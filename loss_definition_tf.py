from random import sample
import tensorflow as tf
from utilstf.params import ParamsPack, parse_param_62
param_pack = ParamsPack()
import math


def WingLoss(target_lmk, pred_lmk):
    
    n_points = target_lmk.shape[2]
    
    target = tf.reshape(target_lmk, [-1, 3*n_points])
    pred = tf.reshape(pred_lmk, [-1, 3*n_points])
    delta = tf.abs(target - pred)
    
    omega = 10
    epsilon = 2
    delta1 = omega * tf.math.log(1 + delta / epsilon)
    delta2 = delta - (omega - (omega * tf.math.log(1 + omega/epsilon)))
    
    losses = tf.where(tf.math.greater_equal(delta, omega), delta2, delta1)
    
    loss = tf.reduce_mean(losses, axis=-1)
    
    return loss

def ParamLoss(target_param, pred_param):
    
    loss = tf.reduce_mean(tf.math.square(pred_param[:,:12] - target_param[:,:12]), axis=-1) \
            + tf.reduce_mean(tf.math.square(pred_param[:,12:] - target_param[:,12:]), axis=-1)

    return loss
class TrainLoss(tf.keras.losses.Loss):
    """Total loss"""
    def call(self, y_true, y_pred):
        [y_pred_param, y_pred_lmk, y_pred_lmk_refine, y_pred_rev_param] = y_pred
        y_pred_param = tf.convert_to_tensor(y_pred_param)
        y_true_param = tf.cast(y_true, y_pred_param.dtype)
        
        param_loss = ParamLoss(y_true_param, y_pred_param)
        
        y_true_lmk = self._reconstruct_vertex_62(y_true_param)
        
        #lmkLoss = self._lmk_loss(y_true_lmk, y_pred_lmk)
        lmk_loss = WingLoss(y_true_lmk, y_pred_lmk)
        
        lmk_refine_loss = WingLoss(y_true_lmk, y_pred_lmk_refine)
        
        param_loss_rev_target = ParamLoss(y_pred_param, y_true_param)
        param_loss_rev_pred = ParamLoss(y_pred_param, y_pred_rev_param)
        
        return 0.02*param_loss + 0.05*lmk_loss + 0.05*lmk_refine_loss \
               + 0.02*param_loss_rev_target + 0.001*param_loss_rev_pred

    
    def _parse_param_62(self, param):
        """Work for only tensor"""
        param = param.numpy()
        p_ = param[:, :12].reshape(-1, 3, 4)
        p = p_[:, :, :3]
        offset = p_[:, :, -1].reshape(-1, 3, 1)
        alpha_shp = param[:, 12:52].reshape(-1, 40, 1)
        alpha_exp = param[:, 52:62].reshape(-1, 10, 1)
        return p, offset, alpha_shp, alpha_exp

    def _reconstruct_vertex_62(self, param, whitening=True, dense=False, transform=True, lmk_pts=68):
        """
        Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
        dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
        image coordinate space, but without alignment caused by face cropping.
        transform: whether transform to image space
        Working with batched tensors. Using Fortan-type reshape.
        """

        if whitening:
            if param.shape[1] == 62:
                param_ = param * param_pack.param_std[:62] + param_pack.param_mean[:62]
            else:
                raise RuntimeError('length of params mismatch')

        p, offset, alpha_shp, alpha_exp = self._parse_param_62(param_)

        if dense:
            
            vertex = p @ (param_pack.u + param_pack.w_shp @ alpha_shp + param_pack.w_exp @ alpha_exp).reshape(-1,68,3, order='F') + offset
            
            if transform: 
                # transform to image coordinate space
                vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :]

        else:
            """For 68 pts"""
            face = (param_pack.u_base + param_pack.w_shp_base @ alpha_shp + param_pack.w_exp_base @ alpha_exp)
            face = tf.reshape(face, [-1, 68, 3])
            face = tf.transpose(face, perm=[0, 2, 1]) #128,3,68
            vertex = p @ face + offset

            if transform: 
                # transform to image coordinate space
                
                #vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :]
                offset_y = tf.constant([0, param_pack.std_size + 1, 0], dtype=tf.float32, shape=[3,1])
                offset_y = tf.tile(offset_y, [1, 68])
                vertex = offset_y - vertex


        return vertex
        
    
class ParamAcc(tf.keras.metrics.Metric):
    """Input and target are all 62-d param"""
    def __init__(self, name="param_acc", **kwargs):
        super(ParamAcc, self).__init__(name=name, **kwargs)
        self.acc = self.add_weight(name="error", initializer="zeros")
        self.total = self.add_weight(name="count", initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        [y_pred_param, y_pred_lmk, _, _] = y_pred
        y_pred_param = tf.convert_to_tensor(y_pred_param)
        y_true_param = y_true[:,:62]
        y_true_param = tf.cast(y_true_param, y_pred_param.dtype)
        
        y_pred_lmk = self._reconstruct_vertex_62(y_pred_param)
        y_true_lmk = self._reconstruct_vertex_62(y_true_param)
        
        param_acc = ParamLoss(y_true_param, y_pred_param)
        lmk_acc = WingLoss(y_true_lmk, y_pred_lmk)
        
        total_acc = 0.02*param_acc + 0.05*lmk_acc
        self.acc.assign_add(tf.reduce_mean(total_acc))
        self.total.assign_add(1)

    def result(self):
        return self.acc/self.total
    
    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.acc.assign(0.0)
        self.total.assign(0)
        
    def _parse_param_62(self, param):
        """Work for only tensor"""
        param = param.numpy()
        p_ = param[:, :12].reshape(-1, 3, 4)
        p = p_[:, :, :3]
        offset = p_[:, :, -1].reshape(-1, 3, 1)
        alpha_shp = param[:, 12:52].reshape(-1, 40, 1)
        alpha_exp = param[:, 52:62].reshape(-1, 10, 1)
        return p, offset, alpha_shp, alpha_exp
            
    def _reconstruct_vertex_62(self, param, whitening=True, dense=False, transform=True, lmk_pts=68):
        """
        Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
        dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
        image coordinate space, but without alignment caused by face cropping.
        transform: whether transform to image space
        Working with batched tensors. Using Fortan-type reshape.
        """

        if whitening:
            if param.shape[1] == 62:
                param_ = param * param_pack.param_std[:62] + param_pack.param_mean[:62]
            else:
                raise RuntimeError('length of params mismatch')

        p, offset, alpha_shp, alpha_exp = self._parse_param_62(param_)

        if dense:
            
            vertex = p @ (param_pack.u + param_pack.w_shp @ alpha_shp + param_pack.w_exp @ alpha_exp).reshape(-1,68,3, order='F') + offset
            
            if transform: 
                # transform to image coordinate space
                vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :]

        else:
            """For 68 pts"""
            face = (param_pack.u_base + param_pack.w_shp_base @ alpha_shp + param_pack.w_exp_base @ alpha_exp)
            face = tf.reshape(face, [-1, 68, 3])
            face = tf.transpose(face, perm=[0, 2, 1]) #128,3,68
            vertex = p @ face + offset

            if transform: 
                # transform to image coordinate space
                offset_y = tf.constant([0, param_pack.std_size + 1, 0], dtype=tf.float32, shape=[3,1])
                offset_y = tf.tile(offset_y, [1, 68])
                vertex = offset_y - vertex

        return vertex        


if __name__ == "__main__":
    pass


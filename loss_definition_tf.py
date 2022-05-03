from random import sample
import tensorflow as tf
from utilstf.params import ParamsPack, parse_param_62
param_pack = ParamsPack()
import math


# class WingLoss(nn.Module):
#     def __init__(self, omega=10, epsilon=2):
#         super(WingLoss, self).__init__()
#         self.omega = omega
#         self.epsilon = epsilon
#         self.log_term = math.log(1 + self.omega / self.epsilon)

#     def forward(self, pred, target, kp=False):
#         n_points = pred.shape[2]
#         pred = pred.transpose(1,2).contiguous().view(-1, 3*n_points)
#         target = target.transpose(1,2).contiguous().view(-1, 3*n_points)
#         y = target
#         y_hat = pred
#         delta_y = (y - y_hat).abs()
#         delta_y1 = delta_y[delta_y < self.omega]
#         delta_y2 = delta_y[delta_y >= self.omega]
#         loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
#         C = self.omega - self.omega * self.log_term
#         loss2 = delta_y2 - C
#         return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

class TrainLoss(tf.keras.losses.Loss):
    """Total loss"""
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        paramLoss = self._param_loss(y_true, y_pred)
        #lmkLoss = self._lmk_loss(y_true, y_pred)
        
        return paramLoss
        
    def _param_loss(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.math.square(y_pred[:,:12] - y_true[:,:12]), axis=-1) + tf.reduce_mean(tf.math.square(y_pred[:,12:] - y_true[:,12:]), axis=-1)
        return mse
    
    def _lmk_loss(self, y_true, y_pred):
        
        lmk_true = self._reconstruct_vertex_62(y_true)
        lmk_pred = self._reconstruct_vertex_62(y_pred)
        sqe = tf.math.square(lmk_pred-lmk_true)
        se = tf.math.sqrt(tf.math.reduce_sum(sqe, axis=1))
        lmk_mse = tf.math.reduce_mean(se, axis=-1)
        
        return lmk_mse
    
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
            vertex = p @ (param_pack.u_base + param_pack.w_shp_base @ alpha_shp + param_pack.w_exp_base @ alpha_exp).reshape(-1, 3, 68) + offset

            if transform: 
                # transform to image coordinate space
                vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :]

        return vertex
        
    
        

class ParamLoss(tf.keras.losses.Loss):
    """Input and target are all 62-d param"""
    def call(self, y_true, y_pred):
        
        y_pred = tf.convert_to_tensor(y_pred)
        
        #y_true = y_true[:,:62]
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(tf.math.square(y_pred-y_true), axis=-1, name="ParamLoss")

    
class ParamAcc(tf.keras.metrics.Metric):
    """Input and target are all 62-d param"""
    def __init__(self, name="param_acc", **kwargs):
        super(ParamAcc, self).__init__(name=name, **kwargs)
        self.acc = self.add_weight(name="pa", initializer="zeros")
        self.total = self.add_weight(name="pa", initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = y_true[:,:62]
        y_true = tf.cast(y_true, y_pred.dtype)
        
        values = tf.reduce_mean(tf.math.square(y_pred-y_true), axis=-1)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            
        self.acc.assign_add(tf.reduce_mean(values))
        self.total.assign_add(1)

    def result(self):
        return self.acc/self.total
    
    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.acc.assign(0.0)
        self.total.assign(0)


if __name__ == "__main__":
    pass


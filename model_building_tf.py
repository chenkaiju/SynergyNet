#!/usr/bin/env python3
# coding: utf-8
import tensorflow as tf

import numpy as np
import scipy.io as sio

# All data parameters import
from utilstf.params import ParamsPack
param_pack = ParamsPack()


def parse_param_62(param):
    """Work for only tensor"""
    p_ = param[:, :12].reshape(-1, 3, 4)
    p = p_[:, :, :3]
    offset = p_[:, :, -1].reshape(-1, 3, 1)
    alpha_shp = param[:, 12:52].reshape(-1, 40, 1)
    alpha_exp = param[:, 52:62].reshape(-1, 10, 1)
    return p, offset, alpha_shp, alpha_exp

class I2P(tf.keras.Model):
    def __init__(self, args):
        super(I2P, self).__init__()
        self.args = args
        if 'mobilenet_v2' in self.args.arch:
            self.backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(
                input_shape = (120,120,3),
                include_top=False
            )
            self.backbone.trainable = True

        self.last_channel = 1280
        
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        
        # building classifier        
        self.num_ori = 12
        self.num_shape = 40
        self.num_exp = 10
        
        self.classifier_ori = tf.keras.Sequential([
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_ori),], name="pose"
        )
        self.classifier_shape = tf.keras.Sequential([
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_shape),], name="shape"
        )
        self.classifier_exp = tf.keras.Sequential([
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_exp),], name="expression"
        )
        
    def call(self, x):
        
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        x = preprocess_input(x)
        x = self.backbone(x)
        
        x = self.global_average_layer(x)
        
        pool_x = x
        
        x_ori = self.classifier_ori(x)
        x_shape = self.classifier_shape(x)
        x_exp = self.classifier_exp(x)
        
        x = tf.concat((x_ori, x_shape, x_exp), axis=1)
        
        return x, pool_x

class SynergyNet(tf.keras.Model):
    def __init__(self, args, name="SynergyNet"):
        super().__init__()
        self.triangles = sio.loadmat('./3dmm_data/tri.mat')['tri'] -1
        self.img_size = args.img_size
        # Image-to-parameter
        self.I2P = I2P(args)
        # Forward: parameter to landmark
        self.forwardDirection = MLP_for(68)
        
        # Reverse: landmark to parameters
        self.backwardDirection = MLP_rev(68)
        
        self.data_param = [param_pack.param_mean, param_pack.param_std, 
                           param_pack.w_shp_base, param_pack.u_base, param_pack.w_exp_base]
        
    def summary(self):
        x = tf.keras.Input(shape=(120, 120, 3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()
    
    def _parse_param_62(self, param):
        """Work for only tensor"""
        
        #p_ = param[:, :12].reshape(-1, 3, 4)
        p_ = tf.reshape(param[:,:12], [-1, 3, 4])
        p = p_[:, :, :3]
        #offset = p_[:, :, -1].reshape(-1, 3, 1)
        offset = tf.reshape(p_[:,:,-1], [-1, 3, 1])
        #alpha_shp = param[:, 12:52].reshape(-1, 40, 1)
        alpha_shp = tf.reshape(param[:, 12:52], [-1, 40, 1])
        #alpha_exp = param[:, 52:62].reshape(-1, 10, 1)
        alpha_exp = tf.reshape(param[:,52:62], [-1, 10, 1])
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
            shp = param_pack.w_shp_base @ alpha_shp
            exp = param_pack.w_exp_base @ alpha_exp
            
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
    
        
    def call(self, input):
        
        x = input
        _3D_attr, avgpool = self.I2P(x)
        
        vertex_lmk = self._reconstruct_vertex_62(_3D_attr, dense=False)
        
        point_residual = self.forwardDirection(vertex_lmk, avgpool, _3D_attr[:,12:52], _3D_attr[:,52:62])
        point_residual = tf.transpose(point_residual, [0, 2, 1])
        vertex_lmk_refine = vertex_lmk + 0.05 * point_residual
        
        _3D_attr_S2 = self.backwardDirection(vertex_lmk_refine)
        
        return _3D_attr, vertex_lmk, vertex_lmk_refine, _3D_attr_S2
        
class MLP_for(tf.keras.layers.Layer):
    def __init__(self, num_pts):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(64, 1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        
        self.conv2 = tf.keras.layers.Conv1D(64, 1)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()        
        
        self.conv3 = tf.keras.layers.Conv1D(64, 1)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()        
        
        self.conv4 = tf.keras.layers.Conv1D(128, 1)
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.relu4 = tf.keras.layers.ReLU()
        
        self.conv5 = tf.keras.layers.Conv1D(1024, 1)
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.relu5 = tf.keras.layers.ReLU()
        
        self.conv6 = tf.keras.layers.Conv1D(512, 1)
        self.bn6 = tf.keras.layers.BatchNormalization()
        self.relu6 = tf.keras.layers.ReLU()
        
        self.conv7 = tf.keras.layers.Conv1D(256, 1)
        self.bn7 = tf.keras.layers.BatchNormalization()
        self.relu7 = tf.keras.layers.ReLU()
        
        self.conv8 = tf.keras.layers.Conv1D(128, 1)
        self.bn8 = tf.keras.layers.BatchNormalization()
        self.relu8 = tf.keras.layers.ReLU()
        
        self.conv9 = tf.keras.layers.Conv1D(3, 1)
        self.bn9 = tf.keras.layers.BatchNormalization()
        self.relu9 = tf.keras.layers.ReLU()        
        
        self.num_pts = num_pts
        
    
    def call(self, input, other_input1=None, other_input2=None, other_input3=None):
        
        Lc = tf.transpose(input, [0, 2, 1])
        out = self.conv1(Lc)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        point_features = out
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)
        
        global_features = tf.keras.layers.MaxPool1D(pool_size=self.num_pts)(out)
        global_features_repeated = tf.tile(global_features, [1, self.num_pts, 1])
        
        #3DMM image
        avgpool = other_input1
        avgpool = tf.expand_dims(input=avgpool, axis=1)
        avgpool = tf.tile(avgpool, [1, self.num_pts, 1])
        
        shape_code = other_input2
        shape_code = tf.expand_dims(input=shape_code, axis=1)
        shape_code = tf.tile(shape_code, [1, self.num_pts, 1])

        expr_code = other_input3
        expr_code = tf.expand_dims(input=expr_code, axis=1)
        expr_code = tf.tile(expr_code, [1, self.num_pts, 1])
        
        out = tf.concat([point_features, global_features_repeated, avgpool, shape_code, expr_code], axis=2)   
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu6(out)
        
        out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu7(out)
        
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu8(out)
        
        out = self.conv9(out)
        out = self.bn9(out)
        out = self.relu9(out)
        
        return out
        
class MLP_rev(tf.keras.layers.Layer):
    def __init__(self, num_pts):
        super().__init__()
        
        self.num_pts = num_pts
        
        self.conv1 = tf.keras.layers.Conv1D(64, 1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        
        self.conv2 = tf.keras.layers.Conv1D(64, 1)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()        

        self.conv3 = tf.keras.layers.Conv1D(128, 1)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()   
        
        self.conv4 = tf.keras.layers.Conv1D(256, 1)
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.relu4 = tf.keras.layers.ReLU()
        
        self.conv5 = tf.keras.layers.Conv1D(1024, 1)
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.relu5 = tf.keras.layers.ReLU()
        
        self.conv6_1 = tf.keras.layers.Conv1D(12, 1)
        self.conv6_2 = tf.keras.layers.Conv1D(40, 1)
        self.conv6_3 = tf.keras.layers.Conv1D(10, 1)
        
        self.bn6_1 = tf.keras.layers.BatchNormalization()
        self.bn6_2 = tf.keras.layers.BatchNormalization()
        self.bn6_3 = tf.keras.layers.BatchNormalization()
        
    def call(self, input, other_input1=None, other_input2=None, other_input3=None):
        
        Lr = tf.transpose(input, [0, 2, 1])
        out = self.conv1(Lr)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)
        
        global_features = tf.keras.layers.MaxPool1D(pool_size=self.num_pts)(out)
        
        out_trans = self.bn6_1(self.conv6_1(global_features))
        out_shape = self.bn6_2(self.conv6_2(global_features))
        out_expr = self.bn6_3(self.conv6_3(global_features))
        
        out = tf.concat([out_trans, out_shape, out_expr], 2)
        out = tf.squeeze(out, axis=[1])
        
        return out
            
if __name__ == '__main__':
    pass    
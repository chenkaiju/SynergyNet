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
        x_ori = self.classifier_ori(x)
        x_shape = self.classifier_shape(x)
        x_exp = self.classifier_exp(x)
        
        x = tf.concat((x_ori, x_shape, x_exp), axis=1)
        
        return x  

class SynergyNet(tf.keras.Model):
    def __init__(self, args, name="SynergyNet"):
        super().__init__()
        self.triangles = sio.loadmat('./3dmm_data/tri.mat')['tri'] -1
        self.img_size = args.img_size
        # Image-to-parameter
        self.I2P = I2P(args)
        
        self.data_param = [param_pack.param_mean, param_pack.param_std, 
                           param_pack.w_shp_base, param_pack.u_base, param_pack.w_exp_base]
        
    def summary(self):
        x = tf.keras.Input(shape=(120, 120, 3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()
        
        
    def call(self, input):
        
        x = input
        #x = tf.keras.layers.Normalization(axis=-1, mean=127.5, variance=128)(input)
        _3D_attr = self.I2P(x)
        
        
        return _3D_attr        
        
    

    
if __name__ == '__main__':
    pass    
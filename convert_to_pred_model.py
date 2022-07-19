import os 
import tensorflow as tf
from tensorflow import keras
import argparse
from synergynet_tf import SynergyNetPred

import tf2onnx
import onnxruntime as onnxrt


# All data parameters import
from utilstf.params import ParamsPack
param_pack = ParamsPack()


# global args (configuration)
args = None # define the static training setting, which wouldn't and shouldn't be changed over the whole experiements.

def parse_args():
    parser = argparse.ArgumentParser(description='3DMM Fitting')

    parser.add_argument('--arch', default='mobilenet_v2', type=str, help="Please choose [mobilenet_v2, mobilenet_1, resnet50, resnet101, or ghostnet]")
    parser.add_argument('--img_size', default=120, type=int)

    global args
    args = parser.parse_args()
    
def convertToSaveModel(ckpt_path):
    
    parse_args()
    
    model_pred = SynergyNetPred(args)
    
    # load pre-tained model    
    model_pred.load_weights(ckpt_path).expect_partial()
    
    model_pred.compute_output_shape(input_shape=(None, 120, 120, 3))
    
    pred_dir = "./pred_model/"
    predict_model_path = os.path.join(pred_dir, "save_model")
    
    pred_weights_dir = os.path.join(pred_dir, "saved_weights")
    if (os.path.exists(pred_weights_dir)==False):
        os.makedirs(pred_weights_dir)
    predict_model_weights = os.path.join(pred_weights_dir, "pred_weight.h5")
    
    model_pred.save(predict_model_path)
    model_pred.save_weights(predict_model_weights)
    
    return model_pred
    
def convertToOnnx(model, onnx_path):
    
    spec = (tf.TensorSpec((None, 120, 120, 3), tf.float32, name="input"),)
    
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=onnx_path)
    output_names = [n.name for n in model_proto.graph.output]
    
    print(output_names)
    
    
if __name__ == '__main__':
    
    dir="./saved_model"
    ckpt_name='cp-0200.ckpt'
    ckpt_path = os.path.join(dir, ckpt_name)
    
    save_model = convertToSaveModel(ckpt_path)
    
    dir="./pred_model/onnx"
    onnx_model_name = "saved_model.onnx"
    onnx_path = os.path.join(dir, onnx_model_name)
    convertToOnnx(save_model, onnx_path)
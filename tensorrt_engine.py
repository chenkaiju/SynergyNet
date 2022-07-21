import tensorrt as trt
from onnx import ModelProto


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
def build_engine(onnx_path, input_shape = [1, 120, 120, 3]):

    """
    This is the function to create the TensorRT engine
    Args:
        onnx_path : Path to onnx_file. 
        input_shape : Shape of the input of the ONNX file. 
    """

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config.max_workspace_size = (256 << 20)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = input_shape
        engine = builder.build_engine(network, config)
        return engine

def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)
        
def load_engine(trt_runtime, plan_path):
    with open(plan_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

if __name__=='__main__':
    
    engine_name = './pred_model/tensorrt/synergy_pred.plan'
    onnx_path = "./pred_model/onnx/saved_model.onnx"
    batch_size = 1
    
    onnx_to_plan(onnx_path, batch_size, engine_name)
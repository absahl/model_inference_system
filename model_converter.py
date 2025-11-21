from optimum.onnxruntime import ORTModel

class ModelConverter:
    def __init__(self, model_name, target_backend):
        print(f'converting model to backend [{target_backend}]')
        if target_backend == 'onnx':
            onnx_model = ORTModel.from_pretrained(model_name, export=True)

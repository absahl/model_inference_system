from model_converter import ModelConverter
from transformers import AutoTokenizer, AutoModel
from optimum.onnxruntime import ORTModelForSequenceClassification

class Model:
    required_keys = ['name', 'type']
    allowed_types = ['text', 'image']
    allowed_backends = ['onnx']
    allowed_providers = ['CPUExecutionProvider' , 'CUDAExecutionProvider']

    def __init__(self, **kwargs):
        print(f'initializing model from args [{kwargs}]')
        missing_keys = [k for k in Model.required_keys if k not in kwargs.keys()]
        if missing_keys:
            raise ValueError(f'failed to initialize model due to missing keys: {missing_keys}')

        if kwargs['type'] not in Model.allowed_types:
            raise ValueError(f"failed to initialize model due to invalid type [{kwargs['type']}]")

        if kwargs['backend'] not in Model.allowed_backends:
            raise ValueError(f"failed to initialize model due to invalid backend [{kwargs['backend']}]")

        if kwargs['provider'] not in Model.allowed_providers:
            raise ValueError(f"failed to initialize model due to invalid provider [{kwargs['provider']}]")

        self.name = kwargs['name']
        self.type = kwargs['type']
        self.backend = kwargs.get('backend', 'onnx')
        self.provider = kwargs.get('provider', 'CPUExecutionProvider')
        self.opt_level = kwargs.get('optimization_level', 99)
        self.quantized = kwargs.get('quantized', False)

        # load tokenizer and model from HF
        print(f'loading tokenizer and model from HF [{self.name}]')
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        default_model = AutoModel.from_pretrained(self.name)
        self.model = ModelConverter(self.name, self.backend)

    def __repr__(self):
        return f"{self.__dict__}"

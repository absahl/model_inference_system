class Model:
    required_keys = ['name', 'type']

    def __init__(self, **kwargs):
        missing_keys = [k for k in Model.required_keys if k not in kwargs.keys()]
        if missing_keys:
            raise ValueError(f'{missing_keys} are missing')

        self.name = kwargs['name']
        self.type = kwargs['type']
        self.backend = kwargs.get('backend', 'onnx')
        self.provider = kwargs.get('provider', 'CPUExecutionProvider')
        self.opt_level = kwargs.get('optimization_level', 99)
        self.quantized = kwargs.get('quantized', False)

    def __repr__(self):
        return f"{self.__dict__}"

class HubertContentVec:
    def __init__(self, *args, **kwargs):
        pass

    def extract(self, wav, sr):
        raise NotImplementedError("ContentVec extraction not needed for ONNX export")

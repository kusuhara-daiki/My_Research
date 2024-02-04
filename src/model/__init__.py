from .torch_model import TorchModel
from .yolo import Yolo


def get_detection_model(model_name: str, dataset):
    if model_name in Yolo.models:
        model = Yolo(model_name, dataset)
    elif model_name in TorchModel.models:
        model = TorchModel(model_name, dataset)
    else:
        raise NotImplementedError(model_name)
    return model

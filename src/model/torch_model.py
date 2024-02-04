import os

import pandas as pd
from pandas import DataFrame
from PIL import Image
from torch.nn import Module
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

from dataset import coco_label


class TorchModel(Module):
    models = ["fasterrcnn_resnet50"]

    def __init__(self, model_name, dataset):
        super().__init__()

        if model_name == "fasterrcnn_resnet50":
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn(weights=weights)
            self.transform = weights.transforms()
        else:
            raise ValueError(model_name)
        self.model.eval()

        self.columns = ["xmin", "ymin", "xmax", "ymax"]

    def forward(self, path) -> DataFrame:
        prediction = DataFrame()
        for p in path:
            image = Image.open(p).convert("RGB")
            image = self.transform(image).unsqueeze(0)
            box, label, score = self.model(image)[0].values()
            pred = DataFrame(box, columns=self.columns)
            pred["confidence"] = score
            pred["label"] = label
            pred.insert(0, "file", os.path.basename(p))
            prediction = pd.concat([prediction, pred])
        prediction["label"] = prediction["label"].map(coco_label)
        return prediction

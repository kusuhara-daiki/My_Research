import os

import pandas as pd
from pandas import DataFrame
from torch.nn import Module
from ultralytics import YOLO


class Yolo(Module):
    models = ["yolov5l"]

    def __init__(self, model_name, dataset):
        super().__init__()

        if model_name == "yolov5l":
            self.model = YOLO("../storage/models/yolov5lu.pt")
        else:
            raise ValueError(model_name)

        class_id = dict(zip(self.model.names.values(), self.model.names.keys()))
        self.classes = [class_id[label] for label in dataset.labels]
        self.columns = ["xmin", "ymin", "xmax", "ymax", "confidence", "label"]

    def forward(self, path):
        prediction = DataFrame()
        pred = self.model(
            path, conf=0, iou=1, classes=self.classes, batch=len(path), verbose=False
        )
        for file, p in zip(path, pred):
            p = DataFrame(p.boxes.data.cpu(), columns=self.columns)
            p.insert(0, "file", os.path.basename(file))
            prediction = pd.concat([prediction, p])
        prediction["label"] = prediction["label"].map(self.model.names)
        return prediction

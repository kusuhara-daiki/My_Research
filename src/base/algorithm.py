import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from mean_average_precision.utils import compute_iou
from numpy.typing import NDArray
from pandas import DataFrame

from utils import ProgressBar, config_parser

from .dataset import Dataset

config = config_parser()


class Algorithm:
    def __init__(self):
        if not hasattr(self, "_run"):
            raise NotImplementedError

    def run(self, dataset: Dataset, all_detection: DataFrame):
        self.dataset = dataset

        # remove small confidence bbox
        remove = all_detection["confidence"] < config.confidence_threshold
        detection = all_detection[~remove].copy()

        # split by file
        detection = detection.groupby("file")
        config.n_examples = len(detection)
        detection = [detection.get_group(f) for f in detection.groups]

        # run algorithm
        self.pbar = ProgressBar(config.n_examples, "suppression", color="cyan")
        stopwatch = time.time()
        if config.thread == 1:
            detection = [self._run(d) for d in detection]
        else:
            with ThreadPoolExecutor(config.thread) as executor:
                detection = list(executor.map(self._run, detection))
        config.computational_time = time.time() - stopwatch

        # return result
        detection = pd.concat(detection)
        self.pbar.end()
        return detection

    """utility functions"""

    def get_max_precision(self, candidates: DataFrame):
        idx = candidates["confidence"].idxmax()
        bbox = candidates.loc[idx, ["xmin", "ymin", "xmax", "ymax"]].values
        bbox = bbox.reshape(1, 4)
        return idx, bbox

    def compute_iou(self, candidates: DataFrame):
        bbox = candidates[["xmin", "ymin", "xmax", "ymax"]].values
        iou = compute_iou(bbox, bbox)
        return iou

    def compute_overlap(self, candidates: DataFrame):
        def area(bbox):
            return (bbox[:, 2] - bbox[:, 0] + 1) * (bbox[:, 3] - bbox[:, 1] + 1)

        bbox = candidates[["xmin", "ymin", "xmax", "ymax"]].values
        bbox1 = np.tile(bbox, (len(bbox), 1))
        bbox2 = np.repeat(bbox, len(bbox), axis=0)
        xmin = np.maximum(bbox1[:, 0], bbox2[:, 0])
        ymin = np.maximum(bbox1[:, 1], bbox2[:, 1])
        xmax = np.minimum(bbox1[:, 2], bbox2[:, 2])
        ymax = np.minimum(bbox1[:, 3], bbox2[:, 3])
        intersection = np.maximum(xmax - xmin + 1, 0) * np.maximum(ymax - ymin + 1, 0)
        scale = np.multiply(area(bbox1), area(bbox2))
        overlap = (intersection / np.sqrt(scale)).reshape(len(bbox), len(bbox))
        return overlap

    def gaussian_weighting(self, iou: NDArray):
        return np.exp(-(iou**2) / config.gaussian_sigma)

    def linear_weighting(self, iou: NDArray):
        return np.where(iou > config.linear_threshold, 1 - iou, 1)

    def soft_suppression(self, detection: DataFrame, candidates: DataFrame):
        for _, bbox in candidates.iterrows():
            confidence = bbox["confidence"]
            _bbox = bbox[["xmin", "ymin", "xmax", "ymax"]].values.reshape(1, 4)
            _candidates = candidates[["xmin", "ymin", "xmax", "ymax"]].values
            iou = compute_iou(_bbox, _candidates).reshape(-1)
            confidence *= self.gaussian_weighting(iou.max())
            if confidence > config.confidence_threshold:
                detection = pd.concat([detection, bbox])
        return detection

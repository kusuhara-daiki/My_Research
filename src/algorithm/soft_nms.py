from mean_average_precision.utils import compute_iou
from numpy.typing import NDArray
from pandas import DataFrame

from base import Algorithm
from utils import config_parser

config = config_parser()


class SoftNMS(Algorithm):
    """
    Soft-NMS -- Improving Object Detection With One Line of Code

    paper: https://arxiv.org/abs/1704.04503
    """

    def __init__(self):
        super().__init__()
        self.scaleing = {
            "linear": self.linear_weighting,
            "gaussian": self.gaussian_weighting,
        }[config.scaling]

    def _run(self, detection: DataFrame):
        for label in detection["label"].unique():
            candidates = detection.groupby("label").get_group(label)
            while len(candidates) > 0:
                idx, bbox = self.get_max_precision(candidates)
                detection.loc[idx, "confidence"] = candidates.loc[idx, "confidence"]
                candidates = candidates.drop(idx)
                candidates = self.soft_scoring(bbox, candidates)
        self.pbar.step()
        detection = detection[detection["confidence"] >= config.confidence_threshold]
        return detection

    def soft_scoring(self, bbox: NDArray, candidates: DataFrame):
        other_bbox = candidates[["xmin", "ymin", "xmax", "ymax"]].values
        iou = compute_iou(bbox, other_bbox).reshape(-1).astype(float)
        candidates["confidence"] *= self.scaleing(iou)
        return candidates

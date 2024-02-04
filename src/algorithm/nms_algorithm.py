from mean_average_precision.utils import compute_iou
from numpy.typing import NDArray
from pandas import DataFrame

from base import Algorithm
from utils import config_parser

config = config_parser()


class NMS_Algorithm(Algorithm):
    """
    Non-Maximum Suppression Algorithm
    """

    def _run(self, detection: DataFrame) -> DataFrame:
        detection["remain"] = False
        for label in detection["label"].unique():
            candidates = detection.groupby("label").get_group(label)
            while len(candidates) > 0:
                idx, bbox = self.get_max_precision(candidates)
                detection.loc[idx, "remain"] = True
                candidates = candidates.drop(idx)
                candidates = self.remove_bbox(bbox, candidates)
        self.pbar.step()
        detection = detection.groupby("remain").get_group(True)
        detection = detection.drop("remain", axis=1)
        return detection

    def remove_bbox(self, bbox: NDArray, candidates: DataFrame) -> DataFrame:
        other_bbox = candidates[["xmin", "ymin", "xmax", "ymax"]].values
        iou = compute_iou(bbox, other_bbox).reshape(-1)
        candidates = candidates[iou <= config.iou_threshold]
        return candidates

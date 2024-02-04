import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from ._oc_cost import get_cmap, get_ot_cost


def oc_cost(files: NDArray, detection: DataFrame, ground_truth: DataFrame) -> str:
    metric = []
    class_id = ground_truth["class_id"].unique()
    ground_truth["confidence"] = 1

    def cmap_function(x, y):
        return get_cmap(x, y, alpha=0.5, beta=0.6)

    for file in files:
        detection_storage, ground_truth_storage = [], []
        _detection = detection[detection["file"] == file]
        _ground_truth = ground_truth[ground_truth["file"] == file]
        for i in class_id:
            d = _detection[_detection["class_id"] == i]
            detection_storage.append(
                d[["xmin", "ymin", "xmax", "ymax", "confidence"]].values
            )
            g = _ground_truth[_ground_truth["class_id"] == i]
            ground_truth_storage.append(
                g[["xmin", "ymin", "xmax", "ymax", "confidence"]].values
            )
        metric.append(
            get_ot_cost(ground_truth_storage, detection_storage, cmap_function)
        )
    metric = np.mean(metric)
    result = f"OC cost = {metric* 100:.2f}%\n"
    return result

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from utils import config_parser

from .mean_average_precision import mean_average_precision
from .oc_cost import oc_cost

config = config_parser()


def evaluate(files: NDArray, detection: DataFrame, ground_truth: DataFrame) -> str:
    # assign class_id
    all_class = np.union1d(detection["label"].unique(), ground_truth["label"].unique())
    class_id = {c: i for i, c in enumerate(all_class)}
    detection["class_id"] = detection["label"].map(class_id)
    ground_truth["class_id"] = ground_truth["label"].map(class_id)

    # evaluate
    result = (
        "\n"
        f"n_examples = {config.n_examples}\n"
        f"computational time = {config.computational_time:.2f} (sec)\n"
    )
    result += oc_cost(files, detection, ground_truth)
    result += mean_average_precision(files, detection, ground_truth)
    print(result, file=open(f"{config.savedir}/summary.txt", "w"))
    return result

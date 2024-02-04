import os
from contextlib import redirect_stdout

from numpy.typing import NDArray
from pandas import DataFrame
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils import config_parser

config = config_parser()


def mean_average_precision(
    files: NDArray, detection: DataFrame, ground_truth: DataFrame
) -> str:
    image_id = {f: i for i, f in enumerate(files)}
    class_id = ground_truth["class_id"].unique()
    coco_format = COCO()
    coco_format.dataset["images"] = [{"id": i} for i in image_id.values()]
    coco_format.dataset["categories"] = [{"id": c} for c in class_id]
    with redirect_stdout(open(os.devnull, "w")):
        coco_format.createIndex()
    detection["width"] = detection["xmax"] - detection["xmin"]
    detection["height"] = detection["ymax"] - detection["ymin"]
    class_id = ground_truth["class_id"].unique()
    detection = [
        {
            "image_id": image_id[d["file"]],
            "category_id": d["class_id"],
            "bbox": d[["xmin", "ymin", "width", "height"]].tolist(),
            "score": d["confidence"],
        }
        for _, d in detection.iterrows()
    ]
    with redirect_stdout(open(os.devnull, "w")):
        detection = coco_format.loadRes(detection)
    ground_truth["width"] = ground_truth["xmax"] - ground_truth["xmin"]
    ground_truth["height"] = ground_truth["ymax"] - ground_truth["ymin"]
    ground_truth = [
        {
            "image_id": image_id[d["file"]],
            "category_id": d["class_id"],
            "bbox": d[["xmin", "ymin", "width", "height"]].tolist(),
        }
        for _, d in ground_truth.iterrows()
    ]
    with redirect_stdout(open(os.devnull, "w")):
        ground_truth = coco_format.loadRes(ground_truth)
    metrics = COCOeval(ground_truth, detection, "bbox")
    with redirect_stdout(open(os.devnull, "w")):
        metrics.evaluate()
        metrics.accumulate()
        metrics.summarize()
    mAP = metrics.stats
    result = (
        f"mAP: {mAP[0] * 100:.2f} %\n"
        f"mAP@50: {mAP[1] * 100:.2f} %\n"
        f"mAP@75: {mAP[2] * 100:.2f} %\n"
        f"mAP@small: {mAP[3] * 100:.2f} %\n"
        f"mAP@medium: {mAP[4] * 100:.2f} %\n"
        f"mAP@large: {mAP[5] * 100:.2f} %\n"
        f"mAR@1: {mAP[6] * 100:.2f} %\n"
        f"mAR@10: {mAP[7] * 100:.2f} %\n"
        f"mAR@100: {mAP[8] * 100:.2f} %\n"
        f"mAR@small: {mAP[9] * 100:.2f} %\n"
        f"mAR@medium: {mAP[10] * 100:.2f} %\n"
        f"mAR@large: {mAP[11] * 100:.2f} %\n"
    )
    return result

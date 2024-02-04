import amplify
import cv2
import imagehash
import numpy as np
import pandas as pd
from pandas import DataFrame
from PIL import Image

from base import Algorithm
from solver import get_solver
from utils import config_parser

config = config_parser()


class QAQS_hash(Algorithm):
    """
    QAQS: Quantum Appearance QUBO Suppression

    appearance feature: image-hash
    """

    def _run(self, detection: DataFrame):
        solver = get_solver()
        result = pd.DataFrame()
        for label in detection["label"].unique():
            candidates = detection.groupby("label").get_group(label)
            problem = self.make_problem(candidates)
            remain = solver.solve(problem).astype(bool)
            result = pd.concat([result, candidates[remain]])
            candidates = candidates[~remain]
            result = self.soft_suppression(result, candidates)
        self.pbar.step()
        return result

    def make_problem(self, candidates: DataFrame) -> amplify.BinaryPoly:
        generator = amplify.SymbolGenerator(amplify.BinaryPoly)
        x = generator.array(len(candidates))
        problem = 0

        # features
        confidence = candidates["confidence"].values
        iou = self.compute_iou(candidates)
        overlap = self.compute_overlap(candidates)
        appearance = self.appearance_feature(candidates)

        for i in range(len(candidates)):
            problem -= config.A1 * confidence[i] * x[i]
            for j in range(i):
                Q_ij = config.A2 * iou[i, j]
                Q_ij += config.A3 * overlap[i, j]
                Q_ij *= appearance[i, j]
                problem += Q_ij * x[i] * x[j]
        return problem

    def appearance_feature(self, candidates: DataFrame):
        image_id = candidates["file"].unique().item()
        image = self.dataset.load_image(image_id, config.color).squeeze()
        all_bbox = candidates[["xmin", "ymin", "xmax", "ymax"]]
        all_bbox = all_bbox.astype(int).values
        similarity = np.zeros((len(candidates), len(candidates)))
        feature = []
        for i, bbox in enumerate(all_bbox):
            bbox = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            if config.resize:
                bbox = cv2.resize(bbox, (config.resize, config.resize))
            bbox = Image.fromarray(bbox)
            feature.append(imagehash.average_hash(bbox, hash_size=config.hash_size))
            for j in range(i):
                similarity[i, j] = 1 - (feature[i] - feature[j]) / config.hash_size
        similarity += similarity.T
        return similarity

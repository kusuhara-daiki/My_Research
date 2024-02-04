import amplify
import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame

from base import Algorithm
from solver import get_solver
from utils import config_parser

config = config_parser()


class QAQS_histogram(Algorithm):
    """
    QAQS: Quantum Appearance QUBO Suppression

    appearance feature: RGB
    """

    def __init__(self):
        super().__init__()
        self.histogram_parametor = {
            "rgb": ([0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256]),
            "gray": ([0], None, [256], [0, 256]),
            "hsv": ([0, 1, 2], None, [180, 256, 256], [0, 180, 0, 256, 0, 256]),
            "lab": ([0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256]),
        }[config.color]

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
        all_bbox = candidates[["xmin", "ymin", "xmax", "ymax"]].astype(int)
        method = cv2.HISTCMP_CORREL
        similarity = np.zeros((len(candidates), len(candidates)))
        feature = []
        for i in range(len(candidates)):
            xmin, ymin, xmax, ymax = all_bbox.iloc[i].values
            bbox = image[ymin:ymax, xmin:xmax]
            feature.append(cv2.calcHist([bbox], *self.histogram_parametor))
            for j in range(i):
                similarity[i, j] = cv2.compareHist(feature[i], feature[j], method)
        similarity += similarity.T
        return similarity

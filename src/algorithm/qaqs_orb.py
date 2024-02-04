import amplify
import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame

from base import Algorithm
from solver import get_solver
from utils import config_parser

config = config_parser()


class QAQS_orb(Algorithm):
    """
    QAQS: Quantum Appearance QUBO Suppression
    
    appearance feature: orb
    """

    def __init__(self):
        super().__init__()
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher().match #マッチの仕方も多数のため調査が必要
    
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
    
    def make_problem(self, candidates:DataFrame) -> amplify.BinaryPoly:
        generator = amplify.SymbolGenerator(amplify.BinaryPoly)
        x = generator.array(len(candidates))
        problem = 0

        #feature
        confidence = candidates["confidence"].values
        iou = self.compute_iou(candidates)
        overlap = self.compute_overlap(candidates)
        appearance = self.appearance_feature(candidates)

        for i in range(len(candidates)):
            problem -= config.A1 * confidence[i] * x[i]
            for j in range(i):
                Q_ij = config.A2 * iou[i, j]
                Q_ij += config.A3 * overlap[i, j]
                Q_ij *= appearance[i ,j]
                problem += Q_ij * x[i] * x[j]
        return problem
    
    def appearance_feature(self, candidates: DataFrame):
        image_id = candidates["file"].unique().item()
        image = self.dataset.load_image(image_id, config.color).squeeze()
        all_box = candidates[["xmin", "ymin", "xmax", "ymax"]]
        all_box = all_box.astype(int).values
        similarity = np.zeros((len(candidates), len(candidates)))
        feature = []
        for i, bbox in enumerate(all_box):
            bbox = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            if config.resize:
                bbox = cv2.resize(bbox, (config.resize, config.resize))
            feature.append(self.orb.detectAndCompute(bbox, None)[1])
            for j in range(i):
                if feature[i] is None or feature[j] is None:
                    continue
                match = self.matcher(feature[i], feature[j])
                similarity[i, j] = np.mean([m.distance for m in match])
        similarity += similarity.T
        similarity = 1 - similarity / (similarity.max() + 1e-10)
        return similarity
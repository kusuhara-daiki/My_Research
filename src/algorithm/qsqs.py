import amplify
import pandas as pd
from pandas import DataFrame

from base import Algorithm
from solver import get_solver
from utils import config_parser

config = config_parser()


class QSQS_Algorithm(Algorithm):
    """
    Quantum-soft QUBO Suppression for Accurate Object Detection

    paper: https://arxiv.org/abs/2007.13992
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

        for i in range(len(candidates)):
            problem -= config.A1 * confidence[i] * x[i]
            for j in range(i):
                Q_ij = config.A2 * iou[i, j]
                Q_ij += config.A3 * overlap[i, j]
                problem += Q_ij * x[i] * x[j]
        return problem

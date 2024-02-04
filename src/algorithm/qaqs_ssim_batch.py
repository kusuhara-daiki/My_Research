import amplify
import gc
import numpy as np
import pandas as pd
from pandas import DataFrame

import torch

from base import Algorithm
from solver import get_solver
from utils import config_parser
from cython_ssim import ssim_batch_cython


config = config_parser()


class QAQS_ssim_batch(Algorithm):
    """
    QAQS: Quantum Appearance QUBO Suppression

    appearance feature: SSIM
    """
    window = None

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
        if self.pbar.iter % 20 == 0: # GPUメモリエラー対策
            gc.collect()
            torch.cuda.empty_cache()
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
        window_size = 11
        channel = image.shape[-1]
        if self.window is None:
            self.window = ssim_batch_cython.create_window(window_size, channel)
        similarity = ssim_batch_cython._appearance_feature_ssim_batch_cython_hybrid(image, all_bbox, self.window, window_size, channel, config.resize).astype(np.float64)
        return similarity
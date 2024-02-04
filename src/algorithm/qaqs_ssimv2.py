import amplify
import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame

import torch
# import torchvision.transforms.functional as F
import torch.nn.functional as F

from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torch.autograd import Variable #多分不要
from torch import optim

from base import Algorithm
from solver import get_solver
from utils import config_parser
import pytorch_ssim


config = config_parser()


class QAQS_ssimv2(Algorithm):
    """
    QAQS: Quantum Appearance QUBO Suppression

    appearance feature: SSIM
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
        
        bbox_list = []
        mu_list = []
        sigma_list = []
        window_size = 11
        
        for i, bbox in enumerate(all_bbox):
            bbox = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            if config.resize:
                bbox = cv2.resize(bbox, (config.resize, config.resize))
            bbox = torch.from_numpy(np.rollaxis(bbox, 2)).float().unsqueeze(0)/255.0
            (_, channel, _, _) = bbox.size()
            
            window = pytorch_ssim.create_window(window_size, channel)
            
            bbox_list.append(bbox)
            
            mu1 = F.conv2d(bbox, window, padding = window_size//2, groups = channel)
            mu_list.append(mu1)
            mu1_sq = mu1.pow(2)
            
            sigma1_sq = F.conv2d(bbox*bbox, window, padding = window_size//2, groups = channel) - mu1_sq
            sigma_list.append(sigma1_sq)
            
            for j in range(i):
                similarity[i, j] = pytorch_ssim.ssim(
                    bbox_list[i], 
                    bbox_list[j],
                    mu_list[i],
                    mu_list[j],
                    sigma_list[i],
                    sigma_list[j]
                    ).item()
        similarity += similarity.T
        return similarity
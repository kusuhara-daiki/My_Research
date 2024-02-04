import json
import os
import random
import re
from glob import glob
from subprocess import check_output
from urllib.request import urlretrieve
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from base import Dataset
from utils import config_parser

config = config_parser()


class CrowdHuman(Dataset):
    def __init__(self):
        super().__init__()
        self.datadir = "../storage/data/crowd_human"
        self.download()
        self.labels = ["person"]

    def get_files(self, n_examples: int):
        path_list = glob(f"{self.datadir}/Images/*.jpg")
        path_list = random.sample(path_list, n_examples)
        return path_list

    def load_image(self, files: NDArray, color: str):
        path = np.unique(files)
        image_path = [f"{self.datadir}/Images/{p}" for p in path]
        image = [cv2.imread(p) for p in image_path]
        image = np.array([cv2.cvtColor(img, self.color[color]) for img in image])
        return image

    def load_ground_truth(self, files: NDArray):
        contents = open(f"{self.datadir}/annotations.odgt", "r").readlines()
        ground_truth = []
        for line in contents:
            image_id, bbox = json.loads(line).values()
            ground_truth += [[image_id, b["tag"], *b["vbox"]] for b in bbox]
        columns = ["file", "label", "xmin", "ymin", "width", "height"]
        ground_truth = pd.DataFrame(ground_truth, columns=columns)
        ground_truth = ground_truth[ground_truth["label"].isin(self.labels)]
        ground_truth["file"] = ground_truth["file"].apply(lambda x: x + ".jpg")
        ground_truth = ground_truth[ground_truth["file"].isin(files)]
        ground_truth["xmax"] = ground_truth["xmin"] + ground_truth["width"]
        ground_truth["ymax"] = ground_truth["ymin"] + ground_truth["height"]
        return ground_truth

    def download(self):
        os.makedirs(self.datadir, exist_ok=True)
        google_url = "https://drive.google.com/uc?export=download&"
        if not os.path.exists(f"{self.datadir}/annotations.odgt"):
            print("Downloading annotations data...")
            url = f"{google_url}id=10WIRwu8ju8GRLuCkZ_vT6hnNxs5ptwoL"
            urlretrieve(url, f"{self.datadir}/annotations.odgt")

        if not os.path.exists(f"{self.datadir}/Images"):
            print("Downloading images...")
            url = f"{google_url}id=18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO"
            confirm = check_output(["wget", "--quiet", url, "-O-"])
            confirm = re.search("confirm=([0-9A-Za-z_]+).*", confirm.decode()).group(1)
            url = f"{google_url}confirm={confirm}&id=18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO"
            urlretrieve(url, f"{self.datadir}/CrowdHuman_val.zip")
            ZipFile(f"{self.datadir}/CrowdHuman_val.zip", "r").extractall(self.datadir)

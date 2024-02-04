import os
import random
from contextlib import redirect_stdout
from glob import glob
from urllib.request import urlretrieve
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pycocotools.coco import COCO

from base import Dataset
from utils import config_parser

config = config_parser()


class Coco2017(Dataset):
    def __init__(self):
        super().__init__()
        self.datadir = "../storage/data/coco2017"
        self.download()
        with redirect_stdout(open(os.devnull, "w")):
            self.annotations = COCO(f"{self.datadir}/annotations.json")
        self.labels = [c["name"] for c in self.annotations.dataset["categories"]]

    def get_files(self, n_examples: int):
        path_list = glob(f"{self.datadir}/val2017/*.jpg")
        path_list = random.sample(path_list, n_examples)
        # path_list = open("../storage/data/test_coco2017.txt").read().splitlines()
        # path_list = [f"{self.datadir}/val2017/{path}" for path in path_list]
        # path_list = path_list[:n_examples]
        return path_list

    def load_image(self, files: NDArray, color: str):
        path = np.unique(files)
        image_path = [f"{self.datadir}/val2017/{p}" for p in path]
        image = [cv2.imread(p) for p in image_path]
        image = np.array([cv2.cvtColor(img, self.color[color]) for img in image])
        return image

    def load_ground_truth(self, files: NDArray):
        categories = self.annotations.dataset["categories"]
        categories = {d["id"]: d["name"] for d in categories}
        image_ids = [int(f[: -len(".jpg")]) for f in files]
        ground_truth = self.annotations.loadAnns(self.annotations.getAnnIds(image_ids))
        ground_truth = pd.DataFrame(
            [(t["image_id"], t["category_id"], *t["bbox"]) for t in ground_truth],
            columns=["file", "label", "xmin", "ymin", "width", "height"],
        )
        ground_truth["file"] = ground_truth["file"].apply(lambda x: f"{x:012}.jpg")
        ground_truth["xmax"] = ground_truth["xmin"] + ground_truth["width"]
        ground_truth["ymax"] = ground_truth["ymin"] + ground_truth["height"]
        ground_truth["label"] = ground_truth["label"].map(categories)
        return ground_truth

    def download(self):
        os.makedirs(self.datadir, exist_ok=True)
        if not os.path.exists(f"{self.datadir}/annotations.json"):
            print("Downloading annotations data...")
            url = (
                "http://images.cocodataset.org/"
                "annotations/annotations_trainval2017.zip"
            )
            file = f"{self.datadir}/annotations_trainval2017.zip"
            urlretrieve(url, file)
            contents = ZipFile(file, "r").open("annotations/instances_val2017.json")
            contents = contents.read().decode("utf-8")
            print(contents, file=open(f"{self.datadir}/annotations.json", "w"))

        if not os.path.exists(f"{self.datadir}/val2017"):
            print("Downloading images...")
            url = "http://images.cocodataset.org/zips/val2017.zip"
            file = f"{self.datadir}/val2017.zip"
            urlretrieve(url, file)
            ZipFile(file, "r").extractall(self.datadir)

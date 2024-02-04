import os
import random
import re
from glob import glob
from shutil import rmtree
from subprocess import check_output
from urllib.request import urlretrieve
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from base import Dataset
from utils import ProgressBar


class WiderPerson(Dataset):
    def __init__(self):
        super().__init__()
        self.datadir = "../storage/data/wider_person"
        self.download()
        self.labels = ["person"]

    def get_files(self, n_examples: int):
        path_list = open(f"{self.datadir}/val.txt").readlines()
        path_list = [f"{self.datadir}/Images/{path[:-1]}.jpg" for path in path_list]
        path_list = random.sample(path_list, n_examples)
        return path_list

    def load_image(self, files: NDArray, color: str):
        path = np.unique(files)
        image_path = [f"{self.datadir}/Images/{p}" for p in path]
        image = [cv2.imread(p) for p in image_path]
        image = np.array([cv2.cvtColor(img, self.color[color]) for img in image])
        return image

    def load_ground_truth(self, files: NDArray):
        ground_truth = pd.read_csv(f"{self.datadir}/annotations.csv")
        ground_truth = ground_truth[ground_truth["file"].isin(files)]
        return ground_truth

    def download(self):
        os.makedirs(self.datadir, exist_ok=True)
        google_url = "https://drive.google.com/uc?export=download&"
        if not os.path.exists(f"{self.datadir}/WiderPerson.zip"):
            print("Downloading data...")
            url = google_url + "id=1I7OjhaomWqd8Quf7o5suwLloRlY0THbp"
            confirm = check_output(["wget", "--quiet", url, "-O-"])
            confirm = re.search("confirm=([0-9A-Za-z_]+).*", confirm.decode()).group(1)
            url = f"{google_url}confirm={confirm}&id=1I7OjhaomWqd8Quf7o5suwLloRlY0THbp"
            urlretrieve(url, f"{self.datadir}/WiderPerson.zip")
            ZipFile(f"{self.datadir}/WiderPerson.zip").extractall(self.datadir)
            rmtree(f"{self.datadir}/Evaluation")
            self.format_annotations()

    def format_annotations(self):
        print("Formatting annotations...")
        annotation_files = glob(f"{self.datadir}/Annotations/*.txt")
        storage = pd.DataFrame()
        columns = ["_label", "xmin", "ymin", "xmax", "ymax"]
        pbar = ProgressBar(len(annotation_files))
        for file in annotation_files:
            annotations = open(file).readlines()[1:]
            annotations = [line.split() for line in annotations]
            annotations = pd.DataFrame(annotations, columns=columns)
            annotations.insert(0, "file", os.path.basename(file)[: -len(".txt")])
            storage = pd.concat([storage, annotations])
            pbar.step()
        storage = storage[storage["_label"].isin(["1", "2"])]
        storage["label"] = "person"
        storage.to_csv(f"{self.datadir}/annotations.csv", index=False)
        rmtree(f"{self.datadir}/Annotations")
        pbar.end()

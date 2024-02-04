import os
from argparse import ArgumentParser

import pandas as pd
from matplotlib import pyplot as plt

from dataset import load_dataset
from utils import ProgressBar


def argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-r",
        "--read",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args


def main():
    os.makedirs(f"{os.path.dirname(args.read)}/bbox", exist_ok=True)
    dataset = load_dataset(args.read)
    detection = pd.read_csv(args.read)
    files = detection["file"].unique()
    pbar = ProgressBar(len(files), "Save images", color="cyan")
    for file in files:
        plt.subplots(figsize=(8, 8))
        plt.imshow(dataset.load_image(file, "rgb").squeeze())
        detected_boxes = detection[detection["file"] == file]
        for _, bbox in detected_boxes.iterrows():
            plt.plot(
                [bbox["xmin"], bbox["xmax"], bbox["xmax"], bbox["xmin"], bbox["xmin"]],
                [bbox["ymin"], bbox["ymin"], bbox["ymax"], bbox["ymax"], bbox["ymin"]],
                label=bbox["label"],
            )
        if len(detected_boxes) > 0:  # escape warning
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.axis("off")
        plt.savefig(f"{os.path.dirname(args.read)}/bbox/{file}", bbox_inches="tight")
        plt.close()
        pbar.step()
    pbar.end()


if __name__ == "__main__":
    args = argparser()
    main()

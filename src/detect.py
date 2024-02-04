from argparse import ArgumentParser

import pandas as pd
import torch

from dataset import load_dataset
from model import get_detection_model
from utils import ProgressBar, reproducibility, save_error


def argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n",
        "--n_examples",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=100,
        help="batch size of detection",
    )
    args = parser.parse_args()
    return args


@torch.no_grad()
@save_error()
def main():
    reproducibility()
    dataset = load_dataset(args.dataset)
    path_list = dataset.get_files(args.n_examples)
    model = get_detection_model(args.model, dataset)

    storage = pd.DataFrame()
    pbar = ProgressBar(args.n_examples, "Detecting...", color="cyan")
    for start in range(0, args.n_examples, args.batch):
        end = min(start + args.batch, args.n_examples)
        prediction = model(path_list[start:end])
        storage = pd.concat([storage, prediction])
        pbar.update(end)
    storage.to_csv(f"{dataset.datadir}/detections_{args.n_examples}.csv", index=False)
    pbar.end()


if __name__ == "__main__":
    args = argparser()
    main()

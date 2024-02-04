from argparse import ArgumentParser

import pandas as pd
import torch

from algorithm import load_algorithm
from dataset import load_dataset
from evaluator import evaluate
from utils import config_parser, printc, rename_dir, save_error


def argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-r",
        "--read",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-p",
        "--param",
        type=str,
        nargs="*",
        help="e.g.) -p solver='gurobi' thread=4",
    )
    args = parser.parse_args()
    return args

torch.set_num_threads(1)

@save_error()
def main():
    # load dataset
    all_detection = pd.read_csv(config.read)
    files = all_detection["file"].unique()
    dataset = load_dataset(config.read)
    config.savedir = rename_dir(f"../result/{config.dataset}/{config.algorithm}")
    config_parser.save(f"{config.savedir}/config.json")

    # run algorithm
    algorithm = load_algorithm()
    detection = algorithm.run(dataset, all_detection)
    detection.to_csv(f"{config.savedir}/result.csv", index=False)

    # evaluate
    ground_truth = dataset.load_ground_truth(files)
    ground_truth.to_csv(f"{config.savedir}/ground_truth.csv", index=False)
    result = evaluate(files, detection, ground_truth)
    printc("yellow", result)
    config_parser.save(f"{config.savedir}/config.json")


if __name__ == "__main__":
    args = argparser()
    config = config_parser.read(args, args.config)
    main()

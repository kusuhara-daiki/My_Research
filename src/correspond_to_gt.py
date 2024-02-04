import os
from argparse import ArgumentParser

import pandas as pd
from matplotlib import pyplot as plt

from dataset import load_dataset
from utils import ProgressBar


def argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-r1",
        "--read1",
        type=str,
        default="../result/coco2017/NMS_Algorithm1/result.csv",
    )
    parser.add_argument(
        "-r2",
        "--read2",
        type=str,
        default="../result/coco2017/QAQS_histogram1/result.csv",
    )
    parser.add_argument(
        "-gt",
        "--groundtruth",
        type=str,
        default="../result/coco2017/gt100.csv",
    )
    args = parser.parse_args()
    return args


def make_df():
    # os.makedirs(f"{os.path.dirname(args.groundtruth)}/DiffGT", exist_ok=True)

    ground_truth_df = pd.read_csv(args.groundtruth)
    dataset1 = load_dataset(args.read1)
    detection1 = pd.read_csv(args.read1)
    detection2 = pd.read_csv(args.read2)
    file_names = detection1["file"].unique()
    diff_df = pd.DataFrame(columns=detection1.columns.values)
    # 画像データごとに小数第1位を四捨五入して誤差を許容し,片方のみに含まれているデータを取得
    for file in file_names:
        diff_detection = detection2[detection2["file"] == file][
            ~detection2[detection2["file"] == file]["xmin"]
            .round()
            .isin(detection1[detection1["file"] == file]["xmin"].round())
        ]
        diff_df = pd.concat([diff_df, diff_detection])
    # それぞれにflag追加
    ground_truth_df = ground_truth_df.assign(diff=0)
    detection1 = detection1.assign(diff=1)
    diff_df = diff_df.assign(diff=2)
    # 結合して新しいdataframeを定義
    new_df = pd.concat([ground_truth_df, detection1])
    new_df = pd.concat([new_df, diff_df])
    return dataset1, new_df


def main():
    algrithm_name1 = f"{os.path.dirname(args.read1)}/".split("/")[-2]
    algrithm_name2 = f"{os.path.dirname(args.read2)}/".split("/")[-2]
    savedir = os.path.dirname(args.groundtruth)
    os.makedirs(f"{savedir}/DiffGT/{algrithm_name1}.vs.{algrithm_name2}", exist_ok=True)
    # 違いを描画するために不足があった画像データの名前だけ取得する
    files = new_df[new_df["diff"] == 2]["file"].unique()
    pbar = ProgressBar(len(files), "Save images", color="cyan")
    for file in files:
        plt.subplots(figsize=(8, 8))
        plt.imshow(dataset1.load_image(file, "rgb").squeeze())
        detected_boxes = new_df[new_df["file"] == file]
        for _, bbox in detected_boxes.iterrows():
            if bbox["diff"] == 0:
                plt.plot(
                    [
                        bbox["xmin"],
                        bbox["xmax"],
                        bbox["xmax"],
                        bbox["xmin"],
                        bbox["xmin"],
                    ],
                    [
                        bbox["ymin"],
                        bbox["ymin"],
                        bbox["ymax"],
                        bbox["ymax"],
                        bbox["ymin"],
                    ],
                    color="red",
                    linewidth=0.8,
                )
            elif bbox["diff"] == 1:
                plt.plot(
                    [
                        bbox["xmin"],
                        bbox["xmax"],
                        bbox["xmax"],
                        bbox["xmin"],
                        bbox["xmin"],
                    ],
                    [
                        bbox["ymin"],
                        bbox["ymin"],
                        bbox["ymax"],
                        bbox["ymax"],
                        bbox["ymin"],
                    ],
                    label=bbox["label"],
                    color="blue",
                    linestyle="--",
                )
            else:
                plt.plot(
                    [
                        bbox["xmin"],
                        bbox["xmax"],
                        bbox["xmax"],
                        bbox["xmin"],
                        bbox["xmin"],
                    ],
                    [
                        bbox["ymin"],
                        bbox["ymin"],
                        bbox["ymax"],
                        bbox["ymax"],
                        bbox["ymin"],
                    ],
                    label=f"only {algrithm_name2}",
                    color="green",
                    linestyle=":",
                )
        if len(detected_boxes) > 0:  # escape warning
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.axis("off")
        plt.savefig(
            f"{savedir}/DiffGT/{algrithm_name1}.vs.{algrithm_name2}/{file}",
            bbox_inches="tight",
        )  # passの訂正
        plt.close()
        pbar.step()
    pbar.end()


if __name__ == "__main__":
    args = argparser()
    dataset1, new_df = make_df()
    main()

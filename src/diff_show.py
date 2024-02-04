import os
from argparse import ArgumentParser

import pandas as pd
from matplotlib import pyplot as plt

# sys.path.append("..")
from dataset import load_dataset
from utils import ProgressBar


def argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-r1",
        "--read1",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-r2",
        "--read2",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args


def main():
    os.makedirs(f"{os.path.dirname(args.read1)}/comparison", exist_ok=True)
    # データ読み込み
    nms_dataset = load_dataset(args.read1)
    nms_detection = pd.read_csv(args.read1)
    qaqs_detection = pd.read_csv(args.read2)
    # 違いを抽出
    diff_detection = qaqs_detection[~qaqs_detection["xmin"].isin(nms_detection["xmin"])]
    # それぞれにflag追加
    nms_detection = nms_detection.assign(diff=0)
    diff_detection = diff_detection.assign(diff=1)
    # 結合して新しいdataframeを定義
    new_df = pd.concat([nms_detection, diff_detection])
    # 違いを描画するために不足があった画像データの名前だけ取得する
    files = new_df[new_df["diff"] == 1]["file"].unique()
    pbar = ProgressBar(len(files), "Save images", color="cyan")

    for file in files:
        plt.subplots(figsize=(8, 8))
        plt.imshow(nms_dataset.load_image(file, "rgb").squeeze())
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
                    label=bbox["label"],
                    color="red",
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
                    label="defect",
                    color="blue",
                )
        if len(detected_boxes) > 0:  # escape warning
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.axis("off")
        plt.savefig(
            f"{os.path.dirname(args.read1)}/comparison/{file}", bbox_inches="tight"
        )
        plt.close()
        pbar.step()
    pbar.end()


if __name__ == "__main__":
    args = argparser()
    main()

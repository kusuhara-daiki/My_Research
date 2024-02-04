import cv2
from numpy.typing import NDArray


class Dataset:
    def __init__(self):
        self.color = {
            "rgb": cv2.COLOR_BGR2RGB,
            "gray": cv2.COLOR_BGR2GRAY,
            "hsv": cv2.COLOR_BGR2HSV,
            "lab": cv2.COLOR_BGR2LAB,
        }

    def get_files(self, n_examples: int):
        raise NotImplementedError

    def load_image(self, image_id: NDArray, color: str):
        raise NotImplementedError

    def load_ground_truth(self, files: NDArray):
        raise NotImplementedError

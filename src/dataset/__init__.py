from utils import config_parser

from ._coco2017 import coco_label
from .city_persons import CityPersons
from .coco2017 import Coco2017
from .crowd_human import CrowdHuman
from .wider_person import WiderPerson

config = config_parser()


def load_dataset(dataset_name: str):
    if "coco2017" in dataset_name:
        config.dataset = "coco2017"
        dataset = Coco2017()
    elif "crowd_human" in dataset_name:
        config.dataset = "crowd_human"
        dataset = CrowdHuman()
    elif "wider_person" in dataset_name:
        config.dataset = "wider_person"
        dataset = WiderPerson()
    elif "city_persons" in dataset_name:
        config.dataset = "city_persons"
        dataset = CityPersons()
    else:
        raise NotImplementedError(dataset_name)
    return dataset


__all__ = [
    "coco_label",
]

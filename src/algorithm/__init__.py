from base import Algorithm
from utils import config_parser

from .nms_algorithm import NMS_Algorithm
from .qaqs_akaze import QAQS_akaze
from .qaqs_orb import QAQS_orb
from .qaqs_hash import QAQS_hash
from .qaqs_histogram import QAQS_histogram
from .qsqs import QSQS_Algorithm
from .soft_nms import SoftNMS
from .qaqs_ssim import QAQS_ssim
from .qaqs_ssimv2 import QAQS_ssimv2
from .qaqs_ssim_batch import QAQS_ssim_batch

config = config_parser()


def load_algorithm() -> Algorithm:
    if config.algorithm == "NMS_Algorithm":
        algorithm = NMS_Algorithm()
    elif config.algorithm == "SoftNMS":
        algorithm = SoftNMS()
    elif config.algorithm == "QSQS_Algorithm":
        algorithm = QSQS_Algorithm()
    elif config.algorithm == "QAQS_histogram":
        algorithm = QAQS_histogram()
    elif config.algorithm == "QAQS_akaze":
        algorithm = QAQS_akaze()
    elif config.algorithm == "QAQS_orb":
        algorithm = QAQS_orb()
    elif config.algorithm == "QAQS_hash":
        algorithm = QAQS_hash()
    elif config.algorithm == "QAQS_ssim":
        algorithm = QAQS_ssim()
    elif config.algorithm == "QAQS_ssimv2":
        algorithm = QAQS_ssimv2()
    elif config.algorithm == "QAQS_ssim_batch":
        algorithm = QAQS_ssim_batch()
    else:
        raise NotImplementedError(config.algorithm)
    return algorithm

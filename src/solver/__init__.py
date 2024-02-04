from base import Solver
from utils import config_parser

from .gurobi import Gurobi

config = config_parser()


def get_solver() -> Solver:
    if config.solver == "gurobi":
        solver = Gurobi()
    else:
        raise NotImplementedError(config.solver)
    return solver

import os
from contextlib import redirect_stdout

import amplify
import gurobipy as gp
import numpy as np

from base import Solver
from utils import config_parser

config = config_parser()


class Gurobi(Solver):
    def __init__(self):
        super().__init__()
        with redirect_stdout(open(os.devnull, "w")):
            gp.setParam(gp.GRB.param.OutputFlag, 0)
            gp.setParam(gp.GRB.param.Threads, config.gurobi_thread)
            gp.setParam(gp.GRB.param.TimeLimit, config.gurobi_timeout)

    def solve(self, problem: amplify.BinaryPoly):
        assert problem.symbol == "q"  # binary problem
        model = gp.Model()
        num_node = problem.max_index() + 1
        x = [model.addVar(vtype=gp.GRB.BINARY) for _ in range(num_node)]
        model.update()
        problem = problem.asdict()
        model.setObjective(
            gp.quicksum(
                v * x[k[0]] if len(k) == 1 else v * x[k[0]] * x[k[1]]
                for k, v in problem.items()
            ),
            gp.GRB.MINIMIZE,
        )
        model.optimize()
        solution = np.array([int(_x.x) for _x in x]) 
        # solution = np.array([round(_x.x) for _x in x]　解を0,1にしたいならroundで丸める？
        return solution

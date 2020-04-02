import gurobipy as gp

from config import common_config


def get_rounded_val(v):
    if(v < 0):
        return 0
    if(v < common_config.ftol):
        return 0
    return v


def get_val(v):
    if(isinstance(v, (float, int))):
        return v
    if(isinstance(v, gp.Var)):
        return v.x
    else:
        return v.getValue()

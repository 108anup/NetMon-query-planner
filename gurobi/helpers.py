import gurobipy as gp

import logging

from common import log
from config import common_config
from functools import partial


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


def log_vars(m, logger=partial(log.log, logging.DEBUG-1)):
    logger("\nVARIABLES "+"-"*30)
    logger("Objective: {}".format(m.objVal))
    for v in m.getVars():
        logger('%s %g' % (v.varName, v.x))

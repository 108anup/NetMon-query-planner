'''
Goal: Find best overlay for a given input
'''
import numpy as np
import random

from scipy.optimize import basinhopping

from input import input_generator
from main import solve
from config import common_config
from common import setup_logging


inp = input_generator[28]
common_config.solver = 'Netmon'
common_config.vertical_partition = True
common_config.prog_dir = None
setup_logging(common_config)


class TakeStep():
    def __init__(self, stepsize):
        self.stepsize = stepsize

    def __call__(self, perm):
        n = len(perm)
        l = random.randint(2, n - 1)
        i = random.randint(0, n - l)
        perm[i: (i + l)] = reversed(perm[i: (i + l)])
        return perm


def func(overlay):
    # import ipdb; ipdb.set_trace()
    # Split into parts with 10 elements each
    splits = np.split(np.array(overlay[:980]), 98)
    splits = [x.astype(int).tolist() for x in splits]
    splits[-1].extend(overlay[980:].astype(int))
    inp.overlay = splits
    (ns, res) = solve(inp)
    return (res / ns)


def flatten(l):
    ret = []
    for x in l:
        if(isinstance(x, list)):
            ret.extend(flatten(x))
        else:
            ret.append(x)
    return ret

# import ipdb; ipdb.set_trace()
ret = basinhopping(func, flatten(inp.overlay),
                   niter=1000)

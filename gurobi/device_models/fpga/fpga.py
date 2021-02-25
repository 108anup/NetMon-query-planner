from collections import namedtuple
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import pprint
from scipy.interpolate import interp1d
from operator import itemgetter
import sys
import os
import csv
import palettable
from util import get_fig_size
from common import (hash_plot, get_hash_params,
                    read_bench_util, CELL_SIZE,
                    KB2B, plot_mem, get_mem_label,
                    evaluation_plot, sketch_params)
from functools import partial

# * Globals
bench_dir = sys.argv[1]
plot_dir = os.path.join(bench_dir, 'plots')
if(not os.path.isdir(plot_dir)):
    os.mkdir(plot_dir)
read_bench = partial(read_bench_util, bench_dir)


# * Start
def update_entry(x):
    x.ns = 1e3/x.mpps
    x.cols = 1 << x.logcols
    if(hasattr(x, 'logcols_emem')):
        x.cols += 1 << x.logcols_emem
    # Mem in KibiBytes
    x.mem = x.rows * x.cols * CELL_SIZE / KB2B


def update_entry_single_hash_unit(x):
    update_entry(x)
    x.ns = (x.hash_units * 1e3)/x.mpps

# ** Hash params
hash_bench = read_bench('hash.csv', update_entry_single_hash_unit)
hash_plot(hash_bench, os.path.join(plot_dir, 'fpga-hash-hs.pdf'))
start_idx = 3  # FIXME: manual
hashing_const, hashing_slope = get_hash_params(hash_bench, start_idx)


def get_hash_time(x, hpr, additional_hashes):
    return (
        (hashing_const
         + hashing_slope * (x.rows * hpr + additional_hashes))
        / x.hash_units
    )


# ** Fwd params
fwd_ns = hash_bench[0].ns

# ** Mem params
lmem_bench = read_bench('lmem.csv', update_entry)
gmem_bench = read_bench('gmem.csv', update_entry)
mem_bench = lmem_bench + gmem_bench
grouped_by_row = {}
for x in mem_bench:
    grouped_by_row.setdefault(x.rows, []).append(x)

rows = list(grouped_by_row.keys())
assert(len(rows) == 2)
mem_bench_1 = grouped_by_row[rows[0]]
mem_bench_2 = grouped_by_row[rows[1]]
plot_mem(mem_bench_1, mem_bench_2,
         os.path.join(plot_dir, 'fpga-mem-half.pdf'))


def get_mem_params(bench_list_1, bench_list_2, mem_const, contract_till):
    # FIXME: manual
    x_vals = list(map(lambda x: x.mem, bench_list_2))
    y_vals = list(map(lambda x: (x.ns-mem_const)/x.rows, bench_list_2))

    ys = []
    xs = []

    cidx = contract_till-1
    xs.extend(x_vals[cidx+1:])
    ys.extend(y_vals[cidx+1:])

    # lmem is never bottleneck
    xs.insert(0, x_vals[cidx])
    ys.insert(0, 0)

    xs.insert(0, 0)
    ys.insert(0, 0)

    MAX_LMEM_COLS = 2**10
    MAX_GMEM_COLS = 2**20
    MAX_ROWS = 12
    X_LAST = MAX_ROWS * CELL_SIZE * (MAX_LMEM_COLS + MAX_GMEM_COLS) / KB2B
    # Y_LAST = ys[-1] + (X_LAST-xs[-1])*((ys[-1] - ys[-2])/(xs[-1] - xs[-2]))
    Y_LAST = np.max(ys)  # FIXME: manual saturated here 

    xs.append(X_LAST)
    ys.append(Y_LAST)

    get_mem_access_time = interp1d(xs, ys)
    print("mem access: xs, ys:")
    pprint.pprint(xs)
    pprint.pprint(ys)
    print("mem const: ", mem_const)

    return xs, ys, get_mem_access_time


mem_const = 0  # FIXME: hit and trial fitting
contract_till = 3  # FIXME: manual (3 points are flat)
xs, ys, get_mem_access_time = get_mem_params(
    mem_bench_1, mem_bench_2, 0, contract_till)


def get_mem_time(x, hpr=1):
    x.m_access_time = get_mem_access_time(x.mem)
    if(hasattr(x, 'levels')):
        x.m_access_time = get_mem_access_time(x.mem * x.levels)
    return (mem_const + x.rows * x.m_access_time)


# ** Full model
def model(x, hpr, additional_hashes, diff):
    x.hash_ns = get_hash_time(x, hpr, additional_hashes)
    x.mem_ns = get_mem_time(x, hpr)
    items = (fwd_ns, x.hash_ns, x.mem_ns)
    x.argmax, x.model_ns = max(enumerate(items), key=itemgetter(1))
    diff.append(abs(x.ns - x.model_ns) / x.ns)


# ** Evaluation
sketches = list(sketch_params.keys())
for skid, sketch in enumerate(sketches):
    ground_truths = {}
    bench_list = []
    ground_truth = read_bench('ground_truth_{}.csv'.format(sketch),
                              update_entry)
    bench_list.extend(ground_truth)
        
    if(sketch == 'count-min-sketch'):
        bench_list.extend(mem_bench)
        bench_list.extend(hash_bench)
        amdahl_bench = read_bench('amdahls_hash.csv', update_entry)
        pprint.pprint(amdahl_bench)
        bench_list.extend(amdahl_bench)

    def get_key(x):
        items = [x.hash_units, x.rows, x.cols]
        return tuple(items)

    bench_list.sort(key=lambda x: (x.rows, x.hash_units, x.cols))
    if(sketch == 'univmon'):
        bench_list.sort(
            key=lambda x: (x.hash_units, x.levels, x.rows, x.cols))

    diff = []
    for x in bench_list:
        model(x, sketch_params[sketch].hpr,
              sketch_params[sketch].additional_hashes, diff)

    # TODO: group by hash_units
    extra = ""
    file_path = os.path.join(
        plot_dir, 'fpga-model-{}-{}-hs.pdf'.format(sketch, extra))
    evaluation_plot([x for x in bench_list], sketch, file_path)

    if(len(diff) > 0):
        print("Relative Error [{}]: ".format(sketch), np.average(diff))

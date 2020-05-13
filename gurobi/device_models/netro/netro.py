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

# TODO: get rid of things done by hand, by using derivatives as done in:
# https://stackoverflow.com/questions/29382903/how-to-apply-piecewise-linear-fit-in-python

# * Config
plt.rcParams.update({'font.size': 10,
                     'axes.linewidth': 1,
                     'xtick.major.size': 5,
                     'xtick.major.width': 1,
                     'xtick.minor.size': 2,
                     'xtick.minor.width': 1,
                     'ytick.major.size': 5,
                     'ytick.major.width': 1,
                     'ytick.minor.size': 2,
                     'ytick.minor.width': 1})

# no-branching
mem_params = {
    'FLAT_REGION': [6, 14],
    'COMBINE': [1, 4, 10],
    'DOMINANT': 7
}

# with-branching
mem_params = {
    'FLAT_REGION': [9, 15],
    'COMBINE': [1, 8, 7],
    'DOMINANT': 14,
    'MEM_CONST': 13.460474472023066
}


# * Constants
CELL_SIZE = 4
KB2B = 1024

# * Globals
bench_dir = sys.argv[1]


def read_bench(fname):
    bench = []
    try:
        with open(os.path.join(bench_dir, fname)) as f:
            csvreader = csv.reader(f)
            for line in csvreader:
                entry = []
                for e in line:
                    entry.append(int(e))
                bench.append(tuple(entry))
    except FileNotFoundError:
        pass
    return bench


# * Start
hash_bench = read_bench('hash.csv')
mem_bench = read_bench('mem.csv')
half_len_mem_bench = int(len(mem_bench)/2)
mem_bench_1 = mem_bench[:half_len_mem_bench]
mem_bench_2 = mem_bench[half_len_mem_bench:]

# * Forwarding params
# Taken from no-branching/hash.csv
fwd_thr = 27564551
fwd_ns = 1e9/fwd_thr
print("fwd_ns: ", fwd_ns)

branching_slope = 1e9/hash_bench[0][2] - fwd_ns
print("branching_slope: ", branching_slope)

# * Hashing params
bench_list = [SimpleNamespace(rows=x[0], cols=x[1], pps=x[2])
              for x in hash_bench]
for x in bench_list:
    x.ns = 1e9/x.pps

# plt.plot(list(map(lambda x: x.rows, bench_list)),
#          list(map(lambda x: x.ns, bench_list)), '*-')
# plt.show()

# Linear model
# Hand identified when hashing is bottleneck
start = 4
end = len(bench_list)
start_idx = start - 1
end_idx = end - 1

hashing_x = start
hashing_y = bench_list[start_idx].ns
hashing_slope = (np.average([((bench_list[i+1].ns - bench_list[i].ns) /
                              (bench_list[i+1].rows - bench_list[i].rows))
                             for i in range(start_idx, end_idx)])
                 - branching_slope)
hashing_const = (hashing_y - (hashing_slope + branching_slope) * hashing_x)
print("hashing: x, y, slope:", hashing_x, hashing_y, hashing_slope)
print("hashing_const: ", hashing_const)

# # None linear model
# # PWL by hand
# xs = [1, 3, 9, 12]
# ys = []
# for r in xs:
#     ys.append(bench_list[r-1].ns)
# xs[0] = 0
# print("hashing non linear: xs, ys: ")
# pprint.pprint(xs)
# pprint.pprint(ys)
# hashing_time = interp1d(xs, ys)

# * Mem params
bench_list_1 = [SimpleNamespace(rows=x[0], cols=x[1], pps=x[2])
                for x in mem_bench_1]
bench_list_2 = [SimpleNamespace(rows=x[0], cols=x[1], pps=x[2])
                for x in mem_bench_2]
for x in bench_list_1:
    x.ns = 1e9/x.pps
    x.mem = x.rows * x.cols * CELL_SIZE / KB2B
for x in bench_list_2:
    x.ns = 1e9/x.pps
    x.mem = x.rows * x.cols * CELL_SIZE / KB2B

bench_1_ns = list(map(lambda x: x.ns, bench_list_1))
bench_2_ns = list(map(lambda x: x.ns, bench_list_2))

FLAT_REGION = mem_params['FLAT_REGION']
p1 = bench_list_1[0]
p2 = bench_list_2[0]
flat_part_slope = ((np.average(bench_2_ns[FLAT_REGION[0]:FLAT_REGION[1] + 1])
                    - np.average(bench_1_ns[FLAT_REGION[0]:FLAT_REGION[1] + 1]))
                   / (p2.rows - p1.rows))
print("Mem: flat part slope: ", flat_part_slope)
DOMINANT = mem_params['DOMINANT']
point = bench_list_2[DOMINANT]
flat_part_val = np.average(bench_2_ns[FLAT_REGION[0]:FLAT_REGION[1] + 1])
mem_const = flat_part_val - (flat_part_slope) * point.rows
if('MEM_CONST' in mem_params):
    mem_const = mem_params['MEM_CONST']
# mem_const = 16

# pprint.pprint(bench_list_1)
# pprint.pprint(bench_list_2)

# x_vals = [(bench_list_1[i].mem + bench_list_2[i].mem)/2
#           for i in range(len(bench_list_1))]
# y_vals = [bench_list_2[i].ns - bench_list_1[i].ns
#           for i in range(len(bench_list_2))]
x_vals = list(map(lambda x: x.mem, bench_list_2))
y_vals = list(map(lambda x: (x.ns-mem_const)/x.rows, bench_list_2))

# pprint.pprint(x_vals)
# pprint.pprint(y_vals)

# plt.plot(x_vals, y_vals)
# plt.show()

COMBINE = mem_params['COMBINE']
# print(x_vals[:COMBINE])
# print(y_vals[:COMBINE])

ys = []
xs = []

idx = 0
for c in COMBINE:
    if c == 1:
        ys.append(y_vals[idx])
        xs.append(x_vals[idx])
        idx += 1
    else:
        y_avg = max(y_vals[idx:idx+c-1])
        ys.extend([y_avg, y_avg])
        xs.append(x_vals[idx])
        xs.append(x_vals[idx+c-1])
        idx += c
xs.extend(x_vals[idx:])
ys.extend(y_vals[idx:])

# y_start = np.average(y_vals[:COMBINE])
# ys = [y_start, y_start]
# ys.extend(y_vals[COMBINE:])
# xs = [0, x_vals[COMBINE-1]]
# xs.extend(x_vals[COMBINE:])

xs.insert(0, 0)
ys.insert(0, ys[0])

X_LAST = 200000
Y_LAST = ys[-1] + (X_LAST-xs[-1])*((ys[-1] - ys[-2])/(xs[-1] - xs[-2]))

xs.append(X_LAST)
ys.append(Y_LAST)

# plt.plot(xs, ys)
# plt.show()

# correct for branching
# branching const will be zero if no-branching
ys = [y - branching_slope for y in ys]

get_mem_access_time = interp1d(xs, ys)
# # By hand
# dominant = 8
# point = bench_list_1[dominant]
# mem_const = (
#     point.ns - point.rows * get_mem_access_time(point.mem))
#     # - (hashing_y + hashing_slope * (point.rows - hashing_x))
# # )
print("mem access: xs, ys:")
pprint.pprint(xs)
pprint.pprint(ys)
print("mem const: ", mem_const)

# Evaluation:
ground_truth_54 = read_bench('ground_truth_54.csv')
ground_truth_36 = read_bench('ground_truth_36.csv')
ground_truth_20 = read_bench('ground_truth_20.csv')

MAX_ME = 54
bench_list = [SimpleNamespace(rows=x[0], cols=x[1], pps=x[2], me=54)
              for x in ground_truth_54]
bench_list.extend([SimpleNamespace(rows=x[0], cols=x[1], pps=x[2], me=54)
                   for x in mem_bench])
bench_list.extend([SimpleNamespace(rows=x[0], cols=x[1], pps=x[2], me=54)
                   for x in hash_bench])
bench_list.extend([SimpleNamespace(rows=x[0], cols=x[1], pps=x[2], me=36)
                   for x in ground_truth_36])
bench_list.extend([SimpleNamespace(rows=x[0], cols=x[1], pps=x[2], me=20)
                   for x in ground_truth_20])
bench_list.sort(key=lambda x: (x.rows, x.cols, x.me))

diff = []
for x in bench_list:
    x.mem = x.rows * x.cols * CELL_SIZE / KB2B
    x.ns = 1e9/x.pps
    x.hash_ns = (
        (hashing_const + hashing_slope * x.rows
         + branching_slope * x.rows)
        * MAX_ME / x.me
    )
    x.m_access_time = get_mem_access_time(x.mem)
    x.mem_ns = (mem_const +
                x.rows * x.m_access_time + x.rows * branching_slope)
    items = (fwd_ns * MAX_ME / x.me,
             # hashing_time(x.rows),
             x.hash_ns, x.mem_ns)
    x.argmax, x.model_ns = max(enumerate(items), key=itemgetter(1))
    x.argmax = x.argmax * x.me / 5
    diff.append(abs(x.ns - x.model_ns) / x.ns)


def myplot(bench_list, me):
    fig = plt.figure(figsize=(4, 4))
    labels = list(map(lambda x: str(x.rows) + ", " + str(x.mem), bench_list))
    plt.plot(labels, list(map(lambda x: x.ns, bench_list)), '.-', linewidth=2,
             label='Ground Truth')
    plt.plot(labels, list(map(lambda x: x.model_ns, bench_list)), linewidth=2,
             label="Model")
    plt.plot(labels, list(map(lambda x: x.argmax, bench_list)), linewidth=2,
             label="Bottleneck")
    # plt.plot(labels, list(map(lambda x: x.hash_ns, bench_list)), linewidth=2,
    #          label="Hashing")
    # plt.plot(labels, list(map(lambda x: x.mem_ns, bench_list)), linewidth=2,
    #          label="Mem")

    num_labels = len(labels)
    plt.xticks(labels[::int(num_labels/20)])
    plt.xticks(rotation=90)
    plt.xlabel("Sketch configuration (rows, mem KB)")
    plt.ylabel("Time per packet (ns)")
    plt.title("Netronome profile for {} MEs".format(me))
    plt.legend()
    # pprint.pprint(bench_list)
    plt.show()
    fig.tight_layout()
    # plt.savefig('netro-model-36me.png')


myplot([x for x in bench_list if x.me == 54], "54")
myplot([x for x in bench_list if x.me == 36], "36")
# myplot([x for x in bench_list if x.me == 20], "20")
print("Relative Error: ", np.average(diff))
# pprint.pprint(bench_list)

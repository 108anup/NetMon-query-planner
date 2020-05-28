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


# * Plottging Config
FONT_SIZE = 8
MARKER_SIZE = 2
LINE_WIDTH = 1
HANDLE_LENGTH = 2.5

colors = palettable.colorbrewer.qualitative.Paired_10.hex_colors
linestyles = ['-', '--']

# plt.rcParams.update({'font.size': 10,
#                      'axes.linewidth': 1,
#                      'xtick.major.size': 5,
#                      'xtick.major.width': 1,
#                      'xtick.minor.size': 2,
#                      'xtick.minor.width': 1,
#                      'ytick.major.size': 5,
#                      'ytick.major.width': 1,
#                      'ytick.minor.size': 2,
#                      'ytick.minor.width': 1})
plt.rc('pdf', fonttype=42)
plt.rc('font', **{'size': FONT_SIZE})

# with-branching
mem_params = {
    'FLAT_REGION': [0, 0],
    'DOMINANT': 0,
}


# * Constants
CELL_SIZE = 4
KB2B = 1024

# * Globals
bench_dir = '.'


def read_bench(fname):
    bench = []
    try:
        with open(os.path.join(bench_dir, fname)) as f:
            csvreader = csv.reader(f)
            for line in csvreader:
                entry = []
                for e in line:
                    entry.append(float(e))
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
fwd_thr_single_core = 5.746337927
fwd_ns = 1000/fwd_thr_single_core
print("fwd_ns: ", fwd_ns)

# * Hashing params
bench_list = [SimpleNamespace(cores=x[0], rows=x[1], cols_per_core=x[2], Mpps=x[3])
              for x in hash_bench]
for x in bench_list:
    x.ns_single = x.cores * 1000/x.Mpps
    x.ns = x.cores * 1000/x.Mpps

fig, ax = plt.subplots(figsize=get_fig_size(0.5, 0.6))
plt.plot(list(map(lambda x: x.rows, bench_list)),
         list(map(lambda x: x.ns_single, bench_list)),
         color=colors[5], marker='s', markersize=MARKER_SIZE,
         lw=LINE_WIDTH, linestyle=linestyles[0], clip_on=False)
ax.set_xlabel('Number of sketch\nupdates per packet', fontsize=FONT_SIZE)
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(labelsize=FONT_SIZE, pad=2)

ax.set_ylabel("Time per\npacket (ns)", fontsize=FONT_SIZE)
ax.yaxis.set_ticks_position('left')

# ax.set_xscale("log", basex=10)
# ax.set_yticklabels(['1M','10M','100M'])
# ax.yaxis.set_label_coords(-0.19, 0.43)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

plt.minorticks_on()
plt.savefig(os.path.join(bench_dir, 'cpu-hash-hs.pdf'), bbox_inches='tight')

# Linear model
# Hand identified when hashing is bottleneck
start = 1
end = len(bench_list)
start_idx = start - 1
end_idx = end - 1

hashing_x = start
hashing_y = bench_list[start_idx].ns
hashing_slope = (np.average([((bench_list[i+1].ns - bench_list[i].ns) /
                              (bench_list[i+1].rows - bench_list[i].rows))
                             for i in range(start_idx, end_idx)]))
hashing_const = (hashing_y - (hashing_slope) * hashing_x)
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
bench_list_1 = [SimpleNamespace(cores=x[0], rows=x[1], cols_per_core=x[2], Mpps=x[3])
                for x in mem_bench_1]
bench_list_2 = [SimpleNamespace(cores=x[0], rows=x[1], cols_per_core=x[2], Mpps=x[3])
                for x in mem_bench_2]
for x in bench_list_1:
    x.ns = 1000/x.Mpps * x.cores
    x.cols = x.cols_per_core * x.cores
    x.mem = x.rows * x.cols * CELL_SIZE / KB2B
for x in bench_list_2:
    x.ns = 1000/x.Mpps * x.cores
    x.cols = x.cols_per_core * x.cores
    x.mem = x.rows * x.cols * CELL_SIZE / KB2B

fig, ax = plt.subplots(figsize=get_fig_size(1, 0.5))
plt.plot(list(map(lambda x: x.mem/1024, bench_list_1)),
         list(map(lambda x: x.ns, bench_list_1)),
         color=colors[5], marker='s', markersize=MARKER_SIZE,
         lw=LINE_WIDTH, linestyle=linestyles[0], clip_on=False,
         label='{} updates per packet'.format(int(bench_list_1[0].rows)))
plt.plot(list(map(lambda x: x.mem/1024, bench_list_2)),
         list(map(lambda x: x.ns, bench_list_2)),
         color=colors[1], marker='^', markersize=MARKER_SIZE,
         lw=LINE_WIDTH, linestyle=linestyles[0], clip_on=False,
         label='{} updates per packet'.format(int(bench_list_2[0].rows)))
# plt.plot(list(map(lambda x: x.mem/1024, bench_list_2)),
#          list(map(lambda x: fwd_ns, bench_list_2)),
#          color=colors[3],
#          lw=LINE_WIDTH, linestyle=linestyles[1], clip_on=False,
#          label='Vanilla OVS-DPDK'.format(bench_list_2[0].rows))

ax.set_xlabel('Total sketch memory (MB)', fontsize=FONT_SIZE)
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(labelsize=FONT_SIZE, pad=2)

ax.set_ylabel("Time per\npacket (ns)", fontsize=FONT_SIZE)
ax.yaxis.set_ticks_position('left')

legend = plt.legend(loc='upper left', numpoints=1,
                    ncol=1, prop={'size': FONT_SIZE}, columnspacing=0.5,
                    handlelength=HANDLE_LENGTH, handletextpad=0.5)
legend.set_frame_on(False)

ax.set_xscale("log", basex=2)
ax.set_xticks([2**(x-6) for x in range(0, 16, 2)])

# ax.set_yscale("log", basey=10)
# ax.set_yticks([200 + 50*x for x in range(9)])
# ax.set_yticks([0.5, 1, 5, 10, 50, 100, 200, 400])
# ax.set_yticklabels(['1M','10M','100M'])
# ax.yaxis.set_label_coords(-0.19, 0.43)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

plt.minorticks_on()
plt.savefig(os.path.join(bench_dir, 'cpu-mem-hs.pdf'), bbox_inches='tight')

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
print("Mem: mem_const: ", mem_const)

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

# plt.subplots()
# plt.plot(x_vals, y_vals)
# plt.xscale('log')
# plt.show()

# COMBINE = mem_params['COMBINE']
# print(x_vals[:COMBINE])
# print(y_vals[:COMBINE])

ys = []
xs = []

idx = 0
# for c in COMBINE:
#     if c == 1:
#         ys.append(y_vals[idx])
#         xs.append(x_vals[idx])
#         idx += 1
#     else:
#         y_avg = max(y_vals[idx:idx+c-1])
#         ys.extend([y_avg, y_avg])
#         xs.append(x_vals[idx])
#         xs.append(x_vals[idx+c-1])
#         idx += c
xs.extend(x_vals[idx:])
ys.extend(y_vals[idx:])

# y_start = np.average(y_vals[:COMBINE])
# ys = [y_start, y_start]
# ys.extend(y_vals[COMBINE:])
# xs = [0, x_vals[COMBINE-1]]
# xs.extend(x_vals[COMBINE:])

xs.insert(0, 0)
ys.insert(0, ys[0])

X_LAST = 4194304 * 12 * 4 * 4 / 1024
Y_LAST = ys[-1] + (X_LAST-xs[-1])*((ys[-1] - ys[-2])/(xs[-1] - xs[-2]))

xs.append(X_LAST)
ys.append(Y_LAST)

# plt.plot(xs, ys)
# plt.show()

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
ground_truth = read_bench('ground-truth.csv')
bench_list = [SimpleNamespace(cores=x[0], rows=x[1], cols_per_core=x[2], Mpps=x[3])
              for x in ground_truth]
bench_list = bench_list[:168:2]
diff = []
for x in bench_list:
    x.mem = x.rows * x.cols_per_core * x.cores * CELL_SIZE / KB2B
    x.ns = 1000/x.Mpps
    x.hash_ns = (hashing_const + hashing_slope * x.rows)
    x.m_access_time = get_mem_access_time(x.mem)
    x.mem_ns = (mem_const + x.rows * x.m_access_time)
    items = (x.hash_ns, x.mem_ns)
    x.argmax, x.model_ns = max(enumerate(items), key=itemgetter(1))
    x.model_ns = (x.model_ns)/x.cores
    # x.argmax_loc = 1500 / x.me + x.argmax * 10 #  * x.me / 5
    diff.append(abs(x.ns - x.model_ns) / x.ns)
    print(x.rows, x.mem, x.ns, x.model_ns)


def get_mem_label(m):
    if(m < 1):
        return "{}".format(int(m * 1024))
    elif(m < 1024):
        return "{}K".format(int(m))
    else:
        return "{}M".format(int(m / 1024))


def myplot(bench_list, cores):

    fig, ax = plt.subplots(figsize=get_fig_size(1, 0.8))
    labels = list(map(lambda x: '{}, {}'.format(int(x.rows), get_mem_label(x.mem)), bench_list))
    ax.plot(labels, list(map(lambda x: x.ns, bench_list)),
            label='Ground Truth', color=colors[5], marker='s',
            markersize=MARKER_SIZE,
            lw=LINE_WIDTH, linestyle=linestyles[0], clip_on=False)

    ax.plot(labels, list(map(lambda x: x.model_ns, bench_list)),
            label="Model",
            color=colors[1], marker='^', markersize=MARKER_SIZE, lw=LINE_WIDTH,
            linestyle=linestyles[0], clip_on=False)
    # ax2 = ax.twinx()
    # ax.plot(labels, list(map(
    #     lambda x: x.argmax_loc, bench_list)), linewidth=2,
    #         label="Bottleneck Operation")
    # ax.plot(labels, list(map(lambda x: x.hash_ns, bench_list)), linewidth=2,
    #          label="Hashing")
    # ax.plot(labels, list(map(lambda x: x.mem_ns, bench_list)), linewidth=2,
    #          label="Mem")
    # make a plot with different y-axis using second axis object
    # ax2.set_ylabel("Bottleneck operation", color="blue", fontsize=14)

    l = plt.legend(loc='upper left', numpoints=1, ncol=1,
                   prop={'size': FONT_SIZE}, columnspacing=0.5,
                   handlelength=2.7, handletextpad=0.5)
    l.set_frame_on(False)

    ax.set_xlabel('Sketch Configuration (rows, mem in Bytes)')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(labelsize=FONT_SIZE, pad=2)

    ax.set_ylabel("Time per\npacket (ns)")
    ax.yaxis.set_ticks_position('left')

    ax.tick_params(labelsize=FONT_SIZE, pad=2)

    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    num_labels = len(labels)
    plt.xticks(labels[::int(num_labels/13)])
    plt.xticks(rotation=90)
    # plt.title("CPU profile for {} cores".format(cores))

    plt.savefig(os.path.join(bench_dir, 'cpu-model-{}cores.pdf'.format(cores)),
                bbox_inches='tight')


myplot([x for x in bench_list if x.cores == 4], "4")
myplot([x for x in bench_list if x.cores == 2], "2")
print("Relative Error: ", np.average(diff))
# pprint.pprint(bench_list)

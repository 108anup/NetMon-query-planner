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

# * Sketch params
sketch_params = {
    'count-min-sketch': SimpleNamespace(hpr=1, additional_hashes=0),
    'count-sketch': SimpleNamespace(hpr=2, additional_hashes=0),
    'univmon': SimpleNamespace(hpr=2, additional_hashes=1)
}

# * Constants
CELL_SIZE = 4
KB2B = 1024

# * Config
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


def hash_plot(bench_list, file_path):
    fig, ax = plt.subplots(figsize=get_fig_size(0.5, 0.6))
    plt.plot(list(map(lambda x: x.rows, bench_list)),
             list(map(lambda x: x.ns, bench_list)),
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
    plt.savefig(file_path, bbox_inches='tight')


# Start index is when hashing starts to become bottleneck
def get_hash_params(bench_list, start_idx):
    end = len(bench_list)
    end_idx = end - 1

    hashing_x = bench_list[start_idx].rows
    hashing_y = bench_list[start_idx].ns
    hashing_slope = np.average([((bench_list[i+1].ns - bench_list[i].ns) /
                                 (bench_list[i+1].rows - bench_list[i].rows))
                                for i in range(start_idx, end_idx)])
    hashing_const = (hashing_y - hashing_slope * hashing_x)
    print("hashing: x, y, slope:", hashing_x, hashing_y, hashing_slope)
    print("hashing_const: ", hashing_const)
    return hashing_const, hashing_slope


def plot_mem(bench_list_1, bench_list_2, file_path):
    fig, ax = plt.subplots(figsize=get_fig_size(0.5/0.9, 0.5))
    plt.plot(list(map(lambda x: x.mem/1024, bench_list_1)),
             list(map(lambda x: x.ns, bench_list_1)),
             color=colors[5], marker='s', markersize=MARKER_SIZE,
             lw=LINE_WIDTH, linestyle=linestyles[0], clip_on=False,
             label='{}'.format(int(bench_list_1[0].rows)))
    plt.plot(list(map(lambda x: x.mem/1024, bench_list_2)),
             list(map(lambda x: x.ns, bench_list_2)),
             color=colors[1], marker='^', markersize=MARKER_SIZE,
             lw=LINE_WIDTH, linestyle=linestyles[0], clip_on=False,
             label='{}'.format(int(bench_list_2[0].rows)))
    ax.set_xlabel('Total sketch\nmemory (MB)', fontsize=FONT_SIZE)
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(labelsize=FONT_SIZE, pad=2)

    ax.set_ylabel("Time per\npacket (ns)", fontsize=FONT_SIZE)
    ax.yaxis.set_ticks_position('left')

    legend = plt.legend(loc='upper left', numpoints=1, bbox_to_anchor=(0, 1.2),
                        ncol=1, prop={'size': FONT_SIZE}, columnspacing=0.5,
                        handlelength=HANDLE_LENGTH, handletextpad=0.5)
    legend.set_frame_on(False)
    ax.set_xscale("log", basex=2)
    ax.set_xticks([2**(x-12) for x in range(0, 22, 4)])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    plt.minorticks_on()
    plt.savefig(file_path, bbox_inches='tight')


def parse_entry(x, header):
    entry = SimpleNamespace()
    for hid, hname in enumerate(header):
        setattr(entry, hname, x[hid])
    return entry


def read_bench_util(bench_dir, fname, update_fun=lambda x: x):
    bench = []
    try:
        with open(os.path.join(bench_dir, fname)) as f:
            csvreader = csv.reader(f)
            for line in csvreader:
                entry = []
                for e in line:
                    if(e.strip().isnumeric()):
                        entry.append(int(e))
                    else:
                        try:
                            entry.append(float(e))
                        except ValueError:
                            entry.append(e.strip())
                bench.append(tuple(entry))
    except FileNotFoundError:
        pass

    if(len(bench) == 0):
        return bench

    # Row 0 is header
    header = bench[0]
    bench_list = bench[1:]
    parsed = [parse_entry(x, header)
              for x in bench_list]
    for x in parsed:
        update_fun(x)
    return parsed


def get_mem_label(m):
    if(m < 1):
        return "{}".format(int(m * 1024))
    elif(m < 1024):
        return "{}K".format(int(m))
    else:
        return "{}M".format(int(m / 1024))


def evaluation_plot(bench_list, sketch, file_path):
    print("SK: {}, list size: {}".format(
        sketch, len(bench_list)))
    if(len(bench_list) == 0):
        return

    fig, ax = plt.subplots(figsize=get_fig_size(1, 0.6))
    labels = list(map(lambda x: '{}, {}'.format(
        int(x.rows), get_mem_label(x.mem)), bench_list))
    if(sketch =='univmon'):
        labels = list(map(lambda x: '{}, {}, {}'.format(
            int(x.levels), int(x.rows),
            get_mem_label(x.mem * x.levels)), bench_list))
    ax.plot(labels, list(map(lambda x: x.ns, bench_list)),
            label='Ground Truth', color=colors[5], marker='s',
            markersize=MARKER_SIZE,
            lw=LINE_WIDTH, linestyle=linestyles[0], clip_on=False)

    ax.plot(labels, list(map(lambda x: x.model_ns, bench_list)),
            label="Model", color=colors[1], marker='^',
            markersize=MARKER_SIZE, lw=LINE_WIDTH,
            linestyle=linestyles[0], clip_on=False)
    
    l = plt.legend(loc='upper left', numpoints=1, ncol=1,
                   prop={'size': FONT_SIZE}, columnspacing=0.5,
                   handlelength=2.7, handletextpad=0.5)
    l.set_frame_on(False)

    ax.set_xlabel('Sketch Configuration (rows, mem in Bytes)')
    if(sketch == 'univmon'):
        ax.set_xlabel('Sketch Configuration (levels, rows, mem in Bytes)')

    ax.set_yscale("log", basey=10)

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

    plt.savefig(file_path, bbox_inches='tight')

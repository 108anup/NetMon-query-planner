from collections import namedtuple
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import pprint
from scipy.interpolate import interp1d
from operator import itemgetter

# TODO: get rid of things done by hand:
# by using derivatives:
# as done in: https://stackoverflow.com/questions/29382903/how-to-apply-piecewise-linear-fit-in-python

# ("rows", "columns", "pps")
# hash_bench = [(1, 1024, 27570015),
#               (2, 1024, 27554591),
#               (3, 1024, 27559059),
#               (4, 1024, 26760321),
#               (5, 1024, 26067310),
#               (6, 1024, 25428039),
#               (7, 1024, 24783708),
#               (8, 1024, 24134861),
#               (9, 1024, 23032677),
#               (10, 1024, 20954514),
#               (11, 1024, 19318893),
#               (12, 1024, 18358842)]

hash_bench = [(1, 4, 27564551),
              (2, 4, 27549165),
              (3, 4, 27547240),
              (4, 4, 26823812),
              (5, 4, 26022178),
              (6, 4, 25391335),
              (7, 4, 24645988),
              (8, 4, 24091511),
              (9, 4, 23519191),
              (10, 4, 23033710),
              (11, 4, 22488571),
              (12, 4, 22029111)]

# ("rows", "columns", "mem", "pps")
mem_bench_1 = [(10, 1, 0.0390625, 26470133),
               (10, 2, 0.078125, 23033864),
               (10, 4, 0.15625, 23034027),
               (10, 8, 0.3125, 23034072),
               (10, 16, 0.625, 21538252),
               (10, 32, 1.25, 21270772),
               (10, 64, 2.5, 21111141),
               (10, 128, 5, 21070769),
               (10, 256, 10, 21009486),
               (10, 512, 20, 21055890),
               (10, 1024, 40, 20946681),
               (10, 2048, 80, 21066369),
               (10, 8192, 320, 21028969),
               (10, 16384, 640, 20980031),
               (10, 32768, 1280, 20967735),
               (10, 131072, 5120, 14009640),
               (10, 262144, 10240, 10104009),
               (10, 524288, 20480, 8567701),
               (10, 1048576, 40960, 7614832),
               (10, 4194304, 163840, 6619393)]

mem_bench_2 = [(12, 1, 0.046875, 25923412),
               (12, 2, 0.09375, 22029132),
               (12, 4, 0.1875, 22029115),
               (12, 8, 0.375, 22029080),
               (12, 16, 0.75, 18601493),
               (12, 32, 1.5, 18318504),
               (12, 64, 3, 18411073),
               (12, 128, 6, 18409627),
               (12, 256, 12, 18394619),
               (12, 512, 24, 18420165),
               (12, 1024, 48, 18346669),
               (12, 2048, 96, 18399047),
               (12, 8192, 384, 18400698),
               (12, 16384, 768, 18377351),
               (12, 32768, 1536, 18344572),
               (12, 131072, 6144, 11719958),
               (12, 262144, 12288, 8633220),
               (12, 524288, 24576, 7249898),
               (12, 1048576, 49152, 6468332),
               (12, 2097152, 98304, 5910759),
               (12, 4194304, 196608, 5565745)]

# [(11, 1, 0.04296875, 25843741),
#  (11, 2, 0.0859375, 22488665),
#  (11, 4, 0.171875, 22488862),
#  (11, 8, 0.34375, 22488642),
#  (11, 16, 0.6875, 19586027),
#  (11, 32, 1.375, 19221237),
#  (11, 64, 2.75, 19365487),
#  (11, 128, 5.5, 19384318),
#  (11, 256, 11, 19343979),
#  (11, 512, 22, 19364502)]

# Assume cols are always power of 2
# Hence not using following
# ("rows", "columns", "pps")
# mem_bench_1 = [(7, 10240, 9139543),
#              (7, 1024, 24783884),
#              (7, 1048576, 10907133),
#              (7, 131072, 21325376),
#              (7, 16384, 24789334),
#              (7, 20480, 9125941),
#              (7, 2048, 24727968),
#              (7, 2097152, 9957436),
#              (7, 245760, 9074668),
#              (7, 262144, 15003552),
#              (7, 3072, 9162595),
#              (7, 32768, 24727878),
#              (7, 4194304, 9403385),
#              (7, 51200, 9106276),
#              (7, 524288, 12370217),
#              (7, 65536, 24701832),
#              (7, 8192, 24740262)]

# mem_bench_1 = [(8, 10240, 8353810),
#                (8, 1024, 24134593),
#                (8, 1048576, 9740216),
#                (8, 131072, 18918742),
#                (8, 16384, 24134640),
#                (8, 20480, 8340701),
#                (8, 2048, 24136108),
#                (8, 2097152, 8884773),
#                (8, 245760, 8289849),
#                (8, 262144, 13547931),
#                (8, 3072, 8377971),
#                (8, 32768, 24134800),
#                (8, 4194304, 8390003),
#                (8, 51200, 8319921),
#                (8, 524288, 11082471),
#                (8, 65536, 24124053),
#                (8, 8192, 24134950)]

"""
Forwarding params
"""

# Hand identified when fwding is bottleneck
bottleneck_idx = 0

fwd_thr = hash_bench[bottleneck_idx][2]
fwd_ns = 1e9/fwd_thr
print("fwd_ns: ", fwd_ns)

"""
# * Hashing params
"""

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
hashing_slope = np.average([((bench_list[i+1].ns - bench_list[i].ns) /
                             (bench_list[i+1].rows - bench_list[i].rows))
                            for i in range(start_idx, end_idx)])
print("hashing: x, y, slope:", hashing_x, hashing_y, hashing_slope)

# None linear model
# PWL by hand
xs = [1, 3, 9, 12]
ys = []
for r in xs:
    ys.append(bench_list[r-1].ns)
xs[0] = 0
print("hashing non linear: xs, ys: ")
pprint.pprint(xs)
pprint.pprint(ys)
hashing_time = interp1d(xs, ys)

"""
# * Mem params
"""
# There are some assumptions here about mem access time being
# similar for memory allocated close by in value
bench_list_1 = [SimpleNamespace(rows=x[0], cols=x[1], mem=x[2], pps=x[3])
                for x in mem_bench_1]
bench_list_2 = [SimpleNamespace(rows=x[0], cols=x[1], mem=x[2], pps=x[3])
                for x in mem_bench_2]

CELL_SIZE = 4
KB2B = 1024
for x in bench_list_1:
    x.ns = 1e9/x.pps
    # x.mem = x.rows * x.cols * CELL_SIZE / KB2B
for x in bench_list_2:
    x.ns = 1e9/x.pps
    # x.mem = x.rows * x.cols * CELL_SIZE / KB2B

bench_1_ns = list(map(lambda x: x.ns, bench_list_1))
bench_2_ns = list(map(lambda x: x.ns, bench_list_2))
FLAT_REGION = [6, 14]
p1 = bench_list_1[0]
p2 = bench_list_2[0]
flat_part_val = ((np.average(bench_2_ns[FLAT_REGION[0]:FLAT_REGION[1] + 1])
                  - np.average(bench_1_ns[FLAT_REGION[0]:FLAT_REGION[1] + 1]))
                 / (p2.rows - p1.rows))
print("Mem: flat part val: ", flat_part_val)
DOMINANT = 7
point = bench_list_2[7]
mem_const = point.ns - flat_part_val * point.rows
# mem_const = 16
# import ipdb; ipdb.set_trace()

# pprint.pprint(bench_list_1)
# pprint.pprint(bench_list_2)

# x_vals = [(bench_list_1[i].mem + bench_list_2[i].mem)/2
#           for i in range(len(bench_list_1))]
# y_vals = [bench_list_2[i].ns - bench_list_1[i].ns
#           for i in range(len(bench_list_2))]
x_vals = list(map(lambda x: x.mem, bench_list_1))
y_vals = list(map(lambda x: (x.ns-mem_const)/x.rows, bench_list_1))

# print(x_vals, y_vals)

# plt.plot(x_vals, y_vals)
# plt.show()

COMBINE = [1, 4, 10]
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
        y_avg = max(y_vals[idx:c])
        ys.extend([y_avg, y_avg])
        xs.append(x_vals[idx])
        xs.append(x_vals[idx + c - 1])
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
ground_truth = [(1, 1024, 4, 27557394),
                (1, 2048, 8, 27554700),
                (1, 8192, 32, 27552922),
                (1, 16384, 64, 27574561),
                (1, 32768, 128, 27559067),
                (1, 65536, 256, 27556674),
                (1, 131072, 512, 27557979),
                (1, 262144, 1024, 27552705),
                (1, 524288, 2048, 27557209),
                (1, 1048576, 4096, 27552689),
                (1, 2097152, 8192, 27558935),
                (1, 4194304, 16384, 27558144),
                (4, 1024, 16, 26760479),
                (4, 2048, 32, 26725093),
                (4, 8192, 128, 26724844),
                (4, 16384, 256, 26760352),
                (4, 32768, 512, 26725045),
                (4, 65536, 1024, 26823498),
                (4, 131072, 2048, 26754960),
                (4, 262144, 4096, 26725760),
                (4, 524288, 8192, 20778847),
                (4, 1048576, 16384, 17972142),
                (4, 2097152, 32768, 16427137),
                (4, 4194304, 65536, 15384414),
                (7, 1024, 28, 24783884),
                (7, 2048, 56, 24727968),
                (7, 8192, 224, 24740262),
                (7, 16384, 448, 24789334),
                (7, 32768, 896, 24727878),
                (7, 65536, 1792, 24701832),
                (7, 131072, 3584, 21325376),
                (7, 262144, 7168, 15003552),
                (7, 524288, 14336, 12370217),
                (7, 1048576, 28672, 10907133),
                (7, 2097152, 57344, 9957436),
                (7, 4194304, 114688, 9403385),
                (8, 1024, 32, 24134593),
                (8, 2048, 64, 24136108),
                (8, 8192, 256, 24134950),
                (8, 16384, 512, 24134640),
                (8, 32768, 1024, 24134800),
                (8, 65536, 2048, 24124053),
                (8, 131072, 4096, 18918742),
                (8, 262144, 8192, 13547931),
                (8, 524288, 16384, 11082471),
                (8, 1048576, 32768, 9740216),
                (8, 2097152, 65536, 8884773),
                (8, 4194304, 131072, 8390003),
                (9, 1024, 36, 23031460),
                (9, 2048, 72, 23164207),
                (9, 8192, 288, 23062167),
                (9, 16384, 576, 23052891),
                (9, 32768, 1152, 23046189),
                (9, 65536, 2304, 23519085),
                (9, 131072, 4608, 16978626),
                (9, 262144, 9216, 12060758),
                (9, 524288, 18432, 10097201),
                (9, 1048576, 36864, 8789591),
                (9, 2097152, 73728, 8027921),
                (9, 4194304, 147456, 7524943),
                (10, 1024, 40, 20946681),
                (10, 2048, 80, 21066369),
                (10, 8192, 320, 21028969),
                (10, 16384, 640, 20980031),
                (10, 32768, 1280, 20967735),
                (10, 65536, 2560, 23034069),
                (10, 131072, 5120, 14009640),
                (10, 262144, 10240, 10104009),
                (10, 524288, 20480, 8567701),
                (10, 1048576, 40960, 7614832),
                (10, 2097152, 81920, 7020178),
                (10, 4194304, 163840, 6619393),
                (11, 1024, 44, 19319091),
                (11, 2048, 88, 19365405),
                (11, 8192, 352, 19368229),
                (11, 16384, 704, 19350983),
                (11, 32768, 1408, 19313930),
                (11, 65536, 2816, 22465120),
                (11, 131072, 5632, 12768181),
                (11, 262144, 11264, 9340859),
                (11, 524288, 22528, 7872020),
                (11, 1048576, 45056, 6991860),
                (11, 2097152, 90112, 6408439),
                (12, 1024, 48, 18346669),
                (12, 2048, 96, 18399047),
                (12, 8192, 384, 18400698),
                (12, 16384, 768, 18377351),
                (12, 32768, 1536, 18344572),
                (12, 65536, 3072, 17134759),
                (12, 131072, 6144, 11719958),
                (12, 262144, 12288, 8633220),
                (12, 524288, 24576, 7249898),
                (12, 1048576, 49152, 6468332),
                (12, 2097152, 98304, 5910759),
                (12, 4194304, 196608, 5565745)]

ground_truth = [(1, 2, 0.0078125, 27546198),
                (1, 4, 0.015625, 27545867),
                (1, 8, 0.03125, 27565566),
                (1, 16, 0.0625, 27546962),
                (1, 32, 0.125, 27546887),
                (1, 64, 0.25, 27547568),
                (1, 128, 0.5, 27549005),
                (1, 256, 1, 27549183),
                (1, 512, 2, 27546034),
                (1, 1024, 4, 27557394),
                (1, 2048, 8, 27554700),
                (1, 8192, 32, 27552922),
                (1, 16384, 64, 27574561),
                (1, 32768, 128, 27559067),
                (1, 65536, 256, 27556674),
                (1, 131072, 512, 27557979),
                (1, 262144, 1024, 27552705),
                (1, 524288, 2048, 27557209),
                (1, 1048576, 4096, 27552689),
                (1, 2097152, 8192, 27558935),
                (1, 4194304, 16384, 27558144),

                (2, 2, 0.015625, 27545362),
                (2, 4, 0.03125, 27549123),
                (2, 8, 0.0625, 27547763),
                (2, 16, 0.125, 27551046),
                (2, 32, 0.25, 27551165),
                (2, 64, 0.5, 27547595),
                (2, 128, 1, 27549180),
                (2, 256, 2, 27548041),
                (2, 512, 4, 27547850),

                (3, 2, 0.0234375, 27547248),
                (3, 4, 0.046875, 27547242),
                (3, 8, 0.09375, 27546917),
                (3, 16, 0.1875, 27546899),
                (3, 32, 0.375, 27547029),
                (3, 64, 0.75, 27546950),
                (3, 128, 1.5, 27546898),
                (3, 256, 3, 27547133),
                (3, 512, 6, 27547565),

                (4, 2, 0.03125, 26823849),
                (4, 4, 0.0625, 26823804),
                (4, 8, 0.125, 26823728),
                (4, 16, 0.25, 26823621),
                (4, 32, 0.5, 26827048),
                (4, 64, 1, 26823323),
                (4, 128, 2, 26823386),
                (4, 256, 4, 26823264),
                (4, 512, 8, 26724870),
                (4, 1024, 16, 26760479),
                (4, 2048, 32, 26725093),
                (4, 8192, 128, 26724844),
                (4, 16384, 256, 26760352),
                (4, 32768, 512, 26725045),
                (4, 65536, 1024, 26823498),
                (4, 131072, 2048, 26754960),
                (4, 262144, 4096, 26725760),
                (4, 524288, 8192, 20778847),
                (4, 1048576, 16384, 17972142),
                (4, 2097152, 32768, 16427137),
                (4, 4194304, 65536, 15384414),

                (5, 2, 0.0390625, 26022029),
                (5, 4, 0.078125, 26022145),
                (5, 8, 0.15625, 26022176),
                (5, 16, 0.3125, 26022287),
                (5, 32, 0.625, 26021822),
                (5, 64, 1.25, 26021744),
                (5, 128, 2.5, 26021784),
                (5, 256, 5, 26076826),
                (5, 512, 10, 26034655),

                (6, 2, 0.046875, 25391283),
                (6, 4, 0.09375, 25391360),
                (6, 8, 0.1875, 25391452),
                (6, 16, 0.375, 25391333),
                (6, 32, 0.75, 25390667),
                (6, 64, 1.5, 25390846),
                (6, 128, 3, 25390600),
                (6, 256, 6, 25390632),
                (6, 512, 12, 25455291),

                (7, 2, 0.0546875, 24646109),
                (7, 4, 0.109375, 24646130),
                (7, 8, 0.21875, 24646186),
                (7, 16, 0.4375, 24646104),
                (7, 32, 0.875, 24643565),
                (7, 64, 1.75, 24644248),
                (7, 128, 3.5, 24644296),
                (7, 256, 7, 24702542),
                (7, 512, 14, 24739883),
                (7, 1024, 28, 24783884),
                (7, 2048, 56, 24727968),
                (7, 8192, 224, 24740262),
                (7, 16384, 448, 24789334),
                (7, 32768, 896, 24727878),
                (7, 65536, 1792, 24701832),
                (7, 131072, 3584, 21325376),
                (7, 262144, 7168, 15003552),
                (7, 524288, 14336, 12370217),
                (7, 1048576, 28672, 10907133),
                (7, 2097152, 57344, 9957436),
                (7, 4194304, 114688, 9403385),

                (8, 2, 0.0625, 24092605),
                (8, 4, 0.125, 24092766),
                (8, 8, 0.25, 24092792),
                (8, 16, 0.5, 24084787),
                (8, 32, 1, 24082188),
                (8, 64, 2, 24080674),
                (8, 128, 4, 24079929),
                (8, 256, 8, 24132845),
                (8, 512, 16, 24134756),
                (8, 1024, 32, 24134593),
                (8, 2048, 64, 24136108),
                (8, 8192, 256, 24134950),
                (8, 16384, 512, 24134640),
                (8, 32768, 1024, 24134800),
                (8, 65536, 2048, 24124053),
                (8, 131072, 4096, 18918742),
                (8, 262144, 8192, 13547931),
                (8, 524288, 16384, 11082471),
                (8, 1048576, 32768, 9740216),
                (8, 2097152, 65536, 8884773),
                (8, 4194304, 131072, 8390003),

                (9, 2, 0.0703125, 23515905),
                (9, 4, 0.140625, 23516002),
                (9, 8, 0.28125, 23515974),
                (9, 16, 0.5625, 23426083),
                (9, 32, 1.125, 23367875),
                (9, 64, 2.25, 23240328),
                (9, 128, 4.5, 23193860),
                (9, 256, 9, 23090537),
                (9, 512, 18, 23041067),
                (9, 1024, 36, 23031460),
                (9, 2048, 72, 23164207),
                (9, 8192, 288, 23062167),
                (9, 16384, 576, 23052891),
                (9, 32768, 1152, 23046189),
                (9, 65536, 2304, 23519085),
                (9, 131072, 4608, 16978626),
                (9, 262144, 9216, 12060758),
                (9, 524288, 18432, 10097201),
                (9, 1048576, 36864, 8789591),
                (9, 2097152, 73728, 8027921),
                (9, 4194304, 147456, 7524943),

                (10, 2, 0.078125, 23033864),
                (10, 4, 0.15625, 23034027),
                (10, 8, 0.3125, 23034072),
                (10, 16, 0.625, 21538252),
                (10, 32, 1.25, 21270772),
                (10, 64, 2.5, 21111141),
                (10, 128, 5, 21070769),
                (10, 256, 10, 21009486),
                (10, 512, 20, 21055890),
                (10, 1024, 40, 20946681),
                (10, 2048, 80, 21066369),
                (10, 8192, 320, 21028969),
                (10, 16384, 640, 20980031),
                (10, 32768, 1280, 20967735),
                (10, 65536, 2560, 23034069),
                (10, 131072, 5120, 14009640),
                (10, 262144, 10240, 10104009),
                (10, 524288, 20480, 8567701),
                (10, 1048576, 40960, 7614832),
                (10, 2097152, 81920, 7020178),
                (10, 4194304, 163840, 6619393),

                (11, 2, 0.0859375, 22488665),
                (11, 4, 0.171875, 22488862),
                (11, 8, 0.34375, 22488642),
                (11, 16, 0.6875, 19586027),
                (11, 32, 1.375, 19221237),
                (11, 64, 2.75, 19365487),
                (11, 128, 5.5, 19384318),
                (11, 256, 11, 19343979),
                (11, 512, 22, 19364502),
                (11, 1024, 44, 19319091),
                (11, 2048, 88, 19365405),
                (11, 8192, 352, 19368229),
                (11, 16384, 704, 19350983),
                (11, 32768, 1408, 19313930),
                (11, 65536, 2816, 22465120),
                (11, 131072, 5632, 12768181),
                (11, 262144, 11264, 9340859),
                (11, 524288, 22528, 7872020),
                (11, 1048576, 45056, 6991860),
                (11, 2097152, 90112, 6408439),

                (12, 2, 0.09375, 22029132),
                (12, 4, 0.1875, 22029115),
                (12, 8, 0.375, 22029080),
                (12, 16, 0.75, 18601493),
                (12, 32, 1.5, 18318504),
                (12, 64, 3, 18411073),
                (12, 128, 6, 18409627),
                (12, 256, 12, 18394619),
                (12, 512, 24, 18420165),
                (12, 1024, 48, 18346669),
                (12, 2048, 96, 18399047),
                (12, 8192, 384, 18400698),
                (12, 16384, 768, 18377351),
                (12, 32768, 1536, 18344572),
                (12, 65536, 3072, 17134759),
                (12, 131072, 6144, 11719958),
                (12, 262144, 12288, 8633220),
                (12, 524288, 24576, 7249898),
                (12, 1048576, 49152, 6468332),
                (12, 2097152, 98304, 5910759),
                (12, 4194304, 196608, 5565745)]

bench_list = [SimpleNamespace(rows=x[0], cols=x[1], mem=x[2], pps=x[3])
              for x in ground_truth]
diff = []
for x in bench_list:
    x.ns = 1e9/x.pps
    x.hash_ns = hashing_y + hashing_slope * (x.rows - hashing_x)
    x.mem_ns = mem_const + x.rows * get_mem_access_time(x.mem)
    items = (fwd_ns,
             # hashing_time(x.rows),
             x.hash_ns, x.mem_ns)
    x.argmax, x.model_ns = max(enumerate(items), key=itemgetter(1))
    x.argmax = x.argmax * 10
    diff.append(abs(x.ns - x.model_ns) / x.ns)

labels = list(map(lambda x: str(x.rows) + ", " + str(x.mem), bench_list))
fig = plt.figure(figsize=(4, 4))
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
plt.xticks(labels[::10])
plt.xticks(rotation=90)
plt.xlabel("Sketch configuration (rows, mem KB)")
plt.ylabel("Time per packet (ns)")
plt.legend()
# pprint.pprint(bench_list)
# plt.show()
fig.tight_layout()
plt.savefig('netro-model.pdf')
print("Relative Error: ", np.average(diff))

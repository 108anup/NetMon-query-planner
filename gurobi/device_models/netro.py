from collections import namedtuple
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import pprint
from scipy.interpolate import interp1d

# ("rows", "columns", "pps")
hash_bench = [(1, 1024, 27570015),
              (2, 1024, 27554591),
              (3, 1024, 27559059),
              (4, 1024, 26760321),
              (5, 1024, 26067310),
              (6, 1024, 25428039),
              (7, 1024, 24783708),
              (8, 1024, 24134861),
              (9, 1024, 23032677),
              (10, 1024, 20954514),
              (11, 1024, 19318893),
              (12, 1024, 18358842)]

# ("rows", "columns", "mem", "pps")
mem_bench_1 = [(7, 1024, 28, 24783884),
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
               (7, 4194304, 114688, 9403385)]

mem_bench_2 = [(8, 1024, 32, 24134593),
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
               (8, 4194304, 131072, 8390003)]

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

# mem_bench_2 = [(8, 10240, 8353810),
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
Hashing params
"""

bench_list = [SimpleNamespace(rows=x[0], cols=x[1], pps=x[2])
              for x in hash_bench]
for x in bench_list:
    x.ns = 1e9/x.pps

# Hand identified when hashing is bottleneck
start = 4
end = 8
start_idx = start - 1
end_idx = end - 1

hashing_x = start
hashing_y = bench_list[start_idx].ns
hashing_slope = np.average([((bench_list[i+1].ns - bench_list[i].ns) /
                           (bench_list[i+1].rows - bench_list[i].rows))
                            for i in range(start_idx, end_idx)])
print("hashing: x, y, slope:", hashing_x, hashing_y, hashing_slope)

"""
Mem params
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

pprint.pprint(bench_list_1)
pprint.pprint(bench_list_2)

x_vals = [(bench_list_1[i].mem + bench_list_2[i].mem)/2
          for i in range(len(bench_list_1))]
y_vals = [bench_list_2[i].ns - bench_list_1[i].ns
          for i in range(len(bench_list_1))]
print(x_vals, y_vals)

# plt.plot(x_vals, y_vals)
# plt.show()

COMBINE = 6
print(x_vals[:COMBINE])
print(y_vals[:COMBINE])
y_start = np.average(y_vals[:COMBINE])
ys = [y_start, y_start]
ys.extend(y_vals[COMBINE:])
xs = [0, x_vals[COMBINE-1]]
xs.extend(x_vals[COMBINE:])
xs.append(200000)
ys.append(ys[-1])
# plt.plot(xs, ys)
# plt.show()

get_mem_access_time = interp1d(xs, ys)
# By hand
dominant = 6
point = bench_list_1[dominant]
mem_const = point.ns - point.rows * get_mem_access_time(point.mem)

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
    x.model_ns = max(fwd_ns,
                     hashing_y + hashing_slope * (x.rows - hashing_x),
                     mem_const + x.rows * get_mem_access_time(x.mem)
    )
    diff.append(abs(x.ns - x.model_ns) / x.ns)

labels = list(map(lambda x: str(x.rows) + ", " + str(x.mem), bench_list))
fig = plt.figure(figsize=(10, 7))
plt.rcParams.update({'font.size': 20,
                     'axes.linewidth': 2,
                     'xtick.major.size': 10,
                     'xtick.major.width': 2,
                     'xtick.minor.size': 5,
                     'xtick.minor.width': 1,
                     'ytick.major.size': 10,
                     'ytick.major.width': 2,
                     'ytick.minor.size': 5,
                     'ytick.minor.width': 1})
plt.plot(labels, list(map(lambda x: x.ns, bench_list)), linewidth=4)
plt.plot(labels, list(map(lambda x: x.model_ns, bench_list)), linewidth=4)
plt.xticks(labels[::6])
plt.xticks(rotation=90)
plt.xlabel("Sketch configuration (rows, mem KB)")
plt.ylabel("Time per packet (ns)")
# plt.show()
fig.tight_layout()
plt.savefig('netro-model.pdf')
print(np.average(diff))

pprint.pprint(bench_list)

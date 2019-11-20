import pprint

def ap(first, last, step=1):
    return [first + x*step for x in range(int((last - first)/step + 1))]

# Optimize for throughput

"""
Notation:
M_i: Memory in KB at i
r_i: number of rows in sketch
f: fraction of flows to monitor at CPU
"""

# Decision variables
ranges = {
    'f': ap(0.01, 0.99, 0.01),
    'M_p4': [2048, 4096, 8192, 16384, 131072],
    'r_p4': ap(4, 9),
    'M_cpu': [10, 32, 80, 160, 240, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
    'r_cpu': ap(4, 9),
    'cpu_cores': ap(2, 8),
    'dpdk_cores': ap(1, 4),

}

keys = list(ranges.keys())
print(keys)

# CPU Model
# For dpdk
fraction_parallel = 3/4 # Amdahls law
single_core_thr = 35

# in KB
L1_size = 32
L2_size = 256
L3_size = 8192

L1_ns = 0.6
L2_ns = 1.5
L3_ns = 3.7
L4_ns = 31

static_loads = [18, 24, 30, 43, 49, 55]

# P4 Model
# in Mpps (if sketch fits then line rate)
throughput_p4 = 148

# Constraints
epsilon_0 = 0.0001

max_thr = -1
solutions = []
def solve(selection, key):
    if key < len(keys):
        for i in ranges[keys[key]]:
            selection[keys[key]] = i
            solve(selection, key + 1)
    else:
        s = selection
        k_cpu = s['M_cpu'] * 1024 / (4 * s['r_cpu'])
        k_p4 = s['M_p4'] * 1024 / (4 * s['r_p4'])
        epsilon_p4 = 2 / k_p4
        epsilon_cpu = 2 / k_cpu
        f = s['f']

        # Check accuracy constraints
        # Note this does nto require to be considered with cores
        # check separately and then use only corresponding selections
        if(epsilon_p4 > epsilon_0 / (1-f) or epsilon_cpu > epsilon_0 / f):
            continue
        # r starts at 4 => assume delta constraint is metadata

        # Memory constrant
        # Change ranges to show memory constraints

        # Constraint on number of core
        # Change ranges to show core constraint
        sketch_cores = s['cpu_cores'] - s['dpdk_cores']
        if(sketch_cores < 1 ):
            continue

        # Constraint to check if check will fit
        # Change ranges to only have num cols which can be supported and
        # num rows which can be supported

        # Objective
        # Want to be within 10% Max throughput, for the optimal solutions choose one
        # with minimum resource utilization on sketch.
        # Point is we don't need to see how much resources are utilized on the sketch,
        # we can just reduce r_p4 or k_p4 and automatically the resources will be minimised.

        r = s['r_cpu']
        M = s['M_cpu']

        # Uniformly random probability model
        t = static_loads[r] - r
        r1 = r * (min(L1_size, M)) / M
        r2 = r * max(0, (min(L2_size - L1_size, M - L1_size), M) / M)
        r3 = r * max(0, (min(L3_size - L2_size, M - L2_size), M) / M)
        r2 = r * max(0, (min(L4_size - L3_size, M - L3_size), M) / M)

        ns_per_packet = (t + r1) * L1_ns + r2 * L2_ns + r3 * L3_ns + r4 * L4_ns;
        ns_per_packet = ns_per_packet / s['cpu_cores']
        # Sketch Model: Fully parallel sketch updates on threads
        throughput_sketch = sketch_cores * 1000 / (ns_per_packet)
        throughput_dpdk = single_core_thr / (1-fraction_parallel + \
                                             fraction_parallel/s['cpu_cores'])
        throughput = min(throughput_dpdk, throughput_sketch, throuhput_p4)
        if(throughput > max_thr):
            max_thr = throughput
            solution = [selection]
        if((max_thr - throughput) / max_thr < 0.1):
            solution.append(selection)

solve({}, 0)
pprint.pprint(solution)

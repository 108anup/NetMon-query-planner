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
    'cpu_cores' ap(1, 4),

}

keys = list(ranges.keys())
print(keys)

# CPU Model
# For dpdk
fraction_parallel = 3/4 # Amdahls law

# in KB
L1_size = 32
L2_size = 256
L3_size = 8192
L4_size = 

# Constraints
epsilon_0 = 0.0001

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

        # Constraint to check if check will fit
        # Change ranges to only have num cols which can be supported and
        # num rows which can be supported

        # Objective
        # Max throughput, for the optimal solutions choose one
        # with minimum resource utilization on sketch.
        # Point is we don't need to see how much rseources are utilized on the sketch,
        # we can just reduce r_p4 or k_p4 and automatically the resources will be minimised.

        r = s['r_cpu']
        M = s['M_cpu']

        r1 = r * (min(L1_size, M)) / M
        r2 = r * max(0, (min(L2_size - L1_size, M - L1_size), M) / M)
        r3 = r * max(0, (min(L3_size - L2_size, M - L2_size), M) / M)
        r2 = r * max(0, (min(L4_size - L3_size, M - L3_size), M) / M)

        ns_per_packet = (t + r1) * L1_ns + r2 * L2_ns + r3 * L3_ns + r4 * L4_ns;





solve({}, 0)


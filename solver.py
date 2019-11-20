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
M_cpu_range = [10, 32, 80, 160, 240, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
r_cpu_range = ap(4, 9)
ranges = {
    'f': ap(0.01, 0.99, 0.05),
    'M_p4': [2048, 4096, 8192, 16384, 65536, 131072],
    'r_p4': ap(4, 9),
    'M_cpu': M_cpu_range,
    'r_cpu': r_cpu_range,
    'cpu_cores': ap(2, 8),
    'dpdk_cores': ap(1, 4),

}

keys = list(ranges.keys())
print(keys)

# CPU Model
# For dpdk
fraction_parallel = 3/4 # Amdahls law
single_core_thr = 35

# Uniformly random probability model
# in KB
L1_size = 32
L2_size = 256
L3_size = 8192

L1_ns = 0.6
L2_ns = 1.5
L3_ns = 3.7
L4_ns = 31
hash_ns = 3.5

static_loads = [18, 24, 30, 43, 49, 55]

def get_ns_per_packet(r, M, sketch_cores):
    t = static_loads[r-4] - r
    r1 = r * (min(L1_size, M)) / M
    r2 = r * (max(0, min(L2_size - L1_size, M - L1_size)) / M)
    r3 = r * (max(0, min(L3_size - L2_size, M - L2_size)) / M)
    r4 = r * (max(0, M - L3_size) / M)

    ns_per_packet = hash_ns * r + (t + r1) * L1_ns + r2 * L2_ns + r3 * L3_ns + r4 * L4_ns;
    ns_per_packet = ns_per_packet / sketch_cores
    return ns_per_packet

# P4 Model
# in Mpps (if sketch fits then line rate)
throughput_p4 = 148

# Constraints
epsilon_0 = 0.0001

# Optimum leeway (relative)
thr_leeway = 0.05
resource_leeway = 0.05

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
            return
        # r starts at 4 => assume delta constraint is metadata

        # Memory constrant
        # Change ranges to show memory constraints

        # Constraint on number of core
        # Change ranges to show core constraint
        sketch_cores = s['cpu_cores'] - s['dpdk_cores']
        if(sketch_cores < 1 ):
            return

        # Constraint to check if check will fit
        # Change ranges to only have num cols which can be supported and
        # num rows which can be supported

        # Objective
        # Want to be within 10% Max throughput, for the optimal solutions choose one
        # with minimum resource utilization on sketch.
        # Point is we don't need to see how much resources are utilized on the sketch,
        # we can just reduce r_p4 or k_p4 and automatically the resources will be minimised.

        ns_per_packet = get_ns_per_packet(s['r_cpu'], s['M_cpu'], sketch_cores)

        # Sketch Model: Fully parallel sketch updates on threads
        throughput_sketch = sketch_cores * 1000 / (ns_per_packet)
        throughput_dpdk = single_core_thr / (1-fraction_parallel + \
                                             fraction_parallel/s['dpdk_cores'])
        throughput = min(throughput_dpdk, throughput_sketch, throughput_p4)

        global max_thr
        global solution

        sol = selection.copy()
        sol['ns_per_packet_single_core'] = ns_per_packet * sketch_cores
        sol['ns_per_packet'] = ns_per_packet
        sol['throughput'] = throughput
        sol['sketch_cores'] = sketch_cores

        if(throughput > max_thr):
            max_thr = throughput
            solution = [sol]
        else:
            # Fix this: prior solutions could also be within 10% of new max
            if((max_thr - throughput) / max_thr < thr_leeway):
                solution.append(sol)
        return

'''
solve({}, 0)
#pprint.pprint(solution)
print(max_thr, len(solution))

min_obj = 1e9
reduced_solution = []
for s in solution:
    obj = s['cpu_cores']
    obj += s['M_p4']/32768 # num stages due to cols
    obj += s['r_p4']/4 # num stages due to rows

    s['obj'] = obj
    if (obj < min_obj):
        reduced_solution = [s]
        min_obj = obj
    else:
        if(obj - min_obj)/min_obj < resource_leeway:
            reduced_solution.append(s)

print(min_obj, len(reduced_solution))
pprint.pprint(reduced_solution)
'''

# Test CPU Model

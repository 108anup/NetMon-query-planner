cell_size = 4


# Sketch model
# Sketches enforce accuracy requirements,
# user can instantiate to provide more leeway
class sketch:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, attr):
        return self.__dict__[attr]


class cm_sketch(sketch):
    # def epsilon(self, r, c, h):
    #     pass

    # def delta(self, r, c, h):
    #     pass

    # def rows(self, epsilon, delta):
    #     pass

    def cols(self, fraction):
        epsilon_max = self.eps0 / fraction
        c = 2 / epsilon_max
        return c


# Device model
# Devices must ensure capacity resource constraint,
# user can modify instantiation to further restrict resources
class device:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, attr):
        return self.__dict__[attr]

    def res_thr(self, args):
        raise NotImplementedError


class cpu(device):
    # TODO: get automatically
    static_loads = [18, 24, 30, 43, 49, 55]
    # TODO:: update with OVS
    fraction_parallel = 3/4

    def single_thread_ns(self, sketches):
        r = 0
        M = 0

        for (rows, cols, hashes) in sketches:
            r += rows
            M += cols * rows * cell_size / 1024  # KB

        # Uniformly random probability model
        # TODO:: Assumption DRAM is unbounded
        # TODO:: verify correctness / accuracy for when multiple sketches are
        # there with different cols, does this still hold as now each access
        # has different amount of possible memory to be found in
        t = cpu.static_loads[r-4] - r
        r1 = r * (min(self.L1_size, M)) / M
        r2 = r * (max(0, min(self.L2_size - self.L1_size,
                             M - self.L1_size)) / M)
        r3 = r * (max(0, min(self.L3_size - self.L2_size,
                             M - self.L2_size)) / M)
        r4 = r * (max(0, M - self.L3_size) / M)
        ns_per_packet = (self.hash_ns * r + (t + r1) * self.L1_ns
                         + r2 * self.L2_ns + r3 * self.L3_ns + r4 * self.L4_ns)
        return (ns_per_packet, M)

    def dpdk_throughput(self, c):
        return (self.dpdk_single_core_thr / (1-cpu.fraction_parallel
                                             + cpu.fraction_parallel/c))

    # Remark: Can add stricter resource constraints depending on need
    def res_thr(self, sketches):
        (ns_per_packet, M) = self.single_thread_ns(sketches)
        allocations = []
        for c in self.cores:
            for dpdk_cores in range(1, c):
                sketch_cores = c - dpdk_cores
                sketch_throughput = ns_per_packet / sketch_cores
                dpdk_thr = self.dpdk_throughput(dpdk_cores)
                alloc_dict = {
                    'cost': 10*c + M/8192,
                    'throughput': min(dpdk_thr,
                                      sketch_throughput),
                    'debug': {
                        'sketch_cores': sketch_cores,
                        'dpdk_cores': dpdk_cores,
                        'sketch_throughput': sketch_throughput,
                        'dpdk_throughput': dpdk_thr
                    }
                }
                allocations.append(alloc_dict)
        return allocations


class p4(device):

    def res_thr(self, sketches):
        r = 0
        M = 0

        for (rows, cols, hashes) in sketches:
            r += rows
            M += cols * rows * cell_size / 1024  # KB
            M_stage = cols * cell_size / 1024
            if(M_stage > self.sram):
                return None

        if(r > self.stages):
            return None

        allocations = [{
            'cost': r/self.meter_alus + M/self.sram,
            'throughput': self.line_thr
        }]
        return allocations


# Query and placement abstraction
eps0 = 1e-5
del0 = 0.96
queries = [cm_sketch(eps0=eps0, del0=del0, sketch_id=1), cm_sketch(eps0=eps0, del0=del0, sketch_id=2)]

# All memory measured in KB unless otherwise specified
# TODO:: update with OVS
devices = [
    cpu(L1_size=32, L2_size=256, L3_size=7775,
        L1_ns=0.6, L2_ns=1.5, L3_ns=3.7, L4_ns=36,
        hash_ns=3.5, cores=8, dpdk_single_core_thr=35),
    p4(meter_alus=4, sram=48, stages=12)
]
num_devices = len(devices)
mappings = []
subsketches = []
num_subsketches = 0

# Sketch partition and one sketch placement generation
def gen_mappings(idx=0, cur_map=list(range(num_devices)), remaining=10):
    if(idx == num_devices):
        mappings.append(cur_map.copy())
    else:
        for val in range(remaining+1):
            cur_map[idx] = val/10
            gen_mappings(idx+1, cur_map, remaining-val)


max_thr = -1
set_thr_solutions = list()
thr_leeway = 0.05

# Full sketch placement
def gen_placements(subsketch_num=0, placements=[]):
    if(subsketch_num < num_subsketches):
        for i in range(len(mappings)):
            placements.append((subsketch[subsketch_num], mappings[i]))
            gen_placements(subsketch_num+1, placements)
        else:
            device_mappings = {}  #list(range(num_devices))
            for placement in placements:
                for dev_num in len(num_devices):
                    dev_fraction = placement[1][dev_num]
                    if(dev_fraction > 0):
                        num_cols = placement[0].cols(dev_fraction)
                        if dev_num in device_mappings:
                            device_mappings[dev_num].append((1, num_cols, 1))
                        else:
                            device_mappings[dev_num] = [(1, num_cols, 1)]

            res_thr_list = []
            for dev_num, sketches in device_mappings.items():
                res_thr = devices[dev_num].res_thr(sketches)
                if(res_thr is not None):
                    res_thr_list.append(res_thr)
                else:
                    # This placement does not satisfy capacity constraints
                    return
            solution = {'placements': placements.copy(),
                        'device_mappings': device_mappings,
                        'res_thr_list': res_thr_list)
            thr_overall = 1e9  # Mpps
            total_cost = 0
            for res_thr in res_thr_list:
                thr_overall = min(thr_overall, res_thr['throughput'])
                total_cost += res_thr['cost']

            solution['thr_overall'] = thr_overall
            solution['total_cost'] = total_cost
            if(thr_overall > max_thr):
                max_thr = thr_overall
                set_thr_solutions = [solution]
            elif((max_thr - thr_overall)/max_thr < thr_leeway):
                set_thr_solutions.append(solution)


gen_mappings()
for sk in queries:
    # Greedy setting of number of rows
    partitions = sk.cols()
    subsketches += [sk for i in range(partitions)]
num_subsketches = len(subsketches)
gen_placements()

import pprint
import math
import itertools
import threading
import ipdb
cell_size = 4


def ap(first, last, step=1):
    return [first + x*step for x in range(int((last - first)/step + 1))]


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

    def rows(self):
        return int(math.log(1/self.del0))

    def cols(self, fraction):
        epsilon_max = self.eps0 / fraction
        c = math.e / epsilon_max
        return int(c)


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

        for sk in sketches:
            rows = sk['rows']
            cols = sk['cols']
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
        return (ns_per_packet, M,
                {'r': r, 'M': M, 'ns_per_packet': ns_per_packet})

    def dpdk_throughput(self, c):
        return (self.dpdk_single_core_thr / (1-cpu.fraction_parallel
                                             + cpu.fraction_parallel/c))

    # Remark: Can add stricter resource constraints depending on need
    def res_thr(self, sketches):
        (ns_per_packet, M, debug) = self.single_thread_ns(sketches)
        allocations = []
        for c in ap(2, self.cores):
            for dpdk_cores in range(1, c):
                sketch_cores = c - dpdk_cores
                sketch_throughput = sketch_cores * 1000 / ns_per_packet
                dpdk_thr = self.dpdk_throughput(dpdk_cores)
                alloc_dict = {
                    'cost': 10*c + M/8192,
                    'throughput': min(dpdk_thr,
                                      sketch_throughput),
                    'debug': {
                        'sketch_cores': sketch_cores,
                        'dpdk_cores': dpdk_cores,
                        'sketch_throughput': sketch_throughput,
                        'dpdk_throughput': dpdk_thr,
                        **debug
                    }
                }
                allocations.append(alloc_dict)
        return allocations


class p4(device):

    def res_thr(self, sketches):
        #ipdb.set_trace()
        r = 0
        M = 0

        for sk in sketches:
            rows = sk['rows']
            cols = sk['cols']
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
eps0 = 1e-4
del0 = 0.01
queries = [cm_sketch(eps0=eps0, del0=del0, sketch_id=1),
           cm_sketch(eps0=eps0, del0=del0, sketch_id=2)]

# All memory measured in KB unless otherwise specified
# TODO:: update with OVS
devices = [
    cpu(L1_size=32, L2_size=256, L3_size=7775,
        L1_ns=0.6, L2_ns=1.5, L3_ns=3.7, L4_ns=36,
        hash_ns=3.5, cores=8, dpdk_single_core_thr=35),
    p4(meter_alus=4, sram=48, stages=12, line_thr=148)
]
num_devices = len(devices)
mappings = []
subsketches = []
num_subsketches = 0


# Sketch partition and one sketch placement generation
def gen_mappings(idx=0, cur_map=list(range(num_devices)), sum_remaining=10):
    if(idx == num_devices):
        mappings.append(cur_map.copy())
    elif (idx == num_devices - 1):
        cur_map[idx] = sum_remaining/10
        gen_mappings(idx+1, cur_map, 0)
    else:
        for val in range(sum_remaining+1):
            cur_map[idx] = val/10
            gen_mappings(idx+1, cur_map, sum_remaining-val)


thr_leeway = 0.05


# Full sketch placement
class Worker(threading.Thread):

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None):
        super(Worker, self).__init__(group=group, target=target,
                                     name=name)
        self.args = args
        self.kwargs = kwargs
        return

    def run(self):
        print(self.args)
        first_map = self.args[0]
        self.max_thr = -1
        self.set_thr_solutions = []
        subsketch_num = 0
        placements = {}
        sk = subsketches[subsketch_num][1]
        placements[subsketches[subsketch_num]] = sk.mappings[first_map]
        self.gen_placements(subsketch_num+1, placements)

    def gen_placements(self, subsketch_num=0, placements={}):
        if(subsketch_num < num_subsketches):
            sk = subsketches[subsketch_num][1]
            for i in range(len(sk.mappings)):
                placements[subsketches[subsketch_num]] = sk.mappings[i]
                self.gen_placements(subsketch_num+1, placements)
        else:
            device_mappings = {}  # list(range(num_devices))
            for subsk, mapping in placements.items():
                for dev_num in range(num_devices):
                    dev_fraction = mapping[dev_num]
                    if(dev_fraction > 0):
                        num_cols = subsk[1].cols(dev_fraction)
                        # print(dev_fraction, num_cols)
                        subsk_el = {
                            'rows': 1,
                            'cols': num_cols,
                            'hashes': 1,
                            'subsk': subsk,
                            'dev_fraction': dev_fraction
                        }
                        device_mappings.setdefault(dev_num,
                                                   []).append(subsk_el)

            res_thr_choices = {}
            for dev_num, sketches in device_mappings.items():
                res_thr = devices[dev_num].res_thr(sketches)
                if(res_thr is not None):
                    res_thr_choices[dev_num] = res_thr
                else:
                    # This placement does not satisfy capacity constraints
                    return

            # TODO:: Can choose best placement for
            # each device and then take product
            res_thr_choices_list_of_lists = list(res_thr_choices.values())
            res_thr_product = list(itertools.product(
                *res_thr_choices_list_of_lists))

            # ipdb.set_trace()
            # placement format (row_num, sk_ptr) => mapping ptr
            solution_outline = {'placements': placements.copy(),
                                'device_mappings': device_mappings}
            for res_thr_instance in res_thr_product:
                thr_overall = 1e9  # Mpps
                total_cost = 0
                for dev_res_thr in res_thr_instance:
                    thr_overall = min(thr_overall, dev_res_thr['throughput'])
                    total_cost += dev_res_thr['cost']
                # ipdb.set_trace()
                if(thr_overall > self.max_thr):
                    max_thr = thr_overall
                    solution = solution_outline.copy()
                    solution['res_thr'] = res_thr_instance
                    solution['thr_overall'] = thr_overall
                    solution['total_cost'] = total_cost
                    self.set_thr_solutions = [solution]
                elif((max_thr - thr_overall)/max_thr < thr_leeway):
                    solution = solution_outline.copy()
                    solution['res_thr'] = res_thr_instance
                    solution['thr_overall'] = thr_overall
                    solution['total_cost'] = total_cost
                    self.set_thr_solutions.append(solution)


gen_mappings()
for sk in queries:
    # Greedy setting of number of rows
    partitions = sk.rows()
    subsketches += [(i+1, sk) for i in range(partitions)]
    sk.mappings = []
    # only consider mappings which at least fit on the device
    # when only one sketch is present.
    for mapping in mappings:
        all_mapped = True
        for dev_num in range(num_devices):
            if(mapping[dev_num] > 0):
                num_cols = sk.cols(mapping[dev_num])
                if(devices[dev_num].res_thr([
                        {'rows': 1, 'cols': num_cols, 'hashes': 1}
                        ]) is None):
                    all_mapped = False
                    break
        if(all_mapped):
            sk.mappings.append(mapping)
    print(sk.mappings)

num_subsketches = len(subsketches)
print("num_subsketches: ", num_subsketches)

workers = []
for i in range(len(subsketches[0][1].mappings)):
    w_i = Worker(args=(i, ))
    workers.append(w_i)
    w_i.start()

for w in workers:
    w.join()
    print(len(w.set_thr_solutions))

# print(len(set_thr_solutions))
# pprint.pprint(set_thr_solutions)
ipdb.set_trace()

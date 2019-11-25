cell_size = 4


# Sketch model
class sketch:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, attr):
        return self.__dict__[attr]


class cm_sketch(sketch):
    def epsilon(self, r, c, h):
        pass

    def delta(self, r, c, h):
        pass

    def rows(self, epsilon, delta):
        pass

    def cols():
        pass


# Device model
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
        return ns_per_packet

    def dpdk_throughput(self, c):
        return (self.dpdk_single_core_thr / (1-cpu.fraction_parallel
                                             + cpu.fraction_parallel/c))

    def res_thr(self, sketches):
        ns_per_packet = self.single_thread_ns(sketches)
        allocations = []
        for c in self.cores:
            for dpdk_cores in range(1, c):
                sketch_cores = c - dpdk_cores
                sketch_throughput = ns_per_packet / sketch_cores
                dpdk_thr = self.dpdk_throughput(dpdk_cores)
                alloc_dict = {
                    'cost': c,
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
    pass


# Query and placement abstraction
eps0 = 1e-5
del0 = 0.96
query = [(cm_sketch, eps0, del0), (cm_sketch, eps0, del0)]

devices = [
    cpu(L1_size=32, L2_size=256, L3_size=7775,
        L1_ns=0.6, L2_ns=1.5, L3_ns=3.7, L4_ns=36,
        hash_ns=3.5, cores=8, dpdk_single_core_thr=35)
    # TODO:: update with OVS
]

# Query partition

# Query placement

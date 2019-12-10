import gurobipy as gp
from gurobipy import GRB
import math
import time
import ipdb

def memoize(func):
    cache = dict()

    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return memoized_func


class param:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, attr):
        return self.__dict__[attr]


class cm_sketch(param):

    def __repr__(self):
        return 'sk_{}'.format(self.sketch_id)
        return "cm: eps0: {}, del0: {}".format(self.eps0, self.del0)

    def rows(self):
        return math.ceil(math.log(1/self.del0))

    def cols(self):
        return math.ceil(math.e / self.eps0)

    @memoize
    def min_mem(self):
        return cell_size * self.cols() / KB2B


class cpu(param):
    # TODO:: update with OVS
    fraction_parallel = 3/4

    def get_pdt_var(self, a, b, pdt_name, m):
        m.update()
        # print(a)
        # print(a.varName)
        # ipdb.set_trace()
        pdt = m.addVar(vtype=GRB.CONTINUOUS,
                       name='pdt_{}_{}'.format(pdt_name, self))
        loga = m.addVar(vtype=GRB.CONTINUOUS,
                        name='log_{}'.format(a.varName))
        logb = m.addVar(vtype=GRB.CONTINUOUS,
                        name='log_{}'.format(b.varName))
        logpdt = m.addVar(vtype=GRB.CONTINUOUS,
                          name='log_pdt_{}_{}'.format(pdt_name, self))
        # m.addGenConstrExpA(logpdt, pdt, 2,
        #                    name='exp_pdt_{}_{}'.format(pdt_name, self))
        m.addGenConstrLogA(pdt, logpdt, 2,
                           name='log_pdt_{}_{}'.format(pdt_name, self),
                           options="FuncPieces=-1 FuncPieceError=0.001")
        m.addGenConstrLogA(a, loga, 2,
                           name='log_{}'.format(a.varName),
                           options="FuncPieces=-1 FuncPieceError=0.001")
        m.addGenConstrLogA(b, logb, 2,
                           name='log_{}'.format(b.varName),
                           options="FuncPieces=-1 FuncPieceError=0.001")
        m.addConstr(logpdt == loga + logb,
                    name='pdt_{}_{}'.format(pdt_name, self))
        return pdt

    def update_ns(self, rows, mem, m):
        # Access time based on mem
        # TODO:: better fits for mem and t
        self.m_access_time = m.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                      name='m_access_time_{}'.format(self))
        m.addGenConstrPWL(mem, self.m_access_time, self.mem_par, self.mem_ns,
                          "mem_access_time_{}".format(self))

        # single core ns model
        t = 18 + (rows-4)*6 - rows
        self.ns_single = m.addVar(vtype=GRB.CONTINUOUS,
                                  name='ns_single_{}'.format(self))
        self.pdt_m_rows = self.get_pdt_var(self.m_access_time,
                                           rows, 'm_rows', m)
        m.addConstr(t * self.mem_ns[0] + self.pdt_m_rows
                    + rows * self.hash_ns
                    == self.ns_single, name='ns_single_{}'.format(self))

        # Multi-core model
        self.cores_sketch = m.addVar(vtype=GRB.INTEGER, lb=1, ub=self.cores,
                                     name='cores_sketch_{}'.format(self))
        self.cores_dpdk = m.addVar(vtype=GRB.INTEGER, lb=1, ub=self.cores,
                                   name='cores_dpdk_{}'.format(self))
        m.addConstr(self.cores_sketch + self.cores_dpdk <= self.cores,
                    name='capacity_cores_{}'.format(self))
        self.ns_sketch = m.addVar(vtype=GRB.CONTINUOUS,
                                  name='ns_sketch_{}'.format(self), lb=0)
        self.ns_dpdk = m.addVar(vtype=GRB.CONTINUOUS,
                                name='ns_dpdk_{}'.format(self))
        self.ns = m.addVar(vtype=GRB.CONTINUOUS,
                           name='ns_{}'.format(self))

        # Multi-core sketching
        self.pdt_nsk_c = self.get_pdt_var(self.ns_sketch,
                                          self.cores_sketch, 'nsk_c', m)
        m.addConstr(self.pdt_nsk_c == self.ns_single,
                    name='ns_sketch_{}'.format(self))

        # Amdahl's law
        self.dpdk_single_ns = 1000/self.dpdk_single_core_thr
        self.pdt_nsc_dpdk = self.get_pdt_var(
            self.ns_dpdk, self.cores_dpdk, 'nsc_dpdk', m)
        m.addConstr(
            (self.cores_dpdk*(1-cpu.fraction_parallel)
             + cpu.fraction_parallel)*self.dpdk_single_ns
            == self.pdt_nsc_dpdk, name='ns_dpdk_{}'.format(self))
        m.addGenConstrMax(self.ns, [self.ns_dpdk, self.ns_sketch],
                          name='ns_{}'.format(self))

    def res(self, rows, mem):
        return 10*(self.cores_dpdk + self.cores_sketch)

    def __repr__(self):
        return self.name


class p4(param):

    def update_ns(self, rows, mem, m):
        self.ns = m.addVar(vtype=GRB.CONTINUOUS, name='ns_{}'.format(self))
        m.addConstr(self.ns == 1000 / self.line_thr, name='ns_{}'.format(self))

    def res(self, rows, mem):
        return rows/self.meter_alus + mem/self.sram

    def __repr__(self):
        return self.name


# Global Params
tolerance = 0.999
invtol = 1-tolerance
cell_size = 4
KB2B = 1024

# Topology and Requirements
# Query and placement abstraction
eps0 = 1e-5
del0 = 0.02
queries = [cm_sketch(eps0=eps0*50, del0=del0),
           cm_sketch(eps0=eps0, del0=del0),
           cm_sketch(eps0=eps0*100, del0=del0/2)]
partitions = []
for (i, q) in enumerate(queries, 1):
    q.sketch_id = i
    num_rows = q.rows()
    partitions += [(r+1, q) for r in range(num_rows)]

# All memory measured in KB unless otherwise specified
devices = [
    cpu(mem_par=[1.1875, 32, 1448.15625, 5792.625, 32768.0, 440871.90625],
        mem_ns=[0.539759, 0.510892, 5.04469, 5.84114, 30.6627, 39.6981],
        hash_ns=3.5, cores=7, dpdk_single_core_thr=35,
        max_mem=32768, max_rows=9, name='cpu_1'),
    p4(meter_alus=4, sram=48, stages=12, line_thr=148,
       max_mpp=48, max_mem=48*12, max_rows=12, name='p4_1')
]
numdevices = len(devices)
numpartitions = len(partitions)

# Build model
m = gp.Model('netmon')

# Decl of vars
frac = m.addVars(numdevices, numpartitions, vtype=GRB.CONTINUOUS,
                 lb=0, ub=1, name='frac')
mem = m.addVars(numdevices, numpartitions, vtype=GRB.CONTINUOUS,
                lb=0, name='mem')
rows = m.addVars(numdevices, numpartitions, vtype=GRB.BINARY,
                 name='rows', lb=0)
m.addConstrs((rows[i, j] >= frac[i, j]
              for i in range(numdevices)
              for j in range(numpartitions)), name='r_ceil0')

m.addConstrs((rows[i, j] <= frac[i, j] + tolerance
              for i in range(numdevices)
              for j in range(numpartitions)), name='r_ceil1')
for pnum in range(numpartitions):
    m.addConstr(frac.sum('*', pnum) == 1,
                name='cov_{}'.format(pnum))

# Accuracy constraints
for (pnum, p) in enumerate(partitions):
    sk = p[1]
    mm = sk.min_mem()
    m.addConstrs((mem[dnum, pnum] >= mm * frac[dnum, pnum]
                  for dnum in range(numdevices)), name='accuracy_{}'.format(p))

# Capacity constraints and throughput
resacc = gp.LinExpr()
for (dnum, d) in enumerate(devices):
    # Simple total model
    rows_tot = m.addVar(vtype=GRB.INTEGER, name='rows_tot_{}'.format(d))
    mem_tot = m.addVar(vtype=GRB.CONTINUOUS, name='mem_tot_{}'.format(d))
    m.addConstr(rows_tot == rows.sum(dnum, '*'), name='rows_tot_{}'.format(d))
    m.addConstr(mem_tot == mem.sum(dnum, '*'), name='mem_tot_{}'.format(d))

    # Capacity constraints
    m.addConstr(mem_tot, GRB.LESS_EQUAL, d.max_mem,
                'capacity_mem_tot_{}'.format(d))
    m.addConstr(rows_tot, GRB.LESS_EQUAL, d.max_rows,
                'capacity_compute_{}'.format(d))

    if isinstance(d, p4):
        for (pnum, p) in enumerate(partitions):
            m.addConstr(mem[dnum, pnum] <= d.max_mpp,
                        'capacity_mem_par_{}'.format(d))

    # Throughput
    # NOTE: Following function updates m
    d.update_ns(rows_tot, mem_tot, m)

    # Resources
    resacc += d.res(rows_tot, mem_tot)

ns_series = [d.ns for d in devices]
ns = m.addVar(vtype=GRB.CONTINUOUS, name='ns')
m.addGenConstrMax(ns, ns_series, name='ns_overall')
res = m.addVar(vtype=GRB.CONTINUOUS, name='res_overall')
m.addConstr(res == resacc, name='res')

# TODO:: fill with multi objective
m.ModelSense = GRB.MINIMIZE
m.setObjectiveN(ns, 0, 10, reltol=0.01, name='ns')
m.setObjectiveN(res, 1, 5, reltol=0.01, name='res')

start = time.time()
m.update()
end = time.time()
print("Model update took: {} seconds".format(end - start))

m.write("prog.lp")

start = time.time()
m.optimize()
end = time.time()
print("Model optimize took: {} seconds".format(end - start))

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))



'''
Extra Codes

        # self.log_m_access_time = m.addVar(vtype=GRB.CONTINUOUS,
        #                                   name='log_m_access_time_{}'.format(self))
        # self.log_rows = m.addVar(vtype=GRB.CONTINUOUS,
        #                          name='log_rows_{}'.format(self))
        # self.log_pdt_m_rows = m.addVar(vtype=GRB.CONTINUOUS,
        #                                name='log_pdt_m_rows{}'.format(self))
        # self.pdt_m_rows = m.addVar(vtype=GRB.CONTINUOUS,
        #                            name='pdt_m_rows_{}'.format(self))
        # m.addGenConstrLogA(self.m_access_time, self.log_m_access_time, 2,
        #                    name='log_mem_{}'.format(self))
        # m.addGenConstrLogA(rows, self.log_rows, 2,
        #                    name='log_rows_{}'.format(self))
        # m.addGenConstrLogA(self.pdt_m_rows, self.log_pdt_m_rows, 2,
        #                    name='log_pdt_m_rows_{}'.format(self))


        # m.addQConstr(t * self.mem_ns[0] + self.m_access_time * rows
        #              + rows * self.hash_ns
        #              - self.ns_single <= invtol,
        #              name='ns_u_single_{}'.format(self))
        # m.addQConstr(t * self.mem_ns[0] + self.m_access_time * rows
        #              + rows * self.hash_ns
        #              - self.ns_single >= -invtol,
        #              name='ns_l_single_{}'.format(self))

        # m.addQConstr(self.ns_sketch*self.cores_sketch
        #              - self.ns_single <= invtol,
        #              name='ns_u_sketch_{}'.format(self))
        # m.addQConstr(self.ns_sketch*self.cores_sketch
        #              - self.ns_single >= invtol,
        #              name='ns_l_sketch_{}'.format(self))


        # m.addQConstr(
        #     ((1-cpu.fraction_parallel)*self.cores_dpdk + cpu.fraction_parallel)
        #     * self.ns_dpdk
        #     - self.dpdk_single_core_thr * self.cores_dpdk <= invtol,
        #     name='ns_u_dpdk_{}'.format(self))
        # m.addQConstr(
        #     ((1-cpu.fraction_parallel)*self.cores_dpdk + cpu.fraction_parallel)
        #     * self.ns_dpdk
        #     - self.dpdk_single_core_thr * self.cores_dpdk >= -invtol,
        #     name='ns_l_dpdk_{}'.format(self))


'''

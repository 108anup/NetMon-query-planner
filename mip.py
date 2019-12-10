import gurobipy as gp
from gurobipy import GRB
import math


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

    def ns(self, rows, mem, m):
        # TODO:: better fits for mem and t
        t = 18 + (rows-4)*6 - rows
        self.m_access_time = m.addVar(vtype=GRB.CONTINUOUS, lb=0)
        m.addGenConstrPWL(mem, self.m_access_time, self.mem_par, self.mem_ns)
        self.ns_single = m.addVar(vtype=GRB.CONTINUOUS)
        m.addConstr(self.ns_single == t * self.mem_ns[0]
                    + self.m_access_time * rows + rows * self.hash_ns)
        self.cores_sketch = m.addVar(vtype=GRB.INTEGER, lb=1, ub=self.cores)
        self.cores_dpdk = m.addVar(vtype=GRB.INTEGER, lb=1, ub=self.cores)
        m.addConstr(self.cores.sketch + self.cores.dpdk <= self.cores)
        self.ns_sketch = m.addVar(vtype=GRB.CONTINUOUS)
        self.ns_dpdk = m.addVar(vtype=GRB.CONTINUOUS)
        self.ns = m.addVar(vtype=GRB.CONTINUOUS)
        m.addConstr(self.ns_sketch*self.cores_sketch == self.ns_single)
        m.addConstr(
            ((1-cpu.fraction_parallel)*self.cores_dpdk + cpu.fraction_parallel)
            * self.ns_dpdk == self.dpdk_single_core_thr * self.cores_dpdk)
        m.addConstr(self.ns == gp.min_(self.ns_dpdk, self.ns_sketch))

    def res(self, rows, mem):
        return self.cores_dpdk + self.cores_sketch


class p4(param):

    def ns(self, rows, mem):
        self.ns = m.addVar(vtype=GRB.CONTINUOUS)
        m.addConstr(self.ns == 1000 / self.line_thr)

    def res(self, rows, mem):
        return rows/self.meter_alus + mem/self.sram


# Global Params
tolerance = 0.999
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
    rows = q.rows()
    partitions += [(r+1, q) for r in range(rows)]

# All memory measured in KB unless otherwise specified
devices = [
    cpu(mem_par=[1.1875, 32, 1448.15625, 5792.625, 32768.0, 440871.90625],
        mem_ns=[0.539759, 0.510892, 5.04469, 5.84114, 30.6627, 39.6981],
        hash_ns=3.5, cores=7, dpdk_single_core_thr=35,
        max_mem=32768, max_rows=9),
    p4(meter_alus=4, sram=48, stages=12, line_thr=148,
       max_mpp=48, max_mem=48*12, max_rows=12)
]

# Build model
m = gp.Model('netmon')

# Decl of vars
frac = m.addVars(devices, partitions, vtype=GRB.CONTINUOUS, lb=0, ub=1)
mem = m.addVars(devices, partitions, vtype=GRB.CONTINUOUS, lb=1)
rows = m.addVars(devices, partitions, vtype=GRB.BINARY)
m.addConstrs((rows[i, j] >= frac[i, j]
              for i in range(len(devices))
              for j in range(len(partitions))), name='r_ceil0')

m.addConstrs((rows[i, j] <= frac[i, j] + tolerance
              for i in range(len(devices))
              for j in range(len(partitions))), name='r_ceil1')

# Accuracy constraints
for (pnum, p) in enumerate(partitions):
    sk = p[1]
    mm = sk.min_mem()
    m.addConstrs((mem[pnum, d] >= mm * frac[pnum][d]
                  for d in range(len(devices))), name='accuracy')

# Capacity constraints and throughput
resacc = gp.QuadExpr()
for (dnum, d) in enumerate(devices):
    # Simple total model
    rows_tot = rows.sum(dnum, '*')
    mem_tot = mem.sum(dnum, '*')

    # Capacity constraints
    m.addConstrs(mem_tot <= d.max_mem)
    m.addConstrs(rows_tot <= d.max_rows)

    if isinstance(d, p4):
        for (pnum, p) in enumerate(partitions):
            m.addConstrs(mem[dnum, pnum] < d.max_mpp)

    # Throughput
    # NOTE: Following function updates m
    d.ns(rows_tot, mem_tot, m)

    # Resources
    resacc += d.res(m)

ns_series = [d.ns for d in devices]
ns = m.addVars(vtype=GRB.CONTINUOUS)
m.addConstr(ns == gp.min_(ns_series))
res = m.addVars(vtype=GRB.CONTINUOUS)
m.addConstr(res == resacc)

# TODO:: fill with multi objective



m.setObjective(obj, GRB.MAXIMIZE)
m.optimize()

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % obj.getValue())



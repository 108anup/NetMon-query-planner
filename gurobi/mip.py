import sys
import time

import gurobipy as gp
import ipdb
from gurobipy import GRB

from config import config, common_config


def get_partitions(queries):
    partitions = []
    for (i, q) in enumerate(queries, 1):
        q.sketch_id = i
        num_rows = q.rows()
        partitions += [(r+1, q) for r in range(num_rows)]
    return partitions


def solve(devices, queries):

    partitions = get_partitions(queries)
    numdevices = len(devices)
    numpartitions = len(partitions)

    # Build model
    m = gp.Model('netmon')

    # Decl of vars
    frac = m.addVars(numdevices, numpartitions, vtype=GRB.CONTINUOUS,
                     lb=0, ub=1, name='frac')
    mem = m.addVars(numdevices, numpartitions, vtype=GRB.CONTINUOUS,
                    lb=0, name='mem')
    # Ceiling
    # rows = m.addVars(numdevices, numpartitions, vtype=GRB.BINARY,
    #                  name='rows', lb=0)
    # m.addConstrs((rows[i, j] >= frac[i, j]
    #               for i in range(numdevices)
    #               for j in range(numpartitions)), name='r_ceil0')

    # m.addConstrs((rows[i, j] <= frac[i, j] + common_config.tolerance
    #               for i in range(numdevices)
    #               for j in range(numpartitions)), name='r_ceil1')

    # Frac == 0 -> mem == 0
    # m.addConstrs(((frac[i, j] == 0) >> (mem[i, j] == 0)
    #               for i in range(numdevices)
    #               for j in range(numpartitions)), name='frac_mem')

    for pnum in range(numpartitions):
        m.addConstr(frac.sum('*', pnum) == 1,
                    name='cov_{}'.format(pnum))

    # Accuracy constraints
    for (pnum, p) in enumerate(partitions):
        sk = p[1]
        mm = sk.min_mem()
        m.addConstrs((mem[dnum, pnum] >= mm * frac[dnum, pnum]
                      for dnum in range(numdevices)),
                     name='accuracy_{}'.format(p))

    # Capacity constraints and throughput
    resacc = gp.LinExpr()
    for (dnum, d) in enumerate(devices):
        # Simple total model
        # Capacity constraints included in bounds
        rows_tot = m.addVar(vtype=GRB.CONTINUOUS,
                            name='rows_tot_{}'.format(d),
                            lb=0, ub=d.max_rows)
        mem_tot = m.addVar(vtype=GRB.CONTINUOUS,
                           name='mem_tot_{}'.format(d),
                           lb=0, ub=d.max_mem)
        m.addConstr(rows_tot == frac.sum(dnum, '*'),
                    name='rows_tot_{}'.format(d))
        m.addConstr(mem_tot == mem.sum(dnum, '*'),
                    name='mem_tot_{}'.format(d))

        # ipdb.set_trace()
        if hasattr(d, 'max_mpp'):
            for (pnum, p) in enumerate(partitions):
                m.addConstr(mem[dnum, pnum] <= d.max_mpp,
                            'capacity_mem_par_{}'.format(d))

        # Throughput
        # NOTE: Following function updates m
        d.update_ns(rows_tot, mem_tot, m)
        d.rows_tot = rows_tot
        d.mem_tot = mem_tot

        # Resources
        resacc += d.res(rows_tot, mem_tot)

    ns_series = [d.ns for d in devices]
    ns = m.addVar(vtype=GRB.CONTINUOUS, name='ns')
    m.addGenConstrMax(ns, ns_series, name='ns_overall')
    res = m.addVar(vtype=GRB.CONTINUOUS, name='res_overall')
    m.addConstr(res == resacc, name='res')

    m.ModelSense = GRB.MINIMIZE
    m.setObjectiveN(ns, 0, 10, reltol=common_config.ns_tol, name='ns')
    m.setObjectiveN(res, 1, 5, reltol=common_config.res_tol, name='res')

    start = time.time()
    m.update()
    end = time.time()
    print("Model update took: {} seconds".format(end - start))

    m.write("progs/prog_{}.lp".format(cfg_num))

    start = time.time()
    m.optimize()
    end = time.time()
    print("Model optimize took: {} seconds".format(end - start))

    if(m.Status == GRB.INFEASIBLE):
        m.computeIIS()
        m.write("progs/infeasible_{}.ilp".format(cfg_num))
    else:
        for v in m.getVars():
            print('%s %g' % (v.varName, v.x))

        # Mapping print:
        print("-----------------------------\n\n"
              "Throughput: {} Mpps, ns per packet: {}".format(1000/ns.x, ns.x))
        print("Resources: {}".format(res.x))

        cur_sketch = 0
        row = 1
        for (pnum, p) in enumerate(partitions):
            if(cur_sketch != p[1].sketch_id):
                print("Sketch {} ({})".format(p[1].sketch_id, p[1].details()))
                row = 1
                cur_sketch = p[1].sketch_id
            print("Row: {}".format(row))
            row += 1

            for (dnum, d) in enumerate(devices):
                print("{}".format(frac[dnum, pnum].x), end='    ')

            print('\n')

        for (dnum, d) in enumerate(devices):
            print("Device {}:".format(d))
            print(d.resource_stats())
            print("Rows total: {}".format(d.rows_tot.x))
            print("Mem total: {}\n".format(d.mem_tot.x))

    # ipdb.set_trace()


cfg_num = 0
if(len(sys.argv) > 1):
    cfg_num = int(sys.argv[1])

cfg = config[cfg_num]
solve(cfg.devices, cfg.queries)

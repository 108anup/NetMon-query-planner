import sys
import time

import gurobipy as gp
import ipdb
from gurobipy import GRB

from config import config, common_config, eps0


def get_partitions(queries):
    partitions = []
    for (i, q) in enumerate(queries):
        q.sketch_id = i
        num_rows = q.rows()
        start_idx = len(partitions)
        q.partitions = [start_idx + r for r in range(num_rows)]
        partitions += [(r, q) for r in range(num_rows)]
    return partitions


def update_flows(flows, queries):
    for f in flows:
        f.partitions = []
        for q in f.queries:
            coverage_requirement = q[1]
            q_idx = q[0]
            for p_idx in queries[q_idx].partitions:
                f.partitions.append((p_idx, coverage_requirement))


def solve(devices, queries, flows):

    partitions = get_partitions(queries)
    update_flows(flows, queries)
    numdevices = len(devices)
    numpartitions = len(partitions)

    # Build model
    m = gp.Model('netmon')

    # Decl of vars
    """
    TODO:
    See what other people define by coverage

    frac represents fraction of row monitored on a device
    We are incorporating coverage by saying the fractions may not sum upto 1
    then a packet may or may not be sampled (coin toss w.p. sample or not)
    given a packet is sampled decide which device will the packet be sampled at
    (use hash function for this)

    To do this 2 approaches:
    1. edge devices have to modify packets to convey whether packet will be
       sampled and decide where to sample and then remove this info from header
    2. each device makes a local decision. First check hash range and then
       coin toss.

    Need to include above costs. (1) is more efficient.
    """
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

    # Coverage Constraints
    # for pnum in range(numpartitions):
    #     m.addConstr(frac.sum('*', pnum) == 1,
    #                 name='cov_{}'.format(pnum))
    for (fnum, f) in enumerate(flows):
        for p in f.partitions:
            pnum = p[0]
            coverage_requirement = p[1]
            sum_expr = gp.quicksum(frac[dnum, pnum] for dnum in f.path)
            # sum_expr = frac[[dnum for dnum in f.path], pnum].sum()
            m.addConstr(sum_expr >= coverage_requirement,
                        name='cov_{}_{}'.format(fnum, pnum))

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

    m.printQuality()

    if(m.Status == GRB.INFEASIBLE):
        m.computeIIS()
        m.write("progs/infeasible_{}.ilp".format(cfg_num))
    else:
        print("\n\nDEBUG -----------------------\n")
        for v in m.getVars():
            print('%s %g' % (v.varName, v.x))

        # Mapping print:
        dbg = None
        if(common_config.fileout == True):
            dbg = open('strobe.out', 'a')
        print("-----------------------------\n\n"
              "Throughput: {} Mpps, ns per packet: {}".format(1000/ns.x, ns.x))
        print("Resources: {}".format(res.x))

        if(common_config.fileout == True):
            dbg.write("-----------------------------\n\n"
                      "Throughput: {} Mpps, ns per packet: {}\n".format(1000/ns.x, ns.x))
            dbg.write("Resources: {}\n".format(res.x))

        cur_sketch = -1
        row = 1
        for (pnum, p) in enumerate(partitions):
            if(cur_sketch != p[1].sketch_id):
                print("Sketch ({}) ({})".format(p[1].sketch_id, p[1].details()))
                if(common_config.fileout == True):
                    dbg.write("Sketch {} ({})\n".format(p[1].sketch_id, p[1].details()))
                row = 1
                cur_sketch = p[1].sketch_id
            print("Row: {}".format(row))
            row += 1

            for (dnum, d) in enumerate(devices):
                print("{}".format(frac[dnum, pnum].x), end='    ')

            print('\n')

        for (dnum, d) in enumerate(devices):
            print("Device ({}) {}:".format(dnum, d))
            print(d.resource_stats())
            print("Rows total: {}".format(d.rows_tot.x))
            print("Mem total: {}\n".format(d.mem_tot.x))

        if(common_config.fileout == True):
            for (dnum, d) in enumerate(devices):
                dbg.write("Device ({}) {}:\n".format(dnum, d))
                dbg.write(d.resource_stats() + "\n")
                dbg.write("Rows total: {}\n".format(d.rows_tot.x))
                dbg.write("Mem total: {}\n\n".format(d.mem_tot.x))
            dbg.close()

        for (fnum, f) in enumerate(flows):
            print("Flow {}:".format(fnum))
            print("queries: {}".format(f.queries))
            print("path: {}".format(f.path))

    # ipdb.set_trace()


cfg_num = 0
if(len(sys.argv) > 1):
    cfg_num = int(sys.argv[1])

cfg = config[cfg_num]
if(cfg_num == 3):
    for eps0_mul in [1, 4, 10, 23]:
        cfg.queries[0].eps0 = eps0 * eps0_mul
        solve(cfg.devices, cfg.queries)
else:
    solve(cfg.devices, cfg.queries, cfg.flows)

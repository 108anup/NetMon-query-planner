import sys
import time
import re

import gurobipy as gp
import ipdb
from gurobipy import GRB

from config import config, common_config, eps0, update_config
from cli import generate_parser
from devices import cpu, p4


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
    then
    (1) a packet may or may not be sampled (coin toss w.p. sample or not)
    given a packet is sampled decide which device will the packet be sampled at
    (use hash function for this)

    To do this 2 approaches:
    1. edge devices have to modify packets to convey whether packet will be
       sampled and decide where to sample and then remove this info from header
    2. each device makes a local decision. First check hash range and then
       coin toss.

    Need to include above costs. 2. is more efficient.

    OR
    (2) some keys may not be sampled ever. based on their hash.
    So we hash into [0,1] and then [0,0.5] -> sketch 1, [0.5,0.9] sketch 2 and
    [0.9,1] are never hashed.

    Need to include cost of branch i.e. if hash lies in relevant range.

    (2) is more efficient overall. But has different tradeoff than (1)
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

    if(common_config.solver == 'univmon'):
        mem_series = [d.mem_tot for d in devices]
        max_mem = m.addVar(vtype=GRB.CONTINUOUS, name='max_mem')
        m.addGenConstrMax(max_mem, mem_series, name='mem_overall')

    if('univmon_greedy' in common_config.solver):
        mem_series_p4 = [d.mem_tot for d in devices if isinstance(d, p4)]
        mem_series_cpu = [d.mem_tot for d in devices if isinstance(d, cpu)]
        max_mem_p4 = m.addVar(vtype=GRB.CONTINUOUS, name='max_mem_p4')
        m.addGenConstrMax(max_mem_p4, mem_series_p4, name='mem_overall_p4')
        max_mem_cpu = m.addVar(vtype=GRB.CONTINUOUS, name='max_mem_cpu')
        m.addGenConstrMax(max_mem_cpu, mem_series_cpu, name='mem_overall_cpu')
        if(common_config.solver == 'univmon_greedy_rows'):
            rows_series_p4 = [d.rows_tot for d in devices if isinstance(d, p4)]
            rows_series_cpu = [d.rows_tot for d in devices if isinstance(d, cpu)]
            max_rows_p4 = m.addVar(vtype=GRB.CONTINUOUS, name='max_rows_p4')
            m.addGenConstrMax(max_rows_p4, rows_series_p4, name='rows_overall_p4')
            max_rows_cpu = m.addVar(vtype=GRB.CONTINUOUS, name='max_rows_cpu')
            m.addGenConstrMax(max_rows_cpu, rows_series_cpu, name='rows_overall_cpu')

    m.ModelSense = GRB.MINIMIZE
    if(common_config.solver == 'univmon'):
        m.setObjective(max_mem)
    elif(common_config.solver == 'univmon_greedy'):
        m.setObjectiveN(max_mem_cpu, 0, 10, name='cpu_mem_load')
        m.setObjectiveN(max_mem_p4, 1, 5, name='p4_mem_load')
    elif(common_config.solver == 'univmon_greedy_rows'):
        m.setObjectiveN(max_rows_cpu, 0, 20, name='cpu_rows_load')
        m.setObjectiveN(max_rows_p4, 1, 15, name='p4_rows_load')
        m.setObjectiveN(max_mem_cpu, 2, 10, name='cpu_load_mem')
        m.setObjectiveN(max_mem_p4, 3, 5, name='p4_load_mem')
    elif(common_config.solver == 'netmon'):
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
        post_optimize(m, devices, partitions, flows, ns, res, frac)
        if('univmon' in common_config.solver):
            '''
            prefixes = ['cores_sketch_cpu',
                        'cores_dpdk_cpu', 'ns_sketch_cpu', 'ns_dpdk_cpu',
                        'ns_cpu', 'pdt_nsk_c_cpu', 'pdt_nsc_dpdk_cpu', 'ns$',
                        'res_overall$']
            regex = '|'.join(prefixes)
            prog = re.compile(regex)
            for v in m.getVars():
                if (not prog.match(v.varName)):
                    print(v.varName)
                    if(abs(v.x) < 1e-5):
                        m.addConstr(v == 0)
                    else:
                        m.addConstr(v == v.x)

            for d in devices:
                if(isinstance(d, cpu)):
                    m.addConstr(d.cores_dpdk == 6)
            '''
            prefixes = ['frac', 'mem\[']
            regex = '|'.join(prefixes)
            prog = re.compile(regex)
            for v in m.getVars():
                if (prog.match(v.varName)):
                    print(v.varName)
                    if(abs(v.x) < 1e-5):
                        m.addConstr(v == 0)
                    else:
                        m.addConstr(v == v.x)

            m.NumObj = 2
            m.setObjectiveN(ns, 0, 10, reltol=common_config.ns_tol, name='ns')
            m.setObjectiveN(res, 1, 5, reltol=common_config.res_tol, name='res')
            # m.setObjectiveN(ns, 0, 10, name='ns')
            # m.setObjectiveN(ns, 0, 10, name='ns')
            m.optimize()
            print('\n')

            if(m.Status == GRB.INFEASIBLE):
                m.computeIIS()
                m.write("progs/infeasible_{}.ilp".format(cfg_num))
            else:
                post_optimize(m, devices, partitions, flows, ns, res, frac)
    # ipdb.set_trace()


def post_optimize(m, devices, partitions, flows, ns, res, frac):

    m.printQuality()
    print("\n\nDEBUG -----------------------\n")
    print("Objective: {}".format(m.objVal))
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
            print("\nSketch ({}) ({})".format(p[1].sketch_id, p[1].details()))
            if(common_config.fileout == True):
                dbg.write("Sketch {} ({})\n".format(p[1].sketch_id, p[1].details()))
            row = 1
            cur_sketch = p[1].sketch_id
        print("Row: {}".format(row))
        row += 1

        for (dnum, d) in enumerate(devices):
            print("{:0.3f}".format(frac[dnum, pnum].x), end='    ')
        tot_frac = 0
        for (dnum, d) in enumerate(devices):
            tot_frac += (frac[dnum, pnum].x)
        print("\nTotal frac: {:0.3f}".format(tot_frac))

    for (dnum, d) in enumerate(devices):
        print("Device ({}) {}:".format(dnum, d))
        print(d.resource_stats())
        print("Rows total: {}".format(d.rows_tot.x))
        print("Mem total: {}".format(d.mem_tot.x))
        print("Throughput: {}\n".format(1000/d.ns.x))

    if(common_config.fileout == True):
        for (dnum, d) in enumerate(devices):
            dbg.write("Device ({}) {}:\n".format(dnum, d))
            dbg.write(d.resource_stats() + "\n")
            dbg.write("Rows total: {}\n".format(d.rows_tot.x))
            dbg.write("Mem total: {}\n\n".format(d.mem_tot.x))
        dbg.close()

    '''
    for (fnum, f) in enumerate(flows):
        print("Flow {}:".format(fnum))
        print("queries: {}".format(f.queries))
        print("path: {}".format(f.path))
    '''

parser = generate_parser()
args = parser.parse_args(sys.argv[1:])
update_config(args)


cfg_num = int(args.config)
cfg = config[cfg_num]
if(cfg_num == 3):
    for eps0_mul in [1, 4, 10, 23]:
        cfg.queries[0].eps0 = eps0 * eps0_mul
        solve(cfg.devices, cfg.queries)
else:
    solve(cfg.devices, cfg.queries, cfg.flows)

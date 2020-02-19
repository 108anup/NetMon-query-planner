import os
import pickle
import sys
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from cli import generate_parser
from common import log, param, setup_logging, log_time
from config import common_config, config, eps0, update_config
from solvers import (add_device_aware_constraints, solver_to_class,
                     solver_to_num)


@log_time
def get_partitions(queries):
    partitions = []
    for (i, q) in enumerate(queries):
        q.sketch_id = i
        num_rows = q.rows()
        start_idx = len(partitions)
        if(common_config.horizontal_partition):
            q.partitions = [start_idx + r for r in range(num_rows)]
            partitions += [param(partition_id=start_idx+r,
                                 sketch=q, num_rows=1)
                           for r in range(num_rows)]
        else:
            q.partitions = [start_idx]
            partitions += [param(partition_id=start_idx, sketch=q,
                                 num_rows=num_rows)]
    return partitions


@log_time
def map_flows_partitions(flows, queries):
    for f in flows:
        f.partitions = []
        for q in f.queries:
            coverage_requirement = q[1]
            q_idx = q[0]
            for p_idx in queries[q_idx].partitions:
                f.partitions.append((p_idx, coverage_requirement))


@log_time
def solve(devices, queries, flows):

    partitions = get_partitions(queries)
    map_flows_partitions(flows, queries)
    numdevices = len(devices)
    numpartitions = len(partitions)

    m = gp.Model('netmon')
    if(not common_config.mipout):
        m.setParam(GRB.Param.LogToConsole, 0)

    log.info("Building model with:\n"
             "{} devices, {} partitions and {} flows"
             .format(numdevices, numpartitions, len(flows)))

    # Fraction of partition on device
    start = time.time()
    if(common_config.vertical_partition):
        frac = m.addVars(numdevices, numpartitions, vtype=GRB.CONTINUOUS,
                         lb=0, ub=1,
                         name='frac')
    else:
        frac = m.addVars(numdevices, numpartitions, vtype=GRB.BINARY,
                         name='frac')
    # Memory taken by partition
    mem = m.addVars(numdevices, numpartitions, vtype=GRB.CONTINUOUS,
                    lb=0, name='mem')
    end = time.time()
    log.info("Adding frac and mem vars took: {}s".format(end - start))
    # Ceiling
    # rows = m.addVars(numdevices, numpartitions, vtype=GRB.BINARY,
    #                  name='rows', lb=0)
    # m.addConstrs((rows[i, j] >= frac[i, j]
    #               for i in range(numdevices)
    #               for j in range(numpartitions)), name='r_ceil0')

    # m.addConstrs((rows[i, j] <= frac[i, j] + common_config.tolerance
    #               for i in range(numdevices)
    #               for j in range(numpartitions)), name='r_ceil1')

    # No memory if not mapped
    # Frac == 0 -> mem == 0
    # m.addConstrs(((frac[i, j] == 0) >> (mem[i, j] == 0)
    #               for i in range(numdevices)
    #               for j in range(numpartitions)), name='frac_mem')

    # Coverage Constraints
    # for pnum in range(numpartitions):
    #     m.addConstr(frac.sum('*', pnum) == 1,
    #                 name='cov_{}'.format(pnum))
    start = time.time()
    keep = np.zeros((numdevices, numpartitions))
    for (fnum, f) in enumerate(flows):
        for p in f.partitions:
            pnum = p[0]
            coverage_requirement = p[1]
            sum_expr = gp.quicksum(frac[dnum, pnum] for dnum in f.path)
            m.addConstr(sum_expr >= coverage_requirement,
                        name='cov_{}_{}'.format(fnum, pnum))
            for dnum in f.path:
                keep[dnum][pnum] = 1
    end = time.time()
    log.info("Adding coverage constraints took: {}s".format(end - start))

    # for dnum in range(numdevices):
    #     for pnum in range(numpartitions):
    #         if(keep[dnum][pnum] == 0):
    #             m.addConstr(frac[dnum, pnum] == 0,
    #                         name='frac_not_reqd_{}_{}' .format(dnum, pnum))
    #             m.addConstr(mem[dnum, pnum] == 0,
    #                         name='mem_not_reqd_{}_{}'.format(dnum, pnum))

    # Accuracy constraints
    start = time.time()
    for (pnum, p) in enumerate(partitions):
        sk = p.sketch
        num_rows = p.num_rows
        mm = sk.min_mem()
        m.addConstrs((mem[dnum, pnum] == mm * frac[dnum, pnum] * num_rows
                      for dnum in range(numdevices)),
                     name='accuracy_{}_{}'.format(pnum, sk))
    end = time.time()
    log.info("Adding accuracy constraints took {}s".format(end - start))

    # Load constraints
    start = time.time()
    for (dnum, d) in enumerate(devices):
        # Memory constraints
        # Capacity constraints included in bounds
        mem_tot = m.addVar(vtype=GRB.CONTINUOUS,
                           name='mem_tot_{}'.format(d),
                           lb=0, ub=d.max_mem)
        m.addConstr(mem_tot == mem.sum(dnum, '*'),
                    name='mem_tot_{}'.format(d))
        d.mem_tot = mem_tot

        # Row constraints
        rows_tot = m.addVar(vtype=GRB.CONTINUOUS,
                            name='rows_tot_{}'.format(d), lb=0)
        rows_series = [p.num_rows * frac[dnum, p.partition_id]
                       for p in partitions]
        m.addConstr(rows_tot == gp.quicksum(rows_series),
                    name='rows_tot_{}'.format(d))
        d.rows_tot = rows_tot
    end = time.time()
    log.info("Adding load constraints took: {}s".format(end-start))

    if(solver_to_num[common_config.solver] > 0):
        add_device_aware_constraints(devices, queries, flows,
                                     partitions, m, frac, mem)

    solver_cls = solver_to_class[common_config.solver]
    solver = solver_cls(devices=devices, queries=queries, flows=flows,
                        partitions=partitions, m=m, frac=frac, mem=mem)
    m.setParam(GRB.Param.TimeLimit, 120)
    # m.setParam(GRB.Param.MIPGapAbs, common_config.mipgapabs)
    # m.setParam(GRB.Param.MIPGap, common_config.mipgap)
    solver.add_constraints()
    m.ModelSense = GRB.MINIMIZE
    solver.add_objective()

    log.info("Starting model update")
    start = time.time()
    m.update()
    end = time.time()
    update_time = end - start

    m.write("progs/prog_{}.lp".format(cfg_num))
    log.info("")
    start = time.time()
    m.optimize()
    end = time.time()

    log.info("-"*50)
    log.info("Model update took: {} seconds".format(update_time))
    log.info("Model optimize took: {} seconds".format(end - start))
    log.info("-"*50)

    if(m.Status == GRB.INFEASIBLE):
        # m.computeIIS()
        # m.write("progs/infeasible_{}.ilp".format(cfg_num))

        if(not (common_config.output_file is None)):
            f = open(common_config.output_file, 'a')
            f.write("-, -, -, -, -, ")
            f.close()

    else:
        solver.post_optimize()

    end = time.time()
    if(not (common_config.output_file is None)):
        f = open(common_config.output_file, 'a')
        f.write("{:06f}, ".format(end - start))
        f.close()


parser = generate_parser()
args = parser.parse_args(sys.argv[1:])
update_config(args)
setup_logging(args)

cfg_num = common_config.cfg_num
cfg = config[cfg_num]
# if(cfg_num == 3):
#     for eps0_mul in [1, 4, 10, 23]:
#         cfg.queries[0].eps0 = eps0 * eps0_mul
#         solve(cfg.devices, cfg.queries)
# else:


# if (cfg_num == 3):
#     if(os.path.exists('pickle_objs/cfg_3')):
#         cfg_file = open('pickle_objs/cfg_3', 'rb')
#         cfg = pickle.load(cfg_file)
#         cfg_file.close()
#     else:
#         cfg_file = open('pickle_objs/cfg_3', 'wb')
#         pickle.dump(cfg, cfg_file)
#         cfg_file.close()

solve(cfg.devices, cfg.queries, cfg.flows)

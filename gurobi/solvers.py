import re
import sys
import inspect

import gurobipy as gp
from gurobipy import GRB

from common import param, log
from config import common_config

from devices import cpu, p4

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


def write_vars(m):
    log.debug("\nVARIABLES "+"-"*30)
    log.debug("Objective: {}".format(m.objVal))
    for v in m.getVars():
        log.debug('%s %g' % (v.varName, v.x))
    log.debug("-"*50)


def add_device_aware_constraints(devices, queries, flows,
                                 partitions, m, frac, mem):
    for (dnum, d) in enumerate(devices):
        # Row constraints
        rows_tot = m.addVar(vtype=GRB.CONTINUOUS,
                            name='rows_tot_{}'.format(d),
                            lb=0, ub=d.max_rows)
        rows_series = [p.num_rows * frac[dnum, p.partition_id]
                       for p in partitions]
        m.addConstr(rows_tot == gp.quicksum(rows_series),
                    name='rows_tot_{}'.format(d))
        d.rows_tot = rows_tot

        if hasattr(d, 'max_mpp'):
            for (pnum, p) in enumerate(partitions):
                m.addConstr(mem[dnum, pnum] <= d.max_mpr * p.num_rows,
                            'capacity_mem_par_{}'.format(d))


def add_device_model_constraints(devices, queries, flows, partitions, m):
    res_acc = gp.LinExpr()
    for d in devices:
        # Throughput
        # Simple total model
        d.add_ns_constraints(m)

        # Resources
        res_acc += d.res()

    ns_series = [d.ns for d in devices]
    ns = m.addVar(vtype=GRB.CONTINUOUS, name='ns')
    m.addGenConstrMax(ns, ns_series, name='ns_overall')
    res = m.addVar(vtype=GRB.CONTINUOUS, name='ns')
    m.addConstr(res_acc == res, name='res_acc')
    return (ns, res)


class netmon(param):
    def __init__(self, *args, **kwargs):
        super(netmon, self).__init__(*args, **kwargs)

    def add_constraints(self):
        (self.ns, self.res) = add_device_model_constraints(
            self.devices, self.queries, self.flows, self.partitions, self.m)

    def add_objective(self):
        self.m.setObjectiveN(self.ns, 0, 10, reltol=common_config.ns_tol,
                             name='ns')
        self.m.setObjectiveN(self.res, 1, 5, reltol=common_config.res_tol,
                             name='res')

    def post_optimize(self):
        self.m.printQuality()
        write_vars(self.m)

        log_results(self.ns, self.res)
        log_placement(self.devices, self.queries, self.flows, self.partitions,
                      self.m, self.frac)


class univmon(param):
    def __init__(self, *args, **kwargs):
        super(univmon, self).__init__(*args, **kwargs)

    def add_constraints(self):
        mem_series = [d.mem_tot for d in self.devices]
        self.max_mem = self.m.addVar(vtype=GRB.CONTINUOUS, name='max_mem')
        self.m.addGenConstrMax(self.max_mem, mem_series, name='mem_overall')

    def add_objective(self):
        self.m.setObjective(self.max_mem)

    def post_optimize(self, aware=False):
        self.m.printQuality()
        write_vars(self.m)

        log_placement(self.devices, self.queries, self.flows, self.partitions,
                      self.m, self.frac)

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
        for v in self.m.getVars():
            if (prog.match(v.varName)):
                if((v.x) < 0):
                    self.m.addConstr(v == 0)
                else:
                    self.m.addConstr(v == v.x)


        # # Compute Best thr / res tradeoff possible
        # if(not aware):
        #     for (dnum, d) in enumerate(devices):

        if(not aware):
            add_device_aware_constraints(
                self.devices, self.queries, self.flows,
                self.partitions, self.m, self.frac, self.mem)

        (ns, res) = add_device_model_constraints(
            self.devices, self.queries, self.flows, self.partitions, self.m)

        self.m.NumObj = 2
        self.m.setObjectiveN(ns, 0, 10, reltol=common_config.ns_tol, name='ns')
        self.m.setObjectiveN(res, 1, 5, reltol=common_config.res_tol,
                             name='res')
        self.m.update()
        self.m.optimize()

        if(self.m.Status == GRB.INFEASIBLE):
            self.m.computeIIS()
            self.m.write("progs/infeasible_placement_{}.ilp"
                         "".format(common_config.cfg_num))
            return

        # optimal_ns = ns.x * (1 + common_config.ns_tol)
        # self.m.addConstr(ns <= optimal_ns)
        # for d in self.devices:
        #     self.m.addConstr(d.ns <= optimal_ns)
        # self.m.setObjectiveN(res, 0, 10, reltol=common_config.res_tol,
        #                      name='res')
        # self.m.update()
        # self.m.optimize()

        # if(self.m.Status == GRB.INFEASIBLE):
        #     self.m.computeIIS()
        #     self.m.write("progs/infeasible_placement_{}.ilp"
        #                  "".format(common_config.cfg_num))
        #     return

        log_results(ns, res)
        log_placement(self.devices, self.queries, self.flows,
                      self.partitions, self.m, self.frac)


class univmon_greedy(univmon):
    def __init__(self, *args, **kwargs):
        super(univmon_greedy, self).__init__(*args, **kwargs)

    def add_constraints(self):
        mem_series_p4 = [d.mem_tot
                         for d in self.devices
                         if isinstance(d, p4)]
        mem_series_cpu = [d.mem_tot
                          for d in self.devices
                          if isinstance(d, cpu)]
        self.max_mem_p4 = self.m.addVar(vtype=GRB.CONTINUOUS,
                                        name='max_mem_p4')
        self.m.addGenConstrMax(self.max_mem_p4, mem_series_p4,
                               name='mem_overall_p4')
        self.max_mem_cpu = self.m.addVar(vtype=GRB.CONTINUOUS,
                                         name='max_mem_cpu')
        self.m.addGenConstrMax(self.max_mem_cpu, mem_series_cpu,
                               name='mem_overall_cpu')

    def add_objective(self):
        self.m.setObjectiveN(self.max_mem_cpu, 0, 10, name='cpu_mem_load')
        self.m.setObjectiveN(self.max_mem_p4, 1, 5, name='p4_mem_load')

    def post_optimize(self):
        super(univmon_greedy, self).post_optimize(True)


class univmon_greedy_rows(univmon_greedy):
    def __init__(self, *args, **kwargs):
        super(univmon_greedy_rows, self).__init__(*args, **kwargs)

    def add_constraints(self):
        super(univmon_greedy_rows, self).add_constraints()
        rows_series_p4 = [d.rows_tot
                          for d in self.devices
                          if isinstance(d, p4)]
        rows_series_cpu = [d.rows_tot
                           for d in self.devices
                           if isinstance(d, cpu)]
        self.max_rows_p4 = self.m.addVar(vtype=GRB.CONTINUOUS,
                                         name='max_rows_p4')
        self.m.addGenConstrMax(self.max_rows_p4, rows_series_p4,
                               name='rows_overall_p4')
        self.max_rows_cpu = self.m.addVar(vtype=GRB.CONTINUOUS,
                                          name='max_rows_cpu')
        self.m.addGenConstrMax(self.max_rows_cpu, rows_series_cpu,
                               name='rows_overall_cpu')

    def add_objective(self):
        self.m.setObjectiveN(self.max_rows_cpu, 0, 20, name='cpu_rows_load')
        self.m.setObjectiveN(self.max_rows_p4, 1, 15, name='p4_rows_load')
        self.m.setObjectiveN(self.max_mem_cpu, 2, 10, name='cpu_load_mem')
        self.m.setObjectiveN(self.max_mem_p4, 3, 5, name='p4_load_mem')


def log_results(ns, res):
    log.info("\nThroughput: {} Mpps, ns per packet: {}".format(
        1000/ns.x, ns.x))
    log.info("Resources: {}".format(res.x))


def log_placement(devices, queries, flows, partitions, m, frac):
    for (qnum, q) in enumerate(queries):
        log.info("\nSketch ({}) ({})".format(q.sketch_id,
                                             q.details()))
        row = 1
        for pnum in q.partitions:
            num_rows = partitions[pnum].num_rows
            log.info("Par: {}, Rows: {}".format(row, num_rows))
            row += 1
            par_info = ""
            total_frac = 0
            for (dnum, d) in enumerate(devices):
                par_info += "{:0.3f}    ".format(frac[dnum, pnum].x)
                total_frac += (frac[dnum, pnum].x)
            log.info(par_info)
            log.info("Total frac: {:0.3f}".format(total_frac))

    for (dnum, d) in enumerate(devices):
        log.info("\nDevice ({}) {}:".format(dnum, d))
        res_stats = d.resource_stats()
        if(res_stats != ""):
            log.info(res_stats)
        if(hasattr(d, 'rows_tot')):
            log.info("Rows total: {}".format(d.rows_tot.x))
        log.info("Mem total: {}".format(d.mem_tot.x))
        if(hasattr(d, 'ns')):
            log.info("Throughput: {}".format(1000/d.ns.x))

    log.debug("")
    for (fnum, f) in enumerate(flows):
        log.debug("Flow {}:".format(fnum))
        log.debug("queries: {}".format(f.queries))
        log.debug("path: {}".format(f.path))
    log.info("-"*50)


solver_names = ['univmon', 'univmon_greedy', 'univmon_greedy_rows', 'netmon']
solver_list = [getattr(sys.modules[__name__], s) for s in solver_names]
solver_to_num = {}
solver_to_class = {}
for (solver_num, solver) in enumerate(solver_list):
    solver_to_num[solver.__name__] = solver_num
    solver_to_class[solver.__name__] = solver

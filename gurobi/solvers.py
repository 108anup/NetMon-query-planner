import re
import sys

import gurobipy as gp
from gurobipy import GRB

from common import log, Namespace
from config import common_config
from devices import CPU, P4

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


def get_rounded_val(v):
    if(v < 0):
        return 0
    if(v < common_config.ftol):
        return 0
    return v


def get_val(v):
    if(isinstance(v, (float, int))):
        return v
    else:
        return v.x


def write_vars(m):
    if(not hasattr(m, 'objVal')):
        import ipdb; ipdb.set_trace()
    log.debug("\nVARIABLES "+"-"*30)
    log.debug("Objective: {}".format(m.objVal))
    for v in m.getVars():
        log.debug('%s %g' % (v.varName, v.x))
    log.debug("-"*50)


def add_device_aware_constraints(devices, queries, flows,
                                 partitions, m, frac, mem):
    for (dnum, d) in enumerate(devices):
        m.addConstr(d.rows_tot <= d.max_rows,
                    name='row_capacity_{}'.format(d))

        if hasattr(d, 'max_mpp'):
            for (pnum, p) in enumerate(partitions):
                m.addConstr(mem[dnum, pnum] <= d.max_mpr * p.num_rows,
                            'capacity_mem_par_{}'.format(d))


def check_device_aware_constraints(devices, partitions, mem):
    for (dnum, d) in enumerate(devices):
        if not (d.rows_tot.x <= d.max_rows):
            return False
        if hasattr(d, 'max_mpp'):
            for (pnum, p) in enumerate(partitions):
                if not (mem[dnum, pnum].x <= d.max_mpr * p.num_rows):
                    return False
    return True


def add_device_model_constraints(devices, queries, flows, partitions, m,
                                 ns_req=None):
    res_acc = gp.LinExpr()
    for d in devices:
        # Throughput
        # Simple total model
        d.add_ns_constraints(m, ns_req)

        # Resources
        res_acc += d.res()

    ns = None
    if(ns_req is None):
        ns_series = [d.ns for d in devices]
        ns = m.addVar(vtype=GRB.CONTINUOUS, name='ns')
        m.addGenConstrMax(ns, ns_series, name='ns_overall')
    res = m.addVar(vtype=GRB.CONTINUOUS, name='res')
    m.addConstr(res_acc == res, name='res_acc')
    return (ns, res)


class MIP(Namespace):

    def solve(self):
        self.m = gp.Model(self.__name__)


class Univmon(MIP):

    def add_constraints(self):
        mem_series = [d.mem_tot for d in self.devices]
        self.max_mem = self.m.addVar(vtype=GRB.CONTINUOUS, name='max_mem')
        self.m.addGenConstrMax(self.max_mem, mem_series, name='mem_overall')
        self.tot_mem = self.m.addVar(vtype=GRB.CONTINUOUS,
                                     name='tot_mem')
        self.m.addConstr(self.tot_mem == gp.quicksum(mem_series),
                         name='tot_mem')

    def add_objective(self):
        self.m.setObjectiveN(self.max_mem, 0, 10, name='max_mem')
        self.m.setObjectiveN(self.tot_mem, 1, 5, name='tot_mem')

    def post_optimize(self, aware=False):
        self.m.printQuality()
        write_vars(self.m)

        log_placement(self.devices, self.queries, self.flows, self.partitions,
                      self.m, self.frac, self.dev_par_tuplelist)

        if(not common_config.use_model):
            if (not aware and
                not check_device_aware_constraints(
                    self.devices, self.partitions, self.mem)):
                log.info("Infeasible placement")

                if(not (common_config.output_file is None)):
                    f = open(common_config.output_file, 'a')
                    f.write("-, -, -, -, -, ")
                    f.close()
                return

            res_acc = 0
            ns_max = 0
            for d in self.devices:
                u = gp.Model(d.name)
                d.u = u
                mem_tot = u.addVar(vtype=GRB.CONTINUOUS,
                                   name='mem_tot_{}'.format(d),
                                   lb=0, ub=d.max_mem)
                u.addConstr(mem_tot == get_rounded_val(d.mem_tot.x),
                            name='mem_tot_{}'.format(d))
                d.mem_tot = mem_tot

                # rows_tot = u.addVar(vtype=GRB.CONTINUOUS,
                #                     name='rows_tot_{}'.format(d), lb=0)
                # u.addConstr(rows_tot == d.rows_tot.x, name='rows_tot_{}'.format(d))
                d.rows_tot = get_rounded_val(d.rows_tot.x)
                d.add_ns_constraints(u)

                u.setObjectiveN(d.ns, 0, 10, reltol=common_config.ns_tol,
                                name='ns')
                u.setObjectiveN(d.res(), 1, 5, reltol=common_config.res_tol,
                                name='res')

                u.ModelSense = GRB.MINIMIZE
                u.setParam(GRB.Param.LogToConsole, 0)
                u.update()
                u.optimize()

                write_vars(u)

                ns_max = max(ns_max, u.getObjective(0).getValue())


            res_acc = 0
            used_cores = 0
            total_CPUs = 0
            switch_memory = 0
            for d in self.devices:
                u = d.u
                if(isinstance(d, CPU)):
                    u.addConstr(d.ns >= ns_max, name='global_ns_req_{}'.format(d))

                    u.update()
                    u.optimize()
                    # import ipdb; ipdb.set_trace()
                    write_vars(u)

                    total_CPUs += 1
                    used_cores += d.cores_sketch.x + d.cores_dpdk.x

                if(isinstance(d, P4)):
                    switch_memory += d.mem_tot.x

                ns_max = max(ns_max, u.getObjective(0).getValue())
                res_acc += u.getObjective(1).getValue()

            log_results(ns_max, res_acc, used_cores, total_CPUs,
                        switch_memory)

        else:
            prefixes = ['frac', 'mem\[']
            regex = '|'.join(prefixes)
            prog = re.compile(regex)
            for v in self.m.getVars():
                if (prog.match(v.varName)):
                    # if((v.x) < 0):
                    #     self.m.addConstr(v == 0)
                    # else:
                    #     self.m.addConstr(v == v.x)
                    self.m.addConstr(v == get_rounded_val(v.x))

            self.m.setParam(GRB.Param.FeasibilityTol, common_config.ftol)
            if(not aware):
                add_device_aware_constraints(
                    self.devices, self.queries, self.flows,
                    self.partitions, self.m, self.frac, self.mem)

            (ns, res) = add_device_model_constraints(
                self.devices, self.queries, self.flows,
                self.partitions, self.m)

            self.m.NumObj = 2
            self.m.setObjectiveN(ns, 0, 10, reltol=common_config.ns_tol,
                                 name='ns')
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

            self.m.printQuality()
            write_vars(self.m)

            log_results(ns, res)

        log_placement(self.devices, self.queries, self.flows,
                      self.partitions, self.m, self.frac,
                      self.dev_par_tuplelist)


class UnivmonGreedy(Univmon):
    def __init__(self, *args, **kwargs):
        super(UnivmonGreedy, self).__init__(*args, **kwargs)

    def add_constraints(self):
        mem_series_P4 = [d.mem_tot
                         for d in self.devices
                         if isinstance(d, P4)]
        mem_series_CPU = [d.mem_tot
                          for d in self.devices
                          if isinstance(d, CPU)]
        if(len(mem_series_P4) > 0):
            self.max_mem_P4 = self.m.addVar(vtype=GRB.CONTINUOUS,
                                            name='max_mem_P4')
            self.m.addGenConstrMax(self.max_mem_P4, mem_series_P4,
                                   name='mem_overall_P4')
            self.tot_mem_P4 = gp.quicksum(mem_series_P4)
        if(len(mem_series_CPU) > 0):
            self.max_mem_CPU = self.m.addVar(vtype=GRB.CONTINUOUS,
                                             name='max_mem_CPU')
            self.m.addGenConstrMax(self.max_mem_CPU, mem_series_CPU,
                                   name='mem_overall_CPU')
            self.tot_mem_CPU = gp.quicksum(mem_series_CPU)
        self.tot_mem = self.m.addVar(vtype=GRB.CONTINUOUS,
                                     name='tot_mem')
        self.m.addConstr(self.tot_mem == gp.quicksum(mem_series_CPU)
                         + gp.quicksum(mem_series_P4), name='tot_mem')

    def add_objective(self):
        if(hasattr(self, 'max_mem_CPU')):
            self.m.setObjectiveN(self.tot_mem_CPU, 0, 20, name='tot_mem_CPU')
            self.m.setObjectiveN(self.max_mem_CPU, 1, 15, name='CPU_mem_load')
        if(hasattr(self, 'max_mem_P4')):
            self.m.setObjectiveN(self.tot_mem_P4, 2, 10, name='tot_mem_P4')
            self.m.setObjectiveN(self.max_mem_P4, 3, 5, name='P4_mem_load')
        # self.m.setObjectiveN(self.tot_mem, 2, 1, name='mem_load')

    def post_optimize(self):
        super(UnivmonGreedy, self).post_optimize(True)


class UnivmonGreedyRows(UnivmonGreedy):
    def __init__(self, *args, **kwargs):
        super(UnivmonGreedyRows, self).__init__(*args, **kwargs)

    def add_constraints(self):
        super(UnivmonGreedyRows, self).add_constraints()
        rows_series_P4 = [d.rows_tot
                          for d in self.devices
                          if isinstance(d, P4)]
        rows_series_CPU = [d.rows_tot
                           for d in self.devices
                           if isinstance(d, CPU)]
        if(len(rows_series_P4) > 0):
            self.max_rows_P4 = self.m.addVar(vtype=GRB.CONTINUOUS,
                                             name='max_rows_P4')
            self.m.addGenConstrMax(self.max_rows_P4, rows_series_P4,
                                   name='rows_overall_P4')
            self.tot_rows_P4 = gp.quicksum(rows_series_P4)
        if(len(rows_series_CPU) > 0):
            self.max_rows_CPU = self.m.addVar(vtype=GRB.CONTINUOUS,
                                              name='max_rows_CPU')
            self.m.addGenConstrMax(self.max_rows_CPU, rows_series_CPU,
                                   name='rows_overall_CPU')
            self.tot_rows_CPU = gp.quicksum(rows_series_CPU)
        self.tot_rows = self.m.addVar(vtype=GRB.CONTINUOUS,
                                      name='tot_rows')
        self.m.addConstr(self.tot_rows == gp.quicksum(rows_series_CPU)
                         + gp.quicksum(rows_series_P4), name='tot_rows')

    def add_objective(self):
        if(hasattr(self, 'max_rows_CPU')):
            self.m.setObjectiveN(self.tot_rows_CPU, 0, 100, name='tot_rows_CPU')
            self.m.setObjectiveN(self.max_rows_CPU, 1, 90, name='CPU_rows_load')
        if(hasattr(self, 'max_rows_P4')):
            self.m.setObjectiveN(self.tot_rows_P4, 2, 80, name='tot_rows_P4')
            self.m.setObjectiveN(self.max_rows_P4, 3, 70, name='P4_rows_load')
        # self.m.setObjectiveN(self.tot_rows, 2, 20, name='rows_load')
        if(hasattr(self, 'max_mem_CPU')):
            self.m.setObjectiveN(self.tot_mem_CPU, 4, 60, name='tot_mem_CPU')
            self.m.setObjectiveN(self.max_mem_CPU, 5, 50, name='CPU_load_mem')
        if(hasattr(self, 'max_mem_P4')):
            self.m.setObjectiveN(self.tot_mem_P4, 6, 40, name='tot_mem_P4')
            self.m.setObjectiveN(self.max_mem_P4, 7, 30, name='P4_load_mem')
        # self.m.setObjectiveN(self.tot_mem, 5, 5, name='mem_load')


class Netmon(UnivmonGreedyRows):
    def __init__(self, *args, **kwargs):
        super(Netmon, self).__init__(*args, **kwargs)

    def add_constraints(self):

        # # Initialize with unimon_greedy_rows solution
        # super(Netmon, self).add_constraints()
        # super(Netmon, self).add_objective()
        # self.m.update()
        # self.m.optimize()

        # # numdevices = len(self.devices)
        # # numpartitions = len(self.partitions)

        # # for dnum in range(numdevices):
        # #     for pnum in range(numpartitions):
        # for (dnum, pnum) in self.dev_par_tuplelist:
        #     self.frac[dnum, pnum].start = self.frac[dnum, pnum].x
        #     self.mem[dnum, pnum].start = self.mem[dnum, pnum].x

        # self.ns_req = 14.3
        (self.ns, self.res) = add_device_model_constraints(
            self.devices, self.queries, self.flows, self.partitions, self.m)

    def add_objective(self):
        if(hasattr(self, 'ns_req') and self.ns_req):
            self.m.NumObj = 1
            self.m.setParam(GRB.Param.MIPGapAbs, common_config.mipgapabs)
            self.m.setObjectiveN(self.res, 0, 10, reltol=common_config.res_tol,
                                 name='res')
        else:
            self.m.NumObj = 2
            self.m.setObjectiveN(self.ns, 0, 10, reltol=common_config.ns_tol,
                                 name='ns')
            self.m.setObjectiveN(self.res, 1, 5, reltol=common_config.res_tol,
                                 name='res')

    def post_optimize(self):
        self.m.printQuality()
        write_vars(self.m)

        total_CPUs = 0
        used_cores = 0
        switch_memory = 0
        for d in self.devices:
            if(isinstance(d, CPU)):
                used_cores += d.cores_sketch.x + d.cores_dpdk.x
                total_CPUs += 1
            if(isinstance(d, P4)):
                switch_memory += d.mem_tot.x

        if('ns_req' in self.__dict__):
            self.ns = self.ns_req
        log_results(self.ns, self.res, used_cores, total_CPUs, switch_memory)
        log_placement(self.devices, self.queries, self.flows, self.partitions,
                      self.m, self.frac, self.dev_par_tuplelist)


def log_results(ns, res, used_cores=None, total_CPUs=None, switch_memory=None):
    log.info("\nThroughput: {} Mpps, ns per packet: {}".format(
        1000/get_val(ns), get_val(ns)))
    log.info("Resources: {}".format(get_val(res)))
    if(total_CPUs is not None and used_cores is not None
       and switch_memory is not None):
        log.info("Used Cores: {}, Total CPUS: {}, Switch Memory: {}"
                 .format(used_cores, total_CPUs, switch_memory))

    if(not (common_config.output_file is None)):
        f = open(common_config.output_file, 'a')
        if(used_cores is not None and total_CPUs is not None
           and switch_memory is not None):
            f.write("{:0.3f}, {:0.3f}, {}, {}, {:0.3f}, ".format(
                1000/get_val(ns), get_val(res),
                used_cores, total_CPUs, switch_memory))
        else:
            f.write("{:0.3f}, {:0.3f}, ".format(
                1000/get_val(ns), get_val(res)))
        f.close()


def log_placement(devices, queries, flows, partitions, m, frac,
                  dev_par_tuplelist):
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
            for (dnum, _) in dev_par_tuplelist.select('*', pnum):
                par_info += "({:0.3f},{})    ".format(frac[dnum, pnum].x, dnum)
                total_frac += (frac[dnum, pnum].x)
            # for (dnum, d) in enumerate(devices):
            #     par_info += "{:0.3f}    ".format(frac[dnum, pnum].x)
            #     total_frac += (frac[dnum, pnum].x)
            log.info(par_info)
            log.info("Total frac: {:0.3f}".format(total_frac))

    for (dnum, d) in enumerate(devices):
        log.info("\nDevice ({}) {}:".format(dnum, d))
        res_stats = d.resource_stats()
        if(res_stats != ""):
            log.info(res_stats)
        log.info("Rows total: {}".format(get_val(d.rows_tot)))
        log.info("Mem total: {}".format(d.mem_tot.x))
        if(hasattr(d, 'ns')):
            log.info("Throughput: {}".format(1000/d.ns.x))

    log.debug("")
    for (fnum, f) in enumerate(flows):
        log.debug("Flow {}:".format(fnum))
        log.debug("queries: {}".format(f.queries))
        log.debug("path: {}".format(f.path))
    log.info("-"*50)


solver_names = ['Univmon', 'UnivmonGreedy', 'UnivmonGreedyRows', 'Netmon']
solver_list = [getattr(sys.modules[__name__], s) for s in solver_names]
solver_to_num = {}
solver_to_class = {}
for (solver_num, solver) in enumerate(solver_list):
    solver_to_num[solver.__name__] = solver_num
    solver_to_class[solver.__name__] = solver

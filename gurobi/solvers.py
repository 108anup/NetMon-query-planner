import re
import sys

import gurobipy as gp
from gurobipy import GRB

from common import log, param
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
    res = m.addVar(vtype=GRB.CONTINUOUS, name='res')
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

        total_cpus = 0
        used_cores = 0
        switch_memory = 0
        for d in self.devices:
            if(isinstance(d, cpu)):
                used_cores += d.cores_sketch.x + d.cores_dpdk.x
                total_cpus += 1
            if(isinstance(d, p4)):
                switch_memory += d.mem_tot.x

        log_results(self.ns, self.res, used_cores, total_cpus, switch_memory)
        log_placement(self.devices, self.queries, self.flows, self.partitions,
                      self.m, self.frac)


class univmon(param):
    def __init__(self, *args, **kwargs):
        super(univmon, self).__init__(*args, **kwargs)

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
                      self.m, self.frac)

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
            total_cpus = 0
            switch_memory = 0
            for d in self.devices:
                u = d.u
                if(isinstance(d, cpu)):
                    u.addConstr(d.ns >= ns_max, name='ns_req_{}'.format(d))

                    u.update()
                    u.optimize()
                    write_vars(u)

                    total_cpus += 1
                    used_cores += d.cores_sketch.x + d.cores_dpdk.x

                if(isinstance(d, p4)):
                    switch_memory += d.mem_tot.x

                ns_max = max(ns_max, u.getObjective(0).getValue())
                res_acc += u.getObjective(1).getValue()

            log_results(ns_max, res_acc, used_cores, total_cpus,
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
        if(len(mem_series_p4) > 0):
            self.max_mem_p4 = self.m.addVar(vtype=GRB.CONTINUOUS,
                                            name='max_mem_p4')
            self.m.addGenConstrMax(self.max_mem_p4, mem_series_p4,
                                   name='mem_overall_p4')
            self.tot_mem_p4 = gp.quicksum(mem_series_p4)
        if(len(mem_series_cpu) > 0):
            self.max_mem_cpu = self.m.addVar(vtype=GRB.CONTINUOUS,
                                             name='max_mem_cpu')
            self.m.addGenConstrMax(self.max_mem_cpu, mem_series_cpu,
                                   name='mem_overall_cpu')
            self.tot_mem_cpu = gp.quicksum(mem_series_cpu)
        self.tot_mem = self.m.addVar(vtype=GRB.CONTINUOUS,
                                     name='tot_mem')
        self.m.addConstr(self.tot_mem == gp.quicksum(mem_series_cpu)
                         + gp.quicksum(mem_series_p4), name='tot_mem')

    def add_objective(self):
        if(hasattr(self, 'max_mem_cpu')):
            self.m.setObjectiveN(self.tot_mem_cpu, 0, 20, name='tot_mem_cpu')
            self.m.setObjectiveN(self.max_mem_cpu, 1, 15, name='cpu_mem_load')
        if(hasattr(self, 'max_mem_p4')):
            self.m.setObjectiveN(self.tot_mem_p4, 2, 10, name='tot_mem_p4')
            self.m.setObjectiveN(self.max_mem_p4, 3, 5, name='p4_mem_load')
        # self.m.setObjectiveN(self.tot_mem, 2, 1, name='mem_load')

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
        if(len(rows_series_p4) > 0):
            self.max_rows_p4 = self.m.addVar(vtype=GRB.CONTINUOUS,
                                             name='max_rows_p4')
            self.m.addGenConstrMax(self.max_rows_p4, rows_series_p4,
                                   name='rows_overall_p4')
            self.tot_rows_p4 = gp.quicksum(rows_series_p4)
        if(len(rows_series_cpu) > 0):
            self.max_rows_cpu = self.m.addVar(vtype=GRB.CONTINUOUS,
                                              name='max_rows_cpu')
            self.m.addGenConstrMax(self.max_rows_cpu, rows_series_cpu,
                                   name='rows_overall_cpu')
            self.tot_rows_cpu = gp.quicksum(rows_series_cpu)
        self.tot_rows = self.m.addVar(vtype=GRB.CONTINUOUS,
                                      name='tot_rows')
        self.m.addConstr(self.tot_rows == gp.quicksum(rows_series_cpu)
                         + gp.quicksum(rows_series_p4), name='tot_rows')

    def add_objective(self):
        if(hasattr(self, 'max_rows_cpu')):
            self.m.setObjectiveN(self.tot_rows_cpu, 0, 100, name='tot_rows_cpu')
            self.m.setObjectiveN(self.max_rows_cpu, 1, 90, name='cpu_rows_load')
        if(hasattr(self, 'max_rows_p4')):
            self.m.setObjectiveN(self.tot_rows_p4, 2, 80, name='tot_rows_p4')
            self.m.setObjectiveN(self.max_rows_p4, 3, 70, name='p4_rows_load')
        # self.m.setObjectiveN(self.tot_rows, 2, 20, name='rows_load')
        if(hasattr(self, 'max_mem_cpu')):
            self.m.setObjectiveN(self.tot_mem_cpu, 4, 60, name='tot_mem_cpu')
            self.m.setObjectiveN(self.max_mem_cpu, 5, 50, name='cpu_load_mem')
        if(hasattr(self, 'max_mem_p4')):
            self.m.setObjectiveN(self.tot_mem_p4, 6, 40, name='tot_mem_p4')
            self.m.setObjectiveN(self.max_mem_p4, 7, 30, name='p4_load_mem')
        # self.m.setObjectiveN(self.tot_mem, 5, 5, name='mem_load')


def log_results(ns, res, used_cores=None, total_cpus=None, switch_memory=None):
    log.info("\nThroughput: {} Mpps, ns per packet: {}".format(
        1000/get_val(ns), get_val(ns)))
    log.info("Resources: {}".format(get_val(res)))
    if(total_cpus is not None and used_cores is not None
       and switch_memory is not None):
        log.info("Used Cores: {}, Total CPUS: {}, Switch Memory: {}"
                 .format(used_cores, total_cpus, switch_memory))

    if(not (common_config.output_file is None)):
        f = open(common_config.output_file, 'a')
        if(used_cores is not None and total_cpus is not None
           and switch_memory is not None):
            f.write("{:0.3f}, {:0.3f}, {}, {}, {:0.3f}, ".format(
                1000/get_val(ns), get_val(res),
                used_cores, total_cpus, switch_memory))
        else:
            f.write("{:0.3f}, {:0.3f}, ".format(
                1000/get_val(ns), get_val(res)))
        f.close()


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


solver_names = ['univmon', 'univmon_greedy', 'univmon_greedy_rows', 'netmon']
solver_list = [getattr(sys.modules[__name__], s) for s in solver_names]
solver_to_num = {}
solver_to_class = {}
for (solver_num, solver) in enumerate(solver_list):
    solver_to_num[solver.__name__] = solver_num
    solver_to_class[solver.__name__] = solver

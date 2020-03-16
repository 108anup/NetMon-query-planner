import os
import re
import sys
import time

import gurobipy as gp
from gurobipy import GRB, tuplelist

from common import log, Namespace, log_time
from config import common_config
from devices import CPU, P4, Cluster

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

'''
Tricks performed:
1. Remove Ceiling
2. Make variables continuous (remove binary and integer variables)
3. Log -INFINITY -> removed
4. Allow non convex problem
5. Use UnivmonGreedyRows output as start
6. Convert ns to constraint

NOTES:
1. With logarithmic constraints, if I make variables integral it seems to
perform better, as long as those vars are not involved in other constraints.
2. We want log to be able to take negative values to allow variables
to take value 0 but some problem take a ton of time to solve in those
scenarios.
Above are not relevant any more
'''


def get_rounded_val(v):
    if(v < 0):
        return 0
    if(v < common_config.ftol):
        return 0
    return v


def get_val(v):
    if(isinstance(v, (float, int))):
        return v
    if(isinstance(v, gp.Var)):
        return v.x
    else:
        return v.getValue()


def write_vars(m):
    log.debug("\nVARIABLES "+"-"*30)
    log.debug("Objective: {}".format(m.objVal))
    for v in m.getVars():
        log.debug('%s %g' % (v.varName, v.x))
    log.debug("-"*50)


class MIP(Namespace):

    def compute_dev_par_tuplelist(self):
        dev_par_tuplelist = []
        for (fnum, f) in enumerate(self.flows):
            for p in f.partitions:
                pnum = p[0]
                dev_par_tuplelist.extend([(dnum, pnum) for dnum in f.path])
        self.dev_par_tuplelist = tuplelist(set(dev_par_tuplelist))

    @log_time
    def add_frac_mem_var(self):
        # Fraction of partition on device
        if(common_config.vertical_partition):
            self.frac = self.m.addVars(
                self.dev_par_tuplelist, vtype=GRB.CONTINUOUS,
                lb=0, ub=1, name='frac')
        else:
            self.frac = self.m.addVars(
                self.dev_par_tuplelist, vtype=GRB.BINARY,
                name='frac')

        # Memory taken by partition
        self.mem = self.m.addVars(
            self.dev_par_tuplelist, vtype=GRB.CONTINUOUS,
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

        # No memory if not mapped
        # Frac == 0 -> mem == 0
        # m.addConstrs(((frac[i, j] == 0) >> (mem[i, j] == 0)
        #               for i in range(numdevices)
        #               for j in range(numpartitions)), name='frac_mem')

    @log_time
    def add_coverage_constraints(self):

        # for pnum in range(numpartitions):
        #     m.addConstr(frac.sum('*', pnum) == 1,
        #                 name='cov_{}'.format(pnum))

        # keep = np.zeros((numdevices, numpartitions))
        for (fnum, f) in enumerate(self.flows):
            for p in f.partitions:
                pnum = p[0]
                coverage_requirement = p[1]
                sum_expr = gp.quicksum(self.frac[dnum, pnum]
                                       for dnum in f.path)
                self.m.addConstr(sum_expr >= coverage_requirement,
                                 name='cov_{}_{}'.format(fnum, pnum))
                # for dnum in f.path:
                #     keep[dnum][pnum] = 1

        # numdevices = len(self.devices)
        # numpartitions = len(self.partitions)
        # for dnum in range(numdevices):
        #     for pnum in range(numpartitions):
        #         if(keep[dnum][pnum] == 0):
        #             self.m.addConstr(self.frac[dnum, pnum] == 0,
        #                              name='frac_not_reqd_{}_{}'
        #                              .format(dnum, pnum))
        #             self.m.addConstr(self.mem[dnum, pnum] == 0,
        #                         name='mem_not_reqd_{}_{}'.format(dnum, pnum))

    @log_time
    def add_accuracy_constraints(self):

        for (dnum, pnum) in self.dev_par_tuplelist:
            p = self.partitions[pnum]
            sk = p.sketch
            num_rows = p.num_rows
            mm = sk.min_mem()
            self.m.addConstr(self.mem[dnum, pnum] ==
                             mm * self.frac[dnum, pnum] * num_rows,
                             name='accuracy_{}_{}'.format(dnum, pnum))

        # numdevices = len(self.devices)
        # for (pnum, p) in enumerate(self.partitions):
        #     sk = p.sketch
        #     num_rows = p.num_rows
        #     mm = sk.min_mem()
        #     self.m.addConstrs((self.mem[dnum, pnum] ==
        #                        mm * self.frac[dnum, pnum] * num_rows
        #                        for dnum in range(numdevices)),
        #                       name='accuracy_{}_{}'.format(pnum, sk))

    @log_time
    def add_capacity_constraints(self):

        for (dnum, d) in enumerate(self.devices):
            # Memory constraints
            # Capacity constraints included in bounds
            mem_tot = self.m.addVar(vtype=GRB.CONTINUOUS,
                                    name='mem_tot_{}'.format(d),
                                    lb=0, ub=d.max_mem)
            self.m.addConstr(mem_tot == self.mem.sum(dnum, '*'),
                             name='mem_tot_{}'.format(d))
            d.mem_tot = mem_tot

            # Row constraints
            rows_tot = self.m.addVar(vtype=GRB.CONTINUOUS,
                                     name='rows_tot_{}'.format(d), lb=0)
            rows_series = []
            for (_, pnum) in self.dev_par_tuplelist.select(dnum, '*'):
                p = self.partitions[pnum]
                rows_series.append(
                    p.num_rows * self.frac[dnum, pnum]
                )
            # rows_series = [p.num_rows * frac[dnum, p.partition_id]
            #                for p in self.partitions]
            self.m.addConstr(rows_tot == gp.quicksum(rows_series),
                             name='rows_tot_{}'.format(d))
            d.rows_tot = rows_tot

    def add_device_aware_constraints(self):
        for (dnum, d) in enumerate(self.devices):
            self.m.addConstr(d.rows_tot <= d.max_rows,
                             name='row_capacity_{}'.format(d))

            if hasattr(d, 'max_mpp'):
                for (pnum, p) in enumerate(self.partitions):
                    self.m.addConstr(
                        self.mem[dnum, pnum] <= d.max_mpr * p.num_rows,
                        'capacity_mem_par_{}'.format(d))

    def add_constraints(self):
        pass

    def add_objective(self):
        pass

    def post_optimize(self):
        pass

    def check_device_aware_constraints(self):
        for (dnum, d) in enumerate(self.devices):
            if not (d.rows_tot.x <= d.max_rows):
                return False
            if hasattr(d, 'max_mpp'):
                for (_, pnum) in self.dev_par_tuplelist.select(dnum, '*'):
                    p = self.partitions[pnum]
                    if not (self.mem[dnum, pnum].x <= d.max_mpr * p.num_rows):
                        return False
        return True

    def add_device_model_constraints(self, ns_req=None):
        res_acc = gp.LinExpr()
        for d in self.devices:
            # Throughput
            # Simple total model
            d.add_ns_constraints(self.m, ns_req)

            # Resources
            res_acc += d.res()
            d.m = self.m

        ns = None
        if(ns_req is None):
            ns_series = [d.ns for d in self.devices]
            ns = self.m.addVar(vtype=GRB.CONTINUOUS, name='ns')
            self.m.addGenConstrMax(ns, ns_series, name='ns_overall')
        res = self.m.addVar(vtype=GRB.CONTINUOUS, name='res')
        self.m.addConstr(res_acc == res, name='res_acc')
        return (ns, res)

    def initialize(self):
        for (dnum, pnum) in self.dev_par_tuplelist:
            self.mem[dnum, pnum].start = self.init.mem[dnum, pnum]
            self.frac[dnum, pnum].start = self.init.frac[dnum, pnum]

    @log_time
    def solve(self):
        log.info("Building model with:\n"
                 "{} devices, {} partitions and {} flows"
                 .format(len(self.devices), len(self.partitions),
                         len(self.flows)))

        self.m = gp.Model(self.__class__.__name__)
        if(not common_config.mipout):
            self.m.setParam(GRB.Param.LogToConsole, 0)

        self.compute_dev_par_tuplelist()
        self.add_frac_mem_var()
        self.add_coverage_constraints()
        self.add_accuracy_constraints()
        self.add_capacity_constraints()
        if(not type(self).__name__ == 'Univmon'):
            self.add_device_aware_constraints()

        if(common_config.time_limit):
            self.m.setParam(GRB.Param.TimeLimit, common_config.time_limit)
        # m.setParam(GRB.Param.MIPGapAbs, common_config.mipgapabs)
        # m.setParam(GRB.Param.MIPGap, common_config.mipgap)

        if(hasattr(self, 'init')):
            self.initialize()
        self.add_constraints()
        self.m.ModelSense = GRB.MINIMIZE
        self.add_objective()

        log.info("-"*50)
        log.info("Starting model update")
        start = time.time()
        self.m.update()
        end = time.time()
        update_time = end - start
        log.info("Model update took: {} s".format(update_time))
        log.info("-"*50)
        if(common_config.prog_dir):
            self.m.write(os.path.join(
                common_config.prog_dir,
                "prog_{}.lp".format(common_config.input_num)))

        log.info("Starting model optimize")
        start = time.time()
        self.m.optimize()
        end = time.time()
        log.info("Model optimize took: {} seconds".format(end - start))
        log.info("-"*50)

        if(self.m.Status == GRB.INFEASIBLE):
            if(common_config.prog_dir):
                self.m.computeIIS()
                self.m.write(
                    os.path.join(
                        common_config.prog_dir,
                        "infeasible_{}.ilp".format(common_config.input_num)
                    )
                )
            if(not (common_config.output_file is None)):
                f = open(common_config.output_file, 'a')
                f.write("-, -, -, -, -, ")
                f.close()
        else:
            self.post_optimize()

        end = time.time()


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

    def post_optimize(self):
        self.m.printQuality()
        write_vars(self.m)

        log_placement(self.devices, self.partitions, self.flows,
                      self.dev_par_tuplelist, self.frac)

        if(not common_config.use_model):
            if (not self.check_device_aware_constraints()):
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
                # u.addConstr(rows_tot == d.rows_tot.x,
                #             name='rows_tot_{}'.format(d))
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
                    u.addConstr(d.ns >= ns_max,
                                name='global_ns_req_{}'.format(d))
                    u.update()
                    u.optimize()
                    write_vars(u)

                    total_CPUs += 1
                    used_cores += d.cores_sketch.x + d.cores_dpdk.x

                if(isinstance(d, P4)):
                    switch_memory += d.mem_tot.x

                ns_max = max(ns_max, u.getObjective(0).getValue())
                res_acc += u.getObjective(1).getValue()

            log_results(self.devices, self.overlay)

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
            if(type(self).__name__ == 'Univmon'):
                self.add_device_aware_constraints(
                    self.devices, self.queries, self.flows,
                    self.partitions, self.m, self.frac, self.mem)

            (ns, res) = self.add_device_model_constraints()
            self.m.NumObj = 2
            self.m.setObjectiveN(ns, 0, 10, reltol=common_config.ns_tol,
                                 name='ns')
            self.m.setObjectiveN(res, 1, 5, reltol=common_config.res_tol,
                                 name='res')
            self.m.update()
            self.m.optimize()

            if(self.m.Status == GRB.INFEASIBLE):
                self.m.computeIIS()
                if(common_config.prog_dir):
                    self.m.write(os.path.join(
                        common_config.prog_dir, "infeasible_placement_{}.ilp"
                        "".format(common_config.cfg_num)))
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

            log_results(self.devices, self.overlay)

        log_placement(self.devices, self.partitions, self.flows,
                      self.dev_par_tuplelist, self.frac)


# TODO:: Add loop for these objectives instead of per device type variables
class UnivmonGreedy(Univmon):

    def add_constraints(self):
        mem_series_P4 = [d.mem_tot
                         for d in self.devices
                         if isinstance(d, P4)]
        mem_series_CPU = [d.mem_tot
                          for d in self.devices
                          if isinstance(d, CPU)]
        mem_series_Cluster = [d.mem_tot
                              for d in self.devices
                              if isinstance(d, Cluster)]
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
        if(len(mem_series_Cluster) > 0):
            self.max_mem_Cluster = self.m.addVar(vtype=GRB.CONTINUOUS,
                                                 name='max_mem_Cluster')
            self.m.addGenConstrMax(self.max_mem_Cluster, mem_series_Cluster,
                                   name='mem_overall_Cluster')
            self.tot_mem_Cluster = gp.quicksum(mem_series_Cluster)
        self.tot_mem = self.m.addVar(vtype=GRB.CONTINUOUS,
                                     name='tot_mem')
        self.m.addConstr(self.tot_mem == gp.quicksum(mem_series_CPU)
                         + gp.quicksum(mem_series_Cluster)
                         + gp.quicksum(mem_series_P4), name='tot_mem')

    def add_objective(self):
        if(hasattr(self, 'max_mem_CPU')):
            self.m.setObjectiveN(self.tot_mem_CPU, 0, 20, name='tot_mem_CPU')
            self.m.setObjectiveN(self.max_mem_CPU, 1, 15, name='CPU_mem_load')
        if(hasattr(self, 'max_mem_Cluster')):
            self.m.setObjectiveN(self.tot_mem_Cluster, 2, 10,
                                 name='tot_mem_Cluster')
            self.m.setObjectiveN(self.max_mem_Cluster, 3, 5,
                                 name='Cluster_mem_load')
        if(hasattr(self, 'max_mem_P4')):
            self.m.setObjectiveN(self.tot_mem_P4, 4, 10, name='tot_mem_P4')
            self.m.setObjectiveN(self.max_mem_P4, 5, 5, name='P4_mem_load')
        # self.m.setObjectiveN(self.tot_mem, 2, 1, name='mem_load')


class UnivmonGreedyRows(UnivmonGreedy):

    def add_constraints(self):
        super(UnivmonGreedyRows, self).add_constraints()
        rows_series_P4 = [d.rows_tot
                          for d in self.devices
                          if isinstance(d, P4)]
        rows_series_CPU = [d.rows_tot
                           for d in self.devices
                           if isinstance(d, CPU)]
        rows_series_Cluster = [d.rows_tot
                               for d in self.devices
                               if isinstance(d, Cluster)]
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
        if(len(rows_series_Cluster) > 0):
            self.max_rows_Cluster = self.m.addVar(vtype=GRB.CONTINUOUS,
                                                  name='max_rows_Cluster')
            self.m.addGenConstrMax(self.max_rows_Cluster, rows_series_Cluster,
                                   name='rows_overall_Cluster')
            self.tot_rows_Cluster = gp.quicksum(rows_series_Cluster)
        self.tot_rows = self.m.addVar(vtype=GRB.CONTINUOUS,
                                      name='tot_rows')
        self.m.addConstr(self.tot_rows == gp.quicksum(rows_series_CPU)
                         + gp.quicksum(rows_series_P4)
                         + gp.quicksum(rows_series_Cluster), name='tot_rows')

    def add_objective(self):
        if(hasattr(self, 'max_rows_CPU')):
            self.m.setObjectiveN(self.tot_rows_CPU, 0, 100,
                                 name='tot_rows_CPU')
            self.m.setObjectiveN(self.max_rows_CPU, 1, 90,
                                 name='CPU_rows_load')
        if(hasattr(self, 'max_rows_Cluster')):
            self.m.setObjectiveN(self.tot_rows_Cluster, 2, 100,
                                 name='tot_rows_Cluster')
            self.m.setObjectiveN(self.max_rows_Cluster, 3, 90,
                                 name='Cluster_rows_load')
        if(hasattr(self, 'max_rows_P4')):
            self.m.setObjectiveN(self.tot_rows_P4, 4, 80, name='tot_rows_P4')
            self.m.setObjectiveN(self.max_rows_P4, 5, 70, name='P4_rows_load')
        # self.m.setObjectiveN(self.tot_rows, 2, 20, name='rows_load')
        if(hasattr(self, 'max_mem_CPU')):
            self.m.setObjectiveN(self.tot_mem_CPU, 6, 60, name='tot_mem_CPU')
            self.m.setObjectiveN(self.max_mem_CPU, 7, 50, name='CPU_load_mem')
        if(hasattr(self, 'max_mem_Cluster')):
            self.m.setObjectiveN(self.tot_mem_Cluster, 8, 60,
                                 name='tot_mem_Cluster')
            self.m.setObjectiveN(self.max_mem_Cluster, 9, 50,
                                 name='Cluster_load_mem')
        if(hasattr(self, 'max_mem_P4')):
            self.m.setObjectiveN(self.tot_mem_P4, 10, 40, name='tot_mem_P4')
            self.m.setObjectiveN(self.max_mem_P4, 11, 30, name='P4_load_mem')
        # self.m.setObjectiveN(self.tot_mem, 5, 5, name='mem_load')


class Netmon(UnivmonGreedyRows):

    def add_constraints(self):

        # Initialize with unimon_greedy_rows solution
        super(Netmon, self).add_constraints()
        super(Netmon, self).add_objective()
        self.m.update()
        self.m.optimize()

        # numdevices = len(self.devices)
        # numpartitions = len(self.partitions)

        # for dnum in range(numdevices):
        #     for pnum in range(numpartitions):
        for (dnum, pnum) in self.dev_par_tuplelist:
            self.frac[dnum, pnum].start = self.frac[dnum, pnum].x
            self.mem[dnum, pnum].start = self.mem[dnum, pnum].x

        # self.ns_req = 14.3
        (self.ns, self.res) = self.add_device_model_constraints()

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

        if(hasattr(self, 'ns_req')):
            self.ns = self.ns_req
        log_results(self.devices, self.overlay)
        log_placement(self.devices, self.partitions, self.flows,
                      self.dev_par_tuplelist, self.frac)


def log_results(devices, overlay=False):

    ns_max = 0
    res = 0
    total_CPUs = 0
    used_cores = 0
    switch_memory = 0

    for d in devices:
        ns_max = max(ns_max, get_val(d.ns))
        res += get_val(d.res())
        if(isinstance(d, CPU)):
            total_CPUs += 1
            used_cores += d.cores_sketch.x + d.cores_dpdk.x
        if(isinstance(d, P4)):
            switch_memory += d.mem_tot.x

    log.info("\nThroughput: {} Mpps, ns per packet: {}".format(
        1000/ns_max, ns_max))
    log.info("Resources: {}".format(res))
    if(total_CPUs is not None and used_cores is not None
       and switch_memory is not None):
        log.info("Used Cores: {}, Total CPUS: {}, Switch Memory: {}"
                 .format(used_cores, total_CPUs, switch_memory))

    if((not (common_config.output_file is None)) and not overlay):
        f = open(common_config.output_file, 'a')
        f.write("{:0.3f}, {:0.3f}, {}, {}, {:0.3f}, ".format(
            1000/ns_max, res,
            used_cores, total_CPUs, switch_memory))
        f.close()


def log_placement(devices, partitions, flows, dev_par_tuplelist, frac):

    # for (qnum, q) in enumerate(queries):
    #     log.info("\nSketch ({}) ({})".format(q.sketch_id,
    #                                          q.details()))
    #     row = 1
    #     for pnum in q.partitions:
    #         num_rows = partitions[pnum].num_rows
    #         log.info("Par: {}, Rows: {}".format(row, num_rows))
    #         row += 1
    #         par_info = ""
    #         total_frac = 0
    #         for (dnum, _) in dev_par_tuplelist.select('*', pnum):
    #             par_info += "({:0.3f},{})    ".format(frac[dnum, pnum].x, dnum)
    #             total_frac += (frac[dnum, pnum].x)
    #         # for (dnum, d) in enumerate(devices):
    #         #     par_info += "{:0.3f}    ".format(frac[dnum, pnum].x)
    #         #     total_frac += (frac[dnum, pnum].x)
    #         log.info(par_info)
    #         log.info("Total frac: {:0.3f}".format(total_frac))

    prev_q_id = None
    for (pnum, p) in enumerate(partitions):
        q = p.sketch
        if(q.sketch_id != prev_q_id):
            log.info("\nSketch ({}) ({})"
                     .format(q.sketch_id, q.details()))
            prev_q_id = q.sketch_id

        log.info("Par{} id: {}, Rows: {}"
                 .format(p.partition_id - q.partitions[0],
                         p.partition_id, p.num_rows))
        par_info = ""
        total_frac = 0
        for (dnum, _) in dev_par_tuplelist.select('*', pnum):
            par_info += "({:0.3f},{})    ".format(
                get_val(frac[dnum, pnum]), dnum)
            total_frac += (get_val(frac[dnum, pnum]))
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
        log.debug("partitions: {}".format(f.partitions))
        log.debug("path: {}".format(f.path))
    log.info("-"*50)


solver_names = ['Univmon', 'UnivmonGreedy', 'UnivmonGreedyRows', 'Netmon']
solver_list = [getattr(sys.modules[__name__], s) for s in solver_names]
solver_to_num = {}  # unused
solver_to_class = {}
for (solver_num, solver) in enumerate(solver_list):
    solver_to_num[solver.__name__] = solver_num
    solver_to_class[solver.__name__] = solver

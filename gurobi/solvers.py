import logging
import os
import re
import sys
import time
from functools import partial
import math

import gurobipy as gp
from gurobipy import GRB, tupledict, tuplelist

from common import Namespace, log, log_time, memoize
from config import common_config
from devices import CPU, P4, Cluster
from helpers import get_rounded_val, get_val


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


def log_vars(m, logger=partial(log.log, logging.DEBUG-1)):
    logger("\nVARIABLES "+"-"*30)
    logger("Objective: {}".format(m.objVal))
    for v in m.getVars():
        logger('%s %g' % (v.varName, v.x))


@log_time(logger=log.info)
def refine_devices(devices):
    res_acc = 0
    ns_max = 0
    for d in devices:

        if(not hasattr(d, 'mem_tot')):
            d.mem_tot = 0
        if(not hasattr(d, 'rows_tot')):
            d.rows_tot = 0

        # TODO: Memoize for faster solution!
        # Also consider memoizing cluster solutions

        # memo1 = tupledict()
        # if((get_val(d.rows_tot), get_val(d.mem_tot)) in memo1):
        #     for d in devices:

        # TODO: Alternate approach for removing side effects only
        # when necessary

        u = gp.Model(d.name)
        if(not common_config.mipout):
            u.setParam(GRB.Param.LogToConsole, 0)
        mem_tot = u.addVar(vtype=GRB.CONTINUOUS,
                           name='mem_tot_{}'.format(d),
                           lb=0, ub=d.max_mem)
        u.addConstr(mem_tot == get_rounded_val(get_val(d.mem_tot)),
                    name='mem_tot_{}'.format(d))
        d.mem_tot_old = d.mem_tot
        d.mem_tot = mem_tot

        # rows_tot = u.addVar(vtype=GRB.CONTINUOUS,
        #                     name='rows_tot_{}'.format(d), lb=0)
        # u.addConstr(rows_tot == d.rows_tot.x,
        #             name='rows_tot_{}'.format(d))
        d.rows_tot_old = d.rows_tot
        d.rows_tot = get_rounded_val(get_val(d.rows_tot))
        d.add_ns_constraints(u)

        u.setObjectiveN(d.ns, 0, 10, reltol=common_config.ns_tol,
                        name='ns')
        u.setObjectiveN(d.res(), 1, 5, reltol=common_config.res_tol,
                        name='res')

        u.ModelSense = GRB.MINIMIZE
        u.setParam(GRB.Param.LogToConsole, 0)
        u.update()
        u.optimize()
        # Need to keep these to retain model values.
        if(hasattr(d, 'u')):
            if(hasattr(d, 'old_u')):
                d.old_u.append(d.u)
            else:
                d.old_u = [d.u]
        d.u = u

        log_vars(u)

        ns_max = max(ns_max, u.getObjective(0).getValue())

    res_acc = 0
    for d in devices:
        u = d.u
        if(isinstance(d, CPU)):
            u.addConstr(d.ns >= ns_max,
                        name='global_ns_req_{}'.format(d))
            u.update()
            u.optimize()
            log_vars(u)

        d.mem_tot = d.mem_tot_old
        d.rows_tot = d.rows_tot_old
        ns_max = max(ns_max, u.getObjective(0).getValue())
        res_acc += u.getObjective(1).getValue()

    return (ns_max, res_acc)


class MIP(Namespace):

    def __init__(self, **kwargs):
        super(MIP, self).__init__(**kwargs)
        self.infeasible = False

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
            if isinstance(self.devices[dnum], P4):
                mm = 2 ** math.ceil(math.log2(mm))
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
            normalized_mem_tot = self.m.addVar(
                vtype=GRB.CONTINUOUS, name='normalized_mem_tot_{}'.format(d),
                lb=0, ub=1)
            self.m.addConstr(normalized_mem_tot == mem_tot / d.max_mem)
            d.normalized_mem_tot = normalized_mem_tot

            # Row constraints
            rows_tot = self.m.addVar(vtype=GRB.CONTINUOUS,
                                     name='rows_tot_{}'.format(d), lb=0)
            normalized_rows_tot = self.m.addVar(
                vtype=GRB.CONTINUOUS, name='normalized_rows_tot_{}'.format(d),
                lb=0, ub=1)
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
            self.m.addConstr(normalized_rows_tot == rows_tot / d.max_rows)
            d.normalized_rows_tot = normalized_rows_tot
            d.m = self.m

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

        # log_initial(self.devices, self.partitions, self.flows,
        #             self.dev_par_tuplelist, self.init)

    @log_time
    def solve(self):
        self.m = gp.Model(self.__class__.__name__)
        log.info("\n" + "-"*80)
        log.info("Model {} with:\n"
                 "{} devices, {} partitions and {} flows"
                 .format(type(self).__name__, len(self.devices),
                         len(self.partitions), len(self.flows)))

        if(not common_config.mipout):
            self.m.setParam(GRB.Param.LogToConsole, 0)
            # gplogger = logging.getLogger('gurobipy')
            # gplogger.disabled = True

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
        if(self.infeasible):
            return

        self.m.ModelSense = GRB.MINIMIZE
        self.add_objective()

        start = time.time()
        self.m.update()
        end = time.time()
        update_time = end - start
        log.info("Model update took: {} s".format(update_time))

        # log.info('='*50)
        # log.info("Beginning tuning")
        # log.info('='*50)
        # self.m.setParam(GRB.Param.TuneOutput, 1)
        # self.m.tune()
        # log.info('='*50)
        # log.info("End tuning")
        # log.info('='*50)
        # log.info("Parameters sets obtained: {}".format(self.m.TuneResultCount))
        # for i in range(self.m.TuneResultCount):
        #     self.m.getTuneResult(i)
        #     self.m.write(os.path.join(common_config.prog_dir,
        #                               'tune'+str(i)+'.prm'))
        # self.m.setParam(GRB.Param.Heuristics, 0.5)
        # self.m.setParam(GRB.Param.PrePasses, 8)

        if(common_config.prog_dir):
            self.m.write(os.path.join(
                common_config.prog_dir,
                "prog_{}.lp".format(common_config.input_num)))

        log.info("Starting model optimize")
        start = time.time()
        self.m.optimize()
        end = time.time()
        log.info("-"*50)
        log.info("Model optimize took: {} seconds".format(end - start))
        log.info("-"*50)

        if(self.m.Status == GRB.INFEASIBLE):
            self.infeasible = True
            self.culprit = self.m
            return
        else:
            log_objectives(self.m)
            self.post_optimize()
            log_placement(self.devices, self.partitions, self.flows,
                          self.dev_par_tuplelist, self.frac,
                          init=getattr(self, 'init', None))
            # Basically if Netmon was run in intermediate and it did not
            # behave like Univmon Greedy Rows
            if(hasattr(self, 'refined')):
                if(self.refined and self.dont_refine):
                    log_results(self.devices, msg="Netmon Intermediate Result")


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
        log_vars(self.m)

        # log_placement(self.devices, self.partitions, self.flows,
        #               self.dev_par_tuplelist, self.frac)

        if(not common_config.use_model):
            if(type(self).__name__ == 'Univmon'):
                if (not self.check_device_aware_constraints()):
                    log.warning("Infeasible placement due to Univmon's "
                             "lack of knowledge")
                    self.infeasible = True
                    self.culprit = self.m
                    return

            if(not self.dont_refine):
                (self.ns_max, self.res) = refine_devices(self.devices)
                self.refined = True

        # else:
        #     prefixes = ['frac', 'mem\[']
        #     regex = '|'.join(prefixes)
        #     prog = re.compile(regex)
        #     for v in self.m.getVars():
        #         if (prog.match(v.varName)):
        #             # if((v.x) < 0):
        #             #     self.m.addConstr(v == 0)
        #             # else:
        #             #     self.m.addConstr(v == v.x)
        #             self.m.addConstr(v == get_rounded_val(v.x))

        #     self.m.setParam(GRB.Param.FeasibilityTol, common_config.ftol)
        #     if(type(self).__name__ == 'Univmon'):
        #         self.add_device_aware_constraints(
        #             self.devices, self.queries, self.flows,
        #             self.partitions, self.m, self.frac, self.mem)

        #     (ns, res) = self.add_device_model_constraints()
        #     self.m.NumObj = 2
        #     self.m.setObjectiveN(ns, 0, 10, reltol=common_config.ns_tol,
        #                          name='ns')
        #     self.m.setObjectiveN(res, 1, 5, reltol=common_config.res_tol,
        #                          name='res')
        #     self.m.update()
        #     self.m.optimize()

        #     if(self.m.Status == GRB.INFEASIBLE):
        #         return handle_infeasible_iis(self.m)

        #     # optimal_ns = ns.x * (1 + common_config.ns_tol)
        #     # self.m.addConstr(ns <= optimal_ns)
        #     # for d in self.devices:
        #     #     self.m.addConstr(d.ns <= optimal_ns)
        #     # self.m.setObjectiveN(res, 0, 10, reltol=common_config.res_tol,
        #     #                      name='res')
        #     # self.m.update()
        #     # self.m.optimize()

        #     # if(self.m.Status == GRB.INFEASIBLE):
        #     #     self.m.computeIIS()
        #     #     self.m.write("progs/infeasible_placement_{}.ilp"
        #     #                  "".format(common_config.cfg_num))
        #     #     return

        #     self.m.printQuality()
        #     log_vars(self.m)

        #     log_results(self.devices)


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

        normalized_mem_series_CPU_Cluster = [
            d.normalized_mem_tot
            for d in self.devices
            if (isinstance(d, Cluster) or isinstance(d, CPU))]

        if(len(mem_series_P4) > 0):
            self.max_mem_P4 = self.m.addVar(vtype=GRB.CONTINUOUS,
                                            name='max_mem_P4')
            self.m.addGenConstrMax(self.max_mem_P4, mem_series_P4,
                                   name='mem_overall_P4')
            self.tot_mem_P4 = gp.quicksum(mem_series_P4)

        if(len(mem_series_CPU) > 0 or len(mem_series_Cluster) > 0):
            self.max_mem_CPU_Cluster = self.m.addVar(
                vtype=GRB.CONTINUOUS, name='max_mem_CPU_Cluster')
            self.m.addGenConstrMax(self.max_mem_CPU_Cluster,
                                   normalized_mem_series_CPU_Cluster,
                                   name='mem_overall_CPU_Cluster')
            self.tot_mem_CPU_Cluster = gp.quicksum(
                mem_series_CPU + mem_series_Cluster)

        # self.tot_mem = self.m.addVar(vtype=GRB.CONTINUOUS,
        #                              name='tot_mem')
        # self.m.addConstr(self.tot_mem == gp.quicksum(mem_series_CPU)
        #                  + gp.quicksum(mem_series_Cluster)
        #                  + gp.quicksum(mem_series_P4), name='tot_mem')

    def add_objective(self):
        if(hasattr(self, 'max_mem_CPU_Cluster')):
            self.m.setObjectiveN(self.tot_mem_CPU_Cluster, 0, 20,
                                 name='tot_mem_CPU_Cluster')
            self.m.setObjectiveN(self.max_mem_CPU_Cluster, 1, 15,
                                 name='Cluster_CPU_mem_load')

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
        normalized_rows_series_CPU_Cluster = [
            d.normalized_rows_tot
            for d in self.devices
            if (isinstance(d, Cluster) or isinstance(d, CPU))]

        if(len(rows_series_P4) > 0):
            self.max_rows_P4 = self.m.addVar(vtype=GRB.CONTINUOUS,
                                             name='max_rows_P4')
            self.m.addGenConstrMax(self.max_rows_P4, rows_series_P4,
                                   name='rows_overall_P4')
            self.tot_rows_P4 = gp.quicksum(rows_series_P4)

        if(len(rows_series_CPU) > 0 or len(rows_series_Cluster) > 0):
            self.max_rows_CPU_Cluster = self.m.addVar(
                vtype=GRB.CONTINUOUS, name='max_rows_CPU_Cluster')
            self.m.addGenConstrMax(self.max_rows_CPU_Cluster,
                                   normalized_rows_series_CPU_Cluster,
                                   name='rows_overall_CPU')
            self.tot_rows_CPU_Cluster = gp.quicksum(
                rows_series_CPU + rows_series_Cluster)

        # self.tot_rows = self.m.addVar(vtype=GRB.CONTINUOUS,
        #                               name='tot_rows')
        # self.m.addConstr(self.tot_rows == gp.quicksum(rows_series_CPU)
        #                  + gp.quicksum(rows_series_P4)
        #                  + gp.quicksum(rows_series_Cluster), name='tot_rows')

    def add_objective(self):
        if(hasattr(self, 'max_rows_CPU_Cluster')):
            self.m.setObjectiveN(self.tot_rows_CPU_Cluster, 0, 100,
                                 name='tot_rows_CPU_Cluster')
            self.m.setObjectiveN(self.max_rows_CPU_Cluster, 1, 90,
                                 name='CPU_Cluster_rows_load')
        if(hasattr(self, 'max_rows_P4')):
            self.m.setObjectiveN(self.tot_rows_P4, 4, 80, name='tot_rows_P4')
            self.m.setObjectiveN(self.max_rows_P4, 5, 70, name='P4_rows_load')
        # self.m.setObjectiveN(self.tot_rows, 2, 20, name='rows_load')
        if(hasattr(self, 'max_mem_CPU_Cluster')):
            self.m.setObjectiveN(self.tot_mem_CPU_Cluster, 6, 60,
                                 name='tot_mem_CPU_Cluster')
            self.m.setObjectiveN(self.max_mem_CPU_Cluster, 7, 50,
                                 name='CPU_Cluster_load_mem')
        if(hasattr(self, 'max_mem_P4')):
            self.m.setObjectiveN(self.tot_mem_P4, 8, 40, name='tot_mem_P4')
            self.m.setObjectiveN(self.max_mem_P4, 9, 30, name='P4_load_mem')
        # self.m.setObjectiveN(self.tot_mem, 5, 5, name='mem_load')


class Netmon(UnivmonGreedyRows):
    @memoize
    def is_clustered(self):
        for d in self.devices:
            if(isinstance(d, Cluster)):
                return True
        return False

    def add_constraints(self):
        if(self.is_clustered()):
            log.info("Netmon behaving like UnivmonGreedyRows")
            return super(Netmon, self).add_constraints()

        # Initialize with unimon_greedy_rows solution
        super(Netmon, self).add_constraints()
        super(Netmon, self).add_objective()
        log.info("-"*50)
        log.info("Running Intermediate Univmon Placement")
        self.m.update()
        # TODO:: Redundancy here. Consider running univmon at Obj init time
        self.m.optimize()
        if(self.m.Status == GRB.INFEASIBLE):
            self.infeasible = True
            self.culprit = self.m
            return

        log_objectives(self.m)
        dont_refine = self.dont_refine
        self.dont_refine = False
        super(Netmon, self).post_optimize()
        self.dont_refine = dont_refine
        # (ns_max, _) = refine_devices(self.devices)
        log_placement(self.devices, self.partitions, self.flows,
                      self.dev_par_tuplelist, self.frac,
                      msg="UnivmonGreedyRows: Intermediate Placement")
        log_results(self.devices, msg="UnivmonGreedyRows: Intermediate Results")

        # numdevices = len(self.devices)
        # numpartitions = len(self.partitions)

        # for dnum in range(numdevices):
        #     for pnum in range(numpartitions):
        for (dnum, pnum) in self.dev_par_tuplelist:
            self.frac[dnum, pnum].start = self.frac[dnum, pnum].x
            self.mem[dnum, pnum].start = self.mem[dnum, pnum].x

        self.ns_req = self.ns_max
        (self.ns, self.res) = self.add_device_model_constraints(self.ns_req)

    def add_objective(self):
        if(self.is_clustered()):
            return super(Netmon, self).add_objective()

        if(getattr(self, 'ns_req', None) is not None):
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
        if(self.is_clustered()):
            return super(Netmon, self).post_optimize()

        self.m.printQuality()
        log_vars(self.m)

        if(self.m.Status == GRB.TIME_LIMIT):
            (self.ns_max, self.res) = refine_devices(self.devices)
        else:
            if(getattr(self, 'ns_req', None) is None):
                self.ns_max = get_val(self.ns)
            self.res = get_val(self.res)
        self.refined = True

# TODO:: Can save results while refining rather than recomputing!
def log_results(devices, logger=log.info, elapsed=None, msg="Results"):

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
            switch_memory += get_val(d.mem_tot)

    log.info("{}:\nThroughput: {} Mpps, ns per packet: {}".format(
        msg, 1000/ns_max, ns_max))
    log.info("Resources: {}".format(res))
    if(total_CPUs is not None and used_cores is not None
       and switch_memory is not None):
        log.info("Used Cores: {}, Total CPUS: {}, Switch Memory: {}"
                 .format(used_cores, total_CPUs, switch_memory))

    if(common_config.results_file is not None and elapsed is not None):
        f = open(common_config.results_file, 'a')
        f.write("{:0.3f}, {:0.3f}, {}, {}, {:0.3f}, {}, ".format(
            1000/ns_max, res,
            used_cores, total_CPUs, switch_memory, elapsed))
        f.close()

    return (ns_max, res)


def log_initial(devices, partitions, flows, dev_par_tuplelist, init):
    log.info("-"*50)
    log.info("Initial UnivmonGreedyRows Fitting:")
    log.info("-"*50)
    for (dnum, d) in enumerate(devices):
        tot_rows = 0
        tot_mem = 0
        for (_, pnum) in dev_par_tuplelist.select(dnum, '*'):
            tot_rows += init.frac[dnum, pnum] * partitions[pnum].num_rows
            tot_mem += init.mem[dnum, pnum]
        log.info("\nDevice ({}) {}:".format(dnum, d))
        log.info("Rows total: {}".format(tot_rows))
        log.info("Mem total: {}".format(tot_mem))


def log_objectives(m):
    log.info("Objectives: ")
    for i in range(m.NumObj):
        m.params.ObjNumber = i
        log.info('{}: {}'.format(m.ObjNName, m.ObjNVal))
    log.info("-"*50)


def log_placement(devices, partitions, flows, dev_par_tuplelist, frac,
                  init=None, msg="Placement"):

    log.debug("-"*50)
    log.debug(msg + ":")
    # # for (qnum, q) in enumerate(queries):
    # #     log.debug("\nSketch ({}) ({})".format(q.sketch_id,
    # #                                          q.details()))
    # #     row = 1
    # #     for pnum in q.partitions:
    # #         num_rows = partitions[pnum].num_rows
    # #         log.debug("Par: {}, Rows: {}".format(row, num_rows))
    # #         row += 1
    # #         par_info = ""
    # #         total_frac = 0
    # #         for (dnum, _) in dev_par_tuplelist.select('*', pnum):
    # #             par_info += "({:0.3f},{})    ".format(frac[dnum, pnum].x, dnum)
    # #             total_frac += (frac[dnum, pnum].x)
    # #         # for (dnum, d) in enumerate(devices):
    # #         #     par_info += "{:0.3f}    ".format(frac[dnum, pnum].x)
    # #         #     total_frac += (frac[dnum, pnum].x)
    # #         log.debug(par_info)
    # #         log.debug("Total frac: {:0.3f}".format(total_frac))

    # prev_q_id = None
    # for (pnum, p) in enumerate(partitions):
    #     q = p.sketch
    #     if(q.sketch_id != prev_q_id):
    #         log.debug("\nSketch ({}) ({})"
    #                  .format(q.sketch_id, q.details()))
    #         prev_q_id = q.sketch_id

    #     log.debug("Par{} id: {}, Rows: {}"
    #              .format(p.partition_id - q.partitions[0],
    #                      p.partition_id, p.num_rows))
    #     par_info = ""
    #     total_frac = 0
    #     for (dnum, _) in dev_par_tuplelist.select('*', pnum):
    #         par_info += "({:0.3f},{})    ".format(
    #             get_val(frac[dnum, pnum]), dnum)
    #         total_frac += (get_val(frac[dnum, pnum]))
    #     log.debug(par_info)
    #     log.debug("Total frac: {:0.3f}".format(total_frac))

    for (dnum, d) in enumerate(devices):
        log.debug("\nDevice ({}) {}:".format(dnum, d))
        res_stats = d.resource_stats()
        if(res_stats != ""):
            log.debug(res_stats)
        log.debug("Rows total: {}".format(get_val(d.rows_tot)))
        log.debug("Mem total: {}".format(get_val(d.mem_tot)))
        if(init):
            tot_rows = 0
            tot_mem = 0
            for (_, pnum) in dev_par_tuplelist.select(dnum, '*'):
                tot_rows += init.frac[dnum, pnum] * partitions[pnum].num_rows
                tot_mem += init.mem[dnum, pnum]
            log.debug("Initial Rows total: {}".format(tot_rows))
            log.debug("Initial Mem total: {}".format(tot_mem))

        if(hasattr(d, 'ns')):
            log.debug("Throughput: {}".format(1000/get_val(d.ns)))

    # log.debug("")
    # for (fnum, f) in enumerate(flows):
    #     log.debug("Flow {}:".format(fnum))
    #     log.debug("partitions: {}".format(f.partitions))
    #     log.debug("path: {}".format(f.path))
    # log.debug("-"*50)


solver_names = ['Univmon', 'UnivmonGreedy', 'UnivmonGreedyRows', 'Netmon']
solver_list = [getattr(sys.modules[__name__], s) for s in solver_names]
solver_to_num = {}  # unused
solver_to_class = {}
for (solver_num, solver) in enumerate(solver_list):
    solver_to_num[solver.__name__] = solver_num
    solver_to_class[solver.__name__] = solver

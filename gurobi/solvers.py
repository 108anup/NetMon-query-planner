import logging
import math
import os
# import re
import sys
import time
from functools import partial
import numpy as np
import pprint

import gurobipy as gp
from gurobipy import GRB, tupledict, tuplelist

from common import Namespace, constants, log, log_time, memoize
from config import common_config
from devices import P4, Cluster, CPU
from helpers import get_rounded_val, get_val, is_infeasible, log_vars
from sketches import all_sketches


"""
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
1. Remove Ceiling (justified)
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


def no_traffic_md(md):
    md.static_hashes_tot = 0
    md.hashes_per_packet_thr = 0
    md.mem_updates_per_packet_thr = 0
    md.static_mem_tot = 0
    md.uniform_mem_tot = 0

    # used for univmon only
    md.normalized_hashes_per_packet_thr_tot = 0
    md.normalized_static_mem_tot = 0

    if(not common_config.perf_obj):
        md.ns_req = constants.NS_LARGEST
    md.ns = constants.NS_SMALLEST

# elements of md_list are updated in place
@log_time(logger=log.info)
def refine_devices(devices, md_list, placement_fixed=True, static=False):
    # TODO: terminology used in this function can be changed
    # This can be made more readable and organized.
    static = static or common_config.static
    #, just_collect=False):
    ns_max = None
    if(common_config.perf_obj):
        ns_max = 0
        for (dnum, d) in enumerate(devices):
            md = md_list[dnum]
            if(not hasattr(md, 'static_mem_tot')):
                assert(not hasattr(md, 'static_hashes_tot'))
                md.static_mem_tot = 0
                md.uniform_mem_tot = 0
                md.static_hashes_tot = 0
                md.hashes_per_packet_thr = 0
                md.mem_updates_per_packet_thr = 0

            ns_max = max(ns_max, d.get_ns(md))

    r = Namespace(ns_max=0, res=0, total_CPUs=0, micro_engines=0,
                  used_cores=0, switch_memory=0, nic_memory=0,
                  fpga_memory=0, fpga_hash_units=0, p10miss_count=0)
    for (dnum, d) in enumerate(devices):
        md = md_list[dnum]

        # refine should not be called with clusters
        assert(hasattr(d, 'fixed_thr'))
        if(not hasattr(md, 'static_mem_tot')):
            assert(not hasattr(md, 'static_hashes_tot'))
            # These are uninitialized when device was never a part of
            # any optimization as it did not have any flows
            # TODO: Assign 0 resources based on device type
            no_traffic_md(md)
            # NS_LARGEST will just allocate minimal forwarding resources
            # Ideally input should make sure there are no disconnected devices
            assert(not common_config.perf_obj)

        if(not d.fixed_thr):
            # static here means that resources are statically allocated by compiler
            static_mem_tot_old = md.static_mem_tot
            static_hashes_tot_old = md.static_hashes_tot
            hashes_per_packet_thr_old = md.hashes_per_packet_thr
            mem_updates_per_packet_thr_old = md.mem_updates_per_packet_thr
            uniform_mem_tot_old = md.uniform_mem_tot

            md.static_mem_tot = get_rounded_val(get_val(md.static_mem_tot))
            md.static_hashes_tot = get_rounded_val(get_val(md.static_hashes_tot))
            md.hashes_per_packet_thr = get_rounded_val(
                get_val(md.hashes_per_packet_thr))
            md.mem_updates_per_packet_thr = get_rounded_val(get_val(
                md.mem_updates_per_packet_thr))
            md.uniform_mem_tot = get_rounded_val(get_val(md.uniform_mem_tot))

            # if(not just_collect):
            # static here means that placement was fixed
            # (used for baseline with allocation)
            if(static):
                ns_max = d.get_ns(md)
                d.add_ns_constraints(None, md, ns_max)
            else:
                d.add_ns_constraints(None, md, getattr(md, 'ns_req', ns_max))
            d.resource_stats(md, r)
            # placement fixed means that just calculated ns_max
            # based on Univmon placement
            if(not placement_fixed):
                # Will be used by Netmon in later optimization
                md.static_mem_tot = static_mem_tot_old
                md.uniform_mem_tot = uniform_mem_tot_old
                md.static_hashes_tot = static_hashes_tot_old
                md.hashes_per_packet_thr = hashes_per_packet_thr_old
                md.mem_updates_per_packet_thr = mem_updates_per_packet_thr_old
        else:
            # if(not just_collect):
            d.add_ns_constraints(None, md, getattr(md, 'ns_req', ns_max))
            d.resource_stats(md, r)

        r.ns_max = max(r.ns_max, get_val(md.ns))
        r.res += get_val(d.res(md))
        if(hasattr(md, 'ns_req')):
            if(get_val(md.ns) * 1.1 > md.ns_req and get_val(md.static_mem_tot > d.cache_size)):
                r.p10miss_count += 1
        if(getattr(md, 'infeasible', None)):
            r.infeasible = True
            r.reason = "Allocating more device resources than available. Possible numerical issue!"

    return r


class MIP(Namespace):

    def __init__(self, **kwargs):
        super(MIP, self).__init__(**kwargs)
        self.infeasible = False

    def compute_dev_par_tuplelist(self):
        self.md_list = [
            Namespace(
                total_thr=0,
                dev_id=getattr(d, 'dev_id', None),
                name=d.name)
            for d in self.devices]
        dev_par_thr = tupledict()
        dev_par_tuplelist = []
        for (fnum, f) in enumerate(self.flows):
            for dnum in f.path:
                for p in f.partitions:
                    pnum = p[0]
                    dev_par_tuplelist.append((dnum, pnum))
                    if((dnum, pnum) in dev_par_thr):
                        dev_par_thr[dnum, pnum] += f.thr
                    else:
                        dev_par_thr[dnum, pnum] = f.thr
                self.md_list[dnum].total_thr += f.thr

        self.dev_par_tuplelist = tuplelist(set(dev_par_tuplelist))
        self.dev_par_thr = dev_par_thr

    @log_time
    def add_frac_mem_var(self):
        # Fraction of partition on device
        if(common_config.vertical_partition):
            # self.frac = self.m.addVars(
            #     self.dev_par_tuplelist, vtype=GRB.CONTINUOUS,
            #     lb=0, ub=1, name='frac')

            # Protocol: No partitioning allowed for P4 switches
            # As they can only have columns which are power of 2
            self.frac = tupledict()
            for (dnum, d) in enumerate(self.devices):
                tuples = self.dev_par_tuplelist.select(dnum, '*')
                if(d.cols_pwr_2 or isinstance(d, Cluster)):
                    self.frac.update(
                        self.m.addVars(tuples, vtype=GRB.BINARY, name='frac')
                    )
                else:
                    self.frac.update(
                        self.m.addVars(tuples, vtype=GRB.CONTINUOUS,
                                       lb=0, ub=1, name='frac')
                    )
        else:
            self.frac = self.m.addVars(
                self.dev_par_tuplelist, vtype=GRB.BINARY,
                name='frac')

        # Memory taken by partition
        self.static_mem = self.m.addVars(
            self.dev_par_tuplelist, vtype=GRB.CONTINUOUS,
            lb=0, name='static_mem')
        self.uniform_mem = self.m.addVars(
            self.dev_par_tuplelist, vtype=GRB.CONTINUOUS,
            lb=0, name='uniform_mem')

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

            self.cols = tupledict()
            self.cols_helper = tupledict()

            p = self.partitions[pnum]
            sk = p.sketch
            num_rows = p.num_rows

            mpr = sk.memory_per_row()
            d = self.devices[dnum]
            power_of_2_multiplier = 1

            if(d.cols_pwr_2):
                power_of_2_multiplier = 2 ** math.ceil(math.log2(mpr)) / mpr
            self.m.addConstr(self.static_mem[dnum, pnum] ==
                             sk.total_mem(num_rows) * power_of_2_multiplier
                             * self.frac[dnum, pnum],
                             name='accuracy_{}_{}'.format(dnum, pnum))
            this_uniform_mem = sk.uniform_mem(num_rows, type(self.devices[dnum]).__name__)
            self.m.addConstr(self.uniform_mem[dnum, pnum] ==
                             this_uniform_mem * power_of_2_multiplier
                             * self.frac[dnum, pnum],
                             name='uniform_mem_{}_{}'.format(dnum, pnum))

            # if(not common_config.vertical_partition):
            #     if isinstance(d, P4):
            #         mm = 2 ** math.ceil(math.log2(mm))

            #     self.m.addConstr(self.mem[dnum, pnum] ==
            #                      mm * self.frac[dnum, pnum] * num_rows,
            #                      name='accuracy_{}_{}'.format(dnum, pnum))
            # else:
            #     if(isinstance(d, P4)):
            #         '''
            #         TODO: For Univmon, this should not be added
            #         This should be checked for the univmon solution later
            #         TODO: see if this is the right place to be doing this

            #         TODO: This also has to be different for each sketch
            #         As not all sketches would preserve accuracy linearly.
            #         TODO: Make modular so that other devices can reuse
            #         '''

            #         mc = sk.cols()  # minimum cols per row
            #         self.cols[dnum, pnum] = self.m.addVar(
            #             vtype=GRB.INTEGER,
            #             name='cols_{}_{}'.format(dnum, pnum),
            #             lb=0, ub=2 ** d.max_col_bits)

            #         self.cols_helper[dnum, pnum] = self.m.addVars(
            #             [x for x in range(d.max_col_bits)], vtype=GRB.BINARY,
            #             name='cols_helper_{}_{}'.format(dnum, pnum))

            #         # can also be zero
            #         # Helper constraints
            #         self.m.addConstr(quicksum(self.cols_helper[dnum, pnum])
            #                          <= 1, name='cols_helper_{}_{}'
            #                          .format(dnum, pnum))
            #         tmp = [self.cols_helper[dnum, pnum][i] * 2**(i+1)
            #                for i in range(d.max_col_bits)]
            #         self.m.addConstr(self.cols[dnum, pnum] == quicksum(tmp),
            #                          name='cols_{}_{}'.format(dnum, pnum))

            #         # Accuracy constraints
            #         self.m.addConstr(self.cols[dnum, pnum] >=
            #                          mc * self.frac[dnum, pnum],
            #                          name='cols_acc_{}_{}'.format(dnum, pnum))
            #         self.m.addConstr(
            #             self.mem[dnum, pnum] ==
            #             (constants.cell_size * self.cols[dnum, pnum]
            #              * num_rows) / constants.KB2B,
            #             name='accuracy_{}_{}'.format(dnum, pnum))

            #     else:
            #         self.m.addConstr(self.mem[dnum, pnum] ==
            #                          mm * self.frac[dnum, pnum] * num_rows,
            #                          name='accuracy_{}_{}'.format(dnum, pnum))

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
        """
        This function apart from adding memory constraints, creates
        variables for book keeping total normalized rows/mem.

        The memory constraints are added using variable upper bounds.

        Note, the rows capacity constraints are added in
        MIP.add_device_aware_constraints

        NOTE: For now Univmon only using static mem for greedy things.
        """
        mem_list = [
            d.max_mem
            for d in self.devices]
        total_avg_mem = int(np.sum(mem_list)/len(mem_list))
        rows_list = [
            d.max_hashes
            for d in self.devices]
        total_avg_rows = int(np.sum(rows_list)/len(rows_list))

        for (dnum, d) in enumerate(self.devices):
            md = self.md_list[dnum]

            if(len(self.dev_par_tuplelist.select(dnum, '*')) > 0):
                # Memory constraints
                # Capacity constraints included in bounds

                uniform_mem_tot = self.m.addVar(
                    vtype=GRB.CONTINUOUS,
                    name='uniform_mem_tot_{}'.format(d),
                    lb=0, ub=d.max_mem)
                self.m.addConstr(
                    uniform_mem_tot == self.uniform_mem.sum(dnum, '*'),
                    name='uniform_mem_{}'.format(d))
                md.uniform_mem_tot = uniform_mem_tot

                static_mem_tot = self.m.addVar(
                    vtype=GRB.CONTINUOUS,
                    name='static_mem_tot_{}'.format(d),
                    lb=0, ub=d.max_mem)
                self.m.addConstr(
                    static_mem_tot == self.static_mem.sum(dnum, '*'),
                    name='static_mem_{}'.format(d))
                md.static_mem_tot = static_mem_tot

                normalized_static_mem_tot = self.m.addVar(
                    vtype=GRB.CONTINUOUS,
                    name='normalized_mem_tot_{}'.format(d),
                    lb=0)
                self.m.addConstr(normalized_static_mem_tot ==
                                 (total_avg_mem * static_mem_tot) / d.max_mem)
                md.normalized_static_mem_tot = normalized_static_mem_tot

                # Row constraints for bookkeeping variables
                static_hashes_tot = self.m.addVar(
                    vtype=GRB.CONTINUOUS,
                    name='static_hashes_tot_{}'.format(d), lb=0)
                normalized_hashes_per_packet_thr_tot = self.m.addVar(
                    vtype=GRB.CONTINUOUS,
                    name='normalized_hashes_per_packet_thr_tot_{}'.format(d),
                    lb=0)
                hashes_per_packet_thr = self.m.addVar(
                    vtype=GRB.CONTINUOUS,
                    name='hashes_per_packet_thr_{}'.format(d), lb=0)
                mem_updates_per_packet_thr = self.m.addVar(
                    vtype=GRB.CONTINUOUS,
                    name='mem_udpates_per_packet_thr_{}'.format(d), lb=0)

                static_hashes_series = []
                hashes_per_packet_thr_series = []
                mem_updates_per_packet_thr_series = []
                for (_, pnum) in self.dev_par_tuplelist.select(dnum, '*'):
                    p = self.partitions[pnum]
                    static_hashes_series.append(
                        p.sketch.hashes_per_packet(p.num_rows)
                        * self.frac[dnum, pnum]
                    )
                    hashes_per_packet_thr_series.append(
                        p.sketch.hashes_per_packet(p.num_rows)
                        * self.frac[dnum, pnum]
                        * self.dev_par_thr[dnum, pnum])
                    mem_updates_per_packet_thr_series.append(
                        p.sketch.mem_updates_per_packet(p.num_rows)
                        * self.frac[dnum, pnum]
                        * self.dev_par_thr[dnum, pnum])
                # rows_series = [p.num_rows * frac[dnum, p.partition_id]
                #                for p in self.partitions]
                if (len(hashes_per_packet_thr_series) > 0):
                    if(md.total_thr == 0):
                        """
                        It should not be the case that there is device
                        without traffic that must hold a sketch.
                        This means that the input is malformed, or there
                        some bug in the input helper functions.
                        """
                        print(static_hashes_series)
                        print(hashes_per_packet_thr_series)
                        assert(False)
                    self.m.addConstr(
                        hashes_per_packet_thr ==
                        gp.quicksum(hashes_per_packet_thr_series)
                        / md.total_thr,
                        name='hashes_per_packet_thr_{}'.format(d))
                    self.m.addConstr(
                        mem_updates_per_packet_thr ==
                        gp.quicksum(mem_updates_per_packet_thr_series)
                        / md.total_thr,
                        name='mem_updates_per_packet_thr_{}'.format(d))
                    self.m.addConstr(static_hashes_tot
                                     == gp.quicksum(static_hashes_series),
                                     name='static_hashes_tot_{}'.format(d))
                else:
                    self.m.addConstr(
                        hashes_per_packet_thr == 0,
                        name='hashes_per_packet_thr_{}'.format(d))
                    self.m.addConstr(
                        mem_updates_per_packet_thr == 0,
                        name='hashes_per_packet_thr_{}'.format(d))
                    self.m.addConstr(static_hashes_tot == 0,
                                     name='static_hashes_tot_{}'.format(d))

                md.static_hashes_tot = static_hashes_tot
                md.hashes_per_packet_thr = hashes_per_packet_thr
                md.mem_updates_per_packet_thr = mem_updates_per_packet_thr
                if(not common_config.perf_obj):
                    md.ns_req = 1000 / md.total_thr
                self.m.addConstr(normalized_hashes_per_packet_thr_tot ==
                                 (total_avg_rows * hashes_per_packet_thr)
                                 / d.max_hashes)
                md.normalized_hashes_per_packet_thr_tot = \
                    normalized_hashes_per_packet_thr_tot
                md.m = self.m
            else:
                no_traffic_md(md)

    def add_device_aware_constraints(self):
        """
        Constraints that mostly basically ensure
        that the sketch fits in P4 switch.
        """
        for (dnum, d) in enumerate(self.devices):
            md = self.md_list[dnum]
            self.m.addConstr(md.static_hashes_tot <= d.max_hashes,
                             name='static_row_capacity_{}'.format(d))

            if hasattr(d, 'max_mpr'):
                for (_, pnum) in self.dev_par_tuplelist.select(dnum, '*'):
                    p = self.partitions[pnum]
                    self.m.addConstr(
                        self.static_mem[dnum, pnum] <= d.max_mpr * p.num_rows,
                        'static_capacity_mem_par_{}_{}'.format(pnum, d))

    def add_constraints(self):
        pass

    def add_objective(self):
        pass

    def post_optimize(self):
        pass

    def check_device_aware_constraints(self):
        for (dnum, d) in enumerate(self.devices):
            md = self.md_list[dnum]
            if not (get_val(md.static_hashes_tot) <= d.max_hashes):
                return False
            if hasattr(d, 'max_mpr'):
                for (_, pnum) in self.dev_par_tuplelist.select(dnum, '*'):
                    p = self.partitions[pnum]
                    if not (get_val(self.static_mem[dnum, pnum])
                            <= d.max_mpr * p.num_rows):
                        return False
        return True

    @log_time
    def add_device_model_constraints(self, ns_max=None):
        res_acc = gp.LinExpr()
        for (dnum, d) in enumerate(self.devices):
            md = self.md_list[dnum]

            # adding these constraints here
            assert(isinstance(md.static_mem_tot, gp.Var)
                   or md.static_mem_tot == 0)
            assert(isinstance(md.static_hashes_tot, gp.Var)
                   or md.static_hashes_tot == 0)

            if(isinstance(md.static_mem_tot, gp.Var)
               or isinstance(md.static_hashes_tot, gp.Var)):
                # Throughput
                # Simple total model
                d.add_ns_constraints(self.m, md, getattr(md, 'ns_req', ns_max))

                # Resources
                res_acc += d.res(md)

        ns = None
        if(common_config.perf_obj):
            ns_series = [md.ns for md in self.md_list]
            ns = self.m.addVar(vtype=GRB.CONTINUOUS, name='ns')
            self.m.addGenConstrMax(ns, ns_series, name='ns_overall')
        res = self.m.addVar(vtype=GRB.CONTINUOUS, name='res')
        self.m.addConstr(res_acc == res, name='res_acc')
        return (ns, res)

    def initialize(self):
        for (dnum, pnum) in self.dev_par_tuplelist:
            if(hasattr(self.init, 'static_mem')):
                self.static_mem[dnum, pnum].start = \
                    get_val(self.init.static_mem[dnum, pnum])
            if((dnum, pnum) in self.init.frac):
                self.frac[dnum, pnum].start = \
                    get_val(self.init.frac[dnum, pnum])
            else:
                self.frac[dnum, pnum].start = 0

        # log_initial(self.devices, self.partitions, self.flows,
        #             self.dev_par_tuplelist, self.init)

    def add_check_constraints(self):
        """
        Used for debugging whether clustering total resources used is
        same as no clustering total resources used.
        """
        for (dnum, pnum) in self.dev_par_tuplelist:
            if((dnum, pnum) in self.check.frac):
                self.m.addConstr(
                    self.frac[dnum, pnum] >=
                    get_val(self.check.frac[dnum, pnum])
                    - common_config.ftol
                )

    @log_time
    def solve(self):
        self.m = gp.Model(self.__class__.__name__)
        # self.m.setParam(GRB.Param.Threads, 2)
        self.m.ModelSense = GRB.MINIMIZE
        log.info("\n" + "-"*80)
        log.info("Model {} with:\n"
                 "{} devices, {} partitions and {} flows"
                 .format(type(self).__name__, len(self.devices),
                         len(self.partitions), len(self.flows)))

        if(not common_config.mipout):
            self.m.setParam(GRB.Param.LogToConsole, 0)
            # gplogger = logging.getLogger('gurobipy')
            # gplogger.disabled = True

        if(common_config.time_limit):
            self.m.setParam(GRB.Param.TimeLimit, common_config.time_limit)
        # m.setParam(GRB.Param.MIPGapAbs, common_config.mipgapabs)
        self.m.setParam(GRB.Param.MIPGap, common_config.MIP_GAP_REL)

        self.compute_dev_par_tuplelist()
        self.add_frac_mem_var()
        self.add_coverage_constraints()
        self.add_accuracy_constraints()
        self.add_capacity_constraints()
        # if(not type(self).__name__ == 'Univmon'):
        self.add_device_aware_constraints()

        if(hasattr(self, 'init')):
            self.initialize()

        self.add_constraints()
        if(self.infeasible):
            return

        if(hasattr(self, 'check')):
            self.add_check_constraints()

        if(not getattr(self, 'objectives_added', False)):
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

        # import ipdb; ipdb.set_trace()

        if(is_infeasible(self.m)):
            self.infeasible = True
            self.reason = ('Infeasible solve: {} devices, {} sketches'
                           .format(len(self.devices), len(self.partitions)))
            self.culprit = self.m
            return
        else:
            log_objectives(self.m)
            self.post_optimize()
            log_placement(self.devices, self.partitions, self.flows,
                          self.dev_par_tuplelist, self.frac, self.md_list,
                          init=getattr(self, 'init', None))
            # Basically if Netmon was run in intermediate and it did not
            # behave like Univmon Greedy Rows
            if(hasattr(self, 'refined')):
                if(self.refined and self.dont_refine):
                    log_results(self.devices, self.r, self.md_list,
                                msg="Netmon Intermediate Result")


class Univmon(MIP):

    def add_constraints(self):
        mem_series = [md.static_mem_tot for md in self.md_list]
        self.max_mem = self.m.addVar(vtype=GRB.CONTINUOUS, name='max_mem')
        self.m.addGenConstrMax(self.max_mem, mem_series, name='mem_overall')
        self.tot_mem = self.m.addVar(vtype=GRB.CONTINUOUS,
                                     name='tot_mem')
        self.m.addConstr(self.tot_mem == gp.quicksum(mem_series),
                         name='tot_mem')

    def add_objective(self):
        self.m.setObjectiveN(self.max_mem, 0, weight=len(self.devices), name='max_mem')
        self.m.setObjectiveN(self.tot_mem, 1, weight=1, name='tot_mem')

    def post_optimize(self):
        self.m.printQuality()
        log_vars(self.m)

        # log_placement(self.devices, self.partitions, self.flows,
        #               self.dev_par_tuplelist, self.frac)

        if(not common_config.use_model):
            if(type(self).__name__ == 'Univmon'):
                if (not self.check_device_aware_constraints()):
                    self.reason = ("Infeasible placement due to Univmon's "
                                   "lack of knowledge")
                    self.infeasible = True
                    self.culprit = self.m
                    return

            if(not self.dont_refine):
                self.r = refine_devices(
                    self.devices, self.md_list,
                    getattr(self, 'placement_fixed', True))
                self.refined = True
                if(getattr(self.r, 'infeasible', None)):
                    self.infeasible = True
                    self.reason = 'Can\'t meet traffic demand'
                    self.culprit = None

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


class UnivmonGreedy(Univmon):
    # Derived from devices.py
    fixed_thr = (P4)

    def add_constraints(self):
        mem_series_fixed_thr = [self.md_list[dnum].static_mem_tot
                                for (dnum, d) in enumerate(self.devices)
                                if isinstance(d, self.fixed_thr)]
        mem_series_others = [self.md_list[dnum].static_mem_tot
                             for (dnum, d) in enumerate(self.devices)
                             if not isinstance(d, self.fixed_thr)]
        normalized_mem_series_others = [
            self.md_list[dnum].normalized_static_mem_tot
            for (dnum, d) in enumerate(self.devices)
            if (not isinstance(d, self.fixed_thr))]

        if(len(mem_series_fixed_thr) > 0):
            self.max_mem_fixed_thr = self.m.addVar(vtype=GRB.CONTINUOUS,
                                                   name='max_mem_fixed_thr')
            self.m.addGenConstrMax(self.max_mem_fixed_thr,
                                   mem_series_fixed_thr,
                                   name='mem_overall_fixed_thr')
            self.tot_mem_fixed_thr = gp.quicksum(mem_series_fixed_thr)

        if(len(mem_series_others) > 0):
            self.max_mem_others = self.m.addVar(
                vtype=GRB.CONTINUOUS, name='max_mem_others')
            self.m.addGenConstrMax(self.max_mem_others,
                                   normalized_mem_series_others,
                                   name='mem_overall_others')
            self.tot_mem_others = gp.quicksum(mem_series_others)

        # self.tot_mem = self.m.addVar(vtype=GRB.CONTINUOUS,
        #                              name='tot_mem')
        # self.m.addConstr(self.tot_mem == gp.quicksum(mem_series_CPU)
        #                  + gp.quicksum(mem_series_Cluster)
        #                  + gp.quicksum(mem_series_P4), name='tot_mem')

    def add_objective(self):

        if(hasattr(self, 'max_mem_others')):
            self.m.setObjectiveN(self.tot_mem_others, 0, 20,
                                 name='tot_mem_others')
            self.m.setObjectiveN(self.max_mem_others, 1, 15,
                                 name='mem_load_others')

        if(hasattr(self, 'max_mem_fixed_thr')):
            self.m.setObjectiveN(self.tot_mem_fixed_thr, 2, 10,
                                 name='tot_mem_fixed_thr')
            self.m.setObjectiveN(self.max_mem_fixed_thr, 3, 5,
                                 name='mem_load_fixed_thr')
        # self.m.setObjectiveN(self.tot_mem, 2, 1, name='mem_load')


class UnivmonGreedyRows(UnivmonGreedy):

    def add_constraints(self):
        super(UnivmonGreedyRows, self).add_constraints()

        """
        The logic here is that, normalization is only needed for others
        as fixed thr is only P4 and does not need to be normalized.

        NOTE: self.fixed_thr is a tuple defined in UnivmonGreedy

        Currently minimizing sum of static_hashes over all devices and
        max of normalized hashes. TODO: Check if this actually makes sense.
        """

        static_rows_series_fixed_thr = [
            self.md_list[dnum].static_hashes_tot
            for (dnum, d) in enumerate(self.devices)
            if isinstance(d, self.fixed_thr)]
        static_rows_series_others = [
            self.md_list[dnum].static_hashes_tot
            for (dnum, d) in enumerate(self.devices)
            if not isinstance(d, self.fixed_thr)]
        normalized_hashes_per_packet_thr_series_others = [
            self.md_list[dnum].normalized_hashes_per_packet_thr_tot
            for (dnum, d) in enumerate(self.devices)
            if (not isinstance(d, self.fixed_thr))]

        if(len(static_rows_series_fixed_thr) > 0):
            self.max_rows_fixed_thr = self.m.addVar(vtype=GRB.CONTINUOUS,
                                                    name='max_rows_fixed_thr')
            self.m.addGenConstrMax(self.max_rows_fixed_thr,
                                   static_rows_series_fixed_thr,
                                   name='rows_overall_fixed_thr')
            self.tot_rows_fixed_thr = gp.quicksum(static_rows_series_fixed_thr)

        if(len(static_rows_series_others) > 0):
            self.max_rows_others = self.m.addVar(
                vtype=GRB.CONTINUOUS, name='max_rows_others')
            self.m.addGenConstrMax(
                self.max_rows_others,
                normalized_hashes_per_packet_thr_series_others,
                name='rows_overall_others')
            self.tot_rows_others = gp.quicksum(static_rows_series_others)

        # self.tot_rows = self.m.addVar(vtype=GRB.CONTINUOUS,
        #                               name='tot_rows')
        # self.m.addConstr(self.tot_rows == gp.quicksum(rows_series_CPU)
        #                  + gp.quicksum(rows_series_P4)
        #                  + gp.quicksum(rows_series_Cluster), name='tot_rows')

    def add_objective(self):

        # MIP Focus 3 to move objective bounds faster
        """
        Root Simplex takes a ton of time for objective 1
        and objective 5. Even with simplex pricing 2
        """
        # self.m.setParam(GRB.Param.SimplexPricing, 2)
        if(hasattr(self, 'max_rows_others')):
            self.m.setObjectiveN(self.tot_rows_others, 0, 100, weight=1,
                                 name='tot_rows_others')
            self.m.setObjectiveN(self.max_rows_others, 1, 90, weight=len(self.devices),
                                 name='others_rows_load')
            env1 = self.m.getMultiobjEnv(1)
            env1.setParam(GRB.Param.MIPFocus, 2)
            if(getattr(common_config, 'ABS_TIME_ON_UNIVMON_BOTTLENECK', None)
               is not None):
                log.error("Setting Univmon Time limit to: {}".format(
                    common_config.ABS_TIME_ON_UNIVMON_BOTTLENECK))
                env1.setParam(GRB.Param.TimeLimit,
                              common_config.ABS_TIME_ON_UNIVMON_BOTTLENECK)
            elif(getattr(common_config,
                         'MIP_GAP_REL_UNIVMON_BOTTLENECK', None)
                 is not None):
                env1.setParam(GRB.Param.MIPGap,
                              common_config.MIP_GAP_REL_UNIVMON_BOTTLENECK)

        if(hasattr(self, 'max_rows_fixed_thr')):
            self.m.setObjectiveN(self.tot_rows_fixed_thr, 2, 80,
                                 name='tot_rows_fixed_thr')
            self.m.setObjectiveN(self.max_rows_fixed_thr, 3, 70,
                                 name='rows_load_fixed_thr')
        # self.m.setObjectiveN(self.tot_rows, 2, 20, name='rows_load')

        if(hasattr(self, 'max_mem_others')):
            self.m.setObjectiveN(self.tot_mem_others, 4, 60, weight=1,
                                 name='tot_mem_others')
            self.m.setObjectiveN(self.max_mem_others, 5, 50, weight=len(self.devices),
                                 name='others_load_mem')
            env5 = self.m.getMultiobjEnv(5)
            env5.setParam(GRB.Param.MIPFocus, 2)
            if(getattr(common_config, 'ABS_TIME_ON_UNIVMON_BOTTLENECK', None)
               is not None):
                log.error("Setting Univmon Time limit to: {}".format(
                    common_config.ABS_TIME_ON_UNIVMON_BOTTLENECK))
                env5.setParam(GRB.Param.TimeLimit,
                              common_config.ABS_TIME_ON_UNIVMON_BOTTLENECK)
            elif(getattr(common_config,
                         'MIP_GAP_REL_UNIVMON_BOTTLENECK', None)
                 is not None):
                env5.setParam(GRB.Param.MIPGap,
                              common_config.MIP_GAP_REL_UNIVMON_BOTTLENECK)

        if(hasattr(self, 'max_mem_fixed_thr')):
            self.m.setObjectiveN(self.tot_mem_fixed_thr, 6, 40,
                                 name='tot_mem_fixed_thr')
            self.m.setObjectiveN(self.max_mem_fixed_thr, 7, 30,
                                 name='mem_load_fixed_thr')
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

        if(not hasattr(self, 'init') and common_config.perf_obj and
           getattr(self, 'ns_req', None) is None):
            # Initialize with unimon_greedy_rows solution
            super(Netmon, self).add_constraints()
            super(Netmon, self).add_objective()
            log.info("-"*50)
            log.info("Running Intermediate Univmon Placement")
            self.m.update()
            # HOLD: Redundancy here. Consider running univmon at Obj init time
            self.m.optimize()
            if(is_infeasible(self.m)):
                self.infeasible = True
                self.culprit = self.m
                self.reason = ('Infeasible solve: {} devices, {} sketches'
                               .format(len(self.devices), len(self.partitions)))
                return

            log_objectives(self.m)
            dont_refine = self.dont_refine
            self.dont_refine = False
            self.placement_fixed = False
            super(Netmon, self).post_optimize()
            self.dont_refine = dont_refine
            log_placement(self.devices, self.partitions, self.flows,
                          self.dev_par_tuplelist, self.frac, self.md_list,
                          msg="UnivmonGreedyRows: Intermediate Placement")
            log_results(self.devices, self.r, self.md_list,
                        msg="UnivmonGreedyRows: Intermediate Results")

            """
            TODO: I am facing an error similar to:
            https://support.gurobi.com/hc/en-us/community/posts/360056102052-Test-if-the-start-solution-is-feasible?page=1
            https://groups.google.com/forum/#!topic/gurobi/C1bIDPFvtKY
            """

            # frac_old_ub = tupledict()
            # mem_old_ub = tupledict()
            for (dnum, pnum) in self.dev_par_tuplelist:
                # frac_old_ub[dnum, pnum] = self.frac[dnum, pnum].ub
                # mem_old_ub[dnum, pnum] = self.mem[dnum, pnum].ub
                # self.frac[dnum, pnum].ub = self.frac[dnum, pnum].x
                # self.frac[dnum, pnum].lb = self.frac[dnum, pnum].x
                # self.mem[dnum, pnum].ub = self.mem[dnum, pnum].x
                # self.mem[dnum, pnum].lb = self.mem[dnum, pnum].x
                self.frac[dnum, pnum].start = self.frac[dnum, pnum].x
                self.static_mem[dnum, pnum].start = self.static_mem[dnum, pnum].x

            # # NOTE:: Check this!
            # With both netro and CPU UGR solution may not be optimal
            # Also with rows_thr also the solution by UGR may not be optimal
            self.ns_req = self.r.ns_max + common_config.ftol

            # self.m.setParam(GRB.Param.MIPFocus, 1)
            # self.m.setParam(GRB.Param.NonConvex, 2)
            # (self.ns, self.res) = self.add_device_model_constraints()

            # self.add_objective()
            # self.objectives_added = True
            # self.m.update()
            # self.m.optimize()
            # self.m.printQuality()
            # if(is_infeasible(self.m)):
            #     self.infeasible = True
            #     self.culprit = self.m
            #     self.reason = 'Constraint violation in UGR -> Netmon'
            #     return

            # for v in self.m.getVars():
            #     v.start = v.x

            # for (dnum, pnum) in self.dev_par_tuplelist:
            #     self.frac[dnum, pnum].ub = frac_old_ub[dnum, pnum]
            #     self.mem[dnum, pnum].ub = mem_old_ub[dnum, pnum]

        self.m.setParam(GRB.Param.NonConvex, 2)
        self.m.setParam(GRB.Param.MIPFocus, common_config.NETMON_RES_FOCUS)
        (self.ns, self.res) = self.add_device_model_constraints(
            getattr(self, 'ns_req', None))

    def add_objective(self):
        if(self.is_clustered()):
            return super(Netmon, self).add_objective()
        # Spend 10% time in finding a good feasible solution using heuristics
        # self.m.setParam(GRB.Param.Heuristics, 0.5)
        # self.m.setParam(GRB.Param.ImproveStartGap, 0.05)
        # MIP Focus 2 to get better quality solutions
        # MIP Focus 1 to get faster to a feasible solution
        if(not common_config.perf_obj or
           getattr(self, 'ns_req', None) is not None):
            self.m.NumObj = 1
            self.m.setObjectiveN(self.res, 0, 10, reltol=common_config.res_tol,
                                 name='res')

            env0 = self.m.getMultiobjEnv(0)
            env0.setParam(GRB.Param.NonConvex, 2)
            env0.setParam(GRB.Param.MIPFocus, common_config.NETMON_RES_FOCUS)
            env0.setParam(GRB.Param.MIPGapAbs, common_config.MIP_GAP_ABS_RES)
            env0.setParam(GRB.Param.MIPGap, 0)
            if(common_config.time_limit):
                env0.setParam(GRB.Param.TimeLimit, common_config.time_limit)

        else:
            self.m.NumObj = 2
            self.m.setObjectiveN(self.ns, 0, 10, reltol=common_config.ns_tol,
                                 name='ns')
            self.m.setObjectiveN(self.res, 1, 5, reltol=common_config.res_tol,
                                 name='res')

            t0 = (common_config.time_limit
                  * common_config.PORTION_TIME_ON_PERF)
            env0 = self.m.getMultiobjEnv(0)
            env0.setParam(GRB.Param.NonConvex, 2)
            env0.setParam(GRB.Param.MIPFocus, 2)
            # env0.setParam(GRB.Param.TimeLimit, t0)
            env0.setParam(GRB.Param.MIPGap, common_config.MIP_GAP_REL)

            t1 = common_config.time_limit - t0
            env1 = self.m.getMultiobjEnv(1)
            env1.setParam(GRB.Param.NonConvex, 2)
            env1.setParam(GRB.Param.MIPFocus, 2)
            # env1.setParam(GRB.Param.TimeLimit, t1)
            env1.setParam(GRB.Param.MIPGapAbs, common_config.MIP_GAP_ABS_RES)
            env1.setParam(GRB.Param.MIPGap, 0)

    def post_optimize(self):
        if(self.is_clustered()):
            return super(Netmon, self).post_optimize()

        self.m.printQuality()
        log_vars(self.m)

        # if(self.m.Status == GRB.TIME_LIMIT):
        #     self.r = refine_devices(
        #         self.devices, self.md_list)

        """
        Following is only used to collect stats
        d.add_ns_constraints is not required to be called
        when calling refine_devices from this location
        The manual resource allocator, even though it modifies the
        resource variables and their types; the values of these variables
        remains the same. Also this helped verify that manual allocator and
        Gurobi were doing the same thing! :)
        """
        self.r = refine_devices(
            self.devices, self.md_list)

        # else:
        #     self.r = Namespace()
        #     self.r.res = get_val(self.res)
        # if(common_config.perf_obj):
        #     self.r.ns_max = self.ns
        self.refined = True


def log_results(devices, r, md_list, logger=log.info,
                elapsed=None, msg="Results"):

    # r is changed in place
    assert(((not common_config.perf_obj) or getattr(r, 'ns_max', None))
           and getattr(r, 'res', None))
    # If not pre-computed results from refine
    if(getattr(r, 'used_cores', None) is None):
        assert(False)
        r = refine_devices(devices, md_list)
        # r.used_cores = 0
        # r.total_CPUs = 0
        # r.res = 0
        # r.ns_max = 0
        # r.nic_memory = 0
        # r.switch_memory = 0
        # r.micro_engines = 0
        # # This would be used by solving runs which did not
        # # explicitly use refine_devices: basically vanilla Netmon
        # for (dnum, d) in enumerate(devices):
        #     md = md_list[dnum]
        #     r.ns_max = max(r.ns_max, get_val(md.ns))
        #     r.res += get_val(d.res(md))
        #     d.resource_stats(md, r)
    if(common_config.perf_obj):
        log.info("{}:\nThroughput: {} Mpps, ns per packet: {}".format(
            msg, 1000/r.ns_max, r.ns_max))
    log.info("Resources: {}".format(r.res))
    if(r.total_CPUs is not None and r.used_cores is not None
       and r.switch_memory is not None and r.micro_engines is not None
       and r.nic_memory is not None and r.fpga_memory is not None
       and r.fpga_hash_units is not None):
        log.info("Used Cores: {}, Total CPUS: {}, "
                 "Switch Memory: {}, \n"
                 "Micro-enginges: {}, NIC Memory: {}\n"
                 "FPGA-hash_units: {}, FPGA Memory: {}"
                 .format(r.used_cores, r.total_CPUs,
                         r.switch_memory, r.micro_engines,
                         r.nic_memory, r.fpga_hash_units,
                         r.fpga_memory))
        log.info("Devices needing more res: {}\n"
                 "Total devices: {}, Devices needing changes: {}%\n"
                 .format(r.p10miss_count, len(devices),
                         round(100 * r.p10miss_count/len(devices), 2))
                 )

    if(common_config.results_file is not None and elapsed is not None):
        f = open(common_config.results_file, 'a')
        f.write("{:0.3f}, {:0.3f}, {}, {}, {:0.3f}, {}, {:0.3f}, {}, {:0.3f}, {}, {}, {}".format(
            1000/r.ns_max, r.res,
            r.used_cores, r.total_CPUs, r.switch_memory,
            r.micro_engines, r.nic_memory, r.fpga_hash_units,
            r.fpga_memory, elapsed,
            r.p10miss_count, len(devices)))
        f.close()

        # Logging for analysis
        # import pickle
        # with open(common_config.results_file + "_result.pickle", 'wb') as f:
        #     pickle.dump(r, f)
        # with open(common_config.results_file + "_devices.pickle", 'wb') as f:
        #     pickle.dump(devices, f)
        # with open(common_config.results_file + "_md_list.pickle", 'wb') as f:
        #     pickle.dump(md_list, f)


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
                  md_list, init=None, msg="Placement"):

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

    sketch_dev_affinity = {}
    for sk in all_sketches:
        sketch_dev_affinity[sk.__name__] = {}

    prev_q_id = None
    logger = partial(log.log, logging.DEBUG-1)
    for (pnum, p) in enumerate(partitions):
        q = p.sketch
        if(q.sketch_id != prev_q_id):
            logger("\nSketch ({}) ({})"
                   .format(q.sketch_id, q.details()))
            prev_q_id = q.sketch_id

        logger("Par{} id: {}, Rows: {}"
               .format(p.partition_id - q.partitions[0],
                       p.partition_id, p.num_rows))
        par_info = ""
        total_frac = 0
        for (dnum, _) in dev_par_tuplelist.select('*', pnum):
            par_info += "({:0.3f},{})    ".format(
                get_val(frac[dnum, pnum]), dnum)
            total_frac += (get_val(frac[dnum, pnum]))
            dtype = type(devices[dnum]).__name__
            stype = type(p.sketch).__name__
            sketch_dev_affinity[stype][dtype] = \
                sketch_dev_affinity[stype].get(dtype, 0) + 1
        logger(par_info)
        logger("Total frac: {:0.3f}".format(total_frac))

    for (dnum, d) in enumerate(devices):
        md = md_list[dnum]
        log.debug("\nDevice ({}) {}:".format(dnum, d))
        res_stats = d.resource_stats(md)
        if(res_stats != ""):
            log.debug(res_stats)
        log.debug("Static Hashes total: {}".format(get_val(md.static_hashes_tot)))
        log.debug("Expected hashes: {}".format(
            get_val(md.hashes_per_packet_thr)))
        log.debug("Static Mem total: {}".format(get_val(md.static_mem_tot)))
        if(hasattr(md, 'ns_sketch')):
            log.debug("ns_sketch: {}".format(get_val(md.ns_sketch)))
        if(hasattr(md, 'ns_dpdk')):
            log.debug("ns_dpdk: {}".format(get_val(md.ns_dpdk)))
        if(init):
            tot_rows = 0
            tot_static_mem = 0
            for (_, pnum) in dev_par_tuplelist.select(dnum, '*'):
                tot_rows += init.frac[dnum, pnum] * partitions[pnum].num_rows
                tot_static_mem += init.static_mem[dnum, pnum]
            log.debug("Initial Rows total: {}".format(tot_rows))
            log.debug("Initial Mem total: {}".format(tot_static_mem))

        if(hasattr(md, 'ns')):
            log.debug("Throughput: {} Mpps".format(1000/get_val(md.ns)))
        if(hasattr(md, 'ns_req')):
            log.debug("Throughput Req: {} Mpps".format(1000/md.ns_req))
            # if(isinstance(d, CPU)):
            #     d.log_vars(md)

    # log.debug("")
    # for (fnum, f) in enumerate(flows):
    #     log.debug("Flow {}:".format(fnum))
    #     log.debug("partitions: {}".format(f.partitions))
    #     log.debug("path: {}".format(f.path))
    # log.debug("-"*50)
    if(msg == "Final Placement"):
        dev_dict = sketch_dev_affinity['cm_sketch']
        dev_names = list(dev_dict.keys())
        header = "Sketch, " + ", ".join(dev_names)
        log.info(header)
        for sk, dev_dict in sketch_dev_affinity.items():
            counts = [str(dev_dict.get(dev, 0)) for dev in dev_names]
            log.info("{}, {}".format(sk, ", ".join(counts)))
        # pprint.pprint(sketch_dev_affinity)

solver_names = ['Univmon', 'UnivmonGreedy', 'UnivmonGreedyRows', 'Netmon']
solver_list = [getattr(sys.modules[__name__], s) for s in solver_names]
solver_to_num = {}  # unused
solver_to_class = {}
for (solver_num, solver) in enumerate(solver_list):
    solver_to_num[solver.__name__] = solver_num
    solver_to_class[solver.__name__] = solver

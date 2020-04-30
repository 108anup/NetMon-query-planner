import math
import gurobipy as gp
from gurobipy import GRB

from config import common_config
from common import Namespace, memoize
from helpers import get_val, get_rounded_val, log_vars


class device(Namespace):

    def add_ns_constraints(self, m):
        pass

    def __init__(self, *args, **kwargs):
        super(device, self).__init__(*args, **kwargs)

    def res(self):
        return 0

    def __repr__(self):
        return self.name

    def resource_stats(self, md):
        return ""

    def get_ns(self, md):
        return 1000 / self.line_thr


class CPU(device):
    # TODO:: update with OVS
    fraction_parallel = 3/4
    static_loads = [0, 6, 12, 18, 24, 30, 43, 49, 55]
    s_rows = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    cache = {}  # TODO: see if there is more performant cache

    def get_pdt_var(self, a, b, pdt_name, m, direction):
        m.update()
        pdt = m.addVar(vtype=GRB.CONTINUOUS,
                       name='pdt_{}_{}'.format(pdt_name, self))
        # loga = m.addVar(vtype=GRB.CONTINUOUS,
        #                 name='log_{}'.format(a.varName), lb=-GRB.INFINITY)
        # logb = m.addVar(vtype=GRB.CONTINUOUS,
        #                 name='log_{}'.format(b.varName), lb=-GRB.INFINITY)
        # logpdt = m.addVar(vtype=GRB.CONTINUOUS,
        #                   name='log_pdt_{}_{}'.format(pdt_name, self),
        #                   lb=-GRB.INFINITY)
        # # m.addGenConstrLogA(pdt, logpdt, 2,
        # #                    name='log_pdt_{}_{}'.format(pdt_name, self),
        # #                    options="FuncPieces=-1 FuncPieceError=0.00001")
        # # m.addGenConstrLogA(a, loga, 2,
        # #                    name='log_{}'.format(a.varName),
        # #                    options="FuncPieces=-1 FuncPieceError=0.00001")
        # # m.addGenConstrLogA(b, logb, 2,
        # #                    name='log_{}'.format(b.varName),
        # #                    options="FuncPieces=-1 FuncPieceError=0.00001")
        # m.addGenConstrExpA(logpdt, pdt, 2,
        #                    name='exp_pdt_{}_{}'.format(pdt_name, self),
        #                    options="FuncPieces=-1 FuncPieceError=0.01")
        # m.addGenConstrExpA(loga, a, 2,
        #                    name='exp_{}'.format(a.varName),
        #                    options="FuncPieces=-1 FuncPieceError=0.01")
        # m.addGenConstrExpA(logb, b, 2,
        #                    name='exp_{}'.format(b.varName),
        #                    options="FuncPieces=-1 FuncPieceError=0.01")
        # m.addConstr(logpdt == loga + logb,
        #             name='pdt_{}_{}'.format(pdt_name, self))

        m.addQConstr(pdt == a * b,
                     name='pdt_{}_{}'.format(pdt_name, self))

        # if (direction == 0):
        #     # convex problem
        #     m.addQConstr(pdt <= a * b,
        #                  name='pdt_{}_{}'.format(pdt_name, self))
        # elif (direction == 1):
        #     # non convex problem
        #     m.addQConstr(pdt >= a * b,
        #                  name='pdt_{}_{}'.format(pdt_name, self))
        # else:
        #     print("Error: Invalid input 'direction' to get_pdt_var")
        #     sys.exit(-1)
        return pdt

    def set_thr(self, md, ns_req):
        md.cores_sketch = math.ceil(md.ns_single / ns_req)
        dpdk_single_ns = 1000/self.dpdk_single_core_thr
        if(md.cores_sketch != 0):
            md.ns_sketch = md.ns_single / md.cores_sketch
        else:
            md.ns_sketch = 0
        f = CPU.fraction_parallel
        dpdk_cores = f/(ns_req/dpdk_single_ns - 1 + f)
        assert(dpdk_cores > 0)
        md.cores_dpdk = math.ceil(dpdk_cores)
        md.ns_dpdk = dpdk_single_ns * (1-f + f/md.cores_dpdk)
        md.ns = max(md.ns_dpdk, md.ns_sketch)

    def add_ns_constraints(self, m, md, ns_req=None):
        rows = md.rows_tot
        mem = md.mem_tot

        # Either both should be True or neither should be True
        assert(isinstance(rows, (int, float)) == isinstance(mem, (int, float)))
        if(isinstance(rows, (int, float))):
            md.m_access_time = self.get_mem_access_time(mem)
            md.ns_single = rows * (md.m_access_time + self.hash_ns)

            # If it is None then directly use set_thr function
            # Using assert because there is no need for creating model m
            if(ns_req is not None):
                assert(m is None)  # TODO: can remove later
                return self.set_thr(md, ns_req)

        else:
            # Access time based on mem
            # TODO:: better fits for mem and t
            md.m_access_time = m.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                        name='m_access_time_{}'.format(self))
            m.addGenConstrPWL(mem, md.m_access_time, self.mem_par, self.mem_ns,
                              "mem_access_time_{}".format(self))
            # single core ns model
            # self.t = m.addVar(vtype=GRB.CONTINUOUS, lb=0)
            # m.addGenConstrPWL(rows, self.t, CPU.s_rows, CPU.static_loads)
            md.ns_single = m.addVar(vtype=GRB.CONTINUOUS,
                                    name='ns_single_{}'.format(self))
            md.pdt_m_rows = self.get_pdt_var(md.m_access_time,
                                             rows, 'm_rows', m, 1)
            # m.addConstr(self.t * self.Li_ns[0] + self.pdt_m_rows
            #             + rows * self.hash_ns
            #             <= self.ns_single, name='ns_single_{}'.format(self))
            m.addConstr(md.pdt_m_rows + rows * self.hash_ns
                        == md.ns_single,
                        name='ns_single_{}'.format(self))

        '''
        If rows and mem are not known then m_access_time * rows requires
        non-convexity
        If ns_req is not present then Amdahl's law requires non-convexity
        If both are known then use set_thr
        '''
        m.setParam(GRB.Param.NonConvex, 2)

        # Multi-core model
        md.cores_sketch = m.addVar(vtype=GRB.INTEGER, lb=0, ub=self.cores,
                                   name='cores_sketch_{}'.format(self))
        # Multi-core sketching
        if(ns_req):
            m.addConstr(md.cores_sketch * ns_req >= md.ns_single,
                        name='ns_sketch_{}'.format(self))
        else:
            md.ns_sketch = m.addVar(vtype=GRB.CONTINUOUS,
                                    name='ns_sketch_{}'.format(self), lb=0)
            md.pdt_nsk_c = self.get_pdt_var(md.ns_sketch,
                                            md.cores_sketch, 'nsk_c', m, 0)
            m.addConstr(md.pdt_nsk_c == md.ns_single,
                        name='ns_sketch_{}'.format(self))

        # Amdahl's law
        dpdk_single_ns = 1000/self.dpdk_single_core_thr
        f = CPU.fraction_parallel
        if(ns_req):
            dpdk_cores = f/(ns_req/dpdk_single_ns - 1 + f)
            assert(dpdk_cores > 0)
            md.cores_dpdk = math.ceil(dpdk_cores)
            # m.addConstr(
            #     (md.cores_dpdk*(1-CPU.fraction_parallel)
            #      + CPU.fraction_parallel)*dpdk_single_ns
            #     <= md.cores_dpdk * ns_req, name='ns_dpdk_{}'.format(self))
        else:
            md.cores_dpdk = m.addVar(vtype=GRB.INTEGER, lb=1, ub=self.cores,
                                     name='cores_dpdk_{}'.format(self))
            md.ns_dpdk = m.addVar(vtype=GRB.CONTINUOUS,
                                  name='ns_dpdk_{}'.format(self))
            md.pdt_nsc_dpdk = self.get_pdt_var(
                md.ns_dpdk, md.cores_dpdk, 'nsc_dpdk', m, 0)
            m.addConstr(
                (md.cores_dpdk*(1-f)+f)*dpdk_single_ns
                == md.pdt_nsc_dpdk, name='ns_dpdk_{}'.format(self))

        m.addConstr(md.cores_sketch + md.cores_dpdk <= self.cores,
                    name='capacity_cores_{}'.format(self))

        if(ns_req is None):
            md.ns = m.addVar(vtype=GRB.CONTINUOUS,
                             name='ns_{}'.format(self))
            m.addGenConstrMax(md.ns, [md.ns_dpdk, md.ns_sketch],
                              name='ns_{}'.format(self))

    def res(self, md):
        return 10*(md.cores_dpdk + md.cores_sketch) \
            + md.mem_tot/self.Li_size[2]

    def resource_stats(self, md):
        if(hasattr(md, 'cores_sketch')):
            return "cores_sketch: {}, cores_dpdk: {}".format(
                get_val(md.cores_sketch), get_val(md.cores_dpdk))
        else:
            return ""

    def get_ns(self, md):
        # TODO: Measure the impact of using int here
        mem_tot = math.ceil(get_rounded_val(get_val(md.mem_tot)))
        rows_tot = math.ceil(get_rounded_val(get_val(md.rows_tot)))
        key = (self.profile_name, mem_tot, rows_tot)
        if(key in self.cache):
            # self.cache['helped'] += 1
            return self.cache[key]

        md_tmp = Namespace()
        # mem_tot = u.addVar(vtype=GRB.CONTINUOUS,
        #                    name='mem_tot_{}'.format(d),
        #                    lb=0, ub=d.max_mem)
        # u.addConstr(mem_tot == get_rounded_val(get_val(d.mem_tot)),
        #             name='mem_tot_{}'.format(d))
        # md.mem_tot_old = md.mem_tot
        md_tmp.mem_tot = mem_tot

        # rows_tot = u.addVar(vtype=GRB.CONTINUOUS,
        #                     name='rows_tot_{}'.format(d), lb=0)
        # u.addConstr(rows_tot == d.rows_tot.x,
        #             name='rows_tot_{}'.format(d))
        # md.rows_tot_old = md.rows_tot
        md_tmp.rows_tot = rows_tot

        u = gp.Model(self.name)
        if(not common_config.mipout):
            u.setParam(GRB.Param.LogToConsole, 0)

        self.add_ns_constraints(u, md_tmp)

        u.setObjectiveN(md_tmp.ns, 0, 10, reltol=common_config.ns_tol,
                        name='ns')
        u.setObjectiveN(self.res(md_tmp), 1, 5, reltol=common_config.res_tol,
                        name='res')

        u.ModelSense = GRB.MINIMIZE
        u.update()
        u.optimize()
        # The solver constraints should guarantee that following holds
        assert(u.Status != GRB.Status.INFEASIBLE)

        # TODO: Is the following needed as we can just keep the new values
        # Need to keep these to retain model values.
        # if(hasattr(md, 'u')):
        #     if(hasattr(md, 'old_u')):
        #         md.old_u.append(md.u)
        #     else:
        #         md.old_u = [md.u]
        # md.u = u
        log_vars(u)
        # return get_val(md_tmp.ns)
        self.cache[key] = get_val(md_tmp.ns)
        return self.cache[key]

    def __init__(self, *args, **kwargs):
        super(CPU, self).__init__(*args, **kwargs)
        from scipy.interpolate import interp1d
        # TODO:: Can memoize, See pickle
        # self.get_mem_access_time = memoize(interp1d(self.mem_par, self.mem_ns))
        self.get_mem_access_time = interp1d(self.mem_par, self.mem_ns)
        # self.weight = 10


class P4(device):

    def add_ns_constraints(self, m, md, ns_req=None):
        md.ns = 1000 / self.line_thr
        if(ns_req):
            assert(isinstance(md.rows_tot, (int, float))
                   == isinstance(md.mem_tot, (int, float)))
            if(isinstance(md.rows_tot, (int, float))):
                assert(m is None)  # TODO: can remove later
            assert(md.ns <= ns_req)
        # md.ns = m.addVar(vtype=GRB.CONTINUOUS, name='ns_{}'.format(self))
        # m.addConstr(md.ns == 1000 / self.line_thr, name='ns_{}'.format(self))

        # TODO: can be removed
        # as will be always true in our cases
        # if(ns_req):
        #     m.addConstr(md.ns <= ns_req, name='max_ns_{}'.format(self))

    def res(self, md):
        return md.rows_tot/self.meter_alus + md.mem_tot/self.sram

    def __init__(self, *args, **kwargs):
        super(P4, self).__init__(*args, **kwargs)
        # self.weight = 1


class Cluster(device):

    # Tree of devices

    @memoize
    def transitive_closure(self):
        closure = []
        self.overlay_closure = []
        for child in self.device_tree:
            if(not isinstance(child, Cluster)):
                closure.append(child)
            else:
                closure.extend(child.transitive_closure())
        return closure

    @memoize
    def dev_id_to_cluster_id(self, inp):
        this_dict = {}
        clusters = self.device_tree
        for (cnum, c) in enumerate(clusters):
            if(isinstance(c, Cluster)):
                for d in c.transitive_closure():
                    this_dict[d.dev_id] = cnum
            else:
                d = c
                this_dict[d.dev_id] = cnum

        return this_dict

    @property
    @memoize
    def max_mem(self):
        mm = 0
        for d in self.transitive_closure():
            mm += d.max_mem
        return mm

    @property
    @memoize
    def max_rows(self):
        mr = 0
        for d in self.transitive_closure():
            mr += d.max_rows
        return mr

    @property
    @memoize
    def name(self):
        name_str = '{' + ','.join(d.name for d in self.device_tree) + '}'
        if(len(name_str) > 200):
            name_str = name_str[:200]
        return name_str

    def res(self, md):
        return md.mem_tot

    @property
    @memoize
    def line_thr(self):
        lthr = self.device_tree[0].line_thr
        for d in self.device_tree:
            lthr = min(lthr, d.line_thr)
        return lthr

    def add_ns_constraints(self, m, md, ns_req=None):
        md.ns = 1000 / self.line_thr
        # m.addVar(vtype=GRB.CONTINUOUS, name='ns_{}'.format(self))
        # m.addConstr(md.ns == 1000 / self.line_thr, name='ns_{}'.format(self))

        if(ns_req):
            assert(m is None)  # TODO: can remove later
            assert(md.ns <= ns_req)

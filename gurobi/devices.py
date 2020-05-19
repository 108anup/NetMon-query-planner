import math
import gurobipy as gp
from gurobipy import GRB

from config import common_config
from common import Namespace, memoize, log
from helpers import get_val, get_rounded_val, is_infeasible


def get_rounded_cores(x):
    f, i = math.modf(x)
    if(f < common_config.ftol):
        return i
    else:
        return i + 1


# * Device
class Device(Namespace):

    def add_ns_constraints(self, m):
        pass

    def __init__(self, *args, **kwargs):
        super(Device, self).__init__(*args, **kwargs)

    def res(self):
        return 0

    def __repr__(self):
        return self.name

    def resource_stats(self, md):
        return ""

    # What is the best ns possible for given placement
    def get_ns(self, md):
        return 1000 / self.line_thr


# * CPU
class CPU(Device):
    fraction_parallel = 1
    # static_loads = [0, 6, 12, 18, 24, 30, 43, 49, 55]
    # s_rows = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    # cache = {}  # HOLD: see if there is more performant cache
    fixed_thr = False
    cols_pwr_2 = False

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

    # What resources are needed for given placement and ns_req
    def set_thr(self, md, ns_req):
        assert(ns_req > 0)
        md.cores_sketch = get_rounded_cores(md.ns_single / ns_req)
        dpdk_single_ns = 1000/self.dpdk_single_core_thr
        if(md.cores_sketch != 0):
            md.ns_sketch = md.ns_single / md.cores_sketch
        else:
            # assert(md.ns_single <= 0.001)
            md.ns_sketch = 0
        f = CPU.fraction_parallel
        den = ns_req/dpdk_single_ns - 1 + f
        if(den <= 0):
            md.infeasible = True
            return
        dpdk_cores = f/den
        md.cores_dpdk = get_rounded_cores(dpdk_cores)
        md.ns_dpdk = dpdk_single_ns * (1-f + f/md.cores_dpdk)
        md.ns = max(md.ns_dpdk, md.ns_sketch)
        if(md.cores_dpdk + md.cores_sketch > self.cores):
            md.infeasible = True

    def add_ns_constraints(self, m, md, ns_req=None):
        # rows = md.rows_tot
        mem = md.mem_tot
        rows_thr = md.rows_thr

        # Either both should be True or neither should be True
        assert(isinstance(rows_thr, (int, float))
               == isinstance(mem, (int, float)))
        if(isinstance(rows_thr, (int, float))):
            md.m_access_time = self.get_mem_access_time(mem)
            md.ns_single = rows_thr * (md.m_access_time + self.hash_ns)

            # If it is None then directly use set_thr function
            # Using assert because there is no need for creating model m
            if(ns_req is not None):
                assert(m is None)  # TODO: can remove later
                return self.set_thr(md, ns_req)
            else:
                assert(False)

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
                                             rows_thr, 'm_rows', m, 1)
            # m.addConstr(self.t * self.Li_ns[0] + self.pdt_m_rows
            #             + rows * self.hash_ns
            #             <= self.ns_single, name='ns_single_{}'.format(self))
            m.addConstr(md.pdt_m_rows + rows_thr * self.hash_ns
                        == md.ns_single,
                        name='ns_single_{}'.format(self))

        '''
        If rows and mem are not known then m_access_time * rows requires
        non-convexity
        If ns_req is not present then Amdahl's law requires non-convexity
        If both are known then use set_thr
        '''

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
            md.cores_dpdk = get_rounded_cores(dpdk_cores)
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
            if(f == 1):
                m.addConstr(
                    dpdk_single_ns
                    == md.pdt_nsc_dpdk, name='ns_dpdk_{}'.format(self))
            else:
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

    def resource_stats(self, md, r=None):
        if(hasattr(md, 'cores_sketch')):
            cores_sketch = get_val(md.cores_sketch)
            cores_dpdk = get_val(md.cores_dpdk)
            if(r):
                # Whenever called with r, assume attrs are set
                r.total_CPUs += 1
                r.used_cores += cores_sketch + cores_dpdk
            return "cores_sketch: {}, cores_dpdk: {}".format(
                cores_sketch, cores_dpdk)
        else:
            assert(r is None)
            return ""

    def log_vars(self, md):
        log.debug("Vars:")
        for k, v in md.__dict__.items():
            if(not isinstance(v, gp.Model)):
                log.debug("{}: {}".format(k, get_val(v)))

    def get_ns(self, md):
        mem_tot = get_rounded_val(get_val(md.mem_tot))
        rows_thr = get_rounded_val(get_val(md.rows_thr))
        m_access_time = self.get_mem_access_time(mem_tot)
        ns_single = rows_thr * (m_access_time + self.hash_ns)
        dpdk_single_ns = 1000/self.dpdk_single_core_thr
        f = CPU.fraction_parallel

        ns_options = []

        def helper_ns(cores_sketch, cores_dpdk):
            if(cores_dpdk >= 1 and cores_sketch >= 0 and
               cores_dpdk + cores_sketch <= self.cores):
                ns_dpdk = dpdk_single_ns * (1-f + f/cores_dpdk)
                if(cores_sketch == 0):
                    if(ns_single == 0):
                        ns_sketch = 0
                        ns_options.append(max(ns_dpdk, ns_sketch))
                else:
                    ns_sketch = ns_single / cores_sketch
                    ns_options.append(max(ns_dpdk, ns_sketch))

        if(f < 1):
            a = dpdk_single_ns * (1-f)
            b = dpdk_single_ns * f
            c = ns_single
            k = self.cores
            x = (a*k+b+c - math.sqrt((a*k+b+c)**2 - 4*a*k*c))/(2 * a)

            # Case 1:
            cores_sketch = int(x)
            cores_dpdk = k - cores_sketch
            helper_ns(cores_sketch, cores_dpdk)

            # Case 2:
            cores_sketch = math.ceil(x)
            cores_dpdk = k - cores_sketch
            helper_ns(cores_sketch, cores_dpdk)

        else:
            # This basically is for OVS which gives
            # almost linear throughput with cores
            ratio = ns_single / dpdk_single_ns

            # Case 1:
            cores_sketch = math.floor(ratio * self.cores / (ratio + 1))
            cores_dpdk = self.cores - cores_sketch
            helper_ns(cores_sketch, cores_dpdk)

            # Case 2:
            cores_sketch = math.ceil(ratio * self.cores / (ratio + 1))
            cores_dpdk = self.cores - cores_sketch
            helper_ns(cores_sketch, cores_dpdk)

        assert(len(ns_options) > 0)
        return min(ns_options)

        # # key = (self.profile_name, mem_tot, rows_tot)
        # # if(key in self.cache):
        # #     # self.cache['helped'] += 1
        # #     return self.cache[key]

        # md_tmp = Namespace()
        # # mem_tot = u.addVar(vtype=GRB.CONTINUOUS,
        # #                    name='mem_tot_{}'.format(d),
        # #                    lb=0, ub=d.max_mem)
        # # u.addConstr(mem_tot == get_rounded_val(get_val(d.mem_tot)),
        # #             name='mem_tot_{}'.format(d))
        # # md.mem_tot_old = md.mem_tot
        # md_tmp.mem_tot = mem_tot

        # # rows_tot = u.addVar(vtype=GRB.CONTINUOUS,
        # #                     name='rows_tot_{}'.format(d), lb=0)
        # # u.addConstr(rows_tot == d.rows_tot.x,
        # #             name='rows_tot_{}'.format(d))
        # # md.rows_tot_old = md.rows_tot
        # md_tmp.rows_tot = rows_tot

        # u = gp.Model(self.name)
        # if(not common_config.mipout):
        #     u.setParam(GRB.Param.LogToConsole, 0)

        # self.add_ns_constraints(u, md_tmp)

        # u.setObjectiveN(md_tmp.ns, 0, 10, reltol=common_config.ns_tol,
        #                 name='ns')
        # u.setObjectiveN(self.res(md_tmp), 1, 5, reltol=common_config.res_tol,
        #                 name='res')

        # u.ModelSense = GRB.MINIMIZE
        # u.update()
        # u.optimize()
        # # The solver constraints should guarantee that following holds
        # assert(not is_infeasible(u))

        # # TODO: Is the following needed as we can just keep the new values
        # # Need to keep these to retain model values.
        # # if(hasattr(md, 'u')):
        # #     if(hasattr(md, 'old_u')):
        # #         md.old_u.append(md.u)
        # #     else:
        # #         md.old_u = [md.u]
        # # md.u = u
        # log_vars(u)
        # return get_val(md_tmp.ns)
        # # self.cache[key] = get_val(md_tmp.ns)
        # # return self.cache[key]

    def __init__(self, *args, **kwargs):
        super(CPU, self).__init__(*args, **kwargs)
        from scipy.interpolate import interp1d
        # TODO:: Can memoize, See pickle
        # self.get_mem_access_time = memoize(interp1d(self.mem_par, self.mem_ns))
        self.get_mem_access_time = interp1d(self.mem_par, self.mem_ns)
        # self.weight = 10


# * Netronome
class Netronome(Device):
    fixed_thr = False
    cols_pwr_2 = True

    def get_pdt_var(self, a, b, pdt_name, m, direction):
        m.update()
        pdt = m.addVar(vtype=GRB.CONTINUOUS,
                       name='pdt_{}_{}'.format(pdt_name, self))
        m.addQConstr(pdt == a * b,
                     name='pdt_{}_{}'.format(pdt_name, self))
        return pdt

    def set_thr(self, md, ns_req):
        md.micro_engines = get_rounded_cores(max(
            md.ns_hash_max * self.total_me / ns_req,
            md.ns_fwd_max * self.total_me / ns_req, self.total_me
        ))
        if(md.micro_engines > self.total_me):
            md.infeasible = True
        md.ns_hash = md.ns_hash_max * self.total_me / md.micro_engines
        md.ns_fwd = md.ns_fwd_max * self.total_me / md.micro_engines
        md.ns = max(md.ns_hash, md.ns_fwd, md.ns_mem_max)

    def add_ns_constraints(self, m, md, ns_req=None):
        rows_thr = md.rows_thr
        mem = md.mem_tot
        md.ns_fwd_max = 1000 / self.line_thr

        assert(isinstance(rows_thr, (int, float))
               == isinstance(mem, (int, float)))
        if(isinstance(rows_thr, (int, float))):
            md.m_access_time = self.get_mem_access_time(mem)
            md.ns_mem_max = self.mem_const + rows_thr * md.m_access_time
            md.ns_hash_max = self.hashing_const + self.hashing_slope * rows_thr

            if(ns_req is not None):
                assert(m is None)
                return self.set_thr(md, ns_req)
            else:
                assert(False)

        else:
            md.m_access_time = m.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                        name='m_access_time_{}'.format(self))
            m.addGenConstrPWL(mem, md.m_access_time, self.mem_par, self.mem_ns,
                              "mem_access_time_{}".format(self))
            md.ns_mem_max = m.addVar(vtype=GRB.CONTINUOUS,
                                     name='ns_mem_max_{}'.format(self))
            md.pdt_m_rows = self.get_pdt_var(md.m_access_time,
                                             rows_thr, 'm_rows', m, 1)
            m.addConstr(self.mem_const + md.pdt_m_rows == md.ns_mem_max,
                        name='ns_mem_max_{}'.format(self))

            md.ns_hash_max = m.addVar(vtype=GRB.CONTINUOUS,
                                      name='ns_hash_max_{}'.format(self))
            m.addConstr(md.ns_hash_max == self.hashing_const
                        + rows_thr * self.hashing_slope,
                        name='ns_hash_max_{}'.format(self))

        '''
        If rows and mem are not known then m_access_time * rows requires
        non-convexity
        If ns_req is not present then Amdahl's law requires non-convexity
        If both are known then use set_thr
        '''

        # Parallelism
        md.micro_engines = m.addVar(vtype=GRB.INTEGER, lb=0, ub=self.total_me,
                                    name='micro_engines_{}'.format(self))
        if(ns_req):
            m.addConstr(ns_req * md.micro_engines >=
                        md.ns_hash_max * self.total_me,
                        name='ns_req_hash_{}'.format(self))
            m.addConstr(md.ns_mem_max <= ns_req,
                        name='ns_req_mem_{}'.format(self))
            m.addConstr(ns_req * md.micro_engines >=
                        md.ns_fwd_max * self.total_me,
                        name='ns_req_fwd_{}'.format(self))
        else:
            md.ns_hash = m.addVar(vtype=GRB.CONTINUOUS,
                                  name='ns_hash_{}'.format(self), lb=0)
            md.ns_mem = m.addVar(vtype=GRB.CONTINUOUS,
                                 name='ns_mem_{}'.format(self), lb=0)
            md.ns_fwd = m.addVar(vtype=GRB.CONTINUOUS,
                                 name='ns_fwd_{}'.format(self), lb=0)
            md.ns = m.addVar(vtype=GRB.CONTINUOUS,
                             name='ns_{}'.format(self))
            md.pdt_ns_hash_me = self.get_pdt_var(
                md.ns_hash_max, md.micro_engines,
                'ns_hash_me', m, 0)
            md.pdt_ns_fwd_me = self.get_pdt_var(
                md.ns_fwd_max, md.micro_engines,
                'ns_fwd_me', m, 0)
            m.addConstr(md.pdt_ns_hash_me == md.ns_hash_max * self.total_me)
            m.addConstr(md.pdt_ns_fwd_me == md.ns_fwd_max * self.total_me)
            m.addGenConstrMax(md.ns, [md.ns_hash, md.ns_mem, md.ns_fwd],
                              name='ns_{}'.format(self))

    def res(self, md):
        return (common_config.ME_WEIGHT * (md.micro_engines)
                + md.mem_tot/self.emem_size)

    def resource_stats(self, md, r=None):
        if(hasattr(md, 'micro_engines')):
            val = get_val(md.micro_engines)
            if(r):
                r.nic_memory += get_val(md.mem_tot)
                r.micro_engines += val
            return "micro_engines: {}".format(val)
        else:
            return ""

    def get_ns(self, md):
        mem_tot = get_rounded_val(get_val(md.mem_tot))
        rows_thr = get_rounded_val(get_val(md.rows_thr))

        m_access_time = self.get_mem_access_time(mem_tot)
        ns_mem_max = self.mem_const + rows_thr * m_access_time
        ns_hash_max = self.hashing_const + self.hashing_slope * rows_thr
        ns_fwd_max = 1000 / self.line_thr
        return max(ns_hash_max, ns_fwd_max, ns_mem_max)

        # md_tmp = Namespace()
        # md_tmp.mem_tot = mem_tot
        # md_tmp.rows_tot = rows_tot

        # u = gp.Model(self.name)
        # if(not common_config.mipout):
        #     u.setParam(GRB.Param.LogToConsole, 0)

        # self.add_ns_constraints(u, md_tmp)

        # u.setObjectiveN(md_tmp.ns, 0, 10, reltol=common_config.ns_tol,
        #                 name='ns')
        # u.setObjectiveN(self.res(md_tmp), 1, 5, reltol=common_config.res_tol,
        #                 name='res')

        # u.ModelSense = GRB.MINIMIZE
        # u.update()
        # u.optimize()
        # # The solver constraints should guarantee that following holds
        # assert(not is_infeasible(u))

        # log_vars(u)
        # return get_val(md_tmp.ns)

    def __init__(self, *args, **kwargs):
        super(Netronome, self).__init__(*args, **kwargs)
        from scipy.interpolate import interp1d
        self.get_mem_access_time = interp1d(self.mem_par, self.mem_ns)


# * P4
class P4(Device):
    fixed_thr = True
    cols_pwr_2 = True

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

    def resource_stats(self, md, r=None):
        if(hasattr(md, 'mem_tot') and r):
            r.switch_memory += get_val(md.mem_tot)
        return ""

    def __init__(self, *args, **kwargs):
        super(P4, self).__init__(*args, **kwargs)
        # self.weight = 1


# * Cluster
class Cluster(Device):
    # Tree of devices
    # HOLD: should Clusters have cols_pwr_2?
    # Ideally clusters will have some CPUs, so not needed for now
    cols_pwr_2 = False

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

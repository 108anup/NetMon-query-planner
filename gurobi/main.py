import sys
from queue import Queue
import time
import ipdb
import traceback

from cli import generate_parser
from common import Namespace, setup_logging, log_time, log
from config import common_config
from input import input_generator
from solvers import (refine_devices, solver_to_class, log_results,
                     log_placement, UnivmonGreedyRows)
from devices import Cluster
from flows import flow
from gurobipy import GRB, tuplelist, tupledict


@log_time
def get_partitions(queries):
    partitions = []
    for (i, q) in enumerate(queries):
        q.sketch_id = i
        num_rows = q.rows()
        start_idx = len(partitions)
        if(common_config.horizontal_partition):
            q.partitions = [start_idx + r for r in range(num_rows)]
            partitions += [Namespace(partition_id=start_idx+r,
                                     sketch=q, num_rows=1)
                           for r in range(num_rows)]
        else:
            q.partitions = [start_idx]
            partitions += [Namespace(partition_id=start_idx, sketch=q,
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


# DFS over overlay tree
def get_cluster_from_overlay(inp, overlay):
    if(isinstance(overlay, list)):
        return Cluster(
            device_tree=[
                get_cluster_from_overlay(inp, roots) for roots in overlay
            ],
            overlay=overlay
        )
    else:
        return inp.devices[overlay]


# Side effects on device only, create new devices (clusters) for each
# invocation of solver
@log_time
def get_partitions_flows(inp, cluster, solver, dnum):
    dev_id_to_cluster_id = cluster.dev_id_to_cluster_id(inp)
    partitions = []
    remaining_frac = {}
    p_to_pnum = {}

    id = 0
    for (_, pnum) in solver.dev_par_tuplelist.select(dnum, '*'):
        if(solver.frac[dnum, pnum].x > 0):
            p = solver.partitions[pnum]
            partitions.append(p)
            p_to_pnum[p] = id
            id += 1
            remaining_frac[p] = solver.frac[dnum, pnum].x

    flows = []
    for f in inp.flows:
        f_new = flow()
        par_new = []
        for fp in f.partitions:
            pnum = fp[0]
            cov = fp[1]
            p = inp.partitions[pnum]
            if(p in partitions):
                cov_new = min(remaining_frac[p], cov)
                par_new.extend([(p_to_pnum[p], cov_new)])

        path_new = set()
        for dnum in f.path:
            if(dnum in dev_id_to_cluster_id):
                path_new.add(dev_id_to_cluster_id[dnum])

        if(len(par_new) > 0 and len(path_new) > 0):
            f_new.partitions = par_new
            f_new.path = path_new
            flows.append(f_new)

    return (partitions, flows)


@log_time
def map_flows_to_cluster(inp):
    dev_id_to_cluster_id = inp.cluster.dev_id_to_cluster_id(inp)
    flows = []
    for f in inp.flows:
        f_new = flow(
            path=set(map(lambda dnum: dev_id_to_cluster_id[dnum], f.path)),
            partitions=f.partitions, queries=f.queries
        )
        flows.append(f_new)
    return flows


@log_time
def map_leaf_solution_to_cluster(solver, cluster):
    dev_id_to_cluster_id = cluster.dev_id_to_cluster_id(inp)
    mem = tupledict()
    frac = tupledict()

    for (dnum, pnum) in solver.dev_par_tuplelist:
        cnum = dev_id_to_cluster_id[dnum]
        mem.setdefault((cnum, pnum), 0)
        frac.setdefault((cnum, pnum), 0)
        mem[cnum, pnum] += solver.mem[dnum, pnum].x
        frac[cnum, pnum] += solver.frac[dnum, pnum].x

    init = Namespace(mem=mem, frac=frac)
    return init


@log_time
def solve(inp):
    #ipdb.set_trace()
    start = time.time()

    for (dnum, d) in enumerate(inp.devices):
        d.dev_id = dnum
    inp.partitions = get_partitions(inp.queries)
    map_flows_partitions(inp.flows, inp.queries)
    Solver = solver_to_class[common_config.solver]

    # Clustering
    if (hasattr(inp, 'overlay')):
        if(common_config.init is True):
            solver = UnivmonGreedyRows(devices=inp.devices,
                                       partitions=inp.partitions,
                                       flows=inp.flows, queries=inp.queries,
                                       overlay=True)
            solver.solve()

        inp.cluster = get_cluster_from_overlay(inp, inp.overlay)
        if(common_config.init is True):
            init = map_leaf_solution_to_cluster(solver, inp.cluster)

        queue = Queue()
        flows = map_flows_to_cluster(inp)
        queue.put(Namespace(devices=inp.cluster.device_tree,
                            partitions=inp.partitions,
                            flows=flows))

        placement = Namespace(frac=tupledict(), mem=tupledict(), res={},
                              dev_par_tuplelist=tuplelist())
        # BFS over device tree
        first_run = True
        while(queue.qsize() > 0):
            front = queue.get()
            devices = front.devices
            partitions = front.partitions
            flows = front.flows
            #import ipdb; ipdb.set_trace()
            solver = Solver(devices=devices, partitions=partitions,
                            flows=flows, queries=inp.queries, overlay=True)
            if(first_run and common_config.init):
                solver = Solver(devices=devices, partitions=partitions,
                                flows=flows, queries=inp.queries, init=init,
                                overlay=True)
            solver.solve()
            first_run = False

            # TODO: Modify Infeasible handling
            # TODO: This breaks abstraction of any type of solver
            if(solver.m.Status == GRB.INFEASIBLE):
                return

            #import ipdb; ipdb.set_trace()
            for (dnum, d) in enumerate(devices):
                if(isinstance(d, Cluster)):
                    (partitions, flows) = get_partitions_flows(
                        inp, d, solver, dnum)
                    if(len(partitions) > 0 and len(flows) > 0):
                        queue.put(Namespace(devices=d.device_tree,
                                            partitions=partitions,
                                            flows=flows))
                else:
                    # Clusters never overlap!!
                    for (_, pnum)in solver.dev_par_tuplelist.select(dnum, '*'):
                        p = solver.partitions[pnum]
                        placement.frac[d.dev_id, p.partition_id] \
                            = solver.frac[dnum, pnum].x
                        placement.mem[d.dev_id, p.partition_id] \
                            = solver.mem[dnum, pnum].x
                        placement.res[d] = d.res().getValue()
                        placement.dev_par_tuplelist.append(
                            (d.dev_id, p.partition_id))
        log.info('-'*50)
        log.info('Clustered Optimization complete')
        log.info('-'*50)

        refine_devices(inp.devices)
        log_results(inp.devices)
        log_placement(inp.devices, inp.partitions, inp.flows,
                      placement.dev_par_tuplelist, placement.frac)

    else:
        solver = Solver(devices=inp.devices, flows=inp.flows,
                        partitions=inp.partitions, queries=inp.queries,
                        overlay=False)
        solver.solve()

    # Move this outside this function
    end = time.time()
    if(not (common_config.output_file is None)):
        f = open(common_config.output_file, 'a')
        f.write("{:06f}, ".format(end - start))
        f.close()


parser = generate_parser()
args = parser.parse_args(sys.argv[1:])
if(hasattr(args, 'config_file')):
    for fpath in args.config_file:
        common_config.load_config_file(fpath)
common_config.update(args)
setup_logging(common_config)

input_num = common_config.input_num
inp = input_generator[input_num]

try:
    solve(inp)
except:
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    ipdb.post_mortem(tb)

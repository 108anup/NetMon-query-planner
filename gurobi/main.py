import sys
from queue import Queue

from cli import generate_parser
from common import Namespace, setup_logging, log_time, log
from config import common_config
from input import input_generator
from solvers import solver_to_class, get_val
from devices import Cluster, CPU, P4
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
    # import ipdb; ipdb.set_trace()
    dev_id_to_cluster_id = inp.cluster.dev_id_to_cluster_id(inp)
    flows = []
    for f in inp.flows:
        f_new = flow(path=set(map(lambda dnum: dev_id_to_cluster_id[dnum], f.path)),
                     partitions=f.partitions, queries=f.queries)
        flows.append(f_new)
    return flows


@log_time
def solve(inp):

    for (dnum, d) in enumerate(inp.devices):
        d.dev_id = dnum
    inp.partitions = get_partitions(inp.queries)
    map_flows_partitions(inp.flows, inp.queries)
    Solver = solver_to_class[common_config.solver]

    # Clustering
    if (hasattr(inp, 'overlay')):
        inp.cluster = get_cluster_from_overlay(inp, inp.overlay)
        queue = Queue()
        flows = map_flows_to_cluster(inp)
        queue.put(Namespace(devices=inp.cluster.device_tree,
                            partitions=inp.partitions,
                            flows=flows))

        placement = Namespace(frac=tupledict(), mem=tupledict(), res={},
                              dev_par_tuplelist=tuplelist())
        # BFS over device tree
        while(queue.qsize() > 0):
            front = queue.get()
            devices = front.devices
            partitions = front.partitions
            flows = front.flows
            solver = Solver(devices=devices, partitions=partitions,
                            flows=flows, queries=inp.queries)
            solver.solve()

            # TODO: This breaks abstraction of any type of solver
            if(solver.m.Status == GRB.INFEASIBLE):
                return

            for (dnum, d) in enumerate(devices):
                if(isinstance(d, Cluster)):
                    (partitions, flows) = get_partitions_flows(
                        inp, d, solver, dnum)
                    queue.put(Namespace(devices=d.device_tree,
                                        partitions=partitions,
                                        flows=flows))
                else:
                    # Clusters never overlap!!
                    for (_, pnum) in solver.dev_par_tuplelist.select(dnum, '*'):
                        p = solver.partitions[pnum]
                        placement.frac[d.dev_id, p.partition_id] = solver.frac[dnum, pnum].x
                        placement.mem[d.dev_id, p.partition_id] = solver.mem[dnum, pnum].x
                        placement.res[d] = d.res().getValue()
                        placement.dev_par_tuplelist.append((d.dev_id, p.partition_id))
        log.info('-'*50)
        log.info('Clustered Optimization complete')
        log.info('-'*50)

        log_results(inp, placement)
        log_placement(inp, placement)

    else:
        solver = Solver(devices=inp.devices, flows=inp.flows,
                        partitions=inp.partitions, queries=inp.queries)
        solver.solve()


# TODO:: Redundancy in Output Logging!!
def log_results(inp, placement):
    ns_max = 0
    res = 0
    total_CPUs = 0
    used_cores = 0
    switch_memory = 0
    for d in inp.devices:
        # import ipdb; ipdb.set_trace()
        ns_max = max(ns_max, get_val(d.ns))
        res += placement.res[d]
        # TODO:: There is a lot of redundancy here, classes calculate this
        if(isinstance(d, CPU)):
            total_CPUs += 1
            used_cores += d.cores_sketch.x + d.cores_dpdk.x
        if(isinstance(d, P4)):
            switch_memory += d.mem_tot.x

    log.info("\nThroughput: {} Mpps, ns per packet: {}".format(
        1000/ns_max, ns_max))
    log.info("Resources: {}".format(res))
    log.info("Used Cores: {}, Total CPUS: {}, Switch Memory: {}"
             .format(used_cores, total_CPUs, switch_memory))


def log_placement(inp, placement):
    for (pnum, p) in enumerate(inp.partitions):
        q = p.sketch
        log.info("\nPartition of Sketch ({}) ({})".format(q.sketch_id,
                                                          q.details()))
        par_info = ""
        total_frac = 0
        for (dnum, _) in placement.dev_par_tuplelist.select('*', pnum):
            par_info += "({:0.3f},{})    ".format(placement.frac[dnum, pnum], dnum)
            total_frac += (placement.frac[dnum, pnum])
        log.info(par_info)
        log.info("Total frac: {:0.3f}".format(total_frac))

    for (dnum, d) in enumerate(inp.devices):
        log.info("\nDevice ({}) {}:".format(dnum, d))
        res_stats = d.resource_stats()
        if(res_stats != ""):
            log.info(res_stats)
        log.info("Rows total: {}".format(get_val(d.rows_tot)))
        log.info("Mem total: {}".format(d.mem_tot.x))
        if(hasattr(d, 'ns')):
            log.info("Throughput: {}".format(1000/d.ns.x))

    log.debug("")
    for (fnum, f) in enumerate(inp.flows):
        log.debug("Flow {}:".format(fnum))
        log.debug("partitions: {}".format(f.partitions))
        log.debug("path: {}".format(f.path))
    log.info("-"*50)


parser = generate_parser()
args = parser.parse_args(sys.argv[1:])
if(hasattr(args, 'config_file')):
    for fpath in args.config_file:
        common_config.load_config_file(fpath)
common_config.update(args)
setup_logging(common_config)

input_num = common_config.input_num
inp = input_generator[input_num]
solve(inp)

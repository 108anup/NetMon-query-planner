import os
import sys
import time
import traceback
from queue import Queue

import ipdb
from gurobipy import GRB, tupledict, tuplelist

from cli import generate_parser
from common import Namespace, log, log_time, setup_logging, freeze_object
from config import common_config
from devices import Cluster
from flows import flow
from input import input_generator
from solvers import (UnivmonGreedyRows, log_placement, log_results,
                     refine_devices, solver_to_class)
import threading


def handle_infeasible(m, iis=True, msg="Infeasible Placement!"):
    log.warning(msg)

    if(not (common_config.results_file is None)):
        f = open(common_config.results_file, 'a')
        f.write("-, -, -, -, -, ")
        f.close()

    if(common_config.prog_dir and iis):
        m.computeIIS()
        m.write(
            os.path.join(
                common_config.prog_dir,
                "infeasible_{}.ilp".format(common_config.input_num)
            )
        )
    return None


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


def get_2_level_overlay(overlay):
    output = []
    for l in overlay:
        if(isinstance(l, list)):
            no_nesting = True
            for e in l:
                if(isinstance(e, list)):
                    no_nesting = False
            if(no_nesting):
                output.append(l)
            else:
                output.extend(get_2_level_overlay(l))
        else:
            output.append(l)
    return output


@log_time
def get_subproblems(inp, solver):
    leaves_overlay = get_2_level_overlay(inp.overlay)
    leaves_cluster = get_cluster_from_overlay(inp, leaves_overlay)

    cluster_list = leaves_cluster.device_tree
    problems = []
    # import ipdb; ipdb.set_trace()
    for c in cluster_list:
        subproblem = Namespace()

        if(isinstance(c, Cluster)):
            devices = c.transitive_closure()
        else:
            devices = [c]
        dev_ids = set(d.dev_id for d in devices)
        dev_id_to_dnum = {}
        for dnum, d in enumerate(devices):
            dev_id_to_dnum[d.dev_id] = dnum

        # TODO:: optimize to only include relevant partitions
        partitions = set()
        flows = []
        for f in inp.flows:
            new_path = set()
            new_par = []
            for fp in f.partitions:
                pnum = fp[0]
                cov = fp[1]
                tot_frac = 0
                for dnum in f.path:
                    if(dnum in dev_ids):
                        new_path.add(dev_id_to_dnum[dnum])
                        tot_frac += solver.frac[dnum, pnum].x
                if(tot_frac > 0):
                    new_par.extend([(pnum, min(cov, tot_frac))])
            if(len(new_path) > 0 and len(new_par) > 0):
                flows.append(flow(
                    path=new_path,
                    partitions=new_par))

        subproblem.devices = devices
        subproblem.partitions = inp.partitions
        subproblem.flows = flows
        if(len(flows) > 0):
            problems.append(subproblem)
    return problems


@log_time
def init_leaf_solution_to_cluster(solver, cluster):
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


def log_step(msg, logger=log.info):
    logger('='*50)
    logger("STEP: " + msg)
    logger('='*50)


def cluster_refinement(inp):
    Solver = solver_to_class[common_config.solver]

    dont_refine = False
    if(common_config.solver == 'Netmon'):
        log_step("Running UnivmonGreedyRows over full topology")
        dont_refine = True
    solver = UnivmonGreedyRows(devices=inp.devices,
                               partitions=inp.partitions,
                               flows=inp.flows, queries=inp.queries,
                               dont_refine=dont_refine)
    solver.solve()
    if(solver.infeasible):
        return handle_infeasible(solver.culprit)

    if(common_config.solver == 'Netmon'):
        log_step("Refining clusters")
        subproblems = get_subproblems(inp, solver)

        frac = tupledict()
        md_list = [Namespace() for i in range(len(inp.devices))]
        for prob in subproblems:
            sol = Solver(devices=prob.devices, partitions=prob.partitions,
                         flows=prob.flows, queries=inp.queries,
                         dont_refine=False)
            sol.solve()
            if(sol.infeasible):
                return handle_infeasible(sol.culprit)
            log_results(sol.devices, msg="Refinement Results")
            frac.update(sol.frac)
            for (dnum, d) in enumerate(prob.devices):
                md_list[d.dev_id] = sol.md_list[dnum]

        log_step("Selective Refinement Complete")
        r = refine_devices(inp.devices, md_list)
        ret = Namespace(results=r, md_list=md_list)
        log_placement(inp.devices, inp.partitions, inp.flows,
                      solver.dev_par_tuplelist, frac, md_list)
    else:
        ret = Namespace(results=solver.r, md_list=solver.md_list)

    return ret


def cluster_optimization(inp):
    assert(not (common_config.parallel and common_config.init))
    Solver = solver_to_class[common_config.solver]

    if(common_config.init is True):
        log_step("Running UnivmonGreedyRows over full topology")
        solver = UnivmonGreedyRows(devices=inp.devices,
                                   partitions=inp.partitions,
                                   flows=inp.flows, queries=inp.queries,
                                   dont_refine=True)
        solver.solve()
        if(solver.infeasible):
            return handle_infeasible(solver.culprit)

    inp.cluster = get_cluster_from_overlay(inp, inp.overlay)
    if(common_config.init is True):
        init = init_leaf_solution_to_cluster(solver, inp.cluster)

    queue = Queue()
    flows = map_flows_to_cluster(inp)
    queue.put(Namespace(devices=inp.cluster.device_tree,
                        partitions=inp.partitions,
                        flows=flows))

    placement = Namespace(frac=tupledict(), mem=tupledict(), res={},
                          dev_par_tuplelist=tuplelist(),
                          md_list=[Namespace()
                                   for i in range(len(inp.devices))])

    # BFS over device tree
    if(common_config.parallel):
        def solver_thread(front):
            devices = front.devices
            partitions = front.partitions
            flows = front.flows
            """
            Python takes a lot of locks while running threads
            Assuming Gurobi spawn a independent dedicated process
            We should be able to overlap the solving times
            """
            solver = Solver(devices=devices, partitions=partitions,
                            flows=flows, queries=inp.queries,
                            dont_refine=True)
            solver.solve()

            # TODO:: check if this will work with multiple threads
            # TODO:: Try futures in python!!
            if(solver.infeasible):
                return handle_infeasible(solver.culprit)

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
                        # placement.res[d] = d.res().getValue()
                        placement.dev_par_tuplelist.append(
                            (d.dev_id, p.partition_id))
                        placement.md_list[d.dev_id] = solver.md_list[dnum]

        # Don't support init here
        while(queue.qsize() > 0):
            threads = []
            num_threads = queue.qsize()
            for thread_id in range(num_threads):
                front = queue.get()
                threads.append(
                    threading.Thread(target=solver_thread, args=(front, ))
                )
            for th in threads:
                th.start()
            for th in threads:
                th.join()

    if(not common_config.parallel):
        first_run = True
        while(queue.qsize() > 0):
            front = queue.get()
            devices = front.devices
            partitions = front.partitions
            flows = front.flows
            if(first_run and common_config.init):
                solver = Solver(devices=devices, partitions=partitions,
                                flows=flows, queries=inp.queries,
                                init=init, dont_refine=True)
            else:
                solver = Solver(devices=devices, partitions=partitions,
                                flows=flows, queries=inp.queries,
                                dont_refine=True)
            solver.solve()
            first_run = False

            if(solver.infeasible):
                return handle_infeasible(solver.culprit)

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
                    for (_, pnum)in solver.dev_par_tuplelist.select(
                            dnum, '*'):
                        p = solver.partitions[pnum]
                        placement.frac[d.dev_id, p.partition_id] \
                            = solver.frac[dnum, pnum].x
                        placement.mem[d.dev_id, p.partition_id] \
                            = solver.mem[dnum, pnum].x
                        # placement.res[d] = d.res().getValue()
                        placement.dev_par_tuplelist.append(
                            (d.dev_id, p.partition_id))
                        placement.md_list[d.dev_id] = solver.md_list[dnum]

    log_step('Clustered Optimization complete')

    r = refine_devices(inp.devices, placement.md_list)
    # TODO:: Put intermediate output to debug!
    # Allow loggers to take input logging level
    log_placement(inp.devices, inp.partitions, inp.flows,
                  placement.dev_par_tuplelist, placement.frac,
                  placement.md_list)
    return Namespace(results=r, md_list=placement.md_list)


# TODO:: Handle disconnected graph in solver
# Final devices will always be refined, just log at the end
@log_time(logger=log.info)
def solve(inp):
    # import ipdb; ipdb.set_trace()
    start = time.time()

    for (dnum, d) in enumerate(inp.devices):
        d.dev_id = dnum
        freeze_object(d)
    inp.partitions = get_partitions(inp.queries)
    map_flows_partitions(inp.flows, inp.queries)
    Solver = solver_to_class[common_config.solver]

    if(getattr(inp, 'refine', None) and getattr(inp, 'overlay', None)):
        ret = cluster_refinement(inp)
    elif (getattr(inp, 'overlay', None)):
        ret = cluster_optimization(inp)
    else:
        solver = Solver(devices=inp.devices, flows=inp.flows,
                        partitions=inp.partitions, queries=inp.queries,
                        dont_refine=False)
        solver.solve()
        if(solver.infeasible):
            return handle_infeasible(solver.culprit)
        ret = Namespace(results=solver.r, md_list=solver.md_list)

    if(ret is None):
        return None

    end = time.time()
    log.info("\n" + "-"*80)
    log_results(inp.devices, ret.results, ret.md_list,
                elapsed=end-start, msg="Final Results")
    # log.info("Memoization resolved {} cases.".format(CPU.cache['helped']))
    return ret


if(__name__ == '__main__'):
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
    except Exception:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)

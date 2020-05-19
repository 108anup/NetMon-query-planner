import concurrent.futures
import os
import sys
# import threading
import time
import traceback
from queue import Queue
# from pathos.pools import ProcessPool

import matplotlib.pyplot as plt
import networkx as nx
from gurobipy import GRB, tupledict, tuplelist
import gurobipy as gp
from orderedset import OrderedSet

from cli import generate_parser
from common import Namespace, freeze_object, log, log_time, setup_logging
from config import common_config
from devices import P4, Cluster
from flows import flow
from input import Input, get_graph
from input_generator import input_generator
from solvers import (UnivmonGreedyRows, log_placement, log_results,
                     refine_devices, solver_to_class)
from helpers import get_val


# * Helper functions
# Run in child process
def extract_solution(solver):
    num_devices = len(solver.devices)
    solution = Namespace(frac=list(range(num_devices)),
                         mem=list(range(num_devices)),
                         dev_par_tuplelist=list(range(num_devices)),
                         md_list=list(range(num_devices)))
    for dnum in range(num_devices):
        solution.dev_par_tuplelist[dnum] = list()
        solution.frac[dnum] = dict()
        solution.mem[dnum] = dict()
        solution.md_list[dnum] = Namespace()
        for k, v in solver.md_list[dnum].__dict__.items():
            if(not isinstance(v, gp.Model)):
                solution.md_list[dnum].__dict__[k] = get_val(v)

    for (dnum, pnum) in solver.dev_par_tuplelist:
        solution.dev_par_tuplelist[dnum].append(pnum)
        solution.frac[dnum][pnum] = solver.frac[dnum, pnum].x
        solution.mem[dnum][pnum] = solver.mem[dnum, pnum].x

    return solution


# Run in parent process
def rebuild_solution(solution):
    new_solution = Namespace(frac=tupledict(),
                             mem=tupledict(),
                             dev_par_tuplelist=tuplelist(),
                             md_list=solution.md_list)

    for dnum, pnums in enumerate(solution.dev_par_tuplelist):
        for pnum in pnums:
            new_solution.dev_par_tuplelist.append((dnum, pnum))
            new_solution.frac[dnum, pnum] = solution.frac[dnum][pnum]
            new_solution.mem[dnum, pnum] = solution.mem[dnum][pnum]

    return new_solution


def runner(solver):
    solver.solve()
    if(solver.infeasible):
        # myinp = Input(devices=solver.devices, flows=solver.flows)
        # g = get_graph(myinp)
        # labels = {}
        # for dnum, d in enumerate(solver.devices):
        #     labels[dnum] = d.name
        # nx.draw(g, labels=labels)
        # plt.show()
        # import ipdb; ipdb.set_trace()
        # return handle_infeasible(solver.culprit)
        solution = Namespace(infeasible=True,
                             reason=solver.reason)
        return solution
    return extract_solution(solver)


def handle_infeasible(m=None, iis=True, msg="Infeasible Placement!"):

    # import ipdb; ipdb.set_trace()
    log.warning(msg)

    if(not (common_config.results_file is None)):
        f = open(common_config.results_file, 'a')
        f.write("-, -, -, -, -, ")
        f.close()

    if(common_config.prog_dir and iis and m):
        m.computeIIS()
        m.write(
            os.path.join(
                common_config.prog_dir,
                "infeasible_{}.ilp".format(common_config.input_num)
            )
        )
    return None


@log_time
def get_partitions(queries, correction=0):
    partitions = []
    for (i, q) in enumerate(queries):
        q.sketch_id = i + correction
        num_rows = q.rows()
        start_idx = len(partitions) + correction
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
def get_partitions_flows(inp, cluster, problem, dnum, solution):
    dev_id_to_cluster_id = cluster.dev_id_to_cluster_id(inp)
    partitions = []
    remaining_frac = {}
    p_to_pnum = {}

    id = 0
    for (_, pnum) in solution.dev_par_tuplelist.select(dnum, '*'):
        if(get_val(solution.frac[dnum, pnum]) > 0):
            p = problem.partitions[pnum]
            partitions.append(p)
            p_to_pnum[p] = id
            id += 1
            remaining_frac[p] = get_val(solution.frac[dnum, pnum])

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

        # Using set here because multiple devices
        # on the path can be in the same cluster
        path_new = OrderedSet()
        for dnum in f.path:
            if(dnum in dev_id_to_cluster_id):
                """
                This optimization basically keeps the sketch in the cluster
                where origin and destinations are instead of keeping in
                a cluster where only the switch is there"""
                d = inp.devices[dnum]
                cnum = dev_id_to_cluster_id[dnum]
                c = cluster.device_tree[cnum]
                if(isinstance(d, P4) and isinstance(c, Cluster)):
                    pass
                else:
                    path_new.add(dev_id_to_cluster_id[dnum])

        if(len(par_new) > 0 and len(path_new) > 0):
            f_new.partitions = par_new
            f_new.path = path_new
            f_new.thr = f.thr
            flows.append(f_new)

    return (partitions, flows)


@log_time
def map_flows_to_cluster(inp):
    dev_id_to_cluster_id = inp.cluster.dev_id_to_cluster_id(inp)
    devices = inp.devices
    flows = []
    for f in inp.flows:
        """
        This optimization basically keeps the sketch in the cluster
        where origin and destinations are instead of keeping in
        a cluster where only the switch is there"""
        f_new = flow(
            path=[dev_id_to_cluster_id[dnum]
                  for dnum in f.path
                  if(not isinstance(devices[dnum], P4))],
            partitions=f.partitions, queries=f.queries,
            thr=f.thr
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
            continue
            devices = [c]
        dev_ids = set(d.dev_id for d in devices)
        dev_id_to_dnum = {}
        for dnum, d in enumerate(devices):
            dev_id_to_dnum[d.dev_id] = dnum

        # HOLD: optimize to only include relevant partitions
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
                    partitions=new_par,
                    thr=f.thr))

        subproblem.devices = devices
        subproblem.partitions = inp.partitions
        subproblem.flows = flows
        if(len(flows) > 0):
            problems.append(subproblem)
    return problems


@log_time
def get_new_problem(old_inp, old_solution, additions):

    assert(not common_config.horizontal_partition)
    assert(getattr(additions, 'devices', None) is None)
    relevant_devices = set()
    old_devices = old_inp.devices
    for f in additions.flows:
        for dnum in f.path:
            relevant_devices.add(old_devices[dnum])

    new_inp = Input()
    new_inp.devices = list(relevant_devices)

    dev_id_to_dnum = {}
    for dnum, d in enumerate(new_inp.devices):
        dev_id_to_dnum[d.dev_id] = dnum

    tmp_flows = []
    # relevant_partitions = set()
    for f in old_inp.flows:
        new_path = set()
        new_par = []
        for fp in f.partitions:
            pnum = fp[0]
            cov = fp[1]
            tot_frac = 0
            for dev_id in f.path:
                new_path.add(dev_id_to_dnum[dev_id])
                tot_frac += get_val(old_solution.frac[dev_id, pnum])
            if(tot_frac > 0):
                new_par.extend([(pnum, min(cov, tot_frac))])
        if(len(new_path) > 0 and len(new_par) > 0):
            # for pnum in new_par:
            #     relevant_partitions.add(pnum)
            tmp_flows.append(flow(
                path=new_path,
                partitions=new_par,
                thr=f.thr))

    # new_inp.partitions = list(relevant_partitions)
    # p_id_to_pnum = {}
    # for pnum, p in enumerate(new_inp.partitions):
    #     p_id_to_pnum[p.partition_id] = pnum

    # # Convert old p_ids to new pnums
    # new_flows = []
    # for f in tmp_flows:
    #     new_flows.append(
    #         path=f.path, thr=f.thr,
    #         partitions=list(
    #             map(lambda x: tuple((p_id_to_pnum[x[0]], x[1])), f.partitions)
    #         )
    #     )

    corr = len(old_inp.partitions)
    new_queries = additions.queries
    new_partitions = get_partitions(new_queries, correction=corr)
    new_inp.queries = old_inp.queries + new_queries
    new_inp.partitions = old_inp.partitions + new_partitions

    # Assumed that no horizontal partitioning
    new_flows = tmp_flows
    for f in additions.flows:
        new_flows.append(flow(
            path=list(
                map(lambda x: dev_id_to_dnum[x], f.path)
            ),
            partitions=[(x[0] + corr, x[1]) for x in f.queries],
            thr=f.thr
        ))

    new_inp.flows = new_flows
    return new_inp


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


# * Clustering Optimizations
# ** Cluster refinement
def cluster_refinement(inp):
    Solver = solver_to_class[common_config.solver]

    # dont_refine = False
    if(common_config.solver == 'Netmon'):
        log_step("Running UnivmonGreedyRows over full topology")
        # dont_refine = True
    solver = UnivmonGreedyRows(devices=inp.devices,
                               partitions=inp.partitions,
                               flows=inp.flows, queries=inp.queries,
                               dont_refine=False)
    solver.solve()
    if(solver.infeasible):
        return handle_infeasible(solver.culprit, solver.reason)

    log_results(inp.devices, solver.r, solver.md_list,
                msg="UnivmonGreedyRows Results")

    md_list = solver.md_list
    frac = solver.frac
    if(common_config.solver == 'Netmon'):
        log_step("Refining clusters")
        subproblems = get_subproblems(inp, solver)

        for prob in subproblems:
            sol = Solver(devices=prob.devices, partitions=prob.partitions,
                         flows=prob.flows, queries=inp.queries,
                         dont_refine=False, ns_req=solver.r.ns_max)
            sol.solve()
            if(sol.infeasible):
                return handle_infeasible(sol.culprit)
            log_results(sol.devices, sol.r, sol.md_list,
                        msg="Refinement Results")
            frac.update(sol.frac)
            for (dnum, d) in enumerate(prob.devices):
                md_list[d.dev_id] = sol.md_list[dnum]

        log_step("Selective Refinement Complete")
        # import ipdb; ipdb.set_trace()
        r = refine_devices(inp.devices, md_list)
        ret = Namespace(results=r, md_list=md_list)
        log_placement(inp.devices, inp.partitions, inp.flows,
                      solver.dev_par_tuplelist, frac, md_list)
    else:
        ret = Namespace(results=solver.r, md_list=solver.md_list)

    return ret


# ** Cluster optimization
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
            return handle_infeasible(getattr(solver, 'culprit', None))

    inp.cluster = get_cluster_from_overlay(inp, inp.overlay)
    if(common_config.init is True):
        init = init_leaf_solution_to_cluster(solver, inp.cluster)

    flows = map_flows_to_cluster(inp)
    queue = Queue()
    queue.put(Namespace(devices=inp.cluster.device_tree,
                        partitions=inp.partitions,
                        flows=flows))

    placement = Namespace(frac=tupledict(), mem=tupledict(), res={},
                          dev_par_tuplelist=tuplelist(),
                          md_list=[Namespace()
                                   for i in range(len(inp.devices))])

    def update_solution(problem, solution):
        for (dnum, d) in enumerate(problem.devices):
            if(isinstance(d, Cluster)):
                (partitions, flows) = get_partitions_flows(
                    inp, d, problem, dnum, solution)
                # Either both have something or both have nothing
                assert(len(partitions) > 0 or len(flows) == 0)
                assert(len(partitions) == 0 or len(flows) > 0)
                if(len(partitions) > 0 and len(flows) > 0):
                    queue.put(Namespace(devices=d.device_tree,
                                        partitions=partitions,
                                        flows=flows))
            else:
                # Clusters never overlap!!
                for (_, pnum)in solution.dev_par_tuplelist.select(
                        dnum, '*'):
                    p = problem.partitions[pnum]
                    placement.frac[d.dev_id, p.partition_id] \
                        = solution.frac[dnum, pnum]
                    placement.mem[d.dev_id, p.partition_id] \
                        = solution.mem[dnum, pnum]
                    # placement.res[d] = d.res().getValue()
                    placement.dev_par_tuplelist.append(
                        (d.dev_id, p.partition_id))
                    placement.md_list[d.dev_id] = solution.md_list[dnum]

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
                # myinp = Input(devices=solver.devices, flows=solver.flows)
                # g = get_graph(myinp)
                # labels = {}
                # for dnum, d in enumerate(solver.devices):
                #     labels[dnum] = d.name
                # nx.draw(g, labels=labels)
                # plt.show()
                # import ipdb; ipdb.set_trace()
                return handle_infeasible(getattr(solver, 'culprit', None),
                                         msg=solver.reason)

            for (dnum, d) in enumerate(solver.devices):
                if(isinstance(d, Cluster)):
                    (partitions, flows) = get_partitions_flows(
                        inp, d, solver, dnum, solver)
                    # Either both have something or both have nothing
                    assert(len(partitions) > 0 or len(flows) == 0)
                    assert(len(partitions) == 0 or len(flows) > 0)
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

    # *** Parallel processing
    else:
        # executer = concurrent.futures.ProcessPoolExecutor(
        #     max_workers=common_config.WORKERS)
        # futures = []
        # pool = ProcessPool(nodes=common_config.WORKERS)
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=common_config.WORKERS) as executor:

            while(queue.qsize() > 0):
                problems = []
                while(queue.qsize() > 0):
                    front = queue.get()
                    devices = front.devices
                    partitions = front.partitions
                    flows = front.flows
                    solver = Solver(devices=devices, partitions=partitions,
                                    flows=flows, queries=inp.queries,
                                    dont_refine=True)
                    problems.append(solver)
                    # futures.append(executer.submit(runner, solver))

                solutions = list(executor.map(runner, problems))
                for prob_num in range(len(problems)):
                    problem = problems[prob_num]
                    solution = solutions[prob_num]
                    if(getattr(solution, 'infeasible', None)):
                        return handle_infeasible(msg=solution.reason)
                    else:
                        new_solution = rebuild_solution(solution)
                        update_solution(problem, new_solution)
                # for future in futures:
                #     solver = future.result()
                #     update_solution(solver)

    # import ipdb; ipdb.set_trace()
    log_step('Clustered Optimization complete')
    # log.debug(placement.md_list)
    r = refine_devices(inp.devices, placement.md_list)
    if(getattr(r, 'infeasible', None)):
        return handle_infeasible(msg=r.reason)
    # TODO:: Put intermediate output to debug!
    # Allow loggers to take input logging level
    log_placement(inp.devices, inp.partitions, inp.flows,
                  placement.dev_par_tuplelist, placement.frac,
                  placement.md_list)
    return Namespace(results=r, md_list=placement.md_list, frac=placement.frac)


# * Main solve function
# HOLD: Handle disconnected graph in solver
# Final devices will always be refined, just log at the end
@log_time(logger=log.info)
def solve(inp, pre_processed=False):
    start = time.time()

    # Assign device ids, if not frozen
    # If frozen then assume ids have been assigned
    try:
        for (dnum, d) in enumerate(inp.devices):
            d.dev_id = dnum
            freeze_object(d)
    except TypeError:
        pass

    if(not pre_processed):
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
            return handle_infeasible(getattr(solver, 'culprit', None),
                                     msg=solver.reason)
        ret = Namespace(results=solver.r, md_list=solver.md_list,
                        frac=solver.frac)

    if(ret is None):
        return None

    end = time.time()
    log.info("\n" + "-"*80)
    log_results(inp.devices, ret.results, ret.md_list,
                elapsed=end-start, msg="Final Results")

    return ret


@log_time(logger=log.info)
def run(inp):
    ret = solve(inp)

    if(getattr(inp, 'additions', None)):
        new_inp = get_new_problem(inp, ret, inp.additions)
        new_ret = solve(new_inp, pre_processed=True)

        # TODO: combine old and new ret
        # TODO: compare with recompute full solution
        return ret
    else:
        return ret


if(__name__ == '__main__'):
    start = time.time()
    parser = generate_parser()
    args = parser.parse_args(sys.argv[1:])
    if(hasattr(args, 'config_file')):
        for fpath in args.config_file:
            common_config.load_config_file(fpath)
    common_config.update(args)

    setup_logging(common_config)

    input_num = common_config.input_num
    inp = input_generator[input_num]
    if(not isinstance(inp, Input)):
        inp = inp.get_input()

    # log.info("Time before solving: {}".format(time.time() - start))
    try:
        run(inp)
    except Exception:
        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
    # log.info("Total time in main.py: {}".format(time.time() - start))

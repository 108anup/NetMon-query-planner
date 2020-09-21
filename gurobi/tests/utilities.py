import random
import time
import os
from main import solve, get_new_problem
from solvers import refine_devices, log_results

from config import common_config
import subprocess
from common import (setup_logging, add_file_logger,
                    remove_all_file_loggers)
from input import Input
from profiles import beluga20, tofino, agiliocx40gbe
from devices import CPU, Netronome, P4
import numpy as np
import pickle

base_dir = 'outputs/tmp'


def flatten_list(l):
    ret = []
    for e in l:
        if(isinstance(e, list)):
            ret.extend(flatten_list(e))
        else:
            ret.append(e)
    return ret


def get_git_revision_short_hash():
    hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    return hash.decode()[:-1]


def get_partition_str(args):
    if(args.horizontal_partition and args.vertical_partition):
        return "hv"
    elif args.horizontal_partition:
        return "h"
    elif args.vertical_partition:
        return "v"
    else:
        return "n"


def get_init_str(args):
    if(args.init):
        return "init"
    else:
        return "noinit"


def run_all_with_input(m, inp, solvers=['UnivmonGreedyRows', 'Netmon']):

    with open(common_config.results_file, 'a') as f:
        f.write("{}, {}, {}, ".format(
            m.test_name + "-" + get_git_revision_short_hash(),
            m.config_str, m.args_str))
        f.close()

    for solver in solvers:
        common_config.solver = solver
        setup_logging(common_config)
        remove_all_file_loggers()
        add_file_logger(os.path.join(
            m.out_dir, '{}-{}-{}.out'
            .format(m.config_str, m.args_str, solver)))

        if(not isinstance(inp, Input)):
            myinp = inp.get_input()
        solve(myinp)

    with open(common_config.results_file, 'a') as f:
        f.write("\n")
        f.close()


def full_rerun_flow_dynamics(m, inp, num_changes, change_size):
    pickle_dir = os.path.join(base_dir,
                              "pickles_" + get_git_revision_short_hash())
    with open(common_config.results_file, 'a') as f:
        f.write("{}, {}, {}\n".format(
            m.test_name + "-" + get_git_revision_short_hash(),
            m.config_str, m.args_str))

    solver = 'Netmon'
    common_config.solver = solver

    if(not isinstance(inp, Input)):
        myinp = inp.get_input()

    inp_flows = myinp.flows
    # import ipdb; ipdb.set_trace()
    myinp.flows = inp_flows[:-change_size * num_changes]
    remaining_flows = inp_flows[-change_size * num_changes:]

    setup_logging(common_config)
    remove_all_file_loggers()
    add_file_logger(os.path.join(
        m.out_dir, '{}-{}-{}-first-run.out'
        .format(m.config_str, m.args_str, solver)))

    ret = solve(myinp)

    with open(common_config.results_file, 'a') as f:
        f.write("\n")

    for change_id in range(num_changes):
        with open(common_config.results_file, 'a') as f:
            f.write("change_id={}, ".format(change_id))

        setup_logging(common_config)
        remove_all_file_loggers()
        add_file_logger(os.path.join(
            m.out_dir, '{}-{}-{}-{}-full-rerun.out'
            .format(m.config_str, m.args_str, solver, change_id)))

        # flows
        this_flows = remaining_flows[
            change_id * change_size:(change_id+1)*change_size]
        myinp.flows += this_flows

        # devices (sanity check)
        # TODO: (ideally changed devices should be the same changes)
        changed_devices = get_changed_devices(myinp.devices, change_size,
                                              change_id, True, pickle_dir)
        for dnum, d in changed_devices.items():
            myinp.devices[dnum] = d

        new_ret = solve(myinp)
        if(new_ret is None):
            return

        with open(common_config.results_file, 'a') as f:
            f.write("\n")
            f.close()


def get_changed_devices(old_devices, change_size, change_id, read, pickle_dir):
    """
    Setup device availability changes:
    """

    if(read):
        change_file = open(
            os.path.join(pickle_dir, "{}.pickle".format(change_id)), 'rb')
        changes = pickle.load(change_file)
        change_file.close()
        changed_device_ids = changes[0]
        change_directions = changes[1]
    else:
        total_devices_count = len(old_devices)
        # Sample change size number of random devices
        changed_device_ids = random.sample(range(total_devices_count),
                                           change_size)
        # Vary availability by 20%
        change_directions = np.round(np.random.rand(change_size)).tolist()
        # [round(random()) for i in range(change_size)]
        # random.sample(range(2), change_size)
        changes = (changed_device_ids, change_directions)
        change_file = open(
            os.path.join(pickle_dir, "{}.pickle".format(change_id)), 'wb')
        pickle.dump(changes, change_file)
        change_file.close()

    changed_devices = {}
    for change_num in range(change_size):
        dnum = changed_device_ids[change_num]
        d = old_devices[dnum]
        if(isinstance(d, CPU)):
            standard = beluga20['cores']
            new = int(standard * 0.8 +
                      standard * 0.4 * change_directions[change_num])
            d.cores = new
        elif(isinstance(d, Netronome)):
            standard = agiliocx40gbe['total_me']
            new = int(standard * 0.8)
            d.total_me = new
        elif(isinstance(d, P4)):
            pass
        else:
            assert(False)
        changed_devices[dnum] = d

    return changed_devices


def run_flow_dynamics(m, inp, num_changes, change_size):

    pickle_dir = os.path.join(base_dir,
                              "pickles_" + get_git_revision_short_hash())
    os.makedirs(pickle_dir, exist_ok=True)
    with open(common_config.results_file, 'a') as f:
        f.write("{}, {}, {}\n".format(
            m.test_name + "-" + get_git_revision_short_hash(),
            m.config_str, m.args_str))

    solver = 'Netmon'
    common_config.solver = solver

    if(not isinstance(inp, Input)):
        myinp = inp.get_input()
    else:
        myinp = inp

    inp_flows = myinp.flows
    # import ipdb; ipdb.set_trace()
    myinp.flows = inp_flows[:-change_size * num_changes]
    remaining_flows = inp_flows[-change_size * num_changes:]

    setup_logging(common_config)
    remove_all_file_loggers()
    add_file_logger(os.path.join(
        m.out_dir, '{}-{}-{}-init.out'
        .format(m.config_str, m.args_str, solver)))

    ret = solve(myinp)

    with open(common_config.results_file, 'a') as f:
        f.write("\n")

    old_inp = myinp
    old_solution = ret
    for change_id in range(num_changes):
        with open(common_config.results_file, 'a') as f:
            f.write("change_id={}, ".format(change_id))

        setup_logging(common_config)
        remove_all_file_loggers()
        add_file_logger(os.path.join(
            m.out_dir, '{}-{}-{}-{}-succ.out'
            .format(m.config_str, m.args_str, solver, change_id)))

        changed_devices = get_changed_devices(old_inp.devices, change_size,
                                              change_id, False, pickle_dir)
        this_flows = remaining_flows[
            change_id * change_size:(change_id+1)*change_size]
        new_inp = get_new_problem(
            old_inp, old_solution,
            Input(flows=this_flows, changed_devices=changed_devices))
        start = time.time()
        new_ret = solve(new_inp, pre_processed=True)
        if(new_ret is None):
            return
        new_solution = new_ret

        # calc full resource util
        complete_md_list = old_solution.md_list
        for dnum, d in enumerate(new_inp.devices):
            old_md = complete_md_list[d.dev_id]
            new_md = new_ret.md_list[dnum]
            old_id = getattr(old_md, 'dev_id', None)
            new_id = getattr(new_md, 'dev_id', None)
            assert(old_id is None or new_id is None or old_id == new_id)
            complete_md_list[d.dev_id] = new_md

        results = refine_devices(myinp.devices, complete_md_list)
        end = time.time()
        log_results(myinp.devices, results, complete_md_list,
                    elapsed=end-start, msg='Computed Full Resource Stat')

        with open(common_config.results_file, 'a') as f:
            f.write("\n")
            f.close()

        # merge (update old_inp and old_solution):
        for f in this_flows:
            f.partitions = [(x[0], x[1]) for x in f.queries]
        old_inp.flows += this_flows

        # Update inp's devices for next itr
        for dnum, d in changed_devices.items():
            old_inp.devices[dnum] = d

        old_solution.results = results  # this value is not used
        old_solution.md_list = complete_md_list
        for (dnum, pnum), f in new_solution.frac.items():
            dev_id = new_inp.devices[dnum].dev_id
            old_solution.frac[dev_id, pnum] = f


def setup_test_meta(m):
    m.out_dir = os.path.join(base_dir, m.test_name
                             + "-" + get_git_revision_short_hash())
    os.makedirs(m.out_dir, exist_ok=True)
    gitignore = os.path.join(m.out_dir, '.gitignore')
    if(not os.path.exists(gitignore)):
        f = open(gitignore, 'w')
        f.write('*.out')
        f.close()
    common_config.results_file = os.path.join(m.out_dir, 'results.csv')
    m.config_str = '{}-{}'.format(get_partition_str(common_config),
                                  get_init_str(common_config))


# l is a list of lists
def combinations(l):
    if(len(l) == 1):
        return [tuple((x, )) for x in l[0]]
    else:
        return [tuple((x, )) + y for x in l[0] for y in combinations(l[1:])]

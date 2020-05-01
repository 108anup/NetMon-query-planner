import os
from main import solve

from config import common_config
import subprocess
from common import (setup_logging, add_file_logger,
                    remove_all_file_loggers)

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
        f.write("{}, {}, {}, ".format(m.test_name, m.config_str, m.args_str))
        f.close()

    for solver in solvers:
        common_config.solver = solver
        setup_logging(common_config)
        remove_all_file_loggers()
        add_file_logger(os.path.join(
            m.out_dir, '{}-{}-{}.out'
            .format(m.config_str, m.args_str, solver)))

        solve(inp)

    with open(common_config.results_file, 'a') as f:
        f.write("\n")
        f.close()


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

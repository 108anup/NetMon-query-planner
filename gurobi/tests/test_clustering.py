import pytest
import os
from main import solve
from input import (beluga20, tofino, Input, eps0, del0,
                   generate_overlay, dc_topology, Input)
from devices import CPU, P4
from sketches import cm_sketch
from flows import flow
from common import (setup_logging, add_file_logger,
                    remove_all_file_loggers, Namespace)
from config import common_config
import numpy as np
import random

base_dir = 'outputs/clustering'


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
    m.out_dir = os.path.join(base_dir, m.test_name)
    common_config.results_file = os.path.join(m.out_dir, 'results.csv')
    m.config_str = '{}-{}'.format(get_partition_str(common_config),
                                  get_init_str(common_config))


# l is a list of lists
def combinations(l):
    if(len(l) == 1):
        return [tuple((x, )) for x in l[0]]
    else:
        return [tuple((x, )) + y for x in l[0] for y in combinations(l[1:])]


@pytest.mark.parametrize(
    "hosts_per_tors, tors_per_l1s, l1s, overlay, refine",
    # [(48, 20, 10, 'none')]
    # combinations(
    #     [[48], [2, 10, 20], [2, 4, 10], ['tenant'], [True, False]]
    # )
    combinations(
        [[48], [50], [20], ['spectralA'], [False]]
    )
)
def test_vary_topo_size_dc_topo_tenant(hosts_per_tors, tors_per_l1s,
                                       l1s, overlay, refine):
    num_hosts = hosts_per_tors*tors_per_l1s*l1s
    inp = dc_topology(hosts_per_tors, tors_per_l1s, l1s,
                      num_queries=int(num_hosts/2),
                      overlay=overlay, tenant=True,
                      refine=refine)

    # Testing: overlay uncorrelated with tenants and traffic

    common_config.vertical_partition = True

    m = Namespace()
    m.test_name = 'vary_topo_size_dc_topo_tenant'
    total_devices = len(inp.devices)
    m.args_str = (
        "overlay={};total_devices={};refine={};"
        "hosts_per_tors={};tors_per_l1s={};l1s={}"
        .format(overlay, total_devices, refine,
                hosts_per_tors, tors_per_l1s, l1s)
    )
    setup_test_meta(m)
    run_all_with_input(m, inp)


@pytest.mark.parametrize("cluster_size, num_cpus", [(0, 20)]
                         + [(2 + i, 20) for i in range(19)])
def test_vary_cluster_size_cpu_triangle(cluster_size, num_cpus):
    overlay = None
    if(cluster_size > 0):
        overlay = generate_overlay([int(num_cpus/cluster_size), cluster_size])
        if(num_cpus % cluster_size != 0):
            overlay += [[num_cpus - 1 - i
                         for i in range(num_cpus % cluster_size)]]
        overlay += [num_cpus]

    common_config.vertical_partition = True

    m = Namespace()
    m.test_name = 'vary_cluster_size_cpu_triangle'
    m.args_str = "cluster_size={};num_cpus={}".format(cluster_size, num_cpus)
    setup_test_meta(m)

    inp = Input(
        devices=(
            [CPU(**beluga20, name='CPU_{}'.format(i))
             for i in range(num_cpus)]
            + [P4(**tofino, name='P4_1')]
        ),
        queries=[
            cm_sketch(eps0=eps0, del0=del0) for i in range(num_cpus)
        ],
        flows=[flow(path=(i, num_cpus, (i + 1) % num_cpus),
                    queries=[(i, 1)])
               for i in range(num_cpus)],
        overlay=overlay
    )
    run_all_with_input(m, inp)


@pytest.mark.parametrize("cluster_size, num_cpus", [(0, 20)]
                         + [(2 + i, 20) for i in range(19)])
def test_vary_cluster_size_cpu_triangle2(cluster_size, num_cpus):
    overlay = None
    if(cluster_size > 0):
        overlay = generate_overlay([int(num_cpus/cluster_size), cluster_size])
        if(num_cpus % cluster_size != 0):
            overlay += [[num_cpus - 1 - i
                         for i in range(num_cpus % cluster_size)]]
        overlay[0] += [num_cpus]

    common_config.vertical_partition = True

    m = Namespace()
    m.test_name = 'vary_cluster_size_cpu_triangle2'
    m.args_str = "cluster_size={};num_cpus={}".format(cluster_size, num_cpus)
    setup_test_meta(m)

    inp = Input(
        devices=(
            [CPU(**beluga20, name='CPU_{}'.format(i))
             for i in range(num_cpus)]
            + [P4(**tofino, name='P4_1')]
        ),
        queries=[
            cm_sketch(eps0=eps0, del0=del0) for i in range(num_cpus)
        ],
        flows=[flow(path=(i, num_cpus, (i + 1) % num_cpus),
                    queries=[(i, 1)])
               for i in range(num_cpus)],
        overlay=overlay
    )
    run_all_with_input(m, inp)


@pytest.mark.parametrize("overlay_num", [i for i in range(6)])
def test_vary_cluster_simple_topo(overlay_num):
    inp = Input(
        devices=(
            [CPU(**beluga20, name='CPU_{}'.format(i)) for i in range(4)] +
            [P4(**tofino, name='P4_{}'.format(i+4)) for i in range(2)]
        ),
        queries=[
            cm_sketch(eps0=eps0, del0=del0) for i in range(2)
        ],
        flows=[
            flow(path=(0, 4, 5, 3), queries=[(1, 1)]),
            flow(path=(1, 4, 5, 3), queries=[(1, 1)]),
            flow(path=(2, 4, 5, 3), queries=[(0, 1)])
        ],
    )

    common_config.vertical_partition = True
    m = Namespace()
    m.test_name = 'vary_cluster_simple_topo'
    m.args_str = "overlay_num={}".format(overlay_num)
    setup_test_meta(m)

    if(overlay_num == 1):
        inp.overlay = [[1, 4, 3, 5], 0, 2]
    elif(overlay_num == 2):
        inp.overlay = [[0, 4, 3, 5], 1, 2]
    elif(overlay_num == 3):
        inp.overlay = [[0, 1, 2, 4], [3, 5]]
    elif(overlay_num == 4):
        inp.overlay = [[0, 4, 1], [3, 5], 2]
    elif(overlay_num == 5):
        inp.overlay = [[0, 1, 2], [4, 3, 5]]

    run_all_with_input(m, inp)

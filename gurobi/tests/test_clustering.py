import pytest
from input import (beluga20, tofino, Input, eps0, del0,
                   generate_overlay, TreeTopology)
from devices import CPU, P4
from sketches import cm_sketch
from flows import flow
from config import common_config
from common import Namespace
import tests.utilities as ut
from tests.utilities import (run_all_with_input, setup_test_meta, combinations)
from clos import Clos
ut.base_dir = 'outputs/clustering'


@pytest.mark.parametrize(
    "hosts_per_tors, tors_per_l1s, l1s, overlay, "
    "refine, devices_per_cluster, clusters_per_cluster, portion_netronome",
    # [(48, 20, 10, 'none')]
    # combinations(
    #     [[48], [2, 10, 20], [2, 4, 10], ['tenant'], [True, False]]
    # )

    # Large
    # combinations(
    #     [[48], [20, 50], [20], ['tenant'], [False],
    #      [25], [200], [0, 0.5]]
    # )

    # Medium
    combinations(
        [[48], [10, 20], [4, 10], ['tenant'], [False],
         [25], [200], [0, 0.5]]
    )
    # combinations(
    #     [[8], [2], [2], ['tenant'], [False]]
    # )
)
def test_vary_topo_size_dc_topo_tenant(
        hosts_per_tors, tors_per_l1s, l1s, overlay, refine,
        devices_per_cluster, clusters_per_cluster, portion_netronome):
    query_density = 3
    common_config.MAX_DEVICES_PER_CLUSTER = devices_per_cluster
    common_config.MAX_CLUSTERS_PER_CLUSTER = clusters_per_cluster
    num_hosts = hosts_per_tors*tors_per_l1s*l1s
    num_queries = int(num_hosts * query_density)
    inp = TreeTopology(
        hosts_per_tors, tors_per_l1s, l1s, num_queries=num_queries,
        overlay=overlay, tenant=True, refine=refine, eps=eps0/10,
        queries_per_tenant=8*query_density,
        portion_netronome=portion_netronome)
    common_config.perf_obj = True
    common_config.parallel = True
    common_config.vertical_partition = True
    # common_config.horizontal_partition = True
    # common_config.mipout = True
    common_config.verbose = 1

    m = Namespace()
    m.test_name = 'vary_topo_size_dc_topo_tenant'
    total_devices = (num_hosts + tors_per_l1s*l1s
                     + l1s + 1 + num_hosts * portion_netronome)
    m.args_str = (
        "overlay={};total_devices={};refine={};"
        "hosts_per_tors={};tors_per_l1s={};l1s={};"
        "num_queries={};devices_per_cluster={};"
        "clusters_per_cluster={}"
        .format(overlay, total_devices, refine,
                hosts_per_tors, tors_per_l1s, l1s,
                num_queries, devices_per_cluster,
                clusters_per_cluster)
    )
    setup_test_meta(m)
    run_all_with_input(m, inp)


@pytest.mark.parametrize(
    "pods, query_density, portion_netronome, overlay, "
    "refine, devices_per_cluster, clusters_per_cluster, perf_obj",
    # Medium
    combinations(
        [[24, 32], [4], [0, 1], ['tenant'], [False],
         [16], [128], [True, False]]
    )
)
def test_vary_topo_size_clos(
        pods, query_density, portion_netronome, overlay, refine,
        devices_per_cluster, clusters_per_cluster, perf_obj):
    common_config.MAX_DEVICES_PER_CLUSTER = devices_per_cluster
    common_config.MAX_CLUSTERS_PER_CLUSTER = clusters_per_cluster
    inp = Clos(
        pods, query_density, portion_netronome=portion_netronome,
        overlay=overlay, eps=eps0/10)
    common_config.perf_obj = perf_obj
    obj = "res"
    if(common_config.perf_obj):
        obj = "perf"
    common_config.parallel = False
    common_config.vertical_partition = True
    # common_config.horizontal_partition = True
    common_config.mipout = False
    common_config.verbose = 1

    m = Namespace()
    m.test_name = 'vary_topo_size_clos'
    m.args_str = (
        "overlay={};total_devices={};refine={};"
        "num_queries={};pods={};obj={};num_netronome={};devices_per_cluster={};"
        "clusters_per_cluster={}"
        .format(overlay, inp.total_devices, refine,
                inp.num_queries, pods, obj, inp.num_netronome, devices_per_cluster,
                clusters_per_cluster)
    )
    setup_test_meta(m)
    run_all_with_input(m, inp)



@pytest.mark.parametrize(
    "devices_per_cluster", [25, 50, 100, 150, 200]
)
def test_devices_per_cluster(devices_per_cluster):
    # common_config.parallel = True
    common_config.vertical_partition = True
    # common_config.horizontal_partition = True
    # common_config.mipout = True
    common_config.verbose = 1
    common_config.MAX_DEVICES_PER_CLUSTER = devices_per_cluster

    inp = TreeTopology(48, 20, 4,
                       num_queries=480*4,
                       overlay='tenant', tenant=True,

                       refine=False, eps=eps0/100,
                       queries_per_tenant=4)

    m = Namespace()
    m.test_name = 'vary_devices_per_cluster'
    m.args_str = (
        "devices_per_cluster={}"
        .format(devices_per_cluster)
    )
    setup_test_meta(m)
    run_all_with_input(m, inp, solvers=['Netmon'])


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

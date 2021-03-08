from types import SimpleNamespace

import pytest
from tests.utilities import combinations, run_all_with_input, setup_test_meta

from common import Namespace
from config import common_config
from topology.clos import Clos

p1 = SimpleNamespace(x=4_500, y=150)
p2 = SimpleNamespace(x=58_176, y=8000)
slope = (p2.y - p1.y) / (p2.x - p1.x)
intercept = p1.y - slope * p1.x


def get_time_limit(topo):
    this_devices = topo.total_devices
    this_time = intercept + slope * this_devices
    return int(this_time)


TIME_LIMIT = {
    16: 150,
    20: 150,
    24: 420,
    32: 2000,
    48: 8000
}

PODS_QD = [
    (16, 1),
    (20, 3),
    (24, 3),
    (32, 4),
    (48, 4)
]

TOPOLOGIES_TO_TEST = [
    Clos(pods=p, query_density=q, portion_netronome=0.5,
         portion_fpga=0.5,
         overlay='none') for p, q in PODS_QD
]

SCHEMES = [
    ('Baseline', 'none'),
    ('Univmon', 'none'),
    ('UnivmonGreedyRows', 'none'),
    ('UnivmonGreedyRows', 'tenant'),
    ('Netmon', 'none'),
    ('Netmon', 'tenant'),
]


@pytest.mark.parametrize(
    "topo, scheme",
    combinations([TOPOLOGIES_TO_TEST[-1:], SCHEMES[-1:]])
)
def test_scale_clos(topo, scheme):
    common_config.parallel = True
    common_config.mipout = False
    common_config.verbose = 1

    m = Namespace()
    m.test_name = 'vary_scale_clos'
    m.args_str = (
        "pods={};sketch_load={};scheme={};overlay={}"
        .format(topo.pods, topo.query_density, scheme[0], scheme[1])
    )
    solvers = [scheme[0]]
    if(scheme[0] == 'Baseline'):
        common_config.static = True
        solvers = ['Univmon']
    else:
        common_config.static = False
    topo.overlay = scheme[1]
    # common_config.time_limit = get_time_limit(topo)
    common_config.time_limit = TIME_LIMIT[topo.pods]
    setup_test_meta(m, "outputs/vary_scale")
    run_all_with_input(m, topo, solvers=solvers)

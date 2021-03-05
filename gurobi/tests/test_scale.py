import pytest
from tests.utilities import (combinations, run_all_with_input, setup_test_meta)

from common import Namespace
from config import common_config
from topology.clos import Clos

PODS_QD = [
    (16, 1),
    (20, 1),
    (24, 3),
    (32, 3),
    (48, 4)
]

TOPOLOGIES_TO_TEST = [
    Clos(pods=p, query_density=q, portion_netronome=0.5, portion_fpga=0.5,
         overlay='none') for p, q in PODS_QD
]

SCHEMES = {
    ('Baseline', 'none'),
    ('Univmon', 'none'),
    ('UnivmonGreedyRows', 'none'),
    ('UnivmonGreedyRows', 'tenant'),
    ('Netmon', 'none'),
    ('Netmon', 'tenant'),
}


@pytest.mark.parametrize(
    "inp, overlay",
    combinations([TOPOLOGIES_TO_TEST[:3], SCHEMES[-2]])
)
def test_scale_clos(inp, scheme):
    common_config.parallel = True
    common_config.mipout = False
    common_config.verbose = 1

    m = Namespace()
    m.test_name = 'vary_scale_clos'
    m.args_str = (
        "pods={};sketch_load={};scheme={};overlay={}"
        .format(inp.pods, inp.query_density, scheme[0], scheme[1])
    )
    setup_test_meta(m)
    solvers = [scheme[0]]
    if(scheme[0] == 'Baseline'):
        common_config.static = True
        solvers = ['Univmon']
    inp.overlay = scheme[1]
    run_all_with_input(m, inp, solvers=solvers)

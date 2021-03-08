import pytest
from tests.utilities import run_all_with_input, setup_test_meta

from common import Namespace
from config import common_config
from topology.clos import Clos
from common import constants

TOPOLOGIES_TO_TEST = [
    Clos(pods=16, query_density=q, portion_netronome=0.5, portion_fpga=0.5,
         overlay='tenant', eps=constants.eps0/8)
    for q in [1, 2, 3, 4, 5, 6]
]

SCHEMES = [
    ('Netmon', 'tenant'),
]


@pytest.mark.parametrize("topo", TOPOLOGIES_TO_TEST)
def test_profiler_error(topo):
    common_config.parallel = True
    common_config.mipout = False
    common_config.verbose = 1

    m = Namespace()
    m.test_name = "profiler_error"
    m.args_str = ("handle={};pickle={}"
                  .format(topo.query_density,
                          topo.get_pickle_name()))
    setup_test_meta(m, "outputs/{}".format(m.test_name))
    run_all_with_input(m, topo, solvers=['Netmon'])

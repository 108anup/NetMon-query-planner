import pytest
from tests.utilities import combinations, run_all_with_input, setup_test_meta

from common import Namespace
from config import common_config
from topology.clos import Clos
from topology.jellyfish import JellyFish
from topology.topology_zoo_wan import TopologyZooWAN

ZOO_SELECTION = [
    'Geant2012.graphml',
    'GtsCe.graphml',
    'Cogentco.graphml',
    'UsCarrier.graphml',
    'TataNld.graphml',
]
BAD_TRAFFIC = [
    'Colt.graphml'
]

TOPOLOGIES_TO_TEST = [
    TopologyZooWAN(zoo_name, overlay='none',
                   query_density=4,
                   portion_netronome=0.5,
                   portion_fpga=0.5) for zoo_name in ZOO_SELECTION
] + [
    JellyFish(tors=500, ports_per_tor=20,
              num_hosts=2000, overlay='none',
              query_density=1,
              portion_netronome=0.5,
              portion_fpga=0.5),
    Clos(pods=20, query_density=1, portion_netronome=0.5, portion_fpga=0.5,
         overlay='none')
]

SCHEMES = [
    ('Baseline', 'none'),
    ('Univmon', 'none'),
    ('UnivmonGreedyRows', 'none'),
    ('UnivmonGreedyRows', 'tenant'),
    ('Netmon', 'none'),
    ('Netmon', 'tenant'),
]

@pytest.mark.parametrize("inp, scheme",
                         combinations([TOPOLOGIES_TO_TEST, SCHEMES]))
def test_vary_topology(inp, scheme):
    common_config.parallel = True
    common_config.mipout = False
    common_config.verbose = 1

    m = Namespace()
    m.test_name = "vary_topology"
    m.args_str = ("handle={};scheme={};overlay={};pickle={}"
                  .format(inp.get_name(), scheme[0], scheme[1],
                          inp.get_pickle_name()))
    # TODO: remove redundancy
    solvers = [scheme[0]]
    if(scheme[0] == 'Baseline'):
        common_config.static = True
        solvers = ['Univmon']
    else:
        common_config.static = False
    inp.overlay = scheme[1]
    common_config.time_limit = 150
    setup_test_meta(m, "outputs/vary_topology")
    run_all_with_input(m, inp, solvers=solvers)

import pytest
from common import Namespace
from topology.jellyfish import JellyFish
from topology.topology_zoo import TopologyZoo
from topology.topology_zoo_wan import TopologyZooWAN
from topology.clos import Clos
from tests.utilities import (run_all_with_input, setup_test_meta,
                             combinations, run_flow_dynamics, full_rerun_flow_dynamics)
from config import common_config

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
    Clos(pods=20, query_density=3, portion_netronome=0.5, portion_fpga=0.5,
         overlay='none')
]

OVERLAYS = [
    # 'none',
    'tenant'
]

@pytest.mark.parametrize("inp, overlay", combinations([TOPOLOGIES_TO_TEST[-1:], OVERLAYS]))
def test_vary_topology(inp, overlay):
    m = Namespace()
    m.test_name = "vary_topology"
    m.args_str = ("handle={};overlay={};pickle={}"
                  .format(inp.get_name(), overlay, inp.get_pickle_name()))
    inp.overlay = overlay
    common_config.time_limit = 420
    setup_test_meta(m, "outputs/vary_topology")
    run_all_with_input(m, inp)

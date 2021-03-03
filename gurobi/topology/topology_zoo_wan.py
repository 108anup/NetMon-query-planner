import sys
import traceback
import math
import os

import networkx as nx
import numpy as np

from common import constants
from profiles import dc_line_rate
from topology import Topology

TOPOLOGY_ZOO_DIRECTORY = "./topology-zoo"
NUM_AGG_SWITCH_PER_CORE = 2
NUM_TOR_SWITCH_PER_CORE = 2
NUM_HOSTS_PER_CORE = NUM_TOR_SWITCH_PER_CORE * 2
MAX_NUM_HOSTS_PER_CORE = NUM_TOR_SWITCH_PER_CORE * 16


class TopologyZooWAN(Topology):

    def __init__(self, topology_gml_name, query_density=2,
                 portion_netronome=0.5, portion_fpga=0.5,
                 eps=constants.eps0, overlay='none',
                 hosts_per_tenant=8):
        if(topology_gml_name.endswith('.gml')):
            g = nx.read_gml(
                os.path.join(TOPOLOGY_ZOO_DIRECTORY, topology_gml_name),
                label='id')
        else:
            assert topology_gml_name.endswith('.graphml')
            g = nx.read_graphml(
                os.path.join(TOPOLOGY_ZOO_DIRECTORY, topology_gml_name))
        if(isinstance(g, nx.MultiGraph)):
            g = nx.Graph(g)
        # Each switch in the original input is connected to 4 switches
        # 2 agg and 2 tor
        self.num_switches = (4 + 1) * len(g.nodes)
        degrees = [d for n,d in g.degree]
        self.max_degree = max(degrees)
        # self.degree_to_host = lambda d: math.ceil(
        #     NUM_HOSTS_PER_CORE*(d + 1)/(self.max_degree + 1))
        self.degree_to_host = lambda d: min(math.ceil(
            NUM_HOSTS_PER_CORE*d), MAX_NUM_HOSTS_PER_CORE)
        self.switch2numhosts = [self.degree_to_host(d) for d in degrees]
        self.num_hosts = np.sum(self.switch2numhosts)
        self.switch_connections = g
        self.num_netronome = int(self.num_hosts * portion_netronome)
        self.num_fpga = int(self.num_hosts * portion_fpga)
        self.num_queries = self.num_hosts * query_density
        self.eps = eps
        self.overlay = overlay
        self.topology_gml_name = topology_gml_name
        self.hosts_per_tenant = hosts_per_tenant
        self.query_density = query_density
        super().__init__()

    def construct_graph(self, devices):
        g = super().construct_graph(devices)

        zoo_node_to_sname = dict()
        sname_to_zoo_node = dict()
        for snum, n in enumerate(self.switch_connections.nodes):
            zoo_node_to_sname[n] = self.switches[snum][0]
            sname_to_zoo_node[self.switches[snum][0]] = n

        for s1, s2 in self.switch_connections.edges:
            g.add_edge(zoo_node_to_sname[s1],
                       zoo_node_to_sname[s2],
                       remaining=dc_line_rate)

        till_now = 0
        num_core_switches = len(self.switch_connections.nodes)
        core_switches = self.switches[till_now:till_now + num_core_switches]
        till_now += num_core_switches

        num_agg_switches = num_core_switches * NUM_AGG_SWITCH_PER_CORE
        agg_switches = self.switches[till_now:till_now + num_agg_switches]
        till_now += num_agg_switches

        num_tor_switches = num_core_switches * NUM_TOR_SWITCH_PER_CORE
        tor_switches = self.switches[till_now:till_now + num_tor_switches]
        till_now += num_tor_switches

        assert(till_now == self.num_switches)

        agg_idx = 0
        tor_idx = 0
        for this_switch in core_switches:
            for i in range(NUM_AGG_SWITCH_PER_CORE):
                # Connect core to all agg switches meant for it
                g.add_edge(this_switch[0], agg_switches[agg_idx+i][0],
                           remaining=dc_line_rate)

                # All to all connections between agg and tor
                for j in range(NUM_TOR_SWITCH_PER_CORE):
                    g.add_edge(agg_switches[agg_idx+i][0], tor_switches[tor_idx+j][0],
                               remaining=dc_line_rate)
            agg_idx+=NUM_AGG_SWITCH_PER_CORE
            tor_idx+=NUM_TOR_SWITCH_PER_CORE

        host_dnum = 0
        tor_switch_start = 0
        for this_switch in core_switches:
            # Assuming switch_connections and core switches mapped one to one
            zoo_sw = sname_to_zoo_node[this_switch[0]]
            zoo_deg = self.switch_connections.degree(zoo_sw)
            this_num_hosts = self.degree_to_host(zoo_deg)
            for i in range(this_num_hosts):
                assert(host_dnum < self.num_hosts)
                tor_switch = tor_switches[tor_switch_start
                                          + i % NUM_TOR_SWITCH_PER_CORE]
                connect_switch_to = self.hosts[host_dnum]
                nic_dnum = self.has_nic[host_dnum]
                if(nic_dnum != -1):
                    g.add_edge(self.hosts[host_dnum][0], self.nics[nic_dnum][0],
                               remaining=dc_line_rate)
                    connect_switch_to = self.nics[nic_dnum]
                g.add_edge(connect_switch_to[0], tor_switch[0],
                           remaining=dc_line_rate)
                host_dnum += 1
            tor_switch_start += NUM_TOR_SWITCH_PER_CORE
        assert(host_dnum == self.num_hosts)
        return g

    def get_pickle_name(self):
        return "zooWan-{}-{}-{}".format(self.topology_gml_name,
                                     self.num_queries,
                                     constants.eps0/self.eps)

    def get_name(self):
        if(self.topology_gml_name.endswith('.graphml')):
            return self.topology_gml_name[:-8]
        if(self.topology_gml_name.endswith('.gml')):
            return self.topology_gml_name[:-4]


if(__name__ == "__main__"):
    try:
        gen = TopologyZooWAN('Geant2012.gml')
        gen = TopologyZooWAN('Arpanet196912.graphml')
        inp = gen.create_inp()
        import ipdb; ipdb.set_trace()
    except Exception:
        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)

import os

import networkx as nx
import numpy as np

from common import constants
from profiles import dc_line_rate
from topology import Topology

TOPOLOGY_ZOO_DIRECTORY = "./topology-zoo"


class TopologyZoo(Topology):

    def __init__(self, topology_gml_name, query_density=2,
                 portion_netronome=0.5, portion_fpga=0.5,
                 eps=constants.eps0, overlay='none',
                 hosts_per_tenant=8):
        g = nx.read_gml(
            os.path.join(TOPOLOGY_ZOO_DIRECTORY, topology_gml_name))
        self.num_switches = len(g.nodes)
        self.max_degree = max(d for (n, d) in g.degree)
        self.num_hosts = np.sum([self.max_degree - d for (n, d) in g.degree])
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
        for snum, n in enumerate(self.switch_connections.nodes):
            zoo_node_to_sname[n] = self.switches[snum][0]

        for s1, s2 in self.switch_connections.edges:
            g.add_edge(zoo_node_to_sname[s1],
                       zoo_node_to_sname[s2],
                       remaining=dc_line_rate)

        host_dnum = 0
        for this_switch in self.switches:
            while(g.degree(this_switch[0]) < self.max_degree
                  and host_dnum < self.num_hosts):
                connect_switch_to = self.hosts[host_dnum]
                nic_dnum = self.has_nic[host_dnum]
                if(nic_dnum != -1):
                    g.add_edge(self.hosts[host_dnum][0], self.nics[nic_dnum][0],
                               remaining=dc_line_rate)
                    connect_switch_to = self.nics[nic_dnum]
                g.add_edge(connect_switch_to[0], this_switch[0],
                           remaining=dc_line_rate)
                host_dnum += 1
        return g

    def get_pickle_name(self):
        return "zoo-{}-{}-{}".format(self.topology_gml_name,
                                     self.num_queries,
                                     constants.eps0/self.eps)


if(__name__ == "__main__"):
    gen = TopologyZoo('Geant2012.gml')
    import ipdb; ipdb.set_trace()

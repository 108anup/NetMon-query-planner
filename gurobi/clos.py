import math
import os
import pickle

import networkx as nx
import numpy as np

from common import constants, freeze_object, log
from devices import CPU, P4, Netronome
from input import Input
from sketches import cm_sketch
from profiles import beluga20, tofino, agiliocx40gbe


eps0 = constants.eps0
del0 = constants.del0


class Clos(object):
    '''
    Adapted from:
    https://github.com/frenetic-lang/ocaml-topology/blob/master/scripts/fattree.py
    Grateful to them for this!!
    '''

    def __init__(self, pods=8, query_density=2, portion_netronome=0,
                 eps=eps0):
        self.pods = pods
        self.podsby2 = int(pods/2)
        self.num_hosts = int((pods**3)/4)
        self.num_agg_sw = pods**2
        self.num_core_sw = int((pods**2)/4)
        self.num_netronome = math.ceil(self.num_hosts * portion_netronome)
        self.query_density = query_density
        self.num_queries = self.num_hosts * self.query_density
        self.eps = eps0

    def construct_graph(self, devices, has_netro):
        start = 0
        hosts = [(devices[i].name,
                  {'capacity': devices[i].line_thr,
                   'remaining': devices[i].line_thr, 'id': i})
                 for i in range(start, start+self.num_hosts)]
        start += self.num_hosts

        netronome_nics = [(devices[i].name,
                           {'capacity': devices[i].line_thr,
                            'remaining': devices[i].line_thr, 'id': i})
                          for i in range(start, start+self.num_netronome)]
        start += self.num_netronome

        core_switches = [(devices[i].name,
                          {'capacity': devices[i].line_thr/self.pods,
                           'remaining': devices[i].line_thr/self.pods,
                           'id': i})
                         for i in range(start, start+self.num_core_sw)]
        start += self.num_core_sw

        # For a given pod:
        # The first self.pods/2 agg sw are in upper layer
        # The next self.pods/2 agg sw are in lower layer
        agg_switches = [(devices[i].name,
                         {'capacity': devices[i].line_thr/self.pods,
                          'remaining': devices[i].line_thr/self.pods,
                          'id': i})
                        for i in range(start, start+self.num_agg_sw)]

        g = nx.Graph()
        g.add_nodes_from(hosts)
        g.add_nodes_from(core_switches)
        g.add_nodes_from(agg_switches)

        host_offset = 0
        for pod in range(self.pods):
            core_offset = 0
            for sw in range(self.podsby2):
                switch = agg_switches[(pod*self.pods) + sw][0]

                # Connect to core switches
                for port in range(self.podsby2):
                    core_switch = core_switches[core_offset][0]
                    g.add_edge(switch, core_switch)
                    core_offset += 1

                # Connect to aggregate switches in same pod
                for port in range(self.podsby2, self.pods):
                    lower_switch = agg_switches[(pod*self.pods) + port][0]
                    g.add_edge(switch, lower_switch)

            for sw in range(self.podsby2, self.pods):
                switch = agg_switches[(pod*self.pods) + sw][0]

                # Connect to hosts
                for port in range(self.podsby2, self.pods):
                    host = hosts[host_offset][0]

                    netro_offset = has_netro[host_offset]
                    if(netro_offset >= 0):
                        netro = netronome_nics[netro_offset][0]
                        g.add_edge(switch, netro)
                        g.add_edge(netro, host)
                    else:
                        g.add_edge(switch, host)
                    host_offset += 1

        self.hosts = hosts
        self.netronome_nics = netronome_nics
        self.core_switches = core_switches
        self.agg_switches = agg_switches
        return g

    def get_overlay(self):
        pass

    def get_path_with_largest_capacity(self, g, h1, h2):
        h1name = self.hosts[h1][0]
        h2name = self.hosts[h2][0]

        node_paths = nx.all_shortest_paths(g, h1name, h2name)
        node_paths_capacity = [
            (
                path,
                min(map(lambda x: g.nodes[x]['capacity'], path))
            )
            for path in node_paths
        ]
        (most_capacity_path, capacity) = max(
            node_paths_capacity,
            key=lambda x: x[1]
        )
        ids_path = list(map(lambda x: g.nodes[x]['id'],
                            most_capacity_path))
        import ipdb; ipdb.set_trace()
        return (ids_path, capacity)

    def get_flows(self, g, inp):
        print(self.get_path_with_largest_capacity(g, 0, self.pods*2+1))

    # FIXME: Remove redundancy
    def create_inp(self):
        inp = Input(
            devices=(
                [CPU(**beluga20, name='CPU_'+str(i+1))
                 for i in range(self.num_hosts)] +
                [Netronome(**agiliocx40gbe, name='Netro'+str(i+1))
                 for i in range(self.num_netronome)] +
                [P4(**tofino, name='core_P4_'+str(i+1))
                 for i in range(self.num_core_sw)] +
                [P4(**tofino, name='agg_P4_'+str(i+1))
                 for i in range(self.num_agg_sw)]
            ),
            queries=(
                [cm_sketch(eps0=self.eps, del0=del0)
                 for i in range(self.num_queries)]
                + []
            ),
        )
        for (dnum, d) in enumerate(inp.devices):
            d.dev_id = dnum
            freeze_object(d)

        # FIXME: different from TreeTopology
        self.has_netro = ([i for i in range(self.num_netronome)]
                          + [-1 for i in
                             range(self.num_hosts - self.num_netronome)])
        np.random.shuffle(self.has_netro)

        g = self.construct_graph(inp.devices, self.has_netro)
        inp.flows = self.get_flows(g, inp)

        return inp

    def get_input(self):
        pickle_name = "pickle_objs/clos-{}-{}-{}-{}".format(
            self.pods, self.query_density, eps0/self.eps, self.num_netronome)
        pickle_loaded = False
        if(os.path.exists(pickle_name)):
            inp_file = open(pickle_name, 'rb')
            inp = pickle.load(inp_file)
            inp_file.close()
            pickle_loaded = True
        else:
            inp = self.create_inp()

        log.info("Building clos topo with {} hosts.".format(self.num_hosts))

        # Recompute overlay
        inp.overlay = self.get_overlay(inp)

        if(pickle_loaded):
            return inp

        # Save inp as pickle
        inp_file = open(pickle_name, 'wb')
        pickle.dump(inp, inp_file)
        inp_file.close()
        return inp

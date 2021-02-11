import math
import os
import pickle
import random
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np

from clustering import Clustering
from common import constants, freeze_object, log
from devices import CPU, P4, Netronome
from flows import flow
from input import Input
from profiles import agiliocx40gbe, beluga20, dc_line_rate, tofino
from sketches import cm_sketch
from traffic import Traffic

eps0 = constants.eps0
del0 = constants.del0


class Topology(ABC):

    def __init__(self):
        self.num_nics = self.num_netronome + self.num_fpga
        self.switches_start_idx = self.num_hosts + self.num_nics

    @abstractmethod
    def get_pickle_name(self):
        raise NotImplementedError

    @abstractmethod
    def construct_graph(self, devices):
        start = 0

        self.hosts = [(devices[i].name, {'id': i, 'remaining': dc_line_rate})
                      for i in range(start, start+self.num_hosts)]
        start += self.num_hosts

        self.nics = [(devices[i].name, {'id': i})
                     for i in range(start, start+self.num_nics)]
        start += self.num_nics

        self.switches = [(devices[i].name, {'id': i})
                         for i in range(start, start+self.num_switches)]

        g = nx.Graph()
        g.add_nodes_from(self.hosts)
        g.add_nodes_from(self.nics)
        g.add_nodes_from(self.switches)
        return g


    def get_device_list(self):
        return (
            [CPU(**beluga20, name='CPU_'+str(i+1))
             for i in range(self.num_hosts)] +
            [Netronome(**agiliocx40gbe, name='Netro'+str(i+1))
             for i in range(self.num_netronome)] +
            [Netronome(**agiliocx40gbe, name='FPGA'+str(i+1))
             for i in range(self.num_fpga)] +
            [P4(**tofino, name='P4_'+str(i+1))
             for i in range(self.num_switches)]
        )

    def supported_overlays(self):
        return ['none', 'tenant', 'spectral', 'spectralA']

    # TODO: Consider splitting this function and moving to Traffic class
    def get_flows(self, g, inp):
        num_tenants = math.ceil(self.num_hosts / self.hosts_per_tenant)
        queries_per_tenant = self.query_density * self.hosts_per_tenant

        flows_per_query = 2
        mean_queries_updated_by_flow = 2
        half_range = mean_queries_updated_by_flow - 1
        low = mean_queries_updated_by_flow - half_range
        high = mean_queries_updated_by_flow + half_range
        assert(queries_per_tenant > high
               and queries_per_tenant <= self.num_queries)

        servers = np.arange(self.num_hosts)
        np.random.shuffle(servers)
        tenant_servers = np.split(servers, num_tenants)

        host_overlay = []
        for x in tenant_servers:
            this_servers = x.tolist()
            this_nics = []
            for s in this_servers:
                nic_id = self.has_nic[s]
                if(nic_id >= 0):
                    dev_id = self.nics[nic_id][1]['id']
                    this_nics.append(dev_id)
            host_overlay.append(this_servers + this_nics)

        inp.tenant_servers = host_overlay
        flows_per_host = (flows_per_query * queries_per_tenant
                          / self.hosts_per_tenant)
        qlist_generator = list(range(queries_per_tenant))

        log.info("Need to generate: {} flows."
                 .format(flows_per_host * self.num_hosts / 2))
        flows = []
        for (tnum, t) in enumerate(tenant_servers):
            query_set = [i + tnum * queries_per_tenant
                         for i in range(queries_per_tenant)]
            for itr in range(int(flows_per_host/2)):
                assigned = t.copy()
                np.random.shuffle(assigned)

                for idx in range(len(t)):
                    h1 = t[idx]
                    h2 = assigned[idx]
                    h1name = self.hosts[h1][0]
                    h2name = self.hosts[h2][0]

                    if(g.nodes[h1name]['remaining'] == 0):
                        continue
                    while(h2 == h1):
                        h2 = t[random.randint(0, len(t)-1)]
                        h2name = self.hosts[h2][0]
                    # Does not seem so correct that you skip the iteration idx
                    if(g.nodes[h2name]['remaining'] == 0):
                        continue

                    (node_path, id_path, capacity) = \
                        Traffic.get_path_with_largest_capacity(
                            g, h1name, h2name)
                    traffic = min(capacity, dc_line_rate/flows_per_host)
                    if(traffic < 0.1):
                        log.error('Need better way to select paths')
                        continue
                    Traffic.update_path_with_traffic(g, node_path, traffic)

                    np.random.shuffle(qlist_generator)
                    queries_for_this_flow = random.randint(low, high)
                    q_list = qlist_generator[:queries_for_this_flow]

                    flows.append(
                        flow(
                            path=id_path,
                            queries=[
                                (
                                    query_set[q_idx],
                                    int(random.random() * 4 + 7)/10
                                )
                                for q_idx in q_list
                            ],
                            thr=traffic
                        )
                    )

        return flows

    def get_input(self):
        pickle_name = self.get_pickle_name()
        pickle_path = os.path.join("pickle_objs", pickle_name)
        pickle_loaded = False
        if(os.path.exists(pickle_path)):
            log.info("Loading " + pickle_name)
            inp_file = open(pickle_path, 'rb')
            inp = pickle.load(inp_file)
            inp_file.close()
            pickle_loaded = True
        else:
            log.info("Building " + pickle_name)
            inp = self.create_inp()

        # Recompute overlay
        inp.overlay = Clustering.get_overlay(inp, self)

        if(pickle_loaded):
            return inp

        # Save inp as pickle
        inp_file = open(pickle_path, 'wb')
        pickle.dump(inp, inp_file)
        inp_file.close()
        return inp

    def create_inp(self):
        inp = Input(
            devices=self.get_device_list(),
            queries=(
                [cm_sketch(eps0=self.eps, del0=del0)
                 for i in range(self.num_queries)]
                + []
            ),
        )
        for (dnum, d) in enumerate(inp.devices):
            d.dev_id = dnum
            freeze_object(d)

        # Move to init
        self.has_nic = ([i for i in range(self.num_nics)]
                        + [-1 for i in
                           range(self.num_hosts
                                 - self.num_nics)])
        # has_nic stores the index of the NIC device relative to all nics
        # to which the host is connected to
        np.random.shuffle(self.has_nic)

        g = self.construct_graph(inp.devices)
        # nx.draw(g, labels=[d.name for d in inp.devices])
        # nx.draw_networkx(g)
        # plt.show()

        flows = self.get_flows(g, inp)
        inp.flows = flows

        return inp

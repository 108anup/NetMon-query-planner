import math
import os
import pickle
import random
import sys

import networkx as nx
import numpy as np

from common import constants, freeze_object, log, log_time
from config import common_config
from devices import CPU, P4, Netronome
from flows import flow
from input import (Input, draw_graph, draw_overlay_over_tenant, fold,
                   generate_overlay, get_complete_graph, get_graph,
                   get_labels_from_overlay, get_spectral_overlay, merge,
                   get_hdbscan_overlay, get_kmedoids_overlay, flatten,
                   get_kmedoids_centers, get_2_level_overlay)
from profiles import agiliocx40gbe, beluga20, tofino, dc_line_rate
from sketches import cm_sketch
import matplotlib.pyplot as plt
import heapq

eps0 = constants.eps0
del0 = constants.del0


class JellyFish(object):

    def __init__(self, tors=20, ports_per_tor=4, num_hosts=16,
                 hosts_per_tenant=8, query_density=2,
                 portion_netronome=0.5, portion_fpga=0.5, overlay='none',
                 eps=eps0):
        # Can't put more than one NIC on one server for now
        # Each server has one port for now
        assert(portion_fpga + portion_netronome <= 1)

        # Built based on my understanding of description from
        # https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final82.pdf
        # All switches have same number of ports
        # Each switch has same number of servers

        self.tors = tors
        self.ports_per_tor = ports_per_tor
        self.num_hosts = num_hosts
        self.query_density = query_density
        self.num_netronome = int(self.num_hosts * portion_netronome)
        self.num_fpga = int(self.num_hosts * portion_fpga)
        self.num_queries = int(query_density * self.num_hosts)
        self.eps = eps

        self.hosts_per_tor = num_hosts / tors
        self.N = tors
        self.k = ports_per_tor
        self.r = int(self.k - self.hosts_per_tor)
        self.overlay = overlay

    def construct_graph(self, devices):
        start = 0
        hosts = [(devices[i].name, {'id': i, 'remaining': dc_line_rate})
                 for i in range(start, start+self.num_hosts)]
        start += self.num_hosts

        nics = [(devices[i].name, {'id': i})
                for i in range(start, start+self.num_netronome + self.num_fpga)]
        start += self.num_netronome + self.num_fpga

        switches = [(devices[i].name, {'id': i})
                    for i in range(start, start+self.tors)]

        g = nx.Graph()
        g.add_nodes_from(hosts)
        g.add_nodes_from(nics)
        g.add_nodes_from(switches)

        # Get a r-regular random graph
        rrg = nx.random_regular_graph(d=self.r, n=self.N)
        for snum1, snum2 in rrg.edges:
            g.add_edge(switches[snum1][0], switches[snum2][0],
                       remaining=dc_line_rate)

        # Connect hosts to switches
        num_switches = len(switches)
        for host_dnum, nic_dnum in enumerate(self.has_nic):
            connect_switch_to = hosts[host_dnum]
            if(nic_dnum != -1):
                g.add_edge(hosts[host_dnum][0], nics[nic_dnum][0],
                           remaining=dc_line_rate)
                connect_switch_to = nics[nic_dnum]
            switch_dnum = host_dnum % num_switches
            if(g.degree(switches[switch_dnum][0]) >= self.ports_per_tor):
                raise AssertionError("Not enough switch "
                                     "ports to support the hosts")
            g.add_edge(connect_switch_to[0], switches[switch_dnum][0],
                       remaining=dc_line_rate)

        switch_degrees = g.degree(list(map(lambda x: x[0], switches)))
        heap = [(degree, sname) for (sname, degree) in switch_degrees]
        heapq.heapify(heap)

        # For the r-regular graph, to be able to keep space for hosts/nics,
        # r was < ports_per_tor. Here we connect any remaining
        # ports on the switches. O(e * nlogn) complexity.
        # (e = O(k - num_hosts/N))
        while(True):
            s1 = heapq.heappop(heap)
            s2 = heapq.heappop(heap)
            if(s1[0] < self.ports_per_tor and s2[0] < self.ports_per_tor):
                g.add_edge(s1[1], s2[1], remaining=dc_line_rate)
                s1 = (s1[0]+1, s1[1])
                s2 = (s2[0]+1, s2[1])
                heapq.heappush(heap, s1)
                heapq.heappush(heap, s2)
            else:
                break

        # Attempt at O(e n**2) version of above.
        # for i in np.arange(num_switches-1, -1, -1):
        #     if(g.degree(switches[i][0]) >= self.ports_per_tor):
        #         continue
        #     for j in np.arange(i-1, -1, -1):
        #         if(g.degree(switches[j][0]) >= self.ports_per_tor):
        #             continue
        #         g.add_edge(switches[i][0], switches[j][0])
        #         break

        self.hosts = hosts
        self.switches = switches
        self.nics = nics

        return g


    # FIXME: Remove redundancy have a common topo class
    def create_inp(self):
        # TODO change to fpga
        inp = Input(
            devices=(
                [CPU(**beluga20, name='CPU_'+str(i+1))
                 for i in range(self.num_hosts)] +
                [Netronome(**agiliocx40gbe, name='Netro'+str(i+1))
                 for i in range(self.num_netronome)] +
                [Netronome(**agiliocx40gbe, name='FPGA'+str(i+1))
                 for i in range(self.num_fpga)] +
                [P4(**tofino, name='P4_'+str(i+1))
                 for i in range(self.tors)]
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

        # FIXME: different from Clos
        self.has_nic = ([i for i in range(self.num_fpga + self.num_netronome)]
                        + [-1 for i in
                           range(self.num_hosts - self.num_netronome - self.num_fpga)])
        # has_nic stores the index of the NIC device relative to all nics
        # to which the host is connected to
        np.random.shuffle(self.has_nic)

        g = self.construct_graph(inp.devices)
        # nx.draw(g, labels=[d.name for d in inp.devices])
        nx.draw_networkx(g)
        plt.show()

        flows = self.get_flows(g, inp)
        inp.flows = flows

        return inp

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

        log.info("Need to generate: {} flows.".format(flows_per_host * self.num_hosts / 2))
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

                    # Clos should ensure if hosts have remaining
                    # capacity then there is a path between them
                    # with enough capacity
                    if(g.nodes[h1name]['remaining'] == 0):
                        continue
                    while(h2 == h1):
                        h2 = t[random.randint(0, len(t)-1)]
                        h2name = self.hosts[h2][0]
                    if(g.nodes[h2name]['remaining'] == 0):
                        continue

                    (node_path, id_path, capacity) = \
                        self.get_path_with_largest_capacity(g, h1, h2)
                    traffic = min(capacity, dc_line_rate/flows_per_host)
                    if(traffic < 0.1):
                        log.error('Need better way to select paths')
                        continue
                    self.update_path_with_traffic(g, node_path, traffic)

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



if (__name__ == '__main__'):
    # rrg = nx.random_regular_graph(d=3, n=20)
    # # nx.draw_networkx(rrg)
    # # plt.show()
    # import ipdb; ipdb.set_trace()

    gen = JellyFish(portion_netronome=0, portion_fpga=0)
    gen.create_inp()

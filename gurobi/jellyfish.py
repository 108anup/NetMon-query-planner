import networkx as nx

from common import constants
from devices import CPU, P4, Netronome
from input import (Input, draw_graph, draw_overlay_over_tenant, fold,
                   generate_overlay, get_complete_graph, get_graph,
                   get_labels_from_overlay, get_spectral_overlay, merge,
                   get_hdbscan_overlay, get_kmedoids_overlay, flatten,
                   get_kmedoids_centers, get_2_level_overlay)
from profiles import agiliocx40gbe, beluga20, tofino, dc_line_rate
import heapq
from topology import Topology

eps0 = constants.eps0
del0 = constants.del0


class JellyFish(Topology):

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
        self.num_nics = self.num_fpga + self.num_netronome
        self.num_queries = int(query_density * self.num_hosts)
        self.eps = eps
        self.hosts_per_tenant = hosts_per_tenant

        self.hosts_per_tor = num_hosts / tors
        self.N = tors
        self.k = ports_per_tor
        self.r = int(self.k - self.hosts_per_tor)
        self.overlay = overlay

        self.switches_start_idx = self.num_hosts + self.num_nics

        # TODO try this
        # super.__init__()

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

    def get_device_list(self):
        # TODO change to FPGA
        return (
                [CPU(**beluga20, name='CPU_'+str(i+1))
                 for i in range(self.num_hosts)] +
                [Netronome(**agiliocx40gbe, name='Netro'+str(i+1))
                 for i in range(self.num_netronome)] +
                [Netronome(**agiliocx40gbe, name='FPGA'+str(i+1))
                 for i in range(self.num_fpga)] +
                [P4(**tofino, name='P4_'+str(i+1))
                 for i in range(self.tors)]
            )

    def get_pickle_name(self):
        return "jelly-{}-{}-{}-{}-{}-{}-{}".format(
            self.N, self.k, self.r, self.query_density, eps0/self.eps,
            self.num_netronome, self.num_fpga)


if (__name__ == '__main__'):
    # rrg = nx.random_regular_graph(d=3, n=20)
    # # nx.draw_networkx(rrg)
    # # plt.show()
    # import ipdb; ipdb.set_trace()

    gen = JellyFish(portion_netronome=0, portion_fpga=0, overlay='tenant')
    inp = gen.get_input()
    import ipdb; ipdb.set_trace()

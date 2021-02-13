import heapq

import networkx as nx

from common import constants
from profiles import dc_line_rate
from topology import Topology


class JellyFish(Topology):

    def __init__(self, tors=20, ports_per_tor=4, num_hosts=16,
                 hosts_per_tenant=8, query_density=2,
                 portion_netronome=0.5, portion_fpga=0.5, overlay='none',
                 eps=constants.eps0):
        # Can't put more than one NIC on one server for now
        # Each server has one port for now
        assert(portion_fpga + portion_netronome <= 1)

        # Built based on my understanding of description from
        # https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final82.pdf
        # All switches have same number of ports
        # Each switch has same number of servers

        self.num_switches = tors
        self.tors = tors
        self.ports_per_tor = ports_per_tor
        self.num_hosts = num_hosts
        self.query_density = query_density
        self.num_netronome = int(self.num_hosts * portion_netronome)
        self.num_fpga = int(self.num_hosts * portion_fpga)
        self.num_queries = int(query_density * self.num_hosts)
        self.eps = eps
        self.hosts_per_tenant = hosts_per_tenant

        self.hosts_per_tor = num_hosts / tors
        self.N = tors
        self.k = ports_per_tor
        self.r = int(self.k - self.hosts_per_tor)
        self.overlay = overlay
        super().__init__()

    def construct_graph(self, devices):
        g = super().construct_graph(devices)
        hosts = self.hosts
        nics = self.nics
        switches = self.switches

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

        return g

    def get_pickle_name(self):
        return "jelly-{}-{}-{}-{}-{}-{}-{}".format(
            self.N, self.k, self.r, self.query_density,
            constants.eps0/self.eps,
            self.num_netronome, self.num_fpga)


if (__name__ == '__main__'):
    # rrg = nx.random_regular_graph(d=3, n=20)
    # # nx.draw_networkx(rrg)
    # # plt.show()
    # import ipdb; ipdb.set_trace()

    gen = JellyFish(portion_netronome=0, portion_fpga=0, overlay='tenant')
    inp = gen.get_input()
    import ipdb; ipdb.set_trace()

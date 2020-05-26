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
                   get_labels_from_overlay, get_spectral_overlay, merge)
from profiles import agiliocx40gbe, beluga20, tofino, dc_line_rate
from sketches import cm_sketch
import matplotlib.pyplot as plt

eps0 = constants.eps0
del0 = constants.del0


class Clos(object):
    '''
    Adapted from:
    https://github.com/frenetic-lang/ocaml-topology/blob/master/scripts/fattree.py
    Grateful to them for this!!
    '''

    def __init__(self, pods=8, query_density=2, hosts_per_tenant=8,
                 portion_netronome=0, overlay='none',
                 eps=eps0):
        assert(pods % 2 == 0)
        self.pods = pods
        self.podsby2 = int(pods/2)
        self.num_hosts = int((pods**3)/4)
        self.num_agg_sw = pods**2
        self.num_core_sw = int((pods**2)/4)
        self.num_netronome = int(self.num_hosts * portion_netronome)
        self.total_devices = (self.num_hosts + self.num_agg_sw
                              + self.num_core_sw + self.num_netronome)
        self.query_density = query_density
        self.num_queries = self.num_hosts * self.query_density
        self.eps = eps0
        self.hosts_per_tenant = min(hosts_per_tenant, self.num_hosts)
        if(self.num_hosts % self.hosts_per_tenant != 0):
            if(pods < 12):
                self.hosts_per_tenant = pods
            elif(pods < 22):
                self.hosts_per_tenant = self.podsby2
        self.overlay = overlay

    def construct_graph(self, devices, has_netro):
        start = 0
        hosts = [(devices[i].name, {'id': i, 'remaining': dc_line_rate})
                 for i in range(start, start+self.num_hosts)]
        start += self.num_hosts

        netronome_nics = [(devices[i].name, {'id': i})
                          for i in range(start, start+self.num_netronome)]
        start += self.num_netronome

        core_switches = [(devices[i].name, {'id': i})
                         for i in range(start, start+self.num_core_sw)]
        start += self.num_core_sw

        # For a given pod:
        # The first self.pods/2 agg sw are in upper layer
        # The next self.pods/2 agg sw are in lower layer
        agg_switches = [(devices[i].name, {'id': i})
                        for i in range(start, start+self.num_agg_sw)]

        g = nx.Graph()
        g.add_nodes_from(hosts)
        g.add_nodes_from(netronome_nics)
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
                    g.add_edge(switch, core_switch,
                               remaining=dc_line_rate)
                    core_offset += 1

                # Connect to aggregate switches in same pod
                for port in range(self.podsby2, self.pods):
                    lower_switch = agg_switches[(pod*self.pods) + port][0]
                    g.add_edge(switch, lower_switch,
                               remaining=dc_line_rate)

            for sw in range(self.podsby2, self.pods):
                switch = agg_switches[(pod*self.pods) + sw][0]

                # Connect to hosts
                for port in range(self.podsby2, self.pods):
                    host = hosts[host_offset][0]

                    netro_offset = has_netro[host_offset]
                    if(netro_offset >= 0):
                        netro = netronome_nics[netro_offset][0]
                        g.add_edge(switch, netro, remaining=dc_line_rate)
                        g.add_edge(netro, host, remaining=dc_line_rate)
                    else:
                        g.add_edge(switch, host, remaining=dc_line_rate)
                    host_offset += 1

        self.hosts = hosts
        self.netronome_nics = netronome_nics
        self.core_switches = core_switches
        self.agg_switches = agg_switches

        # # Check that everything is in order :D
        # pos = nx.spring_layout(g)
        # # labels = nx.draw_networkx_labels(g, pos=pos)
        # labels = {}
        # for d in devices:
        #     labels[d.name] = d.name
        # nx.draw(g, pos=pos, labels=labels)
        # plt.show()
        return g

    @log_time(logger=log.info)
    def get_path_with_largest_capacity(self, g, h1, h2):
        h1name = self.hosts[h1][0]
        h2name = self.hosts[h2][0]

        node_paths = nx.all_shortest_paths(g, h1name, h2name)
        node_paths_capacity = []
        max_capacity = 0
        max_capacity_path = None
        for path in node_paths:
            capacity = dc_line_rate
            for start in range(len(path)-1):
                beg = path[start]
                end = path[start+1]
                capacity = min(capacity, g.edges[beg, end]['remaining'])
            node_paths_capacity.append((path, capacity))
            if(capacity >= max_capacity):
                max_capacity_path = path
                max_capacity = capacity

        # node_paths_capacity = [
        #     (
        #         path,
        #         min(map(lambda x: g.nodes[x]['remaining'], path))
        #     )
        #     for path in node_paths
        # ]
        # (max_capacity_path, capacity) = max(
        #     node_paths_capacity,
        #     key=lambda x: x[1]
        # )
        ids_path = list(map(lambda x: g.nodes[x]['id'],
                            max_capacity_path))
        return (max_capacity_path, ids_path, max_capacity)

    def update_path_with_traffic(self, g, node_path, traffic):
        h1 = node_path[0]
        h2 = node_path[-1]
        g.nodes[h1]['remaining'] -= traffic
        g.nodes[h2]['remaining'] -= traffic
        for start in range(len(node_path)-1):
            beg = node_path[start]
            end = node_path[start+1]
            g.edges[beg, end]['remaining'] -= traffic
            # if('debug' in g.edges[beg, end]):
            #     g.edges[beg, end]['debug'].append((h1, h2, traffic))
            # else:
            #     g.edges[beg, end]['debug'] = [(h1, h2, traffic)]

    # @log_time(logger=log.info)
    def get_flows(self, g, inp):
        # log.info("Generating flows")
        # import ipdb; ipdb.set_trace()
        # query_density means queries per host
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

        # servers = servers.tolist()
        # tenant_servers = []
        # taken = 0
        # total_servers = len(servers)
        # for i in range(num_tenants):
        #     this_num_servers = random.randint(self.hosts_per_tenant-1,
        #                                       self.hosts_per_tenant+1)
        #     if(this_num_servers + taken > total_servers):
        #         tenant_servers.append(servers[taken:])
        #         taken = total_servers
        #         break

        #     tenant_servers.append(servers[taken:this_num_servers+taken])
        #     taken += this_num_servers

        # if(taken < total_servers):
        #     tenant_servers.append(servers[taken:])

        host_overlay = []
        for x in tenant_servers:
            this_servers = x.tolist()
            # this_servers = x
            this_netro = []
            for s in this_servers:
                netro_id = self.has_netro[s]
                if(netro_id >= 0):
                    dev_id = self.netronome_nics[netro_id][1]['id']
                    this_netro.append(dev_id)
            host_overlay.append(this_servers + this_netro)

        inp.tenant_servers = host_overlay
        inp.tenant_overlay = (host_overlay
                              + generate_overlay(
                                  [self.num_core_sw + self.num_agg_sw],
                                  self.num_hosts + self.num_netronome))

        flows_per_host = (flows_per_query * queries_per_tenant
                          / self.hosts_per_tenant)
        qlist_generator = list(range(queries_per_tenant))

        log.info("Need to generate: {} flows.".format(flows_per_host * self.num_hosts / 2))
        flows = []
        for (tnum, t) in enumerate(tenant_servers):
            query_set = [i + tnum * queries_per_tenant
                         for i in range(queries_per_tenant)]
            # for itr in range(queries_per_tenant * flows_per_query):
            #     h1 = t[random.randint(0, self.hosts_per_tenant-1)]
            #     h2 = t[random.randint(0, self.hosts_per_tenant-1)]
            #     while(h2 == h1):
            #         h2 = t[random.randint(0, self.hosts_per_tenant-1)]

            # for h1 in t:
            #     while(self.hosts[h1][1]['remaining_flows'] > 0):
            #         h2 = t[random.randint(0, self.hosts_per_tenant-1)]
            #         while(h2 == h1 or self.hosts[h2][1]['remaining_flows'] == 0):
            #             h2 = t[random.randint(0, self.hosts_per_tenant-1)]

            #         self.hosts[h1][1]['remaining_flows'] -= 1
            #         self.hosts[h2][1]['remaining_flows'] -= 1

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

    def get_tenant_overlay_switches(self, inp):
        devices_per_cluster = common_config.MAX_CLUSTERS_PER_CLUSTER
        if(common_config.solver == 'Netmon'):
            devices_per_cluster = common_config.MAX_DEVICES_PER_CLUSTER
        host_nic_overlay = inp.tenant_servers

        num_tenant_clusters_to_merge = math.ceil(devices_per_cluster
                                                 / self.hosts_per_tenant)
        if(num_tenant_clusters_to_merge > 1):
            host_nic_overlay = merge(host_nic_overlay,
                                     num_tenant_clusters_to_merge)

        # Assign switches to clusters:
        dev_id_to_cluster_id = dict()
        for cnum, c in enumerate(host_nic_overlay):
            for dnum in c:
                dev_id_to_cluster_id[dnum] = cnum

        g = get_complete_graph(inp)
        switches_start_idx = self.num_hosts + self.num_netronome
        total_devices = len(inp.devices)
        seq = 0
        total_clusters = len(host_nic_overlay)

        for snum in range(switches_start_idx, total_devices):
            devs = g.neighbors(snum)
            best_dnum, edge_count = -1, 0
            for dnum in devs:
                if(dnum in dev_id_to_cluster_id):
                    edges = g.number_of_edges(snum, dnum)
                    if(edges > edge_count):
                        edge_count = edges
                        best_dnum = dnum

            if(best_dnum == -1):
                cnum = seq % total_clusters
                seq += 1
            else:
                cnum = dev_id_to_cluster_id[best_dnum]
            host_nic_overlay[cnum].append(snum)
            dev_id_to_cluster_id[snum] = cnum

        if(total_clusters > common_config.MAX_CLUSTERS_PER_CLUSTER):
            host_nic_overlay = fold(host_nic_overlay,
                                    common_config.MAX_CLUSTERS_PER_CLUSTER)

        overlay = host_nic_overlay

        # inp.overlay = overlay
        # node_labels = {}
        # for d in inp.devices:
        #     node_labels[d.dev_id] = d.name
        # node_colors = get_labels_from_overlay(inp, inp.overlay)
        # g = get_graph(inp)
        # draw_graph(g, node_colors, node_labels)

        if(len(overlay) == 1 and isinstance(overlay[0], list)):
            return overlay[0]

        return overlay

    def get_overlay(self, inp):
        assert(self.overlay in ['tenant', 'none', 'spectral'])
        overlay = None
        if('spectral' in self.overlay):
            overlay = self.get_spectral_overlay(inp)
            if(len(overlay) > common_config.MAX_CLUSTERS_PER_CLUSTER):
                overlay = fold(overlay, common_config.MAX_CLUSTER_PER_CLUSTER)
        elif(self.overlay == 'tenant'):
            overlay = self.get_tenant_overlay_switches(inp)

        if(overlay):
            if(len(overlay) == 1
               and isinstance(overlay[0], list)):
                # Ideally this should be handled by prior functions
                assert(False)
                overlay = overlay[0]

            if(len(overlay) == len(inp.devices)):
                log.info("Not clustering as topology is fairly small")
                log.info("-"*50)
                return None
            else:
                return overlay
        return overlay

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
        flows = self.get_flows(g, inp)
        inp.flows = flows

        return inp

    def get_input(self):
        pickle_name = "pickle_objs/clos-{}-{}-{}-{}".format(
            self.pods, self.query_density, eps0/self.eps, self.num_netronome)
        pickle_loaded = False
        if(os.path.exists(pickle_name)):
            log.info("Fetching clos topo with {} hosts.".format(self.num_hosts))
            inp_file = open(pickle_name, 'rb')
            inp = pickle.load(inp_file)
            inp_file.close()
            pickle_loaded = True
        else:
            log.info("Building clos topo with {} hosts.".format(self.num_hosts))
            inp = self.create_inp()

        # Recompute overlay
        inp.overlay = self.get_overlay(inp)

        if(pickle_loaded):
            return inp

        # Save inp as pickle
        inp_file = open(pickle_name, 'wb')
        pickle.dump(inp, inp_file)
        inp_file.close()
        return inp

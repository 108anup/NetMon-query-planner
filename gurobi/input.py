import math
import os
import pickle
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering

from common import Namespace, log, log_time, memoize
from config import common_config
from devices import CPU, P4
from flows import flow
from sketches import cm_sketch

# Stub file for providing input to solver

"""
TODO:
Provide wrapper over flow abstraction.
How to convert to flow abstraction below:

1. specify monitoring based on OD pairs. Then ingress routers figure out
based on src IP, dst IP and routing information what the egress port will be.
They include this information in the packet headers

2. flow filter based specification:
e.g. src IP == x.x.x.x and dst port == xx
Then figure out what all unique paths can such traffic take
and then create a mipflow for each of those paths.
"""

"""
TODO:
Temporal Multiplexing

1. If the operator specifies absolute errors in measurement.
Then we have an opportunity to modify epsilon based on the traffic moment.
Each sketch additionally maintains 'F_1 = m' observed in the measurement
interval based on that epsilon can be updated.
=> Can predict what future F_1 will be using RNNs (short term) OR
time of day methods (long term).
"""

eps0 = 0.1 * 8 / 128  # 1e-5
del0 = 0.05  # 0.02

# One time profiling of each device type
beluga20 = {
    'mem_par': [0, 1.1875, 32, 1448.15625,
                5792.625, 32768.0, 440871.90625],
    'mem_ns': [0, 0.539759, 0.510892, 5.04469,
               5.84114, 30.6627, 39.6981],
    'Li_size': [32, 256, 8192, 32768],
    'Li_ns': [0.53, 1.5, 3.7, 36],
    'hash_ns': 3.5, 'cores': 7, 'dpdk_single_core_thr': 35,
    'max_mem': 32768, 'max_rows': 12, 'line_thr': 98
}

tofino = {
    'meter_alus': 4, 'sram': 48, 'stages': 12, 'line_thr': 148,
    'max_mpr': 48, 'max_mem': 48*12, 'max_rows': 12 * 4
}

'''
    CPU(mem_par=[0, 1.1875, 32, 1448.15625,
                 5792.625, 32768.0, 440871.90625],
        mem_ns=[0, 0.539759, 0.510892, 5.04469,
                5.84114, 30.6627, 39.6981],
        Li_size=[32, 256, 8192, 32768],
        Li_ns=[0.53, 1.5, 3.7, 36],
        hash_ns=3.5, cores=7, dpdk_single_core_thr=35,
        max_mem=32768, max_rows=9, name='CPU_1'),
    P4(meter_alus=4, sram=48, stages=12, line_thr=148,
       max_mpr=48, max_mem=48*12, max_rows=12, name='P4_1'),

'''


def remove_empty(l):
    re = [e for e in l if len(e) > 0]
    # Also flatten single device cluster
    ret = []
    for e in re:
        if(isinstance(e, list) and len(e) == 1):
            ret.extend(e)
        else:
            ret.append(e)
    return ret


def get_graph(inp):
    g = nx.MultiGraph()
    g.add_nodes_from(range(len(inp.devices)))
    for f in inp.flows:
        p = f.path
        a = p[0]
        for b in p[1:]:
            g.add_edge(a, b)
            a = b
    return g


def get_complete_graph(inp):
    g = nx.MultiGraph()
    g.add_nodes_from(range(len(inp.devices)))
    for f in inp.flows:
        p = f.path
        n = len(p)
        for i in range(n):
            for j in range(i+1, n):
                g.add_edge(p[i], p[j])
    return g


def remap_colors(colors):
    m = np.max(colors)
    freq = [0 for i in range(m+1)]
    for cnum in colors:
        freq[cnum] += 1
    remap = [0 for i in range(m+1)]
    cur_clr = 0
    for cnum, f in enumerate(freq):
        if(f > 0):
            remap[cnum] = cur_clr
            cur_clr += 1
    remapped = [0 for i in range(len(colors))]
    for i in range(len(colors)):
        remapped[i] = remap[colors[i]]
    return remapped


def draw_graph(G, colors, labels=None):
    colors = remap_colors(colors)

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    if(labels):
        temp = {x: labels[x] for x in G.nodes}
        labels = temp
    nx.draw(G, pos, node_color=colors, labels=labels,
            cmap=plt.get_cmap('Spectral'))
    plt.show()


# Only for tree (acyclic graph)
def dfs(nodes, data):
    data.color += 1
    for n in nodes:
        if(not isinstance(n, list)):
            data.node_colors[n] = data.color

    for n in nodes:
        if(isinstance(n, list)):
            dfs(n, data)


def get_labels_from_overlay(inp, overlay):
    data = Namespace(color=-1, node_colors=list(range(len(inp.devices))))
    dfs(overlay, data)
    return data.node_colors


def draw_overlay_over_tenant(inp):
    node_labels = get_labels_from_overlay(inp, inp.tenant_overlay)
    node_colors = get_labels_from_overlay(inp, inp.overlay)
    g = get_graph(inp)
    draw_graph(g, node_colors, node_labels)


def draw_overlay(inp):
    g = get_graph(inp)
    node_colors = get_labels_from_overlay(inp, inp.overlay)

    # # Assuming overlay is list of lists
    # color = 0
    # for l in inp.overlay:
    #     for x in l:
    #         node_colors[x] = color
    #     color += 1

    draw_graph(g, node_colors)


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def UnnormalizedSpectral(W, nc=2):
    assert(check_symmetric(W.todense()))
    D = np.diag(np.sum(W, axis=1))
    L = D - W

    e, v = np.linalg.eigh(L)
    idx = e.argsort()
    e = e[idx]
    v = v[:, idx]

    U = np.array(v[:, :nc])
    km = KMeans(init='k-means++', n_clusters=nc)
    r = km.fit_predict(U)
    return r


@log_time(logger=log.info)
def get_spectral_overlay(inp, comp={}, normalized=True, affinity=False):
    log.info("Buliding spectral overlay with {} devices"
             .format(len(inp.devices)))
    if(affinity):
        g = get_complete_graph(inp)
    else:
        g = get_graph(inp)

    overlay = []

    num_comp = 0
    node_colors = list(g.nodes)
    last_inc = 0

    for c in nx.connected_components(g):
        num_comp += 1
        cg = g.subgraph(c)
        i2n = list(cg.nodes)
        if(len(i2n) > 1):
            adj = nx.adjacency_matrix(cg)
            nc = 2 * math.ceil(len(i2n)
                               / common_config.max_devices_per_cluster)
            if(normalized):
                sc = SpectralClustering(nc, affinity='precomputed',
                                        n_init=100, assign_labels='discretize')
                cluster_assignment = sc.fit_predict(adj)
            else:
                cluster_assignment = UnnormalizedSpectral(adj, nc)
            # draw_graph(cg, cluster_assignment)
            sub_overlay = [[] for i in range(nc)]

            for dnum, cnum in enumerate(cluster_assignment):
                node_colors[i2n[dnum]] = last_inc + cnum
                if(not i2n[dnum] in comp):
                    sub_overlay[cnum].append(i2n[dnum])
            last_inc += nc

            filtered_sub_overlay = remove_empty(sub_overlay)
            if(len(filtered_sub_overlay) == 1):
                overlay.extend(filtered_sub_overlay)
            elif(len(filtered_sub_overlay) > 1):
                overlay.append(filtered_sub_overlay)
        else:
            overlay.append(i2n[0])
            node_colors[i2n[0]] = last_inc
            last_inc += 1

    # draw_graph(g, node_colors)

    if(len(overlay) == 1 and isinstance(overlay[0], list)):
        return overlay[0]
    return overlay


def shift_overlay(overlay):
    last = overlay[-1]
    l_len = len(last)
    hlen = int(l_len/2)
    new_overlay = []
    lfh = last[:hlen]
    prev_sh = last[hlen:]
    for ol in overlay[:-1]:
        ol_len = len(ol)
        hlen = int(ol_len/2)
        fh = ol[:hlen]
        sh = ol[hlen:]
        new_overlay.append(prev_sh + fh)
        prev_sh = sh
    new_overlay.append(prev_sh + lfh)
    return new_overlay


def generate_overlay(nesting, start_idx=0):
    if(len(nesting) == 0):
        return int(start_idx)
    # if(len(nesting) == 1):
    #     return [start_idx + i for i in range(nesting[0])]
    else:
        num_elements = np.prod(nesting[1:])
        return [generate_overlay(nesting[1:], start_idx + i*num_elements)
                for i in range(nesting[0])]


def merge(l, n=2):
    ret = []
    cur = []
    itr = 0
    for x in l:
        cur += x
        itr += 1
        if(itr == n):
            ret.append(cur)
            cur = []
            itr = 0
    if(len(cur) > 0):
        ret.append(cur)
    return ret


def fold(l, n=4):
    ret = []
    cur = []
    itr = 0
    for x in l:
        cur.append(x)
        itr += 1
        if(itr == n):
            ret.append(cur)
            cur = []
            itr = 0
    if(len(cur) > 0):
        ret.append(cur)
    return ret


class Input(Namespace):

    @property
    @memoize
    def device_to_id(self):
        _device_to_id = {}
        for (dnum, d) in enumerate(self.devices):
            _device_to_id[d] = dnum
        return _device_to_id


def dc_topology(hosts_per_tors=2, tors_per_l1s=2, l1s=2,
                num_queries=80, eps=eps0, overlay='none', tenant=False,
                refine=False, queries_per_tenant=4, hosts_per_tenant=8):
    pickle_name = "pickle_objs/inp-{}-{}-{}-{}-{}".format(
        hosts_per_tors, tors_per_l1s, l1s, num_queries, eps0/eps)
    pickle_loaded = False
    if(os.path.exists(pickle_name)):
        inp_file = open(pickle_name, 'rb')
        inp = pickle.load(inp_file)
        inp_file.close()
        pickle_loaded = True

    hosts = hosts_per_tors * tors_per_l1s * l1s
    tors = tors_per_l1s * l1s
    hosts_tors = hosts + tors
    hosts_tors_l1s = hosts_tors + l1s

    if(not pickle_loaded):
        def get_path(h1, h2):
            while(h1 == h2):
                h2 = random.randint(0, hosts-1)
            tor1 = int(h1 / hosts_per_tors)
            tor2 = int(h2 / hosts_per_tors)
            l11 = int(tor1 / tors_per_l1s)
            l12 = int(tor2 / tors_per_l1s)
            tor1 = tor1 + hosts
            tor2 = tor2 + hosts
            l11 = l11 + hosts_tors
            l12 = l12 + hosts_tors
            l2 = hosts_tors_l1s
            if(l11 == l12):
                if(tor1 == tor2):
                    if(h1 == h2):
                        return tuple([h1])
                    else:
                        return (h1, tor1, h2)
                else:
                    return (h1, tor1, l11, tor2, h2)
            else:
                return (h1, tor1, l11, l2, l12, tor2, h2)

        inp = Input(
            devices=(
                [CPU(**beluga20, name='CPU'+str(i+1))
                 for i in range(hosts)] +
                [P4(**tofino, name='tor_P4'+str(i+1))
                 for i in range(int(tors))] +
                # [CPU(**beluga20, name='tor_CPU'+str(i+1))
                #  for i in range(int(tors/2))] +
                [P4(**tofino, name='l1_P4'+str(i+1))
                 for i in range(l1s)] +
                [P4(**tofino, name='l2_P4')]
            ),
            queries=(
                [cm_sketch(eps0=eps, del0=del0) for i in range(num_queries)]
                + []
                # [cm_sketch(eps0=eps0*10, del0=del0) for i in range(24)] +
                # [cm_sketch(eps0=eps0, del0=del0) for i in range(32)]
            ),
        )

        if(tenant):
            flows = []

            flows_per_query = 2
            # each tenant exclusively owns 8 hosts
            # and has 4 queries they want measured
            num_tenants = hosts / hosts_per_tenant
            assert(num_queries == hosts / (hosts_per_tenant / queries_per_tenant))

            # the 8 hosts are randomly assigned to tenants
            servers = np.arange(hosts)
            np.random.shuffle(servers)
            tenant_servers = np.split(servers, num_tenants)

            inp.tenant_servers = tenant_servers
            host_overlay = [x.tolist() for x in inp.tenant_servers]
            inp.tenant_overlay = (host_overlay
                                  + generate_overlay([tors + l1s + 1], hosts))

            for (tnum, t) in enumerate(tenant_servers):
                query_set = [i + tnum*queries_per_tenant for i in range(queries_per_tenant)]

                # each tenant has 8 different OD pairs,
                # traffic between which needs to be measured
                for itr in range(queries_per_tenant * flows_per_query):
                    h1 = t[random.randint(0, hosts_per_tenant-1)]
                    h2 = t[random.randint(0, hosts_per_tenant-1)]
                    cov = int(random.random() * 4 + 7)/10
                    q = query_set[random.randint(0, queries_per_tenant-1)]
                    flows.append(
                        flow(
                            path=get_path(h1, h2),
                            queries=[(q, cov)]
                        )
                    )
            inp.flows = flows
        else:
            inp.flows = [
                flow(
                    path=get_path(random.randint(0, hosts-1),
                                  random.randint(0, hosts-1)),
                    queries=[
                        (random.randint(0, num_queries-1),
                         int(random.random() * 4 + 7)/10)
                    ]
                )
                for flownum in range(max(hosts, num_queries) * 5)
            ]

    inp.refine = refine

    if(overlay == 'tor'):
        if(hosts_per_tors <= 8):
            inp.overlay = (generate_overlay([tors, hosts_per_tors])
                           + generate_overlay([tors + l1s + 1], hosts))
        elif(tors <= 20):  # Assuming hosts_per_tors multiple of 8
            inp.overlay = (generate_overlay([tors, int(hosts_per_tors/8), 8])
                           + generate_overlay([tors + l1s + 1], hosts))
        else:  # Assuming tors multiple of 20
            inp.overlay = (generate_overlay([tors, int(hosts_per_tors/8), 8])
                           + generate_overlay([int(tors/20), 20], hosts)
                           + generate_overlay([l1s + 1], hosts + tors))

    elif(overlay == 'shifted'):
        if(hosts_per_tors <= 8):
            inp.overlay = (
                shift_overlay(generate_overlay([tors, hosts_per_tors]))
                + generate_overlay([tors + l1s + 1], hosts)
            )

    # Heuristic: don't cluster nodes with high responsibility:
    elif('spectral' in overlay):
        if(overlay == 'spectralU'):
            host_overlay = get_spectral_overlay(
                inp, comp=set(hosts + i for i in range(tors + l1s + 1)),
                normalized=False)
            inp.overlay = (host_overlay
                           + generate_overlay([tors + l1s + 1], hosts))
        if(overlay == 'spectralA'):
            host_overlay = get_spectral_overlay(
                inp,  # comp=set(hosts + i for i in range(tors + l1s + 1)),
                affinity=True)
            inp.overlay = host_overlay
        else:
            host_overlay = get_spectral_overlay(
                inp, comp=set(hosts + i for i in range(tors + l1s + 1)))
            inp.overlay = (host_overlay
                           + generate_overlay([tors + l1s + 1], hosts))

        #if(num_queries == 480):
        #    draw_overlay_over_tenant(inp)

    elif(overlay == 'tenant'):
        host_overlay = [x.tolist() for x in inp.tenant_servers]
        if(hosts > 10000):
            host_overlay = merge(host_overlay, 120)
            # leaves = hosts / 8 / 120
            tors_per_leaf = math.ceil(tors / len(host_overlay))

            # TODO:: better way to do assign tors to clusters!!
            assert(tors % tors_per_leaf == 0)
            tor_overlay = generate_overlay(
                [int(tors/tors_per_leaf), tors_per_leaf], hosts)

            for i in range(len(tor_overlay)):
                host_overlay[i].extend(tor_overlay[i])

            # tors_overlay = generate_overlay([int(tors/x), x], hosts)
            # host_overlay = fold(host_overlay, 1000)
            inp.overlay = (host_overlay
                           + generate_overlay([l1s + 1], hosts + tors))
        else:
            inp.overlay = (host_overlay
                           + generate_overlay([tors + l1s + 1], hosts))
        # if(tors <= 20):
        #     inp.overlay = (host_overlay
        #                    + generate_overlay([tors + l1s + 1], hosts))
        # else:
        #     # Assuming tors multiple of 20
        #     # TODO:: Redundancy
        #     inp.overlay = (host_overlay
        #                    + generate_overlay([int(tors/20), 20], hosts)
        #                    + generate_overlay([l1s + 1], hosts + tors))
        # draw_overlay(inp)

    elif(overlay == 'random'):
        ov = np.array(range(hosts))
        np.random.shuffle(ov)
        host_overlay = np.array_split(
            ov, math.ceil(hosts/common_config.max_devices_per_cluster))
        ho = [e.tolist() for e in host_overlay]
        inp.overlay = (ho
                       + generate_overlay([tors + l1s + 1], hosts))

    elif(overlay == 'none'):
        inp.overlay = None

    if(pickle_loaded):
        return inp

    inp_file = open(pickle_name, 'wb')
    pickle.dump(inp, inp_file)
    inp_file.close()
    return inp


# All memory measured in KB unless otherwise specified
input_generator = [

    # 0
    # Bad for vanilla Univmon (puts much load on CPU)
    Input(
        # Change when devices are added / removed
        devices=[
            CPU(**beluga20, name='CPU_1'),
            P4(**tofino, name='P4_1'),
        ],
        # Change when metrics are added / removed
        queries=[
            cm_sketch(eps0=eps0, del0=del0),
        ],
        # Change when metric filters are modified
        flows=[
            flow(path=(0, 1), queries=[(0, 1)]),
        ]
    ),

    # 1
    # Bad for UnivmonGreedy (puts too many rows on CPU)
    Input(
        devices=[
            CPU(**beluga20, name='CPU_1'),
            P4(**tofino, name='P4_1'),
        ],
        queries=[
            cm_sketch(eps0=eps0/1000, del0=del0),
            cm_sketch(eps0=eps0/10, del0=del0),
            cm_sketch(eps0=eps0/10, del0=del0),
        ],
        flows=[
            flow(path=(0, 1), queries=[(0, 1), (1, 1), (2, 1)]),
        ]
    ),

    # 2
    # Bad for UnivmonGreedyRows (puts too much load on P4)
    # CPU can handle extra memory load with same core budget
    # P4 memory exhausted!
    Input(
        devices=[
            CPU(**beluga20, name='CPU_1'),
            P4(**tofino, name='P4_1'),
        ],
        queries=[
            cm_sketch(eps0=eps0/5000, del0=del0),
        ],
        flows=[
            flow(path=(0, 1), queries=[(0, 1)])
        ]
    ),

    # 3
    # small dc topology, more sketches

    # Bad for Netmon when very large inputs
    # Partitioning helps for UnivmonGreedyRows

    # Full sketches only Netmon is better than Univmon*
    dc_topology(),

    # 4 - same as 0
    # P4 priority over CPU when everything fits on P4
    # Bad for Univmon
    Input(
        devices=[
            CPU(**beluga20, name='CPU_1'),
            P4(**tofino, name='P4_1'),
        ],
        queries=[
            cm_sketch(eps0=eps0*50, del0=del0),
            cm_sketch(eps0=eps0/5, del0=del0),
            cm_sketch(eps0=eps0*100, del0=del0/2)
        ],
        flows=[
            flow(path=(0, 1), queries=[(0, 1), (1, 1), (2, 1)])
        ]
    ),

    # 5 - same as 11
    # Skewed CPU allocation
    # Bad for Univmon* -> does not know within CPU
    Input(
        devices=[
            CPU(**beluga20, name='CPU_1'),
            P4(**tofino, name='P4_1'),
            CPU(**beluga20, name='CPU_2'),
        ],
        queries=[
            cm_sketch(eps0=eps0*5, del0=del0),
            cm_sketch(eps0=eps0/2000, del0=del0),
            cm_sketch(eps0=eps0*100, del0=del0/2)
        ],
        flows=[
            flow(path=(0, 1, 2), queries=[(0, 1), (1, 1), (2, 1)])
        ]
    ),

    # 6 - same as 11
    # Skewed CPU allocation 2
    # Bad for Univmon* -> does not know within CPU
    Input(
        devices=[
            CPU(**beluga20, name='CPU_1'),
            P4(**tofino, name='P4_1'),
            CPU(**beluga20, name='CPU_2'),
        ],
        queries=[
            cm_sketch(eps0=eps0*5, del0=del0),
            cm_sketch(eps0=eps0/5000, del0=del0),
            cm_sketch(eps0=eps0*100, del0=del0/2)
        ],
        flows=[
            flow(path=(0, 1, 2), queries=[(0, 1), (1, 1), (2, 1)])
        ]
    ),

    # 7 - has both effects of 11 and 12
    # Use small sketches for fully utilizing CPUs
    # Bad for UnivmonGreedyRows exhausts P4 memory
    # Bad for UnivmonGreedy_ns / vanilla Univmon (put many rows on CPU)
    Input(
        devices=[
            CPU(**beluga20, name='CPU_1'),
            CPU(**beluga20, name='CPU_2'),
            P4(**tofino, name='P4_1'),
            P4(**tofino, name='P4_2'),
        ],
        queries=[
            cm_sketch(eps0=eps0*12/1000, del0=del0),
            cm_sketch(eps0=eps0/5000, del0=del0),
            cm_sketch(eps0=eps0/10, del0=del0/2)
        ],
        flows=[
            flow(path=(0, 1, 2, 3), queries=[(0, 1), (1, 1), (2, 1)]),
        ]
        # queries=[cm_sketch(eps0=eps0*12, del0=del0),
        #          cm_sketch(eps0=eps0/5, del0=del0),
        #          cm_sketch(eps0=eps0*100, del0=del0/2)]
    ),

    # 8 - sanity check
    # Multi P4
    # Nothing matters as continuous resource allocation
    Input(
        devices=[
            P4(**tofino, name='P4_1'),
            P4(**tofino, name='P4_2'),
        ],
        queries=[
            cm_sketch(eps0=eps0/20, del0=del0),
        ],
        flows=[
            flow(path=(0, 1), queries=[(0, 1)])
        ]
    ),

    # 9
    # small dc topology, large sketches
    dc_topology(num_queries=64, eps=eps0/10),

    # 10
    # Mem vary - CPU - P4
    Input(
        devices=[
            CPU(**beluga20, name='CPU_1'),
            P4(**tofino, name='P4_1'),
        ],
        queries=[cm_sketch(eps0=eps0*5, del0=del0)]
    ),

    # 11
    # Pressure at network core
    Input(
        devices=(
            [CPU(**beluga20, name='CPU_{}'.format(i)) for i in range(20)] +
            [P4(**tofino, name='P4_1')]
        ),
        queries=[
            cm_sketch(eps0=eps0, del0=del0) for i in range(20)
        ],
        flows=[flow(path=(i, 20, (i + 1) % 20), queries=[(i, 1)])
               for i in range(20)]
    ),

    # 12
    # Pressure at network core, core is now CPU
    Input(
        devices=(
            [P4(**tofino, name='P4_{}'.format(i)) for i in range(20)] +
            [CPU(**beluga20, name='CPU_1')]
        ),
        queries=[
            cm_sketch(eps0=eps0/1000, del0=del0)],
        flows=[flow(path=(i, 20, (i + 1) % 20), queries=[(0, 1)])
               for i in range(20)]
    ),

    # 13
    # Pressure at network core, core is now CPU
    # small sketches
    Input(
        devices=(
            [P4(**tofino, name='P4_{}'.format(i)) for i in range(20)] +
            [CPU(**beluga20, name='CPU_1')]
        ),
        queries=[
            cm_sketch(eps0=eps0, del0=del0)],
        flows=[flow(path=(i, 20, (i + 1) % 20), queries=[(0, 1)])
               for i in range(20)]
    ),

    # 14
    # Large Topo
    dc_topology(hosts_per_tors=48, num_queries=256),

    # 15
    # Very Large
    dc_topology(hosts_per_tors=48, tors_per_l1s=20,
                l1s=10, num_queries=5, overlay='tor'),

    # 16
    # Overlay test 1
    Input(
        devices=(
            [CPU(**beluga20, name='CPU_{}'.format(i)) for i in range(2)] +
            [P4(**tofino, name='P4_{}'.format(i+2)) for i in range(4)]
        ),
        queries=[
            cm_sketch(eps0=eps0, del0=del0) for i in range(3)
        ],
        flows=[
            flow(path=(0, 3, 5, 1), queries=[(0, 0.6), (1, 0.8)]),
            flow(path=(2, 3, 4), queries=[(1, 0.6), (2, 0.9)]),
            flow(path=(5, 1), queries=[(0, 1)])
        ],
    ),

    # 17
    # Overlay test 2
    Input(
        devices=(
            [CPU(**beluga20, name='CPU_{}'.format(i)) for i in range(2)] +
            [P4(**tofino, name='P4_{}'.format(i+2)) for i in range(4)]
        ),
        queries=[
            cm_sketch(eps0=eps0, del0=del0) for i in range(3)
        ],
        flows=[
            flow(path=(0, 3, 5, 1), queries=[(0, 0.6), (1, 0.8)]),
            flow(path=(2, 3, 4), queries=[(1, 0.6), (2, 0.9)]),
            flow(path=(5, 1), queries=[(0, 1)])
        ],
        overlay=[[0, 3, 2], [4, 5, 1]]
    ),

    # 18 overlay on 11
    # Pressure at network core
    Input(
        devices=(
            [CPU(**beluga20, name='CPU_{}'.format(i)) for i in range(20)] +
            [P4(**tofino, name='P4_1')]
        ),
        queries=[
            cm_sketch(eps0=eps0, del0=del0) for i in range(20)
        ],
        flows=[flow(path=(i, 20, (i + 1) % 20), queries=[(i, 1)])
               for i in range(20)],
        overlay=[[i + j*4 for i in range(4)] for j in range(5)] + [20]
    ),

    # 19 overlay on 12
    # Pressure at network core, core is now CPU
    Input(
        devices=(
            [P4(**tofino, name='P4_{}'.format(i)) for i in range(20)] +
            [CPU(**beluga20, name='CPU_1')]
        ),
        queries=[
            cm_sketch(eps0=eps0/1000, del0=del0)],
        flows=[flow(path=(i, 20, (i + 1) % 20), queries=[(0, 1)])
               for i in range(20)],
        overlay=[[i + j*4 for i in range(4)] for j in range(5)] + [20]
    ),

    # 20
    # overlay, small dc
    dc_topology(overlay='shifted'),

    # 21
    Input(
        devices=(
            [CPU(**beluga20, name='CPU_{}'.format(i)) for i in range(5)]
        ),
        queries=[
            cm_sketch(eps0=eps0, del0=del0) for i in range(4)
        ],
        flows=[flow(path=(i, 4, (i + 1) % 4), queries=[(i, 1)])
               for i in range(4)],
    ),

    # 22
    Input(
        devices=(
            [CPU(**beluga20, name='CPU_{}'.format(i)) for i in range(5)]
        ),
        queries=[
            cm_sketch(eps0=eps0, del0=del0) for i in range(4)
        ],
        flows=[flow(path=(i, 4, (i + 1) % 4), queries=[(i, 1)])
               for i in range(4)],
        overlay=[[i + j*2 for i in range(2)] for j in range(2)] + [4]
    ),

    # 23 modified 18 overlay on 11
    # Pressure at network core
    Input(
        devices=(
            [CPU(**beluga20, name='CPU_{}'.format(i)) for i in range(20)] +
            [P4(**tofino, name='P4_1')]
        ),
        queries=[
            cm_sketch(eps0=eps0, del0=del0) for i in range(20)
        ],
        flows=[flow(path=(i, 20, (i + 1) % 20), queries=[(i, 1)])
               for i in range(20)],
        overlay=[[i + j*5 for i in range(5)] for j in range(4)] + [20]
    ),

    # 24
    # Small tenant (100)
    dc_topology(hosts_per_tors=8, num_queries=4*40, tenant=True,
                eps=eps0, overlay='spectralA', refine=False,
                queries_per_tenant=40),

    # 25
    # Large tenant (10K)
    dc_topology(hosts_per_tors=48, tors_per_l1s=20,
                l1s=10, num_queries=4800, tenant=True),

    # 26 Clustering Intuition init cant help
    Input(
        devices=[CPU(**beluga20, name='CPU_{}'.format(i)) for i in range(5)],
        queries=[cm_sketch(eps0=eps0, del0=del0)],
        flows=[flow(path=(0, 3), queries=[(0, 1)]),
               flow(path=(1, 3), queries=[(0, 1)]),
               flow(path=(2, 4), queries=[(0, 1)])],
        overlay=[[0, 1, 2], 3, 4]
    ),

    # 27 Clustering Intuition init could help
    Input(
        devices=[CPU(**beluga20, name='CPU_{}'.format(i)) for i in range(3)],
        queries=[cm_sketch(eps0=eps0, del0=del0)],
        flows=[flow(path=(0, 2), queries=[(0, 1)]),
               flow(path=(1, 2), queries=[(0, 1)])],
        overlay=[[0, 1], 2],
        refine=True
    ),

    # 28
    # Medium tenant (1K)
    # This basically is an example of spectralA which gives infeasible
    dc_topology(hosts_per_tors=48, tors_per_l1s=10,
                l1s=2, num_queries=480, tenant=True,
                overlay='tenant', refine=True),

    # 29
    # Very Large (100K)
    dc_topology(hosts_per_tors=48, tors_per_l1s=50,
                l1s=20, num_queries=24000, tenant=True,
                overlay='tenant', refine=False),

]

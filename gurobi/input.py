import time
import math
import os
import pickle
import random
import hdbscan
import seaborn as sns
import copy

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.cluster.spectral import discretize
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA

from common import Namespace, log, log_time, memoize, freeze_object, constants
from config import common_config
from devices import CPU, P4, Netronome
from flows import flow
from sketches import cm_sketch
from profiles import beluga20, tofino, agiliocx40gbe, dc_line_rate
from util import get_fig_size
import equal_kmedoids
from scipy.sparse import csgraph
from sklearn.manifold import SpectralEmbedding

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

eps0 = constants.eps0
del0 = constants.del0


# * Helpers
def remove_empty(l):
    re = [e for e in l if len(e) > 0]
    # Also flatten single device cluster
    # ret = []
    # for e in re:
    #     if(isinstance(e, list) and len(e) == 1):
    #         ret.extend(e)
    #     else:
    #         ret.append(e)
    # return ret
    return re


def flatten(l):
    ret = []
    for x in l:
        if(isinstance(x, list)):
            ret.extend(flatten(x))
        else:
            ret.append(x)
    return ret


def get_2_level_overlay(overlay):
    output = []
    for l in overlay:
        if(isinstance(l, list)):
            no_nesting = True
            for e in l:
                if(isinstance(e, list)):
                    no_nesting = False
            if(no_nesting):
                output.append(l)
            else:
                output.extend(get_2_level_overlay(l))
        else:
            output.append(l)
    return output


def get_graph(inp):
    g = nx.MultiGraph()
    g.add_nodes_from(range(len(inp.devices)))
    for f in inp.flows:
        p = list(f.path)
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


def draw_graph(G, colors, labels=None, remap=True):
    if(remap):
        colors = remap_colors(colors)

    fig_width = get_fig_size()[0]
    plt.figure(figsize=(fig_width, fig_width))
    pos = nx.spring_layout(G)
    # with open('pickle_objs/cluster-pickle-pos', 'rb') as f:
    #     pos = pickle.load(f)
    # with open('pickle_objs/cluster-pickle-pos', 'wb') as f:
    #     pickle.dump(pos, f)

    if(labels):
        temp = {x: labels[x] for x in G.nodes}
        labels = temp
    nx.draw(G, pos, node_color=colors, labels=labels,
            font_size=8, node_size=150, width=0.5, linewidths=0.8,
            cmap=plt.get_cmap('Spectral'))
    plt.show()
    # plt.savefig('spectral.pdf')

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
    # km = KMeans(init='k-means++', n_clusters=nc)
    # r = km.fit_predict(U)
    r = discretize(U)
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
    num_components = len(list(nx.connected_components(g)))

    for c in nx.connected_components(g):
        num_comp += 1
        cg = g.subgraph(c)
        i2n = list(cg.nodes)
        if(len(i2n) > 1):
            adj = nx.adjacency_matrix(cg)
            devices_per_cluster = common_config.MAX_CLUSTERS_PER_CLUSTER
            if(common_config.solver == 'Netmon'):
                devices_per_cluster = common_config.MAX_DEVICES_PER_CLUSTER
            nc = math.ceil(len(i2n)/devices_per_cluster)
            if(normalized):
                start = time.time()
                log.info("Started SpectralClustering {}/{} size: {}"
                         .format(num_comp, num_components, len(cg)))
                sc = SpectralClustering(nc, affinity='precomputed',
                                        n_init=10, assign_labels='discretize')
                cluster_assignment = sc.fit_predict(adj)
                log.info("Finished SpectralClustering, taking total: {}s"
                         .format(time.time()-start))
                # start = time.time()
                # log.info("Started SpectralClustering {}/{} size: {}"
                #          .format(num_comp, num_components, len(cg)))
                # sc = SpectralClustering(nc, affinity='precomputed',
                #                         n_init=10, assign_labels='discretize')
                # cluster_assignment = sc.fit_predict(adj)
                # log.info("Finished SpectralClustering, taking total: {}s"
                #          .format(time.time()-start))
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


# TODO: remove redundancy in getting overlay for non-connected graphs and such
@log_time(logger=log.info)
def get_hdbscan_overlay(inp, dont_include=lambda x: False):
    log.info("Buliding HDBSCAN overlay with {} devices"
             .format(len(inp.devices)))
    g = get_complete_graph(inp)
    # overlay = []

    # num_comp = 0
    # node_colors = list(g.nodes)
    # last_inc = 0
    # num_components = len(list(nx.connected_components(g)))

    # for c in nx.connected_components(g):
    #     num_comp += 1
    #     cg = g.subgraph(c)
    #     i2n = list(cg.nodes)
    #     if(len(i2n) > 1):
    #         adj = nx.adjacency_matrix(cg)
    #         max_devices_per_cluster = common_config.MAX_CLUSTERS_PER_CLUSTER
    #         if(common_config.solver == 'Netmon'):
    #             max_devices_per_cluster = common_config.MAX_DEVICES_PER_CLUSTER
    #         start = time.time()
    #         log.info("Started HDBSCAN {}/{} size: {}"
    #                  .format(num_comp, num_components, len(cg)))
    #         clusterer = hdbscan.HDBSCAN(
    #             metric='precomputed',
    #             allow_single_cluster=False,
    #             min_cluster_size=int(max_devices_per_cluster/2),
    #             cluster_selection_method='leaf',
    #             min_samples=1,
    #             cluster_selection_epsilon=0.1
    #         )
    #         clusterer.fit(adj)
    #         cluster_assignment = clusterer.labels_
    #         log.info("Finished HDBSCAN, taking total: {}s"
    #                  .format(time.time()-start))
    #         # draw_graph(cg, cluster_assignment)
    #         nc = clusterer.labels_.max() + 1
    #         sub_overlay = [[] for i in range(nc)]
    #         for dnum, cnum in enumerate(cluster_assignment):
    #             node_colors[i2n[dnum]] = last_inc + cnum
    #             sub_overlay[cnum].append(i2n[dnum])
    #         last_inc += nc

    #         filtered_sub_overlay = remove_empty(sub_overlay)
    #         if(len(filtered_sub_overlay) == 1):
    #             overlay.extend(filtered_sub_overlay)
    #         elif(len(filtered_sub_overlay) > 1):
    #             overlay.append(filtered_sub_overlay)
    #     else:
    #         overlay.append(i2n[0])
    #         node_colors[i2n[0]] = last_inc
    #         last_inc += 1

    # # draw_graph(g, node_colors)

    # if(len(overlay) == 1 and isinstance(overlay[0], list)):
    #     return overlay[0]
    # return overlay

    nodes_to_remove = [dont_include(x) for x in g.nodes]
    g.remove_nodes_from(nodes_to_remove)

    adj = nx.adjacency_matrix(g)

    # convert to feature vector matrix
    max_affinity = adj.max()
    num_nodes = len(g.nodes)
    adj = adj.toarray()
    for i in range(num_nodes):
        adj[i, i] = max_affinity

    # convert to distance matrix
    # np.exp(-adj.data, out=adj.data)
    clusterer = hdbscan.HDBSCAN(
        allow_single_cluster=False,
        min_cluster_size=int(common_config.MAX_DEVICES_PER_CLUSTER/2),
        cluster_selection_method='leaf',
        min_samples=1,
        # cluster_selection_epsilon=0.1,
    )
    start = time.time()
    log.info("Starting HDBSCAN")
    clusterer.fit(adj)
    end = time.time()
    log.info("Ended HDBSCAN, Took {} seconds".format(end - start))

    # palette = sns.color_palette()
    # cluster_colors = [sns.desaturate(palette[col], sat)
    #                   if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
    #                   zip(clusterer.labels_, clusterer.probabilities_)]
    # gg = get_graph(inp)
    # node_labels = get_labels_from_overlay(inp, inp.tenant_overlay)
    # draw_graph(gg, cluster_colors, node_labels, remap=False)

    cluster_assignment = clusterer.labels_
    # import ipdb; ipdb.set_trace()
    # TODO: Handle -1's
    nc = clusterer.labels_.max() + 1
    overlay = [[] for i in range(nc)]
    for dnum, cnum in enumerate(cluster_assignment):
        overlay[cnum].append(dnum)

    return overlay

"""
KMedoids does not do a good job of clustering switches into
the server clusters, so we handle it separately
"""
@log_time(logger=log.info)
def get_kmedoids_overlay(inp, dont_include=lambda x: False):
    log.info("Buliding k-medoids cluster overlay with {} devices"
             .format(len(inp.devices)))
    g = get_complete_graph(inp)
    nodes_to_remove = [x for x in g.nodes if dont_include(x)]
    g.remove_nodes_from(nodes_to_remove)

    adj = nx.adjacency_matrix(g)

    # convert to feature vector matrix
    max_affinity = adj.max()
    num_nodes = adj.shape[0]
    adj = adj.toarray()
    for i in range(num_nodes):
        adj[i, i] = max_affinity
    # adj = adj/max_affinity
    # adj = 1 - adj

    # # Visualize affinities
    # pca = PCA(n_components=3)
    # pca.fit(adj)
    # projected = pca.transform(adj)
    # node_labels = get_labels_from_overlay(inp, inp.tenant_overlay)
    # # plt.scatter(projected[:, 0], projected[:, 1],
    # #             c=node_labels,
    # #             cmap=plt.cm.get_cmap('Spectral', 10))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
    #            c=node_labels[:num_nodes],
    #            cmap=plt.cm.get_cmap('Spectral', 10))
    # plt.show()

    devices_per_cluster = common_config.MAX_CLUSTERS_PER_CLUSTER
    if(common_config.solver == 'Netmon'):
        devices_per_cluster = common_config.MAX_DEVICES_PER_CLUSTER
    nc = math.ceil(num_nodes/devices_per_cluster)

    # # import ipdb; ipdb.set_trace()
    # se = SpectralEmbedding(n_components=nc)
    # adj = se.fit_transform(adj)

    start = time.time()
    log.info("Started KMedoids size: {}".format(num_nodes))

    # kmedoids sklearn extra
    # km = KMeans(nc) #, metric='manhattan')
    # km.fit(adj)
    # cluster_assignment = km.labels_

    pca = PCA(n_components=nc)
    pca.fit(adj)
    projected = pca.transform(adj)
    cluster_assignment = discretize(projected)

    log.info("Finished KMedoids, taking total: {}s"
             .format(time.time()-start))

    overlay = [[] for i in range(nc)]

    for dnum, cnum in enumerate(cluster_assignment):
        overlay[cnum].append(dnum)
    overlay = remove_empty(overlay)

    # # Visualize spectral embedding
    # se = SpectralEmbedding(n_components=3, affinity='precomputed')
    # projected = se.fit_transform(adj)
    # node_labels = get_labels_from_overlay(inp, inp.tenant_overlay)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
    #            c=node_labels[:num_nodes],
    #            cmap=plt.cm.get_cmap('Spectral', 10))
    # # ax = fig.add_subplot(111)
    # # ax.scatter(projected[:, 0], projected[:, 1],
    # #            c=cluster_assignment,
    # #            cmap=plt.cm.get_cmap('Spectral', 10))
    # # for i in range(0, num_nodes):
    # #     plt.annotate(node_labels[i], (projected[i, 0], projected[i, 1]))
    # plt.show()

    # num_comp = 0
    # node_colors = list(g.nodes)
    # last_inc = 0
    # num_components = len(list(nx.connected_components(g)))

    # for c in nx.connected_components(g):
    #     num_comp += 1
    #     cg = g.subgraph(c)
    #     i2n = list(cg.nodes)
    #     if(len(i2n) > 1):
    #         adj = nx.adjacency_matrix(cg)

    #         # convert to feature vector matrix
    #         max_affinity = adj.max()
    #         num_nodes = adj.shape[0]
    #         adj = adj.toarray()
    #         for i in range(num_nodes):
    #             adj[i, i] = max_affinity

    #         # convert to distance matrix
    #         # Option 1
    #         # adj = adj/max_affinity
    #         # adj = 1 - adj
    #         # Option 2
    #         # adj = max_affinity - adj
    #         # Option 3
    #         # adj = np.exp(-adj)

    #         devices_per_cluster = common_config.MAX_CLUSTERS_PER_CLUSTER
    #         if(common_config.solver == 'Netmon'):
    #             devices_per_cluster = common_config.MAX_DEVICES_PER_CLUSTER
    #         nc = math.ceil(len(i2n)/devices_per_cluster)
    #         start = time.time()
    #         log.info("Started KMediods {}/{} size: {}"
    #                  .format(num_comp, num_components, len(cg)))

    #         # # equal size github
    #         # dist = pairwise_distances(adj, metric='euclidean')
    #         # km = equal_kmedoids.KMedoids(distance_matrix=dist, n_clusters=nc)
    #         # km.run(max_iterations=30, tolerance=0.001)
    #         # cluster_assignment = list(np.zeros(num_nodes))
    #         # cnum = 0
    #         # for k, v in km.clusters.items():
    #         #     for dnum in v:
    #         #         cluster_assignment[dnum] = cnum
    #         #     cnum += 1

    #         # kmedoids sklearn extra
    #         km = KMedoids(nc)
    #         km.fit(adj)
    #         cluster_assignment = km.labels_

    #         log.info("Finished KMediods, taking total: {}s"
    #                  .format(time.time()-start))

    #         # draw_graph(cg, cluster_assignment)
    #         sub_overlay = [[] for i in range(nc)]

    #         for dnum, cnum in enumerate(cluster_assignment):
    #             if(not dont_include(i2n[dnum])):
    #                 node_colors[i2n[dnum]] = last_inc + cnum
    #                 sub_overlay[cnum].append(i2n[dnum])
    #         last_inc += nc
    #         filtered_sub_overlay = remove_empty(sub_overlay)
    #         if(len(filtered_sub_overlay) == 1):
    #             overlay.extend(filtered_sub_overlay)
    #         elif(len(filtered_sub_overlay) > 1):
    #             import ipdb; ipdb.set_trace()
    #             overlay.append(filtered_sub_overlay)
    #     else:
    #         overlay.append(i2n[0])
    #         node_colors[i2n[0]] = last_inc
    #         last_inc += 1

    # # gg = get_graph(inp)
    # # node_labels = get_labels_from_overlay(inp, inp.tenant_overlay)
    # # draw_graph(gg, node_colors, node_labels)

    if(len(overlay) == 1 and isinstance(overlay[0], list)):
        return overlay[0]
    return overlay


@log_time(logger=log.info)
def get_kmedoids_centers(inp, dont_include=lambda x: False):
    log.info("Buliding k-medoids cluster overlay with {} devices"
             .format(len(inp.devices)))
    g = get_complete_graph(inp)
    nodes_to_remove = [x for x in g.nodes if dont_include(x)]
    g.remove_nodes_from(nodes_to_remove)
    adj = nx.adjacency_matrix(g)

    # convert to feature vector matrix
    max_affinity = adj.max()
    num_nodes = adj.shape[0]
    adj = adj.toarray()
    for i in range(num_nodes):
        adj[i, i] = max_affinity

    # number of clusters
    devices_per_cluster = common_config.MAX_CLUSTERS_PER_CLUSTER
    if(common_config.solver == 'Netmon'):
        devices_per_cluster = common_config.MAX_DEVICES_PER_CLUSTER
    nc = math.ceil(num_nodes/devices_per_cluster)

    # kmedoids sklearn extra
    km = KMedoids(nc)
    km.fit(adj)
    medoid_indices = km.medoid_indices_
    return medoid_indices


# def kmeans_cluster(affinities, n_clusters, rseed=2):
#     assert(len(affinities.shape) == 2)
#     assert(affinities.shape[0] == affinities.shape[1])

#     # 1. Randomly choose clusters
#     rng = np.random.RandomState(rseed)
#     centers = rng.permutation(affinities.shape[0])[:n_clusters]

#     while True:
#         # 2a. Assign labels based on closest center
#         labels = pairwise_distances_argmin(X, centers)
#         # 2b. Find new centers from means of points
#         new_centers = np.array([X[labels == i].mean(0)
#                                 for i in range(n_clusters)])
#         # 2c. Check for convergence
#         if np.all(centers == new_centers):
#             break
#         centers = new_centers
#     return centers, labels


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
    """
    Convert ([[1,2,3], [4,5,6], [7,8,9]], 2) => [[1,2,3,4,5,6], [7,8,9]]
    """
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
    """
    Convert ([[1,2,3], [4,5,6], [7,8,9]], 2) => [[[1,2,3], [4,5,6]], [7,8,9]]
    """
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

    if(len(cur) == 1):
        ret.extend(cur)
    elif(len(cur) > 1):
        ret.append(cur)
    return ret


# * Input
class Input(Namespace):

    @property
    @memoize
    def device_to_id(self):
        _device_to_id = {}
        for (dnum, d) in enumerate(self.devices):
            _device_to_id[d] = dnum
        return _device_to_id


# * TreeTopology
class TreeTopology():

    def __init__(self, hosts_per_tors=2, tors_per_l1s=2, l1s=2,
                 num_queries=80, eps=eps0, overlay='none', tenant=False,
                 refine=False, queries_per_tenant=4, hosts_per_tenant=8,
                 portion_netronome=0):
        self.hosts_per_tors = hosts_per_tors
        self.tors_per_l1s = tors_per_l1s
        self.l1s = l1s
        self.num_queries = num_queries
        self.eps = eps
        self.overlay = overlay
        self.tenant = tenant
        self.refine = refine
        self.queries_per_tenant = queries_per_tenant
        self.hosts_per_tenant = hosts_per_tenant
        self.hosts = hosts_per_tors * tors_per_l1s * l1s
        self.tors = tors_per_l1s * l1s
        self.hosts_tors = self.hosts + self.tors
        self.hosts_tors_l1s = self.hosts_tors + l1s
        self.num_netronome = math.ceil(self.hosts * portion_netronome)

    def get_path_old(self, h1, h2):
        while(h1 == h2):
            h2 = random.randint(0, self.hosts-1)
        tor1 = int(h1 / self.hosts_per_tors)
        tor2 = int(h2 / self.hosts_per_tors)
        l11 = int(tor1 / self.tors_per_l1s)
        l12 = int(tor2 / self.tors_per_l1s)
        tor1 = tor1 + self.hosts
        tor2 = tor2 + self.hosts
        l11 = l11 + self.hosts_tors
        l12 = l12 + self.hosts_tors
        l2 = self.hosts_tors_l1s
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

    def construct_graph(self, devices, has_netro):
        dnames = list(map(lambda x: x.name, devices))
        g = nx.Graph()
        g.add_nodes_from(dnames)

        for i in range(self.l1s):
            l1name = 'l1_P4'+str(i+1)
            g.add_edge('l2_P4', l1name)
            starting_tor = i * self.tors_per_l1s

            for j in range(self.tors_per_l1s):
                tor_idx = starting_tor + j
                tor_name = 'tor_P4'+str(tor_idx + 1)
                g.add_edge(l1name, tor_name)
                starting_host = tor_idx * self.hosts_per_tors

                for k in range(self.hosts_per_tors):
                    host_idx = starting_host + k
                    cpu_name = 'CPU'+str(host_idx + 1)
                    netro_id = has_netro[host_idx]
                    if(netro_id != 0):
                        netro_name = 'Netro'+str(netro_id)
                        g.add_edge(tor_name, netro_name)
                        g.add_edge(netro_name, cpu_name)
                    else:
                        g.add_edge(tor_name, cpu_name)

        return g

    def get_path(self, g, h1, h2):
        h1name = 'CPU'+str(h1)
        h2name = 'CPU'+str(h2)

        try:
            names_path = nx.shortest_path(g, h1name, h2name)
        except:
            print(h1, h2, h1name, h2name)
            pos = nx.spring_layout(g)
            # labels = nx.draw_networkx_labels(g, pos=pos)
            labels = {}
            for key in self.device_dict.keys():
                labels[key] = key
            nx.draw(g, pos=pos, labels=labels)
            plt.show()
            # import ipdb; ipdb.set_trace()
        ids_path = list(map(lambda x: self.device_dict[x].dev_id,
                            names_path))
        return ids_path

    def get_flows(self, inp):
        if(self.tenant):
            flows = []

            flows_per_query = 2
            queries_updated_by_flow = 2
            assert(self.queries_per_tenant > queries_updated_by_flow)
            num_tenants = self.hosts / self.hosts_per_tenant
            assert(
                self.num_queries
                == self.hosts /
                (self.hosts_per_tenant / self.queries_per_tenant))

            servers = np.arange(self.hosts)
            np.random.shuffle(servers)
            tenant_servers = np.split(servers, num_tenants)

            inp.tenant_servers = tenant_servers
            host_overlay = []
            for x in inp.tenant_servers:
                this_servers = x.tolist()
                this_netro = []
                for s in this_servers:
                    netro_id = self.has_netro[s]
                    if(netro_id != 0):
                        netro_name = 'Netro'+str(netro_id)
                        dev_id = self.device_dict[netro_name].dev_id
                        this_netro.append(dev_id)
                host_overlay.append(this_servers + this_netro)

            inp.tenant_servers = host_overlay
            inp.tenant_overlay = (host_overlay
                                  + generate_overlay(
                                      [self.tors + self.l1s + 1],
                                      self.hosts + self.num_netronome))
            flows_per_host = (flows_per_query * self.queries_per_tenant
                              / self.hosts_per_tenant)
            qlist_generator = list(range(self.queries_per_tenant))

            for (tnum, t) in enumerate(tenant_servers):
                query_set = [i + tnum*self.queries_per_tenant
                             for i in range(self.queries_per_tenant)]

                for itr in range(self.queries_per_tenant * flows_per_query):
                    h1 = t[random.randint(0, self.hosts_per_tenant-1)] + 1
                    h2 = t[random.randint(0, self.hosts_per_tenant-1)] + 1
                    while(h2 == h1):
                        h2 = t[random.randint(0, self.hosts_per_tenant-1)] + 1

                    np.random.shuffle(qlist_generator)
                    q_list = qlist_generator[:queries_updated_by_flow]
                    flows.append(
                        flow(
                            path=self.get_path(self.g, h1, h2),
                            queries=[
                                (
                                    query_set[q_idx],
                                    int(random.random() * 4 + 7)/10
                                )
                                for q_idx in q_list
                            ],
                            thr=dc_line_rate/(flows_per_host * 2)
                        )
                    )
        else:
            flows = [
                flow(
                    path=self.get_path(self.g,
                                       random.randint(0, self.hosts-1)+1,
                                       random.randint(0, self.hosts-1)+1
                    ),
                    queries=[
                        (random.randint(0, self.num_queries-1),
                         int(random.random() * 4 + 7)/10)
                    ]
                )
                for flownum in range(max(self.hosts, self.num_queries) * 5)
            ]
        return flows

    def create_inp(self):
        inp = Input(
            devices=(
                [CPU(**beluga20, name='CPU'+str(i+1))
                 for i in range(self.hosts)] +
                [Netronome(**agiliocx40gbe, name='Netro'+str(i+1))
                 for i in range(self.num_netronome)] +
                [P4(**tofino, name='tor_P4'+str(i+1))
                 for i in range(int(self.tors))] +
                [P4(**tofino, name='l1_P4'+str(i+1))
                 for i in range(self.l1s)] +
                [P4(**tofino, name='l2_P4')]
            ),
            queries=(
                [cm_sketch(eps0=self.eps, del0=del0)
                 for i in range(self.num_queries)]
                + []
                # [cm_sketch(eps0=eps0*10, del0=del0) for i in range(24)] +
                # [cm_sketch(eps0=eps0, del0=del0) for i in range(32)]
            ),
        )
        for (dnum, d) in enumerate(inp.devices):
            d.dev_id = dnum
            freeze_object(d)

        self.has_netro = ([i+1 for i in range(self.num_netronome)]
                          + [0 for i in
                             range(self.hosts - self.num_netronome)])
        np.random.shuffle(self.has_netro)

        self.device_dict = {}
        for d in inp.devices:
            self.device_dict[d.name] = d

        self.g = self.construct_graph(inp.devices, self.has_netro)

        inp.flows = self.get_flows(inp)
        return inp

    # ** Old
    def create_inp_old(self):
        inp = Input(
            devices=(
                [CPU(**beluga20, name='CPU'+str(i+1))
                 for i in range(self.hosts)] +
                [P4(**tofino, name='tor_P4'+str(i+1))
                 for i in range(int(self.tors))] +
                # [CPU(**beluga20, name='tor_CPU'+str(i+1))
                #  for i in range(int(tors/2))] +
                [P4(**tofino, name='l1_P4'+str(i+1))
                 for i in range(self.l1s)] +
                [P4(**tofino, name='l2_P4')]
            ),
            queries=(
                [cm_sketch(eps0=self.eps, del0=del0)
                 for i in range(self.num_queries)]
                + []
                # [cm_sketch(eps0=eps0*10, del0=del0) for i in range(24)] +
                # [cm_sketch(eps0=eps0, del0=del0) for i in range(32)]
            ),
        )

        if(self.tenant):
            flows = []

            flows_per_query = 2
            # each tenant exclusively owns 8 hosts
            # and has 4 queries they want measured
            num_tenants = self.hosts / self.hosts_per_tenant
            assert(
                self.num_queries
                == self.hosts /
                (self.hosts_per_tenant / self.queries_per_tenant))

            # the 8 hosts are randomly assigned to tenants
            servers = np.arange(self.hosts)
            np.random.shuffle(servers)
            tenant_servers = np.split(servers, num_tenants)

            inp.tenant_servers = tenant_servers
            host_overlay = [x.tolist() for x in inp.tenant_servers]
            inp.tenant_overlay = (host_overlay
                                  + generate_overlay(
                                      [self.tors + self.l1s + 1], self.hosts))

            for (tnum, t) in enumerate(tenant_servers):
                query_set = [i + tnum*self.queries_per_tenant
                             for i in range(self.queries_per_tenant)]

                # each tenant has 8 different OD pairs,
                # traffic between which needs to be measured
                for itr in range(self.queries_per_tenant * flows_per_query):
                    h1 = t[random.randint(0, self.hosts_per_tenant-1)]
                    h2 = t[random.randint(0, self.hosts_per_tenant-1)]
                    cov = int(random.random() * 4 + 7)/10
                    q = query_set[random.randint(0, self.queries_per_tenant-1)]
                    flows.append(
                        flow(
                            path=self.get_path_old(h1, h2),
                            queries=[(q, cov)]
                        )
                    )
            inp.flows = flows
        else:
            inp.flows = [
                flow(
                    path=self.get_path_old(random.randint(0, self.hosts-1),
                                       random.randint(0, self.hosts-1)),
                    queries=[
                        (random.randint(0, self.num_queries-1),
                         int(random.random() * 4 + 7)/10)
                    ]
                )
                for flownum in range(max(self.hosts, self.num_queries) * 5)
            ]
        return inp

    # TODO: Implement condensing technique to
    # build overlays with larger cluster sizes
    def get_tor_overlay(self):
        if(self.hosts_per_tors <= 8):
            overlay = (
                generate_overlay(
                    [self.tors, self.hosts_per_tors])
                + generate_overlay(
                    [self.tors + self.l1s + 1], self.hosts)
            )
        elif(self.tors <= 20):  # Assuming hosts_per_tors multiple of 8
            overlay = (
                generate_overlay(
                    [self.tors, int(self.hosts_per_tors/8), 8])
                + generate_overlay(
                    [self.tors + self.l1s + 1], self.hosts)
            )
        else:  # Assuming tors multiple of 20
            overlay = (
                generate_overlay(
                    [self.tors, int(self.hosts_per_tors/8), 8])
                + generate_overlay(
                    [int(self.tors/20), 20], self.hosts)
                + generate_overlay(
                    [self.l1s + 1], self.hosts + self.tors)
            )
        return overlay

    def get_spectral_overlay(self, inp):
        # Heuristic: don't cluster nodes with high responsibility:
        if(self.overlay == 'spectralU'):
            host_overlay = get_spectral_overlay(
                inp, comp=set(self.hosts + i
                              for i in range(self.tors + self.l1s + 1)),
                normalized=False)
            overlay = (host_overlay
                       + generate_overlay([self.tors + self.l1s + 1],
                                          self.hosts))
        if(self.overlay == 'spectralA'):
            host_overlay = get_spectral_overlay(
                inp,  # comp=set(hosts + i for i in range(tors + l1s + 1)),
                affinity=True)
            overlay = host_overlay
        else:
            host_overlay = get_spectral_overlay(
                inp, comp=set(self.hosts + i for i in range(
                    self.tors + self.l1s + 1)))
            overlay = (host_overlay
                       + generate_overlay(
                           [self.tors + self.l1s + 1], self.hosts))

        if(len(overlay) > common_config.MAX_CLUSTERS_PER_CLUSTER):
            overlay = fold(overlay,
                           common_config.MAX_CLUSTERS_PER_CLUSTER)

        # inp.overlay = overlay
        # draw_overlay_over_tenant(inp)
        return overlay

    def get_tenant_overlay(self, inp):
        host_overlay = [x.tolist() for x in inp.tenant_servers]
        if(self.hosts > 10000):
            host_overlay = merge(host_overlay, 120)
            # leaves = hosts / 8 / 120
            tors_per_leaf = math.ceil(self.tors / len(host_overlay))

            # TODO:: better way to do assign tors to clusters!!
            assert(self.tors % tors_per_leaf == 0)
            tor_overlay = generate_overlay(
                [int(self.tors/tors_per_leaf), tors_per_leaf],
                self.hosts)

            for i in range(len(tor_overlay)):
                host_overlay[i].extend(tor_overlay[i])

            # tors_overlay = generate_overlay([int(tors/x), x], hosts)
            # host_overlay = fold(host_overlay, 1000)
            overlay = (host_overlay
                       + generate_overlay([self.l1s + 1],
                                          self.hosts + self.tors))
        else:
            overlay = (host_overlay
                       + generate_overlay([self.tors + self.l1s + 1],
                                          self.hosts))
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
        return overlay

    def get_tenant_overlay_switches(self, inp):
        devices_per_cluster = common_config.MAX_CLUSTERS_PER_CLUSTER
        if(common_config.solver == 'Netmon'):
            devices_per_cluster = common_config.MAX_DEVICES_PER_CLUSTER
        host_nic_overlay = copy.deepcopy(inp.tenant_servers)

        # import ipdb; ipdb.set_trace()
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
        switches_start_idx = self.hosts + self.num_netronome
        total_devices = len(inp.devices)
        seq = 0
        total_clusters = len(host_nic_overlay)
        best_dnum_ = {}

        for snum in range(switches_start_idx, total_devices):
            devs = g.neighbors(snum)
            best_dnum, edge_count = -1, 0
            for dnum in devs:
                if(dnum in dev_id_to_cluster_id and dnum < switches_start_idx):
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
            best_dnum_[snum] = best_dnum

        if(total_clusters > common_config.MAX_CLUSTERS_PER_CLUSTER):
            host_nic_overlay = fold(host_nic_overlay,
                                    common_config.MAX_CLUSTERS_PER_CLUSTER)

        overlay = host_nic_overlay
        # inp.overlay = overlay
        # draw_overlay_over_tenant(inp)
        if(len(overlay) == 1 and isinstance(overlay[0], list)):
            return overlay[0]

        return overlay

    def get_kmedoids_overlay(self, inp):
        switches_start_idx = self.hosts + self.num_netronome
        dont_include = lambda x: x >= switches_start_idx
        overlay = get_kmedoids_overlay(inp, dont_include)
        # TODO: Directly get flattened overlay in KMediods instead of flattening yourself
        overlay = get_2_level_overlay(overlay)

        # Assign switches to clusters:
        dev_id_to_cluster_id = dict()
        for cnum, c in enumerate(overlay):
            for dnum in c:
                dev_id_to_cluster_id[dnum] = cnum

        g = get_complete_graph(inp)
        total_devices = len(inp.devices)
        seq = 0
        total_clusters = len(overlay)

        switch_best_dnum_dict = {}
        for snum in range(switches_start_idx, total_devices):
            devs = g.neighbors(snum)
            best_dnum, edge_count = -1, 0
            for dnum in devs:
                if(dnum in dev_id_to_cluster_id and not dont_include(dnum)):
                    edges = g.number_of_edges(snum, dnum)
                    if(edges > edge_count):
                        edge_count = edges
                        best_dnum = dnum

            if(best_dnum == -1):
                cnum = seq % total_clusters
                seq += 1
            else:
                cnum = dev_id_to_cluster_id[best_dnum]
            switch_best_dnum_dict[snum] = best_dnum
            overlay[cnum].append(snum)
            dev_id_to_cluster_id[snum] = cnum

        # TODO: remove all these redundancies on post processing.
        # Move them to a single function
        if(len(overlay) == 1 and isinstance(overlay[0], list)):
            return overlay[0]

        if(len(overlay) > common_config.MAX_CLUSTERS_PER_CLUSTER):
            overlay = fold(overlay, common_config.MAX_CLUSTERS_PER_CLUSTER)

        # inp.overlay = overlay
        # draw_overlay_over_tenant(inp)
        return overlay

    def get_overlay(self, inp):
        assert(self.num_netronome == 0 or
               'spectral' in self.overlay or
               self.overlay in ['hdbscan', 'tenant', 'none'])

        if(self.overlay == 'tor'):
            overlay = self.get_tor_overlay()

        elif(self.overlay == 'shifted'):
            if(self.hosts_per_tors <= 8):
                overlay = (
                    shift_overlay(
                        generate_overlay(
                            [self.tors, self.hosts_per_tors])
                    )
                    + generate_overlay(
                        [self.tors + self.l1s + 1], self.hosts)
                )

        elif('spectral' in self.overlay):
            overlay = self.get_spectral_overlay(inp)
            if(len(overlay) > common_config.MAX_CLUSTERS_PER_CLUSTER):
                overlay = fold(overlay, common_config.MAX_CLUSTER_PER_CLUSTER)

        elif(self.overlay == 'kmedoids'):
            overlay = self.get_kmedoids_overlay(inp)
            if(len(overlay) > common_config.MAX_CLUSTERS_PER_CLUSTER):
                overlay = fold(overlay, common_config.MAX_CLUSTER_PER_CLUSTER)

        elif(self.overlay == 'tenant'):
            overlay = self.get_tenant_overlay_switches(inp)

        elif(self.overlay == 'random'):
            ov = np.array(range(self.hosts))
            np.random.shuffle(ov)
            host_overlay = np.array_split(
                ov, math.ceil(
                    self.hosts/common_config.max_devices_per_cluster))
            ho = [e.tolist() for e in host_overlay]
            overlay = (ho + generate_overlay(
                           [self.tors + self.l1s + 1], self.hosts))

        elif(self.overlay == 'hdbscan'):
            overlay = get_hdbscan_overlay(inp)

        elif(self.overlay == 'none'):
            overlay = None

        # import ipdb; ipdb.set_trace()
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

    def get_input(self):
        pickle_name = "pickle_objs/inp-{}-{}-{}-{}-{}-{}".format(
            self.hosts_per_tors, self.tors_per_l1s, self.l1s,
            self.num_queries, eps0/self.eps, self.num_netronome)
        pickle_loaded = False
        if(os.path.exists(pickle_name)):
            inp_file = open(pickle_name, 'rb')
            inp = pickle.load(inp_file)
            inp_file.close()
            pickle_loaded = True

        log.info("Building tree topo with {} hosts.".format(self.hosts))
        if(not pickle_loaded):
            inp = self.create_inp()

        # Recompute overlay
        inp.refine = self.refine
        inp.overlay = self.get_overlay(inp)

        if(pickle_loaded):
            return inp

        inp_file = open(pickle_name, 'wb')
        pickle.dump(inp, inp_file)
        inp_file.close()
        return inp

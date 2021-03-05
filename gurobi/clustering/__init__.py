import math

from common import log
from config import common_config
from input import (Input, draw_graph, draw_overlay_over_tenant, flatten, fold,
                   generate_overlay, get_2_level_overlay, get_complete_graph,
                   get_graph, get_hdbscan_overlay, get_kmedoids_centers,
                   get_kmedoids_overlay, get_labels_from_overlay,
                   get_spectral_overlay, merge)
from .fast_modularity import FastModularity


class Clustering(object):

    @staticmethod
    def get_tenant_overlay_switches(inp, hosts_per_tenant, switches_start_idx):
        devices_per_cluster = common_config.MAX_CLUSTERS_PER_CLUSTER
        if(common_config.solver == 'Netmon'):
            devices_per_cluster = common_config.MAX_DEVICES_PER_CLUSTER
        host_nic_overlay = inp.tenant_servers

        num_tenant_clusters_to_merge = math.ceil(devices_per_cluster
                                                 / hosts_per_tenant)
        if(num_tenant_clusters_to_merge > 1):
            host_nic_overlay = merge(host_nic_overlay,
                                     num_tenant_clusters_to_merge)

        # Assign switches to clusters:
        dev_id_to_cluster_id = dict()
        for cnum, c in enumerate(host_nic_overlay):
            for dnum in c:
                dev_id_to_cluster_id[dnum] = cnum

        g = get_complete_graph(inp)
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

        overlay = host_nic_overlay
        # inp.overlay = overlay
        # node_labels = {}
        # for d in inp.devices:
        #     node_labels[d.dev_id] = d.name
        # node_colors = get_labels_from_overlay(inp, inp.overlay)
        # g = get_graph(inp)
        # draw_graph(g, node_colors, node_labels)

        return overlay

    @staticmethod
    def get_overlay(inp, topology):
        assert(topology.overlay in topology.supported_overlays())
        overlay = None

        # TODO: consider converting to classes rather
        # than functions
        if('spectral' == topology.overlay):
            overlay = get_spectral_overlay(inp)
        elif('spectralA' == topology.overlay):
            overlay = get_spectral_overlay(inp, normalized=False, affinity=True)
        elif('tenant' == topology.overlay):
            overlay = Clustering.get_tenant_overlay_switches(
                inp, topology.hosts_per_tenant, topology.switches_start_idx)
        elif('kmedoids' == topology.overlay):
            overlay = get_kmedoids_overlay(inp)
        elif('hdbscan' == topology.overlay):
            overlay = get_hdbscan_overlay(inp)
        elif('fast_modularity' == topology.overlay):
            fm = FastModularity()
            overlay = fm.get_overlay(inp, topology)

        # Clean overlay
        # TODO: Move all clustering functions from input.py to here
        # TODO: Move all overlay cleaning from individual functions to here
        if(overlay):
            if(len(overlay) > common_config.MAX_CLUSTERS_PER_CLUSTER):
                overlay = fold(overlay, common_config.MAX_CLUSTERS_PER_CLUSTER)
            if(len(overlay) == 1 and isinstance(overlay[0], list)):
                overlay = overlay[0]
            if(len(overlay) == len(inp.devices)):
                log.info("Not clustering as topology is fairly small")
                log.info("-"*50)
                overlay = None

        # # Visualization
        # inp.overlay = overlay
        # node_labels = {}
        # for d in inp.devices:
        #     node_labels[d.dev_id] = d.name[-2:]
        # node_colors = get_labels_from_overlay(inp, inp.overlay)
        # g = get_graph(inp)
        # draw_graph(g, node_colors, node_labels)

        return overlay

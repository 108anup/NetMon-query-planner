import math
import os
import pickle
import random

import networkx as nx
import numpy as np

from common import log, log_time
from flows import flow
from profiles import dc_line_rate


class Traffic(object):

    @staticmethod
    @log_time(logger=log.info)
    def get_path_with_largest_capacity(g, h1name, h2name):
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

        ids_path = list(map(lambda x: g.nodes[x]['id'],
                            max_capacity_path))
        return (max_capacity_path, ids_path, max_capacity)

    @staticmethod
    def update_path_with_traffic(g, node_path, traffic):
        h1 = node_path[0]
        h2 = node_path[-1]
        g.nodes[h1]['remaining'] -= traffic
        g.nodes[h2]['remaining'] -= traffic
        for start in range(len(node_path)-1):
            beg = node_path[start]
            end = node_path[start+1]
            g.edges[beg, end]['remaining'] -= traffic

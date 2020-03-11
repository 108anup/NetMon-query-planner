import sys
from queue import Queue

from cli import generate_parser
from common import Namespace, setup_logging, log_time
from config import common_config
from input import input_generator
from solvers import solver_to_class
from devices import Cluster


@log_time
def get_partitions(queries):
    partitions = []
    for (i, q) in enumerate(queries):
        q.sketch_id = i
        num_rows = q.rows()
        start_idx = len(partitions)
        if(common_config.horizontal_partition):
            q.partitions = [start_idx + r for r in range(num_rows)]
            partitions += [Namespace(partition_id=start_idx+r,
                                     sketch=q, num_rows=1)
                           for r in range(num_rows)]
        else:
            q.partitions = [start_idx]
            partitions += [Namespace(partition_id=start_idx, sketch=q,
                                     num_rows=num_rows)]
    return partitions


@log_time
def map_flows_partitions(flows, queries):
    for f in flows:
        f.partitions = []
        for q in f.queries:
            coverage_requirement = q[1]
            q_idx = q[0]
            for p_idx in queries[q_idx].partitions:
                f.partitions.append((p_idx, coverage_requirement))


@log_time
def get_cluster_from_overlay(inp, overlay):
    if(isinstance(overlay, list)):
        return Cluster(
            device_tree=[
                get_cluster_from_overlay(inp, roots) for roots in overlay
            ],
            overlay=overlay
        )
    else:
        return inp.devices[overlay]


@log_time
def solve(inp):

    inp.partitions = get_partitions(inp.queries)
    map_flows_partitions(inp.flows, inp.queries)
    Solver = solver_to_class[common_config.solver]

    # Clustering
    if (hasattr(inp, 'overlay')):
        inp.cluster = get_cluster_from_overlay(inp.overlay)
        inp.cluster.build_closures()
        queue = Queue()
        queue.put(Namespace(root=inp.cluster, partitions=inp.partitions))

        while(queue.not_empty()):
            front = queue.get()
            root = front.root  # list of devices (clusters)
            partitions = front.partitions
            # TODO: implement get_flows
            flows = get_flows(inp, root, partitions)
            solver = Solver(devices=root.device_tree, partitions=partitions,
                            flows=flows, queries=inp.queries)
            for d in root.device_tree:
                if(isinstance(d, Cluster)):
                    # TODO: implement d.partitions
                    queue.put(Namespace(root=d, partitions=d.partitions))
    else:
        solver = Solver(devices=inp.devices, flows=inp.flows,
                        partitions=inp.partitions, queries=inp.queries)
        solver.solve()


parser = generate_parser()
args = parser.parse_args(sys.argv[1:])
if(hasattr(args, 'config_file')):
    for fpath in args.config_file:
        common_config.load_config_file(fpath)
common_config.update(args)
setup_logging(common_config)

input_num = common_config.input_num
inp = input_generator[input_num]
solve(inp)

import sys

from cli import generate_parser
from common import Namespace, setup_logging, log_time
from config import common_config
from input import input_generator
from solvers import solver_to_class


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
def solve(devices, queries, flows):

    partitions = get_partitions(queries)
    map_flows_partitions(flows, queries)

    # Clustering
    # for (dnum, d) in enumerate(self.devices):

    Solver = solver_to_class[common_config.solver]
    solver = Solver(devices=devices, flows=flows,
                    partitions=partitions, queries=queries)
    solver.solve()


parser = generate_parser()
args = parser.parse_args(sys.argv[1:])
if('config_file' in args.__dict__):
    for fpath in args.config_file:
        common_config.load_config_file(fpath)
common_config.update(args)
setup_logging(args)

input_num = common_config.input_num
inp = input_generator[input_num]
solve(inp.devices, inp.queries, inp.flows)

import pytest
from main import solve
from input import beluga20, tofino, Input, eps0, del0, generate_overlay
from devices import CPU, P4
from sketches import cm_sketch
from flows import flow


@pytest.mark.parametrize("hosts_per_tor, tors_per_l1s, l1s",
                         [(2, 2, 1), ()])
def test_vary_topology_size(hosts_per_tor, tors_per_l1s, l1s):
    pass


@pytest.mark.parametrize("cluster_size, num_cpus",
                         [(2 + i, 20) for i in range(19)])
def test_vary_cluster_size_cpu_triangle(cluster_size, num_cpus):
    overlay = generate_overlay([int(num_cpus/cluster_size), cluster_size])
    if(num_cpus % cluster_size != 0):
        overlay += [[num_cpus - 1 - i for i in range(num_cpus % cluster_size)]]
    overlay += [num_cpus]



    inp = Input(
        devices=(
            [CPU(**beluga20, name='CPU_{}'.format(i)) for i in range(num_cpus)] +
            [P4(**tofino, name='P4_1')]
        ),
        queries=[
            cm_sketch(eps0=eps0, del0=del0) for i in range(num_cpus)
        ],
        flows=[flow(path=(i, num_cpus, (i + 1) % num_cpus), queries=[(i, 1)])
               for i in range(num_cpus)],
        overlay=overlay
    )
    solve(inp)

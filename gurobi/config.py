from common import param
from devices import cpu, p4
from sketches import cm_sketch
from flows import flow
import random


hosts_per_tors = 4
tors_per_l1s = 2
l1s = 2
hosts = hosts_per_tors * tors_per_l1s * l1s
tors = tors_per_l1s * l1s
hosts_tors = hosts + tors
hosts_tors_l1s = hosts_tors + l1s


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


eps0 = 0.1 * 8 / 128  # 1e-5
del0 = 0.05  # 0.02

"""
TODO:
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

# One time profiling of each device type
beluga20 = {
    'mem_par': [0, 1.1875, 32, 1448.15625,
                5792.625, 32768.0, 440871.90625],
    'mem_ns': [0, 0.539759, 0.510892, 5.04469,
               5.84114, 30.6627, 39.6981],
    'Li_size': [32, 256, 8192, 32768],
    'Li_ns': [0.53, 1.5, 3.7, 36],
    'hash_ns': 3.5, 'cores': 7, 'dpdk_single_core_thr': 35,
    'max_mem': 32768, 'max_rows': 12
}

tofino = {
    'meter_alus': 4, 'sram': 48, 'stages': 12, 'line_thr': 148,
    'max_mpp': 48, 'max_mem': 48*12, 'max_rows': 12 * 4
}

# All memory measured in KB unless otherwise specified
config = [

    # 0
    param(
        # Change when devices are added / removed
        devices=[
            cpu(**beluga20, name='cpu_1'),
            cpu(**beluga20, name='cpu_2'),
            p4(**tofino, name='p4_1'),
            p4(**tofino, name='p4_2'),
        ],
        # Change when metrics are added / removed
        queries=[
            cm_sketch(eps0=eps0*50, del0=del0),
            cm_sketch(eps0=eps0/5, del0=del0),
            cm_sketch(eps0=eps0*100, del0=del0/2)
        ],
        # Change when metric filters are modified
        flows=[
            flow(path=(0, 2, 1), queries=[(1, 1), (2, 1)]),
            flow(path=(0, 2), queries=[(2, 0.9), (0, 1)]),
            flow(path=(0, 3, 1), queries=[(1, 0.8), (2, 0.5)])
        ]
    ),

    # 1
    param(
        devices=[
            cpu(mem_par=[1.1875, 32, 1448.15625,
                         5792.625, 32768.0, 440871.90625],
                mem_ns=[0.539759, 0.510892, 5.04469,
                        5.84114, 30.6627, 39.6981],
                Li_size=[32, 256, 8192, 32768],
                Li_ns=[0.53, 1.5, 3.7, 36],
                hash_ns=3.5, cores=7, dpdk_single_core_thr=35,
                max_mem=32768, max_rows=9, name='cpu_1'),
            p4(meter_alus=4, sram=48, stages=12, line_thr=148,
               max_mpp=48, max_mem=48*12, max_rows=12, name='p4_1')
        ],
        queries=[cm_sketch(eps0=eps0*50, del0=del0),
                 cm_sketch(eps0=eps0/5, del0=del0),
                 cm_sketch(eps0=eps0*100, del0=del0/2)]
    ),

    # 2
    # P4 priority over CPU
    param(
        devices=[
            cpu(mem_par=[1.1875, 32, 1448.15625,
                         5792.625, 32768.0, 440871.90625],
                mem_ns=[0.539759, 0.510892, 5.04469,
                        5.84114, 30.6627, 39.6981],
                Li_size=[32, 256, 8192, 32768],
                Li_ns=[0.53, 1.5, 3.7, 36],
                hash_ns=3.5, cores=7, dpdk_single_core_thr=35,
                max_mem=32768, max_rows=9, name='cpu_1'),
            p4(meter_alus=4, sram=48, stages=12, line_thr=148,
               max_mpp=48, max_mem=48*12, max_rows=12, name='p4_1'),
        ],
        queries=[cm_sketch(eps0=eps0*50, del0=del0),
                 # cm_sketch(eps0=eps0*6, del0=del0),
                 cm_sketch(eps0=eps0*100, del0=del0/2)]
    ),

    # 3
    # Mem vary - CPU - P4
    param(
        devices=[
            cpu(mem_par=[1.1875, 32, 1448.15625,
                         5792.625, 32768.0, 440871.90625],
                mem_ns=[0.50, 0.510892, 5.04469,
                        5.84114, 30.6627, 39.6981],
                Li_size=[32, 256, 8192, 32768],
                Li_ns=[0.53, 1.5, 3.7, 36],
                hash_ns=3.5, cores=7, dpdk_single_core_thr=35,
                max_mem=32768, max_rows=9, name='cpu_1'),
            p4(meter_alus=4, sram=48, stages=12, line_thr=148,
               max_mpp=48, max_mem=48*12, max_rows=12, name='p4_1'),
        ],
        queries=[cm_sketch(eps0=eps0*5, del0=del0)]
    ),

    # 4
    # Skewed CPU allocation 1
    param(
        devices=[
            cpu(mem_par=[0, 1.1875, 32, 1448.15625,
                         5792.625, 32768.0, 440871.90625],
                mem_ns=[0, 0.539759, 0.510892, 5.04469,
                        5.84114, 30.6627, 39.6981],
                Li_size=[32, 256, 8192, 32768],
                Li_ns=[0.53, 1.5, 3.7, 36],
                hash_ns=3.5, cores=7, dpdk_single_core_thr=35,
                max_mem=32768, max_rows=9, name='cpu_1'),
            cpu(mem_par=[0, 1.1875, 32, 1448.15625,
                         5792.625, 32768.0, 440871.90625],
                mem_ns=[0, 0.539759, 0.510892, 5.04469,
                        5.84114, 30.6627, 39.6981],
                Li_size=[32, 256, 8192, 32768],
                Li_ns=[0.53, 1.5, 3.7, 36],
                hash_ns=3.5, cores=7, dpdk_single_core_thr=35,
                max_mem=32768, max_rows=9, name='cpu_2'),
            p4(meter_alus=4, sram=48, stages=12, line_thr=148,
               max_mpp=48, max_mem=48*12, max_rows=12, name='p4_1'),
        ],
        queries=[cm_sketch(eps0=eps0*5, del0=del0),
                 cm_sketch(eps0=eps0/2, del0=del0),
                 cm_sketch(eps0=eps0*100, del0=del0/2)]
    ),

    # 5
    # Skewed CPU allocation 2
    param(
        devices=[
            cpu(mem_par=[0, 1.1875, 32, 1448.15625,
                         5792.625, 32768.0, 440871.90625],
                mem_ns=[0, 0.539759, 0.510892, 5.04469,
                        5.84114, 30.6627, 39.6981],
                Li_size=[32, 256, 8192, 32768],
                Li_ns=[0.53, 1.5, 3.7, 36],
                hash_ns=3.5, cores=7, dpdk_single_core_thr=35,
                max_mem=32768, max_rows=9, name='cpu_1'),
            cpu(mem_par=[0, 1.1875, 32, 1448.15625,
                         5792.625, 32768.0, 440871.90625],
                mem_ns=[0, 0.539759, 0.510892, 5.04469,
                        5.84114, 30.6627, 39.6981],
                Li_size=[32, 256, 8192, 32768],
                Li_ns=[0.53, 1.5, 3.7, 36],
                hash_ns=3.5, cores=7, dpdk_single_core_thr=35,
                max_mem=32768, max_rows=9, name='cpu_2'),
            p4(meter_alus=4, sram=48, stages=12, line_thr=148,
               max_mpp=48, max_mem=48*12, max_rows=12, name='p4_1'),
        ],
        queries=[cm_sketch(eps0=eps0*50, del0=del0),
                 cm_sketch(eps0=eps0/5, del0=del0),
                 cm_sketch(eps0=eps0*100, del0=del0/2)]
    ),

    # 6
    # Use small sketches for fully utilizing CPUs
    param(
        devices=[
            cpu(mem_par=[0, 1.1875, 32, 1448.15625,
                         5792.625, 32768.0, 440871.90625],
                mem_ns=[0, 0.539759, 0.510892, 5.04469,
                        5.84114, 30.6627, 39.6981],
                Li_size=[32, 256, 8192, 32768],
                Li_ns=[0.53, 1.5, 3.7, 36],
                hash_ns=3.5, cores=7, dpdk_single_core_thr=35,
                max_mem=32768, max_rows=9, name='cpu_1'),
            cpu(mem_par=[0, 1.1875, 32, 1448.15625,
                         5792.625, 32768.0, 440871.90625],
                mem_ns=[0, 0.539759, 0.510892, 5.04469,
                        5.84114, 30.6627, 39.6981],
                Li_size=[32, 256, 8192, 32768],
                Li_ns=[0.53, 1.5, 3.7, 36],
                hash_ns=3.5, cores=7, dpdk_single_core_thr=35,
                max_mem=32768, max_rows=9, name='cpu_2'),
            p4(meter_alus=4, sram=48, stages=12, line_thr=148,
               max_mpp=48, max_mem=48*12, max_rows=12, name='p4_1'),
            p4(meter_alus=4, sram=48, stages=12, line_thr=148,
               max_mpp=48, max_mem=48*12, max_rows=12, name='p4_2')
        ],
        queries=[cm_sketch(eps0=eps0*12, del0=del0),
                 cm_sketch(eps0=eps0/5, del0=del0),
                 cm_sketch(eps0=eps0*100, del0=del0/2)]
    ),

    # 7
    # Multi P4
    param(
        devices=[
            p4(meter_alus=4, sram=48, stages=12, line_thr=148,
               max_mpp=48, max_mem=48*12, max_rows=12, name='p4_1'),
            p4(meter_alus=4, sram=48, stages=12, line_thr=148,
               max_mpp=48, max_mem=48*12, max_rows=12, name='p4_2')
        ],
        queries=[cm_sketch(eps0=eps0*12, del0=del0)]
    ),

    # 8
    # Large topo
    param(
        devices=(
            [cpu(**beluga20, name='endhost_cpu'+str(i+1))
             for i in range(hosts)] +
            [p4(**tofino, name='tor_p4'+str(i+1)) for i in range(int(tors))] +
            # [cpu(**beluga20, name='tor_cpu'+str(i+1))
            #  for i in range(int(tors/2))] +
            [p4(**tofino, name='l1_p4'+str(i+1)) for i in range(l1s)] +
            [p4(**tofino, name='l2_p4')]
        ),
        queries=(
            [cm_sketch(eps0=eps0, del0=del0) for i in range(64)] + []
            # [cm_sketch(eps0=eps0*10, del0=del0) for i in range(24)] +
            # [cm_sketch(eps0=eps0, del0=del0) for i in range(32)]
        ),
        flows=[
            flow(
                path=get_path(random.randint(0, hosts-1), random.randint(0, hosts-1)),
                queries=[
                    (random.randint(0, 63), int(random.random() * 4 + 7)/10)
                ]
            )
            for flownum in range(64 * 5)
        ]
    )
]

common_config = param(
    tolerance=0.999,
    ns_tol=0.05,
    res_tol=0.05,
    fileout=False,
    solver='univmon'
)


def update_config(args):
    common_config.solver = args.scheme


'''
Tricks performed:
1. Remove Ceiling
2. Make variables continuous (remove binary and integer variables)
3. Log -INFINITY -> removed
4. Allow non convex problem

NOTES:
1. With logarithmic constraints, if I make variables integral it seems to
perform better, as long as those vars are not involved in other constraints.
2. We want log to be able to take negative values to allow variables
to take value 0 but some problem take a ton of time to solve in those
scenarios.
Above are not relevant any more
'''

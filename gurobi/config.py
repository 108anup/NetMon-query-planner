from common import param
from devices import cpu, p4
from sketches import cm_sketch
from flows import flow


eps0 = 1e-5
del0 = 0.02

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
    'max_mem': 32768, 'max_rows': 9
}

tofino = {
    'meter_alus': 4, 'sram': 48, 'stages': 12, 'line_thr': 148,
    'max_mpp': 48, 'max_mem': 48*12, 'max_rows': 12
}

# All memory measured in KB unless otherwise specified
config = [
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
]

common_config = param(
    tolerance=0.999,
    ns_tol=0.05,
    res_tol=0.05,
    fileout=False
)


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

import os
import random
import pickle

from common import Namespace, memoize
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


class Input(Namespace):

    @property
    @memoize
    def device_to_id(self):
        _device_to_id = {}
        for (dnum, d) in enumerate(self.devices):
            _device_to_id[d] = dnum
        return _device_to_id


def dc_topology(hosts_per_tors=2, tors_per_l1s=2, l1s=2,
                num_queries=80, eps=eps0, overlay='none'):
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

    if(overlay == 'tor'):
        inp.overlay = ([[i + j*hosts_per_tors for i in range(hosts_per_tors)]
                        for j in range(tors)]
                       + [hosts + j for j in range(tors + l1s + 1)])

    if(pickle_loaded):
        return inp

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
        flows=[
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
    )

    if(hasattr(inp, 'overlay')):
        delattr(inp, 'overlay')

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
                l1s=10, num_queries=1024, overlay='tor'),

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
    dc_topology(overlay='tor'),

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

]

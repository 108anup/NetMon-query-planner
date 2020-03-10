import os
import random
import pickle
import yaml

from common import Namespace
from devices import CPU, P4
from flows import flow
from sketches import cm_sketch


eps0 = 0.1 * 8 / 128  # 1e-5
del0 = 0.05  # 0.02


def dc_topology(hosts_per_tors=2, tors_per_l1s=2, l1s=2,
                num_queries=80, eps=eps0):

    pickle_name = "pickle_objs/cfg-{}-{}-{}-{}-{}".format(
        hosts_per_tors, tors_per_l1s, l1s, num_queries, eps0/eps)
    if(os.path.exists(pickle_name)):
        cfg_file = open(pickle_name, 'rb')
        cfg = pickle.load(cfg_file)
        cfg_file.close()
        return cfg

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

    cfg = Namespace(
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

    cfg_file = open(pickle_name, 'wb')
    pickle.dump(cfg, cfg_file)
    cfg_file.close()
    return cfg

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
    'max_mpr': 48, 'max_mem': 48*12, 'max_rows': 12 * 4
}

# All memory measured in KB unless otherwise specified
config = [

    # 0
    # Bad for vanilla Univmon (puts much load on CPU)
    Namespace(
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
    Namespace(
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
    Namespace(
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
    Namespace(
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
    Namespace(
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
    Namespace(
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
    Namespace(
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
    Namespace(
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
    Namespace(
        devices=[
            CPU(mem_par=[1.1875, 32, 1448.15625,
                         5792.625, 32768.0, 440871.90625],
                mem_ns=[0.50, 0.510892, 5.04469,
                        5.84114, 30.6627, 39.6981],
                Li_size=[32, 256, 8192, 32768],
                Li_ns=[0.53, 1.5, 3.7, 36],
                hash_ns=3.5, cores=7, dpdk_single_core_thr=35,
                max_mem=32768, max_rows=9, name='CPU_1'),
            P4(meter_alus=4, sram=48, stages=12, line_thr=148,
               max_mpr=48, max_mem=48*12, max_rows=12, name='P4_1'),
        ],
        queries=[cm_sketch(eps0=eps0*5, del0=del0)]
    ),

    # 11
    # Pressure at network core
    Namespace(
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
    Namespace(
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
    Namespace(
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
    dc_topology(hosts_per_tors=48, tors_per_l1s=20, l1s=10, num_queries=2048)
]


class Config(Namespace):

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def load_config_file(self, fpath='config.yml'):
        if(os.path.exists(fpath)):
            f = open(fpath)
            config_data = yaml.safe_load(f)
            f.close()
            self.update(config_data)

    def update(self, config):
        if(isinstance(config, dict)):
            self.__dict__.update(config)
        else:
            self.__dict__.update(config.__dict__)


'''
Config Priorities:
cli input overrides
cli provided config file overrides
default config file overrides
default config values
'''

default_config = Config(
    tolerance=0.999,
    ns_tol=0,
    res_tol=0,
    use_model=False,
    ftol=6.0e-5,
    mipgapabs=10,
    # mipgap=0.01,

    solver='Netmon',
    cfg_num=0,
    verbose=0,
    mipout=False,
    horizontal_partition=False,
    vertical_partition=False,
    output_file=None,
    config_file=[]
)
common_config = Config()
common_config.update(default_config)
common_config.load_config_file()

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

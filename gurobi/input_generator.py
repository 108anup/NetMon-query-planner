from clos import Clos
from common import constants
from devices import CPU, P4, Netronome
from flows import flow
from input import Input, TreeTopology
from profiles import agiliocx40gbe, beluga20, tofino
from sketches import cm_sketch

# Stub file for providing input to solver

eps0 = constants.eps0
del0 = constants.del0

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
            flow(path=(0, 1), queries=[(0, 1)], thr=70),
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
    TreeTopology(),

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
    # Nothing matters as inuous resource allocation
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
    TreeTopology(num_queries=64, eps=eps0/10),

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
    TreeTopology(hosts_per_tors=48, num_queries=256),

    # 15
    # Very Large
    TreeTopology(hosts_per_tors=48, tors_per_l1s=20,
                 l1s=10, num_queries=5, overlay='tor'),

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
    TreeTopology(overlay='shifted'),

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

    # 23 modified 18 overlay on 11
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
        overlay=[[i + j*5 for i in range(5)] for j in range(4)] + [20]
    ),

    # 24
    # Small tenant (100)
    TreeTopology(hosts_per_tors=8, num_queries=4*40, tenant=True,
                 eps=eps0, overlay='tenant', refine=True,
                 queries_per_tenant=40, portion_netronome=0),

    # 25
    # Large tenant (10K)
    TreeTopology(hosts_per_tors=48, tors_per_l1s=20,
                 l1s=10, num_queries=48*20*10*2, tenant=True, overlay='tenant',
                 eps=eps0/100, portion_netronome=0.5,
                 queries_per_tenant=16),

    # 26 Clustering Intuition init cant help
    Input(
        devices=[CPU(**beluga20, name='CPU_{}'.format(i)) for i in range(5)],
        queries=[cm_sketch(eps0=eps0, del0=del0)],
        flows=[flow(path=(0, 3), queries=[(0, 1)]),
               flow(path=(1, 3), queries=[(0, 1)]),
               flow(path=(2, 4), queries=[(0, 1)])],
        overlay=[[0, 1, 2], 3, 4]
    ),

    # 27 Clustering Intuition init could help
    Input(
        devices=[CPU(**beluga20, name='CPU_{}'.format(i)) for i in range(3)],
        queries=[cm_sketch(eps0=eps0, del0=del0)],
        flows=[flow(path=(0, 2), queries=[(0, 1)]),
               flow(path=(1, 2), queries=[(0, 1)])],
        overlay=[[0, 1], 2],
        refine=True
    ),

    # 28
    # Medium tenant (1K)
    TreeTopology(hosts_per_tors=48, tors_per_l1s=10,
                 l1s=4, num_queries=48*10*4*3, tenant=True,
                 overlay='tenant', refine=False, eps=eps0/10,
                 queries_per_tenant=8*3),

    # 29
    # Very Large (100K)
    TreeTopology(hosts_per_tors=48, tors_per_l1s=50,
                 l1s=40, num_queries=48 * 50 * 40, tenant=True,
                 overlay='tenant', refine=False,
                 queries_per_tenant=8),

    # 30
    # Small tenant with small requirements
    TreeTopology(hosts_per_tors=8, tors_per_l1s=2, l1s=2, num_queries=32 * 2,
                 queries_per_tenant=8 * 2, eps=eps0/100, overlay='tenant',
                 tenant=True, refine=False),
    # 31
    Input(
        devices=[
            CPU(**beluga20, name='CPU_1'),
            Netronome(**agiliocx40gbe, name='Netronome_1'),
            P4(**tofino, name='P4_1'),
        ],
        queries=[
            cm_sketch(eps0=eps0/10000, del0=del0/1.5),
        ],
        flows=[
            flow(path=(0, 1, 2), queries=[(0, 1)]),
        ]
    ),

    # 32
    Input(
        devices=[
            Netronome(**agiliocx40gbe, name='Netronome_1'),
        ],
        queries=[
            cm_sketch(eps0=eps0/1000, del0=del0/1.5),
        ],
        flows=[
            flow(path=(0,), queries=[(0, 1)], thr=20),
        ]
    ),

    # 33
    Input(
        devices=(
            [CPU(**beluga20, name='CPU_{}'.format(i)) for i in range(2)] +
            [P4(**tofino, name='P4_{}'.format(i)) for i in range(2)]
        ),
        queries=[
            cm_sketch(eps0=eps0/100, del0=del0/1.5) for i in range(8)
        ],
        flows=[
            flow(path=(0, 2, 1),
                 queries=[(i, 1) for i in range(8)], thr=20),
            flow(path=(0, 3, 1),
                 queries=[(i, 1) for i in range(7)], thr=40)
        ]
    ),

    # 34
    Clos(pods=4, overlay='tenant', query_density=6),
]

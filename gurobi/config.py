from common import param
from devices import cpu, p4
from sketches import cm_sketch


eps0 = 1e-5
del0 = 0.02

# All memory measured in KB unless otherwise specified
config = [
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
        queries=[cm_sketch(eps0=eps0*50, del0=del0),
                 cm_sketch(eps0=eps0/5, del0=del0),
                 cm_sketch(eps0=eps0*100, del0=del0/2)]
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
        queries=[cm_sketch(eps0=eps0*50, del0=del0)]
    )
]

common_config = param(
    tolerance=0.999,
    ns_tol=0.05,
    res_tol=0.05
)


'''
Tricks performed:
1. Remove Ceiling
2. Make variables continuous (remove binary and integer variables)
3. Log -INFINITY

NOTES:
1. With logarithmic constraints, if I make variables integral it seems to
perform better, as long as those vars are not involved in other constraints.
2. We want log to be able to take negative values to allow variables
to take value 0 but some problem take a ton of time to solve in those
scenarios.
'''

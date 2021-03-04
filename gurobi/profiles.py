dc_line_rate = 22
p4_stages = 2

# * One time profiling of each device type
beluga20 = {
    'profile_name': "beluga20",
    'mem_par': [0, 1.1875, 32, 1448.15625,
                5792.625, 32768.0, 440871.90625],
    'mem_ns': [0, 0.539759, 0.510892, 5.04469,
               5.84114, 30.6627, 39.6981],
    'Li_size': [32, 256, 8192, 32768],
    'Li_ns': [0.53, 1.5, 3.7, 36],
    'hash_ns': 3.5, 'cores': 6, 'dpdk_single_core_thr': 5.746337927,
    'max_mem': 32768, 'max_hashes': 48, 'line_thr': 98,
    'cache_size': 8192
}

tofino = {
    'profile_name': "tofino",
    'meter_alus': 4, 'sram': 512, 'stages': p4_stages, 'line_thr': 5208.33,
    'max_mpr': 512, 'max_mem': 512 * p4_stages, 'max_hashes': p4_stages * 4,
    'max_col_bits': 17
}

agiliocx40gbe = {
    'profile_name': "agiliocx40gbe",
    'mem_const': 13.400362847440405,
    'mem_par': [0, 0.0390625, 0.078125, 0.625, 1.25, 1280, 5120,
                10240, 20480, 40960, 163840, 200000],
    'mem_ns': [2.4378064635338013, 2.4378064635338013, 3.001397701320998,
               3.001397701320998, 3.419718427960497, 3.419718427960497,
               5.7979058750830506, 8.557025366725194, 10.33170622821247,
               11.792229798079585, 13.76708909669199, 14.348232588210744],
    'hashing_const': 33.22322438313901, 'hashing_slope': 1.0142711549803503,
    'emem_size': 3*1024, 'total_me': 54, 'max_mem': 200000,
    'max_hashes': 12, 'line_thr': 27.55,
    'cache_size': 3*1024
}

# Hashes almost always won't become bottleneck
# On the other hand, very low memory
bram_size = 48
alveo_u280 = {
    'profile_name': "alveo_u280",
    'mem_const': 0,
    'mem_par': [0, 48.0, 80.0, 680.0, 10280.0, 49200.0],
    'mem_ns': [0, 0, 457.89396815068494, 565.0799996575344,
               595.8979178082192, 595.8979178082192],
    'hashing_const': 0, 'hashing_slope': 3.424654843444227,
    'bram_size': bram_size, 'total_hash_units': 400, 'max_mem':bram_size,
    'max_hashes': 1200, 'line_thr': 91.438134,
    'cache_size': 48
}

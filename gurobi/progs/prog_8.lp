\ Model netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  cpu_rows_load: Priority=20 Weight=1 AbsTol=0 RelTol=0
   0 m_access_time_cpu1 + 0 ns_sketch_cpu1 + 0 ns_dpdk_cpu1 + 0 ns_cpu1
   + 0 m_access_time_cpu2 + 0 ns_sketch_cpu2 + 0 ns_dpdk_cpu2 + 0 ns_cpu2
   + 0 m_access_time_cpu3 + 0 ns_sketch_cpu3 + 0 ns_dpdk_cpu3 + 0 ns_cpu3
   + 0 m_access_time_cpu4 + 0 ns_sketch_cpu4 + 0 ns_dpdk_cpu4 + 0 ns_cpu4
   + 0 m_access_time_cpu5 + 0 ns_sketch_cpu5 + 0 ns_dpdk_cpu5 + 0 ns_cpu5
   + 0 m_access_time_cpu6 + 0 ns_sketch_cpu6 + 0 ns_dpdk_cpu6 + 0 ns_cpu6
   + 0 m_access_time_cpu7 + 0 ns_sketch_cpu7 + 0 ns_dpdk_cpu7 + 0 ns_cpu7
   + 0 m_access_time_cpu8 + 0 ns_sketch_cpu8 + 0 ns_dpdk_cpu8 + 0 ns_cpu8
   + 0 m_access_time_cpu9 + 0 ns_sketch_cpu9 + 0 ns_dpdk_cpu9 + 0 ns_cpu9
   + 0 m_access_time_cpu10 + 0 ns_sketch_cpu10 + 0 ns_dpdk_cpu10
   + 0 ns_cpu10 + 0 m_access_time_cpu11 + 0 ns_sketch_cpu11
   + 0 ns_dpdk_cpu11 + 0 ns_cpu11 + 0 m_access_time_cpu12
   + 0 ns_sketch_cpu12 + 0 ns_dpdk_cpu12 + 0 ns_cpu12
   + 0 m_access_time_cpu13 + 0 ns_sketch_cpu13 + 0 ns_dpdk_cpu13
   + 0 ns_cpu13 + 0 m_access_time_cpu14 + 0 ns_sketch_cpu14
   + 0 ns_dpdk_cpu14 + 0 ns_cpu14 + 0 m_access_time_cpu15
   + 0 ns_sketch_cpu15 + 0 ns_dpdk_cpu15 + 0 ns_cpu15
   + 0 m_access_time_cpu16 + 0 ns_sketch_cpu16 + 0 ns_dpdk_cpu16
   + 0 ns_cpu16 + 0 ns + max_rows_cpu
  p4_rows_load: Priority=15 Weight=1 AbsTol=0 RelTol=0
   max_rows_p4
  cpu_load_mem: Priority=10 Weight=1 AbsTol=0 RelTol=0
   max_mem_cpu
  p4_load_mem: Priority=5 Weight=1 AbsTol=0 RelTol=0
   max_mem_p4
Subject To
 cov_0_84: frac[10,84] + frac[14,84] + frac[18,84] + frac[19,84]
   + frac[21,84] >= 0.7
 cov_0_85: frac[10,85] + frac[14,85] + frac[18,85] + frac[19,85]
   + frac[21,85] >= 0.7
 cov_0_86: frac[10,86] + frac[14,86] + frac[18,86] + frac[19,86]
   + frac[21,86] >= 0.7
 cov_1_0: frac[4,0] + frac[15,0] + frac[17,0] + frac[19,0] + frac[20,0]
   + frac[21,0] + frac[22,0] >= 0.7
 cov_1_1: frac[4,1] + frac[15,1] + frac[17,1] + frac[19,1] + frac[20,1]
   + frac[21,1] + frac[22,1] >= 0.7
 cov_1_2: frac[4,2] + frac[15,2] + frac[17,2] + frac[19,2] + frac[20,2]
   + frac[21,2] + frac[22,2] >= 0.7
 cov_2_63: frac[4,63] + frac[12,63] + frac[17,63] + frac[19,63]
   + frac[20,63] + frac[21,63] + frac[22,63] >= 0.7
 cov_2_64: frac[4,64] + frac[12,64] + frac[17,64] + frac[19,64]
   + frac[20,64] + frac[21,64] + frac[22,64] >= 0.7
 cov_2_65: frac[4,65] + frac[12,65] + frac[17,65] + frac[19,65]
   + frac[20,65] + frac[21,65] + frac[22,65] >= 0.7
 cov_3_186: frac[4,186] + frac[13,186] + frac[17,186] + frac[19,186]
   + frac[20,186] + frac[21,186] + frac[22,186] >= 0.9
 cov_3_187: frac[4,187] + frac[13,187] + frac[17,187] + frac[19,187]
   + frac[20,187] + frac[21,187] + frac[22,187] >= 0.9
 cov_3_188: frac[4,188] + frac[13,188] + frac[17,188] + frac[19,188]
   + frac[20,188] + frac[21,188] + frac[22,188] >= 0.9
 cov_4_27: frac[4,27] + frac[13,27] + frac[17,27] + frac[19,27]
   + frac[20,27] + frac[21,27] + frac[22,27] >= 0.7
 cov_4_28: frac[4,28] + frac[13,28] + frac[17,28] + frac[19,28]
   + frac[20,28] + frac[21,28] + frac[22,28] >= 0.7
 cov_4_29: frac[4,29] + frac[13,29] + frac[17,29] + frac[19,29]
   + frac[20,29] + frac[21,29] + frac[22,29] >= 0.7
 cov_5_48: frac[9,48] + frac[11,48] + frac[18,48] >= 0.8
 cov_5_49: frac[9,49] + frac[11,49] + frac[18,49] >= 0.8
 cov_5_50: frac[9,50] + frac[11,50] + frac[18,50] >= 0.8
 cov_6_180: frac[0,180] + frac[13,180] + frac[16,180] + frac[19,180]
   + frac[20,180] + frac[21,180] + frac[22,180] >= 1
 cov_6_181: frac[0,181] + frac[13,181] + frac[16,181] + frac[19,181]
   + frac[20,181] + frac[21,181] + frac[22,181] >= 1
 cov_6_182: frac[0,182] + frac[13,182] + frac[16,182] + frac[19,182]
   + frac[20,182] + frac[21,182] + frac[22,182] >= 1
 cov_7_18: frac[0,18] + frac[15,18] + frac[16,18] + frac[19,18]
   + frac[20,18] + frac[21,18] + frac[22,18] >= 0.8
 cov_7_19: frac[0,19] + frac[15,19] + frac[16,19] + frac[19,19]
   + frac[20,19] + frac[21,19] + frac[22,19] >= 0.8
 cov_7_20: frac[0,20] + frac[15,20] + frac[16,20] + frac[19,20]
   + frac[20,20] + frac[21,20] + frac[22,20] >= 0.8
 cov_8_15: frac[1,15] + frac[5,15] + frac[16,15] + frac[17,15]
   + frac[20,15] >= 1
 cov_8_16: frac[1,16] + frac[5,16] + frac[16,16] + frac[17,16]
   + frac[20,16] >= 1
 cov_8_17: frac[1,17] + frac[5,17] + frac[16,17] + frac[17,17]
   + frac[20,17] >= 1
 cov_9_6: frac[2,6] + frac[7,6] + frac[16,6] + frac[17,6] + frac[20,6]
   >= 0.9
 cov_9_7: frac[2,7] + frac[7,7] + frac[16,7] + frac[17,7] + frac[20,7]
   >= 0.9
 cov_9_8: frac[2,8] + frac[7,8] + frac[16,8] + frac[17,8] + frac[20,8]
   >= 0.9
 cov_10_36: frac[1,36] + frac[14,36] + frac[16,36] + frac[19,36]
   + frac[20,36] + frac[21,36] + frac[22,36] >= 1
 cov_10_37: frac[1,37] + frac[14,37] + frac[16,37] + frac[19,37]
   + frac[20,37] + frac[21,37] + frac[22,37] >= 1
 cov_10_38: frac[1,38] + frac[14,38] + frac[16,38] + frac[19,38]
   + frac[20,38] + frac[21,38] + frac[22,38] >= 1
 cov_11_168: frac[0,168] + frac[12,168] + frac[16,168] + frac[19,168]
   + frac[20,168] + frac[21,168] + frac[22,168] >= 1
 cov_11_169: frac[0,169] + frac[12,169] + frac[16,169] + frac[19,169]
   + frac[20,169] + frac[21,169] + frac[22,169] >= 1
 cov_11_170: frac[0,170] + frac[12,170] + frac[16,170] + frac[19,170]
   + frac[20,170] + frac[21,170] + frac[22,170] >= 1
 cov_12_21: frac[0,21] + frac[12,21] + frac[16,21] + frac[19,21]
   + frac[20,21] + frac[21,21] + frac[22,21] >= 1
 cov_12_22: frac[0,22] + frac[12,22] + frac[16,22] + frac[19,22]
   + frac[20,22] + frac[21,22] + frac[22,22] >= 1
 cov_12_23: frac[0,23] + frac[12,23] + frac[16,23] + frac[19,23]
   + frac[20,23] + frac[21,23] + frac[22,23] >= 1
 cov_13_105: frac[2,105] + frac[12,105] + frac[16,105] + frac[19,105]
   + frac[20,105] + frac[21,105] + frac[22,105] >= 0.9
 cov_13_106: frac[2,106] + frac[12,106] + frac[16,106] + frac[19,106]
   + frac[20,106] + frac[21,106] + frac[22,106] >= 0.9
 cov_13_107: frac[2,107] + frac[12,107] + frac[16,107] + frac[19,107]
   + frac[20,107] + frac[21,107] + frac[22,107] >= 0.9
 cov_14_27: frac[9,27] + frac[12,27] + frac[18,27] + frac[19,27]
   + frac[21,27] >= 0.9
 cov_14_28: frac[9,28] + frac[12,28] + frac[18,28] + frac[19,28]
   + frac[21,28] >= 0.9
 cov_14_29: frac[9,29] + frac[12,29] + frac[18,29] + frac[19,29]
   + frac[21,29] >= 0.9
 cov_15_36: frac[3,36] + frac[15,36] + frac[16,36] + frac[19,36]
   + frac[20,36] + frac[21,36] + frac[22,36] >= 0.8
 cov_15_37: frac[3,37] + frac[15,37] + frac[16,37] + frac[19,37]
   + frac[20,37] + frac[21,37] + frac[22,37] >= 0.8
 cov_15_38: frac[3,38] + frac[15,38] + frac[16,38] + frac[19,38]
   + frac[20,38] + frac[21,38] + frac[22,38] >= 0.8
 cov_16_42: frac[6,42] + frac[14,42] + frac[17,42] + frac[19,42]
   + frac[20,42] + frac[21,42] + frac[22,42] >= 0.7
 cov_16_43: frac[6,43] + frac[14,43] + frac[17,43] + frac[19,43]
   + frac[20,43] + frac[21,43] + frac[22,43] >= 0.7
 cov_16_44: frac[6,44] + frac[14,44] + frac[17,44] + frac[19,44]
   + frac[20,44] + frac[21,44] + frac[22,44] >= 0.7
 cov_17_69: frac[5,69] + frac[7,69] + frac[17,69] >= 0.7
 cov_17_70: frac[5,70] + frac[7,70] + frac[17,70] >= 0.7
 cov_17_71: frac[5,71] + frac[7,71] + frac[17,71] >= 0.7
 cov_18_171: frac[4,171] + frac[12,171] + frac[17,171] + frac[19,171]
   + frac[20,171] + frac[21,171] + frac[22,171] >= 0.9
 cov_18_172: frac[4,172] + frac[12,172] + frac[17,172] + frac[19,172]
   + frac[20,172] + frac[21,172] + frac[22,172] >= 0.9
 cov_18_173: frac[4,173] + frac[12,173] + frac[17,173] + frac[19,173]
   + frac[20,173] + frac[21,173] + frac[22,173] >= 0.9
 cov_19_171: frac[5,171] + frac[14,171] + frac[17,171] + frac[19,171]
   + frac[20,171] + frac[21,171] + frac[22,171] >= 0.9
 cov_19_172: frac[5,172] + frac[14,172] + frac[17,172] + frac[19,172]
   + frac[20,172] + frac[21,172] + frac[22,172] >= 0.9
 cov_19_173: frac[5,173] + frac[14,173] + frac[17,173] + frac[19,173]
   + frac[20,173] + frac[21,173] + frac[22,173] >= 0.9
 cov_20_189: frac[3,189] + frac[15,189] + frac[16,189] + frac[19,189]
   + frac[20,189] + frac[21,189] + frac[22,189] >= 0.8
 cov_20_190: frac[3,190] + frac[15,190] + frac[16,190] + frac[19,190]
   + frac[20,190] + frac[21,190] + frac[22,190] >= 0.8
 cov_20_191: frac[3,191] + frac[15,191] + frac[16,191] + frac[19,191]
   + frac[20,191] + frac[21,191] + frac[22,191] >= 0.8
 cov_21_168: frac[2,168] + frac[15,168] + frac[16,168] + frac[19,168]
   + frac[20,168] + frac[21,168] + frac[22,168] >= 0.8
 cov_21_169: frac[2,169] + frac[15,169] + frac[16,169] + frac[19,169]
   + frac[20,169] + frac[21,169] + frac[22,169] >= 0.8
 cov_21_170: frac[2,170] + frac[15,170] + frac[16,170] + frac[19,170]
   + frac[20,170] + frac[21,170] + frac[22,170] >= 0.8
 cov_22_57: frac[1,57] + frac[14,57] + frac[16,57] + frac[19,57]
   + frac[20,57] + frac[21,57] + frac[22,57] >= 0.8
 cov_22_58: frac[1,58] + frac[14,58] + frac[16,58] + frac[19,58]
   + frac[20,58] + frac[21,58] + frac[22,58] >= 0.8
 cov_22_59: frac[1,59] + frac[14,59] + frac[16,59] + frac[19,59]
   + frac[20,59] + frac[21,59] + frac[22,59] >= 0.8
 cov_23_60: frac[4,60] + frac[7,60] + frac[17,60] >= 0.7
 cov_23_61: frac[4,61] + frac[7,61] + frac[17,61] >= 0.7
 cov_23_62: frac[4,62] + frac[7,62] + frac[17,62] >= 0.7
 cov_24_165: frac[1,165] + frac[14,165] + frac[16,165] + frac[19,165]
   + frac[20,165] + frac[21,165] + frac[22,165] >= 0.9
 cov_24_166: frac[1,166] + frac[14,166] + frac[16,166] + frac[19,166]
   + frac[20,166] + frac[21,166] + frac[22,166] >= 0.9
 cov_24_167: frac[1,167] + frac[14,167] + frac[16,167] + frac[19,167]
   + frac[20,167] + frac[21,167] + frac[22,167] >= 0.9
 cov_25_135: frac[8,135] + frac[15,135] + frac[18,135] + frac[19,135]
   + frac[21,135] >= 0.8
 cov_25_136: frac[8,136] + frac[15,136] + frac[18,136] + frac[19,136]
   + frac[21,136] >= 0.8
 cov_25_137: frac[8,137] + frac[15,137] + frac[18,137] + frac[19,137]
   + frac[21,137] >= 0.8
 cov_26_84: frac[1,84] + frac[2,84] + frac[16,84] >= 1
 cov_26_85: frac[1,85] + frac[2,85] + frac[16,85] >= 1
 cov_26_86: frac[1,86] + frac[2,86] + frac[16,86] >= 1
 cov_27_150: frac[4,150] + frac[15,150] + frac[17,150] + frac[19,150]
   + frac[20,150] + frac[21,150] + frac[22,150] >= 0.9
 cov_27_151: frac[4,151] + frac[15,151] + frac[17,151] + frac[19,151]
   + frac[20,151] + frac[21,151] + frac[22,151] >= 0.9
 cov_27_152: frac[4,152] + frac[15,152] + frac[17,152] + frac[19,152]
   + frac[20,152] + frac[21,152] + frac[22,152] >= 0.9
 cov_28_132: frac[6,132] + frac[11,132] + frac[17,132] + frac[18,132]
   + frac[20,132] + frac[21,132] + frac[22,132] >= 1
 cov_28_133: frac[6,133] + frac[11,133] + frac[17,133] + frac[18,133]
   + frac[20,133] + frac[21,133] + frac[22,133] >= 1
 cov_28_134: frac[6,134] + frac[11,134] + frac[17,134] + frac[18,134]
   + frac[20,134] + frac[21,134] + frac[22,134] >= 1
 cov_29_120: frac[6,120] + frac[10,120] + frac[17,120] + frac[18,120]
   + frac[20,120] + frac[21,120] + frac[22,120] >= 0.7
 cov_29_121: frac[6,121] + frac[10,121] + frac[17,121] + frac[18,121]
   + frac[20,121] + frac[21,121] + frac[22,121] >= 0.7
 cov_29_122: frac[6,122] + frac[10,122] + frac[17,122] + frac[18,122]
   + frac[20,122] + frac[21,122] + frac[22,122] >= 0.7
 cov_30_24: frac[2,24] + frac[15,24] + frac[16,24] + frac[19,24]
   + frac[20,24] + frac[21,24] + frac[22,24] >= 0.9
 cov_30_25: frac[2,25] + frac[15,25] + frac[16,25] + frac[19,25]
   + frac[20,25] + frac[21,25] + frac[22,25] >= 0.9
 cov_30_26: frac[2,26] + frac[15,26] + frac[16,26] + frac[19,26]
   + frac[20,26] + frac[21,26] + frac[22,26] >= 0.9
 cov_31_117: frac[4,117] + frac[5,117] + frac[17,117] >= 0.7
 cov_31_118: frac[4,118] + frac[5,118] + frac[17,118] >= 0.7
 cov_31_119: frac[4,119] + frac[5,119] + frac[17,119] >= 0.7
 cov_32_126: frac[0,126] + frac[6,126] + frac[16,126] + frac[17,126]
   + frac[20,126] >= 0.9
 cov_32_127: frac[0,127] + frac[6,127] + frac[16,127] + frac[17,127]
   + frac[20,127] >= 0.9
 cov_32_128: frac[0,128] + frac[6,128] + frac[16,128] + frac[17,128]
   + frac[20,128] >= 0.9
 cov_33_177: frac[2,177] + frac[3,177] + frac[16,177] >= 1
 cov_33_178: frac[2,178] + frac[3,178] + frac[16,178] >= 1
 cov_33_179: frac[2,179] + frac[3,179] + frac[16,179] >= 1
 cov_34_150: frac[8,150] + frac[12,150] + frac[18,150] + frac[19,150]
   + frac[21,150] >= 0.9
 cov_34_151: frac[8,151] + frac[12,151] + frac[18,151] + frac[19,151]
   + frac[21,151] >= 0.9
 cov_34_152: frac[8,152] + frac[12,152] + frac[18,152] + frac[19,152]
   + frac[21,152] >= 0.9
 cov_35_42: frac[12,42] + frac[15,42] + frac[19,42] >= 0.8
 cov_35_43: frac[12,43] + frac[15,43] + frac[19,43] >= 0.8
 cov_35_44: frac[12,44] + frac[15,44] + frac[19,44] >= 0.8
 cov_36_147: frac[3,147] + frac[12,147] + frac[16,147] + frac[19,147]
   + frac[20,147] + frac[21,147] + frac[22,147] >= 0.9
 cov_36_148: frac[3,148] + frac[12,148] + frac[16,148] + frac[19,148]
   + frac[20,148] + frac[21,148] + frac[22,148] >= 0.9
 cov_36_149: frac[3,149] + frac[12,149] + frac[16,149] + frac[19,149]
   + frac[20,149] + frac[21,149] + frac[22,149] >= 0.9
 cov_37_78: frac[5,78] + frac[15,78] + frac[17,78] + frac[19,78]
   + frac[20,78] + frac[21,78] + frac[22,78] >= 0.9
 cov_37_79: frac[5,79] + frac[15,79] + frac[17,79] + frac[19,79]
   + frac[20,79] + frac[21,79] + frac[22,79] >= 0.9
 cov_37_80: frac[5,80] + frac[15,80] + frac[17,80] + frac[19,80]
   + frac[20,80] + frac[21,80] + frac[22,80] >= 0.9
 cov_38_126: frac[9,126] + frac[11,126] + frac[18,126] >= 0.8
 cov_38_127: frac[9,127] + frac[11,127] + frac[18,127] >= 0.8
 cov_38_128: frac[9,128] + frac[11,128] + frac[18,128] >= 0.8
 cov_39_39: frac[1,39] + frac[3,39] + frac[16,39] >= 0.9
 cov_39_40: frac[1,40] + frac[3,40] + frac[16,40] >= 0.9
 cov_39_41: frac[1,41] + frac[3,41] + frac[16,41] >= 0.9
 cov_40_162: frac[1,162] + frac[4,162] + frac[16,162] + frac[17,162]
   + frac[20,162] >= 1
 cov_40_163: frac[1,163] + frac[4,163] + frac[16,163] + frac[17,163]
   + frac[20,163] >= 1
 cov_40_164: frac[1,164] + frac[4,164] + frac[16,164] + frac[17,164]
   + frac[20,164] >= 1
 cov_41_57: frac[4,57] + frac[6,57] + frac[17,57] >= 0.8
 cov_41_58: frac[4,58] + frac[6,58] + frac[17,58] >= 0.8
 cov_41_59: frac[4,59] + frac[6,59] + frac[17,59] >= 0.8
 cov_42_84: frac[6,84] + frac[7,84] + frac[17,84] >= 0.8
 cov_42_85: frac[6,85] + frac[7,85] + frac[17,85] >= 0.8
 cov_42_86: frac[6,86] + frac[7,86] + frac[17,86] >= 0.8
 cov_43_93: frac[3,93] + frac[11,93] + frac[16,93] + frac[18,93]
   + frac[20,93] + frac[21,93] + frac[22,93] >= 0.9
 cov_43_94: frac[3,94] + frac[11,94] + frac[16,94] + frac[18,94]
   + frac[20,94] + frac[21,94] + frac[22,94] >= 0.9
 cov_43_95: frac[3,95] + frac[11,95] + frac[16,95] + frac[18,95]
   + frac[20,95] + frac[21,95] + frac[22,95] >= 0.9
 cov_44_102: frac[0,102] + frac[2,102] + frac[16,102] >= 0.9
 cov_44_103: frac[0,103] + frac[2,103] + frac[16,103] >= 0.9
 cov_44_104: frac[0,104] + frac[2,104] + frac[16,104] >= 0.9
 cov_45_27: frac[6,27] + frac[9,27] + frac[17,27] + frac[18,27]
   + frac[20,27] + frac[21,27] + frac[22,27] >= 0.8
 cov_45_28: frac[6,28] + frac[9,28] + frac[17,28] + frac[18,28]
   + frac[20,28] + frac[21,28] + frac[22,28] >= 0.8
 cov_45_29: frac[6,29] + frac[9,29] + frac[17,29] + frac[18,29]
   + frac[20,29] + frac[21,29] + frac[22,29] >= 0.8
 cov_46_6: frac[1,6] + frac[9,6] + frac[16,6] + frac[18,6] + frac[20,6]
   + frac[21,6] + frac[22,6] >= 0.8
 cov_46_7: frac[1,7] + frac[9,7] + frac[16,7] + frac[18,7] + frac[20,7]
   + frac[21,7] + frac[22,7] >= 0.8
 cov_46_8: frac[1,8] + frac[9,8] + frac[16,8] + frac[18,8] + frac[20,8]
   + frac[21,8] + frac[22,8] >= 0.8
 cov_47_21: frac[9,21] + frac[12,21] + frac[18,21] + frac[19,21]
   + frac[21,21] >= 1
 cov_47_22: frac[9,22] + frac[12,22] + frac[18,22] + frac[19,22]
   + frac[21,22] >= 1
 cov_47_23: frac[9,23] + frac[12,23] + frac[18,23] + frac[19,23]
   + frac[21,23] >= 1
 cov_48_84: frac[10,84] + frac[12,84] + frac[18,84] + frac[19,84]
   + frac[21,84] >= 0.8
 cov_48_85: frac[10,85] + frac[12,85] + frac[18,85] + frac[19,85]
   + frac[21,85] >= 0.8
 cov_48_86: frac[10,86] + frac[12,86] + frac[18,86] + frac[19,86]
   + frac[21,86] >= 0.8
 cov_49_54: frac[8,54] + frac[10,54] + frac[18,54] >= 0.8
 cov_49_55: frac[8,55] + frac[10,55] + frac[18,55] >= 0.8
 cov_49_56: frac[8,56] + frac[10,56] + frac[18,56] >= 0.8
 cov_50_186: frac[1,186] + frac[14,186] + frac[16,186] + frac[19,186]
   + frac[20,186] + frac[21,186] + frac[22,186] >= 0.9
 cov_50_187: frac[1,187] + frac[14,187] + frac[16,187] + frac[19,187]
   + frac[20,187] + frac[21,187] + frac[22,187] >= 0.9
 cov_50_188: frac[1,188] + frac[14,188] + frac[16,188] + frac[19,188]
   + frac[20,188] + frac[21,188] + frac[22,188] >= 0.9
 cov_51_153: frac[6,153] + frac[9,153] + frac[17,153] + frac[18,153]
   + frac[20,153] + frac[21,153] + frac[22,153] >= 1
 cov_51_154: frac[6,154] + frac[9,154] + frac[17,154] + frac[18,154]
   + frac[20,154] + frac[21,154] + frac[22,154] >= 1
 cov_51_155: frac[6,155] + frac[9,155] + frac[17,155] + frac[18,155]
   + frac[20,155] + frac[21,155] + frac[22,155] >= 1
 cov_52_3: frac[8,3] + frac[10,3] + frac[18,3] >= 0.8
 cov_52_4: frac[8,4] + frac[10,4] + frac[18,4] >= 0.8
 cov_52_5: frac[8,5] + frac[10,5] + frac[18,5] >= 0.8
 cov_53_150: frac[8,150] + frac[11,150] + frac[18,150] >= 0.7
 cov_53_151: frac[8,151] + frac[11,151] + frac[18,151] >= 0.7
 cov_53_152: frac[8,152] + frac[11,152] + frac[18,152] >= 0.7
 cov_54_183: frac[8,183] + frac[13,183] + frac[18,183] + frac[19,183]
   + frac[21,183] >= 0.9
 cov_54_184: frac[8,184] + frac[13,184] + frac[18,184] + frac[19,184]
   + frac[21,184] >= 0.9
 cov_54_185: frac[8,185] + frac[13,185] + frac[18,185] + frac[19,185]
   + frac[21,185] >= 0.9
 cov_55_186: frac[2,186] + frac[9,186] + frac[16,186] + frac[18,186]
   + frac[20,186] + frac[21,186] + frac[22,186] >= 0.8
 cov_55_187: frac[2,187] + frac[9,187] + frac[16,187] + frac[18,187]
   + frac[20,187] + frac[21,187] + frac[22,187] >= 0.8
 cov_55_188: frac[2,188] + frac[9,188] + frac[16,188] + frac[18,188]
   + frac[20,188] + frac[21,188] + frac[22,188] >= 0.8
 cov_56_132: frac[5,132] + frac[6,132] + frac[17,132] >= 1
 cov_56_133: frac[5,133] + frac[6,133] + frac[17,133] >= 1
 cov_56_134: frac[5,134] + frac[6,134] + frac[17,134] >= 1
 cov_57_153: frac[0,153] + frac[8,153] + frac[16,153] + frac[18,153]
   + frac[20,153] + frac[21,153] + frac[22,153] >= 1
 cov_57_154: frac[0,154] + frac[8,154] + frac[16,154] + frac[18,154]
   + frac[20,154] + frac[21,154] + frac[22,154] >= 1
 cov_57_155: frac[0,155] + frac[8,155] + frac[16,155] + frac[18,155]
   + frac[20,155] + frac[21,155] + frac[22,155] >= 1
 cov_58_21: frac[1,21] + frac[13,21] + frac[16,21] + frac[19,21]
   + frac[20,21] + frac[21,21] + frac[22,21] >= 0.9
 cov_58_22: frac[1,22] + frac[13,22] + frac[16,22] + frac[19,22]
   + frac[20,22] + frac[21,22] + frac[22,22] >= 0.9
 cov_58_23: frac[1,23] + frac[13,23] + frac[16,23] + frac[19,23]
   + frac[20,23] + frac[21,23] + frac[22,23] >= 0.9
 cov_59_15: frac[11,15] + frac[15,15] + frac[18,15] + frac[19,15]
   + frac[21,15] >= 0.8
 cov_59_16: frac[11,16] + frac[15,16] + frac[18,16] + frac[19,16]
   + frac[21,16] >= 0.8
 cov_59_17: frac[11,17] + frac[15,17] + frac[18,17] + frac[19,17]
   + frac[21,17] >= 0.8
 cov_60_9: frac[9,9] + frac[13,9] + frac[18,9] + frac[19,9] + frac[21,9]
   >= 0.7
 cov_60_10: frac[9,10] + frac[13,10] + frac[18,10] + frac[19,10]
   + frac[21,10] >= 0.7
 cov_60_11: frac[9,11] + frac[13,11] + frac[18,11] + frac[19,11]
   + frac[21,11] >= 0.7
 cov_61_153: frac[4,153] + frac[9,153] + frac[17,153] + frac[18,153]
   + frac[20,153] + frac[21,153] + frac[22,153] >= 0.7
 cov_61_154: frac[4,154] + frac[9,154] + frac[17,154] + frac[18,154]
   + frac[20,154] + frac[21,154] + frac[22,154] >= 0.7
 cov_61_155: frac[4,155] + frac[9,155] + frac[17,155] + frac[18,155]
   + frac[20,155] + frac[21,155] + frac[22,155] >= 0.7
 cov_62_180: frac[8,180] + frac[10,180] + frac[18,180] >= 1
 cov_62_181: frac[8,181] + frac[10,181] + frac[18,181] >= 1
 cov_62_182: frac[8,182] + frac[10,182] + frac[18,182] >= 1
 cov_63_15: frac[6,15] + frac[7,15] + frac[17,15] >= 0.7
 cov_63_16: frac[6,16] + frac[7,16] + frac[17,16] >= 0.7
 cov_63_17: frac[6,17] + frac[7,17] + frac[17,17] >= 0.7
 cov_64_189: frac[4,189] + frac[14,189] + frac[17,189] + frac[19,189]
   + frac[20,189] + frac[21,189] + frac[22,189] >= 0.8
 cov_64_190: frac[4,190] + frac[14,190] + frac[17,190] + frac[19,190]
   + frac[20,190] + frac[21,190] + frac[22,190] >= 0.8
 cov_64_191: frac[4,191] + frac[14,191] + frac[17,191] + frac[19,191]
   + frac[20,191] + frac[21,191] + frac[22,191] >= 0.8
 cov_65_171: frac[9,171] + frac[14,171] + frac[18,171] + frac[19,171]
   + frac[21,171] >= 0.9
 cov_65_172: frac[9,172] + frac[14,172] + frac[18,172] + frac[19,172]
   + frac[21,172] >= 0.9
 cov_65_173: frac[9,173] + frac[14,173] + frac[18,173] + frac[19,173]
   + frac[21,173] >= 0.9
 cov_66_120: frac[8,120] + frac[9,120] + frac[18,120] >= 1
 cov_66_121: frac[8,121] + frac[9,121] + frac[18,121] >= 1
 cov_66_122: frac[8,122] + frac[9,122] + frac[18,122] >= 1
 cov_67_111: frac[12,111] + frac[15,111] + frac[19,111] >= 0.8
 cov_67_112: frac[12,112] + frac[15,112] + frac[19,112] >= 0.8
 cov_67_113: frac[12,113] + frac[15,113] + frac[19,113] >= 0.8
 cov_68_9: frac[0,9] + frac[10,9] + frac[16,9] + frac[18,9] + frac[20,9]
   + frac[21,9] + frac[22,9] >= 0.7
 cov_68_10: frac[0,10] + frac[10,10] + frac[16,10] + frac[18,10]
   + frac[20,10] + frac[21,10] + frac[22,10] >= 0.7
 cov_68_11: frac[0,11] + frac[10,11] + frac[16,11] + frac[18,11]
   + frac[20,11] + frac[21,11] + frac[22,11] >= 0.7
 cov_69_180: frac[11,180] + frac[12,180] + frac[18,180] + frac[19,180]
   + frac[21,180] >= 0.7
 cov_69_181: frac[11,181] + frac[12,181] + frac[18,181] + frac[19,181]
   + frac[21,181] >= 0.7
 cov_69_182: frac[11,182] + frac[12,182] + frac[18,182] + frac[19,182]
   + frac[21,182] >= 0.7
 cov_70_42: frac[7,42] + frac[8,42] + frac[17,42] + frac[18,42]
   + frac[20,42] + frac[21,42] + frac[22,42] >= 0.8
 cov_70_43: frac[7,43] + frac[8,43] + frac[17,43] + frac[18,43]
   + frac[20,43] + frac[21,43] + frac[22,43] >= 0.8
 cov_70_44: frac[7,44] + frac[8,44] + frac[17,44] + frac[18,44]
   + frac[20,44] + frac[21,44] + frac[22,44] >= 0.8
 cov_71_150: frac[2,150] + frac[15,150] + frac[16,150] + frac[19,150]
   + frac[20,150] + frac[21,150] + frac[22,150] >= 0.8
 cov_71_151: frac[2,151] + frac[15,151] + frac[16,151] + frac[19,151]
   + frac[20,151] + frac[21,151] + frac[22,151] >= 0.8
 cov_71_152: frac[2,152] + frac[15,152] + frac[16,152] + frac[19,152]
   + frac[20,152] + frac[21,152] + frac[22,152] >= 0.8
 cov_72_66: frac[6,66] + frac[12,66] + frac[17,66] + frac[19,66]
   + frac[20,66] + frac[21,66] + frac[22,66] >= 0.9
 cov_72_67: frac[6,67] + frac[12,67] + frac[17,67] + frac[19,67]
   + frac[20,67] + frac[21,67] + frac[22,67] >= 0.9
 cov_72_68: frac[6,68] + frac[12,68] + frac[17,68] + frac[19,68]
   + frac[20,68] + frac[21,68] + frac[22,68] >= 0.9
 cov_73_114: frac[8,114] + frac[15,114] + frac[18,114] + frac[19,114]
   + frac[21,114] >= 1
 cov_73_115: frac[8,115] + frac[15,115] + frac[18,115] + frac[19,115]
   + frac[21,115] >= 1
 cov_73_116: frac[8,116] + frac[15,116] + frac[18,116] + frac[19,116]
   + frac[21,116] >= 1
 cov_74_15: frac[3,15] + frac[12,15] + frac[16,15] + frac[19,15]
   + frac[20,15] + frac[21,15] + frac[22,15] >= 0.8
 cov_74_16: frac[3,16] + frac[12,16] + frac[16,16] + frac[19,16]
   + frac[20,16] + frac[21,16] + frac[22,16] >= 0.8
 cov_74_17: frac[3,17] + frac[12,17] + frac[16,17] + frac[19,17]
   + frac[20,17] + frac[21,17] + frac[22,17] >= 0.8
 cov_75_153: frac[0,153] + frac[6,153] + frac[16,153] + frac[17,153]
   + frac[20,153] >= 0.8
 cov_75_154: frac[0,154] + frac[6,154] + frac[16,154] + frac[17,154]
   + frac[20,154] >= 0.8
 cov_75_155: frac[0,155] + frac[6,155] + frac[16,155] + frac[17,155]
   + frac[20,155] >= 0.8
 cov_76_189: frac[12,189] + frac[14,189] + frac[19,189] >= 1
 cov_76_190: frac[12,190] + frac[14,190] + frac[19,190] >= 1
 cov_76_191: frac[12,191] + frac[14,191] + frac[19,191] >= 1
 cov_77_51: frac[1,51] + frac[2,51] + frac[16,51] >= 0.8
 cov_77_52: frac[1,52] + frac[2,52] + frac[16,52] >= 0.8
 cov_77_53: frac[1,53] + frac[2,53] + frac[16,53] >= 0.8
 cov_78_27: frac[6,27] + frac[13,27] + frac[17,27] + frac[19,27]
   + frac[20,27] + frac[21,27] + frac[22,27] >= 0.8
 cov_78_28: frac[6,28] + frac[13,28] + frac[17,28] + frac[19,28]
   + frac[20,28] + frac[21,28] + frac[22,28] >= 0.8
 cov_78_29: frac[6,29] + frac[13,29] + frac[17,29] + frac[19,29]
   + frac[20,29] + frac[21,29] + frac[22,29] >= 0.8
 cov_79_141: frac[9,141] + frac[13,141] + frac[18,141] + frac[19,141]
   + frac[21,141] >= 0.7
 cov_79_142: frac[9,142] + frac[13,142] + frac[18,142] + frac[19,142]
   + frac[21,142] >= 0.7
 cov_79_143: frac[9,143] + frac[13,143] + frac[18,143] + frac[19,143]
   + frac[21,143] >= 0.7
 cov_80_63: frac[0,63] + frac[1,63] + frac[16,63] >= 0.7
 cov_80_64: frac[0,64] + frac[1,64] + frac[16,64] >= 0.7
 cov_80_65: frac[0,65] + frac[1,65] + frac[16,65] >= 0.7
 cov_81_168: frac[2,168] + frac[9,168] + frac[16,168] + frac[18,168]
   + frac[20,168] + frac[21,168] + frac[22,168] >= 0.7
 cov_81_169: frac[2,169] + frac[9,169] + frac[16,169] + frac[18,169]
   + frac[20,169] + frac[21,169] + frac[22,169] >= 0.7
 cov_81_170: frac[2,170] + frac[9,170] + frac[16,170] + frac[18,170]
   + frac[20,170] + frac[21,170] + frac[22,170] >= 0.7
 cov_82_171: frac[9,171] + frac[15,171] + frac[18,171] + frac[19,171]
   + frac[21,171] >= 0.7
 cov_82_172: frac[9,172] + frac[15,172] + frac[18,172] + frac[19,172]
   + frac[21,172] >= 0.7
 cov_82_173: frac[9,173] + frac[15,173] + frac[18,173] + frac[19,173]
   + frac[21,173] >= 0.7
 cov_83_57: frac[1,57] + frac[10,57] + frac[16,57] + frac[18,57]
   + frac[20,57] + frac[21,57] + frac[22,57] >= 0.7
 cov_83_58: frac[1,58] + frac[10,58] + frac[16,58] + frac[18,58]
   + frac[20,58] + frac[21,58] + frac[22,58] >= 0.7
 cov_83_59: frac[1,59] + frac[10,59] + frac[16,59] + frac[18,59]
   + frac[20,59] + frac[21,59] + frac[22,59] >= 0.7
 cov_84_27: frac[0,27] + frac[13,27] + frac[16,27] + frac[19,27]
   + frac[20,27] + frac[21,27] + frac[22,27] >= 0.8
 cov_84_28: frac[0,28] + frac[13,28] + frac[16,28] + frac[19,28]
   + frac[20,28] + frac[21,28] + frac[22,28] >= 0.8
 cov_84_29: frac[0,29] + frac[13,29] + frac[16,29] + frac[19,29]
   + frac[20,29] + frac[21,29] + frac[22,29] >= 0.8
 cov_85_183: frac[6,183] + frac[10,183] + frac[17,183] + frac[18,183]
   + frac[20,183] + frac[21,183] + frac[22,183] >= 0.8
 cov_85_184: frac[6,184] + frac[10,184] + frac[17,184] + frac[18,184]
   + frac[20,184] + frac[21,184] + frac[22,184] >= 0.8
 cov_85_185: frac[6,185] + frac[10,185] + frac[17,185] + frac[18,185]
   + frac[20,185] + frac[21,185] + frac[22,185] >= 0.8
 cov_86_72: frac[4,72] + frac[8,72] + frac[17,72] + frac[18,72]
   + frac[20,72] + frac[21,72] + frac[22,72] >= 0.9
 cov_86_73: frac[4,73] + frac[8,73] + frac[17,73] + frac[18,73]
   + frac[20,73] + frac[21,73] + frac[22,73] >= 0.9
 cov_86_74: frac[4,74] + frac[8,74] + frac[17,74] + frac[18,74]
   + frac[20,74] + frac[21,74] + frac[22,74] >= 0.9
 cov_87_84: frac[3,84] + frac[12,84] + frac[16,84] + frac[19,84]
   + frac[20,84] + frac[21,84] + frac[22,84] >= 0.9
 cov_87_85: frac[3,85] + frac[12,85] + frac[16,85] + frac[19,85]
   + frac[20,85] + frac[21,85] + frac[22,85] >= 0.9
 cov_87_86: frac[3,86] + frac[12,86] + frac[16,86] + frac[19,86]
   + frac[20,86] + frac[21,86] + frac[22,86] >= 0.9
 cov_88_24: frac[7,24] + frac[9,24] + frac[17,24] + frac[18,24]
   + frac[20,24] + frac[21,24] + frac[22,24] >= 0.8
 cov_88_25: frac[7,25] + frac[9,25] + frac[17,25] + frac[18,25]
   + frac[20,25] + frac[21,25] + frac[22,25] >= 0.8
 cov_88_26: frac[7,26] + frac[9,26] + frac[17,26] + frac[18,26]
   + frac[20,26] + frac[21,26] + frac[22,26] >= 0.8
 cov_89_3: frac[14,3] + frac[15,3] + frac[19,3] >= 0.9
 cov_89_4: frac[14,4] + frac[15,4] + frac[19,4] >= 0.9
 cov_89_5: frac[14,5] + frac[15,5] + frac[19,5] >= 0.9
 cov_90_18: frac[7,18] + frac[13,18] + frac[17,18] + frac[19,18]
   + frac[20,18] + frac[21,18] + frac[22,18] >= 1
 cov_90_19: frac[7,19] + frac[13,19] + frac[17,19] + frac[19,19]
   + frac[20,19] + frac[21,19] + frac[22,19] >= 1
 cov_90_20: frac[7,20] + frac[13,20] + frac[17,20] + frac[19,20]
   + frac[20,20] + frac[21,20] + frac[22,20] >= 1
 cov_91_111: frac[11,111] + frac[15,111] + frac[18,111] + frac[19,111]
   + frac[21,111] >= 1
 cov_91_112: frac[11,112] + frac[15,112] + frac[18,112] + frac[19,112]
   + frac[21,112] >= 1
 cov_91_113: frac[11,113] + frac[15,113] + frac[18,113] + frac[19,113]
   + frac[21,113] >= 1
 cov_92_153: frac[12,153] + frac[15,153] + frac[19,153] >= 0.7
 cov_92_154: frac[12,154] + frac[15,154] + frac[19,154] >= 0.7
 cov_92_155: frac[12,155] + frac[15,155] + frac[19,155] >= 0.7
 cov_93_174: frac[6,174] + frac[14,174] + frac[17,174] + frac[19,174]
   + frac[20,174] + frac[21,174] + frac[22,174] >= 0.8
 cov_93_175: frac[6,175] + frac[14,175] + frac[17,175] + frac[19,175]
   + frac[20,175] + frac[21,175] + frac[22,175] >= 0.8
 cov_93_176: frac[6,176] + frac[14,176] + frac[17,176] + frac[19,176]
   + frac[20,176] + frac[21,176] + frac[22,176] >= 0.8
 cov_94_42: frac[7,42] + frac[8,42] + frac[17,42] + frac[18,42]
   + frac[20,42] + frac[21,42] + frac[22,42] >= 0.8
 cov_94_43: frac[7,43] + frac[8,43] + frac[17,43] + frac[18,43]
   + frac[20,43] + frac[21,43] + frac[22,43] >= 0.8
 cov_94_44: frac[7,44] + frac[8,44] + frac[17,44] + frac[18,44]
   + frac[20,44] + frac[21,44] + frac[22,44] >= 0.8
 cov_95_87: frac[11,87] + frac[12,87] + frac[18,87] + frac[19,87]
   + frac[21,87] >= 0.7
 cov_95_88: frac[11,88] + frac[12,88] + frac[18,88] + frac[19,88]
   + frac[21,88] >= 0.7
 cov_95_89: frac[11,89] + frac[12,89] + frac[18,89] + frac[19,89]
   + frac[21,89] >= 0.7
 cov_96_45: frac[5,45] + frac[14,45] + frac[17,45] + frac[19,45]
   + frac[20,45] + frac[21,45] + frac[22,45] >= 1
 cov_96_46: frac[5,46] + frac[14,46] + frac[17,46] + frac[19,46]
   + frac[20,46] + frac[21,46] + frac[22,46] >= 1
 cov_96_47: frac[5,47] + frac[14,47] + frac[17,47] + frac[19,47]
   + frac[20,47] + frac[21,47] + frac[22,47] >= 1
 cov_97_174: frac[2,174] + frac[13,174] + frac[16,174] + frac[19,174]
   + frac[20,174] + frac[21,174] + frac[22,174] >= 1
 cov_97_175: frac[2,175] + frac[13,175] + frac[16,175] + frac[19,175]
   + frac[20,175] + frac[21,175] + frac[22,175] >= 1
 cov_97_176: frac[2,176] + frac[13,176] + frac[16,176] + frac[19,176]
   + frac[20,176] + frac[21,176] + frac[22,176] >= 1
 cov_98_150: frac[7,150] + frac[14,150] + frac[17,150] + frac[19,150]
   + frac[20,150] + frac[21,150] + frac[22,150] >= 0.9
 cov_98_151: frac[7,151] + frac[14,151] + frac[17,151] + frac[19,151]
   + frac[20,151] + frac[21,151] + frac[22,151] >= 0.9
 cov_98_152: frac[7,152] + frac[14,152] + frac[17,152] + frac[19,152]
   + frac[20,152] + frac[21,152] + frac[22,152] >= 0.9
 cov_99_48: frac[4,48] + frac[15,48] + frac[17,48] + frac[19,48]
   + frac[20,48] + frac[21,48] + frac[22,48] >= 0.7
 cov_99_49: frac[4,49] + frac[15,49] + frac[17,49] + frac[19,49]
   + frac[20,49] + frac[21,49] + frac[22,49] >= 0.7
 cov_99_50: frac[4,50] + frac[15,50] + frac[17,50] + frac[19,50]
   + frac[20,50] + frac[21,50] + frac[22,50] >= 0.7
 cov_100_51: frac[10,51] + frac[12,51] + frac[18,51] + frac[19,51]
   + frac[21,51] >= 0.8
 cov_100_52: frac[10,52] + frac[12,52] + frac[18,52] + frac[19,52]
   + frac[21,52] >= 0.8
 cov_100_53: frac[10,53] + frac[12,53] + frac[18,53] + frac[19,53]
   + frac[21,53] >= 0.8
 cov_101_183: frac[5,183] + frac[6,183] + frac[17,183] >= 1
 cov_101_184: frac[5,184] + frac[6,184] + frac[17,184] >= 1
 cov_101_185: frac[5,185] + frac[6,185] + frac[17,185] >= 1
 cov_102_162: frac[6,162] + frac[7,162] + frac[17,162] >= 0.9
 cov_102_163: frac[6,163] + frac[7,163] + frac[17,163] >= 0.9
 cov_102_164: frac[6,164] + frac[7,164] + frac[17,164] >= 0.9
 cov_103_168: frac[10,168] + frac[15,168] + frac[18,168] + frac[19,168]
   + frac[21,168] >= 0.8
 cov_103_169: frac[10,169] + frac[15,169] + frac[18,169] + frac[19,169]
   + frac[21,169] >= 0.8
 cov_103_170: frac[10,170] + frac[15,170] + frac[18,170] + frac[19,170]
   + frac[21,170] >= 0.8
 cov_104_66: frac[1,66] + frac[6,66] + frac[16,66] + frac[17,66]
   + frac[20,66] >= 1
 cov_104_67: frac[1,67] + frac[6,67] + frac[16,67] + frac[17,67]
   + frac[20,67] >= 1
 cov_104_68: frac[1,68] + frac[6,68] + frac[16,68] + frac[17,68]
   + frac[20,68] >= 1
 cov_105_114: frac[2,114] + frac[9,114] + frac[16,114] + frac[18,114]
   + frac[20,114] + frac[21,114] + frac[22,114] >= 1
 cov_105_115: frac[2,115] + frac[9,115] + frac[16,115] + frac[18,115]
   + frac[20,115] + frac[21,115] + frac[22,115] >= 1
 cov_105_116: frac[2,116] + frac[9,116] + frac[16,116] + frac[18,116]
   + frac[20,116] + frac[21,116] + frac[22,116] >= 1
 cov_106_135: frac[6,135] + frac[8,135] + frac[17,135] + frac[18,135]
   + frac[20,135] + frac[21,135] + frac[22,135] >= 0.9
 cov_106_136: frac[6,136] + frac[8,136] + frac[17,136] + frac[18,136]
   + frac[20,136] + frac[21,136] + frac[22,136] >= 0.9
 cov_106_137: frac[6,137] + frac[8,137] + frac[17,137] + frac[18,137]
   + frac[20,137] + frac[21,137] + frac[22,137] >= 0.9
 cov_107_114: frac[10,114] + frac[11,114] + frac[18,114] >= 0.7
 cov_107_115: frac[10,115] + frac[11,115] + frac[18,115] >= 0.7
 cov_107_116: frac[10,116] + frac[11,116] + frac[18,116] >= 0.7
 cov_108_0: frac[1,0] + frac[6,0] + frac[16,0] + frac[17,0] + frac[20,0]
   >= 0.7
 cov_108_1: frac[1,1] + frac[6,1] + frac[16,1] + frac[17,1] + frac[20,1]
   >= 0.7
 cov_108_2: frac[1,2] + frac[6,2] + frac[16,2] + frac[17,2] + frac[20,2]
   >= 0.7
 cov_109_42: frac[0,42] + frac[3,42] + frac[16,42] >= 0.7
 cov_109_43: frac[0,43] + frac[3,43] + frac[16,43] >= 0.7
 cov_109_44: frac[0,44] + frac[3,44] + frac[16,44] >= 0.7
 cov_110_120: frac[2,120] + frac[5,120] + frac[16,120] + frac[17,120]
   + frac[20,120] >= 0.9
 cov_110_121: frac[2,121] + frac[5,121] + frac[16,121] + frac[17,121]
   + frac[20,121] >= 0.9
 cov_110_122: frac[2,122] + frac[5,122] + frac[16,122] + frac[17,122]
   + frac[20,122] >= 0.9
 cov_111_183: frac[0,183] + frac[6,183] + frac[16,183] + frac[17,183]
   + frac[20,183] >= 0.7
 cov_111_184: frac[0,184] + frac[6,184] + frac[16,184] + frac[17,184]
   + frac[20,184] >= 0.7
 cov_111_185: frac[0,185] + frac[6,185] + frac[16,185] + frac[17,185]
   + frac[20,185] >= 0.7
 cov_112_129: frac[2,129] + frac[14,129] + frac[16,129] + frac[19,129]
   + frac[20,129] + frac[21,129] + frac[22,129] >= 1
 cov_112_130: frac[2,130] + frac[14,130] + frac[16,130] + frac[19,130]
   + frac[20,130] + frac[21,130] + frac[22,130] >= 1
 cov_112_131: frac[2,131] + frac[14,131] + frac[16,131] + frac[19,131]
   + frac[20,131] + frac[21,131] + frac[22,131] >= 1
 cov_113_51: frac[2,51] + frac[8,51] + frac[16,51] + frac[18,51]
   + frac[20,51] + frac[21,51] + frac[22,51] >= 1
 cov_113_52: frac[2,52] + frac[8,52] + frac[16,52] + frac[18,52]
   + frac[20,52] + frac[21,52] + frac[22,52] >= 1
 cov_113_53: frac[2,53] + frac[8,53] + frac[16,53] + frac[18,53]
   + frac[20,53] + frac[21,53] + frac[22,53] >= 1
 cov_114_33: frac[3,33] + frac[8,33] + frac[16,33] + frac[18,33]
   + frac[20,33] + frac[21,33] + frac[22,33] >= 0.9
 cov_114_34: frac[3,34] + frac[8,34] + frac[16,34] + frac[18,34]
   + frac[20,34] + frac[21,34] + frac[22,34] >= 0.9
 cov_114_35: frac[3,35] + frac[8,35] + frac[16,35] + frac[18,35]
   + frac[20,35] + frac[21,35] + frac[22,35] >= 0.9
 cov_115_54: frac[3,54] + frac[10,54] + frac[16,54] + frac[18,54]
   + frac[20,54] + frac[21,54] + frac[22,54] >= 0.7
 cov_115_55: frac[3,55] + frac[10,55] + frac[16,55] + frac[18,55]
   + frac[20,55] + frac[21,55] + frac[22,55] >= 0.7
 cov_115_56: frac[3,56] + frac[10,56] + frac[16,56] + frac[18,56]
   + frac[20,56] + frac[21,56] + frac[22,56] >= 0.7
 cov_116_87: frac[3,87] + frac[10,87] + frac[16,87] + frac[18,87]
   + frac[20,87] + frac[21,87] + frac[22,87] >= 0.9
 cov_116_88: frac[3,88] + frac[10,88] + frac[16,88] + frac[18,88]
   + frac[20,88] + frac[21,88] + frac[22,88] >= 0.9
 cov_116_89: frac[3,89] + frac[10,89] + frac[16,89] + frac[18,89]
   + frac[20,89] + frac[21,89] + frac[22,89] >= 0.9
 cov_117_84: frac[4,84] + frac[7,84] + frac[17,84] >= 0.7
 cov_117_85: frac[4,85] + frac[7,85] + frac[17,85] >= 0.7
 cov_117_86: frac[4,86] + frac[7,86] + frac[17,86] >= 0.7
 cov_118_66: frac[5,66] + frac[14,66] + frac[17,66] + frac[19,66]
   + frac[20,66] + frac[21,66] + frac[22,66] >= 0.8
 cov_118_67: frac[5,67] + frac[14,67] + frac[17,67] + frac[19,67]
   + frac[20,67] + frac[21,67] + frac[22,67] >= 0.8
 cov_118_68: frac[5,68] + frac[14,68] + frac[17,68] + frac[19,68]
   + frac[20,68] + frac[21,68] + frac[22,68] >= 0.8
 cov_119_126: frac[2,126] + frac[10,126] + frac[16,126] + frac[18,126]
   + frac[20,126] + frac[21,126] + frac[22,126] >= 1
 cov_119_127: frac[2,127] + frac[10,127] + frac[16,127] + frac[18,127]
   + frac[20,127] + frac[21,127] + frac[22,127] >= 1
 cov_119_128: frac[2,128] + frac[10,128] + frac[16,128] + frac[18,128]
   + frac[20,128] + frac[21,128] + frac[22,128] >= 1
 cov_120_120: frac[7,120] + frac[15,120] + frac[17,120] + frac[19,120]
   + frac[20,120] + frac[21,120] + frac[22,120] >= 0.7
 cov_120_121: frac[7,121] + frac[15,121] + frac[17,121] + frac[19,121]
   + frac[20,121] + frac[21,121] + frac[22,121] >= 0.7
 cov_120_122: frac[7,122] + frac[15,122] + frac[17,122] + frac[19,122]
   + frac[20,122] + frac[21,122] + frac[22,122] >= 0.7
 cov_121_30: frac[4,30] + frac[14,30] + frac[17,30] + frac[19,30]
   + frac[20,30] + frac[21,30] + frac[22,30] >= 0.7
 cov_121_31: frac[4,31] + frac[14,31] + frac[17,31] + frac[19,31]
   + frac[20,31] + frac[21,31] + frac[22,31] >= 0.7
 cov_121_32: frac[4,32] + frac[14,32] + frac[17,32] + frac[19,32]
   + frac[20,32] + frac[21,32] + frac[22,32] >= 0.7
 cov_122_120: frac[0,120] + frac[13,120] + frac[16,120] + frac[19,120]
   + frac[20,120] + frac[21,120] + frac[22,120] >= 0.8
 cov_122_121: frac[0,121] + frac[13,121] + frac[16,121] + frac[19,121]
   + frac[20,121] + frac[21,121] + frac[22,121] >= 0.8
 cov_122_122: frac[0,122] + frac[13,122] + frac[16,122] + frac[19,122]
   + frac[20,122] + frac[21,122] + frac[22,122] >= 0.8
 cov_123_111: frac[3,111] + frac[11,111] + frac[16,111] + frac[18,111]
   + frac[20,111] + frac[21,111] + frac[22,111] >= 0.7
 cov_123_112: frac[3,112] + frac[11,112] + frac[16,112] + frac[18,112]
   + frac[20,112] + frac[21,112] + frac[22,112] >= 0.7
 cov_123_113: frac[3,113] + frac[11,113] + frac[16,113] + frac[18,113]
   + frac[20,113] + frac[21,113] + frac[22,113] >= 0.7
 cov_124_36: frac[13,36] + frac[14,36] + frac[19,36] >= 1
 cov_124_37: frac[13,37] + frac[14,37] + frac[19,37] >= 1
 cov_124_38: frac[13,38] + frac[14,38] + frac[19,38] >= 1
 cov_125_174: frac[3,174] + frac[11,174] + frac[16,174] + frac[18,174]
   + frac[20,174] + frac[21,174] + frac[22,174] >= 1
 cov_125_175: frac[3,175] + frac[11,175] + frac[16,175] + frac[18,175]
   + frac[20,175] + frac[21,175] + frac[22,175] >= 1
 cov_125_176: frac[3,176] + frac[11,176] + frac[16,176] + frac[18,176]
   + frac[20,176] + frac[21,176] + frac[22,176] >= 1
 cov_126_30: frac[8,30] + frac[15,30] + frac[18,30] + frac[19,30]
   + frac[21,30] >= 0.7
 cov_126_31: frac[8,31] + frac[15,31] + frac[18,31] + frac[19,31]
   + frac[21,31] >= 0.7
 cov_126_32: frac[8,32] + frac[15,32] + frac[18,32] + frac[19,32]
   + frac[21,32] >= 0.7
 cov_127_186: frac[8,186] + frac[14,186] + frac[18,186] + frac[19,186]
   + frac[21,186] >= 0.8
 cov_127_187: frac[8,187] + frac[14,187] + frac[18,187] + frac[19,187]
   + frac[21,187] >= 0.8
 cov_127_188: frac[8,188] + frac[14,188] + frac[18,188] + frac[19,188]
   + frac[21,188] >= 0.8
 cov_128_69: frac[13,69] + frac[14,69] + frac[19,69] >= 0.9
 cov_128_70: frac[13,70] + frac[14,70] + frac[19,70] >= 0.9
 cov_128_71: frac[13,71] + frac[14,71] + frac[19,71] >= 0.9
 cov_129_90: frac[2,90] + frac[9,90] + frac[16,90] + frac[18,90]
   + frac[20,90] + frac[21,90] + frac[22,90] >= 1
 cov_129_91: frac[2,91] + frac[9,91] + frac[16,91] + frac[18,91]
   + frac[20,91] + frac[21,91] + frac[22,91] >= 1
 cov_129_92: frac[2,92] + frac[9,92] + frac[16,92] + frac[18,92]
   + frac[20,92] + frac[21,92] + frac[22,92] >= 1
 cov_130_48: frac[12,48] + frac[15,48] + frac[19,48] >= 1
 cov_130_49: frac[12,49] + frac[15,49] + frac[19,49] >= 1
 cov_130_50: frac[12,50] + frac[15,50] + frac[19,50] >= 1
 cov_131_18: frac[1,18] + frac[9,18] + frac[16,18] + frac[18,18]
   + frac[20,18] + frac[21,18] + frac[22,18] >= 0.7
 cov_131_19: frac[1,19] + frac[9,19] + frac[16,19] + frac[18,19]
   + frac[20,19] + frac[21,19] + frac[22,19] >= 0.7
 cov_131_20: frac[1,20] + frac[9,20] + frac[16,20] + frac[18,20]
   + frac[20,20] + frac[21,20] + frac[22,20] >= 0.7
 cov_132_30: frac[1,30] + frac[6,30] + frac[16,30] + frac[17,30]
   + frac[20,30] >= 0.8
 cov_132_31: frac[1,31] + frac[6,31] + frac[16,31] + frac[17,31]
   + frac[20,31] >= 0.8
 cov_132_32: frac[1,32] + frac[6,32] + frac[16,32] + frac[17,32]
   + frac[20,32] >= 0.8
 cov_133_135: frac[2,135] + frac[9,135] + frac[16,135] + frac[18,135]
   + frac[20,135] + frac[21,135] + frac[22,135] >= 1
 cov_133_136: frac[2,136] + frac[9,136] + frac[16,136] + frac[18,136]
   + frac[20,136] + frac[21,136] + frac[22,136] >= 1
 cov_133_137: frac[2,137] + frac[9,137] + frac[16,137] + frac[18,137]
   + frac[20,137] + frac[21,137] + frac[22,137] >= 1
 cov_134_174: frac[2,174] + frac[3,174] + frac[16,174] >= 0.9
 cov_134_175: frac[2,175] + frac[3,175] + frac[16,175] >= 0.9
 cov_134_176: frac[2,176] + frac[3,176] + frac[16,176] >= 0.9
 cov_135_144: frac[0,144] + frac[13,144] + frac[16,144] + frac[19,144]
   + frac[20,144] + frac[21,144] + frac[22,144] >= 0.8
 cov_135_145: frac[0,145] + frac[13,145] + frac[16,145] + frac[19,145]
   + frac[20,145] + frac[21,145] + frac[22,145] >= 0.8
 cov_135_146: frac[0,146] + frac[13,146] + frac[16,146] + frac[19,146]
   + frac[20,146] + frac[21,146] + frac[22,146] >= 0.8
 cov_136_123: frac[11,123] + frac[12,123] + frac[18,123] + frac[19,123]
   + frac[21,123] >= 0.7
 cov_136_124: frac[11,124] + frac[12,124] + frac[18,124] + frac[19,124]
   + frac[21,124] >= 0.7
 cov_136_125: frac[11,125] + frac[12,125] + frac[18,125] + frac[19,125]
   + frac[21,125] >= 0.7
 cov_137_168: frac[0,168] + frac[1,168] + frac[16,168] >= 0.9
 cov_137_169: frac[0,169] + frac[1,169] + frac[16,169] >= 0.9
 cov_137_170: frac[0,170] + frac[1,170] + frac[16,170] >= 0.9
 cov_138_165: frac[5,165] + frac[7,165] + frac[17,165] >= 0.7
 cov_138_166: frac[5,166] + frac[7,166] + frac[17,166] >= 0.7
 cov_138_167: frac[5,167] + frac[7,167] + frac[17,167] >= 0.7
 cov_139_111: frac[5,111] + frac[13,111] + frac[17,111] + frac[19,111]
   + frac[20,111] + frac[21,111] + frac[22,111] >= 0.9
 cov_139_112: frac[5,112] + frac[13,112] + frac[17,112] + frac[19,112]
   + frac[20,112] + frac[21,112] + frac[22,112] >= 0.9
 cov_139_113: frac[5,113] + frac[13,113] + frac[17,113] + frac[19,113]
   + frac[20,113] + frac[21,113] + frac[22,113] >= 0.9
 cov_140_33: frac[5,33] + frac[11,33] + frac[17,33] + frac[18,33]
   + frac[20,33] + frac[21,33] + frac[22,33] >= 1
 cov_140_34: frac[5,34] + frac[11,34] + frac[17,34] + frac[18,34]
   + frac[20,34] + frac[21,34] + frac[22,34] >= 1
 cov_140_35: frac[5,35] + frac[11,35] + frac[17,35] + frac[18,35]
   + frac[20,35] + frac[21,35] + frac[22,35] >= 1
 cov_141_18: frac[6,18] + frac[7,18] + frac[17,18] >= 0.8
 cov_141_19: frac[6,19] + frac[7,19] + frac[17,19] >= 0.8
 cov_141_20: frac[6,20] + frac[7,20] + frac[17,20] >= 0.8
 cov_142_105: frac[7,105] + frac[10,105] + frac[17,105] + frac[18,105]
   + frac[20,105] + frac[21,105] + frac[22,105] >= 1
 cov_142_106: frac[7,106] + frac[10,106] + frac[17,106] + frac[18,106]
   + frac[20,106] + frac[21,106] + frac[22,106] >= 1
 cov_142_107: frac[7,107] + frac[10,107] + frac[17,107] + frac[18,107]
   + frac[20,107] + frac[21,107] + frac[22,107] >= 1
 cov_143_57: frac[5,57] + frac[7,57] + frac[17,57] >= 0.8
 cov_143_58: frac[5,58] + frac[7,58] + frac[17,58] >= 0.8
 cov_143_59: frac[5,59] + frac[7,59] + frac[17,59] >= 0.8
 cov_144_27: frac[4,27] + frac[10,27] + frac[17,27] + frac[18,27]
   + frac[20,27] + frac[21,27] + frac[22,27] >= 0.8
 cov_144_28: frac[4,28] + frac[10,28] + frac[17,28] + frac[18,28]
   + frac[20,28] + frac[21,28] + frac[22,28] >= 0.8
 cov_144_29: frac[4,29] + frac[10,29] + frac[17,29] + frac[18,29]
   + frac[20,29] + frac[21,29] + frac[22,29] >= 0.8
 cov_145_96: frac[8,96] + frac[14,96] + frac[18,96] + frac[19,96]
   + frac[21,96] >= 1
 cov_145_97: frac[8,97] + frac[14,97] + frac[18,97] + frac[19,97]
   + frac[21,97] >= 1
 cov_145_98: frac[8,98] + frac[14,98] + frac[18,98] + frac[19,98]
   + frac[21,98] >= 1
 cov_146_24: frac[8,24] + frac[12,24] + frac[18,24] + frac[19,24]
   + frac[21,24] >= 1
 cov_146_25: frac[8,25] + frac[12,25] + frac[18,25] + frac[19,25]
   + frac[21,25] >= 1
 cov_146_26: frac[8,26] + frac[12,26] + frac[18,26] + frac[19,26]
   + frac[21,26] >= 1
 cov_147_24: frac[4,24] + frac[11,24] + frac[17,24] + frac[18,24]
   + frac[20,24] + frac[21,24] + frac[22,24] >= 0.8
 cov_147_25: frac[4,25] + frac[11,25] + frac[17,25] + frac[18,25]
   + frac[20,25] + frac[21,25] + frac[22,25] >= 0.8
 cov_147_26: frac[4,26] + frac[11,26] + frac[17,26] + frac[18,26]
   + frac[20,26] + frac[21,26] + frac[22,26] >= 0.8
 cov_148_102: frac[2,102] + frac[3,102] + frac[16,102] >= 1
 cov_148_103: frac[2,103] + frac[3,103] + frac[16,103] >= 1
 cov_148_104: frac[2,104] + frac[3,104] + frac[16,104] >= 1
 cov_149_147: frac[9,147] + frac[14,147] + frac[18,147] + frac[19,147]
   + frac[21,147] >= 0.7
 cov_149_148: frac[9,148] + frac[14,148] + frac[18,148] + frac[19,148]
   + frac[21,148] >= 0.7
 cov_149_149: frac[9,149] + frac[14,149] + frac[18,149] + frac[19,149]
   + frac[21,149] >= 0.7
 cov_150_177: frac[1,177] + frac[9,177] + frac[16,177] + frac[18,177]
   + frac[20,177] + frac[21,177] + frac[22,177] >= 1
 cov_150_178: frac[1,178] + frac[9,178] + frac[16,178] + frac[18,178]
   + frac[20,178] + frac[21,178] + frac[22,178] >= 1
 cov_150_179: frac[1,179] + frac[9,179] + frac[16,179] + frac[18,179]
   + frac[20,179] + frac[21,179] + frac[22,179] >= 1
 cov_151_51: frac[1,51] + frac[3,51] + frac[16,51] >= 1
 cov_151_52: frac[1,52] + frac[3,52] + frac[16,52] >= 1
 cov_151_53: frac[1,53] + frac[3,53] + frac[16,53] >= 1
 cov_152_123: frac[5,123] + frac[14,123] + frac[17,123] + frac[19,123]
   + frac[20,123] + frac[21,123] + frac[22,123] >= 0.8
 cov_152_124: frac[5,124] + frac[14,124] + frac[17,124] + frac[19,124]
   + frac[20,124] + frac[21,124] + frac[22,124] >= 0.8
 cov_152_125: frac[5,125] + frac[14,125] + frac[17,125] + frac[19,125]
   + frac[20,125] + frac[21,125] + frac[22,125] >= 0.8
 cov_153_141: frac[1,141] + frac[10,141] + frac[16,141] + frac[18,141]
   + frac[20,141] + frac[21,141] + frac[22,141] >= 0.9
 cov_153_142: frac[1,142] + frac[10,142] + frac[16,142] + frac[18,142]
   + frac[20,142] + frac[21,142] + frac[22,142] >= 0.9
 cov_153_143: frac[1,143] + frac[10,143] + frac[16,143] + frac[18,143]
   + frac[20,143] + frac[21,143] + frac[22,143] >= 0.9
 cov_154_177: frac[8,177] + frac[15,177] + frac[18,177] + frac[19,177]
   + frac[21,177] >= 0.7
 cov_154_178: frac[8,178] + frac[15,178] + frac[18,178] + frac[19,178]
   + frac[21,178] >= 0.7
 cov_154_179: frac[8,179] + frac[15,179] + frac[18,179] + frac[19,179]
   + frac[21,179] >= 0.7
 cov_155_180: frac[3,180] + frac[15,180] + frac[16,180] + frac[19,180]
   + frac[20,180] + frac[21,180] + frac[22,180] >= 0.8
 cov_155_181: frac[3,181] + frac[15,181] + frac[16,181] + frac[19,181]
   + frac[20,181] + frac[21,181] + frac[22,181] >= 0.8
 cov_155_182: frac[3,182] + frac[15,182] + frac[16,182] + frac[19,182]
   + frac[20,182] + frac[21,182] + frac[22,182] >= 0.8
 cov_156_42: frac[1,42] + frac[6,42] + frac[16,42] + frac[17,42]
   + frac[20,42] >= 1
 cov_156_43: frac[1,43] + frac[6,43] + frac[16,43] + frac[17,43]
   + frac[20,43] >= 1
 cov_156_44: frac[1,44] + frac[6,44] + frac[16,44] + frac[17,44]
   + frac[20,44] >= 1
 cov_157_18: frac[0,18] + frac[4,18] + frac[16,18] + frac[17,18]
   + frac[20,18] >= 0.7
 cov_157_19: frac[0,19] + frac[4,19] + frac[16,19] + frac[17,19]
   + frac[20,19] >= 0.7
 cov_157_20: frac[0,20] + frac[4,20] + frac[16,20] + frac[17,20]
   + frac[20,20] >= 0.7
 cov_158_96: frac[3,96] + frac[6,96] + frac[16,96] + frac[17,96]
   + frac[20,96] >= 0.8
 cov_158_97: frac[3,97] + frac[6,97] + frac[16,97] + frac[17,97]
   + frac[20,97] >= 0.8
 cov_158_98: frac[3,98] + frac[6,98] + frac[16,98] + frac[17,98]
   + frac[20,98] >= 0.8
 cov_159_96: frac[1,96] + frac[4,96] + frac[16,96] + frac[17,96]
   + frac[20,96] >= 0.9
 cov_159_97: frac[1,97] + frac[4,97] + frac[16,97] + frac[17,97]
   + frac[20,97] >= 0.9
 cov_159_98: frac[1,98] + frac[4,98] + frac[16,98] + frac[17,98]
   + frac[20,98] >= 0.9
 cov_160_153: frac[0,153] + frac[8,153] + frac[16,153] + frac[18,153]
   + frac[20,153] + frac[21,153] + frac[22,153] >= 0.7
 cov_160_154: frac[0,154] + frac[8,154] + frac[16,154] + frac[18,154]
   + frac[20,154] + frac[21,154] + frac[22,154] >= 0.7
 cov_160_155: frac[0,155] + frac[8,155] + frac[16,155] + frac[18,155]
   + frac[20,155] + frac[21,155] + frac[22,155] >= 0.7
 cov_161_6: frac[0,6] + frac[6,6] + frac[16,6] + frac[17,6] + frac[20,6]
   >= 0.7
 cov_161_7: frac[0,7] + frac[6,7] + frac[16,7] + frac[17,7] + frac[20,7]
   >= 0.7
 cov_161_8: frac[0,8] + frac[6,8] + frac[16,8] + frac[17,8] + frac[20,8]
   >= 0.7
 cov_162_156: frac[2,156] + frac[11,156] + frac[16,156] + frac[18,156]
   + frac[20,156] + frac[21,156] + frac[22,156] >= 0.7
 cov_162_157: frac[2,157] + frac[11,157] + frac[16,157] + frac[18,157]
   + frac[20,157] + frac[21,157] + frac[22,157] >= 0.7
 cov_162_158: frac[2,158] + frac[11,158] + frac[16,158] + frac[18,158]
   + frac[20,158] + frac[21,158] + frac[22,158] >= 0.7
 cov_163_72: frac[3,72] + frac[5,72] + frac[16,72] + frac[17,72]
   + frac[20,72] >= 0.9
 cov_163_73: frac[3,73] + frac[5,73] + frac[16,73] + frac[17,73]
   + frac[20,73] >= 0.9
 cov_163_74: frac[3,74] + frac[5,74] + frac[16,74] + frac[17,74]
   + frac[20,74] >= 0.9
 cov_164_180: frac[0,180] + frac[1,180] + frac[16,180] >= 0.8
 cov_164_181: frac[0,181] + frac[1,181] + frac[16,181] >= 0.8
 cov_164_182: frac[0,182] + frac[1,182] + frac[16,182] >= 0.8
 cov_165_174: frac[12,174] + frac[15,174] + frac[19,174] >= 0.9
 cov_165_175: frac[12,175] + frac[15,175] + frac[19,175] >= 0.9
 cov_165_176: frac[12,176] + frac[15,176] + frac[19,176] >= 0.9
 cov_166_150: frac[4,150] + frac[11,150] + frac[17,150] + frac[18,150]
   + frac[20,150] + frac[21,150] + frac[22,150] >= 1
 cov_166_151: frac[4,151] + frac[11,151] + frac[17,151] + frac[18,151]
   + frac[20,151] + frac[21,151] + frac[22,151] >= 1
 cov_166_152: frac[4,152] + frac[11,152] + frac[17,152] + frac[18,152]
   + frac[20,152] + frac[21,152] + frac[22,152] >= 1
 cov_167_48: frac[10,48] + frac[14,48] + frac[18,48] + frac[19,48]
   + frac[21,48] >= 1
 cov_167_49: frac[10,49] + frac[14,49] + frac[18,49] + frac[19,49]
   + frac[21,49] >= 1
 cov_167_50: frac[10,50] + frac[14,50] + frac[18,50] + frac[19,50]
   + frac[21,50] >= 1
 cov_168_78: frac[0,78] + frac[3,78] + frac[16,78] >= 0.9
 cov_168_79: frac[0,79] + frac[3,79] + frac[16,79] >= 0.9
 cov_168_80: frac[0,80] + frac[3,80] + frac[16,80] >= 0.9
 cov_169_87: frac[6,87] + frac[15,87] + frac[17,87] + frac[19,87]
   + frac[20,87] + frac[21,87] + frac[22,87] >= 0.9
 cov_169_88: frac[6,88] + frac[15,88] + frac[17,88] + frac[19,88]
   + frac[20,88] + frac[21,88] + frac[22,88] >= 0.9
 cov_169_89: frac[6,89] + frac[15,89] + frac[17,89] + frac[19,89]
   + frac[20,89] + frac[21,89] + frac[22,89] >= 0.9
 cov_170_54: frac[1,54] + frac[15,54] + frac[16,54] + frac[19,54]
   + frac[20,54] + frac[21,54] + frac[22,54] >= 0.8
 cov_170_55: frac[1,55] + frac[15,55] + frac[16,55] + frac[19,55]
   + frac[20,55] + frac[21,55] + frac[22,55] >= 0.8
 cov_170_56: frac[1,56] + frac[15,56] + frac[16,56] + frac[19,56]
   + frac[20,56] + frac[21,56] + frac[22,56] >= 0.8
 cov_171_168: frac[7,168] + frac[9,168] + frac[17,168] + frac[18,168]
   + frac[20,168] + frac[21,168] + frac[22,168] >= 0.7
 cov_171_169: frac[7,169] + frac[9,169] + frac[17,169] + frac[18,169]
   + frac[20,169] + frac[21,169] + frac[22,169] >= 0.7
 cov_171_170: frac[7,170] + frac[9,170] + frac[17,170] + frac[18,170]
   + frac[20,170] + frac[21,170] + frac[22,170] >= 0.7
 cov_172_87: frac[0,87] + frac[8,87] + frac[16,87] + frac[18,87]
   + frac[20,87] + frac[21,87] + frac[22,87] >= 0.7
 cov_172_88: frac[0,88] + frac[8,88] + frac[16,88] + frac[18,88]
   + frac[20,88] + frac[21,88] + frac[22,88] >= 0.7
 cov_172_89: frac[0,89] + frac[8,89] + frac[16,89] + frac[18,89]
   + frac[20,89] + frac[21,89] + frac[22,89] >= 0.7
 cov_173_81: frac[7,81] + frac[8,81] + frac[17,81] + frac[18,81]
   + frac[20,81] + frac[21,81] + frac[22,81] >= 0.8
 cov_173_82: frac[7,82] + frac[8,82] + frac[17,82] + frac[18,82]
   + frac[20,82] + frac[21,82] + frac[22,82] >= 0.8
 cov_173_83: frac[7,83] + frac[8,83] + frac[17,83] + frac[18,83]
   + frac[20,83] + frac[21,83] + frac[22,83] >= 0.8
 cov_174_111: frac[10,111] + frac[11,111] + frac[18,111] >= 1
 cov_174_112: frac[10,112] + frac[11,112] + frac[18,112] >= 1
 cov_174_113: frac[10,113] + frac[11,113] + frac[18,113] >= 1
 cov_175_69: frac[0,69] + frac[5,69] + frac[16,69] + frac[17,69]
   + frac[20,69] >= 0.8
 cov_175_70: frac[0,70] + frac[5,70] + frac[16,70] + frac[17,70]
   + frac[20,70] >= 0.8
 cov_175_71: frac[0,71] + frac[5,71] + frac[16,71] + frac[17,71]
   + frac[20,71] >= 0.8
 cov_176_15: frac[0,15] + frac[13,15] + frac[16,15] + frac[19,15]
   + frac[20,15] + frac[21,15] + frac[22,15] >= 0.8
 cov_176_16: frac[0,16] + frac[13,16] + frac[16,16] + frac[19,16]
   + frac[20,16] + frac[21,16] + frac[22,16] >= 0.8
 cov_176_17: frac[0,17] + frac[13,17] + frac[16,17] + frac[19,17]
   + frac[20,17] + frac[21,17] + frac[22,17] >= 0.8
 cov_177_159: frac[1,159] + frac[11,159] + frac[16,159] + frac[18,159]
   + frac[20,159] + frac[21,159] + frac[22,159] >= 0.7
 cov_177_160: frac[1,160] + frac[11,160] + frac[16,160] + frac[18,160]
   + frac[20,160] + frac[21,160] + frac[22,160] >= 0.7
 cov_177_161: frac[1,161] + frac[11,161] + frac[16,161] + frac[18,161]
   + frac[20,161] + frac[21,161] + frac[22,161] >= 0.7
 cov_178_36: frac[1,36] + frac[4,36] + frac[16,36] + frac[17,36]
   + frac[20,36] >= 1
 cov_178_37: frac[1,37] + frac[4,37] + frac[16,37] + frac[17,37]
   + frac[20,37] >= 1
 cov_178_38: frac[1,38] + frac[4,38] + frac[16,38] + frac[17,38]
   + frac[20,38] >= 1
 cov_179_138: frac[10,138] + frac[12,138] + frac[18,138] + frac[19,138]
   + frac[21,138] >= 0.7
 cov_179_139: frac[10,139] + frac[12,139] + frac[18,139] + frac[19,139]
   + frac[21,139] >= 0.7
 cov_179_140: frac[10,140] + frac[12,140] + frac[18,140] + frac[19,140]
   + frac[21,140] >= 0.7
 cov_180_93: frac[10,93] + frac[15,93] + frac[18,93] + frac[19,93]
   + frac[21,93] >= 0.8
 cov_180_94: frac[10,94] + frac[15,94] + frac[18,94] + frac[19,94]
   + frac[21,94] >= 0.8
 cov_180_95: frac[10,95] + frac[15,95] + frac[18,95] + frac[19,95]
   + frac[21,95] >= 0.8
 cov_181_39: frac[2,39] + frac[9,39] + frac[16,39] + frac[18,39]
   + frac[20,39] + frac[21,39] + frac[22,39] >= 0.9
 cov_181_40: frac[2,40] + frac[9,40] + frac[16,40] + frac[18,40]
   + frac[20,40] + frac[21,40] + frac[22,40] >= 0.9
 cov_181_41: frac[2,41] + frac[9,41] + frac[16,41] + frac[18,41]
   + frac[20,41] + frac[21,41] + frac[22,41] >= 0.9
 cov_182_51: frac[3,51] + frac[5,51] + frac[16,51] + frac[17,51]
   + frac[20,51] >= 1
 cov_182_52: frac[3,52] + frac[5,52] + frac[16,52] + frac[17,52]
   + frac[20,52] >= 1
 cov_182_53: frac[3,53] + frac[5,53] + frac[16,53] + frac[17,53]
   + frac[20,53] >= 1
 cov_183_120: frac[0,120] + frac[10,120] + frac[16,120] + frac[18,120]
   + frac[20,120] + frac[21,120] + frac[22,120] >= 0.8
 cov_183_121: frac[0,121] + frac[10,121] + frac[16,121] + frac[18,121]
   + frac[20,121] + frac[21,121] + frac[22,121] >= 0.8
 cov_183_122: frac[0,122] + frac[10,122] + frac[16,122] + frac[18,122]
   + frac[20,122] + frac[21,122] + frac[22,122] >= 0.8
 cov_184_189: frac[0,189] + frac[7,189] + frac[16,189] + frac[17,189]
   + frac[20,189] >= 1
 cov_184_190: frac[0,190] + frac[7,190] + frac[16,190] + frac[17,190]
   + frac[20,190] >= 1
 cov_184_191: frac[0,191] + frac[7,191] + frac[16,191] + frac[17,191]
   + frac[20,191] >= 1
 cov_185_162: frac[5,162] + frac[13,162] + frac[17,162] + frac[19,162]
   + frac[20,162] + frac[21,162] + frac[22,162] >= 0.8
 cov_185_163: frac[5,163] + frac[13,163] + frac[17,163] + frac[19,163]
   + frac[20,163] + frac[21,163] + frac[22,163] >= 0.8
 cov_185_164: frac[5,164] + frac[13,164] + frac[17,164] + frac[19,164]
   + frac[20,164] + frac[21,164] + frac[22,164] >= 0.8
 cov_186_30: frac[2,30] + frac[4,30] + frac[16,30] + frac[17,30]
   + frac[20,30] >= 0.8
 cov_186_31: frac[2,31] + frac[4,31] + frac[16,31] + frac[17,31]
   + frac[20,31] >= 0.8
 cov_186_32: frac[2,32] + frac[4,32] + frac[16,32] + frac[17,32]
   + frac[20,32] >= 0.8
 cov_187_174: frac[4,174] + frac[6,174] + frac[17,174] >= 0.8
 cov_187_175: frac[4,175] + frac[6,175] + frac[17,175] >= 0.8
 cov_187_176: frac[4,176] + frac[6,176] + frac[17,176] >= 0.8
 cov_188_27: frac[7,27] + frac[14,27] + frac[17,27] + frac[19,27]
   + frac[20,27] + frac[21,27] + frac[22,27] >= 0.9
 cov_188_28: frac[7,28] + frac[14,28] + frac[17,28] + frac[19,28]
   + frac[20,28] + frac[21,28] + frac[22,28] >= 0.9
 cov_188_29: frac[7,29] + frac[14,29] + frac[17,29] + frac[19,29]
   + frac[20,29] + frac[21,29] + frac[22,29] >= 0.9
 cov_189_15: frac[1,15] + frac[7,15] + frac[16,15] + frac[17,15]
   + frac[20,15] >= 0.9
 cov_189_16: frac[1,16] + frac[7,16] + frac[16,16] + frac[17,16]
   + frac[20,16] >= 0.9
 cov_189_17: frac[1,17] + frac[7,17] + frac[16,17] + frac[17,17]
   + frac[20,17] >= 0.9
 cov_190_90: frac[9,90] + frac[10,90] + frac[18,90] >= 0.7
 cov_190_91: frac[9,91] + frac[10,91] + frac[18,91] >= 0.7
 cov_190_92: frac[9,92] + frac[10,92] + frac[18,92] >= 0.7
 cov_191_27: frac[0,27] + frac[5,27] + frac[16,27] + frac[17,27]
   + frac[20,27] >= 1
 cov_191_28: frac[0,28] + frac[5,28] + frac[16,28] + frac[17,28]
   + frac[20,28] >= 1
 cov_191_29: frac[0,29] + frac[5,29] + frac[16,29] + frac[17,29]
   + frac[20,29] >= 1
 cov_192_15: frac[1,15] + frac[14,15] + frac[16,15] + frac[19,15]
   + frac[20,15] + frac[21,15] + frac[22,15] >= 0.7
 cov_192_16: frac[1,16] + frac[14,16] + frac[16,16] + frac[19,16]
   + frac[20,16] + frac[21,16] + frac[22,16] >= 0.7
 cov_192_17: frac[1,17] + frac[14,17] + frac[16,17] + frac[19,17]
   + frac[20,17] + frac[21,17] + frac[22,17] >= 0.7
 cov_193_174: frac[2,174] + frac[3,174] + frac[16,174] >= 0.8
 cov_193_175: frac[2,175] + frac[3,175] + frac[16,175] >= 0.8
 cov_193_176: frac[2,176] + frac[3,176] + frac[16,176] >= 0.8
 cov_194_36: frac[2,36] + frac[14,36] + frac[16,36] + frac[19,36]
   + frac[20,36] + frac[21,36] + frac[22,36] >= 1
 cov_194_37: frac[2,37] + frac[14,37] + frac[16,37] + frac[19,37]
   + frac[20,37] + frac[21,37] + frac[22,37] >= 1
 cov_194_38: frac[2,38] + frac[14,38] + frac[16,38] + frac[19,38]
   + frac[20,38] + frac[21,38] + frac[22,38] >= 1
 cov_195_102: frac[0,102] + frac[6,102] + frac[16,102] + frac[17,102]
   + frac[20,102] >= 0.8
 cov_195_103: frac[0,103] + frac[6,103] + frac[16,103] + frac[17,103]
   + frac[20,103] >= 0.8
 cov_195_104: frac[0,104] + frac[6,104] + frac[16,104] + frac[17,104]
   + frac[20,104] >= 0.8
 cov_196_111: frac[4,111] + frac[9,111] + frac[17,111] + frac[18,111]
   + frac[20,111] + frac[21,111] + frac[22,111] >= 0.8
 cov_196_112: frac[4,112] + frac[9,112] + frac[17,112] + frac[18,112]
   + frac[20,112] + frac[21,112] + frac[22,112] >= 0.8
 cov_196_113: frac[4,113] + frac[9,113] + frac[17,113] + frac[18,113]
   + frac[20,113] + frac[21,113] + frac[22,113] >= 0.8
 cov_197_54: frac[0,54] + frac[4,54] + frac[16,54] + frac[17,54]
   + frac[20,54] >= 1
 cov_197_55: frac[0,55] + frac[4,55] + frac[16,55] + frac[17,55]
   + frac[20,55] >= 1
 cov_197_56: frac[0,56] + frac[4,56] + frac[16,56] + frac[17,56]
   + frac[20,56] >= 1
 cov_198_57: frac[3,57] + frac[10,57] + frac[16,57] + frac[18,57]
   + frac[20,57] + frac[21,57] + frac[22,57] >= 0.9
 cov_198_58: frac[3,58] + frac[10,58] + frac[16,58] + frac[18,58]
   + frac[20,58] + frac[21,58] + frac[22,58] >= 0.9
 cov_198_59: frac[3,59] + frac[10,59] + frac[16,59] + frac[18,59]
   + frac[20,59] + frac[21,59] + frac[22,59] >= 0.9
 cov_199_57: frac[2,57] + frac[10,57] + frac[16,57] + frac[18,57]
   + frac[20,57] + frac[21,57] + frac[22,57] >= 0.9
 cov_199_58: frac[2,58] + frac[10,58] + frac[16,58] + frac[18,58]
   + frac[20,58] + frac[21,58] + frac[22,58] >= 0.9
 cov_199_59: frac[2,59] + frac[10,59] + frac[16,59] + frac[18,59]
   + frac[20,59] + frac[21,59] + frac[22,59] >= 0.9
 cov_200_0: frac[4,0] + frac[6,0] + frac[17,0] >= 0.7
 cov_200_1: frac[4,1] + frac[6,1] + frac[17,1] >= 0.7
 cov_200_2: frac[4,2] + frac[6,2] + frac[17,2] >= 0.7
 cov_201_33: frac[7,33] + frac[8,33] + frac[17,33] + frac[18,33]
   + frac[20,33] + frac[21,33] + frac[22,33] >= 0.8
 cov_201_34: frac[7,34] + frac[8,34] + frac[17,34] + frac[18,34]
   + frac[20,34] + frac[21,34] + frac[22,34] >= 0.8
 cov_201_35: frac[7,35] + frac[8,35] + frac[17,35] + frac[18,35]
   + frac[20,35] + frac[21,35] + frac[22,35] >= 0.8
 cov_202_60: frac[1,60] + frac[4,60] + frac[16,60] + frac[17,60]
   + frac[20,60] >= 0.7
 cov_202_61: frac[1,61] + frac[4,61] + frac[16,61] + frac[17,61]
   + frac[20,61] >= 0.7
 cov_202_62: frac[1,62] + frac[4,62] + frac[16,62] + frac[17,62]
   + frac[20,62] >= 0.7
 cov_203_6: frac[1,6] + frac[14,6] + frac[16,6] + frac[19,6] + frac[20,6]
   + frac[21,6] + frac[22,6] >= 0.8
 cov_203_7: frac[1,7] + frac[14,7] + frac[16,7] + frac[19,7] + frac[20,7]
   + frac[21,7] + frac[22,7] >= 0.8
 cov_203_8: frac[1,8] + frac[14,8] + frac[16,8] + frac[19,8] + frac[20,8]
   + frac[21,8] + frac[22,8] >= 0.8
 cov_204_174: frac[13,174] + frac[14,174] + frac[19,174] >= 0.9
 cov_204_175: frac[13,175] + frac[14,175] + frac[19,175] >= 0.9
 cov_204_176: frac[13,176] + frac[14,176] + frac[19,176] >= 0.9
 cov_205_72: frac[2,72] + frac[5,72] + frac[16,72] + frac[17,72]
   + frac[20,72] >= 0.7
 cov_205_73: frac[2,73] + frac[5,73] + frac[16,73] + frac[17,73]
   + frac[20,73] >= 0.7
 cov_205_74: frac[2,74] + frac[5,74] + frac[16,74] + frac[17,74]
   + frac[20,74] >= 0.7
 cov_206_144: frac[10,144] + frac[11,144] + frac[18,144] >= 0.7
 cov_206_145: frac[10,145] + frac[11,145] + frac[18,145] >= 0.7
 cov_206_146: frac[10,146] + frac[11,146] + frac[18,146] >= 0.7
 cov_207_42: frac[5,42] + frac[13,42] + frac[17,42] + frac[19,42]
   + frac[20,42] + frac[21,42] + frac[22,42] >= 0.9
 cov_207_43: frac[5,43] + frac[13,43] + frac[17,43] + frac[19,43]
   + frac[20,43] + frac[21,43] + frac[22,43] >= 0.9
 cov_207_44: frac[5,44] + frac[13,44] + frac[17,44] + frac[19,44]
   + frac[20,44] + frac[21,44] + frac[22,44] >= 0.9
 cov_208_126: frac[9,126] + frac[13,126] + frac[18,126] + frac[19,126]
   + frac[21,126] >= 1
 cov_208_127: frac[9,127] + frac[13,127] + frac[18,127] + frac[19,127]
   + frac[21,127] >= 1
 cov_208_128: frac[9,128] + frac[13,128] + frac[18,128] + frac[19,128]
   + frac[21,128] >= 1
 cov_209_24: frac[2,24] + frac[5,24] + frac[16,24] + frac[17,24]
   + frac[20,24] >= 0.9
 cov_209_25: frac[2,25] + frac[5,25] + frac[16,25] + frac[17,25]
   + frac[20,25] >= 0.9
 cov_209_26: frac[2,26] + frac[5,26] + frac[16,26] + frac[17,26]
   + frac[20,26] >= 0.9
 cov_210_114: frac[4,114] + frac[12,114] + frac[17,114] + frac[19,114]
   + frac[20,114] + frac[21,114] + frac[22,114] >= 1
 cov_210_115: frac[4,115] + frac[12,115] + frac[17,115] + frac[19,115]
   + frac[20,115] + frac[21,115] + frac[22,115] >= 1
 cov_210_116: frac[4,116] + frac[12,116] + frac[17,116] + frac[19,116]
   + frac[20,116] + frac[21,116] + frac[22,116] >= 1
 cov_211_165: frac[2,165] + frac[6,165] + frac[16,165] + frac[17,165]
   + frac[20,165] >= 0.7
 cov_211_166: frac[2,166] + frac[6,166] + frac[16,166] + frac[17,166]
   + frac[20,166] >= 0.7
 cov_211_167: frac[2,167] + frac[6,167] + frac[16,167] + frac[17,167]
   + frac[20,167] >= 0.7
 cov_212_189: frac[7,189] + frac[14,189] + frac[17,189] + frac[19,189]
   + frac[20,189] + frac[21,189] + frac[22,189] >= 0.8
 cov_212_190: frac[7,190] + frac[14,190] + frac[17,190] + frac[19,190]
   + frac[20,190] + frac[21,190] + frac[22,190] >= 0.8
 cov_212_191: frac[7,191] + frac[14,191] + frac[17,191] + frac[19,191]
   + frac[20,191] + frac[21,191] + frac[22,191] >= 0.8
 cov_213_18: frac[6,18] + frac[9,18] + frac[17,18] + frac[18,18]
   + frac[20,18] + frac[21,18] + frac[22,18] >= 1
 cov_213_19: frac[6,19] + frac[9,19] + frac[17,19] + frac[18,19]
   + frac[20,19] + frac[21,19] + frac[22,19] >= 1
 cov_213_20: frac[6,20] + frac[9,20] + frac[17,20] + frac[18,20]
   + frac[20,20] + frac[21,20] + frac[22,20] >= 1
 cov_214_42: frac[13,42] + frac[15,42] + frac[19,42] >= 0.7
 cov_214_43: frac[13,43] + frac[15,43] + frac[19,43] >= 0.7
 cov_214_44: frac[13,44] + frac[15,44] + frac[19,44] >= 0.7
 cov_215_15: frac[3,15] + frac[15,15] + frac[16,15] + frac[19,15]
   + frac[20,15] + frac[21,15] + frac[22,15] >= 0.9
 cov_215_16: frac[3,16] + frac[15,16] + frac[16,16] + frac[19,16]
   + frac[20,16] + frac[21,16] + frac[22,16] >= 0.9
 cov_215_17: frac[3,17] + frac[15,17] + frac[16,17] + frac[19,17]
   + frac[20,17] + frac[21,17] + frac[22,17] >= 0.9
 cov_216_159: frac[5,159] + frac[11,159] + frac[17,159] + frac[18,159]
   + frac[20,159] + frac[21,159] + frac[22,159] >= 0.8
 cov_216_160: frac[5,160] + frac[11,160] + frac[17,160] + frac[18,160]
   + frac[20,160] + frac[21,160] + frac[22,160] >= 0.8
 cov_216_161: frac[5,161] + frac[11,161] + frac[17,161] + frac[18,161]
   + frac[20,161] + frac[21,161] + frac[22,161] >= 0.8
 cov_217_93: frac[5,93] + frac[11,93] + frac[17,93] + frac[18,93]
   + frac[20,93] + frac[21,93] + frac[22,93] >= 0.8
 cov_217_94: frac[5,94] + frac[11,94] + frac[17,94] + frac[18,94]
   + frac[20,94] + frac[21,94] + frac[22,94] >= 0.8
 cov_217_95: frac[5,95] + frac[11,95] + frac[17,95] + frac[18,95]
   + frac[20,95] + frac[21,95] + frac[22,95] >= 0.8
 cov_218_147: frac[7,147] + frac[13,147] + frac[17,147] + frac[19,147]
   + frac[20,147] + frac[21,147] + frac[22,147] >= 0.7
 cov_218_148: frac[7,148] + frac[13,148] + frac[17,148] + frac[19,148]
   + frac[20,148] + frac[21,148] + frac[22,148] >= 0.7
 cov_218_149: frac[7,149] + frac[13,149] + frac[17,149] + frac[19,149]
   + frac[20,149] + frac[21,149] + frac[22,149] >= 0.7
 cov_219_159: frac[4,159] + frac[13,159] + frac[17,159] + frac[19,159]
   + frac[20,159] + frac[21,159] + frac[22,159] >= 0.8
 cov_219_160: frac[4,160] + frac[13,160] + frac[17,160] + frac[19,160]
   + frac[20,160] + frac[21,160] + frac[22,160] >= 0.8
 cov_219_161: frac[4,161] + frac[13,161] + frac[17,161] + frac[19,161]
   + frac[20,161] + frac[21,161] + frac[22,161] >= 0.8
 cov_220_102: frac[1,102] + frac[6,102] + frac[16,102] + frac[17,102]
   + frac[20,102] >= 0.8
 cov_220_103: frac[1,103] + frac[6,103] + frac[16,103] + frac[17,103]
   + frac[20,103] >= 0.8
 cov_220_104: frac[1,104] + frac[6,104] + frac[16,104] + frac[17,104]
   + frac[20,104] >= 0.8
 cov_221_117: frac[3,117] + frac[7,117] + frac[16,117] + frac[17,117]
   + frac[20,117] >= 0.8
 cov_221_118: frac[3,118] + frac[7,118] + frac[16,118] + frac[17,118]
   + frac[20,118] >= 0.8
 cov_221_119: frac[3,119] + frac[7,119] + frac[16,119] + frac[17,119]
   + frac[20,119] >= 0.8
 cov_222_87: frac[3,87] + frac[7,87] + frac[16,87] + frac[17,87]
   + frac[20,87] >= 0.8
 cov_222_88: frac[3,88] + frac[7,88] + frac[16,88] + frac[17,88]
   + frac[20,88] >= 0.8
 cov_222_89: frac[3,89] + frac[7,89] + frac[16,89] + frac[17,89]
   + frac[20,89] >= 0.8
 cov_223_99: frac[3,99] + frac[14,99] + frac[16,99] + frac[19,99]
   + frac[20,99] + frac[21,99] + frac[22,99] >= 0.7
 cov_223_100: frac[3,100] + frac[14,100] + frac[16,100] + frac[19,100]
   + frac[20,100] + frac[21,100] + frac[22,100] >= 0.7
 cov_223_101: frac[3,101] + frac[14,101] + frac[16,101] + frac[19,101]
   + frac[20,101] + frac[21,101] + frac[22,101] >= 0.7
 cov_224_114: frac[9,114] + frac[10,114] + frac[18,114] >= 1
 cov_224_115: frac[9,115] + frac[10,115] + frac[18,115] >= 1
 cov_224_116: frac[9,116] + frac[10,116] + frac[18,116] >= 1
 cov_225_135: frac[12,135] + frac[13,135] + frac[19,135] >= 1
 cov_225_136: frac[12,136] + frac[13,136] + frac[19,136] >= 1
 cov_225_137: frac[12,137] + frac[13,137] + frac[19,137] >= 1
 cov_226_189: frac[2,189] + frac[3,189] + frac[16,189] >= 1
 cov_226_190: frac[2,190] + frac[3,190] + frac[16,190] >= 1
 cov_226_191: frac[2,191] + frac[3,191] + frac[16,191] >= 1
 cov_227_24: frac[13,24] + frac[15,24] + frac[19,24] >= 0.9
 cov_227_25: frac[13,25] + frac[15,25] + frac[19,25] >= 0.9
 cov_227_26: frac[13,26] + frac[15,26] + frac[19,26] >= 0.9
 cov_228_150: frac[7,150] + frac[13,150] + frac[17,150] + frac[19,150]
   + frac[20,150] + frac[21,150] + frac[22,150] >= 0.7
 cov_228_151: frac[7,151] + frac[13,151] + frac[17,151] + frac[19,151]
   + frac[20,151] + frac[21,151] + frac[22,151] >= 0.7
 cov_228_152: frac[7,152] + frac[13,152] + frac[17,152] + frac[19,152]
   + frac[20,152] + frac[21,152] + frac[22,152] >= 0.7
 cov_229_99: frac[0,99] + frac[3,99] + frac[16,99] >= 0.8
 cov_229_100: frac[0,100] + frac[3,100] + frac[16,100] >= 0.8
 cov_229_101: frac[0,101] + frac[3,101] + frac[16,101] >= 0.8
 cov_230_108: frac[3,108] + frac[13,108] + frac[16,108] + frac[19,108]
   + frac[20,108] + frac[21,108] + frac[22,108] >= 0.7
 cov_230_109: frac[3,109] + frac[13,109] + frac[16,109] + frac[19,109]
   + frac[20,109] + frac[21,109] + frac[22,109] >= 0.7
 cov_230_110: frac[3,110] + frac[13,110] + frac[16,110] + frac[19,110]
   + frac[20,110] + frac[21,110] + frac[22,110] >= 0.7
 cov_231_84: frac[4,84] + frac[12,84] + frac[17,84] + frac[19,84]
   + frac[20,84] + frac[21,84] + frac[22,84] >= 1
 cov_231_85: frac[4,85] + frac[12,85] + frac[17,85] + frac[19,85]
   + frac[20,85] + frac[21,85] + frac[22,85] >= 1
 cov_231_86: frac[4,86] + frac[12,86] + frac[17,86] + frac[19,86]
   + frac[20,86] + frac[21,86] + frac[22,86] >= 1
 cov_232_183: frac[8,183] + frac[11,183] + frac[18,183] >= 1
 cov_232_184: frac[8,184] + frac[11,184] + frac[18,184] >= 1
 cov_232_185: frac[8,185] + frac[11,185] + frac[18,185] >= 1
 cov_233_114: frac[5,114] + frac[6,114] + frac[17,114] >= 0.9
 cov_233_115: frac[5,115] + frac[6,115] + frac[17,115] >= 0.9
 cov_233_116: frac[5,116] + frac[6,116] + frac[17,116] >= 0.9
 cov_234_6: frac[11,6] + frac[12,6] + frac[18,6] + frac[19,6] + frac[21,6]
   >= 0.9
 cov_234_7: frac[11,7] + frac[12,7] + frac[18,7] + frac[19,7] + frac[21,7]
   >= 0.9
 cov_234_8: frac[11,8] + frac[12,8] + frac[18,8] + frac[19,8] + frac[21,8]
   >= 0.9
 cov_235_156: frac[6,156] + frac[12,156] + frac[17,156] + frac[19,156]
   + frac[20,156] + frac[21,156] + frac[22,156] >= 1
 cov_235_157: frac[6,157] + frac[12,157] + frac[17,157] + frac[19,157]
   + frac[20,157] + frac[21,157] + frac[22,157] >= 1
 cov_235_158: frac[6,158] + frac[12,158] + frac[17,158] + frac[19,158]
   + frac[20,158] + frac[21,158] + frac[22,158] >= 1
 cov_236_171: frac[11,171] + frac[15,171] + frac[18,171] + frac[19,171]
   + frac[21,171] >= 0.7
 cov_236_172: frac[11,172] + frac[15,172] + frac[18,172] + frac[19,172]
   + frac[21,172] >= 0.7
 cov_236_173: frac[11,173] + frac[15,173] + frac[18,173] + frac[19,173]
   + frac[21,173] >= 0.7
 cov_237_174: frac[9,174] + frac[13,174] + frac[18,174] + frac[19,174]
   + frac[21,174] >= 0.7
 cov_237_175: frac[9,175] + frac[13,175] + frac[18,175] + frac[19,175]
   + frac[21,175] >= 0.7
 cov_237_176: frac[9,176] + frac[13,176] + frac[18,176] + frac[19,176]
   + frac[21,176] >= 0.7
 cov_238_162: frac[0,162] + frac[13,162] + frac[16,162] + frac[19,162]
   + frac[20,162] + frac[21,162] + frac[22,162] >= 0.9
 cov_238_163: frac[0,163] + frac[13,163] + frac[16,163] + frac[19,163]
   + frac[20,163] + frac[21,163] + frac[22,163] >= 0.9
 cov_238_164: frac[0,164] + frac[13,164] + frac[16,164] + frac[19,164]
   + frac[20,164] + frac[21,164] + frac[22,164] >= 0.9
 cov_239_159: frac[0,159] + frac[12,159] + frac[16,159] + frac[19,159]
   + frac[20,159] + frac[21,159] + frac[22,159] >= 1
 cov_239_160: frac[0,160] + frac[12,160] + frac[16,160] + frac[19,160]
   + frac[20,160] + frac[21,160] + frac[22,160] >= 1
 cov_239_161: frac[0,161] + frac[12,161] + frac[16,161] + frac[19,161]
   + frac[20,161] + frac[21,161] + frac[22,161] >= 1
 cov_240_45: frac[7,45] + frac[11,45] + frac[17,45] + frac[18,45]
   + frac[20,45] + frac[21,45] + frac[22,45] >= 0.8
 cov_240_46: frac[7,46] + frac[11,46] + frac[17,46] + frac[18,46]
   + frac[20,46] + frac[21,46] + frac[22,46] >= 0.8
 cov_240_47: frac[7,47] + frac[11,47] + frac[17,47] + frac[18,47]
   + frac[20,47] + frac[21,47] + frac[22,47] >= 0.8
 cov_241_153: frac[7,153] + frac[8,153] + frac[17,153] + frac[18,153]
   + frac[20,153] + frac[21,153] + frac[22,153] >= 0.9
 cov_241_154: frac[7,154] + frac[8,154] + frac[17,154] + frac[18,154]
   + frac[20,154] + frac[21,154] + frac[22,154] >= 0.9
 cov_241_155: frac[7,155] + frac[8,155] + frac[17,155] + frac[18,155]
   + frac[20,155] + frac[21,155] + frac[22,155] >= 0.9
 cov_242_129: frac[11,129] + frac[14,129] + frac[18,129] + frac[19,129]
   + frac[21,129] >= 0.9
 cov_242_130: frac[11,130] + frac[14,130] + frac[18,130] + frac[19,130]
   + frac[21,130] >= 0.9
 cov_242_131: frac[11,131] + frac[14,131] + frac[18,131] + frac[19,131]
   + frac[21,131] >= 0.9
 cov_243_24: frac[2,24] + frac[12,24] + frac[16,24] + frac[19,24]
   + frac[20,24] + frac[21,24] + frac[22,24] >= 1
 cov_243_25: frac[2,25] + frac[12,25] + frac[16,25] + frac[19,25]
   + frac[20,25] + frac[21,25] + frac[22,25] >= 1
 cov_243_26: frac[2,26] + frac[12,26] + frac[16,26] + frac[19,26]
   + frac[20,26] + frac[21,26] + frac[22,26] >= 1
 cov_244_162: frac[6,162] + frac[11,162] + frac[17,162] + frac[18,162]
   + frac[20,162] + frac[21,162] + frac[22,162] >= 0.8
 cov_244_163: frac[6,163] + frac[11,163] + frac[17,163] + frac[18,163]
   + frac[20,163] + frac[21,163] + frac[22,163] >= 0.8
 cov_244_164: frac[6,164] + frac[11,164] + frac[17,164] + frac[18,164]
   + frac[20,164] + frac[21,164] + frac[22,164] >= 0.8
 cov_245_99: frac[4,99] + frac[13,99] + frac[17,99] + frac[19,99]
   + frac[20,99] + frac[21,99] + frac[22,99] >= 1
 cov_245_100: frac[4,100] + frac[13,100] + frac[17,100] + frac[19,100]
   + frac[20,100] + frac[21,100] + frac[22,100] >= 1
 cov_245_101: frac[4,101] + frac[13,101] + frac[17,101] + frac[19,101]
   + frac[20,101] + frac[21,101] + frac[22,101] >= 1
 cov_246_108: frac[4,108] + frac[10,108] + frac[17,108] + frac[18,108]
   + frac[20,108] + frac[21,108] + frac[22,108] >= 0.8
 cov_246_109: frac[4,109] + frac[10,109] + frac[17,109] + frac[18,109]
   + frac[20,109] + frac[21,109] + frac[22,109] >= 0.8
 cov_246_110: frac[4,110] + frac[10,110] + frac[17,110] + frac[18,110]
   + frac[20,110] + frac[21,110] + frac[22,110] >= 0.8
 cov_247_66: frac[10,66] + frac[15,66] + frac[18,66] + frac[19,66]
   + frac[21,66] >= 0.7
 cov_247_67: frac[10,67] + frac[15,67] + frac[18,67] + frac[19,67]
   + frac[21,67] >= 0.7
 cov_247_68: frac[10,68] + frac[15,68] + frac[18,68] + frac[19,68]
   + frac[21,68] >= 0.7
 cov_248_129: frac[10,129] + frac[13,129] + frac[18,129] + frac[19,129]
   + frac[21,129] >= 0.9
 cov_248_130: frac[10,130] + frac[13,130] + frac[18,130] + frac[19,130]
   + frac[21,130] >= 0.9
 cov_248_131: frac[10,131] + frac[13,131] + frac[18,131] + frac[19,131]
   + frac[21,131] >= 0.9
 cov_249_141: frac[1,141] + frac[7,141] + frac[16,141] + frac[17,141]
   + frac[20,141] >= 0.7
 cov_249_142: frac[1,142] + frac[7,142] + frac[16,142] + frac[17,142]
   + frac[20,142] >= 0.7
 cov_249_143: frac[1,143] + frac[7,143] + frac[16,143] + frac[17,143]
   + frac[20,143] >= 0.7
 cov_250_150: frac[4,150] + frac[6,150] + frac[17,150] >= 0.8
 cov_250_151: frac[4,151] + frac[6,151] + frac[17,151] >= 0.8
 cov_250_152: frac[4,152] + frac[6,152] + frac[17,152] >= 0.8
 cov_251_42: frac[8,42] + frac[13,42] + frac[18,42] + frac[19,42]
   + frac[21,42] >= 0.9
 cov_251_43: frac[8,43] + frac[13,43] + frac[18,43] + frac[19,43]
   + frac[21,43] >= 0.9
 cov_251_44: frac[8,44] + frac[13,44] + frac[18,44] + frac[19,44]
   + frac[21,44] >= 0.9
 cov_252_42: frac[12,42] + frac[14,42] + frac[19,42] >= 1
 cov_252_43: frac[12,43] + frac[14,43] + frac[19,43] >= 1
 cov_252_44: frac[12,44] + frac[14,44] + frac[19,44] >= 1
 cov_253_51: frac[3,51] + frac[9,51] + frac[16,51] + frac[18,51]
   + frac[20,51] + frac[21,51] + frac[22,51] >= 0.8
 cov_253_52: frac[3,52] + frac[9,52] + frac[16,52] + frac[18,52]
   + frac[20,52] + frac[21,52] + frac[22,52] >= 0.8
 cov_253_53: frac[3,53] + frac[9,53] + frac[16,53] + frac[18,53]
   + frac[20,53] + frac[21,53] + frac[22,53] >= 0.8
 cov_254_132: frac[0,132] + frac[2,132] + frac[16,132] >= 0.7
 cov_254_133: frac[0,133] + frac[2,133] + frac[16,133] >= 0.7
 cov_254_134: frac[0,134] + frac[2,134] + frac[16,134] >= 0.7
 cov_255_156: frac[14,156] + frac[15,156] + frac[19,156] >= 1
 cov_255_157: frac[14,157] + frac[15,157] + frac[19,157] >= 1
 cov_255_158: frac[14,158] + frac[15,158] + frac[19,158] >= 1
 cov_256_153: frac[12,153] + frac[13,153] + frac[19,153] >= 0.8
 cov_256_154: frac[12,154] + frac[13,154] + frac[19,154] >= 0.8
 cov_256_155: frac[12,155] + frac[13,155] + frac[19,155] >= 0.8
 cov_257_156: frac[3,156] + frac[15,156] + frac[16,156] + frac[19,156]
   + frac[20,156] + frac[21,156] + frac[22,156] >= 0.9
 cov_257_157: frac[3,157] + frac[15,157] + frac[16,157] + frac[19,157]
   + frac[20,157] + frac[21,157] + frac[22,157] >= 0.9
 cov_257_158: frac[3,158] + frac[15,158] + frac[16,158] + frac[19,158]
   + frac[20,158] + frac[21,158] + frac[22,158] >= 0.9
 cov_258_63: frac[8,63] + frac[11,63] + frac[18,63] >= 0.7
 cov_258_64: frac[8,64] + frac[11,64] + frac[18,64] >= 0.7
 cov_258_65: frac[8,65] + frac[11,65] + frac[18,65] >= 0.7
 cov_259_144: frac[4,144] + frac[12,144] + frac[17,144] + frac[19,144]
   + frac[20,144] + frac[21,144] + frac[22,144] >= 0.9
 cov_259_145: frac[4,145] + frac[12,145] + frac[17,145] + frac[19,145]
   + frac[20,145] + frac[21,145] + frac[22,145] >= 0.9
 cov_259_146: frac[4,146] + frac[12,146] + frac[17,146] + frac[19,146]
   + frac[20,146] + frac[21,146] + frac[22,146] >= 0.9
 cov_260_24: frac[4,24] + frac[15,24] + frac[17,24] + frac[19,24]
   + frac[20,24] + frac[21,24] + frac[22,24] >= 1
 cov_260_25: frac[4,25] + frac[15,25] + frac[17,25] + frac[19,25]
   + frac[20,25] + frac[21,25] + frac[22,25] >= 1
 cov_260_26: frac[4,26] + frac[15,26] + frac[17,26] + frac[19,26]
   + frac[20,26] + frac[21,26] + frac[22,26] >= 1
 cov_261_18: frac[8,18] + frac[10,18] + frac[18,18] >= 0.9
 cov_261_19: frac[8,19] + frac[10,19] + frac[18,19] >= 0.9
 cov_261_20: frac[8,20] + frac[10,20] + frac[18,20] >= 0.9
 cov_262_153: frac[4,153] + frac[13,153] + frac[17,153] + frac[19,153]
   + frac[20,153] + frac[21,153] + frac[22,153] >= 0.7
 cov_262_154: frac[4,154] + frac[13,154] + frac[17,154] + frac[19,154]
   + frac[20,154] + frac[21,154] + frac[22,154] >= 0.7
 cov_262_155: frac[4,155] + frac[13,155] + frac[17,155] + frac[19,155]
   + frac[20,155] + frac[21,155] + frac[22,155] >= 0.7
 cov_263_138: frac[0,138] + frac[14,138] + frac[16,138] + frac[19,138]
   + frac[20,138] + frac[21,138] + frac[22,138] >= 0.7
 cov_263_139: frac[0,139] + frac[14,139] + frac[16,139] + frac[19,139]
   + frac[20,139] + frac[21,139] + frac[22,139] >= 0.7
 cov_263_140: frac[0,140] + frac[14,140] + frac[16,140] + frac[19,140]
   + frac[20,140] + frac[21,140] + frac[22,140] >= 0.7
 cov_264_102: frac[3,102] + frac[14,102] + frac[16,102] + frac[19,102]
   + frac[20,102] + frac[21,102] + frac[22,102] >= 1
 cov_264_103: frac[3,103] + frac[14,103] + frac[16,103] + frac[19,103]
   + frac[20,103] + frac[21,103] + frac[22,103] >= 1
 cov_264_104: frac[3,104] + frac[14,104] + frac[16,104] + frac[19,104]
   + frac[20,104] + frac[21,104] + frac[22,104] >= 1
 cov_265_42: frac[3,42] + frac[7,42] + frac[16,42] + frac[17,42]
   + frac[20,42] >= 0.7
 cov_265_43: frac[3,43] + frac[7,43] + frac[16,43] + frac[17,43]
   + frac[20,43] >= 0.7
 cov_265_44: frac[3,44] + frac[7,44] + frac[16,44] + frac[17,44]
   + frac[20,44] >= 0.7
 cov_266_51: frac[1,51] + frac[13,51] + frac[16,51] + frac[19,51]
   + frac[20,51] + frac[21,51] + frac[22,51] >= 1
 cov_266_52: frac[1,52] + frac[13,52] + frac[16,52] + frac[19,52]
   + frac[20,52] + frac[21,52] + frac[22,52] >= 1
 cov_266_53: frac[1,53] + frac[13,53] + frac[16,53] + frac[19,53]
   + frac[20,53] + frac[21,53] + frac[22,53] >= 1
 cov_267_9: frac[2,9] + frac[10,9] + frac[16,9] + frac[18,9] + frac[20,9]
   + frac[21,9] + frac[22,9] >= 1
 cov_267_10: frac[2,10] + frac[10,10] + frac[16,10] + frac[18,10]
   + frac[20,10] + frac[21,10] + frac[22,10] >= 1
 cov_267_11: frac[2,11] + frac[10,11] + frac[16,11] + frac[18,11]
   + frac[20,11] + frac[21,11] + frac[22,11] >= 1
 cov_268_6: frac[9,6] + frac[14,6] + frac[18,6] + frac[19,6] + frac[21,6]
   >= 1
 cov_268_7: frac[9,7] + frac[14,7] + frac[18,7] + frac[19,7] + frac[21,7]
   >= 1
 cov_268_8: frac[9,8] + frac[14,8] + frac[18,8] + frac[19,8] + frac[21,8]
   >= 1
 cov_269_171: frac[4,171] + frac[7,171] + frac[17,171] >= 0.8
 cov_269_172: frac[4,172] + frac[7,172] + frac[17,172] >= 0.8
 cov_269_173: frac[4,173] + frac[7,173] + frac[17,173] >= 0.8
 cov_270_78: frac[6,78] + frac[12,78] + frac[17,78] + frac[19,78]
   + frac[20,78] + frac[21,78] + frac[22,78] >= 0.9
 cov_270_79: frac[6,79] + frac[12,79] + frac[17,79] + frac[19,79]
   + frac[20,79] + frac[21,79] + frac[22,79] >= 0.9
 cov_270_80: frac[6,80] + frac[12,80] + frac[17,80] + frac[19,80]
   + frac[20,80] + frac[21,80] + frac[22,80] >= 0.9
 cov_271_99: frac[8,99] + frac[15,99] + frac[18,99] + frac[19,99]
   + frac[21,99] >= 0.9
 cov_271_100: frac[8,100] + frac[15,100] + frac[18,100] + frac[19,100]
   + frac[21,100] >= 0.9
 cov_271_101: frac[8,101] + frac[15,101] + frac[18,101] + frac[19,101]
   + frac[21,101] >= 0.9
 cov_272_63: frac[12,63] + frac[13,63] + frac[19,63] >= 1
 cov_272_64: frac[12,64] + frac[13,64] + frac[19,64] >= 1
 cov_272_65: frac[12,65] + frac[13,65] + frac[19,65] >= 1
 cov_273_36: frac[5,36] + frac[7,36] + frac[17,36] >= 0.7
 cov_273_37: frac[5,37] + frac[7,37] + frac[17,37] >= 0.7
 cov_273_38: frac[5,38] + frac[7,38] + frac[17,38] >= 0.7
 cov_274_129: frac[0,129] + frac[6,129] + frac[16,129] + frac[17,129]
   + frac[20,129] >= 0.9
 cov_274_130: frac[0,130] + frac[6,130] + frac[16,130] + frac[17,130]
   + frac[20,130] >= 0.9
 cov_274_131: frac[0,131] + frac[6,131] + frac[16,131] + frac[17,131]
   + frac[20,131] >= 0.9
 cov_275_132: frac[1,132] + frac[9,132] + frac[16,132] + frac[18,132]
   + frac[20,132] + frac[21,132] + frac[22,132] >= 0.8
 cov_275_133: frac[1,133] + frac[9,133] + frac[16,133] + frac[18,133]
   + frac[20,133] + frac[21,133] + frac[22,133] >= 0.8
 cov_275_134: frac[1,134] + frac[9,134] + frac[16,134] + frac[18,134]
   + frac[20,134] + frac[21,134] + frac[22,134] >= 0.8
 cov_276_69: frac[14,69] + frac[15,69] + frac[19,69] >= 0.9
 cov_276_70: frac[14,70] + frac[15,70] + frac[19,70] >= 0.9
 cov_276_71: frac[14,71] + frac[15,71] + frac[19,71] >= 0.9
 cov_277_162: frac[2,162] + frac[14,162] + frac[16,162] + frac[19,162]
   + frac[20,162] + frac[21,162] + frac[22,162] >= 0.8
 cov_277_163: frac[2,163] + frac[14,163] + frac[16,163] + frac[19,163]
   + frac[20,163] + frac[21,163] + frac[22,163] >= 0.8
 cov_277_164: frac[2,164] + frac[14,164] + frac[16,164] + frac[19,164]
   + frac[20,164] + frac[21,164] + frac[22,164] >= 0.8
 cov_278_168: frac[13,168] + frac[15,168] + frac[19,168] >= 1
 cov_278_169: frac[13,169] + frac[15,169] + frac[19,169] >= 1
 cov_278_170: frac[13,170] + frac[15,170] + frac[19,170] >= 1
 cov_279_147: frac[4,147] + frac[11,147] + frac[17,147] + frac[18,147]
   + frac[20,147] + frac[21,147] + frac[22,147] >= 0.8
 cov_279_148: frac[4,148] + frac[11,148] + frac[17,148] + frac[18,148]
   + frac[20,148] + frac[21,148] + frac[22,148] >= 0.8
 cov_279_149: frac[4,149] + frac[11,149] + frac[17,149] + frac[18,149]
   + frac[20,149] + frac[21,149] + frac[22,149] >= 0.8
 cov_280_186: frac[2,186] + frac[15,186] + frac[16,186] + frac[19,186]
   + frac[20,186] + frac[21,186] + frac[22,186] >= 0.7
 cov_280_187: frac[2,187] + frac[15,187] + frac[16,187] + frac[19,187]
   + frac[20,187] + frac[21,187] + frac[22,187] >= 0.7
 cov_280_188: frac[2,188] + frac[15,188] + frac[16,188] + frac[19,188]
   + frac[20,188] + frac[21,188] + frac[22,188] >= 0.7
 cov_281_30: frac[4,30] + frac[11,30] + frac[17,30] + frac[18,30]
   + frac[20,30] + frac[21,30] + frac[22,30] >= 1
 cov_281_31: frac[4,31] + frac[11,31] + frac[17,31] + frac[18,31]
   + frac[20,31] + frac[21,31] + frac[22,31] >= 1
 cov_281_32: frac[4,32] + frac[11,32] + frac[17,32] + frac[18,32]
   + frac[20,32] + frac[21,32] + frac[22,32] >= 1
 cov_282_99: frac[8,99] + frac[10,99] + frac[18,99] >= 0.7
 cov_282_100: frac[8,100] + frac[10,100] + frac[18,100] >= 0.7
 cov_282_101: frac[8,101] + frac[10,101] + frac[18,101] >= 0.7
 cov_283_153: frac[6,153] + frac[8,153] + frac[17,153] + frac[18,153]
   + frac[20,153] + frac[21,153] + frac[22,153] >= 1
 cov_283_154: frac[6,154] + frac[8,154] + frac[17,154] + frac[18,154]
   + frac[20,154] + frac[21,154] + frac[22,154] >= 1
 cov_283_155: frac[6,155] + frac[8,155] + frac[17,155] + frac[18,155]
   + frac[20,155] + frac[21,155] + frac[22,155] >= 1
 cov_284_129: frac[10,129] + frac[14,129] + frac[18,129] + frac[19,129]
   + frac[21,129] >= 0.9
 cov_284_130: frac[10,130] + frac[14,130] + frac[18,130] + frac[19,130]
   + frac[21,130] >= 0.9
 cov_284_131: frac[10,131] + frac[14,131] + frac[18,131] + frac[19,131]
   + frac[21,131] >= 0.9
 cov_285_156: frac[9,156] + frac[15,156] + frac[18,156] + frac[19,156]
   + frac[21,156] >= 0.8
 cov_285_157: frac[9,157] + frac[15,157] + frac[18,157] + frac[19,157]
   + frac[21,157] >= 0.8
 cov_285_158: frac[9,158] + frac[15,158] + frac[18,158] + frac[19,158]
   + frac[21,158] >= 0.8
 cov_286_120: frac[4,120] + frac[9,120] + frac[17,120] + frac[18,120]
   + frac[20,120] + frac[21,120] + frac[22,120] >= 0.9
 cov_286_121: frac[4,121] + frac[9,121] + frac[17,121] + frac[18,121]
   + frac[20,121] + frac[21,121] + frac[22,121] >= 0.9
 cov_286_122: frac[4,122] + frac[9,122] + frac[17,122] + frac[18,122]
   + frac[20,122] + frac[21,122] + frac[22,122] >= 0.9
 cov_287_6: frac[5,6] + frac[12,6] + frac[17,6] + frac[19,6] + frac[20,6]
   + frac[21,6] + frac[22,6] >= 0.7
 cov_287_7: frac[5,7] + frac[12,7] + frac[17,7] + frac[19,7] + frac[20,7]
   + frac[21,7] + frac[22,7] >= 0.7
 cov_287_8: frac[5,8] + frac[12,8] + frac[17,8] + frac[19,8] + frac[20,8]
   + frac[21,8] + frac[22,8] >= 0.7
 cov_288_81: frac[9,81] + frac[10,81] + frac[18,81] >= 0.7
 cov_288_82: frac[9,82] + frac[10,82] + frac[18,82] >= 0.7
 cov_288_83: frac[9,83] + frac[10,83] + frac[18,83] >= 0.7
 cov_289_45: frac[8,45] + frac[14,45] + frac[18,45] + frac[19,45]
   + frac[21,45] >= 0.7
 cov_289_46: frac[8,46] + frac[14,46] + frac[18,46] + frac[19,46]
   + frac[21,46] >= 0.7
 cov_289_47: frac[8,47] + frac[14,47] + frac[18,47] + frac[19,47]
   + frac[21,47] >= 0.7
 cov_290_54: frac[6,54] + frac[9,54] + frac[17,54] + frac[18,54]
   + frac[20,54] + frac[21,54] + frac[22,54] >= 0.8
 cov_290_55: frac[6,55] + frac[9,55] + frac[17,55] + frac[18,55]
   + frac[20,55] + frac[21,55] + frac[22,55] >= 0.8
 cov_290_56: frac[6,56] + frac[9,56] + frac[17,56] + frac[18,56]
   + frac[20,56] + frac[21,56] + frac[22,56] >= 0.8
 cov_291_48: frac[8,48] + frac[10,48] + frac[18,48] >= 1
 cov_291_49: frac[8,49] + frac[10,49] + frac[18,49] >= 1
 cov_291_50: frac[8,50] + frac[10,50] + frac[18,50] >= 1
 cov_292_177: frac[10,177] + frac[11,177] + frac[18,177] >= 0.7
 cov_292_178: frac[10,178] + frac[11,178] + frac[18,178] >= 0.7
 cov_292_179: frac[10,179] + frac[11,179] + frac[18,179] >= 0.7
 cov_293_3: frac[6,3] + frac[14,3] + frac[17,3] + frac[19,3] + frac[20,3]
   + frac[21,3] + frac[22,3] >= 0.7
 cov_293_4: frac[6,4] + frac[14,4] + frac[17,4] + frac[19,4] + frac[20,4]
   + frac[21,4] + frac[22,4] >= 0.7
 cov_293_5: frac[6,5] + frac[14,5] + frac[17,5] + frac[19,5] + frac[20,5]
   + frac[21,5] + frac[22,5] >= 0.7
 cov_294_36: frac[8,36] + frac[13,36] + frac[18,36] + frac[19,36]
   + frac[21,36] >= 0.8
 cov_294_37: frac[8,37] + frac[13,37] + frac[18,37] + frac[19,37]
   + frac[21,37] >= 0.8
 cov_294_38: frac[8,38] + frac[13,38] + frac[18,38] + frac[19,38]
   + frac[21,38] >= 0.8
 cov_295_9: frac[4,9] + frac[11,9] + frac[17,9] + frac[18,9] + frac[20,9]
   + frac[21,9] + frac[22,9] >= 1
 cov_295_10: frac[4,10] + frac[11,10] + frac[17,10] + frac[18,10]
   + frac[20,10] + frac[21,10] + frac[22,10] >= 1
 cov_295_11: frac[4,11] + frac[11,11] + frac[17,11] + frac[18,11]
   + frac[20,11] + frac[21,11] + frac[22,11] >= 1
 cov_296_117: frac[13,117] + frac[14,117] + frac[19,117] >= 0.9
 cov_296_118: frac[13,118] + frac[14,118] + frac[19,118] >= 0.9
 cov_296_119: frac[13,119] + frac[14,119] + frac[19,119] >= 0.9
 cov_297_24: frac[2,24] + frac[8,24] + frac[16,24] + frac[18,24]
   + frac[20,24] + frac[21,24] + frac[22,24] >= 0.9
 cov_297_25: frac[2,25] + frac[8,25] + frac[16,25] + frac[18,25]
   + frac[20,25] + frac[21,25] + frac[22,25] >= 0.9
 cov_297_26: frac[2,26] + frac[8,26] + frac[16,26] + frac[18,26]
   + frac[20,26] + frac[21,26] + frac[22,26] >= 0.9
 cov_298_174: frac[1,174] + frac[9,174] + frac[16,174] + frac[18,174]
   + frac[20,174] + frac[21,174] + frac[22,174] >= 0.8
 cov_298_175: frac[1,175] + frac[9,175] + frac[16,175] + frac[18,175]
   + frac[20,175] + frac[21,175] + frac[22,175] >= 0.8
 cov_298_176: frac[1,176] + frac[9,176] + frac[16,176] + frac[18,176]
   + frac[20,176] + frac[21,176] + frac[22,176] >= 0.8
 cov_299_180: frac[6,180] + frac[7,180] + frac[17,180] >= 0.8
 cov_299_181: frac[6,181] + frac[7,181] + frac[17,181] >= 0.8
 cov_299_182: frac[6,182] + frac[7,182] + frac[17,182] >= 0.8
 cov_300_78: frac[0,78] + frac[1,78] + frac[16,78] >= 0.8
 cov_300_79: frac[0,79] + frac[1,79] + frac[16,79] >= 0.8
 cov_300_80: frac[0,80] + frac[1,80] + frac[16,80] >= 0.8
 cov_301_159: frac[4,159] + frac[9,159] + frac[17,159] + frac[18,159]
   + frac[20,159] + frac[21,159] + frac[22,159] >= 0.9
 cov_301_160: frac[4,160] + frac[9,160] + frac[17,160] + frac[18,160]
   + frac[20,160] + frac[21,160] + frac[22,160] >= 0.9
 cov_301_161: frac[4,161] + frac[9,161] + frac[17,161] + frac[18,161]
   + frac[20,161] + frac[21,161] + frac[22,161] >= 0.9
 cov_302_114: frac[8,114] + frac[15,114] + frac[18,114] + frac[19,114]
   + frac[21,114] >= 1
 cov_302_115: frac[8,115] + frac[15,115] + frac[18,115] + frac[19,115]
   + frac[21,115] >= 1
 cov_302_116: frac[8,116] + frac[15,116] + frac[18,116] + frac[19,116]
   + frac[21,116] >= 1
 cov_303_87: frac[4,87] + frac[6,87] + frac[17,87] >= 1
 cov_303_88: frac[4,88] + frac[6,88] + frac[17,88] >= 1
 cov_303_89: frac[4,89] + frac[6,89] + frac[17,89] >= 1
 cov_304_189: frac[1,189] + frac[9,189] + frac[16,189] + frac[18,189]
   + frac[20,189] + frac[21,189] + frac[22,189] >= 0.7
 cov_304_190: frac[1,190] + frac[9,190] + frac[16,190] + frac[18,190]
   + frac[20,190] + frac[21,190] + frac[22,190] >= 0.7
 cov_304_191: frac[1,191] + frac[9,191] + frac[16,191] + frac[18,191]
   + frac[20,191] + frac[21,191] + frac[22,191] >= 0.7
 cov_305_102: frac[11,102] + frac[13,102] + frac[18,102] + frac[19,102]
   + frac[21,102] >= 0.7
 cov_305_103: frac[11,103] + frac[13,103] + frac[18,103] + frac[19,103]
   + frac[21,103] >= 0.7
 cov_305_104: frac[11,104] + frac[13,104] + frac[18,104] + frac[19,104]
   + frac[21,104] >= 0.7
 cov_306_81: frac[1,81] + frac[8,81] + frac[16,81] + frac[18,81]
   + frac[20,81] + frac[21,81] + frac[22,81] >= 0.8
 cov_306_82: frac[1,82] + frac[8,82] + frac[16,82] + frac[18,82]
   + frac[20,82] + frac[21,82] + frac[22,82] >= 0.8
 cov_306_83: frac[1,83] + frac[8,83] + frac[16,83] + frac[18,83]
   + frac[20,83] + frac[21,83] + frac[22,83] >= 0.8
 cov_307_45: frac[0,45] + frac[4,45] + frac[16,45] + frac[17,45]
   + frac[20,45] >= 1
 cov_307_46: frac[0,46] + frac[4,46] + frac[16,46] + frac[17,46]
   + frac[20,46] >= 1
 cov_307_47: frac[0,47] + frac[4,47] + frac[16,47] + frac[17,47]
   + frac[20,47] >= 1
 cov_308_183: frac[7,183] + frac[10,183] + frac[17,183] + frac[18,183]
   + frac[20,183] + frac[21,183] + frac[22,183] >= 0.9
 cov_308_184: frac[7,184] + frac[10,184] + frac[17,184] + frac[18,184]
   + frac[20,184] + frac[21,184] + frac[22,184] >= 0.9
 cov_308_185: frac[7,185] + frac[10,185] + frac[17,185] + frac[18,185]
   + frac[20,185] + frac[21,185] + frac[22,185] >= 0.9
 cov_309_45: frac[5,45] + frac[8,45] + frac[17,45] + frac[18,45]
   + frac[20,45] + frac[21,45] + frac[22,45] >= 0.7
 cov_309_46: frac[5,46] + frac[8,46] + frac[17,46] + frac[18,46]
   + frac[20,46] + frac[21,46] + frac[22,46] >= 0.7
 cov_309_47: frac[5,47] + frac[8,47] + frac[17,47] + frac[18,47]
   + frac[20,47] + frac[21,47] + frac[22,47] >= 0.7
 cov_310_0: frac[9,0] + frac[15,0] + frac[18,0] + frac[19,0] + frac[21,0]
   >= 1
 cov_310_1: frac[9,1] + frac[15,1] + frac[18,1] + frac[19,1] + frac[21,1]
   >= 1
 cov_310_2: frac[9,2] + frac[15,2] + frac[18,2] + frac[19,2] + frac[21,2]
   >= 1
 cov_311_39: frac[3,39] + frac[7,39] + frac[16,39] + frac[17,39]
   + frac[20,39] >= 1
 cov_311_40: frac[3,40] + frac[7,40] + frac[16,40] + frac[17,40]
   + frac[20,40] >= 1
 cov_311_41: frac[3,41] + frac[7,41] + frac[16,41] + frac[17,41]
   + frac[20,41] >= 1
 cov_312_30: frac[0,30] + frac[3,30] + frac[16,30] >= 0.8
 cov_312_31: frac[0,31] + frac[3,31] + frac[16,31] >= 0.8
 cov_312_32: frac[0,32] + frac[3,32] + frac[16,32] >= 0.8
 cov_313_36: frac[2,36] + frac[9,36] + frac[16,36] + frac[18,36]
   + frac[20,36] + frac[21,36] + frac[22,36] >= 0.9
 cov_313_37: frac[2,37] + frac[9,37] + frac[16,37] + frac[18,37]
   + frac[20,37] + frac[21,37] + frac[22,37] >= 0.9
 cov_313_38: frac[2,38] + frac[9,38] + frac[16,38] + frac[18,38]
   + frac[20,38] + frac[21,38] + frac[22,38] >= 0.9
 cov_314_39: frac[10,39] + frac[11,39] + frac[18,39] >= 0.7
 cov_314_40: frac[10,40] + frac[11,40] + frac[18,40] >= 0.7
 cov_314_41: frac[10,41] + frac[11,41] + frac[18,41] >= 0.7
 cov_315_45: frac[2,45] + frac[4,45] + frac[16,45] + frac[17,45]
   + frac[20,45] >= 0.7
 cov_315_46: frac[2,46] + frac[4,46] + frac[16,46] + frac[17,46]
   + frac[20,46] >= 0.7
 cov_315_47: frac[2,47] + frac[4,47] + frac[16,47] + frac[17,47]
   + frac[20,47] >= 0.7
 cov_316_60: frac[0,60] + frac[14,60] + frac[16,60] + frac[19,60]
   + frac[20,60] + frac[21,60] + frac[22,60] >= 0.7
 cov_316_61: frac[0,61] + frac[14,61] + frac[16,61] + frac[19,61]
   + frac[20,61] + frac[21,61] + frac[22,61] >= 0.7
 cov_316_62: frac[0,62] + frac[14,62] + frac[16,62] + frac[19,62]
   + frac[20,62] + frac[21,62] + frac[22,62] >= 0.7
 cov_317_162: frac[8,162] + frac[15,162] + frac[18,162] + frac[19,162]
   + frac[21,162] >= 0.9
 cov_317_163: frac[8,163] + frac[15,163] + frac[18,163] + frac[19,163]
   + frac[21,163] >= 0.9
 cov_317_164: frac[8,164] + frac[15,164] + frac[18,164] + frac[19,164]
   + frac[21,164] >= 0.9
 cov_318_69: frac[2,69] + frac[7,69] + frac[16,69] + frac[17,69]
   + frac[20,69] >= 0.7
 cov_318_70: frac[2,70] + frac[7,70] + frac[16,70] + frac[17,70]
   + frac[20,70] >= 0.7
 cov_318_71: frac[2,71] + frac[7,71] + frac[16,71] + frac[17,71]
   + frac[20,71] >= 0.7
 cov_319_174: frac[11,174] + frac[14,174] + frac[18,174] + frac[19,174]
   + frac[21,174] >= 0.7
 cov_319_175: frac[11,175] + frac[14,175] + frac[18,175] + frac[19,175]
   + frac[21,175] >= 0.7
 cov_319_176: frac[11,176] + frac[14,176] + frac[18,176] + frac[19,176]
   + frac[21,176] >= 0.7
 accuracy_(0,_sk_0)[0]: - 1.69921875 frac[0,0] + mem[0,0] >= 0
 accuracy_(0,_sk_0)[1]: - 1.69921875 frac[1,0] + mem[1,0] >= 0
 accuracy_(0,_sk_0)[2]: - 1.69921875 frac[2,0] + mem[2,0] >= 0
 accuracy_(0,_sk_0)[3]: - 1.69921875 frac[3,0] + mem[3,0] >= 0
 accuracy_(0,_sk_0)[4]: - 1.69921875 frac[4,0] + mem[4,0] >= 0
 accuracy_(0,_sk_0)[5]: - 1.69921875 frac[5,0] + mem[5,0] >= 0
 accuracy_(0,_sk_0)[6]: - 1.69921875 frac[6,0] + mem[6,0] >= 0
 accuracy_(0,_sk_0)[7]: - 1.69921875 frac[7,0] + mem[7,0] >= 0
 accuracy_(0,_sk_0)[8]: - 1.69921875 frac[8,0] + mem[8,0] >= 0
 accuracy_(0,_sk_0)[9]: - 1.69921875 frac[9,0] + mem[9,0] >= 0
 accuracy_(0,_sk_0)[10]: - 1.69921875 frac[10,0] + mem[10,0] >= 0
 accuracy_(0,_sk_0)[11]: - 1.69921875 frac[11,0] + mem[11,0] >= 0
 accuracy_(0,_sk_0)[12]: - 1.69921875 frac[12,0] + mem[12,0] >= 0
 accuracy_(0,_sk_0)[13]: - 1.69921875 frac[13,0] + mem[13,0] >= 0
 accuracy_(0,_sk_0)[14]: - 1.69921875 frac[14,0] + mem[14,0] >= 0
 accuracy_(0,_sk_0)[15]: - 1.69921875 frac[15,0] + mem[15,0] >= 0
 accuracy_(0,_sk_0)[16]: - 1.69921875 frac[16,0] + mem[16,0] >= 0
 accuracy_(0,_sk_0)[17]: - 1.69921875 frac[17,0] + mem[17,0] >= 0
 accuracy_(0,_sk_0)[18]: - 1.69921875 frac[18,0] + mem[18,0] >= 0
 accuracy_(0,_sk_0)[19]: - 1.69921875 frac[19,0] + mem[19,0] >= 0
 accuracy_(0,_sk_0)[20]: - 1.69921875 frac[20,0] + mem[20,0] >= 0
 accuracy_(0,_sk_0)[21]: - 1.69921875 frac[21,0] + mem[21,0] >= 0
 accuracy_(0,_sk_0)[22]: - 1.69921875 frac[22,0] + mem[22,0] >= 0
 accuracy_(1,_sk_0)[0]: - 1.69921875 frac[0,1] + mem[0,1] >= 0
 accuracy_(1,_sk_0)[1]: - 1.69921875 frac[1,1] + mem[1,1] >= 0
 accuracy_(1,_sk_0)[2]: - 1.69921875 frac[2,1] + mem[2,1] >= 0
 accuracy_(1,_sk_0)[3]: - 1.69921875 frac[3,1] + mem[3,1] >= 0
 accuracy_(1,_sk_0)[4]: - 1.69921875 frac[4,1] + mem[4,1] >= 0
 accuracy_(1,_sk_0)[5]: - 1.69921875 frac[5,1] + mem[5,1] >= 0
 accuracy_(1,_sk_0)[6]: - 1.69921875 frac[6,1] + mem[6,1] >= 0
 accuracy_(1,_sk_0)[7]: - 1.69921875 frac[7,1] + mem[7,1] >= 0
 accuracy_(1,_sk_0)[8]: - 1.69921875 frac[8,1] + mem[8,1] >= 0
 accuracy_(1,_sk_0)[9]: - 1.69921875 frac[9,1] + mem[9,1] >= 0
 accuracy_(1,_sk_0)[10]: - 1.69921875 frac[10,1] + mem[10,1] >= 0
 accuracy_(1,_sk_0)[11]: - 1.69921875 frac[11,1] + mem[11,1] >= 0
 accuracy_(1,_sk_0)[12]: - 1.69921875 frac[12,1] + mem[12,1] >= 0
 accuracy_(1,_sk_0)[13]: - 1.69921875 frac[13,1] + mem[13,1] >= 0
 accuracy_(1,_sk_0)[14]: - 1.69921875 frac[14,1] + mem[14,1] >= 0
 accuracy_(1,_sk_0)[15]: - 1.69921875 frac[15,1] + mem[15,1] >= 0
 accuracy_(1,_sk_0)[16]: - 1.69921875 frac[16,1] + mem[16,1] >= 0
 accuracy_(1,_sk_0)[17]: - 1.69921875 frac[17,1] + mem[17,1] >= 0
 accuracy_(1,_sk_0)[18]: - 1.69921875 frac[18,1] + mem[18,1] >= 0
 accuracy_(1,_sk_0)[19]: - 1.69921875 frac[19,1] + mem[19,1] >= 0
 accuracy_(1,_sk_0)[20]: - 1.69921875 frac[20,1] + mem[20,1] >= 0
 accuracy_(1,_sk_0)[21]: - 1.69921875 frac[21,1] + mem[21,1] >= 0
 accuracy_(1,_sk_0)[22]: - 1.69921875 frac[22,1] + mem[22,1] >= 0
 accuracy_(2,_sk_0)[0]: - 1.69921875 frac[0,2] + mem[0,2] >= 0
 accuracy_(2,_sk_0)[1]: - 1.69921875 frac[1,2] + mem[1,2] >= 0
 accuracy_(2,_sk_0)[2]: - 1.69921875 frac[2,2] + mem[2,2] >= 0
 accuracy_(2,_sk_0)[3]: - 1.69921875 frac[3,2] + mem[3,2] >= 0
 accuracy_(2,_sk_0)[4]: - 1.69921875 frac[4,2] + mem[4,2] >= 0
 accuracy_(2,_sk_0)[5]: - 1.69921875 frac[5,2] + mem[5,2] >= 0
 accuracy_(2,_sk_0)[6]: - 1.69921875 frac[6,2] + mem[6,2] >= 0
 accuracy_(2,_sk_0)[7]: - 1.69921875 frac[7,2] + mem[7,2] >= 0
 accuracy_(2,_sk_0)[8]: - 1.69921875 frac[8,2] + mem[8,2] >= 0
 accuracy_(2,_sk_0)[9]: - 1.69921875 frac[9,2] + mem[9,2] >= 0
 accuracy_(2,_sk_0)[10]: - 1.69921875 frac[10,2] + mem[10,2] >= 0
 accuracy_(2,_sk_0)[11]: - 1.69921875 frac[11,2] + mem[11,2] >= 0
 accuracy_(2,_sk_0)[12]: - 1.69921875 frac[12,2] + mem[12,2] >= 0
 accuracy_(2,_sk_0)[13]: - 1.69921875 frac[13,2] + mem[13,2] >= 0
 accuracy_(2,_sk_0)[14]: - 1.69921875 frac[14,2] + mem[14,2] >= 0
 accuracy_(2,_sk_0)[15]: - 1.69921875 frac[15,2] + mem[15,2] >= 0
 accuracy_(2,_sk_0)[16]: - 1.69921875 frac[16,2] + mem[16,2] >= 0
 accuracy_(2,_sk_0)[17]: - 1.69921875 frac[17,2] + mem[17,2] >= 0
 accuracy_(2,_sk_0)[18]: - 1.69921875 frac[18,2] + mem[18,2] >= 0
 accuracy_(2,_sk_0)[19]: - 1.69921875 frac[19,2] + mem[19,2] >= 0
 accuracy_(2,_sk_0)[20]: - 1.69921875 frac[20,2] + mem[20,2] >= 0
 accuracy_(2,_sk_0)[21]: - 1.69921875 frac[21,2] + mem[21,2] >= 0
 accuracy_(2,_sk_0)[22]: - 1.69921875 frac[22,2] + mem[22,2] >= 0
 accuracy_(0,_sk_1)[0]: - 1.69921875 frac[0,3] + mem[0,3] >= 0
 accuracy_(0,_sk_1)[1]: - 1.69921875 frac[1,3] + mem[1,3] >= 0
 accuracy_(0,_sk_1)[2]: - 1.69921875 frac[2,3] + mem[2,3] >= 0
 accuracy_(0,_sk_1)[3]: - 1.69921875 frac[3,3] + mem[3,3] >= 0
 accuracy_(0,_sk_1)[4]: - 1.69921875 frac[4,3] + mem[4,3] >= 0
 accuracy_(0,_sk_1)[5]: - 1.69921875 frac[5,3] + mem[5,3] >= 0
 accuracy_(0,_sk_1)[6]: - 1.69921875 frac[6,3] + mem[6,3] >= 0
 accuracy_(0,_sk_1)[7]: - 1.69921875 frac[7,3] + mem[7,3] >= 0
 accuracy_(0,_sk_1)[8]: - 1.69921875 frac[8,3] + mem[8,3] >= 0
 accuracy_(0,_sk_1)[9]: - 1.69921875 frac[9,3] + mem[9,3] >= 0
 accuracy_(0,_sk_1)[10]: - 1.69921875 frac[10,3] + mem[10,3] >= 0
 accuracy_(0,_sk_1)[11]: - 1.69921875 frac[11,3] + mem[11,3] >= 0
 accuracy_(0,_sk_1)[12]: - 1.69921875 frac[12,3] + mem[12,3] >= 0
 accuracy_(0,_sk_1)[13]: - 1.69921875 frac[13,3] + mem[13,3] >= 0
 accuracy_(0,_sk_1)[14]: - 1.69921875 frac[14,3] + mem[14,3] >= 0
 accuracy_(0,_sk_1)[15]: - 1.69921875 frac[15,3] + mem[15,3] >= 0
 accuracy_(0,_sk_1)[16]: - 1.69921875 frac[16,3] + mem[16,3] >= 0
 accuracy_(0,_sk_1)[17]: - 1.69921875 frac[17,3] + mem[17,3] >= 0
 accuracy_(0,_sk_1)[18]: - 1.69921875 frac[18,3] + mem[18,3] >= 0
 accuracy_(0,_sk_1)[19]: - 1.69921875 frac[19,3] + mem[19,3] >= 0
 accuracy_(0,_sk_1)[20]: - 1.69921875 frac[20,3] + mem[20,3] >= 0
 accuracy_(0,_sk_1)[21]: - 1.69921875 frac[21,3] + mem[21,3] >= 0
 accuracy_(0,_sk_1)[22]: - 1.69921875 frac[22,3] + mem[22,3] >= 0
 accuracy_(1,_sk_1)[0]: - 1.69921875 frac[0,4] + mem[0,4] >= 0
 accuracy_(1,_sk_1)[1]: - 1.69921875 frac[1,4] + mem[1,4] >= 0
 accuracy_(1,_sk_1)[2]: - 1.69921875 frac[2,4] + mem[2,4] >= 0
 accuracy_(1,_sk_1)[3]: - 1.69921875 frac[3,4] + mem[3,4] >= 0
 accuracy_(1,_sk_1)[4]: - 1.69921875 frac[4,4] + mem[4,4] >= 0
 accuracy_(1,_sk_1)[5]: - 1.69921875 frac[5,4] + mem[5,4] >= 0
 accuracy_(1,_sk_1)[6]: - 1.69921875 frac[6,4] + mem[6,4] >= 0
 accuracy_(1,_sk_1)[7]: - 1.69921875 frac[7,4] + mem[7,4] >= 0
 accuracy_(1,_sk_1)[8]: - 1.69921875 frac[8,4] + mem[8,4] >= 0
 accuracy_(1,_sk_1)[9]: - 1.69921875 frac[9,4] + mem[9,4] >= 0
 accuracy_(1,_sk_1)[10]: - 1.69921875 frac[10,4] + mem[10,4] >= 0
 accuracy_(1,_sk_1)[11]: - 1.69921875 frac[11,4] + mem[11,4] >= 0
 accuracy_(1,_sk_1)[12]: - 1.69921875 frac[12,4] + mem[12,4] >= 0
 accuracy_(1,_sk_1)[13]: - 1.69921875 frac[13,4] + mem[13,4] >= 0
 accuracy_(1,_sk_1)[14]: - 1.69921875 frac[14,4] + mem[14,4] >= 0
 accuracy_(1,_sk_1)[15]: - 1.69921875 frac[15,4] + mem[15,4] >= 0
 accuracy_(1,_sk_1)[16]: - 1.69921875 frac[16,4] + mem[16,4] >= 0
 accuracy_(1,_sk_1)[17]: - 1.69921875 frac[17,4] + mem[17,4] >= 0
 accuracy_(1,_sk_1)[18]: - 1.69921875 frac[18,4] + mem[18,4] >= 0
 accuracy_(1,_sk_1)[19]: - 1.69921875 frac[19,4] + mem[19,4] >= 0
 accuracy_(1,_sk_1)[20]: - 1.69921875 frac[20,4] + mem[20,4] >= 0
 accuracy_(1,_sk_1)[21]: - 1.69921875 frac[21,4] + mem[21,4] >= 0
 accuracy_(1,_sk_1)[22]: - 1.69921875 frac[22,4] + mem[22,4] >= 0
 accuracy_(2,_sk_1)[0]: - 1.69921875 frac[0,5] + mem[0,5] >= 0
 accuracy_(2,_sk_1)[1]: - 1.69921875 frac[1,5] + mem[1,5] >= 0
 accuracy_(2,_sk_1)[2]: - 1.69921875 frac[2,5] + mem[2,5] >= 0
 accuracy_(2,_sk_1)[3]: - 1.69921875 frac[3,5] + mem[3,5] >= 0
 accuracy_(2,_sk_1)[4]: - 1.69921875 frac[4,5] + mem[4,5] >= 0
 accuracy_(2,_sk_1)[5]: - 1.69921875 frac[5,5] + mem[5,5] >= 0
 accuracy_(2,_sk_1)[6]: - 1.69921875 frac[6,5] + mem[6,5] >= 0
 accuracy_(2,_sk_1)[7]: - 1.69921875 frac[7,5] + mem[7,5] >= 0
 accuracy_(2,_sk_1)[8]: - 1.69921875 frac[8,5] + mem[8,5] >= 0
 accuracy_(2,_sk_1)[9]: - 1.69921875 frac[9,5] + mem[9,5] >= 0
 accuracy_(2,_sk_1)[10]: - 1.69921875 frac[10,5] + mem[10,5] >= 0
 accuracy_(2,_sk_1)[11]: - 1.69921875 frac[11,5] + mem[11,5] >= 0
 accuracy_(2,_sk_1)[12]: - 1.69921875 frac[12,5] + mem[12,5] >= 0
 accuracy_(2,_sk_1)[13]: - 1.69921875 frac[13,5] + mem[13,5] >= 0
 accuracy_(2,_sk_1)[14]: - 1.69921875 frac[14,5] + mem[14,5] >= 0
 accuracy_(2,_sk_1)[15]: - 1.69921875 frac[15,5] + mem[15,5] >= 0
 accuracy_(2,_sk_1)[16]: - 1.69921875 frac[16,5] + mem[16,5] >= 0
 accuracy_(2,_sk_1)[17]: - 1.69921875 frac[17,5] + mem[17,5] >= 0
 accuracy_(2,_sk_1)[18]: - 1.69921875 frac[18,5] + mem[18,5] >= 0
 accuracy_(2,_sk_1)[19]: - 1.69921875 frac[19,5] + mem[19,5] >= 0
 accuracy_(2,_sk_1)[20]: - 1.69921875 frac[20,5] + mem[20,5] >= 0
 accuracy_(2,_sk_1)[21]: - 1.69921875 frac[21,5] + mem[21,5] >= 0
 accuracy_(2,_sk_1)[22]: - 1.69921875 frac[22,5] + mem[22,5] >= 0
 accuracy_(0,_sk_2)[0]: - 1.69921875 frac[0,6] + mem[0,6] >= 0
 accuracy_(0,_sk_2)[1]: - 1.69921875 frac[1,6] + mem[1,6] >= 0
 accuracy_(0,_sk_2)[2]: - 1.69921875 frac[2,6] + mem[2,6] >= 0
 accuracy_(0,_sk_2)[3]: - 1.69921875 frac[3,6] + mem[3,6] >= 0
 accuracy_(0,_sk_2)[4]: - 1.69921875 frac[4,6] + mem[4,6] >= 0
 accuracy_(0,_sk_2)[5]: - 1.69921875 frac[5,6] + mem[5,6] >= 0
 accuracy_(0,_sk_2)[6]: - 1.69921875 frac[6,6] + mem[6,6] >= 0
 accuracy_(0,_sk_2)[7]: - 1.69921875 frac[7,6] + mem[7,6] >= 0
 accuracy_(0,_sk_2)[8]: - 1.69921875 frac[8,6] + mem[8,6] >= 0
 accuracy_(0,_sk_2)[9]: - 1.69921875 frac[9,6] + mem[9,6] >= 0
 accuracy_(0,_sk_2)[10]: - 1.69921875 frac[10,6] + mem[10,6] >= 0
 accuracy_(0,_sk_2)[11]: - 1.69921875 frac[11,6] + mem[11,6] >= 0
 accuracy_(0,_sk_2)[12]: - 1.69921875 frac[12,6] + mem[12,6] >= 0
 accuracy_(0,_sk_2)[13]: - 1.69921875 frac[13,6] + mem[13,6] >= 0
 accuracy_(0,_sk_2)[14]: - 1.69921875 frac[14,6] + mem[14,6] >= 0
 accuracy_(0,_sk_2)[15]: - 1.69921875 frac[15,6] + mem[15,6] >= 0
 accuracy_(0,_sk_2)[16]: - 1.69921875 frac[16,6] + mem[16,6] >= 0
 accuracy_(0,_sk_2)[17]: - 1.69921875 frac[17,6] + mem[17,6] >= 0
 accuracy_(0,_sk_2)[18]: - 1.69921875 frac[18,6] + mem[18,6] >= 0
 accuracy_(0,_sk_2)[19]: - 1.69921875 frac[19,6] + mem[19,6] >= 0
 accuracy_(0,_sk_2)[20]: - 1.69921875 frac[20,6] + mem[20,6] >= 0
 accuracy_(0,_sk_2)[21]: - 1.69921875 frac[21,6] + mem[21,6] >= 0
 accuracy_(0,_sk_2)[22]: - 1.69921875 frac[22,6] + mem[22,6] >= 0
 accuracy_(1,_sk_2)[0]: - 1.69921875 frac[0,7] + mem[0,7] >= 0
 accuracy_(1,_sk_2)[1]: - 1.69921875 frac[1,7] + mem[1,7] >= 0
 accuracy_(1,_sk_2)[2]: - 1.69921875 frac[2,7] + mem[2,7] >= 0
 accuracy_(1,_sk_2)[3]: - 1.69921875 frac[3,7] + mem[3,7] >= 0
 accuracy_(1,_sk_2)[4]: - 1.69921875 frac[4,7] + mem[4,7] >= 0
 accuracy_(1,_sk_2)[5]: - 1.69921875 frac[5,7] + mem[5,7] >= 0
 accuracy_(1,_sk_2)[6]: - 1.69921875 frac[6,7] + mem[6,7] >= 0
 accuracy_(1,_sk_2)[7]: - 1.69921875 frac[7,7] + mem[7,7] >= 0
 accuracy_(1,_sk_2)[8]: - 1.69921875 frac[8,7] + mem[8,7] >= 0
 accuracy_(1,_sk_2)[9]: - 1.69921875 frac[9,7] + mem[9,7] >= 0
 accuracy_(1,_sk_2)[10]: - 1.69921875 frac[10,7] + mem[10,7] >= 0
 accuracy_(1,_sk_2)[11]: - 1.69921875 frac[11,7] + mem[11,7] >= 0
 accuracy_(1,_sk_2)[12]: - 1.69921875 frac[12,7] + mem[12,7] >= 0
 accuracy_(1,_sk_2)[13]: - 1.69921875 frac[13,7] + mem[13,7] >= 0
 accuracy_(1,_sk_2)[14]: - 1.69921875 frac[14,7] + mem[14,7] >= 0
 accuracy_(1,_sk_2)[15]: - 1.69921875 frac[15,7] + mem[15,7] >= 0
 accuracy_(1,_sk_2)[16]: - 1.69921875 frac[16,7] + mem[16,7] >= 0
 accuracy_(1,_sk_2)[17]: - 1.69921875 frac[17,7] + mem[17,7] >= 0
 accuracy_(1,_sk_2)[18]: - 1.69921875 frac[18,7] + mem[18,7] >= 0
 accuracy_(1,_sk_2)[19]: - 1.69921875 frac[19,7] + mem[19,7] >= 0
 accuracy_(1,_sk_2)[20]: - 1.69921875 frac[20,7] + mem[20,7] >= 0
 accuracy_(1,_sk_2)[21]: - 1.69921875 frac[21,7] + mem[21,7] >= 0
 accuracy_(1,_sk_2)[22]: - 1.69921875 frac[22,7] + mem[22,7] >= 0
 accuracy_(2,_sk_2)[0]: - 1.69921875 frac[0,8] + mem[0,8] >= 0
 accuracy_(2,_sk_2)[1]: - 1.69921875 frac[1,8] + mem[1,8] >= 0
 accuracy_(2,_sk_2)[2]: - 1.69921875 frac[2,8] + mem[2,8] >= 0
 accuracy_(2,_sk_2)[3]: - 1.69921875 frac[3,8] + mem[3,8] >= 0
 accuracy_(2,_sk_2)[4]: - 1.69921875 frac[4,8] + mem[4,8] >= 0
 accuracy_(2,_sk_2)[5]: - 1.69921875 frac[5,8] + mem[5,8] >= 0
 accuracy_(2,_sk_2)[6]: - 1.69921875 frac[6,8] + mem[6,8] >= 0
 accuracy_(2,_sk_2)[7]: - 1.69921875 frac[7,8] + mem[7,8] >= 0
 accuracy_(2,_sk_2)[8]: - 1.69921875 frac[8,8] + mem[8,8] >= 0
 accuracy_(2,_sk_2)[9]: - 1.69921875 frac[9,8] + mem[9,8] >= 0
 accuracy_(2,_sk_2)[10]: - 1.69921875 frac[10,8] + mem[10,8] >= 0
 accuracy_(2,_sk_2)[11]: - 1.69921875 frac[11,8] + mem[11,8] >= 0
 accuracy_(2,_sk_2)[12]: - 1.69921875 frac[12,8] + mem[12,8] >= 0
 accuracy_(2,_sk_2)[13]: - 1.69921875 frac[13,8] + mem[13,8] >= 0
 accuracy_(2,_sk_2)[14]: - 1.69921875 frac[14,8] + mem[14,8] >= 0
 accuracy_(2,_sk_2)[15]: - 1.69921875 frac[15,8] + mem[15,8] >= 0
 accuracy_(2,_sk_2)[16]: - 1.69921875 frac[16,8] + mem[16,8] >= 0
 accuracy_(2,_sk_2)[17]: - 1.69921875 frac[17,8] + mem[17,8] >= 0
 accuracy_(2,_sk_2)[18]: - 1.69921875 frac[18,8] + mem[18,8] >= 0
 accuracy_(2,_sk_2)[19]: - 1.69921875 frac[19,8] + mem[19,8] >= 0
 accuracy_(2,_sk_2)[20]: - 1.69921875 frac[20,8] + mem[20,8] >= 0
 accuracy_(2,_sk_2)[21]: - 1.69921875 frac[21,8] + mem[21,8] >= 0
 accuracy_(2,_sk_2)[22]: - 1.69921875 frac[22,8] + mem[22,8] >= 0
 accuracy_(0,_sk_3)[0]: - 1.69921875 frac[0,9] + mem[0,9] >= 0
 accuracy_(0,_sk_3)[1]: - 1.69921875 frac[1,9] + mem[1,9] >= 0
 accuracy_(0,_sk_3)[2]: - 1.69921875 frac[2,9] + mem[2,9] >= 0
 accuracy_(0,_sk_3)[3]: - 1.69921875 frac[3,9] + mem[3,9] >= 0
 accuracy_(0,_sk_3)[4]: - 1.69921875 frac[4,9] + mem[4,9] >= 0
 accuracy_(0,_sk_3)[5]: - 1.69921875 frac[5,9] + mem[5,9] >= 0
 accuracy_(0,_sk_3)[6]: - 1.69921875 frac[6,9] + mem[6,9] >= 0
 accuracy_(0,_sk_3)[7]: - 1.69921875 frac[7,9] + mem[7,9] >= 0
 accuracy_(0,_sk_3)[8]: - 1.69921875 frac[8,9] + mem[8,9] >= 0
 accuracy_(0,_sk_3)[9]: - 1.69921875 frac[9,9] + mem[9,9] >= 0
 accuracy_(0,_sk_3)[10]: - 1.69921875 frac[10,9] + mem[10,9] >= 0
 accuracy_(0,_sk_3)[11]: - 1.69921875 frac[11,9] + mem[11,9] >= 0
 accuracy_(0,_sk_3)[12]: - 1.69921875 frac[12,9] + mem[12,9] >= 0
 accuracy_(0,_sk_3)[13]: - 1.69921875 frac[13,9] + mem[13,9] >= 0
 accuracy_(0,_sk_3)[14]: - 1.69921875 frac[14,9] + mem[14,9] >= 0
 accuracy_(0,_sk_3)[15]: - 1.69921875 frac[15,9] + mem[15,9] >= 0
 accuracy_(0,_sk_3)[16]: - 1.69921875 frac[16,9] + mem[16,9] >= 0
 accuracy_(0,_sk_3)[17]: - 1.69921875 frac[17,9] + mem[17,9] >= 0
 accuracy_(0,_sk_3)[18]: - 1.69921875 frac[18,9] + mem[18,9] >= 0
 accuracy_(0,_sk_3)[19]: - 1.69921875 frac[19,9] + mem[19,9] >= 0
 accuracy_(0,_sk_3)[20]: - 1.69921875 frac[20,9] + mem[20,9] >= 0
 accuracy_(0,_sk_3)[21]: - 1.69921875 frac[21,9] + mem[21,9] >= 0
 accuracy_(0,_sk_3)[22]: - 1.69921875 frac[22,9] + mem[22,9] >= 0
 accuracy_(1,_sk_3)[0]: - 1.69921875 frac[0,10] + mem[0,10] >= 0
 accuracy_(1,_sk_3)[1]: - 1.69921875 frac[1,10] + mem[1,10] >= 0
 accuracy_(1,_sk_3)[2]: - 1.69921875 frac[2,10] + mem[2,10] >= 0
 accuracy_(1,_sk_3)[3]: - 1.69921875 frac[3,10] + mem[3,10] >= 0
 accuracy_(1,_sk_3)[4]: - 1.69921875 frac[4,10] + mem[4,10] >= 0
 accuracy_(1,_sk_3)[5]: - 1.69921875 frac[5,10] + mem[5,10] >= 0
 accuracy_(1,_sk_3)[6]: - 1.69921875 frac[6,10] + mem[6,10] >= 0
 accuracy_(1,_sk_3)[7]: - 1.69921875 frac[7,10] + mem[7,10] >= 0
 accuracy_(1,_sk_3)[8]: - 1.69921875 frac[8,10] + mem[8,10] >= 0
 accuracy_(1,_sk_3)[9]: - 1.69921875 frac[9,10] + mem[9,10] >= 0
 accuracy_(1,_sk_3)[10]: - 1.69921875 frac[10,10] + mem[10,10] >= 0
 accuracy_(1,_sk_3)[11]: - 1.69921875 frac[11,10] + mem[11,10] >= 0
 accuracy_(1,_sk_3)[12]: - 1.69921875 frac[12,10] + mem[12,10] >= 0
 accuracy_(1,_sk_3)[13]: - 1.69921875 frac[13,10] + mem[13,10] >= 0
 accuracy_(1,_sk_3)[14]: - 1.69921875 frac[14,10] + mem[14,10] >= 0
 accuracy_(1,_sk_3)[15]: - 1.69921875 frac[15,10] + mem[15,10] >= 0
 accuracy_(1,_sk_3)[16]: - 1.69921875 frac[16,10] + mem[16,10] >= 0
 accuracy_(1,_sk_3)[17]: - 1.69921875 frac[17,10] + mem[17,10] >= 0
 accuracy_(1,_sk_3)[18]: - 1.69921875 frac[18,10] + mem[18,10] >= 0
 accuracy_(1,_sk_3)[19]: - 1.69921875 frac[19,10] + mem[19,10] >= 0
 accuracy_(1,_sk_3)[20]: - 1.69921875 frac[20,10] + mem[20,10] >= 0
 accuracy_(1,_sk_3)[21]: - 1.69921875 frac[21,10] + mem[21,10] >= 0
 accuracy_(1,_sk_3)[22]: - 1.69921875 frac[22,10] + mem[22,10] >= 0
 accuracy_(2,_sk_3)[0]: - 1.69921875 frac[0,11] + mem[0,11] >= 0
 accuracy_(2,_sk_3)[1]: - 1.69921875 frac[1,11] + mem[1,11] >= 0
 accuracy_(2,_sk_3)[2]: - 1.69921875 frac[2,11] + mem[2,11] >= 0
 accuracy_(2,_sk_3)[3]: - 1.69921875 frac[3,11] + mem[3,11] >= 0
 accuracy_(2,_sk_3)[4]: - 1.69921875 frac[4,11] + mem[4,11] >= 0
 accuracy_(2,_sk_3)[5]: - 1.69921875 frac[5,11] + mem[5,11] >= 0
 accuracy_(2,_sk_3)[6]: - 1.69921875 frac[6,11] + mem[6,11] >= 0
 accuracy_(2,_sk_3)[7]: - 1.69921875 frac[7,11] + mem[7,11] >= 0
 accuracy_(2,_sk_3)[8]: - 1.69921875 frac[8,11] + mem[8,11] >= 0
 accuracy_(2,_sk_3)[9]: - 1.69921875 frac[9,11] + mem[9,11] >= 0
 accuracy_(2,_sk_3)[10]: - 1.69921875 frac[10,11] + mem[10,11] >= 0
 accuracy_(2,_sk_3)[11]: - 1.69921875 frac[11,11] + mem[11,11] >= 0
 accuracy_(2,_sk_3)[12]: - 1.69921875 frac[12,11] + mem[12,11] >= 0
 accuracy_(2,_sk_3)[13]: - 1.69921875 frac[13,11] + mem[13,11] >= 0
 accuracy_(2,_sk_3)[14]: - 1.69921875 frac[14,11] + mem[14,11] >= 0
 accuracy_(2,_sk_3)[15]: - 1.69921875 frac[15,11] + mem[15,11] >= 0
 accuracy_(2,_sk_3)[16]: - 1.69921875 frac[16,11] + mem[16,11] >= 0
 accuracy_(2,_sk_3)[17]: - 1.69921875 frac[17,11] + mem[17,11] >= 0
 accuracy_(2,_sk_3)[18]: - 1.69921875 frac[18,11] + mem[18,11] >= 0
 accuracy_(2,_sk_3)[19]: - 1.69921875 frac[19,11] + mem[19,11] >= 0
 accuracy_(2,_sk_3)[20]: - 1.69921875 frac[20,11] + mem[20,11] >= 0
 accuracy_(2,_sk_3)[21]: - 1.69921875 frac[21,11] + mem[21,11] >= 0
 accuracy_(2,_sk_3)[22]: - 1.69921875 frac[22,11] + mem[22,11] >= 0
 accuracy_(0,_sk_4)[0]: - 1.69921875 frac[0,12] + mem[0,12] >= 0
 accuracy_(0,_sk_4)[1]: - 1.69921875 frac[1,12] + mem[1,12] >= 0
 accuracy_(0,_sk_4)[2]: - 1.69921875 frac[2,12] + mem[2,12] >= 0
 accuracy_(0,_sk_4)[3]: - 1.69921875 frac[3,12] + mem[3,12] >= 0
 accuracy_(0,_sk_4)[4]: - 1.69921875 frac[4,12] + mem[4,12] >= 0
 accuracy_(0,_sk_4)[5]: - 1.69921875 frac[5,12] + mem[5,12] >= 0
 accuracy_(0,_sk_4)[6]: - 1.69921875 frac[6,12] + mem[6,12] >= 0
 accuracy_(0,_sk_4)[7]: - 1.69921875 frac[7,12] + mem[7,12] >= 0
 accuracy_(0,_sk_4)[8]: - 1.69921875 frac[8,12] + mem[8,12] >= 0
 accuracy_(0,_sk_4)[9]: - 1.69921875 frac[9,12] + mem[9,12] >= 0
 accuracy_(0,_sk_4)[10]: - 1.69921875 frac[10,12] + mem[10,12] >= 0
 accuracy_(0,_sk_4)[11]: - 1.69921875 frac[11,12] + mem[11,12] >= 0
 accuracy_(0,_sk_4)[12]: - 1.69921875 frac[12,12] + mem[12,12] >= 0
 accuracy_(0,_sk_4)[13]: - 1.69921875 frac[13,12] + mem[13,12] >= 0
 accuracy_(0,_sk_4)[14]: - 1.69921875 frac[14,12] + mem[14,12] >= 0
 accuracy_(0,_sk_4)[15]: - 1.69921875 frac[15,12] + mem[15,12] >= 0
 accuracy_(0,_sk_4)[16]: - 1.69921875 frac[16,12] + mem[16,12] >= 0
 accuracy_(0,_sk_4)[17]: - 1.69921875 frac[17,12] + mem[17,12] >= 0
 accuracy_(0,_sk_4)[18]: - 1.69921875 frac[18,12] + mem[18,12] >= 0
 accuracy_(0,_sk_4)[19]: - 1.69921875 frac[19,12] + mem[19,12] >= 0
 accuracy_(0,_sk_4)[20]: - 1.69921875 frac[20,12] + mem[20,12] >= 0
 accuracy_(0,_sk_4)[21]: - 1.69921875 frac[21,12] + mem[21,12] >= 0
 accuracy_(0,_sk_4)[22]: - 1.69921875 frac[22,12] + mem[22,12] >= 0
 accuracy_(1,_sk_4)[0]: - 1.69921875 frac[0,13] + mem[0,13] >= 0
 accuracy_(1,_sk_4)[1]: - 1.69921875 frac[1,13] + mem[1,13] >= 0
 accuracy_(1,_sk_4)[2]: - 1.69921875 frac[2,13] + mem[2,13] >= 0
 accuracy_(1,_sk_4)[3]: - 1.69921875 frac[3,13] + mem[3,13] >= 0
 accuracy_(1,_sk_4)[4]: - 1.69921875 frac[4,13] + mem[4,13] >= 0
 accuracy_(1,_sk_4)[5]: - 1.69921875 frac[5,13] + mem[5,13] >= 0
 accuracy_(1,_sk_4)[6]: - 1.69921875 frac[6,13] + mem[6,13] >= 0
 accuracy_(1,_sk_4)[7]: - 1.69921875 frac[7,13] + mem[7,13] >= 0
 accuracy_(1,_sk_4)[8]: - 1.69921875 frac[8,13] + mem[8,13] >= 0
 accuracy_(1,_sk_4)[9]: - 1.69921875 frac[9,13] + mem[9,13] >= 0
 accuracy_(1,_sk_4)[10]: - 1.69921875 frac[10,13] + mem[10,13] >= 0
 accuracy_(1,_sk_4)[11]: - 1.69921875 frac[11,13] + mem[11,13] >= 0
 accuracy_(1,_sk_4)[12]: - 1.69921875 frac[12,13] + mem[12,13] >= 0
 accuracy_(1,_sk_4)[13]: - 1.69921875 frac[13,13] + mem[13,13] >= 0
 accuracy_(1,_sk_4)[14]: - 1.69921875 frac[14,13] + mem[14,13] >= 0
 accuracy_(1,_sk_4)[15]: - 1.69921875 frac[15,13] + mem[15,13] >= 0
 accuracy_(1,_sk_4)[16]: - 1.69921875 frac[16,13] + mem[16,13] >= 0
 accuracy_(1,_sk_4)[17]: - 1.69921875 frac[17,13] + mem[17,13] >= 0
 accuracy_(1,_sk_4)[18]: - 1.69921875 frac[18,13] + mem[18,13] >= 0
 accuracy_(1,_sk_4)[19]: - 1.69921875 frac[19,13] + mem[19,13] >= 0
 accuracy_(1,_sk_4)[20]: - 1.69921875 frac[20,13] + mem[20,13] >= 0
 accuracy_(1,_sk_4)[21]: - 1.69921875 frac[21,13] + mem[21,13] >= 0
 accuracy_(1,_sk_4)[22]: - 1.69921875 frac[22,13] + mem[22,13] >= 0
 accuracy_(2,_sk_4)[0]: - 1.69921875 frac[0,14] + mem[0,14] >= 0
 accuracy_(2,_sk_4)[1]: - 1.69921875 frac[1,14] + mem[1,14] >= 0
 accuracy_(2,_sk_4)[2]: - 1.69921875 frac[2,14] + mem[2,14] >= 0
 accuracy_(2,_sk_4)[3]: - 1.69921875 frac[3,14] + mem[3,14] >= 0
 accuracy_(2,_sk_4)[4]: - 1.69921875 frac[4,14] + mem[4,14] >= 0
 accuracy_(2,_sk_4)[5]: - 1.69921875 frac[5,14] + mem[5,14] >= 0
 accuracy_(2,_sk_4)[6]: - 1.69921875 frac[6,14] + mem[6,14] >= 0
 accuracy_(2,_sk_4)[7]: - 1.69921875 frac[7,14] + mem[7,14] >= 0
 accuracy_(2,_sk_4)[8]: - 1.69921875 frac[8,14] + mem[8,14] >= 0
 accuracy_(2,_sk_4)[9]: - 1.69921875 frac[9,14] + mem[9,14] >= 0
 accuracy_(2,_sk_4)[10]: - 1.69921875 frac[10,14] + mem[10,14] >= 0
 accuracy_(2,_sk_4)[11]: - 1.69921875 frac[11,14] + mem[11,14] >= 0
 accuracy_(2,_sk_4)[12]: - 1.69921875 frac[12,14] + mem[12,14] >= 0
 accuracy_(2,_sk_4)[13]: - 1.69921875 frac[13,14] + mem[13,14] >= 0
 accuracy_(2,_sk_4)[14]: - 1.69921875 frac[14,14] + mem[14,14] >= 0
 accuracy_(2,_sk_4)[15]: - 1.69921875 frac[15,14] + mem[15,14] >= 0
 accuracy_(2,_sk_4)[16]: - 1.69921875 frac[16,14] + mem[16,14] >= 0
 accuracy_(2,_sk_4)[17]: - 1.69921875 frac[17,14] + mem[17,14] >= 0
 accuracy_(2,_sk_4)[18]: - 1.69921875 frac[18,14] + mem[18,14] >= 0
 accuracy_(2,_sk_4)[19]: - 1.69921875 frac[19,14] + mem[19,14] >= 0
 accuracy_(2,_sk_4)[20]: - 1.69921875 frac[20,14] + mem[20,14] >= 0
 accuracy_(2,_sk_4)[21]: - 1.69921875 frac[21,14] + mem[21,14] >= 0
 accuracy_(2,_sk_4)[22]: - 1.69921875 frac[22,14] + mem[22,14] >= 0
 accuracy_(0,_sk_5)[0]: - 1.69921875 frac[0,15] + mem[0,15] >= 0
 accuracy_(0,_sk_5)[1]: - 1.69921875 frac[1,15] + mem[1,15] >= 0
 accuracy_(0,_sk_5)[2]: - 1.69921875 frac[2,15] + mem[2,15] >= 0
 accuracy_(0,_sk_5)[3]: - 1.69921875 frac[3,15] + mem[3,15] >= 0
 accuracy_(0,_sk_5)[4]: - 1.69921875 frac[4,15] + mem[4,15] >= 0
 accuracy_(0,_sk_5)[5]: - 1.69921875 frac[5,15] + mem[5,15] >= 0
 accuracy_(0,_sk_5)[6]: - 1.69921875 frac[6,15] + mem[6,15] >= 0
 accuracy_(0,_sk_5)[7]: - 1.69921875 frac[7,15] + mem[7,15] >= 0
 accuracy_(0,_sk_5)[8]: - 1.69921875 frac[8,15] + mem[8,15] >= 0
 accuracy_(0,_sk_5)[9]: - 1.69921875 frac[9,15] + mem[9,15] >= 0
 accuracy_(0,_sk_5)[10]: - 1.69921875 frac[10,15] + mem[10,15] >= 0
 accuracy_(0,_sk_5)[11]: - 1.69921875 frac[11,15] + mem[11,15] >= 0
 accuracy_(0,_sk_5)[12]: - 1.69921875 frac[12,15] + mem[12,15] >= 0
 accuracy_(0,_sk_5)[13]: - 1.69921875 frac[13,15] + mem[13,15] >= 0
 accuracy_(0,_sk_5)[14]: - 1.69921875 frac[14,15] + mem[14,15] >= 0
 accuracy_(0,_sk_5)[15]: - 1.69921875 frac[15,15] + mem[15,15] >= 0
 accuracy_(0,_sk_5)[16]: - 1.69921875 frac[16,15] + mem[16,15] >= 0
 accuracy_(0,_sk_5)[17]: - 1.69921875 frac[17,15] + mem[17,15] >= 0
 accuracy_(0,_sk_5)[18]: - 1.69921875 frac[18,15] + mem[18,15] >= 0
 accuracy_(0,_sk_5)[19]: - 1.69921875 frac[19,15] + mem[19,15] >= 0
 accuracy_(0,_sk_5)[20]: - 1.69921875 frac[20,15] + mem[20,15] >= 0
 accuracy_(0,_sk_5)[21]: - 1.69921875 frac[21,15] + mem[21,15] >= 0
 accuracy_(0,_sk_5)[22]: - 1.69921875 frac[22,15] + mem[22,15] >= 0
 accuracy_(1,_sk_5)[0]: - 1.69921875 frac[0,16] + mem[0,16] >= 0
 accuracy_(1,_sk_5)[1]: - 1.69921875 frac[1,16] + mem[1,16] >= 0
 accuracy_(1,_sk_5)[2]: - 1.69921875 frac[2,16] + mem[2,16] >= 0
 accuracy_(1,_sk_5)[3]: - 1.69921875 frac[3,16] + mem[3,16] >= 0
 accuracy_(1,_sk_5)[4]: - 1.69921875 frac[4,16] + mem[4,16] >= 0
 accuracy_(1,_sk_5)[5]: - 1.69921875 frac[5,16] + mem[5,16] >= 0
 accuracy_(1,_sk_5)[6]: - 1.69921875 frac[6,16] + mem[6,16] >= 0
 accuracy_(1,_sk_5)[7]: - 1.69921875 frac[7,16] + mem[7,16] >= 0
 accuracy_(1,_sk_5)[8]: - 1.69921875 frac[8,16] + mem[8,16] >= 0
 accuracy_(1,_sk_5)[9]: - 1.69921875 frac[9,16] + mem[9,16] >= 0
 accuracy_(1,_sk_5)[10]: - 1.69921875 frac[10,16] + mem[10,16] >= 0
 accuracy_(1,_sk_5)[11]: - 1.69921875 frac[11,16] + mem[11,16] >= 0
 accuracy_(1,_sk_5)[12]: - 1.69921875 frac[12,16] + mem[12,16] >= 0
 accuracy_(1,_sk_5)[13]: - 1.69921875 frac[13,16] + mem[13,16] >= 0
 accuracy_(1,_sk_5)[14]: - 1.69921875 frac[14,16] + mem[14,16] >= 0
 accuracy_(1,_sk_5)[15]: - 1.69921875 frac[15,16] + mem[15,16] >= 0
 accuracy_(1,_sk_5)[16]: - 1.69921875 frac[16,16] + mem[16,16] >= 0
 accuracy_(1,_sk_5)[17]: - 1.69921875 frac[17,16] + mem[17,16] >= 0
 accuracy_(1,_sk_5)[18]: - 1.69921875 frac[18,16] + mem[18,16] >= 0
 accuracy_(1,_sk_5)[19]: - 1.69921875 frac[19,16] + mem[19,16] >= 0
 accuracy_(1,_sk_5)[20]: - 1.69921875 frac[20,16] + mem[20,16] >= 0
 accuracy_(1,_sk_5)[21]: - 1.69921875 frac[21,16] + mem[21,16] >= 0
 accuracy_(1,_sk_5)[22]: - 1.69921875 frac[22,16] + mem[22,16] >= 0
 accuracy_(2,_sk_5)[0]: - 1.69921875 frac[0,17] + mem[0,17] >= 0
 accuracy_(2,_sk_5)[1]: - 1.69921875 frac[1,17] + mem[1,17] >= 0
 accuracy_(2,_sk_5)[2]: - 1.69921875 frac[2,17] + mem[2,17] >= 0
 accuracy_(2,_sk_5)[3]: - 1.69921875 frac[3,17] + mem[3,17] >= 0
 accuracy_(2,_sk_5)[4]: - 1.69921875 frac[4,17] + mem[4,17] >= 0
 accuracy_(2,_sk_5)[5]: - 1.69921875 frac[5,17] + mem[5,17] >= 0
 accuracy_(2,_sk_5)[6]: - 1.69921875 frac[6,17] + mem[6,17] >= 0
 accuracy_(2,_sk_5)[7]: - 1.69921875 frac[7,17] + mem[7,17] >= 0
 accuracy_(2,_sk_5)[8]: - 1.69921875 frac[8,17] + mem[8,17] >= 0
 accuracy_(2,_sk_5)[9]: - 1.69921875 frac[9,17] + mem[9,17] >= 0
 accuracy_(2,_sk_5)[10]: - 1.69921875 frac[10,17] + mem[10,17] >= 0
 accuracy_(2,_sk_5)[11]: - 1.69921875 frac[11,17] + mem[11,17] >= 0
 accuracy_(2,_sk_5)[12]: - 1.69921875 frac[12,17] + mem[12,17] >= 0
 accuracy_(2,_sk_5)[13]: - 1.69921875 frac[13,17] + mem[13,17] >= 0
 accuracy_(2,_sk_5)[14]: - 1.69921875 frac[14,17] + mem[14,17] >= 0
 accuracy_(2,_sk_5)[15]: - 1.69921875 frac[15,17] + mem[15,17] >= 0
 accuracy_(2,_sk_5)[16]: - 1.69921875 frac[16,17] + mem[16,17] >= 0
 accuracy_(2,_sk_5)[17]: - 1.69921875 frac[17,17] + mem[17,17] >= 0
 accuracy_(2,_sk_5)[18]: - 1.69921875 frac[18,17] + mem[18,17] >= 0
 accuracy_(2,_sk_5)[19]: - 1.69921875 frac[19,17] + mem[19,17] >= 0
 accuracy_(2,_sk_5)[20]: - 1.69921875 frac[20,17] + mem[20,17] >= 0
 accuracy_(2,_sk_5)[21]: - 1.69921875 frac[21,17] + mem[21,17] >= 0
 accuracy_(2,_sk_5)[22]: - 1.69921875 frac[22,17] + mem[22,17] >= 0
 accuracy_(0,_sk_6)[0]: - 1.69921875 frac[0,18] + mem[0,18] >= 0
 accuracy_(0,_sk_6)[1]: - 1.69921875 frac[1,18] + mem[1,18] >= 0
 accuracy_(0,_sk_6)[2]: - 1.69921875 frac[2,18] + mem[2,18] >= 0
 accuracy_(0,_sk_6)[3]: - 1.69921875 frac[3,18] + mem[3,18] >= 0
 accuracy_(0,_sk_6)[4]: - 1.69921875 frac[4,18] + mem[4,18] >= 0
 accuracy_(0,_sk_6)[5]: - 1.69921875 frac[5,18] + mem[5,18] >= 0
 accuracy_(0,_sk_6)[6]: - 1.69921875 frac[6,18] + mem[6,18] >= 0
 accuracy_(0,_sk_6)[7]: - 1.69921875 frac[7,18] + mem[7,18] >= 0
 accuracy_(0,_sk_6)[8]: - 1.69921875 frac[8,18] + mem[8,18] >= 0
 accuracy_(0,_sk_6)[9]: - 1.69921875 frac[9,18] + mem[9,18] >= 0
 accuracy_(0,_sk_6)[10]: - 1.69921875 frac[10,18] + mem[10,18] >= 0
 accuracy_(0,_sk_6)[11]: - 1.69921875 frac[11,18] + mem[11,18] >= 0
 accuracy_(0,_sk_6)[12]: - 1.69921875 frac[12,18] + mem[12,18] >= 0
 accuracy_(0,_sk_6)[13]: - 1.69921875 frac[13,18] + mem[13,18] >= 0
 accuracy_(0,_sk_6)[14]: - 1.69921875 frac[14,18] + mem[14,18] >= 0
 accuracy_(0,_sk_6)[15]: - 1.69921875 frac[15,18] + mem[15,18] >= 0
 accuracy_(0,_sk_6)[16]: - 1.69921875 frac[16,18] + mem[16,18] >= 0
 accuracy_(0,_sk_6)[17]: - 1.69921875 frac[17,18] + mem[17,18] >= 0
 accuracy_(0,_sk_6)[18]: - 1.69921875 frac[18,18] + mem[18,18] >= 0
 accuracy_(0,_sk_6)[19]: - 1.69921875 frac[19,18] + mem[19,18] >= 0
 accuracy_(0,_sk_6)[20]: - 1.69921875 frac[20,18] + mem[20,18] >= 0
 accuracy_(0,_sk_6)[21]: - 1.69921875 frac[21,18] + mem[21,18] >= 0
 accuracy_(0,_sk_6)[22]: - 1.69921875 frac[22,18] + mem[22,18] >= 0
 accuracy_(1,_sk_6)[0]: - 1.69921875 frac[0,19] + mem[0,19] >= 0
 accuracy_(1,_sk_6)[1]: - 1.69921875 frac[1,19] + mem[1,19] >= 0
 accuracy_(1,_sk_6)[2]: - 1.69921875 frac[2,19] + mem[2,19] >= 0
 accuracy_(1,_sk_6)[3]: - 1.69921875 frac[3,19] + mem[3,19] >= 0
 accuracy_(1,_sk_6)[4]: - 1.69921875 frac[4,19] + mem[4,19] >= 0
 accuracy_(1,_sk_6)[5]: - 1.69921875 frac[5,19] + mem[5,19] >= 0
 accuracy_(1,_sk_6)[6]: - 1.69921875 frac[6,19] + mem[6,19] >= 0
 accuracy_(1,_sk_6)[7]: - 1.69921875 frac[7,19] + mem[7,19] >= 0
 accuracy_(1,_sk_6)[8]: - 1.69921875 frac[8,19] + mem[8,19] >= 0
 accuracy_(1,_sk_6)[9]: - 1.69921875 frac[9,19] + mem[9,19] >= 0
 accuracy_(1,_sk_6)[10]: - 1.69921875 frac[10,19] + mem[10,19] >= 0
 accuracy_(1,_sk_6)[11]: - 1.69921875 frac[11,19] + mem[11,19] >= 0
 accuracy_(1,_sk_6)[12]: - 1.69921875 frac[12,19] + mem[12,19] >= 0
 accuracy_(1,_sk_6)[13]: - 1.69921875 frac[13,19] + mem[13,19] >= 0
 accuracy_(1,_sk_6)[14]: - 1.69921875 frac[14,19] + mem[14,19] >= 0
 accuracy_(1,_sk_6)[15]: - 1.69921875 frac[15,19] + mem[15,19] >= 0
 accuracy_(1,_sk_6)[16]: - 1.69921875 frac[16,19] + mem[16,19] >= 0
 accuracy_(1,_sk_6)[17]: - 1.69921875 frac[17,19] + mem[17,19] >= 0
 accuracy_(1,_sk_6)[18]: - 1.69921875 frac[18,19] + mem[18,19] >= 0
 accuracy_(1,_sk_6)[19]: - 1.69921875 frac[19,19] + mem[19,19] >= 0
 accuracy_(1,_sk_6)[20]: - 1.69921875 frac[20,19] + mem[20,19] >= 0
 accuracy_(1,_sk_6)[21]: - 1.69921875 frac[21,19] + mem[21,19] >= 0
 accuracy_(1,_sk_6)[22]: - 1.69921875 frac[22,19] + mem[22,19] >= 0
 accuracy_(2,_sk_6)[0]: - 1.69921875 frac[0,20] + mem[0,20] >= 0
 accuracy_(2,_sk_6)[1]: - 1.69921875 frac[1,20] + mem[1,20] >= 0
 accuracy_(2,_sk_6)[2]: - 1.69921875 frac[2,20] + mem[2,20] >= 0
 accuracy_(2,_sk_6)[3]: - 1.69921875 frac[3,20] + mem[3,20] >= 0
 accuracy_(2,_sk_6)[4]: - 1.69921875 frac[4,20] + mem[4,20] >= 0
 accuracy_(2,_sk_6)[5]: - 1.69921875 frac[5,20] + mem[5,20] >= 0
 accuracy_(2,_sk_6)[6]: - 1.69921875 frac[6,20] + mem[6,20] >= 0
 accuracy_(2,_sk_6)[7]: - 1.69921875 frac[7,20] + mem[7,20] >= 0
 accuracy_(2,_sk_6)[8]: - 1.69921875 frac[8,20] + mem[8,20] >= 0
 accuracy_(2,_sk_6)[9]: - 1.69921875 frac[9,20] + mem[9,20] >= 0
 accuracy_(2,_sk_6)[10]: - 1.69921875 frac[10,20] + mem[10,20] >= 0
 accuracy_(2,_sk_6)[11]: - 1.69921875 frac[11,20] + mem[11,20] >= 0
 accuracy_(2,_sk_6)[12]: - 1.69921875 frac[12,20] + mem[12,20] >= 0
 accuracy_(2,_sk_6)[13]: - 1.69921875 frac[13,20] + mem[13,20] >= 0
 accuracy_(2,_sk_6)[14]: - 1.69921875 frac[14,20] + mem[14,20] >= 0
 accuracy_(2,_sk_6)[15]: - 1.69921875 frac[15,20] + mem[15,20] >= 0
 accuracy_(2,_sk_6)[16]: - 1.69921875 frac[16,20] + mem[16,20] >= 0
 accuracy_(2,_sk_6)[17]: - 1.69921875 frac[17,20] + mem[17,20] >= 0
 accuracy_(2,_sk_6)[18]: - 1.69921875 frac[18,20] + mem[18,20] >= 0
 accuracy_(2,_sk_6)[19]: - 1.69921875 frac[19,20] + mem[19,20] >= 0
 accuracy_(2,_sk_6)[20]: - 1.69921875 frac[20,20] + mem[20,20] >= 0
 accuracy_(2,_sk_6)[21]: - 1.69921875 frac[21,20] + mem[21,20] >= 0
 accuracy_(2,_sk_6)[22]: - 1.69921875 frac[22,20] + mem[22,20] >= 0
 accuracy_(0,_sk_7)[0]: - 1.69921875 frac[0,21] + mem[0,21] >= 0
 accuracy_(0,_sk_7)[1]: - 1.69921875 frac[1,21] + mem[1,21] >= 0
 accuracy_(0,_sk_7)[2]: - 1.69921875 frac[2,21] + mem[2,21] >= 0
 accuracy_(0,_sk_7)[3]: - 1.69921875 frac[3,21] + mem[3,21] >= 0
 accuracy_(0,_sk_7)[4]: - 1.69921875 frac[4,21] + mem[4,21] >= 0
 accuracy_(0,_sk_7)[5]: - 1.69921875 frac[5,21] + mem[5,21] >= 0
 accuracy_(0,_sk_7)[6]: - 1.69921875 frac[6,21] + mem[6,21] >= 0
 accuracy_(0,_sk_7)[7]: - 1.69921875 frac[7,21] + mem[7,21] >= 0
 accuracy_(0,_sk_7)[8]: - 1.69921875 frac[8,21] + mem[8,21] >= 0
 accuracy_(0,_sk_7)[9]: - 1.69921875 frac[9,21] + mem[9,21] >= 0
 accuracy_(0,_sk_7)[10]: - 1.69921875 frac[10,21] + mem[10,21] >= 0
 accuracy_(0,_sk_7)[11]: - 1.69921875 frac[11,21] + mem[11,21] >= 0
 accuracy_(0,_sk_7)[12]: - 1.69921875 frac[12,21] + mem[12,21] >= 0
 accuracy_(0,_sk_7)[13]: - 1.69921875 frac[13,21] + mem[13,21] >= 0
 accuracy_(0,_sk_7)[14]: - 1.69921875 frac[14,21] + mem[14,21] >= 0
 accuracy_(0,_sk_7)[15]: - 1.69921875 frac[15,21] + mem[15,21] >= 0
 accuracy_(0,_sk_7)[16]: - 1.69921875 frac[16,21] + mem[16,21] >= 0
 accuracy_(0,_sk_7)[17]: - 1.69921875 frac[17,21] + mem[17,21] >= 0
 accuracy_(0,_sk_7)[18]: - 1.69921875 frac[18,21] + mem[18,21] >= 0
 accuracy_(0,_sk_7)[19]: - 1.69921875 frac[19,21] + mem[19,21] >= 0
 accuracy_(0,_sk_7)[20]: - 1.69921875 frac[20,21] + mem[20,21] >= 0
 accuracy_(0,_sk_7)[21]: - 1.69921875 frac[21,21] + mem[21,21] >= 0
 accuracy_(0,_sk_7)[22]: - 1.69921875 frac[22,21] + mem[22,21] >= 0
 accuracy_(1,_sk_7)[0]: - 1.69921875 frac[0,22] + mem[0,22] >= 0
 accuracy_(1,_sk_7)[1]: - 1.69921875 frac[1,22] + mem[1,22] >= 0
 accuracy_(1,_sk_7)[2]: - 1.69921875 frac[2,22] + mem[2,22] >= 0
 accuracy_(1,_sk_7)[3]: - 1.69921875 frac[3,22] + mem[3,22] >= 0
 accuracy_(1,_sk_7)[4]: - 1.69921875 frac[4,22] + mem[4,22] >= 0
 accuracy_(1,_sk_7)[5]: - 1.69921875 frac[5,22] + mem[5,22] >= 0
 accuracy_(1,_sk_7)[6]: - 1.69921875 frac[6,22] + mem[6,22] >= 0
 accuracy_(1,_sk_7)[7]: - 1.69921875 frac[7,22] + mem[7,22] >= 0
 accuracy_(1,_sk_7)[8]: - 1.69921875 frac[8,22] + mem[8,22] >= 0
 accuracy_(1,_sk_7)[9]: - 1.69921875 frac[9,22] + mem[9,22] >= 0
 accuracy_(1,_sk_7)[10]: - 1.69921875 frac[10,22] + mem[10,22] >= 0
 accuracy_(1,_sk_7)[11]: - 1.69921875 frac[11,22] + mem[11,22] >= 0
 accuracy_(1,_sk_7)[12]: - 1.69921875 frac[12,22] + mem[12,22] >= 0
 accuracy_(1,_sk_7)[13]: - 1.69921875 frac[13,22] + mem[13,22] >= 0
 accuracy_(1,_sk_7)[14]: - 1.69921875 frac[14,22] + mem[14,22] >= 0
 accuracy_(1,_sk_7)[15]: - 1.69921875 frac[15,22] + mem[15,22] >= 0
 accuracy_(1,_sk_7)[16]: - 1.69921875 frac[16,22] + mem[16,22] >= 0
 accuracy_(1,_sk_7)[17]: - 1.69921875 frac[17,22] + mem[17,22] >= 0
 accuracy_(1,_sk_7)[18]: - 1.69921875 frac[18,22] + mem[18,22] >= 0
 accuracy_(1,_sk_7)[19]: - 1.69921875 frac[19,22] + mem[19,22] >= 0
 accuracy_(1,_sk_7)[20]: - 1.69921875 frac[20,22] + mem[20,22] >= 0
 accuracy_(1,_sk_7)[21]: - 1.69921875 frac[21,22] + mem[21,22] >= 0
 accuracy_(1,_sk_7)[22]: - 1.69921875 frac[22,22] + mem[22,22] >= 0
 accuracy_(2,_sk_7)[0]: - 1.69921875 frac[0,23] + mem[0,23] >= 0
 accuracy_(2,_sk_7)[1]: - 1.69921875 frac[1,23] + mem[1,23] >= 0
 accuracy_(2,_sk_7)[2]: - 1.69921875 frac[2,23] + mem[2,23] >= 0
 accuracy_(2,_sk_7)[3]: - 1.69921875 frac[3,23] + mem[3,23] >= 0
 accuracy_(2,_sk_7)[4]: - 1.69921875 frac[4,23] + mem[4,23] >= 0
 accuracy_(2,_sk_7)[5]: - 1.69921875 frac[5,23] + mem[5,23] >= 0
 accuracy_(2,_sk_7)[6]: - 1.69921875 frac[6,23] + mem[6,23] >= 0
 accuracy_(2,_sk_7)[7]: - 1.69921875 frac[7,23] + mem[7,23] >= 0
 accuracy_(2,_sk_7)[8]: - 1.69921875 frac[8,23] + mem[8,23] >= 0
 accuracy_(2,_sk_7)[9]: - 1.69921875 frac[9,23] + mem[9,23] >= 0
 accuracy_(2,_sk_7)[10]: - 1.69921875 frac[10,23] + mem[10,23] >= 0
 accuracy_(2,_sk_7)[11]: - 1.69921875 frac[11,23] + mem[11,23] >= 0
 accuracy_(2,_sk_7)[12]: - 1.69921875 frac[12,23] + mem[12,23] >= 0
 accuracy_(2,_sk_7)[13]: - 1.69921875 frac[13,23] + mem[13,23] >= 0
 accuracy_(2,_sk_7)[14]: - 1.69921875 frac[14,23] + mem[14,23] >= 0
 accuracy_(2,_sk_7)[15]: - 1.69921875 frac[15,23] + mem[15,23] >= 0
 accuracy_(2,_sk_7)[16]: - 1.69921875 frac[16,23] + mem[16,23] >= 0
 accuracy_(2,_sk_7)[17]: - 1.69921875 frac[17,23] + mem[17,23] >= 0
 accuracy_(2,_sk_7)[18]: - 1.69921875 frac[18,23] + mem[18,23] >= 0
 accuracy_(2,_sk_7)[19]: - 1.69921875 frac[19,23] + mem[19,23] >= 0
 accuracy_(2,_sk_7)[20]: - 1.69921875 frac[20,23] + mem[20,23] >= 0
 accuracy_(2,_sk_7)[21]: - 1.69921875 frac[21,23] + mem[21,23] >= 0
 accuracy_(2,_sk_7)[22]: - 1.69921875 frac[22,23] + mem[22,23] >= 0
 accuracy_(0,_sk_8)[0]: - 1.69921875 frac[0,24] + mem[0,24] >= 0
 accuracy_(0,_sk_8)[1]: - 1.69921875 frac[1,24] + mem[1,24] >= 0
 accuracy_(0,_sk_8)[2]: - 1.69921875 frac[2,24] + mem[2,24] >= 0
 accuracy_(0,_sk_8)[3]: - 1.69921875 frac[3,24] + mem[3,24] >= 0
 accuracy_(0,_sk_8)[4]: - 1.69921875 frac[4,24] + mem[4,24] >= 0
 accuracy_(0,_sk_8)[5]: - 1.69921875 frac[5,24] + mem[5,24] >= 0
 accuracy_(0,_sk_8)[6]: - 1.69921875 frac[6,24] + mem[6,24] >= 0
 accuracy_(0,_sk_8)[7]: - 1.69921875 frac[7,24] + mem[7,24] >= 0
 accuracy_(0,_sk_8)[8]: - 1.69921875 frac[8,24] + mem[8,24] >= 0
 accuracy_(0,_sk_8)[9]: - 1.69921875 frac[9,24] + mem[9,24] >= 0
 accuracy_(0,_sk_8)[10]: - 1.69921875 frac[10,24] + mem[10,24] >= 0
 accuracy_(0,_sk_8)[11]: - 1.69921875 frac[11,24] + mem[11,24] >= 0
 accuracy_(0,_sk_8)[12]: - 1.69921875 frac[12,24] + mem[12,24] >= 0
 accuracy_(0,_sk_8)[13]: - 1.69921875 frac[13,24] + mem[13,24] >= 0
 accuracy_(0,_sk_8)[14]: - 1.69921875 frac[14,24] + mem[14,24] >= 0
 accuracy_(0,_sk_8)[15]: - 1.69921875 frac[15,24] + mem[15,24] >= 0
 accuracy_(0,_sk_8)[16]: - 1.69921875 frac[16,24] + mem[16,24] >= 0
 accuracy_(0,_sk_8)[17]: - 1.69921875 frac[17,24] + mem[17,24] >= 0
 accuracy_(0,_sk_8)[18]: - 1.69921875 frac[18,24] + mem[18,24] >= 0
 accuracy_(0,_sk_8)[19]: - 1.69921875 frac[19,24] + mem[19,24] >= 0
 accuracy_(0,_sk_8)[20]: - 1.69921875 frac[20,24] + mem[20,24] >= 0
 accuracy_(0,_sk_8)[21]: - 1.69921875 frac[21,24] + mem[21,24] >= 0
 accuracy_(0,_sk_8)[22]: - 1.69921875 frac[22,24] + mem[22,24] >= 0
 accuracy_(1,_sk_8)[0]: - 1.69921875 frac[0,25] + mem[0,25] >= 0
 accuracy_(1,_sk_8)[1]: - 1.69921875 frac[1,25] + mem[1,25] >= 0
 accuracy_(1,_sk_8)[2]: - 1.69921875 frac[2,25] + mem[2,25] >= 0
 accuracy_(1,_sk_8)[3]: - 1.69921875 frac[3,25] + mem[3,25] >= 0
 accuracy_(1,_sk_8)[4]: - 1.69921875 frac[4,25] + mem[4,25] >= 0
 accuracy_(1,_sk_8)[5]: - 1.69921875 frac[5,25] + mem[5,25] >= 0
 accuracy_(1,_sk_8)[6]: - 1.69921875 frac[6,25] + mem[6,25] >= 0
 accuracy_(1,_sk_8)[7]: - 1.69921875 frac[7,25] + mem[7,25] >= 0
 accuracy_(1,_sk_8)[8]: - 1.69921875 frac[8,25] + mem[8,25] >= 0
 accuracy_(1,_sk_8)[9]: - 1.69921875 frac[9,25] + mem[9,25] >= 0
 accuracy_(1,_sk_8)[10]: - 1.69921875 frac[10,25] + mem[10,25] >= 0
 accuracy_(1,_sk_8)[11]: - 1.69921875 frac[11,25] + mem[11,25] >= 0
 accuracy_(1,_sk_8)[12]: - 1.69921875 frac[12,25] + mem[12,25] >= 0
 accuracy_(1,_sk_8)[13]: - 1.69921875 frac[13,25] + mem[13,25] >= 0
 accuracy_(1,_sk_8)[14]: - 1.69921875 frac[14,25] + mem[14,25] >= 0
 accuracy_(1,_sk_8)[15]: - 1.69921875 frac[15,25] + mem[15,25] >= 0
 accuracy_(1,_sk_8)[16]: - 1.69921875 frac[16,25] + mem[16,25] >= 0
 accuracy_(1,_sk_8)[17]: - 1.69921875 frac[17,25] + mem[17,25] >= 0
 accuracy_(1,_sk_8)[18]: - 1.69921875 frac[18,25] + mem[18,25] >= 0
 accuracy_(1,_sk_8)[19]: - 1.69921875 frac[19,25] + mem[19,25] >= 0
 accuracy_(1,_sk_8)[20]: - 1.69921875 frac[20,25] + mem[20,25] >= 0
 accuracy_(1,_sk_8)[21]: - 1.69921875 frac[21,25] + mem[21,25] >= 0
 accuracy_(1,_sk_8)[22]: - 1.69921875 frac[22,25] + mem[22,25] >= 0
 accuracy_(2,_sk_8)[0]: - 1.69921875 frac[0,26] + mem[0,26] >= 0
 accuracy_(2,_sk_8)[1]: - 1.69921875 frac[1,26] + mem[1,26] >= 0
 accuracy_(2,_sk_8)[2]: - 1.69921875 frac[2,26] + mem[2,26] >= 0
 accuracy_(2,_sk_8)[3]: - 1.69921875 frac[3,26] + mem[3,26] >= 0
 accuracy_(2,_sk_8)[4]: - 1.69921875 frac[4,26] + mem[4,26] >= 0
 accuracy_(2,_sk_8)[5]: - 1.69921875 frac[5,26] + mem[5,26] >= 0
 accuracy_(2,_sk_8)[6]: - 1.69921875 frac[6,26] + mem[6,26] >= 0
 accuracy_(2,_sk_8)[7]: - 1.69921875 frac[7,26] + mem[7,26] >= 0
 accuracy_(2,_sk_8)[8]: - 1.69921875 frac[8,26] + mem[8,26] >= 0
 accuracy_(2,_sk_8)[9]: - 1.69921875 frac[9,26] + mem[9,26] >= 0
 accuracy_(2,_sk_8)[10]: - 1.69921875 frac[10,26] + mem[10,26] >= 0
 accuracy_(2,_sk_8)[11]: - 1.69921875 frac[11,26] + mem[11,26] >= 0
 accuracy_(2,_sk_8)[12]: - 1.69921875 frac[12,26] + mem[12,26] >= 0
 accuracy_(2,_sk_8)[13]: - 1.69921875 frac[13,26] + mem[13,26] >= 0
 accuracy_(2,_sk_8)[14]: - 1.69921875 frac[14,26] + mem[14,26] >= 0
 accuracy_(2,_sk_8)[15]: - 1.69921875 frac[15,26] + mem[15,26] >= 0
 accuracy_(2,_sk_8)[16]: - 1.69921875 frac[16,26] + mem[16,26] >= 0
 accuracy_(2,_sk_8)[17]: - 1.69921875 frac[17,26] + mem[17,26] >= 0
 accuracy_(2,_sk_8)[18]: - 1.69921875 frac[18,26] + mem[18,26] >= 0
 accuracy_(2,_sk_8)[19]: - 1.69921875 frac[19,26] + mem[19,26] >= 0
 accuracy_(2,_sk_8)[20]: - 1.69921875 frac[20,26] + mem[20,26] >= 0
 accuracy_(2,_sk_8)[21]: - 1.69921875 frac[21,26] + mem[21,26] >= 0
 accuracy_(2,_sk_8)[22]: - 1.69921875 frac[22,26] + mem[22,26] >= 0
 accuracy_(0,_sk_9)[0]: - 1.69921875 frac[0,27] + mem[0,27] >= 0
 accuracy_(0,_sk_9)[1]: - 1.69921875 frac[1,27] + mem[1,27] >= 0
 accuracy_(0,_sk_9)[2]: - 1.69921875 frac[2,27] + mem[2,27] >= 0
 accuracy_(0,_sk_9)[3]: - 1.69921875 frac[3,27] + mem[3,27] >= 0
 accuracy_(0,_sk_9)[4]: - 1.69921875 frac[4,27] + mem[4,27] >= 0
 accuracy_(0,_sk_9)[5]: - 1.69921875 frac[5,27] + mem[5,27] >= 0
 accuracy_(0,_sk_9)[6]: - 1.69921875 frac[6,27] + mem[6,27] >= 0
 accuracy_(0,_sk_9)[7]: - 1.69921875 frac[7,27] + mem[7,27] >= 0
 accuracy_(0,_sk_9)[8]: - 1.69921875 frac[8,27] + mem[8,27] >= 0
 accuracy_(0,_sk_9)[9]: - 1.69921875 frac[9,27] + mem[9,27] >= 0
 accuracy_(0,_sk_9)[10]: - 1.69921875 frac[10,27] + mem[10,27] >= 0
 accuracy_(0,_sk_9)[11]: - 1.69921875 frac[11,27] + mem[11,27] >= 0
 accuracy_(0,_sk_9)[12]: - 1.69921875 frac[12,27] + mem[12,27] >= 0
 accuracy_(0,_sk_9)[13]: - 1.69921875 frac[13,27] + mem[13,27] >= 0
 accuracy_(0,_sk_9)[14]: - 1.69921875 frac[14,27] + mem[14,27] >= 0
 accuracy_(0,_sk_9)[15]: - 1.69921875 frac[15,27] + mem[15,27] >= 0
 accuracy_(0,_sk_9)[16]: - 1.69921875 frac[16,27] + mem[16,27] >= 0
 accuracy_(0,_sk_9)[17]: - 1.69921875 frac[17,27] + mem[17,27] >= 0
 accuracy_(0,_sk_9)[18]: - 1.69921875 frac[18,27] + mem[18,27] >= 0
 accuracy_(0,_sk_9)[19]: - 1.69921875 frac[19,27] + mem[19,27] >= 0
 accuracy_(0,_sk_9)[20]: - 1.69921875 frac[20,27] + mem[20,27] >= 0
 accuracy_(0,_sk_9)[21]: - 1.69921875 frac[21,27] + mem[21,27] >= 0
 accuracy_(0,_sk_9)[22]: - 1.69921875 frac[22,27] + mem[22,27] >= 0
 accuracy_(1,_sk_9)[0]: - 1.69921875 frac[0,28] + mem[0,28] >= 0
 accuracy_(1,_sk_9)[1]: - 1.69921875 frac[1,28] + mem[1,28] >= 0
 accuracy_(1,_sk_9)[2]: - 1.69921875 frac[2,28] + mem[2,28] >= 0
 accuracy_(1,_sk_9)[3]: - 1.69921875 frac[3,28] + mem[3,28] >= 0
 accuracy_(1,_sk_9)[4]: - 1.69921875 frac[4,28] + mem[4,28] >= 0
 accuracy_(1,_sk_9)[5]: - 1.69921875 frac[5,28] + mem[5,28] >= 0
 accuracy_(1,_sk_9)[6]: - 1.69921875 frac[6,28] + mem[6,28] >= 0
 accuracy_(1,_sk_9)[7]: - 1.69921875 frac[7,28] + mem[7,28] >= 0
 accuracy_(1,_sk_9)[8]: - 1.69921875 frac[8,28] + mem[8,28] >= 0
 accuracy_(1,_sk_9)[9]: - 1.69921875 frac[9,28] + mem[9,28] >= 0
 accuracy_(1,_sk_9)[10]: - 1.69921875 frac[10,28] + mem[10,28] >= 0
 accuracy_(1,_sk_9)[11]: - 1.69921875 frac[11,28] + mem[11,28] >= 0
 accuracy_(1,_sk_9)[12]: - 1.69921875 frac[12,28] + mem[12,28] >= 0
 accuracy_(1,_sk_9)[13]: - 1.69921875 frac[13,28] + mem[13,28] >= 0
 accuracy_(1,_sk_9)[14]: - 1.69921875 frac[14,28] + mem[14,28] >= 0
 accuracy_(1,_sk_9)[15]: - 1.69921875 frac[15,28] + mem[15,28] >= 0
 accuracy_(1,_sk_9)[16]: - 1.69921875 frac[16,28] + mem[16,28] >= 0
 accuracy_(1,_sk_9)[17]: - 1.69921875 frac[17,28] + mem[17,28] >= 0
 accuracy_(1,_sk_9)[18]: - 1.69921875 frac[18,28] + mem[18,28] >= 0
 accuracy_(1,_sk_9)[19]: - 1.69921875 frac[19,28] + mem[19,28] >= 0
 accuracy_(1,_sk_9)[20]: - 1.69921875 frac[20,28] + mem[20,28] >= 0
 accuracy_(1,_sk_9)[21]: - 1.69921875 frac[21,28] + mem[21,28] >= 0
 accuracy_(1,_sk_9)[22]: - 1.69921875 frac[22,28] + mem[22,28] >= 0
 accuracy_(2,_sk_9)[0]: - 1.69921875 frac[0,29] + mem[0,29] >= 0
 accuracy_(2,_sk_9)[1]: - 1.69921875 frac[1,29] + mem[1,29] >= 0
 accuracy_(2,_sk_9)[2]: - 1.69921875 frac[2,29] + mem[2,29] >= 0
 accuracy_(2,_sk_9)[3]: - 1.69921875 frac[3,29] + mem[3,29] >= 0
 accuracy_(2,_sk_9)[4]: - 1.69921875 frac[4,29] + mem[4,29] >= 0
 accuracy_(2,_sk_9)[5]: - 1.69921875 frac[5,29] + mem[5,29] >= 0
 accuracy_(2,_sk_9)[6]: - 1.69921875 frac[6,29] + mem[6,29] >= 0
 accuracy_(2,_sk_9)[7]: - 1.69921875 frac[7,29] + mem[7,29] >= 0
 accuracy_(2,_sk_9)[8]: - 1.69921875 frac[8,29] + mem[8,29] >= 0
 accuracy_(2,_sk_9)[9]: - 1.69921875 frac[9,29] + mem[9,29] >= 0
 accuracy_(2,_sk_9)[10]: - 1.69921875 frac[10,29] + mem[10,29] >= 0
 accuracy_(2,_sk_9)[11]: - 1.69921875 frac[11,29] + mem[11,29] >= 0
 accuracy_(2,_sk_9)[12]: - 1.69921875 frac[12,29] + mem[12,29] >= 0
 accuracy_(2,_sk_9)[13]: - 1.69921875 frac[13,29] + mem[13,29] >= 0
 accuracy_(2,_sk_9)[14]: - 1.69921875 frac[14,29] + mem[14,29] >= 0
 accuracy_(2,_sk_9)[15]: - 1.69921875 frac[15,29] + mem[15,29] >= 0
 accuracy_(2,_sk_9)[16]: - 1.69921875 frac[16,29] + mem[16,29] >= 0
 accuracy_(2,_sk_9)[17]: - 1.69921875 frac[17,29] + mem[17,29] >= 0
 accuracy_(2,_sk_9)[18]: - 1.69921875 frac[18,29] + mem[18,29] >= 0
 accuracy_(2,_sk_9)[19]: - 1.69921875 frac[19,29] + mem[19,29] >= 0
 accuracy_(2,_sk_9)[20]: - 1.69921875 frac[20,29] + mem[20,29] >= 0
 accuracy_(2,_sk_9)[21]: - 1.69921875 frac[21,29] + mem[21,29] >= 0
 accuracy_(2,_sk_9)[22]: - 1.69921875 frac[22,29] + mem[22,29] >= 0
 accuracy_(0,_sk_10)[0]: - 1.69921875 frac[0,30] + mem[0,30] >= 0
 accuracy_(0,_sk_10)[1]: - 1.69921875 frac[1,30] + mem[1,30] >= 0
 accuracy_(0,_sk_10)[2]: - 1.69921875 frac[2,30] + mem[2,30] >= 0
 accuracy_(0,_sk_10)[3]: - 1.69921875 frac[3,30] + mem[3,30] >= 0
 accuracy_(0,_sk_10)[4]: - 1.69921875 frac[4,30] + mem[4,30] >= 0
 accuracy_(0,_sk_10)[5]: - 1.69921875 frac[5,30] + mem[5,30] >= 0
 accuracy_(0,_sk_10)[6]: - 1.69921875 frac[6,30] + mem[6,30] >= 0
 accuracy_(0,_sk_10)[7]: - 1.69921875 frac[7,30] + mem[7,30] >= 0
 accuracy_(0,_sk_10)[8]: - 1.69921875 frac[8,30] + mem[8,30] >= 0
 accuracy_(0,_sk_10)[9]: - 1.69921875 frac[9,30] + mem[9,30] >= 0
 accuracy_(0,_sk_10)[10]: - 1.69921875 frac[10,30] + mem[10,30] >= 0
 accuracy_(0,_sk_10)[11]: - 1.69921875 frac[11,30] + mem[11,30] >= 0
 accuracy_(0,_sk_10)[12]: - 1.69921875 frac[12,30] + mem[12,30] >= 0
 accuracy_(0,_sk_10)[13]: - 1.69921875 frac[13,30] + mem[13,30] >= 0
 accuracy_(0,_sk_10)[14]: - 1.69921875 frac[14,30] + mem[14,30] >= 0
 accuracy_(0,_sk_10)[15]: - 1.69921875 frac[15,30] + mem[15,30] >= 0
 accuracy_(0,_sk_10)[16]: - 1.69921875 frac[16,30] + mem[16,30] >= 0
 accuracy_(0,_sk_10)[17]: - 1.69921875 frac[17,30] + mem[17,30] >= 0
 accuracy_(0,_sk_10)[18]: - 1.69921875 frac[18,30] + mem[18,30] >= 0
 accuracy_(0,_sk_10)[19]: - 1.69921875 frac[19,30] + mem[19,30] >= 0
 accuracy_(0,_sk_10)[20]: - 1.69921875 frac[20,30] + mem[20,30] >= 0
 accuracy_(0,_sk_10)[21]: - 1.69921875 frac[21,30] + mem[21,30] >= 0
 accuracy_(0,_sk_10)[22]: - 1.69921875 frac[22,30] + mem[22,30] >= 0
 accuracy_(1,_sk_10)[0]: - 1.69921875 frac[0,31] + mem[0,31] >= 0
 accuracy_(1,_sk_10)[1]: - 1.69921875 frac[1,31] + mem[1,31] >= 0
 accuracy_(1,_sk_10)[2]: - 1.69921875 frac[2,31] + mem[2,31] >= 0
 accuracy_(1,_sk_10)[3]: - 1.69921875 frac[3,31] + mem[3,31] >= 0
 accuracy_(1,_sk_10)[4]: - 1.69921875 frac[4,31] + mem[4,31] >= 0
 accuracy_(1,_sk_10)[5]: - 1.69921875 frac[5,31] + mem[5,31] >= 0
 accuracy_(1,_sk_10)[6]: - 1.69921875 frac[6,31] + mem[6,31] >= 0
 accuracy_(1,_sk_10)[7]: - 1.69921875 frac[7,31] + mem[7,31] >= 0
 accuracy_(1,_sk_10)[8]: - 1.69921875 frac[8,31] + mem[8,31] >= 0
 accuracy_(1,_sk_10)[9]: - 1.69921875 frac[9,31] + mem[9,31] >= 0
 accuracy_(1,_sk_10)[10]: - 1.69921875 frac[10,31] + mem[10,31] >= 0
 accuracy_(1,_sk_10)[11]: - 1.69921875 frac[11,31] + mem[11,31] >= 0
 accuracy_(1,_sk_10)[12]: - 1.69921875 frac[12,31] + mem[12,31] >= 0
 accuracy_(1,_sk_10)[13]: - 1.69921875 frac[13,31] + mem[13,31] >= 0
 accuracy_(1,_sk_10)[14]: - 1.69921875 frac[14,31] + mem[14,31] >= 0
 accuracy_(1,_sk_10)[15]: - 1.69921875 frac[15,31] + mem[15,31] >= 0
 accuracy_(1,_sk_10)[16]: - 1.69921875 frac[16,31] + mem[16,31] >= 0
 accuracy_(1,_sk_10)[17]: - 1.69921875 frac[17,31] + mem[17,31] >= 0
 accuracy_(1,_sk_10)[18]: - 1.69921875 frac[18,31] + mem[18,31] >= 0
 accuracy_(1,_sk_10)[19]: - 1.69921875 frac[19,31] + mem[19,31] >= 0
 accuracy_(1,_sk_10)[20]: - 1.69921875 frac[20,31] + mem[20,31] >= 0
 accuracy_(1,_sk_10)[21]: - 1.69921875 frac[21,31] + mem[21,31] >= 0
 accuracy_(1,_sk_10)[22]: - 1.69921875 frac[22,31] + mem[22,31] >= 0
 accuracy_(2,_sk_10)[0]: - 1.69921875 frac[0,32] + mem[0,32] >= 0
 accuracy_(2,_sk_10)[1]: - 1.69921875 frac[1,32] + mem[1,32] >= 0
 accuracy_(2,_sk_10)[2]: - 1.69921875 frac[2,32] + mem[2,32] >= 0
 accuracy_(2,_sk_10)[3]: - 1.69921875 frac[3,32] + mem[3,32] >= 0
 accuracy_(2,_sk_10)[4]: - 1.69921875 frac[4,32] + mem[4,32] >= 0
 accuracy_(2,_sk_10)[5]: - 1.69921875 frac[5,32] + mem[5,32] >= 0
 accuracy_(2,_sk_10)[6]: - 1.69921875 frac[6,32] + mem[6,32] >= 0
 accuracy_(2,_sk_10)[7]: - 1.69921875 frac[7,32] + mem[7,32] >= 0
 accuracy_(2,_sk_10)[8]: - 1.69921875 frac[8,32] + mem[8,32] >= 0
 accuracy_(2,_sk_10)[9]: - 1.69921875 frac[9,32] + mem[9,32] >= 0
 accuracy_(2,_sk_10)[10]: - 1.69921875 frac[10,32] + mem[10,32] >= 0
 accuracy_(2,_sk_10)[11]: - 1.69921875 frac[11,32] + mem[11,32] >= 0
 accuracy_(2,_sk_10)[12]: - 1.69921875 frac[12,32] + mem[12,32] >= 0
 accuracy_(2,_sk_10)[13]: - 1.69921875 frac[13,32] + mem[13,32] >= 0
 accuracy_(2,_sk_10)[14]: - 1.69921875 frac[14,32] + mem[14,32] >= 0
 accuracy_(2,_sk_10)[15]: - 1.69921875 frac[15,32] + mem[15,32] >= 0
 accuracy_(2,_sk_10)[16]: - 1.69921875 frac[16,32] + mem[16,32] >= 0
 accuracy_(2,_sk_10)[17]: - 1.69921875 frac[17,32] + mem[17,32] >= 0
 accuracy_(2,_sk_10)[18]: - 1.69921875 frac[18,32] + mem[18,32] >= 0
 accuracy_(2,_sk_10)[19]: - 1.69921875 frac[19,32] + mem[19,32] >= 0
 accuracy_(2,_sk_10)[20]: - 1.69921875 frac[20,32] + mem[20,32] >= 0
 accuracy_(2,_sk_10)[21]: - 1.69921875 frac[21,32] + mem[21,32] >= 0
 accuracy_(2,_sk_10)[22]: - 1.69921875 frac[22,32] + mem[22,32] >= 0
 accuracy_(0,_sk_11)[0]: - 1.69921875 frac[0,33] + mem[0,33] >= 0
 accuracy_(0,_sk_11)[1]: - 1.69921875 frac[1,33] + mem[1,33] >= 0
 accuracy_(0,_sk_11)[2]: - 1.69921875 frac[2,33] + mem[2,33] >= 0
 accuracy_(0,_sk_11)[3]: - 1.69921875 frac[3,33] + mem[3,33] >= 0
 accuracy_(0,_sk_11)[4]: - 1.69921875 frac[4,33] + mem[4,33] >= 0
 accuracy_(0,_sk_11)[5]: - 1.69921875 frac[5,33] + mem[5,33] >= 0
 accuracy_(0,_sk_11)[6]: - 1.69921875 frac[6,33] + mem[6,33] >= 0
 accuracy_(0,_sk_11)[7]: - 1.69921875 frac[7,33] + mem[7,33] >= 0
 accuracy_(0,_sk_11)[8]: - 1.69921875 frac[8,33] + mem[8,33] >= 0
 accuracy_(0,_sk_11)[9]: - 1.69921875 frac[9,33] + mem[9,33] >= 0
 accuracy_(0,_sk_11)[10]: - 1.69921875 frac[10,33] + mem[10,33] >= 0
 accuracy_(0,_sk_11)[11]: - 1.69921875 frac[11,33] + mem[11,33] >= 0
 accuracy_(0,_sk_11)[12]: - 1.69921875 frac[12,33] + mem[12,33] >= 0
 accuracy_(0,_sk_11)[13]: - 1.69921875 frac[13,33] + mem[13,33] >= 0
 accuracy_(0,_sk_11)[14]: - 1.69921875 frac[14,33] + mem[14,33] >= 0
 accuracy_(0,_sk_11)[15]: - 1.69921875 frac[15,33] + mem[15,33] >= 0
 accuracy_(0,_sk_11)[16]: - 1.69921875 frac[16,33] + mem[16,33] >= 0
 accuracy_(0,_sk_11)[17]: - 1.69921875 frac[17,33] + mem[17,33] >= 0
 accuracy_(0,_sk_11)[18]: - 1.69921875 frac[18,33] + mem[18,33] >= 0
 accuracy_(0,_sk_11)[19]: - 1.69921875 frac[19,33] + mem[19,33] >= 0
 accuracy_(0,_sk_11)[20]: - 1.69921875 frac[20,33] + mem[20,33] >= 0
 accuracy_(0,_sk_11)[21]: - 1.69921875 frac[21,33] + mem[21,33] >= 0
 accuracy_(0,_sk_11)[22]: - 1.69921875 frac[22,33] + mem[22,33] >= 0
 accuracy_(1,_sk_11)[0]: - 1.69921875 frac[0,34] + mem[0,34] >= 0
 accuracy_(1,_sk_11)[1]: - 1.69921875 frac[1,34] + mem[1,34] >= 0
 accuracy_(1,_sk_11)[2]: - 1.69921875 frac[2,34] + mem[2,34] >= 0
 accuracy_(1,_sk_11)[3]: - 1.69921875 frac[3,34] + mem[3,34] >= 0
 accuracy_(1,_sk_11)[4]: - 1.69921875 frac[4,34] + mem[4,34] >= 0
 accuracy_(1,_sk_11)[5]: - 1.69921875 frac[5,34] + mem[5,34] >= 0
 accuracy_(1,_sk_11)[6]: - 1.69921875 frac[6,34] + mem[6,34] >= 0
 accuracy_(1,_sk_11)[7]: - 1.69921875 frac[7,34] + mem[7,34] >= 0
 accuracy_(1,_sk_11)[8]: - 1.69921875 frac[8,34] + mem[8,34] >= 0
 accuracy_(1,_sk_11)[9]: - 1.69921875 frac[9,34] + mem[9,34] >= 0
 accuracy_(1,_sk_11)[10]: - 1.69921875 frac[10,34] + mem[10,34] >= 0
 accuracy_(1,_sk_11)[11]: - 1.69921875 frac[11,34] + mem[11,34] >= 0
 accuracy_(1,_sk_11)[12]: - 1.69921875 frac[12,34] + mem[12,34] >= 0
 accuracy_(1,_sk_11)[13]: - 1.69921875 frac[13,34] + mem[13,34] >= 0
 accuracy_(1,_sk_11)[14]: - 1.69921875 frac[14,34] + mem[14,34] >= 0
 accuracy_(1,_sk_11)[15]: - 1.69921875 frac[15,34] + mem[15,34] >= 0
 accuracy_(1,_sk_11)[16]: - 1.69921875 frac[16,34] + mem[16,34] >= 0
 accuracy_(1,_sk_11)[17]: - 1.69921875 frac[17,34] + mem[17,34] >= 0
 accuracy_(1,_sk_11)[18]: - 1.69921875 frac[18,34] + mem[18,34] >= 0
 accuracy_(1,_sk_11)[19]: - 1.69921875 frac[19,34] + mem[19,34] >= 0
 accuracy_(1,_sk_11)[20]: - 1.69921875 frac[20,34] + mem[20,34] >= 0
 accuracy_(1,_sk_11)[21]: - 1.69921875 frac[21,34] + mem[21,34] >= 0
 accuracy_(1,_sk_11)[22]: - 1.69921875 frac[22,34] + mem[22,34] >= 0
 accuracy_(2,_sk_11)[0]: - 1.69921875 frac[0,35] + mem[0,35] >= 0
 accuracy_(2,_sk_11)[1]: - 1.69921875 frac[1,35] + mem[1,35] >= 0
 accuracy_(2,_sk_11)[2]: - 1.69921875 frac[2,35] + mem[2,35] >= 0
 accuracy_(2,_sk_11)[3]: - 1.69921875 frac[3,35] + mem[3,35] >= 0
 accuracy_(2,_sk_11)[4]: - 1.69921875 frac[4,35] + mem[4,35] >= 0
 accuracy_(2,_sk_11)[5]: - 1.69921875 frac[5,35] + mem[5,35] >= 0
 accuracy_(2,_sk_11)[6]: - 1.69921875 frac[6,35] + mem[6,35] >= 0
 accuracy_(2,_sk_11)[7]: - 1.69921875 frac[7,35] + mem[7,35] >= 0
 accuracy_(2,_sk_11)[8]: - 1.69921875 frac[8,35] + mem[8,35] >= 0
 accuracy_(2,_sk_11)[9]: - 1.69921875 frac[9,35] + mem[9,35] >= 0
 accuracy_(2,_sk_11)[10]: - 1.69921875 frac[10,35] + mem[10,35] >= 0
 accuracy_(2,_sk_11)[11]: - 1.69921875 frac[11,35] + mem[11,35] >= 0
 accuracy_(2,_sk_11)[12]: - 1.69921875 frac[12,35] + mem[12,35] >= 0
 accuracy_(2,_sk_11)[13]: - 1.69921875 frac[13,35] + mem[13,35] >= 0
 accuracy_(2,_sk_11)[14]: - 1.69921875 frac[14,35] + mem[14,35] >= 0
 accuracy_(2,_sk_11)[15]: - 1.69921875 frac[15,35] + mem[15,35] >= 0
 accuracy_(2,_sk_11)[16]: - 1.69921875 frac[16,35] + mem[16,35] >= 0
 accuracy_(2,_sk_11)[17]: - 1.69921875 frac[17,35] + mem[17,35] >= 0
 accuracy_(2,_sk_11)[18]: - 1.69921875 frac[18,35] + mem[18,35] >= 0
 accuracy_(2,_sk_11)[19]: - 1.69921875 frac[19,35] + mem[19,35] >= 0
 accuracy_(2,_sk_11)[20]: - 1.69921875 frac[20,35] + mem[20,35] >= 0
 accuracy_(2,_sk_11)[21]: - 1.69921875 frac[21,35] + mem[21,35] >= 0
 accuracy_(2,_sk_11)[22]: - 1.69921875 frac[22,35] + mem[22,35] >= 0
 accuracy_(0,_sk_12)[0]: - 1.69921875 frac[0,36] + mem[0,36] >= 0
 accuracy_(0,_sk_12)[1]: - 1.69921875 frac[1,36] + mem[1,36] >= 0
 accuracy_(0,_sk_12)[2]: - 1.69921875 frac[2,36] + mem[2,36] >= 0
 accuracy_(0,_sk_12)[3]: - 1.69921875 frac[3,36] + mem[3,36] >= 0
 accuracy_(0,_sk_12)[4]: - 1.69921875 frac[4,36] + mem[4,36] >= 0
 accuracy_(0,_sk_12)[5]: - 1.69921875 frac[5,36] + mem[5,36] >= 0
 accuracy_(0,_sk_12)[6]: - 1.69921875 frac[6,36] + mem[6,36] >= 0
 accuracy_(0,_sk_12)[7]: - 1.69921875 frac[7,36] + mem[7,36] >= 0
 accuracy_(0,_sk_12)[8]: - 1.69921875 frac[8,36] + mem[8,36] >= 0
 accuracy_(0,_sk_12)[9]: - 1.69921875 frac[9,36] + mem[9,36] >= 0
 accuracy_(0,_sk_12)[10]: - 1.69921875 frac[10,36] + mem[10,36] >= 0
 accuracy_(0,_sk_12)[11]: - 1.69921875 frac[11,36] + mem[11,36] >= 0
 accuracy_(0,_sk_12)[12]: - 1.69921875 frac[12,36] + mem[12,36] >= 0
 accuracy_(0,_sk_12)[13]: - 1.69921875 frac[13,36] + mem[13,36] >= 0
 accuracy_(0,_sk_12)[14]: - 1.69921875 frac[14,36] + mem[14,36] >= 0
 accuracy_(0,_sk_12)[15]: - 1.69921875 frac[15,36] + mem[15,36] >= 0
 accuracy_(0,_sk_12)[16]: - 1.69921875 frac[16,36] + mem[16,36] >= 0
 accuracy_(0,_sk_12)[17]: - 1.69921875 frac[17,36] + mem[17,36] >= 0
 accuracy_(0,_sk_12)[18]: - 1.69921875 frac[18,36] + mem[18,36] >= 0
 accuracy_(0,_sk_12)[19]: - 1.69921875 frac[19,36] + mem[19,36] >= 0
 accuracy_(0,_sk_12)[20]: - 1.69921875 frac[20,36] + mem[20,36] >= 0
 accuracy_(0,_sk_12)[21]: - 1.69921875 frac[21,36] + mem[21,36] >= 0
 accuracy_(0,_sk_12)[22]: - 1.69921875 frac[22,36] + mem[22,36] >= 0
 accuracy_(1,_sk_12)[0]: - 1.69921875 frac[0,37] + mem[0,37] >= 0
 accuracy_(1,_sk_12)[1]: - 1.69921875 frac[1,37] + mem[1,37] >= 0
 accuracy_(1,_sk_12)[2]: - 1.69921875 frac[2,37] + mem[2,37] >= 0
 accuracy_(1,_sk_12)[3]: - 1.69921875 frac[3,37] + mem[3,37] >= 0
 accuracy_(1,_sk_12)[4]: - 1.69921875 frac[4,37] + mem[4,37] >= 0
 accuracy_(1,_sk_12)[5]: - 1.69921875 frac[5,37] + mem[5,37] >= 0
 accuracy_(1,_sk_12)[6]: - 1.69921875 frac[6,37] + mem[6,37] >= 0
 accuracy_(1,_sk_12)[7]: - 1.69921875 frac[7,37] + mem[7,37] >= 0
 accuracy_(1,_sk_12)[8]: - 1.69921875 frac[8,37] + mem[8,37] >= 0
 accuracy_(1,_sk_12)[9]: - 1.69921875 frac[9,37] + mem[9,37] >= 0
 accuracy_(1,_sk_12)[10]: - 1.69921875 frac[10,37] + mem[10,37] >= 0
 accuracy_(1,_sk_12)[11]: - 1.69921875 frac[11,37] + mem[11,37] >= 0
 accuracy_(1,_sk_12)[12]: - 1.69921875 frac[12,37] + mem[12,37] >= 0
 accuracy_(1,_sk_12)[13]: - 1.69921875 frac[13,37] + mem[13,37] >= 0
 accuracy_(1,_sk_12)[14]: - 1.69921875 frac[14,37] + mem[14,37] >= 0
 accuracy_(1,_sk_12)[15]: - 1.69921875 frac[15,37] + mem[15,37] >= 0
 accuracy_(1,_sk_12)[16]: - 1.69921875 frac[16,37] + mem[16,37] >= 0
 accuracy_(1,_sk_12)[17]: - 1.69921875 frac[17,37] + mem[17,37] >= 0
 accuracy_(1,_sk_12)[18]: - 1.69921875 frac[18,37] + mem[18,37] >= 0
 accuracy_(1,_sk_12)[19]: - 1.69921875 frac[19,37] + mem[19,37] >= 0
 accuracy_(1,_sk_12)[20]: - 1.69921875 frac[20,37] + mem[20,37] >= 0
 accuracy_(1,_sk_12)[21]: - 1.69921875 frac[21,37] + mem[21,37] >= 0
 accuracy_(1,_sk_12)[22]: - 1.69921875 frac[22,37] + mem[22,37] >= 0
 accuracy_(2,_sk_12)[0]: - 1.69921875 frac[0,38] + mem[0,38] >= 0
 accuracy_(2,_sk_12)[1]: - 1.69921875 frac[1,38] + mem[1,38] >= 0
 accuracy_(2,_sk_12)[2]: - 1.69921875 frac[2,38] + mem[2,38] >= 0
 accuracy_(2,_sk_12)[3]: - 1.69921875 frac[3,38] + mem[3,38] >= 0
 accuracy_(2,_sk_12)[4]: - 1.69921875 frac[4,38] + mem[4,38] >= 0
 accuracy_(2,_sk_12)[5]: - 1.69921875 frac[5,38] + mem[5,38] >= 0
 accuracy_(2,_sk_12)[6]: - 1.69921875 frac[6,38] + mem[6,38] >= 0
 accuracy_(2,_sk_12)[7]: - 1.69921875 frac[7,38] + mem[7,38] >= 0
 accuracy_(2,_sk_12)[8]: - 1.69921875 frac[8,38] + mem[8,38] >= 0
 accuracy_(2,_sk_12)[9]: - 1.69921875 frac[9,38] + mem[9,38] >= 0
 accuracy_(2,_sk_12)[10]: - 1.69921875 frac[10,38] + mem[10,38] >= 0
 accuracy_(2,_sk_12)[11]: - 1.69921875 frac[11,38] + mem[11,38] >= 0
 accuracy_(2,_sk_12)[12]: - 1.69921875 frac[12,38] + mem[12,38] >= 0
 accuracy_(2,_sk_12)[13]: - 1.69921875 frac[13,38] + mem[13,38] >= 0
 accuracy_(2,_sk_12)[14]: - 1.69921875 frac[14,38] + mem[14,38] >= 0
 accuracy_(2,_sk_12)[15]: - 1.69921875 frac[15,38] + mem[15,38] >= 0
 accuracy_(2,_sk_12)[16]: - 1.69921875 frac[16,38] + mem[16,38] >= 0
 accuracy_(2,_sk_12)[17]: - 1.69921875 frac[17,38] + mem[17,38] >= 0
 accuracy_(2,_sk_12)[18]: - 1.69921875 frac[18,38] + mem[18,38] >= 0
 accuracy_(2,_sk_12)[19]: - 1.69921875 frac[19,38] + mem[19,38] >= 0
 accuracy_(2,_sk_12)[20]: - 1.69921875 frac[20,38] + mem[20,38] >= 0
 accuracy_(2,_sk_12)[21]: - 1.69921875 frac[21,38] + mem[21,38] >= 0
 accuracy_(2,_sk_12)[22]: - 1.69921875 frac[22,38] + mem[22,38] >= 0
 accuracy_(0,_sk_13)[0]: - 1.69921875 frac[0,39] + mem[0,39] >= 0
 accuracy_(0,_sk_13)[1]: - 1.69921875 frac[1,39] + mem[1,39] >= 0
 accuracy_(0,_sk_13)[2]: - 1.69921875 frac[2,39] + mem[2,39] >= 0
 accuracy_(0,_sk_13)[3]: - 1.69921875 frac[3,39] + mem[3,39] >= 0
 accuracy_(0,_sk_13)[4]: - 1.69921875 frac[4,39] + mem[4,39] >= 0
 accuracy_(0,_sk_13)[5]: - 1.69921875 frac[5,39] + mem[5,39] >= 0
 accuracy_(0,_sk_13)[6]: - 1.69921875 frac[6,39] + mem[6,39] >= 0
 accuracy_(0,_sk_13)[7]: - 1.69921875 frac[7,39] + mem[7,39] >= 0
 accuracy_(0,_sk_13)[8]: - 1.69921875 frac[8,39] + mem[8,39] >= 0
 accuracy_(0,_sk_13)[9]: - 1.69921875 frac[9,39] + mem[9,39] >= 0
 accuracy_(0,_sk_13)[10]: - 1.69921875 frac[10,39] + mem[10,39] >= 0
 accuracy_(0,_sk_13)[11]: - 1.69921875 frac[11,39] + mem[11,39] >= 0
 accuracy_(0,_sk_13)[12]: - 1.69921875 frac[12,39] + mem[12,39] >= 0
 accuracy_(0,_sk_13)[13]: - 1.69921875 frac[13,39] + mem[13,39] >= 0
 accuracy_(0,_sk_13)[14]: - 1.69921875 frac[14,39] + mem[14,39] >= 0
 accuracy_(0,_sk_13)[15]: - 1.69921875 frac[15,39] + mem[15,39] >= 0
 accuracy_(0,_sk_13)[16]: - 1.69921875 frac[16,39] + mem[16,39] >= 0
 accuracy_(0,_sk_13)[17]: - 1.69921875 frac[17,39] + mem[17,39] >= 0
 accuracy_(0,_sk_13)[18]: - 1.69921875 frac[18,39] + mem[18,39] >= 0
 accuracy_(0,_sk_13)[19]: - 1.69921875 frac[19,39] + mem[19,39] >= 0
 accuracy_(0,_sk_13)[20]: - 1.69921875 frac[20,39] + mem[20,39] >= 0
 accuracy_(0,_sk_13)[21]: - 1.69921875 frac[21,39] + mem[21,39] >= 0
 accuracy_(0,_sk_13)[22]: - 1.69921875 frac[22,39] + mem[22,39] >= 0
 accuracy_(1,_sk_13)[0]: - 1.69921875 frac[0,40] + mem[0,40] >= 0
 accuracy_(1,_sk_13)[1]: - 1.69921875 frac[1,40] + mem[1,40] >= 0
 accuracy_(1,_sk_13)[2]: - 1.69921875 frac[2,40] + mem[2,40] >= 0
 accuracy_(1,_sk_13)[3]: - 1.69921875 frac[3,40] + mem[3,40] >= 0
 accuracy_(1,_sk_13)[4]: - 1.69921875 frac[4,40] + mem[4,40] >= 0
 accuracy_(1,_sk_13)[5]: - 1.69921875 frac[5,40] + mem[5,40] >= 0
 accuracy_(1,_sk_13)[6]: - 1.69921875 frac[6,40] + mem[6,40] >= 0
 accuracy_(1,_sk_13)[7]: - 1.69921875 frac[7,40] + mem[7,40] >= 0
 accuracy_(1,_sk_13)[8]: - 1.69921875 frac[8,40] + mem[8,40] >= 0
 accuracy_(1,_sk_13)[9]: - 1.69921875 frac[9,40] + mem[9,40] >= 0
 accuracy_(1,_sk_13)[10]: - 1.69921875 frac[10,40] + mem[10,40] >= 0
 accuracy_(1,_sk_13)[11]: - 1.69921875 frac[11,40] + mem[11,40] >= 0
 accuracy_(1,_sk_13)[12]: - 1.69921875 frac[12,40] + mem[12,40] >= 0
 accuracy_(1,_sk_13)[13]: - 1.69921875 frac[13,40] + mem[13,40] >= 0
 accuracy_(1,_sk_13)[14]: - 1.69921875 frac[14,40] + mem[14,40] >= 0
 accuracy_(1,_sk_13)[15]: - 1.69921875 frac[15,40] + mem[15,40] >= 0
 accuracy_(1,_sk_13)[16]: - 1.69921875 frac[16,40] + mem[16,40] >= 0
 accuracy_(1,_sk_13)[17]: - 1.69921875 frac[17,40] + mem[17,40] >= 0
 accuracy_(1,_sk_13)[18]: - 1.69921875 frac[18,40] + mem[18,40] >= 0
 accuracy_(1,_sk_13)[19]: - 1.69921875 frac[19,40] + mem[19,40] >= 0
 accuracy_(1,_sk_13)[20]: - 1.69921875 frac[20,40] + mem[20,40] >= 0
 accuracy_(1,_sk_13)[21]: - 1.69921875 frac[21,40] + mem[21,40] >= 0
 accuracy_(1,_sk_13)[22]: - 1.69921875 frac[22,40] + mem[22,40] >= 0
 accuracy_(2,_sk_13)[0]: - 1.69921875 frac[0,41] + mem[0,41] >= 0
 accuracy_(2,_sk_13)[1]: - 1.69921875 frac[1,41] + mem[1,41] >= 0
 accuracy_(2,_sk_13)[2]: - 1.69921875 frac[2,41] + mem[2,41] >= 0
 accuracy_(2,_sk_13)[3]: - 1.69921875 frac[3,41] + mem[3,41] >= 0
 accuracy_(2,_sk_13)[4]: - 1.69921875 frac[4,41] + mem[4,41] >= 0
 accuracy_(2,_sk_13)[5]: - 1.69921875 frac[5,41] + mem[5,41] >= 0
 accuracy_(2,_sk_13)[6]: - 1.69921875 frac[6,41] + mem[6,41] >= 0
 accuracy_(2,_sk_13)[7]: - 1.69921875 frac[7,41] + mem[7,41] >= 0
 accuracy_(2,_sk_13)[8]: - 1.69921875 frac[8,41] + mem[8,41] >= 0
 accuracy_(2,_sk_13)[9]: - 1.69921875 frac[9,41] + mem[9,41] >= 0
 accuracy_(2,_sk_13)[10]: - 1.69921875 frac[10,41] + mem[10,41] >= 0
 accuracy_(2,_sk_13)[11]: - 1.69921875 frac[11,41] + mem[11,41] >= 0
 accuracy_(2,_sk_13)[12]: - 1.69921875 frac[12,41] + mem[12,41] >= 0
 accuracy_(2,_sk_13)[13]: - 1.69921875 frac[13,41] + mem[13,41] >= 0
 accuracy_(2,_sk_13)[14]: - 1.69921875 frac[14,41] + mem[14,41] >= 0
 accuracy_(2,_sk_13)[15]: - 1.69921875 frac[15,41] + mem[15,41] >= 0
 accuracy_(2,_sk_13)[16]: - 1.69921875 frac[16,41] + mem[16,41] >= 0
 accuracy_(2,_sk_13)[17]: - 1.69921875 frac[17,41] + mem[17,41] >= 0
 accuracy_(2,_sk_13)[18]: - 1.69921875 frac[18,41] + mem[18,41] >= 0
 accuracy_(2,_sk_13)[19]: - 1.69921875 frac[19,41] + mem[19,41] >= 0
 accuracy_(2,_sk_13)[20]: - 1.69921875 frac[20,41] + mem[20,41] >= 0
 accuracy_(2,_sk_13)[21]: - 1.69921875 frac[21,41] + mem[21,41] >= 0
 accuracy_(2,_sk_13)[22]: - 1.69921875 frac[22,41] + mem[22,41] >= 0
 accuracy_(0,_sk_14)[0]: - 1.69921875 frac[0,42] + mem[0,42] >= 0
 accuracy_(0,_sk_14)[1]: - 1.69921875 frac[1,42] + mem[1,42] >= 0
 accuracy_(0,_sk_14)[2]: - 1.69921875 frac[2,42] + mem[2,42] >= 0
 accuracy_(0,_sk_14)[3]: - 1.69921875 frac[3,42] + mem[3,42] >= 0
 accuracy_(0,_sk_14)[4]: - 1.69921875 frac[4,42] + mem[4,42] >= 0
 accuracy_(0,_sk_14)[5]: - 1.69921875 frac[5,42] + mem[5,42] >= 0
 accuracy_(0,_sk_14)[6]: - 1.69921875 frac[6,42] + mem[6,42] >= 0
 accuracy_(0,_sk_14)[7]: - 1.69921875 frac[7,42] + mem[7,42] >= 0
 accuracy_(0,_sk_14)[8]: - 1.69921875 frac[8,42] + mem[8,42] >= 0
 accuracy_(0,_sk_14)[9]: - 1.69921875 frac[9,42] + mem[9,42] >= 0
 accuracy_(0,_sk_14)[10]: - 1.69921875 frac[10,42] + mem[10,42] >= 0
 accuracy_(0,_sk_14)[11]: - 1.69921875 frac[11,42] + mem[11,42] >= 0
 accuracy_(0,_sk_14)[12]: - 1.69921875 frac[12,42] + mem[12,42] >= 0
 accuracy_(0,_sk_14)[13]: - 1.69921875 frac[13,42] + mem[13,42] >= 0
 accuracy_(0,_sk_14)[14]: - 1.69921875 frac[14,42] + mem[14,42] >= 0
 accuracy_(0,_sk_14)[15]: - 1.69921875 frac[15,42] + mem[15,42] >= 0
 accuracy_(0,_sk_14)[16]: - 1.69921875 frac[16,42] + mem[16,42] >= 0
 accuracy_(0,_sk_14)[17]: - 1.69921875 frac[17,42] + mem[17,42] >= 0
 accuracy_(0,_sk_14)[18]: - 1.69921875 frac[18,42] + mem[18,42] >= 0
 accuracy_(0,_sk_14)[19]: - 1.69921875 frac[19,42] + mem[19,42] >= 0
 accuracy_(0,_sk_14)[20]: - 1.69921875 frac[20,42] + mem[20,42] >= 0
 accuracy_(0,_sk_14)[21]: - 1.69921875 frac[21,42] + mem[21,42] >= 0
 accuracy_(0,_sk_14)[22]: - 1.69921875 frac[22,42] + mem[22,42] >= 0
 accuracy_(1,_sk_14)[0]: - 1.69921875 frac[0,43] + mem[0,43] >= 0
 accuracy_(1,_sk_14)[1]: - 1.69921875 frac[1,43] + mem[1,43] >= 0
 accuracy_(1,_sk_14)[2]: - 1.69921875 frac[2,43] + mem[2,43] >= 0
 accuracy_(1,_sk_14)[3]: - 1.69921875 frac[3,43] + mem[3,43] >= 0
 accuracy_(1,_sk_14)[4]: - 1.69921875 frac[4,43] + mem[4,43] >= 0
 accuracy_(1,_sk_14)[5]: - 1.69921875 frac[5,43] + mem[5,43] >= 0
 accuracy_(1,_sk_14)[6]: - 1.69921875 frac[6,43] + mem[6,43] >= 0
 accuracy_(1,_sk_14)[7]: - 1.69921875 frac[7,43] + mem[7,43] >= 0
 accuracy_(1,_sk_14)[8]: - 1.69921875 frac[8,43] + mem[8,43] >= 0
 accuracy_(1,_sk_14)[9]: - 1.69921875 frac[9,43] + mem[9,43] >= 0
 accuracy_(1,_sk_14)[10]: - 1.69921875 frac[10,43] + mem[10,43] >= 0
 accuracy_(1,_sk_14)[11]: - 1.69921875 frac[11,43] + mem[11,43] >= 0
 accuracy_(1,_sk_14)[12]: - 1.69921875 frac[12,43] + mem[12,43] >= 0
 accuracy_(1,_sk_14)[13]: - 1.69921875 frac[13,43] + mem[13,43] >= 0
 accuracy_(1,_sk_14)[14]: - 1.69921875 frac[14,43] + mem[14,43] >= 0
 accuracy_(1,_sk_14)[15]: - 1.69921875 frac[15,43] + mem[15,43] >= 0
 accuracy_(1,_sk_14)[16]: - 1.69921875 frac[16,43] + mem[16,43] >= 0
 accuracy_(1,_sk_14)[17]: - 1.69921875 frac[17,43] + mem[17,43] >= 0
 accuracy_(1,_sk_14)[18]: - 1.69921875 frac[18,43] + mem[18,43] >= 0
 accuracy_(1,_sk_14)[19]: - 1.69921875 frac[19,43] + mem[19,43] >= 0
 accuracy_(1,_sk_14)[20]: - 1.69921875 frac[20,43] + mem[20,43] >= 0
 accuracy_(1,_sk_14)[21]: - 1.69921875 frac[21,43] + mem[21,43] >= 0
 accuracy_(1,_sk_14)[22]: - 1.69921875 frac[22,43] + mem[22,43] >= 0
 accuracy_(2,_sk_14)[0]: - 1.69921875 frac[0,44] + mem[0,44] >= 0
 accuracy_(2,_sk_14)[1]: - 1.69921875 frac[1,44] + mem[1,44] >= 0
 accuracy_(2,_sk_14)[2]: - 1.69921875 frac[2,44] + mem[2,44] >= 0
 accuracy_(2,_sk_14)[3]: - 1.69921875 frac[3,44] + mem[3,44] >= 0
 accuracy_(2,_sk_14)[4]: - 1.69921875 frac[4,44] + mem[4,44] >= 0
 accuracy_(2,_sk_14)[5]: - 1.69921875 frac[5,44] + mem[5,44] >= 0
 accuracy_(2,_sk_14)[6]: - 1.69921875 frac[6,44] + mem[6,44] >= 0
 accuracy_(2,_sk_14)[7]: - 1.69921875 frac[7,44] + mem[7,44] >= 0
 accuracy_(2,_sk_14)[8]: - 1.69921875 frac[8,44] + mem[8,44] >= 0
 accuracy_(2,_sk_14)[9]: - 1.69921875 frac[9,44] + mem[9,44] >= 0
 accuracy_(2,_sk_14)[10]: - 1.69921875 frac[10,44] + mem[10,44] >= 0
 accuracy_(2,_sk_14)[11]: - 1.69921875 frac[11,44] + mem[11,44] >= 0
 accuracy_(2,_sk_14)[12]: - 1.69921875 frac[12,44] + mem[12,44] >= 0
 accuracy_(2,_sk_14)[13]: - 1.69921875 frac[13,44] + mem[13,44] >= 0
 accuracy_(2,_sk_14)[14]: - 1.69921875 frac[14,44] + mem[14,44] >= 0
 accuracy_(2,_sk_14)[15]: - 1.69921875 frac[15,44] + mem[15,44] >= 0
 accuracy_(2,_sk_14)[16]: - 1.69921875 frac[16,44] + mem[16,44] >= 0
 accuracy_(2,_sk_14)[17]: - 1.69921875 frac[17,44] + mem[17,44] >= 0
 accuracy_(2,_sk_14)[18]: - 1.69921875 frac[18,44] + mem[18,44] >= 0
 accuracy_(2,_sk_14)[19]: - 1.69921875 frac[19,44] + mem[19,44] >= 0
 accuracy_(2,_sk_14)[20]: - 1.69921875 frac[20,44] + mem[20,44] >= 0
 accuracy_(2,_sk_14)[21]: - 1.69921875 frac[21,44] + mem[21,44] >= 0
 accuracy_(2,_sk_14)[22]: - 1.69921875 frac[22,44] + mem[22,44] >= 0
 accuracy_(0,_sk_15)[0]: - 1.69921875 frac[0,45] + mem[0,45] >= 0
 accuracy_(0,_sk_15)[1]: - 1.69921875 frac[1,45] + mem[1,45] >= 0
 accuracy_(0,_sk_15)[2]: - 1.69921875 frac[2,45] + mem[2,45] >= 0
 accuracy_(0,_sk_15)[3]: - 1.69921875 frac[3,45] + mem[3,45] >= 0
 accuracy_(0,_sk_15)[4]: - 1.69921875 frac[4,45] + mem[4,45] >= 0
 accuracy_(0,_sk_15)[5]: - 1.69921875 frac[5,45] + mem[5,45] >= 0
 accuracy_(0,_sk_15)[6]: - 1.69921875 frac[6,45] + mem[6,45] >= 0
 accuracy_(0,_sk_15)[7]: - 1.69921875 frac[7,45] + mem[7,45] >= 0
 accuracy_(0,_sk_15)[8]: - 1.69921875 frac[8,45] + mem[8,45] >= 0
 accuracy_(0,_sk_15)[9]: - 1.69921875 frac[9,45] + mem[9,45] >= 0
 accuracy_(0,_sk_15)[10]: - 1.69921875 frac[10,45] + mem[10,45] >= 0
 accuracy_(0,_sk_15)[11]: - 1.69921875 frac[11,45] + mem[11,45] >= 0
 accuracy_(0,_sk_15)[12]: - 1.69921875 frac[12,45] + mem[12,45] >= 0
 accuracy_(0,_sk_15)[13]: - 1.69921875 frac[13,45] + mem[13,45] >= 0
 accuracy_(0,_sk_15)[14]: - 1.69921875 frac[14,45] + mem[14,45] >= 0
 accuracy_(0,_sk_15)[15]: - 1.69921875 frac[15,45] + mem[15,45] >= 0
 accuracy_(0,_sk_15)[16]: - 1.69921875 frac[16,45] + mem[16,45] >= 0
 accuracy_(0,_sk_15)[17]: - 1.69921875 frac[17,45] + mem[17,45] >= 0
 accuracy_(0,_sk_15)[18]: - 1.69921875 frac[18,45] + mem[18,45] >= 0
 accuracy_(0,_sk_15)[19]: - 1.69921875 frac[19,45] + mem[19,45] >= 0
 accuracy_(0,_sk_15)[20]: - 1.69921875 frac[20,45] + mem[20,45] >= 0
 accuracy_(0,_sk_15)[21]: - 1.69921875 frac[21,45] + mem[21,45] >= 0
 accuracy_(0,_sk_15)[22]: - 1.69921875 frac[22,45] + mem[22,45] >= 0
 accuracy_(1,_sk_15)[0]: - 1.69921875 frac[0,46] + mem[0,46] >= 0
 accuracy_(1,_sk_15)[1]: - 1.69921875 frac[1,46] + mem[1,46] >= 0
 accuracy_(1,_sk_15)[2]: - 1.69921875 frac[2,46] + mem[2,46] >= 0
 accuracy_(1,_sk_15)[3]: - 1.69921875 frac[3,46] + mem[3,46] >= 0
 accuracy_(1,_sk_15)[4]: - 1.69921875 frac[4,46] + mem[4,46] >= 0
 accuracy_(1,_sk_15)[5]: - 1.69921875 frac[5,46] + mem[5,46] >= 0
 accuracy_(1,_sk_15)[6]: - 1.69921875 frac[6,46] + mem[6,46] >= 0
 accuracy_(1,_sk_15)[7]: - 1.69921875 frac[7,46] + mem[7,46] >= 0
 accuracy_(1,_sk_15)[8]: - 1.69921875 frac[8,46] + mem[8,46] >= 0
 accuracy_(1,_sk_15)[9]: - 1.69921875 frac[9,46] + mem[9,46] >= 0
 accuracy_(1,_sk_15)[10]: - 1.69921875 frac[10,46] + mem[10,46] >= 0
 accuracy_(1,_sk_15)[11]: - 1.69921875 frac[11,46] + mem[11,46] >= 0
 accuracy_(1,_sk_15)[12]: - 1.69921875 frac[12,46] + mem[12,46] >= 0
 accuracy_(1,_sk_15)[13]: - 1.69921875 frac[13,46] + mem[13,46] >= 0
 accuracy_(1,_sk_15)[14]: - 1.69921875 frac[14,46] + mem[14,46] >= 0
 accuracy_(1,_sk_15)[15]: - 1.69921875 frac[15,46] + mem[15,46] >= 0
 accuracy_(1,_sk_15)[16]: - 1.69921875 frac[16,46] + mem[16,46] >= 0
 accuracy_(1,_sk_15)[17]: - 1.69921875 frac[17,46] + mem[17,46] >= 0
 accuracy_(1,_sk_15)[18]: - 1.69921875 frac[18,46] + mem[18,46] >= 0
 accuracy_(1,_sk_15)[19]: - 1.69921875 frac[19,46] + mem[19,46] >= 0
 accuracy_(1,_sk_15)[20]: - 1.69921875 frac[20,46] + mem[20,46] >= 0
 accuracy_(1,_sk_15)[21]: - 1.69921875 frac[21,46] + mem[21,46] >= 0
 accuracy_(1,_sk_15)[22]: - 1.69921875 frac[22,46] + mem[22,46] >= 0
 accuracy_(2,_sk_15)[0]: - 1.69921875 frac[0,47] + mem[0,47] >= 0
 accuracy_(2,_sk_15)[1]: - 1.69921875 frac[1,47] + mem[1,47] >= 0
 accuracy_(2,_sk_15)[2]: - 1.69921875 frac[2,47] + mem[2,47] >= 0
 accuracy_(2,_sk_15)[3]: - 1.69921875 frac[3,47] + mem[3,47] >= 0
 accuracy_(2,_sk_15)[4]: - 1.69921875 frac[4,47] + mem[4,47] >= 0
 accuracy_(2,_sk_15)[5]: - 1.69921875 frac[5,47] + mem[5,47] >= 0
 accuracy_(2,_sk_15)[6]: - 1.69921875 frac[6,47] + mem[6,47] >= 0
 accuracy_(2,_sk_15)[7]: - 1.69921875 frac[7,47] + mem[7,47] >= 0
 accuracy_(2,_sk_15)[8]: - 1.69921875 frac[8,47] + mem[8,47] >= 0
 accuracy_(2,_sk_15)[9]: - 1.69921875 frac[9,47] + mem[9,47] >= 0
 accuracy_(2,_sk_15)[10]: - 1.69921875 frac[10,47] + mem[10,47] >= 0
 accuracy_(2,_sk_15)[11]: - 1.69921875 frac[11,47] + mem[11,47] >= 0
 accuracy_(2,_sk_15)[12]: - 1.69921875 frac[12,47] + mem[12,47] >= 0
 accuracy_(2,_sk_15)[13]: - 1.69921875 frac[13,47] + mem[13,47] >= 0
 accuracy_(2,_sk_15)[14]: - 1.69921875 frac[14,47] + mem[14,47] >= 0
 accuracy_(2,_sk_15)[15]: - 1.69921875 frac[15,47] + mem[15,47] >= 0
 accuracy_(2,_sk_15)[16]: - 1.69921875 frac[16,47] + mem[16,47] >= 0
 accuracy_(2,_sk_15)[17]: - 1.69921875 frac[17,47] + mem[17,47] >= 0
 accuracy_(2,_sk_15)[18]: - 1.69921875 frac[18,47] + mem[18,47] >= 0
 accuracy_(2,_sk_15)[19]: - 1.69921875 frac[19,47] + mem[19,47] >= 0
 accuracy_(2,_sk_15)[20]: - 1.69921875 frac[20,47] + mem[20,47] >= 0
 accuracy_(2,_sk_15)[21]: - 1.69921875 frac[21,47] + mem[21,47] >= 0
 accuracy_(2,_sk_15)[22]: - 1.69921875 frac[22,47] + mem[22,47] >= 0
 accuracy_(0,_sk_16)[0]: - 1.69921875 frac[0,48] + mem[0,48] >= 0
 accuracy_(0,_sk_16)[1]: - 1.69921875 frac[1,48] + mem[1,48] >= 0
 accuracy_(0,_sk_16)[2]: - 1.69921875 frac[2,48] + mem[2,48] >= 0
 accuracy_(0,_sk_16)[3]: - 1.69921875 frac[3,48] + mem[3,48] >= 0
 accuracy_(0,_sk_16)[4]: - 1.69921875 frac[4,48] + mem[4,48] >= 0
 accuracy_(0,_sk_16)[5]: - 1.69921875 frac[5,48] + mem[5,48] >= 0
 accuracy_(0,_sk_16)[6]: - 1.69921875 frac[6,48] + mem[6,48] >= 0
 accuracy_(0,_sk_16)[7]: - 1.69921875 frac[7,48] + mem[7,48] >= 0
 accuracy_(0,_sk_16)[8]: - 1.69921875 frac[8,48] + mem[8,48] >= 0
 accuracy_(0,_sk_16)[9]: - 1.69921875 frac[9,48] + mem[9,48] >= 0
 accuracy_(0,_sk_16)[10]: - 1.69921875 frac[10,48] + mem[10,48] >= 0
 accuracy_(0,_sk_16)[11]: - 1.69921875 frac[11,48] + mem[11,48] >= 0
 accuracy_(0,_sk_16)[12]: - 1.69921875 frac[12,48] + mem[12,48] >= 0
 accuracy_(0,_sk_16)[13]: - 1.69921875 frac[13,48] + mem[13,48] >= 0
 accuracy_(0,_sk_16)[14]: - 1.69921875 frac[14,48] + mem[14,48] >= 0
 accuracy_(0,_sk_16)[15]: - 1.69921875 frac[15,48] + mem[15,48] >= 0
 accuracy_(0,_sk_16)[16]: - 1.69921875 frac[16,48] + mem[16,48] >= 0
 accuracy_(0,_sk_16)[17]: - 1.69921875 frac[17,48] + mem[17,48] >= 0
 accuracy_(0,_sk_16)[18]: - 1.69921875 frac[18,48] + mem[18,48] >= 0
 accuracy_(0,_sk_16)[19]: - 1.69921875 frac[19,48] + mem[19,48] >= 0
 accuracy_(0,_sk_16)[20]: - 1.69921875 frac[20,48] + mem[20,48] >= 0
 accuracy_(0,_sk_16)[21]: - 1.69921875 frac[21,48] + mem[21,48] >= 0
 accuracy_(0,_sk_16)[22]: - 1.69921875 frac[22,48] + mem[22,48] >= 0
 accuracy_(1,_sk_16)[0]: - 1.69921875 frac[0,49] + mem[0,49] >= 0
 accuracy_(1,_sk_16)[1]: - 1.69921875 frac[1,49] + mem[1,49] >= 0
 accuracy_(1,_sk_16)[2]: - 1.69921875 frac[2,49] + mem[2,49] >= 0
 accuracy_(1,_sk_16)[3]: - 1.69921875 frac[3,49] + mem[3,49] >= 0
 accuracy_(1,_sk_16)[4]: - 1.69921875 frac[4,49] + mem[4,49] >= 0
 accuracy_(1,_sk_16)[5]: - 1.69921875 frac[5,49] + mem[5,49] >= 0
 accuracy_(1,_sk_16)[6]: - 1.69921875 frac[6,49] + mem[6,49] >= 0
 accuracy_(1,_sk_16)[7]: - 1.69921875 frac[7,49] + mem[7,49] >= 0
 accuracy_(1,_sk_16)[8]: - 1.69921875 frac[8,49] + mem[8,49] >= 0
 accuracy_(1,_sk_16)[9]: - 1.69921875 frac[9,49] + mem[9,49] >= 0
 accuracy_(1,_sk_16)[10]: - 1.69921875 frac[10,49] + mem[10,49] >= 0
 accuracy_(1,_sk_16)[11]: - 1.69921875 frac[11,49] + mem[11,49] >= 0
 accuracy_(1,_sk_16)[12]: - 1.69921875 frac[12,49] + mem[12,49] >= 0
 accuracy_(1,_sk_16)[13]: - 1.69921875 frac[13,49] + mem[13,49] >= 0
 accuracy_(1,_sk_16)[14]: - 1.69921875 frac[14,49] + mem[14,49] >= 0
 accuracy_(1,_sk_16)[15]: - 1.69921875 frac[15,49] + mem[15,49] >= 0
 accuracy_(1,_sk_16)[16]: - 1.69921875 frac[16,49] + mem[16,49] >= 0
 accuracy_(1,_sk_16)[17]: - 1.69921875 frac[17,49] + mem[17,49] >= 0
 accuracy_(1,_sk_16)[18]: - 1.69921875 frac[18,49] + mem[18,49] >= 0
 accuracy_(1,_sk_16)[19]: - 1.69921875 frac[19,49] + mem[19,49] >= 0
 accuracy_(1,_sk_16)[20]: - 1.69921875 frac[20,49] + mem[20,49] >= 0
 accuracy_(1,_sk_16)[21]: - 1.69921875 frac[21,49] + mem[21,49] >= 0
 accuracy_(1,_sk_16)[22]: - 1.69921875 frac[22,49] + mem[22,49] >= 0
 accuracy_(2,_sk_16)[0]: - 1.69921875 frac[0,50] + mem[0,50] >= 0
 accuracy_(2,_sk_16)[1]: - 1.69921875 frac[1,50] + mem[1,50] >= 0
 accuracy_(2,_sk_16)[2]: - 1.69921875 frac[2,50] + mem[2,50] >= 0
 accuracy_(2,_sk_16)[3]: - 1.69921875 frac[3,50] + mem[3,50] >= 0
 accuracy_(2,_sk_16)[4]: - 1.69921875 frac[4,50] + mem[4,50] >= 0
 accuracy_(2,_sk_16)[5]: - 1.69921875 frac[5,50] + mem[5,50] >= 0
 accuracy_(2,_sk_16)[6]: - 1.69921875 frac[6,50] + mem[6,50] >= 0
 accuracy_(2,_sk_16)[7]: - 1.69921875 frac[7,50] + mem[7,50] >= 0
 accuracy_(2,_sk_16)[8]: - 1.69921875 frac[8,50] + mem[8,50] >= 0
 accuracy_(2,_sk_16)[9]: - 1.69921875 frac[9,50] + mem[9,50] >= 0
 accuracy_(2,_sk_16)[10]: - 1.69921875 frac[10,50] + mem[10,50] >= 0
 accuracy_(2,_sk_16)[11]: - 1.69921875 frac[11,50] + mem[11,50] >= 0
 accuracy_(2,_sk_16)[12]: - 1.69921875 frac[12,50] + mem[12,50] >= 0
 accuracy_(2,_sk_16)[13]: - 1.69921875 frac[13,50] + mem[13,50] >= 0
 accuracy_(2,_sk_16)[14]: - 1.69921875 frac[14,50] + mem[14,50] >= 0
 accuracy_(2,_sk_16)[15]: - 1.69921875 frac[15,50] + mem[15,50] >= 0
 accuracy_(2,_sk_16)[16]: - 1.69921875 frac[16,50] + mem[16,50] >= 0
 accuracy_(2,_sk_16)[17]: - 1.69921875 frac[17,50] + mem[17,50] >= 0
 accuracy_(2,_sk_16)[18]: - 1.69921875 frac[18,50] + mem[18,50] >= 0
 accuracy_(2,_sk_16)[19]: - 1.69921875 frac[19,50] + mem[19,50] >= 0
 accuracy_(2,_sk_16)[20]: - 1.69921875 frac[20,50] + mem[20,50] >= 0
 accuracy_(2,_sk_16)[21]: - 1.69921875 frac[21,50] + mem[21,50] >= 0
 accuracy_(2,_sk_16)[22]: - 1.69921875 frac[22,50] + mem[22,50] >= 0
 accuracy_(0,_sk_17)[0]: - 1.69921875 frac[0,51] + mem[0,51] >= 0
 accuracy_(0,_sk_17)[1]: - 1.69921875 frac[1,51] + mem[1,51] >= 0
 accuracy_(0,_sk_17)[2]: - 1.69921875 frac[2,51] + mem[2,51] >= 0
 accuracy_(0,_sk_17)[3]: - 1.69921875 frac[3,51] + mem[3,51] >= 0
 accuracy_(0,_sk_17)[4]: - 1.69921875 frac[4,51] + mem[4,51] >= 0
 accuracy_(0,_sk_17)[5]: - 1.69921875 frac[5,51] + mem[5,51] >= 0
 accuracy_(0,_sk_17)[6]: - 1.69921875 frac[6,51] + mem[6,51] >= 0
 accuracy_(0,_sk_17)[7]: - 1.69921875 frac[7,51] + mem[7,51] >= 0
 accuracy_(0,_sk_17)[8]: - 1.69921875 frac[8,51] + mem[8,51] >= 0
 accuracy_(0,_sk_17)[9]: - 1.69921875 frac[9,51] + mem[9,51] >= 0
 accuracy_(0,_sk_17)[10]: - 1.69921875 frac[10,51] + mem[10,51] >= 0
 accuracy_(0,_sk_17)[11]: - 1.69921875 frac[11,51] + mem[11,51] >= 0
 accuracy_(0,_sk_17)[12]: - 1.69921875 frac[12,51] + mem[12,51] >= 0
 accuracy_(0,_sk_17)[13]: - 1.69921875 frac[13,51] + mem[13,51] >= 0
 accuracy_(0,_sk_17)[14]: - 1.69921875 frac[14,51] + mem[14,51] >= 0
 accuracy_(0,_sk_17)[15]: - 1.69921875 frac[15,51] + mem[15,51] >= 0
 accuracy_(0,_sk_17)[16]: - 1.69921875 frac[16,51] + mem[16,51] >= 0
 accuracy_(0,_sk_17)[17]: - 1.69921875 frac[17,51] + mem[17,51] >= 0
 accuracy_(0,_sk_17)[18]: - 1.69921875 frac[18,51] + mem[18,51] >= 0
 accuracy_(0,_sk_17)[19]: - 1.69921875 frac[19,51] + mem[19,51] >= 0
 accuracy_(0,_sk_17)[20]: - 1.69921875 frac[20,51] + mem[20,51] >= 0
 accuracy_(0,_sk_17)[21]: - 1.69921875 frac[21,51] + mem[21,51] >= 0
 accuracy_(0,_sk_17)[22]: - 1.69921875 frac[22,51] + mem[22,51] >= 0
 accuracy_(1,_sk_17)[0]: - 1.69921875 frac[0,52] + mem[0,52] >= 0
 accuracy_(1,_sk_17)[1]: - 1.69921875 frac[1,52] + mem[1,52] >= 0
 accuracy_(1,_sk_17)[2]: - 1.69921875 frac[2,52] + mem[2,52] >= 0
 accuracy_(1,_sk_17)[3]: - 1.69921875 frac[3,52] + mem[3,52] >= 0
 accuracy_(1,_sk_17)[4]: - 1.69921875 frac[4,52] + mem[4,52] >= 0
 accuracy_(1,_sk_17)[5]: - 1.69921875 frac[5,52] + mem[5,52] >= 0
 accuracy_(1,_sk_17)[6]: - 1.69921875 frac[6,52] + mem[6,52] >= 0
 accuracy_(1,_sk_17)[7]: - 1.69921875 frac[7,52] + mem[7,52] >= 0
 accuracy_(1,_sk_17)[8]: - 1.69921875 frac[8,52] + mem[8,52] >= 0
 accuracy_(1,_sk_17)[9]: - 1.69921875 frac[9,52] + mem[9,52] >= 0
 accuracy_(1,_sk_17)[10]: - 1.69921875 frac[10,52] + mem[10,52] >= 0
 accuracy_(1,_sk_17)[11]: - 1.69921875 frac[11,52] + mem[11,52] >= 0
 accuracy_(1,_sk_17)[12]: - 1.69921875 frac[12,52] + mem[12,52] >= 0
 accuracy_(1,_sk_17)[13]: - 1.69921875 frac[13,52] + mem[13,52] >= 0
 accuracy_(1,_sk_17)[14]: - 1.69921875 frac[14,52] + mem[14,52] >= 0
 accuracy_(1,_sk_17)[15]: - 1.69921875 frac[15,52] + mem[15,52] >= 0
 accuracy_(1,_sk_17)[16]: - 1.69921875 frac[16,52] + mem[16,52] >= 0
 accuracy_(1,_sk_17)[17]: - 1.69921875 frac[17,52] + mem[17,52] >= 0
 accuracy_(1,_sk_17)[18]: - 1.69921875 frac[18,52] + mem[18,52] >= 0
 accuracy_(1,_sk_17)[19]: - 1.69921875 frac[19,52] + mem[19,52] >= 0
 accuracy_(1,_sk_17)[20]: - 1.69921875 frac[20,52] + mem[20,52] >= 0
 accuracy_(1,_sk_17)[21]: - 1.69921875 frac[21,52] + mem[21,52] >= 0
 accuracy_(1,_sk_17)[22]: - 1.69921875 frac[22,52] + mem[22,52] >= 0
 accuracy_(2,_sk_17)[0]: - 1.69921875 frac[0,53] + mem[0,53] >= 0
 accuracy_(2,_sk_17)[1]: - 1.69921875 frac[1,53] + mem[1,53] >= 0
 accuracy_(2,_sk_17)[2]: - 1.69921875 frac[2,53] + mem[2,53] >= 0
 accuracy_(2,_sk_17)[3]: - 1.69921875 frac[3,53] + mem[3,53] >= 0
 accuracy_(2,_sk_17)[4]: - 1.69921875 frac[4,53] + mem[4,53] >= 0
 accuracy_(2,_sk_17)[5]: - 1.69921875 frac[5,53] + mem[5,53] >= 0
 accuracy_(2,_sk_17)[6]: - 1.69921875 frac[6,53] + mem[6,53] >= 0
 accuracy_(2,_sk_17)[7]: - 1.69921875 frac[7,53] + mem[7,53] >= 0
 accuracy_(2,_sk_17)[8]: - 1.69921875 frac[8,53] + mem[8,53] >= 0
 accuracy_(2,_sk_17)[9]: - 1.69921875 frac[9,53] + mem[9,53] >= 0
 accuracy_(2,_sk_17)[10]: - 1.69921875 frac[10,53] + mem[10,53] >= 0
 accuracy_(2,_sk_17)[11]: - 1.69921875 frac[11,53] + mem[11,53] >= 0
 accuracy_(2,_sk_17)[12]: - 1.69921875 frac[12,53] + mem[12,53] >= 0
 accuracy_(2,_sk_17)[13]: - 1.69921875 frac[13,53] + mem[13,53] >= 0
 accuracy_(2,_sk_17)[14]: - 1.69921875 frac[14,53] + mem[14,53] >= 0
 accuracy_(2,_sk_17)[15]: - 1.69921875 frac[15,53] + mem[15,53] >= 0
 accuracy_(2,_sk_17)[16]: - 1.69921875 frac[16,53] + mem[16,53] >= 0
 accuracy_(2,_sk_17)[17]: - 1.69921875 frac[17,53] + mem[17,53] >= 0
 accuracy_(2,_sk_17)[18]: - 1.69921875 frac[18,53] + mem[18,53] >= 0
 accuracy_(2,_sk_17)[19]: - 1.69921875 frac[19,53] + mem[19,53] >= 0
 accuracy_(2,_sk_17)[20]: - 1.69921875 frac[20,53] + mem[20,53] >= 0
 accuracy_(2,_sk_17)[21]: - 1.69921875 frac[21,53] + mem[21,53] >= 0
 accuracy_(2,_sk_17)[22]: - 1.69921875 frac[22,53] + mem[22,53] >= 0
 accuracy_(0,_sk_18)[0]: - 1.69921875 frac[0,54] + mem[0,54] >= 0
 accuracy_(0,_sk_18)[1]: - 1.69921875 frac[1,54] + mem[1,54] >= 0
 accuracy_(0,_sk_18)[2]: - 1.69921875 frac[2,54] + mem[2,54] >= 0
 accuracy_(0,_sk_18)[3]: - 1.69921875 frac[3,54] + mem[3,54] >= 0
 accuracy_(0,_sk_18)[4]: - 1.69921875 frac[4,54] + mem[4,54] >= 0
 accuracy_(0,_sk_18)[5]: - 1.69921875 frac[5,54] + mem[5,54] >= 0
 accuracy_(0,_sk_18)[6]: - 1.69921875 frac[6,54] + mem[6,54] >= 0
 accuracy_(0,_sk_18)[7]: - 1.69921875 frac[7,54] + mem[7,54] >= 0
 accuracy_(0,_sk_18)[8]: - 1.69921875 frac[8,54] + mem[8,54] >= 0
 accuracy_(0,_sk_18)[9]: - 1.69921875 frac[9,54] + mem[9,54] >= 0
 accuracy_(0,_sk_18)[10]: - 1.69921875 frac[10,54] + mem[10,54] >= 0
 accuracy_(0,_sk_18)[11]: - 1.69921875 frac[11,54] + mem[11,54] >= 0
 accuracy_(0,_sk_18)[12]: - 1.69921875 frac[12,54] + mem[12,54] >= 0
 accuracy_(0,_sk_18)[13]: - 1.69921875 frac[13,54] + mem[13,54] >= 0
 accuracy_(0,_sk_18)[14]: - 1.69921875 frac[14,54] + mem[14,54] >= 0
 accuracy_(0,_sk_18)[15]: - 1.69921875 frac[15,54] + mem[15,54] >= 0
 accuracy_(0,_sk_18)[16]: - 1.69921875 frac[16,54] + mem[16,54] >= 0
 accuracy_(0,_sk_18)[17]: - 1.69921875 frac[17,54] + mem[17,54] >= 0
 accuracy_(0,_sk_18)[18]: - 1.69921875 frac[18,54] + mem[18,54] >= 0
 accuracy_(0,_sk_18)[19]: - 1.69921875 frac[19,54] + mem[19,54] >= 0
 accuracy_(0,_sk_18)[20]: - 1.69921875 frac[20,54] + mem[20,54] >= 0
 accuracy_(0,_sk_18)[21]: - 1.69921875 frac[21,54] + mem[21,54] >= 0
 accuracy_(0,_sk_18)[22]: - 1.69921875 frac[22,54] + mem[22,54] >= 0
 accuracy_(1,_sk_18)[0]: - 1.69921875 frac[0,55] + mem[0,55] >= 0
 accuracy_(1,_sk_18)[1]: - 1.69921875 frac[1,55] + mem[1,55] >= 0
 accuracy_(1,_sk_18)[2]: - 1.69921875 frac[2,55] + mem[2,55] >= 0
 accuracy_(1,_sk_18)[3]: - 1.69921875 frac[3,55] + mem[3,55] >= 0
 accuracy_(1,_sk_18)[4]: - 1.69921875 frac[4,55] + mem[4,55] >= 0
 accuracy_(1,_sk_18)[5]: - 1.69921875 frac[5,55] + mem[5,55] >= 0
 accuracy_(1,_sk_18)[6]: - 1.69921875 frac[6,55] + mem[6,55] >= 0
 accuracy_(1,_sk_18)[7]: - 1.69921875 frac[7,55] + mem[7,55] >= 0
 accuracy_(1,_sk_18)[8]: - 1.69921875 frac[8,55] + mem[8,55] >= 0
 accuracy_(1,_sk_18)[9]: - 1.69921875 frac[9,55] + mem[9,55] >= 0
 accuracy_(1,_sk_18)[10]: - 1.69921875 frac[10,55] + mem[10,55] >= 0
 accuracy_(1,_sk_18)[11]: - 1.69921875 frac[11,55] + mem[11,55] >= 0
 accuracy_(1,_sk_18)[12]: - 1.69921875 frac[12,55] + mem[12,55] >= 0
 accuracy_(1,_sk_18)[13]: - 1.69921875 frac[13,55] + mem[13,55] >= 0
 accuracy_(1,_sk_18)[14]: - 1.69921875 frac[14,55] + mem[14,55] >= 0
 accuracy_(1,_sk_18)[15]: - 1.69921875 frac[15,55] + mem[15,55] >= 0
 accuracy_(1,_sk_18)[16]: - 1.69921875 frac[16,55] + mem[16,55] >= 0
 accuracy_(1,_sk_18)[17]: - 1.69921875 frac[17,55] + mem[17,55] >= 0
 accuracy_(1,_sk_18)[18]: - 1.69921875 frac[18,55] + mem[18,55] >= 0
 accuracy_(1,_sk_18)[19]: - 1.69921875 frac[19,55] + mem[19,55] >= 0
 accuracy_(1,_sk_18)[20]: - 1.69921875 frac[20,55] + mem[20,55] >= 0
 accuracy_(1,_sk_18)[21]: - 1.69921875 frac[21,55] + mem[21,55] >= 0
 accuracy_(1,_sk_18)[22]: - 1.69921875 frac[22,55] + mem[22,55] >= 0
 accuracy_(2,_sk_18)[0]: - 1.69921875 frac[0,56] + mem[0,56] >= 0
 accuracy_(2,_sk_18)[1]: - 1.69921875 frac[1,56] + mem[1,56] >= 0
 accuracy_(2,_sk_18)[2]: - 1.69921875 frac[2,56] + mem[2,56] >= 0
 accuracy_(2,_sk_18)[3]: - 1.69921875 frac[3,56] + mem[3,56] >= 0
 accuracy_(2,_sk_18)[4]: - 1.69921875 frac[4,56] + mem[4,56] >= 0
 accuracy_(2,_sk_18)[5]: - 1.69921875 frac[5,56] + mem[5,56] >= 0
 accuracy_(2,_sk_18)[6]: - 1.69921875 frac[6,56] + mem[6,56] >= 0
 accuracy_(2,_sk_18)[7]: - 1.69921875 frac[7,56] + mem[7,56] >= 0
 accuracy_(2,_sk_18)[8]: - 1.69921875 frac[8,56] + mem[8,56] >= 0
 accuracy_(2,_sk_18)[9]: - 1.69921875 frac[9,56] + mem[9,56] >= 0
 accuracy_(2,_sk_18)[10]: - 1.69921875 frac[10,56] + mem[10,56] >= 0
 accuracy_(2,_sk_18)[11]: - 1.69921875 frac[11,56] + mem[11,56] >= 0
 accuracy_(2,_sk_18)[12]: - 1.69921875 frac[12,56] + mem[12,56] >= 0
 accuracy_(2,_sk_18)[13]: - 1.69921875 frac[13,56] + mem[13,56] >= 0
 accuracy_(2,_sk_18)[14]: - 1.69921875 frac[14,56] + mem[14,56] >= 0
 accuracy_(2,_sk_18)[15]: - 1.69921875 frac[15,56] + mem[15,56] >= 0
 accuracy_(2,_sk_18)[16]: - 1.69921875 frac[16,56] + mem[16,56] >= 0
 accuracy_(2,_sk_18)[17]: - 1.69921875 frac[17,56] + mem[17,56] >= 0
 accuracy_(2,_sk_18)[18]: - 1.69921875 frac[18,56] + mem[18,56] >= 0
 accuracy_(2,_sk_18)[19]: - 1.69921875 frac[19,56] + mem[19,56] >= 0
 accuracy_(2,_sk_18)[20]: - 1.69921875 frac[20,56] + mem[20,56] >= 0
 accuracy_(2,_sk_18)[21]: - 1.69921875 frac[21,56] + mem[21,56] >= 0
 accuracy_(2,_sk_18)[22]: - 1.69921875 frac[22,56] + mem[22,56] >= 0
 accuracy_(0,_sk_19)[0]: - 1.69921875 frac[0,57] + mem[0,57] >= 0
 accuracy_(0,_sk_19)[1]: - 1.69921875 frac[1,57] + mem[1,57] >= 0
 accuracy_(0,_sk_19)[2]: - 1.69921875 frac[2,57] + mem[2,57] >= 0
 accuracy_(0,_sk_19)[3]: - 1.69921875 frac[3,57] + mem[3,57] >= 0
 accuracy_(0,_sk_19)[4]: - 1.69921875 frac[4,57] + mem[4,57] >= 0
 accuracy_(0,_sk_19)[5]: - 1.69921875 frac[5,57] + mem[5,57] >= 0
 accuracy_(0,_sk_19)[6]: - 1.69921875 frac[6,57] + mem[6,57] >= 0
 accuracy_(0,_sk_19)[7]: - 1.69921875 frac[7,57] + mem[7,57] >= 0
 accuracy_(0,_sk_19)[8]: - 1.69921875 frac[8,57] + mem[8,57] >= 0
 accuracy_(0,_sk_19)[9]: - 1.69921875 frac[9,57] + mem[9,57] >= 0
 accuracy_(0,_sk_19)[10]: - 1.69921875 frac[10,57] + mem[10,57] >= 0
 accuracy_(0,_sk_19)[11]: - 1.69921875 frac[11,57] + mem[11,57] >= 0
 accuracy_(0,_sk_19)[12]: - 1.69921875 frac[12,57] + mem[12,57] >= 0
 accuracy_(0,_sk_19)[13]: - 1.69921875 frac[13,57] + mem[13,57] >= 0
 accuracy_(0,_sk_19)[14]: - 1.69921875 frac[14,57] + mem[14,57] >= 0
 accuracy_(0,_sk_19)[15]: - 1.69921875 frac[15,57] + mem[15,57] >= 0
 accuracy_(0,_sk_19)[16]: - 1.69921875 frac[16,57] + mem[16,57] >= 0
 accuracy_(0,_sk_19)[17]: - 1.69921875 frac[17,57] + mem[17,57] >= 0
 accuracy_(0,_sk_19)[18]: - 1.69921875 frac[18,57] + mem[18,57] >= 0
 accuracy_(0,_sk_19)[19]: - 1.69921875 frac[19,57] + mem[19,57] >= 0
 accuracy_(0,_sk_19)[20]: - 1.69921875 frac[20,57] + mem[20,57] >= 0
 accuracy_(0,_sk_19)[21]: - 1.69921875 frac[21,57] + mem[21,57] >= 0
 accuracy_(0,_sk_19)[22]: - 1.69921875 frac[22,57] + mem[22,57] >= 0
 accuracy_(1,_sk_19)[0]: - 1.69921875 frac[0,58] + mem[0,58] >= 0
 accuracy_(1,_sk_19)[1]: - 1.69921875 frac[1,58] + mem[1,58] >= 0
 accuracy_(1,_sk_19)[2]: - 1.69921875 frac[2,58] + mem[2,58] >= 0
 accuracy_(1,_sk_19)[3]: - 1.69921875 frac[3,58] + mem[3,58] >= 0
 accuracy_(1,_sk_19)[4]: - 1.69921875 frac[4,58] + mem[4,58] >= 0
 accuracy_(1,_sk_19)[5]: - 1.69921875 frac[5,58] + mem[5,58] >= 0
 accuracy_(1,_sk_19)[6]: - 1.69921875 frac[6,58] + mem[6,58] >= 0
 accuracy_(1,_sk_19)[7]: - 1.69921875 frac[7,58] + mem[7,58] >= 0
 accuracy_(1,_sk_19)[8]: - 1.69921875 frac[8,58] + mem[8,58] >= 0
 accuracy_(1,_sk_19)[9]: - 1.69921875 frac[9,58] + mem[9,58] >= 0
 accuracy_(1,_sk_19)[10]: - 1.69921875 frac[10,58] + mem[10,58] >= 0
 accuracy_(1,_sk_19)[11]: - 1.69921875 frac[11,58] + mem[11,58] >= 0
 accuracy_(1,_sk_19)[12]: - 1.69921875 frac[12,58] + mem[12,58] >= 0
 accuracy_(1,_sk_19)[13]: - 1.69921875 frac[13,58] + mem[13,58] >= 0
 accuracy_(1,_sk_19)[14]: - 1.69921875 frac[14,58] + mem[14,58] >= 0
 accuracy_(1,_sk_19)[15]: - 1.69921875 frac[15,58] + mem[15,58] >= 0
 accuracy_(1,_sk_19)[16]: - 1.69921875 frac[16,58] + mem[16,58] >= 0
 accuracy_(1,_sk_19)[17]: - 1.69921875 frac[17,58] + mem[17,58] >= 0
 accuracy_(1,_sk_19)[18]: - 1.69921875 frac[18,58] + mem[18,58] >= 0
 accuracy_(1,_sk_19)[19]: - 1.69921875 frac[19,58] + mem[19,58] >= 0
 accuracy_(1,_sk_19)[20]: - 1.69921875 frac[20,58] + mem[20,58] >= 0
 accuracy_(1,_sk_19)[21]: - 1.69921875 frac[21,58] + mem[21,58] >= 0
 accuracy_(1,_sk_19)[22]: - 1.69921875 frac[22,58] + mem[22,58] >= 0
 accuracy_(2,_sk_19)[0]: - 1.69921875 frac[0,59] + mem[0,59] >= 0
 accuracy_(2,_sk_19)[1]: - 1.69921875 frac[1,59] + mem[1,59] >= 0
 accuracy_(2,_sk_19)[2]: - 1.69921875 frac[2,59] + mem[2,59] >= 0
 accuracy_(2,_sk_19)[3]: - 1.69921875 frac[3,59] + mem[3,59] >= 0
 accuracy_(2,_sk_19)[4]: - 1.69921875 frac[4,59] + mem[4,59] >= 0
 accuracy_(2,_sk_19)[5]: - 1.69921875 frac[5,59] + mem[5,59] >= 0
 accuracy_(2,_sk_19)[6]: - 1.69921875 frac[6,59] + mem[6,59] >= 0
 accuracy_(2,_sk_19)[7]: - 1.69921875 frac[7,59] + mem[7,59] >= 0
 accuracy_(2,_sk_19)[8]: - 1.69921875 frac[8,59] + mem[8,59] >= 0
 accuracy_(2,_sk_19)[9]: - 1.69921875 frac[9,59] + mem[9,59] >= 0
 accuracy_(2,_sk_19)[10]: - 1.69921875 frac[10,59] + mem[10,59] >= 0
 accuracy_(2,_sk_19)[11]: - 1.69921875 frac[11,59] + mem[11,59] >= 0
 accuracy_(2,_sk_19)[12]: - 1.69921875 frac[12,59] + mem[12,59] >= 0
 accuracy_(2,_sk_19)[13]: - 1.69921875 frac[13,59] + mem[13,59] >= 0
 accuracy_(2,_sk_19)[14]: - 1.69921875 frac[14,59] + mem[14,59] >= 0
 accuracy_(2,_sk_19)[15]: - 1.69921875 frac[15,59] + mem[15,59] >= 0
 accuracy_(2,_sk_19)[16]: - 1.69921875 frac[16,59] + mem[16,59] >= 0
 accuracy_(2,_sk_19)[17]: - 1.69921875 frac[17,59] + mem[17,59] >= 0
 accuracy_(2,_sk_19)[18]: - 1.69921875 frac[18,59] + mem[18,59] >= 0
 accuracy_(2,_sk_19)[19]: - 1.69921875 frac[19,59] + mem[19,59] >= 0
 accuracy_(2,_sk_19)[20]: - 1.69921875 frac[20,59] + mem[20,59] >= 0
 accuracy_(2,_sk_19)[21]: - 1.69921875 frac[21,59] + mem[21,59] >= 0
 accuracy_(2,_sk_19)[22]: - 1.69921875 frac[22,59] + mem[22,59] >= 0
 accuracy_(0,_sk_20)[0]: - 1.69921875 frac[0,60] + mem[0,60] >= 0
 accuracy_(0,_sk_20)[1]: - 1.69921875 frac[1,60] + mem[1,60] >= 0
 accuracy_(0,_sk_20)[2]: - 1.69921875 frac[2,60] + mem[2,60] >= 0
 accuracy_(0,_sk_20)[3]: - 1.69921875 frac[3,60] + mem[3,60] >= 0
 accuracy_(0,_sk_20)[4]: - 1.69921875 frac[4,60] + mem[4,60] >= 0
 accuracy_(0,_sk_20)[5]: - 1.69921875 frac[5,60] + mem[5,60] >= 0
 accuracy_(0,_sk_20)[6]: - 1.69921875 frac[6,60] + mem[6,60] >= 0
 accuracy_(0,_sk_20)[7]: - 1.69921875 frac[7,60] + mem[7,60] >= 0
 accuracy_(0,_sk_20)[8]: - 1.69921875 frac[8,60] + mem[8,60] >= 0
 accuracy_(0,_sk_20)[9]: - 1.69921875 frac[9,60] + mem[9,60] >= 0
 accuracy_(0,_sk_20)[10]: - 1.69921875 frac[10,60] + mem[10,60] >= 0
 accuracy_(0,_sk_20)[11]: - 1.69921875 frac[11,60] + mem[11,60] >= 0
 accuracy_(0,_sk_20)[12]: - 1.69921875 frac[12,60] + mem[12,60] >= 0
 accuracy_(0,_sk_20)[13]: - 1.69921875 frac[13,60] + mem[13,60] >= 0
 accuracy_(0,_sk_20)[14]: - 1.69921875 frac[14,60] + mem[14,60] >= 0
 accuracy_(0,_sk_20)[15]: - 1.69921875 frac[15,60] + mem[15,60] >= 0
 accuracy_(0,_sk_20)[16]: - 1.69921875 frac[16,60] + mem[16,60] >= 0
 accuracy_(0,_sk_20)[17]: - 1.69921875 frac[17,60] + mem[17,60] >= 0
 accuracy_(0,_sk_20)[18]: - 1.69921875 frac[18,60] + mem[18,60] >= 0
 accuracy_(0,_sk_20)[19]: - 1.69921875 frac[19,60] + mem[19,60] >= 0
 accuracy_(0,_sk_20)[20]: - 1.69921875 frac[20,60] + mem[20,60] >= 0
 accuracy_(0,_sk_20)[21]: - 1.69921875 frac[21,60] + mem[21,60] >= 0
 accuracy_(0,_sk_20)[22]: - 1.69921875 frac[22,60] + mem[22,60] >= 0
 accuracy_(1,_sk_20)[0]: - 1.69921875 frac[0,61] + mem[0,61] >= 0
 accuracy_(1,_sk_20)[1]: - 1.69921875 frac[1,61] + mem[1,61] >= 0
 accuracy_(1,_sk_20)[2]: - 1.69921875 frac[2,61] + mem[2,61] >= 0
 accuracy_(1,_sk_20)[3]: - 1.69921875 frac[3,61] + mem[3,61] >= 0
 accuracy_(1,_sk_20)[4]: - 1.69921875 frac[4,61] + mem[4,61] >= 0
 accuracy_(1,_sk_20)[5]: - 1.69921875 frac[5,61] + mem[5,61] >= 0
 accuracy_(1,_sk_20)[6]: - 1.69921875 frac[6,61] + mem[6,61] >= 0
 accuracy_(1,_sk_20)[7]: - 1.69921875 frac[7,61] + mem[7,61] >= 0
 accuracy_(1,_sk_20)[8]: - 1.69921875 frac[8,61] + mem[8,61] >= 0
 accuracy_(1,_sk_20)[9]: - 1.69921875 frac[9,61] + mem[9,61] >= 0
 accuracy_(1,_sk_20)[10]: - 1.69921875 frac[10,61] + mem[10,61] >= 0
 accuracy_(1,_sk_20)[11]: - 1.69921875 frac[11,61] + mem[11,61] >= 0
 accuracy_(1,_sk_20)[12]: - 1.69921875 frac[12,61] + mem[12,61] >= 0
 accuracy_(1,_sk_20)[13]: - 1.69921875 frac[13,61] + mem[13,61] >= 0
 accuracy_(1,_sk_20)[14]: - 1.69921875 frac[14,61] + mem[14,61] >= 0
 accuracy_(1,_sk_20)[15]: - 1.69921875 frac[15,61] + mem[15,61] >= 0
 accuracy_(1,_sk_20)[16]: - 1.69921875 frac[16,61] + mem[16,61] >= 0
 accuracy_(1,_sk_20)[17]: - 1.69921875 frac[17,61] + mem[17,61] >= 0
 accuracy_(1,_sk_20)[18]: - 1.69921875 frac[18,61] + mem[18,61] >= 0
 accuracy_(1,_sk_20)[19]: - 1.69921875 frac[19,61] + mem[19,61] >= 0
 accuracy_(1,_sk_20)[20]: - 1.69921875 frac[20,61] + mem[20,61] >= 0
 accuracy_(1,_sk_20)[21]: - 1.69921875 frac[21,61] + mem[21,61] >= 0
 accuracy_(1,_sk_20)[22]: - 1.69921875 frac[22,61] + mem[22,61] >= 0
 accuracy_(2,_sk_20)[0]: - 1.69921875 frac[0,62] + mem[0,62] >= 0
 accuracy_(2,_sk_20)[1]: - 1.69921875 frac[1,62] + mem[1,62] >= 0
 accuracy_(2,_sk_20)[2]: - 1.69921875 frac[2,62] + mem[2,62] >= 0
 accuracy_(2,_sk_20)[3]: - 1.69921875 frac[3,62] + mem[3,62] >= 0
 accuracy_(2,_sk_20)[4]: - 1.69921875 frac[4,62] + mem[4,62] >= 0
 accuracy_(2,_sk_20)[5]: - 1.69921875 frac[5,62] + mem[5,62] >= 0
 accuracy_(2,_sk_20)[6]: - 1.69921875 frac[6,62] + mem[6,62] >= 0
 accuracy_(2,_sk_20)[7]: - 1.69921875 frac[7,62] + mem[7,62] >= 0
 accuracy_(2,_sk_20)[8]: - 1.69921875 frac[8,62] + mem[8,62] >= 0
 accuracy_(2,_sk_20)[9]: - 1.69921875 frac[9,62] + mem[9,62] >= 0
 accuracy_(2,_sk_20)[10]: - 1.69921875 frac[10,62] + mem[10,62] >= 0
 accuracy_(2,_sk_20)[11]: - 1.69921875 frac[11,62] + mem[11,62] >= 0
 accuracy_(2,_sk_20)[12]: - 1.69921875 frac[12,62] + mem[12,62] >= 0
 accuracy_(2,_sk_20)[13]: - 1.69921875 frac[13,62] + mem[13,62] >= 0
 accuracy_(2,_sk_20)[14]: - 1.69921875 frac[14,62] + mem[14,62] >= 0
 accuracy_(2,_sk_20)[15]: - 1.69921875 frac[15,62] + mem[15,62] >= 0
 accuracy_(2,_sk_20)[16]: - 1.69921875 frac[16,62] + mem[16,62] >= 0
 accuracy_(2,_sk_20)[17]: - 1.69921875 frac[17,62] + mem[17,62] >= 0
 accuracy_(2,_sk_20)[18]: - 1.69921875 frac[18,62] + mem[18,62] >= 0
 accuracy_(2,_sk_20)[19]: - 1.69921875 frac[19,62] + mem[19,62] >= 0
 accuracy_(2,_sk_20)[20]: - 1.69921875 frac[20,62] + mem[20,62] >= 0
 accuracy_(2,_sk_20)[21]: - 1.69921875 frac[21,62] + mem[21,62] >= 0
 accuracy_(2,_sk_20)[22]: - 1.69921875 frac[22,62] + mem[22,62] >= 0
 accuracy_(0,_sk_21)[0]: - 1.69921875 frac[0,63] + mem[0,63] >= 0
 accuracy_(0,_sk_21)[1]: - 1.69921875 frac[1,63] + mem[1,63] >= 0
 accuracy_(0,_sk_21)[2]: - 1.69921875 frac[2,63] + mem[2,63] >= 0
 accuracy_(0,_sk_21)[3]: - 1.69921875 frac[3,63] + mem[3,63] >= 0
 accuracy_(0,_sk_21)[4]: - 1.69921875 frac[4,63] + mem[4,63] >= 0
 accuracy_(0,_sk_21)[5]: - 1.69921875 frac[5,63] + mem[5,63] >= 0
 accuracy_(0,_sk_21)[6]: - 1.69921875 frac[6,63] + mem[6,63] >= 0
 accuracy_(0,_sk_21)[7]: - 1.69921875 frac[7,63] + mem[7,63] >= 0
 accuracy_(0,_sk_21)[8]: - 1.69921875 frac[8,63] + mem[8,63] >= 0
 accuracy_(0,_sk_21)[9]: - 1.69921875 frac[9,63] + mem[9,63] >= 0
 accuracy_(0,_sk_21)[10]: - 1.69921875 frac[10,63] + mem[10,63] >= 0
 accuracy_(0,_sk_21)[11]: - 1.69921875 frac[11,63] + mem[11,63] >= 0
 accuracy_(0,_sk_21)[12]: - 1.69921875 frac[12,63] + mem[12,63] >= 0
 accuracy_(0,_sk_21)[13]: - 1.69921875 frac[13,63] + mem[13,63] >= 0
 accuracy_(0,_sk_21)[14]: - 1.69921875 frac[14,63] + mem[14,63] >= 0
 accuracy_(0,_sk_21)[15]: - 1.69921875 frac[15,63] + mem[15,63] >= 0
 accuracy_(0,_sk_21)[16]: - 1.69921875 frac[16,63] + mem[16,63] >= 0
 accuracy_(0,_sk_21)[17]: - 1.69921875 frac[17,63] + mem[17,63] >= 0
 accuracy_(0,_sk_21)[18]: - 1.69921875 frac[18,63] + mem[18,63] >= 0
 accuracy_(0,_sk_21)[19]: - 1.69921875 frac[19,63] + mem[19,63] >= 0
 accuracy_(0,_sk_21)[20]: - 1.69921875 frac[20,63] + mem[20,63] >= 0
 accuracy_(0,_sk_21)[21]: - 1.69921875 frac[21,63] + mem[21,63] >= 0
 accuracy_(0,_sk_21)[22]: - 1.69921875 frac[22,63] + mem[22,63] >= 0
 accuracy_(1,_sk_21)[0]: - 1.69921875 frac[0,64] + mem[0,64] >= 0
 accuracy_(1,_sk_21)[1]: - 1.69921875 frac[1,64] + mem[1,64] >= 0
 accuracy_(1,_sk_21)[2]: - 1.69921875 frac[2,64] + mem[2,64] >= 0
 accuracy_(1,_sk_21)[3]: - 1.69921875 frac[3,64] + mem[3,64] >= 0
 accuracy_(1,_sk_21)[4]: - 1.69921875 frac[4,64] + mem[4,64] >= 0
 accuracy_(1,_sk_21)[5]: - 1.69921875 frac[5,64] + mem[5,64] >= 0
 accuracy_(1,_sk_21)[6]: - 1.69921875 frac[6,64] + mem[6,64] >= 0
 accuracy_(1,_sk_21)[7]: - 1.69921875 frac[7,64] + mem[7,64] >= 0
 accuracy_(1,_sk_21)[8]: - 1.69921875 frac[8,64] + mem[8,64] >= 0
 accuracy_(1,_sk_21)[9]: - 1.69921875 frac[9,64] + mem[9,64] >= 0
 accuracy_(1,_sk_21)[10]: - 1.69921875 frac[10,64] + mem[10,64] >= 0
 accuracy_(1,_sk_21)[11]: - 1.69921875 frac[11,64] + mem[11,64] >= 0
 accuracy_(1,_sk_21)[12]: - 1.69921875 frac[12,64] + mem[12,64] >= 0
 accuracy_(1,_sk_21)[13]: - 1.69921875 frac[13,64] + mem[13,64] >= 0
 accuracy_(1,_sk_21)[14]: - 1.69921875 frac[14,64] + mem[14,64] >= 0
 accuracy_(1,_sk_21)[15]: - 1.69921875 frac[15,64] + mem[15,64] >= 0
 accuracy_(1,_sk_21)[16]: - 1.69921875 frac[16,64] + mem[16,64] >= 0
 accuracy_(1,_sk_21)[17]: - 1.69921875 frac[17,64] + mem[17,64] >= 0
 accuracy_(1,_sk_21)[18]: - 1.69921875 frac[18,64] + mem[18,64] >= 0
 accuracy_(1,_sk_21)[19]: - 1.69921875 frac[19,64] + mem[19,64] >= 0
 accuracy_(1,_sk_21)[20]: - 1.69921875 frac[20,64] + mem[20,64] >= 0
 accuracy_(1,_sk_21)[21]: - 1.69921875 frac[21,64] + mem[21,64] >= 0
 accuracy_(1,_sk_21)[22]: - 1.69921875 frac[22,64] + mem[22,64] >= 0
 accuracy_(2,_sk_21)[0]: - 1.69921875 frac[0,65] + mem[0,65] >= 0
 accuracy_(2,_sk_21)[1]: - 1.69921875 frac[1,65] + mem[1,65] >= 0
 accuracy_(2,_sk_21)[2]: - 1.69921875 frac[2,65] + mem[2,65] >= 0
 accuracy_(2,_sk_21)[3]: - 1.69921875 frac[3,65] + mem[3,65] >= 0
 accuracy_(2,_sk_21)[4]: - 1.69921875 frac[4,65] + mem[4,65] >= 0
 accuracy_(2,_sk_21)[5]: - 1.69921875 frac[5,65] + mem[5,65] >= 0
 accuracy_(2,_sk_21)[6]: - 1.69921875 frac[6,65] + mem[6,65] >= 0
 accuracy_(2,_sk_21)[7]: - 1.69921875 frac[7,65] + mem[7,65] >= 0
 accuracy_(2,_sk_21)[8]: - 1.69921875 frac[8,65] + mem[8,65] >= 0
 accuracy_(2,_sk_21)[9]: - 1.69921875 frac[9,65] + mem[9,65] >= 0
 accuracy_(2,_sk_21)[10]: - 1.69921875 frac[10,65] + mem[10,65] >= 0
 accuracy_(2,_sk_21)[11]: - 1.69921875 frac[11,65] + mem[11,65] >= 0
 accuracy_(2,_sk_21)[12]: - 1.69921875 frac[12,65] + mem[12,65] >= 0
 accuracy_(2,_sk_21)[13]: - 1.69921875 frac[13,65] + mem[13,65] >= 0
 accuracy_(2,_sk_21)[14]: - 1.69921875 frac[14,65] + mem[14,65] >= 0
 accuracy_(2,_sk_21)[15]: - 1.69921875 frac[15,65] + mem[15,65] >= 0
 accuracy_(2,_sk_21)[16]: - 1.69921875 frac[16,65] + mem[16,65] >= 0
 accuracy_(2,_sk_21)[17]: - 1.69921875 frac[17,65] + mem[17,65] >= 0
 accuracy_(2,_sk_21)[18]: - 1.69921875 frac[18,65] + mem[18,65] >= 0
 accuracy_(2,_sk_21)[19]: - 1.69921875 frac[19,65] + mem[19,65] >= 0
 accuracy_(2,_sk_21)[20]: - 1.69921875 frac[20,65] + mem[20,65] >= 0
 accuracy_(2,_sk_21)[21]: - 1.69921875 frac[21,65] + mem[21,65] >= 0
 accuracy_(2,_sk_21)[22]: - 1.69921875 frac[22,65] + mem[22,65] >= 0
 accuracy_(0,_sk_22)[0]: - 1.69921875 frac[0,66] + mem[0,66] >= 0
 accuracy_(0,_sk_22)[1]: - 1.69921875 frac[1,66] + mem[1,66] >= 0
 accuracy_(0,_sk_22)[2]: - 1.69921875 frac[2,66] + mem[2,66] >= 0
 accuracy_(0,_sk_22)[3]: - 1.69921875 frac[3,66] + mem[3,66] >= 0
 accuracy_(0,_sk_22)[4]: - 1.69921875 frac[4,66] + mem[4,66] >= 0
 accuracy_(0,_sk_22)[5]: - 1.69921875 frac[5,66] + mem[5,66] >= 0
 accuracy_(0,_sk_22)[6]: - 1.69921875 frac[6,66] + mem[6,66] >= 0
 accuracy_(0,_sk_22)[7]: - 1.69921875 frac[7,66] + mem[7,66] >= 0
 accuracy_(0,_sk_22)[8]: - 1.69921875 frac[8,66] + mem[8,66] >= 0
 accuracy_(0,_sk_22)[9]: - 1.69921875 frac[9,66] + mem[9,66] >= 0
 accuracy_(0,_sk_22)[10]: - 1.69921875 frac[10,66] + mem[10,66] >= 0
 accuracy_(0,_sk_22)[11]: - 1.69921875 frac[11,66] + mem[11,66] >= 0
 accuracy_(0,_sk_22)[12]: - 1.69921875 frac[12,66] + mem[12,66] >= 0
 accuracy_(0,_sk_22)[13]: - 1.69921875 frac[13,66] + mem[13,66] >= 0
 accuracy_(0,_sk_22)[14]: - 1.69921875 frac[14,66] + mem[14,66] >= 0
 accuracy_(0,_sk_22)[15]: - 1.69921875 frac[15,66] + mem[15,66] >= 0
 accuracy_(0,_sk_22)[16]: - 1.69921875 frac[16,66] + mem[16,66] >= 0
 accuracy_(0,_sk_22)[17]: - 1.69921875 frac[17,66] + mem[17,66] >= 0
 accuracy_(0,_sk_22)[18]: - 1.69921875 frac[18,66] + mem[18,66] >= 0
 accuracy_(0,_sk_22)[19]: - 1.69921875 frac[19,66] + mem[19,66] >= 0
 accuracy_(0,_sk_22)[20]: - 1.69921875 frac[20,66] + mem[20,66] >= 0
 accuracy_(0,_sk_22)[21]: - 1.69921875 frac[21,66] + mem[21,66] >= 0
 accuracy_(0,_sk_22)[22]: - 1.69921875 frac[22,66] + mem[22,66] >= 0
 accuracy_(1,_sk_22)[0]: - 1.69921875 frac[0,67] + mem[0,67] >= 0
 accuracy_(1,_sk_22)[1]: - 1.69921875 frac[1,67] + mem[1,67] >= 0
 accuracy_(1,_sk_22)[2]: - 1.69921875 frac[2,67] + mem[2,67] >= 0
 accuracy_(1,_sk_22)[3]: - 1.69921875 frac[3,67] + mem[3,67] >= 0
 accuracy_(1,_sk_22)[4]: - 1.69921875 frac[4,67] + mem[4,67] >= 0
 accuracy_(1,_sk_22)[5]: - 1.69921875 frac[5,67] + mem[5,67] >= 0
 accuracy_(1,_sk_22)[6]: - 1.69921875 frac[6,67] + mem[6,67] >= 0
 accuracy_(1,_sk_22)[7]: - 1.69921875 frac[7,67] + mem[7,67] >= 0
 accuracy_(1,_sk_22)[8]: - 1.69921875 frac[8,67] + mem[8,67] >= 0
 accuracy_(1,_sk_22)[9]: - 1.69921875 frac[9,67] + mem[9,67] >= 0
 accuracy_(1,_sk_22)[10]: - 1.69921875 frac[10,67] + mem[10,67] >= 0
 accuracy_(1,_sk_22)[11]: - 1.69921875 frac[11,67] + mem[11,67] >= 0
 accuracy_(1,_sk_22)[12]: - 1.69921875 frac[12,67] + mem[12,67] >= 0
 accuracy_(1,_sk_22)[13]: - 1.69921875 frac[13,67] + mem[13,67] >= 0
 accuracy_(1,_sk_22)[14]: - 1.69921875 frac[14,67] + mem[14,67] >= 0
 accuracy_(1,_sk_22)[15]: - 1.69921875 frac[15,67] + mem[15,67] >= 0
 accuracy_(1,_sk_22)[16]: - 1.69921875 frac[16,67] + mem[16,67] >= 0
 accuracy_(1,_sk_22)[17]: - 1.69921875 frac[17,67] + mem[17,67] >= 0
 accuracy_(1,_sk_22)[18]: - 1.69921875 frac[18,67] + mem[18,67] >= 0
 accuracy_(1,_sk_22)[19]: - 1.69921875 frac[19,67] + mem[19,67] >= 0
 accuracy_(1,_sk_22)[20]: - 1.69921875 frac[20,67] + mem[20,67] >= 0
 accuracy_(1,_sk_22)[21]: - 1.69921875 frac[21,67] + mem[21,67] >= 0
 accuracy_(1,_sk_22)[22]: - 1.69921875 frac[22,67] + mem[22,67] >= 0
 accuracy_(2,_sk_22)[0]: - 1.69921875 frac[0,68] + mem[0,68] >= 0
 accuracy_(2,_sk_22)[1]: - 1.69921875 frac[1,68] + mem[1,68] >= 0
 accuracy_(2,_sk_22)[2]: - 1.69921875 frac[2,68] + mem[2,68] >= 0
 accuracy_(2,_sk_22)[3]: - 1.69921875 frac[3,68] + mem[3,68] >= 0
 accuracy_(2,_sk_22)[4]: - 1.69921875 frac[4,68] + mem[4,68] >= 0
 accuracy_(2,_sk_22)[5]: - 1.69921875 frac[5,68] + mem[5,68] >= 0
 accuracy_(2,_sk_22)[6]: - 1.69921875 frac[6,68] + mem[6,68] >= 0
 accuracy_(2,_sk_22)[7]: - 1.69921875 frac[7,68] + mem[7,68] >= 0
 accuracy_(2,_sk_22)[8]: - 1.69921875 frac[8,68] + mem[8,68] >= 0
 accuracy_(2,_sk_22)[9]: - 1.69921875 frac[9,68] + mem[9,68] >= 0
 accuracy_(2,_sk_22)[10]: - 1.69921875 frac[10,68] + mem[10,68] >= 0
 accuracy_(2,_sk_22)[11]: - 1.69921875 frac[11,68] + mem[11,68] >= 0
 accuracy_(2,_sk_22)[12]: - 1.69921875 frac[12,68] + mem[12,68] >= 0
 accuracy_(2,_sk_22)[13]: - 1.69921875 frac[13,68] + mem[13,68] >= 0
 accuracy_(2,_sk_22)[14]: - 1.69921875 frac[14,68] + mem[14,68] >= 0
 accuracy_(2,_sk_22)[15]: - 1.69921875 frac[15,68] + mem[15,68] >= 0
 accuracy_(2,_sk_22)[16]: - 1.69921875 frac[16,68] + mem[16,68] >= 0
 accuracy_(2,_sk_22)[17]: - 1.69921875 frac[17,68] + mem[17,68] >= 0
 accuracy_(2,_sk_22)[18]: - 1.69921875 frac[18,68] + mem[18,68] >= 0
 accuracy_(2,_sk_22)[19]: - 1.69921875 frac[19,68] + mem[19,68] >= 0
 accuracy_(2,_sk_22)[20]: - 1.69921875 frac[20,68] + mem[20,68] >= 0
 accuracy_(2,_sk_22)[21]: - 1.69921875 frac[21,68] + mem[21,68] >= 0
 accuracy_(2,_sk_22)[22]: - 1.69921875 frac[22,68] + mem[22,68] >= 0
 accuracy_(0,_sk_23)[0]: - 1.69921875 frac[0,69] + mem[0,69] >= 0
 accuracy_(0,_sk_23)[1]: - 1.69921875 frac[1,69] + mem[1,69] >= 0
 accuracy_(0,_sk_23)[2]: - 1.69921875 frac[2,69] + mem[2,69] >= 0
 accuracy_(0,_sk_23)[3]: - 1.69921875 frac[3,69] + mem[3,69] >= 0
 accuracy_(0,_sk_23)[4]: - 1.69921875 frac[4,69] + mem[4,69] >= 0
 accuracy_(0,_sk_23)[5]: - 1.69921875 frac[5,69] + mem[5,69] >= 0
 accuracy_(0,_sk_23)[6]: - 1.69921875 frac[6,69] + mem[6,69] >= 0
 accuracy_(0,_sk_23)[7]: - 1.69921875 frac[7,69] + mem[7,69] >= 0
 accuracy_(0,_sk_23)[8]: - 1.69921875 frac[8,69] + mem[8,69] >= 0
 accuracy_(0,_sk_23)[9]: - 1.69921875 frac[9,69] + mem[9,69] >= 0
 accuracy_(0,_sk_23)[10]: - 1.69921875 frac[10,69] + mem[10,69] >= 0
 accuracy_(0,_sk_23)[11]: - 1.69921875 frac[11,69] + mem[11,69] >= 0
 accuracy_(0,_sk_23)[12]: - 1.69921875 frac[12,69] + mem[12,69] >= 0
 accuracy_(0,_sk_23)[13]: - 1.69921875 frac[13,69] + mem[13,69] >= 0
 accuracy_(0,_sk_23)[14]: - 1.69921875 frac[14,69] + mem[14,69] >= 0
 accuracy_(0,_sk_23)[15]: - 1.69921875 frac[15,69] + mem[15,69] >= 0
 accuracy_(0,_sk_23)[16]: - 1.69921875 frac[16,69] + mem[16,69] >= 0
 accuracy_(0,_sk_23)[17]: - 1.69921875 frac[17,69] + mem[17,69] >= 0
 accuracy_(0,_sk_23)[18]: - 1.69921875 frac[18,69] + mem[18,69] >= 0
 accuracy_(0,_sk_23)[19]: - 1.69921875 frac[19,69] + mem[19,69] >= 0
 accuracy_(0,_sk_23)[20]: - 1.69921875 frac[20,69] + mem[20,69] >= 0
 accuracy_(0,_sk_23)[21]: - 1.69921875 frac[21,69] + mem[21,69] >= 0
 accuracy_(0,_sk_23)[22]: - 1.69921875 frac[22,69] + mem[22,69] >= 0
 accuracy_(1,_sk_23)[0]: - 1.69921875 frac[0,70] + mem[0,70] >= 0
 accuracy_(1,_sk_23)[1]: - 1.69921875 frac[1,70] + mem[1,70] >= 0
 accuracy_(1,_sk_23)[2]: - 1.69921875 frac[2,70] + mem[2,70] >= 0
 accuracy_(1,_sk_23)[3]: - 1.69921875 frac[3,70] + mem[3,70] >= 0
 accuracy_(1,_sk_23)[4]: - 1.69921875 frac[4,70] + mem[4,70] >= 0
 accuracy_(1,_sk_23)[5]: - 1.69921875 frac[5,70] + mem[5,70] >= 0
 accuracy_(1,_sk_23)[6]: - 1.69921875 frac[6,70] + mem[6,70] >= 0
 accuracy_(1,_sk_23)[7]: - 1.69921875 frac[7,70] + mem[7,70] >= 0
 accuracy_(1,_sk_23)[8]: - 1.69921875 frac[8,70] + mem[8,70] >= 0
 accuracy_(1,_sk_23)[9]: - 1.69921875 frac[9,70] + mem[9,70] >= 0
 accuracy_(1,_sk_23)[10]: - 1.69921875 frac[10,70] + mem[10,70] >= 0
 accuracy_(1,_sk_23)[11]: - 1.69921875 frac[11,70] + mem[11,70] >= 0
 accuracy_(1,_sk_23)[12]: - 1.69921875 frac[12,70] + mem[12,70] >= 0
 accuracy_(1,_sk_23)[13]: - 1.69921875 frac[13,70] + mem[13,70] >= 0
 accuracy_(1,_sk_23)[14]: - 1.69921875 frac[14,70] + mem[14,70] >= 0
 accuracy_(1,_sk_23)[15]: - 1.69921875 frac[15,70] + mem[15,70] >= 0
 accuracy_(1,_sk_23)[16]: - 1.69921875 frac[16,70] + mem[16,70] >= 0
 accuracy_(1,_sk_23)[17]: - 1.69921875 frac[17,70] + mem[17,70] >= 0
 accuracy_(1,_sk_23)[18]: - 1.69921875 frac[18,70] + mem[18,70] >= 0
 accuracy_(1,_sk_23)[19]: - 1.69921875 frac[19,70] + mem[19,70] >= 0
 accuracy_(1,_sk_23)[20]: - 1.69921875 frac[20,70] + mem[20,70] >= 0
 accuracy_(1,_sk_23)[21]: - 1.69921875 frac[21,70] + mem[21,70] >= 0
 accuracy_(1,_sk_23)[22]: - 1.69921875 frac[22,70] + mem[22,70] >= 0
 accuracy_(2,_sk_23)[0]: - 1.69921875 frac[0,71] + mem[0,71] >= 0
 accuracy_(2,_sk_23)[1]: - 1.69921875 frac[1,71] + mem[1,71] >= 0
 accuracy_(2,_sk_23)[2]: - 1.69921875 frac[2,71] + mem[2,71] >= 0
 accuracy_(2,_sk_23)[3]: - 1.69921875 frac[3,71] + mem[3,71] >= 0
 accuracy_(2,_sk_23)[4]: - 1.69921875 frac[4,71] + mem[4,71] >= 0
 accuracy_(2,_sk_23)[5]: - 1.69921875 frac[5,71] + mem[5,71] >= 0
 accuracy_(2,_sk_23)[6]: - 1.69921875 frac[6,71] + mem[6,71] >= 0
 accuracy_(2,_sk_23)[7]: - 1.69921875 frac[7,71] + mem[7,71] >= 0
 accuracy_(2,_sk_23)[8]: - 1.69921875 frac[8,71] + mem[8,71] >= 0
 accuracy_(2,_sk_23)[9]: - 1.69921875 frac[9,71] + mem[9,71] >= 0
 accuracy_(2,_sk_23)[10]: - 1.69921875 frac[10,71] + mem[10,71] >= 0
 accuracy_(2,_sk_23)[11]: - 1.69921875 frac[11,71] + mem[11,71] >= 0
 accuracy_(2,_sk_23)[12]: - 1.69921875 frac[12,71] + mem[12,71] >= 0
 accuracy_(2,_sk_23)[13]: - 1.69921875 frac[13,71] + mem[13,71] >= 0
 accuracy_(2,_sk_23)[14]: - 1.69921875 frac[14,71] + mem[14,71] >= 0
 accuracy_(2,_sk_23)[15]: - 1.69921875 frac[15,71] + mem[15,71] >= 0
 accuracy_(2,_sk_23)[16]: - 1.69921875 frac[16,71] + mem[16,71] >= 0
 accuracy_(2,_sk_23)[17]: - 1.69921875 frac[17,71] + mem[17,71] >= 0
 accuracy_(2,_sk_23)[18]: - 1.69921875 frac[18,71] + mem[18,71] >= 0
 accuracy_(2,_sk_23)[19]: - 1.69921875 frac[19,71] + mem[19,71] >= 0
 accuracy_(2,_sk_23)[20]: - 1.69921875 frac[20,71] + mem[20,71] >= 0
 accuracy_(2,_sk_23)[21]: - 1.69921875 frac[21,71] + mem[21,71] >= 0
 accuracy_(2,_sk_23)[22]: - 1.69921875 frac[22,71] + mem[22,71] >= 0
 accuracy_(0,_sk_24)[0]: - 1.69921875 frac[0,72] + mem[0,72] >= 0
 accuracy_(0,_sk_24)[1]: - 1.69921875 frac[1,72] + mem[1,72] >= 0
 accuracy_(0,_sk_24)[2]: - 1.69921875 frac[2,72] + mem[2,72] >= 0
 accuracy_(0,_sk_24)[3]: - 1.69921875 frac[3,72] + mem[3,72] >= 0
 accuracy_(0,_sk_24)[4]: - 1.69921875 frac[4,72] + mem[4,72] >= 0
 accuracy_(0,_sk_24)[5]: - 1.69921875 frac[5,72] + mem[5,72] >= 0
 accuracy_(0,_sk_24)[6]: - 1.69921875 frac[6,72] + mem[6,72] >= 0
 accuracy_(0,_sk_24)[7]: - 1.69921875 frac[7,72] + mem[7,72] >= 0
 accuracy_(0,_sk_24)[8]: - 1.69921875 frac[8,72] + mem[8,72] >= 0
 accuracy_(0,_sk_24)[9]: - 1.69921875 frac[9,72] + mem[9,72] >= 0
 accuracy_(0,_sk_24)[10]: - 1.69921875 frac[10,72] + mem[10,72] >= 0
 accuracy_(0,_sk_24)[11]: - 1.69921875 frac[11,72] + mem[11,72] >= 0
 accuracy_(0,_sk_24)[12]: - 1.69921875 frac[12,72] + mem[12,72] >= 0
 accuracy_(0,_sk_24)[13]: - 1.69921875 frac[13,72] + mem[13,72] >= 0
 accuracy_(0,_sk_24)[14]: - 1.69921875 frac[14,72] + mem[14,72] >= 0
 accuracy_(0,_sk_24)[15]: - 1.69921875 frac[15,72] + mem[15,72] >= 0
 accuracy_(0,_sk_24)[16]: - 1.69921875 frac[16,72] + mem[16,72] >= 0
 accuracy_(0,_sk_24)[17]: - 1.69921875 frac[17,72] + mem[17,72] >= 0
 accuracy_(0,_sk_24)[18]: - 1.69921875 frac[18,72] + mem[18,72] >= 0
 accuracy_(0,_sk_24)[19]: - 1.69921875 frac[19,72] + mem[19,72] >= 0
 accuracy_(0,_sk_24)[20]: - 1.69921875 frac[20,72] + mem[20,72] >= 0
 accuracy_(0,_sk_24)[21]: - 1.69921875 frac[21,72] + mem[21,72] >= 0
 accuracy_(0,_sk_24)[22]: - 1.69921875 frac[22,72] + mem[22,72] >= 0
 accuracy_(1,_sk_24)[0]: - 1.69921875 frac[0,73] + mem[0,73] >= 0
 accuracy_(1,_sk_24)[1]: - 1.69921875 frac[1,73] + mem[1,73] >= 0
 accuracy_(1,_sk_24)[2]: - 1.69921875 frac[2,73] + mem[2,73] >= 0
 accuracy_(1,_sk_24)[3]: - 1.69921875 frac[3,73] + mem[3,73] >= 0
 accuracy_(1,_sk_24)[4]: - 1.69921875 frac[4,73] + mem[4,73] >= 0
 accuracy_(1,_sk_24)[5]: - 1.69921875 frac[5,73] + mem[5,73] >= 0
 accuracy_(1,_sk_24)[6]: - 1.69921875 frac[6,73] + mem[6,73] >= 0
 accuracy_(1,_sk_24)[7]: - 1.69921875 frac[7,73] + mem[7,73] >= 0
 accuracy_(1,_sk_24)[8]: - 1.69921875 frac[8,73] + mem[8,73] >= 0
 accuracy_(1,_sk_24)[9]: - 1.69921875 frac[9,73] + mem[9,73] >= 0
 accuracy_(1,_sk_24)[10]: - 1.69921875 frac[10,73] + mem[10,73] >= 0
 accuracy_(1,_sk_24)[11]: - 1.69921875 frac[11,73] + mem[11,73] >= 0
 accuracy_(1,_sk_24)[12]: - 1.69921875 frac[12,73] + mem[12,73] >= 0
 accuracy_(1,_sk_24)[13]: - 1.69921875 frac[13,73] + mem[13,73] >= 0
 accuracy_(1,_sk_24)[14]: - 1.69921875 frac[14,73] + mem[14,73] >= 0
 accuracy_(1,_sk_24)[15]: - 1.69921875 frac[15,73] + mem[15,73] >= 0
 accuracy_(1,_sk_24)[16]: - 1.69921875 frac[16,73] + mem[16,73] >= 0
 accuracy_(1,_sk_24)[17]: - 1.69921875 frac[17,73] + mem[17,73] >= 0
 accuracy_(1,_sk_24)[18]: - 1.69921875 frac[18,73] + mem[18,73] >= 0
 accuracy_(1,_sk_24)[19]: - 1.69921875 frac[19,73] + mem[19,73] >= 0
 accuracy_(1,_sk_24)[20]: - 1.69921875 frac[20,73] + mem[20,73] >= 0
 accuracy_(1,_sk_24)[21]: - 1.69921875 frac[21,73] + mem[21,73] >= 0
 accuracy_(1,_sk_24)[22]: - 1.69921875 frac[22,73] + mem[22,73] >= 0
 accuracy_(2,_sk_24)[0]: - 1.69921875 frac[0,74] + mem[0,74] >= 0
 accuracy_(2,_sk_24)[1]: - 1.69921875 frac[1,74] + mem[1,74] >= 0
 accuracy_(2,_sk_24)[2]: - 1.69921875 frac[2,74] + mem[2,74] >= 0
 accuracy_(2,_sk_24)[3]: - 1.69921875 frac[3,74] + mem[3,74] >= 0
 accuracy_(2,_sk_24)[4]: - 1.69921875 frac[4,74] + mem[4,74] >= 0
 accuracy_(2,_sk_24)[5]: - 1.69921875 frac[5,74] + mem[5,74] >= 0
 accuracy_(2,_sk_24)[6]: - 1.69921875 frac[6,74] + mem[6,74] >= 0
 accuracy_(2,_sk_24)[7]: - 1.69921875 frac[7,74] + mem[7,74] >= 0
 accuracy_(2,_sk_24)[8]: - 1.69921875 frac[8,74] + mem[8,74] >= 0
 accuracy_(2,_sk_24)[9]: - 1.69921875 frac[9,74] + mem[9,74] >= 0
 accuracy_(2,_sk_24)[10]: - 1.69921875 frac[10,74] + mem[10,74] >= 0
 accuracy_(2,_sk_24)[11]: - 1.69921875 frac[11,74] + mem[11,74] >= 0
 accuracy_(2,_sk_24)[12]: - 1.69921875 frac[12,74] + mem[12,74] >= 0
 accuracy_(2,_sk_24)[13]: - 1.69921875 frac[13,74] + mem[13,74] >= 0
 accuracy_(2,_sk_24)[14]: - 1.69921875 frac[14,74] + mem[14,74] >= 0
 accuracy_(2,_sk_24)[15]: - 1.69921875 frac[15,74] + mem[15,74] >= 0
 accuracy_(2,_sk_24)[16]: - 1.69921875 frac[16,74] + mem[16,74] >= 0
 accuracy_(2,_sk_24)[17]: - 1.69921875 frac[17,74] + mem[17,74] >= 0
 accuracy_(2,_sk_24)[18]: - 1.69921875 frac[18,74] + mem[18,74] >= 0
 accuracy_(2,_sk_24)[19]: - 1.69921875 frac[19,74] + mem[19,74] >= 0
 accuracy_(2,_sk_24)[20]: - 1.69921875 frac[20,74] + mem[20,74] >= 0
 accuracy_(2,_sk_24)[21]: - 1.69921875 frac[21,74] + mem[21,74] >= 0
 accuracy_(2,_sk_24)[22]: - 1.69921875 frac[22,74] + mem[22,74] >= 0
 accuracy_(0,_sk_25)[0]: - 1.69921875 frac[0,75] + mem[0,75] >= 0
 accuracy_(0,_sk_25)[1]: - 1.69921875 frac[1,75] + mem[1,75] >= 0
 accuracy_(0,_sk_25)[2]: - 1.69921875 frac[2,75] + mem[2,75] >= 0
 accuracy_(0,_sk_25)[3]: - 1.69921875 frac[3,75] + mem[3,75] >= 0
 accuracy_(0,_sk_25)[4]: - 1.69921875 frac[4,75] + mem[4,75] >= 0
 accuracy_(0,_sk_25)[5]: - 1.69921875 frac[5,75] + mem[5,75] >= 0
 accuracy_(0,_sk_25)[6]: - 1.69921875 frac[6,75] + mem[6,75] >= 0
 accuracy_(0,_sk_25)[7]: - 1.69921875 frac[7,75] + mem[7,75] >= 0
 accuracy_(0,_sk_25)[8]: - 1.69921875 frac[8,75] + mem[8,75] >= 0
 accuracy_(0,_sk_25)[9]: - 1.69921875 frac[9,75] + mem[9,75] >= 0
 accuracy_(0,_sk_25)[10]: - 1.69921875 frac[10,75] + mem[10,75] >= 0
 accuracy_(0,_sk_25)[11]: - 1.69921875 frac[11,75] + mem[11,75] >= 0
 accuracy_(0,_sk_25)[12]: - 1.69921875 frac[12,75] + mem[12,75] >= 0
 accuracy_(0,_sk_25)[13]: - 1.69921875 frac[13,75] + mem[13,75] >= 0
 accuracy_(0,_sk_25)[14]: - 1.69921875 frac[14,75] + mem[14,75] >= 0
 accuracy_(0,_sk_25)[15]: - 1.69921875 frac[15,75] + mem[15,75] >= 0
 accuracy_(0,_sk_25)[16]: - 1.69921875 frac[16,75] + mem[16,75] >= 0
 accuracy_(0,_sk_25)[17]: - 1.69921875 frac[17,75] + mem[17,75] >= 0
 accuracy_(0,_sk_25)[18]: - 1.69921875 frac[18,75] + mem[18,75] >= 0
 accuracy_(0,_sk_25)[19]: - 1.69921875 frac[19,75] + mem[19,75] >= 0
 accuracy_(0,_sk_25)[20]: - 1.69921875 frac[20,75] + mem[20,75] >= 0
 accuracy_(0,_sk_25)[21]: - 1.69921875 frac[21,75] + mem[21,75] >= 0
 accuracy_(0,_sk_25)[22]: - 1.69921875 frac[22,75] + mem[22,75] >= 0
 accuracy_(1,_sk_25)[0]: - 1.69921875 frac[0,76] + mem[0,76] >= 0
 accuracy_(1,_sk_25)[1]: - 1.69921875 frac[1,76] + mem[1,76] >= 0
 accuracy_(1,_sk_25)[2]: - 1.69921875 frac[2,76] + mem[2,76] >= 0
 accuracy_(1,_sk_25)[3]: - 1.69921875 frac[3,76] + mem[3,76] >= 0
 accuracy_(1,_sk_25)[4]: - 1.69921875 frac[4,76] + mem[4,76] >= 0
 accuracy_(1,_sk_25)[5]: - 1.69921875 frac[5,76] + mem[5,76] >= 0
 accuracy_(1,_sk_25)[6]: - 1.69921875 frac[6,76] + mem[6,76] >= 0
 accuracy_(1,_sk_25)[7]: - 1.69921875 frac[7,76] + mem[7,76] >= 0
 accuracy_(1,_sk_25)[8]: - 1.69921875 frac[8,76] + mem[8,76] >= 0
 accuracy_(1,_sk_25)[9]: - 1.69921875 frac[9,76] + mem[9,76] >= 0
 accuracy_(1,_sk_25)[10]: - 1.69921875 frac[10,76] + mem[10,76] >= 0
 accuracy_(1,_sk_25)[11]: - 1.69921875 frac[11,76] + mem[11,76] >= 0
 accuracy_(1,_sk_25)[12]: - 1.69921875 frac[12,76] + mem[12,76] >= 0
 accuracy_(1,_sk_25)[13]: - 1.69921875 frac[13,76] + mem[13,76] >= 0
 accuracy_(1,_sk_25)[14]: - 1.69921875 frac[14,76] + mem[14,76] >= 0
 accuracy_(1,_sk_25)[15]: - 1.69921875 frac[15,76] + mem[15,76] >= 0
 accuracy_(1,_sk_25)[16]: - 1.69921875 frac[16,76] + mem[16,76] >= 0
 accuracy_(1,_sk_25)[17]: - 1.69921875 frac[17,76] + mem[17,76] >= 0
 accuracy_(1,_sk_25)[18]: - 1.69921875 frac[18,76] + mem[18,76] >= 0
 accuracy_(1,_sk_25)[19]: - 1.69921875 frac[19,76] + mem[19,76] >= 0
 accuracy_(1,_sk_25)[20]: - 1.69921875 frac[20,76] + mem[20,76] >= 0
 accuracy_(1,_sk_25)[21]: - 1.69921875 frac[21,76] + mem[21,76] >= 0
 accuracy_(1,_sk_25)[22]: - 1.69921875 frac[22,76] + mem[22,76] >= 0
 accuracy_(2,_sk_25)[0]: - 1.69921875 frac[0,77] + mem[0,77] >= 0
 accuracy_(2,_sk_25)[1]: - 1.69921875 frac[1,77] + mem[1,77] >= 0
 accuracy_(2,_sk_25)[2]: - 1.69921875 frac[2,77] + mem[2,77] >= 0
 accuracy_(2,_sk_25)[3]: - 1.69921875 frac[3,77] + mem[3,77] >= 0
 accuracy_(2,_sk_25)[4]: - 1.69921875 frac[4,77] + mem[4,77] >= 0
 accuracy_(2,_sk_25)[5]: - 1.69921875 frac[5,77] + mem[5,77] >= 0
 accuracy_(2,_sk_25)[6]: - 1.69921875 frac[6,77] + mem[6,77] >= 0
 accuracy_(2,_sk_25)[7]: - 1.69921875 frac[7,77] + mem[7,77] >= 0
 accuracy_(2,_sk_25)[8]: - 1.69921875 frac[8,77] + mem[8,77] >= 0
 accuracy_(2,_sk_25)[9]: - 1.69921875 frac[9,77] + mem[9,77] >= 0
 accuracy_(2,_sk_25)[10]: - 1.69921875 frac[10,77] + mem[10,77] >= 0
 accuracy_(2,_sk_25)[11]: - 1.69921875 frac[11,77] + mem[11,77] >= 0
 accuracy_(2,_sk_25)[12]: - 1.69921875 frac[12,77] + mem[12,77] >= 0
 accuracy_(2,_sk_25)[13]: - 1.69921875 frac[13,77] + mem[13,77] >= 0
 accuracy_(2,_sk_25)[14]: - 1.69921875 frac[14,77] + mem[14,77] >= 0
 accuracy_(2,_sk_25)[15]: - 1.69921875 frac[15,77] + mem[15,77] >= 0
 accuracy_(2,_sk_25)[16]: - 1.69921875 frac[16,77] + mem[16,77] >= 0
 accuracy_(2,_sk_25)[17]: - 1.69921875 frac[17,77] + mem[17,77] >= 0
 accuracy_(2,_sk_25)[18]: - 1.69921875 frac[18,77] + mem[18,77] >= 0
 accuracy_(2,_sk_25)[19]: - 1.69921875 frac[19,77] + mem[19,77] >= 0
 accuracy_(2,_sk_25)[20]: - 1.69921875 frac[20,77] + mem[20,77] >= 0
 accuracy_(2,_sk_25)[21]: - 1.69921875 frac[21,77] + mem[21,77] >= 0
 accuracy_(2,_sk_25)[22]: - 1.69921875 frac[22,77] + mem[22,77] >= 0
 accuracy_(0,_sk_26)[0]: - 1.69921875 frac[0,78] + mem[0,78] >= 0
 accuracy_(0,_sk_26)[1]: - 1.69921875 frac[1,78] + mem[1,78] >= 0
 accuracy_(0,_sk_26)[2]: - 1.69921875 frac[2,78] + mem[2,78] >= 0
 accuracy_(0,_sk_26)[3]: - 1.69921875 frac[3,78] + mem[3,78] >= 0
 accuracy_(0,_sk_26)[4]: - 1.69921875 frac[4,78] + mem[4,78] >= 0
 accuracy_(0,_sk_26)[5]: - 1.69921875 frac[5,78] + mem[5,78] >= 0
 accuracy_(0,_sk_26)[6]: - 1.69921875 frac[6,78] + mem[6,78] >= 0
 accuracy_(0,_sk_26)[7]: - 1.69921875 frac[7,78] + mem[7,78] >= 0
 accuracy_(0,_sk_26)[8]: - 1.69921875 frac[8,78] + mem[8,78] >= 0
 accuracy_(0,_sk_26)[9]: - 1.69921875 frac[9,78] + mem[9,78] >= 0
 accuracy_(0,_sk_26)[10]: - 1.69921875 frac[10,78] + mem[10,78] >= 0
 accuracy_(0,_sk_26)[11]: - 1.69921875 frac[11,78] + mem[11,78] >= 0
 accuracy_(0,_sk_26)[12]: - 1.69921875 frac[12,78] + mem[12,78] >= 0
 accuracy_(0,_sk_26)[13]: - 1.69921875 frac[13,78] + mem[13,78] >= 0
 accuracy_(0,_sk_26)[14]: - 1.69921875 frac[14,78] + mem[14,78] >= 0
 accuracy_(0,_sk_26)[15]: - 1.69921875 frac[15,78] + mem[15,78] >= 0
 accuracy_(0,_sk_26)[16]: - 1.69921875 frac[16,78] + mem[16,78] >= 0
 accuracy_(0,_sk_26)[17]: - 1.69921875 frac[17,78] + mem[17,78] >= 0
 accuracy_(0,_sk_26)[18]: - 1.69921875 frac[18,78] + mem[18,78] >= 0
 accuracy_(0,_sk_26)[19]: - 1.69921875 frac[19,78] + mem[19,78] >= 0
 accuracy_(0,_sk_26)[20]: - 1.69921875 frac[20,78] + mem[20,78] >= 0
 accuracy_(0,_sk_26)[21]: - 1.69921875 frac[21,78] + mem[21,78] >= 0
 accuracy_(0,_sk_26)[22]: - 1.69921875 frac[22,78] + mem[22,78] >= 0
 accuracy_(1,_sk_26)[0]: - 1.69921875 frac[0,79] + mem[0,79] >= 0
 accuracy_(1,_sk_26)[1]: - 1.69921875 frac[1,79] + mem[1,79] >= 0
 accuracy_(1,_sk_26)[2]: - 1.69921875 frac[2,79] + mem[2,79] >= 0
 accuracy_(1,_sk_26)[3]: - 1.69921875 frac[3,79] + mem[3,79] >= 0
 accuracy_(1,_sk_26)[4]: - 1.69921875 frac[4,79] + mem[4,79] >= 0
 accuracy_(1,_sk_26)[5]: - 1.69921875 frac[5,79] + mem[5,79] >= 0
 accuracy_(1,_sk_26)[6]: - 1.69921875 frac[6,79] + mem[6,79] >= 0
 accuracy_(1,_sk_26)[7]: - 1.69921875 frac[7,79] + mem[7,79] >= 0
 accuracy_(1,_sk_26)[8]: - 1.69921875 frac[8,79] + mem[8,79] >= 0
 accuracy_(1,_sk_26)[9]: - 1.69921875 frac[9,79] + mem[9,79] >= 0
 accuracy_(1,_sk_26)[10]: - 1.69921875 frac[10,79] + mem[10,79] >= 0
 accuracy_(1,_sk_26)[11]: - 1.69921875 frac[11,79] + mem[11,79] >= 0
 accuracy_(1,_sk_26)[12]: - 1.69921875 frac[12,79] + mem[12,79] >= 0
 accuracy_(1,_sk_26)[13]: - 1.69921875 frac[13,79] + mem[13,79] >= 0
 accuracy_(1,_sk_26)[14]: - 1.69921875 frac[14,79] + mem[14,79] >= 0
 accuracy_(1,_sk_26)[15]: - 1.69921875 frac[15,79] + mem[15,79] >= 0
 accuracy_(1,_sk_26)[16]: - 1.69921875 frac[16,79] + mem[16,79] >= 0
 accuracy_(1,_sk_26)[17]: - 1.69921875 frac[17,79] + mem[17,79] >= 0
 accuracy_(1,_sk_26)[18]: - 1.69921875 frac[18,79] + mem[18,79] >= 0
 accuracy_(1,_sk_26)[19]: - 1.69921875 frac[19,79] + mem[19,79] >= 0
 accuracy_(1,_sk_26)[20]: - 1.69921875 frac[20,79] + mem[20,79] >= 0
 accuracy_(1,_sk_26)[21]: - 1.69921875 frac[21,79] + mem[21,79] >= 0
 accuracy_(1,_sk_26)[22]: - 1.69921875 frac[22,79] + mem[22,79] >= 0
 accuracy_(2,_sk_26)[0]: - 1.69921875 frac[0,80] + mem[0,80] >= 0
 accuracy_(2,_sk_26)[1]: - 1.69921875 frac[1,80] + mem[1,80] >= 0
 accuracy_(2,_sk_26)[2]: - 1.69921875 frac[2,80] + mem[2,80] >= 0
 accuracy_(2,_sk_26)[3]: - 1.69921875 frac[3,80] + mem[3,80] >= 0
 accuracy_(2,_sk_26)[4]: - 1.69921875 frac[4,80] + mem[4,80] >= 0
 accuracy_(2,_sk_26)[5]: - 1.69921875 frac[5,80] + mem[5,80] >= 0
 accuracy_(2,_sk_26)[6]: - 1.69921875 frac[6,80] + mem[6,80] >= 0
 accuracy_(2,_sk_26)[7]: - 1.69921875 frac[7,80] + mem[7,80] >= 0
 accuracy_(2,_sk_26)[8]: - 1.69921875 frac[8,80] + mem[8,80] >= 0
 accuracy_(2,_sk_26)[9]: - 1.69921875 frac[9,80] + mem[9,80] >= 0
 accuracy_(2,_sk_26)[10]: - 1.69921875 frac[10,80] + mem[10,80] >= 0
 accuracy_(2,_sk_26)[11]: - 1.69921875 frac[11,80] + mem[11,80] >= 0
 accuracy_(2,_sk_26)[12]: - 1.69921875 frac[12,80] + mem[12,80] >= 0
 accuracy_(2,_sk_26)[13]: - 1.69921875 frac[13,80] + mem[13,80] >= 0
 accuracy_(2,_sk_26)[14]: - 1.69921875 frac[14,80] + mem[14,80] >= 0
 accuracy_(2,_sk_26)[15]: - 1.69921875 frac[15,80] + mem[15,80] >= 0
 accuracy_(2,_sk_26)[16]: - 1.69921875 frac[16,80] + mem[16,80] >= 0
 accuracy_(2,_sk_26)[17]: - 1.69921875 frac[17,80] + mem[17,80] >= 0
 accuracy_(2,_sk_26)[18]: - 1.69921875 frac[18,80] + mem[18,80] >= 0
 accuracy_(2,_sk_26)[19]: - 1.69921875 frac[19,80] + mem[19,80] >= 0
 accuracy_(2,_sk_26)[20]: - 1.69921875 frac[20,80] + mem[20,80] >= 0
 accuracy_(2,_sk_26)[21]: - 1.69921875 frac[21,80] + mem[21,80] >= 0
 accuracy_(2,_sk_26)[22]: - 1.69921875 frac[22,80] + mem[22,80] >= 0
 accuracy_(0,_sk_27)[0]: - 1.69921875 frac[0,81] + mem[0,81] >= 0
 accuracy_(0,_sk_27)[1]: - 1.69921875 frac[1,81] + mem[1,81] >= 0
 accuracy_(0,_sk_27)[2]: - 1.69921875 frac[2,81] + mem[2,81] >= 0
 accuracy_(0,_sk_27)[3]: - 1.69921875 frac[3,81] + mem[3,81] >= 0
 accuracy_(0,_sk_27)[4]: - 1.69921875 frac[4,81] + mem[4,81] >= 0
 accuracy_(0,_sk_27)[5]: - 1.69921875 frac[5,81] + mem[5,81] >= 0
 accuracy_(0,_sk_27)[6]: - 1.69921875 frac[6,81] + mem[6,81] >= 0
 accuracy_(0,_sk_27)[7]: - 1.69921875 frac[7,81] + mem[7,81] >= 0
 accuracy_(0,_sk_27)[8]: - 1.69921875 frac[8,81] + mem[8,81] >= 0
 accuracy_(0,_sk_27)[9]: - 1.69921875 frac[9,81] + mem[9,81] >= 0
 accuracy_(0,_sk_27)[10]: - 1.69921875 frac[10,81] + mem[10,81] >= 0
 accuracy_(0,_sk_27)[11]: - 1.69921875 frac[11,81] + mem[11,81] >= 0
 accuracy_(0,_sk_27)[12]: - 1.69921875 frac[12,81] + mem[12,81] >= 0
 accuracy_(0,_sk_27)[13]: - 1.69921875 frac[13,81] + mem[13,81] >= 0
 accuracy_(0,_sk_27)[14]: - 1.69921875 frac[14,81] + mem[14,81] >= 0
 accuracy_(0,_sk_27)[15]: - 1.69921875 frac[15,81] + mem[15,81] >= 0
 accuracy_(0,_sk_27)[16]: - 1.69921875 frac[16,81] + mem[16,81] >= 0
 accuracy_(0,_sk_27)[17]: - 1.69921875 frac[17,81] + mem[17,81] >= 0
 accuracy_(0,_sk_27)[18]: - 1.69921875 frac[18,81] + mem[18,81] >= 0
 accuracy_(0,_sk_27)[19]: - 1.69921875 frac[19,81] + mem[19,81] >= 0
 accuracy_(0,_sk_27)[20]: - 1.69921875 frac[20,81] + mem[20,81] >= 0
 accuracy_(0,_sk_27)[21]: - 1.69921875 frac[21,81] + mem[21,81] >= 0
 accuracy_(0,_sk_27)[22]: - 1.69921875 frac[22,81] + mem[22,81] >= 0
 accuracy_(1,_sk_27)[0]: - 1.69921875 frac[0,82] + mem[0,82] >= 0
 accuracy_(1,_sk_27)[1]: - 1.69921875 frac[1,82] + mem[1,82] >= 0
 accuracy_(1,_sk_27)[2]: - 1.69921875 frac[2,82] + mem[2,82] >= 0
 accuracy_(1,_sk_27)[3]: - 1.69921875 frac[3,82] + mem[3,82] >= 0
 accuracy_(1,_sk_27)[4]: - 1.69921875 frac[4,82] + mem[4,82] >= 0
 accuracy_(1,_sk_27)[5]: - 1.69921875 frac[5,82] + mem[5,82] >= 0
 accuracy_(1,_sk_27)[6]: - 1.69921875 frac[6,82] + mem[6,82] >= 0
 accuracy_(1,_sk_27)[7]: - 1.69921875 frac[7,82] + mem[7,82] >= 0
 accuracy_(1,_sk_27)[8]: - 1.69921875 frac[8,82] + mem[8,82] >= 0
 accuracy_(1,_sk_27)[9]: - 1.69921875 frac[9,82] + mem[9,82] >= 0
 accuracy_(1,_sk_27)[10]: - 1.69921875 frac[10,82] + mem[10,82] >= 0
 accuracy_(1,_sk_27)[11]: - 1.69921875 frac[11,82] + mem[11,82] >= 0
 accuracy_(1,_sk_27)[12]: - 1.69921875 frac[12,82] + mem[12,82] >= 0
 accuracy_(1,_sk_27)[13]: - 1.69921875 frac[13,82] + mem[13,82] >= 0
 accuracy_(1,_sk_27)[14]: - 1.69921875 frac[14,82] + mem[14,82] >= 0
 accuracy_(1,_sk_27)[15]: - 1.69921875 frac[15,82] + mem[15,82] >= 0
 accuracy_(1,_sk_27)[16]: - 1.69921875 frac[16,82] + mem[16,82] >= 0
 accuracy_(1,_sk_27)[17]: - 1.69921875 frac[17,82] + mem[17,82] >= 0
 accuracy_(1,_sk_27)[18]: - 1.69921875 frac[18,82] + mem[18,82] >= 0
 accuracy_(1,_sk_27)[19]: - 1.69921875 frac[19,82] + mem[19,82] >= 0
 accuracy_(1,_sk_27)[20]: - 1.69921875 frac[20,82] + mem[20,82] >= 0
 accuracy_(1,_sk_27)[21]: - 1.69921875 frac[21,82] + mem[21,82] >= 0
 accuracy_(1,_sk_27)[22]: - 1.69921875 frac[22,82] + mem[22,82] >= 0
 accuracy_(2,_sk_27)[0]: - 1.69921875 frac[0,83] + mem[0,83] >= 0
 accuracy_(2,_sk_27)[1]: - 1.69921875 frac[1,83] + mem[1,83] >= 0
 accuracy_(2,_sk_27)[2]: - 1.69921875 frac[2,83] + mem[2,83] >= 0
 accuracy_(2,_sk_27)[3]: - 1.69921875 frac[3,83] + mem[3,83] >= 0
 accuracy_(2,_sk_27)[4]: - 1.69921875 frac[4,83] + mem[4,83] >= 0
 accuracy_(2,_sk_27)[5]: - 1.69921875 frac[5,83] + mem[5,83] >= 0
 accuracy_(2,_sk_27)[6]: - 1.69921875 frac[6,83] + mem[6,83] >= 0
 accuracy_(2,_sk_27)[7]: - 1.69921875 frac[7,83] + mem[7,83] >= 0
 accuracy_(2,_sk_27)[8]: - 1.69921875 frac[8,83] + mem[8,83] >= 0
 accuracy_(2,_sk_27)[9]: - 1.69921875 frac[9,83] + mem[9,83] >= 0
 accuracy_(2,_sk_27)[10]: - 1.69921875 frac[10,83] + mem[10,83] >= 0
 accuracy_(2,_sk_27)[11]: - 1.69921875 frac[11,83] + mem[11,83] >= 0
 accuracy_(2,_sk_27)[12]: - 1.69921875 frac[12,83] + mem[12,83] >= 0
 accuracy_(2,_sk_27)[13]: - 1.69921875 frac[13,83] + mem[13,83] >= 0
 accuracy_(2,_sk_27)[14]: - 1.69921875 frac[14,83] + mem[14,83] >= 0
 accuracy_(2,_sk_27)[15]: - 1.69921875 frac[15,83] + mem[15,83] >= 0
 accuracy_(2,_sk_27)[16]: - 1.69921875 frac[16,83] + mem[16,83] >= 0
 accuracy_(2,_sk_27)[17]: - 1.69921875 frac[17,83] + mem[17,83] >= 0
 accuracy_(2,_sk_27)[18]: - 1.69921875 frac[18,83] + mem[18,83] >= 0
 accuracy_(2,_sk_27)[19]: - 1.69921875 frac[19,83] + mem[19,83] >= 0
 accuracy_(2,_sk_27)[20]: - 1.69921875 frac[20,83] + mem[20,83] >= 0
 accuracy_(2,_sk_27)[21]: - 1.69921875 frac[21,83] + mem[21,83] >= 0
 accuracy_(2,_sk_27)[22]: - 1.69921875 frac[22,83] + mem[22,83] >= 0
 accuracy_(0,_sk_28)[0]: - 1.69921875 frac[0,84] + mem[0,84] >= 0
 accuracy_(0,_sk_28)[1]: - 1.69921875 frac[1,84] + mem[1,84] >= 0
 accuracy_(0,_sk_28)[2]: - 1.69921875 frac[2,84] + mem[2,84] >= 0
 accuracy_(0,_sk_28)[3]: - 1.69921875 frac[3,84] + mem[3,84] >= 0
 accuracy_(0,_sk_28)[4]: - 1.69921875 frac[4,84] + mem[4,84] >= 0
 accuracy_(0,_sk_28)[5]: - 1.69921875 frac[5,84] + mem[5,84] >= 0
 accuracy_(0,_sk_28)[6]: - 1.69921875 frac[6,84] + mem[6,84] >= 0
 accuracy_(0,_sk_28)[7]: - 1.69921875 frac[7,84] + mem[7,84] >= 0
 accuracy_(0,_sk_28)[8]: - 1.69921875 frac[8,84] + mem[8,84] >= 0
 accuracy_(0,_sk_28)[9]: - 1.69921875 frac[9,84] + mem[9,84] >= 0
 accuracy_(0,_sk_28)[10]: - 1.69921875 frac[10,84] + mem[10,84] >= 0
 accuracy_(0,_sk_28)[11]: - 1.69921875 frac[11,84] + mem[11,84] >= 0
 accuracy_(0,_sk_28)[12]: - 1.69921875 frac[12,84] + mem[12,84] >= 0
 accuracy_(0,_sk_28)[13]: - 1.69921875 frac[13,84] + mem[13,84] >= 0
 accuracy_(0,_sk_28)[14]: - 1.69921875 frac[14,84] + mem[14,84] >= 0
 accuracy_(0,_sk_28)[15]: - 1.69921875 frac[15,84] + mem[15,84] >= 0
 accuracy_(0,_sk_28)[16]: - 1.69921875 frac[16,84] + mem[16,84] >= 0
 accuracy_(0,_sk_28)[17]: - 1.69921875 frac[17,84] + mem[17,84] >= 0
 accuracy_(0,_sk_28)[18]: - 1.69921875 frac[18,84] + mem[18,84] >= 0
 accuracy_(0,_sk_28)[19]: - 1.69921875 frac[19,84] + mem[19,84] >= 0
 accuracy_(0,_sk_28)[20]: - 1.69921875 frac[20,84] + mem[20,84] >= 0
 accuracy_(0,_sk_28)[21]: - 1.69921875 frac[21,84] + mem[21,84] >= 0
 accuracy_(0,_sk_28)[22]: - 1.69921875 frac[22,84] + mem[22,84] >= 0
 accuracy_(1,_sk_28)[0]: - 1.69921875 frac[0,85] + mem[0,85] >= 0
 accuracy_(1,_sk_28)[1]: - 1.69921875 frac[1,85] + mem[1,85] >= 0
 accuracy_(1,_sk_28)[2]: - 1.69921875 frac[2,85] + mem[2,85] >= 0
 accuracy_(1,_sk_28)[3]: - 1.69921875 frac[3,85] + mem[3,85] >= 0
 accuracy_(1,_sk_28)[4]: - 1.69921875 frac[4,85] + mem[4,85] >= 0
 accuracy_(1,_sk_28)[5]: - 1.69921875 frac[5,85] + mem[5,85] >= 0
 accuracy_(1,_sk_28)[6]: - 1.69921875 frac[6,85] + mem[6,85] >= 0
 accuracy_(1,_sk_28)[7]: - 1.69921875 frac[7,85] + mem[7,85] >= 0
 accuracy_(1,_sk_28)[8]: - 1.69921875 frac[8,85] + mem[8,85] >= 0
 accuracy_(1,_sk_28)[9]: - 1.69921875 frac[9,85] + mem[9,85] >= 0
 accuracy_(1,_sk_28)[10]: - 1.69921875 frac[10,85] + mem[10,85] >= 0
 accuracy_(1,_sk_28)[11]: - 1.69921875 frac[11,85] + mem[11,85] >= 0
 accuracy_(1,_sk_28)[12]: - 1.69921875 frac[12,85] + mem[12,85] >= 0
 accuracy_(1,_sk_28)[13]: - 1.69921875 frac[13,85] + mem[13,85] >= 0
 accuracy_(1,_sk_28)[14]: - 1.69921875 frac[14,85] + mem[14,85] >= 0
 accuracy_(1,_sk_28)[15]: - 1.69921875 frac[15,85] + mem[15,85] >= 0
 accuracy_(1,_sk_28)[16]: - 1.69921875 frac[16,85] + mem[16,85] >= 0
 accuracy_(1,_sk_28)[17]: - 1.69921875 frac[17,85] + mem[17,85] >= 0
 accuracy_(1,_sk_28)[18]: - 1.69921875 frac[18,85] + mem[18,85] >= 0
 accuracy_(1,_sk_28)[19]: - 1.69921875 frac[19,85] + mem[19,85] >= 0
 accuracy_(1,_sk_28)[20]: - 1.69921875 frac[20,85] + mem[20,85] >= 0
 accuracy_(1,_sk_28)[21]: - 1.69921875 frac[21,85] + mem[21,85] >= 0
 accuracy_(1,_sk_28)[22]: - 1.69921875 frac[22,85] + mem[22,85] >= 0
 accuracy_(2,_sk_28)[0]: - 1.69921875 frac[0,86] + mem[0,86] >= 0
 accuracy_(2,_sk_28)[1]: - 1.69921875 frac[1,86] + mem[1,86] >= 0
 accuracy_(2,_sk_28)[2]: - 1.69921875 frac[2,86] + mem[2,86] >= 0
 accuracy_(2,_sk_28)[3]: - 1.69921875 frac[3,86] + mem[3,86] >= 0
 accuracy_(2,_sk_28)[4]: - 1.69921875 frac[4,86] + mem[4,86] >= 0
 accuracy_(2,_sk_28)[5]: - 1.69921875 frac[5,86] + mem[5,86] >= 0
 accuracy_(2,_sk_28)[6]: - 1.69921875 frac[6,86] + mem[6,86] >= 0
 accuracy_(2,_sk_28)[7]: - 1.69921875 frac[7,86] + mem[7,86] >= 0
 accuracy_(2,_sk_28)[8]: - 1.69921875 frac[8,86] + mem[8,86] >= 0
 accuracy_(2,_sk_28)[9]: - 1.69921875 frac[9,86] + mem[9,86] >= 0
 accuracy_(2,_sk_28)[10]: - 1.69921875 frac[10,86] + mem[10,86] >= 0
 accuracy_(2,_sk_28)[11]: - 1.69921875 frac[11,86] + mem[11,86] >= 0
 accuracy_(2,_sk_28)[12]: - 1.69921875 frac[12,86] + mem[12,86] >= 0
 accuracy_(2,_sk_28)[13]: - 1.69921875 frac[13,86] + mem[13,86] >= 0
 accuracy_(2,_sk_28)[14]: - 1.69921875 frac[14,86] + mem[14,86] >= 0
 accuracy_(2,_sk_28)[15]: - 1.69921875 frac[15,86] + mem[15,86] >= 0
 accuracy_(2,_sk_28)[16]: - 1.69921875 frac[16,86] + mem[16,86] >= 0
 accuracy_(2,_sk_28)[17]: - 1.69921875 frac[17,86] + mem[17,86] >= 0
 accuracy_(2,_sk_28)[18]: - 1.69921875 frac[18,86] + mem[18,86] >= 0
 accuracy_(2,_sk_28)[19]: - 1.69921875 frac[19,86] + mem[19,86] >= 0
 accuracy_(2,_sk_28)[20]: - 1.69921875 frac[20,86] + mem[20,86] >= 0
 accuracy_(2,_sk_28)[21]: - 1.69921875 frac[21,86] + mem[21,86] >= 0
 accuracy_(2,_sk_28)[22]: - 1.69921875 frac[22,86] + mem[22,86] >= 0
 accuracy_(0,_sk_29)[0]: - 1.69921875 frac[0,87] + mem[0,87] >= 0
 accuracy_(0,_sk_29)[1]: - 1.69921875 frac[1,87] + mem[1,87] >= 0
 accuracy_(0,_sk_29)[2]: - 1.69921875 frac[2,87] + mem[2,87] >= 0
 accuracy_(0,_sk_29)[3]: - 1.69921875 frac[3,87] + mem[3,87] >= 0
 accuracy_(0,_sk_29)[4]: - 1.69921875 frac[4,87] + mem[4,87] >= 0
 accuracy_(0,_sk_29)[5]: - 1.69921875 frac[5,87] + mem[5,87] >= 0
 accuracy_(0,_sk_29)[6]: - 1.69921875 frac[6,87] + mem[6,87] >= 0
 accuracy_(0,_sk_29)[7]: - 1.69921875 frac[7,87] + mem[7,87] >= 0
 accuracy_(0,_sk_29)[8]: - 1.69921875 frac[8,87] + mem[8,87] >= 0
 accuracy_(0,_sk_29)[9]: - 1.69921875 frac[9,87] + mem[9,87] >= 0
 accuracy_(0,_sk_29)[10]: - 1.69921875 frac[10,87] + mem[10,87] >= 0
 accuracy_(0,_sk_29)[11]: - 1.69921875 frac[11,87] + mem[11,87] >= 0
 accuracy_(0,_sk_29)[12]: - 1.69921875 frac[12,87] + mem[12,87] >= 0
 accuracy_(0,_sk_29)[13]: - 1.69921875 frac[13,87] + mem[13,87] >= 0
 accuracy_(0,_sk_29)[14]: - 1.69921875 frac[14,87] + mem[14,87] >= 0
 accuracy_(0,_sk_29)[15]: - 1.69921875 frac[15,87] + mem[15,87] >= 0
 accuracy_(0,_sk_29)[16]: - 1.69921875 frac[16,87] + mem[16,87] >= 0
 accuracy_(0,_sk_29)[17]: - 1.69921875 frac[17,87] + mem[17,87] >= 0
 accuracy_(0,_sk_29)[18]: - 1.69921875 frac[18,87] + mem[18,87] >= 0
 accuracy_(0,_sk_29)[19]: - 1.69921875 frac[19,87] + mem[19,87] >= 0
 accuracy_(0,_sk_29)[20]: - 1.69921875 frac[20,87] + mem[20,87] >= 0
 accuracy_(0,_sk_29)[21]: - 1.69921875 frac[21,87] + mem[21,87] >= 0
 accuracy_(0,_sk_29)[22]: - 1.69921875 frac[22,87] + mem[22,87] >= 0
 accuracy_(1,_sk_29)[0]: - 1.69921875 frac[0,88] + mem[0,88] >= 0
 accuracy_(1,_sk_29)[1]: - 1.69921875 frac[1,88] + mem[1,88] >= 0
 accuracy_(1,_sk_29)[2]: - 1.69921875 frac[2,88] + mem[2,88] >= 0
 accuracy_(1,_sk_29)[3]: - 1.69921875 frac[3,88] + mem[3,88] >= 0
 accuracy_(1,_sk_29)[4]: - 1.69921875 frac[4,88] + mem[4,88] >= 0
 accuracy_(1,_sk_29)[5]: - 1.69921875 frac[5,88] + mem[5,88] >= 0
 accuracy_(1,_sk_29)[6]: - 1.69921875 frac[6,88] + mem[6,88] >= 0
 accuracy_(1,_sk_29)[7]: - 1.69921875 frac[7,88] + mem[7,88] >= 0
 accuracy_(1,_sk_29)[8]: - 1.69921875 frac[8,88] + mem[8,88] >= 0
 accuracy_(1,_sk_29)[9]: - 1.69921875 frac[9,88] + mem[9,88] >= 0
 accuracy_(1,_sk_29)[10]: - 1.69921875 frac[10,88] + mem[10,88] >= 0
 accuracy_(1,_sk_29)[11]: - 1.69921875 frac[11,88] + mem[11,88] >= 0
 accuracy_(1,_sk_29)[12]: - 1.69921875 frac[12,88] + mem[12,88] >= 0
 accuracy_(1,_sk_29)[13]: - 1.69921875 frac[13,88] + mem[13,88] >= 0
 accuracy_(1,_sk_29)[14]: - 1.69921875 frac[14,88] + mem[14,88] >= 0
 accuracy_(1,_sk_29)[15]: - 1.69921875 frac[15,88] + mem[15,88] >= 0
 accuracy_(1,_sk_29)[16]: - 1.69921875 frac[16,88] + mem[16,88] >= 0
 accuracy_(1,_sk_29)[17]: - 1.69921875 frac[17,88] + mem[17,88] >= 0
 accuracy_(1,_sk_29)[18]: - 1.69921875 frac[18,88] + mem[18,88] >= 0
 accuracy_(1,_sk_29)[19]: - 1.69921875 frac[19,88] + mem[19,88] >= 0
 accuracy_(1,_sk_29)[20]: - 1.69921875 frac[20,88] + mem[20,88] >= 0
 accuracy_(1,_sk_29)[21]: - 1.69921875 frac[21,88] + mem[21,88] >= 0
 accuracy_(1,_sk_29)[22]: - 1.69921875 frac[22,88] + mem[22,88] >= 0
 accuracy_(2,_sk_29)[0]: - 1.69921875 frac[0,89] + mem[0,89] >= 0
 accuracy_(2,_sk_29)[1]: - 1.69921875 frac[1,89] + mem[1,89] >= 0
 accuracy_(2,_sk_29)[2]: - 1.69921875 frac[2,89] + mem[2,89] >= 0
 accuracy_(2,_sk_29)[3]: - 1.69921875 frac[3,89] + mem[3,89] >= 0
 accuracy_(2,_sk_29)[4]: - 1.69921875 frac[4,89] + mem[4,89] >= 0
 accuracy_(2,_sk_29)[5]: - 1.69921875 frac[5,89] + mem[5,89] >= 0
 accuracy_(2,_sk_29)[6]: - 1.69921875 frac[6,89] + mem[6,89] >= 0
 accuracy_(2,_sk_29)[7]: - 1.69921875 frac[7,89] + mem[7,89] >= 0
 accuracy_(2,_sk_29)[8]: - 1.69921875 frac[8,89] + mem[8,89] >= 0
 accuracy_(2,_sk_29)[9]: - 1.69921875 frac[9,89] + mem[9,89] >= 0
 accuracy_(2,_sk_29)[10]: - 1.69921875 frac[10,89] + mem[10,89] >= 0
 accuracy_(2,_sk_29)[11]: - 1.69921875 frac[11,89] + mem[11,89] >= 0
 accuracy_(2,_sk_29)[12]: - 1.69921875 frac[12,89] + mem[12,89] >= 0
 accuracy_(2,_sk_29)[13]: - 1.69921875 frac[13,89] + mem[13,89] >= 0
 accuracy_(2,_sk_29)[14]: - 1.69921875 frac[14,89] + mem[14,89] >= 0
 accuracy_(2,_sk_29)[15]: - 1.69921875 frac[15,89] + mem[15,89] >= 0
 accuracy_(2,_sk_29)[16]: - 1.69921875 frac[16,89] + mem[16,89] >= 0
 accuracy_(2,_sk_29)[17]: - 1.69921875 frac[17,89] + mem[17,89] >= 0
 accuracy_(2,_sk_29)[18]: - 1.69921875 frac[18,89] + mem[18,89] >= 0
 accuracy_(2,_sk_29)[19]: - 1.69921875 frac[19,89] + mem[19,89] >= 0
 accuracy_(2,_sk_29)[20]: - 1.69921875 frac[20,89] + mem[20,89] >= 0
 accuracy_(2,_sk_29)[21]: - 1.69921875 frac[21,89] + mem[21,89] >= 0
 accuracy_(2,_sk_29)[22]: - 1.69921875 frac[22,89] + mem[22,89] >= 0
 accuracy_(0,_sk_30)[0]: - 1.69921875 frac[0,90] + mem[0,90] >= 0
 accuracy_(0,_sk_30)[1]: - 1.69921875 frac[1,90] + mem[1,90] >= 0
 accuracy_(0,_sk_30)[2]: - 1.69921875 frac[2,90] + mem[2,90] >= 0
 accuracy_(0,_sk_30)[3]: - 1.69921875 frac[3,90] + mem[3,90] >= 0
 accuracy_(0,_sk_30)[4]: - 1.69921875 frac[4,90] + mem[4,90] >= 0
 accuracy_(0,_sk_30)[5]: - 1.69921875 frac[5,90] + mem[5,90] >= 0
 accuracy_(0,_sk_30)[6]: - 1.69921875 frac[6,90] + mem[6,90] >= 0
 accuracy_(0,_sk_30)[7]: - 1.69921875 frac[7,90] + mem[7,90] >= 0
 accuracy_(0,_sk_30)[8]: - 1.69921875 frac[8,90] + mem[8,90] >= 0
 accuracy_(0,_sk_30)[9]: - 1.69921875 frac[9,90] + mem[9,90] >= 0
 accuracy_(0,_sk_30)[10]: - 1.69921875 frac[10,90] + mem[10,90] >= 0
 accuracy_(0,_sk_30)[11]: - 1.69921875 frac[11,90] + mem[11,90] >= 0
 accuracy_(0,_sk_30)[12]: - 1.69921875 frac[12,90] + mem[12,90] >= 0
 accuracy_(0,_sk_30)[13]: - 1.69921875 frac[13,90] + mem[13,90] >= 0
 accuracy_(0,_sk_30)[14]: - 1.69921875 frac[14,90] + mem[14,90] >= 0
 accuracy_(0,_sk_30)[15]: - 1.69921875 frac[15,90] + mem[15,90] >= 0
 accuracy_(0,_sk_30)[16]: - 1.69921875 frac[16,90] + mem[16,90] >= 0
 accuracy_(0,_sk_30)[17]: - 1.69921875 frac[17,90] + mem[17,90] >= 0
 accuracy_(0,_sk_30)[18]: - 1.69921875 frac[18,90] + mem[18,90] >= 0
 accuracy_(0,_sk_30)[19]: - 1.69921875 frac[19,90] + mem[19,90] >= 0
 accuracy_(0,_sk_30)[20]: - 1.69921875 frac[20,90] + mem[20,90] >= 0
 accuracy_(0,_sk_30)[21]: - 1.69921875 frac[21,90] + mem[21,90] >= 0
 accuracy_(0,_sk_30)[22]: - 1.69921875 frac[22,90] + mem[22,90] >= 0
 accuracy_(1,_sk_30)[0]: - 1.69921875 frac[0,91] + mem[0,91] >= 0
 accuracy_(1,_sk_30)[1]: - 1.69921875 frac[1,91] + mem[1,91] >= 0
 accuracy_(1,_sk_30)[2]: - 1.69921875 frac[2,91] + mem[2,91] >= 0
 accuracy_(1,_sk_30)[3]: - 1.69921875 frac[3,91] + mem[3,91] >= 0
 accuracy_(1,_sk_30)[4]: - 1.69921875 frac[4,91] + mem[4,91] >= 0
 accuracy_(1,_sk_30)[5]: - 1.69921875 frac[5,91] + mem[5,91] >= 0
 accuracy_(1,_sk_30)[6]: - 1.69921875 frac[6,91] + mem[6,91] >= 0
 accuracy_(1,_sk_30)[7]: - 1.69921875 frac[7,91] + mem[7,91] >= 0
 accuracy_(1,_sk_30)[8]: - 1.69921875 frac[8,91] + mem[8,91] >= 0
 accuracy_(1,_sk_30)[9]: - 1.69921875 frac[9,91] + mem[9,91] >= 0
 accuracy_(1,_sk_30)[10]: - 1.69921875 frac[10,91] + mem[10,91] >= 0
 accuracy_(1,_sk_30)[11]: - 1.69921875 frac[11,91] + mem[11,91] >= 0
 accuracy_(1,_sk_30)[12]: - 1.69921875 frac[12,91] + mem[12,91] >= 0
 accuracy_(1,_sk_30)[13]: - 1.69921875 frac[13,91] + mem[13,91] >= 0
 accuracy_(1,_sk_30)[14]: - 1.69921875 frac[14,91] + mem[14,91] >= 0
 accuracy_(1,_sk_30)[15]: - 1.69921875 frac[15,91] + mem[15,91] >= 0
 accuracy_(1,_sk_30)[16]: - 1.69921875 frac[16,91] + mem[16,91] >= 0
 accuracy_(1,_sk_30)[17]: - 1.69921875 frac[17,91] + mem[17,91] >= 0
 accuracy_(1,_sk_30)[18]: - 1.69921875 frac[18,91] + mem[18,91] >= 0
 accuracy_(1,_sk_30)[19]: - 1.69921875 frac[19,91] + mem[19,91] >= 0
 accuracy_(1,_sk_30)[20]: - 1.69921875 frac[20,91] + mem[20,91] >= 0
 accuracy_(1,_sk_30)[21]: - 1.69921875 frac[21,91] + mem[21,91] >= 0
 accuracy_(1,_sk_30)[22]: - 1.69921875 frac[22,91] + mem[22,91] >= 0
 accuracy_(2,_sk_30)[0]: - 1.69921875 frac[0,92] + mem[0,92] >= 0
 accuracy_(2,_sk_30)[1]: - 1.69921875 frac[1,92] + mem[1,92] >= 0
 accuracy_(2,_sk_30)[2]: - 1.69921875 frac[2,92] + mem[2,92] >= 0
 accuracy_(2,_sk_30)[3]: - 1.69921875 frac[3,92] + mem[3,92] >= 0
 accuracy_(2,_sk_30)[4]: - 1.69921875 frac[4,92] + mem[4,92] >= 0
 accuracy_(2,_sk_30)[5]: - 1.69921875 frac[5,92] + mem[5,92] >= 0
 accuracy_(2,_sk_30)[6]: - 1.69921875 frac[6,92] + mem[6,92] >= 0
 accuracy_(2,_sk_30)[7]: - 1.69921875 frac[7,92] + mem[7,92] >= 0
 accuracy_(2,_sk_30)[8]: - 1.69921875 frac[8,92] + mem[8,92] >= 0
 accuracy_(2,_sk_30)[9]: - 1.69921875 frac[9,92] + mem[9,92] >= 0
 accuracy_(2,_sk_30)[10]: - 1.69921875 frac[10,92] + mem[10,92] >= 0
 accuracy_(2,_sk_30)[11]: - 1.69921875 frac[11,92] + mem[11,92] >= 0
 accuracy_(2,_sk_30)[12]: - 1.69921875 frac[12,92] + mem[12,92] >= 0
 accuracy_(2,_sk_30)[13]: - 1.69921875 frac[13,92] + mem[13,92] >= 0
 accuracy_(2,_sk_30)[14]: - 1.69921875 frac[14,92] + mem[14,92] >= 0
 accuracy_(2,_sk_30)[15]: - 1.69921875 frac[15,92] + mem[15,92] >= 0
 accuracy_(2,_sk_30)[16]: - 1.69921875 frac[16,92] + mem[16,92] >= 0
 accuracy_(2,_sk_30)[17]: - 1.69921875 frac[17,92] + mem[17,92] >= 0
 accuracy_(2,_sk_30)[18]: - 1.69921875 frac[18,92] + mem[18,92] >= 0
 accuracy_(2,_sk_30)[19]: - 1.69921875 frac[19,92] + mem[19,92] >= 0
 accuracy_(2,_sk_30)[20]: - 1.69921875 frac[20,92] + mem[20,92] >= 0
 accuracy_(2,_sk_30)[21]: - 1.69921875 frac[21,92] + mem[21,92] >= 0
 accuracy_(2,_sk_30)[22]: - 1.69921875 frac[22,92] + mem[22,92] >= 0
 accuracy_(0,_sk_31)[0]: - 1.69921875 frac[0,93] + mem[0,93] >= 0
 accuracy_(0,_sk_31)[1]: - 1.69921875 frac[1,93] + mem[1,93] >= 0
 accuracy_(0,_sk_31)[2]: - 1.69921875 frac[2,93] + mem[2,93] >= 0
 accuracy_(0,_sk_31)[3]: - 1.69921875 frac[3,93] + mem[3,93] >= 0
 accuracy_(0,_sk_31)[4]: - 1.69921875 frac[4,93] + mem[4,93] >= 0
 accuracy_(0,_sk_31)[5]: - 1.69921875 frac[5,93] + mem[5,93] >= 0
 accuracy_(0,_sk_31)[6]: - 1.69921875 frac[6,93] + mem[6,93] >= 0
 accuracy_(0,_sk_31)[7]: - 1.69921875 frac[7,93] + mem[7,93] >= 0
 accuracy_(0,_sk_31)[8]: - 1.69921875 frac[8,93] + mem[8,93] >= 0
 accuracy_(0,_sk_31)[9]: - 1.69921875 frac[9,93] + mem[9,93] >= 0
 accuracy_(0,_sk_31)[10]: - 1.69921875 frac[10,93] + mem[10,93] >= 0
 accuracy_(0,_sk_31)[11]: - 1.69921875 frac[11,93] + mem[11,93] >= 0
 accuracy_(0,_sk_31)[12]: - 1.69921875 frac[12,93] + mem[12,93] >= 0
 accuracy_(0,_sk_31)[13]: - 1.69921875 frac[13,93] + mem[13,93] >= 0
 accuracy_(0,_sk_31)[14]: - 1.69921875 frac[14,93] + mem[14,93] >= 0
 accuracy_(0,_sk_31)[15]: - 1.69921875 frac[15,93] + mem[15,93] >= 0
 accuracy_(0,_sk_31)[16]: - 1.69921875 frac[16,93] + mem[16,93] >= 0
 accuracy_(0,_sk_31)[17]: - 1.69921875 frac[17,93] + mem[17,93] >= 0
 accuracy_(0,_sk_31)[18]: - 1.69921875 frac[18,93] + mem[18,93] >= 0
 accuracy_(0,_sk_31)[19]: - 1.69921875 frac[19,93] + mem[19,93] >= 0
 accuracy_(0,_sk_31)[20]: - 1.69921875 frac[20,93] + mem[20,93] >= 0
 accuracy_(0,_sk_31)[21]: - 1.69921875 frac[21,93] + mem[21,93] >= 0
 accuracy_(0,_sk_31)[22]: - 1.69921875 frac[22,93] + mem[22,93] >= 0
 accuracy_(1,_sk_31)[0]: - 1.69921875 frac[0,94] + mem[0,94] >= 0
 accuracy_(1,_sk_31)[1]: - 1.69921875 frac[1,94] + mem[1,94] >= 0
 accuracy_(1,_sk_31)[2]: - 1.69921875 frac[2,94] + mem[2,94] >= 0
 accuracy_(1,_sk_31)[3]: - 1.69921875 frac[3,94] + mem[3,94] >= 0
 accuracy_(1,_sk_31)[4]: - 1.69921875 frac[4,94] + mem[4,94] >= 0
 accuracy_(1,_sk_31)[5]: - 1.69921875 frac[5,94] + mem[5,94] >= 0
 accuracy_(1,_sk_31)[6]: - 1.69921875 frac[6,94] + mem[6,94] >= 0
 accuracy_(1,_sk_31)[7]: - 1.69921875 frac[7,94] + mem[7,94] >= 0
 accuracy_(1,_sk_31)[8]: - 1.69921875 frac[8,94] + mem[8,94] >= 0
 accuracy_(1,_sk_31)[9]: - 1.69921875 frac[9,94] + mem[9,94] >= 0
 accuracy_(1,_sk_31)[10]: - 1.69921875 frac[10,94] + mem[10,94] >= 0
 accuracy_(1,_sk_31)[11]: - 1.69921875 frac[11,94] + mem[11,94] >= 0
 accuracy_(1,_sk_31)[12]: - 1.69921875 frac[12,94] + mem[12,94] >= 0
 accuracy_(1,_sk_31)[13]: - 1.69921875 frac[13,94] + mem[13,94] >= 0
 accuracy_(1,_sk_31)[14]: - 1.69921875 frac[14,94] + mem[14,94] >= 0
 accuracy_(1,_sk_31)[15]: - 1.69921875 frac[15,94] + mem[15,94] >= 0
 accuracy_(1,_sk_31)[16]: - 1.69921875 frac[16,94] + mem[16,94] >= 0
 accuracy_(1,_sk_31)[17]: - 1.69921875 frac[17,94] + mem[17,94] >= 0
 accuracy_(1,_sk_31)[18]: - 1.69921875 frac[18,94] + mem[18,94] >= 0
 accuracy_(1,_sk_31)[19]: - 1.69921875 frac[19,94] + mem[19,94] >= 0
 accuracy_(1,_sk_31)[20]: - 1.69921875 frac[20,94] + mem[20,94] >= 0
 accuracy_(1,_sk_31)[21]: - 1.69921875 frac[21,94] + mem[21,94] >= 0
 accuracy_(1,_sk_31)[22]: - 1.69921875 frac[22,94] + mem[22,94] >= 0
 accuracy_(2,_sk_31)[0]: - 1.69921875 frac[0,95] + mem[0,95] >= 0
 accuracy_(2,_sk_31)[1]: - 1.69921875 frac[1,95] + mem[1,95] >= 0
 accuracy_(2,_sk_31)[2]: - 1.69921875 frac[2,95] + mem[2,95] >= 0
 accuracy_(2,_sk_31)[3]: - 1.69921875 frac[3,95] + mem[3,95] >= 0
 accuracy_(2,_sk_31)[4]: - 1.69921875 frac[4,95] + mem[4,95] >= 0
 accuracy_(2,_sk_31)[5]: - 1.69921875 frac[5,95] + mem[5,95] >= 0
 accuracy_(2,_sk_31)[6]: - 1.69921875 frac[6,95] + mem[6,95] >= 0
 accuracy_(2,_sk_31)[7]: - 1.69921875 frac[7,95] + mem[7,95] >= 0
 accuracy_(2,_sk_31)[8]: - 1.69921875 frac[8,95] + mem[8,95] >= 0
 accuracy_(2,_sk_31)[9]: - 1.69921875 frac[9,95] + mem[9,95] >= 0
 accuracy_(2,_sk_31)[10]: - 1.69921875 frac[10,95] + mem[10,95] >= 0
 accuracy_(2,_sk_31)[11]: - 1.69921875 frac[11,95] + mem[11,95] >= 0
 accuracy_(2,_sk_31)[12]: - 1.69921875 frac[12,95] + mem[12,95] >= 0
 accuracy_(2,_sk_31)[13]: - 1.69921875 frac[13,95] + mem[13,95] >= 0
 accuracy_(2,_sk_31)[14]: - 1.69921875 frac[14,95] + mem[14,95] >= 0
 accuracy_(2,_sk_31)[15]: - 1.69921875 frac[15,95] + mem[15,95] >= 0
 accuracy_(2,_sk_31)[16]: - 1.69921875 frac[16,95] + mem[16,95] >= 0
 accuracy_(2,_sk_31)[17]: - 1.69921875 frac[17,95] + mem[17,95] >= 0
 accuracy_(2,_sk_31)[18]: - 1.69921875 frac[18,95] + mem[18,95] >= 0
 accuracy_(2,_sk_31)[19]: - 1.69921875 frac[19,95] + mem[19,95] >= 0
 accuracy_(2,_sk_31)[20]: - 1.69921875 frac[20,95] + mem[20,95] >= 0
 accuracy_(2,_sk_31)[21]: - 1.69921875 frac[21,95] + mem[21,95] >= 0
 accuracy_(2,_sk_31)[22]: - 1.69921875 frac[22,95] + mem[22,95] >= 0
 accuracy_(0,_sk_32)[0]: - 1.69921875 frac[0,96] + mem[0,96] >= 0
 accuracy_(0,_sk_32)[1]: - 1.69921875 frac[1,96] + mem[1,96] >= 0
 accuracy_(0,_sk_32)[2]: - 1.69921875 frac[2,96] + mem[2,96] >= 0
 accuracy_(0,_sk_32)[3]: - 1.69921875 frac[3,96] + mem[3,96] >= 0
 accuracy_(0,_sk_32)[4]: - 1.69921875 frac[4,96] + mem[4,96] >= 0
 accuracy_(0,_sk_32)[5]: - 1.69921875 frac[5,96] + mem[5,96] >= 0
 accuracy_(0,_sk_32)[6]: - 1.69921875 frac[6,96] + mem[6,96] >= 0
 accuracy_(0,_sk_32)[7]: - 1.69921875 frac[7,96] + mem[7,96] >= 0
 accuracy_(0,_sk_32)[8]: - 1.69921875 frac[8,96] + mem[8,96] >= 0
 accuracy_(0,_sk_32)[9]: - 1.69921875 frac[9,96] + mem[9,96] >= 0
 accuracy_(0,_sk_32)[10]: - 1.69921875 frac[10,96] + mem[10,96] >= 0
 accuracy_(0,_sk_32)[11]: - 1.69921875 frac[11,96] + mem[11,96] >= 0
 accuracy_(0,_sk_32)[12]: - 1.69921875 frac[12,96] + mem[12,96] >= 0
 accuracy_(0,_sk_32)[13]: - 1.69921875 frac[13,96] + mem[13,96] >= 0
 accuracy_(0,_sk_32)[14]: - 1.69921875 frac[14,96] + mem[14,96] >= 0
 accuracy_(0,_sk_32)[15]: - 1.69921875 frac[15,96] + mem[15,96] >= 0
 accuracy_(0,_sk_32)[16]: - 1.69921875 frac[16,96] + mem[16,96] >= 0
 accuracy_(0,_sk_32)[17]: - 1.69921875 frac[17,96] + mem[17,96] >= 0
 accuracy_(0,_sk_32)[18]: - 1.69921875 frac[18,96] + mem[18,96] >= 0
 accuracy_(0,_sk_32)[19]: - 1.69921875 frac[19,96] + mem[19,96] >= 0
 accuracy_(0,_sk_32)[20]: - 1.69921875 frac[20,96] + mem[20,96] >= 0
 accuracy_(0,_sk_32)[21]: - 1.69921875 frac[21,96] + mem[21,96] >= 0
 accuracy_(0,_sk_32)[22]: - 1.69921875 frac[22,96] + mem[22,96] >= 0
 accuracy_(1,_sk_32)[0]: - 1.69921875 frac[0,97] + mem[0,97] >= 0
 accuracy_(1,_sk_32)[1]: - 1.69921875 frac[1,97] + mem[1,97] >= 0
 accuracy_(1,_sk_32)[2]: - 1.69921875 frac[2,97] + mem[2,97] >= 0
 accuracy_(1,_sk_32)[3]: - 1.69921875 frac[3,97] + mem[3,97] >= 0
 accuracy_(1,_sk_32)[4]: - 1.69921875 frac[4,97] + mem[4,97] >= 0
 accuracy_(1,_sk_32)[5]: - 1.69921875 frac[5,97] + mem[5,97] >= 0
 accuracy_(1,_sk_32)[6]: - 1.69921875 frac[6,97] + mem[6,97] >= 0
 accuracy_(1,_sk_32)[7]: - 1.69921875 frac[7,97] + mem[7,97] >= 0
 accuracy_(1,_sk_32)[8]: - 1.69921875 frac[8,97] + mem[8,97] >= 0
 accuracy_(1,_sk_32)[9]: - 1.69921875 frac[9,97] + mem[9,97] >= 0
 accuracy_(1,_sk_32)[10]: - 1.69921875 frac[10,97] + mem[10,97] >= 0
 accuracy_(1,_sk_32)[11]: - 1.69921875 frac[11,97] + mem[11,97] >= 0
 accuracy_(1,_sk_32)[12]: - 1.69921875 frac[12,97] + mem[12,97] >= 0
 accuracy_(1,_sk_32)[13]: - 1.69921875 frac[13,97] + mem[13,97] >= 0
 accuracy_(1,_sk_32)[14]: - 1.69921875 frac[14,97] + mem[14,97] >= 0
 accuracy_(1,_sk_32)[15]: - 1.69921875 frac[15,97] + mem[15,97] >= 0
 accuracy_(1,_sk_32)[16]: - 1.69921875 frac[16,97] + mem[16,97] >= 0
 accuracy_(1,_sk_32)[17]: - 1.69921875 frac[17,97] + mem[17,97] >= 0
 accuracy_(1,_sk_32)[18]: - 1.69921875 frac[18,97] + mem[18,97] >= 0
 accuracy_(1,_sk_32)[19]: - 1.69921875 frac[19,97] + mem[19,97] >= 0
 accuracy_(1,_sk_32)[20]: - 1.69921875 frac[20,97] + mem[20,97] >= 0
 accuracy_(1,_sk_32)[21]: - 1.69921875 frac[21,97] + mem[21,97] >= 0
 accuracy_(1,_sk_32)[22]: - 1.69921875 frac[22,97] + mem[22,97] >= 0
 accuracy_(2,_sk_32)[0]: - 1.69921875 frac[0,98] + mem[0,98] >= 0
 accuracy_(2,_sk_32)[1]: - 1.69921875 frac[1,98] + mem[1,98] >= 0
 accuracy_(2,_sk_32)[2]: - 1.69921875 frac[2,98] + mem[2,98] >= 0
 accuracy_(2,_sk_32)[3]: - 1.69921875 frac[3,98] + mem[3,98] >= 0
 accuracy_(2,_sk_32)[4]: - 1.69921875 frac[4,98] + mem[4,98] >= 0
 accuracy_(2,_sk_32)[5]: - 1.69921875 frac[5,98] + mem[5,98] >= 0
 accuracy_(2,_sk_32)[6]: - 1.69921875 frac[6,98] + mem[6,98] >= 0
 accuracy_(2,_sk_32)[7]: - 1.69921875 frac[7,98] + mem[7,98] >= 0
 accuracy_(2,_sk_32)[8]: - 1.69921875 frac[8,98] + mem[8,98] >= 0
 accuracy_(2,_sk_32)[9]: - 1.69921875 frac[9,98] + mem[9,98] >= 0
 accuracy_(2,_sk_32)[10]: - 1.69921875 frac[10,98] + mem[10,98] >= 0
 accuracy_(2,_sk_32)[11]: - 1.69921875 frac[11,98] + mem[11,98] >= 0
 accuracy_(2,_sk_32)[12]: - 1.69921875 frac[12,98] + mem[12,98] >= 0
 accuracy_(2,_sk_32)[13]: - 1.69921875 frac[13,98] + mem[13,98] >= 0
 accuracy_(2,_sk_32)[14]: - 1.69921875 frac[14,98] + mem[14,98] >= 0
 accuracy_(2,_sk_32)[15]: - 1.69921875 frac[15,98] + mem[15,98] >= 0
 accuracy_(2,_sk_32)[16]: - 1.69921875 frac[16,98] + mem[16,98] >= 0
 accuracy_(2,_sk_32)[17]: - 1.69921875 frac[17,98] + mem[17,98] >= 0
 accuracy_(2,_sk_32)[18]: - 1.69921875 frac[18,98] + mem[18,98] >= 0
 accuracy_(2,_sk_32)[19]: - 1.69921875 frac[19,98] + mem[19,98] >= 0
 accuracy_(2,_sk_32)[20]: - 1.69921875 frac[20,98] + mem[20,98] >= 0
 accuracy_(2,_sk_32)[21]: - 1.69921875 frac[21,98] + mem[21,98] >= 0
 accuracy_(2,_sk_32)[22]: - 1.69921875 frac[22,98] + mem[22,98] >= 0
 accuracy_(0,_sk_33)[0]: - 1.69921875 frac[0,99] + mem[0,99] >= 0
 accuracy_(0,_sk_33)[1]: - 1.69921875 frac[1,99] + mem[1,99] >= 0
 accuracy_(0,_sk_33)[2]: - 1.69921875 frac[2,99] + mem[2,99] >= 0
 accuracy_(0,_sk_33)[3]: - 1.69921875 frac[3,99] + mem[3,99] >= 0
 accuracy_(0,_sk_33)[4]: - 1.69921875 frac[4,99] + mem[4,99] >= 0
 accuracy_(0,_sk_33)[5]: - 1.69921875 frac[5,99] + mem[5,99] >= 0
 accuracy_(0,_sk_33)[6]: - 1.69921875 frac[6,99] + mem[6,99] >= 0
 accuracy_(0,_sk_33)[7]: - 1.69921875 frac[7,99] + mem[7,99] >= 0
 accuracy_(0,_sk_33)[8]: - 1.69921875 frac[8,99] + mem[8,99] >= 0
 accuracy_(0,_sk_33)[9]: - 1.69921875 frac[9,99] + mem[9,99] >= 0
 accuracy_(0,_sk_33)[10]: - 1.69921875 frac[10,99] + mem[10,99] >= 0
 accuracy_(0,_sk_33)[11]: - 1.69921875 frac[11,99] + mem[11,99] >= 0
 accuracy_(0,_sk_33)[12]: - 1.69921875 frac[12,99] + mem[12,99] >= 0
 accuracy_(0,_sk_33)[13]: - 1.69921875 frac[13,99] + mem[13,99] >= 0
 accuracy_(0,_sk_33)[14]: - 1.69921875 frac[14,99] + mem[14,99] >= 0
 accuracy_(0,_sk_33)[15]: - 1.69921875 frac[15,99] + mem[15,99] >= 0
 accuracy_(0,_sk_33)[16]: - 1.69921875 frac[16,99] + mem[16,99] >= 0
 accuracy_(0,_sk_33)[17]: - 1.69921875 frac[17,99] + mem[17,99] >= 0
 accuracy_(0,_sk_33)[18]: - 1.69921875 frac[18,99] + mem[18,99] >= 0
 accuracy_(0,_sk_33)[19]: - 1.69921875 frac[19,99] + mem[19,99] >= 0
 accuracy_(0,_sk_33)[20]: - 1.69921875 frac[20,99] + mem[20,99] >= 0
 accuracy_(0,_sk_33)[21]: - 1.69921875 frac[21,99] + mem[21,99] >= 0
 accuracy_(0,_sk_33)[22]: - 1.69921875 frac[22,99] + mem[22,99] >= 0
 accuracy_(1,_sk_33)[0]: - 1.69921875 frac[0,100] + mem[0,100] >= 0
 accuracy_(1,_sk_33)[1]: - 1.69921875 frac[1,100] + mem[1,100] >= 0
 accuracy_(1,_sk_33)[2]: - 1.69921875 frac[2,100] + mem[2,100] >= 0
 accuracy_(1,_sk_33)[3]: - 1.69921875 frac[3,100] + mem[3,100] >= 0
 accuracy_(1,_sk_33)[4]: - 1.69921875 frac[4,100] + mem[4,100] >= 0
 accuracy_(1,_sk_33)[5]: - 1.69921875 frac[5,100] + mem[5,100] >= 0
 accuracy_(1,_sk_33)[6]: - 1.69921875 frac[6,100] + mem[6,100] >= 0
 accuracy_(1,_sk_33)[7]: - 1.69921875 frac[7,100] + mem[7,100] >= 0
 accuracy_(1,_sk_33)[8]: - 1.69921875 frac[8,100] + mem[8,100] >= 0
 accuracy_(1,_sk_33)[9]: - 1.69921875 frac[9,100] + mem[9,100] >= 0
 accuracy_(1,_sk_33)[10]: - 1.69921875 frac[10,100] + mem[10,100] >= 0
 accuracy_(1,_sk_33)[11]: - 1.69921875 frac[11,100] + mem[11,100] >= 0
 accuracy_(1,_sk_33)[12]: - 1.69921875 frac[12,100] + mem[12,100] >= 0
 accuracy_(1,_sk_33)[13]: - 1.69921875 frac[13,100] + mem[13,100] >= 0
 accuracy_(1,_sk_33)[14]: - 1.69921875 frac[14,100] + mem[14,100] >= 0
 accuracy_(1,_sk_33)[15]: - 1.69921875 frac[15,100] + mem[15,100] >= 0
 accuracy_(1,_sk_33)[16]: - 1.69921875 frac[16,100] + mem[16,100] >= 0
 accuracy_(1,_sk_33)[17]: - 1.69921875 frac[17,100] + mem[17,100] >= 0
 accuracy_(1,_sk_33)[18]: - 1.69921875 frac[18,100] + mem[18,100] >= 0
 accuracy_(1,_sk_33)[19]: - 1.69921875 frac[19,100] + mem[19,100] >= 0
 accuracy_(1,_sk_33)[20]: - 1.69921875 frac[20,100] + mem[20,100] >= 0
 accuracy_(1,_sk_33)[21]: - 1.69921875 frac[21,100] + mem[21,100] >= 0
 accuracy_(1,_sk_33)[22]: - 1.69921875 frac[22,100] + mem[22,100] >= 0
 accuracy_(2,_sk_33)[0]: - 1.69921875 frac[0,101] + mem[0,101] >= 0
 accuracy_(2,_sk_33)[1]: - 1.69921875 frac[1,101] + mem[1,101] >= 0
 accuracy_(2,_sk_33)[2]: - 1.69921875 frac[2,101] + mem[2,101] >= 0
 accuracy_(2,_sk_33)[3]: - 1.69921875 frac[3,101] + mem[3,101] >= 0
 accuracy_(2,_sk_33)[4]: - 1.69921875 frac[4,101] + mem[4,101] >= 0
 accuracy_(2,_sk_33)[5]: - 1.69921875 frac[5,101] + mem[5,101] >= 0
 accuracy_(2,_sk_33)[6]: - 1.69921875 frac[6,101] + mem[6,101] >= 0
 accuracy_(2,_sk_33)[7]: - 1.69921875 frac[7,101] + mem[7,101] >= 0
 accuracy_(2,_sk_33)[8]: - 1.69921875 frac[8,101] + mem[8,101] >= 0
 accuracy_(2,_sk_33)[9]: - 1.69921875 frac[9,101] + mem[9,101] >= 0
 accuracy_(2,_sk_33)[10]: - 1.69921875 frac[10,101] + mem[10,101] >= 0
 accuracy_(2,_sk_33)[11]: - 1.69921875 frac[11,101] + mem[11,101] >= 0
 accuracy_(2,_sk_33)[12]: - 1.69921875 frac[12,101] + mem[12,101] >= 0
 accuracy_(2,_sk_33)[13]: - 1.69921875 frac[13,101] + mem[13,101] >= 0
 accuracy_(2,_sk_33)[14]: - 1.69921875 frac[14,101] + mem[14,101] >= 0
 accuracy_(2,_sk_33)[15]: - 1.69921875 frac[15,101] + mem[15,101] >= 0
 accuracy_(2,_sk_33)[16]: - 1.69921875 frac[16,101] + mem[16,101] >= 0
 accuracy_(2,_sk_33)[17]: - 1.69921875 frac[17,101] + mem[17,101] >= 0
 accuracy_(2,_sk_33)[18]: - 1.69921875 frac[18,101] + mem[18,101] >= 0
 accuracy_(2,_sk_33)[19]: - 1.69921875 frac[19,101] + mem[19,101] >= 0
 accuracy_(2,_sk_33)[20]: - 1.69921875 frac[20,101] + mem[20,101] >= 0
 accuracy_(2,_sk_33)[21]: - 1.69921875 frac[21,101] + mem[21,101] >= 0
 accuracy_(2,_sk_33)[22]: - 1.69921875 frac[22,101] + mem[22,101] >= 0
 accuracy_(0,_sk_34)[0]: - 1.69921875 frac[0,102] + mem[0,102] >= 0
 accuracy_(0,_sk_34)[1]: - 1.69921875 frac[1,102] + mem[1,102] >= 0
 accuracy_(0,_sk_34)[2]: - 1.69921875 frac[2,102] + mem[2,102] >= 0
 accuracy_(0,_sk_34)[3]: - 1.69921875 frac[3,102] + mem[3,102] >= 0
 accuracy_(0,_sk_34)[4]: - 1.69921875 frac[4,102] + mem[4,102] >= 0
 accuracy_(0,_sk_34)[5]: - 1.69921875 frac[5,102] + mem[5,102] >= 0
 accuracy_(0,_sk_34)[6]: - 1.69921875 frac[6,102] + mem[6,102] >= 0
 accuracy_(0,_sk_34)[7]: - 1.69921875 frac[7,102] + mem[7,102] >= 0
 accuracy_(0,_sk_34)[8]: - 1.69921875 frac[8,102] + mem[8,102] >= 0
 accuracy_(0,_sk_34)[9]: - 1.69921875 frac[9,102] + mem[9,102] >= 0
 accuracy_(0,_sk_34)[10]: - 1.69921875 frac[10,102] + mem[10,102] >= 0
 accuracy_(0,_sk_34)[11]: - 1.69921875 frac[11,102] + mem[11,102] >= 0
 accuracy_(0,_sk_34)[12]: - 1.69921875 frac[12,102] + mem[12,102] >= 0
 accuracy_(0,_sk_34)[13]: - 1.69921875 frac[13,102] + mem[13,102] >= 0
 accuracy_(0,_sk_34)[14]: - 1.69921875 frac[14,102] + mem[14,102] >= 0
 accuracy_(0,_sk_34)[15]: - 1.69921875 frac[15,102] + mem[15,102] >= 0
 accuracy_(0,_sk_34)[16]: - 1.69921875 frac[16,102] + mem[16,102] >= 0
 accuracy_(0,_sk_34)[17]: - 1.69921875 frac[17,102] + mem[17,102] >= 0
 accuracy_(0,_sk_34)[18]: - 1.69921875 frac[18,102] + mem[18,102] >= 0
 accuracy_(0,_sk_34)[19]: - 1.69921875 frac[19,102] + mem[19,102] >= 0
 accuracy_(0,_sk_34)[20]: - 1.69921875 frac[20,102] + mem[20,102] >= 0
 accuracy_(0,_sk_34)[21]: - 1.69921875 frac[21,102] + mem[21,102] >= 0
 accuracy_(0,_sk_34)[22]: - 1.69921875 frac[22,102] + mem[22,102] >= 0
 accuracy_(1,_sk_34)[0]: - 1.69921875 frac[0,103] + mem[0,103] >= 0
 accuracy_(1,_sk_34)[1]: - 1.69921875 frac[1,103] + mem[1,103] >= 0
 accuracy_(1,_sk_34)[2]: - 1.69921875 frac[2,103] + mem[2,103] >= 0
 accuracy_(1,_sk_34)[3]: - 1.69921875 frac[3,103] + mem[3,103] >= 0
 accuracy_(1,_sk_34)[4]: - 1.69921875 frac[4,103] + mem[4,103] >= 0
 accuracy_(1,_sk_34)[5]: - 1.69921875 frac[5,103] + mem[5,103] >= 0
 accuracy_(1,_sk_34)[6]: - 1.69921875 frac[6,103] + mem[6,103] >= 0
 accuracy_(1,_sk_34)[7]: - 1.69921875 frac[7,103] + mem[7,103] >= 0
 accuracy_(1,_sk_34)[8]: - 1.69921875 frac[8,103] + mem[8,103] >= 0
 accuracy_(1,_sk_34)[9]: - 1.69921875 frac[9,103] + mem[9,103] >= 0
 accuracy_(1,_sk_34)[10]: - 1.69921875 frac[10,103] + mem[10,103] >= 0
 accuracy_(1,_sk_34)[11]: - 1.69921875 frac[11,103] + mem[11,103] >= 0
 accuracy_(1,_sk_34)[12]: - 1.69921875 frac[12,103] + mem[12,103] >= 0
 accuracy_(1,_sk_34)[13]: - 1.69921875 frac[13,103] + mem[13,103] >= 0
 accuracy_(1,_sk_34)[14]: - 1.69921875 frac[14,103] + mem[14,103] >= 0
 accuracy_(1,_sk_34)[15]: - 1.69921875 frac[15,103] + mem[15,103] >= 0
 accuracy_(1,_sk_34)[16]: - 1.69921875 frac[16,103] + mem[16,103] >= 0
 accuracy_(1,_sk_34)[17]: - 1.69921875 frac[17,103] + mem[17,103] >= 0
 accuracy_(1,_sk_34)[18]: - 1.69921875 frac[18,103] + mem[18,103] >= 0
 accuracy_(1,_sk_34)[19]: - 1.69921875 frac[19,103] + mem[19,103] >= 0
 accuracy_(1,_sk_34)[20]: - 1.69921875 frac[20,103] + mem[20,103] >= 0
 accuracy_(1,_sk_34)[21]: - 1.69921875 frac[21,103] + mem[21,103] >= 0
 accuracy_(1,_sk_34)[22]: - 1.69921875 frac[22,103] + mem[22,103] >= 0
 accuracy_(2,_sk_34)[0]: - 1.69921875 frac[0,104] + mem[0,104] >= 0
 accuracy_(2,_sk_34)[1]: - 1.69921875 frac[1,104] + mem[1,104] >= 0
 accuracy_(2,_sk_34)[2]: - 1.69921875 frac[2,104] + mem[2,104] >= 0
 accuracy_(2,_sk_34)[3]: - 1.69921875 frac[3,104] + mem[3,104] >= 0
 accuracy_(2,_sk_34)[4]: - 1.69921875 frac[4,104] + mem[4,104] >= 0
 accuracy_(2,_sk_34)[5]: - 1.69921875 frac[5,104] + mem[5,104] >= 0
 accuracy_(2,_sk_34)[6]: - 1.69921875 frac[6,104] + mem[6,104] >= 0
 accuracy_(2,_sk_34)[7]: - 1.69921875 frac[7,104] + mem[7,104] >= 0
 accuracy_(2,_sk_34)[8]: - 1.69921875 frac[8,104] + mem[8,104] >= 0
 accuracy_(2,_sk_34)[9]: - 1.69921875 frac[9,104] + mem[9,104] >= 0
 accuracy_(2,_sk_34)[10]: - 1.69921875 frac[10,104] + mem[10,104] >= 0
 accuracy_(2,_sk_34)[11]: - 1.69921875 frac[11,104] + mem[11,104] >= 0
 accuracy_(2,_sk_34)[12]: - 1.69921875 frac[12,104] + mem[12,104] >= 0
 accuracy_(2,_sk_34)[13]: - 1.69921875 frac[13,104] + mem[13,104] >= 0
 accuracy_(2,_sk_34)[14]: - 1.69921875 frac[14,104] + mem[14,104] >= 0
 accuracy_(2,_sk_34)[15]: - 1.69921875 frac[15,104] + mem[15,104] >= 0
 accuracy_(2,_sk_34)[16]: - 1.69921875 frac[16,104] + mem[16,104] >= 0
 accuracy_(2,_sk_34)[17]: - 1.69921875 frac[17,104] + mem[17,104] >= 0
 accuracy_(2,_sk_34)[18]: - 1.69921875 frac[18,104] + mem[18,104] >= 0
 accuracy_(2,_sk_34)[19]: - 1.69921875 frac[19,104] + mem[19,104] >= 0
 accuracy_(2,_sk_34)[20]: - 1.69921875 frac[20,104] + mem[20,104] >= 0
 accuracy_(2,_sk_34)[21]: - 1.69921875 frac[21,104] + mem[21,104] >= 0
 accuracy_(2,_sk_34)[22]: - 1.69921875 frac[22,104] + mem[22,104] >= 0
 accuracy_(0,_sk_35)[0]: - 1.69921875 frac[0,105] + mem[0,105] >= 0
 accuracy_(0,_sk_35)[1]: - 1.69921875 frac[1,105] + mem[1,105] >= 0
 accuracy_(0,_sk_35)[2]: - 1.69921875 frac[2,105] + mem[2,105] >= 0
 accuracy_(0,_sk_35)[3]: - 1.69921875 frac[3,105] + mem[3,105] >= 0
 accuracy_(0,_sk_35)[4]: - 1.69921875 frac[4,105] + mem[4,105] >= 0
 accuracy_(0,_sk_35)[5]: - 1.69921875 frac[5,105] + mem[5,105] >= 0
 accuracy_(0,_sk_35)[6]: - 1.69921875 frac[6,105] + mem[6,105] >= 0
 accuracy_(0,_sk_35)[7]: - 1.69921875 frac[7,105] + mem[7,105] >= 0
 accuracy_(0,_sk_35)[8]: - 1.69921875 frac[8,105] + mem[8,105] >= 0
 accuracy_(0,_sk_35)[9]: - 1.69921875 frac[9,105] + mem[9,105] >= 0
 accuracy_(0,_sk_35)[10]: - 1.69921875 frac[10,105] + mem[10,105] >= 0
 accuracy_(0,_sk_35)[11]: - 1.69921875 frac[11,105] + mem[11,105] >= 0
 accuracy_(0,_sk_35)[12]: - 1.69921875 frac[12,105] + mem[12,105] >= 0
 accuracy_(0,_sk_35)[13]: - 1.69921875 frac[13,105] + mem[13,105] >= 0
 accuracy_(0,_sk_35)[14]: - 1.69921875 frac[14,105] + mem[14,105] >= 0
 accuracy_(0,_sk_35)[15]: - 1.69921875 frac[15,105] + mem[15,105] >= 0
 accuracy_(0,_sk_35)[16]: - 1.69921875 frac[16,105] + mem[16,105] >= 0
 accuracy_(0,_sk_35)[17]: - 1.69921875 frac[17,105] + mem[17,105] >= 0
 accuracy_(0,_sk_35)[18]: - 1.69921875 frac[18,105] + mem[18,105] >= 0
 accuracy_(0,_sk_35)[19]: - 1.69921875 frac[19,105] + mem[19,105] >= 0
 accuracy_(0,_sk_35)[20]: - 1.69921875 frac[20,105] + mem[20,105] >= 0
 accuracy_(0,_sk_35)[21]: - 1.69921875 frac[21,105] + mem[21,105] >= 0
 accuracy_(0,_sk_35)[22]: - 1.69921875 frac[22,105] + mem[22,105] >= 0
 accuracy_(1,_sk_35)[0]: - 1.69921875 frac[0,106] + mem[0,106] >= 0
 accuracy_(1,_sk_35)[1]: - 1.69921875 frac[1,106] + mem[1,106] >= 0
 accuracy_(1,_sk_35)[2]: - 1.69921875 frac[2,106] + mem[2,106] >= 0
 accuracy_(1,_sk_35)[3]: - 1.69921875 frac[3,106] + mem[3,106] >= 0
 accuracy_(1,_sk_35)[4]: - 1.69921875 frac[4,106] + mem[4,106] >= 0
 accuracy_(1,_sk_35)[5]: - 1.69921875 frac[5,106] + mem[5,106] >= 0
 accuracy_(1,_sk_35)[6]: - 1.69921875 frac[6,106] + mem[6,106] >= 0
 accuracy_(1,_sk_35)[7]: - 1.69921875 frac[7,106] + mem[7,106] >= 0
 accuracy_(1,_sk_35)[8]: - 1.69921875 frac[8,106] + mem[8,106] >= 0
 accuracy_(1,_sk_35)[9]: - 1.69921875 frac[9,106] + mem[9,106] >= 0
 accuracy_(1,_sk_35)[10]: - 1.69921875 frac[10,106] + mem[10,106] >= 0
 accuracy_(1,_sk_35)[11]: - 1.69921875 frac[11,106] + mem[11,106] >= 0
 accuracy_(1,_sk_35)[12]: - 1.69921875 frac[12,106] + mem[12,106] >= 0
 accuracy_(1,_sk_35)[13]: - 1.69921875 frac[13,106] + mem[13,106] >= 0
 accuracy_(1,_sk_35)[14]: - 1.69921875 frac[14,106] + mem[14,106] >= 0
 accuracy_(1,_sk_35)[15]: - 1.69921875 frac[15,106] + mem[15,106] >= 0
 accuracy_(1,_sk_35)[16]: - 1.69921875 frac[16,106] + mem[16,106] >= 0
 accuracy_(1,_sk_35)[17]: - 1.69921875 frac[17,106] + mem[17,106] >= 0
 accuracy_(1,_sk_35)[18]: - 1.69921875 frac[18,106] + mem[18,106] >= 0
 accuracy_(1,_sk_35)[19]: - 1.69921875 frac[19,106] + mem[19,106] >= 0
 accuracy_(1,_sk_35)[20]: - 1.69921875 frac[20,106] + mem[20,106] >= 0
 accuracy_(1,_sk_35)[21]: - 1.69921875 frac[21,106] + mem[21,106] >= 0
 accuracy_(1,_sk_35)[22]: - 1.69921875 frac[22,106] + mem[22,106] >= 0
 accuracy_(2,_sk_35)[0]: - 1.69921875 frac[0,107] + mem[0,107] >= 0
 accuracy_(2,_sk_35)[1]: - 1.69921875 frac[1,107] + mem[1,107] >= 0
 accuracy_(2,_sk_35)[2]: - 1.69921875 frac[2,107] + mem[2,107] >= 0
 accuracy_(2,_sk_35)[3]: - 1.69921875 frac[3,107] + mem[3,107] >= 0
 accuracy_(2,_sk_35)[4]: - 1.69921875 frac[4,107] + mem[4,107] >= 0
 accuracy_(2,_sk_35)[5]: - 1.69921875 frac[5,107] + mem[5,107] >= 0
 accuracy_(2,_sk_35)[6]: - 1.69921875 frac[6,107] + mem[6,107] >= 0
 accuracy_(2,_sk_35)[7]: - 1.69921875 frac[7,107] + mem[7,107] >= 0
 accuracy_(2,_sk_35)[8]: - 1.69921875 frac[8,107] + mem[8,107] >= 0
 accuracy_(2,_sk_35)[9]: - 1.69921875 frac[9,107] + mem[9,107] >= 0
 accuracy_(2,_sk_35)[10]: - 1.69921875 frac[10,107] + mem[10,107] >= 0
 accuracy_(2,_sk_35)[11]: - 1.69921875 frac[11,107] + mem[11,107] >= 0
 accuracy_(2,_sk_35)[12]: - 1.69921875 frac[12,107] + mem[12,107] >= 0
 accuracy_(2,_sk_35)[13]: - 1.69921875 frac[13,107] + mem[13,107] >= 0
 accuracy_(2,_sk_35)[14]: - 1.69921875 frac[14,107] + mem[14,107] >= 0
 accuracy_(2,_sk_35)[15]: - 1.69921875 frac[15,107] + mem[15,107] >= 0
 accuracy_(2,_sk_35)[16]: - 1.69921875 frac[16,107] + mem[16,107] >= 0
 accuracy_(2,_sk_35)[17]: - 1.69921875 frac[17,107] + mem[17,107] >= 0
 accuracy_(2,_sk_35)[18]: - 1.69921875 frac[18,107] + mem[18,107] >= 0
 accuracy_(2,_sk_35)[19]: - 1.69921875 frac[19,107] + mem[19,107] >= 0
 accuracy_(2,_sk_35)[20]: - 1.69921875 frac[20,107] + mem[20,107] >= 0
 accuracy_(2,_sk_35)[21]: - 1.69921875 frac[21,107] + mem[21,107] >= 0
 accuracy_(2,_sk_35)[22]: - 1.69921875 frac[22,107] + mem[22,107] >= 0
 accuracy_(0,_sk_36)[0]: - 1.69921875 frac[0,108] + mem[0,108] >= 0
 accuracy_(0,_sk_36)[1]: - 1.69921875 frac[1,108] + mem[1,108] >= 0
 accuracy_(0,_sk_36)[2]: - 1.69921875 frac[2,108] + mem[2,108] >= 0
 accuracy_(0,_sk_36)[3]: - 1.69921875 frac[3,108] + mem[3,108] >= 0
 accuracy_(0,_sk_36)[4]: - 1.69921875 frac[4,108] + mem[4,108] >= 0
 accuracy_(0,_sk_36)[5]: - 1.69921875 frac[5,108] + mem[5,108] >= 0
 accuracy_(0,_sk_36)[6]: - 1.69921875 frac[6,108] + mem[6,108] >= 0
 accuracy_(0,_sk_36)[7]: - 1.69921875 frac[7,108] + mem[7,108] >= 0
 accuracy_(0,_sk_36)[8]: - 1.69921875 frac[8,108] + mem[8,108] >= 0
 accuracy_(0,_sk_36)[9]: - 1.69921875 frac[9,108] + mem[9,108] >= 0
 accuracy_(0,_sk_36)[10]: - 1.69921875 frac[10,108] + mem[10,108] >= 0
 accuracy_(0,_sk_36)[11]: - 1.69921875 frac[11,108] + mem[11,108] >= 0
 accuracy_(0,_sk_36)[12]: - 1.69921875 frac[12,108] + mem[12,108] >= 0
 accuracy_(0,_sk_36)[13]: - 1.69921875 frac[13,108] + mem[13,108] >= 0
 accuracy_(0,_sk_36)[14]: - 1.69921875 frac[14,108] + mem[14,108] >= 0
 accuracy_(0,_sk_36)[15]: - 1.69921875 frac[15,108] + mem[15,108] >= 0
 accuracy_(0,_sk_36)[16]: - 1.69921875 frac[16,108] + mem[16,108] >= 0
 accuracy_(0,_sk_36)[17]: - 1.69921875 frac[17,108] + mem[17,108] >= 0
 accuracy_(0,_sk_36)[18]: - 1.69921875 frac[18,108] + mem[18,108] >= 0
 accuracy_(0,_sk_36)[19]: - 1.69921875 frac[19,108] + mem[19,108] >= 0
 accuracy_(0,_sk_36)[20]: - 1.69921875 frac[20,108] + mem[20,108] >= 0
 accuracy_(0,_sk_36)[21]: - 1.69921875 frac[21,108] + mem[21,108] >= 0
 accuracy_(0,_sk_36)[22]: - 1.69921875 frac[22,108] + mem[22,108] >= 0
 accuracy_(1,_sk_36)[0]: - 1.69921875 frac[0,109] + mem[0,109] >= 0
 accuracy_(1,_sk_36)[1]: - 1.69921875 frac[1,109] + mem[1,109] >= 0
 accuracy_(1,_sk_36)[2]: - 1.69921875 frac[2,109] + mem[2,109] >= 0
 accuracy_(1,_sk_36)[3]: - 1.69921875 frac[3,109] + mem[3,109] >= 0
 accuracy_(1,_sk_36)[4]: - 1.69921875 frac[4,109] + mem[4,109] >= 0
 accuracy_(1,_sk_36)[5]: - 1.69921875 frac[5,109] + mem[5,109] >= 0
 accuracy_(1,_sk_36)[6]: - 1.69921875 frac[6,109] + mem[6,109] >= 0
 accuracy_(1,_sk_36)[7]: - 1.69921875 frac[7,109] + mem[7,109] >= 0
 accuracy_(1,_sk_36)[8]: - 1.69921875 frac[8,109] + mem[8,109] >= 0
 accuracy_(1,_sk_36)[9]: - 1.69921875 frac[9,109] + mem[9,109] >= 0
 accuracy_(1,_sk_36)[10]: - 1.69921875 frac[10,109] + mem[10,109] >= 0
 accuracy_(1,_sk_36)[11]: - 1.69921875 frac[11,109] + mem[11,109] >= 0
 accuracy_(1,_sk_36)[12]: - 1.69921875 frac[12,109] + mem[12,109] >= 0
 accuracy_(1,_sk_36)[13]: - 1.69921875 frac[13,109] + mem[13,109] >= 0
 accuracy_(1,_sk_36)[14]: - 1.69921875 frac[14,109] + mem[14,109] >= 0
 accuracy_(1,_sk_36)[15]: - 1.69921875 frac[15,109] + mem[15,109] >= 0
 accuracy_(1,_sk_36)[16]: - 1.69921875 frac[16,109] + mem[16,109] >= 0
 accuracy_(1,_sk_36)[17]: - 1.69921875 frac[17,109] + mem[17,109] >= 0
 accuracy_(1,_sk_36)[18]: - 1.69921875 frac[18,109] + mem[18,109] >= 0
 accuracy_(1,_sk_36)[19]: - 1.69921875 frac[19,109] + mem[19,109] >= 0
 accuracy_(1,_sk_36)[20]: - 1.69921875 frac[20,109] + mem[20,109] >= 0
 accuracy_(1,_sk_36)[21]: - 1.69921875 frac[21,109] + mem[21,109] >= 0
 accuracy_(1,_sk_36)[22]: - 1.69921875 frac[22,109] + mem[22,109] >= 0
 accuracy_(2,_sk_36)[0]: - 1.69921875 frac[0,110] + mem[0,110] >= 0
 accuracy_(2,_sk_36)[1]: - 1.69921875 frac[1,110] + mem[1,110] >= 0
 accuracy_(2,_sk_36)[2]: - 1.69921875 frac[2,110] + mem[2,110] >= 0
 accuracy_(2,_sk_36)[3]: - 1.69921875 frac[3,110] + mem[3,110] >= 0
 accuracy_(2,_sk_36)[4]: - 1.69921875 frac[4,110] + mem[4,110] >= 0
 accuracy_(2,_sk_36)[5]: - 1.69921875 frac[5,110] + mem[5,110] >= 0
 accuracy_(2,_sk_36)[6]: - 1.69921875 frac[6,110] + mem[6,110] >= 0
 accuracy_(2,_sk_36)[7]: - 1.69921875 frac[7,110] + mem[7,110] >= 0
 accuracy_(2,_sk_36)[8]: - 1.69921875 frac[8,110] + mem[8,110] >= 0
 accuracy_(2,_sk_36)[9]: - 1.69921875 frac[9,110] + mem[9,110] >= 0
 accuracy_(2,_sk_36)[10]: - 1.69921875 frac[10,110] + mem[10,110] >= 0
 accuracy_(2,_sk_36)[11]: - 1.69921875 frac[11,110] + mem[11,110] >= 0
 accuracy_(2,_sk_36)[12]: - 1.69921875 frac[12,110] + mem[12,110] >= 0
 accuracy_(2,_sk_36)[13]: - 1.69921875 frac[13,110] + mem[13,110] >= 0
 accuracy_(2,_sk_36)[14]: - 1.69921875 frac[14,110] + mem[14,110] >= 0
 accuracy_(2,_sk_36)[15]: - 1.69921875 frac[15,110] + mem[15,110] >= 0
 accuracy_(2,_sk_36)[16]: - 1.69921875 frac[16,110] + mem[16,110] >= 0
 accuracy_(2,_sk_36)[17]: - 1.69921875 frac[17,110] + mem[17,110] >= 0
 accuracy_(2,_sk_36)[18]: - 1.69921875 frac[18,110] + mem[18,110] >= 0
 accuracy_(2,_sk_36)[19]: - 1.69921875 frac[19,110] + mem[19,110] >= 0
 accuracy_(2,_sk_36)[20]: - 1.69921875 frac[20,110] + mem[20,110] >= 0
 accuracy_(2,_sk_36)[21]: - 1.69921875 frac[21,110] + mem[21,110] >= 0
 accuracy_(2,_sk_36)[22]: - 1.69921875 frac[22,110] + mem[22,110] >= 0
 accuracy_(0,_sk_37)[0]: - 1.69921875 frac[0,111] + mem[0,111] >= 0
 accuracy_(0,_sk_37)[1]: - 1.69921875 frac[1,111] + mem[1,111] >= 0
 accuracy_(0,_sk_37)[2]: - 1.69921875 frac[2,111] + mem[2,111] >= 0
 accuracy_(0,_sk_37)[3]: - 1.69921875 frac[3,111] + mem[3,111] >= 0
 accuracy_(0,_sk_37)[4]: - 1.69921875 frac[4,111] + mem[4,111] >= 0
 accuracy_(0,_sk_37)[5]: - 1.69921875 frac[5,111] + mem[5,111] >= 0
 accuracy_(0,_sk_37)[6]: - 1.69921875 frac[6,111] + mem[6,111] >= 0
 accuracy_(0,_sk_37)[7]: - 1.69921875 frac[7,111] + mem[7,111] >= 0
 accuracy_(0,_sk_37)[8]: - 1.69921875 frac[8,111] + mem[8,111] >= 0
 accuracy_(0,_sk_37)[9]: - 1.69921875 frac[9,111] + mem[9,111] >= 0
 accuracy_(0,_sk_37)[10]: - 1.69921875 frac[10,111] + mem[10,111] >= 0
 accuracy_(0,_sk_37)[11]: - 1.69921875 frac[11,111] + mem[11,111] >= 0
 accuracy_(0,_sk_37)[12]: - 1.69921875 frac[12,111] + mem[12,111] >= 0
 accuracy_(0,_sk_37)[13]: - 1.69921875 frac[13,111] + mem[13,111] >= 0
 accuracy_(0,_sk_37)[14]: - 1.69921875 frac[14,111] + mem[14,111] >= 0
 accuracy_(0,_sk_37)[15]: - 1.69921875 frac[15,111] + mem[15,111] >= 0
 accuracy_(0,_sk_37)[16]: - 1.69921875 frac[16,111] + mem[16,111] >= 0
 accuracy_(0,_sk_37)[17]: - 1.69921875 frac[17,111] + mem[17,111] >= 0
 accuracy_(0,_sk_37)[18]: - 1.69921875 frac[18,111] + mem[18,111] >= 0
 accuracy_(0,_sk_37)[19]: - 1.69921875 frac[19,111] + mem[19,111] >= 0
 accuracy_(0,_sk_37)[20]: - 1.69921875 frac[20,111] + mem[20,111] >= 0
 accuracy_(0,_sk_37)[21]: - 1.69921875 frac[21,111] + mem[21,111] >= 0
 accuracy_(0,_sk_37)[22]: - 1.69921875 frac[22,111] + mem[22,111] >= 0
 accuracy_(1,_sk_37)[0]: - 1.69921875 frac[0,112] + mem[0,112] >= 0
 accuracy_(1,_sk_37)[1]: - 1.69921875 frac[1,112] + mem[1,112] >= 0
 accuracy_(1,_sk_37)[2]: - 1.69921875 frac[2,112] + mem[2,112] >= 0
 accuracy_(1,_sk_37)[3]: - 1.69921875 frac[3,112] + mem[3,112] >= 0
 accuracy_(1,_sk_37)[4]: - 1.69921875 frac[4,112] + mem[4,112] >= 0
 accuracy_(1,_sk_37)[5]: - 1.69921875 frac[5,112] + mem[5,112] >= 0
 accuracy_(1,_sk_37)[6]: - 1.69921875 frac[6,112] + mem[6,112] >= 0
 accuracy_(1,_sk_37)[7]: - 1.69921875 frac[7,112] + mem[7,112] >= 0
 accuracy_(1,_sk_37)[8]: - 1.69921875 frac[8,112] + mem[8,112] >= 0
 accuracy_(1,_sk_37)[9]: - 1.69921875 frac[9,112] + mem[9,112] >= 0
 accuracy_(1,_sk_37)[10]: - 1.69921875 frac[10,112] + mem[10,112] >= 0
 accuracy_(1,_sk_37)[11]: - 1.69921875 frac[11,112] + mem[11,112] >= 0
 accuracy_(1,_sk_37)[12]: - 1.69921875 frac[12,112] + mem[12,112] >= 0
 accuracy_(1,_sk_37)[13]: - 1.69921875 frac[13,112] + mem[13,112] >= 0
 accuracy_(1,_sk_37)[14]: - 1.69921875 frac[14,112] + mem[14,112] >= 0
 accuracy_(1,_sk_37)[15]: - 1.69921875 frac[15,112] + mem[15,112] >= 0
 accuracy_(1,_sk_37)[16]: - 1.69921875 frac[16,112] + mem[16,112] >= 0
 accuracy_(1,_sk_37)[17]: - 1.69921875 frac[17,112] + mem[17,112] >= 0
 accuracy_(1,_sk_37)[18]: - 1.69921875 frac[18,112] + mem[18,112] >= 0
 accuracy_(1,_sk_37)[19]: - 1.69921875 frac[19,112] + mem[19,112] >= 0
 accuracy_(1,_sk_37)[20]: - 1.69921875 frac[20,112] + mem[20,112] >= 0
 accuracy_(1,_sk_37)[21]: - 1.69921875 frac[21,112] + mem[21,112] >= 0
 accuracy_(1,_sk_37)[22]: - 1.69921875 frac[22,112] + mem[22,112] >= 0
 accuracy_(2,_sk_37)[0]: - 1.69921875 frac[0,113] + mem[0,113] >= 0
 accuracy_(2,_sk_37)[1]: - 1.69921875 frac[1,113] + mem[1,113] >= 0
 accuracy_(2,_sk_37)[2]: - 1.69921875 frac[2,113] + mem[2,113] >= 0
 accuracy_(2,_sk_37)[3]: - 1.69921875 frac[3,113] + mem[3,113] >= 0
 accuracy_(2,_sk_37)[4]: - 1.69921875 frac[4,113] + mem[4,113] >= 0
 accuracy_(2,_sk_37)[5]: - 1.69921875 frac[5,113] + mem[5,113] >= 0
 accuracy_(2,_sk_37)[6]: - 1.69921875 frac[6,113] + mem[6,113] >= 0
 accuracy_(2,_sk_37)[7]: - 1.69921875 frac[7,113] + mem[7,113] >= 0
 accuracy_(2,_sk_37)[8]: - 1.69921875 frac[8,113] + mem[8,113] >= 0
 accuracy_(2,_sk_37)[9]: - 1.69921875 frac[9,113] + mem[9,113] >= 0
 accuracy_(2,_sk_37)[10]: - 1.69921875 frac[10,113] + mem[10,113] >= 0
 accuracy_(2,_sk_37)[11]: - 1.69921875 frac[11,113] + mem[11,113] >= 0
 accuracy_(2,_sk_37)[12]: - 1.69921875 frac[12,113] + mem[12,113] >= 0
 accuracy_(2,_sk_37)[13]: - 1.69921875 frac[13,113] + mem[13,113] >= 0
 accuracy_(2,_sk_37)[14]: - 1.69921875 frac[14,113] + mem[14,113] >= 0
 accuracy_(2,_sk_37)[15]: - 1.69921875 frac[15,113] + mem[15,113] >= 0
 accuracy_(2,_sk_37)[16]: - 1.69921875 frac[16,113] + mem[16,113] >= 0
 accuracy_(2,_sk_37)[17]: - 1.69921875 frac[17,113] + mem[17,113] >= 0
 accuracy_(2,_sk_37)[18]: - 1.69921875 frac[18,113] + mem[18,113] >= 0
 accuracy_(2,_sk_37)[19]: - 1.69921875 frac[19,113] + mem[19,113] >= 0
 accuracy_(2,_sk_37)[20]: - 1.69921875 frac[20,113] + mem[20,113] >= 0
 accuracy_(2,_sk_37)[21]: - 1.69921875 frac[21,113] + mem[21,113] >= 0
 accuracy_(2,_sk_37)[22]: - 1.69921875 frac[22,113] + mem[22,113] >= 0
 accuracy_(0,_sk_38)[0]: - 1.69921875 frac[0,114] + mem[0,114] >= 0
 accuracy_(0,_sk_38)[1]: - 1.69921875 frac[1,114] + mem[1,114] >= 0
 accuracy_(0,_sk_38)[2]: - 1.69921875 frac[2,114] + mem[2,114] >= 0
 accuracy_(0,_sk_38)[3]: - 1.69921875 frac[3,114] + mem[3,114] >= 0
 accuracy_(0,_sk_38)[4]: - 1.69921875 frac[4,114] + mem[4,114] >= 0
 accuracy_(0,_sk_38)[5]: - 1.69921875 frac[5,114] + mem[5,114] >= 0
 accuracy_(0,_sk_38)[6]: - 1.69921875 frac[6,114] + mem[6,114] >= 0
 accuracy_(0,_sk_38)[7]: - 1.69921875 frac[7,114] + mem[7,114] >= 0
 accuracy_(0,_sk_38)[8]: - 1.69921875 frac[8,114] + mem[8,114] >= 0
 accuracy_(0,_sk_38)[9]: - 1.69921875 frac[9,114] + mem[9,114] >= 0
 accuracy_(0,_sk_38)[10]: - 1.69921875 frac[10,114] + mem[10,114] >= 0
 accuracy_(0,_sk_38)[11]: - 1.69921875 frac[11,114] + mem[11,114] >= 0
 accuracy_(0,_sk_38)[12]: - 1.69921875 frac[12,114] + mem[12,114] >= 0
 accuracy_(0,_sk_38)[13]: - 1.69921875 frac[13,114] + mem[13,114] >= 0
 accuracy_(0,_sk_38)[14]: - 1.69921875 frac[14,114] + mem[14,114] >= 0
 accuracy_(0,_sk_38)[15]: - 1.69921875 frac[15,114] + mem[15,114] >= 0
 accuracy_(0,_sk_38)[16]: - 1.69921875 frac[16,114] + mem[16,114] >= 0
 accuracy_(0,_sk_38)[17]: - 1.69921875 frac[17,114] + mem[17,114] >= 0
 accuracy_(0,_sk_38)[18]: - 1.69921875 frac[18,114] + mem[18,114] >= 0
 accuracy_(0,_sk_38)[19]: - 1.69921875 frac[19,114] + mem[19,114] >= 0
 accuracy_(0,_sk_38)[20]: - 1.69921875 frac[20,114] + mem[20,114] >= 0
 accuracy_(0,_sk_38)[21]: - 1.69921875 frac[21,114] + mem[21,114] >= 0
 accuracy_(0,_sk_38)[22]: - 1.69921875 frac[22,114] + mem[22,114] >= 0
 accuracy_(1,_sk_38)[0]: - 1.69921875 frac[0,115] + mem[0,115] >= 0
 accuracy_(1,_sk_38)[1]: - 1.69921875 frac[1,115] + mem[1,115] >= 0
 accuracy_(1,_sk_38)[2]: - 1.69921875 frac[2,115] + mem[2,115] >= 0
 accuracy_(1,_sk_38)[3]: - 1.69921875 frac[3,115] + mem[3,115] >= 0
 accuracy_(1,_sk_38)[4]: - 1.69921875 frac[4,115] + mem[4,115] >= 0
 accuracy_(1,_sk_38)[5]: - 1.69921875 frac[5,115] + mem[5,115] >= 0
 accuracy_(1,_sk_38)[6]: - 1.69921875 frac[6,115] + mem[6,115] >= 0
 accuracy_(1,_sk_38)[7]: - 1.69921875 frac[7,115] + mem[7,115] >= 0
 accuracy_(1,_sk_38)[8]: - 1.69921875 frac[8,115] + mem[8,115] >= 0
 accuracy_(1,_sk_38)[9]: - 1.69921875 frac[9,115] + mem[9,115] >= 0
 accuracy_(1,_sk_38)[10]: - 1.69921875 frac[10,115] + mem[10,115] >= 0
 accuracy_(1,_sk_38)[11]: - 1.69921875 frac[11,115] + mem[11,115] >= 0
 accuracy_(1,_sk_38)[12]: - 1.69921875 frac[12,115] + mem[12,115] >= 0
 accuracy_(1,_sk_38)[13]: - 1.69921875 frac[13,115] + mem[13,115] >= 0
 accuracy_(1,_sk_38)[14]: - 1.69921875 frac[14,115] + mem[14,115] >= 0
 accuracy_(1,_sk_38)[15]: - 1.69921875 frac[15,115] + mem[15,115] >= 0
 accuracy_(1,_sk_38)[16]: - 1.69921875 frac[16,115] + mem[16,115] >= 0
 accuracy_(1,_sk_38)[17]: - 1.69921875 frac[17,115] + mem[17,115] >= 0
 accuracy_(1,_sk_38)[18]: - 1.69921875 frac[18,115] + mem[18,115] >= 0
 accuracy_(1,_sk_38)[19]: - 1.69921875 frac[19,115] + mem[19,115] >= 0
 accuracy_(1,_sk_38)[20]: - 1.69921875 frac[20,115] + mem[20,115] >= 0
 accuracy_(1,_sk_38)[21]: - 1.69921875 frac[21,115] + mem[21,115] >= 0
 accuracy_(1,_sk_38)[22]: - 1.69921875 frac[22,115] + mem[22,115] >= 0
 accuracy_(2,_sk_38)[0]: - 1.69921875 frac[0,116] + mem[0,116] >= 0
 accuracy_(2,_sk_38)[1]: - 1.69921875 frac[1,116] + mem[1,116] >= 0
 accuracy_(2,_sk_38)[2]: - 1.69921875 frac[2,116] + mem[2,116] >= 0
 accuracy_(2,_sk_38)[3]: - 1.69921875 frac[3,116] + mem[3,116] >= 0
 accuracy_(2,_sk_38)[4]: - 1.69921875 frac[4,116] + mem[4,116] >= 0
 accuracy_(2,_sk_38)[5]: - 1.69921875 frac[5,116] + mem[5,116] >= 0
 accuracy_(2,_sk_38)[6]: - 1.69921875 frac[6,116] + mem[6,116] >= 0
 accuracy_(2,_sk_38)[7]: - 1.69921875 frac[7,116] + mem[7,116] >= 0
 accuracy_(2,_sk_38)[8]: - 1.69921875 frac[8,116] + mem[8,116] >= 0
 accuracy_(2,_sk_38)[9]: - 1.69921875 frac[9,116] + mem[9,116] >= 0
 accuracy_(2,_sk_38)[10]: - 1.69921875 frac[10,116] + mem[10,116] >= 0
 accuracy_(2,_sk_38)[11]: - 1.69921875 frac[11,116] + mem[11,116] >= 0
 accuracy_(2,_sk_38)[12]: - 1.69921875 frac[12,116] + mem[12,116] >= 0
 accuracy_(2,_sk_38)[13]: - 1.69921875 frac[13,116] + mem[13,116] >= 0
 accuracy_(2,_sk_38)[14]: - 1.69921875 frac[14,116] + mem[14,116] >= 0
 accuracy_(2,_sk_38)[15]: - 1.69921875 frac[15,116] + mem[15,116] >= 0
 accuracy_(2,_sk_38)[16]: - 1.69921875 frac[16,116] + mem[16,116] >= 0
 accuracy_(2,_sk_38)[17]: - 1.69921875 frac[17,116] + mem[17,116] >= 0
 accuracy_(2,_sk_38)[18]: - 1.69921875 frac[18,116] + mem[18,116] >= 0
 accuracy_(2,_sk_38)[19]: - 1.69921875 frac[19,116] + mem[19,116] >= 0
 accuracy_(2,_sk_38)[20]: - 1.69921875 frac[20,116] + mem[20,116] >= 0
 accuracy_(2,_sk_38)[21]: - 1.69921875 frac[21,116] + mem[21,116] >= 0
 accuracy_(2,_sk_38)[22]: - 1.69921875 frac[22,116] + mem[22,116] >= 0
 accuracy_(0,_sk_39)[0]: - 1.69921875 frac[0,117] + mem[0,117] >= 0
 accuracy_(0,_sk_39)[1]: - 1.69921875 frac[1,117] + mem[1,117] >= 0
 accuracy_(0,_sk_39)[2]: - 1.69921875 frac[2,117] + mem[2,117] >= 0
 accuracy_(0,_sk_39)[3]: - 1.69921875 frac[3,117] + mem[3,117] >= 0
 accuracy_(0,_sk_39)[4]: - 1.69921875 frac[4,117] + mem[4,117] >= 0
 accuracy_(0,_sk_39)[5]: - 1.69921875 frac[5,117] + mem[5,117] >= 0
 accuracy_(0,_sk_39)[6]: - 1.69921875 frac[6,117] + mem[6,117] >= 0
 accuracy_(0,_sk_39)[7]: - 1.69921875 frac[7,117] + mem[7,117] >= 0
 accuracy_(0,_sk_39)[8]: - 1.69921875 frac[8,117] + mem[8,117] >= 0
 accuracy_(0,_sk_39)[9]: - 1.69921875 frac[9,117] + mem[9,117] >= 0
 accuracy_(0,_sk_39)[10]: - 1.69921875 frac[10,117] + mem[10,117] >= 0
 accuracy_(0,_sk_39)[11]: - 1.69921875 frac[11,117] + mem[11,117] >= 0
 accuracy_(0,_sk_39)[12]: - 1.69921875 frac[12,117] + mem[12,117] >= 0
 accuracy_(0,_sk_39)[13]: - 1.69921875 frac[13,117] + mem[13,117] >= 0
 accuracy_(0,_sk_39)[14]: - 1.69921875 frac[14,117] + mem[14,117] >= 0
 accuracy_(0,_sk_39)[15]: - 1.69921875 frac[15,117] + mem[15,117] >= 0
 accuracy_(0,_sk_39)[16]: - 1.69921875 frac[16,117] + mem[16,117] >= 0
 accuracy_(0,_sk_39)[17]: - 1.69921875 frac[17,117] + mem[17,117] >= 0
 accuracy_(0,_sk_39)[18]: - 1.69921875 frac[18,117] + mem[18,117] >= 0
 accuracy_(0,_sk_39)[19]: - 1.69921875 frac[19,117] + mem[19,117] >= 0
 accuracy_(0,_sk_39)[20]: - 1.69921875 frac[20,117] + mem[20,117] >= 0
 accuracy_(0,_sk_39)[21]: - 1.69921875 frac[21,117] + mem[21,117] >= 0
 accuracy_(0,_sk_39)[22]: - 1.69921875 frac[22,117] + mem[22,117] >= 0
 accuracy_(1,_sk_39)[0]: - 1.69921875 frac[0,118] + mem[0,118] >= 0
 accuracy_(1,_sk_39)[1]: - 1.69921875 frac[1,118] + mem[1,118] >= 0
 accuracy_(1,_sk_39)[2]: - 1.69921875 frac[2,118] + mem[2,118] >= 0
 accuracy_(1,_sk_39)[3]: - 1.69921875 frac[3,118] + mem[3,118] >= 0
 accuracy_(1,_sk_39)[4]: - 1.69921875 frac[4,118] + mem[4,118] >= 0
 accuracy_(1,_sk_39)[5]: - 1.69921875 frac[5,118] + mem[5,118] >= 0
 accuracy_(1,_sk_39)[6]: - 1.69921875 frac[6,118] + mem[6,118] >= 0
 accuracy_(1,_sk_39)[7]: - 1.69921875 frac[7,118] + mem[7,118] >= 0
 accuracy_(1,_sk_39)[8]: - 1.69921875 frac[8,118] + mem[8,118] >= 0
 accuracy_(1,_sk_39)[9]: - 1.69921875 frac[9,118] + mem[9,118] >= 0
 accuracy_(1,_sk_39)[10]: - 1.69921875 frac[10,118] + mem[10,118] >= 0
 accuracy_(1,_sk_39)[11]: - 1.69921875 frac[11,118] + mem[11,118] >= 0
 accuracy_(1,_sk_39)[12]: - 1.69921875 frac[12,118] + mem[12,118] >= 0
 accuracy_(1,_sk_39)[13]: - 1.69921875 frac[13,118] + mem[13,118] >= 0
 accuracy_(1,_sk_39)[14]: - 1.69921875 frac[14,118] + mem[14,118] >= 0
 accuracy_(1,_sk_39)[15]: - 1.69921875 frac[15,118] + mem[15,118] >= 0
 accuracy_(1,_sk_39)[16]: - 1.69921875 frac[16,118] + mem[16,118] >= 0
 accuracy_(1,_sk_39)[17]: - 1.69921875 frac[17,118] + mem[17,118] >= 0
 accuracy_(1,_sk_39)[18]: - 1.69921875 frac[18,118] + mem[18,118] >= 0
 accuracy_(1,_sk_39)[19]: - 1.69921875 frac[19,118] + mem[19,118] >= 0
 accuracy_(1,_sk_39)[20]: - 1.69921875 frac[20,118] + mem[20,118] >= 0
 accuracy_(1,_sk_39)[21]: - 1.69921875 frac[21,118] + mem[21,118] >= 0
 accuracy_(1,_sk_39)[22]: - 1.69921875 frac[22,118] + mem[22,118] >= 0
 accuracy_(2,_sk_39)[0]: - 1.69921875 frac[0,119] + mem[0,119] >= 0
 accuracy_(2,_sk_39)[1]: - 1.69921875 frac[1,119] + mem[1,119] >= 0
 accuracy_(2,_sk_39)[2]: - 1.69921875 frac[2,119] + mem[2,119] >= 0
 accuracy_(2,_sk_39)[3]: - 1.69921875 frac[3,119] + mem[3,119] >= 0
 accuracy_(2,_sk_39)[4]: - 1.69921875 frac[4,119] + mem[4,119] >= 0
 accuracy_(2,_sk_39)[5]: - 1.69921875 frac[5,119] + mem[5,119] >= 0
 accuracy_(2,_sk_39)[6]: - 1.69921875 frac[6,119] + mem[6,119] >= 0
 accuracy_(2,_sk_39)[7]: - 1.69921875 frac[7,119] + mem[7,119] >= 0
 accuracy_(2,_sk_39)[8]: - 1.69921875 frac[8,119] + mem[8,119] >= 0
 accuracy_(2,_sk_39)[9]: - 1.69921875 frac[9,119] + mem[9,119] >= 0
 accuracy_(2,_sk_39)[10]: - 1.69921875 frac[10,119] + mem[10,119] >= 0
 accuracy_(2,_sk_39)[11]: - 1.69921875 frac[11,119] + mem[11,119] >= 0
 accuracy_(2,_sk_39)[12]: - 1.69921875 frac[12,119] + mem[12,119] >= 0
 accuracy_(2,_sk_39)[13]: - 1.69921875 frac[13,119] + mem[13,119] >= 0
 accuracy_(2,_sk_39)[14]: - 1.69921875 frac[14,119] + mem[14,119] >= 0
 accuracy_(2,_sk_39)[15]: - 1.69921875 frac[15,119] + mem[15,119] >= 0
 accuracy_(2,_sk_39)[16]: - 1.69921875 frac[16,119] + mem[16,119] >= 0
 accuracy_(2,_sk_39)[17]: - 1.69921875 frac[17,119] + mem[17,119] >= 0
 accuracy_(2,_sk_39)[18]: - 1.69921875 frac[18,119] + mem[18,119] >= 0
 accuracy_(2,_sk_39)[19]: - 1.69921875 frac[19,119] + mem[19,119] >= 0
 accuracy_(2,_sk_39)[20]: - 1.69921875 frac[20,119] + mem[20,119] >= 0
 accuracy_(2,_sk_39)[21]: - 1.69921875 frac[21,119] + mem[21,119] >= 0
 accuracy_(2,_sk_39)[22]: - 1.69921875 frac[22,119] + mem[22,119] >= 0
 accuracy_(0,_sk_40)[0]: - 1.69921875 frac[0,120] + mem[0,120] >= 0
 accuracy_(0,_sk_40)[1]: - 1.69921875 frac[1,120] + mem[1,120] >= 0
 accuracy_(0,_sk_40)[2]: - 1.69921875 frac[2,120] + mem[2,120] >= 0
 accuracy_(0,_sk_40)[3]: - 1.69921875 frac[3,120] + mem[3,120] >= 0
 accuracy_(0,_sk_40)[4]: - 1.69921875 frac[4,120] + mem[4,120] >= 0
 accuracy_(0,_sk_40)[5]: - 1.69921875 frac[5,120] + mem[5,120] >= 0
 accuracy_(0,_sk_40)[6]: - 1.69921875 frac[6,120] + mem[6,120] >= 0
 accuracy_(0,_sk_40)[7]: - 1.69921875 frac[7,120] + mem[7,120] >= 0
 accuracy_(0,_sk_40)[8]: - 1.69921875 frac[8,120] + mem[8,120] >= 0
 accuracy_(0,_sk_40)[9]: - 1.69921875 frac[9,120] + mem[9,120] >= 0
 accuracy_(0,_sk_40)[10]: - 1.69921875 frac[10,120] + mem[10,120] >= 0
 accuracy_(0,_sk_40)[11]: - 1.69921875 frac[11,120] + mem[11,120] >= 0
 accuracy_(0,_sk_40)[12]: - 1.69921875 frac[12,120] + mem[12,120] >= 0
 accuracy_(0,_sk_40)[13]: - 1.69921875 frac[13,120] + mem[13,120] >= 0
 accuracy_(0,_sk_40)[14]: - 1.69921875 frac[14,120] + mem[14,120] >= 0
 accuracy_(0,_sk_40)[15]: - 1.69921875 frac[15,120] + mem[15,120] >= 0
 accuracy_(0,_sk_40)[16]: - 1.69921875 frac[16,120] + mem[16,120] >= 0
 accuracy_(0,_sk_40)[17]: - 1.69921875 frac[17,120] + mem[17,120] >= 0
 accuracy_(0,_sk_40)[18]: - 1.69921875 frac[18,120] + mem[18,120] >= 0
 accuracy_(0,_sk_40)[19]: - 1.69921875 frac[19,120] + mem[19,120] >= 0
 accuracy_(0,_sk_40)[20]: - 1.69921875 frac[20,120] + mem[20,120] >= 0
 accuracy_(0,_sk_40)[21]: - 1.69921875 frac[21,120] + mem[21,120] >= 0
 accuracy_(0,_sk_40)[22]: - 1.69921875 frac[22,120] + mem[22,120] >= 0
 accuracy_(1,_sk_40)[0]: - 1.69921875 frac[0,121] + mem[0,121] >= 0
 accuracy_(1,_sk_40)[1]: - 1.69921875 frac[1,121] + mem[1,121] >= 0
 accuracy_(1,_sk_40)[2]: - 1.69921875 frac[2,121] + mem[2,121] >= 0
 accuracy_(1,_sk_40)[3]: - 1.69921875 frac[3,121] + mem[3,121] >= 0
 accuracy_(1,_sk_40)[4]: - 1.69921875 frac[4,121] + mem[4,121] >= 0
 accuracy_(1,_sk_40)[5]: - 1.69921875 frac[5,121] + mem[5,121] >= 0
 accuracy_(1,_sk_40)[6]: - 1.69921875 frac[6,121] + mem[6,121] >= 0
 accuracy_(1,_sk_40)[7]: - 1.69921875 frac[7,121] + mem[7,121] >= 0
 accuracy_(1,_sk_40)[8]: - 1.69921875 frac[8,121] + mem[8,121] >= 0
 accuracy_(1,_sk_40)[9]: - 1.69921875 frac[9,121] + mem[9,121] >= 0
 accuracy_(1,_sk_40)[10]: - 1.69921875 frac[10,121] + mem[10,121] >= 0
 accuracy_(1,_sk_40)[11]: - 1.69921875 frac[11,121] + mem[11,121] >= 0
 accuracy_(1,_sk_40)[12]: - 1.69921875 frac[12,121] + mem[12,121] >= 0
 accuracy_(1,_sk_40)[13]: - 1.69921875 frac[13,121] + mem[13,121] >= 0
 accuracy_(1,_sk_40)[14]: - 1.69921875 frac[14,121] + mem[14,121] >= 0
 accuracy_(1,_sk_40)[15]: - 1.69921875 frac[15,121] + mem[15,121] >= 0
 accuracy_(1,_sk_40)[16]: - 1.69921875 frac[16,121] + mem[16,121] >= 0
 accuracy_(1,_sk_40)[17]: - 1.69921875 frac[17,121] + mem[17,121] >= 0
 accuracy_(1,_sk_40)[18]: - 1.69921875 frac[18,121] + mem[18,121] >= 0
 accuracy_(1,_sk_40)[19]: - 1.69921875 frac[19,121] + mem[19,121] >= 0
 accuracy_(1,_sk_40)[20]: - 1.69921875 frac[20,121] + mem[20,121] >= 0
 accuracy_(1,_sk_40)[21]: - 1.69921875 frac[21,121] + mem[21,121] >= 0
 accuracy_(1,_sk_40)[22]: - 1.69921875 frac[22,121] + mem[22,121] >= 0
 accuracy_(2,_sk_40)[0]: - 1.69921875 frac[0,122] + mem[0,122] >= 0
 accuracy_(2,_sk_40)[1]: - 1.69921875 frac[1,122] + mem[1,122] >= 0
 accuracy_(2,_sk_40)[2]: - 1.69921875 frac[2,122] + mem[2,122] >= 0
 accuracy_(2,_sk_40)[3]: - 1.69921875 frac[3,122] + mem[3,122] >= 0
 accuracy_(2,_sk_40)[4]: - 1.69921875 frac[4,122] + mem[4,122] >= 0
 accuracy_(2,_sk_40)[5]: - 1.69921875 frac[5,122] + mem[5,122] >= 0
 accuracy_(2,_sk_40)[6]: - 1.69921875 frac[6,122] + mem[6,122] >= 0
 accuracy_(2,_sk_40)[7]: - 1.69921875 frac[7,122] + mem[7,122] >= 0
 accuracy_(2,_sk_40)[8]: - 1.69921875 frac[8,122] + mem[8,122] >= 0
 accuracy_(2,_sk_40)[9]: - 1.69921875 frac[9,122] + mem[9,122] >= 0
 accuracy_(2,_sk_40)[10]: - 1.69921875 frac[10,122] + mem[10,122] >= 0
 accuracy_(2,_sk_40)[11]: - 1.69921875 frac[11,122] + mem[11,122] >= 0
 accuracy_(2,_sk_40)[12]: - 1.69921875 frac[12,122] + mem[12,122] >= 0
 accuracy_(2,_sk_40)[13]: - 1.69921875 frac[13,122] + mem[13,122] >= 0
 accuracy_(2,_sk_40)[14]: - 1.69921875 frac[14,122] + mem[14,122] >= 0
 accuracy_(2,_sk_40)[15]: - 1.69921875 frac[15,122] + mem[15,122] >= 0
 accuracy_(2,_sk_40)[16]: - 1.69921875 frac[16,122] + mem[16,122] >= 0
 accuracy_(2,_sk_40)[17]: - 1.69921875 frac[17,122] + mem[17,122] >= 0
 accuracy_(2,_sk_40)[18]: - 1.69921875 frac[18,122] + mem[18,122] >= 0
 accuracy_(2,_sk_40)[19]: - 1.69921875 frac[19,122] + mem[19,122] >= 0
 accuracy_(2,_sk_40)[20]: - 1.69921875 frac[20,122] + mem[20,122] >= 0
 accuracy_(2,_sk_40)[21]: - 1.69921875 frac[21,122] + mem[21,122] >= 0
 accuracy_(2,_sk_40)[22]: - 1.69921875 frac[22,122] + mem[22,122] >= 0
 accuracy_(0,_sk_41)[0]: - 1.69921875 frac[0,123] + mem[0,123] >= 0
 accuracy_(0,_sk_41)[1]: - 1.69921875 frac[1,123] + mem[1,123] >= 0
 accuracy_(0,_sk_41)[2]: - 1.69921875 frac[2,123] + mem[2,123] >= 0
 accuracy_(0,_sk_41)[3]: - 1.69921875 frac[3,123] + mem[3,123] >= 0
 accuracy_(0,_sk_41)[4]: - 1.69921875 frac[4,123] + mem[4,123] >= 0
 accuracy_(0,_sk_41)[5]: - 1.69921875 frac[5,123] + mem[5,123] >= 0
 accuracy_(0,_sk_41)[6]: - 1.69921875 frac[6,123] + mem[6,123] >= 0
 accuracy_(0,_sk_41)[7]: - 1.69921875 frac[7,123] + mem[7,123] >= 0
 accuracy_(0,_sk_41)[8]: - 1.69921875 frac[8,123] + mem[8,123] >= 0
 accuracy_(0,_sk_41)[9]: - 1.69921875 frac[9,123] + mem[9,123] >= 0
 accuracy_(0,_sk_41)[10]: - 1.69921875 frac[10,123] + mem[10,123] >= 0
 accuracy_(0,_sk_41)[11]: - 1.69921875 frac[11,123] + mem[11,123] >= 0
 accuracy_(0,_sk_41)[12]: - 1.69921875 frac[12,123] + mem[12,123] >= 0
 accuracy_(0,_sk_41)[13]: - 1.69921875 frac[13,123] + mem[13,123] >= 0
 accuracy_(0,_sk_41)[14]: - 1.69921875 frac[14,123] + mem[14,123] >= 0
 accuracy_(0,_sk_41)[15]: - 1.69921875 frac[15,123] + mem[15,123] >= 0
 accuracy_(0,_sk_41)[16]: - 1.69921875 frac[16,123] + mem[16,123] >= 0
 accuracy_(0,_sk_41)[17]: - 1.69921875 frac[17,123] + mem[17,123] >= 0
 accuracy_(0,_sk_41)[18]: - 1.69921875 frac[18,123] + mem[18,123] >= 0
 accuracy_(0,_sk_41)[19]: - 1.69921875 frac[19,123] + mem[19,123] >= 0
 accuracy_(0,_sk_41)[20]: - 1.69921875 frac[20,123] + mem[20,123] >= 0
 accuracy_(0,_sk_41)[21]: - 1.69921875 frac[21,123] + mem[21,123] >= 0
 accuracy_(0,_sk_41)[22]: - 1.69921875 frac[22,123] + mem[22,123] >= 0
 accuracy_(1,_sk_41)[0]: - 1.69921875 frac[0,124] + mem[0,124] >= 0
 accuracy_(1,_sk_41)[1]: - 1.69921875 frac[1,124] + mem[1,124] >= 0
 accuracy_(1,_sk_41)[2]: - 1.69921875 frac[2,124] + mem[2,124] >= 0
 accuracy_(1,_sk_41)[3]: - 1.69921875 frac[3,124] + mem[3,124] >= 0
 accuracy_(1,_sk_41)[4]: - 1.69921875 frac[4,124] + mem[4,124] >= 0
 accuracy_(1,_sk_41)[5]: - 1.69921875 frac[5,124] + mem[5,124] >= 0
 accuracy_(1,_sk_41)[6]: - 1.69921875 frac[6,124] + mem[6,124] >= 0
 accuracy_(1,_sk_41)[7]: - 1.69921875 frac[7,124] + mem[7,124] >= 0
 accuracy_(1,_sk_41)[8]: - 1.69921875 frac[8,124] + mem[8,124] >= 0
 accuracy_(1,_sk_41)[9]: - 1.69921875 frac[9,124] + mem[9,124] >= 0
 accuracy_(1,_sk_41)[10]: - 1.69921875 frac[10,124] + mem[10,124] >= 0
 accuracy_(1,_sk_41)[11]: - 1.69921875 frac[11,124] + mem[11,124] >= 0
 accuracy_(1,_sk_41)[12]: - 1.69921875 frac[12,124] + mem[12,124] >= 0
 accuracy_(1,_sk_41)[13]: - 1.69921875 frac[13,124] + mem[13,124] >= 0
 accuracy_(1,_sk_41)[14]: - 1.69921875 frac[14,124] + mem[14,124] >= 0
 accuracy_(1,_sk_41)[15]: - 1.69921875 frac[15,124] + mem[15,124] >= 0
 accuracy_(1,_sk_41)[16]: - 1.69921875 frac[16,124] + mem[16,124] >= 0
 accuracy_(1,_sk_41)[17]: - 1.69921875 frac[17,124] + mem[17,124] >= 0
 accuracy_(1,_sk_41)[18]: - 1.69921875 frac[18,124] + mem[18,124] >= 0
 accuracy_(1,_sk_41)[19]: - 1.69921875 frac[19,124] + mem[19,124] >= 0
 accuracy_(1,_sk_41)[20]: - 1.69921875 frac[20,124] + mem[20,124] >= 0
 accuracy_(1,_sk_41)[21]: - 1.69921875 frac[21,124] + mem[21,124] >= 0
 accuracy_(1,_sk_41)[22]: - 1.69921875 frac[22,124] + mem[22,124] >= 0
 accuracy_(2,_sk_41)[0]: - 1.69921875 frac[0,125] + mem[0,125] >= 0
 accuracy_(2,_sk_41)[1]: - 1.69921875 frac[1,125] + mem[1,125] >= 0
 accuracy_(2,_sk_41)[2]: - 1.69921875 frac[2,125] + mem[2,125] >= 0
 accuracy_(2,_sk_41)[3]: - 1.69921875 frac[3,125] + mem[3,125] >= 0
 accuracy_(2,_sk_41)[4]: - 1.69921875 frac[4,125] + mem[4,125] >= 0
 accuracy_(2,_sk_41)[5]: - 1.69921875 frac[5,125] + mem[5,125] >= 0
 accuracy_(2,_sk_41)[6]: - 1.69921875 frac[6,125] + mem[6,125] >= 0
 accuracy_(2,_sk_41)[7]: - 1.69921875 frac[7,125] + mem[7,125] >= 0
 accuracy_(2,_sk_41)[8]: - 1.69921875 frac[8,125] + mem[8,125] >= 0
 accuracy_(2,_sk_41)[9]: - 1.69921875 frac[9,125] + mem[9,125] >= 0
 accuracy_(2,_sk_41)[10]: - 1.69921875 frac[10,125] + mem[10,125] >= 0
 accuracy_(2,_sk_41)[11]: - 1.69921875 frac[11,125] + mem[11,125] >= 0
 accuracy_(2,_sk_41)[12]: - 1.69921875 frac[12,125] + mem[12,125] >= 0
 accuracy_(2,_sk_41)[13]: - 1.69921875 frac[13,125] + mem[13,125] >= 0
 accuracy_(2,_sk_41)[14]: - 1.69921875 frac[14,125] + mem[14,125] >= 0
 accuracy_(2,_sk_41)[15]: - 1.69921875 frac[15,125] + mem[15,125] >= 0
 accuracy_(2,_sk_41)[16]: - 1.69921875 frac[16,125] + mem[16,125] >= 0
 accuracy_(2,_sk_41)[17]: - 1.69921875 frac[17,125] + mem[17,125] >= 0
 accuracy_(2,_sk_41)[18]: - 1.69921875 frac[18,125] + mem[18,125] >= 0
 accuracy_(2,_sk_41)[19]: - 1.69921875 frac[19,125] + mem[19,125] >= 0
 accuracy_(2,_sk_41)[20]: - 1.69921875 frac[20,125] + mem[20,125] >= 0
 accuracy_(2,_sk_41)[21]: - 1.69921875 frac[21,125] + mem[21,125] >= 0
 accuracy_(2,_sk_41)[22]: - 1.69921875 frac[22,125] + mem[22,125] >= 0
 accuracy_(0,_sk_42)[0]: - 1.69921875 frac[0,126] + mem[0,126] >= 0
 accuracy_(0,_sk_42)[1]: - 1.69921875 frac[1,126] + mem[1,126] >= 0
 accuracy_(0,_sk_42)[2]: - 1.69921875 frac[2,126] + mem[2,126] >= 0
 accuracy_(0,_sk_42)[3]: - 1.69921875 frac[3,126] + mem[3,126] >= 0
 accuracy_(0,_sk_42)[4]: - 1.69921875 frac[4,126] + mem[4,126] >= 0
 accuracy_(0,_sk_42)[5]: - 1.69921875 frac[5,126] + mem[5,126] >= 0
 accuracy_(0,_sk_42)[6]: - 1.69921875 frac[6,126] + mem[6,126] >= 0
 accuracy_(0,_sk_42)[7]: - 1.69921875 frac[7,126] + mem[7,126] >= 0
 accuracy_(0,_sk_42)[8]: - 1.69921875 frac[8,126] + mem[8,126] >= 0
 accuracy_(0,_sk_42)[9]: - 1.69921875 frac[9,126] + mem[9,126] >= 0
 accuracy_(0,_sk_42)[10]: - 1.69921875 frac[10,126] + mem[10,126] >= 0
 accuracy_(0,_sk_42)[11]: - 1.69921875 frac[11,126] + mem[11,126] >= 0
 accuracy_(0,_sk_42)[12]: - 1.69921875 frac[12,126] + mem[12,126] >= 0
 accuracy_(0,_sk_42)[13]: - 1.69921875 frac[13,126] + mem[13,126] >= 0
 accuracy_(0,_sk_42)[14]: - 1.69921875 frac[14,126] + mem[14,126] >= 0
 accuracy_(0,_sk_42)[15]: - 1.69921875 frac[15,126] + mem[15,126] >= 0
 accuracy_(0,_sk_42)[16]: - 1.69921875 frac[16,126] + mem[16,126] >= 0
 accuracy_(0,_sk_42)[17]: - 1.69921875 frac[17,126] + mem[17,126] >= 0
 accuracy_(0,_sk_42)[18]: - 1.69921875 frac[18,126] + mem[18,126] >= 0
 accuracy_(0,_sk_42)[19]: - 1.69921875 frac[19,126] + mem[19,126] >= 0
 accuracy_(0,_sk_42)[20]: - 1.69921875 frac[20,126] + mem[20,126] >= 0
 accuracy_(0,_sk_42)[21]: - 1.69921875 frac[21,126] + mem[21,126] >= 0
 accuracy_(0,_sk_42)[22]: - 1.69921875 frac[22,126] + mem[22,126] >= 0
 accuracy_(1,_sk_42)[0]: - 1.69921875 frac[0,127] + mem[0,127] >= 0
 accuracy_(1,_sk_42)[1]: - 1.69921875 frac[1,127] + mem[1,127] >= 0
 accuracy_(1,_sk_42)[2]: - 1.69921875 frac[2,127] + mem[2,127] >= 0
 accuracy_(1,_sk_42)[3]: - 1.69921875 frac[3,127] + mem[3,127] >= 0
 accuracy_(1,_sk_42)[4]: - 1.69921875 frac[4,127] + mem[4,127] >= 0
 accuracy_(1,_sk_42)[5]: - 1.69921875 frac[5,127] + mem[5,127] >= 0
 accuracy_(1,_sk_42)[6]: - 1.69921875 frac[6,127] + mem[6,127] >= 0
 accuracy_(1,_sk_42)[7]: - 1.69921875 frac[7,127] + mem[7,127] >= 0
 accuracy_(1,_sk_42)[8]: - 1.69921875 frac[8,127] + mem[8,127] >= 0
 accuracy_(1,_sk_42)[9]: - 1.69921875 frac[9,127] + mem[9,127] >= 0
 accuracy_(1,_sk_42)[10]: - 1.69921875 frac[10,127] + mem[10,127] >= 0
 accuracy_(1,_sk_42)[11]: - 1.69921875 frac[11,127] + mem[11,127] >= 0
 accuracy_(1,_sk_42)[12]: - 1.69921875 frac[12,127] + mem[12,127] >= 0
 accuracy_(1,_sk_42)[13]: - 1.69921875 frac[13,127] + mem[13,127] >= 0
 accuracy_(1,_sk_42)[14]: - 1.69921875 frac[14,127] + mem[14,127] >= 0
 accuracy_(1,_sk_42)[15]: - 1.69921875 frac[15,127] + mem[15,127] >= 0
 accuracy_(1,_sk_42)[16]: - 1.69921875 frac[16,127] + mem[16,127] >= 0
 accuracy_(1,_sk_42)[17]: - 1.69921875 frac[17,127] + mem[17,127] >= 0
 accuracy_(1,_sk_42)[18]: - 1.69921875 frac[18,127] + mem[18,127] >= 0
 accuracy_(1,_sk_42)[19]: - 1.69921875 frac[19,127] + mem[19,127] >= 0
 accuracy_(1,_sk_42)[20]: - 1.69921875 frac[20,127] + mem[20,127] >= 0
 accuracy_(1,_sk_42)[21]: - 1.69921875 frac[21,127] + mem[21,127] >= 0
 accuracy_(1,_sk_42)[22]: - 1.69921875 frac[22,127] + mem[22,127] >= 0
 accuracy_(2,_sk_42)[0]: - 1.69921875 frac[0,128] + mem[0,128] >= 0
 accuracy_(2,_sk_42)[1]: - 1.69921875 frac[1,128] + mem[1,128] >= 0
 accuracy_(2,_sk_42)[2]: - 1.69921875 frac[2,128] + mem[2,128] >= 0
 accuracy_(2,_sk_42)[3]: - 1.69921875 frac[3,128] + mem[3,128] >= 0
 accuracy_(2,_sk_42)[4]: - 1.69921875 frac[4,128] + mem[4,128] >= 0
 accuracy_(2,_sk_42)[5]: - 1.69921875 frac[5,128] + mem[5,128] >= 0
 accuracy_(2,_sk_42)[6]: - 1.69921875 frac[6,128] + mem[6,128] >= 0
 accuracy_(2,_sk_42)[7]: - 1.69921875 frac[7,128] + mem[7,128] >= 0
 accuracy_(2,_sk_42)[8]: - 1.69921875 frac[8,128] + mem[8,128] >= 0
 accuracy_(2,_sk_42)[9]: - 1.69921875 frac[9,128] + mem[9,128] >= 0
 accuracy_(2,_sk_42)[10]: - 1.69921875 frac[10,128] + mem[10,128] >= 0
 accuracy_(2,_sk_42)[11]: - 1.69921875 frac[11,128] + mem[11,128] >= 0
 accuracy_(2,_sk_42)[12]: - 1.69921875 frac[12,128] + mem[12,128] >= 0
 accuracy_(2,_sk_42)[13]: - 1.69921875 frac[13,128] + mem[13,128] >= 0
 accuracy_(2,_sk_42)[14]: - 1.69921875 frac[14,128] + mem[14,128] >= 0
 accuracy_(2,_sk_42)[15]: - 1.69921875 frac[15,128] + mem[15,128] >= 0
 accuracy_(2,_sk_42)[16]: - 1.69921875 frac[16,128] + mem[16,128] >= 0
 accuracy_(2,_sk_42)[17]: - 1.69921875 frac[17,128] + mem[17,128] >= 0
 accuracy_(2,_sk_42)[18]: - 1.69921875 frac[18,128] + mem[18,128] >= 0
 accuracy_(2,_sk_42)[19]: - 1.69921875 frac[19,128] + mem[19,128] >= 0
 accuracy_(2,_sk_42)[20]: - 1.69921875 frac[20,128] + mem[20,128] >= 0
 accuracy_(2,_sk_42)[21]: - 1.69921875 frac[21,128] + mem[21,128] >= 0
 accuracy_(2,_sk_42)[22]: - 1.69921875 frac[22,128] + mem[22,128] >= 0
 accuracy_(0,_sk_43)[0]: - 1.69921875 frac[0,129] + mem[0,129] >= 0
 accuracy_(0,_sk_43)[1]: - 1.69921875 frac[1,129] + mem[1,129] >= 0
 accuracy_(0,_sk_43)[2]: - 1.69921875 frac[2,129] + mem[2,129] >= 0
 accuracy_(0,_sk_43)[3]: - 1.69921875 frac[3,129] + mem[3,129] >= 0
 accuracy_(0,_sk_43)[4]: - 1.69921875 frac[4,129] + mem[4,129] >= 0
 accuracy_(0,_sk_43)[5]: - 1.69921875 frac[5,129] + mem[5,129] >= 0
 accuracy_(0,_sk_43)[6]: - 1.69921875 frac[6,129] + mem[6,129] >= 0
 accuracy_(0,_sk_43)[7]: - 1.69921875 frac[7,129] + mem[7,129] >= 0
 accuracy_(0,_sk_43)[8]: - 1.69921875 frac[8,129] + mem[8,129] >= 0
 accuracy_(0,_sk_43)[9]: - 1.69921875 frac[9,129] + mem[9,129] >= 0
 accuracy_(0,_sk_43)[10]: - 1.69921875 frac[10,129] + mem[10,129] >= 0
 accuracy_(0,_sk_43)[11]: - 1.69921875 frac[11,129] + mem[11,129] >= 0
 accuracy_(0,_sk_43)[12]: - 1.69921875 frac[12,129] + mem[12,129] >= 0
 accuracy_(0,_sk_43)[13]: - 1.69921875 frac[13,129] + mem[13,129] >= 0
 accuracy_(0,_sk_43)[14]: - 1.69921875 frac[14,129] + mem[14,129] >= 0
 accuracy_(0,_sk_43)[15]: - 1.69921875 frac[15,129] + mem[15,129] >= 0
 accuracy_(0,_sk_43)[16]: - 1.69921875 frac[16,129] + mem[16,129] >= 0
 accuracy_(0,_sk_43)[17]: - 1.69921875 frac[17,129] + mem[17,129] >= 0
 accuracy_(0,_sk_43)[18]: - 1.69921875 frac[18,129] + mem[18,129] >= 0
 accuracy_(0,_sk_43)[19]: - 1.69921875 frac[19,129] + mem[19,129] >= 0
 accuracy_(0,_sk_43)[20]: - 1.69921875 frac[20,129] + mem[20,129] >= 0
 accuracy_(0,_sk_43)[21]: - 1.69921875 frac[21,129] + mem[21,129] >= 0
 accuracy_(0,_sk_43)[22]: - 1.69921875 frac[22,129] + mem[22,129] >= 0
 accuracy_(1,_sk_43)[0]: - 1.69921875 frac[0,130] + mem[0,130] >= 0
 accuracy_(1,_sk_43)[1]: - 1.69921875 frac[1,130] + mem[1,130] >= 0
 accuracy_(1,_sk_43)[2]: - 1.69921875 frac[2,130] + mem[2,130] >= 0
 accuracy_(1,_sk_43)[3]: - 1.69921875 frac[3,130] + mem[3,130] >= 0
 accuracy_(1,_sk_43)[4]: - 1.69921875 frac[4,130] + mem[4,130] >= 0
 accuracy_(1,_sk_43)[5]: - 1.69921875 frac[5,130] + mem[5,130] >= 0
 accuracy_(1,_sk_43)[6]: - 1.69921875 frac[6,130] + mem[6,130] >= 0
 accuracy_(1,_sk_43)[7]: - 1.69921875 frac[7,130] + mem[7,130] >= 0
 accuracy_(1,_sk_43)[8]: - 1.69921875 frac[8,130] + mem[8,130] >= 0
 accuracy_(1,_sk_43)[9]: - 1.69921875 frac[9,130] + mem[9,130] >= 0
 accuracy_(1,_sk_43)[10]: - 1.69921875 frac[10,130] + mem[10,130] >= 0
 accuracy_(1,_sk_43)[11]: - 1.69921875 frac[11,130] + mem[11,130] >= 0
 accuracy_(1,_sk_43)[12]: - 1.69921875 frac[12,130] + mem[12,130] >= 0
 accuracy_(1,_sk_43)[13]: - 1.69921875 frac[13,130] + mem[13,130] >= 0
 accuracy_(1,_sk_43)[14]: - 1.69921875 frac[14,130] + mem[14,130] >= 0
 accuracy_(1,_sk_43)[15]: - 1.69921875 frac[15,130] + mem[15,130] >= 0
 accuracy_(1,_sk_43)[16]: - 1.69921875 frac[16,130] + mem[16,130] >= 0
 accuracy_(1,_sk_43)[17]: - 1.69921875 frac[17,130] + mem[17,130] >= 0
 accuracy_(1,_sk_43)[18]: - 1.69921875 frac[18,130] + mem[18,130] >= 0
 accuracy_(1,_sk_43)[19]: - 1.69921875 frac[19,130] + mem[19,130] >= 0
 accuracy_(1,_sk_43)[20]: - 1.69921875 frac[20,130] + mem[20,130] >= 0
 accuracy_(1,_sk_43)[21]: - 1.69921875 frac[21,130] + mem[21,130] >= 0
 accuracy_(1,_sk_43)[22]: - 1.69921875 frac[22,130] + mem[22,130] >= 0
 accuracy_(2,_sk_43)[0]: - 1.69921875 frac[0,131] + mem[0,131] >= 0
 accuracy_(2,_sk_43)[1]: - 1.69921875 frac[1,131] + mem[1,131] >= 0
 accuracy_(2,_sk_43)[2]: - 1.69921875 frac[2,131] + mem[2,131] >= 0
 accuracy_(2,_sk_43)[3]: - 1.69921875 frac[3,131] + mem[3,131] >= 0
 accuracy_(2,_sk_43)[4]: - 1.69921875 frac[4,131] + mem[4,131] >= 0
 accuracy_(2,_sk_43)[5]: - 1.69921875 frac[5,131] + mem[5,131] >= 0
 accuracy_(2,_sk_43)[6]: - 1.69921875 frac[6,131] + mem[6,131] >= 0
 accuracy_(2,_sk_43)[7]: - 1.69921875 frac[7,131] + mem[7,131] >= 0
 accuracy_(2,_sk_43)[8]: - 1.69921875 frac[8,131] + mem[8,131] >= 0
 accuracy_(2,_sk_43)[9]: - 1.69921875 frac[9,131] + mem[9,131] >= 0
 accuracy_(2,_sk_43)[10]: - 1.69921875 frac[10,131] + mem[10,131] >= 0
 accuracy_(2,_sk_43)[11]: - 1.69921875 frac[11,131] + mem[11,131] >= 0
 accuracy_(2,_sk_43)[12]: - 1.69921875 frac[12,131] + mem[12,131] >= 0
 accuracy_(2,_sk_43)[13]: - 1.69921875 frac[13,131] + mem[13,131] >= 0
 accuracy_(2,_sk_43)[14]: - 1.69921875 frac[14,131] + mem[14,131] >= 0
 accuracy_(2,_sk_43)[15]: - 1.69921875 frac[15,131] + mem[15,131] >= 0
 accuracy_(2,_sk_43)[16]: - 1.69921875 frac[16,131] + mem[16,131] >= 0
 accuracy_(2,_sk_43)[17]: - 1.69921875 frac[17,131] + mem[17,131] >= 0
 accuracy_(2,_sk_43)[18]: - 1.69921875 frac[18,131] + mem[18,131] >= 0
 accuracy_(2,_sk_43)[19]: - 1.69921875 frac[19,131] + mem[19,131] >= 0
 accuracy_(2,_sk_43)[20]: - 1.69921875 frac[20,131] + mem[20,131] >= 0
 accuracy_(2,_sk_43)[21]: - 1.69921875 frac[21,131] + mem[21,131] >= 0
 accuracy_(2,_sk_43)[22]: - 1.69921875 frac[22,131] + mem[22,131] >= 0
 accuracy_(0,_sk_44)[0]: - 1.69921875 frac[0,132] + mem[0,132] >= 0
 accuracy_(0,_sk_44)[1]: - 1.69921875 frac[1,132] + mem[1,132] >= 0
 accuracy_(0,_sk_44)[2]: - 1.69921875 frac[2,132] + mem[2,132] >= 0
 accuracy_(0,_sk_44)[3]: - 1.69921875 frac[3,132] + mem[3,132] >= 0
 accuracy_(0,_sk_44)[4]: - 1.69921875 frac[4,132] + mem[4,132] >= 0
 accuracy_(0,_sk_44)[5]: - 1.69921875 frac[5,132] + mem[5,132] >= 0
 accuracy_(0,_sk_44)[6]: - 1.69921875 frac[6,132] + mem[6,132] >= 0
 accuracy_(0,_sk_44)[7]: - 1.69921875 frac[7,132] + mem[7,132] >= 0
 accuracy_(0,_sk_44)[8]: - 1.69921875 frac[8,132] + mem[8,132] >= 0
 accuracy_(0,_sk_44)[9]: - 1.69921875 frac[9,132] + mem[9,132] >= 0
 accuracy_(0,_sk_44)[10]: - 1.69921875 frac[10,132] + mem[10,132] >= 0
 accuracy_(0,_sk_44)[11]: - 1.69921875 frac[11,132] + mem[11,132] >= 0
 accuracy_(0,_sk_44)[12]: - 1.69921875 frac[12,132] + mem[12,132] >= 0
 accuracy_(0,_sk_44)[13]: - 1.69921875 frac[13,132] + mem[13,132] >= 0
 accuracy_(0,_sk_44)[14]: - 1.69921875 frac[14,132] + mem[14,132] >= 0
 accuracy_(0,_sk_44)[15]: - 1.69921875 frac[15,132] + mem[15,132] >= 0
 accuracy_(0,_sk_44)[16]: - 1.69921875 frac[16,132] + mem[16,132] >= 0
 accuracy_(0,_sk_44)[17]: - 1.69921875 frac[17,132] + mem[17,132] >= 0
 accuracy_(0,_sk_44)[18]: - 1.69921875 frac[18,132] + mem[18,132] >= 0
 accuracy_(0,_sk_44)[19]: - 1.69921875 frac[19,132] + mem[19,132] >= 0
 accuracy_(0,_sk_44)[20]: - 1.69921875 frac[20,132] + mem[20,132] >= 0
 accuracy_(0,_sk_44)[21]: - 1.69921875 frac[21,132] + mem[21,132] >= 0
 accuracy_(0,_sk_44)[22]: - 1.69921875 frac[22,132] + mem[22,132] >= 0
 accuracy_(1,_sk_44)[0]: - 1.69921875 frac[0,133] + mem[0,133] >= 0
 accuracy_(1,_sk_44)[1]: - 1.69921875 frac[1,133] + mem[1,133] >= 0
 accuracy_(1,_sk_44)[2]: - 1.69921875 frac[2,133] + mem[2,133] >= 0
 accuracy_(1,_sk_44)[3]: - 1.69921875 frac[3,133] + mem[3,133] >= 0
 accuracy_(1,_sk_44)[4]: - 1.69921875 frac[4,133] + mem[4,133] >= 0
 accuracy_(1,_sk_44)[5]: - 1.69921875 frac[5,133] + mem[5,133] >= 0
 accuracy_(1,_sk_44)[6]: - 1.69921875 frac[6,133] + mem[6,133] >= 0
 accuracy_(1,_sk_44)[7]: - 1.69921875 frac[7,133] + mem[7,133] >= 0
 accuracy_(1,_sk_44)[8]: - 1.69921875 frac[8,133] + mem[8,133] >= 0
 accuracy_(1,_sk_44)[9]: - 1.69921875 frac[9,133] + mem[9,133] >= 0
 accuracy_(1,_sk_44)[10]: - 1.69921875 frac[10,133] + mem[10,133] >= 0
 accuracy_(1,_sk_44)[11]: - 1.69921875 frac[11,133] + mem[11,133] >= 0
 accuracy_(1,_sk_44)[12]: - 1.69921875 frac[12,133] + mem[12,133] >= 0
 accuracy_(1,_sk_44)[13]: - 1.69921875 frac[13,133] + mem[13,133] >= 0
 accuracy_(1,_sk_44)[14]: - 1.69921875 frac[14,133] + mem[14,133] >= 0
 accuracy_(1,_sk_44)[15]: - 1.69921875 frac[15,133] + mem[15,133] >= 0
 accuracy_(1,_sk_44)[16]: - 1.69921875 frac[16,133] + mem[16,133] >= 0
 accuracy_(1,_sk_44)[17]: - 1.69921875 frac[17,133] + mem[17,133] >= 0
 accuracy_(1,_sk_44)[18]: - 1.69921875 frac[18,133] + mem[18,133] >= 0
 accuracy_(1,_sk_44)[19]: - 1.69921875 frac[19,133] + mem[19,133] >= 0
 accuracy_(1,_sk_44)[20]: - 1.69921875 frac[20,133] + mem[20,133] >= 0
 accuracy_(1,_sk_44)[21]: - 1.69921875 frac[21,133] + mem[21,133] >= 0
 accuracy_(1,_sk_44)[22]: - 1.69921875 frac[22,133] + mem[22,133] >= 0
 accuracy_(2,_sk_44)[0]: - 1.69921875 frac[0,134] + mem[0,134] >= 0
 accuracy_(2,_sk_44)[1]: - 1.69921875 frac[1,134] + mem[1,134] >= 0
 accuracy_(2,_sk_44)[2]: - 1.69921875 frac[2,134] + mem[2,134] >= 0
 accuracy_(2,_sk_44)[3]: - 1.69921875 frac[3,134] + mem[3,134] >= 0
 accuracy_(2,_sk_44)[4]: - 1.69921875 frac[4,134] + mem[4,134] >= 0
 accuracy_(2,_sk_44)[5]: - 1.69921875 frac[5,134] + mem[5,134] >= 0
 accuracy_(2,_sk_44)[6]: - 1.69921875 frac[6,134] + mem[6,134] >= 0
 accuracy_(2,_sk_44)[7]: - 1.69921875 frac[7,134] + mem[7,134] >= 0
 accuracy_(2,_sk_44)[8]: - 1.69921875 frac[8,134] + mem[8,134] >= 0
 accuracy_(2,_sk_44)[9]: - 1.69921875 frac[9,134] + mem[9,134] >= 0
 accuracy_(2,_sk_44)[10]: - 1.69921875 frac[10,134] + mem[10,134] >= 0
 accuracy_(2,_sk_44)[11]: - 1.69921875 frac[11,134] + mem[11,134] >= 0
 accuracy_(2,_sk_44)[12]: - 1.69921875 frac[12,134] + mem[12,134] >= 0
 accuracy_(2,_sk_44)[13]: - 1.69921875 frac[13,134] + mem[13,134] >= 0
 accuracy_(2,_sk_44)[14]: - 1.69921875 frac[14,134] + mem[14,134] >= 0
 accuracy_(2,_sk_44)[15]: - 1.69921875 frac[15,134] + mem[15,134] >= 0
 accuracy_(2,_sk_44)[16]: - 1.69921875 frac[16,134] + mem[16,134] >= 0
 accuracy_(2,_sk_44)[17]: - 1.69921875 frac[17,134] + mem[17,134] >= 0
 accuracy_(2,_sk_44)[18]: - 1.69921875 frac[18,134] + mem[18,134] >= 0
 accuracy_(2,_sk_44)[19]: - 1.69921875 frac[19,134] + mem[19,134] >= 0
 accuracy_(2,_sk_44)[20]: - 1.69921875 frac[20,134] + mem[20,134] >= 0
 accuracy_(2,_sk_44)[21]: - 1.69921875 frac[21,134] + mem[21,134] >= 0
 accuracy_(2,_sk_44)[22]: - 1.69921875 frac[22,134] + mem[22,134] >= 0
 accuracy_(0,_sk_45)[0]: - 1.69921875 frac[0,135] + mem[0,135] >= 0
 accuracy_(0,_sk_45)[1]: - 1.69921875 frac[1,135] + mem[1,135] >= 0
 accuracy_(0,_sk_45)[2]: - 1.69921875 frac[2,135] + mem[2,135] >= 0
 accuracy_(0,_sk_45)[3]: - 1.69921875 frac[3,135] + mem[3,135] >= 0
 accuracy_(0,_sk_45)[4]: - 1.69921875 frac[4,135] + mem[4,135] >= 0
 accuracy_(0,_sk_45)[5]: - 1.69921875 frac[5,135] + mem[5,135] >= 0
 accuracy_(0,_sk_45)[6]: - 1.69921875 frac[6,135] + mem[6,135] >= 0
 accuracy_(0,_sk_45)[7]: - 1.69921875 frac[7,135] + mem[7,135] >= 0
 accuracy_(0,_sk_45)[8]: - 1.69921875 frac[8,135] + mem[8,135] >= 0
 accuracy_(0,_sk_45)[9]: - 1.69921875 frac[9,135] + mem[9,135] >= 0
 accuracy_(0,_sk_45)[10]: - 1.69921875 frac[10,135] + mem[10,135] >= 0
 accuracy_(0,_sk_45)[11]: - 1.69921875 frac[11,135] + mem[11,135] >= 0
 accuracy_(0,_sk_45)[12]: - 1.69921875 frac[12,135] + mem[12,135] >= 0
 accuracy_(0,_sk_45)[13]: - 1.69921875 frac[13,135] + mem[13,135] >= 0
 accuracy_(0,_sk_45)[14]: - 1.69921875 frac[14,135] + mem[14,135] >= 0
 accuracy_(0,_sk_45)[15]: - 1.69921875 frac[15,135] + mem[15,135] >= 0
 accuracy_(0,_sk_45)[16]: - 1.69921875 frac[16,135] + mem[16,135] >= 0
 accuracy_(0,_sk_45)[17]: - 1.69921875 frac[17,135] + mem[17,135] >= 0
 accuracy_(0,_sk_45)[18]: - 1.69921875 frac[18,135] + mem[18,135] >= 0
 accuracy_(0,_sk_45)[19]: - 1.69921875 frac[19,135] + mem[19,135] >= 0
 accuracy_(0,_sk_45)[20]: - 1.69921875 frac[20,135] + mem[20,135] >= 0
 accuracy_(0,_sk_45)[21]: - 1.69921875 frac[21,135] + mem[21,135] >= 0
 accuracy_(0,_sk_45)[22]: - 1.69921875 frac[22,135] + mem[22,135] >= 0
 accuracy_(1,_sk_45)[0]: - 1.69921875 frac[0,136] + mem[0,136] >= 0
 accuracy_(1,_sk_45)[1]: - 1.69921875 frac[1,136] + mem[1,136] >= 0
 accuracy_(1,_sk_45)[2]: - 1.69921875 frac[2,136] + mem[2,136] >= 0
 accuracy_(1,_sk_45)[3]: - 1.69921875 frac[3,136] + mem[3,136] >= 0
 accuracy_(1,_sk_45)[4]: - 1.69921875 frac[4,136] + mem[4,136] >= 0
 accuracy_(1,_sk_45)[5]: - 1.69921875 frac[5,136] + mem[5,136] >= 0
 accuracy_(1,_sk_45)[6]: - 1.69921875 frac[6,136] + mem[6,136] >= 0
 accuracy_(1,_sk_45)[7]: - 1.69921875 frac[7,136] + mem[7,136] >= 0
 accuracy_(1,_sk_45)[8]: - 1.69921875 frac[8,136] + mem[8,136] >= 0
 accuracy_(1,_sk_45)[9]: - 1.69921875 frac[9,136] + mem[9,136] >= 0
 accuracy_(1,_sk_45)[10]: - 1.69921875 frac[10,136] + mem[10,136] >= 0
 accuracy_(1,_sk_45)[11]: - 1.69921875 frac[11,136] + mem[11,136] >= 0
 accuracy_(1,_sk_45)[12]: - 1.69921875 frac[12,136] + mem[12,136] >= 0
 accuracy_(1,_sk_45)[13]: - 1.69921875 frac[13,136] + mem[13,136] >= 0
 accuracy_(1,_sk_45)[14]: - 1.69921875 frac[14,136] + mem[14,136] >= 0
 accuracy_(1,_sk_45)[15]: - 1.69921875 frac[15,136] + mem[15,136] >= 0
 accuracy_(1,_sk_45)[16]: - 1.69921875 frac[16,136] + mem[16,136] >= 0
 accuracy_(1,_sk_45)[17]: - 1.69921875 frac[17,136] + mem[17,136] >= 0
 accuracy_(1,_sk_45)[18]: - 1.69921875 frac[18,136] + mem[18,136] >= 0
 accuracy_(1,_sk_45)[19]: - 1.69921875 frac[19,136] + mem[19,136] >= 0
 accuracy_(1,_sk_45)[20]: - 1.69921875 frac[20,136] + mem[20,136] >= 0
 accuracy_(1,_sk_45)[21]: - 1.69921875 frac[21,136] + mem[21,136] >= 0
 accuracy_(1,_sk_45)[22]: - 1.69921875 frac[22,136] + mem[22,136] >= 0
 accuracy_(2,_sk_45)[0]: - 1.69921875 frac[0,137] + mem[0,137] >= 0
 accuracy_(2,_sk_45)[1]: - 1.69921875 frac[1,137] + mem[1,137] >= 0
 accuracy_(2,_sk_45)[2]: - 1.69921875 frac[2,137] + mem[2,137] >= 0
 accuracy_(2,_sk_45)[3]: - 1.69921875 frac[3,137] + mem[3,137] >= 0
 accuracy_(2,_sk_45)[4]: - 1.69921875 frac[4,137] + mem[4,137] >= 0
 accuracy_(2,_sk_45)[5]: - 1.69921875 frac[5,137] + mem[5,137] >= 0
 accuracy_(2,_sk_45)[6]: - 1.69921875 frac[6,137] + mem[6,137] >= 0
 accuracy_(2,_sk_45)[7]: - 1.69921875 frac[7,137] + mem[7,137] >= 0
 accuracy_(2,_sk_45)[8]: - 1.69921875 frac[8,137] + mem[8,137] >= 0
 accuracy_(2,_sk_45)[9]: - 1.69921875 frac[9,137] + mem[9,137] >= 0
 accuracy_(2,_sk_45)[10]: - 1.69921875 frac[10,137] + mem[10,137] >= 0
 accuracy_(2,_sk_45)[11]: - 1.69921875 frac[11,137] + mem[11,137] >= 0
 accuracy_(2,_sk_45)[12]: - 1.69921875 frac[12,137] + mem[12,137] >= 0
 accuracy_(2,_sk_45)[13]: - 1.69921875 frac[13,137] + mem[13,137] >= 0
 accuracy_(2,_sk_45)[14]: - 1.69921875 frac[14,137] + mem[14,137] >= 0
 accuracy_(2,_sk_45)[15]: - 1.69921875 frac[15,137] + mem[15,137] >= 0
 accuracy_(2,_sk_45)[16]: - 1.69921875 frac[16,137] + mem[16,137] >= 0
 accuracy_(2,_sk_45)[17]: - 1.69921875 frac[17,137] + mem[17,137] >= 0
 accuracy_(2,_sk_45)[18]: - 1.69921875 frac[18,137] + mem[18,137] >= 0
 accuracy_(2,_sk_45)[19]: - 1.69921875 frac[19,137] + mem[19,137] >= 0
 accuracy_(2,_sk_45)[20]: - 1.69921875 frac[20,137] + mem[20,137] >= 0
 accuracy_(2,_sk_45)[21]: - 1.69921875 frac[21,137] + mem[21,137] >= 0
 accuracy_(2,_sk_45)[22]: - 1.69921875 frac[22,137] + mem[22,137] >= 0
 accuracy_(0,_sk_46)[0]: - 1.69921875 frac[0,138] + mem[0,138] >= 0
 accuracy_(0,_sk_46)[1]: - 1.69921875 frac[1,138] + mem[1,138] >= 0
 accuracy_(0,_sk_46)[2]: - 1.69921875 frac[2,138] + mem[2,138] >= 0
 accuracy_(0,_sk_46)[3]: - 1.69921875 frac[3,138] + mem[3,138] >= 0
 accuracy_(0,_sk_46)[4]: - 1.69921875 frac[4,138] + mem[4,138] >= 0
 accuracy_(0,_sk_46)[5]: - 1.69921875 frac[5,138] + mem[5,138] >= 0
 accuracy_(0,_sk_46)[6]: - 1.69921875 frac[6,138] + mem[6,138] >= 0
 accuracy_(0,_sk_46)[7]: - 1.69921875 frac[7,138] + mem[7,138] >= 0
 accuracy_(0,_sk_46)[8]: - 1.69921875 frac[8,138] + mem[8,138] >= 0
 accuracy_(0,_sk_46)[9]: - 1.69921875 frac[9,138] + mem[9,138] >= 0
 accuracy_(0,_sk_46)[10]: - 1.69921875 frac[10,138] + mem[10,138] >= 0
 accuracy_(0,_sk_46)[11]: - 1.69921875 frac[11,138] + mem[11,138] >= 0
 accuracy_(0,_sk_46)[12]: - 1.69921875 frac[12,138] + mem[12,138] >= 0
 accuracy_(0,_sk_46)[13]: - 1.69921875 frac[13,138] + mem[13,138] >= 0
 accuracy_(0,_sk_46)[14]: - 1.69921875 frac[14,138] + mem[14,138] >= 0
 accuracy_(0,_sk_46)[15]: - 1.69921875 frac[15,138] + mem[15,138] >= 0
 accuracy_(0,_sk_46)[16]: - 1.69921875 frac[16,138] + mem[16,138] >= 0
 accuracy_(0,_sk_46)[17]: - 1.69921875 frac[17,138] + mem[17,138] >= 0
 accuracy_(0,_sk_46)[18]: - 1.69921875 frac[18,138] + mem[18,138] >= 0
 accuracy_(0,_sk_46)[19]: - 1.69921875 frac[19,138] + mem[19,138] >= 0
 accuracy_(0,_sk_46)[20]: - 1.69921875 frac[20,138] + mem[20,138] >= 0
 accuracy_(0,_sk_46)[21]: - 1.69921875 frac[21,138] + mem[21,138] >= 0
 accuracy_(0,_sk_46)[22]: - 1.69921875 frac[22,138] + mem[22,138] >= 0
 accuracy_(1,_sk_46)[0]: - 1.69921875 frac[0,139] + mem[0,139] >= 0
 accuracy_(1,_sk_46)[1]: - 1.69921875 frac[1,139] + mem[1,139] >= 0
 accuracy_(1,_sk_46)[2]: - 1.69921875 frac[2,139] + mem[2,139] >= 0
 accuracy_(1,_sk_46)[3]: - 1.69921875 frac[3,139] + mem[3,139] >= 0
 accuracy_(1,_sk_46)[4]: - 1.69921875 frac[4,139] + mem[4,139] >= 0
 accuracy_(1,_sk_46)[5]: - 1.69921875 frac[5,139] + mem[5,139] >= 0
 accuracy_(1,_sk_46)[6]: - 1.69921875 frac[6,139] + mem[6,139] >= 0
 accuracy_(1,_sk_46)[7]: - 1.69921875 frac[7,139] + mem[7,139] >= 0
 accuracy_(1,_sk_46)[8]: - 1.69921875 frac[8,139] + mem[8,139] >= 0
 accuracy_(1,_sk_46)[9]: - 1.69921875 frac[9,139] + mem[9,139] >= 0
 accuracy_(1,_sk_46)[10]: - 1.69921875 frac[10,139] + mem[10,139] >= 0
 accuracy_(1,_sk_46)[11]: - 1.69921875 frac[11,139] + mem[11,139] >= 0
 accuracy_(1,_sk_46)[12]: - 1.69921875 frac[12,139] + mem[12,139] >= 0
 accuracy_(1,_sk_46)[13]: - 1.69921875 frac[13,139] + mem[13,139] >= 0
 accuracy_(1,_sk_46)[14]: - 1.69921875 frac[14,139] + mem[14,139] >= 0
 accuracy_(1,_sk_46)[15]: - 1.69921875 frac[15,139] + mem[15,139] >= 0
 accuracy_(1,_sk_46)[16]: - 1.69921875 frac[16,139] + mem[16,139] >= 0
 accuracy_(1,_sk_46)[17]: - 1.69921875 frac[17,139] + mem[17,139] >= 0
 accuracy_(1,_sk_46)[18]: - 1.69921875 frac[18,139] + mem[18,139] >= 0
 accuracy_(1,_sk_46)[19]: - 1.69921875 frac[19,139] + mem[19,139] >= 0
 accuracy_(1,_sk_46)[20]: - 1.69921875 frac[20,139] + mem[20,139] >= 0
 accuracy_(1,_sk_46)[21]: - 1.69921875 frac[21,139] + mem[21,139] >= 0
 accuracy_(1,_sk_46)[22]: - 1.69921875 frac[22,139] + mem[22,139] >= 0
 accuracy_(2,_sk_46)[0]: - 1.69921875 frac[0,140] + mem[0,140] >= 0
 accuracy_(2,_sk_46)[1]: - 1.69921875 frac[1,140] + mem[1,140] >= 0
 accuracy_(2,_sk_46)[2]: - 1.69921875 frac[2,140] + mem[2,140] >= 0
 accuracy_(2,_sk_46)[3]: - 1.69921875 frac[3,140] + mem[3,140] >= 0
 accuracy_(2,_sk_46)[4]: - 1.69921875 frac[4,140] + mem[4,140] >= 0
 accuracy_(2,_sk_46)[5]: - 1.69921875 frac[5,140] + mem[5,140] >= 0
 accuracy_(2,_sk_46)[6]: - 1.69921875 frac[6,140] + mem[6,140] >= 0
 accuracy_(2,_sk_46)[7]: - 1.69921875 frac[7,140] + mem[7,140] >= 0
 accuracy_(2,_sk_46)[8]: - 1.69921875 frac[8,140] + mem[8,140] >= 0
 accuracy_(2,_sk_46)[9]: - 1.69921875 frac[9,140] + mem[9,140] >= 0
 accuracy_(2,_sk_46)[10]: - 1.69921875 frac[10,140] + mem[10,140] >= 0
 accuracy_(2,_sk_46)[11]: - 1.69921875 frac[11,140] + mem[11,140] >= 0
 accuracy_(2,_sk_46)[12]: - 1.69921875 frac[12,140] + mem[12,140] >= 0
 accuracy_(2,_sk_46)[13]: - 1.69921875 frac[13,140] + mem[13,140] >= 0
 accuracy_(2,_sk_46)[14]: - 1.69921875 frac[14,140] + mem[14,140] >= 0
 accuracy_(2,_sk_46)[15]: - 1.69921875 frac[15,140] + mem[15,140] >= 0
 accuracy_(2,_sk_46)[16]: - 1.69921875 frac[16,140] + mem[16,140] >= 0
 accuracy_(2,_sk_46)[17]: - 1.69921875 frac[17,140] + mem[17,140] >= 0
 accuracy_(2,_sk_46)[18]: - 1.69921875 frac[18,140] + mem[18,140] >= 0
 accuracy_(2,_sk_46)[19]: - 1.69921875 frac[19,140] + mem[19,140] >= 0
 accuracy_(2,_sk_46)[20]: - 1.69921875 frac[20,140] + mem[20,140] >= 0
 accuracy_(2,_sk_46)[21]: - 1.69921875 frac[21,140] + mem[21,140] >= 0
 accuracy_(2,_sk_46)[22]: - 1.69921875 frac[22,140] + mem[22,140] >= 0
 accuracy_(0,_sk_47)[0]: - 1.69921875 frac[0,141] + mem[0,141] >= 0
 accuracy_(0,_sk_47)[1]: - 1.69921875 frac[1,141] + mem[1,141] >= 0
 accuracy_(0,_sk_47)[2]: - 1.69921875 frac[2,141] + mem[2,141] >= 0
 accuracy_(0,_sk_47)[3]: - 1.69921875 frac[3,141] + mem[3,141] >= 0
 accuracy_(0,_sk_47)[4]: - 1.69921875 frac[4,141] + mem[4,141] >= 0
 accuracy_(0,_sk_47)[5]: - 1.69921875 frac[5,141] + mem[5,141] >= 0
 accuracy_(0,_sk_47)[6]: - 1.69921875 frac[6,141] + mem[6,141] >= 0
 accuracy_(0,_sk_47)[7]: - 1.69921875 frac[7,141] + mem[7,141] >= 0
 accuracy_(0,_sk_47)[8]: - 1.69921875 frac[8,141] + mem[8,141] >= 0
 accuracy_(0,_sk_47)[9]: - 1.69921875 frac[9,141] + mem[9,141] >= 0
 accuracy_(0,_sk_47)[10]: - 1.69921875 frac[10,141] + mem[10,141] >= 0
 accuracy_(0,_sk_47)[11]: - 1.69921875 frac[11,141] + mem[11,141] >= 0
 accuracy_(0,_sk_47)[12]: - 1.69921875 frac[12,141] + mem[12,141] >= 0
 accuracy_(0,_sk_47)[13]: - 1.69921875 frac[13,141] + mem[13,141] >= 0
 accuracy_(0,_sk_47)[14]: - 1.69921875 frac[14,141] + mem[14,141] >= 0
 accuracy_(0,_sk_47)[15]: - 1.69921875 frac[15,141] + mem[15,141] >= 0
 accuracy_(0,_sk_47)[16]: - 1.69921875 frac[16,141] + mem[16,141] >= 0
 accuracy_(0,_sk_47)[17]: - 1.69921875 frac[17,141] + mem[17,141] >= 0
 accuracy_(0,_sk_47)[18]: - 1.69921875 frac[18,141] + mem[18,141] >= 0
 accuracy_(0,_sk_47)[19]: - 1.69921875 frac[19,141] + mem[19,141] >= 0
 accuracy_(0,_sk_47)[20]: - 1.69921875 frac[20,141] + mem[20,141] >= 0
 accuracy_(0,_sk_47)[21]: - 1.69921875 frac[21,141] + mem[21,141] >= 0
 accuracy_(0,_sk_47)[22]: - 1.69921875 frac[22,141] + mem[22,141] >= 0
 accuracy_(1,_sk_47)[0]: - 1.69921875 frac[0,142] + mem[0,142] >= 0
 accuracy_(1,_sk_47)[1]: - 1.69921875 frac[1,142] + mem[1,142] >= 0
 accuracy_(1,_sk_47)[2]: - 1.69921875 frac[2,142] + mem[2,142] >= 0
 accuracy_(1,_sk_47)[3]: - 1.69921875 frac[3,142] + mem[3,142] >= 0
 accuracy_(1,_sk_47)[4]: - 1.69921875 frac[4,142] + mem[4,142] >= 0
 accuracy_(1,_sk_47)[5]: - 1.69921875 frac[5,142] + mem[5,142] >= 0
 accuracy_(1,_sk_47)[6]: - 1.69921875 frac[6,142] + mem[6,142] >= 0
 accuracy_(1,_sk_47)[7]: - 1.69921875 frac[7,142] + mem[7,142] >= 0
 accuracy_(1,_sk_47)[8]: - 1.69921875 frac[8,142] + mem[8,142] >= 0
 accuracy_(1,_sk_47)[9]: - 1.69921875 frac[9,142] + mem[9,142] >= 0
 accuracy_(1,_sk_47)[10]: - 1.69921875 frac[10,142] + mem[10,142] >= 0
 accuracy_(1,_sk_47)[11]: - 1.69921875 frac[11,142] + mem[11,142] >= 0
 accuracy_(1,_sk_47)[12]: - 1.69921875 frac[12,142] + mem[12,142] >= 0
 accuracy_(1,_sk_47)[13]: - 1.69921875 frac[13,142] + mem[13,142] >= 0
 accuracy_(1,_sk_47)[14]: - 1.69921875 frac[14,142] + mem[14,142] >= 0
 accuracy_(1,_sk_47)[15]: - 1.69921875 frac[15,142] + mem[15,142] >= 0
 accuracy_(1,_sk_47)[16]: - 1.69921875 frac[16,142] + mem[16,142] >= 0
 accuracy_(1,_sk_47)[17]: - 1.69921875 frac[17,142] + mem[17,142] >= 0
 accuracy_(1,_sk_47)[18]: - 1.69921875 frac[18,142] + mem[18,142] >= 0
 accuracy_(1,_sk_47)[19]: - 1.69921875 frac[19,142] + mem[19,142] >= 0
 accuracy_(1,_sk_47)[20]: - 1.69921875 frac[20,142] + mem[20,142] >= 0
 accuracy_(1,_sk_47)[21]: - 1.69921875 frac[21,142] + mem[21,142] >= 0
 accuracy_(1,_sk_47)[22]: - 1.69921875 frac[22,142] + mem[22,142] >= 0
 accuracy_(2,_sk_47)[0]: - 1.69921875 frac[0,143] + mem[0,143] >= 0
 accuracy_(2,_sk_47)[1]: - 1.69921875 frac[1,143] + mem[1,143] >= 0
 accuracy_(2,_sk_47)[2]: - 1.69921875 frac[2,143] + mem[2,143] >= 0
 accuracy_(2,_sk_47)[3]: - 1.69921875 frac[3,143] + mem[3,143] >= 0
 accuracy_(2,_sk_47)[4]: - 1.69921875 frac[4,143] + mem[4,143] >= 0
 accuracy_(2,_sk_47)[5]: - 1.69921875 frac[5,143] + mem[5,143] >= 0
 accuracy_(2,_sk_47)[6]: - 1.69921875 frac[6,143] + mem[6,143] >= 0
 accuracy_(2,_sk_47)[7]: - 1.69921875 frac[7,143] + mem[7,143] >= 0
 accuracy_(2,_sk_47)[8]: - 1.69921875 frac[8,143] + mem[8,143] >= 0
 accuracy_(2,_sk_47)[9]: - 1.69921875 frac[9,143] + mem[9,143] >= 0
 accuracy_(2,_sk_47)[10]: - 1.69921875 frac[10,143] + mem[10,143] >= 0
 accuracy_(2,_sk_47)[11]: - 1.69921875 frac[11,143] + mem[11,143] >= 0
 accuracy_(2,_sk_47)[12]: - 1.69921875 frac[12,143] + mem[12,143] >= 0
 accuracy_(2,_sk_47)[13]: - 1.69921875 frac[13,143] + mem[13,143] >= 0
 accuracy_(2,_sk_47)[14]: - 1.69921875 frac[14,143] + mem[14,143] >= 0
 accuracy_(2,_sk_47)[15]: - 1.69921875 frac[15,143] + mem[15,143] >= 0
 accuracy_(2,_sk_47)[16]: - 1.69921875 frac[16,143] + mem[16,143] >= 0
 accuracy_(2,_sk_47)[17]: - 1.69921875 frac[17,143] + mem[17,143] >= 0
 accuracy_(2,_sk_47)[18]: - 1.69921875 frac[18,143] + mem[18,143] >= 0
 accuracy_(2,_sk_47)[19]: - 1.69921875 frac[19,143] + mem[19,143] >= 0
 accuracy_(2,_sk_47)[20]: - 1.69921875 frac[20,143] + mem[20,143] >= 0
 accuracy_(2,_sk_47)[21]: - 1.69921875 frac[21,143] + mem[21,143] >= 0
 accuracy_(2,_sk_47)[22]: - 1.69921875 frac[22,143] + mem[22,143] >= 0
 accuracy_(0,_sk_48)[0]: - 1.69921875 frac[0,144] + mem[0,144] >= 0
 accuracy_(0,_sk_48)[1]: - 1.69921875 frac[1,144] + mem[1,144] >= 0
 accuracy_(0,_sk_48)[2]: - 1.69921875 frac[2,144] + mem[2,144] >= 0
 accuracy_(0,_sk_48)[3]: - 1.69921875 frac[3,144] + mem[3,144] >= 0
 accuracy_(0,_sk_48)[4]: - 1.69921875 frac[4,144] + mem[4,144] >= 0
 accuracy_(0,_sk_48)[5]: - 1.69921875 frac[5,144] + mem[5,144] >= 0
 accuracy_(0,_sk_48)[6]: - 1.69921875 frac[6,144] + mem[6,144] >= 0
 accuracy_(0,_sk_48)[7]: - 1.69921875 frac[7,144] + mem[7,144] >= 0
 accuracy_(0,_sk_48)[8]: - 1.69921875 frac[8,144] + mem[8,144] >= 0
 accuracy_(0,_sk_48)[9]: - 1.69921875 frac[9,144] + mem[9,144] >= 0
 accuracy_(0,_sk_48)[10]: - 1.69921875 frac[10,144] + mem[10,144] >= 0
 accuracy_(0,_sk_48)[11]: - 1.69921875 frac[11,144] + mem[11,144] >= 0
 accuracy_(0,_sk_48)[12]: - 1.69921875 frac[12,144] + mem[12,144] >= 0
 accuracy_(0,_sk_48)[13]: - 1.69921875 frac[13,144] + mem[13,144] >= 0
 accuracy_(0,_sk_48)[14]: - 1.69921875 frac[14,144] + mem[14,144] >= 0
 accuracy_(0,_sk_48)[15]: - 1.69921875 frac[15,144] + mem[15,144] >= 0
 accuracy_(0,_sk_48)[16]: - 1.69921875 frac[16,144] + mem[16,144] >= 0
 accuracy_(0,_sk_48)[17]: - 1.69921875 frac[17,144] + mem[17,144] >= 0
 accuracy_(0,_sk_48)[18]: - 1.69921875 frac[18,144] + mem[18,144] >= 0
 accuracy_(0,_sk_48)[19]: - 1.69921875 frac[19,144] + mem[19,144] >= 0
 accuracy_(0,_sk_48)[20]: - 1.69921875 frac[20,144] + mem[20,144] >= 0
 accuracy_(0,_sk_48)[21]: - 1.69921875 frac[21,144] + mem[21,144] >= 0
 accuracy_(0,_sk_48)[22]: - 1.69921875 frac[22,144] + mem[22,144] >= 0
 accuracy_(1,_sk_48)[0]: - 1.69921875 frac[0,145] + mem[0,145] >= 0
 accuracy_(1,_sk_48)[1]: - 1.69921875 frac[1,145] + mem[1,145] >= 0
 accuracy_(1,_sk_48)[2]: - 1.69921875 frac[2,145] + mem[2,145] >= 0
 accuracy_(1,_sk_48)[3]: - 1.69921875 frac[3,145] + mem[3,145] >= 0
 accuracy_(1,_sk_48)[4]: - 1.69921875 frac[4,145] + mem[4,145] >= 0
 accuracy_(1,_sk_48)[5]: - 1.69921875 frac[5,145] + mem[5,145] >= 0
 accuracy_(1,_sk_48)[6]: - 1.69921875 frac[6,145] + mem[6,145] >= 0
 accuracy_(1,_sk_48)[7]: - 1.69921875 frac[7,145] + mem[7,145] >= 0
 accuracy_(1,_sk_48)[8]: - 1.69921875 frac[8,145] + mem[8,145] >= 0
 accuracy_(1,_sk_48)[9]: - 1.69921875 frac[9,145] + mem[9,145] >= 0
 accuracy_(1,_sk_48)[10]: - 1.69921875 frac[10,145] + mem[10,145] >= 0
 accuracy_(1,_sk_48)[11]: - 1.69921875 frac[11,145] + mem[11,145] >= 0
 accuracy_(1,_sk_48)[12]: - 1.69921875 frac[12,145] + mem[12,145] >= 0
 accuracy_(1,_sk_48)[13]: - 1.69921875 frac[13,145] + mem[13,145] >= 0
 accuracy_(1,_sk_48)[14]: - 1.69921875 frac[14,145] + mem[14,145] >= 0
 accuracy_(1,_sk_48)[15]: - 1.69921875 frac[15,145] + mem[15,145] >= 0
 accuracy_(1,_sk_48)[16]: - 1.69921875 frac[16,145] + mem[16,145] >= 0
 accuracy_(1,_sk_48)[17]: - 1.69921875 frac[17,145] + mem[17,145] >= 0
 accuracy_(1,_sk_48)[18]: - 1.69921875 frac[18,145] + mem[18,145] >= 0
 accuracy_(1,_sk_48)[19]: - 1.69921875 frac[19,145] + mem[19,145] >= 0
 accuracy_(1,_sk_48)[20]: - 1.69921875 frac[20,145] + mem[20,145] >= 0
 accuracy_(1,_sk_48)[21]: - 1.69921875 frac[21,145] + mem[21,145] >= 0
 accuracy_(1,_sk_48)[22]: - 1.69921875 frac[22,145] + mem[22,145] >= 0
 accuracy_(2,_sk_48)[0]: - 1.69921875 frac[0,146] + mem[0,146] >= 0
 accuracy_(2,_sk_48)[1]: - 1.69921875 frac[1,146] + mem[1,146] >= 0
 accuracy_(2,_sk_48)[2]: - 1.69921875 frac[2,146] + mem[2,146] >= 0
 accuracy_(2,_sk_48)[3]: - 1.69921875 frac[3,146] + mem[3,146] >= 0
 accuracy_(2,_sk_48)[4]: - 1.69921875 frac[4,146] + mem[4,146] >= 0
 accuracy_(2,_sk_48)[5]: - 1.69921875 frac[5,146] + mem[5,146] >= 0
 accuracy_(2,_sk_48)[6]: - 1.69921875 frac[6,146] + mem[6,146] >= 0
 accuracy_(2,_sk_48)[7]: - 1.69921875 frac[7,146] + mem[7,146] >= 0
 accuracy_(2,_sk_48)[8]: - 1.69921875 frac[8,146] + mem[8,146] >= 0
 accuracy_(2,_sk_48)[9]: - 1.69921875 frac[9,146] + mem[9,146] >= 0
 accuracy_(2,_sk_48)[10]: - 1.69921875 frac[10,146] + mem[10,146] >= 0
 accuracy_(2,_sk_48)[11]: - 1.69921875 frac[11,146] + mem[11,146] >= 0
 accuracy_(2,_sk_48)[12]: - 1.69921875 frac[12,146] + mem[12,146] >= 0
 accuracy_(2,_sk_48)[13]: - 1.69921875 frac[13,146] + mem[13,146] >= 0
 accuracy_(2,_sk_48)[14]: - 1.69921875 frac[14,146] + mem[14,146] >= 0
 accuracy_(2,_sk_48)[15]: - 1.69921875 frac[15,146] + mem[15,146] >= 0
 accuracy_(2,_sk_48)[16]: - 1.69921875 frac[16,146] + mem[16,146] >= 0
 accuracy_(2,_sk_48)[17]: - 1.69921875 frac[17,146] + mem[17,146] >= 0
 accuracy_(2,_sk_48)[18]: - 1.69921875 frac[18,146] + mem[18,146] >= 0
 accuracy_(2,_sk_48)[19]: - 1.69921875 frac[19,146] + mem[19,146] >= 0
 accuracy_(2,_sk_48)[20]: - 1.69921875 frac[20,146] + mem[20,146] >= 0
 accuracy_(2,_sk_48)[21]: - 1.69921875 frac[21,146] + mem[21,146] >= 0
 accuracy_(2,_sk_48)[22]: - 1.69921875 frac[22,146] + mem[22,146] >= 0
 accuracy_(0,_sk_49)[0]: - 1.69921875 frac[0,147] + mem[0,147] >= 0
 accuracy_(0,_sk_49)[1]: - 1.69921875 frac[1,147] + mem[1,147] >= 0
 accuracy_(0,_sk_49)[2]: - 1.69921875 frac[2,147] + mem[2,147] >= 0
 accuracy_(0,_sk_49)[3]: - 1.69921875 frac[3,147] + mem[3,147] >= 0
 accuracy_(0,_sk_49)[4]: - 1.69921875 frac[4,147] + mem[4,147] >= 0
 accuracy_(0,_sk_49)[5]: - 1.69921875 frac[5,147] + mem[5,147] >= 0
 accuracy_(0,_sk_49)[6]: - 1.69921875 frac[6,147] + mem[6,147] >= 0
 accuracy_(0,_sk_49)[7]: - 1.69921875 frac[7,147] + mem[7,147] >= 0
 accuracy_(0,_sk_49)[8]: - 1.69921875 frac[8,147] + mem[8,147] >= 0
 accuracy_(0,_sk_49)[9]: - 1.69921875 frac[9,147] + mem[9,147] >= 0
 accuracy_(0,_sk_49)[10]: - 1.69921875 frac[10,147] + mem[10,147] >= 0
 accuracy_(0,_sk_49)[11]: - 1.69921875 frac[11,147] + mem[11,147] >= 0
 accuracy_(0,_sk_49)[12]: - 1.69921875 frac[12,147] + mem[12,147] >= 0
 accuracy_(0,_sk_49)[13]: - 1.69921875 frac[13,147] + mem[13,147] >= 0
 accuracy_(0,_sk_49)[14]: - 1.69921875 frac[14,147] + mem[14,147] >= 0
 accuracy_(0,_sk_49)[15]: - 1.69921875 frac[15,147] + mem[15,147] >= 0
 accuracy_(0,_sk_49)[16]: - 1.69921875 frac[16,147] + mem[16,147] >= 0
 accuracy_(0,_sk_49)[17]: - 1.69921875 frac[17,147] + mem[17,147] >= 0
 accuracy_(0,_sk_49)[18]: - 1.69921875 frac[18,147] + mem[18,147] >= 0
 accuracy_(0,_sk_49)[19]: - 1.69921875 frac[19,147] + mem[19,147] >= 0
 accuracy_(0,_sk_49)[20]: - 1.69921875 frac[20,147] + mem[20,147] >= 0
 accuracy_(0,_sk_49)[21]: - 1.69921875 frac[21,147] + mem[21,147] >= 0
 accuracy_(0,_sk_49)[22]: - 1.69921875 frac[22,147] + mem[22,147] >= 0
 accuracy_(1,_sk_49)[0]: - 1.69921875 frac[0,148] + mem[0,148] >= 0
 accuracy_(1,_sk_49)[1]: - 1.69921875 frac[1,148] + mem[1,148] >= 0
 accuracy_(1,_sk_49)[2]: - 1.69921875 frac[2,148] + mem[2,148] >= 0
 accuracy_(1,_sk_49)[3]: - 1.69921875 frac[3,148] + mem[3,148] >= 0
 accuracy_(1,_sk_49)[4]: - 1.69921875 frac[4,148] + mem[4,148] >= 0
 accuracy_(1,_sk_49)[5]: - 1.69921875 frac[5,148] + mem[5,148] >= 0
 accuracy_(1,_sk_49)[6]: - 1.69921875 frac[6,148] + mem[6,148] >= 0
 accuracy_(1,_sk_49)[7]: - 1.69921875 frac[7,148] + mem[7,148] >= 0
 accuracy_(1,_sk_49)[8]: - 1.69921875 frac[8,148] + mem[8,148] >= 0
 accuracy_(1,_sk_49)[9]: - 1.69921875 frac[9,148] + mem[9,148] >= 0
 accuracy_(1,_sk_49)[10]: - 1.69921875 frac[10,148] + mem[10,148] >= 0
 accuracy_(1,_sk_49)[11]: - 1.69921875 frac[11,148] + mem[11,148] >= 0
 accuracy_(1,_sk_49)[12]: - 1.69921875 frac[12,148] + mem[12,148] >= 0
 accuracy_(1,_sk_49)[13]: - 1.69921875 frac[13,148] + mem[13,148] >= 0
 accuracy_(1,_sk_49)[14]: - 1.69921875 frac[14,148] + mem[14,148] >= 0
 accuracy_(1,_sk_49)[15]: - 1.69921875 frac[15,148] + mem[15,148] >= 0
 accuracy_(1,_sk_49)[16]: - 1.69921875 frac[16,148] + mem[16,148] >= 0
 accuracy_(1,_sk_49)[17]: - 1.69921875 frac[17,148] + mem[17,148] >= 0
 accuracy_(1,_sk_49)[18]: - 1.69921875 frac[18,148] + mem[18,148] >= 0
 accuracy_(1,_sk_49)[19]: - 1.69921875 frac[19,148] + mem[19,148] >= 0
 accuracy_(1,_sk_49)[20]: - 1.69921875 frac[20,148] + mem[20,148] >= 0
 accuracy_(1,_sk_49)[21]: - 1.69921875 frac[21,148] + mem[21,148] >= 0
 accuracy_(1,_sk_49)[22]: - 1.69921875 frac[22,148] + mem[22,148] >= 0
 accuracy_(2,_sk_49)[0]: - 1.69921875 frac[0,149] + mem[0,149] >= 0
 accuracy_(2,_sk_49)[1]: - 1.69921875 frac[1,149] + mem[1,149] >= 0
 accuracy_(2,_sk_49)[2]: - 1.69921875 frac[2,149] + mem[2,149] >= 0
 accuracy_(2,_sk_49)[3]: - 1.69921875 frac[3,149] + mem[3,149] >= 0
 accuracy_(2,_sk_49)[4]: - 1.69921875 frac[4,149] + mem[4,149] >= 0
 accuracy_(2,_sk_49)[5]: - 1.69921875 frac[5,149] + mem[5,149] >= 0
 accuracy_(2,_sk_49)[6]: - 1.69921875 frac[6,149] + mem[6,149] >= 0
 accuracy_(2,_sk_49)[7]: - 1.69921875 frac[7,149] + mem[7,149] >= 0
 accuracy_(2,_sk_49)[8]: - 1.69921875 frac[8,149] + mem[8,149] >= 0
 accuracy_(2,_sk_49)[9]: - 1.69921875 frac[9,149] + mem[9,149] >= 0
 accuracy_(2,_sk_49)[10]: - 1.69921875 frac[10,149] + mem[10,149] >= 0
 accuracy_(2,_sk_49)[11]: - 1.69921875 frac[11,149] + mem[11,149] >= 0
 accuracy_(2,_sk_49)[12]: - 1.69921875 frac[12,149] + mem[12,149] >= 0
 accuracy_(2,_sk_49)[13]: - 1.69921875 frac[13,149] + mem[13,149] >= 0
 accuracy_(2,_sk_49)[14]: - 1.69921875 frac[14,149] + mem[14,149] >= 0
 accuracy_(2,_sk_49)[15]: - 1.69921875 frac[15,149] + mem[15,149] >= 0
 accuracy_(2,_sk_49)[16]: - 1.69921875 frac[16,149] + mem[16,149] >= 0
 accuracy_(2,_sk_49)[17]: - 1.69921875 frac[17,149] + mem[17,149] >= 0
 accuracy_(2,_sk_49)[18]: - 1.69921875 frac[18,149] + mem[18,149] >= 0
 accuracy_(2,_sk_49)[19]: - 1.69921875 frac[19,149] + mem[19,149] >= 0
 accuracy_(2,_sk_49)[20]: - 1.69921875 frac[20,149] + mem[20,149] >= 0
 accuracy_(2,_sk_49)[21]: - 1.69921875 frac[21,149] + mem[21,149] >= 0
 accuracy_(2,_sk_49)[22]: - 1.69921875 frac[22,149] + mem[22,149] >= 0
 accuracy_(0,_sk_50)[0]: - 1.69921875 frac[0,150] + mem[0,150] >= 0
 accuracy_(0,_sk_50)[1]: - 1.69921875 frac[1,150] + mem[1,150] >= 0
 accuracy_(0,_sk_50)[2]: - 1.69921875 frac[2,150] + mem[2,150] >= 0
 accuracy_(0,_sk_50)[3]: - 1.69921875 frac[3,150] + mem[3,150] >= 0
 accuracy_(0,_sk_50)[4]: - 1.69921875 frac[4,150] + mem[4,150] >= 0
 accuracy_(0,_sk_50)[5]: - 1.69921875 frac[5,150] + mem[5,150] >= 0
 accuracy_(0,_sk_50)[6]: - 1.69921875 frac[6,150] + mem[6,150] >= 0
 accuracy_(0,_sk_50)[7]: - 1.69921875 frac[7,150] + mem[7,150] >= 0
 accuracy_(0,_sk_50)[8]: - 1.69921875 frac[8,150] + mem[8,150] >= 0
 accuracy_(0,_sk_50)[9]: - 1.69921875 frac[9,150] + mem[9,150] >= 0
 accuracy_(0,_sk_50)[10]: - 1.69921875 frac[10,150] + mem[10,150] >= 0
 accuracy_(0,_sk_50)[11]: - 1.69921875 frac[11,150] + mem[11,150] >= 0
 accuracy_(0,_sk_50)[12]: - 1.69921875 frac[12,150] + mem[12,150] >= 0
 accuracy_(0,_sk_50)[13]: - 1.69921875 frac[13,150] + mem[13,150] >= 0
 accuracy_(0,_sk_50)[14]: - 1.69921875 frac[14,150] + mem[14,150] >= 0
 accuracy_(0,_sk_50)[15]: - 1.69921875 frac[15,150] + mem[15,150] >= 0
 accuracy_(0,_sk_50)[16]: - 1.69921875 frac[16,150] + mem[16,150] >= 0
 accuracy_(0,_sk_50)[17]: - 1.69921875 frac[17,150] + mem[17,150] >= 0
 accuracy_(0,_sk_50)[18]: - 1.69921875 frac[18,150] + mem[18,150] >= 0
 accuracy_(0,_sk_50)[19]: - 1.69921875 frac[19,150] + mem[19,150] >= 0
 accuracy_(0,_sk_50)[20]: - 1.69921875 frac[20,150] + mem[20,150] >= 0
 accuracy_(0,_sk_50)[21]: - 1.69921875 frac[21,150] + mem[21,150] >= 0
 accuracy_(0,_sk_50)[22]: - 1.69921875 frac[22,150] + mem[22,150] >= 0
 accuracy_(1,_sk_50)[0]: - 1.69921875 frac[0,151] + mem[0,151] >= 0
 accuracy_(1,_sk_50)[1]: - 1.69921875 frac[1,151] + mem[1,151] >= 0
 accuracy_(1,_sk_50)[2]: - 1.69921875 frac[2,151] + mem[2,151] >= 0
 accuracy_(1,_sk_50)[3]: - 1.69921875 frac[3,151] + mem[3,151] >= 0
 accuracy_(1,_sk_50)[4]: - 1.69921875 frac[4,151] + mem[4,151] >= 0
 accuracy_(1,_sk_50)[5]: - 1.69921875 frac[5,151] + mem[5,151] >= 0
 accuracy_(1,_sk_50)[6]: - 1.69921875 frac[6,151] + mem[6,151] >= 0
 accuracy_(1,_sk_50)[7]: - 1.69921875 frac[7,151] + mem[7,151] >= 0
 accuracy_(1,_sk_50)[8]: - 1.69921875 frac[8,151] + mem[8,151] >= 0
 accuracy_(1,_sk_50)[9]: - 1.69921875 frac[9,151] + mem[9,151] >= 0
 accuracy_(1,_sk_50)[10]: - 1.69921875 frac[10,151] + mem[10,151] >= 0
 accuracy_(1,_sk_50)[11]: - 1.69921875 frac[11,151] + mem[11,151] >= 0
 accuracy_(1,_sk_50)[12]: - 1.69921875 frac[12,151] + mem[12,151] >= 0
 accuracy_(1,_sk_50)[13]: - 1.69921875 frac[13,151] + mem[13,151] >= 0
 accuracy_(1,_sk_50)[14]: - 1.69921875 frac[14,151] + mem[14,151] >= 0
 accuracy_(1,_sk_50)[15]: - 1.69921875 frac[15,151] + mem[15,151] >= 0
 accuracy_(1,_sk_50)[16]: - 1.69921875 frac[16,151] + mem[16,151] >= 0
 accuracy_(1,_sk_50)[17]: - 1.69921875 frac[17,151] + mem[17,151] >= 0
 accuracy_(1,_sk_50)[18]: - 1.69921875 frac[18,151] + mem[18,151] >= 0
 accuracy_(1,_sk_50)[19]: - 1.69921875 frac[19,151] + mem[19,151] >= 0
 accuracy_(1,_sk_50)[20]: - 1.69921875 frac[20,151] + mem[20,151] >= 0
 accuracy_(1,_sk_50)[21]: - 1.69921875 frac[21,151] + mem[21,151] >= 0
 accuracy_(1,_sk_50)[22]: - 1.69921875 frac[22,151] + mem[22,151] >= 0
 accuracy_(2,_sk_50)[0]: - 1.69921875 frac[0,152] + mem[0,152] >= 0
 accuracy_(2,_sk_50)[1]: - 1.69921875 frac[1,152] + mem[1,152] >= 0
 accuracy_(2,_sk_50)[2]: - 1.69921875 frac[2,152] + mem[2,152] >= 0
 accuracy_(2,_sk_50)[3]: - 1.69921875 frac[3,152] + mem[3,152] >= 0
 accuracy_(2,_sk_50)[4]: - 1.69921875 frac[4,152] + mem[4,152] >= 0
 accuracy_(2,_sk_50)[5]: - 1.69921875 frac[5,152] + mem[5,152] >= 0
 accuracy_(2,_sk_50)[6]: - 1.69921875 frac[6,152] + mem[6,152] >= 0
 accuracy_(2,_sk_50)[7]: - 1.69921875 frac[7,152] + mem[7,152] >= 0
 accuracy_(2,_sk_50)[8]: - 1.69921875 frac[8,152] + mem[8,152] >= 0
 accuracy_(2,_sk_50)[9]: - 1.69921875 frac[9,152] + mem[9,152] >= 0
 accuracy_(2,_sk_50)[10]: - 1.69921875 frac[10,152] + mem[10,152] >= 0
 accuracy_(2,_sk_50)[11]: - 1.69921875 frac[11,152] + mem[11,152] >= 0
 accuracy_(2,_sk_50)[12]: - 1.69921875 frac[12,152] + mem[12,152] >= 0
 accuracy_(2,_sk_50)[13]: - 1.69921875 frac[13,152] + mem[13,152] >= 0
 accuracy_(2,_sk_50)[14]: - 1.69921875 frac[14,152] + mem[14,152] >= 0
 accuracy_(2,_sk_50)[15]: - 1.69921875 frac[15,152] + mem[15,152] >= 0
 accuracy_(2,_sk_50)[16]: - 1.69921875 frac[16,152] + mem[16,152] >= 0
 accuracy_(2,_sk_50)[17]: - 1.69921875 frac[17,152] + mem[17,152] >= 0
 accuracy_(2,_sk_50)[18]: - 1.69921875 frac[18,152] + mem[18,152] >= 0
 accuracy_(2,_sk_50)[19]: - 1.69921875 frac[19,152] + mem[19,152] >= 0
 accuracy_(2,_sk_50)[20]: - 1.69921875 frac[20,152] + mem[20,152] >= 0
 accuracy_(2,_sk_50)[21]: - 1.69921875 frac[21,152] + mem[21,152] >= 0
 accuracy_(2,_sk_50)[22]: - 1.69921875 frac[22,152] + mem[22,152] >= 0
 accuracy_(0,_sk_51)[0]: - 1.69921875 frac[0,153] + mem[0,153] >= 0
 accuracy_(0,_sk_51)[1]: - 1.69921875 frac[1,153] + mem[1,153] >= 0
 accuracy_(0,_sk_51)[2]: - 1.69921875 frac[2,153] + mem[2,153] >= 0
 accuracy_(0,_sk_51)[3]: - 1.69921875 frac[3,153] + mem[3,153] >= 0
 accuracy_(0,_sk_51)[4]: - 1.69921875 frac[4,153] + mem[4,153] >= 0
 accuracy_(0,_sk_51)[5]: - 1.69921875 frac[5,153] + mem[5,153] >= 0
 accuracy_(0,_sk_51)[6]: - 1.69921875 frac[6,153] + mem[6,153] >= 0
 accuracy_(0,_sk_51)[7]: - 1.69921875 frac[7,153] + mem[7,153] >= 0
 accuracy_(0,_sk_51)[8]: - 1.69921875 frac[8,153] + mem[8,153] >= 0
 accuracy_(0,_sk_51)[9]: - 1.69921875 frac[9,153] + mem[9,153] >= 0
 accuracy_(0,_sk_51)[10]: - 1.69921875 frac[10,153] + mem[10,153] >= 0
 accuracy_(0,_sk_51)[11]: - 1.69921875 frac[11,153] + mem[11,153] >= 0
 accuracy_(0,_sk_51)[12]: - 1.69921875 frac[12,153] + mem[12,153] >= 0
 accuracy_(0,_sk_51)[13]: - 1.69921875 frac[13,153] + mem[13,153] >= 0
 accuracy_(0,_sk_51)[14]: - 1.69921875 frac[14,153] + mem[14,153] >= 0
 accuracy_(0,_sk_51)[15]: - 1.69921875 frac[15,153] + mem[15,153] >= 0
 accuracy_(0,_sk_51)[16]: - 1.69921875 frac[16,153] + mem[16,153] >= 0
 accuracy_(0,_sk_51)[17]: - 1.69921875 frac[17,153] + mem[17,153] >= 0
 accuracy_(0,_sk_51)[18]: - 1.69921875 frac[18,153] + mem[18,153] >= 0
 accuracy_(0,_sk_51)[19]: - 1.69921875 frac[19,153] + mem[19,153] >= 0
 accuracy_(0,_sk_51)[20]: - 1.69921875 frac[20,153] + mem[20,153] >= 0
 accuracy_(0,_sk_51)[21]: - 1.69921875 frac[21,153] + mem[21,153] >= 0
 accuracy_(0,_sk_51)[22]: - 1.69921875 frac[22,153] + mem[22,153] >= 0
 accuracy_(1,_sk_51)[0]: - 1.69921875 frac[0,154] + mem[0,154] >= 0
 accuracy_(1,_sk_51)[1]: - 1.69921875 frac[1,154] + mem[1,154] >= 0
 accuracy_(1,_sk_51)[2]: - 1.69921875 frac[2,154] + mem[2,154] >= 0
 accuracy_(1,_sk_51)[3]: - 1.69921875 frac[3,154] + mem[3,154] >= 0
 accuracy_(1,_sk_51)[4]: - 1.69921875 frac[4,154] + mem[4,154] >= 0
 accuracy_(1,_sk_51)[5]: - 1.69921875 frac[5,154] + mem[5,154] >= 0
 accuracy_(1,_sk_51)[6]: - 1.69921875 frac[6,154] + mem[6,154] >= 0
 accuracy_(1,_sk_51)[7]: - 1.69921875 frac[7,154] + mem[7,154] >= 0
 accuracy_(1,_sk_51)[8]: - 1.69921875 frac[8,154] + mem[8,154] >= 0
 accuracy_(1,_sk_51)[9]: - 1.69921875 frac[9,154] + mem[9,154] >= 0
 accuracy_(1,_sk_51)[10]: - 1.69921875 frac[10,154] + mem[10,154] >= 0
 accuracy_(1,_sk_51)[11]: - 1.69921875 frac[11,154] + mem[11,154] >= 0
 accuracy_(1,_sk_51)[12]: - 1.69921875 frac[12,154] + mem[12,154] >= 0
 accuracy_(1,_sk_51)[13]: - 1.69921875 frac[13,154] + mem[13,154] >= 0
 accuracy_(1,_sk_51)[14]: - 1.69921875 frac[14,154] + mem[14,154] >= 0
 accuracy_(1,_sk_51)[15]: - 1.69921875 frac[15,154] + mem[15,154] >= 0
 accuracy_(1,_sk_51)[16]: - 1.69921875 frac[16,154] + mem[16,154] >= 0
 accuracy_(1,_sk_51)[17]: - 1.69921875 frac[17,154] + mem[17,154] >= 0
 accuracy_(1,_sk_51)[18]: - 1.69921875 frac[18,154] + mem[18,154] >= 0
 accuracy_(1,_sk_51)[19]: - 1.69921875 frac[19,154] + mem[19,154] >= 0
 accuracy_(1,_sk_51)[20]: - 1.69921875 frac[20,154] + mem[20,154] >= 0
 accuracy_(1,_sk_51)[21]: - 1.69921875 frac[21,154] + mem[21,154] >= 0
 accuracy_(1,_sk_51)[22]: - 1.69921875 frac[22,154] + mem[22,154] >= 0
 accuracy_(2,_sk_51)[0]: - 1.69921875 frac[0,155] + mem[0,155] >= 0
 accuracy_(2,_sk_51)[1]: - 1.69921875 frac[1,155] + mem[1,155] >= 0
 accuracy_(2,_sk_51)[2]: - 1.69921875 frac[2,155] + mem[2,155] >= 0
 accuracy_(2,_sk_51)[3]: - 1.69921875 frac[3,155] + mem[3,155] >= 0
 accuracy_(2,_sk_51)[4]: - 1.69921875 frac[4,155] + mem[4,155] >= 0
 accuracy_(2,_sk_51)[5]: - 1.69921875 frac[5,155] + mem[5,155] >= 0
 accuracy_(2,_sk_51)[6]: - 1.69921875 frac[6,155] + mem[6,155] >= 0
 accuracy_(2,_sk_51)[7]: - 1.69921875 frac[7,155] + mem[7,155] >= 0
 accuracy_(2,_sk_51)[8]: - 1.69921875 frac[8,155] + mem[8,155] >= 0
 accuracy_(2,_sk_51)[9]: - 1.69921875 frac[9,155] + mem[9,155] >= 0
 accuracy_(2,_sk_51)[10]: - 1.69921875 frac[10,155] + mem[10,155] >= 0
 accuracy_(2,_sk_51)[11]: - 1.69921875 frac[11,155] + mem[11,155] >= 0
 accuracy_(2,_sk_51)[12]: - 1.69921875 frac[12,155] + mem[12,155] >= 0
 accuracy_(2,_sk_51)[13]: - 1.69921875 frac[13,155] + mem[13,155] >= 0
 accuracy_(2,_sk_51)[14]: - 1.69921875 frac[14,155] + mem[14,155] >= 0
 accuracy_(2,_sk_51)[15]: - 1.69921875 frac[15,155] + mem[15,155] >= 0
 accuracy_(2,_sk_51)[16]: - 1.69921875 frac[16,155] + mem[16,155] >= 0
 accuracy_(2,_sk_51)[17]: - 1.69921875 frac[17,155] + mem[17,155] >= 0
 accuracy_(2,_sk_51)[18]: - 1.69921875 frac[18,155] + mem[18,155] >= 0
 accuracy_(2,_sk_51)[19]: - 1.69921875 frac[19,155] + mem[19,155] >= 0
 accuracy_(2,_sk_51)[20]: - 1.69921875 frac[20,155] + mem[20,155] >= 0
 accuracy_(2,_sk_51)[21]: - 1.69921875 frac[21,155] + mem[21,155] >= 0
 accuracy_(2,_sk_51)[22]: - 1.69921875 frac[22,155] + mem[22,155] >= 0
 accuracy_(0,_sk_52)[0]: - 1.69921875 frac[0,156] + mem[0,156] >= 0
 accuracy_(0,_sk_52)[1]: - 1.69921875 frac[1,156] + mem[1,156] >= 0
 accuracy_(0,_sk_52)[2]: - 1.69921875 frac[2,156] + mem[2,156] >= 0
 accuracy_(0,_sk_52)[3]: - 1.69921875 frac[3,156] + mem[3,156] >= 0
 accuracy_(0,_sk_52)[4]: - 1.69921875 frac[4,156] + mem[4,156] >= 0
 accuracy_(0,_sk_52)[5]: - 1.69921875 frac[5,156] + mem[5,156] >= 0
 accuracy_(0,_sk_52)[6]: - 1.69921875 frac[6,156] + mem[6,156] >= 0
 accuracy_(0,_sk_52)[7]: - 1.69921875 frac[7,156] + mem[7,156] >= 0
 accuracy_(0,_sk_52)[8]: - 1.69921875 frac[8,156] + mem[8,156] >= 0
 accuracy_(0,_sk_52)[9]: - 1.69921875 frac[9,156] + mem[9,156] >= 0
 accuracy_(0,_sk_52)[10]: - 1.69921875 frac[10,156] + mem[10,156] >= 0
 accuracy_(0,_sk_52)[11]: - 1.69921875 frac[11,156] + mem[11,156] >= 0
 accuracy_(0,_sk_52)[12]: - 1.69921875 frac[12,156] + mem[12,156] >= 0
 accuracy_(0,_sk_52)[13]: - 1.69921875 frac[13,156] + mem[13,156] >= 0
 accuracy_(0,_sk_52)[14]: - 1.69921875 frac[14,156] + mem[14,156] >= 0
 accuracy_(0,_sk_52)[15]: - 1.69921875 frac[15,156] + mem[15,156] >= 0
 accuracy_(0,_sk_52)[16]: - 1.69921875 frac[16,156] + mem[16,156] >= 0
 accuracy_(0,_sk_52)[17]: - 1.69921875 frac[17,156] + mem[17,156] >= 0
 accuracy_(0,_sk_52)[18]: - 1.69921875 frac[18,156] + mem[18,156] >= 0
 accuracy_(0,_sk_52)[19]: - 1.69921875 frac[19,156] + mem[19,156] >= 0
 accuracy_(0,_sk_52)[20]: - 1.69921875 frac[20,156] + mem[20,156] >= 0
 accuracy_(0,_sk_52)[21]: - 1.69921875 frac[21,156] + mem[21,156] >= 0
 accuracy_(0,_sk_52)[22]: - 1.69921875 frac[22,156] + mem[22,156] >= 0
 accuracy_(1,_sk_52)[0]: - 1.69921875 frac[0,157] + mem[0,157] >= 0
 accuracy_(1,_sk_52)[1]: - 1.69921875 frac[1,157] + mem[1,157] >= 0
 accuracy_(1,_sk_52)[2]: - 1.69921875 frac[2,157] + mem[2,157] >= 0
 accuracy_(1,_sk_52)[3]: - 1.69921875 frac[3,157] + mem[3,157] >= 0
 accuracy_(1,_sk_52)[4]: - 1.69921875 frac[4,157] + mem[4,157] >= 0
 accuracy_(1,_sk_52)[5]: - 1.69921875 frac[5,157] + mem[5,157] >= 0
 accuracy_(1,_sk_52)[6]: - 1.69921875 frac[6,157] + mem[6,157] >= 0
 accuracy_(1,_sk_52)[7]: - 1.69921875 frac[7,157] + mem[7,157] >= 0
 accuracy_(1,_sk_52)[8]: - 1.69921875 frac[8,157] + mem[8,157] >= 0
 accuracy_(1,_sk_52)[9]: - 1.69921875 frac[9,157] + mem[9,157] >= 0
 accuracy_(1,_sk_52)[10]: - 1.69921875 frac[10,157] + mem[10,157] >= 0
 accuracy_(1,_sk_52)[11]: - 1.69921875 frac[11,157] + mem[11,157] >= 0
 accuracy_(1,_sk_52)[12]: - 1.69921875 frac[12,157] + mem[12,157] >= 0
 accuracy_(1,_sk_52)[13]: - 1.69921875 frac[13,157] + mem[13,157] >= 0
 accuracy_(1,_sk_52)[14]: - 1.69921875 frac[14,157] + mem[14,157] >= 0
 accuracy_(1,_sk_52)[15]: - 1.69921875 frac[15,157] + mem[15,157] >= 0
 accuracy_(1,_sk_52)[16]: - 1.69921875 frac[16,157] + mem[16,157] >= 0
 accuracy_(1,_sk_52)[17]: - 1.69921875 frac[17,157] + mem[17,157] >= 0
 accuracy_(1,_sk_52)[18]: - 1.69921875 frac[18,157] + mem[18,157] >= 0
 accuracy_(1,_sk_52)[19]: - 1.69921875 frac[19,157] + mem[19,157] >= 0
 accuracy_(1,_sk_52)[20]: - 1.69921875 frac[20,157] + mem[20,157] >= 0
 accuracy_(1,_sk_52)[21]: - 1.69921875 frac[21,157] + mem[21,157] >= 0
 accuracy_(1,_sk_52)[22]: - 1.69921875 frac[22,157] + mem[22,157] >= 0
 accuracy_(2,_sk_52)[0]: - 1.69921875 frac[0,158] + mem[0,158] >= 0
 accuracy_(2,_sk_52)[1]: - 1.69921875 frac[1,158] + mem[1,158] >= 0
 accuracy_(2,_sk_52)[2]: - 1.69921875 frac[2,158] + mem[2,158] >= 0
 accuracy_(2,_sk_52)[3]: - 1.69921875 frac[3,158] + mem[3,158] >= 0
 accuracy_(2,_sk_52)[4]: - 1.69921875 frac[4,158] + mem[4,158] >= 0
 accuracy_(2,_sk_52)[5]: - 1.69921875 frac[5,158] + mem[5,158] >= 0
 accuracy_(2,_sk_52)[6]: - 1.69921875 frac[6,158] + mem[6,158] >= 0
 accuracy_(2,_sk_52)[7]: - 1.69921875 frac[7,158] + mem[7,158] >= 0
 accuracy_(2,_sk_52)[8]: - 1.69921875 frac[8,158] + mem[8,158] >= 0
 accuracy_(2,_sk_52)[9]: - 1.69921875 frac[9,158] + mem[9,158] >= 0
 accuracy_(2,_sk_52)[10]: - 1.69921875 frac[10,158] + mem[10,158] >= 0
 accuracy_(2,_sk_52)[11]: - 1.69921875 frac[11,158] + mem[11,158] >= 0
 accuracy_(2,_sk_52)[12]: - 1.69921875 frac[12,158] + mem[12,158] >= 0
 accuracy_(2,_sk_52)[13]: - 1.69921875 frac[13,158] + mem[13,158] >= 0
 accuracy_(2,_sk_52)[14]: - 1.69921875 frac[14,158] + mem[14,158] >= 0
 accuracy_(2,_sk_52)[15]: - 1.69921875 frac[15,158] + mem[15,158] >= 0
 accuracy_(2,_sk_52)[16]: - 1.69921875 frac[16,158] + mem[16,158] >= 0
 accuracy_(2,_sk_52)[17]: - 1.69921875 frac[17,158] + mem[17,158] >= 0
 accuracy_(2,_sk_52)[18]: - 1.69921875 frac[18,158] + mem[18,158] >= 0
 accuracy_(2,_sk_52)[19]: - 1.69921875 frac[19,158] + mem[19,158] >= 0
 accuracy_(2,_sk_52)[20]: - 1.69921875 frac[20,158] + mem[20,158] >= 0
 accuracy_(2,_sk_52)[21]: - 1.69921875 frac[21,158] + mem[21,158] >= 0
 accuracy_(2,_sk_52)[22]: - 1.69921875 frac[22,158] + mem[22,158] >= 0
 accuracy_(0,_sk_53)[0]: - 1.69921875 frac[0,159] + mem[0,159] >= 0
 accuracy_(0,_sk_53)[1]: - 1.69921875 frac[1,159] + mem[1,159] >= 0
 accuracy_(0,_sk_53)[2]: - 1.69921875 frac[2,159] + mem[2,159] >= 0
 accuracy_(0,_sk_53)[3]: - 1.69921875 frac[3,159] + mem[3,159] >= 0
 accuracy_(0,_sk_53)[4]: - 1.69921875 frac[4,159] + mem[4,159] >= 0
 accuracy_(0,_sk_53)[5]: - 1.69921875 frac[5,159] + mem[5,159] >= 0
 accuracy_(0,_sk_53)[6]: - 1.69921875 frac[6,159] + mem[6,159] >= 0
 accuracy_(0,_sk_53)[7]: - 1.69921875 frac[7,159] + mem[7,159] >= 0
 accuracy_(0,_sk_53)[8]: - 1.69921875 frac[8,159] + mem[8,159] >= 0
 accuracy_(0,_sk_53)[9]: - 1.69921875 frac[9,159] + mem[9,159] >= 0
 accuracy_(0,_sk_53)[10]: - 1.69921875 frac[10,159] + mem[10,159] >= 0
 accuracy_(0,_sk_53)[11]: - 1.69921875 frac[11,159] + mem[11,159] >= 0
 accuracy_(0,_sk_53)[12]: - 1.69921875 frac[12,159] + mem[12,159] >= 0
 accuracy_(0,_sk_53)[13]: - 1.69921875 frac[13,159] + mem[13,159] >= 0
 accuracy_(0,_sk_53)[14]: - 1.69921875 frac[14,159] + mem[14,159] >= 0
 accuracy_(0,_sk_53)[15]: - 1.69921875 frac[15,159] + mem[15,159] >= 0
 accuracy_(0,_sk_53)[16]: - 1.69921875 frac[16,159] + mem[16,159] >= 0
 accuracy_(0,_sk_53)[17]: - 1.69921875 frac[17,159] + mem[17,159] >= 0
 accuracy_(0,_sk_53)[18]: - 1.69921875 frac[18,159] + mem[18,159] >= 0
 accuracy_(0,_sk_53)[19]: - 1.69921875 frac[19,159] + mem[19,159] >= 0
 accuracy_(0,_sk_53)[20]: - 1.69921875 frac[20,159] + mem[20,159] >= 0
 accuracy_(0,_sk_53)[21]: - 1.69921875 frac[21,159] + mem[21,159] >= 0
 accuracy_(0,_sk_53)[22]: - 1.69921875 frac[22,159] + mem[22,159] >= 0
 accuracy_(1,_sk_53)[0]: - 1.69921875 frac[0,160] + mem[0,160] >= 0
 accuracy_(1,_sk_53)[1]: - 1.69921875 frac[1,160] + mem[1,160] >= 0
 accuracy_(1,_sk_53)[2]: - 1.69921875 frac[2,160] + mem[2,160] >= 0
 accuracy_(1,_sk_53)[3]: - 1.69921875 frac[3,160] + mem[3,160] >= 0
 accuracy_(1,_sk_53)[4]: - 1.69921875 frac[4,160] + mem[4,160] >= 0
 accuracy_(1,_sk_53)[5]: - 1.69921875 frac[5,160] + mem[5,160] >= 0
 accuracy_(1,_sk_53)[6]: - 1.69921875 frac[6,160] + mem[6,160] >= 0
 accuracy_(1,_sk_53)[7]: - 1.69921875 frac[7,160] + mem[7,160] >= 0
 accuracy_(1,_sk_53)[8]: - 1.69921875 frac[8,160] + mem[8,160] >= 0
 accuracy_(1,_sk_53)[9]: - 1.69921875 frac[9,160] + mem[9,160] >= 0
 accuracy_(1,_sk_53)[10]: - 1.69921875 frac[10,160] + mem[10,160] >= 0
 accuracy_(1,_sk_53)[11]: - 1.69921875 frac[11,160] + mem[11,160] >= 0
 accuracy_(1,_sk_53)[12]: - 1.69921875 frac[12,160] + mem[12,160] >= 0
 accuracy_(1,_sk_53)[13]: - 1.69921875 frac[13,160] + mem[13,160] >= 0
 accuracy_(1,_sk_53)[14]: - 1.69921875 frac[14,160] + mem[14,160] >= 0
 accuracy_(1,_sk_53)[15]: - 1.69921875 frac[15,160] + mem[15,160] >= 0
 accuracy_(1,_sk_53)[16]: - 1.69921875 frac[16,160] + mem[16,160] >= 0
 accuracy_(1,_sk_53)[17]: - 1.69921875 frac[17,160] + mem[17,160] >= 0
 accuracy_(1,_sk_53)[18]: - 1.69921875 frac[18,160] + mem[18,160] >= 0
 accuracy_(1,_sk_53)[19]: - 1.69921875 frac[19,160] + mem[19,160] >= 0
 accuracy_(1,_sk_53)[20]: - 1.69921875 frac[20,160] + mem[20,160] >= 0
 accuracy_(1,_sk_53)[21]: - 1.69921875 frac[21,160] + mem[21,160] >= 0
 accuracy_(1,_sk_53)[22]: - 1.69921875 frac[22,160] + mem[22,160] >= 0
 accuracy_(2,_sk_53)[0]: - 1.69921875 frac[0,161] + mem[0,161] >= 0
 accuracy_(2,_sk_53)[1]: - 1.69921875 frac[1,161] + mem[1,161] >= 0
 accuracy_(2,_sk_53)[2]: - 1.69921875 frac[2,161] + mem[2,161] >= 0
 accuracy_(2,_sk_53)[3]: - 1.69921875 frac[3,161] + mem[3,161] >= 0
 accuracy_(2,_sk_53)[4]: - 1.69921875 frac[4,161] + mem[4,161] >= 0
 accuracy_(2,_sk_53)[5]: - 1.69921875 frac[5,161] + mem[5,161] >= 0
 accuracy_(2,_sk_53)[6]: - 1.69921875 frac[6,161] + mem[6,161] >= 0
 accuracy_(2,_sk_53)[7]: - 1.69921875 frac[7,161] + mem[7,161] >= 0
 accuracy_(2,_sk_53)[8]: - 1.69921875 frac[8,161] + mem[8,161] >= 0
 accuracy_(2,_sk_53)[9]: - 1.69921875 frac[9,161] + mem[9,161] >= 0
 accuracy_(2,_sk_53)[10]: - 1.69921875 frac[10,161] + mem[10,161] >= 0
 accuracy_(2,_sk_53)[11]: - 1.69921875 frac[11,161] + mem[11,161] >= 0
 accuracy_(2,_sk_53)[12]: - 1.69921875 frac[12,161] + mem[12,161] >= 0
 accuracy_(2,_sk_53)[13]: - 1.69921875 frac[13,161] + mem[13,161] >= 0
 accuracy_(2,_sk_53)[14]: - 1.69921875 frac[14,161] + mem[14,161] >= 0
 accuracy_(2,_sk_53)[15]: - 1.69921875 frac[15,161] + mem[15,161] >= 0
 accuracy_(2,_sk_53)[16]: - 1.69921875 frac[16,161] + mem[16,161] >= 0
 accuracy_(2,_sk_53)[17]: - 1.69921875 frac[17,161] + mem[17,161] >= 0
 accuracy_(2,_sk_53)[18]: - 1.69921875 frac[18,161] + mem[18,161] >= 0
 accuracy_(2,_sk_53)[19]: - 1.69921875 frac[19,161] + mem[19,161] >= 0
 accuracy_(2,_sk_53)[20]: - 1.69921875 frac[20,161] + mem[20,161] >= 0
 accuracy_(2,_sk_53)[21]: - 1.69921875 frac[21,161] + mem[21,161] >= 0
 accuracy_(2,_sk_53)[22]: - 1.69921875 frac[22,161] + mem[22,161] >= 0
 accuracy_(0,_sk_54)[0]: - 1.69921875 frac[0,162] + mem[0,162] >= 0
 accuracy_(0,_sk_54)[1]: - 1.69921875 frac[1,162] + mem[1,162] >= 0
 accuracy_(0,_sk_54)[2]: - 1.69921875 frac[2,162] + mem[2,162] >= 0
 accuracy_(0,_sk_54)[3]: - 1.69921875 frac[3,162] + mem[3,162] >= 0
 accuracy_(0,_sk_54)[4]: - 1.69921875 frac[4,162] + mem[4,162] >= 0
 accuracy_(0,_sk_54)[5]: - 1.69921875 frac[5,162] + mem[5,162] >= 0
 accuracy_(0,_sk_54)[6]: - 1.69921875 frac[6,162] + mem[6,162] >= 0
 accuracy_(0,_sk_54)[7]: - 1.69921875 frac[7,162] + mem[7,162] >= 0
 accuracy_(0,_sk_54)[8]: - 1.69921875 frac[8,162] + mem[8,162] >= 0
 accuracy_(0,_sk_54)[9]: - 1.69921875 frac[9,162] + mem[9,162] >= 0
 accuracy_(0,_sk_54)[10]: - 1.69921875 frac[10,162] + mem[10,162] >= 0
 accuracy_(0,_sk_54)[11]: - 1.69921875 frac[11,162] + mem[11,162] >= 0
 accuracy_(0,_sk_54)[12]: - 1.69921875 frac[12,162] + mem[12,162] >= 0
 accuracy_(0,_sk_54)[13]: - 1.69921875 frac[13,162] + mem[13,162] >= 0
 accuracy_(0,_sk_54)[14]: - 1.69921875 frac[14,162] + mem[14,162] >= 0
 accuracy_(0,_sk_54)[15]: - 1.69921875 frac[15,162] + mem[15,162] >= 0
 accuracy_(0,_sk_54)[16]: - 1.69921875 frac[16,162] + mem[16,162] >= 0
 accuracy_(0,_sk_54)[17]: - 1.69921875 frac[17,162] + mem[17,162] >= 0
 accuracy_(0,_sk_54)[18]: - 1.69921875 frac[18,162] + mem[18,162] >= 0
 accuracy_(0,_sk_54)[19]: - 1.69921875 frac[19,162] + mem[19,162] >= 0
 accuracy_(0,_sk_54)[20]: - 1.69921875 frac[20,162] + mem[20,162] >= 0
 accuracy_(0,_sk_54)[21]: - 1.69921875 frac[21,162] + mem[21,162] >= 0
 accuracy_(0,_sk_54)[22]: - 1.69921875 frac[22,162] + mem[22,162] >= 0
 accuracy_(1,_sk_54)[0]: - 1.69921875 frac[0,163] + mem[0,163] >= 0
 accuracy_(1,_sk_54)[1]: - 1.69921875 frac[1,163] + mem[1,163] >= 0
 accuracy_(1,_sk_54)[2]: - 1.69921875 frac[2,163] + mem[2,163] >= 0
 accuracy_(1,_sk_54)[3]: - 1.69921875 frac[3,163] + mem[3,163] >= 0
 accuracy_(1,_sk_54)[4]: - 1.69921875 frac[4,163] + mem[4,163] >= 0
 accuracy_(1,_sk_54)[5]: - 1.69921875 frac[5,163] + mem[5,163] >= 0
 accuracy_(1,_sk_54)[6]: - 1.69921875 frac[6,163] + mem[6,163] >= 0
 accuracy_(1,_sk_54)[7]: - 1.69921875 frac[7,163] + mem[7,163] >= 0
 accuracy_(1,_sk_54)[8]: - 1.69921875 frac[8,163] + mem[8,163] >= 0
 accuracy_(1,_sk_54)[9]: - 1.69921875 frac[9,163] + mem[9,163] >= 0
 accuracy_(1,_sk_54)[10]: - 1.69921875 frac[10,163] + mem[10,163] >= 0
 accuracy_(1,_sk_54)[11]: - 1.69921875 frac[11,163] + mem[11,163] >= 0
 accuracy_(1,_sk_54)[12]: - 1.69921875 frac[12,163] + mem[12,163] >= 0
 accuracy_(1,_sk_54)[13]: - 1.69921875 frac[13,163] + mem[13,163] >= 0
 accuracy_(1,_sk_54)[14]: - 1.69921875 frac[14,163] + mem[14,163] >= 0
 accuracy_(1,_sk_54)[15]: - 1.69921875 frac[15,163] + mem[15,163] >= 0
 accuracy_(1,_sk_54)[16]: - 1.69921875 frac[16,163] + mem[16,163] >= 0
 accuracy_(1,_sk_54)[17]: - 1.69921875 frac[17,163] + mem[17,163] >= 0
 accuracy_(1,_sk_54)[18]: - 1.69921875 frac[18,163] + mem[18,163] >= 0
 accuracy_(1,_sk_54)[19]: - 1.69921875 frac[19,163] + mem[19,163] >= 0
 accuracy_(1,_sk_54)[20]: - 1.69921875 frac[20,163] + mem[20,163] >= 0
 accuracy_(1,_sk_54)[21]: - 1.69921875 frac[21,163] + mem[21,163] >= 0
 accuracy_(1,_sk_54)[22]: - 1.69921875 frac[22,163] + mem[22,163] >= 0
 accuracy_(2,_sk_54)[0]: - 1.69921875 frac[0,164] + mem[0,164] >= 0
 accuracy_(2,_sk_54)[1]: - 1.69921875 frac[1,164] + mem[1,164] >= 0
 accuracy_(2,_sk_54)[2]: - 1.69921875 frac[2,164] + mem[2,164] >= 0
 accuracy_(2,_sk_54)[3]: - 1.69921875 frac[3,164] + mem[3,164] >= 0
 accuracy_(2,_sk_54)[4]: - 1.69921875 frac[4,164] + mem[4,164] >= 0
 accuracy_(2,_sk_54)[5]: - 1.69921875 frac[5,164] + mem[5,164] >= 0
 accuracy_(2,_sk_54)[6]: - 1.69921875 frac[6,164] + mem[6,164] >= 0
 accuracy_(2,_sk_54)[7]: - 1.69921875 frac[7,164] + mem[7,164] >= 0
 accuracy_(2,_sk_54)[8]: - 1.69921875 frac[8,164] + mem[8,164] >= 0
 accuracy_(2,_sk_54)[9]: - 1.69921875 frac[9,164] + mem[9,164] >= 0
 accuracy_(2,_sk_54)[10]: - 1.69921875 frac[10,164] + mem[10,164] >= 0
 accuracy_(2,_sk_54)[11]: - 1.69921875 frac[11,164] + mem[11,164] >= 0
 accuracy_(2,_sk_54)[12]: - 1.69921875 frac[12,164] + mem[12,164] >= 0
 accuracy_(2,_sk_54)[13]: - 1.69921875 frac[13,164] + mem[13,164] >= 0
 accuracy_(2,_sk_54)[14]: - 1.69921875 frac[14,164] + mem[14,164] >= 0
 accuracy_(2,_sk_54)[15]: - 1.69921875 frac[15,164] + mem[15,164] >= 0
 accuracy_(2,_sk_54)[16]: - 1.69921875 frac[16,164] + mem[16,164] >= 0
 accuracy_(2,_sk_54)[17]: - 1.69921875 frac[17,164] + mem[17,164] >= 0
 accuracy_(2,_sk_54)[18]: - 1.69921875 frac[18,164] + mem[18,164] >= 0
 accuracy_(2,_sk_54)[19]: - 1.69921875 frac[19,164] + mem[19,164] >= 0
 accuracy_(2,_sk_54)[20]: - 1.69921875 frac[20,164] + mem[20,164] >= 0
 accuracy_(2,_sk_54)[21]: - 1.69921875 frac[21,164] + mem[21,164] >= 0
 accuracy_(2,_sk_54)[22]: - 1.69921875 frac[22,164] + mem[22,164] >= 0
 accuracy_(0,_sk_55)[0]: - 1.69921875 frac[0,165] + mem[0,165] >= 0
 accuracy_(0,_sk_55)[1]: - 1.69921875 frac[1,165] + mem[1,165] >= 0
 accuracy_(0,_sk_55)[2]: - 1.69921875 frac[2,165] + mem[2,165] >= 0
 accuracy_(0,_sk_55)[3]: - 1.69921875 frac[3,165] + mem[3,165] >= 0
 accuracy_(0,_sk_55)[4]: - 1.69921875 frac[4,165] + mem[4,165] >= 0
 accuracy_(0,_sk_55)[5]: - 1.69921875 frac[5,165] + mem[5,165] >= 0
 accuracy_(0,_sk_55)[6]: - 1.69921875 frac[6,165] + mem[6,165] >= 0
 accuracy_(0,_sk_55)[7]: - 1.69921875 frac[7,165] + mem[7,165] >= 0
 accuracy_(0,_sk_55)[8]: - 1.69921875 frac[8,165] + mem[8,165] >= 0
 accuracy_(0,_sk_55)[9]: - 1.69921875 frac[9,165] + mem[9,165] >= 0
 accuracy_(0,_sk_55)[10]: - 1.69921875 frac[10,165] + mem[10,165] >= 0
 accuracy_(0,_sk_55)[11]: - 1.69921875 frac[11,165] + mem[11,165] >= 0
 accuracy_(0,_sk_55)[12]: - 1.69921875 frac[12,165] + mem[12,165] >= 0
 accuracy_(0,_sk_55)[13]: - 1.69921875 frac[13,165] + mem[13,165] >= 0
 accuracy_(0,_sk_55)[14]: - 1.69921875 frac[14,165] + mem[14,165] >= 0
 accuracy_(0,_sk_55)[15]: - 1.69921875 frac[15,165] + mem[15,165] >= 0
 accuracy_(0,_sk_55)[16]: - 1.69921875 frac[16,165] + mem[16,165] >= 0
 accuracy_(0,_sk_55)[17]: - 1.69921875 frac[17,165] + mem[17,165] >= 0
 accuracy_(0,_sk_55)[18]: - 1.69921875 frac[18,165] + mem[18,165] >= 0
 accuracy_(0,_sk_55)[19]: - 1.69921875 frac[19,165] + mem[19,165] >= 0
 accuracy_(0,_sk_55)[20]: - 1.69921875 frac[20,165] + mem[20,165] >= 0
 accuracy_(0,_sk_55)[21]: - 1.69921875 frac[21,165] + mem[21,165] >= 0
 accuracy_(0,_sk_55)[22]: - 1.69921875 frac[22,165] + mem[22,165] >= 0
 accuracy_(1,_sk_55)[0]: - 1.69921875 frac[0,166] + mem[0,166] >= 0
 accuracy_(1,_sk_55)[1]: - 1.69921875 frac[1,166] + mem[1,166] >= 0
 accuracy_(1,_sk_55)[2]: - 1.69921875 frac[2,166] + mem[2,166] >= 0
 accuracy_(1,_sk_55)[3]: - 1.69921875 frac[3,166] + mem[3,166] >= 0
 accuracy_(1,_sk_55)[4]: - 1.69921875 frac[4,166] + mem[4,166] >= 0
 accuracy_(1,_sk_55)[5]: - 1.69921875 frac[5,166] + mem[5,166] >= 0
 accuracy_(1,_sk_55)[6]: - 1.69921875 frac[6,166] + mem[6,166] >= 0
 accuracy_(1,_sk_55)[7]: - 1.69921875 frac[7,166] + mem[7,166] >= 0
 accuracy_(1,_sk_55)[8]: - 1.69921875 frac[8,166] + mem[8,166] >= 0
 accuracy_(1,_sk_55)[9]: - 1.69921875 frac[9,166] + mem[9,166] >= 0
 accuracy_(1,_sk_55)[10]: - 1.69921875 frac[10,166] + mem[10,166] >= 0
 accuracy_(1,_sk_55)[11]: - 1.69921875 frac[11,166] + mem[11,166] >= 0
 accuracy_(1,_sk_55)[12]: - 1.69921875 frac[12,166] + mem[12,166] >= 0
 accuracy_(1,_sk_55)[13]: - 1.69921875 frac[13,166] + mem[13,166] >= 0
 accuracy_(1,_sk_55)[14]: - 1.69921875 frac[14,166] + mem[14,166] >= 0
 accuracy_(1,_sk_55)[15]: - 1.69921875 frac[15,166] + mem[15,166] >= 0
 accuracy_(1,_sk_55)[16]: - 1.69921875 frac[16,166] + mem[16,166] >= 0
 accuracy_(1,_sk_55)[17]: - 1.69921875 frac[17,166] + mem[17,166] >= 0
 accuracy_(1,_sk_55)[18]: - 1.69921875 frac[18,166] + mem[18,166] >= 0
 accuracy_(1,_sk_55)[19]: - 1.69921875 frac[19,166] + mem[19,166] >= 0
 accuracy_(1,_sk_55)[20]: - 1.69921875 frac[20,166] + mem[20,166] >= 0
 accuracy_(1,_sk_55)[21]: - 1.69921875 frac[21,166] + mem[21,166] >= 0
 accuracy_(1,_sk_55)[22]: - 1.69921875 frac[22,166] + mem[22,166] >= 0
 accuracy_(2,_sk_55)[0]: - 1.69921875 frac[0,167] + mem[0,167] >= 0
 accuracy_(2,_sk_55)[1]: - 1.69921875 frac[1,167] + mem[1,167] >= 0
 accuracy_(2,_sk_55)[2]: - 1.69921875 frac[2,167] + mem[2,167] >= 0
 accuracy_(2,_sk_55)[3]: - 1.69921875 frac[3,167] + mem[3,167] >= 0
 accuracy_(2,_sk_55)[4]: - 1.69921875 frac[4,167] + mem[4,167] >= 0
 accuracy_(2,_sk_55)[5]: - 1.69921875 frac[5,167] + mem[5,167] >= 0
 accuracy_(2,_sk_55)[6]: - 1.69921875 frac[6,167] + mem[6,167] >= 0
 accuracy_(2,_sk_55)[7]: - 1.69921875 frac[7,167] + mem[7,167] >= 0
 accuracy_(2,_sk_55)[8]: - 1.69921875 frac[8,167] + mem[8,167] >= 0
 accuracy_(2,_sk_55)[9]: - 1.69921875 frac[9,167] + mem[9,167] >= 0
 accuracy_(2,_sk_55)[10]: - 1.69921875 frac[10,167] + mem[10,167] >= 0
 accuracy_(2,_sk_55)[11]: - 1.69921875 frac[11,167] + mem[11,167] >= 0
 accuracy_(2,_sk_55)[12]: - 1.69921875 frac[12,167] + mem[12,167] >= 0
 accuracy_(2,_sk_55)[13]: - 1.69921875 frac[13,167] + mem[13,167] >= 0
 accuracy_(2,_sk_55)[14]: - 1.69921875 frac[14,167] + mem[14,167] >= 0
 accuracy_(2,_sk_55)[15]: - 1.69921875 frac[15,167] + mem[15,167] >= 0
 accuracy_(2,_sk_55)[16]: - 1.69921875 frac[16,167] + mem[16,167] >= 0
 accuracy_(2,_sk_55)[17]: - 1.69921875 frac[17,167] + mem[17,167] >= 0
 accuracy_(2,_sk_55)[18]: - 1.69921875 frac[18,167] + mem[18,167] >= 0
 accuracy_(2,_sk_55)[19]: - 1.69921875 frac[19,167] + mem[19,167] >= 0
 accuracy_(2,_sk_55)[20]: - 1.69921875 frac[20,167] + mem[20,167] >= 0
 accuracy_(2,_sk_55)[21]: - 1.69921875 frac[21,167] + mem[21,167] >= 0
 accuracy_(2,_sk_55)[22]: - 1.69921875 frac[22,167] + mem[22,167] >= 0
 accuracy_(0,_sk_56)[0]: - 1.69921875 frac[0,168] + mem[0,168] >= 0
 accuracy_(0,_sk_56)[1]: - 1.69921875 frac[1,168] + mem[1,168] >= 0
 accuracy_(0,_sk_56)[2]: - 1.69921875 frac[2,168] + mem[2,168] >= 0
 accuracy_(0,_sk_56)[3]: - 1.69921875 frac[3,168] + mem[3,168] >= 0
 accuracy_(0,_sk_56)[4]: - 1.69921875 frac[4,168] + mem[4,168] >= 0
 accuracy_(0,_sk_56)[5]: - 1.69921875 frac[5,168] + mem[5,168] >= 0
 accuracy_(0,_sk_56)[6]: - 1.69921875 frac[6,168] + mem[6,168] >= 0
 accuracy_(0,_sk_56)[7]: - 1.69921875 frac[7,168] + mem[7,168] >= 0
 accuracy_(0,_sk_56)[8]: - 1.69921875 frac[8,168] + mem[8,168] >= 0
 accuracy_(0,_sk_56)[9]: - 1.69921875 frac[9,168] + mem[9,168] >= 0
 accuracy_(0,_sk_56)[10]: - 1.69921875 frac[10,168] + mem[10,168] >= 0
 accuracy_(0,_sk_56)[11]: - 1.69921875 frac[11,168] + mem[11,168] >= 0
 accuracy_(0,_sk_56)[12]: - 1.69921875 frac[12,168] + mem[12,168] >= 0
 accuracy_(0,_sk_56)[13]: - 1.69921875 frac[13,168] + mem[13,168] >= 0
 accuracy_(0,_sk_56)[14]: - 1.69921875 frac[14,168] + mem[14,168] >= 0
 accuracy_(0,_sk_56)[15]: - 1.69921875 frac[15,168] + mem[15,168] >= 0
 accuracy_(0,_sk_56)[16]: - 1.69921875 frac[16,168] + mem[16,168] >= 0
 accuracy_(0,_sk_56)[17]: - 1.69921875 frac[17,168] + mem[17,168] >= 0
 accuracy_(0,_sk_56)[18]: - 1.69921875 frac[18,168] + mem[18,168] >= 0
 accuracy_(0,_sk_56)[19]: - 1.69921875 frac[19,168] + mem[19,168] >= 0
 accuracy_(0,_sk_56)[20]: - 1.69921875 frac[20,168] + mem[20,168] >= 0
 accuracy_(0,_sk_56)[21]: - 1.69921875 frac[21,168] + mem[21,168] >= 0
 accuracy_(0,_sk_56)[22]: - 1.69921875 frac[22,168] + mem[22,168] >= 0
 accuracy_(1,_sk_56)[0]: - 1.69921875 frac[0,169] + mem[0,169] >= 0
 accuracy_(1,_sk_56)[1]: - 1.69921875 frac[1,169] + mem[1,169] >= 0
 accuracy_(1,_sk_56)[2]: - 1.69921875 frac[2,169] + mem[2,169] >= 0
 accuracy_(1,_sk_56)[3]: - 1.69921875 frac[3,169] + mem[3,169] >= 0
 accuracy_(1,_sk_56)[4]: - 1.69921875 frac[4,169] + mem[4,169] >= 0
 accuracy_(1,_sk_56)[5]: - 1.69921875 frac[5,169] + mem[5,169] >= 0
 accuracy_(1,_sk_56)[6]: - 1.69921875 frac[6,169] + mem[6,169] >= 0
 accuracy_(1,_sk_56)[7]: - 1.69921875 frac[7,169] + mem[7,169] >= 0
 accuracy_(1,_sk_56)[8]: - 1.69921875 frac[8,169] + mem[8,169] >= 0
 accuracy_(1,_sk_56)[9]: - 1.69921875 frac[9,169] + mem[9,169] >= 0
 accuracy_(1,_sk_56)[10]: - 1.69921875 frac[10,169] + mem[10,169] >= 0
 accuracy_(1,_sk_56)[11]: - 1.69921875 frac[11,169] + mem[11,169] >= 0
 accuracy_(1,_sk_56)[12]: - 1.69921875 frac[12,169] + mem[12,169] >= 0
 accuracy_(1,_sk_56)[13]: - 1.69921875 frac[13,169] + mem[13,169] >= 0
 accuracy_(1,_sk_56)[14]: - 1.69921875 frac[14,169] + mem[14,169] >= 0
 accuracy_(1,_sk_56)[15]: - 1.69921875 frac[15,169] + mem[15,169] >= 0
 accuracy_(1,_sk_56)[16]: - 1.69921875 frac[16,169] + mem[16,169] >= 0
 accuracy_(1,_sk_56)[17]: - 1.69921875 frac[17,169] + mem[17,169] >= 0
 accuracy_(1,_sk_56)[18]: - 1.69921875 frac[18,169] + mem[18,169] >= 0
 accuracy_(1,_sk_56)[19]: - 1.69921875 frac[19,169] + mem[19,169] >= 0
 accuracy_(1,_sk_56)[20]: - 1.69921875 frac[20,169] + mem[20,169] >= 0
 accuracy_(1,_sk_56)[21]: - 1.69921875 frac[21,169] + mem[21,169] >= 0
 accuracy_(1,_sk_56)[22]: - 1.69921875 frac[22,169] + mem[22,169] >= 0
 accuracy_(2,_sk_56)[0]: - 1.69921875 frac[0,170] + mem[0,170] >= 0
 accuracy_(2,_sk_56)[1]: - 1.69921875 frac[1,170] + mem[1,170] >= 0
 accuracy_(2,_sk_56)[2]: - 1.69921875 frac[2,170] + mem[2,170] >= 0
 accuracy_(2,_sk_56)[3]: - 1.69921875 frac[3,170] + mem[3,170] >= 0
 accuracy_(2,_sk_56)[4]: - 1.69921875 frac[4,170] + mem[4,170] >= 0
 accuracy_(2,_sk_56)[5]: - 1.69921875 frac[5,170] + mem[5,170] >= 0
 accuracy_(2,_sk_56)[6]: - 1.69921875 frac[6,170] + mem[6,170] >= 0
 accuracy_(2,_sk_56)[7]: - 1.69921875 frac[7,170] + mem[7,170] >= 0
 accuracy_(2,_sk_56)[8]: - 1.69921875 frac[8,170] + mem[8,170] >= 0
 accuracy_(2,_sk_56)[9]: - 1.69921875 frac[9,170] + mem[9,170] >= 0
 accuracy_(2,_sk_56)[10]: - 1.69921875 frac[10,170] + mem[10,170] >= 0
 accuracy_(2,_sk_56)[11]: - 1.69921875 frac[11,170] + mem[11,170] >= 0
 accuracy_(2,_sk_56)[12]: - 1.69921875 frac[12,170] + mem[12,170] >= 0
 accuracy_(2,_sk_56)[13]: - 1.69921875 frac[13,170] + mem[13,170] >= 0
 accuracy_(2,_sk_56)[14]: - 1.69921875 frac[14,170] + mem[14,170] >= 0
 accuracy_(2,_sk_56)[15]: - 1.69921875 frac[15,170] + mem[15,170] >= 0
 accuracy_(2,_sk_56)[16]: - 1.69921875 frac[16,170] + mem[16,170] >= 0
 accuracy_(2,_sk_56)[17]: - 1.69921875 frac[17,170] + mem[17,170] >= 0
 accuracy_(2,_sk_56)[18]: - 1.69921875 frac[18,170] + mem[18,170] >= 0
 accuracy_(2,_sk_56)[19]: - 1.69921875 frac[19,170] + mem[19,170] >= 0
 accuracy_(2,_sk_56)[20]: - 1.69921875 frac[20,170] + mem[20,170] >= 0
 accuracy_(2,_sk_56)[21]: - 1.69921875 frac[21,170] + mem[21,170] >= 0
 accuracy_(2,_sk_56)[22]: - 1.69921875 frac[22,170] + mem[22,170] >= 0
 accuracy_(0,_sk_57)[0]: - 1.69921875 frac[0,171] + mem[0,171] >= 0
 accuracy_(0,_sk_57)[1]: - 1.69921875 frac[1,171] + mem[1,171] >= 0
 accuracy_(0,_sk_57)[2]: - 1.69921875 frac[2,171] + mem[2,171] >= 0
 accuracy_(0,_sk_57)[3]: - 1.69921875 frac[3,171] + mem[3,171] >= 0
 accuracy_(0,_sk_57)[4]: - 1.69921875 frac[4,171] + mem[4,171] >= 0
 accuracy_(0,_sk_57)[5]: - 1.69921875 frac[5,171] + mem[5,171] >= 0
 accuracy_(0,_sk_57)[6]: - 1.69921875 frac[6,171] + mem[6,171] >= 0
 accuracy_(0,_sk_57)[7]: - 1.69921875 frac[7,171] + mem[7,171] >= 0
 accuracy_(0,_sk_57)[8]: - 1.69921875 frac[8,171] + mem[8,171] >= 0
 accuracy_(0,_sk_57)[9]: - 1.69921875 frac[9,171] + mem[9,171] >= 0
 accuracy_(0,_sk_57)[10]: - 1.69921875 frac[10,171] + mem[10,171] >= 0
 accuracy_(0,_sk_57)[11]: - 1.69921875 frac[11,171] + mem[11,171] >= 0
 accuracy_(0,_sk_57)[12]: - 1.69921875 frac[12,171] + mem[12,171] >= 0
 accuracy_(0,_sk_57)[13]: - 1.69921875 frac[13,171] + mem[13,171] >= 0
 accuracy_(0,_sk_57)[14]: - 1.69921875 frac[14,171] + mem[14,171] >= 0
 accuracy_(0,_sk_57)[15]: - 1.69921875 frac[15,171] + mem[15,171] >= 0
 accuracy_(0,_sk_57)[16]: - 1.69921875 frac[16,171] + mem[16,171] >= 0
 accuracy_(0,_sk_57)[17]: - 1.69921875 frac[17,171] + mem[17,171] >= 0
 accuracy_(0,_sk_57)[18]: - 1.69921875 frac[18,171] + mem[18,171] >= 0
 accuracy_(0,_sk_57)[19]: - 1.69921875 frac[19,171] + mem[19,171] >= 0
 accuracy_(0,_sk_57)[20]: - 1.69921875 frac[20,171] + mem[20,171] >= 0
 accuracy_(0,_sk_57)[21]: - 1.69921875 frac[21,171] + mem[21,171] >= 0
 accuracy_(0,_sk_57)[22]: - 1.69921875 frac[22,171] + mem[22,171] >= 0
 accuracy_(1,_sk_57)[0]: - 1.69921875 frac[0,172] + mem[0,172] >= 0
 accuracy_(1,_sk_57)[1]: - 1.69921875 frac[1,172] + mem[1,172] >= 0
 accuracy_(1,_sk_57)[2]: - 1.69921875 frac[2,172] + mem[2,172] >= 0
 accuracy_(1,_sk_57)[3]: - 1.69921875 frac[3,172] + mem[3,172] >= 0
 accuracy_(1,_sk_57)[4]: - 1.69921875 frac[4,172] + mem[4,172] >= 0
 accuracy_(1,_sk_57)[5]: - 1.69921875 frac[5,172] + mem[5,172] >= 0
 accuracy_(1,_sk_57)[6]: - 1.69921875 frac[6,172] + mem[6,172] >= 0
 accuracy_(1,_sk_57)[7]: - 1.69921875 frac[7,172] + mem[7,172] >= 0
 accuracy_(1,_sk_57)[8]: - 1.69921875 frac[8,172] + mem[8,172] >= 0
 accuracy_(1,_sk_57)[9]: - 1.69921875 frac[9,172] + mem[9,172] >= 0
 accuracy_(1,_sk_57)[10]: - 1.69921875 frac[10,172] + mem[10,172] >= 0
 accuracy_(1,_sk_57)[11]: - 1.69921875 frac[11,172] + mem[11,172] >= 0
 accuracy_(1,_sk_57)[12]: - 1.69921875 frac[12,172] + mem[12,172] >= 0
 accuracy_(1,_sk_57)[13]: - 1.69921875 frac[13,172] + mem[13,172] >= 0
 accuracy_(1,_sk_57)[14]: - 1.69921875 frac[14,172] + mem[14,172] >= 0
 accuracy_(1,_sk_57)[15]: - 1.69921875 frac[15,172] + mem[15,172] >= 0
 accuracy_(1,_sk_57)[16]: - 1.69921875 frac[16,172] + mem[16,172] >= 0
 accuracy_(1,_sk_57)[17]: - 1.69921875 frac[17,172] + mem[17,172] >= 0
 accuracy_(1,_sk_57)[18]: - 1.69921875 frac[18,172] + mem[18,172] >= 0
 accuracy_(1,_sk_57)[19]: - 1.69921875 frac[19,172] + mem[19,172] >= 0
 accuracy_(1,_sk_57)[20]: - 1.69921875 frac[20,172] + mem[20,172] >= 0
 accuracy_(1,_sk_57)[21]: - 1.69921875 frac[21,172] + mem[21,172] >= 0
 accuracy_(1,_sk_57)[22]: - 1.69921875 frac[22,172] + mem[22,172] >= 0
 accuracy_(2,_sk_57)[0]: - 1.69921875 frac[0,173] + mem[0,173] >= 0
 accuracy_(2,_sk_57)[1]: - 1.69921875 frac[1,173] + mem[1,173] >= 0
 accuracy_(2,_sk_57)[2]: - 1.69921875 frac[2,173] + mem[2,173] >= 0
 accuracy_(2,_sk_57)[3]: - 1.69921875 frac[3,173] + mem[3,173] >= 0
 accuracy_(2,_sk_57)[4]: - 1.69921875 frac[4,173] + mem[4,173] >= 0
 accuracy_(2,_sk_57)[5]: - 1.69921875 frac[5,173] + mem[5,173] >= 0
 accuracy_(2,_sk_57)[6]: - 1.69921875 frac[6,173] + mem[6,173] >= 0
 accuracy_(2,_sk_57)[7]: - 1.69921875 frac[7,173] + mem[7,173] >= 0
 accuracy_(2,_sk_57)[8]: - 1.69921875 frac[8,173] + mem[8,173] >= 0
 accuracy_(2,_sk_57)[9]: - 1.69921875 frac[9,173] + mem[9,173] >= 0
 accuracy_(2,_sk_57)[10]: - 1.69921875 frac[10,173] + mem[10,173] >= 0
 accuracy_(2,_sk_57)[11]: - 1.69921875 frac[11,173] + mem[11,173] >= 0
 accuracy_(2,_sk_57)[12]: - 1.69921875 frac[12,173] + mem[12,173] >= 0
 accuracy_(2,_sk_57)[13]: - 1.69921875 frac[13,173] + mem[13,173] >= 0
 accuracy_(2,_sk_57)[14]: - 1.69921875 frac[14,173] + mem[14,173] >= 0
 accuracy_(2,_sk_57)[15]: - 1.69921875 frac[15,173] + mem[15,173] >= 0
 accuracy_(2,_sk_57)[16]: - 1.69921875 frac[16,173] + mem[16,173] >= 0
 accuracy_(2,_sk_57)[17]: - 1.69921875 frac[17,173] + mem[17,173] >= 0
 accuracy_(2,_sk_57)[18]: - 1.69921875 frac[18,173] + mem[18,173] >= 0
 accuracy_(2,_sk_57)[19]: - 1.69921875 frac[19,173] + mem[19,173] >= 0
 accuracy_(2,_sk_57)[20]: - 1.69921875 frac[20,173] + mem[20,173] >= 0
 accuracy_(2,_sk_57)[21]: - 1.69921875 frac[21,173] + mem[21,173] >= 0
 accuracy_(2,_sk_57)[22]: - 1.69921875 frac[22,173] + mem[22,173] >= 0
 accuracy_(0,_sk_58)[0]: - 1.69921875 frac[0,174] + mem[0,174] >= 0
 accuracy_(0,_sk_58)[1]: - 1.69921875 frac[1,174] + mem[1,174] >= 0
 accuracy_(0,_sk_58)[2]: - 1.69921875 frac[2,174] + mem[2,174] >= 0
 accuracy_(0,_sk_58)[3]: - 1.69921875 frac[3,174] + mem[3,174] >= 0
 accuracy_(0,_sk_58)[4]: - 1.69921875 frac[4,174] + mem[4,174] >= 0
 accuracy_(0,_sk_58)[5]: - 1.69921875 frac[5,174] + mem[5,174] >= 0
 accuracy_(0,_sk_58)[6]: - 1.69921875 frac[6,174] + mem[6,174] >= 0
 accuracy_(0,_sk_58)[7]: - 1.69921875 frac[7,174] + mem[7,174] >= 0
 accuracy_(0,_sk_58)[8]: - 1.69921875 frac[8,174] + mem[8,174] >= 0
 accuracy_(0,_sk_58)[9]: - 1.69921875 frac[9,174] + mem[9,174] >= 0
 accuracy_(0,_sk_58)[10]: - 1.69921875 frac[10,174] + mem[10,174] >= 0
 accuracy_(0,_sk_58)[11]: - 1.69921875 frac[11,174] + mem[11,174] >= 0
 accuracy_(0,_sk_58)[12]: - 1.69921875 frac[12,174] + mem[12,174] >= 0
 accuracy_(0,_sk_58)[13]: - 1.69921875 frac[13,174] + mem[13,174] >= 0
 accuracy_(0,_sk_58)[14]: - 1.69921875 frac[14,174] + mem[14,174] >= 0
 accuracy_(0,_sk_58)[15]: - 1.69921875 frac[15,174] + mem[15,174] >= 0
 accuracy_(0,_sk_58)[16]: - 1.69921875 frac[16,174] + mem[16,174] >= 0
 accuracy_(0,_sk_58)[17]: - 1.69921875 frac[17,174] + mem[17,174] >= 0
 accuracy_(0,_sk_58)[18]: - 1.69921875 frac[18,174] + mem[18,174] >= 0
 accuracy_(0,_sk_58)[19]: - 1.69921875 frac[19,174] + mem[19,174] >= 0
 accuracy_(0,_sk_58)[20]: - 1.69921875 frac[20,174] + mem[20,174] >= 0
 accuracy_(0,_sk_58)[21]: - 1.69921875 frac[21,174] + mem[21,174] >= 0
 accuracy_(0,_sk_58)[22]: - 1.69921875 frac[22,174] + mem[22,174] >= 0
 accuracy_(1,_sk_58)[0]: - 1.69921875 frac[0,175] + mem[0,175] >= 0
 accuracy_(1,_sk_58)[1]: - 1.69921875 frac[1,175] + mem[1,175] >= 0
 accuracy_(1,_sk_58)[2]: - 1.69921875 frac[2,175] + mem[2,175] >= 0
 accuracy_(1,_sk_58)[3]: - 1.69921875 frac[3,175] + mem[3,175] >= 0
 accuracy_(1,_sk_58)[4]: - 1.69921875 frac[4,175] + mem[4,175] >= 0
 accuracy_(1,_sk_58)[5]: - 1.69921875 frac[5,175] + mem[5,175] >= 0
 accuracy_(1,_sk_58)[6]: - 1.69921875 frac[6,175] + mem[6,175] >= 0
 accuracy_(1,_sk_58)[7]: - 1.69921875 frac[7,175] + mem[7,175] >= 0
 accuracy_(1,_sk_58)[8]: - 1.69921875 frac[8,175] + mem[8,175] >= 0
 accuracy_(1,_sk_58)[9]: - 1.69921875 frac[9,175] + mem[9,175] >= 0
 accuracy_(1,_sk_58)[10]: - 1.69921875 frac[10,175] + mem[10,175] >= 0
 accuracy_(1,_sk_58)[11]: - 1.69921875 frac[11,175] + mem[11,175] >= 0
 accuracy_(1,_sk_58)[12]: - 1.69921875 frac[12,175] + mem[12,175] >= 0
 accuracy_(1,_sk_58)[13]: - 1.69921875 frac[13,175] + mem[13,175] >= 0
 accuracy_(1,_sk_58)[14]: - 1.69921875 frac[14,175] + mem[14,175] >= 0
 accuracy_(1,_sk_58)[15]: - 1.69921875 frac[15,175] + mem[15,175] >= 0
 accuracy_(1,_sk_58)[16]: - 1.69921875 frac[16,175] + mem[16,175] >= 0
 accuracy_(1,_sk_58)[17]: - 1.69921875 frac[17,175] + mem[17,175] >= 0
 accuracy_(1,_sk_58)[18]: - 1.69921875 frac[18,175] + mem[18,175] >= 0
 accuracy_(1,_sk_58)[19]: - 1.69921875 frac[19,175] + mem[19,175] >= 0
 accuracy_(1,_sk_58)[20]: - 1.69921875 frac[20,175] + mem[20,175] >= 0
 accuracy_(1,_sk_58)[21]: - 1.69921875 frac[21,175] + mem[21,175] >= 0
 accuracy_(1,_sk_58)[22]: - 1.69921875 frac[22,175] + mem[22,175] >= 0
 accuracy_(2,_sk_58)[0]: - 1.69921875 frac[0,176] + mem[0,176] >= 0
 accuracy_(2,_sk_58)[1]: - 1.69921875 frac[1,176] + mem[1,176] >= 0
 accuracy_(2,_sk_58)[2]: - 1.69921875 frac[2,176] + mem[2,176] >= 0
 accuracy_(2,_sk_58)[3]: - 1.69921875 frac[3,176] + mem[3,176] >= 0
 accuracy_(2,_sk_58)[4]: - 1.69921875 frac[4,176] + mem[4,176] >= 0
 accuracy_(2,_sk_58)[5]: - 1.69921875 frac[5,176] + mem[5,176] >= 0
 accuracy_(2,_sk_58)[6]: - 1.69921875 frac[6,176] + mem[6,176] >= 0
 accuracy_(2,_sk_58)[7]: - 1.69921875 frac[7,176] + mem[7,176] >= 0
 accuracy_(2,_sk_58)[8]: - 1.69921875 frac[8,176] + mem[8,176] >= 0
 accuracy_(2,_sk_58)[9]: - 1.69921875 frac[9,176] + mem[9,176] >= 0
 accuracy_(2,_sk_58)[10]: - 1.69921875 frac[10,176] + mem[10,176] >= 0
 accuracy_(2,_sk_58)[11]: - 1.69921875 frac[11,176] + mem[11,176] >= 0
 accuracy_(2,_sk_58)[12]: - 1.69921875 frac[12,176] + mem[12,176] >= 0
 accuracy_(2,_sk_58)[13]: - 1.69921875 frac[13,176] + mem[13,176] >= 0
 accuracy_(2,_sk_58)[14]: - 1.69921875 frac[14,176] + mem[14,176] >= 0
 accuracy_(2,_sk_58)[15]: - 1.69921875 frac[15,176] + mem[15,176] >= 0
 accuracy_(2,_sk_58)[16]: - 1.69921875 frac[16,176] + mem[16,176] >= 0
 accuracy_(2,_sk_58)[17]: - 1.69921875 frac[17,176] + mem[17,176] >= 0
 accuracy_(2,_sk_58)[18]: - 1.69921875 frac[18,176] + mem[18,176] >= 0
 accuracy_(2,_sk_58)[19]: - 1.69921875 frac[19,176] + mem[19,176] >= 0
 accuracy_(2,_sk_58)[20]: - 1.69921875 frac[20,176] + mem[20,176] >= 0
 accuracy_(2,_sk_58)[21]: - 1.69921875 frac[21,176] + mem[21,176] >= 0
 accuracy_(2,_sk_58)[22]: - 1.69921875 frac[22,176] + mem[22,176] >= 0
 accuracy_(0,_sk_59)[0]: - 1.69921875 frac[0,177] + mem[0,177] >= 0
 accuracy_(0,_sk_59)[1]: - 1.69921875 frac[1,177] + mem[1,177] >= 0
 accuracy_(0,_sk_59)[2]: - 1.69921875 frac[2,177] + mem[2,177] >= 0
 accuracy_(0,_sk_59)[3]: - 1.69921875 frac[3,177] + mem[3,177] >= 0
 accuracy_(0,_sk_59)[4]: - 1.69921875 frac[4,177] + mem[4,177] >= 0
 accuracy_(0,_sk_59)[5]: - 1.69921875 frac[5,177] + mem[5,177] >= 0
 accuracy_(0,_sk_59)[6]: - 1.69921875 frac[6,177] + mem[6,177] >= 0
 accuracy_(0,_sk_59)[7]: - 1.69921875 frac[7,177] + mem[7,177] >= 0
 accuracy_(0,_sk_59)[8]: - 1.69921875 frac[8,177] + mem[8,177] >= 0
 accuracy_(0,_sk_59)[9]: - 1.69921875 frac[9,177] + mem[9,177] >= 0
 accuracy_(0,_sk_59)[10]: - 1.69921875 frac[10,177] + mem[10,177] >= 0
 accuracy_(0,_sk_59)[11]: - 1.69921875 frac[11,177] + mem[11,177] >= 0
 accuracy_(0,_sk_59)[12]: - 1.69921875 frac[12,177] + mem[12,177] >= 0
 accuracy_(0,_sk_59)[13]: - 1.69921875 frac[13,177] + mem[13,177] >= 0
 accuracy_(0,_sk_59)[14]: - 1.69921875 frac[14,177] + mem[14,177] >= 0
 accuracy_(0,_sk_59)[15]: - 1.69921875 frac[15,177] + mem[15,177] >= 0
 accuracy_(0,_sk_59)[16]: - 1.69921875 frac[16,177] + mem[16,177] >= 0
 accuracy_(0,_sk_59)[17]: - 1.69921875 frac[17,177] + mem[17,177] >= 0
 accuracy_(0,_sk_59)[18]: - 1.69921875 frac[18,177] + mem[18,177] >= 0
 accuracy_(0,_sk_59)[19]: - 1.69921875 frac[19,177] + mem[19,177] >= 0
 accuracy_(0,_sk_59)[20]: - 1.69921875 frac[20,177] + mem[20,177] >= 0
 accuracy_(0,_sk_59)[21]: - 1.69921875 frac[21,177] + mem[21,177] >= 0
 accuracy_(0,_sk_59)[22]: - 1.69921875 frac[22,177] + mem[22,177] >= 0
 accuracy_(1,_sk_59)[0]: - 1.69921875 frac[0,178] + mem[0,178] >= 0
 accuracy_(1,_sk_59)[1]: - 1.69921875 frac[1,178] + mem[1,178] >= 0
 accuracy_(1,_sk_59)[2]: - 1.69921875 frac[2,178] + mem[2,178] >= 0
 accuracy_(1,_sk_59)[3]: - 1.69921875 frac[3,178] + mem[3,178] >= 0
 accuracy_(1,_sk_59)[4]: - 1.69921875 frac[4,178] + mem[4,178] >= 0
 accuracy_(1,_sk_59)[5]: - 1.69921875 frac[5,178] + mem[5,178] >= 0
 accuracy_(1,_sk_59)[6]: - 1.69921875 frac[6,178] + mem[6,178] >= 0
 accuracy_(1,_sk_59)[7]: - 1.69921875 frac[7,178] + mem[7,178] >= 0
 accuracy_(1,_sk_59)[8]: - 1.69921875 frac[8,178] + mem[8,178] >= 0
 accuracy_(1,_sk_59)[9]: - 1.69921875 frac[9,178] + mem[9,178] >= 0
 accuracy_(1,_sk_59)[10]: - 1.69921875 frac[10,178] + mem[10,178] >= 0
 accuracy_(1,_sk_59)[11]: - 1.69921875 frac[11,178] + mem[11,178] >= 0
 accuracy_(1,_sk_59)[12]: - 1.69921875 frac[12,178] + mem[12,178] >= 0
 accuracy_(1,_sk_59)[13]: - 1.69921875 frac[13,178] + mem[13,178] >= 0
 accuracy_(1,_sk_59)[14]: - 1.69921875 frac[14,178] + mem[14,178] >= 0
 accuracy_(1,_sk_59)[15]: - 1.69921875 frac[15,178] + mem[15,178] >= 0
 accuracy_(1,_sk_59)[16]: - 1.69921875 frac[16,178] + mem[16,178] >= 0
 accuracy_(1,_sk_59)[17]: - 1.69921875 frac[17,178] + mem[17,178] >= 0
 accuracy_(1,_sk_59)[18]: - 1.69921875 frac[18,178] + mem[18,178] >= 0
 accuracy_(1,_sk_59)[19]: - 1.69921875 frac[19,178] + mem[19,178] >= 0
 accuracy_(1,_sk_59)[20]: - 1.69921875 frac[20,178] + mem[20,178] >= 0
 accuracy_(1,_sk_59)[21]: - 1.69921875 frac[21,178] + mem[21,178] >= 0
 accuracy_(1,_sk_59)[22]: - 1.69921875 frac[22,178] + mem[22,178] >= 0
 accuracy_(2,_sk_59)[0]: - 1.69921875 frac[0,179] + mem[0,179] >= 0
 accuracy_(2,_sk_59)[1]: - 1.69921875 frac[1,179] + mem[1,179] >= 0
 accuracy_(2,_sk_59)[2]: - 1.69921875 frac[2,179] + mem[2,179] >= 0
 accuracy_(2,_sk_59)[3]: - 1.69921875 frac[3,179] + mem[3,179] >= 0
 accuracy_(2,_sk_59)[4]: - 1.69921875 frac[4,179] + mem[4,179] >= 0
 accuracy_(2,_sk_59)[5]: - 1.69921875 frac[5,179] + mem[5,179] >= 0
 accuracy_(2,_sk_59)[6]: - 1.69921875 frac[6,179] + mem[6,179] >= 0
 accuracy_(2,_sk_59)[7]: - 1.69921875 frac[7,179] + mem[7,179] >= 0
 accuracy_(2,_sk_59)[8]: - 1.69921875 frac[8,179] + mem[8,179] >= 0
 accuracy_(2,_sk_59)[9]: - 1.69921875 frac[9,179] + mem[9,179] >= 0
 accuracy_(2,_sk_59)[10]: - 1.69921875 frac[10,179] + mem[10,179] >= 0
 accuracy_(2,_sk_59)[11]: - 1.69921875 frac[11,179] + mem[11,179] >= 0
 accuracy_(2,_sk_59)[12]: - 1.69921875 frac[12,179] + mem[12,179] >= 0
 accuracy_(2,_sk_59)[13]: - 1.69921875 frac[13,179] + mem[13,179] >= 0
 accuracy_(2,_sk_59)[14]: - 1.69921875 frac[14,179] + mem[14,179] >= 0
 accuracy_(2,_sk_59)[15]: - 1.69921875 frac[15,179] + mem[15,179] >= 0
 accuracy_(2,_sk_59)[16]: - 1.69921875 frac[16,179] + mem[16,179] >= 0
 accuracy_(2,_sk_59)[17]: - 1.69921875 frac[17,179] + mem[17,179] >= 0
 accuracy_(2,_sk_59)[18]: - 1.69921875 frac[18,179] + mem[18,179] >= 0
 accuracy_(2,_sk_59)[19]: - 1.69921875 frac[19,179] + mem[19,179] >= 0
 accuracy_(2,_sk_59)[20]: - 1.69921875 frac[20,179] + mem[20,179] >= 0
 accuracy_(2,_sk_59)[21]: - 1.69921875 frac[21,179] + mem[21,179] >= 0
 accuracy_(2,_sk_59)[22]: - 1.69921875 frac[22,179] + mem[22,179] >= 0
 accuracy_(0,_sk_60)[0]: - 1.69921875 frac[0,180] + mem[0,180] >= 0
 accuracy_(0,_sk_60)[1]: - 1.69921875 frac[1,180] + mem[1,180] >= 0
 accuracy_(0,_sk_60)[2]: - 1.69921875 frac[2,180] + mem[2,180] >= 0
 accuracy_(0,_sk_60)[3]: - 1.69921875 frac[3,180] + mem[3,180] >= 0
 accuracy_(0,_sk_60)[4]: - 1.69921875 frac[4,180] + mem[4,180] >= 0
 accuracy_(0,_sk_60)[5]: - 1.69921875 frac[5,180] + mem[5,180] >= 0
 accuracy_(0,_sk_60)[6]: - 1.69921875 frac[6,180] + mem[6,180] >= 0
 accuracy_(0,_sk_60)[7]: - 1.69921875 frac[7,180] + mem[7,180] >= 0
 accuracy_(0,_sk_60)[8]: - 1.69921875 frac[8,180] + mem[8,180] >= 0
 accuracy_(0,_sk_60)[9]: - 1.69921875 frac[9,180] + mem[9,180] >= 0
 accuracy_(0,_sk_60)[10]: - 1.69921875 frac[10,180] + mem[10,180] >= 0
 accuracy_(0,_sk_60)[11]: - 1.69921875 frac[11,180] + mem[11,180] >= 0
 accuracy_(0,_sk_60)[12]: - 1.69921875 frac[12,180] + mem[12,180] >= 0
 accuracy_(0,_sk_60)[13]: - 1.69921875 frac[13,180] + mem[13,180] >= 0
 accuracy_(0,_sk_60)[14]: - 1.69921875 frac[14,180] + mem[14,180] >= 0
 accuracy_(0,_sk_60)[15]: - 1.69921875 frac[15,180] + mem[15,180] >= 0
 accuracy_(0,_sk_60)[16]: - 1.69921875 frac[16,180] + mem[16,180] >= 0
 accuracy_(0,_sk_60)[17]: - 1.69921875 frac[17,180] + mem[17,180] >= 0
 accuracy_(0,_sk_60)[18]: - 1.69921875 frac[18,180] + mem[18,180] >= 0
 accuracy_(0,_sk_60)[19]: - 1.69921875 frac[19,180] + mem[19,180] >= 0
 accuracy_(0,_sk_60)[20]: - 1.69921875 frac[20,180] + mem[20,180] >= 0
 accuracy_(0,_sk_60)[21]: - 1.69921875 frac[21,180] + mem[21,180] >= 0
 accuracy_(0,_sk_60)[22]: - 1.69921875 frac[22,180] + mem[22,180] >= 0
 accuracy_(1,_sk_60)[0]: - 1.69921875 frac[0,181] + mem[0,181] >= 0
 accuracy_(1,_sk_60)[1]: - 1.69921875 frac[1,181] + mem[1,181] >= 0
 accuracy_(1,_sk_60)[2]: - 1.69921875 frac[2,181] + mem[2,181] >= 0
 accuracy_(1,_sk_60)[3]: - 1.69921875 frac[3,181] + mem[3,181] >= 0
 accuracy_(1,_sk_60)[4]: - 1.69921875 frac[4,181] + mem[4,181] >= 0
 accuracy_(1,_sk_60)[5]: - 1.69921875 frac[5,181] + mem[5,181] >= 0
 accuracy_(1,_sk_60)[6]: - 1.69921875 frac[6,181] + mem[6,181] >= 0
 accuracy_(1,_sk_60)[7]: - 1.69921875 frac[7,181] + mem[7,181] >= 0
 accuracy_(1,_sk_60)[8]: - 1.69921875 frac[8,181] + mem[8,181] >= 0
 accuracy_(1,_sk_60)[9]: - 1.69921875 frac[9,181] + mem[9,181] >= 0
 accuracy_(1,_sk_60)[10]: - 1.69921875 frac[10,181] + mem[10,181] >= 0
 accuracy_(1,_sk_60)[11]: - 1.69921875 frac[11,181] + mem[11,181] >= 0
 accuracy_(1,_sk_60)[12]: - 1.69921875 frac[12,181] + mem[12,181] >= 0
 accuracy_(1,_sk_60)[13]: - 1.69921875 frac[13,181] + mem[13,181] >= 0
 accuracy_(1,_sk_60)[14]: - 1.69921875 frac[14,181] + mem[14,181] >= 0
 accuracy_(1,_sk_60)[15]: - 1.69921875 frac[15,181] + mem[15,181] >= 0
 accuracy_(1,_sk_60)[16]: - 1.69921875 frac[16,181] + mem[16,181] >= 0
 accuracy_(1,_sk_60)[17]: - 1.69921875 frac[17,181] + mem[17,181] >= 0
 accuracy_(1,_sk_60)[18]: - 1.69921875 frac[18,181] + mem[18,181] >= 0
 accuracy_(1,_sk_60)[19]: - 1.69921875 frac[19,181] + mem[19,181] >= 0
 accuracy_(1,_sk_60)[20]: - 1.69921875 frac[20,181] + mem[20,181] >= 0
 accuracy_(1,_sk_60)[21]: - 1.69921875 frac[21,181] + mem[21,181] >= 0
 accuracy_(1,_sk_60)[22]: - 1.69921875 frac[22,181] + mem[22,181] >= 0
 accuracy_(2,_sk_60)[0]: - 1.69921875 frac[0,182] + mem[0,182] >= 0
 accuracy_(2,_sk_60)[1]: - 1.69921875 frac[1,182] + mem[1,182] >= 0
 accuracy_(2,_sk_60)[2]: - 1.69921875 frac[2,182] + mem[2,182] >= 0
 accuracy_(2,_sk_60)[3]: - 1.69921875 frac[3,182] + mem[3,182] >= 0
 accuracy_(2,_sk_60)[4]: - 1.69921875 frac[4,182] + mem[4,182] >= 0
 accuracy_(2,_sk_60)[5]: - 1.69921875 frac[5,182] + mem[5,182] >= 0
 accuracy_(2,_sk_60)[6]: - 1.69921875 frac[6,182] + mem[6,182] >= 0
 accuracy_(2,_sk_60)[7]: - 1.69921875 frac[7,182] + mem[7,182] >= 0
 accuracy_(2,_sk_60)[8]: - 1.69921875 frac[8,182] + mem[8,182] >= 0
 accuracy_(2,_sk_60)[9]: - 1.69921875 frac[9,182] + mem[9,182] >= 0
 accuracy_(2,_sk_60)[10]: - 1.69921875 frac[10,182] + mem[10,182] >= 0
 accuracy_(2,_sk_60)[11]: - 1.69921875 frac[11,182] + mem[11,182] >= 0
 accuracy_(2,_sk_60)[12]: - 1.69921875 frac[12,182] + mem[12,182] >= 0
 accuracy_(2,_sk_60)[13]: - 1.69921875 frac[13,182] + mem[13,182] >= 0
 accuracy_(2,_sk_60)[14]: - 1.69921875 frac[14,182] + mem[14,182] >= 0
 accuracy_(2,_sk_60)[15]: - 1.69921875 frac[15,182] + mem[15,182] >= 0
 accuracy_(2,_sk_60)[16]: - 1.69921875 frac[16,182] + mem[16,182] >= 0
 accuracy_(2,_sk_60)[17]: - 1.69921875 frac[17,182] + mem[17,182] >= 0
 accuracy_(2,_sk_60)[18]: - 1.69921875 frac[18,182] + mem[18,182] >= 0
 accuracy_(2,_sk_60)[19]: - 1.69921875 frac[19,182] + mem[19,182] >= 0
 accuracy_(2,_sk_60)[20]: - 1.69921875 frac[20,182] + mem[20,182] >= 0
 accuracy_(2,_sk_60)[21]: - 1.69921875 frac[21,182] + mem[21,182] >= 0
 accuracy_(2,_sk_60)[22]: - 1.69921875 frac[22,182] + mem[22,182] >= 0
 accuracy_(0,_sk_61)[0]: - 1.69921875 frac[0,183] + mem[0,183] >= 0
 accuracy_(0,_sk_61)[1]: - 1.69921875 frac[1,183] + mem[1,183] >= 0
 accuracy_(0,_sk_61)[2]: - 1.69921875 frac[2,183] + mem[2,183] >= 0
 accuracy_(0,_sk_61)[3]: - 1.69921875 frac[3,183] + mem[3,183] >= 0
 accuracy_(0,_sk_61)[4]: - 1.69921875 frac[4,183] + mem[4,183] >= 0
 accuracy_(0,_sk_61)[5]: - 1.69921875 frac[5,183] + mem[5,183] >= 0
 accuracy_(0,_sk_61)[6]: - 1.69921875 frac[6,183] + mem[6,183] >= 0
 accuracy_(0,_sk_61)[7]: - 1.69921875 frac[7,183] + mem[7,183] >= 0
 accuracy_(0,_sk_61)[8]: - 1.69921875 frac[8,183] + mem[8,183] >= 0
 accuracy_(0,_sk_61)[9]: - 1.69921875 frac[9,183] + mem[9,183] >= 0
 accuracy_(0,_sk_61)[10]: - 1.69921875 frac[10,183] + mem[10,183] >= 0
 accuracy_(0,_sk_61)[11]: - 1.69921875 frac[11,183] + mem[11,183] >= 0
 accuracy_(0,_sk_61)[12]: - 1.69921875 frac[12,183] + mem[12,183] >= 0
 accuracy_(0,_sk_61)[13]: - 1.69921875 frac[13,183] + mem[13,183] >= 0
 accuracy_(0,_sk_61)[14]: - 1.69921875 frac[14,183] + mem[14,183] >= 0
 accuracy_(0,_sk_61)[15]: - 1.69921875 frac[15,183] + mem[15,183] >= 0
 accuracy_(0,_sk_61)[16]: - 1.69921875 frac[16,183] + mem[16,183] >= 0
 accuracy_(0,_sk_61)[17]: - 1.69921875 frac[17,183] + mem[17,183] >= 0
 accuracy_(0,_sk_61)[18]: - 1.69921875 frac[18,183] + mem[18,183] >= 0
 accuracy_(0,_sk_61)[19]: - 1.69921875 frac[19,183] + mem[19,183] >= 0
 accuracy_(0,_sk_61)[20]: - 1.69921875 frac[20,183] + mem[20,183] >= 0
 accuracy_(0,_sk_61)[21]: - 1.69921875 frac[21,183] + mem[21,183] >= 0
 accuracy_(0,_sk_61)[22]: - 1.69921875 frac[22,183] + mem[22,183] >= 0
 accuracy_(1,_sk_61)[0]: - 1.69921875 frac[0,184] + mem[0,184] >= 0
 accuracy_(1,_sk_61)[1]: - 1.69921875 frac[1,184] + mem[1,184] >= 0
 accuracy_(1,_sk_61)[2]: - 1.69921875 frac[2,184] + mem[2,184] >= 0
 accuracy_(1,_sk_61)[3]: - 1.69921875 frac[3,184] + mem[3,184] >= 0
 accuracy_(1,_sk_61)[4]: - 1.69921875 frac[4,184] + mem[4,184] >= 0
 accuracy_(1,_sk_61)[5]: - 1.69921875 frac[5,184] + mem[5,184] >= 0
 accuracy_(1,_sk_61)[6]: - 1.69921875 frac[6,184] + mem[6,184] >= 0
 accuracy_(1,_sk_61)[7]: - 1.69921875 frac[7,184] + mem[7,184] >= 0
 accuracy_(1,_sk_61)[8]: - 1.69921875 frac[8,184] + mem[8,184] >= 0
 accuracy_(1,_sk_61)[9]: - 1.69921875 frac[9,184] + mem[9,184] >= 0
 accuracy_(1,_sk_61)[10]: - 1.69921875 frac[10,184] + mem[10,184] >= 0
 accuracy_(1,_sk_61)[11]: - 1.69921875 frac[11,184] + mem[11,184] >= 0
 accuracy_(1,_sk_61)[12]: - 1.69921875 frac[12,184] + mem[12,184] >= 0
 accuracy_(1,_sk_61)[13]: - 1.69921875 frac[13,184] + mem[13,184] >= 0
 accuracy_(1,_sk_61)[14]: - 1.69921875 frac[14,184] + mem[14,184] >= 0
 accuracy_(1,_sk_61)[15]: - 1.69921875 frac[15,184] + mem[15,184] >= 0
 accuracy_(1,_sk_61)[16]: - 1.69921875 frac[16,184] + mem[16,184] >= 0
 accuracy_(1,_sk_61)[17]: - 1.69921875 frac[17,184] + mem[17,184] >= 0
 accuracy_(1,_sk_61)[18]: - 1.69921875 frac[18,184] + mem[18,184] >= 0
 accuracy_(1,_sk_61)[19]: - 1.69921875 frac[19,184] + mem[19,184] >= 0
 accuracy_(1,_sk_61)[20]: - 1.69921875 frac[20,184] + mem[20,184] >= 0
 accuracy_(1,_sk_61)[21]: - 1.69921875 frac[21,184] + mem[21,184] >= 0
 accuracy_(1,_sk_61)[22]: - 1.69921875 frac[22,184] + mem[22,184] >= 0
 accuracy_(2,_sk_61)[0]: - 1.69921875 frac[0,185] + mem[0,185] >= 0
 accuracy_(2,_sk_61)[1]: - 1.69921875 frac[1,185] + mem[1,185] >= 0
 accuracy_(2,_sk_61)[2]: - 1.69921875 frac[2,185] + mem[2,185] >= 0
 accuracy_(2,_sk_61)[3]: - 1.69921875 frac[3,185] + mem[3,185] >= 0
 accuracy_(2,_sk_61)[4]: - 1.69921875 frac[4,185] + mem[4,185] >= 0
 accuracy_(2,_sk_61)[5]: - 1.69921875 frac[5,185] + mem[5,185] >= 0
 accuracy_(2,_sk_61)[6]: - 1.69921875 frac[6,185] + mem[6,185] >= 0
 accuracy_(2,_sk_61)[7]: - 1.69921875 frac[7,185] + mem[7,185] >= 0
 accuracy_(2,_sk_61)[8]: - 1.69921875 frac[8,185] + mem[8,185] >= 0
 accuracy_(2,_sk_61)[9]: - 1.69921875 frac[9,185] + mem[9,185] >= 0
 accuracy_(2,_sk_61)[10]: - 1.69921875 frac[10,185] + mem[10,185] >= 0
 accuracy_(2,_sk_61)[11]: - 1.69921875 frac[11,185] + mem[11,185] >= 0
 accuracy_(2,_sk_61)[12]: - 1.69921875 frac[12,185] + mem[12,185] >= 0
 accuracy_(2,_sk_61)[13]: - 1.69921875 frac[13,185] + mem[13,185] >= 0
 accuracy_(2,_sk_61)[14]: - 1.69921875 frac[14,185] + mem[14,185] >= 0
 accuracy_(2,_sk_61)[15]: - 1.69921875 frac[15,185] + mem[15,185] >= 0
 accuracy_(2,_sk_61)[16]: - 1.69921875 frac[16,185] + mem[16,185] >= 0
 accuracy_(2,_sk_61)[17]: - 1.69921875 frac[17,185] + mem[17,185] >= 0
 accuracy_(2,_sk_61)[18]: - 1.69921875 frac[18,185] + mem[18,185] >= 0
 accuracy_(2,_sk_61)[19]: - 1.69921875 frac[19,185] + mem[19,185] >= 0
 accuracy_(2,_sk_61)[20]: - 1.69921875 frac[20,185] + mem[20,185] >= 0
 accuracy_(2,_sk_61)[21]: - 1.69921875 frac[21,185] + mem[21,185] >= 0
 accuracy_(2,_sk_61)[22]: - 1.69921875 frac[22,185] + mem[22,185] >= 0
 accuracy_(0,_sk_62)[0]: - 1.69921875 frac[0,186] + mem[0,186] >= 0
 accuracy_(0,_sk_62)[1]: - 1.69921875 frac[1,186] + mem[1,186] >= 0
 accuracy_(0,_sk_62)[2]: - 1.69921875 frac[2,186] + mem[2,186] >= 0
 accuracy_(0,_sk_62)[3]: - 1.69921875 frac[3,186] + mem[3,186] >= 0
 accuracy_(0,_sk_62)[4]: - 1.69921875 frac[4,186] + mem[4,186] >= 0
 accuracy_(0,_sk_62)[5]: - 1.69921875 frac[5,186] + mem[5,186] >= 0
 accuracy_(0,_sk_62)[6]: - 1.69921875 frac[6,186] + mem[6,186] >= 0
 accuracy_(0,_sk_62)[7]: - 1.69921875 frac[7,186] + mem[7,186] >= 0
 accuracy_(0,_sk_62)[8]: - 1.69921875 frac[8,186] + mem[8,186] >= 0
 accuracy_(0,_sk_62)[9]: - 1.69921875 frac[9,186] + mem[9,186] >= 0
 accuracy_(0,_sk_62)[10]: - 1.69921875 frac[10,186] + mem[10,186] >= 0
 accuracy_(0,_sk_62)[11]: - 1.69921875 frac[11,186] + mem[11,186] >= 0
 accuracy_(0,_sk_62)[12]: - 1.69921875 frac[12,186] + mem[12,186] >= 0
 accuracy_(0,_sk_62)[13]: - 1.69921875 frac[13,186] + mem[13,186] >= 0
 accuracy_(0,_sk_62)[14]: - 1.69921875 frac[14,186] + mem[14,186] >= 0
 accuracy_(0,_sk_62)[15]: - 1.69921875 frac[15,186] + mem[15,186] >= 0
 accuracy_(0,_sk_62)[16]: - 1.69921875 frac[16,186] + mem[16,186] >= 0
 accuracy_(0,_sk_62)[17]: - 1.69921875 frac[17,186] + mem[17,186] >= 0
 accuracy_(0,_sk_62)[18]: - 1.69921875 frac[18,186] + mem[18,186] >= 0
 accuracy_(0,_sk_62)[19]: - 1.69921875 frac[19,186] + mem[19,186] >= 0
 accuracy_(0,_sk_62)[20]: - 1.69921875 frac[20,186] + mem[20,186] >= 0
 accuracy_(0,_sk_62)[21]: - 1.69921875 frac[21,186] + mem[21,186] >= 0
 accuracy_(0,_sk_62)[22]: - 1.69921875 frac[22,186] + mem[22,186] >= 0
 accuracy_(1,_sk_62)[0]: - 1.69921875 frac[0,187] + mem[0,187] >= 0
 accuracy_(1,_sk_62)[1]: - 1.69921875 frac[1,187] + mem[1,187] >= 0
 accuracy_(1,_sk_62)[2]: - 1.69921875 frac[2,187] + mem[2,187] >= 0
 accuracy_(1,_sk_62)[3]: - 1.69921875 frac[3,187] + mem[3,187] >= 0
 accuracy_(1,_sk_62)[4]: - 1.69921875 frac[4,187] + mem[4,187] >= 0
 accuracy_(1,_sk_62)[5]: - 1.69921875 frac[5,187] + mem[5,187] >= 0
 accuracy_(1,_sk_62)[6]: - 1.69921875 frac[6,187] + mem[6,187] >= 0
 accuracy_(1,_sk_62)[7]: - 1.69921875 frac[7,187] + mem[7,187] >= 0
 accuracy_(1,_sk_62)[8]: - 1.69921875 frac[8,187] + mem[8,187] >= 0
 accuracy_(1,_sk_62)[9]: - 1.69921875 frac[9,187] + mem[9,187] >= 0
 accuracy_(1,_sk_62)[10]: - 1.69921875 frac[10,187] + mem[10,187] >= 0
 accuracy_(1,_sk_62)[11]: - 1.69921875 frac[11,187] + mem[11,187] >= 0
 accuracy_(1,_sk_62)[12]: - 1.69921875 frac[12,187] + mem[12,187] >= 0
 accuracy_(1,_sk_62)[13]: - 1.69921875 frac[13,187] + mem[13,187] >= 0
 accuracy_(1,_sk_62)[14]: - 1.69921875 frac[14,187] + mem[14,187] >= 0
 accuracy_(1,_sk_62)[15]: - 1.69921875 frac[15,187] + mem[15,187] >= 0
 accuracy_(1,_sk_62)[16]: - 1.69921875 frac[16,187] + mem[16,187] >= 0
 accuracy_(1,_sk_62)[17]: - 1.69921875 frac[17,187] + mem[17,187] >= 0
 accuracy_(1,_sk_62)[18]: - 1.69921875 frac[18,187] + mem[18,187] >= 0
 accuracy_(1,_sk_62)[19]: - 1.69921875 frac[19,187] + mem[19,187] >= 0
 accuracy_(1,_sk_62)[20]: - 1.69921875 frac[20,187] + mem[20,187] >= 0
 accuracy_(1,_sk_62)[21]: - 1.69921875 frac[21,187] + mem[21,187] >= 0
 accuracy_(1,_sk_62)[22]: - 1.69921875 frac[22,187] + mem[22,187] >= 0
 accuracy_(2,_sk_62)[0]: - 1.69921875 frac[0,188] + mem[0,188] >= 0
 accuracy_(2,_sk_62)[1]: - 1.69921875 frac[1,188] + mem[1,188] >= 0
 accuracy_(2,_sk_62)[2]: - 1.69921875 frac[2,188] + mem[2,188] >= 0
 accuracy_(2,_sk_62)[3]: - 1.69921875 frac[3,188] + mem[3,188] >= 0
 accuracy_(2,_sk_62)[4]: - 1.69921875 frac[4,188] + mem[4,188] >= 0
 accuracy_(2,_sk_62)[5]: - 1.69921875 frac[5,188] + mem[5,188] >= 0
 accuracy_(2,_sk_62)[6]: - 1.69921875 frac[6,188] + mem[6,188] >= 0
 accuracy_(2,_sk_62)[7]: - 1.69921875 frac[7,188] + mem[7,188] >= 0
 accuracy_(2,_sk_62)[8]: - 1.69921875 frac[8,188] + mem[8,188] >= 0
 accuracy_(2,_sk_62)[9]: - 1.69921875 frac[9,188] + mem[9,188] >= 0
 accuracy_(2,_sk_62)[10]: - 1.69921875 frac[10,188] + mem[10,188] >= 0
 accuracy_(2,_sk_62)[11]: - 1.69921875 frac[11,188] + mem[11,188] >= 0
 accuracy_(2,_sk_62)[12]: - 1.69921875 frac[12,188] + mem[12,188] >= 0
 accuracy_(2,_sk_62)[13]: - 1.69921875 frac[13,188] + mem[13,188] >= 0
 accuracy_(2,_sk_62)[14]: - 1.69921875 frac[14,188] + mem[14,188] >= 0
 accuracy_(2,_sk_62)[15]: - 1.69921875 frac[15,188] + mem[15,188] >= 0
 accuracy_(2,_sk_62)[16]: - 1.69921875 frac[16,188] + mem[16,188] >= 0
 accuracy_(2,_sk_62)[17]: - 1.69921875 frac[17,188] + mem[17,188] >= 0
 accuracy_(2,_sk_62)[18]: - 1.69921875 frac[18,188] + mem[18,188] >= 0
 accuracy_(2,_sk_62)[19]: - 1.69921875 frac[19,188] + mem[19,188] >= 0
 accuracy_(2,_sk_62)[20]: - 1.69921875 frac[20,188] + mem[20,188] >= 0
 accuracy_(2,_sk_62)[21]: - 1.69921875 frac[21,188] + mem[21,188] >= 0
 accuracy_(2,_sk_62)[22]: - 1.69921875 frac[22,188] + mem[22,188] >= 0
 accuracy_(0,_sk_63)[0]: - 1.69921875 frac[0,189] + mem[0,189] >= 0
 accuracy_(0,_sk_63)[1]: - 1.69921875 frac[1,189] + mem[1,189] >= 0
 accuracy_(0,_sk_63)[2]: - 1.69921875 frac[2,189] + mem[2,189] >= 0
 accuracy_(0,_sk_63)[3]: - 1.69921875 frac[3,189] + mem[3,189] >= 0
 accuracy_(0,_sk_63)[4]: - 1.69921875 frac[4,189] + mem[4,189] >= 0
 accuracy_(0,_sk_63)[5]: - 1.69921875 frac[5,189] + mem[5,189] >= 0
 accuracy_(0,_sk_63)[6]: - 1.69921875 frac[6,189] + mem[6,189] >= 0
 accuracy_(0,_sk_63)[7]: - 1.69921875 frac[7,189] + mem[7,189] >= 0
 accuracy_(0,_sk_63)[8]: - 1.69921875 frac[8,189] + mem[8,189] >= 0
 accuracy_(0,_sk_63)[9]: - 1.69921875 frac[9,189] + mem[9,189] >= 0
 accuracy_(0,_sk_63)[10]: - 1.69921875 frac[10,189] + mem[10,189] >= 0
 accuracy_(0,_sk_63)[11]: - 1.69921875 frac[11,189] + mem[11,189] >= 0
 accuracy_(0,_sk_63)[12]: - 1.69921875 frac[12,189] + mem[12,189] >= 0
 accuracy_(0,_sk_63)[13]: - 1.69921875 frac[13,189] + mem[13,189] >= 0
 accuracy_(0,_sk_63)[14]: - 1.69921875 frac[14,189] + mem[14,189] >= 0
 accuracy_(0,_sk_63)[15]: - 1.69921875 frac[15,189] + mem[15,189] >= 0
 accuracy_(0,_sk_63)[16]: - 1.69921875 frac[16,189] + mem[16,189] >= 0
 accuracy_(0,_sk_63)[17]: - 1.69921875 frac[17,189] + mem[17,189] >= 0
 accuracy_(0,_sk_63)[18]: - 1.69921875 frac[18,189] + mem[18,189] >= 0
 accuracy_(0,_sk_63)[19]: - 1.69921875 frac[19,189] + mem[19,189] >= 0
 accuracy_(0,_sk_63)[20]: - 1.69921875 frac[20,189] + mem[20,189] >= 0
 accuracy_(0,_sk_63)[21]: - 1.69921875 frac[21,189] + mem[21,189] >= 0
 accuracy_(0,_sk_63)[22]: - 1.69921875 frac[22,189] + mem[22,189] >= 0
 accuracy_(1,_sk_63)[0]: - 1.69921875 frac[0,190] + mem[0,190] >= 0
 accuracy_(1,_sk_63)[1]: - 1.69921875 frac[1,190] + mem[1,190] >= 0
 accuracy_(1,_sk_63)[2]: - 1.69921875 frac[2,190] + mem[2,190] >= 0
 accuracy_(1,_sk_63)[3]: - 1.69921875 frac[3,190] + mem[3,190] >= 0
 accuracy_(1,_sk_63)[4]: - 1.69921875 frac[4,190] + mem[4,190] >= 0
 accuracy_(1,_sk_63)[5]: - 1.69921875 frac[5,190] + mem[5,190] >= 0
 accuracy_(1,_sk_63)[6]: - 1.69921875 frac[6,190] + mem[6,190] >= 0
 accuracy_(1,_sk_63)[7]: - 1.69921875 frac[7,190] + mem[7,190] >= 0
 accuracy_(1,_sk_63)[8]: - 1.69921875 frac[8,190] + mem[8,190] >= 0
 accuracy_(1,_sk_63)[9]: - 1.69921875 frac[9,190] + mem[9,190] >= 0
 accuracy_(1,_sk_63)[10]: - 1.69921875 frac[10,190] + mem[10,190] >= 0
 accuracy_(1,_sk_63)[11]: - 1.69921875 frac[11,190] + mem[11,190] >= 0
 accuracy_(1,_sk_63)[12]: - 1.69921875 frac[12,190] + mem[12,190] >= 0
 accuracy_(1,_sk_63)[13]: - 1.69921875 frac[13,190] + mem[13,190] >= 0
 accuracy_(1,_sk_63)[14]: - 1.69921875 frac[14,190] + mem[14,190] >= 0
 accuracy_(1,_sk_63)[15]: - 1.69921875 frac[15,190] + mem[15,190] >= 0
 accuracy_(1,_sk_63)[16]: - 1.69921875 frac[16,190] + mem[16,190] >= 0
 accuracy_(1,_sk_63)[17]: - 1.69921875 frac[17,190] + mem[17,190] >= 0
 accuracy_(1,_sk_63)[18]: - 1.69921875 frac[18,190] + mem[18,190] >= 0
 accuracy_(1,_sk_63)[19]: - 1.69921875 frac[19,190] + mem[19,190] >= 0
 accuracy_(1,_sk_63)[20]: - 1.69921875 frac[20,190] + mem[20,190] >= 0
 accuracy_(1,_sk_63)[21]: - 1.69921875 frac[21,190] + mem[21,190] >= 0
 accuracy_(1,_sk_63)[22]: - 1.69921875 frac[22,190] + mem[22,190] >= 0
 accuracy_(2,_sk_63)[0]: - 1.69921875 frac[0,191] + mem[0,191] >= 0
 accuracy_(2,_sk_63)[1]: - 1.69921875 frac[1,191] + mem[1,191] >= 0
 accuracy_(2,_sk_63)[2]: - 1.69921875 frac[2,191] + mem[2,191] >= 0
 accuracy_(2,_sk_63)[3]: - 1.69921875 frac[3,191] + mem[3,191] >= 0
 accuracy_(2,_sk_63)[4]: - 1.69921875 frac[4,191] + mem[4,191] >= 0
 accuracy_(2,_sk_63)[5]: - 1.69921875 frac[5,191] + mem[5,191] >= 0
 accuracy_(2,_sk_63)[6]: - 1.69921875 frac[6,191] + mem[6,191] >= 0
 accuracy_(2,_sk_63)[7]: - 1.69921875 frac[7,191] + mem[7,191] >= 0
 accuracy_(2,_sk_63)[8]: - 1.69921875 frac[8,191] + mem[8,191] >= 0
 accuracy_(2,_sk_63)[9]: - 1.69921875 frac[9,191] + mem[9,191] >= 0
 accuracy_(2,_sk_63)[10]: - 1.69921875 frac[10,191] + mem[10,191] >= 0
 accuracy_(2,_sk_63)[11]: - 1.69921875 frac[11,191] + mem[11,191] >= 0
 accuracy_(2,_sk_63)[12]: - 1.69921875 frac[12,191] + mem[12,191] >= 0
 accuracy_(2,_sk_63)[13]: - 1.69921875 frac[13,191] + mem[13,191] >= 0
 accuracy_(2,_sk_63)[14]: - 1.69921875 frac[14,191] + mem[14,191] >= 0
 accuracy_(2,_sk_63)[15]: - 1.69921875 frac[15,191] + mem[15,191] >= 0
 accuracy_(2,_sk_63)[16]: - 1.69921875 frac[16,191] + mem[16,191] >= 0
 accuracy_(2,_sk_63)[17]: - 1.69921875 frac[17,191] + mem[17,191] >= 0
 accuracy_(2,_sk_63)[18]: - 1.69921875 frac[18,191] + mem[18,191] >= 0
 accuracy_(2,_sk_63)[19]: - 1.69921875 frac[19,191] + mem[19,191] >= 0
 accuracy_(2,_sk_63)[20]: - 1.69921875 frac[20,191] + mem[20,191] >= 0
 accuracy_(2,_sk_63)[21]: - 1.69921875 frac[21,191] + mem[21,191] >= 0
 accuracy_(2,_sk_63)[22]: - 1.69921875 frac[22,191] + mem[22,191] >= 0
 rows_tot_cpu1: - frac[0,0] - frac[0,1] - frac[0,2] - frac[0,3] - frac[0,4]
   - frac[0,5] - frac[0,6] - frac[0,7] - frac[0,8] - frac[0,9] - frac[0,10]
   - frac[0,11] - frac[0,12] - frac[0,13] - frac[0,14] - frac[0,15]
   - frac[0,16] - frac[0,17] - frac[0,18] - frac[0,19] - frac[0,20]
   - frac[0,21] - frac[0,22] - frac[0,23] - frac[0,24] - frac[0,25]
   - frac[0,26] - frac[0,27] - frac[0,28] - frac[0,29] - frac[0,30]
   - frac[0,31] - frac[0,32] - frac[0,33] - frac[0,34] - frac[0,35]
   - frac[0,36] - frac[0,37] - frac[0,38] - frac[0,39] - frac[0,40]
   - frac[0,41] - frac[0,42] - frac[0,43] - frac[0,44] - frac[0,45]
   - frac[0,46] - frac[0,47] - frac[0,48] - frac[0,49] - frac[0,50]
   - frac[0,51] - frac[0,52] - frac[0,53] - frac[0,54] - frac[0,55]
   - frac[0,56] - frac[0,57] - frac[0,58] - frac[0,59] - frac[0,60]
   - frac[0,61] - frac[0,62] - frac[0,63] - frac[0,64] - frac[0,65]
   - frac[0,66] - frac[0,67] - frac[0,68] - frac[0,69] - frac[0,70]
   - frac[0,71] - frac[0,72] - frac[0,73] - frac[0,74] - frac[0,75]
   - frac[0,76] - frac[0,77] - frac[0,78] - frac[0,79] - frac[0,80]
   - frac[0,81] - frac[0,82] - frac[0,83] - frac[0,84] - frac[0,85]
   - frac[0,86] - frac[0,87] - frac[0,88] - frac[0,89] - frac[0,90]
   - frac[0,91] - frac[0,92] - frac[0,93] - frac[0,94] - frac[0,95]
   - frac[0,96] - frac[0,97] - frac[0,98] - frac[0,99] - frac[0,100]
   - frac[0,101] - frac[0,102] - frac[0,103] - frac[0,104] - frac[0,105]
   - frac[0,106] - frac[0,107] - frac[0,108] - frac[0,109] - frac[0,110]
   - frac[0,111] - frac[0,112] - frac[0,113] - frac[0,114] - frac[0,115]
   - frac[0,116] - frac[0,117] - frac[0,118] - frac[0,119] - frac[0,120]
   - frac[0,121] - frac[0,122] - frac[0,123] - frac[0,124] - frac[0,125]
   - frac[0,126] - frac[0,127] - frac[0,128] - frac[0,129] - frac[0,130]
   - frac[0,131] - frac[0,132] - frac[0,133] - frac[0,134] - frac[0,135]
   - frac[0,136] - frac[0,137] - frac[0,138] - frac[0,139] - frac[0,140]
   - frac[0,141] - frac[0,142] - frac[0,143] - frac[0,144] - frac[0,145]
   - frac[0,146] - frac[0,147] - frac[0,148] - frac[0,149] - frac[0,150]
   - frac[0,151] - frac[0,152] - frac[0,153] - frac[0,154] - frac[0,155]
   - frac[0,156] - frac[0,157] - frac[0,158] - frac[0,159] - frac[0,160]
   - frac[0,161] - frac[0,162] - frac[0,163] - frac[0,164] - frac[0,165]
   - frac[0,166] - frac[0,167] - frac[0,168] - frac[0,169] - frac[0,170]
   - frac[0,171] - frac[0,172] - frac[0,173] - frac[0,174] - frac[0,175]
   - frac[0,176] - frac[0,177] - frac[0,178] - frac[0,179] - frac[0,180]
   - frac[0,181] - frac[0,182] - frac[0,183] - frac[0,184] - frac[0,185]
   - frac[0,186] - frac[0,187] - frac[0,188] - frac[0,189] - frac[0,190]
   - frac[0,191] + rows_tot_cpu1 = 0
 mem_tot_cpu1: - mem[0,0] - mem[0,1] - mem[0,2] - mem[0,3] - mem[0,4]
   - mem[0,5] - mem[0,6] - mem[0,7] - mem[0,8] - mem[0,9] - mem[0,10]
   - mem[0,11] - mem[0,12] - mem[0,13] - mem[0,14] - mem[0,15] - mem[0,16]
   - mem[0,17] - mem[0,18] - mem[0,19] - mem[0,20] - mem[0,21] - mem[0,22]
   - mem[0,23] - mem[0,24] - mem[0,25] - mem[0,26] - mem[0,27] - mem[0,28]
   - mem[0,29] - mem[0,30] - mem[0,31] - mem[0,32] - mem[0,33] - mem[0,34]
   - mem[0,35] - mem[0,36] - mem[0,37] - mem[0,38] - mem[0,39] - mem[0,40]
   - mem[0,41] - mem[0,42] - mem[0,43] - mem[0,44] - mem[0,45] - mem[0,46]
   - mem[0,47] - mem[0,48] - mem[0,49] - mem[0,50] - mem[0,51] - mem[0,52]
   - mem[0,53] - mem[0,54] - mem[0,55] - mem[0,56] - mem[0,57] - mem[0,58]
   - mem[0,59] - mem[0,60] - mem[0,61] - mem[0,62] - mem[0,63] - mem[0,64]
   - mem[0,65] - mem[0,66] - mem[0,67] - mem[0,68] - mem[0,69] - mem[0,70]
   - mem[0,71] - mem[0,72] - mem[0,73] - mem[0,74] - mem[0,75] - mem[0,76]
   - mem[0,77] - mem[0,78] - mem[0,79] - mem[0,80] - mem[0,81] - mem[0,82]
   - mem[0,83] - mem[0,84] - mem[0,85] - mem[0,86] - mem[0,87] - mem[0,88]
   - mem[0,89] - mem[0,90] - mem[0,91] - mem[0,92] - mem[0,93] - mem[0,94]
   - mem[0,95] - mem[0,96] - mem[0,97] - mem[0,98] - mem[0,99] - mem[0,100]
   - mem[0,101] - mem[0,102] - mem[0,103] - mem[0,104] - mem[0,105]
   - mem[0,106] - mem[0,107] - mem[0,108] - mem[0,109] - mem[0,110]
   - mem[0,111] - mem[0,112] - mem[0,113] - mem[0,114] - mem[0,115]
   - mem[0,116] - mem[0,117] - mem[0,118] - mem[0,119] - mem[0,120]
   - mem[0,121] - mem[0,122] - mem[0,123] - mem[0,124] - mem[0,125]
   - mem[0,126] - mem[0,127] - mem[0,128] - mem[0,129] - mem[0,130]
   - mem[0,131] - mem[0,132] - mem[0,133] - mem[0,134] - mem[0,135]
   - mem[0,136] - mem[0,137] - mem[0,138] - mem[0,139] - mem[0,140]
   - mem[0,141] - mem[0,142] - mem[0,143] - mem[0,144] - mem[0,145]
   - mem[0,146] - mem[0,147] - mem[0,148] - mem[0,149] - mem[0,150]
   - mem[0,151] - mem[0,152] - mem[0,153] - mem[0,154] - mem[0,155]
   - mem[0,156] - mem[0,157] - mem[0,158] - mem[0,159] - mem[0,160]
   - mem[0,161] - mem[0,162] - mem[0,163] - mem[0,164] - mem[0,165]
   - mem[0,166] - mem[0,167] - mem[0,168] - mem[0,169] - mem[0,170]
   - mem[0,171] - mem[0,172] - mem[0,173] - mem[0,174] - mem[0,175]
   - mem[0,176] - mem[0,177] - mem[0,178] - mem[0,179] - mem[0,180]
   - mem[0,181] - mem[0,182] - mem[0,183] - mem[0,184] - mem[0,185]
   - mem[0,186] - mem[0,187] - mem[0,188] - mem[0,189] - mem[0,190]
   - mem[0,191] + mem_tot_cpu1 = 0
 ns_single_cpu1: 3.5 rows_tot_cpu1 - ns_single_cpu1 + pdt_m_rows_cpu1 <= 0
 capacity_cores_cpu1: cores_sketch_cpu1 + cores_dpdk_cpu1 <= 7
 ns_sketch_cpu1: - ns_single_cpu1 + pdt_nsk_c_cpu1 = 0
 ns_dpdk_cpu1: 7.142857142857143 cores_dpdk_cpu1 - pdt_nsc_dpdk_cpu1
   = -21.42857142857143
 rows_tot_cpu2: - frac[1,0] - frac[1,1] - frac[1,2] - frac[1,3] - frac[1,4]
   - frac[1,5] - frac[1,6] - frac[1,7] - frac[1,8] - frac[1,9] - frac[1,10]
   - frac[1,11] - frac[1,12] - frac[1,13] - frac[1,14] - frac[1,15]
   - frac[1,16] - frac[1,17] - frac[1,18] - frac[1,19] - frac[1,20]
   - frac[1,21] - frac[1,22] - frac[1,23] - frac[1,24] - frac[1,25]
   - frac[1,26] - frac[1,27] - frac[1,28] - frac[1,29] - frac[1,30]
   - frac[1,31] - frac[1,32] - frac[1,33] - frac[1,34] - frac[1,35]
   - frac[1,36] - frac[1,37] - frac[1,38] - frac[1,39] - frac[1,40]
   - frac[1,41] - frac[1,42] - frac[1,43] - frac[1,44] - frac[1,45]
   - frac[1,46] - frac[1,47] - frac[1,48] - frac[1,49] - frac[1,50]
   - frac[1,51] - frac[1,52] - frac[1,53] - frac[1,54] - frac[1,55]
   - frac[1,56] - frac[1,57] - frac[1,58] - frac[1,59] - frac[1,60]
   - frac[1,61] - frac[1,62] - frac[1,63] - frac[1,64] - frac[1,65]
   - frac[1,66] - frac[1,67] - frac[1,68] - frac[1,69] - frac[1,70]
   - frac[1,71] - frac[1,72] - frac[1,73] - frac[1,74] - frac[1,75]
   - frac[1,76] - frac[1,77] - frac[1,78] - frac[1,79] - frac[1,80]
   - frac[1,81] - frac[1,82] - frac[1,83] - frac[1,84] - frac[1,85]
   - frac[1,86] - frac[1,87] - frac[1,88] - frac[1,89] - frac[1,90]
   - frac[1,91] - frac[1,92] - frac[1,93] - frac[1,94] - frac[1,95]
   - frac[1,96] - frac[1,97] - frac[1,98] - frac[1,99] - frac[1,100]
   - frac[1,101] - frac[1,102] - frac[1,103] - frac[1,104] - frac[1,105]
   - frac[1,106] - frac[1,107] - frac[1,108] - frac[1,109] - frac[1,110]
   - frac[1,111] - frac[1,112] - frac[1,113] - frac[1,114] - frac[1,115]
   - frac[1,116] - frac[1,117] - frac[1,118] - frac[1,119] - frac[1,120]
   - frac[1,121] - frac[1,122] - frac[1,123] - frac[1,124] - frac[1,125]
   - frac[1,126] - frac[1,127] - frac[1,128] - frac[1,129] - frac[1,130]
   - frac[1,131] - frac[1,132] - frac[1,133] - frac[1,134] - frac[1,135]
   - frac[1,136] - frac[1,137] - frac[1,138] - frac[1,139] - frac[1,140]
   - frac[1,141] - frac[1,142] - frac[1,143] - frac[1,144] - frac[1,145]
   - frac[1,146] - frac[1,147] - frac[1,148] - frac[1,149] - frac[1,150]
   - frac[1,151] - frac[1,152] - frac[1,153] - frac[1,154] - frac[1,155]
   - frac[1,156] - frac[1,157] - frac[1,158] - frac[1,159] - frac[1,160]
   - frac[1,161] - frac[1,162] - frac[1,163] - frac[1,164] - frac[1,165]
   - frac[1,166] - frac[1,167] - frac[1,168] - frac[1,169] - frac[1,170]
   - frac[1,171] - frac[1,172] - frac[1,173] - frac[1,174] - frac[1,175]
   - frac[1,176] - frac[1,177] - frac[1,178] - frac[1,179] - frac[1,180]
   - frac[1,181] - frac[1,182] - frac[1,183] - frac[1,184] - frac[1,185]
   - frac[1,186] - frac[1,187] - frac[1,188] - frac[1,189] - frac[1,190]
   - frac[1,191] + rows_tot_cpu2 = 0
 mem_tot_cpu2: - mem[1,0] - mem[1,1] - mem[1,2] - mem[1,3] - mem[1,4]
   - mem[1,5] - mem[1,6] - mem[1,7] - mem[1,8] - mem[1,9] - mem[1,10]
   - mem[1,11] - mem[1,12] - mem[1,13] - mem[1,14] - mem[1,15] - mem[1,16]
   - mem[1,17] - mem[1,18] - mem[1,19] - mem[1,20] - mem[1,21] - mem[1,22]
   - mem[1,23] - mem[1,24] - mem[1,25] - mem[1,26] - mem[1,27] - mem[1,28]
   - mem[1,29] - mem[1,30] - mem[1,31] - mem[1,32] - mem[1,33] - mem[1,34]
   - mem[1,35] - mem[1,36] - mem[1,37] - mem[1,38] - mem[1,39] - mem[1,40]
   - mem[1,41] - mem[1,42] - mem[1,43] - mem[1,44] - mem[1,45] - mem[1,46]
   - mem[1,47] - mem[1,48] - mem[1,49] - mem[1,50] - mem[1,51] - mem[1,52]
   - mem[1,53] - mem[1,54] - mem[1,55] - mem[1,56] - mem[1,57] - mem[1,58]
   - mem[1,59] - mem[1,60] - mem[1,61] - mem[1,62] - mem[1,63] - mem[1,64]
   - mem[1,65] - mem[1,66] - mem[1,67] - mem[1,68] - mem[1,69] - mem[1,70]
   - mem[1,71] - mem[1,72] - mem[1,73] - mem[1,74] - mem[1,75] - mem[1,76]
   - mem[1,77] - mem[1,78] - mem[1,79] - mem[1,80] - mem[1,81] - mem[1,82]
   - mem[1,83] - mem[1,84] - mem[1,85] - mem[1,86] - mem[1,87] - mem[1,88]
   - mem[1,89] - mem[1,90] - mem[1,91] - mem[1,92] - mem[1,93] - mem[1,94]
   - mem[1,95] - mem[1,96] - mem[1,97] - mem[1,98] - mem[1,99] - mem[1,100]
   - mem[1,101] - mem[1,102] - mem[1,103] - mem[1,104] - mem[1,105]
   - mem[1,106] - mem[1,107] - mem[1,108] - mem[1,109] - mem[1,110]
   - mem[1,111] - mem[1,112] - mem[1,113] - mem[1,114] - mem[1,115]
   - mem[1,116] - mem[1,117] - mem[1,118] - mem[1,119] - mem[1,120]
   - mem[1,121] - mem[1,122] - mem[1,123] - mem[1,124] - mem[1,125]
   - mem[1,126] - mem[1,127] - mem[1,128] - mem[1,129] - mem[1,130]
   - mem[1,131] - mem[1,132] - mem[1,133] - mem[1,134] - mem[1,135]
   - mem[1,136] - mem[1,137] - mem[1,138] - mem[1,139] - mem[1,140]
   - mem[1,141] - mem[1,142] - mem[1,143] - mem[1,144] - mem[1,145]
   - mem[1,146] - mem[1,147] - mem[1,148] - mem[1,149] - mem[1,150]
   - mem[1,151] - mem[1,152] - mem[1,153] - mem[1,154] - mem[1,155]
   - mem[1,156] - mem[1,157] - mem[1,158] - mem[1,159] - mem[1,160]
   - mem[1,161] - mem[1,162] - mem[1,163] - mem[1,164] - mem[1,165]
   - mem[1,166] - mem[1,167] - mem[1,168] - mem[1,169] - mem[1,170]
   - mem[1,171] - mem[1,172] - mem[1,173] - mem[1,174] - mem[1,175]
   - mem[1,176] - mem[1,177] - mem[1,178] - mem[1,179] - mem[1,180]
   - mem[1,181] - mem[1,182] - mem[1,183] - mem[1,184] - mem[1,185]
   - mem[1,186] - mem[1,187] - mem[1,188] - mem[1,189] - mem[1,190]
   - mem[1,191] + mem_tot_cpu2 = 0
 ns_single_cpu2: 3.5 rows_tot_cpu2 - ns_single_cpu2 + pdt_m_rows_cpu2 <= 0
 capacity_cores_cpu2: cores_sketch_cpu2 + cores_dpdk_cpu2 <= 7
 ns_sketch_cpu2: - ns_single_cpu2 + pdt_nsk_c_cpu2 = 0
 ns_dpdk_cpu2: 7.142857142857143 cores_dpdk_cpu2 - pdt_nsc_dpdk_cpu2
   = -21.42857142857143
 rows_tot_cpu3: - frac[2,0] - frac[2,1] - frac[2,2] - frac[2,3] - frac[2,4]
   - frac[2,5] - frac[2,6] - frac[2,7] - frac[2,8] - frac[2,9] - frac[2,10]
   - frac[2,11] - frac[2,12] - frac[2,13] - frac[2,14] - frac[2,15]
   - frac[2,16] - frac[2,17] - frac[2,18] - frac[2,19] - frac[2,20]
   - frac[2,21] - frac[2,22] - frac[2,23] - frac[2,24] - frac[2,25]
   - frac[2,26] - frac[2,27] - frac[2,28] - frac[2,29] - frac[2,30]
   - frac[2,31] - frac[2,32] - frac[2,33] - frac[2,34] - frac[2,35]
   - frac[2,36] - frac[2,37] - frac[2,38] - frac[2,39] - frac[2,40]
   - frac[2,41] - frac[2,42] - frac[2,43] - frac[2,44] - frac[2,45]
   - frac[2,46] - frac[2,47] - frac[2,48] - frac[2,49] - frac[2,50]
   - frac[2,51] - frac[2,52] - frac[2,53] - frac[2,54] - frac[2,55]
   - frac[2,56] - frac[2,57] - frac[2,58] - frac[2,59] - frac[2,60]
   - frac[2,61] - frac[2,62] - frac[2,63] - frac[2,64] - frac[2,65]
   - frac[2,66] - frac[2,67] - frac[2,68] - frac[2,69] - frac[2,70]
   - frac[2,71] - frac[2,72] - frac[2,73] - frac[2,74] - frac[2,75]
   - frac[2,76] - frac[2,77] - frac[2,78] - frac[2,79] - frac[2,80]
   - frac[2,81] - frac[2,82] - frac[2,83] - frac[2,84] - frac[2,85]
   - frac[2,86] - frac[2,87] - frac[2,88] - frac[2,89] - frac[2,90]
   - frac[2,91] - frac[2,92] - frac[2,93] - frac[2,94] - frac[2,95]
   - frac[2,96] - frac[2,97] - frac[2,98] - frac[2,99] - frac[2,100]
   - frac[2,101] - frac[2,102] - frac[2,103] - frac[2,104] - frac[2,105]
   - frac[2,106] - frac[2,107] - frac[2,108] - frac[2,109] - frac[2,110]
   - frac[2,111] - frac[2,112] - frac[2,113] - frac[2,114] - frac[2,115]
   - frac[2,116] - frac[2,117] - frac[2,118] - frac[2,119] - frac[2,120]
   - frac[2,121] - frac[2,122] - frac[2,123] - frac[2,124] - frac[2,125]
   - frac[2,126] - frac[2,127] - frac[2,128] - frac[2,129] - frac[2,130]
   - frac[2,131] - frac[2,132] - frac[2,133] - frac[2,134] - frac[2,135]
   - frac[2,136] - frac[2,137] - frac[2,138] - frac[2,139] - frac[2,140]
   - frac[2,141] - frac[2,142] - frac[2,143] - frac[2,144] - frac[2,145]
   - frac[2,146] - frac[2,147] - frac[2,148] - frac[2,149] - frac[2,150]
   - frac[2,151] - frac[2,152] - frac[2,153] - frac[2,154] - frac[2,155]
   - frac[2,156] - frac[2,157] - frac[2,158] - frac[2,159] - frac[2,160]
   - frac[2,161] - frac[2,162] - frac[2,163] - frac[2,164] - frac[2,165]
   - frac[2,166] - frac[2,167] - frac[2,168] - frac[2,169] - frac[2,170]
   - frac[2,171] - frac[2,172] - frac[2,173] - frac[2,174] - frac[2,175]
   - frac[2,176] - frac[2,177] - frac[2,178] - frac[2,179] - frac[2,180]
   - frac[2,181] - frac[2,182] - frac[2,183] - frac[2,184] - frac[2,185]
   - frac[2,186] - frac[2,187] - frac[2,188] - frac[2,189] - frac[2,190]
   - frac[2,191] + rows_tot_cpu3 = 0
 mem_tot_cpu3: - mem[2,0] - mem[2,1] - mem[2,2] - mem[2,3] - mem[2,4]
   - mem[2,5] - mem[2,6] - mem[2,7] - mem[2,8] - mem[2,9] - mem[2,10]
   - mem[2,11] - mem[2,12] - mem[2,13] - mem[2,14] - mem[2,15] - mem[2,16]
   - mem[2,17] - mem[2,18] - mem[2,19] - mem[2,20] - mem[2,21] - mem[2,22]
   - mem[2,23] - mem[2,24] - mem[2,25] - mem[2,26] - mem[2,27] - mem[2,28]
   - mem[2,29] - mem[2,30] - mem[2,31] - mem[2,32] - mem[2,33] - mem[2,34]
   - mem[2,35] - mem[2,36] - mem[2,37] - mem[2,38] - mem[2,39] - mem[2,40]
   - mem[2,41] - mem[2,42] - mem[2,43] - mem[2,44] - mem[2,45] - mem[2,46]
   - mem[2,47] - mem[2,48] - mem[2,49] - mem[2,50] - mem[2,51] - mem[2,52]
   - mem[2,53] - mem[2,54] - mem[2,55] - mem[2,56] - mem[2,57] - mem[2,58]
   - mem[2,59] - mem[2,60] - mem[2,61] - mem[2,62] - mem[2,63] - mem[2,64]
   - mem[2,65] - mem[2,66] - mem[2,67] - mem[2,68] - mem[2,69] - mem[2,70]
   - mem[2,71] - mem[2,72] - mem[2,73] - mem[2,74] - mem[2,75] - mem[2,76]
   - mem[2,77] - mem[2,78] - mem[2,79] - mem[2,80] - mem[2,81] - mem[2,82]
   - mem[2,83] - mem[2,84] - mem[2,85] - mem[2,86] - mem[2,87] - mem[2,88]
   - mem[2,89] - mem[2,90] - mem[2,91] - mem[2,92] - mem[2,93] - mem[2,94]
   - mem[2,95] - mem[2,96] - mem[2,97] - mem[2,98] - mem[2,99] - mem[2,100]
   - mem[2,101] - mem[2,102] - mem[2,103] - mem[2,104] - mem[2,105]
   - mem[2,106] - mem[2,107] - mem[2,108] - mem[2,109] - mem[2,110]
   - mem[2,111] - mem[2,112] - mem[2,113] - mem[2,114] - mem[2,115]
   - mem[2,116] - mem[2,117] - mem[2,118] - mem[2,119] - mem[2,120]
   - mem[2,121] - mem[2,122] - mem[2,123] - mem[2,124] - mem[2,125]
   - mem[2,126] - mem[2,127] - mem[2,128] - mem[2,129] - mem[2,130]
   - mem[2,131] - mem[2,132] - mem[2,133] - mem[2,134] - mem[2,135]
   - mem[2,136] - mem[2,137] - mem[2,138] - mem[2,139] - mem[2,140]
   - mem[2,141] - mem[2,142] - mem[2,143] - mem[2,144] - mem[2,145]
   - mem[2,146] - mem[2,147] - mem[2,148] - mem[2,149] - mem[2,150]
   - mem[2,151] - mem[2,152] - mem[2,153] - mem[2,154] - mem[2,155]
   - mem[2,156] - mem[2,157] - mem[2,158] - mem[2,159] - mem[2,160]
   - mem[2,161] - mem[2,162] - mem[2,163] - mem[2,164] - mem[2,165]
   - mem[2,166] - mem[2,167] - mem[2,168] - mem[2,169] - mem[2,170]
   - mem[2,171] - mem[2,172] - mem[2,173] - mem[2,174] - mem[2,175]
   - mem[2,176] - mem[2,177] - mem[2,178] - mem[2,179] - mem[2,180]
   - mem[2,181] - mem[2,182] - mem[2,183] - mem[2,184] - mem[2,185]
   - mem[2,186] - mem[2,187] - mem[2,188] - mem[2,189] - mem[2,190]
   - mem[2,191] + mem_tot_cpu3 = 0
 ns_single_cpu3: 3.5 rows_tot_cpu3 - ns_single_cpu3 + pdt_m_rows_cpu3 <= 0
 capacity_cores_cpu3: cores_sketch_cpu3 + cores_dpdk_cpu3 <= 7
 ns_sketch_cpu3: - ns_single_cpu3 + pdt_nsk_c_cpu3 = 0
 ns_dpdk_cpu3: 7.142857142857143 cores_dpdk_cpu3 - pdt_nsc_dpdk_cpu3
   = -21.42857142857143
 rows_tot_cpu4: - frac[3,0] - frac[3,1] - frac[3,2] - frac[3,3] - frac[3,4]
   - frac[3,5] - frac[3,6] - frac[3,7] - frac[3,8] - frac[3,9] - frac[3,10]
   - frac[3,11] - frac[3,12] - frac[3,13] - frac[3,14] - frac[3,15]
   - frac[3,16] - frac[3,17] - frac[3,18] - frac[3,19] - frac[3,20]
   - frac[3,21] - frac[3,22] - frac[3,23] - frac[3,24] - frac[3,25]
   - frac[3,26] - frac[3,27] - frac[3,28] - frac[3,29] - frac[3,30]
   - frac[3,31] - frac[3,32] - frac[3,33] - frac[3,34] - frac[3,35]
   - frac[3,36] - frac[3,37] - frac[3,38] - frac[3,39] - frac[3,40]
   - frac[3,41] - frac[3,42] - frac[3,43] - frac[3,44] - frac[3,45]
   - frac[3,46] - frac[3,47] - frac[3,48] - frac[3,49] - frac[3,50]
   - frac[3,51] - frac[3,52] - frac[3,53] - frac[3,54] - frac[3,55]
   - frac[3,56] - frac[3,57] - frac[3,58] - frac[3,59] - frac[3,60]
   - frac[3,61] - frac[3,62] - frac[3,63] - frac[3,64] - frac[3,65]
   - frac[3,66] - frac[3,67] - frac[3,68] - frac[3,69] - frac[3,70]
   - frac[3,71] - frac[3,72] - frac[3,73] - frac[3,74] - frac[3,75]
   - frac[3,76] - frac[3,77] - frac[3,78] - frac[3,79] - frac[3,80]
   - frac[3,81] - frac[3,82] - frac[3,83] - frac[3,84] - frac[3,85]
   - frac[3,86] - frac[3,87] - frac[3,88] - frac[3,89] - frac[3,90]
   - frac[3,91] - frac[3,92] - frac[3,93] - frac[3,94] - frac[3,95]
   - frac[3,96] - frac[3,97] - frac[3,98] - frac[3,99] - frac[3,100]
   - frac[3,101] - frac[3,102] - frac[3,103] - frac[3,104] - frac[3,105]
   - frac[3,106] - frac[3,107] - frac[3,108] - frac[3,109] - frac[3,110]
   - frac[3,111] - frac[3,112] - frac[3,113] - frac[3,114] - frac[3,115]
   - frac[3,116] - frac[3,117] - frac[3,118] - frac[3,119] - frac[3,120]
   - frac[3,121] - frac[3,122] - frac[3,123] - frac[3,124] - frac[3,125]
   - frac[3,126] - frac[3,127] - frac[3,128] - frac[3,129] - frac[3,130]
   - frac[3,131] - frac[3,132] - frac[3,133] - frac[3,134] - frac[3,135]
   - frac[3,136] - frac[3,137] - frac[3,138] - frac[3,139] - frac[3,140]
   - frac[3,141] - frac[3,142] - frac[3,143] - frac[3,144] - frac[3,145]
   - frac[3,146] - frac[3,147] - frac[3,148] - frac[3,149] - frac[3,150]
   - frac[3,151] - frac[3,152] - frac[3,153] - frac[3,154] - frac[3,155]
   - frac[3,156] - frac[3,157] - frac[3,158] - frac[3,159] - frac[3,160]
   - frac[3,161] - frac[3,162] - frac[3,163] - frac[3,164] - frac[3,165]
   - frac[3,166] - frac[3,167] - frac[3,168] - frac[3,169] - frac[3,170]
   - frac[3,171] - frac[3,172] - frac[3,173] - frac[3,174] - frac[3,175]
   - frac[3,176] - frac[3,177] - frac[3,178] - frac[3,179] - frac[3,180]
   - frac[3,181] - frac[3,182] - frac[3,183] - frac[3,184] - frac[3,185]
   - frac[3,186] - frac[3,187] - frac[3,188] - frac[3,189] - frac[3,190]
   - frac[3,191] + rows_tot_cpu4 = 0
 mem_tot_cpu4: - mem[3,0] - mem[3,1] - mem[3,2] - mem[3,3] - mem[3,4]
   - mem[3,5] - mem[3,6] - mem[3,7] - mem[3,8] - mem[3,9] - mem[3,10]
   - mem[3,11] - mem[3,12] - mem[3,13] - mem[3,14] - mem[3,15] - mem[3,16]
   - mem[3,17] - mem[3,18] - mem[3,19] - mem[3,20] - mem[3,21] - mem[3,22]
   - mem[3,23] - mem[3,24] - mem[3,25] - mem[3,26] - mem[3,27] - mem[3,28]
   - mem[3,29] - mem[3,30] - mem[3,31] - mem[3,32] - mem[3,33] - mem[3,34]
   - mem[3,35] - mem[3,36] - mem[3,37] - mem[3,38] - mem[3,39] - mem[3,40]
   - mem[3,41] - mem[3,42] - mem[3,43] - mem[3,44] - mem[3,45] - mem[3,46]
   - mem[3,47] - mem[3,48] - mem[3,49] - mem[3,50] - mem[3,51] - mem[3,52]
   - mem[3,53] - mem[3,54] - mem[3,55] - mem[3,56] - mem[3,57] - mem[3,58]
   - mem[3,59] - mem[3,60] - mem[3,61] - mem[3,62] - mem[3,63] - mem[3,64]
   - mem[3,65] - mem[3,66] - mem[3,67] - mem[3,68] - mem[3,69] - mem[3,70]
   - mem[3,71] - mem[3,72] - mem[3,73] - mem[3,74] - mem[3,75] - mem[3,76]
   - mem[3,77] - mem[3,78] - mem[3,79] - mem[3,80] - mem[3,81] - mem[3,82]
   - mem[3,83] - mem[3,84] - mem[3,85] - mem[3,86] - mem[3,87] - mem[3,88]
   - mem[3,89] - mem[3,90] - mem[3,91] - mem[3,92] - mem[3,93] - mem[3,94]
   - mem[3,95] - mem[3,96] - mem[3,97] - mem[3,98] - mem[3,99] - mem[3,100]
   - mem[3,101] - mem[3,102] - mem[3,103] - mem[3,104] - mem[3,105]
   - mem[3,106] - mem[3,107] - mem[3,108] - mem[3,109] - mem[3,110]
   - mem[3,111] - mem[3,112] - mem[3,113] - mem[3,114] - mem[3,115]
   - mem[3,116] - mem[3,117] - mem[3,118] - mem[3,119] - mem[3,120]
   - mem[3,121] - mem[3,122] - mem[3,123] - mem[3,124] - mem[3,125]
   - mem[3,126] - mem[3,127] - mem[3,128] - mem[3,129] - mem[3,130]
   - mem[3,131] - mem[3,132] - mem[3,133] - mem[3,134] - mem[3,135]
   - mem[3,136] - mem[3,137] - mem[3,138] - mem[3,139] - mem[3,140]
   - mem[3,141] - mem[3,142] - mem[3,143] - mem[3,144] - mem[3,145]
   - mem[3,146] - mem[3,147] - mem[3,148] - mem[3,149] - mem[3,150]
   - mem[3,151] - mem[3,152] - mem[3,153] - mem[3,154] - mem[3,155]
   - mem[3,156] - mem[3,157] - mem[3,158] - mem[3,159] - mem[3,160]
   - mem[3,161] - mem[3,162] - mem[3,163] - mem[3,164] - mem[3,165]
   - mem[3,166] - mem[3,167] - mem[3,168] - mem[3,169] - mem[3,170]
   - mem[3,171] - mem[3,172] - mem[3,173] - mem[3,174] - mem[3,175]
   - mem[3,176] - mem[3,177] - mem[3,178] - mem[3,179] - mem[3,180]
   - mem[3,181] - mem[3,182] - mem[3,183] - mem[3,184] - mem[3,185]
   - mem[3,186] - mem[3,187] - mem[3,188] - mem[3,189] - mem[3,190]
   - mem[3,191] + mem_tot_cpu4 = 0
 ns_single_cpu4: 3.5 rows_tot_cpu4 - ns_single_cpu4 + pdt_m_rows_cpu4 <= 0
 capacity_cores_cpu4: cores_sketch_cpu4 + cores_dpdk_cpu4 <= 7
 ns_sketch_cpu4: - ns_single_cpu4 + pdt_nsk_c_cpu4 = 0
 ns_dpdk_cpu4: 7.142857142857143 cores_dpdk_cpu4 - pdt_nsc_dpdk_cpu4
   = -21.42857142857143
 rows_tot_cpu5: - frac[4,0] - frac[4,1] - frac[4,2] - frac[4,3] - frac[4,4]
   - frac[4,5] - frac[4,6] - frac[4,7] - frac[4,8] - frac[4,9] - frac[4,10]
   - frac[4,11] - frac[4,12] - frac[4,13] - frac[4,14] - frac[4,15]
   - frac[4,16] - frac[4,17] - frac[4,18] - frac[4,19] - frac[4,20]
   - frac[4,21] - frac[4,22] - frac[4,23] - frac[4,24] - frac[4,25]
   - frac[4,26] - frac[4,27] - frac[4,28] - frac[4,29] - frac[4,30]
   - frac[4,31] - frac[4,32] - frac[4,33] - frac[4,34] - frac[4,35]
   - frac[4,36] - frac[4,37] - frac[4,38] - frac[4,39] - frac[4,40]
   - frac[4,41] - frac[4,42] - frac[4,43] - frac[4,44] - frac[4,45]
   - frac[4,46] - frac[4,47] - frac[4,48] - frac[4,49] - frac[4,50]
   - frac[4,51] - frac[4,52] - frac[4,53] - frac[4,54] - frac[4,55]
   - frac[4,56] - frac[4,57] - frac[4,58] - frac[4,59] - frac[4,60]
   - frac[4,61] - frac[4,62] - frac[4,63] - frac[4,64] - frac[4,65]
   - frac[4,66] - frac[4,67] - frac[4,68] - frac[4,69] - frac[4,70]
   - frac[4,71] - frac[4,72] - frac[4,73] - frac[4,74] - frac[4,75]
   - frac[4,76] - frac[4,77] - frac[4,78] - frac[4,79] - frac[4,80]
   - frac[4,81] - frac[4,82] - frac[4,83] - frac[4,84] - frac[4,85]
   - frac[4,86] - frac[4,87] - frac[4,88] - frac[4,89] - frac[4,90]
   - frac[4,91] - frac[4,92] - frac[4,93] - frac[4,94] - frac[4,95]
   - frac[4,96] - frac[4,97] - frac[4,98] - frac[4,99] - frac[4,100]
   - frac[4,101] - frac[4,102] - frac[4,103] - frac[4,104] - frac[4,105]
   - frac[4,106] - frac[4,107] - frac[4,108] - frac[4,109] - frac[4,110]
   - frac[4,111] - frac[4,112] - frac[4,113] - frac[4,114] - frac[4,115]
   - frac[4,116] - frac[4,117] - frac[4,118] - frac[4,119] - frac[4,120]
   - frac[4,121] - frac[4,122] - frac[4,123] - frac[4,124] - frac[4,125]
   - frac[4,126] - frac[4,127] - frac[4,128] - frac[4,129] - frac[4,130]
   - frac[4,131] - frac[4,132] - frac[4,133] - frac[4,134] - frac[4,135]
   - frac[4,136] - frac[4,137] - frac[4,138] - frac[4,139] - frac[4,140]
   - frac[4,141] - frac[4,142] - frac[4,143] - frac[4,144] - frac[4,145]
   - frac[4,146] - frac[4,147] - frac[4,148] - frac[4,149] - frac[4,150]
   - frac[4,151] - frac[4,152] - frac[4,153] - frac[4,154] - frac[4,155]
   - frac[4,156] - frac[4,157] - frac[4,158] - frac[4,159] - frac[4,160]
   - frac[4,161] - frac[4,162] - frac[4,163] - frac[4,164] - frac[4,165]
   - frac[4,166] - frac[4,167] - frac[4,168] - frac[4,169] - frac[4,170]
   - frac[4,171] - frac[4,172] - frac[4,173] - frac[4,174] - frac[4,175]
   - frac[4,176] - frac[4,177] - frac[4,178] - frac[4,179] - frac[4,180]
   - frac[4,181] - frac[4,182] - frac[4,183] - frac[4,184] - frac[4,185]
   - frac[4,186] - frac[4,187] - frac[4,188] - frac[4,189] - frac[4,190]
   - frac[4,191] + rows_tot_cpu5 = 0
 mem_tot_cpu5: - mem[4,0] - mem[4,1] - mem[4,2] - mem[4,3] - mem[4,4]
   - mem[4,5] - mem[4,6] - mem[4,7] - mem[4,8] - mem[4,9] - mem[4,10]
   - mem[4,11] - mem[4,12] - mem[4,13] - mem[4,14] - mem[4,15] - mem[4,16]
   - mem[4,17] - mem[4,18] - mem[4,19] - mem[4,20] - mem[4,21] - mem[4,22]
   - mem[4,23] - mem[4,24] - mem[4,25] - mem[4,26] - mem[4,27] - mem[4,28]
   - mem[4,29] - mem[4,30] - mem[4,31] - mem[4,32] - mem[4,33] - mem[4,34]
   - mem[4,35] - mem[4,36] - mem[4,37] - mem[4,38] - mem[4,39] - mem[4,40]
   - mem[4,41] - mem[4,42] - mem[4,43] - mem[4,44] - mem[4,45] - mem[4,46]
   - mem[4,47] - mem[4,48] - mem[4,49] - mem[4,50] - mem[4,51] - mem[4,52]
   - mem[4,53] - mem[4,54] - mem[4,55] - mem[4,56] - mem[4,57] - mem[4,58]
   - mem[4,59] - mem[4,60] - mem[4,61] - mem[4,62] - mem[4,63] - mem[4,64]
   - mem[4,65] - mem[4,66] - mem[4,67] - mem[4,68] - mem[4,69] - mem[4,70]
   - mem[4,71] - mem[4,72] - mem[4,73] - mem[4,74] - mem[4,75] - mem[4,76]
   - mem[4,77] - mem[4,78] - mem[4,79] - mem[4,80] - mem[4,81] - mem[4,82]
   - mem[4,83] - mem[4,84] - mem[4,85] - mem[4,86] - mem[4,87] - mem[4,88]
   - mem[4,89] - mem[4,90] - mem[4,91] - mem[4,92] - mem[4,93] - mem[4,94]
   - mem[4,95] - mem[4,96] - mem[4,97] - mem[4,98] - mem[4,99] - mem[4,100]
   - mem[4,101] - mem[4,102] - mem[4,103] - mem[4,104] - mem[4,105]
   - mem[4,106] - mem[4,107] - mem[4,108] - mem[4,109] - mem[4,110]
   - mem[4,111] - mem[4,112] - mem[4,113] - mem[4,114] - mem[4,115]
   - mem[4,116] - mem[4,117] - mem[4,118] - mem[4,119] - mem[4,120]
   - mem[4,121] - mem[4,122] - mem[4,123] - mem[4,124] - mem[4,125]
   - mem[4,126] - mem[4,127] - mem[4,128] - mem[4,129] - mem[4,130]
   - mem[4,131] - mem[4,132] - mem[4,133] - mem[4,134] - mem[4,135]
   - mem[4,136] - mem[4,137] - mem[4,138] - mem[4,139] - mem[4,140]
   - mem[4,141] - mem[4,142] - mem[4,143] - mem[4,144] - mem[4,145]
   - mem[4,146] - mem[4,147] - mem[4,148] - mem[4,149] - mem[4,150]
   - mem[4,151] - mem[4,152] - mem[4,153] - mem[4,154] - mem[4,155]
   - mem[4,156] - mem[4,157] - mem[4,158] - mem[4,159] - mem[4,160]
   - mem[4,161] - mem[4,162] - mem[4,163] - mem[4,164] - mem[4,165]
   - mem[4,166] - mem[4,167] - mem[4,168] - mem[4,169] - mem[4,170]
   - mem[4,171] - mem[4,172] - mem[4,173] - mem[4,174] - mem[4,175]
   - mem[4,176] - mem[4,177] - mem[4,178] - mem[4,179] - mem[4,180]
   - mem[4,181] - mem[4,182] - mem[4,183] - mem[4,184] - mem[4,185]
   - mem[4,186] - mem[4,187] - mem[4,188] - mem[4,189] - mem[4,190]
   - mem[4,191] + mem_tot_cpu5 = 0
 ns_single_cpu5: 3.5 rows_tot_cpu5 - ns_single_cpu5 + pdt_m_rows_cpu5 <= 0
 capacity_cores_cpu5: cores_sketch_cpu5 + cores_dpdk_cpu5 <= 7
 ns_sketch_cpu5: - ns_single_cpu5 + pdt_nsk_c_cpu5 = 0
 ns_dpdk_cpu5: 7.142857142857143 cores_dpdk_cpu5 - pdt_nsc_dpdk_cpu5
   = -21.42857142857143
 rows_tot_cpu6: - frac[5,0] - frac[5,1] - frac[5,2] - frac[5,3] - frac[5,4]
   - frac[5,5] - frac[5,6] - frac[5,7] - frac[5,8] - frac[5,9] - frac[5,10]
   - frac[5,11] - frac[5,12] - frac[5,13] - frac[5,14] - frac[5,15]
   - frac[5,16] - frac[5,17] - frac[5,18] - frac[5,19] - frac[5,20]
   - frac[5,21] - frac[5,22] - frac[5,23] - frac[5,24] - frac[5,25]
   - frac[5,26] - frac[5,27] - frac[5,28] - frac[5,29] - frac[5,30]
   - frac[5,31] - frac[5,32] - frac[5,33] - frac[5,34] - frac[5,35]
   - frac[5,36] - frac[5,37] - frac[5,38] - frac[5,39] - frac[5,40]
   - frac[5,41] - frac[5,42] - frac[5,43] - frac[5,44] - frac[5,45]
   - frac[5,46] - frac[5,47] - frac[5,48] - frac[5,49] - frac[5,50]
   - frac[5,51] - frac[5,52] - frac[5,53] - frac[5,54] - frac[5,55]
   - frac[5,56] - frac[5,57] - frac[5,58] - frac[5,59] - frac[5,60]
   - frac[5,61] - frac[5,62] - frac[5,63] - frac[5,64] - frac[5,65]
   - frac[5,66] - frac[5,67] - frac[5,68] - frac[5,69] - frac[5,70]
   - frac[5,71] - frac[5,72] - frac[5,73] - frac[5,74] - frac[5,75]
   - frac[5,76] - frac[5,77] - frac[5,78] - frac[5,79] - frac[5,80]
   - frac[5,81] - frac[5,82] - frac[5,83] - frac[5,84] - frac[5,85]
   - frac[5,86] - frac[5,87] - frac[5,88] - frac[5,89] - frac[5,90]
   - frac[5,91] - frac[5,92] - frac[5,93] - frac[5,94] - frac[5,95]
   - frac[5,96] - frac[5,97] - frac[5,98] - frac[5,99] - frac[5,100]
   - frac[5,101] - frac[5,102] - frac[5,103] - frac[5,104] - frac[5,105]
   - frac[5,106] - frac[5,107] - frac[5,108] - frac[5,109] - frac[5,110]
   - frac[5,111] - frac[5,112] - frac[5,113] - frac[5,114] - frac[5,115]
   - frac[5,116] - frac[5,117] - frac[5,118] - frac[5,119] - frac[5,120]
   - frac[5,121] - frac[5,122] - frac[5,123] - frac[5,124] - frac[5,125]
   - frac[5,126] - frac[5,127] - frac[5,128] - frac[5,129] - frac[5,130]
   - frac[5,131] - frac[5,132] - frac[5,133] - frac[5,134] - frac[5,135]
   - frac[5,136] - frac[5,137] - frac[5,138] - frac[5,139] - frac[5,140]
   - frac[5,141] - frac[5,142] - frac[5,143] - frac[5,144] - frac[5,145]
   - frac[5,146] - frac[5,147] - frac[5,148] - frac[5,149] - frac[5,150]
   - frac[5,151] - frac[5,152] - frac[5,153] - frac[5,154] - frac[5,155]
   - frac[5,156] - frac[5,157] - frac[5,158] - frac[5,159] - frac[5,160]
   - frac[5,161] - frac[5,162] - frac[5,163] - frac[5,164] - frac[5,165]
   - frac[5,166] - frac[5,167] - frac[5,168] - frac[5,169] - frac[5,170]
   - frac[5,171] - frac[5,172] - frac[5,173] - frac[5,174] - frac[5,175]
   - frac[5,176] - frac[5,177] - frac[5,178] - frac[5,179] - frac[5,180]
   - frac[5,181] - frac[5,182] - frac[5,183] - frac[5,184] - frac[5,185]
   - frac[5,186] - frac[5,187] - frac[5,188] - frac[5,189] - frac[5,190]
   - frac[5,191] + rows_tot_cpu6 = 0
 mem_tot_cpu6: - mem[5,0] - mem[5,1] - mem[5,2] - mem[5,3] - mem[5,4]
   - mem[5,5] - mem[5,6] - mem[5,7] - mem[5,8] - mem[5,9] - mem[5,10]
   - mem[5,11] - mem[5,12] - mem[5,13] - mem[5,14] - mem[5,15] - mem[5,16]
   - mem[5,17] - mem[5,18] - mem[5,19] - mem[5,20] - mem[5,21] - mem[5,22]
   - mem[5,23] - mem[5,24] - mem[5,25] - mem[5,26] - mem[5,27] - mem[5,28]
   - mem[5,29] - mem[5,30] - mem[5,31] - mem[5,32] - mem[5,33] - mem[5,34]
   - mem[5,35] - mem[5,36] - mem[5,37] - mem[5,38] - mem[5,39] - mem[5,40]
   - mem[5,41] - mem[5,42] - mem[5,43] - mem[5,44] - mem[5,45] - mem[5,46]
   - mem[5,47] - mem[5,48] - mem[5,49] - mem[5,50] - mem[5,51] - mem[5,52]
   - mem[5,53] - mem[5,54] - mem[5,55] - mem[5,56] - mem[5,57] - mem[5,58]
   - mem[5,59] - mem[5,60] - mem[5,61] - mem[5,62] - mem[5,63] - mem[5,64]
   - mem[5,65] - mem[5,66] - mem[5,67] - mem[5,68] - mem[5,69] - mem[5,70]
   - mem[5,71] - mem[5,72] - mem[5,73] - mem[5,74] - mem[5,75] - mem[5,76]
   - mem[5,77] - mem[5,78] - mem[5,79] - mem[5,80] - mem[5,81] - mem[5,82]
   - mem[5,83] - mem[5,84] - mem[5,85] - mem[5,86] - mem[5,87] - mem[5,88]
   - mem[5,89] - mem[5,90] - mem[5,91] - mem[5,92] - mem[5,93] - mem[5,94]
   - mem[5,95] - mem[5,96] - mem[5,97] - mem[5,98] - mem[5,99] - mem[5,100]
   - mem[5,101] - mem[5,102] - mem[5,103] - mem[5,104] - mem[5,105]
   - mem[5,106] - mem[5,107] - mem[5,108] - mem[5,109] - mem[5,110]
   - mem[5,111] - mem[5,112] - mem[5,113] - mem[5,114] - mem[5,115]
   - mem[5,116] - mem[5,117] - mem[5,118] - mem[5,119] - mem[5,120]
   - mem[5,121] - mem[5,122] - mem[5,123] - mem[5,124] - mem[5,125]
   - mem[5,126] - mem[5,127] - mem[5,128] - mem[5,129] - mem[5,130]
   - mem[5,131] - mem[5,132] - mem[5,133] - mem[5,134] - mem[5,135]
   - mem[5,136] - mem[5,137] - mem[5,138] - mem[5,139] - mem[5,140]
   - mem[5,141] - mem[5,142] - mem[5,143] - mem[5,144] - mem[5,145]
   - mem[5,146] - mem[5,147] - mem[5,148] - mem[5,149] - mem[5,150]
   - mem[5,151] - mem[5,152] - mem[5,153] - mem[5,154] - mem[5,155]
   - mem[5,156] - mem[5,157] - mem[5,158] - mem[5,159] - mem[5,160]
   - mem[5,161] - mem[5,162] - mem[5,163] - mem[5,164] - mem[5,165]
   - mem[5,166] - mem[5,167] - mem[5,168] - mem[5,169] - mem[5,170]
   - mem[5,171] - mem[5,172] - mem[5,173] - mem[5,174] - mem[5,175]
   - mem[5,176] - mem[5,177] - mem[5,178] - mem[5,179] - mem[5,180]
   - mem[5,181] - mem[5,182] - mem[5,183] - mem[5,184] - mem[5,185]
   - mem[5,186] - mem[5,187] - mem[5,188] - mem[5,189] - mem[5,190]
   - mem[5,191] + mem_tot_cpu6 = 0
 ns_single_cpu6: 3.5 rows_tot_cpu6 - ns_single_cpu6 + pdt_m_rows_cpu6 <= 0
 capacity_cores_cpu6: cores_sketch_cpu6 + cores_dpdk_cpu6 <= 7
 ns_sketch_cpu6: - ns_single_cpu6 + pdt_nsk_c_cpu6 = 0
 ns_dpdk_cpu6: 7.142857142857143 cores_dpdk_cpu6 - pdt_nsc_dpdk_cpu6
   = -21.42857142857143
 rows_tot_cpu7: - frac[6,0] - frac[6,1] - frac[6,2] - frac[6,3] - frac[6,4]
   - frac[6,5] - frac[6,6] - frac[6,7] - frac[6,8] - frac[6,9] - frac[6,10]
   - frac[6,11] - frac[6,12] - frac[6,13] - frac[6,14] - frac[6,15]
   - frac[6,16] - frac[6,17] - frac[6,18] - frac[6,19] - frac[6,20]
   - frac[6,21] - frac[6,22] - frac[6,23] - frac[6,24] - frac[6,25]
   - frac[6,26] - frac[6,27] - frac[6,28] - frac[6,29] - frac[6,30]
   - frac[6,31] - frac[6,32] - frac[6,33] - frac[6,34] - frac[6,35]
   - frac[6,36] - frac[6,37] - frac[6,38] - frac[6,39] - frac[6,40]
   - frac[6,41] - frac[6,42] - frac[6,43] - frac[6,44] - frac[6,45]
   - frac[6,46] - frac[6,47] - frac[6,48] - frac[6,49] - frac[6,50]
   - frac[6,51] - frac[6,52] - frac[6,53] - frac[6,54] - frac[6,55]
   - frac[6,56] - frac[6,57] - frac[6,58] - frac[6,59] - frac[6,60]
   - frac[6,61] - frac[6,62] - frac[6,63] - frac[6,64] - frac[6,65]
   - frac[6,66] - frac[6,67] - frac[6,68] - frac[6,69] - frac[6,70]
   - frac[6,71] - frac[6,72] - frac[6,73] - frac[6,74] - frac[6,75]
   - frac[6,76] - frac[6,77] - frac[6,78] - frac[6,79] - frac[6,80]
   - frac[6,81] - frac[6,82] - frac[6,83] - frac[6,84] - frac[6,85]
   - frac[6,86] - frac[6,87] - frac[6,88] - frac[6,89] - frac[6,90]
   - frac[6,91] - frac[6,92] - frac[6,93] - frac[6,94] - frac[6,95]
   - frac[6,96] - frac[6,97] - frac[6,98] - frac[6,99] - frac[6,100]
   - frac[6,101] - frac[6,102] - frac[6,103] - frac[6,104] - frac[6,105]
   - frac[6,106] - frac[6,107] - frac[6,108] - frac[6,109] - frac[6,110]
   - frac[6,111] - frac[6,112] - frac[6,113] - frac[6,114] - frac[6,115]
   - frac[6,116] - frac[6,117] - frac[6,118] - frac[6,119] - frac[6,120]
   - frac[6,121] - frac[6,122] - frac[6,123] - frac[6,124] - frac[6,125]
   - frac[6,126] - frac[6,127] - frac[6,128] - frac[6,129] - frac[6,130]
   - frac[6,131] - frac[6,132] - frac[6,133] - frac[6,134] - frac[6,135]
   - frac[6,136] - frac[6,137] - frac[6,138] - frac[6,139] - frac[6,140]
   - frac[6,141] - frac[6,142] - frac[6,143] - frac[6,144] - frac[6,145]
   - frac[6,146] - frac[6,147] - frac[6,148] - frac[6,149] - frac[6,150]
   - frac[6,151] - frac[6,152] - frac[6,153] - frac[6,154] - frac[6,155]
   - frac[6,156] - frac[6,157] - frac[6,158] - frac[6,159] - frac[6,160]
   - frac[6,161] - frac[6,162] - frac[6,163] - frac[6,164] - frac[6,165]
   - frac[6,166] - frac[6,167] - frac[6,168] - frac[6,169] - frac[6,170]
   - frac[6,171] - frac[6,172] - frac[6,173] - frac[6,174] - frac[6,175]
   - frac[6,176] - frac[6,177] - frac[6,178] - frac[6,179] - frac[6,180]
   - frac[6,181] - frac[6,182] - frac[6,183] - frac[6,184] - frac[6,185]
   - frac[6,186] - frac[6,187] - frac[6,188] - frac[6,189] - frac[6,190]
   - frac[6,191] + rows_tot_cpu7 = 0
 mem_tot_cpu7: - mem[6,0] - mem[6,1] - mem[6,2] - mem[6,3] - mem[6,4]
   - mem[6,5] - mem[6,6] - mem[6,7] - mem[6,8] - mem[6,9] - mem[6,10]
   - mem[6,11] - mem[6,12] - mem[6,13] - mem[6,14] - mem[6,15] - mem[6,16]
   - mem[6,17] - mem[6,18] - mem[6,19] - mem[6,20] - mem[6,21] - mem[6,22]
   - mem[6,23] - mem[6,24] - mem[6,25] - mem[6,26] - mem[6,27] - mem[6,28]
   - mem[6,29] - mem[6,30] - mem[6,31] - mem[6,32] - mem[6,33] - mem[6,34]
   - mem[6,35] - mem[6,36] - mem[6,37] - mem[6,38] - mem[6,39] - mem[6,40]
   - mem[6,41] - mem[6,42] - mem[6,43] - mem[6,44] - mem[6,45] - mem[6,46]
   - mem[6,47] - mem[6,48] - mem[6,49] - mem[6,50] - mem[6,51] - mem[6,52]
   - mem[6,53] - mem[6,54] - mem[6,55] - mem[6,56] - mem[6,57] - mem[6,58]
   - mem[6,59] - mem[6,60] - mem[6,61] - mem[6,62] - mem[6,63] - mem[6,64]
   - mem[6,65] - mem[6,66] - mem[6,67] - mem[6,68] - mem[6,69] - mem[6,70]
   - mem[6,71] - mem[6,72] - mem[6,73] - mem[6,74] - mem[6,75] - mem[6,76]
   - mem[6,77] - mem[6,78] - mem[6,79] - mem[6,80] - mem[6,81] - mem[6,82]
   - mem[6,83] - mem[6,84] - mem[6,85] - mem[6,86] - mem[6,87] - mem[6,88]
   - mem[6,89] - mem[6,90] - mem[6,91] - mem[6,92] - mem[6,93] - mem[6,94]
   - mem[6,95] - mem[6,96] - mem[6,97] - mem[6,98] - mem[6,99] - mem[6,100]
   - mem[6,101] - mem[6,102] - mem[6,103] - mem[6,104] - mem[6,105]
   - mem[6,106] - mem[6,107] - mem[6,108] - mem[6,109] - mem[6,110]
   - mem[6,111] - mem[6,112] - mem[6,113] - mem[6,114] - mem[6,115]
   - mem[6,116] - mem[6,117] - mem[6,118] - mem[6,119] - mem[6,120]
   - mem[6,121] - mem[6,122] - mem[6,123] - mem[6,124] - mem[6,125]
   - mem[6,126] - mem[6,127] - mem[6,128] - mem[6,129] - mem[6,130]
   - mem[6,131] - mem[6,132] - mem[6,133] - mem[6,134] - mem[6,135]
   - mem[6,136] - mem[6,137] - mem[6,138] - mem[6,139] - mem[6,140]
   - mem[6,141] - mem[6,142] - mem[6,143] - mem[6,144] - mem[6,145]
   - mem[6,146] - mem[6,147] - mem[6,148] - mem[6,149] - mem[6,150]
   - mem[6,151] - mem[6,152] - mem[6,153] - mem[6,154] - mem[6,155]
   - mem[6,156] - mem[6,157] - mem[6,158] - mem[6,159] - mem[6,160]
   - mem[6,161] - mem[6,162] - mem[6,163] - mem[6,164] - mem[6,165]
   - mem[6,166] - mem[6,167] - mem[6,168] - mem[6,169] - mem[6,170]
   - mem[6,171] - mem[6,172] - mem[6,173] - mem[6,174] - mem[6,175]
   - mem[6,176] - mem[6,177] - mem[6,178] - mem[6,179] - mem[6,180]
   - mem[6,181] - mem[6,182] - mem[6,183] - mem[6,184] - mem[6,185]
   - mem[6,186] - mem[6,187] - mem[6,188] - mem[6,189] - mem[6,190]
   - mem[6,191] + mem_tot_cpu7 = 0
 ns_single_cpu7: 3.5 rows_tot_cpu7 - ns_single_cpu7 + pdt_m_rows_cpu7 <= 0
 capacity_cores_cpu7: cores_sketch_cpu7 + cores_dpdk_cpu7 <= 7
 ns_sketch_cpu7: - ns_single_cpu7 + pdt_nsk_c_cpu7 = 0
 ns_dpdk_cpu7: 7.142857142857143 cores_dpdk_cpu7 - pdt_nsc_dpdk_cpu7
   = -21.42857142857143
 rows_tot_cpu8: - frac[7,0] - frac[7,1] - frac[7,2] - frac[7,3] - frac[7,4]
   - frac[7,5] - frac[7,6] - frac[7,7] - frac[7,8] - frac[7,9] - frac[7,10]
   - frac[7,11] - frac[7,12] - frac[7,13] - frac[7,14] - frac[7,15]
   - frac[7,16] - frac[7,17] - frac[7,18] - frac[7,19] - frac[7,20]
   - frac[7,21] - frac[7,22] - frac[7,23] - frac[7,24] - frac[7,25]
   - frac[7,26] - frac[7,27] - frac[7,28] - frac[7,29] - frac[7,30]
   - frac[7,31] - frac[7,32] - frac[7,33] - frac[7,34] - frac[7,35]
   - frac[7,36] - frac[7,37] - frac[7,38] - frac[7,39] - frac[7,40]
   - frac[7,41] - frac[7,42] - frac[7,43] - frac[7,44] - frac[7,45]
   - frac[7,46] - frac[7,47] - frac[7,48] - frac[7,49] - frac[7,50]
   - frac[7,51] - frac[7,52] - frac[7,53] - frac[7,54] - frac[7,55]
   - frac[7,56] - frac[7,57] - frac[7,58] - frac[7,59] - frac[7,60]
   - frac[7,61] - frac[7,62] - frac[7,63] - frac[7,64] - frac[7,65]
   - frac[7,66] - frac[7,67] - frac[7,68] - frac[7,69] - frac[7,70]
   - frac[7,71] - frac[7,72] - frac[7,73] - frac[7,74] - frac[7,75]
   - frac[7,76] - frac[7,77] - frac[7,78] - frac[7,79] - frac[7,80]
   - frac[7,81] - frac[7,82] - frac[7,83] - frac[7,84] - frac[7,85]
   - frac[7,86] - frac[7,87] - frac[7,88] - frac[7,89] - frac[7,90]
   - frac[7,91] - frac[7,92] - frac[7,93] - frac[7,94] - frac[7,95]
   - frac[7,96] - frac[7,97] - frac[7,98] - frac[7,99] - frac[7,100]
   - frac[7,101] - frac[7,102] - frac[7,103] - frac[7,104] - frac[7,105]
   - frac[7,106] - frac[7,107] - frac[7,108] - frac[7,109] - frac[7,110]
   - frac[7,111] - frac[7,112] - frac[7,113] - frac[7,114] - frac[7,115]
   - frac[7,116] - frac[7,117] - frac[7,118] - frac[7,119] - frac[7,120]
   - frac[7,121] - frac[7,122] - frac[7,123] - frac[7,124] - frac[7,125]
   - frac[7,126] - frac[7,127] - frac[7,128] - frac[7,129] - frac[7,130]
   - frac[7,131] - frac[7,132] - frac[7,133] - frac[7,134] - frac[7,135]
   - frac[7,136] - frac[7,137] - frac[7,138] - frac[7,139] - frac[7,140]
   - frac[7,141] - frac[7,142] - frac[7,143] - frac[7,144] - frac[7,145]
   - frac[7,146] - frac[7,147] - frac[7,148] - frac[7,149] - frac[7,150]
   - frac[7,151] - frac[7,152] - frac[7,153] - frac[7,154] - frac[7,155]
   - frac[7,156] - frac[7,157] - frac[7,158] - frac[7,159] - frac[7,160]
   - frac[7,161] - frac[7,162] - frac[7,163] - frac[7,164] - frac[7,165]
   - frac[7,166] - frac[7,167] - frac[7,168] - frac[7,169] - frac[7,170]
   - frac[7,171] - frac[7,172] - frac[7,173] - frac[7,174] - frac[7,175]
   - frac[7,176] - frac[7,177] - frac[7,178] - frac[7,179] - frac[7,180]
   - frac[7,181] - frac[7,182] - frac[7,183] - frac[7,184] - frac[7,185]
   - frac[7,186] - frac[7,187] - frac[7,188] - frac[7,189] - frac[7,190]
   - frac[7,191] + rows_tot_cpu8 = 0
 mem_tot_cpu8: - mem[7,0] - mem[7,1] - mem[7,2] - mem[7,3] - mem[7,4]
   - mem[7,5] - mem[7,6] - mem[7,7] - mem[7,8] - mem[7,9] - mem[7,10]
   - mem[7,11] - mem[7,12] - mem[7,13] - mem[7,14] - mem[7,15] - mem[7,16]
   - mem[7,17] - mem[7,18] - mem[7,19] - mem[7,20] - mem[7,21] - mem[7,22]
   - mem[7,23] - mem[7,24] - mem[7,25] - mem[7,26] - mem[7,27] - mem[7,28]
   - mem[7,29] - mem[7,30] - mem[7,31] - mem[7,32] - mem[7,33] - mem[7,34]
   - mem[7,35] - mem[7,36] - mem[7,37] - mem[7,38] - mem[7,39] - mem[7,40]
   - mem[7,41] - mem[7,42] - mem[7,43] - mem[7,44] - mem[7,45] - mem[7,46]
   - mem[7,47] - mem[7,48] - mem[7,49] - mem[7,50] - mem[7,51] - mem[7,52]
   - mem[7,53] - mem[7,54] - mem[7,55] - mem[7,56] - mem[7,57] - mem[7,58]
   - mem[7,59] - mem[7,60] - mem[7,61] - mem[7,62] - mem[7,63] - mem[7,64]
   - mem[7,65] - mem[7,66] - mem[7,67] - mem[7,68] - mem[7,69] - mem[7,70]
   - mem[7,71] - mem[7,72] - mem[7,73] - mem[7,74] - mem[7,75] - mem[7,76]
   - mem[7,77] - mem[7,78] - mem[7,79] - mem[7,80] - mem[7,81] - mem[7,82]
   - mem[7,83] - mem[7,84] - mem[7,85] - mem[7,86] - mem[7,87] - mem[7,88]
   - mem[7,89] - mem[7,90] - mem[7,91] - mem[7,92] - mem[7,93] - mem[7,94]
   - mem[7,95] - mem[7,96] - mem[7,97] - mem[7,98] - mem[7,99] - mem[7,100]
   - mem[7,101] - mem[7,102] - mem[7,103] - mem[7,104] - mem[7,105]
   - mem[7,106] - mem[7,107] - mem[7,108] - mem[7,109] - mem[7,110]
   - mem[7,111] - mem[7,112] - mem[7,113] - mem[7,114] - mem[7,115]
   - mem[7,116] - mem[7,117] - mem[7,118] - mem[7,119] - mem[7,120]
   - mem[7,121] - mem[7,122] - mem[7,123] - mem[7,124] - mem[7,125]
   - mem[7,126] - mem[7,127] - mem[7,128] - mem[7,129] - mem[7,130]
   - mem[7,131] - mem[7,132] - mem[7,133] - mem[7,134] - mem[7,135]
   - mem[7,136] - mem[7,137] - mem[7,138] - mem[7,139] - mem[7,140]
   - mem[7,141] - mem[7,142] - mem[7,143] - mem[7,144] - mem[7,145]
   - mem[7,146] - mem[7,147] - mem[7,148] - mem[7,149] - mem[7,150]
   - mem[7,151] - mem[7,152] - mem[7,153] - mem[7,154] - mem[7,155]
   - mem[7,156] - mem[7,157] - mem[7,158] - mem[7,159] - mem[7,160]
   - mem[7,161] - mem[7,162] - mem[7,163] - mem[7,164] - mem[7,165]
   - mem[7,166] - mem[7,167] - mem[7,168] - mem[7,169] - mem[7,170]
   - mem[7,171] - mem[7,172] - mem[7,173] - mem[7,174] - mem[7,175]
   - mem[7,176] - mem[7,177] - mem[7,178] - mem[7,179] - mem[7,180]
   - mem[7,181] - mem[7,182] - mem[7,183] - mem[7,184] - mem[7,185]
   - mem[7,186] - mem[7,187] - mem[7,188] - mem[7,189] - mem[7,190]
   - mem[7,191] + mem_tot_cpu8 = 0
 ns_single_cpu8: 3.5 rows_tot_cpu8 - ns_single_cpu8 + pdt_m_rows_cpu8 <= 0
 capacity_cores_cpu8: cores_sketch_cpu8 + cores_dpdk_cpu8 <= 7
 ns_sketch_cpu8: - ns_single_cpu8 + pdt_nsk_c_cpu8 = 0
 ns_dpdk_cpu8: 7.142857142857143 cores_dpdk_cpu8 - pdt_nsc_dpdk_cpu8
   = -21.42857142857143
 rows_tot_cpu9: - frac[8,0] - frac[8,1] - frac[8,2] - frac[8,3] - frac[8,4]
   - frac[8,5] - frac[8,6] - frac[8,7] - frac[8,8] - frac[8,9] - frac[8,10]
   - frac[8,11] - frac[8,12] - frac[8,13] - frac[8,14] - frac[8,15]
   - frac[8,16] - frac[8,17] - frac[8,18] - frac[8,19] - frac[8,20]
   - frac[8,21] - frac[8,22] - frac[8,23] - frac[8,24] - frac[8,25]
   - frac[8,26] - frac[8,27] - frac[8,28] - frac[8,29] - frac[8,30]
   - frac[8,31] - frac[8,32] - frac[8,33] - frac[8,34] - frac[8,35]
   - frac[8,36] - frac[8,37] - frac[8,38] - frac[8,39] - frac[8,40]
   - frac[8,41] - frac[8,42] - frac[8,43] - frac[8,44] - frac[8,45]
   - frac[8,46] - frac[8,47] - frac[8,48] - frac[8,49] - frac[8,50]
   - frac[8,51] - frac[8,52] - frac[8,53] - frac[8,54] - frac[8,55]
   - frac[8,56] - frac[8,57] - frac[8,58] - frac[8,59] - frac[8,60]
   - frac[8,61] - frac[8,62] - frac[8,63] - frac[8,64] - frac[8,65]
   - frac[8,66] - frac[8,67] - frac[8,68] - frac[8,69] - frac[8,70]
   - frac[8,71] - frac[8,72] - frac[8,73] - frac[8,74] - frac[8,75]
   - frac[8,76] - frac[8,77] - frac[8,78] - frac[8,79] - frac[8,80]
   - frac[8,81] - frac[8,82] - frac[8,83] - frac[8,84] - frac[8,85]
   - frac[8,86] - frac[8,87] - frac[8,88] - frac[8,89] - frac[8,90]
   - frac[8,91] - frac[8,92] - frac[8,93] - frac[8,94] - frac[8,95]
   - frac[8,96] - frac[8,97] - frac[8,98] - frac[8,99] - frac[8,100]
   - frac[8,101] - frac[8,102] - frac[8,103] - frac[8,104] - frac[8,105]
   - frac[8,106] - frac[8,107] - frac[8,108] - frac[8,109] - frac[8,110]
   - frac[8,111] - frac[8,112] - frac[8,113] - frac[8,114] - frac[8,115]
   - frac[8,116] - frac[8,117] - frac[8,118] - frac[8,119] - frac[8,120]
   - frac[8,121] - frac[8,122] - frac[8,123] - frac[8,124] - frac[8,125]
   - frac[8,126] - frac[8,127] - frac[8,128] - frac[8,129] - frac[8,130]
   - frac[8,131] - frac[8,132] - frac[8,133] - frac[8,134] - frac[8,135]
   - frac[8,136] - frac[8,137] - frac[8,138] - frac[8,139] - frac[8,140]
   - frac[8,141] - frac[8,142] - frac[8,143] - frac[8,144] - frac[8,145]
   - frac[8,146] - frac[8,147] - frac[8,148] - frac[8,149] - frac[8,150]
   - frac[8,151] - frac[8,152] - frac[8,153] - frac[8,154] - frac[8,155]
   - frac[8,156] - frac[8,157] - frac[8,158] - frac[8,159] - frac[8,160]
   - frac[8,161] - frac[8,162] - frac[8,163] - frac[8,164] - frac[8,165]
   - frac[8,166] - frac[8,167] - frac[8,168] - frac[8,169] - frac[8,170]
   - frac[8,171] - frac[8,172] - frac[8,173] - frac[8,174] - frac[8,175]
   - frac[8,176] - frac[8,177] - frac[8,178] - frac[8,179] - frac[8,180]
   - frac[8,181] - frac[8,182] - frac[8,183] - frac[8,184] - frac[8,185]
   - frac[8,186] - frac[8,187] - frac[8,188] - frac[8,189] - frac[8,190]
   - frac[8,191] + rows_tot_cpu9 = 0
 mem_tot_cpu9: - mem[8,0] - mem[8,1] - mem[8,2] - mem[8,3] - mem[8,4]
   - mem[8,5] - mem[8,6] - mem[8,7] - mem[8,8] - mem[8,9] - mem[8,10]
   - mem[8,11] - mem[8,12] - mem[8,13] - mem[8,14] - mem[8,15] - mem[8,16]
   - mem[8,17] - mem[8,18] - mem[8,19] - mem[8,20] - mem[8,21] - mem[8,22]
   - mem[8,23] - mem[8,24] - mem[8,25] - mem[8,26] - mem[8,27] - mem[8,28]
   - mem[8,29] - mem[8,30] - mem[8,31] - mem[8,32] - mem[8,33] - mem[8,34]
   - mem[8,35] - mem[8,36] - mem[8,37] - mem[8,38] - mem[8,39] - mem[8,40]
   - mem[8,41] - mem[8,42] - mem[8,43] - mem[8,44] - mem[8,45] - mem[8,46]
   - mem[8,47] - mem[8,48] - mem[8,49] - mem[8,50] - mem[8,51] - mem[8,52]
   - mem[8,53] - mem[8,54] - mem[8,55] - mem[8,56] - mem[8,57] - mem[8,58]
   - mem[8,59] - mem[8,60] - mem[8,61] - mem[8,62] - mem[8,63] - mem[8,64]
   - mem[8,65] - mem[8,66] - mem[8,67] - mem[8,68] - mem[8,69] - mem[8,70]
   - mem[8,71] - mem[8,72] - mem[8,73] - mem[8,74] - mem[8,75] - mem[8,76]
   - mem[8,77] - mem[8,78] - mem[8,79] - mem[8,80] - mem[8,81] - mem[8,82]
   - mem[8,83] - mem[8,84] - mem[8,85] - mem[8,86] - mem[8,87] - mem[8,88]
   - mem[8,89] - mem[8,90] - mem[8,91] - mem[8,92] - mem[8,93] - mem[8,94]
   - mem[8,95] - mem[8,96] - mem[8,97] - mem[8,98] - mem[8,99] - mem[8,100]
   - mem[8,101] - mem[8,102] - mem[8,103] - mem[8,104] - mem[8,105]
   - mem[8,106] - mem[8,107] - mem[8,108] - mem[8,109] - mem[8,110]
   - mem[8,111] - mem[8,112] - mem[8,113] - mem[8,114] - mem[8,115]
   - mem[8,116] - mem[8,117] - mem[8,118] - mem[8,119] - mem[8,120]
   - mem[8,121] - mem[8,122] - mem[8,123] - mem[8,124] - mem[8,125]
   - mem[8,126] - mem[8,127] - mem[8,128] - mem[8,129] - mem[8,130]
   - mem[8,131] - mem[8,132] - mem[8,133] - mem[8,134] - mem[8,135]
   - mem[8,136] - mem[8,137] - mem[8,138] - mem[8,139] - mem[8,140]
   - mem[8,141] - mem[8,142] - mem[8,143] - mem[8,144] - mem[8,145]
   - mem[8,146] - mem[8,147] - mem[8,148] - mem[8,149] - mem[8,150]
   - mem[8,151] - mem[8,152] - mem[8,153] - mem[8,154] - mem[8,155]
   - mem[8,156] - mem[8,157] - mem[8,158] - mem[8,159] - mem[8,160]
   - mem[8,161] - mem[8,162] - mem[8,163] - mem[8,164] - mem[8,165]
   - mem[8,166] - mem[8,167] - mem[8,168] - mem[8,169] - mem[8,170]
   - mem[8,171] - mem[8,172] - mem[8,173] - mem[8,174] - mem[8,175]
   - mem[8,176] - mem[8,177] - mem[8,178] - mem[8,179] - mem[8,180]
   - mem[8,181] - mem[8,182] - mem[8,183] - mem[8,184] - mem[8,185]
   - mem[8,186] - mem[8,187] - mem[8,188] - mem[8,189] - mem[8,190]
   - mem[8,191] + mem_tot_cpu9 = 0
 ns_single_cpu9: 3.5 rows_tot_cpu9 - ns_single_cpu9 + pdt_m_rows_cpu9 <= 0
 capacity_cores_cpu9: cores_sketch_cpu9 + cores_dpdk_cpu9 <= 7
 ns_sketch_cpu9: - ns_single_cpu9 + pdt_nsk_c_cpu9 = 0
 ns_dpdk_cpu9: 7.142857142857143 cores_dpdk_cpu9 - pdt_nsc_dpdk_cpu9
   = -21.42857142857143
 rows_tot_cpu10: - frac[9,0] - frac[9,1] - frac[9,2] - frac[9,3]
   - frac[9,4] - frac[9,5] - frac[9,6] - frac[9,7] - frac[9,8] - frac[9,9]
   - frac[9,10] - frac[9,11] - frac[9,12] - frac[9,13] - frac[9,14]
   - frac[9,15] - frac[9,16] - frac[9,17] - frac[9,18] - frac[9,19]
   - frac[9,20] - frac[9,21] - frac[9,22] - frac[9,23] - frac[9,24]
   - frac[9,25] - frac[9,26] - frac[9,27] - frac[9,28] - frac[9,29]
   - frac[9,30] - frac[9,31] - frac[9,32] - frac[9,33] - frac[9,34]
   - frac[9,35] - frac[9,36] - frac[9,37] - frac[9,38] - frac[9,39]
   - frac[9,40] - frac[9,41] - frac[9,42] - frac[9,43] - frac[9,44]
   - frac[9,45] - frac[9,46] - frac[9,47] - frac[9,48] - frac[9,49]
   - frac[9,50] - frac[9,51] - frac[9,52] - frac[9,53] - frac[9,54]
   - frac[9,55] - frac[9,56] - frac[9,57] - frac[9,58] - frac[9,59]
   - frac[9,60] - frac[9,61] - frac[9,62] - frac[9,63] - frac[9,64]
   - frac[9,65] - frac[9,66] - frac[9,67] - frac[9,68] - frac[9,69]
   - frac[9,70] - frac[9,71] - frac[9,72] - frac[9,73] - frac[9,74]
   - frac[9,75] - frac[9,76] - frac[9,77] - frac[9,78] - frac[9,79]
   - frac[9,80] - frac[9,81] - frac[9,82] - frac[9,83] - frac[9,84]
   - frac[9,85] - frac[9,86] - frac[9,87] - frac[9,88] - frac[9,89]
   - frac[9,90] - frac[9,91] - frac[9,92] - frac[9,93] - frac[9,94]
   - frac[9,95] - frac[9,96] - frac[9,97] - frac[9,98] - frac[9,99]
   - frac[9,100] - frac[9,101] - frac[9,102] - frac[9,103] - frac[9,104]
   - frac[9,105] - frac[9,106] - frac[9,107] - frac[9,108] - frac[9,109]
   - frac[9,110] - frac[9,111] - frac[9,112] - frac[9,113] - frac[9,114]
   - frac[9,115] - frac[9,116] - frac[9,117] - frac[9,118] - frac[9,119]
   - frac[9,120] - frac[9,121] - frac[9,122] - frac[9,123] - frac[9,124]
   - frac[9,125] - frac[9,126] - frac[9,127] - frac[9,128] - frac[9,129]
   - frac[9,130] - frac[9,131] - frac[9,132] - frac[9,133] - frac[9,134]
   - frac[9,135] - frac[9,136] - frac[9,137] - frac[9,138] - frac[9,139]
   - frac[9,140] - frac[9,141] - frac[9,142] - frac[9,143] - frac[9,144]
   - frac[9,145] - frac[9,146] - frac[9,147] - frac[9,148] - frac[9,149]
   - frac[9,150] - frac[9,151] - frac[9,152] - frac[9,153] - frac[9,154]
   - frac[9,155] - frac[9,156] - frac[9,157] - frac[9,158] - frac[9,159]
   - frac[9,160] - frac[9,161] - frac[9,162] - frac[9,163] - frac[9,164]
   - frac[9,165] - frac[9,166] - frac[9,167] - frac[9,168] - frac[9,169]
   - frac[9,170] - frac[9,171] - frac[9,172] - frac[9,173] - frac[9,174]
   - frac[9,175] - frac[9,176] - frac[9,177] - frac[9,178] - frac[9,179]
   - frac[9,180] - frac[9,181] - frac[9,182] - frac[9,183] - frac[9,184]
   - frac[9,185] - frac[9,186] - frac[9,187] - frac[9,188] - frac[9,189]
   - frac[9,190] - frac[9,191] + rows_tot_cpu10 = 0
 mem_tot_cpu10: - mem[9,0] - mem[9,1] - mem[9,2] - mem[9,3] - mem[9,4]
   - mem[9,5] - mem[9,6] - mem[9,7] - mem[9,8] - mem[9,9] - mem[9,10]
   - mem[9,11] - mem[9,12] - mem[9,13] - mem[9,14] - mem[9,15] - mem[9,16]
   - mem[9,17] - mem[9,18] - mem[9,19] - mem[9,20] - mem[9,21] - mem[9,22]
   - mem[9,23] - mem[9,24] - mem[9,25] - mem[9,26] - mem[9,27] - mem[9,28]
   - mem[9,29] - mem[9,30] - mem[9,31] - mem[9,32] - mem[9,33] - mem[9,34]
   - mem[9,35] - mem[9,36] - mem[9,37] - mem[9,38] - mem[9,39] - mem[9,40]
   - mem[9,41] - mem[9,42] - mem[9,43] - mem[9,44] - mem[9,45] - mem[9,46]
   - mem[9,47] - mem[9,48] - mem[9,49] - mem[9,50] - mem[9,51] - mem[9,52]
   - mem[9,53] - mem[9,54] - mem[9,55] - mem[9,56] - mem[9,57] - mem[9,58]
   - mem[9,59] - mem[9,60] - mem[9,61] - mem[9,62] - mem[9,63] - mem[9,64]
   - mem[9,65] - mem[9,66] - mem[9,67] - mem[9,68] - mem[9,69] - mem[9,70]
   - mem[9,71] - mem[9,72] - mem[9,73] - mem[9,74] - mem[9,75] - mem[9,76]
   - mem[9,77] - mem[9,78] - mem[9,79] - mem[9,80] - mem[9,81] - mem[9,82]
   - mem[9,83] - mem[9,84] - mem[9,85] - mem[9,86] - mem[9,87] - mem[9,88]
   - mem[9,89] - mem[9,90] - mem[9,91] - mem[9,92] - mem[9,93] - mem[9,94]
   - mem[9,95] - mem[9,96] - mem[9,97] - mem[9,98] - mem[9,99] - mem[9,100]
   - mem[9,101] - mem[9,102] - mem[9,103] - mem[9,104] - mem[9,105]
   - mem[9,106] - mem[9,107] - mem[9,108] - mem[9,109] - mem[9,110]
   - mem[9,111] - mem[9,112] - mem[9,113] - mem[9,114] - mem[9,115]
   - mem[9,116] - mem[9,117] - mem[9,118] - mem[9,119] - mem[9,120]
   - mem[9,121] - mem[9,122] - mem[9,123] - mem[9,124] - mem[9,125]
   - mem[9,126] - mem[9,127] - mem[9,128] - mem[9,129] - mem[9,130]
   - mem[9,131] - mem[9,132] - mem[9,133] - mem[9,134] - mem[9,135]
   - mem[9,136] - mem[9,137] - mem[9,138] - mem[9,139] - mem[9,140]
   - mem[9,141] - mem[9,142] - mem[9,143] - mem[9,144] - mem[9,145]
   - mem[9,146] - mem[9,147] - mem[9,148] - mem[9,149] - mem[9,150]
   - mem[9,151] - mem[9,152] - mem[9,153] - mem[9,154] - mem[9,155]
   - mem[9,156] - mem[9,157] - mem[9,158] - mem[9,159] - mem[9,160]
   - mem[9,161] - mem[9,162] - mem[9,163] - mem[9,164] - mem[9,165]
   - mem[9,166] - mem[9,167] - mem[9,168] - mem[9,169] - mem[9,170]
   - mem[9,171] - mem[9,172] - mem[9,173] - mem[9,174] - mem[9,175]
   - mem[9,176] - mem[9,177] - mem[9,178] - mem[9,179] - mem[9,180]
   - mem[9,181] - mem[9,182] - mem[9,183] - mem[9,184] - mem[9,185]
   - mem[9,186] - mem[9,187] - mem[9,188] - mem[9,189] - mem[9,190]
   - mem[9,191] + mem_tot_cpu10 = 0
 ns_single_cpu10: 3.5 rows_tot_cpu10 - ns_single_cpu10 + pdt_m_rows_cpu10
   <= 0
 capacity_cores_cpu10: cores_sketch_cpu10 + cores_dpdk_cpu10 <= 7
 ns_sketch_cpu10: - ns_single_cpu10 + pdt_nsk_c_cpu10 = 0
 ns_dpdk_cpu10: 7.142857142857143 cores_dpdk_cpu10 - pdt_nsc_dpdk_cpu10
   = -21.42857142857143
 rows_tot_cpu11: - frac[10,0] - frac[10,1] - frac[10,2] - frac[10,3]
   - frac[10,4] - frac[10,5] - frac[10,6] - frac[10,7] - frac[10,8]
   - frac[10,9] - frac[10,10] - frac[10,11] - frac[10,12] - frac[10,13]
   - frac[10,14] - frac[10,15] - frac[10,16] - frac[10,17] - frac[10,18]
   - frac[10,19] - frac[10,20] - frac[10,21] - frac[10,22] - frac[10,23]
   - frac[10,24] - frac[10,25] - frac[10,26] - frac[10,27] - frac[10,28]
   - frac[10,29] - frac[10,30] - frac[10,31] - frac[10,32] - frac[10,33]
   - frac[10,34] - frac[10,35] - frac[10,36] - frac[10,37] - frac[10,38]
   - frac[10,39] - frac[10,40] - frac[10,41] - frac[10,42] - frac[10,43]
   - frac[10,44] - frac[10,45] - frac[10,46] - frac[10,47] - frac[10,48]
   - frac[10,49] - frac[10,50] - frac[10,51] - frac[10,52] - frac[10,53]
   - frac[10,54] - frac[10,55] - frac[10,56] - frac[10,57] - frac[10,58]
   - frac[10,59] - frac[10,60] - frac[10,61] - frac[10,62] - frac[10,63]
   - frac[10,64] - frac[10,65] - frac[10,66] - frac[10,67] - frac[10,68]
   - frac[10,69] - frac[10,70] - frac[10,71] - frac[10,72] - frac[10,73]
   - frac[10,74] - frac[10,75] - frac[10,76] - frac[10,77] - frac[10,78]
   - frac[10,79] - frac[10,80] - frac[10,81] - frac[10,82] - frac[10,83]
   - frac[10,84] - frac[10,85] - frac[10,86] - frac[10,87] - frac[10,88]
   - frac[10,89] - frac[10,90] - frac[10,91] - frac[10,92] - frac[10,93]
   - frac[10,94] - frac[10,95] - frac[10,96] - frac[10,97] - frac[10,98]
   - frac[10,99] - frac[10,100] - frac[10,101] - frac[10,102]
   - frac[10,103] - frac[10,104] - frac[10,105] - frac[10,106]
   - frac[10,107] - frac[10,108] - frac[10,109] - frac[10,110]
   - frac[10,111] - frac[10,112] - frac[10,113] - frac[10,114]
   - frac[10,115] - frac[10,116] - frac[10,117] - frac[10,118]
   - frac[10,119] - frac[10,120] - frac[10,121] - frac[10,122]
   - frac[10,123] - frac[10,124] - frac[10,125] - frac[10,126]
   - frac[10,127] - frac[10,128] - frac[10,129] - frac[10,130]
   - frac[10,131] - frac[10,132] - frac[10,133] - frac[10,134]
   - frac[10,135] - frac[10,136] - frac[10,137] - frac[10,138]
   - frac[10,139] - frac[10,140] - frac[10,141] - frac[10,142]
   - frac[10,143] - frac[10,144] - frac[10,145] - frac[10,146]
   - frac[10,147] - frac[10,148] - frac[10,149] - frac[10,150]
   - frac[10,151] - frac[10,152] - frac[10,153] - frac[10,154]
   - frac[10,155] - frac[10,156] - frac[10,157] - frac[10,158]
   - frac[10,159] - frac[10,160] - frac[10,161] - frac[10,162]
   - frac[10,163] - frac[10,164] - frac[10,165] - frac[10,166]
   - frac[10,167] - frac[10,168] - frac[10,169] - frac[10,170]
   - frac[10,171] - frac[10,172] - frac[10,173] - frac[10,174]
   - frac[10,175] - frac[10,176] - frac[10,177] - frac[10,178]
   - frac[10,179] - frac[10,180] - frac[10,181] - frac[10,182]
   - frac[10,183] - frac[10,184] - frac[10,185] - frac[10,186]
   - frac[10,187] - frac[10,188] - frac[10,189] - frac[10,190]
   - frac[10,191] + rows_tot_cpu11 = 0
 mem_tot_cpu11: - mem[10,0] - mem[10,1] - mem[10,2] - mem[10,3] - mem[10,4]
   - mem[10,5] - mem[10,6] - mem[10,7] - mem[10,8] - mem[10,9] - mem[10,10]
   - mem[10,11] - mem[10,12] - mem[10,13] - mem[10,14] - mem[10,15]
   - mem[10,16] - mem[10,17] - mem[10,18] - mem[10,19] - mem[10,20]
   - mem[10,21] - mem[10,22] - mem[10,23] - mem[10,24] - mem[10,25]
   - mem[10,26] - mem[10,27] - mem[10,28] - mem[10,29] - mem[10,30]
   - mem[10,31] - mem[10,32] - mem[10,33] - mem[10,34] - mem[10,35]
   - mem[10,36] - mem[10,37] - mem[10,38] - mem[10,39] - mem[10,40]
   - mem[10,41] - mem[10,42] - mem[10,43] - mem[10,44] - mem[10,45]
   - mem[10,46] - mem[10,47] - mem[10,48] - mem[10,49] - mem[10,50]
   - mem[10,51] - mem[10,52] - mem[10,53] - mem[10,54] - mem[10,55]
   - mem[10,56] - mem[10,57] - mem[10,58] - mem[10,59] - mem[10,60]
   - mem[10,61] - mem[10,62] - mem[10,63] - mem[10,64] - mem[10,65]
   - mem[10,66] - mem[10,67] - mem[10,68] - mem[10,69] - mem[10,70]
   - mem[10,71] - mem[10,72] - mem[10,73] - mem[10,74] - mem[10,75]
   - mem[10,76] - mem[10,77] - mem[10,78] - mem[10,79] - mem[10,80]
   - mem[10,81] - mem[10,82] - mem[10,83] - mem[10,84] - mem[10,85]
   - mem[10,86] - mem[10,87] - mem[10,88] - mem[10,89] - mem[10,90]
   - mem[10,91] - mem[10,92] - mem[10,93] - mem[10,94] - mem[10,95]
   - mem[10,96] - mem[10,97] - mem[10,98] - mem[10,99] - mem[10,100]
   - mem[10,101] - mem[10,102] - mem[10,103] - mem[10,104] - mem[10,105]
   - mem[10,106] - mem[10,107] - mem[10,108] - mem[10,109] - mem[10,110]
   - mem[10,111] - mem[10,112] - mem[10,113] - mem[10,114] - mem[10,115]
   - mem[10,116] - mem[10,117] - mem[10,118] - mem[10,119] - mem[10,120]
   - mem[10,121] - mem[10,122] - mem[10,123] - mem[10,124] - mem[10,125]
   - mem[10,126] - mem[10,127] - mem[10,128] - mem[10,129] - mem[10,130]
   - mem[10,131] - mem[10,132] - mem[10,133] - mem[10,134] - mem[10,135]
   - mem[10,136] - mem[10,137] - mem[10,138] - mem[10,139] - mem[10,140]
   - mem[10,141] - mem[10,142] - mem[10,143] - mem[10,144] - mem[10,145]
   - mem[10,146] - mem[10,147] - mem[10,148] - mem[10,149] - mem[10,150]
   - mem[10,151] - mem[10,152] - mem[10,153] - mem[10,154] - mem[10,155]
   - mem[10,156] - mem[10,157] - mem[10,158] - mem[10,159] - mem[10,160]
   - mem[10,161] - mem[10,162] - mem[10,163] - mem[10,164] - mem[10,165]
   - mem[10,166] - mem[10,167] - mem[10,168] - mem[10,169] - mem[10,170]
   - mem[10,171] - mem[10,172] - mem[10,173] - mem[10,174] - mem[10,175]
   - mem[10,176] - mem[10,177] - mem[10,178] - mem[10,179] - mem[10,180]
   - mem[10,181] - mem[10,182] - mem[10,183] - mem[10,184] - mem[10,185]
   - mem[10,186] - mem[10,187] - mem[10,188] - mem[10,189] - mem[10,190]
   - mem[10,191] + mem_tot_cpu11 = 0
 ns_single_cpu11: 3.5 rows_tot_cpu11 - ns_single_cpu11 + pdt_m_rows_cpu11
   <= 0
 capacity_cores_cpu11: cores_sketch_cpu11 + cores_dpdk_cpu11 <= 7
 ns_sketch_cpu11: - ns_single_cpu11 + pdt_nsk_c_cpu11 = 0
 ns_dpdk_cpu11: 7.142857142857143 cores_dpdk_cpu11 - pdt_nsc_dpdk_cpu11
   = -21.42857142857143
 rows_tot_cpu12: - frac[11,0] - frac[11,1] - frac[11,2] - frac[11,3]
   - frac[11,4] - frac[11,5] - frac[11,6] - frac[11,7] - frac[11,8]
   - frac[11,9] - frac[11,10] - frac[11,11] - frac[11,12] - frac[11,13]
   - frac[11,14] - frac[11,15] - frac[11,16] - frac[11,17] - frac[11,18]
   - frac[11,19] - frac[11,20] - frac[11,21] - frac[11,22] - frac[11,23]
   - frac[11,24] - frac[11,25] - frac[11,26] - frac[11,27] - frac[11,28]
   - frac[11,29] - frac[11,30] - frac[11,31] - frac[11,32] - frac[11,33]
   - frac[11,34] - frac[11,35] - frac[11,36] - frac[11,37] - frac[11,38]
   - frac[11,39] - frac[11,40] - frac[11,41] - frac[11,42] - frac[11,43]
   - frac[11,44] - frac[11,45] - frac[11,46] - frac[11,47] - frac[11,48]
   - frac[11,49] - frac[11,50] - frac[11,51] - frac[11,52] - frac[11,53]
   - frac[11,54] - frac[11,55] - frac[11,56] - frac[11,57] - frac[11,58]
   - frac[11,59] - frac[11,60] - frac[11,61] - frac[11,62] - frac[11,63]
   - frac[11,64] - frac[11,65] - frac[11,66] - frac[11,67] - frac[11,68]
   - frac[11,69] - frac[11,70] - frac[11,71] - frac[11,72] - frac[11,73]
   - frac[11,74] - frac[11,75] - frac[11,76] - frac[11,77] - frac[11,78]
   - frac[11,79] - frac[11,80] - frac[11,81] - frac[11,82] - frac[11,83]
   - frac[11,84] - frac[11,85] - frac[11,86] - frac[11,87] - frac[11,88]
   - frac[11,89] - frac[11,90] - frac[11,91] - frac[11,92] - frac[11,93]
   - frac[11,94] - frac[11,95] - frac[11,96] - frac[11,97] - frac[11,98]
   - frac[11,99] - frac[11,100] - frac[11,101] - frac[11,102]
   - frac[11,103] - frac[11,104] - frac[11,105] - frac[11,106]
   - frac[11,107] - frac[11,108] - frac[11,109] - frac[11,110]
   - frac[11,111] - frac[11,112] - frac[11,113] - frac[11,114]
   - frac[11,115] - frac[11,116] - frac[11,117] - frac[11,118]
   - frac[11,119] - frac[11,120] - frac[11,121] - frac[11,122]
   - frac[11,123] - frac[11,124] - frac[11,125] - frac[11,126]
   - frac[11,127] - frac[11,128] - frac[11,129] - frac[11,130]
   - frac[11,131] - frac[11,132] - frac[11,133] - frac[11,134]
   - frac[11,135] - frac[11,136] - frac[11,137] - frac[11,138]
   - frac[11,139] - frac[11,140] - frac[11,141] - frac[11,142]
   - frac[11,143] - frac[11,144] - frac[11,145] - frac[11,146]
   - frac[11,147] - frac[11,148] - frac[11,149] - frac[11,150]
   - frac[11,151] - frac[11,152] - frac[11,153] - frac[11,154]
   - frac[11,155] - frac[11,156] - frac[11,157] - frac[11,158]
   - frac[11,159] - frac[11,160] - frac[11,161] - frac[11,162]
   - frac[11,163] - frac[11,164] - frac[11,165] - frac[11,166]
   - frac[11,167] - frac[11,168] - frac[11,169] - frac[11,170]
   - frac[11,171] - frac[11,172] - frac[11,173] - frac[11,174]
   - frac[11,175] - frac[11,176] - frac[11,177] - frac[11,178]
   - frac[11,179] - frac[11,180] - frac[11,181] - frac[11,182]
   - frac[11,183] - frac[11,184] - frac[11,185] - frac[11,186]
   - frac[11,187] - frac[11,188] - frac[11,189] - frac[11,190]
   - frac[11,191] + rows_tot_cpu12 = 0
 mem_tot_cpu12: - mem[11,0] - mem[11,1] - mem[11,2] - mem[11,3] - mem[11,4]
   - mem[11,5] - mem[11,6] - mem[11,7] - mem[11,8] - mem[11,9] - mem[11,10]
   - mem[11,11] - mem[11,12] - mem[11,13] - mem[11,14] - mem[11,15]
   - mem[11,16] - mem[11,17] - mem[11,18] - mem[11,19] - mem[11,20]
   - mem[11,21] - mem[11,22] - mem[11,23] - mem[11,24] - mem[11,25]
   - mem[11,26] - mem[11,27] - mem[11,28] - mem[11,29] - mem[11,30]
   - mem[11,31] - mem[11,32] - mem[11,33] - mem[11,34] - mem[11,35]
   - mem[11,36] - mem[11,37] - mem[11,38] - mem[11,39] - mem[11,40]
   - mem[11,41] - mem[11,42] - mem[11,43] - mem[11,44] - mem[11,45]
   - mem[11,46] - mem[11,47] - mem[11,48] - mem[11,49] - mem[11,50]
   - mem[11,51] - mem[11,52] - mem[11,53] - mem[11,54] - mem[11,55]
   - mem[11,56] - mem[11,57] - mem[11,58] - mem[11,59] - mem[11,60]
   - mem[11,61] - mem[11,62] - mem[11,63] - mem[11,64] - mem[11,65]
   - mem[11,66] - mem[11,67] - mem[11,68] - mem[11,69] - mem[11,70]
   - mem[11,71] - mem[11,72] - mem[11,73] - mem[11,74] - mem[11,75]
   - mem[11,76] - mem[11,77] - mem[11,78] - mem[11,79] - mem[11,80]
   - mem[11,81] - mem[11,82] - mem[11,83] - mem[11,84] - mem[11,85]
   - mem[11,86] - mem[11,87] - mem[11,88] - mem[11,89] - mem[11,90]
   - mem[11,91] - mem[11,92] - mem[11,93] - mem[11,94] - mem[11,95]
   - mem[11,96] - mem[11,97] - mem[11,98] - mem[11,99] - mem[11,100]
   - mem[11,101] - mem[11,102] - mem[11,103] - mem[11,104] - mem[11,105]
   - mem[11,106] - mem[11,107] - mem[11,108] - mem[11,109] - mem[11,110]
   - mem[11,111] - mem[11,112] - mem[11,113] - mem[11,114] - mem[11,115]
   - mem[11,116] - mem[11,117] - mem[11,118] - mem[11,119] - mem[11,120]
   - mem[11,121] - mem[11,122] - mem[11,123] - mem[11,124] - mem[11,125]
   - mem[11,126] - mem[11,127] - mem[11,128] - mem[11,129] - mem[11,130]
   - mem[11,131] - mem[11,132] - mem[11,133] - mem[11,134] - mem[11,135]
   - mem[11,136] - mem[11,137] - mem[11,138] - mem[11,139] - mem[11,140]
   - mem[11,141] - mem[11,142] - mem[11,143] - mem[11,144] - mem[11,145]
   - mem[11,146] - mem[11,147] - mem[11,148] - mem[11,149] - mem[11,150]
   - mem[11,151] - mem[11,152] - mem[11,153] - mem[11,154] - mem[11,155]
   - mem[11,156] - mem[11,157] - mem[11,158] - mem[11,159] - mem[11,160]
   - mem[11,161] - mem[11,162] - mem[11,163] - mem[11,164] - mem[11,165]
   - mem[11,166] - mem[11,167] - mem[11,168] - mem[11,169] - mem[11,170]
   - mem[11,171] - mem[11,172] - mem[11,173] - mem[11,174] - mem[11,175]
   - mem[11,176] - mem[11,177] - mem[11,178] - mem[11,179] - mem[11,180]
   - mem[11,181] - mem[11,182] - mem[11,183] - mem[11,184] - mem[11,185]
   - mem[11,186] - mem[11,187] - mem[11,188] - mem[11,189] - mem[11,190]
   - mem[11,191] + mem_tot_cpu12 = 0
 ns_single_cpu12: 3.5 rows_tot_cpu12 - ns_single_cpu12 + pdt_m_rows_cpu12
   <= 0
 capacity_cores_cpu12: cores_sketch_cpu12 + cores_dpdk_cpu12 <= 7
 ns_sketch_cpu12: - ns_single_cpu12 + pdt_nsk_c_cpu12 = 0
 ns_dpdk_cpu12: 7.142857142857143 cores_dpdk_cpu12 - pdt_nsc_dpdk_cpu12
   = -21.42857142857143
 rows_tot_cpu13: - frac[12,0] - frac[12,1] - frac[12,2] - frac[12,3]
   - frac[12,4] - frac[12,5] - frac[12,6] - frac[12,7] - frac[12,8]
   - frac[12,9] - frac[12,10] - frac[12,11] - frac[12,12] - frac[12,13]
   - frac[12,14] - frac[12,15] - frac[12,16] - frac[12,17] - frac[12,18]
   - frac[12,19] - frac[12,20] - frac[12,21] - frac[12,22] - frac[12,23]
   - frac[12,24] - frac[12,25] - frac[12,26] - frac[12,27] - frac[12,28]
   - frac[12,29] - frac[12,30] - frac[12,31] - frac[12,32] - frac[12,33]
   - frac[12,34] - frac[12,35] - frac[12,36] - frac[12,37] - frac[12,38]
   - frac[12,39] - frac[12,40] - frac[12,41] - frac[12,42] - frac[12,43]
   - frac[12,44] - frac[12,45] - frac[12,46] - frac[12,47] - frac[12,48]
   - frac[12,49] - frac[12,50] - frac[12,51] - frac[12,52] - frac[12,53]
   - frac[12,54] - frac[12,55] - frac[12,56] - frac[12,57] - frac[12,58]
   - frac[12,59] - frac[12,60] - frac[12,61] - frac[12,62] - frac[12,63]
   - frac[12,64] - frac[12,65] - frac[12,66] - frac[12,67] - frac[12,68]
   - frac[12,69] - frac[12,70] - frac[12,71] - frac[12,72] - frac[12,73]
   - frac[12,74] - frac[12,75] - frac[12,76] - frac[12,77] - frac[12,78]
   - frac[12,79] - frac[12,80] - frac[12,81] - frac[12,82] - frac[12,83]
   - frac[12,84] - frac[12,85] - frac[12,86] - frac[12,87] - frac[12,88]
   - frac[12,89] - frac[12,90] - frac[12,91] - frac[12,92] - frac[12,93]
   - frac[12,94] - frac[12,95] - frac[12,96] - frac[12,97] - frac[12,98]
   - frac[12,99] - frac[12,100] - frac[12,101] - frac[12,102]
   - frac[12,103] - frac[12,104] - frac[12,105] - frac[12,106]
   - frac[12,107] - frac[12,108] - frac[12,109] - frac[12,110]
   - frac[12,111] - frac[12,112] - frac[12,113] - frac[12,114]
   - frac[12,115] - frac[12,116] - frac[12,117] - frac[12,118]
   - frac[12,119] - frac[12,120] - frac[12,121] - frac[12,122]
   - frac[12,123] - frac[12,124] - frac[12,125] - frac[12,126]
   - frac[12,127] - frac[12,128] - frac[12,129] - frac[12,130]
   - frac[12,131] - frac[12,132] - frac[12,133] - frac[12,134]
   - frac[12,135] - frac[12,136] - frac[12,137] - frac[12,138]
   - frac[12,139] - frac[12,140] - frac[12,141] - frac[12,142]
   - frac[12,143] - frac[12,144] - frac[12,145] - frac[12,146]
   - frac[12,147] - frac[12,148] - frac[12,149] - frac[12,150]
   - frac[12,151] - frac[12,152] - frac[12,153] - frac[12,154]
   - frac[12,155] - frac[12,156] - frac[12,157] - frac[12,158]
   - frac[12,159] - frac[12,160] - frac[12,161] - frac[12,162]
   - frac[12,163] - frac[12,164] - frac[12,165] - frac[12,166]
   - frac[12,167] - frac[12,168] - frac[12,169] - frac[12,170]
   - frac[12,171] - frac[12,172] - frac[12,173] - frac[12,174]
   - frac[12,175] - frac[12,176] - frac[12,177] - frac[12,178]
   - frac[12,179] - frac[12,180] - frac[12,181] - frac[12,182]
   - frac[12,183] - frac[12,184] - frac[12,185] - frac[12,186]
   - frac[12,187] - frac[12,188] - frac[12,189] - frac[12,190]
   - frac[12,191] + rows_tot_cpu13 = 0
 mem_tot_cpu13: - mem[12,0] - mem[12,1] - mem[12,2] - mem[12,3] - mem[12,4]
   - mem[12,5] - mem[12,6] - mem[12,7] - mem[12,8] - mem[12,9] - mem[12,10]
   - mem[12,11] - mem[12,12] - mem[12,13] - mem[12,14] - mem[12,15]
   - mem[12,16] - mem[12,17] - mem[12,18] - mem[12,19] - mem[12,20]
   - mem[12,21] - mem[12,22] - mem[12,23] - mem[12,24] - mem[12,25]
   - mem[12,26] - mem[12,27] - mem[12,28] - mem[12,29] - mem[12,30]
   - mem[12,31] - mem[12,32] - mem[12,33] - mem[12,34] - mem[12,35]
   - mem[12,36] - mem[12,37] - mem[12,38] - mem[12,39] - mem[12,40]
   - mem[12,41] - mem[12,42] - mem[12,43] - mem[12,44] - mem[12,45]
   - mem[12,46] - mem[12,47] - mem[12,48] - mem[12,49] - mem[12,50]
   - mem[12,51] - mem[12,52] - mem[12,53] - mem[12,54] - mem[12,55]
   - mem[12,56] - mem[12,57] - mem[12,58] - mem[12,59] - mem[12,60]
   - mem[12,61] - mem[12,62] - mem[12,63] - mem[12,64] - mem[12,65]
   - mem[12,66] - mem[12,67] - mem[12,68] - mem[12,69] - mem[12,70]
   - mem[12,71] - mem[12,72] - mem[12,73] - mem[12,74] - mem[12,75]
   - mem[12,76] - mem[12,77] - mem[12,78] - mem[12,79] - mem[12,80]
   - mem[12,81] - mem[12,82] - mem[12,83] - mem[12,84] - mem[12,85]
   - mem[12,86] - mem[12,87] - mem[12,88] - mem[12,89] - mem[12,90]
   - mem[12,91] - mem[12,92] - mem[12,93] - mem[12,94] - mem[12,95]
   - mem[12,96] - mem[12,97] - mem[12,98] - mem[12,99] - mem[12,100]
   - mem[12,101] - mem[12,102] - mem[12,103] - mem[12,104] - mem[12,105]
   - mem[12,106] - mem[12,107] - mem[12,108] - mem[12,109] - mem[12,110]
   - mem[12,111] - mem[12,112] - mem[12,113] - mem[12,114] - mem[12,115]
   - mem[12,116] - mem[12,117] - mem[12,118] - mem[12,119] - mem[12,120]
   - mem[12,121] - mem[12,122] - mem[12,123] - mem[12,124] - mem[12,125]
   - mem[12,126] - mem[12,127] - mem[12,128] - mem[12,129] - mem[12,130]
   - mem[12,131] - mem[12,132] - mem[12,133] - mem[12,134] - mem[12,135]
   - mem[12,136] - mem[12,137] - mem[12,138] - mem[12,139] - mem[12,140]
   - mem[12,141] - mem[12,142] - mem[12,143] - mem[12,144] - mem[12,145]
   - mem[12,146] - mem[12,147] - mem[12,148] - mem[12,149] - mem[12,150]
   - mem[12,151] - mem[12,152] - mem[12,153] - mem[12,154] - mem[12,155]
   - mem[12,156] - mem[12,157] - mem[12,158] - mem[12,159] - mem[12,160]
   - mem[12,161] - mem[12,162] - mem[12,163] - mem[12,164] - mem[12,165]
   - mem[12,166] - mem[12,167] - mem[12,168] - mem[12,169] - mem[12,170]
   - mem[12,171] - mem[12,172] - mem[12,173] - mem[12,174] - mem[12,175]
   - mem[12,176] - mem[12,177] - mem[12,178] - mem[12,179] - mem[12,180]
   - mem[12,181] - mem[12,182] - mem[12,183] - mem[12,184] - mem[12,185]
   - mem[12,186] - mem[12,187] - mem[12,188] - mem[12,189] - mem[12,190]
   - mem[12,191] + mem_tot_cpu13 = 0
 ns_single_cpu13: 3.5 rows_tot_cpu13 - ns_single_cpu13 + pdt_m_rows_cpu13
   <= 0
 capacity_cores_cpu13: cores_sketch_cpu13 + cores_dpdk_cpu13 <= 7
 ns_sketch_cpu13: - ns_single_cpu13 + pdt_nsk_c_cpu13 = 0
 ns_dpdk_cpu13: 7.142857142857143 cores_dpdk_cpu13 - pdt_nsc_dpdk_cpu13
   = -21.42857142857143
 rows_tot_cpu14: - frac[13,0] - frac[13,1] - frac[13,2] - frac[13,3]
   - frac[13,4] - frac[13,5] - frac[13,6] - frac[13,7] - frac[13,8]
   - frac[13,9] - frac[13,10] - frac[13,11] - frac[13,12] - frac[13,13]
   - frac[13,14] - frac[13,15] - frac[13,16] - frac[13,17] - frac[13,18]
   - frac[13,19] - frac[13,20] - frac[13,21] - frac[13,22] - frac[13,23]
   - frac[13,24] - frac[13,25] - frac[13,26] - frac[13,27] - frac[13,28]
   - frac[13,29] - frac[13,30] - frac[13,31] - frac[13,32] - frac[13,33]
   - frac[13,34] - frac[13,35] - frac[13,36] - frac[13,37] - frac[13,38]
   - frac[13,39] - frac[13,40] - frac[13,41] - frac[13,42] - frac[13,43]
   - frac[13,44] - frac[13,45] - frac[13,46] - frac[13,47] - frac[13,48]
   - frac[13,49] - frac[13,50] - frac[13,51] - frac[13,52] - frac[13,53]
   - frac[13,54] - frac[13,55] - frac[13,56] - frac[13,57] - frac[13,58]
   - frac[13,59] - frac[13,60] - frac[13,61] - frac[13,62] - frac[13,63]
   - frac[13,64] - frac[13,65] - frac[13,66] - frac[13,67] - frac[13,68]
   - frac[13,69] - frac[13,70] - frac[13,71] - frac[13,72] - frac[13,73]
   - frac[13,74] - frac[13,75] - frac[13,76] - frac[13,77] - frac[13,78]
   - frac[13,79] - frac[13,80] - frac[13,81] - frac[13,82] - frac[13,83]
   - frac[13,84] - frac[13,85] - frac[13,86] - frac[13,87] - frac[13,88]
   - frac[13,89] - frac[13,90] - frac[13,91] - frac[13,92] - frac[13,93]
   - frac[13,94] - frac[13,95] - frac[13,96] - frac[13,97] - frac[13,98]
   - frac[13,99] - frac[13,100] - frac[13,101] - frac[13,102]
   - frac[13,103] - frac[13,104] - frac[13,105] - frac[13,106]
   - frac[13,107] - frac[13,108] - frac[13,109] - frac[13,110]
   - frac[13,111] - frac[13,112] - frac[13,113] - frac[13,114]
   - frac[13,115] - frac[13,116] - frac[13,117] - frac[13,118]
   - frac[13,119] - frac[13,120] - frac[13,121] - frac[13,122]
   - frac[13,123] - frac[13,124] - frac[13,125] - frac[13,126]
   - frac[13,127] - frac[13,128] - frac[13,129] - frac[13,130]
   - frac[13,131] - frac[13,132] - frac[13,133] - frac[13,134]
   - frac[13,135] - frac[13,136] - frac[13,137] - frac[13,138]
   - frac[13,139] - frac[13,140] - frac[13,141] - frac[13,142]
   - frac[13,143] - frac[13,144] - frac[13,145] - frac[13,146]
   - frac[13,147] - frac[13,148] - frac[13,149] - frac[13,150]
   - frac[13,151] - frac[13,152] - frac[13,153] - frac[13,154]
   - frac[13,155] - frac[13,156] - frac[13,157] - frac[13,158]
   - frac[13,159] - frac[13,160] - frac[13,161] - frac[13,162]
   - frac[13,163] - frac[13,164] - frac[13,165] - frac[13,166]
   - frac[13,167] - frac[13,168] - frac[13,169] - frac[13,170]
   - frac[13,171] - frac[13,172] - frac[13,173] - frac[13,174]
   - frac[13,175] - frac[13,176] - frac[13,177] - frac[13,178]
   - frac[13,179] - frac[13,180] - frac[13,181] - frac[13,182]
   - frac[13,183] - frac[13,184] - frac[13,185] - frac[13,186]
   - frac[13,187] - frac[13,188] - frac[13,189] - frac[13,190]
   - frac[13,191] + rows_tot_cpu14 = 0
 mem_tot_cpu14: - mem[13,0] - mem[13,1] - mem[13,2] - mem[13,3] - mem[13,4]
   - mem[13,5] - mem[13,6] - mem[13,7] - mem[13,8] - mem[13,9] - mem[13,10]
   - mem[13,11] - mem[13,12] - mem[13,13] - mem[13,14] - mem[13,15]
   - mem[13,16] - mem[13,17] - mem[13,18] - mem[13,19] - mem[13,20]
   - mem[13,21] - mem[13,22] - mem[13,23] - mem[13,24] - mem[13,25]
   - mem[13,26] - mem[13,27] - mem[13,28] - mem[13,29] - mem[13,30]
   - mem[13,31] - mem[13,32] - mem[13,33] - mem[13,34] - mem[13,35]
   - mem[13,36] - mem[13,37] - mem[13,38] - mem[13,39] - mem[13,40]
   - mem[13,41] - mem[13,42] - mem[13,43] - mem[13,44] - mem[13,45]
   - mem[13,46] - mem[13,47] - mem[13,48] - mem[13,49] - mem[13,50]
   - mem[13,51] - mem[13,52] - mem[13,53] - mem[13,54] - mem[13,55]
   - mem[13,56] - mem[13,57] - mem[13,58] - mem[13,59] - mem[13,60]
   - mem[13,61] - mem[13,62] - mem[13,63] - mem[13,64] - mem[13,65]
   - mem[13,66] - mem[13,67] - mem[13,68] - mem[13,69] - mem[13,70]
   - mem[13,71] - mem[13,72] - mem[13,73] - mem[13,74] - mem[13,75]
   - mem[13,76] - mem[13,77] - mem[13,78] - mem[13,79] - mem[13,80]
   - mem[13,81] - mem[13,82] - mem[13,83] - mem[13,84] - mem[13,85]
   - mem[13,86] - mem[13,87] - mem[13,88] - mem[13,89] - mem[13,90]
   - mem[13,91] - mem[13,92] - mem[13,93] - mem[13,94] - mem[13,95]
   - mem[13,96] - mem[13,97] - mem[13,98] - mem[13,99] - mem[13,100]
   - mem[13,101] - mem[13,102] - mem[13,103] - mem[13,104] - mem[13,105]
   - mem[13,106] - mem[13,107] - mem[13,108] - mem[13,109] - mem[13,110]
   - mem[13,111] - mem[13,112] - mem[13,113] - mem[13,114] - mem[13,115]
   - mem[13,116] - mem[13,117] - mem[13,118] - mem[13,119] - mem[13,120]
   - mem[13,121] - mem[13,122] - mem[13,123] - mem[13,124] - mem[13,125]
   - mem[13,126] - mem[13,127] - mem[13,128] - mem[13,129] - mem[13,130]
   - mem[13,131] - mem[13,132] - mem[13,133] - mem[13,134] - mem[13,135]
   - mem[13,136] - mem[13,137] - mem[13,138] - mem[13,139] - mem[13,140]
   - mem[13,141] - mem[13,142] - mem[13,143] - mem[13,144] - mem[13,145]
   - mem[13,146] - mem[13,147] - mem[13,148] - mem[13,149] - mem[13,150]
   - mem[13,151] - mem[13,152] - mem[13,153] - mem[13,154] - mem[13,155]
   - mem[13,156] - mem[13,157] - mem[13,158] - mem[13,159] - mem[13,160]
   - mem[13,161] - mem[13,162] - mem[13,163] - mem[13,164] - mem[13,165]
   - mem[13,166] - mem[13,167] - mem[13,168] - mem[13,169] - mem[13,170]
   - mem[13,171] - mem[13,172] - mem[13,173] - mem[13,174] - mem[13,175]
   - mem[13,176] - mem[13,177] - mem[13,178] - mem[13,179] - mem[13,180]
   - mem[13,181] - mem[13,182] - mem[13,183] - mem[13,184] - mem[13,185]
   - mem[13,186] - mem[13,187] - mem[13,188] - mem[13,189] - mem[13,190]
   - mem[13,191] + mem_tot_cpu14 = 0
 ns_single_cpu14: 3.5 rows_tot_cpu14 - ns_single_cpu14 + pdt_m_rows_cpu14
   <= 0
 capacity_cores_cpu14: cores_sketch_cpu14 + cores_dpdk_cpu14 <= 7
 ns_sketch_cpu14: - ns_single_cpu14 + pdt_nsk_c_cpu14 = 0
 ns_dpdk_cpu14: 7.142857142857143 cores_dpdk_cpu14 - pdt_nsc_dpdk_cpu14
   = -21.42857142857143
 rows_tot_cpu15: - frac[14,0] - frac[14,1] - frac[14,2] - frac[14,3]
   - frac[14,4] - frac[14,5] - frac[14,6] - frac[14,7] - frac[14,8]
   - frac[14,9] - frac[14,10] - frac[14,11] - frac[14,12] - frac[14,13]
   - frac[14,14] - frac[14,15] - frac[14,16] - frac[14,17] - frac[14,18]
   - frac[14,19] - frac[14,20] - frac[14,21] - frac[14,22] - frac[14,23]
   - frac[14,24] - frac[14,25] - frac[14,26] - frac[14,27] - frac[14,28]
   - frac[14,29] - frac[14,30] - frac[14,31] - frac[14,32] - frac[14,33]
   - frac[14,34] - frac[14,35] - frac[14,36] - frac[14,37] - frac[14,38]
   - frac[14,39] - frac[14,40] - frac[14,41] - frac[14,42] - frac[14,43]
   - frac[14,44] - frac[14,45] - frac[14,46] - frac[14,47] - frac[14,48]
   - frac[14,49] - frac[14,50] - frac[14,51] - frac[14,52] - frac[14,53]
   - frac[14,54] - frac[14,55] - frac[14,56] - frac[14,57] - frac[14,58]
   - frac[14,59] - frac[14,60] - frac[14,61] - frac[14,62] - frac[14,63]
   - frac[14,64] - frac[14,65] - frac[14,66] - frac[14,67] - frac[14,68]
   - frac[14,69] - frac[14,70] - frac[14,71] - frac[14,72] - frac[14,73]
   - frac[14,74] - frac[14,75] - frac[14,76] - frac[14,77] - frac[14,78]
   - frac[14,79] - frac[14,80] - frac[14,81] - frac[14,82] - frac[14,83]
   - frac[14,84] - frac[14,85] - frac[14,86] - frac[14,87] - frac[14,88]
   - frac[14,89] - frac[14,90] - frac[14,91] - frac[14,92] - frac[14,93]
   - frac[14,94] - frac[14,95] - frac[14,96] - frac[14,97] - frac[14,98]
   - frac[14,99] - frac[14,100] - frac[14,101] - frac[14,102]
   - frac[14,103] - frac[14,104] - frac[14,105] - frac[14,106]
   - frac[14,107] - frac[14,108] - frac[14,109] - frac[14,110]
   - frac[14,111] - frac[14,112] - frac[14,113] - frac[14,114]
   - frac[14,115] - frac[14,116] - frac[14,117] - frac[14,118]
   - frac[14,119] - frac[14,120] - frac[14,121] - frac[14,122]
   - frac[14,123] - frac[14,124] - frac[14,125] - frac[14,126]
   - frac[14,127] - frac[14,128] - frac[14,129] - frac[14,130]
   - frac[14,131] - frac[14,132] - frac[14,133] - frac[14,134]
   - frac[14,135] - frac[14,136] - frac[14,137] - frac[14,138]
   - frac[14,139] - frac[14,140] - frac[14,141] - frac[14,142]
   - frac[14,143] - frac[14,144] - frac[14,145] - frac[14,146]
   - frac[14,147] - frac[14,148] - frac[14,149] - frac[14,150]
   - frac[14,151] - frac[14,152] - frac[14,153] - frac[14,154]
   - frac[14,155] - frac[14,156] - frac[14,157] - frac[14,158]
   - frac[14,159] - frac[14,160] - frac[14,161] - frac[14,162]
   - frac[14,163] - frac[14,164] - frac[14,165] - frac[14,166]
   - frac[14,167] - frac[14,168] - frac[14,169] - frac[14,170]
   - frac[14,171] - frac[14,172] - frac[14,173] - frac[14,174]
   - frac[14,175] - frac[14,176] - frac[14,177] - frac[14,178]
   - frac[14,179] - frac[14,180] - frac[14,181] - frac[14,182]
   - frac[14,183] - frac[14,184] - frac[14,185] - frac[14,186]
   - frac[14,187] - frac[14,188] - frac[14,189] - frac[14,190]
   - frac[14,191] + rows_tot_cpu15 = 0
 mem_tot_cpu15: - mem[14,0] - mem[14,1] - mem[14,2] - mem[14,3] - mem[14,4]
   - mem[14,5] - mem[14,6] - mem[14,7] - mem[14,8] - mem[14,9] - mem[14,10]
   - mem[14,11] - mem[14,12] - mem[14,13] - mem[14,14] - mem[14,15]
   - mem[14,16] - mem[14,17] - mem[14,18] - mem[14,19] - mem[14,20]
   - mem[14,21] - mem[14,22] - mem[14,23] - mem[14,24] - mem[14,25]
   - mem[14,26] - mem[14,27] - mem[14,28] - mem[14,29] - mem[14,30]
   - mem[14,31] - mem[14,32] - mem[14,33] - mem[14,34] - mem[14,35]
   - mem[14,36] - mem[14,37] - mem[14,38] - mem[14,39] - mem[14,40]
   - mem[14,41] - mem[14,42] - mem[14,43] - mem[14,44] - mem[14,45]
   - mem[14,46] - mem[14,47] - mem[14,48] - mem[14,49] - mem[14,50]
   - mem[14,51] - mem[14,52] - mem[14,53] - mem[14,54] - mem[14,55]
   - mem[14,56] - mem[14,57] - mem[14,58] - mem[14,59] - mem[14,60]
   - mem[14,61] - mem[14,62] - mem[14,63] - mem[14,64] - mem[14,65]
   - mem[14,66] - mem[14,67] - mem[14,68] - mem[14,69] - mem[14,70]
   - mem[14,71] - mem[14,72] - mem[14,73] - mem[14,74] - mem[14,75]
   - mem[14,76] - mem[14,77] - mem[14,78] - mem[14,79] - mem[14,80]
   - mem[14,81] - mem[14,82] - mem[14,83] - mem[14,84] - mem[14,85]
   - mem[14,86] - mem[14,87] - mem[14,88] - mem[14,89] - mem[14,90]
   - mem[14,91] - mem[14,92] - mem[14,93] - mem[14,94] - mem[14,95]
   - mem[14,96] - mem[14,97] - mem[14,98] - mem[14,99] - mem[14,100]
   - mem[14,101] - mem[14,102] - mem[14,103] - mem[14,104] - mem[14,105]
   - mem[14,106] - mem[14,107] - mem[14,108] - mem[14,109] - mem[14,110]
   - mem[14,111] - mem[14,112] - mem[14,113] - mem[14,114] - mem[14,115]
   - mem[14,116] - mem[14,117] - mem[14,118] - mem[14,119] - mem[14,120]
   - mem[14,121] - mem[14,122] - mem[14,123] - mem[14,124] - mem[14,125]
   - mem[14,126] - mem[14,127] - mem[14,128] - mem[14,129] - mem[14,130]
   - mem[14,131] - mem[14,132] - mem[14,133] - mem[14,134] - mem[14,135]
   - mem[14,136] - mem[14,137] - mem[14,138] - mem[14,139] - mem[14,140]
   - mem[14,141] - mem[14,142] - mem[14,143] - mem[14,144] - mem[14,145]
   - mem[14,146] - mem[14,147] - mem[14,148] - mem[14,149] - mem[14,150]
   - mem[14,151] - mem[14,152] - mem[14,153] - mem[14,154] - mem[14,155]
   - mem[14,156] - mem[14,157] - mem[14,158] - mem[14,159] - mem[14,160]
   - mem[14,161] - mem[14,162] - mem[14,163] - mem[14,164] - mem[14,165]
   - mem[14,166] - mem[14,167] - mem[14,168] - mem[14,169] - mem[14,170]
   - mem[14,171] - mem[14,172] - mem[14,173] - mem[14,174] - mem[14,175]
   - mem[14,176] - mem[14,177] - mem[14,178] - mem[14,179] - mem[14,180]
   - mem[14,181] - mem[14,182] - mem[14,183] - mem[14,184] - mem[14,185]
   - mem[14,186] - mem[14,187] - mem[14,188] - mem[14,189] - mem[14,190]
   - mem[14,191] + mem_tot_cpu15 = 0
 ns_single_cpu15: 3.5 rows_tot_cpu15 - ns_single_cpu15 + pdt_m_rows_cpu15
   <= 0
 capacity_cores_cpu15: cores_sketch_cpu15 + cores_dpdk_cpu15 <= 7
 ns_sketch_cpu15: - ns_single_cpu15 + pdt_nsk_c_cpu15 = 0
 ns_dpdk_cpu15: 7.142857142857143 cores_dpdk_cpu15 - pdt_nsc_dpdk_cpu15
   = -21.42857142857143
 rows_tot_cpu16: - frac[15,0] - frac[15,1] - frac[15,2] - frac[15,3]
   - frac[15,4] - frac[15,5] - frac[15,6] - frac[15,7] - frac[15,8]
   - frac[15,9] - frac[15,10] - frac[15,11] - frac[15,12] - frac[15,13]
   - frac[15,14] - frac[15,15] - frac[15,16] - frac[15,17] - frac[15,18]
   - frac[15,19] - frac[15,20] - frac[15,21] - frac[15,22] - frac[15,23]
   - frac[15,24] - frac[15,25] - frac[15,26] - frac[15,27] - frac[15,28]
   - frac[15,29] - frac[15,30] - frac[15,31] - frac[15,32] - frac[15,33]
   - frac[15,34] - frac[15,35] - frac[15,36] - frac[15,37] - frac[15,38]
   - frac[15,39] - frac[15,40] - frac[15,41] - frac[15,42] - frac[15,43]
   - frac[15,44] - frac[15,45] - frac[15,46] - frac[15,47] - frac[15,48]
   - frac[15,49] - frac[15,50] - frac[15,51] - frac[15,52] - frac[15,53]
   - frac[15,54] - frac[15,55] - frac[15,56] - frac[15,57] - frac[15,58]
   - frac[15,59] - frac[15,60] - frac[15,61] - frac[15,62] - frac[15,63]
   - frac[15,64] - frac[15,65] - frac[15,66] - frac[15,67] - frac[15,68]
   - frac[15,69] - frac[15,70] - frac[15,71] - frac[15,72] - frac[15,73]
   - frac[15,74] - frac[15,75] - frac[15,76] - frac[15,77] - frac[15,78]
   - frac[15,79] - frac[15,80] - frac[15,81] - frac[15,82] - frac[15,83]
   - frac[15,84] - frac[15,85] - frac[15,86] - frac[15,87] - frac[15,88]
   - frac[15,89] - frac[15,90] - frac[15,91] - frac[15,92] - frac[15,93]
   - frac[15,94] - frac[15,95] - frac[15,96] - frac[15,97] - frac[15,98]
   - frac[15,99] - frac[15,100] - frac[15,101] - frac[15,102]
   - frac[15,103] - frac[15,104] - frac[15,105] - frac[15,106]
   - frac[15,107] - frac[15,108] - frac[15,109] - frac[15,110]
   - frac[15,111] - frac[15,112] - frac[15,113] - frac[15,114]
   - frac[15,115] - frac[15,116] - frac[15,117] - frac[15,118]
   - frac[15,119] - frac[15,120] - frac[15,121] - frac[15,122]
   - frac[15,123] - frac[15,124] - frac[15,125] - frac[15,126]
   - frac[15,127] - frac[15,128] - frac[15,129] - frac[15,130]
   - frac[15,131] - frac[15,132] - frac[15,133] - frac[15,134]
   - frac[15,135] - frac[15,136] - frac[15,137] - frac[15,138]
   - frac[15,139] - frac[15,140] - frac[15,141] - frac[15,142]
   - frac[15,143] - frac[15,144] - frac[15,145] - frac[15,146]
   - frac[15,147] - frac[15,148] - frac[15,149] - frac[15,150]
   - frac[15,151] - frac[15,152] - frac[15,153] - frac[15,154]
   - frac[15,155] - frac[15,156] - frac[15,157] - frac[15,158]
   - frac[15,159] - frac[15,160] - frac[15,161] - frac[15,162]
   - frac[15,163] - frac[15,164] - frac[15,165] - frac[15,166]
   - frac[15,167] - frac[15,168] - frac[15,169] - frac[15,170]
   - frac[15,171] - frac[15,172] - frac[15,173] - frac[15,174]
   - frac[15,175] - frac[15,176] - frac[15,177] - frac[15,178]
   - frac[15,179] - frac[15,180] - frac[15,181] - frac[15,182]
   - frac[15,183] - frac[15,184] - frac[15,185] - frac[15,186]
   - frac[15,187] - frac[15,188] - frac[15,189] - frac[15,190]
   - frac[15,191] + rows_tot_cpu16 = 0
 mem_tot_cpu16: - mem[15,0] - mem[15,1] - mem[15,2] - mem[15,3] - mem[15,4]
   - mem[15,5] - mem[15,6] - mem[15,7] - mem[15,8] - mem[15,9] - mem[15,10]
   - mem[15,11] - mem[15,12] - mem[15,13] - mem[15,14] - mem[15,15]
   - mem[15,16] - mem[15,17] - mem[15,18] - mem[15,19] - mem[15,20]
   - mem[15,21] - mem[15,22] - mem[15,23] - mem[15,24] - mem[15,25]
   - mem[15,26] - mem[15,27] - mem[15,28] - mem[15,29] - mem[15,30]
   - mem[15,31] - mem[15,32] - mem[15,33] - mem[15,34] - mem[15,35]
   - mem[15,36] - mem[15,37] - mem[15,38] - mem[15,39] - mem[15,40]
   - mem[15,41] - mem[15,42] - mem[15,43] - mem[15,44] - mem[15,45]
   - mem[15,46] - mem[15,47] - mem[15,48] - mem[15,49] - mem[15,50]
   - mem[15,51] - mem[15,52] - mem[15,53] - mem[15,54] - mem[15,55]
   - mem[15,56] - mem[15,57] - mem[15,58] - mem[15,59] - mem[15,60]
   - mem[15,61] - mem[15,62] - mem[15,63] - mem[15,64] - mem[15,65]
   - mem[15,66] - mem[15,67] - mem[15,68] - mem[15,69] - mem[15,70]
   - mem[15,71] - mem[15,72] - mem[15,73] - mem[15,74] - mem[15,75]
   - mem[15,76] - mem[15,77] - mem[15,78] - mem[15,79] - mem[15,80]
   - mem[15,81] - mem[15,82] - mem[15,83] - mem[15,84] - mem[15,85]
   - mem[15,86] - mem[15,87] - mem[15,88] - mem[15,89] - mem[15,90]
   - mem[15,91] - mem[15,92] - mem[15,93] - mem[15,94] - mem[15,95]
   - mem[15,96] - mem[15,97] - mem[15,98] - mem[15,99] - mem[15,100]
   - mem[15,101] - mem[15,102] - mem[15,103] - mem[15,104] - mem[15,105]
   - mem[15,106] - mem[15,107] - mem[15,108] - mem[15,109] - mem[15,110]
   - mem[15,111] - mem[15,112] - mem[15,113] - mem[15,114] - mem[15,115]
   - mem[15,116] - mem[15,117] - mem[15,118] - mem[15,119] - mem[15,120]
   - mem[15,121] - mem[15,122] - mem[15,123] - mem[15,124] - mem[15,125]
   - mem[15,126] - mem[15,127] - mem[15,128] - mem[15,129] - mem[15,130]
   - mem[15,131] - mem[15,132] - mem[15,133] - mem[15,134] - mem[15,135]
   - mem[15,136] - mem[15,137] - mem[15,138] - mem[15,139] - mem[15,140]
   - mem[15,141] - mem[15,142] - mem[15,143] - mem[15,144] - mem[15,145]
   - mem[15,146] - mem[15,147] - mem[15,148] - mem[15,149] - mem[15,150]
   - mem[15,151] - mem[15,152] - mem[15,153] - mem[15,154] - mem[15,155]
   - mem[15,156] - mem[15,157] - mem[15,158] - mem[15,159] - mem[15,160]
   - mem[15,161] - mem[15,162] - mem[15,163] - mem[15,164] - mem[15,165]
   - mem[15,166] - mem[15,167] - mem[15,168] - mem[15,169] - mem[15,170]
   - mem[15,171] - mem[15,172] - mem[15,173] - mem[15,174] - mem[15,175]
   - mem[15,176] - mem[15,177] - mem[15,178] - mem[15,179] - mem[15,180]
   - mem[15,181] - mem[15,182] - mem[15,183] - mem[15,184] - mem[15,185]
   - mem[15,186] - mem[15,187] - mem[15,188] - mem[15,189] - mem[15,190]
   - mem[15,191] + mem_tot_cpu16 = 0
 ns_single_cpu16: 3.5 rows_tot_cpu16 - ns_single_cpu16 + pdt_m_rows_cpu16
   <= 0
 capacity_cores_cpu16: cores_sketch_cpu16 + cores_dpdk_cpu16 <= 7
 ns_sketch_cpu16: - ns_single_cpu16 + pdt_nsk_c_cpu16 = 0
 ns_dpdk_cpu16: 7.142857142857143 cores_dpdk_cpu16 - pdt_nsc_dpdk_cpu16
   = -21.42857142857143
 rows_tot_tor_p41: - frac[16,0] - frac[16,1] - frac[16,2] - frac[16,3]
   - frac[16,4] - frac[16,5] - frac[16,6] - frac[16,7] - frac[16,8]
   - frac[16,9] - frac[16,10] - frac[16,11] - frac[16,12] - frac[16,13]
   - frac[16,14] - frac[16,15] - frac[16,16] - frac[16,17] - frac[16,18]
   - frac[16,19] - frac[16,20] - frac[16,21] - frac[16,22] - frac[16,23]
   - frac[16,24] - frac[16,25] - frac[16,26] - frac[16,27] - frac[16,28]
   - frac[16,29] - frac[16,30] - frac[16,31] - frac[16,32] - frac[16,33]
   - frac[16,34] - frac[16,35] - frac[16,36] - frac[16,37] - frac[16,38]
   - frac[16,39] - frac[16,40] - frac[16,41] - frac[16,42] - frac[16,43]
   - frac[16,44] - frac[16,45] - frac[16,46] - frac[16,47] - frac[16,48]
   - frac[16,49] - frac[16,50] - frac[16,51] - frac[16,52] - frac[16,53]
   - frac[16,54] - frac[16,55] - frac[16,56] - frac[16,57] - frac[16,58]
   - frac[16,59] - frac[16,60] - frac[16,61] - frac[16,62] - frac[16,63]
   - frac[16,64] - frac[16,65] - frac[16,66] - frac[16,67] - frac[16,68]
   - frac[16,69] - frac[16,70] - frac[16,71] - frac[16,72] - frac[16,73]
   - frac[16,74] - frac[16,75] - frac[16,76] - frac[16,77] - frac[16,78]
   - frac[16,79] - frac[16,80] - frac[16,81] - frac[16,82] - frac[16,83]
   - frac[16,84] - frac[16,85] - frac[16,86] - frac[16,87] - frac[16,88]
   - frac[16,89] - frac[16,90] - frac[16,91] - frac[16,92] - frac[16,93]
   - frac[16,94] - frac[16,95] - frac[16,96] - frac[16,97] - frac[16,98]
   - frac[16,99] - frac[16,100] - frac[16,101] - frac[16,102]
   - frac[16,103] - frac[16,104] - frac[16,105] - frac[16,106]
   - frac[16,107] - frac[16,108] - frac[16,109] - frac[16,110]
   - frac[16,111] - frac[16,112] - frac[16,113] - frac[16,114]
   - frac[16,115] - frac[16,116] - frac[16,117] - frac[16,118]
   - frac[16,119] - frac[16,120] - frac[16,121] - frac[16,122]
   - frac[16,123] - frac[16,124] - frac[16,125] - frac[16,126]
   - frac[16,127] - frac[16,128] - frac[16,129] - frac[16,130]
   - frac[16,131] - frac[16,132] - frac[16,133] - frac[16,134]
   - frac[16,135] - frac[16,136] - frac[16,137] - frac[16,138]
   - frac[16,139] - frac[16,140] - frac[16,141] - frac[16,142]
   - frac[16,143] - frac[16,144] - frac[16,145] - frac[16,146]
   - frac[16,147] - frac[16,148] - frac[16,149] - frac[16,150]
   - frac[16,151] - frac[16,152] - frac[16,153] - frac[16,154]
   - frac[16,155] - frac[16,156] - frac[16,157] - frac[16,158]
   - frac[16,159] - frac[16,160] - frac[16,161] - frac[16,162]
   - frac[16,163] - frac[16,164] - frac[16,165] - frac[16,166]
   - frac[16,167] - frac[16,168] - frac[16,169] - frac[16,170]
   - frac[16,171] - frac[16,172] - frac[16,173] - frac[16,174]
   - frac[16,175] - frac[16,176] - frac[16,177] - frac[16,178]
   - frac[16,179] - frac[16,180] - frac[16,181] - frac[16,182]
   - frac[16,183] - frac[16,184] - frac[16,185] - frac[16,186]
   - frac[16,187] - frac[16,188] - frac[16,189] - frac[16,190]
   - frac[16,191] + rows_tot_tor_p41 = 0
 mem_tot_tor_p41: - mem[16,0] - mem[16,1] - mem[16,2] - mem[16,3]
   - mem[16,4] - mem[16,5] - mem[16,6] - mem[16,7] - mem[16,8] - mem[16,9]
   - mem[16,10] - mem[16,11] - mem[16,12] - mem[16,13] - mem[16,14]
   - mem[16,15] - mem[16,16] - mem[16,17] - mem[16,18] - mem[16,19]
   - mem[16,20] - mem[16,21] - mem[16,22] - mem[16,23] - mem[16,24]
   - mem[16,25] - mem[16,26] - mem[16,27] - mem[16,28] - mem[16,29]
   - mem[16,30] - mem[16,31] - mem[16,32] - mem[16,33] - mem[16,34]
   - mem[16,35] - mem[16,36] - mem[16,37] - mem[16,38] - mem[16,39]
   - mem[16,40] - mem[16,41] - mem[16,42] - mem[16,43] - mem[16,44]
   - mem[16,45] - mem[16,46] - mem[16,47] - mem[16,48] - mem[16,49]
   - mem[16,50] - mem[16,51] - mem[16,52] - mem[16,53] - mem[16,54]
   - mem[16,55] - mem[16,56] - mem[16,57] - mem[16,58] - mem[16,59]
   - mem[16,60] - mem[16,61] - mem[16,62] - mem[16,63] - mem[16,64]
   - mem[16,65] - mem[16,66] - mem[16,67] - mem[16,68] - mem[16,69]
   - mem[16,70] - mem[16,71] - mem[16,72] - mem[16,73] - mem[16,74]
   - mem[16,75] - mem[16,76] - mem[16,77] - mem[16,78] - mem[16,79]
   - mem[16,80] - mem[16,81] - mem[16,82] - mem[16,83] - mem[16,84]
   - mem[16,85] - mem[16,86] - mem[16,87] - mem[16,88] - mem[16,89]
   - mem[16,90] - mem[16,91] - mem[16,92] - mem[16,93] - mem[16,94]
   - mem[16,95] - mem[16,96] - mem[16,97] - mem[16,98] - mem[16,99]
   - mem[16,100] - mem[16,101] - mem[16,102] - mem[16,103] - mem[16,104]
   - mem[16,105] - mem[16,106] - mem[16,107] - mem[16,108] - mem[16,109]
   - mem[16,110] - mem[16,111] - mem[16,112] - mem[16,113] - mem[16,114]
   - mem[16,115] - mem[16,116] - mem[16,117] - mem[16,118] - mem[16,119]
   - mem[16,120] - mem[16,121] - mem[16,122] - mem[16,123] - mem[16,124]
   - mem[16,125] - mem[16,126] - mem[16,127] - mem[16,128] - mem[16,129]
   - mem[16,130] - mem[16,131] - mem[16,132] - mem[16,133] - mem[16,134]
   - mem[16,135] - mem[16,136] - mem[16,137] - mem[16,138] - mem[16,139]
   - mem[16,140] - mem[16,141] - mem[16,142] - mem[16,143] - mem[16,144]
   - mem[16,145] - mem[16,146] - mem[16,147] - mem[16,148] - mem[16,149]
   - mem[16,150] - mem[16,151] - mem[16,152] - mem[16,153] - mem[16,154]
   - mem[16,155] - mem[16,156] - mem[16,157] - mem[16,158] - mem[16,159]
   - mem[16,160] - mem[16,161] - mem[16,162] - mem[16,163] - mem[16,164]
   - mem[16,165] - mem[16,166] - mem[16,167] - mem[16,168] - mem[16,169]
   - mem[16,170] - mem[16,171] - mem[16,172] - mem[16,173] - mem[16,174]
   - mem[16,175] - mem[16,176] - mem[16,177] - mem[16,178] - mem[16,179]
   - mem[16,180] - mem[16,181] - mem[16,182] - mem[16,183] - mem[16,184]
   - mem[16,185] - mem[16,186] - mem[16,187] - mem[16,188] - mem[16,189]
   - mem[16,190] - mem[16,191] + mem_tot_tor_p41 = 0
 capacity_mem_par_tor_p41: mem[16,0] <= 48
 capacity_mem_par_tor_p41: mem[16,1] <= 48
 capacity_mem_par_tor_p41: mem[16,2] <= 48
 capacity_mem_par_tor_p41: mem[16,3] <= 48
 capacity_mem_par_tor_p41: mem[16,4] <= 48
 capacity_mem_par_tor_p41: mem[16,5] <= 48
 capacity_mem_par_tor_p41: mem[16,6] <= 48
 capacity_mem_par_tor_p41: mem[16,7] <= 48
 capacity_mem_par_tor_p41: mem[16,8] <= 48
 capacity_mem_par_tor_p41: mem[16,9] <= 48
 capacity_mem_par_tor_p41: mem[16,10] <= 48
 capacity_mem_par_tor_p41: mem[16,11] <= 48
 capacity_mem_par_tor_p41: mem[16,12] <= 48
 capacity_mem_par_tor_p41: mem[16,13] <= 48
 capacity_mem_par_tor_p41: mem[16,14] <= 48
 capacity_mem_par_tor_p41: mem[16,15] <= 48
 capacity_mem_par_tor_p41: mem[16,16] <= 48
 capacity_mem_par_tor_p41: mem[16,17] <= 48
 capacity_mem_par_tor_p41: mem[16,18] <= 48
 capacity_mem_par_tor_p41: mem[16,19] <= 48
 capacity_mem_par_tor_p41: mem[16,20] <= 48
 capacity_mem_par_tor_p41: mem[16,21] <= 48
 capacity_mem_par_tor_p41: mem[16,22] <= 48
 capacity_mem_par_tor_p41: mem[16,23] <= 48
 capacity_mem_par_tor_p41: mem[16,24] <= 48
 capacity_mem_par_tor_p41: mem[16,25] <= 48
 capacity_mem_par_tor_p41: mem[16,26] <= 48
 capacity_mem_par_tor_p41: mem[16,27] <= 48
 capacity_mem_par_tor_p41: mem[16,28] <= 48
 capacity_mem_par_tor_p41: mem[16,29] <= 48
 capacity_mem_par_tor_p41: mem[16,30] <= 48
 capacity_mem_par_tor_p41: mem[16,31] <= 48
 capacity_mem_par_tor_p41: mem[16,32] <= 48
 capacity_mem_par_tor_p41: mem[16,33] <= 48
 capacity_mem_par_tor_p41: mem[16,34] <= 48
 capacity_mem_par_tor_p41: mem[16,35] <= 48
 capacity_mem_par_tor_p41: mem[16,36] <= 48
 capacity_mem_par_tor_p41: mem[16,37] <= 48
 capacity_mem_par_tor_p41: mem[16,38] <= 48
 capacity_mem_par_tor_p41: mem[16,39] <= 48
 capacity_mem_par_tor_p41: mem[16,40] <= 48
 capacity_mem_par_tor_p41: mem[16,41] <= 48
 capacity_mem_par_tor_p41: mem[16,42] <= 48
 capacity_mem_par_tor_p41: mem[16,43] <= 48
 capacity_mem_par_tor_p41: mem[16,44] <= 48
 capacity_mem_par_tor_p41: mem[16,45] <= 48
 capacity_mem_par_tor_p41: mem[16,46] <= 48
 capacity_mem_par_tor_p41: mem[16,47] <= 48
 capacity_mem_par_tor_p41: mem[16,48] <= 48
 capacity_mem_par_tor_p41: mem[16,49] <= 48
 capacity_mem_par_tor_p41: mem[16,50] <= 48
 capacity_mem_par_tor_p41: mem[16,51] <= 48
 capacity_mem_par_tor_p41: mem[16,52] <= 48
 capacity_mem_par_tor_p41: mem[16,53] <= 48
 capacity_mem_par_tor_p41: mem[16,54] <= 48
 capacity_mem_par_tor_p41: mem[16,55] <= 48
 capacity_mem_par_tor_p41: mem[16,56] <= 48
 capacity_mem_par_tor_p41: mem[16,57] <= 48
 capacity_mem_par_tor_p41: mem[16,58] <= 48
 capacity_mem_par_tor_p41: mem[16,59] <= 48
 capacity_mem_par_tor_p41: mem[16,60] <= 48
 capacity_mem_par_tor_p41: mem[16,61] <= 48
 capacity_mem_par_tor_p41: mem[16,62] <= 48
 capacity_mem_par_tor_p41: mem[16,63] <= 48
 capacity_mem_par_tor_p41: mem[16,64] <= 48
 capacity_mem_par_tor_p41: mem[16,65] <= 48
 capacity_mem_par_tor_p41: mem[16,66] <= 48
 capacity_mem_par_tor_p41: mem[16,67] <= 48
 capacity_mem_par_tor_p41: mem[16,68] <= 48
 capacity_mem_par_tor_p41: mem[16,69] <= 48
 capacity_mem_par_tor_p41: mem[16,70] <= 48
 capacity_mem_par_tor_p41: mem[16,71] <= 48
 capacity_mem_par_tor_p41: mem[16,72] <= 48
 capacity_mem_par_tor_p41: mem[16,73] <= 48
 capacity_mem_par_tor_p41: mem[16,74] <= 48
 capacity_mem_par_tor_p41: mem[16,75] <= 48
 capacity_mem_par_tor_p41: mem[16,76] <= 48
 capacity_mem_par_tor_p41: mem[16,77] <= 48
 capacity_mem_par_tor_p41: mem[16,78] <= 48
 capacity_mem_par_tor_p41: mem[16,79] <= 48
 capacity_mem_par_tor_p41: mem[16,80] <= 48
 capacity_mem_par_tor_p41: mem[16,81] <= 48
 capacity_mem_par_tor_p41: mem[16,82] <= 48
 capacity_mem_par_tor_p41: mem[16,83] <= 48
 capacity_mem_par_tor_p41: mem[16,84] <= 48
 capacity_mem_par_tor_p41: mem[16,85] <= 48
 capacity_mem_par_tor_p41: mem[16,86] <= 48
 capacity_mem_par_tor_p41: mem[16,87] <= 48
 capacity_mem_par_tor_p41: mem[16,88] <= 48
 capacity_mem_par_tor_p41: mem[16,89] <= 48
 capacity_mem_par_tor_p41: mem[16,90] <= 48
 capacity_mem_par_tor_p41: mem[16,91] <= 48
 capacity_mem_par_tor_p41: mem[16,92] <= 48
 capacity_mem_par_tor_p41: mem[16,93] <= 48
 capacity_mem_par_tor_p41: mem[16,94] <= 48
 capacity_mem_par_tor_p41: mem[16,95] <= 48
 capacity_mem_par_tor_p41: mem[16,96] <= 48
 capacity_mem_par_tor_p41: mem[16,97] <= 48
 capacity_mem_par_tor_p41: mem[16,98] <= 48
 capacity_mem_par_tor_p41: mem[16,99] <= 48
 capacity_mem_par_tor_p41: mem[16,100] <= 48
 capacity_mem_par_tor_p41: mem[16,101] <= 48
 capacity_mem_par_tor_p41: mem[16,102] <= 48
 capacity_mem_par_tor_p41: mem[16,103] <= 48
 capacity_mem_par_tor_p41: mem[16,104] <= 48
 capacity_mem_par_tor_p41: mem[16,105] <= 48
 capacity_mem_par_tor_p41: mem[16,106] <= 48
 capacity_mem_par_tor_p41: mem[16,107] <= 48
 capacity_mem_par_tor_p41: mem[16,108] <= 48
 capacity_mem_par_tor_p41: mem[16,109] <= 48
 capacity_mem_par_tor_p41: mem[16,110] <= 48
 capacity_mem_par_tor_p41: mem[16,111] <= 48
 capacity_mem_par_tor_p41: mem[16,112] <= 48
 capacity_mem_par_tor_p41: mem[16,113] <= 48
 capacity_mem_par_tor_p41: mem[16,114] <= 48
 capacity_mem_par_tor_p41: mem[16,115] <= 48
 capacity_mem_par_tor_p41: mem[16,116] <= 48
 capacity_mem_par_tor_p41: mem[16,117] <= 48
 capacity_mem_par_tor_p41: mem[16,118] <= 48
 capacity_mem_par_tor_p41: mem[16,119] <= 48
 capacity_mem_par_tor_p41: mem[16,120] <= 48
 capacity_mem_par_tor_p41: mem[16,121] <= 48
 capacity_mem_par_tor_p41: mem[16,122] <= 48
 capacity_mem_par_tor_p41: mem[16,123] <= 48
 capacity_mem_par_tor_p41: mem[16,124] <= 48
 capacity_mem_par_tor_p41: mem[16,125] <= 48
 capacity_mem_par_tor_p41: mem[16,126] <= 48
 capacity_mem_par_tor_p41: mem[16,127] <= 48
 capacity_mem_par_tor_p41: mem[16,128] <= 48
 capacity_mem_par_tor_p41: mem[16,129] <= 48
 capacity_mem_par_tor_p41: mem[16,130] <= 48
 capacity_mem_par_tor_p41: mem[16,131] <= 48
 capacity_mem_par_tor_p41: mem[16,132] <= 48
 capacity_mem_par_tor_p41: mem[16,133] <= 48
 capacity_mem_par_tor_p41: mem[16,134] <= 48
 capacity_mem_par_tor_p41: mem[16,135] <= 48
 capacity_mem_par_tor_p41: mem[16,136] <= 48
 capacity_mem_par_tor_p41: mem[16,137] <= 48
 capacity_mem_par_tor_p41: mem[16,138] <= 48
 capacity_mem_par_tor_p41: mem[16,139] <= 48
 capacity_mem_par_tor_p41: mem[16,140] <= 48
 capacity_mem_par_tor_p41: mem[16,141] <= 48
 capacity_mem_par_tor_p41: mem[16,142] <= 48
 capacity_mem_par_tor_p41: mem[16,143] <= 48
 capacity_mem_par_tor_p41: mem[16,144] <= 48
 capacity_mem_par_tor_p41: mem[16,145] <= 48
 capacity_mem_par_tor_p41: mem[16,146] <= 48
 capacity_mem_par_tor_p41: mem[16,147] <= 48
 capacity_mem_par_tor_p41: mem[16,148] <= 48
 capacity_mem_par_tor_p41: mem[16,149] <= 48
 capacity_mem_par_tor_p41: mem[16,150] <= 48
 capacity_mem_par_tor_p41: mem[16,151] <= 48
 capacity_mem_par_tor_p41: mem[16,152] <= 48
 capacity_mem_par_tor_p41: mem[16,153] <= 48
 capacity_mem_par_tor_p41: mem[16,154] <= 48
 capacity_mem_par_tor_p41: mem[16,155] <= 48
 capacity_mem_par_tor_p41: mem[16,156] <= 48
 capacity_mem_par_tor_p41: mem[16,157] <= 48
 capacity_mem_par_tor_p41: mem[16,158] <= 48
 capacity_mem_par_tor_p41: mem[16,159] <= 48
 capacity_mem_par_tor_p41: mem[16,160] <= 48
 capacity_mem_par_tor_p41: mem[16,161] <= 48
 capacity_mem_par_tor_p41: mem[16,162] <= 48
 capacity_mem_par_tor_p41: mem[16,163] <= 48
 capacity_mem_par_tor_p41: mem[16,164] <= 48
 capacity_mem_par_tor_p41: mem[16,165] <= 48
 capacity_mem_par_tor_p41: mem[16,166] <= 48
 capacity_mem_par_tor_p41: mem[16,167] <= 48
 capacity_mem_par_tor_p41: mem[16,168] <= 48
 capacity_mem_par_tor_p41: mem[16,169] <= 48
 capacity_mem_par_tor_p41: mem[16,170] <= 48
 capacity_mem_par_tor_p41: mem[16,171] <= 48
 capacity_mem_par_tor_p41: mem[16,172] <= 48
 capacity_mem_par_tor_p41: mem[16,173] <= 48
 capacity_mem_par_tor_p41: mem[16,174] <= 48
 capacity_mem_par_tor_p41: mem[16,175] <= 48
 capacity_mem_par_tor_p41: mem[16,176] <= 48
 capacity_mem_par_tor_p41: mem[16,177] <= 48
 capacity_mem_par_tor_p41: mem[16,178] <= 48
 capacity_mem_par_tor_p41: mem[16,179] <= 48
 capacity_mem_par_tor_p41: mem[16,180] <= 48
 capacity_mem_par_tor_p41: mem[16,181] <= 48
 capacity_mem_par_tor_p41: mem[16,182] <= 48
 capacity_mem_par_tor_p41: mem[16,183] <= 48
 capacity_mem_par_tor_p41: mem[16,184] <= 48
 capacity_mem_par_tor_p41: mem[16,185] <= 48
 capacity_mem_par_tor_p41: mem[16,186] <= 48
 capacity_mem_par_tor_p41: mem[16,187] <= 48
 capacity_mem_par_tor_p41: mem[16,188] <= 48
 capacity_mem_par_tor_p41: mem[16,189] <= 48
 capacity_mem_par_tor_p41: mem[16,190] <= 48
 capacity_mem_par_tor_p41: mem[16,191] <= 48
 ns_tor_p41: ns_tor_p41 = 6.756756756756757
 rows_tot_tor_p42: - frac[17,0] - frac[17,1] - frac[17,2] - frac[17,3]
   - frac[17,4] - frac[17,5] - frac[17,6] - frac[17,7] - frac[17,8]
   - frac[17,9] - frac[17,10] - frac[17,11] - frac[17,12] - frac[17,13]
   - frac[17,14] - frac[17,15] - frac[17,16] - frac[17,17] - frac[17,18]
   - frac[17,19] - frac[17,20] - frac[17,21] - frac[17,22] - frac[17,23]
   - frac[17,24] - frac[17,25] - frac[17,26] - frac[17,27] - frac[17,28]
   - frac[17,29] - frac[17,30] - frac[17,31] - frac[17,32] - frac[17,33]
   - frac[17,34] - frac[17,35] - frac[17,36] - frac[17,37] - frac[17,38]
   - frac[17,39] - frac[17,40] - frac[17,41] - frac[17,42] - frac[17,43]
   - frac[17,44] - frac[17,45] - frac[17,46] - frac[17,47] - frac[17,48]
   - frac[17,49] - frac[17,50] - frac[17,51] - frac[17,52] - frac[17,53]
   - frac[17,54] - frac[17,55] - frac[17,56] - frac[17,57] - frac[17,58]
   - frac[17,59] - frac[17,60] - frac[17,61] - frac[17,62] - frac[17,63]
   - frac[17,64] - frac[17,65] - frac[17,66] - frac[17,67] - frac[17,68]
   - frac[17,69] - frac[17,70] - frac[17,71] - frac[17,72] - frac[17,73]
   - frac[17,74] - frac[17,75] - frac[17,76] - frac[17,77] - frac[17,78]
   - frac[17,79] - frac[17,80] - frac[17,81] - frac[17,82] - frac[17,83]
   - frac[17,84] - frac[17,85] - frac[17,86] - frac[17,87] - frac[17,88]
   - frac[17,89] - frac[17,90] - frac[17,91] - frac[17,92] - frac[17,93]
   - frac[17,94] - frac[17,95] - frac[17,96] - frac[17,97] - frac[17,98]
   - frac[17,99] - frac[17,100] - frac[17,101] - frac[17,102]
   - frac[17,103] - frac[17,104] - frac[17,105] - frac[17,106]
   - frac[17,107] - frac[17,108] - frac[17,109] - frac[17,110]
   - frac[17,111] - frac[17,112] - frac[17,113] - frac[17,114]
   - frac[17,115] - frac[17,116] - frac[17,117] - frac[17,118]
   - frac[17,119] - frac[17,120] - frac[17,121] - frac[17,122]
   - frac[17,123] - frac[17,124] - frac[17,125] - frac[17,126]
   - frac[17,127] - frac[17,128] - frac[17,129] - frac[17,130]
   - frac[17,131] - frac[17,132] - frac[17,133] - frac[17,134]
   - frac[17,135] - frac[17,136] - frac[17,137] - frac[17,138]
   - frac[17,139] - frac[17,140] - frac[17,141] - frac[17,142]
   - frac[17,143] - frac[17,144] - frac[17,145] - frac[17,146]
   - frac[17,147] - frac[17,148] - frac[17,149] - frac[17,150]
   - frac[17,151] - frac[17,152] - frac[17,153] - frac[17,154]
   - frac[17,155] - frac[17,156] - frac[17,157] - frac[17,158]
   - frac[17,159] - frac[17,160] - frac[17,161] - frac[17,162]
   - frac[17,163] - frac[17,164] - frac[17,165] - frac[17,166]
   - frac[17,167] - frac[17,168] - frac[17,169] - frac[17,170]
   - frac[17,171] - frac[17,172] - frac[17,173] - frac[17,174]
   - frac[17,175] - frac[17,176] - frac[17,177] - frac[17,178]
   - frac[17,179] - frac[17,180] - frac[17,181] - frac[17,182]
   - frac[17,183] - frac[17,184] - frac[17,185] - frac[17,186]
   - frac[17,187] - frac[17,188] - frac[17,189] - frac[17,190]
   - frac[17,191] + rows_tot_tor_p42 = 0
 mem_tot_tor_p42: - mem[17,0] - mem[17,1] - mem[17,2] - mem[17,3]
   - mem[17,4] - mem[17,5] - mem[17,6] - mem[17,7] - mem[17,8] - mem[17,9]
   - mem[17,10] - mem[17,11] - mem[17,12] - mem[17,13] - mem[17,14]
   - mem[17,15] - mem[17,16] - mem[17,17] - mem[17,18] - mem[17,19]
   - mem[17,20] - mem[17,21] - mem[17,22] - mem[17,23] - mem[17,24]
   - mem[17,25] - mem[17,26] - mem[17,27] - mem[17,28] - mem[17,29]
   - mem[17,30] - mem[17,31] - mem[17,32] - mem[17,33] - mem[17,34]
   - mem[17,35] - mem[17,36] - mem[17,37] - mem[17,38] - mem[17,39]
   - mem[17,40] - mem[17,41] - mem[17,42] - mem[17,43] - mem[17,44]
   - mem[17,45] - mem[17,46] - mem[17,47] - mem[17,48] - mem[17,49]
   - mem[17,50] - mem[17,51] - mem[17,52] - mem[17,53] - mem[17,54]
   - mem[17,55] - mem[17,56] - mem[17,57] - mem[17,58] - mem[17,59]
   - mem[17,60] - mem[17,61] - mem[17,62] - mem[17,63] - mem[17,64]
   - mem[17,65] - mem[17,66] - mem[17,67] - mem[17,68] - mem[17,69]
   - mem[17,70] - mem[17,71] - mem[17,72] - mem[17,73] - mem[17,74]
   - mem[17,75] - mem[17,76] - mem[17,77] - mem[17,78] - mem[17,79]
   - mem[17,80] - mem[17,81] - mem[17,82] - mem[17,83] - mem[17,84]
   - mem[17,85] - mem[17,86] - mem[17,87] - mem[17,88] - mem[17,89]
   - mem[17,90] - mem[17,91] - mem[17,92] - mem[17,93] - mem[17,94]
   - mem[17,95] - mem[17,96] - mem[17,97] - mem[17,98] - mem[17,99]
   - mem[17,100] - mem[17,101] - mem[17,102] - mem[17,103] - mem[17,104]
   - mem[17,105] - mem[17,106] - mem[17,107] - mem[17,108] - mem[17,109]
   - mem[17,110] - mem[17,111] - mem[17,112] - mem[17,113] - mem[17,114]
   - mem[17,115] - mem[17,116] - mem[17,117] - mem[17,118] - mem[17,119]
   - mem[17,120] - mem[17,121] - mem[17,122] - mem[17,123] - mem[17,124]
   - mem[17,125] - mem[17,126] - mem[17,127] - mem[17,128] - mem[17,129]
   - mem[17,130] - mem[17,131] - mem[17,132] - mem[17,133] - mem[17,134]
   - mem[17,135] - mem[17,136] - mem[17,137] - mem[17,138] - mem[17,139]
   - mem[17,140] - mem[17,141] - mem[17,142] - mem[17,143] - mem[17,144]
   - mem[17,145] - mem[17,146] - mem[17,147] - mem[17,148] - mem[17,149]
   - mem[17,150] - mem[17,151] - mem[17,152] - mem[17,153] - mem[17,154]
   - mem[17,155] - mem[17,156] - mem[17,157] - mem[17,158] - mem[17,159]
   - mem[17,160] - mem[17,161] - mem[17,162] - mem[17,163] - mem[17,164]
   - mem[17,165] - mem[17,166] - mem[17,167] - mem[17,168] - mem[17,169]
   - mem[17,170] - mem[17,171] - mem[17,172] - mem[17,173] - mem[17,174]
   - mem[17,175] - mem[17,176] - mem[17,177] - mem[17,178] - mem[17,179]
   - mem[17,180] - mem[17,181] - mem[17,182] - mem[17,183] - mem[17,184]
   - mem[17,185] - mem[17,186] - mem[17,187] - mem[17,188] - mem[17,189]
   - mem[17,190] - mem[17,191] + mem_tot_tor_p42 = 0
 capacity_mem_par_tor_p42: mem[17,0] <= 48
 capacity_mem_par_tor_p42: mem[17,1] <= 48
 capacity_mem_par_tor_p42: mem[17,2] <= 48
 capacity_mem_par_tor_p42: mem[17,3] <= 48
 capacity_mem_par_tor_p42: mem[17,4] <= 48
 capacity_mem_par_tor_p42: mem[17,5] <= 48
 capacity_mem_par_tor_p42: mem[17,6] <= 48
 capacity_mem_par_tor_p42: mem[17,7] <= 48
 capacity_mem_par_tor_p42: mem[17,8] <= 48
 capacity_mem_par_tor_p42: mem[17,9] <= 48
 capacity_mem_par_tor_p42: mem[17,10] <= 48
 capacity_mem_par_tor_p42: mem[17,11] <= 48
 capacity_mem_par_tor_p42: mem[17,12] <= 48
 capacity_mem_par_tor_p42: mem[17,13] <= 48
 capacity_mem_par_tor_p42: mem[17,14] <= 48
 capacity_mem_par_tor_p42: mem[17,15] <= 48
 capacity_mem_par_tor_p42: mem[17,16] <= 48
 capacity_mem_par_tor_p42: mem[17,17] <= 48
 capacity_mem_par_tor_p42: mem[17,18] <= 48
 capacity_mem_par_tor_p42: mem[17,19] <= 48
 capacity_mem_par_tor_p42: mem[17,20] <= 48
 capacity_mem_par_tor_p42: mem[17,21] <= 48
 capacity_mem_par_tor_p42: mem[17,22] <= 48
 capacity_mem_par_tor_p42: mem[17,23] <= 48
 capacity_mem_par_tor_p42: mem[17,24] <= 48
 capacity_mem_par_tor_p42: mem[17,25] <= 48
 capacity_mem_par_tor_p42: mem[17,26] <= 48
 capacity_mem_par_tor_p42: mem[17,27] <= 48
 capacity_mem_par_tor_p42: mem[17,28] <= 48
 capacity_mem_par_tor_p42: mem[17,29] <= 48
 capacity_mem_par_tor_p42: mem[17,30] <= 48
 capacity_mem_par_tor_p42: mem[17,31] <= 48
 capacity_mem_par_tor_p42: mem[17,32] <= 48
 capacity_mem_par_tor_p42: mem[17,33] <= 48
 capacity_mem_par_tor_p42: mem[17,34] <= 48
 capacity_mem_par_tor_p42: mem[17,35] <= 48
 capacity_mem_par_tor_p42: mem[17,36] <= 48
 capacity_mem_par_tor_p42: mem[17,37] <= 48
 capacity_mem_par_tor_p42: mem[17,38] <= 48
 capacity_mem_par_tor_p42: mem[17,39] <= 48
 capacity_mem_par_tor_p42: mem[17,40] <= 48
 capacity_mem_par_tor_p42: mem[17,41] <= 48
 capacity_mem_par_tor_p42: mem[17,42] <= 48
 capacity_mem_par_tor_p42: mem[17,43] <= 48
 capacity_mem_par_tor_p42: mem[17,44] <= 48
 capacity_mem_par_tor_p42: mem[17,45] <= 48
 capacity_mem_par_tor_p42: mem[17,46] <= 48
 capacity_mem_par_tor_p42: mem[17,47] <= 48
 capacity_mem_par_tor_p42: mem[17,48] <= 48
 capacity_mem_par_tor_p42: mem[17,49] <= 48
 capacity_mem_par_tor_p42: mem[17,50] <= 48
 capacity_mem_par_tor_p42: mem[17,51] <= 48
 capacity_mem_par_tor_p42: mem[17,52] <= 48
 capacity_mem_par_tor_p42: mem[17,53] <= 48
 capacity_mem_par_tor_p42: mem[17,54] <= 48
 capacity_mem_par_tor_p42: mem[17,55] <= 48
 capacity_mem_par_tor_p42: mem[17,56] <= 48
 capacity_mem_par_tor_p42: mem[17,57] <= 48
 capacity_mem_par_tor_p42: mem[17,58] <= 48
 capacity_mem_par_tor_p42: mem[17,59] <= 48
 capacity_mem_par_tor_p42: mem[17,60] <= 48
 capacity_mem_par_tor_p42: mem[17,61] <= 48
 capacity_mem_par_tor_p42: mem[17,62] <= 48
 capacity_mem_par_tor_p42: mem[17,63] <= 48
 capacity_mem_par_tor_p42: mem[17,64] <= 48
 capacity_mem_par_tor_p42: mem[17,65] <= 48
 capacity_mem_par_tor_p42: mem[17,66] <= 48
 capacity_mem_par_tor_p42: mem[17,67] <= 48
 capacity_mem_par_tor_p42: mem[17,68] <= 48
 capacity_mem_par_tor_p42: mem[17,69] <= 48
 capacity_mem_par_tor_p42: mem[17,70] <= 48
 capacity_mem_par_tor_p42: mem[17,71] <= 48
 capacity_mem_par_tor_p42: mem[17,72] <= 48
 capacity_mem_par_tor_p42: mem[17,73] <= 48
 capacity_mem_par_tor_p42: mem[17,74] <= 48
 capacity_mem_par_tor_p42: mem[17,75] <= 48
 capacity_mem_par_tor_p42: mem[17,76] <= 48
 capacity_mem_par_tor_p42: mem[17,77] <= 48
 capacity_mem_par_tor_p42: mem[17,78] <= 48
 capacity_mem_par_tor_p42: mem[17,79] <= 48
 capacity_mem_par_tor_p42: mem[17,80] <= 48
 capacity_mem_par_tor_p42: mem[17,81] <= 48
 capacity_mem_par_tor_p42: mem[17,82] <= 48
 capacity_mem_par_tor_p42: mem[17,83] <= 48
 capacity_mem_par_tor_p42: mem[17,84] <= 48
 capacity_mem_par_tor_p42: mem[17,85] <= 48
 capacity_mem_par_tor_p42: mem[17,86] <= 48
 capacity_mem_par_tor_p42: mem[17,87] <= 48
 capacity_mem_par_tor_p42: mem[17,88] <= 48
 capacity_mem_par_tor_p42: mem[17,89] <= 48
 capacity_mem_par_tor_p42: mem[17,90] <= 48
 capacity_mem_par_tor_p42: mem[17,91] <= 48
 capacity_mem_par_tor_p42: mem[17,92] <= 48
 capacity_mem_par_tor_p42: mem[17,93] <= 48
 capacity_mem_par_tor_p42: mem[17,94] <= 48
 capacity_mem_par_tor_p42: mem[17,95] <= 48
 capacity_mem_par_tor_p42: mem[17,96] <= 48
 capacity_mem_par_tor_p42: mem[17,97] <= 48
 capacity_mem_par_tor_p42: mem[17,98] <= 48
 capacity_mem_par_tor_p42: mem[17,99] <= 48
 capacity_mem_par_tor_p42: mem[17,100] <= 48
 capacity_mem_par_tor_p42: mem[17,101] <= 48
 capacity_mem_par_tor_p42: mem[17,102] <= 48
 capacity_mem_par_tor_p42: mem[17,103] <= 48
 capacity_mem_par_tor_p42: mem[17,104] <= 48
 capacity_mem_par_tor_p42: mem[17,105] <= 48
 capacity_mem_par_tor_p42: mem[17,106] <= 48
 capacity_mem_par_tor_p42: mem[17,107] <= 48
 capacity_mem_par_tor_p42: mem[17,108] <= 48
 capacity_mem_par_tor_p42: mem[17,109] <= 48
 capacity_mem_par_tor_p42: mem[17,110] <= 48
 capacity_mem_par_tor_p42: mem[17,111] <= 48
 capacity_mem_par_tor_p42: mem[17,112] <= 48
 capacity_mem_par_tor_p42: mem[17,113] <= 48
 capacity_mem_par_tor_p42: mem[17,114] <= 48
 capacity_mem_par_tor_p42: mem[17,115] <= 48
 capacity_mem_par_tor_p42: mem[17,116] <= 48
 capacity_mem_par_tor_p42: mem[17,117] <= 48
 capacity_mem_par_tor_p42: mem[17,118] <= 48
 capacity_mem_par_tor_p42: mem[17,119] <= 48
 capacity_mem_par_tor_p42: mem[17,120] <= 48
 capacity_mem_par_tor_p42: mem[17,121] <= 48
 capacity_mem_par_tor_p42: mem[17,122] <= 48
 capacity_mem_par_tor_p42: mem[17,123] <= 48
 capacity_mem_par_tor_p42: mem[17,124] <= 48
 capacity_mem_par_tor_p42: mem[17,125] <= 48
 capacity_mem_par_tor_p42: mem[17,126] <= 48
 capacity_mem_par_tor_p42: mem[17,127] <= 48
 capacity_mem_par_tor_p42: mem[17,128] <= 48
 capacity_mem_par_tor_p42: mem[17,129] <= 48
 capacity_mem_par_tor_p42: mem[17,130] <= 48
 capacity_mem_par_tor_p42: mem[17,131] <= 48
 capacity_mem_par_tor_p42: mem[17,132] <= 48
 capacity_mem_par_tor_p42: mem[17,133] <= 48
 capacity_mem_par_tor_p42: mem[17,134] <= 48
 capacity_mem_par_tor_p42: mem[17,135] <= 48
 capacity_mem_par_tor_p42: mem[17,136] <= 48
 capacity_mem_par_tor_p42: mem[17,137] <= 48
 capacity_mem_par_tor_p42: mem[17,138] <= 48
 capacity_mem_par_tor_p42: mem[17,139] <= 48
 capacity_mem_par_tor_p42: mem[17,140] <= 48
 capacity_mem_par_tor_p42: mem[17,141] <= 48
 capacity_mem_par_tor_p42: mem[17,142] <= 48
 capacity_mem_par_tor_p42: mem[17,143] <= 48
 capacity_mem_par_tor_p42: mem[17,144] <= 48
 capacity_mem_par_tor_p42: mem[17,145] <= 48
 capacity_mem_par_tor_p42: mem[17,146] <= 48
 capacity_mem_par_tor_p42: mem[17,147] <= 48
 capacity_mem_par_tor_p42: mem[17,148] <= 48
 capacity_mem_par_tor_p42: mem[17,149] <= 48
 capacity_mem_par_tor_p42: mem[17,150] <= 48
 capacity_mem_par_tor_p42: mem[17,151] <= 48
 capacity_mem_par_tor_p42: mem[17,152] <= 48
 capacity_mem_par_tor_p42: mem[17,153] <= 48
 capacity_mem_par_tor_p42: mem[17,154] <= 48
 capacity_mem_par_tor_p42: mem[17,155] <= 48
 capacity_mem_par_tor_p42: mem[17,156] <= 48
 capacity_mem_par_tor_p42: mem[17,157] <= 48
 capacity_mem_par_tor_p42: mem[17,158] <= 48
 capacity_mem_par_tor_p42: mem[17,159] <= 48
 capacity_mem_par_tor_p42: mem[17,160] <= 48
 capacity_mem_par_tor_p42: mem[17,161] <= 48
 capacity_mem_par_tor_p42: mem[17,162] <= 48
 capacity_mem_par_tor_p42: mem[17,163] <= 48
 capacity_mem_par_tor_p42: mem[17,164] <= 48
 capacity_mem_par_tor_p42: mem[17,165] <= 48
 capacity_mem_par_tor_p42: mem[17,166] <= 48
 capacity_mem_par_tor_p42: mem[17,167] <= 48
 capacity_mem_par_tor_p42: mem[17,168] <= 48
 capacity_mem_par_tor_p42: mem[17,169] <= 48
 capacity_mem_par_tor_p42: mem[17,170] <= 48
 capacity_mem_par_tor_p42: mem[17,171] <= 48
 capacity_mem_par_tor_p42: mem[17,172] <= 48
 capacity_mem_par_tor_p42: mem[17,173] <= 48
 capacity_mem_par_tor_p42: mem[17,174] <= 48
 capacity_mem_par_tor_p42: mem[17,175] <= 48
 capacity_mem_par_tor_p42: mem[17,176] <= 48
 capacity_mem_par_tor_p42: mem[17,177] <= 48
 capacity_mem_par_tor_p42: mem[17,178] <= 48
 capacity_mem_par_tor_p42: mem[17,179] <= 48
 capacity_mem_par_tor_p42: mem[17,180] <= 48
 capacity_mem_par_tor_p42: mem[17,181] <= 48
 capacity_mem_par_tor_p42: mem[17,182] <= 48
 capacity_mem_par_tor_p42: mem[17,183] <= 48
 capacity_mem_par_tor_p42: mem[17,184] <= 48
 capacity_mem_par_tor_p42: mem[17,185] <= 48
 capacity_mem_par_tor_p42: mem[17,186] <= 48
 capacity_mem_par_tor_p42: mem[17,187] <= 48
 capacity_mem_par_tor_p42: mem[17,188] <= 48
 capacity_mem_par_tor_p42: mem[17,189] <= 48
 capacity_mem_par_tor_p42: mem[17,190] <= 48
 capacity_mem_par_tor_p42: mem[17,191] <= 48
 ns_tor_p42: ns_tor_p42 = 6.756756756756757
 rows_tot_tor_p43: - frac[18,0] - frac[18,1] - frac[18,2] - frac[18,3]
   - frac[18,4] - frac[18,5] - frac[18,6] - frac[18,7] - frac[18,8]
   - frac[18,9] - frac[18,10] - frac[18,11] - frac[18,12] - frac[18,13]
   - frac[18,14] - frac[18,15] - frac[18,16] - frac[18,17] - frac[18,18]
   - frac[18,19] - frac[18,20] - frac[18,21] - frac[18,22] - frac[18,23]
   - frac[18,24] - frac[18,25] - frac[18,26] - frac[18,27] - frac[18,28]
   - frac[18,29] - frac[18,30] - frac[18,31] - frac[18,32] - frac[18,33]
   - frac[18,34] - frac[18,35] - frac[18,36] - frac[18,37] - frac[18,38]
   - frac[18,39] - frac[18,40] - frac[18,41] - frac[18,42] - frac[18,43]
   - frac[18,44] - frac[18,45] - frac[18,46] - frac[18,47] - frac[18,48]
   - frac[18,49] - frac[18,50] - frac[18,51] - frac[18,52] - frac[18,53]
   - frac[18,54] - frac[18,55] - frac[18,56] - frac[18,57] - frac[18,58]
   - frac[18,59] - frac[18,60] - frac[18,61] - frac[18,62] - frac[18,63]
   - frac[18,64] - frac[18,65] - frac[18,66] - frac[18,67] - frac[18,68]
   - frac[18,69] - frac[18,70] - frac[18,71] - frac[18,72] - frac[18,73]
   - frac[18,74] - frac[18,75] - frac[18,76] - frac[18,77] - frac[18,78]
   - frac[18,79] - frac[18,80] - frac[18,81] - frac[18,82] - frac[18,83]
   - frac[18,84] - frac[18,85] - frac[18,86] - frac[18,87] - frac[18,88]
   - frac[18,89] - frac[18,90] - frac[18,91] - frac[18,92] - frac[18,93]
   - frac[18,94] - frac[18,95] - frac[18,96] - frac[18,97] - frac[18,98]
   - frac[18,99] - frac[18,100] - frac[18,101] - frac[18,102]
   - frac[18,103] - frac[18,104] - frac[18,105] - frac[18,106]
   - frac[18,107] - frac[18,108] - frac[18,109] - frac[18,110]
   - frac[18,111] - frac[18,112] - frac[18,113] - frac[18,114]
   - frac[18,115] - frac[18,116] - frac[18,117] - frac[18,118]
   - frac[18,119] - frac[18,120] - frac[18,121] - frac[18,122]
   - frac[18,123] - frac[18,124] - frac[18,125] - frac[18,126]
   - frac[18,127] - frac[18,128] - frac[18,129] - frac[18,130]
   - frac[18,131] - frac[18,132] - frac[18,133] - frac[18,134]
   - frac[18,135] - frac[18,136] - frac[18,137] - frac[18,138]
   - frac[18,139] - frac[18,140] - frac[18,141] - frac[18,142]
   - frac[18,143] - frac[18,144] - frac[18,145] - frac[18,146]
   - frac[18,147] - frac[18,148] - frac[18,149] - frac[18,150]
   - frac[18,151] - frac[18,152] - frac[18,153] - frac[18,154]
   - frac[18,155] - frac[18,156] - frac[18,157] - frac[18,158]
   - frac[18,159] - frac[18,160] - frac[18,161] - frac[18,162]
   - frac[18,163] - frac[18,164] - frac[18,165] - frac[18,166]
   - frac[18,167] - frac[18,168] - frac[18,169] - frac[18,170]
   - frac[18,171] - frac[18,172] - frac[18,173] - frac[18,174]
   - frac[18,175] - frac[18,176] - frac[18,177] - frac[18,178]
   - frac[18,179] - frac[18,180] - frac[18,181] - frac[18,182]
   - frac[18,183] - frac[18,184] - frac[18,185] - frac[18,186]
   - frac[18,187] - frac[18,188] - frac[18,189] - frac[18,190]
   - frac[18,191] + rows_tot_tor_p43 = 0
 mem_tot_tor_p43: - mem[18,0] - mem[18,1] - mem[18,2] - mem[18,3]
   - mem[18,4] - mem[18,5] - mem[18,6] - mem[18,7] - mem[18,8] - mem[18,9]
   - mem[18,10] - mem[18,11] - mem[18,12] - mem[18,13] - mem[18,14]
   - mem[18,15] - mem[18,16] - mem[18,17] - mem[18,18] - mem[18,19]
   - mem[18,20] - mem[18,21] - mem[18,22] - mem[18,23] - mem[18,24]
   - mem[18,25] - mem[18,26] - mem[18,27] - mem[18,28] - mem[18,29]
   - mem[18,30] - mem[18,31] - mem[18,32] - mem[18,33] - mem[18,34]
   - mem[18,35] - mem[18,36] - mem[18,37] - mem[18,38] - mem[18,39]
   - mem[18,40] - mem[18,41] - mem[18,42] - mem[18,43] - mem[18,44]
   - mem[18,45] - mem[18,46] - mem[18,47] - mem[18,48] - mem[18,49]
   - mem[18,50] - mem[18,51] - mem[18,52] - mem[18,53] - mem[18,54]
   - mem[18,55] - mem[18,56] - mem[18,57] - mem[18,58] - mem[18,59]
   - mem[18,60] - mem[18,61] - mem[18,62] - mem[18,63] - mem[18,64]
   - mem[18,65] - mem[18,66] - mem[18,67] - mem[18,68] - mem[18,69]
   - mem[18,70] - mem[18,71] - mem[18,72] - mem[18,73] - mem[18,74]
   - mem[18,75] - mem[18,76] - mem[18,77] - mem[18,78] - mem[18,79]
   - mem[18,80] - mem[18,81] - mem[18,82] - mem[18,83] - mem[18,84]
   - mem[18,85] - mem[18,86] - mem[18,87] - mem[18,88] - mem[18,89]
   - mem[18,90] - mem[18,91] - mem[18,92] - mem[18,93] - mem[18,94]
   - mem[18,95] - mem[18,96] - mem[18,97] - mem[18,98] - mem[18,99]
   - mem[18,100] - mem[18,101] - mem[18,102] - mem[18,103] - mem[18,104]
   - mem[18,105] - mem[18,106] - mem[18,107] - mem[18,108] - mem[18,109]
   - mem[18,110] - mem[18,111] - mem[18,112] - mem[18,113] - mem[18,114]
   - mem[18,115] - mem[18,116] - mem[18,117] - mem[18,118] - mem[18,119]
   - mem[18,120] - mem[18,121] - mem[18,122] - mem[18,123] - mem[18,124]
   - mem[18,125] - mem[18,126] - mem[18,127] - mem[18,128] - mem[18,129]
   - mem[18,130] - mem[18,131] - mem[18,132] - mem[18,133] - mem[18,134]
   - mem[18,135] - mem[18,136] - mem[18,137] - mem[18,138] - mem[18,139]
   - mem[18,140] - mem[18,141] - mem[18,142] - mem[18,143] - mem[18,144]
   - mem[18,145] - mem[18,146] - mem[18,147] - mem[18,148] - mem[18,149]
   - mem[18,150] - mem[18,151] - mem[18,152] - mem[18,153] - mem[18,154]
   - mem[18,155] - mem[18,156] - mem[18,157] - mem[18,158] - mem[18,159]
   - mem[18,160] - mem[18,161] - mem[18,162] - mem[18,163] - mem[18,164]
   - mem[18,165] - mem[18,166] - mem[18,167] - mem[18,168] - mem[18,169]
   - mem[18,170] - mem[18,171] - mem[18,172] - mem[18,173] - mem[18,174]
   - mem[18,175] - mem[18,176] - mem[18,177] - mem[18,178] - mem[18,179]
   - mem[18,180] - mem[18,181] - mem[18,182] - mem[18,183] - mem[18,184]
   - mem[18,185] - mem[18,186] - mem[18,187] - mem[18,188] - mem[18,189]
   - mem[18,190] - mem[18,191] + mem_tot_tor_p43 = 0
 capacity_mem_par_tor_p43: mem[18,0] <= 48
 capacity_mem_par_tor_p43: mem[18,1] <= 48
 capacity_mem_par_tor_p43: mem[18,2] <= 48
 capacity_mem_par_tor_p43: mem[18,3] <= 48
 capacity_mem_par_tor_p43: mem[18,4] <= 48
 capacity_mem_par_tor_p43: mem[18,5] <= 48
 capacity_mem_par_tor_p43: mem[18,6] <= 48
 capacity_mem_par_tor_p43: mem[18,7] <= 48
 capacity_mem_par_tor_p43: mem[18,8] <= 48
 capacity_mem_par_tor_p43: mem[18,9] <= 48
 capacity_mem_par_tor_p43: mem[18,10] <= 48
 capacity_mem_par_tor_p43: mem[18,11] <= 48
 capacity_mem_par_tor_p43: mem[18,12] <= 48
 capacity_mem_par_tor_p43: mem[18,13] <= 48
 capacity_mem_par_tor_p43: mem[18,14] <= 48
 capacity_mem_par_tor_p43: mem[18,15] <= 48
 capacity_mem_par_tor_p43: mem[18,16] <= 48
 capacity_mem_par_tor_p43: mem[18,17] <= 48
 capacity_mem_par_tor_p43: mem[18,18] <= 48
 capacity_mem_par_tor_p43: mem[18,19] <= 48
 capacity_mem_par_tor_p43: mem[18,20] <= 48
 capacity_mem_par_tor_p43: mem[18,21] <= 48
 capacity_mem_par_tor_p43: mem[18,22] <= 48
 capacity_mem_par_tor_p43: mem[18,23] <= 48
 capacity_mem_par_tor_p43: mem[18,24] <= 48
 capacity_mem_par_tor_p43: mem[18,25] <= 48
 capacity_mem_par_tor_p43: mem[18,26] <= 48
 capacity_mem_par_tor_p43: mem[18,27] <= 48
 capacity_mem_par_tor_p43: mem[18,28] <= 48
 capacity_mem_par_tor_p43: mem[18,29] <= 48
 capacity_mem_par_tor_p43: mem[18,30] <= 48
 capacity_mem_par_tor_p43: mem[18,31] <= 48
 capacity_mem_par_tor_p43: mem[18,32] <= 48
 capacity_mem_par_tor_p43: mem[18,33] <= 48
 capacity_mem_par_tor_p43: mem[18,34] <= 48
 capacity_mem_par_tor_p43: mem[18,35] <= 48
 capacity_mem_par_tor_p43: mem[18,36] <= 48
 capacity_mem_par_tor_p43: mem[18,37] <= 48
 capacity_mem_par_tor_p43: mem[18,38] <= 48
 capacity_mem_par_tor_p43: mem[18,39] <= 48
 capacity_mem_par_tor_p43: mem[18,40] <= 48
 capacity_mem_par_tor_p43: mem[18,41] <= 48
 capacity_mem_par_tor_p43: mem[18,42] <= 48
 capacity_mem_par_tor_p43: mem[18,43] <= 48
 capacity_mem_par_tor_p43: mem[18,44] <= 48
 capacity_mem_par_tor_p43: mem[18,45] <= 48
 capacity_mem_par_tor_p43: mem[18,46] <= 48
 capacity_mem_par_tor_p43: mem[18,47] <= 48
 capacity_mem_par_tor_p43: mem[18,48] <= 48
 capacity_mem_par_tor_p43: mem[18,49] <= 48
 capacity_mem_par_tor_p43: mem[18,50] <= 48
 capacity_mem_par_tor_p43: mem[18,51] <= 48
 capacity_mem_par_tor_p43: mem[18,52] <= 48
 capacity_mem_par_tor_p43: mem[18,53] <= 48
 capacity_mem_par_tor_p43: mem[18,54] <= 48
 capacity_mem_par_tor_p43: mem[18,55] <= 48
 capacity_mem_par_tor_p43: mem[18,56] <= 48
 capacity_mem_par_tor_p43: mem[18,57] <= 48
 capacity_mem_par_tor_p43: mem[18,58] <= 48
 capacity_mem_par_tor_p43: mem[18,59] <= 48
 capacity_mem_par_tor_p43: mem[18,60] <= 48
 capacity_mem_par_tor_p43: mem[18,61] <= 48
 capacity_mem_par_tor_p43: mem[18,62] <= 48
 capacity_mem_par_tor_p43: mem[18,63] <= 48
 capacity_mem_par_tor_p43: mem[18,64] <= 48
 capacity_mem_par_tor_p43: mem[18,65] <= 48
 capacity_mem_par_tor_p43: mem[18,66] <= 48
 capacity_mem_par_tor_p43: mem[18,67] <= 48
 capacity_mem_par_tor_p43: mem[18,68] <= 48
 capacity_mem_par_tor_p43: mem[18,69] <= 48
 capacity_mem_par_tor_p43: mem[18,70] <= 48
 capacity_mem_par_tor_p43: mem[18,71] <= 48
 capacity_mem_par_tor_p43: mem[18,72] <= 48
 capacity_mem_par_tor_p43: mem[18,73] <= 48
 capacity_mem_par_tor_p43: mem[18,74] <= 48
 capacity_mem_par_tor_p43: mem[18,75] <= 48
 capacity_mem_par_tor_p43: mem[18,76] <= 48
 capacity_mem_par_tor_p43: mem[18,77] <= 48
 capacity_mem_par_tor_p43: mem[18,78] <= 48
 capacity_mem_par_tor_p43: mem[18,79] <= 48
 capacity_mem_par_tor_p43: mem[18,80] <= 48
 capacity_mem_par_tor_p43: mem[18,81] <= 48
 capacity_mem_par_tor_p43: mem[18,82] <= 48
 capacity_mem_par_tor_p43: mem[18,83] <= 48
 capacity_mem_par_tor_p43: mem[18,84] <= 48
 capacity_mem_par_tor_p43: mem[18,85] <= 48
 capacity_mem_par_tor_p43: mem[18,86] <= 48
 capacity_mem_par_tor_p43: mem[18,87] <= 48
 capacity_mem_par_tor_p43: mem[18,88] <= 48
 capacity_mem_par_tor_p43: mem[18,89] <= 48
 capacity_mem_par_tor_p43: mem[18,90] <= 48
 capacity_mem_par_tor_p43: mem[18,91] <= 48
 capacity_mem_par_tor_p43: mem[18,92] <= 48
 capacity_mem_par_tor_p43: mem[18,93] <= 48
 capacity_mem_par_tor_p43: mem[18,94] <= 48
 capacity_mem_par_tor_p43: mem[18,95] <= 48
 capacity_mem_par_tor_p43: mem[18,96] <= 48
 capacity_mem_par_tor_p43: mem[18,97] <= 48
 capacity_mem_par_tor_p43: mem[18,98] <= 48
 capacity_mem_par_tor_p43: mem[18,99] <= 48
 capacity_mem_par_tor_p43: mem[18,100] <= 48
 capacity_mem_par_tor_p43: mem[18,101] <= 48
 capacity_mem_par_tor_p43: mem[18,102] <= 48
 capacity_mem_par_tor_p43: mem[18,103] <= 48
 capacity_mem_par_tor_p43: mem[18,104] <= 48
 capacity_mem_par_tor_p43: mem[18,105] <= 48
 capacity_mem_par_tor_p43: mem[18,106] <= 48
 capacity_mem_par_tor_p43: mem[18,107] <= 48
 capacity_mem_par_tor_p43: mem[18,108] <= 48
 capacity_mem_par_tor_p43: mem[18,109] <= 48
 capacity_mem_par_tor_p43: mem[18,110] <= 48
 capacity_mem_par_tor_p43: mem[18,111] <= 48
 capacity_mem_par_tor_p43: mem[18,112] <= 48
 capacity_mem_par_tor_p43: mem[18,113] <= 48
 capacity_mem_par_tor_p43: mem[18,114] <= 48
 capacity_mem_par_tor_p43: mem[18,115] <= 48
 capacity_mem_par_tor_p43: mem[18,116] <= 48
 capacity_mem_par_tor_p43: mem[18,117] <= 48
 capacity_mem_par_tor_p43: mem[18,118] <= 48
 capacity_mem_par_tor_p43: mem[18,119] <= 48
 capacity_mem_par_tor_p43: mem[18,120] <= 48
 capacity_mem_par_tor_p43: mem[18,121] <= 48
 capacity_mem_par_tor_p43: mem[18,122] <= 48
 capacity_mem_par_tor_p43: mem[18,123] <= 48
 capacity_mem_par_tor_p43: mem[18,124] <= 48
 capacity_mem_par_tor_p43: mem[18,125] <= 48
 capacity_mem_par_tor_p43: mem[18,126] <= 48
 capacity_mem_par_tor_p43: mem[18,127] <= 48
 capacity_mem_par_tor_p43: mem[18,128] <= 48
 capacity_mem_par_tor_p43: mem[18,129] <= 48
 capacity_mem_par_tor_p43: mem[18,130] <= 48
 capacity_mem_par_tor_p43: mem[18,131] <= 48
 capacity_mem_par_tor_p43: mem[18,132] <= 48
 capacity_mem_par_tor_p43: mem[18,133] <= 48
 capacity_mem_par_tor_p43: mem[18,134] <= 48
 capacity_mem_par_tor_p43: mem[18,135] <= 48
 capacity_mem_par_tor_p43: mem[18,136] <= 48
 capacity_mem_par_tor_p43: mem[18,137] <= 48
 capacity_mem_par_tor_p43: mem[18,138] <= 48
 capacity_mem_par_tor_p43: mem[18,139] <= 48
 capacity_mem_par_tor_p43: mem[18,140] <= 48
 capacity_mem_par_tor_p43: mem[18,141] <= 48
 capacity_mem_par_tor_p43: mem[18,142] <= 48
 capacity_mem_par_tor_p43: mem[18,143] <= 48
 capacity_mem_par_tor_p43: mem[18,144] <= 48
 capacity_mem_par_tor_p43: mem[18,145] <= 48
 capacity_mem_par_tor_p43: mem[18,146] <= 48
 capacity_mem_par_tor_p43: mem[18,147] <= 48
 capacity_mem_par_tor_p43: mem[18,148] <= 48
 capacity_mem_par_tor_p43: mem[18,149] <= 48
 capacity_mem_par_tor_p43: mem[18,150] <= 48
 capacity_mem_par_tor_p43: mem[18,151] <= 48
 capacity_mem_par_tor_p43: mem[18,152] <= 48
 capacity_mem_par_tor_p43: mem[18,153] <= 48
 capacity_mem_par_tor_p43: mem[18,154] <= 48
 capacity_mem_par_tor_p43: mem[18,155] <= 48
 capacity_mem_par_tor_p43: mem[18,156] <= 48
 capacity_mem_par_tor_p43: mem[18,157] <= 48
 capacity_mem_par_tor_p43: mem[18,158] <= 48
 capacity_mem_par_tor_p43: mem[18,159] <= 48
 capacity_mem_par_tor_p43: mem[18,160] <= 48
 capacity_mem_par_tor_p43: mem[18,161] <= 48
 capacity_mem_par_tor_p43: mem[18,162] <= 48
 capacity_mem_par_tor_p43: mem[18,163] <= 48
 capacity_mem_par_tor_p43: mem[18,164] <= 48
 capacity_mem_par_tor_p43: mem[18,165] <= 48
 capacity_mem_par_tor_p43: mem[18,166] <= 48
 capacity_mem_par_tor_p43: mem[18,167] <= 48
 capacity_mem_par_tor_p43: mem[18,168] <= 48
 capacity_mem_par_tor_p43: mem[18,169] <= 48
 capacity_mem_par_tor_p43: mem[18,170] <= 48
 capacity_mem_par_tor_p43: mem[18,171] <= 48
 capacity_mem_par_tor_p43: mem[18,172] <= 48
 capacity_mem_par_tor_p43: mem[18,173] <= 48
 capacity_mem_par_tor_p43: mem[18,174] <= 48
 capacity_mem_par_tor_p43: mem[18,175] <= 48
 capacity_mem_par_tor_p43: mem[18,176] <= 48
 capacity_mem_par_tor_p43: mem[18,177] <= 48
 capacity_mem_par_tor_p43: mem[18,178] <= 48
 capacity_mem_par_tor_p43: mem[18,179] <= 48
 capacity_mem_par_tor_p43: mem[18,180] <= 48
 capacity_mem_par_tor_p43: mem[18,181] <= 48
 capacity_mem_par_tor_p43: mem[18,182] <= 48
 capacity_mem_par_tor_p43: mem[18,183] <= 48
 capacity_mem_par_tor_p43: mem[18,184] <= 48
 capacity_mem_par_tor_p43: mem[18,185] <= 48
 capacity_mem_par_tor_p43: mem[18,186] <= 48
 capacity_mem_par_tor_p43: mem[18,187] <= 48
 capacity_mem_par_tor_p43: mem[18,188] <= 48
 capacity_mem_par_tor_p43: mem[18,189] <= 48
 capacity_mem_par_tor_p43: mem[18,190] <= 48
 capacity_mem_par_tor_p43: mem[18,191] <= 48
 ns_tor_p43: ns_tor_p43 = 6.756756756756757
 rows_tot_tor_p44: - frac[19,0] - frac[19,1] - frac[19,2] - frac[19,3]
   - frac[19,4] - frac[19,5] - frac[19,6] - frac[19,7] - frac[19,8]
   - frac[19,9] - frac[19,10] - frac[19,11] - frac[19,12] - frac[19,13]
   - frac[19,14] - frac[19,15] - frac[19,16] - frac[19,17] - frac[19,18]
   - frac[19,19] - frac[19,20] - frac[19,21] - frac[19,22] - frac[19,23]
   - frac[19,24] - frac[19,25] - frac[19,26] - frac[19,27] - frac[19,28]
   - frac[19,29] - frac[19,30] - frac[19,31] - frac[19,32] - frac[19,33]
   - frac[19,34] - frac[19,35] - frac[19,36] - frac[19,37] - frac[19,38]
   - frac[19,39] - frac[19,40] - frac[19,41] - frac[19,42] - frac[19,43]
   - frac[19,44] - frac[19,45] - frac[19,46] - frac[19,47] - frac[19,48]
   - frac[19,49] - frac[19,50] - frac[19,51] - frac[19,52] - frac[19,53]
   - frac[19,54] - frac[19,55] - frac[19,56] - frac[19,57] - frac[19,58]
   - frac[19,59] - frac[19,60] - frac[19,61] - frac[19,62] - frac[19,63]
   - frac[19,64] - frac[19,65] - frac[19,66] - frac[19,67] - frac[19,68]
   - frac[19,69] - frac[19,70] - frac[19,71] - frac[19,72] - frac[19,73]
   - frac[19,74] - frac[19,75] - frac[19,76] - frac[19,77] - frac[19,78]
   - frac[19,79] - frac[19,80] - frac[19,81] - frac[19,82] - frac[19,83]
   - frac[19,84] - frac[19,85] - frac[19,86] - frac[19,87] - frac[19,88]
   - frac[19,89] - frac[19,90] - frac[19,91] - frac[19,92] - frac[19,93]
   - frac[19,94] - frac[19,95] - frac[19,96] - frac[19,97] - frac[19,98]
   - frac[19,99] - frac[19,100] - frac[19,101] - frac[19,102]
   - frac[19,103] - frac[19,104] - frac[19,105] - frac[19,106]
   - frac[19,107] - frac[19,108] - frac[19,109] - frac[19,110]
   - frac[19,111] - frac[19,112] - frac[19,113] - frac[19,114]
   - frac[19,115] - frac[19,116] - frac[19,117] - frac[19,118]
   - frac[19,119] - frac[19,120] - frac[19,121] - frac[19,122]
   - frac[19,123] - frac[19,124] - frac[19,125] - frac[19,126]
   - frac[19,127] - frac[19,128] - frac[19,129] - frac[19,130]
   - frac[19,131] - frac[19,132] - frac[19,133] - frac[19,134]
   - frac[19,135] - frac[19,136] - frac[19,137] - frac[19,138]
   - frac[19,139] - frac[19,140] - frac[19,141] - frac[19,142]
   - frac[19,143] - frac[19,144] - frac[19,145] - frac[19,146]
   - frac[19,147] - frac[19,148] - frac[19,149] - frac[19,150]
   - frac[19,151] - frac[19,152] - frac[19,153] - frac[19,154]
   - frac[19,155] - frac[19,156] - frac[19,157] - frac[19,158]
   - frac[19,159] - frac[19,160] - frac[19,161] - frac[19,162]
   - frac[19,163] - frac[19,164] - frac[19,165] - frac[19,166]
   - frac[19,167] - frac[19,168] - frac[19,169] - frac[19,170]
   - frac[19,171] - frac[19,172] - frac[19,173] - frac[19,174]
   - frac[19,175] - frac[19,176] - frac[19,177] - frac[19,178]
   - frac[19,179] - frac[19,180] - frac[19,181] - frac[19,182]
   - frac[19,183] - frac[19,184] - frac[19,185] - frac[19,186]
   - frac[19,187] - frac[19,188] - frac[19,189] - frac[19,190]
   - frac[19,191] + rows_tot_tor_p44 = 0
 mem_tot_tor_p44: - mem[19,0] - mem[19,1] - mem[19,2] - mem[19,3]
   - mem[19,4] - mem[19,5] - mem[19,6] - mem[19,7] - mem[19,8] - mem[19,9]
   - mem[19,10] - mem[19,11] - mem[19,12] - mem[19,13] - mem[19,14]
   - mem[19,15] - mem[19,16] - mem[19,17] - mem[19,18] - mem[19,19]
   - mem[19,20] - mem[19,21] - mem[19,22] - mem[19,23] - mem[19,24]
   - mem[19,25] - mem[19,26] - mem[19,27] - mem[19,28] - mem[19,29]
   - mem[19,30] - mem[19,31] - mem[19,32] - mem[19,33] - mem[19,34]
   - mem[19,35] - mem[19,36] - mem[19,37] - mem[19,38] - mem[19,39]
   - mem[19,40] - mem[19,41] - mem[19,42] - mem[19,43] - mem[19,44]
   - mem[19,45] - mem[19,46] - mem[19,47] - mem[19,48] - mem[19,49]
   - mem[19,50] - mem[19,51] - mem[19,52] - mem[19,53] - mem[19,54]
   - mem[19,55] - mem[19,56] - mem[19,57] - mem[19,58] - mem[19,59]
   - mem[19,60] - mem[19,61] - mem[19,62] - mem[19,63] - mem[19,64]
   - mem[19,65] - mem[19,66] - mem[19,67] - mem[19,68] - mem[19,69]
   - mem[19,70] - mem[19,71] - mem[19,72] - mem[19,73] - mem[19,74]
   - mem[19,75] - mem[19,76] - mem[19,77] - mem[19,78] - mem[19,79]
   - mem[19,80] - mem[19,81] - mem[19,82] - mem[19,83] - mem[19,84]
   - mem[19,85] - mem[19,86] - mem[19,87] - mem[19,88] - mem[19,89]
   - mem[19,90] - mem[19,91] - mem[19,92] - mem[19,93] - mem[19,94]
   - mem[19,95] - mem[19,96] - mem[19,97] - mem[19,98] - mem[19,99]
   - mem[19,100] - mem[19,101] - mem[19,102] - mem[19,103] - mem[19,104]
   - mem[19,105] - mem[19,106] - mem[19,107] - mem[19,108] - mem[19,109]
   - mem[19,110] - mem[19,111] - mem[19,112] - mem[19,113] - mem[19,114]
   - mem[19,115] - mem[19,116] - mem[19,117] - mem[19,118] - mem[19,119]
   - mem[19,120] - mem[19,121] - mem[19,122] - mem[19,123] - mem[19,124]
   - mem[19,125] - mem[19,126] - mem[19,127] - mem[19,128] - mem[19,129]
   - mem[19,130] - mem[19,131] - mem[19,132] - mem[19,133] - mem[19,134]
   - mem[19,135] - mem[19,136] - mem[19,137] - mem[19,138] - mem[19,139]
   - mem[19,140] - mem[19,141] - mem[19,142] - mem[19,143] - mem[19,144]
   - mem[19,145] - mem[19,146] - mem[19,147] - mem[19,148] - mem[19,149]
   - mem[19,150] - mem[19,151] - mem[19,152] - mem[19,153] - mem[19,154]
   - mem[19,155] - mem[19,156] - mem[19,157] - mem[19,158] - mem[19,159]
   - mem[19,160] - mem[19,161] - mem[19,162] - mem[19,163] - mem[19,164]
   - mem[19,165] - mem[19,166] - mem[19,167] - mem[19,168] - mem[19,169]
   - mem[19,170] - mem[19,171] - mem[19,172] - mem[19,173] - mem[19,174]
   - mem[19,175] - mem[19,176] - mem[19,177] - mem[19,178] - mem[19,179]
   - mem[19,180] - mem[19,181] - mem[19,182] - mem[19,183] - mem[19,184]
   - mem[19,185] - mem[19,186] - mem[19,187] - mem[19,188] - mem[19,189]
   - mem[19,190] - mem[19,191] + mem_tot_tor_p44 = 0
 capacity_mem_par_tor_p44: mem[19,0] <= 48
 capacity_mem_par_tor_p44: mem[19,1] <= 48
 capacity_mem_par_tor_p44: mem[19,2] <= 48
 capacity_mem_par_tor_p44: mem[19,3] <= 48
 capacity_mem_par_tor_p44: mem[19,4] <= 48
 capacity_mem_par_tor_p44: mem[19,5] <= 48
 capacity_mem_par_tor_p44: mem[19,6] <= 48
 capacity_mem_par_tor_p44: mem[19,7] <= 48
 capacity_mem_par_tor_p44: mem[19,8] <= 48
 capacity_mem_par_tor_p44: mem[19,9] <= 48
 capacity_mem_par_tor_p44: mem[19,10] <= 48
 capacity_mem_par_tor_p44: mem[19,11] <= 48
 capacity_mem_par_tor_p44: mem[19,12] <= 48
 capacity_mem_par_tor_p44: mem[19,13] <= 48
 capacity_mem_par_tor_p44: mem[19,14] <= 48
 capacity_mem_par_tor_p44: mem[19,15] <= 48
 capacity_mem_par_tor_p44: mem[19,16] <= 48
 capacity_mem_par_tor_p44: mem[19,17] <= 48
 capacity_mem_par_tor_p44: mem[19,18] <= 48
 capacity_mem_par_tor_p44: mem[19,19] <= 48
 capacity_mem_par_tor_p44: mem[19,20] <= 48
 capacity_mem_par_tor_p44: mem[19,21] <= 48
 capacity_mem_par_tor_p44: mem[19,22] <= 48
 capacity_mem_par_tor_p44: mem[19,23] <= 48
 capacity_mem_par_tor_p44: mem[19,24] <= 48
 capacity_mem_par_tor_p44: mem[19,25] <= 48
 capacity_mem_par_tor_p44: mem[19,26] <= 48
 capacity_mem_par_tor_p44: mem[19,27] <= 48
 capacity_mem_par_tor_p44: mem[19,28] <= 48
 capacity_mem_par_tor_p44: mem[19,29] <= 48
 capacity_mem_par_tor_p44: mem[19,30] <= 48
 capacity_mem_par_tor_p44: mem[19,31] <= 48
 capacity_mem_par_tor_p44: mem[19,32] <= 48
 capacity_mem_par_tor_p44: mem[19,33] <= 48
 capacity_mem_par_tor_p44: mem[19,34] <= 48
 capacity_mem_par_tor_p44: mem[19,35] <= 48
 capacity_mem_par_tor_p44: mem[19,36] <= 48
 capacity_mem_par_tor_p44: mem[19,37] <= 48
 capacity_mem_par_tor_p44: mem[19,38] <= 48
 capacity_mem_par_tor_p44: mem[19,39] <= 48
 capacity_mem_par_tor_p44: mem[19,40] <= 48
 capacity_mem_par_tor_p44: mem[19,41] <= 48
 capacity_mem_par_tor_p44: mem[19,42] <= 48
 capacity_mem_par_tor_p44: mem[19,43] <= 48
 capacity_mem_par_tor_p44: mem[19,44] <= 48
 capacity_mem_par_tor_p44: mem[19,45] <= 48
 capacity_mem_par_tor_p44: mem[19,46] <= 48
 capacity_mem_par_tor_p44: mem[19,47] <= 48
 capacity_mem_par_tor_p44: mem[19,48] <= 48
 capacity_mem_par_tor_p44: mem[19,49] <= 48
 capacity_mem_par_tor_p44: mem[19,50] <= 48
 capacity_mem_par_tor_p44: mem[19,51] <= 48
 capacity_mem_par_tor_p44: mem[19,52] <= 48
 capacity_mem_par_tor_p44: mem[19,53] <= 48
 capacity_mem_par_tor_p44: mem[19,54] <= 48
 capacity_mem_par_tor_p44: mem[19,55] <= 48
 capacity_mem_par_tor_p44: mem[19,56] <= 48
 capacity_mem_par_tor_p44: mem[19,57] <= 48
 capacity_mem_par_tor_p44: mem[19,58] <= 48
 capacity_mem_par_tor_p44: mem[19,59] <= 48
 capacity_mem_par_tor_p44: mem[19,60] <= 48
 capacity_mem_par_tor_p44: mem[19,61] <= 48
 capacity_mem_par_tor_p44: mem[19,62] <= 48
 capacity_mem_par_tor_p44: mem[19,63] <= 48
 capacity_mem_par_tor_p44: mem[19,64] <= 48
 capacity_mem_par_tor_p44: mem[19,65] <= 48
 capacity_mem_par_tor_p44: mem[19,66] <= 48
 capacity_mem_par_tor_p44: mem[19,67] <= 48
 capacity_mem_par_tor_p44: mem[19,68] <= 48
 capacity_mem_par_tor_p44: mem[19,69] <= 48
 capacity_mem_par_tor_p44: mem[19,70] <= 48
 capacity_mem_par_tor_p44: mem[19,71] <= 48
 capacity_mem_par_tor_p44: mem[19,72] <= 48
 capacity_mem_par_tor_p44: mem[19,73] <= 48
 capacity_mem_par_tor_p44: mem[19,74] <= 48
 capacity_mem_par_tor_p44: mem[19,75] <= 48
 capacity_mem_par_tor_p44: mem[19,76] <= 48
 capacity_mem_par_tor_p44: mem[19,77] <= 48
 capacity_mem_par_tor_p44: mem[19,78] <= 48
 capacity_mem_par_tor_p44: mem[19,79] <= 48
 capacity_mem_par_tor_p44: mem[19,80] <= 48
 capacity_mem_par_tor_p44: mem[19,81] <= 48
 capacity_mem_par_tor_p44: mem[19,82] <= 48
 capacity_mem_par_tor_p44: mem[19,83] <= 48
 capacity_mem_par_tor_p44: mem[19,84] <= 48
 capacity_mem_par_tor_p44: mem[19,85] <= 48
 capacity_mem_par_tor_p44: mem[19,86] <= 48
 capacity_mem_par_tor_p44: mem[19,87] <= 48
 capacity_mem_par_tor_p44: mem[19,88] <= 48
 capacity_mem_par_tor_p44: mem[19,89] <= 48
 capacity_mem_par_tor_p44: mem[19,90] <= 48
 capacity_mem_par_tor_p44: mem[19,91] <= 48
 capacity_mem_par_tor_p44: mem[19,92] <= 48
 capacity_mem_par_tor_p44: mem[19,93] <= 48
 capacity_mem_par_tor_p44: mem[19,94] <= 48
 capacity_mem_par_tor_p44: mem[19,95] <= 48
 capacity_mem_par_tor_p44: mem[19,96] <= 48
 capacity_mem_par_tor_p44: mem[19,97] <= 48
 capacity_mem_par_tor_p44: mem[19,98] <= 48
 capacity_mem_par_tor_p44: mem[19,99] <= 48
 capacity_mem_par_tor_p44: mem[19,100] <= 48
 capacity_mem_par_tor_p44: mem[19,101] <= 48
 capacity_mem_par_tor_p44: mem[19,102] <= 48
 capacity_mem_par_tor_p44: mem[19,103] <= 48
 capacity_mem_par_tor_p44: mem[19,104] <= 48
 capacity_mem_par_tor_p44: mem[19,105] <= 48
 capacity_mem_par_tor_p44: mem[19,106] <= 48
 capacity_mem_par_tor_p44: mem[19,107] <= 48
 capacity_mem_par_tor_p44: mem[19,108] <= 48
 capacity_mem_par_tor_p44: mem[19,109] <= 48
 capacity_mem_par_tor_p44: mem[19,110] <= 48
 capacity_mem_par_tor_p44: mem[19,111] <= 48
 capacity_mem_par_tor_p44: mem[19,112] <= 48
 capacity_mem_par_tor_p44: mem[19,113] <= 48
 capacity_mem_par_tor_p44: mem[19,114] <= 48
 capacity_mem_par_tor_p44: mem[19,115] <= 48
 capacity_mem_par_tor_p44: mem[19,116] <= 48
 capacity_mem_par_tor_p44: mem[19,117] <= 48
 capacity_mem_par_tor_p44: mem[19,118] <= 48
 capacity_mem_par_tor_p44: mem[19,119] <= 48
 capacity_mem_par_tor_p44: mem[19,120] <= 48
 capacity_mem_par_tor_p44: mem[19,121] <= 48
 capacity_mem_par_tor_p44: mem[19,122] <= 48
 capacity_mem_par_tor_p44: mem[19,123] <= 48
 capacity_mem_par_tor_p44: mem[19,124] <= 48
 capacity_mem_par_tor_p44: mem[19,125] <= 48
 capacity_mem_par_tor_p44: mem[19,126] <= 48
 capacity_mem_par_tor_p44: mem[19,127] <= 48
 capacity_mem_par_tor_p44: mem[19,128] <= 48
 capacity_mem_par_tor_p44: mem[19,129] <= 48
 capacity_mem_par_tor_p44: mem[19,130] <= 48
 capacity_mem_par_tor_p44: mem[19,131] <= 48
 capacity_mem_par_tor_p44: mem[19,132] <= 48
 capacity_mem_par_tor_p44: mem[19,133] <= 48
 capacity_mem_par_tor_p44: mem[19,134] <= 48
 capacity_mem_par_tor_p44: mem[19,135] <= 48
 capacity_mem_par_tor_p44: mem[19,136] <= 48
 capacity_mem_par_tor_p44: mem[19,137] <= 48
 capacity_mem_par_tor_p44: mem[19,138] <= 48
 capacity_mem_par_tor_p44: mem[19,139] <= 48
 capacity_mem_par_tor_p44: mem[19,140] <= 48
 capacity_mem_par_tor_p44: mem[19,141] <= 48
 capacity_mem_par_tor_p44: mem[19,142] <= 48
 capacity_mem_par_tor_p44: mem[19,143] <= 48
 capacity_mem_par_tor_p44: mem[19,144] <= 48
 capacity_mem_par_tor_p44: mem[19,145] <= 48
 capacity_mem_par_tor_p44: mem[19,146] <= 48
 capacity_mem_par_tor_p44: mem[19,147] <= 48
 capacity_mem_par_tor_p44: mem[19,148] <= 48
 capacity_mem_par_tor_p44: mem[19,149] <= 48
 capacity_mem_par_tor_p44: mem[19,150] <= 48
 capacity_mem_par_tor_p44: mem[19,151] <= 48
 capacity_mem_par_tor_p44: mem[19,152] <= 48
 capacity_mem_par_tor_p44: mem[19,153] <= 48
 capacity_mem_par_tor_p44: mem[19,154] <= 48
 capacity_mem_par_tor_p44: mem[19,155] <= 48
 capacity_mem_par_tor_p44: mem[19,156] <= 48
 capacity_mem_par_tor_p44: mem[19,157] <= 48
 capacity_mem_par_tor_p44: mem[19,158] <= 48
 capacity_mem_par_tor_p44: mem[19,159] <= 48
 capacity_mem_par_tor_p44: mem[19,160] <= 48
 capacity_mem_par_tor_p44: mem[19,161] <= 48
 capacity_mem_par_tor_p44: mem[19,162] <= 48
 capacity_mem_par_tor_p44: mem[19,163] <= 48
 capacity_mem_par_tor_p44: mem[19,164] <= 48
 capacity_mem_par_tor_p44: mem[19,165] <= 48
 capacity_mem_par_tor_p44: mem[19,166] <= 48
 capacity_mem_par_tor_p44: mem[19,167] <= 48
 capacity_mem_par_tor_p44: mem[19,168] <= 48
 capacity_mem_par_tor_p44: mem[19,169] <= 48
 capacity_mem_par_tor_p44: mem[19,170] <= 48
 capacity_mem_par_tor_p44: mem[19,171] <= 48
 capacity_mem_par_tor_p44: mem[19,172] <= 48
 capacity_mem_par_tor_p44: mem[19,173] <= 48
 capacity_mem_par_tor_p44: mem[19,174] <= 48
 capacity_mem_par_tor_p44: mem[19,175] <= 48
 capacity_mem_par_tor_p44: mem[19,176] <= 48
 capacity_mem_par_tor_p44: mem[19,177] <= 48
 capacity_mem_par_tor_p44: mem[19,178] <= 48
 capacity_mem_par_tor_p44: mem[19,179] <= 48
 capacity_mem_par_tor_p44: mem[19,180] <= 48
 capacity_mem_par_tor_p44: mem[19,181] <= 48
 capacity_mem_par_tor_p44: mem[19,182] <= 48
 capacity_mem_par_tor_p44: mem[19,183] <= 48
 capacity_mem_par_tor_p44: mem[19,184] <= 48
 capacity_mem_par_tor_p44: mem[19,185] <= 48
 capacity_mem_par_tor_p44: mem[19,186] <= 48
 capacity_mem_par_tor_p44: mem[19,187] <= 48
 capacity_mem_par_tor_p44: mem[19,188] <= 48
 capacity_mem_par_tor_p44: mem[19,189] <= 48
 capacity_mem_par_tor_p44: mem[19,190] <= 48
 capacity_mem_par_tor_p44: mem[19,191] <= 48
 ns_tor_p44: ns_tor_p44 = 6.756756756756757
 rows_tot_l1_p41: - frac[20,0] - frac[20,1] - frac[20,2] - frac[20,3]
   - frac[20,4] - frac[20,5] - frac[20,6] - frac[20,7] - frac[20,8]
   - frac[20,9] - frac[20,10] - frac[20,11] - frac[20,12] - frac[20,13]
   - frac[20,14] - frac[20,15] - frac[20,16] - frac[20,17] - frac[20,18]
   - frac[20,19] - frac[20,20] - frac[20,21] - frac[20,22] - frac[20,23]
   - frac[20,24] - frac[20,25] - frac[20,26] - frac[20,27] - frac[20,28]
   - frac[20,29] - frac[20,30] - frac[20,31] - frac[20,32] - frac[20,33]
   - frac[20,34] - frac[20,35] - frac[20,36] - frac[20,37] - frac[20,38]
   - frac[20,39] - frac[20,40] - frac[20,41] - frac[20,42] - frac[20,43]
   - frac[20,44] - frac[20,45] - frac[20,46] - frac[20,47] - frac[20,48]
   - frac[20,49] - frac[20,50] - frac[20,51] - frac[20,52] - frac[20,53]
   - frac[20,54] - frac[20,55] - frac[20,56] - frac[20,57] - frac[20,58]
   - frac[20,59] - frac[20,60] - frac[20,61] - frac[20,62] - frac[20,63]
   - frac[20,64] - frac[20,65] - frac[20,66] - frac[20,67] - frac[20,68]
   - frac[20,69] - frac[20,70] - frac[20,71] - frac[20,72] - frac[20,73]
   - frac[20,74] - frac[20,75] - frac[20,76] - frac[20,77] - frac[20,78]
   - frac[20,79] - frac[20,80] - frac[20,81] - frac[20,82] - frac[20,83]
   - frac[20,84] - frac[20,85] - frac[20,86] - frac[20,87] - frac[20,88]
   - frac[20,89] - frac[20,90] - frac[20,91] - frac[20,92] - frac[20,93]
   - frac[20,94] - frac[20,95] - frac[20,96] - frac[20,97] - frac[20,98]
   - frac[20,99] - frac[20,100] - frac[20,101] - frac[20,102]
   - frac[20,103] - frac[20,104] - frac[20,105] - frac[20,106]
   - frac[20,107] - frac[20,108] - frac[20,109] - frac[20,110]
   - frac[20,111] - frac[20,112] - frac[20,113] - frac[20,114]
   - frac[20,115] - frac[20,116] - frac[20,117] - frac[20,118]
   - frac[20,119] - frac[20,120] - frac[20,121] - frac[20,122]
   - frac[20,123] - frac[20,124] - frac[20,125] - frac[20,126]
   - frac[20,127] - frac[20,128] - frac[20,129] - frac[20,130]
   - frac[20,131] - frac[20,132] - frac[20,133] - frac[20,134]
   - frac[20,135] - frac[20,136] - frac[20,137] - frac[20,138]
   - frac[20,139] - frac[20,140] - frac[20,141] - frac[20,142]
   - frac[20,143] - frac[20,144] - frac[20,145] - frac[20,146]
   - frac[20,147] - frac[20,148] - frac[20,149] - frac[20,150]
   - frac[20,151] - frac[20,152] - frac[20,153] - frac[20,154]
   - frac[20,155] - frac[20,156] - frac[20,157] - frac[20,158]
   - frac[20,159] - frac[20,160] - frac[20,161] - frac[20,162]
   - frac[20,163] - frac[20,164] - frac[20,165] - frac[20,166]
   - frac[20,167] - frac[20,168] - frac[20,169] - frac[20,170]
   - frac[20,171] - frac[20,172] - frac[20,173] - frac[20,174]
   - frac[20,175] - frac[20,176] - frac[20,177] - frac[20,178]
   - frac[20,179] - frac[20,180] - frac[20,181] - frac[20,182]
   - frac[20,183] - frac[20,184] - frac[20,185] - frac[20,186]
   - frac[20,187] - frac[20,188] - frac[20,189] - frac[20,190]
   - frac[20,191] + rows_tot_l1_p41 = 0
 mem_tot_l1_p41: - mem[20,0] - mem[20,1] - mem[20,2] - mem[20,3]
   - mem[20,4] - mem[20,5] - mem[20,6] - mem[20,7] - mem[20,8] - mem[20,9]
   - mem[20,10] - mem[20,11] - mem[20,12] - mem[20,13] - mem[20,14]
   - mem[20,15] - mem[20,16] - mem[20,17] - mem[20,18] - mem[20,19]
   - mem[20,20] - mem[20,21] - mem[20,22] - mem[20,23] - mem[20,24]
   - mem[20,25] - mem[20,26] - mem[20,27] - mem[20,28] - mem[20,29]
   - mem[20,30] - mem[20,31] - mem[20,32] - mem[20,33] - mem[20,34]
   - mem[20,35] - mem[20,36] - mem[20,37] - mem[20,38] - mem[20,39]
   - mem[20,40] - mem[20,41] - mem[20,42] - mem[20,43] - mem[20,44]
   - mem[20,45] - mem[20,46] - mem[20,47] - mem[20,48] - mem[20,49]
   - mem[20,50] - mem[20,51] - mem[20,52] - mem[20,53] - mem[20,54]
   - mem[20,55] - mem[20,56] - mem[20,57] - mem[20,58] - mem[20,59]
   - mem[20,60] - mem[20,61] - mem[20,62] - mem[20,63] - mem[20,64]
   - mem[20,65] - mem[20,66] - mem[20,67] - mem[20,68] - mem[20,69]
   - mem[20,70] - mem[20,71] - mem[20,72] - mem[20,73] - mem[20,74]
   - mem[20,75] - mem[20,76] - mem[20,77] - mem[20,78] - mem[20,79]
   - mem[20,80] - mem[20,81] - mem[20,82] - mem[20,83] - mem[20,84]
   - mem[20,85] - mem[20,86] - mem[20,87] - mem[20,88] - mem[20,89]
   - mem[20,90] - mem[20,91] - mem[20,92] - mem[20,93] - mem[20,94]
   - mem[20,95] - mem[20,96] - mem[20,97] - mem[20,98] - mem[20,99]
   - mem[20,100] - mem[20,101] - mem[20,102] - mem[20,103] - mem[20,104]
   - mem[20,105] - mem[20,106] - mem[20,107] - mem[20,108] - mem[20,109]
   - mem[20,110] - mem[20,111] - mem[20,112] - mem[20,113] - mem[20,114]
   - mem[20,115] - mem[20,116] - mem[20,117] - mem[20,118] - mem[20,119]
   - mem[20,120] - mem[20,121] - mem[20,122] - mem[20,123] - mem[20,124]
   - mem[20,125] - mem[20,126] - mem[20,127] - mem[20,128] - mem[20,129]
   - mem[20,130] - mem[20,131] - mem[20,132] - mem[20,133] - mem[20,134]
   - mem[20,135] - mem[20,136] - mem[20,137] - mem[20,138] - mem[20,139]
   - mem[20,140] - mem[20,141] - mem[20,142] - mem[20,143] - mem[20,144]
   - mem[20,145] - mem[20,146] - mem[20,147] - mem[20,148] - mem[20,149]
   - mem[20,150] - mem[20,151] - mem[20,152] - mem[20,153] - mem[20,154]
   - mem[20,155] - mem[20,156] - mem[20,157] - mem[20,158] - mem[20,159]
   - mem[20,160] - mem[20,161] - mem[20,162] - mem[20,163] - mem[20,164]
   - mem[20,165] - mem[20,166] - mem[20,167] - mem[20,168] - mem[20,169]
   - mem[20,170] - mem[20,171] - mem[20,172] - mem[20,173] - mem[20,174]
   - mem[20,175] - mem[20,176] - mem[20,177] - mem[20,178] - mem[20,179]
   - mem[20,180] - mem[20,181] - mem[20,182] - mem[20,183] - mem[20,184]
   - mem[20,185] - mem[20,186] - mem[20,187] - mem[20,188] - mem[20,189]
   - mem[20,190] - mem[20,191] + mem_tot_l1_p41 = 0
 capacity_mem_par_l1_p41: mem[20,0] <= 48
 capacity_mem_par_l1_p41: mem[20,1] <= 48
 capacity_mem_par_l1_p41: mem[20,2] <= 48
 capacity_mem_par_l1_p41: mem[20,3] <= 48
 capacity_mem_par_l1_p41: mem[20,4] <= 48
 capacity_mem_par_l1_p41: mem[20,5] <= 48
 capacity_mem_par_l1_p41: mem[20,6] <= 48
 capacity_mem_par_l1_p41: mem[20,7] <= 48
 capacity_mem_par_l1_p41: mem[20,8] <= 48
 capacity_mem_par_l1_p41: mem[20,9] <= 48
 capacity_mem_par_l1_p41: mem[20,10] <= 48
 capacity_mem_par_l1_p41: mem[20,11] <= 48
 capacity_mem_par_l1_p41: mem[20,12] <= 48
 capacity_mem_par_l1_p41: mem[20,13] <= 48
 capacity_mem_par_l1_p41: mem[20,14] <= 48
 capacity_mem_par_l1_p41: mem[20,15] <= 48
 capacity_mem_par_l1_p41: mem[20,16] <= 48
 capacity_mem_par_l1_p41: mem[20,17] <= 48
 capacity_mem_par_l1_p41: mem[20,18] <= 48
 capacity_mem_par_l1_p41: mem[20,19] <= 48
 capacity_mem_par_l1_p41: mem[20,20] <= 48
 capacity_mem_par_l1_p41: mem[20,21] <= 48
 capacity_mem_par_l1_p41: mem[20,22] <= 48
 capacity_mem_par_l1_p41: mem[20,23] <= 48
 capacity_mem_par_l1_p41: mem[20,24] <= 48
 capacity_mem_par_l1_p41: mem[20,25] <= 48
 capacity_mem_par_l1_p41: mem[20,26] <= 48
 capacity_mem_par_l1_p41: mem[20,27] <= 48
 capacity_mem_par_l1_p41: mem[20,28] <= 48
 capacity_mem_par_l1_p41: mem[20,29] <= 48
 capacity_mem_par_l1_p41: mem[20,30] <= 48
 capacity_mem_par_l1_p41: mem[20,31] <= 48
 capacity_mem_par_l1_p41: mem[20,32] <= 48
 capacity_mem_par_l1_p41: mem[20,33] <= 48
 capacity_mem_par_l1_p41: mem[20,34] <= 48
 capacity_mem_par_l1_p41: mem[20,35] <= 48
 capacity_mem_par_l1_p41: mem[20,36] <= 48
 capacity_mem_par_l1_p41: mem[20,37] <= 48
 capacity_mem_par_l1_p41: mem[20,38] <= 48
 capacity_mem_par_l1_p41: mem[20,39] <= 48
 capacity_mem_par_l1_p41: mem[20,40] <= 48
 capacity_mem_par_l1_p41: mem[20,41] <= 48
 capacity_mem_par_l1_p41: mem[20,42] <= 48
 capacity_mem_par_l1_p41: mem[20,43] <= 48
 capacity_mem_par_l1_p41: mem[20,44] <= 48
 capacity_mem_par_l1_p41: mem[20,45] <= 48
 capacity_mem_par_l1_p41: mem[20,46] <= 48
 capacity_mem_par_l1_p41: mem[20,47] <= 48
 capacity_mem_par_l1_p41: mem[20,48] <= 48
 capacity_mem_par_l1_p41: mem[20,49] <= 48
 capacity_mem_par_l1_p41: mem[20,50] <= 48
 capacity_mem_par_l1_p41: mem[20,51] <= 48
 capacity_mem_par_l1_p41: mem[20,52] <= 48
 capacity_mem_par_l1_p41: mem[20,53] <= 48
 capacity_mem_par_l1_p41: mem[20,54] <= 48
 capacity_mem_par_l1_p41: mem[20,55] <= 48
 capacity_mem_par_l1_p41: mem[20,56] <= 48
 capacity_mem_par_l1_p41: mem[20,57] <= 48
 capacity_mem_par_l1_p41: mem[20,58] <= 48
 capacity_mem_par_l1_p41: mem[20,59] <= 48
 capacity_mem_par_l1_p41: mem[20,60] <= 48
 capacity_mem_par_l1_p41: mem[20,61] <= 48
 capacity_mem_par_l1_p41: mem[20,62] <= 48
 capacity_mem_par_l1_p41: mem[20,63] <= 48
 capacity_mem_par_l1_p41: mem[20,64] <= 48
 capacity_mem_par_l1_p41: mem[20,65] <= 48
 capacity_mem_par_l1_p41: mem[20,66] <= 48
 capacity_mem_par_l1_p41: mem[20,67] <= 48
 capacity_mem_par_l1_p41: mem[20,68] <= 48
 capacity_mem_par_l1_p41: mem[20,69] <= 48
 capacity_mem_par_l1_p41: mem[20,70] <= 48
 capacity_mem_par_l1_p41: mem[20,71] <= 48
 capacity_mem_par_l1_p41: mem[20,72] <= 48
 capacity_mem_par_l1_p41: mem[20,73] <= 48
 capacity_mem_par_l1_p41: mem[20,74] <= 48
 capacity_mem_par_l1_p41: mem[20,75] <= 48
 capacity_mem_par_l1_p41: mem[20,76] <= 48
 capacity_mem_par_l1_p41: mem[20,77] <= 48
 capacity_mem_par_l1_p41: mem[20,78] <= 48
 capacity_mem_par_l1_p41: mem[20,79] <= 48
 capacity_mem_par_l1_p41: mem[20,80] <= 48
 capacity_mem_par_l1_p41: mem[20,81] <= 48
 capacity_mem_par_l1_p41: mem[20,82] <= 48
 capacity_mem_par_l1_p41: mem[20,83] <= 48
 capacity_mem_par_l1_p41: mem[20,84] <= 48
 capacity_mem_par_l1_p41: mem[20,85] <= 48
 capacity_mem_par_l1_p41: mem[20,86] <= 48
 capacity_mem_par_l1_p41: mem[20,87] <= 48
 capacity_mem_par_l1_p41: mem[20,88] <= 48
 capacity_mem_par_l1_p41: mem[20,89] <= 48
 capacity_mem_par_l1_p41: mem[20,90] <= 48
 capacity_mem_par_l1_p41: mem[20,91] <= 48
 capacity_mem_par_l1_p41: mem[20,92] <= 48
 capacity_mem_par_l1_p41: mem[20,93] <= 48
 capacity_mem_par_l1_p41: mem[20,94] <= 48
 capacity_mem_par_l1_p41: mem[20,95] <= 48
 capacity_mem_par_l1_p41: mem[20,96] <= 48
 capacity_mem_par_l1_p41: mem[20,97] <= 48
 capacity_mem_par_l1_p41: mem[20,98] <= 48
 capacity_mem_par_l1_p41: mem[20,99] <= 48
 capacity_mem_par_l1_p41: mem[20,100] <= 48
 capacity_mem_par_l1_p41: mem[20,101] <= 48
 capacity_mem_par_l1_p41: mem[20,102] <= 48
 capacity_mem_par_l1_p41: mem[20,103] <= 48
 capacity_mem_par_l1_p41: mem[20,104] <= 48
 capacity_mem_par_l1_p41: mem[20,105] <= 48
 capacity_mem_par_l1_p41: mem[20,106] <= 48
 capacity_mem_par_l1_p41: mem[20,107] <= 48
 capacity_mem_par_l1_p41: mem[20,108] <= 48
 capacity_mem_par_l1_p41: mem[20,109] <= 48
 capacity_mem_par_l1_p41: mem[20,110] <= 48
 capacity_mem_par_l1_p41: mem[20,111] <= 48
 capacity_mem_par_l1_p41: mem[20,112] <= 48
 capacity_mem_par_l1_p41: mem[20,113] <= 48
 capacity_mem_par_l1_p41: mem[20,114] <= 48
 capacity_mem_par_l1_p41: mem[20,115] <= 48
 capacity_mem_par_l1_p41: mem[20,116] <= 48
 capacity_mem_par_l1_p41: mem[20,117] <= 48
 capacity_mem_par_l1_p41: mem[20,118] <= 48
 capacity_mem_par_l1_p41: mem[20,119] <= 48
 capacity_mem_par_l1_p41: mem[20,120] <= 48
 capacity_mem_par_l1_p41: mem[20,121] <= 48
 capacity_mem_par_l1_p41: mem[20,122] <= 48
 capacity_mem_par_l1_p41: mem[20,123] <= 48
 capacity_mem_par_l1_p41: mem[20,124] <= 48
 capacity_mem_par_l1_p41: mem[20,125] <= 48
 capacity_mem_par_l1_p41: mem[20,126] <= 48
 capacity_mem_par_l1_p41: mem[20,127] <= 48
 capacity_mem_par_l1_p41: mem[20,128] <= 48
 capacity_mem_par_l1_p41: mem[20,129] <= 48
 capacity_mem_par_l1_p41: mem[20,130] <= 48
 capacity_mem_par_l1_p41: mem[20,131] <= 48
 capacity_mem_par_l1_p41: mem[20,132] <= 48
 capacity_mem_par_l1_p41: mem[20,133] <= 48
 capacity_mem_par_l1_p41: mem[20,134] <= 48
 capacity_mem_par_l1_p41: mem[20,135] <= 48
 capacity_mem_par_l1_p41: mem[20,136] <= 48
 capacity_mem_par_l1_p41: mem[20,137] <= 48
 capacity_mem_par_l1_p41: mem[20,138] <= 48
 capacity_mem_par_l1_p41: mem[20,139] <= 48
 capacity_mem_par_l1_p41: mem[20,140] <= 48
 capacity_mem_par_l1_p41: mem[20,141] <= 48
 capacity_mem_par_l1_p41: mem[20,142] <= 48
 capacity_mem_par_l1_p41: mem[20,143] <= 48
 capacity_mem_par_l1_p41: mem[20,144] <= 48
 capacity_mem_par_l1_p41: mem[20,145] <= 48
 capacity_mem_par_l1_p41: mem[20,146] <= 48
 capacity_mem_par_l1_p41: mem[20,147] <= 48
 capacity_mem_par_l1_p41: mem[20,148] <= 48
 capacity_mem_par_l1_p41: mem[20,149] <= 48
 capacity_mem_par_l1_p41: mem[20,150] <= 48
 capacity_mem_par_l1_p41: mem[20,151] <= 48
 capacity_mem_par_l1_p41: mem[20,152] <= 48
 capacity_mem_par_l1_p41: mem[20,153] <= 48
 capacity_mem_par_l1_p41: mem[20,154] <= 48
 capacity_mem_par_l1_p41: mem[20,155] <= 48
 capacity_mem_par_l1_p41: mem[20,156] <= 48
 capacity_mem_par_l1_p41: mem[20,157] <= 48
 capacity_mem_par_l1_p41: mem[20,158] <= 48
 capacity_mem_par_l1_p41: mem[20,159] <= 48
 capacity_mem_par_l1_p41: mem[20,160] <= 48
 capacity_mem_par_l1_p41: mem[20,161] <= 48
 capacity_mem_par_l1_p41: mem[20,162] <= 48
 capacity_mem_par_l1_p41: mem[20,163] <= 48
 capacity_mem_par_l1_p41: mem[20,164] <= 48
 capacity_mem_par_l1_p41: mem[20,165] <= 48
 capacity_mem_par_l1_p41: mem[20,166] <= 48
 capacity_mem_par_l1_p41: mem[20,167] <= 48
 capacity_mem_par_l1_p41: mem[20,168] <= 48
 capacity_mem_par_l1_p41: mem[20,169] <= 48
 capacity_mem_par_l1_p41: mem[20,170] <= 48
 capacity_mem_par_l1_p41: mem[20,171] <= 48
 capacity_mem_par_l1_p41: mem[20,172] <= 48
 capacity_mem_par_l1_p41: mem[20,173] <= 48
 capacity_mem_par_l1_p41: mem[20,174] <= 48
 capacity_mem_par_l1_p41: mem[20,175] <= 48
 capacity_mem_par_l1_p41: mem[20,176] <= 48
 capacity_mem_par_l1_p41: mem[20,177] <= 48
 capacity_mem_par_l1_p41: mem[20,178] <= 48
 capacity_mem_par_l1_p41: mem[20,179] <= 48
 capacity_mem_par_l1_p41: mem[20,180] <= 48
 capacity_mem_par_l1_p41: mem[20,181] <= 48
 capacity_mem_par_l1_p41: mem[20,182] <= 48
 capacity_mem_par_l1_p41: mem[20,183] <= 48
 capacity_mem_par_l1_p41: mem[20,184] <= 48
 capacity_mem_par_l1_p41: mem[20,185] <= 48
 capacity_mem_par_l1_p41: mem[20,186] <= 48
 capacity_mem_par_l1_p41: mem[20,187] <= 48
 capacity_mem_par_l1_p41: mem[20,188] <= 48
 capacity_mem_par_l1_p41: mem[20,189] <= 48
 capacity_mem_par_l1_p41: mem[20,190] <= 48
 capacity_mem_par_l1_p41: mem[20,191] <= 48
 ns_l1_p41: ns_l1_p41 = 6.756756756756757
 rows_tot_l1_p42: - frac[21,0] - frac[21,1] - frac[21,2] - frac[21,3]
   - frac[21,4] - frac[21,5] - frac[21,6] - frac[21,7] - frac[21,8]
   - frac[21,9] - frac[21,10] - frac[21,11] - frac[21,12] - frac[21,13]
   - frac[21,14] - frac[21,15] - frac[21,16] - frac[21,17] - frac[21,18]
   - frac[21,19] - frac[21,20] - frac[21,21] - frac[21,22] - frac[21,23]
   - frac[21,24] - frac[21,25] - frac[21,26] - frac[21,27] - frac[21,28]
   - frac[21,29] - frac[21,30] - frac[21,31] - frac[21,32] - frac[21,33]
   - frac[21,34] - frac[21,35] - frac[21,36] - frac[21,37] - frac[21,38]
   - frac[21,39] - frac[21,40] - frac[21,41] - frac[21,42] - frac[21,43]
   - frac[21,44] - frac[21,45] - frac[21,46] - frac[21,47] - frac[21,48]
   - frac[21,49] - frac[21,50] - frac[21,51] - frac[21,52] - frac[21,53]
   - frac[21,54] - frac[21,55] - frac[21,56] - frac[21,57] - frac[21,58]
   - frac[21,59] - frac[21,60] - frac[21,61] - frac[21,62] - frac[21,63]
   - frac[21,64] - frac[21,65] - frac[21,66] - frac[21,67] - frac[21,68]
   - frac[21,69] - frac[21,70] - frac[21,71] - frac[21,72] - frac[21,73]
   - frac[21,74] - frac[21,75] - frac[21,76] - frac[21,77] - frac[21,78]
   - frac[21,79] - frac[21,80] - frac[21,81] - frac[21,82] - frac[21,83]
   - frac[21,84] - frac[21,85] - frac[21,86] - frac[21,87] - frac[21,88]
   - frac[21,89] - frac[21,90] - frac[21,91] - frac[21,92] - frac[21,93]
   - frac[21,94] - frac[21,95] - frac[21,96] - frac[21,97] - frac[21,98]
   - frac[21,99] - frac[21,100] - frac[21,101] - frac[21,102]
   - frac[21,103] - frac[21,104] - frac[21,105] - frac[21,106]
   - frac[21,107] - frac[21,108] - frac[21,109] - frac[21,110]
   - frac[21,111] - frac[21,112] - frac[21,113] - frac[21,114]
   - frac[21,115] - frac[21,116] - frac[21,117] - frac[21,118]
   - frac[21,119] - frac[21,120] - frac[21,121] - frac[21,122]
   - frac[21,123] - frac[21,124] - frac[21,125] - frac[21,126]
   - frac[21,127] - frac[21,128] - frac[21,129] - frac[21,130]
   - frac[21,131] - frac[21,132] - frac[21,133] - frac[21,134]
   - frac[21,135] - frac[21,136] - frac[21,137] - frac[21,138]
   - frac[21,139] - frac[21,140] - frac[21,141] - frac[21,142]
   - frac[21,143] - frac[21,144] - frac[21,145] - frac[21,146]
   - frac[21,147] - frac[21,148] - frac[21,149] - frac[21,150]
   - frac[21,151] - frac[21,152] - frac[21,153] - frac[21,154]
   - frac[21,155] - frac[21,156] - frac[21,157] - frac[21,158]
   - frac[21,159] - frac[21,160] - frac[21,161] - frac[21,162]
   - frac[21,163] - frac[21,164] - frac[21,165] - frac[21,166]
   - frac[21,167] - frac[21,168] - frac[21,169] - frac[21,170]
   - frac[21,171] - frac[21,172] - frac[21,173] - frac[21,174]
   - frac[21,175] - frac[21,176] - frac[21,177] - frac[21,178]
   - frac[21,179] - frac[21,180] - frac[21,181] - frac[21,182]
   - frac[21,183] - frac[21,184] - frac[21,185] - frac[21,186]
   - frac[21,187] - frac[21,188] - frac[21,189] - frac[21,190]
   - frac[21,191] + rows_tot_l1_p42 = 0
 mem_tot_l1_p42: - mem[21,0] - mem[21,1] - mem[21,2] - mem[21,3]
   - mem[21,4] - mem[21,5] - mem[21,6] - mem[21,7] - mem[21,8] - mem[21,9]
   - mem[21,10] - mem[21,11] - mem[21,12] - mem[21,13] - mem[21,14]
   - mem[21,15] - mem[21,16] - mem[21,17] - mem[21,18] - mem[21,19]
   - mem[21,20] - mem[21,21] - mem[21,22] - mem[21,23] - mem[21,24]
   - mem[21,25] - mem[21,26] - mem[21,27] - mem[21,28] - mem[21,29]
   - mem[21,30] - mem[21,31] - mem[21,32] - mem[21,33] - mem[21,34]
   - mem[21,35] - mem[21,36] - mem[21,37] - mem[21,38] - mem[21,39]
   - mem[21,40] - mem[21,41] - mem[21,42] - mem[21,43] - mem[21,44]
   - mem[21,45] - mem[21,46] - mem[21,47] - mem[21,48] - mem[21,49]
   - mem[21,50] - mem[21,51] - mem[21,52] - mem[21,53] - mem[21,54]
   - mem[21,55] - mem[21,56] - mem[21,57] - mem[21,58] - mem[21,59]
   - mem[21,60] - mem[21,61] - mem[21,62] - mem[21,63] - mem[21,64]
   - mem[21,65] - mem[21,66] - mem[21,67] - mem[21,68] - mem[21,69]
   - mem[21,70] - mem[21,71] - mem[21,72] - mem[21,73] - mem[21,74]
   - mem[21,75] - mem[21,76] - mem[21,77] - mem[21,78] - mem[21,79]
   - mem[21,80] - mem[21,81] - mem[21,82] - mem[21,83] - mem[21,84]
   - mem[21,85] - mem[21,86] - mem[21,87] - mem[21,88] - mem[21,89]
   - mem[21,90] - mem[21,91] - mem[21,92] - mem[21,93] - mem[21,94]
   - mem[21,95] - mem[21,96] - mem[21,97] - mem[21,98] - mem[21,99]
   - mem[21,100] - mem[21,101] - mem[21,102] - mem[21,103] - mem[21,104]
   - mem[21,105] - mem[21,106] - mem[21,107] - mem[21,108] - mem[21,109]
   - mem[21,110] - mem[21,111] - mem[21,112] - mem[21,113] - mem[21,114]
   - mem[21,115] - mem[21,116] - mem[21,117] - mem[21,118] - mem[21,119]
   - mem[21,120] - mem[21,121] - mem[21,122] - mem[21,123] - mem[21,124]
   - mem[21,125] - mem[21,126] - mem[21,127] - mem[21,128] - mem[21,129]
   - mem[21,130] - mem[21,131] - mem[21,132] - mem[21,133] - mem[21,134]
   - mem[21,135] - mem[21,136] - mem[21,137] - mem[21,138] - mem[21,139]
   - mem[21,140] - mem[21,141] - mem[21,142] - mem[21,143] - mem[21,144]
   - mem[21,145] - mem[21,146] - mem[21,147] - mem[21,148] - mem[21,149]
   - mem[21,150] - mem[21,151] - mem[21,152] - mem[21,153] - mem[21,154]
   - mem[21,155] - mem[21,156] - mem[21,157] - mem[21,158] - mem[21,159]
   - mem[21,160] - mem[21,161] - mem[21,162] - mem[21,163] - mem[21,164]
   - mem[21,165] - mem[21,166] - mem[21,167] - mem[21,168] - mem[21,169]
   - mem[21,170] - mem[21,171] - mem[21,172] - mem[21,173] - mem[21,174]
   - mem[21,175] - mem[21,176] - mem[21,177] - mem[21,178] - mem[21,179]
   - mem[21,180] - mem[21,181] - mem[21,182] - mem[21,183] - mem[21,184]
   - mem[21,185] - mem[21,186] - mem[21,187] - mem[21,188] - mem[21,189]
   - mem[21,190] - mem[21,191] + mem_tot_l1_p42 = 0
 capacity_mem_par_l1_p42: mem[21,0] <= 48
 capacity_mem_par_l1_p42: mem[21,1] <= 48
 capacity_mem_par_l1_p42: mem[21,2] <= 48
 capacity_mem_par_l1_p42: mem[21,3] <= 48
 capacity_mem_par_l1_p42: mem[21,4] <= 48
 capacity_mem_par_l1_p42: mem[21,5] <= 48
 capacity_mem_par_l1_p42: mem[21,6] <= 48
 capacity_mem_par_l1_p42: mem[21,7] <= 48
 capacity_mem_par_l1_p42: mem[21,8] <= 48
 capacity_mem_par_l1_p42: mem[21,9] <= 48
 capacity_mem_par_l1_p42: mem[21,10] <= 48
 capacity_mem_par_l1_p42: mem[21,11] <= 48
 capacity_mem_par_l1_p42: mem[21,12] <= 48
 capacity_mem_par_l1_p42: mem[21,13] <= 48
 capacity_mem_par_l1_p42: mem[21,14] <= 48
 capacity_mem_par_l1_p42: mem[21,15] <= 48
 capacity_mem_par_l1_p42: mem[21,16] <= 48
 capacity_mem_par_l1_p42: mem[21,17] <= 48
 capacity_mem_par_l1_p42: mem[21,18] <= 48
 capacity_mem_par_l1_p42: mem[21,19] <= 48
 capacity_mem_par_l1_p42: mem[21,20] <= 48
 capacity_mem_par_l1_p42: mem[21,21] <= 48
 capacity_mem_par_l1_p42: mem[21,22] <= 48
 capacity_mem_par_l1_p42: mem[21,23] <= 48
 capacity_mem_par_l1_p42: mem[21,24] <= 48
 capacity_mem_par_l1_p42: mem[21,25] <= 48
 capacity_mem_par_l1_p42: mem[21,26] <= 48
 capacity_mem_par_l1_p42: mem[21,27] <= 48
 capacity_mem_par_l1_p42: mem[21,28] <= 48
 capacity_mem_par_l1_p42: mem[21,29] <= 48
 capacity_mem_par_l1_p42: mem[21,30] <= 48
 capacity_mem_par_l1_p42: mem[21,31] <= 48
 capacity_mem_par_l1_p42: mem[21,32] <= 48
 capacity_mem_par_l1_p42: mem[21,33] <= 48
 capacity_mem_par_l1_p42: mem[21,34] <= 48
 capacity_mem_par_l1_p42: mem[21,35] <= 48
 capacity_mem_par_l1_p42: mem[21,36] <= 48
 capacity_mem_par_l1_p42: mem[21,37] <= 48
 capacity_mem_par_l1_p42: mem[21,38] <= 48
 capacity_mem_par_l1_p42: mem[21,39] <= 48
 capacity_mem_par_l1_p42: mem[21,40] <= 48
 capacity_mem_par_l1_p42: mem[21,41] <= 48
 capacity_mem_par_l1_p42: mem[21,42] <= 48
 capacity_mem_par_l1_p42: mem[21,43] <= 48
 capacity_mem_par_l1_p42: mem[21,44] <= 48
 capacity_mem_par_l1_p42: mem[21,45] <= 48
 capacity_mem_par_l1_p42: mem[21,46] <= 48
 capacity_mem_par_l1_p42: mem[21,47] <= 48
 capacity_mem_par_l1_p42: mem[21,48] <= 48
 capacity_mem_par_l1_p42: mem[21,49] <= 48
 capacity_mem_par_l1_p42: mem[21,50] <= 48
 capacity_mem_par_l1_p42: mem[21,51] <= 48
 capacity_mem_par_l1_p42: mem[21,52] <= 48
 capacity_mem_par_l1_p42: mem[21,53] <= 48
 capacity_mem_par_l1_p42: mem[21,54] <= 48
 capacity_mem_par_l1_p42: mem[21,55] <= 48
 capacity_mem_par_l1_p42: mem[21,56] <= 48
 capacity_mem_par_l1_p42: mem[21,57] <= 48
 capacity_mem_par_l1_p42: mem[21,58] <= 48
 capacity_mem_par_l1_p42: mem[21,59] <= 48
 capacity_mem_par_l1_p42: mem[21,60] <= 48
 capacity_mem_par_l1_p42: mem[21,61] <= 48
 capacity_mem_par_l1_p42: mem[21,62] <= 48
 capacity_mem_par_l1_p42: mem[21,63] <= 48
 capacity_mem_par_l1_p42: mem[21,64] <= 48
 capacity_mem_par_l1_p42: mem[21,65] <= 48
 capacity_mem_par_l1_p42: mem[21,66] <= 48
 capacity_mem_par_l1_p42: mem[21,67] <= 48
 capacity_mem_par_l1_p42: mem[21,68] <= 48
 capacity_mem_par_l1_p42: mem[21,69] <= 48
 capacity_mem_par_l1_p42: mem[21,70] <= 48
 capacity_mem_par_l1_p42: mem[21,71] <= 48
 capacity_mem_par_l1_p42: mem[21,72] <= 48
 capacity_mem_par_l1_p42: mem[21,73] <= 48
 capacity_mem_par_l1_p42: mem[21,74] <= 48
 capacity_mem_par_l1_p42: mem[21,75] <= 48
 capacity_mem_par_l1_p42: mem[21,76] <= 48
 capacity_mem_par_l1_p42: mem[21,77] <= 48
 capacity_mem_par_l1_p42: mem[21,78] <= 48
 capacity_mem_par_l1_p42: mem[21,79] <= 48
 capacity_mem_par_l1_p42: mem[21,80] <= 48
 capacity_mem_par_l1_p42: mem[21,81] <= 48
 capacity_mem_par_l1_p42: mem[21,82] <= 48
 capacity_mem_par_l1_p42: mem[21,83] <= 48
 capacity_mem_par_l1_p42: mem[21,84] <= 48
 capacity_mem_par_l1_p42: mem[21,85] <= 48
 capacity_mem_par_l1_p42: mem[21,86] <= 48
 capacity_mem_par_l1_p42: mem[21,87] <= 48
 capacity_mem_par_l1_p42: mem[21,88] <= 48
 capacity_mem_par_l1_p42: mem[21,89] <= 48
 capacity_mem_par_l1_p42: mem[21,90] <= 48
 capacity_mem_par_l1_p42: mem[21,91] <= 48
 capacity_mem_par_l1_p42: mem[21,92] <= 48
 capacity_mem_par_l1_p42: mem[21,93] <= 48
 capacity_mem_par_l1_p42: mem[21,94] <= 48
 capacity_mem_par_l1_p42: mem[21,95] <= 48
 capacity_mem_par_l1_p42: mem[21,96] <= 48
 capacity_mem_par_l1_p42: mem[21,97] <= 48
 capacity_mem_par_l1_p42: mem[21,98] <= 48
 capacity_mem_par_l1_p42: mem[21,99] <= 48
 capacity_mem_par_l1_p42: mem[21,100] <= 48
 capacity_mem_par_l1_p42: mem[21,101] <= 48
 capacity_mem_par_l1_p42: mem[21,102] <= 48
 capacity_mem_par_l1_p42: mem[21,103] <= 48
 capacity_mem_par_l1_p42: mem[21,104] <= 48
 capacity_mem_par_l1_p42: mem[21,105] <= 48
 capacity_mem_par_l1_p42: mem[21,106] <= 48
 capacity_mem_par_l1_p42: mem[21,107] <= 48
 capacity_mem_par_l1_p42: mem[21,108] <= 48
 capacity_mem_par_l1_p42: mem[21,109] <= 48
 capacity_mem_par_l1_p42: mem[21,110] <= 48
 capacity_mem_par_l1_p42: mem[21,111] <= 48
 capacity_mem_par_l1_p42: mem[21,112] <= 48
 capacity_mem_par_l1_p42: mem[21,113] <= 48
 capacity_mem_par_l1_p42: mem[21,114] <= 48
 capacity_mem_par_l1_p42: mem[21,115] <= 48
 capacity_mem_par_l1_p42: mem[21,116] <= 48
 capacity_mem_par_l1_p42: mem[21,117] <= 48
 capacity_mem_par_l1_p42: mem[21,118] <= 48
 capacity_mem_par_l1_p42: mem[21,119] <= 48
 capacity_mem_par_l1_p42: mem[21,120] <= 48
 capacity_mem_par_l1_p42: mem[21,121] <= 48
 capacity_mem_par_l1_p42: mem[21,122] <= 48
 capacity_mem_par_l1_p42: mem[21,123] <= 48
 capacity_mem_par_l1_p42: mem[21,124] <= 48
 capacity_mem_par_l1_p42: mem[21,125] <= 48
 capacity_mem_par_l1_p42: mem[21,126] <= 48
 capacity_mem_par_l1_p42: mem[21,127] <= 48
 capacity_mem_par_l1_p42: mem[21,128] <= 48
 capacity_mem_par_l1_p42: mem[21,129] <= 48
 capacity_mem_par_l1_p42: mem[21,130] <= 48
 capacity_mem_par_l1_p42: mem[21,131] <= 48
 capacity_mem_par_l1_p42: mem[21,132] <= 48
 capacity_mem_par_l1_p42: mem[21,133] <= 48
 capacity_mem_par_l1_p42: mem[21,134] <= 48
 capacity_mem_par_l1_p42: mem[21,135] <= 48
 capacity_mem_par_l1_p42: mem[21,136] <= 48
 capacity_mem_par_l1_p42: mem[21,137] <= 48
 capacity_mem_par_l1_p42: mem[21,138] <= 48
 capacity_mem_par_l1_p42: mem[21,139] <= 48
 capacity_mem_par_l1_p42: mem[21,140] <= 48
 capacity_mem_par_l1_p42: mem[21,141] <= 48
 capacity_mem_par_l1_p42: mem[21,142] <= 48
 capacity_mem_par_l1_p42: mem[21,143] <= 48
 capacity_mem_par_l1_p42: mem[21,144] <= 48
 capacity_mem_par_l1_p42: mem[21,145] <= 48
 capacity_mem_par_l1_p42: mem[21,146] <= 48
 capacity_mem_par_l1_p42: mem[21,147] <= 48
 capacity_mem_par_l1_p42: mem[21,148] <= 48
 capacity_mem_par_l1_p42: mem[21,149] <= 48
 capacity_mem_par_l1_p42: mem[21,150] <= 48
 capacity_mem_par_l1_p42: mem[21,151] <= 48
 capacity_mem_par_l1_p42: mem[21,152] <= 48
 capacity_mem_par_l1_p42: mem[21,153] <= 48
 capacity_mem_par_l1_p42: mem[21,154] <= 48
 capacity_mem_par_l1_p42: mem[21,155] <= 48
 capacity_mem_par_l1_p42: mem[21,156] <= 48
 capacity_mem_par_l1_p42: mem[21,157] <= 48
 capacity_mem_par_l1_p42: mem[21,158] <= 48
 capacity_mem_par_l1_p42: mem[21,159] <= 48
 capacity_mem_par_l1_p42: mem[21,160] <= 48
 capacity_mem_par_l1_p42: mem[21,161] <= 48
 capacity_mem_par_l1_p42: mem[21,162] <= 48
 capacity_mem_par_l1_p42: mem[21,163] <= 48
 capacity_mem_par_l1_p42: mem[21,164] <= 48
 capacity_mem_par_l1_p42: mem[21,165] <= 48
 capacity_mem_par_l1_p42: mem[21,166] <= 48
 capacity_mem_par_l1_p42: mem[21,167] <= 48
 capacity_mem_par_l1_p42: mem[21,168] <= 48
 capacity_mem_par_l1_p42: mem[21,169] <= 48
 capacity_mem_par_l1_p42: mem[21,170] <= 48
 capacity_mem_par_l1_p42: mem[21,171] <= 48
 capacity_mem_par_l1_p42: mem[21,172] <= 48
 capacity_mem_par_l1_p42: mem[21,173] <= 48
 capacity_mem_par_l1_p42: mem[21,174] <= 48
 capacity_mem_par_l1_p42: mem[21,175] <= 48
 capacity_mem_par_l1_p42: mem[21,176] <= 48
 capacity_mem_par_l1_p42: mem[21,177] <= 48
 capacity_mem_par_l1_p42: mem[21,178] <= 48
 capacity_mem_par_l1_p42: mem[21,179] <= 48
 capacity_mem_par_l1_p42: mem[21,180] <= 48
 capacity_mem_par_l1_p42: mem[21,181] <= 48
 capacity_mem_par_l1_p42: mem[21,182] <= 48
 capacity_mem_par_l1_p42: mem[21,183] <= 48
 capacity_mem_par_l1_p42: mem[21,184] <= 48
 capacity_mem_par_l1_p42: mem[21,185] <= 48
 capacity_mem_par_l1_p42: mem[21,186] <= 48
 capacity_mem_par_l1_p42: mem[21,187] <= 48
 capacity_mem_par_l1_p42: mem[21,188] <= 48
 capacity_mem_par_l1_p42: mem[21,189] <= 48
 capacity_mem_par_l1_p42: mem[21,190] <= 48
 capacity_mem_par_l1_p42: mem[21,191] <= 48
 ns_l1_p42: ns_l1_p42 = 6.756756756756757
 rows_tot_l2_p4: - frac[22,0] - frac[22,1] - frac[22,2] - frac[22,3]
   - frac[22,4] - frac[22,5] - frac[22,6] - frac[22,7] - frac[22,8]
   - frac[22,9] - frac[22,10] - frac[22,11] - frac[22,12] - frac[22,13]
   - frac[22,14] - frac[22,15] - frac[22,16] - frac[22,17] - frac[22,18]
   - frac[22,19] - frac[22,20] - frac[22,21] - frac[22,22] - frac[22,23]
   - frac[22,24] - frac[22,25] - frac[22,26] - frac[22,27] - frac[22,28]
   - frac[22,29] - frac[22,30] - frac[22,31] - frac[22,32] - frac[22,33]
   - frac[22,34] - frac[22,35] - frac[22,36] - frac[22,37] - frac[22,38]
   - frac[22,39] - frac[22,40] - frac[22,41] - frac[22,42] - frac[22,43]
   - frac[22,44] - frac[22,45] - frac[22,46] - frac[22,47] - frac[22,48]
   - frac[22,49] - frac[22,50] - frac[22,51] - frac[22,52] - frac[22,53]
   - frac[22,54] - frac[22,55] - frac[22,56] - frac[22,57] - frac[22,58]
   - frac[22,59] - frac[22,60] - frac[22,61] - frac[22,62] - frac[22,63]
   - frac[22,64] - frac[22,65] - frac[22,66] - frac[22,67] - frac[22,68]
   - frac[22,69] - frac[22,70] - frac[22,71] - frac[22,72] - frac[22,73]
   - frac[22,74] - frac[22,75] - frac[22,76] - frac[22,77] - frac[22,78]
   - frac[22,79] - frac[22,80] - frac[22,81] - frac[22,82] - frac[22,83]
   - frac[22,84] - frac[22,85] - frac[22,86] - frac[22,87] - frac[22,88]
   - frac[22,89] - frac[22,90] - frac[22,91] - frac[22,92] - frac[22,93]
   - frac[22,94] - frac[22,95] - frac[22,96] - frac[22,97] - frac[22,98]
   - frac[22,99] - frac[22,100] - frac[22,101] - frac[22,102]
   - frac[22,103] - frac[22,104] - frac[22,105] - frac[22,106]
   - frac[22,107] - frac[22,108] - frac[22,109] - frac[22,110]
   - frac[22,111] - frac[22,112] - frac[22,113] - frac[22,114]
   - frac[22,115] - frac[22,116] - frac[22,117] - frac[22,118]
   - frac[22,119] - frac[22,120] - frac[22,121] - frac[22,122]
   - frac[22,123] - frac[22,124] - frac[22,125] - frac[22,126]
   - frac[22,127] - frac[22,128] - frac[22,129] - frac[22,130]
   - frac[22,131] - frac[22,132] - frac[22,133] - frac[22,134]
   - frac[22,135] - frac[22,136] - frac[22,137] - frac[22,138]
   - frac[22,139] - frac[22,140] - frac[22,141] - frac[22,142]
   - frac[22,143] - frac[22,144] - frac[22,145] - frac[22,146]
   - frac[22,147] - frac[22,148] - frac[22,149] - frac[22,150]
   - frac[22,151] - frac[22,152] - frac[22,153] - frac[22,154]
   - frac[22,155] - frac[22,156] - frac[22,157] - frac[22,158]
   - frac[22,159] - frac[22,160] - frac[22,161] - frac[22,162]
   - frac[22,163] - frac[22,164] - frac[22,165] - frac[22,166]
   - frac[22,167] - frac[22,168] - frac[22,169] - frac[22,170]
   - frac[22,171] - frac[22,172] - frac[22,173] - frac[22,174]
   - frac[22,175] - frac[22,176] - frac[22,177] - frac[22,178]
   - frac[22,179] - frac[22,180] - frac[22,181] - frac[22,182]
   - frac[22,183] - frac[22,184] - frac[22,185] - frac[22,186]
   - frac[22,187] - frac[22,188] - frac[22,189] - frac[22,190]
   - frac[22,191] + rows_tot_l2_p4 = 0
 mem_tot_l2_p4: - mem[22,0] - mem[22,1] - mem[22,2] - mem[22,3] - mem[22,4]
   - mem[22,5] - mem[22,6] - mem[22,7] - mem[22,8] - mem[22,9] - mem[22,10]
   - mem[22,11] - mem[22,12] - mem[22,13] - mem[22,14] - mem[22,15]
   - mem[22,16] - mem[22,17] - mem[22,18] - mem[22,19] - mem[22,20]
   - mem[22,21] - mem[22,22] - mem[22,23] - mem[22,24] - mem[22,25]
   - mem[22,26] - mem[22,27] - mem[22,28] - mem[22,29] - mem[22,30]
   - mem[22,31] - mem[22,32] - mem[22,33] - mem[22,34] - mem[22,35]
   - mem[22,36] - mem[22,37] - mem[22,38] - mem[22,39] - mem[22,40]
   - mem[22,41] - mem[22,42] - mem[22,43] - mem[22,44] - mem[22,45]
   - mem[22,46] - mem[22,47] - mem[22,48] - mem[22,49] - mem[22,50]
   - mem[22,51] - mem[22,52] - mem[22,53] - mem[22,54] - mem[22,55]
   - mem[22,56] - mem[22,57] - mem[22,58] - mem[22,59] - mem[22,60]
   - mem[22,61] - mem[22,62] - mem[22,63] - mem[22,64] - mem[22,65]
   - mem[22,66] - mem[22,67] - mem[22,68] - mem[22,69] - mem[22,70]
   - mem[22,71] - mem[22,72] - mem[22,73] - mem[22,74] - mem[22,75]
   - mem[22,76] - mem[22,77] - mem[22,78] - mem[22,79] - mem[22,80]
   - mem[22,81] - mem[22,82] - mem[22,83] - mem[22,84] - mem[22,85]
   - mem[22,86] - mem[22,87] - mem[22,88] - mem[22,89] - mem[22,90]
   - mem[22,91] - mem[22,92] - mem[22,93] - mem[22,94] - mem[22,95]
   - mem[22,96] - mem[22,97] - mem[22,98] - mem[22,99] - mem[22,100]
   - mem[22,101] - mem[22,102] - mem[22,103] - mem[22,104] - mem[22,105]
   - mem[22,106] - mem[22,107] - mem[22,108] - mem[22,109] - mem[22,110]
   - mem[22,111] - mem[22,112] - mem[22,113] - mem[22,114] - mem[22,115]
   - mem[22,116] - mem[22,117] - mem[22,118] - mem[22,119] - mem[22,120]
   - mem[22,121] - mem[22,122] - mem[22,123] - mem[22,124] - mem[22,125]
   - mem[22,126] - mem[22,127] - mem[22,128] - mem[22,129] - mem[22,130]
   - mem[22,131] - mem[22,132] - mem[22,133] - mem[22,134] - mem[22,135]
   - mem[22,136] - mem[22,137] - mem[22,138] - mem[22,139] - mem[22,140]
   - mem[22,141] - mem[22,142] - mem[22,143] - mem[22,144] - mem[22,145]
   - mem[22,146] - mem[22,147] - mem[22,148] - mem[22,149] - mem[22,150]
   - mem[22,151] - mem[22,152] - mem[22,153] - mem[22,154] - mem[22,155]
   - mem[22,156] - mem[22,157] - mem[22,158] - mem[22,159] - mem[22,160]
   - mem[22,161] - mem[22,162] - mem[22,163] - mem[22,164] - mem[22,165]
   - mem[22,166] - mem[22,167] - mem[22,168] - mem[22,169] - mem[22,170]
   - mem[22,171] - mem[22,172] - mem[22,173] - mem[22,174] - mem[22,175]
   - mem[22,176] - mem[22,177] - mem[22,178] - mem[22,179] - mem[22,180]
   - mem[22,181] - mem[22,182] - mem[22,183] - mem[22,184] - mem[22,185]
   - mem[22,186] - mem[22,187] - mem[22,188] - mem[22,189] - mem[22,190]
   - mem[22,191] + mem_tot_l2_p4 = 0
 capacity_mem_par_l2_p4: mem[22,0] <= 48
 capacity_mem_par_l2_p4: mem[22,1] <= 48
 capacity_mem_par_l2_p4: mem[22,2] <= 48
 capacity_mem_par_l2_p4: mem[22,3] <= 48
 capacity_mem_par_l2_p4: mem[22,4] <= 48
 capacity_mem_par_l2_p4: mem[22,5] <= 48
 capacity_mem_par_l2_p4: mem[22,6] <= 48
 capacity_mem_par_l2_p4: mem[22,7] <= 48
 capacity_mem_par_l2_p4: mem[22,8] <= 48
 capacity_mem_par_l2_p4: mem[22,9] <= 48
 capacity_mem_par_l2_p4: mem[22,10] <= 48
 capacity_mem_par_l2_p4: mem[22,11] <= 48
 capacity_mem_par_l2_p4: mem[22,12] <= 48
 capacity_mem_par_l2_p4: mem[22,13] <= 48
 capacity_mem_par_l2_p4: mem[22,14] <= 48
 capacity_mem_par_l2_p4: mem[22,15] <= 48
 capacity_mem_par_l2_p4: mem[22,16] <= 48
 capacity_mem_par_l2_p4: mem[22,17] <= 48
 capacity_mem_par_l2_p4: mem[22,18] <= 48
 capacity_mem_par_l2_p4: mem[22,19] <= 48
 capacity_mem_par_l2_p4: mem[22,20] <= 48
 capacity_mem_par_l2_p4: mem[22,21] <= 48
 capacity_mem_par_l2_p4: mem[22,22] <= 48
 capacity_mem_par_l2_p4: mem[22,23] <= 48
 capacity_mem_par_l2_p4: mem[22,24] <= 48
 capacity_mem_par_l2_p4: mem[22,25] <= 48
 capacity_mem_par_l2_p4: mem[22,26] <= 48
 capacity_mem_par_l2_p4: mem[22,27] <= 48
 capacity_mem_par_l2_p4: mem[22,28] <= 48
 capacity_mem_par_l2_p4: mem[22,29] <= 48
 capacity_mem_par_l2_p4: mem[22,30] <= 48
 capacity_mem_par_l2_p4: mem[22,31] <= 48
 capacity_mem_par_l2_p4: mem[22,32] <= 48
 capacity_mem_par_l2_p4: mem[22,33] <= 48
 capacity_mem_par_l2_p4: mem[22,34] <= 48
 capacity_mem_par_l2_p4: mem[22,35] <= 48
 capacity_mem_par_l2_p4: mem[22,36] <= 48
 capacity_mem_par_l2_p4: mem[22,37] <= 48
 capacity_mem_par_l2_p4: mem[22,38] <= 48
 capacity_mem_par_l2_p4: mem[22,39] <= 48
 capacity_mem_par_l2_p4: mem[22,40] <= 48
 capacity_mem_par_l2_p4: mem[22,41] <= 48
 capacity_mem_par_l2_p4: mem[22,42] <= 48
 capacity_mem_par_l2_p4: mem[22,43] <= 48
 capacity_mem_par_l2_p4: mem[22,44] <= 48
 capacity_mem_par_l2_p4: mem[22,45] <= 48
 capacity_mem_par_l2_p4: mem[22,46] <= 48
 capacity_mem_par_l2_p4: mem[22,47] <= 48
 capacity_mem_par_l2_p4: mem[22,48] <= 48
 capacity_mem_par_l2_p4: mem[22,49] <= 48
 capacity_mem_par_l2_p4: mem[22,50] <= 48
 capacity_mem_par_l2_p4: mem[22,51] <= 48
 capacity_mem_par_l2_p4: mem[22,52] <= 48
 capacity_mem_par_l2_p4: mem[22,53] <= 48
 capacity_mem_par_l2_p4: mem[22,54] <= 48
 capacity_mem_par_l2_p4: mem[22,55] <= 48
 capacity_mem_par_l2_p4: mem[22,56] <= 48
 capacity_mem_par_l2_p4: mem[22,57] <= 48
 capacity_mem_par_l2_p4: mem[22,58] <= 48
 capacity_mem_par_l2_p4: mem[22,59] <= 48
 capacity_mem_par_l2_p4: mem[22,60] <= 48
 capacity_mem_par_l2_p4: mem[22,61] <= 48
 capacity_mem_par_l2_p4: mem[22,62] <= 48
 capacity_mem_par_l2_p4: mem[22,63] <= 48
 capacity_mem_par_l2_p4: mem[22,64] <= 48
 capacity_mem_par_l2_p4: mem[22,65] <= 48
 capacity_mem_par_l2_p4: mem[22,66] <= 48
 capacity_mem_par_l2_p4: mem[22,67] <= 48
 capacity_mem_par_l2_p4: mem[22,68] <= 48
 capacity_mem_par_l2_p4: mem[22,69] <= 48
 capacity_mem_par_l2_p4: mem[22,70] <= 48
 capacity_mem_par_l2_p4: mem[22,71] <= 48
 capacity_mem_par_l2_p4: mem[22,72] <= 48
 capacity_mem_par_l2_p4: mem[22,73] <= 48
 capacity_mem_par_l2_p4: mem[22,74] <= 48
 capacity_mem_par_l2_p4: mem[22,75] <= 48
 capacity_mem_par_l2_p4: mem[22,76] <= 48
 capacity_mem_par_l2_p4: mem[22,77] <= 48
 capacity_mem_par_l2_p4: mem[22,78] <= 48
 capacity_mem_par_l2_p4: mem[22,79] <= 48
 capacity_mem_par_l2_p4: mem[22,80] <= 48
 capacity_mem_par_l2_p4: mem[22,81] <= 48
 capacity_mem_par_l2_p4: mem[22,82] <= 48
 capacity_mem_par_l2_p4: mem[22,83] <= 48
 capacity_mem_par_l2_p4: mem[22,84] <= 48
 capacity_mem_par_l2_p4: mem[22,85] <= 48
 capacity_mem_par_l2_p4: mem[22,86] <= 48
 capacity_mem_par_l2_p4: mem[22,87] <= 48
 capacity_mem_par_l2_p4: mem[22,88] <= 48
 capacity_mem_par_l2_p4: mem[22,89] <= 48
 capacity_mem_par_l2_p4: mem[22,90] <= 48
 capacity_mem_par_l2_p4: mem[22,91] <= 48
 capacity_mem_par_l2_p4: mem[22,92] <= 48
 capacity_mem_par_l2_p4: mem[22,93] <= 48
 capacity_mem_par_l2_p4: mem[22,94] <= 48
 capacity_mem_par_l2_p4: mem[22,95] <= 48
 capacity_mem_par_l2_p4: mem[22,96] <= 48
 capacity_mem_par_l2_p4: mem[22,97] <= 48
 capacity_mem_par_l2_p4: mem[22,98] <= 48
 capacity_mem_par_l2_p4: mem[22,99] <= 48
 capacity_mem_par_l2_p4: mem[22,100] <= 48
 capacity_mem_par_l2_p4: mem[22,101] <= 48
 capacity_mem_par_l2_p4: mem[22,102] <= 48
 capacity_mem_par_l2_p4: mem[22,103] <= 48
 capacity_mem_par_l2_p4: mem[22,104] <= 48
 capacity_mem_par_l2_p4: mem[22,105] <= 48
 capacity_mem_par_l2_p4: mem[22,106] <= 48
 capacity_mem_par_l2_p4: mem[22,107] <= 48
 capacity_mem_par_l2_p4: mem[22,108] <= 48
 capacity_mem_par_l2_p4: mem[22,109] <= 48
 capacity_mem_par_l2_p4: mem[22,110] <= 48
 capacity_mem_par_l2_p4: mem[22,111] <= 48
 capacity_mem_par_l2_p4: mem[22,112] <= 48
 capacity_mem_par_l2_p4: mem[22,113] <= 48
 capacity_mem_par_l2_p4: mem[22,114] <= 48
 capacity_mem_par_l2_p4: mem[22,115] <= 48
 capacity_mem_par_l2_p4: mem[22,116] <= 48
 capacity_mem_par_l2_p4: mem[22,117] <= 48
 capacity_mem_par_l2_p4: mem[22,118] <= 48
 capacity_mem_par_l2_p4: mem[22,119] <= 48
 capacity_mem_par_l2_p4: mem[22,120] <= 48
 capacity_mem_par_l2_p4: mem[22,121] <= 48
 capacity_mem_par_l2_p4: mem[22,122] <= 48
 capacity_mem_par_l2_p4: mem[22,123] <= 48
 capacity_mem_par_l2_p4: mem[22,124] <= 48
 capacity_mem_par_l2_p4: mem[22,125] <= 48
 capacity_mem_par_l2_p4: mem[22,126] <= 48
 capacity_mem_par_l2_p4: mem[22,127] <= 48
 capacity_mem_par_l2_p4: mem[22,128] <= 48
 capacity_mem_par_l2_p4: mem[22,129] <= 48
 capacity_mem_par_l2_p4: mem[22,130] <= 48
 capacity_mem_par_l2_p4: mem[22,131] <= 48
 capacity_mem_par_l2_p4: mem[22,132] <= 48
 capacity_mem_par_l2_p4: mem[22,133] <= 48
 capacity_mem_par_l2_p4: mem[22,134] <= 48
 capacity_mem_par_l2_p4: mem[22,135] <= 48
 capacity_mem_par_l2_p4: mem[22,136] <= 48
 capacity_mem_par_l2_p4: mem[22,137] <= 48
 capacity_mem_par_l2_p4: mem[22,138] <= 48
 capacity_mem_par_l2_p4: mem[22,139] <= 48
 capacity_mem_par_l2_p4: mem[22,140] <= 48
 capacity_mem_par_l2_p4: mem[22,141] <= 48
 capacity_mem_par_l2_p4: mem[22,142] <= 48
 capacity_mem_par_l2_p4: mem[22,143] <= 48
 capacity_mem_par_l2_p4: mem[22,144] <= 48
 capacity_mem_par_l2_p4: mem[22,145] <= 48
 capacity_mem_par_l2_p4: mem[22,146] <= 48
 capacity_mem_par_l2_p4: mem[22,147] <= 48
 capacity_mem_par_l2_p4: mem[22,148] <= 48
 capacity_mem_par_l2_p4: mem[22,149] <= 48
 capacity_mem_par_l2_p4: mem[22,150] <= 48
 capacity_mem_par_l2_p4: mem[22,151] <= 48
 capacity_mem_par_l2_p4: mem[22,152] <= 48
 capacity_mem_par_l2_p4: mem[22,153] <= 48
 capacity_mem_par_l2_p4: mem[22,154] <= 48
 capacity_mem_par_l2_p4: mem[22,155] <= 48
 capacity_mem_par_l2_p4: mem[22,156] <= 48
 capacity_mem_par_l2_p4: mem[22,157] <= 48
 capacity_mem_par_l2_p4: mem[22,158] <= 48
 capacity_mem_par_l2_p4: mem[22,159] <= 48
 capacity_mem_par_l2_p4: mem[22,160] <= 48
 capacity_mem_par_l2_p4: mem[22,161] <= 48
 capacity_mem_par_l2_p4: mem[22,162] <= 48
 capacity_mem_par_l2_p4: mem[22,163] <= 48
 capacity_mem_par_l2_p4: mem[22,164] <= 48
 capacity_mem_par_l2_p4: mem[22,165] <= 48
 capacity_mem_par_l2_p4: mem[22,166] <= 48
 capacity_mem_par_l2_p4: mem[22,167] <= 48
 capacity_mem_par_l2_p4: mem[22,168] <= 48
 capacity_mem_par_l2_p4: mem[22,169] <= 48
 capacity_mem_par_l2_p4: mem[22,170] <= 48
 capacity_mem_par_l2_p4: mem[22,171] <= 48
 capacity_mem_par_l2_p4: mem[22,172] <= 48
 capacity_mem_par_l2_p4: mem[22,173] <= 48
 capacity_mem_par_l2_p4: mem[22,174] <= 48
 capacity_mem_par_l2_p4: mem[22,175] <= 48
 capacity_mem_par_l2_p4: mem[22,176] <= 48
 capacity_mem_par_l2_p4: mem[22,177] <= 48
 capacity_mem_par_l2_p4: mem[22,178] <= 48
 capacity_mem_par_l2_p4: mem[22,179] <= 48
 capacity_mem_par_l2_p4: mem[22,180] <= 48
 capacity_mem_par_l2_p4: mem[22,181] <= 48
 capacity_mem_par_l2_p4: mem[22,182] <= 48
 capacity_mem_par_l2_p4: mem[22,183] <= 48
 capacity_mem_par_l2_p4: mem[22,184] <= 48
 capacity_mem_par_l2_p4: mem[22,185] <= 48
 capacity_mem_par_l2_p4: mem[22,186] <= 48
 capacity_mem_par_l2_p4: mem[22,187] <= 48
 capacity_mem_par_l2_p4: mem[22,188] <= 48
 capacity_mem_par_l2_p4: mem[22,189] <= 48
 capacity_mem_par_l2_p4: mem[22,190] <= 48
 capacity_mem_par_l2_p4: mem[22,191] <= 48
 ns_l2_p4: ns_l2_p4 = 6.756756756756757
 res: - 1.220703125e-04 mem_tot_cpu1 - 10 cores_sketch_cpu1
   - 10 cores_dpdk_cpu1 - 1.220703125e-04 mem_tot_cpu2
   - 10 cores_sketch_cpu2 - 10 cores_dpdk_cpu2
   - 1.220703125e-04 mem_tot_cpu3 - 10 cores_sketch_cpu3
   - 10 cores_dpdk_cpu3 - 1.220703125e-04 mem_tot_cpu4
   - 10 cores_sketch_cpu4 - 10 cores_dpdk_cpu4
   - 1.220703125e-04 mem_tot_cpu5 - 10 cores_sketch_cpu5
   - 10 cores_dpdk_cpu5 - 1.220703125e-04 mem_tot_cpu6
   - 10 cores_sketch_cpu6 - 10 cores_dpdk_cpu6
   - 1.220703125e-04 mem_tot_cpu7 - 10 cores_sketch_cpu7
   - 10 cores_dpdk_cpu7 - 1.220703125e-04 mem_tot_cpu8
   - 10 cores_sketch_cpu8 - 10 cores_dpdk_cpu8
   - 1.220703125e-04 mem_tot_cpu9 - 10 cores_sketch_cpu9
   - 10 cores_dpdk_cpu9 - 1.220703125e-04 mem_tot_cpu10
   - 10 cores_sketch_cpu10 - 10 cores_dpdk_cpu10
   - 1.220703125e-04 mem_tot_cpu11 - 10 cores_sketch_cpu11
   - 10 cores_dpdk_cpu11 - 1.220703125e-04 mem_tot_cpu12
   - 10 cores_sketch_cpu12 - 10 cores_dpdk_cpu12
   - 1.220703125e-04 mem_tot_cpu13 - 10 cores_sketch_cpu13
   - 10 cores_dpdk_cpu13 - 1.220703125e-04 mem_tot_cpu14
   - 10 cores_sketch_cpu14 - 10 cores_dpdk_cpu14
   - 1.220703125e-04 mem_tot_cpu15 - 10 cores_sketch_cpu15
   - 10 cores_dpdk_cpu15 - 1.220703125e-04 mem_tot_cpu16
   - 10 cores_sketch_cpu16 - 10 cores_dpdk_cpu16 - 0.25 rows_tot_tor_p41
   - 0.0208333333333333 mem_tot_tor_p41 - 0.25 rows_tot_tor_p42
   - 0.0208333333333333 mem_tot_tor_p42 - 0.25 rows_tot_tor_p43
   - 0.0208333333333333 mem_tot_tor_p43 - 0.25 rows_tot_tor_p44
   - 0.0208333333333333 mem_tot_tor_p44 - 0.25 rows_tot_l1_p41
   - 0.0208333333333333 mem_tot_l1_p41 - 0.25 rows_tot_l1_p42
   - 0.0208333333333333 mem_tot_l1_p42 - 0.25 rows_tot_l2_p4
   - 0.0208333333333333 mem_tot_l2_p4 + res_overall = 0
 pdt_m_rows_cpu1: pdt_m_rows_cpu1 + [ - rows_tot_cpu1 * m_access_time_cpu1
   ] = 0
 pdt_nsk_c_cpu1: pdt_nsk_c_cpu1 + [ - cores_sketch_cpu1 * ns_sketch_cpu1 ]
   = 0
 pdt_nsc_dpdk_cpu1: pdt_nsc_dpdk_cpu1 + [ - cores_dpdk_cpu1 * ns_dpdk_cpu1
   ] = 0
 pdt_m_rows_cpu2: pdt_m_rows_cpu2 + [ - rows_tot_cpu2 * m_access_time_cpu2
   ] = 0
 pdt_nsk_c_cpu2: pdt_nsk_c_cpu2 + [ - cores_sketch_cpu2 * ns_sketch_cpu2 ]
   = 0
 pdt_nsc_dpdk_cpu2: pdt_nsc_dpdk_cpu2 + [ - cores_dpdk_cpu2 * ns_dpdk_cpu2
   ] = 0
 pdt_m_rows_cpu3: pdt_m_rows_cpu3 + [ - rows_tot_cpu3 * m_access_time_cpu3
   ] = 0
 pdt_nsk_c_cpu3: pdt_nsk_c_cpu3 + [ - cores_sketch_cpu3 * ns_sketch_cpu3 ]
   = 0
 pdt_nsc_dpdk_cpu3: pdt_nsc_dpdk_cpu3 + [ - cores_dpdk_cpu3 * ns_dpdk_cpu3
   ] = 0
 pdt_m_rows_cpu4: pdt_m_rows_cpu4 + [ - rows_tot_cpu4 * m_access_time_cpu4
   ] = 0
 pdt_nsk_c_cpu4: pdt_nsk_c_cpu4 + [ - cores_sketch_cpu4 * ns_sketch_cpu4 ]
   = 0
 pdt_nsc_dpdk_cpu4: pdt_nsc_dpdk_cpu4 + [ - cores_dpdk_cpu4 * ns_dpdk_cpu4
   ] = 0
 pdt_m_rows_cpu5: pdt_m_rows_cpu5 + [ - rows_tot_cpu5 * m_access_time_cpu5
   ] = 0
 pdt_nsk_c_cpu5: pdt_nsk_c_cpu5 + [ - cores_sketch_cpu5 * ns_sketch_cpu5 ]
   = 0
 pdt_nsc_dpdk_cpu5: pdt_nsc_dpdk_cpu5 + [ - cores_dpdk_cpu5 * ns_dpdk_cpu5
   ] = 0
 pdt_m_rows_cpu6: pdt_m_rows_cpu6 + [ - rows_tot_cpu6 * m_access_time_cpu6
   ] = 0
 pdt_nsk_c_cpu6: pdt_nsk_c_cpu6 + [ - cores_sketch_cpu6 * ns_sketch_cpu6 ]
   = 0
 pdt_nsc_dpdk_cpu6: pdt_nsc_dpdk_cpu6 + [ - cores_dpdk_cpu6 * ns_dpdk_cpu6
   ] = 0
 pdt_m_rows_cpu7: pdt_m_rows_cpu7 + [ - rows_tot_cpu7 * m_access_time_cpu7
   ] = 0
 pdt_nsk_c_cpu7: pdt_nsk_c_cpu7 + [ - cores_sketch_cpu7 * ns_sketch_cpu7 ]
   = 0
 pdt_nsc_dpdk_cpu7: pdt_nsc_dpdk_cpu7 + [ - cores_dpdk_cpu7 * ns_dpdk_cpu7
   ] = 0
 pdt_m_rows_cpu8: pdt_m_rows_cpu8 + [ - rows_tot_cpu8 * m_access_time_cpu8
   ] = 0
 pdt_nsk_c_cpu8: pdt_nsk_c_cpu8 + [ - cores_sketch_cpu8 * ns_sketch_cpu8 ]
   = 0
 pdt_nsc_dpdk_cpu8: pdt_nsc_dpdk_cpu8 + [ - cores_dpdk_cpu8 * ns_dpdk_cpu8
   ] = 0
 pdt_m_rows_cpu9: pdt_m_rows_cpu9 + [ - rows_tot_cpu9 * m_access_time_cpu9
   ] = 0
 pdt_nsk_c_cpu9: pdt_nsk_c_cpu9 + [ - cores_sketch_cpu9 * ns_sketch_cpu9 ]
   = 0
 pdt_nsc_dpdk_cpu9: pdt_nsc_dpdk_cpu9 + [ - cores_dpdk_cpu9 * ns_dpdk_cpu9
   ] = 0
 pdt_m_rows_cpu10: pdt_m_rows_cpu10 + [
   - rows_tot_cpu10 * m_access_time_cpu10 ] = 0
 pdt_nsk_c_cpu10: pdt_nsk_c_cpu10 + [
   - cores_sketch_cpu10 * ns_sketch_cpu10 ] = 0
 pdt_nsc_dpdk_cpu10: pdt_nsc_dpdk_cpu10 + [
   - cores_dpdk_cpu10 * ns_dpdk_cpu10 ] = 0
 pdt_m_rows_cpu11: pdt_m_rows_cpu11 + [
   - rows_tot_cpu11 * m_access_time_cpu11 ] = 0
 pdt_nsk_c_cpu11: pdt_nsk_c_cpu11 + [
   - cores_sketch_cpu11 * ns_sketch_cpu11 ] = 0
 pdt_nsc_dpdk_cpu11: pdt_nsc_dpdk_cpu11 + [
   - cores_dpdk_cpu11 * ns_dpdk_cpu11 ] = 0
 pdt_m_rows_cpu12: pdt_m_rows_cpu12 + [
   - rows_tot_cpu12 * m_access_time_cpu12 ] = 0
 pdt_nsk_c_cpu12: pdt_nsk_c_cpu12 + [
   - cores_sketch_cpu12 * ns_sketch_cpu12 ] = 0
 pdt_nsc_dpdk_cpu12: pdt_nsc_dpdk_cpu12 + [
   - cores_dpdk_cpu12 * ns_dpdk_cpu12 ] = 0
 pdt_m_rows_cpu13: pdt_m_rows_cpu13 + [
   - rows_tot_cpu13 * m_access_time_cpu13 ] = 0
 pdt_nsk_c_cpu13: pdt_nsk_c_cpu13 + [
   - cores_sketch_cpu13 * ns_sketch_cpu13 ] = 0
 pdt_nsc_dpdk_cpu13: pdt_nsc_dpdk_cpu13 + [
   - cores_dpdk_cpu13 * ns_dpdk_cpu13 ] = 0
 pdt_m_rows_cpu14: pdt_m_rows_cpu14 + [
   - rows_tot_cpu14 * m_access_time_cpu14 ] = 0
 pdt_nsk_c_cpu14: pdt_nsk_c_cpu14 + [
   - cores_sketch_cpu14 * ns_sketch_cpu14 ] = 0
 pdt_nsc_dpdk_cpu14: pdt_nsc_dpdk_cpu14 + [
   - cores_dpdk_cpu14 * ns_dpdk_cpu14 ] = 0
 pdt_m_rows_cpu15: pdt_m_rows_cpu15 + [
   - rows_tot_cpu15 * m_access_time_cpu15 ] = 0
 pdt_nsk_c_cpu15: pdt_nsk_c_cpu15 + [
   - cores_sketch_cpu15 * ns_sketch_cpu15 ] = 0
 pdt_nsc_dpdk_cpu15: pdt_nsc_dpdk_cpu15 + [
   - cores_dpdk_cpu15 * ns_dpdk_cpu15 ] = 0
 pdt_m_rows_cpu16: pdt_m_rows_cpu16 + [
   - rows_tot_cpu16 * m_access_time_cpu16 ] = 0
 pdt_nsk_c_cpu16: pdt_nsk_c_cpu16 + [
   - cores_sketch_cpu16 * ns_sketch_cpu16 ] = 0
 pdt_nsc_dpdk_cpu16: pdt_nsc_dpdk_cpu16 + [
   - cores_dpdk_cpu16 * ns_dpdk_cpu16 ] = 0
Bounds
 frac[0,0] <= 1
 frac[0,1] <= 1
 frac[0,2] <= 1
 frac[0,3] <= 1
 frac[0,4] <= 1
 frac[0,5] <= 1
 frac[0,6] <= 1
 frac[0,7] <= 1
 frac[0,8] <= 1
 frac[0,9] <= 1
 frac[0,10] <= 1
 frac[0,11] <= 1
 frac[0,12] <= 1
 frac[0,13] <= 1
 frac[0,14] <= 1
 frac[0,15] <= 1
 frac[0,16] <= 1
 frac[0,17] <= 1
 frac[0,18] <= 1
 frac[0,19] <= 1
 frac[0,20] <= 1
 frac[0,21] <= 1
 frac[0,22] <= 1
 frac[0,23] <= 1
 frac[0,24] <= 1
 frac[0,25] <= 1
 frac[0,26] <= 1
 frac[0,27] <= 1
 frac[0,28] <= 1
 frac[0,29] <= 1
 frac[0,30] <= 1
 frac[0,31] <= 1
 frac[0,32] <= 1
 frac[0,33] <= 1
 frac[0,34] <= 1
 frac[0,35] <= 1
 frac[0,36] <= 1
 frac[0,37] <= 1
 frac[0,38] <= 1
 frac[0,39] <= 1
 frac[0,40] <= 1
 frac[0,41] <= 1
 frac[0,42] <= 1
 frac[0,43] <= 1
 frac[0,44] <= 1
 frac[0,45] <= 1
 frac[0,46] <= 1
 frac[0,47] <= 1
 frac[0,48] <= 1
 frac[0,49] <= 1
 frac[0,50] <= 1
 frac[0,51] <= 1
 frac[0,52] <= 1
 frac[0,53] <= 1
 frac[0,54] <= 1
 frac[0,55] <= 1
 frac[0,56] <= 1
 frac[0,57] <= 1
 frac[0,58] <= 1
 frac[0,59] <= 1
 frac[0,60] <= 1
 frac[0,61] <= 1
 frac[0,62] <= 1
 frac[0,63] <= 1
 frac[0,64] <= 1
 frac[0,65] <= 1
 frac[0,66] <= 1
 frac[0,67] <= 1
 frac[0,68] <= 1
 frac[0,69] <= 1
 frac[0,70] <= 1
 frac[0,71] <= 1
 frac[0,72] <= 1
 frac[0,73] <= 1
 frac[0,74] <= 1
 frac[0,75] <= 1
 frac[0,76] <= 1
 frac[0,77] <= 1
 frac[0,78] <= 1
 frac[0,79] <= 1
 frac[0,80] <= 1
 frac[0,81] <= 1
 frac[0,82] <= 1
 frac[0,83] <= 1
 frac[0,84] <= 1
 frac[0,85] <= 1
 frac[0,86] <= 1
 frac[0,87] <= 1
 frac[0,88] <= 1
 frac[0,89] <= 1
 frac[0,90] <= 1
 frac[0,91] <= 1
 frac[0,92] <= 1
 frac[0,93] <= 1
 frac[0,94] <= 1
 frac[0,95] <= 1
 frac[0,96] <= 1
 frac[0,97] <= 1
 frac[0,98] <= 1
 frac[0,99] <= 1
 frac[0,100] <= 1
 frac[0,101] <= 1
 frac[0,102] <= 1
 frac[0,103] <= 1
 frac[0,104] <= 1
 frac[0,105] <= 1
 frac[0,106] <= 1
 frac[0,107] <= 1
 frac[0,108] <= 1
 frac[0,109] <= 1
 frac[0,110] <= 1
 frac[0,111] <= 1
 frac[0,112] <= 1
 frac[0,113] <= 1
 frac[0,114] <= 1
 frac[0,115] <= 1
 frac[0,116] <= 1
 frac[0,117] <= 1
 frac[0,118] <= 1
 frac[0,119] <= 1
 frac[0,120] <= 1
 frac[0,121] <= 1
 frac[0,122] <= 1
 frac[0,123] <= 1
 frac[0,124] <= 1
 frac[0,125] <= 1
 frac[0,126] <= 1
 frac[0,127] <= 1
 frac[0,128] <= 1
 frac[0,129] <= 1
 frac[0,130] <= 1
 frac[0,131] <= 1
 frac[0,132] <= 1
 frac[0,133] <= 1
 frac[0,134] <= 1
 frac[0,135] <= 1
 frac[0,136] <= 1
 frac[0,137] <= 1
 frac[0,138] <= 1
 frac[0,139] <= 1
 frac[0,140] <= 1
 frac[0,141] <= 1
 frac[0,142] <= 1
 frac[0,143] <= 1
 frac[0,144] <= 1
 frac[0,145] <= 1
 frac[0,146] <= 1
 frac[0,147] <= 1
 frac[0,148] <= 1
 frac[0,149] <= 1
 frac[0,150] <= 1
 frac[0,151] <= 1
 frac[0,152] <= 1
 frac[0,153] <= 1
 frac[0,154] <= 1
 frac[0,155] <= 1
 frac[0,156] <= 1
 frac[0,157] <= 1
 frac[0,158] <= 1
 frac[0,159] <= 1
 frac[0,160] <= 1
 frac[0,161] <= 1
 frac[0,162] <= 1
 frac[0,163] <= 1
 frac[0,164] <= 1
 frac[0,165] <= 1
 frac[0,166] <= 1
 frac[0,167] <= 1
 frac[0,168] <= 1
 frac[0,169] <= 1
 frac[0,170] <= 1
 frac[0,171] <= 1
 frac[0,172] <= 1
 frac[0,173] <= 1
 frac[0,174] <= 1
 frac[0,175] <= 1
 frac[0,176] <= 1
 frac[0,177] <= 1
 frac[0,178] <= 1
 frac[0,179] <= 1
 frac[0,180] <= 1
 frac[0,181] <= 1
 frac[0,182] <= 1
 frac[0,183] <= 1
 frac[0,184] <= 1
 frac[0,185] <= 1
 frac[0,186] <= 1
 frac[0,187] <= 1
 frac[0,188] <= 1
 frac[0,189] <= 1
 frac[0,190] <= 1
 frac[0,191] <= 1
 frac[1,0] <= 1
 frac[1,1] <= 1
 frac[1,2] <= 1
 frac[1,3] <= 1
 frac[1,4] <= 1
 frac[1,5] <= 1
 frac[1,6] <= 1
 frac[1,7] <= 1
 frac[1,8] <= 1
 frac[1,9] <= 1
 frac[1,10] <= 1
 frac[1,11] <= 1
 frac[1,12] <= 1
 frac[1,13] <= 1
 frac[1,14] <= 1
 frac[1,15] <= 1
 frac[1,16] <= 1
 frac[1,17] <= 1
 frac[1,18] <= 1
 frac[1,19] <= 1
 frac[1,20] <= 1
 frac[1,21] <= 1
 frac[1,22] <= 1
 frac[1,23] <= 1
 frac[1,24] <= 1
 frac[1,25] <= 1
 frac[1,26] <= 1
 frac[1,27] <= 1
 frac[1,28] <= 1
 frac[1,29] <= 1
 frac[1,30] <= 1
 frac[1,31] <= 1
 frac[1,32] <= 1
 frac[1,33] <= 1
 frac[1,34] <= 1
 frac[1,35] <= 1
 frac[1,36] <= 1
 frac[1,37] <= 1
 frac[1,38] <= 1
 frac[1,39] <= 1
 frac[1,40] <= 1
 frac[1,41] <= 1
 frac[1,42] <= 1
 frac[1,43] <= 1
 frac[1,44] <= 1
 frac[1,45] <= 1
 frac[1,46] <= 1
 frac[1,47] <= 1
 frac[1,48] <= 1
 frac[1,49] <= 1
 frac[1,50] <= 1
 frac[1,51] <= 1
 frac[1,52] <= 1
 frac[1,53] <= 1
 frac[1,54] <= 1
 frac[1,55] <= 1
 frac[1,56] <= 1
 frac[1,57] <= 1
 frac[1,58] <= 1
 frac[1,59] <= 1
 frac[1,60] <= 1
 frac[1,61] <= 1
 frac[1,62] <= 1
 frac[1,63] <= 1
 frac[1,64] <= 1
 frac[1,65] <= 1
 frac[1,66] <= 1
 frac[1,67] <= 1
 frac[1,68] <= 1
 frac[1,69] <= 1
 frac[1,70] <= 1
 frac[1,71] <= 1
 frac[1,72] <= 1
 frac[1,73] <= 1
 frac[1,74] <= 1
 frac[1,75] <= 1
 frac[1,76] <= 1
 frac[1,77] <= 1
 frac[1,78] <= 1
 frac[1,79] <= 1
 frac[1,80] <= 1
 frac[1,81] <= 1
 frac[1,82] <= 1
 frac[1,83] <= 1
 frac[1,84] <= 1
 frac[1,85] <= 1
 frac[1,86] <= 1
 frac[1,87] <= 1
 frac[1,88] <= 1
 frac[1,89] <= 1
 frac[1,90] <= 1
 frac[1,91] <= 1
 frac[1,92] <= 1
 frac[1,93] <= 1
 frac[1,94] <= 1
 frac[1,95] <= 1
 frac[1,96] <= 1
 frac[1,97] <= 1
 frac[1,98] <= 1
 frac[1,99] <= 1
 frac[1,100] <= 1
 frac[1,101] <= 1
 frac[1,102] <= 1
 frac[1,103] <= 1
 frac[1,104] <= 1
 frac[1,105] <= 1
 frac[1,106] <= 1
 frac[1,107] <= 1
 frac[1,108] <= 1
 frac[1,109] <= 1
 frac[1,110] <= 1
 frac[1,111] <= 1
 frac[1,112] <= 1
 frac[1,113] <= 1
 frac[1,114] <= 1
 frac[1,115] <= 1
 frac[1,116] <= 1
 frac[1,117] <= 1
 frac[1,118] <= 1
 frac[1,119] <= 1
 frac[1,120] <= 1
 frac[1,121] <= 1
 frac[1,122] <= 1
 frac[1,123] <= 1
 frac[1,124] <= 1
 frac[1,125] <= 1
 frac[1,126] <= 1
 frac[1,127] <= 1
 frac[1,128] <= 1
 frac[1,129] <= 1
 frac[1,130] <= 1
 frac[1,131] <= 1
 frac[1,132] <= 1
 frac[1,133] <= 1
 frac[1,134] <= 1
 frac[1,135] <= 1
 frac[1,136] <= 1
 frac[1,137] <= 1
 frac[1,138] <= 1
 frac[1,139] <= 1
 frac[1,140] <= 1
 frac[1,141] <= 1
 frac[1,142] <= 1
 frac[1,143] <= 1
 frac[1,144] <= 1
 frac[1,145] <= 1
 frac[1,146] <= 1
 frac[1,147] <= 1
 frac[1,148] <= 1
 frac[1,149] <= 1
 frac[1,150] <= 1
 frac[1,151] <= 1
 frac[1,152] <= 1
 frac[1,153] <= 1
 frac[1,154] <= 1
 frac[1,155] <= 1
 frac[1,156] <= 1
 frac[1,157] <= 1
 frac[1,158] <= 1
 frac[1,159] <= 1
 frac[1,160] <= 1
 frac[1,161] <= 1
 frac[1,162] <= 1
 frac[1,163] <= 1
 frac[1,164] <= 1
 frac[1,165] <= 1
 frac[1,166] <= 1
 frac[1,167] <= 1
 frac[1,168] <= 1
 frac[1,169] <= 1
 frac[1,170] <= 1
 frac[1,171] <= 1
 frac[1,172] <= 1
 frac[1,173] <= 1
 frac[1,174] <= 1
 frac[1,175] <= 1
 frac[1,176] <= 1
 frac[1,177] <= 1
 frac[1,178] <= 1
 frac[1,179] <= 1
 frac[1,180] <= 1
 frac[1,181] <= 1
 frac[1,182] <= 1
 frac[1,183] <= 1
 frac[1,184] <= 1
 frac[1,185] <= 1
 frac[1,186] <= 1
 frac[1,187] <= 1
 frac[1,188] <= 1
 frac[1,189] <= 1
 frac[1,190] <= 1
 frac[1,191] <= 1
 frac[2,0] <= 1
 frac[2,1] <= 1
 frac[2,2] <= 1
 frac[2,3] <= 1
 frac[2,4] <= 1
 frac[2,5] <= 1
 frac[2,6] <= 1
 frac[2,7] <= 1
 frac[2,8] <= 1
 frac[2,9] <= 1
 frac[2,10] <= 1
 frac[2,11] <= 1
 frac[2,12] <= 1
 frac[2,13] <= 1
 frac[2,14] <= 1
 frac[2,15] <= 1
 frac[2,16] <= 1
 frac[2,17] <= 1
 frac[2,18] <= 1
 frac[2,19] <= 1
 frac[2,20] <= 1
 frac[2,21] <= 1
 frac[2,22] <= 1
 frac[2,23] <= 1
 frac[2,24] <= 1
 frac[2,25] <= 1
 frac[2,26] <= 1
 frac[2,27] <= 1
 frac[2,28] <= 1
 frac[2,29] <= 1
 frac[2,30] <= 1
 frac[2,31] <= 1
 frac[2,32] <= 1
 frac[2,33] <= 1
 frac[2,34] <= 1
 frac[2,35] <= 1
 frac[2,36] <= 1
 frac[2,37] <= 1
 frac[2,38] <= 1
 frac[2,39] <= 1
 frac[2,40] <= 1
 frac[2,41] <= 1
 frac[2,42] <= 1
 frac[2,43] <= 1
 frac[2,44] <= 1
 frac[2,45] <= 1
 frac[2,46] <= 1
 frac[2,47] <= 1
 frac[2,48] <= 1
 frac[2,49] <= 1
 frac[2,50] <= 1
 frac[2,51] <= 1
 frac[2,52] <= 1
 frac[2,53] <= 1
 frac[2,54] <= 1
 frac[2,55] <= 1
 frac[2,56] <= 1
 frac[2,57] <= 1
 frac[2,58] <= 1
 frac[2,59] <= 1
 frac[2,60] <= 1
 frac[2,61] <= 1
 frac[2,62] <= 1
 frac[2,63] <= 1
 frac[2,64] <= 1
 frac[2,65] <= 1
 frac[2,66] <= 1
 frac[2,67] <= 1
 frac[2,68] <= 1
 frac[2,69] <= 1
 frac[2,70] <= 1
 frac[2,71] <= 1
 frac[2,72] <= 1
 frac[2,73] <= 1
 frac[2,74] <= 1
 frac[2,75] <= 1
 frac[2,76] <= 1
 frac[2,77] <= 1
 frac[2,78] <= 1
 frac[2,79] <= 1
 frac[2,80] <= 1
 frac[2,81] <= 1
 frac[2,82] <= 1
 frac[2,83] <= 1
 frac[2,84] <= 1
 frac[2,85] <= 1
 frac[2,86] <= 1
 frac[2,87] <= 1
 frac[2,88] <= 1
 frac[2,89] <= 1
 frac[2,90] <= 1
 frac[2,91] <= 1
 frac[2,92] <= 1
 frac[2,93] <= 1
 frac[2,94] <= 1
 frac[2,95] <= 1
 frac[2,96] <= 1
 frac[2,97] <= 1
 frac[2,98] <= 1
 frac[2,99] <= 1
 frac[2,100] <= 1
 frac[2,101] <= 1
 frac[2,102] <= 1
 frac[2,103] <= 1
 frac[2,104] <= 1
 frac[2,105] <= 1
 frac[2,106] <= 1
 frac[2,107] <= 1
 frac[2,108] <= 1
 frac[2,109] <= 1
 frac[2,110] <= 1
 frac[2,111] <= 1
 frac[2,112] <= 1
 frac[2,113] <= 1
 frac[2,114] <= 1
 frac[2,115] <= 1
 frac[2,116] <= 1
 frac[2,117] <= 1
 frac[2,118] <= 1
 frac[2,119] <= 1
 frac[2,120] <= 1
 frac[2,121] <= 1
 frac[2,122] <= 1
 frac[2,123] <= 1
 frac[2,124] <= 1
 frac[2,125] <= 1
 frac[2,126] <= 1
 frac[2,127] <= 1
 frac[2,128] <= 1
 frac[2,129] <= 1
 frac[2,130] <= 1
 frac[2,131] <= 1
 frac[2,132] <= 1
 frac[2,133] <= 1
 frac[2,134] <= 1
 frac[2,135] <= 1
 frac[2,136] <= 1
 frac[2,137] <= 1
 frac[2,138] <= 1
 frac[2,139] <= 1
 frac[2,140] <= 1
 frac[2,141] <= 1
 frac[2,142] <= 1
 frac[2,143] <= 1
 frac[2,144] <= 1
 frac[2,145] <= 1
 frac[2,146] <= 1
 frac[2,147] <= 1
 frac[2,148] <= 1
 frac[2,149] <= 1
 frac[2,150] <= 1
 frac[2,151] <= 1
 frac[2,152] <= 1
 frac[2,153] <= 1
 frac[2,154] <= 1
 frac[2,155] <= 1
 frac[2,156] <= 1
 frac[2,157] <= 1
 frac[2,158] <= 1
 frac[2,159] <= 1
 frac[2,160] <= 1
 frac[2,161] <= 1
 frac[2,162] <= 1
 frac[2,163] <= 1
 frac[2,164] <= 1
 frac[2,165] <= 1
 frac[2,166] <= 1
 frac[2,167] <= 1
 frac[2,168] <= 1
 frac[2,169] <= 1
 frac[2,170] <= 1
 frac[2,171] <= 1
 frac[2,172] <= 1
 frac[2,173] <= 1
 frac[2,174] <= 1
 frac[2,175] <= 1
 frac[2,176] <= 1
 frac[2,177] <= 1
 frac[2,178] <= 1
 frac[2,179] <= 1
 frac[2,180] <= 1
 frac[2,181] <= 1
 frac[2,182] <= 1
 frac[2,183] <= 1
 frac[2,184] <= 1
 frac[2,185] <= 1
 frac[2,186] <= 1
 frac[2,187] <= 1
 frac[2,188] <= 1
 frac[2,189] <= 1
 frac[2,190] <= 1
 frac[2,191] <= 1
 frac[3,0] <= 1
 frac[3,1] <= 1
 frac[3,2] <= 1
 frac[3,3] <= 1
 frac[3,4] <= 1
 frac[3,5] <= 1
 frac[3,6] <= 1
 frac[3,7] <= 1
 frac[3,8] <= 1
 frac[3,9] <= 1
 frac[3,10] <= 1
 frac[3,11] <= 1
 frac[3,12] <= 1
 frac[3,13] <= 1
 frac[3,14] <= 1
 frac[3,15] <= 1
 frac[3,16] <= 1
 frac[3,17] <= 1
 frac[3,18] <= 1
 frac[3,19] <= 1
 frac[3,20] <= 1
 frac[3,21] <= 1
 frac[3,22] <= 1
 frac[3,23] <= 1
 frac[3,24] <= 1
 frac[3,25] <= 1
 frac[3,26] <= 1
 frac[3,27] <= 1
 frac[3,28] <= 1
 frac[3,29] <= 1
 frac[3,30] <= 1
 frac[3,31] <= 1
 frac[3,32] <= 1
 frac[3,33] <= 1
 frac[3,34] <= 1
 frac[3,35] <= 1
 frac[3,36] <= 1
 frac[3,37] <= 1
 frac[3,38] <= 1
 frac[3,39] <= 1
 frac[3,40] <= 1
 frac[3,41] <= 1
 frac[3,42] <= 1
 frac[3,43] <= 1
 frac[3,44] <= 1
 frac[3,45] <= 1
 frac[3,46] <= 1
 frac[3,47] <= 1
 frac[3,48] <= 1
 frac[3,49] <= 1
 frac[3,50] <= 1
 frac[3,51] <= 1
 frac[3,52] <= 1
 frac[3,53] <= 1
 frac[3,54] <= 1
 frac[3,55] <= 1
 frac[3,56] <= 1
 frac[3,57] <= 1
 frac[3,58] <= 1
 frac[3,59] <= 1
 frac[3,60] <= 1
 frac[3,61] <= 1
 frac[3,62] <= 1
 frac[3,63] <= 1
 frac[3,64] <= 1
 frac[3,65] <= 1
 frac[3,66] <= 1
 frac[3,67] <= 1
 frac[3,68] <= 1
 frac[3,69] <= 1
 frac[3,70] <= 1
 frac[3,71] <= 1
 frac[3,72] <= 1
 frac[3,73] <= 1
 frac[3,74] <= 1
 frac[3,75] <= 1
 frac[3,76] <= 1
 frac[3,77] <= 1
 frac[3,78] <= 1
 frac[3,79] <= 1
 frac[3,80] <= 1
 frac[3,81] <= 1
 frac[3,82] <= 1
 frac[3,83] <= 1
 frac[3,84] <= 1
 frac[3,85] <= 1
 frac[3,86] <= 1
 frac[3,87] <= 1
 frac[3,88] <= 1
 frac[3,89] <= 1
 frac[3,90] <= 1
 frac[3,91] <= 1
 frac[3,92] <= 1
 frac[3,93] <= 1
 frac[3,94] <= 1
 frac[3,95] <= 1
 frac[3,96] <= 1
 frac[3,97] <= 1
 frac[3,98] <= 1
 frac[3,99] <= 1
 frac[3,100] <= 1
 frac[3,101] <= 1
 frac[3,102] <= 1
 frac[3,103] <= 1
 frac[3,104] <= 1
 frac[3,105] <= 1
 frac[3,106] <= 1
 frac[3,107] <= 1
 frac[3,108] <= 1
 frac[3,109] <= 1
 frac[3,110] <= 1
 frac[3,111] <= 1
 frac[3,112] <= 1
 frac[3,113] <= 1
 frac[3,114] <= 1
 frac[3,115] <= 1
 frac[3,116] <= 1
 frac[3,117] <= 1
 frac[3,118] <= 1
 frac[3,119] <= 1
 frac[3,120] <= 1
 frac[3,121] <= 1
 frac[3,122] <= 1
 frac[3,123] <= 1
 frac[3,124] <= 1
 frac[3,125] <= 1
 frac[3,126] <= 1
 frac[3,127] <= 1
 frac[3,128] <= 1
 frac[3,129] <= 1
 frac[3,130] <= 1
 frac[3,131] <= 1
 frac[3,132] <= 1
 frac[3,133] <= 1
 frac[3,134] <= 1
 frac[3,135] <= 1
 frac[3,136] <= 1
 frac[3,137] <= 1
 frac[3,138] <= 1
 frac[3,139] <= 1
 frac[3,140] <= 1
 frac[3,141] <= 1
 frac[3,142] <= 1
 frac[3,143] <= 1
 frac[3,144] <= 1
 frac[3,145] <= 1
 frac[3,146] <= 1
 frac[3,147] <= 1
 frac[3,148] <= 1
 frac[3,149] <= 1
 frac[3,150] <= 1
 frac[3,151] <= 1
 frac[3,152] <= 1
 frac[3,153] <= 1
 frac[3,154] <= 1
 frac[3,155] <= 1
 frac[3,156] <= 1
 frac[3,157] <= 1
 frac[3,158] <= 1
 frac[3,159] <= 1
 frac[3,160] <= 1
 frac[3,161] <= 1
 frac[3,162] <= 1
 frac[3,163] <= 1
 frac[3,164] <= 1
 frac[3,165] <= 1
 frac[3,166] <= 1
 frac[3,167] <= 1
 frac[3,168] <= 1
 frac[3,169] <= 1
 frac[3,170] <= 1
 frac[3,171] <= 1
 frac[3,172] <= 1
 frac[3,173] <= 1
 frac[3,174] <= 1
 frac[3,175] <= 1
 frac[3,176] <= 1
 frac[3,177] <= 1
 frac[3,178] <= 1
 frac[3,179] <= 1
 frac[3,180] <= 1
 frac[3,181] <= 1
 frac[3,182] <= 1
 frac[3,183] <= 1
 frac[3,184] <= 1
 frac[3,185] <= 1
 frac[3,186] <= 1
 frac[3,187] <= 1
 frac[3,188] <= 1
 frac[3,189] <= 1
 frac[3,190] <= 1
 frac[3,191] <= 1
 frac[4,0] <= 1
 frac[4,1] <= 1
 frac[4,2] <= 1
 frac[4,3] <= 1
 frac[4,4] <= 1
 frac[4,5] <= 1
 frac[4,6] <= 1
 frac[4,7] <= 1
 frac[4,8] <= 1
 frac[4,9] <= 1
 frac[4,10] <= 1
 frac[4,11] <= 1
 frac[4,12] <= 1
 frac[4,13] <= 1
 frac[4,14] <= 1
 frac[4,15] <= 1
 frac[4,16] <= 1
 frac[4,17] <= 1
 frac[4,18] <= 1
 frac[4,19] <= 1
 frac[4,20] <= 1
 frac[4,21] <= 1
 frac[4,22] <= 1
 frac[4,23] <= 1
 frac[4,24] <= 1
 frac[4,25] <= 1
 frac[4,26] <= 1
 frac[4,27] <= 1
 frac[4,28] <= 1
 frac[4,29] <= 1
 frac[4,30] <= 1
 frac[4,31] <= 1
 frac[4,32] <= 1
 frac[4,33] <= 1
 frac[4,34] <= 1
 frac[4,35] <= 1
 frac[4,36] <= 1
 frac[4,37] <= 1
 frac[4,38] <= 1
 frac[4,39] <= 1
 frac[4,40] <= 1
 frac[4,41] <= 1
 frac[4,42] <= 1
 frac[4,43] <= 1
 frac[4,44] <= 1
 frac[4,45] <= 1
 frac[4,46] <= 1
 frac[4,47] <= 1
 frac[4,48] <= 1
 frac[4,49] <= 1
 frac[4,50] <= 1
 frac[4,51] <= 1
 frac[4,52] <= 1
 frac[4,53] <= 1
 frac[4,54] <= 1
 frac[4,55] <= 1
 frac[4,56] <= 1
 frac[4,57] <= 1
 frac[4,58] <= 1
 frac[4,59] <= 1
 frac[4,60] <= 1
 frac[4,61] <= 1
 frac[4,62] <= 1
 frac[4,63] <= 1
 frac[4,64] <= 1
 frac[4,65] <= 1
 frac[4,66] <= 1
 frac[4,67] <= 1
 frac[4,68] <= 1
 frac[4,69] <= 1
 frac[4,70] <= 1
 frac[4,71] <= 1
 frac[4,72] <= 1
 frac[4,73] <= 1
 frac[4,74] <= 1
 frac[4,75] <= 1
 frac[4,76] <= 1
 frac[4,77] <= 1
 frac[4,78] <= 1
 frac[4,79] <= 1
 frac[4,80] <= 1
 frac[4,81] <= 1
 frac[4,82] <= 1
 frac[4,83] <= 1
 frac[4,84] <= 1
 frac[4,85] <= 1
 frac[4,86] <= 1
 frac[4,87] <= 1
 frac[4,88] <= 1
 frac[4,89] <= 1
 frac[4,90] <= 1
 frac[4,91] <= 1
 frac[4,92] <= 1
 frac[4,93] <= 1
 frac[4,94] <= 1
 frac[4,95] <= 1
 frac[4,96] <= 1
 frac[4,97] <= 1
 frac[4,98] <= 1
 frac[4,99] <= 1
 frac[4,100] <= 1
 frac[4,101] <= 1
 frac[4,102] <= 1
 frac[4,103] <= 1
 frac[4,104] <= 1
 frac[4,105] <= 1
 frac[4,106] <= 1
 frac[4,107] <= 1
 frac[4,108] <= 1
 frac[4,109] <= 1
 frac[4,110] <= 1
 frac[4,111] <= 1
 frac[4,112] <= 1
 frac[4,113] <= 1
 frac[4,114] <= 1
 frac[4,115] <= 1
 frac[4,116] <= 1
 frac[4,117] <= 1
 frac[4,118] <= 1
 frac[4,119] <= 1
 frac[4,120] <= 1
 frac[4,121] <= 1
 frac[4,122] <= 1
 frac[4,123] <= 1
 frac[4,124] <= 1
 frac[4,125] <= 1
 frac[4,126] <= 1
 frac[4,127] <= 1
 frac[4,128] <= 1
 frac[4,129] <= 1
 frac[4,130] <= 1
 frac[4,131] <= 1
 frac[4,132] <= 1
 frac[4,133] <= 1
 frac[4,134] <= 1
 frac[4,135] <= 1
 frac[4,136] <= 1
 frac[4,137] <= 1
 frac[4,138] <= 1
 frac[4,139] <= 1
 frac[4,140] <= 1
 frac[4,141] <= 1
 frac[4,142] <= 1
 frac[4,143] <= 1
 frac[4,144] <= 1
 frac[4,145] <= 1
 frac[4,146] <= 1
 frac[4,147] <= 1
 frac[4,148] <= 1
 frac[4,149] <= 1
 frac[4,150] <= 1
 frac[4,151] <= 1
 frac[4,152] <= 1
 frac[4,153] <= 1
 frac[4,154] <= 1
 frac[4,155] <= 1
 frac[4,156] <= 1
 frac[4,157] <= 1
 frac[4,158] <= 1
 frac[4,159] <= 1
 frac[4,160] <= 1
 frac[4,161] <= 1
 frac[4,162] <= 1
 frac[4,163] <= 1
 frac[4,164] <= 1
 frac[4,165] <= 1
 frac[4,166] <= 1
 frac[4,167] <= 1
 frac[4,168] <= 1
 frac[4,169] <= 1
 frac[4,170] <= 1
 frac[4,171] <= 1
 frac[4,172] <= 1
 frac[4,173] <= 1
 frac[4,174] <= 1
 frac[4,175] <= 1
 frac[4,176] <= 1
 frac[4,177] <= 1
 frac[4,178] <= 1
 frac[4,179] <= 1
 frac[4,180] <= 1
 frac[4,181] <= 1
 frac[4,182] <= 1
 frac[4,183] <= 1
 frac[4,184] <= 1
 frac[4,185] <= 1
 frac[4,186] <= 1
 frac[4,187] <= 1
 frac[4,188] <= 1
 frac[4,189] <= 1
 frac[4,190] <= 1
 frac[4,191] <= 1
 frac[5,0] <= 1
 frac[5,1] <= 1
 frac[5,2] <= 1
 frac[5,3] <= 1
 frac[5,4] <= 1
 frac[5,5] <= 1
 frac[5,6] <= 1
 frac[5,7] <= 1
 frac[5,8] <= 1
 frac[5,9] <= 1
 frac[5,10] <= 1
 frac[5,11] <= 1
 frac[5,12] <= 1
 frac[5,13] <= 1
 frac[5,14] <= 1
 frac[5,15] <= 1
 frac[5,16] <= 1
 frac[5,17] <= 1
 frac[5,18] <= 1
 frac[5,19] <= 1
 frac[5,20] <= 1
 frac[5,21] <= 1
 frac[5,22] <= 1
 frac[5,23] <= 1
 frac[5,24] <= 1
 frac[5,25] <= 1
 frac[5,26] <= 1
 frac[5,27] <= 1
 frac[5,28] <= 1
 frac[5,29] <= 1
 frac[5,30] <= 1
 frac[5,31] <= 1
 frac[5,32] <= 1
 frac[5,33] <= 1
 frac[5,34] <= 1
 frac[5,35] <= 1
 frac[5,36] <= 1
 frac[5,37] <= 1
 frac[5,38] <= 1
 frac[5,39] <= 1
 frac[5,40] <= 1
 frac[5,41] <= 1
 frac[5,42] <= 1
 frac[5,43] <= 1
 frac[5,44] <= 1
 frac[5,45] <= 1
 frac[5,46] <= 1
 frac[5,47] <= 1
 frac[5,48] <= 1
 frac[5,49] <= 1
 frac[5,50] <= 1
 frac[5,51] <= 1
 frac[5,52] <= 1
 frac[5,53] <= 1
 frac[5,54] <= 1
 frac[5,55] <= 1
 frac[5,56] <= 1
 frac[5,57] <= 1
 frac[5,58] <= 1
 frac[5,59] <= 1
 frac[5,60] <= 1
 frac[5,61] <= 1
 frac[5,62] <= 1
 frac[5,63] <= 1
 frac[5,64] <= 1
 frac[5,65] <= 1
 frac[5,66] <= 1
 frac[5,67] <= 1
 frac[5,68] <= 1
 frac[5,69] <= 1
 frac[5,70] <= 1
 frac[5,71] <= 1
 frac[5,72] <= 1
 frac[5,73] <= 1
 frac[5,74] <= 1
 frac[5,75] <= 1
 frac[5,76] <= 1
 frac[5,77] <= 1
 frac[5,78] <= 1
 frac[5,79] <= 1
 frac[5,80] <= 1
 frac[5,81] <= 1
 frac[5,82] <= 1
 frac[5,83] <= 1
 frac[5,84] <= 1
 frac[5,85] <= 1
 frac[5,86] <= 1
 frac[5,87] <= 1
 frac[5,88] <= 1
 frac[5,89] <= 1
 frac[5,90] <= 1
 frac[5,91] <= 1
 frac[5,92] <= 1
 frac[5,93] <= 1
 frac[5,94] <= 1
 frac[5,95] <= 1
 frac[5,96] <= 1
 frac[5,97] <= 1
 frac[5,98] <= 1
 frac[5,99] <= 1
 frac[5,100] <= 1
 frac[5,101] <= 1
 frac[5,102] <= 1
 frac[5,103] <= 1
 frac[5,104] <= 1
 frac[5,105] <= 1
 frac[5,106] <= 1
 frac[5,107] <= 1
 frac[5,108] <= 1
 frac[5,109] <= 1
 frac[5,110] <= 1
 frac[5,111] <= 1
 frac[5,112] <= 1
 frac[5,113] <= 1
 frac[5,114] <= 1
 frac[5,115] <= 1
 frac[5,116] <= 1
 frac[5,117] <= 1
 frac[5,118] <= 1
 frac[5,119] <= 1
 frac[5,120] <= 1
 frac[5,121] <= 1
 frac[5,122] <= 1
 frac[5,123] <= 1
 frac[5,124] <= 1
 frac[5,125] <= 1
 frac[5,126] <= 1
 frac[5,127] <= 1
 frac[5,128] <= 1
 frac[5,129] <= 1
 frac[5,130] <= 1
 frac[5,131] <= 1
 frac[5,132] <= 1
 frac[5,133] <= 1
 frac[5,134] <= 1
 frac[5,135] <= 1
 frac[5,136] <= 1
 frac[5,137] <= 1
 frac[5,138] <= 1
 frac[5,139] <= 1
 frac[5,140] <= 1
 frac[5,141] <= 1
 frac[5,142] <= 1
 frac[5,143] <= 1
 frac[5,144] <= 1
 frac[5,145] <= 1
 frac[5,146] <= 1
 frac[5,147] <= 1
 frac[5,148] <= 1
 frac[5,149] <= 1
 frac[5,150] <= 1
 frac[5,151] <= 1
 frac[5,152] <= 1
 frac[5,153] <= 1
 frac[5,154] <= 1
 frac[5,155] <= 1
 frac[5,156] <= 1
 frac[5,157] <= 1
 frac[5,158] <= 1
 frac[5,159] <= 1
 frac[5,160] <= 1
 frac[5,161] <= 1
 frac[5,162] <= 1
 frac[5,163] <= 1
 frac[5,164] <= 1
 frac[5,165] <= 1
 frac[5,166] <= 1
 frac[5,167] <= 1
 frac[5,168] <= 1
 frac[5,169] <= 1
 frac[5,170] <= 1
 frac[5,171] <= 1
 frac[5,172] <= 1
 frac[5,173] <= 1
 frac[5,174] <= 1
 frac[5,175] <= 1
 frac[5,176] <= 1
 frac[5,177] <= 1
 frac[5,178] <= 1
 frac[5,179] <= 1
 frac[5,180] <= 1
 frac[5,181] <= 1
 frac[5,182] <= 1
 frac[5,183] <= 1
 frac[5,184] <= 1
 frac[5,185] <= 1
 frac[5,186] <= 1
 frac[5,187] <= 1
 frac[5,188] <= 1
 frac[5,189] <= 1
 frac[5,190] <= 1
 frac[5,191] <= 1
 frac[6,0] <= 1
 frac[6,1] <= 1
 frac[6,2] <= 1
 frac[6,3] <= 1
 frac[6,4] <= 1
 frac[6,5] <= 1
 frac[6,6] <= 1
 frac[6,7] <= 1
 frac[6,8] <= 1
 frac[6,9] <= 1
 frac[6,10] <= 1
 frac[6,11] <= 1
 frac[6,12] <= 1
 frac[6,13] <= 1
 frac[6,14] <= 1
 frac[6,15] <= 1
 frac[6,16] <= 1
 frac[6,17] <= 1
 frac[6,18] <= 1
 frac[6,19] <= 1
 frac[6,20] <= 1
 frac[6,21] <= 1
 frac[6,22] <= 1
 frac[6,23] <= 1
 frac[6,24] <= 1
 frac[6,25] <= 1
 frac[6,26] <= 1
 frac[6,27] <= 1
 frac[6,28] <= 1
 frac[6,29] <= 1
 frac[6,30] <= 1
 frac[6,31] <= 1
 frac[6,32] <= 1
 frac[6,33] <= 1
 frac[6,34] <= 1
 frac[6,35] <= 1
 frac[6,36] <= 1
 frac[6,37] <= 1
 frac[6,38] <= 1
 frac[6,39] <= 1
 frac[6,40] <= 1
 frac[6,41] <= 1
 frac[6,42] <= 1
 frac[6,43] <= 1
 frac[6,44] <= 1
 frac[6,45] <= 1
 frac[6,46] <= 1
 frac[6,47] <= 1
 frac[6,48] <= 1
 frac[6,49] <= 1
 frac[6,50] <= 1
 frac[6,51] <= 1
 frac[6,52] <= 1
 frac[6,53] <= 1
 frac[6,54] <= 1
 frac[6,55] <= 1
 frac[6,56] <= 1
 frac[6,57] <= 1
 frac[6,58] <= 1
 frac[6,59] <= 1
 frac[6,60] <= 1
 frac[6,61] <= 1
 frac[6,62] <= 1
 frac[6,63] <= 1
 frac[6,64] <= 1
 frac[6,65] <= 1
 frac[6,66] <= 1
 frac[6,67] <= 1
 frac[6,68] <= 1
 frac[6,69] <= 1
 frac[6,70] <= 1
 frac[6,71] <= 1
 frac[6,72] <= 1
 frac[6,73] <= 1
 frac[6,74] <= 1
 frac[6,75] <= 1
 frac[6,76] <= 1
 frac[6,77] <= 1
 frac[6,78] <= 1
 frac[6,79] <= 1
 frac[6,80] <= 1
 frac[6,81] <= 1
 frac[6,82] <= 1
 frac[6,83] <= 1
 frac[6,84] <= 1
 frac[6,85] <= 1
 frac[6,86] <= 1
 frac[6,87] <= 1
 frac[6,88] <= 1
 frac[6,89] <= 1
 frac[6,90] <= 1
 frac[6,91] <= 1
 frac[6,92] <= 1
 frac[6,93] <= 1
 frac[6,94] <= 1
 frac[6,95] <= 1
 frac[6,96] <= 1
 frac[6,97] <= 1
 frac[6,98] <= 1
 frac[6,99] <= 1
 frac[6,100] <= 1
 frac[6,101] <= 1
 frac[6,102] <= 1
 frac[6,103] <= 1
 frac[6,104] <= 1
 frac[6,105] <= 1
 frac[6,106] <= 1
 frac[6,107] <= 1
 frac[6,108] <= 1
 frac[6,109] <= 1
 frac[6,110] <= 1
 frac[6,111] <= 1
 frac[6,112] <= 1
 frac[6,113] <= 1
 frac[6,114] <= 1
 frac[6,115] <= 1
 frac[6,116] <= 1
 frac[6,117] <= 1
 frac[6,118] <= 1
 frac[6,119] <= 1
 frac[6,120] <= 1
 frac[6,121] <= 1
 frac[6,122] <= 1
 frac[6,123] <= 1
 frac[6,124] <= 1
 frac[6,125] <= 1
 frac[6,126] <= 1
 frac[6,127] <= 1
 frac[6,128] <= 1
 frac[6,129] <= 1
 frac[6,130] <= 1
 frac[6,131] <= 1
 frac[6,132] <= 1
 frac[6,133] <= 1
 frac[6,134] <= 1
 frac[6,135] <= 1
 frac[6,136] <= 1
 frac[6,137] <= 1
 frac[6,138] <= 1
 frac[6,139] <= 1
 frac[6,140] <= 1
 frac[6,141] <= 1
 frac[6,142] <= 1
 frac[6,143] <= 1
 frac[6,144] <= 1
 frac[6,145] <= 1
 frac[6,146] <= 1
 frac[6,147] <= 1
 frac[6,148] <= 1
 frac[6,149] <= 1
 frac[6,150] <= 1
 frac[6,151] <= 1
 frac[6,152] <= 1
 frac[6,153] <= 1
 frac[6,154] <= 1
 frac[6,155] <= 1
 frac[6,156] <= 1
 frac[6,157] <= 1
 frac[6,158] <= 1
 frac[6,159] <= 1
 frac[6,160] <= 1
 frac[6,161] <= 1
 frac[6,162] <= 1
 frac[6,163] <= 1
 frac[6,164] <= 1
 frac[6,165] <= 1
 frac[6,166] <= 1
 frac[6,167] <= 1
 frac[6,168] <= 1
 frac[6,169] <= 1
 frac[6,170] <= 1
 frac[6,171] <= 1
 frac[6,172] <= 1
 frac[6,173] <= 1
 frac[6,174] <= 1
 frac[6,175] <= 1
 frac[6,176] <= 1
 frac[6,177] <= 1
 frac[6,178] <= 1
 frac[6,179] <= 1
 frac[6,180] <= 1
 frac[6,181] <= 1
 frac[6,182] <= 1
 frac[6,183] <= 1
 frac[6,184] <= 1
 frac[6,185] <= 1
 frac[6,186] <= 1
 frac[6,187] <= 1
 frac[6,188] <= 1
 frac[6,189] <= 1
 frac[6,190] <= 1
 frac[6,191] <= 1
 frac[7,0] <= 1
 frac[7,1] <= 1
 frac[7,2] <= 1
 frac[7,3] <= 1
 frac[7,4] <= 1
 frac[7,5] <= 1
 frac[7,6] <= 1
 frac[7,7] <= 1
 frac[7,8] <= 1
 frac[7,9] <= 1
 frac[7,10] <= 1
 frac[7,11] <= 1
 frac[7,12] <= 1
 frac[7,13] <= 1
 frac[7,14] <= 1
 frac[7,15] <= 1
 frac[7,16] <= 1
 frac[7,17] <= 1
 frac[7,18] <= 1
 frac[7,19] <= 1
 frac[7,20] <= 1
 frac[7,21] <= 1
 frac[7,22] <= 1
 frac[7,23] <= 1
 frac[7,24] <= 1
 frac[7,25] <= 1
 frac[7,26] <= 1
 frac[7,27] <= 1
 frac[7,28] <= 1
 frac[7,29] <= 1
 frac[7,30] <= 1
 frac[7,31] <= 1
 frac[7,32] <= 1
 frac[7,33] <= 1
 frac[7,34] <= 1
 frac[7,35] <= 1
 frac[7,36] <= 1
 frac[7,37] <= 1
 frac[7,38] <= 1
 frac[7,39] <= 1
 frac[7,40] <= 1
 frac[7,41] <= 1
 frac[7,42] <= 1
 frac[7,43] <= 1
 frac[7,44] <= 1
 frac[7,45] <= 1
 frac[7,46] <= 1
 frac[7,47] <= 1
 frac[7,48] <= 1
 frac[7,49] <= 1
 frac[7,50] <= 1
 frac[7,51] <= 1
 frac[7,52] <= 1
 frac[7,53] <= 1
 frac[7,54] <= 1
 frac[7,55] <= 1
 frac[7,56] <= 1
 frac[7,57] <= 1
 frac[7,58] <= 1
 frac[7,59] <= 1
 frac[7,60] <= 1
 frac[7,61] <= 1
 frac[7,62] <= 1
 frac[7,63] <= 1
 frac[7,64] <= 1
 frac[7,65] <= 1
 frac[7,66] <= 1
 frac[7,67] <= 1
 frac[7,68] <= 1
 frac[7,69] <= 1
 frac[7,70] <= 1
 frac[7,71] <= 1
 frac[7,72] <= 1
 frac[7,73] <= 1
 frac[7,74] <= 1
 frac[7,75] <= 1
 frac[7,76] <= 1
 frac[7,77] <= 1
 frac[7,78] <= 1
 frac[7,79] <= 1
 frac[7,80] <= 1
 frac[7,81] <= 1
 frac[7,82] <= 1
 frac[7,83] <= 1
 frac[7,84] <= 1
 frac[7,85] <= 1
 frac[7,86] <= 1
 frac[7,87] <= 1
 frac[7,88] <= 1
 frac[7,89] <= 1
 frac[7,90] <= 1
 frac[7,91] <= 1
 frac[7,92] <= 1
 frac[7,93] <= 1
 frac[7,94] <= 1
 frac[7,95] <= 1
 frac[7,96] <= 1
 frac[7,97] <= 1
 frac[7,98] <= 1
 frac[7,99] <= 1
 frac[7,100] <= 1
 frac[7,101] <= 1
 frac[7,102] <= 1
 frac[7,103] <= 1
 frac[7,104] <= 1
 frac[7,105] <= 1
 frac[7,106] <= 1
 frac[7,107] <= 1
 frac[7,108] <= 1
 frac[7,109] <= 1
 frac[7,110] <= 1
 frac[7,111] <= 1
 frac[7,112] <= 1
 frac[7,113] <= 1
 frac[7,114] <= 1
 frac[7,115] <= 1
 frac[7,116] <= 1
 frac[7,117] <= 1
 frac[7,118] <= 1
 frac[7,119] <= 1
 frac[7,120] <= 1
 frac[7,121] <= 1
 frac[7,122] <= 1
 frac[7,123] <= 1
 frac[7,124] <= 1
 frac[7,125] <= 1
 frac[7,126] <= 1
 frac[7,127] <= 1
 frac[7,128] <= 1
 frac[7,129] <= 1
 frac[7,130] <= 1
 frac[7,131] <= 1
 frac[7,132] <= 1
 frac[7,133] <= 1
 frac[7,134] <= 1
 frac[7,135] <= 1
 frac[7,136] <= 1
 frac[7,137] <= 1
 frac[7,138] <= 1
 frac[7,139] <= 1
 frac[7,140] <= 1
 frac[7,141] <= 1
 frac[7,142] <= 1
 frac[7,143] <= 1
 frac[7,144] <= 1
 frac[7,145] <= 1
 frac[7,146] <= 1
 frac[7,147] <= 1
 frac[7,148] <= 1
 frac[7,149] <= 1
 frac[7,150] <= 1
 frac[7,151] <= 1
 frac[7,152] <= 1
 frac[7,153] <= 1
 frac[7,154] <= 1
 frac[7,155] <= 1
 frac[7,156] <= 1
 frac[7,157] <= 1
 frac[7,158] <= 1
 frac[7,159] <= 1
 frac[7,160] <= 1
 frac[7,161] <= 1
 frac[7,162] <= 1
 frac[7,163] <= 1
 frac[7,164] <= 1
 frac[7,165] <= 1
 frac[7,166] <= 1
 frac[7,167] <= 1
 frac[7,168] <= 1
 frac[7,169] <= 1
 frac[7,170] <= 1
 frac[7,171] <= 1
 frac[7,172] <= 1
 frac[7,173] <= 1
 frac[7,174] <= 1
 frac[7,175] <= 1
 frac[7,176] <= 1
 frac[7,177] <= 1
 frac[7,178] <= 1
 frac[7,179] <= 1
 frac[7,180] <= 1
 frac[7,181] <= 1
 frac[7,182] <= 1
 frac[7,183] <= 1
 frac[7,184] <= 1
 frac[7,185] <= 1
 frac[7,186] <= 1
 frac[7,187] <= 1
 frac[7,188] <= 1
 frac[7,189] <= 1
 frac[7,190] <= 1
 frac[7,191] <= 1
 frac[8,0] <= 1
 frac[8,1] <= 1
 frac[8,2] <= 1
 frac[8,3] <= 1
 frac[8,4] <= 1
 frac[8,5] <= 1
 frac[8,6] <= 1
 frac[8,7] <= 1
 frac[8,8] <= 1
 frac[8,9] <= 1
 frac[8,10] <= 1
 frac[8,11] <= 1
 frac[8,12] <= 1
 frac[8,13] <= 1
 frac[8,14] <= 1
 frac[8,15] <= 1
 frac[8,16] <= 1
 frac[8,17] <= 1
 frac[8,18] <= 1
 frac[8,19] <= 1
 frac[8,20] <= 1
 frac[8,21] <= 1
 frac[8,22] <= 1
 frac[8,23] <= 1
 frac[8,24] <= 1
 frac[8,25] <= 1
 frac[8,26] <= 1
 frac[8,27] <= 1
 frac[8,28] <= 1
 frac[8,29] <= 1
 frac[8,30] <= 1
 frac[8,31] <= 1
 frac[8,32] <= 1
 frac[8,33] <= 1
 frac[8,34] <= 1
 frac[8,35] <= 1
 frac[8,36] <= 1
 frac[8,37] <= 1
 frac[8,38] <= 1
 frac[8,39] <= 1
 frac[8,40] <= 1
 frac[8,41] <= 1
 frac[8,42] <= 1
 frac[8,43] <= 1
 frac[8,44] <= 1
 frac[8,45] <= 1
 frac[8,46] <= 1
 frac[8,47] <= 1
 frac[8,48] <= 1
 frac[8,49] <= 1
 frac[8,50] <= 1
 frac[8,51] <= 1
 frac[8,52] <= 1
 frac[8,53] <= 1
 frac[8,54] <= 1
 frac[8,55] <= 1
 frac[8,56] <= 1
 frac[8,57] <= 1
 frac[8,58] <= 1
 frac[8,59] <= 1
 frac[8,60] <= 1
 frac[8,61] <= 1
 frac[8,62] <= 1
 frac[8,63] <= 1
 frac[8,64] <= 1
 frac[8,65] <= 1
 frac[8,66] <= 1
 frac[8,67] <= 1
 frac[8,68] <= 1
 frac[8,69] <= 1
 frac[8,70] <= 1
 frac[8,71] <= 1
 frac[8,72] <= 1
 frac[8,73] <= 1
 frac[8,74] <= 1
 frac[8,75] <= 1
 frac[8,76] <= 1
 frac[8,77] <= 1
 frac[8,78] <= 1
 frac[8,79] <= 1
 frac[8,80] <= 1
 frac[8,81] <= 1
 frac[8,82] <= 1
 frac[8,83] <= 1
 frac[8,84] <= 1
 frac[8,85] <= 1
 frac[8,86] <= 1
 frac[8,87] <= 1
 frac[8,88] <= 1
 frac[8,89] <= 1
 frac[8,90] <= 1
 frac[8,91] <= 1
 frac[8,92] <= 1
 frac[8,93] <= 1
 frac[8,94] <= 1
 frac[8,95] <= 1
 frac[8,96] <= 1
 frac[8,97] <= 1
 frac[8,98] <= 1
 frac[8,99] <= 1
 frac[8,100] <= 1
 frac[8,101] <= 1
 frac[8,102] <= 1
 frac[8,103] <= 1
 frac[8,104] <= 1
 frac[8,105] <= 1
 frac[8,106] <= 1
 frac[8,107] <= 1
 frac[8,108] <= 1
 frac[8,109] <= 1
 frac[8,110] <= 1
 frac[8,111] <= 1
 frac[8,112] <= 1
 frac[8,113] <= 1
 frac[8,114] <= 1
 frac[8,115] <= 1
 frac[8,116] <= 1
 frac[8,117] <= 1
 frac[8,118] <= 1
 frac[8,119] <= 1
 frac[8,120] <= 1
 frac[8,121] <= 1
 frac[8,122] <= 1
 frac[8,123] <= 1
 frac[8,124] <= 1
 frac[8,125] <= 1
 frac[8,126] <= 1
 frac[8,127] <= 1
 frac[8,128] <= 1
 frac[8,129] <= 1
 frac[8,130] <= 1
 frac[8,131] <= 1
 frac[8,132] <= 1
 frac[8,133] <= 1
 frac[8,134] <= 1
 frac[8,135] <= 1
 frac[8,136] <= 1
 frac[8,137] <= 1
 frac[8,138] <= 1
 frac[8,139] <= 1
 frac[8,140] <= 1
 frac[8,141] <= 1
 frac[8,142] <= 1
 frac[8,143] <= 1
 frac[8,144] <= 1
 frac[8,145] <= 1
 frac[8,146] <= 1
 frac[8,147] <= 1
 frac[8,148] <= 1
 frac[8,149] <= 1
 frac[8,150] <= 1
 frac[8,151] <= 1
 frac[8,152] <= 1
 frac[8,153] <= 1
 frac[8,154] <= 1
 frac[8,155] <= 1
 frac[8,156] <= 1
 frac[8,157] <= 1
 frac[8,158] <= 1
 frac[8,159] <= 1
 frac[8,160] <= 1
 frac[8,161] <= 1
 frac[8,162] <= 1
 frac[8,163] <= 1
 frac[8,164] <= 1
 frac[8,165] <= 1
 frac[8,166] <= 1
 frac[8,167] <= 1
 frac[8,168] <= 1
 frac[8,169] <= 1
 frac[8,170] <= 1
 frac[8,171] <= 1
 frac[8,172] <= 1
 frac[8,173] <= 1
 frac[8,174] <= 1
 frac[8,175] <= 1
 frac[8,176] <= 1
 frac[8,177] <= 1
 frac[8,178] <= 1
 frac[8,179] <= 1
 frac[8,180] <= 1
 frac[8,181] <= 1
 frac[8,182] <= 1
 frac[8,183] <= 1
 frac[8,184] <= 1
 frac[8,185] <= 1
 frac[8,186] <= 1
 frac[8,187] <= 1
 frac[8,188] <= 1
 frac[8,189] <= 1
 frac[8,190] <= 1
 frac[8,191] <= 1
 frac[9,0] <= 1
 frac[9,1] <= 1
 frac[9,2] <= 1
 frac[9,3] <= 1
 frac[9,4] <= 1
 frac[9,5] <= 1
 frac[9,6] <= 1
 frac[9,7] <= 1
 frac[9,8] <= 1
 frac[9,9] <= 1
 frac[9,10] <= 1
 frac[9,11] <= 1
 frac[9,12] <= 1
 frac[9,13] <= 1
 frac[9,14] <= 1
 frac[9,15] <= 1
 frac[9,16] <= 1
 frac[9,17] <= 1
 frac[9,18] <= 1
 frac[9,19] <= 1
 frac[9,20] <= 1
 frac[9,21] <= 1
 frac[9,22] <= 1
 frac[9,23] <= 1
 frac[9,24] <= 1
 frac[9,25] <= 1
 frac[9,26] <= 1
 frac[9,27] <= 1
 frac[9,28] <= 1
 frac[9,29] <= 1
 frac[9,30] <= 1
 frac[9,31] <= 1
 frac[9,32] <= 1
 frac[9,33] <= 1
 frac[9,34] <= 1
 frac[9,35] <= 1
 frac[9,36] <= 1
 frac[9,37] <= 1
 frac[9,38] <= 1
 frac[9,39] <= 1
 frac[9,40] <= 1
 frac[9,41] <= 1
 frac[9,42] <= 1
 frac[9,43] <= 1
 frac[9,44] <= 1
 frac[9,45] <= 1
 frac[9,46] <= 1
 frac[9,47] <= 1
 frac[9,48] <= 1
 frac[9,49] <= 1
 frac[9,50] <= 1
 frac[9,51] <= 1
 frac[9,52] <= 1
 frac[9,53] <= 1
 frac[9,54] <= 1
 frac[9,55] <= 1
 frac[9,56] <= 1
 frac[9,57] <= 1
 frac[9,58] <= 1
 frac[9,59] <= 1
 frac[9,60] <= 1
 frac[9,61] <= 1
 frac[9,62] <= 1
 frac[9,63] <= 1
 frac[9,64] <= 1
 frac[9,65] <= 1
 frac[9,66] <= 1
 frac[9,67] <= 1
 frac[9,68] <= 1
 frac[9,69] <= 1
 frac[9,70] <= 1
 frac[9,71] <= 1
 frac[9,72] <= 1
 frac[9,73] <= 1
 frac[9,74] <= 1
 frac[9,75] <= 1
 frac[9,76] <= 1
 frac[9,77] <= 1
 frac[9,78] <= 1
 frac[9,79] <= 1
 frac[9,80] <= 1
 frac[9,81] <= 1
 frac[9,82] <= 1
 frac[9,83] <= 1
 frac[9,84] <= 1
 frac[9,85] <= 1
 frac[9,86] <= 1
 frac[9,87] <= 1
 frac[9,88] <= 1
 frac[9,89] <= 1
 frac[9,90] <= 1
 frac[9,91] <= 1
 frac[9,92] <= 1
 frac[9,93] <= 1
 frac[9,94] <= 1
 frac[9,95] <= 1
 frac[9,96] <= 1
 frac[9,97] <= 1
 frac[9,98] <= 1
 frac[9,99] <= 1
 frac[9,100] <= 1
 frac[9,101] <= 1
 frac[9,102] <= 1
 frac[9,103] <= 1
 frac[9,104] <= 1
 frac[9,105] <= 1
 frac[9,106] <= 1
 frac[9,107] <= 1
 frac[9,108] <= 1
 frac[9,109] <= 1
 frac[9,110] <= 1
 frac[9,111] <= 1
 frac[9,112] <= 1
 frac[9,113] <= 1
 frac[9,114] <= 1
 frac[9,115] <= 1
 frac[9,116] <= 1
 frac[9,117] <= 1
 frac[9,118] <= 1
 frac[9,119] <= 1
 frac[9,120] <= 1
 frac[9,121] <= 1
 frac[9,122] <= 1
 frac[9,123] <= 1
 frac[9,124] <= 1
 frac[9,125] <= 1
 frac[9,126] <= 1
 frac[9,127] <= 1
 frac[9,128] <= 1
 frac[9,129] <= 1
 frac[9,130] <= 1
 frac[9,131] <= 1
 frac[9,132] <= 1
 frac[9,133] <= 1
 frac[9,134] <= 1
 frac[9,135] <= 1
 frac[9,136] <= 1
 frac[9,137] <= 1
 frac[9,138] <= 1
 frac[9,139] <= 1
 frac[9,140] <= 1
 frac[9,141] <= 1
 frac[9,142] <= 1
 frac[9,143] <= 1
 frac[9,144] <= 1
 frac[9,145] <= 1
 frac[9,146] <= 1
 frac[9,147] <= 1
 frac[9,148] <= 1
 frac[9,149] <= 1
 frac[9,150] <= 1
 frac[9,151] <= 1
 frac[9,152] <= 1
 frac[9,153] <= 1
 frac[9,154] <= 1
 frac[9,155] <= 1
 frac[9,156] <= 1
 frac[9,157] <= 1
 frac[9,158] <= 1
 frac[9,159] <= 1
 frac[9,160] <= 1
 frac[9,161] <= 1
 frac[9,162] <= 1
 frac[9,163] <= 1
 frac[9,164] <= 1
 frac[9,165] <= 1
 frac[9,166] <= 1
 frac[9,167] <= 1
 frac[9,168] <= 1
 frac[9,169] <= 1
 frac[9,170] <= 1
 frac[9,171] <= 1
 frac[9,172] <= 1
 frac[9,173] <= 1
 frac[9,174] <= 1
 frac[9,175] <= 1
 frac[9,176] <= 1
 frac[9,177] <= 1
 frac[9,178] <= 1
 frac[9,179] <= 1
 frac[9,180] <= 1
 frac[9,181] <= 1
 frac[9,182] <= 1
 frac[9,183] <= 1
 frac[9,184] <= 1
 frac[9,185] <= 1
 frac[9,186] <= 1
 frac[9,187] <= 1
 frac[9,188] <= 1
 frac[9,189] <= 1
 frac[9,190] <= 1
 frac[9,191] <= 1
 frac[10,0] <= 1
 frac[10,1] <= 1
 frac[10,2] <= 1
 frac[10,3] <= 1
 frac[10,4] <= 1
 frac[10,5] <= 1
 frac[10,6] <= 1
 frac[10,7] <= 1
 frac[10,8] <= 1
 frac[10,9] <= 1
 frac[10,10] <= 1
 frac[10,11] <= 1
 frac[10,12] <= 1
 frac[10,13] <= 1
 frac[10,14] <= 1
 frac[10,15] <= 1
 frac[10,16] <= 1
 frac[10,17] <= 1
 frac[10,18] <= 1
 frac[10,19] <= 1
 frac[10,20] <= 1
 frac[10,21] <= 1
 frac[10,22] <= 1
 frac[10,23] <= 1
 frac[10,24] <= 1
 frac[10,25] <= 1
 frac[10,26] <= 1
 frac[10,27] <= 1
 frac[10,28] <= 1
 frac[10,29] <= 1
 frac[10,30] <= 1
 frac[10,31] <= 1
 frac[10,32] <= 1
 frac[10,33] <= 1
 frac[10,34] <= 1
 frac[10,35] <= 1
 frac[10,36] <= 1
 frac[10,37] <= 1
 frac[10,38] <= 1
 frac[10,39] <= 1
 frac[10,40] <= 1
 frac[10,41] <= 1
 frac[10,42] <= 1
 frac[10,43] <= 1
 frac[10,44] <= 1
 frac[10,45] <= 1
 frac[10,46] <= 1
 frac[10,47] <= 1
 frac[10,48] <= 1
 frac[10,49] <= 1
 frac[10,50] <= 1
 frac[10,51] <= 1
 frac[10,52] <= 1
 frac[10,53] <= 1
 frac[10,54] <= 1
 frac[10,55] <= 1
 frac[10,56] <= 1
 frac[10,57] <= 1
 frac[10,58] <= 1
 frac[10,59] <= 1
 frac[10,60] <= 1
 frac[10,61] <= 1
 frac[10,62] <= 1
 frac[10,63] <= 1
 frac[10,64] <= 1
 frac[10,65] <= 1
 frac[10,66] <= 1
 frac[10,67] <= 1
 frac[10,68] <= 1
 frac[10,69] <= 1
 frac[10,70] <= 1
 frac[10,71] <= 1
 frac[10,72] <= 1
 frac[10,73] <= 1
 frac[10,74] <= 1
 frac[10,75] <= 1
 frac[10,76] <= 1
 frac[10,77] <= 1
 frac[10,78] <= 1
 frac[10,79] <= 1
 frac[10,80] <= 1
 frac[10,81] <= 1
 frac[10,82] <= 1
 frac[10,83] <= 1
 frac[10,84] <= 1
 frac[10,85] <= 1
 frac[10,86] <= 1
 frac[10,87] <= 1
 frac[10,88] <= 1
 frac[10,89] <= 1
 frac[10,90] <= 1
 frac[10,91] <= 1
 frac[10,92] <= 1
 frac[10,93] <= 1
 frac[10,94] <= 1
 frac[10,95] <= 1
 frac[10,96] <= 1
 frac[10,97] <= 1
 frac[10,98] <= 1
 frac[10,99] <= 1
 frac[10,100] <= 1
 frac[10,101] <= 1
 frac[10,102] <= 1
 frac[10,103] <= 1
 frac[10,104] <= 1
 frac[10,105] <= 1
 frac[10,106] <= 1
 frac[10,107] <= 1
 frac[10,108] <= 1
 frac[10,109] <= 1
 frac[10,110] <= 1
 frac[10,111] <= 1
 frac[10,112] <= 1
 frac[10,113] <= 1
 frac[10,114] <= 1
 frac[10,115] <= 1
 frac[10,116] <= 1
 frac[10,117] <= 1
 frac[10,118] <= 1
 frac[10,119] <= 1
 frac[10,120] <= 1
 frac[10,121] <= 1
 frac[10,122] <= 1
 frac[10,123] <= 1
 frac[10,124] <= 1
 frac[10,125] <= 1
 frac[10,126] <= 1
 frac[10,127] <= 1
 frac[10,128] <= 1
 frac[10,129] <= 1
 frac[10,130] <= 1
 frac[10,131] <= 1
 frac[10,132] <= 1
 frac[10,133] <= 1
 frac[10,134] <= 1
 frac[10,135] <= 1
 frac[10,136] <= 1
 frac[10,137] <= 1
 frac[10,138] <= 1
 frac[10,139] <= 1
 frac[10,140] <= 1
 frac[10,141] <= 1
 frac[10,142] <= 1
 frac[10,143] <= 1
 frac[10,144] <= 1
 frac[10,145] <= 1
 frac[10,146] <= 1
 frac[10,147] <= 1
 frac[10,148] <= 1
 frac[10,149] <= 1
 frac[10,150] <= 1
 frac[10,151] <= 1
 frac[10,152] <= 1
 frac[10,153] <= 1
 frac[10,154] <= 1
 frac[10,155] <= 1
 frac[10,156] <= 1
 frac[10,157] <= 1
 frac[10,158] <= 1
 frac[10,159] <= 1
 frac[10,160] <= 1
 frac[10,161] <= 1
 frac[10,162] <= 1
 frac[10,163] <= 1
 frac[10,164] <= 1
 frac[10,165] <= 1
 frac[10,166] <= 1
 frac[10,167] <= 1
 frac[10,168] <= 1
 frac[10,169] <= 1
 frac[10,170] <= 1
 frac[10,171] <= 1
 frac[10,172] <= 1
 frac[10,173] <= 1
 frac[10,174] <= 1
 frac[10,175] <= 1
 frac[10,176] <= 1
 frac[10,177] <= 1
 frac[10,178] <= 1
 frac[10,179] <= 1
 frac[10,180] <= 1
 frac[10,181] <= 1
 frac[10,182] <= 1
 frac[10,183] <= 1
 frac[10,184] <= 1
 frac[10,185] <= 1
 frac[10,186] <= 1
 frac[10,187] <= 1
 frac[10,188] <= 1
 frac[10,189] <= 1
 frac[10,190] <= 1
 frac[10,191] <= 1
 frac[11,0] <= 1
 frac[11,1] <= 1
 frac[11,2] <= 1
 frac[11,3] <= 1
 frac[11,4] <= 1
 frac[11,5] <= 1
 frac[11,6] <= 1
 frac[11,7] <= 1
 frac[11,8] <= 1
 frac[11,9] <= 1
 frac[11,10] <= 1
 frac[11,11] <= 1
 frac[11,12] <= 1
 frac[11,13] <= 1
 frac[11,14] <= 1
 frac[11,15] <= 1
 frac[11,16] <= 1
 frac[11,17] <= 1
 frac[11,18] <= 1
 frac[11,19] <= 1
 frac[11,20] <= 1
 frac[11,21] <= 1
 frac[11,22] <= 1
 frac[11,23] <= 1
 frac[11,24] <= 1
 frac[11,25] <= 1
 frac[11,26] <= 1
 frac[11,27] <= 1
 frac[11,28] <= 1
 frac[11,29] <= 1
 frac[11,30] <= 1
 frac[11,31] <= 1
 frac[11,32] <= 1
 frac[11,33] <= 1
 frac[11,34] <= 1
 frac[11,35] <= 1
 frac[11,36] <= 1
 frac[11,37] <= 1
 frac[11,38] <= 1
 frac[11,39] <= 1
 frac[11,40] <= 1
 frac[11,41] <= 1
 frac[11,42] <= 1
 frac[11,43] <= 1
 frac[11,44] <= 1
 frac[11,45] <= 1
 frac[11,46] <= 1
 frac[11,47] <= 1
 frac[11,48] <= 1
 frac[11,49] <= 1
 frac[11,50] <= 1
 frac[11,51] <= 1
 frac[11,52] <= 1
 frac[11,53] <= 1
 frac[11,54] <= 1
 frac[11,55] <= 1
 frac[11,56] <= 1
 frac[11,57] <= 1
 frac[11,58] <= 1
 frac[11,59] <= 1
 frac[11,60] <= 1
 frac[11,61] <= 1
 frac[11,62] <= 1
 frac[11,63] <= 1
 frac[11,64] <= 1
 frac[11,65] <= 1
 frac[11,66] <= 1
 frac[11,67] <= 1
 frac[11,68] <= 1
 frac[11,69] <= 1
 frac[11,70] <= 1
 frac[11,71] <= 1
 frac[11,72] <= 1
 frac[11,73] <= 1
 frac[11,74] <= 1
 frac[11,75] <= 1
 frac[11,76] <= 1
 frac[11,77] <= 1
 frac[11,78] <= 1
 frac[11,79] <= 1
 frac[11,80] <= 1
 frac[11,81] <= 1
 frac[11,82] <= 1
 frac[11,83] <= 1
 frac[11,84] <= 1
 frac[11,85] <= 1
 frac[11,86] <= 1
 frac[11,87] <= 1
 frac[11,88] <= 1
 frac[11,89] <= 1
 frac[11,90] <= 1
 frac[11,91] <= 1
 frac[11,92] <= 1
 frac[11,93] <= 1
 frac[11,94] <= 1
 frac[11,95] <= 1
 frac[11,96] <= 1
 frac[11,97] <= 1
 frac[11,98] <= 1
 frac[11,99] <= 1
 frac[11,100] <= 1
 frac[11,101] <= 1
 frac[11,102] <= 1
 frac[11,103] <= 1
 frac[11,104] <= 1
 frac[11,105] <= 1
 frac[11,106] <= 1
 frac[11,107] <= 1
 frac[11,108] <= 1
 frac[11,109] <= 1
 frac[11,110] <= 1
 frac[11,111] <= 1
 frac[11,112] <= 1
 frac[11,113] <= 1
 frac[11,114] <= 1
 frac[11,115] <= 1
 frac[11,116] <= 1
 frac[11,117] <= 1
 frac[11,118] <= 1
 frac[11,119] <= 1
 frac[11,120] <= 1
 frac[11,121] <= 1
 frac[11,122] <= 1
 frac[11,123] <= 1
 frac[11,124] <= 1
 frac[11,125] <= 1
 frac[11,126] <= 1
 frac[11,127] <= 1
 frac[11,128] <= 1
 frac[11,129] <= 1
 frac[11,130] <= 1
 frac[11,131] <= 1
 frac[11,132] <= 1
 frac[11,133] <= 1
 frac[11,134] <= 1
 frac[11,135] <= 1
 frac[11,136] <= 1
 frac[11,137] <= 1
 frac[11,138] <= 1
 frac[11,139] <= 1
 frac[11,140] <= 1
 frac[11,141] <= 1
 frac[11,142] <= 1
 frac[11,143] <= 1
 frac[11,144] <= 1
 frac[11,145] <= 1
 frac[11,146] <= 1
 frac[11,147] <= 1
 frac[11,148] <= 1
 frac[11,149] <= 1
 frac[11,150] <= 1
 frac[11,151] <= 1
 frac[11,152] <= 1
 frac[11,153] <= 1
 frac[11,154] <= 1
 frac[11,155] <= 1
 frac[11,156] <= 1
 frac[11,157] <= 1
 frac[11,158] <= 1
 frac[11,159] <= 1
 frac[11,160] <= 1
 frac[11,161] <= 1
 frac[11,162] <= 1
 frac[11,163] <= 1
 frac[11,164] <= 1
 frac[11,165] <= 1
 frac[11,166] <= 1
 frac[11,167] <= 1
 frac[11,168] <= 1
 frac[11,169] <= 1
 frac[11,170] <= 1
 frac[11,171] <= 1
 frac[11,172] <= 1
 frac[11,173] <= 1
 frac[11,174] <= 1
 frac[11,175] <= 1
 frac[11,176] <= 1
 frac[11,177] <= 1
 frac[11,178] <= 1
 frac[11,179] <= 1
 frac[11,180] <= 1
 frac[11,181] <= 1
 frac[11,182] <= 1
 frac[11,183] <= 1
 frac[11,184] <= 1
 frac[11,185] <= 1
 frac[11,186] <= 1
 frac[11,187] <= 1
 frac[11,188] <= 1
 frac[11,189] <= 1
 frac[11,190] <= 1
 frac[11,191] <= 1
 frac[12,0] <= 1
 frac[12,1] <= 1
 frac[12,2] <= 1
 frac[12,3] <= 1
 frac[12,4] <= 1
 frac[12,5] <= 1
 frac[12,6] <= 1
 frac[12,7] <= 1
 frac[12,8] <= 1
 frac[12,9] <= 1
 frac[12,10] <= 1
 frac[12,11] <= 1
 frac[12,12] <= 1
 frac[12,13] <= 1
 frac[12,14] <= 1
 frac[12,15] <= 1
 frac[12,16] <= 1
 frac[12,17] <= 1
 frac[12,18] <= 1
 frac[12,19] <= 1
 frac[12,20] <= 1
 frac[12,21] <= 1
 frac[12,22] <= 1
 frac[12,23] <= 1
 frac[12,24] <= 1
 frac[12,25] <= 1
 frac[12,26] <= 1
 frac[12,27] <= 1
 frac[12,28] <= 1
 frac[12,29] <= 1
 frac[12,30] <= 1
 frac[12,31] <= 1
 frac[12,32] <= 1
 frac[12,33] <= 1
 frac[12,34] <= 1
 frac[12,35] <= 1
 frac[12,36] <= 1
 frac[12,37] <= 1
 frac[12,38] <= 1
 frac[12,39] <= 1
 frac[12,40] <= 1
 frac[12,41] <= 1
 frac[12,42] <= 1
 frac[12,43] <= 1
 frac[12,44] <= 1
 frac[12,45] <= 1
 frac[12,46] <= 1
 frac[12,47] <= 1
 frac[12,48] <= 1
 frac[12,49] <= 1
 frac[12,50] <= 1
 frac[12,51] <= 1
 frac[12,52] <= 1
 frac[12,53] <= 1
 frac[12,54] <= 1
 frac[12,55] <= 1
 frac[12,56] <= 1
 frac[12,57] <= 1
 frac[12,58] <= 1
 frac[12,59] <= 1
 frac[12,60] <= 1
 frac[12,61] <= 1
 frac[12,62] <= 1
 frac[12,63] <= 1
 frac[12,64] <= 1
 frac[12,65] <= 1
 frac[12,66] <= 1
 frac[12,67] <= 1
 frac[12,68] <= 1
 frac[12,69] <= 1
 frac[12,70] <= 1
 frac[12,71] <= 1
 frac[12,72] <= 1
 frac[12,73] <= 1
 frac[12,74] <= 1
 frac[12,75] <= 1
 frac[12,76] <= 1
 frac[12,77] <= 1
 frac[12,78] <= 1
 frac[12,79] <= 1
 frac[12,80] <= 1
 frac[12,81] <= 1
 frac[12,82] <= 1
 frac[12,83] <= 1
 frac[12,84] <= 1
 frac[12,85] <= 1
 frac[12,86] <= 1
 frac[12,87] <= 1
 frac[12,88] <= 1
 frac[12,89] <= 1
 frac[12,90] <= 1
 frac[12,91] <= 1
 frac[12,92] <= 1
 frac[12,93] <= 1
 frac[12,94] <= 1
 frac[12,95] <= 1
 frac[12,96] <= 1
 frac[12,97] <= 1
 frac[12,98] <= 1
 frac[12,99] <= 1
 frac[12,100] <= 1
 frac[12,101] <= 1
 frac[12,102] <= 1
 frac[12,103] <= 1
 frac[12,104] <= 1
 frac[12,105] <= 1
 frac[12,106] <= 1
 frac[12,107] <= 1
 frac[12,108] <= 1
 frac[12,109] <= 1
 frac[12,110] <= 1
 frac[12,111] <= 1
 frac[12,112] <= 1
 frac[12,113] <= 1
 frac[12,114] <= 1
 frac[12,115] <= 1
 frac[12,116] <= 1
 frac[12,117] <= 1
 frac[12,118] <= 1
 frac[12,119] <= 1
 frac[12,120] <= 1
 frac[12,121] <= 1
 frac[12,122] <= 1
 frac[12,123] <= 1
 frac[12,124] <= 1
 frac[12,125] <= 1
 frac[12,126] <= 1
 frac[12,127] <= 1
 frac[12,128] <= 1
 frac[12,129] <= 1
 frac[12,130] <= 1
 frac[12,131] <= 1
 frac[12,132] <= 1
 frac[12,133] <= 1
 frac[12,134] <= 1
 frac[12,135] <= 1
 frac[12,136] <= 1
 frac[12,137] <= 1
 frac[12,138] <= 1
 frac[12,139] <= 1
 frac[12,140] <= 1
 frac[12,141] <= 1
 frac[12,142] <= 1
 frac[12,143] <= 1
 frac[12,144] <= 1
 frac[12,145] <= 1
 frac[12,146] <= 1
 frac[12,147] <= 1
 frac[12,148] <= 1
 frac[12,149] <= 1
 frac[12,150] <= 1
 frac[12,151] <= 1
 frac[12,152] <= 1
 frac[12,153] <= 1
 frac[12,154] <= 1
 frac[12,155] <= 1
 frac[12,156] <= 1
 frac[12,157] <= 1
 frac[12,158] <= 1
 frac[12,159] <= 1
 frac[12,160] <= 1
 frac[12,161] <= 1
 frac[12,162] <= 1
 frac[12,163] <= 1
 frac[12,164] <= 1
 frac[12,165] <= 1
 frac[12,166] <= 1
 frac[12,167] <= 1
 frac[12,168] <= 1
 frac[12,169] <= 1
 frac[12,170] <= 1
 frac[12,171] <= 1
 frac[12,172] <= 1
 frac[12,173] <= 1
 frac[12,174] <= 1
 frac[12,175] <= 1
 frac[12,176] <= 1
 frac[12,177] <= 1
 frac[12,178] <= 1
 frac[12,179] <= 1
 frac[12,180] <= 1
 frac[12,181] <= 1
 frac[12,182] <= 1
 frac[12,183] <= 1
 frac[12,184] <= 1
 frac[12,185] <= 1
 frac[12,186] <= 1
 frac[12,187] <= 1
 frac[12,188] <= 1
 frac[12,189] <= 1
 frac[12,190] <= 1
 frac[12,191] <= 1
 frac[13,0] <= 1
 frac[13,1] <= 1
 frac[13,2] <= 1
 frac[13,3] <= 1
 frac[13,4] <= 1
 frac[13,5] <= 1
 frac[13,6] <= 1
 frac[13,7] <= 1
 frac[13,8] <= 1
 frac[13,9] <= 1
 frac[13,10] <= 1
 frac[13,11] <= 1
 frac[13,12] <= 1
 frac[13,13] <= 1
 frac[13,14] <= 1
 frac[13,15] <= 1
 frac[13,16] <= 1
 frac[13,17] <= 1
 frac[13,18] <= 1
 frac[13,19] <= 1
 frac[13,20] <= 1
 frac[13,21] <= 1
 frac[13,22] <= 1
 frac[13,23] <= 1
 frac[13,24] <= 1
 frac[13,25] <= 1
 frac[13,26] <= 1
 frac[13,27] <= 1
 frac[13,28] <= 1
 frac[13,29] <= 1
 frac[13,30] <= 1
 frac[13,31] <= 1
 frac[13,32] <= 1
 frac[13,33] <= 1
 frac[13,34] <= 1
 frac[13,35] <= 1
 frac[13,36] <= 1
 frac[13,37] <= 1
 frac[13,38] <= 1
 frac[13,39] <= 1
 frac[13,40] <= 1
 frac[13,41] <= 1
 frac[13,42] <= 1
 frac[13,43] <= 1
 frac[13,44] <= 1
 frac[13,45] <= 1
 frac[13,46] <= 1
 frac[13,47] <= 1
 frac[13,48] <= 1
 frac[13,49] <= 1
 frac[13,50] <= 1
 frac[13,51] <= 1
 frac[13,52] <= 1
 frac[13,53] <= 1
 frac[13,54] <= 1
 frac[13,55] <= 1
 frac[13,56] <= 1
 frac[13,57] <= 1
 frac[13,58] <= 1
 frac[13,59] <= 1
 frac[13,60] <= 1
 frac[13,61] <= 1
 frac[13,62] <= 1
 frac[13,63] <= 1
 frac[13,64] <= 1
 frac[13,65] <= 1
 frac[13,66] <= 1
 frac[13,67] <= 1
 frac[13,68] <= 1
 frac[13,69] <= 1
 frac[13,70] <= 1
 frac[13,71] <= 1
 frac[13,72] <= 1
 frac[13,73] <= 1
 frac[13,74] <= 1
 frac[13,75] <= 1
 frac[13,76] <= 1
 frac[13,77] <= 1
 frac[13,78] <= 1
 frac[13,79] <= 1
 frac[13,80] <= 1
 frac[13,81] <= 1
 frac[13,82] <= 1
 frac[13,83] <= 1
 frac[13,84] <= 1
 frac[13,85] <= 1
 frac[13,86] <= 1
 frac[13,87] <= 1
 frac[13,88] <= 1
 frac[13,89] <= 1
 frac[13,90] <= 1
 frac[13,91] <= 1
 frac[13,92] <= 1
 frac[13,93] <= 1
 frac[13,94] <= 1
 frac[13,95] <= 1
 frac[13,96] <= 1
 frac[13,97] <= 1
 frac[13,98] <= 1
 frac[13,99] <= 1
 frac[13,100] <= 1
 frac[13,101] <= 1
 frac[13,102] <= 1
 frac[13,103] <= 1
 frac[13,104] <= 1
 frac[13,105] <= 1
 frac[13,106] <= 1
 frac[13,107] <= 1
 frac[13,108] <= 1
 frac[13,109] <= 1
 frac[13,110] <= 1
 frac[13,111] <= 1
 frac[13,112] <= 1
 frac[13,113] <= 1
 frac[13,114] <= 1
 frac[13,115] <= 1
 frac[13,116] <= 1
 frac[13,117] <= 1
 frac[13,118] <= 1
 frac[13,119] <= 1
 frac[13,120] <= 1
 frac[13,121] <= 1
 frac[13,122] <= 1
 frac[13,123] <= 1
 frac[13,124] <= 1
 frac[13,125] <= 1
 frac[13,126] <= 1
 frac[13,127] <= 1
 frac[13,128] <= 1
 frac[13,129] <= 1
 frac[13,130] <= 1
 frac[13,131] <= 1
 frac[13,132] <= 1
 frac[13,133] <= 1
 frac[13,134] <= 1
 frac[13,135] <= 1
 frac[13,136] <= 1
 frac[13,137] <= 1
 frac[13,138] <= 1
 frac[13,139] <= 1
 frac[13,140] <= 1
 frac[13,141] <= 1
 frac[13,142] <= 1
 frac[13,143] <= 1
 frac[13,144] <= 1
 frac[13,145] <= 1
 frac[13,146] <= 1
 frac[13,147] <= 1
 frac[13,148] <= 1
 frac[13,149] <= 1
 frac[13,150] <= 1
 frac[13,151] <= 1
 frac[13,152] <= 1
 frac[13,153] <= 1
 frac[13,154] <= 1
 frac[13,155] <= 1
 frac[13,156] <= 1
 frac[13,157] <= 1
 frac[13,158] <= 1
 frac[13,159] <= 1
 frac[13,160] <= 1
 frac[13,161] <= 1
 frac[13,162] <= 1
 frac[13,163] <= 1
 frac[13,164] <= 1
 frac[13,165] <= 1
 frac[13,166] <= 1
 frac[13,167] <= 1
 frac[13,168] <= 1
 frac[13,169] <= 1
 frac[13,170] <= 1
 frac[13,171] <= 1
 frac[13,172] <= 1
 frac[13,173] <= 1
 frac[13,174] <= 1
 frac[13,175] <= 1
 frac[13,176] <= 1
 frac[13,177] <= 1
 frac[13,178] <= 1
 frac[13,179] <= 1
 frac[13,180] <= 1
 frac[13,181] <= 1
 frac[13,182] <= 1
 frac[13,183] <= 1
 frac[13,184] <= 1
 frac[13,185] <= 1
 frac[13,186] <= 1
 frac[13,187] <= 1
 frac[13,188] <= 1
 frac[13,189] <= 1
 frac[13,190] <= 1
 frac[13,191] <= 1
 frac[14,0] <= 1
 frac[14,1] <= 1
 frac[14,2] <= 1
 frac[14,3] <= 1
 frac[14,4] <= 1
 frac[14,5] <= 1
 frac[14,6] <= 1
 frac[14,7] <= 1
 frac[14,8] <= 1
 frac[14,9] <= 1
 frac[14,10] <= 1
 frac[14,11] <= 1
 frac[14,12] <= 1
 frac[14,13] <= 1
 frac[14,14] <= 1
 frac[14,15] <= 1
 frac[14,16] <= 1
 frac[14,17] <= 1
 frac[14,18] <= 1
 frac[14,19] <= 1
 frac[14,20] <= 1
 frac[14,21] <= 1
 frac[14,22] <= 1
 frac[14,23] <= 1
 frac[14,24] <= 1
 frac[14,25] <= 1
 frac[14,26] <= 1
 frac[14,27] <= 1
 frac[14,28] <= 1
 frac[14,29] <= 1
 frac[14,30] <= 1
 frac[14,31] <= 1
 frac[14,32] <= 1
 frac[14,33] <= 1
 frac[14,34] <= 1
 frac[14,35] <= 1
 frac[14,36] <= 1
 frac[14,37] <= 1
 frac[14,38] <= 1
 frac[14,39] <= 1
 frac[14,40] <= 1
 frac[14,41] <= 1
 frac[14,42] <= 1
 frac[14,43] <= 1
 frac[14,44] <= 1
 frac[14,45] <= 1
 frac[14,46] <= 1
 frac[14,47] <= 1
 frac[14,48] <= 1
 frac[14,49] <= 1
 frac[14,50] <= 1
 frac[14,51] <= 1
 frac[14,52] <= 1
 frac[14,53] <= 1
 frac[14,54] <= 1
 frac[14,55] <= 1
 frac[14,56] <= 1
 frac[14,57] <= 1
 frac[14,58] <= 1
 frac[14,59] <= 1
 frac[14,60] <= 1
 frac[14,61] <= 1
 frac[14,62] <= 1
 frac[14,63] <= 1
 frac[14,64] <= 1
 frac[14,65] <= 1
 frac[14,66] <= 1
 frac[14,67] <= 1
 frac[14,68] <= 1
 frac[14,69] <= 1
 frac[14,70] <= 1
 frac[14,71] <= 1
 frac[14,72] <= 1
 frac[14,73] <= 1
 frac[14,74] <= 1
 frac[14,75] <= 1
 frac[14,76] <= 1
 frac[14,77] <= 1
 frac[14,78] <= 1
 frac[14,79] <= 1
 frac[14,80] <= 1
 frac[14,81] <= 1
 frac[14,82] <= 1
 frac[14,83] <= 1
 frac[14,84] <= 1
 frac[14,85] <= 1
 frac[14,86] <= 1
 frac[14,87] <= 1
 frac[14,88] <= 1
 frac[14,89] <= 1
 frac[14,90] <= 1
 frac[14,91] <= 1
 frac[14,92] <= 1
 frac[14,93] <= 1
 frac[14,94] <= 1
 frac[14,95] <= 1
 frac[14,96] <= 1
 frac[14,97] <= 1
 frac[14,98] <= 1
 frac[14,99] <= 1
 frac[14,100] <= 1
 frac[14,101] <= 1
 frac[14,102] <= 1
 frac[14,103] <= 1
 frac[14,104] <= 1
 frac[14,105] <= 1
 frac[14,106] <= 1
 frac[14,107] <= 1
 frac[14,108] <= 1
 frac[14,109] <= 1
 frac[14,110] <= 1
 frac[14,111] <= 1
 frac[14,112] <= 1
 frac[14,113] <= 1
 frac[14,114] <= 1
 frac[14,115] <= 1
 frac[14,116] <= 1
 frac[14,117] <= 1
 frac[14,118] <= 1
 frac[14,119] <= 1
 frac[14,120] <= 1
 frac[14,121] <= 1
 frac[14,122] <= 1
 frac[14,123] <= 1
 frac[14,124] <= 1
 frac[14,125] <= 1
 frac[14,126] <= 1
 frac[14,127] <= 1
 frac[14,128] <= 1
 frac[14,129] <= 1
 frac[14,130] <= 1
 frac[14,131] <= 1
 frac[14,132] <= 1
 frac[14,133] <= 1
 frac[14,134] <= 1
 frac[14,135] <= 1
 frac[14,136] <= 1
 frac[14,137] <= 1
 frac[14,138] <= 1
 frac[14,139] <= 1
 frac[14,140] <= 1
 frac[14,141] <= 1
 frac[14,142] <= 1
 frac[14,143] <= 1
 frac[14,144] <= 1
 frac[14,145] <= 1
 frac[14,146] <= 1
 frac[14,147] <= 1
 frac[14,148] <= 1
 frac[14,149] <= 1
 frac[14,150] <= 1
 frac[14,151] <= 1
 frac[14,152] <= 1
 frac[14,153] <= 1
 frac[14,154] <= 1
 frac[14,155] <= 1
 frac[14,156] <= 1
 frac[14,157] <= 1
 frac[14,158] <= 1
 frac[14,159] <= 1
 frac[14,160] <= 1
 frac[14,161] <= 1
 frac[14,162] <= 1
 frac[14,163] <= 1
 frac[14,164] <= 1
 frac[14,165] <= 1
 frac[14,166] <= 1
 frac[14,167] <= 1
 frac[14,168] <= 1
 frac[14,169] <= 1
 frac[14,170] <= 1
 frac[14,171] <= 1
 frac[14,172] <= 1
 frac[14,173] <= 1
 frac[14,174] <= 1
 frac[14,175] <= 1
 frac[14,176] <= 1
 frac[14,177] <= 1
 frac[14,178] <= 1
 frac[14,179] <= 1
 frac[14,180] <= 1
 frac[14,181] <= 1
 frac[14,182] <= 1
 frac[14,183] <= 1
 frac[14,184] <= 1
 frac[14,185] <= 1
 frac[14,186] <= 1
 frac[14,187] <= 1
 frac[14,188] <= 1
 frac[14,189] <= 1
 frac[14,190] <= 1
 frac[14,191] <= 1
 frac[15,0] <= 1
 frac[15,1] <= 1
 frac[15,2] <= 1
 frac[15,3] <= 1
 frac[15,4] <= 1
 frac[15,5] <= 1
 frac[15,6] <= 1
 frac[15,7] <= 1
 frac[15,8] <= 1
 frac[15,9] <= 1
 frac[15,10] <= 1
 frac[15,11] <= 1
 frac[15,12] <= 1
 frac[15,13] <= 1
 frac[15,14] <= 1
 frac[15,15] <= 1
 frac[15,16] <= 1
 frac[15,17] <= 1
 frac[15,18] <= 1
 frac[15,19] <= 1
 frac[15,20] <= 1
 frac[15,21] <= 1
 frac[15,22] <= 1
 frac[15,23] <= 1
 frac[15,24] <= 1
 frac[15,25] <= 1
 frac[15,26] <= 1
 frac[15,27] <= 1
 frac[15,28] <= 1
 frac[15,29] <= 1
 frac[15,30] <= 1
 frac[15,31] <= 1
 frac[15,32] <= 1
 frac[15,33] <= 1
 frac[15,34] <= 1
 frac[15,35] <= 1
 frac[15,36] <= 1
 frac[15,37] <= 1
 frac[15,38] <= 1
 frac[15,39] <= 1
 frac[15,40] <= 1
 frac[15,41] <= 1
 frac[15,42] <= 1
 frac[15,43] <= 1
 frac[15,44] <= 1
 frac[15,45] <= 1
 frac[15,46] <= 1
 frac[15,47] <= 1
 frac[15,48] <= 1
 frac[15,49] <= 1
 frac[15,50] <= 1
 frac[15,51] <= 1
 frac[15,52] <= 1
 frac[15,53] <= 1
 frac[15,54] <= 1
 frac[15,55] <= 1
 frac[15,56] <= 1
 frac[15,57] <= 1
 frac[15,58] <= 1
 frac[15,59] <= 1
 frac[15,60] <= 1
 frac[15,61] <= 1
 frac[15,62] <= 1
 frac[15,63] <= 1
 frac[15,64] <= 1
 frac[15,65] <= 1
 frac[15,66] <= 1
 frac[15,67] <= 1
 frac[15,68] <= 1
 frac[15,69] <= 1
 frac[15,70] <= 1
 frac[15,71] <= 1
 frac[15,72] <= 1
 frac[15,73] <= 1
 frac[15,74] <= 1
 frac[15,75] <= 1
 frac[15,76] <= 1
 frac[15,77] <= 1
 frac[15,78] <= 1
 frac[15,79] <= 1
 frac[15,80] <= 1
 frac[15,81] <= 1
 frac[15,82] <= 1
 frac[15,83] <= 1
 frac[15,84] <= 1
 frac[15,85] <= 1
 frac[15,86] <= 1
 frac[15,87] <= 1
 frac[15,88] <= 1
 frac[15,89] <= 1
 frac[15,90] <= 1
 frac[15,91] <= 1
 frac[15,92] <= 1
 frac[15,93] <= 1
 frac[15,94] <= 1
 frac[15,95] <= 1
 frac[15,96] <= 1
 frac[15,97] <= 1
 frac[15,98] <= 1
 frac[15,99] <= 1
 frac[15,100] <= 1
 frac[15,101] <= 1
 frac[15,102] <= 1
 frac[15,103] <= 1
 frac[15,104] <= 1
 frac[15,105] <= 1
 frac[15,106] <= 1
 frac[15,107] <= 1
 frac[15,108] <= 1
 frac[15,109] <= 1
 frac[15,110] <= 1
 frac[15,111] <= 1
 frac[15,112] <= 1
 frac[15,113] <= 1
 frac[15,114] <= 1
 frac[15,115] <= 1
 frac[15,116] <= 1
 frac[15,117] <= 1
 frac[15,118] <= 1
 frac[15,119] <= 1
 frac[15,120] <= 1
 frac[15,121] <= 1
 frac[15,122] <= 1
 frac[15,123] <= 1
 frac[15,124] <= 1
 frac[15,125] <= 1
 frac[15,126] <= 1
 frac[15,127] <= 1
 frac[15,128] <= 1
 frac[15,129] <= 1
 frac[15,130] <= 1
 frac[15,131] <= 1
 frac[15,132] <= 1
 frac[15,133] <= 1
 frac[15,134] <= 1
 frac[15,135] <= 1
 frac[15,136] <= 1
 frac[15,137] <= 1
 frac[15,138] <= 1
 frac[15,139] <= 1
 frac[15,140] <= 1
 frac[15,141] <= 1
 frac[15,142] <= 1
 frac[15,143] <= 1
 frac[15,144] <= 1
 frac[15,145] <= 1
 frac[15,146] <= 1
 frac[15,147] <= 1
 frac[15,148] <= 1
 frac[15,149] <= 1
 frac[15,150] <= 1
 frac[15,151] <= 1
 frac[15,152] <= 1
 frac[15,153] <= 1
 frac[15,154] <= 1
 frac[15,155] <= 1
 frac[15,156] <= 1
 frac[15,157] <= 1
 frac[15,158] <= 1
 frac[15,159] <= 1
 frac[15,160] <= 1
 frac[15,161] <= 1
 frac[15,162] <= 1
 frac[15,163] <= 1
 frac[15,164] <= 1
 frac[15,165] <= 1
 frac[15,166] <= 1
 frac[15,167] <= 1
 frac[15,168] <= 1
 frac[15,169] <= 1
 frac[15,170] <= 1
 frac[15,171] <= 1
 frac[15,172] <= 1
 frac[15,173] <= 1
 frac[15,174] <= 1
 frac[15,175] <= 1
 frac[15,176] <= 1
 frac[15,177] <= 1
 frac[15,178] <= 1
 frac[15,179] <= 1
 frac[15,180] <= 1
 frac[15,181] <= 1
 frac[15,182] <= 1
 frac[15,183] <= 1
 frac[15,184] <= 1
 frac[15,185] <= 1
 frac[15,186] <= 1
 frac[15,187] <= 1
 frac[15,188] <= 1
 frac[15,189] <= 1
 frac[15,190] <= 1
 frac[15,191] <= 1
 frac[16,0] <= 1
 frac[16,1] <= 1
 frac[16,2] <= 1
 frac[16,3] <= 1
 frac[16,4] <= 1
 frac[16,5] <= 1
 frac[16,6] <= 1
 frac[16,7] <= 1
 frac[16,8] <= 1
 frac[16,9] <= 1
 frac[16,10] <= 1
 frac[16,11] <= 1
 frac[16,12] <= 1
 frac[16,13] <= 1
 frac[16,14] <= 1
 frac[16,15] <= 1
 frac[16,16] <= 1
 frac[16,17] <= 1
 frac[16,18] <= 1
 frac[16,19] <= 1
 frac[16,20] <= 1
 frac[16,21] <= 1
 frac[16,22] <= 1
 frac[16,23] <= 1
 frac[16,24] <= 1
 frac[16,25] <= 1
 frac[16,26] <= 1
 frac[16,27] <= 1
 frac[16,28] <= 1
 frac[16,29] <= 1
 frac[16,30] <= 1
 frac[16,31] <= 1
 frac[16,32] <= 1
 frac[16,33] <= 1
 frac[16,34] <= 1
 frac[16,35] <= 1
 frac[16,36] <= 1
 frac[16,37] <= 1
 frac[16,38] <= 1
 frac[16,39] <= 1
 frac[16,40] <= 1
 frac[16,41] <= 1
 frac[16,42] <= 1
 frac[16,43] <= 1
 frac[16,44] <= 1
 frac[16,45] <= 1
 frac[16,46] <= 1
 frac[16,47] <= 1
 frac[16,48] <= 1
 frac[16,49] <= 1
 frac[16,50] <= 1
 frac[16,51] <= 1
 frac[16,52] <= 1
 frac[16,53] <= 1
 frac[16,54] <= 1
 frac[16,55] <= 1
 frac[16,56] <= 1
 frac[16,57] <= 1
 frac[16,58] <= 1
 frac[16,59] <= 1
 frac[16,60] <= 1
 frac[16,61] <= 1
 frac[16,62] <= 1
 frac[16,63] <= 1
 frac[16,64] <= 1
 frac[16,65] <= 1
 frac[16,66] <= 1
 frac[16,67] <= 1
 frac[16,68] <= 1
 frac[16,69] <= 1
 frac[16,70] <= 1
 frac[16,71] <= 1
 frac[16,72] <= 1
 frac[16,73] <= 1
 frac[16,74] <= 1
 frac[16,75] <= 1
 frac[16,76] <= 1
 frac[16,77] <= 1
 frac[16,78] <= 1
 frac[16,79] <= 1
 frac[16,80] <= 1
 frac[16,81] <= 1
 frac[16,82] <= 1
 frac[16,83] <= 1
 frac[16,84] <= 1
 frac[16,85] <= 1
 frac[16,86] <= 1
 frac[16,87] <= 1
 frac[16,88] <= 1
 frac[16,89] <= 1
 frac[16,90] <= 1
 frac[16,91] <= 1
 frac[16,92] <= 1
 frac[16,93] <= 1
 frac[16,94] <= 1
 frac[16,95] <= 1
 frac[16,96] <= 1
 frac[16,97] <= 1
 frac[16,98] <= 1
 frac[16,99] <= 1
 frac[16,100] <= 1
 frac[16,101] <= 1
 frac[16,102] <= 1
 frac[16,103] <= 1
 frac[16,104] <= 1
 frac[16,105] <= 1
 frac[16,106] <= 1
 frac[16,107] <= 1
 frac[16,108] <= 1
 frac[16,109] <= 1
 frac[16,110] <= 1
 frac[16,111] <= 1
 frac[16,112] <= 1
 frac[16,113] <= 1
 frac[16,114] <= 1
 frac[16,115] <= 1
 frac[16,116] <= 1
 frac[16,117] <= 1
 frac[16,118] <= 1
 frac[16,119] <= 1
 frac[16,120] <= 1
 frac[16,121] <= 1
 frac[16,122] <= 1
 frac[16,123] <= 1
 frac[16,124] <= 1
 frac[16,125] <= 1
 frac[16,126] <= 1
 frac[16,127] <= 1
 frac[16,128] <= 1
 frac[16,129] <= 1
 frac[16,130] <= 1
 frac[16,131] <= 1
 frac[16,132] <= 1
 frac[16,133] <= 1
 frac[16,134] <= 1
 frac[16,135] <= 1
 frac[16,136] <= 1
 frac[16,137] <= 1
 frac[16,138] <= 1
 frac[16,139] <= 1
 frac[16,140] <= 1
 frac[16,141] <= 1
 frac[16,142] <= 1
 frac[16,143] <= 1
 frac[16,144] <= 1
 frac[16,145] <= 1
 frac[16,146] <= 1
 frac[16,147] <= 1
 frac[16,148] <= 1
 frac[16,149] <= 1
 frac[16,150] <= 1
 frac[16,151] <= 1
 frac[16,152] <= 1
 frac[16,153] <= 1
 frac[16,154] <= 1
 frac[16,155] <= 1
 frac[16,156] <= 1
 frac[16,157] <= 1
 frac[16,158] <= 1
 frac[16,159] <= 1
 frac[16,160] <= 1
 frac[16,161] <= 1
 frac[16,162] <= 1
 frac[16,163] <= 1
 frac[16,164] <= 1
 frac[16,165] <= 1
 frac[16,166] <= 1
 frac[16,167] <= 1
 frac[16,168] <= 1
 frac[16,169] <= 1
 frac[16,170] <= 1
 frac[16,171] <= 1
 frac[16,172] <= 1
 frac[16,173] <= 1
 frac[16,174] <= 1
 frac[16,175] <= 1
 frac[16,176] <= 1
 frac[16,177] <= 1
 frac[16,178] <= 1
 frac[16,179] <= 1
 frac[16,180] <= 1
 frac[16,181] <= 1
 frac[16,182] <= 1
 frac[16,183] <= 1
 frac[16,184] <= 1
 frac[16,185] <= 1
 frac[16,186] <= 1
 frac[16,187] <= 1
 frac[16,188] <= 1
 frac[16,189] <= 1
 frac[16,190] <= 1
 frac[16,191] <= 1
 frac[17,0] <= 1
 frac[17,1] <= 1
 frac[17,2] <= 1
 frac[17,3] <= 1
 frac[17,4] <= 1
 frac[17,5] <= 1
 frac[17,6] <= 1
 frac[17,7] <= 1
 frac[17,8] <= 1
 frac[17,9] <= 1
 frac[17,10] <= 1
 frac[17,11] <= 1
 frac[17,12] <= 1
 frac[17,13] <= 1
 frac[17,14] <= 1
 frac[17,15] <= 1
 frac[17,16] <= 1
 frac[17,17] <= 1
 frac[17,18] <= 1
 frac[17,19] <= 1
 frac[17,20] <= 1
 frac[17,21] <= 1
 frac[17,22] <= 1
 frac[17,23] <= 1
 frac[17,24] <= 1
 frac[17,25] <= 1
 frac[17,26] <= 1
 frac[17,27] <= 1
 frac[17,28] <= 1
 frac[17,29] <= 1
 frac[17,30] <= 1
 frac[17,31] <= 1
 frac[17,32] <= 1
 frac[17,33] <= 1
 frac[17,34] <= 1
 frac[17,35] <= 1
 frac[17,36] <= 1
 frac[17,37] <= 1
 frac[17,38] <= 1
 frac[17,39] <= 1
 frac[17,40] <= 1
 frac[17,41] <= 1
 frac[17,42] <= 1
 frac[17,43] <= 1
 frac[17,44] <= 1
 frac[17,45] <= 1
 frac[17,46] <= 1
 frac[17,47] <= 1
 frac[17,48] <= 1
 frac[17,49] <= 1
 frac[17,50] <= 1
 frac[17,51] <= 1
 frac[17,52] <= 1
 frac[17,53] <= 1
 frac[17,54] <= 1
 frac[17,55] <= 1
 frac[17,56] <= 1
 frac[17,57] <= 1
 frac[17,58] <= 1
 frac[17,59] <= 1
 frac[17,60] <= 1
 frac[17,61] <= 1
 frac[17,62] <= 1
 frac[17,63] <= 1
 frac[17,64] <= 1
 frac[17,65] <= 1
 frac[17,66] <= 1
 frac[17,67] <= 1
 frac[17,68] <= 1
 frac[17,69] <= 1
 frac[17,70] <= 1
 frac[17,71] <= 1
 frac[17,72] <= 1
 frac[17,73] <= 1
 frac[17,74] <= 1
 frac[17,75] <= 1
 frac[17,76] <= 1
 frac[17,77] <= 1
 frac[17,78] <= 1
 frac[17,79] <= 1
 frac[17,80] <= 1
 frac[17,81] <= 1
 frac[17,82] <= 1
 frac[17,83] <= 1
 frac[17,84] <= 1
 frac[17,85] <= 1
 frac[17,86] <= 1
 frac[17,87] <= 1
 frac[17,88] <= 1
 frac[17,89] <= 1
 frac[17,90] <= 1
 frac[17,91] <= 1
 frac[17,92] <= 1
 frac[17,93] <= 1
 frac[17,94] <= 1
 frac[17,95] <= 1
 frac[17,96] <= 1
 frac[17,97] <= 1
 frac[17,98] <= 1
 frac[17,99] <= 1
 frac[17,100] <= 1
 frac[17,101] <= 1
 frac[17,102] <= 1
 frac[17,103] <= 1
 frac[17,104] <= 1
 frac[17,105] <= 1
 frac[17,106] <= 1
 frac[17,107] <= 1
 frac[17,108] <= 1
 frac[17,109] <= 1
 frac[17,110] <= 1
 frac[17,111] <= 1
 frac[17,112] <= 1
 frac[17,113] <= 1
 frac[17,114] <= 1
 frac[17,115] <= 1
 frac[17,116] <= 1
 frac[17,117] <= 1
 frac[17,118] <= 1
 frac[17,119] <= 1
 frac[17,120] <= 1
 frac[17,121] <= 1
 frac[17,122] <= 1
 frac[17,123] <= 1
 frac[17,124] <= 1
 frac[17,125] <= 1
 frac[17,126] <= 1
 frac[17,127] <= 1
 frac[17,128] <= 1
 frac[17,129] <= 1
 frac[17,130] <= 1
 frac[17,131] <= 1
 frac[17,132] <= 1
 frac[17,133] <= 1
 frac[17,134] <= 1
 frac[17,135] <= 1
 frac[17,136] <= 1
 frac[17,137] <= 1
 frac[17,138] <= 1
 frac[17,139] <= 1
 frac[17,140] <= 1
 frac[17,141] <= 1
 frac[17,142] <= 1
 frac[17,143] <= 1
 frac[17,144] <= 1
 frac[17,145] <= 1
 frac[17,146] <= 1
 frac[17,147] <= 1
 frac[17,148] <= 1
 frac[17,149] <= 1
 frac[17,150] <= 1
 frac[17,151] <= 1
 frac[17,152] <= 1
 frac[17,153] <= 1
 frac[17,154] <= 1
 frac[17,155] <= 1
 frac[17,156] <= 1
 frac[17,157] <= 1
 frac[17,158] <= 1
 frac[17,159] <= 1
 frac[17,160] <= 1
 frac[17,161] <= 1
 frac[17,162] <= 1
 frac[17,163] <= 1
 frac[17,164] <= 1
 frac[17,165] <= 1
 frac[17,166] <= 1
 frac[17,167] <= 1
 frac[17,168] <= 1
 frac[17,169] <= 1
 frac[17,170] <= 1
 frac[17,171] <= 1
 frac[17,172] <= 1
 frac[17,173] <= 1
 frac[17,174] <= 1
 frac[17,175] <= 1
 frac[17,176] <= 1
 frac[17,177] <= 1
 frac[17,178] <= 1
 frac[17,179] <= 1
 frac[17,180] <= 1
 frac[17,181] <= 1
 frac[17,182] <= 1
 frac[17,183] <= 1
 frac[17,184] <= 1
 frac[17,185] <= 1
 frac[17,186] <= 1
 frac[17,187] <= 1
 frac[17,188] <= 1
 frac[17,189] <= 1
 frac[17,190] <= 1
 frac[17,191] <= 1
 frac[18,0] <= 1
 frac[18,1] <= 1
 frac[18,2] <= 1
 frac[18,3] <= 1
 frac[18,4] <= 1
 frac[18,5] <= 1
 frac[18,6] <= 1
 frac[18,7] <= 1
 frac[18,8] <= 1
 frac[18,9] <= 1
 frac[18,10] <= 1
 frac[18,11] <= 1
 frac[18,12] <= 1
 frac[18,13] <= 1
 frac[18,14] <= 1
 frac[18,15] <= 1
 frac[18,16] <= 1
 frac[18,17] <= 1
 frac[18,18] <= 1
 frac[18,19] <= 1
 frac[18,20] <= 1
 frac[18,21] <= 1
 frac[18,22] <= 1
 frac[18,23] <= 1
 frac[18,24] <= 1
 frac[18,25] <= 1
 frac[18,26] <= 1
 frac[18,27] <= 1
 frac[18,28] <= 1
 frac[18,29] <= 1
 frac[18,30] <= 1
 frac[18,31] <= 1
 frac[18,32] <= 1
 frac[18,33] <= 1
 frac[18,34] <= 1
 frac[18,35] <= 1
 frac[18,36] <= 1
 frac[18,37] <= 1
 frac[18,38] <= 1
 frac[18,39] <= 1
 frac[18,40] <= 1
 frac[18,41] <= 1
 frac[18,42] <= 1
 frac[18,43] <= 1
 frac[18,44] <= 1
 frac[18,45] <= 1
 frac[18,46] <= 1
 frac[18,47] <= 1
 frac[18,48] <= 1
 frac[18,49] <= 1
 frac[18,50] <= 1
 frac[18,51] <= 1
 frac[18,52] <= 1
 frac[18,53] <= 1
 frac[18,54] <= 1
 frac[18,55] <= 1
 frac[18,56] <= 1
 frac[18,57] <= 1
 frac[18,58] <= 1
 frac[18,59] <= 1
 frac[18,60] <= 1
 frac[18,61] <= 1
 frac[18,62] <= 1
 frac[18,63] <= 1
 frac[18,64] <= 1
 frac[18,65] <= 1
 frac[18,66] <= 1
 frac[18,67] <= 1
 frac[18,68] <= 1
 frac[18,69] <= 1
 frac[18,70] <= 1
 frac[18,71] <= 1
 frac[18,72] <= 1
 frac[18,73] <= 1
 frac[18,74] <= 1
 frac[18,75] <= 1
 frac[18,76] <= 1
 frac[18,77] <= 1
 frac[18,78] <= 1
 frac[18,79] <= 1
 frac[18,80] <= 1
 frac[18,81] <= 1
 frac[18,82] <= 1
 frac[18,83] <= 1
 frac[18,84] <= 1
 frac[18,85] <= 1
 frac[18,86] <= 1
 frac[18,87] <= 1
 frac[18,88] <= 1
 frac[18,89] <= 1
 frac[18,90] <= 1
 frac[18,91] <= 1
 frac[18,92] <= 1
 frac[18,93] <= 1
 frac[18,94] <= 1
 frac[18,95] <= 1
 frac[18,96] <= 1
 frac[18,97] <= 1
 frac[18,98] <= 1
 frac[18,99] <= 1
 frac[18,100] <= 1
 frac[18,101] <= 1
 frac[18,102] <= 1
 frac[18,103] <= 1
 frac[18,104] <= 1
 frac[18,105] <= 1
 frac[18,106] <= 1
 frac[18,107] <= 1
 frac[18,108] <= 1
 frac[18,109] <= 1
 frac[18,110] <= 1
 frac[18,111] <= 1
 frac[18,112] <= 1
 frac[18,113] <= 1
 frac[18,114] <= 1
 frac[18,115] <= 1
 frac[18,116] <= 1
 frac[18,117] <= 1
 frac[18,118] <= 1
 frac[18,119] <= 1
 frac[18,120] <= 1
 frac[18,121] <= 1
 frac[18,122] <= 1
 frac[18,123] <= 1
 frac[18,124] <= 1
 frac[18,125] <= 1
 frac[18,126] <= 1
 frac[18,127] <= 1
 frac[18,128] <= 1
 frac[18,129] <= 1
 frac[18,130] <= 1
 frac[18,131] <= 1
 frac[18,132] <= 1
 frac[18,133] <= 1
 frac[18,134] <= 1
 frac[18,135] <= 1
 frac[18,136] <= 1
 frac[18,137] <= 1
 frac[18,138] <= 1
 frac[18,139] <= 1
 frac[18,140] <= 1
 frac[18,141] <= 1
 frac[18,142] <= 1
 frac[18,143] <= 1
 frac[18,144] <= 1
 frac[18,145] <= 1
 frac[18,146] <= 1
 frac[18,147] <= 1
 frac[18,148] <= 1
 frac[18,149] <= 1
 frac[18,150] <= 1
 frac[18,151] <= 1
 frac[18,152] <= 1
 frac[18,153] <= 1
 frac[18,154] <= 1
 frac[18,155] <= 1
 frac[18,156] <= 1
 frac[18,157] <= 1
 frac[18,158] <= 1
 frac[18,159] <= 1
 frac[18,160] <= 1
 frac[18,161] <= 1
 frac[18,162] <= 1
 frac[18,163] <= 1
 frac[18,164] <= 1
 frac[18,165] <= 1
 frac[18,166] <= 1
 frac[18,167] <= 1
 frac[18,168] <= 1
 frac[18,169] <= 1
 frac[18,170] <= 1
 frac[18,171] <= 1
 frac[18,172] <= 1
 frac[18,173] <= 1
 frac[18,174] <= 1
 frac[18,175] <= 1
 frac[18,176] <= 1
 frac[18,177] <= 1
 frac[18,178] <= 1
 frac[18,179] <= 1
 frac[18,180] <= 1
 frac[18,181] <= 1
 frac[18,182] <= 1
 frac[18,183] <= 1
 frac[18,184] <= 1
 frac[18,185] <= 1
 frac[18,186] <= 1
 frac[18,187] <= 1
 frac[18,188] <= 1
 frac[18,189] <= 1
 frac[18,190] <= 1
 frac[18,191] <= 1
 frac[19,0] <= 1
 frac[19,1] <= 1
 frac[19,2] <= 1
 frac[19,3] <= 1
 frac[19,4] <= 1
 frac[19,5] <= 1
 frac[19,6] <= 1
 frac[19,7] <= 1
 frac[19,8] <= 1
 frac[19,9] <= 1
 frac[19,10] <= 1
 frac[19,11] <= 1
 frac[19,12] <= 1
 frac[19,13] <= 1
 frac[19,14] <= 1
 frac[19,15] <= 1
 frac[19,16] <= 1
 frac[19,17] <= 1
 frac[19,18] <= 1
 frac[19,19] <= 1
 frac[19,20] <= 1
 frac[19,21] <= 1
 frac[19,22] <= 1
 frac[19,23] <= 1
 frac[19,24] <= 1
 frac[19,25] <= 1
 frac[19,26] <= 1
 frac[19,27] <= 1
 frac[19,28] <= 1
 frac[19,29] <= 1
 frac[19,30] <= 1
 frac[19,31] <= 1
 frac[19,32] <= 1
 frac[19,33] <= 1
 frac[19,34] <= 1
 frac[19,35] <= 1
 frac[19,36] <= 1
 frac[19,37] <= 1
 frac[19,38] <= 1
 frac[19,39] <= 1
 frac[19,40] <= 1
 frac[19,41] <= 1
 frac[19,42] <= 1
 frac[19,43] <= 1
 frac[19,44] <= 1
 frac[19,45] <= 1
 frac[19,46] <= 1
 frac[19,47] <= 1
 frac[19,48] <= 1
 frac[19,49] <= 1
 frac[19,50] <= 1
 frac[19,51] <= 1
 frac[19,52] <= 1
 frac[19,53] <= 1
 frac[19,54] <= 1
 frac[19,55] <= 1
 frac[19,56] <= 1
 frac[19,57] <= 1
 frac[19,58] <= 1
 frac[19,59] <= 1
 frac[19,60] <= 1
 frac[19,61] <= 1
 frac[19,62] <= 1
 frac[19,63] <= 1
 frac[19,64] <= 1
 frac[19,65] <= 1
 frac[19,66] <= 1
 frac[19,67] <= 1
 frac[19,68] <= 1
 frac[19,69] <= 1
 frac[19,70] <= 1
 frac[19,71] <= 1
 frac[19,72] <= 1
 frac[19,73] <= 1
 frac[19,74] <= 1
 frac[19,75] <= 1
 frac[19,76] <= 1
 frac[19,77] <= 1
 frac[19,78] <= 1
 frac[19,79] <= 1
 frac[19,80] <= 1
 frac[19,81] <= 1
 frac[19,82] <= 1
 frac[19,83] <= 1
 frac[19,84] <= 1
 frac[19,85] <= 1
 frac[19,86] <= 1
 frac[19,87] <= 1
 frac[19,88] <= 1
 frac[19,89] <= 1
 frac[19,90] <= 1
 frac[19,91] <= 1
 frac[19,92] <= 1
 frac[19,93] <= 1
 frac[19,94] <= 1
 frac[19,95] <= 1
 frac[19,96] <= 1
 frac[19,97] <= 1
 frac[19,98] <= 1
 frac[19,99] <= 1
 frac[19,100] <= 1
 frac[19,101] <= 1
 frac[19,102] <= 1
 frac[19,103] <= 1
 frac[19,104] <= 1
 frac[19,105] <= 1
 frac[19,106] <= 1
 frac[19,107] <= 1
 frac[19,108] <= 1
 frac[19,109] <= 1
 frac[19,110] <= 1
 frac[19,111] <= 1
 frac[19,112] <= 1
 frac[19,113] <= 1
 frac[19,114] <= 1
 frac[19,115] <= 1
 frac[19,116] <= 1
 frac[19,117] <= 1
 frac[19,118] <= 1
 frac[19,119] <= 1
 frac[19,120] <= 1
 frac[19,121] <= 1
 frac[19,122] <= 1
 frac[19,123] <= 1
 frac[19,124] <= 1
 frac[19,125] <= 1
 frac[19,126] <= 1
 frac[19,127] <= 1
 frac[19,128] <= 1
 frac[19,129] <= 1
 frac[19,130] <= 1
 frac[19,131] <= 1
 frac[19,132] <= 1
 frac[19,133] <= 1
 frac[19,134] <= 1
 frac[19,135] <= 1
 frac[19,136] <= 1
 frac[19,137] <= 1
 frac[19,138] <= 1
 frac[19,139] <= 1
 frac[19,140] <= 1
 frac[19,141] <= 1
 frac[19,142] <= 1
 frac[19,143] <= 1
 frac[19,144] <= 1
 frac[19,145] <= 1
 frac[19,146] <= 1
 frac[19,147] <= 1
 frac[19,148] <= 1
 frac[19,149] <= 1
 frac[19,150] <= 1
 frac[19,151] <= 1
 frac[19,152] <= 1
 frac[19,153] <= 1
 frac[19,154] <= 1
 frac[19,155] <= 1
 frac[19,156] <= 1
 frac[19,157] <= 1
 frac[19,158] <= 1
 frac[19,159] <= 1
 frac[19,160] <= 1
 frac[19,161] <= 1
 frac[19,162] <= 1
 frac[19,163] <= 1
 frac[19,164] <= 1
 frac[19,165] <= 1
 frac[19,166] <= 1
 frac[19,167] <= 1
 frac[19,168] <= 1
 frac[19,169] <= 1
 frac[19,170] <= 1
 frac[19,171] <= 1
 frac[19,172] <= 1
 frac[19,173] <= 1
 frac[19,174] <= 1
 frac[19,175] <= 1
 frac[19,176] <= 1
 frac[19,177] <= 1
 frac[19,178] <= 1
 frac[19,179] <= 1
 frac[19,180] <= 1
 frac[19,181] <= 1
 frac[19,182] <= 1
 frac[19,183] <= 1
 frac[19,184] <= 1
 frac[19,185] <= 1
 frac[19,186] <= 1
 frac[19,187] <= 1
 frac[19,188] <= 1
 frac[19,189] <= 1
 frac[19,190] <= 1
 frac[19,191] <= 1
 frac[20,0] <= 1
 frac[20,1] <= 1
 frac[20,2] <= 1
 frac[20,3] <= 1
 frac[20,4] <= 1
 frac[20,5] <= 1
 frac[20,6] <= 1
 frac[20,7] <= 1
 frac[20,8] <= 1
 frac[20,9] <= 1
 frac[20,10] <= 1
 frac[20,11] <= 1
 frac[20,12] <= 1
 frac[20,13] <= 1
 frac[20,14] <= 1
 frac[20,15] <= 1
 frac[20,16] <= 1
 frac[20,17] <= 1
 frac[20,18] <= 1
 frac[20,19] <= 1
 frac[20,20] <= 1
 frac[20,21] <= 1
 frac[20,22] <= 1
 frac[20,23] <= 1
 frac[20,24] <= 1
 frac[20,25] <= 1
 frac[20,26] <= 1
 frac[20,27] <= 1
 frac[20,28] <= 1
 frac[20,29] <= 1
 frac[20,30] <= 1
 frac[20,31] <= 1
 frac[20,32] <= 1
 frac[20,33] <= 1
 frac[20,34] <= 1
 frac[20,35] <= 1
 frac[20,36] <= 1
 frac[20,37] <= 1
 frac[20,38] <= 1
 frac[20,39] <= 1
 frac[20,40] <= 1
 frac[20,41] <= 1
 frac[20,42] <= 1
 frac[20,43] <= 1
 frac[20,44] <= 1
 frac[20,45] <= 1
 frac[20,46] <= 1
 frac[20,47] <= 1
 frac[20,48] <= 1
 frac[20,49] <= 1
 frac[20,50] <= 1
 frac[20,51] <= 1
 frac[20,52] <= 1
 frac[20,53] <= 1
 frac[20,54] <= 1
 frac[20,55] <= 1
 frac[20,56] <= 1
 frac[20,57] <= 1
 frac[20,58] <= 1
 frac[20,59] <= 1
 frac[20,60] <= 1
 frac[20,61] <= 1
 frac[20,62] <= 1
 frac[20,63] <= 1
 frac[20,64] <= 1
 frac[20,65] <= 1
 frac[20,66] <= 1
 frac[20,67] <= 1
 frac[20,68] <= 1
 frac[20,69] <= 1
 frac[20,70] <= 1
 frac[20,71] <= 1
 frac[20,72] <= 1
 frac[20,73] <= 1
 frac[20,74] <= 1
 frac[20,75] <= 1
 frac[20,76] <= 1
 frac[20,77] <= 1
 frac[20,78] <= 1
 frac[20,79] <= 1
 frac[20,80] <= 1
 frac[20,81] <= 1
 frac[20,82] <= 1
 frac[20,83] <= 1
 frac[20,84] <= 1
 frac[20,85] <= 1
 frac[20,86] <= 1
 frac[20,87] <= 1
 frac[20,88] <= 1
 frac[20,89] <= 1
 frac[20,90] <= 1
 frac[20,91] <= 1
 frac[20,92] <= 1
 frac[20,93] <= 1
 frac[20,94] <= 1
 frac[20,95] <= 1
 frac[20,96] <= 1
 frac[20,97] <= 1
 frac[20,98] <= 1
 frac[20,99] <= 1
 frac[20,100] <= 1
 frac[20,101] <= 1
 frac[20,102] <= 1
 frac[20,103] <= 1
 frac[20,104] <= 1
 frac[20,105] <= 1
 frac[20,106] <= 1
 frac[20,107] <= 1
 frac[20,108] <= 1
 frac[20,109] <= 1
 frac[20,110] <= 1
 frac[20,111] <= 1
 frac[20,112] <= 1
 frac[20,113] <= 1
 frac[20,114] <= 1
 frac[20,115] <= 1
 frac[20,116] <= 1
 frac[20,117] <= 1
 frac[20,118] <= 1
 frac[20,119] <= 1
 frac[20,120] <= 1
 frac[20,121] <= 1
 frac[20,122] <= 1
 frac[20,123] <= 1
 frac[20,124] <= 1
 frac[20,125] <= 1
 frac[20,126] <= 1
 frac[20,127] <= 1
 frac[20,128] <= 1
 frac[20,129] <= 1
 frac[20,130] <= 1
 frac[20,131] <= 1
 frac[20,132] <= 1
 frac[20,133] <= 1
 frac[20,134] <= 1
 frac[20,135] <= 1
 frac[20,136] <= 1
 frac[20,137] <= 1
 frac[20,138] <= 1
 frac[20,139] <= 1
 frac[20,140] <= 1
 frac[20,141] <= 1
 frac[20,142] <= 1
 frac[20,143] <= 1
 frac[20,144] <= 1
 frac[20,145] <= 1
 frac[20,146] <= 1
 frac[20,147] <= 1
 frac[20,148] <= 1
 frac[20,149] <= 1
 frac[20,150] <= 1
 frac[20,151] <= 1
 frac[20,152] <= 1
 frac[20,153] <= 1
 frac[20,154] <= 1
 frac[20,155] <= 1
 frac[20,156] <= 1
 frac[20,157] <= 1
 frac[20,158] <= 1
 frac[20,159] <= 1
 frac[20,160] <= 1
 frac[20,161] <= 1
 frac[20,162] <= 1
 frac[20,163] <= 1
 frac[20,164] <= 1
 frac[20,165] <= 1
 frac[20,166] <= 1
 frac[20,167] <= 1
 frac[20,168] <= 1
 frac[20,169] <= 1
 frac[20,170] <= 1
 frac[20,171] <= 1
 frac[20,172] <= 1
 frac[20,173] <= 1
 frac[20,174] <= 1
 frac[20,175] <= 1
 frac[20,176] <= 1
 frac[20,177] <= 1
 frac[20,178] <= 1
 frac[20,179] <= 1
 frac[20,180] <= 1
 frac[20,181] <= 1
 frac[20,182] <= 1
 frac[20,183] <= 1
 frac[20,184] <= 1
 frac[20,185] <= 1
 frac[20,186] <= 1
 frac[20,187] <= 1
 frac[20,188] <= 1
 frac[20,189] <= 1
 frac[20,190] <= 1
 frac[20,191] <= 1
 frac[21,0] <= 1
 frac[21,1] <= 1
 frac[21,2] <= 1
 frac[21,3] <= 1
 frac[21,4] <= 1
 frac[21,5] <= 1
 frac[21,6] <= 1
 frac[21,7] <= 1
 frac[21,8] <= 1
 frac[21,9] <= 1
 frac[21,10] <= 1
 frac[21,11] <= 1
 frac[21,12] <= 1
 frac[21,13] <= 1
 frac[21,14] <= 1
 frac[21,15] <= 1
 frac[21,16] <= 1
 frac[21,17] <= 1
 frac[21,18] <= 1
 frac[21,19] <= 1
 frac[21,20] <= 1
 frac[21,21] <= 1
 frac[21,22] <= 1
 frac[21,23] <= 1
 frac[21,24] <= 1
 frac[21,25] <= 1
 frac[21,26] <= 1
 frac[21,27] <= 1
 frac[21,28] <= 1
 frac[21,29] <= 1
 frac[21,30] <= 1
 frac[21,31] <= 1
 frac[21,32] <= 1
 frac[21,33] <= 1
 frac[21,34] <= 1
 frac[21,35] <= 1
 frac[21,36] <= 1
 frac[21,37] <= 1
 frac[21,38] <= 1
 frac[21,39] <= 1
 frac[21,40] <= 1
 frac[21,41] <= 1
 frac[21,42] <= 1
 frac[21,43] <= 1
 frac[21,44] <= 1
 frac[21,45] <= 1
 frac[21,46] <= 1
 frac[21,47] <= 1
 frac[21,48] <= 1
 frac[21,49] <= 1
 frac[21,50] <= 1
 frac[21,51] <= 1
 frac[21,52] <= 1
 frac[21,53] <= 1
 frac[21,54] <= 1
 frac[21,55] <= 1
 frac[21,56] <= 1
 frac[21,57] <= 1
 frac[21,58] <= 1
 frac[21,59] <= 1
 frac[21,60] <= 1
 frac[21,61] <= 1
 frac[21,62] <= 1
 frac[21,63] <= 1
 frac[21,64] <= 1
 frac[21,65] <= 1
 frac[21,66] <= 1
 frac[21,67] <= 1
 frac[21,68] <= 1
 frac[21,69] <= 1
 frac[21,70] <= 1
 frac[21,71] <= 1
 frac[21,72] <= 1
 frac[21,73] <= 1
 frac[21,74] <= 1
 frac[21,75] <= 1
 frac[21,76] <= 1
 frac[21,77] <= 1
 frac[21,78] <= 1
 frac[21,79] <= 1
 frac[21,80] <= 1
 frac[21,81] <= 1
 frac[21,82] <= 1
 frac[21,83] <= 1
 frac[21,84] <= 1
 frac[21,85] <= 1
 frac[21,86] <= 1
 frac[21,87] <= 1
 frac[21,88] <= 1
 frac[21,89] <= 1
 frac[21,90] <= 1
 frac[21,91] <= 1
 frac[21,92] <= 1
 frac[21,93] <= 1
 frac[21,94] <= 1
 frac[21,95] <= 1
 frac[21,96] <= 1
 frac[21,97] <= 1
 frac[21,98] <= 1
 frac[21,99] <= 1
 frac[21,100] <= 1
 frac[21,101] <= 1
 frac[21,102] <= 1
 frac[21,103] <= 1
 frac[21,104] <= 1
 frac[21,105] <= 1
 frac[21,106] <= 1
 frac[21,107] <= 1
 frac[21,108] <= 1
 frac[21,109] <= 1
 frac[21,110] <= 1
 frac[21,111] <= 1
 frac[21,112] <= 1
 frac[21,113] <= 1
 frac[21,114] <= 1
 frac[21,115] <= 1
 frac[21,116] <= 1
 frac[21,117] <= 1
 frac[21,118] <= 1
 frac[21,119] <= 1
 frac[21,120] <= 1
 frac[21,121] <= 1
 frac[21,122] <= 1
 frac[21,123] <= 1
 frac[21,124] <= 1
 frac[21,125] <= 1
 frac[21,126] <= 1
 frac[21,127] <= 1
 frac[21,128] <= 1
 frac[21,129] <= 1
 frac[21,130] <= 1
 frac[21,131] <= 1
 frac[21,132] <= 1
 frac[21,133] <= 1
 frac[21,134] <= 1
 frac[21,135] <= 1
 frac[21,136] <= 1
 frac[21,137] <= 1
 frac[21,138] <= 1
 frac[21,139] <= 1
 frac[21,140] <= 1
 frac[21,141] <= 1
 frac[21,142] <= 1
 frac[21,143] <= 1
 frac[21,144] <= 1
 frac[21,145] <= 1
 frac[21,146] <= 1
 frac[21,147] <= 1
 frac[21,148] <= 1
 frac[21,149] <= 1
 frac[21,150] <= 1
 frac[21,151] <= 1
 frac[21,152] <= 1
 frac[21,153] <= 1
 frac[21,154] <= 1
 frac[21,155] <= 1
 frac[21,156] <= 1
 frac[21,157] <= 1
 frac[21,158] <= 1
 frac[21,159] <= 1
 frac[21,160] <= 1
 frac[21,161] <= 1
 frac[21,162] <= 1
 frac[21,163] <= 1
 frac[21,164] <= 1
 frac[21,165] <= 1
 frac[21,166] <= 1
 frac[21,167] <= 1
 frac[21,168] <= 1
 frac[21,169] <= 1
 frac[21,170] <= 1
 frac[21,171] <= 1
 frac[21,172] <= 1
 frac[21,173] <= 1
 frac[21,174] <= 1
 frac[21,175] <= 1
 frac[21,176] <= 1
 frac[21,177] <= 1
 frac[21,178] <= 1
 frac[21,179] <= 1
 frac[21,180] <= 1
 frac[21,181] <= 1
 frac[21,182] <= 1
 frac[21,183] <= 1
 frac[21,184] <= 1
 frac[21,185] <= 1
 frac[21,186] <= 1
 frac[21,187] <= 1
 frac[21,188] <= 1
 frac[21,189] <= 1
 frac[21,190] <= 1
 frac[21,191] <= 1
 frac[22,0] <= 1
 frac[22,1] <= 1
 frac[22,2] <= 1
 frac[22,3] <= 1
 frac[22,4] <= 1
 frac[22,5] <= 1
 frac[22,6] <= 1
 frac[22,7] <= 1
 frac[22,8] <= 1
 frac[22,9] <= 1
 frac[22,10] <= 1
 frac[22,11] <= 1
 frac[22,12] <= 1
 frac[22,13] <= 1
 frac[22,14] <= 1
 frac[22,15] <= 1
 frac[22,16] <= 1
 frac[22,17] <= 1
 frac[22,18] <= 1
 frac[22,19] <= 1
 frac[22,20] <= 1
 frac[22,21] <= 1
 frac[22,22] <= 1
 frac[22,23] <= 1
 frac[22,24] <= 1
 frac[22,25] <= 1
 frac[22,26] <= 1
 frac[22,27] <= 1
 frac[22,28] <= 1
 frac[22,29] <= 1
 frac[22,30] <= 1
 frac[22,31] <= 1
 frac[22,32] <= 1
 frac[22,33] <= 1
 frac[22,34] <= 1
 frac[22,35] <= 1
 frac[22,36] <= 1
 frac[22,37] <= 1
 frac[22,38] <= 1
 frac[22,39] <= 1
 frac[22,40] <= 1
 frac[22,41] <= 1
 frac[22,42] <= 1
 frac[22,43] <= 1
 frac[22,44] <= 1
 frac[22,45] <= 1
 frac[22,46] <= 1
 frac[22,47] <= 1
 frac[22,48] <= 1
 frac[22,49] <= 1
 frac[22,50] <= 1
 frac[22,51] <= 1
 frac[22,52] <= 1
 frac[22,53] <= 1
 frac[22,54] <= 1
 frac[22,55] <= 1
 frac[22,56] <= 1
 frac[22,57] <= 1
 frac[22,58] <= 1
 frac[22,59] <= 1
 frac[22,60] <= 1
 frac[22,61] <= 1
 frac[22,62] <= 1
 frac[22,63] <= 1
 frac[22,64] <= 1
 frac[22,65] <= 1
 frac[22,66] <= 1
 frac[22,67] <= 1
 frac[22,68] <= 1
 frac[22,69] <= 1
 frac[22,70] <= 1
 frac[22,71] <= 1
 frac[22,72] <= 1
 frac[22,73] <= 1
 frac[22,74] <= 1
 frac[22,75] <= 1
 frac[22,76] <= 1
 frac[22,77] <= 1
 frac[22,78] <= 1
 frac[22,79] <= 1
 frac[22,80] <= 1
 frac[22,81] <= 1
 frac[22,82] <= 1
 frac[22,83] <= 1
 frac[22,84] <= 1
 frac[22,85] <= 1
 frac[22,86] <= 1
 frac[22,87] <= 1
 frac[22,88] <= 1
 frac[22,89] <= 1
 frac[22,90] <= 1
 frac[22,91] <= 1
 frac[22,92] <= 1
 frac[22,93] <= 1
 frac[22,94] <= 1
 frac[22,95] <= 1
 frac[22,96] <= 1
 frac[22,97] <= 1
 frac[22,98] <= 1
 frac[22,99] <= 1
 frac[22,100] <= 1
 frac[22,101] <= 1
 frac[22,102] <= 1
 frac[22,103] <= 1
 frac[22,104] <= 1
 frac[22,105] <= 1
 frac[22,106] <= 1
 frac[22,107] <= 1
 frac[22,108] <= 1
 frac[22,109] <= 1
 frac[22,110] <= 1
 frac[22,111] <= 1
 frac[22,112] <= 1
 frac[22,113] <= 1
 frac[22,114] <= 1
 frac[22,115] <= 1
 frac[22,116] <= 1
 frac[22,117] <= 1
 frac[22,118] <= 1
 frac[22,119] <= 1
 frac[22,120] <= 1
 frac[22,121] <= 1
 frac[22,122] <= 1
 frac[22,123] <= 1
 frac[22,124] <= 1
 frac[22,125] <= 1
 frac[22,126] <= 1
 frac[22,127] <= 1
 frac[22,128] <= 1
 frac[22,129] <= 1
 frac[22,130] <= 1
 frac[22,131] <= 1
 frac[22,132] <= 1
 frac[22,133] <= 1
 frac[22,134] <= 1
 frac[22,135] <= 1
 frac[22,136] <= 1
 frac[22,137] <= 1
 frac[22,138] <= 1
 frac[22,139] <= 1
 frac[22,140] <= 1
 frac[22,141] <= 1
 frac[22,142] <= 1
 frac[22,143] <= 1
 frac[22,144] <= 1
 frac[22,145] <= 1
 frac[22,146] <= 1
 frac[22,147] <= 1
 frac[22,148] <= 1
 frac[22,149] <= 1
 frac[22,150] <= 1
 frac[22,151] <= 1
 frac[22,152] <= 1
 frac[22,153] <= 1
 frac[22,154] <= 1
 frac[22,155] <= 1
 frac[22,156] <= 1
 frac[22,157] <= 1
 frac[22,158] <= 1
 frac[22,159] <= 1
 frac[22,160] <= 1
 frac[22,161] <= 1
 frac[22,162] <= 1
 frac[22,163] <= 1
 frac[22,164] <= 1
 frac[22,165] <= 1
 frac[22,166] <= 1
 frac[22,167] <= 1
 frac[22,168] <= 1
 frac[22,169] <= 1
 frac[22,170] <= 1
 frac[22,171] <= 1
 frac[22,172] <= 1
 frac[22,173] <= 1
 frac[22,174] <= 1
 frac[22,175] <= 1
 frac[22,176] <= 1
 frac[22,177] <= 1
 frac[22,178] <= 1
 frac[22,179] <= 1
 frac[22,180] <= 1
 frac[22,181] <= 1
 frac[22,182] <= 1
 frac[22,183] <= 1
 frac[22,184] <= 1
 frac[22,185] <= 1
 frac[22,186] <= 1
 frac[22,187] <= 1
 frac[22,188] <= 1
 frac[22,189] <= 1
 frac[22,190] <= 1
 frac[22,191] <= 1
 rows_tot_cpu1 <= 12
 mem_tot_cpu1 <= 32768
 cores_sketch_cpu1 <= 7
 1 <= cores_dpdk_cpu1 <= 7
 rows_tot_cpu2 <= 12
 mem_tot_cpu2 <= 32768
 cores_sketch_cpu2 <= 7
 1 <= cores_dpdk_cpu2 <= 7
 rows_tot_cpu3 <= 12
 mem_tot_cpu3 <= 32768
 cores_sketch_cpu3 <= 7
 1 <= cores_dpdk_cpu3 <= 7
 rows_tot_cpu4 <= 12
 mem_tot_cpu4 <= 32768
 cores_sketch_cpu4 <= 7
 1 <= cores_dpdk_cpu4 <= 7
 rows_tot_cpu5 <= 12
 mem_tot_cpu5 <= 32768
 cores_sketch_cpu5 <= 7
 1 <= cores_dpdk_cpu5 <= 7
 rows_tot_cpu6 <= 12
 mem_tot_cpu6 <= 32768
 cores_sketch_cpu6 <= 7
 1 <= cores_dpdk_cpu6 <= 7
 rows_tot_cpu7 <= 12
 mem_tot_cpu7 <= 32768
 cores_sketch_cpu7 <= 7
 1 <= cores_dpdk_cpu7 <= 7
 rows_tot_cpu8 <= 12
 mem_tot_cpu8 <= 32768
 cores_sketch_cpu8 <= 7
 1 <= cores_dpdk_cpu8 <= 7
 rows_tot_cpu9 <= 12
 mem_tot_cpu9 <= 32768
 cores_sketch_cpu9 <= 7
 1 <= cores_dpdk_cpu9 <= 7
 rows_tot_cpu10 <= 12
 mem_tot_cpu10 <= 32768
 cores_sketch_cpu10 <= 7
 1 <= cores_dpdk_cpu10 <= 7
 rows_tot_cpu11 <= 12
 mem_tot_cpu11 <= 32768
 cores_sketch_cpu11 <= 7
 1 <= cores_dpdk_cpu11 <= 7
 rows_tot_cpu12 <= 12
 mem_tot_cpu12 <= 32768
 cores_sketch_cpu12 <= 7
 1 <= cores_dpdk_cpu12 <= 7
 rows_tot_cpu13 <= 12
 mem_tot_cpu13 <= 32768
 cores_sketch_cpu13 <= 7
 1 <= cores_dpdk_cpu13 <= 7
 rows_tot_cpu14 <= 12
 mem_tot_cpu14 <= 32768
 cores_sketch_cpu14 <= 7
 1 <= cores_dpdk_cpu14 <= 7
 rows_tot_cpu15 <= 12
 mem_tot_cpu15 <= 32768
 cores_sketch_cpu15 <= 7
 1 <= cores_dpdk_cpu15 <= 7
 rows_tot_cpu16 <= 12
 mem_tot_cpu16 <= 32768
 cores_sketch_cpu16 <= 7
 1 <= cores_dpdk_cpu16 <= 7
 rows_tot_tor_p41 <= 48
 mem_tot_tor_p41 <= 576
 rows_tot_tor_p42 <= 48
 mem_tot_tor_p42 <= 576
 rows_tot_tor_p43 <= 48
 mem_tot_tor_p43 <= 576
 rows_tot_tor_p44 <= 48
 mem_tot_tor_p44 <= 576
 rows_tot_l1_p41 <= 48
 mem_tot_l1_p41 <= 576
 rows_tot_l1_p42 <= 48
 mem_tot_l1_p42 <= 576
 rows_tot_l2_p4 <= 48
 mem_tot_l2_p4 <= 576
Generals
 cores_sketch_cpu1 cores_dpdk_cpu1 cores_sketch_cpu2 cores_dpdk_cpu2
 cores_sketch_cpu3 cores_dpdk_cpu3 cores_sketch_cpu4 cores_dpdk_cpu4
 cores_sketch_cpu5 cores_dpdk_cpu5 cores_sketch_cpu6 cores_dpdk_cpu6
 cores_sketch_cpu7 cores_dpdk_cpu7 cores_sketch_cpu8 cores_dpdk_cpu8
 cores_sketch_cpu9 cores_dpdk_cpu9 cores_sketch_cpu10 cores_dpdk_cpu10
 cores_sketch_cpu11 cores_dpdk_cpu11 cores_sketch_cpu12 cores_dpdk_cpu12
 cores_sketch_cpu13 cores_dpdk_cpu13 cores_sketch_cpu14 cores_dpdk_cpu14
 cores_sketch_cpu15 cores_dpdk_cpu15 cores_sketch_cpu16 cores_dpdk_cpu16
General Constraints
 mem_access_time_cpu1: m_access_time_cpu1 = PWL ( mem_tot_cpu1 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu1: ns_cpu1 = MAX ( ns_dpdk_cpu1 , ns_sketch_cpu1 )
 mem_access_time_cpu2: m_access_time_cpu2 = PWL ( mem_tot_cpu2 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu2: ns_cpu2 = MAX ( ns_dpdk_cpu2 , ns_sketch_cpu2 )
 mem_access_time_cpu3: m_access_time_cpu3 = PWL ( mem_tot_cpu3 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu3: ns_cpu3 = MAX ( ns_dpdk_cpu3 , ns_sketch_cpu3 )
 mem_access_time_cpu4: m_access_time_cpu4 = PWL ( mem_tot_cpu4 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu4: ns_cpu4 = MAX ( ns_dpdk_cpu4 , ns_sketch_cpu4 )
 mem_access_time_cpu5: m_access_time_cpu5 = PWL ( mem_tot_cpu5 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu5: ns_cpu5 = MAX ( ns_dpdk_cpu5 , ns_sketch_cpu5 )
 mem_access_time_cpu6: m_access_time_cpu6 = PWL ( mem_tot_cpu6 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu6: ns_cpu6 = MAX ( ns_dpdk_cpu6 , ns_sketch_cpu6 )
 mem_access_time_cpu7: m_access_time_cpu7 = PWL ( mem_tot_cpu7 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu7: ns_cpu7 = MAX ( ns_dpdk_cpu7 , ns_sketch_cpu7 )
 mem_access_time_cpu8: m_access_time_cpu8 = PWL ( mem_tot_cpu8 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu8: ns_cpu8 = MAX ( ns_dpdk_cpu8 , ns_sketch_cpu8 )
 mem_access_time_cpu9: m_access_time_cpu9 = PWL ( mem_tot_cpu9 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu9: ns_cpu9 = MAX ( ns_dpdk_cpu9 , ns_sketch_cpu9 )
 mem_access_time_cpu10: m_access_time_cpu10 = PWL ( mem_tot_cpu10 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu10: ns_cpu10 = MAX ( ns_dpdk_cpu10 , ns_sketch_cpu10 )
 mem_access_time_cpu11: m_access_time_cpu11 = PWL ( mem_tot_cpu11 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu11: ns_cpu11 = MAX ( ns_dpdk_cpu11 , ns_sketch_cpu11 )
 mem_access_time_cpu12: m_access_time_cpu12 = PWL ( mem_tot_cpu12 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu12: ns_cpu12 = MAX ( ns_dpdk_cpu12 , ns_sketch_cpu12 )
 mem_access_time_cpu13: m_access_time_cpu13 = PWL ( mem_tot_cpu13 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu13: ns_cpu13 = MAX ( ns_dpdk_cpu13 , ns_sketch_cpu13 )
 mem_access_time_cpu14: m_access_time_cpu14 = PWL ( mem_tot_cpu14 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu14: ns_cpu14 = MAX ( ns_dpdk_cpu14 , ns_sketch_cpu14 )
 mem_access_time_cpu15: m_access_time_cpu15 = PWL ( mem_tot_cpu15 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu15: ns_cpu15 = MAX ( ns_dpdk_cpu15 , ns_sketch_cpu15 )
 mem_access_time_cpu16: m_access_time_cpu16 = PWL ( mem_tot_cpu16 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu16: ns_cpu16 = MAX ( ns_dpdk_cpu16 , ns_sketch_cpu16 )
 ns_overall: ns = MAX ( ns_cpu1 , ns_cpu2 , ns_cpu3 , ns_cpu4 , ns_cpu5 ,
   ns_cpu6 , ns_cpu7 , ns_cpu8 , ns_cpu9 , ns_cpu10 , ns_cpu11 , ns_cpu12 ,
   ns_cpu13 , ns_cpu14 , ns_cpu15 , ns_cpu16 , ns_tor_p41 , ns_tor_p42 ,
   ns_tor_p43 , ns_tor_p44 , ns_l1_p41 , ns_l1_p42 , ns_l2_p4 )
 mem_overall_p4: max_mem_p4 = MAX ( mem_tot_tor_p41 , mem_tot_tor_p42 ,
   mem_tot_tor_p43 , mem_tot_tor_p44 , mem_tot_l1_p41 , mem_tot_l1_p42 ,
   mem_tot_l2_p4 )
 mem_overall_cpu: max_mem_cpu = MAX ( mem_tot_cpu1 , mem_tot_cpu2 ,
   mem_tot_cpu3 , mem_tot_cpu4 , mem_tot_cpu5 , mem_tot_cpu6 ,
   mem_tot_cpu7 , mem_tot_cpu8 , mem_tot_cpu9 , mem_tot_cpu10 ,
   mem_tot_cpu11 , mem_tot_cpu12 , mem_tot_cpu13 , mem_tot_cpu14 ,
   mem_tot_cpu15 , mem_tot_cpu16 )
 rows_overall_p4: max_rows_p4 = MAX ( rows_tot_tor_p41 , rows_tot_tor_p42 ,
   rows_tot_tor_p43 , rows_tot_tor_p44 , rows_tot_l1_p41 ,
   rows_tot_l1_p42 , rows_tot_l2_p4 )
 rows_overall_cpu: max_rows_cpu = MAX ( rows_tot_cpu1 , rows_tot_cpu2 ,
   rows_tot_cpu3 , rows_tot_cpu4 , rows_tot_cpu5 , rows_tot_cpu6 ,
   rows_tot_cpu7 , rows_tot_cpu8 , rows_tot_cpu9 , rows_tot_cpu10 ,
   rows_tot_cpu11 , rows_tot_cpu12 , rows_tot_cpu13 , rows_tot_cpu14 ,
   rows_tot_cpu15 , rows_tot_cpu16 )
End

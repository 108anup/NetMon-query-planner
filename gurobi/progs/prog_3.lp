\ Model netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0.05
   0 m_access_time_cpu1 + 0 ns_sketch_cpu1 + 0 ns_dpdk_cpu1 + 0 ns_cpu1
   + 0 m_access_time_cpu2 + 0 ns_sketch_cpu2 + 0 ns_dpdk_cpu2 + 0 ns_cpu2
   + 0 m_access_time_cpu3 + 0 ns_sketch_cpu3 + 0 ns_dpdk_cpu3 + 0 ns_cpu3
   + 0 m_access_time_cpu4 + 0 ns_sketch_cpu4 + 0 ns_dpdk_cpu4 + 0 ns_cpu4
   + 0 m_access_time_cpu5 + 0 ns_sketch_cpu5 + 0 ns_dpdk_cpu5 + 0 ns_cpu5
   + 0 m_access_time_cpu6 + 0 ns_sketch_cpu6 + 0 ns_dpdk_cpu6 + 0 ns_cpu6
   + 0 m_access_time_cpu7 + 0 ns_sketch_cpu7 + 0 ns_dpdk_cpu7 + 0 ns_cpu7
   + 0 m_access_time_cpu8 + 0 ns_sketch_cpu8 + 0 ns_dpdk_cpu8 + 0 ns_cpu8
   + ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0.05
   res
Subject To
 cov_0_22: frac[3,22] + frac[7,22] + frac[9,22] + frac[11,22] + frac[12,22]
   + frac[13,22] + frac[14,22] >= 0.8
 cov_1_65: frac[1,65] + frac[5,65] + frac[8,65] + frac[10,65] + frac[12,65]
   + frac[13,65] + frac[14,65] >= 1
 cov_2_61: frac[3,61] + frac[5,61] + frac[9,61] + frac[10,61] + frac[12,61]
   + frac[13,61] + frac[14,61] >= 1
 cov_3_6: frac[5,6] + frac[6,6] + frac[10,6] + frac[11,6] + frac[13,6]
   >= 0.9
 cov_4_12: frac[0,12] + frac[2,12] + frac[8,12] + frac[9,12] + frac[12,12]
   >= 0.7
 cov_5_65: frac[5,65] + frac[6,65] + frac[10,65] + frac[11,65]
   + frac[13,65] >= 0.9
 cov_6_17: frac[2,17] + frac[3,17] + frac[9,17] >= 0.9
 cov_7_7: frac[0,7] + frac[6,7] + frac[8,7] + frac[11,7] + frac[12,7]
   + frac[13,7] + frac[14,7] >= 1
 cov_8_35: frac[3,35] + frac[5,35] + frac[9,35] + frac[10,35] + frac[12,35]
   + frac[13,35] + frac[14,35] >= 0.8
 cov_9_1: frac[4,1] + frac[7,1] + frac[10,1] + frac[11,1] + frac[13,1]
   >= 0.9
 cov_10_62: frac[4,62] + frac[6,62] + frac[10,62] + frac[11,62]
   + frac[13,62] >= 0.9
 cov_11_32: frac[3,32] + frac[7,32] + frac[9,32] + frac[11,32]
   + frac[12,32] + frac[13,32] + frac[14,32] >= 0.9
 cov_12_69: frac[1,69] + frac[3,69] + frac[8,69] + frac[9,69] + frac[12,69]
   >= 0.7
 cov_13_22: frac[0,22] + frac[7,22] + frac[8,22] + frac[11,22]
   + frac[12,22] + frac[13,22] + frac[14,22] >= 1
 cov_14_12: frac[1,12] + frac[6,12] + frac[8,12] + frac[11,12]
   + frac[12,12] + frac[13,12] + frac[14,12] >= 0.8
 cov_15_41: frac[0,41] + frac[4,41] + frac[8,41] + frac[10,41]
   + frac[12,41] + frac[13,41] + frac[14,41] >= 0.7
 cov_16_77: frac[4,77] + frac[7,77] + frac[10,77] + frac[11,77]
   + frac[13,77] >= 0.9
 cov_17_44: frac[2,44] + frac[6,44] + frac[9,44] + frac[11,44]
   + frac[12,44] + frac[13,44] + frac[14,44] >= 0.9
 cov_18_41: frac[1,41] + frac[7,41] + frac[8,41] + frac[11,41]
   + frac[12,41] + frac[13,41] + frac[14,41] >= 0.9
 cov_19_18: frac[6,18] + frac[7,18] + frac[11,18] >= 0.9
 cov_20_46: frac[6,46] + frac[7,46] + frac[11,46] >= 0.7
 cov_21_33: frac[0,33] + frac[7,33] + frac[8,33] + frac[11,33]
   + frac[12,33] + frac[13,33] + frac[14,33] >= 0.7
 cov_22_19: frac[3,19] + frac[6,19] + frac[9,19] + frac[11,19]
   + frac[12,19] + frac[13,19] + frac[14,19] >= 0.9
 cov_23_69: frac[1,69] + frac[3,69] + frac[8,69] + frac[9,69] + frac[12,69]
   >= 0.8
 cov_24_10: frac[2,10] + frac[3,10] + frac[9,10] >= 0.8
 cov_25_26: frac[0,26] + frac[1,26] + frac[8,26] >= 0.7
 cov_26_52: frac[2,52] + frac[6,52] + frac[9,52] + frac[11,52]
   + frac[12,52] + frac[13,52] + frac[14,52] >= 1
 cov_27_13: frac[2,13] + frac[7,13] + frac[9,13] + frac[11,13]
   + frac[12,13] + frac[13,13] + frac[14,13] >= 0.8
 cov_28_73: frac[0,73] + frac[3,73] + frac[8,73] + frac[9,73] + frac[12,73]
   >= 0.8
 cov_29_64: frac[2,64] + frac[7,64] + frac[9,64] + frac[11,64]
   + frac[12,64] + frac[13,64] + frac[14,64] >= 1
 cov_30_25: frac[0,25] + frac[6,25] + frac[8,25] + frac[11,25]
   + frac[12,25] + frac[13,25] + frac[14,25] >= 0.9
 cov_31_59: frac[0,59] + frac[7,59] + frac[8,59] + frac[11,59]
   + frac[12,59] + frac[13,59] + frac[14,59] >= 0.9
 cov_32_79: frac[4,79] + frac[5,79] + frac[10,79] >= 1
 cov_33_76: frac[1,76] + frac[2,76] + frac[8,76] + frac[9,76] + frac[12,76]
   >= 1
 cov_34_70: frac[0,70] + frac[4,70] + frac[8,70] + frac[10,70]
   + frac[12,70] + frac[13,70] + frac[14,70] >= 0.8
 cov_35_73: frac[1,73] + frac[7,73] + frac[8,73] + frac[11,73]
   + frac[12,73] + frac[13,73] + frac[14,73] >= 0.8
 cov_36_60: frac[4,60] + frac[7,60] + frac[10,60] + frac[11,60]
   + frac[13,60] >= 0.7
 cov_37_75: frac[4,75] + frac[5,75] + frac[10,75] >= 0.9
 cov_38_71: frac[1,71] + frac[6,71] + frac[8,71] + frac[11,71]
   + frac[12,71] + frac[13,71] + frac[14,71] >= 1
 cov_39_61: frac[3,61] + frac[6,61] + frac[9,61] + frac[11,61]
   + frac[12,61] + frac[13,61] + frac[14,61] >= 1
 cov_40_25: frac[4,25] + frac[7,25] + frac[10,25] + frac[11,25]
   + frac[13,25] >= 0.7
 cov_41_12: frac[2,12] + frac[6,12] + frac[9,12] + frac[11,12]
   + frac[12,12] + frac[13,12] + frac[14,12] >= 1
 cov_42_8: frac[2,8] + frac[5,8] + frac[9,8] + frac[10,8] + frac[12,8]
   + frac[13,8] + frac[14,8] >= 1
 cov_43_63: frac[0,63] + frac[5,63] + frac[8,63] + frac[10,63]
   + frac[12,63] + frac[13,63] + frac[14,63] >= 0.8
 cov_44_18: frac[0,18] + frac[2,18] + frac[8,18] + frac[9,18] + frac[12,18]
   >= 0.7
 cov_45_58: frac[0,58] + frac[2,58] + frac[8,58] + frac[9,58] + frac[12,58]
   >= 0.8
 cov_46_23: frac[5,23] + frac[6,23] + frac[10,23] + frac[11,23]
   + frac[13,23] >= 1
 cov_47_51: frac[4,51] + frac[5,51] + frac[10,51] >= 0.7
 cov_48_57: frac[1,57] + frac[6,57] + frac[8,57] + frac[11,57]
   + frac[12,57] + frac[13,57] + frac[14,57] >= 0.7
 cov_49_34: frac[1,34] + frac[4,34] + frac[8,34] + frac[10,34]
   + frac[12,34] + frac[13,34] + frac[14,34] >= 0.7
 cov_50_70: frac[2,70] + frac[6,70] + frac[9,70] + frac[11,70]
   + frac[12,70] + frac[13,70] + frac[14,70] >= 1
 cov_51_59: frac[5,59] + frac[6,59] + frac[10,59] + frac[11,59]
   + frac[13,59] >= 0.8
 cov_52_37: frac[0,37] + frac[5,37] + frac[8,37] + frac[10,37]
   + frac[12,37] + frac[13,37] + frac[14,37] >= 0.7
 cov_53_71: frac[3,71] + frac[4,71] + frac[9,71] + frac[10,71]
   + frac[12,71] + frac[13,71] + frac[14,71] >= 0.7
 cov_54_21: frac[2,21] + frac[3,21] + frac[9,21] >= 0.7
 cov_55_73: frac[5,73] + frac[7,73] + frac[10,73] + frac[11,73]
   + frac[13,73] >= 0.7
 cov_56_22: frac[5,22] + frac[6,22] + frac[10,22] + frac[11,22]
   + frac[13,22] >= 0.9
 cov_57_37: frac[0,37] + frac[7,37] + frac[8,37] + frac[11,37]
   + frac[12,37] + frac[13,37] + frac[14,37] >= 1
 cov_58_61: frac[3,61] + frac[4,61] + frac[9,61] + frac[10,61]
   + frac[12,61] + frac[13,61] + frac[14,61] >= 0.8
 cov_59_18: frac[6,18] + frac[7,18] + frac[11,18] >= 0.7
 cov_60_21: frac[1,21] + frac[7,21] + frac[8,21] + frac[11,21]
   + frac[12,21] + frac[13,21] + frac[14,21] >= 1
 cov_61_51: frac[0,51] + frac[5,51] + frac[8,51] + frac[10,51]
   + frac[12,51] + frac[13,51] + frac[14,51] >= 0.8
 cov_62_19: frac[0,19] + frac[4,19] + frac[8,19] + frac[10,19]
   + frac[12,19] + frac[13,19] + frac[14,19] >= 0.8
 cov_63_47: frac[4,47] + frac[6,47] + frac[10,47] + frac[11,47]
   + frac[13,47] >= 0.7
 cov_64_53: frac[3,53] + frac[7,53] + frac[9,53] + frac[11,53]
   + frac[12,53] + frac[13,53] + frac[14,53] >= 0.8
 cov_65_1: frac[2,1] + frac[7,1] + frac[9,1] + frac[11,1] + frac[12,1]
   + frac[13,1] + frac[14,1] >= 1
 cov_66_59: frac[2,59] + frac[3,59] + frac[9,59] >= 1
 cov_67_77: frac[2,77] + frac[5,77] + frac[9,77] + frac[10,77]
   + frac[12,77] + frac[13,77] + frac[14,77] >= 1
 cov_68_38: frac[0,38] + frac[2,38] + frac[8,38] + frac[9,38] + frac[12,38]
   >= 0.7
 cov_69_13: frac[4,13] + frac[6,13] + frac[10,13] + frac[11,13]
   + frac[13,13] >= 0.8
 cov_70_30: frac[4,30] + frac[5,30] + frac[10,30] >= 0.9
 cov_71_0: frac[1,0] + frac[3,0] + frac[8,0] + frac[9,0] + frac[12,0] >= 1
 cov_72_25: frac[1,25] + frac[6,25] + frac[8,25] + frac[11,25]
   + frac[12,25] + frac[13,25] + frac[14,25] >= 1
 cov_73_24: frac[0,24] + frac[4,24] + frac[8,24] + frac[10,24]
   + frac[12,24] + frac[13,24] + frac[14,24] >= 0.9
 cov_74_2: frac[3,2] + frac[4,2] + frac[9,2] + frac[10,2] + frac[12,2]
   + frac[13,2] + frac[14,2] >= 0.7
 cov_75_77: frac[1,77] + frac[3,77] + frac[8,77] + frac[9,77] + frac[12,77]
   >= 0.9
 cov_76_20: frac[0,20] + frac[2,20] + frac[8,20] + frac[9,20] + frac[12,20]
   >= 0.8
 cov_77_49: frac[2,49] + frac[7,49] + frac[9,49] + frac[11,49]
   + frac[12,49] + frac[13,49] + frac[14,49] >= 0.8
 cov_78_37: frac[5,37] + frac[6,37] + frac[10,37] + frac[11,37]
   + frac[13,37] >= 1
 cov_79_61: frac[0,61] + frac[5,61] + frac[8,61] + frac[10,61]
   + frac[12,61] + frac[13,61] + frac[14,61] >= 1
 cov_80_25: frac[3,25] + frac[4,25] + frac[9,25] + frac[10,25]
   + frac[12,25] + frac[13,25] + frac[14,25] >= 0.9
 cov_81_57: frac[1,57] + frac[5,57] + frac[8,57] + frac[10,57]
   + frac[12,57] + frac[13,57] + frac[14,57] >= 1
 cov_82_42: frac[3,42] + frac[7,42] + frac[9,42] + frac[11,42]
   + frac[12,42] + frac[13,42] + frac[14,42] >= 0.7
 cov_83_37: frac[4,37] + frac[6,37] + frac[10,37] + frac[11,37]
   + frac[13,37] >= 0.8
 cov_84_59: frac[1,59] + frac[5,59] + frac[8,59] + frac[10,59]
   + frac[12,59] + frac[13,59] + frac[14,59] >= 0.7
 cov_85_56: frac[0,56] + frac[2,56] + frac[8,56] + frac[9,56] + frac[12,56]
   >= 1
 cov_86_27: frac[3,27] + frac[4,27] + frac[9,27] + frac[10,27]
   + frac[12,27] + frac[13,27] + frac[14,27] >= 0.8
 cov_87_70: frac[3,70] + frac[7,70] + frac[9,70] + frac[11,70]
   + frac[12,70] + frac[13,70] + frac[14,70] >= 0.9
 cov_88_78: frac[0,78] + frac[2,78] + frac[8,78] + frac[9,78] + frac[12,78]
   >= 1
 cov_89_11: frac[2,11] + frac[5,11] + frac[9,11] + frac[10,11]
   + frac[12,11] + frac[13,11] + frac[14,11] >= 0.9
 cov_90_61: frac[1,61] + frac[7,61] + frac[8,61] + frac[11,61]
   + frac[12,61] + frac[13,61] + frac[14,61] >= 0.8
 cov_91_38: frac[1,38] + frac[2,38] + frac[8,38] + frac[9,38] + frac[12,38]
   >= 0.9
 cov_92_31: frac[6,31] + frac[7,31] + frac[11,31] >= 0.7
 cov_93_29: frac[3,29] + frac[6,29] + frac[9,29] + frac[11,29]
   + frac[12,29] + frac[13,29] + frac[14,29] >= 0.7
 cov_94_57: frac[4,57] + frac[7,57] + frac[10,57] + frac[11,57]
   + frac[13,57] >= 0.8
 cov_95_13: frac[4,13] + frac[5,13] + frac[10,13] >= 0.9
 cov_96_15: frac[2,15] + frac[7,15] + frac[9,15] + frac[11,15]
   + frac[12,15] + frac[13,15] + frac[14,15] >= 1
 cov_97_53: frac[1,53] + frac[4,53] + frac[8,53] + frac[10,53]
   + frac[12,53] + frac[13,53] + frac[14,53] >= 0.7
 cov_98_20: frac[2,20] + frac[4,20] + frac[9,20] + frac[10,20]
   + frac[12,20] + frac[13,20] + frac[14,20] >= 0.7
 cov_99_49: frac[5,49] + frac[6,49] + frac[10,49] + frac[11,49]
   + frac[13,49] >= 1
 cov_100_24: frac[1,24] + frac[2,24] + frac[8,24] + frac[9,24]
   + frac[12,24] >= 1
 cov_101_16: frac[2,16] + frac[3,16] + frac[9,16] >= 0.7
 cov_102_55: frac[6,55] + frac[7,55] + frac[11,55] >= 0.7
 cov_103_4: frac[1,4] + frac[2,4] + frac[8,4] + frac[9,4] + frac[12,4]
   >= 0.9
 cov_104_47: frac[5,47] + frac[7,47] + frac[10,47] + frac[11,47]
   + frac[13,47] >= 0.9
 cov_105_55: frac[2,55] + frac[5,55] + frac[9,55] + frac[10,55]
   + frac[12,55] + frac[13,55] + frac[14,55] >= 0.7
 cov_106_27: frac[1,27] + frac[5,27] + frac[8,27] + frac[10,27]
   + frac[12,27] + frac[13,27] + frac[14,27] >= 0.7
 cov_107_32: frac[1,32] + frac[7,32] + frac[8,32] + frac[11,32]
   + frac[12,32] + frac[13,32] + frac[14,32] >= 0.8
 cov_108_70: frac[1,70] + frac[2,70] + frac[8,70] + frac[9,70]
   + frac[12,70] >= 0.8
 cov_109_31: frac[1,31] + frac[6,31] + frac[8,31] + frac[11,31]
   + frac[12,31] + frac[13,31] + frac[14,31] >= 0.9
 cov_110_53: frac[0,53] + frac[1,53] + frac[8,53] >= 1
 cov_111_48: frac[0,48] + frac[6,48] + frac[8,48] + frac[11,48]
   + frac[12,48] + frac[13,48] + frac[14,48] >= 1
 cov_112_43: frac[3,43] + frac[5,43] + frac[9,43] + frac[10,43]
   + frac[12,43] + frac[13,43] + frac[14,43] >= 0.8
 cov_113_58: frac[1,58] + frac[5,58] + frac[8,58] + frac[10,58]
   + frac[12,58] + frac[13,58] + frac[14,58] >= 0.7
 cov_114_21: frac[3,21] + frac[6,21] + frac[9,21] + frac[11,21]
   + frac[12,21] + frac[13,21] + frac[14,21] >= 0.9
 cov_115_26: frac[1,26] + frac[7,26] + frac[8,26] + frac[11,26]
   + frac[12,26] + frac[13,26] + frac[14,26] >= 0.8
 cov_116_79: frac[6,79] + frac[7,79] + frac[11,79] >= 0.9
 cov_117_64: frac[0,64] + frac[6,64] + frac[8,64] + frac[11,64]
   + frac[12,64] + frac[13,64] + frac[14,64] >= 0.7
 cov_118_72: frac[5,72] + frac[6,72] + frac[10,72] + frac[11,72]
   + frac[13,72] >= 1
 cov_119_44: frac[2,44] + frac[6,44] + frac[9,44] + frac[11,44]
   + frac[12,44] + frac[13,44] + frac[14,44] >= 0.7
 cov_120_40: frac[5,40] + frac[6,40] + frac[10,40] + frac[11,40]
   + frac[13,40] >= 0.7
 cov_121_67: frac[5,67] + frac[7,67] + frac[10,67] + frac[11,67]
   + frac[13,67] >= 0.7
 cov_122_38: frac[0,38] + frac[3,38] + frac[8,38] + frac[9,38]
   + frac[12,38] >= 0.8
 cov_123_27: frac[5,27] + frac[6,27] + frac[10,27] + frac[11,27]
   + frac[13,27] >= 0.7
 cov_124_3: frac[6,3] + frac[7,3] + frac[11,3] >= 0.9
 cov_125_17: frac[3,17] + frac[6,17] + frac[9,17] + frac[11,17]
   + frac[12,17] + frac[13,17] + frac[14,17] >= 1
 cov_126_43: frac[4,43] + frac[6,43] + frac[10,43] + frac[11,43]
   + frac[13,43] >= 1
 cov_127_36: frac[2,36] + frac[7,36] + frac[9,36] + frac[11,36]
   + frac[12,36] + frac[13,36] + frac[14,36] >= 0.8
 cov_128_78: frac[5,78] + frac[6,78] + frac[10,78] + frac[11,78]
   + frac[13,78] >= 0.7
 cov_129_7: frac[3,7] + frac[7,7] + frac[9,7] + frac[11,7] + frac[12,7]
   + frac[13,7] + frac[14,7] >= 0.8
 cov_130_59: frac[3,59] + frac[7,59] + frac[9,59] + frac[11,59]
   + frac[12,59] + frac[13,59] + frac[14,59] >= 0.7
 cov_131_19: frac[0,19] + frac[2,19] + frac[8,19] + frac[9,19]
   + frac[12,19] >= 1
 cov_132_31: frac[4,31] + frac[7,31] + frac[10,31] + frac[11,31]
   + frac[13,31] >= 0.9
 cov_133_69: frac[1,69] + frac[2,69] + frac[8,69] + frac[9,69]
   + frac[12,69] >= 0.9
 cov_134_7: frac[2,7] + frac[3,7] + frac[9,7] >= 0.7
 cov_135_63: frac[5,63] + frac[7,63] + frac[10,63] + frac[11,63]
   + frac[13,63] >= 0.8
 cov_136_13: frac[1,13] + frac[4,13] + frac[8,13] + frac[10,13]
   + frac[12,13] + frac[13,13] + frac[14,13] >= 0.9
 cov_137_31: frac[3,31] + frac[4,31] + frac[9,31] + frac[10,31]
   + frac[12,31] + frac[13,31] + frac[14,31] >= 0.7
 cov_138_1: frac[0,1] + frac[2,1] + frac[8,1] + frac[9,1] + frac[12,1]
   >= 0.9
 cov_139_50: frac[1,50] + frac[5,50] + frac[8,50] + frac[10,50]
   + frac[12,50] + frac[13,50] + frac[14,50] >= 0.8
 cov_140_78: frac[3,78] + frac[7,78] + frac[9,78] + frac[11,78]
   + frac[12,78] + frac[13,78] + frac[14,78] >= 0.9
 cov_141_45: frac[0,45] + frac[2,45] + frac[8,45] + frac[9,45]
   + frac[12,45] >= 0.9
 cov_142_42: frac[5,42] + frac[7,42] + frac[10,42] + frac[11,42]
   + frac[13,42] >= 0.7
 cov_143_29: frac[0,29] + frac[6,29] + frac[8,29] + frac[11,29]
   + frac[12,29] + frac[13,29] + frac[14,29] >= 0.8
 cov_144_48: frac[4,48] + frac[7,48] + frac[10,48] + frac[11,48]
   + frac[13,48] >= 0.8
 cov_145_8: frac[5,8] + frac[7,8] + frac[10,8] + frac[11,8] + frac[13,8]
   >= 0.9
 cov_146_53: frac[2,53] + frac[6,53] + frac[9,53] + frac[11,53]
   + frac[12,53] + frac[13,53] + frac[14,53] >= 0.9
 cov_147_2: frac[1,2] + frac[7,2] + frac[8,2] + frac[11,2] + frac[12,2]
   + frac[13,2] + frac[14,2] >= 0.8
 cov_148_19: frac[0,19] + frac[7,19] + frac[8,19] + frac[11,19]
   + frac[12,19] + frac[13,19] + frac[14,19] >= 0.8
 cov_149_31: frac[2,31] + frac[7,31] + frac[9,31] + frac[11,31]
   + frac[12,31] + frac[13,31] + frac[14,31] >= 1
 cov_150_49: frac[3,49] + frac[5,49] + frac[9,49] + frac[10,49]
   + frac[12,49] + frac[13,49] + frac[14,49] >= 0.8
 cov_151_64: frac[6,64] + frac[7,64] + frac[11,64] >= 0.9
 cov_152_50: frac[3,50] + frac[7,50] + frac[9,50] + frac[11,50]
   + frac[12,50] + frac[13,50] + frac[14,50] >= 0.9
 cov_153_62: frac[4,62] + frac[7,62] + frac[10,62] + frac[11,62]
   + frac[13,62] >= 0.8
 cov_154_57: frac[0,57] + frac[1,57] + frac[8,57] >= 0.8
 cov_155_72: frac[2,72] + frac[3,72] + frac[9,72] >= 0.7
 cov_156_26: frac[2,26] + frac[4,26] + frac[9,26] + frac[10,26]
   + frac[12,26] + frac[13,26] + frac[14,26] >= 0.9
 cov_157_73: frac[5,73] + frac[6,73] + frac[10,73] + frac[11,73]
   + frac[13,73] >= 1
 cov_158_45: frac[0,45] + frac[5,45] + frac[8,45] + frac[10,45]
   + frac[12,45] + frac[13,45] + frac[14,45] >= 0.8
 cov_159_27: frac[5,27] + frac[7,27] + frac[10,27] + frac[11,27]
   + frac[13,27] >= 1
 cov_160_54: frac[1,54] + frac[4,54] + frac[8,54] + frac[10,54]
   + frac[12,54] + frac[13,54] + frac[14,54] >= 0.9
 cov_161_76: frac[0,76] + frac[2,76] + frac[8,76] + frac[9,76]
   + frac[12,76] >= 1
 cov_162_56: frac[0,56] + frac[4,56] + frac[8,56] + frac[10,56]
   + frac[12,56] + frac[13,56] + frac[14,56] >= 1
 cov_163_49: frac[4,49] + frac[7,49] + frac[10,49] + frac[11,49]
   + frac[13,49] >= 1
 cov_164_48: frac[3,48] + frac[6,48] + frac[9,48] + frac[11,48]
   + frac[12,48] + frac[13,48] + frac[14,48] >= 1
 cov_165_4: frac[1,4] + frac[3,4] + frac[8,4] + frac[9,4] + frac[12,4]
   >= 0.7
 cov_166_8: frac[1,8] + frac[6,8] + frac[8,8] + frac[11,8] + frac[12,8]
   + frac[13,8] + frac[14,8] >= 0.9
 cov_167_38: frac[2,38] + frac[6,38] + frac[9,38] + frac[11,38]
   + frac[12,38] + frac[13,38] + frac[14,38] >= 0.9
 cov_168_59: frac[3,59] + frac[4,59] + frac[9,59] + frac[10,59]
   + frac[12,59] + frac[13,59] + frac[14,59] >= 0.8
 cov_169_72: frac[1,72] + frac[2,72] + frac[8,72] + frac[9,72]
   + frac[12,72] >= 0.9
 cov_170_27: frac[3,27] + frac[6,27] + frac[9,27] + frac[11,27]
   + frac[12,27] + frac[13,27] + frac[14,27] >= 0.9
 cov_171_54: frac[0,54] + frac[2,54] + frac[8,54] + frac[9,54]
   + frac[12,54] >= 1
 cov_172_69: frac[1,69] + frac[7,69] + frac[8,69] + frac[11,69]
   + frac[12,69] + frac[13,69] + frac[14,69] >= 0.8
 cov_173_58: frac[3,58] + frac[7,58] + frac[9,58] + frac[11,58]
   + frac[12,58] + frac[13,58] + frac[14,58] >= 0.8
 cov_174_24: frac[6,24] + frac[7,24] + frac[11,24] >= 1
 cov_175_12: frac[0,12] + frac[7,12] + frac[8,12] + frac[11,12]
   + frac[12,12] + frac[13,12] + frac[14,12] >= 0.8
 cov_176_16: frac[1,16] + frac[6,16] + frac[8,16] + frac[11,16]
   + frac[12,16] + frac[13,16] + frac[14,16] >= 0.7
 cov_177_49: frac[0,49] + frac[1,49] + frac[8,49] >= 0.7
 cov_178_28: frac[2,28] + frac[3,28] + frac[9,28] >= 0.9
 cov_179_38: frac[0,38] + frac[7,38] + frac[8,38] + frac[11,38]
   + frac[12,38] + frac[13,38] + frac[14,38] >= 0.8
 cov_180_41: frac[2,41] + frac[4,41] + frac[9,41] + frac[10,41]
   + frac[12,41] + frac[13,41] + frac[14,41] >= 0.8
 cov_181_70: frac[0,70] + frac[1,70] + frac[8,70] >= 0.8
 cov_182_72: frac[6,72] + frac[7,72] + frac[11,72] >= 0.7
 cov_183_10: frac[2,10] + frac[5,10] + frac[9,10] + frac[10,10]
   + frac[12,10] + frac[13,10] + frac[14,10] >= 0.8
 cov_184_65: frac[2,65] + frac[3,65] + frac[9,65] >= 0.9
 cov_185_46: frac[1,46] + frac[4,46] + frac[8,46] + frac[10,46]
   + frac[12,46] + frac[13,46] + frac[14,46] >= 0.9
 cov_186_46: frac[1,46] + frac[2,46] + frac[8,46] + frac[9,46]
   + frac[12,46] >= 0.7
 cov_187_73: frac[1,73] + frac[6,73] + frac[8,73] + frac[11,73]
   + frac[12,73] + frac[13,73] + frac[14,73] >= 0.8
 cov_188_43: frac[4,43] + frac[7,43] + frac[10,43] + frac[11,43]
   + frac[13,43] >= 0.9
 cov_189_10: frac[1,10] + frac[5,10] + frac[8,10] + frac[10,10]
   + frac[12,10] + frac[13,10] + frac[14,10] >= 0.9
 cov_190_60: frac[0,60] + frac[6,60] + frac[8,60] + frac[11,60]
   + frac[12,60] + frac[13,60] + frac[14,60] >= 0.7
 cov_191_4: frac[2,4] + frac[7,4] + frac[9,4] + frac[11,4] + frac[12,4]
   + frac[13,4] + frac[14,4] >= 0.9
 cov_192_9: frac[2,9] + frac[3,9] + frac[9,9] >= 0.8
 cov_193_65: frac[3,65] + frac[5,65] + frac[9,65] + frac[10,65]
   + frac[12,65] + frac[13,65] + frac[14,65] >= 1
 cov_194_73: frac[3,73] + frac[5,73] + frac[9,73] + frac[10,73]
   + frac[12,73] + frac[13,73] + frac[14,73] >= 0.9
 cov_195_18: frac[3,18] + frac[6,18] + frac[9,18] + frac[11,18]
   + frac[12,18] + frac[13,18] + frac[14,18] >= 0.8
 cov_196_30: frac[5,30] + frac[7,30] + frac[10,30] + frac[11,30]
   + frac[13,30] >= 0.9
 cov_197_61: frac[3,61] + frac[5,61] + frac[9,61] + frac[10,61]
   + frac[12,61] + frac[13,61] + frac[14,61] >= 0.8
 cov_198_1: frac[4,1] + frac[5,1] + frac[10,1] >= 0.7
 cov_199_59: frac[3,59] + frac[7,59] + frac[9,59] + frac[11,59]
   + frac[12,59] + frac[13,59] + frac[14,59] >= 0.7
 cov_200_31: frac[3,31] + frac[5,31] + frac[9,31] + frac[10,31]
   + frac[12,31] + frac[13,31] + frac[14,31] >= 0.8
 cov_201_8: frac[5,8] + frac[6,8] + frac[10,8] + frac[11,8] + frac[13,8]
   >= 0.9
 cov_202_58: frac[5,58] + frac[6,58] + frac[10,58] + frac[11,58]
   + frac[13,58] >= 0.8
 cov_203_2: frac[2,2] + frac[5,2] + frac[9,2] + frac[10,2] + frac[12,2]
   + frac[13,2] + frac[14,2] >= 0.9
 cov_204_74: frac[0,74] + frac[3,74] + frac[8,74] + frac[9,74]
   + frac[12,74] >= 0.8
 cov_205_4: frac[1,4] + frac[3,4] + frac[8,4] + frac[9,4] + frac[12,4]
   >= 0.8
 cov_206_50: frac[5,50] + frac[6,50] + frac[10,50] + frac[11,50]
   + frac[13,50] >= 0.9
 cov_207_29: frac[2,29] + frac[6,29] + frac[9,29] + frac[11,29]
   + frac[12,29] + frac[13,29] + frac[14,29] >= 0.8
 cov_208_33: frac[1,33] + frac[3,33] + frac[8,33] + frac[9,33]
   + frac[12,33] >= 0.9
 cov_209_48: frac[3,48] + frac[5,48] + frac[9,48] + frac[10,48]
   + frac[12,48] + frac[13,48] + frac[14,48] >= 0.8
 cov_210_36: frac[6,36] + frac[7,36] + frac[11,36] >= 0.8
 cov_211_21: frac[1,21] + frac[6,21] + frac[8,21] + frac[11,21]
   + frac[12,21] + frac[13,21] + frac[14,21] >= 0.7
 cov_212_66: frac[4,66] + frac[5,66] + frac[10,66] >= 0.7
 cov_213_73: frac[4,73] + frac[5,73] + frac[10,73] >= 0.8
 cov_214_44: frac[2,44] + frac[7,44] + frac[9,44] + frac[11,44]
   + frac[12,44] + frac[13,44] + frac[14,44] >= 0.8
 cov_215_65: frac[0,65] + frac[2,65] + frac[8,65] + frac[9,65]
   + frac[12,65] >= 0.7
 cov_216_28: frac[1,28] + frac[2,28] + frac[8,28] + frac[9,28]
   + frac[12,28] >= 1
 cov_217_2: frac[1,2] + frac[5,2] + frac[8,2] + frac[10,2] + frac[12,2]
   + frac[13,2] + frac[14,2] >= 0.7
 cov_218_33: frac[1,33] + frac[7,33] + frac[8,33] + frac[11,33]
   + frac[12,33] + frac[13,33] + frac[14,33] >= 0.7
 cov_219_36: frac[0,36] + frac[1,36] + frac[8,36] >= 1
 cov_220_49: frac[1,49] + frac[6,49] + frac[8,49] + frac[11,49]
   + frac[12,49] + frac[13,49] + frac[14,49] >= 0.9
 cov_221_0: frac[3,0] + frac[5,0] + frac[9,0] + frac[10,0] + frac[12,0]
   + frac[13,0] + frac[14,0] >= 0.7
 cov_222_20: frac[0,20] + frac[2,20] + frac[8,20] + frac[9,20]
   + frac[12,20] >= 0.7
 cov_223_66: frac[4,66] + frac[5,66] + frac[10,66] >= 0.9
 cov_224_26: frac[1,26] + frac[2,26] + frac[8,26] + frac[9,26]
   + frac[12,26] >= 1
 cov_225_32: frac[0,32] + frac[2,32] + frac[8,32] + frac[9,32]
   + frac[12,32] >= 1
 cov_226_57: frac[3,57] + frac[7,57] + frac[9,57] + frac[11,57]
   + frac[12,57] + frac[13,57] + frac[14,57] >= 0.7
 cov_227_79: frac[2,79] + frac[4,79] + frac[9,79] + frac[10,79]
   + frac[12,79] + frac[13,79] + frac[14,79] >= 1
 cov_228_58: frac[3,58] + frac[6,58] + frac[9,58] + frac[11,58]
   + frac[12,58] + frac[13,58] + frac[14,58] >= 0.9
 cov_229_57: frac[0,57] + frac[5,57] + frac[8,57] + frac[10,57]
   + frac[12,57] + frac[13,57] + frac[14,57] >= 0.9
 cov_230_77: frac[2,77] + frac[7,77] + frac[9,77] + frac[11,77]
   + frac[12,77] + frac[13,77] + frac[14,77] >= 0.7
 cov_231_46: frac[1,46] + frac[7,46] + frac[8,46] + frac[11,46]
   + frac[12,46] + frac[13,46] + frac[14,46] >= 1
 cov_232_66: frac[2,66] + frac[6,66] + frac[9,66] + frac[11,66]
   + frac[12,66] + frac[13,66] + frac[14,66] >= 1
 cov_233_26: frac[2,26] + frac[4,26] + frac[9,26] + frac[10,26]
   + frac[12,26] + frac[13,26] + frac[14,26] >= 0.8
 cov_234_59: frac[4,59] + frac[6,59] + frac[10,59] + frac[11,59]
   + frac[13,59] >= 0.7
 cov_235_28: frac[2,28] + frac[6,28] + frac[9,28] + frac[11,28]
   + frac[12,28] + frac[13,28] + frac[14,28] >= 1
 cov_236_70: frac[4,70] + frac[6,70] + frac[10,70] + frac[11,70]
   + frac[13,70] >= 0.8
 cov_237_26: frac[1,26] + frac[7,26] + frac[8,26] + frac[11,26]
   + frac[12,26] + frac[13,26] + frac[14,26] >= 0.8
 cov_238_25: frac[0,25] + frac[7,25] + frac[8,25] + frac[11,25]
   + frac[12,25] + frac[13,25] + frac[14,25] >= 1
 cov_239_65: frac[0,65] + frac[6,65] + frac[8,65] + frac[11,65]
   + frac[12,65] + frac[13,65] + frac[14,65] >= 0.7
 cov_240_53: frac[0,53] + frac[1,53] + frac[8,53] >= 0.9
 cov_241_40: frac[6,40] + frac[7,40] + frac[11,40] >= 1
 cov_242_61: frac[4,61] + frac[5,61] + frac[10,61] >= 0.9
 cov_243_32: frac[1,32] + frac[6,32] + frac[8,32] + frac[11,32]
   + frac[12,32] + frac[13,32] + frac[14,32] >= 0.9
 cov_244_17: frac[2,17] + frac[4,17] + frac[9,17] + frac[10,17]
   + frac[12,17] + frac[13,17] + frac[14,17] >= 0.7
 cov_245_47: frac[1,47] + frac[6,47] + frac[8,47] + frac[11,47]
   + frac[12,47] + frac[13,47] + frac[14,47] >= 0.8
 cov_246_46: frac[6,46] + frac[7,46] + frac[11,46] >= 0.8
 cov_247_58: frac[1,58] + frac[5,58] + frac[8,58] + frac[10,58]
   + frac[12,58] + frac[13,58] + frac[14,58] >= 0.7
 cov_248_58: frac[0,58] + frac[4,58] + frac[8,58] + frac[10,58]
   + frac[12,58] + frac[13,58] + frac[14,58] >= 0.9
 cov_249_19: frac[1,19] + frac[5,19] + frac[8,19] + frac[10,19]
   + frac[12,19] + frac[13,19] + frac[14,19] >= 0.9
 cov_250_14: frac[5,14] + frac[7,14] + frac[10,14] + frac[11,14]
   + frac[13,14] >= 1
 cov_251_60: frac[0,60] + frac[2,60] + frac[8,60] + frac[9,60]
   + frac[12,60] >= 1
 cov_252_28: frac[1,28] + frac[7,28] + frac[8,28] + frac[11,28]
   + frac[12,28] + frac[13,28] + frac[14,28] >= 0.8
 cov_253_36: frac[4,36] + frac[7,36] + frac[10,36] + frac[11,36]
   + frac[13,36] >= 0.8
 cov_254_47: frac[1,47] + frac[5,47] + frac[8,47] + frac[10,47]
   + frac[12,47] + frac[13,47] + frac[14,47] >= 1
 cov_255_59: frac[3,59] + frac[6,59] + frac[9,59] + frac[11,59]
   + frac[12,59] + frac[13,59] + frac[14,59] >= 1
 cov_256_32: frac[6,32] + frac[7,32] + frac[11,32] >= 0.9
 cov_257_57: frac[3,57] + frac[5,57] + frac[9,57] + frac[10,57]
   + frac[12,57] + frac[13,57] + frac[14,57] >= 1
 cov_258_4: frac[0,4] + frac[4,4] + frac[8,4] + frac[10,4] + frac[12,4]
   + frac[13,4] + frac[14,4] >= 0.8
 cov_259_18: frac[0,18] + frac[4,18] + frac[8,18] + frac[10,18]
   + frac[12,18] + frac[13,18] + frac[14,18] >= 0.8
 cov_260_78: frac[0,78] + frac[1,78] + frac[8,78] >= 0.8
 cov_261_19: frac[3,19] + frac[7,19] + frac[9,19] + frac[11,19]
   + frac[12,19] + frac[13,19] + frac[14,19] >= 1
 cov_262_40: frac[4,40] + frac[5,40] + frac[10,40] >= 0.9
 cov_263_76: frac[0,76] + frac[1,76] + frac[8,76] >= 1
 cov_264_65: frac[4,65] + frac[7,65] + frac[10,65] + frac[11,65]
   + frac[13,65] >= 1
 cov_265_4: frac[2,4] + frac[6,4] + frac[9,4] + frac[11,4] + frac[12,4]
   + frac[13,4] + frac[14,4] >= 0.9
 cov_266_69: frac[3,69] + frac[7,69] + frac[9,69] + frac[11,69]
   + frac[12,69] + frac[13,69] + frac[14,69] >= 0.9
 cov_267_13: frac[6,13] + frac[7,13] + frac[11,13] >= 0.7
 cov_268_21: frac[3,21] + frac[5,21] + frac[9,21] + frac[10,21]
   + frac[12,21] + frac[13,21] + frac[14,21] >= 0.8
 cov_269_50: frac[4,50] + frac[5,50] + frac[10,50] >= 0.8
 cov_270_13: frac[1,13] + frac[7,13] + frac[8,13] + frac[11,13]
   + frac[12,13] + frac[13,13] + frac[14,13] >= 0.7
 cov_271_10: frac[2,10] + frac[5,10] + frac[9,10] + frac[10,10]
   + frac[12,10] + frac[13,10] + frac[14,10] >= 0.8
 cov_272_63: frac[0,63] + frac[4,63] + frac[8,63] + frac[10,63]
   + frac[12,63] + frac[13,63] + frac[14,63] >= 0.7
 cov_273_33: frac[0,33] + frac[2,33] + frac[8,33] + frac[9,33]
   + frac[12,33] >= 0.9
 cov_274_56: frac[0,56] + frac[2,56] + frac[8,56] + frac[9,56]
   + frac[12,56] >= 1
 cov_275_78: frac[5,78] + frac[6,78] + frac[10,78] + frac[11,78]
   + frac[13,78] >= 0.7
 cov_276_72: frac[1,72] + frac[2,72] + frac[8,72] + frac[9,72]
   + frac[12,72] >= 0.7
 cov_277_57: frac[3,57] + frac[6,57] + frac[9,57] + frac[11,57]
   + frac[12,57] + frac[13,57] + frac[14,57] >= 0.9
 cov_278_66: frac[2,66] + frac[7,66] + frac[9,66] + frac[11,66]
   + frac[12,66] + frac[13,66] + frac[14,66] >= 0.8
 cov_279_69: frac[2,69] + frac[7,69] + frac[9,69] + frac[11,69]
   + frac[12,69] + frac[13,69] + frac[14,69] >= 0.9
 cov_280_20: frac[0,20] + frac[6,20] + frac[8,20] + frac[11,20]
   + frac[12,20] + frac[13,20] + frac[14,20] >= 0.7
 cov_281_67: frac[0,67] + frac[7,67] + frac[8,67] + frac[11,67]
   + frac[12,67] + frac[13,67] + frac[14,67] >= 0.8
 cov_282_14: frac[2,14] + frac[7,14] + frac[9,14] + frac[11,14]
   + frac[12,14] + frac[13,14] + frac[14,14] >= 0.7
 cov_283_60: frac[0,60] + frac[7,60] + frac[8,60] + frac[11,60]
   + frac[12,60] + frac[13,60] + frac[14,60] >= 1
 cov_284_13: frac[1,13] + frac[5,13] + frac[8,13] + frac[10,13]
   + frac[12,13] + frac[13,13] + frac[14,13] >= 1
 cov_285_14: frac[5,14] + frac[7,14] + frac[10,14] + frac[11,14]
   + frac[13,14] >= 0.7
 cov_286_4: frac[0,4] + frac[2,4] + frac[8,4] + frac[9,4] + frac[12,4]
   >= 0.7
 cov_287_10: frac[3,10] + frac[6,10] + frac[9,10] + frac[11,10]
   + frac[12,10] + frac[13,10] + frac[14,10] >= 0.8
 cov_288_35: frac[1,35] + frac[7,35] + frac[8,35] + frac[11,35]
   + frac[12,35] + frac[13,35] + frac[14,35] >= 1
 cov_289_43: frac[1,43] + frac[4,43] + frac[8,43] + frac[10,43]
   + frac[12,43] + frac[13,43] + frac[14,43] >= 0.8
 cov_290_47: frac[0,47] + frac[4,47] + frac[8,47] + frac[10,47]
   + frac[12,47] + frac[13,47] + frac[14,47] >= 1
 cov_291_63: frac[4,63] + frac[6,63] + frac[10,63] + frac[11,63]
   + frac[13,63] >= 1
 cov_292_63: frac[0,63] + frac[4,63] + frac[8,63] + frac[10,63]
   + frac[12,63] + frac[13,63] + frac[14,63] >= 0.8
 cov_293_25: frac[1,25] + frac[4,25] + frac[8,25] + frac[10,25]
   + frac[12,25] + frac[13,25] + frac[14,25] >= 0.9
 cov_294_14: frac[3,14] + frac[4,14] + frac[9,14] + frac[10,14]
   + frac[12,14] + frac[13,14] + frac[14,14] >= 1
 cov_295_25: frac[0,25] + frac[6,25] + frac[8,25] + frac[11,25]
   + frac[12,25] + frac[13,25] + frac[14,25] >= 1
 cov_296_7: frac[2,7] + frac[7,7] + frac[9,7] + frac[11,7] + frac[12,7]
   + frac[13,7] + frac[14,7] >= 0.8
 cov_297_35: frac[6,35] + frac[7,35] + frac[11,35] >= 1
 cov_298_36: frac[1,36] + frac[2,36] + frac[8,36] + frac[9,36]
   + frac[12,36] >= 0.9
 cov_299_21: frac[0,21] + frac[5,21] + frac[8,21] + frac[10,21]
   + frac[12,21] + frac[13,21] + frac[14,21] >= 1
 cov_300_5: frac[0,5] + frac[1,5] + frac[8,5] >= 0.9
 cov_301_29: frac[0,29] + frac[3,29] + frac[8,29] + frac[9,29]
   + frac[12,29] >= 0.8
 cov_302_15: frac[1,15] + frac[4,15] + frac[8,15] + frac[10,15]
   + frac[12,15] + frac[13,15] + frac[14,15] >= 1
 cov_303_66: frac[2,66] + frac[5,66] + frac[9,66] + frac[10,66]
   + frac[12,66] + frac[13,66] + frac[14,66] >= 0.9
 cov_304_66: frac[4,66] + frac[7,66] + frac[10,66] + frac[11,66]
   + frac[13,66] >= 0.9
 cov_305_68: frac[0,68] + frac[6,68] + frac[8,68] + frac[11,68]
   + frac[12,68] + frac[13,68] + frac[14,68] >= 0.9
 cov_306_64: frac[1,64] + frac[6,64] + frac[8,64] + frac[11,64]
   + frac[12,64] + frac[13,64] + frac[14,64] >= 0.9
 cov_307_38: frac[5,38] + frac[6,38] + frac[10,38] + frac[11,38]
   + frac[13,38] >= 1
 cov_308_79: frac[1,79] + frac[4,79] + frac[8,79] + frac[10,79]
   + frac[12,79] + frac[13,79] + frac[14,79] >= 0.7
 cov_309_30: frac[4,30] + frac[5,30] + frac[10,30] >= 0.8
 cov_310_41: frac[1,41] + frac[6,41] + frac[8,41] + frac[11,41]
   + frac[12,41] + frac[13,41] + frac[14,41] >= 1
 cov_311_54: frac[0,54] + frac[4,54] + frac[8,54] + frac[10,54]
   + frac[12,54] + frac[13,54] + frac[14,54] >= 0.8
 cov_312_37: frac[5,37] + frac[6,37] + frac[10,37] + frac[11,37]
   + frac[13,37] >= 0.8
 cov_313_43: frac[4,43] + frac[5,43] + frac[10,43] >= 0.9
 cov_314_79: frac[4,79] + frac[6,79] + frac[10,79] + frac[11,79]
   + frac[13,79] >= 1
 cov_315_71: frac[2,71] + frac[5,71] + frac[9,71] + frac[10,71]
   + frac[12,71] + frac[13,71] + frac[14,71] >= 0.9
 cov_316_17: frac[4,17] + frac[6,17] + frac[10,17] + frac[11,17]
   + frac[13,17] >= 0.8
 cov_317_49: frac[0,49] + frac[7,49] + frac[8,49] + frac[11,49]
   + frac[12,49] + frac[13,49] + frac[14,49] >= 0.8
 cov_318_12: frac[1,12] + frac[5,12] + frac[8,12] + frac[10,12]
   + frac[12,12] + frac[13,12] + frac[14,12] >= 0.8
 cov_319_4: frac[3,4] + frac[6,4] + frac[9,4] + frac[11,4] + frac[12,4]
   + frac[13,4] + frac[14,4] >= 1
 cov_320_66: frac[2,66] + frac[7,66] + frac[9,66] + frac[11,66]
   + frac[12,66] + frac[13,66] + frac[14,66] >= 0.9
 cov_321_61: frac[0,61] + frac[2,61] + frac[8,61] + frac[9,61]
   + frac[12,61] >= 0.9
 cov_322_16: frac[3,16] + frac[5,16] + frac[9,16] + frac[10,16]
   + frac[12,16] + frac[13,16] + frac[14,16] >= 0.7
 cov_323_69: frac[2,69] + frac[4,69] + frac[9,69] + frac[10,69]
   + frac[12,69] + frac[13,69] + frac[14,69] >= 0.9
 cov_324_5: frac[1,5] + frac[4,5] + frac[8,5] + frac[10,5] + frac[12,5]
   + frac[13,5] + frac[14,5] >= 1
 cov_325_67: frac[3,67] + frac[5,67] + frac[9,67] + frac[10,67]
   + frac[12,67] + frac[13,67] + frac[14,67] >= 0.7
 cov_326_20: frac[1,20] + frac[7,20] + frac[8,20] + frac[11,20]
   + frac[12,20] + frac[13,20] + frac[14,20] >= 0.8
 cov_327_37: frac[2,37] + frac[3,37] + frac[9,37] >= 1
 cov_328_56: frac[3,56] + frac[6,56] + frac[9,56] + frac[11,56]
   + frac[12,56] + frac[13,56] + frac[14,56] >= 1
 cov_329_18: frac[4,18] + frac[5,18] + frac[10,18] >= 1
 cov_330_39: frac[1,39] + frac[6,39] + frac[8,39] + frac[11,39]
   + frac[12,39] + frac[13,39] + frac[14,39] >= 0.7
 cov_331_57: frac[3,57] + frac[5,57] + frac[9,57] + frac[10,57]
   + frac[12,57] + frac[13,57] + frac[14,57] >= 0.9
 cov_332_60: frac[5,60] + frac[7,60] + frac[10,60] + frac[11,60]
   + frac[13,60] >= 0.8
 cov_333_36: frac[2,36] + frac[4,36] + frac[9,36] + frac[10,36]
   + frac[12,36] + frac[13,36] + frac[14,36] >= 1
 cov_334_30: frac[2,30] + frac[6,30] + frac[9,30] + frac[11,30]
   + frac[12,30] + frac[13,30] + frac[14,30] >= 0.7
 cov_335_45: frac[0,45] + frac[3,45] + frac[8,45] + frac[9,45]
   + frac[12,45] >= 0.7
 cov_336_69: frac[3,69] + frac[4,69] + frac[9,69] + frac[10,69]
   + frac[12,69] + frac[13,69] + frac[14,69] >= 0.9
 cov_337_19: frac[3,19] + frac[6,19] + frac[9,19] + frac[11,19]
   + frac[12,19] + frac[13,19] + frac[14,19] >= 0.8
 cov_338_10: frac[0,10] + frac[7,10] + frac[8,10] + frac[11,10]
   + frac[12,10] + frac[13,10] + frac[14,10] >= 0.9
 cov_339_65: frac[1,65] + frac[4,65] + frac[8,65] + frac[10,65]
   + frac[12,65] + frac[13,65] + frac[14,65] >= 0.7
 cov_340_8: frac[4,8] + frac[6,8] + frac[10,8] + frac[11,8] + frac[13,8]
   >= 1
 cov_341_64: frac[2,64] + frac[4,64] + frac[9,64] + frac[10,64]
   + frac[12,64] + frac[13,64] + frac[14,64] >= 0.9
 cov_342_21: frac[3,21] + frac[6,21] + frac[9,21] + frac[11,21]
   + frac[12,21] + frac[13,21] + frac[14,21] >= 0.9
 cov_343_57: frac[1,57] + frac[2,57] + frac[8,57] + frac[9,57]
   + frac[12,57] >= 0.7
 cov_344_5: frac[0,5] + frac[2,5] + frac[8,5] + frac[9,5] + frac[12,5]
   >= 0.7
 cov_345_50: frac[1,50] + frac[6,50] + frac[8,50] + frac[11,50]
   + frac[12,50] + frac[13,50] + frac[14,50] >= 1
 cov_346_24: frac[2,24] + frac[6,24] + frac[9,24] + frac[11,24]
   + frac[12,24] + frac[13,24] + frac[14,24] >= 1
 cov_347_14: frac[3,14] + frac[5,14] + frac[9,14] + frac[10,14]
   + frac[12,14] + frac[13,14] + frac[14,14] >= 1
 cov_348_39: frac[4,39] + frac[7,39] + frac[10,39] + frac[11,39]
   + frac[13,39] >= 0.8
 cov_349_6: frac[2,6] + frac[4,6] + frac[9,6] + frac[10,6] + frac[12,6]
   + frac[13,6] + frac[14,6] >= 0.9
 cov_350_35: frac[1,35] + frac[5,35] + frac[8,35] + frac[10,35]
   + frac[12,35] + frac[13,35] + frac[14,35] >= 0.9
 cov_351_60: frac[0,60] + frac[6,60] + frac[8,60] + frac[11,60]
   + frac[12,60] + frac[13,60] + frac[14,60] >= 0.8
 cov_352_64: frac[2,64] + frac[5,64] + frac[9,64] + frac[10,64]
   + frac[12,64] + frac[13,64] + frac[14,64] >= 0.7
 cov_353_40: frac[1,40] + frac[5,40] + frac[8,40] + frac[10,40]
   + frac[12,40] + frac[13,40] + frac[14,40] >= 0.7
 cov_354_54: frac[0,54] + frac[6,54] + frac[8,54] + frac[11,54]
   + frac[12,54] + frac[13,54] + frac[14,54] >= 0.7
 cov_355_49: frac[3,49] + frac[6,49] + frac[9,49] + frac[11,49]
   + frac[12,49] + frac[13,49] + frac[14,49] >= 0.7
 cov_356_20: frac[0,20] + frac[6,20] + frac[8,20] + frac[11,20]
   + frac[12,20] + frac[13,20] + frac[14,20] >= 0.7
 cov_357_58: frac[6,58] + frac[7,58] + frac[11,58] >= 1
 cov_358_65: frac[1,65] + frac[7,65] + frac[8,65] + frac[11,65]
   + frac[12,65] + frac[13,65] + frac[14,65] >= 0.7
 cov_359_78: frac[3,78] + frac[7,78] + frac[9,78] + frac[11,78]
   + frac[12,78] + frac[13,78] + frac[14,78] >= 0.8
 cov_360_68: frac[0,68] + frac[2,68] + frac[8,68] + frac[9,68]
   + frac[12,68] >= 0.9
 cov_361_54: frac[2,54] + frac[4,54] + frac[9,54] + frac[10,54]
   + frac[12,54] + frac[13,54] + frac[14,54] >= 1
 cov_362_41: frac[1,41] + frac[3,41] + frac[8,41] + frac[9,41]
   + frac[12,41] >= 0.8
 cov_363_10: frac[4,10] + frac[5,10] + frac[10,10] >= 0.8
 cov_364_43: frac[4,43] + frac[5,43] + frac[10,43] >= 0.7
 cov_365_48: frac[0,48] + frac[3,48] + frac[8,48] + frac[9,48]
   + frac[12,48] >= 0.7
 cov_366_69: frac[0,69] + frac[2,69] + frac[8,69] + frac[9,69]
   + frac[12,69] >= 0.7
 cov_367_20: frac[0,20] + frac[6,20] + frac[8,20] + frac[11,20]
   + frac[12,20] + frac[13,20] + frac[14,20] >= 1
 cov_368_22: frac[2,22] + frac[4,22] + frac[9,22] + frac[10,22]
   + frac[12,22] + frac[13,22] + frac[14,22] >= 1
 cov_369_38: frac[2,38] + frac[7,38] + frac[9,38] + frac[11,38]
   + frac[12,38] + frac[13,38] + frac[14,38] >= 0.8
 cov_370_39: frac[2,39] + frac[5,39] + frac[9,39] + frac[10,39]
   + frac[12,39] + frac[13,39] + frac[14,39] >= 0.9
 cov_371_42: frac[4,42] + frac[7,42] + frac[10,42] + frac[11,42]
   + frac[13,42] >= 0.8
 cov_372_52: frac[1,52] + frac[7,52] + frac[8,52] + frac[11,52]
   + frac[12,52] + frac[13,52] + frac[14,52] >= 1
 cov_373_39: frac[1,39] + frac[5,39] + frac[8,39] + frac[10,39]
   + frac[12,39] + frac[13,39] + frac[14,39] >= 1
 cov_374_47: frac[2,47] + frac[5,47] + frac[9,47] + frac[10,47]
   + frac[12,47] + frac[13,47] + frac[14,47] >= 0.8
 cov_375_41: frac[0,41] + frac[7,41] + frac[8,41] + frac[11,41]
   + frac[12,41] + frac[13,41] + frac[14,41] >= 1
 cov_376_51: frac[2,51] + frac[3,51] + frac[9,51] >= 0.9
 cov_377_74: frac[0,74] + frac[4,74] + frac[8,74] + frac[10,74]
   + frac[12,74] + frac[13,74] + frac[14,74] >= 0.7
 cov_378_71: frac[1,71] + frac[6,71] + frac[8,71] + frac[11,71]
   + frac[12,71] + frac[13,71] + frac[14,71] >= 0.8
 cov_379_29: frac[2,29] + frac[6,29] + frac[9,29] + frac[11,29]
   + frac[12,29] + frac[13,29] + frac[14,29] >= 1
 cov_380_33: frac[1,33] + frac[4,33] + frac[8,33] + frac[10,33]
   + frac[12,33] + frac[13,33] + frac[14,33] >= 1
 cov_381_1: frac[3,1] + frac[4,1] + frac[9,1] + frac[10,1] + frac[12,1]
   + frac[13,1] + frac[14,1] >= 0.9
 cov_382_33: frac[3,33] + frac[6,33] + frac[9,33] + frac[11,33]
   + frac[12,33] + frac[13,33] + frac[14,33] >= 1
 cov_383_4: frac[5,4] + frac[6,4] + frac[10,4] + frac[11,4] + frac[13,4]
   >= 0.9
 cov_384_25: frac[2,25] + frac[6,25] + frac[9,25] + frac[11,25]
   + frac[12,25] + frac[13,25] + frac[14,25] >= 1
 cov_385_66: frac[2,66] + frac[4,66] + frac[9,66] + frac[10,66]
   + frac[12,66] + frac[13,66] + frac[14,66] >= 0.8
 cov_386_54: frac[2,54] + frac[7,54] + frac[9,54] + frac[11,54]
   + frac[12,54] + frac[13,54] + frac[14,54] >= 1
 cov_387_55: frac[0,55] + frac[1,55] + frac[8,55] >= 0.7
 cov_388_10: frac[0,10] + frac[1,10] + frac[8,10] >= 0.7
 cov_389_10: frac[2,10] + frac[7,10] + frac[9,10] + frac[11,10]
   + frac[12,10] + frac[13,10] + frac[14,10] >= 0.9
 cov_390_69: frac[2,69] + frac[3,69] + frac[9,69] >= 1
 cov_391_33: frac[1,33] + frac[7,33] + frac[8,33] + frac[11,33]
   + frac[12,33] + frac[13,33] + frac[14,33] >= 0.9
 cov_392_9: frac[6,9] + frac[7,9] + frac[11,9] >= 0.9
 cov_393_20: frac[2,20] + frac[3,20] + frac[9,20] >= 1
 cov_394_73: frac[0,73] + frac[5,73] + frac[8,73] + frac[10,73]
   + frac[12,73] + frac[13,73] + frac[14,73] >= 1
 cov_395_28: frac[1,28] + frac[5,28] + frac[8,28] + frac[10,28]
   + frac[12,28] + frac[13,28] + frac[14,28] >= 0.7
 cov_396_72: frac[0,72] + frac[7,72] + frac[8,72] + frac[11,72]
   + frac[12,72] + frac[13,72] + frac[14,72] >= 0.9
 cov_397_69: frac[2,69] + frac[3,69] + frac[9,69] >= 0.9
 cov_398_75: frac[5,75] + frac[7,75] + frac[10,75] + frac[11,75]
   + frac[13,75] >= 1
 cov_399_58: frac[4,58] + frac[7,58] + frac[10,58] + frac[11,58]
   + frac[13,58] >= 0.8
 accuracy_0_sk_0[0]: - 5.09765625 frac[0,0] + mem[0,0] >= 0
 accuracy_0_sk_0[1]: - 5.09765625 frac[1,0] + mem[1,0] >= 0
 accuracy_0_sk_0[2]: - 5.09765625 frac[2,0] + mem[2,0] >= 0
 accuracy_0_sk_0[3]: - 5.09765625 frac[3,0] + mem[3,0] >= 0
 accuracy_0_sk_0[4]: - 5.09765625 frac[4,0] + mem[4,0] >= 0
 accuracy_0_sk_0[5]: - 5.09765625 frac[5,0] + mem[5,0] >= 0
 accuracy_0_sk_0[6]: - 5.09765625 frac[6,0] + mem[6,0] >= 0
 accuracy_0_sk_0[7]: - 5.09765625 frac[7,0] + mem[7,0] >= 0
 accuracy_0_sk_0[8]: - 5.09765625 frac[8,0] + mem[8,0] >= 0
 accuracy_0_sk_0[9]: - 5.09765625 frac[9,0] + mem[9,0] >= 0
 accuracy_0_sk_0[10]: - 5.09765625 frac[10,0] + mem[10,0] >= 0
 accuracy_0_sk_0[11]: - 5.09765625 frac[11,0] + mem[11,0] >= 0
 accuracy_0_sk_0[12]: - 5.09765625 frac[12,0] + mem[12,0] >= 0
 accuracy_0_sk_0[13]: - 5.09765625 frac[13,0] + mem[13,0] >= 0
 accuracy_0_sk_0[14]: - 5.09765625 frac[14,0] + mem[14,0] >= 0
 accuracy_1_sk_1[0]: - 5.09765625 frac[0,1] + mem[0,1] >= 0
 accuracy_1_sk_1[1]: - 5.09765625 frac[1,1] + mem[1,1] >= 0
 accuracy_1_sk_1[2]: - 5.09765625 frac[2,1] + mem[2,1] >= 0
 accuracy_1_sk_1[3]: - 5.09765625 frac[3,1] + mem[3,1] >= 0
 accuracy_1_sk_1[4]: - 5.09765625 frac[4,1] + mem[4,1] >= 0
 accuracy_1_sk_1[5]: - 5.09765625 frac[5,1] + mem[5,1] >= 0
 accuracy_1_sk_1[6]: - 5.09765625 frac[6,1] + mem[6,1] >= 0
 accuracy_1_sk_1[7]: - 5.09765625 frac[7,1] + mem[7,1] >= 0
 accuracy_1_sk_1[8]: - 5.09765625 frac[8,1] + mem[8,1] >= 0
 accuracy_1_sk_1[9]: - 5.09765625 frac[9,1] + mem[9,1] >= 0
 accuracy_1_sk_1[10]: - 5.09765625 frac[10,1] + mem[10,1] >= 0
 accuracy_1_sk_1[11]: - 5.09765625 frac[11,1] + mem[11,1] >= 0
 accuracy_1_sk_1[12]: - 5.09765625 frac[12,1] + mem[12,1] >= 0
 accuracy_1_sk_1[13]: - 5.09765625 frac[13,1] + mem[13,1] >= 0
 accuracy_1_sk_1[14]: - 5.09765625 frac[14,1] + mem[14,1] >= 0
 accuracy_2_sk_2[0]: - 5.09765625 frac[0,2] + mem[0,2] >= 0
 accuracy_2_sk_2[1]: - 5.09765625 frac[1,2] + mem[1,2] >= 0
 accuracy_2_sk_2[2]: - 5.09765625 frac[2,2] + mem[2,2] >= 0
 accuracy_2_sk_2[3]: - 5.09765625 frac[3,2] + mem[3,2] >= 0
 accuracy_2_sk_2[4]: - 5.09765625 frac[4,2] + mem[4,2] >= 0
 accuracy_2_sk_2[5]: - 5.09765625 frac[5,2] + mem[5,2] >= 0
 accuracy_2_sk_2[6]: - 5.09765625 frac[6,2] + mem[6,2] >= 0
 accuracy_2_sk_2[7]: - 5.09765625 frac[7,2] + mem[7,2] >= 0
 accuracy_2_sk_2[8]: - 5.09765625 frac[8,2] + mem[8,2] >= 0
 accuracy_2_sk_2[9]: - 5.09765625 frac[9,2] + mem[9,2] >= 0
 accuracy_2_sk_2[10]: - 5.09765625 frac[10,2] + mem[10,2] >= 0
 accuracy_2_sk_2[11]: - 5.09765625 frac[11,2] + mem[11,2] >= 0
 accuracy_2_sk_2[12]: - 5.09765625 frac[12,2] + mem[12,2] >= 0
 accuracy_2_sk_2[13]: - 5.09765625 frac[13,2] + mem[13,2] >= 0
 accuracy_2_sk_2[14]: - 5.09765625 frac[14,2] + mem[14,2] >= 0
 accuracy_3_sk_3[0]: - 5.09765625 frac[0,3] + mem[0,3] >= 0
 accuracy_3_sk_3[1]: - 5.09765625 frac[1,3] + mem[1,3] >= 0
 accuracy_3_sk_3[2]: - 5.09765625 frac[2,3] + mem[2,3] >= 0
 accuracy_3_sk_3[3]: - 5.09765625 frac[3,3] + mem[3,3] >= 0
 accuracy_3_sk_3[4]: - 5.09765625 frac[4,3] + mem[4,3] >= 0
 accuracy_3_sk_3[5]: - 5.09765625 frac[5,3] + mem[5,3] >= 0
 accuracy_3_sk_3[6]: - 5.09765625 frac[6,3] + mem[6,3] >= 0
 accuracy_3_sk_3[7]: - 5.09765625 frac[7,3] + mem[7,3] >= 0
 accuracy_3_sk_3[8]: - 5.09765625 frac[8,3] + mem[8,3] >= 0
 accuracy_3_sk_3[9]: - 5.09765625 frac[9,3] + mem[9,3] >= 0
 accuracy_3_sk_3[10]: - 5.09765625 frac[10,3] + mem[10,3] >= 0
 accuracy_3_sk_3[11]: - 5.09765625 frac[11,3] + mem[11,3] >= 0
 accuracy_3_sk_3[12]: - 5.09765625 frac[12,3] + mem[12,3] >= 0
 accuracy_3_sk_3[13]: - 5.09765625 frac[13,3] + mem[13,3] >= 0
 accuracy_3_sk_3[14]: - 5.09765625 frac[14,3] + mem[14,3] >= 0
 accuracy_4_sk_4[0]: - 5.09765625 frac[0,4] + mem[0,4] >= 0
 accuracy_4_sk_4[1]: - 5.09765625 frac[1,4] + mem[1,4] >= 0
 accuracy_4_sk_4[2]: - 5.09765625 frac[2,4] + mem[2,4] >= 0
 accuracy_4_sk_4[3]: - 5.09765625 frac[3,4] + mem[3,4] >= 0
 accuracy_4_sk_4[4]: - 5.09765625 frac[4,4] + mem[4,4] >= 0
 accuracy_4_sk_4[5]: - 5.09765625 frac[5,4] + mem[5,4] >= 0
 accuracy_4_sk_4[6]: - 5.09765625 frac[6,4] + mem[6,4] >= 0
 accuracy_4_sk_4[7]: - 5.09765625 frac[7,4] + mem[7,4] >= 0
 accuracy_4_sk_4[8]: - 5.09765625 frac[8,4] + mem[8,4] >= 0
 accuracy_4_sk_4[9]: - 5.09765625 frac[9,4] + mem[9,4] >= 0
 accuracy_4_sk_4[10]: - 5.09765625 frac[10,4] + mem[10,4] >= 0
 accuracy_4_sk_4[11]: - 5.09765625 frac[11,4] + mem[11,4] >= 0
 accuracy_4_sk_4[12]: - 5.09765625 frac[12,4] + mem[12,4] >= 0
 accuracy_4_sk_4[13]: - 5.09765625 frac[13,4] + mem[13,4] >= 0
 accuracy_4_sk_4[14]: - 5.09765625 frac[14,4] + mem[14,4] >= 0
 accuracy_5_sk_5[0]: - 5.09765625 frac[0,5] + mem[0,5] >= 0
 accuracy_5_sk_5[1]: - 5.09765625 frac[1,5] + mem[1,5] >= 0
 accuracy_5_sk_5[2]: - 5.09765625 frac[2,5] + mem[2,5] >= 0
 accuracy_5_sk_5[3]: - 5.09765625 frac[3,5] + mem[3,5] >= 0
 accuracy_5_sk_5[4]: - 5.09765625 frac[4,5] + mem[4,5] >= 0
 accuracy_5_sk_5[5]: - 5.09765625 frac[5,5] + mem[5,5] >= 0
 accuracy_5_sk_5[6]: - 5.09765625 frac[6,5] + mem[6,5] >= 0
 accuracy_5_sk_5[7]: - 5.09765625 frac[7,5] + mem[7,5] >= 0
 accuracy_5_sk_5[8]: - 5.09765625 frac[8,5] + mem[8,5] >= 0
 accuracy_5_sk_5[9]: - 5.09765625 frac[9,5] + mem[9,5] >= 0
 accuracy_5_sk_5[10]: - 5.09765625 frac[10,5] + mem[10,5] >= 0
 accuracy_5_sk_5[11]: - 5.09765625 frac[11,5] + mem[11,5] >= 0
 accuracy_5_sk_5[12]: - 5.09765625 frac[12,5] + mem[12,5] >= 0
 accuracy_5_sk_5[13]: - 5.09765625 frac[13,5] + mem[13,5] >= 0
 accuracy_5_sk_5[14]: - 5.09765625 frac[14,5] + mem[14,5] >= 0
 accuracy_6_sk_6[0]: - 5.09765625 frac[0,6] + mem[0,6] >= 0
 accuracy_6_sk_6[1]: - 5.09765625 frac[1,6] + mem[1,6] >= 0
 accuracy_6_sk_6[2]: - 5.09765625 frac[2,6] + mem[2,6] >= 0
 accuracy_6_sk_6[3]: - 5.09765625 frac[3,6] + mem[3,6] >= 0
 accuracy_6_sk_6[4]: - 5.09765625 frac[4,6] + mem[4,6] >= 0
 accuracy_6_sk_6[5]: - 5.09765625 frac[5,6] + mem[5,6] >= 0
 accuracy_6_sk_6[6]: - 5.09765625 frac[6,6] + mem[6,6] >= 0
 accuracy_6_sk_6[7]: - 5.09765625 frac[7,6] + mem[7,6] >= 0
 accuracy_6_sk_6[8]: - 5.09765625 frac[8,6] + mem[8,6] >= 0
 accuracy_6_sk_6[9]: - 5.09765625 frac[9,6] + mem[9,6] >= 0
 accuracy_6_sk_6[10]: - 5.09765625 frac[10,6] + mem[10,6] >= 0
 accuracy_6_sk_6[11]: - 5.09765625 frac[11,6] + mem[11,6] >= 0
 accuracy_6_sk_6[12]: - 5.09765625 frac[12,6] + mem[12,6] >= 0
 accuracy_6_sk_6[13]: - 5.09765625 frac[13,6] + mem[13,6] >= 0
 accuracy_6_sk_6[14]: - 5.09765625 frac[14,6] + mem[14,6] >= 0
 accuracy_7_sk_7[0]: - 5.09765625 frac[0,7] + mem[0,7] >= 0
 accuracy_7_sk_7[1]: - 5.09765625 frac[1,7] + mem[1,7] >= 0
 accuracy_7_sk_7[2]: - 5.09765625 frac[2,7] + mem[2,7] >= 0
 accuracy_7_sk_7[3]: - 5.09765625 frac[3,7] + mem[3,7] >= 0
 accuracy_7_sk_7[4]: - 5.09765625 frac[4,7] + mem[4,7] >= 0
 accuracy_7_sk_7[5]: - 5.09765625 frac[5,7] + mem[5,7] >= 0
 accuracy_7_sk_7[6]: - 5.09765625 frac[6,7] + mem[6,7] >= 0
 accuracy_7_sk_7[7]: - 5.09765625 frac[7,7] + mem[7,7] >= 0
 accuracy_7_sk_7[8]: - 5.09765625 frac[8,7] + mem[8,7] >= 0
 accuracy_7_sk_7[9]: - 5.09765625 frac[9,7] + mem[9,7] >= 0
 accuracy_7_sk_7[10]: - 5.09765625 frac[10,7] + mem[10,7] >= 0
 accuracy_7_sk_7[11]: - 5.09765625 frac[11,7] + mem[11,7] >= 0
 accuracy_7_sk_7[12]: - 5.09765625 frac[12,7] + mem[12,7] >= 0
 accuracy_7_sk_7[13]: - 5.09765625 frac[13,7] + mem[13,7] >= 0
 accuracy_7_sk_7[14]: - 5.09765625 frac[14,7] + mem[14,7] >= 0
 accuracy_8_sk_8[0]: - 5.09765625 frac[0,8] + mem[0,8] >= 0
 accuracy_8_sk_8[1]: - 5.09765625 frac[1,8] + mem[1,8] >= 0
 accuracy_8_sk_8[2]: - 5.09765625 frac[2,8] + mem[2,8] >= 0
 accuracy_8_sk_8[3]: - 5.09765625 frac[3,8] + mem[3,8] >= 0
 accuracy_8_sk_8[4]: - 5.09765625 frac[4,8] + mem[4,8] >= 0
 accuracy_8_sk_8[5]: - 5.09765625 frac[5,8] + mem[5,8] >= 0
 accuracy_8_sk_8[6]: - 5.09765625 frac[6,8] + mem[6,8] >= 0
 accuracy_8_sk_8[7]: - 5.09765625 frac[7,8] + mem[7,8] >= 0
 accuracy_8_sk_8[8]: - 5.09765625 frac[8,8] + mem[8,8] >= 0
 accuracy_8_sk_8[9]: - 5.09765625 frac[9,8] + mem[9,8] >= 0
 accuracy_8_sk_8[10]: - 5.09765625 frac[10,8] + mem[10,8] >= 0
 accuracy_8_sk_8[11]: - 5.09765625 frac[11,8] + mem[11,8] >= 0
 accuracy_8_sk_8[12]: - 5.09765625 frac[12,8] + mem[12,8] >= 0
 accuracy_8_sk_8[13]: - 5.09765625 frac[13,8] + mem[13,8] >= 0
 accuracy_8_sk_8[14]: - 5.09765625 frac[14,8] + mem[14,8] >= 0
 accuracy_9_sk_9[0]: - 5.09765625 frac[0,9] + mem[0,9] >= 0
 accuracy_9_sk_9[1]: - 5.09765625 frac[1,9] + mem[1,9] >= 0
 accuracy_9_sk_9[2]: - 5.09765625 frac[2,9] + mem[2,9] >= 0
 accuracy_9_sk_9[3]: - 5.09765625 frac[3,9] + mem[3,9] >= 0
 accuracy_9_sk_9[4]: - 5.09765625 frac[4,9] + mem[4,9] >= 0
 accuracy_9_sk_9[5]: - 5.09765625 frac[5,9] + mem[5,9] >= 0
 accuracy_9_sk_9[6]: - 5.09765625 frac[6,9] + mem[6,9] >= 0
 accuracy_9_sk_9[7]: - 5.09765625 frac[7,9] + mem[7,9] >= 0
 accuracy_9_sk_9[8]: - 5.09765625 frac[8,9] + mem[8,9] >= 0
 accuracy_9_sk_9[9]: - 5.09765625 frac[9,9] + mem[9,9] >= 0
 accuracy_9_sk_9[10]: - 5.09765625 frac[10,9] + mem[10,9] >= 0
 accuracy_9_sk_9[11]: - 5.09765625 frac[11,9] + mem[11,9] >= 0
 accuracy_9_sk_9[12]: - 5.09765625 frac[12,9] + mem[12,9] >= 0
 accuracy_9_sk_9[13]: - 5.09765625 frac[13,9] + mem[13,9] >= 0
 accuracy_9_sk_9[14]: - 5.09765625 frac[14,9] + mem[14,9] >= 0
 accuracy_10_sk_10[0]: - 5.09765625 frac[0,10] + mem[0,10] >= 0
 accuracy_10_sk_10[1]: - 5.09765625 frac[1,10] + mem[1,10] >= 0
 accuracy_10_sk_10[2]: - 5.09765625 frac[2,10] + mem[2,10] >= 0
 accuracy_10_sk_10[3]: - 5.09765625 frac[3,10] + mem[3,10] >= 0
 accuracy_10_sk_10[4]: - 5.09765625 frac[4,10] + mem[4,10] >= 0
 accuracy_10_sk_10[5]: - 5.09765625 frac[5,10] + mem[5,10] >= 0
 accuracy_10_sk_10[6]: - 5.09765625 frac[6,10] + mem[6,10] >= 0
 accuracy_10_sk_10[7]: - 5.09765625 frac[7,10] + mem[7,10] >= 0
 accuracy_10_sk_10[8]: - 5.09765625 frac[8,10] + mem[8,10] >= 0
 accuracy_10_sk_10[9]: - 5.09765625 frac[9,10] + mem[9,10] >= 0
 accuracy_10_sk_10[10]: - 5.09765625 frac[10,10] + mem[10,10] >= 0
 accuracy_10_sk_10[11]: - 5.09765625 frac[11,10] + mem[11,10] >= 0
 accuracy_10_sk_10[12]: - 5.09765625 frac[12,10] + mem[12,10] >= 0
 accuracy_10_sk_10[13]: - 5.09765625 frac[13,10] + mem[13,10] >= 0
 accuracy_10_sk_10[14]: - 5.09765625 frac[14,10] + mem[14,10] >= 0
 accuracy_11_sk_11[0]: - 5.09765625 frac[0,11] + mem[0,11] >= 0
 accuracy_11_sk_11[1]: - 5.09765625 frac[1,11] + mem[1,11] >= 0
 accuracy_11_sk_11[2]: - 5.09765625 frac[2,11] + mem[2,11] >= 0
 accuracy_11_sk_11[3]: - 5.09765625 frac[3,11] + mem[3,11] >= 0
 accuracy_11_sk_11[4]: - 5.09765625 frac[4,11] + mem[4,11] >= 0
 accuracy_11_sk_11[5]: - 5.09765625 frac[5,11] + mem[5,11] >= 0
 accuracy_11_sk_11[6]: - 5.09765625 frac[6,11] + mem[6,11] >= 0
 accuracy_11_sk_11[7]: - 5.09765625 frac[7,11] + mem[7,11] >= 0
 accuracy_11_sk_11[8]: - 5.09765625 frac[8,11] + mem[8,11] >= 0
 accuracy_11_sk_11[9]: - 5.09765625 frac[9,11] + mem[9,11] >= 0
 accuracy_11_sk_11[10]: - 5.09765625 frac[10,11] + mem[10,11] >= 0
 accuracy_11_sk_11[11]: - 5.09765625 frac[11,11] + mem[11,11] >= 0
 accuracy_11_sk_11[12]: - 5.09765625 frac[12,11] + mem[12,11] >= 0
 accuracy_11_sk_11[13]: - 5.09765625 frac[13,11] + mem[13,11] >= 0
 accuracy_11_sk_11[14]: - 5.09765625 frac[14,11] + mem[14,11] >= 0
 accuracy_12_sk_12[0]: - 5.09765625 frac[0,12] + mem[0,12] >= 0
 accuracy_12_sk_12[1]: - 5.09765625 frac[1,12] + mem[1,12] >= 0
 accuracy_12_sk_12[2]: - 5.09765625 frac[2,12] + mem[2,12] >= 0
 accuracy_12_sk_12[3]: - 5.09765625 frac[3,12] + mem[3,12] >= 0
 accuracy_12_sk_12[4]: - 5.09765625 frac[4,12] + mem[4,12] >= 0
 accuracy_12_sk_12[5]: - 5.09765625 frac[5,12] + mem[5,12] >= 0
 accuracy_12_sk_12[6]: - 5.09765625 frac[6,12] + mem[6,12] >= 0
 accuracy_12_sk_12[7]: - 5.09765625 frac[7,12] + mem[7,12] >= 0
 accuracy_12_sk_12[8]: - 5.09765625 frac[8,12] + mem[8,12] >= 0
 accuracy_12_sk_12[9]: - 5.09765625 frac[9,12] + mem[9,12] >= 0
 accuracy_12_sk_12[10]: - 5.09765625 frac[10,12] + mem[10,12] >= 0
 accuracy_12_sk_12[11]: - 5.09765625 frac[11,12] + mem[11,12] >= 0
 accuracy_12_sk_12[12]: - 5.09765625 frac[12,12] + mem[12,12] >= 0
 accuracy_12_sk_12[13]: - 5.09765625 frac[13,12] + mem[13,12] >= 0
 accuracy_12_sk_12[14]: - 5.09765625 frac[14,12] + mem[14,12] >= 0
 accuracy_13_sk_13[0]: - 5.09765625 frac[0,13] + mem[0,13] >= 0
 accuracy_13_sk_13[1]: - 5.09765625 frac[1,13] + mem[1,13] >= 0
 accuracy_13_sk_13[2]: - 5.09765625 frac[2,13] + mem[2,13] >= 0
 accuracy_13_sk_13[3]: - 5.09765625 frac[3,13] + mem[3,13] >= 0
 accuracy_13_sk_13[4]: - 5.09765625 frac[4,13] + mem[4,13] >= 0
 accuracy_13_sk_13[5]: - 5.09765625 frac[5,13] + mem[5,13] >= 0
 accuracy_13_sk_13[6]: - 5.09765625 frac[6,13] + mem[6,13] >= 0
 accuracy_13_sk_13[7]: - 5.09765625 frac[7,13] + mem[7,13] >= 0
 accuracy_13_sk_13[8]: - 5.09765625 frac[8,13] + mem[8,13] >= 0
 accuracy_13_sk_13[9]: - 5.09765625 frac[9,13] + mem[9,13] >= 0
 accuracy_13_sk_13[10]: - 5.09765625 frac[10,13] + mem[10,13] >= 0
 accuracy_13_sk_13[11]: - 5.09765625 frac[11,13] + mem[11,13] >= 0
 accuracy_13_sk_13[12]: - 5.09765625 frac[12,13] + mem[12,13] >= 0
 accuracy_13_sk_13[13]: - 5.09765625 frac[13,13] + mem[13,13] >= 0
 accuracy_13_sk_13[14]: - 5.09765625 frac[14,13] + mem[14,13] >= 0
 accuracy_14_sk_14[0]: - 5.09765625 frac[0,14] + mem[0,14] >= 0
 accuracy_14_sk_14[1]: - 5.09765625 frac[1,14] + mem[1,14] >= 0
 accuracy_14_sk_14[2]: - 5.09765625 frac[2,14] + mem[2,14] >= 0
 accuracy_14_sk_14[3]: - 5.09765625 frac[3,14] + mem[3,14] >= 0
 accuracy_14_sk_14[4]: - 5.09765625 frac[4,14] + mem[4,14] >= 0
 accuracy_14_sk_14[5]: - 5.09765625 frac[5,14] + mem[5,14] >= 0
 accuracy_14_sk_14[6]: - 5.09765625 frac[6,14] + mem[6,14] >= 0
 accuracy_14_sk_14[7]: - 5.09765625 frac[7,14] + mem[7,14] >= 0
 accuracy_14_sk_14[8]: - 5.09765625 frac[8,14] + mem[8,14] >= 0
 accuracy_14_sk_14[9]: - 5.09765625 frac[9,14] + mem[9,14] >= 0
 accuracy_14_sk_14[10]: - 5.09765625 frac[10,14] + mem[10,14] >= 0
 accuracy_14_sk_14[11]: - 5.09765625 frac[11,14] + mem[11,14] >= 0
 accuracy_14_sk_14[12]: - 5.09765625 frac[12,14] + mem[12,14] >= 0
 accuracy_14_sk_14[13]: - 5.09765625 frac[13,14] + mem[13,14] >= 0
 accuracy_14_sk_14[14]: - 5.09765625 frac[14,14] + mem[14,14] >= 0
 accuracy_15_sk_15[0]: - 5.09765625 frac[0,15] + mem[0,15] >= 0
 accuracy_15_sk_15[1]: - 5.09765625 frac[1,15] + mem[1,15] >= 0
 accuracy_15_sk_15[2]: - 5.09765625 frac[2,15] + mem[2,15] >= 0
 accuracy_15_sk_15[3]: - 5.09765625 frac[3,15] + mem[3,15] >= 0
 accuracy_15_sk_15[4]: - 5.09765625 frac[4,15] + mem[4,15] >= 0
 accuracy_15_sk_15[5]: - 5.09765625 frac[5,15] + mem[5,15] >= 0
 accuracy_15_sk_15[6]: - 5.09765625 frac[6,15] + mem[6,15] >= 0
 accuracy_15_sk_15[7]: - 5.09765625 frac[7,15] + mem[7,15] >= 0
 accuracy_15_sk_15[8]: - 5.09765625 frac[8,15] + mem[8,15] >= 0
 accuracy_15_sk_15[9]: - 5.09765625 frac[9,15] + mem[9,15] >= 0
 accuracy_15_sk_15[10]: - 5.09765625 frac[10,15] + mem[10,15] >= 0
 accuracy_15_sk_15[11]: - 5.09765625 frac[11,15] + mem[11,15] >= 0
 accuracy_15_sk_15[12]: - 5.09765625 frac[12,15] + mem[12,15] >= 0
 accuracy_15_sk_15[13]: - 5.09765625 frac[13,15] + mem[13,15] >= 0
 accuracy_15_sk_15[14]: - 5.09765625 frac[14,15] + mem[14,15] >= 0
 accuracy_16_sk_16[0]: - 5.09765625 frac[0,16] + mem[0,16] >= 0
 accuracy_16_sk_16[1]: - 5.09765625 frac[1,16] + mem[1,16] >= 0
 accuracy_16_sk_16[2]: - 5.09765625 frac[2,16] + mem[2,16] >= 0
 accuracy_16_sk_16[3]: - 5.09765625 frac[3,16] + mem[3,16] >= 0
 accuracy_16_sk_16[4]: - 5.09765625 frac[4,16] + mem[4,16] >= 0
 accuracy_16_sk_16[5]: - 5.09765625 frac[5,16] + mem[5,16] >= 0
 accuracy_16_sk_16[6]: - 5.09765625 frac[6,16] + mem[6,16] >= 0
 accuracy_16_sk_16[7]: - 5.09765625 frac[7,16] + mem[7,16] >= 0
 accuracy_16_sk_16[8]: - 5.09765625 frac[8,16] + mem[8,16] >= 0
 accuracy_16_sk_16[9]: - 5.09765625 frac[9,16] + mem[9,16] >= 0
 accuracy_16_sk_16[10]: - 5.09765625 frac[10,16] + mem[10,16] >= 0
 accuracy_16_sk_16[11]: - 5.09765625 frac[11,16] + mem[11,16] >= 0
 accuracy_16_sk_16[12]: - 5.09765625 frac[12,16] + mem[12,16] >= 0
 accuracy_16_sk_16[13]: - 5.09765625 frac[13,16] + mem[13,16] >= 0
 accuracy_16_sk_16[14]: - 5.09765625 frac[14,16] + mem[14,16] >= 0
 accuracy_17_sk_17[0]: - 5.09765625 frac[0,17] + mem[0,17] >= 0
 accuracy_17_sk_17[1]: - 5.09765625 frac[1,17] + mem[1,17] >= 0
 accuracy_17_sk_17[2]: - 5.09765625 frac[2,17] + mem[2,17] >= 0
 accuracy_17_sk_17[3]: - 5.09765625 frac[3,17] + mem[3,17] >= 0
 accuracy_17_sk_17[4]: - 5.09765625 frac[4,17] + mem[4,17] >= 0
 accuracy_17_sk_17[5]: - 5.09765625 frac[5,17] + mem[5,17] >= 0
 accuracy_17_sk_17[6]: - 5.09765625 frac[6,17] + mem[6,17] >= 0
 accuracy_17_sk_17[7]: - 5.09765625 frac[7,17] + mem[7,17] >= 0
 accuracy_17_sk_17[8]: - 5.09765625 frac[8,17] + mem[8,17] >= 0
 accuracy_17_sk_17[9]: - 5.09765625 frac[9,17] + mem[9,17] >= 0
 accuracy_17_sk_17[10]: - 5.09765625 frac[10,17] + mem[10,17] >= 0
 accuracy_17_sk_17[11]: - 5.09765625 frac[11,17] + mem[11,17] >= 0
 accuracy_17_sk_17[12]: - 5.09765625 frac[12,17] + mem[12,17] >= 0
 accuracy_17_sk_17[13]: - 5.09765625 frac[13,17] + mem[13,17] >= 0
 accuracy_17_sk_17[14]: - 5.09765625 frac[14,17] + mem[14,17] >= 0
 accuracy_18_sk_18[0]: - 5.09765625 frac[0,18] + mem[0,18] >= 0
 accuracy_18_sk_18[1]: - 5.09765625 frac[1,18] + mem[1,18] >= 0
 accuracy_18_sk_18[2]: - 5.09765625 frac[2,18] + mem[2,18] >= 0
 accuracy_18_sk_18[3]: - 5.09765625 frac[3,18] + mem[3,18] >= 0
 accuracy_18_sk_18[4]: - 5.09765625 frac[4,18] + mem[4,18] >= 0
 accuracy_18_sk_18[5]: - 5.09765625 frac[5,18] + mem[5,18] >= 0
 accuracy_18_sk_18[6]: - 5.09765625 frac[6,18] + mem[6,18] >= 0
 accuracy_18_sk_18[7]: - 5.09765625 frac[7,18] + mem[7,18] >= 0
 accuracy_18_sk_18[8]: - 5.09765625 frac[8,18] + mem[8,18] >= 0
 accuracy_18_sk_18[9]: - 5.09765625 frac[9,18] + mem[9,18] >= 0
 accuracy_18_sk_18[10]: - 5.09765625 frac[10,18] + mem[10,18] >= 0
 accuracy_18_sk_18[11]: - 5.09765625 frac[11,18] + mem[11,18] >= 0
 accuracy_18_sk_18[12]: - 5.09765625 frac[12,18] + mem[12,18] >= 0
 accuracy_18_sk_18[13]: - 5.09765625 frac[13,18] + mem[13,18] >= 0
 accuracy_18_sk_18[14]: - 5.09765625 frac[14,18] + mem[14,18] >= 0
 accuracy_19_sk_19[0]: - 5.09765625 frac[0,19] + mem[0,19] >= 0
 accuracy_19_sk_19[1]: - 5.09765625 frac[1,19] + mem[1,19] >= 0
 accuracy_19_sk_19[2]: - 5.09765625 frac[2,19] + mem[2,19] >= 0
 accuracy_19_sk_19[3]: - 5.09765625 frac[3,19] + mem[3,19] >= 0
 accuracy_19_sk_19[4]: - 5.09765625 frac[4,19] + mem[4,19] >= 0
 accuracy_19_sk_19[5]: - 5.09765625 frac[5,19] + mem[5,19] >= 0
 accuracy_19_sk_19[6]: - 5.09765625 frac[6,19] + mem[6,19] >= 0
 accuracy_19_sk_19[7]: - 5.09765625 frac[7,19] + mem[7,19] >= 0
 accuracy_19_sk_19[8]: - 5.09765625 frac[8,19] + mem[8,19] >= 0
 accuracy_19_sk_19[9]: - 5.09765625 frac[9,19] + mem[9,19] >= 0
 accuracy_19_sk_19[10]: - 5.09765625 frac[10,19] + mem[10,19] >= 0
 accuracy_19_sk_19[11]: - 5.09765625 frac[11,19] + mem[11,19] >= 0
 accuracy_19_sk_19[12]: - 5.09765625 frac[12,19] + mem[12,19] >= 0
 accuracy_19_sk_19[13]: - 5.09765625 frac[13,19] + mem[13,19] >= 0
 accuracy_19_sk_19[14]: - 5.09765625 frac[14,19] + mem[14,19] >= 0
 accuracy_20_sk_20[0]: - 5.09765625 frac[0,20] + mem[0,20] >= 0
 accuracy_20_sk_20[1]: - 5.09765625 frac[1,20] + mem[1,20] >= 0
 accuracy_20_sk_20[2]: - 5.09765625 frac[2,20] + mem[2,20] >= 0
 accuracy_20_sk_20[3]: - 5.09765625 frac[3,20] + mem[3,20] >= 0
 accuracy_20_sk_20[4]: - 5.09765625 frac[4,20] + mem[4,20] >= 0
 accuracy_20_sk_20[5]: - 5.09765625 frac[5,20] + mem[5,20] >= 0
 accuracy_20_sk_20[6]: - 5.09765625 frac[6,20] + mem[6,20] >= 0
 accuracy_20_sk_20[7]: - 5.09765625 frac[7,20] + mem[7,20] >= 0
 accuracy_20_sk_20[8]: - 5.09765625 frac[8,20] + mem[8,20] >= 0
 accuracy_20_sk_20[9]: - 5.09765625 frac[9,20] + mem[9,20] >= 0
 accuracy_20_sk_20[10]: - 5.09765625 frac[10,20] + mem[10,20] >= 0
 accuracy_20_sk_20[11]: - 5.09765625 frac[11,20] + mem[11,20] >= 0
 accuracy_20_sk_20[12]: - 5.09765625 frac[12,20] + mem[12,20] >= 0
 accuracy_20_sk_20[13]: - 5.09765625 frac[13,20] + mem[13,20] >= 0
 accuracy_20_sk_20[14]: - 5.09765625 frac[14,20] + mem[14,20] >= 0
 accuracy_21_sk_21[0]: - 5.09765625 frac[0,21] + mem[0,21] >= 0
 accuracy_21_sk_21[1]: - 5.09765625 frac[1,21] + mem[1,21] >= 0
 accuracy_21_sk_21[2]: - 5.09765625 frac[2,21] + mem[2,21] >= 0
 accuracy_21_sk_21[3]: - 5.09765625 frac[3,21] + mem[3,21] >= 0
 accuracy_21_sk_21[4]: - 5.09765625 frac[4,21] + mem[4,21] >= 0
 accuracy_21_sk_21[5]: - 5.09765625 frac[5,21] + mem[5,21] >= 0
 accuracy_21_sk_21[6]: - 5.09765625 frac[6,21] + mem[6,21] >= 0
 accuracy_21_sk_21[7]: - 5.09765625 frac[7,21] + mem[7,21] >= 0
 accuracy_21_sk_21[8]: - 5.09765625 frac[8,21] + mem[8,21] >= 0
 accuracy_21_sk_21[9]: - 5.09765625 frac[9,21] + mem[9,21] >= 0
 accuracy_21_sk_21[10]: - 5.09765625 frac[10,21] + mem[10,21] >= 0
 accuracy_21_sk_21[11]: - 5.09765625 frac[11,21] + mem[11,21] >= 0
 accuracy_21_sk_21[12]: - 5.09765625 frac[12,21] + mem[12,21] >= 0
 accuracy_21_sk_21[13]: - 5.09765625 frac[13,21] + mem[13,21] >= 0
 accuracy_21_sk_21[14]: - 5.09765625 frac[14,21] + mem[14,21] >= 0
 accuracy_22_sk_22[0]: - 5.09765625 frac[0,22] + mem[0,22] >= 0
 accuracy_22_sk_22[1]: - 5.09765625 frac[1,22] + mem[1,22] >= 0
 accuracy_22_sk_22[2]: - 5.09765625 frac[2,22] + mem[2,22] >= 0
 accuracy_22_sk_22[3]: - 5.09765625 frac[3,22] + mem[3,22] >= 0
 accuracy_22_sk_22[4]: - 5.09765625 frac[4,22] + mem[4,22] >= 0
 accuracy_22_sk_22[5]: - 5.09765625 frac[5,22] + mem[5,22] >= 0
 accuracy_22_sk_22[6]: - 5.09765625 frac[6,22] + mem[6,22] >= 0
 accuracy_22_sk_22[7]: - 5.09765625 frac[7,22] + mem[7,22] >= 0
 accuracy_22_sk_22[8]: - 5.09765625 frac[8,22] + mem[8,22] >= 0
 accuracy_22_sk_22[9]: - 5.09765625 frac[9,22] + mem[9,22] >= 0
 accuracy_22_sk_22[10]: - 5.09765625 frac[10,22] + mem[10,22] >= 0
 accuracy_22_sk_22[11]: - 5.09765625 frac[11,22] + mem[11,22] >= 0
 accuracy_22_sk_22[12]: - 5.09765625 frac[12,22] + mem[12,22] >= 0
 accuracy_22_sk_22[13]: - 5.09765625 frac[13,22] + mem[13,22] >= 0
 accuracy_22_sk_22[14]: - 5.09765625 frac[14,22] + mem[14,22] >= 0
 accuracy_23_sk_23[0]: - 5.09765625 frac[0,23] + mem[0,23] >= 0
 accuracy_23_sk_23[1]: - 5.09765625 frac[1,23] + mem[1,23] >= 0
 accuracy_23_sk_23[2]: - 5.09765625 frac[2,23] + mem[2,23] >= 0
 accuracy_23_sk_23[3]: - 5.09765625 frac[3,23] + mem[3,23] >= 0
 accuracy_23_sk_23[4]: - 5.09765625 frac[4,23] + mem[4,23] >= 0
 accuracy_23_sk_23[5]: - 5.09765625 frac[5,23] + mem[5,23] >= 0
 accuracy_23_sk_23[6]: - 5.09765625 frac[6,23] + mem[6,23] >= 0
 accuracy_23_sk_23[7]: - 5.09765625 frac[7,23] + mem[7,23] >= 0
 accuracy_23_sk_23[8]: - 5.09765625 frac[8,23] + mem[8,23] >= 0
 accuracy_23_sk_23[9]: - 5.09765625 frac[9,23] + mem[9,23] >= 0
 accuracy_23_sk_23[10]: - 5.09765625 frac[10,23] + mem[10,23] >= 0
 accuracy_23_sk_23[11]: - 5.09765625 frac[11,23] + mem[11,23] >= 0
 accuracy_23_sk_23[12]: - 5.09765625 frac[12,23] + mem[12,23] >= 0
 accuracy_23_sk_23[13]: - 5.09765625 frac[13,23] + mem[13,23] >= 0
 accuracy_23_sk_23[14]: - 5.09765625 frac[14,23] + mem[14,23] >= 0
 accuracy_24_sk_24[0]: - 5.09765625 frac[0,24] + mem[0,24] >= 0
 accuracy_24_sk_24[1]: - 5.09765625 frac[1,24] + mem[1,24] >= 0
 accuracy_24_sk_24[2]: - 5.09765625 frac[2,24] + mem[2,24] >= 0
 accuracy_24_sk_24[3]: - 5.09765625 frac[3,24] + mem[3,24] >= 0
 accuracy_24_sk_24[4]: - 5.09765625 frac[4,24] + mem[4,24] >= 0
 accuracy_24_sk_24[5]: - 5.09765625 frac[5,24] + mem[5,24] >= 0
 accuracy_24_sk_24[6]: - 5.09765625 frac[6,24] + mem[6,24] >= 0
 accuracy_24_sk_24[7]: - 5.09765625 frac[7,24] + mem[7,24] >= 0
 accuracy_24_sk_24[8]: - 5.09765625 frac[8,24] + mem[8,24] >= 0
 accuracy_24_sk_24[9]: - 5.09765625 frac[9,24] + mem[9,24] >= 0
 accuracy_24_sk_24[10]: - 5.09765625 frac[10,24] + mem[10,24] >= 0
 accuracy_24_sk_24[11]: - 5.09765625 frac[11,24] + mem[11,24] >= 0
 accuracy_24_sk_24[12]: - 5.09765625 frac[12,24] + mem[12,24] >= 0
 accuracy_24_sk_24[13]: - 5.09765625 frac[13,24] + mem[13,24] >= 0
 accuracy_24_sk_24[14]: - 5.09765625 frac[14,24] + mem[14,24] >= 0
 accuracy_25_sk_25[0]: - 5.09765625 frac[0,25] + mem[0,25] >= 0
 accuracy_25_sk_25[1]: - 5.09765625 frac[1,25] + mem[1,25] >= 0
 accuracy_25_sk_25[2]: - 5.09765625 frac[2,25] + mem[2,25] >= 0
 accuracy_25_sk_25[3]: - 5.09765625 frac[3,25] + mem[3,25] >= 0
 accuracy_25_sk_25[4]: - 5.09765625 frac[4,25] + mem[4,25] >= 0
 accuracy_25_sk_25[5]: - 5.09765625 frac[5,25] + mem[5,25] >= 0
 accuracy_25_sk_25[6]: - 5.09765625 frac[6,25] + mem[6,25] >= 0
 accuracy_25_sk_25[7]: - 5.09765625 frac[7,25] + mem[7,25] >= 0
 accuracy_25_sk_25[8]: - 5.09765625 frac[8,25] + mem[8,25] >= 0
 accuracy_25_sk_25[9]: - 5.09765625 frac[9,25] + mem[9,25] >= 0
 accuracy_25_sk_25[10]: - 5.09765625 frac[10,25] + mem[10,25] >= 0
 accuracy_25_sk_25[11]: - 5.09765625 frac[11,25] + mem[11,25] >= 0
 accuracy_25_sk_25[12]: - 5.09765625 frac[12,25] + mem[12,25] >= 0
 accuracy_25_sk_25[13]: - 5.09765625 frac[13,25] + mem[13,25] >= 0
 accuracy_25_sk_25[14]: - 5.09765625 frac[14,25] + mem[14,25] >= 0
 accuracy_26_sk_26[0]: - 5.09765625 frac[0,26] + mem[0,26] >= 0
 accuracy_26_sk_26[1]: - 5.09765625 frac[1,26] + mem[1,26] >= 0
 accuracy_26_sk_26[2]: - 5.09765625 frac[2,26] + mem[2,26] >= 0
 accuracy_26_sk_26[3]: - 5.09765625 frac[3,26] + mem[3,26] >= 0
 accuracy_26_sk_26[4]: - 5.09765625 frac[4,26] + mem[4,26] >= 0
 accuracy_26_sk_26[5]: - 5.09765625 frac[5,26] + mem[5,26] >= 0
 accuracy_26_sk_26[6]: - 5.09765625 frac[6,26] + mem[6,26] >= 0
 accuracy_26_sk_26[7]: - 5.09765625 frac[7,26] + mem[7,26] >= 0
 accuracy_26_sk_26[8]: - 5.09765625 frac[8,26] + mem[8,26] >= 0
 accuracy_26_sk_26[9]: - 5.09765625 frac[9,26] + mem[9,26] >= 0
 accuracy_26_sk_26[10]: - 5.09765625 frac[10,26] + mem[10,26] >= 0
 accuracy_26_sk_26[11]: - 5.09765625 frac[11,26] + mem[11,26] >= 0
 accuracy_26_sk_26[12]: - 5.09765625 frac[12,26] + mem[12,26] >= 0
 accuracy_26_sk_26[13]: - 5.09765625 frac[13,26] + mem[13,26] >= 0
 accuracy_26_sk_26[14]: - 5.09765625 frac[14,26] + mem[14,26] >= 0
 accuracy_27_sk_27[0]: - 5.09765625 frac[0,27] + mem[0,27] >= 0
 accuracy_27_sk_27[1]: - 5.09765625 frac[1,27] + mem[1,27] >= 0
 accuracy_27_sk_27[2]: - 5.09765625 frac[2,27] + mem[2,27] >= 0
 accuracy_27_sk_27[3]: - 5.09765625 frac[3,27] + mem[3,27] >= 0
 accuracy_27_sk_27[4]: - 5.09765625 frac[4,27] + mem[4,27] >= 0
 accuracy_27_sk_27[5]: - 5.09765625 frac[5,27] + mem[5,27] >= 0
 accuracy_27_sk_27[6]: - 5.09765625 frac[6,27] + mem[6,27] >= 0
 accuracy_27_sk_27[7]: - 5.09765625 frac[7,27] + mem[7,27] >= 0
 accuracy_27_sk_27[8]: - 5.09765625 frac[8,27] + mem[8,27] >= 0
 accuracy_27_sk_27[9]: - 5.09765625 frac[9,27] + mem[9,27] >= 0
 accuracy_27_sk_27[10]: - 5.09765625 frac[10,27] + mem[10,27] >= 0
 accuracy_27_sk_27[11]: - 5.09765625 frac[11,27] + mem[11,27] >= 0
 accuracy_27_sk_27[12]: - 5.09765625 frac[12,27] + mem[12,27] >= 0
 accuracy_27_sk_27[13]: - 5.09765625 frac[13,27] + mem[13,27] >= 0
 accuracy_27_sk_27[14]: - 5.09765625 frac[14,27] + mem[14,27] >= 0
 accuracy_28_sk_28[0]: - 5.09765625 frac[0,28] + mem[0,28] >= 0
 accuracy_28_sk_28[1]: - 5.09765625 frac[1,28] + mem[1,28] >= 0
 accuracy_28_sk_28[2]: - 5.09765625 frac[2,28] + mem[2,28] >= 0
 accuracy_28_sk_28[3]: - 5.09765625 frac[3,28] + mem[3,28] >= 0
 accuracy_28_sk_28[4]: - 5.09765625 frac[4,28] + mem[4,28] >= 0
 accuracy_28_sk_28[5]: - 5.09765625 frac[5,28] + mem[5,28] >= 0
 accuracy_28_sk_28[6]: - 5.09765625 frac[6,28] + mem[6,28] >= 0
 accuracy_28_sk_28[7]: - 5.09765625 frac[7,28] + mem[7,28] >= 0
 accuracy_28_sk_28[8]: - 5.09765625 frac[8,28] + mem[8,28] >= 0
 accuracy_28_sk_28[9]: - 5.09765625 frac[9,28] + mem[9,28] >= 0
 accuracy_28_sk_28[10]: - 5.09765625 frac[10,28] + mem[10,28] >= 0
 accuracy_28_sk_28[11]: - 5.09765625 frac[11,28] + mem[11,28] >= 0
 accuracy_28_sk_28[12]: - 5.09765625 frac[12,28] + mem[12,28] >= 0
 accuracy_28_sk_28[13]: - 5.09765625 frac[13,28] + mem[13,28] >= 0
 accuracy_28_sk_28[14]: - 5.09765625 frac[14,28] + mem[14,28] >= 0
 accuracy_29_sk_29[0]: - 5.09765625 frac[0,29] + mem[0,29] >= 0
 accuracy_29_sk_29[1]: - 5.09765625 frac[1,29] + mem[1,29] >= 0
 accuracy_29_sk_29[2]: - 5.09765625 frac[2,29] + mem[2,29] >= 0
 accuracy_29_sk_29[3]: - 5.09765625 frac[3,29] + mem[3,29] >= 0
 accuracy_29_sk_29[4]: - 5.09765625 frac[4,29] + mem[4,29] >= 0
 accuracy_29_sk_29[5]: - 5.09765625 frac[5,29] + mem[5,29] >= 0
 accuracy_29_sk_29[6]: - 5.09765625 frac[6,29] + mem[6,29] >= 0
 accuracy_29_sk_29[7]: - 5.09765625 frac[7,29] + mem[7,29] >= 0
 accuracy_29_sk_29[8]: - 5.09765625 frac[8,29] + mem[8,29] >= 0
 accuracy_29_sk_29[9]: - 5.09765625 frac[9,29] + mem[9,29] >= 0
 accuracy_29_sk_29[10]: - 5.09765625 frac[10,29] + mem[10,29] >= 0
 accuracy_29_sk_29[11]: - 5.09765625 frac[11,29] + mem[11,29] >= 0
 accuracy_29_sk_29[12]: - 5.09765625 frac[12,29] + mem[12,29] >= 0
 accuracy_29_sk_29[13]: - 5.09765625 frac[13,29] + mem[13,29] >= 0
 accuracy_29_sk_29[14]: - 5.09765625 frac[14,29] + mem[14,29] >= 0
 accuracy_30_sk_30[0]: - 5.09765625 frac[0,30] + mem[0,30] >= 0
 accuracy_30_sk_30[1]: - 5.09765625 frac[1,30] + mem[1,30] >= 0
 accuracy_30_sk_30[2]: - 5.09765625 frac[2,30] + mem[2,30] >= 0
 accuracy_30_sk_30[3]: - 5.09765625 frac[3,30] + mem[3,30] >= 0
 accuracy_30_sk_30[4]: - 5.09765625 frac[4,30] + mem[4,30] >= 0
 accuracy_30_sk_30[5]: - 5.09765625 frac[5,30] + mem[5,30] >= 0
 accuracy_30_sk_30[6]: - 5.09765625 frac[6,30] + mem[6,30] >= 0
 accuracy_30_sk_30[7]: - 5.09765625 frac[7,30] + mem[7,30] >= 0
 accuracy_30_sk_30[8]: - 5.09765625 frac[8,30] + mem[8,30] >= 0
 accuracy_30_sk_30[9]: - 5.09765625 frac[9,30] + mem[9,30] >= 0
 accuracy_30_sk_30[10]: - 5.09765625 frac[10,30] + mem[10,30] >= 0
 accuracy_30_sk_30[11]: - 5.09765625 frac[11,30] + mem[11,30] >= 0
 accuracy_30_sk_30[12]: - 5.09765625 frac[12,30] + mem[12,30] >= 0
 accuracy_30_sk_30[13]: - 5.09765625 frac[13,30] + mem[13,30] >= 0
 accuracy_30_sk_30[14]: - 5.09765625 frac[14,30] + mem[14,30] >= 0
 accuracy_31_sk_31[0]: - 5.09765625 frac[0,31] + mem[0,31] >= 0
 accuracy_31_sk_31[1]: - 5.09765625 frac[1,31] + mem[1,31] >= 0
 accuracy_31_sk_31[2]: - 5.09765625 frac[2,31] + mem[2,31] >= 0
 accuracy_31_sk_31[3]: - 5.09765625 frac[3,31] + mem[3,31] >= 0
 accuracy_31_sk_31[4]: - 5.09765625 frac[4,31] + mem[4,31] >= 0
 accuracy_31_sk_31[5]: - 5.09765625 frac[5,31] + mem[5,31] >= 0
 accuracy_31_sk_31[6]: - 5.09765625 frac[6,31] + mem[6,31] >= 0
 accuracy_31_sk_31[7]: - 5.09765625 frac[7,31] + mem[7,31] >= 0
 accuracy_31_sk_31[8]: - 5.09765625 frac[8,31] + mem[8,31] >= 0
 accuracy_31_sk_31[9]: - 5.09765625 frac[9,31] + mem[9,31] >= 0
 accuracy_31_sk_31[10]: - 5.09765625 frac[10,31] + mem[10,31] >= 0
 accuracy_31_sk_31[11]: - 5.09765625 frac[11,31] + mem[11,31] >= 0
 accuracy_31_sk_31[12]: - 5.09765625 frac[12,31] + mem[12,31] >= 0
 accuracy_31_sk_31[13]: - 5.09765625 frac[13,31] + mem[13,31] >= 0
 accuracy_31_sk_31[14]: - 5.09765625 frac[14,31] + mem[14,31] >= 0
 accuracy_32_sk_32[0]: - 5.09765625 frac[0,32] + mem[0,32] >= 0
 accuracy_32_sk_32[1]: - 5.09765625 frac[1,32] + mem[1,32] >= 0
 accuracy_32_sk_32[2]: - 5.09765625 frac[2,32] + mem[2,32] >= 0
 accuracy_32_sk_32[3]: - 5.09765625 frac[3,32] + mem[3,32] >= 0
 accuracy_32_sk_32[4]: - 5.09765625 frac[4,32] + mem[4,32] >= 0
 accuracy_32_sk_32[5]: - 5.09765625 frac[5,32] + mem[5,32] >= 0
 accuracy_32_sk_32[6]: - 5.09765625 frac[6,32] + mem[6,32] >= 0
 accuracy_32_sk_32[7]: - 5.09765625 frac[7,32] + mem[7,32] >= 0
 accuracy_32_sk_32[8]: - 5.09765625 frac[8,32] + mem[8,32] >= 0
 accuracy_32_sk_32[9]: - 5.09765625 frac[9,32] + mem[9,32] >= 0
 accuracy_32_sk_32[10]: - 5.09765625 frac[10,32] + mem[10,32] >= 0
 accuracy_32_sk_32[11]: - 5.09765625 frac[11,32] + mem[11,32] >= 0
 accuracy_32_sk_32[12]: - 5.09765625 frac[12,32] + mem[12,32] >= 0
 accuracy_32_sk_32[13]: - 5.09765625 frac[13,32] + mem[13,32] >= 0
 accuracy_32_sk_32[14]: - 5.09765625 frac[14,32] + mem[14,32] >= 0
 accuracy_33_sk_33[0]: - 5.09765625 frac[0,33] + mem[0,33] >= 0
 accuracy_33_sk_33[1]: - 5.09765625 frac[1,33] + mem[1,33] >= 0
 accuracy_33_sk_33[2]: - 5.09765625 frac[2,33] + mem[2,33] >= 0
 accuracy_33_sk_33[3]: - 5.09765625 frac[3,33] + mem[3,33] >= 0
 accuracy_33_sk_33[4]: - 5.09765625 frac[4,33] + mem[4,33] >= 0
 accuracy_33_sk_33[5]: - 5.09765625 frac[5,33] + mem[5,33] >= 0
 accuracy_33_sk_33[6]: - 5.09765625 frac[6,33] + mem[6,33] >= 0
 accuracy_33_sk_33[7]: - 5.09765625 frac[7,33] + mem[7,33] >= 0
 accuracy_33_sk_33[8]: - 5.09765625 frac[8,33] + mem[8,33] >= 0
 accuracy_33_sk_33[9]: - 5.09765625 frac[9,33] + mem[9,33] >= 0
 accuracy_33_sk_33[10]: - 5.09765625 frac[10,33] + mem[10,33] >= 0
 accuracy_33_sk_33[11]: - 5.09765625 frac[11,33] + mem[11,33] >= 0
 accuracy_33_sk_33[12]: - 5.09765625 frac[12,33] + mem[12,33] >= 0
 accuracy_33_sk_33[13]: - 5.09765625 frac[13,33] + mem[13,33] >= 0
 accuracy_33_sk_33[14]: - 5.09765625 frac[14,33] + mem[14,33] >= 0
 accuracy_34_sk_34[0]: - 5.09765625 frac[0,34] + mem[0,34] >= 0
 accuracy_34_sk_34[1]: - 5.09765625 frac[1,34] + mem[1,34] >= 0
 accuracy_34_sk_34[2]: - 5.09765625 frac[2,34] + mem[2,34] >= 0
 accuracy_34_sk_34[3]: - 5.09765625 frac[3,34] + mem[3,34] >= 0
 accuracy_34_sk_34[4]: - 5.09765625 frac[4,34] + mem[4,34] >= 0
 accuracy_34_sk_34[5]: - 5.09765625 frac[5,34] + mem[5,34] >= 0
 accuracy_34_sk_34[6]: - 5.09765625 frac[6,34] + mem[6,34] >= 0
 accuracy_34_sk_34[7]: - 5.09765625 frac[7,34] + mem[7,34] >= 0
 accuracy_34_sk_34[8]: - 5.09765625 frac[8,34] + mem[8,34] >= 0
 accuracy_34_sk_34[9]: - 5.09765625 frac[9,34] + mem[9,34] >= 0
 accuracy_34_sk_34[10]: - 5.09765625 frac[10,34] + mem[10,34] >= 0
 accuracy_34_sk_34[11]: - 5.09765625 frac[11,34] + mem[11,34] >= 0
 accuracy_34_sk_34[12]: - 5.09765625 frac[12,34] + mem[12,34] >= 0
 accuracy_34_sk_34[13]: - 5.09765625 frac[13,34] + mem[13,34] >= 0
 accuracy_34_sk_34[14]: - 5.09765625 frac[14,34] + mem[14,34] >= 0
 accuracy_35_sk_35[0]: - 5.09765625 frac[0,35] + mem[0,35] >= 0
 accuracy_35_sk_35[1]: - 5.09765625 frac[1,35] + mem[1,35] >= 0
 accuracy_35_sk_35[2]: - 5.09765625 frac[2,35] + mem[2,35] >= 0
 accuracy_35_sk_35[3]: - 5.09765625 frac[3,35] + mem[3,35] >= 0
 accuracy_35_sk_35[4]: - 5.09765625 frac[4,35] + mem[4,35] >= 0
 accuracy_35_sk_35[5]: - 5.09765625 frac[5,35] + mem[5,35] >= 0
 accuracy_35_sk_35[6]: - 5.09765625 frac[6,35] + mem[6,35] >= 0
 accuracy_35_sk_35[7]: - 5.09765625 frac[7,35] + mem[7,35] >= 0
 accuracy_35_sk_35[8]: - 5.09765625 frac[8,35] + mem[8,35] >= 0
 accuracy_35_sk_35[9]: - 5.09765625 frac[9,35] + mem[9,35] >= 0
 accuracy_35_sk_35[10]: - 5.09765625 frac[10,35] + mem[10,35] >= 0
 accuracy_35_sk_35[11]: - 5.09765625 frac[11,35] + mem[11,35] >= 0
 accuracy_35_sk_35[12]: - 5.09765625 frac[12,35] + mem[12,35] >= 0
 accuracy_35_sk_35[13]: - 5.09765625 frac[13,35] + mem[13,35] >= 0
 accuracy_35_sk_35[14]: - 5.09765625 frac[14,35] + mem[14,35] >= 0
 accuracy_36_sk_36[0]: - 5.09765625 frac[0,36] + mem[0,36] >= 0
 accuracy_36_sk_36[1]: - 5.09765625 frac[1,36] + mem[1,36] >= 0
 accuracy_36_sk_36[2]: - 5.09765625 frac[2,36] + mem[2,36] >= 0
 accuracy_36_sk_36[3]: - 5.09765625 frac[3,36] + mem[3,36] >= 0
 accuracy_36_sk_36[4]: - 5.09765625 frac[4,36] + mem[4,36] >= 0
 accuracy_36_sk_36[5]: - 5.09765625 frac[5,36] + mem[5,36] >= 0
 accuracy_36_sk_36[6]: - 5.09765625 frac[6,36] + mem[6,36] >= 0
 accuracy_36_sk_36[7]: - 5.09765625 frac[7,36] + mem[7,36] >= 0
 accuracy_36_sk_36[8]: - 5.09765625 frac[8,36] + mem[8,36] >= 0
 accuracy_36_sk_36[9]: - 5.09765625 frac[9,36] + mem[9,36] >= 0
 accuracy_36_sk_36[10]: - 5.09765625 frac[10,36] + mem[10,36] >= 0
 accuracy_36_sk_36[11]: - 5.09765625 frac[11,36] + mem[11,36] >= 0
 accuracy_36_sk_36[12]: - 5.09765625 frac[12,36] + mem[12,36] >= 0
 accuracy_36_sk_36[13]: - 5.09765625 frac[13,36] + mem[13,36] >= 0
 accuracy_36_sk_36[14]: - 5.09765625 frac[14,36] + mem[14,36] >= 0
 accuracy_37_sk_37[0]: - 5.09765625 frac[0,37] + mem[0,37] >= 0
 accuracy_37_sk_37[1]: - 5.09765625 frac[1,37] + mem[1,37] >= 0
 accuracy_37_sk_37[2]: - 5.09765625 frac[2,37] + mem[2,37] >= 0
 accuracy_37_sk_37[3]: - 5.09765625 frac[3,37] + mem[3,37] >= 0
 accuracy_37_sk_37[4]: - 5.09765625 frac[4,37] + mem[4,37] >= 0
 accuracy_37_sk_37[5]: - 5.09765625 frac[5,37] + mem[5,37] >= 0
 accuracy_37_sk_37[6]: - 5.09765625 frac[6,37] + mem[6,37] >= 0
 accuracy_37_sk_37[7]: - 5.09765625 frac[7,37] + mem[7,37] >= 0
 accuracy_37_sk_37[8]: - 5.09765625 frac[8,37] + mem[8,37] >= 0
 accuracy_37_sk_37[9]: - 5.09765625 frac[9,37] + mem[9,37] >= 0
 accuracy_37_sk_37[10]: - 5.09765625 frac[10,37] + mem[10,37] >= 0
 accuracy_37_sk_37[11]: - 5.09765625 frac[11,37] + mem[11,37] >= 0
 accuracy_37_sk_37[12]: - 5.09765625 frac[12,37] + mem[12,37] >= 0
 accuracy_37_sk_37[13]: - 5.09765625 frac[13,37] + mem[13,37] >= 0
 accuracy_37_sk_37[14]: - 5.09765625 frac[14,37] + mem[14,37] >= 0
 accuracy_38_sk_38[0]: - 5.09765625 frac[0,38] + mem[0,38] >= 0
 accuracy_38_sk_38[1]: - 5.09765625 frac[1,38] + mem[1,38] >= 0
 accuracy_38_sk_38[2]: - 5.09765625 frac[2,38] + mem[2,38] >= 0
 accuracy_38_sk_38[3]: - 5.09765625 frac[3,38] + mem[3,38] >= 0
 accuracy_38_sk_38[4]: - 5.09765625 frac[4,38] + mem[4,38] >= 0
 accuracy_38_sk_38[5]: - 5.09765625 frac[5,38] + mem[5,38] >= 0
 accuracy_38_sk_38[6]: - 5.09765625 frac[6,38] + mem[6,38] >= 0
 accuracy_38_sk_38[7]: - 5.09765625 frac[7,38] + mem[7,38] >= 0
 accuracy_38_sk_38[8]: - 5.09765625 frac[8,38] + mem[8,38] >= 0
 accuracy_38_sk_38[9]: - 5.09765625 frac[9,38] + mem[9,38] >= 0
 accuracy_38_sk_38[10]: - 5.09765625 frac[10,38] + mem[10,38] >= 0
 accuracy_38_sk_38[11]: - 5.09765625 frac[11,38] + mem[11,38] >= 0
 accuracy_38_sk_38[12]: - 5.09765625 frac[12,38] + mem[12,38] >= 0
 accuracy_38_sk_38[13]: - 5.09765625 frac[13,38] + mem[13,38] >= 0
 accuracy_38_sk_38[14]: - 5.09765625 frac[14,38] + mem[14,38] >= 0
 accuracy_39_sk_39[0]: - 5.09765625 frac[0,39] + mem[0,39] >= 0
 accuracy_39_sk_39[1]: - 5.09765625 frac[1,39] + mem[1,39] >= 0
 accuracy_39_sk_39[2]: - 5.09765625 frac[2,39] + mem[2,39] >= 0
 accuracy_39_sk_39[3]: - 5.09765625 frac[3,39] + mem[3,39] >= 0
 accuracy_39_sk_39[4]: - 5.09765625 frac[4,39] + mem[4,39] >= 0
 accuracy_39_sk_39[5]: - 5.09765625 frac[5,39] + mem[5,39] >= 0
 accuracy_39_sk_39[6]: - 5.09765625 frac[6,39] + mem[6,39] >= 0
 accuracy_39_sk_39[7]: - 5.09765625 frac[7,39] + mem[7,39] >= 0
 accuracy_39_sk_39[8]: - 5.09765625 frac[8,39] + mem[8,39] >= 0
 accuracy_39_sk_39[9]: - 5.09765625 frac[9,39] + mem[9,39] >= 0
 accuracy_39_sk_39[10]: - 5.09765625 frac[10,39] + mem[10,39] >= 0
 accuracy_39_sk_39[11]: - 5.09765625 frac[11,39] + mem[11,39] >= 0
 accuracy_39_sk_39[12]: - 5.09765625 frac[12,39] + mem[12,39] >= 0
 accuracy_39_sk_39[13]: - 5.09765625 frac[13,39] + mem[13,39] >= 0
 accuracy_39_sk_39[14]: - 5.09765625 frac[14,39] + mem[14,39] >= 0
 accuracy_40_sk_40[0]: - 5.09765625 frac[0,40] + mem[0,40] >= 0
 accuracy_40_sk_40[1]: - 5.09765625 frac[1,40] + mem[1,40] >= 0
 accuracy_40_sk_40[2]: - 5.09765625 frac[2,40] + mem[2,40] >= 0
 accuracy_40_sk_40[3]: - 5.09765625 frac[3,40] + mem[3,40] >= 0
 accuracy_40_sk_40[4]: - 5.09765625 frac[4,40] + mem[4,40] >= 0
 accuracy_40_sk_40[5]: - 5.09765625 frac[5,40] + mem[5,40] >= 0
 accuracy_40_sk_40[6]: - 5.09765625 frac[6,40] + mem[6,40] >= 0
 accuracy_40_sk_40[7]: - 5.09765625 frac[7,40] + mem[7,40] >= 0
 accuracy_40_sk_40[8]: - 5.09765625 frac[8,40] + mem[8,40] >= 0
 accuracy_40_sk_40[9]: - 5.09765625 frac[9,40] + mem[9,40] >= 0
 accuracy_40_sk_40[10]: - 5.09765625 frac[10,40] + mem[10,40] >= 0
 accuracy_40_sk_40[11]: - 5.09765625 frac[11,40] + mem[11,40] >= 0
 accuracy_40_sk_40[12]: - 5.09765625 frac[12,40] + mem[12,40] >= 0
 accuracy_40_sk_40[13]: - 5.09765625 frac[13,40] + mem[13,40] >= 0
 accuracy_40_sk_40[14]: - 5.09765625 frac[14,40] + mem[14,40] >= 0
 accuracy_41_sk_41[0]: - 5.09765625 frac[0,41] + mem[0,41] >= 0
 accuracy_41_sk_41[1]: - 5.09765625 frac[1,41] + mem[1,41] >= 0
 accuracy_41_sk_41[2]: - 5.09765625 frac[2,41] + mem[2,41] >= 0
 accuracy_41_sk_41[3]: - 5.09765625 frac[3,41] + mem[3,41] >= 0
 accuracy_41_sk_41[4]: - 5.09765625 frac[4,41] + mem[4,41] >= 0
 accuracy_41_sk_41[5]: - 5.09765625 frac[5,41] + mem[5,41] >= 0
 accuracy_41_sk_41[6]: - 5.09765625 frac[6,41] + mem[6,41] >= 0
 accuracy_41_sk_41[7]: - 5.09765625 frac[7,41] + mem[7,41] >= 0
 accuracy_41_sk_41[8]: - 5.09765625 frac[8,41] + mem[8,41] >= 0
 accuracy_41_sk_41[9]: - 5.09765625 frac[9,41] + mem[9,41] >= 0
 accuracy_41_sk_41[10]: - 5.09765625 frac[10,41] + mem[10,41] >= 0
 accuracy_41_sk_41[11]: - 5.09765625 frac[11,41] + mem[11,41] >= 0
 accuracy_41_sk_41[12]: - 5.09765625 frac[12,41] + mem[12,41] >= 0
 accuracy_41_sk_41[13]: - 5.09765625 frac[13,41] + mem[13,41] >= 0
 accuracy_41_sk_41[14]: - 5.09765625 frac[14,41] + mem[14,41] >= 0
 accuracy_42_sk_42[0]: - 5.09765625 frac[0,42] + mem[0,42] >= 0
 accuracy_42_sk_42[1]: - 5.09765625 frac[1,42] + mem[1,42] >= 0
 accuracy_42_sk_42[2]: - 5.09765625 frac[2,42] + mem[2,42] >= 0
 accuracy_42_sk_42[3]: - 5.09765625 frac[3,42] + mem[3,42] >= 0
 accuracy_42_sk_42[4]: - 5.09765625 frac[4,42] + mem[4,42] >= 0
 accuracy_42_sk_42[5]: - 5.09765625 frac[5,42] + mem[5,42] >= 0
 accuracy_42_sk_42[6]: - 5.09765625 frac[6,42] + mem[6,42] >= 0
 accuracy_42_sk_42[7]: - 5.09765625 frac[7,42] + mem[7,42] >= 0
 accuracy_42_sk_42[8]: - 5.09765625 frac[8,42] + mem[8,42] >= 0
 accuracy_42_sk_42[9]: - 5.09765625 frac[9,42] + mem[9,42] >= 0
 accuracy_42_sk_42[10]: - 5.09765625 frac[10,42] + mem[10,42] >= 0
 accuracy_42_sk_42[11]: - 5.09765625 frac[11,42] + mem[11,42] >= 0
 accuracy_42_sk_42[12]: - 5.09765625 frac[12,42] + mem[12,42] >= 0
 accuracy_42_sk_42[13]: - 5.09765625 frac[13,42] + mem[13,42] >= 0
 accuracy_42_sk_42[14]: - 5.09765625 frac[14,42] + mem[14,42] >= 0
 accuracy_43_sk_43[0]: - 5.09765625 frac[0,43] + mem[0,43] >= 0
 accuracy_43_sk_43[1]: - 5.09765625 frac[1,43] + mem[1,43] >= 0
 accuracy_43_sk_43[2]: - 5.09765625 frac[2,43] + mem[2,43] >= 0
 accuracy_43_sk_43[3]: - 5.09765625 frac[3,43] + mem[3,43] >= 0
 accuracy_43_sk_43[4]: - 5.09765625 frac[4,43] + mem[4,43] >= 0
 accuracy_43_sk_43[5]: - 5.09765625 frac[5,43] + mem[5,43] >= 0
 accuracy_43_sk_43[6]: - 5.09765625 frac[6,43] + mem[6,43] >= 0
 accuracy_43_sk_43[7]: - 5.09765625 frac[7,43] + mem[7,43] >= 0
 accuracy_43_sk_43[8]: - 5.09765625 frac[8,43] + mem[8,43] >= 0
 accuracy_43_sk_43[9]: - 5.09765625 frac[9,43] + mem[9,43] >= 0
 accuracy_43_sk_43[10]: - 5.09765625 frac[10,43] + mem[10,43] >= 0
 accuracy_43_sk_43[11]: - 5.09765625 frac[11,43] + mem[11,43] >= 0
 accuracy_43_sk_43[12]: - 5.09765625 frac[12,43] + mem[12,43] >= 0
 accuracy_43_sk_43[13]: - 5.09765625 frac[13,43] + mem[13,43] >= 0
 accuracy_43_sk_43[14]: - 5.09765625 frac[14,43] + mem[14,43] >= 0
 accuracy_44_sk_44[0]: - 5.09765625 frac[0,44] + mem[0,44] >= 0
 accuracy_44_sk_44[1]: - 5.09765625 frac[1,44] + mem[1,44] >= 0
 accuracy_44_sk_44[2]: - 5.09765625 frac[2,44] + mem[2,44] >= 0
 accuracy_44_sk_44[3]: - 5.09765625 frac[3,44] + mem[3,44] >= 0
 accuracy_44_sk_44[4]: - 5.09765625 frac[4,44] + mem[4,44] >= 0
 accuracy_44_sk_44[5]: - 5.09765625 frac[5,44] + mem[5,44] >= 0
 accuracy_44_sk_44[6]: - 5.09765625 frac[6,44] + mem[6,44] >= 0
 accuracy_44_sk_44[7]: - 5.09765625 frac[7,44] + mem[7,44] >= 0
 accuracy_44_sk_44[8]: - 5.09765625 frac[8,44] + mem[8,44] >= 0
 accuracy_44_sk_44[9]: - 5.09765625 frac[9,44] + mem[9,44] >= 0
 accuracy_44_sk_44[10]: - 5.09765625 frac[10,44] + mem[10,44] >= 0
 accuracy_44_sk_44[11]: - 5.09765625 frac[11,44] + mem[11,44] >= 0
 accuracy_44_sk_44[12]: - 5.09765625 frac[12,44] + mem[12,44] >= 0
 accuracy_44_sk_44[13]: - 5.09765625 frac[13,44] + mem[13,44] >= 0
 accuracy_44_sk_44[14]: - 5.09765625 frac[14,44] + mem[14,44] >= 0
 accuracy_45_sk_45[0]: - 5.09765625 frac[0,45] + mem[0,45] >= 0
 accuracy_45_sk_45[1]: - 5.09765625 frac[1,45] + mem[1,45] >= 0
 accuracy_45_sk_45[2]: - 5.09765625 frac[2,45] + mem[2,45] >= 0
 accuracy_45_sk_45[3]: - 5.09765625 frac[3,45] + mem[3,45] >= 0
 accuracy_45_sk_45[4]: - 5.09765625 frac[4,45] + mem[4,45] >= 0
 accuracy_45_sk_45[5]: - 5.09765625 frac[5,45] + mem[5,45] >= 0
 accuracy_45_sk_45[6]: - 5.09765625 frac[6,45] + mem[6,45] >= 0
 accuracy_45_sk_45[7]: - 5.09765625 frac[7,45] + mem[7,45] >= 0
 accuracy_45_sk_45[8]: - 5.09765625 frac[8,45] + mem[8,45] >= 0
 accuracy_45_sk_45[9]: - 5.09765625 frac[9,45] + mem[9,45] >= 0
 accuracy_45_sk_45[10]: - 5.09765625 frac[10,45] + mem[10,45] >= 0
 accuracy_45_sk_45[11]: - 5.09765625 frac[11,45] + mem[11,45] >= 0
 accuracy_45_sk_45[12]: - 5.09765625 frac[12,45] + mem[12,45] >= 0
 accuracy_45_sk_45[13]: - 5.09765625 frac[13,45] + mem[13,45] >= 0
 accuracy_45_sk_45[14]: - 5.09765625 frac[14,45] + mem[14,45] >= 0
 accuracy_46_sk_46[0]: - 5.09765625 frac[0,46] + mem[0,46] >= 0
 accuracy_46_sk_46[1]: - 5.09765625 frac[1,46] + mem[1,46] >= 0
 accuracy_46_sk_46[2]: - 5.09765625 frac[2,46] + mem[2,46] >= 0
 accuracy_46_sk_46[3]: - 5.09765625 frac[3,46] + mem[3,46] >= 0
 accuracy_46_sk_46[4]: - 5.09765625 frac[4,46] + mem[4,46] >= 0
 accuracy_46_sk_46[5]: - 5.09765625 frac[5,46] + mem[5,46] >= 0
 accuracy_46_sk_46[6]: - 5.09765625 frac[6,46] + mem[6,46] >= 0
 accuracy_46_sk_46[7]: - 5.09765625 frac[7,46] + mem[7,46] >= 0
 accuracy_46_sk_46[8]: - 5.09765625 frac[8,46] + mem[8,46] >= 0
 accuracy_46_sk_46[9]: - 5.09765625 frac[9,46] + mem[9,46] >= 0
 accuracy_46_sk_46[10]: - 5.09765625 frac[10,46] + mem[10,46] >= 0
 accuracy_46_sk_46[11]: - 5.09765625 frac[11,46] + mem[11,46] >= 0
 accuracy_46_sk_46[12]: - 5.09765625 frac[12,46] + mem[12,46] >= 0
 accuracy_46_sk_46[13]: - 5.09765625 frac[13,46] + mem[13,46] >= 0
 accuracy_46_sk_46[14]: - 5.09765625 frac[14,46] + mem[14,46] >= 0
 accuracy_47_sk_47[0]: - 5.09765625 frac[0,47] + mem[0,47] >= 0
 accuracy_47_sk_47[1]: - 5.09765625 frac[1,47] + mem[1,47] >= 0
 accuracy_47_sk_47[2]: - 5.09765625 frac[2,47] + mem[2,47] >= 0
 accuracy_47_sk_47[3]: - 5.09765625 frac[3,47] + mem[3,47] >= 0
 accuracy_47_sk_47[4]: - 5.09765625 frac[4,47] + mem[4,47] >= 0
 accuracy_47_sk_47[5]: - 5.09765625 frac[5,47] + mem[5,47] >= 0
 accuracy_47_sk_47[6]: - 5.09765625 frac[6,47] + mem[6,47] >= 0
 accuracy_47_sk_47[7]: - 5.09765625 frac[7,47] + mem[7,47] >= 0
 accuracy_47_sk_47[8]: - 5.09765625 frac[8,47] + mem[8,47] >= 0
 accuracy_47_sk_47[9]: - 5.09765625 frac[9,47] + mem[9,47] >= 0
 accuracy_47_sk_47[10]: - 5.09765625 frac[10,47] + mem[10,47] >= 0
 accuracy_47_sk_47[11]: - 5.09765625 frac[11,47] + mem[11,47] >= 0
 accuracy_47_sk_47[12]: - 5.09765625 frac[12,47] + mem[12,47] >= 0
 accuracy_47_sk_47[13]: - 5.09765625 frac[13,47] + mem[13,47] >= 0
 accuracy_47_sk_47[14]: - 5.09765625 frac[14,47] + mem[14,47] >= 0
 accuracy_48_sk_48[0]: - 5.09765625 frac[0,48] + mem[0,48] >= 0
 accuracy_48_sk_48[1]: - 5.09765625 frac[1,48] + mem[1,48] >= 0
 accuracy_48_sk_48[2]: - 5.09765625 frac[2,48] + mem[2,48] >= 0
 accuracy_48_sk_48[3]: - 5.09765625 frac[3,48] + mem[3,48] >= 0
 accuracy_48_sk_48[4]: - 5.09765625 frac[4,48] + mem[4,48] >= 0
 accuracy_48_sk_48[5]: - 5.09765625 frac[5,48] + mem[5,48] >= 0
 accuracy_48_sk_48[6]: - 5.09765625 frac[6,48] + mem[6,48] >= 0
 accuracy_48_sk_48[7]: - 5.09765625 frac[7,48] + mem[7,48] >= 0
 accuracy_48_sk_48[8]: - 5.09765625 frac[8,48] + mem[8,48] >= 0
 accuracy_48_sk_48[9]: - 5.09765625 frac[9,48] + mem[9,48] >= 0
 accuracy_48_sk_48[10]: - 5.09765625 frac[10,48] + mem[10,48] >= 0
 accuracy_48_sk_48[11]: - 5.09765625 frac[11,48] + mem[11,48] >= 0
 accuracy_48_sk_48[12]: - 5.09765625 frac[12,48] + mem[12,48] >= 0
 accuracy_48_sk_48[13]: - 5.09765625 frac[13,48] + mem[13,48] >= 0
 accuracy_48_sk_48[14]: - 5.09765625 frac[14,48] + mem[14,48] >= 0
 accuracy_49_sk_49[0]: - 5.09765625 frac[0,49] + mem[0,49] >= 0
 accuracy_49_sk_49[1]: - 5.09765625 frac[1,49] + mem[1,49] >= 0
 accuracy_49_sk_49[2]: - 5.09765625 frac[2,49] + mem[2,49] >= 0
 accuracy_49_sk_49[3]: - 5.09765625 frac[3,49] + mem[3,49] >= 0
 accuracy_49_sk_49[4]: - 5.09765625 frac[4,49] + mem[4,49] >= 0
 accuracy_49_sk_49[5]: - 5.09765625 frac[5,49] + mem[5,49] >= 0
 accuracy_49_sk_49[6]: - 5.09765625 frac[6,49] + mem[6,49] >= 0
 accuracy_49_sk_49[7]: - 5.09765625 frac[7,49] + mem[7,49] >= 0
 accuracy_49_sk_49[8]: - 5.09765625 frac[8,49] + mem[8,49] >= 0
 accuracy_49_sk_49[9]: - 5.09765625 frac[9,49] + mem[9,49] >= 0
 accuracy_49_sk_49[10]: - 5.09765625 frac[10,49] + mem[10,49] >= 0
 accuracy_49_sk_49[11]: - 5.09765625 frac[11,49] + mem[11,49] >= 0
 accuracy_49_sk_49[12]: - 5.09765625 frac[12,49] + mem[12,49] >= 0
 accuracy_49_sk_49[13]: - 5.09765625 frac[13,49] + mem[13,49] >= 0
 accuracy_49_sk_49[14]: - 5.09765625 frac[14,49] + mem[14,49] >= 0
 accuracy_50_sk_50[0]: - 5.09765625 frac[0,50] + mem[0,50] >= 0
 accuracy_50_sk_50[1]: - 5.09765625 frac[1,50] + mem[1,50] >= 0
 accuracy_50_sk_50[2]: - 5.09765625 frac[2,50] + mem[2,50] >= 0
 accuracy_50_sk_50[3]: - 5.09765625 frac[3,50] + mem[3,50] >= 0
 accuracy_50_sk_50[4]: - 5.09765625 frac[4,50] + mem[4,50] >= 0
 accuracy_50_sk_50[5]: - 5.09765625 frac[5,50] + mem[5,50] >= 0
 accuracy_50_sk_50[6]: - 5.09765625 frac[6,50] + mem[6,50] >= 0
 accuracy_50_sk_50[7]: - 5.09765625 frac[7,50] + mem[7,50] >= 0
 accuracy_50_sk_50[8]: - 5.09765625 frac[8,50] + mem[8,50] >= 0
 accuracy_50_sk_50[9]: - 5.09765625 frac[9,50] + mem[9,50] >= 0
 accuracy_50_sk_50[10]: - 5.09765625 frac[10,50] + mem[10,50] >= 0
 accuracy_50_sk_50[11]: - 5.09765625 frac[11,50] + mem[11,50] >= 0
 accuracy_50_sk_50[12]: - 5.09765625 frac[12,50] + mem[12,50] >= 0
 accuracy_50_sk_50[13]: - 5.09765625 frac[13,50] + mem[13,50] >= 0
 accuracy_50_sk_50[14]: - 5.09765625 frac[14,50] + mem[14,50] >= 0
 accuracy_51_sk_51[0]: - 5.09765625 frac[0,51] + mem[0,51] >= 0
 accuracy_51_sk_51[1]: - 5.09765625 frac[1,51] + mem[1,51] >= 0
 accuracy_51_sk_51[2]: - 5.09765625 frac[2,51] + mem[2,51] >= 0
 accuracy_51_sk_51[3]: - 5.09765625 frac[3,51] + mem[3,51] >= 0
 accuracy_51_sk_51[4]: - 5.09765625 frac[4,51] + mem[4,51] >= 0
 accuracy_51_sk_51[5]: - 5.09765625 frac[5,51] + mem[5,51] >= 0
 accuracy_51_sk_51[6]: - 5.09765625 frac[6,51] + mem[6,51] >= 0
 accuracy_51_sk_51[7]: - 5.09765625 frac[7,51] + mem[7,51] >= 0
 accuracy_51_sk_51[8]: - 5.09765625 frac[8,51] + mem[8,51] >= 0
 accuracy_51_sk_51[9]: - 5.09765625 frac[9,51] + mem[9,51] >= 0
 accuracy_51_sk_51[10]: - 5.09765625 frac[10,51] + mem[10,51] >= 0
 accuracy_51_sk_51[11]: - 5.09765625 frac[11,51] + mem[11,51] >= 0
 accuracy_51_sk_51[12]: - 5.09765625 frac[12,51] + mem[12,51] >= 0
 accuracy_51_sk_51[13]: - 5.09765625 frac[13,51] + mem[13,51] >= 0
 accuracy_51_sk_51[14]: - 5.09765625 frac[14,51] + mem[14,51] >= 0
 accuracy_52_sk_52[0]: - 5.09765625 frac[0,52] + mem[0,52] >= 0
 accuracy_52_sk_52[1]: - 5.09765625 frac[1,52] + mem[1,52] >= 0
 accuracy_52_sk_52[2]: - 5.09765625 frac[2,52] + mem[2,52] >= 0
 accuracy_52_sk_52[3]: - 5.09765625 frac[3,52] + mem[3,52] >= 0
 accuracy_52_sk_52[4]: - 5.09765625 frac[4,52] + mem[4,52] >= 0
 accuracy_52_sk_52[5]: - 5.09765625 frac[5,52] + mem[5,52] >= 0
 accuracy_52_sk_52[6]: - 5.09765625 frac[6,52] + mem[6,52] >= 0
 accuracy_52_sk_52[7]: - 5.09765625 frac[7,52] + mem[7,52] >= 0
 accuracy_52_sk_52[8]: - 5.09765625 frac[8,52] + mem[8,52] >= 0
 accuracy_52_sk_52[9]: - 5.09765625 frac[9,52] + mem[9,52] >= 0
 accuracy_52_sk_52[10]: - 5.09765625 frac[10,52] + mem[10,52] >= 0
 accuracy_52_sk_52[11]: - 5.09765625 frac[11,52] + mem[11,52] >= 0
 accuracy_52_sk_52[12]: - 5.09765625 frac[12,52] + mem[12,52] >= 0
 accuracy_52_sk_52[13]: - 5.09765625 frac[13,52] + mem[13,52] >= 0
 accuracy_52_sk_52[14]: - 5.09765625 frac[14,52] + mem[14,52] >= 0
 accuracy_53_sk_53[0]: - 5.09765625 frac[0,53] + mem[0,53] >= 0
 accuracy_53_sk_53[1]: - 5.09765625 frac[1,53] + mem[1,53] >= 0
 accuracy_53_sk_53[2]: - 5.09765625 frac[2,53] + mem[2,53] >= 0
 accuracy_53_sk_53[3]: - 5.09765625 frac[3,53] + mem[3,53] >= 0
 accuracy_53_sk_53[4]: - 5.09765625 frac[4,53] + mem[4,53] >= 0
 accuracy_53_sk_53[5]: - 5.09765625 frac[5,53] + mem[5,53] >= 0
 accuracy_53_sk_53[6]: - 5.09765625 frac[6,53] + mem[6,53] >= 0
 accuracy_53_sk_53[7]: - 5.09765625 frac[7,53] + mem[7,53] >= 0
 accuracy_53_sk_53[8]: - 5.09765625 frac[8,53] + mem[8,53] >= 0
 accuracy_53_sk_53[9]: - 5.09765625 frac[9,53] + mem[9,53] >= 0
 accuracy_53_sk_53[10]: - 5.09765625 frac[10,53] + mem[10,53] >= 0
 accuracy_53_sk_53[11]: - 5.09765625 frac[11,53] + mem[11,53] >= 0
 accuracy_53_sk_53[12]: - 5.09765625 frac[12,53] + mem[12,53] >= 0
 accuracy_53_sk_53[13]: - 5.09765625 frac[13,53] + mem[13,53] >= 0
 accuracy_53_sk_53[14]: - 5.09765625 frac[14,53] + mem[14,53] >= 0
 accuracy_54_sk_54[0]: - 5.09765625 frac[0,54] + mem[0,54] >= 0
 accuracy_54_sk_54[1]: - 5.09765625 frac[1,54] + mem[1,54] >= 0
 accuracy_54_sk_54[2]: - 5.09765625 frac[2,54] + mem[2,54] >= 0
 accuracy_54_sk_54[3]: - 5.09765625 frac[3,54] + mem[3,54] >= 0
 accuracy_54_sk_54[4]: - 5.09765625 frac[4,54] + mem[4,54] >= 0
 accuracy_54_sk_54[5]: - 5.09765625 frac[5,54] + mem[5,54] >= 0
 accuracy_54_sk_54[6]: - 5.09765625 frac[6,54] + mem[6,54] >= 0
 accuracy_54_sk_54[7]: - 5.09765625 frac[7,54] + mem[7,54] >= 0
 accuracy_54_sk_54[8]: - 5.09765625 frac[8,54] + mem[8,54] >= 0
 accuracy_54_sk_54[9]: - 5.09765625 frac[9,54] + mem[9,54] >= 0
 accuracy_54_sk_54[10]: - 5.09765625 frac[10,54] + mem[10,54] >= 0
 accuracy_54_sk_54[11]: - 5.09765625 frac[11,54] + mem[11,54] >= 0
 accuracy_54_sk_54[12]: - 5.09765625 frac[12,54] + mem[12,54] >= 0
 accuracy_54_sk_54[13]: - 5.09765625 frac[13,54] + mem[13,54] >= 0
 accuracy_54_sk_54[14]: - 5.09765625 frac[14,54] + mem[14,54] >= 0
 accuracy_55_sk_55[0]: - 5.09765625 frac[0,55] + mem[0,55] >= 0
 accuracy_55_sk_55[1]: - 5.09765625 frac[1,55] + mem[1,55] >= 0
 accuracy_55_sk_55[2]: - 5.09765625 frac[2,55] + mem[2,55] >= 0
 accuracy_55_sk_55[3]: - 5.09765625 frac[3,55] + mem[3,55] >= 0
 accuracy_55_sk_55[4]: - 5.09765625 frac[4,55] + mem[4,55] >= 0
 accuracy_55_sk_55[5]: - 5.09765625 frac[5,55] + mem[5,55] >= 0
 accuracy_55_sk_55[6]: - 5.09765625 frac[6,55] + mem[6,55] >= 0
 accuracy_55_sk_55[7]: - 5.09765625 frac[7,55] + mem[7,55] >= 0
 accuracy_55_sk_55[8]: - 5.09765625 frac[8,55] + mem[8,55] >= 0
 accuracy_55_sk_55[9]: - 5.09765625 frac[9,55] + mem[9,55] >= 0
 accuracy_55_sk_55[10]: - 5.09765625 frac[10,55] + mem[10,55] >= 0
 accuracy_55_sk_55[11]: - 5.09765625 frac[11,55] + mem[11,55] >= 0
 accuracy_55_sk_55[12]: - 5.09765625 frac[12,55] + mem[12,55] >= 0
 accuracy_55_sk_55[13]: - 5.09765625 frac[13,55] + mem[13,55] >= 0
 accuracy_55_sk_55[14]: - 5.09765625 frac[14,55] + mem[14,55] >= 0
 accuracy_56_sk_56[0]: - 5.09765625 frac[0,56] + mem[0,56] >= 0
 accuracy_56_sk_56[1]: - 5.09765625 frac[1,56] + mem[1,56] >= 0
 accuracy_56_sk_56[2]: - 5.09765625 frac[2,56] + mem[2,56] >= 0
 accuracy_56_sk_56[3]: - 5.09765625 frac[3,56] + mem[3,56] >= 0
 accuracy_56_sk_56[4]: - 5.09765625 frac[4,56] + mem[4,56] >= 0
 accuracy_56_sk_56[5]: - 5.09765625 frac[5,56] + mem[5,56] >= 0
 accuracy_56_sk_56[6]: - 5.09765625 frac[6,56] + mem[6,56] >= 0
 accuracy_56_sk_56[7]: - 5.09765625 frac[7,56] + mem[7,56] >= 0
 accuracy_56_sk_56[8]: - 5.09765625 frac[8,56] + mem[8,56] >= 0
 accuracy_56_sk_56[9]: - 5.09765625 frac[9,56] + mem[9,56] >= 0
 accuracy_56_sk_56[10]: - 5.09765625 frac[10,56] + mem[10,56] >= 0
 accuracy_56_sk_56[11]: - 5.09765625 frac[11,56] + mem[11,56] >= 0
 accuracy_56_sk_56[12]: - 5.09765625 frac[12,56] + mem[12,56] >= 0
 accuracy_56_sk_56[13]: - 5.09765625 frac[13,56] + mem[13,56] >= 0
 accuracy_56_sk_56[14]: - 5.09765625 frac[14,56] + mem[14,56] >= 0
 accuracy_57_sk_57[0]: - 5.09765625 frac[0,57] + mem[0,57] >= 0
 accuracy_57_sk_57[1]: - 5.09765625 frac[1,57] + mem[1,57] >= 0
 accuracy_57_sk_57[2]: - 5.09765625 frac[2,57] + mem[2,57] >= 0
 accuracy_57_sk_57[3]: - 5.09765625 frac[3,57] + mem[3,57] >= 0
 accuracy_57_sk_57[4]: - 5.09765625 frac[4,57] + mem[4,57] >= 0
 accuracy_57_sk_57[5]: - 5.09765625 frac[5,57] + mem[5,57] >= 0
 accuracy_57_sk_57[6]: - 5.09765625 frac[6,57] + mem[6,57] >= 0
 accuracy_57_sk_57[7]: - 5.09765625 frac[7,57] + mem[7,57] >= 0
 accuracy_57_sk_57[8]: - 5.09765625 frac[8,57] + mem[8,57] >= 0
 accuracy_57_sk_57[9]: - 5.09765625 frac[9,57] + mem[9,57] >= 0
 accuracy_57_sk_57[10]: - 5.09765625 frac[10,57] + mem[10,57] >= 0
 accuracy_57_sk_57[11]: - 5.09765625 frac[11,57] + mem[11,57] >= 0
 accuracy_57_sk_57[12]: - 5.09765625 frac[12,57] + mem[12,57] >= 0
 accuracy_57_sk_57[13]: - 5.09765625 frac[13,57] + mem[13,57] >= 0
 accuracy_57_sk_57[14]: - 5.09765625 frac[14,57] + mem[14,57] >= 0
 accuracy_58_sk_58[0]: - 5.09765625 frac[0,58] + mem[0,58] >= 0
 accuracy_58_sk_58[1]: - 5.09765625 frac[1,58] + mem[1,58] >= 0
 accuracy_58_sk_58[2]: - 5.09765625 frac[2,58] + mem[2,58] >= 0
 accuracy_58_sk_58[3]: - 5.09765625 frac[3,58] + mem[3,58] >= 0
 accuracy_58_sk_58[4]: - 5.09765625 frac[4,58] + mem[4,58] >= 0
 accuracy_58_sk_58[5]: - 5.09765625 frac[5,58] + mem[5,58] >= 0
 accuracy_58_sk_58[6]: - 5.09765625 frac[6,58] + mem[6,58] >= 0
 accuracy_58_sk_58[7]: - 5.09765625 frac[7,58] + mem[7,58] >= 0
 accuracy_58_sk_58[8]: - 5.09765625 frac[8,58] + mem[8,58] >= 0
 accuracy_58_sk_58[9]: - 5.09765625 frac[9,58] + mem[9,58] >= 0
 accuracy_58_sk_58[10]: - 5.09765625 frac[10,58] + mem[10,58] >= 0
 accuracy_58_sk_58[11]: - 5.09765625 frac[11,58] + mem[11,58] >= 0
 accuracy_58_sk_58[12]: - 5.09765625 frac[12,58] + mem[12,58] >= 0
 accuracy_58_sk_58[13]: - 5.09765625 frac[13,58] + mem[13,58] >= 0
 accuracy_58_sk_58[14]: - 5.09765625 frac[14,58] + mem[14,58] >= 0
 accuracy_59_sk_59[0]: - 5.09765625 frac[0,59] + mem[0,59] >= 0
 accuracy_59_sk_59[1]: - 5.09765625 frac[1,59] + mem[1,59] >= 0
 accuracy_59_sk_59[2]: - 5.09765625 frac[2,59] + mem[2,59] >= 0
 accuracy_59_sk_59[3]: - 5.09765625 frac[3,59] + mem[3,59] >= 0
 accuracy_59_sk_59[4]: - 5.09765625 frac[4,59] + mem[4,59] >= 0
 accuracy_59_sk_59[5]: - 5.09765625 frac[5,59] + mem[5,59] >= 0
 accuracy_59_sk_59[6]: - 5.09765625 frac[6,59] + mem[6,59] >= 0
 accuracy_59_sk_59[7]: - 5.09765625 frac[7,59] + mem[7,59] >= 0
 accuracy_59_sk_59[8]: - 5.09765625 frac[8,59] + mem[8,59] >= 0
 accuracy_59_sk_59[9]: - 5.09765625 frac[9,59] + mem[9,59] >= 0
 accuracy_59_sk_59[10]: - 5.09765625 frac[10,59] + mem[10,59] >= 0
 accuracy_59_sk_59[11]: - 5.09765625 frac[11,59] + mem[11,59] >= 0
 accuracy_59_sk_59[12]: - 5.09765625 frac[12,59] + mem[12,59] >= 0
 accuracy_59_sk_59[13]: - 5.09765625 frac[13,59] + mem[13,59] >= 0
 accuracy_59_sk_59[14]: - 5.09765625 frac[14,59] + mem[14,59] >= 0
 accuracy_60_sk_60[0]: - 5.09765625 frac[0,60] + mem[0,60] >= 0
 accuracy_60_sk_60[1]: - 5.09765625 frac[1,60] + mem[1,60] >= 0
 accuracy_60_sk_60[2]: - 5.09765625 frac[2,60] + mem[2,60] >= 0
 accuracy_60_sk_60[3]: - 5.09765625 frac[3,60] + mem[3,60] >= 0
 accuracy_60_sk_60[4]: - 5.09765625 frac[4,60] + mem[4,60] >= 0
 accuracy_60_sk_60[5]: - 5.09765625 frac[5,60] + mem[5,60] >= 0
 accuracy_60_sk_60[6]: - 5.09765625 frac[6,60] + mem[6,60] >= 0
 accuracy_60_sk_60[7]: - 5.09765625 frac[7,60] + mem[7,60] >= 0
 accuracy_60_sk_60[8]: - 5.09765625 frac[8,60] + mem[8,60] >= 0
 accuracy_60_sk_60[9]: - 5.09765625 frac[9,60] + mem[9,60] >= 0
 accuracy_60_sk_60[10]: - 5.09765625 frac[10,60] + mem[10,60] >= 0
 accuracy_60_sk_60[11]: - 5.09765625 frac[11,60] + mem[11,60] >= 0
 accuracy_60_sk_60[12]: - 5.09765625 frac[12,60] + mem[12,60] >= 0
 accuracy_60_sk_60[13]: - 5.09765625 frac[13,60] + mem[13,60] >= 0
 accuracy_60_sk_60[14]: - 5.09765625 frac[14,60] + mem[14,60] >= 0
 accuracy_61_sk_61[0]: - 5.09765625 frac[0,61] + mem[0,61] >= 0
 accuracy_61_sk_61[1]: - 5.09765625 frac[1,61] + mem[1,61] >= 0
 accuracy_61_sk_61[2]: - 5.09765625 frac[2,61] + mem[2,61] >= 0
 accuracy_61_sk_61[3]: - 5.09765625 frac[3,61] + mem[3,61] >= 0
 accuracy_61_sk_61[4]: - 5.09765625 frac[4,61] + mem[4,61] >= 0
 accuracy_61_sk_61[5]: - 5.09765625 frac[5,61] + mem[5,61] >= 0
 accuracy_61_sk_61[6]: - 5.09765625 frac[6,61] + mem[6,61] >= 0
 accuracy_61_sk_61[7]: - 5.09765625 frac[7,61] + mem[7,61] >= 0
 accuracy_61_sk_61[8]: - 5.09765625 frac[8,61] + mem[8,61] >= 0
 accuracy_61_sk_61[9]: - 5.09765625 frac[9,61] + mem[9,61] >= 0
 accuracy_61_sk_61[10]: - 5.09765625 frac[10,61] + mem[10,61] >= 0
 accuracy_61_sk_61[11]: - 5.09765625 frac[11,61] + mem[11,61] >= 0
 accuracy_61_sk_61[12]: - 5.09765625 frac[12,61] + mem[12,61] >= 0
 accuracy_61_sk_61[13]: - 5.09765625 frac[13,61] + mem[13,61] >= 0
 accuracy_61_sk_61[14]: - 5.09765625 frac[14,61] + mem[14,61] >= 0
 accuracy_62_sk_62[0]: - 5.09765625 frac[0,62] + mem[0,62] >= 0
 accuracy_62_sk_62[1]: - 5.09765625 frac[1,62] + mem[1,62] >= 0
 accuracy_62_sk_62[2]: - 5.09765625 frac[2,62] + mem[2,62] >= 0
 accuracy_62_sk_62[3]: - 5.09765625 frac[3,62] + mem[3,62] >= 0
 accuracy_62_sk_62[4]: - 5.09765625 frac[4,62] + mem[4,62] >= 0
 accuracy_62_sk_62[5]: - 5.09765625 frac[5,62] + mem[5,62] >= 0
 accuracy_62_sk_62[6]: - 5.09765625 frac[6,62] + mem[6,62] >= 0
 accuracy_62_sk_62[7]: - 5.09765625 frac[7,62] + mem[7,62] >= 0
 accuracy_62_sk_62[8]: - 5.09765625 frac[8,62] + mem[8,62] >= 0
 accuracy_62_sk_62[9]: - 5.09765625 frac[9,62] + mem[9,62] >= 0
 accuracy_62_sk_62[10]: - 5.09765625 frac[10,62] + mem[10,62] >= 0
 accuracy_62_sk_62[11]: - 5.09765625 frac[11,62] + mem[11,62] >= 0
 accuracy_62_sk_62[12]: - 5.09765625 frac[12,62] + mem[12,62] >= 0
 accuracy_62_sk_62[13]: - 5.09765625 frac[13,62] + mem[13,62] >= 0
 accuracy_62_sk_62[14]: - 5.09765625 frac[14,62] + mem[14,62] >= 0
 accuracy_63_sk_63[0]: - 5.09765625 frac[0,63] + mem[0,63] >= 0
 accuracy_63_sk_63[1]: - 5.09765625 frac[1,63] + mem[1,63] >= 0
 accuracy_63_sk_63[2]: - 5.09765625 frac[2,63] + mem[2,63] >= 0
 accuracy_63_sk_63[3]: - 5.09765625 frac[3,63] + mem[3,63] >= 0
 accuracy_63_sk_63[4]: - 5.09765625 frac[4,63] + mem[4,63] >= 0
 accuracy_63_sk_63[5]: - 5.09765625 frac[5,63] + mem[5,63] >= 0
 accuracy_63_sk_63[6]: - 5.09765625 frac[6,63] + mem[6,63] >= 0
 accuracy_63_sk_63[7]: - 5.09765625 frac[7,63] + mem[7,63] >= 0
 accuracy_63_sk_63[8]: - 5.09765625 frac[8,63] + mem[8,63] >= 0
 accuracy_63_sk_63[9]: - 5.09765625 frac[9,63] + mem[9,63] >= 0
 accuracy_63_sk_63[10]: - 5.09765625 frac[10,63] + mem[10,63] >= 0
 accuracy_63_sk_63[11]: - 5.09765625 frac[11,63] + mem[11,63] >= 0
 accuracy_63_sk_63[12]: - 5.09765625 frac[12,63] + mem[12,63] >= 0
 accuracy_63_sk_63[13]: - 5.09765625 frac[13,63] + mem[13,63] >= 0
 accuracy_63_sk_63[14]: - 5.09765625 frac[14,63] + mem[14,63] >= 0
 accuracy_64_sk_64[0]: - 5.09765625 frac[0,64] + mem[0,64] >= 0
 accuracy_64_sk_64[1]: - 5.09765625 frac[1,64] + mem[1,64] >= 0
 accuracy_64_sk_64[2]: - 5.09765625 frac[2,64] + mem[2,64] >= 0
 accuracy_64_sk_64[3]: - 5.09765625 frac[3,64] + mem[3,64] >= 0
 accuracy_64_sk_64[4]: - 5.09765625 frac[4,64] + mem[4,64] >= 0
 accuracy_64_sk_64[5]: - 5.09765625 frac[5,64] + mem[5,64] >= 0
 accuracy_64_sk_64[6]: - 5.09765625 frac[6,64] + mem[6,64] >= 0
 accuracy_64_sk_64[7]: - 5.09765625 frac[7,64] + mem[7,64] >= 0
 accuracy_64_sk_64[8]: - 5.09765625 frac[8,64] + mem[8,64] >= 0
 accuracy_64_sk_64[9]: - 5.09765625 frac[9,64] + mem[9,64] >= 0
 accuracy_64_sk_64[10]: - 5.09765625 frac[10,64] + mem[10,64] >= 0
 accuracy_64_sk_64[11]: - 5.09765625 frac[11,64] + mem[11,64] >= 0
 accuracy_64_sk_64[12]: - 5.09765625 frac[12,64] + mem[12,64] >= 0
 accuracy_64_sk_64[13]: - 5.09765625 frac[13,64] + mem[13,64] >= 0
 accuracy_64_sk_64[14]: - 5.09765625 frac[14,64] + mem[14,64] >= 0
 accuracy_65_sk_65[0]: - 5.09765625 frac[0,65] + mem[0,65] >= 0
 accuracy_65_sk_65[1]: - 5.09765625 frac[1,65] + mem[1,65] >= 0
 accuracy_65_sk_65[2]: - 5.09765625 frac[2,65] + mem[2,65] >= 0
 accuracy_65_sk_65[3]: - 5.09765625 frac[3,65] + mem[3,65] >= 0
 accuracy_65_sk_65[4]: - 5.09765625 frac[4,65] + mem[4,65] >= 0
 accuracy_65_sk_65[5]: - 5.09765625 frac[5,65] + mem[5,65] >= 0
 accuracy_65_sk_65[6]: - 5.09765625 frac[6,65] + mem[6,65] >= 0
 accuracy_65_sk_65[7]: - 5.09765625 frac[7,65] + mem[7,65] >= 0
 accuracy_65_sk_65[8]: - 5.09765625 frac[8,65] + mem[8,65] >= 0
 accuracy_65_sk_65[9]: - 5.09765625 frac[9,65] + mem[9,65] >= 0
 accuracy_65_sk_65[10]: - 5.09765625 frac[10,65] + mem[10,65] >= 0
 accuracy_65_sk_65[11]: - 5.09765625 frac[11,65] + mem[11,65] >= 0
 accuracy_65_sk_65[12]: - 5.09765625 frac[12,65] + mem[12,65] >= 0
 accuracy_65_sk_65[13]: - 5.09765625 frac[13,65] + mem[13,65] >= 0
 accuracy_65_sk_65[14]: - 5.09765625 frac[14,65] + mem[14,65] >= 0
 accuracy_66_sk_66[0]: - 5.09765625 frac[0,66] + mem[0,66] >= 0
 accuracy_66_sk_66[1]: - 5.09765625 frac[1,66] + mem[1,66] >= 0
 accuracy_66_sk_66[2]: - 5.09765625 frac[2,66] + mem[2,66] >= 0
 accuracy_66_sk_66[3]: - 5.09765625 frac[3,66] + mem[3,66] >= 0
 accuracy_66_sk_66[4]: - 5.09765625 frac[4,66] + mem[4,66] >= 0
 accuracy_66_sk_66[5]: - 5.09765625 frac[5,66] + mem[5,66] >= 0
 accuracy_66_sk_66[6]: - 5.09765625 frac[6,66] + mem[6,66] >= 0
 accuracy_66_sk_66[7]: - 5.09765625 frac[7,66] + mem[7,66] >= 0
 accuracy_66_sk_66[8]: - 5.09765625 frac[8,66] + mem[8,66] >= 0
 accuracy_66_sk_66[9]: - 5.09765625 frac[9,66] + mem[9,66] >= 0
 accuracy_66_sk_66[10]: - 5.09765625 frac[10,66] + mem[10,66] >= 0
 accuracy_66_sk_66[11]: - 5.09765625 frac[11,66] + mem[11,66] >= 0
 accuracy_66_sk_66[12]: - 5.09765625 frac[12,66] + mem[12,66] >= 0
 accuracy_66_sk_66[13]: - 5.09765625 frac[13,66] + mem[13,66] >= 0
 accuracy_66_sk_66[14]: - 5.09765625 frac[14,66] + mem[14,66] >= 0
 accuracy_67_sk_67[0]: - 5.09765625 frac[0,67] + mem[0,67] >= 0
 accuracy_67_sk_67[1]: - 5.09765625 frac[1,67] + mem[1,67] >= 0
 accuracy_67_sk_67[2]: - 5.09765625 frac[2,67] + mem[2,67] >= 0
 accuracy_67_sk_67[3]: - 5.09765625 frac[3,67] + mem[3,67] >= 0
 accuracy_67_sk_67[4]: - 5.09765625 frac[4,67] + mem[4,67] >= 0
 accuracy_67_sk_67[5]: - 5.09765625 frac[5,67] + mem[5,67] >= 0
 accuracy_67_sk_67[6]: - 5.09765625 frac[6,67] + mem[6,67] >= 0
 accuracy_67_sk_67[7]: - 5.09765625 frac[7,67] + mem[7,67] >= 0
 accuracy_67_sk_67[8]: - 5.09765625 frac[8,67] + mem[8,67] >= 0
 accuracy_67_sk_67[9]: - 5.09765625 frac[9,67] + mem[9,67] >= 0
 accuracy_67_sk_67[10]: - 5.09765625 frac[10,67] + mem[10,67] >= 0
 accuracy_67_sk_67[11]: - 5.09765625 frac[11,67] + mem[11,67] >= 0
 accuracy_67_sk_67[12]: - 5.09765625 frac[12,67] + mem[12,67] >= 0
 accuracy_67_sk_67[13]: - 5.09765625 frac[13,67] + mem[13,67] >= 0
 accuracy_67_sk_67[14]: - 5.09765625 frac[14,67] + mem[14,67] >= 0
 accuracy_68_sk_68[0]: - 5.09765625 frac[0,68] + mem[0,68] >= 0
 accuracy_68_sk_68[1]: - 5.09765625 frac[1,68] + mem[1,68] >= 0
 accuracy_68_sk_68[2]: - 5.09765625 frac[2,68] + mem[2,68] >= 0
 accuracy_68_sk_68[3]: - 5.09765625 frac[3,68] + mem[3,68] >= 0
 accuracy_68_sk_68[4]: - 5.09765625 frac[4,68] + mem[4,68] >= 0
 accuracy_68_sk_68[5]: - 5.09765625 frac[5,68] + mem[5,68] >= 0
 accuracy_68_sk_68[6]: - 5.09765625 frac[6,68] + mem[6,68] >= 0
 accuracy_68_sk_68[7]: - 5.09765625 frac[7,68] + mem[7,68] >= 0
 accuracy_68_sk_68[8]: - 5.09765625 frac[8,68] + mem[8,68] >= 0
 accuracy_68_sk_68[9]: - 5.09765625 frac[9,68] + mem[9,68] >= 0
 accuracy_68_sk_68[10]: - 5.09765625 frac[10,68] + mem[10,68] >= 0
 accuracy_68_sk_68[11]: - 5.09765625 frac[11,68] + mem[11,68] >= 0
 accuracy_68_sk_68[12]: - 5.09765625 frac[12,68] + mem[12,68] >= 0
 accuracy_68_sk_68[13]: - 5.09765625 frac[13,68] + mem[13,68] >= 0
 accuracy_68_sk_68[14]: - 5.09765625 frac[14,68] + mem[14,68] >= 0
 accuracy_69_sk_69[0]: - 5.09765625 frac[0,69] + mem[0,69] >= 0
 accuracy_69_sk_69[1]: - 5.09765625 frac[1,69] + mem[1,69] >= 0
 accuracy_69_sk_69[2]: - 5.09765625 frac[2,69] + mem[2,69] >= 0
 accuracy_69_sk_69[3]: - 5.09765625 frac[3,69] + mem[3,69] >= 0
 accuracy_69_sk_69[4]: - 5.09765625 frac[4,69] + mem[4,69] >= 0
 accuracy_69_sk_69[5]: - 5.09765625 frac[5,69] + mem[5,69] >= 0
 accuracy_69_sk_69[6]: - 5.09765625 frac[6,69] + mem[6,69] >= 0
 accuracy_69_sk_69[7]: - 5.09765625 frac[7,69] + mem[7,69] >= 0
 accuracy_69_sk_69[8]: - 5.09765625 frac[8,69] + mem[8,69] >= 0
 accuracy_69_sk_69[9]: - 5.09765625 frac[9,69] + mem[9,69] >= 0
 accuracy_69_sk_69[10]: - 5.09765625 frac[10,69] + mem[10,69] >= 0
 accuracy_69_sk_69[11]: - 5.09765625 frac[11,69] + mem[11,69] >= 0
 accuracy_69_sk_69[12]: - 5.09765625 frac[12,69] + mem[12,69] >= 0
 accuracy_69_sk_69[13]: - 5.09765625 frac[13,69] + mem[13,69] >= 0
 accuracy_69_sk_69[14]: - 5.09765625 frac[14,69] + mem[14,69] >= 0
 accuracy_70_sk_70[0]: - 5.09765625 frac[0,70] + mem[0,70] >= 0
 accuracy_70_sk_70[1]: - 5.09765625 frac[1,70] + mem[1,70] >= 0
 accuracy_70_sk_70[2]: - 5.09765625 frac[2,70] + mem[2,70] >= 0
 accuracy_70_sk_70[3]: - 5.09765625 frac[3,70] + mem[3,70] >= 0
 accuracy_70_sk_70[4]: - 5.09765625 frac[4,70] + mem[4,70] >= 0
 accuracy_70_sk_70[5]: - 5.09765625 frac[5,70] + mem[5,70] >= 0
 accuracy_70_sk_70[6]: - 5.09765625 frac[6,70] + mem[6,70] >= 0
 accuracy_70_sk_70[7]: - 5.09765625 frac[7,70] + mem[7,70] >= 0
 accuracy_70_sk_70[8]: - 5.09765625 frac[8,70] + mem[8,70] >= 0
 accuracy_70_sk_70[9]: - 5.09765625 frac[9,70] + mem[9,70] >= 0
 accuracy_70_sk_70[10]: - 5.09765625 frac[10,70] + mem[10,70] >= 0
 accuracy_70_sk_70[11]: - 5.09765625 frac[11,70] + mem[11,70] >= 0
 accuracy_70_sk_70[12]: - 5.09765625 frac[12,70] + mem[12,70] >= 0
 accuracy_70_sk_70[13]: - 5.09765625 frac[13,70] + mem[13,70] >= 0
 accuracy_70_sk_70[14]: - 5.09765625 frac[14,70] + mem[14,70] >= 0
 accuracy_71_sk_71[0]: - 5.09765625 frac[0,71] + mem[0,71] >= 0
 accuracy_71_sk_71[1]: - 5.09765625 frac[1,71] + mem[1,71] >= 0
 accuracy_71_sk_71[2]: - 5.09765625 frac[2,71] + mem[2,71] >= 0
 accuracy_71_sk_71[3]: - 5.09765625 frac[3,71] + mem[3,71] >= 0
 accuracy_71_sk_71[4]: - 5.09765625 frac[4,71] + mem[4,71] >= 0
 accuracy_71_sk_71[5]: - 5.09765625 frac[5,71] + mem[5,71] >= 0
 accuracy_71_sk_71[6]: - 5.09765625 frac[6,71] + mem[6,71] >= 0
 accuracy_71_sk_71[7]: - 5.09765625 frac[7,71] + mem[7,71] >= 0
 accuracy_71_sk_71[8]: - 5.09765625 frac[8,71] + mem[8,71] >= 0
 accuracy_71_sk_71[9]: - 5.09765625 frac[9,71] + mem[9,71] >= 0
 accuracy_71_sk_71[10]: - 5.09765625 frac[10,71] + mem[10,71] >= 0
 accuracy_71_sk_71[11]: - 5.09765625 frac[11,71] + mem[11,71] >= 0
 accuracy_71_sk_71[12]: - 5.09765625 frac[12,71] + mem[12,71] >= 0
 accuracy_71_sk_71[13]: - 5.09765625 frac[13,71] + mem[13,71] >= 0
 accuracy_71_sk_71[14]: - 5.09765625 frac[14,71] + mem[14,71] >= 0
 accuracy_72_sk_72[0]: - 5.09765625 frac[0,72] + mem[0,72] >= 0
 accuracy_72_sk_72[1]: - 5.09765625 frac[1,72] + mem[1,72] >= 0
 accuracy_72_sk_72[2]: - 5.09765625 frac[2,72] + mem[2,72] >= 0
 accuracy_72_sk_72[3]: - 5.09765625 frac[3,72] + mem[3,72] >= 0
 accuracy_72_sk_72[4]: - 5.09765625 frac[4,72] + mem[4,72] >= 0
 accuracy_72_sk_72[5]: - 5.09765625 frac[5,72] + mem[5,72] >= 0
 accuracy_72_sk_72[6]: - 5.09765625 frac[6,72] + mem[6,72] >= 0
 accuracy_72_sk_72[7]: - 5.09765625 frac[7,72] + mem[7,72] >= 0
 accuracy_72_sk_72[8]: - 5.09765625 frac[8,72] + mem[8,72] >= 0
 accuracy_72_sk_72[9]: - 5.09765625 frac[9,72] + mem[9,72] >= 0
 accuracy_72_sk_72[10]: - 5.09765625 frac[10,72] + mem[10,72] >= 0
 accuracy_72_sk_72[11]: - 5.09765625 frac[11,72] + mem[11,72] >= 0
 accuracy_72_sk_72[12]: - 5.09765625 frac[12,72] + mem[12,72] >= 0
 accuracy_72_sk_72[13]: - 5.09765625 frac[13,72] + mem[13,72] >= 0
 accuracy_72_sk_72[14]: - 5.09765625 frac[14,72] + mem[14,72] >= 0
 accuracy_73_sk_73[0]: - 5.09765625 frac[0,73] + mem[0,73] >= 0
 accuracy_73_sk_73[1]: - 5.09765625 frac[1,73] + mem[1,73] >= 0
 accuracy_73_sk_73[2]: - 5.09765625 frac[2,73] + mem[2,73] >= 0
 accuracy_73_sk_73[3]: - 5.09765625 frac[3,73] + mem[3,73] >= 0
 accuracy_73_sk_73[4]: - 5.09765625 frac[4,73] + mem[4,73] >= 0
 accuracy_73_sk_73[5]: - 5.09765625 frac[5,73] + mem[5,73] >= 0
 accuracy_73_sk_73[6]: - 5.09765625 frac[6,73] + mem[6,73] >= 0
 accuracy_73_sk_73[7]: - 5.09765625 frac[7,73] + mem[7,73] >= 0
 accuracy_73_sk_73[8]: - 5.09765625 frac[8,73] + mem[8,73] >= 0
 accuracy_73_sk_73[9]: - 5.09765625 frac[9,73] + mem[9,73] >= 0
 accuracy_73_sk_73[10]: - 5.09765625 frac[10,73] + mem[10,73] >= 0
 accuracy_73_sk_73[11]: - 5.09765625 frac[11,73] + mem[11,73] >= 0
 accuracy_73_sk_73[12]: - 5.09765625 frac[12,73] + mem[12,73] >= 0
 accuracy_73_sk_73[13]: - 5.09765625 frac[13,73] + mem[13,73] >= 0
 accuracy_73_sk_73[14]: - 5.09765625 frac[14,73] + mem[14,73] >= 0
 accuracy_74_sk_74[0]: - 5.09765625 frac[0,74] + mem[0,74] >= 0
 accuracy_74_sk_74[1]: - 5.09765625 frac[1,74] + mem[1,74] >= 0
 accuracy_74_sk_74[2]: - 5.09765625 frac[2,74] + mem[2,74] >= 0
 accuracy_74_sk_74[3]: - 5.09765625 frac[3,74] + mem[3,74] >= 0
 accuracy_74_sk_74[4]: - 5.09765625 frac[4,74] + mem[4,74] >= 0
 accuracy_74_sk_74[5]: - 5.09765625 frac[5,74] + mem[5,74] >= 0
 accuracy_74_sk_74[6]: - 5.09765625 frac[6,74] + mem[6,74] >= 0
 accuracy_74_sk_74[7]: - 5.09765625 frac[7,74] + mem[7,74] >= 0
 accuracy_74_sk_74[8]: - 5.09765625 frac[8,74] + mem[8,74] >= 0
 accuracy_74_sk_74[9]: - 5.09765625 frac[9,74] + mem[9,74] >= 0
 accuracy_74_sk_74[10]: - 5.09765625 frac[10,74] + mem[10,74] >= 0
 accuracy_74_sk_74[11]: - 5.09765625 frac[11,74] + mem[11,74] >= 0
 accuracy_74_sk_74[12]: - 5.09765625 frac[12,74] + mem[12,74] >= 0
 accuracy_74_sk_74[13]: - 5.09765625 frac[13,74] + mem[13,74] >= 0
 accuracy_74_sk_74[14]: - 5.09765625 frac[14,74] + mem[14,74] >= 0
 accuracy_75_sk_75[0]: - 5.09765625 frac[0,75] + mem[0,75] >= 0
 accuracy_75_sk_75[1]: - 5.09765625 frac[1,75] + mem[1,75] >= 0
 accuracy_75_sk_75[2]: - 5.09765625 frac[2,75] + mem[2,75] >= 0
 accuracy_75_sk_75[3]: - 5.09765625 frac[3,75] + mem[3,75] >= 0
 accuracy_75_sk_75[4]: - 5.09765625 frac[4,75] + mem[4,75] >= 0
 accuracy_75_sk_75[5]: - 5.09765625 frac[5,75] + mem[5,75] >= 0
 accuracy_75_sk_75[6]: - 5.09765625 frac[6,75] + mem[6,75] >= 0
 accuracy_75_sk_75[7]: - 5.09765625 frac[7,75] + mem[7,75] >= 0
 accuracy_75_sk_75[8]: - 5.09765625 frac[8,75] + mem[8,75] >= 0
 accuracy_75_sk_75[9]: - 5.09765625 frac[9,75] + mem[9,75] >= 0
 accuracy_75_sk_75[10]: - 5.09765625 frac[10,75] + mem[10,75] >= 0
 accuracy_75_sk_75[11]: - 5.09765625 frac[11,75] + mem[11,75] >= 0
 accuracy_75_sk_75[12]: - 5.09765625 frac[12,75] + mem[12,75] >= 0
 accuracy_75_sk_75[13]: - 5.09765625 frac[13,75] + mem[13,75] >= 0
 accuracy_75_sk_75[14]: - 5.09765625 frac[14,75] + mem[14,75] >= 0
 accuracy_76_sk_76[0]: - 5.09765625 frac[0,76] + mem[0,76] >= 0
 accuracy_76_sk_76[1]: - 5.09765625 frac[1,76] + mem[1,76] >= 0
 accuracy_76_sk_76[2]: - 5.09765625 frac[2,76] + mem[2,76] >= 0
 accuracy_76_sk_76[3]: - 5.09765625 frac[3,76] + mem[3,76] >= 0
 accuracy_76_sk_76[4]: - 5.09765625 frac[4,76] + mem[4,76] >= 0
 accuracy_76_sk_76[5]: - 5.09765625 frac[5,76] + mem[5,76] >= 0
 accuracy_76_sk_76[6]: - 5.09765625 frac[6,76] + mem[6,76] >= 0
 accuracy_76_sk_76[7]: - 5.09765625 frac[7,76] + mem[7,76] >= 0
 accuracy_76_sk_76[8]: - 5.09765625 frac[8,76] + mem[8,76] >= 0
 accuracy_76_sk_76[9]: - 5.09765625 frac[9,76] + mem[9,76] >= 0
 accuracy_76_sk_76[10]: - 5.09765625 frac[10,76] + mem[10,76] >= 0
 accuracy_76_sk_76[11]: - 5.09765625 frac[11,76] + mem[11,76] >= 0
 accuracy_76_sk_76[12]: - 5.09765625 frac[12,76] + mem[12,76] >= 0
 accuracy_76_sk_76[13]: - 5.09765625 frac[13,76] + mem[13,76] >= 0
 accuracy_76_sk_76[14]: - 5.09765625 frac[14,76] + mem[14,76] >= 0
 accuracy_77_sk_77[0]: - 5.09765625 frac[0,77] + mem[0,77] >= 0
 accuracy_77_sk_77[1]: - 5.09765625 frac[1,77] + mem[1,77] >= 0
 accuracy_77_sk_77[2]: - 5.09765625 frac[2,77] + mem[2,77] >= 0
 accuracy_77_sk_77[3]: - 5.09765625 frac[3,77] + mem[3,77] >= 0
 accuracy_77_sk_77[4]: - 5.09765625 frac[4,77] + mem[4,77] >= 0
 accuracy_77_sk_77[5]: - 5.09765625 frac[5,77] + mem[5,77] >= 0
 accuracy_77_sk_77[6]: - 5.09765625 frac[6,77] + mem[6,77] >= 0
 accuracy_77_sk_77[7]: - 5.09765625 frac[7,77] + mem[7,77] >= 0
 accuracy_77_sk_77[8]: - 5.09765625 frac[8,77] + mem[8,77] >= 0
 accuracy_77_sk_77[9]: - 5.09765625 frac[9,77] + mem[9,77] >= 0
 accuracy_77_sk_77[10]: - 5.09765625 frac[10,77] + mem[10,77] >= 0
 accuracy_77_sk_77[11]: - 5.09765625 frac[11,77] + mem[11,77] >= 0
 accuracy_77_sk_77[12]: - 5.09765625 frac[12,77] + mem[12,77] >= 0
 accuracy_77_sk_77[13]: - 5.09765625 frac[13,77] + mem[13,77] >= 0
 accuracy_77_sk_77[14]: - 5.09765625 frac[14,77] + mem[14,77] >= 0
 accuracy_78_sk_78[0]: - 5.09765625 frac[0,78] + mem[0,78] >= 0
 accuracy_78_sk_78[1]: - 5.09765625 frac[1,78] + mem[1,78] >= 0
 accuracy_78_sk_78[2]: - 5.09765625 frac[2,78] + mem[2,78] >= 0
 accuracy_78_sk_78[3]: - 5.09765625 frac[3,78] + mem[3,78] >= 0
 accuracy_78_sk_78[4]: - 5.09765625 frac[4,78] + mem[4,78] >= 0
 accuracy_78_sk_78[5]: - 5.09765625 frac[5,78] + mem[5,78] >= 0
 accuracy_78_sk_78[6]: - 5.09765625 frac[6,78] + mem[6,78] >= 0
 accuracy_78_sk_78[7]: - 5.09765625 frac[7,78] + mem[7,78] >= 0
 accuracy_78_sk_78[8]: - 5.09765625 frac[8,78] + mem[8,78] >= 0
 accuracy_78_sk_78[9]: - 5.09765625 frac[9,78] + mem[9,78] >= 0
 accuracy_78_sk_78[10]: - 5.09765625 frac[10,78] + mem[10,78] >= 0
 accuracy_78_sk_78[11]: - 5.09765625 frac[11,78] + mem[11,78] >= 0
 accuracy_78_sk_78[12]: - 5.09765625 frac[12,78] + mem[12,78] >= 0
 accuracy_78_sk_78[13]: - 5.09765625 frac[13,78] + mem[13,78] >= 0
 accuracy_78_sk_78[14]: - 5.09765625 frac[14,78] + mem[14,78] >= 0
 accuracy_79_sk_79[0]: - 5.09765625 frac[0,79] + mem[0,79] >= 0
 accuracy_79_sk_79[1]: - 5.09765625 frac[1,79] + mem[1,79] >= 0
 accuracy_79_sk_79[2]: - 5.09765625 frac[2,79] + mem[2,79] >= 0
 accuracy_79_sk_79[3]: - 5.09765625 frac[3,79] + mem[3,79] >= 0
 accuracy_79_sk_79[4]: - 5.09765625 frac[4,79] + mem[4,79] >= 0
 accuracy_79_sk_79[5]: - 5.09765625 frac[5,79] + mem[5,79] >= 0
 accuracy_79_sk_79[6]: - 5.09765625 frac[6,79] + mem[6,79] >= 0
 accuracy_79_sk_79[7]: - 5.09765625 frac[7,79] + mem[7,79] >= 0
 accuracy_79_sk_79[8]: - 5.09765625 frac[8,79] + mem[8,79] >= 0
 accuracy_79_sk_79[9]: - 5.09765625 frac[9,79] + mem[9,79] >= 0
 accuracy_79_sk_79[10]: - 5.09765625 frac[10,79] + mem[10,79] >= 0
 accuracy_79_sk_79[11]: - 5.09765625 frac[11,79] + mem[11,79] >= 0
 accuracy_79_sk_79[12]: - 5.09765625 frac[12,79] + mem[12,79] >= 0
 accuracy_79_sk_79[13]: - 5.09765625 frac[13,79] + mem[13,79] >= 0
 accuracy_79_sk_79[14]: - 5.09765625 frac[14,79] + mem[14,79] >= 0
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
   - mem[0,77] - mem[0,78] - mem[0,79] + mem_tot_cpu1 = 0
 rows_tot_cpu1: - 3 frac[0,0] - 3 frac[0,1] - 3 frac[0,2] - 3 frac[0,3]
   - 3 frac[0,4] - 3 frac[0,5] - 3 frac[0,6] - 3 frac[0,7] - 3 frac[0,8]
   - 3 frac[0,9] - 3 frac[0,10] - 3 frac[0,11] - 3 frac[0,12]
   - 3 frac[0,13] - 3 frac[0,14] - 3 frac[0,15] - 3 frac[0,16]
   - 3 frac[0,17] - 3 frac[0,18] - 3 frac[0,19] - 3 frac[0,20]
   - 3 frac[0,21] - 3 frac[0,22] - 3 frac[0,23] - 3 frac[0,24]
   - 3 frac[0,25] - 3 frac[0,26] - 3 frac[0,27] - 3 frac[0,28]
   - 3 frac[0,29] - 3 frac[0,30] - 3 frac[0,31] - 3 frac[0,32]
   - 3 frac[0,33] - 3 frac[0,34] - 3 frac[0,35] - 3 frac[0,36]
   - 3 frac[0,37] - 3 frac[0,38] - 3 frac[0,39] - 3 frac[0,40]
   - 3 frac[0,41] - 3 frac[0,42] - 3 frac[0,43] - 3 frac[0,44]
   - 3 frac[0,45] - 3 frac[0,46] - 3 frac[0,47] - 3 frac[0,48]
   - 3 frac[0,49] - 3 frac[0,50] - 3 frac[0,51] - 3 frac[0,52]
   - 3 frac[0,53] - 3 frac[0,54] - 3 frac[0,55] - 3 frac[0,56]
   - 3 frac[0,57] - 3 frac[0,58] - 3 frac[0,59] - 3 frac[0,60]
   - 3 frac[0,61] - 3 frac[0,62] - 3 frac[0,63] - 3 frac[0,64]
   - 3 frac[0,65] - 3 frac[0,66] - 3 frac[0,67] - 3 frac[0,68]
   - 3 frac[0,69] - 3 frac[0,70] - 3 frac[0,71] - 3 frac[0,72]
   - 3 frac[0,73] - 3 frac[0,74] - 3 frac[0,75] - 3 frac[0,76]
   - 3 frac[0,77] - 3 frac[0,78] - 3 frac[0,79] + rows_tot_cpu1 = 0
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
   - mem[1,77] - mem[1,78] - mem[1,79] + mem_tot_cpu2 = 0
 rows_tot_cpu2: - 3 frac[1,0] - 3 frac[1,1] - 3 frac[1,2] - 3 frac[1,3]
   - 3 frac[1,4] - 3 frac[1,5] - 3 frac[1,6] - 3 frac[1,7] - 3 frac[1,8]
   - 3 frac[1,9] - 3 frac[1,10] - 3 frac[1,11] - 3 frac[1,12]
   - 3 frac[1,13] - 3 frac[1,14] - 3 frac[1,15] - 3 frac[1,16]
   - 3 frac[1,17] - 3 frac[1,18] - 3 frac[1,19] - 3 frac[1,20]
   - 3 frac[1,21] - 3 frac[1,22] - 3 frac[1,23] - 3 frac[1,24]
   - 3 frac[1,25] - 3 frac[1,26] - 3 frac[1,27] - 3 frac[1,28]
   - 3 frac[1,29] - 3 frac[1,30] - 3 frac[1,31] - 3 frac[1,32]
   - 3 frac[1,33] - 3 frac[1,34] - 3 frac[1,35] - 3 frac[1,36]
   - 3 frac[1,37] - 3 frac[1,38] - 3 frac[1,39] - 3 frac[1,40]
   - 3 frac[1,41] - 3 frac[1,42] - 3 frac[1,43] - 3 frac[1,44]
   - 3 frac[1,45] - 3 frac[1,46] - 3 frac[1,47] - 3 frac[1,48]
   - 3 frac[1,49] - 3 frac[1,50] - 3 frac[1,51] - 3 frac[1,52]
   - 3 frac[1,53] - 3 frac[1,54] - 3 frac[1,55] - 3 frac[1,56]
   - 3 frac[1,57] - 3 frac[1,58] - 3 frac[1,59] - 3 frac[1,60]
   - 3 frac[1,61] - 3 frac[1,62] - 3 frac[1,63] - 3 frac[1,64]
   - 3 frac[1,65] - 3 frac[1,66] - 3 frac[1,67] - 3 frac[1,68]
   - 3 frac[1,69] - 3 frac[1,70] - 3 frac[1,71] - 3 frac[1,72]
   - 3 frac[1,73] - 3 frac[1,74] - 3 frac[1,75] - 3 frac[1,76]
   - 3 frac[1,77] - 3 frac[1,78] - 3 frac[1,79] + rows_tot_cpu2 = 0
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
   - mem[2,77] - mem[2,78] - mem[2,79] + mem_tot_cpu3 = 0
 rows_tot_cpu3: - 3 frac[2,0] - 3 frac[2,1] - 3 frac[2,2] - 3 frac[2,3]
   - 3 frac[2,4] - 3 frac[2,5] - 3 frac[2,6] - 3 frac[2,7] - 3 frac[2,8]
   - 3 frac[2,9] - 3 frac[2,10] - 3 frac[2,11] - 3 frac[2,12]
   - 3 frac[2,13] - 3 frac[2,14] - 3 frac[2,15] - 3 frac[2,16]
   - 3 frac[2,17] - 3 frac[2,18] - 3 frac[2,19] - 3 frac[2,20]
   - 3 frac[2,21] - 3 frac[2,22] - 3 frac[2,23] - 3 frac[2,24]
   - 3 frac[2,25] - 3 frac[2,26] - 3 frac[2,27] - 3 frac[2,28]
   - 3 frac[2,29] - 3 frac[2,30] - 3 frac[2,31] - 3 frac[2,32]
   - 3 frac[2,33] - 3 frac[2,34] - 3 frac[2,35] - 3 frac[2,36]
   - 3 frac[2,37] - 3 frac[2,38] - 3 frac[2,39] - 3 frac[2,40]
   - 3 frac[2,41] - 3 frac[2,42] - 3 frac[2,43] - 3 frac[2,44]
   - 3 frac[2,45] - 3 frac[2,46] - 3 frac[2,47] - 3 frac[2,48]
   - 3 frac[2,49] - 3 frac[2,50] - 3 frac[2,51] - 3 frac[2,52]
   - 3 frac[2,53] - 3 frac[2,54] - 3 frac[2,55] - 3 frac[2,56]
   - 3 frac[2,57] - 3 frac[2,58] - 3 frac[2,59] - 3 frac[2,60]
   - 3 frac[2,61] - 3 frac[2,62] - 3 frac[2,63] - 3 frac[2,64]
   - 3 frac[2,65] - 3 frac[2,66] - 3 frac[2,67] - 3 frac[2,68]
   - 3 frac[2,69] - 3 frac[2,70] - 3 frac[2,71] - 3 frac[2,72]
   - 3 frac[2,73] - 3 frac[2,74] - 3 frac[2,75] - 3 frac[2,76]
   - 3 frac[2,77] - 3 frac[2,78] - 3 frac[2,79] + rows_tot_cpu3 = 0
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
   - mem[3,77] - mem[3,78] - mem[3,79] + mem_tot_cpu4 = 0
 rows_tot_cpu4: - 3 frac[3,0] - 3 frac[3,1] - 3 frac[3,2] - 3 frac[3,3]
   - 3 frac[3,4] - 3 frac[3,5] - 3 frac[3,6] - 3 frac[3,7] - 3 frac[3,8]
   - 3 frac[3,9] - 3 frac[3,10] - 3 frac[3,11] - 3 frac[3,12]
   - 3 frac[3,13] - 3 frac[3,14] - 3 frac[3,15] - 3 frac[3,16]
   - 3 frac[3,17] - 3 frac[3,18] - 3 frac[3,19] - 3 frac[3,20]
   - 3 frac[3,21] - 3 frac[3,22] - 3 frac[3,23] - 3 frac[3,24]
   - 3 frac[3,25] - 3 frac[3,26] - 3 frac[3,27] - 3 frac[3,28]
   - 3 frac[3,29] - 3 frac[3,30] - 3 frac[3,31] - 3 frac[3,32]
   - 3 frac[3,33] - 3 frac[3,34] - 3 frac[3,35] - 3 frac[3,36]
   - 3 frac[3,37] - 3 frac[3,38] - 3 frac[3,39] - 3 frac[3,40]
   - 3 frac[3,41] - 3 frac[3,42] - 3 frac[3,43] - 3 frac[3,44]
   - 3 frac[3,45] - 3 frac[3,46] - 3 frac[3,47] - 3 frac[3,48]
   - 3 frac[3,49] - 3 frac[3,50] - 3 frac[3,51] - 3 frac[3,52]
   - 3 frac[3,53] - 3 frac[3,54] - 3 frac[3,55] - 3 frac[3,56]
   - 3 frac[3,57] - 3 frac[3,58] - 3 frac[3,59] - 3 frac[3,60]
   - 3 frac[3,61] - 3 frac[3,62] - 3 frac[3,63] - 3 frac[3,64]
   - 3 frac[3,65] - 3 frac[3,66] - 3 frac[3,67] - 3 frac[3,68]
   - 3 frac[3,69] - 3 frac[3,70] - 3 frac[3,71] - 3 frac[3,72]
   - 3 frac[3,73] - 3 frac[3,74] - 3 frac[3,75] - 3 frac[3,76]
   - 3 frac[3,77] - 3 frac[3,78] - 3 frac[3,79] + rows_tot_cpu4 = 0
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
   - mem[4,77] - mem[4,78] - mem[4,79] + mem_tot_cpu5 = 0
 rows_tot_cpu5: - 3 frac[4,0] - 3 frac[4,1] - 3 frac[4,2] - 3 frac[4,3]
   - 3 frac[4,4] - 3 frac[4,5] - 3 frac[4,6] - 3 frac[4,7] - 3 frac[4,8]
   - 3 frac[4,9] - 3 frac[4,10] - 3 frac[4,11] - 3 frac[4,12]
   - 3 frac[4,13] - 3 frac[4,14] - 3 frac[4,15] - 3 frac[4,16]
   - 3 frac[4,17] - 3 frac[4,18] - 3 frac[4,19] - 3 frac[4,20]
   - 3 frac[4,21] - 3 frac[4,22] - 3 frac[4,23] - 3 frac[4,24]
   - 3 frac[4,25] - 3 frac[4,26] - 3 frac[4,27] - 3 frac[4,28]
   - 3 frac[4,29] - 3 frac[4,30] - 3 frac[4,31] - 3 frac[4,32]
   - 3 frac[4,33] - 3 frac[4,34] - 3 frac[4,35] - 3 frac[4,36]
   - 3 frac[4,37] - 3 frac[4,38] - 3 frac[4,39] - 3 frac[4,40]
   - 3 frac[4,41] - 3 frac[4,42] - 3 frac[4,43] - 3 frac[4,44]
   - 3 frac[4,45] - 3 frac[4,46] - 3 frac[4,47] - 3 frac[4,48]
   - 3 frac[4,49] - 3 frac[4,50] - 3 frac[4,51] - 3 frac[4,52]
   - 3 frac[4,53] - 3 frac[4,54] - 3 frac[4,55] - 3 frac[4,56]
   - 3 frac[4,57] - 3 frac[4,58] - 3 frac[4,59] - 3 frac[4,60]
   - 3 frac[4,61] - 3 frac[4,62] - 3 frac[4,63] - 3 frac[4,64]
   - 3 frac[4,65] - 3 frac[4,66] - 3 frac[4,67] - 3 frac[4,68]
   - 3 frac[4,69] - 3 frac[4,70] - 3 frac[4,71] - 3 frac[4,72]
   - 3 frac[4,73] - 3 frac[4,74] - 3 frac[4,75] - 3 frac[4,76]
   - 3 frac[4,77] - 3 frac[4,78] - 3 frac[4,79] + rows_tot_cpu5 = 0
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
   - mem[5,77] - mem[5,78] - mem[5,79] + mem_tot_cpu6 = 0
 rows_tot_cpu6: - 3 frac[5,0] - 3 frac[5,1] - 3 frac[5,2] - 3 frac[5,3]
   - 3 frac[5,4] - 3 frac[5,5] - 3 frac[5,6] - 3 frac[5,7] - 3 frac[5,8]
   - 3 frac[5,9] - 3 frac[5,10] - 3 frac[5,11] - 3 frac[5,12]
   - 3 frac[5,13] - 3 frac[5,14] - 3 frac[5,15] - 3 frac[5,16]
   - 3 frac[5,17] - 3 frac[5,18] - 3 frac[5,19] - 3 frac[5,20]
   - 3 frac[5,21] - 3 frac[5,22] - 3 frac[5,23] - 3 frac[5,24]
   - 3 frac[5,25] - 3 frac[5,26] - 3 frac[5,27] - 3 frac[5,28]
   - 3 frac[5,29] - 3 frac[5,30] - 3 frac[5,31] - 3 frac[5,32]
   - 3 frac[5,33] - 3 frac[5,34] - 3 frac[5,35] - 3 frac[5,36]
   - 3 frac[5,37] - 3 frac[5,38] - 3 frac[5,39] - 3 frac[5,40]
   - 3 frac[5,41] - 3 frac[5,42] - 3 frac[5,43] - 3 frac[5,44]
   - 3 frac[5,45] - 3 frac[5,46] - 3 frac[5,47] - 3 frac[5,48]
   - 3 frac[5,49] - 3 frac[5,50] - 3 frac[5,51] - 3 frac[5,52]
   - 3 frac[5,53] - 3 frac[5,54] - 3 frac[5,55] - 3 frac[5,56]
   - 3 frac[5,57] - 3 frac[5,58] - 3 frac[5,59] - 3 frac[5,60]
   - 3 frac[5,61] - 3 frac[5,62] - 3 frac[5,63] - 3 frac[5,64]
   - 3 frac[5,65] - 3 frac[5,66] - 3 frac[5,67] - 3 frac[5,68]
   - 3 frac[5,69] - 3 frac[5,70] - 3 frac[5,71] - 3 frac[5,72]
   - 3 frac[5,73] - 3 frac[5,74] - 3 frac[5,75] - 3 frac[5,76]
   - 3 frac[5,77] - 3 frac[5,78] - 3 frac[5,79] + rows_tot_cpu6 = 0
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
   - mem[6,77] - mem[6,78] - mem[6,79] + mem_tot_cpu7 = 0
 rows_tot_cpu7: - 3 frac[6,0] - 3 frac[6,1] - 3 frac[6,2] - 3 frac[6,3]
   - 3 frac[6,4] - 3 frac[6,5] - 3 frac[6,6] - 3 frac[6,7] - 3 frac[6,8]
   - 3 frac[6,9] - 3 frac[6,10] - 3 frac[6,11] - 3 frac[6,12]
   - 3 frac[6,13] - 3 frac[6,14] - 3 frac[6,15] - 3 frac[6,16]
   - 3 frac[6,17] - 3 frac[6,18] - 3 frac[6,19] - 3 frac[6,20]
   - 3 frac[6,21] - 3 frac[6,22] - 3 frac[6,23] - 3 frac[6,24]
   - 3 frac[6,25] - 3 frac[6,26] - 3 frac[6,27] - 3 frac[6,28]
   - 3 frac[6,29] - 3 frac[6,30] - 3 frac[6,31] - 3 frac[6,32]
   - 3 frac[6,33] - 3 frac[6,34] - 3 frac[6,35] - 3 frac[6,36]
   - 3 frac[6,37] - 3 frac[6,38] - 3 frac[6,39] - 3 frac[6,40]
   - 3 frac[6,41] - 3 frac[6,42] - 3 frac[6,43] - 3 frac[6,44]
   - 3 frac[6,45] - 3 frac[6,46] - 3 frac[6,47] - 3 frac[6,48]
   - 3 frac[6,49] - 3 frac[6,50] - 3 frac[6,51] - 3 frac[6,52]
   - 3 frac[6,53] - 3 frac[6,54] - 3 frac[6,55] - 3 frac[6,56]
   - 3 frac[6,57] - 3 frac[6,58] - 3 frac[6,59] - 3 frac[6,60]
   - 3 frac[6,61] - 3 frac[6,62] - 3 frac[6,63] - 3 frac[6,64]
   - 3 frac[6,65] - 3 frac[6,66] - 3 frac[6,67] - 3 frac[6,68]
   - 3 frac[6,69] - 3 frac[6,70] - 3 frac[6,71] - 3 frac[6,72]
   - 3 frac[6,73] - 3 frac[6,74] - 3 frac[6,75] - 3 frac[6,76]
   - 3 frac[6,77] - 3 frac[6,78] - 3 frac[6,79] + rows_tot_cpu7 = 0
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
   - mem[7,77] - mem[7,78] - mem[7,79] + mem_tot_cpu8 = 0
 rows_tot_cpu8: - 3 frac[7,0] - 3 frac[7,1] - 3 frac[7,2] - 3 frac[7,3]
   - 3 frac[7,4] - 3 frac[7,5] - 3 frac[7,6] - 3 frac[7,7] - 3 frac[7,8]
   - 3 frac[7,9] - 3 frac[7,10] - 3 frac[7,11] - 3 frac[7,12]
   - 3 frac[7,13] - 3 frac[7,14] - 3 frac[7,15] - 3 frac[7,16]
   - 3 frac[7,17] - 3 frac[7,18] - 3 frac[7,19] - 3 frac[7,20]
   - 3 frac[7,21] - 3 frac[7,22] - 3 frac[7,23] - 3 frac[7,24]
   - 3 frac[7,25] - 3 frac[7,26] - 3 frac[7,27] - 3 frac[7,28]
   - 3 frac[7,29] - 3 frac[7,30] - 3 frac[7,31] - 3 frac[7,32]
   - 3 frac[7,33] - 3 frac[7,34] - 3 frac[7,35] - 3 frac[7,36]
   - 3 frac[7,37] - 3 frac[7,38] - 3 frac[7,39] - 3 frac[7,40]
   - 3 frac[7,41] - 3 frac[7,42] - 3 frac[7,43] - 3 frac[7,44]
   - 3 frac[7,45] - 3 frac[7,46] - 3 frac[7,47] - 3 frac[7,48]
   - 3 frac[7,49] - 3 frac[7,50] - 3 frac[7,51] - 3 frac[7,52]
   - 3 frac[7,53] - 3 frac[7,54] - 3 frac[7,55] - 3 frac[7,56]
   - 3 frac[7,57] - 3 frac[7,58] - 3 frac[7,59] - 3 frac[7,60]
   - 3 frac[7,61] - 3 frac[7,62] - 3 frac[7,63] - 3 frac[7,64]
   - 3 frac[7,65] - 3 frac[7,66] - 3 frac[7,67] - 3 frac[7,68]
   - 3 frac[7,69] - 3 frac[7,70] - 3 frac[7,71] - 3 frac[7,72]
   - 3 frac[7,73] - 3 frac[7,74] - 3 frac[7,75] - 3 frac[7,76]
   - 3 frac[7,77] - 3 frac[7,78] - 3 frac[7,79] + rows_tot_cpu8 = 0
 mem_tot_tor_p41: - mem[8,0] - mem[8,1] - mem[8,2] - mem[8,3] - mem[8,4]
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
   - mem[8,77] - mem[8,78] - mem[8,79] + mem_tot_tor_p41 = 0
 rows_tot_tor_p41: - 3 frac[8,0] - 3 frac[8,1] - 3 frac[8,2] - 3 frac[8,3]
   - 3 frac[8,4] - 3 frac[8,5] - 3 frac[8,6] - 3 frac[8,7] - 3 frac[8,8]
   - 3 frac[8,9] - 3 frac[8,10] - 3 frac[8,11] - 3 frac[8,12]
   - 3 frac[8,13] - 3 frac[8,14] - 3 frac[8,15] - 3 frac[8,16]
   - 3 frac[8,17] - 3 frac[8,18] - 3 frac[8,19] - 3 frac[8,20]
   - 3 frac[8,21] - 3 frac[8,22] - 3 frac[8,23] - 3 frac[8,24]
   - 3 frac[8,25] - 3 frac[8,26] - 3 frac[8,27] - 3 frac[8,28]
   - 3 frac[8,29] - 3 frac[8,30] - 3 frac[8,31] - 3 frac[8,32]
   - 3 frac[8,33] - 3 frac[8,34] - 3 frac[8,35] - 3 frac[8,36]
   - 3 frac[8,37] - 3 frac[8,38] - 3 frac[8,39] - 3 frac[8,40]
   - 3 frac[8,41] - 3 frac[8,42] - 3 frac[8,43] - 3 frac[8,44]
   - 3 frac[8,45] - 3 frac[8,46] - 3 frac[8,47] - 3 frac[8,48]
   - 3 frac[8,49] - 3 frac[8,50] - 3 frac[8,51] - 3 frac[8,52]
   - 3 frac[8,53] - 3 frac[8,54] - 3 frac[8,55] - 3 frac[8,56]
   - 3 frac[8,57] - 3 frac[8,58] - 3 frac[8,59] - 3 frac[8,60]
   - 3 frac[8,61] - 3 frac[8,62] - 3 frac[8,63] - 3 frac[8,64]
   - 3 frac[8,65] - 3 frac[8,66] - 3 frac[8,67] - 3 frac[8,68]
   - 3 frac[8,69] - 3 frac[8,70] - 3 frac[8,71] - 3 frac[8,72]
   - 3 frac[8,73] - 3 frac[8,74] - 3 frac[8,75] - 3 frac[8,76]
   - 3 frac[8,77] - 3 frac[8,78] - 3 frac[8,79] + rows_tot_tor_p41 = 0
 mem_tot_tor_p42: - mem[9,0] - mem[9,1] - mem[9,2] - mem[9,3] - mem[9,4]
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
   - mem[9,77] - mem[9,78] - mem[9,79] + mem_tot_tor_p42 = 0
 rows_tot_tor_p42: - 3 frac[9,0] - 3 frac[9,1] - 3 frac[9,2] - 3 frac[9,3]
   - 3 frac[9,4] - 3 frac[9,5] - 3 frac[9,6] - 3 frac[9,7] - 3 frac[9,8]
   - 3 frac[9,9] - 3 frac[9,10] - 3 frac[9,11] - 3 frac[9,12]
   - 3 frac[9,13] - 3 frac[9,14] - 3 frac[9,15] - 3 frac[9,16]
   - 3 frac[9,17] - 3 frac[9,18] - 3 frac[9,19] - 3 frac[9,20]
   - 3 frac[9,21] - 3 frac[9,22] - 3 frac[9,23] - 3 frac[9,24]
   - 3 frac[9,25] - 3 frac[9,26] - 3 frac[9,27] - 3 frac[9,28]
   - 3 frac[9,29] - 3 frac[9,30] - 3 frac[9,31] - 3 frac[9,32]
   - 3 frac[9,33] - 3 frac[9,34] - 3 frac[9,35] - 3 frac[9,36]
   - 3 frac[9,37] - 3 frac[9,38] - 3 frac[9,39] - 3 frac[9,40]
   - 3 frac[9,41] - 3 frac[9,42] - 3 frac[9,43] - 3 frac[9,44]
   - 3 frac[9,45] - 3 frac[9,46] - 3 frac[9,47] - 3 frac[9,48]
   - 3 frac[9,49] - 3 frac[9,50] - 3 frac[9,51] - 3 frac[9,52]
   - 3 frac[9,53] - 3 frac[9,54] - 3 frac[9,55] - 3 frac[9,56]
   - 3 frac[9,57] - 3 frac[9,58] - 3 frac[9,59] - 3 frac[9,60]
   - 3 frac[9,61] - 3 frac[9,62] - 3 frac[9,63] - 3 frac[9,64]
   - 3 frac[9,65] - 3 frac[9,66] - 3 frac[9,67] - 3 frac[9,68]
   - 3 frac[9,69] - 3 frac[9,70] - 3 frac[9,71] - 3 frac[9,72]
   - 3 frac[9,73] - 3 frac[9,74] - 3 frac[9,75] - 3 frac[9,76]
   - 3 frac[9,77] - 3 frac[9,78] - 3 frac[9,79] + rows_tot_tor_p42 = 0
 mem_tot_tor_p43: - mem[10,0] - mem[10,1] - mem[10,2] - mem[10,3]
   - mem[10,4] - mem[10,5] - mem[10,6] - mem[10,7] - mem[10,8] - mem[10,9]
   - mem[10,10] - mem[10,11] - mem[10,12] - mem[10,13] - mem[10,14]
   - mem[10,15] - mem[10,16] - mem[10,17] - mem[10,18] - mem[10,19]
   - mem[10,20] - mem[10,21] - mem[10,22] - mem[10,23] - mem[10,24]
   - mem[10,25] - mem[10,26] - mem[10,27] - mem[10,28] - mem[10,29]
   - mem[10,30] - mem[10,31] - mem[10,32] - mem[10,33] - mem[10,34]
   - mem[10,35] - mem[10,36] - mem[10,37] - mem[10,38] - mem[10,39]
   - mem[10,40] - mem[10,41] - mem[10,42] - mem[10,43] - mem[10,44]
   - mem[10,45] - mem[10,46] - mem[10,47] - mem[10,48] - mem[10,49]
   - mem[10,50] - mem[10,51] - mem[10,52] - mem[10,53] - mem[10,54]
   - mem[10,55] - mem[10,56] - mem[10,57] - mem[10,58] - mem[10,59]
   - mem[10,60] - mem[10,61] - mem[10,62] - mem[10,63] - mem[10,64]
   - mem[10,65] - mem[10,66] - mem[10,67] - mem[10,68] - mem[10,69]
   - mem[10,70] - mem[10,71] - mem[10,72] - mem[10,73] - mem[10,74]
   - mem[10,75] - mem[10,76] - mem[10,77] - mem[10,78] - mem[10,79]
   + mem_tot_tor_p43 = 0
 rows_tot_tor_p43: - 3 frac[10,0] - 3 frac[10,1] - 3 frac[10,2]
   - 3 frac[10,3] - 3 frac[10,4] - 3 frac[10,5] - 3 frac[10,6]
   - 3 frac[10,7] - 3 frac[10,8] - 3 frac[10,9] - 3 frac[10,10]
   - 3 frac[10,11] - 3 frac[10,12] - 3 frac[10,13] - 3 frac[10,14]
   - 3 frac[10,15] - 3 frac[10,16] - 3 frac[10,17] - 3 frac[10,18]
   - 3 frac[10,19] - 3 frac[10,20] - 3 frac[10,21] - 3 frac[10,22]
   - 3 frac[10,23] - 3 frac[10,24] - 3 frac[10,25] - 3 frac[10,26]
   - 3 frac[10,27] - 3 frac[10,28] - 3 frac[10,29] - 3 frac[10,30]
   - 3 frac[10,31] - 3 frac[10,32] - 3 frac[10,33] - 3 frac[10,34]
   - 3 frac[10,35] - 3 frac[10,36] - 3 frac[10,37] - 3 frac[10,38]
   - 3 frac[10,39] - 3 frac[10,40] - 3 frac[10,41] - 3 frac[10,42]
   - 3 frac[10,43] - 3 frac[10,44] - 3 frac[10,45] - 3 frac[10,46]
   - 3 frac[10,47] - 3 frac[10,48] - 3 frac[10,49] - 3 frac[10,50]
   - 3 frac[10,51] - 3 frac[10,52] - 3 frac[10,53] - 3 frac[10,54]
   - 3 frac[10,55] - 3 frac[10,56] - 3 frac[10,57] - 3 frac[10,58]
   - 3 frac[10,59] - 3 frac[10,60] - 3 frac[10,61] - 3 frac[10,62]
   - 3 frac[10,63] - 3 frac[10,64] - 3 frac[10,65] - 3 frac[10,66]
   - 3 frac[10,67] - 3 frac[10,68] - 3 frac[10,69] - 3 frac[10,70]
   - 3 frac[10,71] - 3 frac[10,72] - 3 frac[10,73] - 3 frac[10,74]
   - 3 frac[10,75] - 3 frac[10,76] - 3 frac[10,77] - 3 frac[10,78]
   - 3 frac[10,79] + rows_tot_tor_p43 = 0
 mem_tot_tor_p44: - mem[11,0] - mem[11,1] - mem[11,2] - mem[11,3]
   - mem[11,4] - mem[11,5] - mem[11,6] - mem[11,7] - mem[11,8] - mem[11,9]
   - mem[11,10] - mem[11,11] - mem[11,12] - mem[11,13] - mem[11,14]
   - mem[11,15] - mem[11,16] - mem[11,17] - mem[11,18] - mem[11,19]
   - mem[11,20] - mem[11,21] - mem[11,22] - mem[11,23] - mem[11,24]
   - mem[11,25] - mem[11,26] - mem[11,27] - mem[11,28] - mem[11,29]
   - mem[11,30] - mem[11,31] - mem[11,32] - mem[11,33] - mem[11,34]
   - mem[11,35] - mem[11,36] - mem[11,37] - mem[11,38] - mem[11,39]
   - mem[11,40] - mem[11,41] - mem[11,42] - mem[11,43] - mem[11,44]
   - mem[11,45] - mem[11,46] - mem[11,47] - mem[11,48] - mem[11,49]
   - mem[11,50] - mem[11,51] - mem[11,52] - mem[11,53] - mem[11,54]
   - mem[11,55] - mem[11,56] - mem[11,57] - mem[11,58] - mem[11,59]
   - mem[11,60] - mem[11,61] - mem[11,62] - mem[11,63] - mem[11,64]
   - mem[11,65] - mem[11,66] - mem[11,67] - mem[11,68] - mem[11,69]
   - mem[11,70] - mem[11,71] - mem[11,72] - mem[11,73] - mem[11,74]
   - mem[11,75] - mem[11,76] - mem[11,77] - mem[11,78] - mem[11,79]
   + mem_tot_tor_p44 = 0
 rows_tot_tor_p44: - 3 frac[11,0] - 3 frac[11,1] - 3 frac[11,2]
   - 3 frac[11,3] - 3 frac[11,4] - 3 frac[11,5] - 3 frac[11,6]
   - 3 frac[11,7] - 3 frac[11,8] - 3 frac[11,9] - 3 frac[11,10]
   - 3 frac[11,11] - 3 frac[11,12] - 3 frac[11,13] - 3 frac[11,14]
   - 3 frac[11,15] - 3 frac[11,16] - 3 frac[11,17] - 3 frac[11,18]
   - 3 frac[11,19] - 3 frac[11,20] - 3 frac[11,21] - 3 frac[11,22]
   - 3 frac[11,23] - 3 frac[11,24] - 3 frac[11,25] - 3 frac[11,26]
   - 3 frac[11,27] - 3 frac[11,28] - 3 frac[11,29] - 3 frac[11,30]
   - 3 frac[11,31] - 3 frac[11,32] - 3 frac[11,33] - 3 frac[11,34]
   - 3 frac[11,35] - 3 frac[11,36] - 3 frac[11,37] - 3 frac[11,38]
   - 3 frac[11,39] - 3 frac[11,40] - 3 frac[11,41] - 3 frac[11,42]
   - 3 frac[11,43] - 3 frac[11,44] - 3 frac[11,45] - 3 frac[11,46]
   - 3 frac[11,47] - 3 frac[11,48] - 3 frac[11,49] - 3 frac[11,50]
   - 3 frac[11,51] - 3 frac[11,52] - 3 frac[11,53] - 3 frac[11,54]
   - 3 frac[11,55] - 3 frac[11,56] - 3 frac[11,57] - 3 frac[11,58]
   - 3 frac[11,59] - 3 frac[11,60] - 3 frac[11,61] - 3 frac[11,62]
   - 3 frac[11,63] - 3 frac[11,64] - 3 frac[11,65] - 3 frac[11,66]
   - 3 frac[11,67] - 3 frac[11,68] - 3 frac[11,69] - 3 frac[11,70]
   - 3 frac[11,71] - 3 frac[11,72] - 3 frac[11,73] - 3 frac[11,74]
   - 3 frac[11,75] - 3 frac[11,76] - 3 frac[11,77] - 3 frac[11,78]
   - 3 frac[11,79] + rows_tot_tor_p44 = 0
 mem_tot_l1_p41: - mem[12,0] - mem[12,1] - mem[12,2] - mem[12,3]
   - mem[12,4] - mem[12,5] - mem[12,6] - mem[12,7] - mem[12,8] - mem[12,9]
   - mem[12,10] - mem[12,11] - mem[12,12] - mem[12,13] - mem[12,14]
   - mem[12,15] - mem[12,16] - mem[12,17] - mem[12,18] - mem[12,19]
   - mem[12,20] - mem[12,21] - mem[12,22] - mem[12,23] - mem[12,24]
   - mem[12,25] - mem[12,26] - mem[12,27] - mem[12,28] - mem[12,29]
   - mem[12,30] - mem[12,31] - mem[12,32] - mem[12,33] - mem[12,34]
   - mem[12,35] - mem[12,36] - mem[12,37] - mem[12,38] - mem[12,39]
   - mem[12,40] - mem[12,41] - mem[12,42] - mem[12,43] - mem[12,44]
   - mem[12,45] - mem[12,46] - mem[12,47] - mem[12,48] - mem[12,49]
   - mem[12,50] - mem[12,51] - mem[12,52] - mem[12,53] - mem[12,54]
   - mem[12,55] - mem[12,56] - mem[12,57] - mem[12,58] - mem[12,59]
   - mem[12,60] - mem[12,61] - mem[12,62] - mem[12,63] - mem[12,64]
   - mem[12,65] - mem[12,66] - mem[12,67] - mem[12,68] - mem[12,69]
   - mem[12,70] - mem[12,71] - mem[12,72] - mem[12,73] - mem[12,74]
   - mem[12,75] - mem[12,76] - mem[12,77] - mem[12,78] - mem[12,79]
   + mem_tot_l1_p41 = 0
 rows_tot_l1_p41: - 3 frac[12,0] - 3 frac[12,1] - 3 frac[12,2]
   - 3 frac[12,3] - 3 frac[12,4] - 3 frac[12,5] - 3 frac[12,6]
   - 3 frac[12,7] - 3 frac[12,8] - 3 frac[12,9] - 3 frac[12,10]
   - 3 frac[12,11] - 3 frac[12,12] - 3 frac[12,13] - 3 frac[12,14]
   - 3 frac[12,15] - 3 frac[12,16] - 3 frac[12,17] - 3 frac[12,18]
   - 3 frac[12,19] - 3 frac[12,20] - 3 frac[12,21] - 3 frac[12,22]
   - 3 frac[12,23] - 3 frac[12,24] - 3 frac[12,25] - 3 frac[12,26]
   - 3 frac[12,27] - 3 frac[12,28] - 3 frac[12,29] - 3 frac[12,30]
   - 3 frac[12,31] - 3 frac[12,32] - 3 frac[12,33] - 3 frac[12,34]
   - 3 frac[12,35] - 3 frac[12,36] - 3 frac[12,37] - 3 frac[12,38]
   - 3 frac[12,39] - 3 frac[12,40] - 3 frac[12,41] - 3 frac[12,42]
   - 3 frac[12,43] - 3 frac[12,44] - 3 frac[12,45] - 3 frac[12,46]
   - 3 frac[12,47] - 3 frac[12,48] - 3 frac[12,49] - 3 frac[12,50]
   - 3 frac[12,51] - 3 frac[12,52] - 3 frac[12,53] - 3 frac[12,54]
   - 3 frac[12,55] - 3 frac[12,56] - 3 frac[12,57] - 3 frac[12,58]
   - 3 frac[12,59] - 3 frac[12,60] - 3 frac[12,61] - 3 frac[12,62]
   - 3 frac[12,63] - 3 frac[12,64] - 3 frac[12,65] - 3 frac[12,66]
   - 3 frac[12,67] - 3 frac[12,68] - 3 frac[12,69] - 3 frac[12,70]
   - 3 frac[12,71] - 3 frac[12,72] - 3 frac[12,73] - 3 frac[12,74]
   - 3 frac[12,75] - 3 frac[12,76] - 3 frac[12,77] - 3 frac[12,78]
   - 3 frac[12,79] + rows_tot_l1_p41 = 0
 mem_tot_l1_p42: - mem[13,0] - mem[13,1] - mem[13,2] - mem[13,3]
   - mem[13,4] - mem[13,5] - mem[13,6] - mem[13,7] - mem[13,8] - mem[13,9]
   - mem[13,10] - mem[13,11] - mem[13,12] - mem[13,13] - mem[13,14]
   - mem[13,15] - mem[13,16] - mem[13,17] - mem[13,18] - mem[13,19]
   - mem[13,20] - mem[13,21] - mem[13,22] - mem[13,23] - mem[13,24]
   - mem[13,25] - mem[13,26] - mem[13,27] - mem[13,28] - mem[13,29]
   - mem[13,30] - mem[13,31] - mem[13,32] - mem[13,33] - mem[13,34]
   - mem[13,35] - mem[13,36] - mem[13,37] - mem[13,38] - mem[13,39]
   - mem[13,40] - mem[13,41] - mem[13,42] - mem[13,43] - mem[13,44]
   - mem[13,45] - mem[13,46] - mem[13,47] - mem[13,48] - mem[13,49]
   - mem[13,50] - mem[13,51] - mem[13,52] - mem[13,53] - mem[13,54]
   - mem[13,55] - mem[13,56] - mem[13,57] - mem[13,58] - mem[13,59]
   - mem[13,60] - mem[13,61] - mem[13,62] - mem[13,63] - mem[13,64]
   - mem[13,65] - mem[13,66] - mem[13,67] - mem[13,68] - mem[13,69]
   - mem[13,70] - mem[13,71] - mem[13,72] - mem[13,73] - mem[13,74]
   - mem[13,75] - mem[13,76] - mem[13,77] - mem[13,78] - mem[13,79]
   + mem_tot_l1_p42 = 0
 rows_tot_l1_p42: - 3 frac[13,0] - 3 frac[13,1] - 3 frac[13,2]
   - 3 frac[13,3] - 3 frac[13,4] - 3 frac[13,5] - 3 frac[13,6]
   - 3 frac[13,7] - 3 frac[13,8] - 3 frac[13,9] - 3 frac[13,10]
   - 3 frac[13,11] - 3 frac[13,12] - 3 frac[13,13] - 3 frac[13,14]
   - 3 frac[13,15] - 3 frac[13,16] - 3 frac[13,17] - 3 frac[13,18]
   - 3 frac[13,19] - 3 frac[13,20] - 3 frac[13,21] - 3 frac[13,22]
   - 3 frac[13,23] - 3 frac[13,24] - 3 frac[13,25] - 3 frac[13,26]
   - 3 frac[13,27] - 3 frac[13,28] - 3 frac[13,29] - 3 frac[13,30]
   - 3 frac[13,31] - 3 frac[13,32] - 3 frac[13,33] - 3 frac[13,34]
   - 3 frac[13,35] - 3 frac[13,36] - 3 frac[13,37] - 3 frac[13,38]
   - 3 frac[13,39] - 3 frac[13,40] - 3 frac[13,41] - 3 frac[13,42]
   - 3 frac[13,43] - 3 frac[13,44] - 3 frac[13,45] - 3 frac[13,46]
   - 3 frac[13,47] - 3 frac[13,48] - 3 frac[13,49] - 3 frac[13,50]
   - 3 frac[13,51] - 3 frac[13,52] - 3 frac[13,53] - 3 frac[13,54]
   - 3 frac[13,55] - 3 frac[13,56] - 3 frac[13,57] - 3 frac[13,58]
   - 3 frac[13,59] - 3 frac[13,60] - 3 frac[13,61] - 3 frac[13,62]
   - 3 frac[13,63] - 3 frac[13,64] - 3 frac[13,65] - 3 frac[13,66]
   - 3 frac[13,67] - 3 frac[13,68] - 3 frac[13,69] - 3 frac[13,70]
   - 3 frac[13,71] - 3 frac[13,72] - 3 frac[13,73] - 3 frac[13,74]
   - 3 frac[13,75] - 3 frac[13,76] - 3 frac[13,77] - 3 frac[13,78]
   - 3 frac[13,79] + rows_tot_l1_p42 = 0
 mem_tot_l2_p4: - mem[14,0] - mem[14,1] - mem[14,2] - mem[14,3] - mem[14,4]
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
   - mem[14,76] - mem[14,77] - mem[14,78] - mem[14,79] + mem_tot_l2_p4 = 0
 rows_tot_l2_p4: - 3 frac[14,0] - 3 frac[14,1] - 3 frac[14,2]
   - 3 frac[14,3] - 3 frac[14,4] - 3 frac[14,5] - 3 frac[14,6]
   - 3 frac[14,7] - 3 frac[14,8] - 3 frac[14,9] - 3 frac[14,10]
   - 3 frac[14,11] - 3 frac[14,12] - 3 frac[14,13] - 3 frac[14,14]
   - 3 frac[14,15] - 3 frac[14,16] - 3 frac[14,17] - 3 frac[14,18]
   - 3 frac[14,19] - 3 frac[14,20] - 3 frac[14,21] - 3 frac[14,22]
   - 3 frac[14,23] - 3 frac[14,24] - 3 frac[14,25] - 3 frac[14,26]
   - 3 frac[14,27] - 3 frac[14,28] - 3 frac[14,29] - 3 frac[14,30]
   - 3 frac[14,31] - 3 frac[14,32] - 3 frac[14,33] - 3 frac[14,34]
   - 3 frac[14,35] - 3 frac[14,36] - 3 frac[14,37] - 3 frac[14,38]
   - 3 frac[14,39] - 3 frac[14,40] - 3 frac[14,41] - 3 frac[14,42]
   - 3 frac[14,43] - 3 frac[14,44] - 3 frac[14,45] - 3 frac[14,46]
   - 3 frac[14,47] - 3 frac[14,48] - 3 frac[14,49] - 3 frac[14,50]
   - 3 frac[14,51] - 3 frac[14,52] - 3 frac[14,53] - 3 frac[14,54]
   - 3 frac[14,55] - 3 frac[14,56] - 3 frac[14,57] - 3 frac[14,58]
   - 3 frac[14,59] - 3 frac[14,60] - 3 frac[14,61] - 3 frac[14,62]
   - 3 frac[14,63] - 3 frac[14,64] - 3 frac[14,65] - 3 frac[14,66]
   - 3 frac[14,67] - 3 frac[14,68] - 3 frac[14,69] - 3 frac[14,70]
   - 3 frac[14,71] - 3 frac[14,72] - 3 frac[14,73] - 3 frac[14,74]
   - 3 frac[14,75] - 3 frac[14,76] - 3 frac[14,77] - 3 frac[14,78]
   - 3 frac[14,79] + rows_tot_l2_p4 = 0
 row_capacity_cpu1: rows_tot_cpu1 <= 12
 row_capacity_cpu2: rows_tot_cpu2 <= 12
 row_capacity_cpu3: rows_tot_cpu3 <= 12
 row_capacity_cpu4: rows_tot_cpu4 <= 12
 row_capacity_cpu5: rows_tot_cpu5 <= 12
 row_capacity_cpu6: rows_tot_cpu6 <= 12
 row_capacity_cpu7: rows_tot_cpu7 <= 12
 row_capacity_cpu8: rows_tot_cpu8 <= 12
 row_capacity_tor_p41: rows_tot_tor_p41 <= 48
 row_capacity_tor_p42: rows_tot_tor_p42 <= 48
 row_capacity_tor_p43: rows_tot_tor_p43 <= 48
 row_capacity_tor_p44: rows_tot_tor_p44 <= 48
 row_capacity_l1_p41: rows_tot_l1_p41 <= 48
 row_capacity_l1_p42: rows_tot_l1_p42 <= 48
 row_capacity_l2_p4: rows_tot_l2_p4 <= 48
 ns_single_cpu1: 3.5 rows_tot_cpu1 - ns_single_cpu1 + pdt_m_rows_cpu1 = 0
 capacity_cores_cpu1: cores_sketch_cpu1 + cores_dpdk_cpu1 <= 7
 ns_sketch_cpu1: - ns_single_cpu1 + pdt_nsk_c_cpu1 = 0
 ns_dpdk_cpu1: 7.142857142857143 cores_dpdk_cpu1 - pdt_nsc_dpdk_cpu1
   = -21.42857142857143
 ns_single_cpu2: 3.5 rows_tot_cpu2 - ns_single_cpu2 + pdt_m_rows_cpu2 = 0
 capacity_cores_cpu2: cores_sketch_cpu2 + cores_dpdk_cpu2 <= 7
 ns_sketch_cpu2: - ns_single_cpu2 + pdt_nsk_c_cpu2 = 0
 ns_dpdk_cpu2: 7.142857142857143 cores_dpdk_cpu2 - pdt_nsc_dpdk_cpu2
   = -21.42857142857143
 ns_single_cpu3: 3.5 rows_tot_cpu3 - ns_single_cpu3 + pdt_m_rows_cpu3 = 0
 capacity_cores_cpu3: cores_sketch_cpu3 + cores_dpdk_cpu3 <= 7
 ns_sketch_cpu3: - ns_single_cpu3 + pdt_nsk_c_cpu3 = 0
 ns_dpdk_cpu3: 7.142857142857143 cores_dpdk_cpu3 - pdt_nsc_dpdk_cpu3
   = -21.42857142857143
 ns_single_cpu4: 3.5 rows_tot_cpu4 - ns_single_cpu4 + pdt_m_rows_cpu4 = 0
 capacity_cores_cpu4: cores_sketch_cpu4 + cores_dpdk_cpu4 <= 7
 ns_sketch_cpu4: - ns_single_cpu4 + pdt_nsk_c_cpu4 = 0
 ns_dpdk_cpu4: 7.142857142857143 cores_dpdk_cpu4 - pdt_nsc_dpdk_cpu4
   = -21.42857142857143
 ns_single_cpu5: 3.5 rows_tot_cpu5 - ns_single_cpu5 + pdt_m_rows_cpu5 = 0
 capacity_cores_cpu5: cores_sketch_cpu5 + cores_dpdk_cpu5 <= 7
 ns_sketch_cpu5: - ns_single_cpu5 + pdt_nsk_c_cpu5 = 0
 ns_dpdk_cpu5: 7.142857142857143 cores_dpdk_cpu5 - pdt_nsc_dpdk_cpu5
   = -21.42857142857143
 ns_single_cpu6: 3.5 rows_tot_cpu6 - ns_single_cpu6 + pdt_m_rows_cpu6 = 0
 capacity_cores_cpu6: cores_sketch_cpu6 + cores_dpdk_cpu6 <= 7
 ns_sketch_cpu6: - ns_single_cpu6 + pdt_nsk_c_cpu6 = 0
 ns_dpdk_cpu6: 7.142857142857143 cores_dpdk_cpu6 - pdt_nsc_dpdk_cpu6
   = -21.42857142857143
 ns_single_cpu7: 3.5 rows_tot_cpu7 - ns_single_cpu7 + pdt_m_rows_cpu7 = 0
 capacity_cores_cpu7: cores_sketch_cpu7 + cores_dpdk_cpu7 <= 7
 ns_sketch_cpu7: - ns_single_cpu7 + pdt_nsk_c_cpu7 = 0
 ns_dpdk_cpu7: 7.142857142857143 cores_dpdk_cpu7 - pdt_nsc_dpdk_cpu7
   = -21.42857142857143
 ns_single_cpu8: 3.5 rows_tot_cpu8 - ns_single_cpu8 + pdt_m_rows_cpu8 = 0
 capacity_cores_cpu8: cores_sketch_cpu8 + cores_dpdk_cpu8 <= 7
 ns_sketch_cpu8: - ns_single_cpu8 + pdt_nsk_c_cpu8 = 0
 ns_dpdk_cpu8: 7.142857142857143 cores_dpdk_cpu8 - pdt_nsc_dpdk_cpu8
   = -21.42857142857143
 ns_tor_p41: ns_tor_p41 = 6.756756756756757
 ns_tor_p42: ns_tor_p42 = 6.756756756756757
 ns_tor_p43: ns_tor_p43 = 6.756756756756757
 ns_tor_p44: ns_tor_p44 = 6.756756756756757
 ns_l1_p41: ns_l1_p41 = 6.756756756756757
 ns_l1_p42: ns_l1_p42 = 6.756756756756757
 ns_l2_p4: ns_l2_p4 = 6.756756756756757
 res_acc: 1.220703125e-04 mem_tot_cpu1 + 1.220703125e-04 mem_tot_cpu2
   + 1.220703125e-04 mem_tot_cpu3 + 1.220703125e-04 mem_tot_cpu4
   + 1.220703125e-04 mem_tot_cpu5 + 1.220703125e-04 mem_tot_cpu6
   + 1.220703125e-04 mem_tot_cpu7 + 1.220703125e-04 mem_tot_cpu8
   + 0.0208333333333333 mem_tot_tor_p41 + 0.25 rows_tot_tor_p41
   + 0.0208333333333333 mem_tot_tor_p42 + 0.25 rows_tot_tor_p42
   + 0.0208333333333333 mem_tot_tor_p43 + 0.25 rows_tot_tor_p43
   + 0.0208333333333333 mem_tot_tor_p44 + 0.25 rows_tot_tor_p44
   + 0.0208333333333333 mem_tot_l1_p41 + 0.25 rows_tot_l1_p41
   + 0.0208333333333333 mem_tot_l1_p42 + 0.25 rows_tot_l1_p42
   + 0.0208333333333333 mem_tot_l2_p4 + 0.25 rows_tot_l2_p4
   + 10 cores_sketch_cpu1 + 10 cores_dpdk_cpu1 + 10 cores_sketch_cpu2
   + 10 cores_dpdk_cpu2 + 10 cores_sketch_cpu3 + 10 cores_dpdk_cpu3
   + 10 cores_sketch_cpu4 + 10 cores_dpdk_cpu4 + 10 cores_sketch_cpu5
   + 10 cores_dpdk_cpu5 + 10 cores_sketch_cpu6 + 10 cores_dpdk_cpu6
   + 10 cores_sketch_cpu7 + 10 cores_dpdk_cpu7 + 10 cores_sketch_cpu8
   + 10 cores_dpdk_cpu8 - res = 0
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
Bounds
 mem_tot_cpu1 <= 32768
 mem_tot_cpu2 <= 32768
 mem_tot_cpu3 <= 32768
 mem_tot_cpu4 <= 32768
 mem_tot_cpu5 <= 32768
 mem_tot_cpu6 <= 32768
 mem_tot_cpu7 <= 32768
 mem_tot_cpu8 <= 32768
 mem_tot_tor_p41 <= 576
 mem_tot_tor_p42 <= 576
 mem_tot_tor_p43 <= 576
 mem_tot_tor_p44 <= 576
 mem_tot_l1_p41 <= 576
 mem_tot_l1_p42 <= 576
 mem_tot_l2_p4 <= 576
 cores_sketch_cpu1 <= 7
 1 <= cores_dpdk_cpu1 <= 7
 cores_sketch_cpu2 <= 7
 1 <= cores_dpdk_cpu2 <= 7
 cores_sketch_cpu3 <= 7
 1 <= cores_dpdk_cpu3 <= 7
 cores_sketch_cpu4 <= 7
 1 <= cores_dpdk_cpu4 <= 7
 cores_sketch_cpu5 <= 7
 1 <= cores_dpdk_cpu5 <= 7
 cores_sketch_cpu6 <= 7
 1 <= cores_dpdk_cpu6 <= 7
 cores_sketch_cpu7 <= 7
 1 <= cores_dpdk_cpu7 <= 7
 cores_sketch_cpu8 <= 7
 1 <= cores_dpdk_cpu8 <= 7
Binaries
 frac[0,0] frac[0,1] frac[0,2] frac[0,3] frac[0,4] frac[0,5] frac[0,6]
 frac[0,7] frac[0,8] frac[0,9] frac[0,10] frac[0,11] frac[0,12] frac[0,13]
 frac[0,14] frac[0,15] frac[0,16] frac[0,17] frac[0,18] frac[0,19]
 frac[0,20] frac[0,21] frac[0,22] frac[0,23] frac[0,24] frac[0,25]
 frac[0,26] frac[0,27] frac[0,28] frac[0,29] frac[0,30] frac[0,31]
 frac[0,32] frac[0,33] frac[0,34] frac[0,35] frac[0,36] frac[0,37]
 frac[0,38] frac[0,39] frac[0,40] frac[0,41] frac[0,42] frac[0,43]
 frac[0,44] frac[0,45] frac[0,46] frac[0,47] frac[0,48] frac[0,49]
 frac[0,50] frac[0,51] frac[0,52] frac[0,53] frac[0,54] frac[0,55]
 frac[0,56] frac[0,57] frac[0,58] frac[0,59] frac[0,60] frac[0,61]
 frac[0,62] frac[0,63] frac[0,64] frac[0,65] frac[0,66] frac[0,67]
 frac[0,68] frac[0,69] frac[0,70] frac[0,71] frac[0,72] frac[0,73]
 frac[0,74] frac[0,75] frac[0,76] frac[0,77] frac[0,78] frac[0,79]
 frac[1,0] frac[1,1] frac[1,2] frac[1,3] frac[1,4] frac[1,5] frac[1,6]
 frac[1,7] frac[1,8] frac[1,9] frac[1,10] frac[1,11] frac[1,12] frac[1,13]
 frac[1,14] frac[1,15] frac[1,16] frac[1,17] frac[1,18] frac[1,19]
 frac[1,20] frac[1,21] frac[1,22] frac[1,23] frac[1,24] frac[1,25]
 frac[1,26] frac[1,27] frac[1,28] frac[1,29] frac[1,30] frac[1,31]
 frac[1,32] frac[1,33] frac[1,34] frac[1,35] frac[1,36] frac[1,37]
 frac[1,38] frac[1,39] frac[1,40] frac[1,41] frac[1,42] frac[1,43]
 frac[1,44] frac[1,45] frac[1,46] frac[1,47] frac[1,48] frac[1,49]
 frac[1,50] frac[1,51] frac[1,52] frac[1,53] frac[1,54] frac[1,55]
 frac[1,56] frac[1,57] frac[1,58] frac[1,59] frac[1,60] frac[1,61]
 frac[1,62] frac[1,63] frac[1,64] frac[1,65] frac[1,66] frac[1,67]
 frac[1,68] frac[1,69] frac[1,70] frac[1,71] frac[1,72] frac[1,73]
 frac[1,74] frac[1,75] frac[1,76] frac[1,77] frac[1,78] frac[1,79]
 frac[2,0] frac[2,1] frac[2,2] frac[2,3] frac[2,4] frac[2,5] frac[2,6]
 frac[2,7] frac[2,8] frac[2,9] frac[2,10] frac[2,11] frac[2,12] frac[2,13]
 frac[2,14] frac[2,15] frac[2,16] frac[2,17] frac[2,18] frac[2,19]
 frac[2,20] frac[2,21] frac[2,22] frac[2,23] frac[2,24] frac[2,25]
 frac[2,26] frac[2,27] frac[2,28] frac[2,29] frac[2,30] frac[2,31]
 frac[2,32] frac[2,33] frac[2,34] frac[2,35] frac[2,36] frac[2,37]
 frac[2,38] frac[2,39] frac[2,40] frac[2,41] frac[2,42] frac[2,43]
 frac[2,44] frac[2,45] frac[2,46] frac[2,47] frac[2,48] frac[2,49]
 frac[2,50] frac[2,51] frac[2,52] frac[2,53] frac[2,54] frac[2,55]
 frac[2,56] frac[2,57] frac[2,58] frac[2,59] frac[2,60] frac[2,61]
 frac[2,62] frac[2,63] frac[2,64] frac[2,65] frac[2,66] frac[2,67]
 frac[2,68] frac[2,69] frac[2,70] frac[2,71] frac[2,72] frac[2,73]
 frac[2,74] frac[2,75] frac[2,76] frac[2,77] frac[2,78] frac[2,79]
 frac[3,0] frac[3,1] frac[3,2] frac[3,3] frac[3,4] frac[3,5] frac[3,6]
 frac[3,7] frac[3,8] frac[3,9] frac[3,10] frac[3,11] frac[3,12] frac[3,13]
 frac[3,14] frac[3,15] frac[3,16] frac[3,17] frac[3,18] frac[3,19]
 frac[3,20] frac[3,21] frac[3,22] frac[3,23] frac[3,24] frac[3,25]
 frac[3,26] frac[3,27] frac[3,28] frac[3,29] frac[3,30] frac[3,31]
 frac[3,32] frac[3,33] frac[3,34] frac[3,35] frac[3,36] frac[3,37]
 frac[3,38] frac[3,39] frac[3,40] frac[3,41] frac[3,42] frac[3,43]
 frac[3,44] frac[3,45] frac[3,46] frac[3,47] frac[3,48] frac[3,49]
 frac[3,50] frac[3,51] frac[3,52] frac[3,53] frac[3,54] frac[3,55]
 frac[3,56] frac[3,57] frac[3,58] frac[3,59] frac[3,60] frac[3,61]
 frac[3,62] frac[3,63] frac[3,64] frac[3,65] frac[3,66] frac[3,67]
 frac[3,68] frac[3,69] frac[3,70] frac[3,71] frac[3,72] frac[3,73]
 frac[3,74] frac[3,75] frac[3,76] frac[3,77] frac[3,78] frac[3,79]
 frac[4,0] frac[4,1] frac[4,2] frac[4,3] frac[4,4] frac[4,5] frac[4,6]
 frac[4,7] frac[4,8] frac[4,9] frac[4,10] frac[4,11] frac[4,12] frac[4,13]
 frac[4,14] frac[4,15] frac[4,16] frac[4,17] frac[4,18] frac[4,19]
 frac[4,20] frac[4,21] frac[4,22] frac[4,23] frac[4,24] frac[4,25]
 frac[4,26] frac[4,27] frac[4,28] frac[4,29] frac[4,30] frac[4,31]
 frac[4,32] frac[4,33] frac[4,34] frac[4,35] frac[4,36] frac[4,37]
 frac[4,38] frac[4,39] frac[4,40] frac[4,41] frac[4,42] frac[4,43]
 frac[4,44] frac[4,45] frac[4,46] frac[4,47] frac[4,48] frac[4,49]
 frac[4,50] frac[4,51] frac[4,52] frac[4,53] frac[4,54] frac[4,55]
 frac[4,56] frac[4,57] frac[4,58] frac[4,59] frac[4,60] frac[4,61]
 frac[4,62] frac[4,63] frac[4,64] frac[4,65] frac[4,66] frac[4,67]
 frac[4,68] frac[4,69] frac[4,70] frac[4,71] frac[4,72] frac[4,73]
 frac[4,74] frac[4,75] frac[4,76] frac[4,77] frac[4,78] frac[4,79]
 frac[5,0] frac[5,1] frac[5,2] frac[5,3] frac[5,4] frac[5,5] frac[5,6]
 frac[5,7] frac[5,8] frac[5,9] frac[5,10] frac[5,11] frac[5,12] frac[5,13]
 frac[5,14] frac[5,15] frac[5,16] frac[5,17] frac[5,18] frac[5,19]
 frac[5,20] frac[5,21] frac[5,22] frac[5,23] frac[5,24] frac[5,25]
 frac[5,26] frac[5,27] frac[5,28] frac[5,29] frac[5,30] frac[5,31]
 frac[5,32] frac[5,33] frac[5,34] frac[5,35] frac[5,36] frac[5,37]
 frac[5,38] frac[5,39] frac[5,40] frac[5,41] frac[5,42] frac[5,43]
 frac[5,44] frac[5,45] frac[5,46] frac[5,47] frac[5,48] frac[5,49]
 frac[5,50] frac[5,51] frac[5,52] frac[5,53] frac[5,54] frac[5,55]
 frac[5,56] frac[5,57] frac[5,58] frac[5,59] frac[5,60] frac[5,61]
 frac[5,62] frac[5,63] frac[5,64] frac[5,65] frac[5,66] frac[5,67]
 frac[5,68] frac[5,69] frac[5,70] frac[5,71] frac[5,72] frac[5,73]
 frac[5,74] frac[5,75] frac[5,76] frac[5,77] frac[5,78] frac[5,79]
 frac[6,0] frac[6,1] frac[6,2] frac[6,3] frac[6,4] frac[6,5] frac[6,6]
 frac[6,7] frac[6,8] frac[6,9] frac[6,10] frac[6,11] frac[6,12] frac[6,13]
 frac[6,14] frac[6,15] frac[6,16] frac[6,17] frac[6,18] frac[6,19]
 frac[6,20] frac[6,21] frac[6,22] frac[6,23] frac[6,24] frac[6,25]
 frac[6,26] frac[6,27] frac[6,28] frac[6,29] frac[6,30] frac[6,31]
 frac[6,32] frac[6,33] frac[6,34] frac[6,35] frac[6,36] frac[6,37]
 frac[6,38] frac[6,39] frac[6,40] frac[6,41] frac[6,42] frac[6,43]
 frac[6,44] frac[6,45] frac[6,46] frac[6,47] frac[6,48] frac[6,49]
 frac[6,50] frac[6,51] frac[6,52] frac[6,53] frac[6,54] frac[6,55]
 frac[6,56] frac[6,57] frac[6,58] frac[6,59] frac[6,60] frac[6,61]
 frac[6,62] frac[6,63] frac[6,64] frac[6,65] frac[6,66] frac[6,67]
 frac[6,68] frac[6,69] frac[6,70] frac[6,71] frac[6,72] frac[6,73]
 frac[6,74] frac[6,75] frac[6,76] frac[6,77] frac[6,78] frac[6,79]
 frac[7,0] frac[7,1] frac[7,2] frac[7,3] frac[7,4] frac[7,5] frac[7,6]
 frac[7,7] frac[7,8] frac[7,9] frac[7,10] frac[7,11] frac[7,12] frac[7,13]
 frac[7,14] frac[7,15] frac[7,16] frac[7,17] frac[7,18] frac[7,19]
 frac[7,20] frac[7,21] frac[7,22] frac[7,23] frac[7,24] frac[7,25]
 frac[7,26] frac[7,27] frac[7,28] frac[7,29] frac[7,30] frac[7,31]
 frac[7,32] frac[7,33] frac[7,34] frac[7,35] frac[7,36] frac[7,37]
 frac[7,38] frac[7,39] frac[7,40] frac[7,41] frac[7,42] frac[7,43]
 frac[7,44] frac[7,45] frac[7,46] frac[7,47] frac[7,48] frac[7,49]
 frac[7,50] frac[7,51] frac[7,52] frac[7,53] frac[7,54] frac[7,55]
 frac[7,56] frac[7,57] frac[7,58] frac[7,59] frac[7,60] frac[7,61]
 frac[7,62] frac[7,63] frac[7,64] frac[7,65] frac[7,66] frac[7,67]
 frac[7,68] frac[7,69] frac[7,70] frac[7,71] frac[7,72] frac[7,73]
 frac[7,74] frac[7,75] frac[7,76] frac[7,77] frac[7,78] frac[7,79]
 frac[8,0] frac[8,1] frac[8,2] frac[8,3] frac[8,4] frac[8,5] frac[8,6]
 frac[8,7] frac[8,8] frac[8,9] frac[8,10] frac[8,11] frac[8,12] frac[8,13]
 frac[8,14] frac[8,15] frac[8,16] frac[8,17] frac[8,18] frac[8,19]
 frac[8,20] frac[8,21] frac[8,22] frac[8,23] frac[8,24] frac[8,25]
 frac[8,26] frac[8,27] frac[8,28] frac[8,29] frac[8,30] frac[8,31]
 frac[8,32] frac[8,33] frac[8,34] frac[8,35] frac[8,36] frac[8,37]
 frac[8,38] frac[8,39] frac[8,40] frac[8,41] frac[8,42] frac[8,43]
 frac[8,44] frac[8,45] frac[8,46] frac[8,47] frac[8,48] frac[8,49]
 frac[8,50] frac[8,51] frac[8,52] frac[8,53] frac[8,54] frac[8,55]
 frac[8,56] frac[8,57] frac[8,58] frac[8,59] frac[8,60] frac[8,61]
 frac[8,62] frac[8,63] frac[8,64] frac[8,65] frac[8,66] frac[8,67]
 frac[8,68] frac[8,69] frac[8,70] frac[8,71] frac[8,72] frac[8,73]
 frac[8,74] frac[8,75] frac[8,76] frac[8,77] frac[8,78] frac[8,79]
 frac[9,0] frac[9,1] frac[9,2] frac[9,3] frac[9,4] frac[9,5] frac[9,6]
 frac[9,7] frac[9,8] frac[9,9] frac[9,10] frac[9,11] frac[9,12] frac[9,13]
 frac[9,14] frac[9,15] frac[9,16] frac[9,17] frac[9,18] frac[9,19]
 frac[9,20] frac[9,21] frac[9,22] frac[9,23] frac[9,24] frac[9,25]
 frac[9,26] frac[9,27] frac[9,28] frac[9,29] frac[9,30] frac[9,31]
 frac[9,32] frac[9,33] frac[9,34] frac[9,35] frac[9,36] frac[9,37]
 frac[9,38] frac[9,39] frac[9,40] frac[9,41] frac[9,42] frac[9,43]
 frac[9,44] frac[9,45] frac[9,46] frac[9,47] frac[9,48] frac[9,49]
 frac[9,50] frac[9,51] frac[9,52] frac[9,53] frac[9,54] frac[9,55]
 frac[9,56] frac[9,57] frac[9,58] frac[9,59] frac[9,60] frac[9,61]
 frac[9,62] frac[9,63] frac[9,64] frac[9,65] frac[9,66] frac[9,67]
 frac[9,68] frac[9,69] frac[9,70] frac[9,71] frac[9,72] frac[9,73]
 frac[9,74] frac[9,75] frac[9,76] frac[9,77] frac[9,78] frac[9,79]
 frac[10,0] frac[10,1] frac[10,2] frac[10,3] frac[10,4] frac[10,5]
 frac[10,6] frac[10,7] frac[10,8] frac[10,9] frac[10,10] frac[10,11]
 frac[10,12] frac[10,13] frac[10,14] frac[10,15] frac[10,16] frac[10,17]
 frac[10,18] frac[10,19] frac[10,20] frac[10,21] frac[10,22] frac[10,23]
 frac[10,24] frac[10,25] frac[10,26] frac[10,27] frac[10,28] frac[10,29]
 frac[10,30] frac[10,31] frac[10,32] frac[10,33] frac[10,34] frac[10,35]
 frac[10,36] frac[10,37] frac[10,38] frac[10,39] frac[10,40] frac[10,41]
 frac[10,42] frac[10,43] frac[10,44] frac[10,45] frac[10,46] frac[10,47]
 frac[10,48] frac[10,49] frac[10,50] frac[10,51] frac[10,52] frac[10,53]
 frac[10,54] frac[10,55] frac[10,56] frac[10,57] frac[10,58] frac[10,59]
 frac[10,60] frac[10,61] frac[10,62] frac[10,63] frac[10,64] frac[10,65]
 frac[10,66] frac[10,67] frac[10,68] frac[10,69] frac[10,70] frac[10,71]
 frac[10,72] frac[10,73] frac[10,74] frac[10,75] frac[10,76] frac[10,77]
 frac[10,78] frac[10,79] frac[11,0] frac[11,1] frac[11,2] frac[11,3]
 frac[11,4] frac[11,5] frac[11,6] frac[11,7] frac[11,8] frac[11,9]
 frac[11,10] frac[11,11] frac[11,12] frac[11,13] frac[11,14] frac[11,15]
 frac[11,16] frac[11,17] frac[11,18] frac[11,19] frac[11,20] frac[11,21]
 frac[11,22] frac[11,23] frac[11,24] frac[11,25] frac[11,26] frac[11,27]
 frac[11,28] frac[11,29] frac[11,30] frac[11,31] frac[11,32] frac[11,33]
 frac[11,34] frac[11,35] frac[11,36] frac[11,37] frac[11,38] frac[11,39]
 frac[11,40] frac[11,41] frac[11,42] frac[11,43] frac[11,44] frac[11,45]
 frac[11,46] frac[11,47] frac[11,48] frac[11,49] frac[11,50] frac[11,51]
 frac[11,52] frac[11,53] frac[11,54] frac[11,55] frac[11,56] frac[11,57]
 frac[11,58] frac[11,59] frac[11,60] frac[11,61] frac[11,62] frac[11,63]
 frac[11,64] frac[11,65] frac[11,66] frac[11,67] frac[11,68] frac[11,69]
 frac[11,70] frac[11,71] frac[11,72] frac[11,73] frac[11,74] frac[11,75]
 frac[11,76] frac[11,77] frac[11,78] frac[11,79] frac[12,0] frac[12,1]
 frac[12,2] frac[12,3] frac[12,4] frac[12,5] frac[12,6] frac[12,7]
 frac[12,8] frac[12,9] frac[12,10] frac[12,11] frac[12,12] frac[12,13]
 frac[12,14] frac[12,15] frac[12,16] frac[12,17] frac[12,18] frac[12,19]
 frac[12,20] frac[12,21] frac[12,22] frac[12,23] frac[12,24] frac[12,25]
 frac[12,26] frac[12,27] frac[12,28] frac[12,29] frac[12,30] frac[12,31]
 frac[12,32] frac[12,33] frac[12,34] frac[12,35] frac[12,36] frac[12,37]
 frac[12,38] frac[12,39] frac[12,40] frac[12,41] frac[12,42] frac[12,43]
 frac[12,44] frac[12,45] frac[12,46] frac[12,47] frac[12,48] frac[12,49]
 frac[12,50] frac[12,51] frac[12,52] frac[12,53] frac[12,54] frac[12,55]
 frac[12,56] frac[12,57] frac[12,58] frac[12,59] frac[12,60] frac[12,61]
 frac[12,62] frac[12,63] frac[12,64] frac[12,65] frac[12,66] frac[12,67]
 frac[12,68] frac[12,69] frac[12,70] frac[12,71] frac[12,72] frac[12,73]
 frac[12,74] frac[12,75] frac[12,76] frac[12,77] frac[12,78] frac[12,79]
 frac[13,0] frac[13,1] frac[13,2] frac[13,3] frac[13,4] frac[13,5]
 frac[13,6] frac[13,7] frac[13,8] frac[13,9] frac[13,10] frac[13,11]
 frac[13,12] frac[13,13] frac[13,14] frac[13,15] frac[13,16] frac[13,17]
 frac[13,18] frac[13,19] frac[13,20] frac[13,21] frac[13,22] frac[13,23]
 frac[13,24] frac[13,25] frac[13,26] frac[13,27] frac[13,28] frac[13,29]
 frac[13,30] frac[13,31] frac[13,32] frac[13,33] frac[13,34] frac[13,35]
 frac[13,36] frac[13,37] frac[13,38] frac[13,39] frac[13,40] frac[13,41]
 frac[13,42] frac[13,43] frac[13,44] frac[13,45] frac[13,46] frac[13,47]
 frac[13,48] frac[13,49] frac[13,50] frac[13,51] frac[13,52] frac[13,53]
 frac[13,54] frac[13,55] frac[13,56] frac[13,57] frac[13,58] frac[13,59]
 frac[13,60] frac[13,61] frac[13,62] frac[13,63] frac[13,64] frac[13,65]
 frac[13,66] frac[13,67] frac[13,68] frac[13,69] frac[13,70] frac[13,71]
 frac[13,72] frac[13,73] frac[13,74] frac[13,75] frac[13,76] frac[13,77]
 frac[13,78] frac[13,79] frac[14,0] frac[14,1] frac[14,2] frac[14,3]
 frac[14,4] frac[14,5] frac[14,6] frac[14,7] frac[14,8] frac[14,9]
 frac[14,10] frac[14,11] frac[14,12] frac[14,13] frac[14,14] frac[14,15]
 frac[14,16] frac[14,17] frac[14,18] frac[14,19] frac[14,20] frac[14,21]
 frac[14,22] frac[14,23] frac[14,24] frac[14,25] frac[14,26] frac[14,27]
 frac[14,28] frac[14,29] frac[14,30] frac[14,31] frac[14,32] frac[14,33]
 frac[14,34] frac[14,35] frac[14,36] frac[14,37] frac[14,38] frac[14,39]
 frac[14,40] frac[14,41] frac[14,42] frac[14,43] frac[14,44] frac[14,45]
 frac[14,46] frac[14,47] frac[14,48] frac[14,49] frac[14,50] frac[14,51]
 frac[14,52] frac[14,53] frac[14,54] frac[14,55] frac[14,56] frac[14,57]
 frac[14,58] frac[14,59] frac[14,60] frac[14,61] frac[14,62] frac[14,63]
 frac[14,64] frac[14,65] frac[14,66] frac[14,67] frac[14,68] frac[14,69]
 frac[14,70] frac[14,71] frac[14,72] frac[14,73] frac[14,74] frac[14,75]
 frac[14,76] frac[14,77] frac[14,78] frac[14,79]
Generals
 cores_sketch_cpu1 cores_dpdk_cpu1 cores_sketch_cpu2 cores_dpdk_cpu2
 cores_sketch_cpu3 cores_dpdk_cpu3 cores_sketch_cpu4 cores_dpdk_cpu4
 cores_sketch_cpu5 cores_dpdk_cpu5 cores_sketch_cpu6 cores_dpdk_cpu6
 cores_sketch_cpu7 cores_dpdk_cpu7 cores_sketch_cpu8 cores_dpdk_cpu8
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
 ns_overall: ns = MAX ( ns_cpu1 , ns_cpu2 , ns_cpu3 , ns_cpu4 , ns_cpu5 ,
   ns_cpu6 , ns_cpu7 , ns_cpu8 , ns_tor_p41 , ns_tor_p42 , ns_tor_p43 ,
   ns_tor_p44 , ns_l1_p41 , ns_l1_p42 , ns_l2_p4 )
End

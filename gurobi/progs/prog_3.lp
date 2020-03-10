\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 m_access_time_CPU1 + 0 ns_sketch_CPU1 + 0 ns_dpdk_CPU1 + 0 ns_CPU1
   + 0 m_access_time_CPU2 + 0 ns_sketch_CPU2 + 0 ns_dpdk_CPU2 + 0 ns_CPU2
   + 0 m_access_time_CPU3 + 0 ns_sketch_CPU3 + 0 ns_dpdk_CPU3 + 0 ns_CPU3
   + 0 m_access_time_CPU4 + 0 ns_sketch_CPU4 + 0 ns_dpdk_CPU4 + 0 ns_CPU4
   + 0 m_access_time_CPU5 + 0 ns_sketch_CPU5 + 0 ns_dpdk_CPU5 + 0 ns_CPU5
   + 0 m_access_time_CPU6 + 0 ns_sketch_CPU6 + 0 ns_dpdk_CPU6 + 0 ns_CPU6
   + 0 m_access_time_CPU7 + 0 ns_sketch_CPU7 + 0 ns_dpdk_CPU7 + 0 ns_CPU7
   + 0 m_access_time_CPU8 + 0 ns_sketch_CPU8 + 0 ns_dpdk_CPU8 + 0 ns_CPU8
   + ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0
   res
Subject To
 cov_0_47: frac[10,47] + frac[8,47] + frac[14,47] + frac[13,47]
   + frac[1,47] + frac[5,47] + frac[12,47] >= 0.8
 cov_1_35: frac[13,35] + frac[5,35] + frac[6,35] + frac[10,35]
   + frac[11,35] >= 0.7
 cov_2_5: frac[8,5] + frac[0,5] + frac[14,5] + frac[6,5] + frac[11,5]
   + frac[13,5] + frac[12,5] >= 1
 cov_3_35: frac[4,35] + frac[13,35] + frac[7,35] + frac[10,35]
   + frac[11,35] >= 1
 cov_4_13: frac[13,13] + frac[3,13] + frac[12,13] + frac[9,13]
   + frac[11,13] + frac[7,13] + frac[14,13] >= 0.7
 cov_5_25: frac[8,25] + frac[1,25] + frac[12,25] + frac[3,25] + frac[9,25]
   >= 0.8
 cov_6_40: frac[0,40] + frac[8,40] + frac[13,40] + frac[12,40]
   + frac[14,40] + frac[5,40] + frac[10,40] >= 0.7
 cov_7_0: frac[8,0] + frac[0,0] + frac[14,0] + frac[13,0] + frac[5,0]
   + frac[12,0] + frac[10,0] >= 0.9
 cov_8_36: frac[4,36] + frac[10,36] + frac[0,36] + frac[14,36]
   + frac[13,36] + frac[12,36] + frac[8,36] >= 0.8
 cov_9_23: frac[13,23] + frac[14,23] + frac[11,23] + frac[7,23]
   + frac[12,23] + frac[1,23] + frac[8,23] >= 0.9
 cov_10_57: frac[5,57] + frac[3,57] + frac[9,57] + frac[10,57]
   + frac[14,57] + frac[13,57] + frac[12,57] >= 0.8
 cov_11_24: frac[7,24] + frac[6,24] + frac[11,24] >= 0.8
 cov_12_17: frac[12,17] + frac[9,17] + frac[14,17] + frac[10,17]
   + frac[13,17] + frac[2,17] + frac[5,17] >= 0.9
 cov_13_37: frac[13,37] + frac[3,37] + frac[9,37] + frac[12,37]
   + frac[14,37] + frac[6,37] + frac[11,37] >= 0.7
 cov_14_46: frac[11,46] + frac[1,46] + frac[7,46] + frac[12,46]
   + frac[8,46] + frac[14,46] + frac[13,46] >= 1
 cov_15_48: frac[3,48] + frac[9,48] + frac[8,48] + frac[12,48] + frac[0,48]
   >= 0.9
 cov_16_37: frac[13,37] + frac[4,37] + frac[10,37] + frac[6,37]
   + frac[11,37] >= 0.7
 cov_17_48: frac[10,48] + frac[14,48] + frac[8,48] + frac[13,48]
   + frac[1,48] + frac[12,48] + frac[5,48] >= 0.8
 cov_18_31: frac[3,31] + frac[2,31] + frac[9,31] >= 0.7
 cov_19_39: frac[9,39] + frac[11,39] + frac[3,39] + frac[12,39]
   + frac[14,39] + frac[6,39] + frac[13,39] >= 1
 cov_20_42: frac[6,42] + frac[7,42] + frac[11,42] >= 0.7
 cov_21_3: frac[1,3] + frac[8,3] + frac[0,3] >= 0.8
 cov_22_30: frac[0,30] + frac[9,30] + frac[12,30] + frac[3,30] + frac[8,30]
   >= 0.8
 cov_23_53: frac[14,53] + frac[12,53] + frac[6,53] + frac[11,53]
   + frac[13,53] + frac[3,53] + frac[9,53] >= 0.7
 cov_24_39: frac[9,39] + frac[5,39] + frac[3,39] + frac[10,39]
   + frac[12,39] + frac[14,39] + frac[13,39] >= 1
 cov_25_58: frac[13,58] + frac[1,58] + frac[8,58] + frac[12,58]
   + frac[10,58] + frac[14,58] + frac[4,58] >= 1
 cov_26_63: frac[4,63] + frac[13,63] + frac[11,63] + frac[10,63]
   + frac[6,63] >= 0.9
 cov_27_5: frac[14,5] + frac[11,5] + frac[7,5] + frac[3,5] + frac[13,5]
   + frac[9,5] + frac[12,5] >= 0.7
 cov_28_3: frac[11,3] + frac[1,3] + frac[12,3] + frac[13,3] + frac[8,3]
   + frac[14,3] + frac[6,3] >= 0.8
 cov_29_26: frac[9,26] + frac[2,26] + frac[1,26] + frac[8,26] + frac[12,26]
   >= 0.7
 cov_30_2: frac[8,2] + frac[12,2] + frac[2,2] + frac[9,2] + frac[1,2]
   >= 0.7
 cov_31_42: frac[13,42] + frac[7,42] + frac[5,42] + frac[11,42]
   + frac[10,42] >= 1
 cov_32_56: frac[13,56] + frac[6,56] + frac[11,56] + frac[5,56]
   + frac[10,56] >= 1
 cov_33_35: frac[4,35] + frac[13,35] + frac[6,35] + frac[10,35]
   + frac[11,35] >= 0.8
 cov_34_54: frac[6,54] + frac[1,54] + frac[13,54] + frac[14,54]
   + frac[11,54] + frac[12,54] + frac[8,54] >= 0.9
 cov_35_39: frac[11,39] + frac[1,39] + frac[7,39] + frac[12,39]
   + frac[8,39] + frac[14,39] + frac[13,39] >= 0.9
 cov_36_18: frac[8,18] + frac[1,18] + frac[13,18] + frac[7,18]
   + frac[14,18] + frac[12,18] + frac[11,18] >= 0.8
 cov_37_23: frac[13,23] + frac[5,23] + frac[11,23] + frac[7,23]
   + frac[10,23] >= 1
 cov_38_10: frac[5,10] + frac[12,10] + frac[10,10] + frac[0,10]
   + frac[13,10] + frac[8,10] + frac[14,10] >= 0.7
 cov_39_0: frac[7,0] + frac[6,0] + frac[11,0] >= 0.7
 cov_40_47: frac[10,47] + frac[8,47] + frac[14,47] + frac[4,47]
   + frac[13,47] + frac[1,47] + frac[12,47] >= 0.9
 cov_41_19: frac[9,19] + frac[14,19] + frac[11,19] + frac[12,19]
   + frac[13,19] + frac[2,19] + frac[7,19] >= 0.9
 cov_42_35: frac[6,35] + frac[7,35] + frac[11,35] >= 0.9
 cov_43_51: frac[2,51] + frac[9,51] + frac[3,51] >= 0.7
 cov_44_51: frac[13,51] + frac[5,51] + frac[7,51] + frac[10,51]
   + frac[11,51] >= 0.9
 cov_45_13: frac[13,13] + frac[12,13] + frac[9,13] + frac[4,13]
   + frac[2,13] + frac[10,13] + frac[14,13] >= 0.7
 cov_46_16: frac[13,16] + frac[5,16] + frac[0,16] + frac[12,16]
   + frac[14,16] + frac[10,16] + frac[8,16] >= 0.9
 cov_47_35: frac[13,35] + frac[14,35] + frac[6,35] + frac[9,35]
   + frac[11,35] + frac[12,35] + frac[2,35] >= 0.8
 cov_48_23: frac[13,23] + frac[9,23] + frac[5,23] + frac[14,23]
   + frac[3,23] + frac[10,23] + frac[12,23] >= 1
 cov_49_54: frac[6,54] + frac[13,54] + frac[11,54] + frac[5,54]
   + frac[10,54] >= 0.7
 cov_50_12: frac[6,12] + frac[11,12] + frac[7,12] >= 0.7
 cov_51_33: frac[5,33] + frac[4,33] + frac[10,33] >= 0.7
 cov_52_40: frac[1,40] + frac[8,40] + frac[13,40] + frac[6,40]
   + frac[12,40] + frac[14,40] + frac[11,40] >= 0.7
 cov_53_1: frac[12,1] + frac[9,1] + frac[2,1] + frac[8,1] + frac[0,1]
   >= 0.7
 cov_54_39: frac[9,39] + frac[3,39] + frac[12,39] + frac[8,39] + frac[0,39]
   >= 0.7
 cov_55_45: frac[8,45] + frac[4,45] + frac[10,45] + frac[14,45]
   + frac[13,45] + frac[1,45] + frac[12,45] >= 1
 cov_56_26: frac[1,26] + frac[8,26] + frac[0,26] >= 0.9
 cov_57_26: frac[11,26] + frac[14,26] + frac[13,26] + frac[8,26]
   + frac[7,26] + frac[0,26] + frac[12,26] >= 0.7
 cov_58_14: frac[10,14] + frac[14,14] + frac[4,14] + frac[13,14]
   + frac[9,14] + frac[12,14] + frac[2,14] >= 0.9
 cov_59_46: frac[9,46] + frac[5,46] + frac[12,46] + frac[3,46]
   + frac[10,46] + frac[14,46] + frac[13,46] >= 0.7
 cov_60_4: frac[11,4] + frac[7,4] + frac[13,4] + frac[9,4] + frac[12,4]
   + frac[2,4] + frac[14,4] >= 0.7
 cov_61_29: frac[12,29] + frac[9,29] + frac[8,29] + frac[1,29] + frac[3,29]
   >= 0.7
 cov_62_16: frac[7,16] + frac[13,16] + frac[5,16] + frac[11,16]
   + frac[10,16] >= 1
 cov_63_19: frac[9,19] + frac[14,19] + frac[10,19] + frac[12,19]
   + frac[13,19] + frac[2,19] + frac[5,19] >= 0.8
 cov_64_53: frac[5,53] + frac[14,53] + frac[12,53] + frac[8,53]
   + frac[10,53] + frac[1,53] + frac[13,53] >= 1
 cov_65_62: frac[9,62] + frac[3,62] + frac[2,62] >= 0.8
 cov_66_9: frac[9,9] + frac[10,9] + frac[13,9] + frac[14,9] + frac[12,9]
   + frac[5,9] + frac[3,9] >= 0.8
 cov_67_43: frac[14,43] + frac[12,43] + frac[2,43] + frac[9,43]
   + frac[10,43] + frac[4,43] + frac[13,43] >= 0.8
 cov_68_52: frac[1,52] + frac[2,52] + frac[9,52] + frac[12,52] + frac[8,52]
   >= 0.9
 cov_69_35: frac[13,35] + frac[14,35] + frac[6,35] + frac[9,35]
   + frac[11,35] + frac[12,35] + frac[2,35] >= 0.7
 cov_70_53: frac[8,53] + frac[1,53] + frac[0,53] >= 0.8
 cov_71_39: frac[1,39] + frac[10,39] + frac[12,39] + frac[8,39]
   + frac[14,39] + frac[4,39] + frac[13,39] >= 1
 cov_72_18: frac[8,18] + frac[13,18] + frac[14,18] + frac[10,18]
   + frac[0,18] + frac[12,18] + frac[4,18] >= 0.9
 cov_73_27: frac[5,27] + frac[10,27] + frac[4,27] >= 1
 cov_74_39: frac[11,39] + frac[5,39] + frac[7,39] + frac[10,39]
   + frac[13,39] >= 0.9
 cov_75_30: frac[13,30] + frac[11,30] + frac[10,30] + frac[6,30]
   + frac[4,30] >= 0.7
 cov_76_45: frac[2,45] + frac[3,45] + frac[9,45] >= 0.8
 cov_77_31: frac[5,31] + frac[10,31] + frac[6,31] + frac[13,31]
   + frac[11,31] >= 0.8
 cov_78_28: frac[13,28] + frac[12,28] + frac[3,28] + frac[14,28]
   + frac[9,28] + frac[5,28] + frac[10,28] >= 1
 cov_79_59: frac[6,59] + frac[7,59] + frac[11,59] >= 0.8
 cov_80_21: frac[0,21] + frac[13,21] + frac[14,21] + frac[12,21]
   + frac[8,21] + frac[4,21] + frac[10,21] >= 0.7
 cov_81_63: frac[8,63] + frac[13,63] + frac[11,63] + frac[7,63]
   + frac[14,63] + frac[12,63] + frac[1,63] >= 1
 cov_82_44: frac[5,44] + frac[4,44] + frac[10,44] >= 0.8
 cov_83_1: frac[14,1] + frac[6,1] + frac[11,1] + frac[12,1] + frac[3,1]
   + frac[9,1] + frac[13,1] >= 0.9
 cov_84_7: frac[10,7] + frac[6,7] + frac[13,7] + frac[5,7] + frac[11,7]
   >= 0.7
 cov_85_1: frac[14,1] + frac[11,1] + frac[12,1] + frac[1,1] + frac[8,1]
   + frac[7,1] + frac[13,1] >= 0.7
 cov_86_5: frac[10,5] + frac[14,5] + frac[3,5] + frac[13,5] + frac[9,5]
   + frac[5,5] + frac[12,5] >= 0.8
 cov_87_5: frac[14,5] + frac[6,5] + frac[11,5] + frac[3,5] + frac[13,5]
   + frac[9,5] + frac[12,5] >= 1
 cov_88_42: frac[13,42] + frac[14,42] + frac[7,42] + frac[12,42]
   + frac[2,42] + frac[11,42] + frac[9,42] >= 0.8
 cov_89_25: frac[8,25] + frac[0,25] + frac[12,25] + frac[3,25] + frac[9,25]
   >= 0.7
 cov_90_24: frac[5,24] + frac[14,24] + frac[10,24] + frac[8,24]
   + frac[1,24] + frac[13,24] + frac[12,24] >= 0.8
 cov_91_28: frac[1,28] + frac[11,28] + frac[7,28] + frac[13,28]
   + frac[12,28] + frac[14,28] + frac[8,28] >= 1
 cov_92_57: frac[9,57] + frac[14,57] + frac[13,57] + frac[2,57]
   + frac[6,57] + frac[11,57] + frac[12,57] >= 1
 cov_93_6: frac[13,6] + frac[11,6] + frac[7,6] + frac[12,6] + frac[8,6]
   + frac[14,6] + frac[0,6] >= 0.8
 cov_94_43: frac[14,43] + frac[11,43] + frac[12,43] + frac[2,43]
   + frac[9,43] + frac[13,43] + frac[7,43] >= 0.8
 cov_95_0: frac[9,0] + frac[7,0] + frac[14,0] + frac[13,0] + frac[12,0]
   + frac[11,0] + frac[3,0] >= 0.8
 cov_96_2: frac[8,2] + frac[14,2] + frac[6,2] + frac[13,2] + frac[12,2]
   + frac[11,2] + frac[1,2] >= 1
 cov_97_55: frac[0,55] + frac[10,55] + frac[12,55] + frac[8,55]
   + frac[13,55] + frac[5,55] + frac[14,55] >= 0.7
 cov_98_55: frac[3,55] + frac[12,55] + frac[6,55] + frac[13,55]
   + frac[9,55] + frac[11,55] + frac[14,55] >= 0.8
 cov_99_19: frac[10,19] + frac[4,19] + frac[5,19] >= 0.8
 cov_100_46: frac[11,46] + frac[7,46] + frac[9,46] + frac[12,46]
   + frac[2,46] + frac[14,46] + frac[13,46] >= 0.9
 cov_101_26: frac[9,26] + frac[3,26] + frac[8,26] + frac[0,26]
   + frac[12,26] >= 0.8
 cov_102_21: frac[0,21] + frac[13,21] + frac[14,21] + frac[12,21]
   + frac[8,21] + frac[4,21] + frac[10,21] >= 1
 cov_103_54: frac[13,54] + frac[14,54] + frac[0,54] + frac[5,54]
   + frac[10,54] + frac[12,54] + frac[8,54] >= 0.7
 cov_104_16: frac[2,16] + frac[1,16] + frac[12,16] + frac[9,16]
   + frac[8,16] >= 0.8
 cov_105_46: frac[11,46] + frac[7,46] + frac[9,46] + frac[12,46]
   + frac[3,46] + frac[14,46] + frac[13,46] >= 0.8
 cov_106_44: frac[12,44] + frac[3,44] + frac[1,44] + frac[9,44]
   + frac[8,44] >= 0.9
 cov_107_26: frac[11,26] + frac[14,26] + frac[6,26] + frac[1,26]
   + frac[13,26] + frac[8,26] + frac[12,26] >= 1
 cov_108_61: frac[10,61] + frac[0,61] + frac[13,61] + frac[12,61]
   + frac[14,61] + frac[8,61] + frac[4,61] >= 1
 cov_109_53: frac[2,53] + frac[3,53] + frac[9,53] >= 1
 cov_110_13: frac[13,13] + frac[12,13] + frac[0,13] + frac[8,13]
   + frac[6,13] + frac[11,13] + frac[14,13] >= 0.8
 cov_111_55: frac[10,55] + frac[6,55] + frac[13,55] + frac[11,55]
   + frac[5,55] >= 0.9
 cov_112_48: frac[11,48] + frac[7,48] + frac[6,48] >= 0.7
 cov_113_50: frac[11,50] + frac[4,50] + frac[13,50] + frac[7,50]
   + frac[10,50] >= 0.9
 cov_114_18: frac[2,18] + frac[13,18] + frac[14,18] + frac[10,18]
   + frac[12,18] + frac[9,18] + frac[4,18] >= 0.7
 cov_115_33: frac[2,33] + frac[13,33] + frac[14,33] + frac[5,33]
   + frac[12,33] + frac[9,33] + frac[10,33] >= 0.8
 cov_116_63: frac[8,63] + frac[0,63] + frac[1,63] >= 1
 cov_117_21: frac[6,21] + frac[11,21] + frac[13,21] + frac[4,21]
   + frac[10,21] >= 1
 cov_118_30: frac[9,30] + frac[12,30] + frac[3,30] + frac[1,30]
   + frac[8,30] >= 0.9
 cov_119_48: frac[11,48] + frac[10,48] + frac[7,48] + frac[13,48]
   + frac[5,48] >= 1
 cov_120_31: frac[5,31] + frac[14,31] + frac[12,31] + frac[10,31]
   + frac[13,31] + frac[2,31] + frac[9,31] >= 0.7
 cov_121_37: frac[13,37] + frac[12,37] + frac[8,37] + frac[4,37]
   + frac[10,37] + frac[0,37] + frac[14,37] >= 0.8
 cov_122_42: frac[13,42] + frac[8,42] + frac[14,42] + frac[5,42]
   + frac[12,42] + frac[10,42] + frac[1,42] >= 0.7
 cov_123_36: frac[0,36] + frac[2,36] + frac[9,36] + frac[12,36]
   + frac[8,36] >= 1
 cov_124_53: frac[5,53] + frac[14,53] + frac[12,53] + frac[8,53]
   + frac[10,53] + frac[1,53] + frac[13,53] >= 1
 cov_125_57: frac[5,57] + frac[10,57] + frac[13,57] + frac[6,57]
   + frac[11,57] >= 1
 cov_126_24: frac[5,24] + frac[14,24] + frac[10,24] + frac[8,24]
   + frac[13,24] + frac[0,24] + frac[12,24] >= 0.7
 cov_127_5: frac[4,5] + frac[10,5] + frac[11,5] + frac[7,5] + frac[13,5]
   >= 0.7
 cov_128_47: frac[10,47] + frac[14,47] + frac[4,47] + frac[13,47]
   + frac[9,47] + frac[3,47] + frac[12,47] >= 0.9
 cov_129_3: frac[7,3] + frac[11,3] + frac[1,3] + frac[12,3] + frac[13,3]
   + frac[8,3] + frac[14,3] >= 1
 cov_130_36: frac[14,36] + frac[6,36] + frac[11,36] + frac[13,36]
   + frac[3,36] + frac[9,36] + frac[12,36] >= 0.8
 cov_131_21: frac[13,21] + frac[2,21] + frac[9,21] + frac[5,21]
   + frac[14,21] + frac[12,21] + frac[10,21] >= 1
 cov_132_17: frac[12,17] + frac[14,17] + frac[8,17] + frac[7,17]
   + frac[1,17] + frac[13,17] + frac[11,17] >= 1
 cov_133_24: frac[3,24] + frac[9,24] + frac[8,24] + frac[1,24]
   + frac[12,24] >= 0.8
 cov_134_59: frac[12,59] + frac[9,59] + frac[10,59] + frac[14,59]
   + frac[3,59] + frac[4,59] + frac[13,59] >= 0.9
 cov_135_17: frac[12,17] + frac[9,17] + frac[8,17] + frac[1,17]
   + frac[2,17] >= 0.8
 cov_136_15: frac[9,15] + frac[3,15] + frac[12,15] + frac[0,15]
   + frac[8,15] >= 0.8
 cov_137_21: frac[3,21] + frac[13,21] + frac[9,21] + frac[5,21]
   + frac[14,21] + frac[12,21] + frac[10,21] >= 0.7
 cov_138_5: frac[4,5] + frac[10,5] + frac[6,5] + frac[11,5] + frac[13,5]
   >= 0.7
 cov_139_22: frac[11,22] + frac[10,22] + frac[4,22] + frac[6,22]
   + frac[13,22] >= 0.7
 cov_140_7: frac[12,7] + frac[8,7] + frac[0,7] + frac[9,7] + frac[3,7]
   >= 0.9
 cov_141_19: frac[9,19] + frac[12,19] + frac[3,19] + frac[1,19]
   + frac[8,19] >= 1
 cov_142_58: frac[13,58] + frac[8,58] + frac[12,58] + frac[0,58]
   + frac[10,58] + frac[14,58] + frac[4,58] >= 0.9
 cov_143_24: frac[14,24] + frac[8,24] + frac[7,24] + frac[13,24]
   + frac[0,24] + frac[12,24] + frac[11,24] >= 1
 cov_144_12: frac[8,12] + frac[1,12] + frac[0,12] >= 0.9
 cov_145_31: frac[14,31] + frac[12,31] + frac[6,31] + frac[13,31]
   + frac[2,31] + frac[11,31] + frac[9,31] >= 0.8
 cov_146_30: frac[13,30] + frac[14,30] + frac[9,30] + frac[12,30]
   + frac[3,30] + frac[10,30] + frac[4,30] >= 0.9
 cov_147_59: frac[10,59] + frac[4,59] + frac[5,59] >= 0.8
 cov_148_51: frac[5,51] + frac[10,51] + frac[4,51] >= 0.7
 cov_149_50: frac[12,50] + frac[14,50] + frac[0,50] + frac[11,50]
   + frac[8,50] + frac[13,50] + frac[7,50] >= 0.7
 cov_150_40: frac[3,40] + frac[9,40] + frac[13,40] + frac[12,40]
   + frac[14,40] + frac[4,40] + frac[10,40] >= 0.7
 cov_151_32: frac[13,32] + frac[12,32] + frac[5,32] + frac[10,32]
   + frac[8,32] + frac[1,32] + frac[14,32] >= 1
 cov_152_51: frac[13,51] + frac[9,51] + frac[10,51] + frac[12,51]
   + frac[14,51] + frac[3,51] + frac[4,51] >= 1
 cov_153_29: frac[12,29] + frac[9,29] + frac[8,29] + frac[1,29]
   + frac[2,29] >= 0.7
 cov_154_60: frac[14,60] + frac[9,60] + frac[11,60] + frac[7,60]
   + frac[2,60] + frac[13,60] + frac[12,60] >= 0.9
 cov_155_38: frac[8,38] + frac[14,38] + frac[0,38] + frac[13,38]
   + frac[11,38] + frac[7,38] + frac[12,38] >= 1
 cov_156_9: frac[7,9] + frac[8,9] + frac[13,9] + frac[11,9] + frac[14,9]
   + frac[12,9] + frac[1,9] >= 0.9
 cov_157_62: frac[9,62] + frac[5,62] + frac[12,62] + frac[10,62]
   + frac[2,62] + frac[13,62] + frac[14,62] >= 0.9
 cov_158_36: frac[10,36] + frac[14,36] + frac[13,36] + frac[3,36]
   + frac[9,36] + frac[12,36] + frac[5,36] >= 1
 cov_159_55: frac[7,55] + frac[10,55] + frac[4,55] + frac[13,55]
   + frac[11,55] >= 1
 cov_160_21: frac[3,21] + frac[13,21] + frac[9,21] + frac[5,21]
   + frac[14,21] + frac[12,21] + frac[10,21] >= 0.7
 cov_161_52: frac[11,52] + frac[7,52] + frac[13,52] + frac[4,52]
   + frac[10,52] >= 0.9
 cov_162_63: frac[4,63] + frac[13,63] + frac[11,63] + frac[7,63]
   + frac[10,63] >= 0.9
 cov_163_21: frac[1,21] + frac[6,21] + frac[11,21] + frac[13,21]
   + frac[14,21] + frac[12,21] + frac[8,21] >= 0.7
 cov_164_19: frac[14,19] + frac[10,19] + frac[12,19] + frac[4,19]
   + frac[1,19] + frac[13,19] + frac[8,19] >= 0.8
 cov_165_18: frac[2,18] + frac[13,18] + frac[5,18] + frac[14,18]
   + frac[10,18] + frac[12,18] + frac[9,18] >= 1
 cov_166_12: frac[6,12] + frac[14,12] + frac[11,12] + frac[13,12]
   + frac[12,12] + frac[3,12] + frac[9,12] >= 0.9
 cov_167_42: frac[4,42] + frac[3,42] + frac[13,42] + frac[14,42]
   + frac[12,42] + frac[10,42] + frac[9,42] >= 0.7
 cov_168_14: frac[14,14] + frac[13,14] + frac[11,14] + frac[7,14]
   + frac[9,14] + frac[12,14] + frac[3,14] >= 1
 cov_169_51: frac[13,51] + frac[2,51] + frac[7,51] + frac[9,51]
   + frac[11,51] + frac[12,51] + frac[14,51] >= 0.9
 cov_170_56: frac[13,56] + frac[6,56] + frac[12,56] + frac[11,56]
   + frac[3,56] + frac[14,56] + frac[9,56] >= 0.8
 cov_171_29: frac[12,29] + frac[9,29] + frac[14,29] + frac[11,29]
   + frac[7,29] + frac[13,29] + frac[3,29] >= 1
 cov_172_34: frac[12,34] + frac[9,34] + frac[2,34] + frac[11,34]
   + frac[14,34] + frac[13,34] + frac[6,34] >= 0.8
 cov_173_53: frac[5,53] + frac[4,53] + frac[10,53] >= 0.7
 cov_174_22: frac[11,22] + frac[7,22] + frac[12,22] + frac[2,22]
   + frac[14,22] + frac[13,22] + frac[9,22] >= 0.9
 cov_175_20: frac[9,20] + frac[2,20] + frac[3,20] >= 0.8
 cov_176_35: frac[6,35] + frac[7,35] + frac[11,35] >= 1
 cov_177_13: frac[13,13] + frac[3,13] + frac[12,13] + frac[9,13]
   + frac[6,13] + frac[11,13] + frac[14,13] >= 0.7
 cov_178_38: frac[13,38] + frac[11,38] + frac[5,38] + frac[7,38]
   + frac[10,38] >= 0.9
 cov_179_60: frac[14,60] + frac[8,60] + frac[5,60] + frac[10,60]
   + frac[1,60] + frac[13,60] + frac[12,60] >= 0.7
 cov_180_28: frac[11,28] + frac[7,28] + frac[13,28] + frac[5,28]
   + frac[10,28] >= 1
 cov_181_53: frac[2,53] + frac[12,53] + frac[8,53] + frac[1,53]
   + frac[9,53] >= 0.8
 cov_182_44: frac[12,44] + frac[9,44] + frac[0,44] + frac[8,44]
   + frac[2,44] >= 1
 cov_183_35: frac[3,35] + frac[4,35] + frac[13,35] + frac[14,35]
   + frac[9,35] + frac[10,35] + frac[12,35] >= 0.7
 cov_184_31: frac[14,31] + frac[12,31] + frac[13,31] + frac[2,31]
   + frac[11,31] + frac[7,31] + frac[9,31] >= 0.8
 cov_185_24: frac[5,24] + frac[14,24] + frac[10,24] + frac[9,24]
   + frac[2,24] + frac[13,24] + frac[12,24] >= 0.9
 cov_186_58: frac[8,58] + frac[12,58] + frac[0,58] + frac[9,58]
   + frac[3,58] >= 1
 cov_187_62: frac[9,62] + frac[5,62] + frac[12,62] + frac[10,62]
   + frac[2,62] + frac[13,62] + frac[14,62] >= 1
 cov_188_8: frac[14,8] + frac[12,8] + frac[11,8] + frac[1,8] + frac[8,8]
   + frac[6,8] + frac[13,8] >= 0.9
 cov_189_12: frac[4,12] + frac[10,12] + frac[5,12] >= 0.9
 cov_190_7: frac[12,7] + frac[14,7] + frac[13,7] + frac[9,7] + frac[11,7]
   + frac[7,7] + frac[3,7] >= 1
 cov_191_11: frac[4,11] + frac[5,11] + frac[10,11] >= 0.9
 cov_192_55: frac[3,55] + frac[12,55] + frac[6,55] + frac[13,55]
   + frac[9,55] + frac[11,55] + frac[14,55] >= 0.8
 cov_193_43: frac[14,43] + frac[11,43] + frac[12,43] + frac[13,43]
   + frac[0,43] + frac[6,43] + frac[8,43] >= 0.7
 cov_194_34: frac[10,34] + frac[12,34] + frac[9,34] + frac[4,34]
   + frac[3,34] + frac[14,34] + frac[13,34] >= 0.7
 cov_195_28: frac[0,28] + frac[11,28] + frac[7,28] + frac[13,28]
   + frac[12,28] + frac[14,28] + frac[8,28] >= 0.8
 cov_196_1: frac[14,1] + frac[5,1] + frac[12,1] + frac[9,1] + frac[10,1]
   + frac[2,1] + frac[13,1] >= 0.7
 cov_197_48: frac[11,48] + frac[14,48] + frac[8,48] + frac[13,48]
   + frac[6,48] + frac[1,48] + frac[12,48] >= 0.9
 cov_198_27: frac[7,27] + frac[8,27] + frac[11,27] + frac[12,27]
   + frac[14,27] + frac[1,27] + frac[13,27] >= 0.9
 cov_199_52: frac[0,52] + frac[2,52] + frac[9,52] + frac[12,52]
   + frac[8,52] >= 0.9
 cov_200_3: frac[11,3] + frac[12,3] + frac[13,3] + frac[8,3] + frac[14,3]
   + frac[0,3] + frac[6,3] >= 0.8
 cov_201_33: frac[2,33] + frac[13,33] + frac[14,33] + frac[6,33]
   + frac[11,33] + frac[12,33] + frac[9,33] >= 0.9
 cov_202_40: frac[7,40] + frac[13,40] + frac[11,40] + frac[4,40]
   + frac[10,40] >= 0.8
 cov_203_21: frac[1,21] + frac[13,21] + frac[5,21] + frac[14,21]
   + frac[12,21] + frac[8,21] + frac[10,21] >= 0.9
 cov_204_1: frac[14,1] + frac[5,1] + frac[12,1] + frac[9,1] + frac[10,1]
   + frac[2,1] + frac[13,1] >= 1
 cov_205_57: frac[5,57] + frac[9,57] + frac[10,57] + frac[14,57]
   + frac[13,57] + frac[2,57] + frac[12,57] >= 1
 cov_206_13: frac[13,13] + frac[12,13] + frac[9,13] + frac[4,13]
   + frac[2,13] + frac[10,13] + frac[14,13] >= 0.9
 cov_207_21: frac[6,21] + frac[11,21] + frac[7,21] >= 0.7
 cov_208_0: frac[1,0] + frac[8,0] + frac[0,0] >= 0.9
 cov_209_40: frac[7,40] + frac[13,40] + frac[11,40] + frac[4,40]
   + frac[10,40] >= 1
 cov_210_2: frac[14,2] + frac[13,2] + frac[10,2] + frac[12,2] + frac[2,2]
   + frac[9,2] + frac[4,2] >= 1
 cov_211_8: frac[2,8] + frac[3,8] + frac[9,8] >= 0.9
 cov_212_61: frac[11,61] + frac[7,61] + frac[0,61] + frac[13,61]
   + frac[12,61] + frac[14,61] + frac[8,61] >= 0.9
 cov_213_31: frac[14,31] + frac[12,31] + frac[1,31] + frac[8,31]
   + frac[13,31] + frac[11,31] + frac[7,31] >= 0.9
 cov_214_31: frac[12,31] + frac[1,31] + frac[8,31] + frac[2,31]
   + frac[9,31] >= 0.7
 cov_215_61: frac[6,61] + frac[11,61] + frac[13,61] + frac[3,61]
   + frac[12,61] + frac[9,61] + frac[14,61] >= 0.8
 cov_216_47: frac[10,47] + frac[6,47] + frac[13,47] + frac[11,47]
   + frac[5,47] >= 0.9
 cov_217_54: frac[6,54] + frac[2,54] + frac[13,54] + frac[14,54]
   + frac[9,54] + frac[11,54] + frac[12,54] >= 0.8
 cov_218_39: frac[5,39] + frac[10,39] + frac[4,39] >= 0.9
 cov_219_25: frac[2,25] + frac[3,25] + frac[9,25] >= 0.9
 cov_220_41: frac[11,41] + frac[12,41] + frac[14,41] + frac[0,41]
   + frac[8,41] + frac[7,41] + frac[13,41] >= 0.9
 cov_221_40: frac[3,40] + frac[9,40] + frac[13,40] + frac[12,40]
   + frac[14,40] + frac[4,40] + frac[10,40] >= 0.8
 cov_222_49: frac[2,49] + frac[12,49] + frac[0,49] + frac[9,49]
   + frac[8,49] >= 0.8
 cov_223_32: frac[13,32] + frac[6,32] + frac[12,32] + frac[11,32]
   + frac[8,32] + frac[1,32] + frac[14,32] >= 1
 cov_224_21: frac[6,21] + frac[11,21] + frac[13,21] + frac[4,21]
   + frac[10,21] >= 0.7
 cov_225_36: frac[10,36] + frac[6,36] + frac[11,36] + frac[13,36]
   + frac[5,36] >= 1
 cov_226_6: frac[9,6] + frac[1,6] + frac[12,6] + frac[2,6] + frac[8,6]
   >= 1
 cov_227_15: frac[11,15] + frac[9,15] + frac[3,15] + frac[12,15]
   + frac[6,15] + frac[14,15] + frac[13,15] >= 0.9
 cov_228_34: frac[5,34] + frac[10,34] + frac[12,34] + frac[9,34]
   + frac[3,34] + frac[14,34] + frac[13,34] >= 0.9
 cov_229_28: frac[13,28] + frac[12,28] + frac[3,28] + frac[14,28]
   + frac[9,28] + frac[5,28] + frac[10,28] >= 0.8
 cov_230_63: frac[8,63] + frac[13,63] + frac[11,63] + frac[14,63]
   + frac[12,63] + frac[1,63] + frac[6,63] >= 0.9
 cov_231_29: frac[12,29] + frac[5,29] + frac[14,29] + frac[8,29]
   + frac[1,29] + frac[10,29] + frac[13,29] >= 0.7
 cov_232_59: frac[10,59] + frac[13,59] + frac[7,59] + frac[5,59]
   + frac[11,59] >= 0.8
 cov_233_63: frac[4,63] + frac[5,63] + frac[10,63] >= 0.8
 cov_234_23: frac[13,23] + frac[11,23] + frac[7,23] + frac[10,23]
   + frac[4,23] >= 0.9
 cov_235_52: frac[0,52] + frac[11,52] + frac[7,52] + frac[13,52]
   + frac[14,52] + frac[12,52] + frac[8,52] >= 0.8
 cov_236_37: frac[13,37] + frac[5,37] + frac[10,37] + frac[6,37]
   + frac[11,37] >= 1
 cov_237_41: frac[12,41] + frac[14,41] + frac[5,41] + frac[10,41]
   + frac[0,41] + frac[8,41] + frac[13,41] >= 0.8
 cov_238_53: frac[14,53] + frac[12,53] + frac[4,53] + frac[10,53]
   + frac[13,53] + frac[3,53] + frac[9,53] >= 0.8
 cov_239_21: frac[0,21] + frac[13,21] + frac[5,21] + frac[14,21]
   + frac[12,21] + frac[8,21] + frac[10,21] >= 1
 cov_240_32: frac[13,32] + frac[6,32] + frac[5,32] + frac[11,32]
   + frac[10,32] >= 0.7
 cov_241_6: frac[13,6] + frac[9,6] + frac[3,6] + frac[10,6] + frac[12,6]
   + frac[14,6] + frac[4,6] >= 0.8
 cov_242_22: frac[3,22] + frac[10,22] + frac[12,22] + frac[4,22]
   + frac[14,22] + frac[13,22] + frac[9,22] >= 0.9
 cov_243_14: frac[8,14] + frac[14,14] + frac[13,14] + frac[11,14]
   + frac[1,14] + frac[7,14] + frac[12,14] >= 0.9
 cov_244_50: frac[12,50] + frac[9,50] + frac[8,50] + frac[2,50]
   + frac[1,50] >= 1
 cov_245_16: frac[6,16] + frac[1,16] + frac[13,16] + frac[12,16]
   + frac[11,16] + frac[14,16] + frac[8,16] >= 0.7
 cov_246_46: frac[12,46] + frac[10,46] + frac[0,46] + frac[8,46]
   + frac[14,46] + frac[4,46] + frac[13,46] >= 0.7
 cov_247_45: frac[6,45] + frac[11,45] + frac[7,45] >= 0.9
 cov_248_12: frac[2,12] + frac[3,12] + frac[9,12] >= 0.7
 cov_249_49: frac[7,49] + frac[13,49] + frac[11,49] + frac[12,49]
   + frac[3,49] + frac[9,49] + frac[14,49] >= 0.9
 cov_250_58: frac[13,58] + frac[6,58] + frac[8,58] + frac[12,58]
   + frac[0,58] + frac[11,58] + frac[14,58] >= 0.8
 cov_251_51: frac[13,51] + frac[2,51] + frac[5,51] + frac[9,51]
   + frac[10,51] + frac[12,51] + frac[14,51] >= 1
 cov_252_27: frac[8,27] + frac[0,27] + frac[1,27] >= 0.7
 cov_253_26: frac[11,26] + frac[14,26] + frac[6,26] + frac[13,26]
   + frac[8,26] + frac[0,26] + frac[12,26] >= 0.9
 cov_254_24: frac[14,24] + frac[8,24] + frac[7,24] + frac[1,24]
   + frac[13,24] + frac[12,24] + frac[11,24] >= 0.9
 cov_255_57: frac[14,57] + frac[8,57] + frac[7,57] + frac[13,57]
   + frac[11,57] + frac[12,57] + frac[0,57] >= 1
 cov_256_10: frac[5,10] + frac[11,10] + frac[10,10] + frac[13,10]
   + frac[7,10] >= 0.7
 cov_257_56: frac[13,56] + frac[6,56] + frac[1,56] + frac[12,56]
   + frac[11,56] + frac[14,56] + frac[8,56] >= 0.9
 cov_258_5: frac[8,5] + frac[2,5] + frac[9,5] + frac[1,5] + frac[12,5]
   >= 1
 cov_259_51: frac[13,51] + frac[9,51] + frac[10,51] + frac[12,51]
   + frac[14,51] + frac[3,51] + frac[4,51] >= 0.8
 cov_260_62: frac[5,62] + frac[10,62] + frac[4,62] >= 1
 cov_261_1: frac[14,1] + frac[12,1] + frac[4,1] + frac[9,1] + frac[10,1]
   + frac[2,1] + frac[13,1] >= 1
 cov_262_54: frac[6,54] + frac[13,54] + frac[14,54] + frac[9,54]
   + frac[11,54] + frac[3,54] + frac[12,54] >= 0.9
 cov_263_35: frac[13,35] + frac[5,35] + frac[7,35] + frac[10,35]
   + frac[11,35] >= 0.7
 cov_264_8: frac[14,8] + frac[12,8] + frac[11,8] + frac[2,8] + frac[9,8]
   + frac[6,8] + frac[13,8] >= 0.7
 cov_265_2: frac[6,2] + frac[13,2] + frac[10,2] + frac[4,2] + frac[11,2]
   >= 0.9
 cov_266_29: frac[12,29] + frac[14,29] + frac[8,29] + frac[1,29]
   + frac[6,29] + frac[11,29] + frac[13,29] >= 0.7
 cov_267_47: frac[2,47] + frac[0,47] + frac[8,47] + frac[9,47]
   + frac[12,47] >= 0.9
 cov_268_3: frac[10,3] + frac[1,3] + frac[12,3] + frac[4,3] + frac[13,3]
   + frac[8,3] + frac[14,3] >= 0.9
 cov_269_41: frac[11,41] + frac[12,41] + frac[14,41] + frac[0,41]
   + frac[8,41] + frac[7,41] + frac[13,41] >= 0.9
 cov_270_9: frac[9,9] + frac[7,9] + frac[13,9] + frac[11,9] + frac[14,9]
   + frac[12,9] + frac[2,9] >= 0.7
 cov_271_60: frac[4,60] + frac[10,60] + frac[6,60] + frac[11,60]
   + frac[13,60] >= 0.9
 cov_272_6: frac[13,6] + frac[11,6] + frac[1,6] + frac[7,6] + frac[12,6]
   + frac[8,6] + frac[14,6] >= 0.8
 cov_273_40: frac[3,40] + frac[9,40] + frac[0,40] + frac[8,40]
   + frac[12,40] >= 0.8
 cov_274_22: frac[11,22] + frac[12,22] + frac[6,22] + frac[2,22]
   + frac[14,22] + frac[13,22] + frac[9,22] >= 1
 cov_275_21: frac[11,21] + frac[7,21] + frac[3,21] + frac[13,21]
   + frac[9,21] + frac[14,21] + frac[12,21] >= 0.8
 cov_276_37: frac[1,37] + frac[8,37] + frac[0,37] >= 0.7
 cov_277_62: frac[12,62] + frac[1,62] + frac[8,62] + frac[13,62]
   + frac[11,62] + frac[7,62] + frac[14,62] >= 0.8
 cov_278_13: frac[5,13] + frac[4,13] + frac[10,13] >= 1
 cov_279_48: frac[11,48] + frac[3,48] + frac[14,48] + frac[9,48]
   + frac[13,48] + frac[6,48] + frac[12,48] >= 0.8
 cov_280_20: frac[13,20] + frac[9,20] + frac[5,20] + frac[12,20]
   + frac[10,20] + frac[3,20] + frac[14,20] >= 0.7
 cov_281_61: frac[6,61] + frac[11,61] + frac[13,61] + frac[3,61]
   + frac[12,61] + frac[9,61] + frac[14,61] >= 0.7
 cov_282_37: frac[3,37] + frac[9,37] + frac[2,37] >= 1
 cov_283_62: frac[9,62] + frac[12,62] + frac[2,62] + frac[13,62]
   + frac[11,62] + frac[7,62] + frac[14,62] >= 0.8
 cov_284_55: frac[7,55] + frac[10,55] + frac[13,55] + frac[11,55]
   + frac[5,55] >= 0.7
 cov_285_58: frac[13,58] + frac[1,58] + frac[8,58] + frac[12,58]
   + frac[10,58] + frac[14,58] + frac[4,58] >= 0.8
 cov_286_0: frac[1,0] + frac[8,0] + frac[0,0] >= 0.8
 cov_287_55: frac[10,55] + frac[12,55] + frac[4,55] + frac[13,55]
   + frac[9,55] + frac[2,55] + frac[14,55] >= 1
 cov_288_29: frac[12,29] + frac[14,29] + frac[8,29] + frac[4,29]
   + frac[1,29] + frac[10,29] + frac[13,29] >= 0.8
 cov_289_44: frac[13,44] + frac[12,44] + frac[0,44] + frac[8,44]
   + frac[6,44] + frac[14,44] + frac[11,44] >= 1
 cov_290_43: frac[14,43] + frac[11,43] + frac[12,43] + frac[2,43]
   + frac[9,43] + frac[13,43] + frac[7,43] >= 0.7
 cov_291_24: frac[14,24] + frac[8,24] + frac[6,24] + frac[13,24]
   + frac[0,24] + frac[12,24] + frac[11,24] >= 1
 cov_292_31: frac[14,31] + frac[12,31] + frac[6,31] + frac[13,31]
   + frac[2,31] + frac[11,31] + frac[9,31] >= 1
 cov_293_7: frac[10,7] + frac[13,7] + frac[5,7] + frac[11,7] + frac[7,7]
   >= 0.7
 cov_294_58: frac[13,58] + frac[7,58] + frac[12,58] + frac[11,58]
   + frac[14,58] + frac[9,58] + frac[2,58] >= 0.7
 cov_295_33: frac[1,33] + frac[8,33] + frac[0,33] >= 1
 cov_296_40: frac[0,40] + frac[8,40] + frac[13,40] + frac[12,40]
   + frac[14,40] + frac[4,40] + frac[10,40] >= 1
 cov_297_51: frac[8,51] + frac[1,51] + frac[2,51] + frac[9,51]
   + frac[12,51] >= 0.9
 cov_298_24: frac[14,24] + frac[8,24] + frac[7,24] + frac[1,24]
   + frac[13,24] + frac[12,24] + frac[11,24] >= 0.7
 cov_299_45: frac[11,45] + frac[7,45] + frac[14,45] + frac[13,45]
   + frac[3,45] + frac[12,45] + frac[9,45] >= 0.9
 cov_300_58: frac[13,58] + frac[1,58] + frac[8,58] + frac[12,58]
   + frac[10,58] + frac[14,58] + frac[4,58] >= 0.8
 cov_301_45: frac[2,45] + frac[10,45] + frac[14,45] + frac[13,45]
   + frac[12,45] + frac[9,45] + frac[5,45] >= 0.9
 cov_302_52: frac[13,52] + frac[3,52] + frac[14,52] + frac[9,52]
   + frac[12,52] + frac[4,52] + frac[10,52] >= 1
 cov_303_10: frac[12,10] + frac[9,10] + frac[3,10] + frac[0,10]
   + frac[8,10] >= 0.7
 cov_304_7: frac[10,7] + frac[12,7] + frac[14,7] + frac[13,7] + frac[9,7]
   + frac[5,7] + frac[3,7] >= 0.7
 cov_305_38: frac[4,38] + frac[5,38] + frac[10,38] >= 0.8
 cov_306_43: frac[12,43] + frac[1,43] + frac[2,43] + frac[9,43]
   + frac[8,43] >= 0.7
 cov_307_46: frac[9,46] + frac[5,46] + frac[12,46] + frac[3,46]
   + frac[10,46] + frac[14,46] + frac[13,46] >= 0.9
 cov_308_13: frac[13,13] + frac[3,13] + frac[12,13] + frac[9,13]
   + frac[6,13] + frac[11,13] + frac[14,13] >= 0.9
 cov_309_49: frac[12,49] + frac[0,49] + frac[3,49] + frac[9,49]
   + frac[8,49] >= 0.9
 cov_310_61: frac[10,61] + frac[13,61] + frac[3,61] + frac[12,61]
   + frac[9,61] + frac[14,61] + frac[4,61] >= 0.9
 cov_311_46: frac[11,46] + frac[1,46] + frac[7,46] + frac[12,46]
   + frac[8,46] + frac[14,46] + frac[13,46] >= 0.7
 cov_312_58: frac[13,58] + frac[8,58] + frac[5,58] + frac[12,58]
   + frac[0,58] + frac[10,58] + frac[14,58] >= 0.7
 cov_313_1: frac[12,1] + frac[3,1] + frac[9,1] + frac[8,1] + frac[0,1]
   >= 0.9
 cov_314_53: frac[12,53] + frac[8,53] + frac[1,53] + frac[3,53]
   + frac[9,53] >= 1
 cov_315_40: frac[3,40] + frac[9,40] + frac[13,40] + frac[12,40]
   + frac[14,40] + frac[5,40] + frac[10,40] >= 0.9
 cov_316_17: frac[0,17] + frac[12,17] + frac[14,17] + frac[10,17]
   + frac[8,17] + frac[13,17] + frac[5,17] >= 0.7
 cov_317_21: frac[3,21] + frac[2,21] + frac[9,21] >= 1
 cov_318_51: frac[13,51] + frac[2,51] + frac[5,51] + frac[9,51]
   + frac[10,51] + frac[12,51] + frac[14,51] >= 0.8
 cov_319_33: frac[2,33] + frac[13,33] + frac[14,33] + frac[5,33]
   + frac[12,33] + frac[9,33] + frac[10,33] >= 1
 accuracy_7_3: - 5.09765625 frac[7,3] + mem[7,3] = 0
 accuracy_5_31: - 5.09765625 frac[5,31] + mem[5,31] = 0
 accuracy_3_35: - 5.09765625 frac[3,35] + mem[3,35] = 0
 accuracy_4_36: - 5.09765625 frac[4,36] + mem[4,36] = 0
 accuracy_8_63: - 5.09765625 frac[8,63] + mem[8,63] = 0
 accuracy_5_57: - 5.09765625 frac[5,57] + mem[5,57] = 0
 accuracy_9_62: - 5.09765625 frac[9,62] + mem[9,62] = 0
 accuracy_6_54: - 5.09765625 frac[6,54] + mem[6,54] = 0
 accuracy_10_61: - 5.09765625 frac[10,61] + mem[10,61] = 0
 accuracy_8_5: - 5.09765625 frac[8,5] + mem[8,5] = 0
 accuracy_7_55: - 5.09765625 frac[7,55] + mem[7,55] = 0
 accuracy_13_32: - 5.09765625 frac[13,32] + mem[13,32] = 0
 accuracy_11_48: - 5.09765625 frac[11,48] + mem[11,48] = 0
 accuracy_9_0: - 5.09765625 frac[9,0] + mem[9,0] = 0
 accuracy_14_31: - 5.09765625 frac[14,31] + mem[14,31] = 0
 accuracy_12_59: - 5.09765625 frac[12,59] + mem[12,59] = 0
 accuracy_10_7: - 5.09765625 frac[10,7] + mem[10,7] = 0
 accuracy_13_58: - 5.09765625 frac[13,58] + mem[13,58] = 0
 accuracy_11_22: - 5.09765625 frac[11,22] + mem[11,22] = 0
 accuracy_0_17: - 5.09765625 frac[0,17] + mem[0,17] = 0
 accuracy_14_1: - 5.09765625 frac[14,1] + mem[14,1] = 0
 accuracy_12_17: - 5.09765625 frac[12,17] + mem[12,17] = 0
 accuracy_1_28: - 5.09765625 frac[1,28] + mem[1,28] = 0
 accuracy_13_20: - 5.09765625 frac[13,20] + mem[13,20] = 0
 accuracy_0_55: - 5.09765625 frac[0,55] + mem[0,55] = 0
 accuracy_14_43: - 5.09765625 frac[14,43] + mem[14,43] = 0
 accuracy_1_54: - 5.09765625 frac[1,54] + mem[1,54] = 0
 accuracy_4_5: - 5.09765625 frac[4,5] + mem[4,5] = 0
 accuracy_2_53: - 5.09765625 frac[2,53] + mem[2,53] = 0
 accuracy_5_24: - 5.09765625 frac[5,24] + mem[5,24] = 0
 accuracy_3_40: - 5.09765625 frac[3,40] + mem[3,40] = 0
 accuracy_1_40: - 5.09765625 frac[1,40] + mem[1,40] = 0
 accuracy_4_35: - 5.09765625 frac[4,35] + mem[4,35] = 0
 accuracy_2_47: - 5.09765625 frac[2,47] + mem[2,47] = 0
 accuracy_8_38: - 5.09765625 frac[8,38] + mem[8,38] = 0
 accuracy_7_22: - 5.09765625 frac[7,22] + mem[7,22] = 0
 accuracy_5_34: - 5.09765625 frac[5,34] + mem[5,34] = 0
 accuracy_9_39: - 5.09765625 frac[9,39] + mem[9,39] = 0
 accuracy_10_36: - 5.09765625 frac[10,36] + mem[10,36] = 0
 accuracy_8_12: - 5.09765625 frac[8,12] + mem[8,12] = 0
 accuracy_11_41: - 5.09765625 frac[11,41] + mem[11,41] = 0
 accuracy_9_9: - 5.09765625 frac[9,9] + mem[9,9] = 0
 accuracy_12_50: - 5.09765625 frac[12,50] + mem[12,50] = 0
 accuracy_10_14: - 5.09765625 frac[10,14] + mem[10,14] = 0
 accuracy_8_18: - 5.09765625 frac[8,18] + mem[8,18] = 0
 accuracy_13_51: - 5.09765625 frac[13,51] + mem[13,51] = 0
 accuracy_11_15: - 5.09765625 frac[11,15] + mem[11,15] = 0
 accuracy_9_19: - 5.09765625 frac[9,19] + mem[9,19] = 0
 accuracy_14_8: - 5.09765625 frac[14,8] + mem[14,8] = 0
 accuracy_12_8: - 5.09765625 frac[12,8] + mem[12,8] = 0
 accuracy_1_21: - 5.09765625 frac[1,21] + mem[1,21] = 0
 accuracy_13_13: - 5.09765625 frac[13,13] + mem[13,13] = 0
 accuracy_2_18: - 5.09765625 frac[2,18] + mem[2,18] = 0
 accuracy_14_50: - 5.09765625 frac[14,50] + mem[14,50] = 0
 accuracy_4_12: - 5.09765625 frac[4,12] + mem[4,12] = 0
 accuracy_2_12: - 5.09765625 frac[2,12] + mem[2,12] = 0
 accuracy_0_36: - 5.09765625 frac[0,36] + mem[0,36] = 0
 accuracy_5_1: - 5.09765625 frac[5,1] + mem[5,1] = 0
 accuracy_1_33: - 5.09765625 frac[1,33] + mem[1,33] = 0
 accuracy_4_42: - 5.09765625 frac[4,42] + mem[4,42] = 0
 accuracy_8_45: - 5.09765625 frac[8,45] + mem[8,45] = 0
 accuracy_3_55: - 5.09765625 frac[3,55] + mem[3,55] = 0
 accuracy_9_40: - 5.09765625 frac[9,40] + mem[9,40] = 0
 accuracy_6_32: - 5.09765625 frac[6,32] + mem[6,32] = 0
 accuracy_10_47: - 5.09765625 frac[10,47] + mem[10,47] = 0
 accuracy_8_51: - 5.09765625 frac[8,51] + mem[8,51] = 0
 accuracy_5_53: - 5.09765625 frac[5,53] + mem[5,53] = 0
 accuracy_11_46: - 5.09765625 frac[11,46] + mem[11,46] = 0
 accuracy_9_50: - 5.09765625 frac[9,50] + mem[9,50] = 0
 accuracy_6_58: - 5.09765625 frac[6,58] + mem[6,58] = 0
 accuracy_12_41: - 5.09765625 frac[12,41] + mem[12,41] = 0
 accuracy_10_9: - 5.09765625 frac[10,9] + mem[10,9] = 0
 accuracy_8_25: - 5.09765625 frac[8,25] + mem[8,25] = 0
 accuracy_13_44: - 5.09765625 frac[13,44] + mem[13,44] = 0
 accuracy_11_4: - 5.09765625 frac[11,4] + mem[11,4] = 0
 accuracy_9_20: - 5.09765625 frac[9,20] + mem[9,20] = 0
 accuracy_14_19: - 5.09765625 frac[14,19] + mem[14,19] = 0
 accuracy_12_7: - 5.09765625 frac[12,7] + mem[12,7] = 0
 accuracy_10_19: - 5.09765625 frac[10,19] + mem[10,19] = 0
 accuracy_13_6: - 5.09765625 frac[13,6] + mem[13,6] = 0
 accuracy_11_26: - 5.09765625 frac[11,26] + mem[11,26] = 0
 accuracy_0_5: - 5.09765625 frac[0,5] + mem[0,5] = 0
 accuracy_14_53: - 5.09765625 frac[14,53] + mem[14,53] = 0
 accuracy_12_29: - 5.09765625 frac[12,29] + mem[12,29] = 0
 accuracy_1_0: - 5.09765625 frac[1,0] + mem[1,0] = 0
 accuracy_4_11: - 5.09765625 frac[4,11] + mem[4,11] = 0
 accuracy_5_10: - 5.09765625 frac[5,10] + mem[5,10] = 0
 accuracy_3_22: - 5.09765625 frac[3,22] + mem[3,22] = 0
 accuracy_1_58: - 5.09765625 frac[1,58] + mem[1,58] = 0
 accuracy_6_1: - 5.09765625 frac[6,1] + mem[6,1] = 0
 accuracy_2_33: - 5.09765625 frac[2,33] + mem[2,33] = 0
 accuracy_7_4: - 5.09765625 frac[7,4] + mem[7,4] = 0
 accuracy_5_20: - 5.09765625 frac[5,20] + mem[5,20] = 0
 accuracy_4_63: - 5.09765625 frac[4,63] + mem[4,63] = 0
 accuracy_8_58: - 5.09765625 frac[8,58] + mem[8,58] = 0
 accuracy_5_62: - 5.09765625 frac[5,62] + mem[5,62] = 0
 accuracy_11_39: - 5.09765625 frac[11,39] + mem[11,39] = 0
 accuracy_9_59: - 5.09765625 frac[9,59] + mem[9,59] = 0
 accuracy_6_61: - 5.09765625 frac[6,61] + mem[6,61] = 0
 accuracy_12_32: - 5.09765625 frac[12,32] + mem[12,32] = 0
 accuracy_10_48: - 5.09765625 frac[10,48] + mem[10,48] = 0
 accuracy_8_0: - 5.09765625 frac[8,0] + mem[8,0] = 0
 accuracy_13_37: - 5.09765625 frac[13,37] + mem[13,37] = 0
 accuracy_11_61: - 5.09765625 frac[11,61] + mem[11,61] = 0
 accuracy_9_29: - 5.09765625 frac[9,29] + mem[9,29] = 0
 accuracy_14_26: - 5.09765625 frac[14,26] + mem[14,26] = 0
 accuracy_12_62: - 5.09765625 frac[12,62] + mem[12,62] = 0
 accuracy_13_63: - 5.09765625 frac[13,63] + mem[13,63] = 0
 accuracy_11_19: - 5.09765625 frac[11,19] + mem[11,19] = 0
 accuracy_0_28: - 5.09765625 frac[0,28] + mem[0,28] = 0
 accuracy_14_60: - 5.09765625 frac[14,60] + mem[14,60] = 0
 accuracy_12_20: - 5.09765625 frac[12,20] + mem[12,20] = 0
 accuracy_1_25: - 5.09765625 frac[1,25] + mem[1,25] = 0
 accuracy_0_50: - 5.09765625 frac[0,50] + mem[0,50] = 0
 accuracy_14_38: - 5.09765625 frac[14,38] + mem[14,38] = 0
 accuracy_3_31: - 5.09765625 frac[3,31] + mem[3,31] = 0
 accuracy_1_51: - 5.09765625 frac[1,51] + mem[1,51] = 0
 accuracy_0_40: - 5.09765625 frac[0,40] + mem[0,40] = 0
 accuracy_5_29: - 5.09765625 frac[5,29] + mem[5,29] = 0
 accuracy_3_37: - 5.09765625 frac[3,37] + mem[3,37] = 0
 accuracy_4_38: - 5.09765625 frac[4,38] + mem[4,38] = 0
 accuracy_8_33: - 5.09765625 frac[8,33] + mem[8,33] = 0
 accuracy_7_27: - 5.09765625 frac[7,27] + mem[7,27] = 0
 accuracy_5_39: - 5.09765625 frac[5,39] + mem[5,39] = 0
 accuracy_9_60: - 5.09765625 frac[9,60] + mem[9,60] = 0
 accuracy_10_59: - 5.09765625 frac[10,59] + mem[10,59] = 0
 accuracy_8_7: - 5.09765625 frac[8,7] + mem[8,7] = 0
 accuracy_7_49: - 5.09765625 frac[7,49] + mem[7,49] = 0
 accuracy_11_50: - 5.09765625 frac[11,50] + mem[11,50] = 0
 accuracy_9_6: - 5.09765625 frac[9,6] + mem[9,6] = 0
 accuracy_14_29: - 5.09765625 frac[14,29] + mem[14,29] = 0
 accuracy_12_53: - 5.09765625 frac[12,53] + mem[12,53] = 0
 accuracy_10_5: - 5.09765625 frac[10,5] + mem[10,5] = 0
 accuracy_13_56: - 5.09765625 frac[13,56] + mem[13,56] = 0
 accuracy_11_8: - 5.09765625 frac[11,8] + mem[11,8] = 0
 accuracy_14_7: - 5.09765625 frac[14,7] + mem[14,7] = 0
 accuracy_12_19: - 5.09765625 frac[12,19] + mem[12,19] = 0
 accuracy_1_18: - 5.09765625 frac[1,18] + mem[1,18] = 0
 accuracy_13_18: - 5.09765625 frac[13,18] + mem[13,18] = 0
 accuracy_2_25: - 5.09765625 frac[2,25] + mem[2,25] = 0
 accuracy_14_41: - 5.09765625 frac[14,41] + mem[14,41] = 0
 accuracy_1_52: - 5.09765625 frac[1,52] + mem[1,52] = 0
 accuracy_2_51: - 5.09765625 frac[2,51] + mem[2,51] = 0
 accuracy_0_47: - 5.09765625 frac[0,47] + mem[0,47] = 0
 accuracy_3_42: - 5.09765625 frac[3,42] + mem[3,42] = 0
 accuracy_1_46: - 5.09765625 frac[1,46] + mem[1,46] = 0
 accuracy_6_21: - 5.09765625 frac[6,21] + mem[6,21] = 0
 accuracy_4_45: - 5.09765625 frac[4,45] + mem[4,45] = 0
 accuracy_2_45: - 5.09765625 frac[2,45] + mem[2,45] = 0
 accuracy_8_40: - 5.09765625 frac[8,40] + mem[8,40] = 0
 accuracy_7_16: - 5.09765625 frac[7,16] + mem[7,16] = 0
 accuracy_5_32: - 5.09765625 frac[5,32] + mem[5,32] = 0
 accuracy_3_48: - 5.09765625 frac[3,48] + mem[3,48] = 0
 accuracy_9_37: - 5.09765625 frac[9,37] + mem[9,37] = 0
 accuracy_6_47: - 5.09765625 frac[6,47] + mem[6,47] = 0
 accuracy_10_34: - 5.09765625 frac[10,34] + mem[10,34] = 0
 accuracy_8_14: - 5.09765625 frac[8,14] + mem[8,14] = 0
 accuracy_7_46: - 5.09765625 frac[7,46] + mem[7,46] = 0
 accuracy_11_43: - 5.09765625 frac[11,43] + mem[11,43] = 0
 accuracy_9_15: - 5.09765625 frac[9,15] + mem[9,15] = 0
 accuracy_12_44: - 5.09765625 frac[12,44] + mem[12,44] = 0
 accuracy_10_12: - 5.09765625 frac[10,12] + mem[10,12] = 0
 accuracy_13_49: - 5.09765625 frac[13,49] + mem[13,49] = 0
 accuracy_11_1: - 5.09765625 frac[11,1] + mem[11,1] = 0
 accuracy_9_17: - 5.09765625 frac[9,17] + mem[9,17] = 0
 accuracy_14_14: - 5.09765625 frac[14,14] + mem[14,14] = 0
 accuracy_12_10: - 5.09765625 frac[12,10] + mem[12,10] = 0
 accuracy_10_22: - 5.09765625 frac[10,22] + mem[10,22] = 0
 accuracy_2_16: - 5.09765625 frac[2,16] + mem[2,16] = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 accuracy_14_48: - 5.09765625 frac[14,48] + mem[14,48] = 0
 accuracy_3_13: - 5.09765625 frac[3,13] + mem[3,13] = 0
 accuracy_4_14: - 5.09765625 frac[4,14] + mem[4,14] = 0
 accuracy_0_38: - 5.09765625 frac[0,38] + mem[0,38] = 0
 accuracy_3_19: - 5.09765625 frac[3,19] + mem[3,19] = 0
 accuracy_1_39: - 5.09765625 frac[1,39] + mem[1,39] = 0
 accuracy_6_12: - 5.09765625 frac[6,12] + mem[6,12] = 0
 accuracy_2_36: - 5.09765625 frac[2,36] + mem[2,36] = 0
 accuracy_8_47: - 5.09765625 frac[8,47] + mem[8,47] = 0
 accuracy_7_9: - 5.09765625 frac[7,9] + mem[7,9] = 0
 accuracy_5_41: - 5.09765625 frac[5,41] + mem[5,41] = 0
 accuracy_3_57: - 5.09765625 frac[3,57] + mem[3,57] = 0
 accuracy_9_46: - 5.09765625 frac[9,46] + mem[9,46] = 0
 accuracy_4_50: - 5.09765625 frac[4,50] + mem[4,50] = 0
 accuracy_10_45: - 5.09765625 frac[10,45] + mem[10,45] = 0
 accuracy_8_53: - 5.09765625 frac[8,53] + mem[8,53] = 0
 accuracy_7_39: - 5.09765625 frac[7,39] + mem[7,39] = 0
 accuracy_5_51: - 5.09765625 frac[5,51] + mem[5,51] = 0
 accuracy_11_32: - 5.09765625 frac[11,32] + mem[11,32] = 0
 accuracy_9_48: - 5.09765625 frac[9,48] + mem[9,48] = 0
 accuracy_6_56: - 5.09765625 frac[6,56] + mem[6,56] = 0
 accuracy_12_43: - 5.09765625 frac[12,43] + mem[12,43] = 0
 accuracy_10_55: - 5.09765625 frac[10,55] + mem[10,55] = 0
 accuracy_8_27: - 5.09765625 frac[8,27] + mem[8,27] = 0
 accuracy_7_61: - 5.09765625 frac[7,61] + mem[7,61] = 0
 accuracy_13_42: - 5.09765625 frac[13,42] + mem[13,42] = 0
 accuracy_11_6: - 5.09765625 frac[11,6] + mem[11,6] = 0
 accuracy_9_26: - 5.09765625 frac[9,26] + mem[9,26] = 0
 accuracy_14_17: - 5.09765625 frac[14,17] + mem[14,17] = 0
 accuracy_12_1: - 5.09765625 frac[12,1] + mem[12,1] = 0
 accuracy_10_17: - 5.09765625 frac[10,17] + mem[10,17] = 0
 accuracy_13_4: - 5.09765625 frac[13,4] + mem[13,4] = 0
 accuracy_11_28: - 5.09765625 frac[11,28] + mem[11,28] = 0
 accuracy_0_7: - 5.09765625 frac[0,7] + mem[0,7] = 0
 accuracy_14_59: - 5.09765625 frac[14,59] + mem[14,59] = 0
 accuracy_12_31: - 5.09765625 frac[12,31] + mem[12,31] = 0
 accuracy_1_6: - 5.09765625 frac[1,6] + mem[1,6] = 0
 accuracy_13_30: - 5.09765625 frac[13,30] + mem[13,30] = 0
 accuracy_2_5: - 5.09765625 frac[2,5] + mem[2,5] = 0
 accuracy_0_61: - 5.09765625 frac[0,61] + mem[0,61] = 0
 accuracy_3_24: - 5.09765625 frac[3,24] + mem[3,24] = 0
 accuracy_1_56: - 5.09765625 frac[1,56] + mem[1,56] = 0
 accuracy_6_7: - 5.09765625 frac[6,7] + mem[6,7] = 0
 accuracy_4_19: - 5.09765625 frac[4,19] + mem[4,19] = 0
 accuracy_7_6: - 5.09765625 frac[7,6] + mem[7,6] = 0
 accuracy_5_18: - 5.09765625 frac[5,18] + mem[5,18] = 0
 accuracy_3_62: - 5.09765625 frac[3,62] + mem[3,62] = 0
 accuracy_8_60: - 5.09765625 frac[8,60] + mem[8,60] = 0
 accuracy_7_28: - 5.09765625 frac[7,28] + mem[7,28] = 0
 accuracy_5_60: - 5.09765625 frac[5,60] + mem[5,60] = 0
 accuracy_9_57: - 5.09765625 frac[9,57] + mem[9,57] = 0
 accuracy_12_34: - 5.09765625 frac[12,34] + mem[12,34] = 0
 accuracy_10_62: - 5.09765625 frac[10,62] + mem[10,62] = 0
 accuracy_8_2: - 5.09765625 frac[8,2] + mem[8,2] = 0
 accuracy_7_58: - 5.09765625 frac[7,58] + mem[7,58] = 0
 accuracy_13_35: - 5.09765625 frac[13,35] + mem[13,35] = 0
 accuracy_11_63: - 5.09765625 frac[11,63] + mem[11,63] = 0
 accuracy_14_24: - 5.09765625 frac[14,24] + mem[14,24] = 0
 accuracy_12_56: - 5.09765625 frac[12,56] + mem[12,56] = 0
 accuracy_10_24: - 5.09765625 frac[10,24] + mem[10,24] = 0
 accuracy_13_61: - 5.09765625 frac[13,61] + mem[13,61] = 0
 accuracy_11_21: - 5.09765625 frac[11,21] + mem[11,21] = 0
 accuracy_0_30: - 5.09765625 frac[0,30] + mem[0,30] = 0
 accuracy_14_2: - 5.09765625 frac[14,2] + mem[14,2] = 0
 accuracy_12_22: - 5.09765625 frac[12,22] + mem[12,22] = 0
 accuracy_1_31: - 5.09765625 frac[1,31] + mem[1,31] = 0
 accuracy_13_23: - 5.09765625 frac[13,23] + mem[13,23] = 0
 accuracy_0_52: - 5.09765625 frac[0,52] + mem[0,52] = 0
 accuracy_14_36: - 5.09765625 frac[14,36] + mem[14,36] = 0
 accuracy_3_1: - 5.09765625 frac[3,1] + mem[3,1] = 0
 accuracy_2_54: - 5.09765625 frac[2,54] + mem[2,54] = 0
 accuracy_5_27: - 5.09765625 frac[5,27] + mem[5,27] = 0
 accuracy_3_39: - 5.09765625 frac[3,39] + mem[3,39] = 0
 accuracy_1_43: - 5.09765625 frac[1,43] + mem[1,43] = 0
 accuracy_6_16: - 5.09765625 frac[6,16] + mem[6,16] = 0
 accuracy_7_21: - 5.09765625 frac[7,21] + mem[7,21] = 0
 accuracy_5_37: - 5.09765625 frac[5,37] + mem[5,37] = 0
 accuracy_9_34: - 5.09765625 frac[9,34] + mem[9,34] = 0
 accuracy_6_42: - 5.09765625 frac[6,42] + mem[6,42] = 0
 accuracy_10_57: - 5.09765625 frac[10,57] + mem[10,57] = 0
 accuracy_8_9: - 5.09765625 frac[8,9] + mem[8,9] = 0
 accuracy_7_51: - 5.09765625 frac[7,51] + mem[7,51] = 0
 accuracy_11_52: - 5.09765625 frac[11,52] + mem[11,52] = 0
 accuracy_9_4: - 5.09765625 frac[9,4] + mem[9,4] = 0
 accuracy_12_55: - 5.09765625 frac[12,55] + mem[12,55] = 0
 accuracy_10_3: - 5.09765625 frac[10,3] + mem[10,3] = 0
 accuracy_13_54: - 5.09765625 frac[13,54] + mem[13,54] = 0
 accuracy_11_10: - 5.09765625 frac[11,10] + mem[11,10] = 0
 accuracy_0_21: - 5.09765625 frac[0,21] + mem[0,21] = 0
 accuracy_14_5: - 5.09765625 frac[14,5] + mem[14,5] = 0
 accuracy_12_13: - 5.09765625 frac[12,13] + mem[12,13] = 0
 accuracy_1_16: - 5.09765625 frac[1,16] + mem[1,16] = 0
 accuracy_13_16: - 5.09765625 frac[13,16] + mem[13,16] = 0
 accuracy_14_47: - 5.09765625 frac[14,47] + mem[14,47] = 0
 accuracy_3_6: - 5.09765625 frac[3,6] + mem[3,6] = 0
 accuracy_4_1: - 5.09765625 frac[4,1] + mem[4,1] = 0
 accuracy_2_49: - 5.09765625 frac[2,49] + mem[2,49] = 0
 accuracy_0_33: - 5.09765625 frac[0,33] + mem[0,33] = 0
 accuracy_3_44: - 5.09765625 frac[3,44] + mem[3,44] = 0
 accuracy_1_44: - 5.09765625 frac[1,44] + mem[1,44] = 0
 accuracy_4_47: - 5.09765625 frac[4,47] + mem[4,47] = 0
 accuracy_2_43: - 5.09765625 frac[2,43] + mem[2,43] = 0
 accuracy_8_42: - 5.09765625 frac[8,42] + mem[8,42] = 0
 accuracy_7_18: - 5.09765625 frac[7,18] + mem[7,18] = 0
 accuracy_5_46: - 5.09765625 frac[5,46] + mem[5,46] = 0
 accuracy_9_43: - 5.09765625 frac[9,43] + mem[9,43] = 0
 accuracy_6_45: - 5.09765625 frac[6,45] + mem[6,45] = 0
 accuracy_4_53: - 5.09765625 frac[4,53] + mem[4,53] = 0
 accuracy_10_32: - 5.09765625 frac[10,32] + mem[10,32] = 0
 accuracy_8_48: - 5.09765625 frac[8,48] + mem[8,48] = 0
 accuracy_7_40: - 5.09765625 frac[7,40] + mem[7,40] = 0
 accuracy_11_45: - 5.09765625 frac[11,45] + mem[11,45] = 0
 accuracy_9_13: - 5.09765625 frac[9,13] + mem[9,13] = 0
 accuracy_12_46: - 5.09765625 frac[12,46] + mem[12,46] = 0
 accuracy_10_10: - 5.09765625 frac[10,10] + mem[10,10] = 0
 accuracy_13_47: - 5.09765625 frac[13,47] + mem[13,47] = 0
 accuracy_11_3: - 5.09765625 frac[11,3] + mem[11,3] = 0
 accuracy_9_23: - 5.09765625 frac[9,23] + mem[9,23] = 0
 accuracy_14_12: - 5.09765625 frac[14,12] + mem[14,12] = 0
 accuracy_12_4: - 5.09765625 frac[12,4] + mem[12,4] = 0
 accuracy_10_20: - 5.09765625 frac[10,20] + mem[10,20] = 0
 accuracy_13_9: - 5.09765625 frac[13,9] + mem[13,9] = 0
 accuracy_14_54: - 5.09765625 frac[14,54] + mem[14,54] = 0
 accuracy_3_15: - 5.09765625 frac[3,15] + mem[3,15] = 0
 accuracy_1_3: - 5.09765625 frac[1,3] + mem[1,3] = 0
 accuracy_2_8: - 5.09765625 frac[2,8] + mem[2,8] = 0
 accuracy_5_13: - 5.09765625 frac[5,13] + mem[5,13] = 0
 accuracy_3_21: - 5.09765625 frac[3,21] + mem[3,21] = 0
 accuracy_1_37: - 5.09765625 frac[1,37] + mem[1,37] = 0
 accuracy_6_2: - 5.09765625 frac[6,2] + mem[6,2] = 0
 accuracy_4_22: - 5.09765625 frac[4,22] + mem[4,22] = 0
 accuracy_2_34: - 5.09765625 frac[2,34] + mem[2,34] = 0
 accuracy_5_23: - 5.09765625 frac[5,23] + mem[5,23] = 0
 accuracy_3_59: - 5.09765625 frac[3,59] + mem[3,59] = 0
 accuracy_9_44: - 5.09765625 frac[9,44] + mem[9,44] = 0
 accuracy_6_36: - 5.09765625 frac[6,36] + mem[6,36] = 0
 accuracy_4_60: - 5.09765625 frac[4,60] + mem[4,60] = 0
 accuracy_10_43: - 5.09765625 frac[10,43] + mem[10,43] = 0
 accuracy_8_55: - 5.09765625 frac[8,55] + mem[8,55] = 0
 accuracy_11_34: - 5.09765625 frac[11,34] + mem[11,34] = 0
 accuracy_9_54: - 5.09765625 frac[9,54] + mem[9,54] = 0
 accuracy_12_37: - 5.09765625 frac[12,37] + mem[12,37] = 0
 accuracy_10_53: - 5.09765625 frac[10,53] + mem[10,53] = 0
 accuracy_8_29: - 5.09765625 frac[8,29] + mem[8,29] = 0
 accuracy_7_63: - 5.09765625 frac[7,63] + mem[7,63] = 0
 accuracy_13_40: - 5.09765625 frac[13,40] + mem[13,40] = 0
 accuracy_11_56: - 5.09765625 frac[11,56] + mem[11,56] = 0
 accuracy_9_24: - 5.09765625 frac[9,24] + mem[9,24] = 0
 accuracy_14_23: - 5.09765625 frac[14,23] + mem[14,23] = 0
 accuracy_12_3: - 5.09765625 frac[12,3] + mem[12,3] = 0
 accuracy_10_31: - 5.09765625 frac[10,31] + mem[10,31] = 0
 accuracy_13_2: - 5.09765625 frac[13,2] + mem[13,2] = 0
 accuracy_11_30: - 5.09765625 frac[11,30] + mem[11,30] = 0
 accuracy_0_25: - 5.09765625 frac[0,25] + mem[0,25] = 0
 accuracy_14_57: - 5.09765625 frac[14,57] + mem[14,57] = 0
 accuracy_12_25: - 5.09765625 frac[12,25] + mem[12,25] = 0
 accuracy_13_28: - 5.09765625 frac[13,28] + mem[13,28] = 0
 accuracy_0_63: - 5.09765625 frac[0,63] + mem[0,63] = 0
 accuracy_14_35: - 5.09765625 frac[14,35] + mem[14,35] = 0
 accuracy_3_26: - 5.09765625 frac[3,26] + mem[3,26] = 0
 accuracy_1_62: - 5.09765625 frac[1,62] + mem[1,62] = 0
 accuracy_6_5: - 5.09765625 frac[6,5] + mem[6,5] = 0
 accuracy_4_29: - 5.09765625 frac[4,29] + mem[4,29] = 0
 accuracy_7_0: - 5.09765625 frac[7,0] + mem[7,0] = 0
 accuracy_5_16: - 5.09765625 frac[5,16] + mem[5,16] = 0
 accuracy_6_31: - 5.09765625 frac[6,31] + mem[6,31] = 0
 accuracy_4_59: - 5.09765625 frac[4,59] + mem[4,59] = 0
 accuracy_8_62: - 5.09765625 frac[8,62] + mem[8,62] = 0
 accuracy_5_58: - 5.09765625 frac[5,58] + mem[5,58] = 0
 accuracy_10_60: - 5.09765625 frac[10,60] + mem[10,60] = 0
 accuracy_7_52: - 5.09765625 frac[7,52] + mem[7,52] = 0
 accuracy_13_33: - 5.09765625 frac[13,33] + mem[13,33] = 0
 accuracy_11_49: - 5.09765625 frac[11,49] + mem[11,49] = 0
 accuracy_9_1: - 5.09765625 frac[9,1] + mem[9,1] = 0
 accuracy_14_30: - 5.09765625 frac[14,30] + mem[14,30] = 0
 accuracy_12_58: - 5.09765625 frac[12,58] + mem[12,58] = 0
 accuracy_10_6: - 5.09765625 frac[10,6] + mem[10,6] = 0
 accuracy_13_59: - 5.09765625 frac[13,59] + mem[13,59] = 0
 accuracy_11_23: - 5.09765625 frac[11,23] + mem[11,23] = 0
 accuracy_0_16: - 5.09765625 frac[0,16] + mem[0,16] = 0
 accuracy_14_0: - 5.09765625 frac[14,0] + mem[14,0] = 0
 accuracy_12_16: - 5.09765625 frac[12,16] + mem[12,16] = 0
 accuracy_1_29: - 5.09765625 frac[1,29] + mem[1,29] = 0
 accuracy_13_21: - 5.09765625 frac[13,21] + mem[13,21] = 0
 accuracy_2_26: - 5.09765625 frac[2,26] + mem[2,26] = 0
 accuracy_0_54: - 5.09765625 frac[0,54] + mem[0,54] = 0
 accuracy_14_42: - 5.09765625 frac[14,42] + mem[14,42] = 0
 accuracy_2_52: - 5.09765625 frac[2,52] + mem[2,52] = 0
 accuracy_0_44: - 5.09765625 frac[0,44] + mem[0,44] = 0
 accuracy_6_22: - 5.09765625 frac[6,22] + mem[6,22] = 0
 accuracy_4_34: - 5.09765625 frac[4,34] + mem[4,34] = 0
 accuracy_2_46: - 5.09765625 frac[2,46] + mem[2,46] = 0
 accuracy_8_37: - 5.09765625 frac[8,37] + mem[8,37] = 0
 accuracy_7_23: - 5.09765625 frac[7,23] + mem[7,23] = 0
 accuracy_5_35: - 5.09765625 frac[5,35] + mem[5,35] = 0
 accuracy_6_40: - 5.09765625 frac[6,40] + mem[6,40] = 0
 accuracy_10_39: - 5.09765625 frac[10,39] + mem[10,39] = 0
 accuracy_7_45: - 5.09765625 frac[7,45] + mem[7,45] = 0
 accuracy_11_54: - 5.09765625 frac[11,54] + mem[11,54] = 0
 accuracy_9_10: - 5.09765625 frac[9,10] + mem[9,10] = 0
 accuracy_12_49: - 5.09765625 frac[12,49] + mem[12,49] = 0
 accuracy_10_1: - 5.09765625 frac[10,1] + mem[10,1] = 0
 accuracy_8_17: - 5.09765625 frac[8,17] + mem[8,17] = 0
 accuracy_13_52: - 5.09765625 frac[13,52] + mem[13,52] = 0
 accuracy_11_12: - 5.09765625 frac[11,12] + mem[11,12] = 0
 accuracy_12_15: - 5.09765625 frac[12,15] + mem[12,15] = 0
 accuracy_13_14: - 5.09765625 frac[13,14] + mem[13,14] = 0
 accuracy_2_21: - 5.09765625 frac[2,21] + mem[2,21] = 0
 accuracy_0_13: - 5.09765625 frac[0,13] + mem[0,13] = 0
 accuracy_14_45: - 5.09765625 frac[14,45] + mem[14,45] = 0
 accuracy_3_8: - 5.09765625 frac[3,8] + mem[3,8] = 0
 accuracy_1_8: - 5.09765625 frac[1,8] + mem[1,8] = 0
 accuracy_4_3: - 5.09765625 frac[4,3] + mem[4,3] = 0
 accuracy_3_46: - 5.09765625 frac[3,46] + mem[3,46] = 0
 accuracy_8_44: - 5.09765625 frac[8,44] + mem[8,44] = 0
 accuracy_7_12: - 5.09765625 frac[7,12] + mem[7,12] = 0
 accuracy_5_44: - 5.09765625 frac[5,44] + mem[5,44] = 0
 accuracy_3_52: - 5.09765625 frac[3,52] + mem[3,52] = 0
 accuracy_6_35: - 5.09765625 frac[6,35] + mem[6,35] = 0
 accuracy_4_55: - 5.09765625 frac[4,55] + mem[4,55] = 0
 accuracy_10_46: - 5.09765625 frac[10,46] + mem[10,46] = 0
 accuracy_8_50: - 5.09765625 frac[8,50] + mem[8,50] = 0
 accuracy_7_42: - 5.09765625 frac[7,42] + mem[7,42] = 0
 accuracy_5_54: - 5.09765625 frac[5,54] + mem[5,54] = 0
 accuracy_11_47: - 5.09765625 frac[11,47] + mem[11,47] = 0
 accuracy_9_51: - 5.09765625 frac[9,51] + mem[9,51] = 0
 accuracy_12_40: - 5.09765625 frac[12,40] + mem[12,40] = 0
 accuracy_8_24: - 5.09765625 frac[8,24] + mem[8,24] = 0
 accuracy_13_45: - 5.09765625 frac[13,45] + mem[13,45] = 0
 accuracy_11_5: - 5.09765625 frac[11,5] + mem[11,5] = 0
 accuracy_9_21: - 5.09765625 frac[9,21] + mem[9,21] = 0
 accuracy_14_18: - 5.09765625 frac[14,18] + mem[14,18] = 0
 accuracy_12_6: - 5.09765625 frac[12,6] + mem[12,6] = 0
 accuracy_10_18: - 5.09765625 frac[10,18] + mem[10,18] = 0
 accuracy_13_7: - 5.09765625 frac[13,7] + mem[13,7] = 0
 accuracy_11_27: - 5.09765625 frac[11,27] + mem[11,27] = 0
 accuracy_14_52: - 5.09765625 frac[14,52] + mem[14,52] = 0
 accuracy_12_28: - 5.09765625 frac[12,28] + mem[12,28] = 0
 accuracy_1_1: - 5.09765625 frac[1,1] + mem[1,1] = 0
 accuracy_2_6: - 5.09765625 frac[2,6] + mem[2,6] = 0
 accuracy_0_58: - 5.09765625 frac[0,58] + mem[0,58] = 0
 accuracy_5_11: - 5.09765625 frac[5,11] + mem[5,11] = 0
 accuracy_3_23: - 5.09765625 frac[3,23] + mem[3,23] = 0
 accuracy_6_0: - 5.09765625 frac[6,0] + mem[6,0] = 0
 accuracy_7_5: - 5.09765625 frac[7,5] + mem[7,5] = 0
 accuracy_5_21: - 5.09765625 frac[5,21] + mem[5,21] = 0
 accuracy_3_61: - 5.09765625 frac[3,61] + mem[3,61] = 0
 accuracy_6_26: - 5.09765625 frac[6,26] + mem[6,26] = 0
 accuracy_4_62: - 5.09765625 frac[4,62] + mem[4,62] = 0
 accuracy_10_41: - 5.09765625 frac[10,41] + mem[10,41] = 0
 accuracy_8_57: - 5.09765625 frac[8,57] + mem[8,57] = 0
 accuracy_7_35: - 5.09765625 frac[7,35] + mem[7,35] = 0
 accuracy_5_63: - 5.09765625 frac[5,63] + mem[5,63] = 0
 accuracy_11_36: - 5.09765625 frac[11,36] + mem[11,36] = 0
 accuracy_9_52: - 5.09765625 frac[9,52] + mem[9,52] = 0
 accuracy_6_60: - 5.09765625 frac[6,60] + mem[6,60] = 0
 accuracy_12_39: - 5.09765625 frac[12,39] + mem[12,39] = 0
 accuracy_10_51: - 5.09765625 frac[10,51] + mem[10,51] = 0
 accuracy_8_31: - 5.09765625 frac[8,31] + mem[8,31] = 0
 accuracy_7_57: - 5.09765625 frac[7,57] + mem[7,57] = 0
 accuracy_13_38: - 5.09765625 frac[13,38] + mem[13,38] = 0
 accuracy_11_58: - 5.09765625 frac[11,58] + mem[11,58] = 0
 accuracy_9_30: - 5.09765625 frac[9,30] + mem[9,30] = 0
 accuracy_14_21: - 5.09765625 frac[14,21] + mem[14,21] = 0
 accuracy_12_61: - 5.09765625 frac[12,61] + mem[12,61] = 0
 accuracy_10_29: - 5.09765625 frac[10,29] + mem[10,29] = 0
 accuracy_13_0: - 5.09765625 frac[13,0] + mem[13,0] = 0
 accuracy_11_16: - 5.09765625 frac[11,16] + mem[11,16] = 0
 accuracy_0_27: - 5.09765625 frac[0,27] + mem[0,27] = 0
 accuracy_14_63: - 5.09765625 frac[14,63] + mem[14,63] = 0
 accuracy_12_27: - 5.09765625 frac[12,27] + mem[12,27] = 0
 accuracy_1_26: - 5.09765625 frac[1,26] + mem[1,26] = 0
 accuracy_13_26: - 5.09765625 frac[13,26] + mem[13,26] = 0
 accuracy_2_1: - 5.09765625 frac[2,1] + mem[2,1] = 0
 accuracy_0_49: - 5.09765625 frac[0,49] + mem[0,49] = 0
 accuracy_14_33: - 5.09765625 frac[14,33] + mem[14,33] = 0
 accuracy_3_28: - 5.09765625 frac[3,28] + mem[3,28] = 0
 accuracy_1_60: - 5.09765625 frac[1,60] + mem[1,60] = 0
 accuracy_3_34: - 5.09765625 frac[3,34] + mem[3,34] = 0
 accuracy_6_29: - 5.09765625 frac[6,29] + mem[6,29] = 0
 accuracy_4_37: - 5.09765625 frac[4,37] + mem[4,37] = 0
 accuracy_8_32: - 5.09765625 frac[8,32] + mem[8,32] = 0
 accuracy_7_24: - 5.09765625 frac[7,24] + mem[7,24] = 0
 accuracy_5_56: - 5.09765625 frac[5,56] + mem[5,56] = 0
 accuracy_9_61: - 5.09765625 frac[9,61] + mem[9,61] = 0
 accuracy_6_55: - 5.09765625 frac[6,55] + mem[6,55] = 0
 accuracy_10_58: - 5.09765625 frac[10,58] + mem[10,58] = 0
 accuracy_8_6: - 5.09765625 frac[8,6] + mem[8,6] = 0
 accuracy_11_51: - 5.09765625 frac[11,51] + mem[11,51] = 0
 accuracy_9_7: - 5.09765625 frac[9,7] + mem[9,7] = 0
 accuracy_14_28: - 5.09765625 frac[14,28] + mem[14,28] = 0
 accuracy_12_52: - 5.09765625 frac[12,52] + mem[12,52] = 0
 accuracy_13_57: - 5.09765625 frac[13,57] + mem[13,57] = 0
 accuracy_11_9: - 5.09765625 frac[11,9] + mem[11,9] = 0
 accuracy_0_18: - 5.09765625 frac[0,18] + mem[0,18] = 0
 accuracy_14_6: - 5.09765625 frac[14,6] + mem[14,6] = 0
 accuracy_12_18: - 5.09765625 frac[12,18] + mem[12,18] = 0
 accuracy_1_19: - 5.09765625 frac[1,19] + mem[1,19] = 0
 accuracy_13_19: - 5.09765625 frac[13,19] + mem[13,19] = 0
 accuracy_2_24: - 5.09765625 frac[2,24] + mem[2,24] = 0
 accuracy_14_40: - 5.09765625 frac[14,40] + mem[14,40] = 0
 accuracy_3_5: - 5.09765625 frac[3,5] + mem[3,5] = 0
 accuracy_1_53: - 5.09765625 frac[1,53] + mem[1,53] = 0
 accuracy_4_6: - 5.09765625 frac[4,6] + mem[4,6] = 0
 accuracy_2_50: - 5.09765625 frac[2,50] + mem[2,50] = 0
 accuracy_0_46: - 5.09765625 frac[0,46] + mem[0,46] = 0
 accuracy_5_7: - 5.09765625 frac[5,7] + mem[5,7] = 0
 accuracy_1_47: - 5.09765625 frac[1,47] + mem[1,47] = 0
 accuracy_4_44: - 5.09765625 frac[4,44] + mem[4,44] = 0
 accuracy_2_44: - 5.09765625 frac[2,44] + mem[2,44] = 0
 accuracy_8_39: - 5.09765625 frac[8,39] + mem[8,39] = 0
 accuracy_7_17: - 5.09765625 frac[7,17] + mem[7,17] = 0
 accuracy_5_33: - 5.09765625 frac[5,33] + mem[5,33] = 0
 accuracy_3_49: - 5.09765625 frac[3,49] + mem[3,49] = 0
 accuracy_10_37: - 5.09765625 frac[10,37] + mem[10,37] = 0
 accuracy_8_13: - 5.09765625 frac[8,13] + mem[8,13] = 0
 accuracy_11_40: - 5.09765625 frac[11,40] + mem[11,40] = 0
 accuracy_9_8: - 5.09765625 frac[9,8] + mem[9,8] = 0
 accuracy_12_51: - 5.09765625 frac[12,51] + mem[12,51] = 0
 accuracy_8_19: - 5.09765625 frac[8,19] + mem[8,19] = 0
 accuracy_13_50: - 5.09765625 frac[13,50] + mem[13,50] = 0
 accuracy_11_14: - 5.09765625 frac[11,14] + mem[11,14] = 0
 accuracy_9_18: - 5.09765625 frac[9,18] + mem[9,18] = 0
 accuracy_14_9: - 5.09765625 frac[14,9] + mem[14,9] = 0
 accuracy_12_9: - 5.09765625 frac[12,9] + mem[12,9] = 0
 accuracy_13_12: - 5.09765625 frac[13,12] + mem[13,12] = 0
 accuracy_2_19: - 5.09765625 frac[2,19] + mem[2,19] = 0
 accuracy_0_15: - 5.09765625 frac[0,15] + mem[0,15] = 0
 accuracy_14_51: - 5.09765625 frac[14,51] + mem[14,51] = 0
 accuracy_3_10: - 5.09765625 frac[3,10] + mem[3,10] = 0
 accuracy_1_14: - 5.09765625 frac[1,14] + mem[1,14] = 0
 accuracy_4_13: - 5.09765625 frac[4,13] + mem[4,13] = 0
 accuracy_2_13: - 5.09765625 frac[2,13] + mem[2,13] = 0
 accuracy_0_37: - 5.09765625 frac[0,37] + mem[0,37] = 0
 accuracy_5_0: - 5.09765625 frac[5,0] + mem[5,0] = 0
 accuracy_1_32: - 5.09765625 frac[1,32] + mem[1,32] = 0
 accuracy_6_15: - 5.09765625 frac[6,15] + mem[6,15] = 0
 accuracy_4_43: - 5.09765625 frac[4,43] + mem[4,43] = 0
 accuracy_8_46: - 5.09765625 frac[8,46] + mem[8,46] = 0
 accuracy_7_14: - 5.09765625 frac[7,14] + mem[7,14] = 0
 accuracy_5_42: - 5.09765625 frac[5,42] + mem[5,42] = 0
 accuracy_3_54: - 5.09765625 frac[3,54] + mem[3,54] = 0
 accuracy_9_47: - 5.09765625 frac[9,47] + mem[9,47] = 0
 accuracy_6_33: - 5.09765625 frac[6,33] + mem[6,33] = 0
 accuracy_10_44: - 5.09765625 frac[10,44] + mem[10,44] = 0
 accuracy_8_52: - 5.09765625 frac[8,52] + mem[8,52] = 0
 accuracy_11_33: - 5.09765625 frac[11,33] + mem[11,33] = 0
 accuracy_9_49: - 5.09765625 frac[9,49] + mem[9,49] = 0
 accuracy_6_59: - 5.09765625 frac[6,59] + mem[6,59] = 0
 accuracy_12_42: - 5.09765625 frac[12,42] + mem[12,42] = 0
 accuracy_10_54: - 5.09765625 frac[10,54] + mem[10,54] = 0
 accuracy_8_26: - 5.09765625 frac[8,26] + mem[8,26] = 0
 accuracy_13_43: - 5.09765625 frac[13,43] + mem[13,43] = 0
 accuracy_11_7: - 5.09765625 frac[11,7] + mem[11,7] = 0
 accuracy_14_16: - 5.09765625 frac[14,16] + mem[14,16] = 0
 accuracy_12_0: - 5.09765625 frac[12,0] + mem[12,0] = 0
 accuracy_10_16: - 5.09765625 frac[10,16] + mem[10,16] = 0
 accuracy_13_5: - 5.09765625 frac[13,5] + mem[13,5] = 0
 accuracy_11_29: - 5.09765625 frac[11,29] + mem[11,29] = 0
 accuracy_0_6: - 5.09765625 frac[0,6] + mem[0,6] = 0
 accuracy_14_58: - 5.09765625 frac[14,58] + mem[14,58] = 0
 accuracy_12_30: - 5.09765625 frac[12,30] + mem[12,30] = 0
 accuracy_13_31: - 5.09765625 frac[13,31] + mem[13,31] = 0
 accuracy_2_4: - 5.09765625 frac[2,4] + mem[2,4] = 0
 accuracy_5_9: - 5.09765625 frac[5,9] + mem[5,9] = 0
 accuracy_3_25: - 5.09765625 frac[3,25] + mem[3,25] = 0
 accuracy_4_18: - 5.09765625 frac[4,18] + mem[4,18] = 0
 accuracy_2_62: - 5.09765625 frac[2,62] + mem[2,62] = 0
 accuracy_7_7: - 5.09765625 frac[7,7] + mem[7,7] = 0
 accuracy_5_19: - 5.09765625 frac[5,19] + mem[5,19] = 0
 accuracy_6_24: - 5.09765625 frac[6,24] + mem[6,24] = 0
 accuracy_7_29: - 5.09765625 frac[7,29] + mem[7,29] = 0
 accuracy_11_38: - 5.09765625 frac[11,38] + mem[11,38] = 0
 accuracy_9_58: - 5.09765625 frac[9,58] + mem[9,58] = 0
 accuracy_12_33: - 5.09765625 frac[12,33] + mem[12,33] = 0
 accuracy_8_1: - 5.09765625 frac[8,1] + mem[8,1] = 0
 accuracy_7_59: - 5.09765625 frac[7,59] + mem[7,59] = 0
 accuracy_13_36: - 5.09765625 frac[13,36] + mem[13,36] = 0
 accuracy_11_60: - 5.09765625 frac[11,60] + mem[11,60] = 0
 accuracy_9_28: - 5.09765625 frac[9,28] + mem[9,28] = 0
 accuracy_14_27: - 5.09765625 frac[14,27] + mem[14,27] = 0
 accuracy_12_63: - 5.09765625 frac[12,63] + mem[12,63] = 0
 accuracy_10_27: - 5.09765625 frac[10,27] + mem[10,27] = 0
 accuracy_13_62: - 5.09765625 frac[13,62] + mem[13,62] = 0
 accuracy_11_18: - 5.09765625 frac[11,18] + mem[11,18] = 0
 accuracy_14_61: - 5.09765625 frac[14,61] + mem[14,61] = 0
 accuracy_12_21: - 5.09765625 frac[12,21] + mem[12,21] = 0
 accuracy_1_24: - 5.09765625 frac[1,24] + mem[1,24] = 0
 accuracy_13_24: - 5.09765625 frac[13,24] + mem[13,24] = 0
 accuracy_2_31: - 5.09765625 frac[2,31] + mem[2,31] = 0
 accuracy_14_39: - 5.09765625 frac[14,39] + mem[14,39] = 0
 accuracy_3_30: - 5.09765625 frac[3,30] + mem[3,30] = 0
 accuracy_1_50: - 5.09765625 frac[1,50] + mem[1,50] = 0
 accuracy_2_57: - 5.09765625 frac[2,57] + mem[2,57] = 0
 accuracy_0_41: - 5.09765625 frac[0,41] + mem[0,41] = 0
 accuracy_5_28: - 5.09765625 frac[5,28] + mem[5,28] = 0
 accuracy_3_36: - 5.09765625 frac[3,36] + mem[3,36] = 0
 accuracy_4_39: - 5.09765625 frac[4,39] + mem[4,39] = 0
 accuracy_7_26: - 5.09765625 frac[7,26] + mem[7,26] = 0
 accuracy_5_38: - 5.09765625 frac[5,38] + mem[5,38] = 0
 accuracy_9_35: - 5.09765625 frac[9,35] + mem[9,35] = 0
 accuracy_6_53: - 5.09765625 frac[6,53] + mem[6,53] = 0
 accuracy_10_56: - 5.09765625 frac[10,56] + mem[10,56] = 0
 accuracy_8_8: - 5.09765625 frac[8,8] + mem[8,8] = 0
 accuracy_7_48: - 5.09765625 frac[7,48] + mem[7,48] = 0
 accuracy_11_53: - 5.09765625 frac[11,53] + mem[11,53] = 0
 accuracy_9_5: - 5.09765625 frac[9,5] + mem[9,5] = 0
 accuracy_12_54: - 5.09765625 frac[12,54] + mem[12,54] = 0
 accuracy_10_2: - 5.09765625 frac[10,2] + mem[10,2] = 0
 accuracy_13_55: - 5.09765625 frac[13,55] + mem[13,55] = 0
 accuracy_14_4: - 5.09765625 frac[14,4] + mem[14,4] = 0
 accuracy_12_12: - 5.09765625 frac[12,12] + mem[12,12] = 0
 accuracy_1_17: - 5.09765625 frac[1,17] + mem[1,17] = 0
 accuracy_13_17: - 5.09765625 frac[13,17] + mem[13,17] = 0
 accuracy_2_22: - 5.09765625 frac[2,22] + mem[2,22] = 0
 accuracy_0_10: - 5.09765625 frac[0,10] + mem[0,10] = 0
 accuracy_14_46: - 5.09765625 frac[14,46] + mem[14,46] = 0
 accuracy_3_7: - 5.09765625 frac[3,7] + mem[3,7] = 0
 accuracy_5_5: - 5.09765625 frac[5,5] + mem[5,5] = 0
 accuracy_3_45: - 5.09765625 frac[3,45] + mem[3,45] = 0
 accuracy_1_45: - 5.09765625 frac[1,45] + mem[1,45] = 0
 accuracy_4_46: - 5.09765625 frac[4,46] + mem[4,46] = 0
 accuracy_2_42: - 5.09765625 frac[2,42] + mem[2,42] = 0
 accuracy_8_41: - 5.09765625 frac[8,41] + mem[8,41] = 0
 accuracy_7_19: - 5.09765625 frac[7,19] + mem[7,19] = 0
 accuracy_5_47: - 5.09765625 frac[5,47] + mem[5,47] = 0
 accuracy_3_51: - 5.09765625 frac[3,51] + mem[3,51] = 0
 accuracy_9_36: - 5.09765625 frac[9,36] + mem[9,36] = 0
 accuracy_6_44: - 5.09765625 frac[6,44] + mem[6,44] = 0
 accuracy_4_52: - 5.09765625 frac[4,52] + mem[4,52] = 0
 accuracy_10_35: - 5.09765625 frac[10,35] + mem[10,35] = 0
 accuracy_8_15: - 5.09765625 frac[8,15] + mem[8,15] = 0
 accuracy_7_41: - 5.09765625 frac[7,41] + mem[7,41] = 0
 accuracy_11_42: - 5.09765625 frac[11,42] + mem[11,42] = 0
 accuracy_9_14: - 5.09765625 frac[9,14] + mem[9,14] = 0
 accuracy_12_45: - 5.09765625 frac[12,45] + mem[12,45] = 0
 accuracy_10_13: - 5.09765625 frac[10,13] + mem[10,13] = 0
 accuracy_8_21: - 5.09765625 frac[8,21] + mem[8,21] = 0
 accuracy_13_48: - 5.09765625 frac[13,48] + mem[13,48] = 0
 accuracy_11_0: - 5.09765625 frac[11,0] + mem[11,0] = 0
 accuracy_9_16: - 5.09765625 frac[9,16] + mem[9,16] = 0
 accuracy_14_15: - 5.09765625 frac[14,15] + mem[14,15] = 0
 accuracy_10_23: - 5.09765625 frac[10,23] + mem[10,23] = 0
 accuracy_13_10: - 5.09765625 frac[13,10] + mem[13,10] = 0
 accuracy_2_17: - 5.09765625 frac[2,17] + mem[2,17] = 0
 accuracy_0_1: - 5.09765625 frac[0,1] + mem[0,1] = 0
 accuracy_14_49: - 5.09765625 frac[14,49] + mem[14,49] = 0
 accuracy_3_12: - 5.09765625 frac[3,12] + mem[3,12] = 0
 accuracy_1_12: - 5.09765625 frac[1,12] + mem[1,12] = 0
 accuracy_0_39: - 5.09765625 frac[0,39] + mem[0,39] = 0
 accuracy_6_13: - 5.09765625 frac[6,13] + mem[6,13] = 0
 accuracy_4_21: - 5.09765625 frac[4,21] + mem[4,21] = 0
 accuracy_2_37: - 5.09765625 frac[2,37] + mem[2,37] = 0
 accuracy_5_40: - 5.09765625 frac[5,40] + mem[5,40] = 0
 accuracy_3_56: - 5.09765625 frac[3,56] + mem[3,56] = 0
 accuracy_9_45: - 5.09765625 frac[9,45] + mem[9,45] = 0
 accuracy_6_39: - 5.09765625 frac[6,39] + mem[6,39] = 0
 accuracy_4_51: - 5.09765625 frac[4,51] + mem[4,51] = 0
 accuracy_10_42: - 5.09765625 frac[10,42] + mem[10,42] = 0
 accuracy_8_54: - 5.09765625 frac[8,54] + mem[8,54] = 0
 accuracy_7_38: - 5.09765625 frac[7,38] + mem[7,38] = 0
 accuracy_11_35: - 5.09765625 frac[11,35] + mem[11,35] = 0
 accuracy_9_55: - 5.09765625 frac[9,55] + mem[9,55] = 0
 accuracy_6_57: - 5.09765625 frac[6,57] + mem[6,57] = 0
 accuracy_12_36: - 5.09765625 frac[12,36] + mem[12,36] = 0
 accuracy_10_52: - 5.09765625 frac[10,52] + mem[10,52] = 0
 accuracy_8_28: - 5.09765625 frac[8,28] + mem[8,28] = 0
 accuracy_7_60: - 5.09765625 frac[7,60] + mem[7,60] = 0
 accuracy_13_41: - 5.09765625 frac[13,41] + mem[13,41] = 0
 accuracy_11_57: - 5.09765625 frac[11,57] + mem[11,57] = 0
 accuracy_9_25: - 5.09765625 frac[9,25] + mem[9,25] = 0
 accuracy_14_22: - 5.09765625 frac[14,22] + mem[14,22] = 0
 accuracy_12_2: - 5.09765625 frac[12,2] + mem[12,2] = 0
 accuracy_10_30: - 5.09765625 frac[10,30] + mem[10,30] = 0
 accuracy_13_3: - 5.09765625 frac[13,3] + mem[13,3] = 0
 accuracy_11_31: - 5.09765625 frac[11,31] + mem[11,31] = 0
 accuracy_0_24: - 5.09765625 frac[0,24] + mem[0,24] = 0
 accuracy_14_56: - 5.09765625 frac[14,56] + mem[14,56] = 0
 accuracy_12_24: - 5.09765625 frac[12,24] + mem[12,24] = 0
 accuracy_1_5: - 5.09765625 frac[1,5] + mem[1,5] = 0
 accuracy_13_29: - 5.09765625 frac[13,29] + mem[13,29] = 0
 accuracy_2_2: - 5.09765625 frac[2,2] + mem[2,2] = 0
 accuracy_14_34: - 5.09765625 frac[14,34] + mem[14,34] = 0
 accuracy_1_63: - 5.09765625 frac[1,63] + mem[1,63] = 0
 accuracy_2_60: - 5.09765625 frac[2,60] + mem[2,60] = 0
 accuracy_7_1: - 5.09765625 frac[7,1] + mem[7,1] = 0
 accuracy_5_17: - 5.09765625 frac[5,17] + mem[5,17] = 0
 accuracy_6_30: - 5.09765625 frac[6,30] + mem[6,30] = 0
 accuracy_4_58: - 5.09765625 frac[4,58] + mem[4,58] = 0
 accuracy_8_61: - 5.09765625 frac[8,61] + mem[8,61] = 0
 accuracy_7_31: - 5.09765625 frac[7,31] + mem[7,31] = 0
 accuracy_5_59: - 5.09765625 frac[5,59] + mem[5,59] = 0
 accuracy_9_56: - 5.09765625 frac[9,56] + mem[9,56] = 0
 accuracy_6_48: - 5.09765625 frac[6,48] + mem[6,48] = 0
 accuracy_12_35: - 5.09765625 frac[12,35] + mem[12,35] = 0
 accuracy_10_63: - 5.09765625 frac[10,63] + mem[10,63] = 0
 accuracy_8_3: - 5.09765625 frac[8,3] + mem[8,3] = 0
 accuracy_13_34: - 5.09765625 frac[13,34] + mem[13,34] = 0
 accuracy_11_62: - 5.09765625 frac[11,62] + mem[11,62] = 0
 accuracy_9_2: - 5.09765625 frac[9,2] + mem[9,2] = 0
 accuracy_12_57: - 5.09765625 frac[12,57] + mem[12,57] = 0
 accuracy_13_60: - 5.09765625 frac[13,60] + mem[13,60] = 0
 accuracy_14_3: - 5.09765625 frac[14,3] + mem[14,3] = 0
 accuracy_12_23: - 5.09765625 frac[12,23] + mem[12,23] = 0
 accuracy_1_30: - 5.09765625 frac[1,30] + mem[1,30] = 0
 accuracy_13_22: - 5.09765625 frac[13,22] + mem[13,22] = 0
 accuracy_2_29: - 5.09765625 frac[2,29] + mem[2,29] = 0
 accuracy_0_53: - 5.09765625 frac[0,53] + mem[0,53] = 0
 accuracy_14_37: - 5.09765625 frac[14,37] + mem[14,37] = 0
 accuracy_3_0: - 5.09765625 frac[3,0] + mem[3,0] = 0
 accuracy_1_48: - 5.09765625 frac[1,48] + mem[1,48] = 0
 accuracy_4_27: - 5.09765625 frac[4,27] + mem[4,27] = 0
 accuracy_2_55: - 5.09765625 frac[2,55] + mem[2,55] = 0
 accuracy_0_43: - 5.09765625 frac[0,43] + mem[0,43] = 0
 accuracy_1_42: - 5.09765625 frac[1,42] + mem[1,42] = 0
 accuracy_4_33: - 5.09765625 frac[4,33] + mem[4,33] = 0
 accuracy_8_36: - 5.09765625 frac[8,36] + mem[8,36] = 0
 accuracy_5_36: - 5.09765625 frac[5,36] + mem[5,36] = 0
 accuracy_9_33: - 5.09765625 frac[9,33] + mem[9,33] = 0
 accuracy_6_43: - 5.09765625 frac[6,43] + mem[6,43] = 0
 accuracy_10_38: - 5.09765625 frac[10,38] + mem[10,38] = 0
 accuracy_8_10: - 5.09765625 frac[8,10] + mem[8,10] = 0
 accuracy_7_50: - 5.09765625 frac[7,50] + mem[7,50] = 0
 accuracy_11_55: - 5.09765625 frac[11,55] + mem[11,55] = 0
 accuracy_12_48: - 5.09765625 frac[12,48] + mem[12,48] = 0
 accuracy_10_0: - 5.09765625 frac[10,0] + mem[10,0] = 0
 accuracy_8_16: - 5.09765625 frac[8,16] + mem[8,16] = 0
 accuracy_13_53: - 5.09765625 frac[13,53] + mem[13,53] = 0
 accuracy_11_13: - 5.09765625 frac[11,13] + mem[11,13] = 0
 accuracy_14_10: - 5.09765625 frac[14,10] + mem[14,10] = 0
 accuracy_12_14: - 5.09765625 frac[12,14] + mem[12,14] = 0
 accuracy_1_23: - 5.09765625 frac[1,23] + mem[1,23] = 0
 accuracy_13_15: - 5.09765625 frac[13,15] + mem[13,15] = 0
 accuracy_2_20: - 5.09765625 frac[2,20] + mem[2,20] = 0
 accuracy_0_12: - 5.09765625 frac[0,12] + mem[0,12] = 0
 accuracy_14_44: - 5.09765625 frac[14,44] + mem[14,44] = 0
 accuracy_3_9: - 5.09765625 frac[3,9] + mem[3,9] = 0
 accuracy_1_9: - 5.09765625 frac[1,9] + mem[1,9] = 0
 accuracy_4_2: - 5.09765625 frac[4,2] + mem[4,2] = 0
 accuracy_2_14: - 5.09765625 frac[2,14] + mem[2,14] = 0
 accuracy_3_47: - 5.09765625 frac[3,47] + mem[3,47] = 0
 accuracy_6_8: - 5.09765625 frac[6,8] + mem[6,8] = 0
 accuracy_4_40: - 5.09765625 frac[4,40] + mem[4,40] = 0
 accuracy_8_43: - 5.09765625 frac[8,43] + mem[8,43] = 0
 accuracy_7_13: - 5.09765625 frac[7,13] + mem[7,13] = 0
 accuracy_5_45: - 5.09765625 frac[5,45] + mem[5,45] = 0
 accuracy_3_53: - 5.09765625 frac[3,53] + mem[3,53] = 0
 accuracy_9_42: - 5.09765625 frac[9,42] + mem[9,42] = 0
 accuracy_6_34: - 5.09765625 frac[6,34] + mem[6,34] = 0
 accuracy_10_33: - 5.09765625 frac[10,33] + mem[10,33] = 0
 accuracy_8_49: - 5.09765625 frac[8,49] + mem[8,49] = 0
 accuracy_7_43: - 5.09765625 frac[7,43] + mem[7,43] = 0
 accuracy_5_55: - 5.09765625 frac[5,55] + mem[5,55] = 0
 accuracy_11_44: - 5.09765625 frac[11,44] + mem[11,44] = 0
 accuracy_9_12: - 5.09765625 frac[9,12] + mem[9,12] = 0
 accuracy_12_47: - 5.09765625 frac[12,47] + mem[12,47] = 0
 accuracy_10_11: - 5.09765625 frac[10,11] + mem[10,11] = 0
 accuracy_8_23: - 5.09765625 frac[8,23] + mem[8,23] = 0
 accuracy_13_46: - 5.09765625 frac[13,46] + mem[13,46] = 0
 accuracy_11_2: - 5.09765625 frac[11,2] + mem[11,2] = 0
 accuracy_9_22: - 5.09765625 frac[9,22] + mem[9,22] = 0
 accuracy_14_13: - 5.09765625 frac[14,13] + mem[14,13] = 0
 accuracy_12_5: - 5.09765625 frac[12,5] + mem[12,5] = 0
 accuracy_10_21: - 5.09765625 frac[10,21] + mem[10,21] = 0
 accuracy_13_8: - 5.09765625 frac[13,8] + mem[13,8] = 0
 accuracy_11_24: - 5.09765625 frac[11,24] + mem[11,24] = 0
 accuracy_0_3: - 5.09765625 frac[0,3] + mem[0,3] = 0
 accuracy_14_55: - 5.09765625 frac[14,55] + mem[14,55] = 0
 accuracy_3_14: - 5.09765625 frac[3,14] + mem[3,14] = 0
 accuracy_1_2: - 5.09765625 frac[1,2] + mem[1,2] = 0
 accuracy_2_9: - 5.09765625 frac[2,9] + mem[2,9] = 0
 accuracy_0_57: - 5.09765625 frac[0,57] + mem[0,57] = 0
 accuracy_5_12: - 5.09765625 frac[5,12] + mem[5,12] = 0
 accuracy_3_20: - 5.09765625 frac[3,20] + mem[3,20] = 0
 accuracy_6_3: - 5.09765625 frac[6,3] + mem[6,3] = 0
 accuracy_4_23: - 5.09765625 frac[4,23] + mem[4,23] = 0
 accuracy_2_35: - 5.09765625 frac[2,35] + mem[2,35] = 0
 accuracy_7_10: - 5.09765625 frac[7,10] + mem[7,10] = 0
 accuracy_3_58: - 5.09765625 frac[3,58] + mem[3,58] = 0
 accuracy_6_37: - 5.09765625 frac[6,37] + mem[6,37] = 0
 accuracy_4_61: - 5.09765625 frac[4,61] + mem[4,61] = 0
 accuracy_10_40: - 5.09765625 frac[10,40] + mem[10,40] = 0
 accuracy_8_56: - 5.09765625 frac[8,56] + mem[8,56] = 0
 accuracy_5_48: - 5.09765625 frac[5,48] + mem[5,48] = 0
 accuracy_11_37: - 5.09765625 frac[11,37] + mem[11,37] = 0
 accuracy_9_53: - 5.09765625 frac[9,53] + mem[9,53] = 0
 accuracy_6_63: - 5.09765625 frac[6,63] + mem[6,63] = 0
 accuracy_12_38: - 5.09765625 frac[12,38] + mem[12,38] = 0
 accuracy_10_50: - 5.09765625 frac[10,50] + mem[10,50] = 0
 accuracy_8_30: - 5.09765625 frac[8,30] + mem[8,30] = 0
 accuracy_7_62: - 5.09765625 frac[7,62] + mem[7,62] = 0
 accuracy_13_39: - 5.09765625 frac[13,39] + mem[13,39] = 0
 accuracy_11_59: - 5.09765625 frac[11,59] + mem[11,59] = 0
 accuracy_9_31: - 5.09765625 frac[9,31] + mem[9,31] = 0
 accuracy_14_20: - 5.09765625 frac[14,20] + mem[14,20] = 0
 accuracy_12_60: - 5.09765625 frac[12,60] + mem[12,60] = 0
 accuracy_10_28: - 5.09765625 frac[10,28] + mem[10,28] = 0
 accuracy_13_1: - 5.09765625 frac[13,1] + mem[13,1] = 0
 accuracy_11_17: - 5.09765625 frac[11,17] + mem[11,17] = 0
 accuracy_0_26: - 5.09765625 frac[0,26] + mem[0,26] = 0
 accuracy_14_62: - 5.09765625 frac[14,62] + mem[14,62] = 0
 accuracy_12_26: - 5.09765625 frac[12,26] + mem[12,26] = 0
 accuracy_1_27: - 5.09765625 frac[1,27] + mem[1,27] = 0
 accuracy_13_27: - 5.09765625 frac[13,27] + mem[13,27] = 0
 accuracy_0_48: - 5.09765625 frac[0,48] + mem[0,48] = 0
 accuracy_14_32: - 5.09765625 frac[14,32] + mem[14,32] = 0
 accuracy_3_29: - 5.09765625 frac[3,29] + mem[3,29] = 0
 accuracy_4_30: - 5.09765625 frac[4,30] + mem[4,30] = 0
 accuracy_2_58: - 5.09765625 frac[2,58] + mem[2,58] = 0
 mem_tot_CPU1: - mem[0,17] - mem[0,55] - mem[0,36] - mem[0,5] - mem[0,28]
   - mem[0,50] - mem[0,40] - mem[0,47] - mem[0,0] - mem[0,38] - mem[0,7]
   - mem[0,61] - mem[0,30] - mem[0,52] - mem[0,21] - mem[0,33] - mem[0,25]
   - mem[0,63] - mem[0,16] - mem[0,54] - mem[0,44] - mem[0,13] - mem[0,58]
   - mem[0,27] - mem[0,49] - mem[0,18] - mem[0,46] - mem[0,15] - mem[0,37]
   - mem[0,6] - mem[0,41] - mem[0,10] - mem[0,1] - mem[0,39] - mem[0,24]
   - mem[0,53] - mem[0,43] - mem[0,12] - mem[0,3] - mem[0,57] - mem[0,26]
   - mem[0,48] + mem_tot_CPU1 = 0
 rows_tot_CPU1: - 3 frac[0,17] - 3 frac[0,55] - 3 frac[0,36] - 3 frac[0,5]
   - 3 frac[0,28] - 3 frac[0,50] - 3 frac[0,40] - 3 frac[0,47]
   - 3 frac[0,0] - 3 frac[0,38] - 3 frac[0,7] - 3 frac[0,61] - 3 frac[0,30]
   - 3 frac[0,52] - 3 frac[0,21] - 3 frac[0,33] - 3 frac[0,25]
   - 3 frac[0,63] - 3 frac[0,16] - 3 frac[0,54] - 3 frac[0,44]
   - 3 frac[0,13] - 3 frac[0,58] - 3 frac[0,27] - 3 frac[0,49]
   - 3 frac[0,18] - 3 frac[0,46] - 3 frac[0,15] - 3 frac[0,37]
   - 3 frac[0,6] - 3 frac[0,41] - 3 frac[0,10] - 3 frac[0,1] - 3 frac[0,39]
   - 3 frac[0,24] - 3 frac[0,53] - 3 frac[0,43] - 3 frac[0,12]
   - 3 frac[0,3] - 3 frac[0,57] - 3 frac[0,26] - 3 frac[0,48]
   + rows_tot_CPU1 = 0
 mem_tot_CPU2: - mem[1,28] - mem[1,54] - mem[1,40] - mem[1,21] - mem[1,33]
   - mem[1,0] - mem[1,58] - mem[1,25] - mem[1,51] - mem[1,18] - mem[1,52]
   - mem[1,46] - mem[1,39] - mem[1,6] - mem[1,56] - mem[1,31] - mem[1,43]
   - mem[1,16] - mem[1,44] - mem[1,3] - mem[1,37] - mem[1,62] - mem[1,29]
   - mem[1,8] - mem[1,1] - mem[1,26] - mem[1,60] - mem[1,19] - mem[1,53]
   - mem[1,47] - mem[1,14] - mem[1,32] - mem[1,24] - mem[1,50] - mem[1,17]
   - mem[1,45] - mem[1,12] - mem[1,5] - mem[1,63] - mem[1,30] - mem[1,48]
   - mem[1,42] - mem[1,23] - mem[1,9] - mem[1,2] - mem[1,27] + mem_tot_CPU2
   = 0
 rows_tot_CPU2: - 3 frac[1,28] - 3 frac[1,54] - 3 frac[1,40] - 3 frac[1,21]
   - 3 frac[1,33] - 3 frac[1,0] - 3 frac[1,58] - 3 frac[1,25]
   - 3 frac[1,51] - 3 frac[1,18] - 3 frac[1,52] - 3 frac[1,46]
   - 3 frac[1,39] - 3 frac[1,6] - 3 frac[1,56] - 3 frac[1,31]
   - 3 frac[1,43] - 3 frac[1,16] - 3 frac[1,44] - 3 frac[1,3]
   - 3 frac[1,37] - 3 frac[1,62] - 3 frac[1,29] - 3 frac[1,8] - 3 frac[1,1]
   - 3 frac[1,26] - 3 frac[1,60] - 3 frac[1,19] - 3 frac[1,53]
   - 3 frac[1,47] - 3 frac[1,14] - 3 frac[1,32] - 3 frac[1,24]
   - 3 frac[1,50] - 3 frac[1,17] - 3 frac[1,45] - 3 frac[1,12]
   - 3 frac[1,5] - 3 frac[1,63] - 3 frac[1,30] - 3 frac[1,48]
   - 3 frac[1,42] - 3 frac[1,23] - 3 frac[1,9] - 3 frac[1,2] - 3 frac[1,27]
   + rows_tot_CPU2 = 0
 mem_tot_CPU3: - mem[2,53] - mem[2,47] - mem[2,18] - mem[2,12] - mem[2,33]
   - mem[2,25] - mem[2,51] - mem[2,45] - mem[2,16] - mem[2,36] - mem[2,5]
   - mem[2,54] - mem[2,49] - mem[2,43] - mem[2,8] - mem[2,34] - mem[2,26]
   - mem[2,52] - mem[2,46] - mem[2,21] - mem[2,6] - mem[2,1] - mem[2,24]
   - mem[2,50] - mem[2,44] - mem[2,19] - mem[2,13] - mem[2,4] - mem[2,62]
   - mem[2,31] - mem[2,57] - mem[2,22] - mem[2,42] - mem[2,17] - mem[2,37]
   - mem[2,2] - mem[2,60] - mem[2,29] - mem[2,55] - mem[2,20] - mem[2,14]
   - mem[2,9] - mem[2,35] - mem[2,58] + mem_tot_CPU3 = 0
 rows_tot_CPU3: - 3 frac[2,53] - 3 frac[2,47] - 3 frac[2,18] - 3 frac[2,12]
   - 3 frac[2,33] - 3 frac[2,25] - 3 frac[2,51] - 3 frac[2,45]
   - 3 frac[2,16] - 3 frac[2,36] - 3 frac[2,5] - 3 frac[2,54]
   - 3 frac[2,49] - 3 frac[2,43] - 3 frac[2,8] - 3 frac[2,34]
   - 3 frac[2,26] - 3 frac[2,52] - 3 frac[2,46] - 3 frac[2,21]
   - 3 frac[2,6] - 3 frac[2,1] - 3 frac[2,24] - 3 frac[2,50] - 3 frac[2,44]
   - 3 frac[2,19] - 3 frac[2,13] - 3 frac[2,4] - 3 frac[2,62]
   - 3 frac[2,31] - 3 frac[2,57] - 3 frac[2,22] - 3 frac[2,42]
   - 3 frac[2,17] - 3 frac[2,37] - 3 frac[2,2] - 3 frac[2,60]
   - 3 frac[2,29] - 3 frac[2,55] - 3 frac[2,20] - 3 frac[2,14]
   - 3 frac[2,9] - 3 frac[2,35] - 3 frac[2,58] + rows_tot_CPU3 = 0
 mem_tot_CPU4: - mem[3,35] - mem[3,40] - mem[3,55] - mem[3,22] - mem[3,31]
   - mem[3,37] - mem[3,42] - mem[3,48] - mem[3,13] - mem[3,19] - mem[3,57]
   - mem[3,24] - mem[3,62] - mem[3,1] - mem[3,39] - mem[3,6] - mem[3,44]
   - mem[3,15] - mem[3,21] - mem[3,59] - mem[3,26] - mem[3,8] - mem[3,46]
   - mem[3,52] - mem[3,23] - mem[3,61] - mem[3,28] - mem[3,34] - mem[3,5]
   - mem[3,49] - mem[3,10] - mem[3,54] - mem[3,25] - mem[3,30] - mem[3,36]
   - mem[3,7] - mem[3,45] - mem[3,51] - mem[3,12] - mem[3,56] - mem[3,0]
   - mem[3,9] - mem[3,47] - mem[3,53] - mem[3,14] - mem[3,20] - mem[3,58]
   - mem[3,29] + mem_tot_CPU4 = 0
 rows_tot_CPU4: - 3 frac[3,35] - 3 frac[3,40] - 3 frac[3,55] - 3 frac[3,22]
   - 3 frac[3,31] - 3 frac[3,37] - 3 frac[3,42] - 3 frac[3,48]
   - 3 frac[3,13] - 3 frac[3,19] - 3 frac[3,57] - 3 frac[3,24]
   - 3 frac[3,62] - 3 frac[3,1] - 3 frac[3,39] - 3 frac[3,6] - 3 frac[3,44]
   - 3 frac[3,15] - 3 frac[3,21] - 3 frac[3,59] - 3 frac[3,26]
   - 3 frac[3,8] - 3 frac[3,46] - 3 frac[3,52] - 3 frac[3,23]
   - 3 frac[3,61] - 3 frac[3,28] - 3 frac[3,34] - 3 frac[3,5]
   - 3 frac[3,49] - 3 frac[3,10] - 3 frac[3,54] - 3 frac[3,25]
   - 3 frac[3,30] - 3 frac[3,36] - 3 frac[3,7] - 3 frac[3,45]
   - 3 frac[3,51] - 3 frac[3,12] - 3 frac[3,56] - 3 frac[3,0] - 3 frac[3,9]
   - 3 frac[3,47] - 3 frac[3,53] - 3 frac[3,14] - 3 frac[3,20]
   - 3 frac[3,58] - 3 frac[3,29] + rows_tot_CPU4 = 0
 mem_tot_CPU5: - mem[4,36] - mem[4,5] - mem[4,35] - mem[4,12] - mem[4,42]
   - mem[4,11] - mem[4,63] - mem[4,38] - mem[4,45] - mem[4,14] - mem[4,50]
   - mem[4,19] - mem[4,1] - mem[4,47] - mem[4,53] - mem[4,22] - mem[4,60]
   - mem[4,29] - mem[4,59] - mem[4,34] - mem[4,3] - mem[4,55] - mem[4,62]
   - mem[4,37] - mem[4,6] - mem[4,44] - mem[4,13] - mem[4,43] - mem[4,18]
   - mem[4,39] - mem[4,46] - mem[4,52] - mem[4,21] - mem[4,51] - mem[4,58]
   - mem[4,27] - mem[4,33] - mem[4,2] - mem[4,40] - mem[4,23] - mem[4,61]
   - mem[4,30] + mem_tot_CPU5 = 0
 rows_tot_CPU5: - 3 frac[4,36] - 3 frac[4,5] - 3 frac[4,35] - 3 frac[4,12]
   - 3 frac[4,42] - 3 frac[4,11] - 3 frac[4,63] - 3 frac[4,38]
   - 3 frac[4,45] - 3 frac[4,14] - 3 frac[4,50] - 3 frac[4,19]
   - 3 frac[4,1] - 3 frac[4,47] - 3 frac[4,53] - 3 frac[4,22]
   - 3 frac[4,60] - 3 frac[4,29] - 3 frac[4,59] - 3 frac[4,34]
   - 3 frac[4,3] - 3 frac[4,55] - 3 frac[4,62] - 3 frac[4,37] - 3 frac[4,6]
   - 3 frac[4,44] - 3 frac[4,13] - 3 frac[4,43] - 3 frac[4,18]
   - 3 frac[4,39] - 3 frac[4,46] - 3 frac[4,52] - 3 frac[4,21]
   - 3 frac[4,51] - 3 frac[4,58] - 3 frac[4,27] - 3 frac[4,33]
   - 3 frac[4,2] - 3 frac[4,40] - 3 frac[4,23] - 3 frac[4,61]
   - 3 frac[4,30] + rows_tot_CPU5 = 0
 mem_tot_CPU6: - mem[5,31] - mem[5,57] - mem[5,24] - mem[5,34] - mem[5,1]
   - mem[5,53] - mem[5,10] - mem[5,20] - mem[5,62] - mem[5,29] - mem[5,39]
   - mem[5,32] - mem[5,41] - mem[5,51] - mem[5,18] - mem[5,60] - mem[5,27]
   - mem[5,37] - mem[5,46] - mem[5,13] - mem[5,23] - mem[5,16] - mem[5,58]
   - mem[5,35] - mem[5,44] - mem[5,54] - mem[5,11] - mem[5,21] - mem[5,63]
   - mem[5,56] - mem[5,7] - mem[5,33] - mem[5,0] - mem[5,42] - mem[5,9]
   - mem[5,19] - mem[5,28] - mem[5,38] - mem[5,5] - mem[5,47] - mem[5,40]
   - mem[5,17] - mem[5,59] - mem[5,36] - mem[5,45] - mem[5,55] - mem[5,12]
   - mem[5,48] + mem_tot_CPU6 = 0
 rows_tot_CPU6: - 3 frac[5,31] - 3 frac[5,57] - 3 frac[5,24] - 3 frac[5,34]
   - 3 frac[5,1] - 3 frac[5,53] - 3 frac[5,10] - 3 frac[5,20]
   - 3 frac[5,62] - 3 frac[5,29] - 3 frac[5,39] - 3 frac[5,32]
   - 3 frac[5,41] - 3 frac[5,51] - 3 frac[5,18] - 3 frac[5,60]
   - 3 frac[5,27] - 3 frac[5,37] - 3 frac[5,46] - 3 frac[5,13]
   - 3 frac[5,23] - 3 frac[5,16] - 3 frac[5,58] - 3 frac[5,35]
   - 3 frac[5,44] - 3 frac[5,54] - 3 frac[5,11] - 3 frac[5,21]
   - 3 frac[5,63] - 3 frac[5,56] - 3 frac[5,7] - 3 frac[5,33] - 3 frac[5,0]
   - 3 frac[5,42] - 3 frac[5,9] - 3 frac[5,19] - 3 frac[5,28]
   - 3 frac[5,38] - 3 frac[5,5] - 3 frac[5,47] - 3 frac[5,40]
   - 3 frac[5,17] - 3 frac[5,59] - 3 frac[5,36] - 3 frac[5,45]
   - 3 frac[5,55] - 3 frac[5,12] - 3 frac[5,48] + rows_tot_CPU6 = 0
 mem_tot_CPU7: - mem[6,54] - mem[6,32] - mem[6,58] - mem[6,1] - mem[6,61]
   - mem[6,21] - mem[6,47] - mem[6,12] - mem[6,56] - mem[6,7] - mem[6,16]
   - mem[6,42] - mem[6,45] - mem[6,2] - mem[6,36] - mem[6,5] - mem[6,31]
   - mem[6,22] - mem[6,40] - mem[6,35] - mem[6,0] - mem[6,26] - mem[6,60]
   - mem[6,29] - mem[6,55] - mem[6,15] - mem[6,33] - mem[6,59] - mem[6,24]
   - mem[6,53] - mem[6,44] - mem[6,13] - mem[6,39] - mem[6,57] - mem[6,30]
   - mem[6,48] - mem[6,43] - mem[6,8] - mem[6,34] - mem[6,3] - mem[6,37]
   - mem[6,63] + mem_tot_CPU7 = 0
 rows_tot_CPU7: - 3 frac[6,54] - 3 frac[6,32] - 3 frac[6,58] - 3 frac[6,1]
   - 3 frac[6,61] - 3 frac[6,21] - 3 frac[6,47] - 3 frac[6,12]
   - 3 frac[6,56] - 3 frac[6,7] - 3 frac[6,16] - 3 frac[6,42]
   - 3 frac[6,45] - 3 frac[6,2] - 3 frac[6,36] - 3 frac[6,5] - 3 frac[6,31]
   - 3 frac[6,22] - 3 frac[6,40] - 3 frac[6,35] - 3 frac[6,0]
   - 3 frac[6,26] - 3 frac[6,60] - 3 frac[6,29] - 3 frac[6,55]
   - 3 frac[6,15] - 3 frac[6,33] - 3 frac[6,59] - 3 frac[6,24]
   - 3 frac[6,53] - 3 frac[6,44] - 3 frac[6,13] - 3 frac[6,39]
   - 3 frac[6,57] - 3 frac[6,30] - 3 frac[6,48] - 3 frac[6,43]
   - 3 frac[6,8] - 3 frac[6,34] - 3 frac[6,3] - 3 frac[6,37] - 3 frac[6,63]
   + rows_tot_CPU7 = 0
 mem_tot_CPU8: - mem[7,3] - mem[7,55] - mem[7,22] - mem[7,4] - mem[7,27]
   - mem[7,49] - mem[7,16] - mem[7,46] - mem[7,9] - mem[7,39] - mem[7,61]
   - mem[7,6] - mem[7,28] - mem[7,58] - mem[7,21] - mem[7,51] - mem[7,18]
   - mem[7,40] - mem[7,63] - mem[7,0] - mem[7,52] - mem[7,23] - mem[7,45]
   - mem[7,12] - mem[7,42] - mem[7,5] - mem[7,35] - mem[7,57] - mem[7,24]
   - mem[7,17] - mem[7,14] - mem[7,7] - mem[7,29] - mem[7,59] - mem[7,26]
   - mem[7,48] - mem[7,19] - mem[7,41] - mem[7,38] - mem[7,60] - mem[7,1]
   - mem[7,31] - mem[7,50] - mem[7,13] - mem[7,43] - mem[7,10] - mem[7,62]
   + mem_tot_CPU8 = 0
 rows_tot_CPU8: - 3 frac[7,3] - 3 frac[7,55] - 3 frac[7,22] - 3 frac[7,4]
   - 3 frac[7,27] - 3 frac[7,49] - 3 frac[7,16] - 3 frac[7,46]
   - 3 frac[7,9] - 3 frac[7,39] - 3 frac[7,61] - 3 frac[7,6] - 3 frac[7,28]
   - 3 frac[7,58] - 3 frac[7,21] - 3 frac[7,51] - 3 frac[7,18]
   - 3 frac[7,40] - 3 frac[7,63] - 3 frac[7,0] - 3 frac[7,52]
   - 3 frac[7,23] - 3 frac[7,45] - 3 frac[7,12] - 3 frac[7,42]
   - 3 frac[7,5] - 3 frac[7,35] - 3 frac[7,57] - 3 frac[7,24]
   - 3 frac[7,17] - 3 frac[7,14] - 3 frac[7,7] - 3 frac[7,29]
   - 3 frac[7,59] - 3 frac[7,26] - 3 frac[7,48] - 3 frac[7,19]
   - 3 frac[7,41] - 3 frac[7,38] - 3 frac[7,60] - 3 frac[7,1]
   - 3 frac[7,31] - 3 frac[7,50] - 3 frac[7,13] - 3 frac[7,43]
   - 3 frac[7,10] - 3 frac[7,62] + rows_tot_CPU8 = 0
 mem_tot_tor_P41: - mem[8,63] - mem[8,5] - mem[8,38] - mem[8,12]
   - mem[8,18] - mem[8,45] - mem[8,51] - mem[8,25] - mem[8,58] - mem[8,0]
   - mem[8,33] - mem[8,7] - mem[8,40] - mem[8,14] - mem[8,47] - mem[8,53]
   - mem[8,27] - mem[8,60] - mem[8,2] - mem[8,9] - mem[8,42] - mem[8,48]
   - mem[8,55] - mem[8,29] - mem[8,62] - mem[8,37] - mem[8,17] - mem[8,44]
   - mem[8,50] - mem[8,24] - mem[8,57] - mem[8,31] - mem[8,32] - mem[8,6]
   - mem[8,39] - mem[8,13] - mem[8,19] - mem[8,46] - mem[8,52] - mem[8,26]
   - mem[8,1] - mem[8,8] - mem[8,41] - mem[8,15] - mem[8,21] - mem[8,54]
   - mem[8,28] - mem[8,61] - mem[8,3] - mem[8,36] - mem[8,10] - mem[8,16]
   - mem[8,43] - mem[8,49] - mem[8,23] - mem[8,56] - mem[8,30]
   + mem_tot_tor_P41 = 0
 rows_tot_tor_P41: - 3 frac[8,63] - 3 frac[8,5] - 3 frac[8,38]
   - 3 frac[8,12] - 3 frac[8,18] - 3 frac[8,45] - 3 frac[8,51]
   - 3 frac[8,25] - 3 frac[8,58] - 3 frac[8,0] - 3 frac[8,33] - 3 frac[8,7]
   - 3 frac[8,40] - 3 frac[8,14] - 3 frac[8,47] - 3 frac[8,53]
   - 3 frac[8,27] - 3 frac[8,60] - 3 frac[8,2] - 3 frac[8,9] - 3 frac[8,42]
   - 3 frac[8,48] - 3 frac[8,55] - 3 frac[8,29] - 3 frac[8,62]
   - 3 frac[8,37] - 3 frac[8,17] - 3 frac[8,44] - 3 frac[8,50]
   - 3 frac[8,24] - 3 frac[8,57] - 3 frac[8,31] - 3 frac[8,32]
   - 3 frac[8,6] - 3 frac[8,39] - 3 frac[8,13] - 3 frac[8,19]
   - 3 frac[8,46] - 3 frac[8,52] - 3 frac[8,26] - 3 frac[8,1] - 3 frac[8,8]
   - 3 frac[8,41] - 3 frac[8,15] - 3 frac[8,21] - 3 frac[8,54]
   - 3 frac[8,28] - 3 frac[8,61] - 3 frac[8,3] - 3 frac[8,36]
   - 3 frac[8,10] - 3 frac[8,16] - 3 frac[8,43] - 3 frac[8,49]
   - 3 frac[8,23] - 3 frac[8,56] - 3 frac[8,30] + rows_tot_tor_P41 = 0
 mem_tot_tor_P42: - mem[9,62] - mem[9,0] - mem[9,39] - mem[9,9] - mem[9,19]
   - mem[9,40] - mem[9,50] - mem[9,20] - mem[9,59] - mem[9,29] - mem[9,60]
   - mem[9,6] - mem[9,37] - mem[9,15] - mem[9,17] - mem[9,46] - mem[9,48]
   - mem[9,26] - mem[9,57] - mem[9,34] - mem[9,4] - mem[9,43] - mem[9,13]
   - mem[9,23] - mem[9,44] - mem[9,54] - mem[9,24] - mem[9,1] - mem[9,10]
   - mem[9,51] - mem[9,21] - mem[9,52] - mem[9,30] - mem[9,61] - mem[9,7]
   - mem[9,8] - mem[9,18] - mem[9,47] - mem[9,49] - mem[9,58] - mem[9,28]
   - mem[9,35] - mem[9,5] - mem[9,36] - mem[9,14] - mem[9,16] - mem[9,45]
   - mem[9,55] - mem[9,25] - mem[9,56] - mem[9,2] - mem[9,33] - mem[9,42]
   - mem[9,12] - mem[9,22] - mem[9,53] - mem[9,31] + mem_tot_tor_P42 = 0
 rows_tot_tor_P42: - 3 frac[9,62] - 3 frac[9,0] - 3 frac[9,39]
   - 3 frac[9,9] - 3 frac[9,19] - 3 frac[9,40] - 3 frac[9,50]
   - 3 frac[9,20] - 3 frac[9,59] - 3 frac[9,29] - 3 frac[9,60]
   - 3 frac[9,6] - 3 frac[9,37] - 3 frac[9,15] - 3 frac[9,17]
   - 3 frac[9,46] - 3 frac[9,48] - 3 frac[9,26] - 3 frac[9,57]
   - 3 frac[9,34] - 3 frac[9,4] - 3 frac[9,43] - 3 frac[9,13]
   - 3 frac[9,23] - 3 frac[9,44] - 3 frac[9,54] - 3 frac[9,24]
   - 3 frac[9,1] - 3 frac[9,10] - 3 frac[9,51] - 3 frac[9,21]
   - 3 frac[9,52] - 3 frac[9,30] - 3 frac[9,61] - 3 frac[9,7] - 3 frac[9,8]
   - 3 frac[9,18] - 3 frac[9,47] - 3 frac[9,49] - 3 frac[9,58]
   - 3 frac[9,28] - 3 frac[9,35] - 3 frac[9,5] - 3 frac[9,36]
   - 3 frac[9,14] - 3 frac[9,16] - 3 frac[9,45] - 3 frac[9,55]
   - 3 frac[9,25] - 3 frac[9,56] - 3 frac[9,2] - 3 frac[9,33]
   - 3 frac[9,42] - 3 frac[9,12] - 3 frac[9,22] - 3 frac[9,53]
   - 3 frac[9,31] + rows_tot_tor_P42 = 0
 mem_tot_tor_P43: - mem[10,61] - mem[10,7] - mem[10,36] - mem[10,14]
   - mem[10,47] - mem[10,9] - mem[10,19] - mem[10,48] - mem[10,59]
   - mem[10,5] - mem[10,34] - mem[10,12] - mem[10,22] - mem[10,45]
   - mem[10,55] - mem[10,17] - mem[10,62] - mem[10,24] - mem[10,57]
   - mem[10,3] - mem[10,32] - mem[10,10] - mem[10,20] - mem[10,43]
   - mem[10,53] - mem[10,31] - mem[10,60] - mem[10,6] - mem[10,39]
   - mem[10,1] - mem[10,46] - mem[10,18] - mem[10,41] - mem[10,51]
   - mem[10,29] - mem[10,58] - mem[10,37] - mem[10,44] - mem[10,54]
   - mem[10,16] - mem[10,27] - mem[10,56] - mem[10,2] - mem[10,35]
   - mem[10,13] - mem[10,23] - mem[10,42] - mem[10,52] - mem[10,30]
   - mem[10,63] - mem[10,38] - mem[10,0] - mem[10,33] - mem[10,11]
   - mem[10,21] - mem[10,40] - mem[10,50] - mem[10,28] + mem_tot_tor_P43
   = 0
 rows_tot_tor_P43: - 3 frac[10,61] - 3 frac[10,7] - 3 frac[10,36]
   - 3 frac[10,14] - 3 frac[10,47] - 3 frac[10,9] - 3 frac[10,19]
   - 3 frac[10,48] - 3 frac[10,59] - 3 frac[10,5] - 3 frac[10,34]
   - 3 frac[10,12] - 3 frac[10,22] - 3 frac[10,45] - 3 frac[10,55]
   - 3 frac[10,17] - 3 frac[10,62] - 3 frac[10,24] - 3 frac[10,57]
   - 3 frac[10,3] - 3 frac[10,32] - 3 frac[10,10] - 3 frac[10,20]
   - 3 frac[10,43] - 3 frac[10,53] - 3 frac[10,31] - 3 frac[10,60]
   - 3 frac[10,6] - 3 frac[10,39] - 3 frac[10,1] - 3 frac[10,46]
   - 3 frac[10,18] - 3 frac[10,41] - 3 frac[10,51] - 3 frac[10,29]
   - 3 frac[10,58] - 3 frac[10,37] - 3 frac[10,44] - 3 frac[10,54]
   - 3 frac[10,16] - 3 frac[10,27] - 3 frac[10,56] - 3 frac[10,2]
   - 3 frac[10,35] - 3 frac[10,13] - 3 frac[10,23] - 3 frac[10,42]
   - 3 frac[10,52] - 3 frac[10,30] - 3 frac[10,63] - 3 frac[10,38]
   - 3 frac[10,0] - 3 frac[10,33] - 3 frac[10,11] - 3 frac[10,21]
   - 3 frac[10,40] - 3 frac[10,50] - 3 frac[10,28] + rows_tot_tor_P43 = 0
 mem_tot_tor_P44: - mem[11,48] - mem[11,22] - mem[11,41] - mem[11,15]
   - mem[11,46] - mem[11,4] - mem[11,26] - mem[11,39] - mem[11,61]
   - mem[11,19] - mem[11,50] - mem[11,8] - mem[11,43] - mem[11,1]
   - mem[11,32] - mem[11,6] - mem[11,28] - mem[11,63] - mem[11,21]
   - mem[11,52] - mem[11,10] - mem[11,45] - mem[11,3] - mem[11,34]
   - mem[11,56] - mem[11,30] - mem[11,49] - mem[11,23] - mem[11,54]
   - mem[11,12] - mem[11,47] - mem[11,5] - mem[11,27] - mem[11,36]
   - mem[11,58] - mem[11,16] - mem[11,51] - mem[11,9] - mem[11,40]
   - mem[11,14] - mem[11,33] - mem[11,7] - mem[11,29] - mem[11,38]
   - mem[11,60] - mem[11,18] - mem[11,53] - mem[11,42] - mem[11,0]
   - mem[11,35] - mem[11,57] - mem[11,31] - mem[11,62] - mem[11,55]
   - mem[11,13] - mem[11,44] - mem[11,2] - mem[11,24] - mem[11,37]
   - mem[11,59] - mem[11,17] + mem_tot_tor_P44 = 0
 rows_tot_tor_P44: - 3 frac[11,48] - 3 frac[11,22] - 3 frac[11,41]
   - 3 frac[11,15] - 3 frac[11,46] - 3 frac[11,4] - 3 frac[11,26]
   - 3 frac[11,39] - 3 frac[11,61] - 3 frac[11,19] - 3 frac[11,50]
   - 3 frac[11,8] - 3 frac[11,43] - 3 frac[11,1] - 3 frac[11,32]
   - 3 frac[11,6] - 3 frac[11,28] - 3 frac[11,63] - 3 frac[11,21]
   - 3 frac[11,52] - 3 frac[11,10] - 3 frac[11,45] - 3 frac[11,3]
   - 3 frac[11,34] - 3 frac[11,56] - 3 frac[11,30] - 3 frac[11,49]
   - 3 frac[11,23] - 3 frac[11,54] - 3 frac[11,12] - 3 frac[11,47]
   - 3 frac[11,5] - 3 frac[11,27] - 3 frac[11,36] - 3 frac[11,58]
   - 3 frac[11,16] - 3 frac[11,51] - 3 frac[11,9] - 3 frac[11,40]
   - 3 frac[11,14] - 3 frac[11,33] - 3 frac[11,7] - 3 frac[11,29]
   - 3 frac[11,38] - 3 frac[11,60] - 3 frac[11,18] - 3 frac[11,53]
   - 3 frac[11,42] - 3 frac[11,0] - 3 frac[11,35] - 3 frac[11,57]
   - 3 frac[11,31] - 3 frac[11,62] - 3 frac[11,55] - 3 frac[11,13]
   - 3 frac[11,44] - 3 frac[11,2] - 3 frac[11,24] - 3 frac[11,37]
   - 3 frac[11,59] - 3 frac[11,17] + rows_tot_tor_P44 = 0
 mem_tot_l1_P41: - mem[12,59] - mem[12,17] - mem[12,50] - mem[12,8]
   - mem[12,41] - mem[12,7] - mem[12,29] - mem[12,32] - mem[12,62]
   - mem[12,20] - mem[12,53] - mem[12,19] - mem[12,44] - mem[12,10]
   - mem[12,43] - mem[12,1] - mem[12,31] - mem[12,34] - mem[12,56]
   - mem[12,22] - mem[12,55] - mem[12,13] - mem[12,46] - mem[12,4]
   - mem[12,37] - mem[12,3] - mem[12,25] - mem[12,58] - mem[12,16]
   - mem[12,49] - mem[12,15] - mem[12,40] - mem[12,6] - mem[12,28]
   - mem[12,39] - mem[12,61] - mem[12,27] - mem[12,52] - mem[12,18]
   - mem[12,51] - mem[12,9] - mem[12,42] - mem[12,0] - mem[12,30]
   - mem[12,33] - mem[12,63] - mem[12,21] - mem[12,54] - mem[12,12]
   - mem[12,45] - mem[12,36] - mem[12,2] - mem[12,24] - mem[12,35]
   - mem[12,57] - mem[12,23] - mem[12,48] - mem[12,14] - mem[12,47]
   - mem[12,5] - mem[12,38] - mem[12,60] - mem[12,26] + mem_tot_l1_P41 = 0
 rows_tot_l1_P41: - 3 frac[12,59] - 3 frac[12,17] - 3 frac[12,50]
   - 3 frac[12,8] - 3 frac[12,41] - 3 frac[12,7] - 3 frac[12,29]
   - 3 frac[12,32] - 3 frac[12,62] - 3 frac[12,20] - 3 frac[12,53]
   - 3 frac[12,19] - 3 frac[12,44] - 3 frac[12,10] - 3 frac[12,43]
   - 3 frac[12,1] - 3 frac[12,31] - 3 frac[12,34] - 3 frac[12,56]
   - 3 frac[12,22] - 3 frac[12,55] - 3 frac[12,13] - 3 frac[12,46]
   - 3 frac[12,4] - 3 frac[12,37] - 3 frac[12,3] - 3 frac[12,25]
   - 3 frac[12,58] - 3 frac[12,16] - 3 frac[12,49] - 3 frac[12,15]
   - 3 frac[12,40] - 3 frac[12,6] - 3 frac[12,28] - 3 frac[12,39]
   - 3 frac[12,61] - 3 frac[12,27] - 3 frac[12,52] - 3 frac[12,18]
   - 3 frac[12,51] - 3 frac[12,9] - 3 frac[12,42] - 3 frac[12,0]
   - 3 frac[12,30] - 3 frac[12,33] - 3 frac[12,63] - 3 frac[12,21]
   - 3 frac[12,54] - 3 frac[12,12] - 3 frac[12,45] - 3 frac[12,36]
   - 3 frac[12,2] - 3 frac[12,24] - 3 frac[12,35] - 3 frac[12,57]
   - 3 frac[12,23] - 3 frac[12,48] - 3 frac[12,14] - 3 frac[12,47]
   - 3 frac[12,5] - 3 frac[12,38] - 3 frac[12,60] - 3 frac[12,26]
   + rows_tot_l1_P41 = 0
 mem_tot_l1_P42: - mem[13,32] - mem[13,58] - mem[13,20] - mem[13,51]
   - mem[13,13] - mem[13,44] - mem[13,6] - mem[13,37] - mem[13,63]
   - mem[13,56] - mem[13,18] - mem[13,49] - mem[13,42] - mem[13,4]
   - mem[13,30] - mem[13,35] - mem[13,61] - mem[13,23] - mem[13,54]
   - mem[13,16] - mem[13,47] - mem[13,9] - mem[13,40] - mem[13,2]
   - mem[13,28] - mem[13,33] - mem[13,59] - mem[13,21] - mem[13,52]
   - mem[13,14] - mem[13,45] - mem[13,7] - mem[13,38] - mem[13,0]
   - mem[13,26] - mem[13,57] - mem[13,19] - mem[13,50] - mem[13,12]
   - mem[13,43] - mem[13,5] - mem[13,31] - mem[13,36] - mem[13,62]
   - mem[13,24] - mem[13,55] - mem[13,17] - mem[13,48] - mem[13,10]
   - mem[13,41] - mem[13,3] - mem[13,29] - mem[13,34] - mem[13,60]
   - mem[13,22] - mem[13,53] - mem[13,15] - mem[13,46] - mem[13,8]
   - mem[13,39] - mem[13,1] - mem[13,27] + mem_tot_l1_P42 = 0
 rows_tot_l1_P42: - 3 frac[13,32] - 3 frac[13,58] - 3 frac[13,20]
   - 3 frac[13,51] - 3 frac[13,13] - 3 frac[13,44] - 3 frac[13,6]
   - 3 frac[13,37] - 3 frac[13,63] - 3 frac[13,56] - 3 frac[13,18]
   - 3 frac[13,49] - 3 frac[13,42] - 3 frac[13,4] - 3 frac[13,30]
   - 3 frac[13,35] - 3 frac[13,61] - 3 frac[13,23] - 3 frac[13,54]
   - 3 frac[13,16] - 3 frac[13,47] - 3 frac[13,9] - 3 frac[13,40]
   - 3 frac[13,2] - 3 frac[13,28] - 3 frac[13,33] - 3 frac[13,59]
   - 3 frac[13,21] - 3 frac[13,52] - 3 frac[13,14] - 3 frac[13,45]
   - 3 frac[13,7] - 3 frac[13,38] - 3 frac[13,0] - 3 frac[13,26]
   - 3 frac[13,57] - 3 frac[13,19] - 3 frac[13,50] - 3 frac[13,12]
   - 3 frac[13,43] - 3 frac[13,5] - 3 frac[13,31] - 3 frac[13,36]
   - 3 frac[13,62] - 3 frac[13,24] - 3 frac[13,55] - 3 frac[13,17]
   - 3 frac[13,48] - 3 frac[13,10] - 3 frac[13,41] - 3 frac[13,3]
   - 3 frac[13,29] - 3 frac[13,34] - 3 frac[13,60] - 3 frac[13,22]
   - 3 frac[13,53] - 3 frac[13,15] - 3 frac[13,46] - 3 frac[13,8]
   - 3 frac[13,39] - 3 frac[13,1] - 3 frac[13,27] + rows_tot_l1_P42 = 0
 mem_tot_l2_P4: - mem[14,31] - mem[14,1] - mem[14,43] - mem[14,8]
   - mem[14,50] - mem[14,19] - mem[14,53] - mem[14,26] - mem[14,60]
   - mem[14,38] - mem[14,29] - mem[14,7] - mem[14,41] - mem[14,14]
   - mem[14,48] - mem[14,17] - mem[14,59] - mem[14,24] - mem[14,2]
   - mem[14,36] - mem[14,5] - mem[14,47] - mem[14,12] - mem[14,54]
   - mem[14,23] - mem[14,57] - mem[14,35] - mem[14,30] - mem[14,0]
   - mem[14,42] - mem[14,45] - mem[14,18] - mem[14,52] - mem[14,21]
   - mem[14,63] - mem[14,33] - mem[14,28] - mem[14,6] - mem[14,40]
   - mem[14,9] - mem[14,51] - mem[14,16] - mem[14,58] - mem[14,27]
   - mem[14,61] - mem[14,39] - mem[14,4] - mem[14,46] - mem[14,15]
   - mem[14,49] - mem[14,22] - mem[14,56] - mem[14,34] - mem[14,3]
   - mem[14,37] - mem[14,10] - mem[14,44] - mem[14,13] - mem[14,55]
   - mem[14,20] - mem[14,62] - mem[14,32] + mem_tot_l2_P4 = 0
 rows_tot_l2_P4: - 3 frac[14,31] - 3 frac[14,1] - 3 frac[14,43]
   - 3 frac[14,8] - 3 frac[14,50] - 3 frac[14,19] - 3 frac[14,53]
   - 3 frac[14,26] - 3 frac[14,60] - 3 frac[14,38] - 3 frac[14,29]
   - 3 frac[14,7] - 3 frac[14,41] - 3 frac[14,14] - 3 frac[14,48]
   - 3 frac[14,17] - 3 frac[14,59] - 3 frac[14,24] - 3 frac[14,2]
   - 3 frac[14,36] - 3 frac[14,5] - 3 frac[14,47] - 3 frac[14,12]
   - 3 frac[14,54] - 3 frac[14,23] - 3 frac[14,57] - 3 frac[14,35]
   - 3 frac[14,30] - 3 frac[14,0] - 3 frac[14,42] - 3 frac[14,45]
   - 3 frac[14,18] - 3 frac[14,52] - 3 frac[14,21] - 3 frac[14,63]
   - 3 frac[14,33] - 3 frac[14,28] - 3 frac[14,6] - 3 frac[14,40]
   - 3 frac[14,9] - 3 frac[14,51] - 3 frac[14,16] - 3 frac[14,58]
   - 3 frac[14,27] - 3 frac[14,61] - 3 frac[14,39] - 3 frac[14,4]
   - 3 frac[14,46] - 3 frac[14,15] - 3 frac[14,49] - 3 frac[14,22]
   - 3 frac[14,56] - 3 frac[14,34] - 3 frac[14,3] - 3 frac[14,37]
   - 3 frac[14,10] - 3 frac[14,44] - 3 frac[14,13] - 3 frac[14,55]
   - 3 frac[14,20] - 3 frac[14,62] - 3 frac[14,32] + rows_tot_l2_P4 = 0
 ns_single_CPU1: 3.5 rows_tot_CPU1 - ns_single_CPU1 + pdt_m_rows_CPU1 = 0
 capacity_cores_CPU1: cores_sketch_CPU1 + cores_dpdk_CPU1 <= 7
 ns_sketch_CPU1: - ns_single_CPU1 + pdt_nsk_c_CPU1 = 0
 ns_dpdk_CPU1: 7.142857142857143 cores_dpdk_CPU1 - pdt_nsc_dpdk_CPU1
   = -21.42857142857143
 ns_single_CPU2: 3.5 rows_tot_CPU2 - ns_single_CPU2 + pdt_m_rows_CPU2 = 0
 capacity_cores_CPU2: cores_sketch_CPU2 + cores_dpdk_CPU2 <= 7
 ns_sketch_CPU2: - ns_single_CPU2 + pdt_nsk_c_CPU2 = 0
 ns_dpdk_CPU2: 7.142857142857143 cores_dpdk_CPU2 - pdt_nsc_dpdk_CPU2
   = -21.42857142857143
 ns_single_CPU3: 3.5 rows_tot_CPU3 - ns_single_CPU3 + pdt_m_rows_CPU3 = 0
 capacity_cores_CPU3: cores_sketch_CPU3 + cores_dpdk_CPU3 <= 7
 ns_sketch_CPU3: - ns_single_CPU3 + pdt_nsk_c_CPU3 = 0
 ns_dpdk_CPU3: 7.142857142857143 cores_dpdk_CPU3 - pdt_nsc_dpdk_CPU3
   = -21.42857142857143
 ns_single_CPU4: 3.5 rows_tot_CPU4 - ns_single_CPU4 + pdt_m_rows_CPU4 = 0
 capacity_cores_CPU4: cores_sketch_CPU4 + cores_dpdk_CPU4 <= 7
 ns_sketch_CPU4: - ns_single_CPU4 + pdt_nsk_c_CPU4 = 0
 ns_dpdk_CPU4: 7.142857142857143 cores_dpdk_CPU4 - pdt_nsc_dpdk_CPU4
   = -21.42857142857143
 ns_single_CPU5: 3.5 rows_tot_CPU5 - ns_single_CPU5 + pdt_m_rows_CPU5 = 0
 capacity_cores_CPU5: cores_sketch_CPU5 + cores_dpdk_CPU5 <= 7
 ns_sketch_CPU5: - ns_single_CPU5 + pdt_nsk_c_CPU5 = 0
 ns_dpdk_CPU5: 7.142857142857143 cores_dpdk_CPU5 - pdt_nsc_dpdk_CPU5
   = -21.42857142857143
 ns_single_CPU6: 3.5 rows_tot_CPU6 - ns_single_CPU6 + pdt_m_rows_CPU6 = 0
 capacity_cores_CPU6: cores_sketch_CPU6 + cores_dpdk_CPU6 <= 7
 ns_sketch_CPU6: - ns_single_CPU6 + pdt_nsk_c_CPU6 = 0
 ns_dpdk_CPU6: 7.142857142857143 cores_dpdk_CPU6 - pdt_nsc_dpdk_CPU6
   = -21.42857142857143
 ns_single_CPU7: 3.5 rows_tot_CPU7 - ns_single_CPU7 + pdt_m_rows_CPU7 = 0
 capacity_cores_CPU7: cores_sketch_CPU7 + cores_dpdk_CPU7 <= 7
 ns_sketch_CPU7: - ns_single_CPU7 + pdt_nsk_c_CPU7 = 0
 ns_dpdk_CPU7: 7.142857142857143 cores_dpdk_CPU7 - pdt_nsc_dpdk_CPU7
   = -21.42857142857143
 ns_single_CPU8: 3.5 rows_tot_CPU8 - ns_single_CPU8 + pdt_m_rows_CPU8 = 0
 capacity_cores_CPU8: cores_sketch_CPU8 + cores_dpdk_CPU8 <= 7
 ns_sketch_CPU8: - ns_single_CPU8 + pdt_nsk_c_CPU8 = 0
 ns_dpdk_CPU8: 7.142857142857143 cores_dpdk_CPU8 - pdt_nsc_dpdk_CPU8
   = -21.42857142857143
 ns_tor_P41: ns_tor_P41 = 6.756756756756757
 ns_tor_P42: ns_tor_P42 = 6.756756756756757
 ns_tor_P43: ns_tor_P43 = 6.756756756756757
 ns_tor_P44: ns_tor_P44 = 6.756756756756757
 ns_l1_P41: ns_l1_P41 = 6.756756756756757
 ns_l1_P42: ns_l1_P42 = 6.756756756756757
 ns_l2_P4: ns_l2_P4 = 6.756756756756757
 res_acc: 1.220703125e-04 mem_tot_CPU1 + 1.220703125e-04 mem_tot_CPU2
   + 1.220703125e-04 mem_tot_CPU3 + 1.220703125e-04 mem_tot_CPU4
   + 1.220703125e-04 mem_tot_CPU5 + 1.220703125e-04 mem_tot_CPU6
   + 1.220703125e-04 mem_tot_CPU7 + 1.220703125e-04 mem_tot_CPU8
   + 0.0208333333333333 mem_tot_tor_P41 + 0.25 rows_tot_tor_P41
   + 0.0208333333333333 mem_tot_tor_P42 + 0.25 rows_tot_tor_P42
   + 0.0208333333333333 mem_tot_tor_P43 + 0.25 rows_tot_tor_P43
   + 0.0208333333333333 mem_tot_tor_P44 + 0.25 rows_tot_tor_P44
   + 0.0208333333333333 mem_tot_l1_P41 + 0.25 rows_tot_l1_P41
   + 0.0208333333333333 mem_tot_l1_P42 + 0.25 rows_tot_l1_P42
   + 0.0208333333333333 mem_tot_l2_P4 + 0.25 rows_tot_l2_P4
   + 10 cores_sketch_CPU1 + 10 cores_dpdk_CPU1 + 10 cores_sketch_CPU2
   + 10 cores_dpdk_CPU2 + 10 cores_sketch_CPU3 + 10 cores_dpdk_CPU3
   + 10 cores_sketch_CPU4 + 10 cores_dpdk_CPU4 + 10 cores_sketch_CPU5
   + 10 cores_dpdk_CPU5 + 10 cores_sketch_CPU6 + 10 cores_dpdk_CPU6
   + 10 cores_sketch_CPU7 + 10 cores_dpdk_CPU7 + 10 cores_sketch_CPU8
   + 10 cores_dpdk_CPU8 - res = 0
 pdt_m_rows_CPU1: pdt_m_rows_CPU1 + [ - rows_tot_CPU1 * m_access_time_CPU1
   ] = 0
 pdt_nsk_c_CPU1: pdt_nsk_c_CPU1 + [ - cores_sketch_CPU1 * ns_sketch_CPU1 ]
   = 0
 pdt_nsc_dpdk_CPU1: pdt_nsc_dpdk_CPU1 + [ - cores_dpdk_CPU1 * ns_dpdk_CPU1
   ] = 0
 pdt_m_rows_CPU2: pdt_m_rows_CPU2 + [ - rows_tot_CPU2 * m_access_time_CPU2
   ] = 0
 pdt_nsk_c_CPU2: pdt_nsk_c_CPU2 + [ - cores_sketch_CPU2 * ns_sketch_CPU2 ]
   = 0
 pdt_nsc_dpdk_CPU2: pdt_nsc_dpdk_CPU2 + [ - cores_dpdk_CPU2 * ns_dpdk_CPU2
   ] = 0
 pdt_m_rows_CPU3: pdt_m_rows_CPU3 + [ - rows_tot_CPU3 * m_access_time_CPU3
   ] = 0
 pdt_nsk_c_CPU3: pdt_nsk_c_CPU3 + [ - cores_sketch_CPU3 * ns_sketch_CPU3 ]
   = 0
 pdt_nsc_dpdk_CPU3: pdt_nsc_dpdk_CPU3 + [ - cores_dpdk_CPU3 * ns_dpdk_CPU3
   ] = 0
 pdt_m_rows_CPU4: pdt_m_rows_CPU4 + [ - rows_tot_CPU4 * m_access_time_CPU4
   ] = 0
 pdt_nsk_c_CPU4: pdt_nsk_c_CPU4 + [ - cores_sketch_CPU4 * ns_sketch_CPU4 ]
   = 0
 pdt_nsc_dpdk_CPU4: pdt_nsc_dpdk_CPU4 + [ - cores_dpdk_CPU4 * ns_dpdk_CPU4
   ] = 0
 pdt_m_rows_CPU5: pdt_m_rows_CPU5 + [ - rows_tot_CPU5 * m_access_time_CPU5
   ] = 0
 pdt_nsk_c_CPU5: pdt_nsk_c_CPU5 + [ - cores_sketch_CPU5 * ns_sketch_CPU5 ]
   = 0
 pdt_nsc_dpdk_CPU5: pdt_nsc_dpdk_CPU5 + [ - cores_dpdk_CPU5 * ns_dpdk_CPU5
   ] = 0
 pdt_m_rows_CPU6: pdt_m_rows_CPU6 + [ - rows_tot_CPU6 * m_access_time_CPU6
   ] = 0
 pdt_nsk_c_CPU6: pdt_nsk_c_CPU6 + [ - cores_sketch_CPU6 * ns_sketch_CPU6 ]
   = 0
 pdt_nsc_dpdk_CPU6: pdt_nsc_dpdk_CPU6 + [ - cores_dpdk_CPU6 * ns_dpdk_CPU6
   ] = 0
 pdt_m_rows_CPU7: pdt_m_rows_CPU7 + [ - rows_tot_CPU7 * m_access_time_CPU7
   ] = 0
 pdt_nsk_c_CPU7: pdt_nsk_c_CPU7 + [ - cores_sketch_CPU7 * ns_sketch_CPU7 ]
   = 0
 pdt_nsc_dpdk_CPU7: pdt_nsc_dpdk_CPU7 + [ - cores_dpdk_CPU7 * ns_dpdk_CPU7
   ] = 0
 pdt_m_rows_CPU8: pdt_m_rows_CPU8 + [ - rows_tot_CPU8 * m_access_time_CPU8
   ] = 0
 pdt_nsk_c_CPU8: pdt_nsk_c_CPU8 + [ - cores_sketch_CPU8 * ns_sketch_CPU8 ]
   = 0
 pdt_nsc_dpdk_CPU8: pdt_nsc_dpdk_CPU8 + [ - cores_dpdk_CPU8 * ns_dpdk_CPU8
   ] = 0
Bounds
 mem_tot_CPU1 <= 32768
 mem_tot_CPU2 <= 32768
 mem_tot_CPU3 <= 32768
 mem_tot_CPU4 <= 32768
 mem_tot_CPU5 <= 32768
 mem_tot_CPU6 <= 32768
 mem_tot_CPU7 <= 32768
 mem_tot_CPU8 <= 32768
 mem_tot_tor_P41 <= 576
 mem_tot_tor_P42 <= 576
 mem_tot_tor_P43 <= 576
 mem_tot_tor_P44 <= 576
 mem_tot_l1_P41 <= 576
 mem_tot_l1_P42 <= 576
 mem_tot_l2_P4 <= 576
 cores_sketch_CPU1 <= 7
 1 <= cores_dpdk_CPU1 <= 7
 cores_sketch_CPU2 <= 7
 1 <= cores_dpdk_CPU2 <= 7
 cores_sketch_CPU3 <= 7
 1 <= cores_dpdk_CPU3 <= 7
 cores_sketch_CPU4 <= 7
 1 <= cores_dpdk_CPU4 <= 7
 cores_sketch_CPU5 <= 7
 1 <= cores_dpdk_CPU5 <= 7
 cores_sketch_CPU6 <= 7
 1 <= cores_dpdk_CPU6 <= 7
 cores_sketch_CPU7 <= 7
 1 <= cores_dpdk_CPU7 <= 7
 cores_sketch_CPU8 <= 7
 1 <= cores_dpdk_CPU8 <= 7
Binaries
 frac[7,3] frac[5,31] frac[3,35] frac[4,36] frac[8,63] frac[5,57]
 frac[9,62] frac[6,54] frac[10,61] frac[8,5] frac[7,55] frac[13,32]
 frac[11,48] frac[9,0] frac[14,31] frac[12,59] frac[10,7] frac[13,58]
 frac[11,22] frac[0,17] frac[14,1] frac[12,17] frac[1,28] frac[13,20]
 frac[0,55] frac[14,43] frac[1,54] frac[4,5] frac[2,53] frac[5,24]
 frac[3,40] frac[1,40] frac[4,35] frac[2,47] frac[8,38] frac[7,22]
 frac[5,34] frac[9,39] frac[10,36] frac[8,12] frac[11,41] frac[9,9]
 frac[12,50] frac[10,14] frac[8,18] frac[13,51] frac[11,15] frac[9,19]
 frac[14,8] frac[12,8] frac[1,21] frac[13,13] frac[2,18] frac[14,50]
 frac[4,12] frac[2,12] frac[0,36] frac[5,1] frac[1,33] frac[4,42]
 frac[8,45] frac[3,55] frac[9,40] frac[6,32] frac[10,47] frac[8,51]
 frac[5,53] frac[11,46] frac[9,50] frac[6,58] frac[12,41] frac[10,9]
 frac[8,25] frac[13,44] frac[11,4] frac[9,20] frac[14,19] frac[12,7]
 frac[10,19] frac[13,6] frac[11,26] frac[0,5] frac[14,53] frac[12,29]
 frac[1,0] frac[4,11] frac[5,10] frac[3,22] frac[1,58] frac[6,1] frac[2,33]
 frac[7,4] frac[5,20] frac[4,63] frac[8,58] frac[5,62] frac[11,39]
 frac[9,59] frac[6,61] frac[12,32] frac[10,48] frac[8,0] frac[13,37]
 frac[11,61] frac[9,29] frac[14,26] frac[12,62] frac[13,63] frac[11,19]
 frac[0,28] frac[14,60] frac[12,20] frac[1,25] frac[0,50] frac[14,38]
 frac[3,31] frac[1,51] frac[0,40] frac[5,29] frac[3,37] frac[4,38]
 frac[8,33] frac[7,27] frac[5,39] frac[9,60] frac[10,59] frac[8,7]
 frac[7,49] frac[11,50] frac[9,6] frac[14,29] frac[12,53] frac[10,5]
 frac[13,56] frac[11,8] frac[14,7] frac[12,19] frac[1,18] frac[13,18]
 frac[2,25] frac[14,41] frac[1,52] frac[2,51] frac[0,47] frac[3,42]
 frac[1,46] frac[6,21] frac[4,45] frac[2,45] frac[8,40] frac[7,16]
 frac[5,32] frac[3,48] frac[9,37] frac[6,47] frac[10,34] frac[8,14]
 frac[7,46] frac[11,43] frac[9,15] frac[12,44] frac[10,12] frac[13,49]
 frac[11,1] frac[9,17] frac[14,14] frac[12,10] frac[10,22] frac[2,16]
 frac[0,0] frac[14,48] frac[3,13] frac[4,14] frac[0,38] frac[3,19]
 frac[1,39] frac[6,12] frac[2,36] frac[8,47] frac[7,9] frac[5,41]
 frac[3,57] frac[9,46] frac[4,50] frac[10,45] frac[8,53] frac[7,39]
 frac[5,51] frac[11,32] frac[9,48] frac[6,56] frac[12,43] frac[10,55]
 frac[8,27] frac[7,61] frac[13,42] frac[11,6] frac[9,26] frac[14,17]
 frac[12,1] frac[10,17] frac[13,4] frac[11,28] frac[0,7] frac[14,59]
 frac[12,31] frac[1,6] frac[13,30] frac[2,5] frac[0,61] frac[3,24]
 frac[1,56] frac[6,7] frac[4,19] frac[7,6] frac[5,18] frac[3,62] frac[8,60]
 frac[7,28] frac[5,60] frac[9,57] frac[12,34] frac[10,62] frac[8,2]
 frac[7,58] frac[13,35] frac[11,63] frac[14,24] frac[12,56] frac[10,24]
 frac[13,61] frac[11,21] frac[0,30] frac[14,2] frac[12,22] frac[1,31]
 frac[13,23] frac[0,52] frac[14,36] frac[3,1] frac[2,54] frac[5,27]
 frac[3,39] frac[1,43] frac[6,16] frac[7,21] frac[5,37] frac[9,34]
 frac[6,42] frac[10,57] frac[8,9] frac[7,51] frac[11,52] frac[9,4]
 frac[12,55] frac[10,3] frac[13,54] frac[11,10] frac[0,21] frac[14,5]
 frac[12,13] frac[1,16] frac[13,16] frac[14,47] frac[3,6] frac[4,1]
 frac[2,49] frac[0,33] frac[3,44] frac[1,44] frac[4,47] frac[2,43]
 frac[8,42] frac[7,18] frac[5,46] frac[9,43] frac[6,45] frac[4,53]
 frac[10,32] frac[8,48] frac[7,40] frac[11,45] frac[9,13] frac[12,46]
 frac[10,10] frac[13,47] frac[11,3] frac[9,23] frac[14,12] frac[12,4]
 frac[10,20] frac[13,9] frac[14,54] frac[3,15] frac[1,3] frac[2,8]
 frac[5,13] frac[3,21] frac[1,37] frac[6,2] frac[4,22] frac[2,34]
 frac[5,23] frac[3,59] frac[9,44] frac[6,36] frac[4,60] frac[10,43]
 frac[8,55] frac[11,34] frac[9,54] frac[12,37] frac[10,53] frac[8,29]
 frac[7,63] frac[13,40] frac[11,56] frac[9,24] frac[14,23] frac[12,3]
 frac[10,31] frac[13,2] frac[11,30] frac[0,25] frac[14,57] frac[12,25]
 frac[13,28] frac[0,63] frac[14,35] frac[3,26] frac[1,62] frac[6,5]
 frac[4,29] frac[7,0] frac[5,16] frac[6,31] frac[4,59] frac[8,62]
 frac[5,58] frac[10,60] frac[7,52] frac[13,33] frac[11,49] frac[9,1]
 frac[14,30] frac[12,58] frac[10,6] frac[13,59] frac[11,23] frac[0,16]
 frac[14,0] frac[12,16] frac[1,29] frac[13,21] frac[2,26] frac[0,54]
 frac[14,42] frac[2,52] frac[0,44] frac[6,22] frac[4,34] frac[2,46]
 frac[8,37] frac[7,23] frac[5,35] frac[6,40] frac[10,39] frac[7,45]
 frac[11,54] frac[9,10] frac[12,49] frac[10,1] frac[8,17] frac[13,52]
 frac[11,12] frac[12,15] frac[13,14] frac[2,21] frac[0,13] frac[14,45]
 frac[3,8] frac[1,8] frac[4,3] frac[3,46] frac[8,44] frac[7,12] frac[5,44]
 frac[3,52] frac[6,35] frac[4,55] frac[10,46] frac[8,50] frac[7,42]
 frac[5,54] frac[11,47] frac[9,51] frac[12,40] frac[8,24] frac[13,45]
 frac[11,5] frac[9,21] frac[14,18] frac[12,6] frac[10,18] frac[13,7]
 frac[11,27] frac[14,52] frac[12,28] frac[1,1] frac[2,6] frac[0,58]
 frac[5,11] frac[3,23] frac[6,0] frac[7,5] frac[5,21] frac[3,61] frac[6,26]
 frac[4,62] frac[10,41] frac[8,57] frac[7,35] frac[5,63] frac[11,36]
 frac[9,52] frac[6,60] frac[12,39] frac[10,51] frac[8,31] frac[7,57]
 frac[13,38] frac[11,58] frac[9,30] frac[14,21] frac[12,61] frac[10,29]
 frac[13,0] frac[11,16] frac[0,27] frac[14,63] frac[12,27] frac[1,26]
 frac[13,26] frac[2,1] frac[0,49] frac[14,33] frac[3,28] frac[1,60]
 frac[3,34] frac[6,29] frac[4,37] frac[8,32] frac[7,24] frac[5,56]
 frac[9,61] frac[6,55] frac[10,58] frac[8,6] frac[11,51] frac[9,7]
 frac[14,28] frac[12,52] frac[13,57] frac[11,9] frac[0,18] frac[14,6]
 frac[12,18] frac[1,19] frac[13,19] frac[2,24] frac[14,40] frac[3,5]
 frac[1,53] frac[4,6] frac[2,50] frac[0,46] frac[5,7] frac[1,47] frac[4,44]
 frac[2,44] frac[8,39] frac[7,17] frac[5,33] frac[3,49] frac[10,37]
 frac[8,13] frac[11,40] frac[9,8] frac[12,51] frac[8,19] frac[13,50]
 frac[11,14] frac[9,18] frac[14,9] frac[12,9] frac[13,12] frac[2,19]
 frac[0,15] frac[14,51] frac[3,10] frac[1,14] frac[4,13] frac[2,13]
 frac[0,37] frac[5,0] frac[1,32] frac[6,15] frac[4,43] frac[8,46]
 frac[7,14] frac[5,42] frac[3,54] frac[9,47] frac[6,33] frac[10,44]
 frac[8,52] frac[11,33] frac[9,49] frac[6,59] frac[12,42] frac[10,54]
 frac[8,26] frac[13,43] frac[11,7] frac[14,16] frac[12,0] frac[10,16]
 frac[13,5] frac[11,29] frac[0,6] frac[14,58] frac[12,30] frac[13,31]
 frac[2,4] frac[5,9] frac[3,25] frac[4,18] frac[2,62] frac[7,7] frac[5,19]
 frac[6,24] frac[7,29] frac[11,38] frac[9,58] frac[12,33] frac[8,1]
 frac[7,59] frac[13,36] frac[11,60] frac[9,28] frac[14,27] frac[12,63]
 frac[10,27] frac[13,62] frac[11,18] frac[14,61] frac[12,21] frac[1,24]
 frac[13,24] frac[2,31] frac[14,39] frac[3,30] frac[1,50] frac[2,57]
 frac[0,41] frac[5,28] frac[3,36] frac[4,39] frac[7,26] frac[5,38]
 frac[9,35] frac[6,53] frac[10,56] frac[8,8] frac[7,48] frac[11,53]
 frac[9,5] frac[12,54] frac[10,2] frac[13,55] frac[14,4] frac[12,12]
 frac[1,17] frac[13,17] frac[2,22] frac[0,10] frac[14,46] frac[3,7]
 frac[5,5] frac[3,45] frac[1,45] frac[4,46] frac[2,42] frac[8,41]
 frac[7,19] frac[5,47] frac[3,51] frac[9,36] frac[6,44] frac[4,52]
 frac[10,35] frac[8,15] frac[7,41] frac[11,42] frac[9,14] frac[12,45]
 frac[10,13] frac[8,21] frac[13,48] frac[11,0] frac[9,16] frac[14,15]
 frac[10,23] frac[13,10] frac[2,17] frac[0,1] frac[14,49] frac[3,12]
 frac[1,12] frac[0,39] frac[6,13] frac[4,21] frac[2,37] frac[5,40]
 frac[3,56] frac[9,45] frac[6,39] frac[4,51] frac[10,42] frac[8,54]
 frac[7,38] frac[11,35] frac[9,55] frac[6,57] frac[12,36] frac[10,52]
 frac[8,28] frac[7,60] frac[13,41] frac[11,57] frac[9,25] frac[14,22]
 frac[12,2] frac[10,30] frac[13,3] frac[11,31] frac[0,24] frac[14,56]
 frac[12,24] frac[1,5] frac[13,29] frac[2,2] frac[14,34] frac[1,63]
 frac[2,60] frac[7,1] frac[5,17] frac[6,30] frac[4,58] frac[8,61]
 frac[7,31] frac[5,59] frac[9,56] frac[6,48] frac[12,35] frac[10,63]
 frac[8,3] frac[13,34] frac[11,62] frac[9,2] frac[12,57] frac[13,60]
 frac[14,3] frac[12,23] frac[1,30] frac[13,22] frac[2,29] frac[0,53]
 frac[14,37] frac[3,0] frac[1,48] frac[4,27] frac[2,55] frac[0,43]
 frac[1,42] frac[4,33] frac[8,36] frac[5,36] frac[9,33] frac[6,43]
 frac[10,38] frac[8,10] frac[7,50] frac[11,55] frac[12,48] frac[10,0]
 frac[8,16] frac[13,53] frac[11,13] frac[14,10] frac[12,14] frac[1,23]
 frac[13,15] frac[2,20] frac[0,12] frac[14,44] frac[3,9] frac[1,9]
 frac[4,2] frac[2,14] frac[3,47] frac[6,8] frac[4,40] frac[8,43] frac[7,13]
 frac[5,45] frac[3,53] frac[9,42] frac[6,34] frac[10,33] frac[8,49]
 frac[7,43] frac[5,55] frac[11,44] frac[9,12] frac[12,47] frac[10,11]
 frac[8,23] frac[13,46] frac[11,2] frac[9,22] frac[14,13] frac[12,5]
 frac[10,21] frac[13,8] frac[11,24] frac[0,3] frac[14,55] frac[3,14]
 frac[1,2] frac[2,9] frac[0,57] frac[5,12] frac[3,20] frac[6,3] frac[4,23]
 frac[2,35] frac[7,10] frac[3,58] frac[6,37] frac[4,61] frac[10,40]
 frac[8,56] frac[5,48] frac[11,37] frac[9,53] frac[6,63] frac[12,38]
 frac[10,50] frac[8,30] frac[7,62] frac[13,39] frac[11,59] frac[9,31]
 frac[14,20] frac[12,60] frac[10,28] frac[13,1] frac[11,17] frac[0,26]
 frac[14,62] frac[12,26] frac[1,27] frac[13,27] frac[0,48] frac[14,32]
 frac[3,29] frac[4,30] frac[2,58]
Generals
 cores_sketch_CPU1 cores_dpdk_CPU1 cores_sketch_CPU2 cores_dpdk_CPU2
 cores_sketch_CPU3 cores_dpdk_CPU3 cores_sketch_CPU4 cores_dpdk_CPU4
 cores_sketch_CPU5 cores_dpdk_CPU5 cores_sketch_CPU6 cores_dpdk_CPU6
 cores_sketch_CPU7 cores_dpdk_CPU7 cores_sketch_CPU8 cores_dpdk_CPU8
General Constraints
 mem_access_time_CPU1: m_access_time_CPU1 = PWL ( mem_tot_CPU1 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU1: ns_CPU1 = MAX ( ns_dpdk_CPU1 , ns_sketch_CPU1 )
 mem_access_time_CPU2: m_access_time_CPU2 = PWL ( mem_tot_CPU2 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU2: ns_CPU2 = MAX ( ns_dpdk_CPU2 , ns_sketch_CPU2 )
 mem_access_time_CPU3: m_access_time_CPU3 = PWL ( mem_tot_CPU3 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU3: ns_CPU3 = MAX ( ns_dpdk_CPU3 , ns_sketch_CPU3 )
 mem_access_time_CPU4: m_access_time_CPU4 = PWL ( mem_tot_CPU4 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU4: ns_CPU4 = MAX ( ns_dpdk_CPU4 , ns_sketch_CPU4 )
 mem_access_time_CPU5: m_access_time_CPU5 = PWL ( mem_tot_CPU5 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU5: ns_CPU5 = MAX ( ns_dpdk_CPU5 , ns_sketch_CPU5 )
 mem_access_time_CPU6: m_access_time_CPU6 = PWL ( mem_tot_CPU6 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU6: ns_CPU6 = MAX ( ns_dpdk_CPU6 , ns_sketch_CPU6 )
 mem_access_time_CPU7: m_access_time_CPU7 = PWL ( mem_tot_CPU7 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU7: ns_CPU7 = MAX ( ns_dpdk_CPU7 , ns_sketch_CPU7 )
 mem_access_time_CPU8: m_access_time_CPU8 = PWL ( mem_tot_CPU8 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU8: ns_CPU8 = MAX ( ns_dpdk_CPU8 , ns_sketch_CPU8 )
 ns_overall: ns = MAX ( ns_CPU1 , ns_CPU2 , ns_CPU3 , ns_CPU4 , ns_CPU5 ,
   ns_CPU6 , ns_CPU7 , ns_CPU8 , ns_tor_P41 , ns_tor_P42 , ns_tor_P43 ,
   ns_tor_P44 , ns_l1_P41 , ns_l1_P42 , ns_l2_P4 )
End

\ Model netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 m_access_time_cpu1 + 0 ns_sketch_cpu1 + 0 ns_dpdk_cpu1 + 0 ns_cpu1
   + 0 m_access_time_cpu2 + 0 ns_sketch_cpu2 + 0 ns_dpdk_cpu2 + 0 ns_cpu2
   + 0 m_access_time_cpu3 + 0 ns_sketch_cpu3 + 0 ns_dpdk_cpu3 + 0 ns_cpu3
   + 0 m_access_time_cpu4 + 0 ns_sketch_cpu4 + 0 ns_dpdk_cpu4 + 0 ns_cpu4
   + 0 m_access_time_cpu5 + 0 ns_sketch_cpu5 + 0 ns_dpdk_cpu5 + 0 ns_cpu5
   + 0 m_access_time_cpu6 + 0 ns_sketch_cpu6 + 0 ns_dpdk_cpu6 + 0 ns_cpu6
   + 0 m_access_time_cpu7 + 0 ns_sketch_cpu7 + 0 ns_dpdk_cpu7 + 0 ns_cpu7
   + 0 m_access_time_cpu8 + 0 ns_sketch_cpu8 + 0 ns_dpdk_cpu8 + 0 ns_cpu8
   + ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0
   res
Subject To
 cov_0_26: frac[11,26] + frac[14,26] + frac[9,26] + frac[3,26] + frac[6,26]
   + frac[13,26] + frac[12,26] >= 1
 cov_1_53: frac[5,53] + frac[10,53] + frac[11,53] + frac[7,53]
   + frac[13,53] >= 0.9
 cov_2_39: frac[9,39] + frac[11,39] + frac[7,39] + frac[3,39] + frac[12,39]
   + frac[14,39] + frac[13,39] >= 1
 cov_3_47: frac[10,47] + frac[8,47] + frac[14,47] + frac[13,47]
   + frac[1,47] + frac[5,47] + frac[12,47] >= 0.7
 cov_4_17: frac[7,17] + frac[6,17] + frac[11,17] >= 0.9
 cov_5_57: frac[3,57] + frac[9,57] + frac[8,57] + frac[12,57] + frac[0,57]
   >= 0.9
 cov_6_36: frac[0,36] + frac[14,36] + frac[11,36] + frac[7,36]
   + frac[13,36] + frac[12,36] + frac[8,36] >= 0.7
 cov_7_3: frac[11,3] + frac[1,3] + frac[12,3] + frac[13,3] + frac[8,3]
   + frac[14,3] + frac[6,3] >= 0.7
 cov_8_36: frac[2,36] + frac[9,36] + frac[12,36] + frac[8,36] + frac[1,36]
   >= 1
 cov_9_57: frac[5,57] + frac[10,57] + frac[13,57] + frac[6,57]
   + frac[11,57] >= 0.7
 cov_10_5: frac[8,5] + frac[0,5] + frac[2,5] + frac[9,5] + frac[12,5]
   >= 0.9
 cov_11_59: frac[0,59] + frac[1,59] + frac[8,59] >= 1
 cov_12_27: frac[8,27] + frac[12,27] + frac[14,27] + frac[10,27]
   + frac[4,27] + frac[1,27] + frac[13,27] >= 1
 cov_13_37: frac[7,37] + frac[13,37] + frac[4,37] + frac[10,37]
   + frac[11,37] >= 0.9
 cov_14_35: frac[13,35] + frac[8,35] + frac[14,35] + frac[0,35]
   + frac[7,35] + frac[11,35] + frac[12,35] >= 0.9
 cov_15_41: frac[6,41] + frac[11,41] + frac[12,41] + frac[14,41]
   + frac[0,41] + frac[8,41] + frac[13,41] >= 1
 cov_16_27: frac[2,27] + frac[5,27] + frac[12,27] + frac[9,27]
   + frac[14,27] + frac[10,27] + frac[13,27] >= 1
 cov_17_56: frac[13,56] + frac[6,56] + frac[11,56] + frac[4,56]
   + frac[10,56] >= 0.8
 cov_18_50: frac[12,50] + frac[14,50] + frac[11,50] + frac[8,50]
   + frac[13,50] + frac[6,50] + frac[1,50] >= 0.9
 cov_19_17: frac[0,17] + frac[12,17] + frac[14,17] + frac[8,17]
   + frac[13,17] + frac[6,17] + frac[11,17] >= 0.7
 cov_20_37: frac[13,37] + frac[1,37] + frac[12,37] + frac[8,37]
   + frac[14,37] + frac[6,37] + frac[11,37] >= 0.8
 cov_21_29: frac[4,29] + frac[10,29] + frac[6,29] + frac[11,29]
   + frac[13,29] >= 0.8
 cov_22_61: frac[2,61] + frac[12,61] + frac[9,61] + frac[8,61] + frac[1,61]
   >= 0.8
 cov_23_59: frac[12,59] + frac[0,59] + frac[9,59] + frac[2,59] + frac[8,59]
   >= 1
 cov_24_52: frac[6,52] + frac[11,52] + frac[13,52] + frac[5,52]
   + frac[10,52] >= 0.7
 cov_25_30: frac[0,30] + frac[1,30] + frac[8,30] >= 0.7
 cov_26_31: frac[14,31] + frac[12,31] + frac[13,31] + frac[2,31]
   + frac[11,31] + frac[7,31] + frac[9,31] >= 1
 cov_27_43: frac[14,43] + frac[11,43] + frac[12,43] + frac[2,43]
   + frac[9,43] + frac[13,43] + frac[6,43] >= 1
 cov_28_28: frac[6,28] + frac[11,28] + frac[13,28] + frac[5,28]
   + frac[10,28] >= 0.8
 cov_29_25: frac[13,25] + frac[2,25] + frac[6,25] + frac[11,25]
   + frac[12,25] + frac[9,25] + frac[14,25] >= 1
 cov_30_58: frac[9,58] + frac[3,58] + frac[2,58] >= 0.7
 cov_31_21: frac[6,21] + frac[11,21] + frac[13,21] + frac[5,21]
   + frac[10,21] >= 0.8
 cov_32_2: frac[6,2] + frac[7,2] + frac[11,2] >= 0.9
 cov_33_31: frac[1,31] + frac[8,31] + frac[0,31] >= 0.7
 cov_34_15: frac[9,15] + frac[3,15] + frac[12,15] + frac[10,15]
   + frac[14,15] + frac[4,15] + frac[13,15] >= 0.8
 cov_35_59: frac[10,59] + frac[13,59] + frac[7,59] + frac[5,59]
   + frac[11,59] >= 0.9
 cov_36_36: frac[14,36] + frac[6,36] + frac[11,36] + frac[13,36]
   + frac[3,36] + frac[9,36] + frac[12,36] >= 0.7
 cov_37_56: frac[12,56] + frac[0,56] + frac[3,56] + frac[9,56] + frac[8,56]
   >= 0.7
 cov_38_27: frac[2,27] + frac[5,27] + frac[12,27] + frac[9,27]
   + frac[14,27] + frac[10,27] + frac[13,27] >= 0.7
 cov_39_3: frac[9,3] + frac[10,3] + frac[12,3] + frac[2,3] + frac[4,3]
   + frac[13,3] + frac[14,3] >= 0.8
 cov_40_45: frac[6,45] + frac[11,45] + frac[14,45] + frac[13,45]
   + frac[3,45] + frac[12,45] + frac[9,45] >= 0.8
 cov_41_15: frac[1,15] + frac[12,15] + frac[10,15] + frac[8,15]
   + frac[14,15] + frac[4,15] + frac[13,15] >= 0.8
 cov_42_24: frac[4,24] + frac[3,24] + frac[14,24] + frac[10,24]
   + frac[9,24] + frac[13,24] + frac[12,24] >= 0.7
 cov_43_46: frac[9,46] + frac[12,46] + frac[3,46] + frac[10,46]
   + frac[14,46] + frac[4,46] + frac[13,46] >= 1
 cov_44_22: frac[11,22] + frac[12,22] + frac[6,22] + frac[2,22]
   + frac[14,22] + frac[13,22] + frac[9,22] >= 0.7
 cov_45_5: frac[4,5] + frac[10,5] + frac[5,5] >= 0.9
 cov_46_2: frac[6,2] + frac[13,2] + frac[5,2] + frac[10,2] + frac[11,2]
   >= 0.8
 cov_47_18: frac[6,18] + frac[7,18] + frac[11,18] >= 0.9
 cov_48_14: frac[10,14] + frac[4,14] + frac[5,14] >= 1
 cov_49_11: frac[13,11] + frac[7,11] + frac[5,11] + frac[11,11]
   + frac[10,11] >= 0.7
 cov_50_9: frac[10,9] + frac[8,9] + frac[13,9] + frac[14,9] + frac[12,9]
   + frac[1,9] + frac[4,9] >= 1
 cov_51_2: frac[14,2] + frac[13,2] + frac[5,2] + frac[10,2] + frac[12,2]
   + frac[2,2] + frac[9,2] >= 0.7
 cov_52_52: frac[1,52] + frac[13,52] + frac[14,52] + frac[12,52]
   + frac[8,52] + frac[5,52] + frac[10,52] >= 0.8
 cov_53_38: frac[8,38] + frac[14,38] + frac[13,38] + frac[11,38]
   + frac[1,38] + frac[7,38] + frac[12,38] >= 0.7
 cov_54_46: frac[5,46] + frac[10,46] + frac[4,46] >= 0.7
 cov_55_60: frac[14,60] + frac[8,60] + frac[6,60] + frac[0,60]
   + frac[11,60] + frac[13,60] + frac[12,60] >= 1
 cov_56_37: frac[7,37] + frac[6,37] + frac[11,37] >= 0.9
 cov_57_14: frac[10,14] + frac[14,14] + frac[13,14] + frac[9,14]
   + frac[5,14] + frac[12,14] + frac[3,14] >= 0.8
 cov_58_46: frac[11,46] + frac[12,46] + frac[0,46] + frac[6,46]
   + frac[8,46] + frac[14,46] + frac[13,46] >= 0.8
 cov_59_41: frac[6,41] + frac[11,41] + frac[5,41] + frac[10,41]
   + frac[13,41] >= 0.9
 cov_60_34: frac[10,34] + frac[12,34] + frac[9,34] + frac[4,34]
   + frac[3,34] + frac[14,34] + frac[13,34] >= 0.9
 cov_61_38: frac[8,38] + frac[14,38] + frac[13,38] + frac[5,38]
   + frac[1,38] + frac[10,38] + frac[12,38] >= 0.7
 cov_62_10: frac[12,10] + frac[11,10] + frac[0,10] + frac[6,10]
   + frac[13,10] + frac[8,10] + frac[14,10] >= 1
 cov_63_5: frac[4,5] + frac[10,5] + frac[6,5] + frac[11,5] + frac[13,5]
   >= 0.9
 cov_64_53: frac[14,53] + frac[12,53] + frac[11,53] + frac[7,53]
   + frac[13,53] + frac[3,53] + frac[9,53] >= 0.9
 cov_65_5: frac[4,5] + frac[10,5] + frac[6,5] + frac[11,5] + frac[13,5]
   >= 0.8
 cov_66_52: frac[6,52] + frac[11,52] + frac[13,52] + frac[5,52]
   + frac[10,52] >= 1
 cov_67_28: frac[0,28] + frac[13,28] + frac[12,28] + frac[14,28]
   + frac[8,28] + frac[4,28] + frac[10,28] >= 0.9
 cov_68_27: frac[6,27] + frac[8,27] + frac[11,27] + frac[0,27]
   + frac[12,27] + frac[14,27] + frac[13,27] >= 0.9
 cov_69_23: frac[6,23] + frac[13,23] + frac[2,23] + frac[9,23]
   + frac[14,23] + frac[11,23] + frac[12,23] >= 0.8
 cov_70_18: frac[8,18] + frac[1,18] + frac[13,18] + frac[14,18]
   + frac[10,18] + frac[12,18] + frac[4,18] >= 0.9
 cov_71_36: frac[10,36] + frac[2,36] + frac[14,36] + frac[13,36]
   + frac[9,36] + frac[12,36] + frac[5,36] >= 1
 cov_72_61: frac[3,61] + frac[12,61] + frac[9,61] + frac[8,61] + frac[1,61]
   >= 0.9
 cov_73_41: frac[11,41] + frac[12,41] + frac[14,41] + frac[1,41]
   + frac[8,41] + frac[7,41] + frac[13,41] >= 0.9
 cov_74_40: frac[3,40] + frac[9,40] + frac[0,40] + frac[8,40] + frac[12,40]
   >= 1
 cov_75_12: frac[8,12] + frac[14,12] + frac[11,12] + frac[7,12]
   + frac[13,12] + frac[12,12] + frac[0,12] >= 0.9
 cov_76_30: frac[13,30] + frac[0,30] + frac[14,30] + frac[5,30]
   + frac[12,30] + frac[10,30] + frac[8,30] >= 0.7
 cov_77_21: frac[6,21] + frac[11,21] + frac[13,21] + frac[2,21]
   + frac[9,21] + frac[14,21] + frac[12,21] >= 0.7
 cov_78_49: frac[7,49] + frac[13,49] + frac[5,49] + frac[11,49]
   + frac[10,49] >= 0.9
 cov_79_2: frac[3,2] + frac[14,2] + frac[13,2] + frac[7,2] + frac[12,2]
   + frac[9,2] + frac[11,2] >= 0.7
 cov_80_32: frac[13,32] + frac[12,32] + frac[4,32] + frac[10,32]
   + frac[8,32] + frac[0,32] + frac[14,32] >= 0.7
 cov_81_53: frac[14,53] + frac[12,53] + frac[8,53] + frac[6,53]
   + frac[11,53] + frac[0,53] + frac[13,53] >= 0.7
 cov_82_34: frac[5,34] + frac[10,34] + frac[11,34] + frac[13,34]
   + frac[6,34] >= 0.8
 cov_83_34: frac[5,34] + frac[10,34] + frac[12,34] + frac[9,34]
   + frac[2,34] + frac[14,34] + frac[13,34] >= 0.9
 cov_84_48: frac[11,48] + frac[14,48] + frac[9,48] + frac[2,48]
   + frac[13,48] + frac[6,48] + frac[12,48] >= 0.9
 cov_85_5: frac[4,5] + frac[10,5] + frac[5,5] >= 0.9
 cov_86_7: frac[12,7] + frac[8,7] + frac[14,7] + frac[0,7] + frac[13,7]
   + frac[11,7] + frac[7,7] >= 0.8
 cov_87_40: frac[9,40] + frac[13,40] + frac[12,40] + frac[14,40]
   + frac[4,40] + frac[2,40] + frac[10,40] >= 0.9
 cov_88_30: frac[5,30] + frac[10,30] + frac[4,30] >= 0.7
 cov_89_25: frac[7,25] + frac[13,25] + frac[11,25] + frac[5,25]
   + frac[10,25] >= 0.9
 cov_90_6: frac[13,6] + frac[11,6] + frac[12,6] + frac[8,6] + frac[14,6]
   + frac[0,6] + frac[6,6] >= 0.7
 cov_91_61: frac[10,61] + frac[13,61] + frac[2,61] + frac[12,61]
   + frac[9,61] + frac[14,61] + frac[4,61] >= 0.8
 cov_92_45: frac[0,45] + frac[8,45] + frac[1,45] >= 0.9
 cov_93_50: frac[11,50] + frac[4,50] + frac[13,50] + frac[7,50]
   + frac[10,50] >= 0.8
 cov_94_30: frac[13,30] + frac[11,30] + frac[7,30] + frac[10,30]
   + frac[4,30] >= 1
 cov_95_32: frac[13,32] + frac[5,32] + frac[11,32] + frac[10,32]
   + frac[7,32] >= 1
 cov_96_55: frac[10,55] + frac[4,55] + frac[5,55] >= 0.7
 cov_97_54: frac[6,54] + frac[2,54] + frac[13,54] + frac[14,54]
   + frac[9,54] + frac[11,54] + frac[12,54] >= 0.8
 cov_98_22: frac[10,22] + frac[4,22] + frac[5,22] >= 0.8
 cov_99_6: frac[13,6] + frac[11,6] + frac[12,6] + frac[8,6] + frac[14,6]
   + frac[0,6] + frac[6,6] >= 0.9
 cov_100_37: frac[7,37] + frac[13,37] + frac[3,37] + frac[9,37]
   + frac[12,37] + frac[14,37] + frac[11,37] >= 0.7
 cov_101_4: frac[13,4] + frac[9,4] + frac[5,4] + frac[12,4] + frac[10,4]
   + frac[2,4] + frac[14,4] >= 1
 cov_102_0: frac[1,0] + frac[8,0] + frac[14,0] + frac[13,0] + frac[12,0]
   + frac[4,0] + frac[10,0] >= 0.7
 cov_103_35: frac[13,35] + frac[8,35] + frac[14,35] + frac[7,35]
   + frac[11,35] + frac[12,35] + frac[1,35] >= 0.7
 cov_104_12: frac[8,12] + frac[12,12] + frac[3,12] + frac[0,12]
   + frac[9,12] >= 0.8
 cov_105_45: frac[4,45] + frac[10,45] + frac[6,45] + frac[11,45]
   + frac[13,45] >= 0.9
 cov_106_38: frac[8,38] + frac[14,38] + frac[0,38] + frac[13,38]
   + frac[5,38] + frac[10,38] + frac[12,38] >= 0.7
 cov_107_45: frac[10,45] + frac[6,45] + frac[11,45] + frac[13,45]
   + frac[5,45] >= 0.9
 cov_108_51: frac[13,51] + frac[8,51] + frac[1,51] + frac[5,51]
   + frac[10,51] + frac[12,51] + frac[14,51] >= 0.8
 cov_109_58: frac[13,58] + frac[12,58] + frac[10,58] + frac[14,58]
   + frac[9,58] + frac[4,58] + frac[3,58] >= 0.8
 cov_110_4: frac[11,4] + frac[13,4] + frac[9,4] + frac[12,4] + frac[2,4]
   + frac[14,4] + frac[6,4] >= 0.7
 cov_111_33: frac[1,33] + frac[8,33] + frac[13,33] + frac[14,33]
   + frac[6,33] + frac[11,33] + frac[12,33] >= 1
 cov_112_55: frac[10,55] + frac[12,55] + frac[13,55] + frac[9,55]
   + frac[2,55] + frac[5,55] + frac[14,55] >= 0.7
 cov_113_40: frac[3,40] + frac[1,40] + frac[9,40] + frac[8,40]
   + frac[12,40] >= 0.9
 cov_114_38: frac[14,38] + frac[4,38] + frac[13,38] + frac[9,38]
   + frac[3,38] + frac[10,38] + frac[12,38] >= 0.7
 cov_115_53: frac[5,53] + frac[14,53] + frac[12,53] + frac[8,53]
   + frac[10,53] + frac[1,53] + frac[13,53] >= 1
 cov_116_24: frac[5,24] + frac[14,24] + frac[10,24] + frac[8,24]
   + frac[13,24] + frac[0,24] + frac[12,24] >= 0.7
 cov_117_49: frac[7,49] + frac[13,49] + frac[11,49] + frac[12,49]
   + frac[3,49] + frac[9,49] + frac[14,49] >= 0.9
 cov_118_44: frac[12,44] + frac[3,44] + frac[9,44] + frac[0,44]
   + frac[8,44] >= 0.9
 cov_119_33: frac[13,33] + frac[14,33] + frac[12,33] + frac[3,33]
   + frac[4,33] + frac[9,33] + frac[10,33] >= 1
 cov_120_1: frac[14,1] + frac[11,1] + frac[12,1] + frac[3,1] + frac[9,1]
   + frac[7,1] + frac[13,1] >= 0.8
 cov_121_24: frac[4,24] + frac[10,24] + frac[6,24] + frac[13,24]
   + frac[11,24] >= 0.7
 cov_122_35: frac[4,35] + frac[13,35] + frac[7,35] + frac[10,35]
   + frac[11,35] >= 0.7
 cov_123_51: frac[13,51] + frac[5,51] + frac[7,51] + frac[10,51]
   + frac[11,51] >= 0.9
 cov_124_24: frac[9,24] + frac[8,24] + frac[2,24] + frac[1,24]
   + frac[12,24] >= 1
 cov_125_0: frac[8,0] + frac[0,0] + frac[7,0] + frac[14,0] + frac[13,0]
   + frac[12,0] + frac[11,0] >= 0.7
 cov_126_15: frac[11,15] + frac[5,15] + frac[10,15] + frac[6,15]
   + frac[13,15] >= 0.7
 cov_127_43: frac[12,43] + frac[1,43] + frac[9,43] + frac[3,43]
   + frac[8,43] >= 1
 cov_128_41: frac[12,41] + frac[2,41] + frac[9,41] + frac[0,41]
   + frac[8,41] >= 1
 cov_129_36: frac[10,36] + frac[6,36] + frac[11,36] + frac[13,36]
   + frac[5,36] >= 0.9
 cov_130_30: frac[13,30] + frac[11,30] + frac[5,30] + frac[10,30]
   + frac[6,30] >= 1
 cov_131_51: frac[13,51] + frac[8,51] + frac[5,51] + frac[10,51]
   + frac[12,51] + frac[14,51] + frac[0,51] >= 0.7
 cov_132_8: frac[11,8] + frac[4,8] + frac[10,8] + frac[7,8] + frac[13,8]
   >= 0.9
 cov_133_1: frac[14,1] + frac[11,1] + frac[12,1] + frac[8,1] + frac[0,1]
   + frac[7,1] + frac[13,1] >= 0.7
 cov_134_9: frac[0,9] + frac[8,9] + frac[1,9] >= 0.7
 cov_135_54: frac[9,54] + frac[0,54] + frac[3,54] + frac[12,54]
   + frac[8,54] >= 0.8
 cov_136_32: frac[13,32] + frac[12,32] + frac[4,32] + frac[10,32]
   + frac[9,32] + frac[2,32] + frac[14,32] >= 0.9
 cov_137_18: frac[8,18] + frac[6,18] + frac[13,18] + frac[14,18]
   + frac[0,18] + frac[12,18] + frac[11,18] >= 0.9
 cov_138_48: frac[8,48] + frac[1,48] + frac[0,48] >= 1
 cov_139_2: frac[3,2] + frac[8,2] + frac[12,2] + frac[9,2] + frac[1,2]
   >= 1
 cov_140_1: frac[12,1] + frac[3,1] + frac[9,1] + frac[8,1] + frac[0,1]
   >= 0.8
 cov_141_10: frac[12,10] + frac[2,10] + frac[1,10] + frac[9,10]
   + frac[8,10] >= 0.9
 cov_142_50: frac[12,50] + frac[14,50] + frac[4,50] + frac[8,50]
   + frac[13,50] + frac[1,50] + frac[10,50] >= 0.7
 cov_143_46: frac[11,46] + frac[5,46] + frac[10,46] + frac[6,46]
   + frac[13,46] >= 0.8
 cov_144_15: frac[9,15] + frac[3,15] + frac[2,15] >= 1
 cov_145_57: frac[5,57] + frac[10,57] + frac[13,57] + frac[6,57]
   + frac[11,57] >= 0.8
 cov_146_23: frac[5,23] + frac[10,23] + frac[4,23] >= 0.7
 cov_147_9: frac[9,9] + frac[10,9] + frac[13,9] + frac[14,9] + frac[12,9]
   + frac[4,9] + frac[2,9] >= 0.7
 cov_148_52: frac[1,52] + frac[2,52] + frac[9,52] + frac[12,52]
   + frac[8,52] >= 0.9
 cov_149_25: frac[13,25] + frac[2,25] + frac[12,25] + frac[4,25]
   + frac[9,25] + frac[14,25] + frac[10,25] >= 1
 cov_150_57: frac[3,57] + frac[9,57] + frac[8,57] + frac[1,57]
   + frac[12,57] >= 1
 cov_151_59: frac[12,59] + frac[0,59] + frac[10,59] + frac[14,59]
   + frac[4,59] + frac[13,59] + frac[8,59] >= 0.7
 cov_152_33: frac[2,33] + frac[13,33] + frac[14,33] + frac[12,33]
   + frac[4,33] + frac[9,33] + frac[10,33] >= 0.9
 cov_153_0: frac[7,0] + frac[13,0] + frac[4,0] + frac[11,0] + frac[10,0]
   >= 0.7
 cov_154_48: frac[4,48] + frac[10,48] + frac[14,48] + frac[8,48]
   + frac[13,48] + frac[1,48] + frac[12,48] >= 0.7
 cov_155_18: frac[8,18] + frac[1,18] + frac[13,18] + frac[5,18]
   + frac[14,18] + frac[10,18] + frac[12,18] >= 1
 cov_156_46: frac[11,46] + frac[7,46] + frac[12,46] + frac[0,46]
   + frac[8,46] + frac[14,46] + frac[13,46] >= 0.9
 cov_157_44: frac[13,44] + frac[12,44] + frac[1,44] + frac[8,44]
   + frac[4,44] + frac[10,44] + frac[14,44] >= 0.7
 cov_158_42: frac[3,42] + frac[2,42] + frac[9,42] >= 0.8
 cov_159_45: frac[2,45] + frac[11,45] + frac[7,45] + frac[14,45]
   + frac[13,45] + frac[12,45] + frac[9,45] >= 0.9
 cov_160_24: frac[4,24] + frac[14,24] + frac[10,24] + frac[8,24]
   + frac[13,24] + frac[0,24] + frac[12,24] >= 0.9
 cov_161_19: frac[9,19] + frac[0,19] + frac[12,19] + frac[3,19]
   + frac[8,19] >= 1
 cov_162_41: frac[6,41] + frac[11,41] + frac[4,41] + frac[10,41]
   + frac[13,41] >= 1
 cov_163_26: frac[10,26] + frac[4,26] + frac[5,26] >= 0.7
 cov_164_25: frac[8,25] + frac[1,25] + frac[12,25] + frac[3,25]
   + frac[9,25] >= 0.7
 cov_165_15: frac[1,15] + frac[0,15] + frac[8,15] >= 0.8
 cov_166_42: frac[8,42] + frac[12,42] + frac[2,42] + frac[1,42]
   + frac[9,42] >= 0.9
 cov_167_47: frac[10,47] + frac[6,47] + frac[13,47] + frac[11,47]
   + frac[5,47] >= 0.7
 cov_168_34: frac[9,34] + frac[2,34] + frac[3,34] >= 0.7
 cov_169_39: frac[9,39] + frac[5,39] + frac[3,39] + frac[10,39]
   + frac[12,39] + frac[14,39] + frac[13,39] >= 0.9
 cov_170_59: frac[6,59] + frac[7,59] + frac[11,59] >= 0.9
 cov_171_39: frac[10,39] + frac[12,39] + frac[8,39] + frac[14,39]
   + frac[4,39] + frac[0,39] + frac[13,39] >= 0.8
 cov_172_59: frac[12,59] + frac[9,59] + frac[14,59] + frac[13,59]
   + frac[2,59] + frac[6,59] + frac[11,59] >= 1
 cov_173_32: frac[13,32] + frac[12,32] + frac[11,32] + frac[8,32]
   + frac[0,32] + frac[7,32] + frac[14,32] >= 0.9
 cov_174_21: frac[3,21] + frac[13,21] + frac[9,21] + frac[14,21]
   + frac[12,21] + frac[4,21] + frac[10,21] >= 0.9
 cov_175_12: frac[6,12] + frac[14,12] + frac[11,12] + frac[13,12]
   + frac[12,12] + frac[3,12] + frac[9,12] >= 1
 cov_176_56: frac[13,56] + frac[12,56] + frac[5,56] + frac[10,56]
   + frac[3,56] + frac[14,56] + frac[9,56] >= 0.8
 cov_177_6: frac[1,6] + frac[8,6] + frac[0,6] >= 0.8
 cov_178_59: frac[12,59] + frac[9,59] + frac[1,59] + frac[2,59]
   + frac[8,59] >= 0.7
 cov_179_34: frac[12,34] + frac[9,34] + frac[1,34] + frac[3,34]
   + frac[8,34] >= 0.8
 cov_180_25: frac[13,25] + frac[2,25] + frac[12,25] + frac[4,25]
   + frac[9,25] + frac[14,25] + frac[10,25] >= 1
 cov_181_12: frac[8,12] + frac[12,12] + frac[3,12] + frac[0,12]
   + frac[9,12] >= 0.9
 cov_182_54: frac[13,54] + frac[11,54] + frac[7,54] + frac[10,54]
   + frac[4,54] >= 0.9
 cov_183_16: frac[13,16] + frac[12,16] + frac[4,16] + frac[3,16]
   + frac[14,16] + frac[10,16] + frac[9,16] >= 1
 cov_184_5: frac[8,5] + frac[0,5] + frac[2,5] + frac[9,5] + frac[12,5]
   >= 0.8
 cov_185_37: frac[13,37] + frac[4,37] + frac[10,37] + frac[6,37]
   + frac[11,37] >= 1
 cov_186_37: frac[13,37] + frac[9,37] + frac[5,37] + frac[12,37]
   + frac[10,37] + frac[2,37] + frac[14,37] >= 0.9
 cov_187_22: frac[11,22] + frac[7,22] + frac[10,22] + frac[13,22]
   + frac[5,22] >= 1
 cov_188_13: frac[6,13] + frac[11,13] + frac[7,13] >= 1
 cov_189_15: frac[1,15] + frac[0,15] + frac[8,15] >= 0.9
 cov_190_63: frac[13,63] + frac[9,63] + frac[5,63] + frac[14,63]
   + frac[3,63] + frac[12,63] + frac[10,63] >= 1
 cov_191_45: frac[0,45] + frac[8,45] + frac[10,45] + frac[14,45]
   + frac[13,45] + frac[12,45] + frac[5,45] >= 0.8
 cov_192_13: frac[3,13] + frac[9,13] + frac[2,13] >= 1
 cov_193_49: frac[12,49] + frac[0,49] + frac[3,49] + frac[9,49]
   + frac[8,49] >= 0.8
 cov_194_37: frac[3,37] + frac[9,37] + frac[2,37] >= 0.8
 cov_195_63: frac[11,63] + frac[7,63] + frac[6,63] >= 1
 cov_196_40: frac[1,40] + frac[0,40] + frac[8,40] >= 0.9
 cov_197_13: frac[13,13] + frac[12,13] + frac[9,13] + frac[5,13]
   + frac[2,13] + frac[10,13] + frac[14,13] >= 0.8
 cov_198_28: frac[13,28] + frac[12,28] + frac[3,28] + frac[14,28]
   + frac[9,28] + frac[5,28] + frac[10,28] >= 0.9
 cov_199_10: frac[12,10] + frac[11,10] + frac[1,10] + frac[6,10]
   + frac[13,10] + frac[8,10] + frac[14,10] >= 0.7
 cov_200_5: frac[10,5] + frac[14,5] + frac[3,5] + frac[13,5] + frac[9,5]
   + frac[5,5] + frac[12,5] >= 1
 cov_201_42: frac[13,42] + frac[6,42] + frac[8,42] + frac[14,42]
   + frac[12,42] + frac[11,42] + frac[1,42] >= 0.7
 cov_202_53: frac[2,53] + frac[14,53] + frac[12,53] + frac[11,53]
   + frac[7,53] + frac[13,53] + frac[9,53] >= 0.7
 cov_203_11: frac[4,11] + frac[5,11] + frac[10,11] >= 0.9
 cov_204_8: frac[11,8] + frac[5,8] + frac[10,8] + frac[7,8] + frac[13,8]
   >= 1
 cov_205_31: frac[14,31] + frac[12,31] + frac[1,31] + frac[8,31]
   + frac[13,31] + frac[11,31] + frac[7,31] >= 1
 cov_206_13: frac[13,13] + frac[5,13] + frac[10,13] + frac[11,13]
   + frac[7,13] >= 0.8
 cov_207_15: frac[1,15] + frac[5,15] + frac[12,15] + frac[10,15]
   + frac[8,15] + frac[14,15] + frac[13,15] >= 0.8
 cov_208_49: frac[7,49] + frac[6,49] + frac[11,49] >= 0.7
 cov_209_1: frac[11,1] + frac[4,1] + frac[10,1] + frac[7,1] + frac[13,1]
   >= 1
 cov_210_10: frac[5,10] + frac[11,10] + frac[10,10] + frac[6,10]
   + frac[13,10] >= 0.8
 cov_211_16: frac[7,16] + frac[2,16] + frac[13,16] + frac[12,16]
   + frac[11,16] + frac[14,16] + frac[9,16] >= 0.8
 cov_212_15: frac[12,15] + frac[10,15] + frac[0,15] + frac[8,15]
   + frac[14,15] + frac[4,15] + frac[13,15] >= 1
 cov_213_19: frac[14,19] + frac[11,19] + frac[0,19] + frac[12,19]
   + frac[13,19] + frac[8,19] + frac[7,19] >= 0.8
 cov_214_33: frac[8,33] + frac[0,33] + frac[7,33] + frac[13,33]
   + frac[14,33] + frac[11,33] + frac[12,33] >= 0.9
 cov_215_17: frac[0,17] + frac[12,17] + frac[4,17] + frac[14,17]
   + frac[10,17] + frac[8,17] + frac[13,17] >= 0.7
 cov_216_35: frac[13,35] + frac[5,35] + frac[6,35] + frac[10,35]
   + frac[11,35] >= 0.9
 cov_217_11: frac[4,11] + frac[13,11] + frac[14,11] + frac[12,11]
   + frac[2,11] + frac[9,11] + frac[10,11] >= 0.7
 cov_218_14: frac[6,14] + frac[8,14] + frac[14,14] + frac[13,14]
   + frac[11,14] + frac[1,14] + frac[12,14] >= 0.8
 cov_219_2: frac[6,2] + frac[13,2] + frac[10,2] + frac[4,2] + frac[11,2]
   >= 1
 cov_220_61: frac[10,61] + frac[5,61] + frac[4,61] >= 1
 cov_221_59: frac[10,59] + frac[4,59] + frac[5,59] >= 0.7
 cov_222_5: frac[14,5] + frac[11,5] + frac[7,5] + frac[3,5] + frac[13,5]
   + frac[9,5] + frac[12,5] >= 0.8
 cov_223_16: frac[6,16] + frac[1,16] + frac[13,16] + frac[12,16]
   + frac[11,16] + frac[14,16] + frac[8,16] >= 0.7
 cov_224_47: frac[2,47] + frac[10,47] + frac[14,47] + frac[13,47]
   + frac[9,47] + frac[5,47] + frac[12,47] >= 0.9
 cov_225_31: frac[14,31] + frac[12,31] + frac[1,31] + frac[10,31]
   + frac[8,31] + frac[4,31] + frac[13,31] >= 0.8
 cov_226_56: frac[2,56] + frac[12,56] + frac[0,56] + frac[9,56]
   + frac[8,56] >= 0.8
 cov_227_41: frac[5,41] + frac[4,41] + frac[10,41] >= 1
 cov_228_53: frac[5,53] + frac[14,53] + frac[12,53] + frac[10,53]
   + frac[13,53] + frac[3,53] + frac[9,53] >= 0.7
 cov_229_52: frac[6,52] + frac[11,52] + frac[13,52] + frac[4,52]
   + frac[10,52] >= 1
 cov_230_20: frac[9,20] + frac[12,20] + frac[8,20] + frac[0,20]
   + frac[2,20] >= 1
 cov_231_16: frac[2,16] + frac[6,16] + frac[13,16] + frac[12,16]
   + frac[11,16] + frac[14,16] + frac[9,16] >= 0.8
 cov_232_36: frac[3,36] + frac[9,36] + frac[12,36] + frac[8,36]
   + frac[1,36] >= 0.8
 cov_233_35: frac[3,35] + frac[9,35] + frac[2,35] >= 1
 cov_234_14: frac[6,14] + frac[11,14] + frac[7,14] >= 0.9
 cov_235_3: frac[7,3] + frac[9,3] + frac[11,3] + frac[12,3] + frac[2,3]
   + frac[13,3] + frac[14,3] >= 0.9
 cov_236_22: frac[11,22] + frac[7,22] + frac[10,22] + frac[13,22]
   + frac[5,22] >= 0.9
 cov_237_46: frac[1,46] + frac[0,46] + frac[8,46] >= 0.7
 cov_238_13: frac[1,13] + frac[12,13] + frac[9,13] + frac[8,13]
   + frac[2,13] >= 1
 cov_239_60: frac[8,60] + frac[1,60] + frac[0,60] >= 0.7
 cov_240_18: frac[8,18] + frac[13,18] + frac[7,18] + frac[14,18]
   + frac[0,18] + frac[12,18] + frac[11,18] >= 0.7
 cov_241_46: frac[11,46] + frac[9,46] + frac[12,46] + frac[3,46]
   + frac[6,46] + frac[14,46] + frac[13,46] >= 1
 cov_242_12: frac[2,12] + frac[6,12] + frac[14,12] + frac[11,12]
   + frac[13,12] + frac[12,12] + frac[9,12] >= 0.9
 cov_243_20: frac[13,20] + frac[9,20] + frac[12,20] + frac[11,20]
   + frac[7,20] + frac[2,20] + frac[14,20] >= 0.8
 cov_244_54: frac[6,54] + frac[13,54] + frac[14,54] + frac[0,54]
   + frac[11,54] + frac[12,54] + frac[8,54] >= 0.9
 cov_245_4: frac[11,4] + frac[7,4] + frac[13,4] + frac[4,4] + frac[10,4]
   >= 0.7
 cov_246_45: frac[0,45] + frac[8,45] + frac[6,45] + frac[11,45]
   + frac[14,45] + frac[13,45] + frac[12,45] >= 0.8
 cov_247_55: frac[7,55] + frac[12,55] + frac[8,55] + frac[1,55]
   + frac[13,55] + frac[11,55] + frac[14,55] >= 0.8
 cov_248_0: frac[9,0] + frac[14,0] + frac[13,0] + frac[12,0] + frac[4,0]
   + frac[10,0] + frac[2,0] >= 1
 cov_249_26: frac[11,26] + frac[6,26] + frac[7,26] >= 0.9
 cov_250_57: frac[14,57] + frac[8,57] + frac[13,57] + frac[6,57]
   + frac[11,57] + frac[12,57] + frac[0,57] >= 1
 cov_251_43: frac[14,43] + frac[12,43] + frac[2,43] + frac[9,43]
   + frac[10,43] + frac[4,43] + frac[13,43] >= 0.7
 cov_252_63: frac[13,63] + frac[2,63] + frac[9,63] + frac[5,63]
   + frac[14,63] + frac[12,63] + frac[10,63] >= 0.7
 cov_253_13: frac[1,13] + frac[12,13] + frac[9,13] + frac[8,13]
   + frac[2,13] >= 1
 cov_254_54: frac[13,54] + frac[14,54] + frac[0,54] + frac[10,54]
   + frac[12,54] + frac[8,54] + frac[4,54] >= 0.7
 cov_255_61: frac[3,61] + frac[12,61] + frac[9,61] + frac[8,61]
   + frac[1,61] >= 1
 cov_256_53: frac[5,53] + frac[4,53] + frac[10,53] >= 0.9
 cov_257_45: frac[4,45] + frac[2,45] + frac[10,45] + frac[14,45]
   + frac[13,45] + frac[12,45] + frac[9,45] >= 1
 cov_258_19: frac[9,19] + frac[14,19] + frac[11,19] + frac[12,19]
   + frac[3,19] + frac[13,19] + frac[6,19] >= 1
 cov_259_50: frac[4,50] + frac[5,50] + frac[10,50] >= 0.8
 cov_260_9: frac[0,9] + frac[7,9] + frac[8,9] + frac[13,9] + frac[11,9]
   + frac[14,9] + frac[12,9] >= 0.8
 cov_261_31: frac[6,31] + frac[11,31] + frac[7,31] >= 0.8
 cov_262_42: frac[4,42] + frac[13,42] + frac[0,42] + frac[8,42]
   + frac[14,42] + frac[12,42] + frac[10,42] >= 1
 cov_263_2: frac[3,2] + frac[14,2] + frac[13,2] + frac[7,2] + frac[12,2]
   + frac[9,2] + frac[11,2] >= 0.9
 cov_264_23: frac[13,23] + frac[11,23] + frac[7,23] + frac[10,23]
   + frac[4,23] >= 1
 cov_265_52: frac[1,52] + frac[11,52] + frac[7,52] + frac[13,52]
   + frac[14,52] + frac[12,52] + frac[8,52] >= 0.9
 cov_266_30: frac[13,30] + frac[11,30] + frac[7,30] + frac[14,30]
   + frac[12,30] + frac[1,30] + frac[8,30] >= 0.9
 cov_267_39: frac[9,39] + frac[3,39] + frac[10,39] + frac[12,39]
   + frac[14,39] + frac[4,39] + frac[13,39] >= 0.8
 cov_268_32: frac[12,32] + frac[9,32] + frac[2,32] + frac[8,32]
   + frac[1,32] >= 0.9
 cov_269_51: frac[6,51] + frac[7,51] + frac[11,51] >= 0.9
 cov_270_56: frac[13,56] + frac[6,56] + frac[1,56] + frac[12,56]
   + frac[11,56] + frac[14,56] + frac[8,56] >= 0.8
 cov_271_60: frac[3,60] + frac[9,60] + frac[8,60] + frac[1,60]
   + frac[12,60] >= 0.9
 cov_272_56: frac[13,56] + frac[12,56] + frac[5,56] + frac[10,56]
   + frac[3,56] + frac[14,56] + frac[9,56] >= 0.7
 cov_273_45: frac[8,45] + frac[10,45] + frac[14,45] + frac[13,45]
   + frac[1,45] + frac[12,45] + frac[5,45] >= 0.9
 cov_274_39: frac[11,39] + frac[1,39] + frac[7,39] + frac[12,39]
   + frac[8,39] + frac[14,39] + frac[13,39] >= 0.9
 cov_275_4: frac[11,4] + frac[13,4] + frac[9,4] + frac[12,4] + frac[2,4]
   + frac[14,4] + frac[6,4] >= 1
 cov_276_14: frac[10,14] + frac[6,14] + frac[4,14] + frac[13,14]
   + frac[11,14] >= 0.9
 cov_277_40: frac[13,40] + frac[6,40] + frac[11,40] + frac[5,40]
   + frac[10,40] >= 0.7
 cov_278_39: frac[9,39] + frac[11,39] + frac[7,39] + frac[12,39]
   + frac[2,39] + frac[14,39] + frac[13,39] >= 1
 cov_279_21: frac[1,21] + frac[13,21] + frac[14,21] + frac[12,21]
   + frac[8,21] + frac[4,21] + frac[10,21] >= 0.8
 cov_280_3: frac[7,3] + frac[11,3] + frac[6,3] >= 0.9
 cov_281_16: frac[0,16] + frac[12,16] + frac[3,16] + frac[9,16]
   + frac[8,16] >= 0.8
 cov_282_60: frac[6,60] + frac[11,60] + frac[7,60] >= 0.7
 cov_283_57: frac[3,57] + frac[9,57] + frac[8,57] + frac[1,57]
   + frac[12,57] >= 1
 cov_284_5: frac[8,5] + frac[3,5] + frac[9,5] + frac[1,5] + frac[12,5]
   >= 0.7
 cov_285_6: frac[13,6] + frac[11,6] + frac[7,6] + frac[12,6] + frac[8,6]
   + frac[14,6] + frac[0,6] >= 0.9
 cov_286_11: frac[13,11] + frac[8,11] + frac[14,11] + frac[5,11]
   + frac[1,11] + frac[12,11] + frac[10,11] >= 1
 cov_287_24: frac[4,24] + frac[10,24] + frac[7,24] + frac[13,24]
   + frac[11,24] >= 0.7
 cov_288_17: frac[0,17] + frac[12,17] + frac[14,17] + frac[8,17]
   + frac[13,17] + frac[6,17] + frac[11,17] >= 0.7
 cov_289_17: frac[3,17] + frac[9,17] + frac[2,17] >= 0.7
 cov_290_16: frac[6,16] + frac[13,16] + frac[0,16] + frac[12,16]
   + frac[11,16] + frac[14,16] + frac[8,16] >= 0.9
 cov_291_22: frac[3,22] + frac[2,22] + frac[9,22] >= 0.7
 cov_292_17: frac[12,17] + frac[4,17] + frac[14,17] + frac[10,17]
   + frac[8,17] + frac[1,17] + frac[13,17] >= 0.8
 cov_293_32: frac[5,32] + frac[4,32] + frac[10,32] >= 1
 cov_294_31: frac[5,31] + frac[14,31] + frac[3,31] + frac[12,31]
   + frac[10,31] + frac[13,31] + frac[9,31] >= 0.7
 cov_295_5: frac[4,5] + frac[10,5] + frac[5,5] >= 0.9
 cov_296_34: frac[10,34] + frac[12,34] + frac[9,34] + frac[4,34]
   + frac[3,34] + frac[14,34] + frac[13,34] >= 0.8
 cov_297_22: frac[3,22] + frac[12,22] + frac[8,22] + frac[1,22]
   + frac[9,22] >= 1
 cov_298_26: frac[14,26] + frac[10,26] + frac[4,26] + frac[13,26]
   + frac[8,26] + frac[0,26] + frac[12,26] >= 0.7
 cov_299_28: frac[0,28] + frac[13,28] + frac[12,28] + frac[14,28]
   + frac[5,28] + frac[8,28] + frac[10,28] >= 0.7
 cov_300_36: frac[0,36] + frac[2,36] + frac[9,36] + frac[12,36]
   + frac[8,36] >= 0.7
 cov_301_63: frac[8,63] + frac[9,63] + frac[3,63] + frac[12,63]
   + frac[1,63] >= 1
 cov_302_39: frac[9,39] + frac[3,39] + frac[2,39] >= 1
 cov_303_11: frac[0,11] + frac[8,11] + frac[12,11] + frac[2,11]
   + frac[9,11] >= 1
 cov_304_45: frac[8,45] + frac[6,45] + frac[11,45] + frac[14,45]
   + frac[13,45] + frac[1,45] + frac[12,45] >= 0.9
 cov_305_14: frac[14,14] + frac[13,14] + frac[11,14] + frac[7,14]
   + frac[9,14] + frac[12,14] + frac[3,14] >= 0.8
 cov_306_14: frac[6,14] + frac[8,14] + frac[14,14] + frac[13,14]
   + frac[11,14] + frac[1,14] + frac[12,14] >= 0.9
 cov_307_35: frac[4,35] + frac[13,35] + frac[8,35] + frac[14,35]
   + frac[10,35] + frac[12,35] + frac[1,35] >= 0.9
 cov_308_21: frac[6,21] + frac[11,21] + frac[3,21] + frac[13,21]
   + frac[9,21] + frac[14,21] + frac[12,21] >= 0.7
 cov_309_4: frac[11,4] + frac[13,4] + frac[9,4] + frac[12,4] + frac[2,4]
   + frac[14,4] + frac[6,4] >= 1
 cov_310_4: frac[11,4] + frac[7,4] + frac[13,4] + frac[12,4] + frac[1,4]
   + frac[8,4] + frac[14,4] >= 0.8
 cov_311_24: frac[4,24] + frac[10,24] + frac[6,24] + frac[13,24]
   + frac[11,24] >= 1
 cov_312_24: frac[4,24] + frac[10,24] + frac[7,24] + frac[13,24]
   + frac[11,24] >= 1
 cov_313_36: frac[0,36] + frac[2,36] + frac[9,36] + frac[12,36]
   + frac[8,36] >= 0.7
 cov_314_61: frac[0,61] + frac[8,61] + frac[1,61] >= 0.8
 cov_315_7: frac[10,7] + frac[6,7] + frac[13,7] + frac[5,7] + frac[11,7]
   >= 0.8
 cov_316_48: frac[4,48] + frac[10,48] + frac[14,48] + frac[9,48]
   + frac[2,48] + frac[13,48] + frac[12,48] >= 0.9
 cov_317_38: frac[14,38] + frac[4,38] + frac[13,38] + frac[9,38]
   + frac[3,38] + frac[10,38] + frac[12,38] >= 1
 cov_318_52: frac[1,52] + frac[13,52] + frac[14,52] + frac[12,52]
   + frac[8,52] + frac[4,52] + frac[10,52] >= 1
 cov_319_60: frac[9,60] + frac[8,60] + frac[1,60] + frac[2,60]
   + frac[12,60] >= 0.7
 accuracy_7_3: - 50.9765625 frac[7,3] + mem[7,3] = 0
 accuracy_5_31: - 50.9765625 frac[5,31] + mem[5,31] = 0
 accuracy_3_35: - 50.9765625 frac[3,35] + mem[3,35] = 0
 accuracy_6_28: - 50.9765625 frac[6,28] + mem[6,28] = 0
 accuracy_8_63: - 50.9765625 frac[8,63] + mem[8,63] = 0
 accuracy_7_25: - 50.9765625 frac[7,25] + mem[7,25] = 0
 accuracy_5_57: - 50.9765625 frac[5,57] + mem[5,57] = 0
 accuracy_6_54: - 50.9765625 frac[6,54] + mem[6,54] = 0
 accuracy_10_61: - 50.9765625 frac[10,61] + mem[10,61] = 0
 accuracy_8_5: - 50.9765625 frac[8,5] + mem[8,5] = 0
 accuracy_7_55: - 50.9765625 frac[7,55] + mem[7,55] = 0
 accuracy_13_32: - 50.9765625 frac[13,32] + mem[13,32] = 0
 accuracy_11_48: - 50.9765625 frac[11,48] + mem[11,48] = 0
 accuracy_9_0: - 50.9765625 frac[9,0] + mem[9,0] = 0
 accuracy_14_31: - 50.9765625 frac[14,31] + mem[14,31] = 0
 accuracy_12_59: - 50.9765625 frac[12,59] + mem[12,59] = 0
 accuracy_10_7: - 50.9765625 frac[10,7] + mem[10,7] = 0
 accuracy_13_58: - 50.9765625 frac[13,58] + mem[13,58] = 0
 accuracy_11_22: - 50.9765625 frac[11,22] + mem[11,22] = 0
 accuracy_0_17: - 50.9765625 frac[0,17] + mem[0,17] = 0
 accuracy_14_1: - 50.9765625 frac[14,1] + mem[14,1] = 0
 accuracy_12_17: - 50.9765625 frac[12,17] + mem[12,17] = 0
 accuracy_13_20: - 50.9765625 frac[13,20] + mem[13,20] = 0
 accuracy_2_27: - 50.9765625 frac[2,27] + mem[2,27] = 0
 accuracy_14_43: - 50.9765625 frac[14,43] + mem[14,43] = 0
 accuracy_3_2: - 50.9765625 frac[3,2] + mem[3,2] = 0
 accuracy_4_5: - 50.9765625 frac[4,5] + mem[4,5] = 0
 accuracy_2_53: - 50.9765625 frac[2,53] + mem[2,53] = 0
 accuracy_0_45: - 50.9765625 frac[0,45] + mem[0,45] = 0
 accuracy_5_24: - 50.9765625 frac[5,24] + mem[5,24] = 0
 accuracy_3_40: - 50.9765625 frac[3,40] + mem[3,40] = 0
 accuracy_1_40: - 50.9765625 frac[1,40] + mem[1,40] = 0
 accuracy_6_23: - 50.9765625 frac[6,23] + mem[6,23] = 0
 accuracy_4_35: - 50.9765625 frac[4,35] + mem[4,35] = 0
 accuracy_2_47: - 50.9765625 frac[2,47] + mem[2,47] = 0
 accuracy_8_38: - 50.9765625 frac[8,38] + mem[8,38] = 0
 accuracy_7_22: - 50.9765625 frac[7,22] + mem[7,22] = 0
 accuracy_5_34: - 50.9765625 frac[5,34] + mem[5,34] = 0
 accuracy_9_39: - 50.9765625 frac[9,39] + mem[9,39] = 0
 accuracy_6_41: - 50.9765625 frac[6,41] + mem[6,41] = 0
 accuracy_10_36: - 50.9765625 frac[10,36] + mem[10,36] = 0
 accuracy_8_12: - 50.9765625 frac[8,12] + mem[8,12] = 0
 accuracy_11_41: - 50.9765625 frac[11,41] + mem[11,41] = 0
 accuracy_9_9: - 50.9765625 frac[9,9] + mem[9,9] = 0
 accuracy_12_50: - 50.9765625 frac[12,50] + mem[12,50] = 0
 accuracy_10_14: - 50.9765625 frac[10,14] + mem[10,14] = 0
 accuracy_8_18: - 50.9765625 frac[8,18] + mem[8,18] = 0
 accuracy_13_51: - 50.9765625 frac[13,51] + mem[13,51] = 0
 accuracy_11_15: - 50.9765625 frac[11,15] + mem[11,15] = 0
 accuracy_9_19: - 50.9765625 frac[9,19] + mem[9,19] = 0
 accuracy_1_21: - 50.9765625 frac[1,21] + mem[1,21] = 0
 accuracy_13_13: - 50.9765625 frac[13,13] + mem[13,13] = 0
 accuracy_14_50: - 50.9765625 frac[14,50] + mem[14,50] = 0
 accuracy_1_15: - 50.9765625 frac[1,15] + mem[1,15] = 0
 accuracy_2_12: - 50.9765625 frac[2,12] + mem[2,12] = 0
 accuracy_0_36: - 50.9765625 frac[0,36] + mem[0,36] = 0
 accuracy_3_17: - 50.9765625 frac[3,17] + mem[3,17] = 0
 accuracy_1_33: - 50.9765625 frac[1,33] + mem[1,33] = 0
 accuracy_6_14: - 50.9765625 frac[6,14] + mem[6,14] = 0
 accuracy_4_42: - 50.9765625 frac[4,42] + mem[4,42] = 0
 accuracy_8_45: - 50.9765625 frac[8,45] + mem[8,45] = 0
 accuracy_9_40: - 50.9765625 frac[9,40] + mem[9,40] = 0
 accuracy_4_48: - 50.9765625 frac[4,48] + mem[4,48] = 0
 accuracy_10_47: - 50.9765625 frac[10,47] + mem[10,47] = 0
 accuracy_8_51: - 50.9765625 frac[8,51] + mem[8,51] = 0
 accuracy_7_37: - 50.9765625 frac[7,37] + mem[7,37] = 0
 accuracy_5_53: - 50.9765625 frac[5,53] + mem[5,53] = 0
 accuracy_11_46: - 50.9765625 frac[11,46] + mem[11,46] = 0
 accuracy_12_41: - 50.9765625 frac[12,41] + mem[12,41] = 0
 accuracy_10_9: - 50.9765625 frac[10,9] + mem[10,9] = 0
 accuracy_8_25: - 50.9765625 frac[8,25] + mem[8,25] = 0
 accuracy_13_44: - 50.9765625 frac[13,44] + mem[13,44] = 0
 accuracy_11_4: - 50.9765625 frac[11,4] + mem[11,4] = 0
 accuracy_9_20: - 50.9765625 frac[9,20] + mem[9,20] = 0
 accuracy_14_19: - 50.9765625 frac[14,19] + mem[14,19] = 0
 accuracy_12_7: - 50.9765625 frac[12,7] + mem[12,7] = 0
 accuracy_13_6: - 50.9765625 frac[13,6] + mem[13,6] = 0
 accuracy_11_26: - 50.9765625 frac[11,26] + mem[11,26] = 0
 accuracy_0_5: - 50.9765625 frac[0,5] + mem[0,5] = 0
 accuracy_14_53: - 50.9765625 frac[14,53] + mem[14,53] = 0
 accuracy_1_0: - 50.9765625 frac[1,0] + mem[1,0] = 0
 accuracy_4_11: - 50.9765625 frac[4,11] + mem[4,11] = 0
 accuracy_0_59: - 50.9765625 frac[0,59] + mem[0,59] = 0
 accuracy_5_10: - 50.9765625 frac[5,10] + mem[5,10] = 0
 accuracy_3_22: - 50.9765625 frac[3,22] + mem[3,22] = 0
 accuracy_4_17: - 50.9765625 frac[4,17] + mem[4,17] = 0
 accuracy_2_33: - 50.9765625 frac[2,33] + mem[2,33] = 0
 accuracy_7_4: - 50.9765625 frac[7,4] + mem[7,4] = 0
 accuracy_3_60: - 50.9765625 frac[3,60] + mem[3,60] = 0
 accuracy_6_27: - 50.9765625 frac[6,27] + mem[6,27] = 0
 accuracy_11_39: - 50.9765625 frac[11,39] + mem[11,39] = 0
 accuracy_9_59: - 50.9765625 frac[9,59] + mem[9,59] = 0
 accuracy_12_32: - 50.9765625 frac[12,32] + mem[12,32] = 0
 accuracy_10_48: - 50.9765625 frac[10,48] + mem[10,48] = 0
 accuracy_8_0: - 50.9765625 frac[8,0] + mem[8,0] = 0
 accuracy_13_37: - 50.9765625 frac[13,37] + mem[13,37] = 0
 accuracy_14_26: - 50.9765625 frac[14,26] + mem[14,26] = 0
 accuracy_10_26: - 50.9765625 frac[10,26] + mem[10,26] = 0
 accuracy_13_63: - 50.9765625 frac[13,63] + mem[13,63] = 0
 accuracy_11_19: - 50.9765625 frac[11,19] + mem[11,19] = 0
 accuracy_0_28: - 50.9765625 frac[0,28] + mem[0,28] = 0
 accuracy_14_60: - 50.9765625 frac[14,60] + mem[14,60] = 0
 accuracy_12_20: - 50.9765625 frac[12,20] + mem[12,20] = 0
 accuracy_1_25: - 50.9765625 frac[1,25] + mem[1,25] = 0
 accuracy_13_25: - 50.9765625 frac[13,25] + mem[13,25] = 0
 accuracy_14_38: - 50.9765625 frac[14,38] + mem[14,38] = 0
 accuracy_3_31: - 50.9765625 frac[3,31] + mem[3,31] = 0
 accuracy_1_51: - 50.9765625 frac[1,51] + mem[1,51] = 0
 accuracy_4_24: - 50.9765625 frac[4,24] + mem[4,24] = 0
 accuracy_2_56: - 50.9765625 frac[2,56] + mem[2,56] = 0
 accuracy_0_40: - 50.9765625 frac[0,40] + mem[0,40] = 0
 accuracy_3_37: - 50.9765625 frac[3,37] + mem[3,37] = 0
 accuracy_6_18: - 50.9765625 frac[6,18] + mem[6,18] = 0
 accuracy_4_38: - 50.9765625 frac[4,38] + mem[4,38] = 0
 accuracy_8_33: - 50.9765625 frac[8,33] + mem[8,33] = 0
 accuracy_5_39: - 50.9765625 frac[5,39] + mem[5,39] = 0
 accuracy_9_60: - 50.9765625 frac[9,60] + mem[9,60] = 0
 accuracy_6_52: - 50.9765625 frac[6,52] + mem[6,52] = 0
 accuracy_10_59: - 50.9765625 frac[10,59] + mem[10,59] = 0
 accuracy_8_7: - 50.9765625 frac[8,7] + mem[8,7] = 0
 accuracy_7_49: - 50.9765625 frac[7,49] + mem[7,49] = 0
 accuracy_11_50: - 50.9765625 frac[11,50] + mem[11,50] = 0
 accuracy_12_53: - 50.9765625 frac[12,53] + mem[12,53] = 0
 accuracy_10_5: - 50.9765625 frac[10,5] + mem[10,5] = 0
 accuracy_13_56: - 50.9765625 frac[13,56] + mem[13,56] = 0
 accuracy_11_8: - 50.9765625 frac[11,8] + mem[11,8] = 0
 accuracy_0_19: - 50.9765625 frac[0,19] + mem[0,19] = 0
 accuracy_14_7: - 50.9765625 frac[14,7] + mem[14,7] = 0
 accuracy_12_19: - 50.9765625 frac[12,19] + mem[12,19] = 0
 accuracy_1_18: - 50.9765625 frac[1,18] + mem[1,18] = 0
 accuracy_13_18: - 50.9765625 frac[13,18] + mem[13,18] = 0
 accuracy_2_25: - 50.9765625 frac[2,25] + mem[2,25] = 0
 accuracy_0_9: - 50.9765625 frac[0,9] + mem[0,9] = 0
 accuracy_14_41: - 50.9765625 frac[14,41] + mem[14,41] = 0
 accuracy_1_52: - 50.9765625 frac[1,52] + mem[1,52] = 0
 accuracy_3_42: - 50.9765625 frac[3,42] + mem[3,42] = 0
 accuracy_1_46: - 50.9765625 frac[1,46] + mem[1,46] = 0
 accuracy_6_21: - 50.9765625 frac[6,21] + mem[6,21] = 0
 accuracy_4_45: - 50.9765625 frac[4,45] + mem[4,45] = 0
 accuracy_2_45: - 50.9765625 frac[2,45] + mem[2,45] = 0
 accuracy_8_40: - 50.9765625 frac[8,40] + mem[8,40] = 0
 accuracy_7_16: - 50.9765625 frac[7,16] + mem[7,16] = 0
 accuracy_5_32: - 50.9765625 frac[5,32] + mem[5,32] = 0
 accuracy_9_37: - 50.9765625 frac[9,37] + mem[9,37] = 0
 accuracy_6_47: - 50.9765625 frac[6,47] + mem[6,47] = 0
 accuracy_10_34: - 50.9765625 frac[10,34] + mem[10,34] = 0
 accuracy_8_14: - 50.9765625 frac[8,14] + mem[8,14] = 0
 accuracy_7_46: - 50.9765625 frac[7,46] + mem[7,46] = 0
 accuracy_11_43: - 50.9765625 frac[11,43] + mem[11,43] = 0
 accuracy_9_15: - 50.9765625 frac[9,15] + mem[9,15] = 0
 accuracy_12_44: - 50.9765625 frac[12,44] + mem[12,44] = 0
 accuracy_8_20: - 50.9765625 frac[8,20] + mem[8,20] = 0
 accuracy_13_49: - 50.9765625 frac[13,49] + mem[13,49] = 0
 accuracy_11_1: - 50.9765625 frac[11,1] + mem[11,1] = 0
 accuracy_9_17: - 50.9765625 frac[9,17] + mem[9,17] = 0
 accuracy_14_14: - 50.9765625 frac[14,14] + mem[14,14] = 0
 accuracy_12_10: - 50.9765625 frac[12,10] + mem[12,10] = 0
 accuracy_10_22: - 50.9765625 frac[10,22] + mem[10,22] = 0
 accuracy_13_11: - 50.9765625 frac[13,11] + mem[13,11] = 0
 accuracy_2_16: - 50.9765625 frac[2,16] + mem[2,16] = 0
 accuracy_0_0: - 50.9765625 frac[0,0] + mem[0,0] = 0
 accuracy_14_48: - 50.9765625 frac[14,48] + mem[14,48] = 0
 accuracy_3_13: - 50.9765625 frac[3,13] + mem[3,13] = 0
 accuracy_1_13: - 50.9765625 frac[1,13] + mem[1,13] = 0
 accuracy_4_14: - 50.9765625 frac[4,14] + mem[4,14] = 0
 accuracy_2_10: - 50.9765625 frac[2,10] + mem[2,10] = 0
 accuracy_0_38: - 50.9765625 frac[0,38] + mem[0,38] = 0
 accuracy_5_15: - 50.9765625 frac[5,15] + mem[5,15] = 0
 accuracy_3_19: - 50.9765625 frac[3,19] + mem[3,19] = 0
 accuracy_1_39: - 50.9765625 frac[1,39] + mem[1,39] = 0
 accuracy_6_12: - 50.9765625 frac[6,12] + mem[6,12] = 0
 accuracy_2_36: - 50.9765625 frac[2,36] + mem[2,36] = 0
 accuracy_8_47: - 50.9765625 frac[8,47] + mem[8,47] = 0
 accuracy_7_9: - 50.9765625 frac[7,9] + mem[7,9] = 0
 accuracy_5_41: - 50.9765625 frac[5,41] + mem[5,41] = 0
 accuracy_3_57: - 50.9765625 frac[3,57] + mem[3,57] = 0
 accuracy_9_46: - 50.9765625 frac[9,46] + mem[9,46] = 0
 accuracy_4_50: - 50.9765625 frac[4,50] + mem[4,50] = 0
 accuracy_10_45: - 50.9765625 frac[10,45] + mem[10,45] = 0
 accuracy_8_53: - 50.9765625 frac[8,53] + mem[8,53] = 0
 accuracy_7_39: - 50.9765625 frac[7,39] + mem[7,39] = 0
 accuracy_5_51: - 50.9765625 frac[5,51] + mem[5,51] = 0
 accuracy_11_32: - 50.9765625 frac[11,32] + mem[11,32] = 0
 accuracy_9_48: - 50.9765625 frac[9,48] + mem[9,48] = 0
 accuracy_6_56: - 50.9765625 frac[6,56] + mem[6,56] = 0
 accuracy_12_43: - 50.9765625 frac[12,43] + mem[12,43] = 0
 accuracy_10_55: - 50.9765625 frac[10,55] + mem[10,55] = 0
 accuracy_8_27: - 50.9765625 frac[8,27] + mem[8,27] = 0
 accuracy_13_42: - 50.9765625 frac[13,42] + mem[13,42] = 0
 accuracy_11_6: - 50.9765625 frac[11,6] + mem[11,6] = 0
 accuracy_9_26: - 50.9765625 frac[9,26] + mem[9,26] = 0
 accuracy_14_17: - 50.9765625 frac[14,17] + mem[14,17] = 0
 accuracy_12_1: - 50.9765625 frac[12,1] + mem[12,1] = 0
 accuracy_10_17: - 50.9765625 frac[10,17] + mem[10,17] = 0
 accuracy_13_4: - 50.9765625 frac[13,4] + mem[13,4] = 0
 accuracy_11_28: - 50.9765625 frac[11,28] + mem[11,28] = 0
 accuracy_0_7: - 50.9765625 frac[0,7] + mem[0,7] = 0
 accuracy_14_59: - 50.9765625 frac[14,59] + mem[14,59] = 0
 accuracy_12_31: - 50.9765625 frac[12,31] + mem[12,31] = 0
 accuracy_1_6: - 50.9765625 frac[1,6] + mem[1,6] = 0
 accuracy_13_30: - 50.9765625 frac[13,30] + mem[13,30] = 0
 accuracy_2_5: - 50.9765625 frac[2,5] + mem[2,5] = 0
 accuracy_0_61: - 50.9765625 frac[0,61] + mem[0,61] = 0
 accuracy_5_8: - 50.9765625 frac[5,8] + mem[5,8] = 0
 accuracy_3_24: - 50.9765625 frac[3,24] + mem[3,24] = 0
 accuracy_1_56: - 50.9765625 frac[1,56] + mem[1,56] = 0
 accuracy_6_7: - 50.9765625 frac[6,7] + mem[6,7] = 0
 accuracy_2_63: - 50.9765625 frac[2,63] + mem[2,63] = 0
 accuracy_7_6: - 50.9765625 frac[7,6] + mem[7,6] = 0
 accuracy_5_18: - 50.9765625 frac[5,18] + mem[5,18] = 0
 accuracy_6_25: - 50.9765625 frac[6,25] + mem[6,25] = 0
 accuracy_8_60: - 50.9765625 frac[8,60] + mem[8,60] = 0
 accuracy_9_57: - 50.9765625 frac[9,57] + mem[9,57] = 0
 accuracy_6_51: - 50.9765625 frac[6,51] + mem[6,51] = 0
 accuracy_12_34: - 50.9765625 frac[12,34] + mem[12,34] = 0
 accuracy_8_2: - 50.9765625 frac[8,2] + mem[8,2] = 0
 accuracy_13_35: - 50.9765625 frac[13,35] + mem[13,35] = 0
 accuracy_11_63: - 50.9765625 frac[11,63] + mem[11,63] = 0
 accuracy_9_3: - 50.9765625 frac[9,3] + mem[9,3] = 0
 accuracy_14_24: - 50.9765625 frac[14,24] + mem[14,24] = 0
 accuracy_12_56: - 50.9765625 frac[12,56] + mem[12,56] = 0
 accuracy_10_24: - 50.9765625 frac[10,24] + mem[10,24] = 0
 accuracy_13_61: - 50.9765625 frac[13,61] + mem[13,61] = 0
 accuracy_11_21: - 50.9765625 frac[11,21] + mem[11,21] = 0
 accuracy_0_30: - 50.9765625 frac[0,30] + mem[0,30] = 0
 accuracy_14_2: - 50.9765625 frac[14,2] + mem[14,2] = 0
 accuracy_12_22: - 50.9765625 frac[12,22] + mem[12,22] = 0
 accuracy_1_31: - 50.9765625 frac[1,31] + mem[1,31] = 0
 accuracy_13_23: - 50.9765625 frac[13,23] + mem[13,23] = 0
 accuracy_14_36: - 50.9765625 frac[14,36] + mem[14,36] = 0
 accuracy_3_1: - 50.9765625 frac[3,1] + mem[3,1] = 0
 accuracy_4_26: - 50.9765625 frac[4,26] + mem[4,26] = 0
 accuracy_2_54: - 50.9765625 frac[2,54] + mem[2,54] = 0
 accuracy_0_42: - 50.9765625 frac[0,42] + mem[0,42] = 0
 accuracy_5_27: - 50.9765625 frac[5,27] + mem[5,27] = 0
 accuracy_3_39: - 50.9765625 frac[3,39] + mem[3,39] = 0
 accuracy_1_43: - 50.9765625 frac[1,43] + mem[1,43] = 0
 accuracy_6_16: - 50.9765625 frac[6,16] + mem[6,16] = 0
 accuracy_4_32: - 50.9765625 frac[4,32] + mem[4,32] = 0
 accuracy_8_35: - 50.9765625 frac[8,35] + mem[8,35] = 0
 accuracy_5_37: - 50.9765625 frac[5,37] + mem[5,37] = 0
 accuracy_9_34: - 50.9765625 frac[9,34] + mem[9,34] = 0
 accuracy_6_42: - 50.9765625 frac[6,42] + mem[6,42] = 0
 accuracy_10_57: - 50.9765625 frac[10,57] + mem[10,57] = 0
 accuracy_8_9: - 50.9765625 frac[8,9] + mem[8,9] = 0
 accuracy_7_51: - 50.9765625 frac[7,51] + mem[7,51] = 0
 accuracy_11_52: - 50.9765625 frac[11,52] + mem[11,52] = 0
 accuracy_9_4: - 50.9765625 frac[9,4] + mem[9,4] = 0
 accuracy_12_55: - 50.9765625 frac[12,55] + mem[12,55] = 0
 accuracy_10_3: - 50.9765625 frac[10,3] + mem[10,3] = 0
 accuracy_13_54: - 50.9765625 frac[13,54] + mem[13,54] = 0
 accuracy_11_10: - 50.9765625 frac[11,10] + mem[11,10] = 0
 accuracy_14_5: - 50.9765625 frac[14,5] + mem[14,5] = 0
 accuracy_12_13: - 50.9765625 frac[12,13] + mem[12,13] = 0
 accuracy_1_16: - 50.9765625 frac[1,16] + mem[1,16] = 0
 accuracy_13_16: - 50.9765625 frac[13,16] + mem[13,16] = 0
 accuracy_2_23: - 50.9765625 frac[2,23] + mem[2,23] = 0
 accuracy_0_11: - 50.9765625 frac[0,11] + mem[0,11] = 0
 accuracy_14_47: - 50.9765625 frac[14,47] + mem[14,47] = 0
 accuracy_1_10: - 50.9765625 frac[1,10] + mem[1,10] = 0
 accuracy_4_1: - 50.9765625 frac[4,1] + mem[4,1] = 0
 accuracy_0_33: - 50.9765625 frac[0,33] + mem[0,33] = 0
 accuracy_5_4: - 50.9765625 frac[5,4] + mem[5,4] = 0
 accuracy_3_44: - 50.9765625 frac[3,44] + mem[3,44] = 0
 accuracy_1_44: - 50.9765625 frac[1,44] + mem[1,44] = 0
 accuracy_2_43: - 50.9765625 frac[2,43] + mem[2,43] = 0
 accuracy_8_42: - 50.9765625 frac[8,42] + mem[8,42] = 0
 accuracy_7_18: - 50.9765625 frac[7,18] + mem[7,18] = 0
 accuracy_5_46: - 50.9765625 frac[5,46] + mem[5,46] = 0
 accuracy_9_43: - 50.9765625 frac[9,43] + mem[9,43] = 0
 accuracy_6_45: - 50.9765625 frac[6,45] + mem[6,45] = 0
 accuracy_4_53: - 50.9765625 frac[4,53] + mem[4,53] = 0
 accuracy_10_32: - 50.9765625 frac[10,32] + mem[10,32] = 0
 accuracy_8_48: - 50.9765625 frac[8,48] + mem[8,48] = 0
 accuracy_11_45: - 50.9765625 frac[11,45] + mem[11,45] = 0
 accuracy_9_13: - 50.9765625 frac[9,13] + mem[9,13] = 0
 accuracy_12_46: - 50.9765625 frac[12,46] + mem[12,46] = 0
 accuracy_10_10: - 50.9765625 frac[10,10] + mem[10,10] = 0
 accuracy_8_22: - 50.9765625 frac[8,22] + mem[8,22] = 0
 accuracy_13_47: - 50.9765625 frac[13,47] + mem[13,47] = 0
 accuracy_11_3: - 50.9765625 frac[11,3] + mem[11,3] = 0
 accuracy_9_23: - 50.9765625 frac[9,23] + mem[9,23] = 0
 accuracy_14_12: - 50.9765625 frac[14,12] + mem[14,12] = 0
 accuracy_12_4: - 50.9765625 frac[12,4] + mem[12,4] = 0
 accuracy_13_9: - 50.9765625 frac[13,9] + mem[13,9] = 0
 accuracy_11_25: - 50.9765625 frac[11,25] + mem[11,25] = 0
 accuracy_14_54: - 50.9765625 frac[14,54] + mem[14,54] = 0
 accuracy_3_15: - 50.9765625 frac[3,15] + mem[3,15] = 0
 accuracy_1_3: - 50.9765625 frac[1,3] + mem[1,3] = 0
 accuracy_4_8: - 50.9765625 frac[4,8] + mem[4,8] = 0
 accuracy_0_56: - 50.9765625 frac[0,56] + mem[0,56] = 0
 accuracy_5_13: - 50.9765625 frac[5,13] + mem[5,13] = 0
 accuracy_3_21: - 50.9765625 frac[3,21] + mem[3,21] = 0
 accuracy_1_37: - 50.9765625 frac[1,37] + mem[1,37] = 0
 accuracy_6_2: - 50.9765625 frac[6,2] + mem[6,2] = 0
 accuracy_4_22: - 50.9765625 frac[4,22] + mem[4,22] = 0
 accuracy_2_34: - 50.9765625 frac[2,34] + mem[2,34] = 0
 accuracy_7_11: - 50.9765625 frac[7,11] + mem[7,11] = 0
 accuracy_5_23: - 50.9765625 frac[5,23] + mem[5,23] = 0
 accuracy_9_44: - 50.9765625 frac[9,44] + mem[9,44] = 0
 accuracy_6_36: - 50.9765625 frac[6,36] + mem[6,36] = 0
 accuracy_10_43: - 50.9765625 frac[10,43] + mem[10,43] = 0
 accuracy_8_55: - 50.9765625 frac[8,55] + mem[8,55] = 0
 accuracy_7_33: - 50.9765625 frac[7,33] + mem[7,33] = 0
 accuracy_5_49: - 50.9765625 frac[5,49] + mem[5,49] = 0
 accuracy_11_34: - 50.9765625 frac[11,34] + mem[11,34] = 0
 accuracy_9_54: - 50.9765625 frac[9,54] + mem[9,54] = 0
 accuracy_12_37: - 50.9765625 frac[12,37] + mem[12,37] = 0
 accuracy_10_53: - 50.9765625 frac[10,53] + mem[10,53] = 0
 accuracy_7_63: - 50.9765625 frac[7,63] + mem[7,63] = 0
 accuracy_13_40: - 50.9765625 frac[13,40] + mem[13,40] = 0
 accuracy_11_56: - 50.9765625 frac[11,56] + mem[11,56] = 0
 accuracy_9_24: - 50.9765625 frac[9,24] + mem[9,24] = 0
 accuracy_14_23: - 50.9765625 frac[14,23] + mem[14,23] = 0
 accuracy_12_3: - 50.9765625 frac[12,3] + mem[12,3] = 0
 accuracy_10_31: - 50.9765625 frac[10,31] + mem[10,31] = 0
 accuracy_13_2: - 50.9765625 frac[13,2] + mem[13,2] = 0
 accuracy_11_30: - 50.9765625 frac[11,30] + mem[11,30] = 0
 accuracy_14_57: - 50.9765625 frac[14,57] + mem[14,57] = 0
 accuracy_12_25: - 50.9765625 frac[12,25] + mem[12,25] = 0
 accuracy_1_4: - 50.9765625 frac[1,4] + mem[1,4] = 0
 accuracy_13_28: - 50.9765625 frac[13,28] + mem[13,28] = 0
 accuracy_2_3: - 50.9765625 frac[2,3] + mem[2,3] = 0
 accuracy_14_35: - 50.9765625 frac[14,35] + mem[14,35] = 0
 accuracy_3_26: - 50.9765625 frac[3,26] + mem[3,26] = 0
 accuracy_6_5: - 50.9765625 frac[6,5] + mem[6,5] = 0
 accuracy_4_29: - 50.9765625 frac[4,29] + mem[4,29] = 0
 accuracy_2_61: - 50.9765625 frac[2,61] + mem[2,61] = 0
 accuracy_7_0: - 50.9765625 frac[7,0] + mem[7,0] = 0
 accuracy_6_31: - 50.9765625 frac[6,31] + mem[6,31] = 0
 accuracy_4_59: - 50.9765625 frac[4,59] + mem[4,59] = 0
 accuracy_7_30: - 50.9765625 frac[7,30] + mem[7,30] = 0
 accuracy_9_63: - 50.9765625 frac[9,63] + mem[9,63] = 0
 accuracy_6_49: - 50.9765625 frac[6,49] + mem[6,49] = 0
 accuracy_8_4: - 50.9765625 frac[8,4] + mem[8,4] = 0
 accuracy_7_52: - 50.9765625 frac[7,52] + mem[7,52] = 0
 accuracy_13_33: - 50.9765625 frac[13,33] + mem[13,33] = 0
 accuracy_11_49: - 50.9765625 frac[11,49] + mem[11,49] = 0
 accuracy_9_1: - 50.9765625 frac[9,1] + mem[9,1] = 0
 accuracy_14_30: - 50.9765625 frac[14,30] + mem[14,30] = 0
 accuracy_12_58: - 50.9765625 frac[12,58] + mem[12,58] = 0
 accuracy_13_59: - 50.9765625 frac[13,59] + mem[13,59] = 0
 accuracy_11_23: - 50.9765625 frac[11,23] + mem[11,23] = 0
 accuracy_0_16: - 50.9765625 frac[0,16] + mem[0,16] = 0
 accuracy_14_0: - 50.9765625 frac[14,0] + mem[14,0] = 0
 accuracy_12_16: - 50.9765625 frac[12,16] + mem[12,16] = 0
 accuracy_13_21: - 50.9765625 frac[13,21] + mem[13,21] = 0
 accuracy_0_54: - 50.9765625 frac[0,54] + mem[0,54] = 0
 accuracy_14_42: - 50.9765625 frac[14,42] + mem[14,42] = 0
 accuracy_1_55: - 50.9765625 frac[1,55] + mem[1,55] = 0
 accuracy_4_4: - 50.9765625 frac[4,4] + mem[4,4] = 0
 accuracy_2_52: - 50.9765625 frac[2,52] + mem[2,52] = 0
 accuracy_0_44: - 50.9765625 frac[0,44] + mem[0,44] = 0
 accuracy_5_25: - 50.9765625 frac[5,25] + mem[5,25] = 0
 accuracy_1_41: - 50.9765625 frac[1,41] + mem[1,41] = 0
 accuracy_6_22: - 50.9765625 frac[6,22] + mem[6,22] = 0
 accuracy_4_34: - 50.9765625 frac[4,34] + mem[4,34] = 0
 accuracy_8_37: - 50.9765625 frac[8,37] + mem[8,37] = 0
 accuracy_7_23: - 50.9765625 frac[7,23] + mem[7,23] = 0
 accuracy_5_35: - 50.9765625 frac[5,35] + mem[5,35] = 0
 accuracy_9_32: - 50.9765625 frac[9,32] + mem[9,32] = 0
 accuracy_6_40: - 50.9765625 frac[6,40] + mem[6,40] = 0
 accuracy_10_39: - 50.9765625 frac[10,39] + mem[10,39] = 0
 accuracy_8_11: - 50.9765625 frac[8,11] + mem[8,11] = 0
 accuracy_7_45: - 50.9765625 frac[7,45] + mem[7,45] = 0
 accuracy_11_54: - 50.9765625 frac[11,54] + mem[11,54] = 0
 accuracy_9_10: - 50.9765625 frac[9,10] + mem[9,10] = 0
 accuracy_12_49: - 50.9765625 frac[12,49] + mem[12,49] = 0
 accuracy_10_1: - 50.9765625 frac[10,1] + mem[10,1] = 0
 accuracy_8_17: - 50.9765625 frac[8,17] + mem[8,17] = 0
 accuracy_13_52: - 50.9765625 frac[13,52] + mem[13,52] = 0
 accuracy_11_12: - 50.9765625 frac[11,12] + mem[11,12] = 0
 accuracy_14_11: - 50.9765625 frac[14,11] + mem[14,11] = 0
 accuracy_12_15: - 50.9765625 frac[12,15] + mem[12,15] = 0
 accuracy_1_22: - 50.9765625 frac[1,22] + mem[1,22] = 0
 accuracy_13_14: - 50.9765625 frac[13,14] + mem[13,14] = 0
 accuracy_2_21: - 50.9765625 frac[2,21] + mem[2,21] = 0
 accuracy_14_45: - 50.9765625 frac[14,45] + mem[14,45] = 0
 accuracy_4_3: - 50.9765625 frac[4,3] + mem[4,3] = 0
 accuracy_2_15: - 50.9765625 frac[2,15] + mem[2,15] = 0
 accuracy_0_35: - 50.9765625 frac[0,35] + mem[0,35] = 0
 accuracy_5_2: - 50.9765625 frac[5,2] + mem[5,2] = 0
 accuracy_3_46: - 50.9765625 frac[3,46] + mem[3,46] = 0
 accuracy_1_34: - 50.9765625 frac[1,34] + mem[1,34] = 0
 accuracy_4_41: - 50.9765625 frac[4,41] + mem[4,41] = 0
 accuracy_2_41: - 50.9765625 frac[2,41] + mem[2,41] = 0
 accuracy_8_44: - 50.9765625 frac[8,44] + mem[8,44] = 0
 accuracy_7_12: - 50.9765625 frac[7,12] + mem[7,12] = 0
 accuracy_9_41: - 50.9765625 frac[9,41] + mem[9,41] = 0
 accuracy_6_35: - 50.9765625 frac[6,35] + mem[6,35] = 0
 accuracy_4_55: - 50.9765625 frac[4,55] + mem[4,55] = 0
 accuracy_10_46: - 50.9765625 frac[10,46] + mem[10,46] = 0
 accuracy_8_50: - 50.9765625 frac[8,50] + mem[8,50] = 0
 accuracy_11_47: - 50.9765625 frac[11,47] + mem[11,47] = 0
 accuracy_12_40: - 50.9765625 frac[12,40] + mem[12,40] = 0
 accuracy_10_8: - 50.9765625 frac[10,8] + mem[10,8] = 0
 accuracy_8_24: - 50.9765625 frac[8,24] + mem[8,24] = 0
 accuracy_13_45: - 50.9765625 frac[13,45] + mem[13,45] = 0
 accuracy_11_5: - 50.9765625 frac[11,5] + mem[11,5] = 0
 accuracy_9_21: - 50.9765625 frac[9,21] + mem[9,21] = 0
 accuracy_14_18: - 50.9765625 frac[14,18] + mem[14,18] = 0
 accuracy_12_6: - 50.9765625 frac[12,6] + mem[12,6] = 0
 accuracy_10_18: - 50.9765625 frac[10,18] + mem[10,18] = 0
 accuracy_13_7: - 50.9765625 frac[13,7] + mem[13,7] = 0
 accuracy_11_27: - 50.9765625 frac[11,27] + mem[11,27] = 0
 accuracy_14_52: - 50.9765625 frac[14,52] + mem[14,52] = 0
 accuracy_12_28: - 50.9765625 frac[12,28] + mem[12,28] = 0
 accuracy_5_11: - 50.9765625 frac[5,11] + mem[5,11] = 0
 accuracy_1_59: - 50.9765625 frac[1,59] + mem[1,59] = 0
 accuracy_4_16: - 50.9765625 frac[4,16] + mem[4,16] = 0
 accuracy_2_32: - 50.9765625 frac[2,32] + mem[2,32] = 0
 accuracy_7_5: - 50.9765625 frac[7,5] + mem[7,5] = 0
 accuracy_5_21: - 50.9765625 frac[5,21] + mem[5,21] = 0
 accuracy_3_61: - 50.9765625 frac[3,61] + mem[3,61] = 0
 accuracy_6_26: - 50.9765625 frac[6,26] + mem[6,26] = 0
 accuracy_10_41: - 50.9765625 frac[10,41] + mem[10,41] = 0
 accuracy_8_57: - 50.9765625 frac[8,57] + mem[8,57] = 0
 accuracy_7_35: - 50.9765625 frac[7,35] + mem[7,35] = 0
 accuracy_5_63: - 50.9765625 frac[5,63] + mem[5,63] = 0
 accuracy_11_36: - 50.9765625 frac[11,36] + mem[11,36] = 0
 accuracy_9_52: - 50.9765625 frac[9,52] + mem[9,52] = 0
 accuracy_6_60: - 50.9765625 frac[6,60] + mem[6,60] = 0
 accuracy_12_39: - 50.9765625 frac[12,39] + mem[12,39] = 0
 accuracy_10_51: - 50.9765625 frac[10,51] + mem[10,51] = 0
 accuracy_8_31: - 50.9765625 frac[8,31] + mem[8,31] = 0
 accuracy_13_38: - 50.9765625 frac[13,38] + mem[13,38] = 0
 accuracy_14_21: - 50.9765625 frac[14,21] + mem[14,21] = 0
 accuracy_12_61: - 50.9765625 frac[12,61] + mem[12,61] = 0
 accuracy_10_29: - 50.9765625 frac[10,29] + mem[10,29] = 0
 accuracy_13_0: - 50.9765625 frac[13,0] + mem[13,0] = 0
 accuracy_11_16: - 50.9765625 frac[11,16] + mem[11,16] = 0
 accuracy_0_27: - 50.9765625 frac[0,27] + mem[0,27] = 0
 accuracy_14_63: - 50.9765625 frac[14,63] + mem[14,63] = 0
 accuracy_12_27: - 50.9765625 frac[12,27] + mem[12,27] = 0
 accuracy_13_26: - 50.9765625 frac[13,26] + mem[13,26] = 0
 accuracy_0_49: - 50.9765625 frac[0,49] + mem[0,49] = 0
 accuracy_14_33: - 50.9765625 frac[14,33] + mem[14,33] = 0
 accuracy_3_28: - 50.9765625 frac[3,28] + mem[3,28] = 0
 accuracy_1_60: - 50.9765625 frac[1,60] + mem[1,60] = 0
 accuracy_4_31: - 50.9765625 frac[4,31] + mem[4,31] = 0
 accuracy_2_59: - 50.9765625 frac[2,59] + mem[2,59] = 0
 accuracy_7_2: - 50.9765625 frac[7,2] + mem[7,2] = 0
 accuracy_5_30: - 50.9765625 frac[5,30] + mem[5,30] = 0
 accuracy_3_34: - 50.9765625 frac[3,34] + mem[3,34] = 0
 accuracy_6_29: - 50.9765625 frac[6,29] + mem[6,29] = 0
 accuracy_4_37: - 50.9765625 frac[4,37] + mem[4,37] = 0
 accuracy_8_32: - 50.9765625 frac[8,32] + mem[8,32] = 0
 accuracy_7_24: - 50.9765625 frac[7,24] + mem[7,24] = 0
 accuracy_5_56: - 50.9765625 frac[5,56] + mem[5,56] = 0
 accuracy_9_61: - 50.9765625 frac[9,61] + mem[9,61] = 0
 accuracy_10_58: - 50.9765625 frac[10,58] + mem[10,58] = 0
 accuracy_8_6: - 50.9765625 frac[8,6] + mem[8,6] = 0
 accuracy_7_54: - 50.9765625 frac[7,54] + mem[7,54] = 0
 accuracy_11_51: - 50.9765625 frac[11,51] + mem[11,51] = 0
 accuracy_14_28: - 50.9765625 frac[14,28] + mem[14,28] = 0
 accuracy_12_52: - 50.9765625 frac[12,52] + mem[12,52] = 0
 accuracy_10_4: - 50.9765625 frac[10,4] + mem[10,4] = 0
 accuracy_13_57: - 50.9765625 frac[13,57] + mem[13,57] = 0
 accuracy_11_9: - 50.9765625 frac[11,9] + mem[11,9] = 0
 accuracy_0_18: - 50.9765625 frac[0,18] + mem[0,18] = 0
 accuracy_14_6: - 50.9765625 frac[14,6] + mem[14,6] = 0
 accuracy_12_18: - 50.9765625 frac[12,18] + mem[12,18] = 0
 accuracy_13_19: - 50.9765625 frac[13,19] + mem[13,19] = 0
 accuracy_2_24: - 50.9765625 frac[2,24] + mem[2,24] = 0
 accuracy_14_40: - 50.9765625 frac[14,40] + mem[14,40] = 0
 accuracy_3_5: - 50.9765625 frac[3,5] + mem[3,5] = 0
 accuracy_1_53: - 50.9765625 frac[1,53] + mem[1,53] = 0
 accuracy_0_46: - 50.9765625 frac[0,46] + mem[0,46] = 0
 accuracy_5_7: - 50.9765625 frac[5,7] + mem[5,7] = 0
 accuracy_3_43: - 50.9765625 frac[3,43] + mem[3,43] = 0
 accuracy_1_47: - 50.9765625 frac[1,47] + mem[1,47] = 0
 accuracy_4_44: - 50.9765625 frac[4,44] + mem[4,44] = 0
 accuracy_8_39: - 50.9765625 frac[8,39] + mem[8,39] = 0
 accuracy_7_17: - 50.9765625 frac[7,17] + mem[7,17] = 0
 accuracy_3_49: - 50.9765625 frac[3,49] + mem[3,49] = 0
 accuracy_9_38: - 50.9765625 frac[9,38] + mem[9,38] = 0
 accuracy_6_46: - 50.9765625 frac[6,46] + mem[6,46] = 0
 accuracy_10_37: - 50.9765625 frac[10,37] + mem[10,37] = 0
 accuracy_8_13: - 50.9765625 frac[8,13] + mem[8,13] = 0
 accuracy_11_40: - 50.9765625 frac[11,40] + mem[11,40] = 0
 accuracy_12_51: - 50.9765625 frac[12,51] + mem[12,51] = 0
 accuracy_10_15: - 50.9765625 frac[10,15] + mem[10,15] = 0
 accuracy_8_19: - 50.9765625 frac[8,19] + mem[8,19] = 0
 accuracy_13_50: - 50.9765625 frac[13,50] + mem[13,50] = 0
 accuracy_11_14: - 50.9765625 frac[11,14] + mem[11,14] = 0
 accuracy_14_9: - 50.9765625 frac[14,9] + mem[14,9] = 0
 accuracy_12_9: - 50.9765625 frac[12,9] + mem[12,9] = 0
 accuracy_13_12: - 50.9765625 frac[13,12] + mem[13,12] = 0
 accuracy_0_15: - 50.9765625 frac[0,15] + mem[0,15] = 0
 accuracy_14_51: - 50.9765625 frac[14,51] + mem[14,51] = 0
 accuracy_1_14: - 50.9765625 frac[1,14] + mem[1,14] = 0
 accuracy_2_13: - 50.9765625 frac[2,13] + mem[2,13] = 0
 accuracy_3_16: - 50.9765625 frac[3,16] + mem[3,16] = 0
 accuracy_1_32: - 50.9765625 frac[1,32] + mem[1,32] = 0
 accuracy_6_15: - 50.9765625 frac[6,15] + mem[6,15] = 0
 accuracy_4_43: - 50.9765625 frac[4,43] + mem[4,43] = 0
 accuracy_2_39: - 50.9765625 frac[2,39] + mem[2,39] = 0
 accuracy_8_46: - 50.9765625 frac[8,46] + mem[8,46] = 0
 accuracy_7_14: - 50.9765625 frac[7,14] + mem[7,14] = 0
 accuracy_3_54: - 50.9765625 frac[3,54] + mem[3,54] = 0
 accuracy_9_47: - 50.9765625 frac[9,47] + mem[9,47] = 0
 accuracy_6_33: - 50.9765625 frac[6,33] + mem[6,33] = 0
 accuracy_10_44: - 50.9765625 frac[10,44] + mem[10,44] = 0
 accuracy_8_52: - 50.9765625 frac[8,52] + mem[8,52] = 0
 accuracy_7_36: - 50.9765625 frac[7,36] + mem[7,36] = 0
 accuracy_5_52: - 50.9765625 frac[5,52] + mem[5,52] = 0
 accuracy_11_33: - 50.9765625 frac[11,33] + mem[11,33] = 0
 accuracy_9_49: - 50.9765625 frac[9,49] + mem[9,49] = 0
 accuracy_6_59: - 50.9765625 frac[6,59] + mem[6,59] = 0
 accuracy_12_42: - 50.9765625 frac[12,42] + mem[12,42] = 0
 accuracy_10_54: - 50.9765625 frac[10,54] + mem[10,54] = 0
 accuracy_8_26: - 50.9765625 frac[8,26] + mem[8,26] = 0
 accuracy_13_43: - 50.9765625 frac[13,43] + mem[13,43] = 0
 accuracy_11_7: - 50.9765625 frac[11,7] + mem[11,7] = 0
 accuracy_9_27: - 50.9765625 frac[9,27] + mem[9,27] = 0
 accuracy_14_16: - 50.9765625 frac[14,16] + mem[14,16] = 0
 accuracy_12_0: - 50.9765625 frac[12,0] + mem[12,0] = 0
 accuracy_10_16: - 50.9765625 frac[10,16] + mem[10,16] = 0
 accuracy_13_5: - 50.9765625 frac[13,5] + mem[13,5] = 0
 accuracy_11_29: - 50.9765625 frac[11,29] + mem[11,29] = 0
 accuracy_0_6: - 50.9765625 frac[0,6] + mem[0,6] = 0
 accuracy_14_58: - 50.9765625 frac[14,58] + mem[14,58] = 0
 accuracy_12_30: - 50.9765625 frac[12,30] + mem[12,30] = 0
 accuracy_13_31: - 50.9765625 frac[13,31] + mem[13,31] = 0
 accuracy_2_4: - 50.9765625 frac[2,4] + mem[2,4] = 0
 accuracy_0_60: - 50.9765625 frac[0,60] + mem[0,60] = 0
 accuracy_3_25: - 50.9765625 frac[3,25] + mem[3,25] = 0
 accuracy_1_57: - 50.9765625 frac[1,57] + mem[1,57] = 0
 accuracy_6_6: - 50.9765625 frac[6,6] + mem[6,6] = 0
 accuracy_4_18: - 50.9765625 frac[4,18] + mem[4,18] = 0
 accuracy_7_7: - 50.9765625 frac[7,7] + mem[7,7] = 0
 accuracy_3_63: - 50.9765625 frac[3,63] + mem[3,63] = 0
 accuracy_6_24: - 50.9765625 frac[6,24] + mem[6,24] = 0
 accuracy_4_56: - 50.9765625 frac[4,56] + mem[4,56] = 0
 accuracy_8_59: - 50.9765625 frac[8,59] + mem[8,59] = 0
 accuracy_5_61: - 50.9765625 frac[5,61] + mem[5,61] = 0
 accuracy_11_38: - 50.9765625 frac[11,38] + mem[11,38] = 0
 accuracy_9_58: - 50.9765625 frac[9,58] + mem[9,58] = 0
 accuracy_6_50: - 50.9765625 frac[6,50] + mem[6,50] = 0
 accuracy_12_33: - 50.9765625 frac[12,33] + mem[12,33] = 0
 accuracy_10_49: - 50.9765625 frac[10,49] + mem[10,49] = 0
 accuracy_8_1: - 50.9765625 frac[8,1] + mem[8,1] = 0
 accuracy_7_59: - 50.9765625 frac[7,59] + mem[7,59] = 0
 accuracy_13_36: - 50.9765625 frac[13,36] + mem[13,36] = 0
 accuracy_11_60: - 50.9765625 frac[11,60] + mem[11,60] = 0
 accuracy_9_28: - 50.9765625 frac[9,28] + mem[9,28] = 0
 accuracy_14_27: - 50.9765625 frac[14,27] + mem[14,27] = 0
 accuracy_12_63: - 50.9765625 frac[12,63] + mem[12,63] = 0
 accuracy_10_27: - 50.9765625 frac[10,27] + mem[10,27] = 0
 accuracy_11_18: - 50.9765625 frac[11,18] + mem[11,18] = 0
 accuracy_14_61: - 50.9765625 frac[14,61] + mem[14,61] = 0
 accuracy_12_21: - 50.9765625 frac[12,21] + mem[12,21] = 0
 accuracy_1_24: - 50.9765625 frac[1,24] + mem[1,24] = 0
 accuracy_13_24: - 50.9765625 frac[13,24] + mem[13,24] = 0
 accuracy_2_31: - 50.9765625 frac[2,31] + mem[2,31] = 0
 accuracy_0_51: - 50.9765625 frac[0,51] + mem[0,51] = 0
 accuracy_14_39: - 50.9765625 frac[14,39] + mem[14,39] = 0
 accuracy_1_50: - 50.9765625 frac[1,50] + mem[1,50] = 0
 accuracy_4_25: - 50.9765625 frac[4,25] + mem[4,25] = 0
 accuracy_0_41: - 50.9765625 frac[0,41] + mem[0,41] = 0
 accuracy_5_28: - 50.9765625 frac[5,28] + mem[5,28] = 0
 accuracy_3_36: - 50.9765625 frac[3,36] + mem[3,36] = 0
 accuracy_6_19: - 50.9765625 frac[6,19] + mem[6,19] = 0
 accuracy_4_39: - 50.9765625 frac[4,39] + mem[4,39] = 0
 accuracy_8_34: - 50.9765625 frac[8,34] + mem[8,34] = 0
 accuracy_7_26: - 50.9765625 frac[7,26] + mem[7,26] = 0
 accuracy_5_38: - 50.9765625 frac[5,38] + mem[5,38] = 0
 accuracy_9_35: - 50.9765625 frac[9,35] + mem[9,35] = 0
 accuracy_6_53: - 50.9765625 frac[6,53] + mem[6,53] = 0
 accuracy_10_56: - 50.9765625 frac[10,56] + mem[10,56] = 0
 accuracy_11_53: - 50.9765625 frac[11,53] + mem[11,53] = 0
 accuracy_9_5: - 50.9765625 frac[9,5] + mem[9,5] = 0
 accuracy_12_54: - 50.9765625 frac[12,54] + mem[12,54] = 0
 accuracy_10_2: - 50.9765625 frac[10,2] + mem[10,2] = 0
 accuracy_13_55: - 50.9765625 frac[13,55] + mem[13,55] = 0
 accuracy_11_11: - 50.9765625 frac[11,11] + mem[11,11] = 0
 accuracy_0_20: - 50.9765625 frac[0,20] + mem[0,20] = 0
 accuracy_14_4: - 50.9765625 frac[14,4] + mem[14,4] = 0
 accuracy_12_12: - 50.9765625 frac[12,12] + mem[12,12] = 0
 accuracy_1_17: - 50.9765625 frac[1,17] + mem[1,17] = 0
 accuracy_13_17: - 50.9765625 frac[13,17] + mem[13,17] = 0
 accuracy_2_22: - 50.9765625 frac[2,22] + mem[2,22] = 0
 accuracy_0_10: - 50.9765625 frac[0,10] + mem[0,10] = 0
 accuracy_14_46: - 50.9765625 frac[14,46] + mem[14,46] = 0
 accuracy_1_11: - 50.9765625 frac[1,11] + mem[1,11] = 0
 accuracy_4_0: - 50.9765625 frac[4,0] + mem[4,0] = 0
 accuracy_2_48: - 50.9765625 frac[2,48] + mem[2,48] = 0
 accuracy_0_32: - 50.9765625 frac[0,32] + mem[0,32] = 0
 accuracy_5_5: - 50.9765625 frac[5,5] + mem[5,5] = 0
 accuracy_3_45: - 50.9765625 frac[3,45] + mem[3,45] = 0
 accuracy_1_45: - 50.9765625 frac[1,45] + mem[1,45] = 0
 accuracy_6_10: - 50.9765625 frac[6,10] + mem[6,10] = 0
 accuracy_4_46: - 50.9765625 frac[4,46] + mem[4,46] = 0
 accuracy_2_42: - 50.9765625 frac[2,42] + mem[2,42] = 0
 accuracy_8_41: - 50.9765625 frac[8,41] + mem[8,41] = 0
 accuracy_7_19: - 50.9765625 frac[7,19] + mem[7,19] = 0
 accuracy_5_47: - 50.9765625 frac[5,47] + mem[5,47] = 0
 accuracy_9_36: - 50.9765625 frac[9,36] + mem[9,36] = 0
 accuracy_4_52: - 50.9765625 frac[4,52] + mem[4,52] = 0
 accuracy_10_35: - 50.9765625 frac[10,35] + mem[10,35] = 0
 accuracy_8_15: - 50.9765625 frac[8,15] + mem[8,15] = 0
 accuracy_7_41: - 50.9765625 frac[7,41] + mem[7,41] = 0
 accuracy_11_42: - 50.9765625 frac[11,42] + mem[11,42] = 0
 accuracy_9_14: - 50.9765625 frac[9,14] + mem[9,14] = 0
 accuracy_12_45: - 50.9765625 frac[12,45] + mem[12,45] = 0
 accuracy_10_13: - 50.9765625 frac[10,13] + mem[10,13] = 0
 accuracy_8_21: - 50.9765625 frac[8,21] + mem[8,21] = 0
 accuracy_13_48: - 50.9765625 frac[13,48] + mem[13,48] = 0
 accuracy_11_0: - 50.9765625 frac[11,0] + mem[11,0] = 0
 accuracy_9_16: - 50.9765625 frac[9,16] + mem[9,16] = 0
 accuracy_14_15: - 50.9765625 frac[14,15] + mem[14,15] = 0
 accuracy_12_11: - 50.9765625 frac[12,11] + mem[12,11] = 0
 accuracy_10_23: - 50.9765625 frac[10,23] + mem[10,23] = 0
 accuracy_13_10: - 50.9765625 frac[13,10] + mem[13,10] = 0
 accuracy_2_17: - 50.9765625 frac[2,17] + mem[2,17] = 0
 accuracy_0_1: - 50.9765625 frac[0,1] + mem[0,1] = 0
 accuracy_14_49: - 50.9765625 frac[14,49] + mem[14,49] = 0
 accuracy_3_12: - 50.9765625 frac[3,12] + mem[3,12] = 0
 accuracy_4_15: - 50.9765625 frac[4,15] + mem[4,15] = 0
 accuracy_2_11: - 50.9765625 frac[2,11] + mem[2,11] = 0
 accuracy_0_39: - 50.9765625 frac[0,39] + mem[0,39] = 0
 accuracy_5_14: - 50.9765625 frac[5,14] + mem[5,14] = 0
 accuracy_1_38: - 50.9765625 frac[1,38] + mem[1,38] = 0
 accuracy_6_13: - 50.9765625 frac[6,13] + mem[6,13] = 0
 accuracy_4_21: - 50.9765625 frac[4,21] + mem[4,21] = 0
 accuracy_2_37: - 50.9765625 frac[2,37] + mem[2,37] = 0
 accuracy_7_8: - 50.9765625 frac[7,8] + mem[7,8] = 0
 accuracy_5_40: - 50.9765625 frac[5,40] + mem[5,40] = 0
 accuracy_3_56: - 50.9765625 frac[3,56] + mem[3,56] = 0
 accuracy_9_45: - 50.9765625 frac[9,45] + mem[9,45] = 0
 accuracy_10_42: - 50.9765625 frac[10,42] + mem[10,42] = 0
 accuracy_8_54: - 50.9765625 frac[8,54] + mem[8,54] = 0
 accuracy_7_38: - 50.9765625 frac[7,38] + mem[7,38] = 0
 accuracy_5_50: - 50.9765625 frac[5,50] + mem[5,50] = 0
 accuracy_11_35: - 50.9765625 frac[11,35] + mem[11,35] = 0
 accuracy_9_55: - 50.9765625 frac[9,55] + mem[9,55] = 0
 accuracy_6_57: - 50.9765625 frac[6,57] + mem[6,57] = 0
 accuracy_12_36: - 50.9765625 frac[12,36] + mem[12,36] = 0
 accuracy_10_52: - 50.9765625 frac[10,52] + mem[10,52] = 0
 accuracy_8_28: - 50.9765625 frac[8,28] + mem[8,28] = 0
 accuracy_7_60: - 50.9765625 frac[7,60] + mem[7,60] = 0
 accuracy_13_41: - 50.9765625 frac[13,41] + mem[13,41] = 0
 accuracy_11_57: - 50.9765625 frac[11,57] + mem[11,57] = 0
 accuracy_9_25: - 50.9765625 frac[9,25] + mem[9,25] = 0
 accuracy_14_22: - 50.9765625 frac[14,22] + mem[14,22] = 0
 accuracy_12_2: - 50.9765625 frac[12,2] + mem[12,2] = 0
 accuracy_10_30: - 50.9765625 frac[10,30] + mem[10,30] = 0
 accuracy_13_3: - 50.9765625 frac[13,3] + mem[13,3] = 0
 accuracy_11_31: - 50.9765625 frac[11,31] + mem[11,31] = 0
 accuracy_0_24: - 50.9765625 frac[0,24] + mem[0,24] = 0
 accuracy_14_56: - 50.9765625 frac[14,56] + mem[14,56] = 0
 accuracy_12_24: - 50.9765625 frac[12,24] + mem[12,24] = 0
 accuracy_1_5: - 50.9765625 frac[1,5] + mem[1,5] = 0
 accuracy_13_29: - 50.9765625 frac[13,29] + mem[13,29] = 0
 accuracy_2_2: - 50.9765625 frac[2,2] + mem[2,2] = 0
 accuracy_14_34: - 50.9765625 frac[14,34] + mem[14,34] = 0
 accuracy_1_63: - 50.9765625 frac[1,63] + mem[1,63] = 0
 accuracy_6_4: - 50.9765625 frac[6,4] + mem[6,4] = 0
 accuracy_4_28: - 50.9765625 frac[4,28] + mem[4,28] = 0
 accuracy_2_60: - 50.9765625 frac[2,60] + mem[2,60] = 0
 accuracy_7_1: - 50.9765625 frac[7,1] + mem[7,1] = 0
 accuracy_3_33: - 50.9765625 frac[3,33] + mem[3,33] = 0
 accuracy_6_30: - 50.9765625 frac[6,30] + mem[6,30] = 0
 accuracy_4_58: - 50.9765625 frac[4,58] + mem[4,58] = 0
 accuracy_8_61: - 50.9765625 frac[8,61] + mem[8,61] = 0
 accuracy_7_31: - 50.9765625 frac[7,31] + mem[7,31] = 0
 accuracy_5_59: - 50.9765625 frac[5,59] + mem[5,59] = 0
 accuracy_9_56: - 50.9765625 frac[9,56] + mem[9,56] = 0
 accuracy_6_48: - 50.9765625 frac[6,48] + mem[6,48] = 0
 accuracy_12_35: - 50.9765625 frac[12,35] + mem[12,35] = 0
 accuracy_10_63: - 50.9765625 frac[10,63] + mem[10,63] = 0
 accuracy_8_3: - 50.9765625 frac[8,3] + mem[8,3] = 0
 accuracy_7_53: - 50.9765625 frac[7,53] + mem[7,53] = 0
 accuracy_13_34: - 50.9765625 frac[13,34] + mem[13,34] = 0
 accuracy_9_2: - 50.9765625 frac[9,2] + mem[9,2] = 0
 accuracy_14_25: - 50.9765625 frac[14,25] + mem[14,25] = 0
 accuracy_12_57: - 50.9765625 frac[12,57] + mem[12,57] = 0
 accuracy_10_25: - 50.9765625 frac[10,25] + mem[10,25] = 0
 accuracy_13_60: - 50.9765625 frac[13,60] + mem[13,60] = 0
 accuracy_11_20: - 50.9765625 frac[11,20] + mem[11,20] = 0
 accuracy_0_31: - 50.9765625 frac[0,31] + mem[0,31] = 0
 accuracy_14_3: - 50.9765625 frac[14,3] + mem[14,3] = 0
 accuracy_12_23: - 50.9765625 frac[12,23] + mem[12,23] = 0
 accuracy_1_30: - 50.9765625 frac[1,30] + mem[1,30] = 0
 accuracy_13_22: - 50.9765625 frac[13,22] + mem[13,22] = 0
 accuracy_0_53: - 50.9765625 frac[0,53] + mem[0,53] = 0
 accuracy_14_37: - 50.9765625 frac[14,37] + mem[14,37] = 0
 accuracy_1_48: - 50.9765625 frac[1,48] + mem[1,48] = 0
 accuracy_4_27: - 50.9765625 frac[4,27] + mem[4,27] = 0
 accuracy_2_55: - 50.9765625 frac[2,55] + mem[2,55] = 0
 accuracy_5_26: - 50.9765625 frac[5,26] + mem[5,26] = 0
 accuracy_3_38: - 50.9765625 frac[3,38] + mem[3,38] = 0
 accuracy_1_42: - 50.9765625 frac[1,42] + mem[1,42] = 0
 accuracy_6_17: - 50.9765625 frac[6,17] + mem[6,17] = 0
 accuracy_4_33: - 50.9765625 frac[4,33] + mem[4,33] = 0
 accuracy_8_36: - 50.9765625 frac[8,36] + mem[8,36] = 0
 accuracy_7_20: - 50.9765625 frac[7,20] + mem[7,20] = 0
 accuracy_5_36: - 50.9765625 frac[5,36] + mem[5,36] = 0
 accuracy_9_33: - 50.9765625 frac[9,33] + mem[9,33] = 0
 accuracy_6_43: - 50.9765625 frac[6,43] + mem[6,43] = 0
 accuracy_10_38: - 50.9765625 frac[10,38] + mem[10,38] = 0
 accuracy_8_10: - 50.9765625 frac[8,10] + mem[8,10] = 0
 accuracy_7_50: - 50.9765625 frac[7,50] + mem[7,50] = 0
 accuracy_11_55: - 50.9765625 frac[11,55] + mem[11,55] = 0
 accuracy_9_11: - 50.9765625 frac[9,11] + mem[9,11] = 0
 accuracy_12_48: - 50.9765625 frac[12,48] + mem[12,48] = 0
 accuracy_10_0: - 50.9765625 frac[10,0] + mem[10,0] = 0
 accuracy_8_16: - 50.9765625 frac[8,16] + mem[8,16] = 0
 accuracy_13_53: - 50.9765625 frac[13,53] + mem[13,53] = 0
 accuracy_11_13: - 50.9765625 frac[11,13] + mem[11,13] = 0
 accuracy_14_10: - 50.9765625 frac[14,10] + mem[14,10] = 0
 accuracy_12_14: - 50.9765625 frac[12,14] + mem[12,14] = 0
 accuracy_13_15: - 50.9765625 frac[13,15] + mem[13,15] = 0
 accuracy_2_20: - 50.9765625 frac[2,20] + mem[2,20] = 0
 accuracy_0_12: - 50.9765625 frac[0,12] + mem[0,12] = 0
 accuracy_14_44: - 50.9765625 frac[14,44] + mem[14,44] = 0
 accuracy_1_9: - 50.9765625 frac[1,9] + mem[1,9] = 0
 accuracy_4_2: - 50.9765625 frac[4,2] + mem[4,2] = 0
 accuracy_1_35: - 50.9765625 frac[1,35] + mem[1,35] = 0
 accuracy_4_40: - 50.9765625 frac[4,40] + mem[4,40] = 0
 accuracy_2_40: - 50.9765625 frac[2,40] + mem[2,40] = 0
 accuracy_8_43: - 50.9765625 frac[8,43] + mem[8,43] = 0
 accuracy_7_13: - 50.9765625 frac[7,13] + mem[7,13] = 0
 accuracy_5_45: - 50.9765625 frac[5,45] + mem[5,45] = 0
 accuracy_3_53: - 50.9765625 frac[3,53] + mem[3,53] = 0
 accuracy_9_42: - 50.9765625 frac[9,42] + mem[9,42] = 0
 accuracy_6_34: - 50.9765625 frac[6,34] + mem[6,34] = 0
 accuracy_4_54: - 50.9765625 frac[4,54] + mem[4,54] = 0
 accuracy_10_33: - 50.9765625 frac[10,33] + mem[10,33] = 0
 accuracy_8_49: - 50.9765625 frac[8,49] + mem[8,49] = 0
 accuracy_5_55: - 50.9765625 frac[5,55] + mem[5,55] = 0
 accuracy_9_12: - 50.9765625 frac[9,12] + mem[9,12] = 0
 accuracy_12_47: - 50.9765625 frac[12,47] + mem[12,47] = 0
 accuracy_10_11: - 50.9765625 frac[10,11] + mem[10,11] = 0
 accuracy_13_46: - 50.9765625 frac[13,46] + mem[13,46] = 0
 accuracy_11_2: - 50.9765625 frac[11,2] + mem[11,2] = 0
 accuracy_9_22: - 50.9765625 frac[9,22] + mem[9,22] = 0
 accuracy_14_13: - 50.9765625 frac[14,13] + mem[14,13] = 0
 accuracy_12_5: - 50.9765625 frac[12,5] + mem[12,5] = 0
 accuracy_10_21: - 50.9765625 frac[10,21] + mem[10,21] = 0
 accuracy_13_8: - 50.9765625 frac[13,8] + mem[13,8] = 0
 accuracy_11_24: - 50.9765625 frac[11,24] + mem[11,24] = 0
 accuracy_14_55: - 50.9765625 frac[14,55] + mem[14,55] = 0
 accuracy_3_14: - 50.9765625 frac[3,14] + mem[3,14] = 0
 accuracy_1_2: - 50.9765625 frac[1,2] + mem[1,2] = 0
 accuracy_4_9: - 50.9765625 frac[4,9] + mem[4,9] = 0
 accuracy_2_9: - 50.9765625 frac[2,9] + mem[2,9] = 0
 accuracy_0_57: - 50.9765625 frac[0,57] + mem[0,57] = 0
 accuracy_1_36: - 50.9765625 frac[1,36] + mem[1,36] = 0
 accuracy_6_3: - 50.9765625 frac[6,3] + mem[6,3] = 0
 accuracy_4_23: - 50.9765625 frac[4,23] + mem[4,23] = 0
 accuracy_2_35: - 50.9765625 frac[2,35] + mem[2,35] = 0
 accuracy_5_22: - 50.9765625 frac[5,22] + mem[5,22] = 0
 accuracy_3_58: - 50.9765625 frac[3,58] + mem[3,58] = 0
 accuracy_6_37: - 50.9765625 frac[6,37] + mem[6,37] = 0
 accuracy_4_61: - 50.9765625 frac[4,61] + mem[4,61] = 0
 accuracy_10_40: - 50.9765625 frac[10,40] + mem[10,40] = 0
 accuracy_8_56: - 50.9765625 frac[8,56] + mem[8,56] = 0
 accuracy_7_32: - 50.9765625 frac[7,32] + mem[7,32] = 0
 accuracy_11_37: - 50.9765625 frac[11,37] + mem[11,37] = 0
 accuracy_9_53: - 50.9765625 frac[9,53] + mem[9,53] = 0
 accuracy_6_63: - 50.9765625 frac[6,63] + mem[6,63] = 0
 accuracy_12_38: - 50.9765625 frac[12,38] + mem[12,38] = 0
 accuracy_10_50: - 50.9765625 frac[10,50] + mem[10,50] = 0
 accuracy_8_30: - 50.9765625 frac[8,30] + mem[8,30] = 0
 accuracy_13_39: - 50.9765625 frac[13,39] + mem[13,39] = 0
 accuracy_11_59: - 50.9765625 frac[11,59] + mem[11,59] = 0
 accuracy_9_31: - 50.9765625 frac[9,31] + mem[9,31] = 0
 accuracy_14_20: - 50.9765625 frac[14,20] + mem[14,20] = 0
 accuracy_12_60: - 50.9765625 frac[12,60] + mem[12,60] = 0
 accuracy_10_28: - 50.9765625 frac[10,28] + mem[10,28] = 0
 accuracy_13_1: - 50.9765625 frac[13,1] + mem[13,1] = 0
 accuracy_11_17: - 50.9765625 frac[11,17] + mem[11,17] = 0
 accuracy_0_26: - 50.9765625 frac[0,26] + mem[0,26] = 0
 accuracy_12_26: - 50.9765625 frac[12,26] + mem[12,26] = 0
 accuracy_1_27: - 50.9765625 frac[1,27] + mem[1,27] = 0
 accuracy_13_27: - 50.9765625 frac[13,27] + mem[13,27] = 0
 accuracy_2_0: - 50.9765625 frac[2,0] + mem[2,0] = 0
 accuracy_0_48: - 50.9765625 frac[0,48] + mem[0,48] = 0
 accuracy_14_32: - 50.9765625 frac[14,32] + mem[14,32] = 0
 accuracy_1_61: - 50.9765625 frac[1,61] + mem[1,61] = 0
 accuracy_4_30: - 50.9765625 frac[4,30] + mem[4,30] = 0
 accuracy_2_58: - 50.9765625 frac[2,58] + mem[2,58] = 0
 mem_tot_cpu1: - mem[0,17] - mem[0,45] - mem[0,36] - mem[0,5] - mem[0,59]
   - mem[0,28] - mem[0,40] - mem[0,19] - mem[0,9] - mem[0,0] - mem[0,38]
   - mem[0,7] - mem[0,61] - mem[0,30] - mem[0,42] - mem[0,11] - mem[0,33]
   - mem[0,56] - mem[0,16] - mem[0,54] - mem[0,44] - mem[0,35] - mem[0,27]
   - mem[0,49] - mem[0,18] - mem[0,46] - mem[0,15] - mem[0,6] - mem[0,60]
   - mem[0,51] - mem[0,41] - mem[0,20] - mem[0,10] - mem[0,32] - mem[0,1]
   - mem[0,39] - mem[0,24] - mem[0,31] - mem[0,53] - mem[0,12] - mem[0,57]
   - mem[0,26] - mem[0,48] + mem_tot_cpu1 = 0
 rows_tot_cpu1: - 3 frac[0,17] - 3 frac[0,45] - 3 frac[0,36] - 3 frac[0,5]
   - 3 frac[0,59] - 3 frac[0,28] - 3 frac[0,40] - 3 frac[0,19]
   - 3 frac[0,9] - 3 frac[0,0] - 3 frac[0,38] - 3 frac[0,7] - 3 frac[0,61]
   - 3 frac[0,30] - 3 frac[0,42] - 3 frac[0,11] - 3 frac[0,33]
   - 3 frac[0,56] - 3 frac[0,16] - 3 frac[0,54] - 3 frac[0,44]
   - 3 frac[0,35] - 3 frac[0,27] - 3 frac[0,49] - 3 frac[0,18]
   - 3 frac[0,46] - 3 frac[0,15] - 3 frac[0,6] - 3 frac[0,60]
   - 3 frac[0,51] - 3 frac[0,41] - 3 frac[0,20] - 3 frac[0,10]
   - 3 frac[0,32] - 3 frac[0,1] - 3 frac[0,39] - 3 frac[0,24]
   - 3 frac[0,31] - 3 frac[0,53] - 3 frac[0,12] - 3 frac[0,57]
   - 3 frac[0,26] - 3 frac[0,48] + rows_tot_cpu1 = 0
 mem_tot_cpu2: - mem[1,40] - mem[1,21] - mem[1,15] - mem[1,33] - mem[1,0]
   - mem[1,25] - mem[1,51] - mem[1,18] - mem[1,52] - mem[1,46] - mem[1,13]
   - mem[1,39] - mem[1,6] - mem[1,56] - mem[1,31] - mem[1,43] - mem[1,16]
   - mem[1,10] - mem[1,44] - mem[1,3] - mem[1,37] - mem[1,4] - mem[1,55]
   - mem[1,41] - mem[1,22] - mem[1,34] - mem[1,59] - mem[1,60] - mem[1,53]
   - mem[1,47] - mem[1,14] - mem[1,32] - mem[1,57] - mem[1,24] - mem[1,50]
   - mem[1,17] - mem[1,11] - mem[1,45] - mem[1,38] - mem[1,5] - mem[1,63]
   - mem[1,30] - mem[1,48] - mem[1,42] - mem[1,9] - mem[1,35] - mem[1,2]
   - mem[1,36] - mem[1,27] - mem[1,61] + mem_tot_cpu2 = 0
 rows_tot_cpu2: - 3 frac[1,40] - 3 frac[1,21] - 3 frac[1,15] - 3 frac[1,33]
   - 3 frac[1,0] - 3 frac[1,25] - 3 frac[1,51] - 3 frac[1,18]
   - 3 frac[1,52] - 3 frac[1,46] - 3 frac[1,13] - 3 frac[1,39]
   - 3 frac[1,6] - 3 frac[1,56] - 3 frac[1,31] - 3 frac[1,43]
   - 3 frac[1,16] - 3 frac[1,10] - 3 frac[1,44] - 3 frac[1,3]
   - 3 frac[1,37] - 3 frac[1,4] - 3 frac[1,55] - 3 frac[1,41]
   - 3 frac[1,22] - 3 frac[1,34] - 3 frac[1,59] - 3 frac[1,60]
   - 3 frac[1,53] - 3 frac[1,47] - 3 frac[1,14] - 3 frac[1,32]
   - 3 frac[1,57] - 3 frac[1,24] - 3 frac[1,50] - 3 frac[1,17]
   - 3 frac[1,11] - 3 frac[1,45] - 3 frac[1,38] - 3 frac[1,5]
   - 3 frac[1,63] - 3 frac[1,30] - 3 frac[1,48] - 3 frac[1,42]
   - 3 frac[1,9] - 3 frac[1,35] - 3 frac[1,2] - 3 frac[1,36] - 3 frac[1,27]
   - 3 frac[1,61] + rows_tot_cpu2 = 0
 mem_tot_cpu3: - mem[2,27] - mem[2,53] - mem[2,47] - mem[2,12] - mem[2,33]
   - mem[2,56] - mem[2,25] - mem[2,45] - mem[2,16] - mem[2,10] - mem[2,36]
   - mem[2,5] - mem[2,63] - mem[2,54] - mem[2,23] - mem[2,43] - mem[2,34]
   - mem[2,3] - mem[2,61] - mem[2,52] - mem[2,21] - mem[2,15] - mem[2,41]
   - mem[2,32] - mem[2,59] - mem[2,24] - mem[2,13] - mem[2,39] - mem[2,4]
   - mem[2,31] - mem[2,22] - mem[2,48] - mem[2,42] - mem[2,17] - mem[2,11]
   - mem[2,37] - mem[2,2] - mem[2,60] - mem[2,55] - mem[2,20] - mem[2,40]
   - mem[2,9] - mem[2,35] - mem[2,0] - mem[2,58] + mem_tot_cpu3 = 0
 rows_tot_cpu3: - 3 frac[2,27] - 3 frac[2,53] - 3 frac[2,47] - 3 frac[2,12]
   - 3 frac[2,33] - 3 frac[2,56] - 3 frac[2,25] - 3 frac[2,45]
   - 3 frac[2,16] - 3 frac[2,10] - 3 frac[2,36] - 3 frac[2,5]
   - 3 frac[2,63] - 3 frac[2,54] - 3 frac[2,23] - 3 frac[2,43]
   - 3 frac[2,34] - 3 frac[2,3] - 3 frac[2,61] - 3 frac[2,52]
   - 3 frac[2,21] - 3 frac[2,15] - 3 frac[2,41] - 3 frac[2,32]
   - 3 frac[2,59] - 3 frac[2,24] - 3 frac[2,13] - 3 frac[2,39]
   - 3 frac[2,4] - 3 frac[2,31] - 3 frac[2,22] - 3 frac[2,48]
   - 3 frac[2,42] - 3 frac[2,17] - 3 frac[2,11] - 3 frac[2,37]
   - 3 frac[2,2] - 3 frac[2,60] - 3 frac[2,55] - 3 frac[2,20]
   - 3 frac[2,40] - 3 frac[2,9] - 3 frac[2,35] - 3 frac[2,0] - 3 frac[2,58]
   + rows_tot_cpu3 = 0
 mem_tot_cpu4: - mem[3,35] - mem[3,2] - mem[3,40] - mem[3,17] - mem[3,22]
   - mem[3,60] - mem[3,31] - mem[3,37] - mem[3,42] - mem[3,13] - mem[3,19]
   - mem[3,57] - mem[3,24] - mem[3,1] - mem[3,39] - mem[3,44] - mem[3,15]
   - mem[3,21] - mem[3,26] - mem[3,46] - mem[3,61] - mem[3,28] - mem[3,34]
   - mem[3,5] - mem[3,43] - mem[3,49] - mem[3,16] - mem[3,54] - mem[3,25]
   - mem[3,63] - mem[3,36] - mem[3,45] - mem[3,12] - mem[3,56] - mem[3,33]
   - mem[3,38] - mem[3,53] - mem[3,14] - mem[3,58] + mem_tot_cpu4 = 0
 rows_tot_cpu4: - 3 frac[3,35] - 3 frac[3,2] - 3 frac[3,40] - 3 frac[3,17]
   - 3 frac[3,22] - 3 frac[3,60] - 3 frac[3,31] - 3 frac[3,37]
   - 3 frac[3,42] - 3 frac[3,13] - 3 frac[3,19] - 3 frac[3,57]
   - 3 frac[3,24] - 3 frac[3,1] - 3 frac[3,39] - 3 frac[3,44]
   - 3 frac[3,15] - 3 frac[3,21] - 3 frac[3,26] - 3 frac[3,46]
   - 3 frac[3,61] - 3 frac[3,28] - 3 frac[3,34] - 3 frac[3,5]
   - 3 frac[3,43] - 3 frac[3,49] - 3 frac[3,16] - 3 frac[3,54]
   - 3 frac[3,25] - 3 frac[3,63] - 3 frac[3,36] - 3 frac[3,45]
   - 3 frac[3,12] - 3 frac[3,56] - 3 frac[3,33] - 3 frac[3,38]
   - 3 frac[3,53] - 3 frac[3,14] - 3 frac[3,58] + rows_tot_cpu4 = 0
 mem_tot_cpu5: - mem[4,5] - mem[4,35] - mem[4,42] - mem[4,48] - mem[4,11]
   - mem[4,17] - mem[4,24] - mem[4,38] - mem[4,45] - mem[4,14] - mem[4,50]
   - mem[4,26] - mem[4,32] - mem[4,1] - mem[4,53] - mem[4,8] - mem[4,22]
   - mem[4,29] - mem[4,59] - mem[4,4] - mem[4,34] - mem[4,3] - mem[4,41]
   - mem[4,55] - mem[4,16] - mem[4,31] - mem[4,37] - mem[4,44] - mem[4,43]
   - mem[4,18] - mem[4,56] - mem[4,25] - mem[4,39] - mem[4,0] - mem[4,46]
   - mem[4,52] - mem[4,15] - mem[4,21] - mem[4,28] - mem[4,58] - mem[4,27]
   - mem[4,33] - mem[4,2] - mem[4,40] - mem[4,54] - mem[4,9] - mem[4,23]
   - mem[4,61] - mem[4,30] + mem_tot_cpu5 = 0
 rows_tot_cpu5: - 3 frac[4,5] - 3 frac[4,35] - 3 frac[4,42] - 3 frac[4,48]
   - 3 frac[4,11] - 3 frac[4,17] - 3 frac[4,24] - 3 frac[4,38]
   - 3 frac[4,45] - 3 frac[4,14] - 3 frac[4,50] - 3 frac[4,26]
   - 3 frac[4,32] - 3 frac[4,1] - 3 frac[4,53] - 3 frac[4,8] - 3 frac[4,22]
   - 3 frac[4,29] - 3 frac[4,59] - 3 frac[4,4] - 3 frac[4,34] - 3 frac[4,3]
   - 3 frac[4,41] - 3 frac[4,55] - 3 frac[4,16] - 3 frac[4,31]
   - 3 frac[4,37] - 3 frac[4,44] - 3 frac[4,43] - 3 frac[4,18]
   - 3 frac[4,56] - 3 frac[4,25] - 3 frac[4,39] - 3 frac[4,0]
   - 3 frac[4,46] - 3 frac[4,52] - 3 frac[4,15] - 3 frac[4,21]
   - 3 frac[4,28] - 3 frac[4,58] - 3 frac[4,27] - 3 frac[4,33]
   - 3 frac[4,2] - 3 frac[4,40] - 3 frac[4,54] - 3 frac[4,9] - 3 frac[4,23]
   - 3 frac[4,61] - 3 frac[4,30] + rows_tot_cpu5 = 0
 mem_tot_cpu6: - mem[5,31] - mem[5,57] - mem[5,24] - mem[5,34] - mem[5,53]
   - mem[5,10] - mem[5,39] - mem[5,32] - mem[5,15] - mem[5,41] - mem[5,51]
   - mem[5,8] - mem[5,18] - mem[5,27] - mem[5,37] - mem[5,4] - mem[5,46]
   - mem[5,13] - mem[5,23] - mem[5,49] - mem[5,25] - mem[5,35] - mem[5,2]
   - mem[5,11] - mem[5,21] - mem[5,63] - mem[5,30] - mem[5,56] - mem[5,7]
   - mem[5,52] - mem[5,61] - mem[5,28] - mem[5,38] - mem[5,5] - mem[5,47]
   - mem[5,14] - mem[5,40] - mem[5,50] - mem[5,59] - mem[5,26] - mem[5,36]
   - mem[5,45] - mem[5,55] - mem[5,22] + mem_tot_cpu6 = 0
 rows_tot_cpu6: - 3 frac[5,31] - 3 frac[5,57] - 3 frac[5,24] - 3 frac[5,34]
   - 3 frac[5,53] - 3 frac[5,10] - 3 frac[5,39] - 3 frac[5,32]
   - 3 frac[5,15] - 3 frac[5,41] - 3 frac[5,51] - 3 frac[5,8]
   - 3 frac[5,18] - 3 frac[5,27] - 3 frac[5,37] - 3 frac[5,4]
   - 3 frac[5,46] - 3 frac[5,13] - 3 frac[5,23] - 3 frac[5,49]
   - 3 frac[5,25] - 3 frac[5,35] - 3 frac[5,2] - 3 frac[5,11]
   - 3 frac[5,21] - 3 frac[5,63] - 3 frac[5,30] - 3 frac[5,56]
   - 3 frac[5,7] - 3 frac[5,52] - 3 frac[5,61] - 3 frac[5,28]
   - 3 frac[5,38] - 3 frac[5,5] - 3 frac[5,47] - 3 frac[5,14]
   - 3 frac[5,40] - 3 frac[5,50] - 3 frac[5,59] - 3 frac[5,26]
   - 3 frac[5,36] - 3 frac[5,45] - 3 frac[5,55] - 3 frac[5,22]
   + rows_tot_cpu6 = 0
 mem_tot_cpu7: - mem[6,28] - mem[6,54] - mem[6,23] - mem[6,41] - mem[6,14]
   - mem[6,27] - mem[6,18] - mem[6,52] - mem[6,21] - mem[6,47] - mem[6,12]
   - mem[6,56] - mem[6,7] - mem[6,25] - mem[6,51] - mem[6,16] - mem[6,42]
   - mem[6,45] - mem[6,2] - mem[6,36] - mem[6,5] - mem[6,31] - mem[6,49]
   - mem[6,22] - mem[6,40] - mem[6,35] - mem[6,26] - mem[6,60] - mem[6,29]
   - mem[6,46] - mem[6,15] - mem[6,33] - mem[6,59] - mem[6,6] - mem[6,24]
   - mem[6,50] - mem[6,19] - mem[6,53] - mem[6,10] - mem[6,13] - mem[6,57]
   - mem[6,4] - mem[6,30] - mem[6,48] - mem[6,17] - mem[6,43] - mem[6,34]
   - mem[6,3] - mem[6,37] - mem[6,63] + mem_tot_cpu7 = 0
 rows_tot_cpu7: - 3 frac[6,28] - 3 frac[6,54] - 3 frac[6,23] - 3 frac[6,41]
   - 3 frac[6,14] - 3 frac[6,27] - 3 frac[6,18] - 3 frac[6,52]
   - 3 frac[6,21] - 3 frac[6,47] - 3 frac[6,12] - 3 frac[6,56]
   - 3 frac[6,7] - 3 frac[6,25] - 3 frac[6,51] - 3 frac[6,16]
   - 3 frac[6,42] - 3 frac[6,45] - 3 frac[6,2] - 3 frac[6,36] - 3 frac[6,5]
   - 3 frac[6,31] - 3 frac[6,49] - 3 frac[6,22] - 3 frac[6,40]
   - 3 frac[6,35] - 3 frac[6,26] - 3 frac[6,60] - 3 frac[6,29]
   - 3 frac[6,46] - 3 frac[6,15] - 3 frac[6,33] - 3 frac[6,59]
   - 3 frac[6,6] - 3 frac[6,24] - 3 frac[6,50] - 3 frac[6,19]
   - 3 frac[6,53] - 3 frac[6,10] - 3 frac[6,13] - 3 frac[6,57]
   - 3 frac[6,4] - 3 frac[6,30] - 3 frac[6,48] - 3 frac[6,17]
   - 3 frac[6,43] - 3 frac[6,34] - 3 frac[6,3] - 3 frac[6,37]
   - 3 frac[6,63] + rows_tot_cpu7 = 0
 mem_tot_cpu8: - mem[7,3] - mem[7,25] - mem[7,55] - mem[7,22] - mem[7,37]
   - mem[7,4] - mem[7,49] - mem[7,16] - mem[7,46] - mem[7,9] - mem[7,39]
   - mem[7,6] - mem[7,51] - mem[7,18] - mem[7,11] - mem[7,33] - mem[7,63]
   - mem[7,0] - mem[7,30] - mem[7,52] - mem[7,23] - mem[7,45] - mem[7,12]
   - mem[7,5] - mem[7,35] - mem[7,2] - mem[7,24] - mem[7,54] - mem[7,17]
   - mem[7,14] - mem[7,36] - mem[7,7] - mem[7,59] - mem[7,26] - mem[7,19]
   - mem[7,41] - mem[7,8] - mem[7,38] - mem[7,60] - mem[7,1] - mem[7,31]
   - mem[7,53] - mem[7,20] - mem[7,50] - mem[7,13] - mem[7,32]
   + mem_tot_cpu8 = 0
 rows_tot_cpu8: - 3 frac[7,3] - 3 frac[7,25] - 3 frac[7,55] - 3 frac[7,22]
   - 3 frac[7,37] - 3 frac[7,4] - 3 frac[7,49] - 3 frac[7,16]
   - 3 frac[7,46] - 3 frac[7,9] - 3 frac[7,39] - 3 frac[7,6] - 3 frac[7,51]
   - 3 frac[7,18] - 3 frac[7,11] - 3 frac[7,33] - 3 frac[7,63]
   - 3 frac[7,0] - 3 frac[7,30] - 3 frac[7,52] - 3 frac[7,23]
   - 3 frac[7,45] - 3 frac[7,12] - 3 frac[7,5] - 3 frac[7,35] - 3 frac[7,2]
   - 3 frac[7,24] - 3 frac[7,54] - 3 frac[7,17] - 3 frac[7,14]
   - 3 frac[7,36] - 3 frac[7,7] - 3 frac[7,59] - 3 frac[7,26]
   - 3 frac[7,19] - 3 frac[7,41] - 3 frac[7,8] - 3 frac[7,38]
   - 3 frac[7,60] - 3 frac[7,1] - 3 frac[7,31] - 3 frac[7,53]
   - 3 frac[7,20] - 3 frac[7,50] - 3 frac[7,13] - 3 frac[7,32]
   + rows_tot_cpu8 = 0
 mem_tot_tor_p41: - mem[8,63] - mem[8,5] - mem[8,38] - mem[8,12]
   - mem[8,18] - mem[8,45] - mem[8,51] - mem[8,25] - mem[8,0] - mem[8,33]
   - mem[8,7] - mem[8,40] - mem[8,14] - mem[8,20] - mem[8,47] - mem[8,53]
   - mem[8,27] - mem[8,60] - mem[8,2] - mem[8,35] - mem[8,9] - mem[8,42]
   - mem[8,48] - mem[8,22] - mem[8,55] - mem[8,4] - mem[8,37] - mem[8,11]
   - mem[8,17] - mem[8,44] - mem[8,50] - mem[8,24] - mem[8,57] - mem[8,31]
   - mem[8,32] - mem[8,6] - mem[8,39] - mem[8,13] - mem[8,19] - mem[8,46]
   - mem[8,52] - mem[8,26] - mem[8,59] - mem[8,1] - mem[8,34] - mem[8,41]
   - mem[8,15] - mem[8,21] - mem[8,54] - mem[8,28] - mem[8,61] - mem[8,3]
   - mem[8,36] - mem[8,10] - mem[8,16] - mem[8,43] - mem[8,49] - mem[8,56]
   - mem[8,30] + mem_tot_tor_p41 = 0
 rows_tot_tor_p41: - 3 frac[8,63] - 3 frac[8,5] - 3 frac[8,38]
   - 3 frac[8,12] - 3 frac[8,18] - 3 frac[8,45] - 3 frac[8,51]
   - 3 frac[8,25] - 3 frac[8,0] - 3 frac[8,33] - 3 frac[8,7] - 3 frac[8,40]
   - 3 frac[8,14] - 3 frac[8,20] - 3 frac[8,47] - 3 frac[8,53]
   - 3 frac[8,27] - 3 frac[8,60] - 3 frac[8,2] - 3 frac[8,35] - 3 frac[8,9]
   - 3 frac[8,42] - 3 frac[8,48] - 3 frac[8,22] - 3 frac[8,55]
   - 3 frac[8,4] - 3 frac[8,37] - 3 frac[8,11] - 3 frac[8,17]
   - 3 frac[8,44] - 3 frac[8,50] - 3 frac[8,24] - 3 frac[8,57]
   - 3 frac[8,31] - 3 frac[8,32] - 3 frac[8,6] - 3 frac[8,39]
   - 3 frac[8,13] - 3 frac[8,19] - 3 frac[8,46] - 3 frac[8,52]
   - 3 frac[8,26] - 3 frac[8,59] - 3 frac[8,1] - 3 frac[8,34]
   - 3 frac[8,41] - 3 frac[8,15] - 3 frac[8,21] - 3 frac[8,54]
   - 3 frac[8,28] - 3 frac[8,61] - 3 frac[8,3] - 3 frac[8,36]
   - 3 frac[8,10] - 3 frac[8,16] - 3 frac[8,43] - 3 frac[8,49]
   - 3 frac[8,56] - 3 frac[8,30] + rows_tot_tor_p41 = 0
 mem_tot_tor_p42: - mem[9,0] - mem[9,39] - mem[9,9] - mem[9,19] - mem[9,40]
   - mem[9,20] - mem[9,59] - mem[9,60] - mem[9,37] - mem[9,15] - mem[9,17]
   - mem[9,46] - mem[9,48] - mem[9,26] - mem[9,57] - mem[9,3] - mem[9,34]
   - mem[9,4] - mem[9,43] - mem[9,13] - mem[9,23] - mem[9,44] - mem[9,54]
   - mem[9,24] - mem[9,63] - mem[9,1] - mem[9,32] - mem[9,10] - mem[9,41]
   - mem[9,21] - mem[9,52] - mem[9,61] - mem[9,38] - mem[9,47] - mem[9,49]
   - mem[9,27] - mem[9,58] - mem[9,28] - mem[9,35] - mem[9,5] - mem[9,36]
   - mem[9,14] - mem[9,16] - mem[9,45] - mem[9,55] - mem[9,25] - mem[9,56]
   - mem[9,2] - mem[9,33] - mem[9,11] - mem[9,42] - mem[9,12] - mem[9,22]
   - mem[9,53] - mem[9,31] + mem_tot_tor_p42 = 0
 rows_tot_tor_p42: - 3 frac[9,0] - 3 frac[9,39] - 3 frac[9,9]
   - 3 frac[9,19] - 3 frac[9,40] - 3 frac[9,20] - 3 frac[9,59]
   - 3 frac[9,60] - 3 frac[9,37] - 3 frac[9,15] - 3 frac[9,17]
   - 3 frac[9,46] - 3 frac[9,48] - 3 frac[9,26] - 3 frac[9,57]
   - 3 frac[9,3] - 3 frac[9,34] - 3 frac[9,4] - 3 frac[9,43] - 3 frac[9,13]
   - 3 frac[9,23] - 3 frac[9,44] - 3 frac[9,54] - 3 frac[9,24]
   - 3 frac[9,63] - 3 frac[9,1] - 3 frac[9,32] - 3 frac[9,10]
   - 3 frac[9,41] - 3 frac[9,21] - 3 frac[9,52] - 3 frac[9,61]
   - 3 frac[9,38] - 3 frac[9,47] - 3 frac[9,49] - 3 frac[9,27]
   - 3 frac[9,58] - 3 frac[9,28] - 3 frac[9,35] - 3 frac[9,5]
   - 3 frac[9,36] - 3 frac[9,14] - 3 frac[9,16] - 3 frac[9,45]
   - 3 frac[9,55] - 3 frac[9,25] - 3 frac[9,56] - 3 frac[9,2]
   - 3 frac[9,33] - 3 frac[9,11] - 3 frac[9,42] - 3 frac[9,12]
   - 3 frac[9,22] - 3 frac[9,53] - 3 frac[9,31] + rows_tot_tor_p42 = 0
 mem_tot_tor_p43: - mem[10,61] - mem[10,7] - mem[10,36] - mem[10,14]
   - mem[10,47] - mem[10,9] - mem[10,48] - mem[10,26] - mem[10,59]
   - mem[10,5] - mem[10,34] - mem[10,22] - mem[10,45] - mem[10,55]
   - mem[10,17] - mem[10,24] - mem[10,57] - mem[10,3] - mem[10,32]
   - mem[10,10] - mem[10,43] - mem[10,53] - mem[10,31] - mem[10,39]
   - mem[10,1] - mem[10,46] - mem[10,8] - mem[10,18] - mem[10,41]
   - mem[10,51] - mem[10,29] - mem[10,58] - mem[10,4] - mem[10,37]
   - mem[10,15] - mem[10,44] - mem[10,54] - mem[10,16] - mem[10,49]
   - mem[10,27] - mem[10,56] - mem[10,2] - mem[10,35] - mem[10,13]
   - mem[10,23] - mem[10,42] - mem[10,52] - mem[10,30] - mem[10,63]
   - mem[10,25] - mem[10,38] - mem[10,0] - mem[10,33] - mem[10,11]
   - mem[10,21] - mem[10,40] - mem[10,50] - mem[10,28] + mem_tot_tor_p43
   = 0
 rows_tot_tor_p43: - 3 frac[10,61] - 3 frac[10,7] - 3 frac[10,36]
   - 3 frac[10,14] - 3 frac[10,47] - 3 frac[10,9] - 3 frac[10,48]
   - 3 frac[10,26] - 3 frac[10,59] - 3 frac[10,5] - 3 frac[10,34]
   - 3 frac[10,22] - 3 frac[10,45] - 3 frac[10,55] - 3 frac[10,17]
   - 3 frac[10,24] - 3 frac[10,57] - 3 frac[10,3] - 3 frac[10,32]
   - 3 frac[10,10] - 3 frac[10,43] - 3 frac[10,53] - 3 frac[10,31]
   - 3 frac[10,39] - 3 frac[10,1] - 3 frac[10,46] - 3 frac[10,8]
   - 3 frac[10,18] - 3 frac[10,41] - 3 frac[10,51] - 3 frac[10,29]
   - 3 frac[10,58] - 3 frac[10,4] - 3 frac[10,37] - 3 frac[10,15]
   - 3 frac[10,44] - 3 frac[10,54] - 3 frac[10,16] - 3 frac[10,49]
   - 3 frac[10,27] - 3 frac[10,56] - 3 frac[10,2] - 3 frac[10,35]
   - 3 frac[10,13] - 3 frac[10,23] - 3 frac[10,42] - 3 frac[10,52]
   - 3 frac[10,30] - 3 frac[10,63] - 3 frac[10,25] - 3 frac[10,38]
   - 3 frac[10,0] - 3 frac[10,33] - 3 frac[10,11] - 3 frac[10,21]
   - 3 frac[10,40] - 3 frac[10,50] - 3 frac[10,28] + rows_tot_tor_p43 = 0
 mem_tot_tor_p44: - mem[11,48] - mem[11,22] - mem[11,41] - mem[11,15]
   - mem[11,46] - mem[11,4] - mem[11,26] - mem[11,39] - mem[11,19]
   - mem[11,50] - mem[11,8] - mem[11,43] - mem[11,1] - mem[11,32]
   - mem[11,6] - mem[11,28] - mem[11,63] - mem[11,21] - mem[11,52]
   - mem[11,10] - mem[11,45] - mem[11,3] - mem[11,25] - mem[11,34]
   - mem[11,56] - mem[11,30] - mem[11,49] - mem[11,23] - mem[11,54]
   - mem[11,12] - mem[11,47] - mem[11,5] - mem[11,27] - mem[11,36]
   - mem[11,16] - mem[11,51] - mem[11,9] - mem[11,40] - mem[11,14]
   - mem[11,33] - mem[11,7] - mem[11,29] - mem[11,38] - mem[11,60]
   - mem[11,18] - mem[11,53] - mem[11,11] - mem[11,42] - mem[11,0]
   - mem[11,35] - mem[11,57] - mem[11,31] - mem[11,20] - mem[11,55]
   - mem[11,13] - mem[11,2] - mem[11,24] - mem[11,37] - mem[11,59]
   - mem[11,17] + mem_tot_tor_p44 = 0
 rows_tot_tor_p44: - 3 frac[11,48] - 3 frac[11,22] - 3 frac[11,41]
   - 3 frac[11,15] - 3 frac[11,46] - 3 frac[11,4] - 3 frac[11,26]
   - 3 frac[11,39] - 3 frac[11,19] - 3 frac[11,50] - 3 frac[11,8]
   - 3 frac[11,43] - 3 frac[11,1] - 3 frac[11,32] - 3 frac[11,6]
   - 3 frac[11,28] - 3 frac[11,63] - 3 frac[11,21] - 3 frac[11,52]
   - 3 frac[11,10] - 3 frac[11,45] - 3 frac[11,3] - 3 frac[11,25]
   - 3 frac[11,34] - 3 frac[11,56] - 3 frac[11,30] - 3 frac[11,49]
   - 3 frac[11,23] - 3 frac[11,54] - 3 frac[11,12] - 3 frac[11,47]
   - 3 frac[11,5] - 3 frac[11,27] - 3 frac[11,36] - 3 frac[11,16]
   - 3 frac[11,51] - 3 frac[11,9] - 3 frac[11,40] - 3 frac[11,14]
   - 3 frac[11,33] - 3 frac[11,7] - 3 frac[11,29] - 3 frac[11,38]
   - 3 frac[11,60] - 3 frac[11,18] - 3 frac[11,53] - 3 frac[11,11]
   - 3 frac[11,42] - 3 frac[11,0] - 3 frac[11,35] - 3 frac[11,57]
   - 3 frac[11,31] - 3 frac[11,20] - 3 frac[11,55] - 3 frac[11,13]
   - 3 frac[11,2] - 3 frac[11,24] - 3 frac[11,37] - 3 frac[11,59]
   - 3 frac[11,17] + rows_tot_tor_p44 = 0
 mem_tot_l1_p41: - mem[12,59] - mem[12,17] - mem[12,50] - mem[12,41]
   - mem[12,7] - mem[12,32] - mem[12,20] - mem[12,53] - mem[12,19]
   - mem[12,44] - mem[12,10] - mem[12,43] - mem[12,1] - mem[12,31]
   - mem[12,34] - mem[12,56] - mem[12,22] - mem[12,55] - mem[12,13]
   - mem[12,46] - mem[12,4] - mem[12,37] - mem[12,3] - mem[12,25]
   - mem[12,58] - mem[12,16] - mem[12,49] - mem[12,15] - mem[12,40]
   - mem[12,6] - mem[12,28] - mem[12,39] - mem[12,61] - mem[12,27]
   - mem[12,52] - mem[12,18] - mem[12,51] - mem[12,9] - mem[12,42]
   - mem[12,0] - mem[12,30] - mem[12,33] - mem[12,63] - mem[12,21]
   - mem[12,54] - mem[12,12] - mem[12,45] - mem[12,11] - mem[12,36]
   - mem[12,2] - mem[12,24] - mem[12,35] - mem[12,57] - mem[12,23]
   - mem[12,48] - mem[12,14] - mem[12,47] - mem[12,5] - mem[12,38]
   - mem[12,60] - mem[12,26] + mem_tot_l1_p41 = 0
 rows_tot_l1_p41: - 3 frac[12,59] - 3 frac[12,17] - 3 frac[12,50]
   - 3 frac[12,41] - 3 frac[12,7] - 3 frac[12,32] - 3 frac[12,20]
   - 3 frac[12,53] - 3 frac[12,19] - 3 frac[12,44] - 3 frac[12,10]
   - 3 frac[12,43] - 3 frac[12,1] - 3 frac[12,31] - 3 frac[12,34]
   - 3 frac[12,56] - 3 frac[12,22] - 3 frac[12,55] - 3 frac[12,13]
   - 3 frac[12,46] - 3 frac[12,4] - 3 frac[12,37] - 3 frac[12,3]
   - 3 frac[12,25] - 3 frac[12,58] - 3 frac[12,16] - 3 frac[12,49]
   - 3 frac[12,15] - 3 frac[12,40] - 3 frac[12,6] - 3 frac[12,28]
   - 3 frac[12,39] - 3 frac[12,61] - 3 frac[12,27] - 3 frac[12,52]
   - 3 frac[12,18] - 3 frac[12,51] - 3 frac[12,9] - 3 frac[12,42]
   - 3 frac[12,0] - 3 frac[12,30] - 3 frac[12,33] - 3 frac[12,63]
   - 3 frac[12,21] - 3 frac[12,54] - 3 frac[12,12] - 3 frac[12,45]
   - 3 frac[12,11] - 3 frac[12,36] - 3 frac[12,2] - 3 frac[12,24]
   - 3 frac[12,35] - 3 frac[12,57] - 3 frac[12,23] - 3 frac[12,48]
   - 3 frac[12,14] - 3 frac[12,47] - 3 frac[12,5] - 3 frac[12,38]
   - 3 frac[12,60] - 3 frac[12,26] + rows_tot_l1_p41 = 0
 mem_tot_l1_p42: - mem[13,32] - mem[13,58] - mem[13,20] - mem[13,51]
   - mem[13,13] - mem[13,44] - mem[13,6] - mem[13,37] - mem[13,63]
   - mem[13,25] - mem[13,56] - mem[13,18] - mem[13,49] - mem[13,11]
   - mem[13,42] - mem[13,4] - mem[13,30] - mem[13,35] - mem[13,61]
   - mem[13,23] - mem[13,54] - mem[13,16] - mem[13,47] - mem[13,9]
   - mem[13,40] - mem[13,2] - mem[13,28] - mem[13,33] - mem[13,59]
   - mem[13,21] - mem[13,52] - mem[13,14] - mem[13,45] - mem[13,7]
   - mem[13,38] - mem[13,0] - mem[13,26] - mem[13,57] - mem[13,19]
   - mem[13,50] - mem[13,12] - mem[13,43] - mem[13,5] - mem[13,31]
   - mem[13,36] - mem[13,24] - mem[13,55] - mem[13,17] - mem[13,48]
   - mem[13,10] - mem[13,41] - mem[13,3] - mem[13,29] - mem[13,34]
   - mem[13,60] - mem[13,22] - mem[13,53] - mem[13,15] - mem[13,46]
   - mem[13,8] - mem[13,39] - mem[13,1] - mem[13,27] + mem_tot_l1_p42 = 0
 rows_tot_l1_p42: - 3 frac[13,32] - 3 frac[13,58] - 3 frac[13,20]
   - 3 frac[13,51] - 3 frac[13,13] - 3 frac[13,44] - 3 frac[13,6]
   - 3 frac[13,37] - 3 frac[13,63] - 3 frac[13,25] - 3 frac[13,56]
   - 3 frac[13,18] - 3 frac[13,49] - 3 frac[13,11] - 3 frac[13,42]
   - 3 frac[13,4] - 3 frac[13,30] - 3 frac[13,35] - 3 frac[13,61]
   - 3 frac[13,23] - 3 frac[13,54] - 3 frac[13,16] - 3 frac[13,47]
   - 3 frac[13,9] - 3 frac[13,40] - 3 frac[13,2] - 3 frac[13,28]
   - 3 frac[13,33] - 3 frac[13,59] - 3 frac[13,21] - 3 frac[13,52]
   - 3 frac[13,14] - 3 frac[13,45] - 3 frac[13,7] - 3 frac[13,38]
   - 3 frac[13,0] - 3 frac[13,26] - 3 frac[13,57] - 3 frac[13,19]
   - 3 frac[13,50] - 3 frac[13,12] - 3 frac[13,43] - 3 frac[13,5]
   - 3 frac[13,31] - 3 frac[13,36] - 3 frac[13,24] - 3 frac[13,55]
   - 3 frac[13,17] - 3 frac[13,48] - 3 frac[13,10] - 3 frac[13,41]
   - 3 frac[13,3] - 3 frac[13,29] - 3 frac[13,34] - 3 frac[13,60]
   - 3 frac[13,22] - 3 frac[13,53] - 3 frac[13,15] - 3 frac[13,46]
   - 3 frac[13,8] - 3 frac[13,39] - 3 frac[13,1] - 3 frac[13,27]
   + rows_tot_l1_p42 = 0
 mem_tot_l2_p4: - mem[14,31] - mem[14,1] - mem[14,43] - mem[14,50]
   - mem[14,19] - mem[14,53] - mem[14,26] - mem[14,60] - mem[14,38]
   - mem[14,7] - mem[14,41] - mem[14,14] - mem[14,48] - mem[14,17]
   - mem[14,59] - mem[14,24] - mem[14,2] - mem[14,36] - mem[14,5]
   - mem[14,47] - mem[14,12] - mem[14,54] - mem[14,23] - mem[14,57]
   - mem[14,35] - mem[14,30] - mem[14,0] - mem[14,42] - mem[14,11]
   - mem[14,45] - mem[14,18] - mem[14,52] - mem[14,21] - mem[14,63]
   - mem[14,33] - mem[14,28] - mem[14,6] - mem[14,40] - mem[14,9]
   - mem[14,51] - mem[14,16] - mem[14,58] - mem[14,27] - mem[14,61]
   - mem[14,39] - mem[14,4] - mem[14,46] - mem[14,15] - mem[14,49]
   - mem[14,22] - mem[14,56] - mem[14,34] - mem[14,25] - mem[14,3]
   - mem[14,37] - mem[14,10] - mem[14,44] - mem[14,13] - mem[14,55]
   - mem[14,20] - mem[14,32] + mem_tot_l2_p4 = 0
 rows_tot_l2_p4: - 3 frac[14,31] - 3 frac[14,1] - 3 frac[14,43]
   - 3 frac[14,50] - 3 frac[14,19] - 3 frac[14,53] - 3 frac[14,26]
   - 3 frac[14,60] - 3 frac[14,38] - 3 frac[14,7] - 3 frac[14,41]
   - 3 frac[14,14] - 3 frac[14,48] - 3 frac[14,17] - 3 frac[14,59]
   - 3 frac[14,24] - 3 frac[14,2] - 3 frac[14,36] - 3 frac[14,5]
   - 3 frac[14,47] - 3 frac[14,12] - 3 frac[14,54] - 3 frac[14,23]
   - 3 frac[14,57] - 3 frac[14,35] - 3 frac[14,30] - 3 frac[14,0]
   - 3 frac[14,42] - 3 frac[14,11] - 3 frac[14,45] - 3 frac[14,18]
   - 3 frac[14,52] - 3 frac[14,21] - 3 frac[14,63] - 3 frac[14,33]
   - 3 frac[14,28] - 3 frac[14,6] - 3 frac[14,40] - 3 frac[14,9]
   - 3 frac[14,51] - 3 frac[14,16] - 3 frac[14,58] - 3 frac[14,27]
   - 3 frac[14,61] - 3 frac[14,39] - 3 frac[14,4] - 3 frac[14,46]
   - 3 frac[14,15] - 3 frac[14,49] - 3 frac[14,22] - 3 frac[14,56]
   - 3 frac[14,34] - 3 frac[14,25] - 3 frac[14,3] - 3 frac[14,37]
   - 3 frac[14,10] - 3 frac[14,44] - 3 frac[14,13] - 3 frac[14,55]
   - 3 frac[14,20] - 3 frac[14,32] + rows_tot_l2_p4 = 0
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
 frac[7,3] frac[5,31] frac[3,35] frac[6,28] frac[8,63] frac[7,25]
 frac[5,57] frac[6,54] frac[10,61] frac[8,5] frac[7,55] frac[13,32]
 frac[11,48] frac[9,0] frac[14,31] frac[12,59] frac[10,7] frac[13,58]
 frac[11,22] frac[0,17] frac[14,1] frac[12,17] frac[13,20] frac[2,27]
 frac[14,43] frac[3,2] frac[4,5] frac[2,53] frac[0,45] frac[5,24]
 frac[3,40] frac[1,40] frac[6,23] frac[4,35] frac[2,47] frac[8,38]
 frac[7,22] frac[5,34] frac[9,39] frac[6,41] frac[10,36] frac[8,12]
 frac[11,41] frac[9,9] frac[12,50] frac[10,14] frac[8,18] frac[13,51]
 frac[11,15] frac[9,19] frac[1,21] frac[13,13] frac[14,50] frac[1,15]
 frac[2,12] frac[0,36] frac[3,17] frac[1,33] frac[6,14] frac[4,42]
 frac[8,45] frac[9,40] frac[4,48] frac[10,47] frac[8,51] frac[7,37]
 frac[5,53] frac[11,46] frac[12,41] frac[10,9] frac[8,25] frac[13,44]
 frac[11,4] frac[9,20] frac[14,19] frac[12,7] frac[13,6] frac[11,26]
 frac[0,5] frac[14,53] frac[1,0] frac[4,11] frac[0,59] frac[5,10]
 frac[3,22] frac[4,17] frac[2,33] frac[7,4] frac[3,60] frac[6,27]
 frac[11,39] frac[9,59] frac[12,32] frac[10,48] frac[8,0] frac[13,37]
 frac[14,26] frac[10,26] frac[13,63] frac[11,19] frac[0,28] frac[14,60]
 frac[12,20] frac[1,25] frac[13,25] frac[14,38] frac[3,31] frac[1,51]
 frac[4,24] frac[2,56] frac[0,40] frac[3,37] frac[6,18] frac[4,38]
 frac[8,33] frac[5,39] frac[9,60] frac[6,52] frac[10,59] frac[8,7]
 frac[7,49] frac[11,50] frac[12,53] frac[10,5] frac[13,56] frac[11,8]
 frac[0,19] frac[14,7] frac[12,19] frac[1,18] frac[13,18] frac[2,25]
 frac[0,9] frac[14,41] frac[1,52] frac[3,42] frac[1,46] frac[6,21]
 frac[4,45] frac[2,45] frac[8,40] frac[7,16] frac[5,32] frac[9,37]
 frac[6,47] frac[10,34] frac[8,14] frac[7,46] frac[11,43] frac[9,15]
 frac[12,44] frac[8,20] frac[13,49] frac[11,1] frac[9,17] frac[14,14]
 frac[12,10] frac[10,22] frac[13,11] frac[2,16] frac[0,0] frac[14,48]
 frac[3,13] frac[1,13] frac[4,14] frac[2,10] frac[0,38] frac[5,15]
 frac[3,19] frac[1,39] frac[6,12] frac[2,36] frac[8,47] frac[7,9]
 frac[5,41] frac[3,57] frac[9,46] frac[4,50] frac[10,45] frac[8,53]
 frac[7,39] frac[5,51] frac[11,32] frac[9,48] frac[6,56] frac[12,43]
 frac[10,55] frac[8,27] frac[13,42] frac[11,6] frac[9,26] frac[14,17]
 frac[12,1] frac[10,17] frac[13,4] frac[11,28] frac[0,7] frac[14,59]
 frac[12,31] frac[1,6] frac[13,30] frac[2,5] frac[0,61] frac[5,8]
 frac[3,24] frac[1,56] frac[6,7] frac[2,63] frac[7,6] frac[5,18] frac[6,25]
 frac[8,60] frac[9,57] frac[6,51] frac[12,34] frac[8,2] frac[13,35]
 frac[11,63] frac[9,3] frac[14,24] frac[12,56] frac[10,24] frac[13,61]
 frac[11,21] frac[0,30] frac[14,2] frac[12,22] frac[1,31] frac[13,23]
 frac[14,36] frac[3,1] frac[4,26] frac[2,54] frac[0,42] frac[5,27]
 frac[3,39] frac[1,43] frac[6,16] frac[4,32] frac[8,35] frac[5,37]
 frac[9,34] frac[6,42] frac[10,57] frac[8,9] frac[7,51] frac[11,52]
 frac[9,4] frac[12,55] frac[10,3] frac[13,54] frac[11,10] frac[14,5]
 frac[12,13] frac[1,16] frac[13,16] frac[2,23] frac[0,11] frac[14,47]
 frac[1,10] frac[4,1] frac[0,33] frac[5,4] frac[3,44] frac[1,44] frac[2,43]
 frac[8,42] frac[7,18] frac[5,46] frac[9,43] frac[6,45] frac[4,53]
 frac[10,32] frac[8,48] frac[11,45] frac[9,13] frac[12,46] frac[10,10]
 frac[8,22] frac[13,47] frac[11,3] frac[9,23] frac[14,12] frac[12,4]
 frac[13,9] frac[11,25] frac[14,54] frac[3,15] frac[1,3] frac[4,8]
 frac[0,56] frac[5,13] frac[3,21] frac[1,37] frac[6,2] frac[4,22]
 frac[2,34] frac[7,11] frac[5,23] frac[9,44] frac[6,36] frac[10,43]
 frac[8,55] frac[7,33] frac[5,49] frac[11,34] frac[9,54] frac[12,37]
 frac[10,53] frac[7,63] frac[13,40] frac[11,56] frac[9,24] frac[14,23]
 frac[12,3] frac[10,31] frac[13,2] frac[11,30] frac[14,57] frac[12,25]
 frac[1,4] frac[13,28] frac[2,3] frac[14,35] frac[3,26] frac[6,5]
 frac[4,29] frac[2,61] frac[7,0] frac[6,31] frac[4,59] frac[7,30]
 frac[9,63] frac[6,49] frac[8,4] frac[7,52] frac[13,33] frac[11,49]
 frac[9,1] frac[14,30] frac[12,58] frac[13,59] frac[11,23] frac[0,16]
 frac[14,0] frac[12,16] frac[13,21] frac[0,54] frac[14,42] frac[1,55]
 frac[4,4] frac[2,52] frac[0,44] frac[5,25] frac[1,41] frac[6,22]
 frac[4,34] frac[8,37] frac[7,23] frac[5,35] frac[9,32] frac[6,40]
 frac[10,39] frac[8,11] frac[7,45] frac[11,54] frac[9,10] frac[12,49]
 frac[10,1] frac[8,17] frac[13,52] frac[11,12] frac[14,11] frac[12,15]
 frac[1,22] frac[13,14] frac[2,21] frac[14,45] frac[4,3] frac[2,15]
 frac[0,35] frac[5,2] frac[3,46] frac[1,34] frac[4,41] frac[2,41]
 frac[8,44] frac[7,12] frac[9,41] frac[6,35] frac[4,55] frac[10,46]
 frac[8,50] frac[11,47] frac[12,40] frac[10,8] frac[8,24] frac[13,45]
 frac[11,5] frac[9,21] frac[14,18] frac[12,6] frac[10,18] frac[13,7]
 frac[11,27] frac[14,52] frac[12,28] frac[5,11] frac[1,59] frac[4,16]
 frac[2,32] frac[7,5] frac[5,21] frac[3,61] frac[6,26] frac[10,41]
 frac[8,57] frac[7,35] frac[5,63] frac[11,36] frac[9,52] frac[6,60]
 frac[12,39] frac[10,51] frac[8,31] frac[13,38] frac[14,21] frac[12,61]
 frac[10,29] frac[13,0] frac[11,16] frac[0,27] frac[14,63] frac[12,27]
 frac[13,26] frac[0,49] frac[14,33] frac[3,28] frac[1,60] frac[4,31]
 frac[2,59] frac[7,2] frac[5,30] frac[3,34] frac[6,29] frac[4,37]
 frac[8,32] frac[7,24] frac[5,56] frac[9,61] frac[10,58] frac[8,6]
 frac[7,54] frac[11,51] frac[14,28] frac[12,52] frac[10,4] frac[13,57]
 frac[11,9] frac[0,18] frac[14,6] frac[12,18] frac[13,19] frac[2,24]
 frac[14,40] frac[3,5] frac[1,53] frac[0,46] frac[5,7] frac[3,43]
 frac[1,47] frac[4,44] frac[8,39] frac[7,17] frac[3,49] frac[9,38]
 frac[6,46] frac[10,37] frac[8,13] frac[11,40] frac[12,51] frac[10,15]
 frac[8,19] frac[13,50] frac[11,14] frac[14,9] frac[12,9] frac[13,12]
 frac[0,15] frac[14,51] frac[1,14] frac[2,13] frac[3,16] frac[1,32]
 frac[6,15] frac[4,43] frac[2,39] frac[8,46] frac[7,14] frac[3,54]
 frac[9,47] frac[6,33] frac[10,44] frac[8,52] frac[7,36] frac[5,52]
 frac[11,33] frac[9,49] frac[6,59] frac[12,42] frac[10,54] frac[8,26]
 frac[13,43] frac[11,7] frac[9,27] frac[14,16] frac[12,0] frac[10,16]
 frac[13,5] frac[11,29] frac[0,6] frac[14,58] frac[12,30] frac[13,31]
 frac[2,4] frac[0,60] frac[3,25] frac[1,57] frac[6,6] frac[4,18] frac[7,7]
 frac[3,63] frac[6,24] frac[4,56] frac[8,59] frac[5,61] frac[11,38]
 frac[9,58] frac[6,50] frac[12,33] frac[10,49] frac[8,1] frac[7,59]
 frac[13,36] frac[11,60] frac[9,28] frac[14,27] frac[12,63] frac[10,27]
 frac[11,18] frac[14,61] frac[12,21] frac[1,24] frac[13,24] frac[2,31]
 frac[0,51] frac[14,39] frac[1,50] frac[4,25] frac[0,41] frac[5,28]
 frac[3,36] frac[6,19] frac[4,39] frac[8,34] frac[7,26] frac[5,38]
 frac[9,35] frac[6,53] frac[10,56] frac[11,53] frac[9,5] frac[12,54]
 frac[10,2] frac[13,55] frac[11,11] frac[0,20] frac[14,4] frac[12,12]
 frac[1,17] frac[13,17] frac[2,22] frac[0,10] frac[14,46] frac[1,11]
 frac[4,0] frac[2,48] frac[0,32] frac[5,5] frac[3,45] frac[1,45] frac[6,10]
 frac[4,46] frac[2,42] frac[8,41] frac[7,19] frac[5,47] frac[9,36]
 frac[4,52] frac[10,35] frac[8,15] frac[7,41] frac[11,42] frac[9,14]
 frac[12,45] frac[10,13] frac[8,21] frac[13,48] frac[11,0] frac[9,16]
 frac[14,15] frac[12,11] frac[10,23] frac[13,10] frac[2,17] frac[0,1]
 frac[14,49] frac[3,12] frac[4,15] frac[2,11] frac[0,39] frac[5,14]
 frac[1,38] frac[6,13] frac[4,21] frac[2,37] frac[7,8] frac[5,40]
 frac[3,56] frac[9,45] frac[10,42] frac[8,54] frac[7,38] frac[5,50]
 frac[11,35] frac[9,55] frac[6,57] frac[12,36] frac[10,52] frac[8,28]
 frac[7,60] frac[13,41] frac[11,57] frac[9,25] frac[14,22] frac[12,2]
 frac[10,30] frac[13,3] frac[11,31] frac[0,24] frac[14,56] frac[12,24]
 frac[1,5] frac[13,29] frac[2,2] frac[14,34] frac[1,63] frac[6,4]
 frac[4,28] frac[2,60] frac[7,1] frac[3,33] frac[6,30] frac[4,58]
 frac[8,61] frac[7,31] frac[5,59] frac[9,56] frac[6,48] frac[12,35]
 frac[10,63] frac[8,3] frac[7,53] frac[13,34] frac[9,2] frac[14,25]
 frac[12,57] frac[10,25] frac[13,60] frac[11,20] frac[0,31] frac[14,3]
 frac[12,23] frac[1,30] frac[13,22] frac[0,53] frac[14,37] frac[1,48]
 frac[4,27] frac[2,55] frac[5,26] frac[3,38] frac[1,42] frac[6,17]
 frac[4,33] frac[8,36] frac[7,20] frac[5,36] frac[9,33] frac[6,43]
 frac[10,38] frac[8,10] frac[7,50] frac[11,55] frac[9,11] frac[12,48]
 frac[10,0] frac[8,16] frac[13,53] frac[11,13] frac[14,10] frac[12,14]
 frac[13,15] frac[2,20] frac[0,12] frac[14,44] frac[1,9] frac[4,2]
 frac[1,35] frac[4,40] frac[2,40] frac[8,43] frac[7,13] frac[5,45]
 frac[3,53] frac[9,42] frac[6,34] frac[4,54] frac[10,33] frac[8,49]
 frac[5,55] frac[9,12] frac[12,47] frac[10,11] frac[13,46] frac[11,2]
 frac[9,22] frac[14,13] frac[12,5] frac[10,21] frac[13,8] frac[11,24]
 frac[14,55] frac[3,14] frac[1,2] frac[4,9] frac[2,9] frac[0,57] frac[1,36]
 frac[6,3] frac[4,23] frac[2,35] frac[5,22] frac[3,58] frac[6,37]
 frac[4,61] frac[10,40] frac[8,56] frac[7,32] frac[11,37] frac[9,53]
 frac[6,63] frac[12,38] frac[10,50] frac[8,30] frac[13,39] frac[11,59]
 frac[9,31] frac[14,20] frac[12,60] frac[10,28] frac[13,1] frac[11,17]
 frac[0,26] frac[12,26] frac[1,27] frac[13,27] frac[2,0] frac[0,48]
 frac[14,32] frac[1,61] frac[4,30] frac[2,58]
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

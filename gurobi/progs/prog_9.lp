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
 cov_0_55: frac[0,55] + frac[12,55] + frac[8,55] + frac[6,55] + frac[13,55]
   + frac[11,55] + frac[14,55] >= 0.8
 cov_1_20: frac[13,20] + frac[5,20] + frac[10,20] + frac[6,20]
   + frac[11,20] >= 0.9
 cov_2_7: frac[10,7] + frac[12,7] + frac[8,7] + frac[14,7] + frac[0,7]
   + frac[13,7] + frac[5,7] >= 0.7
 cov_3_48: frac[11,48] + frac[3,48] + frac[14,48] + frac[9,48] + frac[7,48]
   + frac[13,48] + frac[12,48] >= 0.7
 cov_4_41: frac[12,41] + frac[14,41] + frac[4,41] + frac[2,41] + frac[9,41]
   + frac[10,41] + frac[13,41] >= 0.8
 cov_5_16: frac[7,16] + frac[1,16] + frac[13,16] + frac[12,16]
   + frac[11,16] + frac[14,16] + frac[8,16] >= 0.7
 cov_6_25: frac[8,25] + frac[1,25] + frac[12,25] + frac[3,25] + frac[9,25]
   >= 0.7
 cov_7_45: frac[11,45] + frac[7,45] + frac[14,45] + frac[13,45]
   + frac[3,45] + frac[12,45] + frac[9,45] >= 0.8
 cov_8_17: frac[12,17] + frac[14,17] + frac[10,17] + frac[8,17]
   + frac[1,17] + frac[13,17] + frac[5,17] >= 1
 cov_9_23: frac[6,23] + frac[13,23] + frac[5,23] + frac[11,23]
   + frac[10,23] >= 1
 cov_10_10: frac[11,10] + frac[10,10] + frac[4,10] + frac[13,10]
   + frac[7,10] >= 0.8
 cov_11_42: frac[13,42] + frac[8,42] + frac[14,42] + frac[5,42]
   + frac[12,42] + frac[10,42] + frac[1,42] >= 0.9
 cov_12_32: frac[13,32] + frac[6,32] + frac[12,32] + frac[11,32]
   + frac[8,32] + frac[0,32] + frac[14,32] >= 1
 cov_13_1: frac[11,1] + frac[4,1] + frac[10,1] + frac[7,1] + frac[13,1]
   >= 0.9
 cov_14_42: frac[4,42] + frac[13,42] + frac[8,42] + frac[14,42]
   + frac[12,42] + frac[10,42] + frac[1,42] >= 0.8
 cov_15_57: frac[4,57] + frac[10,57] + frac[7,57] + frac[13,57]
   + frac[11,57] >= 0.8
 cov_16_24: frac[5,24] + frac[10,24] + frac[7,24] + frac[13,24]
   + frac[11,24] >= 0.9
 cov_17_63: frac[8,63] + frac[13,63] + frac[11,63] + frac[7,63]
   + frac[14,63] + frac[12,63] + frac[1,63] >= 0.8
 cov_18_32: frac[13,32] + frac[6,32] + frac[12,32] + frac[11,32]
   + frac[3,32] + frac[9,32] + frac[14,32] >= 1
 cov_19_29: frac[8,29] + frac[1,29] + frac[0,29] >= 0.7
 cov_20_5: frac[8,5] + frac[0,5] + frac[10,5] + frac[14,5] + frac[13,5]
   + frac[5,5] + frac[12,5] >= 0.8
 cov_21_49: frac[1,49] + frac[0,49] + frac[8,49] >= 0.8
 cov_22_56: frac[7,56] + frac[6,56] + frac[11,56] >= 0.9
 cov_23_18: frac[8,18] + frac[6,18] + frac[1,18] + frac[13,18]
   + frac[14,18] + frac[12,18] + frac[11,18] >= 1
 cov_24_24: frac[4,24] + frac[14,24] + frac[10,24] + frac[8,24]
   + frac[1,24] + frac[13,24] + frac[12,24] >= 1
 cov_25_24: frac[3,24] + frac[9,24] + frac[2,24] >= 0.9
 cov_26_22: frac[3,22] + frac[12,22] + frac[8,22] + frac[1,22] + frac[9,22]
   >= 0.7
 cov_27_32: frac[13,32] + frac[12,32] + frac[5,32] + frac[10,32]
   + frac[8,32] + frac[1,32] + frac[14,32] >= 0.7
 cov_28_48: frac[10,48] + frac[14,48] + frac[8,48] + frac[13,48]
   + frac[1,48] + frac[12,48] + frac[5,48] >= 0.7
 cov_29_3: frac[10,3] + frac[11,3] + frac[4,3] + frac[13,3] + frac[6,3]
   >= 0.8
 cov_30_23: frac[2,23] + frac[9,23] + frac[3,23] >= 0.7
 cov_31_16: frac[13,16] + frac[5,16] + frac[12,16] + frac[3,16]
   + frac[14,16] + frac[10,16] + frac[9,16] >= 0.8
 cov_32_29: frac[12,29] + frac[9,29] + frac[14,29] + frac[11,29]
   + frac[7,29] + frac[13,29] + frac[2,29] >= 0.9
 cov_33_8: frac[12,8] + frac[2,8] + frac[1,8] + frac[9,8] + frac[8,8] >= 1
 cov_34_59: frac[10,59] + frac[4,59] + frac[13,59] + frac[7,59]
   + frac[11,59] >= 1
 cov_35_55: frac[10,55] + frac[4,55] + frac[6,55] + frac[13,55]
   + frac[11,55] >= 0.7
 cov_36_27: frac[6,27] + frac[8,27] + frac[11,27] + frac[12,27]
   + frac[14,27] + frac[1,27] + frac[13,27] >= 0.9
 cov_37_29: frac[12,29] + frac[9,29] + frac[14,29] + frac[11,29]
   + frac[7,29] + frac[13,29] + frac[2,29] >= 0.8
 cov_38_0: frac[9,0] + frac[8,0] + frac[0,0] + frac[12,0] + frac[2,0]
   >= 0.8
 cov_39_2: frac[5,2] + frac[10,2] + frac[4,2] >= 0.7
 cov_40_55: frac[0,55] + frac[10,55] + frac[12,55] + frac[8,55]
   + frac[4,55] + frac[13,55] + frac[14,55] >= 0.7
 cov_41_11: frac[13,11] + frac[6,11] + frac[14,11] + frac[11,11]
   + frac[12,11] + frac[2,11] + frac[9,11] >= 0.9
 cov_42_3: frac[9,3] + frac[12,3] + frac[2,3] + frac[8,3] + frac[0,3]
   >= 0.7
 cov_43_13: frac[13,13] + frac[3,13] + frac[12,13] + frac[9,13]
   + frac[5,13] + frac[10,13] + frac[14,13] >= 0.7
 cov_44_47: frac[0,47] + frac[8,47] + frac[9,47] + frac[3,47] + frac[12,47]
   >= 1
 cov_45_27: frac[6,27] + frac[11,27] + frac[10,27] + frac[4,27]
   + frac[13,27] >= 0.9
 cov_46_8: frac[14,8] + frac[12,8] + frac[11,8] + frac[3,8] + frac[9,8]
   + frac[6,8] + frac[13,8] >= 0.8
 cov_47_1: frac[14,1] + frac[12,1] + frac[4,1] + frac[10,1] + frac[1,1]
   + frac[8,1] + frac[13,1] >= 0.7
 cov_48_32: frac[13,32] + frac[12,32] + frac[5,32] + frac[10,32]
   + frac[3,32] + frac[9,32] + frac[14,32] >= 0.9
 cov_49_15: frac[1,15] + frac[12,15] + frac[10,15] + frac[8,15]
   + frac[14,15] + frac[4,15] + frac[13,15] >= 0.7
 cov_50_1: frac[14,1] + frac[12,1] + frac[3,1] + frac[4,1] + frac[9,1]
   + frac[10,1] + frac[13,1] >= 0.7
 cov_51_45: frac[0,45] + frac[8,45] + frac[3,45] + frac[12,45] + frac[9,45]
   >= 0.9
 cov_52_42: frac[0,42] + frac[8,42] + frac[12,42] + frac[2,42] + frac[9,42]
   >= 0.8
 cov_53_16: frac[7,16] + frac[6,16] + frac[11,16] >= 1
 cov_54_33: frac[1,33] + frac[2,33] + frac[8,33] + frac[12,33] + frac[9,33]
   >= 0.9
 cov_55_42: frac[13,42] + frac[0,42] + frac[8,42] + frac[14,42]
   + frac[7,42] + frac[12,42] + frac[11,42] >= 1
 cov_56_2: frac[3,2] + frac[14,2] + frac[6,2] + frac[13,2] + frac[12,2]
   + frac[9,2] + frac[11,2] >= 0.9
 cov_57_55: frac[3,55] + frac[10,55] + frac[12,55] + frac[13,55]
   + frac[9,55] + frac[5,55] + frac[14,55] >= 0.7
 cov_58_55: frac[12,55] + frac[6,55] + frac[13,55] + frac[9,55]
   + frac[2,55] + frac[11,55] + frac[14,55] >= 0.8
 cov_59_63: frac[4,63] + frac[13,63] + frac[11,63] + frac[7,63]
   + frac[10,63] >= 0.7
 cov_60_53: frac[2,53] + frac[14,53] + frac[12,53] + frac[4,53]
   + frac[10,53] + frac[13,53] + frac[9,53] >= 0.7
 cov_61_62: frac[12,62] + frac[10,62] + frac[8,62] + frac[4,62]
   + frac[13,62] + frac[0,62] + frac[14,62] >= 0.9
 cov_62_16: frac[6,16] + frac[13,16] + frac[0,16] + frac[12,16]
   + frac[11,16] + frac[14,16] + frac[8,16] >= 0.7
 cov_63_31: frac[14,31] + frac[12,31] + frac[10,31] + frac[4,31]
   + frac[13,31] + frac[2,31] + frac[9,31] >= 0.8
 cov_64_43: frac[2,43] + frac[9,43] + frac[3,43] >= 0.9
 cov_65_42: frac[4,42] + frac[13,42] + frac[7,42] + frac[11,42]
   + frac[10,42] >= 0.9
 cov_66_27: frac[2,27] + frac[8,27] + frac[12,27] + frac[9,27] + frac[1,27]
   >= 0.8
 cov_67_59: frac[10,59] + frac[13,59] + frac[7,59] + frac[5,59]
   + frac[11,59] >= 0.8
 cov_68_43: frac[14,43] + frac[12,43] + frac[10,43] + frac[4,43]
   + frac[13,43] + frac[0,43] + frac[8,43] >= 0.8
 cov_69_31: frac[5,31] + frac[14,31] + frac[12,31] + frac[10,31]
   + frac[13,31] + frac[2,31] + frac[9,31] >= 0.7
 cov_70_61: frac[11,61] + frac[7,61] + frac[13,61] + frac[2,61]
   + frac[12,61] + frac[9,61] + frac[14,61] >= 0.8
 cov_71_44: frac[13,44] + frac[12,44] + frac[1,44] + frac[8,44]
   + frac[5,44] + frac[10,44] + frac[14,44] >= 1
 cov_72_26: frac[11,26] + frac[14,26] + frac[9,26] + frac[3,26]
   + frac[13,26] + frac[7,26] + frac[12,26] >= 0.9
 cov_73_36: frac[10,36] + frac[11,36] + frac[7,36] + frac[13,36]
   + frac[5,36] >= 1
 cov_74_10: frac[12,10] + frac[11,10] + frac[9,10] + frac[3,10]
   + frac[13,10] + frac[14,10] + frac[7,10] >= 0.9
 cov_75_28: frac[13,28] + frac[12,28] + frac[3,28] + frac[14,28]
   + frac[9,28] + frac[5,28] + frac[10,28] >= 0.7
 cov_76_38: frac[14,38] + frac[4,38] + frac[13,38] + frac[9,38]
   + frac[3,38] + frac[10,38] + frac[12,38] >= 0.8
 cov_77_54: frac[13,54] + frac[14,54] + frac[9,54] + frac[5,54]
   + frac[3,54] + frac[10,54] + frac[12,54] >= 0.7
 cov_78_19: frac[10,19] + frac[4,19] + frac[5,19] >= 0.7
 cov_79_23: frac[6,23] + frac[13,23] + frac[14,23] + frac[11,23]
   + frac[0,23] + frac[12,23] + frac[8,23] >= 1
 cov_80_59: frac[12,59] + frac[0,59] + frac[14,59] + frac[13,59]
   + frac[6,59] + frac[8,59] + frac[11,59] >= 0.7
 cov_81_5: frac[8,5] + frac[0,5] + frac[1,5] >= 0.8
 cov_82_58: frac[13,58] + frac[1,58] + frac[8,58] + frac[12,58]
   + frac[10,58] + frac[14,58] + frac[4,58] >= 0.9
 cov_83_35: frac[4,35] + frac[13,35] + frac[8,35] + frac[14,35]
   + frac[0,35] + frac[10,35] + frac[12,35] >= 0.9
 cov_84_48: frac[11,48] + frac[14,48] + frac[8,48] + frac[13,48]
   + frac[6,48] + frac[12,48] + frac[0,48] >= 0.9
 cov_85_28: frac[6,28] + frac[1,28] + frac[11,28] + frac[13,28]
   + frac[12,28] + frac[14,28] + frac[8,28] >= 0.8
 cov_86_58: frac[13,58] + frac[5,58] + frac[12,58] + frac[10,58]
   + frac[14,58] + frac[9,58] + frac[2,58] >= 0.9
 cov_87_35: frac[13,35] + frac[14,35] + frac[7,35] + frac[9,35]
   + frac[11,35] + frac[12,35] + frac[2,35] >= 0.9
 cov_88_49: frac[2,49] + frac[3,49] + frac[9,49] >= 1
 cov_89_59: frac[6,59] + frac[7,59] + frac[11,59] >= 1
 cov_90_51: frac[8,51] + frac[1,51] + frac[9,51] + frac[12,51] + frac[3,51]
   >= 0.8
 cov_91_63: frac[13,63] + frac[2,63] + frac[11,63] + frac[7,63]
   + frac[9,63] + frac[14,63] + frac[12,63] >= 0.7
 cov_92_20: frac[9,20] + frac[2,20] + frac[3,20] >= 1
 cov_93_3: frac[10,3] + frac[12,3] + frac[4,3] + frac[13,3] + frac[8,3]
   + frac[14,3] + frac[0,3] >= 0.8
 cov_94_35: frac[13,35] + frac[8,35] + frac[14,35] + frac[5,35]
   + frac[0,35] + frac[10,35] + frac[12,35] >= 0.8
 cov_95_27: frac[8,27] + frac[5,27] + frac[0,27] + frac[12,27]
   + frac[14,27] + frac[10,27] + frac[13,27] >= 0.7
 cov_96_18: frac[2,18] + frac[13,18] + frac[14,18] + frac[10,18]
   + frac[12,18] + frac[9,18] + frac[4,18] >= 0.9
 cov_97_35: frac[13,35] + frac[8,35] + frac[14,35] + frac[6,35]
   + frac[11,35] + frac[12,35] + frac[1,35] >= 0.8
 cov_98_55: frac[10,55] + frac[4,55] + frac[5,55] >= 0.9
 cov_99_34: frac[5,34] + frac[10,34] + frac[12,34] + frac[9,34]
   + frac[2,34] + frac[14,34] + frac[13,34] >= 1
 cov_100_62: frac[9,62] + frac[12,62] + frac[10,62] + frac[4,62]
   + frac[2,62] + frac[13,62] + frac[14,62] >= 1
 cov_101_63: frac[13,63] + frac[2,63] + frac[11,63] + frac[9,63]
   + frac[14,63] + frac[12,63] + frac[6,63] >= 1
 cov_102_59: frac[12,59] + frac[9,59] + frac[14,59] + frac[3,59]
   + frac[13,59] + frac[6,59] + frac[11,59] >= 1
 cov_103_6: frac[13,6] + frac[11,6] + frac[12,6] + frac[8,6] + frac[14,6]
   + frac[0,6] + frac[6,6] >= 0.9
 cov_104_9: frac[9,9] + frac[8,9] + frac[12,9] + frac[1,9] + frac[2,9]
   >= 0.7
 cov_105_53: frac[14,53] + frac[12,53] + frac[8,53] + frac[6,53]
   + frac[11,53] + frac[0,53] + frac[13,53] >= 0.9
 cov_106_55: frac[0,55] + frac[10,55] + frac[12,55] + frac[8,55]
   + frac[4,55] + frac[13,55] + frac[14,55] >= 1
 cov_107_10: frac[12,10] + frac[2,10] + frac[9,10] + frac[0,10]
   + frac[8,10] >= 0.9
 cov_108_46: frac[9,46] + frac[12,46] + frac[2,46] + frac[0,46]
   + frac[8,46] >= 0.7
 cov_109_3: frac[11,3] + frac[1,3] + frac[12,3] + frac[13,3] + frac[8,3]
   + frac[14,3] + frac[6,3] >= 0.9
 cov_110_24: frac[3,24] + frac[14,24] + frac[9,24] + frac[7,24]
   + frac[13,24] + frac[12,24] + frac[11,24] >= 0.8
 cov_111_48: frac[10,48] + frac[3,48] + frac[14,48] + frac[9,48]
   + frac[13,48] + frac[12,48] + frac[5,48] >= 0.7
 cov_112_41: frac[11,41] + frac[12,41] + frac[14,41] + frac[2,41]
   + frac[9,41] + frac[7,41] + frac[13,41] >= 0.9
 cov_113_32: frac[13,32] + frac[12,32] + frac[5,32] + frac[10,32]
   + frac[3,32] + frac[9,32] + frac[14,32] >= 0.9
 cov_114_47: frac[10,47] + frac[6,47] + frac[13,47] + frac[11,47]
   + frac[5,47] >= 1
 cov_115_3: frac[7,3] + frac[10,3] + frac[11,3] + frac[4,3] + frac[13,3]
   >= 0.7
 cov_116_18: frac[2,18] + frac[13,18] + frac[7,18] + frac[14,18]
   + frac[12,18] + frac[9,18] + frac[11,18] >= 0.7
 cov_117_27: frac[6,27] + frac[11,27] + frac[10,27] + frac[4,27]
   + frac[13,27] >= 1
 cov_118_4: frac[11,4] + frac[7,4] + frac[13,4] + frac[4,4] + frac[10,4]
   >= 0.9
 cov_119_17: frac[0,17] + frac[12,17] + frac[14,17] + frac[8,17]
   + frac[7,17] + frac[13,17] + frac[11,17] >= 0.9
 cov_120_10: frac[5,10] + frac[12,10] + frac[1,10] + frac[10,10]
   + frac[13,10] + frac[8,10] + frac[14,10] >= 0.8
 cov_121_6: frac[13,6] + frac[5,6] + frac[10,6] + frac[12,6] + frac[8,6]
   + frac[14,6] + frac[0,6] >= 0.7
 cov_122_0: frac[6,0] + frac[13,0] + frac[4,0] + frac[11,0] + frac[10,0]
   >= 0.9
 cov_123_14: frac[10,14] + frac[0,14] + frac[8,14] + frac[14,14]
   + frac[13,14] + frac[5,14] + frac[12,14] >= 0.8
 cov_124_59: frac[12,59] + frac[14,59] + frac[13,59] + frac[1,59]
   + frac[8,59] + frac[7,59] + frac[11,59] >= 1
 cov_125_57: frac[4,57] + frac[10,57] + frac[13,57] + frac[6,57]
   + frac[11,57] >= 0.8
 cov_126_53: frac[14,53] + frac[12,53] + frac[8,53] + frac[6,53]
   + frac[11,53] + frac[0,53] + frac[13,53] >= 0.7
 cov_127_40: frac[13,40] + frac[6,40] + frac[11,40] + frac[5,40]
   + frac[10,40] >= 0.9
 cov_128_7: frac[12,7] + frac[14,7] + frac[13,7] + frac[9,7] + frac[11,7]
   + frac[7,7] + frac[3,7] >= 0.9
 cov_129_6: frac[9,6] + frac[12,6] + frac[2,6] + frac[8,6] + frac[0,6]
   >= 0.7
 cov_130_6: frac[13,6] + frac[11,6] + frac[7,6] + frac[12,6] + frac[8,6]
   + frac[14,6] + frac[0,6] >= 0.7
 cov_131_7: frac[2,7] + frac[9,7] + frac[3,7] >= 0.7
 cov_132_37: frac[5,37] + frac[4,37] + frac[10,37] >= 0.8
 cov_133_19: frac[9,19] + frac[12,19] + frac[3,19] + frac[1,19]
   + frac[8,19] >= 0.7
 cov_134_4: frac[3,4] + frac[9,4] + frac[2,4] >= 1
 cov_135_26: frac[14,26] + frac[10,26] + frac[4,26] + frac[13,26]
   + frac[8,26] + frac[0,26] + frac[12,26] >= 0.7
 cov_136_62: frac[12,62] + frac[10,62] + frac[1,62] + frac[8,62]
   + frac[4,62] + frac[13,62] + frac[14,62] >= 0.7
 cov_137_62: frac[12,62] + frac[10,62] + frac[1,62] + frac[8,62]
   + frac[4,62] + frac[13,62] + frac[14,62] >= 0.8
 cov_138_7: frac[10,7] + frac[12,7] + frac[2,7] + frac[14,7] + frac[4,7]
   + frac[13,7] + frac[9,7] >= 1
 cov_139_18: frac[8,18] + frac[1,18] + frac[13,18] + frac[5,18]
   + frac[14,18] + frac[10,18] + frac[12,18] >= 0.7
 cov_140_34: frac[5,34] + frac[10,34] + frac[11,34] + frac[13,34]
   + frac[6,34] >= 0.7
 cov_141_40: frac[0,40] + frac[8,40] + frac[13,40] + frac[12,40]
   + frac[14,40] + frac[5,40] + frac[10,40] >= 1
 cov_142_59: frac[10,59] + frac[13,59] + frac[6,59] + frac[5,59]
   + frac[11,59] >= 0.7
 cov_143_4: frac[11,4] + frac[3,4] + frac[13,4] + frac[9,4] + frac[12,4]
   + frac[14,4] + frac[6,4] >= 0.7
 cov_144_30: frac[13,30] + frac[11,30] + frac[14,30] + frac[12,30]
   + frac[6,30] + frac[1,30] + frac[8,30] >= 0.9
 cov_145_19: frac[9,19] + frac[14,19] + frac[11,19] + frac[12,19]
   + frac[3,19] + frac[13,19] + frac[7,19] >= 1
 cov_146_39: frac[9,39] + frac[3,39] + frac[2,39] >= 0.9
 cov_147_28: frac[11,28] + frac[7,28] + frac[2,28] + frac[13,28]
   + frac[12,28] + frac[14,28] + frac[9,28] >= 0.7
 cov_148_37: frac[3,37] + frac[9,37] + frac[1,37] + frac[12,37]
   + frac[8,37] >= 0.8
 cov_149_11: frac[3,11] + frac[8,11] + frac[1,11] + frac[12,11]
   + frac[9,11] >= 0.9
 cov_150_55: frac[10,55] + frac[12,55] + frac[13,55] + frac[9,55]
   + frac[2,55] + frac[5,55] + frac[14,55] >= 0.8
 cov_151_1: frac[3,1] + frac[9,1] + frac[2,1] >= 0.9
 cov_152_48: frac[10,48] + frac[14,48] + frac[9,48] + frac[2,48]
   + frac[13,48] + frac[12,48] + frac[5,48] >= 0.8
 cov_153_28: frac[0,28] + frac[13,28] + frac[12,28] + frac[14,28]
   + frac[8,28] + frac[4,28] + frac[10,28] >= 0.8
 cov_154_6: frac[13,6] + frac[9,6] + frac[10,6] + frac[12,6] + frac[2,6]
   + frac[14,6] + frac[4,6] >= 0.7
 cov_155_37: frac[13,37] + frac[9,37] + frac[12,37] + frac[4,37]
   + frac[10,37] + frac[2,37] + frac[14,37] >= 1
 cov_156_32: frac[3,32] + frac[9,32] + frac[2,32] >= 0.8
 cov_157_19: frac[10,19] + frac[11,19] + frac[13,19] + frac[5,19]
   + frac[6,19] >= 0.8
 cov_158_53: frac[4,53] + frac[10,53] + frac[11,53] + frac[7,53]
   + frac[13,53] >= 0.7
 cov_159_60: frac[3,60] + frac[14,60] + frac[9,60] + frac[5,60]
   + frac[10,60] + frac[13,60] + frac[12,60] >= 1
 cov_160_51: frac[8,51] + frac[2,51] + frac[9,51] + frac[12,51]
   + frac[0,51] >= 0.7
 cov_161_42: frac[13,42] + frac[6,42] + frac[8,42] + frac[14,42]
   + frac[12,42] + frac[11,42] + frac[1,42] >= 1
 cov_162_60: frac[14,60] + frac[8,60] + frac[5,60] + frac[10,60]
   + frac[1,60] + frac[13,60] + frac[12,60] >= 1
 cov_163_15: frac[9,15] + frac[3,15] + frac[12,15] + frac[10,15]
   + frac[14,15] + frac[4,15] + frac[13,15] >= 0.8
 cov_164_36: frac[2,36] + frac[14,36] + frac[6,36] + frac[11,36]
   + frac[13,36] + frac[9,36] + frac[12,36] >= 1
 cov_165_62: frac[9,62] + frac[12,62] + frac[3,62] + frac[1,62]
   + frac[8,62] >= 0.9
 cov_166_53: frac[2,53] + frac[12,53] + frac[8,53] + frac[1,53]
   + frac[9,53] >= 0.7
 cov_167_34: frac[12,34] + frac[9,34] + frac[1,34] + frac[3,34]
   + frac[8,34] >= 0.9
 cov_168_17: frac[12,17] + frac[3,17] + frac[9,17] + frac[8,17]
   + frac[1,17] >= 1
 cov_169_27: frac[2,27] + frac[7,27] + frac[11,27] + frac[12,27]
   + frac[9,27] + frac[14,27] + frac[13,27] >= 0.7
 cov_170_59: frac[12,59] + frac[0,59] + frac[10,59] + frac[14,59]
   + frac[13,59] + frac[8,59] + frac[5,59] >= 0.8
 cov_171_55: frac[3,55] + frac[12,55] + frac[6,55] + frac[13,55]
   + frac[9,55] + frac[11,55] + frac[14,55] >= 0.7
 cov_172_30: frac[13,30] + frac[0,30] + frac[11,30] + frac[7,30]
   + frac[14,30] + frac[12,30] + frac[8,30] >= 0.7
 cov_173_29: frac[12,29] + frac[9,29] + frac[14,29] + frac[4,29]
   + frac[10,29] + frac[13,29] + frac[2,29] >= 0.7
 cov_174_56: frac[7,56] + frac[13,56] + frac[11,56] + frac[4,56]
   + frac[10,56] >= 0.9
 cov_175_10: frac[5,10] + frac[12,10] + frac[10,10] + frac[0,10]
   + frac[13,10] + frac[8,10] + frac[14,10] >= 1
 cov_176_29: frac[8,29] + frac[1,29] + frac[0,29] >= 0.8
 cov_177_6: frac[13,6] + frac[9,6] + frac[10,6] + frac[12,6] + frac[2,6]
   + frac[14,6] + frac[4,6] >= 1
 cov_178_1: frac[5,1] + frac[6,1] + frac[11,1] + frac[10,1] + frac[13,1]
   >= 0.9
 cov_179_4: frac[1,4] + frac[8,4] + frac[0,4] >= 0.7
 cov_180_41: frac[12,41] + frac[14,41] + frac[5,41] + frac[3,41]
   + frac[9,41] + frac[10,41] + frac[13,41] >= 0.7
 cov_181_51: frac[13,51] + frac[8,51] + frac[6,51] + frac[11,51]
   + frac[12,51] + frac[14,51] + frac[0,51] >= 0.9
 cov_182_50: frac[12,50] + frac[14,50] + frac[9,50] + frac[3,50]
   + frac[13,50] + frac[5,50] + frac[10,50] >= 0.9
 cov_183_29: frac[12,29] + frac[9,29] + frac[14,29] + frac[4,29]
   + frac[10,29] + frac[13,29] + frac[3,29] >= 0.8
 cov_184_5: frac[4,5] + frac[10,5] + frac[11,5] + frac[7,5] + frac[13,5]
   >= 1
 cov_185_33: frac[2,33] + frac[7,33] + frac[13,33] + frac[14,33]
   + frac[11,33] + frac[12,33] + frac[9,33] >= 0.9
 cov_186_12: frac[8,12] + frac[6,12] + frac[14,12] + frac[11,12]
   + frac[13,12] + frac[12,12] + frac[0,12] >= 0.7
 cov_187_33: frac[7,33] + frac[6,33] + frac[11,33] >= 0.8
 cov_188_40: frac[9,40] + frac[13,40] + frac[12,40] + frac[14,40]
   + frac[4,40] + frac[2,40] + frac[10,40] >= 0.8
 cov_189_0: frac[8,0] + frac[0,0] + frac[14,0] + frac[6,0] + frac[13,0]
   + frac[12,0] + frac[11,0] >= 0.7
 cov_190_30: frac[2,30] + frac[13,30] + frac[14,30] + frac[9,30]
   + frac[5,30] + frac[12,30] + frac[10,30] >= 0.9
 cov_191_17: frac[12,17] + frac[9,17] + frac[14,17] + frac[10,17]
   + frac[13,17] + frac[2,17] + frac[5,17] >= 1
 cov_192_11: frac[3,11] + frac[4,11] + frac[13,11] + frac[14,11]
   + frac[12,11] + frac[9,11] + frac[10,11] >= 1
 cov_193_1: frac[14,1] + frac[12,1] + frac[4,1] + frac[9,1] + frac[10,1]
   + frac[2,1] + frac[13,1] >= 0.8
 cov_194_6: frac[13,6] + frac[11,6] + frac[7,6] + frac[10,6] + frac[4,6]
   >= 0.7
 cov_195_33: frac[7,33] + frac[13,33] + frac[11,33] + frac[4,33]
   + frac[10,33] >= 0.8
 cov_196_21: frac[0,21] + frac[2,21] + frac[9,21] + frac[12,21]
   + frac[8,21] >= 0.7
 cov_197_1: frac[5,1] + frac[11,1] + frac[10,1] + frac[7,1] + frac[13,1]
   >= 0.7
 cov_198_7: frac[10,7] + frac[13,7] + frac[5,7] + frac[11,7] + frac[7,7]
   >= 0.9
 cov_199_25: frac[8,25] + frac[1,25] + frac[2,25] + frac[12,25]
   + frac[9,25] >= 0.8
 cov_200_5: frac[14,5] + frac[6,5] + frac[11,5] + frac[3,5] + frac[13,5]
   + frac[9,5] + frac[12,5] >= 1
 cov_201_30: frac[13,30] + frac[11,30] + frac[7,30] + frac[14,30]
   + frac[9,30] + frac[12,30] + frac[3,30] >= 0.8
 cov_202_3: frac[7,3] + frac[11,3] + frac[12,3] + frac[13,3] + frac[8,3]
   + frac[14,3] + frac[0,3] >= 0.7
 cov_203_46: frac[11,46] + frac[1,46] + frac[12,46] + frac[6,46]
   + frac[8,46] + frac[14,46] + frac[13,46] >= 0.7
 cov_204_49: frac[13,49] + frac[6,49] + frac[11,49] + frac[12,49]
   + frac[0,49] + frac[14,49] + frac[8,49] >= 0.8
 cov_205_2: frac[13,2] + frac[5,2] + frac[7,2] + frac[10,2] + frac[11,2]
   >= 0.9
 cov_206_34: frac[12,34] + frac[11,34] + frac[1,34] + frac[8,34]
   + frac[14,34] + frac[13,34] + frac[6,34] >= 0.9
 cov_207_23: frac[6,23] + frac[11,23] + frac[7,23] >= 0.9
 cov_208_5: frac[2,5] + frac[14,5] + frac[6,5] + frac[11,5] + frac[13,5]
   + frac[9,5] + frac[12,5] >= 0.9
 cov_209_21: frac[3,21] + frac[13,21] + frac[9,21] + frac[5,21]
   + frac[14,21] + frac[12,21] + frac[10,21] >= 0.7
 cov_210_37: frac[7,37] + frac[13,37] + frac[12,37] + frac[8,37]
   + frac[0,37] + frac[14,37] + frac[11,37] >= 1
 cov_211_46: frac[9,46] + frac[12,46] + frac[2,46] + frac[0,46]
   + frac[8,46] >= 0.8
 cov_212_0: frac[9,0] + frac[1,0] + frac[8,0] + frac[12,0] + frac[3,0]
   >= 0.7
 cov_213_18: frac[8,18] + frac[2,18] + frac[0,18] + frac[12,18]
   + frac[9,18] >= 1
 cov_214_40: frac[3,40] + frac[1,40] + frac[9,40] + frac[8,40]
   + frac[12,40] >= 0.8
 cov_215_26: frac[11,26] + frac[10,26] + frac[4,26] + frac[6,26]
   + frac[13,26] >= 0.7
 cov_216_22: frac[10,22] + frac[12,22] + frac[8,22] + frac[4,22]
   + frac[1,22] + frac[14,22] + frac[13,22] >= 0.9
 cov_217_55: frac[10,55] + frac[12,55] + frac[13,55] + frac[9,55]
   + frac[2,55] + frac[5,55] + frac[14,55] >= 0.8
 cov_218_57: frac[5,57] + frac[10,57] + frac[13,57] + frac[6,57]
   + frac[11,57] >= 0.7
 cov_219_44: frac[7,44] + frac[13,44] + frac[12,44] + frac[3,44]
   + frac[9,44] + frac[14,44] + frac[11,44] >= 1
 cov_220_12: frac[4,12] + frac[10,12] + frac[6,12] + frac[11,12]
   + frac[13,12] >= 0.7
 cov_221_23: frac[13,23] + frac[5,23] + frac[11,23] + frac[7,23]
   + frac[10,23] >= 0.7
 cov_222_20: frac[13,20] + frac[9,20] + frac[12,20] + frac[4,20]
   + frac[10,20] + frac[3,20] + frac[14,20] >= 0.8
 cov_223_55: frac[0,55] + frac[12,55] + frac[8,55] + frac[6,55]
   + frac[13,55] + frac[11,55] + frac[14,55] >= 0.9
 cov_224_38: frac[14,38] + frac[6,38] + frac[13,38] + frac[9,38]
   + frac[11,38] + frac[3,38] + frac[12,38] >= 1
 cov_225_62: frac[12,62] + frac[10,62] + frac[8,62] + frac[4,62]
   + frac[13,62] + frac[0,62] + frac[14,62] >= 0.7
 cov_226_20: frac[13,20] + frac[12,20] + frac[8,20] + frac[6,20]
   + frac[1,20] + frac[11,20] + frac[14,20] >= 1
 cov_227_47: frac[6,47] + frac[8,47] + frac[14,47] + frac[13,47]
   + frac[11,47] + frac[1,47] + frac[12,47] >= 0.9
 cov_228_12: frac[2,12] + frac[10,12] + frac[14,12] + frac[13,12]
   + frac[12,12] + frac[9,12] + frac[5,12] >= 1
 cov_229_23: frac[13,23] + frac[9,23] + frac[14,23] + frac[11,23]
   + frac[7,23] + frac[3,23] + frac[12,23] >= 1
 cov_230_41: frac[12,41] + frac[14,41] + frac[5,41] + frac[2,41]
   + frac[9,41] + frac[10,41] + frac[13,41] >= 0.8
 cov_231_11: frac[13,11] + frac[8,11] + frac[14,11] + frac[5,11]
   + frac[1,11] + frac[12,11] + frac[10,11] >= 1
 cov_232_46: frac[11,46] + frac[9,46] + frac[12,46] + frac[3,46]
   + frac[6,46] + frac[14,46] + frac[13,46] >= 1
 cov_233_51: frac[8,51] + frac[2,51] + frac[9,51] + frac[12,51]
   + frac[0,51] >= 0.7
 cov_234_46: frac[11,46] + frac[9,46] + frac[12,46] + frac[3,46]
   + frac[6,46] + frac[14,46] + frac[13,46] >= 1
 cov_235_38: frac[14,38] + frac[13,38] + frac[9,38] + frac[11,38]
   + frac[7,38] + frac[3,38] + frac[12,38] >= 0.8
 cov_236_16: frac[2,16] + frac[13,16] + frac[12,16] + frac[4,16]
   + frac[14,16] + frac[10,16] + frac[9,16] >= 0.8
 cov_237_29: frac[9,29] + frac[2,29] + frac[3,29] >= 0.9
 cov_238_27: frac[7,27] + frac[11,27] + frac[10,27] + frac[4,27]
   + frac[13,27] >= 0.8
 cov_239_2: frac[14,2] + frac[13,2] + frac[10,2] + frac[12,2] + frac[2,2]
   + frac[9,2] + frac[4,2] >= 1
 cov_240_44: frac[12,44] + frac[1,44] + frac[9,44] + frac[8,44]
   + frac[2,44] >= 0.8
 cov_241_13: frac[13,13] + frac[3,13] + frac[12,13] + frac[9,13]
   + frac[5,13] + frac[10,13] + frac[14,13] >= 1
 cov_242_7: frac[10,7] + frac[12,7] + frac[14,7] + frac[13,7] + frac[9,7]
   + frac[5,7] + frac[3,7] >= 0.9
 cov_243_24: frac[3,24] + frac[9,24] + frac[2,24] >= 0.8
 cov_244_6: frac[13,6] + frac[11,6] + frac[10,6] + frac[4,6] + frac[6,6]
   >= 0.8
 cov_245_2: frac[8,2] + frac[0,2] + frac[12,2] + frac[2,2] + frac[9,2]
   >= 1
 cov_246_33: frac[1,33] + frac[8,33] + frac[7,33] + frac[13,33]
   + frac[14,33] + frac[11,33] + frac[12,33] >= 1
 cov_247_60: frac[9,60] + frac[8,60] + frac[0,60] + frac[2,60]
   + frac[12,60] >= 1
 cov_248_2: frac[8,2] + frac[14,2] + frac[0,2] + frac[13,2] + frac[10,2]
   + frac[12,2] + frac[4,2] >= 0.7
 cov_249_55: frac[7,55] + frac[12,55] + frac[8,55] + frac[1,55]
   + frac[13,55] + frac[11,55] + frac[14,55] >= 0.9
 cov_250_32: frac[13,32] + frac[6,32] + frac[11,32] + frac[4,32]
   + frac[10,32] >= 0.9
 cov_251_39: frac[11,39] + frac[5,39] + frac[7,39] + frac[10,39]
   + frac[13,39] >= 0.8
 cov_252_13: frac[13,13] + frac[12,13] + frac[9,13] + frac[2,13]
   + frac[11,13] + frac[7,13] + frac[14,13] >= 0.9
 cov_253_14: frac[10,14] + frac[6,14] + frac[13,14] + frac[11,14]
   + frac[5,14] >= 0.9
 cov_254_60: frac[14,60] + frac[8,60] + frac[5,60] + frac[10,60]
   + frac[1,60] + frac[13,60] + frac[12,60] >= 0.7
 cov_255_28: frac[6,28] + frac[11,28] + frac[13,28] + frac[5,28]
   + frac[10,28] >= 0.8
 cov_256_39: frac[9,39] + frac[3,39] + frac[2,39] >= 1
 cov_257_58: frac[13,58] + frac[6,58] + frac[12,58] + frac[11,58]
   + frac[14,58] + frac[9,58] + frac[2,58] >= 1
 cov_258_57: frac[14,57] + frac[8,57] + frac[7,57] + frac[13,57]
   + frac[1,57] + frac[11,57] + frac[12,57] >= 0.8
 cov_259_10: frac[11,10] + frac[6,10] + frac[7,10] >= 0.9
 cov_260_58: frac[13,58] + frac[12,58] + frac[10,58] + frac[14,58]
   + frac[9,58] + frac[4,58] + frac[2,58] >= 0.8
 cov_261_41: frac[6,41] + frac[11,41] + frac[12,41] + frac[14,41]
   + frac[0,41] + frac[8,41] + frac[13,41] >= 0.8
 cov_262_20: frac[13,20] + frac[9,20] + frac[12,20] + frac[4,20]
   + frac[10,20] + frac[3,20] + frac[14,20] >= 1
 cov_263_45: frac[0,45] + frac[8,45] + frac[10,45] + frac[14,45]
   + frac[13,45] + frac[12,45] + frac[5,45] >= 1
 cov_264_3: frac[9,3] + frac[10,3] + frac[12,3] + frac[3,3] + frac[13,3]
   + frac[14,3] + frac[5,3] >= 1
 cov_265_24: frac[14,24] + frac[8,24] + frac[7,24] + frac[1,24]
   + frac[13,24] + frac[12,24] + frac[11,24] >= 0.8
 cov_266_23: frac[13,23] + frac[14,23] + frac[11,23] + frac[7,23]
   + frac[0,23] + frac[12,23] + frac[8,23] >= 0.8
 cov_267_29: frac[12,29] + frac[9,29] + frac[8,29] + frac[1,29]
   + frac[2,29] >= 1
 cov_268_56: frac[2,56] + frac[3,56] + frac[9,56] >= 0.8
 cov_269_4: frac[11,4] + frac[7,4] + frac[13,4] + frac[5,4] + frac[10,4]
   >= 0.9
 cov_270_47: frac[10,47] + frac[0,47] + frac[8,47] + frac[14,47]
   + frac[4,47] + frac[13,47] + frac[12,47] >= 0.7
 cov_271_49: frac[13,49] + frac[6,49] + frac[11,49] + frac[12,49]
   + frac[0,49] + frac[14,49] + frac[8,49] >= 1
 cov_272_4: frac[3,4] + frac[13,4] + frac[9,4] + frac[5,4] + frac[12,4]
   + frac[10,4] + frac[14,4] >= 0.7
 cov_273_34: frac[12,34] + frac[11,34] + frac[1,34] + frac[8,34]
   + frac[14,34] + frac[13,34] + frac[6,34] >= 0.8
 cov_274_19: frac[9,19] + frac[14,19] + frac[10,19] + frac[12,19]
   + frac[3,19] + frac[4,19] + frac[13,19] >= 0.7
 cov_275_35: frac[13,35] + frac[8,35] + frac[14,35] + frac[5,35]
   + frac[0,35] + frac[10,35] + frac[12,35] >= 0.9
 cov_276_58: frac[13,58] + frac[7,58] + frac[11,58] + frac[10,58]
   + frac[4,58] >= 0.7
 cov_277_11: frac[3,11] + frac[13,11] + frac[14,11] + frac[5,11]
   + frac[12,11] + frac[9,11] + frac[10,11] >= 1
 cov_278_48: frac[10,48] + frac[14,48] + frac[8,48] + frac[13,48]
   + frac[12,48] + frac[5,48] + frac[0,48] >= 0.9
 cov_279_60: frac[14,60] + frac[9,60] + frac[11,60] + frac[7,60]
   + frac[2,60] + frac[13,60] + frac[12,60] >= 1
 cov_280_52: frac[6,52] + frac[11,52] + frac[2,52] + frac[13,52]
   + frac[14,52] + frac[9,52] + frac[12,52] >= 1
 cov_281_52: frac[13,52] + frac[3,52] + frac[14,52] + frac[9,52]
   + frac[12,52] + frac[5,52] + frac[10,52] >= 0.9
 cov_282_55: frac[0,55] + frac[10,55] + frac[12,55] + frac[8,55]
   + frac[4,55] + frac[13,55] + frac[14,55] >= 1
 cov_283_51: frac[8,51] + frac[9,51] + frac[12,51] + frac[0,51]
   + frac[3,51] >= 0.8
 cov_284_39: frac[5,39] + frac[1,39] + frac[10,39] + frac[12,39]
   + frac[8,39] + frac[14,39] + frac[13,39] >= 0.8
 cov_285_39: frac[11,39] + frac[7,39] + frac[10,39] + frac[4,39]
   + frac[13,39] >= 1
 cov_286_55: frac[7,55] + frac[0,55] + frac[12,55] + frac[8,55]
   + frac[13,55] + frac[11,55] + frac[14,55] >= 1
 cov_287_23: frac[6,23] + frac[13,23] + frac[2,23] + frac[9,23]
   + frac[14,23] + frac[11,23] + frac[12,23] >= 0.9
 cov_288_44: frac[13,44] + frac[12,44] + frac[0,44] + frac[8,44]
   + frac[5,44] + frac[10,44] + frac[14,44] >= 1
 cov_289_46: frac[1,46] + frac[12,46] + frac[10,46] + frac[8,46]
   + frac[14,46] + frac[4,46] + frac[13,46] >= 0.7
 cov_290_46: frac[5,46] + frac[12,46] + frac[10,46] + frac[0,46]
   + frac[8,46] + frac[14,46] + frac[13,46] >= 0.9
 cov_291_50: frac[12,50] + frac[14,50] + frac[11,50] + frac[8,50]
   + frac[13,50] + frac[1,50] + frac[7,50] >= 0.8
 cov_292_9: frac[10,9] + frac[7,9] + frac[13,9] + frac[11,9] + frac[5,9]
   >= 0.7
 cov_293_18: frac[2,18] + frac[13,18] + frac[14,18] + frac[10,18]
   + frac[12,18] + frac[9,18] + frac[4,18] >= 0.9
 cov_294_32: frac[6,32] + frac[11,32] + frac[7,32] >= 0.9
 cov_295_52: frac[11,52] + frac[7,52] + frac[13,52] + frac[5,52]
   + frac[10,52] >= 1
 cov_296_15: frac[9,15] + frac[3,15] + frac[2,15] >= 0.9
 cov_297_22: frac[3,22] + frac[2,22] + frac[9,22] >= 0.7
 cov_298_54: frac[6,54] + frac[13,54] + frac[14,54] + frac[9,54]
   + frac[11,54] + frac[3,54] + frac[12,54] >= 0.7
 cov_299_22: frac[8,22] + frac[1,22] + frac[0,22] >= 1
 cov_300_25: frac[8,25] + frac[1,25] + frac[2,25] + frac[12,25]
   + frac[9,25] >= 0.7
 cov_301_53: frac[5,53] + frac[14,53] + frac[12,53] + frac[8,53]
   + frac[10,53] + frac[1,53] + frac[13,53] >= 0.9
 cov_302_26: frac[14,26] + frac[10,26] + frac[13,26] + frac[8,26]
   + frac[5,26] + frac[0,26] + frac[12,26] >= 0.7
 cov_303_16: frac[7,16] + frac[6,16] + frac[11,16] >= 1
 cov_304_46: frac[9,46] + frac[12,46] + frac[2,46] + frac[10,46]
   + frac[14,46] + frac[4,46] + frac[13,46] >= 0.9
 cov_305_11: frac[4,11] + frac[13,11] + frac[7,11] + frac[11,11]
   + frac[10,11] >= 0.9
 cov_306_54: frac[1,54] + frac[13,54] + frac[14,54] + frac[11,54]
   + frac[7,54] + frac[12,54] + frac[8,54] >= 0.7
 cov_307_10: frac[5,10] + frac[11,10] + frac[10,10] + frac[6,10]
   + frac[13,10] >= 0.8
 cov_308_56: frac[12,56] + frac[0,56] + frac[3,56] + frac[9,56]
   + frac[8,56] >= 0.7
 cov_309_16: frac[7,16] + frac[2,16] + frac[13,16] + frac[12,16]
   + frac[11,16] + frac[14,16] + frac[9,16] >= 0.8
 cov_310_17: frac[7,17] + frac[6,17] + frac[11,17] >= 0.7
 cov_311_38: frac[8,38] + frac[0,38] + frac[1,38] >= 0.9
 cov_312_16: frac[13,16] + frac[5,16] + frac[12,16] + frac[3,16]
   + frac[14,16] + frac[10,16] + frac[9,16] >= 0.8
 cov_313_36: frac[2,36] + frac[9,36] + frac[12,36] + frac[8,36]
   + frac[1,36] >= 0.8
 cov_314_9: frac[9,9] + frac[7,9] + frac[13,9] + frac[11,9] + frac[14,9]
   + frac[12,9] + frac[2,9] >= 1
 cov_315_7: frac[12,7] + frac[14,7] + frac[13,7] + frac[9,7] + frac[11,7]
   + frac[7,7] + frac[3,7] >= 0.7
 cov_316_15: frac[11,15] + frac[7,15] + frac[5,15] + frac[10,15]
   + frac[13,15] >= 0.8
 cov_317_54: frac[13,54] + frac[14,54] + frac[9,54] + frac[3,54]
   + frac[10,54] + frac[12,54] + frac[4,54] >= 1
 cov_318_43: frac[14,43] + frac[11,43] + frac[12,43] + frac[13,43]
   + frac[0,43] + frac[6,43] + frac[8,43] >= 0.8
 cov_319_29: frac[4,29] + frac[10,29] + frac[6,29] + frac[11,29]
   + frac[13,29] >= 0.7
 accuracy_7_3: - 50.9765625 frac[7,3] + mem[7,3] = 0
 accuracy_5_31: - 50.9765625 frac[5,31] + mem[5,31] = 0
 accuracy_6_28: - 50.9765625 frac[6,28] + mem[6,28] = 0
 accuracy_8_63: - 50.9765625 frac[8,63] + mem[8,63] = 0
 accuracy_5_57: - 50.9765625 frac[5,57] + mem[5,57] = 0
 accuracy_9_62: - 50.9765625 frac[9,62] + mem[9,62] = 0
 accuracy_6_54: - 50.9765625 frac[6,54] + mem[6,54] = 0
 accuracy_8_5: - 50.9765625 frac[8,5] + mem[8,5] = 0
 accuracy_7_55: - 50.9765625 frac[7,55] + mem[7,55] = 0
 accuracy_13_32: - 50.9765625 frac[13,32] + mem[13,32] = 0
 accuracy_11_48: - 50.9765625 frac[11,48] + mem[11,48] = 0
 accuracy_9_0: - 50.9765625 frac[9,0] + mem[9,0] = 0
 accuracy_14_31: - 50.9765625 frac[14,31] + mem[14,31] = 0
 accuracy_12_59: - 50.9765625 frac[12,59] + mem[12,59] = 0
 accuracy_10_7: - 50.9765625 frac[10,7] + mem[10,7] = 0
 accuracy_13_58: - 50.9765625 frac[13,58] + mem[13,58] = 0
 accuracy_0_17: - 50.9765625 frac[0,17] + mem[0,17] = 0
 accuracy_14_1: - 50.9765625 frac[14,1] + mem[14,1] = 0
 accuracy_12_17: - 50.9765625 frac[12,17] + mem[12,17] = 0
 accuracy_1_28: - 50.9765625 frac[1,28] + mem[1,28] = 0
 accuracy_13_20: - 50.9765625 frac[13,20] + mem[13,20] = 0
 accuracy_2_27: - 50.9765625 frac[2,27] + mem[2,27] = 0
 accuracy_0_55: - 50.9765625 frac[0,55] + mem[0,55] = 0
 accuracy_14_43: - 50.9765625 frac[14,43] + mem[14,43] = 0
 accuracy_3_2: - 50.9765625 frac[3,2] + mem[3,2] = 0
 accuracy_1_54: - 50.9765625 frac[1,54] + mem[1,54] = 0
 accuracy_4_5: - 50.9765625 frac[4,5] + mem[4,5] = 0
 accuracy_2_53: - 50.9765625 frac[2,53] + mem[2,53] = 0
 accuracy_0_45: - 50.9765625 frac[0,45] + mem[0,45] = 0
 accuracy_5_24: - 50.9765625 frac[5,24] + mem[5,24] = 0
 accuracy_3_40: - 50.9765625 frac[3,40] + mem[3,40] = 0
 accuracy_1_40: - 50.9765625 frac[1,40] + mem[1,40] = 0
 accuracy_6_23: - 50.9765625 frac[6,23] + mem[6,23] = 0
 accuracy_4_35: - 50.9765625 frac[4,35] + mem[4,35] = 0
 accuracy_8_38: - 50.9765625 frac[8,38] + mem[8,38] = 0
 accuracy_5_34: - 50.9765625 frac[5,34] + mem[5,34] = 0
 accuracy_9_39: - 50.9765625 frac[9,39] + mem[9,39] = 0
 accuracy_6_41: - 50.9765625 frac[6,41] + mem[6,41] = 0
 accuracy_10_36: - 50.9765625 frac[10,36] + mem[10,36] = 0
 accuracy_8_12: - 50.9765625 frac[8,12] + mem[8,12] = 0
 accuracy_7_44: - 50.9765625 frac[7,44] + mem[7,44] = 0
 accuracy_11_41: - 50.9765625 frac[11,41] + mem[11,41] = 0
 accuracy_9_9: - 50.9765625 frac[9,9] + mem[9,9] = 0
 accuracy_12_50: - 50.9765625 frac[12,50] + mem[12,50] = 0
 accuracy_10_14: - 50.9765625 frac[10,14] + mem[10,14] = 0
 accuracy_8_18: - 50.9765625 frac[8,18] + mem[8,18] = 0
 accuracy_13_51: - 50.9765625 frac[13,51] + mem[13,51] = 0
 accuracy_11_15: - 50.9765625 frac[11,15] + mem[11,15] = 0
 accuracy_9_19: - 50.9765625 frac[9,19] + mem[9,19] = 0
 accuracy_14_8: - 50.9765625 frac[14,8] + mem[14,8] = 0
 accuracy_12_8: - 50.9765625 frac[12,8] + mem[12,8] = 0
 accuracy_13_13: - 50.9765625 frac[13,13] + mem[13,13] = 0
 accuracy_2_18: - 50.9765625 frac[2,18] + mem[2,18] = 0
 accuracy_0_14: - 50.9765625 frac[0,14] + mem[0,14] = 0
 accuracy_14_50: - 50.9765625 frac[14,50] + mem[14,50] = 0
 accuracy_3_11: - 50.9765625 frac[3,11] + mem[3,11] = 0
 accuracy_1_15: - 50.9765625 frac[1,15] + mem[1,15] = 0
 accuracy_4_12: - 50.9765625 frac[4,12] + mem[4,12] = 0
 accuracy_2_12: - 50.9765625 frac[2,12] + mem[2,12] = 0
 accuracy_5_1: - 50.9765625 frac[5,1] + mem[5,1] = 0
 accuracy_3_17: - 50.9765625 frac[3,17] + mem[3,17] = 0
 accuracy_1_33: - 50.9765625 frac[1,33] + mem[1,33] = 0
 accuracy_6_14: - 50.9765625 frac[6,14] + mem[6,14] = 0
 accuracy_4_42: - 50.9765625 frac[4,42] + mem[4,42] = 0
 accuracy_8_45: - 50.9765625 frac[8,45] + mem[8,45] = 0
 accuracy_7_15: - 50.9765625 frac[7,15] + mem[7,15] = 0
 accuracy_3_55: - 50.9765625 frac[3,55] + mem[3,55] = 0
 accuracy_9_40: - 50.9765625 frac[9,40] + mem[9,40] = 0
 accuracy_6_32: - 50.9765625 frac[6,32] + mem[6,32] = 0
 accuracy_10_47: - 50.9765625 frac[10,47] + mem[10,47] = 0
 accuracy_8_51: - 50.9765625 frac[8,51] + mem[8,51] = 0
 accuracy_7_37: - 50.9765625 frac[7,37] + mem[7,37] = 0
 accuracy_5_53: - 50.9765625 frac[5,53] + mem[5,53] = 0
 accuracy_11_46: - 50.9765625 frac[11,46] + mem[11,46] = 0
 accuracy_9_50: - 50.9765625 frac[9,50] + mem[9,50] = 0
 accuracy_6_58: - 50.9765625 frac[6,58] + mem[6,58] = 0
 accuracy_12_41: - 50.9765625 frac[12,41] + mem[12,41] = 0
 accuracy_10_9: - 50.9765625 frac[10,9] + mem[10,9] = 0
 accuracy_8_25: - 50.9765625 frac[8,25] + mem[8,25] = 0
 accuracy_13_44: - 50.9765625 frac[13,44] + mem[13,44] = 0
 accuracy_11_4: - 50.9765625 frac[11,4] + mem[11,4] = 0
 accuracy_9_20: - 50.9765625 frac[9,20] + mem[9,20] = 0
 accuracy_14_19: - 50.9765625 frac[14,19] + mem[14,19] = 0
 accuracy_12_7: - 50.9765625 frac[12,7] + mem[12,7] = 0
 accuracy_10_19: - 50.9765625 frac[10,19] + mem[10,19] = 0
 accuracy_13_6: - 50.9765625 frac[13,6] + mem[13,6] = 0
 accuracy_11_26: - 50.9765625 frac[11,26] + mem[11,26] = 0
 accuracy_0_5: - 50.9765625 frac[0,5] + mem[0,5] = 0
 accuracy_14_53: - 50.9765625 frac[14,53] + mem[14,53] = 0
 accuracy_12_29: - 50.9765625 frac[12,29] + mem[12,29] = 0
 accuracy_1_0: - 50.9765625 frac[1,0] + mem[1,0] = 0
 accuracy_4_11: - 50.9765625 frac[4,11] + mem[4,11] = 0
 accuracy_2_7: - 50.9765625 frac[2,7] + mem[2,7] = 0
 accuracy_0_59: - 50.9765625 frac[0,59] + mem[0,59] = 0
 accuracy_5_10: - 50.9765625 frac[5,10] + mem[5,10] = 0
 accuracy_3_22: - 50.9765625 frac[3,22] + mem[3,22] = 0
 accuracy_1_58: - 50.9765625 frac[1,58] + mem[1,58] = 0
 accuracy_6_1: - 50.9765625 frac[6,1] + mem[6,1] = 0
 accuracy_2_33: - 50.9765625 frac[2,33] + mem[2,33] = 0
 accuracy_7_4: - 50.9765625 frac[7,4] + mem[7,4] = 0
 accuracy_5_20: - 50.9765625 frac[5,20] + mem[5,20] = 0
 accuracy_3_60: - 50.9765625 frac[3,60] + mem[3,60] = 0
 accuracy_6_27: - 50.9765625 frac[6,27] + mem[6,27] = 0
 accuracy_4_63: - 50.9765625 frac[4,63] + mem[4,63] = 0
 accuracy_8_58: - 50.9765625 frac[8,58] + mem[8,58] = 0
 accuracy_11_39: - 50.9765625 frac[11,39] + mem[11,39] = 0
 accuracy_9_59: - 50.9765625 frac[9,59] + mem[9,59] = 0
 accuracy_12_32: - 50.9765625 frac[12,32] + mem[12,32] = 0
 accuracy_10_48: - 50.9765625 frac[10,48] + mem[10,48] = 0
 accuracy_8_0: - 50.9765625 frac[8,0] + mem[8,0] = 0
 accuracy_7_56: - 50.9765625 frac[7,56] + mem[7,56] = 0
 accuracy_13_37: - 50.9765625 frac[13,37] + mem[13,37] = 0
 accuracy_11_61: - 50.9765625 frac[11,61] + mem[11,61] = 0
 accuracy_9_29: - 50.9765625 frac[9,29] + mem[9,29] = 0
 accuracy_14_26: - 50.9765625 frac[14,26] + mem[14,26] = 0
 accuracy_12_62: - 50.9765625 frac[12,62] + mem[12,62] = 0
 accuracy_10_26: - 50.9765625 frac[10,26] + mem[10,26] = 0
 accuracy_13_63: - 50.9765625 frac[13,63] + mem[13,63] = 0
 accuracy_11_19: - 50.9765625 frac[11,19] + mem[11,19] = 0
 accuracy_0_28: - 50.9765625 frac[0,28] + mem[0,28] = 0
 accuracy_14_60: - 50.9765625 frac[14,60] + mem[14,60] = 0
 accuracy_12_20: - 50.9765625 frac[12,20] + mem[12,20] = 0
 accuracy_1_25: - 50.9765625 frac[1,25] + mem[1,25] = 0
 accuracy_2_30: - 50.9765625 frac[2,30] + mem[2,30] = 0
 accuracy_14_38: - 50.9765625 frac[14,38] + mem[14,38] = 0
 accuracy_1_51: - 50.9765625 frac[1,51] + mem[1,51] = 0
 accuracy_4_24: - 50.9765625 frac[4,24] + mem[4,24] = 0
 accuracy_2_56: - 50.9765625 frac[2,56] + mem[2,56] = 0
 accuracy_0_40: - 50.9765625 frac[0,40] + mem[0,40] = 0
 accuracy_3_37: - 50.9765625 frac[3,37] + mem[3,37] = 0
 accuracy_6_18: - 50.9765625 frac[6,18] + mem[6,18] = 0
 accuracy_4_38: - 50.9765625 frac[4,38] + mem[4,38] = 0
 accuracy_8_33: - 50.9765625 frac[8,33] + mem[8,33] = 0
 accuracy_7_27: - 50.9765625 frac[7,27] + mem[7,27] = 0
 accuracy_5_39: - 50.9765625 frac[5,39] + mem[5,39] = 0
 accuracy_9_60: - 50.9765625 frac[9,60] + mem[9,60] = 0
 accuracy_6_52: - 50.9765625 frac[6,52] + mem[6,52] = 0
 accuracy_10_59: - 50.9765625 frac[10,59] + mem[10,59] = 0
 accuracy_8_7: - 50.9765625 frac[8,7] + mem[8,7] = 0
 accuracy_11_50: - 50.9765625 frac[11,50] + mem[11,50] = 0
 accuracy_9_6: - 50.9765625 frac[9,6] + mem[9,6] = 0
 accuracy_14_29: - 50.9765625 frac[14,29] + mem[14,29] = 0
 accuracy_12_53: - 50.9765625 frac[12,53] + mem[12,53] = 0
 accuracy_10_5: - 50.9765625 frac[10,5] + mem[10,5] = 0
 accuracy_13_56: - 50.9765625 frac[13,56] + mem[13,56] = 0
 accuracy_11_8: - 50.9765625 frac[11,8] + mem[11,8] = 0
 accuracy_14_7: - 50.9765625 frac[14,7] + mem[14,7] = 0
 accuracy_12_19: - 50.9765625 frac[12,19] + mem[12,19] = 0
 accuracy_1_18: - 50.9765625 frac[1,18] + mem[1,18] = 0
 accuracy_13_18: - 50.9765625 frac[13,18] + mem[13,18] = 0
 accuracy_2_25: - 50.9765625 frac[2,25] + mem[2,25] = 0
 accuracy_14_41: - 50.9765625 frac[14,41] + mem[14,41] = 0
 accuracy_3_4: - 50.9765625 frac[3,4] + mem[3,4] = 0
 accuracy_4_7: - 50.9765625 frac[4,7] + mem[4,7] = 0
 accuracy_2_51: - 50.9765625 frac[2,51] + mem[2,51] = 0
 accuracy_0_47: - 50.9765625 frac[0,47] + mem[0,47] = 0
 accuracy_5_6: - 50.9765625 frac[5,6] + mem[5,6] = 0
 accuracy_1_46: - 50.9765625 frac[1,46] + mem[1,46] = 0
 accuracy_8_40: - 50.9765625 frac[8,40] + mem[8,40] = 0
 accuracy_7_16: - 50.9765625 frac[7,16] + mem[7,16] = 0
 accuracy_5_32: - 50.9765625 frac[5,32] + mem[5,32] = 0
 accuracy_3_48: - 50.9765625 frac[3,48] + mem[3,48] = 0
 accuracy_9_37: - 50.9765625 frac[9,37] + mem[9,37] = 0
 accuracy_6_47: - 50.9765625 frac[6,47] + mem[6,47] = 0
 accuracy_10_34: - 50.9765625 frac[10,34] + mem[10,34] = 0
 accuracy_8_14: - 50.9765625 frac[8,14] + mem[8,14] = 0
 accuracy_11_43: - 50.9765625 frac[11,43] + mem[11,43] = 0
 accuracy_9_15: - 50.9765625 frac[9,15] + mem[9,15] = 0
 accuracy_12_44: - 50.9765625 frac[12,44] + mem[12,44] = 0
 accuracy_10_12: - 50.9765625 frac[10,12] + mem[10,12] = 0
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
 accuracy_2_10: - 50.9765625 frac[2,10] + mem[2,10] = 0
 accuracy_0_38: - 50.9765625 frac[0,38] + mem[0,38] = 0
 accuracy_5_15: - 50.9765625 frac[5,15] + mem[5,15] = 0
 accuracy_3_19: - 50.9765625 frac[3,19] + mem[3,19] = 0
 accuracy_1_39: - 50.9765625 frac[1,39] + mem[1,39] = 0
 accuracy_6_12: - 50.9765625 frac[6,12] + mem[6,12] = 0
 accuracy_4_20: - 50.9765625 frac[4,20] + mem[4,20] = 0
 accuracy_2_36: - 50.9765625 frac[2,36] + mem[2,36] = 0
 accuracy_8_47: - 50.9765625 frac[8,47] + mem[8,47] = 0
 accuracy_7_9: - 50.9765625 frac[7,9] + mem[7,9] = 0
 accuracy_5_41: - 50.9765625 frac[5,41] + mem[5,41] = 0
 accuracy_9_46: - 50.9765625 frac[9,46] + mem[9,46] = 0
 accuracy_6_38: - 50.9765625 frac[6,38] + mem[6,38] = 0
 accuracy_10_45: - 50.9765625 frac[10,45] + mem[10,45] = 0
 accuracy_8_53: - 50.9765625 frac[8,53] + mem[8,53] = 0
 accuracy_7_39: - 50.9765625 frac[7,39] + mem[7,39] = 0
 accuracy_11_32: - 50.9765625 frac[11,32] + mem[11,32] = 0
 accuracy_9_48: - 50.9765625 frac[9,48] + mem[9,48] = 0
 accuracy_6_56: - 50.9765625 frac[6,56] + mem[6,56] = 0
 accuracy_12_43: - 50.9765625 frac[12,43] + mem[12,43] = 0
 accuracy_10_55: - 50.9765625 frac[10,55] + mem[10,55] = 0
 accuracy_8_27: - 50.9765625 frac[8,27] + mem[8,27] = 0
 accuracy_7_61: - 50.9765625 frac[7,61] + mem[7,61] = 0
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
 accuracy_13_30: - 50.9765625 frac[13,30] + mem[13,30] = 0
 accuracy_2_5: - 50.9765625 frac[2,5] + mem[2,5] = 0
 accuracy_3_24: - 50.9765625 frac[3,24] + mem[3,24] = 0
 accuracy_4_19: - 50.9765625 frac[4,19] + mem[4,19] = 0
 accuracy_2_63: - 50.9765625 frac[2,63] + mem[2,63] = 0
 accuracy_7_6: - 50.9765625 frac[7,6] + mem[7,6] = 0
 accuracy_5_18: - 50.9765625 frac[5,18] + mem[5,18] = 0
 accuracy_3_62: - 50.9765625 frac[3,62] + mem[3,62] = 0
 accuracy_4_57: - 50.9765625 frac[4,57] + mem[4,57] = 0
 accuracy_8_60: - 50.9765625 frac[8,60] + mem[8,60] = 0
 accuracy_7_28: - 50.9765625 frac[7,28] + mem[7,28] = 0
 accuracy_5_60: - 50.9765625 frac[5,60] + mem[5,60] = 0
 accuracy_6_51: - 50.9765625 frac[6,51] + mem[6,51] = 0
 accuracy_12_34: - 50.9765625 frac[12,34] + mem[12,34] = 0
 accuracy_10_62: - 50.9765625 frac[10,62] + mem[10,62] = 0
 accuracy_8_2: - 50.9765625 frac[8,2] + mem[8,2] = 0
 accuracy_7_58: - 50.9765625 frac[7,58] + mem[7,58] = 0
 accuracy_13_35: - 50.9765625 frac[13,35] + mem[13,35] = 0
 accuracy_11_63: - 50.9765625 frac[11,63] + mem[11,63] = 0
 accuracy_9_3: - 50.9765625 frac[9,3] + mem[9,3] = 0
 accuracy_14_24: - 50.9765625 frac[14,24] + mem[14,24] = 0
 accuracy_12_56: - 50.9765625 frac[12,56] + mem[12,56] = 0
 accuracy_10_24: - 50.9765625 frac[10,24] + mem[10,24] = 0
 accuracy_13_61: - 50.9765625 frac[13,61] + mem[13,61] = 0
 accuracy_0_30: - 50.9765625 frac[0,30] + mem[0,30] = 0
 accuracy_14_2: - 50.9765625 frac[14,2] + mem[14,2] = 0
 accuracy_12_22: - 50.9765625 frac[12,22] + mem[12,22] = 0
 accuracy_13_23: - 50.9765625 frac[13,23] + mem[13,23] = 0
 accuracy_2_28: - 50.9765625 frac[2,28] + mem[2,28] = 0
 accuracy_14_36: - 50.9765625 frac[14,36] + mem[14,36] = 0
 accuracy_3_1: - 50.9765625 frac[3,1] + mem[3,1] = 0
 accuracy_1_49: - 50.9765625 frac[1,49] + mem[1,49] = 0
 accuracy_4_26: - 50.9765625 frac[4,26] + mem[4,26] = 0
 accuracy_0_42: - 50.9765625 frac[0,42] + mem[0,42] = 0
 accuracy_5_27: - 50.9765625 frac[5,27] + mem[5,27] = 0
 accuracy_3_39: - 50.9765625 frac[3,39] + mem[3,39] = 0
 accuracy_6_16: - 50.9765625 frac[6,16] + mem[6,16] = 0
 accuracy_4_32: - 50.9765625 frac[4,32] + mem[4,32] = 0
 accuracy_8_35: - 50.9765625 frac[8,35] + mem[8,35] = 0
 accuracy_5_37: - 50.9765625 frac[5,37] + mem[5,37] = 0
 accuracy_9_34: - 50.9765625 frac[9,34] + mem[9,34] = 0
 accuracy_6_42: - 50.9765625 frac[6,42] + mem[6,42] = 0
 accuracy_10_57: - 50.9765625 frac[10,57] + mem[10,57] = 0
 accuracy_8_9: - 50.9765625 frac[8,9] + mem[8,9] = 0
 accuracy_11_52: - 50.9765625 frac[11,52] + mem[11,52] = 0
 accuracy_9_4: - 50.9765625 frac[9,4] + mem[9,4] = 0
 accuracy_12_55: - 50.9765625 frac[12,55] + mem[12,55] = 0
 accuracy_10_3: - 50.9765625 frac[10,3] + mem[10,3] = 0
 accuracy_13_54: - 50.9765625 frac[13,54] + mem[13,54] = 0
 accuracy_11_10: - 50.9765625 frac[11,10] + mem[11,10] = 0
 accuracy_0_21: - 50.9765625 frac[0,21] + mem[0,21] = 0
 accuracy_14_5: - 50.9765625 frac[14,5] + mem[14,5] = 0
 accuracy_12_13: - 50.9765625 frac[12,13] + mem[12,13] = 0
 accuracy_1_16: - 50.9765625 frac[1,16] + mem[1,16] = 0
 accuracy_13_16: - 50.9765625 frac[13,16] + mem[13,16] = 0
 accuracy_2_23: - 50.9765625 frac[2,23] + mem[2,23] = 0
 accuracy_14_47: - 50.9765625 frac[14,47] + mem[14,47] = 0
 accuracy_1_10: - 50.9765625 frac[1,10] + mem[1,10] = 0
 accuracy_4_1: - 50.9765625 frac[4,1] + mem[4,1] = 0
 accuracy_2_49: - 50.9765625 frac[2,49] + mem[2,49] = 0
 accuracy_5_4: - 50.9765625 frac[5,4] + mem[5,4] = 0
 accuracy_3_44: - 50.9765625 frac[3,44] + mem[3,44] = 0
 accuracy_1_44: - 50.9765625 frac[1,44] + mem[1,44] = 0
 accuracy_6_11: - 50.9765625 frac[6,11] + mem[6,11] = 0
 accuracy_4_47: - 50.9765625 frac[4,47] + mem[4,47] = 0
 accuracy_2_43: - 50.9765625 frac[2,43] + mem[2,43] = 0
 accuracy_8_42: - 50.9765625 frac[8,42] + mem[8,42] = 0
 accuracy_7_18: - 50.9765625 frac[7,18] + mem[7,18] = 0
 accuracy_5_46: - 50.9765625 frac[5,46] + mem[5,46] = 0
 accuracy_3_50: - 50.9765625 frac[3,50] + mem[3,50] = 0
 accuracy_9_43: - 50.9765625 frac[9,43] + mem[9,43] = 0
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
 accuracy_10_20: - 50.9765625 frac[10,20] + mem[10,20] = 0
 accuracy_13_9: - 50.9765625 frac[13,9] + mem[13,9] = 0
 accuracy_0_2: - 50.9765625 frac[0,2] + mem[0,2] = 0
 accuracy_14_54: - 50.9765625 frac[14,54] + mem[14,54] = 0
 accuracy_3_15: - 50.9765625 frac[3,15] + mem[3,15] = 0
 accuracy_1_3: - 50.9765625 frac[1,3] + mem[1,3] = 0
 accuracy_2_8: - 50.9765625 frac[2,8] + mem[2,8] = 0
 accuracy_0_56: - 50.9765625 frac[0,56] + mem[0,56] = 0
 accuracy_5_13: - 50.9765625 frac[5,13] + mem[5,13] = 0
 accuracy_3_21: - 50.9765625 frac[3,21] + mem[3,21] = 0
 accuracy_1_37: - 50.9765625 frac[1,37] + mem[1,37] = 0
 accuracy_6_2: - 50.9765625 frac[6,2] + mem[6,2] = 0
 accuracy_4_22: - 50.9765625 frac[4,22] + mem[4,22] = 0
 accuracy_2_34: - 50.9765625 frac[2,34] + mem[2,34] = 0
 accuracy_7_11: - 50.9765625 frac[7,11] + mem[7,11] = 0
 accuracy_5_23: - 50.9765625 frac[5,23] + mem[5,23] = 0
 accuracy_3_59: - 50.9765625 frac[3,59] + mem[3,59] = 0
 accuracy_9_44: - 50.9765625 frac[9,44] + mem[9,44] = 0
 accuracy_6_36: - 50.9765625 frac[6,36] + mem[6,36] = 0
 accuracy_10_43: - 50.9765625 frac[10,43] + mem[10,43] = 0
 accuracy_8_55: - 50.9765625 frac[8,55] + mem[8,55] = 0
 accuracy_7_33: - 50.9765625 frac[7,33] + mem[7,33] = 0
 accuracy_11_34: - 50.9765625 frac[11,34] + mem[11,34] = 0
 accuracy_9_54: - 50.9765625 frac[9,54] + mem[9,54] = 0
 accuracy_12_37: - 50.9765625 frac[12,37] + mem[12,37] = 0
 accuracy_10_53: - 50.9765625 frac[10,53] + mem[10,53] = 0
 accuracy_8_29: - 50.9765625 frac[8,29] + mem[8,29] = 0
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
 accuracy_1_62: - 50.9765625 frac[1,62] + mem[1,62] = 0
 accuracy_6_5: - 50.9765625 frac[6,5] + mem[6,5] = 0
 accuracy_4_29: - 50.9765625 frac[4,29] + mem[4,29] = 0
 accuracy_2_61: - 50.9765625 frac[2,61] + mem[2,61] = 0
 accuracy_5_16: - 50.9765625 frac[5,16] + mem[5,16] = 0
 accuracy_3_32: - 50.9765625 frac[3,32] + mem[3,32] = 0
 accuracy_4_59: - 50.9765625 frac[4,59] + mem[4,59] = 0
 accuracy_8_62: - 50.9765625 frac[8,62] + mem[8,62] = 0
 accuracy_7_30: - 50.9765625 frac[7,30] + mem[7,30] = 0
 accuracy_5_58: - 50.9765625 frac[5,58] + mem[5,58] = 0
 accuracy_9_63: - 50.9765625 frac[9,63] + mem[9,63] = 0
 accuracy_6_49: - 50.9765625 frac[6,49] + mem[6,49] = 0
 accuracy_10_60: - 50.9765625 frac[10,60] + mem[10,60] = 0
 accuracy_8_4: - 50.9765625 frac[8,4] + mem[8,4] = 0
 accuracy_7_52: - 50.9765625 frac[7,52] + mem[7,52] = 0
 accuracy_13_33: - 50.9765625 frac[13,33] + mem[13,33] = 0
 accuracy_11_49: - 50.9765625 frac[11,49] + mem[11,49] = 0
 accuracy_9_1: - 50.9765625 frac[9,1] + mem[9,1] = 0
 accuracy_14_30: - 50.9765625 frac[14,30] + mem[14,30] = 0
 accuracy_12_58: - 50.9765625 frac[12,58] + mem[12,58] = 0
 accuracy_10_6: - 50.9765625 frac[10,6] + mem[10,6] = 0
 accuracy_13_59: - 50.9765625 frac[13,59] + mem[13,59] = 0
 accuracy_11_23: - 50.9765625 frac[11,23] + mem[11,23] = 0
 accuracy_0_16: - 50.9765625 frac[0,16] + mem[0,16] = 0
 accuracy_14_0: - 50.9765625 frac[14,0] + mem[14,0] = 0
 accuracy_12_16: - 50.9765625 frac[12,16] + mem[12,16] = 0
 accuracy_1_29: - 50.9765625 frac[1,29] + mem[1,29] = 0
 accuracy_13_21: - 50.9765625 frac[13,21] + mem[13,21] = 0
 accuracy_14_42: - 50.9765625 frac[14,42] + mem[14,42] = 0
 accuracy_3_3: - 50.9765625 frac[3,3] + mem[3,3] = 0
 accuracy_1_55: - 50.9765625 frac[1,55] + mem[1,55] = 0
 accuracy_4_4: - 50.9765625 frac[4,4] + mem[4,4] = 0
 accuracy_2_52: - 50.9765625 frac[2,52] + mem[2,52] = 0
 accuracy_0_44: - 50.9765625 frac[0,44] + mem[0,44] = 0
 accuracy_3_41: - 50.9765625 frac[3,41] + mem[3,41] = 0
 accuracy_2_46: - 50.9765625 frac[2,46] + mem[2,46] = 0
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
 accuracy_0_23: - 50.9765625 frac[0,23] + mem[0,23] = 0
 accuracy_14_11: - 50.9765625 frac[14,11] + mem[14,11] = 0
 accuracy_12_15: - 50.9765625 frac[12,15] + mem[12,15] = 0
 accuracy_1_22: - 50.9765625 frac[1,22] + mem[1,22] = 0
 accuracy_13_14: - 50.9765625 frac[13,14] + mem[13,14] = 0
 accuracy_2_21: - 50.9765625 frac[2,21] + mem[2,21] = 0
 accuracy_14_45: - 50.9765625 frac[14,45] + mem[14,45] = 0
 accuracy_3_8: - 50.9765625 frac[3,8] + mem[3,8] = 0
 accuracy_1_8: - 50.9765625 frac[1,8] + mem[1,8] = 0
 accuracy_4_3: - 50.9765625 frac[4,3] + mem[4,3] = 0
 accuracy_2_15: - 50.9765625 frac[2,15] + mem[2,15] = 0
 accuracy_0_35: - 50.9765625 frac[0,35] + mem[0,35] = 0
 accuracy_5_2: - 50.9765625 frac[5,2] + mem[5,2] = 0
 accuracy_3_46: - 50.9765625 frac[3,46] + mem[3,46] = 0
 accuracy_1_34: - 50.9765625 frac[1,34] + mem[1,34] = 0
 accuracy_4_41: - 50.9765625 frac[4,41] + mem[4,41] = 0
 accuracy_2_41: - 50.9765625 frac[2,41] + mem[2,41] = 0
 accuracy_8_44: - 50.9765625 frac[8,44] + mem[8,44] = 0
 accuracy_5_44: - 50.9765625 frac[5,44] + mem[5,44] = 0
 accuracy_3_52: - 50.9765625 frac[3,52] + mem[3,52] = 0
 accuracy_9_41: - 50.9765625 frac[9,41] + mem[9,41] = 0
 accuracy_6_35: - 50.9765625 frac[6,35] + mem[6,35] = 0
 accuracy_4_55: - 50.9765625 frac[4,55] + mem[4,55] = 0
 accuracy_10_46: - 50.9765625 frac[10,46] + mem[10,46] = 0
 accuracy_8_50: - 50.9765625 frac[8,50] + mem[8,50] = 0
 accuracy_7_42: - 50.9765625 frac[7,42] + mem[7,42] = 0
 accuracy_5_54: - 50.9765625 frac[5,54] + mem[5,54] = 0
 accuracy_11_47: - 50.9765625 frac[11,47] + mem[11,47] = 0
 accuracy_9_51: - 50.9765625 frac[9,51] + mem[9,51] = 0
 accuracy_12_40: - 50.9765625 frac[12,40] + mem[12,40] = 0
 accuracy_8_24: - 50.9765625 frac[8,24] + mem[8,24] = 0
 accuracy_13_45: - 50.9765625 frac[13,45] + mem[13,45] = 0
 accuracy_11_5: - 50.9765625 frac[11,5] + mem[11,5] = 0
 accuracy_9_21: - 50.9765625 frac[9,21] + mem[9,21] = 0
 accuracy_14_18: - 50.9765625 frac[14,18] + mem[14,18] = 0
 accuracy_12_6: - 50.9765625 frac[12,6] + mem[12,6] = 0
 accuracy_10_18: - 50.9765625 frac[10,18] + mem[10,18] = 0
 accuracy_13_7: - 50.9765625 frac[13,7] + mem[13,7] = 0
 accuracy_11_27: - 50.9765625 frac[11,27] + mem[11,27] = 0
 accuracy_0_4: - 50.9765625 frac[0,4] + mem[0,4] = 0
 accuracy_14_52: - 50.9765625 frac[14,52] + mem[14,52] = 0
 accuracy_12_28: - 50.9765625 frac[12,28] + mem[12,28] = 0
 accuracy_1_1: - 50.9765625 frac[1,1] + mem[1,1] = 0
 accuracy_4_10: - 50.9765625 frac[4,10] + mem[4,10] = 0
 accuracy_2_6: - 50.9765625 frac[2,6] + mem[2,6] = 0
 accuracy_5_11: - 50.9765625 frac[5,11] + mem[5,11] = 0
 accuracy_3_23: - 50.9765625 frac[3,23] + mem[3,23] = 0
 accuracy_1_59: - 50.9765625 frac[1,59] + mem[1,59] = 0
 accuracy_6_0: - 50.9765625 frac[6,0] + mem[6,0] = 0
 accuracy_4_16: - 50.9765625 frac[4,16] + mem[4,16] = 0
 accuracy_2_32: - 50.9765625 frac[2,32] + mem[2,32] = 0
 accuracy_7_5: - 50.9765625 frac[7,5] + mem[7,5] = 0
 accuracy_5_21: - 50.9765625 frac[5,21] + mem[5,21] = 0
 accuracy_6_26: - 50.9765625 frac[6,26] + mem[6,26] = 0
 accuracy_4_62: - 50.9765625 frac[4,62] + mem[4,62] = 0
 accuracy_10_41: - 50.9765625 frac[10,41] + mem[10,41] = 0
 accuracy_8_57: - 50.9765625 frac[8,57] + mem[8,57] = 0
 accuracy_7_35: - 50.9765625 frac[7,35] + mem[7,35] = 0
 accuracy_11_36: - 50.9765625 frac[11,36] + mem[11,36] = 0
 accuracy_9_52: - 50.9765625 frac[9,52] + mem[9,52] = 0
 accuracy_12_39: - 50.9765625 frac[12,39] + mem[12,39] = 0
 accuracy_7_57: - 50.9765625 frac[7,57] + mem[7,57] = 0
 accuracy_13_38: - 50.9765625 frac[13,38] + mem[13,38] = 0
 accuracy_11_58: - 50.9765625 frac[11,58] + mem[11,58] = 0
 accuracy_9_30: - 50.9765625 frac[9,30] + mem[9,30] = 0
 accuracy_14_21: - 50.9765625 frac[14,21] + mem[14,21] = 0
 accuracy_12_61: - 50.9765625 frac[12,61] + mem[12,61] = 0
 accuracy_10_29: - 50.9765625 frac[10,29] + mem[10,29] = 0
 accuracy_13_0: - 50.9765625 frac[13,0] + mem[13,0] = 0
 accuracy_11_16: - 50.9765625 frac[11,16] + mem[11,16] = 0
 accuracy_0_27: - 50.9765625 frac[0,27] + mem[0,27] = 0
 accuracy_14_63: - 50.9765625 frac[14,63] + mem[14,63] = 0
 accuracy_12_27: - 50.9765625 frac[12,27] + mem[12,27] = 0
 accuracy_13_26: - 50.9765625 frac[13,26] + mem[13,26] = 0
 accuracy_2_1: - 50.9765625 frac[2,1] + mem[2,1] = 0
 accuracy_0_49: - 50.9765625 frac[0,49] + mem[0,49] = 0
 accuracy_14_33: - 50.9765625 frac[14,33] + mem[14,33] = 0
 accuracy_3_28: - 50.9765625 frac[3,28] + mem[3,28] = 0
 accuracy_1_60: - 50.9765625 frac[1,60] + mem[1,60] = 0
 accuracy_4_31: - 50.9765625 frac[4,31] + mem[4,31] = 0
 accuracy_7_2: - 50.9765625 frac[7,2] + mem[7,2] = 0
 accuracy_5_30: - 50.9765625 frac[5,30] + mem[5,30] = 0
 accuracy_3_34: - 50.9765625 frac[3,34] + mem[3,34] = 0
 accuracy_6_29: - 50.9765625 frac[6,29] + mem[6,29] = 0
 accuracy_4_37: - 50.9765625 frac[4,37] + mem[4,37] = 0
 accuracy_8_32: - 50.9765625 frac[8,32] + mem[8,32] = 0
 accuracy_7_24: - 50.9765625 frac[7,24] + mem[7,24] = 0
 accuracy_9_61: - 50.9765625 frac[9,61] + mem[9,61] = 0
 accuracy_6_55: - 50.9765625 frac[6,55] + mem[6,55] = 0
 accuracy_10_58: - 50.9765625 frac[10,58] + mem[10,58] = 0
 accuracy_8_6: - 50.9765625 frac[8,6] + mem[8,6] = 0
 accuracy_7_54: - 50.9765625 frac[7,54] + mem[7,54] = 0
 accuracy_11_51: - 50.9765625 frac[11,51] + mem[11,51] = 0
 accuracy_9_7: - 50.9765625 frac[9,7] + mem[9,7] = 0
 accuracy_14_28: - 50.9765625 frac[14,28] + mem[14,28] = 0
 accuracy_12_52: - 50.9765625 frac[12,52] + mem[12,52] = 0
 accuracy_10_4: - 50.9765625 frac[10,4] + mem[10,4] = 0
 accuracy_13_57: - 50.9765625 frac[13,57] + mem[13,57] = 0
 accuracy_11_9: - 50.9765625 frac[11,9] + mem[11,9] = 0
 accuracy_0_18: - 50.9765625 frac[0,18] + mem[0,18] = 0
 accuracy_14_6: - 50.9765625 frac[14,6] + mem[14,6] = 0
 accuracy_12_18: - 50.9765625 frac[12,18] + mem[12,18] = 0
 accuracy_1_19: - 50.9765625 frac[1,19] + mem[1,19] = 0
 accuracy_13_19: - 50.9765625 frac[13,19] + mem[13,19] = 0
 accuracy_2_24: - 50.9765625 frac[2,24] + mem[2,24] = 0
 accuracy_14_40: - 50.9765625 frac[14,40] + mem[14,40] = 0
 accuracy_3_5: - 50.9765625 frac[3,5] + mem[3,5] = 0
 accuracy_1_53: - 50.9765625 frac[1,53] + mem[1,53] = 0
 accuracy_4_6: - 50.9765625 frac[4,6] + mem[4,6] = 0
 accuracy_0_46: - 50.9765625 frac[0,46] + mem[0,46] = 0
 accuracy_5_7: - 50.9765625 frac[5,7] + mem[5,7] = 0
 accuracy_3_43: - 50.9765625 frac[3,43] + mem[3,43] = 0
 accuracy_1_47: - 50.9765625 frac[1,47] + mem[1,47] = 0
 accuracy_6_20: - 50.9765625 frac[6,20] + mem[6,20] = 0
 accuracy_2_44: - 50.9765625 frac[2,44] + mem[2,44] = 0
 accuracy_8_39: - 50.9765625 frac[8,39] + mem[8,39] = 0
 accuracy_7_17: - 50.9765625 frac[7,17] + mem[7,17] = 0
 accuracy_3_49: - 50.9765625 frac[3,49] + mem[3,49] = 0
 accuracy_9_38: - 50.9765625 frac[9,38] + mem[9,38] = 0
 accuracy_6_46: - 50.9765625 frac[6,46] + mem[6,46] = 0
 accuracy_10_37: - 50.9765625 frac[10,37] + mem[10,37] = 0
 accuracy_11_40: - 50.9765625 frac[11,40] + mem[11,40] = 0
 accuracy_9_8: - 50.9765625 frac[9,8] + mem[9,8] = 0
 accuracy_12_51: - 50.9765625 frac[12,51] + mem[12,51] = 0
 accuracy_10_15: - 50.9765625 frac[10,15] + mem[10,15] = 0
 accuracy_8_19: - 50.9765625 frac[8,19] + mem[8,19] = 0
 accuracy_13_50: - 50.9765625 frac[13,50] + mem[13,50] = 0
 accuracy_11_14: - 50.9765625 frac[11,14] + mem[11,14] = 0
 accuracy_9_18: - 50.9765625 frac[9,18] + mem[9,18] = 0
 accuracy_14_9: - 50.9765625 frac[14,9] + mem[14,9] = 0
 accuracy_12_9: - 50.9765625 frac[12,9] + mem[12,9] = 0
 accuracy_1_20: - 50.9765625 frac[1,20] + mem[1,20] = 0
 accuracy_13_12: - 50.9765625 frac[13,12] + mem[13,12] = 0
 accuracy_14_51: - 50.9765625 frac[14,51] + mem[14,51] = 0
 accuracy_3_10: - 50.9765625 frac[3,10] + mem[3,10] = 0
 accuracy_2_13: - 50.9765625 frac[2,13] + mem[2,13] = 0
 accuracy_0_37: - 50.9765625 frac[0,37] + mem[0,37] = 0
 accuracy_3_16: - 50.9765625 frac[3,16] + mem[3,16] = 0
 accuracy_1_32: - 50.9765625 frac[1,32] + mem[1,32] = 0
 accuracy_4_43: - 50.9765625 frac[4,43] + mem[4,43] = 0
 accuracy_2_39: - 50.9765625 frac[2,39] + mem[2,39] = 0
 accuracy_8_46: - 50.9765625 frac[8,46] + mem[8,46] = 0
 accuracy_5_42: - 50.9765625 frac[5,42] + mem[5,42] = 0
 accuracy_3_54: - 50.9765625 frac[3,54] + mem[3,54] = 0
 accuracy_9_47: - 50.9765625 frac[9,47] + mem[9,47] = 0
 accuracy_6_33: - 50.9765625 frac[6,33] + mem[6,33] = 0
 accuracy_10_44: - 50.9765625 frac[10,44] + mem[10,44] = 0
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
 accuracy_5_9: - 50.9765625 frac[5,9] + mem[5,9] = 0
 accuracy_3_25: - 50.9765625 frac[3,25] + mem[3,25] = 0
 accuracy_1_57: - 50.9765625 frac[1,57] + mem[1,57] = 0
 accuracy_6_6: - 50.9765625 frac[6,6] + mem[6,6] = 0
 accuracy_4_18: - 50.9765625 frac[4,18] + mem[4,18] = 0
 accuracy_2_62: - 50.9765625 frac[2,62] + mem[2,62] = 0
 accuracy_7_7: - 50.9765625 frac[7,7] + mem[7,7] = 0
 accuracy_5_19: - 50.9765625 frac[5,19] + mem[5,19] = 0
 accuracy_4_56: - 50.9765625 frac[4,56] + mem[4,56] = 0
 accuracy_8_59: - 50.9765625 frac[8,59] + mem[8,59] = 0
 accuracy_7_29: - 50.9765625 frac[7,29] + mem[7,29] = 0
 accuracy_11_38: - 50.9765625 frac[11,38] + mem[11,38] = 0
 accuracy_9_58: - 50.9765625 frac[9,58] + mem[9,58] = 0
 accuracy_12_33: - 50.9765625 frac[12,33] + mem[12,33] = 0
 accuracy_8_1: - 50.9765625 frac[8,1] + mem[8,1] = 0
 accuracy_7_59: - 50.9765625 frac[7,59] + mem[7,59] = 0
 accuracy_13_36: - 50.9765625 frac[13,36] + mem[13,36] = 0
 accuracy_11_60: - 50.9765625 frac[11,60] + mem[11,60] = 0
 accuracy_9_28: - 50.9765625 frac[9,28] + mem[9,28] = 0
 accuracy_14_27: - 50.9765625 frac[14,27] + mem[14,27] = 0
 accuracy_12_63: - 50.9765625 frac[12,63] + mem[12,63] = 0
 accuracy_10_27: - 50.9765625 frac[10,27] + mem[10,27] = 0
 accuracy_13_62: - 50.9765625 frac[13,62] + mem[13,62] = 0
 accuracy_11_18: - 50.9765625 frac[11,18] + mem[11,18] = 0
 accuracy_0_29: - 50.9765625 frac[0,29] + mem[0,29] = 0
 accuracy_14_61: - 50.9765625 frac[14,61] + mem[14,61] = 0
 accuracy_12_21: - 50.9765625 frac[12,21] + mem[12,21] = 0
 accuracy_1_24: - 50.9765625 frac[1,24] + mem[1,24] = 0
 accuracy_13_24: - 50.9765625 frac[13,24] + mem[13,24] = 0
 accuracy_2_31: - 50.9765625 frac[2,31] + mem[2,31] = 0
 accuracy_0_51: - 50.9765625 frac[0,51] + mem[0,51] = 0
 accuracy_14_39: - 50.9765625 frac[14,39] + mem[14,39] = 0
 accuracy_3_30: - 50.9765625 frac[3,30] + mem[3,30] = 0
 accuracy_1_50: - 50.9765625 frac[1,50] + mem[1,50] = 0
 accuracy_0_41: - 50.9765625 frac[0,41] + mem[0,41] = 0
 accuracy_5_28: - 50.9765625 frac[5,28] + mem[5,28] = 0
 accuracy_6_19: - 50.9765625 frac[6,19] + mem[6,19] = 0
 accuracy_4_39: - 50.9765625 frac[4,39] + mem[4,39] = 0
 accuracy_8_34: - 50.9765625 frac[8,34] + mem[8,34] = 0
 accuracy_7_26: - 50.9765625 frac[7,26] + mem[7,26] = 0
 accuracy_9_35: - 50.9765625 frac[9,35] + mem[9,35] = 0
 accuracy_6_53: - 50.9765625 frac[6,53] + mem[6,53] = 0
 accuracy_10_56: - 50.9765625 frac[10,56] + mem[10,56] = 0
 accuracy_8_8: - 50.9765625 frac[8,8] + mem[8,8] = 0
 accuracy_7_48: - 50.9765625 frac[7,48] + mem[7,48] = 0
 accuracy_11_53: - 50.9765625 frac[11,53] + mem[11,53] = 0
 accuracy_9_5: - 50.9765625 frac[9,5] + mem[9,5] = 0
 accuracy_12_54: - 50.9765625 frac[12,54] + mem[12,54] = 0
 accuracy_10_2: - 50.9765625 frac[10,2] + mem[10,2] = 0
 accuracy_13_55: - 50.9765625 frac[13,55] + mem[13,55] = 0
 accuracy_11_11: - 50.9765625 frac[11,11] + mem[11,11] = 0
 accuracy_14_4: - 50.9765625 frac[14,4] + mem[14,4] = 0
 accuracy_12_12: - 50.9765625 frac[12,12] + mem[12,12] = 0
 accuracy_1_17: - 50.9765625 frac[1,17] + mem[1,17] = 0
 accuracy_13_17: - 50.9765625 frac[13,17] + mem[13,17] = 0
 accuracy_2_22: - 50.9765625 frac[2,22] + mem[2,22] = 0
 accuracy_0_10: - 50.9765625 frac[0,10] + mem[0,10] = 0
 accuracy_14_46: - 50.9765625 frac[14,46] + mem[14,46] = 0
 accuracy_3_7: - 50.9765625 frac[3,7] + mem[3,7] = 0
 accuracy_1_11: - 50.9765625 frac[1,11] + mem[1,11] = 0
 accuracy_4_0: - 50.9765625 frac[4,0] + mem[4,0] = 0
 accuracy_2_48: - 50.9765625 frac[2,48] + mem[2,48] = 0
 accuracy_0_32: - 50.9765625 frac[0,32] + mem[0,32] = 0
 accuracy_5_5: - 50.9765625 frac[5,5] + mem[5,5] = 0
 accuracy_3_45: - 50.9765625 frac[3,45] + mem[3,45] = 0
 accuracy_6_10: - 50.9765625 frac[6,10] + mem[6,10] = 0
 accuracy_4_46: - 50.9765625 frac[4,46] + mem[4,46] = 0
 accuracy_2_42: - 50.9765625 frac[2,42] + mem[2,42] = 0
 accuracy_8_41: - 50.9765625 frac[8,41] + mem[8,41] = 0
 accuracy_7_19: - 50.9765625 frac[7,19] + mem[7,19] = 0
 accuracy_5_47: - 50.9765625 frac[5,47] + mem[5,47] = 0
 accuracy_3_51: - 50.9765625 frac[3,51] + mem[3,51] = 0
 accuracy_9_36: - 50.9765625 frac[9,36] + mem[9,36] = 0
 accuracy_10_35: - 50.9765625 frac[10,35] + mem[10,35] = 0
 accuracy_8_15: - 50.9765625 frac[8,15] + mem[8,15] = 0
 accuracy_7_41: - 50.9765625 frac[7,41] + mem[7,41] = 0
 accuracy_11_42: - 50.9765625 frac[11,42] + mem[11,42] = 0
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
 accuracy_14_49: - 50.9765625 frac[14,49] + mem[14,49] = 0
 accuracy_4_15: - 50.9765625 frac[4,15] + mem[4,15] = 0
 accuracy_2_11: - 50.9765625 frac[2,11] + mem[2,11] = 0
 accuracy_5_14: - 50.9765625 frac[5,14] + mem[5,14] = 0
 accuracy_1_38: - 50.9765625 frac[1,38] + mem[1,38] = 0
 accuracy_2_37: - 50.9765625 frac[2,37] + mem[2,37] = 0
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
 accuracy_12_24: - 50.9765625 frac[12,24] + mem[12,24] = 0
 accuracy_1_5: - 50.9765625 frac[1,5] + mem[1,5] = 0
 accuracy_13_29: - 50.9765625 frac[13,29] + mem[13,29] = 0
 accuracy_2_2: - 50.9765625 frac[2,2] + mem[2,2] = 0
 accuracy_0_62: - 50.9765625 frac[0,62] + mem[0,62] = 0
 accuracy_14_34: - 50.9765625 frac[14,34] + mem[14,34] = 0
 accuracy_1_63: - 50.9765625 frac[1,63] + mem[1,63] = 0
 accuracy_6_4: - 50.9765625 frac[6,4] + mem[6,4] = 0
 accuracy_4_28: - 50.9765625 frac[4,28] + mem[4,28] = 0
 accuracy_2_60: - 50.9765625 frac[2,60] + mem[2,60] = 0
 accuracy_7_1: - 50.9765625 frac[7,1] + mem[7,1] = 0
 accuracy_5_17: - 50.9765625 frac[5,17] + mem[5,17] = 0
 accuracy_6_30: - 50.9765625 frac[6,30] + mem[6,30] = 0
 accuracy_4_58: - 50.9765625 frac[4,58] + mem[4,58] = 0
 accuracy_5_59: - 50.9765625 frac[5,59] + mem[5,59] = 0
 accuracy_9_56: - 50.9765625 frac[9,56] + mem[9,56] = 0
 accuracy_6_48: - 50.9765625 frac[6,48] + mem[6,48] = 0
 accuracy_12_35: - 50.9765625 frac[12,35] + mem[12,35] = 0
 accuracy_10_63: - 50.9765625 frac[10,63] + mem[10,63] = 0
 accuracy_8_3: - 50.9765625 frac[8,3] + mem[8,3] = 0
 accuracy_7_53: - 50.9765625 frac[7,53] + mem[7,53] = 0
 accuracy_13_34: - 50.9765625 frac[13,34] + mem[13,34] = 0
 accuracy_9_2: - 50.9765625 frac[9,2] + mem[9,2] = 0
 accuracy_12_57: - 50.9765625 frac[12,57] + mem[12,57] = 0
 accuracy_13_60: - 50.9765625 frac[13,60] + mem[13,60] = 0
 accuracy_11_20: - 50.9765625 frac[11,20] + mem[11,20] = 0
 accuracy_14_3: - 50.9765625 frac[14,3] + mem[14,3] = 0
 accuracy_12_23: - 50.9765625 frac[12,23] + mem[12,23] = 0
 accuracy_1_30: - 50.9765625 frac[1,30] + mem[1,30] = 0
 accuracy_13_22: - 50.9765625 frac[13,22] + mem[13,22] = 0
 accuracy_2_29: - 50.9765625 frac[2,29] + mem[2,29] = 0
 accuracy_0_53: - 50.9765625 frac[0,53] + mem[0,53] = 0
 accuracy_14_37: - 50.9765625 frac[14,37] + mem[14,37] = 0
 accuracy_3_0: - 50.9765625 frac[3,0] + mem[3,0] = 0
 accuracy_1_48: - 50.9765625 frac[1,48] + mem[1,48] = 0
 accuracy_4_27: - 50.9765625 frac[4,27] + mem[4,27] = 0
 accuracy_2_55: - 50.9765625 frac[2,55] + mem[2,55] = 0
 accuracy_0_43: - 50.9765625 frac[0,43] + mem[0,43] = 0
 accuracy_5_26: - 50.9765625 frac[5,26] + mem[5,26] = 0
 accuracy_3_38: - 50.9765625 frac[3,38] + mem[3,38] = 0
 accuracy_1_42: - 50.9765625 frac[1,42] + mem[1,42] = 0
 accuracy_6_17: - 50.9765625 frac[6,17] + mem[6,17] = 0
 accuracy_4_33: - 50.9765625 frac[4,33] + mem[4,33] = 0
 accuracy_8_36: - 50.9765625 frac[8,36] + mem[8,36] = 0
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
 accuracy_0_22: - 50.9765625 frac[0,22] + mem[0,22] = 0
 accuracy_14_10: - 50.9765625 frac[14,10] + mem[14,10] = 0
 accuracy_12_14: - 50.9765625 frac[12,14] + mem[12,14] = 0
 accuracy_13_15: - 50.9765625 frac[13,15] + mem[13,15] = 0
 accuracy_2_20: - 50.9765625 frac[2,20] + mem[2,20] = 0
 accuracy_0_12: - 50.9765625 frac[0,12] + mem[0,12] = 0
 accuracy_14_44: - 50.9765625 frac[14,44] + mem[14,44] = 0
 accuracy_1_9: - 50.9765625 frac[1,9] + mem[1,9] = 0
 accuracy_4_2: - 50.9765625 frac[4,2] + mem[4,2] = 0
 accuracy_5_3: - 50.9765625 frac[5,3] + mem[5,3] = 0
 accuracy_3_47: - 50.9765625 frac[3,47] + mem[3,47] = 0
 accuracy_1_35: - 50.9765625 frac[1,35] + mem[1,35] = 0
 accuracy_6_8: - 50.9765625 frac[6,8] + mem[6,8] = 0
 accuracy_4_40: - 50.9765625 frac[4,40] + mem[4,40] = 0
 accuracy_2_40: - 50.9765625 frac[2,40] + mem[2,40] = 0
 accuracy_8_43: - 50.9765625 frac[8,43] + mem[8,43] = 0
 accuracy_7_13: - 50.9765625 frac[7,13] + mem[7,13] = 0
 accuracy_5_45: - 50.9765625 frac[5,45] + mem[5,45] = 0
 accuracy_9_42: - 50.9765625 frac[9,42] + mem[9,42] = 0
 accuracy_6_34: - 50.9765625 frac[6,34] + mem[6,34] = 0
 accuracy_4_54: - 50.9765625 frac[4,54] + mem[4,54] = 0
 accuracy_10_33: - 50.9765625 frac[10,33] + mem[10,33] = 0
 accuracy_8_49: - 50.9765625 frac[8,49] + mem[8,49] = 0
 accuracy_5_55: - 50.9765625 frac[5,55] + mem[5,55] = 0
 accuracy_11_44: - 50.9765625 frac[11,44] + mem[11,44] = 0
 accuracy_9_12: - 50.9765625 frac[9,12] + mem[9,12] = 0
 accuracy_12_47: - 50.9765625 frac[12,47] + mem[12,47] = 0
 accuracy_10_11: - 50.9765625 frac[10,11] + mem[10,11] = 0
 accuracy_8_23: - 50.9765625 frac[8,23] + mem[8,23] = 0
 accuracy_13_46: - 50.9765625 frac[13,46] + mem[13,46] = 0
 accuracy_11_2: - 50.9765625 frac[11,2] + mem[11,2] = 0
 accuracy_9_22: - 50.9765625 frac[9,22] + mem[9,22] = 0
 accuracy_14_13: - 50.9765625 frac[14,13] + mem[14,13] = 0
 accuracy_12_5: - 50.9765625 frac[12,5] + mem[12,5] = 0
 accuracy_10_21: - 50.9765625 frac[10,21] + mem[10,21] = 0
 accuracy_13_8: - 50.9765625 frac[13,8] + mem[13,8] = 0
 accuracy_11_24: - 50.9765625 frac[11,24] + mem[11,24] = 0
 accuracy_0_3: - 50.9765625 frac[0,3] + mem[0,3] = 0
 accuracy_14_55: - 50.9765625 frac[14,55] + mem[14,55] = 0
 accuracy_2_9: - 50.9765625 frac[2,9] + mem[2,9] = 0
 accuracy_5_12: - 50.9765625 frac[5,12] + mem[5,12] = 0
 accuracy_3_20: - 50.9765625 frac[3,20] + mem[3,20] = 0
 accuracy_1_36: - 50.9765625 frac[1,36] + mem[1,36] = 0
 accuracy_6_3: - 50.9765625 frac[6,3] + mem[6,3] = 0
 accuracy_2_35: - 50.9765625 frac[2,35] + mem[2,35] = 0
 accuracy_7_10: - 50.9765625 frac[7,10] + mem[7,10] = 0
 accuracy_10_40: - 50.9765625 frac[10,40] + mem[10,40] = 0
 accuracy_8_56: - 50.9765625 frac[8,56] + mem[8,56] = 0
 accuracy_7_32: - 50.9765625 frac[7,32] + mem[7,32] = 0
 accuracy_5_48: - 50.9765625 frac[5,48] + mem[5,48] = 0
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
 accuracy_14_62: - 50.9765625 frac[14,62] + mem[14,62] = 0
 accuracy_12_26: - 50.9765625 frac[12,26] + mem[12,26] = 0
 accuracy_1_27: - 50.9765625 frac[1,27] + mem[1,27] = 0
 accuracy_13_27: - 50.9765625 frac[13,27] + mem[13,27] = 0
 accuracy_2_0: - 50.9765625 frac[2,0] + mem[2,0] = 0
 accuracy_0_48: - 50.9765625 frac[0,48] + mem[0,48] = 0
 accuracy_14_32: - 50.9765625 frac[14,32] + mem[14,32] = 0
 accuracy_3_29: - 50.9765625 frac[3,29] + mem[3,29] = 0
 accuracy_2_58: - 50.9765625 frac[2,58] + mem[2,58] = 0
 mem_tot_CPU1: - mem[0,17] - mem[0,55] - mem[0,45] - mem[0,14] - mem[0,5]
   - mem[0,59] - mem[0,28] - mem[0,40] - mem[0,47] - mem[0,0] - mem[0,38]
   - mem[0,7] - mem[0,30] - mem[0,42] - mem[0,21] - mem[0,2] - mem[0,56]
   - mem[0,16] - mem[0,44] - mem[0,23] - mem[0,35] - mem[0,4] - mem[0,27]
   - mem[0,49] - mem[0,18] - mem[0,46] - mem[0,37] - mem[0,6] - mem[0,60]
   - mem[0,29] - mem[0,51] - mem[0,41] - mem[0,10] - mem[0,32] - mem[0,62]
   - mem[0,53] - mem[0,43] - mem[0,22] - mem[0,12] - mem[0,3] - mem[0,26]
   - mem[0,48] + mem_tot_CPU1 = 0
 rows_tot_CPU1: - 3 frac[0,17] - 3 frac[0,55] - 3 frac[0,45] - 3 frac[0,14]
   - 3 frac[0,5] - 3 frac[0,59] - 3 frac[0,28] - 3 frac[0,40]
   - 3 frac[0,47] - 3 frac[0,0] - 3 frac[0,38] - 3 frac[0,7] - 3 frac[0,30]
   - 3 frac[0,42] - 3 frac[0,21] - 3 frac[0,2] - 3 frac[0,56]
   - 3 frac[0,16] - 3 frac[0,44] - 3 frac[0,23] - 3 frac[0,35]
   - 3 frac[0,4] - 3 frac[0,27] - 3 frac[0,49] - 3 frac[0,18]
   - 3 frac[0,46] - 3 frac[0,37] - 3 frac[0,6] - 3 frac[0,60]
   - 3 frac[0,29] - 3 frac[0,51] - 3 frac[0,41] - 3 frac[0,10]
   - 3 frac[0,32] - 3 frac[0,62] - 3 frac[0,53] - 3 frac[0,43]
   - 3 frac[0,22] - 3 frac[0,12] - 3 frac[0,3] - 3 frac[0,26]
   - 3 frac[0,48] + rows_tot_CPU1 = 0
 mem_tot_CPU2: - mem[1,28] - mem[1,54] - mem[1,40] - mem[1,15] - mem[1,33]
   - mem[1,0] - mem[1,58] - mem[1,25] - mem[1,51] - mem[1,18] - mem[1,46]
   - mem[1,39] - mem[1,49] - mem[1,16] - mem[1,10] - mem[1,44] - mem[1,3]
   - mem[1,37] - mem[1,4] - mem[1,62] - mem[1,29] - mem[1,55] - mem[1,22]
   - mem[1,8] - mem[1,34] - mem[1,1] - mem[1,59] - mem[1,60] - mem[1,19]
   - mem[1,53] - mem[1,47] - mem[1,20] - mem[1,32] - mem[1,57] - mem[1,24]
   - mem[1,50] - mem[1,17] - mem[1,11] - mem[1,38] - mem[1,5] - mem[1,63]
   - mem[1,30] - mem[1,48] - mem[1,42] - mem[1,9] - mem[1,35] - mem[1,36]
   - mem[1,27] + mem_tot_CPU2 = 0
 rows_tot_CPU2: - 3 frac[1,28] - 3 frac[1,54] - 3 frac[1,40] - 3 frac[1,15]
   - 3 frac[1,33] - 3 frac[1,0] - 3 frac[1,58] - 3 frac[1,25]
   - 3 frac[1,51] - 3 frac[1,18] - 3 frac[1,46] - 3 frac[1,39]
   - 3 frac[1,49] - 3 frac[1,16] - 3 frac[1,10] - 3 frac[1,44]
   - 3 frac[1,3] - 3 frac[1,37] - 3 frac[1,4] - 3 frac[1,62] - 3 frac[1,29]
   - 3 frac[1,55] - 3 frac[1,22] - 3 frac[1,8] - 3 frac[1,34] - 3 frac[1,1]
   - 3 frac[1,59] - 3 frac[1,60] - 3 frac[1,19] - 3 frac[1,53]
   - 3 frac[1,47] - 3 frac[1,20] - 3 frac[1,32] - 3 frac[1,57]
   - 3 frac[1,24] - 3 frac[1,50] - 3 frac[1,17] - 3 frac[1,11]
   - 3 frac[1,38] - 3 frac[1,5] - 3 frac[1,63] - 3 frac[1,30]
   - 3 frac[1,48] - 3 frac[1,42] - 3 frac[1,9] - 3 frac[1,35]
   - 3 frac[1,36] - 3 frac[1,27] + rows_tot_CPU2 = 0
 mem_tot_CPU3: - mem[2,27] - mem[2,53] - mem[2,18] - mem[2,12] - mem[2,7]
   - mem[2,33] - mem[2,30] - mem[2,56] - mem[2,25] - mem[2,51] - mem[2,16]
   - mem[2,10] - mem[2,36] - mem[2,5] - mem[2,63] - mem[2,28] - mem[2,23]
   - mem[2,49] - mem[2,43] - mem[2,8] - mem[2,34] - mem[2,3] - mem[2,61]
   - mem[2,52] - mem[2,46] - mem[2,21] - mem[2,15] - mem[2,41] - mem[2,6]
   - mem[2,32] - mem[2,1] - mem[2,24] - mem[2,44] - mem[2,13] - mem[2,39]
   - mem[2,4] - mem[2,62] - mem[2,31] - mem[2,22] - mem[2,48] - mem[2,42]
   - mem[2,17] - mem[2,11] - mem[2,37] - mem[2,2] - mem[2,60] - mem[2,29]
   - mem[2,55] - mem[2,20] - mem[2,40] - mem[2,9] - mem[2,35] - mem[2,0]
   - mem[2,58] + mem_tot_CPU3 = 0
 rows_tot_CPU3: - 3 frac[2,27] - 3 frac[2,53] - 3 frac[2,18] - 3 frac[2,12]
   - 3 frac[2,7] - 3 frac[2,33] - 3 frac[2,30] - 3 frac[2,56]
   - 3 frac[2,25] - 3 frac[2,51] - 3 frac[2,16] - 3 frac[2,10]
   - 3 frac[2,36] - 3 frac[2,5] - 3 frac[2,63] - 3 frac[2,28]
   - 3 frac[2,23] - 3 frac[2,49] - 3 frac[2,43] - 3 frac[2,8]
   - 3 frac[2,34] - 3 frac[2,3] - 3 frac[2,61] - 3 frac[2,52]
   - 3 frac[2,46] - 3 frac[2,21] - 3 frac[2,15] - 3 frac[2,41]
   - 3 frac[2,6] - 3 frac[2,32] - 3 frac[2,1] - 3 frac[2,24] - 3 frac[2,44]
   - 3 frac[2,13] - 3 frac[2,39] - 3 frac[2,4] - 3 frac[2,62]
   - 3 frac[2,31] - 3 frac[2,22] - 3 frac[2,48] - 3 frac[2,42]
   - 3 frac[2,17] - 3 frac[2,11] - 3 frac[2,37] - 3 frac[2,2]
   - 3 frac[2,60] - 3 frac[2,29] - 3 frac[2,55] - 3 frac[2,20]
   - 3 frac[2,40] - 3 frac[2,9] - 3 frac[2,35] - 3 frac[2,0] - 3 frac[2,58]
   + rows_tot_CPU3 = 0
 mem_tot_CPU4: - mem[3,2] - mem[3,40] - mem[3,11] - mem[3,17] - mem[3,55]
   - mem[3,22] - mem[3,60] - mem[3,37] - mem[3,4] - mem[3,48] - mem[3,13]
   - mem[3,19] - mem[3,24] - mem[3,62] - mem[3,1] - mem[3,39] - mem[3,44]
   - mem[3,50] - mem[3,15] - mem[3,21] - mem[3,59] - mem[3,26] - mem[3,32]
   - mem[3,3] - mem[3,41] - mem[3,8] - mem[3,46] - mem[3,52] - mem[3,23]
   - mem[3,28] - mem[3,34] - mem[3,5] - mem[3,43] - mem[3,49] - mem[3,10]
   - mem[3,16] - mem[3,54] - mem[3,25] - mem[3,30] - mem[3,7] - mem[3,45]
   - mem[3,51] - mem[3,56] - mem[3,0] - mem[3,38] - mem[3,47] - mem[3,20]
   - mem[3,29] + mem_tot_CPU4 = 0
 rows_tot_CPU4: - 3 frac[3,2] - 3 frac[3,40] - 3 frac[3,11] - 3 frac[3,17]
   - 3 frac[3,55] - 3 frac[3,22] - 3 frac[3,60] - 3 frac[3,37]
   - 3 frac[3,4] - 3 frac[3,48] - 3 frac[3,13] - 3 frac[3,19]
   - 3 frac[3,24] - 3 frac[3,62] - 3 frac[3,1] - 3 frac[3,39]
   - 3 frac[3,44] - 3 frac[3,50] - 3 frac[3,15] - 3 frac[3,21]
   - 3 frac[3,59] - 3 frac[3,26] - 3 frac[3,32] - 3 frac[3,3]
   - 3 frac[3,41] - 3 frac[3,8] - 3 frac[3,46] - 3 frac[3,52]
   - 3 frac[3,23] - 3 frac[3,28] - 3 frac[3,34] - 3 frac[3,5]
   - 3 frac[3,43] - 3 frac[3,49] - 3 frac[3,10] - 3 frac[3,16]
   - 3 frac[3,54] - 3 frac[3,25] - 3 frac[3,30] - 3 frac[3,7]
   - 3 frac[3,45] - 3 frac[3,51] - 3 frac[3,56] - 3 frac[3,0]
   - 3 frac[3,38] - 3 frac[3,47] - 3 frac[3,20] - 3 frac[3,29]
   + rows_tot_CPU4 = 0
 mem_tot_CPU5: - mem[4,5] - mem[4,35] - mem[4,12] - mem[4,42] - mem[4,11]
   - mem[4,63] - mem[4,24] - mem[4,38] - mem[4,7] - mem[4,20] - mem[4,19]
   - mem[4,57] - mem[4,26] - mem[4,32] - mem[4,1] - mem[4,47] - mem[4,53]
   - mem[4,22] - mem[4,29] - mem[4,59] - mem[4,4] - mem[4,3] - mem[4,41]
   - mem[4,55] - mem[4,10] - mem[4,16] - mem[4,62] - mem[4,31] - mem[4,37]
   - mem[4,6] - mem[4,43] - mem[4,18] - mem[4,56] - mem[4,39] - mem[4,0]
   - mem[4,46] - mem[4,15] - mem[4,28] - mem[4,58] - mem[4,27] - mem[4,33]
   - mem[4,2] - mem[4,40] - mem[4,54] + mem_tot_CPU5 = 0
 rows_tot_CPU5: - 3 frac[4,5] - 3 frac[4,35] - 3 frac[4,12] - 3 frac[4,42]
   - 3 frac[4,11] - 3 frac[4,63] - 3 frac[4,24] - 3 frac[4,38]
   - 3 frac[4,7] - 3 frac[4,20] - 3 frac[4,19] - 3 frac[4,57]
   - 3 frac[4,26] - 3 frac[4,32] - 3 frac[4,1] - 3 frac[4,47]
   - 3 frac[4,53] - 3 frac[4,22] - 3 frac[4,29] - 3 frac[4,59]
   - 3 frac[4,4] - 3 frac[4,3] - 3 frac[4,41] - 3 frac[4,55] - 3 frac[4,10]
   - 3 frac[4,16] - 3 frac[4,62] - 3 frac[4,31] - 3 frac[4,37]
   - 3 frac[4,6] - 3 frac[4,43] - 3 frac[4,18] - 3 frac[4,56]
   - 3 frac[4,39] - 3 frac[4,0] - 3 frac[4,46] - 3 frac[4,15]
   - 3 frac[4,28] - 3 frac[4,58] - 3 frac[4,27] - 3 frac[4,33]
   - 3 frac[4,2] - 3 frac[4,40] - 3 frac[4,54] + rows_tot_CPU5 = 0
 mem_tot_CPU6: - mem[5,31] - mem[5,57] - mem[5,24] - mem[5,34] - mem[5,1]
   - mem[5,53] - mem[5,10] - mem[5,20] - mem[5,39] - mem[5,6] - mem[5,32]
   - mem[5,15] - mem[5,41] - mem[5,18] - mem[5,60] - mem[5,27] - mem[5,37]
   - mem[5,4] - mem[5,46] - mem[5,13] - mem[5,23] - mem[5,16] - mem[5,58]
   - mem[5,35] - mem[5,2] - mem[5,44] - mem[5,54] - mem[5,11] - mem[5,21]
   - mem[5,30] - mem[5,7] - mem[5,42] - mem[5,52] - mem[5,9] - mem[5,19]
   - mem[5,28] - mem[5,5] - mem[5,47] - mem[5,14] - mem[5,40] - mem[5,50]
   - mem[5,17] - mem[5,59] - mem[5,26] - mem[5,36] - mem[5,3] - mem[5,45]
   - mem[5,55] - mem[5,12] - mem[5,48] + mem_tot_CPU6 = 0
 rows_tot_CPU6: - 3 frac[5,31] - 3 frac[5,57] - 3 frac[5,24] - 3 frac[5,34]
   - 3 frac[5,1] - 3 frac[5,53] - 3 frac[5,10] - 3 frac[5,20]
   - 3 frac[5,39] - 3 frac[5,6] - 3 frac[5,32] - 3 frac[5,15]
   - 3 frac[5,41] - 3 frac[5,18] - 3 frac[5,60] - 3 frac[5,27]
   - 3 frac[5,37] - 3 frac[5,4] - 3 frac[5,46] - 3 frac[5,13]
   - 3 frac[5,23] - 3 frac[5,16] - 3 frac[5,58] - 3 frac[5,35]
   - 3 frac[5,2] - 3 frac[5,44] - 3 frac[5,54] - 3 frac[5,11]
   - 3 frac[5,21] - 3 frac[5,30] - 3 frac[5,7] - 3 frac[5,42]
   - 3 frac[5,52] - 3 frac[5,9] - 3 frac[5,19] - 3 frac[5,28] - 3 frac[5,5]
   - 3 frac[5,47] - 3 frac[5,14] - 3 frac[5,40] - 3 frac[5,50]
   - 3 frac[5,17] - 3 frac[5,59] - 3 frac[5,26] - 3 frac[5,36]
   - 3 frac[5,3] - 3 frac[5,45] - 3 frac[5,55] - 3 frac[5,12]
   - 3 frac[5,48] + rows_tot_CPU6 = 0
 mem_tot_CPU7: - mem[6,28] - mem[6,54] - mem[6,23] - mem[6,41] - mem[6,14]
   - mem[6,32] - mem[6,58] - mem[6,1] - mem[6,27] - mem[6,18] - mem[6,52]
   - mem[6,47] - mem[6,12] - mem[6,38] - mem[6,56] - mem[6,51] - mem[6,16]
   - mem[6,42] - mem[6,11] - mem[6,2] - mem[6,36] - mem[6,5] - mem[6,49]
   - mem[6,40] - mem[6,35] - mem[6,0] - mem[6,26] - mem[6,29] - mem[6,55]
   - mem[6,20] - mem[6,46] - mem[6,33] - mem[6,59] - mem[6,6] - mem[6,19]
   - mem[6,53] - mem[6,10] - mem[6,57] - mem[6,4] - mem[6,30] - mem[6,48]
   - mem[6,17] - mem[6,43] - mem[6,8] - mem[6,34] - mem[6,3] - mem[6,63]
   + mem_tot_CPU7 = 0
 rows_tot_CPU7: - 3 frac[6,28] - 3 frac[6,54] - 3 frac[6,23] - 3 frac[6,41]
   - 3 frac[6,14] - 3 frac[6,32] - 3 frac[6,58] - 3 frac[6,1]
   - 3 frac[6,27] - 3 frac[6,18] - 3 frac[6,52] - 3 frac[6,47]
   - 3 frac[6,12] - 3 frac[6,38] - 3 frac[6,56] - 3 frac[6,51]
   - 3 frac[6,16] - 3 frac[6,42] - 3 frac[6,11] - 3 frac[6,2]
   - 3 frac[6,36] - 3 frac[6,5] - 3 frac[6,49] - 3 frac[6,40]
   - 3 frac[6,35] - 3 frac[6,0] - 3 frac[6,26] - 3 frac[6,29]
   - 3 frac[6,55] - 3 frac[6,20] - 3 frac[6,46] - 3 frac[6,33]
   - 3 frac[6,59] - 3 frac[6,6] - 3 frac[6,19] - 3 frac[6,53]
   - 3 frac[6,10] - 3 frac[6,57] - 3 frac[6,4] - 3 frac[6,30]
   - 3 frac[6,48] - 3 frac[6,17] - 3 frac[6,43] - 3 frac[6,8]
   - 3 frac[6,34] - 3 frac[6,3] - 3 frac[6,63] + rows_tot_CPU7 = 0
 mem_tot_CPU8: - mem[7,3] - mem[7,55] - mem[7,44] - mem[7,15] - mem[7,37]
   - mem[7,4] - mem[7,56] - mem[7,27] - mem[7,16] - mem[7,9] - mem[7,39]
   - mem[7,61] - mem[7,6] - mem[7,28] - mem[7,58] - mem[7,18] - mem[7,11]
   - mem[7,33] - mem[7,63] - mem[7,30] - mem[7,52] - mem[7,23] - mem[7,45]
   - mem[7,42] - mem[7,5] - mem[7,35] - mem[7,57] - mem[7,2] - mem[7,24]
   - mem[7,54] - mem[7,17] - mem[7,36] - mem[7,7] - mem[7,29] - mem[7,59]
   - mem[7,26] - mem[7,48] - mem[7,19] - mem[7,41] - mem[7,38] - mem[7,60]
   - mem[7,1] - mem[7,53] - mem[7,50] - mem[7,13] - mem[7,10] - mem[7,32]
   + mem_tot_CPU8 = 0
 rows_tot_CPU8: - 3 frac[7,3] - 3 frac[7,55] - 3 frac[7,44] - 3 frac[7,15]
   - 3 frac[7,37] - 3 frac[7,4] - 3 frac[7,56] - 3 frac[7,27]
   - 3 frac[7,16] - 3 frac[7,9] - 3 frac[7,39] - 3 frac[7,61] - 3 frac[7,6]
   - 3 frac[7,28] - 3 frac[7,58] - 3 frac[7,18] - 3 frac[7,11]
   - 3 frac[7,33] - 3 frac[7,63] - 3 frac[7,30] - 3 frac[7,52]
   - 3 frac[7,23] - 3 frac[7,45] - 3 frac[7,42] - 3 frac[7,5]
   - 3 frac[7,35] - 3 frac[7,57] - 3 frac[7,2] - 3 frac[7,24]
   - 3 frac[7,54] - 3 frac[7,17] - 3 frac[7,36] - 3 frac[7,7]
   - 3 frac[7,29] - 3 frac[7,59] - 3 frac[7,26] - 3 frac[7,48]
   - 3 frac[7,19] - 3 frac[7,41] - 3 frac[7,38] - 3 frac[7,60]
   - 3 frac[7,1] - 3 frac[7,53] - 3 frac[7,50] - 3 frac[7,13]
   - 3 frac[7,10] - 3 frac[7,32] + rows_tot_CPU8 = 0
 mem_tot_tor_P41: - mem[8,63] - mem[8,5] - mem[8,38] - mem[8,12]
   - mem[8,18] - mem[8,45] - mem[8,51] - mem[8,25] - mem[8,58] - mem[8,0]
   - mem[8,33] - mem[8,7] - mem[8,40] - mem[8,14] - mem[8,20] - mem[8,47]
   - mem[8,53] - mem[8,27] - mem[8,60] - mem[8,2] - mem[8,35] - mem[8,9]
   - mem[8,42] - mem[8,48] - mem[8,22] - mem[8,55] - mem[8,29] - mem[8,62]
   - mem[8,4] - mem[8,37] - mem[8,11] - mem[8,17] - mem[8,44] - mem[8,50]
   - mem[8,24] - mem[8,57] - mem[8,32] - mem[8,6] - mem[8,39] - mem[8,19]
   - mem[8,46] - mem[8,26] - mem[8,59] - mem[8,1] - mem[8,34] - mem[8,8]
   - mem[8,41] - mem[8,15] - mem[8,21] - mem[8,54] - mem[8,28] - mem[8,3]
   - mem[8,36] - mem[8,10] - mem[8,16] - mem[8,43] - mem[8,49] - mem[8,23]
   - mem[8,56] - mem[8,30] + mem_tot_tor_P41 = 0
 rows_tot_tor_P41: - 3 frac[8,63] - 3 frac[8,5] - 3 frac[8,38]
   - 3 frac[8,12] - 3 frac[8,18] - 3 frac[8,45] - 3 frac[8,51]
   - 3 frac[8,25] - 3 frac[8,58] - 3 frac[8,0] - 3 frac[8,33] - 3 frac[8,7]
   - 3 frac[8,40] - 3 frac[8,14] - 3 frac[8,20] - 3 frac[8,47]
   - 3 frac[8,53] - 3 frac[8,27] - 3 frac[8,60] - 3 frac[8,2]
   - 3 frac[8,35] - 3 frac[8,9] - 3 frac[8,42] - 3 frac[8,48]
   - 3 frac[8,22] - 3 frac[8,55] - 3 frac[8,29] - 3 frac[8,62]
   - 3 frac[8,4] - 3 frac[8,37] - 3 frac[8,11] - 3 frac[8,17]
   - 3 frac[8,44] - 3 frac[8,50] - 3 frac[8,24] - 3 frac[8,57]
   - 3 frac[8,32] - 3 frac[8,6] - 3 frac[8,39] - 3 frac[8,19]
   - 3 frac[8,46] - 3 frac[8,26] - 3 frac[8,59] - 3 frac[8,1]
   - 3 frac[8,34] - 3 frac[8,8] - 3 frac[8,41] - 3 frac[8,15]
   - 3 frac[8,21] - 3 frac[8,54] - 3 frac[8,28] - 3 frac[8,3]
   - 3 frac[8,36] - 3 frac[8,10] - 3 frac[8,16] - 3 frac[8,43]
   - 3 frac[8,49] - 3 frac[8,23] - 3 frac[8,56] - 3 frac[8,30]
   + rows_tot_tor_P41 = 0
 mem_tot_tor_P42: - mem[9,62] - mem[9,0] - mem[9,39] - mem[9,9] - mem[9,19]
   - mem[9,40] - mem[9,50] - mem[9,20] - mem[9,59] - mem[9,29] - mem[9,60]
   - mem[9,6] - mem[9,37] - mem[9,15] - mem[9,17] - mem[9,46] - mem[9,48]
   - mem[9,26] - mem[9,3] - mem[9,34] - mem[9,4] - mem[9,43] - mem[9,13]
   - mem[9,23] - mem[9,44] - mem[9,54] - mem[9,24] - mem[9,63] - mem[9,1]
   - mem[9,32] - mem[9,10] - mem[9,41] - mem[9,51] - mem[9,21] - mem[9,52]
   - mem[9,30] - mem[9,61] - mem[9,7] - mem[9,38] - mem[9,8] - mem[9,18]
   - mem[9,47] - mem[9,49] - mem[9,27] - mem[9,58] - mem[9,28] - mem[9,35]
   - mem[9,5] - mem[9,36] - mem[9,16] - mem[9,45] - mem[9,55] - mem[9,25]
   - mem[9,56] - mem[9,2] - mem[9,33] - mem[9,11] - mem[9,42] - mem[9,12]
   - mem[9,22] - mem[9,53] - mem[9,31] + mem_tot_tor_P42 = 0
 rows_tot_tor_P42: - 3 frac[9,62] - 3 frac[9,0] - 3 frac[9,39]
   - 3 frac[9,9] - 3 frac[9,19] - 3 frac[9,40] - 3 frac[9,50]
   - 3 frac[9,20] - 3 frac[9,59] - 3 frac[9,29] - 3 frac[9,60]
   - 3 frac[9,6] - 3 frac[9,37] - 3 frac[9,15] - 3 frac[9,17]
   - 3 frac[9,46] - 3 frac[9,48] - 3 frac[9,26] - 3 frac[9,3]
   - 3 frac[9,34] - 3 frac[9,4] - 3 frac[9,43] - 3 frac[9,13]
   - 3 frac[9,23] - 3 frac[9,44] - 3 frac[9,54] - 3 frac[9,24]
   - 3 frac[9,63] - 3 frac[9,1] - 3 frac[9,32] - 3 frac[9,10]
   - 3 frac[9,41] - 3 frac[9,51] - 3 frac[9,21] - 3 frac[9,52]
   - 3 frac[9,30] - 3 frac[9,61] - 3 frac[9,7] - 3 frac[9,38] - 3 frac[9,8]
   - 3 frac[9,18] - 3 frac[9,47] - 3 frac[9,49] - 3 frac[9,27]
   - 3 frac[9,58] - 3 frac[9,28] - 3 frac[9,35] - 3 frac[9,5]
   - 3 frac[9,36] - 3 frac[9,16] - 3 frac[9,45] - 3 frac[9,55]
   - 3 frac[9,25] - 3 frac[9,56] - 3 frac[9,2] - 3 frac[9,33]
   - 3 frac[9,11] - 3 frac[9,42] - 3 frac[9,12] - 3 frac[9,22]
   - 3 frac[9,53] - 3 frac[9,31] + rows_tot_tor_P42 = 0
 mem_tot_tor_P43: - mem[10,7] - mem[10,36] - mem[10,14] - mem[10,47]
   - mem[10,9] - mem[10,19] - mem[10,48] - mem[10,26] - mem[10,59]
   - mem[10,5] - mem[10,34] - mem[10,12] - mem[10,22] - mem[10,45]
   - mem[10,55] - mem[10,17] - mem[10,62] - mem[10,24] - mem[10,57]
   - mem[10,3] - mem[10,32] - mem[10,10] - mem[10,20] - mem[10,43]
   - mem[10,53] - mem[10,31] - mem[10,60] - mem[10,6] - mem[10,39]
   - mem[10,1] - mem[10,46] - mem[10,18] - mem[10,41] - mem[10,29]
   - mem[10,58] - mem[10,4] - mem[10,37] - mem[10,15] - mem[10,44]
   - mem[10,54] - mem[10,16] - mem[10,27] - mem[10,56] - mem[10,2]
   - mem[10,35] - mem[10,13] - mem[10,23] - mem[10,42] - mem[10,52]
   - mem[10,30] - mem[10,63] - mem[10,38] - mem[10,0] - mem[10,33]
   - mem[10,11] - mem[10,21] - mem[10,40] - mem[10,50] - mem[10,28]
   + mem_tot_tor_P43 = 0
 rows_tot_tor_P43: - 3 frac[10,7] - 3 frac[10,36] - 3 frac[10,14]
   - 3 frac[10,47] - 3 frac[10,9] - 3 frac[10,19] - 3 frac[10,48]
   - 3 frac[10,26] - 3 frac[10,59] - 3 frac[10,5] - 3 frac[10,34]
   - 3 frac[10,12] - 3 frac[10,22] - 3 frac[10,45] - 3 frac[10,55]
   - 3 frac[10,17] - 3 frac[10,62] - 3 frac[10,24] - 3 frac[10,57]
   - 3 frac[10,3] - 3 frac[10,32] - 3 frac[10,10] - 3 frac[10,20]
   - 3 frac[10,43] - 3 frac[10,53] - 3 frac[10,31] - 3 frac[10,60]
   - 3 frac[10,6] - 3 frac[10,39] - 3 frac[10,1] - 3 frac[10,46]
   - 3 frac[10,18] - 3 frac[10,41] - 3 frac[10,29] - 3 frac[10,58]
   - 3 frac[10,4] - 3 frac[10,37] - 3 frac[10,15] - 3 frac[10,44]
   - 3 frac[10,54] - 3 frac[10,16] - 3 frac[10,27] - 3 frac[10,56]
   - 3 frac[10,2] - 3 frac[10,35] - 3 frac[10,13] - 3 frac[10,23]
   - 3 frac[10,42] - 3 frac[10,52] - 3 frac[10,30] - 3 frac[10,63]
   - 3 frac[10,38] - 3 frac[10,0] - 3 frac[10,33] - 3 frac[10,11]
   - 3 frac[10,21] - 3 frac[10,40] - 3 frac[10,50] - 3 frac[10,28]
   + rows_tot_tor_P43 = 0
 mem_tot_tor_P44: - mem[11,48] - mem[11,41] - mem[11,15] - mem[11,46]
   - mem[11,4] - mem[11,26] - mem[11,39] - mem[11,61] - mem[11,19]
   - mem[11,50] - mem[11,8] - mem[11,43] - mem[11,1] - mem[11,32]
   - mem[11,6] - mem[11,28] - mem[11,63] - mem[11,52] - mem[11,10]
   - mem[11,45] - mem[11,3] - mem[11,34] - mem[11,56] - mem[11,30]
   - mem[11,49] - mem[11,23] - mem[11,54] - mem[11,12] - mem[11,47]
   - mem[11,5] - mem[11,27] - mem[11,36] - mem[11,58] - mem[11,16]
   - mem[11,51] - mem[11,9] - mem[11,40] - mem[11,14] - mem[11,33]
   - mem[11,7] - mem[11,29] - mem[11,38] - mem[11,60] - mem[11,18]
   - mem[11,53] - mem[11,11] - mem[11,42] - mem[11,0] - mem[11,35]
   - mem[11,57] - mem[11,20] - mem[11,55] - mem[11,13] - mem[11,44]
   - mem[11,2] - mem[11,24] - mem[11,37] - mem[11,59] - mem[11,17]
   + mem_tot_tor_P44 = 0
 rows_tot_tor_P44: - 3 frac[11,48] - 3 frac[11,41] - 3 frac[11,15]
   - 3 frac[11,46] - 3 frac[11,4] - 3 frac[11,26] - 3 frac[11,39]
   - 3 frac[11,61] - 3 frac[11,19] - 3 frac[11,50] - 3 frac[11,8]
   - 3 frac[11,43] - 3 frac[11,1] - 3 frac[11,32] - 3 frac[11,6]
   - 3 frac[11,28] - 3 frac[11,63] - 3 frac[11,52] - 3 frac[11,10]
   - 3 frac[11,45] - 3 frac[11,3] - 3 frac[11,34] - 3 frac[11,56]
   - 3 frac[11,30] - 3 frac[11,49] - 3 frac[11,23] - 3 frac[11,54]
   - 3 frac[11,12] - 3 frac[11,47] - 3 frac[11,5] - 3 frac[11,27]
   - 3 frac[11,36] - 3 frac[11,58] - 3 frac[11,16] - 3 frac[11,51]
   - 3 frac[11,9] - 3 frac[11,40] - 3 frac[11,14] - 3 frac[11,33]
   - 3 frac[11,7] - 3 frac[11,29] - 3 frac[11,38] - 3 frac[11,60]
   - 3 frac[11,18] - 3 frac[11,53] - 3 frac[11,11] - 3 frac[11,42]
   - 3 frac[11,0] - 3 frac[11,35] - 3 frac[11,57] - 3 frac[11,20]
   - 3 frac[11,55] - 3 frac[11,13] - 3 frac[11,44] - 3 frac[11,2]
   - 3 frac[11,24] - 3 frac[11,37] - 3 frac[11,59] - 3 frac[11,17]
   + rows_tot_tor_P44 = 0
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
   - mem[12,45] - mem[12,11] - mem[12,36] - mem[12,2] - mem[12,24]
   - mem[12,35] - mem[12,57] - mem[12,23] - mem[12,48] - mem[12,14]
   - mem[12,47] - mem[12,5] - mem[12,38] - mem[12,60] - mem[12,26]
   + mem_tot_l1_P41 = 0
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
   - 3 frac[12,54] - 3 frac[12,12] - 3 frac[12,45] - 3 frac[12,11]
   - 3 frac[12,36] - 3 frac[12,2] - 3 frac[12,24] - 3 frac[12,35]
   - 3 frac[12,57] - 3 frac[12,23] - 3 frac[12,48] - 3 frac[12,14]
   - 3 frac[12,47] - 3 frac[12,5] - 3 frac[12,38] - 3 frac[12,60]
   - 3 frac[12,26] + rows_tot_l1_P41 = 0
 mem_tot_l1_P42: - mem[13,32] - mem[13,58] - mem[13,20] - mem[13,51]
   - mem[13,13] - mem[13,44] - mem[13,6] - mem[13,37] - mem[13,63]
   - mem[13,56] - mem[13,18] - mem[13,49] - mem[13,11] - mem[13,42]
   - mem[13,4] - mem[13,30] - mem[13,35] - mem[13,61] - mem[13,23]
   - mem[13,54] - mem[13,16] - mem[13,47] - mem[13,9] - mem[13,40]
   - mem[13,2] - mem[13,28] - mem[13,33] - mem[13,59] - mem[13,21]
   - mem[13,52] - mem[13,14] - mem[13,45] - mem[13,7] - mem[13,38]
   - mem[13,0] - mem[13,26] - mem[13,57] - mem[13,19] - mem[13,50]
   - mem[13,12] - mem[13,43] - mem[13,5] - mem[13,31] - mem[13,36]
   - mem[13,62] - mem[13,24] - mem[13,55] - mem[13,17] - mem[13,48]
   - mem[13,10] - mem[13,41] - mem[13,3] - mem[13,29] - mem[13,34]
   - mem[13,60] - mem[13,22] - mem[13,53] - mem[13,15] - mem[13,46]
   - mem[13,8] - mem[13,39] - mem[13,1] - mem[13,27] + mem_tot_l1_P42 = 0
 rows_tot_l1_P42: - 3 frac[13,32] - 3 frac[13,58] - 3 frac[13,20]
   - 3 frac[13,51] - 3 frac[13,13] - 3 frac[13,44] - 3 frac[13,6]
   - 3 frac[13,37] - 3 frac[13,63] - 3 frac[13,56] - 3 frac[13,18]
   - 3 frac[13,49] - 3 frac[13,11] - 3 frac[13,42] - 3 frac[13,4]
   - 3 frac[13,30] - 3 frac[13,35] - 3 frac[13,61] - 3 frac[13,23]
   - 3 frac[13,54] - 3 frac[13,16] - 3 frac[13,47] - 3 frac[13,9]
   - 3 frac[13,40] - 3 frac[13,2] - 3 frac[13,28] - 3 frac[13,33]
   - 3 frac[13,59] - 3 frac[13,21] - 3 frac[13,52] - 3 frac[13,14]
   - 3 frac[13,45] - 3 frac[13,7] - 3 frac[13,38] - 3 frac[13,0]
   - 3 frac[13,26] - 3 frac[13,57] - 3 frac[13,19] - 3 frac[13,50]
   - 3 frac[13,12] - 3 frac[13,43] - 3 frac[13,5] - 3 frac[13,31]
   - 3 frac[13,36] - 3 frac[13,62] - 3 frac[13,24] - 3 frac[13,55]
   - 3 frac[13,17] - 3 frac[13,48] - 3 frac[13,10] - 3 frac[13,41]
   - 3 frac[13,3] - 3 frac[13,29] - 3 frac[13,34] - 3 frac[13,60]
   - 3 frac[13,22] - 3 frac[13,53] - 3 frac[13,15] - 3 frac[13,46]
   - 3 frac[13,8] - 3 frac[13,39] - 3 frac[13,1] - 3 frac[13,27]
   + rows_tot_l1_P42 = 0
 mem_tot_l2_P4: - mem[14,31] - mem[14,1] - mem[14,43] - mem[14,8]
   - mem[14,50] - mem[14,19] - mem[14,53] - mem[14,26] - mem[14,60]
   - mem[14,38] - mem[14,29] - mem[14,7] - mem[14,41] - mem[14,14]
   - mem[14,48] - mem[14,17] - mem[14,59] - mem[14,24] - mem[14,2]
   - mem[14,36] - mem[14,5] - mem[14,47] - mem[14,12] - mem[14,54]
   - mem[14,23] - mem[14,57] - mem[14,35] - mem[14,30] - mem[14,0]
   - mem[14,42] - mem[14,11] - mem[14,45] - mem[14,18] - mem[14,52]
   - mem[14,21] - mem[14,63] - mem[14,33] - mem[14,28] - mem[14,6]
   - mem[14,40] - mem[14,9] - mem[14,51] - mem[14,16] - mem[14,58]
   - mem[14,27] - mem[14,61] - mem[14,39] - mem[14,4] - mem[14,46]
   - mem[14,15] - mem[14,49] - mem[14,22] - mem[14,34] - mem[14,3]
   - mem[14,37] - mem[14,10] - mem[14,44] - mem[14,13] - mem[14,55]
   - mem[14,20] - mem[14,62] - mem[14,32] + mem_tot_l2_P4 = 0
 rows_tot_l2_P4: - 3 frac[14,31] - 3 frac[14,1] - 3 frac[14,43]
   - 3 frac[14,8] - 3 frac[14,50] - 3 frac[14,19] - 3 frac[14,53]
   - 3 frac[14,26] - 3 frac[14,60] - 3 frac[14,38] - 3 frac[14,29]
   - 3 frac[14,7] - 3 frac[14,41] - 3 frac[14,14] - 3 frac[14,48]
   - 3 frac[14,17] - 3 frac[14,59] - 3 frac[14,24] - 3 frac[14,2]
   - 3 frac[14,36] - 3 frac[14,5] - 3 frac[14,47] - 3 frac[14,12]
   - 3 frac[14,54] - 3 frac[14,23] - 3 frac[14,57] - 3 frac[14,35]
   - 3 frac[14,30] - 3 frac[14,0] - 3 frac[14,42] - 3 frac[14,11]
   - 3 frac[14,45] - 3 frac[14,18] - 3 frac[14,52] - 3 frac[14,21]
   - 3 frac[14,63] - 3 frac[14,33] - 3 frac[14,28] - 3 frac[14,6]
   - 3 frac[14,40] - 3 frac[14,9] - 3 frac[14,51] - 3 frac[14,16]
   - 3 frac[14,58] - 3 frac[14,27] - 3 frac[14,61] - 3 frac[14,39]
   - 3 frac[14,4] - 3 frac[14,46] - 3 frac[14,15] - 3 frac[14,49]
   - 3 frac[14,22] - 3 frac[14,34] - 3 frac[14,3] - 3 frac[14,37]
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
 frac[7,3] frac[5,31] frac[6,28] frac[8,63] frac[5,57] frac[9,62]
 frac[6,54] frac[8,5] frac[7,55] frac[13,32] frac[11,48] frac[9,0]
 frac[14,31] frac[12,59] frac[10,7] frac[13,58] frac[0,17] frac[14,1]
 frac[12,17] frac[1,28] frac[13,20] frac[2,27] frac[0,55] frac[14,43]
 frac[3,2] frac[1,54] frac[4,5] frac[2,53] frac[0,45] frac[5,24] frac[3,40]
 frac[1,40] frac[6,23] frac[4,35] frac[8,38] frac[5,34] frac[9,39]
 frac[6,41] frac[10,36] frac[8,12] frac[7,44] frac[11,41] frac[9,9]
 frac[12,50] frac[10,14] frac[8,18] frac[13,51] frac[11,15] frac[9,19]
 frac[14,8] frac[12,8] frac[13,13] frac[2,18] frac[0,14] frac[14,50]
 frac[3,11] frac[1,15] frac[4,12] frac[2,12] frac[5,1] frac[3,17]
 frac[1,33] frac[6,14] frac[4,42] frac[8,45] frac[7,15] frac[3,55]
 frac[9,40] frac[6,32] frac[10,47] frac[8,51] frac[7,37] frac[5,53]
 frac[11,46] frac[9,50] frac[6,58] frac[12,41] frac[10,9] frac[8,25]
 frac[13,44] frac[11,4] frac[9,20] frac[14,19] frac[12,7] frac[10,19]
 frac[13,6] frac[11,26] frac[0,5] frac[14,53] frac[12,29] frac[1,0]
 frac[4,11] frac[2,7] frac[0,59] frac[5,10] frac[3,22] frac[1,58] frac[6,1]
 frac[2,33] frac[7,4] frac[5,20] frac[3,60] frac[6,27] frac[4,63]
 frac[8,58] frac[11,39] frac[9,59] frac[12,32] frac[10,48] frac[8,0]
 frac[7,56] frac[13,37] frac[11,61] frac[9,29] frac[14,26] frac[12,62]
 frac[10,26] frac[13,63] frac[11,19] frac[0,28] frac[14,60] frac[12,20]
 frac[1,25] frac[2,30] frac[14,38] frac[1,51] frac[4,24] frac[2,56]
 frac[0,40] frac[3,37] frac[6,18] frac[4,38] frac[8,33] frac[7,27]
 frac[5,39] frac[9,60] frac[6,52] frac[10,59] frac[8,7] frac[11,50]
 frac[9,6] frac[14,29] frac[12,53] frac[10,5] frac[13,56] frac[11,8]
 frac[14,7] frac[12,19] frac[1,18] frac[13,18] frac[2,25] frac[14,41]
 frac[3,4] frac[4,7] frac[2,51] frac[0,47] frac[5,6] frac[1,46] frac[8,40]
 frac[7,16] frac[5,32] frac[3,48] frac[9,37] frac[6,47] frac[10,34]
 frac[8,14] frac[11,43] frac[9,15] frac[12,44] frac[10,12] frac[8,20]
 frac[13,49] frac[11,1] frac[9,17] frac[14,14] frac[12,10] frac[10,22]
 frac[13,11] frac[2,16] frac[0,0] frac[14,48] frac[3,13] frac[2,10]
 frac[0,38] frac[5,15] frac[3,19] frac[1,39] frac[6,12] frac[4,20]
 frac[2,36] frac[8,47] frac[7,9] frac[5,41] frac[9,46] frac[6,38]
 frac[10,45] frac[8,53] frac[7,39] frac[11,32] frac[9,48] frac[6,56]
 frac[12,43] frac[10,55] frac[8,27] frac[7,61] frac[13,42] frac[11,6]
 frac[9,26] frac[14,17] frac[12,1] frac[10,17] frac[13,4] frac[11,28]
 frac[0,7] frac[14,59] frac[12,31] frac[13,30] frac[2,5] frac[3,24]
 frac[4,19] frac[2,63] frac[7,6] frac[5,18] frac[3,62] frac[4,57]
 frac[8,60] frac[7,28] frac[5,60] frac[6,51] frac[12,34] frac[10,62]
 frac[8,2] frac[7,58] frac[13,35] frac[11,63] frac[9,3] frac[14,24]
 frac[12,56] frac[10,24] frac[13,61] frac[0,30] frac[14,2] frac[12,22]
 frac[13,23] frac[2,28] frac[14,36] frac[3,1] frac[1,49] frac[4,26]
 frac[0,42] frac[5,27] frac[3,39] frac[6,16] frac[4,32] frac[8,35]
 frac[5,37] frac[9,34] frac[6,42] frac[10,57] frac[8,9] frac[11,52]
 frac[9,4] frac[12,55] frac[10,3] frac[13,54] frac[11,10] frac[0,21]
 frac[14,5] frac[12,13] frac[1,16] frac[13,16] frac[2,23] frac[14,47]
 frac[1,10] frac[4,1] frac[2,49] frac[5,4] frac[3,44] frac[1,44] frac[6,11]
 frac[4,47] frac[2,43] frac[8,42] frac[7,18] frac[5,46] frac[3,50]
 frac[9,43] frac[4,53] frac[10,32] frac[8,48] frac[11,45] frac[9,13]
 frac[12,46] frac[10,10] frac[8,22] frac[13,47] frac[11,3] frac[9,23]
 frac[14,12] frac[12,4] frac[10,20] frac[13,9] frac[0,2] frac[14,54]
 frac[3,15] frac[1,3] frac[2,8] frac[0,56] frac[5,13] frac[3,21] frac[1,37]
 frac[6,2] frac[4,22] frac[2,34] frac[7,11] frac[5,23] frac[3,59]
 frac[9,44] frac[6,36] frac[10,43] frac[8,55] frac[7,33] frac[11,34]
 frac[9,54] frac[12,37] frac[10,53] frac[8,29] frac[7,63] frac[13,40]
 frac[11,56] frac[9,24] frac[14,23] frac[12,3] frac[10,31] frac[13,2]
 frac[11,30] frac[14,57] frac[12,25] frac[1,4] frac[13,28] frac[2,3]
 frac[14,35] frac[3,26] frac[1,62] frac[6,5] frac[4,29] frac[2,61]
 frac[5,16] frac[3,32] frac[4,59] frac[8,62] frac[7,30] frac[5,58]
 frac[9,63] frac[6,49] frac[10,60] frac[8,4] frac[7,52] frac[13,33]
 frac[11,49] frac[9,1] frac[14,30] frac[12,58] frac[10,6] frac[13,59]
 frac[11,23] frac[0,16] frac[14,0] frac[12,16] frac[1,29] frac[13,21]
 frac[14,42] frac[3,3] frac[1,55] frac[4,4] frac[2,52] frac[0,44]
 frac[3,41] frac[2,46] frac[8,37] frac[7,23] frac[5,35] frac[9,32]
 frac[6,40] frac[10,39] frac[8,11] frac[7,45] frac[11,54] frac[9,10]
 frac[12,49] frac[10,1] frac[8,17] frac[13,52] frac[11,12] frac[0,23]
 frac[14,11] frac[12,15] frac[1,22] frac[13,14] frac[2,21] frac[14,45]
 frac[3,8] frac[1,8] frac[4,3] frac[2,15] frac[0,35] frac[5,2] frac[3,46]
 frac[1,34] frac[4,41] frac[2,41] frac[8,44] frac[5,44] frac[3,52]
 frac[9,41] frac[6,35] frac[4,55] frac[10,46] frac[8,50] frac[7,42]
 frac[5,54] frac[11,47] frac[9,51] frac[12,40] frac[8,24] frac[13,45]
 frac[11,5] frac[9,21] frac[14,18] frac[12,6] frac[10,18] frac[13,7]
 frac[11,27] frac[0,4] frac[14,52] frac[12,28] frac[1,1] frac[4,10]
 frac[2,6] frac[5,11] frac[3,23] frac[1,59] frac[6,0] frac[4,16] frac[2,32]
 frac[7,5] frac[5,21] frac[6,26] frac[4,62] frac[10,41] frac[8,57]
 frac[7,35] frac[11,36] frac[9,52] frac[12,39] frac[7,57] frac[13,38]
 frac[11,58] frac[9,30] frac[14,21] frac[12,61] frac[10,29] frac[13,0]
 frac[11,16] frac[0,27] frac[14,63] frac[12,27] frac[13,26] frac[2,1]
 frac[0,49] frac[14,33] frac[3,28] frac[1,60] frac[4,31] frac[7,2]
 frac[5,30] frac[3,34] frac[6,29] frac[4,37] frac[8,32] frac[7,24]
 frac[9,61] frac[6,55] frac[10,58] frac[8,6] frac[7,54] frac[11,51]
 frac[9,7] frac[14,28] frac[12,52] frac[10,4] frac[13,57] frac[11,9]
 frac[0,18] frac[14,6] frac[12,18] frac[1,19] frac[13,19] frac[2,24]
 frac[14,40] frac[3,5] frac[1,53] frac[4,6] frac[0,46] frac[5,7] frac[3,43]
 frac[1,47] frac[6,20] frac[2,44] frac[8,39] frac[7,17] frac[3,49]
 frac[9,38] frac[6,46] frac[10,37] frac[11,40] frac[9,8] frac[12,51]
 frac[10,15] frac[8,19] frac[13,50] frac[11,14] frac[9,18] frac[14,9]
 frac[12,9] frac[1,20] frac[13,12] frac[14,51] frac[3,10] frac[2,13]
 frac[0,37] frac[3,16] frac[1,32] frac[4,43] frac[2,39] frac[8,46]
 frac[5,42] frac[3,54] frac[9,47] frac[6,33] frac[10,44] frac[7,36]
 frac[5,52] frac[11,33] frac[9,49] frac[6,59] frac[12,42] frac[10,54]
 frac[8,26] frac[13,43] frac[11,7] frac[9,27] frac[14,16] frac[12,0]
 frac[10,16] frac[13,5] frac[11,29] frac[0,6] frac[14,58] frac[12,30]
 frac[13,31] frac[2,4] frac[0,60] frac[5,9] frac[3,25] frac[1,57] frac[6,6]
 frac[4,18] frac[2,62] frac[7,7] frac[5,19] frac[4,56] frac[8,59]
 frac[7,29] frac[11,38] frac[9,58] frac[12,33] frac[8,1] frac[7,59]
 frac[13,36] frac[11,60] frac[9,28] frac[14,27] frac[12,63] frac[10,27]
 frac[13,62] frac[11,18] frac[0,29] frac[14,61] frac[12,21] frac[1,24]
 frac[13,24] frac[2,31] frac[0,51] frac[14,39] frac[3,30] frac[1,50]
 frac[0,41] frac[5,28] frac[6,19] frac[4,39] frac[8,34] frac[7,26]
 frac[9,35] frac[6,53] frac[10,56] frac[8,8] frac[7,48] frac[11,53]
 frac[9,5] frac[12,54] frac[10,2] frac[13,55] frac[11,11] frac[14,4]
 frac[12,12] frac[1,17] frac[13,17] frac[2,22] frac[0,10] frac[14,46]
 frac[3,7] frac[1,11] frac[4,0] frac[2,48] frac[0,32] frac[5,5] frac[3,45]
 frac[6,10] frac[4,46] frac[2,42] frac[8,41] frac[7,19] frac[5,47]
 frac[3,51] frac[9,36] frac[10,35] frac[8,15] frac[7,41] frac[11,42]
 frac[12,45] frac[10,13] frac[8,21] frac[13,48] frac[11,0] frac[9,16]
 frac[14,15] frac[12,11] frac[10,23] frac[13,10] frac[2,17] frac[14,49]
 frac[4,15] frac[2,11] frac[5,14] frac[1,38] frac[2,37] frac[5,40]
 frac[3,56] frac[9,45] frac[10,42] frac[8,54] frac[7,38] frac[5,50]
 frac[11,35] frac[9,55] frac[6,57] frac[12,36] frac[10,52] frac[8,28]
 frac[7,60] frac[13,41] frac[11,57] frac[9,25] frac[14,22] frac[12,2]
 frac[10,30] frac[13,3] frac[12,24] frac[1,5] frac[13,29] frac[2,2]
 frac[0,62] frac[14,34] frac[1,63] frac[6,4] frac[4,28] frac[2,60]
 frac[7,1] frac[5,17] frac[6,30] frac[4,58] frac[5,59] frac[9,56]
 frac[6,48] frac[12,35] frac[10,63] frac[8,3] frac[7,53] frac[13,34]
 frac[9,2] frac[12,57] frac[13,60] frac[11,20] frac[14,3] frac[12,23]
 frac[1,30] frac[13,22] frac[2,29] frac[0,53] frac[14,37] frac[3,0]
 frac[1,48] frac[4,27] frac[2,55] frac[0,43] frac[5,26] frac[3,38]
 frac[1,42] frac[6,17] frac[4,33] frac[8,36] frac[5,36] frac[9,33]
 frac[6,43] frac[10,38] frac[8,10] frac[7,50] frac[11,55] frac[9,11]
 frac[12,48] frac[10,0] frac[8,16] frac[13,53] frac[11,13] frac[0,22]
 frac[14,10] frac[12,14] frac[13,15] frac[2,20] frac[0,12] frac[14,44]
 frac[1,9] frac[4,2] frac[5,3] frac[3,47] frac[1,35] frac[6,8] frac[4,40]
 frac[2,40] frac[8,43] frac[7,13] frac[5,45] frac[9,42] frac[6,34]
 frac[4,54] frac[10,33] frac[8,49] frac[5,55] frac[11,44] frac[9,12]
 frac[12,47] frac[10,11] frac[8,23] frac[13,46] frac[11,2] frac[9,22]
 frac[14,13] frac[12,5] frac[10,21] frac[13,8] frac[11,24] frac[0,3]
 frac[14,55] frac[2,9] frac[5,12] frac[3,20] frac[1,36] frac[6,3]
 frac[2,35] frac[7,10] frac[10,40] frac[8,56] frac[7,32] frac[5,48]
 frac[11,37] frac[9,53] frac[6,63] frac[12,38] frac[10,50] frac[8,30]
 frac[13,39] frac[11,59] frac[9,31] frac[14,20] frac[12,60] frac[10,28]
 frac[13,1] frac[11,17] frac[0,26] frac[14,62] frac[12,26] frac[1,27]
 frac[13,27] frac[2,0] frac[0,48] frac[14,32] frac[3,29] frac[2,58]
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

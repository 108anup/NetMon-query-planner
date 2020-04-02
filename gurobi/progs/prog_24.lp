\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_P4 + 0 max_mem_CPU_Cluster + 0 max_rows_P4
   + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU5 + 0 m_access_time_CPU9
   + 0 m_access_time_CPU16 + 0 m_access_time_CPU19 + 0 m_access_time_CPU20
   + 0 m_access_time_CPU22 + 0 m_access_time_CPU23 + 0 m_access_time_CPU30
   + res
Subject To
 cov_0_52: frac[9,52] + frac[8,52] >= 1
 cov_1_28: frac[9,28] + frac[10,28] >= 0.7
 cov_2_42: frac[10,42] + frac[9,42] >= 0.8
 cov_3_52: frac[9,52] + frac[10,52] + frac[8,52] >= 0.9
 cov_4_38: frac[9,38] + frac[10,38] >= 0.9
 cov_5_0: frac[9,0] + frac[10,0] >= 0.7
 cov_6_52: frac[2,52] >= 1
 cov_7_0: frac[9,0] + frac[10,0] >= 0.8
 cov_8_11: frac[9,11] + frac[8,11] >= 0.1
 cov_9_11: frac[9,11] + frac[10,11] >= 0.1
 cov_10_21: frac[9,21] + frac[10,21] >= 0.7
 cov_11_37: frac[10,37] + frac[9,37] >= 0.2
 cov_12_38: frac[8,38] + frac[4,38] >= 0.8
 cov_13_55: frac[9,55] + frac[10,55] >= 0.8
 cov_14_44: frac[8,44] + frac[9,44] + frac[10,44] >= 0.7
 cov_15_52: frac[9,52] + frac[10,52] >= 1
 cov_16_52: frac[9,52] + frac[10,52] >= 1
 cov_17_33: frac[5,33] + frac[8,33] >= 0.7
 cov_18_42: frac[10,42] + frac[8,42] + frac[9,42] >= 0.7
 cov_19_49: frac[1,49] + frac[3,49] + frac[9,49] + frac[8,49] + frac[10,49]
   >= 1
 cov_20_46: frac[4,46] + frac[8,46] + frac[3,46] >= 0.9
 cov_21_8: frac[5,8] + frac[3,8] + frac[8,8] >= 0.8
 cov_22_6: frac[1,6] + frac[8,6] + frac[9,6] + frac[6,6] + frac[10,6]
   >= 0.9
 cov_23_39: frac[9,39] + frac[0,39] + frac[3,39] + frac[8,39] + frac[10,39]
   >= 0.9533333333333334
 cov_24_6: frac[8,6] + frac[4,6] + frac[5,6] >= 1
 cov_25_50: frac[8,50] + frac[9,50] + frac[10,50] + frac[1,50] + frac[4,50]
   >= 0.8
 cov_26_14: frac[9,14] + frac[10,14] + frac[1,14] + frac[8,14] + frac[4,14]
   >= 0.9
 cov_27_15: frac[8,15] + frac[4,15] + frac[10,15] + frac[0,15] + frac[9,15]
   >= 0.8
 cov_28_25: frac[9,25] + frac[8,25] + frac[10,25] + frac[0,25] + frac[5,25]
   >= 0.7
 cov_29_40: frac[3,40] + frac[4,40] + frac[8,40] >= 0.8
 cov_30_19: frac[7,19] + frac[9,19] + frac[8,19] + frac[3,19] >= 0.8
 cov_31_19: frac[2,19] >= 0.9
 cov_32_15: frac[0,15] + frac[2,15] >= 0.9
 cov_33_30: frac[9,30] + frac[10,30] + frac[1,30] + frac[8,30] + frac[4,30]
   >= 0.7
 cov_34_53: frac[2,53] + frac[7,53] + frac[10,53] + frac[9,53] >= 0.8
 cov_35_36: frac[9,36] + frac[8,36] + frac[5,36] + frac[7,36] >= 0.7
 cov_36_49: frac[5,49] + frac[4,49] + frac[8,49] >= 0.8
 cov_37_35: frac[4,35] + frac[7,35] + frac[8,35] + frac[9,35] >= 0.7
 cov_38_5: frac[8,5] + frac[7,5] + frac[3,5] + frac[9,5] >= 0.7
 cov_39_35: frac[4,35] + frac[7,35] + frac[8,35] + frac[9,35] >= 1
 cov_40_24: frac[8,24] + frac[5,24] + frac[6,24] >= 1
 cov_41_46: frac[10,46] + frac[5,46] + frac[0,46] + frac[8,46] + frac[9,46]
   >= 0.9
 cov_42_14: frac[9,14] + frac[10,14] + frac[0,14] + frac[7,14] >= 0.9
 cov_43_36: frac[4,36] + frac[8,36] + frac[6,36] >= 0.7
 cov_44_12: frac[7,12] + frac[0,12] + frac[9,12] + frac[10,12] >= 0.8
 cov_45_9: frac[6,9] + frac[8,9] + frac[5,9] >= 0.7
 cov_46_54: frac[5,54] + frac[2,54] + frac[8,54] + frac[9,54] + frac[10,54]
   >= 0.9
 cov_47_12: frac[3,12] + frac[8,12] + frac[9,12] >= 0.8
 cov_48_19: frac[7,19] + frac[9,19] + frac[10,19] + frac[2,19] >= 0.9
 cov_49_30: frac[9,30] + frac[10,30] + frac[1,30] + frac[8,30] + frac[4,30]
   >= 0.9
 cov_50_46: frac[10,46] + frac[8,46] + frac[1,46] + frac[9,46] + frac[3,46]
   >= 0.8
 cov_51_23: frac[6,23] + frac[5,23] + frac[8,23] >= 1
 cov_52_8: frac[7,8] + frac[9,8] + frac[6,8] + frac[8,8] >= 0.9
 cov_53_13: frac[10,13] + frac[9,13] + frac[8,13] + frac[2,13] + frac[3,13]
   >= 0.9
 cov_54_32: frac[0,32] + frac[4,32] + frac[8,32] + frac[10,32] + frac[9,32]
   >= 0.9533333333333334
 cov_55_25: frac[7,25] + frac[9,25] + frac[8,25] + frac[5,25] >= 0.7
 cov_56_12: frac[2,12] + frac[0,12] >= 0.8
 cov_57_13: frac[10,13] + frac[9,13] + frac[8,13] + frac[5,13] + frac[1,13]
   >= 0.7
 cov_58_14: frac[9,14] + frac[6,14] + frac[7,14] + frac[8,14] >= 0.9
 cov_59_25: frac[7,25] + frac[6,25] + frac[9,25] + frac[8,25] >= 0.7
 cov_60_6: frac[2,6] + frac[8,6] + frac[4,6] + frac[9,6] + frac[10,6]
   >= 0.8
 cov_61_45: frac[3,45] + frac[8,45] + frac[4,45] >= 1
 cov_62_18: frac[2,18] + frac[1,18] >= 0.9
 cov_63_50: frac[8,50] + frac[9,50] + frac[3,50] + frac[2,50] + frac[10,50]
   >= 0.8
 cov_64_5: frac[8,5] + frac[4,5] + frac[10,5] + frac[9,5] >= 0.8
 cov_65_12: frac[8,12] + frac[1,12] + frac[4,12] + frac[9,12] + frac[10,12]
   >= 0.8
 cov_66_24: frac[8,24] + frac[5,24] + frac[10,24] + frac[2,24] + frac[9,24]
   >= 1
 cov_67_35: frac[1,35] + frac[0,35] >= 0.9
 cov_68_53: frac[4,53] + frac[8,53] + frac[6,53] >= 0.8
 cov_69_5: frac[8,5] + frac[5,5] + frac[4,5] >= 0.8
 cov_70_13: frac[10,13] + frac[9,13] + frac[8,13] + frac[3,13] + frac[0,13]
   >= 0.9
 cov_71_12: frac[3,12] + frac[8,12] + frac[0,12] + frac[9,12] + frac[10,12]
   >= 0.8
 cov_72_31: frac[8,31] + frac[7,31] + frac[3,31] + frac[9,31] >= 0.7
 cov_73_53: frac[2,53] + frac[7,53] + frac[10,53] + frac[9,53] >= 0.8
 cov_74_50: frac[8,50] + frac[9,50] + frac[3,50] + frac[7,50] >= 0.8
 cov_75_13: frac[10,13] + frac[6,13] + frac[9,13] + frac[8,13] + frac[0,13]
   >= 0.9533333333333334
 cov_76_35: frac[3,35] + frac[8,35] + frac[5,35] >= 1
 cov_77_19: frac[7,19] + frac[9,19] + frac[8,19] + frac[6,19] >= 0.9
 cov_78_15: frac[8,15] + frac[7,15] + frac[3,15] + frac[9,15] >= 0.8
 cov_79_31: frac[8,31] + frac[4,31] + frac[6,31] >= 0.7
 cov_80_19: frac[1,19] + frac[2,19] >= 0.9
 cov_81_46: frac[10,46] + frac[5,46] + frac[0,46] + frac[8,46] + frac[9,46]
   >= 0.8
 cov_82_3: frac[7,3] + frac[9,3] + frac[10,3] >= 0.7
 cov_83_47: frac[10,47] + frac[9,47] >= 0.9533333333333334
 cov_84_17: frac[10,17] + frac[9,17] >= 0.3
 cov_85_26: frac[9,26] + frac[10,26] >= 0.3
 cov_86_47: frac[10,47] + frac[9,47] + frac[8,47] >= 0.9533333333333334
 cov_87_51: frac[9,51] + frac[10,51] >= 0.2
 cov_88_7: frac[10,7] + frac[9,7] >= 0.3
 cov_89_48: frac[8,48] + frac[10,48] + frac[9,48] >= 0.2
 cov_90_16: frac[9,16] + frac[10,16] >= 0.8
 cov_91_26: frac[9,26] + frac[10,26] + frac[8,26] >= 0.3
 cov_92_26: frac[9,26] + frac[10,26] + frac[8,26] >= 0.3
 cov_93_48: frac[8,48] + frac[9,48] >= 0.2
 cov_94_51: frac[9,51] + frac[10,51] >= 0.2
 cov_95_7: frac[9,7] + frac[8,7] >= 0.3
 cov_96_48: frac[10,48] + frac[9,48] >= 0.2
 cov_97_17: frac[10,17] + frac[9,17] >= 0.3
 cov_98_27: frac[9,27] + frac[10,27] >= 0.8
 cov_99_47: frac[9,47] + frac[6,47] + frac[8,47] >= 0.9533333333333334
 cov_100_17: frac[10,17] + frac[9,17] >= 0.3
 cov_101_20: frac[9,20] + frac[10,20] >= 0.2
 cov_102_26: frac[9,26] + frac[10,26] >= 0.3
 cov_103_2: frac[9,2] + frac[10,2] >= 0.1
 cov_104_41: frac[9,41] + frac[8,41] + frac[10,41] >= 0.9533333333333334
 cov_105_2: frac[9,2] + frac[10,2] >= 0.1
 cov_106_10: frac[10,10] + frac[9,10] >= 0.7
 cov_107_22: frac[8,22] + frac[9,22] >= 0.7
 cov_108_43: frac[9,43] + frac[10,43] + frac[8,43] >= 0.7
 cov_109_2: frac[9,2] + frac[10,2] >= 0.1
 cov_110_2: frac[3,2] + frac[8,2] + frac[9,2] + frac[10,2] >= 0.1
 cov_111_34: frac[9,34] + frac[10,34] + frac[8,34] >= 0.1
 cov_112_29: frac[10,29] + frac[9,29] + frac[8,29] >= 1
 cov_113_4: frac[9,4] + frac[10,4] >= 1
 cov_114_34: frac[9,34] + frac[10,34] + frac[8,34] >= 0.1
 cov_115_1: frac[9,1] + frac[8,1] + frac[10,1] >= 0.2
 cov_116_29: frac[10,29] + frac[9,29] + frac[8,29] >= 1
 cov_117_1: frac[9,1] + frac[8,1] + frac[10,1] >= 0.2
 cov_118_4: frac[9,4] + frac[10,4] + frac[8,4] >= 1
 cov_119_4: frac[9,4] + frac[8,4] >= 0.7
 cov_120_4: frac[9,4] + frac[10,4] + frac[8,4] >= 0.9
 accuracy_9_26: - 5.09765625 frac[9,26] + mem[9,26] = 0
 accuracy_7_3: - 5.09765625 frac[7,3] + mem[7,3] = 0
 accuracy_6_9: - 5.09765625 frac[6,9] + mem[6,9] = 0
 accuracy_3_35: - 5.09765625 frac[3,35] + mem[3,35] = 0
 accuracy_10_17: - 5.09765625 frac[10,17] + mem[10,17] = 0
 accuracy_8_44: - 5.09765625 frac[8,44] + mem[8,44] = 0
 accuracy_7_12: - 5.09765625 frac[7,12] + mem[7,12] = 0
 accuracy_4_36: - 5.09765625 frac[4,36] + mem[4,36] = 0
 accuracy_9_41: - 5.09765625 frac[9,41] + mem[9,41] = 0
 accuracy_7_25: - 5.09765625 frac[7,25] + mem[7,25] = 0
 accuracy_10_46: - 5.09765625 frac[10,46] + mem[10,46] = 0
 accuracy_1_6: - 5.09765625 frac[1,6] + mem[1,6] = 0
 accuracy_8_50: - 5.09765625 frac[8,50] + mem[8,50] = 0
 accuracy_5_54: - 5.09765625 frac[5,54] + mem[5,54] = 0
 accuracy_9_51: - 5.09765625 frac[9,51] + mem[9,51] = 0
 accuracy_8_5: - 5.09765625 frac[8,5] + mem[8,5] = 0
 accuracy_5_8: - 5.09765625 frac[5,8] + mem[5,8] = 0
 accuracy_9_0: - 5.09765625 frac[9,0] + mem[9,0] = 0
 accuracy_8_24: - 5.09765625 frac[8,24] + mem[8,24] = 0
 accuracy_0_32: - 5.09765625 frac[0,32] + mem[0,32] = 0
 accuracy_5_5: - 5.09765625 frac[5,5] + mem[5,5] = 0
 accuracy_3_45: - 5.09765625 frac[3,45] + mem[3,45] = 0
 accuracy_10_7: - 5.09765625 frac[10,7] + mem[10,7] = 0
 accuracy_9_21: - 5.09765625 frac[9,21] + mem[9,21] = 0
 accuracy_4_46: - 5.09765625 frac[4,46] + mem[4,46] = 0
 accuracy_8_41: - 5.09765625 frac[8,41] + mem[8,41] = 0
 accuracy_7_19: - 5.09765625 frac[7,19] + mem[7,19] = 0
 accuracy_6_25: - 5.09765625 frac[6,25] + mem[6,25] = 0
 accuracy_9_36: - 5.09765625 frac[9,36] + mem[9,36] = 0
 accuracy_8_15: - 5.09765625 frac[8,15] + mem[8,15] = 0
 accuracy_3_2: - 5.09765625 frac[3,2] + mem[3,2] = 0
 accuracy_2_6: - 5.09765625 frac[2,6] + mem[2,6] = 0
 accuracy_9_14: - 5.09765625 frac[9,14] + mem[9,14] = 0
 accuracy_8_2: - 5.09765625 frac[8,2] + mem[8,2] = 0
 accuracy_4_5: - 5.09765625 frac[4,5] + mem[4,5] = 0
 accuracy_2_53: - 5.09765625 frac[2,53] + mem[2,53] = 0
 accuracy_10_13: - 5.09765625 frac[10,13] + mem[10,13] = 0
 accuracy_9_3: - 5.09765625 frac[9,3] + mem[9,3] = 0
 accuracy_5_24: - 5.09765625 frac[5,24] + mem[5,24] = 0
 accuracy_3_40: - 5.09765625 frac[3,40] + mem[3,40] = 0
 accuracy_10_24: - 5.09765625 frac[10,24] + mem[10,24] = 0
 accuracy_9_16: - 5.09765625 frac[9,16] + mem[9,16] = 0
 accuracy_7_5: - 5.09765625 frac[7,5] + mem[7,5] = 0
 accuracy_6_23: - 5.09765625 frac[6,23] + mem[6,23] = 0
 accuracy_4_35: - 5.09765625 frac[4,35] + mem[4,35] = 0
 accuracy_8_38: - 5.09765625 frac[8,38] + mem[8,38] = 0
 accuracy_10_41: - 5.09765625 frac[10,41] + mem[10,41] = 0
 accuracy_9_39: - 5.09765625 frac[9,39] + mem[9,39] = 0
 accuracy_7_35: - 5.09765625 frac[7,35] + mem[7,35] = 0
 accuracy_3_12: - 5.09765625 frac[3,12] + mem[3,12] = 0
 accuracy_9_52: - 5.09765625 frac[9,52] + mem[9,52] = 0
 accuracy_8_12: - 5.09765625 frac[8,12] + mem[8,12] = 0
 accuracy_1_12: - 5.09765625 frac[1,12] + mem[1,12] = 0
 accuracy_4_15: - 5.09765625 frac[4,15] + mem[4,15] = 0
 accuracy_10_51: - 5.09765625 frac[10,51] + mem[10,51] = 0
 accuracy_1_49: - 5.09765625 frac[1,49] + mem[1,49] = 0
 accuracy_0_39: - 5.09765625 frac[0,39] + mem[0,39] = 0
 accuracy_8_31: - 5.09765625 frac[8,31] + mem[8,31] = 0
 accuracy_10_14: - 5.09765625 frac[10,14] + mem[10,14] = 0
 accuracy_9_30: - 5.09765625 frac[9,30] + mem[9,30] = 0
 accuracy_2_54: - 5.09765625 frac[2,54] + mem[2,54] = 0
 accuracy_6_13: - 5.09765625 frac[6,13] + mem[6,13] = 0
 accuracy_3_39: - 5.09765625 frac[3,39] + mem[3,39] = 0
 accuracy_10_29: - 5.09765625 frac[10,29] + mem[10,29] = 0
 accuracy_9_19: - 5.09765625 frac[9,19] + mem[9,19] = 0
 accuracy_7_8: - 5.09765625 frac[7,8] + mem[7,8] = 0
 accuracy_4_32: - 5.09765625 frac[4,32] + mem[4,32] = 0
 accuracy_8_35: - 5.09765625 frac[8,35] + mem[8,35] = 0
 accuracy_10_42: - 5.09765625 frac[10,42] + mem[10,42] = 0
 accuracy_2_18: - 5.09765625 frac[2,18] + mem[2,18] = 0
 accuracy_0_14: - 5.09765625 frac[0,14] + mem[0,14] = 0
 accuracy_8_54: - 5.09765625 frac[8,54] + mem[8,54] = 0
 accuracy_9_34: - 5.09765625 frac[9,34] + mem[9,34] = 0
 accuracy_9_55: - 5.09765625 frac[9,55] + mem[9,55] = 0
 accuracy_8_9: - 5.09765625 frac[8,9] + mem[8,9] = 0
 accuracy_4_12: - 5.09765625 frac[4,12] + mem[4,12] = 0
 accuracy_10_52: - 5.09765625 frac[10,52] + mem[10,52] = 0
 accuracy_2_12: - 5.09765625 frac[2,12] + mem[2,12] = 0
 accuracy_9_4: - 5.09765625 frac[9,4] + mem[9,4] = 0
 accuracy_4_31: - 5.09765625 frac[4,31] + mem[4,31] = 0
 accuracy_10_3: - 5.09765625 frac[10,3] + mem[10,3] = 0
 accuracy_9_25: - 5.09765625 frac[9,25] + mem[9,25] = 0
 accuracy_6_14: - 5.09765625 frac[6,14] + mem[6,14] = 0
 accuracy_10_30: - 5.09765625 frac[10,30] + mem[10,30] = 0
 accuracy_8_45: - 5.09765625 frac[8,45] + mem[8,45] = 0
 accuracy_7_15: - 5.09765625 frac[7,15] + mem[7,15] = 0
 accuracy_8_32: - 5.09765625 frac[8,32] + mem[8,32] = 0
 accuracy_10_47: - 5.09765625 frac[10,47] + mem[10,47] = 0
 accuracy_9_50: - 5.09765625 frac[9,50] + mem[9,50] = 0
 accuracy_8_6: - 5.09765625 frac[8,6] + mem[8,6] = 0
 accuracy_9_7: - 5.09765625 frac[9,7] + mem[9,7] = 0
 accuracy_8_25: - 5.09765625 frac[8,25] + mem[8,25] = 0
 accuracy_10_4: - 5.09765625 frac[10,4] + mem[10,4] = 0
 accuracy_9_20: - 5.09765625 frac[9,20] + mem[9,20] = 0
 accuracy_10_19: - 5.09765625 frac[10,19] + mem[10,19] = 0
 accuracy_8_42: - 5.09765625 frac[8,42] + mem[8,42] = 0
 accuracy_5_46: - 5.09765625 frac[5,46] + mem[5,46] = 0
 accuracy_3_50: - 5.09765625 frac[3,50] + mem[3,50] = 0
 accuracy_1_19: - 5.09765625 frac[1,19] + mem[1,19] = 0
 accuracy_9_43: - 5.09765625 frac[9,43] + mem[9,43] = 0
 accuracy_7_31: - 5.09765625 frac[7,31] + mem[7,31] = 0
 accuracy_4_53: - 5.09765625 frac[4,53] + mem[4,53] = 0
 accuracy_10_32: - 5.09765625 frac[10,32] + mem[10,32] = 0
 accuracy_2_24: - 5.09765625 frac[2,24] + mem[2,24] = 0
 accuracy_8_48: - 5.09765625 frac[8,48] + mem[8,48] = 0
 accuracy_3_5: - 5.09765625 frac[3,5] + mem[3,5] = 0
 accuracy_9_13: - 5.09765625 frac[9,13] + mem[9,13] = 0
 accuracy_7_53: - 5.09765625 frac[7,53] + mem[7,53] = 0
 accuracy_4_6: - 5.09765625 frac[4,6] + mem[4,6] = 0
 accuracy_2_50: - 5.09765625 frac[2,50] + mem[2,50] = 0
 accuracy_9_2: - 5.09765625 frac[9,2] + mem[9,2] = 0
 accuracy_0_46: - 5.09765625 frac[0,46] + mem[0,46] = 0
 accuracy_10_10: - 5.09765625 frac[10,10] + mem[10,10] = 0
 accuracy_8_22: - 5.09765625 frac[8,22] + mem[8,22] = 0
 accuracy_10_25: - 5.09765625 frac[10,25] + mem[10,25] = 0
 accuracy_10_20: - 5.09765625 frac[10,20] + mem[10,20] = 0
 accuracy_8_39: - 5.09765625 frac[8,39] + mem[8,39] = 0
 accuracy_5_33: - 5.09765625 frac[5,33] + mem[5,33] = 0
 accuracy_3_49: - 5.09765625 frac[3,49] + mem[3,49] = 0
 accuracy_9_38: - 5.09765625 frac[9,38] + mem[9,38] = 0
 accuracy_1_30: - 5.09765625 frac[1,30] + mem[1,30] = 0
 accuracy_3_15: - 5.09765625 frac[3,15] + mem[3,15] = 0
 accuracy_10_37: - 5.09765625 frac[10,37] + mem[10,37] = 0
 accuracy_8_13: - 5.09765625 frac[8,13] + mem[8,13] = 0
 accuracy_10_48: - 5.09765625 frac[10,48] + mem[10,48] = 0
 accuracy_9_8: - 5.09765625 frac[9,8] + mem[9,8] = 0
 accuracy_5_13: - 5.09765625 frac[5,13] + mem[5,13] = 0
 accuracy_10_15: - 5.09765625 frac[10,15] + mem[10,15] = 0
 accuracy_9_29: - 5.09765625 frac[9,29] + mem[9,29] = 0
 accuracy_8_19: - 5.09765625 frac[8,19] + mem[8,19] = 0
 accuracy_10_26: - 5.09765625 frac[10,26] + mem[10,26] = 0
 accuracy_5_23: - 5.09765625 frac[5,23] + mem[5,23] = 0
 accuracy_9_44: - 5.09765625 frac[9,44] + mem[9,44] = 0
 accuracy_8_36: - 5.09765625 frac[8,36] + mem[8,36] = 0
 accuracy_6_36: - 5.09765625 frac[6,36] + mem[6,36] = 0
 accuracy_5_36: - 5.09765625 frac[5,36] + mem[5,36] = 0
 accuracy_2_19: - 5.09765625 frac[2,19] + mem[2,19] = 0
 accuracy_10_43: - 5.09765625 frac[10,43] + mem[10,43] = 0
 accuracy_0_15: - 5.09765625 frac[0,15] + mem[0,15] = 0
 accuracy_5_49: - 5.09765625 frac[5,49] + mem[5,49] = 0
 accuracy_10_38: - 5.09765625 frac[10,38] + mem[10,38] = 0
 accuracy_1_14: - 5.09765625 frac[1,14] + mem[1,14] = 0
 accuracy_9_54: - 5.09765625 frac[9,54] + mem[9,54] = 0
 accuracy_7_50: - 5.09765625 frac[7,50] + mem[7,50] = 0
 accuracy_3_31: - 5.09765625 frac[3,31] + mem[3,31] = 0
 accuracy_10_53: - 5.09765625 frac[10,53] + mem[10,53] = 0
 accuracy_9_11: - 5.09765625 frac[9,11] + mem[9,11] = 0
 accuracy_2_13: - 5.09765625 frac[2,13] + mem[2,13] = 0
 accuracy_8_29: - 5.09765625 frac[8,29] + mem[8,29] = 0
 accuracy_2_15: - 5.09765625 frac[2,15] + mem[2,15] = 0
 accuracy_10_0: - 5.09765625 frac[10,0] + mem[10,0] = 0
 accuracy_9_24: - 5.09765625 frac[9,24] + mem[9,24] = 0
 accuracy_8_46: - 5.09765625 frac[8,46] + mem[8,46] = 0
 accuracy_7_14: - 5.09765625 frac[7,14] + mem[7,14] = 0
 accuracy_4_38: - 5.09765625 frac[4,38] + mem[4,38] = 0
 accuracy_9_47: - 5.09765625 frac[9,47] + mem[9,47] = 0
 accuracy_8_33: - 5.09765625 frac[8,33] + mem[8,33] = 0
 accuracy_0_25: - 5.09765625 frac[0,25] + mem[0,25] = 0
 accuracy_4_49: - 5.09765625 frac[4,49] + mem[4,49] = 0
 accuracy_10_44: - 5.09765625 frac[10,44] + mem[10,44] = 0
 accuracy_8_52: - 5.09765625 frac[8,52] + mem[8,52] = 0
 accuracy_7_36: - 5.09765625 frac[7,36] + mem[7,36] = 0
 accuracy_0_12: - 5.09765625 frac[0,12] + mem[0,12] = 0
 accuracy_9_49: - 5.09765625 frac[9,49] + mem[9,49] = 0
 accuracy_8_7: - 5.09765625 frac[8,7] + mem[8,7] = 0
 accuracy_10_54: - 5.09765625 frac[10,54] + mem[10,54] = 0
 accuracy_9_6: - 5.09765625 frac[9,6] + mem[9,6] = 0
 accuracy_8_26: - 5.09765625 frac[8,26] + mem[8,26] = 0
 accuracy_10_5: - 5.09765625 frac[10,5] + mem[10,5] = 0
 accuracy_1_35: - 5.09765625 frac[1,35] + mem[1,35] = 0
 accuracy_9_27: - 5.09765625 frac[9,27] + mem[9,27] = 0
 accuracy_6_8: - 5.09765625 frac[6,8] + mem[6,8] = 0
 accuracy_4_40: - 5.09765625 frac[4,40] + mem[4,40] = 0
 accuracy_10_16: - 5.09765625 frac[10,16] + mem[10,16] = 0
 accuracy_8_43: - 5.09765625 frac[8,43] + mem[8,43] = 0
 accuracy_6_31: - 5.09765625 frac[6,31] + mem[6,31] = 0
 accuracy_9_42: - 5.09765625 frac[9,42] + mem[9,42] = 0
 accuracy_1_18: - 5.09765625 frac[1,18] + mem[1,18] = 0
 accuracy_8_49: - 5.09765625 frac[8,49] + mem[8,49] = 0
 accuracy_9_12: - 5.09765625 frac[9,12] + mem[9,12] = 0
 accuracy_8_4: - 5.09765625 frac[8,4] + mem[8,4] = 0
 accuracy_5_9: - 5.09765625 frac[5,9] + mem[5,9] = 0
 accuracy_10_11: - 5.09765625 frac[10,11] + mem[10,11] = 0
 accuracy_9_1: - 5.09765625 frac[9,1] + mem[9,1] = 0
 accuracy_8_23: - 5.09765625 frac[8,23] + mem[8,23] = 0
 accuracy_6_6: - 5.09765625 frac[6,6] + mem[6,6] = 0
 accuracy_5_6: - 5.09765625 frac[5,6] + mem[5,6] = 0
 accuracy_10_6: - 5.09765625 frac[10,6] + mem[10,6] = 0
 accuracy_1_46: - 5.09765625 frac[1,46] + mem[1,46] = 0
 accuracy_9_22: - 5.09765625 frac[9,22] + mem[9,22] = 0
 accuracy_4_45: - 5.09765625 frac[4,45] + mem[4,45] = 0
 accuracy_10_21: - 5.09765625 frac[10,21] + mem[10,21] = 0
 accuracy_8_40: - 5.09765625 frac[8,40] + mem[8,40] = 0
 accuracy_6_24: - 5.09765625 frac[6,24] + mem[6,24] = 0
 accuracy_9_37: - 5.09765625 frac[9,37] + mem[9,37] = 0
 accuracy_6_47: - 5.09765625 frac[6,47] + mem[6,47] = 0
 accuracy_10_34: - 5.09765625 frac[10,34] + mem[10,34] = 0
 accuracy_8_14: - 5.09765625 frac[8,14] + mem[8,14] = 0
 accuracy_10_49: - 5.09765625 frac[10,49] + mem[10,49] = 0
 accuracy_9_15: - 5.09765625 frac[9,15] + mem[9,15] = 0
 accuracy_8_1: - 5.09765625 frac[8,1] + mem[8,1] = 0
 accuracy_2_52: - 5.09765625 frac[2,52] + mem[2,52] = 0
 accuracy_9_28: - 5.09765625 frac[9,28] + mem[9,28] = 0
 accuracy_10_12: - 5.09765625 frac[10,12] + mem[10,12] = 0
 accuracy_5_25: - 5.09765625 frac[5,25] + mem[5,25] = 0
 accuracy_10_27: - 5.09765625 frac[10,27] + mem[10,27] = 0
 accuracy_9_17: - 5.09765625 frac[9,17] + mem[9,17] = 0
 accuracy_5_35: - 5.09765625 frac[5,35] + mem[5,35] = 0
 accuracy_9_32: - 5.09765625 frac[9,32] + mem[9,32] = 0
 accuracy_3_13: - 5.09765625 frac[3,13] + mem[3,13] = 0
 accuracy_10_39: - 5.09765625 frac[10,39] + mem[10,39] = 0
 accuracy_9_53: - 5.09765625 frac[9,53] + mem[9,53] = 0
 accuracy_8_11: - 5.09765625 frac[8,11] + mem[8,11] = 0
 accuracy_1_13: - 5.09765625 frac[1,13] + mem[1,13] = 0
 accuracy_4_14: - 5.09765625 frac[4,14] + mem[4,14] = 0
 accuracy_10_50: - 5.09765625 frac[10,50] + mem[10,50] = 0
 accuracy_1_50: - 5.09765625 frac[1,50] + mem[1,50] = 0
 accuracy_8_30: - 5.09765625 frac[8,30] + mem[8,30] = 0
 accuracy_9_10: - 5.09765625 frac[9,10] + mem[9,10] = 0
 accuracy_3_19: - 5.09765625 frac[3,19] + mem[3,19] = 0
 accuracy_10_1: - 5.09765625 frac[10,1] + mem[10,1] = 0
 accuracy_9_31: - 5.09765625 frac[9,31] + mem[9,31] = 0
 accuracy_10_28: - 5.09765625 frac[10,28] + mem[10,28] = 0
 accuracy_8_47: - 5.09765625 frac[8,47] + mem[8,47] = 0
 accuracy_6_19: - 5.09765625 frac[6,19] + mem[6,19] = 0
 accuracy_9_46: - 5.09765625 frac[9,46] + mem[9,46] = 0
 accuracy_8_34: - 5.09765625 frac[8,34] + mem[8,34] = 0
 accuracy_4_50: - 5.09765625 frac[4,50] + mem[4,50] = 0
 accuracy_9_35: - 5.09765625 frac[9,35] + mem[9,35] = 0
 accuracy_8_53: - 5.09765625 frac[8,53] + mem[8,53] = 0
 accuracy_0_13: - 5.09765625 frac[0,13] + mem[0,13] = 0
 accuracy_6_53: - 5.09765625 frac[6,53] + mem[6,53] = 0
 accuracy_3_8: - 5.09765625 frac[3,8] + mem[3,8] = 0
 accuracy_9_48: - 5.09765625 frac[9,48] + mem[9,48] = 0
 accuracy_8_8: - 5.09765625 frac[8,8] + mem[8,8] = 0
 accuracy_10_55: - 5.09765625 frac[10,55] + mem[10,55] = 0
 accuracy_9_5: - 5.09765625 frac[9,5] + mem[9,5] = 0
 accuracy_0_35: - 5.09765625 frac[0,35] + mem[0,35] = 0
 accuracy_4_30: - 5.09765625 frac[4,30] + mem[4,30] = 0
 accuracy_3_46: - 5.09765625 frac[3,46] + mem[3,46] = 0
 accuracy_10_2: - 5.09765625 frac[10,2] + mem[10,2] = 0
 mem_tot_CPU5: - mem[0,32] - mem[0,39] - mem[0,14] - mem[0,46] - mem[0,15]
   - mem[0,25] - mem[0,12] - mem[0,13] - mem[0,35] + mem_tot_CPU5 = 0
 R363: - 3.05176e-05 mem_tot_CPU5 + normalized_mem_tot_CPU5 = 0
 rows_tot_CPU5: - 3 frac[0,32] - 3 frac[0,39] - 3 frac[0,14] - 3 frac[0,46]
   - 3 frac[0,15] - 3 frac[0,25] - 3 frac[0,12] - 3 frac[0,13]
   - 3 frac[0,35] + rows_tot_CPU5 = 0
 R365: - 0.0833333333333333 rows_tot_CPU5 + normalized_rows_tot_CPU5 = 0
 mem_tot_CPU9: - mem[1,6] - mem[1,12] - mem[1,49] - mem[1,19] - mem[1,30]
   - mem[1,14] - mem[1,35] - mem[1,18] - mem[1,46] - mem[1,13] - mem[1,50]
   + mem_tot_CPU9 = 0
 R367: - 3.05176e-05 mem_tot_CPU9 + normalized_mem_tot_CPU9 = 0
 rows_tot_CPU9: - 3 frac[1,6] - 3 frac[1,12] - 3 frac[1,49] - 3 frac[1,19]
   - 3 frac[1,30] - 3 frac[1,14] - 3 frac[1,35] - 3 frac[1,18]
   - 3 frac[1,46] - 3 frac[1,13] - 3 frac[1,50] + rows_tot_CPU9 = 0
 R369: - 0.0833333333333333 rows_tot_CPU9 + normalized_rows_tot_CPU9 = 0
 mem_tot_CPU16: - mem[2,6] - mem[2,53] - mem[2,54] - mem[2,18] - mem[2,12]
   - mem[2,24] - mem[2,50] - mem[2,19] - mem[2,13] - mem[2,15] - mem[2,52]
   + mem_tot_CPU16 = 0
 R371: - 3.05176e-05 mem_tot_CPU16 + normalized_mem_tot_CPU16 = 0
 rows_tot_CPU16: - 3 frac[2,6] - 3 frac[2,53] - 3 frac[2,54] - 3 frac[2,18]
   - 3 frac[2,12] - 3 frac[2,24] - 3 frac[2,50] - 3 frac[2,19]
   - 3 frac[2,13] - 3 frac[2,15] - 3 frac[2,52] + rows_tot_CPU16 = 0
 R373: - 0.0833333333333333 rows_tot_CPU16 + normalized_rows_tot_CPU16 = 0
 mem_tot_CPU19: - mem[3,35] - mem[3,45] - mem[3,2] - mem[3,40] - mem[3,12]
   - mem[3,39] - mem[3,50] - mem[3,5] - mem[3,49] - mem[3,15] - mem[3,31]
   - mem[3,13] - mem[3,19] - mem[3,8] - mem[3,46] + mem_tot_CPU19 = 0
 R375: - 3.05176e-05 mem_tot_CPU19 + normalized_mem_tot_CPU19 = 0
 rows_tot_CPU19: - 3 frac[3,35] - 3 frac[3,45] - 3 frac[3,2] - 3 frac[3,40]
   - 3 frac[3,12] - 3 frac[3,39] - 3 frac[3,50] - 3 frac[3,5]
   - 3 frac[3,49] - 3 frac[3,15] - 3 frac[3,31] - 3 frac[3,13]
   - 3 frac[3,19] - 3 frac[3,8] - 3 frac[3,46] + rows_tot_CPU19 = 0
 R377: - 0.0833333333333333 rows_tot_CPU19 + normalized_rows_tot_CPU19 = 0
 mem_tot_CPU20: - mem[4,36] - mem[4,46] - mem[4,5] - mem[4,35] - mem[4,15]
   - mem[4,32] - mem[4,12] - mem[4,31] - mem[4,53] - mem[4,6] - mem[4,38]
   - mem[4,49] - mem[4,40] - mem[4,45] - mem[4,14] - mem[4,50] - mem[4,30]
   + mem_tot_CPU20 = 0
 R379: - 3.05176e-05 mem_tot_CPU20 + normalized_mem_tot_CPU20 = 0
 rows_tot_CPU20: - 3 frac[4,36] - 3 frac[4,46] - 3 frac[4,5] - 3 frac[4,35]
   - 3 frac[4,15] - 3 frac[4,32] - 3 frac[4,12] - 3 frac[4,31]
   - 3 frac[4,53] - 3 frac[4,6] - 3 frac[4,38] - 3 frac[4,49]
   - 3 frac[4,40] - 3 frac[4,45] - 3 frac[4,14] - 3 frac[4,50]
   - 3 frac[4,30] + rows_tot_CPU20 = 0
 R381: - 0.0833333333333333 rows_tot_CPU20 + normalized_rows_tot_CPU20 = 0
 mem_tot_CPU22: - mem[5,54] - mem[5,8] - mem[5,5] - mem[5,24] - mem[5,46]
   - mem[5,33] - mem[5,13] - mem[5,23] - mem[5,36] - mem[5,49] - mem[5,9]
   - mem[5,6] - mem[5,25] - mem[5,35] + mem_tot_CPU22 = 0
 R383: - 3.05176e-05 mem_tot_CPU22 + normalized_mem_tot_CPU22 = 0
 rows_tot_CPU22: - 3 frac[5,54] - 3 frac[5,8] - 3 frac[5,5] - 3 frac[5,24]
   - 3 frac[5,46] - 3 frac[5,33] - 3 frac[5,13] - 3 frac[5,23]
   - 3 frac[5,36] - 3 frac[5,49] - 3 frac[5,9] - 3 frac[5,6] - 3 frac[5,25]
   - 3 frac[5,35] + rows_tot_CPU22 = 0
 R385: - 0.0833333333333333 rows_tot_CPU22 + normalized_rows_tot_CPU22 = 0
 mem_tot_CPU23: - mem[6,9] - mem[6,25] - mem[6,23] - mem[6,13] - mem[6,14]
   - mem[6,36] - mem[6,8] - mem[6,31] - mem[6,6] - mem[6,24] - mem[6,47]
   - mem[6,19] - mem[6,53] + mem_tot_CPU23 = 0
 R387: - 3.05176e-05 mem_tot_CPU23 + normalized_mem_tot_CPU23 = 0
 rows_tot_CPU23: - 3 frac[6,9] - 3 frac[6,25] - 3 frac[6,23] - 3 frac[6,13]
   - 3 frac[6,14] - 3 frac[6,36] - 3 frac[6,8] - 3 frac[6,31] - 3 frac[6,6]
   - 3 frac[6,24] - 3 frac[6,47] - 3 frac[6,19] - 3 frac[6,53]
   + rows_tot_CPU23 = 0
 R389: - 0.0833333333333333 rows_tot_CPU23 + normalized_rows_tot_CPU23 = 0
 mem_tot_CPU30: - mem[7,3] - mem[7,12] - mem[7,25] - mem[7,19] - mem[7,5]
   - mem[7,35] - mem[7,8] - mem[7,15] - mem[7,31] - mem[7,53] - mem[7,50]
   - mem[7,14] - mem[7,36] + mem_tot_CPU30 = 0
 R391: - 3.05176e-05 mem_tot_CPU30 + normalized_mem_tot_CPU30 = 0
 rows_tot_CPU30: - 3 frac[7,3] - 3 frac[7,12] - 3 frac[7,25] - 3 frac[7,19]
   - 3 frac[7,5] - 3 frac[7,35] - 3 frac[7,8] - 3 frac[7,15] - 3 frac[7,31]
   - 3 frac[7,53] - 3 frac[7,50] - 3 frac[7,14] - 3 frac[7,36]
   + rows_tot_CPU30 = 0
 R393: - 0.0833333333333333 rows_tot_CPU30 + normalized_rows_tot_CPU30 = 0
 mem_tot_tor_P43: - mem[8,44] - mem[8,50] - mem[8,5] - mem[8,24]
   - mem[8,41] - mem[8,15] - mem[8,2] - mem[8,38] - mem[8,12] - mem[8,31]
   - mem[8,35] - mem[8,54] - mem[8,9] - mem[8,45] - mem[8,32] - mem[8,6]
   - mem[8,25] - mem[8,42] - mem[8,48] - mem[8,22] - mem[8,39] - mem[8,13]
   - mem[8,19] - mem[8,36] - mem[8,29] - mem[8,46] - mem[8,33] - mem[8,52]
   - mem[8,7] - mem[8,26] - mem[8,43] - mem[8,49] - mem[8,4] - mem[8,23]
   - mem[8,40] - mem[8,14] - mem[8,1] - mem[8,11] - mem[8,30] - mem[8,47]
   - mem[8,34] - mem[8,53] - mem[8,8] + mem_tot_tor_P43 = 0
 R395: - 0.00173611111111111 mem_tot_tor_P43 + normalized_mem_tot_tor_P43
   = 0
 rows_tot_tor_P43: - 3 frac[8,44] - 3 frac[8,50] - 3 frac[8,5]
   - 3 frac[8,24] - 3 frac[8,41] - 3 frac[8,15] - 3 frac[8,2]
   - 3 frac[8,38] - 3 frac[8,12] - 3 frac[8,31] - 3 frac[8,35]
   - 3 frac[8,54] - 3 frac[8,9] - 3 frac[8,45] - 3 frac[8,32] - 3 frac[8,6]
   - 3 frac[8,25] - 3 frac[8,42] - 3 frac[8,48] - 3 frac[8,22]
   - 3 frac[8,39] - 3 frac[8,13] - 3 frac[8,19] - 3 frac[8,36]
   - 3 frac[8,29] - 3 frac[8,46] - 3 frac[8,33] - 3 frac[8,52]
   - 3 frac[8,7] - 3 frac[8,26] - 3 frac[8,43] - 3 frac[8,49] - 3 frac[8,4]
   - 3 frac[8,23] - 3 frac[8,40] - 3 frac[8,14] - 3 frac[8,1]
   - 3 frac[8,11] - 3 frac[8,30] - 3 frac[8,47] - 3 frac[8,34]
   - 3 frac[8,53] - 3 frac[8,8] + rows_tot_tor_P43 = 0
 R397: - 0.0208333333333333 rows_tot_tor_P43 + normalized_rows_tot_tor_P43
   = 0
 mem_tot_l1_P42: - mem[9,26] - mem[9,41] - mem[9,51] - mem[9,0] - mem[9,21]
   - mem[9,36] - mem[9,14] - mem[9,3] - mem[9,16] - mem[9,39] - mem[9,52]
   - mem[9,30] - mem[9,19] - mem[9,34] - mem[9,55] - mem[9,4] - mem[9,25]
   - mem[9,50] - mem[9,7] - mem[9,20] - mem[9,43] - mem[9,13] - mem[9,2]
   - mem[9,38] - mem[9,8] - mem[9,29] - mem[9,44] - mem[9,54] - mem[9,11]
   - mem[9,24] - mem[9,47] - mem[9,49] - mem[9,6] - mem[9,27] - mem[9,42]
   - mem[9,12] - mem[9,1] - mem[9,22] - mem[9,37] - mem[9,15] - mem[9,28]
   - mem[9,17] - mem[9,32] - mem[9,53] - mem[9,10] - mem[9,31] - mem[9,46]
   - mem[9,35] - mem[9,48] - mem[9,5] + mem_tot_l1_P42 = 0
 R399: - 0.00173611111111111 mem_tot_l1_P42 + normalized_mem_tot_l1_P42
   = 0
 rows_tot_l1_P42: - 3 frac[9,26] - 3 frac[9,41] - 3 frac[9,51]
   - 3 frac[9,0] - 3 frac[9,21] - 3 frac[9,36] - 3 frac[9,14] - 3 frac[9,3]
   - 3 frac[9,16] - 3 frac[9,39] - 3 frac[9,52] - 3 frac[9,30]
   - 3 frac[9,19] - 3 frac[9,34] - 3 frac[9,55] - 3 frac[9,4]
   - 3 frac[9,25] - 3 frac[9,50] - 3 frac[9,7] - 3 frac[9,20]
   - 3 frac[9,43] - 3 frac[9,13] - 3 frac[9,2] - 3 frac[9,38] - 3 frac[9,8]
   - 3 frac[9,29] - 3 frac[9,44] - 3 frac[9,54] - 3 frac[9,11]
   - 3 frac[9,24] - 3 frac[9,47] - 3 frac[9,49] - 3 frac[9,6]
   - 3 frac[9,27] - 3 frac[9,42] - 3 frac[9,12] - 3 frac[9,1]
   - 3 frac[9,22] - 3 frac[9,37] - 3 frac[9,15] - 3 frac[9,28]
   - 3 frac[9,17] - 3 frac[9,32] - 3 frac[9,53] - 3 frac[9,10]
   - 3 frac[9,31] - 3 frac[9,46] - 3 frac[9,35] - 3 frac[9,48]
   - 3 frac[9,5] + rows_tot_l1_P42 = 0
 R401: - 0.0208333333333333 rows_tot_l1_P42 + normalized_rows_tot_l1_P42
   = 0
 mem_tot_l2_P4: - mem[10,17] - mem[10,46] - mem[10,7] - mem[10,13]
   - mem[10,24] - mem[10,41] - mem[10,51] - mem[10,14] - mem[10,29]
   - mem[10,42] - mem[10,52] - mem[10,3] - mem[10,30] - mem[10,47]
   - mem[10,4] - mem[10,19] - mem[10,32] - mem[10,10] - mem[10,25]
   - mem[10,20] - mem[10,37] - mem[10,48] - mem[10,15] - mem[10,26]
   - mem[10,43] - mem[10,38] - mem[10,53] - mem[10,0] - mem[10,44]
   - mem[10,54] - mem[10,5] - mem[10,16] - mem[10,11] - mem[10,6]
   - mem[10,21] - mem[10,34] - mem[10,49] - mem[10,12] - mem[10,27]
   - mem[10,39] - mem[10,50] - mem[10,1] - mem[10,28] - mem[10,55]
   - mem[10,2] + mem_tot_l2_P4 = 0
 R403: - 0.00173611111111111 mem_tot_l2_P4 + normalized_mem_tot_l2_P4 = 0
 rows_tot_l2_P4: - 3 frac[10,17] - 3 frac[10,46] - 3 frac[10,7]
   - 3 frac[10,13] - 3 frac[10,24] - 3 frac[10,41] - 3 frac[10,51]
   - 3 frac[10,14] - 3 frac[10,29] - 3 frac[10,42] - 3 frac[10,52]
   - 3 frac[10,3] - 3 frac[10,30] - 3 frac[10,47] - 3 frac[10,4]
   - 3 frac[10,19] - 3 frac[10,32] - 3 frac[10,10] - 3 frac[10,25]
   - 3 frac[10,20] - 3 frac[10,37] - 3 frac[10,48] - 3 frac[10,15]
   - 3 frac[10,26] - 3 frac[10,43] - 3 frac[10,38] - 3 frac[10,53]
   - 3 frac[10,0] - 3 frac[10,44] - 3 frac[10,54] - 3 frac[10,5]
   - 3 frac[10,16] - 3 frac[10,11] - 3 frac[10,6] - 3 frac[10,21]
   - 3 frac[10,34] - 3 frac[10,49] - 3 frac[10,12] - 3 frac[10,27]
   - 3 frac[10,39] - 3 frac[10,50] - 3 frac[10,1] - 3 frac[10,28]
   - 3 frac[10,55] - 3 frac[10,2] + rows_tot_l2_P4 = 0
 R405: - 0.0208333333333333 rows_tot_l2_P4 + normalized_rows_tot_l2_P4 = 0
 row_capacity_CPU5: rows_tot_CPU5 <= 12
 row_capacity_CPU9: rows_tot_CPU9 <= 12
 row_capacity_CPU16: rows_tot_CPU16 <= 12
 row_capacity_CPU19: rows_tot_CPU19 <= 12
 row_capacity_CPU20: rows_tot_CPU20 <= 12
 row_capacity_CPU22: rows_tot_CPU22 <= 12
 row_capacity_CPU23: rows_tot_CPU23 <= 12
 row_capacity_CPU30: rows_tot_CPU30 <= 12
 row_capacity_tor_P43: rows_tot_tor_P43 <= 48
 row_capacity_l1_P42: rows_tot_l1_P42 <= 48
 row_capacity_l2_P4: rows_tot_l2_P4 <= 48
 ns_single_CPU5: 3.5 rows_tot_CPU5 - ns_single_CPU5 + pdt_m_rows_CPU5 = 0
 capacity_cores_CPU5: cores_sketch_CPU5 + cores_dpdk_CPU5 <= 7
 ns_sketch_CPU5: - ns_single_CPU5 + 12.5 cores_sketch_CPU5 >= 0
 ns_dpdk_CPU5: - 5.357142857142711 cores_dpdk_CPU5 <= -21.42857142857143
 ns_single_CPU9: 3.5 rows_tot_CPU9 - ns_single_CPU9 + pdt_m_rows_CPU9 = 0
 capacity_cores_CPU9: cores_sketch_CPU9 + cores_dpdk_CPU9 <= 7
 ns_sketch_CPU9: - ns_single_CPU9 + 12.5 cores_sketch_CPU9 >= 0
 ns_dpdk_CPU9: - 5.357142857142711 cores_dpdk_CPU9 <= -21.42857142857143
 ns_single_CPU16: 3.5 rows_tot_CPU16 - ns_single_CPU16 + pdt_m_rows_CPU16
   = 0
 capacity_cores_CPU16: cores_sketch_CPU16 + cores_dpdk_CPU16 <= 7
 ns_sketch_CPU16: - ns_single_CPU16 + 12.5 cores_sketch_CPU16 >= 0
 ns_dpdk_CPU16: - 5.357142857142711 cores_dpdk_CPU16 <= -21.42857142857143
 ns_single_CPU19: 3.5 rows_tot_CPU19 - ns_single_CPU19 + pdt_m_rows_CPU19
   = 0
 capacity_cores_CPU19: cores_sketch_CPU19 + cores_dpdk_CPU19 <= 7
 ns_sketch_CPU19: - ns_single_CPU19 + 12.5 cores_sketch_CPU19 >= 0
 ns_dpdk_CPU19: - 5.357142857142711 cores_dpdk_CPU19 <= -21.42857142857143
 ns_single_CPU20: 3.5 rows_tot_CPU20 - ns_single_CPU20 + pdt_m_rows_CPU20
   = 0
 capacity_cores_CPU20: cores_sketch_CPU20 + cores_dpdk_CPU20 <= 7
 ns_sketch_CPU20: - ns_single_CPU20 + 12.5 cores_sketch_CPU20 >= 0
 ns_dpdk_CPU20: - 5.357142857142711 cores_dpdk_CPU20 <= -21.42857142857143
 ns_single_CPU22: 3.5 rows_tot_CPU22 - ns_single_CPU22 + pdt_m_rows_CPU22
   = 0
 capacity_cores_CPU22: cores_sketch_CPU22 + cores_dpdk_CPU22 <= 7
 ns_sketch_CPU22: - ns_single_CPU22 + 12.5 cores_sketch_CPU22 >= 0
 ns_dpdk_CPU22: - 5.357142857142711 cores_dpdk_CPU22 <= -21.42857142857143
 ns_single_CPU23: 3.5 rows_tot_CPU23 - ns_single_CPU23 + pdt_m_rows_CPU23
   = 0
 capacity_cores_CPU23: cores_sketch_CPU23 + cores_dpdk_CPU23 <= 7
 ns_sketch_CPU23: - ns_single_CPU23 + 12.5 cores_sketch_CPU23 >= 0
 ns_dpdk_CPU23: - 5.357142857142711 cores_dpdk_CPU23 <= -21.42857142857143
 ns_single_CPU30: 3.5 rows_tot_CPU30 - ns_single_CPU30 + pdt_m_rows_CPU30
   = 0
 capacity_cores_CPU30: cores_sketch_CPU30 + cores_dpdk_CPU30 <= 7
 ns_sketch_CPU30: - ns_single_CPU30 + 12.5 cores_sketch_CPU30 >= 0
 ns_dpdk_CPU30: - 5.357142857142711 cores_dpdk_CPU30 <= -21.42857142857143
 ns_tor_P43: ns_tor_P43 = 6.756756756756757
 max_ns_tor_P43: ns_tor_P43 <= 12.5
 ns_l1_P42: ns_l1_P42 = 6.756756756756757
 max_ns_l1_P42: ns_l1_P42 <= 12.5
 ns_l2_P4: ns_l2_P4 = 6.756756756756757
 max_ns_l2_P4: ns_l2_P4 <= 12.5
 res_acc: 1.220703125e-04 mem_tot_CPU5 + 1.220703125e-04 mem_tot_CPU9
   + 1.220703125e-04 mem_tot_CPU16 + 1.220703125e-04 mem_tot_CPU19
   + 1.220703125e-04 mem_tot_CPU20 + 1.220703125e-04 mem_tot_CPU22
   + 1.220703125e-04 mem_tot_CPU23 + 1.220703125e-04 mem_tot_CPU30
   + 0.0208333333333333 mem_tot_tor_P43 + 0.25 rows_tot_tor_P43
   + 0.0208333333333333 mem_tot_l1_P42 + 0.25 rows_tot_l1_P42
   + 0.0208333333333333 mem_tot_l2_P4 + 0.25 rows_tot_l2_P4
   + 10 cores_sketch_CPU5 + 10 cores_dpdk_CPU5 + 10 cores_sketch_CPU9
   + 10 cores_dpdk_CPU9 + 10 cores_sketch_CPU16 + 10 cores_dpdk_CPU16
   + 10 cores_sketch_CPU19 + 10 cores_dpdk_CPU19 + 10 cores_sketch_CPU20
   + 10 cores_dpdk_CPU20 + 10 cores_sketch_CPU22 + 10 cores_dpdk_CPU22
   + 10 cores_sketch_CPU23 + 10 cores_dpdk_CPU23 + 10 cores_sketch_CPU30
   + 10 cores_dpdk_CPU30 - res = 0
 pdt_m_rows_CPU5: pdt_m_rows_CPU5 + [ - rows_tot_CPU5 * m_access_time_CPU5
   ] = 0
 pdt_m_rows_CPU9: pdt_m_rows_CPU9 + [ - rows_tot_CPU9 * m_access_time_CPU9
   ] = 0
 pdt_m_rows_CPU16: pdt_m_rows_CPU16 + [
   - rows_tot_CPU16 * m_access_time_CPU16 ] = 0
 pdt_m_rows_CPU19: pdt_m_rows_CPU19 + [
   - rows_tot_CPU19 * m_access_time_CPU19 ] = 0
 pdt_m_rows_CPU20: pdt_m_rows_CPU20 + [
   - rows_tot_CPU20 * m_access_time_CPU20 ] = 0
 pdt_m_rows_CPU22: pdt_m_rows_CPU22 + [
   - rows_tot_CPU22 * m_access_time_CPU22 ] = 0
 pdt_m_rows_CPU23: pdt_m_rows_CPU23 + [
   - rows_tot_CPU23 * m_access_time_CPU23 ] = 0
 pdt_m_rows_CPU30: pdt_m_rows_CPU30 + [
   - rows_tot_CPU30 * m_access_time_CPU30 ] = 0
Bounds
 frac[9,26] <= 1
 frac[7,3] <= 1
 frac[6,9] <= 1
 frac[3,35] <= 1
 frac[10,17] <= 1
 frac[8,44] <= 1
 frac[7,12] <= 1
 frac[4,36] <= 1
 frac[9,41] <= 1
 frac[7,25] <= 1
 frac[10,46] <= 1
 frac[1,6] <= 1
 frac[8,50] <= 1
 frac[5,54] <= 1
 frac[9,51] <= 1
 frac[8,5] <= 1
 frac[5,8] <= 1
 frac[9,0] <= 1
 frac[8,24] <= 1
 frac[0,32] <= 1
 frac[5,5] <= 1
 frac[3,45] <= 1
 frac[10,7] <= 1
 frac[9,21] <= 1
 frac[4,46] <= 1
 frac[8,41] <= 1
 frac[7,19] <= 1
 frac[6,25] <= 1
 frac[9,36] <= 1
 frac[8,15] <= 1
 frac[3,2] <= 1
 frac[2,6] <= 1
 frac[9,14] <= 1
 frac[8,2] <= 1
 frac[4,5] <= 1
 frac[2,53] <= 1
 frac[10,13] <= 1
 frac[9,3] <= 1
 frac[5,24] <= 1
 frac[3,40] <= 1
 frac[10,24] <= 1
 frac[9,16] <= 1
 frac[7,5] <= 1
 frac[6,23] <= 1
 frac[4,35] <= 1
 frac[8,38] <= 1
 frac[10,41] <= 1
 frac[9,39] <= 1
 frac[7,35] <= 1
 frac[3,12] <= 1
 frac[9,52] <= 1
 frac[8,12] <= 1
 frac[1,12] <= 1
 frac[4,15] <= 1
 frac[10,51] <= 1
 frac[1,49] <= 1
 frac[0,39] <= 1
 frac[8,31] <= 1
 frac[10,14] <= 1
 frac[9,30] <= 1
 frac[2,54] <= 1
 frac[6,13] <= 1
 frac[3,39] <= 1
 frac[10,29] <= 1
 frac[9,19] <= 1
 frac[7,8] <= 1
 frac[4,32] <= 1
 frac[8,35] <= 1
 frac[10,42] <= 1
 frac[2,18] <= 1
 frac[0,14] <= 1
 frac[8,54] <= 1
 frac[9,34] <= 1
 frac[9,55] <= 1
 frac[8,9] <= 1
 frac[4,12] <= 1
 frac[10,52] <= 1
 frac[2,12] <= 1
 frac[9,4] <= 1
 frac[4,31] <= 1
 frac[10,3] <= 1
 frac[9,25] <= 1
 frac[6,14] <= 1
 frac[10,30] <= 1
 frac[8,45] <= 1
 frac[7,15] <= 1
 frac[8,32] <= 1
 frac[10,47] <= 1
 frac[9,50] <= 1
 frac[8,6] <= 1
 frac[9,7] <= 1
 frac[8,25] <= 1
 frac[10,4] <= 1
 frac[9,20] <= 1
 frac[10,19] <= 1
 frac[8,42] <= 1
 frac[5,46] <= 1
 frac[3,50] <= 1
 frac[1,19] <= 1
 frac[9,43] <= 1
 frac[7,31] <= 1
 frac[4,53] <= 1
 frac[10,32] <= 1
 frac[2,24] <= 1
 frac[8,48] <= 1
 frac[3,5] <= 1
 frac[9,13] <= 1
 frac[7,53] <= 1
 frac[4,6] <= 1
 frac[2,50] <= 1
 frac[9,2] <= 1
 frac[0,46] <= 1
 frac[10,10] <= 1
 frac[8,22] <= 1
 frac[10,25] <= 1
 frac[10,20] <= 1
 frac[8,39] <= 1
 frac[5,33] <= 1
 frac[3,49] <= 1
 frac[9,38] <= 1
 frac[1,30] <= 1
 frac[3,15] <= 1
 frac[10,37] <= 1
 frac[8,13] <= 1
 frac[10,48] <= 1
 frac[9,8] <= 1
 frac[5,13] <= 1
 frac[10,15] <= 1
 frac[9,29] <= 1
 frac[8,19] <= 1
 frac[10,26] <= 1
 frac[5,23] <= 1
 frac[9,44] <= 1
 frac[8,36] <= 1
 frac[6,36] <= 1
 frac[5,36] <= 1
 frac[2,19] <= 1
 frac[10,43] <= 1
 frac[0,15] <= 1
 frac[5,49] <= 1
 frac[10,38] <= 1
 frac[1,14] <= 1
 frac[9,54] <= 1
 frac[7,50] <= 1
 frac[3,31] <= 1
 frac[10,53] <= 1
 frac[9,11] <= 1
 frac[2,13] <= 1
 frac[8,29] <= 1
 frac[2,15] <= 1
 frac[10,0] <= 1
 frac[9,24] <= 1
 frac[8,46] <= 1
 frac[7,14] <= 1
 frac[4,38] <= 1
 frac[9,47] <= 1
 frac[8,33] <= 1
 frac[0,25] <= 1
 frac[4,49] <= 1
 frac[10,44] <= 1
 frac[8,52] <= 1
 frac[7,36] <= 1
 frac[0,12] <= 1
 frac[9,49] <= 1
 frac[8,7] <= 1
 frac[10,54] <= 1
 frac[9,6] <= 1
 frac[8,26] <= 1
 frac[10,5] <= 1
 frac[1,35] <= 1
 frac[9,27] <= 1
 frac[6,8] <= 1
 frac[4,40] <= 1
 frac[10,16] <= 1
 frac[8,43] <= 1
 frac[6,31] <= 1
 frac[9,42] <= 1
 frac[1,18] <= 1
 frac[8,49] <= 1
 frac[9,12] <= 1
 frac[8,4] <= 1
 frac[5,9] <= 1
 frac[10,11] <= 1
 frac[9,1] <= 1
 frac[8,23] <= 1
 frac[6,6] <= 1
 frac[5,6] <= 1
 frac[10,6] <= 1
 frac[1,46] <= 1
 frac[9,22] <= 1
 frac[4,45] <= 1
 frac[10,21] <= 1
 frac[8,40] <= 1
 frac[6,24] <= 1
 frac[9,37] <= 1
 frac[6,47] <= 1
 frac[10,34] <= 1
 frac[8,14] <= 1
 frac[10,49] <= 1
 frac[9,15] <= 1
 frac[8,1] <= 1
 frac[2,52] <= 1
 frac[9,28] <= 1
 frac[10,12] <= 1
 frac[5,25] <= 1
 frac[10,27] <= 1
 frac[9,17] <= 1
 frac[5,35] <= 1
 frac[9,32] <= 1
 frac[3,13] <= 1
 frac[10,39] <= 1
 frac[9,53] <= 1
 frac[8,11] <= 1
 frac[1,13] <= 1
 frac[4,14] <= 1
 frac[10,50] <= 1
 frac[1,50] <= 1
 frac[8,30] <= 1
 frac[9,10] <= 1
 frac[3,19] <= 1
 frac[10,1] <= 1
 frac[9,31] <= 1
 frac[10,28] <= 1
 frac[8,47] <= 1
 frac[6,19] <= 1
 frac[9,46] <= 1
 frac[8,34] <= 1
 frac[4,50] <= 1
 frac[9,35] <= 1
 frac[8,53] <= 1
 frac[0,13] <= 1
 frac[6,53] <= 1
 frac[3,8] <= 1
 frac[9,48] <= 1
 frac[8,8] <= 1
 frac[10,55] <= 1
 frac[9,5] <= 1
 frac[0,35] <= 1
 frac[4,30] <= 1
 frac[3,46] <= 1
 frac[10,2] <= 1
 mem_tot_CPU5 <= 32768
 normalized_mem_tot_CPU5 <= 1
 normalized_rows_tot_CPU5 <= 1
 mem_tot_CPU9 <= 32768
 normalized_mem_tot_CPU9 <= 1
 normalized_rows_tot_CPU9 <= 1
 mem_tot_CPU16 <= 32768
 normalized_mem_tot_CPU16 <= 1
 normalized_rows_tot_CPU16 <= 1
 mem_tot_CPU19 <= 32768
 normalized_mem_tot_CPU19 <= 1
 normalized_rows_tot_CPU19 <= 1
 mem_tot_CPU20 <= 32768
 normalized_mem_tot_CPU20 <= 1
 normalized_rows_tot_CPU20 <= 1
 mem_tot_CPU22 <= 32768
 normalized_mem_tot_CPU22 <= 1
 normalized_rows_tot_CPU22 <= 1
 mem_tot_CPU23 <= 32768
 normalized_mem_tot_CPU23 <= 1
 normalized_rows_tot_CPU23 <= 1
 mem_tot_CPU30 <= 32768
 normalized_mem_tot_CPU30 <= 1
 normalized_rows_tot_CPU30 <= 1
 mem_tot_tor_P43 <= 576
 normalized_mem_tot_tor_P43 <= 1
 normalized_rows_tot_tor_P43 <= 1
 mem_tot_l1_P42 <= 576
 normalized_mem_tot_l1_P42 <= 1
 normalized_rows_tot_l1_P42 <= 1
 mem_tot_l2_P4 <= 576
 normalized_mem_tot_l2_P4 <= 1
 normalized_rows_tot_l2_P4 <= 1
 cores_sketch_CPU5 <= 7
 1 <= cores_dpdk_CPU5 <= 7
 cores_sketch_CPU9 <= 7
 1 <= cores_dpdk_CPU9 <= 7
 cores_sketch_CPU16 <= 7
 1 <= cores_dpdk_CPU16 <= 7
 cores_sketch_CPU19 <= 7
 1 <= cores_dpdk_CPU19 <= 7
 cores_sketch_CPU20 <= 7
 1 <= cores_dpdk_CPU20 <= 7
 cores_sketch_CPU22 <= 7
 1 <= cores_dpdk_CPU22 <= 7
 cores_sketch_CPU23 <= 7
 1 <= cores_dpdk_CPU23 <= 7
 cores_sketch_CPU30 <= 7
 1 <= cores_dpdk_CPU30 <= 7
Generals
 cores_sketch_CPU5 cores_dpdk_CPU5 cores_sketch_CPU9 cores_dpdk_CPU9
 cores_sketch_CPU16 cores_dpdk_CPU16 cores_sketch_CPU19 cores_dpdk_CPU19
 cores_sketch_CPU20 cores_dpdk_CPU20 cores_sketch_CPU22 cores_dpdk_CPU22
 cores_sketch_CPU23 cores_dpdk_CPU23 cores_sketch_CPU30 cores_dpdk_CPU30
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_tor_P43 , mem_tot_l1_P42 ,
   mem_tot_l2_P4 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU5 , normalized_mem_tot_CPU9 ,
   normalized_mem_tot_CPU16 , normalized_mem_tot_CPU19 ,
   normalized_mem_tot_CPU20 , normalized_mem_tot_CPU22 ,
   normalized_mem_tot_CPU23 , normalized_mem_tot_CPU30 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_tor_P43 , rows_tot_l1_P42 ,
   rows_tot_l2_P4 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU5 ,
   normalized_rows_tot_CPU9 , normalized_rows_tot_CPU16 ,
   normalized_rows_tot_CPU19 , normalized_rows_tot_CPU20 ,
   normalized_rows_tot_CPU22 , normalized_rows_tot_CPU23 ,
   normalized_rows_tot_CPU30 )
 mem_access_time_CPU5: m_access_time_CPU5 = PWL ( mem_tot_CPU5 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU9: m_access_time_CPU9 = PWL ( mem_tot_CPU9 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU16: m_access_time_CPU16 = PWL ( mem_tot_CPU16 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU19: m_access_time_CPU19 = PWL ( mem_tot_CPU19 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU20: m_access_time_CPU20 = PWL ( mem_tot_CPU20 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU22: m_access_time_CPU22 = PWL ( mem_tot_CPU22 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU23: m_access_time_CPU23 = PWL ( mem_tot_CPU23 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU30: m_access_time_CPU30 = PWL ( mem_tot_CPU30 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_fixed_thr + 0 max_mem_others + 0 max_rows_fixed_thr
   + 0 max_rows_others + 0 m_access_time_CPU1 + 0 m_access_time_CPU2
   + 0 m_access_time_CPU3 + 0 m_access_time_CPU4 + 0 m_access_time_CPU5
   + 0 m_access_time_CPU6 + 0 m_access_time_CPU7 + 0 m_access_time_CPU8
   + 0 m_access_time_CPU9 + 0 m_access_time_CPU10 + 0 m_access_time_CPU11
   + 0 m_access_time_CPU12 + 0 m_access_time_CPU13 + 0 m_access_time_CPU14
   + 0 m_access_time_CPU15 + 0 m_access_time_CPU16 + 0 m_access_time_CPU17
   + 0 m_access_time_CPU18 + 0 m_access_time_CPU19 + 0 m_access_time_CPU20
   + 0 m_access_time_CPU21 + 0 m_access_time_CPU22 + 0 m_access_time_CPU23
   + 0 m_access_time_CPU24 + 0 m_access_time_CPU25 + 0 m_access_time_CPU26
   + 0 m_access_time_CPU27 + 0 m_access_time_CPU28 + 0 m_access_time_CPU29
   + 0 m_access_time_CPU30 + 0 m_access_time_CPU31 + 0 m_access_time_CPU32
   + res
Subject To
 cov_0_24: frac[8,24] + frac[16,24] + frac[33,24] + frac[34,24]
   + frac[36,24] + frac[37,24] + frac[38,24] >= 0.8
 cov_1_26: frac[8,26] + frac[16,26] + frac[33,26] + frac[34,26]
   + frac[36,26] + frac[37,26] + frac[38,26] >= 1
 cov_2_30: frac[8,30] + frac[16,30] + frac[33,30] + frac[34,30]
   + frac[36,30] + frac[37,30] + frac[38,30] >= 0.8
 cov_3_9: frac[6,9] + frac[15,9] + frac[32,9] + frac[33,9] + frac[36,9]
   >= 0.7
 cov_4_20: frac[6,20] + frac[15,20] + frac[32,20] + frac[33,20]
   + frac[36,20] >= 0.8
 cov_5_7: frac[21,7] + frac[28,7] + frac[34,7] + frac[35,7] + frac[37,7]
   >= 0.7
 cov_6_12: frac[15,12] + frac[22,12] + frac[33,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.7
 cov_7_25: frac[4,25] + frac[24,25] + frac[32,25] + frac[35,25]
   + frac[36,25] + frac[37,25] + frac[38,25] >= 0.9
 cov_8_28: frac[6,28] + frac[15,28] + frac[32,28] + frac[33,28]
   + frac[36,28] >= 0.9
 cov_9_22: frac[4,22] + frac[21,22] + frac[32,22] + frac[34,22]
   + frac[36,22] + frac[37,22] + frac[38,22] >= 1
 cov_10_18: frac[4,18] + frac[16,18] + frac[32,18] + frac[34,18]
   + frac[36,18] + frac[37,18] + frac[38,18] >= 0.9
 cov_11_15: frac[2,15] + frac[15,15] + frac[32,15] + frac[33,15]
   + frac[36,15] >= 0.7
 cov_12_31: frac[8,31] + frac[15,31] + frac[33,31] >= 1
 cov_13_26: frac[15,26] + frac[21,26] + frac[33,26] + frac[34,26]
   + frac[36,26] + frac[37,26] + frac[38,26] >= 0.7
 cov_14_11: frac[6,11] + frac[24,11] + frac[32,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.8
 cov_15_28: frac[4,28] + frac[22,28] + frac[32,28] + frac[34,28]
   + frac[36,28] + frac[37,28] + frac[38,28] >= 0.8
 cov_16_16: frac[21,16] + frac[24,16] + frac[34,16] + frac[35,16]
   + frac[37,16] >= 0.8
 cov_17_18: frac[6,18] + frac[8,18] + frac[32,18] + frac[33,18]
   + frac[36,18] >= 0.9
 cov_18_23: frac[15,23] + frac[20,23] + frac[33,23] + frac[34,23]
   + frac[36,23] + frac[37,23] + frac[38,23] >= 0.7
 cov_19_24: frac[6,24] + frac[22,24] + frac[32,24] + frac[34,24]
   + frac[36,24] + frac[37,24] + frac[38,24] >= 0.8
 cov_20_14: frac[8,14] + frac[16,14] + frac[33,14] + frac[34,14]
   + frac[36,14] + frac[37,14] + frac[38,14] >= 0.7
 cov_21_11: frac[9,11] + frac[15,11] + frac[33,11] >= 0.9
 cov_22_5: frac[6,5] + frac[22,5] + frac[32,5] + frac[34,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 1
 cov_23_30: frac[6,30] + frac[21,30] + frac[32,30] + frac[34,30]
   + frac[36,30] + frac[37,30] + frac[38,30] >= 1
 cov_24_21: frac[6,21] + frac[24,21] + frac[32,21] + frac[35,21]
   + frac[36,21] + frac[37,21] + frac[38,21] >= 0.8
 cov_25_16: frac[4,16] + frac[22,16] + frac[32,16] + frac[34,16]
   + frac[36,16] + frac[37,16] + frac[38,16] >= 0.8
 cov_26_5: frac[13,5] + frac[15,5] + frac[33,5] >= 0.9
 cov_27_20: frac[6,20] + frac[22,20] + frac[32,20] + frac[34,20]
   + frac[36,20] + frac[37,20] + frac[38,20] >= 0.8
 cov_28_30: frac[15,30] + frac[22,30] + frac[33,30] + frac[34,30]
   + frac[36,30] + frac[37,30] + frac[38,30] >= 0.9
 cov_29_14: frac[15,14] + frac[21,14] + frac[33,14] + frac[34,14]
   + frac[36,14] + frac[37,14] + frac[38,14] >= 1
 cov_30_18: frac[4,18] + frac[24,18] + frac[32,18] + frac[35,18]
   + frac[36,18] + frac[37,18] + frac[38,18] >= 0.9
 cov_31_18: frac[15,18] + frac[24,18] + frac[33,18] + frac[35,18]
   + frac[36,18] + frac[37,18] + frac[38,18] >= 0.7
 cov_32_9: frac[15,9] + frac[22,9] + frac[33,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 1
 cov_33_28: frac[8,28] + frac[16,28] + frac[33,28] + frac[34,28]
   + frac[36,28] + frac[37,28] + frac[38,28] >= 1
 cov_34_1: frac[16,1] + frac[22,1] + frac[34,1] >= 0.8
 cov_35_5: frac[4,5] + frac[6,5] + frac[32,5] >= 0.7
 cov_36_10: frac[6,10] + frac[8,10] + frac[32,10] + frac[33,10]
   + frac[36,10] >= 1
 cov_37_29: frac[8,29] + frac[21,29] + frac[33,29] + frac[34,29]
   + frac[36,29] + frac[37,29] + frac[38,29] >= 0.7
 cov_38_25: frac[15,25] + frac[16,25] + frac[33,25] + frac[34,25]
   + frac[36,25] + frac[37,25] + frac[38,25] >= 0.7
 cov_39_17: frac[8,17] + frac[26,17] + frac[33,17] + frac[35,17]
   + frac[36,17] + frac[37,17] + frac[38,17] >= 0.8
 cov_40_29: frac[6,29] + frac[16,29] + frac[32,29] + frac[34,29]
   + frac[36,29] + frac[37,29] + frac[38,29] >= 0.9
 cov_41_22: frac[6,22] + frac[15,22] + frac[32,22] + frac[33,22]
   + frac[36,22] >= 0.9
 cov_42_30: frac[4,30] + frac[22,30] + frac[32,30] + frac[34,30]
   + frac[36,30] + frac[37,30] + frac[38,30] >= 0.7
 cov_43_2: frac[16,2] + frac[24,2] + frac[34,2] + frac[35,2] + frac[37,2]
   >= 1
 cov_44_20: frac[8,20] + frac[21,20] + frac[33,20] + frac[34,20]
   + frac[36,20] + frac[37,20] + frac[38,20] >= 1
 cov_45_22: frac[15,22] + frac[24,22] + frac[33,22] + frac[35,22]
   + frac[36,22] + frac[37,22] + frac[38,22] >= 0.9
 cov_46_17: frac[6,17] + frac[22,17] + frac[32,17] + frac[34,17]
   + frac[36,17] + frac[37,17] + frac[38,17] >= 1
 cov_47_9: frac[15,9] + frac[22,9] + frac[33,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.7
 cov_48_29: frac[6,29] + frac[8,29] + frac[32,29] + frac[33,29]
   + frac[36,29] >= 1
 cov_49_17: frac[4,17] + frac[24,17] + frac[32,17] + frac[35,17]
   + frac[36,17] + frac[37,17] + frac[38,17] >= 0.8
 cov_50_25: frac[4,25] + frac[22,25] + frac[32,25] + frac[34,25]
   + frac[36,25] + frac[37,25] + frac[38,25] >= 1
 cov_51_27: frac[8,27] + frac[15,27] + frac[33,27] >= 0.7
 cov_52_11: frac[4,11] + frac[22,11] + frac[32,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.7
 cov_53_22: frac[4,22] + frac[8,22] + frac[32,22] + frac[33,22]
   + frac[36,22] >= 0.9
 cov_54_23: frac[16,23] + frac[21,23] + frac[34,23] >= 1
 cov_55_21: frac[8,21] + frac[21,21] + frac[33,21] + frac[34,21]
   + frac[36,21] + frac[37,21] + frac[38,21] >= 1
 cov_56_3: frac[4,3] + frac[22,3] + frac[32,3] + frac[34,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.8
 cov_57_14: frac[6,14] + frac[24,14] + frac[32,14] + frac[35,14]
   + frac[36,14] + frac[37,14] + frac[38,14] >= 0.9
 cov_58_18: frac[6,18] + frac[8,18] + frac[32,18] + frac[33,18]
   + frac[36,18] >= 1
 cov_59_29: frac[21,29] + frac[24,29] + frac[34,29] + frac[35,29]
   + frac[37,29] >= 1
 cov_60_17: frac[4,17] + frac[22,17] + frac[32,17] + frac[34,17]
   + frac[36,17] + frac[37,17] + frac[38,17] >= 0.7
 cov_61_12: frac[5,12] + frac[21,12] + frac[32,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.9
 cov_62_11: frac[4,11] + frac[21,11] + frac[32,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.9
 cov_63_19: frac[8,19] + frac[24,19] + frac[33,19] + frac[35,19]
   + frac[36,19] + frac[37,19] + frac[38,19] >= 0.9
 cov_64_57: frac[9,57] + frac[11,57] + frac[33,57] >= 0.9
 cov_65_36: frac[9,36] + frac[27,36] + frac[33,36] + frac[35,36]
   + frac[36,36] + frac[37,36] + frac[38,36] >= 0.8
 cov_66_36: frac[11,36] + frac[14,36] + frac[33,36] >= 0.8
 cov_67_42: frac[14,42] + frac[29,42] + frac[33,42] + frac[35,42]
   + frac[36,42] + frac[37,42] + frac[38,42] >= 0.9
 cov_68_34: frac[7,34] + frac[11,34] + frac[32,34] + frac[33,34]
   + frac[36,34] >= 0.7
 cov_69_47: frac[7,47] + frac[18,47] + frac[32,47] + frac[34,47]
   + frac[36,47] + frac[37,47] + frac[38,47] >= 0.7
 cov_70_35: frac[27,35] + frac[29,35] + frac[35,35] >= 0.9
 cov_71_58: frac[9,58] + frac[18,58] + frac[33,58] + frac[34,58]
   + frac[36,58] + frac[37,58] + frac[38,58] >= 0.7
 cov_72_62: frac[9,62] + frac[11,62] + frac[33,62] >= 0.7
 cov_73_59: frac[18,59] + frac[27,59] + frac[34,59] + frac[35,59]
   + frac[37,59] >= 1
 cov_74_34: frac[27,34] + frac[30,34] + frac[35,34] >= 0.8
 cov_75_46: frac[14,46] + frac[29,46] + frac[33,46] + frac[35,46]
   + frac[36,46] + frac[37,46] + frac[38,46] >= 0.9
 cov_76_51: frac[18,51] + frac[27,51] + frac[34,51] + frac[35,51]
   + frac[37,51] >= 0.7
 cov_77_56: frac[14,56] + frac[29,56] + frac[33,56] + frac[35,56]
   + frac[36,56] + frac[37,56] + frac[38,56] >= 0.8
 cov_78_36: frac[14,36] + frac[30,36] + frac[33,36] + frac[35,36]
   + frac[36,36] + frac[37,36] + frac[38,36] >= 0.8
 cov_79_51: frac[29,51] + frac[30,51] + frac[35,51] >= 1
 cov_80_34: frac[7,34] + frac[18,34] + frac[32,34] + frac[34,34]
   + frac[36,34] + frac[37,34] + frac[38,34] >= 1
 cov_81_63: frac[29,63] + frac[30,63] + frac[35,63] >= 0.9
 cov_82_33: frac[14,33] + frac[29,33] + frac[33,33] + frac[35,33]
   + frac[36,33] + frac[37,33] + frac[38,33] >= 0.8
 cov_83_59: frac[9,59] + frac[18,59] + frac[33,59] + frac[34,59]
   + frac[36,59] + frac[37,59] + frac[38,59] >= 1
 cov_84_45: frac[7,45] + frac[9,45] + frac[32,45] + frac[33,45]
   + frac[36,45] >= 0.8
 cov_85_42: frac[11,42] + frac[14,42] + frac[33,42] >= 0.9
 cov_86_55: frac[7,55] + frac[14,55] + frac[32,55] + frac[33,55]
   + frac[36,55] >= 0.7
 cov_87_45: frac[11,45] + frac[27,45] + frac[33,45] + frac[35,45]
   + frac[36,45] + frac[37,45] + frac[38,45] >= 0.7
 cov_88_49: frac[7,49] + frac[9,49] + frac[32,49] + frac[33,49]
   + frac[36,49] >= 1
 cov_89_39: frac[9,39] + frac[14,39] + frac[33,39] >= 0.8
 cov_90_42: frac[9,42] + frac[18,42] + frac[33,42] + frac[34,42]
   + frac[36,42] + frac[37,42] + frac[38,42] >= 0.8
 cov_91_48: frac[7,48] + frac[18,48] + frac[32,48] + frac[34,48]
   + frac[36,48] + frac[37,48] + frac[38,48] >= 0.8
 cov_92_56: frac[9,56] + frac[29,56] + frac[33,56] + frac[35,56]
   + frac[36,56] + frac[37,56] + frac[38,56] >= 0.7
 cov_93_60: frac[9,60] + frac[27,60] + frac[33,60] + frac[35,60]
   + frac[36,60] + frac[37,60] + frac[38,60] >= 0.8
 cov_94_33: frac[11,33] + frac[18,33] + frac[33,33] + frac[34,33]
   + frac[36,33] + frac[37,33] + frac[38,33] >= 0.9
 cov_95_48: frac[7,48] + frac[29,48] + frac[32,48] + frac[35,48]
   + frac[36,48] + frac[37,48] + frac[38,48] >= 0.7
 cov_96_46: frac[11,46] + frac[30,46] + frac[33,46] + frac[35,46]
   + frac[36,46] + frac[37,46] + frac[38,46] >= 1
 cov_97_62: frac[7,62] + frac[18,62] + frac[32,62] + frac[34,62]
   + frac[36,62] + frac[37,62] + frac[38,62] >= 0.9
 cov_98_35: frac[9,35] + frac[11,35] + frac[33,35] >= 0.7
 cov_99_49: frac[27,49] + frac[30,49] + frac[35,49] >= 0.7
 cov_100_52: frac[14,52] + frac[16,52] + frac[33,52] + frac[34,52]
   + frac[36,52] + frac[37,52] + frac[38,52] >= 0.7
 cov_101_34: frac[15,34] + frac[27,34] + frac[33,34] + frac[35,34]
   + frac[36,34] + frac[37,34] + frac[38,34] >= 0.9
 cov_102_32: frac[3,32] + frac[30,32] + frac[32,32] + frac[35,32]
   + frac[36,32] + frac[37,32] + frac[38,32] >= 1
 cov_103_52: frac[9,52] + frac[30,52] + frac[33,52] + frac[35,52]
   + frac[36,52] + frac[37,52] + frac[38,52] >= 0.8
 cov_104_47: frac[9,47] + frac[11,47] + frac[33,47] >= 0.9
 cov_105_40: frac[27,40] + frac[29,40] + frac[35,40] >= 0.9
 cov_106_59: frac[9,59] + frac[12,59] + frac[33,59] >= 0.8
 cov_107_56: frac[11,56] + frac[14,56] + frac[33,56] >= 0.8
 cov_108_59: frac[14,59] + frac[30,59] + frac[33,59] + frac[35,59]
   + frac[36,59] + frac[37,59] + frac[38,59] >= 0.8
 cov_109_62: frac[22,62] + frac[30,62] + frac[34,62] + frac[35,62]
   + frac[37,62] >= 1
 cov_110_38: frac[18,38] + frac[29,38] + frac[34,38] + frac[35,38]
   + frac[37,38] >= 0.8
 cov_111_39: frac[6,39] + frac[7,39] + frac[32,39] >= 0.7
 cov_112_43: frac[14,43] + frac[27,43] + frac[33,43] + frac[35,43]
   + frac[36,43] + frac[37,43] + frac[38,43] >= 1
 cov_113_51: frac[29,51] + frac[30,51] + frac[35,51] >= 0.8
 cov_114_34: frac[7,34] + frac[30,34] + frac[32,34] + frac[35,34]
   + frac[36,34] + frac[37,34] + frac[38,34] >= 1
 cov_115_51: frac[11,51] + frac[30,51] + frac[33,51] + frac[35,51]
   + frac[36,51] + frac[37,51] + frac[38,51] >= 0.7
 cov_116_45: frac[9,45] + frac[27,45] + frac[33,45] + frac[35,45]
   + frac[36,45] + frac[37,45] + frac[38,45] >= 1
 cov_117_48: frac[7,48] + frac[9,48] + frac[32,48] + frac[33,48]
   + frac[36,48] >= 0.9
 cov_118_59: frac[29,59] + frac[30,59] + frac[35,59] >= 0.9
 cov_119_42: frac[9,42] + frac[14,42] + frac[33,42] >= 0.9
 cov_120_56: frac[7,56] + frac[30,56] + frac[32,56] + frac[35,56]
   + frac[36,56] + frac[37,56] + frac[38,56] >= 0.7
 cov_121_47: frac[7,47] + frac[31,47] + frac[32,47] + frac[35,47]
   + frac[36,47] + frac[37,47] + frac[38,47] >= 0.8
 cov_122_61: frac[7,61] + frac[9,61] + frac[32,61] + frac[33,61]
   + frac[36,61] >= 0.7
 cov_123_54: frac[7,54] + frac[14,54] + frac[32,54] + frac[33,54]
   + frac[36,54] >= 0.8
 cov_124_51: frac[11,51] + frac[30,51] + frac[33,51] + frac[35,51]
   + frac[36,51] + frac[37,51] + frac[38,51] >= 0.9
 cov_125_47: frac[18,47] + frac[27,47] + frac[34,47] + frac[35,47]
   + frac[37,47] >= 0.9
 cov_126_45: frac[9,45] + frac[27,45] + frac[33,45] + frac[35,45]
   + frac[36,45] + frac[37,45] + frac[38,45] >= 1
 cov_127_38: frac[8,38] + frac[30,38] + frac[33,38] + frac[35,38]
   + frac[36,38] + frac[37,38] + frac[38,38] >= 0.8
 cov_128_66: frac[19,66] + frac[26,66] + frac[34,66] + frac[35,66]
   + frac[37,66] >= 0.7
 cov_129_77: frac[26,77] + frac[28,77] + frac[35,77] >= 0.8
 cov_130_88: frac[2,88] + frac[7,88] + frac[32,88] >= 1
 cov_131_89: frac[5,89] + frac[30,89] + frac[32,89] + frac[35,89]
   + frac[36,89] + frac[37,89] + frac[38,89] >= 0.7
 cov_132_74: frac[2,74] + frac[19,74] + frac[32,74] + frac[34,74]
   + frac[36,74] + frac[37,74] + frac[38,74] >= 0.9
 cov_133_83: frac[0,83] + frac[28,83] + frac[32,83] + frac[35,83]
   + frac[36,83] + frac[37,83] + frac[38,83] >= 0.9
 cov_134_64: frac[5,64] + frac[28,64] + frac[32,64] + frac[35,64]
   + frac[36,64] + frac[37,64] + frac[38,64] >= 0.9
 cov_135_76: frac[2,76] + frac[26,76] + frac[32,76] + frac[35,76]
   + frac[36,76] + frac[37,76] + frac[38,76] >= 0.7
 cov_136_77: frac[5,77] + frac[10,77] + frac[32,77] + frac[33,77]
   + frac[36,77] >= 1
 cov_137_90: frac[5,90] + frac[13,90] + frac[32,90] + frac[33,90]
   + frac[36,90] >= 0.8
 cov_138_88: frac[0,88] + frac[26,88] + frac[32,88] + frac[35,88]
   + frac[36,88] + frac[37,88] + frac[38,88] >= 1
 cov_139_76: frac[10,76] + frac[28,76] + frac[33,76] + frac[35,76]
   + frac[36,76] + frac[37,76] + frac[38,76] >= 0.9
 cov_140_93: frac[5,93] + frac[28,93] + frac[32,93] + frac[35,93]
   + frac[36,93] + frac[37,93] + frac[38,93] >= 0.9
 cov_141_81: frac[10,81] + frac[19,81] + frac[33,81] + frac[34,81]
   + frac[36,81] + frac[37,81] + frac[38,81] >= 0.9
 cov_142_65: frac[5,65] + frac[10,65] + frac[32,65] + frac[33,65]
   + frac[36,65] >= 0.7
 cov_143_77: frac[10,77] + frac[19,77] + frac[33,77] + frac[34,77]
   + frac[36,77] + frac[37,77] + frac[38,77] >= 0.8
 cov_144_88: frac[5,88] + frac[19,88] + frac[32,88] + frac[34,88]
   + frac[36,88] + frac[37,88] + frac[38,88] >= 0.9
 cov_145_75: frac[19,75] + frac[28,75] + frac[34,75] + frac[35,75]
   + frac[37,75] >= 1
 cov_146_95: frac[10,95] + frac[19,95] + frac[33,95] + frac[34,95]
   + frac[36,95] + frac[37,95] + frac[38,95] >= 0.9
 cov_147_71: frac[2,71] + frac[5,71] + frac[32,71] >= 1
 cov_148_84: frac[0,84] + frac[19,84] + frac[32,84] + frac[34,84]
   + frac[36,84] + frac[37,84] + frac[38,84] >= 0.9
 cov_149_72: frac[0,72] + frac[5,72] + frac[32,72] >= 0.9
 cov_150_94: frac[19,94] + frac[28,94] + frac[34,94] + frac[35,94]
   + frac[37,94] >= 0.9
 cov_151_72: frac[5,72] + frac[10,72] + frac[32,72] + frac[33,72]
   + frac[36,72] >= 0.9
 cov_152_92: frac[2,92] + frac[26,92] + frac[32,92] + frac[35,92]
   + frac[36,92] + frac[37,92] + frac[38,92] >= 1
 cov_153_85: frac[5,85] + frac[28,85] + frac[32,85] + frac[35,85]
   + frac[36,85] + frac[37,85] + frac[38,85] >= 1
 cov_154_89: frac[0,89] + frac[13,89] + frac[32,89] + frac[33,89]
   + frac[36,89] >= 1
 cov_155_74: frac[0,74] + frac[10,74] + frac[32,74] + frac[33,74]
   + frac[36,74] >= 0.7
 cov_156_94: frac[2,94] + frac[10,94] + frac[32,94] + frac[33,94]
   + frac[36,94] >= 0.8
 cov_157_91: frac[7,91] + frac[28,91] + frac[32,91] + frac[35,91]
   + frac[36,91] + frac[37,91] + frac[38,91] >= 0.9
 cov_158_80: frac[19,80] + frac[26,80] + frac[34,80] + frac[35,80]
   + frac[37,80] >= 0.9
 cov_159_77: frac[0,77] + frac[19,77] + frac[32,77] + frac[34,77]
   + frac[36,77] + frac[37,77] + frac[38,77] >= 0.7
 cov_160_73: frac[10,73] + frac[26,73] + frac[33,73] + frac[35,73]
   + frac[36,73] + frac[37,73] + frac[38,73] >= 0.8
 cov_161_82: frac[0,82] + frac[18,82] + frac[32,82] + frac[34,82]
   + frac[36,82] + frac[37,82] + frac[38,82] >= 0.8
 cov_162_77: frac[5,77] + frac[25,77] + frac[32,77] + frac[35,77]
   + frac[36,77] + frac[37,77] + frac[38,77] >= 1
 cov_163_72: frac[19,72] + frac[26,72] + frac[34,72] + frac[35,72]
   + frac[37,72] >= 0.7
 cov_164_93: frac[10,93] + frac[13,93] + frac[33,93] >= 0.8
 cov_165_66: frac[2,66] + frac[13,66] + frac[32,66] + frac[33,66]
   + frac[36,66] >= 1
 cov_166_89: frac[5,89] + frac[19,89] + frac[32,89] + frac[34,89]
   + frac[36,89] + frac[37,89] + frac[38,89] >= 0.7
 cov_167_93: frac[0,93] + frac[2,93] + frac[32,93] >= 1
 cov_168_79: frac[0,79] + frac[28,79] + frac[32,79] + frac[35,79]
   + frac[36,79] + frac[37,79] + frac[38,79] >= 1
 cov_169_82: frac[0,82] + frac[28,82] + frac[32,82] + frac[35,82]
   + frac[36,82] + frac[37,82] + frac[38,82] >= 0.7
 cov_170_92: frac[19,92] + frac[24,92] + frac[34,92] + frac[35,92]
   + frac[37,92] >= 1
 cov_171_73: frac[13,73] + frac[19,73] + frac[33,73] + frac[34,73]
   + frac[36,73] + frac[37,73] + frac[38,73] >= 0.8
 cov_172_77: frac[13,77] + frac[28,77] + frac[33,77] + frac[35,77]
   + frac[36,77] + frac[37,77] + frac[38,77] >= 0.9
 cov_173_95: frac[0,95] + frac[19,95] + frac[32,95] + frac[34,95]
   + frac[36,95] + frac[37,95] + frac[38,95] >= 0.9
 cov_174_75: frac[0,75] + frac[10,75] + frac[32,75] + frac[33,75]
   + frac[36,75] >= 0.9
 cov_175_65: frac[0,65] + frac[28,65] + frac[32,65] + frac[35,65]
   + frac[36,65] + frac[37,65] + frac[38,65] >= 0.9
 cov_176_92: frac[0,92] + frac[19,92] + frac[32,92] + frac[34,92]
   + frac[36,92] + frac[37,92] + frac[38,92] >= 1
 cov_177_88: frac[13,88] + frac[26,88] + frac[33,88] + frac[35,88]
   + frac[36,88] + frac[37,88] + frac[38,88] >= 1
 cov_178_72: frac[13,72] + frac[16,72] + frac[33,72] + frac[34,72]
   + frac[36,72] + frac[37,72] + frac[38,72] >= 0.7
 cov_179_86: frac[5,86] + frac[19,86] + frac[32,86] + frac[34,86]
   + frac[36,86] + frac[37,86] + frac[38,86] >= 0.7
 cov_180_80: frac[2,80] + frac[5,80] + frac[32,80] >= 0.8
 cov_181_77: frac[0,77] + frac[5,77] + frac[32,77] >= 0.8
 cov_182_77: frac[19,77] + frac[26,77] + frac[34,77] + frac[35,77]
   + frac[37,77] >= 0.7
 cov_183_66: frac[19,66] + frac[28,66] + frac[34,66] + frac[35,66]
   + frac[37,66] >= 0.9
 cov_184_69: frac[10,69] + frac[19,69] + frac[33,69] + frac[34,69]
   + frac[36,69] + frac[37,69] + frac[38,69] >= 0.8
 cov_185_90: frac[0,90] + frac[5,90] + frac[32,90] >= 0.9
 cov_186_92: frac[2,92] + frac[13,92] + frac[32,92] + frac[33,92]
   + frac[36,92] >= 0.8
 cov_187_93: frac[2,93] + frac[10,93] + frac[32,93] + frac[33,93]
   + frac[36,93] >= 1
 cov_188_68: frac[5,68] + frac[13,68] + frac[32,68] + frac[33,68]
   + frac[36,68] >= 1
 cov_189_65: frac[0,65] + frac[28,65] + frac[32,65] + frac[35,65]
   + frac[36,65] + frac[37,65] + frac[38,65] >= 0.9
 cov_190_68: frac[2,68] + frac[26,68] + frac[32,68] + frac[35,68]
   + frac[36,68] + frac[37,68] + frac[38,68] >= 0.8
 cov_191_80: frac[0,80] + frac[2,80] + frac[32,80] >= 0.7
 cov_192_97: frac[20,97] + frac[31,97] + frac[34,97] + frac[35,97]
   + frac[37,97] >= 1
 cov_193_105: frac[3,105] + frac[17,105] + frac[32,105] + frac[34,105]
   + frac[36,105] + frac[37,105] + frac[38,105] >= 0.8
 cov_194_107: frac[12,107] + frac[25,107] + frac[33,107] + frac[35,107]
   + frac[36,107] + frac[37,107] + frac[38,107] >= 0.9
 cov_195_107: frac[23,107] + frac[28,107] + frac[34,107] + frac[35,107]
   + frac[37,107] >= 0.7
 cov_196_111: frac[12,111] + frac[31,111] + frac[33,111] + frac[35,111]
   + frac[36,111] + frac[37,111] + frac[38,111] >= 1
 cov_197_106: frac[20,106] + frac[31,106] + frac[34,106] + frac[35,106]
   + frac[37,106] >= 0.8
 cov_198_106: frac[12,106] + frac[20,106] + frac[33,106] + frac[34,106]
   + frac[36,106] + frac[37,106] + frac[38,106] >= 0.7
 cov_199_120: frac[1,120] + frac[3,120] + frac[32,120] >= 1
 cov_200_118: frac[25,118] + frac[31,118] + frac[35,118] >= 0.7
 cov_201_117: frac[1,117] + frac[20,117] + frac[32,117] + frac[34,117]
   + frac[36,117] + frac[37,117] + frac[38,117] >= 0.9
 cov_202_107: frac[1,107] + frac[12,107] + frac[32,107] + frac[33,107]
   + frac[36,107] >= 0.8
 cov_203_98: frac[3,98] + frac[31,98] + frac[32,98] + frac[35,98]
   + frac[36,98] + frac[37,98] + frac[38,98] >= 1
 cov_204_122: frac[1,122] + frac[14,122] + frac[32,122] + frac[33,122]
   + frac[36,122] >= 0.9
 cov_205_110: frac[3,110] + frac[12,110] + frac[32,110] + frac[33,110]
   + frac[36,110] >= 1
 cov_206_114: frac[1,114] + frac[3,114] + frac[32,114] >= 0.9
 cov_207_120: frac[17,120] + frac[31,120] + frac[34,120] + frac[35,120]
   + frac[37,120] >= 0.8
 cov_208_102: frac[12,102] + frac[17,102] + frac[33,102] + frac[34,102]
   + frac[36,102] + frac[37,102] + frac[38,102] >= 0.7
 cov_209_122: frac[17,122] + frac[31,122] + frac[34,122] + frac[35,122]
   + frac[37,122] >= 0.7
 cov_210_120: frac[12,120] + frac[20,120] + frac[33,120] + frac[34,120]
   + frac[36,120] + frac[37,120] + frac[38,120] >= 0.9
 cov_211_123: frac[25,123] + frac[26,123] + frac[35,123] >= 0.8
 cov_212_117: frac[1,117] + frac[3,117] + frac[32,117] >= 0.9
 cov_213_103: frac[1,103] + frac[3,103] + frac[32,103] >= 0.9
 cov_214_96: frac[1,96] + frac[3,96] + frac[32,96] >= 1
 cov_215_103: frac[12,103] + frac[17,103] + frac[33,103] + frac[34,103]
   + frac[36,103] + frac[37,103] + frac[38,103] >= 0.9
 cov_216_126: frac[17,126] + frac[31,126] + frac[34,126] + frac[35,126]
   + frac[37,126] >= 0.7
 cov_217_110: frac[12,110] + frac[23,110] + frac[33,110] + frac[34,110]
   + frac[36,110] + frac[37,110] + frac[38,110] >= 0.8
 cov_218_106: frac[20,106] + frac[31,106] + frac[34,106] + frac[35,106]
   + frac[37,106] >= 0.9
 cov_219_99: frac[1,99] + frac[3,99] + frac[32,99] >= 1
 cov_220_102: frac[3,102] + frac[17,102] + frac[32,102] + frac[34,102]
   + frac[36,102] + frac[37,102] + frac[38,102] >= 1
 cov_221_115: frac[3,115] + frac[17,115] + frac[32,115] + frac[34,115]
   + frac[36,115] + frac[37,115] + frac[38,115] >= 0.8
 cov_222_106: frac[20,106] + frac[31,106] + frac[34,106] + frac[35,106]
   + frac[37,106] >= 0.7
 cov_223_102: frac[3,102] + frac[25,102] + frac[32,102] + frac[35,102]
   + frac[36,102] + frac[37,102] + frac[38,102] >= 0.7
 cov_224_101: frac[1,101] + frac[10,101] + frac[32,101] + frac[33,101]
   + frac[36,101] >= 0.8
 cov_225_113: frac[1,113] + frac[23,113] + frac[32,113] + frac[34,113]
   + frac[36,113] + frac[37,113] + frac[38,113] >= 0.9
 cov_226_114: frac[3,114] + frac[20,114] + frac[32,114] + frac[34,114]
   + frac[36,114] + frac[37,114] + frac[38,114] >= 0.7
 cov_227_120: frac[3,120] + frac[20,120] + frac[32,120] + frac[34,120]
   + frac[36,120] + frac[37,120] + frac[38,120] >= 1
 cov_228_106: frac[12,106] + frac[31,106] + frac[33,106] + frac[35,106]
   + frac[36,106] + frac[37,106] + frac[38,106] >= 0.9
 cov_229_100: frac[12,100] + frac[23,100] + frac[33,100] + frac[34,100]
   + frac[36,100] + frac[37,100] + frac[38,100] >= 0.7
 cov_230_120: frac[1,120] + frac[23,120] + frac[32,120] + frac[34,120]
   + frac[36,120] + frac[37,120] + frac[38,120] >= 0.8
 cov_231_126: frac[12,126] + frac[20,126] + frac[33,126] + frac[34,126]
   + frac[36,126] + frac[37,126] + frac[38,126] >= 1
 cov_232_118: frac[20,118] + frac[31,118] + frac[34,118] + frac[35,118]
   + frac[37,118] >= 0.8
 cov_233_98: frac[1,98] + frac[23,98] + frac[32,98] + frac[34,98]
   + frac[36,98] + frac[37,98] + frac[38,98] >= 0.7
 cov_234_99: frac[12,99] + frac[17,99] + frac[33,99] + frac[34,99]
   + frac[36,99] + frac[37,99] + frac[38,99] >= 0.7
 cov_235_126: frac[19,126] + frac[20,126] + frac[34,126] >= 1
 cov_236_121: frac[12,121] + frac[17,121] + frac[33,121] + frac[34,121]
   + frac[36,121] + frac[37,121] + frac[38,121] >= 0.8
 cov_237_100: frac[1,100] + frac[23,100] + frac[32,100] + frac[34,100]
   + frac[36,100] + frac[37,100] + frac[38,100] >= 1
 cov_238_117: frac[17,117] + frac[25,117] + frac[34,117] + frac[35,117]
   + frac[37,117] >= 0.7
 cov_239_100: frac[17,100] + frac[23,100] + frac[34,100] >= 1
 cov_240_101: frac[17,101] + frac[31,101] + frac[34,101] + frac[35,101]
   + frac[37,101] >= 0.7
 cov_241_108: frac[1,108] + frac[20,108] + frac[32,108] + frac[34,108]
   + frac[36,108] + frac[37,108] + frac[38,108] >= 0.7
 cov_242_119: frac[1,119] + frac[31,119] + frac[32,119] + frac[35,119]
   + frac[36,119] + frac[37,119] + frac[38,119] >= 0.7
 cov_243_105: frac[1,105] + frac[17,105] + frac[32,105] + frac[34,105]
   + frac[36,105] + frac[37,105] + frac[38,105] >= 0.8
 cov_244_117: frac[12,117] + frac[31,117] + frac[33,117] + frac[35,117]
   + frac[36,117] + frac[37,117] + frac[38,117] >= 0.9
 cov_245_99: frac[17,99] + frac[20,99] + frac[34,99] >= 0.8
 cov_246_105: frac[12,105] + frac[30,105] + frac[33,105] + frac[35,105]
   + frac[36,105] + frac[37,105] + frac[38,105] >= 1
 cov_247_110: frac[20,110] + frac[31,110] + frac[34,110] + frac[35,110]
   + frac[37,110] >= 0.7
 cov_248_118: frac[12,118] + frac[17,118] + frac[33,118] + frac[34,118]
   + frac[36,118] + frac[37,118] + frac[38,118] >= 1
 cov_249_110: frac[20,110] + frac[23,110] + frac[34,110] >= 0.8
 cov_250_107: frac[6,107] + frac[23,107] + frac[32,107] + frac[34,107]
   + frac[36,107] + frac[37,107] + frac[38,107] >= 0.9
 cov_251_98: frac[3,98] + frac[17,98] + frac[32,98] + frac[34,98]
   + frac[36,98] + frac[37,98] + frac[38,98] >= 0.9
 cov_252_105: frac[12,105] + frac[20,105] + frac[33,105] + frac[34,105]
   + frac[36,105] + frac[37,105] + frac[38,105] >= 1
 cov_253_121: frac[4,121] + frac[23,121] + frac[32,121] + frac[34,121]
   + frac[36,121] + frac[37,121] + frac[38,121] >= 0.8
 cov_254_100: frac[3,100] + frac[17,100] + frac[32,100] + frac[34,100]
   + frac[36,100] + frac[37,100] + frac[38,100] >= 1
 cov_255_126: frac[12,126] + frac[16,126] + frac[33,126] + frac[34,126]
   + frac[36,126] + frac[37,126] + frac[38,126] >= 0.8
 accuracy_35_88: - 6 frac[35,88] + mem[35,88] = 0
 accuracy_32_54: - 6 frac[32,54] + mem[32,54] = 0
 accuracy_33_55: - 6 frac[33,55] + mem[33,55] = 0
 accuracy_24_92: - 5.09765625 frac[24,92] + mem[24,92] = 0
 accuracy_6_28: - 5.09765625 frac[6,28] + mem[6,28] = 0
 accuracy_36_68: - 6 frac[36,68] + mem[36,68] = 0
 accuracy_37_82: - 6 frac[37,82] + mem[37,82] = 0
 accuracy_34_52: - 6 frac[34,52] + mem[34,52] = 0
 accuracy_9_62: - 5.09765625 frac[9,62] + mem[9,62] = 0
 accuracy_38_86: - 6 frac[38,86] + mem[38,86] = 0
 accuracy_30_51: - 5.09765625 frac[30,51] + mem[30,51] = 0
 accuracy_36_34: - 6 frac[36,34] + mem[36,34] = 0
 accuracy_7_55: - 5.09765625 frac[7,55] + mem[7,55] = 0
 accuracy_12_59: - 5.09765625 frac[12,59] + mem[12,59] = 0
 accuracy_1_122: - 5.09765625 frac[1,122] + mem[1,122] = 0
 accuracy_15_30: - 5.09765625 frac[15,30] + mem[15,30] = 0
 accuracy_35_21: - 6 frac[35,21] + mem[35,21] = 0
 accuracy_32_113: - 6 frac[32,113] + mem[32,113] = 0
 accuracy_24_14: - 5.09765625 frac[24,14] + mem[24,14] = 0
 accuracy_33_101: - 6 frac[33,101] + mem[33,101] = 0
 accuracy_36_22: - 6 frac[36,22] + mem[36,22] = 0
 accuracy_31_126: - 5.09765625 frac[31,126] + mem[31,126] = 0
 accuracy_37_23: - 6 frac[37,23] + mem[37,23] = 0
 accuracy_35_123: - 6 frac[35,123] + mem[35,123] = 0
 accuracy_33_95: - 6 frac[33,95] + mem[33,95] = 0
 accuracy_14_43: - 5.09765625 frac[14,43] + mem[14,43] = 0
 accuracy_12_111: - 5.09765625 frac[12,111] + mem[12,111] = 0
 accuracy_10_75: - 5.09765625 frac[10,75] + mem[10,75] = 0
 accuracy_15_34: - 5.09765625 frac[15,34] + mem[15,34] = 0
 accuracy_4_5: - 5.09765625 frac[4,5] + mem[4,5] = 0
 accuracy_36_101: - 6 frac[36,101] + mem[36,101] = 0
 accuracy_37_113: - 6 frac[37,113] + mem[37,113] = 0
 accuracy_35_65: - 6 frac[35,65] + mem[35,65] = 0
 accuracy_32_77: - 6 frac[32,77] + mem[32,77] = 0
 accuracy_38_119: - 6 frac[38,119] + mem[38,119] = 0
 accuracy_37_91: - 6 frac[37,91] + mem[37,91] = 0
 accuracy_8_38: - 5.09765625 frac[8,38] + mem[8,38] = 0
 accuracy_35_46: - 6 frac[35,46] + mem[35,46] = 0
 accuracy_9_39: - 5.09765625 frac[9,39] + mem[9,39] = 0
 accuracy_33_34: - 6 frac[33,34] + mem[33,34] = 0
 accuracy_38_73: - 6 frac[38,73] + mem[38,73] = 0
 accuracy_32_25: - 6 frac[32,25] + mem[32,25] = 0
 accuracy_22_12: - 5.09765625 frac[22,12] + mem[22,12] = 0
 accuracy_13_89: - 5.09765625 frac[13,89] + mem[13,89] = 0
 accuracy_35_52: - 6 frac[35,52] + mem[35,52] = 0
 accuracy_36_55: - 6 frac[36,55] + mem[36,55] = 0
 accuracy_34_3: - 6 frac[34,3] + mem[34,3] = 0
 accuracy_8_18: - 5.09765625 frac[8,18] + mem[8,18] = 0
 accuracy_28_77: - 5.09765625 frac[28,77] + mem[28,77] = 0
 accuracy_18_62: - 5.09765625 frac[18,62] + mem[18,62] = 0
 accuracy_24_21: - 5.09765625 frac[24,21] + mem[24,21] = 0
 accuracy_5_93: - 5.09765625 frac[5,93] + mem[5,93] = 0
 accuracy_38_5: - 6 frac[38,5] + mem[38,5] = 0
 accuracy_33_126: - 6 frac[33,126] + mem[33,126] = 0
 accuracy_36_29: - 6 frac[36,29] + mem[36,29] = 0
 accuracy_37_16: - 6 frac[37,16] + mem[37,16] = 0
 accuracy_28_107: - 5.09765625 frac[28,107] + mem[28,107] = 0
 accuracy_34_114: - 6 frac[34,114] + mem[34,114] = 0
 accuracy_7_91: - 5.09765625 frac[7,91] + mem[7,91] = 0
 accuracy_32_110: - 6 frac[32,110] + mem[32,110] = 0
 accuracy_12_102: - 5.09765625 frac[12,102] + mem[12,102] = 0
 accuracy_17_117: - 5.09765625 frac[17,117] + mem[17,117] = 0
 accuracy_36_108: - 6 frac[36,108] + mem[36,108] = 0
 accuracy_37_122: - 6 frac[37,122] + mem[37,122] = 0
 accuracy_34_108: - 6 frac[34,108] + mem[34,108] = 0
 accuracy_32_68: - 6 frac[32,68] + mem[32,68] = 0
 accuracy_33_74: - 6 frac[33,74] + mem[33,74] = 0
 accuracy_27_60: - 5.09765625 frac[27,60] + mem[27,60] = 0
 accuracy_6_14: - 5.09765625 frac[6,14] + mem[6,14] = 0
 accuracy_38_110: - 6 frac[38,110] + mem[38,110] = 0
 accuracy_36_74: - 6 frac[36,74] + mem[36,74] = 0
 accuracy_37_68: - 6 frac[37,68] + mem[37,68] = 0
 accuracy_0_90: - 5.09765625 frac[0,90] + mem[0,90] = 0
 accuracy_33_59: - 6 frac[33,59] + mem[33,59] = 0
 accuracy_16_29: - 5.09765625 frac[16,29] + mem[16,29] = 0
 accuracy_38_64: - 6 frac[38,64] + mem[38,64] = 0
 accuracy_32_16: - 6 frac[32,16] + mem[32,16] = 0
 accuracy_11_46: - 5.09765625 frac[11,46] + mem[11,46] = 0
 accuracy_33_29: - 6 frac[33,29] + mem[33,29] = 0
 accuracy_36_62: - 6 frac[36,62] + mem[36,62] = 0
 accuracy_34_26: - 6 frac[34,26] + mem[34,26] = 0
 accuracy_38_60: - 6 frac[38,60] + mem[38,60] = 0
 accuracy_37_25: - 6 frac[37,25] + mem[37,25] = 0
 accuracy_35_105: - 6 frac[35,105] + mem[35,105] = 0
 accuracy_32_101: - 6 frac[32,101] + mem[32,101] = 0
 accuracy_24_2: - 5.09765625 frac[24,2] + mem[24,2] = 0
 accuracy_33_105: - 6 frac[33,105] + mem[33,105] = 0
 accuracy_38_22: - 6 frac[38,22] + mem[38,22] = 0
 accuracy_4_11: - 5.09765625 frac[4,11] + mem[4,11] = 0
 accuracy_36_107: - 6 frac[36,107] + mem[36,107] = 0
 accuracy_34_103: - 6 frac[34,103] + mem[34,103] = 0
 accuracy_35_79: - 6 frac[35,79] + mem[35,79] = 0
 accuracy_32_91: - 6 frac[32,91] + mem[32,91] = 0
 accuracy_37_99: - 6 frac[37,99] + mem[37,99] = 0
 accuracy_16_126: - 5.09765625 frac[16,126] + mem[16,126] = 0
 accuracy_4_17: - 5.09765625 frac[4,17] + mem[4,17] = 0
 accuracy_37_77: - 6 frac[37,77] + mem[37,77] = 0
 accuracy_36_113: - 6 frac[36,113] + mem[36,113] = 0
 accuracy_35_85: - 6 frac[35,85] + mem[35,85] = 0
 accuracy_32_49: - 6 frac[32,49] + mem[32,49] = 0
 accuracy_27_43: - 5.09765625 frac[27,43] + mem[27,43] = 0
 accuracy_33_60: - 6 frac[33,60] + mem[33,60] = 0
 accuracy_30_36: - 5.09765625 frac[30,36] + mem[30,36] = 0
 accuracy_36_95: - 6 frac[36,95] + mem[36,95] = 0
 accuracy_34_59: - 6 frac[34,59] + mem[34,59] = 0
 accuracy_7_34: - 5.09765625 frac[7,34] + mem[7,34] = 0
 accuracy_22_30: - 5.09765625 frac[22,30] + mem[22,30] = 0
 accuracy_35_34: - 6 frac[35,34] + mem[35,34] = 0
 accuracy_9_59: - 5.09765625 frac[9,59] + mem[9,59] = 0
 accuracy_33_22: - 6 frac[33,22] + mem[33,22] = 0
 accuracy_38_93: - 6 frac[38,93] + mem[38,93] = 0
 accuracy_34_21: - 6 frac[34,21] + mem[34,21] = 0
 accuracy_7_56: - 5.09765625 frac[7,56] + mem[7,56] = 0
 accuracy_37_56: - 6 frac[37,56] + mem[37,56] = 0
 accuracy_18_48: - 5.09765625 frac[18,48] + mem[18,48] = 0
 accuracy_36_3: - 6 frac[36,3] + mem[36,3] = 0
 accuracy_1_103: - 5.09765625 frac[1,103] + mem[1,103] = 0
 accuracy_30_89: - 5.09765625 frac[30,89] + mem[30,89] = 0
 accuracy_37_2: - 6 frac[37,2] + mem[37,2] = 0
 accuracy_19_84: - 5.09765625 frac[19,84] + mem[19,84] = 0
 accuracy_35_110: - 6 frac[35,110] + mem[35,110] = 0
 accuracy_38_9: - 6 frac[38,9] + mem[38,9] = 0
 accuracy_17_103: - 5.09765625 frac[17,103] + mem[17,103] = 0
 accuracy_37_108: - 6 frac[37,108] + mem[37,108] = 0
 accuracy_34_126: - 6 frac[34,126] + mem[34,126] = 0
 accuracy_32_82: - 6 frac[32,82] + mem[32,82] = 0
 accuracy_33_68: - 6 frac[33,68] + mem[33,68] = 0
 accuracy_17_121: - 5.09765625 frac[17,121] + mem[17,121] = 0
 accuracy_36_120: - 6 frac[36,120] + mem[36,120] = 0
 accuracy_37_118: - 6 frac[37,118] + mem[37,118] = 0
 accuracy_34_88: - 6 frac[34,88] + mem[34,88] = 0
 accuracy_32_72: - 6 frac[32,72] + mem[32,72] = 0
 accuracy_29_48: - 5.09765625 frac[29,48] + mem[29,48] = 0
 accuracy_6_18: - 5.09765625 frac[6,18] + mem[6,18] = 0
 accuracy_21_26: - 5.09765625 frac[21,26] + mem[21,26] = 0
 accuracy_38_114: - 6 frac[38,114] + mem[38,114] = 0
 accuracy_37_80: - 6 frac[37,80] + mem[37,80] = 0
 accuracy_22_17: - 5.09765625 frac[22,17] + mem[22,17] = 0
 accuracy_13_68: - 5.09765625 frac[13,68] + mem[13,68] = 0
 accuracy_29_42: - 5.09765625 frac[29,42] + mem[29,42] = 0
 accuracy_35_43: - 6 frac[35,43] + mem[35,43] = 0
 accuracy_9_60: - 5.09765625 frac[9,60] + mem[9,60] = 0
 accuracy_33_47: - 6 frac[33,47] + mem[33,47] = 0
 accuracy_26_92: - 5.09765625 frac[26,92] + mem[26,92] = 0
 accuracy_38_84: - 6 frac[38,84] + mem[38,84] = 0
 accuracy_30_49: - 5.09765625 frac[30,49] + mem[30,49] = 0
 accuracy_2_76: - 5.09765625 frac[2,76] + mem[2,76] = 0
 accuracy_7_49: - 5.09765625 frac[7,49] + mem[7,49] = 0
 accuracy_22_11: - 5.09765625 frac[22,11] + mem[22,11] = 0
 accuracy_37_33: - 6 frac[37,33] + mem[37,33] = 0
 accuracy_35_49: - 6 frac[35,49] + mem[35,49] = 0
 accuracy_5_72: - 5.09765625 frac[5,72] + mem[5,72] = 0
 accuracy_38_46: - 6 frac[38,46] + mem[38,46] = 0
 accuracy_36_10: - 6 frac[36,10] + mem[36,10] = 0
 accuracy_1_120: - 5.09765625 frac[1,120] + mem[1,120] = 0
 accuracy_37_11: - 6 frac[37,11] + mem[37,11] = 0
 accuracy_32_115: - 6 frac[32,115] + mem[32,115] = 0
 accuracy_24_16: - 5.09765625 frac[24,16] + mem[24,16] = 0
 accuracy_36_16: - 6 frac[36,16] + mem[36,16] = 0
 accuracy_31_120: - 5.09765625 frac[31,120] + mem[31,120] = 0
 accuracy_37_21: - 6 frac[37,21] + mem[37,21] = 0
 accuracy_34_121: - 6 frac[34,121] + mem[34,121] = 0
 accuracy_32_105: - 6 frac[32,105] + mem[32,105] = 0
 accuracy_33_93: - 6 frac[33,93] + mem[33,93] = 0
 accuracy_38_26: - 6 frac[38,26] + mem[38,26] = 0
 accuracy_12_105: - 5.09765625 frac[12,105] + mem[12,105] = 0
 accuracy_10_73: - 5.09765625 frac[10,73] + mem[10,73] = 0
 accuracy_36_103: - 6 frac[36,103] + mem[36,103] = 0
 accuracy_32_79: - 6 frac[32,79] + mem[32,79] = 0
 accuracy_16_2: - 5.09765625 frac[16,2] + mem[16,2] = 0
 accuracy_6_21: - 5.09765625 frac[6,21] + mem[6,21] = 0
 accuracy_38_117: - 6 frac[38,117] + mem[38,117] = 0
 accuracy_36_77: - 6 frac[36,77] + mem[36,77] = 0
 accuracy_37_89: - 6 frac[37,89] + mem[37,89] = 0
 accuracy_34_77: - 6 frac[34,77] + mem[34,77] = 0
 accuracy_29_35: - 5.09765625 frac[29,35] + mem[29,35] = 0
 accuracy_16_24: - 5.09765625 frac[16,24] + mem[16,24] = 0
 accuracy_38_79: - 6 frac[38,79] + mem[38,79] = 0
 accuracy_30_56: - 5.09765625 frac[30,56] + mem[30,56] = 0
 accuracy_36_43: - 6 frac[36,43] + mem[36,43] = 0
 accuracy_2_71: - 5.09765625 frac[2,71] + mem[2,71] = 0
 accuracy_8_14: - 5.09765625 frac[8,14] + mem[8,14] = 0
 accuracy_37_42: - 6 frac[37,42] + mem[37,42] = 0
 accuracy_20_118: - 5.09765625 frac[20,118] + mem[20,118] = 0
 accuracy_18_34: - 5.09765625 frac[18,34] + mem[18,34] = 0
 accuracy_33_26: - 6 frac[33,26] + mem[33,26] = 0
 accuracy_38_33: - 6 frac[38,33] + mem[38,33] = 0
 accuracy_36_49: - 6 frac[36,49] + mem[36,49] = 0
 accuracy_1_113: - 5.09765625 frac[1,113] + mem[1,113] = 0
 accuracy_34_1: - 6 frac[34,1] + mem[34,1] = 0
 accuracy_8_20: - 5.09765625 frac[8,20] + mem[8,20] = 0
 accuracy_37_52: - 6 frac[37,52] + mem[37,52] = 0
 accuracy_28_79: - 5.09765625 frac[28,79] + mem[28,79] = 0
 accuracy_38_59: - 6 frac[38,59] + mem[38,59] = 0
 accuracy_3_103: - 5.09765625 frac[3,103] + mem[3,103] = 0
 accuracy_1_107: - 5.09765625 frac[1,107] + mem[1,107] = 0
 accuracy_23_120: - 5.09765625 frac[23,120] + mem[23,120] = 0
 accuracy_15_15: - 5.09765625 frac[15,15] + mem[15,15] = 0
 accuracy_31_97: - 5.09765625 frac[31,97] + mem[31,97] = 0
 accuracy_37_30: - 6 frac[37,30] + mem[37,30] = 0
 accuracy_19_72: - 5.09765625 frac[19,72] + mem[19,72] = 0
 accuracy_35_98: - 6 frac[35,98] + mem[35,98] = 0
 accuracy_32_96: - 6 frac[32,96] + mem[32,96] = 0
 accuracy_38_29: - 6 frac[38,29] + mem[38,29] = 0
 accuracy_36_110: - 6 frac[36,110] + mem[36,110] = 0
 accuracy_37_120: - 6 frac[37,120] + mem[37,120] = 0
 accuracy_34_106: - 6 frac[34,106] + mem[34,106] = 0
 accuracy_35_72: - 6 frac[35,72] + mem[35,72] = 0
 accuracy_33_72: - 6 frac[33,72] + mem[33,72] = 0
 accuracy_12_126: - 5.09765625 frac[12,126] + mem[12,126] = 0
 accuracy_38_108: - 6 frac[38,108] + mem[38,108] = 0
 accuracy_37_66: - 6 frac[37,66] + mem[37,66] = 0
 accuracy_0_92: - 5.09765625 frac[0,92] + mem[0,92] = 0
 accuracy_20_23: - 5.09765625 frac[20,23] + mem[20,23] = 0
 accuracy_33_57: - 6 frac[33,57] + mem[33,57] = 0
 accuracy_21_22: - 5.09765625 frac[21,22] + mem[21,22] = 0
 accuracy_36_82: - 6 frac[36,82] + mem[36,82] = 0
 accuracy_34_62: - 6 frac[34,62] + mem[34,62] = 0
 accuracy_2_94: - 5.09765625 frac[2,94] + mem[2,94] = 0
 accuracy_32_18: - 6 frac[32,18] + mem[32,18] = 0
 accuracy_22_5: - 5.09765625 frac[22,5] + mem[22,5] = 0
 accuracy_7_39: - 5.09765625 frac[7,39] + mem[7,39] = 0
 accuracy_35_63: - 6 frac[35,63] + mem[35,63] = 0
 accuracy_9_48: - 5.09765625 frac[9,48] + mem[9,48] = 0
 accuracy_33_19: - 6 frac[33,19] + mem[33,19] = 0
 accuracy_26_72: - 5.09765625 frac[26,72] + mem[26,72] = 0
 accuracy_38_88: - 6 frac[38,88] + mem[38,88] = 0
 accuracy_36_56: - 6 frac[36,56] + mem[36,56] = 0
 accuracy_34_24: - 6 frac[34,24] + mem[34,24] = 0
 accuracy_8_27: - 5.09765625 frac[8,27] + mem[8,27] = 0
 accuracy_7_61: - 5.09765625 frac[7,61] + mem[7,61] = 0
 accuracy_33_117: - 6 frac[33,117] + mem[33,117] = 0
 accuracy_5_68: - 5.09765625 frac[5,68] + mem[5,68] = 0
 accuracy_1_108: - 5.09765625 frac[1,108] + mem[1,108] = 0
 accuracy_23_113: - 5.09765625 frac[23,113] + mem[23,113] = 0
 accuracy_15_20: - 5.09765625 frac[15,20] + mem[15,20] = 0
 accuracy_31_110: - 5.09765625 frac[31,110] + mem[31,110] = 0
 accuracy_37_7: - 6 frac[37,7] + mem[37,7] = 0
 accuracy_19_81: - 5.09765625 frac[19,81] + mem[19,81] = 0
 accuracy_35_107: - 6 frac[35,107] + mem[35,107] = 0
 accuracy_32_103: - 6 frac[32,103] + mem[32,103] = 0
 accuracy_33_111: - 6 frac[33,111] + mem[33,111] = 0
 accuracy_14_59: - 5.09765625 frac[14,59] + mem[14,59] = 0
 accuracy_38_20: - 6 frac[38,20] + mem[38,20] = 0
 accuracy_3_114: - 5.09765625 frac[3,114] + mem[3,114] = 0
 accuracy_37_97: - 6 frac[37,97] + mem[37,97] = 0
 accuracy_34_101: - 6 frac[34,101] + mem[34,101] = 0
 accuracy_32_93: - 6 frac[32,93] + mem[32,93] = 0
 accuracy_33_65: - 6 frac[33,65] + mem[33,65] = 0
 accuracy_12_117: - 5.09765625 frac[12,117] + mem[12,117] = 0
 accuracy_10_69: - 5.09765625 frac[10,69] + mem[10,69] = 0
 accuracy_38_103: - 6 frac[38,103] + mem[38,103] = 0
 accuracy_36_115: - 6 frac[36,115] + mem[36,115] = 0
 accuracy_37_75: - 6 frac[37,75] + mem[37,75] = 0
 accuracy_34_95: - 6 frac[34,95] + mem[34,95] = 0
 accuracy_0_83: - 5.09765625 frac[0,83] + mem[0,83] = 0
 accuracy_27_45: - 5.09765625 frac[27,45] + mem[27,45] = 0
 accuracy_25_77: - 5.09765625 frac[25,77] + mem[25,77] = 0
 accuracy_38_121: - 6 frac[38,121] + mem[38,121] = 0
 accuracy_36_89: - 6 frac[36,89] + mem[36,89] = 0
 accuracy_37_85: - 6 frac[37,85] + mem[37,85] = 0
 accuracy_22_28: - 5.09765625 frac[22,28] + mem[22,28] = 0
 accuracy_13_73: - 5.09765625 frac[13,73] + mem[13,73] = 0
 accuracy_35_36: - 6 frac[35,36] + mem[35,36] = 0
 accuracy_9_57: - 5.09765625 frac[9,57] + mem[9,57] = 0
 accuracy_33_20: - 6 frac[33,20] + mem[33,20] = 0
 accuracy_38_83: - 6 frac[38,83] + mem[38,83] = 0
 accuracy_32_15: - 6 frac[32,15] + mem[32,15] = 0
 accuracy_37_38: - 6 frac[37,38] + mem[37,38] = 0
 accuracy_28_93: - 5.09765625 frac[28,93] + mem[28,93] = 0
 accuracy_33_14: - 6 frac[33,14] + mem[33,14] = 0
 accuracy_5_77: - 5.09765625 frac[5,77] + mem[5,77] = 0
 accuracy_1_101: - 5.09765625 frac[1,101] + mem[1,101] = 0
 accuracy_19_86: - 5.09765625 frac[19,86] + mem[19,86] = 0
 accuracy_35_16: - 6 frac[35,16] + mem[35,16] = 0
 accuracy_24_11: - 5.09765625 frac[24,11] + mem[24,11] = 0
 accuracy_23_100: - 5.09765625 frac[23,100] + mem[23,100] = 0
 accuracy_17_101: - 5.09765625 frac[17,101] + mem[17,101] = 0
 accuracy_37_106: - 6 frac[37,106] + mem[37,106] = 0
 accuracy_35_118: - 6 frac[35,118] + mem[35,118] = 0
 accuracy_32_84: - 6 frac[32,84] + mem[32,84] = 0
 accuracy_33_90: - 6 frac[33,90] + mem[33,90] = 0
 accuracy_14_36: - 5.09765625 frac[14,36] + mem[14,36] = 0
 accuracy_10_76: - 5.09765625 frac[10,76] + mem[10,76] = 0
 accuracy_36_122: - 6 frac[36,122] + mem[36,122] = 0
 accuracy_34_86: - 6 frac[34,86] + mem[34,86] = 0
 accuracy_35_92: - 6 frac[35,92] + mem[35,92] = 0
 accuracy_32_74: - 6 frac[32,74] + mem[32,74] = 0
 accuracy_27_34: - 5.09765625 frac[27,34] + mem[27,34] = 0
 accuracy_36_64: - 6 frac[36,64] + mem[36,64] = 0
 accuracy_37_94: - 6 frac[37,94] + mem[37,94] = 0
 accuracy_34_48: - 6 frac[34,48] + mem[34,48] = 0
 accuracy_2_80: - 5.09765625 frac[2,80] + mem[2,80] = 0
 accuracy_32_32: - 6 frac[32,32] + mem[32,32] = 0
 accuracy_13_66: - 5.09765625 frac[13,66] + mem[13,66] = 0
 accuracy_29_40: - 5.09765625 frac[29,40] + mem[29,40] = 0
 accuracy_35_45: - 6 frac[35,45] + mem[35,45] = 0
 accuracy_33_45: - 6 frac[33,45] + mem[33,45] = 0
 accuracy_38_74: - 6 frac[38,74] + mem[38,74] = 0
 accuracy_36_46: - 6 frac[36,46] + mem[36,46] = 0
 accuracy_34_42: - 6 frac[34,42] + mem[34,42] = 0
 accuracy_2_74: - 5.09765625 frac[2,74] + mem[2,74] = 0
 accuracy_22_9: - 5.09765625 frac[22,9] + mem[22,9] = 0
 accuracy_37_47: - 6 frac[37,47] + mem[37,47] = 0
 accuracy_13_92: - 5.09765625 frac[13,92] + mem[13,92] = 0
 accuracy_35_51: - 6 frac[35,51] + mem[35,51] = 0
 accuracy_26_68: - 5.09765625 frac[26,68] + mem[26,68] = 0
 accuracy_20_105: - 5.09765625 frac[20,105] + mem[20,105] = 0
 accuracy_36_52: - 6 frac[36,52] + mem[36,52] = 0
 accuracy_15_26: - 5.09765625 frac[15,26] + mem[15,26] = 0
 accuracy_37_9: - 6 frac[37,9] + mem[37,9] = 0
 accuracy_19_95: - 5.09765625 frac[19,95] + mem[19,95] = 0
 accuracy_35_25: - 6 frac[35,25] + mem[35,25] = 0
 accuracy_32_117: - 6 frac[32,117] + mem[32,117] = 0
 accuracy_24_18: - 5.09765625 frac[24,18] + mem[24,18] = 0
 accuracy_5_80: - 5.09765625 frac[5,80] + mem[5,80] = 0
 accuracy_33_121: - 6 frac[33,121] + mem[33,121] = 0
 accuracy_3_96: - 5.09765625 frac[3,96] + mem[3,96] = 0
 accuracy_36_18: - 6 frac[36,18] + mem[36,18] = 0
 accuracy_16_72: - 5.09765625 frac[16,72] + mem[16,72] = 0
 accuracy_31_122: - 5.09765625 frac[31,122] + mem[31,122] = 0
 accuracy_37_19: - 6 frac[37,19] + mem[37,19] = 0
 accuracy_19_69: - 5.09765625 frac[19,69] + mem[19,69] = 0
 accuracy_32_107: - 6 frac[32,107] + mem[32,107] = 0
 accuracy_38_24: - 6 frac[38,24] + mem[38,24] = 0
 accuracy_12_107: - 5.09765625 frac[12,107] + mem[12,107] = 0
 accuracy_18_82: - 5.09765625 frac[18,82] + mem[18,82] = 0
 accuracy_34_81: - 6 frac[34,81] + mem[34,81] = 0
 accuracy_32_65: - 6 frac[32,65] + mem[32,65] = 0
 accuracy_27_59: - 5.09765625 frac[27,59] + mem[27,59] = 0
 accuracy_10_81: - 5.09765625 frac[10,81] + mem[10,81] = 0
 accuracy_6_11: - 5.09765625 frac[6,11] + mem[6,11] = 0
 accuracy_38_107: - 6 frac[38,107] + mem[38,107] = 0
 accuracy_36_79: - 6 frac[36,79] + mem[36,79] = 0
 accuracy_34_75: - 6 frac[34,75] + mem[34,75] = 0
 accuracy_32_39: - 6 frac[32,39] + mem[32,39] = 0
 accuracy_29_33: - 5.09765625 frac[29,33] + mem[29,33] = 0
 accuracy_33_38: - 6 frac[33,38] + mem[33,38] = 0
 accuracy_16_26: - 5.09765625 frac[16,26] + mem[16,26] = 0
 accuracy_38_77: - 6 frac[38,77] + mem[38,77] = 0
 accuracy_21_11: - 5.09765625 frac[21,11] + mem[21,11] = 0
 accuracy_30_62: - 5.09765625 frac[30,62] + mem[30,62] = 0
 accuracy_36_85: - 6 frac[36,85] + mem[36,85] = 0
 accuracy_32_29: - 6 frac[32,29] + mem[32,29] = 0
 accuracy_28_83: - 5.09765625 frac[28,83] + mem[28,83] = 0
 accuracy_11_45: - 5.09765625 frac[11,45] + mem[11,45] = 0
 accuracy_35_56: - 6 frac[35,56] + mem[35,56] = 0
 accuracy_19_126: - 5.09765625 frac[19,126] + mem[19,126] = 0
 accuracy_33_24: - 6 frac[33,24] + mem[33,24] = 0
 accuracy_25_107: - 5.09765625 frac[25,107] + mem[25,107] = 0
 accuracy_36_51: - 6 frac[36,51] + mem[36,51] = 0
 accuracy_1_119: - 5.09765625 frac[1,119] + mem[1,119] = 0
 accuracy_8_22: - 5.09765625 frac[8,22] + mem[8,22] = 0
 accuracy_18_58: - 5.09765625 frac[18,58] + mem[18,58] = 0
 accuracy_24_25: - 5.09765625 frac[24,25] + mem[24,25] = 0
 accuracy_5_89: - 5.09765625 frac[5,89] + mem[5,89] = 0
 accuracy_3_105: - 5.09765625 frac[3,105] + mem[3,105] = 0
 accuracy_36_25: - 6 frac[36,25] + mem[36,25] = 0
 accuracy_1_105: - 5.09765625 frac[1,105] + mem[1,105] = 0
 accuracy_4_121: - 5.09765625 frac[4,121] + mem[4,121] = 0
 accuracy_15_9: - 5.09765625 frac[15,9] + mem[15,9] = 0
 accuracy_37_28: - 6 frac[37,28] + mem[37,28] = 0
 accuracy_19_74: - 5.09765625 frac[19,74] + mem[19,74] = 0
 accuracy_32_98: - 6 frac[32,98] + mem[32,98] = 0
 accuracy_14_54: - 5.09765625 frac[14,54] + mem[14,54] = 0
 accuracy_38_19: - 6 frac[38,19] + mem[38,19] = 0
 accuracy_17_105: - 5.09765625 frac[17,105] + mem[17,105] = 0
 accuracy_37_102: - 6 frac[37,102] + mem[37,102] = 0
 accuracy_32_88: - 6 frac[32,88] + mem[32,88] = 0
 accuracy_12_120: - 5.09765625 frac[12,120] + mem[12,120] = 0
 accuracy_38_98: - 6 frac[38,98] + mem[38,98] = 0
 accuracy_4_22: - 5.09765625 frac[4,22] + mem[4,22] = 0
 accuracy_37_64: - 6 frac[37,64] + mem[37,64] = 0
 accuracy_34_66: - 6 frac[34,66] + mem[34,66] = 0
 accuracy_35_80: - 6 frac[35,80] + mem[35,80] = 0
 accuracy_32_62: - 6 frac[32,62] + mem[32,62] = 0
 accuracy_36_118: - 6 frac[36,118] + mem[36,118] = 0
 accuracy_38_68: - 6 frac[38,68] + mem[38,68] = 0
 accuracy_21_20: - 5.09765625 frac[21,20] + mem[21,20] = 0
 accuracy_36_92: - 6 frac[36,92] + mem[36,92] = 0
 accuracy_2_92: - 5.09765625 frac[2,92] + mem[2,92] = 0
 accuracy_32_20: - 6 frac[32,20] + mem[32,20] = 0
 accuracy_11_34: - 5.09765625 frac[11,34] + mem[11,34] = 0
 accuracy_35_33: - 6 frac[35,33] + mem[35,33] = 0
 accuracy_33_17: - 6 frac[33,17] + mem[33,17] = 0
 accuracy_36_58: - 6 frac[36,58] + mem[36,58] = 0
 accuracy_34_22: - 6 frac[34,22] + mem[34,22] = 0
 accuracy_8_29: - 5.09765625 frac[8,29] + mem[8,29] = 0
 accuracy_32_10: - 6 frac[32,10] + mem[32,10] = 0
 accuracy_37_59: - 6 frac[37,59] + mem[37,59] = 0
 accuracy_11_56: - 5.09765625 frac[11,56] + mem[11,56] = 0
 accuracy_35_7: - 6 frac[35,7] + mem[35,7] = 0
 accuracy_28_64: - 5.09765625 frac[28,64] + mem[28,64] = 0
 accuracy_33_11: - 6 frac[33,11] + mem[33,11] = 0
 accuracy_38_48: - 6 frac[38,48] + mem[38,48] = 0
 accuracy_3_110: - 5.09765625 frac[3,110] + mem[3,110] = 0
 accuracy_1_98: - 5.09765625 frac[1,98] + mem[1,98] = 0
 accuracy_15_22: - 5.09765625 frac[15,22] + mem[15,22] = 0
 accuracy_37_5: - 6 frac[37,5] + mem[37,5] = 0
 accuracy_32_121: - 6 frac[32,121] + mem[32,121] = 0
 accuracy_17_98: - 5.09765625 frac[17,98] + mem[17,98] = 0
 accuracy_31_118: - 5.09765625 frac[31,118] + mem[31,118] = 0
 accuracy_37_111: - 6 frac[37,111] + mem[37,111] = 0
 accuracy_34_99: - 6 frac[34,99] + mem[34,99] = 0
 accuracy_32_95: - 6 frac[32,95] + mem[32,95] = 0
 accuracy_6_5: - 5.09765625 frac[6,5] + mem[6,5] = 0
 accuracy_37_73: - 6 frac[37,73] + mem[37,73] = 0
 accuracy_35_89: - 6 frac[35,89] + mem[35,89] = 0
 accuracy_29_51: - 5.09765625 frac[29,51] + mem[29,51] = 0
 accuracy_3_32: - 5.09765625 frac[3,32] + mem[3,32] = 0
 accuracy_27_47: - 5.09765625 frac[27,47] + mem[27,47] = 0
 accuracy_33_48: - 6 frac[33,48] + mem[33,48] = 0
 accuracy_21_29: - 5.09765625 frac[21,29] + mem[21,29] = 0
 accuracy_36_91: - 6 frac[36,91] + mem[36,91] = 0
 accuracy_37_83: - 6 frac[37,83] + mem[37,83] = 0
 accuracy_0_75: - 5.09765625 frac[0,75] + mem[0,75] = 0
 accuracy_35_38: - 6 frac[35,38] + mem[35,38] = 0
 accuracy_33_42: - 6 frac[33,42] + mem[33,42] = 0
 accuracy_25_117: - 5.09765625 frac[25,117] + mem[25,117] = 0
 accuracy_38_81: - 6 frac[38,81] + mem[38,81] = 0
 accuracy_21_7: - 5.09765625 frac[21,7] + mem[21,7] = 0
 accuracy_36_33: - 6 frac[36,33] + mem[36,33] = 0
 accuracy_34_17: - 6 frac[34,17] + mem[34,17] = 0
 accuracy_37_36: - 6 frac[37,36] + mem[37,36] = 0
 accuracy_20_108: - 5.09765625 frac[20,108] + mem[20,108] = 0
 accuracy_26_123: - 5.09765625 frac[26,123] + mem[26,123] = 0
 accuracy_33_12: - 6 frac[33,12] + mem[33,12] = 0
 accuracy_38_43: - 6 frac[38,43] + mem[38,43] = 0
 accuracy_36_15: - 6 frac[36,15] + mem[36,15] = 0
 accuracy_34_11: - 6 frac[34,11] + mem[34,11] = 0
 accuracy_15_31: - 5.09765625 frac[15,31] + mem[15,31] = 0
 accuracy_37_14: - 6 frac[37,14] + mem[37,14] = 0
 accuracy_19_88: - 5.09765625 frac[19,88] + mem[19,88] = 0
 accuracy_35_18: - 6 frac[35,18] + mem[35,18] = 0
 accuracy_33_102: - 6 frac[33,102] + mem[33,102] = 0
 accuracy_5_85: - 5.09765625 frac[5,85] + mem[5,85] = 0
 accuracy_36_21: - 6 frac[36,21] + mem[36,21] = 0
 accuracy_15_5: - 5.09765625 frac[15,5] + mem[15,5] = 0
 accuracy_34_122: - 6 frac[34,122] + mem[34,122] = 0
 accuracy_35_120: - 6 frac[35,120] + mem[35,120] = 0
 accuracy_32_86: - 6 frac[32,86] + mem[32,86] = 0
 accuracy_33_88: - 6 frac[33,88] + mem[33,88] = 0
 accuracy_14_42: - 5.09765625 frac[14,42] + mem[14,42] = 0
 accuracy_12_110: - 5.09765625 frac[12,110] + mem[12,110] = 0
 accuracy_10_74: - 5.09765625 frac[10,74] + mem[10,74] = 0
 accuracy_36_100: - 6 frac[36,100] + mem[36,100] = 0
 accuracy_37_114: - 6 frac[37,114] + mem[37,114] = 0
 accuracy_34_84: - 6 frac[34,84] + mem[34,84] = 0
 accuracy_35_94: - 6 frac[35,94] + mem[35,94] = 0
 accuracy_32_76: - 6 frac[32,76] + mem[32,76] = 0
 accuracy_27_36: - 5.09765625 frac[27,36] + mem[27,36] = 0
 accuracy_6_22: - 5.09765625 frac[6,22] + mem[6,22] = 0
 accuracy_38_118: - 6 frac[38,118] + mem[38,118] = 0
 accuracy_36_66: - 6 frac[36,66] + mem[36,66] = 0
 accuracy_37_92: - 6 frac[37,92] + mem[37,92] = 0
 accuracy_32_34: - 6 frac[32,34] + mem[32,34] = 0
 accuracy_29_38: - 5.09765625 frac[29,38] + mem[29,38] = 0
 accuracy_35_47: - 6 frac[35,47] + mem[35,47] = 0
 accuracy_26_88: - 5.09765625 frac[26,88] + mem[26,88] = 0
 accuracy_33_35: - 6 frac[33,35] + mem[33,35] = 0
 accuracy_38_72: - 6 frac[38,72] + mem[38,72] = 0
 accuracy_32_24: - 6 frac[32,24] + mem[32,24] = 0
 accuracy_7_45: - 5.09765625 frac[7,45] + mem[7,45] = 0
 accuracy_37_45: - 6 frac[37,45] + mem[37,45] = 0
 accuracy_13_90: - 5.09765625 frac[13,90] + mem[13,90] = 0
 accuracy_26_66: - 5.09765625 frac[26,66] + mem[26,66] = 0
 accuracy_33_5: - 6 frac[33,5] + mem[33,5] = 0
 accuracy_38_34: - 6 frac[38,34] + mem[38,34] = 0
 accuracy_36_54: - 6 frac[36,54] + mem[36,54] = 0
 accuracy_34_2: - 6 frac[34,2] + mem[34,2] = 0
 accuracy_8_17: - 5.09765625 frac[8,17] + mem[8,17] = 0
 accuracy_28_76: - 5.09765625 frac[28,76] + mem[28,76] = 0
 accuracy_32_119: - 6 frac[32,119] + mem[32,119] = 0
 accuracy_3_98: - 5.09765625 frac[3,98] + mem[3,98] = 0
 accuracy_36_28: - 6 frac[36,28] + mem[36,28] = 0
 accuracy_37_17: - 6 frac[37,17] + mem[37,17] = 0
 accuracy_34_117: - 6 frac[34,117] + mem[34,117] = 0
 accuracy_35_97: - 6 frac[35,97] + mem[35,97] = 0
 accuracy_7_88: - 5.09765625 frac[7,88] + mem[7,88] = 0
 accuracy_33_81: - 6 frac[33,81] + mem[33,81] = 0
 accuracy_38_30: - 6 frac[38,30] + mem[38,30] = 0
 accuracy_17_118: - 5.09765625 frac[17,118] + mem[17,118] = 0
 accuracy_4_3: - 5.09765625 frac[4,3] + mem[4,3] = 0
 accuracy_36_99: - 6 frac[36,99] + mem[36,99] = 0
 accuracy_2_15: - 5.09765625 frac[2,15] + mem[2,15] = 0
 accuracy_33_75: - 6 frac[33,75] + mem[33,75] = 0
 accuracy_10_95: - 5.09765625 frac[10,95] + mem[10,95] = 0
 accuracy_6_9: - 5.09765625 frac[6,9] + mem[6,9] = 0
 accuracy_38_105: - 6 frac[38,105] + mem[38,105] = 0
 accuracy_36_73: - 6 frac[36,73] + mem[36,73] = 0
 accuracy_37_69: - 6 frac[37,69] + mem[37,69] = 0
 accuracy_34_73: - 6 frac[34,73] + mem[34,73] = 0
 accuracy_0_89: - 5.09765625 frac[0,89] + mem[0,89] = 0
 accuracy_29_63: - 5.09765625 frac[29,63] + mem[29,63] = 0
 accuracy_33_36: - 6 frac[33,36] + mem[33,36] = 0
 accuracy_16_28: - 5.09765625 frac[16,28] + mem[16,28] = 0
 accuracy_11_47: - 5.09765625 frac[11,47] + mem[11,47] = 0
 accuracy_26_77: - 5.09765625 frac[26,77] + mem[26,77] = 0
 accuracy_20_114: - 5.09765625 frac[20,114] + mem[20,114] = 0
 accuracy_33_30: - 6 frac[33,30] + mem[33,30] = 0
 accuracy_36_61: - 6 frac[36,61] + mem[36,61] = 0
 accuracy_1_117: - 5.09765625 frac[1,117] + mem[1,117] = 0
 accuracy_34_29: - 6 frac[34,29] + mem[34,29] = 0
 accuracy_8_24: - 5.09765625 frac[8,24] + mem[8,24] = 0
 accuracy_37_48: - 6 frac[37,48] + mem[37,48] = 0
 accuracy_28_75: - 5.09765625 frac[28,75] + mem[28,75] = 0
 accuracy_5_71: - 5.09765625 frac[5,71] + mem[5,71] = 0
 accuracy_15_11: - 5.09765625 frac[15,11] + mem[15,11] = 0
 accuracy_37_26: - 6 frac[37,26] + mem[37,26] = 0
 accuracy_26_17: - 5.09765625 frac[26,17] + mem[26,17] = 0
 accuracy_35_102: - 6 frac[35,102] + mem[35,102] = 0
 accuracy_33_106: - 6 frac[33,106] + mem[33,106] = 0
 accuracy_14_52: - 5.09765625 frac[14,52] + mem[14,52] = 0
 accuracy_38_17: - 6 frac[38,17] + mem[38,17] = 0
 accuracy_32_100: - 6 frac[32,100] + mem[32,100] = 0
 accuracy_36_106: - 6 frac[36,106] + mem[36,106] = 0
 accuracy_37_100: - 6 frac[37,100] + mem[37,100] = 0
 accuracy_34_102: - 6 frac[34,102] + mem[34,102] = 0
 accuracy_35_76: - 6 frac[35,76] + mem[35,76] = 0
 accuracy_32_90: - 6 frac[32,90] + mem[32,90] = 0
 accuracy_33_76: - 6 frac[33,76] + mem[33,76] = 0
 accuracy_4_16: - 5.09765625 frac[4,16] + mem[4,16] = 0
 accuracy_35_82: - 6 frac[35,82] + mem[35,82] = 0
 accuracy_32_48: - 6 frac[32,48] + mem[32,48] = 0
 accuracy_0_80: - 5.09765625 frac[0,80] + mem[0,80] = 0
 accuracy_29_56: - 5.09765625 frac[29,56] + mem[29,56] = 0
 accuracy_27_40: - 5.09765625 frac[27,40] + mem[27,40] = 0
 accuracy_33_61: - 6 frac[33,61] + mem[33,61] = 0
 accuracy_36_94: - 6 frac[36,94] + mem[36,94] = 0
 accuracy_34_58: - 6 frac[34,58] + mem[34,58] = 0
 accuracy_32_22: - 6 frac[32,22] + mem[32,22] = 0
 accuracy_22_25: - 5.09765625 frac[22,25] + mem[22,25] = 0
 accuracy_11_36: - 5.09765625 frac[11,36] + mem[11,36] = 0
 accuracy_35_35: - 6 frac[35,35] + mem[35,35] = 0
 accuracy_9_52: - 5.09765625 frac[9,52] + mem[9,52] = 0
 accuracy_33_23: - 6 frac[33,23] + mem[33,23] = 0
 accuracy_38_92: - 6 frac[38,92] + mem[38,92] = 0
 accuracy_36_36: - 6 frac[36,36] + mem[36,36] = 0
 accuracy_34_20: - 6 frac[34,20] + mem[34,20] = 0
 accuracy_8_31: - 5.09765625 frac[8,31] + mem[8,31] = 0
 accuracy_32_12: - 6 frac[32,12] + mem[32,12] = 0
 accuracy_28_66: - 5.09765625 frac[28,66] + mem[28,66] = 0
 accuracy_18_51: - 5.09765625 frac[18,51] + mem[18,51] = 0
 accuracy_33_9: - 6 frac[33,9] + mem[33,9] = 0
 accuracy_5_64: - 5.09765625 frac[5,64] + mem[5,64] = 0
 accuracy_1_96: - 5.09765625 frac[1,96] + mem[1,96] = 0
 accuracy_34_14: - 6 frac[34,14] + mem[34,14] = 0
 accuracy_31_106: - 5.09765625 frac[31,106] + mem[31,106] = 0
 accuracy_37_3: - 6 frac[37,3] + mem[37,3] = 0
 accuracy_35_111: - 6 frac[35,111] + mem[35,111] = 0
 accuracy_33_99: - 6 frac[33,99] + mem[33,99] = 0
 accuracy_23_107: - 5.09765625 frac[23,107] + mem[23,107] = 0
 accuracy_34_97: - 6 frac[34,97] + mem[34,97] = 0
 accuracy_35_117: - 6 frac[35,117] + mem[35,117] = 0
 accuracy_33_69: - 6 frac[33,69] + mem[33,69] = 0
 accuracy_14_33: - 5.09765625 frac[14,33] + mem[14,33] = 0
 accuracy_10_65: - 5.09765625 frac[10,65] + mem[10,65] = 0
 accuracy_17_122: - 5.09765625 frac[17,122] + mem[17,122] = 0
 accuracy_37_119: - 6 frac[37,119] + mem[37,119] = 0
 accuracy_35_91: - 6 frac[35,91] + mem[35,91] = 0
 accuracy_32_55: - 6 frac[32,55] + mem[32,55] = 0
 accuracy_22_62: - 5.09765625 frac[22,62] + mem[22,62] = 0
 accuracy_33_54: - 6 frac[33,54] + mem[33,54] = 0
 accuracy_6_29: - 5.09765625 frac[6,29] + mem[6,29] = 0
 accuracy_30_46: - 5.09765625 frac[30,46] + mem[30,46] = 0
 accuracy_36_69: - 6 frac[36,69] + mem[36,69] = 0
 accuracy_37_81: - 6 frac[37,81] + mem[37,81] = 0
 accuracy_32_45: - 6 frac[32,45] + mem[32,45] = 0
 accuracy_22_16: - 5.09765625 frac[22,16] + mem[22,16] = 0
 accuracy_31_47: - 5.09765625 frac[31,47] + mem[31,47] = 0
 accuracy_0_77: - 5.09765625 frac[0,77] + mem[0,77] = 0
 accuracy_35_40: - 6 frac[35,40] + mem[35,40] = 0
 accuracy_9_61: - 5.09765625 frac[9,61] + mem[9,61] = 0
 accuracy_25_123: - 5.09765625 frac[25,123] + mem[25,123] = 0
 accuracy_34_47: - 6 frac[34,47] + mem[34,47] = 0
 accuracy_32_3: - 6 frac[32,3] + mem[32,3] = 0
 accuracy_7_54: - 5.09765625 frac[7,54] + mem[7,54] = 0
 accuracy_37_34: - 6 frac[37,34] + mem[37,34] = 0
 accuracy_11_51: - 5.09765625 frac[11,51] + mem[11,51] = 0
 accuracy_35_14: - 6 frac[35,14] + mem[35,14] = 0
 accuracy_18_42: - 5.09765625 frac[18,42] + mem[18,42] = 0
 accuracy_20_110: - 5.09765625 frac[20,110] + mem[20,110] = 0
 accuracy_36_9: - 6 frac[36,9] + mem[36,9] = 0
 accuracy_34_9: - 6 frac[34,9] + mem[34,9] = 0
 accuracy_15_25: - 5.09765625 frac[15,25] + mem[15,25] = 0
 accuracy_37_12: - 6 frac[37,12] + mem[37,12] = 0
 accuracy_32_114: - 6 frac[32,114] + mem[32,114] = 0
 accuracy_33_100: - 6 frac[33,100] + mem[33,100] = 0
 accuracy_38_3: - 6 frac[38,3] + mem[38,3] = 0
 accuracy_36_23: - 6 frac[36,23] + mem[36,23] = 0
 accuracy_37_22: - 6 frac[37,22] + mem[37,22] = 0
 accuracy_34_120: - 6 frac[34,120] + mem[34,120] = 0
 accuracy_35_122: - 6 frac[35,122] + mem[35,122] = 0
 accuracy_33_94: - 6 frac[33,94] + mem[33,94] = 0
 accuracy_10_72: - 5.09765625 frac[10,72] + mem[10,72] = 0
 accuracy_17_115: - 5.09765625 frac[17,115] + mem[17,115] = 0
 accuracy_36_102: - 6 frac[36,102] + mem[36,102] = 0
 accuracy_34_82: - 6 frac[34,82] + mem[34,82] = 0
 accuracy_35_64: - 6 frac[35,64] + mem[35,64] = 0
 accuracy_16_1: - 5.09765625 frac[16,1] + mem[16,1] = 0
 accuracy_6_20: - 5.09765625 frac[6,20] + mem[6,20] = 0
 accuracy_36_76: - 6 frac[36,76] + mem[36,76] = 0
 accuracy_33_33: - 6 frac[33,33] + mem[33,33] = 0
 accuracy_21_14: - 5.09765625 frac[21,14] + mem[21,14] = 0
 accuracy_36_42: - 6 frac[36,42] + mem[36,42] = 0
 accuracy_30_59: - 5.09765625 frac[30,59] + mem[30,59] = 0
 accuracy_34_38: - 6 frac[34,38] + mem[34,38] = 0
 accuracy_7_47: - 5.09765625 frac[7,47] + mem[7,47] = 0
 accuracy_13_88: - 5.09765625 frac[13,88] + mem[13,88] = 0
 accuracy_37_43: - 6 frac[37,43] + mem[37,43] = 0
 accuracy_20_117: - 5.09765625 frac[20,117] + mem[20,117] = 0
 accuracy_33_27: - 6 frac[33,27] + mem[33,27] = 0
 accuracy_25_102: - 5.09765625 frac[25,102] + mem[25,102] = 0
 accuracy_38_32: - 6 frac[38,32] + mem[38,32] = 0
 accuracy_36_48: - 6 frac[36,48] + mem[36,48] = 0
 accuracy_1_114: - 5.09765625 frac[1,114] + mem[1,114] = 0
 accuracy_8_19: - 5.09765625 frac[8,19] + mem[8,19] = 0
 accuracy_35_29: - 6 frac[35,29] + mem[35,29] = 0
 accuracy_24_22: - 5.09765625 frac[24,22] + mem[24,22] = 0
 accuracy_38_58: - 6 frac[38,58] + mem[38,58] = 0
 accuracy_3_100: - 5.09765625 frac[3,100] + mem[3,100] = 0
 accuracy_36_30: - 6 frac[36,30] + mem[36,30] = 0
 accuracy_23_121: - 5.09765625 frac[23,121] + mem[23,121] = 0
 accuracy_15_12: - 5.09765625 frac[15,12] + mem[15,12] = 0
 accuracy_19_73: - 5.09765625 frac[19,73] + mem[19,73] = 0
 accuracy_34_115: - 6 frac[34,115] + mem[34,115] = 0
 accuracy_38_28: - 6 frac[38,28] + mem[38,28] = 0
 accuracy_12_103: - 5.09765625 frac[12,103] + mem[12,103] = 0
 accuracy_37_121: - 6 frac[37,121] + mem[37,121] = 0
 accuracy_35_73: - 6 frac[35,73] + mem[35,73] = 0
 accuracy_33_73: - 6 frac[33,73] + mem[33,73] = 0
 accuracy_10_93: - 5.09765625 frac[10,93] + mem[10,93] = 0
 accuracy_38_111: - 6 frac[38,111] + mem[38,111] = 0
 accuracy_36_75: - 6 frac[36,75] + mem[36,75] = 0
 accuracy_9_47: - 5.09765625 frac[9,47] + mem[9,47] = 0
 accuracy_33_58: - 6 frac[33,58] + mem[33,58] = 0
 accuracy_16_30: - 5.09765625 frac[16,30] + mem[16,30] = 0
 accuracy_38_65: - 6 frac[38,65] + mem[38,65] = 0
 accuracy_21_23: - 5.09765625 frac[21,23] + mem[21,23] = 0
 accuracy_30_34: - 5.09765625 frac[30,34] + mem[30,34] = 0
 accuracy_36_81: - 6 frac[36,81] + mem[36,81] = 0
 accuracy_34_33: - 6 frac[34,33] + mem[34,33] = 0
 accuracy_32_17: - 6 frac[32,17] + mem[32,17] = 0
 accuracy_11_33: - 5.09765625 frac[11,33] + mem[11,33] = 0
 accuracy_35_60: - 6 frac[35,60] + mem[35,60] = 0
 accuracy_9_49: - 5.09765625 frac[9,49] + mem[9,49] = 0
 accuracy_33_28: - 6 frac[33,28] + mem[33,28] = 0
 accuracy_16_52: - 5.09765625 frac[16,52] + mem[16,52] = 0
 accuracy_38_91: - 6 frac[38,91] + mem[38,91] = 0
 accuracy_8_26: - 5.09765625 frac[8,26] + mem[8,26] = 0
 accuracy_37_62: - 6 frac[37,62] + mem[37,62] = 0
 accuracy_35_2: - 6 frac[35,2] + mem[35,2] = 0
 accuracy_24_29: - 5.09765625 frac[24,29] + mem[24,29] = 0
 accuracy_33_118: - 6 frac[33,118] + mem[33,118] = 0
 accuracy_36_5: - 6 frac[36,5] + mem[36,5] = 0
 accuracy_13_5: - 5.09765625 frac[13,5] + mem[13,5] = 0
 accuracy_37_24: - 6 frac[37,24] + mem[37,24] = 0
 accuracy_31_111: - 5.09765625 frac[31,111] + mem[31,111] = 0
 accuracy_32_102: - 6 frac[32,102] + mem[32,102] = 0
 accuracy_36_126: - 6 frac[36,126] + mem[36,126] = 0
 accuracy_38_23: - 6 frac[38,23] + mem[38,23] = 0
 accuracy_3_115: - 5.09765625 frac[3,115] + mem[3,115] = 0
 accuracy_31_117: - 5.09765625 frac[31,117] + mem[31,117] = 0
 accuracy_37_98: - 6 frac[37,98] + mem[37,98] = 0
 accuracy_34_100: - 6 frac[34,100] + mem[34,100] = 0
 accuracy_32_92: - 6 frac[32,92] + mem[32,92] = 0
 accuracy_33_66: - 6 frac[33,66] + mem[33,66] = 0
 accuracy_38_102: - 6 frac[38,102] + mem[38,102] = 0
 accuracy_4_18: - 5.09765625 frac[4,18] + mem[4,18] = 0
 accuracy_37_76: - 6 frac[37,76] + mem[37,76] = 0
 accuracy_34_94: - 6 frac[34,94] + mem[34,94] = 0
 accuracy_36_114: - 6 frac[36,114] + mem[36,114] = 0
 accuracy_0_82: - 5.09765625 frac[0,82] + mem[0,82] = 0
 accuracy_33_51: - 6 frac[33,51] + mem[33,51] = 0
 accuracy_6_24: - 5.09765625 frac[6,24] + mem[6,24] = 0
 accuracy_21_16: - 5.09765625 frac[21,16] + mem[21,16] = 0
 accuracy_36_88: - 6 frac[36,88] + mem[36,88] = 0
 accuracy_37_86: - 6 frac[37,86] + mem[37,86] = 0
 accuracy_2_88: - 5.09765625 frac[2,88] + mem[2,88] = 0
 accuracy_38_120: - 6 frac[38,120] + mem[38,120] = 0
 accuracy_0_72: - 5.09765625 frac[0,72] + mem[0,72] = 0
 accuracy_9_58: - 5.09765625 frac[9,58] + mem[9,58] = 0
 accuracy_33_21: - 6 frac[33,21] + mem[33,21] = 0
 accuracy_38_82: - 6 frac[38,82] + mem[38,82] = 0
 accuracy_36_38: - 6 frac[36,38] + mem[36,38] = 0
 accuracy_34_18: - 6 frac[34,18] + mem[34,18] = 0
 accuracy_32_14: - 6 frac[32,14] + mem[32,14] = 0
 accuracy_20_97: - 5.09765625 frac[20,97] + mem[20,97] = 0
 accuracy_35_11: - 6 frac[35,11] + mem[35,11] = 0
 accuracy_33_15: - 6 frac[33,15] + mem[33,15] = 0
 accuracy_38_52: - 6 frac[38,52] + mem[38,52] = 0
 accuracy_36_12: - 6 frac[36,12] + mem[36,12] = 0
 accuracy_34_12: - 6 frac[34,12] + mem[34,12] = 0
 accuracy_15_18: - 5.09765625 frac[15,18] + mem[15,18] = 0
 accuracy_35_17: - 6 frac[35,17] + mem[35,17] = 0
 accuracy_38_14: - 6 frac[38,14] + mem[38,14] = 0
 accuracy_3_120: - 5.09765625 frac[3,120] + mem[3,120] = 0
 accuracy_10_101: - 5.09765625 frac[10,101] + mem[10,101] = 0
 accuracy_17_102: - 5.09765625 frac[17,102] + mem[17,102] = 0
 accuracy_37_107: - 6 frac[37,107] + mem[37,107] = 0
 accuracy_35_119: - 6 frac[35,119] + mem[35,119] = 0
 accuracy_32_83: - 6 frac[32,83] + mem[32,83] = 0
 accuracy_14_39: - 5.09765625 frac[14,39] + mem[14,39] = 0
 accuracy_17_120: - 5.09765625 frac[17,120] + mem[17,120] = 0
 accuracy_4_25: - 5.09765625 frac[4,25] + mem[4,25] = 0
 accuracy_37_117: - 6 frac[37,117] + mem[37,117] = 0
 accuracy_34_89: - 6 frac[34,89] + mem[34,89] = 0
 accuracy_35_93: - 6 frac[35,93] + mem[35,93] = 0
 accuracy_36_121: - 6 frac[36,121] + mem[36,121] = 0
 accuracy_27_35: - 5.09765625 frac[27,35] + mem[27,35] = 0
 accuracy_33_52: - 6 frac[33,52] + mem[33,52] = 0
 accuracy_38_115: - 6 frac[38,115] + mem[38,115] = 0
 accuracy_37_95: - 6 frac[37,95] + mem[37,95] = 0
 accuracy_34_51: - 6 frac[34,51] + mem[34,51] = 0
 accuracy_32_47: - 6 frac[32,47] + mem[32,47] = 0
 accuracy_0_79: - 5.09765625 frac[0,79] + mem[0,79] = 0
 accuracy_35_42: - 6 frac[35,42] + mem[35,42] = 0
 accuracy_9_35: - 5.09765625 frac[9,35] + mem[9,35] = 0
 accuracy_33_46: - 6 frac[33,46] + mem[33,46] = 0
 accuracy_38_85: - 6 frac[38,85] + mem[38,85] = 0
 accuracy_36_45: - 6 frac[36,45] + mem[36,45] = 0
 accuracy_32_5: - 6 frac[32,5] + mem[32,5] = 0
 accuracy_7_48: - 5.09765625 frac[7,48] + mem[7,48] = 0
 accuracy_37_32: - 6 frac[37,32] + mem[37,32] = 0
 accuracy_28_91: - 5.09765625 frac[28,91] + mem[28,91] = 0
 accuracy_35_48: - 6 frac[35,48] + mem[35,48] = 0
 accuracy_13_93: - 5.09765625 frac[13,93] + mem[13,93] = 0
 accuracy_38_47: - 6 frac[38,47] + mem[38,47] = 0
 accuracy_36_11: - 6 frac[36,11] + mem[36,11] = 0
 accuracy_34_7: - 6 frac[34,7] + mem[34,7] = 0
 accuracy_15_27: - 5.09765625 frac[15,27] + mem[15,27] = 0
 accuracy_19_92: - 5.09765625 frac[19,92] + mem[19,92] = 0
 accuracy_35_22: - 6 frac[35,22] + mem[35,22] = 0
 accuracy_24_17: - 5.09765625 frac[24,17] + mem[24,17] = 0
 accuracy_33_122: - 6 frac[33,122] + mem[33,122] = 0
 accuracy_36_17: - 6 frac[36,17] + mem[36,17] = 0
 accuracy_23_98: - 5.09765625 frac[23,98] + mem[23,98] = 0
 accuracy_37_20: - 6 frac[37,20] + mem[37,20] = 0
 accuracy_19_66: - 5.09765625 frac[19,66] + mem[19,66] = 0
 accuracy_34_118: - 6 frac[34,118] + mem[34,118] = 0
 accuracy_33_92: - 6 frac[33,92] + mem[33,92] = 0
 accuracy_14_46: - 5.09765625 frac[14,46] + mem[14,46] = 0
 accuracy_12_106: - 5.09765625 frac[12,106] + mem[12,106] = 0
 accuracy_37_126: - 6 frac[37,126] + mem[37,126] = 0
 accuracy_34_80: - 6 frac[34,80] + mem[34,80] = 0
 accuracy_35_66: - 6 frac[35,66] + mem[35,66] = 0
 accuracy_32_64: - 6 frac[32,64] + mem[32,64] = 0
 accuracy_6_10: - 5.09765625 frac[6,10] + mem[6,10] = 0
 accuracy_38_106: - 6 frac[38,106] + mem[38,106] = 0
 accuracy_37_88: - 6 frac[37,88] + mem[37,88] = 0
 accuracy_34_74: - 6 frac[34,74] + mem[34,74] = 0
 accuracy_14_122: - 5.09765625 frac[14,122] + mem[14,122] = 0
 accuracy_9_36: - 5.09765625 frac[9,36] + mem[9,36] = 0
 accuracy_33_39: - 6 frac[33,39] + mem[33,39] = 0
 accuracy_16_25: - 5.09765625 frac[16,25] + mem[16,25] = 0
 accuracy_38_76: - 6 frac[38,76] + mem[38,76] = 0
 accuracy_21_12: - 5.09765625 frac[21,12] + mem[21,12] = 0
 accuracy_36_84: - 6 frac[36,84] + mem[36,84] = 0
 accuracy_2_68: - 5.09765625 frac[2,68] + mem[2,68] = 0
 accuracy_32_28: - 6 frac[32,28] + mem[32,28] = 0
 accuracy_22_3: - 5.09765625 frac[22,3] + mem[22,3] = 0
 accuracy_28_82: - 5.09765625 frac[28,82] + mem[28,82] = 0
 accuracy_11_42: - 5.09765625 frac[11,42] + mem[11,42] = 0
 accuracy_33_25: - 6 frac[33,25] + mem[33,25] = 0
 accuracy_38_38: - 6 frac[38,38] + mem[38,38] = 0
 accuracy_34_30: - 6 frac[34,30] + mem[34,30] = 0
 accuracy_8_21: - 5.09765625 frac[8,21] + mem[8,21] = 0
 accuracy_37_51: - 6 frac[37,51] + mem[37,51] = 0
 accuracy_5_90: - 5.09765625 frac[5,90] + mem[5,90] = 0
 accuracy_38_56: - 6 frac[38,56] + mem[38,56] = 0
 accuracy_3_102: - 5.09765625 frac[3,102] + mem[3,102] = 0
 accuracy_36_24: - 6 frac[36,24] + mem[36,24] = 0
 accuracy_15_14: - 5.09765625 frac[15,14] + mem[15,14] = 0
 accuracy_37_29: - 6 frac[37,29] + mem[37,29] = 0
 accuracy_19_75: - 5.09765625 frac[19,75] + mem[19,75] = 0
 accuracy_34_113: - 6 frac[34,113] + mem[34,113] = 0
 accuracy_35_101: - 6 frac[35,101] + mem[35,101] = 0
 accuracy_38_18: - 6 frac[38,18] + mem[38,18] = 0
 accuracy_36_111: - 6 frac[36,111] + mem[36,111] = 0
 accuracy_37_103: - 6 frac[37,103] + mem[37,103] = 0
 accuracy_34_107: - 6 frac[34,107] + mem[34,107] = 0
 accuracy_35_75: - 6 frac[35,75] + mem[35,75] = 0
 accuracy_32_71: - 6 frac[32,71] + mem[32,71] = 0
 accuracy_27_49: - 5.09765625 frac[27,49] + mem[27,49] = 0
 accuracy_36_117: - 6 frac[36,117] + mem[36,117] = 0
 accuracy_37_65: - 6 frac[37,65] + mem[37,65] = 0
 accuracy_34_69: - 6 frac[34,69] + mem[34,69] = 0
 accuracy_32_61: - 6 frac[32,61] + mem[32,61] = 0
 accuracy_0_93: - 5.09765625 frac[0,93] + mem[0,93] = 0
 accuracy_29_59: - 5.09765625 frac[29,59] + mem[29,59] = 0
 accuracy_9_45: - 5.09765625 frac[9,45] + mem[9,45] = 0
 accuracy_33_56: - 6 frac[33,56] + mem[33,56] = 0
 accuracy_6_39: - 5.09765625 frac[6,39] + mem[6,39] = 0
 accuracy_21_21: - 5.09765625 frac[21,21] + mem[21,21] = 0
 accuracy_30_32: - 5.09765625 frac[30,32] + mem[30,32] = 0
 accuracy_36_83: - 6 frac[36,83] + mem[36,83] = 0
 accuracy_11_35: - 5.09765625 frac[11,35] + mem[11,35] = 0
 accuracy_35_62: - 6 frac[35,62] + mem[35,62] = 0
 accuracy_26_73: - 5.09765625 frac[26,73] + mem[26,73] = 0
 accuracy_33_18: - 6 frac[33,18] + mem[33,18] = 0
 accuracy_20_126: - 5.09765625 frac[20,126] + mem[20,126] = 0
 accuracy_38_89: - 6 frac[38,89] + mem[38,89] = 0
 accuracy_34_25: - 6 frac[34,25] + mem[34,25] = 0
 accuracy_8_28: - 5.09765625 frac[8,28] + mem[8,28] = 0
 accuracy_32_9: - 6 frac[32,9] + mem[32,9] = 0
 accuracy_37_60: - 6 frac[37,60] + mem[37,60] = 0
 accuracy_11_57: - 5.09765625 frac[11,57] + mem[11,57] = 0
 accuracy_38_51: - 6 frac[38,51] + mem[38,51] = 0
 accuracy_1_99: - 5.09765625 frac[1,99] + mem[1,99] = 0
 accuracy_15_23: - 5.09765625 frac[15,23] + mem[15,23] = 0
 accuracy_19_80: - 5.09765625 frac[19,80] + mem[19,80] = 0
 accuracy_35_106: - 6 frac[35,106] + mem[35,106] = 0
 accuracy_32_120: - 6 frac[32,120] + mem[32,120] = 0
 accuracy_33_110: - 6 frac[33,110] + mem[33,110] = 0
 accuracy_14_56: - 5.09765625 frac[14,56] + mem[14,56] = 0
 accuracy_38_21: - 6 frac[38,21] + mem[38,21] = 0
 accuracy_3_117: - 5.09765625 frac[3,117] + mem[3,117] = 0
 accuracy_23_110: - 5.09765625 frac[23,110] + mem[23,110] = 0
 accuracy_17_99: - 5.09765625 frac[17,99] + mem[17,99] = 0
 accuracy_31_119: - 5.09765625 frac[31,119] + mem[31,119] = 0
 accuracy_34_98: - 6 frac[34,98] + mem[34,98] = 0
 accuracy_32_94: - 6 frac[32,94] + mem[32,94] = 0
 accuracy_12_118: - 5.09765625 frac[12,118] + mem[12,118] = 0
 accuracy_38_100: - 6 frac[38,100] + mem[38,100] = 0
 accuracy_4_28: - 5.09765625 frac[4,28] + mem[4,28] = 0
 accuracy_37_74: - 6 frac[37,74] + mem[37,74] = 0
 accuracy_34_92: - 6 frac[34,92] + mem[34,92] = 0
 accuracy_0_84: - 5.09765625 frac[0,84] + mem[0,84] = 0
 accuracy_33_49: - 6 frac[33,49] + mem[33,49] = 0
 accuracy_16_23: - 5.09765625 frac[16,23] + mem[16,23] = 0
 accuracy_6_30: - 5.09765625 frac[6,30] + mem[6,30] = 0
 accuracy_21_30: - 5.09765625 frac[21,30] + mem[21,30] = 0
 accuracy_36_90: - 6 frac[36,90] + mem[36,90] = 0
 accuracy_37_84: - 6 frac[37,84] + mem[37,84] = 0
 accuracy_38_126: - 6 frac[38,126] + mem[38,126] = 0
 accuracy_0_74: - 5.09765625 frac[0,74] + mem[0,74] = 0
 accuracy_13_72: - 5.09765625 frac[13,72] + mem[13,72] = 0
 accuracy_29_46: - 5.09765625 frac[29,46] + mem[29,46] = 0
 accuracy_26_80: - 5.09765625 frac[26,80] + mem[26,80] = 0
 accuracy_9_56: - 5.09765625 frac[9,56] + mem[9,56] = 0
 accuracy_33_43: - 6 frac[33,43] + mem[33,43] = 0
 accuracy_25_118: - 5.09765625 frac[25,118] + mem[25,118] = 0
 accuracy_36_32: - 6 frac[36,32] + mem[36,32] = 0
 accuracy_34_16: - 6 frac[34,16] + mem[34,16] = 0
 accuracy_28_94: - 5.09765625 frac[28,94] + mem[28,94] = 0
 accuracy_11_62: - 5.09765625 frac[11,62] + mem[11,62] = 0
 accuracy_20_99: - 5.09765625 frac[20,99] + mem[20,99] = 0
 accuracy_18_47: - 5.09765625 frac[18,47] + mem[18,47] = 0
 accuracy_38_42: - 6 frac[38,42] + mem[38,42] = 0
 accuracy_36_14: - 6 frac[36,14] + mem[36,14] = 0
 accuracy_1_100: - 5.09765625 frac[1,100] + mem[1,100] = 0
 accuracy_15_28: - 5.09765625 frac[15,28] + mem[15,28] = 0
 accuracy_19_89: - 5.09765625 frac[19,89] + mem[19,89] = 0
 accuracy_35_19: - 6 frac[35,19] + mem[35,19] = 0
 accuracy_33_103: - 6 frac[33,103] + mem[33,103] = 0
 accuracy_5_86: - 5.09765625 frac[5,86] + mem[5,86] = 0
 accuracy_38_12: - 6 frac[38,12] + mem[38,12] = 0
 accuracy_36_20: - 6 frac[36,20] + mem[36,20] = 0
 accuracy_17_100: - 5.09765625 frac[17,100] + mem[17,100] = 0
 accuracy_37_105: - 6 frac[37,105] + mem[37,105] = 0
 accuracy_32_85: - 6 frac[32,85] + mem[32,85] = 0
 accuracy_33_89: - 6 frac[33,89] + mem[33,89] = 0
 accuracy_10_77: - 5.09765625 frac[10,77] + mem[10,77] = 0
 accuracy_17_126: - 5.09765625 frac[17,126] + mem[17,126] = 0
 accuracy_37_115: - 6 frac[37,115] + mem[37,115] = 0
 accuracy_32_75: - 6 frac[32,75] + mem[32,75] = 0
 accuracy_16_14: - 5.09765625 frac[16,14] + mem[16,14] = 0
 accuracy_6_17: - 5.09765625 frac[6,17] + mem[6,17] = 0
 accuracy_38_113: - 6 frac[38,113] + mem[38,113] = 0
 accuracy_36_65: - 6 frac[36,65] + mem[36,65] = 0
 accuracy_37_93: - 6 frac[37,93] + mem[37,93] = 0
 accuracy_0_65: - 5.09765625 frac[0,65] + mem[0,65] = 0
 accuracy_22_20: - 5.09765625 frac[22,20] + mem[22,20] = 0
 accuracy_30_52: - 5.09765625 frac[30,52] + mem[30,52] = 0
 accuracy_36_47: - 6 frac[36,47] + mem[36,47] = 0
 accuracy_8_10: - 5.09765625 frac[8,10] + mem[8,10] = 0
 accuracy_37_46: - 6 frac[37,46] + mem[37,46] = 0
 accuracy_28_85: - 5.09765625 frac[28,85] + mem[28,85] = 0
 accuracy_20_106: - 5.09765625 frac[20,106] + mem[20,106] = 0
 accuracy_9_11: - 5.09765625 frac[9,11] + mem[9,11] = 0
 accuracy_18_38: - 5.09765625 frac[18,38] + mem[18,38] = 0
 accuracy_38_45: - 6 frac[38,45] + mem[38,45] = 0
 accuracy_34_5: - 6 frac[34,5] + mem[34,5] = 0
 accuracy_19_94: - 5.09765625 frac[19,94] + mem[19,94] = 0
 accuracy_24_19: - 5.09765625 frac[24,19] + mem[24,19] = 0
 accuracy_33_120: - 6 frac[33,120] + mem[33,120] = 0
 accuracy_3_99: - 5.09765625 frac[3,99] + mem[3,99] = 0
 accuracy_36_19: - 6 frac[36,19] + mem[36,19] = 0
 accuracy_30_105: - 5.09765625 frac[30,105] + mem[30,105] = 0
 accuracy_31_101: - 5.09765625 frac[31,101] + mem[31,101] = 0
 accuracy_37_18: - 6 frac[37,18] + mem[37,18] = 0
 accuracy_35_126: - 6 frac[35,126] + mem[35,126] = 0
 accuracy_32_108: - 6 frac[32,108] + mem[32,108] = 0
 accuracy_38_25: - 6 frac[38,25] + mem[38,25] = 0
 accuracy_12_100: - 5.09765625 frac[12,100] + mem[12,100] = 0
 accuracy_36_98: - 6 frac[36,98] + mem[36,98] = 0
 accuracy_28_7: - 5.09765625 frac[28,7] + mem[28,7] = 0
 accuracy_34_110: - 6 frac[34,110] + mem[34,110] = 0
 accuracy_35_68: - 6 frac[35,68] + mem[35,68] = 0
 accuracy_32_66: - 6 frac[32,66] + mem[32,66] = 0
 accuracy_10_94: - 5.09765625 frac[10,94] + mem[10,94] = 0
 accuracy_36_72: - 6 frac[36,72] + mem[36,72] = 0
 accuracy_34_72: - 6 frac[34,72] + mem[34,72] = 0
 accuracy_32_56: - 6 frac[32,56] + mem[32,56] = 0
 accuracy_0_88: - 5.09765625 frac[0,88] + mem[0,88] = 0
 accuracy_9_42: - 5.09765625 frac[9,42] + mem[9,42] = 0
 accuracy_30_63: - 5.09765625 frac[30,63] + mem[30,63] = 0
 accuracy_36_86: - 6 frac[36,86] + mem[36,86] = 0
 accuracy_34_34: - 6 frac[34,34] + mem[34,34] = 0
 accuracy_2_66: - 5.09765625 frac[2,66] + mem[2,66] = 0
 accuracy_32_30: - 6 frac[32,30] + mem[32,30] = 0
 accuracy_22_1: - 5.09765625 frac[22,1] + mem[22,1] = 0
 accuracy_35_59: - 6 frac[35,59] + mem[35,59] = 0
 accuracy_18_33: - 5.09765625 frac[18,33] + mem[18,33] = 0
 accuracy_33_31: - 6 frac[33,31] + mem[33,31] = 0
 accuracy_26_76: - 5.09765625 frac[26,76] + mem[26,76] = 0
 accuracy_38_36: - 6 frac[38,36] + mem[38,36] = 0
 accuracy_36_60: - 6 frac[36,60] + mem[36,60] = 0
 accuracy_34_28: - 6 frac[34,28] + mem[34,28] = 0
 accuracy_18_59: - 5.09765625 frac[18,59] + mem[18,59] = 0
 accuracy_5_88: - 5.09765625 frac[5,88] + mem[5,88] = 0
 accuracy_38_62: - 6 frac[38,62] + mem[38,62] = 0
 accuracy_36_26: - 6 frac[36,26] + mem[36,26] = 0
 accuracy_31_98: - 5.09765625 frac[31,98] + mem[31,98] = 0
 accuracy_19_77: - 5.09765625 frac[19,77] + mem[19,77] = 0
 accuracy_32_99: - 6 frac[32,99] + mem[32,99] = 0
 accuracy_33_107: - 6 frac[33,107] + mem[33,107] = 0
 accuracy_14_55: - 5.09765625 frac[14,55] + mem[14,55] = 0
 accuracy_38_16: - 6 frac[38,16] + mem[38,16] = 0
 accuracy_12_99: - 5.09765625 frac[12,99] + mem[12,99] = 0
 accuracy_36_105: - 6 frac[36,105] + mem[36,105] = 0
 accuracy_37_101: - 6 frac[37,101] + mem[37,101] = 0
 accuracy_34_105: - 6 frac[34,105] + mem[34,105] = 0
 accuracy_35_77: - 6 frac[35,77] + mem[35,77] = 0
 accuracy_32_89: - 6 frac[32,89] + mem[32,89] = 0
 accuracy_33_77: - 6 frac[33,77] + mem[33,77] = 0
 accuracy_5_12: - 5.09765625 frac[5,12] + mem[5,12] = 0
 accuracy_12_121: - 5.09765625 frac[12,121] + mem[12,121] = 0
 accuracy_27_51: - 5.09765625 frac[27,51] + mem[27,51] = 0
 accuracy_38_99: - 6 frac[38,99] + mem[38,99] = 0
 accuracy_36_119: - 6 frac[36,119] + mem[36,119] = 0
 accuracy_37_79: - 6 frac[37,79] + mem[37,79] = 0
 accuracy_35_83: - 6 frac[35,83] + mem[35,83] = 0
 accuracy_0_95: - 5.09765625 frac[0,95] + mem[0,95] = 0
 accuracy_33_62: - 6 frac[33,62] + mem[33,62] = 0
 accuracy_16_18: - 5.09765625 frac[16,18] + mem[16,18] = 0
 accuracy_38_69: - 6 frac[38,69] + mem[38,69] = 0
 accuracy_30_38: - 5.09765625 frac[30,38] + mem[30,38] = 0
 accuracy_36_93: - 6 frac[36,93] + mem[36,93] = 0
 accuracy_2_93: - 5.09765625 frac[2,93] + mem[2,93] = 0
 accuracy_32_21: - 6 frac[32,21] + mem[32,21] = 0
 accuracy_22_24: - 5.09765625 frac[22,24] + mem[22,24] = 0
 accuracy_13_77: - 5.09765625 frac[13,77] + mem[13,77] = 0
 accuracy_20_120: - 5.09765625 frac[20,120] + mem[20,120] = 0
 accuracy_35_32: - 6 frac[35,32] + mem[35,32] = 0
 accuracy_38_95: - 6 frac[38,95] + mem[38,95] = 0
 accuracy_36_59: - 6 frac[36,59] + mem[36,59] = 0
 accuracy_34_23: - 6 frac[34,23] + mem[34,23] = 0
 accuracy_8_30: - 5.09765625 frac[8,30] + mem[8,30] = 0
 accuracy_32_11: - 6 frac[32,11] + mem[32,11] = 0
 accuracy_7_62: - 5.09765625 frac[7,62] + mem[7,62] = 0
 accuracy_37_58: - 6 frac[37,58] + mem[37,58] = 0
 accuracy_28_65: - 5.09765625 frac[28,65] + mem[28,65] = 0
 accuracy_33_10: - 6 frac[33,10] + mem[33,10] = 0
 accuracy_5_65: - 5.09765625 frac[5,65] + mem[5,65] = 0
 accuracy_32_122: - 6 frac[32,122] + mem[32,122] = 0
 accuracy_38_11: - 6 frac[38,11] + mem[38,11] = 0
 accuracy_6_107: - 5.09765625 frac[6,107] + mem[6,107] = 0
 accuracy_37_110: - 6 frac[37,110] + mem[37,110] = 0
 accuracy_32_80: - 6 frac[32,80] + mem[32,80] = 0
 accuracy_4_30: - 5.09765625 frac[4,30] + mem[4,30] = 0
 accuracy_37_72: - 6 frac[37,72] + mem[37,72] = 0
 mem_tot_CPU1: - mem[0,90] - mem[0,92] - mem[0,83] - mem[0,75] - mem[0,89]
   - mem[0,80] - mem[0,77] - mem[0,82] - mem[0,72] - mem[0,79] - mem[0,93]
   - mem[0,84] - mem[0,74] - mem[0,65] - mem[0,88] - mem[0,95]
   + mem_tot_CPU1 = 0
 R1229: - 3.05176e-05 mem_tot_CPU1 + normalized_mem_tot_CPU1 = 0
 rows_tot_CPU1: - 3 frac[0,90] - 3 frac[0,92] - 3 frac[0,83] - 3 frac[0,75]
   - 3 frac[0,89] - 3 frac[0,80] - 3 frac[0,77] - 3 frac[0,82]
   - 3 frac[0,72] - 3 frac[0,79] - 3 frac[0,93] - 3 frac[0,84]
   - 3 frac[0,74] - 3 frac[0,65] - 3 frac[0,88] - 3 frac[0,95]
   + rows_tot_CPU1 = 0
 R1231: - 0.0833333333333333 rows_tot_CPU1 + normalized_rows_tot_CPU1 = 0
 mem_tot_CPU2: - mem[1,122] - mem[1,103] - mem[1,120] - mem[1,113]
   - mem[1,107] - mem[1,108] - mem[1,101] - mem[1,119] - mem[1,105]
   - mem[1,98] - mem[1,117] - mem[1,96] - mem[1,114] - mem[1,99]
   - mem[1,100] + mem_tot_CPU2 = 0
 R1233: - 3.05176e-05 mem_tot_CPU2 + normalized_mem_tot_CPU2 = 0
 rows_tot_CPU2: - 3 frac[1,122] - 3 frac[1,103] - 3 frac[1,120]
   - 3 frac[1,113] - 3 frac[1,107] - 3 frac[1,108] - 3 frac[1,101]
   - 3 frac[1,119] - 3 frac[1,105] - 3 frac[1,98] - 3 frac[1,117]
   - 3 frac[1,96] - 3 frac[1,114] - 3 frac[1,99] - 3 frac[1,100]
   + rows_tot_CPU2 = 0
 R1235: - 0.0833333333333333 rows_tot_CPU2 + normalized_rows_tot_CPU2 = 0
 mem_tot_CPU3: - mem[2,76] - mem[2,71] - mem[2,94] - mem[2,80] - mem[2,74]
   - mem[2,92] - mem[2,15] - mem[2,88] - mem[2,68] - mem[2,66] - mem[2,93]
   + mem_tot_CPU3 = 0
 R1237: - 3.05176e-05 mem_tot_CPU3 + normalized_mem_tot_CPU3 = 0
 rows_tot_CPU3: - 3 frac[2,76] - 3 frac[2,71] - 3 frac[2,94] - 3 frac[2,80]
   - 3 frac[2,74] - 3 frac[2,92] - 3 frac[2,15] - 3 frac[2,88]
   - 3 frac[2,68] - 3 frac[2,66] - 3 frac[2,93] + rows_tot_CPU3 = 0
 R1239: - 0.0833333333333333 rows_tot_CPU3 + normalized_rows_tot_CPU3 = 0
 mem_tot_CPU4: - mem[3,103] - mem[3,114] - mem[3,96] - mem[3,105]
   - mem[3,110] - mem[3,32] - mem[3,98] - mem[3,100] - mem[3,115]
   - mem[3,120] - mem[3,102] - mem[3,117] - mem[3,99] + mem_tot_CPU4 = 0
 R1241: - 3.05176e-05 mem_tot_CPU4 + normalized_mem_tot_CPU4 = 0
 rows_tot_CPU4: - 3 frac[3,103] - 3 frac[3,114] - 3 frac[3,96]
   - 3 frac[3,105] - 3 frac[3,110] - 3 frac[3,32] - 3 frac[3,98]
   - 3 frac[3,100] - 3 frac[3,115] - 3 frac[3,120] - 3 frac[3,102]
   - 3 frac[3,117] - 3 frac[3,99] + rows_tot_CPU4 = 0
 R1243: - 0.0833333333333333 rows_tot_CPU4 + normalized_rows_tot_CPU4 = 0
 mem_tot_CPU5: - mem[4,5] - mem[4,11] - mem[4,17] - mem[4,121] - mem[4,22]
   - mem[4,3] - mem[4,16] - mem[4,18] - mem[4,25] - mem[4,28] - mem[4,30]
   + mem_tot_CPU5 = 0
 R1245: - 3.05176e-05 mem_tot_CPU5 + normalized_mem_tot_CPU5 = 0
 rows_tot_CPU5: - 3 frac[4,5] - 3 frac[4,11] - 3 frac[4,17] - 3 frac[4,121]
   - 3 frac[4,22] - 3 frac[4,3] - 3 frac[4,16] - 3 frac[4,18]
   - 3 frac[4,25] - 3 frac[4,28] - 3 frac[4,30] + rows_tot_CPU5 = 0
 R1247: - 0.0833333333333333 rows_tot_CPU5 + normalized_rows_tot_CPU5 = 0
 mem_tot_CPU6: - mem[5,93] - mem[5,72] - mem[5,68] - mem[5,77] - mem[5,80]
   - mem[5,89] - mem[5,85] - mem[5,71] - mem[5,64] - mem[5,90] - mem[5,86]
   - mem[5,88] - mem[5,12] - mem[5,65] + mem_tot_CPU6 = 0
 R1249: - 3.05176e-05 mem_tot_CPU6 + normalized_mem_tot_CPU6 = 0
 rows_tot_CPU6: - 3 frac[5,93] - 3 frac[5,72] - 3 frac[5,68] - 3 frac[5,77]
   - 3 frac[5,80] - 3 frac[5,89] - 3 frac[5,85] - 3 frac[5,71]
   - 3 frac[5,64] - 3 frac[5,90] - 3 frac[5,86] - 3 frac[5,88]
   - 3 frac[5,12] - 3 frac[5,65] + rows_tot_CPU6 = 0
 R1251: - 0.0833333333333333 rows_tot_CPU6 + normalized_rows_tot_CPU6 = 0
 mem_tot_CPU7: - mem[6,28] - mem[6,14] - mem[6,18] - mem[6,21] - mem[6,11]
   - mem[6,5] - mem[6,22] - mem[6,9] - mem[6,29] - mem[6,20] - mem[6,24]
   - mem[6,10] - mem[6,39] - mem[6,30] - mem[6,17] - mem[6,107]
   + mem_tot_CPU7 = 0
 R1253: - 3.05176e-05 mem_tot_CPU7 + normalized_mem_tot_CPU7 = 0
 rows_tot_CPU7: - 3 frac[6,28] - 3 frac[6,14] - 3 frac[6,18] - 3 frac[6,21]
   - 3 frac[6,11] - 3 frac[6,5] - 3 frac[6,22] - 3 frac[6,9] - 3 frac[6,29]
   - 3 frac[6,20] - 3 frac[6,24] - 3 frac[6,10] - 3 frac[6,39]
   - 3 frac[6,30] - 3 frac[6,17] - 3 frac[6,107] + rows_tot_CPU7 = 0
 R1255: - 0.0833333333333333 rows_tot_CPU7 + normalized_rows_tot_CPU7 = 0
 mem_tot_CPU8: - mem[7,55] - mem[7,91] - mem[7,34] - mem[7,56] - mem[7,49]
   - mem[7,39] - mem[7,61] - mem[7,45] - mem[7,88] - mem[7,54] - mem[7,47]
   - mem[7,48] - mem[7,62] + mem_tot_CPU8 = 0
 R1257: - 3.05176e-05 mem_tot_CPU8 + normalized_mem_tot_CPU8 = 0
 rows_tot_CPU8: - 3 frac[7,55] - 3 frac[7,91] - 3 frac[7,34] - 3 frac[7,56]
   - 3 frac[7,49] - 3 frac[7,39] - 3 frac[7,61] - 3 frac[7,45]
   - 3 frac[7,88] - 3 frac[7,54] - 3 frac[7,47] - 3 frac[7,48]
   - 3 frac[7,62] + rows_tot_CPU8 = 0
 R1259: - 0.0833333333333333 rows_tot_CPU8 + normalized_rows_tot_CPU8 = 0
 mem_tot_CPU9: - mem[8,38] - mem[8,18] - mem[8,14] - mem[8,20] - mem[8,27]
   - mem[8,22] - mem[8,29] - mem[8,17] - mem[8,24] - mem[8,31] - mem[8,19]
   - mem[8,26] - mem[8,21] - mem[8,28] - mem[8,10] - mem[8,30]
   + mem_tot_CPU9 = 0
 R1261: - 3.05176e-05 mem_tot_CPU9 + normalized_mem_tot_CPU9 = 0
 rows_tot_CPU9: - 3 frac[8,38] - 3 frac[8,18] - 3 frac[8,14] - 3 frac[8,20]
   - 3 frac[8,27] - 3 frac[8,22] - 3 frac[8,29] - 3 frac[8,17]
   - 3 frac[8,24] - 3 frac[8,31] - 3 frac[8,19] - 3 frac[8,26]
   - 3 frac[8,21] - 3 frac[8,28] - 3 frac[8,10] - 3 frac[8,30]
   + rows_tot_CPU9 = 0
 R1263: - 0.0833333333333333 rows_tot_CPU9 + normalized_rows_tot_CPU9 = 0
 mem_tot_CPU10: - mem[9,62] - mem[9,39] - mem[9,59] - mem[9,60] - mem[9,48]
   - mem[9,57] - mem[9,52] - mem[9,61] - mem[9,47] - mem[9,49] - mem[9,58]
   - mem[9,35] - mem[9,36] - mem[9,45] - mem[9,56] - mem[9,11] - mem[9,42]
   + mem_tot_CPU10 = 0
 R1265: - 3.05176e-05 mem_tot_CPU10 + normalized_mem_tot_CPU10 = 0
 rows_tot_CPU10: - 3 frac[9,62] - 3 frac[9,39] - 3 frac[9,59]
   - 3 frac[9,60] - 3 frac[9,48] - 3 frac[9,57] - 3 frac[9,52]
   - 3 frac[9,61] - 3 frac[9,47] - 3 frac[9,49] - 3 frac[9,58]
   - 3 frac[9,35] - 3 frac[9,36] - 3 frac[9,45] - 3 frac[9,56]
   - 3 frac[9,11] - 3 frac[9,42] + rows_tot_CPU10 = 0
 R1267: - 0.0833333333333333 rows_tot_CPU10 + normalized_rows_tot_CPU10
   = 0
 mem_tot_CPU11: - mem[10,75] - mem[10,73] - mem[10,69] - mem[10,76]
   - mem[10,81] - mem[10,74] - mem[10,95] - mem[10,65] - mem[10,72]
   - mem[10,93] - mem[10,101] - mem[10,77] - mem[10,94] + mem_tot_CPU11
   = 0
 R1269: - 3.05176e-05 mem_tot_CPU11 + normalized_mem_tot_CPU11 = 0
 rows_tot_CPU11: - 3 frac[10,75] - 3 frac[10,73] - 3 frac[10,69]
   - 3 frac[10,76] - 3 frac[10,81] - 3 frac[10,74] - 3 frac[10,95]
   - 3 frac[10,65] - 3 frac[10,72] - 3 frac[10,93] - 3 frac[10,101]
   - 3 frac[10,77] - 3 frac[10,94] + rows_tot_CPU11 = 0
 R1271: - 0.0833333333333333 rows_tot_CPU11 + normalized_rows_tot_CPU11
   = 0
 mem_tot_CPU12: - mem[11,46] - mem[11,45] - mem[11,34] - mem[11,56]
   - mem[11,47] - mem[11,36] - mem[11,51] - mem[11,33] - mem[11,42]
   - mem[11,35] - mem[11,57] - mem[11,62] + mem_tot_CPU12 = 0
 R1273: - 3.05176e-05 mem_tot_CPU12 + normalized_mem_tot_CPU12 = 0
 rows_tot_CPU12: - 3 frac[11,46] - 3 frac[11,45] - 3 frac[11,34]
   - 3 frac[11,56] - 3 frac[11,47] - 3 frac[11,36] - 3 frac[11,51]
   - 3 frac[11,33] - 3 frac[11,42] - 3 frac[11,35] - 3 frac[11,57]
   - 3 frac[11,62] + rows_tot_CPU12 = 0
 R1275: - 0.0833333333333333 rows_tot_CPU12 + normalized_rows_tot_CPU12
   = 0
 mem_tot_CPU13: - mem[12,59] - mem[12,111] - mem[12,102] - mem[12,105]
   - mem[12,126] - mem[12,117] - mem[12,107] - mem[12,120] - mem[12,110]
   - mem[12,103] - mem[12,106] - mem[12,118] - mem[12,100] - mem[12,99]
   - mem[12,121] + mem_tot_CPU13 = 0
 R1277: - 3.05176e-05 mem_tot_CPU13 + normalized_mem_tot_CPU13 = 0
 rows_tot_CPU13: - 3 frac[12,59] - 3 frac[12,111] - 3 frac[12,102]
   - 3 frac[12,105] - 3 frac[12,126] - 3 frac[12,117] - 3 frac[12,107]
   - 3 frac[12,120] - 3 frac[12,110] - 3 frac[12,103] - 3 frac[12,106]
   - 3 frac[12,118] - 3 frac[12,100] - 3 frac[12,99] - 3 frac[12,121]
   + rows_tot_CPU13 = 0
 R1279: - 0.0833333333333333 rows_tot_CPU13 + normalized_rows_tot_CPU13
   = 0
 mem_tot_CPU14: - mem[13,89] - mem[13,68] - mem[13,73] - mem[13,66]
   - mem[13,92] - mem[13,90] - mem[13,88] - mem[13,5] - mem[13,93]
   - mem[13,72] - mem[13,77] + mem_tot_CPU14 = 0
 R1281: - 3.05176e-05 mem_tot_CPU14 + normalized_mem_tot_CPU14 = 0
 rows_tot_CPU14: - 3 frac[13,89] - 3 frac[13,68] - 3 frac[13,73]
   - 3 frac[13,66] - 3 frac[13,92] - 3 frac[13,90] - 3 frac[13,88]
   - 3 frac[13,5] - 3 frac[13,93] - 3 frac[13,72] - 3 frac[13,77]
   + rows_tot_CPU14 = 0
 R1283: - 0.0833333333333333 rows_tot_CPU14 + normalized_rows_tot_CPU14
   = 0
 mem_tot_CPU15: - mem[14,43] - mem[14,59] - mem[14,36] - mem[14,54]
   - mem[14,42] - mem[14,52] - mem[14,33] - mem[14,39] - mem[14,46]
   - mem[14,122] - mem[14,56] - mem[14,55] + mem_tot_CPU15 = 0
 R1285: - 3.05176e-05 mem_tot_CPU15 + normalized_mem_tot_CPU15 = 0
 rows_tot_CPU15: - 3 frac[14,43] - 3 frac[14,59] - 3 frac[14,36]
   - 3 frac[14,54] - 3 frac[14,42] - 3 frac[14,52] - 3 frac[14,33]
   - 3 frac[14,39] - 3 frac[14,46] - 3 frac[14,122] - 3 frac[14,56]
   - 3 frac[14,55] + rows_tot_CPU15 = 0
 R1287: - 0.0833333333333333 rows_tot_CPU15 + normalized_rows_tot_CPU15
   = 0
 mem_tot_CPU16: - mem[15,30] - mem[15,34] - mem[15,15] - mem[15,20]
   - mem[15,26] - mem[15,9] - mem[15,22] - mem[15,31] - mem[15,5]
   - mem[15,11] - mem[15,25] - mem[15,12] - mem[15,18] - mem[15,27]
   - mem[15,14] - mem[15,23] - mem[15,28] + mem_tot_CPU16 = 0
 R1289: - 3.05176e-05 mem_tot_CPU16 + normalized_mem_tot_CPU16 = 0
 rows_tot_CPU16: - 3 frac[15,30] - 3 frac[15,34] - 3 frac[15,15]
   - 3 frac[15,20] - 3 frac[15,26] - 3 frac[15,9] - 3 frac[15,22]
   - 3 frac[15,31] - 3 frac[15,5] - 3 frac[15,11] - 3 frac[15,25]
   - 3 frac[15,12] - 3 frac[15,18] - 3 frac[15,27] - 3 frac[15,14]
   - 3 frac[15,23] - 3 frac[15,28] + rows_tot_CPU16 = 0
 R1291: - 0.0833333333333333 rows_tot_CPU16 + normalized_rows_tot_CPU16
   = 0
 mem_tot_CPU17: - mem[16,29] - mem[16,126] - mem[16,2] - mem[16,24]
   - mem[16,72] - mem[16,26] - mem[16,28] - mem[16,1] - mem[16,30]
   - mem[16,52] - mem[16,25] - mem[16,23] - mem[16,14] - mem[16,18]
   + mem_tot_CPU17 = 0
 R1293: - 3.05176e-05 mem_tot_CPU17 + normalized_mem_tot_CPU17 = 0
 rows_tot_CPU17: - 3 frac[16,29] - 3 frac[16,126] - 3 frac[16,2]
   - 3 frac[16,24] - 3 frac[16,72] - 3 frac[16,26] - 3 frac[16,28]
   - 3 frac[16,1] - 3 frac[16,30] - 3 frac[16,52] - 3 frac[16,25]
   - 3 frac[16,23] - 3 frac[16,14] - 3 frac[16,18] + rows_tot_CPU17 = 0
 R1295: - 0.0833333333333333 rows_tot_CPU17 + normalized_rows_tot_CPU17
   = 0
 mem_tot_CPU18: - mem[17,117] - mem[17,103] - mem[17,121] - mem[17,101]
   - mem[17,105] - mem[17,98] - mem[17,118] - mem[17,122] - mem[17,115]
   - mem[17,102] - mem[17,120] - mem[17,99] - mem[17,100] - mem[17,126]
   + mem_tot_CPU18 = 0
 R1297: - 3.05176e-05 mem_tot_CPU18 + normalized_mem_tot_CPU18 = 0
 rows_tot_CPU18: - 3 frac[17,117] - 3 frac[17,103] - 3 frac[17,121]
   - 3 frac[17,101] - 3 frac[17,105] - 3 frac[17,98] - 3 frac[17,118]
   - 3 frac[17,122] - 3 frac[17,115] - 3 frac[17,102] - 3 frac[17,120]
   - 3 frac[17,99] - 3 frac[17,100] - 3 frac[17,126] + rows_tot_CPU18 = 0
 R1299: - 0.0833333333333333 rows_tot_CPU18 + normalized_rows_tot_CPU18
   = 0
 mem_tot_CPU19: - mem[18,62] - mem[18,48] - mem[18,34] - mem[18,82]
   - mem[18,58] - mem[18,51] - mem[18,42] - mem[18,47] - mem[18,38]
   - mem[18,33] - mem[18,59] + mem_tot_CPU19 = 0
 R1301: - 3.05176e-05 mem_tot_CPU19 + normalized_mem_tot_CPU19 = 0
 rows_tot_CPU19: - 3 frac[18,62] - 3 frac[18,48] - 3 frac[18,34]
   - 3 frac[18,82] - 3 frac[18,58] - 3 frac[18,51] - 3 frac[18,42]
   - 3 frac[18,47] - 3 frac[18,38] - 3 frac[18,33] - 3 frac[18,59]
   + rows_tot_CPU19 = 0
 R1303: - 0.0833333333333333 rows_tot_CPU19 + normalized_rows_tot_CPU19
   = 0
 mem_tot_CPU20: - mem[19,84] - mem[19,72] - mem[19,81] - mem[19,86]
   - mem[19,95] - mem[19,69] - mem[19,126] - mem[19,74] - mem[19,88]
   - mem[19,73] - mem[19,92] - mem[19,66] - mem[19,75] - mem[19,80]
   - mem[19,89] - mem[19,94] - mem[19,77] + mem_tot_CPU20 = 0
 R1305: - 3.05176e-05 mem_tot_CPU20 + normalized_mem_tot_CPU20 = 0
 rows_tot_CPU20: - 3 frac[19,84] - 3 frac[19,72] - 3 frac[19,81]
   - 3 frac[19,86] - 3 frac[19,95] - 3 frac[19,69] - 3 frac[19,126]
   - 3 frac[19,74] - 3 frac[19,88] - 3 frac[19,73] - 3 frac[19,92]
   - 3 frac[19,66] - 3 frac[19,75] - 3 frac[19,80] - 3 frac[19,89]
   - 3 frac[19,94] - 3 frac[19,77] + rows_tot_CPU20 = 0
 R1307: - 0.0833333333333333 rows_tot_CPU20 + normalized_rows_tot_CPU20
   = 0
 mem_tot_CPU21: - mem[20,118] - mem[20,23] - mem[20,105] - mem[20,108]
   - mem[20,114] - mem[20,110] - mem[20,117] - mem[20,97] - mem[20,126]
   - mem[20,99] - mem[20,106] - mem[20,120] + mem_tot_CPU21 = 0
 R1309: - 3.05176e-05 mem_tot_CPU21 + normalized_mem_tot_CPU21 = 0
 rows_tot_CPU21: - 3 frac[20,118] - 3 frac[20,23] - 3 frac[20,105]
   - 3 frac[20,108] - 3 frac[20,114] - 3 frac[20,110] - 3 frac[20,117]
   - 3 frac[20,97] - 3 frac[20,126] - 3 frac[20,99] - 3 frac[20,106]
   - 3 frac[20,120] + rows_tot_CPU21 = 0
 R1311: - 0.0833333333333333 rows_tot_CPU21 + normalized_rows_tot_CPU21
   = 0
 mem_tot_CPU22: - mem[21,26] - mem[21,22] - mem[21,11] - mem[21,20]
   - mem[21,29] - mem[21,7] - mem[21,14] - mem[21,23] - mem[21,16]
   - mem[21,12] - mem[21,21] - mem[21,30] + mem_tot_CPU22 = 0
 R1313: - 3.05176e-05 mem_tot_CPU22 + normalized_mem_tot_CPU22 = 0
 rows_tot_CPU22: - 3 frac[21,26] - 3 frac[21,22] - 3 frac[21,11]
   - 3 frac[21,20] - 3 frac[21,29] - 3 frac[21,7] - 3 frac[21,14]
   - 3 frac[21,23] - 3 frac[21,16] - 3 frac[21,12] - 3 frac[21,21]
   - 3 frac[21,30] + rows_tot_CPU22 = 0
 R1315: - 0.0833333333333333 rows_tot_CPU22 + normalized_rows_tot_CPU22
   = 0
 mem_tot_CPU23: - mem[22,12] - mem[22,30] - mem[22,17] - mem[22,11]
   - mem[22,5] - mem[22,28] - mem[22,9] - mem[22,25] - mem[22,62]
   - mem[22,16] - mem[22,3] - mem[22,20] - mem[22,1] - mem[22,24]
   + mem_tot_CPU23 = 0
 R1317: - 3.05176e-05 mem_tot_CPU23 + normalized_mem_tot_CPU23 = 0
 rows_tot_CPU23: - 3 frac[22,12] - 3 frac[22,30] - 3 frac[22,17]
   - 3 frac[22,11] - 3 frac[22,5] - 3 frac[22,28] - 3 frac[22,9]
   - 3 frac[22,25] - 3 frac[22,62] - 3 frac[22,16] - 3 frac[22,3]
   - 3 frac[22,20] - 3 frac[22,1] - 3 frac[22,24] + rows_tot_CPU23 = 0
 R1319: - 0.0833333333333333 rows_tot_CPU23 + normalized_rows_tot_CPU23
   = 0
 mem_tot_CPU24: - mem[23,120] - mem[23,113] - mem[23,100] - mem[23,107]
   - mem[23,121] - mem[23,98] - mem[23,110] + mem_tot_CPU24 = 0
 R1321: - 3.05176e-05 mem_tot_CPU24 + normalized_mem_tot_CPU24 = 0
 rows_tot_CPU24: - 3 frac[23,120] - 3 frac[23,113] - 3 frac[23,100]
   - 3 frac[23,107] - 3 frac[23,121] - 3 frac[23,98] - 3 frac[23,110]
   + rows_tot_CPU24 = 0
 R1323: - 0.0833333333333333 rows_tot_CPU24 + normalized_rows_tot_CPU24
   = 0
 mem_tot_CPU25: - mem[24,92] - mem[24,14] - mem[24,21] - mem[24,2]
   - mem[24,16] - mem[24,11] - mem[24,18] - mem[24,25] - mem[24,22]
   - mem[24,29] - mem[24,17] - mem[24,19] + mem_tot_CPU25 = 0
 R1325: - 3.05176e-05 mem_tot_CPU25 + normalized_mem_tot_CPU25 = 0
 rows_tot_CPU25: - 3 frac[24,92] - 3 frac[24,14] - 3 frac[24,21]
   - 3 frac[24,2] - 3 frac[24,16] - 3 frac[24,11] - 3 frac[24,18]
   - 3 frac[24,25] - 3 frac[24,22] - 3 frac[24,29] - 3 frac[24,17]
   - 3 frac[24,19] + rows_tot_CPU25 = 0
 R1327: - 0.0833333333333333 rows_tot_CPU25 + normalized_rows_tot_CPU25
   = 0
 mem_tot_CPU26: - mem[25,77] - mem[25,107] - mem[25,117] - mem[25,123]
   - mem[25,102] - mem[25,118] + mem_tot_CPU26 = 0
 R1329: - 3.05176e-05 mem_tot_CPU26 + normalized_mem_tot_CPU26 = 0
 rows_tot_CPU26: - 3 frac[25,77] - 3 frac[25,107] - 3 frac[25,117]
   - 3 frac[25,123] - 3 frac[25,102] - 3 frac[25,118] + rows_tot_CPU26 = 0
 R1331: - 0.0833333333333333 rows_tot_CPU26 + normalized_rows_tot_CPU26
   = 0
 mem_tot_CPU27: - mem[26,92] - mem[26,72] - mem[26,68] - mem[26,123]
   - mem[26,88] - mem[26,66] - mem[26,77] - mem[26,17] - mem[26,73]
   - mem[26,80] - mem[26,76] + mem_tot_CPU27 = 0
 R1333: - 3.05176e-05 mem_tot_CPU27 + normalized_mem_tot_CPU27 = 0
 rows_tot_CPU27: - 3 frac[26,92] - 3 frac[26,72] - 3 frac[26,68]
   - 3 frac[26,123] - 3 frac[26,88] - 3 frac[26,66] - 3 frac[26,77]
   - 3 frac[26,17] - 3 frac[26,73] - 3 frac[26,80] - 3 frac[26,76]
   + rows_tot_CPU27 = 0
 R1335: - 0.0833333333333333 rows_tot_CPU27 + normalized_rows_tot_CPU27
   = 0
 mem_tot_CPU28: - mem[27,60] - mem[27,43] - mem[27,45] - mem[27,34]
   - mem[27,59] - mem[27,47] - mem[27,36] - mem[27,40] - mem[27,35]
   - mem[27,49] - mem[27,51] + mem_tot_CPU28 = 0
 R1337: - 3.05176e-05 mem_tot_CPU28 + normalized_mem_tot_CPU28 = 0
 rows_tot_CPU28: - 3 frac[27,60] - 3 frac[27,43] - 3 frac[27,45]
   - 3 frac[27,34] - 3 frac[27,59] - 3 frac[27,47] - 3 frac[27,36]
   - 3 frac[27,40] - 3 frac[27,35] - 3 frac[27,49] - 3 frac[27,51]
   + rows_tot_CPU28 = 0
 R1339: - 0.0833333333333333 rows_tot_CPU28 + normalized_rows_tot_CPU28
   = 0
 mem_tot_CPU29: - mem[28,77] - mem[28,107] - mem[28,79] - mem[28,93]
   - mem[28,83] - mem[28,64] - mem[28,76] - mem[28,75] - mem[28,66]
   - mem[28,91] - mem[28,82] - mem[28,94] - mem[28,85] - mem[28,7]
   - mem[28,65] + mem_tot_CPU29 = 0
 R1341: - 3.05176e-05 mem_tot_CPU29 + normalized_mem_tot_CPU29 = 0
 rows_tot_CPU29: - 3 frac[28,77] - 3 frac[28,107] - 3 frac[28,79]
   - 3 frac[28,93] - 3 frac[28,83] - 3 frac[28,64] - 3 frac[28,76]
   - 3 frac[28,75] - 3 frac[28,66] - 3 frac[28,91] - 3 frac[28,82]
   - 3 frac[28,94] - 3 frac[28,85] - 3 frac[28,7] - 3 frac[28,65]
   + rows_tot_CPU29 = 0
 R1343: - 0.0833333333333333 rows_tot_CPU29 + normalized_rows_tot_CPU29
   = 0
 mem_tot_CPU30: - mem[29,48] - mem[29,42] - mem[29,35] - mem[29,40]
   - mem[29,33] - mem[29,51] - mem[29,38] - mem[29,63] - mem[29,56]
   - mem[29,59] - mem[29,46] + mem_tot_CPU30 = 0
 R1345: - 3.05176e-05 mem_tot_CPU30 + normalized_mem_tot_CPU30 = 0
 rows_tot_CPU30: - 3 frac[29,48] - 3 frac[29,42] - 3 frac[29,35]
   - 3 frac[29,40] - 3 frac[29,33] - 3 frac[29,51] - 3 frac[29,38]
   - 3 frac[29,63] - 3 frac[29,56] - 3 frac[29,59] - 3 frac[29,46]
   + rows_tot_CPU30 = 0
 R1347: - 0.0833333333333333 rows_tot_CPU30 + normalized_rows_tot_CPU30
   = 0
 mem_tot_CPU31: - mem[30,51] - mem[30,36] - mem[30,89] - mem[30,49]
   - mem[30,56] - mem[30,62] - mem[30,46] - mem[30,59] - mem[30,34]
   - mem[30,32] - mem[30,52] - mem[30,105] - mem[30,63] - mem[30,38]
   + mem_tot_CPU31 = 0
 R1349: - 3.05176e-05 mem_tot_CPU31 + normalized_mem_tot_CPU31 = 0
 rows_tot_CPU31: - 3 frac[30,51] - 3 frac[30,36] - 3 frac[30,89]
   - 3 frac[30,49] - 3 frac[30,56] - 3 frac[30,62] - 3 frac[30,46]
   - 3 frac[30,59] - 3 frac[30,34] - 3 frac[30,32] - 3 frac[30,52]
   - 3 frac[30,105] - 3 frac[30,63] - 3 frac[30,38] + rows_tot_CPU31 = 0
 R1351: - 0.0833333333333333 rows_tot_CPU31 + normalized_rows_tot_CPU31
   = 0
 mem_tot_CPU32: - mem[31,126] - mem[31,120] - mem[31,97] - mem[31,110]
   - mem[31,122] - mem[31,118] - mem[31,106] - mem[31,47] - mem[31,111]
   - mem[31,117] - mem[31,119] - mem[31,101] - mem[31,98] + mem_tot_CPU32
   = 0
 R1353: - 3.05176e-05 mem_tot_CPU32 + normalized_mem_tot_CPU32 = 0
 rows_tot_CPU32: - 3 frac[31,126] - 3 frac[31,120] - 3 frac[31,97]
   - 3 frac[31,110] - 3 frac[31,122] - 3 frac[31,118] - 3 frac[31,106]
   - 3 frac[31,47] - 3 frac[31,111] - 3 frac[31,117] - 3 frac[31,119]
   - 3 frac[31,101] - 3 frac[31,98] + rows_tot_CPU32 = 0
 R1355: - 0.0833333333333333 rows_tot_CPU32 + normalized_rows_tot_CPU32
   = 0
 mem_tot_tor_P41: - mem[32,54] - mem[32,113] - mem[32,77] - mem[32,25]
   - mem[32,110] - mem[32,68] - mem[32,16] - mem[32,101] - mem[32,91]
   - mem[32,49] - mem[32,82] - mem[32,72] - mem[32,115] - mem[32,105]
   - mem[32,79] - mem[32,96] - mem[32,18] - mem[32,103] - mem[32,93]
   - mem[32,15] - mem[32,84] - mem[32,74] - mem[32,32] - mem[32,117]
   - mem[32,107] - mem[32,65] - mem[32,39] - mem[32,29] - mem[32,98]
   - mem[32,88] - mem[32,62] - mem[32,20] - mem[32,10] - mem[32,121]
   - mem[32,95] - mem[32,86] - mem[32,76] - mem[32,34] - mem[32,24]
   - mem[32,119] - mem[32,100] - mem[32,90] - mem[32,48] - mem[32,22]
   - mem[32,12] - mem[32,55] - mem[32,45] - mem[32,3] - mem[32,114]
   - mem[32,17] - mem[32,102] - mem[32,92] - mem[32,14] - mem[32,83]
   - mem[32,47] - mem[32,5] - mem[32,64] - mem[32,28] - mem[32,71]
   - mem[32,61] - mem[32,9] - mem[32,120] - mem[32,94] - mem[32,85]
   - mem[32,75] - mem[32,108] - mem[32,66] - mem[32,56] - mem[32,30]
   - mem[32,99] - mem[32,89] - mem[32,21] - mem[32,11] - mem[32,122]
   - mem[32,80] + mem_tot_tor_P41 = 0
 R1357: - 1.6276041666666666e-04 mem_tot_tor_P41
   + normalized_mem_tot_tor_P41 = 0
 rows_tot_tor_P41: - 3 frac[32,54] - 3 frac[32,113] - 3 frac[32,77]
   - 3 frac[32,25] - 3 frac[32,110] - 3 frac[32,68] - 3 frac[32,16]
   - 3 frac[32,101] - 3 frac[32,91] - 3 frac[32,49] - 3 frac[32,82]
   - 3 frac[32,72] - 3 frac[32,115] - 3 frac[32,105] - 3 frac[32,79]
   - 3 frac[32,96] - 3 frac[32,18] - 3 frac[32,103] - 3 frac[32,93]
   - 3 frac[32,15] - 3 frac[32,84] - 3 frac[32,74] - 3 frac[32,32]
   - 3 frac[32,117] - 3 frac[32,107] - 3 frac[32,65] - 3 frac[32,39]
   - 3 frac[32,29] - 3 frac[32,98] - 3 frac[32,88] - 3 frac[32,62]
   - 3 frac[32,20] - 3 frac[32,10] - 3 frac[32,121] - 3 frac[32,95]
   - 3 frac[32,86] - 3 frac[32,76] - 3 frac[32,34] - 3 frac[32,24]
   - 3 frac[32,119] - 3 frac[32,100] - 3 frac[32,90] - 3 frac[32,48]
   - 3 frac[32,22] - 3 frac[32,12] - 3 frac[32,55] - 3 frac[32,45]
   - 3 frac[32,3] - 3 frac[32,114] - 3 frac[32,17] - 3 frac[32,102]
   - 3 frac[32,92] - 3 frac[32,14] - 3 frac[32,83] - 3 frac[32,47]
   - 3 frac[32,5] - 3 frac[32,64] - 3 frac[32,28] - 3 frac[32,71]
   - 3 frac[32,61] - 3 frac[32,9] - 3 frac[32,120] - 3 frac[32,94]
   - 3 frac[32,85] - 3 frac[32,75] - 3 frac[32,108] - 3 frac[32,66]
   - 3 frac[32,56] - 3 frac[32,30] - 3 frac[32,99] - 3 frac[32,89]
   - 3 frac[32,21] - 3 frac[32,11] - 3 frac[32,122] - 3 frac[32,80]
   + rows_tot_tor_P41 = 0
 R1359: - 0.0208333333333333 rows_tot_tor_P41 + normalized_rows_tot_tor_P41
   = 0
 mem_tot_tor_P42: - mem[33,55] - mem[33,101] - mem[33,95] - mem[33,34]
   - mem[33,126] - mem[33,74] - mem[33,59] - mem[33,29] - mem[33,105]
   - mem[33,60] - mem[33,22] - mem[33,68] - mem[33,47] - mem[33,93]
   - mem[33,26] - mem[33,72] - mem[33,57] - mem[33,19] - mem[33,117]
   - mem[33,111] - mem[33,65] - mem[33,20] - mem[33,14] - mem[33,90]
   - mem[33,45] - mem[33,121] - mem[33,38] - mem[33,24] - mem[33,17]
   - mem[33,11] - mem[33,48] - mem[33,42] - mem[33,12] - mem[33,102]
   - mem[33,88] - mem[33,35] - mem[33,5] - mem[33,81] - mem[33,75]
   - mem[33,36] - mem[33,30] - mem[33,106] - mem[33,76] - mem[33,61]
   - mem[33,23] - mem[33,9] - mem[33,99] - mem[33,69] - mem[33,54]
   - mem[33,100] - mem[33,94] - mem[33,33] - mem[33,27] - mem[33,73]
   - mem[33,58] - mem[33,28] - mem[33,118] - mem[33,66] - mem[33,51]
   - mem[33,21] - mem[33,15] - mem[33,52] - mem[33,46] - mem[33,122]
   - mem[33,92] - mem[33,39] - mem[33,25] - mem[33,56] - mem[33,18]
   - mem[33,110] - mem[33,49] - mem[33,43] - mem[33,103] - mem[33,89]
   - mem[33,120] - mem[33,31] - mem[33,107] - mem[33,77] - mem[33,62]
   - mem[33,10] + mem_tot_tor_P42 = 0
 R1361: - 1.6276041666666666e-04 mem_tot_tor_P42
   + normalized_mem_tot_tor_P42 = 0
 rows_tot_tor_P42: - 3 frac[33,55] - 3 frac[33,101] - 3 frac[33,95]
   - 3 frac[33,34] - 3 frac[33,126] - 3 frac[33,74] - 3 frac[33,59]
   - 3 frac[33,29] - 3 frac[33,105] - 3 frac[33,60] - 3 frac[33,22]
   - 3 frac[33,68] - 3 frac[33,47] - 3 frac[33,93] - 3 frac[33,26]
   - 3 frac[33,72] - 3 frac[33,57] - 3 frac[33,19] - 3 frac[33,117]
   - 3 frac[33,111] - 3 frac[33,65] - 3 frac[33,20] - 3 frac[33,14]
   - 3 frac[33,90] - 3 frac[33,45] - 3 frac[33,121] - 3 frac[33,38]
   - 3 frac[33,24] - 3 frac[33,17] - 3 frac[33,11] - 3 frac[33,48]
   - 3 frac[33,42] - 3 frac[33,12] - 3 frac[33,102] - 3 frac[33,88]
   - 3 frac[33,35] - 3 frac[33,5] - 3 frac[33,81] - 3 frac[33,75]
   - 3 frac[33,36] - 3 frac[33,30] - 3 frac[33,106] - 3 frac[33,76]
   - 3 frac[33,61] - 3 frac[33,23] - 3 frac[33,9] - 3 frac[33,99]
   - 3 frac[33,69] - 3 frac[33,54] - 3 frac[33,100] - 3 frac[33,94]
   - 3 frac[33,33] - 3 frac[33,27] - 3 frac[33,73] - 3 frac[33,58]
   - 3 frac[33,28] - 3 frac[33,118] - 3 frac[33,66] - 3 frac[33,51]
   - 3 frac[33,21] - 3 frac[33,15] - 3 frac[33,52] - 3 frac[33,46]
   - 3 frac[33,122] - 3 frac[33,92] - 3 frac[33,39] - 3 frac[33,25]
   - 3 frac[33,56] - 3 frac[33,18] - 3 frac[33,110] - 3 frac[33,49]
   - 3 frac[33,43] - 3 frac[33,103] - 3 frac[33,89] - 3 frac[33,120]
   - 3 frac[33,31] - 3 frac[33,107] - 3 frac[33,77] - 3 frac[33,62]
   - 3 frac[33,10] + rows_tot_tor_P42 = 0
 R1363: - 0.0208333333333333 rows_tot_tor_P42 + normalized_rows_tot_tor_P42
   = 0
 mem_tot_tor_P43: - mem[34,52] - mem[34,3] - mem[34,114] - mem[34,108]
   - mem[34,26] - mem[34,103] - mem[34,59] - mem[34,21] - mem[34,126]
   - mem[34,88] - mem[34,121] - mem[34,77] - mem[34,1] - mem[34,106]
   - mem[34,62] - mem[34,24] - mem[34,101] - mem[34,95] - mem[34,86]
   - mem[34,48] - mem[34,42] - mem[34,81] - mem[34,75] - mem[34,66]
   - mem[34,22] - mem[34,99] - mem[34,17] - mem[34,11] - mem[34,122]
   - mem[34,84] - mem[34,2] - mem[34,117] - mem[34,73] - mem[34,29]
   - mem[34,102] - mem[34,58] - mem[34,20] - mem[34,14] - mem[34,97]
   - mem[34,47] - mem[34,9] - mem[34,120] - mem[34,82] - mem[34,38]
   - mem[34,115] - mem[34,33] - mem[34,100] - mem[34,94] - mem[34,18]
   - mem[34,12] - mem[34,89] - mem[34,51] - mem[34,7] - mem[34,118]
   - mem[34,80] - mem[34,74] - mem[34,30] - mem[34,113] - mem[34,107]
   - mem[34,69] - mem[34,25] - mem[34,98] - mem[34,92] - mem[34,16]
   - mem[34,5] - mem[34,110] - mem[34,72] - mem[34,34] - mem[34,28]
   - mem[34,105] - mem[34,23] + mem_tot_tor_P43 = 0
 R1365: - 1.6276041666666666e-04 mem_tot_tor_P43
   + normalized_mem_tot_tor_P43 = 0
 rows_tot_tor_P43: - 3 frac[34,52] - 3 frac[34,3] - 3 frac[34,114]
   - 3 frac[34,108] - 3 frac[34,26] - 3 frac[34,103] - 3 frac[34,59]
   - 3 frac[34,21] - 3 frac[34,126] - 3 frac[34,88] - 3 frac[34,121]
   - 3 frac[34,77] - 3 frac[34,1] - 3 frac[34,106] - 3 frac[34,62]
   - 3 frac[34,24] - 3 frac[34,101] - 3 frac[34,95] - 3 frac[34,86]
   - 3 frac[34,48] - 3 frac[34,42] - 3 frac[34,81] - 3 frac[34,75]
   - 3 frac[34,66] - 3 frac[34,22] - 3 frac[34,99] - 3 frac[34,17]
   - 3 frac[34,11] - 3 frac[34,122] - 3 frac[34,84] - 3 frac[34,2]
   - 3 frac[34,117] - 3 frac[34,73] - 3 frac[34,29] - 3 frac[34,102]
   - 3 frac[34,58] - 3 frac[34,20] - 3 frac[34,14] - 3 frac[34,97]
   - 3 frac[34,47] - 3 frac[34,9] - 3 frac[34,120] - 3 frac[34,82]
   - 3 frac[34,38] - 3 frac[34,115] - 3 frac[34,33] - 3 frac[34,100]
   - 3 frac[34,94] - 3 frac[34,18] - 3 frac[34,12] - 3 frac[34,89]
   - 3 frac[34,51] - 3 frac[34,7] - 3 frac[34,118] - 3 frac[34,80]
   - 3 frac[34,74] - 3 frac[34,30] - 3 frac[34,113] - 3 frac[34,107]
   - 3 frac[34,69] - 3 frac[34,25] - 3 frac[34,98] - 3 frac[34,92]
   - 3 frac[34,16] - 3 frac[34,5] - 3 frac[34,110] - 3 frac[34,72]
   - 3 frac[34,34] - 3 frac[34,28] - 3 frac[34,105] - 3 frac[34,23]
   + rows_tot_tor_P43 = 0
 R1367: - 0.0208333333333333 rows_tot_tor_P43 + normalized_rows_tot_tor_P43
   = 0
 mem_tot_tor_P44: - mem[35,88] - mem[35,21] - mem[35,123] - mem[35,65]
   - mem[35,46] - mem[35,52] - mem[35,105] - mem[35,79] - mem[35,85]
   - mem[35,34] - mem[35,110] - mem[35,43] - mem[35,49] - mem[35,98]
   - mem[35,72] - mem[35,63] - mem[35,107] - mem[35,36] - mem[35,16]
   - mem[35,118] - mem[35,92] - mem[35,45] - mem[35,51] - mem[35,25]
   - mem[35,56] - mem[35,80] - mem[35,33] - mem[35,7] - mem[35,89]
   - mem[35,38] - mem[35,18] - mem[35,120] - mem[35,94] - mem[35,47]
   - mem[35,97] - mem[35,102] - mem[35,76] - mem[35,82] - mem[35,35]
   - mem[35,111] - mem[35,117] - mem[35,91] - mem[35,40] - mem[35,14]
   - mem[35,122] - mem[35,64] - mem[35,29] - mem[35,73] - mem[35,60]
   - mem[35,2] - mem[35,11] - mem[35,17] - mem[35,119] - mem[35,93]
   - mem[35,42] - mem[35,48] - mem[35,22] - mem[35,66] - mem[35,101]
   - mem[35,75] - mem[35,62] - mem[35,106] - mem[35,19] - mem[35,126]
   - mem[35,68] - mem[35,59] - mem[35,77] - mem[35,83] - mem[35,32]
   + mem_tot_tor_P44 = 0
 R1369: - 1.6276041666666666e-04 mem_tot_tor_P44
   + normalized_mem_tot_tor_P44 = 0
 rows_tot_tor_P44: - 3 frac[35,88] - 3 frac[35,21] - 3 frac[35,123]
   - 3 frac[35,65] - 3 frac[35,46] - 3 frac[35,52] - 3 frac[35,105]
   - 3 frac[35,79] - 3 frac[35,85] - 3 frac[35,34] - 3 frac[35,110]
   - 3 frac[35,43] - 3 frac[35,49] - 3 frac[35,98] - 3 frac[35,72]
   - 3 frac[35,63] - 3 frac[35,107] - 3 frac[35,36] - 3 frac[35,16]
   - 3 frac[35,118] - 3 frac[35,92] - 3 frac[35,45] - 3 frac[35,51]
   - 3 frac[35,25] - 3 frac[35,56] - 3 frac[35,80] - 3 frac[35,33]
   - 3 frac[35,7] - 3 frac[35,89] - 3 frac[35,38] - 3 frac[35,18]
   - 3 frac[35,120] - 3 frac[35,94] - 3 frac[35,47] - 3 frac[35,97]
   - 3 frac[35,102] - 3 frac[35,76] - 3 frac[35,82] - 3 frac[35,35]
   - 3 frac[35,111] - 3 frac[35,117] - 3 frac[35,91] - 3 frac[35,40]
   - 3 frac[35,14] - 3 frac[35,122] - 3 frac[35,64] - 3 frac[35,29]
   - 3 frac[35,73] - 3 frac[35,60] - 3 frac[35,2] - 3 frac[35,11]
   - 3 frac[35,17] - 3 frac[35,119] - 3 frac[35,93] - 3 frac[35,42]
   - 3 frac[35,48] - 3 frac[35,22] - 3 frac[35,66] - 3 frac[35,101]
   - 3 frac[35,75] - 3 frac[35,62] - 3 frac[35,106] - 3 frac[35,19]
   - 3 frac[35,126] - 3 frac[35,68] - 3 frac[35,59] - 3 frac[35,77]
   - 3 frac[35,83] - 3 frac[35,32] + rows_tot_tor_P44 = 0
 R1371: - 0.0208333333333333 rows_tot_tor_P44 + normalized_rows_tot_tor_P44
   = 0
 mem_tot_l1_P41: - mem[36,68] - mem[36,34] - mem[36,22] - mem[36,101]
   - mem[36,55] - mem[36,29] - mem[36,108] - mem[36,74] - mem[36,62]
   - mem[36,107] - mem[36,113] - mem[36,95] - mem[36,3] - mem[36,120]
   - mem[36,10] - mem[36,16] - mem[36,103] - mem[36,77] - mem[36,43]
   - mem[36,49] - mem[36,110] - mem[36,82] - mem[36,56] - mem[36,115]
   - mem[36,89] - mem[36,122] - mem[36,64] - mem[36,46] - mem[36,52]
   - mem[36,18] - mem[36,79] - mem[36,85] - mem[36,51] - mem[36,25]
   - mem[36,118] - mem[36,92] - mem[36,58] - mem[36,91] - mem[36,33]
   - mem[36,15] - mem[36,21] - mem[36,100] - mem[36,66] - mem[36,54]
   - mem[36,28] - mem[36,99] - mem[36,73] - mem[36,61] - mem[36,106]
   - mem[36,94] - mem[36,36] - mem[36,69] - mem[36,9] - mem[36,23]
   - mem[36,102] - mem[36,76] - mem[36,42] - mem[36,48] - mem[36,30]
   - mem[36,75] - mem[36,81] - mem[36,5] - mem[36,126] - mem[36,114]
   - mem[36,88] - mem[36,38] - mem[36,12] - mem[36,121] - mem[36,45]
   - mem[36,11] - mem[36,17] - mem[36,84] - mem[36,24] - mem[36,111]
   - mem[36,117] - mem[36,83] - mem[36,90] - mem[36,32] - mem[36,14]
   - mem[36,20] - mem[36,65] - mem[36,47] - mem[36,19] - mem[36,98]
   - mem[36,72] - mem[36,86] - mem[36,60] - mem[36,26] - mem[36,105]
   - mem[36,119] - mem[36,93] - mem[36,59] + mem_tot_l1_P41 = 0
 R1373: - 1.6276041666666666e-04 mem_tot_l1_P41 + normalized_mem_tot_l1_P41
   = 0
 rows_tot_l1_P41: - 3 frac[36,68] - 3 frac[36,34] - 3 frac[36,22]
   - 3 frac[36,101] - 3 frac[36,55] - 3 frac[36,29] - 3 frac[36,108]
   - 3 frac[36,74] - 3 frac[36,62] - 3 frac[36,107] - 3 frac[36,113]
   - 3 frac[36,95] - 3 frac[36,3] - 3 frac[36,120] - 3 frac[36,10]
   - 3 frac[36,16] - 3 frac[36,103] - 3 frac[36,77] - 3 frac[36,43]
   - 3 frac[36,49] - 3 frac[36,110] - 3 frac[36,82] - 3 frac[36,56]
   - 3 frac[36,115] - 3 frac[36,89] - 3 frac[36,122] - 3 frac[36,64]
   - 3 frac[36,46] - 3 frac[36,52] - 3 frac[36,18] - 3 frac[36,79]
   - 3 frac[36,85] - 3 frac[36,51] - 3 frac[36,25] - 3 frac[36,118]
   - 3 frac[36,92] - 3 frac[36,58] - 3 frac[36,91] - 3 frac[36,33]
   - 3 frac[36,15] - 3 frac[36,21] - 3 frac[36,100] - 3 frac[36,66]
   - 3 frac[36,54] - 3 frac[36,28] - 3 frac[36,99] - 3 frac[36,73]
   - 3 frac[36,61] - 3 frac[36,106] - 3 frac[36,94] - 3 frac[36,36]
   - 3 frac[36,69] - 3 frac[36,9] - 3 frac[36,23] - 3 frac[36,102]
   - 3 frac[36,76] - 3 frac[36,42] - 3 frac[36,48] - 3 frac[36,30]
   - 3 frac[36,75] - 3 frac[36,81] - 3 frac[36,5] - 3 frac[36,126]
   - 3 frac[36,114] - 3 frac[36,88] - 3 frac[36,38] - 3 frac[36,12]
   - 3 frac[36,121] - 3 frac[36,45] - 3 frac[36,11] - 3 frac[36,17]
   - 3 frac[36,84] - 3 frac[36,24] - 3 frac[36,111] - 3 frac[36,117]
   - 3 frac[36,83] - 3 frac[36,90] - 3 frac[36,32] - 3 frac[36,14]
   - 3 frac[36,20] - 3 frac[36,65] - 3 frac[36,47] - 3 frac[36,19]
   - 3 frac[36,98] - 3 frac[36,72] - 3 frac[36,86] - 3 frac[36,60]
   - 3 frac[36,26] - 3 frac[36,105] - 3 frac[36,119] - 3 frac[36,93]
   - 3 frac[36,59] + rows_tot_l1_P41 = 0
 R1375: - 0.0208333333333333 rows_tot_l1_P41 + normalized_rows_tot_l1_P41
   = 0
 mem_tot_l1_P42: - mem[37,82] - mem[37,23] - mem[37,113] - mem[37,91]
   - mem[37,16] - mem[37,122] - mem[37,68] - mem[37,25] - mem[37,99]
   - mem[37,77] - mem[37,56] - mem[37,2] - mem[37,108] - mem[37,118]
   - mem[37,80] - mem[37,33] - mem[37,11] - mem[37,21] - mem[37,89]
   - mem[37,42] - mem[37,52] - mem[37,30] - mem[37,120] - mem[37,66]
   - mem[37,7] - mem[37,97] - mem[37,75] - mem[37,85] - mem[37,38]
   - mem[37,106] - mem[37,94] - mem[37,47] - mem[37,9] - mem[37,19]
   - mem[37,28] - mem[37,102] - mem[37,64] - mem[37,59] - mem[37,5]
   - mem[37,111] - mem[37,73] - mem[37,83] - mem[37,36] - mem[37,14]
   - mem[37,114] - mem[37,92] - mem[37,45] - mem[37,17] - mem[37,69]
   - mem[37,48] - mem[37,26] - mem[37,100] - mem[37,3] - mem[37,119]
   - mem[37,81] - mem[37,34] - mem[37,12] - mem[37,22] - mem[37,43]
   - mem[37,121] - mem[37,62] - mem[37,24] - mem[37,98] - mem[37,76]
   - mem[37,86] - mem[37,107] - mem[37,117] - mem[37,95] - mem[37,32]
   - mem[37,20] - mem[37,126] - mem[37,88] - mem[37,51] - mem[37,29]
   - mem[37,103] - mem[37,65] - mem[37,60] - mem[37,74] - mem[37,84]
   - mem[37,105] - mem[37,115] - mem[37,93] - mem[37,46] - mem[37,18]
   - mem[37,101] - mem[37,79] - mem[37,58] - mem[37,110] - mem[37,72]
   + mem_tot_l1_P42 = 0
 R1377: - 1.6276041666666666e-04 mem_tot_l1_P42 + normalized_mem_tot_l1_P42
   = 0
 rows_tot_l1_P42: - 3 frac[37,82] - 3 frac[37,23] - 3 frac[37,113]
   - 3 frac[37,91] - 3 frac[37,16] - 3 frac[37,122] - 3 frac[37,68]
   - 3 frac[37,25] - 3 frac[37,99] - 3 frac[37,77] - 3 frac[37,56]
   - 3 frac[37,2] - 3 frac[37,108] - 3 frac[37,118] - 3 frac[37,80]
   - 3 frac[37,33] - 3 frac[37,11] - 3 frac[37,21] - 3 frac[37,89]
   - 3 frac[37,42] - 3 frac[37,52] - 3 frac[37,30] - 3 frac[37,120]
   - 3 frac[37,66] - 3 frac[37,7] - 3 frac[37,97] - 3 frac[37,75]
   - 3 frac[37,85] - 3 frac[37,38] - 3 frac[37,106] - 3 frac[37,94]
   - 3 frac[37,47] - 3 frac[37,9] - 3 frac[37,19] - 3 frac[37,28]
   - 3 frac[37,102] - 3 frac[37,64] - 3 frac[37,59] - 3 frac[37,5]
   - 3 frac[37,111] - 3 frac[37,73] - 3 frac[37,83] - 3 frac[37,36]
   - 3 frac[37,14] - 3 frac[37,114] - 3 frac[37,92] - 3 frac[37,45]
   - 3 frac[37,17] - 3 frac[37,69] - 3 frac[37,48] - 3 frac[37,26]
   - 3 frac[37,100] - 3 frac[37,3] - 3 frac[37,119] - 3 frac[37,81]
   - 3 frac[37,34] - 3 frac[37,12] - 3 frac[37,22] - 3 frac[37,43]
   - 3 frac[37,121] - 3 frac[37,62] - 3 frac[37,24] - 3 frac[37,98]
   - 3 frac[37,76] - 3 frac[37,86] - 3 frac[37,107] - 3 frac[37,117]
   - 3 frac[37,95] - 3 frac[37,32] - 3 frac[37,20] - 3 frac[37,126]
   - 3 frac[37,88] - 3 frac[37,51] - 3 frac[37,29] - 3 frac[37,103]
   - 3 frac[37,65] - 3 frac[37,60] - 3 frac[37,74] - 3 frac[37,84]
   - 3 frac[37,105] - 3 frac[37,115] - 3 frac[37,93] - 3 frac[37,46]
   - 3 frac[37,18] - 3 frac[37,101] - 3 frac[37,79] - 3 frac[37,58]
   - 3 frac[37,110] - 3 frac[37,72] + rows_tot_l1_P42 = 0
 R1379: - 0.0208333333333333 rows_tot_l1_P42 + normalized_rows_tot_l1_P42
   = 0
 mem_tot_l2_P4: - mem[38,86] - mem[38,119] - mem[38,73] - mem[38,5]
   - mem[38,110] - mem[38,64] - mem[38,60] - mem[38,22] - mem[38,93]
   - mem[38,9] - mem[38,114] - mem[38,84] - mem[38,46] - mem[38,26]
   - mem[38,117] - mem[38,79] - mem[38,33] - mem[38,59] - mem[38,29]
   - mem[38,108] - mem[38,88] - mem[38,20] - mem[38,103] - mem[38,121]
   - mem[38,83] - mem[38,74] - mem[38,24] - mem[38,107] - mem[38,77]
   - mem[38,19] - mem[38,98] - mem[38,68] - mem[38,48] - mem[38,81]
   - mem[38,43] - mem[38,118] - mem[38,72] - mem[38,34] - mem[38,30]
   - mem[38,105] - mem[38,17] - mem[38,92] - mem[38,3] - mem[38,32]
   - mem[38,58] - mem[38,28] - mem[38,111] - mem[38,65] - mem[38,91]
   - mem[38,23] - mem[38,102] - mem[38,120] - mem[38,82] - mem[38,52]
   - mem[38,14] - mem[38,115] - mem[38,85] - mem[38,47] - mem[38,106]
   - mem[38,76] - mem[38,38] - mem[38,56] - mem[38,18] - mem[38,89]
   - mem[38,51] - mem[38,21] - mem[38,100] - mem[38,126] - mem[38,42]
   - mem[38,12] - mem[38,113] - mem[38,45] - mem[38,25] - mem[38,36]
   - mem[38,62] - mem[38,16] - mem[38,99] - mem[38,69] - mem[38,95]
   - mem[38,11] + mem_tot_l2_P4 = 0
 R1381: - 1.6276041666666666e-04 mem_tot_l2_P4 + normalized_mem_tot_l2_P4
   = 0
 rows_tot_l2_P4: - 3 frac[38,86] - 3 frac[38,119] - 3 frac[38,73]
   - 3 frac[38,5] - 3 frac[38,110] - 3 frac[38,64] - 3 frac[38,60]
   - 3 frac[38,22] - 3 frac[38,93] - 3 frac[38,9] - 3 frac[38,114]
   - 3 frac[38,84] - 3 frac[38,46] - 3 frac[38,26] - 3 frac[38,117]
   - 3 frac[38,79] - 3 frac[38,33] - 3 frac[38,59] - 3 frac[38,29]
   - 3 frac[38,108] - 3 frac[38,88] - 3 frac[38,20] - 3 frac[38,103]
   - 3 frac[38,121] - 3 frac[38,83] - 3 frac[38,74] - 3 frac[38,24]
   - 3 frac[38,107] - 3 frac[38,77] - 3 frac[38,19] - 3 frac[38,98]
   - 3 frac[38,68] - 3 frac[38,48] - 3 frac[38,81] - 3 frac[38,43]
   - 3 frac[38,118] - 3 frac[38,72] - 3 frac[38,34] - 3 frac[38,30]
   - 3 frac[38,105] - 3 frac[38,17] - 3 frac[38,92] - 3 frac[38,3]
   - 3 frac[38,32] - 3 frac[38,58] - 3 frac[38,28] - 3 frac[38,111]
   - 3 frac[38,65] - 3 frac[38,91] - 3 frac[38,23] - 3 frac[38,102]
   - 3 frac[38,120] - 3 frac[38,82] - 3 frac[38,52] - 3 frac[38,14]
   - 3 frac[38,115] - 3 frac[38,85] - 3 frac[38,47] - 3 frac[38,106]
   - 3 frac[38,76] - 3 frac[38,38] - 3 frac[38,56] - 3 frac[38,18]
   - 3 frac[38,89] - 3 frac[38,51] - 3 frac[38,21] - 3 frac[38,100]
   - 3 frac[38,126] - 3 frac[38,42] - 3 frac[38,12] - 3 frac[38,113]
   - 3 frac[38,45] - 3 frac[38,25] - 3 frac[38,36] - 3 frac[38,62]
   - 3 frac[38,16] - 3 frac[38,99] - 3 frac[38,69] - 3 frac[38,95]
   - 3 frac[38,11] + rows_tot_l2_P4 = 0
 R1383: - 0.0208333333333333 rows_tot_l2_P4 + normalized_rows_tot_l2_P4
   = 0
 row_capacity_CPU1: rows_tot_CPU1 <= 12
 row_capacity_CPU2: rows_tot_CPU2 <= 12
 row_capacity_CPU3: rows_tot_CPU3 <= 12
 row_capacity_CPU4: rows_tot_CPU4 <= 12
 row_capacity_CPU5: rows_tot_CPU5 <= 12
 row_capacity_CPU6: rows_tot_CPU6 <= 12
 row_capacity_CPU7: rows_tot_CPU7 <= 12
 row_capacity_CPU8: rows_tot_CPU8 <= 12
 row_capacity_CPU9: rows_tot_CPU9 <= 12
 row_capacity_CPU10: rows_tot_CPU10 <= 12
 row_capacity_CPU11: rows_tot_CPU11 <= 12
 row_capacity_CPU12: rows_tot_CPU12 <= 12
 row_capacity_CPU13: rows_tot_CPU13 <= 12
 row_capacity_CPU14: rows_tot_CPU14 <= 12
 row_capacity_CPU15: rows_tot_CPU15 <= 12
 row_capacity_CPU16: rows_tot_CPU16 <= 12
 row_capacity_CPU17: rows_tot_CPU17 <= 12
 row_capacity_CPU18: rows_tot_CPU18 <= 12
 row_capacity_CPU19: rows_tot_CPU19 <= 12
 row_capacity_CPU20: rows_tot_CPU20 <= 12
 row_capacity_CPU21: rows_tot_CPU21 <= 12
 row_capacity_CPU22: rows_tot_CPU22 <= 12
 row_capacity_CPU23: rows_tot_CPU23 <= 12
 row_capacity_CPU24: rows_tot_CPU24 <= 12
 row_capacity_CPU25: rows_tot_CPU25 <= 12
 row_capacity_CPU26: rows_tot_CPU26 <= 12
 row_capacity_CPU27: rows_tot_CPU27 <= 12
 row_capacity_CPU28: rows_tot_CPU28 <= 12
 row_capacity_CPU29: rows_tot_CPU29 <= 12
 row_capacity_CPU30: rows_tot_CPU30 <= 12
 row_capacity_CPU31: rows_tot_CPU31 <= 12
 row_capacity_CPU32: rows_tot_CPU32 <= 12
 row_capacity_tor_P41: rows_tot_tor_P41 <= 48
 capacity_mem_par_tor_P41: mem[32,54] <= 1536
 capacity_mem_par_tor_P41: mem[32,113] <= 1536
 capacity_mem_par_tor_P41: mem[32,77] <= 1536
 capacity_mem_par_tor_P41: mem[32,25] <= 1536
 capacity_mem_par_tor_P41: mem[32,110] <= 1536
 capacity_mem_par_tor_P41: mem[32,68] <= 1536
 capacity_mem_par_tor_P41: mem[32,16] <= 1536
 capacity_mem_par_tor_P41: mem[32,101] <= 1536
 capacity_mem_par_tor_P41: mem[32,91] <= 1536
 capacity_mem_par_tor_P41: mem[32,49] <= 1536
 capacity_mem_par_tor_P41: mem[32,82] <= 1536
 capacity_mem_par_tor_P41: mem[32,72] <= 1536
 capacity_mem_par_tor_P41: mem[32,115] <= 1536
 capacity_mem_par_tor_P41: mem[32,105] <= 1536
 capacity_mem_par_tor_P41: mem[32,79] <= 1536
 capacity_mem_par_tor_P41: mem[32,96] <= 1536
 capacity_mem_par_tor_P41: mem[32,18] <= 1536
 capacity_mem_par_tor_P41: mem[32,103] <= 1536
 capacity_mem_par_tor_P41: mem[32,93] <= 1536
 capacity_mem_par_tor_P41: mem[32,15] <= 1536
 capacity_mem_par_tor_P41: mem[32,84] <= 1536
 capacity_mem_par_tor_P41: mem[32,74] <= 1536
 capacity_mem_par_tor_P41: mem[32,32] <= 1536
 capacity_mem_par_tor_P41: mem[32,117] <= 1536
 capacity_mem_par_tor_P41: mem[32,107] <= 1536
 capacity_mem_par_tor_P41: mem[32,65] <= 1536
 capacity_mem_par_tor_P41: mem[32,39] <= 1536
 capacity_mem_par_tor_P41: mem[32,29] <= 1536
 capacity_mem_par_tor_P41: mem[32,98] <= 1536
 capacity_mem_par_tor_P41: mem[32,88] <= 1536
 capacity_mem_par_tor_P41: mem[32,62] <= 1536
 capacity_mem_par_tor_P41: mem[32,20] <= 1536
 capacity_mem_par_tor_P41: mem[32,10] <= 1536
 capacity_mem_par_tor_P41: mem[32,121] <= 1536
 capacity_mem_par_tor_P41: mem[32,95] <= 1536
 capacity_mem_par_tor_P41: mem[32,86] <= 1536
 capacity_mem_par_tor_P41: mem[32,76] <= 1536
 capacity_mem_par_tor_P41: mem[32,34] <= 1536
 capacity_mem_par_tor_P41: mem[32,24] <= 1536
 capacity_mem_par_tor_P41: mem[32,119] <= 1536
 capacity_mem_par_tor_P41: mem[32,100] <= 1536
 capacity_mem_par_tor_P41: mem[32,90] <= 1536
 capacity_mem_par_tor_P41: mem[32,48] <= 1536
 capacity_mem_par_tor_P41: mem[32,22] <= 1536
 capacity_mem_par_tor_P41: mem[32,12] <= 1536
 capacity_mem_par_tor_P41: mem[32,55] <= 1536
 capacity_mem_par_tor_P41: mem[32,45] <= 1536
 capacity_mem_par_tor_P41: mem[32,3] <= 1536
 capacity_mem_par_tor_P41: mem[32,114] <= 1536
 capacity_mem_par_tor_P41: mem[32,17] <= 1536
 capacity_mem_par_tor_P41: mem[32,102] <= 1536
 capacity_mem_par_tor_P41: mem[32,92] <= 1536
 capacity_mem_par_tor_P41: mem[32,14] <= 1536
 capacity_mem_par_tor_P41: mem[32,83] <= 1536
 capacity_mem_par_tor_P41: mem[32,47] <= 1536
 capacity_mem_par_tor_P41: mem[32,5] <= 1536
 capacity_mem_par_tor_P41: mem[32,64] <= 1536
 capacity_mem_par_tor_P41: mem[32,28] <= 1536
 capacity_mem_par_tor_P41: mem[32,71] <= 1536
 capacity_mem_par_tor_P41: mem[32,61] <= 1536
 capacity_mem_par_tor_P41: mem[32,9] <= 1536
 capacity_mem_par_tor_P41: mem[32,120] <= 1536
 capacity_mem_par_tor_P41: mem[32,94] <= 1536
 capacity_mem_par_tor_P41: mem[32,85] <= 1536
 capacity_mem_par_tor_P41: mem[32,75] <= 1536
 capacity_mem_par_tor_P41: mem[32,108] <= 1536
 capacity_mem_par_tor_P41: mem[32,66] <= 1536
 capacity_mem_par_tor_P41: mem[32,56] <= 1536
 capacity_mem_par_tor_P41: mem[32,30] <= 1536
 capacity_mem_par_tor_P41: mem[32,99] <= 1536
 capacity_mem_par_tor_P41: mem[32,89] <= 1536
 capacity_mem_par_tor_P41: mem[32,21] <= 1536
 capacity_mem_par_tor_P41: mem[32,11] <= 1536
 capacity_mem_par_tor_P41: mem[32,122] <= 1536
 capacity_mem_par_tor_P41: mem[32,80] <= 1536
 row_capacity_tor_P42: rows_tot_tor_P42 <= 48
 capacity_mem_par_tor_P42: mem[33,55] <= 1536
 capacity_mem_par_tor_P42: mem[33,101] <= 1536
 capacity_mem_par_tor_P42: mem[33,95] <= 1536
 capacity_mem_par_tor_P42: mem[33,34] <= 1536
 capacity_mem_par_tor_P42: mem[33,126] <= 1536
 capacity_mem_par_tor_P42: mem[33,74] <= 1536
 capacity_mem_par_tor_P42: mem[33,59] <= 1536
 capacity_mem_par_tor_P42: mem[33,29] <= 1536
 capacity_mem_par_tor_P42: mem[33,105] <= 1536
 capacity_mem_par_tor_P42: mem[33,60] <= 1536
 capacity_mem_par_tor_P42: mem[33,22] <= 1536
 capacity_mem_par_tor_P42: mem[33,68] <= 1536
 capacity_mem_par_tor_P42: mem[33,47] <= 1536
 capacity_mem_par_tor_P42: mem[33,93] <= 1536
 capacity_mem_par_tor_P42: mem[33,26] <= 1536
 capacity_mem_par_tor_P42: mem[33,72] <= 1536
 capacity_mem_par_tor_P42: mem[33,57] <= 1536
 capacity_mem_par_tor_P42: mem[33,19] <= 1536
 capacity_mem_par_tor_P42: mem[33,117] <= 1536
 capacity_mem_par_tor_P42: mem[33,111] <= 1536
 capacity_mem_par_tor_P42: mem[33,65] <= 1536
 capacity_mem_par_tor_P42: mem[33,20] <= 1536
 capacity_mem_par_tor_P42: mem[33,14] <= 1536
 capacity_mem_par_tor_P42: mem[33,90] <= 1536
 capacity_mem_par_tor_P42: mem[33,45] <= 1536
 capacity_mem_par_tor_P42: mem[33,121] <= 1536
 capacity_mem_par_tor_P42: mem[33,38] <= 1536
 capacity_mem_par_tor_P42: mem[33,24] <= 1536
 capacity_mem_par_tor_P42: mem[33,17] <= 1536
 capacity_mem_par_tor_P42: mem[33,11] <= 1536
 capacity_mem_par_tor_P42: mem[33,48] <= 1536
 capacity_mem_par_tor_P42: mem[33,42] <= 1536
 capacity_mem_par_tor_P42: mem[33,12] <= 1536
 capacity_mem_par_tor_P42: mem[33,102] <= 1536
 capacity_mem_par_tor_P42: mem[33,88] <= 1536
 capacity_mem_par_tor_P42: mem[33,35] <= 1536
 capacity_mem_par_tor_P42: mem[33,5] <= 1536
 capacity_mem_par_tor_P42: mem[33,81] <= 1536
 capacity_mem_par_tor_P42: mem[33,75] <= 1536
 capacity_mem_par_tor_P42: mem[33,36] <= 1536
 capacity_mem_par_tor_P42: mem[33,30] <= 1536
 capacity_mem_par_tor_P42: mem[33,106] <= 1536
 capacity_mem_par_tor_P42: mem[33,76] <= 1536
 capacity_mem_par_tor_P42: mem[33,61] <= 1536
 capacity_mem_par_tor_P42: mem[33,23] <= 1536
 capacity_mem_par_tor_P42: mem[33,9] <= 1536
 capacity_mem_par_tor_P42: mem[33,99] <= 1536
 capacity_mem_par_tor_P42: mem[33,69] <= 1536
 capacity_mem_par_tor_P42: mem[33,54] <= 1536
 capacity_mem_par_tor_P42: mem[33,100] <= 1536
 capacity_mem_par_tor_P42: mem[33,94] <= 1536
 capacity_mem_par_tor_P42: mem[33,33] <= 1536
 capacity_mem_par_tor_P42: mem[33,27] <= 1536
 capacity_mem_par_tor_P42: mem[33,73] <= 1536
 capacity_mem_par_tor_P42: mem[33,58] <= 1536
 capacity_mem_par_tor_P42: mem[33,28] <= 1536
 capacity_mem_par_tor_P42: mem[33,118] <= 1536
 capacity_mem_par_tor_P42: mem[33,66] <= 1536
 capacity_mem_par_tor_P42: mem[33,51] <= 1536
 capacity_mem_par_tor_P42: mem[33,21] <= 1536
 capacity_mem_par_tor_P42: mem[33,15] <= 1536
 capacity_mem_par_tor_P42: mem[33,52] <= 1536
 capacity_mem_par_tor_P42: mem[33,46] <= 1536
 capacity_mem_par_tor_P42: mem[33,122] <= 1536
 capacity_mem_par_tor_P42: mem[33,92] <= 1536
 capacity_mem_par_tor_P42: mem[33,39] <= 1536
 capacity_mem_par_tor_P42: mem[33,25] <= 1536
 capacity_mem_par_tor_P42: mem[33,56] <= 1536
 capacity_mem_par_tor_P42: mem[33,18] <= 1536
 capacity_mem_par_tor_P42: mem[33,110] <= 1536
 capacity_mem_par_tor_P42: mem[33,49] <= 1536
 capacity_mem_par_tor_P42: mem[33,43] <= 1536
 capacity_mem_par_tor_P42: mem[33,103] <= 1536
 capacity_mem_par_tor_P42: mem[33,89] <= 1536
 capacity_mem_par_tor_P42: mem[33,120] <= 1536
 capacity_mem_par_tor_P42: mem[33,31] <= 1536
 capacity_mem_par_tor_P42: mem[33,107] <= 1536
 capacity_mem_par_tor_P42: mem[33,77] <= 1536
 capacity_mem_par_tor_P42: mem[33,62] <= 1536
 capacity_mem_par_tor_P42: mem[33,10] <= 1536
 row_capacity_tor_P43: rows_tot_tor_P43 <= 48
 capacity_mem_par_tor_P43: mem[34,52] <= 1536
 capacity_mem_par_tor_P43: mem[34,3] <= 1536
 capacity_mem_par_tor_P43: mem[34,114] <= 1536
 capacity_mem_par_tor_P43: mem[34,108] <= 1536
 capacity_mem_par_tor_P43: mem[34,26] <= 1536
 capacity_mem_par_tor_P43: mem[34,103] <= 1536
 capacity_mem_par_tor_P43: mem[34,59] <= 1536
 capacity_mem_par_tor_P43: mem[34,21] <= 1536
 capacity_mem_par_tor_P43: mem[34,126] <= 1536
 capacity_mem_par_tor_P43: mem[34,88] <= 1536
 capacity_mem_par_tor_P43: mem[34,121] <= 1536
 capacity_mem_par_tor_P43: mem[34,77] <= 1536
 capacity_mem_par_tor_P43: mem[34,1] <= 1536
 capacity_mem_par_tor_P43: mem[34,106] <= 1536
 capacity_mem_par_tor_P43: mem[34,62] <= 1536
 capacity_mem_par_tor_P43: mem[34,24] <= 1536
 capacity_mem_par_tor_P43: mem[34,101] <= 1536
 capacity_mem_par_tor_P43: mem[34,95] <= 1536
 capacity_mem_par_tor_P43: mem[34,86] <= 1536
 capacity_mem_par_tor_P43: mem[34,48] <= 1536
 capacity_mem_par_tor_P43: mem[34,42] <= 1536
 capacity_mem_par_tor_P43: mem[34,81] <= 1536
 capacity_mem_par_tor_P43: mem[34,75] <= 1536
 capacity_mem_par_tor_P43: mem[34,66] <= 1536
 capacity_mem_par_tor_P43: mem[34,22] <= 1536
 capacity_mem_par_tor_P43: mem[34,99] <= 1536
 capacity_mem_par_tor_P43: mem[34,17] <= 1536
 capacity_mem_par_tor_P43: mem[34,11] <= 1536
 capacity_mem_par_tor_P43: mem[34,122] <= 1536
 capacity_mem_par_tor_P43: mem[34,84] <= 1536
 capacity_mem_par_tor_P43: mem[34,2] <= 1536
 capacity_mem_par_tor_P43: mem[34,117] <= 1536
 capacity_mem_par_tor_P43: mem[34,73] <= 1536
 capacity_mem_par_tor_P43: mem[34,29] <= 1536
 capacity_mem_par_tor_P43: mem[34,102] <= 1536
 capacity_mem_par_tor_P43: mem[34,58] <= 1536
 capacity_mem_par_tor_P43: mem[34,20] <= 1536
 capacity_mem_par_tor_P43: mem[34,14] <= 1536
 capacity_mem_par_tor_P43: mem[34,97] <= 1536
 capacity_mem_par_tor_P43: mem[34,47] <= 1536
 capacity_mem_par_tor_P43: mem[34,9] <= 1536
 capacity_mem_par_tor_P43: mem[34,120] <= 1536
 capacity_mem_par_tor_P43: mem[34,82] <= 1536
 capacity_mem_par_tor_P43: mem[34,38] <= 1536
 capacity_mem_par_tor_P43: mem[34,115] <= 1536
 capacity_mem_par_tor_P43: mem[34,33] <= 1536
 capacity_mem_par_tor_P43: mem[34,100] <= 1536
 capacity_mem_par_tor_P43: mem[34,94] <= 1536
 capacity_mem_par_tor_P43: mem[34,18] <= 1536
 capacity_mem_par_tor_P43: mem[34,12] <= 1536
 capacity_mem_par_tor_P43: mem[34,89] <= 1536
 capacity_mem_par_tor_P43: mem[34,51] <= 1536
 capacity_mem_par_tor_P43: mem[34,7] <= 1536
 capacity_mem_par_tor_P43: mem[34,118] <= 1536
 capacity_mem_par_tor_P43: mem[34,80] <= 1536
 capacity_mem_par_tor_P43: mem[34,74] <= 1536
 capacity_mem_par_tor_P43: mem[34,30] <= 1536
 capacity_mem_par_tor_P43: mem[34,113] <= 1536
 capacity_mem_par_tor_P43: mem[34,107] <= 1536
 capacity_mem_par_tor_P43: mem[34,69] <= 1536
 capacity_mem_par_tor_P43: mem[34,25] <= 1536
 capacity_mem_par_tor_P43: mem[34,98] <= 1536
 capacity_mem_par_tor_P43: mem[34,92] <= 1536
 capacity_mem_par_tor_P43: mem[34,16] <= 1536
 capacity_mem_par_tor_P43: mem[34,5] <= 1536
 capacity_mem_par_tor_P43: mem[34,110] <= 1536
 capacity_mem_par_tor_P43: mem[34,72] <= 1536
 capacity_mem_par_tor_P43: mem[34,34] <= 1536
 capacity_mem_par_tor_P43: mem[34,28] <= 1536
 capacity_mem_par_tor_P43: mem[34,105] <= 1536
 capacity_mem_par_tor_P43: mem[34,23] <= 1536
 row_capacity_tor_P44: rows_tot_tor_P44 <= 48
 capacity_mem_par_tor_P44: mem[35,88] <= 1536
 capacity_mem_par_tor_P44: mem[35,21] <= 1536
 capacity_mem_par_tor_P44: mem[35,123] <= 1536
 capacity_mem_par_tor_P44: mem[35,65] <= 1536
 capacity_mem_par_tor_P44: mem[35,46] <= 1536
 capacity_mem_par_tor_P44: mem[35,52] <= 1536
 capacity_mem_par_tor_P44: mem[35,105] <= 1536
 capacity_mem_par_tor_P44: mem[35,79] <= 1536
 capacity_mem_par_tor_P44: mem[35,85] <= 1536
 capacity_mem_par_tor_P44: mem[35,34] <= 1536
 capacity_mem_par_tor_P44: mem[35,110] <= 1536
 capacity_mem_par_tor_P44: mem[35,43] <= 1536
 capacity_mem_par_tor_P44: mem[35,49] <= 1536
 capacity_mem_par_tor_P44: mem[35,98] <= 1536
 capacity_mem_par_tor_P44: mem[35,72] <= 1536
 capacity_mem_par_tor_P44: mem[35,63] <= 1536
 capacity_mem_par_tor_P44: mem[35,107] <= 1536
 capacity_mem_par_tor_P44: mem[35,36] <= 1536
 capacity_mem_par_tor_P44: mem[35,16] <= 1536
 capacity_mem_par_tor_P44: mem[35,118] <= 1536
 capacity_mem_par_tor_P44: mem[35,92] <= 1536
 capacity_mem_par_tor_P44: mem[35,45] <= 1536
 capacity_mem_par_tor_P44: mem[35,51] <= 1536
 capacity_mem_par_tor_P44: mem[35,25] <= 1536
 capacity_mem_par_tor_P44: mem[35,56] <= 1536
 capacity_mem_par_tor_P44: mem[35,80] <= 1536
 capacity_mem_par_tor_P44: mem[35,33] <= 1536
 capacity_mem_par_tor_P44: mem[35,7] <= 1536
 capacity_mem_par_tor_P44: mem[35,89] <= 1536
 capacity_mem_par_tor_P44: mem[35,38] <= 1536
 capacity_mem_par_tor_P44: mem[35,18] <= 1536
 capacity_mem_par_tor_P44: mem[35,120] <= 1536
 capacity_mem_par_tor_P44: mem[35,94] <= 1536
 capacity_mem_par_tor_P44: mem[35,47] <= 1536
 capacity_mem_par_tor_P44: mem[35,97] <= 1536
 capacity_mem_par_tor_P44: mem[35,102] <= 1536
 capacity_mem_par_tor_P44: mem[35,76] <= 1536
 capacity_mem_par_tor_P44: mem[35,82] <= 1536
 capacity_mem_par_tor_P44: mem[35,35] <= 1536
 capacity_mem_par_tor_P44: mem[35,111] <= 1536
 capacity_mem_par_tor_P44: mem[35,117] <= 1536
 capacity_mem_par_tor_P44: mem[35,91] <= 1536
 capacity_mem_par_tor_P44: mem[35,40] <= 1536
 capacity_mem_par_tor_P44: mem[35,14] <= 1536
 capacity_mem_par_tor_P44: mem[35,122] <= 1536
 capacity_mem_par_tor_P44: mem[35,64] <= 1536
 capacity_mem_par_tor_P44: mem[35,29] <= 1536
 capacity_mem_par_tor_P44: mem[35,73] <= 1536
 capacity_mem_par_tor_P44: mem[35,60] <= 1536
 capacity_mem_par_tor_P44: mem[35,2] <= 1536
 capacity_mem_par_tor_P44: mem[35,11] <= 1536
 capacity_mem_par_tor_P44: mem[35,17] <= 1536
 capacity_mem_par_tor_P44: mem[35,119] <= 1536
 capacity_mem_par_tor_P44: mem[35,93] <= 1536
 capacity_mem_par_tor_P44: mem[35,42] <= 1536
 capacity_mem_par_tor_P44: mem[35,48] <= 1536
 capacity_mem_par_tor_P44: mem[35,22] <= 1536
 capacity_mem_par_tor_P44: mem[35,66] <= 1536
 capacity_mem_par_tor_P44: mem[35,101] <= 1536
 capacity_mem_par_tor_P44: mem[35,75] <= 1536
 capacity_mem_par_tor_P44: mem[35,62] <= 1536
 capacity_mem_par_tor_P44: mem[35,106] <= 1536
 capacity_mem_par_tor_P44: mem[35,19] <= 1536
 capacity_mem_par_tor_P44: mem[35,126] <= 1536
 capacity_mem_par_tor_P44: mem[35,68] <= 1536
 capacity_mem_par_tor_P44: mem[35,59] <= 1536
 capacity_mem_par_tor_P44: mem[35,77] <= 1536
 capacity_mem_par_tor_P44: mem[35,83] <= 1536
 capacity_mem_par_tor_P44: mem[35,32] <= 1536
 row_capacity_l1_P41: rows_tot_l1_P41 <= 48
 capacity_mem_par_l1_P41: mem[36,68] <= 1536
 capacity_mem_par_l1_P41: mem[36,34] <= 1536
 capacity_mem_par_l1_P41: mem[36,22] <= 1536
 capacity_mem_par_l1_P41: mem[36,101] <= 1536
 capacity_mem_par_l1_P41: mem[36,55] <= 1536
 capacity_mem_par_l1_P41: mem[36,29] <= 1536
 capacity_mem_par_l1_P41: mem[36,108] <= 1536
 capacity_mem_par_l1_P41: mem[36,74] <= 1536
 capacity_mem_par_l1_P41: mem[36,62] <= 1536
 capacity_mem_par_l1_P41: mem[36,107] <= 1536
 capacity_mem_par_l1_P41: mem[36,113] <= 1536
 capacity_mem_par_l1_P41: mem[36,95] <= 1536
 capacity_mem_par_l1_P41: mem[36,3] <= 1536
 capacity_mem_par_l1_P41: mem[36,120] <= 1536
 capacity_mem_par_l1_P41: mem[36,10] <= 1536
 capacity_mem_par_l1_P41: mem[36,16] <= 1536
 capacity_mem_par_l1_P41: mem[36,103] <= 1536
 capacity_mem_par_l1_P41: mem[36,77] <= 1536
 capacity_mem_par_l1_P41: mem[36,43] <= 1536
 capacity_mem_par_l1_P41: mem[36,49] <= 1536
 capacity_mem_par_l1_P41: mem[36,110] <= 1536
 capacity_mem_par_l1_P41: mem[36,82] <= 1536
 capacity_mem_par_l1_P41: mem[36,56] <= 1536
 capacity_mem_par_l1_P41: mem[36,115] <= 1536
 capacity_mem_par_l1_P41: mem[36,89] <= 1536
 capacity_mem_par_l1_P41: mem[36,122] <= 1536
 capacity_mem_par_l1_P41: mem[36,64] <= 1536
 capacity_mem_par_l1_P41: mem[36,46] <= 1536
 capacity_mem_par_l1_P41: mem[36,52] <= 1536
 capacity_mem_par_l1_P41: mem[36,18] <= 1536
 capacity_mem_par_l1_P41: mem[36,79] <= 1536
 capacity_mem_par_l1_P41: mem[36,85] <= 1536
 capacity_mem_par_l1_P41: mem[36,51] <= 1536
 capacity_mem_par_l1_P41: mem[36,25] <= 1536
 capacity_mem_par_l1_P41: mem[36,118] <= 1536
 capacity_mem_par_l1_P41: mem[36,92] <= 1536
 capacity_mem_par_l1_P41: mem[36,58] <= 1536
 capacity_mem_par_l1_P41: mem[36,91] <= 1536
 capacity_mem_par_l1_P41: mem[36,33] <= 1536
 capacity_mem_par_l1_P41: mem[36,15] <= 1536
 capacity_mem_par_l1_P41: mem[36,21] <= 1536
 capacity_mem_par_l1_P41: mem[36,100] <= 1536
 capacity_mem_par_l1_P41: mem[36,66] <= 1536
 capacity_mem_par_l1_P41: mem[36,54] <= 1536
 capacity_mem_par_l1_P41: mem[36,28] <= 1536
 capacity_mem_par_l1_P41: mem[36,99] <= 1536
 capacity_mem_par_l1_P41: mem[36,73] <= 1536
 capacity_mem_par_l1_P41: mem[36,61] <= 1536
 capacity_mem_par_l1_P41: mem[36,106] <= 1536
 capacity_mem_par_l1_P41: mem[36,94] <= 1536
 capacity_mem_par_l1_P41: mem[36,36] <= 1536
 capacity_mem_par_l1_P41: mem[36,69] <= 1536
 capacity_mem_par_l1_P41: mem[36,9] <= 1536
 capacity_mem_par_l1_P41: mem[36,23] <= 1536
 capacity_mem_par_l1_P41: mem[36,102] <= 1536
 capacity_mem_par_l1_P41: mem[36,76] <= 1536
 capacity_mem_par_l1_P41: mem[36,42] <= 1536
 capacity_mem_par_l1_P41: mem[36,48] <= 1536
 capacity_mem_par_l1_P41: mem[36,30] <= 1536
 capacity_mem_par_l1_P41: mem[36,75] <= 1536
 capacity_mem_par_l1_P41: mem[36,81] <= 1536
 capacity_mem_par_l1_P41: mem[36,5] <= 1536
 capacity_mem_par_l1_P41: mem[36,126] <= 1536
 capacity_mem_par_l1_P41: mem[36,114] <= 1536
 capacity_mem_par_l1_P41: mem[36,88] <= 1536
 capacity_mem_par_l1_P41: mem[36,38] <= 1536
 capacity_mem_par_l1_P41: mem[36,12] <= 1536
 capacity_mem_par_l1_P41: mem[36,121] <= 1536
 capacity_mem_par_l1_P41: mem[36,45] <= 1536
 capacity_mem_par_l1_P41: mem[36,11] <= 1536
 capacity_mem_par_l1_P41: mem[36,17] <= 1536
 capacity_mem_par_l1_P41: mem[36,84] <= 1536
 capacity_mem_par_l1_P41: mem[36,24] <= 1536
 capacity_mem_par_l1_P41: mem[36,111] <= 1536
 capacity_mem_par_l1_P41: mem[36,117] <= 1536
 capacity_mem_par_l1_P41: mem[36,83] <= 1536
 capacity_mem_par_l1_P41: mem[36,90] <= 1536
 capacity_mem_par_l1_P41: mem[36,32] <= 1536
 capacity_mem_par_l1_P41: mem[36,14] <= 1536
 capacity_mem_par_l1_P41: mem[36,20] <= 1536
 capacity_mem_par_l1_P41: mem[36,65] <= 1536
 capacity_mem_par_l1_P41: mem[36,47] <= 1536
 capacity_mem_par_l1_P41: mem[36,19] <= 1536
 capacity_mem_par_l1_P41: mem[36,98] <= 1536
 capacity_mem_par_l1_P41: mem[36,72] <= 1536
 capacity_mem_par_l1_P41: mem[36,86] <= 1536
 capacity_mem_par_l1_P41: mem[36,60] <= 1536
 capacity_mem_par_l1_P41: mem[36,26] <= 1536
 capacity_mem_par_l1_P41: mem[36,105] <= 1536
 capacity_mem_par_l1_P41: mem[36,119] <= 1536
 capacity_mem_par_l1_P41: mem[36,93] <= 1536
 capacity_mem_par_l1_P41: mem[36,59] <= 1536
 row_capacity_l1_P42: rows_tot_l1_P42 <= 48
 capacity_mem_par_l1_P42: mem[37,82] <= 1536
 capacity_mem_par_l1_P42: mem[37,23] <= 1536
 capacity_mem_par_l1_P42: mem[37,113] <= 1536
 capacity_mem_par_l1_P42: mem[37,91] <= 1536
 capacity_mem_par_l1_P42: mem[37,16] <= 1536
 capacity_mem_par_l1_P42: mem[37,122] <= 1536
 capacity_mem_par_l1_P42: mem[37,68] <= 1536
 capacity_mem_par_l1_P42: mem[37,25] <= 1536
 capacity_mem_par_l1_P42: mem[37,99] <= 1536
 capacity_mem_par_l1_P42: mem[37,77] <= 1536
 capacity_mem_par_l1_P42: mem[37,56] <= 1536
 capacity_mem_par_l1_P42: mem[37,2] <= 1536
 capacity_mem_par_l1_P42: mem[37,108] <= 1536
 capacity_mem_par_l1_P42: mem[37,118] <= 1536
 capacity_mem_par_l1_P42: mem[37,80] <= 1536
 capacity_mem_par_l1_P42: mem[37,33] <= 1536
 capacity_mem_par_l1_P42: mem[37,11] <= 1536
 capacity_mem_par_l1_P42: mem[37,21] <= 1536
 capacity_mem_par_l1_P42: mem[37,89] <= 1536
 capacity_mem_par_l1_P42: mem[37,42] <= 1536
 capacity_mem_par_l1_P42: mem[37,52] <= 1536
 capacity_mem_par_l1_P42: mem[37,30] <= 1536
 capacity_mem_par_l1_P42: mem[37,120] <= 1536
 capacity_mem_par_l1_P42: mem[37,66] <= 1536
 capacity_mem_par_l1_P42: mem[37,7] <= 1536
 capacity_mem_par_l1_P42: mem[37,97] <= 1536
 capacity_mem_par_l1_P42: mem[37,75] <= 1536
 capacity_mem_par_l1_P42: mem[37,85] <= 1536
 capacity_mem_par_l1_P42: mem[37,38] <= 1536
 capacity_mem_par_l1_P42: mem[37,106] <= 1536
 capacity_mem_par_l1_P42: mem[37,94] <= 1536
 capacity_mem_par_l1_P42: mem[37,47] <= 1536
 capacity_mem_par_l1_P42: mem[37,9] <= 1536
 capacity_mem_par_l1_P42: mem[37,19] <= 1536
 capacity_mem_par_l1_P42: mem[37,28] <= 1536
 capacity_mem_par_l1_P42: mem[37,102] <= 1536
 capacity_mem_par_l1_P42: mem[37,64] <= 1536
 capacity_mem_par_l1_P42: mem[37,59] <= 1536
 capacity_mem_par_l1_P42: mem[37,5] <= 1536
 capacity_mem_par_l1_P42: mem[37,111] <= 1536
 capacity_mem_par_l1_P42: mem[37,73] <= 1536
 capacity_mem_par_l1_P42: mem[37,83] <= 1536
 capacity_mem_par_l1_P42: mem[37,36] <= 1536
 capacity_mem_par_l1_P42: mem[37,14] <= 1536
 capacity_mem_par_l1_P42: mem[37,114] <= 1536
 capacity_mem_par_l1_P42: mem[37,92] <= 1536
 capacity_mem_par_l1_P42: mem[37,45] <= 1536
 capacity_mem_par_l1_P42: mem[37,17] <= 1536
 capacity_mem_par_l1_P42: mem[37,69] <= 1536
 capacity_mem_par_l1_P42: mem[37,48] <= 1536
 capacity_mem_par_l1_P42: mem[37,26] <= 1536
 capacity_mem_par_l1_P42: mem[37,100] <= 1536
 capacity_mem_par_l1_P42: mem[37,3] <= 1536
 capacity_mem_par_l1_P42: mem[37,119] <= 1536
 capacity_mem_par_l1_P42: mem[37,81] <= 1536
 capacity_mem_par_l1_P42: mem[37,34] <= 1536
 capacity_mem_par_l1_P42: mem[37,12] <= 1536
 capacity_mem_par_l1_P42: mem[37,22] <= 1536
 capacity_mem_par_l1_P42: mem[37,43] <= 1536
 capacity_mem_par_l1_P42: mem[37,121] <= 1536
 capacity_mem_par_l1_P42: mem[37,62] <= 1536
 capacity_mem_par_l1_P42: mem[37,24] <= 1536
 capacity_mem_par_l1_P42: mem[37,98] <= 1536
 capacity_mem_par_l1_P42: mem[37,76] <= 1536
 capacity_mem_par_l1_P42: mem[37,86] <= 1536
 capacity_mem_par_l1_P42: mem[37,107] <= 1536
 capacity_mem_par_l1_P42: mem[37,117] <= 1536
 capacity_mem_par_l1_P42: mem[37,95] <= 1536
 capacity_mem_par_l1_P42: mem[37,32] <= 1536
 capacity_mem_par_l1_P42: mem[37,20] <= 1536
 capacity_mem_par_l1_P42: mem[37,126] <= 1536
 capacity_mem_par_l1_P42: mem[37,88] <= 1536
 capacity_mem_par_l1_P42: mem[37,51] <= 1536
 capacity_mem_par_l1_P42: mem[37,29] <= 1536
 capacity_mem_par_l1_P42: mem[37,103] <= 1536
 capacity_mem_par_l1_P42: mem[37,65] <= 1536
 capacity_mem_par_l1_P42: mem[37,60] <= 1536
 capacity_mem_par_l1_P42: mem[37,74] <= 1536
 capacity_mem_par_l1_P42: mem[37,84] <= 1536
 capacity_mem_par_l1_P42: mem[37,105] <= 1536
 capacity_mem_par_l1_P42: mem[37,115] <= 1536
 capacity_mem_par_l1_P42: mem[37,93] <= 1536
 capacity_mem_par_l1_P42: mem[37,46] <= 1536
 capacity_mem_par_l1_P42: mem[37,18] <= 1536
 capacity_mem_par_l1_P42: mem[37,101] <= 1536
 capacity_mem_par_l1_P42: mem[37,79] <= 1536
 capacity_mem_par_l1_P42: mem[37,58] <= 1536
 capacity_mem_par_l1_P42: mem[37,110] <= 1536
 capacity_mem_par_l1_P42: mem[37,72] <= 1536
 row_capacity_l2_P4: rows_tot_l2_P4 <= 48
 capacity_mem_par_l2_P4: mem[38,86] <= 1536
 capacity_mem_par_l2_P4: mem[38,119] <= 1536
 capacity_mem_par_l2_P4: mem[38,73] <= 1536
 capacity_mem_par_l2_P4: mem[38,5] <= 1536
 capacity_mem_par_l2_P4: mem[38,110] <= 1536
 capacity_mem_par_l2_P4: mem[38,64] <= 1536
 capacity_mem_par_l2_P4: mem[38,60] <= 1536
 capacity_mem_par_l2_P4: mem[38,22] <= 1536
 capacity_mem_par_l2_P4: mem[38,93] <= 1536
 capacity_mem_par_l2_P4: mem[38,9] <= 1536
 capacity_mem_par_l2_P4: mem[38,114] <= 1536
 capacity_mem_par_l2_P4: mem[38,84] <= 1536
 capacity_mem_par_l2_P4: mem[38,46] <= 1536
 capacity_mem_par_l2_P4: mem[38,26] <= 1536
 capacity_mem_par_l2_P4: mem[38,117] <= 1536
 capacity_mem_par_l2_P4: mem[38,79] <= 1536
 capacity_mem_par_l2_P4: mem[38,33] <= 1536
 capacity_mem_par_l2_P4: mem[38,59] <= 1536
 capacity_mem_par_l2_P4: mem[38,29] <= 1536
 capacity_mem_par_l2_P4: mem[38,108] <= 1536
 capacity_mem_par_l2_P4: mem[38,88] <= 1536
 capacity_mem_par_l2_P4: mem[38,20] <= 1536
 capacity_mem_par_l2_P4: mem[38,103] <= 1536
 capacity_mem_par_l2_P4: mem[38,121] <= 1536
 capacity_mem_par_l2_P4: mem[38,83] <= 1536
 capacity_mem_par_l2_P4: mem[38,74] <= 1536
 capacity_mem_par_l2_P4: mem[38,24] <= 1536
 capacity_mem_par_l2_P4: mem[38,107] <= 1536
 capacity_mem_par_l2_P4: mem[38,77] <= 1536
 capacity_mem_par_l2_P4: mem[38,19] <= 1536
 capacity_mem_par_l2_P4: mem[38,98] <= 1536
 capacity_mem_par_l2_P4: mem[38,68] <= 1536
 capacity_mem_par_l2_P4: mem[38,48] <= 1536
 capacity_mem_par_l2_P4: mem[38,81] <= 1536
 capacity_mem_par_l2_P4: mem[38,43] <= 1536
 capacity_mem_par_l2_P4: mem[38,118] <= 1536
 capacity_mem_par_l2_P4: mem[38,72] <= 1536
 capacity_mem_par_l2_P4: mem[38,34] <= 1536
 capacity_mem_par_l2_P4: mem[38,30] <= 1536
 capacity_mem_par_l2_P4: mem[38,105] <= 1536
 capacity_mem_par_l2_P4: mem[38,17] <= 1536
 capacity_mem_par_l2_P4: mem[38,92] <= 1536
 capacity_mem_par_l2_P4: mem[38,3] <= 1536
 capacity_mem_par_l2_P4: mem[38,32] <= 1536
 capacity_mem_par_l2_P4: mem[38,58] <= 1536
 capacity_mem_par_l2_P4: mem[38,28] <= 1536
 capacity_mem_par_l2_P4: mem[38,111] <= 1536
 capacity_mem_par_l2_P4: mem[38,65] <= 1536
 capacity_mem_par_l2_P4: mem[38,91] <= 1536
 capacity_mem_par_l2_P4: mem[38,23] <= 1536
 capacity_mem_par_l2_P4: mem[38,102] <= 1536
 capacity_mem_par_l2_P4: mem[38,120] <= 1536
 capacity_mem_par_l2_P4: mem[38,82] <= 1536
 capacity_mem_par_l2_P4: mem[38,52] <= 1536
 capacity_mem_par_l2_P4: mem[38,14] <= 1536
 capacity_mem_par_l2_P4: mem[38,115] <= 1536
 capacity_mem_par_l2_P4: mem[38,85] <= 1536
 capacity_mem_par_l2_P4: mem[38,47] <= 1536
 capacity_mem_par_l2_P4: mem[38,106] <= 1536
 capacity_mem_par_l2_P4: mem[38,76] <= 1536
 capacity_mem_par_l2_P4: mem[38,38] <= 1536
 capacity_mem_par_l2_P4: mem[38,56] <= 1536
 capacity_mem_par_l2_P4: mem[38,18] <= 1536
 capacity_mem_par_l2_P4: mem[38,89] <= 1536
 capacity_mem_par_l2_P4: mem[38,51] <= 1536
 capacity_mem_par_l2_P4: mem[38,21] <= 1536
 capacity_mem_par_l2_P4: mem[38,100] <= 1536
 capacity_mem_par_l2_P4: mem[38,126] <= 1536
 capacity_mem_par_l2_P4: mem[38,42] <= 1536
 capacity_mem_par_l2_P4: mem[38,12] <= 1536
 capacity_mem_par_l2_P4: mem[38,113] <= 1536
 capacity_mem_par_l2_P4: mem[38,45] <= 1536
 capacity_mem_par_l2_P4: mem[38,25] <= 1536
 capacity_mem_par_l2_P4: mem[38,36] <= 1536
 capacity_mem_par_l2_P4: mem[38,62] <= 1536
 capacity_mem_par_l2_P4: mem[38,16] <= 1536
 capacity_mem_par_l2_P4: mem[38,99] <= 1536
 capacity_mem_par_l2_P4: mem[38,69] <= 1536
 capacity_mem_par_l2_P4: mem[38,95] <= 1536
 capacity_mem_par_l2_P4: mem[38,11] <= 1536
 ns_single_CPU1: 3.5 rows_tot_CPU1 - ns_single_CPU1 + pdt_m_rows_CPU1 = 0
 ns_sketch_CPU1: - ns_single_CPU1 + 10.71434571428571 cores_sketch_CPU1
   >= 0
 capacity_cores_CPU1: cores_sketch_CPU1 <= 1
 ns_single_CPU2: 3.5 rows_tot_CPU2 - ns_single_CPU2 + pdt_m_rows_CPU2 = 0
 ns_sketch_CPU2: - ns_single_CPU2 + 10.71434571428571 cores_sketch_CPU2
   >= 0
 capacity_cores_CPU2: cores_sketch_CPU2 <= 1
 ns_single_CPU3: 3.5 rows_tot_CPU3 - ns_single_CPU3 + pdt_m_rows_CPU3 = 0
 ns_sketch_CPU3: - ns_single_CPU3 + 10.71434571428571 cores_sketch_CPU3
   >= 0
 capacity_cores_CPU3: cores_sketch_CPU3 <= 1
 ns_single_CPU4: 3.5 rows_tot_CPU4 - ns_single_CPU4 + pdt_m_rows_CPU4 = 0
 ns_sketch_CPU4: - ns_single_CPU4 + 10.71434571428571 cores_sketch_CPU4
   >= 0
 capacity_cores_CPU4: cores_sketch_CPU4 <= 1
 ns_single_CPU5: 3.5 rows_tot_CPU5 - ns_single_CPU5 + pdt_m_rows_CPU5 = 0
 ns_sketch_CPU5: - ns_single_CPU5 + 10.71434571428571 cores_sketch_CPU5
   >= 0
 capacity_cores_CPU5: cores_sketch_CPU5 <= 1
 ns_single_CPU6: 3.5 rows_tot_CPU6 - ns_single_CPU6 + pdt_m_rows_CPU6 = 0
 ns_sketch_CPU6: - ns_single_CPU6 + 10.71434571428571 cores_sketch_CPU6
   >= 0
 capacity_cores_CPU6: cores_sketch_CPU6 <= 1
 ns_single_CPU7: 3.5 rows_tot_CPU7 - ns_single_CPU7 + pdt_m_rows_CPU7 = 0
 ns_sketch_CPU7: - ns_single_CPU7 + 10.71434571428571 cores_sketch_CPU7
   >= 0
 capacity_cores_CPU7: cores_sketch_CPU7 <= 1
 ns_single_CPU8: 3.5 rows_tot_CPU8 - ns_single_CPU8 + pdt_m_rows_CPU8 = 0
 ns_sketch_CPU8: - ns_single_CPU8 + 10.71434571428571 cores_sketch_CPU8
   >= 0
 capacity_cores_CPU8: cores_sketch_CPU8 <= 1
 ns_single_CPU9: 3.5 rows_tot_CPU9 - ns_single_CPU9 + pdt_m_rows_CPU9 = 0
 ns_sketch_CPU9: - ns_single_CPU9 + 10.71434571428571 cores_sketch_CPU9
   >= 0
 capacity_cores_CPU9: cores_sketch_CPU9 <= 1
 ns_single_CPU10: 3.5 rows_tot_CPU10 - ns_single_CPU10 + pdt_m_rows_CPU10
   = 0
 ns_sketch_CPU10: - ns_single_CPU10 + 10.71434571428571 cores_sketch_CPU10
   >= 0
 capacity_cores_CPU10: cores_sketch_CPU10 <= 1
 ns_single_CPU11: 3.5 rows_tot_CPU11 - ns_single_CPU11 + pdt_m_rows_CPU11
   = 0
 ns_sketch_CPU11: - ns_single_CPU11 + 10.71434571428571 cores_sketch_CPU11
   >= 0
 capacity_cores_CPU11: cores_sketch_CPU11 <= 1
 ns_single_CPU12: 3.5 rows_tot_CPU12 - ns_single_CPU12 + pdt_m_rows_CPU12
   = 0
 ns_sketch_CPU12: - ns_single_CPU12 + 10.71434571428571 cores_sketch_CPU12
   >= 0
 capacity_cores_CPU12: cores_sketch_CPU12 <= 1
 ns_single_CPU13: 3.5 rows_tot_CPU13 - ns_single_CPU13 + pdt_m_rows_CPU13
   = 0
 ns_sketch_CPU13: - ns_single_CPU13 + 10.71434571428571 cores_sketch_CPU13
   >= 0
 capacity_cores_CPU13: cores_sketch_CPU13 <= 1
 ns_single_CPU14: 3.5 rows_tot_CPU14 - ns_single_CPU14 + pdt_m_rows_CPU14
   = 0
 ns_sketch_CPU14: - ns_single_CPU14 + 10.71434571428571 cores_sketch_CPU14
   >= 0
 capacity_cores_CPU14: cores_sketch_CPU14 <= 1
 ns_single_CPU15: 3.5 rows_tot_CPU15 - ns_single_CPU15 + pdt_m_rows_CPU15
   = 0
 ns_sketch_CPU15: - ns_single_CPU15 + 10.71434571428571 cores_sketch_CPU15
   >= 0
 capacity_cores_CPU15: cores_sketch_CPU15 <= 1
 ns_single_CPU16: 3.5 rows_tot_CPU16 - ns_single_CPU16 + pdt_m_rows_CPU16
   = 0
 ns_sketch_CPU16: - ns_single_CPU16 + 10.71434571428571 cores_sketch_CPU16
   >= 0
 capacity_cores_CPU16: cores_sketch_CPU16 <= 1
 ns_single_CPU17: 3.5 rows_tot_CPU17 - ns_single_CPU17 + pdt_m_rows_CPU17
   = 0
 ns_sketch_CPU17: - ns_single_CPU17 + 10.71434571428571 cores_sketch_CPU17
   >= 0
 capacity_cores_CPU17: cores_sketch_CPU17 <= 1
 ns_single_CPU18: 3.5 rows_tot_CPU18 - ns_single_CPU18 + pdt_m_rows_CPU18
   = 0
 ns_sketch_CPU18: - ns_single_CPU18 + 10.71434571428571 cores_sketch_CPU18
   >= 0
 capacity_cores_CPU18: cores_sketch_CPU18 <= 1
 ns_single_CPU19: 3.5 rows_tot_CPU19 - ns_single_CPU19 + pdt_m_rows_CPU19
   = 0
 ns_sketch_CPU19: - ns_single_CPU19 + 10.71434571428571 cores_sketch_CPU19
   >= 0
 capacity_cores_CPU19: cores_sketch_CPU19 <= 1
 ns_single_CPU20: 3.5 rows_tot_CPU20 - ns_single_CPU20 + pdt_m_rows_CPU20
   = 0
 ns_sketch_CPU20: - ns_single_CPU20 + 10.71434571428571 cores_sketch_CPU20
   >= 0
 capacity_cores_CPU20: cores_sketch_CPU20 <= 1
 ns_single_CPU21: 3.5 rows_tot_CPU21 - ns_single_CPU21 + pdt_m_rows_CPU21
   = 0
 ns_sketch_CPU21: - ns_single_CPU21 + 10.71434571428571 cores_sketch_CPU21
   >= 0
 capacity_cores_CPU21: cores_sketch_CPU21 <= 1
 ns_single_CPU22: 3.5 rows_tot_CPU22 - ns_single_CPU22 + pdt_m_rows_CPU22
   = 0
 ns_sketch_CPU22: - ns_single_CPU22 + 10.71434571428571 cores_sketch_CPU22
   >= 0
 capacity_cores_CPU22: cores_sketch_CPU22 <= 1
 ns_single_CPU23: 3.5 rows_tot_CPU23 - ns_single_CPU23 + pdt_m_rows_CPU23
   = 0
 ns_sketch_CPU23: - ns_single_CPU23 + 10.71434571428571 cores_sketch_CPU23
   >= 0
 capacity_cores_CPU23: cores_sketch_CPU23 <= 1
 ns_single_CPU24: 3.5 rows_tot_CPU24 - ns_single_CPU24 + pdt_m_rows_CPU24
   = 0
 ns_sketch_CPU24: - ns_single_CPU24 + 10.71434571428571 cores_sketch_CPU24
   >= 0
 capacity_cores_CPU24: cores_sketch_CPU24 <= 1
 ns_single_CPU25: 3.5 rows_tot_CPU25 - ns_single_CPU25 + pdt_m_rows_CPU25
   = 0
 ns_sketch_CPU25: - ns_single_CPU25 + 10.71434571428571 cores_sketch_CPU25
   >= 0
 capacity_cores_CPU25: cores_sketch_CPU25 <= 1
 ns_single_CPU26: 3.5 rows_tot_CPU26 - ns_single_CPU26 + pdt_m_rows_CPU26
   = 0
 ns_sketch_CPU26: - ns_single_CPU26 + 10.71434571428571 cores_sketch_CPU26
   >= 0
 capacity_cores_CPU26: cores_sketch_CPU26 <= 1
 ns_single_CPU27: 3.5 rows_tot_CPU27 - ns_single_CPU27 + pdt_m_rows_CPU27
   = 0
 ns_sketch_CPU27: - ns_single_CPU27 + 10.71434571428571 cores_sketch_CPU27
   >= 0
 capacity_cores_CPU27: cores_sketch_CPU27 <= 1
 ns_single_CPU28: 3.5 rows_tot_CPU28 - ns_single_CPU28 + pdt_m_rows_CPU28
   = 0
 ns_sketch_CPU28: - ns_single_CPU28 + 10.71434571428571 cores_sketch_CPU28
   >= 0
 capacity_cores_CPU28: cores_sketch_CPU28 <= 1
 ns_single_CPU29: 3.5 rows_tot_CPU29 - ns_single_CPU29 + pdt_m_rows_CPU29
   = 0
 ns_sketch_CPU29: - ns_single_CPU29 + 10.71434571428571 cores_sketch_CPU29
   >= 0
 capacity_cores_CPU29: cores_sketch_CPU29 <= 1
 ns_single_CPU30: 3.5 rows_tot_CPU30 - ns_single_CPU30 + pdt_m_rows_CPU30
   = 0
 ns_sketch_CPU30: - ns_single_CPU30 + 10.71434571428571 cores_sketch_CPU30
   >= 0
 capacity_cores_CPU30: cores_sketch_CPU30 <= 1
 ns_single_CPU31: 3.5 rows_tot_CPU31 - ns_single_CPU31 + pdt_m_rows_CPU31
   = 0
 ns_sketch_CPU31: - ns_single_CPU31 + 10.71434571428571 cores_sketch_CPU31
   >= 0
 capacity_cores_CPU31: cores_sketch_CPU31 <= 1
 ns_single_CPU32: 3.5 rows_tot_CPU32 - ns_single_CPU32 + pdt_m_rows_CPU32
   = 0
 ns_sketch_CPU32: - ns_single_CPU32 + 10.71434571428571 cores_sketch_CPU32
   >= 0
 capacity_cores_CPU32: cores_sketch_CPU32 <= 1
 res_acc: 1.220703125e-04 mem_tot_CPU1 + 1.220703125e-04 mem_tot_CPU2
   + 1.220703125e-04 mem_tot_CPU3 + 1.220703125e-04 mem_tot_CPU4
   + 1.220703125e-04 mem_tot_CPU5 + 1.220703125e-04 mem_tot_CPU6
   + 1.220703125e-04 mem_tot_CPU7 + 1.220703125e-04 mem_tot_CPU8
   + 1.220703125e-04 mem_tot_CPU9 + 1.220703125e-04 mem_tot_CPU10
   + 1.220703125e-04 mem_tot_CPU11 + 1.220703125e-04 mem_tot_CPU12
   + 1.220703125e-04 mem_tot_CPU13 + 1.220703125e-04 mem_tot_CPU14
   + 1.220703125e-04 mem_tot_CPU15 + 1.220703125e-04 mem_tot_CPU16
   + 1.220703125e-04 mem_tot_CPU17 + 1.220703125e-04 mem_tot_CPU18
   + 1.220703125e-04 mem_tot_CPU19 + 1.220703125e-04 mem_tot_CPU20
   + 1.220703125e-04 mem_tot_CPU21 + 1.220703125e-04 mem_tot_CPU22
   + 1.220703125e-04 mem_tot_CPU23 + 1.220703125e-04 mem_tot_CPU24
   + 1.220703125e-04 mem_tot_CPU25 + 1.220703125e-04 mem_tot_CPU26
   + 1.220703125e-04 mem_tot_CPU27 + 1.220703125e-04 mem_tot_CPU28
   + 1.220703125e-04 mem_tot_CPU29 + 1.220703125e-04 mem_tot_CPU30
   + 1.220703125e-04 mem_tot_CPU31 + 1.220703125e-04 mem_tot_CPU32
   + 0.001953125 mem_tot_tor_P41 + 0.25 rows_tot_tor_P41
   + 0.001953125 mem_tot_tor_P42 + 0.25 rows_tot_tor_P42
   + 0.001953125 mem_tot_tor_P43 + 0.25 rows_tot_tor_P43
   + 0.001953125 mem_tot_tor_P44 + 0.25 rows_tot_tor_P44
   + 0.001953125 mem_tot_l1_P41 + 0.25 rows_tot_l1_P41
   + 0.001953125 mem_tot_l1_P42 + 0.25 rows_tot_l1_P42
   + 0.001953125 mem_tot_l2_P4 + 0.25 rows_tot_l2_P4 + 10 cores_sketch_CPU1
   + 10 cores_sketch_CPU2 + 10 cores_sketch_CPU3 + 10 cores_sketch_CPU4
   + 10 cores_sketch_CPU5 + 10 cores_sketch_CPU6 + 10 cores_sketch_CPU7
   + 10 cores_sketch_CPU8 + 10 cores_sketch_CPU9 + 10 cores_sketch_CPU10
   + 10 cores_sketch_CPU11 + 10 cores_sketch_CPU12 + 10 cores_sketch_CPU13
   + 10 cores_sketch_CPU14 + 10 cores_sketch_CPU15 + 10 cores_sketch_CPU16
   + 10 cores_sketch_CPU17 + 10 cores_sketch_CPU18 + 10 cores_sketch_CPU19
   + 10 cores_sketch_CPU20 + 10 cores_sketch_CPU21 + 10 cores_sketch_CPU22
   + 10 cores_sketch_CPU23 + 10 cores_sketch_CPU24 + 10 cores_sketch_CPU25
   + 10 cores_sketch_CPU26 + 10 cores_sketch_CPU27 + 10 cores_sketch_CPU28
   + 10 cores_sketch_CPU29 + 10 cores_sketch_CPU30 + 10 cores_sketch_CPU31
   + 10 cores_sketch_CPU32 - res = -1920
 pdt_m_rows_CPU1: pdt_m_rows_CPU1 + [ - rows_tot_CPU1 * m_access_time_CPU1
   ] = 0
 pdt_m_rows_CPU2: pdt_m_rows_CPU2 + [ - rows_tot_CPU2 * m_access_time_CPU2
   ] = 0
 pdt_m_rows_CPU3: pdt_m_rows_CPU3 + [ - rows_tot_CPU3 * m_access_time_CPU3
   ] = 0
 pdt_m_rows_CPU4: pdt_m_rows_CPU4 + [ - rows_tot_CPU4 * m_access_time_CPU4
   ] = 0
 pdt_m_rows_CPU5: pdt_m_rows_CPU5 + [ - rows_tot_CPU5 * m_access_time_CPU5
   ] = 0
 pdt_m_rows_CPU6: pdt_m_rows_CPU6 + [ - rows_tot_CPU6 * m_access_time_CPU6
   ] = 0
 pdt_m_rows_CPU7: pdt_m_rows_CPU7 + [ - rows_tot_CPU7 * m_access_time_CPU7
   ] = 0
 pdt_m_rows_CPU8: pdt_m_rows_CPU8 + [ - rows_tot_CPU8 * m_access_time_CPU8
   ] = 0
 pdt_m_rows_CPU9: pdt_m_rows_CPU9 + [ - rows_tot_CPU9 * m_access_time_CPU9
   ] = 0
 pdt_m_rows_CPU10: pdt_m_rows_CPU10 + [
   - rows_tot_CPU10 * m_access_time_CPU10 ] = 0
 pdt_m_rows_CPU11: pdt_m_rows_CPU11 + [
   - rows_tot_CPU11 * m_access_time_CPU11 ] = 0
 pdt_m_rows_CPU12: pdt_m_rows_CPU12 + [
   - rows_tot_CPU12 * m_access_time_CPU12 ] = 0
 pdt_m_rows_CPU13: pdt_m_rows_CPU13 + [
   - rows_tot_CPU13 * m_access_time_CPU13 ] = 0
 pdt_m_rows_CPU14: pdt_m_rows_CPU14 + [
   - rows_tot_CPU14 * m_access_time_CPU14 ] = 0
 pdt_m_rows_CPU15: pdt_m_rows_CPU15 + [
   - rows_tot_CPU15 * m_access_time_CPU15 ] = 0
 pdt_m_rows_CPU16: pdt_m_rows_CPU16 + [
   - rows_tot_CPU16 * m_access_time_CPU16 ] = 0
 pdt_m_rows_CPU17: pdt_m_rows_CPU17 + [
   - rows_tot_CPU17 * m_access_time_CPU17 ] = 0
 pdt_m_rows_CPU18: pdt_m_rows_CPU18 + [
   - rows_tot_CPU18 * m_access_time_CPU18 ] = 0
 pdt_m_rows_CPU19: pdt_m_rows_CPU19 + [
   - rows_tot_CPU19 * m_access_time_CPU19 ] = 0
 pdt_m_rows_CPU20: pdt_m_rows_CPU20 + [
   - rows_tot_CPU20 * m_access_time_CPU20 ] = 0
 pdt_m_rows_CPU21: pdt_m_rows_CPU21 + [
   - rows_tot_CPU21 * m_access_time_CPU21 ] = 0
 pdt_m_rows_CPU22: pdt_m_rows_CPU22 + [
   - rows_tot_CPU22 * m_access_time_CPU22 ] = 0
 pdt_m_rows_CPU23: pdt_m_rows_CPU23 + [
   - rows_tot_CPU23 * m_access_time_CPU23 ] = 0
 pdt_m_rows_CPU24: pdt_m_rows_CPU24 + [
   - rows_tot_CPU24 * m_access_time_CPU24 ] = 0
 pdt_m_rows_CPU25: pdt_m_rows_CPU25 + [
   - rows_tot_CPU25 * m_access_time_CPU25 ] = 0
 pdt_m_rows_CPU26: pdt_m_rows_CPU26 + [
   - rows_tot_CPU26 * m_access_time_CPU26 ] = 0
 pdt_m_rows_CPU27: pdt_m_rows_CPU27 + [
   - rows_tot_CPU27 * m_access_time_CPU27 ] = 0
 pdt_m_rows_CPU28: pdt_m_rows_CPU28 + [
   - rows_tot_CPU28 * m_access_time_CPU28 ] = 0
 pdt_m_rows_CPU29: pdt_m_rows_CPU29 + [
   - rows_tot_CPU29 * m_access_time_CPU29 ] = 0
 pdt_m_rows_CPU30: pdt_m_rows_CPU30 + [
   - rows_tot_CPU30 * m_access_time_CPU30 ] = 0
 pdt_m_rows_CPU31: pdt_m_rows_CPU31 + [
   - rows_tot_CPU31 * m_access_time_CPU31 ] = 0
 pdt_m_rows_CPU32: pdt_m_rows_CPU32 + [
   - rows_tot_CPU32 * m_access_time_CPU32 ] = 0
Bounds
 frac[0,90] <= 1
 frac[0,92] <= 1
 frac[0,83] <= 1
 frac[0,75] <= 1
 frac[0,89] <= 1
 frac[0,80] <= 1
 frac[0,77] <= 1
 frac[0,82] <= 1
 frac[0,72] <= 1
 frac[0,79] <= 1
 frac[0,93] <= 1
 frac[0,84] <= 1
 frac[0,74] <= 1
 frac[0,65] <= 1
 frac[0,88] <= 1
 frac[0,95] <= 1
 frac[1,122] <= 1
 frac[1,103] <= 1
 frac[1,120] <= 1
 frac[1,113] <= 1
 frac[1,107] <= 1
 frac[1,108] <= 1
 frac[1,101] <= 1
 frac[1,119] <= 1
 frac[1,105] <= 1
 frac[1,98] <= 1
 frac[1,117] <= 1
 frac[1,96] <= 1
 frac[1,114] <= 1
 frac[1,99] <= 1
 frac[1,100] <= 1
 frac[2,76] <= 1
 frac[2,71] <= 1
 frac[2,94] <= 1
 frac[2,80] <= 1
 frac[2,74] <= 1
 frac[2,92] <= 1
 frac[2,15] <= 1
 frac[2,88] <= 1
 frac[2,68] <= 1
 frac[2,66] <= 1
 frac[2,93] <= 1
 frac[3,103] <= 1
 frac[3,114] <= 1
 frac[3,96] <= 1
 frac[3,105] <= 1
 frac[3,110] <= 1
 frac[3,32] <= 1
 frac[3,98] <= 1
 frac[3,100] <= 1
 frac[3,115] <= 1
 frac[3,120] <= 1
 frac[3,102] <= 1
 frac[3,117] <= 1
 frac[3,99] <= 1
 frac[4,5] <= 1
 frac[4,11] <= 1
 frac[4,17] <= 1
 frac[4,121] <= 1
 frac[4,22] <= 1
 frac[4,3] <= 1
 frac[4,16] <= 1
 frac[4,18] <= 1
 frac[4,25] <= 1
 frac[4,28] <= 1
 frac[4,30] <= 1
 frac[5,93] <= 1
 frac[5,72] <= 1
 frac[5,68] <= 1
 frac[5,77] <= 1
 frac[5,80] <= 1
 frac[5,89] <= 1
 frac[5,85] <= 1
 frac[5,71] <= 1
 frac[5,64] <= 1
 frac[5,90] <= 1
 frac[5,86] <= 1
 frac[5,88] <= 1
 frac[5,12] <= 1
 frac[5,65] <= 1
 frac[6,28] <= 1
 frac[6,14] <= 1
 frac[6,18] <= 1
 frac[6,21] <= 1
 frac[6,11] <= 1
 frac[6,5] <= 1
 frac[6,22] <= 1
 frac[6,9] <= 1
 frac[6,29] <= 1
 frac[6,20] <= 1
 frac[6,24] <= 1
 frac[6,10] <= 1
 frac[6,39] <= 1
 frac[6,30] <= 1
 frac[6,17] <= 1
 frac[6,107] <= 1
 frac[7,55] <= 1
 frac[7,91] <= 1
 frac[7,34] <= 1
 frac[7,56] <= 1
 frac[7,49] <= 1
 frac[7,39] <= 1
 frac[7,61] <= 1
 frac[7,45] <= 1
 frac[7,88] <= 1
 frac[7,54] <= 1
 frac[7,47] <= 1
 frac[7,48] <= 1
 frac[7,62] <= 1
 frac[8,38] <= 1
 frac[8,18] <= 1
 frac[8,14] <= 1
 frac[8,20] <= 1
 frac[8,27] <= 1
 frac[8,22] <= 1
 frac[8,29] <= 1
 frac[8,17] <= 1
 frac[8,24] <= 1
 frac[8,31] <= 1
 frac[8,19] <= 1
 frac[8,26] <= 1
 frac[8,21] <= 1
 frac[8,28] <= 1
 frac[8,10] <= 1
 frac[8,30] <= 1
 frac[9,62] <= 1
 frac[9,39] <= 1
 frac[9,59] <= 1
 frac[9,60] <= 1
 frac[9,48] <= 1
 frac[9,57] <= 1
 frac[9,52] <= 1
 frac[9,61] <= 1
 frac[9,47] <= 1
 frac[9,49] <= 1
 frac[9,58] <= 1
 frac[9,35] <= 1
 frac[9,36] <= 1
 frac[9,45] <= 1
 frac[9,56] <= 1
 frac[9,11] <= 1
 frac[9,42] <= 1
 frac[10,75] <= 1
 frac[10,73] <= 1
 frac[10,69] <= 1
 frac[10,76] <= 1
 frac[10,81] <= 1
 frac[10,74] <= 1
 frac[10,95] <= 1
 frac[10,65] <= 1
 frac[10,72] <= 1
 frac[10,93] <= 1
 frac[10,101] <= 1
 frac[10,77] <= 1
 frac[10,94] <= 1
 frac[11,46] <= 1
 frac[11,45] <= 1
 frac[11,34] <= 1
 frac[11,56] <= 1
 frac[11,47] <= 1
 frac[11,36] <= 1
 frac[11,51] <= 1
 frac[11,33] <= 1
 frac[11,42] <= 1
 frac[11,35] <= 1
 frac[11,57] <= 1
 frac[11,62] <= 1
 frac[12,59] <= 1
 frac[12,111] <= 1
 frac[12,102] <= 1
 frac[12,105] <= 1
 frac[12,126] <= 1
 frac[12,117] <= 1
 frac[12,107] <= 1
 frac[12,120] <= 1
 frac[12,110] <= 1
 frac[12,103] <= 1
 frac[12,106] <= 1
 frac[12,118] <= 1
 frac[12,100] <= 1
 frac[12,99] <= 1
 frac[12,121] <= 1
 frac[13,89] <= 1
 frac[13,68] <= 1
 frac[13,73] <= 1
 frac[13,66] <= 1
 frac[13,92] <= 1
 frac[13,90] <= 1
 frac[13,88] <= 1
 frac[13,5] <= 1
 frac[13,93] <= 1
 frac[13,72] <= 1
 frac[13,77] <= 1
 frac[14,43] <= 1
 frac[14,59] <= 1
 frac[14,36] <= 1
 frac[14,54] <= 1
 frac[14,42] <= 1
 frac[14,52] <= 1
 frac[14,33] <= 1
 frac[14,39] <= 1
 frac[14,46] <= 1
 frac[14,122] <= 1
 frac[14,56] <= 1
 frac[14,55] <= 1
 frac[15,30] <= 1
 frac[15,34] <= 1
 frac[15,15] <= 1
 frac[15,20] <= 1
 frac[15,26] <= 1
 frac[15,9] <= 1
 frac[15,22] <= 1
 frac[15,31] <= 1
 frac[15,5] <= 1
 frac[15,11] <= 1
 frac[15,25] <= 1
 frac[15,12] <= 1
 frac[15,18] <= 1
 frac[15,27] <= 1
 frac[15,14] <= 1
 frac[15,23] <= 1
 frac[15,28] <= 1
 frac[16,29] <= 1
 frac[16,126] <= 1
 frac[16,2] <= 1
 frac[16,24] <= 1
 frac[16,72] <= 1
 frac[16,26] <= 1
 frac[16,28] <= 1
 frac[16,1] <= 1
 frac[16,30] <= 1
 frac[16,52] <= 1
 frac[16,25] <= 1
 frac[16,23] <= 1
 frac[16,14] <= 1
 frac[16,18] <= 1
 frac[17,117] <= 1
 frac[17,103] <= 1
 frac[17,121] <= 1
 frac[17,101] <= 1
 frac[17,105] <= 1
 frac[17,98] <= 1
 frac[17,118] <= 1
 frac[17,122] <= 1
 frac[17,115] <= 1
 frac[17,102] <= 1
 frac[17,120] <= 1
 frac[17,99] <= 1
 frac[17,100] <= 1
 frac[17,126] <= 1
 frac[18,62] <= 1
 frac[18,48] <= 1
 frac[18,34] <= 1
 frac[18,82] <= 1
 frac[18,58] <= 1
 frac[18,51] <= 1
 frac[18,42] <= 1
 frac[18,47] <= 1
 frac[18,38] <= 1
 frac[18,33] <= 1
 frac[18,59] <= 1
 frac[19,84] <= 1
 frac[19,72] <= 1
 frac[19,81] <= 1
 frac[19,86] <= 1
 frac[19,95] <= 1
 frac[19,69] <= 1
 frac[19,126] <= 1
 frac[19,74] <= 1
 frac[19,88] <= 1
 frac[19,73] <= 1
 frac[19,92] <= 1
 frac[19,66] <= 1
 frac[19,75] <= 1
 frac[19,80] <= 1
 frac[19,89] <= 1
 frac[19,94] <= 1
 frac[19,77] <= 1
 frac[20,118] <= 1
 frac[20,23] <= 1
 frac[20,105] <= 1
 frac[20,108] <= 1
 frac[20,114] <= 1
 frac[20,110] <= 1
 frac[20,117] <= 1
 frac[20,97] <= 1
 frac[20,126] <= 1
 frac[20,99] <= 1
 frac[20,106] <= 1
 frac[20,120] <= 1
 frac[21,26] <= 1
 frac[21,22] <= 1
 frac[21,11] <= 1
 frac[21,20] <= 1
 frac[21,29] <= 1
 frac[21,7] <= 1
 frac[21,14] <= 1
 frac[21,23] <= 1
 frac[21,16] <= 1
 frac[21,12] <= 1
 frac[21,21] <= 1
 frac[21,30] <= 1
 frac[22,12] <= 1
 frac[22,30] <= 1
 frac[22,17] <= 1
 frac[22,11] <= 1
 frac[22,5] <= 1
 frac[22,28] <= 1
 frac[22,9] <= 1
 frac[22,25] <= 1
 frac[22,62] <= 1
 frac[22,16] <= 1
 frac[22,3] <= 1
 frac[22,20] <= 1
 frac[22,1] <= 1
 frac[22,24] <= 1
 frac[23,120] <= 1
 frac[23,113] <= 1
 frac[23,100] <= 1
 frac[23,107] <= 1
 frac[23,121] <= 1
 frac[23,98] <= 1
 frac[23,110] <= 1
 frac[24,92] <= 1
 frac[24,14] <= 1
 frac[24,21] <= 1
 frac[24,2] <= 1
 frac[24,16] <= 1
 frac[24,11] <= 1
 frac[24,18] <= 1
 frac[24,25] <= 1
 frac[24,22] <= 1
 frac[24,29] <= 1
 frac[24,17] <= 1
 frac[24,19] <= 1
 frac[25,77] <= 1
 frac[25,107] <= 1
 frac[25,117] <= 1
 frac[25,123] <= 1
 frac[25,102] <= 1
 frac[25,118] <= 1
 frac[26,92] <= 1
 frac[26,72] <= 1
 frac[26,68] <= 1
 frac[26,123] <= 1
 frac[26,88] <= 1
 frac[26,66] <= 1
 frac[26,77] <= 1
 frac[26,17] <= 1
 frac[26,73] <= 1
 frac[26,80] <= 1
 frac[26,76] <= 1
 frac[27,60] <= 1
 frac[27,43] <= 1
 frac[27,45] <= 1
 frac[27,34] <= 1
 frac[27,59] <= 1
 frac[27,47] <= 1
 frac[27,36] <= 1
 frac[27,40] <= 1
 frac[27,35] <= 1
 frac[27,49] <= 1
 frac[27,51] <= 1
 frac[28,77] <= 1
 frac[28,107] <= 1
 frac[28,79] <= 1
 frac[28,93] <= 1
 frac[28,83] <= 1
 frac[28,64] <= 1
 frac[28,76] <= 1
 frac[28,75] <= 1
 frac[28,66] <= 1
 frac[28,91] <= 1
 frac[28,82] <= 1
 frac[28,94] <= 1
 frac[28,85] <= 1
 frac[28,7] <= 1
 frac[28,65] <= 1
 frac[29,48] <= 1
 frac[29,42] <= 1
 frac[29,35] <= 1
 frac[29,40] <= 1
 frac[29,33] <= 1
 frac[29,51] <= 1
 frac[29,38] <= 1
 frac[29,63] <= 1
 frac[29,56] <= 1
 frac[29,59] <= 1
 frac[29,46] <= 1
 frac[30,51] <= 1
 frac[30,36] <= 1
 frac[30,89] <= 1
 frac[30,49] <= 1
 frac[30,56] <= 1
 frac[30,62] <= 1
 frac[30,46] <= 1
 frac[30,59] <= 1
 frac[30,34] <= 1
 frac[30,32] <= 1
 frac[30,52] <= 1
 frac[30,105] <= 1
 frac[30,63] <= 1
 frac[30,38] <= 1
 frac[31,126] <= 1
 frac[31,120] <= 1
 frac[31,97] <= 1
 frac[31,110] <= 1
 frac[31,122] <= 1
 frac[31,118] <= 1
 frac[31,106] <= 1
 frac[31,47] <= 1
 frac[31,111] <= 1
 frac[31,117] <= 1
 frac[31,119] <= 1
 frac[31,101] <= 1
 frac[31,98] <= 1
 mem_tot_CPU1 <= 32768
 normalized_mem_tot_CPU1 <= 1
 normalized_rows_tot_CPU1 <= 1
 mem_tot_CPU2 <= 32768
 normalized_mem_tot_CPU2 <= 1
 normalized_rows_tot_CPU2 <= 1
 mem_tot_CPU3 <= 32768
 normalized_mem_tot_CPU3 <= 1
 normalized_rows_tot_CPU3 <= 1
 mem_tot_CPU4 <= 32768
 normalized_mem_tot_CPU4 <= 1
 normalized_rows_tot_CPU4 <= 1
 mem_tot_CPU5 <= 32768
 normalized_mem_tot_CPU5 <= 1
 normalized_rows_tot_CPU5 <= 1
 mem_tot_CPU6 <= 32768
 normalized_mem_tot_CPU6 <= 1
 normalized_rows_tot_CPU6 <= 1
 mem_tot_CPU7 <= 32768
 normalized_mem_tot_CPU7 <= 1
 normalized_rows_tot_CPU7 <= 1
 mem_tot_CPU8 <= 32768
 normalized_mem_tot_CPU8 <= 1
 normalized_rows_tot_CPU8 <= 1
 mem_tot_CPU9 <= 32768
 normalized_mem_tot_CPU9 <= 1
 normalized_rows_tot_CPU9 <= 1
 mem_tot_CPU10 <= 32768
 normalized_mem_tot_CPU10 <= 1
 normalized_rows_tot_CPU10 <= 1
 mem_tot_CPU11 <= 32768
 normalized_mem_tot_CPU11 <= 1
 normalized_rows_tot_CPU11 <= 1
 mem_tot_CPU12 <= 32768
 normalized_mem_tot_CPU12 <= 1
 normalized_rows_tot_CPU12 <= 1
 mem_tot_CPU13 <= 32768
 normalized_mem_tot_CPU13 <= 1
 normalized_rows_tot_CPU13 <= 1
 mem_tot_CPU14 <= 32768
 normalized_mem_tot_CPU14 <= 1
 normalized_rows_tot_CPU14 <= 1
 mem_tot_CPU15 <= 32768
 normalized_mem_tot_CPU15 <= 1
 normalized_rows_tot_CPU15 <= 1
 mem_tot_CPU16 <= 32768
 normalized_mem_tot_CPU16 <= 1
 normalized_rows_tot_CPU16 <= 1
 mem_tot_CPU17 <= 32768
 normalized_mem_tot_CPU17 <= 1
 normalized_rows_tot_CPU17 <= 1
 mem_tot_CPU18 <= 32768
 normalized_mem_tot_CPU18 <= 1
 normalized_rows_tot_CPU18 <= 1
 mem_tot_CPU19 <= 32768
 normalized_mem_tot_CPU19 <= 1
 normalized_rows_tot_CPU19 <= 1
 mem_tot_CPU20 <= 32768
 normalized_mem_tot_CPU20 <= 1
 normalized_rows_tot_CPU20 <= 1
 mem_tot_CPU21 <= 32768
 normalized_mem_tot_CPU21 <= 1
 normalized_rows_tot_CPU21 <= 1
 mem_tot_CPU22 <= 32768
 normalized_mem_tot_CPU22 <= 1
 normalized_rows_tot_CPU22 <= 1
 mem_tot_CPU23 <= 32768
 normalized_mem_tot_CPU23 <= 1
 normalized_rows_tot_CPU23 <= 1
 mem_tot_CPU24 <= 32768
 normalized_mem_tot_CPU24 <= 1
 normalized_rows_tot_CPU24 <= 1
 mem_tot_CPU25 <= 32768
 normalized_mem_tot_CPU25 <= 1
 normalized_rows_tot_CPU25 <= 1
 mem_tot_CPU26 <= 32768
 normalized_mem_tot_CPU26 <= 1
 normalized_rows_tot_CPU26 <= 1
 mem_tot_CPU27 <= 32768
 normalized_mem_tot_CPU27 <= 1
 normalized_rows_tot_CPU27 <= 1
 mem_tot_CPU28 <= 32768
 normalized_mem_tot_CPU28 <= 1
 normalized_rows_tot_CPU28 <= 1
 mem_tot_CPU29 <= 32768
 normalized_mem_tot_CPU29 <= 1
 normalized_rows_tot_CPU29 <= 1
 mem_tot_CPU30 <= 32768
 normalized_mem_tot_CPU30 <= 1
 normalized_rows_tot_CPU30 <= 1
 mem_tot_CPU31 <= 32768
 normalized_mem_tot_CPU31 <= 1
 normalized_rows_tot_CPU31 <= 1
 mem_tot_CPU32 <= 32768
 normalized_mem_tot_CPU32 <= 1
 normalized_rows_tot_CPU32 <= 1
 mem_tot_tor_P41 <= 6144
 normalized_mem_tot_tor_P41 <= 1
 normalized_rows_tot_tor_P41 <= 1
 mem_tot_tor_P42 <= 6144
 normalized_mem_tot_tor_P42 <= 1
 normalized_rows_tot_tor_P42 <= 1
 mem_tot_tor_P43 <= 6144
 normalized_mem_tot_tor_P43 <= 1
 normalized_rows_tot_tor_P43 <= 1
 mem_tot_tor_P44 <= 6144
 normalized_mem_tot_tor_P44 <= 1
 normalized_rows_tot_tor_P44 <= 1
 mem_tot_l1_P41 <= 6144
 normalized_mem_tot_l1_P41 <= 1
 normalized_rows_tot_l1_P41 <= 1
 mem_tot_l1_P42 <= 6144
 normalized_mem_tot_l1_P42 <= 1
 normalized_rows_tot_l1_P42 <= 1
 mem_tot_l2_P4 <= 6144
 normalized_mem_tot_l2_P4 <= 1
 normalized_rows_tot_l2_P4 <= 1
 cores_sketch_CPU1 <= 7
 cores_sketch_CPU2 <= 7
 cores_sketch_CPU3 <= 7
 cores_sketch_CPU4 <= 7
 cores_sketch_CPU5 <= 7
 cores_sketch_CPU6 <= 7
 cores_sketch_CPU7 <= 7
 cores_sketch_CPU8 <= 7
 cores_sketch_CPU9 <= 7
 cores_sketch_CPU10 <= 7
 cores_sketch_CPU11 <= 7
 cores_sketch_CPU12 <= 7
 cores_sketch_CPU13 <= 7
 cores_sketch_CPU14 <= 7
 cores_sketch_CPU15 <= 7
 cores_sketch_CPU16 <= 7
 cores_sketch_CPU17 <= 7
 cores_sketch_CPU18 <= 7
 cores_sketch_CPU19 <= 7
 cores_sketch_CPU20 <= 7
 cores_sketch_CPU21 <= 7
 cores_sketch_CPU22 <= 7
 cores_sketch_CPU23 <= 7
 cores_sketch_CPU24 <= 7
 cores_sketch_CPU25 <= 7
 cores_sketch_CPU26 <= 7
 cores_sketch_CPU27 <= 7
 cores_sketch_CPU28 <= 7
 cores_sketch_CPU29 <= 7
 cores_sketch_CPU30 <= 7
 cores_sketch_CPU31 <= 7
 cores_sketch_CPU32 <= 7
Binaries
 frac[32,54] frac[32,113] frac[32,77] frac[32,25] frac[32,110] frac[32,68]
 frac[32,16] frac[32,101] frac[32,91] frac[32,49] frac[32,82] frac[32,72]
 frac[32,115] frac[32,105] frac[32,79] frac[32,96] frac[32,18] frac[32,103]
 frac[32,93] frac[32,15] frac[32,84] frac[32,74] frac[32,32] frac[32,117]
 frac[32,107] frac[32,65] frac[32,39] frac[32,29] frac[32,98] frac[32,88]
 frac[32,62] frac[32,20] frac[32,10] frac[32,121] frac[32,95] frac[32,86]
 frac[32,76] frac[32,34] frac[32,24] frac[32,119] frac[32,100] frac[32,90]
 frac[32,48] frac[32,22] frac[32,12] frac[32,55] frac[32,45] frac[32,3]
 frac[32,114] frac[32,17] frac[32,102] frac[32,92] frac[32,14] frac[32,83]
 frac[32,47] frac[32,5] frac[32,64] frac[32,28] frac[32,71] frac[32,61]
 frac[32,9] frac[32,120] frac[32,94] frac[32,85] frac[32,75] frac[32,108]
 frac[32,66] frac[32,56] frac[32,30] frac[32,99] frac[32,89] frac[32,21]
 frac[32,11] frac[32,122] frac[32,80] frac[33,55] frac[33,101] frac[33,95]
 frac[33,34] frac[33,126] frac[33,74] frac[33,59] frac[33,29] frac[33,105]
 frac[33,60] frac[33,22] frac[33,68] frac[33,47] frac[33,93] frac[33,26]
 frac[33,72] frac[33,57] frac[33,19] frac[33,117] frac[33,111] frac[33,65]
 frac[33,20] frac[33,14] frac[33,90] frac[33,45] frac[33,121] frac[33,38]
 frac[33,24] frac[33,17] frac[33,11] frac[33,48] frac[33,42] frac[33,12]
 frac[33,102] frac[33,88] frac[33,35] frac[33,5] frac[33,81] frac[33,75]
 frac[33,36] frac[33,30] frac[33,106] frac[33,76] frac[33,61] frac[33,23]
 frac[33,9] frac[33,99] frac[33,69] frac[33,54] frac[33,100] frac[33,94]
 frac[33,33] frac[33,27] frac[33,73] frac[33,58] frac[33,28] frac[33,118]
 frac[33,66] frac[33,51] frac[33,21] frac[33,15] frac[33,52] frac[33,46]
 frac[33,122] frac[33,92] frac[33,39] frac[33,25] frac[33,56] frac[33,18]
 frac[33,110] frac[33,49] frac[33,43] frac[33,103] frac[33,89] frac[33,120]
 frac[33,31] frac[33,107] frac[33,77] frac[33,62] frac[33,10] frac[34,52]
 frac[34,3] frac[34,114] frac[34,108] frac[34,26] frac[34,103] frac[34,59]
 frac[34,21] frac[34,126] frac[34,88] frac[34,121] frac[34,77] frac[34,1]
 frac[34,106] frac[34,62] frac[34,24] frac[34,101] frac[34,95] frac[34,86]
 frac[34,48] frac[34,42] frac[34,81] frac[34,75] frac[34,66] frac[34,22]
 frac[34,99] frac[34,17] frac[34,11] frac[34,122] frac[34,84] frac[34,2]
 frac[34,117] frac[34,73] frac[34,29] frac[34,102] frac[34,58] frac[34,20]
 frac[34,14] frac[34,97] frac[34,47] frac[34,9] frac[34,120] frac[34,82]
 frac[34,38] frac[34,115] frac[34,33] frac[34,100] frac[34,94] frac[34,18]
 frac[34,12] frac[34,89] frac[34,51] frac[34,7] frac[34,118] frac[34,80]
 frac[34,74] frac[34,30] frac[34,113] frac[34,107] frac[34,69] frac[34,25]
 frac[34,98] frac[34,92] frac[34,16] frac[34,5] frac[34,110] frac[34,72]
 frac[34,34] frac[34,28] frac[34,105] frac[34,23] frac[35,88] frac[35,21]
 frac[35,123] frac[35,65] frac[35,46] frac[35,52] frac[35,105] frac[35,79]
 frac[35,85] frac[35,34] frac[35,110] frac[35,43] frac[35,49] frac[35,98]
 frac[35,72] frac[35,63] frac[35,107] frac[35,36] frac[35,16] frac[35,118]
 frac[35,92] frac[35,45] frac[35,51] frac[35,25] frac[35,56] frac[35,80]
 frac[35,33] frac[35,7] frac[35,89] frac[35,38] frac[35,18] frac[35,120]
 frac[35,94] frac[35,47] frac[35,97] frac[35,102] frac[35,76] frac[35,82]
 frac[35,35] frac[35,111] frac[35,117] frac[35,91] frac[35,40] frac[35,14]
 frac[35,122] frac[35,64] frac[35,29] frac[35,73] frac[35,60] frac[35,2]
 frac[35,11] frac[35,17] frac[35,119] frac[35,93] frac[35,42] frac[35,48]
 frac[35,22] frac[35,66] frac[35,101] frac[35,75] frac[35,62] frac[35,106]
 frac[35,19] frac[35,126] frac[35,68] frac[35,59] frac[35,77] frac[35,83]
 frac[35,32] frac[36,68] frac[36,34] frac[36,22] frac[36,101] frac[36,55]
 frac[36,29] frac[36,108] frac[36,74] frac[36,62] frac[36,107] frac[36,113]
 frac[36,95] frac[36,3] frac[36,120] frac[36,10] frac[36,16] frac[36,103]
 frac[36,77] frac[36,43] frac[36,49] frac[36,110] frac[36,82] frac[36,56]
 frac[36,115] frac[36,89] frac[36,122] frac[36,64] frac[36,46] frac[36,52]
 frac[36,18] frac[36,79] frac[36,85] frac[36,51] frac[36,25] frac[36,118]
 frac[36,92] frac[36,58] frac[36,91] frac[36,33] frac[36,15] frac[36,21]
 frac[36,100] frac[36,66] frac[36,54] frac[36,28] frac[36,99] frac[36,73]
 frac[36,61] frac[36,106] frac[36,94] frac[36,36] frac[36,69] frac[36,9]
 frac[36,23] frac[36,102] frac[36,76] frac[36,42] frac[36,48] frac[36,30]
 frac[36,75] frac[36,81] frac[36,5] frac[36,126] frac[36,114] frac[36,88]
 frac[36,38] frac[36,12] frac[36,121] frac[36,45] frac[36,11] frac[36,17]
 frac[36,84] frac[36,24] frac[36,111] frac[36,117] frac[36,83] frac[36,90]
 frac[36,32] frac[36,14] frac[36,20] frac[36,65] frac[36,47] frac[36,19]
 frac[36,98] frac[36,72] frac[36,86] frac[36,60] frac[36,26] frac[36,105]
 frac[36,119] frac[36,93] frac[36,59] frac[37,82] frac[37,23] frac[37,113]
 frac[37,91] frac[37,16] frac[37,122] frac[37,68] frac[37,25] frac[37,99]
 frac[37,77] frac[37,56] frac[37,2] frac[37,108] frac[37,118] frac[37,80]
 frac[37,33] frac[37,11] frac[37,21] frac[37,89] frac[37,42] frac[37,52]
 frac[37,30] frac[37,120] frac[37,66] frac[37,7] frac[37,97] frac[37,75]
 frac[37,85] frac[37,38] frac[37,106] frac[37,94] frac[37,47] frac[37,9]
 frac[37,19] frac[37,28] frac[37,102] frac[37,64] frac[37,59] frac[37,5]
 frac[37,111] frac[37,73] frac[37,83] frac[37,36] frac[37,14] frac[37,114]
 frac[37,92] frac[37,45] frac[37,17] frac[37,69] frac[37,48] frac[37,26]
 frac[37,100] frac[37,3] frac[37,119] frac[37,81] frac[37,34] frac[37,12]
 frac[37,22] frac[37,43] frac[37,121] frac[37,62] frac[37,24] frac[37,98]
 frac[37,76] frac[37,86] frac[37,107] frac[37,117] frac[37,95] frac[37,32]
 frac[37,20] frac[37,126] frac[37,88] frac[37,51] frac[37,29] frac[37,103]
 frac[37,65] frac[37,60] frac[37,74] frac[37,84] frac[37,105] frac[37,115]
 frac[37,93] frac[37,46] frac[37,18] frac[37,101] frac[37,79] frac[37,58]
 frac[37,110] frac[37,72] frac[38,86] frac[38,119] frac[38,73] frac[38,5]
 frac[38,110] frac[38,64] frac[38,60] frac[38,22] frac[38,93] frac[38,9]
 frac[38,114] frac[38,84] frac[38,46] frac[38,26] frac[38,117] frac[38,79]
 frac[38,33] frac[38,59] frac[38,29] frac[38,108] frac[38,88] frac[38,20]
 frac[38,103] frac[38,121] frac[38,83] frac[38,74] frac[38,24] frac[38,107]
 frac[38,77] frac[38,19] frac[38,98] frac[38,68] frac[38,48] frac[38,81]
 frac[38,43] frac[38,118] frac[38,72] frac[38,34] frac[38,30] frac[38,105]
 frac[38,17] frac[38,92] frac[38,3] frac[38,32] frac[38,58] frac[38,28]
 frac[38,111] frac[38,65] frac[38,91] frac[38,23] frac[38,102] frac[38,120]
 frac[38,82] frac[38,52] frac[38,14] frac[38,115] frac[38,85] frac[38,47]
 frac[38,106] frac[38,76] frac[38,38] frac[38,56] frac[38,18] frac[38,89]
 frac[38,51] frac[38,21] frac[38,100] frac[38,126] frac[38,42] frac[38,12]
 frac[38,113] frac[38,45] frac[38,25] frac[38,36] frac[38,62] frac[38,16]
 frac[38,99] frac[38,69] frac[38,95] frac[38,11]
Generals
 cores_sketch_CPU1 cores_sketch_CPU2 cores_sketch_CPU3 cores_sketch_CPU4
 cores_sketch_CPU5 cores_sketch_CPU6 cores_sketch_CPU7 cores_sketch_CPU8
 cores_sketch_CPU9 cores_sketch_CPU10 cores_sketch_CPU11 cores_sketch_CPU12
 cores_sketch_CPU13 cores_sketch_CPU14 cores_sketch_CPU15
 cores_sketch_CPU16 cores_sketch_CPU17 cores_sketch_CPU18
 cores_sketch_CPU19 cores_sketch_CPU20 cores_sketch_CPU21
 cores_sketch_CPU22 cores_sketch_CPU23 cores_sketch_CPU24
 cores_sketch_CPU25 cores_sketch_CPU26 cores_sketch_CPU27
 cores_sketch_CPU28 cores_sketch_CPU29 cores_sketch_CPU30
 cores_sketch_CPU31 cores_sketch_CPU32
General Constraints
 mem_overall_fixed_thr: max_mem_fixed_thr = MAX ( mem_tot_tor_P41 ,
   mem_tot_tor_P42 , mem_tot_tor_P43 , mem_tot_tor_P44 , mem_tot_l1_P41 ,
   mem_tot_l1_P42 , mem_tot_l2_P4 )
 mem_overall_others: max_mem_others = MAX ( normalized_mem_tot_CPU1 ,
   normalized_mem_tot_CPU2 , normalized_mem_tot_CPU3 ,
   normalized_mem_tot_CPU4 , normalized_mem_tot_CPU5 ,
   normalized_mem_tot_CPU6 , normalized_mem_tot_CPU7 ,
   normalized_mem_tot_CPU8 , normalized_mem_tot_CPU9 ,
   normalized_mem_tot_CPU10 , normalized_mem_tot_CPU11 ,
   normalized_mem_tot_CPU12 , normalized_mem_tot_CPU13 ,
   normalized_mem_tot_CPU14 , normalized_mem_tot_CPU15 ,
   normalized_mem_tot_CPU16 , normalized_mem_tot_CPU17 ,
   normalized_mem_tot_CPU18 , normalized_mem_tot_CPU19 ,
   normalized_mem_tot_CPU20 , normalized_mem_tot_CPU21 ,
   normalized_mem_tot_CPU22 , normalized_mem_tot_CPU23 ,
   normalized_mem_tot_CPU24 , normalized_mem_tot_CPU25 ,
   normalized_mem_tot_CPU26 , normalized_mem_tot_CPU27 ,
   normalized_mem_tot_CPU28 , normalized_mem_tot_CPU29 ,
   normalized_mem_tot_CPU30 , normalized_mem_tot_CPU31 ,
   normalized_mem_tot_CPU32 )
 rows_overall_fixed_thr: max_rows_fixed_thr = MAX ( rows_tot_tor_P41 ,
   rows_tot_tor_P42 , rows_tot_tor_P43 , rows_tot_tor_P44 ,
   rows_tot_l1_P41 , rows_tot_l1_P42 , rows_tot_l2_P4 )
 rows_overall_others: max_rows_others = MAX ( normalized_rows_tot_CPU1 ,
   normalized_rows_tot_CPU2 , normalized_rows_tot_CPU3 ,
   normalized_rows_tot_CPU4 , normalized_rows_tot_CPU5 ,
   normalized_rows_tot_CPU6 , normalized_rows_tot_CPU7 ,
   normalized_rows_tot_CPU8 , normalized_rows_tot_CPU9 ,
   normalized_rows_tot_CPU10 , normalized_rows_tot_CPU11 ,
   normalized_rows_tot_CPU12 , normalized_rows_tot_CPU13 ,
   normalized_rows_tot_CPU14 , normalized_rows_tot_CPU15 ,
   normalized_rows_tot_CPU16 , normalized_rows_tot_CPU17 ,
   normalized_rows_tot_CPU18 , normalized_rows_tot_CPU19 ,
   normalized_rows_tot_CPU20 , normalized_rows_tot_CPU21 ,
   normalized_rows_tot_CPU22 , normalized_rows_tot_CPU23 ,
   normalized_rows_tot_CPU24 , normalized_rows_tot_CPU25 ,
   normalized_rows_tot_CPU26 , normalized_rows_tot_CPU27 ,
   normalized_rows_tot_CPU28 , normalized_rows_tot_CPU29 ,
   normalized_rows_tot_CPU30 , normalized_rows_tot_CPU31 ,
   normalized_rows_tot_CPU32 )
 mem_access_time_CPU1: m_access_time_CPU1 = PWL ( mem_tot_CPU1 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU2: m_access_time_CPU2 = PWL ( mem_tot_CPU2 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU3: m_access_time_CPU3 = PWL ( mem_tot_CPU3 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU4: m_access_time_CPU4 = PWL ( mem_tot_CPU4 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU5: m_access_time_CPU5 = PWL ( mem_tot_CPU5 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU6: m_access_time_CPU6 = PWL ( mem_tot_CPU6 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU7: m_access_time_CPU7 = PWL ( mem_tot_CPU7 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU8: m_access_time_CPU8 = PWL ( mem_tot_CPU8 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU9: m_access_time_CPU9 = PWL ( mem_tot_CPU9 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU10: m_access_time_CPU10 = PWL ( mem_tot_CPU10 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU11: m_access_time_CPU11 = PWL ( mem_tot_CPU11 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU12: m_access_time_CPU12 = PWL ( mem_tot_CPU12 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU13: m_access_time_CPU13 = PWL ( mem_tot_CPU13 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU14: m_access_time_CPU14 = PWL ( mem_tot_CPU14 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU15: m_access_time_CPU15 = PWL ( mem_tot_CPU15 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU16: m_access_time_CPU16 = PWL ( mem_tot_CPU16 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU17: m_access_time_CPU17 = PWL ( mem_tot_CPU17 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU18: m_access_time_CPU18 = PWL ( mem_tot_CPU18 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU19: m_access_time_CPU19 = PWL ( mem_tot_CPU19 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU20: m_access_time_CPU20 = PWL ( mem_tot_CPU20 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU21: m_access_time_CPU21 = PWL ( mem_tot_CPU21 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU22: m_access_time_CPU22 = PWL ( mem_tot_CPU22 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU23: m_access_time_CPU23 = PWL ( mem_tot_CPU23 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU24: m_access_time_CPU24 = PWL ( mem_tot_CPU24 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU25: m_access_time_CPU25 = PWL ( mem_tot_CPU25 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU26: m_access_time_CPU26 = PWL ( mem_tot_CPU26 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU27: m_access_time_CPU27 = PWL ( mem_tot_CPU27 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU28: m_access_time_CPU28 = PWL ( mem_tot_CPU28 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU29: m_access_time_CPU29 = PWL ( mem_tot_CPU29 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU30: m_access_time_CPU30 = PWL ( mem_tot_CPU30 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU31: m_access_time_CPU31 = PWL ( mem_tot_CPU31 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU32: m_access_time_CPU32 = PWL ( mem_tot_CPU32 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

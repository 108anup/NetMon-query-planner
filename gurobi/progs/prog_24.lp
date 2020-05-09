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
 cov_0_3: frac[0,3] + frac[11,3] + frac[32,3] + frac[33,3] + frac[36,3]
   >= 0.9
 cov_1_21: frac[19,21] + frac[24,21] + frac[34,21] + frac[35,21]
   + frac[37,21] >= 0.7
 cov_2_0: frac[22,0] + frac[28,0] + frac[34,0] + frac[35,0] + frac[37,0]
   >= 0.8
 cov_3_16: frac[11,16] + frac[19,16] + frac[33,16] + frac[34,16]
   + frac[36,16] + frac[37,16] + frac[38,16] >= 0.9
 cov_4_8: frac[0,8] + frac[19,8] + frac[32,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 1
 cov_5_25: frac[22,25] + frac[28,25] + frac[34,25] + frac[35,25]
   + frac[37,25] >= 0.7
 cov_6_8: frac[23,8] + frac[24,8] + frac[34,8] + frac[35,8] + frac[37,8]
   >= 1
 cov_7_29: frac[10,29] + frac[23,29] + frac[33,29] + frac[34,29]
   + frac[36,29] + frac[37,29] + frac[38,29] >= 0.8
 cov_8_13: frac[0,13] + frac[11,13] + frac[32,13] + frac[33,13]
   + frac[36,13] >= 0.7
 cov_9_16: frac[11,16] + frac[24,16] + frac[33,16] + frac[35,16]
   + frac[36,16] + frac[37,16] + frac[38,16] >= 0.9
 cov_10_22: frac[11,22] + frac[19,22] + frac[33,22] + frac[34,22]
   + frac[36,22] + frac[37,22] + frac[38,22] >= 1
 cov_11_11: frac[10,11] + frac[23,11] + frac[33,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.8
 cov_12_21: frac[19,21] + frac[23,21] + frac[34,21] >= 0.7
 cov_13_4: frac[23,4] + frac[28,4] + frac[34,4] + frac[35,4] + frac[37,4]
   >= 0.7
 cov_14_24: frac[10,24] + frac[28,24] + frac[33,24] + frac[35,24]
   + frac[36,24] + frac[37,24] + frac[38,24] >= 1
 cov_15_8: frac[11,8] + frac[19,8] + frac[33,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.7
 cov_16_19: frac[23,19] + frac[24,19] + frac[34,19] + frac[35,19]
   + frac[37,19] >= 0.7
 cov_17_29: frac[22,29] + frac[23,29] + frac[34,29] >= 0.8
 cov_18_11: frac[22,11] + frac[23,11] + frac[34,11] >= 1
 cov_19_9: frac[24,9] + frac[28,9] + frac[35,9] >= 0.8
 cov_20_8: frac[11,8] + frac[28,8] + frac[33,8] + frac[35,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.9
 cov_21_3: frac[22,3] + frac[23,3] + frac[34,3] >= 1
 cov_22_7: frac[10,7] + frac[24,7] + frac[33,7] + frac[35,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.7
 cov_23_13: frac[19,13] + frac[22,13] + frac[34,13] >= 0.8
 cov_24_7: frac[23,7] + frac[24,7] + frac[34,7] + frac[35,7] + frac[37,7]
   >= 1
 cov_25_25: frac[11,25] + frac[24,25] + frac[33,25] + frac[35,25]
   + frac[36,25] + frac[37,25] + frac[38,25] >= 0.8
 cov_26_28: frac[10,28] + frac[11,28] + frac[33,28] >= 0.8
 cov_27_17: frac[19,17] + frac[28,17] + frac[34,17] + frac[35,17]
   + frac[37,17] >= 0.7
 cov_28_23: frac[19,23] + frac[22,23] + frac[34,23] >= 0.9
 cov_29_25: frac[0,25] + frac[22,25] + frac[32,25] + frac[34,25]
   + frac[36,25] + frac[37,25] + frac[38,25] >= 0.7
 cov_30_6: frac[22,6] + frac[24,6] + frac[34,6] + frac[35,6] + frac[37,6]
   >= 0.9
 cov_31_7: frac[22,7] + frac[28,7] + frac[34,7] + frac[35,7] + frac[37,7]
   >= 0.9
 cov_32_14: frac[11,14] + frac[28,14] + frac[33,14] + frac[35,14]
   + frac[36,14] + frac[37,14] + frac[38,14] >= 0.7
 cov_33_0: frac[10,0] + frac[28,0] + frac[33,0] + frac[35,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.8
 cov_34_3: frac[10,3] + frac[23,3] + frac[33,3] + frac[34,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.7
 cov_35_19: frac[10,19] + frac[19,19] + frac[33,19] + frac[34,19]
   + frac[36,19] + frac[37,19] + frac[38,19] >= 0.8
 cov_36_0: frac[0,0] + frac[23,0] + frac[32,0] + frac[34,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 1
 cov_37_23: frac[22,23] + frac[23,23] + frac[34,23] >= 1
 cov_38_17: frac[10,17] + frac[11,17] + frac[33,17] >= 0.9
 cov_39_5: frac[23,5] + frac[28,5] + frac[34,5] + frac[35,5] + frac[37,5]
   >= 0.8
 cov_40_1: frac[10,1] + frac[28,1] + frac[33,1] + frac[35,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.7
 cov_41_14: frac[22,14] + frac[23,14] + frac[34,14] >= 0.9
 cov_42_17: frac[0,17] + frac[24,17] + frac[32,17] + frac[35,17]
   + frac[36,17] + frac[37,17] + frac[38,17] >= 0.7
 cov_43_24: frac[0,24] + frac[19,24] + frac[32,24] + frac[34,24]
   + frac[36,24] + frac[37,24] + frac[38,24] >= 1
 cov_44_27: frac[0,27] + frac[10,27] + frac[32,27] + frac[33,27]
   + frac[36,27] >= 0.8
 cov_45_6: frac[0,6] + frac[22,6] + frac[32,6] + frac[34,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 1
 cov_46_8: frac[22,8] + frac[23,8] + frac[34,8] >= 0.8
 cov_47_6: frac[23,6] + frac[24,6] + frac[34,6] + frac[35,6] + frac[37,6]
   >= 0.7
 cov_48_23: frac[19,23] + frac[23,23] + frac[34,23] >= 1
 cov_49_22: frac[0,22] + frac[23,22] + frac[32,22] + frac[34,22]
   + frac[36,22] + frac[37,22] + frac[38,22] >= 0.7
 cov_50_11: frac[19,11] + frac[28,11] + frac[34,11] + frac[35,11]
   + frac[37,11] >= 1
 cov_51_17: frac[10,17] + frac[22,17] + frac[33,17] + frac[34,17]
   + frac[36,17] + frac[37,17] + frac[38,17] >= 0.8
 cov_52_22: frac[22,22] + frac[24,22] + frac[34,22] + frac[35,22]
   + frac[37,22] >= 0.7
 cov_53_12: frac[22,12] + frac[24,12] + frac[34,12] + frac[35,12]
   + frac[37,12] >= 0.7
 cov_54_6: frac[22,6] + frac[24,6] + frac[34,6] + frac[35,6] + frac[37,6]
   >= 1
 cov_55_20: frac[11,20] + frac[22,20] + frac[33,20] + frac[34,20]
   + frac[36,20] + frac[37,20] + frac[38,20] >= 0.7
 cov_56_27: frac[19,27] + frac[22,27] + frac[34,27] >= 0.9
 cov_57_16: frac[0,16] + frac[28,16] + frac[32,16] + frac[35,16]
   + frac[36,16] + frac[37,16] + frac[38,16] >= 0.8
 cov_58_23: frac[19,23] + frac[23,23] + frac[34,23] >= 0.8
 cov_59_11: frac[11,11] + frac[28,11] + frac[33,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.8
 cov_60_44: frac[3,44] + frac[20,44] + frac[32,44] + frac[34,44]
   + frac[36,44] + frac[37,44] + frac[38,44] >= 1
 cov_61_52: frac[15,52] + frac[30,52] + frac[33,52] + frac[35,52]
   + frac[36,52] + frac[37,52] + frac[38,52] >= 1
 cov_62_46: frac[6,46] + frac[30,46] + frac[32,46] + frac[35,46]
   + frac[36,46] + frac[37,46] + frac[38,46] >= 0.7
 cov_63_36: frac[2,36] + frac[6,36] + frac[32,36] >= 0.7
 cov_64_44: frac[3,44] + frac[30,44] + frac[32,44] + frac[35,44]
   + frac[36,44] + frac[37,44] + frac[38,44] >= 0.8
 cov_65_30: frac[20,30] + frac[27,30] + frac[34,30] + frac[35,30]
   + frac[37,30] >= 0.7
 cov_66_35: frac[2,35] + frac[20,35] + frac[32,35] + frac[34,35]
   + frac[36,35] + frac[37,35] + frac[38,35] >= 1
 cov_67_51: frac[6,51] + frac[30,51] + frac[32,51] + frac[35,51]
   + frac[36,51] + frac[37,51] + frac[38,51] >= 0.7
 cov_68_33: frac[3,33] + frac[7,33] + frac[32,33] >= 1
 cov_69_37: frac[15,37] + frac[20,37] + frac[33,37] + frac[34,37]
   + frac[36,37] + frac[37,37] + frac[38,37] >= 0.9
 cov_70_31: frac[2,31] + frac[15,31] + frac[32,31] + frac[33,31]
   + frac[36,31] >= 0.8
 cov_71_37: frac[2,37] + frac[20,37] + frac[32,37] + frac[34,37]
   + frac[36,37] + frac[37,37] + frac[38,37] >= 0.8
 cov_72_35: frac[7,35] + frac[27,35] + frac[32,35] + frac[35,35]
   + frac[36,35] + frac[37,35] + frac[38,35] >= 0.7
 cov_73_56: frac[15,56] + frac[30,56] + frac[33,56] + frac[35,56]
   + frac[36,56] + frac[37,56] + frac[38,56] >= 0.8
 cov_74_39: frac[3,39] + frac[30,39] + frac[32,39] + frac[35,39]
   + frac[36,39] + frac[37,39] + frac[38,39] >= 0.9
 cov_75_39: frac[2,39] + frac[15,39] + frac[32,39] + frac[33,39]
   + frac[36,39] >= 0.8
 cov_76_39: frac[20,39] + frac[27,39] + frac[34,39] + frac[35,39]
   + frac[37,39] >= 0.8
 cov_77_47: frac[7,47] + frac[15,47] + frac[32,47] + frac[33,47]
   + frac[36,47] >= 0.9
 cov_78_43: frac[2,43] + frac[27,43] + frac[32,43] + frac[35,43]
   + frac[36,43] + frac[37,43] + frac[38,43] >= 0.7
 cov_79_57: frac[3,57] + frac[6,57] + frac[32,57] >= 0.9
 cov_80_39: frac[2,39] + frac[3,39] + frac[32,39] >= 1
 cov_81_37: frac[2,37] + frac[7,37] + frac[32,37] >= 0.7
 cov_82_48: frac[3,48] + frac[20,48] + frac[32,48] + frac[34,48]
   + frac[36,48] + frac[37,48] + frac[38,48] >= 0.8
 cov_83_48: frac[7,48] + frac[27,48] + frac[32,48] + frac[35,48]
   + frac[36,48] + frac[37,48] + frac[38,48] >= 0.8
 cov_84_39: frac[3,39] + frac[7,39] + frac[32,39] >= 1
 cov_85_35: frac[2,35] + frac[27,35] + frac[32,35] + frac[35,35]
   + frac[36,35] + frac[37,35] + frac[38,35] >= 0.9
 cov_86_39: frac[2,39] + frac[7,39] + frac[32,39] >= 1
 cov_87_41: frac[3,41] + frac[6,41] + frac[32,41] >= 1
 cov_88_55: frac[3,55] + frac[30,55] + frac[32,55] + frac[35,55]
   + frac[36,55] + frac[37,55] + frac[38,55] >= 0.8
 cov_89_50: frac[6,50] + frac[15,50] + frac[32,50] + frac[33,50]
   + frac[36,50] >= 1
 cov_90_50: frac[2,50] + frac[20,50] + frac[32,50] + frac[34,50]
   + frac[36,50] + frac[37,50] + frac[38,50] >= 1
 cov_91_48: frac[6,48] + frac[7,48] + frac[32,48] >= 0.9
 cov_92_34: frac[20,34] + frac[27,34] + frac[34,34] + frac[35,34]
   + frac[37,34] >= 0.8
 cov_93_35: frac[7,35] + frac[20,35] + frac[32,35] + frac[34,35]
   + frac[36,35] + frac[37,35] + frac[38,35] >= 1
 cov_94_38: frac[6,38] + frac[7,38] + frac[32,38] >= 0.8
 cov_95_32: frac[2,32] + frac[27,32] + frac[32,32] + frac[35,32]
   + frac[36,32] + frac[37,32] + frac[38,32] >= 0.9
 cov_96_34: frac[2,34] + frac[6,34] + frac[32,34] >= 0.8
 cov_97_49: frac[3,49] + frac[20,49] + frac[32,49] + frac[34,49]
   + frac[36,49] + frac[37,49] + frac[38,49] >= 0.7
 cov_98_31: frac[2,31] + frac[15,31] + frac[32,31] + frac[33,31]
   + frac[36,31] >= 0.7
 cov_99_58: frac[6,58] + frac[30,58] + frac[32,58] + frac[35,58]
   + frac[36,58] + frac[37,58] + frac[38,58] >= 0.7
 cov_100_50: frac[6,50] + frac[7,50] + frac[32,50] >= 1
 cov_101_37: frac[2,37] + frac[30,37] + frac[32,37] + frac[35,37]
   + frac[36,37] + frac[37,37] + frac[38,37] >= 0.7
 cov_102_44: frac[3,44] + frac[7,44] + frac[32,44] >= 0.7
 cov_103_55: frac[15,55] + frac[27,55] + frac[33,55] + frac[35,55]
   + frac[36,55] + frac[37,55] + frac[38,55] >= 1
 cov_104_50: frac[3,50] + frac[27,50] + frac[32,50] + frac[35,50]
   + frac[36,50] + frac[37,50] + frac[38,50] >= 0.8
 cov_105_38: frac[6,38] + frac[30,38] + frac[32,38] + frac[35,38]
   + frac[36,38] + frac[37,38] + frac[38,38] >= 1
 cov_106_38: frac[6,38] + frac[20,38] + frac[32,38] + frac[34,38]
   + frac[36,38] + frac[37,38] + frac[38,38] >= 1
 cov_107_49: frac[7,49] + frac[20,49] + frac[32,49] + frac[34,49]
   + frac[36,49] + frac[37,49] + frac[38,49] >= 0.8
 cov_108_55: frac[6,55] + frac[20,55] + frac[32,55] + frac[34,55]
   + frac[36,55] + frac[37,55] + frac[38,55] >= 1
 cov_109_59: frac[3,59] + frac[15,59] + frac[32,59] + frac[33,59]
   + frac[36,59] >= 1
 cov_110_31: frac[2,31] + frac[6,31] + frac[32,31] >= 0.7
 cov_111_49: frac[20,49] + frac[30,49] + frac[34,49] + frac[35,49]
   + frac[37,49] >= 0.8
 cov_112_38: frac[6,38] + frac[20,38] + frac[32,38] + frac[34,38]
   + frac[36,38] + frac[37,38] + frac[38,38] >= 1
 cov_113_40: frac[2,40] + frac[7,40] + frac[32,40] >= 0.9
 cov_114_53: frac[2,53] + frac[3,53] + frac[32,53] >= 0.9
 cov_115_59: frac[3,59] + frac[15,59] + frac[32,59] + frac[33,59]
   + frac[36,59] >= 0.8
 cov_116_46: frac[3,46] + frac[6,46] + frac[32,46] >= 0.8
 cov_117_41: frac[6,41] + frac[15,41] + frac[32,41] + frac[33,41]
   + frac[36,41] >= 1
 cov_118_38: frac[2,38] + frac[20,38] + frac[32,38] + frac[34,38]
   + frac[36,38] + frac[37,38] + frac[38,38] >= 0.9
 cov_119_31: frac[7,31] + frac[30,31] + frac[32,31] + frac[35,31]
   + frac[36,31] + frac[37,31] + frac[38,31] >= 1
 cov_120_84: frac[8,84] + frac[12,84] + frac[33,84] >= 0.8
 cov_121_69: frac[5,69] + frac[14,69] + frac[32,69] + frac[33,69]
   + frac[36,69] >= 0.7
 cov_122_83: frac[1,83] + frac[8,83] + frac[32,83] + frac[33,83]
   + frac[36,83] >= 0.8
 cov_123_84: frac[8,84] + frac[21,84] + frac[33,84] + frac[34,84]
   + frac[36,84] + frac[37,84] + frac[38,84] >= 1
 cov_124_65: frac[5,65] + frac[8,65] + frac[32,65] + frac[33,65]
   + frac[36,65] >= 1
 cov_125_71: frac[9,71] + frac[12,71] + frac[33,71] >= 0.8
 cov_126_72: frac[4,72] + frac[5,72] + frac[32,72] >= 1
 cov_127_77: frac[9,77] + frac[12,77] + frac[33,77] >= 0.7
 cov_128_75: frac[4,75] + frac[5,75] + frac[32,75] >= 0.8
 cov_129_71: frac[9,71] + frac[12,71] + frac[33,71] >= 0.7
 cov_130_73: frac[1,73] + frac[8,73] + frac[32,73] + frac[33,73]
   + frac[36,73] >= 0.7
 cov_131_89: frac[9,89] + frac[12,89] + frac[33,89] >= 1
 cov_132_74: frac[5,74] + frac[8,74] + frac[32,74] + frac[33,74]
   + frac[36,74] >= 0.9
 cov_133_87: frac[12,87] + frac[21,87] + frac[33,87] + frac[34,87]
   + frac[36,87] + frac[37,87] + frac[38,87] >= 1
 cov_134_89: frac[1,89] + frac[8,89] + frac[32,89] + frac[33,89]
   + frac[36,89] >= 1
 cov_135_60: frac[1,60] + frac[5,60] + frac[32,60] >= 0.8
 cov_136_74: frac[5,74] + frac[12,74] + frac[32,74] + frac[33,74]
   + frac[36,74] >= 0.8
 cov_137_77: frac[4,77] + frac[9,77] + frac[32,77] + frac[33,77]
   + frac[36,77] >= 0.8
 cov_138_85: frac[9,85] + frac[12,85] + frac[33,85] >= 0.7
 cov_139_70: frac[5,70] + frac[12,70] + frac[32,70] + frac[33,70]
   + frac[36,70] >= 0.9
 cov_140_87: frac[4,87] + frac[9,87] + frac[32,87] + frac[33,87]
   + frac[36,87] >= 0.7
 cov_141_62: frac[1,62] + frac[8,62] + frac[32,62] + frac[33,62]
   + frac[36,62] >= 0.7
 cov_142_77: frac[1,77] + frac[12,77] + frac[32,77] + frac[33,77]
   + frac[36,77] >= 0.9
 cov_143_80: frac[9,80] + frac[12,80] + frac[33,80] >= 0.8
 cov_144_61: frac[5,61] + frac[8,61] + frac[32,61] + frac[33,61]
   + frac[36,61] >= 0.9
 cov_145_60: frac[5,60] + frac[12,60] + frac[32,60] + frac[33,60]
   + frac[36,60] >= 0.7
 cov_146_67: frac[5,67] + frac[12,67] + frac[32,67] + frac[33,67]
   + frac[36,67] >= 0.9
 cov_147_73: frac[8,73] + frac[21,73] + frac[33,73] + frac[34,73]
   + frac[36,73] + frac[37,73] + frac[38,73] >= 0.8
 cov_148_62: frac[4,62] + frac[21,62] + frac[32,62] + frac[34,62]
   + frac[36,62] + frac[37,62] + frac[38,62] >= 0.8
 cov_149_66: frac[1,66] + frac[21,66] + frac[32,66] + frac[34,66]
   + frac[36,66] + frac[37,66] + frac[38,66] >= 0.7
 cov_150_77: frac[8,77] + frac[21,77] + frac[33,77] + frac[34,77]
   + frac[36,77] + frac[37,77] + frac[38,77] >= 0.7
 cov_151_68: frac[5,68] + frac[12,68] + frac[32,68] + frac[33,68]
   + frac[36,68] >= 0.7
 cov_152_67: frac[9,67] + frac[14,67] + frac[33,67] >= 0.8
 cov_153_75: frac[8,75] + frac[12,75] + frac[33,75] >= 0.9
 cov_154_79: frac[9,79] + frac[21,79] + frac[33,79] + frac[34,79]
   + frac[36,79] + frac[37,79] + frac[38,79] >= 0.9
 cov_155_76: frac[12,76] + frac[14,76] + frac[33,76] >= 0.8
 cov_156_64: frac[1,64] + frac[21,64] + frac[32,64] + frac[34,64]
   + frac[36,64] + frac[37,64] + frac[38,64] >= 0.7
 cov_157_83: frac[1,83] + frac[14,83] + frac[32,83] + frac[33,83]
   + frac[36,83] >= 1
 cov_158_78: frac[4,78] + frac[8,78] + frac[32,78] + frac[33,78]
   + frac[36,78] >= 1
 cov_159_72: frac[14,72] + frac[21,72] + frac[33,72] + frac[34,72]
   + frac[36,72] + frac[37,72] + frac[38,72] >= 0.9
 cov_160_64: frac[1,64] + frac[4,64] + frac[32,64] >= 0.7
 cov_161_62: frac[9,62] + frac[21,62] + frac[33,62] + frac[34,62]
   + frac[36,62] + frac[37,62] + frac[38,62] >= 0.9
 cov_162_64: frac[8,64] + frac[21,64] + frac[33,64] + frac[34,64]
   + frac[36,64] + frac[37,64] + frac[38,64] >= 1
 cov_163_64: frac[12,64] + frac[21,64] + frac[33,64] + frac[34,64]
   + frac[36,64] + frac[37,64] + frac[38,64] >= 0.7
 cov_164_70: frac[9,70] + frac[21,70] + frac[33,70] + frac[34,70]
   + frac[36,70] + frac[37,70] + frac[38,70] >= 0.7
 cov_165_88: frac[1,88] + frac[14,88] + frac[32,88] + frac[33,88]
   + frac[36,88] >= 0.8
 cov_166_76: frac[9,76] + frac[14,76] + frac[33,76] >= 1
 cov_167_66: frac[9,66] + frac[14,66] + frac[33,66] >= 1
 cov_168_71: frac[12,71] + frac[21,71] + frac[33,71] + frac[34,71]
   + frac[36,71] + frac[37,71] + frac[38,71] >= 0.8
 cov_169_89: frac[14,89] + frac[21,89] + frac[33,89] + frac[34,89]
   + frac[36,89] + frac[37,89] + frac[38,89] >= 0.8
 cov_170_60: frac[4,60] + frac[21,60] + frac[32,60] + frac[34,60]
   + frac[36,60] + frac[37,60] + frac[38,60] >= 0.7
 cov_171_67: frac[4,67] + frac[14,67] + frac[32,67] + frac[33,67]
   + frac[36,67] >= 0.8
 cov_172_61: frac[4,61] + frac[21,61] + frac[32,61] + frac[34,61]
   + frac[36,61] + frac[37,61] + frac[38,61] >= 0.8
 cov_173_71: frac[4,71] + frac[21,71] + frac[32,71] + frac[34,71]
   + frac[36,71] + frac[37,71] + frac[38,71] >= 0.8
 cov_174_85: frac[5,85] + frac[12,85] + frac[32,85] + frac[33,85]
   + frac[36,85] >= 1
 cov_175_62: frac[8,62] + frac[21,62] + frac[33,62] + frac[34,62]
   + frac[36,62] + frac[37,62] + frac[38,62] >= 0.7
 cov_176_82: frac[9,82] + frac[12,82] + frac[33,82] >= 0.7
 cov_177_71: frac[5,71] + frac[21,71] + frac[32,71] + frac[34,71]
   + frac[36,71] + frac[37,71] + frac[38,71] >= 0.9
 cov_178_62: frac[1,62] + frac[21,62] + frac[32,62] + frac[34,62]
   + frac[36,62] + frac[37,62] + frac[38,62] >= 0.8
 cov_179_63: frac[4,63] + frac[12,63] + frac[32,63] + frac[33,63]
   + frac[36,63] >= 0.8
 cov_180_113: frac[18,113] + frac[31,113] + frac[34,113] + frac[35,113]
   + frac[37,113] >= 1
 cov_181_101: frac[13,101] + frac[29,101] + frac[33,101] + frac[35,101]
   + frac[36,101] + frac[37,101] + frac[38,101] >= 1
 cov_182_107: frac[18,107] + frac[25,107] + frac[34,107] + frac[35,107]
   + frac[37,107] >= 0.9
 cov_183_107: frac[26,107] + frac[31,107] + frac[35,107] >= 0.7
 cov_184_112: frac[17,112] + frac[31,112] + frac[34,112] + frac[35,112]
   + frac[37,112] >= 0.9
 cov_185_103: frac[26,103] + frac[29,103] + frac[35,103] >= 0.9
 cov_186_92: frac[13,92] + frac[16,92] + frac[33,92] + frac[34,92]
   + frac[36,92] + frac[37,92] + frac[38,92] >= 0.7
 cov_187_104: frac[13,104] + frac[18,104] + frac[33,104] + frac[34,104]
   + frac[36,104] + frac[37,104] + frac[38,104] >= 0.9
 cov_188_107: frac[17,107] + frac[25,107] + frac[34,107] + frac[35,107]
   + frac[37,107] >= 0.8
 cov_189_118: frac[25,118] + frac[29,118] + frac[35,118] >= 0.7
 cov_190_100: frac[13,100] + frac[31,100] + frac[33,100] + frac[35,100]
   + frac[36,100] + frac[37,100] + frac[38,100] >= 0.8
 cov_191_105: frac[16,105] + frac[25,105] + frac[34,105] + frac[35,105]
   + frac[37,105] >= 0.7
 cov_192_100: frac[16,100] + frac[26,100] + frac[34,100] + frac[35,100]
   + frac[37,100] >= 1
 cov_193_110: frac[26,110] + frac[31,110] + frac[35,110] >= 0.8
 cov_194_103: frac[18,103] + frac[31,103] + frac[34,103] + frac[35,103]
   + frac[37,103] >= 0.9
 cov_195_93: frac[25,93] + frac[26,93] + frac[35,93] >= 0.8
 cov_196_98: frac[29,98] + frac[31,98] + frac[35,98] >= 0.8
 cov_197_93: frac[18,93] + frac[29,93] + frac[34,93] + frac[35,93]
   + frac[37,93] >= 0.7
 cov_198_117: frac[13,117] + frac[26,117] + frac[33,117] + frac[35,117]
   + frac[36,117] + frac[37,117] + frac[38,117] >= 0.7
 cov_199_97: frac[13,97] + frac[25,97] + frac[33,97] + frac[35,97]
   + frac[36,97] + frac[37,97] + frac[38,97] >= 0.8
 cov_200_102: frac[16,102] + frac[17,102] + frac[34,102] >= 1
 cov_201_91: frac[18,91] + frac[29,91] + frac[34,91] + frac[35,91]
   + frac[37,91] >= 0.9
 cov_202_104: frac[13,104] + frac[18,104] + frac[33,104] + frac[34,104]
   + frac[36,104] + frac[37,104] + frac[38,104] >= 0.8
 cov_203_112: frac[18,112] + frac[26,112] + frac[34,112] + frac[35,112]
   + frac[37,112] >= 0.9
 cov_204_102: frac[26,102] + frac[29,102] + frac[35,102] >= 0.7
 cov_205_94: frac[18,94] + frac[31,94] + frac[34,94] + frac[35,94]
   + frac[37,94] >= 1
 cov_206_107: frac[17,107] + frac[31,107] + frac[34,107] + frac[35,107]
   + frac[37,107] >= 0.9
 cov_207_115: frac[18,115] + frac[26,115] + frac[34,115] + frac[35,115]
   + frac[37,115] >= 0.9
 cov_208_114: frac[18,114] + frac[26,114] + frac[34,114] + frac[35,114]
   + frac[37,114] >= 0.9
 cov_209_111: frac[17,111] + frac[29,111] + frac[34,111] + frac[35,111]
   + frac[37,111] >= 0.9
 cov_210_105: frac[13,105] + frac[18,105] + frac[33,105] + frac[34,105]
   + frac[36,105] + frac[37,105] + frac[38,105] >= 0.7
 cov_211_117: frac[16,117] + frac[31,117] + frac[34,117] + frac[35,117]
   + frac[37,117] >= 0.7
 cov_212_115: frac[17,115] + frac[26,115] + frac[34,115] + frac[35,115]
   + frac[37,115] >= 0.7
 cov_213_109: frac[16,109] + frac[26,109] + frac[34,109] + frac[35,109]
   + frac[37,109] >= 0.7
 cov_214_99: frac[26,99] + frac[31,99] + frac[35,99] >= 0.8
 cov_215_110: frac[13,110] + frac[25,110] + frac[33,110] + frac[35,110]
   + frac[36,110] + frac[37,110] + frac[38,110] >= 0.7
 cov_216_96: frac[13,96] + frac[31,96] + frac[33,96] + frac[35,96]
   + frac[36,96] + frac[37,96] + frac[38,96] >= 0.8
 cov_217_90: frac[13,90] + frac[26,90] + frac[33,90] + frac[35,90]
   + frac[36,90] + frac[37,90] + frac[38,90] >= 0.7
 cov_218_96: frac[16,96] + frac[17,96] + frac[34,96] >= 0.7
 cov_219_116: frac[16,116] + frac[17,116] + frac[34,116] >= 1
 cov_220_91: frac[13,91] + frac[18,91] + frac[33,91] + frac[34,91]
   + frac[36,91] + frac[37,91] + frac[38,91] >= 0.8
 cov_221_106: frac[18,106] + frac[29,106] + frac[34,106] + frac[35,106]
   + frac[37,106] >= 0.9
 cov_222_110: frac[13,110] + frac[29,110] + frac[33,110] + frac[35,110]
   + frac[36,110] + frac[37,110] + frac[38,110] >= 1
 cov_223_95: frac[13,95] + frac[25,95] + frac[33,95] + frac[35,95]
   + frac[36,95] + frac[37,95] + frac[38,95] >= 0.8
 cov_224_119: frac[13,119] + frac[25,119] + frac[33,119] + frac[35,119]
   + frac[36,119] + frac[37,119] + frac[38,119] >= 0.7
 cov_225_109: frac[13,109] + frac[16,109] + frac[33,109] + frac[34,109]
   + frac[36,109] + frac[37,109] + frac[38,109] >= 0.9
 cov_226_93: frac[13,93] + frac[29,93] + frac[33,93] + frac[35,93]
   + frac[36,93] + frac[37,93] + frac[38,93] >= 0.9
 cov_227_112: frac[17,112] + frac[29,112] + frac[34,112] + frac[35,112]
   + frac[37,112] >= 0.8
 cov_228_97: frac[25,97] + frac[31,97] + frac[35,97] >= 0.9
 cov_229_106: frac[13,106] + frac[29,106] + frac[33,106] + frac[35,106]
   + frac[36,106] + frac[37,106] + frac[38,106] >= 0.8
 cov_230_103: frac[16,103] + frac[31,103] + frac[34,103] + frac[35,103]
   + frac[37,103] >= 1
 cov_231_101: frac[13,101] + frac[29,101] + frac[33,101] + frac[35,101]
   + frac[36,101] + frac[37,101] + frac[38,101] >= 0.9
 cov_232_111: frac[13,111] + frac[31,111] + frac[33,111] + frac[35,111]
   + frac[36,111] + frac[37,111] + frac[38,111] >= 1
 cov_233_107: frac[17,107] + frac[25,107] + frac[34,107] + frac[35,107]
   + frac[37,107] >= 1
 cov_234_110: frac[16,110] + frac[18,110] + frac[34,110] >= 0.8
 cov_235_90: frac[17,90] + frac[18,90] + frac[34,90] >= 0.9
 cov_236_110: frac[16,110] + frac[31,110] + frac[34,110] + frac[35,110]
   + frac[37,110] >= 0.9
 cov_237_93: frac[13,93] + frac[26,93] + frac[33,93] + frac[35,93]
   + frac[36,93] + frac[37,93] + frac[38,93] >= 0.7
 cov_238_113: frac[25,113] + frac[31,113] + frac[35,113] >= 1
 cov_239_104: frac[16,104] + frac[29,104] + frac[34,104] + frac[35,104]
   + frac[37,104] >= 0.8
 accuracy_33_55: - 6 frac[33,55] + mem[33,55] = 0
 accuracy_36_68: - 6 frac[36,68] + mem[36,68] = 0
 accuracy_32_44: - 6 frac[32,44] + mem[32,44] = 0
 accuracy_9_62: - 5.09765625 frac[9,62] + mem[9,62] = 0
 accuracy_33_41: - 6 frac[33,41] + mem[33,41] = 0
 accuracy_30_51: - 5.09765625 frac[30,51] + mem[30,51] = 0
 accuracy_1_64: - 5.09765625 frac[1,64] + mem[1,64] = 0
 accuracy_37_35: - 6 frac[37,35] + mem[37,35] = 0
 accuracy_33_3: - 6 frac[33,3] + mem[33,3] = 0
 accuracy_5_74: - 5.09765625 frac[5,74] + mem[5,74] = 0
 accuracy_36_8: - 6 frac[36,8] + mem[36,8] = 0
 accuracy_10_7: - 5.09765625 frac[10,7] + mem[10,7] = 0
 accuracy_34_8: - 6 frac[34,8] + mem[34,8] = 0
 accuracy_11_22: - 5.09765625 frac[11,22] + mem[11,22] = 0
 accuracy_35_21: - 6 frac[35,21] + mem[35,21] = 0
 accuracy_0_17: - 5.09765625 frac[0,17] + mem[0,17] = 0
 accuracy_33_101: - 6 frac[33,101] + mem[33,101] = 0
 accuracy_36_22: - 6 frac[36,22] + mem[36,22] = 0
 accuracy_17_90: - 5.09765625 frac[17,90] + mem[17,90] = 0
 accuracy_32_87: - 6 frac[32,87] + mem[32,87] = 0
 accuracy_33_95: - 6 frac[33,95] + mem[33,95] = 0
 accuracy_36_101: - 6 frac[36,101] + mem[36,101] = 0
 accuracy_37_113: - 6 frac[37,113] + mem[37,113] = 0
 accuracy_2_53: - 5.09765625 frac[2,53] + mem[2,53] = 0
 accuracy_32_77: - 6 frac[32,77] + mem[32,77] = 0
 accuracy_13_101: - 5.09765625 frac[13,101] + mem[13,101] = 0
 accuracy_27_39: - 5.09765625 frac[27,39] + mem[27,39] = 0
 accuracy_18_112: - 5.09765625 frac[18,112] + mem[18,112] = 0
 accuracy_38_119: - 6 frac[38,119] + mem[38,119] = 0
 accuracy_36_67: - 6 frac[36,67] + mem[36,67] = 0
 accuracy_19_13: - 5.09765625 frac[19,13] + mem[19,13] = 0
 accuracy_34_79: - 6 frac[34,79] + mem[34,79] = 0
 accuracy_37_91: - 6 frac[37,91] + mem[37,91] = 0
 accuracy_32_35: - 6 frac[32,35] + mem[32,35] = 0
 accuracy_35_46: - 6 frac[35,46] + mem[35,46] = 0
 accuracy_23_19: - 5.09765625 frac[23,19] + mem[23,19] = 0
 accuracy_6_41: - 5.09765625 frac[6,41] + mem[6,41] = 0
 accuracy_30_58: - 5.09765625 frac[30,58] + mem[30,58] = 0
 accuracy_36_41: - 6 frac[36,41] + mem[36,41] = 0
 accuracy_1_89: - 5.09765625 frac[1,89] + mem[1,89] = 0
 accuracy_38_73: - 6 frac[38,73] + mem[38,73] = 0
 accuracy_32_25: - 6 frac[32,25] + mem[32,25] = 0
 accuracy_22_12: - 5.09765625 frac[22,12] + mem[22,12] = 0
 accuracy_37_44: - 6 frac[37,44] + mem[37,44] = 0
 accuracy_7_44: - 5.09765625 frac[7,44] + mem[7,44] = 0
 accuracy_35_52: - 6 frac[35,52] + mem[35,52] = 0
 accuracy_38_35: - 6 frac[38,35] + mem[38,35] = 0
 accuracy_36_55: - 6 frac[36,55] + mem[36,55] = 0
 accuracy_4_87: - 5.09765625 frac[4,87] + mem[4,87] = 0
 accuracy_34_3: - 6 frac[34,3] + mem[34,3] = 0
 accuracy_26_109: - 5.09765625 frac[26,109] + mem[26,109] = 0
 accuracy_24_21: - 5.09765625 frac[24,21] + mem[24,21] = 0
 accuracy_36_29: - 6 frac[36,29] + mem[36,29] = 0
 accuracy_31_103: - 5.09765625 frac[31,103] + mem[31,103] = 0
 accuracy_37_16: - 6 frac[37,16] + mem[37,16] = 0
 accuracy_34_114: - 6 frac[34,114] + mem[34,114] = 0
 accuracy_35_96: - 6 frac[35,96] + mem[35,96] = 0
 accuracy_33_80: - 6 frac[33,80] + mem[33,80] = 0
 accuracy_38_31: - 6 frac[38,31] + mem[38,31] = 0
 accuracy_28_1: - 5.09765625 frac[28,1] + mem[28,1] = 0
 accuracy_32_68: - 6 frac[32,68] + mem[32,68] = 0
 accuracy_33_74: - 6 frac[33,74] + mem[33,74] = 0
 accuracy_9_70: - 5.09765625 frac[9,70] + mem[9,70] = 0
 accuracy_18_107: - 5.09765625 frac[18,107] + mem[18,107] = 0
 accuracy_38_110: - 6 frac[38,110] + mem[38,110] = 0
 accuracy_36_74: - 6 frac[36,74] + mem[36,74] = 0
 accuracy_2_38: - 5.09765625 frac[2,38] + mem[2,38] = 0
 accuracy_34_70: - 6 frac[34,70] + mem[34,70] = 0
 accuracy_32_58: - 6 frac[32,58] + mem[32,58] = 0
 accuracy_3_55: - 5.09765625 frac[3,55] + mem[3,55] = 0
 accuracy_33_59: - 6 frac[33,59] + mem[33,59] = 0
 accuracy_38_64: - 6 frac[38,64] + mem[38,64] = 0
 accuracy_32_16: - 6 frac[32,16] + mem[32,16] = 0
 accuracy_22_7: - 5.09765625 frac[22,7] + mem[22,7] = 0
 accuracy_7_37: - 5.09765625 frac[7,37] + mem[7,37] = 0
 accuracy_33_29: - 6 frac[33,29] + mem[33,29] = 0
 accuracy_23_6: - 5.09765625 frac[23,6] + mem[23,6] = 0
 accuracy_6_58: - 5.09765625 frac[6,58] + mem[6,58] = 0
 accuracy_38_90: - 6 frac[38,90] + mem[38,90] = 0
 accuracy_36_62: - 6 frac[36,62] + mem[36,62] = 0
 accuracy_33_119: - 6 frac[33,119] + mem[33,119] = 0
 accuracy_5_70: - 5.09765625 frac[5,70] + mem[5,70] = 0
 accuracy_38_60: - 6 frac[38,60] + mem[38,60] = 0
 accuracy_10_19: - 5.09765625 frac[10,19] + mem[10,19] = 0
 accuracy_37_25: - 6 frac[37,25] + mem[37,25] = 0
 accuracy_35_105: - 6 frac[35,105] + mem[35,105] = 0
 accuracy_33_105: - 6 frac[33,105] + mem[33,105] = 0
 accuracy_38_22: - 6 frac[38,22] + mem[38,22] = 0
 accuracy_21_77: - 5.09765625 frac[21,77] + mem[21,77] = 0
 accuracy_28_24: - 5.09765625 frac[28,24] + mem[28,24] = 0
 accuracy_34_103: - 6 frac[34,103] + mem[34,103] = 0
 accuracy_33_67: - 6 frac[33,67] + mem[33,67] = 0
 accuracy_9_79: - 5.09765625 frac[9,79] + mem[9,79] = 0
 accuracy_38_97: - 6 frac[38,97] + mem[38,97] = 0
 accuracy_19_27: - 5.09765625 frac[19,27] + mem[19,27] = 0
 accuracy_37_77: - 6 frac[37,77] + mem[37,77] = 0
 accuracy_32_49: - 6 frac[32,49] + mem[32,49] = 0
 accuracy_27_43: - 5.09765625 frac[27,43] + mem[27,43] = 0
 accuracy_33_60: - 6 frac[33,60] + mem[33,60] = 0
 accuracy_4_63: - 5.09765625 frac[4,63] + mem[4,63] = 0
 accuracy_37_87: - 6 frac[37,87] + mem[37,87] = 0
 accuracy_36_95: - 6 frac[36,95] + mem[36,95] = 0
 accuracy_35_34: - 6 frac[35,34] + mem[35,34] = 0
 accuracy_33_22: - 6 frac[33,22] + mem[33,22] = 0
 accuracy_25_113: - 5.09765625 frac[25,113] + mem[25,113] = 0
 accuracy_38_93: - 6 frac[38,93] + mem[38,93] = 0
 accuracy_36_37: - 6 frac[36,37] + mem[36,37] = 0
 accuracy_1_77: - 5.09765625 frac[1,77] + mem[1,77] = 0
 accuracy_34_21: - 6 frac[34,21] + mem[34,21] = 0
 accuracy_32_13: - 6 frac[32,13] + mem[32,13] = 0
 accuracy_37_56: - 6 frac[37,56] + mem[37,56] = 0
 accuracy_35_8: - 6 frac[35,8] + mem[35,8] = 0
 accuracy_33_8: - 6 frac[33,8] + mem[33,8] = 0
 accuracy_38_55: - 6 frac[38,55] + mem[38,55] = 0
 accuracy_36_3: - 6 frac[36,3] + mem[36,3] = 0
 accuracy_29_101: - 5.09765625 frac[29,101] + mem[29,101] = 0
 accuracy_35_110: - 6 frac[35,110] + mem[35,110] = 0
 accuracy_24_9: - 5.09765625 frac[24,9] + mem[24,9] = 0
 accuracy_33_68: - 6 frac[33,68] + mem[33,68] = 0
 accuracy_18_93: - 5.09765625 frac[18,93] + mem[18,93] = 0
 accuracy_16_117: - 5.09765625 frac[16,117] + mem[16,117] = 0
 accuracy_15_39: - 5.09765625 frac[15,39] + mem[15,39] = 0
 accuracy_35_90: - 6 frac[35,90] + mem[35,90] = 0
 accuracy_32_72: - 6 frac[32,72] + mem[32,72] = 0
 accuracy_13_106: - 5.09765625 frac[13,106] + mem[13,106] = 0
 accuracy_14_72: - 5.09765625 frac[14,72] + mem[14,72] = 0
 accuracy_27_32: - 5.09765625 frac[27,32] + mem[27,32] = 0
 accuracy_36_70: - 6 frac[36,70] + mem[36,70] = 0
 accuracy_34_50: - 6 frac[34,50] + mem[34,50] = 0
 accuracy_32_46: - 6 frac[32,46] + mem[32,46] = 0
 accuracy_22_17: - 5.09765625 frac[22,17] + mem[22,17] = 0
 accuracy_35_43: - 6 frac[35,43] + mem[35,43] = 0
 accuracy_33_47: - 6 frac[33,47] + mem[33,47] = 0
 accuracy_38_84: - 6 frac[38,84] + mem[38,84] = 0
 accuracy_30_49: - 5.09765625 frac[30,49] + mem[30,49] = 0
 accuracy_36_44: - 6 frac[36,44] + mem[36,44] = 0
 accuracy_34_44: - 6 frac[34,44] + mem[34,44] = 0
 accuracy_7_49: - 5.09765625 frac[7,49] + mem[7,49] = 0
 accuracy_22_11: - 5.09765625 frac[22,11] + mem[22,11] = 0
 accuracy_35_49: - 6 frac[35,49] + mem[35,49] = 0
 accuracy_33_1: - 6 frac[33,1] + mem[33,1] = 0
 accuracy_5_72: - 5.09765625 frac[5,72] + mem[5,72] = 0
 accuracy_38_46: - 6 frac[38,46] + mem[38,46] = 0
 accuracy_34_6: - 6 frac[34,6] + mem[34,6] = 0
 accuracy_37_11: - 6 frac[37,11] + mem[37,11] = 0
 accuracy_11_8: - 5.09765625 frac[11,8] + mem[11,8] = 0
 accuracy_24_16: - 5.09765625 frac[24,16] + mem[24,16] = 0
 accuracy_38_0: - 6 frac[38,0] + mem[38,0] = 0
 accuracy_36_16: - 6 frac[36,16] + mem[36,16] = 0
 accuracy_37_21: - 6 frac[37,21] + mem[37,21] = 0
 accuracy_33_93: - 6 frac[33,93] + mem[33,93] = 0
 accuracy_8_89: - 5.09765625 frac[8,89] + mem[8,89] = 0
 accuracy_28_4: - 5.09765625 frac[28,4] + mem[28,4] = 0
 accuracy_14_83: - 5.09765625 frac[14,83] + mem[14,83] = 0
 accuracy_20_34: - 5.09765625 frac[20,34] + mem[20,34] = 0
 accuracy_12_71: - 5.09765625 frac[12,71] + mem[12,71] = 0
 accuracy_9_67: - 5.09765625 frac[9,67] + mem[9,67] = 0
 accuracy_18_110: - 5.09765625 frac[18,110] + mem[18,110] = 0
 accuracy_38_117: - 6 frac[38,117] + mem[38,117] = 0
 accuracy_36_77: - 6 frac[36,77] + mem[36,77] = 0
 accuracy_37_89: - 6 frac[37,89] + mem[37,89] = 0
 accuracy_34_77: - 6 frac[34,77] + mem[34,77] = 0
 accuracy_32_37: - 6 frac[32,37] + mem[32,37] = 0
 accuracy_3_48: - 5.09765625 frac[3,48] + mem[3,48] = 0
 accuracy_38_79: - 6 frac[38,79] + mem[38,79] = 0
 accuracy_30_56: - 5.09765625 frac[30,56] + mem[30,56] = 0
 accuracy_36_43: - 6 frac[36,43] + mem[36,43] = 0
 accuracy_4_75: - 5.09765625 frac[4,75] + mem[4,75] = 0
 accuracy_34_39: - 6 frac[34,39] + mem[34,39] = 0
 accuracy_32_27: - 6 frac[32,27] + mem[32,27] = 0
 accuracy_29_93: - 5.09765625 frac[29,93] + mem[29,93] = 0
 accuracy_23_11: - 5.09765625 frac[23,11] + mem[23,11] = 0
 accuracy_36_49: - 6 frac[36,49] + mem[36,49] = 0
 accuracy_37_52: - 6 frac[37,52] + mem[37,52] = 0
 accuracy_26_107: - 5.09765625 frac[26,107] + mem[26,107] = 0
 accuracy_36_31: - 6 frac[36,31] + mem[36,31] = 0
 accuracy_31_97: - 5.09765625 frac[31,97] + mem[31,97] = 0
 accuracy_37_30: - 6 frac[37,30] + mem[37,30] = 0
 accuracy_34_112: - 6 frac[34,112] + mem[34,112] = 0
 accuracy_35_98: - 6 frac[35,98] + mem[35,98] = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 accuracy_38_29: - 6 frac[38,29] + mem[38,29] = 0
 accuracy_8_64: - 5.09765625 frac[8,64] + mem[8,64] = 0
 accuracy_21_66: - 5.09765625 frac[21,66] + mem[21,66] = 0
 accuracy_17_107: - 5.09765625 frac[17,107] + mem[17,107] = 0
 accuracy_36_110: - 6 frac[36,110] + mem[36,110] = 0
 accuracy_34_106: - 6 frac[34,106] + mem[34,106] = 0
 accuracy_32_70: - 6 frac[32,70] + mem[32,70] = 0
 accuracy_33_72: - 6 frac[33,72] + mem[33,72] = 0
 accuracy_18_105: - 5.09765625 frac[18,105] + mem[18,105] = 0
 accuracy_37_66: - 6 frac[37,66] + mem[37,66] = 0
 accuracy_2_36: - 5.09765625 frac[2,36] + mem[2,36] = 0
 accuracy_32_60: - 6 frac[32,60] + mem[32,60] = 0
 accuracy_3_57: - 5.09765625 frac[3,57] + mem[3,57] = 0
 accuracy_12_84: - 5.09765625 frac[12,84] + mem[12,84] = 0
 accuracy_6_38: - 5.09765625 frac[6,38] + mem[6,38] = 0
 accuracy_38_70: - 6 frac[38,70] + mem[38,70] = 0
 accuracy_34_62: - 6 frac[34,62] + mem[34,62] = 0
 accuracy_7_39: - 5.09765625 frac[7,39] + mem[7,39] = 0
 accuracy_33_19: - 6 frac[33,19] + mem[33,19] = 0
 accuracy_23_0: - 5.09765625 frac[23,0] + mem[23,0] = 0
 accuracy_25_110: - 5.09765625 frac[25,110] + mem[25,110] = 0
 accuracy_36_56: - 6 frac[36,56] + mem[36,56] = 0
 accuracy_34_24: - 6 frac[34,24] + mem[34,24] = 0
 accuracy_32_8: - 6 frac[32,8] + mem[32,8] = 0
 accuracy_37_61: - 6 frac[37,61] + mem[37,61] = 0
 accuracy_29_112: - 5.09765625 frac[29,112] + mem[29,112] = 0
 accuracy_35_5: - 6 frac[35,5] + mem[35,5] = 0
 accuracy_26_114: - 5.09765625 frac[26,114] + mem[26,114] = 0
 accuracy_33_117: - 6 frac[33,117] + mem[33,117] = 0
 accuracy_5_68: - 5.09765625 frac[5,68] + mem[5,68] = 0
 accuracy_38_50: - 6 frac[38,50] + mem[38,50] = 0
 accuracy_36_6: - 6 frac[36,6] + mem[36,6] = 0
 accuracy_10_17: - 5.09765625 frac[10,17] + mem[10,17] = 0
 accuracy_31_110: - 5.09765625 frac[31,110] + mem[31,110] = 0
 accuracy_37_7: - 6 frac[37,7] + mem[37,7] = 0
 accuracy_11_28: - 5.09765625 frac[11,28] + mem[11,28] = 0
 accuracy_35_107: - 6 frac[35,107] + mem[35,107] = 0
 accuracy_29_106: - 5.09765625 frac[29,106] + mem[29,106] = 0
 accuracy_33_111: - 6 frac[33,111] + mem[33,111] = 0
 accuracy_38_20: - 6 frac[38,20] + mem[38,20] = 0
 accuracy_15_50: - 5.09765625 frac[15,50] + mem[15,50] = 0
 accuracy_37_97: - 6 frac[37,97] + mem[37,97] = 0
 accuracy_35_113: - 6 frac[35,113] + mem[35,113] = 0
 accuracy_33_65: - 6 frac[33,65] + mem[33,65] = 0
 accuracy_20_48: - 5.09765625 frac[20,48] + mem[20,48] = 0
 accuracy_27_55: - 5.09765625 frac[27,55] + mem[27,55] = 0
 accuracy_9_77: - 5.09765625 frac[9,77] + mem[9,77] = 0
 accuracy_32_51: - 6 frac[32,51] + mem[32,51] = 0
 accuracy_13_111: - 5.09765625 frac[13,111] + mem[13,111] = 0
 accuracy_20_30: - 5.09765625 frac[20,30] + mem[20,30] = 0
 accuracy_9_87: - 5.09765625 frac[9,87] + mem[9,87] = 0
 accuracy_33_50: - 6 frac[33,50] + mem[33,50] = 0
 accuracy_36_89: - 6 frac[36,89] + mem[36,89] = 0
 accuracy_32_41: - 6 frac[32,41] + mem[32,41] = 0
 accuracy_5_60: - 5.09765625 frac[5,60] + mem[5,60] = 0
 accuracy_33_20: - 6 frac[33,20] + mem[33,20] = 0
 accuracy_25_119: - 5.09765625 frac[25,119] + mem[25,119] = 0
 accuracy_6_51: - 5.09765625 frac[6,51] + mem[6,51] = 0
 accuracy_36_39: - 6 frac[36,39] + mem[36,39] = 0
 accuracy_4_71: - 5.09765625 frac[4,71] + mem[4,71] = 0
 accuracy_34_19: - 6 frac[34,19] + mem[34,19] = 0
 accuracy_37_38: - 6 frac[37,38] + mem[37,38] = 0
 accuracy_33_14: - 6 frac[33,14] + mem[33,14] = 0
 accuracy_36_13: - 6 frac[36,13] + mem[36,13] = 0
 accuracy_10_24: - 5.09765625 frac[10,24] + mem[10,24] = 0
 accuracy_34_13: - 6 frac[34,13] + mem[34,13] = 0
 accuracy_37_0: - 6 frac[37,0] + mem[37,0] = 0
 accuracy_35_16: - 6 frac[35,16] + mem[35,16] = 0
 accuracy_26_103: - 5.09765625 frac[26,103] + mem[26,103] = 0
 accuracy_33_96: - 6 frac[33,96] + mem[33,96] = 0
 accuracy_15_59: - 5.09765625 frac[15,59] + mem[15,59] = 0
 accuracy_21_84: - 5.09765625 frac[21,84] + mem[21,84] = 0
 accuracy_8_78: - 5.09765625 frac[8,78] + mem[8,78] = 0
 accuracy_28_17: - 5.09765625 frac[28,17] + mem[28,17] = 0
 accuracy_37_106: - 6 frac[37,106] + mem[37,106] = 0
 accuracy_35_118: - 6 frac[35,118] + mem[35,118] = 0
 accuracy_33_90: - 6 frac[33,90] + mem[33,90] = 0
 accuracy_18_91: - 5.09765625 frac[18,91] + mem[18,91] = 0
 accuracy_8_84: - 5.09765625 frac[8,84] + mem[8,84] = 0
 accuracy_21_62: - 5.09765625 frac[21,62] + mem[21,62] = 0
 accuracy_32_74: - 6 frac[32,74] + mem[32,74] = 0
 accuracy_13_104: - 5.09765625 frac[13,104] + mem[13,104] = 0
 accuracy_20_37: - 5.09765625 frac[20,37] + mem[20,37] = 0
 accuracy_3_39: - 5.09765625 frac[3,39] + mem[3,39] = 0
 accuracy_27_34: - 5.09765625 frac[27,34] + mem[27,34] = 0
 accuracy_12_74: - 5.09765625 frac[12,74] + mem[12,74] = 0
 accuracy_36_64: - 6 frac[36,64] + mem[36,64] = 0
 accuracy_19_8: - 5.09765625 frac[19,8] + mem[19,8] = 0
 accuracy_34_48: - 6 frac[34,48] + mem[34,48] = 0
 accuracy_37_94: - 6 frac[37,94] + mem[37,94] = 0
 accuracy_32_32: - 6 frac[32,32] + mem[32,32] = 0
 accuracy_22_23: - 5.09765625 frac[22,23] + mem[22,23] = 0
 accuracy_26_90: - 5.09765625 frac[26,90] + mem[26,90] = 0
 accuracy_23_22: - 5.09765625 frac[23,22] + mem[23,22] = 0
 accuracy_30_55: - 5.09765625 frac[30,55] + mem[30,55] = 0
 accuracy_36_46: - 6 frac[36,46] + mem[36,46] = 0
 accuracy_4_78: - 5.09765625 frac[4,78] + mem[4,78] = 0
 accuracy_32_6: - 6 frac[32,6] + mem[32,6] = 0
 accuracy_13_92: - 5.09765625 frac[13,92] + mem[13,92] = 0
 accuracy_35_51: - 6 frac[35,51] + mem[35,51] = 0
 accuracy_33_7: - 6 frac[33,7] + mem[33,7] = 0
 accuracy_38_44: - 6 frac[38,44] + mem[38,44] = 0
 accuracy_36_52: - 6 frac[36,52] + mem[36,52] = 0
 accuracy_10_3: - 5.09765625 frac[10,3] + mem[10,3] = 0
 accuracy_34_4: - 6 frac[34,4] + mem[34,4] = 0
 accuracy_35_25: - 6 frac[35,25] + mem[35,25] = 0
 accuracy_26_110: - 5.09765625 frac[26,110] + mem[26,110] = 0
 accuracy_38_6: - 6 frac[38,6] + mem[38,6] = 0
 accuracy_37_19: - 6 frac[37,19] + mem[37,19] = 0
 accuracy_33_83: - 6 frac[33,83] + mem[33,83] = 0
 accuracy_38_24: - 6 frac[38,24] + mem[38,24] = 0
 accuracy_16_110: - 5.09765625 frac[16,110] + mem[16,110] = 0
 accuracy_17_112: - 5.09765625 frac[17,112] + mem[17,112] = 0
 accuracy_21_71: - 5.09765625 frac[21,71] + mem[21,71] = 0
 accuracy_36_97: - 6 frac[36,97] + mem[36,97] = 0
 accuracy_32_65: - 6 frac[32,65] + mem[32,65] = 0
 accuracy_13_97: - 5.09765625 frac[13,97] + mem[13,97] = 0
 accuracy_20_44: - 5.09765625 frac[20,44] + mem[20,44] = 0
 accuracy_3_44: - 5.09765625 frac[3,44] + mem[3,44] = 0
 accuracy_25_95: - 5.09765625 frac[25,95] + mem[25,95] = 0
 accuracy_36_79: - 6 frac[36,79] + mem[36,79] = 0
 accuracy_19_17: - 5.09765625 frac[19,17] + mem[19,17] = 0
 accuracy_2_43: - 5.09765625 frac[2,43] + mem[2,43] = 0
 accuracy_37_71: - 6 frac[37,71] + mem[37,71] = 0
 accuracy_32_39: - 6 frac[32,39] + mem[32,39] = 0
 accuracy_3_50: - 5.09765625 frac[3,50] + mem[3,50] = 0
 accuracy_38_77: - 6 frac[38,77] + mem[38,77] = 0
 accuracy_36_85: - 6 frac[36,85] + mem[36,85] = 0
 accuracy_34_37: - 6 frac[34,37] + mem[34,37] = 0
 accuracy_7_40: - 5.09765625 frac[7,40] + mem[7,40] = 0
 accuracy_22_0: - 5.09765625 frac[22,0] + mem[22,0] = 0
 accuracy_29_91: - 5.09765625 frac[29,91] + mem[29,91] = 0
 accuracy_35_56: - 6 frac[35,56] + mem[35,56] = 0
 accuracy_33_24: - 6 frac[33,24] + mem[33,24] = 0
 accuracy_23_5: - 5.09765625 frac[23,5] + mem[23,5] = 0
 accuracy_38_39: - 6 frac[38,39] + mem[38,39] = 0
 accuracy_25_107: - 5.09765625 frac[25,107] + mem[25,107] = 0
 accuracy_36_51: - 6 frac[36,51] + mem[36,51] = 0
 accuracy_37_50: - 6 frac[37,50] + mem[37,50] = 0
 accuracy_11_3: - 5.09765625 frac[11,3] + mem[11,3] = 0
 accuracy_35_30: - 6 frac[35,30] + mem[35,30] = 0
 accuracy_24_25: - 5.09765625 frac[24,25] + mem[24,25] = 0
 accuracy_36_25: - 6 frac[36,25] + mem[36,25] = 0
 accuracy_31_99: - 5.09765625 frac[31,99] + mem[31,99] = 0
 accuracy_11_25: - 5.09765625 frac[11,25] + mem[11,25] = 0
 accuracy_35_100: - 6 frac[35,100] + mem[35,100] = 0
 accuracy_29_111: - 5.09765625 frac[29,111] + mem[29,111] = 0
 accuracy_33_84: - 6 frac[33,84] + mem[33,84] = 0
 accuracy_38_19: - 6 frac[38,19] + mem[38,19] = 0
 accuracy_15_55: - 5.09765625 frac[15,55] + mem[15,55] = 0
 accuracy_21_64: - 5.09765625 frac[21,64] + mem[21,64] = 0
 accuracy_36_104: - 6 frac[36,104] + mem[36,104] = 0
 accuracy_34_104: - 6 frac[34,104] + mem[34,104] = 0
 accuracy_32_88: - 6 frac[32,88] + mem[32,88] = 0
 accuracy_33_78: - 6 frac[33,78] + mem[33,78] = 0
 accuracy_14_88: - 5.09765625 frac[14,88] + mem[14,88] = 0
 accuracy_27_48: - 5.09765625 frac[27,48] + mem[27,48] = 0
 accuracy_18_103: - 5.09765625 frac[18,103] + mem[18,103] = 0
 accuracy_30_31: - 5.09765625 frac[30,31] + mem[30,31] = 0
 accuracy_19_22: - 5.09765625 frac[19,22] + mem[19,22] = 0
 accuracy_2_34: - 5.09765625 frac[2,34] + mem[2,34] = 0
 accuracy_34_66: - 6 frac[34,66] + mem[34,66] = 0
 accuracy_32_62: - 6 frac[32,62] + mem[32,62] = 0
 accuracy_37_64: - 6 frac[37,64] + mem[37,64] = 0
 accuracy_14_66: - 5.09765625 frac[14,66] + mem[14,66] = 0
 accuracy_3_59: - 5.09765625 frac[3,59] + mem[3,59] = 0
 accuracy_33_63: - 6 frac[33,63] + mem[33,63] = 0
 accuracy_6_36: - 5.09765625 frac[6,36] + mem[6,36] = 0
 accuracy_4_60: - 5.09765625 frac[4,60] + mem[4,60] = 0
 accuracy_36_92: - 6 frac[36,92] + mem[36,92] = 0
 accuracy_34_60: - 6 frac[34,60] + mem[34,60] = 0
 accuracy_7_33: - 5.09765625 frac[7,33] + mem[7,33] = 0
 accuracy_22_27: - 5.09765625 frac[22,27] + mem[22,27] = 0
 accuracy_33_17: - 6 frac[33,17] + mem[33,17] = 0
 accuracy_36_58: - 6 frac[36,58] + mem[36,58] = 0
 accuracy_34_22: - 6 frac[34,22] + mem[34,22] = 0
 accuracy_35_7: - 6 frac[35,7] + mem[35,7] = 0
 accuracy_26_112: - 5.09765625 frac[26,112] + mem[26,112] = 0
 accuracy_33_11: - 6 frac[33,11] + mem[33,11] = 0
 accuracy_38_48: - 6 frac[38,48] + mem[38,48] = 0
 accuracy_36_0: - 6 frac[36,0] + mem[36,0] = 0
 accuracy_37_5: - 6 frac[37,5] + mem[37,5] = 0
 accuracy_29_104: - 5.09765625 frac[29,104] + mem[29,104] = 0
 accuracy_35_109: - 6 frac[35,109] + mem[35,109] = 0
 accuracy_0_25: - 5.09765625 frac[0,25] + mem[0,25] = 0
 accuracy_24_6: - 5.09765625 frac[24,6] + mem[24,6] = 0
 accuracy_33_109: - 6 frac[33,109] + mem[33,109] = 0
 accuracy_16_92: - 5.09765625 frac[16,92] + mem[16,92] = 0
 accuracy_8_73: - 5.09765625 frac[8,73] + mem[8,73] = 0
 accuracy_21_73: - 5.09765625 frac[21,73] + mem[21,73] = 0
 accuracy_37_111: - 6 frac[37,111] + mem[37,111] = 0
 accuracy_35_115: - 6 frac[35,115] + mem[35,115] = 0
 accuracy_33_71: - 6 frac[33,71] + mem[33,71] = 0
 accuracy_20_50: - 5.09765625 frac[20,50] + mem[20,50] = 0
 accuracy_18_94: - 5.09765625 frac[18,94] + mem[18,94] = 0
 accuracy_1_62: - 5.09765625 frac[1,62] + mem[1,62] = 0
 accuracy_38_101: - 6 frac[38,101] + mem[38,101] = 0
 accuracy_37_73: - 6 frac[37,73] + mem[37,73] = 0
 accuracy_34_93: - 6 frac[34,93] + mem[34,93] = 0
 accuracy_32_53: - 6 frac[32,53] + mem[32,53] = 0
 accuracy_13_109: - 5.09765625 frac[13,109] + mem[13,109] = 0
 accuracy_14_69: - 5.09765625 frac[14,69] + mem[14,69] = 0
 accuracy_12_77: - 5.09765625 frac[12,77] + mem[12,77] = 0
 accuracy_9_85: - 5.09765625 frac[9,85] + mem[9,85] = 0
 accuracy_6_31: - 5.09765625 frac[6,31] + mem[6,31] = 0
 accuracy_36_91: - 6 frac[36,91] + mem[36,91] = 0
 accuracy_34_55: - 6 frac[34,55] + mem[34,55] = 0
 accuracy_8_62: - 5.09765625 frac[8,62] + mem[8,62] = 0
 accuracy_32_43: - 6 frac[32,43] + mem[32,43] = 0
 accuracy_35_38: - 6 frac[35,38] + mem[35,38] = 0
 accuracy_34_17: - 6 frac[34,17] + mem[34,17] = 0
 accuracy_35_12: - 6 frac[35,12] + mem[35,12] = 0
 accuracy_5_75: - 5.09765625 frac[5,75] + mem[5,75] = 0
 accuracy_38_43: - 6 frac[38,43] + mem[38,43] = 0
 accuracy_34_11: - 6 frac[34,11] + mem[34,11] = 0
 accuracy_15_31: - 5.09765625 frac[15,31] + mem[15,31] = 0
 accuracy_37_14: - 6 frac[37,14] + mem[37,14] = 0
 accuracy_0_16: - 5.09765625 frac[0,16] + mem[0,16] = 0
 accuracy_5_85: - 5.09765625 frac[5,85] + mem[5,85] = 0
 accuracy_37_104: - 6 frac[37,104] + mem[37,104] = 0
 accuracy_33_88: - 6 frac[33,88] + mem[33,88] = 0
 accuracy_16_105: - 5.09765625 frac[16,105] + mem[16,105] = 0
 accuracy_21_60: - 5.09765625 frac[21,60] + mem[21,60] = 0
 accuracy_36_100: - 6 frac[36,100] + mem[36,100] = 0
 accuracy_28_9: - 5.09765625 frac[28,9] + mem[28,9] = 0
 accuracy_34_84: - 6 frac[34,84] + mem[34,84] = 0
 accuracy_35_94: - 6 frac[35,94] + mem[35,94] = 0
 accuracy_37_114: - 6 frac[37,114] + mem[37,114] = 0
 accuracy_14_76: - 5.09765625 frac[14,76] + mem[14,76] = 0
 accuracy_20_39: - 5.09765625 frac[20,39] + mem[20,39] = 0
 accuracy_3_41: - 5.09765625 frac[3,41] + mem[3,41] = 0
 accuracy_12_68: - 5.09765625 frac[12,68] + mem[12,68] = 0
 accuracy_18_115: - 5.09765625 frac[18,115] + mem[18,115] = 0
 accuracy_36_66: - 6 frac[36,66] + mem[36,66] = 0
 accuracy_37_92: - 6 frac[37,92] + mem[37,92] = 0
 accuracy_32_34: - 6 frac[32,34] + mem[32,34] = 0
 accuracy_38_72: - 6 frac[38,72] + mem[38,72] = 0
 accuracy_4_72: - 5.09765625 frac[4,72] + mem[4,72] = 0
 accuracy_32_24: - 6 frac[32,24] + mem[32,24] = 0
 accuracy_13_90: - 5.09765625 frac[13,90] + mem[13,90] = 0
 accuracy_23_14: - 5.09765625 frac[23,14] + mem[23,14] = 0
 accuracy_10_1: - 5.09765625 frac[10,1] + mem[10,1] = 0
 accuracy_31_94: - 5.09765625 frac[31,94] + mem[31,94] = 0
 accuracy_37_55: - 6 frac[37,55] + mem[37,55] = 0
 accuracy_31_100: - 5.09765625 frac[31,100] + mem[31,100] = 0
 accuracy_37_17: - 6 frac[37,17] + mem[37,17] = 0
 accuracy_34_117: - 6 frac[34,117] + mem[34,117] = 0
 accuracy_35_97: - 6 frac[35,97] + mem[35,97] = 0
 accuracy_0_13: - 5.09765625 frac[0,13] + mem[0,13] = 0
 accuracy_16_96: - 5.09765625 frac[16,96] + mem[16,96] = 0
 accuracy_28_0: - 5.09765625 frac[28,0] + mem[28,0] = 0
 accuracy_34_111: - 6 frac[34,111] + mem[34,111] = 0
 accuracy_32_67: - 6 frac[32,67] + mem[32,67] = 0
 accuracy_33_75: - 6 frac[33,75] + mem[33,75] = 0
 accuracy_3_46: - 5.09765625 frac[3,46] + mem[3,46] = 0
 accuracy_9_71: - 5.09765625 frac[9,71] + mem[9,71] = 0
 accuracy_12_67: - 5.09765625 frac[12,67] + mem[12,67] = 0
 accuracy_25_93: - 5.09765625 frac[25,93] + mem[25,93] = 0
 accuracy_38_105: - 6 frac[38,105] + mem[38,105] = 0
 accuracy_18_106: - 5.09765625 frac[18,106] + mem[18,106] = 0
 accuracy_36_73: - 6 frac[36,73] + mem[36,73] = 0
 accuracy_19_19: - 5.09765625 frac[19,19] + mem[19,19] = 0
 accuracy_34_73: - 6 frac[34,73] + mem[34,73] = 0
 accuracy_32_57: - 6 frac[32,57] + mem[32,57] = 0
 accuracy_12_89: - 5.09765625 frac[12,89] + mem[12,89] = 0
 accuracy_36_87: - 6 frac[36,87] + mem[36,87] = 0
 accuracy_1_83: - 5.09765625 frac[1,83] + mem[1,83] = 0
 accuracy_34_35: - 6 frac[34,35] + mem[34,35] = 0
 accuracy_32_31: - 6 frac[32,31] + mem[32,31] = 0
 accuracy_22_6: - 5.09765625 frac[22,6] + mem[22,6] = 0
 accuracy_35_58: - 6 frac[35,58] + mem[35,58] = 0
 accuracy_23_7: - 5.09765625 frac[23,7] + mem[23,7] = 0
 accuracy_38_37: - 6 frac[38,37] + mem[38,37] = 0
 accuracy_25_105: - 5.09765625 frac[25,105] + mem[25,105] = 0
 accuracy_36_61: - 6 frac[36,61] + mem[36,61] = 0
 accuracy_34_29: - 6 frac[34,29] + mem[34,29] = 0
 accuracy_37_48: - 6 frac[37,48] + mem[37,48] = 0
 accuracy_35_0: - 6 frac[35,0] + mem[35,0] = 0
 accuracy_5_71: - 5.09765625 frac[5,71] + mem[5,71] = 0
 accuracy_36_27: - 6 frac[36,27] + mem[36,27] = 0
 accuracy_35_102: - 6 frac[35,102] + mem[35,102] = 0
 accuracy_33_106: - 6 frac[33,106] + mem[33,106] = 0
 accuracy_38_17: - 6 frac[38,17] + mem[38,17] = 0
 accuracy_16_103: - 5.09765625 frac[16,103] + mem[16,103] = 0
 accuracy_17_111: - 5.09765625 frac[17,111] + mem[17,111] = 0
 accuracy_36_106: - 6 frac[36,106] + mem[36,106] = 0
 accuracy_37_100: - 6 frac[37,100] + mem[37,100] = 0
 accuracy_34_102: - 6 frac[34,102] + mem[34,102] = 0
 accuracy_33_76: - 6 frac[33,76] + mem[33,76] = 0
 accuracy_27_50: - 5.09765625 frac[27,50] + mem[27,50] = 0
 accuracy_38_96: - 6 frac[38,96] + mem[38,96] = 0
 accuracy_19_24: - 5.09765625 frac[19,24] + mem[19,24] = 0
 accuracy_2_32: - 5.09765625 frac[2,32] + mem[2,32] = 0
 accuracy_34_64: - 6 frac[34,64] + mem[34,64] = 0
 accuracy_32_48: - 6 frac[32,48] + mem[32,48] = 0
 accuracy_12_80: - 5.09765625 frac[12,80] + mem[12,80] = 0
 accuracy_9_82: - 5.09765625 frac[9,82] + mem[9,82] = 0
 accuracy_33_61: - 6 frac[33,61] + mem[33,61] = 0
 accuracy_30_39: - 5.09765625 frac[30,39] + mem[30,39] = 0
 accuracy_4_62: - 5.09765625 frac[4,62] + mem[4,62] = 0
 accuracy_32_22: - 6 frac[32,22] + mem[32,22] = 0
 accuracy_22_25: - 5.09765625 frac[22,25] + mem[22,25] = 0
 accuracy_7_35: - 5.09765625 frac[7,35] + mem[7,35] = 0
 accuracy_35_35: - 6 frac[35,35] + mem[35,35] = 0
 accuracy_38_92: - 6 frac[38,92] + mem[38,92] = 0
 accuracy_34_20: - 6 frac[34,20] + mem[34,20] = 0
 accuracy_35_9: - 6 frac[35,9] + mem[35,9] = 0
 accuracy_10_29: - 5.09765625 frac[10,29] + mem[10,29] = 0
 accuracy_34_14: - 6 frac[34,14] + mem[34,14] = 0
 accuracy_37_3: - 6 frac[37,3] + mem[37,3] = 0
 accuracy_11_16: - 5.09765625 frac[11,16] + mem[11,16] = 0
 accuracy_29_102: - 5.09765625 frac[29,102] + mem[29,102] = 0
 accuracy_0_27: - 5.09765625 frac[0,27] + mem[0,27] = 0
 accuracy_24_8: - 5.09765625 frac[24,8] + mem[24,8] = 0
 accuracy_35_111: - 6 frac[35,111] + mem[35,111] = 0
 accuracy_38_8: - 6 frac[38,8] + mem[38,8] = 0
 accuracy_8_75: - 5.09765625 frac[8,75] + mem[8,75] = 0
 accuracy_21_87: - 5.09765625 frac[21,87] + mem[21,87] = 0
 accuracy_31_112: - 5.09765625 frac[31,112] + mem[31,112] = 0
 accuracy_37_109: - 6 frac[37,109] + mem[37,109] = 0
 accuracy_17_96: - 5.09765625 frac[17,96] + mem[17,96] = 0
 accuracy_35_117: - 6 frac[35,117] + mem[35,117] = 0
 accuracy_33_69: - 6 frac[33,69] + mem[33,69] = 0
 accuracy_1_60: - 5.09765625 frac[1,60] + mem[1,60] = 0
 accuracy_16_116: - 5.09765625 frac[16,116] + mem[16,116] = 0
 accuracy_37_119: - 6 frac[37,119] + mem[37,119] = 0
 accuracy_34_91: - 6 frac[34,91] + mem[34,91] = 0
 accuracy_35_91: - 6 frac[35,91] + mem[35,91] = 0
 accuracy_32_55: - 6 frac[32,55] + mem[32,55] = 0
 accuracy_30_46: - 5.09765625 frac[30,46] + mem[30,46] = 0
 accuracy_36_69: - 6 frac[36,69] + mem[36,69] = 0
 accuracy_23_21: - 5.09765625 frac[23,21] + mem[23,21] = 0
 accuracy_6_55: - 5.09765625 frac[6,55] + mem[6,55] = 0
 accuracy_38_87: - 6 frac[38,87] + mem[38,87] = 0
 accuracy_36_35: - 6 frac[36,35] + mem[36,35] = 0
 accuracy_4_67: - 5.09765625 frac[4,67] + mem[4,67] = 0
 accuracy_32_3: - 6 frac[32,3] + mem[32,3] = 0
 accuracy_13_95: - 5.09765625 frac[13,95] + mem[13,95] = 0
 accuracy_37_34: - 6 frac[37,34] + mem[37,34] = 0
 accuracy_35_14: - 6 frac[35,14] + mem[35,14] = 0
 accuracy_37_12: - 6 frac[37,12] + mem[37,12] = 0
 accuracy_26_99: - 5.09765625 frac[26,99] + mem[26,99] = 0
 accuracy_33_100: - 6 frac[33,100] + mem[33,100] = 0
 accuracy_38_3: - 6 frac[38,3] + mem[38,3] = 0
 accuracy_37_22: - 6 frac[37,22] + mem[37,22] = 0
 accuracy_0_8: - 5.09765625 frac[0,8] + mem[0,8] = 0
 accuracy_17_115: - 5.09765625 frac[17,115] + mem[17,115] = 0
 accuracy_28_11: - 5.09765625 frac[28,11] + mem[28,11] = 0
 accuracy_2_50: - 5.09765625 frac[2,50] + mem[2,50] = 0
 accuracy_37_112: - 6 frac[37,112] + mem[37,112] = 0
 accuracy_32_78: - 6 frac[32,78] + mem[32,78] = 0
 accuracy_13_100: - 5.09765625 frac[13,100] + mem[13,100] = 0
 accuracy_12_70: - 5.09765625 frac[12,70] + mem[12,70] = 0
 accuracy_18_113: - 5.09765625 frac[18,113] + mem[18,113] = 0
 accuracy_37_90: - 6 frac[37,90] + mem[37,90] = 0
 accuracy_32_36: - 6 frac[32,36] + mem[32,36] = 0
 accuracy_3_49: - 5.09765625 frac[3,49] + mem[3,49] = 0
 accuracy_6_46: - 5.09765625 frac[6,46] + mem[6,46] = 0
 accuracy_1_88: - 5.09765625 frac[1,88] + mem[1,88] = 0
 accuracy_34_38: - 6 frac[34,38] + mem[34,38] = 0
 accuracy_7_47: - 5.09765625 frac[7,47] + mem[7,47] = 0
 accuracy_22_13: - 5.09765625 frac[22,13] + mem[22,13] = 0
 accuracy_37_43: - 6 frac[37,43] + mem[37,43] = 0
 accuracy_35_55: - 6 frac[35,55] + mem[35,55] = 0
 accuracy_33_27: - 6 frac[33,27] + mem[33,27] = 0
 accuracy_23_8: - 5.09765625 frac[23,8] + mem[23,8] = 0
 accuracy_38_32: - 6 frac[38,32] + mem[38,32] = 0
 accuracy_36_48: - 6 frac[36,48] + mem[36,48] = 0
 accuracy_34_0: - 6 frac[34,0] + mem[34,0] = 0
 accuracy_11_14: - 5.09765625 frac[11,14] + mem[11,14] = 0
 accuracy_24_22: - 5.09765625 frac[24,22] + mem[24,22] = 0
 accuracy_38_58: - 6 frac[38,58] + mem[38,58] = 0
 accuracy_21_89: - 5.09765625 frac[21,89] + mem[21,89] = 0
 accuracy_37_31: - 6 frac[37,31] + mem[37,31] = 0
 accuracy_34_115: - 6 frac[34,115] + mem[34,115] = 0
 accuracy_35_99: - 6 frac[35,99] + mem[35,99] = 0
 accuracy_33_87: - 6 frac[33,87] + mem[33,87] = 0
 accuracy_17_116: - 5.09765625 frac[17,116] + mem[17,116] = 0
 accuracy_36_109: - 6 frac[36,109] + mem[36,109] = 0
 accuracy_34_109: - 6 frac[34,109] + mem[34,109] = 0
 accuracy_32_69: - 6 frac[32,69] + mem[32,69] = 0
 accuracy_33_73: - 6 frac[33,73] + mem[33,73] = 0
 accuracy_18_104: - 5.09765625 frac[18,104] + mem[18,104] = 0
 accuracy_38_111: - 6 frac[38,111] + mem[38,111] = 0
 accuracy_19_21: - 5.09765625 frac[19,21] + mem[19,21] = 0
 accuracy_2_39: - 5.09765625 frac[2,39] + mem[2,39] = 0
 accuracy_34_71: - 6 frac[34,71] + mem[34,71] = 0
 accuracy_32_59: - 6 frac[32,59] + mem[32,59] = 0
 accuracy_13_119: - 5.09765625 frac[13,119] + mem[13,119] = 0
 accuracy_32_17: - 6 frac[32,17] + mem[32,17] = 0
 accuracy_33_28: - 6 frac[33,28] + mem[33,28] = 0
 accuracy_38_91: - 6 frac[38,91] + mem[38,91] = 0
 accuracy_36_63: - 6 frac[36,63] + mem[36,63] = 0
 accuracy_34_27: - 6 frac[34,27] + mem[34,27] = 0
 accuracy_37_62: - 6 frac[37,62] + mem[37,62] = 0
 accuracy_26_117: - 5.09765625 frac[26,117] + mem[26,117] = 0
 accuracy_5_69: - 5.09765625 frac[5,69] + mem[5,69] = 0
 accuracy_38_61: - 6 frac[38,61] + mem[38,61] = 0
 accuracy_31_111: - 5.09765625 frac[31,111] + mem[31,111] = 0
 accuracy_37_24: - 6 frac[37,24] + mem[37,24] = 0
 accuracy_35_104: - 6 frac[35,104] + mem[35,104] = 0
 accuracy_0_6: - 5.09765625 frac[0,6] + mem[0,6] = 0
 accuracy_33_104: - 6 frac[33,104] + mem[33,104] = 0
 accuracy_27_30: - 5.09765625 frac[27,30] + mem[27,30] = 0
 accuracy_31_117: - 5.09765625 frac[31,117] + mem[31,117] = 0
 accuracy_28_25: - 5.09765625 frac[28,25] + mem[28,25] = 0
 accuracy_34_100: - 6 frac[34,100] + mem[34,100] = 0
 accuracy_33_66: - 6 frac[33,66] + mem[33,66] = 0
 accuracy_20_55: - 5.09765625 frac[20,55] + mem[20,55] = 0
 accuracy_34_94: - 6 frac[34,94] + mem[34,94] = 0
 accuracy_32_50: - 6 frac[32,50] + mem[32,50] = 0
 accuracy_12_82: - 5.09765625 frac[12,82] + mem[12,82] = 0
 accuracy_9_80: - 5.09765625 frac[9,80] + mem[9,80] = 0
 accuracy_30_37: - 5.09765625 frac[30,37] + mem[30,37] = 0
 accuracy_36_88: - 6 frac[36,88] + mem[36,88] = 0
 accuracy_32_40: - 6 frac[32,40] + mem[32,40] = 0
 accuracy_5_61: - 5.09765625 frac[5,61] + mem[5,61] = 0
 accuracy_35_37: - 6 frac[35,37] + mem[35,37] = 0
 accuracy_6_50: - 5.09765625 frac[6,50] + mem[6,50] = 0
 accuracy_36_38: - 6 frac[36,38] + mem[36,38] = 0
 accuracy_37_39: - 6 frac[37,39] + mem[37,39] = 0
 accuracy_35_11: - 6 frac[35,11] + mem[35,11] = 0
 accuracy_38_52: - 6 frac[38,52] + mem[38,52] = 0
 accuracy_12_63: - 5.09765625 frac[12,63] + mem[12,63] = 0
 accuracy_10_27: - 5.09765625 frac[10,27] + mem[10,27] = 0
 accuracy_34_12: - 6 frac[34,12] + mem[34,12] = 0
 accuracy_37_1: - 6 frac[37,1] + mem[37,1] = 0
 accuracy_35_17: - 6 frac[35,17] + mem[35,17] = 0
 accuracy_26_102: - 5.09765625 frac[26,102] + mem[26,102] = 0
 accuracy_33_97: - 6 frac[33,97] + mem[33,97] = 0
 accuracy_38_14: - 6 frac[38,14] + mem[38,14] = 0
 accuracy_15_56: - 5.09765625 frac[15,56] + mem[15,56] = 0
 accuracy_8_77: - 5.09765625 frac[8,77] + mem[8,77] = 0
 accuracy_17_102: - 5.09765625 frac[17,102] + mem[17,102] = 0
 accuracy_28_16: - 5.09765625 frac[28,16] + mem[28,16] = 0
 accuracy_2_31: - 5.09765625 frac[2,31] + mem[2,31] = 0
 accuracy_37_107: - 6 frac[37,107] + mem[37,107] = 0
 accuracy_32_83: - 6 frac[32,83] + mem[32,83] = 0
 accuracy_33_91: - 6 frac[33,91] + mem[33,91] = 0
 accuracy_35_119: - 6 frac[35,119] + mem[35,119] = 0
 accuracy_18_90: - 5.09765625 frac[18,90] + mem[18,90] = 0
 accuracy_8_83: - 5.09765625 frac[8,83] + mem[8,83] = 0
 accuracy_28_14: - 5.09765625 frac[28,14] + mem[28,14] = 0
 accuracy_34_89: - 6 frac[34,89] + mem[34,89] = 0
 accuracy_35_93: - 6 frac[35,93] + mem[35,93] = 0
 accuracy_32_73: - 6 frac[32,73] + mem[32,73] = 0
 accuracy_37_117: - 6 frac[37,117] + mem[37,117] = 0
 accuracy_13_105: - 5.09765625 frac[13,105] + mem[13,105] = 0
 accuracy_27_35: - 5.09765625 frac[27,35] + mem[27,35] = 0
 accuracy_33_52: - 6 frac[33,52] + mem[33,52] = 0
 accuracy_9_89: - 5.09765625 frac[9,89] + mem[9,89] = 0
 accuracy_30_44: - 5.09765625 frac[30,44] + mem[30,44] = 0
 accuracy_36_71: - 6 frac[36,71] + mem[36,71] = 0
 accuracy_37_95: - 6 frac[37,95] + mem[37,95] = 0
 accuracy_32_47: - 6 frac[32,47] + mem[32,47] = 0
 accuracy_22_22: - 5.09765625 frac[22,22] + mem[22,22] = 0
 accuracy_26_93: - 5.09765625 frac[26,93] + mem[26,93] = 0
 accuracy_23_23: - 5.09765625 frac[23,23] + mem[23,23] = 0
 accuracy_4_77: - 5.09765625 frac[4,77] + mem[4,77] = 0
 accuracy_7_48: - 5.09765625 frac[7,48] + mem[7,48] = 0
 accuracy_22_8: - 5.09765625 frac[22,8] + mem[22,8] = 0
 accuracy_37_32: - 6 frac[37,32] + mem[37,32] = 0
 accuracy_13_93: - 5.09765625 frac[13,93] + mem[13,93] = 0
 accuracy_35_48: - 6 frac[35,48] + mem[35,48] = 0
 accuracy_33_0: - 6 frac[33,0] + mem[33,0] = 0
 accuracy_36_11: - 6 frac[36,11] + mem[36,11] = 0
 accuracy_34_7: - 6 frac[34,7] + mem[34,7] = 0
 accuracy_11_11: - 5.09765625 frac[11,11] + mem[11,11] = 0
 accuracy_35_22: - 6 frac[35,22] + mem[35,22] = 0
 accuracy_24_17: - 5.09765625 frac[24,17] + mem[24,17] = 0
 accuracy_38_1: - 6 frac[38,1] + mem[38,1] = 0
 accuracy_36_17: - 6 frac[36,17] + mem[36,17] = 0
 accuracy_37_20: - 6 frac[37,20] + mem[37,20] = 0
 accuracy_33_92: - 6 frac[33,92] + mem[33,92] = 0
 accuracy_16_109: - 5.09765625 frac[16,109] + mem[16,109] = 0
 accuracy_15_47: - 5.09765625 frac[15,47] + mem[15,47] = 0
 accuracy_36_96: - 6 frac[36,96] + mem[36,96] = 0
 accuracy_28_5: - 5.09765625 frac[28,5] + mem[28,5] = 0
 accuracy_32_64: - 6 frac[32,64] + mem[32,64] = 0
 accuracy_20_35: - 5.09765625 frac[20,35] + mem[20,35] = 0
 accuracy_12_64: - 5.09765625 frac[12,64] + mem[12,64] = 0
 accuracy_9_66: - 5.09765625 frac[9,66] + mem[9,66] = 0
 accuracy_38_106: - 6 frac[38,106] + mem[38,106] = 0
 accuracy_36_78: - 6 frac[36,78] + mem[36,78] = 0
 accuracy_32_38: - 6 frac[32,38] + mem[32,38] = 0
 accuracy_33_39: - 6 frac[33,39] + mem[33,39] = 0
 accuracy_36_84: - 6 frac[36,84] + mem[36,84] = 0
 accuracy_22_3: - 5.09765625 frac[22,3] + mem[22,3] = 0
 accuracy_33_25: - 6 frac[33,25] + mem[33,25] = 0
 accuracy_38_38: - 6 frac[38,38] + mem[38,38] = 0
 accuracy_36_50: - 6 frac[36,50] + mem[36,50] = 0
 accuracy_34_30: - 6 frac[34,30] + mem[34,30] = 0
 accuracy_37_51: - 6 frac[37,51] + mem[37,51] = 0
 accuracy_29_118: - 5.09765625 frac[29,118] + mem[29,118] = 0
 accuracy_35_31: - 6 frac[35,31] + mem[35,31] = 0
 accuracy_38_56: - 6 frac[38,56] + mem[38,56] = 0
 accuracy_36_24: - 6 frac[36,24] + mem[36,24] = 0
 accuracy_31_96: - 5.09765625 frac[31,96] + mem[31,96] = 0
 accuracy_37_29: - 6 frac[37,29] + mem[37,29] = 0
 accuracy_34_113: - 6 frac[34,113] + mem[34,113] = 0
 accuracy_35_101: - 6 frac[35,101] + mem[35,101] = 0
 accuracy_33_85: - 6 frac[33,85] + mem[33,85] = 0
 accuracy_16_100: - 5.09765625 frac[16,100] + mem[16,100] = 0
 accuracy_15_52: - 5.09765625 frac[15,52] + mem[15,52] = 0
 accuracy_8_65: - 5.09765625 frac[8,65] + mem[8,65] = 0
 accuracy_36_111: - 6 frac[36,111] + mem[36,111] = 0
 accuracy_37_103: - 6 frac[37,103] + mem[37,103] = 0
 accuracy_34_107: - 6 frac[34,107] + mem[34,107] = 0
 accuracy_32_71: - 6 frac[32,71] + mem[32,71] = 0
 accuracy_33_79: - 6 frac[33,79] + mem[33,79] = 0
 accuracy_38_109: - 6 frac[38,109] + mem[38,109] = 0
 accuracy_36_117: - 6 frac[36,117] + mem[36,117] = 0
 accuracy_19_23: - 5.09765625 frac[19,23] + mem[19,23] = 0
 accuracy_2_37: - 5.09765625 frac[2,37] + mem[2,37] = 0
 accuracy_32_61: - 6 frac[32,61] + mem[32,61] = 0
 accuracy_13_117: - 5.09765625 frac[13,117] + mem[13,117] = 0
 accuracy_12_85: - 5.09765625 frac[12,85] + mem[12,85] = 0
 accuracy_33_56: - 6 frac[33,56] + mem[33,56] = 0
 accuracy_38_71: - 6 frac[38,71] + mem[38,71] = 0
 accuracy_36_83: - 6 frac[36,83] + mem[36,83] = 0
 accuracy_7_38: - 5.09765625 frac[7,38] + mem[7,38] = 0
 accuracy_23_3: - 5.09765625 frac[23,3] + mem[23,3] = 0
 accuracy_6_57: - 5.09765625 frac[6,57] + mem[6,57] = 0
 accuracy_38_89: - 6 frac[38,89] + mem[38,89] = 0
 accuracy_1_73: - 5.09765625 frac[1,73] + mem[1,73] = 0
 accuracy_34_25: - 6 frac[34,25] + mem[34,25] = 0
 accuracy_37_60: - 6 frac[37,60] + mem[37,60] = 0
 accuracy_35_4: - 6 frac[35,4] + mem[35,4] = 0
 accuracy_26_115: - 5.09765625 frac[26,115] + mem[26,115] = 0
 accuracy_5_67: - 5.09765625 frac[5,67] + mem[5,67] = 0
 accuracy_38_51: - 6 frac[38,51] + mem[38,51] = 0
 accuracy_36_7: - 6 frac[36,7] + mem[36,7] = 0
 accuracy_37_6: - 6 frac[37,6] + mem[37,6] = 0
 accuracy_35_106: - 6 frac[35,106] + mem[35,106] = 0
 accuracy_0_24: - 5.09765625 frac[0,24] + mem[0,24] = 0
 accuracy_33_110: - 6 frac[33,110] + mem[33,110] = 0
 accuracy_37_96: - 6 frac[37,96] + mem[37,96] = 0
 accuracy_35_112: - 6 frac[35,112] + mem[35,112] = 0
 accuracy_33_64: - 6 frac[33,64] + mem[33,64] = 0
 accuracy_20_49: - 5.09765625 frac[20,49] + mem[20,49] = 0
 accuracy_9_76: - 5.09765625 frac[9,76] + mem[9,76] = 0
 accuracy_38_100: - 6 frac[38,100] + mem[38,100] = 0
 accuracy_34_92: - 6 frac[34,92] + mem[34,92] = 0
 accuracy_13_110: - 5.09765625 frac[13,110] + mem[13,110] = 0
 accuracy_3_33: - 5.09765625 frac[3,33] + mem[3,33] = 0
 accuracy_12_76: - 5.09765625 frac[12,76] + mem[12,76] = 0
 accuracy_36_90: - 6 frac[36,90] + mem[36,90] = 0
 accuracy_37_84: - 6 frac[37,84] + mem[37,84] = 0
 accuracy_8_61: - 5.09765625 frac[8,61] + mem[8,61] = 0
 accuracy_7_31: - 5.09765625 frac[7,31] + mem[7,31] = 0
 accuracy_22_29: - 5.09765625 frac[22,29] + mem[22,29] = 0
 accuracy_35_39: - 6 frac[35,39] + mem[35,39] = 0
 accuracy_25_118: - 5.09765625 frac[25,118] + mem[25,118] = 0
 accuracy_6_48: - 5.09765625 frac[6,48] + mem[6,48] = 0
 accuracy_36_32: - 6 frac[36,32] + mem[36,32] = 0
 accuracy_1_66: - 5.09765625 frac[1,66] + mem[1,66] = 0
 accuracy_34_16: - 6 frac[34,16] + mem[34,16] = 0
 accuracy_4_64: - 5.09765625 frac[4,64] + mem[4,64] = 0
 accuracy_32_0: - 6 frac[32,0] + mem[32,0] = 0
 accuracy_37_37: - 6 frac[37,37] + mem[37,37] = 0
 accuracy_33_13: - 6 frac[33,13] + mem[33,13] = 0
 accuracy_36_14: - 6 frac[36,14] + mem[36,14] = 0
 accuracy_11_20: - 5.09765625 frac[11,20] + mem[11,20] = 0
 accuracy_35_19: - 6 frac[35,19] + mem[35,19] = 0
 accuracy_26_100: - 5.09765625 frac[26,100] + mem[26,100] = 0
 accuracy_24_12: - 5.09765625 frac[24,12] + mem[24,12] = 0
 accuracy_29_98: - 5.09765625 frac[29,98] + mem[29,98] = 0
 accuracy_36_20: - 6 frac[36,20] + mem[36,20] = 0
 accuracy_37_105: - 6 frac[37,105] + mem[37,105] = 0
 accuracy_32_85: - 6 frac[32,85] + mem[32,85] = 0
 accuracy_33_89: - 6 frac[33,89] + mem[33,89] = 0
 accuracy_16_104: - 5.09765625 frac[16,104] + mem[16,104] = 0
 accuracy_21_61: - 5.09765625 frac[21,61] + mem[21,61] = 0
 accuracy_28_8: - 5.09765625 frac[28,8] + mem[28,8] = 0
 accuracy_34_87: - 6 frac[34,87] + mem[34,87] = 0
 accuracy_37_115: - 6 frac[37,115] + mem[37,115] = 0
 accuracy_32_75: - 6 frac[32,75] + mem[32,75] = 0
 accuracy_35_95: - 6 frac[35,95] + mem[35,95] = 0
 accuracy_20_38: - 5.09765625 frac[20,38] + mem[20,38] = 0
 accuracy_12_75: - 5.09765625 frac[12,75] + mem[12,75] = 0
 accuracy_18_114: - 5.09765625 frac[18,114] + mem[18,114] = 0
 accuracy_36_65: - 6 frac[36,65] + mem[36,65] = 0
 accuracy_19_11: - 5.09765625 frac[19,11] + mem[19,11] = 0
 accuracy_34_49: - 6 frac[34,49] + mem[34,49] = 0
 accuracy_37_93: - 6 frac[37,93] + mem[37,93] = 0
 accuracy_32_33: - 6 frac[32,33] + mem[32,33] = 0
 accuracy_22_20: - 5.09765625 frac[22,20] + mem[22,20] = 0
 accuracy_35_44: - 6 frac[35,44] + mem[35,44] = 0
 accuracy_30_52: - 5.09765625 frac[30,52] + mem[30,52] = 0
 accuracy_36_47: - 6 frac[36,47] + mem[36,47] = 0
 accuracy_7_50: - 5.09765625 frac[7,50] + mem[7,50] = 0
 accuracy_22_14: - 5.09765625 frac[22,14] + mem[22,14] = 0
 accuracy_37_46: - 6 frac[37,46] + mem[37,46] = 0
 accuracy_13_91: - 5.09765625 frac[13,91] + mem[13,91] = 0
 accuracy_35_50: - 6 frac[35,50] + mem[35,50] = 0
 accuracy_25_97: - 5.09765625 frac[25,97] + mem[25,97] = 0
 accuracy_10_0: - 5.09765625 frac[10,0] + mem[10,0] = 0
 accuracy_34_5: - 6 frac[34,5] + mem[34,5] = 0
 accuracy_37_8: - 6 frac[37,8] + mem[37,8] = 0
 accuracy_11_13: - 5.09765625 frac[11,13] + mem[11,13] = 0
 accuracy_35_24: - 6 frac[35,24] + mem[35,24] = 0
 accuracy_0_22: - 5.09765625 frac[0,22] + mem[0,22] = 0
 accuracy_24_19: - 5.09765625 frac[24,19] + mem[24,19] = 0
 accuracy_38_7: - 6 frac[38,7] + mem[38,7] = 0
 accuracy_36_19: - 6 frac[36,19] + mem[36,19] = 0
 accuracy_34_116: - 6 frac[34,116] + mem[34,116] = 0
 accuracy_33_82: - 6 frac[33,82] + mem[33,82] = 0
 accuracy_38_25: - 6 frac[38,25] + mem[38,25] = 0
 accuracy_15_41: - 5.09765625 frac[15,41] + mem[15,41] = 0
 accuracy_21_70: - 5.09765625 frac[21,70] + mem[21,70] = 0
 accuracy_28_7: - 5.09765625 frac[28,7] + mem[28,7] = 0
 accuracy_34_110: - 6 frac[34,110] + mem[34,110] = 0
 accuracy_32_66: - 6 frac[32,66] + mem[32,66] = 0
 accuracy_13_96: - 5.09765625 frac[13,96] + mem[13,96] = 0
 accuracy_38_104: - 6 frac[38,104] + mem[38,104] = 0
 accuracy_36_72: - 6 frac[36,72] + mem[36,72] = 0
 accuracy_19_16: - 5.09765625 frac[19,16] + mem[19,16] = 0
 accuracy_2_40: - 5.09765625 frac[2,40] + mem[2,40] = 0
 accuracy_34_72: - 6 frac[34,72] + mem[34,72] = 0
 accuracy_37_70: - 6 frac[37,70] + mem[37,70] = 0
 accuracy_3_53: - 5.09765625 frac[3,53] + mem[3,53] = 0
 accuracy_33_37: - 6 frac[33,37] + mem[33,37] = 0
 accuracy_6_34: - 5.09765625 frac[6,34] + mem[6,34] = 0
 accuracy_38_66: - 6 frac[38,66] + mem[38,66] = 0
 accuracy_34_34: - 6 frac[34,34] + mem[34,34] = 0
 accuracy_33_31: - 6 frac[33,31] + mem[33,31] = 0
 accuracy_23_4: - 5.09765625 frac[23,4] + mem[23,4] = 0
 accuracy_36_60: - 6 frac[36,60] + mem[36,60] = 0
 accuracy_10_11: - 5.09765625 frac[10,11] + mem[10,11] = 0
 accuracy_37_49: - 6 frac[37,49] + mem[37,49] = 0
 accuracy_35_1: - 6 frac[35,1] + mem[35,1] = 0
 accuracy_38_62: - 6 frac[38,62] + mem[38,62] = 0
 accuracy_31_98: - 5.09765625 frac[31,98] + mem[31,98] = 0
 accuracy_29_110: - 5.09765625 frac[29,110] + mem[29,110] = 0
 accuracy_35_103: - 6 frac[35,103] + mem[35,103] = 0
 accuracy_0_3: - 5.09765625 frac[0,3] + mem[0,3] = 0
 accuracy_38_16: - 6 frac[38,16] + mem[38,16] = 0
 accuracy_16_102: - 5.09765625 frac[16,102] + mem[16,102] = 0
 accuracy_21_79: - 5.09765625 frac[21,79] + mem[21,79] = 0
 accuracy_36_105: - 6 frac[36,105] + mem[36,105] = 0
 accuracy_37_101: - 6 frac[37,101] + mem[37,101] = 0
 accuracy_34_105: - 6 frac[34,105] + mem[34,105] = 0
 accuracy_32_89: - 6 frac[32,89] + mem[32,89] = 0
 accuracy_33_77: - 6 frac[33,77] + mem[33,77] = 0
 accuracy_14_89: - 5.09765625 frac[14,89] + mem[14,89] = 0
 accuracy_36_119: - 6 frac[36,119] + mem[36,119] = 0
 accuracy_37_79: - 6 frac[37,79] + mem[37,79] = 0
 accuracy_2_35: - 5.09765625 frac[2,35] + mem[2,35] = 0
 accuracy_32_63: - 6 frac[32,63] + mem[32,63] = 0
 accuracy_14_67: - 5.09765625 frac[14,67] + mem[14,67] = 0
 accuracy_12_87: - 5.09765625 frac[12,87] + mem[12,87] = 0
 accuracy_33_62: - 6 frac[33,62] + mem[33,62] = 0
 accuracy_30_38: - 5.09765625 frac[30,38] + mem[30,38] = 0
 accuracy_4_61: - 5.09765625 frac[4,61] + mem[4,61] = 0
 accuracy_36_93: - 6 frac[36,93] + mem[36,93] = 0
 accuracy_34_61: - 6 frac[34,61] + mem[34,61] = 0
 accuracy_35_32: - 6 frac[35,32] + mem[35,32] = 0
 accuracy_33_16: - 6 frac[33,16] + mem[33,16] = 0
 accuracy_23_29: - 5.09765625 frac[23,29] + mem[23,29] = 0
 accuracy_38_95: - 6 frac[38,95] + mem[38,95] = 0
 accuracy_36_59: - 6 frac[36,59] + mem[36,59] = 0
 accuracy_34_23: - 6 frac[34,23] + mem[34,23] = 0
 accuracy_37_58: - 6 frac[37,58] + mem[37,58] = 0
 accuracy_35_6: - 6 frac[35,6] + mem[35,6] = 0
 accuracy_5_65: - 5.09765625 frac[5,65] + mem[5,65] = 0
 accuracy_38_49: - 6 frac[38,49] + mem[38,49] = 0
 accuracy_12_60: - 5.09765625 frac[12,60] + mem[12,60] = 0
 accuracy_36_1: - 6 frac[36,1] + mem[36,1] = 0
 accuracy_10_28: - 5.09765625 frac[10,28] + mem[10,28] = 0
 accuracy_31_107: - 5.09765625 frac[31,107] + mem[31,107] = 0
 accuracy_37_4: - 6 frac[37,4] + mem[37,4] = 0
 accuracy_11_17: - 5.09765625 frac[11,17] + mem[11,17] = 0
 accuracy_29_103: - 5.09765625 frac[29,103] + mem[29,103] = 0
 accuracy_24_7: - 5.09765625 frac[24,7] + mem[24,7] = 0
 accuracy_38_11: - 6 frac[38,11] + mem[38,11] = 0
 accuracy_8_74: - 5.09765625 frac[8,74] + mem[8,74] = 0
 accuracy_21_72: - 5.09765625 frac[21,72] + mem[21,72] = 0
 accuracy_31_113: - 5.09765625 frac[31,113] + mem[31,113] = 0
 accuracy_37_110: - 6 frac[37,110] + mem[37,110] = 0
 accuracy_34_96: - 6 frac[34,96] + mem[34,96] = 0
 accuracy_35_114: - 6 frac[35,114] + mem[35,114] = 0
 accuracy_33_70: - 6 frac[33,70] + mem[33,70] = 0
 accuracy_15_37: - 5.09765625 frac[15,37] + mem[15,37] = 0
 accuracy_37_72: - 6 frac[37,72] + mem[37,72] = 0
 accuracy_34_90: - 6 frac[34,90] + mem[34,90] = 0
 mem_tot_CPU1: - mem[0,17] - mem[0,0] - mem[0,25] - mem[0,16] - mem[0,13]
   - mem[0,27] - mem[0,8] - mem[0,6] - mem[0,24] - mem[0,22] - mem[0,3]
   + mem_tot_CPU1 = 0
 R1107: - 3.05176e-05 mem_tot_CPU1 + normalized_mem_tot_CPU1 = 0
 rows_tot_CPU1: - 3 frac[0,17] - 3 frac[0,0] - 3 frac[0,25] - 3 frac[0,16]
   - 3 frac[0,13] - 3 frac[0,27] - 3 frac[0,8] - 3 frac[0,6] - 3 frac[0,24]
   - 3 frac[0,22] - 3 frac[0,3] + rows_tot_CPU1 = 0
 R1109: - 0.0833333333333333 rows_tot_CPU1 + normalized_rows_tot_CPU1 = 0
 mem_tot_CPU2: - mem[1,64] - mem[1,89] - mem[1,77] - mem[1,62] - mem[1,83]
   - mem[1,60] - mem[1,88] - mem[1,73] - mem[1,66] + mem_tot_CPU2 = 0
 R1111: - 3.05176e-05 mem_tot_CPU2 + normalized_mem_tot_CPU2 = 0
 rows_tot_CPU2: - 3 frac[1,64] - 3 frac[1,89] - 3 frac[1,77] - 3 frac[1,62]
   - 3 frac[1,83] - 3 frac[1,60] - 3 frac[1,88] - 3 frac[1,73]
   - 3 frac[1,66] + rows_tot_CPU2 = 0
 R1113: - 0.0833333333333333 rows_tot_CPU2 + normalized_rows_tot_CPU2 = 0
 mem_tot_CPU3: - mem[2,53] - mem[2,38] - mem[2,36] - mem[2,43] - mem[2,34]
   - mem[2,32] - mem[2,50] - mem[2,39] - mem[2,31] - mem[2,37] - mem[2,40]
   - mem[2,35] + mem_tot_CPU3 = 0
 R1115: - 3.05176e-05 mem_tot_CPU3 + normalized_mem_tot_CPU3 = 0
 rows_tot_CPU3: - 3 frac[2,53] - 3 frac[2,38] - 3 frac[2,36] - 3 frac[2,43]
   - 3 frac[2,34] - 3 frac[2,32] - 3 frac[2,50] - 3 frac[2,39]
   - 3 frac[2,31] - 3 frac[2,37] - 3 frac[2,40] - 3 frac[2,35]
   + rows_tot_CPU3 = 0
 R1117: - 0.0833333333333333 rows_tot_CPU3 + normalized_rows_tot_CPU3 = 0
 mem_tot_CPU4: - mem[3,55] - mem[3,48] - mem[3,57] - mem[3,39] - mem[3,44]
   - mem[3,50] - mem[3,59] - mem[3,41] - mem[3,46] - mem[3,49] - mem[3,33]
   - mem[3,53] + mem_tot_CPU4 = 0
 R1119: - 3.05176e-05 mem_tot_CPU4 + normalized_mem_tot_CPU4 = 0
 rows_tot_CPU4: - 3 frac[3,55] - 3 frac[3,48] - 3 frac[3,57] - 3 frac[3,39]
   - 3 frac[3,44] - 3 frac[3,50] - 3 frac[3,59] - 3 frac[3,41]
   - 3 frac[3,46] - 3 frac[3,49] - 3 frac[3,33] - 3 frac[3,53]
   + rows_tot_CPU4 = 0
 R1121: - 0.0833333333333333 rows_tot_CPU4 + normalized_rows_tot_CPU4 = 0
 mem_tot_CPU5: - mem[4,87] - mem[4,63] - mem[4,75] - mem[4,71] - mem[4,78]
   - mem[4,60] - mem[4,72] - mem[4,62] - mem[4,67] - mem[4,77] - mem[4,64]
   - mem[4,61] + mem_tot_CPU5 = 0
 R1123: - 3.05176e-05 mem_tot_CPU5 + normalized_mem_tot_CPU5 = 0
 rows_tot_CPU5: - 3 frac[4,87] - 3 frac[4,63] - 3 frac[4,75] - 3 frac[4,71]
   - 3 frac[4,78] - 3 frac[4,60] - 3 frac[4,72] - 3 frac[4,62]
   - 3 frac[4,67] - 3 frac[4,77] - 3 frac[4,64] - 3 frac[4,61]
   + rows_tot_CPU5 = 0
 R1125: - 0.0833333333333333 rows_tot_CPU5 + normalized_rows_tot_CPU5 = 0
 mem_tot_CPU6: - mem[5,74] - mem[5,70] - mem[5,72] - mem[5,68] - mem[5,60]
   - mem[5,75] - mem[5,85] - mem[5,71] - mem[5,69] - mem[5,61] - mem[5,67]
   - mem[5,65] + mem_tot_CPU6 = 0
 R1127: - 3.05176e-05 mem_tot_CPU6 + normalized_mem_tot_CPU6 = 0
 rows_tot_CPU6: - 3 frac[5,74] - 3 frac[5,70] - 3 frac[5,72] - 3 frac[5,68]
   - 3 frac[5,60] - 3 frac[5,75] - 3 frac[5,85] - 3 frac[5,71]
   - 3 frac[5,69] - 3 frac[5,61] - 3 frac[5,67] - 3 frac[5,65]
   + rows_tot_CPU6 = 0
 R1129: - 0.0833333333333333 rows_tot_CPU6 + normalized_rows_tot_CPU6 = 0
 mem_tot_CPU7: - mem[6,41] - mem[6,58] - mem[6,38] - mem[6,51] - mem[6,36]
   - mem[6,31] - mem[6,55] - mem[6,46] - mem[6,50] - mem[6,57] - mem[6,48]
   - mem[6,34] + mem_tot_CPU7 = 0
 R1131: - 3.05176e-05 mem_tot_CPU7 + normalized_mem_tot_CPU7 = 0
 rows_tot_CPU7: - 3 frac[6,41] - 3 frac[6,58] - 3 frac[6,38] - 3 frac[6,51]
   - 3 frac[6,36] - 3 frac[6,31] - 3 frac[6,55] - 3 frac[6,46]
   - 3 frac[6,50] - 3 frac[6,57] - 3 frac[6,48] - 3 frac[6,34]
   + rows_tot_CPU7 = 0
 R1133: - 0.0833333333333333 rows_tot_CPU7 + normalized_rows_tot_CPU7 = 0
 mem_tot_CPU8: - mem[7,44] - mem[7,37] - mem[7,49] - mem[7,39] - mem[7,40]
   - mem[7,33] - mem[7,35] - mem[7,47] - mem[7,48] - mem[7,38] - mem[7,31]
   - mem[7,50] + mem_tot_CPU8 = 0
 R1135: - 3.05176e-05 mem_tot_CPU8 + normalized_mem_tot_CPU8 = 0
 rows_tot_CPU8: - 3 frac[7,44] - 3 frac[7,37] - 3 frac[7,49] - 3 frac[7,39]
   - 3 frac[7,40] - 3 frac[7,33] - 3 frac[7,35] - 3 frac[7,47]
   - 3 frac[7,48] - 3 frac[7,38] - 3 frac[7,31] - 3 frac[7,50]
   + rows_tot_CPU8 = 0
 R1137: - 0.0833333333333333 rows_tot_CPU8 + normalized_rows_tot_CPU8 = 0
 mem_tot_CPU9: - mem[8,89] - mem[8,64] - mem[8,78] - mem[8,84] - mem[8,73]
   - mem[8,62] - mem[8,75] - mem[8,77] - mem[8,83] - mem[8,65] - mem[8,61]
   - mem[8,74] + mem_tot_CPU9 = 0
 R1139: - 3.05176e-05 mem_tot_CPU9 + normalized_mem_tot_CPU9 = 0
 rows_tot_CPU9: - 3 frac[8,89] - 3 frac[8,64] - 3 frac[8,78] - 3 frac[8,84]
   - 3 frac[8,73] - 3 frac[8,62] - 3 frac[8,75] - 3 frac[8,77]
   - 3 frac[8,83] - 3 frac[8,65] - 3 frac[8,61] - 3 frac[8,74]
   + rows_tot_CPU9 = 0
 R1141: - 0.0833333333333333 rows_tot_CPU9 + normalized_rows_tot_CPU9 = 0
 mem_tot_CPU10: - mem[9,62] - mem[9,70] - mem[9,79] - mem[9,67] - mem[9,77]
   - mem[9,87] - mem[9,85] - mem[9,71] - mem[9,82] - mem[9,80] - mem[9,89]
   - mem[9,66] - mem[9,76] + mem_tot_CPU10 = 0
 R1143: - 3.05176e-05 mem_tot_CPU10 + normalized_mem_tot_CPU10 = 0
 rows_tot_CPU10: - 3 frac[9,62] - 3 frac[9,70] - 3 frac[9,79]
   - 3 frac[9,67] - 3 frac[9,77] - 3 frac[9,87] - 3 frac[9,85]
   - 3 frac[9,71] - 3 frac[9,82] - 3 frac[9,80] - 3 frac[9,89]
   - 3 frac[9,66] - 3 frac[9,76] + rows_tot_CPU10 = 0
 R1145: - 0.0833333333333333 rows_tot_CPU10 + normalized_rows_tot_CPU10
   = 0
 mem_tot_CPU11: - mem[10,7] - mem[10,19] - mem[10,17] - mem[10,24]
   - mem[10,3] - mem[10,1] - mem[10,29] - mem[10,27] - mem[10,0]
   - mem[10,11] - mem[10,28] + mem_tot_CPU11 = 0
 R1147: - 3.05176e-05 mem_tot_CPU11 + normalized_mem_tot_CPU11 = 0
 rows_tot_CPU11: - 3 frac[10,7] - 3 frac[10,19] - 3 frac[10,17]
   - 3 frac[10,24] - 3 frac[10,3] - 3 frac[10,1] - 3 frac[10,29]
   - 3 frac[10,27] - 3 frac[10,0] - 3 frac[10,11] - 3 frac[10,28]
   + rows_tot_CPU11 = 0
 R1149: - 0.0833333333333333 rows_tot_CPU11 + normalized_rows_tot_CPU11
   = 0
 mem_tot_CPU12: - mem[11,22] - mem[11,8] - mem[11,28] - mem[11,3]
   - mem[11,25] - mem[11,16] - mem[11,14] - mem[11,11] - mem[11,20]
   - mem[11,13] - mem[11,17] + mem_tot_CPU12 = 0
 R1151: - 3.05176e-05 mem_tot_CPU12 + normalized_mem_tot_CPU12 = 0
 rows_tot_CPU12: - 3 frac[11,22] - 3 frac[11,8] - 3 frac[11,28]
   - 3 frac[11,3] - 3 frac[11,25] - 3 frac[11,16] - 3 frac[11,14]
   - 3 frac[11,11] - 3 frac[11,20] - 3 frac[11,13] - 3 frac[11,17]
   + rows_tot_CPU12 = 0
 R1153: - 0.0833333333333333 rows_tot_CPU12 + normalized_rows_tot_CPU12
   = 0
 mem_tot_CPU13: - mem[12,71] - mem[12,84] - mem[12,74] - mem[12,77]
   - mem[12,68] - mem[12,67] - mem[12,89] - mem[12,80] - mem[12,70]
   - mem[12,82] - mem[12,63] - mem[12,64] - mem[12,85] - mem[12,76]
   - mem[12,75] - mem[12,87] - mem[12,60] + mem_tot_CPU13 = 0
 R1155: - 3.05176e-05 mem_tot_CPU13 + normalized_mem_tot_CPU13 = 0
 rows_tot_CPU13: - 3 frac[12,71] - 3 frac[12,84] - 3 frac[12,74]
   - 3 frac[12,77] - 3 frac[12,68] - 3 frac[12,67] - 3 frac[12,89]
   - 3 frac[12,80] - 3 frac[12,70] - 3 frac[12,82] - 3 frac[12,63]
   - 3 frac[12,64] - 3 frac[12,85] - 3 frac[12,76] - 3 frac[12,75]
   - 3 frac[12,87] - 3 frac[12,60] + rows_tot_CPU13 = 0
 R1157: - 0.0833333333333333 rows_tot_CPU13 + normalized_rows_tot_CPU13
   = 0
 mem_tot_CPU14: - mem[13,101] - mem[13,106] - mem[13,111] - mem[13,104]
   - mem[13,92] - mem[13,97] - mem[13,109] - mem[13,90] - mem[13,95]
   - mem[13,100] - mem[13,119] - mem[13,105] - mem[13,93] - mem[13,117]
   - mem[13,110] - mem[13,91] - mem[13,96] + mem_tot_CPU14 = 0
 R1159: - 3.05176e-05 mem_tot_CPU14 + normalized_mem_tot_CPU14 = 0
 rows_tot_CPU14: - 3 frac[13,101] - 3 frac[13,106] - 3 frac[13,111]
   - 3 frac[13,104] - 3 frac[13,92] - 3 frac[13,97] - 3 frac[13,109]
   - 3 frac[13,90] - 3 frac[13,95] - 3 frac[13,100] - 3 frac[13,119]
   - 3 frac[13,105] - 3 frac[13,93] - 3 frac[13,117] - 3 frac[13,110]
   - 3 frac[13,91] - 3 frac[13,96] + rows_tot_CPU14 = 0
 R1161: - 0.0833333333333333 rows_tot_CPU14 + normalized_rows_tot_CPU14
   = 0
 mem_tot_CPU15: - mem[14,72] - mem[14,83] - mem[14,88] - mem[14,66]
   - mem[14,69] - mem[14,76] - mem[14,89] - mem[14,67] + mem_tot_CPU15 = 0
 R1163: - 3.05176e-05 mem_tot_CPU15 + normalized_mem_tot_CPU15 = 0
 rows_tot_CPU15: - 3 frac[14,72] - 3 frac[14,83] - 3 frac[14,88]
   - 3 frac[14,66] - 3 frac[14,69] - 3 frac[14,76] - 3 frac[14,89]
   - 3 frac[14,67] + rows_tot_CPU15 = 0
 R1165: - 0.0833333333333333 rows_tot_CPU15 + normalized_rows_tot_CPU15
   = 0
 mem_tot_CPU16: - mem[15,39] - mem[15,50] - mem[15,59] - mem[15,55]
   - mem[15,31] - mem[15,56] - mem[15,47] - mem[15,52] - mem[15,41]
   - mem[15,37] + mem_tot_CPU16 = 0
 R1167: - 3.05176e-05 mem_tot_CPU16 + normalized_mem_tot_CPU16 = 0
 rows_tot_CPU16: - 3 frac[15,39] - 3 frac[15,50] - 3 frac[15,59]
   - 3 frac[15,55] - 3 frac[15,31] - 3 frac[15,56] - 3 frac[15,47]
   - 3 frac[15,52] - 3 frac[15,41] - 3 frac[15,37] + rows_tot_CPU16 = 0
 R1169: - 0.0833333333333333 rows_tot_CPU16 + normalized_rows_tot_CPU16
   = 0
 mem_tot_CPU17: - mem[16,117] - mem[16,110] - mem[16,92] - mem[16,105]
   - mem[16,96] - mem[16,103] - mem[16,116] - mem[16,109] - mem[16,100]
   - mem[16,104] - mem[16,102] + mem_tot_CPU17 = 0
 R1171: - 3.05176e-05 mem_tot_CPU17 + normalized_mem_tot_CPU17 = 0
 rows_tot_CPU17: - 3 frac[16,117] - 3 frac[16,110] - 3 frac[16,92]
   - 3 frac[16,105] - 3 frac[16,96] - 3 frac[16,103] - 3 frac[16,116]
   - 3 frac[16,109] - 3 frac[16,100] - 3 frac[16,104] - 3 frac[16,102]
   + rows_tot_CPU17 = 0
 R1173: - 0.0833333333333333 rows_tot_CPU17 + normalized_rows_tot_CPU17
   = 0
 mem_tot_CPU18: - mem[17,90] - mem[17,107] - mem[17,112] - mem[17,111]
   - mem[17,96] - mem[17,115] - mem[17,116] - mem[17,102] + mem_tot_CPU18
   = 0
 R1175: - 3.05176e-05 mem_tot_CPU18 + normalized_mem_tot_CPU18 = 0
 rows_tot_CPU18: - 3 frac[17,90] - 3 frac[17,107] - 3 frac[17,112]
   - 3 frac[17,111] - 3 frac[17,96] - 3 frac[17,115] - 3 frac[17,116]
   - 3 frac[17,102] + rows_tot_CPU18 = 0
 R1177: - 0.0833333333333333 rows_tot_CPU18 + normalized_rows_tot_CPU18
   = 0
 mem_tot_CPU19: - mem[18,112] - mem[18,107] - mem[18,93] - mem[18,110]
   - mem[18,105] - mem[18,91] - mem[18,103] - mem[18,94] - mem[18,115]
   - mem[18,106] - mem[18,113] - mem[18,104] - mem[18,90] - mem[18,114]
   + mem_tot_CPU19 = 0
 R1179: - 3.05176e-05 mem_tot_CPU19 + normalized_mem_tot_CPU19 = 0
 rows_tot_CPU19: - 3 frac[18,112] - 3 frac[18,107] - 3 frac[18,93]
   - 3 frac[18,110] - 3 frac[18,105] - 3 frac[18,91] - 3 frac[18,103]
   - 3 frac[18,94] - 3 frac[18,115] - 3 frac[18,106] - 3 frac[18,113]
   - 3 frac[18,104] - 3 frac[18,90] - 3 frac[18,114] + rows_tot_CPU19 = 0
 R1181: - 0.0833333333333333 rows_tot_CPU19 + normalized_rows_tot_CPU19
   = 0
 mem_tot_CPU20: - mem[19,13] - mem[19,27] - mem[19,8] - mem[19,17]
   - mem[19,22] - mem[19,19] - mem[19,24] - mem[19,21] - mem[19,23]
   - mem[19,11] - mem[19,16] + mem_tot_CPU20 = 0
 R1183: - 3.05176e-05 mem_tot_CPU20 + normalized_mem_tot_CPU20 = 0
 rows_tot_CPU20: - 3 frac[19,13] - 3 frac[19,27] - 3 frac[19,8]
   - 3 frac[19,17] - 3 frac[19,22] - 3 frac[19,19] - 3 frac[19,24]
   - 3 frac[19,21] - 3 frac[19,23] - 3 frac[19,11] - 3 frac[19,16]
   + rows_tot_CPU20 = 0
 R1185: - 0.0833333333333333 rows_tot_CPU20 + normalized_rows_tot_CPU20
   = 0
 mem_tot_CPU21: - mem[20,34] - mem[20,48] - mem[20,30] - mem[20,37]
   - mem[20,44] - mem[20,50] - mem[20,39] - mem[20,55] - mem[20,35]
   - mem[20,49] - mem[20,38] + mem_tot_CPU21 = 0
 R1187: - 3.05176e-05 mem_tot_CPU21 + normalized_mem_tot_CPU21 = 0
 rows_tot_CPU21: - 3 frac[20,34] - 3 frac[20,48] - 3 frac[20,30]
   - 3 frac[20,37] - 3 frac[20,44] - 3 frac[20,50] - 3 frac[20,39]
   - 3 frac[20,55] - 3 frac[20,35] - 3 frac[20,49] - 3 frac[20,38]
   + rows_tot_CPU21 = 0
 R1189: - 0.0833333333333333 rows_tot_CPU21 + normalized_rows_tot_CPU21
   = 0
 mem_tot_CPU22: - mem[21,77] - mem[21,66] - mem[21,84] - mem[21,62]
   - mem[21,71] - mem[21,64] - mem[21,73] - mem[21,60] - mem[21,87]
   - mem[21,89] - mem[21,61] - mem[21,70] - mem[21,79] - mem[21,72]
   + mem_tot_CPU22 = 0
 R1191: - 3.05176e-05 mem_tot_CPU22 + normalized_mem_tot_CPU22 = 0
 rows_tot_CPU22: - 3 frac[21,77] - 3 frac[21,66] - 3 frac[21,84]
   - 3 frac[21,62] - 3 frac[21,71] - 3 frac[21,64] - 3 frac[21,73]
   - 3 frac[21,60] - 3 frac[21,87] - 3 frac[21,89] - 3 frac[21,61]
   - 3 frac[21,70] - 3 frac[21,79] - 3 frac[21,72] + rows_tot_CPU22 = 0
 R1193: - 0.0833333333333333 rows_tot_CPU22 + normalized_rows_tot_CPU22
   = 0
 mem_tot_CPU23: - mem[22,12] - mem[22,7] - mem[22,17] - mem[22,11]
   - mem[22,23] - mem[22,0] - mem[22,27] - mem[22,6] - mem[22,25]
   - mem[22,13] - mem[22,22] - mem[22,8] - mem[22,3] - mem[22,29]
   - mem[22,20] - mem[22,14] + mem_tot_CPU23 = 0
 R1195: - 3.05176e-05 mem_tot_CPU23 + normalized_mem_tot_CPU23 = 0
 rows_tot_CPU23: - 3 frac[22,12] - 3 frac[22,7] - 3 frac[22,17]
   - 3 frac[22,11] - 3 frac[22,23] - 3 frac[22,0] - 3 frac[22,27]
   - 3 frac[22,6] - 3 frac[22,25] - 3 frac[22,13] - 3 frac[22,22]
   - 3 frac[22,8] - 3 frac[22,3] - 3 frac[22,29] - 3 frac[22,20]
   - 3 frac[22,14] + rows_tot_CPU23 = 0
 R1197: - 0.0833333333333333 rows_tot_CPU23 + normalized_rows_tot_CPU23
   = 0
 mem_tot_CPU24: - mem[23,19] - mem[23,6] - mem[23,11] - mem[23,0]
   - mem[23,22] - mem[23,5] - mem[23,14] - mem[23,7] - mem[23,21]
   - mem[23,8] - mem[23,23] - mem[23,3] - mem[23,4] - mem[23,29]
   + mem_tot_CPU24 = 0
 R1199: - 3.05176e-05 mem_tot_CPU24 + normalized_mem_tot_CPU24 = 0
 rows_tot_CPU24: - 3 frac[23,19] - 3 frac[23,6] - 3 frac[23,11]
   - 3 frac[23,0] - 3 frac[23,22] - 3 frac[23,5] - 3 frac[23,14]
   - 3 frac[23,7] - 3 frac[23,21] - 3 frac[23,8] - 3 frac[23,23]
   - 3 frac[23,3] - 3 frac[23,4] - 3 frac[23,29] + rows_tot_CPU24 = 0
 R1201: - 0.0833333333333333 rows_tot_CPU24 + normalized_rows_tot_CPU24
   = 0
 mem_tot_CPU25: - mem[24,21] - mem[24,9] - mem[24,16] - mem[24,25]
   - mem[24,6] - mem[24,8] - mem[24,22] - mem[24,17] - mem[24,12]
   - mem[24,19] - mem[24,7] + mem_tot_CPU25 = 0
 R1203: - 3.05176e-05 mem_tot_CPU25 + normalized_mem_tot_CPU25 = 0
 rows_tot_CPU25: - 3 frac[24,21] - 3 frac[24,9] - 3 frac[24,16]
   - 3 frac[24,25] - 3 frac[24,6] - 3 frac[24,8] - 3 frac[24,22]
   - 3 frac[24,17] - 3 frac[24,12] - 3 frac[24,19] - 3 frac[24,7]
   + rows_tot_CPU25 = 0
 R1205: - 0.0833333333333333 rows_tot_CPU25 + normalized_rows_tot_CPU25
   = 0
 mem_tot_CPU26: - mem[25,113] - mem[25,110] - mem[25,119] - mem[25,95]
   - mem[25,107] - mem[25,93] - mem[25,105] - mem[25,118] - mem[25,97]
   + mem_tot_CPU26 = 0
 R1207: - 3.05176e-05 mem_tot_CPU26 + normalized_mem_tot_CPU26 = 0
 rows_tot_CPU26: - 3 frac[25,113] - 3 frac[25,110] - 3 frac[25,119]
   - 3 frac[25,95] - 3 frac[25,107] - 3 frac[25,93] - 3 frac[25,105]
   - 3 frac[25,118] - 3 frac[25,97] + rows_tot_CPU26 = 0
 R1209: - 0.0833333333333333 rows_tot_CPU26 + normalized_rows_tot_CPU26
   = 0
 mem_tot_CPU27: - mem[26,109] - mem[26,107] - mem[26,114] - mem[26,103]
   - mem[26,90] - mem[26,110] - mem[26,112] - mem[26,99] - mem[26,117]
   - mem[26,102] - mem[26,93] - mem[26,115] - mem[26,100] + mem_tot_CPU27
   = 0
 R1211: - 3.05176e-05 mem_tot_CPU27 + normalized_mem_tot_CPU27 = 0
 rows_tot_CPU27: - 3 frac[26,109] - 3 frac[26,107] - 3 frac[26,114]
   - 3 frac[26,103] - 3 frac[26,90] - 3 frac[26,110] - 3 frac[26,112]
   - 3 frac[26,99] - 3 frac[26,117] - 3 frac[26,102] - 3 frac[26,93]
   - 3 frac[26,115] - 3 frac[26,100] + rows_tot_CPU27 = 0
 R1213: - 0.0833333333333333 rows_tot_CPU27 + normalized_rows_tot_CPU27
   = 0
 mem_tot_CPU28: - mem[27,39] - mem[27,43] - mem[27,32] - mem[27,55]
   - mem[27,34] - mem[27,48] - mem[27,50] - mem[27,30] - mem[27,35]
   + mem_tot_CPU28 = 0
 R1215: - 3.05176e-05 mem_tot_CPU28 + normalized_mem_tot_CPU28 = 0
 rows_tot_CPU28: - 3 frac[27,39] - 3 frac[27,43] - 3 frac[27,32]
   - 3 frac[27,55] - 3 frac[27,34] - 3 frac[27,48] - 3 frac[27,50]
   - 3 frac[27,30] - 3 frac[27,35] + rows_tot_CPU28 = 0
 R1217: - 0.0833333333333333 rows_tot_CPU28 + normalized_rows_tot_CPU28
   = 0
 mem_tot_CPU29: - mem[28,1] - mem[28,24] - mem[28,4] - mem[28,17]
   - mem[28,9] - mem[28,0] - mem[28,11] - mem[28,25] - mem[28,16]
   - mem[28,14] - mem[28,5] - mem[28,8] - mem[28,7] + mem_tot_CPU29 = 0
 R1219: - 3.05176e-05 mem_tot_CPU29 + normalized_mem_tot_CPU29 = 0
 rows_tot_CPU29: - 3 frac[28,1] - 3 frac[28,24] - 3 frac[28,4]
   - 3 frac[28,17] - 3 frac[28,9] - 3 frac[28,0] - 3 frac[28,11]
   - 3 frac[28,25] - 3 frac[28,16] - 3 frac[28,14] - 3 frac[28,5]
   - 3 frac[28,8] - 3 frac[28,7] + rows_tot_CPU29 = 0
 R1221: - 0.0833333333333333 rows_tot_CPU29 + normalized_rows_tot_CPU29
   = 0
 mem_tot_CPU30: - mem[29,101] - mem[29,93] - mem[29,112] - mem[29,106]
   - mem[29,91] - mem[29,111] - mem[29,104] - mem[29,102] - mem[29,118]
   - mem[29,98] - mem[29,110] - mem[29,103] + mem_tot_CPU30 = 0
 R1223: - 3.05176e-05 mem_tot_CPU30 + normalized_mem_tot_CPU30 = 0
 rows_tot_CPU30: - 3 frac[29,101] - 3 frac[29,93] - 3 frac[29,112]
   - 3 frac[29,106] - 3 frac[29,91] - 3 frac[29,111] - 3 frac[29,104]
   - 3 frac[29,102] - 3 frac[29,118] - 3 frac[29,98] - 3 frac[29,110]
   - 3 frac[29,103] + rows_tot_CPU30 = 0
 R1225: - 0.0833333333333333 rows_tot_CPU30 + normalized_rows_tot_CPU30
   = 0
 mem_tot_CPU31: - mem[30,51] - mem[30,58] - mem[30,49] - mem[30,56]
   - mem[30,55] - mem[30,31] - mem[30,39] - mem[30,46] - mem[30,37]
   - mem[30,44] - mem[30,52] - mem[30,38] + mem_tot_CPU31 = 0
 R1227: - 3.05176e-05 mem_tot_CPU31 + normalized_mem_tot_CPU31 = 0
 rows_tot_CPU31: - 3 frac[30,51] - 3 frac[30,58] - 3 frac[30,49]
   - 3 frac[30,56] - 3 frac[30,55] - 3 frac[30,31] - 3 frac[30,39]
   - 3 frac[30,46] - 3 frac[30,37] - 3 frac[30,44] - 3 frac[30,52]
   - 3 frac[30,38] + rows_tot_CPU31 = 0
 R1229: - 0.0833333333333333 rows_tot_CPU31 + normalized_rows_tot_CPU31
   = 0
 mem_tot_CPU32: - mem[31,103] - mem[31,97] - mem[31,110] - mem[31,99]
   - mem[31,94] - mem[31,100] - mem[31,112] - mem[31,111] - mem[31,117]
   - mem[31,96] - mem[31,98] - mem[31,107] - mem[31,113] + mem_tot_CPU32
   = 0
 R1231: - 3.05176e-05 mem_tot_CPU32 + normalized_mem_tot_CPU32 = 0
 rows_tot_CPU32: - 3 frac[31,103] - 3 frac[31,97] - 3 frac[31,110]
   - 3 frac[31,99] - 3 frac[31,94] - 3 frac[31,100] - 3 frac[31,112]
   - 3 frac[31,111] - 3 frac[31,117] - 3 frac[31,96] - 3 frac[31,98]
   - 3 frac[31,107] - 3 frac[31,113] + rows_tot_CPU32 = 0
 R1233: - 0.0833333333333333 rows_tot_CPU32 + normalized_rows_tot_CPU32
   = 0
 mem_tot_tor_P41: - mem[32,44] - mem[32,87] - mem[32,77] - mem[32,35]
   - mem[32,25] - mem[32,68] - mem[32,58] - mem[32,16] - mem[32,49]
   - mem[32,13] - mem[32,72] - mem[32,46] - mem[32,37] - mem[32,27]
   - mem[32,70] - mem[32,60] - mem[32,8] - mem[32,51] - mem[32,41]
   - mem[32,74] - mem[32,32] - mem[32,6] - mem[32,65] - mem[32,39]
   - mem[32,88] - mem[32,62] - mem[32,53] - mem[32,43] - mem[32,34]
   - mem[32,24] - mem[32,67] - mem[32,57] - mem[32,31] - mem[32,48]
   - mem[32,22] - mem[32,55] - mem[32,3] - mem[32,78] - mem[32,36]
   - mem[32,69] - mem[32,59] - mem[32,17] - mem[32,50] - mem[32,40]
   - mem[32,83] - mem[32,73] - mem[32,47] - mem[32,64] - mem[32,38]
   - mem[32,71] - mem[32,61] - mem[32,0] - mem[32,85] - mem[32,75]
   - mem[32,33] - mem[32,66] - mem[32,89] - mem[32,63] + mem_tot_tor_P41
   = 0
 R1235: - 1.6276041666666666e-04 mem_tot_tor_P41
   + normalized_mem_tot_tor_P41 = 0
 rows_tot_tor_P41: - 3 frac[32,44] - 3 frac[32,87] - 3 frac[32,77]
   - 3 frac[32,35] - 3 frac[32,25] - 3 frac[32,68] - 3 frac[32,58]
   - 3 frac[32,16] - 3 frac[32,49] - 3 frac[32,13] - 3 frac[32,72]
   - 3 frac[32,46] - 3 frac[32,37] - 3 frac[32,27] - 3 frac[32,70]
   - 3 frac[32,60] - 3 frac[32,8] - 3 frac[32,51] - 3 frac[32,41]
   - 3 frac[32,74] - 3 frac[32,32] - 3 frac[32,6] - 3 frac[32,65]
   - 3 frac[32,39] - 3 frac[32,88] - 3 frac[32,62] - 3 frac[32,53]
   - 3 frac[32,43] - 3 frac[32,34] - 3 frac[32,24] - 3 frac[32,67]
   - 3 frac[32,57] - 3 frac[32,31] - 3 frac[32,48] - 3 frac[32,22]
   - 3 frac[32,55] - 3 frac[32,3] - 3 frac[32,78] - 3 frac[32,36]
   - 3 frac[32,69] - 3 frac[32,59] - 3 frac[32,17] - 3 frac[32,50]
   - 3 frac[32,40] - 3 frac[32,83] - 3 frac[32,73] - 3 frac[32,47]
   - 3 frac[32,64] - 3 frac[32,38] - 3 frac[32,71] - 3 frac[32,61]
   - 3 frac[32,0] - 3 frac[32,85] - 3 frac[32,75] - 3 frac[32,33]
   - 3 frac[32,66] - 3 frac[32,89] - 3 frac[32,63] + rows_tot_tor_P41 = 0
 R1237: - 0.0208333333333333 rows_tot_tor_P41 + normalized_rows_tot_tor_P41
   = 0
 mem_tot_tor_P42: - mem[33,55] - mem[33,41] - mem[33,3] - mem[33,101]
   - mem[33,95] - mem[33,80] - mem[33,74] - mem[33,59] - mem[33,29]
   - mem[33,119] - mem[33,105] - mem[33,67] - mem[33,60] - mem[33,22]
   - mem[33,8] - mem[33,68] - mem[33,47] - mem[33,1] - mem[33,93]
   - mem[33,72] - mem[33,19] - mem[33,117] - mem[33,111] - mem[33,65]
   - mem[33,50] - mem[33,20] - mem[33,14] - mem[33,96] - mem[33,90]
   - mem[33,7] - mem[33,83] - mem[33,24] - mem[33,84] - mem[33,78]
   - mem[33,63] - mem[33,17] - mem[33,11] - mem[33,109] - mem[33,71]
   - mem[33,88] - mem[33,75] - mem[33,106] - mem[33,76] - mem[33,61]
   - mem[33,69] - mem[33,100] - mem[33,27] - mem[33,87] - mem[33,73]
   - mem[33,28] - mem[33,104] - mem[33,66] - mem[33,97] - mem[33,91]
   - mem[33,52] - mem[33,0] - mem[33,92] - mem[33,39] - mem[33,25]
   - mem[33,85] - mem[33,79] - mem[33,56] - mem[33,110] - mem[33,64]
   - mem[33,13] - mem[33,89] - mem[33,82] - mem[33,37] - mem[33,31]
   - mem[33,77] - mem[33,62] - mem[33,16] - mem[33,70] + mem_tot_tor_P42
   = 0
 R1239: - 1.6276041666666666e-04 mem_tot_tor_P42
   + normalized_mem_tot_tor_P42 = 0
 rows_tot_tor_P42: - 3 frac[33,55] - 3 frac[33,41] - 3 frac[33,3]
   - 3 frac[33,101] - 3 frac[33,95] - 3 frac[33,80] - 3 frac[33,74]
   - 3 frac[33,59] - 3 frac[33,29] - 3 frac[33,119] - 3 frac[33,105]
   - 3 frac[33,67] - 3 frac[33,60] - 3 frac[33,22] - 3 frac[33,8]
   - 3 frac[33,68] - 3 frac[33,47] - 3 frac[33,1] - 3 frac[33,93]
   - 3 frac[33,72] - 3 frac[33,19] - 3 frac[33,117] - 3 frac[33,111]
   - 3 frac[33,65] - 3 frac[33,50] - 3 frac[33,20] - 3 frac[33,14]
   - 3 frac[33,96] - 3 frac[33,90] - 3 frac[33,7] - 3 frac[33,83]
   - 3 frac[33,24] - 3 frac[33,84] - 3 frac[33,78] - 3 frac[33,63]
   - 3 frac[33,17] - 3 frac[33,11] - 3 frac[33,109] - 3 frac[33,71]
   - 3 frac[33,88] - 3 frac[33,75] - 3 frac[33,106] - 3 frac[33,76]
   - 3 frac[33,61] - 3 frac[33,69] - 3 frac[33,100] - 3 frac[33,27]
   - 3 frac[33,87] - 3 frac[33,73] - 3 frac[33,28] - 3 frac[33,104]
   - 3 frac[33,66] - 3 frac[33,97] - 3 frac[33,91] - 3 frac[33,52]
   - 3 frac[33,0] - 3 frac[33,92] - 3 frac[33,39] - 3 frac[33,25]
   - 3 frac[33,85] - 3 frac[33,79] - 3 frac[33,56] - 3 frac[33,110]
   - 3 frac[33,64] - 3 frac[33,13] - 3 frac[33,89] - 3 frac[33,82]
   - 3 frac[33,37] - 3 frac[33,31] - 3 frac[33,77] - 3 frac[33,62]
   - 3 frac[33,16] - 3 frac[33,70] + rows_tot_tor_P42 = 0
 R1241: - 0.0208333333333333 rows_tot_tor_P42 + normalized_rows_tot_tor_P42
   = 0
 mem_tot_tor_P43: - mem[34,8] - mem[34,79] - mem[34,3] - mem[34,114]
   - mem[34,70] - mem[34,103] - mem[34,21] - mem[34,50] - mem[34,44]
   - mem[34,6] - mem[34,77] - mem[34,39] - mem[34,112] - mem[34,106]
   - mem[34,62] - mem[34,24] - mem[34,19] - mem[34,13] - mem[34,48]
   - mem[34,4] - mem[34,37] - mem[34,104] - mem[34,66] - mem[34,60]
   - mem[34,22] - mem[34,93] - mem[34,55] - mem[34,17] - mem[34,11]
   - mem[34,84] - mem[34,117] - mem[34,111] - mem[34,73] - mem[34,35]
   - mem[34,29] - mem[34,102] - mem[34,64] - mem[34,20] - mem[34,14]
   - mem[34,91] - mem[34,38] - mem[34,0] - mem[34,115] - mem[34,109]
   - mem[34,71] - mem[34,27] - mem[34,100] - mem[34,94] - mem[34,12]
   - mem[34,89] - mem[34,7] - mem[34,30] - mem[34,113] - mem[34,107]
   - mem[34,25] - mem[34,92] - mem[34,16] - mem[34,87] - mem[34,49]
   - mem[34,5] - mem[34,116] - mem[34,110] - mem[34,72] - mem[34,34]
   - mem[34,105] - mem[34,61] - mem[34,23] - mem[34,96] - mem[34,90]
   + mem_tot_tor_P43 = 0
 R1243: - 1.6276041666666666e-04 mem_tot_tor_P43
   + normalized_mem_tot_tor_P43 = 0
 rows_tot_tor_P43: - 3 frac[34,8] - 3 frac[34,79] - 3 frac[34,3]
   - 3 frac[34,114] - 3 frac[34,70] - 3 frac[34,103] - 3 frac[34,21]
   - 3 frac[34,50] - 3 frac[34,44] - 3 frac[34,6] - 3 frac[34,77]
   - 3 frac[34,39] - 3 frac[34,112] - 3 frac[34,106] - 3 frac[34,62]
   - 3 frac[34,24] - 3 frac[34,19] - 3 frac[34,13] - 3 frac[34,48]
   - 3 frac[34,4] - 3 frac[34,37] - 3 frac[34,104] - 3 frac[34,66]
   - 3 frac[34,60] - 3 frac[34,22] - 3 frac[34,93] - 3 frac[34,55]
   - 3 frac[34,17] - 3 frac[34,11] - 3 frac[34,84] - 3 frac[34,117]
   - 3 frac[34,111] - 3 frac[34,73] - 3 frac[34,35] - 3 frac[34,29]
   - 3 frac[34,102] - 3 frac[34,64] - 3 frac[34,20] - 3 frac[34,14]
   - 3 frac[34,91] - 3 frac[34,38] - 3 frac[34,0] - 3 frac[34,115]
   - 3 frac[34,109] - 3 frac[34,71] - 3 frac[34,27] - 3 frac[34,100]
   - 3 frac[34,94] - 3 frac[34,12] - 3 frac[34,89] - 3 frac[34,7]
   - 3 frac[34,30] - 3 frac[34,113] - 3 frac[34,107] - 3 frac[34,25]
   - 3 frac[34,92] - 3 frac[34,16] - 3 frac[34,87] - 3 frac[34,49]
   - 3 frac[34,5] - 3 frac[34,116] - 3 frac[34,110] - 3 frac[34,72]
   - 3 frac[34,34] - 3 frac[34,105] - 3 frac[34,61] - 3 frac[34,23]
   - 3 frac[34,96] - 3 frac[34,90] + rows_tot_tor_P43 = 0
 R1245: - 0.0208333333333333 rows_tot_tor_P43 + normalized_rows_tot_tor_P43
   = 0
 mem_tot_tor_P44: - mem[35,21] - mem[35,46] - mem[35,52] - mem[35,96]
   - mem[35,105] - mem[35,34] - mem[35,8] - mem[35,110] - mem[35,90]
   - mem[35,43] - mem[35,49] - mem[35,98] - mem[35,5] - mem[35,107]
   - mem[35,113] - mem[35,16] - mem[35,118] - mem[35,51] - mem[35,25]
   - mem[35,56] - mem[35,30] - mem[35,100] - mem[35,7] - mem[35,109]
   - mem[35,115] - mem[35,38] - mem[35,12] - mem[35,94] - mem[35,97]
   - mem[35,58] - mem[35,0] - mem[35,102] - mem[35,35] - mem[35,9]
   - mem[35,111] - mem[35,117] - mem[35,91] - mem[35,14] - mem[35,55]
   - mem[35,99] - mem[35,104] - mem[35,37] - mem[35,11] - mem[35,17]
   - mem[35,119] - mem[35,93] - mem[35,48] - mem[35,22] - mem[35,31]
   - mem[35,101] - mem[35,4] - mem[35,106] - mem[35,112] - mem[35,39]
   - mem[35,19] - mem[35,95] - mem[35,44] - mem[35,50] - mem[35,24]
   - mem[35,1] - mem[35,103] - mem[35,32] - mem[35,6] - mem[35,114]
   + mem_tot_tor_P44 = 0
 R1247: - 1.6276041666666666e-04 mem_tot_tor_P44
   + normalized_mem_tot_tor_P44 = 0
 rows_tot_tor_P44: - 3 frac[35,21] - 3 frac[35,46] - 3 frac[35,52]
   - 3 frac[35,96] - 3 frac[35,105] - 3 frac[35,34] - 3 frac[35,8]
   - 3 frac[35,110] - 3 frac[35,90] - 3 frac[35,43] - 3 frac[35,49]
   - 3 frac[35,98] - 3 frac[35,5] - 3 frac[35,107] - 3 frac[35,113]
   - 3 frac[35,16] - 3 frac[35,118] - 3 frac[35,51] - 3 frac[35,25]
   - 3 frac[35,56] - 3 frac[35,30] - 3 frac[35,100] - 3 frac[35,7]
   - 3 frac[35,109] - 3 frac[35,115] - 3 frac[35,38] - 3 frac[35,12]
   - 3 frac[35,94] - 3 frac[35,97] - 3 frac[35,58] - 3 frac[35,0]
   - 3 frac[35,102] - 3 frac[35,35] - 3 frac[35,9] - 3 frac[35,111]
   - 3 frac[35,117] - 3 frac[35,91] - 3 frac[35,14] - 3 frac[35,55]
   - 3 frac[35,99] - 3 frac[35,104] - 3 frac[35,37] - 3 frac[35,11]
   - 3 frac[35,17] - 3 frac[35,119] - 3 frac[35,93] - 3 frac[35,48]
   - 3 frac[35,22] - 3 frac[35,31] - 3 frac[35,101] - 3 frac[35,4]
   - 3 frac[35,106] - 3 frac[35,112] - 3 frac[35,39] - 3 frac[35,19]
   - 3 frac[35,95] - 3 frac[35,44] - 3 frac[35,50] - 3 frac[35,24]
   - 3 frac[35,1] - 3 frac[35,103] - 3 frac[35,32] - 3 frac[35,6]
   - 3 frac[35,114] + rows_tot_tor_P44 = 0
 R1249: - 0.0208333333333333 rows_tot_tor_P44 + normalized_rows_tot_tor_P44
   = 0
 mem_tot_l1_P41: - mem[36,68] - mem[36,8] - mem[36,22] - mem[36,101]
   - mem[36,67] - mem[36,41] - mem[36,55] - mem[36,29] - mem[36,74]
   - mem[36,62] - mem[36,95] - mem[36,37] - mem[36,3] - mem[36,70]
   - mem[36,44] - mem[36,16] - mem[36,77] - mem[36,43] - mem[36,49]
   - mem[36,31] - mem[36,110] - mem[36,56] - mem[36,6] - mem[36,89]
   - mem[36,39] - mem[36,13] - mem[36,64] - mem[36,46] - mem[36,52]
   - mem[36,97] - mem[36,79] - mem[36,85] - mem[36,51] - mem[36,25]
   - mem[36,104] - mem[36,92] - mem[36,58] - mem[36,0] - mem[36,91]
   - mem[36,100] - mem[36,66] - mem[36,73] - mem[36,87] - mem[36,61]
   - mem[36,27] - mem[36,106] - mem[36,69] - mem[36,35] - mem[36,48]
   - mem[36,109] - mem[36,63] - mem[36,88] - mem[36,38] - mem[36,71]
   - mem[36,11] - mem[36,17] - mem[36,96] - mem[36,78] - mem[36,84]
   - mem[36,50] - mem[36,24] - mem[36,111] - mem[36,117] - mem[36,83]
   - mem[36,7] - mem[36,90] - mem[36,32] - mem[36,14] - mem[36,20]
   - mem[36,65] - mem[36,47] - mem[36,19] - mem[36,72] - mem[36,60]
   - mem[36,105] - mem[36,119] - mem[36,93] - mem[36,59] - mem[36,1]
   + mem_tot_l1_P41 = 0
 R1251: - 1.6276041666666666e-04 mem_tot_l1_P41 + normalized_mem_tot_l1_P41
   = 0
 rows_tot_l1_P41: - 3 frac[36,68] - 3 frac[36,8] - 3 frac[36,22]
   - 3 frac[36,101] - 3 frac[36,67] - 3 frac[36,41] - 3 frac[36,55]
   - 3 frac[36,29] - 3 frac[36,74] - 3 frac[36,62] - 3 frac[36,95]
   - 3 frac[36,37] - 3 frac[36,3] - 3 frac[36,70] - 3 frac[36,44]
   - 3 frac[36,16] - 3 frac[36,77] - 3 frac[36,43] - 3 frac[36,49]
   - 3 frac[36,31] - 3 frac[36,110] - 3 frac[36,56] - 3 frac[36,6]
   - 3 frac[36,89] - 3 frac[36,39] - 3 frac[36,13] - 3 frac[36,64]
   - 3 frac[36,46] - 3 frac[36,52] - 3 frac[36,97] - 3 frac[36,79]
   - 3 frac[36,85] - 3 frac[36,51] - 3 frac[36,25] - 3 frac[36,104]
   - 3 frac[36,92] - 3 frac[36,58] - 3 frac[36,0] - 3 frac[36,91]
   - 3 frac[36,100] - 3 frac[36,66] - 3 frac[36,73] - 3 frac[36,87]
   - 3 frac[36,61] - 3 frac[36,27] - 3 frac[36,106] - 3 frac[36,69]
   - 3 frac[36,35] - 3 frac[36,48] - 3 frac[36,109] - 3 frac[36,63]
   - 3 frac[36,88] - 3 frac[36,38] - 3 frac[36,71] - 3 frac[36,11]
   - 3 frac[36,17] - 3 frac[36,96] - 3 frac[36,78] - 3 frac[36,84]
   - 3 frac[36,50] - 3 frac[36,24] - 3 frac[36,111] - 3 frac[36,117]
   - 3 frac[36,83] - 3 frac[36,7] - 3 frac[36,90] - 3 frac[36,32]
   - 3 frac[36,14] - 3 frac[36,20] - 3 frac[36,65] - 3 frac[36,47]
   - 3 frac[36,19] - 3 frac[36,72] - 3 frac[36,60] - 3 frac[36,105]
   - 3 frac[36,119] - 3 frac[36,93] - 3 frac[36,59] - 3 frac[36,1]
   + rows_tot_l1_P41 = 0
 R1253: - 0.0208333333333333 rows_tot_l1_P41 + normalized_rows_tot_l1_P41
   = 0
 mem_tot_l1_P42: - mem[37,35] - mem[37,113] - mem[37,91] - mem[37,44]
   - mem[37,16] - mem[37,25] - mem[37,77] - mem[37,87] - mem[37,56]
   - mem[37,11] - mem[37,21] - mem[37,89] - mem[37,52] - mem[37,30]
   - mem[37,66] - mem[37,61] - mem[37,7] - mem[37,97] - mem[37,38]
   - mem[37,0] - mem[37,106] - mem[37,94] - mem[37,19] - mem[37,71]
   - mem[37,50] - mem[37,64] - mem[37,5] - mem[37,111] - mem[37,73]
   - mem[37,14] - mem[37,104] - mem[37,114] - mem[37,92] - mem[37,55]
   - mem[37,17] - mem[37,48] - mem[37,100] - mem[37,3] - mem[37,109]
   - mem[37,119] - mem[37,34] - mem[37,12] - mem[37,22] - mem[37,112]
   - mem[37,90] - mem[37,43] - mem[37,31] - mem[37,62] - mem[37,24]
   - mem[37,39] - mem[37,1] - mem[37,107] - mem[37,117] - mem[37,95]
   - mem[37,32] - mem[37,20] - mem[37,51] - mem[37,29] - mem[37,103]
   - mem[37,60] - mem[37,6] - mem[37,96] - mem[37,84] - mem[37,37]
   - mem[37,105] - mem[37,115] - mem[37,93] - mem[37,46] - mem[37,8]
   - mem[37,70] - mem[37,49] - mem[37,101] - mem[37,79] - mem[37,58]
   - mem[37,4] - mem[37,110] - mem[37,72] + mem_tot_l1_P42 = 0
 R1255: - 1.6276041666666666e-04 mem_tot_l1_P42 + normalized_mem_tot_l1_P42
   = 0
 rows_tot_l1_P42: - 3 frac[37,35] - 3 frac[37,113] - 3 frac[37,91]
   - 3 frac[37,44] - 3 frac[37,16] - 3 frac[37,25] - 3 frac[37,77]
   - 3 frac[37,87] - 3 frac[37,56] - 3 frac[37,11] - 3 frac[37,21]
   - 3 frac[37,89] - 3 frac[37,52] - 3 frac[37,30] - 3 frac[37,66]
   - 3 frac[37,61] - 3 frac[37,7] - 3 frac[37,97] - 3 frac[37,38]
   - 3 frac[37,0] - 3 frac[37,106] - 3 frac[37,94] - 3 frac[37,19]
   - 3 frac[37,71] - 3 frac[37,50] - 3 frac[37,64] - 3 frac[37,5]
   - 3 frac[37,111] - 3 frac[37,73] - 3 frac[37,14] - 3 frac[37,104]
   - 3 frac[37,114] - 3 frac[37,92] - 3 frac[37,55] - 3 frac[37,17]
   - 3 frac[37,48] - 3 frac[37,100] - 3 frac[37,3] - 3 frac[37,109]
   - 3 frac[37,119] - 3 frac[37,34] - 3 frac[37,12] - 3 frac[37,22]
   - 3 frac[37,112] - 3 frac[37,90] - 3 frac[37,43] - 3 frac[37,31]
   - 3 frac[37,62] - 3 frac[37,24] - 3 frac[37,39] - 3 frac[37,1]
   - 3 frac[37,107] - 3 frac[37,117] - 3 frac[37,95] - 3 frac[37,32]
   - 3 frac[37,20] - 3 frac[37,51] - 3 frac[37,29] - 3 frac[37,103]
   - 3 frac[37,60] - 3 frac[37,6] - 3 frac[37,96] - 3 frac[37,84]
   - 3 frac[37,37] - 3 frac[37,105] - 3 frac[37,115] - 3 frac[37,93]
   - 3 frac[37,46] - 3 frac[37,8] - 3 frac[37,70] - 3 frac[37,49]
   - 3 frac[37,101] - 3 frac[37,79] - 3 frac[37,58] - 3 frac[37,4]
   - 3 frac[37,110] - 3 frac[37,72] + rows_tot_l1_P42 = 0
 R1257: - 0.0208333333333333 rows_tot_l1_P42 + normalized_rows_tot_l1_P42
   = 0
 mem_tot_l2_P4: - mem[38,119] - mem[38,73] - mem[38,35] - mem[38,31]
   - mem[38,110] - mem[38,64] - mem[38,90] - mem[38,60] - mem[38,22]
   - mem[38,97] - mem[38,93] - mem[38,55] - mem[38,84] - mem[38,46]
   - mem[38,0] - mem[38,117] - mem[38,79] - mem[38,29] - mem[38,70]
   - mem[38,50] - mem[38,20] - mem[38,44] - mem[38,6] - mem[38,24]
   - mem[38,77] - mem[38,39] - mem[38,19] - mem[38,48] - mem[38,101]
   - mem[38,43] - mem[38,72] - mem[38,105] - mem[38,37] - mem[38,17]
   - mem[38,96] - mem[38,92] - mem[38,8] - mem[38,87] - mem[38,3]
   - mem[38,32] - mem[38,58] - mem[38,111] - mem[38,91] - mem[38,61]
   - mem[38,52] - mem[38,14] - mem[38,1] - mem[38,106] - mem[38,38]
   - mem[38,56] - mem[38,109] - mem[38,71] - mem[38,89] - mem[38,51]
   - mem[38,100] - mem[38,7] - mem[38,25] - mem[38,104] - mem[38,66]
   - mem[38,62] - mem[38,16] - mem[38,95] - mem[38,49] - mem[38,11]
   + mem_tot_l2_P4 = 0
 R1259: - 1.6276041666666666e-04 mem_tot_l2_P4 + normalized_mem_tot_l2_P4
   = 0
 rows_tot_l2_P4: - 3 frac[38,119] - 3 frac[38,73] - 3 frac[38,35]
   - 3 frac[38,31] - 3 frac[38,110] - 3 frac[38,64] - 3 frac[38,90]
   - 3 frac[38,60] - 3 frac[38,22] - 3 frac[38,97] - 3 frac[38,93]
   - 3 frac[38,55] - 3 frac[38,84] - 3 frac[38,46] - 3 frac[38,0]
   - 3 frac[38,117] - 3 frac[38,79] - 3 frac[38,29] - 3 frac[38,70]
   - 3 frac[38,50] - 3 frac[38,20] - 3 frac[38,44] - 3 frac[38,6]
   - 3 frac[38,24] - 3 frac[38,77] - 3 frac[38,39] - 3 frac[38,19]
   - 3 frac[38,48] - 3 frac[38,101] - 3 frac[38,43] - 3 frac[38,72]
   - 3 frac[38,105] - 3 frac[38,37] - 3 frac[38,17] - 3 frac[38,96]
   - 3 frac[38,92] - 3 frac[38,8] - 3 frac[38,87] - 3 frac[38,3]
   - 3 frac[38,32] - 3 frac[38,58] - 3 frac[38,111] - 3 frac[38,91]
   - 3 frac[38,61] - 3 frac[38,52] - 3 frac[38,14] - 3 frac[38,1]
   - 3 frac[38,106] - 3 frac[38,38] - 3 frac[38,56] - 3 frac[38,109]
   - 3 frac[38,71] - 3 frac[38,89] - 3 frac[38,51] - 3 frac[38,100]
   - 3 frac[38,7] - 3 frac[38,25] - 3 frac[38,104] - 3 frac[38,66]
   - 3 frac[38,62] - 3 frac[38,16] - 3 frac[38,95] - 3 frac[38,49]
   - 3 frac[38,11] + rows_tot_l2_P4 = 0
 R1261: - 0.0208333333333333 rows_tot_l2_P4 + normalized_rows_tot_l2_P4
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
 capacity_mem_par_tor_P41: mem[32,44] <= 1536
 capacity_mem_par_tor_P41: mem[32,87] <= 1536
 capacity_mem_par_tor_P41: mem[32,77] <= 1536
 capacity_mem_par_tor_P41: mem[32,35] <= 1536
 capacity_mem_par_tor_P41: mem[32,25] <= 1536
 capacity_mem_par_tor_P41: mem[32,68] <= 1536
 capacity_mem_par_tor_P41: mem[32,58] <= 1536
 capacity_mem_par_tor_P41: mem[32,16] <= 1536
 capacity_mem_par_tor_P41: mem[32,49] <= 1536
 capacity_mem_par_tor_P41: mem[32,13] <= 1536
 capacity_mem_par_tor_P41: mem[32,72] <= 1536
 capacity_mem_par_tor_P41: mem[32,46] <= 1536
 capacity_mem_par_tor_P41: mem[32,37] <= 1536
 capacity_mem_par_tor_P41: mem[32,27] <= 1536
 capacity_mem_par_tor_P41: mem[32,70] <= 1536
 capacity_mem_par_tor_P41: mem[32,60] <= 1536
 capacity_mem_par_tor_P41: mem[32,8] <= 1536
 capacity_mem_par_tor_P41: mem[32,51] <= 1536
 capacity_mem_par_tor_P41: mem[32,41] <= 1536
 capacity_mem_par_tor_P41: mem[32,74] <= 1536
 capacity_mem_par_tor_P41: mem[32,32] <= 1536
 capacity_mem_par_tor_P41: mem[32,6] <= 1536
 capacity_mem_par_tor_P41: mem[32,65] <= 1536
 capacity_mem_par_tor_P41: mem[32,39] <= 1536
 capacity_mem_par_tor_P41: mem[32,88] <= 1536
 capacity_mem_par_tor_P41: mem[32,62] <= 1536
 capacity_mem_par_tor_P41: mem[32,53] <= 1536
 capacity_mem_par_tor_P41: mem[32,43] <= 1536
 capacity_mem_par_tor_P41: mem[32,34] <= 1536
 capacity_mem_par_tor_P41: mem[32,24] <= 1536
 capacity_mem_par_tor_P41: mem[32,67] <= 1536
 capacity_mem_par_tor_P41: mem[32,57] <= 1536
 capacity_mem_par_tor_P41: mem[32,31] <= 1536
 capacity_mem_par_tor_P41: mem[32,48] <= 1536
 capacity_mem_par_tor_P41: mem[32,22] <= 1536
 capacity_mem_par_tor_P41: mem[32,55] <= 1536
 capacity_mem_par_tor_P41: mem[32,3] <= 1536
 capacity_mem_par_tor_P41: mem[32,78] <= 1536
 capacity_mem_par_tor_P41: mem[32,36] <= 1536
 capacity_mem_par_tor_P41: mem[32,69] <= 1536
 capacity_mem_par_tor_P41: mem[32,59] <= 1536
 capacity_mem_par_tor_P41: mem[32,17] <= 1536
 capacity_mem_par_tor_P41: mem[32,50] <= 1536
 capacity_mem_par_tor_P41: mem[32,40] <= 1536
 capacity_mem_par_tor_P41: mem[32,83] <= 1536
 capacity_mem_par_tor_P41: mem[32,73] <= 1536
 capacity_mem_par_tor_P41: mem[32,47] <= 1536
 capacity_mem_par_tor_P41: mem[32,64] <= 1536
 capacity_mem_par_tor_P41: mem[32,38] <= 1536
 capacity_mem_par_tor_P41: mem[32,71] <= 1536
 capacity_mem_par_tor_P41: mem[32,61] <= 1536
 capacity_mem_par_tor_P41: mem[32,0] <= 1536
 capacity_mem_par_tor_P41: mem[32,85] <= 1536
 capacity_mem_par_tor_P41: mem[32,75] <= 1536
 capacity_mem_par_tor_P41: mem[32,33] <= 1536
 capacity_mem_par_tor_P41: mem[32,66] <= 1536
 capacity_mem_par_tor_P41: mem[32,89] <= 1536
 capacity_mem_par_tor_P41: mem[32,63] <= 1536
 row_capacity_tor_P42: rows_tot_tor_P42 <= 48
 capacity_mem_par_tor_P42: mem[33,55] <= 1536
 capacity_mem_par_tor_P42: mem[33,41] <= 1536
 capacity_mem_par_tor_P42: mem[33,3] <= 1536
 capacity_mem_par_tor_P42: mem[33,101] <= 1536
 capacity_mem_par_tor_P42: mem[33,95] <= 1536
 capacity_mem_par_tor_P42: mem[33,80] <= 1536
 capacity_mem_par_tor_P42: mem[33,74] <= 1536
 capacity_mem_par_tor_P42: mem[33,59] <= 1536
 capacity_mem_par_tor_P42: mem[33,29] <= 1536
 capacity_mem_par_tor_P42: mem[33,119] <= 1536
 capacity_mem_par_tor_P42: mem[33,105] <= 1536
 capacity_mem_par_tor_P42: mem[33,67] <= 1536
 capacity_mem_par_tor_P42: mem[33,60] <= 1536
 capacity_mem_par_tor_P42: mem[33,22] <= 1536
 capacity_mem_par_tor_P42: mem[33,8] <= 1536
 capacity_mem_par_tor_P42: mem[33,68] <= 1536
 capacity_mem_par_tor_P42: mem[33,47] <= 1536
 capacity_mem_par_tor_P42: mem[33,1] <= 1536
 capacity_mem_par_tor_P42: mem[33,93] <= 1536
 capacity_mem_par_tor_P42: mem[33,72] <= 1536
 capacity_mem_par_tor_P42: mem[33,19] <= 1536
 capacity_mem_par_tor_P42: mem[33,117] <= 1536
 capacity_mem_par_tor_P42: mem[33,111] <= 1536
 capacity_mem_par_tor_P42: mem[33,65] <= 1536
 capacity_mem_par_tor_P42: mem[33,50] <= 1536
 capacity_mem_par_tor_P42: mem[33,20] <= 1536
 capacity_mem_par_tor_P42: mem[33,14] <= 1536
 capacity_mem_par_tor_P42: mem[33,96] <= 1536
 capacity_mem_par_tor_P42: mem[33,90] <= 1536
 capacity_mem_par_tor_P42: mem[33,7] <= 1536
 capacity_mem_par_tor_P42: mem[33,83] <= 1536
 capacity_mem_par_tor_P42: mem[33,24] <= 1536
 capacity_mem_par_tor_P42: mem[33,84] <= 1536
 capacity_mem_par_tor_P42: mem[33,78] <= 1536
 capacity_mem_par_tor_P42: mem[33,63] <= 1536
 capacity_mem_par_tor_P42: mem[33,17] <= 1536
 capacity_mem_par_tor_P42: mem[33,11] <= 1536
 capacity_mem_par_tor_P42: mem[33,109] <= 1536
 capacity_mem_par_tor_P42: mem[33,71] <= 1536
 capacity_mem_par_tor_P42: mem[33,88] <= 1536
 capacity_mem_par_tor_P42: mem[33,75] <= 1536
 capacity_mem_par_tor_P42: mem[33,106] <= 1536
 capacity_mem_par_tor_P42: mem[33,76] <= 1536
 capacity_mem_par_tor_P42: mem[33,61] <= 1536
 capacity_mem_par_tor_P42: mem[33,69] <= 1536
 capacity_mem_par_tor_P42: mem[33,100] <= 1536
 capacity_mem_par_tor_P42: mem[33,27] <= 1536
 capacity_mem_par_tor_P42: mem[33,87] <= 1536
 capacity_mem_par_tor_P42: mem[33,73] <= 1536
 capacity_mem_par_tor_P42: mem[33,28] <= 1536
 capacity_mem_par_tor_P42: mem[33,104] <= 1536
 capacity_mem_par_tor_P42: mem[33,66] <= 1536
 capacity_mem_par_tor_P42: mem[33,97] <= 1536
 capacity_mem_par_tor_P42: mem[33,91] <= 1536
 capacity_mem_par_tor_P42: mem[33,52] <= 1536
 capacity_mem_par_tor_P42: mem[33,0] <= 1536
 capacity_mem_par_tor_P42: mem[33,92] <= 1536
 capacity_mem_par_tor_P42: mem[33,39] <= 1536
 capacity_mem_par_tor_P42: mem[33,25] <= 1536
 capacity_mem_par_tor_P42: mem[33,85] <= 1536
 capacity_mem_par_tor_P42: mem[33,79] <= 1536
 capacity_mem_par_tor_P42: mem[33,56] <= 1536
 capacity_mem_par_tor_P42: mem[33,110] <= 1536
 capacity_mem_par_tor_P42: mem[33,64] <= 1536
 capacity_mem_par_tor_P42: mem[33,13] <= 1536
 capacity_mem_par_tor_P42: mem[33,89] <= 1536
 capacity_mem_par_tor_P42: mem[33,82] <= 1536
 capacity_mem_par_tor_P42: mem[33,37] <= 1536
 capacity_mem_par_tor_P42: mem[33,31] <= 1536
 capacity_mem_par_tor_P42: mem[33,77] <= 1536
 capacity_mem_par_tor_P42: mem[33,62] <= 1536
 capacity_mem_par_tor_P42: mem[33,16] <= 1536
 capacity_mem_par_tor_P42: mem[33,70] <= 1536
 row_capacity_tor_P43: rows_tot_tor_P43 <= 48
 capacity_mem_par_tor_P43: mem[34,8] <= 1536
 capacity_mem_par_tor_P43: mem[34,79] <= 1536
 capacity_mem_par_tor_P43: mem[34,3] <= 1536
 capacity_mem_par_tor_P43: mem[34,114] <= 1536
 capacity_mem_par_tor_P43: mem[34,70] <= 1536
 capacity_mem_par_tor_P43: mem[34,103] <= 1536
 capacity_mem_par_tor_P43: mem[34,21] <= 1536
 capacity_mem_par_tor_P43: mem[34,50] <= 1536
 capacity_mem_par_tor_P43: mem[34,44] <= 1536
 capacity_mem_par_tor_P43: mem[34,6] <= 1536
 capacity_mem_par_tor_P43: mem[34,77] <= 1536
 capacity_mem_par_tor_P43: mem[34,39] <= 1536
 capacity_mem_par_tor_P43: mem[34,112] <= 1536
 capacity_mem_par_tor_P43: mem[34,106] <= 1536
 capacity_mem_par_tor_P43: mem[34,62] <= 1536
 capacity_mem_par_tor_P43: mem[34,24] <= 1536
 capacity_mem_par_tor_P43: mem[34,19] <= 1536
 capacity_mem_par_tor_P43: mem[34,13] <= 1536
 capacity_mem_par_tor_P43: mem[34,48] <= 1536
 capacity_mem_par_tor_P43: mem[34,4] <= 1536
 capacity_mem_par_tor_P43: mem[34,37] <= 1536
 capacity_mem_par_tor_P43: mem[34,104] <= 1536
 capacity_mem_par_tor_P43: mem[34,66] <= 1536
 capacity_mem_par_tor_P43: mem[34,60] <= 1536
 capacity_mem_par_tor_P43: mem[34,22] <= 1536
 capacity_mem_par_tor_P43: mem[34,93] <= 1536
 capacity_mem_par_tor_P43: mem[34,55] <= 1536
 capacity_mem_par_tor_P43: mem[34,17] <= 1536
 capacity_mem_par_tor_P43: mem[34,11] <= 1536
 capacity_mem_par_tor_P43: mem[34,84] <= 1536
 capacity_mem_par_tor_P43: mem[34,117] <= 1536
 capacity_mem_par_tor_P43: mem[34,111] <= 1536
 capacity_mem_par_tor_P43: mem[34,73] <= 1536
 capacity_mem_par_tor_P43: mem[34,35] <= 1536
 capacity_mem_par_tor_P43: mem[34,29] <= 1536
 capacity_mem_par_tor_P43: mem[34,102] <= 1536
 capacity_mem_par_tor_P43: mem[34,64] <= 1536
 capacity_mem_par_tor_P43: mem[34,20] <= 1536
 capacity_mem_par_tor_P43: mem[34,14] <= 1536
 capacity_mem_par_tor_P43: mem[34,91] <= 1536
 capacity_mem_par_tor_P43: mem[34,38] <= 1536
 capacity_mem_par_tor_P43: mem[34,0] <= 1536
 capacity_mem_par_tor_P43: mem[34,115] <= 1536
 capacity_mem_par_tor_P43: mem[34,109] <= 1536
 capacity_mem_par_tor_P43: mem[34,71] <= 1536
 capacity_mem_par_tor_P43: mem[34,27] <= 1536
 capacity_mem_par_tor_P43: mem[34,100] <= 1536
 capacity_mem_par_tor_P43: mem[34,94] <= 1536
 capacity_mem_par_tor_P43: mem[34,12] <= 1536
 capacity_mem_par_tor_P43: mem[34,89] <= 1536
 capacity_mem_par_tor_P43: mem[34,7] <= 1536
 capacity_mem_par_tor_P43: mem[34,30] <= 1536
 capacity_mem_par_tor_P43: mem[34,113] <= 1536
 capacity_mem_par_tor_P43: mem[34,107] <= 1536
 capacity_mem_par_tor_P43: mem[34,25] <= 1536
 capacity_mem_par_tor_P43: mem[34,92] <= 1536
 capacity_mem_par_tor_P43: mem[34,16] <= 1536
 capacity_mem_par_tor_P43: mem[34,87] <= 1536
 capacity_mem_par_tor_P43: mem[34,49] <= 1536
 capacity_mem_par_tor_P43: mem[34,5] <= 1536
 capacity_mem_par_tor_P43: mem[34,116] <= 1536
 capacity_mem_par_tor_P43: mem[34,110] <= 1536
 capacity_mem_par_tor_P43: mem[34,72] <= 1536
 capacity_mem_par_tor_P43: mem[34,34] <= 1536
 capacity_mem_par_tor_P43: mem[34,105] <= 1536
 capacity_mem_par_tor_P43: mem[34,61] <= 1536
 capacity_mem_par_tor_P43: mem[34,23] <= 1536
 capacity_mem_par_tor_P43: mem[34,96] <= 1536
 capacity_mem_par_tor_P43: mem[34,90] <= 1536
 row_capacity_tor_P44: rows_tot_tor_P44 <= 48
 capacity_mem_par_tor_P44: mem[35,21] <= 1536
 capacity_mem_par_tor_P44: mem[35,46] <= 1536
 capacity_mem_par_tor_P44: mem[35,52] <= 1536
 capacity_mem_par_tor_P44: mem[35,96] <= 1536
 capacity_mem_par_tor_P44: mem[35,105] <= 1536
 capacity_mem_par_tor_P44: mem[35,34] <= 1536
 capacity_mem_par_tor_P44: mem[35,8] <= 1536
 capacity_mem_par_tor_P44: mem[35,110] <= 1536
 capacity_mem_par_tor_P44: mem[35,90] <= 1536
 capacity_mem_par_tor_P44: mem[35,43] <= 1536
 capacity_mem_par_tor_P44: mem[35,49] <= 1536
 capacity_mem_par_tor_P44: mem[35,98] <= 1536
 capacity_mem_par_tor_P44: mem[35,5] <= 1536
 capacity_mem_par_tor_P44: mem[35,107] <= 1536
 capacity_mem_par_tor_P44: mem[35,113] <= 1536
 capacity_mem_par_tor_P44: mem[35,16] <= 1536
 capacity_mem_par_tor_P44: mem[35,118] <= 1536
 capacity_mem_par_tor_P44: mem[35,51] <= 1536
 capacity_mem_par_tor_P44: mem[35,25] <= 1536
 capacity_mem_par_tor_P44: mem[35,56] <= 1536
 capacity_mem_par_tor_P44: mem[35,30] <= 1536
 capacity_mem_par_tor_P44: mem[35,100] <= 1536
 capacity_mem_par_tor_P44: mem[35,7] <= 1536
 capacity_mem_par_tor_P44: mem[35,109] <= 1536
 capacity_mem_par_tor_P44: mem[35,115] <= 1536
 capacity_mem_par_tor_P44: mem[35,38] <= 1536
 capacity_mem_par_tor_P44: mem[35,12] <= 1536
 capacity_mem_par_tor_P44: mem[35,94] <= 1536
 capacity_mem_par_tor_P44: mem[35,97] <= 1536
 capacity_mem_par_tor_P44: mem[35,58] <= 1536
 capacity_mem_par_tor_P44: mem[35,0] <= 1536
 capacity_mem_par_tor_P44: mem[35,102] <= 1536
 capacity_mem_par_tor_P44: mem[35,35] <= 1536
 capacity_mem_par_tor_P44: mem[35,9] <= 1536
 capacity_mem_par_tor_P44: mem[35,111] <= 1536
 capacity_mem_par_tor_P44: mem[35,117] <= 1536
 capacity_mem_par_tor_P44: mem[35,91] <= 1536
 capacity_mem_par_tor_P44: mem[35,14] <= 1536
 capacity_mem_par_tor_P44: mem[35,55] <= 1536
 capacity_mem_par_tor_P44: mem[35,99] <= 1536
 capacity_mem_par_tor_P44: mem[35,104] <= 1536
 capacity_mem_par_tor_P44: mem[35,37] <= 1536
 capacity_mem_par_tor_P44: mem[35,11] <= 1536
 capacity_mem_par_tor_P44: mem[35,17] <= 1536
 capacity_mem_par_tor_P44: mem[35,119] <= 1536
 capacity_mem_par_tor_P44: mem[35,93] <= 1536
 capacity_mem_par_tor_P44: mem[35,48] <= 1536
 capacity_mem_par_tor_P44: mem[35,22] <= 1536
 capacity_mem_par_tor_P44: mem[35,31] <= 1536
 capacity_mem_par_tor_P44: mem[35,101] <= 1536
 capacity_mem_par_tor_P44: mem[35,4] <= 1536
 capacity_mem_par_tor_P44: mem[35,106] <= 1536
 capacity_mem_par_tor_P44: mem[35,112] <= 1536
 capacity_mem_par_tor_P44: mem[35,39] <= 1536
 capacity_mem_par_tor_P44: mem[35,19] <= 1536
 capacity_mem_par_tor_P44: mem[35,95] <= 1536
 capacity_mem_par_tor_P44: mem[35,44] <= 1536
 capacity_mem_par_tor_P44: mem[35,50] <= 1536
 capacity_mem_par_tor_P44: mem[35,24] <= 1536
 capacity_mem_par_tor_P44: mem[35,1] <= 1536
 capacity_mem_par_tor_P44: mem[35,103] <= 1536
 capacity_mem_par_tor_P44: mem[35,32] <= 1536
 capacity_mem_par_tor_P44: mem[35,6] <= 1536
 capacity_mem_par_tor_P44: mem[35,114] <= 1536
 row_capacity_l1_P41: rows_tot_l1_P41 <= 48
 capacity_mem_par_l1_P41: mem[36,68] <= 1536
 capacity_mem_par_l1_P41: mem[36,8] <= 1536
 capacity_mem_par_l1_P41: mem[36,22] <= 1536
 capacity_mem_par_l1_P41: mem[36,101] <= 1536
 capacity_mem_par_l1_P41: mem[36,67] <= 1536
 capacity_mem_par_l1_P41: mem[36,41] <= 1536
 capacity_mem_par_l1_P41: mem[36,55] <= 1536
 capacity_mem_par_l1_P41: mem[36,29] <= 1536
 capacity_mem_par_l1_P41: mem[36,74] <= 1536
 capacity_mem_par_l1_P41: mem[36,62] <= 1536
 capacity_mem_par_l1_P41: mem[36,95] <= 1536
 capacity_mem_par_l1_P41: mem[36,37] <= 1536
 capacity_mem_par_l1_P41: mem[36,3] <= 1536
 capacity_mem_par_l1_P41: mem[36,70] <= 1536
 capacity_mem_par_l1_P41: mem[36,44] <= 1536
 capacity_mem_par_l1_P41: mem[36,16] <= 1536
 capacity_mem_par_l1_P41: mem[36,77] <= 1536
 capacity_mem_par_l1_P41: mem[36,43] <= 1536
 capacity_mem_par_l1_P41: mem[36,49] <= 1536
 capacity_mem_par_l1_P41: mem[36,31] <= 1536
 capacity_mem_par_l1_P41: mem[36,110] <= 1536
 capacity_mem_par_l1_P41: mem[36,56] <= 1536
 capacity_mem_par_l1_P41: mem[36,6] <= 1536
 capacity_mem_par_l1_P41: mem[36,89] <= 1536
 capacity_mem_par_l1_P41: mem[36,39] <= 1536
 capacity_mem_par_l1_P41: mem[36,13] <= 1536
 capacity_mem_par_l1_P41: mem[36,64] <= 1536
 capacity_mem_par_l1_P41: mem[36,46] <= 1536
 capacity_mem_par_l1_P41: mem[36,52] <= 1536
 capacity_mem_par_l1_P41: mem[36,97] <= 1536
 capacity_mem_par_l1_P41: mem[36,79] <= 1536
 capacity_mem_par_l1_P41: mem[36,85] <= 1536
 capacity_mem_par_l1_P41: mem[36,51] <= 1536
 capacity_mem_par_l1_P41: mem[36,25] <= 1536
 capacity_mem_par_l1_P41: mem[36,104] <= 1536
 capacity_mem_par_l1_P41: mem[36,92] <= 1536
 capacity_mem_par_l1_P41: mem[36,58] <= 1536
 capacity_mem_par_l1_P41: mem[36,0] <= 1536
 capacity_mem_par_l1_P41: mem[36,91] <= 1536
 capacity_mem_par_l1_P41: mem[36,100] <= 1536
 capacity_mem_par_l1_P41: mem[36,66] <= 1536
 capacity_mem_par_l1_P41: mem[36,73] <= 1536
 capacity_mem_par_l1_P41: mem[36,87] <= 1536
 capacity_mem_par_l1_P41: mem[36,61] <= 1536
 capacity_mem_par_l1_P41: mem[36,27] <= 1536
 capacity_mem_par_l1_P41: mem[36,106] <= 1536
 capacity_mem_par_l1_P41: mem[36,69] <= 1536
 capacity_mem_par_l1_P41: mem[36,35] <= 1536
 capacity_mem_par_l1_P41: mem[36,48] <= 1536
 capacity_mem_par_l1_P41: mem[36,109] <= 1536
 capacity_mem_par_l1_P41: mem[36,63] <= 1536
 capacity_mem_par_l1_P41: mem[36,88] <= 1536
 capacity_mem_par_l1_P41: mem[36,38] <= 1536
 capacity_mem_par_l1_P41: mem[36,71] <= 1536
 capacity_mem_par_l1_P41: mem[36,11] <= 1536
 capacity_mem_par_l1_P41: mem[36,17] <= 1536
 capacity_mem_par_l1_P41: mem[36,96] <= 1536
 capacity_mem_par_l1_P41: mem[36,78] <= 1536
 capacity_mem_par_l1_P41: mem[36,84] <= 1536
 capacity_mem_par_l1_P41: mem[36,50] <= 1536
 capacity_mem_par_l1_P41: mem[36,24] <= 1536
 capacity_mem_par_l1_P41: mem[36,111] <= 1536
 capacity_mem_par_l1_P41: mem[36,117] <= 1536
 capacity_mem_par_l1_P41: mem[36,83] <= 1536
 capacity_mem_par_l1_P41: mem[36,7] <= 1536
 capacity_mem_par_l1_P41: mem[36,90] <= 1536
 capacity_mem_par_l1_P41: mem[36,32] <= 1536
 capacity_mem_par_l1_P41: mem[36,14] <= 1536
 capacity_mem_par_l1_P41: mem[36,20] <= 1536
 capacity_mem_par_l1_P41: mem[36,65] <= 1536
 capacity_mem_par_l1_P41: mem[36,47] <= 1536
 capacity_mem_par_l1_P41: mem[36,19] <= 1536
 capacity_mem_par_l1_P41: mem[36,72] <= 1536
 capacity_mem_par_l1_P41: mem[36,60] <= 1536
 capacity_mem_par_l1_P41: mem[36,105] <= 1536
 capacity_mem_par_l1_P41: mem[36,119] <= 1536
 capacity_mem_par_l1_P41: mem[36,93] <= 1536
 capacity_mem_par_l1_P41: mem[36,59] <= 1536
 capacity_mem_par_l1_P41: mem[36,1] <= 1536
 row_capacity_l1_P42: rows_tot_l1_P42 <= 48
 capacity_mem_par_l1_P42: mem[37,35] <= 1536
 capacity_mem_par_l1_P42: mem[37,113] <= 1536
 capacity_mem_par_l1_P42: mem[37,91] <= 1536
 capacity_mem_par_l1_P42: mem[37,44] <= 1536
 capacity_mem_par_l1_P42: mem[37,16] <= 1536
 capacity_mem_par_l1_P42: mem[37,25] <= 1536
 capacity_mem_par_l1_P42: mem[37,77] <= 1536
 capacity_mem_par_l1_P42: mem[37,87] <= 1536
 capacity_mem_par_l1_P42: mem[37,56] <= 1536
 capacity_mem_par_l1_P42: mem[37,11] <= 1536
 capacity_mem_par_l1_P42: mem[37,21] <= 1536
 capacity_mem_par_l1_P42: mem[37,89] <= 1536
 capacity_mem_par_l1_P42: mem[37,52] <= 1536
 capacity_mem_par_l1_P42: mem[37,30] <= 1536
 capacity_mem_par_l1_P42: mem[37,66] <= 1536
 capacity_mem_par_l1_P42: mem[37,61] <= 1536
 capacity_mem_par_l1_P42: mem[37,7] <= 1536
 capacity_mem_par_l1_P42: mem[37,97] <= 1536
 capacity_mem_par_l1_P42: mem[37,38] <= 1536
 capacity_mem_par_l1_P42: mem[37,0] <= 1536
 capacity_mem_par_l1_P42: mem[37,106] <= 1536
 capacity_mem_par_l1_P42: mem[37,94] <= 1536
 capacity_mem_par_l1_P42: mem[37,19] <= 1536
 capacity_mem_par_l1_P42: mem[37,71] <= 1536
 capacity_mem_par_l1_P42: mem[37,50] <= 1536
 capacity_mem_par_l1_P42: mem[37,64] <= 1536
 capacity_mem_par_l1_P42: mem[37,5] <= 1536
 capacity_mem_par_l1_P42: mem[37,111] <= 1536
 capacity_mem_par_l1_P42: mem[37,73] <= 1536
 capacity_mem_par_l1_P42: mem[37,14] <= 1536
 capacity_mem_par_l1_P42: mem[37,104] <= 1536
 capacity_mem_par_l1_P42: mem[37,114] <= 1536
 capacity_mem_par_l1_P42: mem[37,92] <= 1536
 capacity_mem_par_l1_P42: mem[37,55] <= 1536
 capacity_mem_par_l1_P42: mem[37,17] <= 1536
 capacity_mem_par_l1_P42: mem[37,48] <= 1536
 capacity_mem_par_l1_P42: mem[37,100] <= 1536
 capacity_mem_par_l1_P42: mem[37,3] <= 1536
 capacity_mem_par_l1_P42: mem[37,109] <= 1536
 capacity_mem_par_l1_P42: mem[37,119] <= 1536
 capacity_mem_par_l1_P42: mem[37,34] <= 1536
 capacity_mem_par_l1_P42: mem[37,12] <= 1536
 capacity_mem_par_l1_P42: mem[37,22] <= 1536
 capacity_mem_par_l1_P42: mem[37,112] <= 1536
 capacity_mem_par_l1_P42: mem[37,90] <= 1536
 capacity_mem_par_l1_P42: mem[37,43] <= 1536
 capacity_mem_par_l1_P42: mem[37,31] <= 1536
 capacity_mem_par_l1_P42: mem[37,62] <= 1536
 capacity_mem_par_l1_P42: mem[37,24] <= 1536
 capacity_mem_par_l1_P42: mem[37,39] <= 1536
 capacity_mem_par_l1_P42: mem[37,1] <= 1536
 capacity_mem_par_l1_P42: mem[37,107] <= 1536
 capacity_mem_par_l1_P42: mem[37,117] <= 1536
 capacity_mem_par_l1_P42: mem[37,95] <= 1536
 capacity_mem_par_l1_P42: mem[37,32] <= 1536
 capacity_mem_par_l1_P42: mem[37,20] <= 1536
 capacity_mem_par_l1_P42: mem[37,51] <= 1536
 capacity_mem_par_l1_P42: mem[37,29] <= 1536
 capacity_mem_par_l1_P42: mem[37,103] <= 1536
 capacity_mem_par_l1_P42: mem[37,60] <= 1536
 capacity_mem_par_l1_P42: mem[37,6] <= 1536
 capacity_mem_par_l1_P42: mem[37,96] <= 1536
 capacity_mem_par_l1_P42: mem[37,84] <= 1536
 capacity_mem_par_l1_P42: mem[37,37] <= 1536
 capacity_mem_par_l1_P42: mem[37,105] <= 1536
 capacity_mem_par_l1_P42: mem[37,115] <= 1536
 capacity_mem_par_l1_P42: mem[37,93] <= 1536
 capacity_mem_par_l1_P42: mem[37,46] <= 1536
 capacity_mem_par_l1_P42: mem[37,8] <= 1536
 capacity_mem_par_l1_P42: mem[37,70] <= 1536
 capacity_mem_par_l1_P42: mem[37,49] <= 1536
 capacity_mem_par_l1_P42: mem[37,101] <= 1536
 capacity_mem_par_l1_P42: mem[37,79] <= 1536
 capacity_mem_par_l1_P42: mem[37,58] <= 1536
 capacity_mem_par_l1_P42: mem[37,4] <= 1536
 capacity_mem_par_l1_P42: mem[37,110] <= 1536
 capacity_mem_par_l1_P42: mem[37,72] <= 1536
 row_capacity_l2_P4: rows_tot_l2_P4 <= 48
 capacity_mem_par_l2_P4: mem[38,119] <= 1536
 capacity_mem_par_l2_P4: mem[38,73] <= 1536
 capacity_mem_par_l2_P4: mem[38,35] <= 1536
 capacity_mem_par_l2_P4: mem[38,31] <= 1536
 capacity_mem_par_l2_P4: mem[38,110] <= 1536
 capacity_mem_par_l2_P4: mem[38,64] <= 1536
 capacity_mem_par_l2_P4: mem[38,90] <= 1536
 capacity_mem_par_l2_P4: mem[38,60] <= 1536
 capacity_mem_par_l2_P4: mem[38,22] <= 1536
 capacity_mem_par_l2_P4: mem[38,97] <= 1536
 capacity_mem_par_l2_P4: mem[38,93] <= 1536
 capacity_mem_par_l2_P4: mem[38,55] <= 1536
 capacity_mem_par_l2_P4: mem[38,84] <= 1536
 capacity_mem_par_l2_P4: mem[38,46] <= 1536
 capacity_mem_par_l2_P4: mem[38,0] <= 1536
 capacity_mem_par_l2_P4: mem[38,117] <= 1536
 capacity_mem_par_l2_P4: mem[38,79] <= 1536
 capacity_mem_par_l2_P4: mem[38,29] <= 1536
 capacity_mem_par_l2_P4: mem[38,70] <= 1536
 capacity_mem_par_l2_P4: mem[38,50] <= 1536
 capacity_mem_par_l2_P4: mem[38,20] <= 1536
 capacity_mem_par_l2_P4: mem[38,44] <= 1536
 capacity_mem_par_l2_P4: mem[38,6] <= 1536
 capacity_mem_par_l2_P4: mem[38,24] <= 1536
 capacity_mem_par_l2_P4: mem[38,77] <= 1536
 capacity_mem_par_l2_P4: mem[38,39] <= 1536
 capacity_mem_par_l2_P4: mem[38,19] <= 1536
 capacity_mem_par_l2_P4: mem[38,48] <= 1536
 capacity_mem_par_l2_P4: mem[38,101] <= 1536
 capacity_mem_par_l2_P4: mem[38,43] <= 1536
 capacity_mem_par_l2_P4: mem[38,72] <= 1536
 capacity_mem_par_l2_P4: mem[38,105] <= 1536
 capacity_mem_par_l2_P4: mem[38,37] <= 1536
 capacity_mem_par_l2_P4: mem[38,17] <= 1536
 capacity_mem_par_l2_P4: mem[38,96] <= 1536
 capacity_mem_par_l2_P4: mem[38,92] <= 1536
 capacity_mem_par_l2_P4: mem[38,8] <= 1536
 capacity_mem_par_l2_P4: mem[38,87] <= 1536
 capacity_mem_par_l2_P4: mem[38,3] <= 1536
 capacity_mem_par_l2_P4: mem[38,32] <= 1536
 capacity_mem_par_l2_P4: mem[38,58] <= 1536
 capacity_mem_par_l2_P4: mem[38,111] <= 1536
 capacity_mem_par_l2_P4: mem[38,91] <= 1536
 capacity_mem_par_l2_P4: mem[38,61] <= 1536
 capacity_mem_par_l2_P4: mem[38,52] <= 1536
 capacity_mem_par_l2_P4: mem[38,14] <= 1536
 capacity_mem_par_l2_P4: mem[38,1] <= 1536
 capacity_mem_par_l2_P4: mem[38,106] <= 1536
 capacity_mem_par_l2_P4: mem[38,38] <= 1536
 capacity_mem_par_l2_P4: mem[38,56] <= 1536
 capacity_mem_par_l2_P4: mem[38,109] <= 1536
 capacity_mem_par_l2_P4: mem[38,71] <= 1536
 capacity_mem_par_l2_P4: mem[38,89] <= 1536
 capacity_mem_par_l2_P4: mem[38,51] <= 1536
 capacity_mem_par_l2_P4: mem[38,100] <= 1536
 capacity_mem_par_l2_P4: mem[38,7] <= 1536
 capacity_mem_par_l2_P4: mem[38,25] <= 1536
 capacity_mem_par_l2_P4: mem[38,104] <= 1536
 capacity_mem_par_l2_P4: mem[38,66] <= 1536
 capacity_mem_par_l2_P4: mem[38,62] <= 1536
 capacity_mem_par_l2_P4: mem[38,16] <= 1536
 capacity_mem_par_l2_P4: mem[38,95] <= 1536
 capacity_mem_par_l2_P4: mem[38,49] <= 1536
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
 frac[0,17] <= 1
 frac[0,0] <= 1
 frac[0,25] <= 1
 frac[0,16] <= 1
 frac[0,13] <= 1
 frac[0,27] <= 1
 frac[0,8] <= 1
 frac[0,6] <= 1
 frac[0,24] <= 1
 frac[0,22] <= 1
 frac[0,3] <= 1
 frac[1,64] <= 1
 frac[1,89] <= 1
 frac[1,77] <= 1
 frac[1,62] <= 1
 frac[1,83] <= 1
 frac[1,60] <= 1
 frac[1,88] <= 1
 frac[1,73] <= 1
 frac[1,66] <= 1
 frac[2,53] <= 1
 frac[2,38] <= 1
 frac[2,36] <= 1
 frac[2,43] <= 1
 frac[2,34] <= 1
 frac[2,32] <= 1
 frac[2,50] <= 1
 frac[2,39] <= 1
 frac[2,31] <= 1
 frac[2,37] <= 1
 frac[2,40] <= 1
 frac[2,35] <= 1
 frac[3,55] <= 1
 frac[3,48] <= 1
 frac[3,57] <= 1
 frac[3,39] <= 1
 frac[3,44] <= 1
 frac[3,50] <= 1
 frac[3,59] <= 1
 frac[3,41] <= 1
 frac[3,46] <= 1
 frac[3,49] <= 1
 frac[3,33] <= 1
 frac[3,53] <= 1
 frac[4,87] <= 1
 frac[4,63] <= 1
 frac[4,75] <= 1
 frac[4,71] <= 1
 frac[4,78] <= 1
 frac[4,60] <= 1
 frac[4,72] <= 1
 frac[4,62] <= 1
 frac[4,67] <= 1
 frac[4,77] <= 1
 frac[4,64] <= 1
 frac[4,61] <= 1
 frac[5,74] <= 1
 frac[5,70] <= 1
 frac[5,72] <= 1
 frac[5,68] <= 1
 frac[5,60] <= 1
 frac[5,75] <= 1
 frac[5,85] <= 1
 frac[5,71] <= 1
 frac[5,69] <= 1
 frac[5,61] <= 1
 frac[5,67] <= 1
 frac[5,65] <= 1
 frac[6,41] <= 1
 frac[6,58] <= 1
 frac[6,38] <= 1
 frac[6,51] <= 1
 frac[6,36] <= 1
 frac[6,31] <= 1
 frac[6,55] <= 1
 frac[6,46] <= 1
 frac[6,50] <= 1
 frac[6,57] <= 1
 frac[6,48] <= 1
 frac[6,34] <= 1
 frac[7,44] <= 1
 frac[7,37] <= 1
 frac[7,49] <= 1
 frac[7,39] <= 1
 frac[7,40] <= 1
 frac[7,33] <= 1
 frac[7,35] <= 1
 frac[7,47] <= 1
 frac[7,48] <= 1
 frac[7,38] <= 1
 frac[7,31] <= 1
 frac[7,50] <= 1
 frac[8,89] <= 1
 frac[8,64] <= 1
 frac[8,78] <= 1
 frac[8,84] <= 1
 frac[8,73] <= 1
 frac[8,62] <= 1
 frac[8,75] <= 1
 frac[8,77] <= 1
 frac[8,83] <= 1
 frac[8,65] <= 1
 frac[8,61] <= 1
 frac[8,74] <= 1
 frac[9,62] <= 1
 frac[9,70] <= 1
 frac[9,79] <= 1
 frac[9,67] <= 1
 frac[9,77] <= 1
 frac[9,87] <= 1
 frac[9,85] <= 1
 frac[9,71] <= 1
 frac[9,82] <= 1
 frac[9,80] <= 1
 frac[9,89] <= 1
 frac[9,66] <= 1
 frac[9,76] <= 1
 frac[10,7] <= 1
 frac[10,19] <= 1
 frac[10,17] <= 1
 frac[10,24] <= 1
 frac[10,3] <= 1
 frac[10,1] <= 1
 frac[10,29] <= 1
 frac[10,27] <= 1
 frac[10,0] <= 1
 frac[10,11] <= 1
 frac[10,28] <= 1
 frac[11,22] <= 1
 frac[11,8] <= 1
 frac[11,28] <= 1
 frac[11,3] <= 1
 frac[11,25] <= 1
 frac[11,16] <= 1
 frac[11,14] <= 1
 frac[11,11] <= 1
 frac[11,20] <= 1
 frac[11,13] <= 1
 frac[11,17] <= 1
 frac[12,71] <= 1
 frac[12,84] <= 1
 frac[12,74] <= 1
 frac[12,77] <= 1
 frac[12,68] <= 1
 frac[12,67] <= 1
 frac[12,89] <= 1
 frac[12,80] <= 1
 frac[12,70] <= 1
 frac[12,82] <= 1
 frac[12,63] <= 1
 frac[12,64] <= 1
 frac[12,85] <= 1
 frac[12,76] <= 1
 frac[12,75] <= 1
 frac[12,87] <= 1
 frac[12,60] <= 1
 frac[13,101] <= 1
 frac[13,106] <= 1
 frac[13,111] <= 1
 frac[13,104] <= 1
 frac[13,92] <= 1
 frac[13,97] <= 1
 frac[13,109] <= 1
 frac[13,90] <= 1
 frac[13,95] <= 1
 frac[13,100] <= 1
 frac[13,119] <= 1
 frac[13,105] <= 1
 frac[13,93] <= 1
 frac[13,117] <= 1
 frac[13,110] <= 1
 frac[13,91] <= 1
 frac[13,96] <= 1
 frac[14,72] <= 1
 frac[14,83] <= 1
 frac[14,88] <= 1
 frac[14,66] <= 1
 frac[14,69] <= 1
 frac[14,76] <= 1
 frac[14,89] <= 1
 frac[14,67] <= 1
 frac[15,39] <= 1
 frac[15,50] <= 1
 frac[15,59] <= 1
 frac[15,55] <= 1
 frac[15,31] <= 1
 frac[15,56] <= 1
 frac[15,47] <= 1
 frac[15,52] <= 1
 frac[15,41] <= 1
 frac[15,37] <= 1
 frac[16,117] <= 1
 frac[16,110] <= 1
 frac[16,92] <= 1
 frac[16,105] <= 1
 frac[16,96] <= 1
 frac[16,103] <= 1
 frac[16,116] <= 1
 frac[16,109] <= 1
 frac[16,100] <= 1
 frac[16,104] <= 1
 frac[16,102] <= 1
 frac[17,90] <= 1
 frac[17,107] <= 1
 frac[17,112] <= 1
 frac[17,111] <= 1
 frac[17,96] <= 1
 frac[17,115] <= 1
 frac[17,116] <= 1
 frac[17,102] <= 1
 frac[18,112] <= 1
 frac[18,107] <= 1
 frac[18,93] <= 1
 frac[18,110] <= 1
 frac[18,105] <= 1
 frac[18,91] <= 1
 frac[18,103] <= 1
 frac[18,94] <= 1
 frac[18,115] <= 1
 frac[18,106] <= 1
 frac[18,113] <= 1
 frac[18,104] <= 1
 frac[18,90] <= 1
 frac[18,114] <= 1
 frac[19,13] <= 1
 frac[19,27] <= 1
 frac[19,8] <= 1
 frac[19,17] <= 1
 frac[19,22] <= 1
 frac[19,19] <= 1
 frac[19,24] <= 1
 frac[19,21] <= 1
 frac[19,23] <= 1
 frac[19,11] <= 1
 frac[19,16] <= 1
 frac[20,34] <= 1
 frac[20,48] <= 1
 frac[20,30] <= 1
 frac[20,37] <= 1
 frac[20,44] <= 1
 frac[20,50] <= 1
 frac[20,39] <= 1
 frac[20,55] <= 1
 frac[20,35] <= 1
 frac[20,49] <= 1
 frac[20,38] <= 1
 frac[21,77] <= 1
 frac[21,66] <= 1
 frac[21,84] <= 1
 frac[21,62] <= 1
 frac[21,71] <= 1
 frac[21,64] <= 1
 frac[21,73] <= 1
 frac[21,60] <= 1
 frac[21,87] <= 1
 frac[21,89] <= 1
 frac[21,61] <= 1
 frac[21,70] <= 1
 frac[21,79] <= 1
 frac[21,72] <= 1
 frac[22,12] <= 1
 frac[22,7] <= 1
 frac[22,17] <= 1
 frac[22,11] <= 1
 frac[22,23] <= 1
 frac[22,0] <= 1
 frac[22,27] <= 1
 frac[22,6] <= 1
 frac[22,25] <= 1
 frac[22,13] <= 1
 frac[22,22] <= 1
 frac[22,8] <= 1
 frac[22,3] <= 1
 frac[22,29] <= 1
 frac[22,20] <= 1
 frac[22,14] <= 1
 frac[23,19] <= 1
 frac[23,6] <= 1
 frac[23,11] <= 1
 frac[23,0] <= 1
 frac[23,22] <= 1
 frac[23,5] <= 1
 frac[23,14] <= 1
 frac[23,7] <= 1
 frac[23,21] <= 1
 frac[23,8] <= 1
 frac[23,23] <= 1
 frac[23,3] <= 1
 frac[23,4] <= 1
 frac[23,29] <= 1
 frac[24,21] <= 1
 frac[24,9] <= 1
 frac[24,16] <= 1
 frac[24,25] <= 1
 frac[24,6] <= 1
 frac[24,8] <= 1
 frac[24,22] <= 1
 frac[24,17] <= 1
 frac[24,12] <= 1
 frac[24,19] <= 1
 frac[24,7] <= 1
 frac[25,113] <= 1
 frac[25,110] <= 1
 frac[25,119] <= 1
 frac[25,95] <= 1
 frac[25,107] <= 1
 frac[25,93] <= 1
 frac[25,105] <= 1
 frac[25,118] <= 1
 frac[25,97] <= 1
 frac[26,109] <= 1
 frac[26,107] <= 1
 frac[26,114] <= 1
 frac[26,103] <= 1
 frac[26,90] <= 1
 frac[26,110] <= 1
 frac[26,112] <= 1
 frac[26,99] <= 1
 frac[26,117] <= 1
 frac[26,102] <= 1
 frac[26,93] <= 1
 frac[26,115] <= 1
 frac[26,100] <= 1
 frac[27,39] <= 1
 frac[27,43] <= 1
 frac[27,32] <= 1
 frac[27,55] <= 1
 frac[27,34] <= 1
 frac[27,48] <= 1
 frac[27,50] <= 1
 frac[27,30] <= 1
 frac[27,35] <= 1
 frac[28,1] <= 1
 frac[28,24] <= 1
 frac[28,4] <= 1
 frac[28,17] <= 1
 frac[28,9] <= 1
 frac[28,0] <= 1
 frac[28,11] <= 1
 frac[28,25] <= 1
 frac[28,16] <= 1
 frac[28,14] <= 1
 frac[28,5] <= 1
 frac[28,8] <= 1
 frac[28,7] <= 1
 frac[29,101] <= 1
 frac[29,93] <= 1
 frac[29,112] <= 1
 frac[29,106] <= 1
 frac[29,91] <= 1
 frac[29,111] <= 1
 frac[29,104] <= 1
 frac[29,102] <= 1
 frac[29,118] <= 1
 frac[29,98] <= 1
 frac[29,110] <= 1
 frac[29,103] <= 1
 frac[30,51] <= 1
 frac[30,58] <= 1
 frac[30,49] <= 1
 frac[30,56] <= 1
 frac[30,55] <= 1
 frac[30,31] <= 1
 frac[30,39] <= 1
 frac[30,46] <= 1
 frac[30,37] <= 1
 frac[30,44] <= 1
 frac[30,52] <= 1
 frac[30,38] <= 1
 frac[31,103] <= 1
 frac[31,97] <= 1
 frac[31,110] <= 1
 frac[31,99] <= 1
 frac[31,94] <= 1
 frac[31,100] <= 1
 frac[31,112] <= 1
 frac[31,111] <= 1
 frac[31,117] <= 1
 frac[31,96] <= 1
 frac[31,98] <= 1
 frac[31,107] <= 1
 frac[31,113] <= 1
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
 frac[32,44] frac[32,87] frac[32,77] frac[32,35] frac[32,25] frac[32,68]
 frac[32,58] frac[32,16] frac[32,49] frac[32,13] frac[32,72] frac[32,46]
 frac[32,37] frac[32,27] frac[32,70] frac[32,60] frac[32,8] frac[32,51]
 frac[32,41] frac[32,74] frac[32,32] frac[32,6] frac[32,65] frac[32,39]
 frac[32,88] frac[32,62] frac[32,53] frac[32,43] frac[32,34] frac[32,24]
 frac[32,67] frac[32,57] frac[32,31] frac[32,48] frac[32,22] frac[32,55]
 frac[32,3] frac[32,78] frac[32,36] frac[32,69] frac[32,59] frac[32,17]
 frac[32,50] frac[32,40] frac[32,83] frac[32,73] frac[32,47] frac[32,64]
 frac[32,38] frac[32,71] frac[32,61] frac[32,0] frac[32,85] frac[32,75]
 frac[32,33] frac[32,66] frac[32,89] frac[32,63] frac[33,55] frac[33,41]
 frac[33,3] frac[33,101] frac[33,95] frac[33,80] frac[33,74] frac[33,59]
 frac[33,29] frac[33,119] frac[33,105] frac[33,67] frac[33,60] frac[33,22]
 frac[33,8] frac[33,68] frac[33,47] frac[33,1] frac[33,93] frac[33,72]
 frac[33,19] frac[33,117] frac[33,111] frac[33,65] frac[33,50] frac[33,20]
 frac[33,14] frac[33,96] frac[33,90] frac[33,7] frac[33,83] frac[33,24]
 frac[33,84] frac[33,78] frac[33,63] frac[33,17] frac[33,11] frac[33,109]
 frac[33,71] frac[33,88] frac[33,75] frac[33,106] frac[33,76] frac[33,61]
 frac[33,69] frac[33,100] frac[33,27] frac[33,87] frac[33,73] frac[33,28]
 frac[33,104] frac[33,66] frac[33,97] frac[33,91] frac[33,52] frac[33,0]
 frac[33,92] frac[33,39] frac[33,25] frac[33,85] frac[33,79] frac[33,56]
 frac[33,110] frac[33,64] frac[33,13] frac[33,89] frac[33,82] frac[33,37]
 frac[33,31] frac[33,77] frac[33,62] frac[33,16] frac[33,70] frac[34,8]
 frac[34,79] frac[34,3] frac[34,114] frac[34,70] frac[34,103] frac[34,21]
 frac[34,50] frac[34,44] frac[34,6] frac[34,77] frac[34,39] frac[34,112]
 frac[34,106] frac[34,62] frac[34,24] frac[34,19] frac[34,13] frac[34,48]
 frac[34,4] frac[34,37] frac[34,104] frac[34,66] frac[34,60] frac[34,22]
 frac[34,93] frac[34,55] frac[34,17] frac[34,11] frac[34,84] frac[34,117]
 frac[34,111] frac[34,73] frac[34,35] frac[34,29] frac[34,102] frac[34,64]
 frac[34,20] frac[34,14] frac[34,91] frac[34,38] frac[34,0] frac[34,115]
 frac[34,109] frac[34,71] frac[34,27] frac[34,100] frac[34,94] frac[34,12]
 frac[34,89] frac[34,7] frac[34,30] frac[34,113] frac[34,107] frac[34,25]
 frac[34,92] frac[34,16] frac[34,87] frac[34,49] frac[34,5] frac[34,116]
 frac[34,110] frac[34,72] frac[34,34] frac[34,105] frac[34,61] frac[34,23]
 frac[34,96] frac[34,90] frac[35,21] frac[35,46] frac[35,52] frac[35,96]
 frac[35,105] frac[35,34] frac[35,8] frac[35,110] frac[35,90] frac[35,43]
 frac[35,49] frac[35,98] frac[35,5] frac[35,107] frac[35,113] frac[35,16]
 frac[35,118] frac[35,51] frac[35,25] frac[35,56] frac[35,30] frac[35,100]
 frac[35,7] frac[35,109] frac[35,115] frac[35,38] frac[35,12] frac[35,94]
 frac[35,97] frac[35,58] frac[35,0] frac[35,102] frac[35,35] frac[35,9]
 frac[35,111] frac[35,117] frac[35,91] frac[35,14] frac[35,55] frac[35,99]
 frac[35,104] frac[35,37] frac[35,11] frac[35,17] frac[35,119] frac[35,93]
 frac[35,48] frac[35,22] frac[35,31] frac[35,101] frac[35,4] frac[35,106]
 frac[35,112] frac[35,39] frac[35,19] frac[35,95] frac[35,44] frac[35,50]
 frac[35,24] frac[35,1] frac[35,103] frac[35,32] frac[35,6] frac[35,114]
 frac[36,68] frac[36,8] frac[36,22] frac[36,101] frac[36,67] frac[36,41]
 frac[36,55] frac[36,29] frac[36,74] frac[36,62] frac[36,95] frac[36,37]
 frac[36,3] frac[36,70] frac[36,44] frac[36,16] frac[36,77] frac[36,43]
 frac[36,49] frac[36,31] frac[36,110] frac[36,56] frac[36,6] frac[36,89]
 frac[36,39] frac[36,13] frac[36,64] frac[36,46] frac[36,52] frac[36,97]
 frac[36,79] frac[36,85] frac[36,51] frac[36,25] frac[36,104] frac[36,92]
 frac[36,58] frac[36,0] frac[36,91] frac[36,100] frac[36,66] frac[36,73]
 frac[36,87] frac[36,61] frac[36,27] frac[36,106] frac[36,69] frac[36,35]
 frac[36,48] frac[36,109] frac[36,63] frac[36,88] frac[36,38] frac[36,71]
 frac[36,11] frac[36,17] frac[36,96] frac[36,78] frac[36,84] frac[36,50]
 frac[36,24] frac[36,111] frac[36,117] frac[36,83] frac[36,7] frac[36,90]
 frac[36,32] frac[36,14] frac[36,20] frac[36,65] frac[36,47] frac[36,19]
 frac[36,72] frac[36,60] frac[36,105] frac[36,119] frac[36,93] frac[36,59]
 frac[36,1] frac[37,35] frac[37,113] frac[37,91] frac[37,44] frac[37,16]
 frac[37,25] frac[37,77] frac[37,87] frac[37,56] frac[37,11] frac[37,21]
 frac[37,89] frac[37,52] frac[37,30] frac[37,66] frac[37,61] frac[37,7]
 frac[37,97] frac[37,38] frac[37,0] frac[37,106] frac[37,94] frac[37,19]
 frac[37,71] frac[37,50] frac[37,64] frac[37,5] frac[37,111] frac[37,73]
 frac[37,14] frac[37,104] frac[37,114] frac[37,92] frac[37,55] frac[37,17]
 frac[37,48] frac[37,100] frac[37,3] frac[37,109] frac[37,119] frac[37,34]
 frac[37,12] frac[37,22] frac[37,112] frac[37,90] frac[37,43] frac[37,31]
 frac[37,62] frac[37,24] frac[37,39] frac[37,1] frac[37,107] frac[37,117]
 frac[37,95] frac[37,32] frac[37,20] frac[37,51] frac[37,29] frac[37,103]
 frac[37,60] frac[37,6] frac[37,96] frac[37,84] frac[37,37] frac[37,105]
 frac[37,115] frac[37,93] frac[37,46] frac[37,8] frac[37,70] frac[37,49]
 frac[37,101] frac[37,79] frac[37,58] frac[37,4] frac[37,110] frac[37,72]
 frac[38,119] frac[38,73] frac[38,35] frac[38,31] frac[38,110] frac[38,64]
 frac[38,90] frac[38,60] frac[38,22] frac[38,97] frac[38,93] frac[38,55]
 frac[38,84] frac[38,46] frac[38,0] frac[38,117] frac[38,79] frac[38,29]
 frac[38,70] frac[38,50] frac[38,20] frac[38,44] frac[38,6] frac[38,24]
 frac[38,77] frac[38,39] frac[38,19] frac[38,48] frac[38,101] frac[38,43]
 frac[38,72] frac[38,105] frac[38,37] frac[38,17] frac[38,96] frac[38,92]
 frac[38,8] frac[38,87] frac[38,3] frac[38,32] frac[38,58] frac[38,111]
 frac[38,91] frac[38,61] frac[38,52] frac[38,14] frac[38,1] frac[38,106]
 frac[38,38] frac[38,56] frac[38,109] frac[38,71] frac[38,89] frac[38,51]
 frac[38,100] frac[38,7] frac[38,25] frac[38,104] frac[38,66] frac[38,62]
 frac[38,16] frac[38,95] frac[38,49] frac[38,11]
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

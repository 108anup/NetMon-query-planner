\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_fixed_thr + 0 max_mem_others + 0 max_rows_fixed_thr
   + 0 max_rows_others + 0 m_access_time_CPU12 + 0 m_access_time_CPU26
   + 0 m_access_time_CPU1 + 0 m_access_time_CPU4 + 0 m_access_time_CPU24
   + 0 m_access_time_CPU21 + 0 m_access_time_CPU2 + 0 m_access_time_CPU32
   + 0 m_access_time_CPU11 + 0 m_access_time_CPU14 + 0 m_access_time_CPU10
   + 0 m_access_time_CPU23 + 0 m_access_time_CPU31 + 0 m_access_time_CPU13
   + 0 m_access_time_CPU16 + 0 m_access_time_CPU5 + res
Subject To
 cov_0_48: frac[17,48] + frac[19,48] + frac[20,48] + frac[21,48]
   >= 0.0174942857142859
 cov_1_19: frac[17,19] + frac[18,19] + frac[20,19] >= 0.9
 cov_2_19: frac[17,19] + frac[18,19] + frac[20,19] >= 0.7
 cov_3_12: frac[18,12] >= 0.8
 cov_4_23: frac[16,23] + frac[18,23] + frac[19,23] + frac[20,23]
   + frac[21,23] >= 0.0174942857142859
 cov_5_48: frac[17,48] + frac[18,48] + frac[20,48] >= 0.0174942857142859
 cov_6_32: frac[18,32] >= 0.0174942857142859
 cov_7_29: frac[16,29] + frac[17,29] + frac[19,29] + frac[20,29]
   + frac[21,29] >= 0.7
 cov_8_2: frac[16,2] + frac[17,2] + frac[19,2] + frac[20,2] + frac[21,2]
   >= 0.0174942857142859
 cov_9_19: frac[16,19] + frac[19,19] >= 1
 cov_10_2: frac[18,2] >= 0.0174942857142859
 cov_11_32: frac[18,32] + frac[19,32] + frac[20,32] + frac[21,32]
   >= 0.0174942857142859
 cov_12_40: frac[17,40] + frac[18,40] + frac[20,40] >= 0.0174942857142859
 cov_13_43: frac[17,43] + frac[18,43] + frac[20,43] >= 1
 cov_14_43: frac[17,43] + frac[18,43] + frac[20,43] >= 0.8
 cov_15_23: frac[16,23] + frac[18,23] + frac[19,23] + frac[20,23]
   + frac[21,23] >= 0.0174942857142859
 cov_16_40: frac[17,40] + frac[19,40] + frac[20,40] + frac[21,40]
   >= 0.0174942857142859
 cov_17_16: frac[18,16] >= 1
 cov_18_2: frac[17,2] + frac[19,2] + frac[20,2] + frac[21,2]
   >= 0.0174942857142859
 cov_19_23: frac[17,23] + frac[18,23] + frac[20,23] >= 0.0174942857142859
 cov_20_29: frac[17,29] + frac[18,29] + frac[20,29] >= 1
 cov_21_3: frac[16,3] + frac[18,3] + frac[19,3] + frac[20,3] + frac[21,3]
   >= 0.0174942857142859
 cov_22_48: frac[17,48] + frac[18,48] + frac[20,48] >= 0.0174942857142859
 cov_23_36: frac[18,36] + frac[19,36] + frac[20,36] + frac[21,36] >= 1
 cov_24_12: frac[16,12] + frac[17,12] + frac[19,12] + frac[20,12]
   + frac[21,12] >= 1
 cov_25_40: frac[18,40] >= 0.0174942857142859
 cov_26_3: frac[17,3] + frac[18,3] + frac[20,3] >= 0.0174942857142859
 cov_27_3: frac[17,3] + frac[18,3] + frac[20,3] >= 0.0174942857142859
 cov_28_16: frac[16,16] + frac[17,16] + frac[19,16] + frac[20,16]
   + frac[21,16] >= 0.7
 cov_29_28: frac[16,28] + frac[19,28] >= 0.9
 cov_30_39: frac[16,39] >= 1
 cov_31_18: frac[16,18] + frac[17,18] + frac[19,18] + frac[20,18]
   + frac[21,18] >= 0.9
 cov_32_35: frac[16,35] + frac[19,35] >= 1
 cov_33_46: frac[16,46] + frac[17,46] + frac[19,46] + frac[20,46]
   + frac[21,46] >= 0.0174942857142859
 cov_34_15: frac[18,15] >= 1
 cov_35_15: frac[16,15] + frac[18,15] + frac[19,15] + frac[20,15]
   + frac[21,15] >= 1
 cov_36_47: frac[17,47] + frac[18,47] + frac[20,47] >= 0.0174942857142859
 cov_37_18: frac[16,18] + frac[17,18] + frac[19,18] + frac[20,18]
   + frac[21,18] >= 0.7
 cov_38_11: frac[18,11] + frac[19,11] + frac[20,11] + frac[21,11] >= 1
 cov_39_11: frac[16,11] + frac[19,11] >= 0.9
 cov_40_46: frac[16,46] + frac[18,46] + frac[19,46] + frac[20,46]
   + frac[21,46] >= 0.0174942857142859
 cov_41_27: frac[18,27] + frac[19,27] + frac[20,27] + frac[21,27] >= 1
 cov_42_35: frac[16,35] >= 1
 cov_43_18: frac[17,18] + frac[18,18] + frac[20,18] >= 1
 cov_44_18: frac[18,18] + frac[19,18] + frac[20,18] + frac[21,18] >= 0.8
 cov_45_22: frac[17,22] + frac[18,22] + frac[20,22] >= 0.9
 cov_46_22: frac[16,22] + frac[18,22] + frac[19,22] + frac[20,22]
   + frac[21,22] >= 0.9
 cov_47_15: frac[16,15] + frac[17,15] + frac[19,15] + frac[20,15]
   + frac[21,15] >= 0.9
 cov_48_22: frac[17,22] + frac[19,22] + frac[20,22] + frac[21,22] >= 1
 cov_49_11: frac[16,11] + frac[19,11] >= 1
 cov_50_1: frac[17,1] + frac[19,1] + frac[20,1] + frac[21,1]
   >= 0.0174942857142859
 cov_51_18: frac[18,18] + frac[19,18] + frac[20,18] + frac[21,18] >= 0.8
 cov_52_15: frac[16,15] + frac[19,15] >= 0.7
 cov_53_11: frac[18,11] + frac[19,11] + frac[20,11] + frac[21,11] >= 1
 cov_54_28: frac[16,28] + frac[19,28] >= 1
 cov_55_22: frac[16,22] + frac[17,22] + frac[19,22] + frac[20,22]
   + frac[21,22] >= 0.9
 cov_56_42: frac[6,42] + frac[7,42] + frac[16,42] + frac[18,42]
   + frac[19,42] + frac[20,42] + frac[21,42] >= 0.9
 cov_57_21: frac[1,21] + frac[3,21] + frac[16,21] + frac[18,21]
   + frac[19,21] + frac[20,21] + frac[21,21] >= 0.8
 cov_58_10: frac[5,10] + frac[6,10] + frac[16,10] + frac[17,10]
   + frac[19,10] + frac[20,10] + frac[21,10] >= 0.9
 cov_59_25: frac[1,25] + frac[4,25] + frac[17,25] + frac[18,25]
   + frac[20,25] >= 0.7
 cov_60_38: frac[0,38] + frac[4,38] + frac[17,38] + frac[19,38]
   + frac[20,38] + frac[21,38] >= 0.0174942857142859
 cov_61_45: frac[2,45] + frac[7,45] + frac[16,45] + frac[18,45]
   + frac[19,45] + frac[20,45] + frac[21,45] >= 0.9
 cov_62_7: frac[6,7] + frac[7,7] + frac[16,7] + frac[18,7] + frac[19,7]
   + frac[20,7] + frac[21,7] >= 1
 cov_63_42: frac[0,42] + frac[3,42] + frac[16,42] + frac[19,42] >= 0.8
 cov_64_9: frac[6,9] + frac[7,9] + frac[16,9] + frac[18,9] + frac[19,9]
   + frac[20,9] + frac[21,9] >= 1
 cov_65_34: frac[2,34] + frac[7,34] + frac[16,34] + frac[18,34]
   + frac[19,34] + frac[20,34] + frac[21,34] >= 1
 cov_66_7: frac[6,7] + frac[7,7] + frac[16,7] + frac[18,7] + frac[19,7]
   + frac[20,7] + frac[21,7] >= 0.9
 cov_67_5: frac[1,5] + frac[7,5] + frac[18,5] >= 0.8
 cov_68_21: frac[2,21] + frac[7,21] + frac[16,21] + frac[18,21]
   + frac[19,21] + frac[20,21] + frac[21,21] >= 1
 cov_69_21: frac[1,21] + frac[4,21] + frac[17,21] + frac[18,21]
   + frac[20,21] >= 1
 cov_70_31: frac[4,31] + frac[6,31] + frac[16,31] + frac[17,31]
   + frac[19,31] + frac[20,31] + frac[21,31] >= 0.9
 cov_71_10: frac[3,10] + frac[7,10] + frac[16,10] + frac[18,10]
   + frac[19,10] + frac[20,10] + frac[21,10] >= 0.8
 cov_72_26: frac[0,26] + frac[1,26] + frac[18,26] + frac[19,26]
   + frac[20,26] + frac[21,26] >= 1
 cov_73_21: frac[3,21] + frac[5,21] + frac[16,21] + frac[17,21]
   + frac[19,21] + frac[20,21] + frac[21,21] >= 1
 cov_74_42: frac[5,42] + frac[6,42] + frac[16,42] + frac[17,42]
   + frac[19,42] + frac[20,42] + frac[21,42] >= 1
 cov_75_25: frac[2,25] + frac[3,25] + frac[16,25] >= 1
 cov_76_7: frac[0,7] + frac[5,7] + frac[17,7] + frac[19,7] + frac[20,7]
   + frac[21,7] >= 0.8
 cov_77_45: frac[5,45] + frac[6,45] + frac[16,45] + frac[17,45]
   + frac[19,45] + frac[20,45] + frac[21,45] >= 0.8
 cov_78_25: frac[0,25] + frac[1,25] + frac[18,25] + frac[19,25]
   + frac[20,25] + frac[21,25] >= 0.9
 cov_79_10: frac[0,10] + frac[7,10] + frac[18,10] + frac[19,10]
   + frac[20,10] + frac[21,10] >= 0.8
 cov_80_26: frac[3,26] + frac[6,26] + frac[16,26] >= 1
 cov_81_21: frac[4,21] + frac[6,21] + frac[16,21] + frac[17,21]
   + frac[19,21] + frac[20,21] + frac[21,21] >= 0.8
 cov_82_21: frac[1,21] + frac[4,21] + frac[17,21] + frac[18,21]
   + frac[20,21] >= 0.9
 cov_83_45: frac[1,45] + frac[3,45] + frac[16,45] + frac[18,45]
   + frac[19,45] + frac[20,45] + frac[21,45] >= 0.7
 cov_84_10: frac[3,10] + frac[4,10] + frac[16,10] + frac[17,10]
   + frac[19,10] + frac[20,10] + frac[21,10] >= 0.9
 cov_85_0: frac[3,0] + frac[6,0] + frac[16,0] >= 0.7
 cov_86_14: frac[4,14] + frac[7,14] + frac[17,14] + frac[18,14]
   + frac[20,14] >= 1
 cov_87_34: frac[3,34] + frac[5,34] + frac[16,34] + frac[17,34]
   + frac[19,34] + frac[20,34] + frac[21,34] >= 0.7
 cov_88_33: frac[11,33] + frac[12,33] + frac[17,33] + frac[18,33]
   + frac[20,33] >= 1
 cov_89_49: frac[8,49] + frac[15,49] + frac[16,49] + frac[19,49]
   >= 0.0174942857142859
 cov_90_20: frac[10,20] + frac[15,20] + frac[16,20] + frac[19,20] >= 0.7
 cov_91_37: frac[8,37] + frac[11,37] + frac[17,37] + frac[19,37]
   + frac[20,37] + frac[21,37] >= 0.7
 cov_92_24: frac[10,24] + frac[14,24] >= 0.9
 cov_93_41: frac[13,41] + frac[15,41] + frac[16,41] + frac[19,41] >= 0.8
 cov_94_41: frac[8,41] + frac[9,41] >= 1
 cov_95_41: frac[12,41] + frac[15,41] + frac[16,41] + frac[18,41]
   + frac[19,41] + frac[20,41] + frac[21,41] >= 1
 cov_96_13: frac[8,13] + frac[14,13] >= 0.0174942857142859
 cov_97_6: frac[11,6] + frac[13,6] + frac[17,6] + frac[19,6] + frac[20,6]
   + frac[21,6] >= 1
 cov_98_33: frac[9,33] + frac[12,33] + frac[18,33] + frac[19,33]
   + frac[20,33] + frac[21,33] >= 1
 cov_99_30: frac[8,30] + frac[11,30] + frac[17,30] + frac[19,30]
   + frac[20,30] + frac[21,30] >= 0.7
 cov_100_20: frac[8,20] + frac[10,20] >= 1
 cov_101_44: frac[9,44] + frac[11,44] + frac[17,44] + frac[19,44]
   + frac[20,44] + frac[21,44] >= 1
 cov_102_17: frac[12,17] + frac[14,17] + frac[18,17] + frac[19,17]
   + frac[20,17] + frac[21,17] >= 0.0174942857142859
 cov_103_4: frac[12,4] + frac[15,4] + frac[16,4] + frac[18,4] + frac[19,4]
   + frac[20,4] + frac[21,4] >= 0.8
 cov_104_30: frac[10,30] + frac[13,30] >= 1
 cov_105_24: frac[8,24] + frac[11,24] + frac[17,24] + frac[19,24]
   + frac[20,24] + frac[21,24] >= 1
 cov_106_41: frac[10,41] + frac[15,41] + frac[16,41] + frac[19,41] >= 0.7
 cov_107_20: frac[9,20] + frac[10,20] >= 0.9
 cov_108_20: frac[12,20] + frac[14,20] + frac[18,20] + frac[19,20]
   + frac[20,20] + frac[21,20] >= 0.7
 cov_109_24: frac[14,24] + frac[15,24] + frac[16,24] + frac[19,24] >= 0.7
 cov_110_37: frac[11,37] + frac[15,37] + frac[16,37] + frac[17,37]
   + frac[19,37] + frac[20,37] + frac[21,37] >= 1
 cov_111_8: frac[10,8] + frac[11,8] + frac[17,8] + frac[19,8] + frac[20,8]
   + frac[21,8] >= 0.0174942857142859
 cov_112_44: frac[13,44] + frac[15,44] + frac[16,44] + frac[19,44] >= 1
 cov_113_17: frac[8,17] + frac[12,17] + frac[18,17] + frac[19,17]
   + frac[20,17] + frac[21,17] >= 0.0174942857142859
 cov_114_37: frac[11,37] + frac[14,37] + frac[17,37] + frac[19,37]
   + frac[20,37] + frac[21,37] >= 0.7
 cov_115_20: frac[8,20] + frac[14,20] >= 0.7
 cov_116_37: frac[8,37] + frac[12,37] + frac[18,37] + frac[19,37]
   + frac[20,37] + frac[21,37] >= 0.7
 accuracy_20_25: - 768 frac[20,25] + mem[20,25] = 0
 accuracy_16_9: - 768 frac[16,9] + mem[16,9] = 0
 accuracy_19_4: - 768 frac[19,4] + mem[19,4] = 0
 accuracy_20_7: - 768 frac[20,7] + mem[20,7] = 0
 accuracy_18_19: - 768 frac[18,19] + mem[18,19] = 0
 accuracy_21_6: - 768 frac[21,6] + mem[21,6] = 0
 accuracy_17_46: - 768 frac[17,46] + mem[17,46] = 0
 accuracy_18_45: - 768 frac[18,45] + mem[18,45] = 0
 accuracy_12_17: - 509.68359375 frac[12,17] + mem[12,17] = 0
 accuracy_15_4: - 509.68359375 frac[15,4] + mem[15,4] = 0
 accuracy_20_32: - 768 frac[20,32] + mem[20,32] = 0
 accuracy_16_0: - 768 frac[16,0] + mem[16,0] = 0
 accuracy_21_37: - 768 frac[21,37] + mem[21,37] = 0
 accuracy_5_34: - 509.68359375 frac[5,34] + mem[5,34] = 0
 accuracy_20_14: - 768 frac[20,14] + mem[20,14] = 0
 accuracy_18_10: - 768 frac[18,10] + mem[18,10] = 0
 accuracy_21_15: - 768 frac[21,15] + mem[21,15] = 0
 accuracy_18_36: - 768 frac[18,36] + mem[18,36] = 0
 accuracy_1_21: - 509.68359375 frac[1,21] + mem[1,21] = 0
 accuracy_19_44: - 768 frac[19,44] + mem[19,44] = 0
 accuracy_20_47: - 768 frac[20,47] + mem[20,47] = 0
 accuracy_16_7: - 768 frac[16,7] + mem[16,7] = 0
 accuracy_21_46: - 768 frac[21,46] + mem[21,46] = 0
 accuracy_19_18: - 768 frac[19,18] + mem[19,18] = 0
 accuracy_17_6: - 768 frac[17,6] + mem[17,6] = 0
 accuracy_20_21: - 768 frac[20,21] + mem[20,21] = 0
 accuracy_18_5: - 768 frac[18,5] + mem[18,5] = 0
 accuracy_16_29: - 768 frac[16,29] + mem[16,29] = 0
 accuracy_21_8: - 768 frac[21,8] + mem[21,8] = 0
 accuracy_17_24: - 768 frac[17,24] + mem[17,24] = 0
 accuracy_12_41: - 509.68359375 frac[12,41] + mem[12,41] = 0
 accuracy_13_44: - 509.68359375 frac[13,44] + mem[13,44] = 0
 accuracy_9_20: - 509.68359375 frac[9,20] + mem[9,20] = 0
 accuracy_13_6: - 509.68359375 frac[13,6] + mem[13,6] = 0
 accuracy_5_10: - 509.68359375 frac[5,10] + mem[5,10] = 0
 accuracy_19_27: - 768 frac[19,27] + mem[19,27] = 0
 accuracy_16_20: - 768 frac[16,20] + mem[16,20] = 0
 accuracy_21_17: - 768 frac[21,17] + mem[21,17] = 0
 accuracy_19_1: - 768 frac[19,1] + mem[19,1] = 0
 accuracy_7_34: - 509.68359375 frac[7,34] + mem[7,34] = 0
 accuracy_18_22: - 768 frac[18,22] + mem[18,22] = 0
 accuracy_16_42: - 768 frac[16,42] + mem[16,42] = 0
 accuracy_17_43: - 768 frac[17,43] + mem[17,43] = 0
 accuracy_18_48: - 768 frac[18,48] + mem[18,48] = 0
 accuracy_12_20: - 509.68359375 frac[12,20] + mem[12,20] = 0
 accuracy_1_25: - 509.68359375 frac[1,25] + mem[1,25] = 0
 accuracy_21_48: - 768 frac[21,48] + mem[21,48] = 0
 accuracy_19_32: - 768 frac[19,32] + mem[19,32] = 0
 accuracy_20_27: - 768 frac[20,27] + mem[20,27] = 0
 accuracy_16_11: - 768 frac[16,11] + mem[16,11] = 0
 accuracy_21_26: - 768 frac[21,26] + mem[21,26] = 0
 accuracy_4_38: - 509.68359375 frac[4,38] + mem[4,38] = 0
 accuracy_19_6: - 768 frac[19,6] + mem[19,6] = 0
 accuracy_17_10: - 768 frac[17,10] + mem[17,10] = 0
 accuracy_20_1: - 768 frac[20,1] + mem[20,1] = 0
 accuracy_18_17: - 768 frac[18,17] + mem[18,17] = 0
 accuracy_21_4: - 768 frac[21,4] + mem[21,4] = 0
 accuracy_17_44: - 768 frac[17,44] + mem[17,44] = 0
 accuracy_18_43: - 768 frac[18,43] + mem[18,43] = 0
 accuracy_15_24: - 509.68359375 frac[15,24] + mem[15,24] = 0
 accuracy_11_8: - 509.68359375 frac[11,8] + mem[11,8] = 0
 accuracy_2_25: - 509.68359375 frac[2,25] + mem[2,25] = 0
 accuracy_15_44: - 509.68359375 frac[15,44] + mem[15,44] = 0
 accuracy_19_41: - 768 frac[19,41] + mem[19,41] = 0
 accuracy_20_34: - 768 frac[20,34] + mem[20,34] = 0
 accuracy_3_42: - 509.68359375 frac[3,42] + mem[3,42] = 0
 accuracy_16_2: - 768 frac[16,2] + mem[16,2] = 0
 accuracy_6_21: - 509.68359375 frac[6,21] + mem[6,21] = 0
 accuracy_19_15: - 768 frac[19,15] + mem[19,15] = 0
 accuracy_2_45: - 509.68359375 frac[2,45] + mem[2,45] = 0
 accuracy_17_3: - 768 frac[17,3] + mem[17,3] = 0
 accuracy_20_8: - 768 frac[20,8] + mem[20,8] = 0
 accuracy_16_24: - 768 frac[16,24] + mem[16,24] = 0
 accuracy_17_37: - 768 frac[17,37] + mem[17,37] = 0
 accuracy_18_34: - 768 frac[18,34] + mem[18,34] = 0
 accuracy_8_20: - 509.68359375 frac[8,20] + mem[8,20] = 0
 accuracy_4_14: - 509.68359375 frac[4,14] + mem[4,14] = 0
 accuracy_19_46: - 768 frac[19,46] + mem[19,46] = 0
 accuracy_0_38: - 509.68359375 frac[0,38] + mem[0,38] = 0
 accuracy_20_41: - 768 frac[20,41] + mem[20,41] = 0
 accuracy_21_44: - 768 frac[21,44] + mem[21,44] = 0
 accuracy_19_20: - 768 frac[19,20] + mem[19,20] = 0
 accuracy_7_9: - 509.68359375 frac[7,9] + mem[7,9] = 0
 accuracy_20_23: - 768 frac[20,23] + mem[20,23] = 0
 accuracy_18_3: - 768 frac[18,3] + mem[18,3] = 0
 accuracy_16_31: - 768 frac[16,31] + mem[16,31] = 0
 accuracy_21_22: - 768 frac[21,22] + mem[21,22] = 0
 accuracy_17_30: - 768 frac[17,30] + mem[17,30] = 0
 accuracy_18_29: - 768 frac[18,29] + mem[18,29] = 0
 accuracy_17_48: - 768 frac[17,48] + mem[17,48] = 0
 accuracy_11_6: - 509.68359375 frac[11,6] + mem[11,6] = 0
 accuracy_14_17: - 509.68359375 frac[14,17] + mem[14,17] = 0
 accuracy_15_20: - 509.68359375 frac[15,20] + mem[15,20] = 0
 accuracy_0_7: - 509.68359375 frac[0,7] + mem[0,7] = 0
 accuracy_13_30: - 509.68359375 frac[13,30] + mem[13,30] = 0
 accuracy_20_48: - 768 frac[20,48] + mem[20,48] = 0
 accuracy_6_7: - 509.68359375 frac[6,7] + mem[6,7] = 0
 accuracy_19_29: - 768 frac[19,29] + mem[19,29] = 0
 accuracy_20_30: - 768 frac[20,30] + mem[20,30] = 0
 accuracy_16_22: - 768 frac[16,22] + mem[16,22] = 0
 accuracy_21_31: - 768 frac[21,31] + mem[21,31] = 0
 accuracy_19_3: - 768 frac[19,3] + mem[19,3] = 0
 accuracy_17_23: - 768 frac[17,23] + mem[17,23] = 0
 accuracy_20_4: - 768 frac[20,4] + mem[20,4] = 0
 accuracy_18_20: - 768 frac[18,20] + mem[18,20] = 0
 accuracy_16_44: - 768 frac[16,44] + mem[16,44] = 0
 accuracy_18_46: - 768 frac[18,46] + mem[18,46] = 0
 accuracy_14_24: - 509.68359375 frac[14,24] + mem[14,24] = 0
 accuracy_10_24: - 509.68359375 frac[10,24] + mem[10,24] = 0
 accuracy_19_34: - 768 frac[19,34] + mem[19,34] = 0
 accuracy_0_42: - 509.68359375 frac[0,42] + mem[0,42] = 0
 accuracy_20_37: - 768 frac[20,37] + mem[20,37] = 0
 accuracy_21_24: - 768 frac[21,24] + mem[21,24] = 0
 accuracy_19_8: - 768 frac[19,8] + mem[19,8] = 0
 accuracy_17_8: - 768 frac[17,8] + mem[17,8] = 0
 accuracy_7_21: - 509.68359375 frac[7,21] + mem[7,21] = 0
 accuracy_20_3: - 768 frac[20,3] + mem[20,3] = 0
 accuracy_18_15: - 768 frac[18,15] + mem[18,15] = 0
 accuracy_16_35: - 768 frac[16,35] + mem[16,35] = 0
 accuracy_6_42: - 509.68359375 frac[6,42] + mem[6,42] = 0
 accuracy_21_2: - 768 frac[21,2] + mem[21,2] = 0
 accuracy_17_34: - 768 frac[17,34] + mem[17,34] = 0
 accuracy_18_41: - 768 frac[18,41] + mem[18,41] = 0
 accuracy_20_44: - 768 frac[20,44] + mem[20,44] = 0
 accuracy_16_4: - 768 frac[16,4] + mem[16,4] = 0
 accuracy_21_33: - 768 frac[21,33] + mem[21,33] = 0
 accuracy_19_17: - 768 frac[19,17] + mem[19,17] = 0
 accuracy_17_1: - 768 frac[17,1] + mem[17,1] = 0
 accuracy_20_10: - 768 frac[20,10] + mem[20,10] = 0
 accuracy_16_26: - 768 frac[16,26] + mem[16,26] = 0
 accuracy_6_45: - 509.68359375 frac[6,45] + mem[6,45] = 0
 accuracy_21_11: - 768 frac[21,11] + mem[21,11] = 0
 accuracy_18_32: - 768 frac[18,32] + mem[18,32] = 0
 accuracy_12_4: - 509.68359375 frac[12,4] + mem[12,4] = 0
 accuracy_10_20: - 509.68359375 frac[10,20] + mem[10,20] = 0
 accuracy_19_48: - 768 frac[19,48] + mem[19,48] = 0
 accuracy_20_43: - 768 frac[20,43] + mem[20,43] = 0
 accuracy_3_21: - 509.68359375 frac[3,21] + mem[3,21] = 0
 accuracy_21_42: - 768 frac[21,42] + mem[21,42] = 0
 accuracy_19_22: - 768 frac[19,22] + mem[19,22] = 0
 accuracy_2_34: - 509.68359375 frac[2,34] + mem[2,34] = 0
 accuracy_20_17: - 768 frac[20,17] + mem[20,17] = 0
 accuracy_9_44: - 509.68359375 frac[9,44] + mem[9,44] = 0
 accuracy_21_20: - 768 frac[21,20] + mem[21,20] = 0
 accuracy_18_27: - 768 frac[18,27] + mem[18,27] = 0
 accuracy_12_37: - 509.68359375 frac[12,37] + mem[12,37] = 0
 accuracy_11_30: - 509.68359375 frac[11,30] + mem[11,30] = 0
 accuracy_0_25: - 509.68359375 frac[0,25] + mem[0,25] = 0
 accuracy_3_26: - 509.68359375 frac[3,26] + mem[3,26] = 0
 accuracy_19_31: - 768 frac[19,31] + mem[19,31] = 0
 accuracy_20_24: - 768 frac[20,24] + mem[20,24] = 0
 accuracy_6_31: - 509.68359375 frac[6,31] + mem[6,31] = 0
 accuracy_21_29: - 768 frac[21,29] + mem[21,29] = 0
 accuracy_17_21: - 768 frac[17,21] + mem[17,21] = 0
 accuracy_20_6: - 768 frac[20,6] + mem[20,6] = 0
 accuracy_18_18: - 768 frac[18,18] + mem[18,18] = 0
 accuracy_16_46: - 768 frac[16,46] + mem[16,46] = 0
 accuracy_21_7: - 768 frac[21,7] + mem[21,7] = 0
 accuracy_17_47: - 768 frac[17,47] + mem[17,47] = 0
 accuracy_19_36: - 768 frac[19,36] + mem[19,36] = 0
 accuracy_16_15: - 768 frac[16,15] + mem[16,15] = 0
 accuracy_21_38: - 768 frac[21,38] + mem[21,38] = 0
 accuracy_19_10: - 768 frac[19,10] + mem[19,10] = 0
 accuracy_17_14: - 768 frac[17,14] + mem[17,14] = 0
 accuracy_8_37: - 509.68359375 frac[8,37] + mem[8,37] = 0
 accuracy_16_37: - 768 frac[16,37] + mem[16,37] = 0
 accuracy_7_45: - 509.68359375 frac[7,45] + mem[7,45] = 0
 accuracy_8_17: - 509.68359375 frac[8,17] + mem[8,17] = 0
 accuracy_2_21: - 509.68359375 frac[2,21] + mem[2,21] = 0
 accuracy_19_45: - 768 frac[19,45] + mem[19,45] = 0
 accuracy_20_46: - 768 frac[20,46] + mem[20,46] = 0
 accuracy_6_9: - 509.68359375 frac[6,9] + mem[6,9] = 0
 accuracy_19_19: - 768 frac[19,19] + mem[19,19] = 0
 accuracy_17_7: - 768 frac[17,7] + mem[17,7] = 0
 accuracy_20_20: - 768 frac[20,20] + mem[20,20] = 0
 accuracy_9_41: - 509.68359375 frac[9,41] + mem[9,41] = 0
 accuracy_18_4: - 768 frac[18,4] + mem[18,4] = 0
 accuracy_16_28: - 768 frac[16,28] + mem[16,28] = 0
 accuracy_21_9: - 768 frac[21,9] + mem[21,9] = 0
 accuracy_17_25: - 768 frac[17,25] + mem[17,25] = 0
 accuracy_7_42: - 509.68359375 frac[7,42] + mem[7,42] = 0
 accuracy_10_8: - 509.68359375 frac[10,8] + mem[10,8] = 0
 accuracy_8_24: - 509.68359375 frac[8,24] + mem[8,24] = 0
 accuracy_15_49: - 509.68359375 frac[15,49] + mem[15,49] = 0
 accuracy_4_10: - 509.68359375 frac[4,10] + mem[4,10] = 0
 accuracy_6_0: - 509.68359375 frac[6,0] + mem[6,0] = 0
 accuracy_21_40: - 768 frac[21,40] + mem[21,40] = 0
 accuracy_19_24: - 768 frac[19,24] + mem[19,24] = 0
 accuracy_7_5: - 509.68359375 frac[7,5] + mem[7,5] = 0
 accuracy_5_21: - 509.68359375 frac[5,21] + mem[5,21] = 0
 accuracy_20_19: - 768 frac[20,19] + mem[20,19] = 0
 accuracy_16_19: - 768 frac[16,19] + mem[16,19] = 0
 accuracy_6_26: - 509.68359375 frac[6,26] + mem[6,26] = 0
 accuracy_21_18: - 768 frac[21,18] + mem[21,18] = 0
 accuracy_10_41: - 509.68359375 frac[10,41] + mem[10,41] = 0
 accuracy_17_18: - 768 frac[17,18] + mem[17,18] = 0
 accuracy_18_25: - 768 frac[18,25] + mem[18,25] = 0
 accuracy_16_41: - 768 frac[16,41] + mem[16,41] = 0
 accuracy_1_26: - 509.68359375 frac[1,26] + mem[1,26] = 0
 accuracy_4_31: - 509.68359375 frac[4,31] + mem[4,31] = 0
 accuracy_19_33: - 768 frac[19,33] + mem[19,33] = 0
 accuracy_20_26: - 768 frac[20,26] + mem[20,26] = 0
 accuracy_3_34: - 509.68359375 frac[3,34] + mem[3,34] = 0
 accuracy_16_10: - 768 frac[16,10] + mem[16,10] = 0
 accuracy_21_27: - 768 frac[21,27] + mem[21,27] = 0
 accuracy_19_7: - 768 frac[19,7] + mem[19,7] = 0
 accuracy_18_16: - 768 frac[18,16] + mem[18,16] = 0
 accuracy_17_45: - 768 frac[17,45] + mem[17,45] = 0
 accuracy_18_42: - 768 frac[18,42] + mem[18,42] = 0
 accuracy_19_38: - 768 frac[19,38] + mem[19,38] = 0
 accuracy_5_7: - 509.68359375 frac[5,7] + mem[5,7] = 0
 accuracy_20_33: - 768 frac[20,33] + mem[20,33] = 0
 accuracy_21_36: - 768 frac[21,36] + mem[21,36] = 0
 accuracy_19_12: - 768 frac[19,12] + mem[19,12] = 0
 accuracy_17_12: - 768 frac[17,12] + mem[17,12] = 0
 accuracy_20_15: - 768 frac[20,15] + mem[20,15] = 0
 accuracy_18_11: - 768 frac[18,11] + mem[18,11] = 0
 accuracy_16_39: - 768 frac[16,39] + mem[16,39] = 0
 accuracy_17_38: - 768 frac[17,38] + mem[17,38] = 0
 accuracy_8_13: - 509.68359375 frac[8,13] + mem[8,13] = 0
 accuracy_18_37: - 768 frac[18,37] + mem[18,37] = 0
 accuracy_3_10: - 509.68359375 frac[3,10] + mem[3,10] = 0
 accuracy_20_40: - 768 frac[20,40] + mem[20,40] = 0
 accuracy_21_45: - 768 frac[21,45] + mem[21,45] = 0
 accuracy_19_21: - 768 frac[19,21] + mem[19,21] = 0
 accuracy_7_14: - 509.68359375 frac[7,14] + mem[7,14] = 0
 accuracy_5_42: - 509.68359375 frac[5,42] + mem[5,42] = 0
 accuracy_20_22: - 768 frac[20,22] + mem[20,22] = 0
 accuracy_18_2: - 768 frac[18,2] + mem[18,2] = 0
 accuracy_21_23: - 768 frac[21,23] + mem[21,23] = 0
 accuracy_17_31: - 768 frac[17,31] + mem[17,31] = 0
 accuracy_11_33: - 509.68359375 frac[11,33] + mem[11,33] = 0
 accuracy_3_25: - 509.68359375 frac[3,25] + mem[3,25] = 0
 accuracy_19_26: - 768 frac[19,26] + mem[19,26] = 0
 accuracy_7_7: - 509.68359375 frac[7,7] + mem[7,7] = 0
 accuracy_20_29: - 768 frac[20,29] + mem[20,29] = 0
 accuracy_16_21: - 768 frac[16,21] + mem[16,21] = 0
 accuracy_21_16: - 768 frac[21,16] + mem[21,16] = 0
 accuracy_17_16: - 768 frac[17,16] + mem[17,16] = 0
 accuracy_18_23: - 768 frac[18,23] + mem[18,23] = 0
 accuracy_12_33: - 509.68359375 frac[12,33] + mem[12,33] = 0
 accuracy_17_42: - 768 frac[17,42] + mem[17,42] = 0
 accuracy_4_25: - 509.68359375 frac[4,25] + mem[4,25] = 0
 accuracy_19_35: - 768 frac[19,35] + mem[19,35] = 0
 accuracy_20_36: - 768 frac[20,36] + mem[20,36] = 0
 accuracy_16_12: - 768 frac[16,12] + mem[16,12] = 0
 accuracy_21_25: - 768 frac[21,25] + mem[21,25] = 0
 accuracy_19_9: - 768 frac[19,9] + mem[19,9] = 0
 accuracy_20_2: - 768 frac[20,2] + mem[20,2] = 0
 accuracy_18_14: - 768 frac[18,14] + mem[18,14] = 0
 accuracy_16_34: - 768 frac[16,34] + mem[16,34] = 0
 accuracy_21_3: - 768 frac[21,3] + mem[21,3] = 0
 accuracy_18_40: - 768 frac[18,40] + mem[18,40] = 0
 accuracy_0_10: - 509.68359375 frac[0,10] + mem[0,10] = 0
 accuracy_19_40: - 768 frac[19,40] + mem[19,40] = 0
 accuracy_3_45: - 509.68359375 frac[3,45] + mem[3,45] = 0
 accuracy_1_45: - 509.68359375 frac[1,45] + mem[1,45] = 0
 accuracy_16_3: - 768 frac[16,3] + mem[16,3] = 0
 accuracy_6_10: - 509.68359375 frac[6,10] + mem[6,10] = 0
 accuracy_21_34: - 768 frac[21,34] + mem[21,34] = 0
 accuracy_17_2: - 768 frac[17,2] + mem[17,2] = 0
 accuracy_8_41: - 509.68359375 frac[8,41] + mem[8,41] = 0
 accuracy_20_9: - 768 frac[20,9] + mem[20,9] = 0
 accuracy_18_9: - 768 frac[18,9] + mem[18,9] = 0
 accuracy_16_25: - 768 frac[16,25] + mem[16,25] = 0
 accuracy_21_12: - 768 frac[21,12] + mem[21,12] = 0
 accuracy_19_49: - 768 frac[19,49] + mem[19,49] = 0
 accuracy_20_42: - 768 frac[20,42] + mem[20,42] = 0
 accuracy_4_21: - 509.68359375 frac[4,21] + mem[4,21] = 0
 accuracy_19_23: - 768 frac[19,23] + mem[19,23] = 0
 accuracy_20_16: - 768 frac[20,16] + mem[20,16] = 0
 accuracy_16_16: - 768 frac[16,16] + mem[16,16] = 0
 accuracy_21_21: - 768 frac[21,21] + mem[21,21] = 0
 accuracy_17_29: - 768 frac[17,29] + mem[17,29] = 0
 accuracy_18_26: - 768 frac[18,26] + mem[18,26] = 0
 accuracy_13_41: - 509.68359375 frac[13,41] + mem[13,41] = 0
 accuracy_10_30: - 509.68359375 frac[10,30] + mem[10,30] = 0
 accuracy_1_5: - 509.68359375 frac[1,5] + mem[1,5] = 0
 accuracy_19_28: - 768 frac[19,28] + mem[19,28] = 0
 accuracy_20_31: - 768 frac[20,31] + mem[20,31] = 0
 accuracy_16_23: - 768 frac[16,23] + mem[16,23] = 0
 accuracy_21_30: - 768 frac[21,30] + mem[21,30] = 0
 accuracy_19_2: - 768 frac[19,2] + mem[19,2] = 0
 accuracy_17_22: - 768 frac[17,22] + mem[17,22] = 0
 accuracy_18_21: - 768 frac[18,21] + mem[18,21] = 0
 accuracy_16_45: - 768 frac[16,45] + mem[16,45] = 0
 accuracy_17_40: - 768 frac[17,40] + mem[17,40] = 0
 accuracy_18_47: - 768 frac[18,47] + mem[18,47] = 0
 accuracy_14_37: - 509.68359375 frac[14,37] + mem[14,37] = 0
 accuracy_3_0: - 509.68359375 frac[3,0] + mem[3,0] = 0
 accuracy_19_37: - 768 frac[19,37] + mem[19,37] = 0
 accuracy_20_38: - 768 frac[20,38] + mem[20,38] = 0
 accuracy_19_11: - 768 frac[19,11] + mem[19,11] = 0
 accuracy_17_15: - 768 frac[17,15] + mem[17,15] = 0
 accuracy_20_12: - 768 frac[20,12] + mem[20,12] = 0
 accuracy_18_12: - 768 frac[18,12] + mem[18,12] = 0
 accuracy_9_33: - 509.68359375 frac[9,33] + mem[9,33] = 0
 accuracy_21_1: - 768 frac[21,1] + mem[21,1] = 0
 accuracy_17_33: - 768 frac[17,33] + mem[17,33] = 0
 accuracy_15_41: - 509.68359375 frac[15,41] + mem[15,41] = 0
 accuracy_19_42: - 768 frac[19,42] + mem[19,42] = 0
 accuracy_20_45: - 768 frac[20,45] + mem[20,45] = 0
 accuracy_21_32: - 768 frac[21,32] + mem[21,32] = 0
 accuracy_19_16: - 768 frac[19,16] + mem[19,16] = 0
 accuracy_5_45: - 509.68359375 frac[5,45] + mem[5,45] = 0
 accuracy_20_11: - 768 frac[20,11] + mem[20,11] = 0
 accuracy_18_7: - 768 frac[18,7] + mem[18,7] = 0
 accuracy_21_10: - 768 frac[21,10] + mem[21,10] = 0
 accuracy_8_49: - 509.68359375 frac[8,49] + mem[8,49] = 0
 accuracy_11_44: - 509.68359375 frac[11,44] + mem[11,44] = 0
 accuracy_18_33: - 768 frac[18,33] + mem[18,33] = 0
 accuracy_16_49: - 768 frac[16,49] + mem[16,49] = 0
 accuracy_14_13: - 509.68359375 frac[14,13] + mem[14,13] = 0
 accuracy_11_24: - 509.68359375 frac[11,24] + mem[11,24] = 0
 accuracy_21_41: - 768 frac[21,41] + mem[21,41] = 0
 accuracy_19_25: - 768 frac[19,25] + mem[19,25] = 0
 accuracy_7_10: - 509.68359375 frac[7,10] + mem[7,10] = 0
 accuracy_20_18: - 768 frac[20,18] + mem[20,18] = 0
 accuracy_16_18: - 768 frac[16,18] + mem[16,18] = 0
 accuracy_17_19: - 768 frac[17,19] + mem[17,19] = 0
 accuracy_11_37: - 509.68359375 frac[11,37] + mem[11,37] = 0
 accuracy_8_30: - 509.68359375 frac[8,30] + mem[8,30] = 0
 accuracy_14_20: - 509.68359375 frac[14,20] + mem[14,20] = 0
 accuracy_0_26: - 509.68359375 frac[0,26] + mem[0,26] = 0
 accuracy_15_37: - 509.68359375 frac[15,37] + mem[15,37] = 0
 accuracy_19_30: - 768 frac[19,30] + mem[19,30] = 0
 mem_tot_CPU12: - mem[0,38] - mem[0,7] - mem[0,42] - mem[0,25] - mem[0,10]
   - mem[0,26] + mem_tot_CPU12 = 0
 R444: - 3.05176e-05 mem_tot_CPU12 + normalized_mem_tot_CPU12 = 0
 rows_tot_CPU12: - 3 frac[0,38] - 3 frac[0,7] - 3 frac[0,42] - 3 frac[0,25]
   - 3 frac[0,10] - 3 frac[0,26] + rows_tot_CPU12 = 0
 R446: - 0.0833333333333333 rows_tot_CPU12 + normalized_rows_tot_CPU12 = 0
 mem_tot_CPU26: - mem[1,21] - mem[1,25] - mem[1,26] - mem[1,45] - mem[1,5]
   + mem_tot_CPU26 = 0
 R448: - 3.05176e-05 mem_tot_CPU26 + normalized_mem_tot_CPU26 = 0
 rows_tot_CPU26: - 3 frac[1,21] - 3 frac[1,25] - 3 frac[1,26]
   - 3 frac[1,45] - 3 frac[1,5] + rows_tot_CPU26 = 0
 R450: - 0.0833333333333333 rows_tot_CPU26 + normalized_rows_tot_CPU26 = 0
 mem_tot_CPU1: - mem[2,25] - mem[2,45] - mem[2,34] - mem[2,21]
   + mem_tot_CPU1 = 0
 R452: - 3.05176e-05 mem_tot_CPU1 + normalized_mem_tot_CPU1 = 0
 rows_tot_CPU1: - 3 frac[2,25] - 3 frac[2,45] - 3 frac[2,34] - 3 frac[2,21]
   + rows_tot_CPU1 = 0
 R454: - 0.0833333333333333 rows_tot_CPU1 + normalized_rows_tot_CPU1 = 0
 mem_tot_CPU4: - mem[3,42] - mem[3,21] - mem[3,26] - mem[3,34] - mem[3,10]
   - mem[3,25] - mem[3,45] - mem[3,0] + mem_tot_CPU4 = 0
 R456: - 3.05176e-05 mem_tot_CPU4 + normalized_mem_tot_CPU4 = 0
 rows_tot_CPU4: - 3 frac[3,42] - 3 frac[3,21] - 3 frac[3,26] - 3 frac[3,34]
   - 3 frac[3,10] - 3 frac[3,25] - 3 frac[3,45] - 3 frac[3,0]
   + rows_tot_CPU4 = 0
 R458: - 0.0833333333333333 rows_tot_CPU4 + normalized_rows_tot_CPU4 = 0
 mem_tot_CPU24: - mem[4,38] - mem[4,14] - mem[4,10] - mem[4,31] - mem[4,25]
   - mem[4,21] + mem_tot_CPU24 = 0
 R460: - 3.05176e-05 mem_tot_CPU24 + normalized_mem_tot_CPU24 = 0
 rows_tot_CPU24: - 3 frac[4,38] - 3 frac[4,14] - 3 frac[4,10]
   - 3 frac[4,31] - 3 frac[4,25] - 3 frac[4,21] + rows_tot_CPU24 = 0
 R462: - 0.0833333333333333 rows_tot_CPU24 + normalized_rows_tot_CPU24 = 0
 mem_tot_CPU21: - mem[5,34] - mem[5,10] - mem[5,21] - mem[5,7] - mem[5,42]
   - mem[5,45] + mem_tot_CPU21 = 0
 R464: - 3.05176e-05 mem_tot_CPU21 + normalized_mem_tot_CPU21 = 0
 rows_tot_CPU21: - 3 frac[5,34] - 3 frac[5,10] - 3 frac[5,21] - 3 frac[5,7]
   - 3 frac[5,42] - 3 frac[5,45] + rows_tot_CPU21 = 0
 R466: - 0.0833333333333333 rows_tot_CPU21 + normalized_rows_tot_CPU21 = 0
 mem_tot_CPU2: - mem[6,21] - mem[6,7] - mem[6,42] - mem[6,45] - mem[6,31]
   - mem[6,9] - mem[6,0] - mem[6,26] - mem[6,10] + mem_tot_CPU2 = 0
 R468: - 3.05176e-05 mem_tot_CPU2 + normalized_mem_tot_CPU2 = 0
 rows_tot_CPU2: - 3 frac[6,21] - 3 frac[6,7] - 3 frac[6,42] - 3 frac[6,45]
   - 3 frac[6,31] - 3 frac[6,9] - 3 frac[6,0] - 3 frac[6,26] - 3 frac[6,10]
   + rows_tot_CPU2 = 0
 R470: - 0.0833333333333333 rows_tot_CPU2 + normalized_rows_tot_CPU2 = 0
 mem_tot_CPU32: - mem[7,34] - mem[7,9] - mem[7,21] - mem[7,45] - mem[7,42]
   - mem[7,5] - mem[7,14] - mem[7,7] - mem[7,10] + mem_tot_CPU32 = 0
 R472: - 3.05176e-05 mem_tot_CPU32 + normalized_mem_tot_CPU32 = 0
 rows_tot_CPU32: - 3 frac[7,34] - 3 frac[7,9] - 3 frac[7,21] - 3 frac[7,45]
   - 3 frac[7,42] - 3 frac[7,5] - 3 frac[7,14] - 3 frac[7,7] - 3 frac[7,10]
   + rows_tot_CPU32 = 0
 R474: - 0.0833333333333333 rows_tot_CPU32 + normalized_rows_tot_CPU32 = 0
 mem_tot_CPU11: - mem[8,20] - mem[8,37] - mem[8,17] - mem[8,24] - mem[8,13]
   - mem[8,41] - mem[8,49] - mem[8,30] + mem_tot_CPU11 = 0
 R476: - 3.05176e-05 mem_tot_CPU11 + normalized_mem_tot_CPU11 = 0
 rows_tot_CPU11: - 3 frac[8,20] - 3 frac[8,37] - 3 frac[8,17]
   - 3 frac[8,24] - 3 frac[8,13] - 3 frac[8,41] - 3 frac[8,49]
   - 3 frac[8,30] + rows_tot_CPU11 = 0
 R478: - 0.0833333333333333 rows_tot_CPU11 + normalized_rows_tot_CPU11 = 0
 mem_tot_CPU14: - mem[9,20] - mem[9,44] - mem[9,41] - mem[9,33]
   + mem_tot_CPU14 = 0
 R480: - 3.05176e-05 mem_tot_CPU14 + normalized_mem_tot_CPU14 = 0
 rows_tot_CPU14: - 3 frac[9,20] - 3 frac[9,44] - 3 frac[9,41]
   - 3 frac[9,33] + rows_tot_CPU14 = 0
 R482: - 0.0833333333333333 rows_tot_CPU14 + normalized_rows_tot_CPU14 = 0
 mem_tot_CPU10: - mem[10,24] - mem[10,20] - mem[10,8] - mem[10,41]
   - mem[10,30] + mem_tot_CPU10 = 0
 R484: - 3.05176e-05 mem_tot_CPU10 + normalized_mem_tot_CPU10 = 0
 rows_tot_CPU10: - 3 frac[10,24] - 3 frac[10,20] - 3 frac[10,8]
   - 3 frac[10,41] - 3 frac[10,30] + rows_tot_CPU10 = 0
 R486: - 0.0833333333333333 rows_tot_CPU10 + normalized_rows_tot_CPU10 = 0
 mem_tot_CPU23: - mem[11,8] - mem[11,6] - mem[11,30] - mem[11,33]
   - mem[11,44] - mem[11,24] - mem[11,37] + mem_tot_CPU23 = 0
 R488: - 3.05176e-05 mem_tot_CPU23 + normalized_mem_tot_CPU23 = 0
 rows_tot_CPU23: - 3 frac[11,8] - 3 frac[11,6] - 3 frac[11,30]
   - 3 frac[11,33] - 3 frac[11,44] - 3 frac[11,24] - 3 frac[11,37]
   + rows_tot_CPU23 = 0
 R490: - 0.0833333333333333 rows_tot_CPU23 + normalized_rows_tot_CPU23 = 0
 mem_tot_CPU31: - mem[12,17] - mem[12,41] - mem[12,20] - mem[12,4]
   - mem[12,37] - mem[12,33] + mem_tot_CPU31 = 0
 R492: - 3.05176e-05 mem_tot_CPU31 + normalized_mem_tot_CPU31 = 0
 rows_tot_CPU31: - 3 frac[12,17] - 3 frac[12,41] - 3 frac[12,20]
   - 3 frac[12,4] - 3 frac[12,37] - 3 frac[12,33] + rows_tot_CPU31 = 0
 R494: - 0.0833333333333333 rows_tot_CPU31 + normalized_rows_tot_CPU31 = 0
 mem_tot_CPU13: - mem[13,44] - mem[13,6] - mem[13,30] - mem[13,41]
   + mem_tot_CPU13 = 0
 R496: - 3.05176e-05 mem_tot_CPU13 + normalized_mem_tot_CPU13 = 0
 rows_tot_CPU13: - 3 frac[13,44] - 3 frac[13,6] - 3 frac[13,30]
   - 3 frac[13,41] + rows_tot_CPU13 = 0
 R498: - 0.0833333333333333 rows_tot_CPU13 + normalized_rows_tot_CPU13 = 0
 mem_tot_CPU16: - mem[14,17] - mem[14,24] - mem[14,37] - mem[14,13]
   - mem[14,20] + mem_tot_CPU16 = 0
 R500: - 3.05176e-05 mem_tot_CPU16 + normalized_mem_tot_CPU16 = 0
 rows_tot_CPU16: - 3 frac[14,17] - 3 frac[14,24] - 3 frac[14,37]
   - 3 frac[14,13] - 3 frac[14,20] + rows_tot_CPU16 = 0
 R502: - 0.0833333333333333 rows_tot_CPU16 + normalized_rows_tot_CPU16 = 0
 mem_tot_CPU5: - mem[15,4] - mem[15,24] - mem[15,44] - mem[15,20]
   - mem[15,49] - mem[15,41] - mem[15,37] + mem_tot_CPU5 = 0
 R504: - 3.05176e-05 mem_tot_CPU5 + normalized_mem_tot_CPU5 = 0
 rows_tot_CPU5: - 3 frac[15,4] - 3 frac[15,24] - 3 frac[15,44]
   - 3 frac[15,20] - 3 frac[15,49] - 3 frac[15,41] - 3 frac[15,37]
   + rows_tot_CPU5 = 0
 R506: - 0.0833333333333333 rows_tot_CPU5 + normalized_rows_tot_CPU5 = 0
 mem_tot_tor_P41: - mem[16,9] - mem[16,0] - mem[16,7] - mem[16,29]
   - mem[16,20] - mem[16,42] - mem[16,11] - mem[16,2] - mem[16,24]
   - mem[16,31] - mem[16,22] - mem[16,44] - mem[16,35] - mem[16,4]
   - mem[16,26] - mem[16,46] - mem[16,15] - mem[16,37] - mem[16,28]
   - mem[16,19] - mem[16,41] - mem[16,10] - mem[16,39] - mem[16,21]
   - mem[16,12] - mem[16,34] - mem[16,3] - mem[16,25] - mem[16,16]
   - mem[16,23] - mem[16,45] - mem[16,49] - mem[16,18] + mem_tot_tor_P41
   = 0
 R508: - 1.6276041666666666e-04 mem_tot_tor_P41
   + normalized_mem_tot_tor_P41 = 0
 rows_tot_tor_P41: - 3 frac[16,9] - 3 frac[16,0] - 3 frac[16,7]
   - 3 frac[16,29] - 3 frac[16,20] - 3 frac[16,42] - 3 frac[16,11]
   - 3 frac[16,2] - 3 frac[16,24] - 3 frac[16,31] - 3 frac[16,22]
   - 3 frac[16,44] - 3 frac[16,35] - 3 frac[16,4] - 3 frac[16,26]
   - 3 frac[16,46] - 3 frac[16,15] - 3 frac[16,37] - 3 frac[16,28]
   - 3 frac[16,19] - 3 frac[16,41] - 3 frac[16,10] - 3 frac[16,39]
   - 3 frac[16,21] - 3 frac[16,12] - 3 frac[16,34] - 3 frac[16,3]
   - 3 frac[16,25] - 3 frac[16,16] - 3 frac[16,23] - 3 frac[16,45]
   - 3 frac[16,49] - 3 frac[16,18] + rows_tot_tor_P41 = 0
 R510: - 0.0208333333333333 rows_tot_tor_P41 + normalized_rows_tot_tor_P41
   = 0
 mem_tot_tor_P43: - mem[17,46] - mem[17,6] - mem[17,24] - mem[17,43]
   - mem[17,10] - mem[17,44] - mem[17,3] - mem[17,37] - mem[17,30]
   - mem[17,48] - mem[17,23] - mem[17,8] - mem[17,34] - mem[17,1]
   - mem[17,21] - mem[17,47] - mem[17,14] - mem[17,7] - mem[17,25]
   - mem[17,18] - mem[17,45] - mem[17,12] - mem[17,38] - mem[17,31]
   - mem[17,16] - mem[17,42] - mem[17,2] - mem[17,29] - mem[17,22]
   - mem[17,40] - mem[17,15] - mem[17,33] - mem[17,19] + mem_tot_tor_P43
   = 0
 R512: - 1.6276041666666666e-04 mem_tot_tor_P43
   + normalized_mem_tot_tor_P43 = 0
 rows_tot_tor_P43: - 3 frac[17,46] - 3 frac[17,6] - 3 frac[17,24]
   - 3 frac[17,43] - 3 frac[17,10] - 3 frac[17,44] - 3 frac[17,3]
   - 3 frac[17,37] - 3 frac[17,30] - 3 frac[17,48] - 3 frac[17,23]
   - 3 frac[17,8] - 3 frac[17,34] - 3 frac[17,1] - 3 frac[17,21]
   - 3 frac[17,47] - 3 frac[17,14] - 3 frac[17,7] - 3 frac[17,25]
   - 3 frac[17,18] - 3 frac[17,45] - 3 frac[17,12] - 3 frac[17,38]
   - 3 frac[17,31] - 3 frac[17,16] - 3 frac[17,42] - 3 frac[17,2]
   - 3 frac[17,29] - 3 frac[17,22] - 3 frac[17,40] - 3 frac[17,15]
   - 3 frac[17,33] - 3 frac[17,19] + rows_tot_tor_P43 = 0
 R514: - 0.0208333333333333 rows_tot_tor_P43 + normalized_rows_tot_tor_P43
   = 0
 mem_tot_tor_P44: - mem[18,19] - mem[18,45] - mem[18,10] - mem[18,36]
   - mem[18,5] - mem[18,22] - mem[18,48] - mem[18,17] - mem[18,43]
   - mem[18,34] - mem[18,3] - mem[18,29] - mem[18,20] - mem[18,46]
   - mem[18,15] - mem[18,41] - mem[18,32] - mem[18,27] - mem[18,18]
   - mem[18,4] - mem[18,25] - mem[18,16] - mem[18,42] - mem[18,11]
   - mem[18,37] - mem[18,2] - mem[18,23] - mem[18,14] - mem[18,40]
   - mem[18,9] - mem[18,26] - mem[18,21] - mem[18,47] - mem[18,12]
   - mem[18,7] - mem[18,33] + mem_tot_tor_P44 = 0
 R516: - 1.6276041666666666e-04 mem_tot_tor_P44
   + normalized_mem_tot_tor_P44 = 0
 rows_tot_tor_P44: - 3 frac[18,19] - 3 frac[18,45] - 3 frac[18,10]
   - 3 frac[18,36] - 3 frac[18,5] - 3 frac[18,22] - 3 frac[18,48]
   - 3 frac[18,17] - 3 frac[18,43] - 3 frac[18,34] - 3 frac[18,3]
   - 3 frac[18,29] - 3 frac[18,20] - 3 frac[18,46] - 3 frac[18,15]
   - 3 frac[18,41] - 3 frac[18,32] - 3 frac[18,27] - 3 frac[18,18]
   - 3 frac[18,4] - 3 frac[18,25] - 3 frac[18,16] - 3 frac[18,42]
   - 3 frac[18,11] - 3 frac[18,37] - 3 frac[18,2] - 3 frac[18,23]
   - 3 frac[18,14] - 3 frac[18,40] - 3 frac[18,9] - 3 frac[18,26]
   - 3 frac[18,21] - 3 frac[18,47] - 3 frac[18,12] - 3 frac[18,7]
   - 3 frac[18,33] + rows_tot_tor_P44 = 0
 R518: - 0.0208333333333333 rows_tot_tor_P44 + normalized_rows_tot_tor_P44
   = 0
 mem_tot_l1_P41: - mem[19,4] - mem[19,44] - mem[19,18] - mem[19,27]
   - mem[19,1] - mem[19,32] - mem[19,6] - mem[19,41] - mem[19,15]
   - mem[19,46] - mem[19,20] - mem[19,29] - mem[19,3] - mem[19,34]
   - mem[19,8] - mem[19,17] - mem[19,48] - mem[19,22] - mem[19,31]
   - mem[19,36] - mem[19,10] - mem[19,45] - mem[19,19] - mem[19,24]
   - mem[19,33] - mem[19,7] - mem[19,38] - mem[19,12] - mem[19,21]
   - mem[19,26] - mem[19,35] - mem[19,9] - mem[19,40] - mem[19,49]
   - mem[19,23] - mem[19,28] - mem[19,2] - mem[19,37] - mem[19,11]
   - mem[19,42] - mem[19,16] - mem[19,25] - mem[19,30] + mem_tot_l1_P41
   = 0
 R520: - 1.6276041666666666e-04 mem_tot_l1_P41 + normalized_mem_tot_l1_P41
   = 0
 rows_tot_l1_P41: - 3 frac[19,4] - 3 frac[19,44] - 3 frac[19,18]
   - 3 frac[19,27] - 3 frac[19,1] - 3 frac[19,32] - 3 frac[19,6]
   - 3 frac[19,41] - 3 frac[19,15] - 3 frac[19,46] - 3 frac[19,20]
   - 3 frac[19,29] - 3 frac[19,3] - 3 frac[19,34] - 3 frac[19,8]
   - 3 frac[19,17] - 3 frac[19,48] - 3 frac[19,22] - 3 frac[19,31]
   - 3 frac[19,36] - 3 frac[19,10] - 3 frac[19,45] - 3 frac[19,19]
   - 3 frac[19,24] - 3 frac[19,33] - 3 frac[19,7] - 3 frac[19,38]
   - 3 frac[19,12] - 3 frac[19,21] - 3 frac[19,26] - 3 frac[19,35]
   - 3 frac[19,9] - 3 frac[19,40] - 3 frac[19,49] - 3 frac[19,23]
   - 3 frac[19,28] - 3 frac[19,2] - 3 frac[19,37] - 3 frac[19,11]
   - 3 frac[19,42] - 3 frac[19,16] - 3 frac[19,25] - 3 frac[19,30]
   + rows_tot_l1_P41 = 0
 R522: - 0.0208333333333333 rows_tot_l1_P41 + normalized_rows_tot_l1_P41
   = 0
 mem_tot_l1_P42: - mem[20,25] - mem[20,7] - mem[20,32] - mem[20,14]
   - mem[20,47] - mem[20,21] - mem[20,27] - mem[20,1] - mem[20,34]
   - mem[20,8] - mem[20,41] - mem[20,23] - mem[20,48] - mem[20,30]
   - mem[20,4] - mem[20,37] - mem[20,3] - mem[20,44] - mem[20,10]
   - mem[20,43] - mem[20,17] - mem[20,24] - mem[20,6] - mem[20,46]
   - mem[20,20] - mem[20,19] - mem[20,26] - mem[20,33] - mem[20,15]
   - mem[20,40] - mem[20,22] - mem[20,29] - mem[20,36] - mem[20,2]
   - mem[20,9] - mem[20,42] - mem[20,16] - mem[20,31] - mem[20,38]
   - mem[20,12] - mem[20,45] - mem[20,11] - mem[20,18] + mem_tot_l1_P42
   = 0
 R524: - 1.6276041666666666e-04 mem_tot_l1_P42 + normalized_mem_tot_l1_P42
   = 0
 rows_tot_l1_P42: - 3 frac[20,25] - 3 frac[20,7] - 3 frac[20,32]
   - 3 frac[20,14] - 3 frac[20,47] - 3 frac[20,21] - 3 frac[20,27]
   - 3 frac[20,1] - 3 frac[20,34] - 3 frac[20,8] - 3 frac[20,41]
   - 3 frac[20,23] - 3 frac[20,48] - 3 frac[20,30] - 3 frac[20,4]
   - 3 frac[20,37] - 3 frac[20,3] - 3 frac[20,44] - 3 frac[20,10]
   - 3 frac[20,43] - 3 frac[20,17] - 3 frac[20,24] - 3 frac[20,6]
   - 3 frac[20,46] - 3 frac[20,20] - 3 frac[20,19] - 3 frac[20,26]
   - 3 frac[20,33] - 3 frac[20,15] - 3 frac[20,40] - 3 frac[20,22]
   - 3 frac[20,29] - 3 frac[20,36] - 3 frac[20,2] - 3 frac[20,9]
   - 3 frac[20,42] - 3 frac[20,16] - 3 frac[20,31] - 3 frac[20,38]
   - 3 frac[20,12] - 3 frac[20,45] - 3 frac[20,11] - 3 frac[20,18]
   + rows_tot_l1_P42 = 0
 R526: - 0.0208333333333333 rows_tot_l1_P42 + normalized_rows_tot_l1_P42
   = 0
 mem_tot_l2_P4: - mem[21,6] - mem[21,37] - mem[21,15] - mem[21,46]
   - mem[21,8] - mem[21,17] - mem[21,48] - mem[21,26] - mem[21,4]
   - mem[21,44] - mem[21,22] - mem[21,31] - mem[21,24] - mem[21,2]
   - mem[21,33] - mem[21,11] - mem[21,42] - mem[21,20] - mem[21,29]
   - mem[21,7] - mem[21,38] - mem[21,9] - mem[21,40] - mem[21,18]
   - mem[21,27] - mem[21,36] - mem[21,45] - mem[21,23] - mem[21,16]
   - mem[21,25] - mem[21,3] - mem[21,34] - mem[21,12] - mem[21,21]
   - mem[21,30] - mem[21,1] - mem[21,32] - mem[21,10] - mem[21,41]
   + mem_tot_l2_P4 = 0
 R528: - 1.6276041666666666e-04 mem_tot_l2_P4 + normalized_mem_tot_l2_P4
   = 0
 rows_tot_l2_P4: - 3 frac[21,6] - 3 frac[21,37] - 3 frac[21,15]
   - 3 frac[21,46] - 3 frac[21,8] - 3 frac[21,17] - 3 frac[21,48]
   - 3 frac[21,26] - 3 frac[21,4] - 3 frac[21,44] - 3 frac[21,22]
   - 3 frac[21,31] - 3 frac[21,24] - 3 frac[21,2] - 3 frac[21,33]
   - 3 frac[21,11] - 3 frac[21,42] - 3 frac[21,20] - 3 frac[21,29]
   - 3 frac[21,7] - 3 frac[21,38] - 3 frac[21,9] - 3 frac[21,40]
   - 3 frac[21,18] - 3 frac[21,27] - 3 frac[21,36] - 3 frac[21,45]
   - 3 frac[21,23] - 3 frac[21,16] - 3 frac[21,25] - 3 frac[21,3]
   - 3 frac[21,34] - 3 frac[21,12] - 3 frac[21,21] - 3 frac[21,30]
   - 3 frac[21,1] - 3 frac[21,32] - 3 frac[21,10] - 3 frac[21,41]
   + rows_tot_l2_P4 = 0
 R530: - 0.0208333333333333 rows_tot_l2_P4 + normalized_rows_tot_l2_P4 = 0
 row_capacity_CPU12: rows_tot_CPU12 <= 12
 row_capacity_CPU26: rows_tot_CPU26 <= 12
 row_capacity_CPU1: rows_tot_CPU1 <= 12
 row_capacity_CPU4: rows_tot_CPU4 <= 12
 row_capacity_CPU24: rows_tot_CPU24 <= 12
 row_capacity_CPU21: rows_tot_CPU21 <= 12
 row_capacity_CPU2: rows_tot_CPU2 <= 12
 row_capacity_CPU32: rows_tot_CPU32 <= 12
 row_capacity_CPU11: rows_tot_CPU11 <= 12
 row_capacity_CPU14: rows_tot_CPU14 <= 12
 row_capacity_CPU10: rows_tot_CPU10 <= 12
 row_capacity_CPU23: rows_tot_CPU23 <= 12
 row_capacity_CPU31: rows_tot_CPU31 <= 12
 row_capacity_CPU13: rows_tot_CPU13 <= 12
 row_capacity_CPU16: rows_tot_CPU16 <= 12
 row_capacity_CPU5: rows_tot_CPU5 <= 12
 row_capacity_tor_P41: rows_tot_tor_P41 <= 48
 capacity_mem_par_tor_P41: mem[16,9] <= 1536
 capacity_mem_par_tor_P41: mem[16,0] <= 1536
 capacity_mem_par_tor_P41: mem[16,7] <= 1536
 capacity_mem_par_tor_P41: mem[16,29] <= 1536
 capacity_mem_par_tor_P41: mem[16,20] <= 1536
 capacity_mem_par_tor_P41: mem[16,42] <= 1536
 capacity_mem_par_tor_P41: mem[16,11] <= 1536
 capacity_mem_par_tor_P41: mem[16,2] <= 1536
 capacity_mem_par_tor_P41: mem[16,24] <= 1536
 capacity_mem_par_tor_P41: mem[16,31] <= 1536
 capacity_mem_par_tor_P41: mem[16,22] <= 1536
 capacity_mem_par_tor_P41: mem[16,44] <= 1536
 capacity_mem_par_tor_P41: mem[16,35] <= 1536
 capacity_mem_par_tor_P41: mem[16,4] <= 1536
 capacity_mem_par_tor_P41: mem[16,26] <= 1536
 capacity_mem_par_tor_P41: mem[16,46] <= 1536
 capacity_mem_par_tor_P41: mem[16,15] <= 1536
 capacity_mem_par_tor_P41: mem[16,37] <= 1536
 capacity_mem_par_tor_P41: mem[16,28] <= 1536
 capacity_mem_par_tor_P41: mem[16,19] <= 1536
 capacity_mem_par_tor_P41: mem[16,41] <= 1536
 capacity_mem_par_tor_P41: mem[16,10] <= 1536
 capacity_mem_par_tor_P41: mem[16,39] <= 1536
 capacity_mem_par_tor_P41: mem[16,21] <= 1536
 capacity_mem_par_tor_P41: mem[16,12] <= 1536
 capacity_mem_par_tor_P41: mem[16,34] <= 1536
 capacity_mem_par_tor_P41: mem[16,3] <= 1536
 capacity_mem_par_tor_P41: mem[16,25] <= 1536
 capacity_mem_par_tor_P41: mem[16,16] <= 1536
 capacity_mem_par_tor_P41: mem[16,23] <= 1536
 capacity_mem_par_tor_P41: mem[16,45] <= 1536
 capacity_mem_par_tor_P41: mem[16,49] <= 1536
 capacity_mem_par_tor_P41: mem[16,18] <= 1536
 row_capacity_tor_P43: rows_tot_tor_P43 <= 48
 capacity_mem_par_tor_P43: mem[17,46] <= 1536
 capacity_mem_par_tor_P43: mem[17,6] <= 1536
 capacity_mem_par_tor_P43: mem[17,24] <= 1536
 capacity_mem_par_tor_P43: mem[17,43] <= 1536
 capacity_mem_par_tor_P43: mem[17,10] <= 1536
 capacity_mem_par_tor_P43: mem[17,44] <= 1536
 capacity_mem_par_tor_P43: mem[17,3] <= 1536
 capacity_mem_par_tor_P43: mem[17,37] <= 1536
 capacity_mem_par_tor_P43: mem[17,30] <= 1536
 capacity_mem_par_tor_P43: mem[17,48] <= 1536
 capacity_mem_par_tor_P43: mem[17,23] <= 1536
 capacity_mem_par_tor_P43: mem[17,8] <= 1536
 capacity_mem_par_tor_P43: mem[17,34] <= 1536
 capacity_mem_par_tor_P43: mem[17,1] <= 1536
 capacity_mem_par_tor_P43: mem[17,21] <= 1536
 capacity_mem_par_tor_P43: mem[17,47] <= 1536
 capacity_mem_par_tor_P43: mem[17,14] <= 1536
 capacity_mem_par_tor_P43: mem[17,7] <= 1536
 capacity_mem_par_tor_P43: mem[17,25] <= 1536
 capacity_mem_par_tor_P43: mem[17,18] <= 1536
 capacity_mem_par_tor_P43: mem[17,45] <= 1536
 capacity_mem_par_tor_P43: mem[17,12] <= 1536
 capacity_mem_par_tor_P43: mem[17,38] <= 1536
 capacity_mem_par_tor_P43: mem[17,31] <= 1536
 capacity_mem_par_tor_P43: mem[17,16] <= 1536
 capacity_mem_par_tor_P43: mem[17,42] <= 1536
 capacity_mem_par_tor_P43: mem[17,2] <= 1536
 capacity_mem_par_tor_P43: mem[17,29] <= 1536
 capacity_mem_par_tor_P43: mem[17,22] <= 1536
 capacity_mem_par_tor_P43: mem[17,40] <= 1536
 capacity_mem_par_tor_P43: mem[17,15] <= 1536
 capacity_mem_par_tor_P43: mem[17,33] <= 1536
 capacity_mem_par_tor_P43: mem[17,19] <= 1536
 row_capacity_tor_P44: rows_tot_tor_P44 <= 48
 capacity_mem_par_tor_P44: mem[18,19] <= 1536
 capacity_mem_par_tor_P44: mem[18,45] <= 1536
 capacity_mem_par_tor_P44: mem[18,10] <= 1536
 capacity_mem_par_tor_P44: mem[18,36] <= 1536
 capacity_mem_par_tor_P44: mem[18,5] <= 1536
 capacity_mem_par_tor_P44: mem[18,22] <= 1536
 capacity_mem_par_tor_P44: mem[18,48] <= 1536
 capacity_mem_par_tor_P44: mem[18,17] <= 1536
 capacity_mem_par_tor_P44: mem[18,43] <= 1536
 capacity_mem_par_tor_P44: mem[18,34] <= 1536
 capacity_mem_par_tor_P44: mem[18,3] <= 1536
 capacity_mem_par_tor_P44: mem[18,29] <= 1536
 capacity_mem_par_tor_P44: mem[18,20] <= 1536
 capacity_mem_par_tor_P44: mem[18,46] <= 1536
 capacity_mem_par_tor_P44: mem[18,15] <= 1536
 capacity_mem_par_tor_P44: mem[18,41] <= 1536
 capacity_mem_par_tor_P44: mem[18,32] <= 1536
 capacity_mem_par_tor_P44: mem[18,27] <= 1536
 capacity_mem_par_tor_P44: mem[18,18] <= 1536
 capacity_mem_par_tor_P44: mem[18,4] <= 1536
 capacity_mem_par_tor_P44: mem[18,25] <= 1536
 capacity_mem_par_tor_P44: mem[18,16] <= 1536
 capacity_mem_par_tor_P44: mem[18,42] <= 1536
 capacity_mem_par_tor_P44: mem[18,11] <= 1536
 capacity_mem_par_tor_P44: mem[18,37] <= 1536
 capacity_mem_par_tor_P44: mem[18,2] <= 1536
 capacity_mem_par_tor_P44: mem[18,23] <= 1536
 capacity_mem_par_tor_P44: mem[18,14] <= 1536
 capacity_mem_par_tor_P44: mem[18,40] <= 1536
 capacity_mem_par_tor_P44: mem[18,9] <= 1536
 capacity_mem_par_tor_P44: mem[18,26] <= 1536
 capacity_mem_par_tor_P44: mem[18,21] <= 1536
 capacity_mem_par_tor_P44: mem[18,47] <= 1536
 capacity_mem_par_tor_P44: mem[18,12] <= 1536
 capacity_mem_par_tor_P44: mem[18,7] <= 1536
 capacity_mem_par_tor_P44: mem[18,33] <= 1536
 row_capacity_l1_P41: rows_tot_l1_P41 <= 48
 capacity_mem_par_l1_P41: mem[19,4] <= 1536
 capacity_mem_par_l1_P41: mem[19,44] <= 1536
 capacity_mem_par_l1_P41: mem[19,18] <= 1536
 capacity_mem_par_l1_P41: mem[19,27] <= 1536
 capacity_mem_par_l1_P41: mem[19,1] <= 1536
 capacity_mem_par_l1_P41: mem[19,32] <= 1536
 capacity_mem_par_l1_P41: mem[19,6] <= 1536
 capacity_mem_par_l1_P41: mem[19,41] <= 1536
 capacity_mem_par_l1_P41: mem[19,15] <= 1536
 capacity_mem_par_l1_P41: mem[19,46] <= 1536
 capacity_mem_par_l1_P41: mem[19,20] <= 1536
 capacity_mem_par_l1_P41: mem[19,29] <= 1536
 capacity_mem_par_l1_P41: mem[19,3] <= 1536
 capacity_mem_par_l1_P41: mem[19,34] <= 1536
 capacity_mem_par_l1_P41: mem[19,8] <= 1536
 capacity_mem_par_l1_P41: mem[19,17] <= 1536
 capacity_mem_par_l1_P41: mem[19,48] <= 1536
 capacity_mem_par_l1_P41: mem[19,22] <= 1536
 capacity_mem_par_l1_P41: mem[19,31] <= 1536
 capacity_mem_par_l1_P41: mem[19,36] <= 1536
 capacity_mem_par_l1_P41: mem[19,10] <= 1536
 capacity_mem_par_l1_P41: mem[19,45] <= 1536
 capacity_mem_par_l1_P41: mem[19,19] <= 1536
 capacity_mem_par_l1_P41: mem[19,24] <= 1536
 capacity_mem_par_l1_P41: mem[19,33] <= 1536
 capacity_mem_par_l1_P41: mem[19,7] <= 1536
 capacity_mem_par_l1_P41: mem[19,38] <= 1536
 capacity_mem_par_l1_P41: mem[19,12] <= 1536
 capacity_mem_par_l1_P41: mem[19,21] <= 1536
 capacity_mem_par_l1_P41: mem[19,26] <= 1536
 capacity_mem_par_l1_P41: mem[19,35] <= 1536
 capacity_mem_par_l1_P41: mem[19,9] <= 1536
 capacity_mem_par_l1_P41: mem[19,40] <= 1536
 capacity_mem_par_l1_P41: mem[19,49] <= 1536
 capacity_mem_par_l1_P41: mem[19,23] <= 1536
 capacity_mem_par_l1_P41: mem[19,28] <= 1536
 capacity_mem_par_l1_P41: mem[19,2] <= 1536
 capacity_mem_par_l1_P41: mem[19,37] <= 1536
 capacity_mem_par_l1_P41: mem[19,11] <= 1536
 capacity_mem_par_l1_P41: mem[19,42] <= 1536
 capacity_mem_par_l1_P41: mem[19,16] <= 1536
 capacity_mem_par_l1_P41: mem[19,25] <= 1536
 capacity_mem_par_l1_P41: mem[19,30] <= 1536
 row_capacity_l1_P42: rows_tot_l1_P42 <= 48
 capacity_mem_par_l1_P42: mem[20,25] <= 1536
 capacity_mem_par_l1_P42: mem[20,7] <= 1536
 capacity_mem_par_l1_P42: mem[20,32] <= 1536
 capacity_mem_par_l1_P42: mem[20,14] <= 1536
 capacity_mem_par_l1_P42: mem[20,47] <= 1536
 capacity_mem_par_l1_P42: mem[20,21] <= 1536
 capacity_mem_par_l1_P42: mem[20,27] <= 1536
 capacity_mem_par_l1_P42: mem[20,1] <= 1536
 capacity_mem_par_l1_P42: mem[20,34] <= 1536
 capacity_mem_par_l1_P42: mem[20,8] <= 1536
 capacity_mem_par_l1_P42: mem[20,41] <= 1536
 capacity_mem_par_l1_P42: mem[20,23] <= 1536
 capacity_mem_par_l1_P42: mem[20,48] <= 1536
 capacity_mem_par_l1_P42: mem[20,30] <= 1536
 capacity_mem_par_l1_P42: mem[20,4] <= 1536
 capacity_mem_par_l1_P42: mem[20,37] <= 1536
 capacity_mem_par_l1_P42: mem[20,3] <= 1536
 capacity_mem_par_l1_P42: mem[20,44] <= 1536
 capacity_mem_par_l1_P42: mem[20,10] <= 1536
 capacity_mem_par_l1_P42: mem[20,43] <= 1536
 capacity_mem_par_l1_P42: mem[20,17] <= 1536
 capacity_mem_par_l1_P42: mem[20,24] <= 1536
 capacity_mem_par_l1_P42: mem[20,6] <= 1536
 capacity_mem_par_l1_P42: mem[20,46] <= 1536
 capacity_mem_par_l1_P42: mem[20,20] <= 1536
 capacity_mem_par_l1_P42: mem[20,19] <= 1536
 capacity_mem_par_l1_P42: mem[20,26] <= 1536
 capacity_mem_par_l1_P42: mem[20,33] <= 1536
 capacity_mem_par_l1_P42: mem[20,15] <= 1536
 capacity_mem_par_l1_P42: mem[20,40] <= 1536
 capacity_mem_par_l1_P42: mem[20,22] <= 1536
 capacity_mem_par_l1_P42: mem[20,29] <= 1536
 capacity_mem_par_l1_P42: mem[20,36] <= 1536
 capacity_mem_par_l1_P42: mem[20,2] <= 1536
 capacity_mem_par_l1_P42: mem[20,9] <= 1536
 capacity_mem_par_l1_P42: mem[20,42] <= 1536
 capacity_mem_par_l1_P42: mem[20,16] <= 1536
 capacity_mem_par_l1_P42: mem[20,31] <= 1536
 capacity_mem_par_l1_P42: mem[20,38] <= 1536
 capacity_mem_par_l1_P42: mem[20,12] <= 1536
 capacity_mem_par_l1_P42: mem[20,45] <= 1536
 capacity_mem_par_l1_P42: mem[20,11] <= 1536
 capacity_mem_par_l1_P42: mem[20,18] <= 1536
 row_capacity_l2_P4: rows_tot_l2_P4 <= 48
 capacity_mem_par_l2_P4: mem[21,6] <= 1536
 capacity_mem_par_l2_P4: mem[21,37] <= 1536
 capacity_mem_par_l2_P4: mem[21,15] <= 1536
 capacity_mem_par_l2_P4: mem[21,46] <= 1536
 capacity_mem_par_l2_P4: mem[21,8] <= 1536
 capacity_mem_par_l2_P4: mem[21,17] <= 1536
 capacity_mem_par_l2_P4: mem[21,48] <= 1536
 capacity_mem_par_l2_P4: mem[21,26] <= 1536
 capacity_mem_par_l2_P4: mem[21,4] <= 1536
 capacity_mem_par_l2_P4: mem[21,44] <= 1536
 capacity_mem_par_l2_P4: mem[21,22] <= 1536
 capacity_mem_par_l2_P4: mem[21,31] <= 1536
 capacity_mem_par_l2_P4: mem[21,24] <= 1536
 capacity_mem_par_l2_P4: mem[21,2] <= 1536
 capacity_mem_par_l2_P4: mem[21,33] <= 1536
 capacity_mem_par_l2_P4: mem[21,11] <= 1536
 capacity_mem_par_l2_P4: mem[21,42] <= 1536
 capacity_mem_par_l2_P4: mem[21,20] <= 1536
 capacity_mem_par_l2_P4: mem[21,29] <= 1536
 capacity_mem_par_l2_P4: mem[21,7] <= 1536
 capacity_mem_par_l2_P4: mem[21,38] <= 1536
 capacity_mem_par_l2_P4: mem[21,9] <= 1536
 capacity_mem_par_l2_P4: mem[21,40] <= 1536
 capacity_mem_par_l2_P4: mem[21,18] <= 1536
 capacity_mem_par_l2_P4: mem[21,27] <= 1536
 capacity_mem_par_l2_P4: mem[21,36] <= 1536
 capacity_mem_par_l2_P4: mem[21,45] <= 1536
 capacity_mem_par_l2_P4: mem[21,23] <= 1536
 capacity_mem_par_l2_P4: mem[21,16] <= 1536
 capacity_mem_par_l2_P4: mem[21,25] <= 1536
 capacity_mem_par_l2_P4: mem[21,3] <= 1536
 capacity_mem_par_l2_P4: mem[21,34] <= 1536
 capacity_mem_par_l2_P4: mem[21,12] <= 1536
 capacity_mem_par_l2_P4: mem[21,21] <= 1536
 capacity_mem_par_l2_P4: mem[21,30] <= 1536
 capacity_mem_par_l2_P4: mem[21,1] <= 1536
 capacity_mem_par_l2_P4: mem[21,32] <= 1536
 capacity_mem_par_l2_P4: mem[21,10] <= 1536
 capacity_mem_par_l2_P4: mem[21,41] <= 1536
 ns_single_CPU12: 3.5 rows_tot_CPU12 - ns_single_CPU12 + pdt_m_rows_CPU12
   = 0
 ns_sketch_CPU12: - ns_single_CPU12 + 11.42863142857143 cores_sketch_CPU12
   >= 0
 capacity_cores_CPU12: cores_sketch_CPU12 <= 2
 ns_single_CPU26: 3.5 rows_tot_CPU26 - ns_single_CPU26 + pdt_m_rows_CPU26
   = 0
 ns_sketch_CPU26: - ns_single_CPU26 + 11.42863142857143 cores_sketch_CPU26
   >= 0
 capacity_cores_CPU26: cores_sketch_CPU26 <= 2
 ns_single_CPU1: 3.5 rows_tot_CPU1 - ns_single_CPU1 + pdt_m_rows_CPU1 = 0
 ns_sketch_CPU1: - ns_single_CPU1 + 11.42863142857143 cores_sketch_CPU1
   >= 0
 capacity_cores_CPU1: cores_sketch_CPU1 <= 2
 ns_single_CPU4: 3.5 rows_tot_CPU4 - ns_single_CPU4 + pdt_m_rows_CPU4 = 0
 ns_sketch_CPU4: - ns_single_CPU4 + 11.42863142857143 cores_sketch_CPU4
   >= 0
 capacity_cores_CPU4: cores_sketch_CPU4 <= 2
 ns_single_CPU24: 3.5 rows_tot_CPU24 - ns_single_CPU24 + pdt_m_rows_CPU24
   = 0
 ns_sketch_CPU24: - ns_single_CPU24 + 11.42863142857143 cores_sketch_CPU24
   >= 0
 capacity_cores_CPU24: cores_sketch_CPU24 <= 2
 ns_single_CPU21: 3.5 rows_tot_CPU21 - ns_single_CPU21 + pdt_m_rows_CPU21
   = 0
 ns_sketch_CPU21: - ns_single_CPU21 + 11.42863142857143 cores_sketch_CPU21
   >= 0
 capacity_cores_CPU21: cores_sketch_CPU21 <= 2
 ns_single_CPU2: 3.5 rows_tot_CPU2 - ns_single_CPU2 + pdt_m_rows_CPU2 = 0
 ns_sketch_CPU2: - ns_single_CPU2 + 11.42863142857143 cores_sketch_CPU2
   >= 0
 capacity_cores_CPU2: cores_sketch_CPU2 <= 2
 ns_single_CPU32: 3.5 rows_tot_CPU32 - ns_single_CPU32 + pdt_m_rows_CPU32
   = 0
 ns_sketch_CPU32: - ns_single_CPU32 + 11.42863142857143 cores_sketch_CPU32
   >= 0
 capacity_cores_CPU32: cores_sketch_CPU32 <= 2
 ns_single_CPU11: 3.5 rows_tot_CPU11 - ns_single_CPU11 + pdt_m_rows_CPU11
   = 0
 ns_sketch_CPU11: - ns_single_CPU11 + 11.42863142857143 cores_sketch_CPU11
   >= 0
 capacity_cores_CPU11: cores_sketch_CPU11 <= 2
 ns_single_CPU14: 3.5 rows_tot_CPU14 - ns_single_CPU14 + pdt_m_rows_CPU14
   = 0
 ns_sketch_CPU14: - ns_single_CPU14 + 11.42863142857143 cores_sketch_CPU14
   >= 0
 capacity_cores_CPU14: cores_sketch_CPU14 <= 2
 ns_single_CPU10: 3.5 rows_tot_CPU10 - ns_single_CPU10 + pdt_m_rows_CPU10
   = 0
 ns_sketch_CPU10: - ns_single_CPU10 + 11.42863142857143 cores_sketch_CPU10
   >= 0
 capacity_cores_CPU10: cores_sketch_CPU10 <= 2
 ns_single_CPU23: 3.5 rows_tot_CPU23 - ns_single_CPU23 + pdt_m_rows_CPU23
   = 0
 ns_sketch_CPU23: - ns_single_CPU23 + 11.42863142857143 cores_sketch_CPU23
   >= 0
 capacity_cores_CPU23: cores_sketch_CPU23 <= 2
 ns_single_CPU31: 3.5 rows_tot_CPU31 - ns_single_CPU31 + pdt_m_rows_CPU31
   = 0
 ns_sketch_CPU31: - ns_single_CPU31 + 11.42863142857143 cores_sketch_CPU31
   >= 0
 capacity_cores_CPU31: cores_sketch_CPU31 <= 2
 ns_single_CPU13: 3.5 rows_tot_CPU13 - ns_single_CPU13 + pdt_m_rows_CPU13
   = 0
 ns_sketch_CPU13: - ns_single_CPU13 + 11.42863142857143 cores_sketch_CPU13
   >= 0
 capacity_cores_CPU13: cores_sketch_CPU13 <= 2
 ns_single_CPU16: 3.5 rows_tot_CPU16 - ns_single_CPU16 + pdt_m_rows_CPU16
   = 0
 ns_sketch_CPU16: - ns_single_CPU16 + 11.42863142857143 cores_sketch_CPU16
   >= 0
 capacity_cores_CPU16: cores_sketch_CPU16 <= 2
 ns_single_CPU5: 3.5 rows_tot_CPU5 - ns_single_CPU5 + pdt_m_rows_CPU5 = 0
 ns_sketch_CPU5: - ns_single_CPU5 + 11.42863142857143 cores_sketch_CPU5
   >= 0
 capacity_cores_CPU5: cores_sketch_CPU5 <= 2
 res_acc: 1.220703125e-04 mem_tot_CPU12 + 1.220703125e-04 mem_tot_CPU26
   + 1.220703125e-04 mem_tot_CPU1 + 1.220703125e-04 mem_tot_CPU4
   + 1.220703125e-04 mem_tot_CPU24 + 1.220703125e-04 mem_tot_CPU21
   + 1.220703125e-04 mem_tot_CPU2 + 1.220703125e-04 mem_tot_CPU32
   + 1.220703125e-04 mem_tot_CPU11 + 1.220703125e-04 mem_tot_CPU14
   + 1.220703125e-04 mem_tot_CPU10 + 1.220703125e-04 mem_tot_CPU23
   + 1.220703125e-04 mem_tot_CPU31 + 1.220703125e-04 mem_tot_CPU13
   + 1.220703125e-04 mem_tot_CPU16 + 1.220703125e-04 mem_tot_CPU5
   + 0.001953125 mem_tot_tor_P41 + 0.25 rows_tot_tor_P41
   + 0.001953125 mem_tot_tor_P43 + 0.25 rows_tot_tor_P43
   + 0.001953125 mem_tot_tor_P44 + 0.25 rows_tot_tor_P44
   + 0.001953125 mem_tot_l1_P41 + 0.25 rows_tot_l1_P41
   + 0.001953125 mem_tot_l1_P42 + 0.25 rows_tot_l1_P42
   + 0.001953125 mem_tot_l2_P4 + 0.25 rows_tot_l2_P4
   + 10 cores_sketch_CPU12 + 10 cores_sketch_CPU26 + 10 cores_sketch_CPU1
   + 10 cores_sketch_CPU4 + 10 cores_sketch_CPU24 + 10 cores_sketch_CPU21
   + 10 cores_sketch_CPU2 + 10 cores_sketch_CPU32 + 10 cores_sketch_CPU11
   + 10 cores_sketch_CPU14 + 10 cores_sketch_CPU10 + 10 cores_sketch_CPU23
   + 10 cores_sketch_CPU31 + 10 cores_sketch_CPU13 + 10 cores_sketch_CPU16
   + 10 cores_sketch_CPU5 - res = -800
 pdt_m_rows_CPU12: pdt_m_rows_CPU12 + [
   - rows_tot_CPU12 * m_access_time_CPU12 ] = 0
 pdt_m_rows_CPU26: pdt_m_rows_CPU26 + [
   - rows_tot_CPU26 * m_access_time_CPU26 ] = 0
 pdt_m_rows_CPU1: pdt_m_rows_CPU1 + [ - rows_tot_CPU1 * m_access_time_CPU1
   ] = 0
 pdt_m_rows_CPU4: pdt_m_rows_CPU4 + [ - rows_tot_CPU4 * m_access_time_CPU4
   ] = 0
 pdt_m_rows_CPU24: pdt_m_rows_CPU24 + [
   - rows_tot_CPU24 * m_access_time_CPU24 ] = 0
 pdt_m_rows_CPU21: pdt_m_rows_CPU21 + [
   - rows_tot_CPU21 * m_access_time_CPU21 ] = 0
 pdt_m_rows_CPU2: pdt_m_rows_CPU2 + [ - rows_tot_CPU2 * m_access_time_CPU2
   ] = 0
 pdt_m_rows_CPU32: pdt_m_rows_CPU32 + [
   - rows_tot_CPU32 * m_access_time_CPU32 ] = 0
 pdt_m_rows_CPU11: pdt_m_rows_CPU11 + [
   - rows_tot_CPU11 * m_access_time_CPU11 ] = 0
 pdt_m_rows_CPU14: pdt_m_rows_CPU14 + [
   - rows_tot_CPU14 * m_access_time_CPU14 ] = 0
 pdt_m_rows_CPU10: pdt_m_rows_CPU10 + [
   - rows_tot_CPU10 * m_access_time_CPU10 ] = 0
 pdt_m_rows_CPU23: pdt_m_rows_CPU23 + [
   - rows_tot_CPU23 * m_access_time_CPU23 ] = 0
 pdt_m_rows_CPU31: pdt_m_rows_CPU31 + [
   - rows_tot_CPU31 * m_access_time_CPU31 ] = 0
 pdt_m_rows_CPU13: pdt_m_rows_CPU13 + [
   - rows_tot_CPU13 * m_access_time_CPU13 ] = 0
 pdt_m_rows_CPU16: pdt_m_rows_CPU16 + [
   - rows_tot_CPU16 * m_access_time_CPU16 ] = 0
 pdt_m_rows_CPU5: pdt_m_rows_CPU5 + [ - rows_tot_CPU5 * m_access_time_CPU5
   ] = 0
Bounds
 frac[0,38] <= 1
 frac[0,7] <= 1
 frac[0,42] <= 1
 frac[0,25] <= 1
 frac[0,10] <= 1
 frac[0,26] <= 1
 frac[1,21] <= 1
 frac[1,25] <= 1
 frac[1,26] <= 1
 frac[1,45] <= 1
 frac[1,5] <= 1
 frac[2,25] <= 1
 frac[2,45] <= 1
 frac[2,34] <= 1
 frac[2,21] <= 1
 frac[3,42] <= 1
 frac[3,21] <= 1
 frac[3,26] <= 1
 frac[3,34] <= 1
 frac[3,10] <= 1
 frac[3,25] <= 1
 frac[3,45] <= 1
 frac[3,0] <= 1
 frac[4,38] <= 1
 frac[4,14] <= 1
 frac[4,10] <= 1
 frac[4,31] <= 1
 frac[4,25] <= 1
 frac[4,21] <= 1
 frac[5,34] <= 1
 frac[5,10] <= 1
 frac[5,21] <= 1
 frac[5,7] <= 1
 frac[5,42] <= 1
 frac[5,45] <= 1
 frac[6,21] <= 1
 frac[6,7] <= 1
 frac[6,42] <= 1
 frac[6,45] <= 1
 frac[6,31] <= 1
 frac[6,9] <= 1
 frac[6,0] <= 1
 frac[6,26] <= 1
 frac[6,10] <= 1
 frac[7,34] <= 1
 frac[7,9] <= 1
 frac[7,21] <= 1
 frac[7,45] <= 1
 frac[7,42] <= 1
 frac[7,5] <= 1
 frac[7,14] <= 1
 frac[7,7] <= 1
 frac[7,10] <= 1
 frac[8,20] <= 1
 frac[8,37] <= 1
 frac[8,17] <= 1
 frac[8,24] <= 1
 frac[8,13] <= 1
 frac[8,41] <= 1
 frac[8,49] <= 1
 frac[8,30] <= 1
 frac[9,20] <= 1
 frac[9,44] <= 1
 frac[9,41] <= 1
 frac[9,33] <= 1
 frac[10,24] <= 1
 frac[10,20] <= 1
 frac[10,8] <= 1
 frac[10,41] <= 1
 frac[10,30] <= 1
 frac[11,8] <= 1
 frac[11,6] <= 1
 frac[11,30] <= 1
 frac[11,33] <= 1
 frac[11,44] <= 1
 frac[11,24] <= 1
 frac[11,37] <= 1
 frac[12,17] <= 1
 frac[12,41] <= 1
 frac[12,20] <= 1
 frac[12,4] <= 1
 frac[12,37] <= 1
 frac[12,33] <= 1
 frac[13,44] <= 1
 frac[13,6] <= 1
 frac[13,30] <= 1
 frac[13,41] <= 1
 frac[14,17] <= 1
 frac[14,24] <= 1
 frac[14,37] <= 1
 frac[14,13] <= 1
 frac[14,20] <= 1
 frac[15,4] <= 1
 frac[15,24] <= 1
 frac[15,44] <= 1
 frac[15,20] <= 1
 frac[15,49] <= 1
 frac[15,41] <= 1
 frac[15,37] <= 1
 mem_tot_CPU12 <= 32768
 normalized_mem_tot_CPU12 <= 1
 normalized_rows_tot_CPU12 <= 1
 mem_tot_CPU26 <= 32768
 normalized_mem_tot_CPU26 <= 1
 normalized_rows_tot_CPU26 <= 1
 mem_tot_CPU1 <= 32768
 normalized_mem_tot_CPU1 <= 1
 normalized_rows_tot_CPU1 <= 1
 mem_tot_CPU4 <= 32768
 normalized_mem_tot_CPU4 <= 1
 normalized_rows_tot_CPU4 <= 1
 mem_tot_CPU24 <= 32768
 normalized_mem_tot_CPU24 <= 1
 normalized_rows_tot_CPU24 <= 1
 mem_tot_CPU21 <= 32768
 normalized_mem_tot_CPU21 <= 1
 normalized_rows_tot_CPU21 <= 1
 mem_tot_CPU2 <= 32768
 normalized_mem_tot_CPU2 <= 1
 normalized_rows_tot_CPU2 <= 1
 mem_tot_CPU32 <= 32768
 normalized_mem_tot_CPU32 <= 1
 normalized_rows_tot_CPU32 <= 1
 mem_tot_CPU11 <= 32768
 normalized_mem_tot_CPU11 <= 1
 normalized_rows_tot_CPU11 <= 1
 mem_tot_CPU14 <= 32768
 normalized_mem_tot_CPU14 <= 1
 normalized_rows_tot_CPU14 <= 1
 mem_tot_CPU10 <= 32768
 normalized_mem_tot_CPU10 <= 1
 normalized_rows_tot_CPU10 <= 1
 mem_tot_CPU23 <= 32768
 normalized_mem_tot_CPU23 <= 1
 normalized_rows_tot_CPU23 <= 1
 mem_tot_CPU31 <= 32768
 normalized_mem_tot_CPU31 <= 1
 normalized_rows_tot_CPU31 <= 1
 mem_tot_CPU13 <= 32768
 normalized_mem_tot_CPU13 <= 1
 normalized_rows_tot_CPU13 <= 1
 mem_tot_CPU16 <= 32768
 normalized_mem_tot_CPU16 <= 1
 normalized_rows_tot_CPU16 <= 1
 mem_tot_CPU5 <= 32768
 normalized_mem_tot_CPU5 <= 1
 normalized_rows_tot_CPU5 <= 1
 mem_tot_tor_P41 <= 6144
 normalized_mem_tot_tor_P41 <= 1
 normalized_rows_tot_tor_P41 <= 1
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
 cores_sketch_CPU12 <= 7
 cores_sketch_CPU26 <= 7
 cores_sketch_CPU1 <= 7
 cores_sketch_CPU4 <= 7
 cores_sketch_CPU24 <= 7
 cores_sketch_CPU21 <= 7
 cores_sketch_CPU2 <= 7
 cores_sketch_CPU32 <= 7
 cores_sketch_CPU11 <= 7
 cores_sketch_CPU14 <= 7
 cores_sketch_CPU10 <= 7
 cores_sketch_CPU23 <= 7
 cores_sketch_CPU31 <= 7
 cores_sketch_CPU13 <= 7
 cores_sketch_CPU16 <= 7
 cores_sketch_CPU5 <= 7
Binaries
 frac[16,9] frac[16,0] frac[16,7] frac[16,29] frac[16,20] frac[16,42]
 frac[16,11] frac[16,2] frac[16,24] frac[16,31] frac[16,22] frac[16,44]
 frac[16,35] frac[16,4] frac[16,26] frac[16,46] frac[16,15] frac[16,37]
 frac[16,28] frac[16,19] frac[16,41] frac[16,10] frac[16,39] frac[16,21]
 frac[16,12] frac[16,34] frac[16,3] frac[16,25] frac[16,16] frac[16,23]
 frac[16,45] frac[16,49] frac[16,18] frac[17,46] frac[17,6] frac[17,24]
 frac[17,43] frac[17,10] frac[17,44] frac[17,3] frac[17,37] frac[17,30]
 frac[17,48] frac[17,23] frac[17,8] frac[17,34] frac[17,1] frac[17,21]
 frac[17,47] frac[17,14] frac[17,7] frac[17,25] frac[17,18] frac[17,45]
 frac[17,12] frac[17,38] frac[17,31] frac[17,16] frac[17,42] frac[17,2]
 frac[17,29] frac[17,22] frac[17,40] frac[17,15] frac[17,33] frac[17,19]
 frac[18,19] frac[18,45] frac[18,10] frac[18,36] frac[18,5] frac[18,22]
 frac[18,48] frac[18,17] frac[18,43] frac[18,34] frac[18,3] frac[18,29]
 frac[18,20] frac[18,46] frac[18,15] frac[18,41] frac[18,32] frac[18,27]
 frac[18,18] frac[18,4] frac[18,25] frac[18,16] frac[18,42] frac[18,11]
 frac[18,37] frac[18,2] frac[18,23] frac[18,14] frac[18,40] frac[18,9]
 frac[18,26] frac[18,21] frac[18,47] frac[18,12] frac[18,7] frac[18,33]
 frac[19,4] frac[19,44] frac[19,18] frac[19,27] frac[19,1] frac[19,32]
 frac[19,6] frac[19,41] frac[19,15] frac[19,46] frac[19,20] frac[19,29]
 frac[19,3] frac[19,34] frac[19,8] frac[19,17] frac[19,48] frac[19,22]
 frac[19,31] frac[19,36] frac[19,10] frac[19,45] frac[19,19] frac[19,24]
 frac[19,33] frac[19,7] frac[19,38] frac[19,12] frac[19,21] frac[19,26]
 frac[19,35] frac[19,9] frac[19,40] frac[19,49] frac[19,23] frac[19,28]
 frac[19,2] frac[19,37] frac[19,11] frac[19,42] frac[19,16] frac[19,25]
 frac[19,30] frac[20,25] frac[20,7] frac[20,32] frac[20,14] frac[20,47]
 frac[20,21] frac[20,27] frac[20,1] frac[20,34] frac[20,8] frac[20,41]
 frac[20,23] frac[20,48] frac[20,30] frac[20,4] frac[20,37] frac[20,3]
 frac[20,44] frac[20,10] frac[20,43] frac[20,17] frac[20,24] frac[20,6]
 frac[20,46] frac[20,20] frac[20,19] frac[20,26] frac[20,33] frac[20,15]
 frac[20,40] frac[20,22] frac[20,29] frac[20,36] frac[20,2] frac[20,9]
 frac[20,42] frac[20,16] frac[20,31] frac[20,38] frac[20,12] frac[20,45]
 frac[20,11] frac[20,18] frac[21,6] frac[21,37] frac[21,15] frac[21,46]
 frac[21,8] frac[21,17] frac[21,48] frac[21,26] frac[21,4] frac[21,44]
 frac[21,22] frac[21,31] frac[21,24] frac[21,2] frac[21,33] frac[21,11]
 frac[21,42] frac[21,20] frac[21,29] frac[21,7] frac[21,38] frac[21,9]
 frac[21,40] frac[21,18] frac[21,27] frac[21,36] frac[21,45] frac[21,23]
 frac[21,16] frac[21,25] frac[21,3] frac[21,34] frac[21,12] frac[21,21]
 frac[21,30] frac[21,1] frac[21,32] frac[21,10] frac[21,41]
Generals
 cores_sketch_CPU12 cores_sketch_CPU26 cores_sketch_CPU1 cores_sketch_CPU4
 cores_sketch_CPU24 cores_sketch_CPU21 cores_sketch_CPU2 cores_sketch_CPU32
 cores_sketch_CPU11 cores_sketch_CPU14 cores_sketch_CPU10
 cores_sketch_CPU23 cores_sketch_CPU31 cores_sketch_CPU13
 cores_sketch_CPU16 cores_sketch_CPU5
General Constraints
 mem_overall_fixed_thr: max_mem_fixed_thr = MAX ( mem_tot_tor_P41 ,
   mem_tot_tor_P43 , mem_tot_tor_P44 , mem_tot_l1_P41 , mem_tot_l1_P42 ,
   mem_tot_l2_P4 )
 mem_overall_others: max_mem_others = MAX ( normalized_mem_tot_CPU12 ,
   normalized_mem_tot_CPU26 , normalized_mem_tot_CPU1 ,
   normalized_mem_tot_CPU4 , normalized_mem_tot_CPU24 ,
   normalized_mem_tot_CPU21 , normalized_mem_tot_CPU2 ,
   normalized_mem_tot_CPU32 , normalized_mem_tot_CPU11 ,
   normalized_mem_tot_CPU14 , normalized_mem_tot_CPU10 ,
   normalized_mem_tot_CPU23 , normalized_mem_tot_CPU31 ,
   normalized_mem_tot_CPU13 , normalized_mem_tot_CPU16 ,
   normalized_mem_tot_CPU5 )
 rows_overall_fixed_thr: max_rows_fixed_thr = MAX ( rows_tot_tor_P41 ,
   rows_tot_tor_P43 , rows_tot_tor_P44 , rows_tot_l1_P41 ,
   rows_tot_l1_P42 , rows_tot_l2_P4 )
 rows_overall_others: max_rows_others = MAX ( normalized_rows_tot_CPU12 ,
   normalized_rows_tot_CPU26 , normalized_rows_tot_CPU1 ,
   normalized_rows_tot_CPU4 , normalized_rows_tot_CPU24 ,
   normalized_rows_tot_CPU21 , normalized_rows_tot_CPU2 ,
   normalized_rows_tot_CPU32 , normalized_rows_tot_CPU11 ,
   normalized_rows_tot_CPU14 , normalized_rows_tot_CPU10 ,
   normalized_rows_tot_CPU23 , normalized_rows_tot_CPU31 ,
   normalized_rows_tot_CPU13 , normalized_rows_tot_CPU16 ,
   normalized_rows_tot_CPU5 )
 mem_access_time_CPU12: m_access_time_CPU12 = PWL ( mem_tot_CPU12 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU26: m_access_time_CPU26 = PWL ( mem_tot_CPU26 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1: m_access_time_CPU1 = PWL ( mem_tot_CPU1 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU4: m_access_time_CPU4 = PWL ( mem_tot_CPU4 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU24: m_access_time_CPU24 = PWL ( mem_tot_CPU24 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU21: m_access_time_CPU21 = PWL ( mem_tot_CPU21 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU2: m_access_time_CPU2 = PWL ( mem_tot_CPU2 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU32: m_access_time_CPU32 = PWL ( mem_tot_CPU32 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU11: m_access_time_CPU11 = PWL ( mem_tot_CPU11 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU14: m_access_time_CPU14 = PWL ( mem_tot_CPU14 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU10: m_access_time_CPU10 = PWL ( mem_tot_CPU10 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU23: m_access_time_CPU23 = PWL ( mem_tot_CPU23 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU31: m_access_time_CPU31 = PWL ( mem_tot_CPU31 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU13: m_access_time_CPU13 = PWL ( mem_tot_CPU13 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU16: m_access_time_CPU16 = PWL ( mem_tot_CPU16 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU5: m_access_time_CPU5 = PWL ( mem_tot_CPU5 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

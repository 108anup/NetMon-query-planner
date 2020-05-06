\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 m_access_time_CPU5 + 0 m_access_time_CPU7 + 0 m_access_time_CPU9
   + 0 m_access_time_CPU16 + 0 m_access_time_CPU17 + 0 m_access_time_CPU22
   + 0 m_access_time_CPU23 + 0 m_access_time_CPU25 + res
Subject To
 cov_0_24: frac[2,24] + frac[4,24] + frac[8,24] + frac[9,24] >= 0.8
 cov_1_26: frac[2,26] + frac[4,26] + frac[8,26] + frac[9,26] >= 1
 cov_2_9: frac[1,9] + frac[3,9] + frac[8,9] >= 0.7
 cov_3_20: frac[1,20] + frac[3,20] + frac[8,20] >= 0.8
 cov_4_7: frac[5,7] >= 0.7
 cov_5_25: frac[0,25] + frac[7,25] + frac[8,25] + frac[9,25] >= 0.9
 cov_6_28: frac[1,28] + frac[3,28] + frac[8,28] >= 0.9
 cov_7_22: frac[0,22] + frac[5,22] + frac[8,22] + frac[9,22] >= 1
 cov_8_18: frac[0,18] + frac[4,18] + frac[8,18] + frac[9,18] >= 0.9
 cov_9_15: frac[3,15] + frac[8,15] >= 0.7
 cov_10_26: frac[3,26] + frac[5,26] + frac[8,26] + frac[9,26] >= 0.7
 cov_11_11: frac[1,11] + frac[7,11] + frac[8,11] + frac[9,11] >= 0.8
 cov_12_28: frac[0,28] + frac[6,28] + frac[8,28] + frac[9,28] >= 0.8
 cov_13_18: frac[1,18] + frac[2,18] + frac[8,18] >= 0.9
 cov_14_24: frac[1,24] + frac[6,24] + frac[8,24] + frac[9,24] >= 0.8
 cov_15_14: frac[2,14] + frac[4,14] + frac[8,14] + frac[9,14] >= 0.7
 cov_16_20: frac[1,20] + frac[6,20] + frac[8,20] + frac[9,20] >= 0.8
 cov_17_14: frac[3,14] + frac[5,14] + frac[8,14] + frac[9,14] >= 1
 cov_18_18: frac[0,18] + frac[7,18] + frac[8,18] + frac[9,18] >= 0.9
 cov_19_18: frac[3,18] + frac[7,18] + frac[8,18] + frac[9,18] >= 0.7
 cov_20_9: frac[3,9] + frac[6,9] + frac[8,9] + frac[9,9] >= 1
 cov_21_28: frac[2,28] + frac[4,28] + frac[8,28] + frac[9,28] >= 1
 cov_22_1: frac[4,1] + frac[6,1] >= 0.8
 cov_23_10: frac[1,10] + frac[2,10] + frac[8,10] >= 1
 cov_24_29: frac[2,29] + frac[5,29] + frac[8,29] + frac[9,29] >= 0.7
 cov_25_25: frac[3,25] + frac[4,25] + frac[8,25] + frac[9,25] >= 0.7
 cov_26_17: frac[2,17] + frac[8,17] + frac[9,17] >= 0.8
 cov_27_29: frac[1,29] + frac[4,29] + frac[8,29] + frac[9,29] >= 0.9
 cov_28_22: frac[1,22] + frac[3,22] + frac[8,22] >= 0.9
 cov_29_20: frac[2,20] + frac[5,20] + frac[8,20] + frac[9,20] >= 1
 cov_30_22: frac[3,22] + frac[7,22] + frac[8,22] + frac[9,22] >= 0.9
 cov_31_17: frac[1,17] + frac[6,17] + frac[8,17] + frac[9,17] >= 1
 cov_32_9: frac[3,9] + frac[6,9] + frac[8,9] + frac[9,9] >= 0.7
 cov_33_29: frac[1,29] + frac[2,29] + frac[8,29] >= 1
 cov_34_17: frac[0,17] + frac[7,17] + frac[8,17] + frac[9,17] >= 0.8
 cov_35_25: frac[0,25] + frac[6,25] + frac[8,25] + frac[9,25] >= 1
 cov_36_27: frac[2,27] + frac[3,27] >= 0.7
 cov_37_11: frac[0,11] + frac[6,11] + frac[8,11] + frac[9,11] >= 0.7
 cov_38_22: frac[0,22] + frac[2,22] + frac[8,22] >= 0.9
 cov_39_3: frac[0,3] + frac[6,3] + frac[8,3] + frac[9,3] >= 0.8
 cov_40_14: frac[1,14] + frac[7,14] + frac[8,14] + frac[9,14] >= 0.9
 cov_41_18: frac[1,18] + frac[2,18] + frac[8,18] >= 1
 cov_42_17: frac[0,17] + frac[6,17] + frac[8,17] + frac[9,17] >= 0.7
 cov_43_11: frac[0,11] + frac[5,11] + frac[8,11] + frac[9,11] >= 0.9
 cov_44_49: frac[8,49] >= 1
 cov_45_52: frac[4,52] + frac[8,52] + frac[9,52] >= 0.7
 cov_46_32: frac[8,32] + frac[9,32] >= 1
 cov_47_52: frac[8,52] + frac[9,52] >= 0.8
 cov_48_39: frac[1,39] >= 0.7
 cov_49_89: frac[8,89] + frac[9,89] >= 0.7
 cov_50_77: frac[8,77] >= 1
 cov_51_77: frac[8,77] + frac[9,77] >= 0.8
 cov_52_95: frac[8,95] + frac[9,95] >= 0.9
 cov_53_92: frac[8,92] + frac[9,92] >= 1
 cov_54_89: frac[8,89] >= 1
 cov_55_91: frac[8,91] + frac[9,91] >= 0.9
 cov_56_77: frac[8,77] + frac[9,77] >= 0.7
 cov_57_73: frac[8,73] + frac[9,73] >= 0.8
 cov_58_82: frac[8,82] + frac[9,82] >= 0.8
 cov_59_77: frac[8,77] + frac[9,77] >= 1
 cov_60_66: frac[8,66] >= 1
 cov_61_89: frac[8,89] + frac[9,89] >= 0.7
 cov_62_82: frac[8,82] + frac[9,82] >= 0.7
 cov_63_73: frac[8,73] + frac[9,73] >= 0.8
 cov_64_77: frac[8,77] + frac[9,77] >= 0.9
 cov_65_95: frac[8,95] + frac[9,95] >= 0.9
 cov_66_75: frac[8,75] >= 0.9
 cov_67_92: frac[8,92] + frac[9,92] >= 1
 cov_68_92: frac[8,92] >= 0.8
 cov_69_68: frac[8,68] >= 1
 cov_70_68: frac[8,68] + frac[9,68] >= 0.8
 cov_71_105: frac[8,105] + frac[9,105] >= 0.8
 cov_72_107: frac[8,107] + frac[9,107] >= 0.9
 cov_73_111: frac[8,111] + frac[9,111] >= 1
 cov_74_107: frac[8,107] >= 0.8
 cov_75_122: frac[8,122] >= 0.9
 cov_76_115: frac[8,115] + frac[9,115] >= 0.8
 cov_77_121: frac[8,121] + frac[9,121] >= 0.8
 cov_78_105: frac[8,105] + frac[9,105] >= 0.8
 cov_79_105: frac[8,105] + frac[9,105] >= 1
 cov_80_107: frac[1,107] + frac[8,107] + frac[9,107] >= 0.9
 cov_81_105: frac[8,105] + frac[9,105] >= 1
 cov_82_121: frac[0,121] + frac[8,121] + frac[9,121] >= 0.8
 accuracy_9_26: - 6 frac[9,26] + mem[9,26] = 0
 accuracy_6_9: - 5.09765625 frac[6,9] + mem[6,9] = 0
 accuracy_6_28: - 5.09765625 frac[6,28] + mem[6,28] = 0
 accuracy_1_17: - 5.09765625 frac[1,17] + mem[1,17] = 0
 accuracy_7_25: - 5.09765625 frac[7,25] + mem[7,25] = 0
 accuracy_2_22: - 5.09765625 frac[2,22] + mem[2,22] = 0
 accuracy_9_107: - 6 frac[9,107] + mem[9,107] = 0
 accuracy_1_11: - 5.09765625 frac[1,11] + mem[1,11] = 0
 accuracy_9_77: - 6 frac[9,77] + mem[9,77] = 0
 accuracy_8_24: - 6 frac[8,24] + mem[8,24] = 0
 accuracy_8_107: - 6 frac[8,107] + mem[8,107] = 0
 accuracy_0_17: - 5.09765625 frac[0,17] + mem[0,17] = 0
 accuracy_6_25: - 5.09765625 frac[6,25] + mem[6,25] = 0
 accuracy_1_28: - 5.09765625 frac[1,28] + mem[1,28] = 0
 accuracy_4_52: - 5.09765625 frac[4,52] + mem[4,52] = 0
 accuracy_2_27: - 5.09765625 frac[2,27] + mem[2,27] = 0
 accuracy_8_15: - 6 frac[8,15] + mem[8,15] = 0
 accuracy_8_68: - 6 frac[8,68] + mem[8,68] = 0
 accuracy_9_14: - 6 frac[9,14] + mem[9,14] = 0
 accuracy_5_11: - 5.09765625 frac[5,11] + mem[5,11] = 0
 accuracy_9_3: - 6 frac[9,3] + mem[9,3] = 0
 accuracy_9_82: - 6 frac[9,82] + mem[9,82] = 0
 accuracy_7_22: - 5.09765625 frac[7,22] + mem[7,22] = 0
 accuracy_2_17: - 5.09765625 frac[2,17] + mem[2,17] = 0
 accuracy_2_28: - 5.09765625 frac[2,28] + mem[2,28] = 0
 accuracy_9_52: - 6 frac[9,52] + mem[9,52] = 0
 accuracy_0_121: - 5.09765625 frac[0,121] + mem[0,121] = 0
 accuracy_9_9: - 6 frac[9,9] + mem[9,9] = 0
 accuracy_5_14: - 5.09765625 frac[5,14] + mem[5,14] = 0
 accuracy_4_26: - 5.09765625 frac[4,26] + mem[4,26] = 0
 accuracy_3_18: - 5.09765625 frac[3,18] + mem[3,18] = 0
 accuracy_8_18: - 6 frac[8,18] + mem[8,18] = 0
 accuracy_2_18: - 5.09765625 frac[2,18] + mem[2,18] = 0
 accuracy_9_111: - 6 frac[9,111] + mem[9,111] = 0
 accuracy_8_75: - 6 frac[8,75] + mem[8,75] = 0
 accuracy_8_9: - 6 frac[8,9] + mem[8,9] = 0
 accuracy_3_28: - 5.09765625 frac[3,28] + mem[3,28] = 0
 accuracy_8_28: - 6 frac[8,28] + mem[8,28] = 0
 accuracy_9_25: - 6 frac[9,25] + mem[9,25] = 0
 accuracy_8_111: - 6 frac[8,111] + mem[8,111] = 0
 accuracy_9_91: - 6 frac[9,91] + mem[9,91] = 0
 accuracy_8_32: - 6 frac[8,32] + mem[8,32] = 0
 accuracy_0_11: - 5.09765625 frac[0,11] + mem[0,11] = 0
 accuracy_1_10: - 5.09765625 frac[1,10] + mem[1,10] = 0
 accuracy_8_91: - 6 frac[8,91] + mem[8,91] = 0
 accuracy_4_1: - 5.09765625 frac[4,1] + mem[4,1] = 0
 accuracy_3_27: - 5.09765625 frac[3,27] + mem[3,27] = 0
 accuracy_8_25: - 6 frac[8,25] + mem[8,25] = 0
 accuracy_4_28: - 5.09765625 frac[4,28] + mem[4,28] = 0
 accuracy_9_20: - 6 frac[9,20] + mem[9,20] = 0
 accuracy_6_11: - 5.09765625 frac[6,11] + mem[6,11] = 0
 accuracy_0_18: - 5.09765625 frac[0,18] + mem[0,18] = 0
 accuracy_7_18: - 5.09765625 frac[7,18] + mem[7,18] = 0
 accuracy_2_24: - 5.09765625 frac[2,24] + mem[2,24] = 0
 accuracy_8_3: - 6 frac[8,3] + mem[8,3] = 0
 accuracy_3_22: - 5.09765625 frac[3,22] + mem[3,22] = 0
 accuracy_8_22: - 6 frac[8,22] + mem[8,22] = 0
 accuracy_6_1: - 5.09765625 frac[6,1] + mem[6,1] = 0
 accuracy_5_7: - 5.09765625 frac[5,7] + mem[5,7] = 0
 accuracy_9_92: - 6 frac[9,92] + mem[9,92] = 0
 accuracy_8_105: - 6 frac[8,105] + mem[8,105] = 0
 accuracy_6_20: - 5.09765625 frac[6,20] + mem[6,20] = 0
 accuracy_5_20: - 5.09765625 frac[5,20] + mem[5,20] = 0
 accuracy_7_17: - 5.09765625 frac[7,17] + mem[7,17] = 0
 accuracy_3_15: - 5.09765625 frac[3,15] + mem[3,15] = 0
 accuracy_2_29: - 5.09765625 frac[2,29] + mem[2,29] = 0
 accuracy_8_66: - 6 frac[8,66] + mem[8,66] = 0
 accuracy_9_29: - 6 frac[9,29] + mem[9,29] = 0
 accuracy_5_26: - 5.09765625 frac[5,26] + mem[5,26] = 0
 accuracy_9_95: - 6 frac[9,95] + mem[9,95] = 0
 accuracy_9_18: - 6 frac[9,18] + mem[9,18] = 0
 accuracy_7_11: - 5.09765625 frac[7,11] + mem[7,11] = 0
 accuracy_6_17: - 5.09765625 frac[6,17] + mem[6,17] = 0
 accuracy_1_20: - 5.09765625 frac[1,20] + mem[1,20] = 0
 accuracy_0_28: - 5.09765625 frac[0,28] + mem[0,28] = 0
 accuracy_8_121: - 6 frac[8,121] + mem[8,121] = 0
 accuracy_1_14: - 5.09765625 frac[1,14] + mem[1,14] = 0
 accuracy_8_10: - 6 frac[8,10] + mem[8,10] = 0
 accuracy_8_95: - 6 frac[8,95] + mem[8,95] = 0
 accuracy_9_11: - 6 frac[9,11] + mem[9,11] = 0
 accuracy_8_29: - 6 frac[8,29] + mem[8,29] = 0
 accuracy_8_82: - 6 frac[8,82] + mem[8,82] = 0
 accuracy_4_24: - 5.09765625 frac[4,24] + mem[4,24] = 0
 accuracy_9_24: - 6 frac[9,24] + mem[9,24] = 0
 accuracy_5_29: - 5.09765625 frac[5,29] + mem[5,29] = 0
 accuracy_0_22: - 5.09765625 frac[0,22] + mem[0,22] = 0
 accuracy_7_14: - 5.09765625 frac[7,14] + mem[7,14] = 0
 accuracy_0_25: - 5.09765625 frac[0,25] + mem[0,25] = 0
 accuracy_2_20: - 5.09765625 frac[2,20] + mem[2,20] = 0
 accuracy_9_105: - 6 frac[9,105] + mem[9,105] = 0
 accuracy_8_52: - 6 frac[8,52] + mem[8,52] = 0
 accuracy_8_73: - 6 frac[8,73] + mem[8,73] = 0
 accuracy_3_9: - 5.09765625 frac[3,9] + mem[3,9] = 0
 accuracy_1_9: - 5.09765625 frac[1,9] + mem[1,9] = 0
 accuracy_8_92: - 6 frac[8,92] + mem[8,92] = 0
 accuracy_3_26: - 5.09765625 frac[3,26] + mem[3,26] = 0
 accuracy_2_14: - 5.09765625 frac[2,14] + mem[2,14] = 0
 accuracy_9_115: - 6 frac[9,115] + mem[9,115] = 0
 accuracy_8_26: - 6 frac[8,26] + mem[8,26] = 0
 accuracy_4_29: - 5.09765625 frac[4,29] + mem[4,29] = 0
 accuracy_1_18: - 5.09765625 frac[1,18] + mem[1,18] = 0
 accuracy_8_115: - 6 frac[8,115] + mem[8,115] = 0
 accuracy_8_49: - 6 frac[8,49] + mem[8,49] = 0
 accuracy_9_121: - 6 frac[9,121] + mem[9,121] = 0
 accuracy_8_89: - 6 frac[8,89] + mem[8,89] = 0
 accuracy_3_25: - 5.09765625 frac[3,25] + mem[3,25] = 0
 accuracy_4_18: - 5.09765625 frac[4,18] + mem[4,18] = 0
 accuracy_9_22: - 6 frac[9,22] + mem[9,22] = 0
 accuracy_6_24: - 5.09765625 frac[6,24] + mem[6,24] = 0
 accuracy_1_29: - 5.09765625 frac[1,29] + mem[1,29] = 0
 accuracy_0_3: - 5.09765625 frac[0,3] + mem[0,3] = 0
 accuracy_3_14: - 5.09765625 frac[3,14] + mem[3,14] = 0
 accuracy_2_26: - 5.09765625 frac[2,26] + mem[2,26] = 0
 accuracy_8_14: - 6 frac[8,14] + mem[8,14] = 0
 accuracy_3_20: - 5.09765625 frac[3,20] + mem[3,20] = 0
 accuracy_9_73: - 6 frac[9,73] + mem[9,73] = 0
 accuracy_9_28: - 6 frac[9,28] + mem[9,28] = 0
 accuracy_8_20: - 6 frac[8,20] + mem[8,20] = 0
 accuracy_6_3: - 5.09765625 frac[6,3] + mem[6,3] = 0
 accuracy_9_17: - 6 frac[9,17] + mem[9,17] = 0
 accuracy_5_22: - 5.09765625 frac[5,22] + mem[5,22] = 0
 accuracy_1_107: - 5.09765625 frac[1,107] + mem[1,107] = 0
 accuracy_8_122: - 6 frac[8,122] + mem[8,122] = 0
 accuracy_1_24: - 5.09765625 frac[1,24] + mem[1,24] = 0
 accuracy_9_32: - 6 frac[9,32] + mem[9,32] = 0
 accuracy_8_77: - 6 frac[8,77] + mem[8,77] = 0
 accuracy_8_11: - 6 frac[8,11] + mem[8,11] = 0
 accuracy_4_14: - 5.09765625 frac[4,14] + mem[4,14] = 0
 accuracy_2_10: - 5.09765625 frac[2,10] + mem[2,10] = 0
 accuracy_4_25: - 5.09765625 frac[4,25] + mem[4,25] = 0
 accuracy_9_68: - 6 frac[9,68] + mem[9,68] = 0
 accuracy_1_39: - 5.09765625 frac[1,39] + mem[1,39] = 0
 accuracy_8_17: - 6 frac[8,17] + mem[8,17] = 0
 accuracy_9_89: - 6 frac[9,89] + mem[9,89] = 0
 accuracy_1_22: - 5.09765625 frac[1,22] + mem[1,22] = 0
 mem_tot_CPU5: - mem[0,17] - mem[0,121] - mem[0,11] - mem[0,18] - mem[0,28]
   - mem[0,22] - mem[0,25] - mem[0,3] + mem_tot_CPU5 = 0
 R219: - 3.05176e-05 mem_tot_CPU5 + normalized_mem_tot_CPU5 = 0
 rows_tot_CPU5: - 3 frac[0,17] - 3 frac[0,121] - 3 frac[0,11]
   - 3 frac[0,18] - 3 frac[0,28] - 3 frac[0,22] - 3 frac[0,25]
   - 3 frac[0,3] + rows_tot_CPU5 = 0
 R221: - 0.0833333333333333 rows_tot_CPU5 + normalized_rows_tot_CPU5 = 0
 mem_tot_CPU7: - mem[1,17] - mem[1,11] - mem[1,28] - mem[1,10] - mem[1,20]
   - mem[1,14] - mem[1,9] - mem[1,18] - mem[1,29] - mem[1,107] - mem[1,24]
   - mem[1,39] - mem[1,22] + mem_tot_CPU7 = 0
 R223: - 3.05176e-05 mem_tot_CPU7 + normalized_mem_tot_CPU7 = 0
 rows_tot_CPU7: - 3 frac[1,17] - 3 frac[1,11] - 3 frac[1,28] - 3 frac[1,10]
   - 3 frac[1,20] - 3 frac[1,14] - 3 frac[1,9] - 3 frac[1,18]
   - 3 frac[1,29] - 3 frac[1,107] - 3 frac[1,24] - 3 frac[1,39]
   - 3 frac[1,22] + rows_tot_CPU7 = 0
 R225: - 0.0833333333333333 rows_tot_CPU7 + normalized_rows_tot_CPU7 = 0
 mem_tot_CPU9: - mem[2,22] - mem[2,27] - mem[2,17] - mem[2,28] - mem[2,18]
   - mem[2,24] - mem[2,29] - mem[2,20] - mem[2,14] - mem[2,26] - mem[2,10]
   + mem_tot_CPU9 = 0
 R227: - 3.05176e-05 mem_tot_CPU9 + normalized_mem_tot_CPU9 = 0
 rows_tot_CPU9: - 3 frac[2,22] - 3 frac[2,27] - 3 frac[2,17] - 3 frac[2,28]
   - 3 frac[2,18] - 3 frac[2,24] - 3 frac[2,29] - 3 frac[2,20]
   - 3 frac[2,14] - 3 frac[2,26] - 3 frac[2,10] + rows_tot_CPU9 = 0
 R229: - 0.0833333333333333 rows_tot_CPU9 + normalized_rows_tot_CPU9 = 0
 mem_tot_CPU16: - mem[3,18] - mem[3,28] - mem[3,27] - mem[3,22] - mem[3,15]
   - mem[3,9] - mem[3,26] - mem[3,25] - mem[3,14] - mem[3,20]
   + mem_tot_CPU16 = 0
 R231: - 3.05176e-05 mem_tot_CPU16 + normalized_mem_tot_CPU16 = 0
 rows_tot_CPU16: - 3 frac[3,18] - 3 frac[3,28] - 3 frac[3,27]
   - 3 frac[3,22] - 3 frac[3,15] - 3 frac[3,9] - 3 frac[3,26]
   - 3 frac[3,25] - 3 frac[3,14] - 3 frac[3,20] + rows_tot_CPU16 = 0
 R233: - 0.0833333333333333 rows_tot_CPU16 + normalized_rows_tot_CPU16 = 0
 mem_tot_CPU17: - mem[4,52] - mem[4,26] - mem[4,1] - mem[4,28] - mem[4,24]
   - mem[4,29] - mem[4,18] - mem[4,14] - mem[4,25] + mem_tot_CPU17 = 0
 R235: - 3.05176e-05 mem_tot_CPU17 + normalized_mem_tot_CPU17 = 0
 rows_tot_CPU17: - 3 frac[4,52] - 3 frac[4,26] - 3 frac[4,1] - 3 frac[4,28]
   - 3 frac[4,24] - 3 frac[4,29] - 3 frac[4,18] - 3 frac[4,14]
   - 3 frac[4,25] + rows_tot_CPU17 = 0
 R237: - 0.0833333333333333 rows_tot_CPU17 + normalized_rows_tot_CPU17 = 0
 mem_tot_CPU22: - mem[5,11] - mem[5,14] - mem[5,7] - mem[5,20] - mem[5,26]
   - mem[5,29] - mem[5,22] + mem_tot_CPU22 = 0
 R239: - 3.05176e-05 mem_tot_CPU22 + normalized_mem_tot_CPU22 = 0
 rows_tot_CPU22: - 3 frac[5,11] - 3 frac[5,14] - 3 frac[5,7] - 3 frac[5,20]
   - 3 frac[5,26] - 3 frac[5,29] - 3 frac[5,22] + rows_tot_CPU22 = 0
 R241: - 0.0833333333333333 rows_tot_CPU22 + normalized_rows_tot_CPU22 = 0
 mem_tot_CPU23: - mem[6,9] - mem[6,28] - mem[6,25] - mem[6,11] - mem[6,1]
   - mem[6,20] - mem[6,17] - mem[6,24] - mem[6,3] + mem_tot_CPU23 = 0
 R243: - 3.05176e-05 mem_tot_CPU23 + normalized_mem_tot_CPU23 = 0
 rows_tot_CPU23: - 3 frac[6,9] - 3 frac[6,28] - 3 frac[6,25] - 3 frac[6,11]
   - 3 frac[6,1] - 3 frac[6,20] - 3 frac[6,17] - 3 frac[6,24] - 3 frac[6,3]
   + rows_tot_CPU23 = 0
 R245: - 0.0833333333333333 rows_tot_CPU23 + normalized_rows_tot_CPU23 = 0
 mem_tot_CPU25: - mem[7,25] - mem[7,22] - mem[7,18] - mem[7,17] - mem[7,11]
   - mem[7,14] + mem_tot_CPU25 = 0
 R247: - 3.05176e-05 mem_tot_CPU25 + normalized_mem_tot_CPU25 = 0
 rows_tot_CPU25: - 3 frac[7,25] - 3 frac[7,22] - 3 frac[7,18]
   - 3 frac[7,17] - 3 frac[7,11] - 3 frac[7,14] + rows_tot_CPU25 = 0
 R249: - 0.0833333333333333 rows_tot_CPU25 + normalized_rows_tot_CPU25 = 0
 mem_tot_l1_P41: - mem[8,24] - mem[8,107] - mem[8,15] - mem[8,68]
   - mem[8,18] - mem[8,75] - mem[8,9] - mem[8,28] - mem[8,111] - mem[8,32]
   - mem[8,91] - mem[8,25] - mem[8,3] - mem[8,22] - mem[8,105] - mem[8,66]
   - mem[8,121] - mem[8,10] - mem[8,95] - mem[8,29] - mem[8,82] - mem[8,52]
   - mem[8,73] - mem[8,92] - mem[8,26] - mem[8,115] - mem[8,49] - mem[8,89]
   - mem[8,14] - mem[8,20] - mem[8,122] - mem[8,77] - mem[8,11] - mem[8,17]
   + mem_tot_l1_P41 = 0
 R251: - 1.6276041666666666e-04 mem_tot_l1_P41 + normalized_mem_tot_l1_P41
   = 0
 rows_tot_l1_P41: - 3 frac[8,24] - 3 frac[8,107] - 3 frac[8,15]
   - 3 frac[8,68] - 3 frac[8,18] - 3 frac[8,75] - 3 frac[8,9]
   - 3 frac[8,28] - 3 frac[8,111] - 3 frac[8,32] - 3 frac[8,91]
   - 3 frac[8,25] - 3 frac[8,3] - 3 frac[8,22] - 3 frac[8,105]
   - 3 frac[8,66] - 3 frac[8,121] - 3 frac[8,10] - 3 frac[8,95]
   - 3 frac[8,29] - 3 frac[8,82] - 3 frac[8,52] - 3 frac[8,73]
   - 3 frac[8,92] - 3 frac[8,26] - 3 frac[8,115] - 3 frac[8,49]
   - 3 frac[8,89] - 3 frac[8,14] - 3 frac[8,20] - 3 frac[8,122]
   - 3 frac[8,77] - 3 frac[8,11] - 3 frac[8,17] + rows_tot_l1_P41 = 0
 R253: - 0.0208333333333333 rows_tot_l1_P41 + normalized_rows_tot_l1_P41
   = 0
 mem_tot_l2_P4: - mem[9,26] - mem[9,107] - mem[9,77] - mem[9,14] - mem[9,3]
   - mem[9,82] - mem[9,52] - mem[9,9] - mem[9,111] - mem[9,25] - mem[9,91]
   - mem[9,20] - mem[9,92] - mem[9,29] - mem[9,95] - mem[9,18] - mem[9,11]
   - mem[9,24] - mem[9,105] - mem[9,115] - mem[9,121] - mem[9,22]
   - mem[9,73] - mem[9,28] - mem[9,17] - mem[9,32] - mem[9,68] - mem[9,89]
   + mem_tot_l2_P4 = 0
 R255: - 1.6276041666666666e-04 mem_tot_l2_P4 + normalized_mem_tot_l2_P4
   = 0
 rows_tot_l2_P4: - 3 frac[9,26] - 3 frac[9,107] - 3 frac[9,77]
   - 3 frac[9,14] - 3 frac[9,3] - 3 frac[9,82] - 3 frac[9,52] - 3 frac[9,9]
   - 3 frac[9,111] - 3 frac[9,25] - 3 frac[9,91] - 3 frac[9,20]
   - 3 frac[9,92] - 3 frac[9,29] - 3 frac[9,95] - 3 frac[9,18]
   - 3 frac[9,11] - 3 frac[9,24] - 3 frac[9,105] - 3 frac[9,115]
   - 3 frac[9,121] - 3 frac[9,22] - 3 frac[9,73] - 3 frac[9,28]
   - 3 frac[9,17] - 3 frac[9,32] - 3 frac[9,68] - 3 frac[9,89]
   + rows_tot_l2_P4 = 0
 R257: - 0.0208333333333333 rows_tot_l2_P4 + normalized_rows_tot_l2_P4 = 0
 row_capacity_CPU5: rows_tot_CPU5 <= 12
 row_capacity_CPU7: rows_tot_CPU7 <= 12
 row_capacity_CPU9: rows_tot_CPU9 <= 12
 row_capacity_CPU16: rows_tot_CPU16 <= 12
 row_capacity_CPU17: rows_tot_CPU17 <= 12
 row_capacity_CPU22: rows_tot_CPU22 <= 12
 row_capacity_CPU23: rows_tot_CPU23 <= 12
 row_capacity_CPU25: rows_tot_CPU25 <= 12
 row_capacity_l1_P41: rows_tot_l1_P41 <= 48
 capacity_mem_par_l1_P41: mem[8,24] <= 1536
 capacity_mem_par_l1_P41: mem[8,107] <= 1536
 capacity_mem_par_l1_P41: mem[8,15] <= 1536
 capacity_mem_par_l1_P41: mem[8,68] <= 1536
 capacity_mem_par_l1_P41: mem[8,18] <= 1536
 capacity_mem_par_l1_P41: mem[8,75] <= 1536
 capacity_mem_par_l1_P41: mem[8,9] <= 1536
 capacity_mem_par_l1_P41: mem[8,28] <= 1536
 capacity_mem_par_l1_P41: mem[8,111] <= 1536
 capacity_mem_par_l1_P41: mem[8,32] <= 1536
 capacity_mem_par_l1_P41: mem[8,91] <= 1536
 capacity_mem_par_l1_P41: mem[8,25] <= 1536
 capacity_mem_par_l1_P41: mem[8,3] <= 1536
 capacity_mem_par_l1_P41: mem[8,22] <= 1536
 capacity_mem_par_l1_P41: mem[8,105] <= 1536
 capacity_mem_par_l1_P41: mem[8,66] <= 1536
 capacity_mem_par_l1_P41: mem[8,121] <= 1536
 capacity_mem_par_l1_P41: mem[8,10] <= 1536
 capacity_mem_par_l1_P41: mem[8,95] <= 1536
 capacity_mem_par_l1_P41: mem[8,29] <= 1536
 capacity_mem_par_l1_P41: mem[8,82] <= 1536
 capacity_mem_par_l1_P41: mem[8,52] <= 1536
 capacity_mem_par_l1_P41: mem[8,73] <= 1536
 capacity_mem_par_l1_P41: mem[8,92] <= 1536
 capacity_mem_par_l1_P41: mem[8,26] <= 1536
 capacity_mem_par_l1_P41: mem[8,115] <= 1536
 capacity_mem_par_l1_P41: mem[8,49] <= 1536
 capacity_mem_par_l1_P41: mem[8,89] <= 1536
 capacity_mem_par_l1_P41: mem[8,14] <= 1536
 capacity_mem_par_l1_P41: mem[8,20] <= 1536
 capacity_mem_par_l1_P41: mem[8,122] <= 1536
 capacity_mem_par_l1_P41: mem[8,77] <= 1536
 capacity_mem_par_l1_P41: mem[8,11] <= 1536
 capacity_mem_par_l1_P41: mem[8,17] <= 1536
 row_capacity_l2_P4: rows_tot_l2_P4 <= 48
 capacity_mem_par_l2_P4: mem[9,26] <= 1536
 capacity_mem_par_l2_P4: mem[9,107] <= 1536
 capacity_mem_par_l2_P4: mem[9,77] <= 1536
 capacity_mem_par_l2_P4: mem[9,14] <= 1536
 capacity_mem_par_l2_P4: mem[9,3] <= 1536
 capacity_mem_par_l2_P4: mem[9,82] <= 1536
 capacity_mem_par_l2_P4: mem[9,52] <= 1536
 capacity_mem_par_l2_P4: mem[9,9] <= 1536
 capacity_mem_par_l2_P4: mem[9,111] <= 1536
 capacity_mem_par_l2_P4: mem[9,25] <= 1536
 capacity_mem_par_l2_P4: mem[9,91] <= 1536
 capacity_mem_par_l2_P4: mem[9,20] <= 1536
 capacity_mem_par_l2_P4: mem[9,92] <= 1536
 capacity_mem_par_l2_P4: mem[9,29] <= 1536
 capacity_mem_par_l2_P4: mem[9,95] <= 1536
 capacity_mem_par_l2_P4: mem[9,18] <= 1536
 capacity_mem_par_l2_P4: mem[9,11] <= 1536
 capacity_mem_par_l2_P4: mem[9,24] <= 1536
 capacity_mem_par_l2_P4: mem[9,105] <= 1536
 capacity_mem_par_l2_P4: mem[9,115] <= 1536
 capacity_mem_par_l2_P4: mem[9,121] <= 1536
 capacity_mem_par_l2_P4: mem[9,22] <= 1536
 capacity_mem_par_l2_P4: mem[9,73] <= 1536
 capacity_mem_par_l2_P4: mem[9,28] <= 1536
 capacity_mem_par_l2_P4: mem[9,17] <= 1536
 capacity_mem_par_l2_P4: mem[9,32] <= 1536
 capacity_mem_par_l2_P4: mem[9,68] <= 1536
 capacity_mem_par_l2_P4: mem[9,89] <= 1536
 ns_single_CPU5: 3.5 rows_tot_CPU5 - ns_single_CPU5 + pdt_m_rows_CPU5 = 0
 ns_sketch_CPU5: - ns_single_CPU5 + 10.71428571428572 cores_sketch_CPU5
   >= 0
 capacity_cores_CPU5: cores_sketch_CPU5 <= 1
 ns_single_CPU7: 3.5 rows_tot_CPU7 - ns_single_CPU7 + pdt_m_rows_CPU7 = 0
 ns_sketch_CPU7: - ns_single_CPU7 + 10.71428571428572 cores_sketch_CPU7
   >= 0
 capacity_cores_CPU7: cores_sketch_CPU7 <= 1
 ns_single_CPU9: 3.5 rows_tot_CPU9 - ns_single_CPU9 + pdt_m_rows_CPU9 = 0
 ns_sketch_CPU9: - ns_single_CPU9 + 10.71428571428572 cores_sketch_CPU9
   >= 0
 capacity_cores_CPU9: cores_sketch_CPU9 <= 1
 ns_single_CPU16: 3.5 rows_tot_CPU16 - ns_single_CPU16 + pdt_m_rows_CPU16
   = 0
 ns_sketch_CPU16: - ns_single_CPU16 + 10.71428571428572 cores_sketch_CPU16
   >= 0
 capacity_cores_CPU16: cores_sketch_CPU16 <= 1
 ns_single_CPU17: 3.5 rows_tot_CPU17 - ns_single_CPU17 + pdt_m_rows_CPU17
   = 0
 ns_sketch_CPU17: - ns_single_CPU17 + 10.71428571428572 cores_sketch_CPU17
   >= 0
 capacity_cores_CPU17: cores_sketch_CPU17 <= 1
 ns_single_CPU22: 3.5 rows_tot_CPU22 - ns_single_CPU22 + pdt_m_rows_CPU22
   = 0
 ns_sketch_CPU22: - ns_single_CPU22 + 10.71428571428572 cores_sketch_CPU22
   >= 0
 capacity_cores_CPU22: cores_sketch_CPU22 <= 1
 ns_single_CPU23: 3.5 rows_tot_CPU23 - ns_single_CPU23 + pdt_m_rows_CPU23
   = 0
 ns_sketch_CPU23: - ns_single_CPU23 + 10.71428571428572 cores_sketch_CPU23
   >= 0
 capacity_cores_CPU23: cores_sketch_CPU23 <= 1
 ns_single_CPU25: 3.5 rows_tot_CPU25 - ns_single_CPU25 + pdt_m_rows_CPU25
   = 0
 ns_sketch_CPU25: - ns_single_CPU25 + 10.71428571428572 cores_sketch_CPU25
   >= 0
 capacity_cores_CPU25: cores_sketch_CPU25 <= 1
 res_acc: 1.220703125e-04 mem_tot_CPU5 + 1.220703125e-04 mem_tot_CPU7
   + 1.220703125e-04 mem_tot_CPU9 + 1.220703125e-04 mem_tot_CPU16
   + 1.220703125e-04 mem_tot_CPU17 + 1.220703125e-04 mem_tot_CPU22
   + 1.220703125e-04 mem_tot_CPU23 + 1.220703125e-04 mem_tot_CPU25
   + 0.001953125 mem_tot_l1_P41 + 0.25 rows_tot_l1_P41
   + 0.001953125 mem_tot_l2_P4 + 0.25 rows_tot_l2_P4 + 10 cores_sketch_CPU5
   + 10 cores_sketch_CPU7 + 10 cores_sketch_CPU9 + 10 cores_sketch_CPU16
   + 10 cores_sketch_CPU17 + 10 cores_sketch_CPU22 + 10 cores_sketch_CPU23
   + 10 cores_sketch_CPU25 - res = -480
 pdt_m_rows_CPU5: pdt_m_rows_CPU5 + [ - rows_tot_CPU5 * m_access_time_CPU5
   ] = 0
 pdt_m_rows_CPU7: pdt_m_rows_CPU7 + [ - rows_tot_CPU7 * m_access_time_CPU7
   ] = 0
 pdt_m_rows_CPU9: pdt_m_rows_CPU9 + [ - rows_tot_CPU9 * m_access_time_CPU9
   ] = 0
 pdt_m_rows_CPU16: pdt_m_rows_CPU16 + [
   - rows_tot_CPU16 * m_access_time_CPU16 ] = 0
 pdt_m_rows_CPU17: pdt_m_rows_CPU17 + [
   - rows_tot_CPU17 * m_access_time_CPU17 ] = 0
 pdt_m_rows_CPU22: pdt_m_rows_CPU22 + [
   - rows_tot_CPU22 * m_access_time_CPU22 ] = 0
 pdt_m_rows_CPU23: pdt_m_rows_CPU23 + [
   - rows_tot_CPU23 * m_access_time_CPU23 ] = 0
 pdt_m_rows_CPU25: pdt_m_rows_CPU25 + [
   - rows_tot_CPU25 * m_access_time_CPU25 ] = 0
Bounds
 frac[0,17] <= 1
 frac[0,121] <= 1
 frac[0,11] <= 1
 frac[0,18] <= 1
 frac[0,28] <= 1
 frac[0,22] <= 1
 frac[0,25] <= 1
 frac[0,3] <= 1
 frac[1,17] <= 1
 frac[1,11] <= 1
 frac[1,28] <= 1
 frac[1,10] <= 1
 frac[1,20] <= 1
 frac[1,14] <= 1
 frac[1,9] <= 1
 frac[1,18] <= 1
 frac[1,29] <= 1
 frac[1,107] <= 1
 frac[1,24] <= 1
 frac[1,39] <= 1
 frac[1,22] <= 1
 frac[2,22] <= 1
 frac[2,27] <= 1
 frac[2,17] <= 1
 frac[2,28] <= 1
 frac[2,18] <= 1
 frac[2,24] <= 1
 frac[2,29] <= 1
 frac[2,20] <= 1
 frac[2,14] <= 1
 frac[2,26] <= 1
 frac[2,10] <= 1
 frac[3,18] <= 1
 frac[3,28] <= 1
 frac[3,27] <= 1
 frac[3,22] <= 1
 frac[3,15] <= 1
 frac[3,9] <= 1
 frac[3,26] <= 1
 frac[3,25] <= 1
 frac[3,14] <= 1
 frac[3,20] <= 1
 frac[4,52] <= 1
 frac[4,26] <= 1
 frac[4,1] <= 1
 frac[4,28] <= 1
 frac[4,24] <= 1
 frac[4,29] <= 1
 frac[4,18] <= 1
 frac[4,14] <= 1
 frac[4,25] <= 1
 frac[5,11] <= 1
 frac[5,14] <= 1
 frac[5,7] <= 1
 frac[5,20] <= 1
 frac[5,26] <= 1
 frac[5,29] <= 1
 frac[5,22] <= 1
 frac[6,9] <= 1
 frac[6,28] <= 1
 frac[6,25] <= 1
 frac[6,11] <= 1
 frac[6,1] <= 1
 frac[6,20] <= 1
 frac[6,17] <= 1
 frac[6,24] <= 1
 frac[6,3] <= 1
 frac[7,25] <= 1
 frac[7,22] <= 1
 frac[7,18] <= 1
 frac[7,17] <= 1
 frac[7,11] <= 1
 frac[7,14] <= 1
 mem_tot_CPU5 <= 32768
 normalized_mem_tot_CPU5 <= 1
 normalized_rows_tot_CPU5 <= 1
 mem_tot_CPU7 <= 32768
 normalized_mem_tot_CPU7 <= 1
 normalized_rows_tot_CPU7 <= 1
 mem_tot_CPU9 <= 32768
 normalized_mem_tot_CPU9 <= 1
 normalized_rows_tot_CPU9 <= 1
 mem_tot_CPU16 <= 32768
 normalized_mem_tot_CPU16 <= 1
 normalized_rows_tot_CPU16 <= 1
 mem_tot_CPU17 <= 32768
 normalized_mem_tot_CPU17 <= 1
 normalized_rows_tot_CPU17 <= 1
 mem_tot_CPU22 <= 32768
 normalized_mem_tot_CPU22 <= 1
 normalized_rows_tot_CPU22 <= 1
 mem_tot_CPU23 <= 32768
 normalized_mem_tot_CPU23 <= 1
 normalized_rows_tot_CPU23 <= 1
 mem_tot_CPU25 <= 32768
 normalized_mem_tot_CPU25 <= 1
 normalized_rows_tot_CPU25 <= 1
 mem_tot_l1_P41 <= 6144
 normalized_mem_tot_l1_P41 <= 1
 normalized_rows_tot_l1_P41 <= 1
 mem_tot_l2_P4 <= 6144
 normalized_mem_tot_l2_P4 <= 1
 normalized_rows_tot_l2_P4 <= 1
 cores_sketch_CPU5 <= 7
 cores_sketch_CPU7 <= 7
 cores_sketch_CPU9 <= 7
 cores_sketch_CPU16 <= 7
 cores_sketch_CPU17 <= 7
 cores_sketch_CPU22 <= 7
 cores_sketch_CPU23 <= 7
 cores_sketch_CPU25 <= 7
Binaries
 frac[8,24] frac[8,107] frac[8,15] frac[8,68] frac[8,18] frac[8,75]
 frac[8,9] frac[8,28] frac[8,111] frac[8,32] frac[8,91] frac[8,25]
 frac[8,3] frac[8,22] frac[8,105] frac[8,66] frac[8,121] frac[8,10]
 frac[8,95] frac[8,29] frac[8,82] frac[8,52] frac[8,73] frac[8,92]
 frac[8,26] frac[8,115] frac[8,49] frac[8,89] frac[8,14] frac[8,20]
 frac[8,122] frac[8,77] frac[8,11] frac[8,17] frac[9,26] frac[9,107]
 frac[9,77] frac[9,14] frac[9,3] frac[9,82] frac[9,52] frac[9,9]
 frac[9,111] frac[9,25] frac[9,91] frac[9,20] frac[9,92] frac[9,29]
 frac[9,95] frac[9,18] frac[9,11] frac[9,24] frac[9,105] frac[9,115]
 frac[9,121] frac[9,22] frac[9,73] frac[9,28] frac[9,17] frac[9,32]
 frac[9,68] frac[9,89]
Generals
 cores_sketch_CPU5 cores_sketch_CPU7 cores_sketch_CPU9 cores_sketch_CPU16
 cores_sketch_CPU17 cores_sketch_CPU22 cores_sketch_CPU23
 cores_sketch_CPU25
General Constraints
 mem_access_time_CPU5: m_access_time_CPU5 = PWL ( mem_tot_CPU5 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU7: m_access_time_CPU7 = PWL ( mem_tot_CPU7 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU9: m_access_time_CPU9 = PWL ( mem_tot_CPU9 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU16: m_access_time_CPU16 = PWL ( mem_tot_CPU16 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU17: m_access_time_CPU17 = PWL ( mem_tot_CPU17 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU22: m_access_time_CPU22 = PWL ( mem_tot_CPU22 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU23: m_access_time_CPU23 = PWL ( mem_tot_CPU23 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU25: m_access_time_CPU25 = PWL ( mem_tot_CPU25 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

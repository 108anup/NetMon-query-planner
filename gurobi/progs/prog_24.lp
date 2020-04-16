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
 cov_0_23: frac[8,23] >= 0.7
 cov_1_26: frac[1,26] + frac[3,26] + frac[8,26] >= 0.8
 cov_2_25: frac[8,25] + frac[3,25] + frac[4,25] >= 0.9
 cov_3_4: frac[5,4] + frac[3,4] + frac[8,4] >= 0.8
 cov_4_3: frac[8,3] + frac[1,3] + frac[6,3] >= 0.9
 cov_5_20: frac[0,20] + frac[3,20] + frac[8,20] >= 1
 cov_6_3: frac[8,3] + frac[5,3] + frac[4,3] >= 1
 cov_7_27: frac[4,27] + frac[1,27] + frac[8,27] >= 0.8
 cov_8_8: frac[4,8] + frac[1,8] + frac[8,8] >= 0.682476
 cov_9_9: frac[8,9] + frac[0,9] + frac[4,9] >= 0.8
 cov_10_14: frac[5,14] + frac[0,14] + frac[8,14] >= 1
 cov_11_21: frac[8,21] + frac[4,21] + frac[3,21] >= 0.8
 cov_12_10: frac[3,10] + frac[8,10] + frac[7,10] >= 0.8
 cov_13_10: frac[2,10] >= 0.9
 cov_14_9: frac[0,9] + frac[2,9] >= 0.9
 cov_15_28: frac[7,28] + frac[2,28] >= 0.8
 cov_16_19: frac[7,19] + frac[8,19] + frac[5,19] >= 0.9
 cov_17_26: frac[4,26] + frac[5,26] + frac[8,26] >= 0.8
 cov_18_18: frac[8,18] + frac[7,18] + frac[4,18] >= 0.7
 cov_19_2: frac[3,2] + frac[8,2] + frac[7,2] >= 0.7
 cov_20_18: frac[8,18] + frac[7,18] + frac[4,18] >= 1
 cov_21_13: frac[6,13] + frac[8,13] + frac[5,13] >= 1
 cov_22_25: frac[8,25] + frac[0,25] + frac[5,25] >= 0.9
 cov_23_8: frac[7,8] + frac[0,8] >= 0.682476
 cov_24_19: frac[4,19] + frac[8,19] + frac[6,19] >= 0.7
 cov_25_6: frac[7,6] + frac[0,6] >= 1
 cov_26_5: frac[8,5] + frac[5,5] + frac[6,5] >= 0.7
 cov_27_6: frac[3,6] + frac[8,6] >= 0.8
 cov_28_10: frac[7,10] + frac[2,10] >= 0.9
 cov_29_25: frac[8,25] + frac[1,25] + frac[3,25] >= 0.8
 cov_30_12: frac[8,12] + frac[5,12] + frac[6,12] >= 1
 cov_31_4: frac[6,4] + frac[7,4] + frac[8,4] >= 0.9
 cov_32_7: frac[3,7] + frac[2,7] + frac[8,7] >= 0.9
 cov_33_17: frac[0,17] + frac[4,17] + frac[8,17] >= 1
 cov_34_14: frac[5,14] + frac[7,14] + frac[8,14] >= 0.7
 cov_35_6: frac[2,6] + frac[0,6] >= 0.9
 cov_36_7: frac[5,7] + frac[8,7] + frac[1,7] >= 0.7
 cov_37_8: frac[7,8] + frac[6,8] + frac[8,8] >= 0.682476
 cov_38_14: frac[6,14] + frac[7,14] + frac[8,14] >= 0.7
 cov_39_3: frac[8,3] + frac[2,3] + frac[4,3] >= 0.8
 cov_40_24: frac[3,24] + frac[8,24] + frac[4,24] >= 1
 cov_41_27: frac[2,27] + frac[3,27] + frac[8,27] >= 0.8
 cov_42_2: frac[8,2] + frac[4,2] >= 0.9
 cov_43_6: frac[1,6] + frac[8,6] + frac[4,6] >= 0.9
 cov_44_13: frac[8,13] + frac[5,13] + frac[2,13] >= 1
 cov_45_18: frac[0,18] + frac[1,18] >= 0.9
 cov_46_28: frac[6,28] + frac[8,28] + frac[4,28] >= 0.8
 cov_47_2: frac[8,2] + frac[4,2] + frac[5,2] >= 0.8
 cov_48_7: frac[0,7] + frac[3,7] + frac[8,7] >= 0.9
 cov_49_6: frac[3,6] + frac[8,6] + frac[0,6] >= 1
 cov_50_16: frac[3,16] + frac[8,16] + frac[7,16] >= 0.7
 cov_51_28: frac[7,28] + frac[2,28] >= 0.8
 cov_52_27: frac[3,27] + frac[7,27] + frac[8,27] >= 0.8
 cov_53_7: frac[0,7] + frac[6,7] + frac[8,7] >= 1
 cov_54_18: frac[5,18] + frac[3,18] + frac[8,18] >= 1
 cov_55_10: frac[6,10] + frac[8,10] + frac[7,10] >= 0.9
 cov_56_9: frac[8,9] + frac[3,9] + frac[7,9] >= 0.8
 cov_57_16: frac[4,16] + frac[6,16] + frac[8,16] >= 0.7
 cov_58_10: frac[1,10] + frac[2,10] >= 0.9
 cov_59_25: frac[8,25] + frac[0,25] + frac[5,25] >= 0.8
 cov_60_1: frac[7,1] >= 0.7
 cov_61_29: frac[8,29] >= 0.2
 cov_62_11: frac[8,11] >= 0.7
 cov_63_22: frac[8,22] >= 0.7
 cov_64_0: frac[8,0] >= 0.2
 cov_65_0: frac[8,0] >= 0.2
 cov_66_30: frac[8,30] >= 0.2
 cov_67_15: frac[8,15] >= 0.2
 accuracy_0_20: - 5.09765625 frac[0,20] + mem[0,20] = 0
 accuracy_6_28: - 5.09765625 frac[6,28] + mem[6,28] = 0
 accuracy_0_7: - 5.09765625 frac[0,7] + mem[0,7] = 0
 accuracy_1_6: - 5.09765625 frac[1,6] + mem[1,6] = 0
 accuracy_3_7: - 5.09765625 frac[3,7] + mem[3,7] = 0
 accuracy_8_5: - 5.09765625 frac[8,5] + mem[8,5] = 0
 accuracy_3_24: - 5.09765625 frac[3,24] + mem[3,24] = 0
 accuracy_8_24: - 5.09765625 frac[8,24] + mem[8,24] = 0
 accuracy_6_7: - 5.09765625 frac[6,7] + mem[6,7] = 0
 accuracy_5_5: - 5.09765625 frac[5,5] + mem[5,5] = 0
 accuracy_4_19: - 5.09765625 frac[4,19] + mem[4,19] = 0
 accuracy_7_6: - 5.09765625 frac[7,6] + mem[7,6] = 0
 accuracy_6_10: - 5.09765625 frac[6,10] + mem[6,10] = 0
 accuracy_5_18: - 5.09765625 frac[5,18] + mem[5,18] = 0
 accuracy_0_17: - 5.09765625 frac[0,17] + mem[0,17] = 0
 accuracy_7_19: - 5.09765625 frac[7,19] + mem[7,19] = 0
 accuracy_7_28: - 5.09765625 frac[7,28] + mem[7,28] = 0
 accuracy_2_27: - 5.09765625 frac[2,27] + mem[2,27] = 0
 accuracy_8_15: - 5.09765625 frac[8,15] + mem[8,15] = 0
 accuracy_3_2: - 5.09765625 frac[3,2] + mem[3,2] = 0
 accuracy_2_6: - 5.09765625 frac[2,6] + mem[2,6] = 0
 accuracy_8_2: - 5.09765625 frac[8,2] + mem[8,2] = 0
 accuracy_8_21: - 5.09765625 frac[8,21] + mem[8,21] = 0
 accuracy_4_16: - 5.09765625 frac[4,16] + mem[4,16] = 0
 accuracy_2_28: - 5.09765625 frac[2,28] + mem[2,28] = 0
 accuracy_8_12: - 5.09765625 frac[8,12] + mem[8,12] = 0
 accuracy_5_14: - 5.09765625 frac[5,14] + mem[5,14] = 0
 accuracy_4_26: - 5.09765625 frac[4,26] + mem[4,26] = 0
 accuracy_3_18: - 5.09765625 frac[3,18] + mem[3,18] = 0
 accuracy_8_18: - 5.09765625 frac[8,18] + mem[8,18] = 0
 accuracy_6_13: - 5.09765625 frac[6,13] + mem[6,13] = 0
 accuracy_4_21: - 5.09765625 frac[4,21] + mem[4,21] = 0
 accuracy_7_8: - 5.09765625 frac[7,8] + mem[7,8] = 0
 accuracy_6_16: - 5.09765625 frac[6,16] + mem[6,16] = 0
 accuracy_1_26: - 5.09765625 frac[1,26] + mem[1,26] = 0
 accuracy_0_14: - 5.09765625 frac[0,14] + mem[0,14] = 0
 accuracy_8_9: - 5.09765625 frac[8,9] + mem[8,9] = 0
 accuracy_8_28: - 5.09765625 frac[8,28] + mem[8,28] = 0
 accuracy_7_2: - 5.09765625 frac[7,2] + mem[7,2] = 0
 accuracy_6_14: - 5.09765625 frac[6,14] + mem[6,14] = 0
 accuracy_3_6: - 5.09765625 frac[3,6] + mem[3,6] = 0
 accuracy_1_10: - 5.09765625 frac[1,10] + mem[1,10] = 0
 accuracy_8_6: - 5.09765625 frac[8,6] + mem[8,6] = 0
 accuracy_3_27: - 5.09765625 frac[3,27] + mem[3,27] = 0
 accuracy_8_25: - 5.09765625 frac[8,25] + mem[8,25] = 0
 accuracy_6_4: - 5.09765625 frac[6,4] + mem[6,4] = 0
 accuracy_5_4: - 5.09765625 frac[5,4] + mem[5,4] = 0
 accuracy_4_28: - 5.09765625 frac[4,28] + mem[4,28] = 0
 accuracy_7_1: - 5.09765625 frac[7,1] + mem[7,1] = 0
 accuracy_0_18: - 5.09765625 frac[0,18] + mem[0,18] = 0
 accuracy_7_18: - 5.09765625 frac[7,18] + mem[7,18] = 0
 accuracy_0_8: - 5.09765625 frac[0,8] + mem[0,8] = 0
 accuracy_2_7: - 5.09765625 frac[2,7] + mem[2,7] = 0
 accuracy_8_3: - 5.09765625 frac[8,3] + mem[8,3] = 0
 accuracy_4_6: - 5.09765625 frac[4,6] + mem[4,6] = 0
 accuracy_8_22: - 5.09765625 frac[8,22] + mem[8,22] = 0
 accuracy_5_7: - 5.09765625 frac[5,7] + mem[5,7] = 0
 accuracy_4_17: - 5.09765625 frac[4,17] + mem[4,17] = 0
 accuracy_7_4: - 5.09765625 frac[7,4] + mem[7,4] = 0
 accuracy_1_3: - 5.09765625 frac[1,3] + mem[1,3] = 0
 accuracy_8_13: - 5.09765625 frac[8,13] + mem[8,13] = 0
 accuracy_4_8: - 5.09765625 frac[4,8] + mem[4,8] = 0
 accuracy_8_0: - 5.09765625 frac[8,0] + mem[8,0] = 0
 accuracy_5_13: - 5.09765625 frac[5,13] + mem[5,13] = 0
 accuracy_4_27: - 5.09765625 frac[4,27] + mem[4,27] = 0
 accuracy_3_21: - 5.09765625 frac[3,21] + mem[3,21] = 0
 accuracy_8_19: - 5.09765625 frac[8,19] + mem[8,19] = 0
 accuracy_5_26: - 5.09765625 frac[5,26] + mem[5,26] = 0
 accuracy_1_25: - 5.09765625 frac[1,25] + mem[1,25] = 0
 accuracy_3_10: - 5.09765625 frac[3,10] + mem[3,10] = 0
 accuracy_8_10: - 5.09765625 frac[8,10] + mem[8,10] = 0
 accuracy_2_13: - 5.09765625 frac[2,13] + mem[2,13] = 0
 accuracy_8_29: - 5.09765625 frac[8,29] + mem[8,29] = 0
 accuracy_4_24: - 5.09765625 frac[4,24] + mem[4,24] = 0
 accuracy_3_16: - 5.09765625 frac[3,16] + mem[3,16] = 0
 accuracy_8_16: - 5.09765625 frac[8,16] + mem[8,16] = 0
 accuracy_7_14: - 5.09765625 frac[7,14] + mem[7,14] = 0
 accuracy_0_25: - 5.09765625 frac[0,25] + mem[0,25] = 0
 accuracy_7_27: - 5.09765625 frac[7,27] + mem[7,27] = 0
 accuracy_3_9: - 5.09765625 frac[3,9] + mem[3,9] = 0
 accuracy_2_3: - 5.09765625 frac[2,3] + mem[2,3] = 0
 accuracy_8_7: - 5.09765625 frac[8,7] + mem[8,7] = 0
 accuracy_4_2: - 5.09765625 frac[4,2] + mem[4,2] = 0
 accuracy_3_26: - 5.09765625 frac[3,26] + mem[3,26] = 0
 accuracy_8_26: - 5.09765625 frac[8,26] + mem[8,26] = 0
 accuracy_6_5: - 5.09765625 frac[6,5] + mem[6,5] = 0
 accuracy_5_3: - 5.09765625 frac[5,3] + mem[5,3] = 0
 accuracy_6_8: - 5.09765625 frac[6,8] + mem[6,8] = 0
 accuracy_1_18: - 5.09765625 frac[1,18] + mem[1,18] = 0
 accuracy_0_6: - 5.09765625 frac[0,6] + mem[0,6] = 0
 accuracy_1_7: - 5.09765625 frac[1,7] + mem[1,7] = 0
 accuracy_0_9: - 5.09765625 frac[0,9] + mem[0,9] = 0
 accuracy_3_4: - 5.09765625 frac[3,4] + mem[3,4] = 0
 accuracy_8_4: - 5.09765625 frac[8,4] + mem[8,4] = 0
 accuracy_3_25: - 5.09765625 frac[3,25] + mem[3,25] = 0
 accuracy_8_23: - 5.09765625 frac[8,23] + mem[8,23] = 0
 accuracy_4_18: - 5.09765625 frac[4,18] + mem[4,18] = 0
 accuracy_5_19: - 5.09765625 frac[5,19] + mem[5,19] = 0
 accuracy_7_16: - 5.09765625 frac[7,16] + mem[7,16] = 0
 accuracy_8_14: - 5.09765625 frac[8,14] + mem[8,14] = 0
 accuracy_4_9: - 5.09765625 frac[4,9] + mem[4,9] = 0
 accuracy_2_9: - 5.09765625 frac[2,9] + mem[2,9] = 0
 accuracy_5_12: - 5.09765625 frac[5,12] + mem[5,12] = 0
 accuracy_3_20: - 5.09765625 frac[3,20] + mem[3,20] = 0
 accuracy_8_20: - 5.09765625 frac[8,20] + mem[8,20] = 0
 accuracy_6_3: - 5.09765625 frac[6,3] + mem[6,3] = 0
 accuracy_5_25: - 5.09765625 frac[5,25] + mem[5,25] = 0
 accuracy_7_10: - 5.09765625 frac[7,10] + mem[7,10] = 0
 accuracy_8_11: - 5.09765625 frac[8,11] + mem[8,11] = 0
 accuracy_2_10: - 5.09765625 frac[2,10] + mem[2,10] = 0
 accuracy_8_30: - 5.09765625 frac[8,30] + mem[8,30] = 0
 accuracy_4_25: - 5.09765625 frac[4,25] + mem[4,25] = 0
 accuracy_8_17: - 5.09765625 frac[8,17] + mem[8,17] = 0
 accuracy_6_12: - 5.09765625 frac[6,12] + mem[6,12] = 0
 accuracy_7_9: - 5.09765625 frac[7,9] + mem[7,9] = 0
 accuracy_6_19: - 5.09765625 frac[6,19] + mem[6,19] = 0
 accuracy_1_27: - 5.09765625 frac[1,27] + mem[1,27] = 0
 accuracy_1_8: - 5.09765625 frac[1,8] + mem[1,8] = 0
 accuracy_8_8: - 5.09765625 frac[8,8] + mem[8,8] = 0
 accuracy_4_3: - 5.09765625 frac[4,3] + mem[4,3] = 0
 accuracy_8_27: - 5.09765625 frac[8,27] + mem[8,27] = 0
 accuracy_5_2: - 5.09765625 frac[5,2] + mem[5,2] = 0
 mem_tot_CPU5: - mem[0,20] - mem[0,7] - mem[0,17] - mem[0,14] - mem[0,18]
   - mem[0,8] - mem[0,25] - mem[0,6] - mem[0,9] + mem_tot_CPU5 = 0
 R191: - 3.05176e-05 mem_tot_CPU5 + normalized_mem_tot_CPU5 = 0
 rows_tot_CPU5: - 3 frac[0,20] - 3 frac[0,7] - 3 frac[0,17] - 3 frac[0,14]
   - 3 frac[0,18] - 3 frac[0,8] - 3 frac[0,25] - 3 frac[0,6] - 3 frac[0,9]
   + rows_tot_CPU5 = 0
 R193: - 0.0833333333333333 rows_tot_CPU5 + normalized_rows_tot_CPU5 = 0
 mem_tot_CPU9: - mem[1,6] - mem[1,26] - mem[1,10] - mem[1,3] - mem[1,25]
   - mem[1,18] - mem[1,7] - mem[1,27] - mem[1,8] + mem_tot_CPU9 = 0
 R195: - 3.05176e-05 mem_tot_CPU9 + normalized_mem_tot_CPU9 = 0
 rows_tot_CPU9: - 3 frac[1,6] - 3 frac[1,26] - 3 frac[1,10] - 3 frac[1,3]
   - 3 frac[1,25] - 3 frac[1,18] - 3 frac[1,7] - 3 frac[1,27] - 3 frac[1,8]
   + rows_tot_CPU9 = 0
 R197: - 0.0833333333333333 rows_tot_CPU9 + normalized_rows_tot_CPU9 = 0
 mem_tot_CPU16: - mem[2,27] - mem[2,6] - mem[2,28] - mem[2,7] - mem[2,13]
   - mem[2,3] - mem[2,9] - mem[2,10] + mem_tot_CPU16 = 0
 R199: - 3.05176e-05 mem_tot_CPU16 + normalized_mem_tot_CPU16 = 0
 rows_tot_CPU16: - 3 frac[2,27] - 3 frac[2,6] - 3 frac[2,28] - 3 frac[2,7]
   - 3 frac[2,13] - 3 frac[2,3] - 3 frac[2,9] - 3 frac[2,10]
   + rows_tot_CPU16 = 0
 R201: - 0.0833333333333333 rows_tot_CPU16 + normalized_rows_tot_CPU16 = 0
 mem_tot_CPU19: - mem[3,7] - mem[3,24] - mem[3,2] - mem[3,18] - mem[3,6]
   - mem[3,27] - mem[3,21] - mem[3,10] - mem[3,16] - mem[3,9] - mem[3,26]
   - mem[3,4] - mem[3,25] - mem[3,20] + mem_tot_CPU19 = 0
 R203: - 3.05176e-05 mem_tot_CPU19 + normalized_mem_tot_CPU19 = 0
 rows_tot_CPU19: - 3 frac[3,7] - 3 frac[3,24] - 3 frac[3,2] - 3 frac[3,18]
   - 3 frac[3,6] - 3 frac[3,27] - 3 frac[3,21] - 3 frac[3,10]
   - 3 frac[3,16] - 3 frac[3,9] - 3 frac[3,26] - 3 frac[3,4] - 3 frac[3,25]
   - 3 frac[3,20] + rows_tot_CPU19 = 0
 R205: - 0.0833333333333333 rows_tot_CPU19 + normalized_rows_tot_CPU19 = 0
 mem_tot_CPU20: - mem[4,19] - mem[4,16] - mem[4,26] - mem[4,21] - mem[4,28]
   - mem[4,6] - mem[4,17] - mem[4,8] - mem[4,27] - mem[4,24] - mem[4,2]
   - mem[4,18] - mem[4,9] - mem[4,25] - mem[4,3] + mem_tot_CPU20 = 0
 R207: - 3.05176e-05 mem_tot_CPU20 + normalized_mem_tot_CPU20 = 0
 rows_tot_CPU20: - 3 frac[4,19] - 3 frac[4,16] - 3 frac[4,26]
   - 3 frac[4,21] - 3 frac[4,28] - 3 frac[4,6] - 3 frac[4,17] - 3 frac[4,8]
   - 3 frac[4,27] - 3 frac[4,24] - 3 frac[4,2] - 3 frac[4,18] - 3 frac[4,9]
   - 3 frac[4,25] - 3 frac[4,3] + rows_tot_CPU20 = 0
 R209: - 0.0833333333333333 rows_tot_CPU20 + normalized_rows_tot_CPU20 = 0
 mem_tot_CPU22: - mem[5,5] - mem[5,18] - mem[5,14] - mem[5,4] - mem[5,7]
   - mem[5,13] - mem[5,26] - mem[5,3] - mem[5,19] - mem[5,12] - mem[5,25]
   - mem[5,2] + mem_tot_CPU22 = 0
 R211: - 3.05176e-05 mem_tot_CPU22 + normalized_mem_tot_CPU22 = 0
 rows_tot_CPU22: - 3 frac[5,5] - 3 frac[5,18] - 3 frac[5,14] - 3 frac[5,4]
   - 3 frac[5,7] - 3 frac[5,13] - 3 frac[5,26] - 3 frac[5,3] - 3 frac[5,19]
   - 3 frac[5,12] - 3 frac[5,25] - 3 frac[5,2] + rows_tot_CPU22 = 0
 R213: - 0.0833333333333333 rows_tot_CPU22 + normalized_rows_tot_CPU22 = 0
 mem_tot_CPU23: - mem[6,28] - mem[6,7] - mem[6,10] - mem[6,13] - mem[6,16]
   - mem[6,14] - mem[6,4] - mem[6,5] - mem[6,8] - mem[6,3] - mem[6,12]
   - mem[6,19] + mem_tot_CPU23 = 0
 R215: - 3.05176e-05 mem_tot_CPU23 + normalized_mem_tot_CPU23 = 0
 rows_tot_CPU23: - 3 frac[6,28] - 3 frac[6,7] - 3 frac[6,10] - 3 frac[6,13]
   - 3 frac[6,16] - 3 frac[6,14] - 3 frac[6,4] - 3 frac[6,5] - 3 frac[6,8]
   - 3 frac[6,3] - 3 frac[6,12] - 3 frac[6,19] + rows_tot_CPU23 = 0
 R217: - 0.0833333333333333 rows_tot_CPU23 + normalized_rows_tot_CPU23 = 0
 mem_tot_CPU30: - mem[7,6] - mem[7,19] - mem[7,28] - mem[7,8] - mem[7,2]
   - mem[7,1] - mem[7,18] - mem[7,4] - mem[7,14] - mem[7,27] - mem[7,16]
   - mem[7,10] - mem[7,9] + mem_tot_CPU30 = 0
 R219: - 3.05176e-05 mem_tot_CPU30 + normalized_mem_tot_CPU30 = 0
 rows_tot_CPU30: - 3 frac[7,6] - 3 frac[7,19] - 3 frac[7,28] - 3 frac[7,8]
   - 3 frac[7,2] - 3 frac[7,1] - 3 frac[7,18] - 3 frac[7,4] - 3 frac[7,14]
   - 3 frac[7,27] - 3 frac[7,16] - 3 frac[7,10] - 3 frac[7,9]
   + rows_tot_CPU30 = 0
 R221: - 0.0833333333333333 rows_tot_CPU30 + normalized_rows_tot_CPU30 = 0
 mem_tot_tor_P43: - mem[8,5] - mem[8,24] - mem[8,15] - mem[8,2] - mem[8,21]
   - mem[8,12] - mem[8,18] - mem[8,9] - mem[8,28] - mem[8,6] - mem[8,25]
   - mem[8,3] - mem[8,22] - mem[8,13] - mem[8,0] - mem[8,19] - mem[8,10]
   - mem[8,29] - mem[8,16] - mem[8,7] - mem[8,26] - mem[8,4] - mem[8,23]
   - mem[8,14] - mem[8,20] - mem[8,11] - mem[8,30] - mem[8,17] - mem[8,8]
   - mem[8,27] + mem_tot_tor_P43 = 0
 R223: - 0.00173611111111111 mem_tot_tor_P43 + normalized_mem_tot_tor_P43
   = 0
 rows_tot_tor_P43: - 3 frac[8,5] - 3 frac[8,24] - 3 frac[8,15]
   - 3 frac[8,2] - 3 frac[8,21] - 3 frac[8,12] - 3 frac[8,18] - 3 frac[8,9]
   - 3 frac[8,28] - 3 frac[8,6] - 3 frac[8,25] - 3 frac[8,3] - 3 frac[8,22]
   - 3 frac[8,13] - 3 frac[8,0] - 3 frac[8,19] - 3 frac[8,10]
   - 3 frac[8,29] - 3 frac[8,16] - 3 frac[8,7] - 3 frac[8,26] - 3 frac[8,4]
   - 3 frac[8,23] - 3 frac[8,14] - 3 frac[8,20] - 3 frac[8,11]
   - 3 frac[8,30] - 3 frac[8,17] - 3 frac[8,8] - 3 frac[8,27]
   + rows_tot_tor_P43 = 0
 R225: - 0.0208333333333333 rows_tot_tor_P43 + normalized_rows_tot_tor_P43
   = 0
 row_capacity_CPU5: rows_tot_CPU5 <= 12
 row_capacity_CPU9: rows_tot_CPU9 <= 12
 row_capacity_CPU16: rows_tot_CPU16 <= 12
 row_capacity_CPU19: rows_tot_CPU19 <= 12
 row_capacity_CPU20: rows_tot_CPU20 <= 12
 row_capacity_CPU22: rows_tot_CPU22 <= 12
 row_capacity_CPU23: rows_tot_CPU23 <= 12
 row_capacity_CPU30: rows_tot_CPU30 <= 12
 row_capacity_tor_P43: rows_tot_tor_P43 <= 48
 ns_single_CPU5: 3.5 rows_tot_CPU5 - ns_single_CPU5 + pdt_m_rows_CPU5 = 0
 capacity_cores_CPU5: cores_sketch_CPU5 + cores_dpdk_CPU5 <= 7
 ns_sketch_CPU5: - ns_single_CPU5 + 11.42857142857143 cores_sketch_CPU5
   >= 0
 ns_dpdk_CPU5: - 4.285714285714287 cores_dpdk_CPU5 <= -21.42857142857143
 ns_single_CPU9: 3.5 rows_tot_CPU9 - ns_single_CPU9 + pdt_m_rows_CPU9 = 0
 capacity_cores_CPU9: cores_sketch_CPU9 + cores_dpdk_CPU9 <= 7
 ns_sketch_CPU9: - ns_single_CPU9 + 11.42857142857143 cores_sketch_CPU9
   >= 0
 ns_dpdk_CPU9: - 4.285714285714287 cores_dpdk_CPU9 <= -21.42857142857143
 ns_single_CPU16: 3.5 rows_tot_CPU16 - ns_single_CPU16 + pdt_m_rows_CPU16
   = 0
 capacity_cores_CPU16: cores_sketch_CPU16 + cores_dpdk_CPU16 <= 7
 ns_sketch_CPU16: - ns_single_CPU16 + 11.42857142857143 cores_sketch_CPU16
   >= 0
 ns_dpdk_CPU16: - 4.285714285714287 cores_dpdk_CPU16 <= -21.42857142857143
 ns_single_CPU19: 3.5 rows_tot_CPU19 - ns_single_CPU19 + pdt_m_rows_CPU19
   = 0
 capacity_cores_CPU19: cores_sketch_CPU19 + cores_dpdk_CPU19 <= 7
 ns_sketch_CPU19: - ns_single_CPU19 + 11.42857142857143 cores_sketch_CPU19
   >= 0
 ns_dpdk_CPU19: - 4.285714285714287 cores_dpdk_CPU19 <= -21.42857142857143
 ns_single_CPU20: 3.5 rows_tot_CPU20 - ns_single_CPU20 + pdt_m_rows_CPU20
   = 0
 capacity_cores_CPU20: cores_sketch_CPU20 + cores_dpdk_CPU20 <= 7
 ns_sketch_CPU20: - ns_single_CPU20 + 11.42857142857143 cores_sketch_CPU20
   >= 0
 ns_dpdk_CPU20: - 4.285714285714287 cores_dpdk_CPU20 <= -21.42857142857143
 ns_single_CPU22: 3.5 rows_tot_CPU22 - ns_single_CPU22 + pdt_m_rows_CPU22
   = 0
 capacity_cores_CPU22: cores_sketch_CPU22 + cores_dpdk_CPU22 <= 7
 ns_sketch_CPU22: - ns_single_CPU22 + 11.42857142857143 cores_sketch_CPU22
   >= 0
 ns_dpdk_CPU22: - 4.285714285714287 cores_dpdk_CPU22 <= -21.42857142857143
 ns_single_CPU23: 3.5 rows_tot_CPU23 - ns_single_CPU23 + pdt_m_rows_CPU23
   = 0
 capacity_cores_CPU23: cores_sketch_CPU23 + cores_dpdk_CPU23 <= 7
 ns_sketch_CPU23: - ns_single_CPU23 + 11.42857142857143 cores_sketch_CPU23
   >= 0
 ns_dpdk_CPU23: - 4.285714285714287 cores_dpdk_CPU23 <= -21.42857142857143
 ns_single_CPU30: 3.5 rows_tot_CPU30 - ns_single_CPU30 + pdt_m_rows_CPU30
   = 0
 capacity_cores_CPU30: cores_sketch_CPU30 + cores_dpdk_CPU30 <= 7
 ns_sketch_CPU30: - ns_single_CPU30 + 11.42857142857143 cores_sketch_CPU30
   >= 0
 ns_dpdk_CPU30: - 4.285714285714287 cores_dpdk_CPU30 <= -21.42857142857143
 ns_tor_P43: ns_tor_P43 = 6.756756756756757
 max_ns_tor_P43: ns_tor_P43 <= 11.42857142857143
 res_acc: 1.220703125e-04 mem_tot_CPU5 + 1.220703125e-04 mem_tot_CPU9
   + 1.220703125e-04 mem_tot_CPU16 + 1.220703125e-04 mem_tot_CPU19
   + 1.220703125e-04 mem_tot_CPU20 + 1.220703125e-04 mem_tot_CPU22
   + 1.220703125e-04 mem_tot_CPU23 + 1.220703125e-04 mem_tot_CPU30
   + 0.0208333333333333 mem_tot_tor_P43 + 0.25 rows_tot_tor_P43
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
 frac[0,20] <= 1
 frac[6,28] <= 1
 frac[0,7] <= 1
 frac[1,6] <= 1
 frac[3,7] <= 1
 frac[8,5] <= 1
 frac[3,24] <= 1
 frac[8,24] <= 1
 frac[6,7] <= 1
 frac[5,5] <= 1
 frac[4,19] <= 1
 frac[7,6] <= 1
 frac[6,10] <= 1
 frac[5,18] <= 1
 frac[0,17] <= 1
 frac[7,19] <= 1
 frac[7,28] <= 1
 frac[2,27] <= 1
 frac[8,15] <= 1
 frac[3,2] <= 1
 frac[2,6] <= 1
 frac[8,2] <= 1
 frac[8,21] <= 1
 frac[4,16] <= 1
 frac[2,28] <= 1
 frac[8,12] <= 1
 frac[5,14] <= 1
 frac[4,26] <= 1
 frac[3,18] <= 1
 frac[8,18] <= 1
 frac[6,13] <= 1
 frac[4,21] <= 1
 frac[7,8] <= 1
 frac[6,16] <= 1
 frac[1,26] <= 1
 frac[0,14] <= 1
 frac[8,9] <= 1
 frac[8,28] <= 1
 frac[7,2] <= 1
 frac[6,14] <= 1
 frac[3,6] <= 1
 frac[1,10] <= 1
 frac[8,6] <= 1
 frac[3,27] <= 1
 frac[8,25] <= 1
 frac[6,4] <= 1
 frac[5,4] <= 1
 frac[4,28] <= 1
 frac[7,1] <= 1
 frac[0,18] <= 1
 frac[7,18] <= 1
 frac[0,8] <= 1
 frac[2,7] <= 1
 frac[8,3] <= 1
 frac[4,6] <= 1
 frac[8,22] <= 1
 frac[5,7] <= 1
 frac[4,17] <= 1
 frac[7,4] <= 1
 frac[1,3] <= 1
 frac[8,13] <= 1
 frac[4,8] <= 1
 frac[8,0] <= 1
 frac[5,13] <= 1
 frac[4,27] <= 1
 frac[3,21] <= 1
 frac[8,19] <= 1
 frac[5,26] <= 1
 frac[1,25] <= 1
 frac[3,10] <= 1
 frac[8,10] <= 1
 frac[2,13] <= 1
 frac[8,29] <= 1
 frac[4,24] <= 1
 frac[3,16] <= 1
 frac[8,16] <= 1
 frac[7,14] <= 1
 frac[0,25] <= 1
 frac[7,27] <= 1
 frac[3,9] <= 1
 frac[2,3] <= 1
 frac[8,7] <= 1
 frac[4,2] <= 1
 frac[3,26] <= 1
 frac[8,26] <= 1
 frac[6,5] <= 1
 frac[5,3] <= 1
 frac[6,8] <= 1
 frac[1,18] <= 1
 frac[0,6] <= 1
 frac[1,7] <= 1
 frac[0,9] <= 1
 frac[3,4] <= 1
 frac[8,4] <= 1
 frac[3,25] <= 1
 frac[8,23] <= 1
 frac[4,18] <= 1
 frac[5,19] <= 1
 frac[7,16] <= 1
 frac[8,14] <= 1
 frac[4,9] <= 1
 frac[2,9] <= 1
 frac[5,12] <= 1
 frac[3,20] <= 1
 frac[8,20] <= 1
 frac[6,3] <= 1
 frac[5,25] <= 1
 frac[7,10] <= 1
 frac[8,11] <= 1
 frac[2,10] <= 1
 frac[8,30] <= 1
 frac[4,25] <= 1
 frac[8,17] <= 1
 frac[6,12] <= 1
 frac[7,9] <= 1
 frac[6,19] <= 1
 frac[1,27] <= 1
 frac[1,8] <= 1
 frac[8,8] <= 1
 frac[4,3] <= 1
 frac[8,27] <= 1
 frac[5,2] <= 1
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
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_tor_P43 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU5 , normalized_mem_tot_CPU9 ,
   normalized_mem_tot_CPU16 , normalized_mem_tot_CPU19 ,
   normalized_mem_tot_CPU20 , normalized_mem_tot_CPU22 ,
   normalized_mem_tot_CPU23 , normalized_mem_tot_CPU30 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_tor_P43 )
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

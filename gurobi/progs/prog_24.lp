\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_P4 + 0 max_mem_CPU_Cluster + 0 max_rows_P4
   + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU3 + 0 m_access_time_CPU5
   + 0 m_access_time_CPU6 + 0 m_access_time_CPU7 + 0 m_access_time_CPU10
   + 0 m_access_time_CPU26 + 0 m_access_time_CPU28 + 0 m_access_time_CPU29
   + res
Subject To
 cov_0_25: frac[7,25] + frac[8,25] + frac[2,25] >= 0.9
 cov_1_18: frac[5,18] + frac[7,18] >= 0.9
 cov_2_25: frac[8,25] + frac[2,25] + frac[3,25] >= 0.8
 cov_3_2: frac[7,2] + frac[4,2] >= 0.7
 cov_4_20: frac[1,20] + frac[7,20] + frac[8,20] >= 0.7
 cov_5_13: frac[8,13] + frac[1,13] + frac[0,13] >= 0.9
 cov_6_15: frac[8,15] + frac[6,15] + frac[2,15] >= 0.9
 cov_7_12: frac[1,12] + frac[8,12] + frac[2,12] >= 1
 cov_8_22: frac[7,22] + frac[8,22] + frac[0,22] >= 0.9
 cov_9_0: frac[4,0] + frac[5,0] >= 0.8
 cov_10_10: frac[0,10] + frac[1,10] + frac[8,10] >= 0.9
 cov_11_20: frac[0,20] + frac[6,20] + frac[8,20] >= 0.7
 cov_12_18: frac[8,18] + frac[2,18] + frac[1,18] >= 0.8
 cov_13_24: frac[8,24] + frac[0,24] + frac[1,24] >= 0.7
 cov_14_6: frac[8,6] + frac[4,6] + frac[0,6] >= 0.8
 cov_15_22: frac[2,22] + frac[8,22] + frac[0,22] >= 0.9
 cov_16_26: frac[3,26] + frac[8,26] + frac[2,26] >= 0.8
 cov_17_5: frac[8,5] + frac[5,5] + frac[0,5] >= 0.7
 cov_18_0: frac[4,0] + frac[6,0] >= 0.8
 cov_19_16: frac[3,16] + frac[8,16] + frac[0,16] >= 0.7
 cov_20_2: frac[8,2] + frac[2,2] + frac[6,2] >= 0.7
 cov_21_23: frac[3,23] + frac[8,23] + frac[0,23] >= 0.9
 cov_22_0: frac[6,0] + frac[5,0] >= 0.8
 cov_23_25: frac[7,25] + frac[8,25] + frac[0,25] >= 1
 cov_24_5: frac[8,5] + frac[7,5] + frac[1,5] >= 1
 cov_25_23: frac[2,23] + frac[8,23] + frac[4,23] >= 0.9
 cov_26_7: frac[0,7] + frac[8,7] + frac[1,7] >= 0.7
 cov_27_12: frac[7,12] + frac[6,12] >= 0.9
 cov_28_27: frac[0,27] + frac[1,27] + frac[8,27] >= 0.7
 cov_29_23: frac[8,23] + frac[7,23] + frac[0,23] >= 1
 cov_30_26: frac[6,26] + frac[7,26] >= 0.8
 cov_31_11: frac[1,11] + frac[3,11] + frac[8,11] >= 1
 cov_32_7: frac[4,7] + frac[7,7] >= 0.8
 cov_33_9: frac[8,9] + frac[0,9] + frac[2,9] >= 0.8
 cov_34_3: frac[8,3] + frac[0,3] + frac[6,3] >= 0.9
 cov_35_22: frac[6,22] + frac[5,22] >= 0.8
 cov_36_3: frac[8,3] + frac[0,3] + frac[3,3] >= 0.7
 cov_37_4: frac[0,4] + frac[3,4] + frac[8,4] >= 0.8
 cov_38_19: frac[8,19] + frac[0,19] + frac[3,19] >= 1
 cov_39_0: frac[3,0] + frac[8,0] + frac[2,0] >= 0.8
 cov_40_5: frac[8,5] + frac[1,5] + frac[6,5] >= 0.9
 cov_41_16: frac[6,16] + frac[1,16] + frac[8,16] >= 0.7
 cov_42_15: frac[7,15] + frac[6,15] >= 0.9
 cov_43_8: frac[2,8] + frac[1,8] + frac[8,8] >= 0.8
 cov_44_9: frac[6,9] + frac[7,9] >= 0.9
 cov_45_20: frac[0,20] + frac[3,20] + frac[8,20] >= 1
 cov_46_18: frac[3,18] + frac[8,18] + frac[4,18] >= 0.7
 cov_47_27: frac[3,27] + frac[7,27] + frac[8,27] >= 0.9
 cov_48_19: frac[8,19] + frac[3,19] + frac[6,19] >= 0.8
 cov_49_6: frac[3,6] + frac[8,6] + frac[0,6] >= 0.7
 cov_50_25: frac[6,25] + frac[8,25] + frac[0,25] >= 0.7
 cov_51_17: frac[1,17] + frac[3,17] + frac[8,17] >= 0.8
 cov_52_27: frac[6,27] + frac[1,27] + frac[8,27] >= 0.7
 cov_53_5: frac[8,5] + frac[0,5] + frac[3,5] >= 0.7
 cov_54_27: frac[3,27] + frac[7,27] + frac[8,27] >= 1
 cov_55_15: frac[8,15] + frac[0,15] + frac[2,15] >= 0.7
 cov_56_14: frac[8,14] >= 0.9
 cov_57_21: frac[8,21] >= 0.8
 cov_58_1: frac[8,1] >= 0.7
 cov_59_21: frac[8,21] >= 0.8
 cov_60_14: frac[8,14] >= 0.8
 accuracy_6_9: - 5.09765625 frac[6,9] + mem[6,9] = 0
 accuracy_0_20: - 5.09765625 frac[0,20] + mem[0,20] = 0
 accuracy_7_12: - 5.09765625 frac[7,12] + mem[7,12] = 0
 accuracy_1_17: - 5.09765625 frac[1,17] + mem[1,17] = 0
 accuracy_0_7: - 5.09765625 frac[0,7] + mem[0,7] = 0
 accuracy_7_25: - 5.09765625 frac[7,25] + mem[7,25] = 0
 accuracy_2_22: - 5.09765625 frac[2,22] + mem[2,22] = 0
 accuracy_0_10: - 5.09765625 frac[0,10] + mem[0,10] = 0
 accuracy_1_11: - 5.09765625 frac[1,11] + mem[1,11] = 0
 accuracy_8_5: - 6 frac[8,5] + mem[8,5] = 0
 accuracy_4_0: - 5.09765625 frac[4,0] + mem[4,0] = 0
 accuracy_8_24: - 6 frac[8,24] + mem[8,24] = 0
 accuracy_5_5: - 5.09765625 frac[5,5] + mem[5,5] = 0
 accuracy_5_18: - 5.09765625 frac[5,18] + mem[5,18] = 0
 accuracy_6_25: - 5.09765625 frac[6,25] + mem[6,25] = 0
 accuracy_0_4: - 5.09765625 frac[0,4] + mem[0,4] = 0
 accuracy_8_15: - 6 frac[8,15] + mem[8,15] = 0
 accuracy_8_2: - 6 frac[8,2] + mem[8,2] = 0
 accuracy_3_23: - 5.09765625 frac[3,23] + mem[3,23] = 0
 accuracy_8_21: - 6 frac[8,21] + mem[8,21] = 0
 accuracy_6_0: - 5.09765625 frac[6,0] + mem[6,0] = 0
 accuracy_7_5: - 5.09765625 frac[7,5] + mem[7,5] = 0
 accuracy_7_22: - 5.09765625 frac[7,22] + mem[7,22] = 0
 accuracy_6_26: - 5.09765625 frac[6,26] + mem[6,26] = 0
 accuracy_1_12: - 5.09765625 frac[1,12] + mem[1,12] = 0
 accuracy_8_12: - 6 frac[8,12] + mem[8,12] = 0
 accuracy_3_18: - 5.09765625 frac[3,18] + mem[3,18] = 0
 accuracy_8_18: - 6 frac[8,18] + mem[8,18] = 0
 accuracy_6_16: - 5.09765625 frac[6,16] + mem[6,16] = 0
 accuracy_0_27: - 5.09765625 frac[0,27] + mem[0,27] = 0
 accuracy_2_18: - 5.09765625 frac[2,18] + mem[2,18] = 0
 accuracy_3_11: - 5.09765625 frac[3,11] + mem[3,11] = 0
 accuracy_8_9: - 6 frac[8,9] + mem[8,9] = 0
 accuracy_2_12: - 5.09765625 frac[2,12] + mem[2,12] = 0
 accuracy_3_17: - 5.09765625 frac[3,17] + mem[3,17] = 0
 accuracy_7_2: - 5.09765625 frac[7,2] + mem[7,2] = 0
 accuracy_7_15: - 5.09765625 frac[7,15] + mem[7,15] = 0
 accuracy_1_16: - 5.09765625 frac[1,16] + mem[1,16] = 0
 accuracy_0_24: - 5.09765625 frac[0,24] + mem[0,24] = 0
 accuracy_2_23: - 5.09765625 frac[2,23] + mem[2,23] = 0
 accuracy_1_5: - 5.09765625 frac[1,5] + mem[1,5] = 0
 accuracy_3_6: - 5.09765625 frac[3,6] + mem[3,6] = 0
 accuracy_2_2: - 5.09765625 frac[2,2] + mem[2,2] = 0
 accuracy_1_10: - 5.09765625 frac[1,10] + mem[1,10] = 0
 accuracy_8_6: - 6 frac[8,6] + mem[8,6] = 0
 accuracy_3_27: - 5.09765625 frac[3,27] + mem[3,27] = 0
 accuracy_8_25: - 6 frac[8,25] + mem[8,25] = 0
 accuracy_7_18: - 5.09765625 frac[7,18] + mem[7,18] = 0
 accuracy_0_5: - 5.09765625 frac[0,5] + mem[0,5] = 0
 accuracy_3_5: - 5.09765625 frac[3,5] + mem[3,5] = 0
 accuracy_8_3: - 6 frac[8,3] + mem[8,3] = 0
 accuracy_4_6: - 5.09765625 frac[4,6] + mem[4,6] = 0
 accuracy_8_22: - 6 frac[8,22] + mem[8,22] = 0
 accuracy_6_20: - 5.09765625 frac[6,20] + mem[6,20] = 0
 accuracy_6_27: - 5.09765625 frac[6,27] + mem[6,27] = 0
 accuracy_8_13: - 6 frac[8,13] + mem[8,13] = 0
 accuracy_3_0: - 5.09765625 frac[3,0] + mem[3,0] = 0
 accuracy_2_8: - 5.09765625 frac[2,8] + mem[2,8] = 0
 accuracy_8_0: - 6 frac[8,0] + mem[8,0] = 0
 accuracy_8_19: - 6 frac[8,19] + mem[8,19] = 0
 accuracy_6_2: - 5.09765625 frac[6,2] + mem[6,2] = 0
 accuracy_1_20: - 5.09765625 frac[1,20] + mem[1,20] = 0
 accuracy_7_20: - 5.09765625 frac[7,20] + mem[7,20] = 0
 accuracy_0_15: - 5.09765625 frac[0,15] + mem[0,15] = 0
 accuracy_8_10: - 6 frac[8,10] + mem[8,10] = 0
 accuracy_5_0: - 5.09765625 frac[5,0] + mem[5,0] = 0
 accuracy_3_16: - 5.09765625 frac[3,16] + mem[3,16] = 0
 accuracy_8_16: - 6 frac[8,16] + mem[8,16] = 0
 accuracy_6_15: - 5.09765625 frac[6,15] + mem[6,15] = 0
 accuracy_0_22: - 5.09765625 frac[0,22] + mem[0,22] = 0
 accuracy_0_25: - 5.09765625 frac[0,25] + mem[0,25] = 0
 accuracy_7_27: - 5.09765625 frac[7,27] + mem[7,27] = 0
 accuracy_8_7: - 6 frac[8,7] + mem[8,7] = 0
 accuracy_4_2: - 5.09765625 frac[4,2] + mem[4,2] = 0
 accuracy_3_26: - 5.09765625 frac[3,26] + mem[3,26] = 0
 accuracy_8_26: - 6 frac[8,26] + mem[8,26] = 0
 accuracy_6_5: - 5.09765625 frac[6,5] + mem[6,5] = 0
 accuracy_0_19: - 5.09765625 frac[0,19] + mem[0,19] = 0
 accuracy_1_18: - 5.09765625 frac[1,18] + mem[1,18] = 0
 accuracy_0_6: - 5.09765625 frac[0,6] + mem[0,6] = 0
 accuracy_2_25: - 5.09765625 frac[2,25] + mem[2,25] = 0
 accuracy_1_7: - 5.09765625 frac[1,7] + mem[1,7] = 0
 accuracy_0_9: - 5.09765625 frac[0,9] + mem[0,9] = 0
 accuracy_3_4: - 5.09765625 frac[3,4] + mem[3,4] = 0
 accuracy_8_4: - 6 frac[8,4] + mem[8,4] = 0
 accuracy_4_7: - 5.09765625 frac[4,7] + mem[4,7] = 0
 accuracy_3_25: - 5.09765625 frac[3,25] + mem[3,25] = 0
 accuracy_8_23: - 6 frac[8,23] + mem[8,23] = 0
 accuracy_4_18: - 5.09765625 frac[4,18] + mem[4,18] = 0
 accuracy_7_7: - 5.09765625 frac[7,7] + mem[7,7] = 0
 accuracy_0_16: - 5.09765625 frac[0,16] + mem[0,16] = 0
 accuracy_0_3: - 5.09765625 frac[0,3] + mem[0,3] = 0
 accuracy_2_26: - 5.09765625 frac[2,26] + mem[2,26] = 0
 accuracy_8_14: - 6 frac[8,14] + mem[8,14] = 0
 accuracy_3_3: - 5.09765625 frac[3,3] + mem[3,3] = 0
 accuracy_2_9: - 5.09765625 frac[2,9] + mem[2,9] = 0
 accuracy_8_1: - 6 frac[8,1] + mem[8,1] = 0
 accuracy_3_20: - 5.09765625 frac[3,20] + mem[3,20] = 0
 accuracy_8_20: - 6 frac[8,20] + mem[8,20] = 0
 accuracy_6_3: - 5.09765625 frac[6,3] + mem[6,3] = 0
 accuracy_4_23: - 5.09765625 frac[4,23] + mem[4,23] = 0
 accuracy_6_22: - 5.09765625 frac[6,22] + mem[6,22] = 0
 accuracy_5_22: - 5.09765625 frac[5,22] + mem[5,22] = 0
 accuracy_7_23: - 5.09765625 frac[7,23] + mem[7,23] = 0
 accuracy_1_24: - 5.09765625 frac[1,24] + mem[1,24] = 0
 accuracy_1_13: - 5.09765625 frac[1,13] + mem[1,13] = 0
 accuracy_8_11: - 6 frac[8,11] + mem[8,11] = 0
 accuracy_3_19: - 5.09765625 frac[3,19] + mem[3,19] = 0
 accuracy_8_17: - 6 frac[8,17] + mem[8,17] = 0
 accuracy_6_12: - 5.09765625 frac[6,12] + mem[6,12] = 0
 accuracy_0_23: - 5.09765625 frac[0,23] + mem[0,23] = 0
 accuracy_7_9: - 5.09765625 frac[7,9] + mem[7,9] = 0
 accuracy_6_19: - 5.09765625 frac[6,19] + mem[6,19] = 0
 accuracy_7_26: - 5.09765625 frac[7,26] + mem[7,26] = 0
 accuracy_1_27: - 5.09765625 frac[1,27] + mem[1,27] = 0
 accuracy_0_13: - 5.09765625 frac[0,13] + mem[0,13] = 0
 accuracy_2_0: - 5.09765625 frac[2,0] + mem[2,0] = 0
 accuracy_1_8: - 5.09765625 frac[1,8] + mem[1,8] = 0
 accuracy_8_8: - 6 frac[8,8] + mem[8,8] = 0
 accuracy_2_15: - 5.09765625 frac[2,15] + mem[2,15] = 0
 accuracy_8_27: - 6 frac[8,27] + mem[8,27] = 0
 mem_tot_CPU3: - mem[0,20] - mem[0,7] - mem[0,10] - mem[0,4] - mem[0,27]
   - mem[0,24] - mem[0,5] - mem[0,15] - mem[0,22] - mem[0,25] - mem[0,19]
   - mem[0,6] - mem[0,9] - mem[0,16] - mem[0,3] - mem[0,23] - mem[0,13]
   + mem_tot_CPU3 = 0
 R183: - 3.05176e-05 mem_tot_CPU3 + normalized_mem_tot_CPU3 = 0
 rows_tot_CPU3: - 3 frac[0,20] - 3 frac[0,7] - 3 frac[0,10] - 3 frac[0,4]
   - 3 frac[0,27] - 3 frac[0,24] - 3 frac[0,5] - 3 frac[0,15]
   - 3 frac[0,22] - 3 frac[0,25] - 3 frac[0,19] - 3 frac[0,6] - 3 frac[0,9]
   - 3 frac[0,16] - 3 frac[0,3] - 3 frac[0,23] - 3 frac[0,13]
   + rows_tot_CPU3 = 0
 R185: - 0.0833333333333333 rows_tot_CPU3 + normalized_rows_tot_CPU3 = 0
 mem_tot_CPU5: - mem[1,17] - mem[1,11] - mem[1,12] - mem[1,16] - mem[1,5]
   - mem[1,10] - mem[1,20] - mem[1,18] - mem[1,7] - mem[1,24] - mem[1,13]
   - mem[1,27] - mem[1,8] + mem_tot_CPU5 = 0
 R187: - 3.05176e-05 mem_tot_CPU5 + normalized_mem_tot_CPU5 = 0
 rows_tot_CPU5: - 3 frac[1,17] - 3 frac[1,11] - 3 frac[1,12] - 3 frac[1,16]
   - 3 frac[1,5] - 3 frac[1,10] - 3 frac[1,20] - 3 frac[1,18] - 3 frac[1,7]
   - 3 frac[1,24] - 3 frac[1,13] - 3 frac[1,27] - 3 frac[1,8]
   + rows_tot_CPU5 = 0
 R189: - 0.0833333333333333 rows_tot_CPU5 + normalized_rows_tot_CPU5 = 0
 mem_tot_CPU6: - mem[2,22] - mem[2,18] - mem[2,12] - mem[2,23] - mem[2,2]
   - mem[2,8] - mem[2,25] - mem[2,26] - mem[2,9] - mem[2,0] - mem[2,15]
   + mem_tot_CPU6 = 0
 R191: - 3.05176e-05 mem_tot_CPU6 + normalized_mem_tot_CPU6 = 0
 rows_tot_CPU6: - 3 frac[2,22] - 3 frac[2,18] - 3 frac[2,12] - 3 frac[2,23]
   - 3 frac[2,2] - 3 frac[2,8] - 3 frac[2,25] - 3 frac[2,26] - 3 frac[2,9]
   - 3 frac[2,0] - 3 frac[2,15] + rows_tot_CPU6 = 0
 R193: - 0.0833333333333333 rows_tot_CPU6 + normalized_rows_tot_CPU6 = 0
 mem_tot_CPU7: - mem[3,23] - mem[3,18] - mem[3,11] - mem[3,17] - mem[3,6]
   - mem[3,27] - mem[3,5] - mem[3,0] - mem[3,16] - mem[3,26] - mem[3,4]
   - mem[3,25] - mem[3,3] - mem[3,20] - mem[3,19] + mem_tot_CPU7 = 0
 R195: - 3.05176e-05 mem_tot_CPU7 + normalized_mem_tot_CPU7 = 0
 rows_tot_CPU7: - 3 frac[3,23] - 3 frac[3,18] - 3 frac[3,11] - 3 frac[3,17]
   - 3 frac[3,6] - 3 frac[3,27] - 3 frac[3,5] - 3 frac[3,0] - 3 frac[3,16]
   - 3 frac[3,26] - 3 frac[3,4] - 3 frac[3,25] - 3 frac[3,3] - 3 frac[3,20]
   - 3 frac[3,19] + rows_tot_CPU7 = 0
 R197: - 0.0833333333333333 rows_tot_CPU7 + normalized_rows_tot_CPU7 = 0
 mem_tot_CPU10: - mem[4,0] - mem[4,6] - mem[4,2] - mem[4,7] - mem[4,18]
   - mem[4,23] + mem_tot_CPU10 = 0
 R199: - 3.05176e-05 mem_tot_CPU10 + normalized_mem_tot_CPU10 = 0
 rows_tot_CPU10: - 3 frac[4,0] - 3 frac[4,6] - 3 frac[4,2] - 3 frac[4,7]
   - 3 frac[4,18] - 3 frac[4,23] + rows_tot_CPU10 = 0
 R201: - 0.0833333333333333 rows_tot_CPU10 + normalized_rows_tot_CPU10 = 0
 mem_tot_CPU26: - mem[5,5] - mem[5,18] - mem[5,0] - mem[5,22]
   + mem_tot_CPU26 = 0
 R203: - 3.05176e-05 mem_tot_CPU26 + normalized_mem_tot_CPU26 = 0
 rows_tot_CPU26: - 3 frac[5,5] - 3 frac[5,18] - 3 frac[5,0] - 3 frac[5,22]
   + rows_tot_CPU26 = 0
 R205: - 0.0833333333333333 rows_tot_CPU26 + normalized_rows_tot_CPU26 = 0
 mem_tot_CPU28: - mem[6,9] - mem[6,25] - mem[6,0] - mem[6,26] - mem[6,16]
   - mem[6,20] - mem[6,27] - mem[6,2] - mem[6,15] - mem[6,5] - mem[6,3]
   - mem[6,22] - mem[6,12] - mem[6,19] + mem_tot_CPU28 = 0
 R207: - 3.05176e-05 mem_tot_CPU28 + normalized_mem_tot_CPU28 = 0
 rows_tot_CPU28: - 3 frac[6,9] - 3 frac[6,25] - 3 frac[6,0] - 3 frac[6,26]
   - 3 frac[6,16] - 3 frac[6,20] - 3 frac[6,27] - 3 frac[6,2]
   - 3 frac[6,15] - 3 frac[6,5] - 3 frac[6,3] - 3 frac[6,22] - 3 frac[6,12]
   - 3 frac[6,19] + rows_tot_CPU28 = 0
 R209: - 0.0833333333333333 rows_tot_CPU28 + normalized_rows_tot_CPU28 = 0
 mem_tot_CPU29: - mem[7,12] - mem[7,25] - mem[7,5] - mem[7,22] - mem[7,2]
   - mem[7,15] - mem[7,18] - mem[7,20] - mem[7,27] - mem[7,7] - mem[7,23]
   - mem[7,9] - mem[7,26] + mem_tot_CPU29 = 0
 R211: - 3.05176e-05 mem_tot_CPU29 + normalized_mem_tot_CPU29 = 0
 rows_tot_CPU29: - 3 frac[7,12] - 3 frac[7,25] - 3 frac[7,5] - 3 frac[7,22]
   - 3 frac[7,2] - 3 frac[7,15] - 3 frac[7,18] - 3 frac[7,20]
   - 3 frac[7,27] - 3 frac[7,7] - 3 frac[7,23] - 3 frac[7,9] - 3 frac[7,26]
   + rows_tot_CPU29 = 0
 R213: - 0.0833333333333333 rows_tot_CPU29 + normalized_rows_tot_CPU29 = 0
 mem_tot_tor_P41: - mem[8,5] - mem[8,24] - mem[8,15] - mem[8,2] - mem[8,21]
   - mem[8,12] - mem[8,18] - mem[8,9] - mem[8,6] - mem[8,25] - mem[8,3]
   - mem[8,22] - mem[8,13] - mem[8,0] - mem[8,19] - mem[8,10] - mem[8,16]
   - mem[8,7] - mem[8,26] - mem[8,4] - mem[8,23] - mem[8,14] - mem[8,1]
   - mem[8,20] - mem[8,11] - mem[8,17] - mem[8,8] - mem[8,27]
   + mem_tot_tor_P41 = 0
 R215: - 1.6276041666666666e-04 mem_tot_tor_P41
   + normalized_mem_tot_tor_P41 = 0
 rows_tot_tor_P41: - 3 frac[8,5] - 3 frac[8,24] - 3 frac[8,15]
   - 3 frac[8,2] - 3 frac[8,21] - 3 frac[8,12] - 3 frac[8,18] - 3 frac[8,9]
   - 3 frac[8,6] - 3 frac[8,25] - 3 frac[8,3] - 3 frac[8,22] - 3 frac[8,13]
   - 3 frac[8,0] - 3 frac[8,19] - 3 frac[8,10] - 3 frac[8,16] - 3 frac[8,7]
   - 3 frac[8,26] - 3 frac[8,4] - 3 frac[8,23] - 3 frac[8,14] - 3 frac[8,1]
   - 3 frac[8,20] - 3 frac[8,11] - 3 frac[8,17] - 3 frac[8,8]
   - 3 frac[8,27] + rows_tot_tor_P41 = 0
 R217: - 0.0208333333333333 rows_tot_tor_P41 + normalized_rows_tot_tor_P41
   = 0
 row_capacity_CPU3: rows_tot_CPU3 <= 12
 row_capacity_CPU5: rows_tot_CPU5 <= 12
 row_capacity_CPU6: rows_tot_CPU6 <= 12
 row_capacity_CPU7: rows_tot_CPU7 <= 12
 row_capacity_CPU10: rows_tot_CPU10 <= 12
 row_capacity_CPU26: rows_tot_CPU26 <= 12
 row_capacity_CPU28: rows_tot_CPU28 <= 12
 row_capacity_CPU29: rows_tot_CPU29 <= 12
 row_capacity_tor_P41: rows_tot_tor_P41 <= 48
 capacity_mem_par_tor_P41: mem[8,5] <= 1536
 capacity_mem_par_tor_P41: mem[8,24] <= 1536
 capacity_mem_par_tor_P41: mem[8,15] <= 1536
 capacity_mem_par_tor_P41: mem[8,2] <= 1536
 capacity_mem_par_tor_P41: mem[8,21] <= 1536
 capacity_mem_par_tor_P41: mem[8,12] <= 1536
 capacity_mem_par_tor_P41: mem[8,18] <= 1536
 capacity_mem_par_tor_P41: mem[8,9] <= 1536
 capacity_mem_par_tor_P41: mem[8,6] <= 1536
 capacity_mem_par_tor_P41: mem[8,25] <= 1536
 capacity_mem_par_tor_P41: mem[8,3] <= 1536
 capacity_mem_par_tor_P41: mem[8,22] <= 1536
 capacity_mem_par_tor_P41: mem[8,13] <= 1536
 capacity_mem_par_tor_P41: mem[8,0] <= 1536
 capacity_mem_par_tor_P41: mem[8,19] <= 1536
 capacity_mem_par_tor_P41: mem[8,10] <= 1536
 capacity_mem_par_tor_P41: mem[8,16] <= 1536
 capacity_mem_par_tor_P41: mem[8,7] <= 1536
 capacity_mem_par_tor_P41: mem[8,26] <= 1536
 capacity_mem_par_tor_P41: mem[8,4] <= 1536
 capacity_mem_par_tor_P41: mem[8,23] <= 1536
 capacity_mem_par_tor_P41: mem[8,14] <= 1536
 capacity_mem_par_tor_P41: mem[8,1] <= 1536
 capacity_mem_par_tor_P41: mem[8,20] <= 1536
 capacity_mem_par_tor_P41: mem[8,11] <= 1536
 capacity_mem_par_tor_P41: mem[8,17] <= 1536
 capacity_mem_par_tor_P41: mem[8,8] <= 1536
 capacity_mem_par_tor_P41: mem[8,27] <= 1536
 ns_single_CPU3: 3.5 rows_tot_CPU3 - ns_single_CPU3 + pdt_m_rows_CPU3 = 0
 ns_sketch_CPU3: - ns_single_CPU3 + 12.09395982124746 cores_sketch_CPU3
   >= 0
 capacity_cores_CPU3: cores_sketch_CPU3 <= 2
 ns_single_CPU5: 3.5 rows_tot_CPU5 - ns_single_CPU5 + pdt_m_rows_CPU5 = 0
 ns_sketch_CPU5: - ns_single_CPU5 + 12.09395982124746 cores_sketch_CPU5
   >= 0
 capacity_cores_CPU5: cores_sketch_CPU5 <= 2
 ns_single_CPU6: 3.5 rows_tot_CPU6 - ns_single_CPU6 + pdt_m_rows_CPU6 = 0
 ns_sketch_CPU6: - ns_single_CPU6 + 12.09395982124746 cores_sketch_CPU6
   >= 0
 capacity_cores_CPU6: cores_sketch_CPU6 <= 2
 ns_single_CPU7: 3.5 rows_tot_CPU7 - ns_single_CPU7 + pdt_m_rows_CPU7 = 0
 ns_sketch_CPU7: - ns_single_CPU7 + 12.09395982124746 cores_sketch_CPU7
   >= 0
 capacity_cores_CPU7: cores_sketch_CPU7 <= 2
 ns_single_CPU10: 3.5 rows_tot_CPU10 - ns_single_CPU10 + pdt_m_rows_CPU10
   = 0
 ns_sketch_CPU10: - ns_single_CPU10 + 12.09395982124746 cores_sketch_CPU10
   >= 0
 capacity_cores_CPU10: cores_sketch_CPU10 <= 2
 ns_single_CPU26: 3.5 rows_tot_CPU26 - ns_single_CPU26 + pdt_m_rows_CPU26
   = 0
 ns_sketch_CPU26: - ns_single_CPU26 + 12.09395982124746 cores_sketch_CPU26
   >= 0
 capacity_cores_CPU26: cores_sketch_CPU26 <= 2
 ns_single_CPU28: 3.5 rows_tot_CPU28 - ns_single_CPU28 + pdt_m_rows_CPU28
   = 0
 ns_sketch_CPU28: - ns_single_CPU28 + 12.09395982124746 cores_sketch_CPU28
   >= 0
 capacity_cores_CPU28: cores_sketch_CPU28 <= 2
 ns_single_CPU29: 3.5 rows_tot_CPU29 - ns_single_CPU29 + pdt_m_rows_CPU29
   = 0
 ns_sketch_CPU29: - ns_single_CPU29 + 12.09395982124746 cores_sketch_CPU29
   >= 0
 capacity_cores_CPU29: cores_sketch_CPU29 <= 2
 res_acc: 1.220703125e-04 mem_tot_CPU3 + 1.220703125e-04 mem_tot_CPU5
   + 1.220703125e-04 mem_tot_CPU6 + 1.220703125e-04 mem_tot_CPU7
   + 1.220703125e-04 mem_tot_CPU10 + 1.220703125e-04 mem_tot_CPU26
   + 1.220703125e-04 mem_tot_CPU28 + 1.220703125e-04 mem_tot_CPU29
   + 0.001953125 mem_tot_tor_P41 + 0.25 rows_tot_tor_P41
   + 10 cores_sketch_CPU3 + 10 cores_sketch_CPU5 + 10 cores_sketch_CPU6
   + 10 cores_sketch_CPU7 + 10 cores_sketch_CPU10 + 10 cores_sketch_CPU26
   + 10 cores_sketch_CPU28 + 10 cores_sketch_CPU29 - res = -400
 pdt_m_rows_CPU3: pdt_m_rows_CPU3 + [ - rows_tot_CPU3 * m_access_time_CPU3
   ] = 0
 pdt_m_rows_CPU5: pdt_m_rows_CPU5 + [ - rows_tot_CPU5 * m_access_time_CPU5
   ] = 0
 pdt_m_rows_CPU6: pdt_m_rows_CPU6 + [ - rows_tot_CPU6 * m_access_time_CPU6
   ] = 0
 pdt_m_rows_CPU7: pdt_m_rows_CPU7 + [ - rows_tot_CPU7 * m_access_time_CPU7
   ] = 0
 pdt_m_rows_CPU10: pdt_m_rows_CPU10 + [
   - rows_tot_CPU10 * m_access_time_CPU10 ] = 0
 pdt_m_rows_CPU26: pdt_m_rows_CPU26 + [
   - rows_tot_CPU26 * m_access_time_CPU26 ] = 0
 pdt_m_rows_CPU28: pdt_m_rows_CPU28 + [
   - rows_tot_CPU28 * m_access_time_CPU28 ] = 0
 pdt_m_rows_CPU29: pdt_m_rows_CPU29 + [
   - rows_tot_CPU29 * m_access_time_CPU29 ] = 0
Bounds
 mem_tot_CPU3 <= 32768
 normalized_mem_tot_CPU3 <= 1
 normalized_rows_tot_CPU3 <= 1
 mem_tot_CPU5 <= 32768
 normalized_mem_tot_CPU5 <= 1
 normalized_rows_tot_CPU5 <= 1
 mem_tot_CPU6 <= 32768
 normalized_mem_tot_CPU6 <= 1
 normalized_rows_tot_CPU6 <= 1
 mem_tot_CPU7 <= 32768
 normalized_mem_tot_CPU7 <= 1
 normalized_rows_tot_CPU7 <= 1
 mem_tot_CPU10 <= 32768
 normalized_mem_tot_CPU10 <= 1
 normalized_rows_tot_CPU10 <= 1
 mem_tot_CPU26 <= 32768
 normalized_mem_tot_CPU26 <= 1
 normalized_rows_tot_CPU26 <= 1
 mem_tot_CPU28 <= 32768
 normalized_mem_tot_CPU28 <= 1
 normalized_rows_tot_CPU28 <= 1
 mem_tot_CPU29 <= 32768
 normalized_mem_tot_CPU29 <= 1
 normalized_rows_tot_CPU29 <= 1
 mem_tot_tor_P41 <= 6144
 normalized_mem_tot_tor_P41 <= 1
 normalized_rows_tot_tor_P41 <= 1
 cores_sketch_CPU3 <= 7
 cores_sketch_CPU5 <= 7
 cores_sketch_CPU6 <= 7
 cores_sketch_CPU7 <= 7
 cores_sketch_CPU10 <= 7
 cores_sketch_CPU26 <= 7
 cores_sketch_CPU28 <= 7
 cores_sketch_CPU29 <= 7
Binaries
 frac[6,9] frac[0,20] frac[7,12] frac[1,17] frac[0,7] frac[7,25] frac[2,22]
 frac[0,10] frac[1,11] frac[8,5] frac[4,0] frac[8,24] frac[5,5] frac[5,18]
 frac[6,25] frac[0,4] frac[8,15] frac[8,2] frac[3,23] frac[8,21] frac[6,0]
 frac[7,5] frac[7,22] frac[6,26] frac[1,12] frac[8,12] frac[3,18]
 frac[8,18] frac[6,16] frac[0,27] frac[2,18] frac[3,11] frac[8,9]
 frac[2,12] frac[3,17] frac[7,2] frac[7,15] frac[1,16] frac[0,24]
 frac[2,23] frac[1,5] frac[3,6] frac[2,2] frac[1,10] frac[8,6] frac[3,27]
 frac[8,25] frac[7,18] frac[0,5] frac[3,5] frac[8,3] frac[4,6] frac[8,22]
 frac[6,20] frac[6,27] frac[8,13] frac[3,0] frac[2,8] frac[8,0] frac[8,19]
 frac[6,2] frac[1,20] frac[7,20] frac[0,15] frac[8,10] frac[5,0] frac[3,16]
 frac[8,16] frac[6,15] frac[0,22] frac[0,25] frac[7,27] frac[8,7] frac[4,2]
 frac[3,26] frac[8,26] frac[6,5] frac[0,19] frac[1,18] frac[0,6] frac[2,25]
 frac[1,7] frac[0,9] frac[3,4] frac[8,4] frac[4,7] frac[3,25] frac[8,23]
 frac[4,18] frac[7,7] frac[0,16] frac[0,3] frac[2,26] frac[8,14] frac[3,3]
 frac[2,9] frac[8,1] frac[3,20] frac[8,20] frac[6,3] frac[4,23] frac[6,22]
 frac[5,22] frac[7,23] frac[1,24] frac[1,13] frac[8,11] frac[3,19]
 frac[8,17] frac[6,12] frac[0,23] frac[7,9] frac[6,19] frac[7,26]
 frac[1,27] frac[0,13] frac[2,0] frac[1,8] frac[8,8] frac[2,15] frac[8,27]
Generals
 cores_sketch_CPU3 cores_sketch_CPU5 cores_sketch_CPU6 cores_sketch_CPU7
 cores_sketch_CPU10 cores_sketch_CPU26 cores_sketch_CPU28
 cores_sketch_CPU29
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_tor_P41 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU3 , normalized_mem_tot_CPU5 ,
   normalized_mem_tot_CPU6 , normalized_mem_tot_CPU7 ,
   normalized_mem_tot_CPU10 , normalized_mem_tot_CPU26 ,
   normalized_mem_tot_CPU28 , normalized_mem_tot_CPU29 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_tor_P41 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU3 ,
   normalized_rows_tot_CPU5 , normalized_rows_tot_CPU6 ,
   normalized_rows_tot_CPU7 , normalized_rows_tot_CPU10 ,
   normalized_rows_tot_CPU26 , normalized_rows_tot_CPU28 ,
   normalized_rows_tot_CPU29 )
 mem_access_time_CPU3: m_access_time_CPU3 = PWL ( mem_tot_CPU3 ) : (0, 0) 
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
 mem_access_time_CPU10: m_access_time_CPU10 = PWL ( mem_tot_CPU10 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU26: m_access_time_CPU26 = PWL ( mem_tot_CPU26 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU28: m_access_time_CPU28 = PWL ( mem_tot_CPU28 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU29: m_access_time_CPU29 = PWL ( mem_tot_CPU29 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

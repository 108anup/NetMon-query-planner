\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_others + 0 max_rows_others + 0 m_access_time_CPU17
   + 0 m_access_time_CPU29 + 0 m_access_time_CPU28 + 0 m_access_time_CPU8
   + 0 m_access_time_CPU6 + 0 m_access_time_CPU23 + 0 m_access_time_CPU7
   + 0 m_access_time_CPU30 + 0 m_access_time_CPU10 + 0 m_access_time_CPU31
   + 0 m_access_time_CPU27 + 0 m_access_time_CPU19 + 0 m_access_time_CPU1
   + 0 m_access_time_CPU15 + 0 m_access_time_CPU3 + 0 m_access_time_CPU9
   + res
Subject To
 cov_0_17: frac[3,17] + frac[7,17] >= 0.6051264
 cov_1_10: frac[3,10] + frac[5,10] >= 0.6051264
 cov_2_17: frac[2,17] + frac[6,17] >= 0.6051264
 cov_3_1: frac[4,1] + frac[6,1] >= 0.8
 cov_4_17: frac[3,17] + frac[4,17] >= 0.6051264
 cov_5_3: frac[2,3] + frac[3,3] >= 0.8
 cov_6_10: frac[0,10] + frac[6,10] >= 0.6051264
 cov_7_14: frac[1,14] + frac[2,14] >= 0.6051264
 cov_8_9: frac[1,9] + frac[4,9] >= 0.8
 cov_9_14: frac[2,14] + frac[3,14] >= 0.6051264
 cov_10_3: frac[2,3] + frac[3,3] >= 0.7
 cov_11_12: frac[4,12] + frac[7,12] >= 0.9
 cov_12_7: frac[5,7] + frac[7,7] >= 0.6051264
 cov_13_14: frac[4,14] + frac[5,14] >= 0.6051264
 cov_14_7: frac[1,7] + frac[6,7] >= 0.6051264
 cov_15_7: frac[5,7] + frac[7,7] >= 0.6051264
 cov_16_1: frac[0,1] + frac[7,1] >= 0.8
 cov_17_9: frac[4,9] + frac[6,9] >= 0.9
 cov_18_10: frac[1,10] + frac[4,10] >= 0.6051264
 cov_19_14: frac[6,14] + frac[7,14] >= 0.6051264
 cov_20_1: frac[1,1] + frac[7,1] >= 0.8
 cov_21_9: frac[0,9] + frac[2,9] >= 0.7
 cov_22_1: frac[5,1] + frac[7,1] >= 0.9
 cov_23_0: frac[4,0] + frac[6,0] >= 0.6051264
 cov_24_3: frac[0,3] + frac[4,3] >= 0.9
 cov_25_17: frac[0,17] + frac[3,17] >= 0.6051264
 cov_26_9: frac[2,9] + frac[3,9] >= 0.7
 cov_27_2: frac[11,2] + frac[12,2] >= 0.8
 cov_28_6: frac[14,6] + frac[15,6] >= 0.7
 cov_29_13: frac[9,13] + frac[15,13] >= 0.6051264
 cov_30_5: frac[13,5] + frac[14,5] >= 0.6051264
 cov_31_11: frac[10,11] + frac[13,11] >= 0.6051264
 cov_32_16: frac[11,16] + frac[12,16] >= 0.7
 cov_33_13: frac[10,13] + frac[13,13] >= 0.6051264
 cov_34_15: frac[10,15] + frac[12,15] >= 0.6051264
 cov_35_2: frac[10,2] + frac[11,2] >= 0.7
 cov_36_4: frac[9,4] + frac[14,4] >= 0.9
 cov_37_2: frac[11,2] + frac[15,2] >= 0.7
 cov_38_2: frac[12,2] + frac[15,2] >= 0.8
 cov_39_8: frac[14,8] + frac[15,8] >= 0.6051264
 cov_40_5: frac[14,5] + frac[15,5] >= 0.6051264
 cov_41_15: frac[14,15] + frac[15,15] >= 0.6051264
 cov_42_5: frac[13,5] + frac[15,5] >= 0.6051264
 cov_43_2: frac[11,2] + frac[15,2] >= 0.9
 cov_44_2: frac[10,2] + frac[14,2] >= 0.8
 cov_45_8: frac[10,8] + frac[13,8] >= 0.6051264
 cov_46_16: frac[12,16] + frac[13,16] >= 0.9
 cov_47_11: frac[10,11] + frac[12,11] >= 0.6051264
 cov_48_16: frac[12,16] + frac[14,16] >= 0.9
 cov_49_16: frac[14,16] + frac[15,16] >= 0.7
 cov_50_6: frac[8,6] + frac[14,6] >= 0.9
 cov_51_15: frac[10,15] + frac[12,15] >= 0.6051264
 cov_52_5: frac[10,5] + frac[11,5] >= 0.6051264
 cov_53_8: frac[8,8] + frac[13,8] >= 0.6051264
 cov_54_5: frac[13,5] + frac[15,5] >= 0.6051264
 accuracy_6_9: - 509.68359375 frac[6,9] + mem[6,9] = 0
 accuracy_7_12: - 509.68359375 frac[7,12] + mem[7,12] = 0
 accuracy_14_4: - 509.68359375 frac[14,4] + mem[14,4] = 0
 accuracy_0_10: - 509.68359375 frac[0,10] + mem[0,10] = 0
 accuracy_4_0: - 509.68359375 frac[4,0] + mem[4,0] = 0
 accuracy_10_8: - 509.68359375 frac[10,8] + mem[10,8] = 0
 accuracy_6_7: - 509.68359375 frac[6,7] + mem[6,7] = 0
 accuracy_11_5: - 509.68359375 frac[11,5] + mem[11,5] = 0
 accuracy_6_10: - 509.68359375 frac[6,10] + mem[6,10] = 0
 accuracy_0_17: - 509.68359375 frac[0,17] + mem[0,17] = 0
 accuracy_1_1: - 509.68359375 frac[1,1] + mem[1,1] = 0
 accuracy_4_10: - 509.68359375 frac[4,10] + mem[4,10] = 0
 accuracy_10_13: - 509.68359375 frac[10,13] + mem[10,13] = 0
 accuracy_6_0: - 509.68359375 frac[6,0] + mem[6,0] = 0
 accuracy_14_15: - 509.68359375 frac[14,15] + mem[14,15] = 0
 accuracy_12_11: - 509.68359375 frac[12,11] + mem[12,11] = 0
 accuracy_14_2: - 509.68359375 frac[14,2] + mem[14,2] = 0
 accuracy_2_17: - 509.68359375 frac[2,17] + mem[2,17] = 0
 accuracy_0_1: - 509.68359375 frac[0,1] + mem[0,1] = 0
 accuracy_5_14: - 509.68359375 frac[5,14] + mem[5,14] = 0
 accuracy_15_16: - 509.68359375 frac[15,16] + mem[15,16] = 0
 accuracy_14_8: - 509.68359375 frac[14,8] + mem[14,8] = 0
 accuracy_11_16: - 509.68359375 frac[11,16] + mem[11,16] = 0
 accuracy_15_13: - 509.68359375 frac[15,13] + mem[15,13] = 0
 accuracy_13_13: - 509.68359375 frac[13,13] + mem[13,13] = 0
 accuracy_4_12: - 509.68359375 frac[4,12] + mem[4,12] = 0
 accuracy_9_4: - 509.68359375 frac[9,4] + mem[9,4] = 0
 accuracy_5_1: - 509.68359375 frac[5,1] + mem[5,1] = 0
 accuracy_3_17: - 509.68359375 frac[3,17] + mem[3,17] = 0
 accuracy_6_14: - 509.68359375 frac[6,14] + mem[6,14] = 0
 accuracy_12_2: - 509.68359375 frac[12,2] + mem[12,2] = 0
 accuracy_14_5: - 509.68359375 frac[14,5] + mem[14,5] = 0
 accuracy_13_16: - 509.68359375 frac[13,16] + mem[13,16] = 0
 accuracy_1_10: - 509.68359375 frac[1,10] + mem[1,10] = 0
 accuracy_8_6: - 509.68359375 frac[8,6] + mem[8,6] = 0
 accuracy_4_1: - 509.68359375 frac[4,1] + mem[4,1] = 0
 accuracy_7_1: - 509.68359375 frac[7,1] + mem[7,1] = 0
 accuracy_14_6: - 509.68359375 frac[14,6] + mem[14,6] = 0
 accuracy_9_13: - 509.68359375 frac[9,13] + mem[9,13] = 0
 accuracy_5_10: - 509.68359375 frac[5,10] + mem[5,10] = 0
 accuracy_6_1: - 509.68359375 frac[6,1] + mem[6,1] = 0
 accuracy_5_7: - 509.68359375 frac[5,7] + mem[5,7] = 0
 accuracy_4_17: - 509.68359375 frac[4,17] + mem[4,17] = 0
 accuracy_7_17: - 509.68359375 frac[7,17] + mem[7,17] = 0
 accuracy_10_15: - 509.68359375 frac[10,15] + mem[10,15] = 0
 accuracy_6_17: - 509.68359375 frac[6,17] + mem[6,17] = 0
 accuracy_3_10: - 509.68359375 frac[3,10] + mem[3,10] = 0
 accuracy_1_14: - 509.68359375 frac[1,14] + mem[1,14] = 0
 accuracy_7_14: - 509.68359375 frac[7,14] + mem[7,14] = 0
 accuracy_3_9: - 509.68359375 frac[3,9] + mem[3,9] = 0
 accuracy_2_3: - 509.68359375 frac[2,3] + mem[2,3] = 0
 accuracy_1_9: - 509.68359375 frac[1,9] + mem[1,9] = 0
 accuracy_2_14: - 509.68359375 frac[2,14] + mem[2,14] = 0
 accuracy_10_5: - 509.68359375 frac[10,5] + mem[10,5] = 0
 accuracy_14_16: - 509.68359375 frac[14,16] + mem[14,16] = 0
 accuracy_13_5: - 509.68359375 frac[13,5] + mem[13,5] = 0
 accuracy_15_6: - 509.68359375 frac[15,6] + mem[15,6] = 0
 accuracy_1_7: - 509.68359375 frac[1,7] + mem[1,7] = 0
 accuracy_0_9: - 509.68359375 frac[0,9] + mem[0,9] = 0
 accuracy_10_11: - 509.68359375 frac[10,11] + mem[10,11] = 0
 accuracy_11_2: - 509.68359375 frac[11,2] + mem[11,2] = 0
 accuracy_7_7: - 509.68359375 frac[7,7] + mem[7,7] = 0
 accuracy_15_8: - 509.68359375 frac[15,8] + mem[15,8] = 0
 accuracy_13_8: - 509.68359375 frac[13,8] + mem[13,8] = 0
 accuracy_12_16: - 509.68359375 frac[12,16] + mem[12,16] = 0
 accuracy_0_3: - 509.68359375 frac[0,3] + mem[0,3] = 0
 accuracy_15_5: - 509.68359375 frac[15,5] + mem[15,5] = 0
 accuracy_3_14: - 509.68359375 frac[3,14] + mem[3,14] = 0
 accuracy_4_9: - 509.68359375 frac[4,9] + mem[4,9] = 0
 accuracy_3_3: - 509.68359375 frac[3,3] + mem[3,3] = 0
 accuracy_2_9: - 509.68359375 frac[2,9] + mem[2,9] = 0
 accuracy_15_15: - 509.68359375 frac[15,15] + mem[15,15] = 0
 accuracy_13_11: - 509.68359375 frac[13,11] + mem[13,11] = 0
 accuracy_4_14: - 509.68359375 frac[4,14] + mem[4,14] = 0
 accuracy_12_15: - 509.68359375 frac[12,15] + mem[12,15] = 0
 accuracy_15_2: - 509.68359375 frac[15,2] + mem[15,2] = 0
 accuracy_8_8: - 509.68359375 frac[8,8] + mem[8,8] = 0
 accuracy_4_3: - 509.68359375 frac[4,3] + mem[4,3] = 0
 accuracy_10_2: - 509.68359375 frac[10,2] + mem[10,2] = 0
 mem_tot_CPU17: - mem[0,10] - mem[0,17] - mem[0,1] - mem[0,9] - mem[0,3]
   + mem_tot_CPU17 = 0
 R135: - 3.05176e-05 mem_tot_CPU17 + normalized_mem_tot_CPU17 = 0
 rows_tot_CPU17: - 3 frac[0,10] - 3 frac[0,17] - 3 frac[0,1] - 3 frac[0,9]
   - 3 frac[0,3] + rows_tot_CPU17 = 0
 R137: - 0.0833333333333333 rows_tot_CPU17 + normalized_rows_tot_CPU17 = 0
 mem_tot_CPU29: - mem[1,1] - mem[1,10] - mem[1,14] - mem[1,9] - mem[1,7]
   + mem_tot_CPU29 = 0
 R139: - 3.05176e-05 mem_tot_CPU29 + normalized_mem_tot_CPU29 = 0
 rows_tot_CPU29: - 3 frac[1,1] - 3 frac[1,10] - 3 frac[1,14] - 3 frac[1,9]
   - 3 frac[1,7] + rows_tot_CPU29 = 0
 R141: - 0.0833333333333333 rows_tot_CPU29 + normalized_rows_tot_CPU29 = 0
 mem_tot_CPU28: - mem[2,17] - mem[2,3] - mem[2,14] - mem[2,9]
   + mem_tot_CPU28 = 0
 R143: - 3.05176e-05 mem_tot_CPU28 + normalized_mem_tot_CPU28 = 0
 rows_tot_CPU28: - 3 frac[2,17] - 3 frac[2,3] - 3 frac[2,14] - 3 frac[2,9]
   + rows_tot_CPU28 = 0
 R145: - 0.0833333333333333 rows_tot_CPU28 + normalized_rows_tot_CPU28 = 0
 mem_tot_CPU8: - mem[3,17] - mem[3,10] - mem[3,9] - mem[3,14] - mem[3,3]
   + mem_tot_CPU8 = 0
 R147: - 3.05176e-05 mem_tot_CPU8 + normalized_mem_tot_CPU8 = 0
 rows_tot_CPU8: - 3 frac[3,17] - 3 frac[3,10] - 3 frac[3,9] - 3 frac[3,14]
   - 3 frac[3,3] + rows_tot_CPU8 = 0
 R149: - 0.0833333333333333 rows_tot_CPU8 + normalized_rows_tot_CPU8 = 0
 mem_tot_CPU6: - mem[4,0] - mem[4,10] - mem[4,12] - mem[4,1] - mem[4,17]
   - mem[4,9] - mem[4,14] - mem[4,3] + mem_tot_CPU6 = 0
 R151: - 3.05176e-05 mem_tot_CPU6 + normalized_mem_tot_CPU6 = 0
 rows_tot_CPU6: - 3 frac[4,0] - 3 frac[4,10] - 3 frac[4,12] - 3 frac[4,1]
   - 3 frac[4,17] - 3 frac[4,9] - 3 frac[4,14] - 3 frac[4,3]
   + rows_tot_CPU6 = 0
 R153: - 0.0833333333333333 rows_tot_CPU6 + normalized_rows_tot_CPU6 = 0
 mem_tot_CPU23: - mem[5,14] - mem[5,1] - mem[5,10] - mem[5,7]
   + mem_tot_CPU23 = 0
 R155: - 3.05176e-05 mem_tot_CPU23 + normalized_mem_tot_CPU23 = 0
 rows_tot_CPU23: - 3 frac[5,14] - 3 frac[5,1] - 3 frac[5,10] - 3 frac[5,7]
   + rows_tot_CPU23 = 0
 R157: - 0.0833333333333333 rows_tot_CPU23 + normalized_rows_tot_CPU23 = 0
 mem_tot_CPU7: - mem[6,9] - mem[6,7] - mem[6,10] - mem[6,0] - mem[6,14]
   - mem[6,1] - mem[6,17] + mem_tot_CPU7 = 0
 R159: - 3.05176e-05 mem_tot_CPU7 + normalized_mem_tot_CPU7 = 0
 rows_tot_CPU7: - 3 frac[6,9] - 3 frac[6,7] - 3 frac[6,10] - 3 frac[6,0]
   - 3 frac[6,14] - 3 frac[6,1] - 3 frac[6,17] + rows_tot_CPU7 = 0
 R161: - 0.0833333333333333 rows_tot_CPU7 + normalized_rows_tot_CPU7 = 0
 mem_tot_CPU30: - mem[7,12] - mem[7,1] - mem[7,17] - mem[7,14] - mem[7,7]
   + mem_tot_CPU30 = 0
 R163: - 3.05176e-05 mem_tot_CPU30 + normalized_mem_tot_CPU30 = 0
 rows_tot_CPU30: - 3 frac[7,12] - 3 frac[7,1] - 3 frac[7,17] - 3 frac[7,14]
   - 3 frac[7,7] + rows_tot_CPU30 = 0
 R165: - 0.0833333333333333 rows_tot_CPU30 + normalized_rows_tot_CPU30 = 0
 mem_tot_CPU10: - mem[8,6] - mem[8,8] + mem_tot_CPU10 = 0
 R167: - 3.05176e-05 mem_tot_CPU10 + normalized_mem_tot_CPU10 = 0
 rows_tot_CPU10: - 3 frac[8,6] - 3 frac[8,8] + rows_tot_CPU10 = 0
 R169: - 0.0833333333333333 rows_tot_CPU10 + normalized_rows_tot_CPU10 = 0
 mem_tot_CPU31: - mem[9,4] - mem[9,13] + mem_tot_CPU31 = 0
 R171: - 3.05176e-05 mem_tot_CPU31 + normalized_mem_tot_CPU31 = 0
 rows_tot_CPU31: - 3 frac[9,4] - 3 frac[9,13] + rows_tot_CPU31 = 0
 R173: - 0.0833333333333333 rows_tot_CPU31 + normalized_rows_tot_CPU31 = 0
 mem_tot_CPU27: - mem[10,8] - mem[10,13] - mem[10,15] - mem[10,5]
   - mem[10,11] - mem[10,2] + mem_tot_CPU27 = 0
 R175: - 3.05176e-05 mem_tot_CPU27 + normalized_mem_tot_CPU27 = 0
 rows_tot_CPU27: - 3 frac[10,8] - 3 frac[10,13] - 3 frac[10,15]
   - 3 frac[10,5] - 3 frac[10,11] - 3 frac[10,2] + rows_tot_CPU27 = 0
 R177: - 0.0833333333333333 rows_tot_CPU27 + normalized_rows_tot_CPU27 = 0
 mem_tot_CPU19: - mem[11,5] - mem[11,16] - mem[11,2] + mem_tot_CPU19 = 0
 R179: - 3.05176e-05 mem_tot_CPU19 + normalized_mem_tot_CPU19 = 0
 rows_tot_CPU19: - 3 frac[11,5] - 3 frac[11,16] - 3 frac[11,2]
   + rows_tot_CPU19 = 0
 R181: - 0.0833333333333333 rows_tot_CPU19 + normalized_rows_tot_CPU19 = 0
 mem_tot_CPU1: - mem[12,11] - mem[12,2] - mem[12,16] - mem[12,15]
   + mem_tot_CPU1 = 0
 R183: - 3.05176e-05 mem_tot_CPU1 + normalized_mem_tot_CPU1 = 0
 rows_tot_CPU1: - 3 frac[12,11] - 3 frac[12,2] - 3 frac[12,16]
   - 3 frac[12,15] + rows_tot_CPU1 = 0
 R185: - 0.0833333333333333 rows_tot_CPU1 + normalized_rows_tot_CPU1 = 0
 mem_tot_CPU15: - mem[13,13] - mem[13,16] - mem[13,5] - mem[13,8]
   - mem[13,11] + mem_tot_CPU15 = 0
 R187: - 3.05176e-05 mem_tot_CPU15 + normalized_mem_tot_CPU15 = 0
 rows_tot_CPU15: - 3 frac[13,13] - 3 frac[13,16] - 3 frac[13,5]
   - 3 frac[13,8] - 3 frac[13,11] + rows_tot_CPU15 = 0
 R189: - 0.0833333333333333 rows_tot_CPU15 + normalized_rows_tot_CPU15 = 0
 mem_tot_CPU3: - mem[14,4] - mem[14,15] - mem[14,2] - mem[14,8] - mem[14,5]
   - mem[14,6] - mem[14,16] + mem_tot_CPU3 = 0
 R191: - 3.05176e-05 mem_tot_CPU3 + normalized_mem_tot_CPU3 = 0
 rows_tot_CPU3: - 3 frac[14,4] - 3 frac[14,15] - 3 frac[14,2]
   - 3 frac[14,8] - 3 frac[14,5] - 3 frac[14,6] - 3 frac[14,16]
   + rows_tot_CPU3 = 0
 R193: - 0.0833333333333333 rows_tot_CPU3 + normalized_rows_tot_CPU3 = 0
 mem_tot_CPU9: - mem[15,16] - mem[15,13] - mem[15,6] - mem[15,8]
   - mem[15,5] - mem[15,15] - mem[15,2] + mem_tot_CPU9 = 0
 R195: - 3.05176e-05 mem_tot_CPU9 + normalized_mem_tot_CPU9 = 0
 rows_tot_CPU9: - 3 frac[15,16] - 3 frac[15,13] - 3 frac[15,6]
   - 3 frac[15,8] - 3 frac[15,5] - 3 frac[15,15] - 3 frac[15,2]
   + rows_tot_CPU9 = 0
 R197: - 0.0833333333333333 rows_tot_CPU9 + normalized_rows_tot_CPU9 = 0
 row_capacity_CPU17: rows_tot_CPU17 <= 12
 row_capacity_CPU29: rows_tot_CPU29 <= 12
 row_capacity_CPU28: rows_tot_CPU28 <= 12
 row_capacity_CPU8: rows_tot_CPU8 <= 12
 row_capacity_CPU6: rows_tot_CPU6 <= 12
 row_capacity_CPU23: rows_tot_CPU23 <= 12
 row_capacity_CPU7: rows_tot_CPU7 <= 12
 row_capacity_CPU30: rows_tot_CPU30 <= 12
 row_capacity_CPU10: rows_tot_CPU10 <= 12
 row_capacity_CPU31: rows_tot_CPU31 <= 12
 row_capacity_CPU27: rows_tot_CPU27 <= 12
 row_capacity_CPU19: rows_tot_CPU19 <= 12
 row_capacity_CPU1: rows_tot_CPU1 <= 12
 row_capacity_CPU15: rows_tot_CPU15 <= 12
 row_capacity_CPU3: rows_tot_CPU3 <= 12
 row_capacity_CPU9: rows_tot_CPU9 <= 12
 ns_single_CPU17: 3.5 rows_tot_CPU17 - ns_single_CPU17 + pdt_m_rows_CPU17
   = 0
 ns_sketch_CPU17: - ns_single_CPU17 + 12.50006 cores_sketch_CPU17 >= 0
 capacity_cores_CPU17: cores_sketch_CPU17 <= 3
 ns_single_CPU29: 3.5 rows_tot_CPU29 - ns_single_CPU29 + pdt_m_rows_CPU29
   = 0
 ns_sketch_CPU29: - ns_single_CPU29 + 12.50006 cores_sketch_CPU29 >= 0
 capacity_cores_CPU29: cores_sketch_CPU29 <= 3
 ns_single_CPU28: 3.5 rows_tot_CPU28 - ns_single_CPU28 + pdt_m_rows_CPU28
   = 0
 ns_sketch_CPU28: - ns_single_CPU28 + 12.50006 cores_sketch_CPU28 >= 0
 capacity_cores_CPU28: cores_sketch_CPU28 <= 3
 ns_single_CPU8: 3.5 rows_tot_CPU8 - ns_single_CPU8 + pdt_m_rows_CPU8 = 0
 ns_sketch_CPU8: - ns_single_CPU8 + 12.50006 cores_sketch_CPU8 >= 0
 capacity_cores_CPU8: cores_sketch_CPU8 <= 3
 ns_single_CPU6: 3.5 rows_tot_CPU6 - ns_single_CPU6 + pdt_m_rows_CPU6 = 0
 ns_sketch_CPU6: - ns_single_CPU6 + 12.50006 cores_sketch_CPU6 >= 0
 capacity_cores_CPU6: cores_sketch_CPU6 <= 3
 ns_single_CPU23: 3.5 rows_tot_CPU23 - ns_single_CPU23 + pdt_m_rows_CPU23
   = 0
 ns_sketch_CPU23: - ns_single_CPU23 + 12.50006 cores_sketch_CPU23 >= 0
 capacity_cores_CPU23: cores_sketch_CPU23 <= 3
 ns_single_CPU7: 3.5 rows_tot_CPU7 - ns_single_CPU7 + pdt_m_rows_CPU7 = 0
 ns_sketch_CPU7: - ns_single_CPU7 + 12.50006 cores_sketch_CPU7 >= 0
 capacity_cores_CPU7: cores_sketch_CPU7 <= 3
 ns_single_CPU30: 3.5 rows_tot_CPU30 - ns_single_CPU30 + pdt_m_rows_CPU30
   = 0
 ns_sketch_CPU30: - ns_single_CPU30 + 12.50006 cores_sketch_CPU30 >= 0
 capacity_cores_CPU30: cores_sketch_CPU30 <= 3
 ns_single_CPU10: 3.5 rows_tot_CPU10 - ns_single_CPU10 + pdt_m_rows_CPU10
   = 0
 ns_sketch_CPU10: - ns_single_CPU10 + 12.50006 cores_sketch_CPU10 >= 0
 capacity_cores_CPU10: cores_sketch_CPU10 <= 3
 ns_single_CPU31: 3.5 rows_tot_CPU31 - ns_single_CPU31 + pdt_m_rows_CPU31
   = 0
 ns_sketch_CPU31: - ns_single_CPU31 + 12.50006 cores_sketch_CPU31 >= 0
 capacity_cores_CPU31: cores_sketch_CPU31 <= 3
 ns_single_CPU27: 3.5 rows_tot_CPU27 - ns_single_CPU27 + pdt_m_rows_CPU27
   = 0
 ns_sketch_CPU27: - ns_single_CPU27 + 12.50006 cores_sketch_CPU27 >= 0
 capacity_cores_CPU27: cores_sketch_CPU27 <= 3
 ns_single_CPU19: 3.5 rows_tot_CPU19 - ns_single_CPU19 + pdt_m_rows_CPU19
   = 0
 ns_sketch_CPU19: - ns_single_CPU19 + 12.50006 cores_sketch_CPU19 >= 0
 capacity_cores_CPU19: cores_sketch_CPU19 <= 3
 ns_single_CPU1: 3.5 rows_tot_CPU1 - ns_single_CPU1 + pdt_m_rows_CPU1 = 0
 ns_sketch_CPU1: - ns_single_CPU1 + 12.50006 cores_sketch_CPU1 >= 0
 capacity_cores_CPU1: cores_sketch_CPU1 <= 3
 ns_single_CPU15: 3.5 rows_tot_CPU15 - ns_single_CPU15 + pdt_m_rows_CPU15
   = 0
 ns_sketch_CPU15: - ns_single_CPU15 + 12.50006 cores_sketch_CPU15 >= 0
 capacity_cores_CPU15: cores_sketch_CPU15 <= 3
 ns_single_CPU3: 3.5 rows_tot_CPU3 - ns_single_CPU3 + pdt_m_rows_CPU3 = 0
 ns_sketch_CPU3: - ns_single_CPU3 + 12.50006 cores_sketch_CPU3 >= 0
 capacity_cores_CPU3: cores_sketch_CPU3 <= 3
 ns_single_CPU9: 3.5 rows_tot_CPU9 - ns_single_CPU9 + pdt_m_rows_CPU9 = 0
 ns_sketch_CPU9: - ns_single_CPU9 + 12.50006 cores_sketch_CPU9 >= 0
 capacity_cores_CPU9: cores_sketch_CPU9 <= 3
 res_acc: 1.220703125e-04 mem_tot_CPU17 + 1.220703125e-04 mem_tot_CPU29
   + 1.220703125e-04 mem_tot_CPU28 + 1.220703125e-04 mem_tot_CPU8
   + 1.220703125e-04 mem_tot_CPU6 + 1.220703125e-04 mem_tot_CPU23
   + 1.220703125e-04 mem_tot_CPU7 + 1.220703125e-04 mem_tot_CPU30
   + 1.220703125e-04 mem_tot_CPU10 + 1.220703125e-04 mem_tot_CPU31
   + 1.220703125e-04 mem_tot_CPU27 + 1.220703125e-04 mem_tot_CPU19
   + 1.220703125e-04 mem_tot_CPU1 + 1.220703125e-04 mem_tot_CPU15
   + 1.220703125e-04 mem_tot_CPU3 + 1.220703125e-04 mem_tot_CPU9
   + 10 cores_sketch_CPU17 + 10 cores_sketch_CPU29 + 10 cores_sketch_CPU28
   + 10 cores_sketch_CPU8 + 10 cores_sketch_CPU6 + 10 cores_sketch_CPU23
   + 10 cores_sketch_CPU7 + 10 cores_sketch_CPU30 + 10 cores_sketch_CPU10
   + 10 cores_sketch_CPU31 + 10 cores_sketch_CPU27 + 10 cores_sketch_CPU19
   + 10 cores_sketch_CPU1 + 10 cores_sketch_CPU15 + 10 cores_sketch_CPU3
   + 10 cores_sketch_CPU9 - res = -640
 pdt_m_rows_CPU17: pdt_m_rows_CPU17 + [
   - rows_tot_CPU17 * m_access_time_CPU17 ] = 0
 pdt_m_rows_CPU29: pdt_m_rows_CPU29 + [
   - rows_tot_CPU29 * m_access_time_CPU29 ] = 0
 pdt_m_rows_CPU28: pdt_m_rows_CPU28 + [
   - rows_tot_CPU28 * m_access_time_CPU28 ] = 0
 pdt_m_rows_CPU8: pdt_m_rows_CPU8 + [ - rows_tot_CPU8 * m_access_time_CPU8
   ] = 0
 pdt_m_rows_CPU6: pdt_m_rows_CPU6 + [ - rows_tot_CPU6 * m_access_time_CPU6
   ] = 0
 pdt_m_rows_CPU23: pdt_m_rows_CPU23 + [
   - rows_tot_CPU23 * m_access_time_CPU23 ] = 0
 pdt_m_rows_CPU7: pdt_m_rows_CPU7 + [ - rows_tot_CPU7 * m_access_time_CPU7
   ] = 0
 pdt_m_rows_CPU30: pdt_m_rows_CPU30 + [
   - rows_tot_CPU30 * m_access_time_CPU30 ] = 0
 pdt_m_rows_CPU10: pdt_m_rows_CPU10 + [
   - rows_tot_CPU10 * m_access_time_CPU10 ] = 0
 pdt_m_rows_CPU31: pdt_m_rows_CPU31 + [
   - rows_tot_CPU31 * m_access_time_CPU31 ] = 0
 pdt_m_rows_CPU27: pdt_m_rows_CPU27 + [
   - rows_tot_CPU27 * m_access_time_CPU27 ] = 0
 pdt_m_rows_CPU19: pdt_m_rows_CPU19 + [
   - rows_tot_CPU19 * m_access_time_CPU19 ] = 0
 pdt_m_rows_CPU1: pdt_m_rows_CPU1 + [ - rows_tot_CPU1 * m_access_time_CPU1
   ] = 0
 pdt_m_rows_CPU15: pdt_m_rows_CPU15 + [
   - rows_tot_CPU15 * m_access_time_CPU15 ] = 0
 pdt_m_rows_CPU3: pdt_m_rows_CPU3 + [ - rows_tot_CPU3 * m_access_time_CPU3
   ] = 0
 pdt_m_rows_CPU9: pdt_m_rows_CPU9 + [ - rows_tot_CPU9 * m_access_time_CPU9
   ] = 0
Bounds
 frac[0,10] <= 1
 frac[0,17] <= 1
 frac[0,1] <= 1
 frac[0,9] <= 1
 frac[0,3] <= 1
 frac[1,1] <= 1
 frac[1,10] <= 1
 frac[1,14] <= 1
 frac[1,9] <= 1
 frac[1,7] <= 1
 frac[2,17] <= 1
 frac[2,3] <= 1
 frac[2,14] <= 1
 frac[2,9] <= 1
 frac[3,17] <= 1
 frac[3,10] <= 1
 frac[3,9] <= 1
 frac[3,14] <= 1
 frac[3,3] <= 1
 frac[4,0] <= 1
 frac[4,10] <= 1
 frac[4,12] <= 1
 frac[4,1] <= 1
 frac[4,17] <= 1
 frac[4,9] <= 1
 frac[4,14] <= 1
 frac[4,3] <= 1
 frac[5,14] <= 1
 frac[5,1] <= 1
 frac[5,10] <= 1
 frac[5,7] <= 1
 frac[6,9] <= 1
 frac[6,7] <= 1
 frac[6,10] <= 1
 frac[6,0] <= 1
 frac[6,14] <= 1
 frac[6,1] <= 1
 frac[6,17] <= 1
 frac[7,12] <= 1
 frac[7,1] <= 1
 frac[7,17] <= 1
 frac[7,14] <= 1
 frac[7,7] <= 1
 frac[8,6] <= 1
 frac[8,8] <= 1
 frac[9,4] <= 1
 frac[9,13] <= 1
 frac[10,8] <= 1
 frac[10,13] <= 1
 frac[10,15] <= 1
 frac[10,5] <= 1
 frac[10,11] <= 1
 frac[10,2] <= 1
 frac[11,5] <= 1
 frac[11,16] <= 1
 frac[11,2] <= 1
 frac[12,11] <= 1
 frac[12,2] <= 1
 frac[12,16] <= 1
 frac[12,15] <= 1
 frac[13,13] <= 1
 frac[13,16] <= 1
 frac[13,5] <= 1
 frac[13,8] <= 1
 frac[13,11] <= 1
 frac[14,4] <= 1
 frac[14,15] <= 1
 frac[14,2] <= 1
 frac[14,8] <= 1
 frac[14,5] <= 1
 frac[14,6] <= 1
 frac[14,16] <= 1
 frac[15,16] <= 1
 frac[15,13] <= 1
 frac[15,6] <= 1
 frac[15,8] <= 1
 frac[15,5] <= 1
 frac[15,15] <= 1
 frac[15,2] <= 1
 mem_tot_CPU17 <= 32768
 normalized_mem_tot_CPU17 <= 1
 normalized_rows_tot_CPU17 <= 1
 mem_tot_CPU29 <= 32768
 normalized_mem_tot_CPU29 <= 1
 normalized_rows_tot_CPU29 <= 1
 mem_tot_CPU28 <= 32768
 normalized_mem_tot_CPU28 <= 1
 normalized_rows_tot_CPU28 <= 1
 mem_tot_CPU8 <= 32768
 normalized_mem_tot_CPU8 <= 1
 normalized_rows_tot_CPU8 <= 1
 mem_tot_CPU6 <= 32768
 normalized_mem_tot_CPU6 <= 1
 normalized_rows_tot_CPU6 <= 1
 mem_tot_CPU23 <= 32768
 normalized_mem_tot_CPU23 <= 1
 normalized_rows_tot_CPU23 <= 1
 mem_tot_CPU7 <= 32768
 normalized_mem_tot_CPU7 <= 1
 normalized_rows_tot_CPU7 <= 1
 mem_tot_CPU30 <= 32768
 normalized_mem_tot_CPU30 <= 1
 normalized_rows_tot_CPU30 <= 1
 mem_tot_CPU10 <= 32768
 normalized_mem_tot_CPU10 <= 1
 normalized_rows_tot_CPU10 <= 1
 mem_tot_CPU31 <= 32768
 normalized_mem_tot_CPU31 <= 1
 normalized_rows_tot_CPU31 <= 1
 mem_tot_CPU27 <= 32768
 normalized_mem_tot_CPU27 <= 1
 normalized_rows_tot_CPU27 <= 1
 mem_tot_CPU19 <= 32768
 normalized_mem_tot_CPU19 <= 1
 normalized_rows_tot_CPU19 <= 1
 mem_tot_CPU1 <= 32768
 normalized_mem_tot_CPU1 <= 1
 normalized_rows_tot_CPU1 <= 1
 mem_tot_CPU15 <= 32768
 normalized_mem_tot_CPU15 <= 1
 normalized_rows_tot_CPU15 <= 1
 mem_tot_CPU3 <= 32768
 normalized_mem_tot_CPU3 <= 1
 normalized_rows_tot_CPU3 <= 1
 mem_tot_CPU9 <= 32768
 normalized_mem_tot_CPU9 <= 1
 normalized_rows_tot_CPU9 <= 1
 cores_sketch_CPU17 <= 7
 cores_sketch_CPU29 <= 7
 cores_sketch_CPU28 <= 7
 cores_sketch_CPU8 <= 7
 cores_sketch_CPU6 <= 7
 cores_sketch_CPU23 <= 7
 cores_sketch_CPU7 <= 7
 cores_sketch_CPU30 <= 7
 cores_sketch_CPU10 <= 7
 cores_sketch_CPU31 <= 7
 cores_sketch_CPU27 <= 7
 cores_sketch_CPU19 <= 7
 cores_sketch_CPU1 <= 7
 cores_sketch_CPU15 <= 7
 cores_sketch_CPU3 <= 7
 cores_sketch_CPU9 <= 7
Generals
 cores_sketch_CPU17 cores_sketch_CPU29 cores_sketch_CPU28 cores_sketch_CPU8
 cores_sketch_CPU6 cores_sketch_CPU23 cores_sketch_CPU7 cores_sketch_CPU30
 cores_sketch_CPU10 cores_sketch_CPU31 cores_sketch_CPU27
 cores_sketch_CPU19 cores_sketch_CPU1 cores_sketch_CPU15 cores_sketch_CPU3
 cores_sketch_CPU9
General Constraints
 mem_overall_others: max_mem_others = MAX ( normalized_mem_tot_CPU17 ,
   normalized_mem_tot_CPU29 , normalized_mem_tot_CPU28 ,
   normalized_mem_tot_CPU8 , normalized_mem_tot_CPU6 ,
   normalized_mem_tot_CPU23 , normalized_mem_tot_CPU7 ,
   normalized_mem_tot_CPU30 , normalized_mem_tot_CPU10 ,
   normalized_mem_tot_CPU31 , normalized_mem_tot_CPU27 ,
   normalized_mem_tot_CPU19 , normalized_mem_tot_CPU1 ,
   normalized_mem_tot_CPU15 , normalized_mem_tot_CPU3 ,
   normalized_mem_tot_CPU9 )
 rows_overall_others: max_rows_others = MAX ( normalized_rows_tot_CPU17 ,
   normalized_rows_tot_CPU29 , normalized_rows_tot_CPU28 ,
   normalized_rows_tot_CPU8 , normalized_rows_tot_CPU6 ,
   normalized_rows_tot_CPU23 , normalized_rows_tot_CPU7 ,
   normalized_rows_tot_CPU30 , normalized_rows_tot_CPU10 ,
   normalized_rows_tot_CPU31 , normalized_rows_tot_CPU27 ,
   normalized_rows_tot_CPU19 , normalized_rows_tot_CPU1 ,
   normalized_rows_tot_CPU15 , normalized_rows_tot_CPU3 ,
   normalized_rows_tot_CPU9 )
 mem_access_time_CPU17: m_access_time_CPU17 = PWL ( mem_tot_CPU17 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU29: m_access_time_CPU29 = PWL ( mem_tot_CPU29 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU28: m_access_time_CPU28 = PWL ( mem_tot_CPU28 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU8: m_access_time_CPU8 = PWL ( mem_tot_CPU8 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU6: m_access_time_CPU6 = PWL ( mem_tot_CPU6 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU23: m_access_time_CPU23 = PWL ( mem_tot_CPU23 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU7: m_access_time_CPU7 = PWL ( mem_tot_CPU7 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU30: m_access_time_CPU30 = PWL ( mem_tot_CPU30 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU10: m_access_time_CPU10 = PWL ( mem_tot_CPU10 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU31: m_access_time_CPU31 = PWL ( mem_tot_CPU31 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU27: m_access_time_CPU27 = PWL ( mem_tot_CPU27 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU19: m_access_time_CPU19 = PWL ( mem_tot_CPU19 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1: m_access_time_CPU1 = PWL ( mem_tot_CPU1 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU15: m_access_time_CPU15 = PWL ( mem_tot_CPU15 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU3: m_access_time_CPU3 = PWL ( mem_tot_CPU3 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU9: m_access_time_CPU9 = PWL ( mem_tot_CPU9 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

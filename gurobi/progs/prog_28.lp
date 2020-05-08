\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_others + 0 max_rows_others + 0 m_access_time_CPU258
   + 0 m_access_time_CPU600 + 0 m_access_time_CPU19
   + 0 m_access_time_CPU743 + 0 m_access_time_CPU207
   + 0 m_access_time_CPU427 + 0 m_access_time_CPU80
   + 0 m_access_time_CPU164 + 0 m_access_time_CPU468
   + 0 m_access_time_CPU241 + 0 m_access_time_CPU734
   + 0 m_access_time_CPU151 + 0 m_access_time_CPU423
   + 0 m_access_time_CPU452 + 0 m_access_time_CPU635
   + 0 m_access_time_CPU198 + res
Subject To
 cov_0_23: frac[3,23] + frac[5,23] >= 0.9
 cov_1_2: frac[3,2] + frac[6,2] >= 0.9
 cov_2_18: frac[0,18] + frac[7,18] >= 0.7
 cov_3_18: frac[6,18] + frac[7,18] >= 0.7
 cov_4_12: frac[2,12] + frac[4,12] >= 0.8
 cov_5_7: frac[4,7] + frac[6,7] >= 0.8
 cov_6_7: frac[2,7] + frac[3,7] >= 0.8
 cov_7_8: frac[4,8] + frac[5,8] >= 0.8
 cov_8_19: frac[2,19] + frac[3,19] >= 0.7
 cov_9_1: frac[1,1] + frac[4,1] >= 0.8
 cov_10_4: frac[4,4] + frac[7,4] >= 0.9
 cov_11_13: frac[0,13] + frac[1,13] >= 0.9
 cov_12_18: frac[2,18] + frac[4,18] >= 1
 cov_13_18: frac[0,18] + frac[3,18] >= 0.7
 cov_14_1: frac[4,1] + frac[5,1] >= 0.9
 cov_15_4: frac[1,4] + frac[5,4] >= 0.9
 cov_16_24: frac[1,24] + frac[4,24] >= 0.441032
 cov_17_12: frac[1,12] + frac[6,12] >= 0.7
 cov_18_12: frac[4,12] + frac[5,12] >= 1
 cov_19_8: frac[2,8] + frac[5,8] >= 0.8
 cov_20_1: frac[6,1] + frac[7,1] >= 1
 cov_21_22: frac[2,22] + frac[7,22] >= 1
 cov_22_22: frac[0,22] + frac[5,22] >= 0.9
 cov_23_18: frac[1,18] + frac[6,18] >= 0.7
 cov_24_24: frac[2,24] + frac[6,24] >= 0.441032
 cov_25_19: frac[2,19] + frac[6,19] >= 0.8
 cov_26_7: frac[5,7] + frac[6,7] >= 0.7
 cov_27_1: frac[1,1] + frac[3,1] >= 1
 cov_28_7: frac[0,7] + frac[7,7] >= 0.8
 cov_29_19: frac[1,19] + frac[3,19] >= 0.8
 cov_30_1: frac[2,1] + frac[4,1] >= 0.9
 cov_31_4: frac[2,4] + frac[6,4] >= 1
 cov_32_0: frac[10,0] + frac[13,0] >= 0.9
 cov_33_3: frac[8,3] + frac[14,3] >= 0.8
 cov_34_6: frac[8,6] + frac[15,6] >= 0.8
 cov_35_9: frac[8,9] + frac[15,9] >= 0.9
 cov_36_17: frac[8,17] + frac[14,17] >= 0.7
 cov_37_16: frac[8,16] + frac[11,16] >= 0.8
 cov_38_17: frac[12,17] + frac[13,17] >= 1
 cov_39_16: frac[11,16] + frac[12,16] >= 1
 cov_40_25: frac[9,25] + frac[14,25] >= 0.9
 cov_41_10: frac[9,10] + frac[11,10] >= 1
 cov_42_21: frac[10,21] + frac[13,21] >= 0.9
 cov_43_21: frac[9,21] + frac[15,21] >= 0.7
 cov_44_9: frac[12,9] + frac[13,9] >= 0.9
 cov_45_25: frac[9,25] + frac[10,25] >= 0.9
 cov_46_5: frac[12,5] + frac[14,5] >= 1
 cov_47_11: frac[10,11] + frac[12,11] >= 0.441032
 cov_48_6: frac[9,6] + frac[10,6] >= 0.9
 cov_49_20: frac[9,20] + frac[14,20] >= 0.9
 cov_50_9: frac[9,9] + frac[10,9] >= 1
 cov_51_14: frac[11,14] + frac[15,14] >= 1
 cov_52_0: frac[9,0] + frac[11,0] >= 0.7
 cov_53_21: frac[11,21] + frac[13,21] >= 0.8
 cov_54_10: frac[12,10] + frac[13,10] >= 0.7
 cov_55_15: frac[10,15] + frac[12,15] >= 0.8
 cov_56_17: frac[8,17] + frac[11,17] >= 1
 cov_57_10: frac[12,10] + frac[13,10] >= 0.8
 cov_58_25: frac[10,25] + frac[14,25] >= 1
 cov_59_16: frac[9,16] + frac[15,16] >= 1
 cov_60_6: frac[10,6] + frac[11,6] >= 0.7
 cov_61_6: frac[9,6] + frac[14,6] >= 0.7
 cov_62_21: frac[12,21] + frac[14,21] >= 0.7
 cov_63_5: frac[13,5] + frac[14,5] >= 1
 accuracy_14_17: - 50.9765625 frac[14,17] + mem[14,17] = 0
 accuracy_0_7: - 50.9765625 frac[0,7] + mem[0,7] = 0
 accuracy_13_17: - 50.9765625 frac[13,17] + mem[13,17] = 0
 accuracy_2_22: - 50.9765625 frac[2,22] + mem[2,22] = 0
 accuracy_3_7: - 50.9765625 frac[3,7] + mem[3,7] = 0
 accuracy_5_8: - 50.9765625 frac[5,8] + mem[5,8] = 0
 accuracy_9_0: - 50.9765625 frac[9,0] + mem[9,0] = 0
 accuracy_6_7: - 50.9765625 frac[6,7] + mem[6,7] = 0
 accuracy_9_21: - 50.9765625 frac[9,21] + mem[9,21] = 0
 accuracy_12_17: - 50.9765625 frac[12,17] + mem[12,17] = 0
 accuracy_1_1: - 50.9765625 frac[1,1] + mem[1,1] = 0
 accuracy_3_2: - 50.9765625 frac[3,2] + mem[3,2] = 0
 accuracy_3_23: - 50.9765625 frac[3,23] + mem[3,23] = 0
 accuracy_11_0: - 50.9765625 frac[11,0] + mem[11,0] = 0
 accuracy_9_16: - 50.9765625 frac[9,16] + mem[9,16] = 0
 accuracy_12_11: - 50.9765625 frac[12,11] + mem[12,11] = 0
 accuracy_11_21: - 50.9765625 frac[11,21] + mem[11,21] = 0
 accuracy_7_22: - 50.9765625 frac[7,22] + mem[7,22] = 0
 accuracy_15_14: - 50.9765625 frac[15,14] + mem[15,14] = 0
 accuracy_13_10: - 50.9765625 frac[13,10] + mem[13,10] = 0
 accuracy_1_12: - 50.9765625 frac[1,12] + mem[1,12] = 0
 accuracy_3_1: - 50.9765625 frac[3,1] + mem[3,1] = 0
 accuracy_9_9: - 50.9765625 frac[9,9] + mem[9,9] = 0
 accuracy_3_18: - 50.9765625 frac[3,18] + mem[3,18] = 0
 accuracy_14_21: - 50.9765625 frac[14,21] + mem[14,21] = 0
 accuracy_15_16: - 50.9765625 frac[15,16] + mem[15,16] = 0
 accuracy_13_0: - 50.9765625 frac[13,0] + mem[13,0] = 0
 accuracy_11_16: - 50.9765625 frac[11,16] + mem[11,16] = 0
 accuracy_2_18: - 50.9765625 frac[2,18] + mem[2,18] = 0
 accuracy_2_1: - 50.9765625 frac[2,1] + mem[2,1] = 0
 accuracy_8_9: - 50.9765625 frac[8,9] + mem[8,9] = 0
 accuracy_4_12: - 50.9765625 frac[4,12] + mem[4,12] = 0
 accuracy_2_12: - 50.9765625 frac[2,12] + mem[2,12] = 0
 accuracy_5_1: - 50.9765625 frac[5,1] + mem[5,1] = 0
 accuracy_9_25: - 50.9765625 frac[9,25] + mem[9,25] = 0
 accuracy_11_10: - 50.9765625 frac[11,10] + mem[11,10] = 0
 accuracy_14_5: - 50.9765625 frac[14,5] + mem[14,5] = 0
 accuracy_8_6: - 50.9765625 frac[8,6] + mem[8,6] = 0
 accuracy_4_1: - 50.9765625 frac[4,1] + mem[4,1] = 0
 accuracy_10_9: - 50.9765625 frac[10,9] + mem[10,9] = 0
 accuracy_6_4: - 50.9765625 frac[6,4] + mem[6,4] = 0
 accuracy_5_4: - 50.9765625 frac[5,4] + mem[5,4] = 0
 accuracy_9_20: - 50.9765625 frac[9,20] + mem[9,20] = 0
 accuracy_7_1: - 50.9765625 frac[7,1] + mem[7,1] = 0
 accuracy_0_18: - 50.9765625 frac[0,18] + mem[0,18] = 0
 accuracy_7_18: - 50.9765625 frac[7,18] + mem[7,18] = 0
 accuracy_14_6: - 50.9765625 frac[14,6] + mem[14,6] = 0
 accuracy_1_19: - 50.9765625 frac[1,19] + mem[1,19] = 0
 accuracy_2_24: - 50.9765625 frac[2,24] + mem[2,24] = 0
 accuracy_2_7: - 50.9765625 frac[2,7] + mem[2,7] = 0
 accuracy_8_3: - 50.9765625 frac[8,3] + mem[8,3] = 0
 accuracy_6_1: - 50.9765625 frac[6,1] + mem[6,1] = 0
 accuracy_5_7: - 50.9765625 frac[5,7] + mem[5,7] = 0
 accuracy_14_25: - 50.9765625 frac[14,25] + mem[14,25] = 0
 accuracy_10_25: - 50.9765625 frac[10,25] + mem[10,25] = 0
 accuracy_7_4: - 50.9765625 frac[7,4] + mem[7,4] = 0
 accuracy_15_9: - 50.9765625 frac[15,9] + mem[15,9] = 0
 accuracy_14_3: - 50.9765625 frac[14,3] + mem[14,3] = 0
 accuracy_13_9: - 50.9765625 frac[13,9] + mem[13,9] = 0
 accuracy_4_8: - 50.9765625 frac[4,8] + mem[4,8] = 0
 accuracy_2_8: - 50.9765625 frac[2,8] + mem[2,8] = 0
 accuracy_10_15: - 50.9765625 frac[10,15] + mem[10,15] = 0
 accuracy_6_2: - 50.9765625 frac[6,2] + mem[6,2] = 0
 accuracy_11_14: - 50.9765625 frac[11,14] + mem[11,14] = 0
 accuracy_5_23: - 50.9765625 frac[5,23] + mem[5,23] = 0
 accuracy_12_9: - 50.9765625 frac[12,9] + mem[12,9] = 0
 accuracy_2_19: - 50.9765625 frac[2,19] + mem[2,19] = 0
 accuracy_4_24: - 50.9765625 frac[4,24] + mem[4,24] = 0
 accuracy_10_0: - 50.9765625 frac[10,0] + mem[10,0] = 0
 accuracy_8_16: - 50.9765625 frac[8,16] + mem[8,16] = 0
 accuracy_0_22: - 50.9765625 frac[0,22] + mem[0,22] = 0
 accuracy_6_18: - 50.9765625 frac[6,18] + mem[6,18] = 0
 accuracy_1_4: - 50.9765625 frac[1,4] + mem[1,4] = 0
 accuracy_9_6: - 50.9765625 frac[9,6] + mem[9,6] = 0
 accuracy_15_21: - 50.9765625 frac[15,21] + mem[15,21] = 0
 accuracy_13_5: - 50.9765625 frac[13,5] + mem[13,5] = 0
 accuracy_1_18: - 50.9765625 frac[1,18] + mem[1,18] = 0
 accuracy_15_6: - 50.9765625 frac[15,6] + mem[15,6] = 0
 accuracy_2_4: - 50.9765625 frac[2,4] + mem[2,4] = 0
 accuracy_4_7: - 50.9765625 frac[4,7] + mem[4,7] = 0
 accuracy_10_11: - 50.9765625 frac[10,11] + mem[10,11] = 0
 accuracy_4_18: - 50.9765625 frac[4,18] + mem[4,18] = 0
 accuracy_10_6: - 50.9765625 frac[10,6] + mem[10,6] = 0
 accuracy_7_7: - 50.9765625 frac[7,7] + mem[7,7] = 0
 accuracy_12_5: - 50.9765625 frac[12,5] + mem[12,5] = 0
 accuracy_10_21: - 50.9765625 frac[10,21] + mem[10,21] = 0
 accuracy_6_24: - 50.9765625 frac[6,24] + mem[6,24] = 0
 accuracy_12_16: - 50.9765625 frac[12,16] + mem[12,16] = 0
 accuracy_13_21: - 50.9765625 frac[13,21] + mem[13,21] = 0
 accuracy_5_12: - 50.9765625 frac[5,12] + mem[5,12] = 0
 accuracy_4_4: - 50.9765625 frac[4,4] + mem[4,4] = 0
 accuracy_5_22: - 50.9765625 frac[5,22] + mem[5,22] = 0
 accuracy_12_10: - 50.9765625 frac[12,10] + mem[12,10] = 0
 accuracy_12_21: - 50.9765625 frac[12,21] + mem[12,21] = 0
 accuracy_1_24: - 50.9765625 frac[1,24] + mem[1,24] = 0
 accuracy_1_13: - 50.9765625 frac[1,13] + mem[1,13] = 0
 accuracy_9_10: - 50.9765625 frac[9,10] + mem[9,10] = 0
 accuracy_3_19: - 50.9765625 frac[3,19] + mem[3,19] = 0
 accuracy_8_17: - 50.9765625 frac[8,17] + mem[8,17] = 0
 accuracy_6_12: - 50.9765625 frac[6,12] + mem[6,12] = 0
 accuracy_14_20: - 50.9765625 frac[14,20] + mem[14,20] = 0
 accuracy_6_19: - 50.9765625 frac[6,19] + mem[6,19] = 0
 accuracy_12_15: - 50.9765625 frac[12,15] + mem[12,15] = 0
 accuracy_11_17: - 50.9765625 frac[11,17] + mem[11,17] = 0
 accuracy_0_13: - 50.9765625 frac[0,13] + mem[0,13] = 0
 accuracy_11_6: - 50.9765625 frac[11,6] + mem[11,6] = 0
 mem_tot_CPU258: - mem[0,7] - mem[0,18] - mem[0,22] - mem[0,13]
   + mem_tot_CPU258 = 0
 R171: - 3.05176e-05 mem_tot_CPU258 + normalized_mem_tot_CPU258 = 0
 rows_tot_CPU258: - 3 frac[0,7] - 3 frac[0,18] - 3 frac[0,22]
   - 3 frac[0,13] + rows_tot_CPU258 = 0
 R173: - 0.0833333333333333 rows_tot_CPU258 + normalized_rows_tot_CPU258
   = 0
 mem_tot_CPU600: - mem[1,1] - mem[1,12] - mem[1,19] - mem[1,4] - mem[1,18]
   - mem[1,24] - mem[1,13] + mem_tot_CPU600 = 0
 R175: - 3.05176e-05 mem_tot_CPU600 + normalized_mem_tot_CPU600 = 0
 rows_tot_CPU600: - 3 frac[1,1] - 3 frac[1,12] - 3 frac[1,19] - 3 frac[1,4]
   - 3 frac[1,18] - 3 frac[1,24] - 3 frac[1,13] + rows_tot_CPU600 = 0
 R177: - 0.0833333333333333 rows_tot_CPU600 + normalized_rows_tot_CPU600
   = 0
 mem_tot_CPU19: - mem[2,22] - mem[2,18] - mem[2,1] - mem[2,12] - mem[2,24]
   - mem[2,7] - mem[2,8] - mem[2,19] - mem[2,4] + mem_tot_CPU19 = 0
 R179: - 3.05176e-05 mem_tot_CPU19 + normalized_mem_tot_CPU19 = 0
 rows_tot_CPU19: - 3 frac[2,22] - 3 frac[2,18] - 3 frac[2,1] - 3 frac[2,12]
   - 3 frac[2,24] - 3 frac[2,7] - 3 frac[2,8] - 3 frac[2,19] - 3 frac[2,4]
   + rows_tot_CPU19 = 0
 R181: - 0.0833333333333333 rows_tot_CPU19 + normalized_rows_tot_CPU19 = 0
 mem_tot_CPU743: - mem[3,7] - mem[3,2] - mem[3,23] - mem[3,1] - mem[3,18]
   - mem[3,19] + mem_tot_CPU743 = 0
 R183: - 3.05176e-05 mem_tot_CPU743 + normalized_mem_tot_CPU743 = 0
 rows_tot_CPU743: - 3 frac[3,7] - 3 frac[3,2] - 3 frac[3,23] - 3 frac[3,1]
   - 3 frac[3,18] - 3 frac[3,19] + rows_tot_CPU743 = 0
 R185: - 0.0833333333333333 rows_tot_CPU743 + normalized_rows_tot_CPU743
   = 0
 mem_tot_CPU207: - mem[4,12] - mem[4,1] - mem[4,8] - mem[4,24] - mem[4,7]
   - mem[4,18] - mem[4,4] + mem_tot_CPU207 = 0
 R187: - 3.05176e-05 mem_tot_CPU207 + normalized_mem_tot_CPU207 = 0
 rows_tot_CPU207: - 3 frac[4,12] - 3 frac[4,1] - 3 frac[4,8] - 3 frac[4,24]
   - 3 frac[4,7] - 3 frac[4,18] - 3 frac[4,4] + rows_tot_CPU207 = 0
 R189: - 0.0833333333333333 rows_tot_CPU207 + normalized_rows_tot_CPU207
   = 0
 mem_tot_CPU427: - mem[5,8] - mem[5,1] - mem[5,4] - mem[5,7] - mem[5,23]
   - mem[5,12] - mem[5,22] + mem_tot_CPU427 = 0
 R191: - 3.05176e-05 mem_tot_CPU427 + normalized_mem_tot_CPU427 = 0
 rows_tot_CPU427: - 3 frac[5,8] - 3 frac[5,1] - 3 frac[5,4] - 3 frac[5,7]
   - 3 frac[5,23] - 3 frac[5,12] - 3 frac[5,22] + rows_tot_CPU427 = 0
 R193: - 0.0833333333333333 rows_tot_CPU427 + normalized_rows_tot_CPU427
   = 0
 mem_tot_CPU80: - mem[6,7] - mem[6,4] - mem[6,1] - mem[6,2] - mem[6,18]
   - mem[6,24] - mem[6,12] - mem[6,19] + mem_tot_CPU80 = 0
 R195: - 3.05176e-05 mem_tot_CPU80 + normalized_mem_tot_CPU80 = 0
 rows_tot_CPU80: - 3 frac[6,7] - 3 frac[6,4] - 3 frac[6,1] - 3 frac[6,2]
   - 3 frac[6,18] - 3 frac[6,24] - 3 frac[6,12] - 3 frac[6,19]
   + rows_tot_CPU80 = 0
 R197: - 0.0833333333333333 rows_tot_CPU80 + normalized_rows_tot_CPU80 = 0
 mem_tot_CPU164: - mem[7,22] - mem[7,1] - mem[7,18] - mem[7,4] - mem[7,7]
   + mem_tot_CPU164 = 0
 R199: - 3.05176e-05 mem_tot_CPU164 + normalized_mem_tot_CPU164 = 0
 rows_tot_CPU164: - 3 frac[7,22] - 3 frac[7,1] - 3 frac[7,18] - 3 frac[7,4]
   - 3 frac[7,7] + rows_tot_CPU164 = 0
 R201: - 0.0833333333333333 rows_tot_CPU164 + normalized_rows_tot_CPU164
   = 0
 mem_tot_CPU468: - mem[8,9] - mem[8,6] - mem[8,3] - mem[8,16] - mem[8,17]
   + mem_tot_CPU468 = 0
 R203: - 3.05176e-05 mem_tot_CPU468 + normalized_mem_tot_CPU468 = 0
 rows_tot_CPU468: - 3 frac[8,9] - 3 frac[8,6] - 3 frac[8,3] - 3 frac[8,16]
   - 3 frac[8,17] + rows_tot_CPU468 = 0
 R205: - 0.0833333333333333 rows_tot_CPU468 + normalized_rows_tot_CPU468
   = 0
 mem_tot_CPU241: - mem[9,0] - mem[9,21] - mem[9,16] - mem[9,9] - mem[9,25]
   - mem[9,20] - mem[9,6] - mem[9,10] + mem_tot_CPU241 = 0
 R207: - 3.05176e-05 mem_tot_CPU241 + normalized_mem_tot_CPU241 = 0
 rows_tot_CPU241: - 3 frac[9,0] - 3 frac[9,21] - 3 frac[9,16] - 3 frac[9,9]
   - 3 frac[9,25] - 3 frac[9,20] - 3 frac[9,6] - 3 frac[9,10]
   + rows_tot_CPU241 = 0
 R209: - 0.0833333333333333 rows_tot_CPU241 + normalized_rows_tot_CPU241
   = 0
 mem_tot_CPU734: - mem[10,9] - mem[10,25] - mem[10,15] - mem[10,0]
   - mem[10,11] - mem[10,6] - mem[10,21] + mem_tot_CPU734 = 0
 R211: - 3.05176e-05 mem_tot_CPU734 + normalized_mem_tot_CPU734 = 0
 rows_tot_CPU734: - 3 frac[10,9] - 3 frac[10,25] - 3 frac[10,15]
   - 3 frac[10,0] - 3 frac[10,11] - 3 frac[10,6] - 3 frac[10,21]
   + rows_tot_CPU734 = 0
 R213: - 0.0833333333333333 rows_tot_CPU734 + normalized_rows_tot_CPU734
   = 0
 mem_tot_CPU151: - mem[11,0] - mem[11,21] - mem[11,16] - mem[11,10]
   - mem[11,14] - mem[11,17] - mem[11,6] + mem_tot_CPU151 = 0
 R215: - 3.05176e-05 mem_tot_CPU151 + normalized_mem_tot_CPU151 = 0
 rows_tot_CPU151: - 3 frac[11,0] - 3 frac[11,21] - 3 frac[11,16]
   - 3 frac[11,10] - 3 frac[11,14] - 3 frac[11,17] - 3 frac[11,6]
   + rows_tot_CPU151 = 0
 R217: - 0.0833333333333333 rows_tot_CPU151 + normalized_rows_tot_CPU151
   = 0
 mem_tot_CPU423: - mem[12,17] - mem[12,11] - mem[12,9] - mem[12,5]
   - mem[12,16] - mem[12,10] - mem[12,21] - mem[12,15] + mem_tot_CPU423
   = 0
 R219: - 3.05176e-05 mem_tot_CPU423 + normalized_mem_tot_CPU423 = 0
 rows_tot_CPU423: - 3 frac[12,17] - 3 frac[12,11] - 3 frac[12,9]
   - 3 frac[12,5] - 3 frac[12,16] - 3 frac[12,10] - 3 frac[12,21]
   - 3 frac[12,15] + rows_tot_CPU423 = 0
 R221: - 0.0833333333333333 rows_tot_CPU423 + normalized_rows_tot_CPU423
   = 0
 mem_tot_CPU452: - mem[13,17] - mem[13,10] - mem[13,0] - mem[13,9]
   - mem[13,5] - mem[13,21] + mem_tot_CPU452 = 0
 R223: - 3.05176e-05 mem_tot_CPU452 + normalized_mem_tot_CPU452 = 0
 rows_tot_CPU452: - 3 frac[13,17] - 3 frac[13,10] - 3 frac[13,0]
   - 3 frac[13,9] - 3 frac[13,5] - 3 frac[13,21] + rows_tot_CPU452 = 0
 R225: - 0.0833333333333333 rows_tot_CPU452 + normalized_rows_tot_CPU452
   = 0
 mem_tot_CPU635: - mem[14,17] - mem[14,21] - mem[14,5] - mem[14,6]
   - mem[14,25] - mem[14,3] - mem[14,20] + mem_tot_CPU635 = 0
 R227: - 3.05176e-05 mem_tot_CPU635 + normalized_mem_tot_CPU635 = 0
 rows_tot_CPU635: - 3 frac[14,17] - 3 frac[14,21] - 3 frac[14,5]
   - 3 frac[14,6] - 3 frac[14,25] - 3 frac[14,3] - 3 frac[14,20]
   + rows_tot_CPU635 = 0
 R229: - 0.0833333333333333 rows_tot_CPU635 + normalized_rows_tot_CPU635
   = 0
 mem_tot_CPU198: - mem[15,14] - mem[15,16] - mem[15,9] - mem[15,21]
   - mem[15,6] + mem_tot_CPU198 = 0
 R231: - 3.05176e-05 mem_tot_CPU198 + normalized_mem_tot_CPU198 = 0
 rows_tot_CPU198: - 3 frac[15,14] - 3 frac[15,16] - 3 frac[15,9]
   - 3 frac[15,21] - 3 frac[15,6] + rows_tot_CPU198 = 0
 R233: - 0.0833333333333333 rows_tot_CPU198 + normalized_rows_tot_CPU198
   = 0
 row_capacity_CPU258: rows_tot_CPU258 <= 12
 row_capacity_CPU600: rows_tot_CPU600 <= 12
 row_capacity_CPU19: rows_tot_CPU19 <= 12
 row_capacity_CPU743: rows_tot_CPU743 <= 12
 row_capacity_CPU207: rows_tot_CPU207 <= 12
 row_capacity_CPU427: rows_tot_CPU427 <= 12
 row_capacity_CPU80: rows_tot_CPU80 <= 12
 row_capacity_CPU164: rows_tot_CPU164 <= 12
 row_capacity_CPU468: rows_tot_CPU468 <= 12
 row_capacity_CPU241: rows_tot_CPU241 <= 12
 row_capacity_CPU734: rows_tot_CPU734 <= 12
 row_capacity_CPU151: rows_tot_CPU151 <= 12
 row_capacity_CPU423: rows_tot_CPU423 <= 12
 row_capacity_CPU452: rows_tot_CPU452 <= 12
 row_capacity_CPU635: rows_tot_CPU635 <= 12
 row_capacity_CPU198: rows_tot_CPU198 <= 12
 ns_single_CPU258: 3.5 rows_tot_CPU258 - ns_single_CPU258
   + pdt_m_rows_CPU258 = 0
 ns_sketch_CPU258: - ns_single_CPU258 + 12.50006 cores_sketch_CPU258 >= 0
 capacity_cores_CPU258: cores_sketch_CPU258 <= 3
 ns_single_CPU600: 3.5 rows_tot_CPU600 - ns_single_CPU600
   + pdt_m_rows_CPU600 = 0
 ns_sketch_CPU600: - ns_single_CPU600 + 12.50006 cores_sketch_CPU600 >= 0
 capacity_cores_CPU600: cores_sketch_CPU600 <= 3
 ns_single_CPU19: 3.5 rows_tot_CPU19 - ns_single_CPU19 + pdt_m_rows_CPU19
   = 0
 ns_sketch_CPU19: - ns_single_CPU19 + 12.50006 cores_sketch_CPU19 >= 0
 capacity_cores_CPU19: cores_sketch_CPU19 <= 3
 ns_single_CPU743: 3.5 rows_tot_CPU743 - ns_single_CPU743
   + pdt_m_rows_CPU743 = 0
 ns_sketch_CPU743: - ns_single_CPU743 + 12.50006 cores_sketch_CPU743 >= 0
 capacity_cores_CPU743: cores_sketch_CPU743 <= 3
 ns_single_CPU207: 3.5 rows_tot_CPU207 - ns_single_CPU207
   + pdt_m_rows_CPU207 = 0
 ns_sketch_CPU207: - ns_single_CPU207 + 12.50006 cores_sketch_CPU207 >= 0
 capacity_cores_CPU207: cores_sketch_CPU207 <= 3
 ns_single_CPU427: 3.5 rows_tot_CPU427 - ns_single_CPU427
   + pdt_m_rows_CPU427 = 0
 ns_sketch_CPU427: - ns_single_CPU427 + 12.50006 cores_sketch_CPU427 >= 0
 capacity_cores_CPU427: cores_sketch_CPU427 <= 3
 ns_single_CPU80: 3.5 rows_tot_CPU80 - ns_single_CPU80 + pdt_m_rows_CPU80
   = 0
 ns_sketch_CPU80: - ns_single_CPU80 + 12.50006 cores_sketch_CPU80 >= 0
 capacity_cores_CPU80: cores_sketch_CPU80 <= 3
 ns_single_CPU164: 3.5 rows_tot_CPU164 - ns_single_CPU164
   + pdt_m_rows_CPU164 = 0
 ns_sketch_CPU164: - ns_single_CPU164 + 12.50006 cores_sketch_CPU164 >= 0
 capacity_cores_CPU164: cores_sketch_CPU164 <= 3
 ns_single_CPU468: 3.5 rows_tot_CPU468 - ns_single_CPU468
   + pdt_m_rows_CPU468 = 0
 ns_sketch_CPU468: - ns_single_CPU468 + 12.50006 cores_sketch_CPU468 >= 0
 capacity_cores_CPU468: cores_sketch_CPU468 <= 3
 ns_single_CPU241: 3.5 rows_tot_CPU241 - ns_single_CPU241
   + pdt_m_rows_CPU241 = 0
 ns_sketch_CPU241: - ns_single_CPU241 + 12.50006 cores_sketch_CPU241 >= 0
 capacity_cores_CPU241: cores_sketch_CPU241 <= 3
 ns_single_CPU734: 3.5 rows_tot_CPU734 - ns_single_CPU734
   + pdt_m_rows_CPU734 = 0
 ns_sketch_CPU734: - ns_single_CPU734 + 12.50006 cores_sketch_CPU734 >= 0
 capacity_cores_CPU734: cores_sketch_CPU734 <= 3
 ns_single_CPU151: 3.5 rows_tot_CPU151 - ns_single_CPU151
   + pdt_m_rows_CPU151 = 0
 ns_sketch_CPU151: - ns_single_CPU151 + 12.50006 cores_sketch_CPU151 >= 0
 capacity_cores_CPU151: cores_sketch_CPU151 <= 3
 ns_single_CPU423: 3.5 rows_tot_CPU423 - ns_single_CPU423
   + pdt_m_rows_CPU423 = 0
 ns_sketch_CPU423: - ns_single_CPU423 + 12.50006 cores_sketch_CPU423 >= 0
 capacity_cores_CPU423: cores_sketch_CPU423 <= 3
 ns_single_CPU452: 3.5 rows_tot_CPU452 - ns_single_CPU452
   + pdt_m_rows_CPU452 = 0
 ns_sketch_CPU452: - ns_single_CPU452 + 12.50006 cores_sketch_CPU452 >= 0
 capacity_cores_CPU452: cores_sketch_CPU452 <= 3
 ns_single_CPU635: 3.5 rows_tot_CPU635 - ns_single_CPU635
   + pdt_m_rows_CPU635 = 0
 ns_sketch_CPU635: - ns_single_CPU635 + 12.50006 cores_sketch_CPU635 >= 0
 capacity_cores_CPU635: cores_sketch_CPU635 <= 3
 ns_single_CPU198: 3.5 rows_tot_CPU198 - ns_single_CPU198
   + pdt_m_rows_CPU198 = 0
 ns_sketch_CPU198: - ns_single_CPU198 + 12.50006 cores_sketch_CPU198 >= 0
 capacity_cores_CPU198: cores_sketch_CPU198 <= 3
 res_acc: 1.220703125e-04 mem_tot_CPU258 + 1.220703125e-04 mem_tot_CPU600
   + 1.220703125e-04 mem_tot_CPU19 + 1.220703125e-04 mem_tot_CPU743
   + 1.220703125e-04 mem_tot_CPU207 + 1.220703125e-04 mem_tot_CPU427
   + 1.220703125e-04 mem_tot_CPU80 + 1.220703125e-04 mem_tot_CPU164
   + 1.220703125e-04 mem_tot_CPU468 + 1.220703125e-04 mem_tot_CPU241
   + 1.220703125e-04 mem_tot_CPU734 + 1.220703125e-04 mem_tot_CPU151
   + 1.220703125e-04 mem_tot_CPU423 + 1.220703125e-04 mem_tot_CPU452
   + 1.220703125e-04 mem_tot_CPU635 + 1.220703125e-04 mem_tot_CPU198
   + 10 cores_sketch_CPU258 + 10 cores_sketch_CPU600
   + 10 cores_sketch_CPU19 + 10 cores_sketch_CPU743
   + 10 cores_sketch_CPU207 + 10 cores_sketch_CPU427
   + 10 cores_sketch_CPU80 + 10 cores_sketch_CPU164
   + 10 cores_sketch_CPU468 + 10 cores_sketch_CPU241
   + 10 cores_sketch_CPU734 + 10 cores_sketch_CPU151
   + 10 cores_sketch_CPU423 + 10 cores_sketch_CPU452
   + 10 cores_sketch_CPU635 + 10 cores_sketch_CPU198 - res = -640
 pdt_m_rows_CPU258: pdt_m_rows_CPU258 + [
   - rows_tot_CPU258 * m_access_time_CPU258 ] = 0
 pdt_m_rows_CPU600: pdt_m_rows_CPU600 + [
   - rows_tot_CPU600 * m_access_time_CPU600 ] = 0
 pdt_m_rows_CPU19: pdt_m_rows_CPU19 + [
   - rows_tot_CPU19 * m_access_time_CPU19 ] = 0
 pdt_m_rows_CPU743: pdt_m_rows_CPU743 + [
   - rows_tot_CPU743 * m_access_time_CPU743 ] = 0
 pdt_m_rows_CPU207: pdt_m_rows_CPU207 + [
   - rows_tot_CPU207 * m_access_time_CPU207 ] = 0
 pdt_m_rows_CPU427: pdt_m_rows_CPU427 + [
   - rows_tot_CPU427 * m_access_time_CPU427 ] = 0
 pdt_m_rows_CPU80: pdt_m_rows_CPU80 + [
   - rows_tot_CPU80 * m_access_time_CPU80 ] = 0
 pdt_m_rows_CPU164: pdt_m_rows_CPU164 + [
   - rows_tot_CPU164 * m_access_time_CPU164 ] = 0
 pdt_m_rows_CPU468: pdt_m_rows_CPU468 + [
   - rows_tot_CPU468 * m_access_time_CPU468 ] = 0
 pdt_m_rows_CPU241: pdt_m_rows_CPU241 + [
   - rows_tot_CPU241 * m_access_time_CPU241 ] = 0
 pdt_m_rows_CPU734: pdt_m_rows_CPU734 + [
   - rows_tot_CPU734 * m_access_time_CPU734 ] = 0
 pdt_m_rows_CPU151: pdt_m_rows_CPU151 + [
   - rows_tot_CPU151 * m_access_time_CPU151 ] = 0
 pdt_m_rows_CPU423: pdt_m_rows_CPU423 + [
   - rows_tot_CPU423 * m_access_time_CPU423 ] = 0
 pdt_m_rows_CPU452: pdt_m_rows_CPU452 + [
   - rows_tot_CPU452 * m_access_time_CPU452 ] = 0
 pdt_m_rows_CPU635: pdt_m_rows_CPU635 + [
   - rows_tot_CPU635 * m_access_time_CPU635 ] = 0
 pdt_m_rows_CPU198: pdt_m_rows_CPU198 + [
   - rows_tot_CPU198 * m_access_time_CPU198 ] = 0
Bounds
 frac[0,7] <= 1
 frac[0,18] <= 1
 frac[0,22] <= 1
 frac[0,13] <= 1
 frac[1,1] <= 1
 frac[1,12] <= 1
 frac[1,19] <= 1
 frac[1,4] <= 1
 frac[1,18] <= 1
 frac[1,24] <= 1
 frac[1,13] <= 1
 frac[2,22] <= 1
 frac[2,18] <= 1
 frac[2,1] <= 1
 frac[2,12] <= 1
 frac[2,24] <= 1
 frac[2,7] <= 1
 frac[2,8] <= 1
 frac[2,19] <= 1
 frac[2,4] <= 1
 frac[3,7] <= 1
 frac[3,2] <= 1
 frac[3,23] <= 1
 frac[3,1] <= 1
 frac[3,18] <= 1
 frac[3,19] <= 1
 frac[4,12] <= 1
 frac[4,1] <= 1
 frac[4,8] <= 1
 frac[4,24] <= 1
 frac[4,7] <= 1
 frac[4,18] <= 1
 frac[4,4] <= 1
 frac[5,8] <= 1
 frac[5,1] <= 1
 frac[5,4] <= 1
 frac[5,7] <= 1
 frac[5,23] <= 1
 frac[5,12] <= 1
 frac[5,22] <= 1
 frac[6,7] <= 1
 frac[6,4] <= 1
 frac[6,1] <= 1
 frac[6,2] <= 1
 frac[6,18] <= 1
 frac[6,24] <= 1
 frac[6,12] <= 1
 frac[6,19] <= 1
 frac[7,22] <= 1
 frac[7,1] <= 1
 frac[7,18] <= 1
 frac[7,4] <= 1
 frac[7,7] <= 1
 frac[8,9] <= 1
 frac[8,6] <= 1
 frac[8,3] <= 1
 frac[8,16] <= 1
 frac[8,17] <= 1
 frac[9,0] <= 1
 frac[9,21] <= 1
 frac[9,16] <= 1
 frac[9,9] <= 1
 frac[9,25] <= 1
 frac[9,20] <= 1
 frac[9,6] <= 1
 frac[9,10] <= 1
 frac[10,9] <= 1
 frac[10,25] <= 1
 frac[10,15] <= 1
 frac[10,0] <= 1
 frac[10,11] <= 1
 frac[10,6] <= 1
 frac[10,21] <= 1
 frac[11,0] <= 1
 frac[11,21] <= 1
 frac[11,16] <= 1
 frac[11,10] <= 1
 frac[11,14] <= 1
 frac[11,17] <= 1
 frac[11,6] <= 1
 frac[12,17] <= 1
 frac[12,11] <= 1
 frac[12,9] <= 1
 frac[12,5] <= 1
 frac[12,16] <= 1
 frac[12,10] <= 1
 frac[12,21] <= 1
 frac[12,15] <= 1
 frac[13,17] <= 1
 frac[13,10] <= 1
 frac[13,0] <= 1
 frac[13,9] <= 1
 frac[13,5] <= 1
 frac[13,21] <= 1
 frac[14,17] <= 1
 frac[14,21] <= 1
 frac[14,5] <= 1
 frac[14,6] <= 1
 frac[14,25] <= 1
 frac[14,3] <= 1
 frac[14,20] <= 1
 frac[15,14] <= 1
 frac[15,16] <= 1
 frac[15,9] <= 1
 frac[15,21] <= 1
 frac[15,6] <= 1
 mem_tot_CPU258 <= 32768
 normalized_mem_tot_CPU258 <= 1
 normalized_rows_tot_CPU258 <= 1
 mem_tot_CPU600 <= 32768
 normalized_mem_tot_CPU600 <= 1
 normalized_rows_tot_CPU600 <= 1
 mem_tot_CPU19 <= 32768
 normalized_mem_tot_CPU19 <= 1
 normalized_rows_tot_CPU19 <= 1
 mem_tot_CPU743 <= 32768
 normalized_mem_tot_CPU743 <= 1
 normalized_rows_tot_CPU743 <= 1
 mem_tot_CPU207 <= 32768
 normalized_mem_tot_CPU207 <= 1
 normalized_rows_tot_CPU207 <= 1
 mem_tot_CPU427 <= 32768
 normalized_mem_tot_CPU427 <= 1
 normalized_rows_tot_CPU427 <= 1
 mem_tot_CPU80 <= 32768
 normalized_mem_tot_CPU80 <= 1
 normalized_rows_tot_CPU80 <= 1
 mem_tot_CPU164 <= 32768
 normalized_mem_tot_CPU164 <= 1
 normalized_rows_tot_CPU164 <= 1
 mem_tot_CPU468 <= 32768
 normalized_mem_tot_CPU468 <= 1
 normalized_rows_tot_CPU468 <= 1
 mem_tot_CPU241 <= 32768
 normalized_mem_tot_CPU241 <= 1
 normalized_rows_tot_CPU241 <= 1
 mem_tot_CPU734 <= 32768
 normalized_mem_tot_CPU734 <= 1
 normalized_rows_tot_CPU734 <= 1
 mem_tot_CPU151 <= 32768
 normalized_mem_tot_CPU151 <= 1
 normalized_rows_tot_CPU151 <= 1
 mem_tot_CPU423 <= 32768
 normalized_mem_tot_CPU423 <= 1
 normalized_rows_tot_CPU423 <= 1
 mem_tot_CPU452 <= 32768
 normalized_mem_tot_CPU452 <= 1
 normalized_rows_tot_CPU452 <= 1
 mem_tot_CPU635 <= 32768
 normalized_mem_tot_CPU635 <= 1
 normalized_rows_tot_CPU635 <= 1
 mem_tot_CPU198 <= 32768
 normalized_mem_tot_CPU198 <= 1
 normalized_rows_tot_CPU198 <= 1
 cores_sketch_CPU258 <= 7
 cores_sketch_CPU600 <= 7
 cores_sketch_CPU19 <= 7
 cores_sketch_CPU743 <= 7
 cores_sketch_CPU207 <= 7
 cores_sketch_CPU427 <= 7
 cores_sketch_CPU80 <= 7
 cores_sketch_CPU164 <= 7
 cores_sketch_CPU468 <= 7
 cores_sketch_CPU241 <= 7
 cores_sketch_CPU734 <= 7
 cores_sketch_CPU151 <= 7
 cores_sketch_CPU423 <= 7
 cores_sketch_CPU452 <= 7
 cores_sketch_CPU635 <= 7
 cores_sketch_CPU198 <= 7
Generals
 cores_sketch_CPU258 cores_sketch_CPU600 cores_sketch_CPU19
 cores_sketch_CPU743 cores_sketch_CPU207 cores_sketch_CPU427
 cores_sketch_CPU80 cores_sketch_CPU164 cores_sketch_CPU468
 cores_sketch_CPU241 cores_sketch_CPU734 cores_sketch_CPU151
 cores_sketch_CPU423 cores_sketch_CPU452 cores_sketch_CPU635
 cores_sketch_CPU198
General Constraints
 mem_overall_others: max_mem_others = MAX ( normalized_mem_tot_CPU258 ,
   normalized_mem_tot_CPU600 , normalized_mem_tot_CPU19 ,
   normalized_mem_tot_CPU743 , normalized_mem_tot_CPU207 ,
   normalized_mem_tot_CPU427 , normalized_mem_tot_CPU80 ,
   normalized_mem_tot_CPU164 , normalized_mem_tot_CPU468 ,
   normalized_mem_tot_CPU241 , normalized_mem_tot_CPU734 ,
   normalized_mem_tot_CPU151 , normalized_mem_tot_CPU423 ,
   normalized_mem_tot_CPU452 , normalized_mem_tot_CPU635 ,
   normalized_mem_tot_CPU198 )
 rows_overall_others: max_rows_others = MAX ( normalized_rows_tot_CPU258 ,
   normalized_rows_tot_CPU600 , normalized_rows_tot_CPU19 ,
   normalized_rows_tot_CPU743 , normalized_rows_tot_CPU207 ,
   normalized_rows_tot_CPU427 , normalized_rows_tot_CPU80 ,
   normalized_rows_tot_CPU164 , normalized_rows_tot_CPU468 ,
   normalized_rows_tot_CPU241 , normalized_rows_tot_CPU734 ,
   normalized_rows_tot_CPU151 , normalized_rows_tot_CPU423 ,
   normalized_rows_tot_CPU452 , normalized_rows_tot_CPU635 ,
   normalized_rows_tot_CPU198 )
 mem_access_time_CPU258: m_access_time_CPU258 = PWL ( mem_tot_CPU258 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU600: m_access_time_CPU600 = PWL ( mem_tot_CPU600 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU19: m_access_time_CPU19 = PWL ( mem_tot_CPU19 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU743: m_access_time_CPU743 = PWL ( mem_tot_CPU743 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU207: m_access_time_CPU207 = PWL ( mem_tot_CPU207 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU427: m_access_time_CPU427 = PWL ( mem_tot_CPU427 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU80: m_access_time_CPU80 = PWL ( mem_tot_CPU80 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU164: m_access_time_CPU164 = PWL ( mem_tot_CPU164 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU468: m_access_time_CPU468 = PWL ( mem_tot_CPU468 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU241: m_access_time_CPU241 = PWL ( mem_tot_CPU241 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU734: m_access_time_CPU734 = PWL ( mem_tot_CPU734 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU151: m_access_time_CPU151 = PWL ( mem_tot_CPU151 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU423: m_access_time_CPU423 = PWL ( mem_tot_CPU423 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU452: m_access_time_CPU452 = PWL ( mem_tot_CPU452 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU635: m_access_time_CPU635 = PWL ( mem_tot_CPU635 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU198: m_access_time_CPU198 = PWL ( mem_tot_CPU198 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

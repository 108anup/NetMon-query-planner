\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU_Cluster: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU1 + rows_tot_CPU2 + rows_tot_CPU3 + rows_tot_CPU5
   + rows_tot_CPU7 + rows_tot_CPU8 + rows_tot_CPU9 + rows_tot_CPU10
   + rows_tot_CPU11 + rows_tot_CPU12 + rows_tot_CPU14 + rows_tot_CPU15
   + rows_tot_CPU16 + rows_tot_CPU17 + rows_tot_CPU18 + rows_tot_CPU19
   + rows_tot_CPU22 + rows_tot_CPU23 + rows_tot_CPU24 + rows_tot_CPU25
   + rows_tot_CPU26 + rows_tot_CPU27 + rows_tot_CPU28 + rows_tot_CPU29
   + rows_tot_CPU30 + rows_tot_CPU31 + rows_tot_CPU32
  CPU_Cluster_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU_Cluster
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_rows_P4: Priority=80 Weight=1 AbsTol=0 RelTol=0
   rows_tot_tor_P41 + rows_tot_tor_P42 + rows_tot_tor_P43 + rows_tot_tor_P44
   + rows_tot_l1_P41 + rows_tot_l1_P42 + rows_tot_l2_P4
  P4_rows_load: Priority=70 Weight=1 AbsTol=0 RelTol=0
   max_rows_P4
  tot_mem_CPU_Cluster: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU1 + mem_tot_CPU2 + mem_tot_CPU3 + mem_tot_CPU5 + mem_tot_CPU7
   + mem_tot_CPU8 + mem_tot_CPU9 + mem_tot_CPU10 + mem_tot_CPU11
   + mem_tot_CPU12 + mem_tot_CPU14 + mem_tot_CPU15 + mem_tot_CPU16
   + mem_tot_CPU17 + mem_tot_CPU18 + mem_tot_CPU19 + mem_tot_CPU22
   + mem_tot_CPU23 + mem_tot_CPU24 + mem_tot_CPU25 + mem_tot_CPU26
   + mem_tot_CPU27 + mem_tot_CPU28 + mem_tot_CPU29 + mem_tot_CPU30
   + mem_tot_CPU31 + mem_tot_CPU32
  CPU_Cluster_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
  tot_mem_P4: Priority=40 Weight=1 AbsTol=0 RelTol=0
   mem_tot_tor_P41 + mem_tot_tor_P42 + mem_tot_tor_P43 + mem_tot_tor_P44
   + mem_tot_l1_P41 + mem_tot_l1_P42 + mem_tot_l2_P4
  P4_load_mem: Priority=30 Weight=1 AbsTol=0 RelTol=0
   max_mem_P4
Subject To
 cov_0_5: frac[28,5] + frac[31,5] + frac[4,5] + frac[27,5] + frac[10,5]
   >= 0.8
 cov_1_0: frac[4,0] + frac[27,0] + frac[5,0] >= 0.9
 cov_2_0: frac[30,0] + frac[13,0] + frac[22,0] + frac[32,0] + frac[29,0]
   >= 0.7
 cov_3_2: frac[27,2] + frac[4,2] + frac[5,2] >= 0.7
 cov_4_1: frac[15,1] + frac[29,1] + frac[28,1] + frac[31,1] + frac[6,1]
   + frac[33,1] + frac[32,1] >= 0.9
 cov_5_0: frac[4,0] + frac[15,0] + frac[27,0] + frac[32,0] + frac[29,0]
   + frac[31,0] + frac[33,0] >= 0.8
 cov_6_5: frac[28,5] + frac[6,5] + frac[10,5] >= 1
 cov_7_0: frac[30,0] + frac[13,0] + frac[22,0] + frac[32,0] + frac[29,0]
   >= 1
 cov_8_10: frac[25,10] + frac[16,10] + frac[29,10] + frac[32,10]
   + frac[30,10] >= 0.7
 cov_9_6: frac[31,6] + frac[7,6] + frac[27,6] + frac[28,6] + frac[0,6]
   >= 1
 cov_10_10: frac[25,10] + frac[16,10] + frac[29,10] + frac[32,10]
   + frac[30,10] >= 1
 cov_11_13: frac[8,13] + frac[31,13] + frac[28,13] + frac[27,13]
   + frac[0,13] >= 0.8
 cov_12_6: frac[31,6] + frac[27,6] + frac[8,6] + frac[28,6] + frac[0,6]
   >= 0.9
 cov_13_10: frac[25,10] + frac[20,10] + frac[30,10] >= 0.8
 cov_14_9: frac[20,9] + frac[33,9] + frac[8,9] + frac[32,9] + frac[31,9]
   + frac[30,9] + frac[28,9] >= 0.9
 cov_15_6: frac[16,6] + frac[32,6] + frac[30,6] + frac[29,6] + frac[20,6]
   >= 0.9
 cov_16_11: frac[29,11] + frac[2,11] + frac[27,11] + frac[17,11]
   + frac[31,11] + frac[33,11] + frac[32,11] >= 0.7
 cov_17_3: frac[9,3] + frac[2,3] + frac[27,3] + frac[28,3] + frac[31,3]
   >= 0.7
 cov_18_11: frac[29,11] + frac[12,11] + frac[17,11] + frac[31,11]
   + frac[28,11] + frac[33,11] + frac[32,11] >= 0.7
 cov_19_4: frac[18,4] + frac[33,4] + frac[29,4] + frac[31,4] + frac[12,4]
   + frac[32,4] + frac[28,4] >= 0.7
 cov_20_7: frac[17,7] + frac[30,7] + frac[26,7] + frac[29,7] + frac[32,7]
   >= 0.8
 cov_21_3: frac[23,3] + frac[26,3] + frac[30,3] >= 0.8
 cov_22_7: frac[33,7] + frac[9,7] + frac[29,7] + frac[32,7] + frac[28,7]
   + frac[31,7] + frac[18,7] >= 1
 cov_23_4: frac[33,4] + frac[30,4] + frac[31,4] + frac[12,4] + frac[26,4]
   + frac[32,4] + frac[28,4] >= 0.7
 cov_24_12: frac[32,12] + frac[14,12] + frac[19,12] + frac[30,12]
   + frac[29,12] >= 0.7
 cov_25_8: frac[31,8] + frac[29,8] + frac[14,8] + frac[33,8] + frac[3,8]
   + frac[27,8] + frac[32,8] >= 0.7
 cov_26_12: frac[21,12] + frac[32,12] + frac[14,12] + frac[30,12]
   + frac[29,12] >= 1
 cov_27_14: frac[24,14] + frac[30,14] + frac[33,14] + frac[31,14]
   + frac[11,14] + frac[32,14] + frac[28,14] >= 0.7
 cov_28_8: frac[31,8] + frac[24,8] + frac[30,8] + frac[33,8] + frac[1,8]
   + frac[27,8] + frac[32,8] >= 0.8
 cov_29_8: frac[31,8] + frac[29,8] + frac[14,8] + frac[33,8] + frac[3,8]
   + frac[27,8] + frac[32,8] >= 0.9
 cov_30_12: frac[21,12] + frac[32,12] + frac[14,12] + frac[30,12]
   + frac[29,12] >= 1
 cov_31_12: frac[32,12] + frac[28,12] + frac[24,12] + frac[30,12]
   + frac[33,12] + frac[31,12] + frac[11,12] >= 0.7
 accuracy_16_6: - 5.09765625 frac[16,6] + mem[16,6] = 0
 accuracy_31_6: - 5.09765625 frac[31,6] + mem[31,6] = 0
 accuracy_17_7: - 5.09765625 frac[17,7] + mem[17,7] = 0
 accuracy_18_4: - 5.09765625 frac[18,4] + mem[18,4] = 0
 accuracy_15_1: - 5.09765625 frac[15,1] + mem[15,1] = 0
 accuracy_27_2: - 5.09765625 frac[27,2] + mem[27,2] = 0
 accuracy_4_0: - 5.09765625 frac[4,0] + mem[4,0] = 0
 accuracy_28_5: - 5.09765625 frac[28,5] + mem[28,5] = 0
 accuracy_31_8: - 5.09765625 frac[31,8] + mem[31,8] = 0
 accuracy_30_0: - 5.09765625 frac[30,0] + mem[30,0] = 0
 accuracy_29_8: - 5.09765625 frac[29,8] + mem[29,8] = 0
 accuracy_7_6: - 5.09765625 frac[7,6] + mem[7,6] = 0
 accuracy_31_5: - 5.09765625 frac[31,5] + mem[31,5] = 0
 accuracy_30_7: - 5.09765625 frac[30,7] + mem[30,7] = 0
 accuracy_24_14: - 5.09765625 frac[24,14] + mem[24,14] = 0
 accuracy_20_9: - 5.09765625 frac[20,9] + mem[20,9] = 0
 accuracy_21_12: - 5.09765625 frac[21,12] + mem[21,12] = 0
 accuracy_30_14: - 5.09765625 frac[30,14] + mem[30,14] = 0
 accuracy_4_5: - 5.09765625 frac[4,5] + mem[4,5] = 0
 accuracy_9_3: - 5.09765625 frac[9,3] + mem[9,3] = 0
 accuracy_33_14: - 5.09765625 frac[33,14] + mem[33,14] = 0
 accuracy_29_11: - 5.09765625 frac[29,11] + mem[29,11] = 0
 accuracy_12_11: - 5.09765625 frac[12,11] + mem[12,11] = 0
 accuracy_25_10: - 5.09765625 frac[25,10] + mem[25,10] = 0
 accuracy_2_11: - 5.09765625 frac[2,11] + mem[2,11] = 0
 accuracy_33_4: - 5.09765625 frac[33,4] + mem[33,4] = 0
 accuracy_32_12: - 5.09765625 frac[32,12] + mem[32,12] = 0
 accuracy_29_1: - 5.09765625 frac[29,1] + mem[29,1] = 0
 accuracy_33_9: - 5.09765625 frac[33,9] + mem[33,9] = 0
 accuracy_14_8: - 5.09765625 frac[14,8] + mem[14,8] = 0
 accuracy_13_0: - 5.09765625 frac[13,0] + mem[13,0] = 0
 accuracy_24_8: - 5.09765625 frac[24,8] + mem[24,8] = 0
 accuracy_27_6: - 5.09765625 frac[27,6] + mem[27,6] = 0
 accuracy_8_9: - 5.09765625 frac[8,9] + mem[8,9] = 0
 accuracy_32_6: - 5.09765625 frac[32,6] + mem[32,6] = 0
 accuracy_23_3: - 5.09765625 frac[23,3] + mem[23,3] = 0
 accuracy_28_1: - 5.09765625 frac[28,1] + mem[28,1] = 0
 accuracy_27_11: - 5.09765625 frac[27,11] + mem[27,11] = 0
 accuracy_33_7: - 5.09765625 frac[33,7] + mem[33,7] = 0
 accuracy_32_9: - 5.09765625 frac[32,9] + mem[32,9] = 0
 accuracy_30_4: - 5.09765625 frac[30,4] + mem[30,4] = 0
 accuracy_29_4: - 5.09765625 frac[29,4] + mem[29,4] = 0
 accuracy_28_12: - 5.09765625 frac[28,12] + mem[28,12] = 0
 accuracy_31_1: - 5.09765625 frac[31,1] + mem[31,1] = 0
 accuracy_16_10: - 5.09765625 frac[16,10] + mem[16,10] = 0
 accuracy_17_11: - 5.09765625 frac[17,11] + mem[17,11] = 0
 accuracy_15_0: - 5.09765625 frac[15,0] + mem[15,0] = 0
 accuracy_27_5: - 5.09765625 frac[27,5] + mem[27,5] = 0
 accuracy_26_7: - 5.09765625 frac[26,7] + mem[26,7] = 0
 accuracy_8_6: - 5.09765625 frac[8,6] + mem[8,6] = 0
 accuracy_28_6: - 5.09765625 frac[28,6] + mem[28,6] = 0
 accuracy_9_7: - 5.09765625 frac[9,7] + mem[9,7] = 0
 accuracy_31_11: - 5.09765625 frac[31,11] + mem[31,11] = 0
 accuracy_29_7: - 5.09765625 frac[29,7] + mem[29,7] = 0
 accuracy_31_4: - 5.09765625 frac[31,4] + mem[31,4] = 0
 accuracy_20_10: - 5.09765625 frac[20,10] + mem[20,10] = 0
 accuracy_22_0: - 5.09765625 frac[22,0] + mem[22,0] = 0
 accuracy_27_0: - 5.09765625 frac[27,0] + mem[27,0] = 0
 accuracy_32_0: - 5.09765625 frac[32,0] + mem[32,0] = 0
 accuracy_28_11: - 5.09765625 frac[28,11] + mem[28,11] = 0
 accuracy_6_1: - 5.09765625 frac[6,1] + mem[6,1] = 0
 accuracy_31_14: - 5.09765625 frac[31,14] + mem[31,14] = 0
 accuracy_29_10: - 5.09765625 frac[29,10] + mem[29,10] = 0
 accuracy_14_12: - 5.09765625 frac[14,12] + mem[14,12] = 0
 accuracy_12_4: - 5.09765625 frac[12,4] + mem[12,4] = 0
 accuracy_19_12: - 5.09765625 frac[19,12] + mem[19,12] = 0
 accuracy_24_12: - 5.09765625 frac[24,12] + mem[24,12] = 0
 accuracy_8_13: - 5.09765625 frac[8,13] + mem[8,13] = 0
 accuracy_30_8: - 5.09765625 frac[30,8] + mem[30,8] = 0
 accuracy_29_0: - 5.09765625 frac[29,0] + mem[29,0] = 0
 accuracy_33_8: - 5.09765625 frac[33,8] + mem[33,8] = 0
 accuracy_31_13: - 5.09765625 frac[31,13] + mem[31,13] = 0
 accuracy_11_14: - 5.09765625 frac[11,14] + mem[11,14] = 0
 accuracy_26_3: - 5.09765625 frac[26,3] + mem[26,3] = 0
 accuracy_32_7: - 5.09765625 frac[32,7] + mem[32,7] = 0
 accuracy_32_10: - 5.09765625 frac[32,10] + mem[32,10] = 0
 accuracy_5_0: - 5.09765625 frac[5,0] + mem[5,0] = 0
 accuracy_28_13: - 5.09765625 frac[28,13] + mem[28,13] = 0
 accuracy_33_11: - 5.09765625 frac[33,11] + mem[33,11] = 0
 accuracy_31_0: - 5.09765625 frac[31,0] + mem[31,0] = 0
 accuracy_2_3: - 5.09765625 frac[2,3] + mem[2,3] = 0
 accuracy_26_4: - 5.09765625 frac[26,4] + mem[26,4] = 0
 accuracy_32_4: - 5.09765625 frac[32,4] + mem[32,4] = 0
 accuracy_4_2: - 5.09765625 frac[4,2] + mem[4,2] = 0
 accuracy_28_7: - 5.09765625 frac[28,7] + mem[28,7] = 0
 accuracy_33_1: - 5.09765625 frac[33,1] + mem[33,1] = 0
 accuracy_6_5: - 5.09765625 frac[6,5] + mem[6,5] = 0
 accuracy_30_6: - 5.09765625 frac[30,6] + mem[30,6] = 0
 accuracy_29_6: - 5.09765625 frac[29,6] + mem[29,6] = 0
 accuracy_10_5: - 5.09765625 frac[10,5] + mem[10,5] = 0
 accuracy_31_7: - 5.09765625 frac[31,7] + mem[31,7] = 0
 accuracy_18_7: - 5.09765625 frac[18,7] + mem[18,7] = 0
 accuracy_0_6: - 5.09765625 frac[0,6] + mem[0,6] = 0
 accuracy_20_6: - 5.09765625 frac[20,6] + mem[20,6] = 0
 accuracy_27_3: - 5.09765625 frac[27,3] + mem[27,3] = 0
 accuracy_32_1: - 5.09765625 frac[32,1] + mem[32,1] = 0
 accuracy_30_12: - 5.09765625 frac[30,12] + mem[30,12] = 0
 accuracy_28_4: - 5.09765625 frac[28,4] + mem[28,4] = 0
 accuracy_33_12: - 5.09765625 frac[33,12] + mem[33,12] = 0
 accuracy_31_9: - 5.09765625 frac[31,9] + mem[31,9] = 0
 accuracy_30_3: - 5.09765625 frac[30,3] + mem[30,3] = 0
 accuracy_32_14: - 5.09765625 frac[32,14] + mem[32,14] = 0
 accuracy_30_9: - 5.09765625 frac[30,9] + mem[30,9] = 0
 accuracy_28_9: - 5.09765625 frac[28,9] + mem[28,9] = 0
 accuracy_31_12: - 5.09765625 frac[31,12] + mem[31,12] = 0
 accuracy_29_12: - 5.09765625 frac[29,12] + mem[29,12] = 0
 accuracy_28_3: - 5.09765625 frac[28,3] + mem[28,3] = 0
 accuracy_27_13: - 5.09765625 frac[27,13] + mem[27,13] = 0
 accuracy_32_11: - 5.09765625 frac[32,11] + mem[32,11] = 0
 accuracy_30_10: - 5.09765625 frac[30,10] + mem[30,10] = 0
 accuracy_28_14: - 5.09765625 frac[28,14] + mem[28,14] = 0
 accuracy_31_3: - 5.09765625 frac[31,3] + mem[31,3] = 0
 accuracy_11_12: - 5.09765625 frac[11,12] + mem[11,12] = 0
 accuracy_0_13: - 5.09765625 frac[0,13] + mem[0,13] = 0
 accuracy_3_8: - 5.09765625 frac[3,8] + mem[3,8] = 0
 accuracy_1_8: - 5.09765625 frac[1,8] + mem[1,8] = 0
 accuracy_27_8: - 5.09765625 frac[27,8] + mem[27,8] = 0
 accuracy_33_0: - 5.09765625 frac[33,0] + mem[33,0] = 0
 accuracy_32_8: - 5.09765625 frac[32,8] + mem[32,8] = 0
 accuracy_5_2: - 5.09765625 frac[5,2] + mem[5,2] = 0
 mem_tot_CPU1: - mem[0,6] - mem[0,13] + mem_tot_CPU1 = 0
 R153: - 3.05176e-05 mem_tot_CPU1 + normalized_mem_tot_CPU1 = 0
 rows_tot_CPU1: - 3 frac[0,6] - 3 frac[0,13] + rows_tot_CPU1 = 0
 R155: - 0.0833333333333333 rows_tot_CPU1 + normalized_rows_tot_CPU1 = 0
 mem_tot_CPU2: - mem[1,8] + mem_tot_CPU2 = 0
 R157: - 3.05176e-05 mem_tot_CPU2 + normalized_mem_tot_CPU2 = 0
 rows_tot_CPU2: - 3 frac[1,8] + rows_tot_CPU2 = 0
 R159: - 0.0833333333333333 rows_tot_CPU2 + normalized_rows_tot_CPU2 = 0
 mem_tot_CPU3: - mem[2,11] - mem[2,3] + mem_tot_CPU3 = 0
 R161: - 3.05176e-05 mem_tot_CPU3 + normalized_mem_tot_CPU3 = 0
 rows_tot_CPU3: - 3 frac[2,11] - 3 frac[2,3] + rows_tot_CPU3 = 0
 R163: - 0.0833333333333333 rows_tot_CPU3 + normalized_rows_tot_CPU3 = 0
 mem_tot_CPU5: - mem[3,8] + mem_tot_CPU5 = 0
 R165: - 3.05176e-05 mem_tot_CPU5 + normalized_mem_tot_CPU5 = 0
 rows_tot_CPU5: - 3 frac[3,8] + rows_tot_CPU5 = 0
 R167: - 0.0833333333333333 rows_tot_CPU5 + normalized_rows_tot_CPU5 = 0
 mem_tot_CPU7: - mem[4,0] - mem[4,5] - mem[4,2] + mem_tot_CPU7 = 0
 R169: - 3.05176e-05 mem_tot_CPU7 + normalized_mem_tot_CPU7 = 0
 rows_tot_CPU7: - 3 frac[4,0] - 3 frac[4,5] - 3 frac[4,2] + rows_tot_CPU7
   = 0
 R171: - 0.0833333333333333 rows_tot_CPU7 + normalized_rows_tot_CPU7 = 0
 mem_tot_CPU8: - mem[5,0] - mem[5,2] + mem_tot_CPU8 = 0
 R173: - 3.05176e-05 mem_tot_CPU8 + normalized_mem_tot_CPU8 = 0
 rows_tot_CPU8: - 3 frac[5,0] - 3 frac[5,2] + rows_tot_CPU8 = 0
 R175: - 0.0833333333333333 rows_tot_CPU8 + normalized_rows_tot_CPU8 = 0
 mem_tot_CPU9: - mem[6,1] - mem[6,5] + mem_tot_CPU9 = 0
 R177: - 3.05176e-05 mem_tot_CPU9 + normalized_mem_tot_CPU9 = 0
 rows_tot_CPU9: - 3 frac[6,1] - 3 frac[6,5] + rows_tot_CPU9 = 0
 R179: - 0.0833333333333333 rows_tot_CPU9 + normalized_rows_tot_CPU9 = 0
 mem_tot_CPU10: - mem[7,6] + mem_tot_CPU10 = 0
 R181: - 3.05176e-05 mem_tot_CPU10 + normalized_mem_tot_CPU10 = 0
 rows_tot_CPU10: - 3 frac[7,6] + rows_tot_CPU10 = 0
 R183: - 0.0833333333333333 rows_tot_CPU10 + normalized_rows_tot_CPU10 = 0
 mem_tot_CPU11: - mem[8,9] - mem[8,6] - mem[8,13] + mem_tot_CPU11 = 0
 R185: - 3.05176e-05 mem_tot_CPU11 + normalized_mem_tot_CPU11 = 0
 rows_tot_CPU11: - 3 frac[8,9] - 3 frac[8,6] - 3 frac[8,13]
   + rows_tot_CPU11 = 0
 R187: - 0.0833333333333333 rows_tot_CPU11 + normalized_rows_tot_CPU11 = 0
 mem_tot_CPU12: - mem[9,3] - mem[9,7] + mem_tot_CPU12 = 0
 R189: - 3.05176e-05 mem_tot_CPU12 + normalized_mem_tot_CPU12 = 0
 rows_tot_CPU12: - 3 frac[9,3] - 3 frac[9,7] + rows_tot_CPU12 = 0
 R191: - 0.0833333333333333 rows_tot_CPU12 + normalized_rows_tot_CPU12 = 0
 mem_tot_CPU14: - mem[10,5] + mem_tot_CPU14 = 0
 R193: - 3.05176e-05 mem_tot_CPU14 + normalized_mem_tot_CPU14 = 0
 rows_tot_CPU14: - 3 frac[10,5] + rows_tot_CPU14 = 0
 R195: - 0.0833333333333333 rows_tot_CPU14 + normalized_rows_tot_CPU14 = 0
 mem_tot_CPU15: - mem[11,14] - mem[11,12] + mem_tot_CPU15 = 0
 R197: - 3.05176e-05 mem_tot_CPU15 + normalized_mem_tot_CPU15 = 0
 rows_tot_CPU15: - 3 frac[11,14] - 3 frac[11,12] + rows_tot_CPU15 = 0
 R199: - 0.0833333333333333 rows_tot_CPU15 + normalized_rows_tot_CPU15 = 0
 mem_tot_CPU16: - mem[12,11] - mem[12,4] + mem_tot_CPU16 = 0
 R201: - 3.05176e-05 mem_tot_CPU16 + normalized_mem_tot_CPU16 = 0
 rows_tot_CPU16: - 3 frac[12,11] - 3 frac[12,4] + rows_tot_CPU16 = 0
 R203: - 0.0833333333333333 rows_tot_CPU16 + normalized_rows_tot_CPU16 = 0
 mem_tot_CPU17: - mem[13,0] + mem_tot_CPU17 = 0
 R205: - 3.05176e-05 mem_tot_CPU17 + normalized_mem_tot_CPU17 = 0
 rows_tot_CPU17: - 3 frac[13,0] + rows_tot_CPU17 = 0
 R207: - 0.0833333333333333 rows_tot_CPU17 + normalized_rows_tot_CPU17 = 0
 mem_tot_CPU18: - mem[14,8] - mem[14,12] + mem_tot_CPU18 = 0
 R209: - 3.05176e-05 mem_tot_CPU18 + normalized_mem_tot_CPU18 = 0
 rows_tot_CPU18: - 3 frac[14,8] - 3 frac[14,12] + rows_tot_CPU18 = 0
 R211: - 0.0833333333333333 rows_tot_CPU18 + normalized_rows_tot_CPU18 = 0
 mem_tot_CPU19: - mem[15,1] - mem[15,0] + mem_tot_CPU19 = 0
 R213: - 3.05176e-05 mem_tot_CPU19 + normalized_mem_tot_CPU19 = 0
 rows_tot_CPU19: - 3 frac[15,1] - 3 frac[15,0] + rows_tot_CPU19 = 0
 R215: - 0.0833333333333333 rows_tot_CPU19 + normalized_rows_tot_CPU19 = 0
 mem_tot_CPU22: - mem[16,6] - mem[16,10] + mem_tot_CPU22 = 0
 R217: - 3.05176e-05 mem_tot_CPU22 + normalized_mem_tot_CPU22 = 0
 rows_tot_CPU22: - 3 frac[16,6] - 3 frac[16,10] + rows_tot_CPU22 = 0
 R219: - 0.0833333333333333 rows_tot_CPU22 + normalized_rows_tot_CPU22 = 0
 mem_tot_CPU23: - mem[17,7] - mem[17,11] + mem_tot_CPU23 = 0
 R221: - 3.05176e-05 mem_tot_CPU23 + normalized_mem_tot_CPU23 = 0
 rows_tot_CPU23: - 3 frac[17,7] - 3 frac[17,11] + rows_tot_CPU23 = 0
 R223: - 0.0833333333333333 rows_tot_CPU23 + normalized_rows_tot_CPU23 = 0
 mem_tot_CPU24: - mem[18,4] - mem[18,7] + mem_tot_CPU24 = 0
 R225: - 3.05176e-05 mem_tot_CPU24 + normalized_mem_tot_CPU24 = 0
 rows_tot_CPU24: - 3 frac[18,4] - 3 frac[18,7] + rows_tot_CPU24 = 0
 R227: - 0.0833333333333333 rows_tot_CPU24 + normalized_rows_tot_CPU24 = 0
 mem_tot_CPU25: - mem[19,12] + mem_tot_CPU25 = 0
 R229: - 3.05176e-05 mem_tot_CPU25 + normalized_mem_tot_CPU25 = 0
 rows_tot_CPU25: - 3 frac[19,12] + rows_tot_CPU25 = 0
 R231: - 0.0833333333333333 rows_tot_CPU25 + normalized_rows_tot_CPU25 = 0
 mem_tot_CPU26: - mem[20,9] - mem[20,10] - mem[20,6] + mem_tot_CPU26 = 0
 R233: - 3.05176e-05 mem_tot_CPU26 + normalized_mem_tot_CPU26 = 0
 rows_tot_CPU26: - 3 frac[20,9] - 3 frac[20,10] - 3 frac[20,6]
   + rows_tot_CPU26 = 0
 R235: - 0.0833333333333333 rows_tot_CPU26 + normalized_rows_tot_CPU26 = 0
 mem_tot_CPU27: - mem[21,12] + mem_tot_CPU27 = 0
 R237: - 3.05176e-05 mem_tot_CPU27 + normalized_mem_tot_CPU27 = 0
 rows_tot_CPU27: - 3 frac[21,12] + rows_tot_CPU27 = 0
 R239: - 0.0833333333333333 rows_tot_CPU27 + normalized_rows_tot_CPU27 = 0
 mem_tot_CPU28: - mem[22,0] + mem_tot_CPU28 = 0
 R241: - 3.05176e-05 mem_tot_CPU28 + normalized_mem_tot_CPU28 = 0
 rows_tot_CPU28: - 3 frac[22,0] + rows_tot_CPU28 = 0
 R243: - 0.0833333333333333 rows_tot_CPU28 + normalized_rows_tot_CPU28 = 0
 mem_tot_CPU29: - mem[23,3] + mem_tot_CPU29 = 0
 R245: - 3.05176e-05 mem_tot_CPU29 + normalized_mem_tot_CPU29 = 0
 rows_tot_CPU29: - 3 frac[23,3] + rows_tot_CPU29 = 0
 R247: - 0.0833333333333333 rows_tot_CPU29 + normalized_rows_tot_CPU29 = 0
 mem_tot_CPU30: - mem[24,14] - mem[24,8] - mem[24,12] + mem_tot_CPU30 = 0
 R249: - 3.05176e-05 mem_tot_CPU30 + normalized_mem_tot_CPU30 = 0
 rows_tot_CPU30: - 3 frac[24,14] - 3 frac[24,8] - 3 frac[24,12]
   + rows_tot_CPU30 = 0
 R251: - 0.0833333333333333 rows_tot_CPU30 + normalized_rows_tot_CPU30 = 0
 mem_tot_CPU31: - mem[25,10] + mem_tot_CPU31 = 0
 R253: - 3.05176e-05 mem_tot_CPU31 + normalized_mem_tot_CPU31 = 0
 rows_tot_CPU31: - 3 frac[25,10] + rows_tot_CPU31 = 0
 R255: - 0.0833333333333333 rows_tot_CPU31 + normalized_rows_tot_CPU31 = 0
 mem_tot_CPU32: - mem[26,7] - mem[26,3] - mem[26,4] + mem_tot_CPU32 = 0
 R257: - 3.05176e-05 mem_tot_CPU32 + normalized_mem_tot_CPU32 = 0
 rows_tot_CPU32: - 3 frac[26,7] - 3 frac[26,3] - 3 frac[26,4]
   + rows_tot_CPU32 = 0
 R259: - 0.0833333333333333 rows_tot_CPU32 + normalized_rows_tot_CPU32 = 0
 mem_tot_tor_P41: - mem[27,2] - mem[27,6] - mem[27,11] - mem[27,5]
   - mem[27,0] - mem[27,3] - mem[27,13] - mem[27,8] + mem_tot_tor_P41 = 0
 R261: - 0.00173611111111111 mem_tot_tor_P41 + normalized_mem_tot_tor_P41
   = 0
 rows_tot_tor_P41: - 3 frac[27,2] - 3 frac[27,6] - 3 frac[27,11]
   - 3 frac[27,5] - 3 frac[27,0] - 3 frac[27,3] - 3 frac[27,13]
   - 3 frac[27,8] + rows_tot_tor_P41 = 0
 R263: - 0.0208333333333333 rows_tot_tor_P41 + normalized_rows_tot_tor_P41
   = 0
 mem_tot_tor_P42: - mem[28,5] - mem[28,1] - mem[28,12] - mem[28,6]
   - mem[28,11] - mem[28,13] - mem[28,7] - mem[28,4] - mem[28,9]
   - mem[28,3] - mem[28,14] + mem_tot_tor_P42 = 0
 R265: - 0.00173611111111111 mem_tot_tor_P42 + normalized_mem_tot_tor_P42
   = 0
 rows_tot_tor_P42: - 3 frac[28,5] - 3 frac[28,1] - 3 frac[28,12]
   - 3 frac[28,6] - 3 frac[28,11] - 3 frac[28,13] - 3 frac[28,7]
   - 3 frac[28,4] - 3 frac[28,9] - 3 frac[28,3] - 3 frac[28,14]
   + rows_tot_tor_P42 = 0
 R267: - 0.0208333333333333 rows_tot_tor_P42 + normalized_rows_tot_tor_P42
   = 0
 mem_tot_tor_P43: - mem[29,8] - mem[29,11] - mem[29,1] - mem[29,4]
   - mem[29,7] - mem[29,10] - mem[29,0] - mem[29,6] - mem[29,12]
   + mem_tot_tor_P43 = 0
 R269: - 0.00173611111111111 mem_tot_tor_P43 + normalized_mem_tot_tor_P43
   = 0
 rows_tot_tor_P43: - 3 frac[29,8] - 3 frac[29,11] - 3 frac[29,1]
   - 3 frac[29,4] - 3 frac[29,7] - 3 frac[29,10] - 3 frac[29,0]
   - 3 frac[29,6] - 3 frac[29,12] + rows_tot_tor_P43 = 0
 R271: - 0.0208333333333333 rows_tot_tor_P43 + normalized_rows_tot_tor_P43
   = 0
 mem_tot_tor_P44: - mem[30,0] - mem[30,7] - mem[30,14] - mem[30,4]
   - mem[30,8] - mem[30,6] - mem[30,12] - mem[30,3] - mem[30,9]
   - mem[30,10] + mem_tot_tor_P44 = 0
 R273: - 0.00173611111111111 mem_tot_tor_P44 + normalized_mem_tot_tor_P44
   = 0
 rows_tot_tor_P44: - 3 frac[30,0] - 3 frac[30,7] - 3 frac[30,14]
   - 3 frac[30,4] - 3 frac[30,8] - 3 frac[30,6] - 3 frac[30,12]
   - 3 frac[30,3] - 3 frac[30,9] - 3 frac[30,10] + rows_tot_tor_P44 = 0
 R275: - 0.0208333333333333 rows_tot_tor_P44 + normalized_rows_tot_tor_P44
   = 0
 mem_tot_l1_P41: - mem[31,6] - mem[31,8] - mem[31,5] - mem[31,1]
   - mem[31,11] - mem[31,4] - mem[31,14] - mem[31,13] - mem[31,0]
   - mem[31,7] - mem[31,9] - mem[31,12] - mem[31,3] + mem_tot_l1_P41 = 0
 R277: - 0.00173611111111111 mem_tot_l1_P41 + normalized_mem_tot_l1_P41
   = 0
 rows_tot_l1_P41: - 3 frac[31,6] - 3 frac[31,8] - 3 frac[31,5]
   - 3 frac[31,1] - 3 frac[31,11] - 3 frac[31,4] - 3 frac[31,14]
   - 3 frac[31,13] - 3 frac[31,0] - 3 frac[31,7] - 3 frac[31,9]
   - 3 frac[31,12] - 3 frac[31,3] + rows_tot_l1_P41 = 0
 R279: - 0.0208333333333333 rows_tot_l1_P41 + normalized_rows_tot_l1_P41
   = 0
 mem_tot_l1_P42: - mem[32,12] - mem[32,6] - mem[32,9] - mem[32,0]
   - mem[32,7] - mem[32,10] - mem[32,4] - mem[32,1] - mem[32,14]
   - mem[32,11] - mem[32,8] + mem_tot_l1_P42 = 0
 R281: - 0.00173611111111111 mem_tot_l1_P42 + normalized_mem_tot_l1_P42
   = 0
 rows_tot_l1_P42: - 3 frac[32,12] - 3 frac[32,6] - 3 frac[32,9]
   - 3 frac[32,0] - 3 frac[32,7] - 3 frac[32,10] - 3 frac[32,4]
   - 3 frac[32,1] - 3 frac[32,14] - 3 frac[32,11] - 3 frac[32,8]
   + rows_tot_l1_P42 = 0
 R283: - 0.0208333333333333 rows_tot_l1_P42 + normalized_rows_tot_l1_P42
   = 0
 mem_tot_l2_P4: - mem[33,14] - mem[33,4] - mem[33,9] - mem[33,7]
   - mem[33,8] - mem[33,11] - mem[33,1] - mem[33,12] - mem[33,0]
   + mem_tot_l2_P4 = 0
 R285: - 0.00173611111111111 mem_tot_l2_P4 + normalized_mem_tot_l2_P4 = 0
 rows_tot_l2_P4: - 3 frac[33,14] - 3 frac[33,4] - 3 frac[33,9]
   - 3 frac[33,7] - 3 frac[33,8] - 3 frac[33,11] - 3 frac[33,1]
   - 3 frac[33,12] - 3 frac[33,0] + rows_tot_l2_P4 = 0
 R287: - 0.0208333333333333 rows_tot_l2_P4 + normalized_rows_tot_l2_P4 = 0
 row_capacity_CPU1: rows_tot_CPU1 <= 12
 row_capacity_CPU2: rows_tot_CPU2 <= 12
 row_capacity_CPU3: rows_tot_CPU3 <= 12
 row_capacity_CPU5: rows_tot_CPU5 <= 12
 row_capacity_CPU7: rows_tot_CPU7 <= 12
 row_capacity_CPU8: rows_tot_CPU8 <= 12
 row_capacity_CPU9: rows_tot_CPU9 <= 12
 row_capacity_CPU10: rows_tot_CPU10 <= 12
 row_capacity_CPU11: rows_tot_CPU11 <= 12
 row_capacity_CPU12: rows_tot_CPU12 <= 12
 row_capacity_CPU14: rows_tot_CPU14 <= 12
 row_capacity_CPU15: rows_tot_CPU15 <= 12
 row_capacity_CPU16: rows_tot_CPU16 <= 12
 row_capacity_CPU17: rows_tot_CPU17 <= 12
 row_capacity_CPU18: rows_tot_CPU18 <= 12
 row_capacity_CPU19: rows_tot_CPU19 <= 12
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
 row_capacity_tor_P42: rows_tot_tor_P42 <= 48
 row_capacity_tor_P43: rows_tot_tor_P43 <= 48
 row_capacity_tor_P44: rows_tot_tor_P44 <= 48
 row_capacity_l1_P41: rows_tot_l1_P41 <= 48
 row_capacity_l1_P42: rows_tot_l1_P42 <= 48
 row_capacity_l2_P4: rows_tot_l2_P4 <= 48
Bounds
 mem_tot_CPU1 <= 32768
 normalized_mem_tot_CPU1 <= 1
 normalized_rows_tot_CPU1 <= 1
 mem_tot_CPU2 <= 32768
 normalized_mem_tot_CPU2 <= 1
 normalized_rows_tot_CPU2 <= 1
 mem_tot_CPU3 <= 32768
 normalized_mem_tot_CPU3 <= 1
 normalized_rows_tot_CPU3 <= 1
 mem_tot_CPU5 <= 32768
 normalized_mem_tot_CPU5 <= 1
 normalized_rows_tot_CPU5 <= 1
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
 mem_tot_tor_P41 <= 576
 normalized_mem_tot_tor_P41 <= 1
 normalized_rows_tot_tor_P41 <= 1
 mem_tot_tor_P42 <= 576
 normalized_mem_tot_tor_P42 <= 1
 normalized_rows_tot_tor_P42 <= 1
 mem_tot_tor_P43 <= 576
 normalized_mem_tot_tor_P43 <= 1
 normalized_rows_tot_tor_P43 <= 1
 mem_tot_tor_P44 <= 576
 normalized_mem_tot_tor_P44 <= 1
 normalized_rows_tot_tor_P44 <= 1
 mem_tot_l1_P41 <= 576
 normalized_mem_tot_l1_P41 <= 1
 normalized_rows_tot_l1_P41 <= 1
 mem_tot_l1_P42 <= 576
 normalized_mem_tot_l1_P42 <= 1
 normalized_rows_tot_l1_P42 <= 1
 mem_tot_l2_P4 <= 576
 normalized_mem_tot_l2_P4 <= 1
 normalized_rows_tot_l2_P4 <= 1
Binaries
 frac[16,6] frac[31,6] frac[17,7] frac[18,4] frac[15,1] frac[27,2]
 frac[4,0] frac[28,5] frac[31,8] frac[30,0] frac[29,8] frac[7,6] frac[31,5]
 frac[30,7] frac[24,14] frac[20,9] frac[21,12] frac[30,14] frac[4,5]
 frac[9,3] frac[33,14] frac[29,11] frac[12,11] frac[25,10] frac[2,11]
 frac[33,4] frac[32,12] frac[29,1] frac[33,9] frac[14,8] frac[13,0]
 frac[24,8] frac[27,6] frac[8,9] frac[32,6] frac[23,3] frac[28,1]
 frac[27,11] frac[33,7] frac[32,9] frac[30,4] frac[29,4] frac[28,12]
 frac[31,1] frac[16,10] frac[17,11] frac[15,0] frac[27,5] frac[26,7]
 frac[8,6] frac[28,6] frac[9,7] frac[31,11] frac[29,7] frac[31,4]
 frac[20,10] frac[22,0] frac[27,0] frac[32,0] frac[28,11] frac[6,1]
 frac[31,14] frac[29,10] frac[14,12] frac[12,4] frac[19,12] frac[24,12]
 frac[8,13] frac[30,8] frac[29,0] frac[33,8] frac[31,13] frac[11,14]
 frac[26,3] frac[32,7] frac[32,10] frac[5,0] frac[28,13] frac[33,11]
 frac[31,0] frac[2,3] frac[26,4] frac[32,4] frac[4,2] frac[28,7] frac[33,1]
 frac[6,5] frac[30,6] frac[29,6] frac[10,5] frac[31,7] frac[18,7] frac[0,6]
 frac[20,6] frac[27,3] frac[32,1] frac[30,12] frac[28,4] frac[33,12]
 frac[31,9] frac[30,3] frac[32,14] frac[30,9] frac[28,9] frac[31,12]
 frac[29,12] frac[28,3] frac[27,13] frac[32,11] frac[30,10] frac[28,14]
 frac[31,3] frac[11,12] frac[0,13] frac[3,8] frac[1,8] frac[27,8]
 frac[33,0] frac[32,8] frac[5,2]
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_tor_P41 , mem_tot_tor_P42 ,
   mem_tot_tor_P43 , mem_tot_tor_P44 , mem_tot_l1_P41 , mem_tot_l1_P42 ,
   mem_tot_l2_P4 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU1 , normalized_mem_tot_CPU2 ,
   normalized_mem_tot_CPU3 , normalized_mem_tot_CPU5 ,
   normalized_mem_tot_CPU7 , normalized_mem_tot_CPU8 ,
   normalized_mem_tot_CPU9 , normalized_mem_tot_CPU10 ,
   normalized_mem_tot_CPU11 , normalized_mem_tot_CPU12 ,
   normalized_mem_tot_CPU14 , normalized_mem_tot_CPU15 ,
   normalized_mem_tot_CPU16 , normalized_mem_tot_CPU17 ,
   normalized_mem_tot_CPU18 , normalized_mem_tot_CPU19 ,
   normalized_mem_tot_CPU22 , normalized_mem_tot_CPU23 ,
   normalized_mem_tot_CPU24 , normalized_mem_tot_CPU25 ,
   normalized_mem_tot_CPU26 , normalized_mem_tot_CPU27 ,
   normalized_mem_tot_CPU28 , normalized_mem_tot_CPU29 ,
   normalized_mem_tot_CPU30 , normalized_mem_tot_CPU31 ,
   normalized_mem_tot_CPU32 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_tor_P41 , rows_tot_tor_P42 ,
   rows_tot_tor_P43 , rows_tot_tor_P44 , rows_tot_l1_P41 ,
   rows_tot_l1_P42 , rows_tot_l2_P4 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU1 ,
   normalized_rows_tot_CPU2 , normalized_rows_tot_CPU3 ,
   normalized_rows_tot_CPU5 , normalized_rows_tot_CPU7 ,
   normalized_rows_tot_CPU8 , normalized_rows_tot_CPU9 ,
   normalized_rows_tot_CPU10 , normalized_rows_tot_CPU11 ,
   normalized_rows_tot_CPU12 , normalized_rows_tot_CPU14 ,
   normalized_rows_tot_CPU15 , normalized_rows_tot_CPU16 ,
   normalized_rows_tot_CPU17 , normalized_rows_tot_CPU18 ,
   normalized_rows_tot_CPU19 , normalized_rows_tot_CPU22 ,
   normalized_rows_tot_CPU23 , normalized_rows_tot_CPU24 ,
   normalized_rows_tot_CPU25 , normalized_rows_tot_CPU26 ,
   normalized_rows_tot_CPU27 , normalized_rows_tot_CPU28 ,
   normalized_rows_tot_CPU29 , normalized_rows_tot_CPU30 ,
   normalized_rows_tot_CPU31 , normalized_rows_tot_CPU32 )
End

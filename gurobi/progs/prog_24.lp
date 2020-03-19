\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU_Cluster: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU1 + rows_tot_CPU2 + rows_tot_CPU3 + rows_tot_CPU4
   + rows_tot_CPU5 + rows_tot_CPU6 + rows_tot_CPU7 + rows_tot_CPU8
   + rows_tot_CPU9 + rows_tot_CPU10 + rows_tot_CPU11 + rows_tot_CPU12
   + rows_tot_CPU13 + rows_tot_CPU14 + rows_tot_CPU15 + rows_tot_CPU16
   + rows_tot_CPU17 + rows_tot_CPU18 + rows_tot_CPU19 + rows_tot_CPU20
   + rows_tot_CPU21 + rows_tot_CPU22 + rows_tot_CPU23 + rows_tot_CPU24
   + rows_tot_CPU25 + rows_tot_CPU26 + rows_tot_CPU27 + rows_tot_CPU28
   + rows_tot_CPU29 + rows_tot_CPU30 + rows_tot_CPU31 + rows_tot_CPU32
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
   mem_tot_CPU1 + mem_tot_CPU2 + mem_tot_CPU3 + mem_tot_CPU4 + mem_tot_CPU5
   + mem_tot_CPU6 + mem_tot_CPU7 + mem_tot_CPU8 + mem_tot_CPU9
   + mem_tot_CPU10 + mem_tot_CPU11 + mem_tot_CPU12 + mem_tot_CPU13
   + mem_tot_CPU14 + mem_tot_CPU15 + mem_tot_CPU16 + mem_tot_CPU17
   + mem_tot_CPU18 + mem_tot_CPU19 + mem_tot_CPU20 + mem_tot_CPU21
   + mem_tot_CPU22 + mem_tot_CPU23 + mem_tot_CPU24 + mem_tot_CPU25
   + mem_tot_CPU26 + mem_tot_CPU27 + mem_tot_CPU28 + mem_tot_CPU29
   + mem_tot_CPU30 + mem_tot_CPU31 + mem_tot_CPU32
  CPU_Cluster_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
  tot_mem_P4: Priority=40 Weight=1 AbsTol=0 RelTol=0
   mem_tot_tor_P41 + mem_tot_tor_P42 + mem_tot_tor_P43 + mem_tot_tor_P44
   + mem_tot_l1_P41 + mem_tot_l1_P42 + mem_tot_l2_P4
  P4_load_mem: Priority=30 Weight=1 AbsTol=0 RelTol=0
   max_mem_P4
Subject To
 cov_0_4: frac[15,4] + frac[33,4] + frac[5,4] + frac[36,4] + frac[32,4]
   >= 0.9
 cov_1_0: frac[32,0] + frac[3,0] + frac[7,0] >= 1
 cov_2_7: frac[34,7] + frac[37,7] + frac[23,7] + frac[36,7] + frac[5,7]
   + frac[32,7] + frac[38,7] >= 0.9
 cov_3_1: frac[31,1] + frac[22,1] + frac[35,1] + frac[34,1] + frac[37,1]
   >= 0.7
 cov_4_5: frac[11,5] + frac[38,5] + frac[18,5] + frac[34,5] + frac[37,5]
   + frac[36,5] + frac[33,5] >= 1
 cov_5_7: frac[37,7] + frac[36,7] + frac[32,7] + frac[35,7] + frac[38,7]
   + frac[31,7] + frac[7,7] >= 1
 cov_6_0: frac[37,0] + frac[18,0] + frac[32,0] + frac[34,0] + frac[5,0]
   + frac[36,0] + frac[38,0] >= 1
 cov_7_7: frac[34,7] + frac[23,7] + frac[18,7] >= 0.7
 cov_8_14: frac[33,14] + frac[10,14] + frac[35,14] + frac[36,14]
   + frac[26,14] + frac[37,14] + frac[38,14] >= 1
 cov_9_10: frac[37,10] + frac[35,10] + frac[15,10] + frac[38,10]
   + frac[36,10] + frac[30,10] + frac[33,10] >= 1
 cov_10_13: frac[37,13] + frac[36,13] + frac[35,13] + frac[32,13]
   + frac[4,13] + frac[38,13] + frac[27,13] >= 0.8
 cov_11_11: frac[26,11] + frac[19,11] + frac[37,11] + frac[34,11]
   + frac[35,11] >= 1
 cov_12_9: frac[35,9] + frac[33,9] + frac[37,9] + frac[36,9] + frac[15,9]
   + frac[38,9] + frac[30,9] >= 1
 cov_13_12: frac[37,12] + frac[19,12] + frac[38,12] + frac[15,12]
   + frac[33,12] + frac[36,12] + frac[34,12] >= 0.9
 cov_14_15: frac[35,15] + frac[38,15] + frac[37,15] + frac[27,15]
   + frac[36,15] + frac[33,15] + frac[15,15] >= 0.7
 cov_15_15: frac[35,15] + frac[38,15] + frac[37,15] + frac[36,15]
   + frac[26,15] + frac[33,15] + frac[15,15] >= 1
 cov_16_17: frac[28,17] + frac[16,17] + frac[34,17] + frac[35,17]
   + frac[37,17] >= 0.8
 cov_17_16: frac[32,16] + frac[8,16] + frac[36,16] + frac[0,16]
   + frac[33,16] >= 0.8
 cov_18_21: frac[36,21] + frac[33,21] + frac[12,21] + frac[32,21]
   + frac[2,21] >= 1
 cov_19_17: frac[36,17] + frac[38,17] + frac[28,17] + frac[33,17]
   + frac[35,17] + frac[8,17] + frac[37,17] >= 0.8
 cov_20_22: frac[35,22] + frac[36,22] + frac[28,22] + frac[14,22]
   + frac[38,22] + frac[37,22] + frac[33,22] >= 0.9
 cov_21_21: frac[36,21] + frac[33,21] + frac[12,21] + frac[32,21]
   + frac[2,21] >= 0.7
 cov_22_17: frac[14,17] + frac[12,17] + frac[33,17] >= 0.9
 cov_23_22: frac[2,22] + frac[36,22] + frac[32,22] + frac[38,22]
   + frac[37,22] + frac[34,22] + frac[20,22] >= 0.7
 cov_24_26: frac[9,26] + frac[1,26] + frac[32,26] + frac[36,26]
   + frac[33,26] >= 1
 cov_25_27: frac[38,27] + frac[36,27] + frac[33,27] + frac[34,27]
   + frac[9,27] + frac[16,27] + frac[37,27] >= 0.7
 cov_26_30: frac[34,30] + frac[36,30] + frac[16,30] + frac[7,30]
   + frac[32,30] + frac[37,30] + frac[38,30] >= 0.8
 cov_27_27: frac[21,27] + frac[22,27] + frac[34,27] >= 0.9
 cov_28_29: frac[34,29] + frac[37,29] + frac[36,29] + frac[6,29]
   + frac[16,29] + frac[32,29] + frac[38,29] >= 0.8
 cov_29_25: frac[32,25] + frac[35,25] + frac[37,25] + frac[24,25]
   + frac[36,25] + frac[1,25] + frac[38,25] >= 1
 cov_30_28: frac[37,28] + frac[38,28] + frac[13,28] + frac[33,28]
   + frac[30,28] + frac[35,28] + frac[36,28] >= 0.7
 cov_31_25: frac[33,25] + frac[35,25] + frac[37,25] + frac[24,25]
   + frac[36,25] + frac[13,25] + frac[38,25] >= 0.7
 accuracy_34_7: - 5.09765625 frac[34,7] + mem[34,7] = 0
 accuracy_9_26: - 5.09765625 frac[9,26] + mem[9,26] = 0
 accuracy_14_17: - 5.09765625 frac[14,17] + mem[14,17] = 0
 accuracy_37_10: - 5.09765625 frac[37,10] + mem[37,10] = 0
 accuracy_35_22: - 5.09765625 frac[35,22] + mem[35,22] = 0
 accuracy_37_7: - 5.09765625 frac[37,7] + mem[37,7] = 0
 accuracy_36_17: - 5.09765625 frac[36,17] + mem[36,17] = 0
 accuracy_2_22: - 5.09765625 frac[2,22] + mem[2,22] = 0
 accuracy_26_11: - 5.09765625 frac[26,11] + mem[26,11] = 0
 accuracy_38_27: - 5.09765625 frac[38,27] + mem[38,27] = 0
 accuracy_23_7: - 5.09765625 frac[23,7] + mem[23,7] = 0
 accuracy_35_15: - 5.09765625 frac[35,15] + mem[35,15] = 0
 accuracy_34_29: - 5.09765625 frac[34,29] + mem[34,29] = 0
 accuracy_11_5: - 5.09765625 frac[11,5] + mem[11,5] = 0
 accuracy_37_13: - 5.09765625 frac[37,13] + mem[37,13] = 0
 accuracy_36_27: - 5.09765625 frac[36,27] + mem[36,27] = 0
 accuracy_12_17: - 5.09765625 frac[12,17] + mem[12,17] = 0
 accuracy_36_22: - 5.09765625 frac[36,22] + mem[36,22] = 0
 accuracy_15_4: - 5.09765625 frac[15,4] + mem[15,4] = 0
 accuracy_38_17: - 5.09765625 frac[38,17] + mem[38,17] = 0
 accuracy_33_25: - 5.09765625 frac[33,25] + mem[33,25] = 0
 accuracy_35_10: - 5.09765625 frac[35,10] + mem[35,10] = 0
 accuracy_34_30: - 5.09765625 frac[34,30] + mem[34,30] = 0
 accuracy_33_14: - 5.09765625 frac[33,14] + mem[33,14] = 0
 accuracy_36_13: - 5.09765625 frac[36,13] + mem[36,13] = 0
 accuracy_37_0: - 5.09765625 frac[37,0] + mem[37,0] = 0
 accuracy_38_15: - 5.09765625 frac[38,15] + mem[38,15] = 0
 accuracy_37_29: - 5.09765625 frac[37,29] + mem[37,29] = 0
 accuracy_32_22: - 5.09765625 frac[32,22] + mem[32,22] = 0
 accuracy_28_17: - 5.09765625 frac[28,17] + mem[28,17] = 0
 accuracy_32_25: - 5.09765625 frac[32,25] + mem[32,25] = 0
 accuracy_33_4: - 5.09765625 frac[33,4] + mem[33,4] = 0
 accuracy_10_14: - 5.09765625 frac[10,14] + mem[10,14] = 0
 accuracy_35_9: - 5.09765625 frac[35,9] + mem[35,9] = 0
 accuracy_33_9: - 5.09765625 frac[33,9] + mem[33,9] = 0
 accuracy_38_5: - 5.09765625 frac[38,5] + mem[38,5] = 0
 accuracy_36_29: - 5.09765625 frac[36,29] + mem[36,29] = 0
 accuracy_18_0: - 5.09765625 frac[18,0] + mem[18,0] = 0
 accuracy_1_26: - 5.09765625 frac[1,26] + mem[1,26] = 0
 accuracy_28_22: - 5.09765625 frac[28,22] + mem[28,22] = 0
 accuracy_31_1: - 5.09765625 frac[31,1] + mem[31,1] = 0
 accuracy_14_22: - 5.09765625 frac[14,22] + mem[14,22] = 0
 accuracy_37_9: - 5.09765625 frac[37,9] + mem[37,9] = 0
 accuracy_36_7: - 5.09765625 frac[36,7] + mem[36,7] = 0
 accuracy_35_25: - 5.09765625 frac[35,25] + mem[35,25] = 0
 accuracy_6_29: - 5.09765625 frac[6,29] + mem[6,29] = 0
 accuracy_21_27: - 5.09765625 frac[21,27] + mem[21,27] = 0
 accuracy_18_5: - 5.09765625 frac[18,5] + mem[18,5] = 0
 accuracy_16_29: - 5.09765625 frac[16,29] + mem[16,29] = 0
 accuracy_32_16: - 5.09765625 frac[32,16] + mem[32,16] = 0
 accuracy_35_14: - 5.09765625 frac[35,14] + mem[35,14] = 0
 accuracy_5_4: - 5.09765625 frac[5,4] + mem[5,4] = 0
 accuracy_36_9: - 5.09765625 frac[36,9] + mem[36,9] = 0
 accuracy_37_12: - 5.09765625 frac[37,12] + mem[37,12] = 0
 accuracy_36_4: - 5.09765625 frac[36,4] + mem[36,4] = 0
 accuracy_15_10: - 5.09765625 frac[15,10] + mem[15,10] = 0
 accuracy_37_25: - 5.09765625 frac[37,25] + mem[37,25] = 0
 accuracy_38_22: - 5.09765625 frac[38,22] + mem[38,22] = 0
 accuracy_37_22: - 5.09765625 frac[37,22] + mem[37,22] = 0
 accuracy_32_29: - 5.09765625 frac[32,29] + mem[32,29] = 0
 accuracy_32_0: - 5.09765625 frac[32,0] + mem[32,0] = 0
 accuracy_35_13: - 5.09765625 frac[35,13] + mem[35,13] = 0
 accuracy_5_7: - 5.09765625 frac[5,7] + mem[5,7] = 0
 accuracy_36_14: - 5.09765625 frac[36,14] + mem[36,14] = 0
 accuracy_24_25: - 5.09765625 frac[24,25] + mem[24,25] = 0
 accuracy_37_15: - 5.09765625 frac[37,15] + mem[37,15] = 0
 accuracy_19_12: - 5.09765625 frac[19,12] + mem[19,12] = 0
 accuracy_36_25: - 5.09765625 frac[36,25] + mem[36,25] = 0
 accuracy_15_9: - 5.09765625 frac[15,9] + mem[15,9] = 0
 accuracy_38_12: - 5.09765625 frac[38,12] + mem[38,12] = 0
 accuracy_37_28: - 5.09765625 frac[37,28] + mem[37,28] = 0
 accuracy_26_14: - 5.09765625 frac[26,14] + mem[26,14] = 0
 accuracy_33_22: - 5.09765625 frac[33,22] + mem[33,22] = 0
 accuracy_32_26: - 5.09765625 frac[32,26] + mem[32,26] = 0
 accuracy_3_0: - 5.09765625 frac[3,0] + mem[3,0] = 0
 accuracy_27_15: - 5.09765625 frac[27,15] + mem[27,15] = 0
 accuracy_33_27: - 5.09765625 frac[33,27] + mem[33,27] = 0
 accuracy_32_13: - 5.09765625 frac[32,13] + mem[32,13] = 0
 accuracy_34_0: - 5.09765625 frac[34,0] + mem[34,0] = 0
 accuracy_19_11: - 5.09765625 frac[19,11] + mem[19,11] = 0
 accuracy_36_30: - 5.09765625 frac[36,30] + mem[36,30] = 0
 accuracy_15_12: - 5.09765625 frac[15,12] + mem[15,12] = 0
 accuracy_16_17: - 5.09765625 frac[16,17] + mem[16,17] = 0
 accuracy_38_9: - 5.09765625 frac[38,9] + mem[38,9] = 0
 accuracy_1_25: - 5.09765625 frac[1,25] + mem[1,25] = 0
 accuracy_22_27: - 5.09765625 frac[22,27] + mem[22,27] = 0
 accuracy_38_28: - 5.09765625 frac[38,28] + mem[38,28] = 0
 accuracy_13_25: - 5.09765625 frac[13,25] + mem[13,25] = 0
 accuracy_33_17: - 5.09765625 frac[33,17] + mem[33,17] = 0
 accuracy_32_7: - 5.09765625 frac[32,7] + mem[32,7] = 0
 accuracy_4_13: - 5.09765625 frac[4,13] + mem[4,13] = 0
 accuracy_34_22: - 5.09765625 frac[34,22] + mem[34,22] = 0
 accuracy_5_0: - 5.09765625 frac[5,0] + mem[5,0] = 0
 accuracy_35_7: - 5.09765625 frac[35,7] + mem[35,7] = 0
 accuracy_34_5: - 5.09765625 frac[34,5] + mem[34,5] = 0
 accuracy_8_16: - 5.09765625 frac[8,16] + mem[8,16] = 0
 accuracy_36_0: - 5.09765625 frac[36,0] + mem[36,0] = 0
 accuracy_38_7: - 5.09765625 frac[38,7] + mem[38,7] = 0
 accuracy_37_5: - 5.09765625 frac[37,5] + mem[37,5] = 0
 accuracy_20_22: - 5.09765625 frac[20,22] + mem[20,22] = 0
 accuracy_16_30: - 5.09765625 frac[16,30] + mem[16,30] = 0
 accuracy_38_10: - 5.09765625 frac[38,10] + mem[38,10] = 0
 accuracy_38_25: - 5.09765625 frac[38,25] + mem[38,25] = 0
 accuracy_13_28: - 5.09765625 frac[13,28] + mem[13,28] = 0
 accuracy_33_28: - 5.09765625 frac[33,28] + mem[33,28] = 0
 accuracy_32_4: - 5.09765625 frac[32,4] + mem[32,4] = 0
 accuracy_34_27: - 5.09765625 frac[34,27] + mem[34,27] = 0
 accuracy_36_10: - 5.09765625 frac[36,10] + mem[36,10] = 0
 accuracy_9_27: - 5.09765625 frac[9,27] + mem[9,27] = 0
 accuracy_7_0: - 5.09765625 frac[7,0] + mem[7,0] = 0
 accuracy_31_7: - 5.09765625 frac[31,7] + mem[31,7] = 0
 accuracy_37_11: - 5.09765625 frac[37,11] + mem[37,11] = 0
 accuracy_36_5: - 5.09765625 frac[36,5] + mem[36,5] = 0
 accuracy_38_0: - 5.09765625 frac[38,0] + mem[38,0] = 0
 accuracy_36_16: - 5.09765625 frac[36,16] + mem[36,16] = 0
 accuracy_18_7: - 5.09765625 frac[18,7] + mem[18,7] = 0
 accuracy_16_27: - 5.09765625 frac[16,27] + mem[16,27] = 0
 accuracy_7_30: - 5.09765625 frac[7,30] + mem[7,30] = 0
 accuracy_32_30: - 5.09765625 frac[32,30] + mem[32,30] = 0
 accuracy_22_1: - 5.09765625 frac[22,1] + mem[22,1] = 0
 accuracy_34_17: - 5.09765625 frac[34,17] + mem[34,17] = 0
 accuracy_33_12: - 5.09765625 frac[33,12] + mem[33,12] = 0
 accuracy_36_15: - 5.09765625 frac[36,15] + mem[36,15] = 0
 accuracy_35_1: - 5.09765625 frac[35,1] + mem[35,1] = 0
 accuracy_34_11: - 5.09765625 frac[34,11] + mem[34,11] = 0
 accuracy_7_7: - 5.09765625 frac[7,7] + mem[7,7] = 0
 accuracy_37_14: - 5.09765625 frac[37,14] + mem[37,14] = 0
 accuracy_36_26: - 5.09765625 frac[36,26] + mem[36,26] = 0
 accuracy_0_16: - 5.09765625 frac[0,16] + mem[0,16] = 0
 accuracy_38_13: - 5.09765625 frac[38,13] + mem[38,13] = 0
 accuracy_37_27: - 5.09765625 frac[37,27] + mem[37,27] = 0
 accuracy_36_21: - 5.09765625 frac[36,21] + mem[36,21] = 0
 accuracy_26_15: - 5.09765625 frac[26,15] + mem[26,15] = 0
 accuracy_33_21: - 5.09765625 frac[33,21] + mem[33,21] = 0
 accuracy_33_26: - 5.09765625 frac[33,26] + mem[33,26] = 0
 accuracy_30_9: - 5.09765625 frac[30,9] + mem[30,9] = 0
 accuracy_35_11: - 5.09765625 frac[35,11] + mem[35,11] = 0
 accuracy_34_1: - 5.09765625 frac[34,1] + mem[34,1] = 0
 accuracy_33_15: - 5.09765625 frac[33,15] + mem[33,15] = 0
 accuracy_30_28: - 5.09765625 frac[30,28] + mem[30,28] = 0
 accuracy_36_12: - 5.09765625 frac[36,12] + mem[36,12] = 0
 accuracy_35_28: - 5.09765625 frac[35,28] + mem[35,28] = 0
 accuracy_34_12: - 5.09765625 frac[34,12] + mem[34,12] = 0
 accuracy_37_1: - 5.09765625 frac[37,1] + mem[37,1] = 0
 accuracy_35_17: - 5.09765625 frac[35,17] + mem[35,17] = 0
 accuracy_15_15: - 5.09765625 frac[15,15] + mem[15,15] = 0
 accuracy_38_14: - 5.09765625 frac[38,14] + mem[38,14] = 0
 accuracy_12_21: - 5.09765625 frac[12,21] + mem[12,21] = 0
 accuracy_37_30: - 5.09765625 frac[37,30] + mem[37,30] = 0
 accuracy_32_21: - 5.09765625 frac[32,21] + mem[32,21] = 0
 accuracy_38_29: - 5.09765625 frac[38,29] + mem[38,29] = 0
 accuracy_33_16: - 5.09765625 frac[33,16] + mem[33,16] = 0
 accuracy_27_13: - 5.09765625 frac[27,13] + mem[27,13] = 0
 accuracy_33_5: - 5.09765625 frac[33,5] + mem[33,5] = 0
 accuracy_30_10: - 5.09765625 frac[30,10] + mem[30,10] = 0
 accuracy_33_10: - 5.09765625 frac[33,10] + mem[33,10] = 0
 accuracy_8_17: - 5.09765625 frac[8,17] + mem[8,17] = 0
 accuracy_36_28: - 5.09765625 frac[36,28] + mem[36,28] = 0
 accuracy_37_17: - 5.09765625 frac[37,17] + mem[37,17] = 0
 accuracy_2_21: - 5.09765625 frac[2,21] + mem[2,21] = 0
 accuracy_38_30: - 5.09765625 frac[38,30] + mem[38,30] = 0
 mem_tot_CPU1: - mem[0,16] + mem_tot_CPU1 = 0
 R194: - 3.05176e-05 mem_tot_CPU1 + normalized_mem_tot_CPU1 = 0
 rows_tot_CPU1: - 3 frac[0,16] + rows_tot_CPU1 = 0
 R196: - 0.0833333333333333 rows_tot_CPU1 + normalized_rows_tot_CPU1 = 0
 mem_tot_CPU2: - mem[1,26] - mem[1,25] + mem_tot_CPU2 = 0
 R198: - 3.05176e-05 mem_tot_CPU2 + normalized_mem_tot_CPU2 = 0
 rows_tot_CPU2: - 3 frac[1,26] - 3 frac[1,25] + rows_tot_CPU2 = 0
 R200: - 0.0833333333333333 rows_tot_CPU2 + normalized_rows_tot_CPU2 = 0
 mem_tot_CPU3: - mem[2,22] - mem[2,21] + mem_tot_CPU3 = 0
 R202: - 3.05176e-05 mem_tot_CPU3 + normalized_mem_tot_CPU3 = 0
 rows_tot_CPU3: - 3 frac[2,22] - 3 frac[2,21] + rows_tot_CPU3 = 0
 R204: - 0.0833333333333333 rows_tot_CPU3 + normalized_rows_tot_CPU3 = 0
 mem_tot_CPU4: - mem[3,0] + mem_tot_CPU4 = 0
 R206: - 3.05176e-05 mem_tot_CPU4 + normalized_mem_tot_CPU4 = 0
 rows_tot_CPU4: - 3 frac[3,0] + rows_tot_CPU4 = 0
 R208: - 0.0833333333333333 rows_tot_CPU4 + normalized_rows_tot_CPU4 = 0
 mem_tot_CPU5: - mem[4,13] + mem_tot_CPU5 = 0
 R210: - 3.05176e-05 mem_tot_CPU5 + normalized_mem_tot_CPU5 = 0
 rows_tot_CPU5: - 3 frac[4,13] + rows_tot_CPU5 = 0
 R212: - 0.0833333333333333 rows_tot_CPU5 + normalized_rows_tot_CPU5 = 0
 mem_tot_CPU6: - mem[5,4] - mem[5,7] - mem[5,0] + mem_tot_CPU6 = 0
 R214: - 3.05176e-05 mem_tot_CPU6 + normalized_mem_tot_CPU6 = 0
 rows_tot_CPU6: - 3 frac[5,4] - 3 frac[5,7] - 3 frac[5,0] + rows_tot_CPU6
   = 0
 R216: - 0.0833333333333333 rows_tot_CPU6 + normalized_rows_tot_CPU6 = 0
 mem_tot_CPU7: - mem[6,29] + mem_tot_CPU7 = 0
 R218: - 3.05176e-05 mem_tot_CPU7 + normalized_mem_tot_CPU7 = 0
 rows_tot_CPU7: - 3 frac[6,29] + rows_tot_CPU7 = 0
 R220: - 0.0833333333333333 rows_tot_CPU7 + normalized_rows_tot_CPU7 = 0
 mem_tot_CPU8: - mem[7,0] - mem[7,30] - mem[7,7] + mem_tot_CPU8 = 0
 R222: - 3.05176e-05 mem_tot_CPU8 + normalized_mem_tot_CPU8 = 0
 rows_tot_CPU8: - 3 frac[7,0] - 3 frac[7,30] - 3 frac[7,7] + rows_tot_CPU8
   = 0
 R224: - 0.0833333333333333 rows_tot_CPU8 + normalized_rows_tot_CPU8 = 0
 mem_tot_CPU9: - mem[8,16] - mem[8,17] + mem_tot_CPU9 = 0
 R226: - 3.05176e-05 mem_tot_CPU9 + normalized_mem_tot_CPU9 = 0
 rows_tot_CPU9: - 3 frac[8,16] - 3 frac[8,17] + rows_tot_CPU9 = 0
 R228: - 0.0833333333333333 rows_tot_CPU9 + normalized_rows_tot_CPU9 = 0
 mem_tot_CPU10: - mem[9,26] - mem[9,27] + mem_tot_CPU10 = 0
 R230: - 3.05176e-05 mem_tot_CPU10 + normalized_mem_tot_CPU10 = 0
 rows_tot_CPU10: - 3 frac[9,26] - 3 frac[9,27] + rows_tot_CPU10 = 0
 R232: - 0.0833333333333333 rows_tot_CPU10 + normalized_rows_tot_CPU10 = 0
 mem_tot_CPU11: - mem[10,14] + mem_tot_CPU11 = 0
 R234: - 3.05176e-05 mem_tot_CPU11 + normalized_mem_tot_CPU11 = 0
 rows_tot_CPU11: - 3 frac[10,14] + rows_tot_CPU11 = 0
 R236: - 0.0833333333333333 rows_tot_CPU11 + normalized_rows_tot_CPU11 = 0
 mem_tot_CPU12: - mem[11,5] + mem_tot_CPU12 = 0
 R238: - 3.05176e-05 mem_tot_CPU12 + normalized_mem_tot_CPU12 = 0
 rows_tot_CPU12: - 3 frac[11,5] + rows_tot_CPU12 = 0
 R240: - 0.0833333333333333 rows_tot_CPU12 + normalized_rows_tot_CPU12 = 0
 mem_tot_CPU13: - mem[12,17] - mem[12,21] + mem_tot_CPU13 = 0
 R242: - 3.05176e-05 mem_tot_CPU13 + normalized_mem_tot_CPU13 = 0
 rows_tot_CPU13: - 3 frac[12,17] - 3 frac[12,21] + rows_tot_CPU13 = 0
 R244: - 0.0833333333333333 rows_tot_CPU13 + normalized_rows_tot_CPU13 = 0
 mem_tot_CPU14: - mem[13,25] - mem[13,28] + mem_tot_CPU14 = 0
 R246: - 3.05176e-05 mem_tot_CPU14 + normalized_mem_tot_CPU14 = 0
 rows_tot_CPU14: - 3 frac[13,25] - 3 frac[13,28] + rows_tot_CPU14 = 0
 R248: - 0.0833333333333333 rows_tot_CPU14 + normalized_rows_tot_CPU14 = 0
 mem_tot_CPU15: - mem[14,17] - mem[14,22] + mem_tot_CPU15 = 0
 R250: - 3.05176e-05 mem_tot_CPU15 + normalized_mem_tot_CPU15 = 0
 rows_tot_CPU15: - 3 frac[14,17] - 3 frac[14,22] + rows_tot_CPU15 = 0
 R252: - 0.0833333333333333 rows_tot_CPU15 + normalized_rows_tot_CPU15 = 0
 mem_tot_CPU16: - mem[15,4] - mem[15,10] - mem[15,9] - mem[15,12]
   - mem[15,15] + mem_tot_CPU16 = 0
 R254: - 3.05176e-05 mem_tot_CPU16 + normalized_mem_tot_CPU16 = 0
 rows_tot_CPU16: - 3 frac[15,4] - 3 frac[15,10] - 3 frac[15,9]
   - 3 frac[15,12] - 3 frac[15,15] + rows_tot_CPU16 = 0
 R256: - 0.0833333333333333 rows_tot_CPU16 + normalized_rows_tot_CPU16 = 0
 mem_tot_CPU17: - mem[16,29] - mem[16,17] - mem[16,30] - mem[16,27]
   + mem_tot_CPU17 = 0
 R258: - 3.05176e-05 mem_tot_CPU17 + normalized_mem_tot_CPU17 = 0
 rows_tot_CPU17: - 3 frac[16,29] - 3 frac[16,17] - 3 frac[16,30]
   - 3 frac[16,27] + rows_tot_CPU17 = 0
 R260: - 0.0833333333333333 rows_tot_CPU17 + normalized_rows_tot_CPU17 = 0
 mem_tot_CPU18: mem_tot_CPU18 = 0
 R262: - 3.05176e-05 mem_tot_CPU18 + normalized_mem_tot_CPU18 = 0
 rows_tot_CPU18: rows_tot_CPU18 = 0
 R264: - 0.0833333333333333 rows_tot_CPU18 + normalized_rows_tot_CPU18 = 0
 mem_tot_CPU19: - mem[18,0] - mem[18,5] - mem[18,7] + mem_tot_CPU19 = 0
 R266: - 3.05176e-05 mem_tot_CPU19 + normalized_mem_tot_CPU19 = 0
 rows_tot_CPU19: - 3 frac[18,0] - 3 frac[18,5] - 3 frac[18,7]
   + rows_tot_CPU19 = 0
 R268: - 0.0833333333333333 rows_tot_CPU19 + normalized_rows_tot_CPU19 = 0
 mem_tot_CPU20: - mem[19,12] - mem[19,11] + mem_tot_CPU20 = 0
 R270: - 3.05176e-05 mem_tot_CPU20 + normalized_mem_tot_CPU20 = 0
 rows_tot_CPU20: - 3 frac[19,12] - 3 frac[19,11] + rows_tot_CPU20 = 0
 R272: - 0.0833333333333333 rows_tot_CPU20 + normalized_rows_tot_CPU20 = 0
 mem_tot_CPU21: - mem[20,22] + mem_tot_CPU21 = 0
 R274: - 3.05176e-05 mem_tot_CPU21 + normalized_mem_tot_CPU21 = 0
 rows_tot_CPU21: - 3 frac[20,22] + rows_tot_CPU21 = 0
 R276: - 0.0833333333333333 rows_tot_CPU21 + normalized_rows_tot_CPU21 = 0
 mem_tot_CPU22: - mem[21,27] + mem_tot_CPU22 = 0
 R278: - 3.05176e-05 mem_tot_CPU22 + normalized_mem_tot_CPU22 = 0
 rows_tot_CPU22: - 3 frac[21,27] + rows_tot_CPU22 = 0
 R280: - 0.0833333333333333 rows_tot_CPU22 + normalized_rows_tot_CPU22 = 0
 mem_tot_CPU23: - mem[22,27] - mem[22,1] + mem_tot_CPU23 = 0
 R282: - 3.05176e-05 mem_tot_CPU23 + normalized_mem_tot_CPU23 = 0
 rows_tot_CPU23: - 3 frac[22,27] - 3 frac[22,1] + rows_tot_CPU23 = 0
 R284: - 0.0833333333333333 rows_tot_CPU23 + normalized_rows_tot_CPU23 = 0
 mem_tot_CPU24: - mem[23,7] + mem_tot_CPU24 = 0
 R286: - 3.05176e-05 mem_tot_CPU24 + normalized_mem_tot_CPU24 = 0
 rows_tot_CPU24: - 3 frac[23,7] + rows_tot_CPU24 = 0
 R288: - 0.0833333333333333 rows_tot_CPU24 + normalized_rows_tot_CPU24 = 0
 mem_tot_CPU25: - mem[24,25] + mem_tot_CPU25 = 0
 R290: - 3.05176e-05 mem_tot_CPU25 + normalized_mem_tot_CPU25 = 0
 rows_tot_CPU25: - 3 frac[24,25] + rows_tot_CPU25 = 0
 R292: - 0.0833333333333333 rows_tot_CPU25 + normalized_rows_tot_CPU25 = 0
 mem_tot_CPU26: mem_tot_CPU26 = 0
 R294: - 3.05176e-05 mem_tot_CPU26 + normalized_mem_tot_CPU26 = 0
 rows_tot_CPU26: rows_tot_CPU26 = 0
 R296: - 0.0833333333333333 rows_tot_CPU26 + normalized_rows_tot_CPU26 = 0
 mem_tot_CPU27: - mem[26,11] - mem[26,14] - mem[26,15] + mem_tot_CPU27 = 0
 R298: - 3.05176e-05 mem_tot_CPU27 + normalized_mem_tot_CPU27 = 0
 rows_tot_CPU27: - 3 frac[26,11] - 3 frac[26,14] - 3 frac[26,15]
   + rows_tot_CPU27 = 0
 R300: - 0.0833333333333333 rows_tot_CPU27 + normalized_rows_tot_CPU27 = 0
 mem_tot_CPU28: - mem[27,15] - mem[27,13] + mem_tot_CPU28 = 0
 R302: - 3.05176e-05 mem_tot_CPU28 + normalized_mem_tot_CPU28 = 0
 rows_tot_CPU28: - 3 frac[27,15] - 3 frac[27,13] + rows_tot_CPU28 = 0
 R304: - 0.0833333333333333 rows_tot_CPU28 + normalized_rows_tot_CPU28 = 0
 mem_tot_CPU29: - mem[28,17] - mem[28,22] + mem_tot_CPU29 = 0
 R306: - 3.05176e-05 mem_tot_CPU29 + normalized_mem_tot_CPU29 = 0
 rows_tot_CPU29: - 3 frac[28,17] - 3 frac[28,22] + rows_tot_CPU29 = 0
 R308: - 0.0833333333333333 rows_tot_CPU29 + normalized_rows_tot_CPU29 = 0
 mem_tot_CPU30: mem_tot_CPU30 = 0
 R310: - 3.05176e-05 mem_tot_CPU30 + normalized_mem_tot_CPU30 = 0
 rows_tot_CPU30: rows_tot_CPU30 = 0
 R312: - 0.0833333333333333 rows_tot_CPU30 + normalized_rows_tot_CPU30 = 0
 mem_tot_CPU31: - mem[30,9] - mem[30,28] - mem[30,10] + mem_tot_CPU31 = 0
 R314: - 3.05176e-05 mem_tot_CPU31 + normalized_mem_tot_CPU31 = 0
 rows_tot_CPU31: - 3 frac[30,9] - 3 frac[30,28] - 3 frac[30,10]
   + rows_tot_CPU31 = 0
 R316: - 0.0833333333333333 rows_tot_CPU31 + normalized_rows_tot_CPU31 = 0
 mem_tot_CPU32: - mem[31,1] - mem[31,7] + mem_tot_CPU32 = 0
 R318: - 3.05176e-05 mem_tot_CPU32 + normalized_mem_tot_CPU32 = 0
 rows_tot_CPU32: - 3 frac[31,1] - 3 frac[31,7] + rows_tot_CPU32 = 0
 R320: - 0.0833333333333333 rows_tot_CPU32 + normalized_rows_tot_CPU32 = 0
 mem_tot_tor_P41: - mem[32,22] - mem[32,25] - mem[32,16] - mem[32,29]
   - mem[32,0] - mem[32,26] - mem[32,13] - mem[32,7] - mem[32,4]
   - mem[32,30] - mem[32,21] + mem_tot_tor_P41 = 0
 R322: - 0.00173611111111111 mem_tot_tor_P41 + normalized_mem_tot_tor_P41
   = 0
 rows_tot_tor_P41: - 3 frac[32,22] - 3 frac[32,25] - 3 frac[32,16]
   - 3 frac[32,29] - 3 frac[32,0] - 3 frac[32,26] - 3 frac[32,13]
   - 3 frac[32,7] - 3 frac[32,4] - 3 frac[32,30] - 3 frac[32,21]
   + rows_tot_tor_P41 = 0
 R324: - 0.0208333333333333 rows_tot_tor_P41 + normalized_rows_tot_tor_P41
   = 0
 mem_tot_tor_P42: - mem[33,25] - mem[33,14] - mem[33,4] - mem[33,9]
   - mem[33,22] - mem[33,27] - mem[33,17] - mem[33,28] - mem[33,12]
   - mem[33,21] - mem[33,26] - mem[33,15] - mem[33,16] - mem[33,5]
   - mem[33,10] + mem_tot_tor_P42 = 0
 R326: - 0.00173611111111111 mem_tot_tor_P42 + normalized_mem_tot_tor_P42
   = 0
 rows_tot_tor_P42: - 3 frac[33,25] - 3 frac[33,14] - 3 frac[33,4]
   - 3 frac[33,9] - 3 frac[33,22] - 3 frac[33,27] - 3 frac[33,17]
   - 3 frac[33,28] - 3 frac[33,12] - 3 frac[33,21] - 3 frac[33,26]
   - 3 frac[33,15] - 3 frac[33,16] - 3 frac[33,5] - 3 frac[33,10]
   + rows_tot_tor_P42 = 0
 R328: - 0.0208333333333333 rows_tot_tor_P42 + normalized_rows_tot_tor_P42
   = 0
 mem_tot_tor_P43: - mem[34,7] - mem[34,29] - mem[34,30] - mem[34,0]
   - mem[34,22] - mem[34,5] - mem[34,27] - mem[34,17] - mem[34,11]
   - mem[34,1] - mem[34,12] + mem_tot_tor_P43 = 0
 R330: - 0.00173611111111111 mem_tot_tor_P43 + normalized_mem_tot_tor_P43
   = 0
 rows_tot_tor_P43: - 3 frac[34,7] - 3 frac[34,29] - 3 frac[34,30]
   - 3 frac[34,0] - 3 frac[34,22] - 3 frac[34,5] - 3 frac[34,27]
   - 3 frac[34,17] - 3 frac[34,11] - 3 frac[34,1] - 3 frac[34,12]
   + rows_tot_tor_P43 = 0
 R332: - 0.0208333333333333 rows_tot_tor_P43 + normalized_rows_tot_tor_P43
   = 0
 mem_tot_tor_P44: - mem[35,22] - mem[35,15] - mem[35,10] - mem[35,9]
   - mem[35,25] - mem[35,14] - mem[35,13] - mem[35,7] - mem[35,1]
   - mem[35,11] - mem[35,28] - mem[35,17] + mem_tot_tor_P44 = 0
 R334: - 0.00173611111111111 mem_tot_tor_P44 + normalized_mem_tot_tor_P44
   = 0
 rows_tot_tor_P44: - 3 frac[35,22] - 3 frac[35,15] - 3 frac[35,10]
   - 3 frac[35,9] - 3 frac[35,25] - 3 frac[35,14] - 3 frac[35,13]
   - 3 frac[35,7] - 3 frac[35,1] - 3 frac[35,11] - 3 frac[35,28]
   - 3 frac[35,17] + rows_tot_tor_P44 = 0
 R336: - 0.0208333333333333 rows_tot_tor_P44 + normalized_rows_tot_tor_P44
   = 0
 mem_tot_l1_P41: - mem[36,17] - mem[36,27] - mem[36,22] - mem[36,13]
   - mem[36,29] - mem[36,7] - mem[36,9] - mem[36,4] - mem[36,14]
   - mem[36,25] - mem[36,30] - mem[36,0] - mem[36,10] - mem[36,5]
   - mem[36,16] - mem[36,15] - mem[36,26] - mem[36,21] - mem[36,12]
   - mem[36,28] + mem_tot_l1_P41 = 0
 R338: - 0.00173611111111111 mem_tot_l1_P41 + normalized_mem_tot_l1_P41
   = 0
 rows_tot_l1_P41: - 3 frac[36,17] - 3 frac[36,27] - 3 frac[36,22]
   - 3 frac[36,13] - 3 frac[36,29] - 3 frac[36,7] - 3 frac[36,9]
   - 3 frac[36,4] - 3 frac[36,14] - 3 frac[36,25] - 3 frac[36,30]
   - 3 frac[36,0] - 3 frac[36,10] - 3 frac[36,5] - 3 frac[36,16]
   - 3 frac[36,15] - 3 frac[36,26] - 3 frac[36,21] - 3 frac[36,12]
   - 3 frac[36,28] + rows_tot_l1_P41 = 0
 R340: - 0.0208333333333333 rows_tot_l1_P41 + normalized_rows_tot_l1_P41
   = 0
 mem_tot_l1_P42: - mem[37,10] - mem[37,7] - mem[37,13] - mem[37,0]
   - mem[37,29] - mem[37,9] - mem[37,12] - mem[37,25] - mem[37,22]
   - mem[37,15] - mem[37,28] - mem[37,5] - mem[37,11] - mem[37,14]
   - mem[37,27] - mem[37,1] - mem[37,30] - mem[37,17] + mem_tot_l1_P42 = 0
 R342: - 0.00173611111111111 mem_tot_l1_P42 + normalized_mem_tot_l1_P42
   = 0
 rows_tot_l1_P42: - 3 frac[37,10] - 3 frac[37,7] - 3 frac[37,13]
   - 3 frac[37,0] - 3 frac[37,29] - 3 frac[37,9] - 3 frac[37,12]
   - 3 frac[37,25] - 3 frac[37,22] - 3 frac[37,15] - 3 frac[37,28]
   - 3 frac[37,5] - 3 frac[37,11] - 3 frac[37,14] - 3 frac[37,27]
   - 3 frac[37,1] - 3 frac[37,30] - 3 frac[37,17] + rows_tot_l1_P42 = 0
 R344: - 0.0208333333333333 rows_tot_l1_P42 + normalized_rows_tot_l1_P42
   = 0
 mem_tot_l2_P4: - mem[38,27] - mem[38,17] - mem[38,15] - mem[38,5]
   - mem[38,22] - mem[38,12] - mem[38,9] - mem[38,28] - mem[38,7]
   - mem[38,10] - mem[38,25] - mem[38,0] - mem[38,13] - mem[38,14]
   - mem[38,29] - mem[38,30] + mem_tot_l2_P4 = 0
 R346: - 0.00173611111111111 mem_tot_l2_P4 + normalized_mem_tot_l2_P4 = 0
 rows_tot_l2_P4: - 3 frac[38,27] - 3 frac[38,17] - 3 frac[38,15]
   - 3 frac[38,5] - 3 frac[38,22] - 3 frac[38,12] - 3 frac[38,9]
   - 3 frac[38,28] - 3 frac[38,7] - 3 frac[38,10] - 3 frac[38,25]
   - 3 frac[38,0] - 3 frac[38,13] - 3 frac[38,14] - 3 frac[38,29]
   - 3 frac[38,30] + rows_tot_l2_P4 = 0
 R348: - 0.0208333333333333 rows_tot_l2_P4 + normalized_rows_tot_l2_P4 = 0
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
 row_capacity_tor_P42: rows_tot_tor_P42 <= 48
 row_capacity_tor_P43: rows_tot_tor_P43 <= 48
 row_capacity_tor_P44: rows_tot_tor_P44 <= 48
 row_capacity_l1_P41: rows_tot_l1_P41 <= 48
 row_capacity_l1_P42: rows_tot_l1_P42 <= 48
 row_capacity_l2_P4: rows_tot_l2_P4 <= 48
Bounds
 frac[34,7] <= 1
 frac[9,26] <= 1
 frac[14,17] <= 1
 frac[37,10] <= 1
 frac[35,22] <= 1
 frac[37,7] <= 1
 frac[36,17] <= 1
 frac[2,22] <= 1
 frac[26,11] <= 1
 frac[38,27] <= 1
 frac[23,7] <= 1
 frac[35,15] <= 1
 frac[34,29] <= 1
 frac[11,5] <= 1
 frac[37,13] <= 1
 frac[36,27] <= 1
 frac[12,17] <= 1
 frac[36,22] <= 1
 frac[15,4] <= 1
 frac[38,17] <= 1
 frac[33,25] <= 1
 frac[35,10] <= 1
 frac[34,30] <= 1
 frac[33,14] <= 1
 frac[36,13] <= 1
 frac[37,0] <= 1
 frac[38,15] <= 1
 frac[37,29] <= 1
 frac[32,22] <= 1
 frac[28,17] <= 1
 frac[32,25] <= 1
 frac[33,4] <= 1
 frac[10,14] <= 1
 frac[35,9] <= 1
 frac[33,9] <= 1
 frac[38,5] <= 1
 frac[36,29] <= 1
 frac[18,0] <= 1
 frac[1,26] <= 1
 frac[28,22] <= 1
 frac[31,1] <= 1
 frac[14,22] <= 1
 frac[37,9] <= 1
 frac[36,7] <= 1
 frac[35,25] <= 1
 frac[6,29] <= 1
 frac[21,27] <= 1
 frac[18,5] <= 1
 frac[16,29] <= 1
 frac[32,16] <= 1
 frac[35,14] <= 1
 frac[5,4] <= 1
 frac[36,9] <= 1
 frac[37,12] <= 1
 frac[36,4] <= 1
 frac[15,10] <= 1
 frac[37,25] <= 1
 frac[38,22] <= 1
 frac[37,22] <= 1
 frac[32,29] <= 1
 frac[32,0] <= 1
 frac[35,13] <= 1
 frac[5,7] <= 1
 frac[36,14] <= 1
 frac[24,25] <= 1
 frac[37,15] <= 1
 frac[19,12] <= 1
 frac[36,25] <= 1
 frac[15,9] <= 1
 frac[38,12] <= 1
 frac[37,28] <= 1
 frac[26,14] <= 1
 frac[33,22] <= 1
 frac[32,26] <= 1
 frac[3,0] <= 1
 frac[27,15] <= 1
 frac[33,27] <= 1
 frac[32,13] <= 1
 frac[34,0] <= 1
 frac[19,11] <= 1
 frac[36,30] <= 1
 frac[15,12] <= 1
 frac[16,17] <= 1
 frac[38,9] <= 1
 frac[1,25] <= 1
 frac[22,27] <= 1
 frac[38,28] <= 1
 frac[13,25] <= 1
 frac[33,17] <= 1
 frac[32,7] <= 1
 frac[4,13] <= 1
 frac[34,22] <= 1
 frac[5,0] <= 1
 frac[35,7] <= 1
 frac[34,5] <= 1
 frac[8,16] <= 1
 frac[36,0] <= 1
 frac[38,7] <= 1
 frac[37,5] <= 1
 frac[20,22] <= 1
 frac[16,30] <= 1
 frac[38,10] <= 1
 frac[38,25] <= 1
 frac[13,28] <= 1
 frac[33,28] <= 1
 frac[32,4] <= 1
 frac[34,27] <= 1
 frac[36,10] <= 1
 frac[9,27] <= 1
 frac[7,0] <= 1
 frac[31,7] <= 1
 frac[37,11] <= 1
 frac[36,5] <= 1
 frac[38,0] <= 1
 frac[36,16] <= 1
 frac[18,7] <= 1
 frac[16,27] <= 1
 frac[7,30] <= 1
 frac[32,30] <= 1
 frac[22,1] <= 1
 frac[34,17] <= 1
 frac[33,12] <= 1
 frac[36,15] <= 1
 frac[35,1] <= 1
 frac[34,11] <= 1
 frac[7,7] <= 1
 frac[37,14] <= 1
 frac[36,26] <= 1
 frac[0,16] <= 1
 frac[38,13] <= 1
 frac[37,27] <= 1
 frac[36,21] <= 1
 frac[26,15] <= 1
 frac[33,21] <= 1
 frac[33,26] <= 1
 frac[30,9] <= 1
 frac[35,11] <= 1
 frac[34,1] <= 1
 frac[33,15] <= 1
 frac[30,28] <= 1
 frac[36,12] <= 1
 frac[35,28] <= 1
 frac[34,12] <= 1
 frac[37,1] <= 1
 frac[35,17] <= 1
 frac[15,15] <= 1
 frac[38,14] <= 1
 frac[12,21] <= 1
 frac[37,30] <= 1
 frac[32,21] <= 1
 frac[38,29] <= 1
 frac[33,16] <= 1
 frac[27,13] <= 1
 frac[33,5] <= 1
 frac[30,10] <= 1
 frac[33,10] <= 1
 frac[8,17] <= 1
 frac[36,28] <= 1
 frac[37,17] <= 1
 frac[2,21] <= 1
 frac[38,30] <= 1
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
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_tor_P41 , mem_tot_tor_P42 ,
   mem_tot_tor_P43 , mem_tot_tor_P44 , mem_tot_l1_P41 , mem_tot_l1_P42 ,
   mem_tot_l2_P4 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU1 , normalized_mem_tot_CPU2 ,
   normalized_mem_tot_CPU3 , normalized_mem_tot_CPU4 ,
   normalized_mem_tot_CPU5 , normalized_mem_tot_CPU6 ,
   normalized_mem_tot_CPU7 , normalized_mem_tot_CPU8 ,
   normalized_mem_tot_CPU9 , normalized_mem_tot_CPU10 ,
   normalized_mem_tot_CPU11 , normalized_mem_tot_CPU12 ,
   normalized_mem_tot_CPU13 , normalized_mem_tot_CPU14 ,
   normalized_mem_tot_CPU15 , normalized_mem_tot_CPU16 ,
   normalized_mem_tot_CPU17 , normalized_mem_tot_CPU18 ,
   normalized_mem_tot_CPU19 , normalized_mem_tot_CPU20 ,
   normalized_mem_tot_CPU21 , normalized_mem_tot_CPU22 ,
   normalized_mem_tot_CPU23 , normalized_mem_tot_CPU24 ,
   normalized_mem_tot_CPU25 , normalized_mem_tot_CPU26 ,
   normalized_mem_tot_CPU27 , normalized_mem_tot_CPU28 ,
   normalized_mem_tot_CPU29 , normalized_mem_tot_CPU30 ,
   normalized_mem_tot_CPU31 , normalized_mem_tot_CPU32 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_tor_P41 , rows_tot_tor_P42 ,
   rows_tot_tor_P43 , rows_tot_tor_P44 , rows_tot_l1_P41 ,
   rows_tot_l1_P42 , rows_tot_l2_P4 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU1 ,
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
End

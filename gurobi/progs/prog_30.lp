\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU_Cluster: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_{CPU15,CPU3,CPU25,CPU28,CPU30,CPU31,CPU13,CPU11}
   + rows_tot_{CPU5,CPU24,CPU16,CPU9,CPU26,CPU8,CPU10,CPU12}
   + rows_tot_{CPU7,CPU22,CPU14,CPU17,CPU19,CPU18,CPU2,CPU20}
   + rows_tot_{CPU29,CPU1,CPU4,CPU21,CPU23,CPU32,CPU27,CPU6}
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
   mem_tot_{CPU15,CPU3,CPU25,CPU28,CPU30,CPU31,CPU13,CPU11}
   + mem_tot_{CPU5,CPU24,CPU16,CPU9,CPU26,CPU8,CPU10,CPU12}
   + mem_tot_{CPU7,CPU22,CPU14,CPU17,CPU19,CPU18,CPU2,CPU20}
   + mem_tot_{CPU29,CPU1,CPU4,CPU21,CPU23,CPU32,CPU27,CPU6}
  CPU_Cluster_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
  tot_mem_P4: Priority=40 Weight=1 AbsTol=0 RelTol=0
   mem_tot_tor_P41 + mem_tot_tor_P42 + mem_tot_tor_P43 + mem_tot_tor_P44
   + mem_tot_l1_P41 + mem_tot_l1_P42 + mem_tot_l2_P4
  P4_load_mem: Priority=30 Weight=1 AbsTol=0 RelTol=0
   max_mem_P4
Subject To
 cov_0_2: frac[0,2] + frac[7,2] >= 1
 cov_1_0: frac[0,0] + frac[7,0] >= 0.7
 cov_2_1: frac[0,1] + frac[7,1] >= 0.7
 cov_3_0: frac[0,0] + frac[7,0] >= 0.9
 cov_4_0: frac[0,0] + frac[7,0] >= 0.9
 cov_5_0: frac[0,0] + frac[5,0] >= 0.9
 cov_6_0: frac[0,0] + frac[4,0] + frac[5,0] + frac[8,0] >= 0.8
 cov_7_1: frac[0,1] + frac[7,1] >= 0.8
 cov_8_6: frac[1,6] + frac[4,6] + frac[5,6] + frac[8,6] >= 0.9
 cov_9_7: frac[1,7] + frac[5,7] >= 0.9
 cov_10_6: frac[1,6] + frac[5,6] + frac[6,6] + frac[8,6] + frac[9,6]
   + frac[10,6] >= 1
 cov_11_4: frac[0,4] + frac[1,4] + frac[5,4] >= 0.9
 cov_12_5: frac[1,5] + frac[2,5] + frac[6,5] + frac[7,5] + frac[9,5]
   >= 0.7
 cov_13_7: frac[1,7] + frac[5,7] + frac[7,7] + frac[8,7] + frac[9,7]
   + frac[10,7] >= 0.8
 cov_14_5: frac[1,5] + frac[5,5] >= 1
 cov_15_6: frac[1,6] + frac[4,6] + frac[5,6] + frac[8,6] >= 0.8
 cov_16_11: frac[2,11] + frac[4,11] + frac[6,11] + frac[8,11] + frac[9,11]
   + frac[10,11] >= 0.8
 cov_17_9: frac[2,9] + frac[4,9] + frac[6,9] + frac[8,9] + frac[9,9]
   + frac[10,9] >= 0.7
 cov_18_10: frac[2,10] + frac[4,10] + frac[5,10] + frac[8,10] >= 0.7
 cov_19_11: frac[2,11] + frac[6,11] >= 0.9
 cov_20_10: frac[2,10] + frac[5,10] + frac[6,10] + frac[8,10] + frac[9,10]
   + frac[10,10] >= 0.9
 cov_21_8: frac[2,8] + frac[3,8] + frac[4,8] + frac[7,8] + frac[8,8]
   + frac[9,8] + frac[10,8] >= 0.9
 cov_22_10: frac[0,10] + frac[2,10] + frac[4,10] + frac[5,10] + frac[8,10]
   >= 0.8
 cov_23_11: frac[2,11] + frac[4,11] + frac[6,11] + frac[8,11] + frac[9,11]
   + frac[10,11] >= 0.8
 cov_24_12: frac[3,12] + frac[4,12] + frac[7,12] + frac[8,12] + frac[9,12]
   + frac[10,12] >= 0.9
 cov_25_12: frac[3,12] + frac[4,12] + frac[6,12] + frac[8,12] + frac[9,12]
   + frac[10,12] >= 0.9
 cov_26_13: frac[1,13] + frac[3,13] + frac[5,13] + frac[6,13] + frac[8,13]
   + frac[9,13] + frac[10,13] >= 0.8
 cov_27_15: frac[3,15] + frac[4,15] + frac[7,15] + frac[8,15] + frac[9,15]
   + frac[10,15] >= 0.9
 cov_28_15: frac[3,15] + frac[7,15] >= 0.8
 cov_29_15: frac[3,15] + frac[6,15] >= 0.9
 cov_30_13: frac[3,13] + frac[7,13] >= 0.8
 cov_31_14: frac[3,14] + frac[4,14] + frac[6,14] + frac[8,14] + frac[9,14]
   + frac[10,14] >= 1
 accuracy_6_9: - 96 frac[6,9] + mem[6,9] = 0
 accuracy_7_12: - 96 frac[7,12] + mem[7,12] = 0
 accuracy_1_6: - 50.9765625 frac[1,6] + mem[1,6] = 0
 accuracy_0_10: - 50.9765625 frac[0,10] + mem[0,10] = 0
 accuracy_2_5: - 50.9765625 frac[2,5] + mem[2,5] = 0
 accuracy_4_0: - 96 frac[4,0] + mem[4,0] = 0
 accuracy_10_8: - 96 frac[10,8] + mem[10,8] = 0
 accuracy_5_5: - 96 frac[5,5] + mem[5,5] = 0
 accuracy_10_7: - 96 frac[10,7] + mem[10,7] = 0
 accuracy_6_10: - 96 frac[6,10] + mem[6,10] = 0
 accuracy_0_4: - 50.9765625 frac[0,4] + mem[0,4] = 0
 accuracy_8_15: - 96 frac[8,15] + mem[8,15] = 0
 accuracy_4_10: - 96 frac[4,10] + mem[4,10] = 0
 accuracy_9_14: - 96 frac[9,14] + mem[9,14] = 0
 accuracy_10_13: - 96 frac[10,13] + mem[10,13] = 0
 accuracy_7_5: - 96 frac[7,5] + mem[7,5] = 0
 accuracy_0_1: - 50.9765625 frac[0,1] + mem[0,1] = 0
 accuracy_3_12: - 50.9765625 frac[3,12] + mem[3,12] = 0
 accuracy_8_12: - 96 frac[8,12] + mem[8,12] = 0
 accuracy_4_15: - 96 frac[4,15] + mem[4,15] = 0
 accuracy_2_11: - 50.9765625 frac[2,11] + mem[2,11] = 0
 accuracy_9_9: - 96 frac[9,9] + mem[9,9] = 0
 accuracy_10_14: - 96 frac[10,14] + mem[10,14] = 0
 accuracy_6_13: - 96 frac[6,13] + mem[6,13] = 0
 accuracy_7_8: - 96 frac[7,8] + mem[7,8] = 0
 accuracy_8_9: - 96 frac[8,9] + mem[8,9] = 0
 accuracy_4_12: - 96 frac[4,12] + mem[4,12] = 0
 accuracy_7_2: - 96 frac[7,2] + mem[7,2] = 0
 accuracy_6_14: - 96 frac[6,14] + mem[6,14] = 0
 accuracy_7_15: - 96 frac[7,15] + mem[7,15] = 0
 accuracy_1_5: - 50.9765625 frac[1,5] + mem[1,5] = 0
 accuracy_8_6: - 96 frac[8,6] + mem[8,6] = 0
 accuracy_10_9: - 96 frac[10,9] + mem[10,9] = 0
 accuracy_9_7: - 96 frac[9,7] + mem[9,7] = 0
 accuracy_5_4: - 96 frac[5,4] + mem[5,4] = 0
 accuracy_7_1: - 96 frac[7,1] + mem[7,1] = 0
 accuracy_6_11: - 96 frac[6,11] + mem[6,11] = 0
 accuracy_4_11: - 96 frac[4,11] + mem[4,11] = 0
 accuracy_9_13: - 96 frac[9,13] + mem[9,13] = 0
 accuracy_5_10: - 96 frac[5,10] + mem[5,10] = 0
 accuracy_4_6: - 96 frac[4,6] + mem[4,6] = 0
 accuracy_10_10: - 96 frac[10,10] + mem[10,10] = 0
 accuracy_5_7: - 96 frac[5,7] + mem[5,7] = 0
 accuracy_0_2: - 50.9765625 frac[0,2] + mem[0,2] = 0
 accuracy_3_15: - 50.9765625 frac[3,15] + mem[3,15] = 0
 accuracy_8_13: - 96 frac[8,13] + mem[8,13] = 0
 accuracy_4_8: - 96 frac[4,8] + mem[4,8] = 0
 accuracy_2_8: - 50.9765625 frac[2,8] + mem[2,8] = 0
 accuracy_9_8: - 96 frac[9,8] + mem[9,8] = 0
 accuracy_8_0: - 96 frac[8,0] + mem[8,0] = 0
 accuracy_5_13: - 96 frac[5,13] + mem[5,13] = 0
 accuracy_10_15: - 96 frac[10,15] + mem[10,15] = 0
 accuracy_8_10: - 96 frac[8,10] + mem[8,10] = 0
 accuracy_9_11: - 96 frac[9,11] + mem[9,11] = 0
 accuracy_5_0: - 96 frac[5,0] + mem[5,0] = 0
 accuracy_6_15: - 96 frac[6,15] + mem[6,15] = 0
 accuracy_1_4: - 50.9765625 frac[1,4] + mem[1,4] = 0
 accuracy_8_7: - 96 frac[8,7] + mem[8,7] = 0
 accuracy_9_6: - 96 frac[9,6] + mem[9,6] = 0
 accuracy_6_5: - 96 frac[6,5] + mem[6,5] = 0
 accuracy_7_0: - 96 frac[7,0] + mem[7,0] = 0
 accuracy_7_13: - 96 frac[7,13] + mem[7,13] = 0
 accuracy_1_7: - 50.9765625 frac[1,7] + mem[1,7] = 0
 accuracy_9_12: - 96 frac[9,12] + mem[9,12] = 0
 accuracy_10_11: - 96 frac[10,11] + mem[10,11] = 0
 accuracy_6_6: - 96 frac[6,6] + mem[6,6] = 0
 accuracy_5_6: - 96 frac[5,6] + mem[5,6] = 0
 accuracy_10_6: - 96 frac[10,6] + mem[10,6] = 0
 accuracy_7_7: - 96 frac[7,7] + mem[7,7] = 0
 accuracy_3_14: - 50.9765625 frac[3,14] + mem[3,14] = 0
 accuracy_8_14: - 96 frac[8,14] + mem[8,14] = 0
 accuracy_4_9: - 96 frac[4,9] + mem[4,9] = 0
 accuracy_2_9: - 50.9765625 frac[2,9] + mem[2,9] = 0
 accuracy_9_15: - 96 frac[9,15] + mem[9,15] = 0
 accuracy_10_12: - 96 frac[10,12] + mem[10,12] = 0
 accuracy_0_0: - 50.9765625 frac[0,0] + mem[0,0] = 0
 accuracy_3_13: - 50.9765625 frac[3,13] + mem[3,13] = 0
 accuracy_1_13: - 50.9765625 frac[1,13] + mem[1,13] = 0
 accuracy_8_11: - 96 frac[8,11] + mem[8,11] = 0
 accuracy_4_14: - 96 frac[4,14] + mem[4,14] = 0
 accuracy_2_10: - 50.9765625 frac[2,10] + mem[2,10] = 0
 accuracy_9_10: - 96 frac[9,10] + mem[9,10] = 0
 accuracy_6_12: - 96 frac[6,12] + mem[6,12] = 0
 accuracy_3_8: - 50.9765625 frac[3,8] + mem[3,8] = 0
 accuracy_8_8: - 96 frac[8,8] + mem[8,8] = 0
 accuracy_9_5: - 96 frac[9,5] + mem[9,5] = 0
 mem_tot_{CPU15,CPU3,CPU25,CPU28,CPU30,CPU31,CPU13,CPU11}: - mem[0,10]
   - mem[0,4] - mem[0,1] - mem[0,2] - mem[0,0]
   + mem_tot_{CPU15,CPU3,CPU25,CPU28,CPU30,CPU31,CPU13,CPU11} = 0
 R119:
   - 3.8147e-06 mem_tot_{CPU15,CPU3,CPU25,CPU28,CPU30,CPU31,CPU13,CPU11}
   + normalized_mem_tot_{CPU15,CPU3,CPU25,CPU28,CPU30,CPU31,CPU13,CPU11}
   = 0
 rows_tot_{CPU15,CPU3,CPU25,CPU28,CPU30,CPU31,CPU13,CPU11}: - 3 frac[0,10]
   - 3 frac[0,4] - 3 frac[0,1] - 3 frac[0,2] - 3 frac[0,0]
   + rows_tot_{CPU15,CPU3,CPU25,CPU28,CPU30,CPU31,CPU13,CPU11} = 0
 R121:
   - 0.0104166666666667 rows_tot_{CPU15,CPU3,CPU25,CPU28,CPU30,CPU31,CPU13,CPU11}
   + normalized_rows_tot_{CPU15,CPU3,CPU25,CPU28,CPU30,CPU31,CPU13,CPU11}
   = 0
 mem_tot_{CPU5,CPU24,CPU16,CPU9,CPU26,CPU8,CPU10,CPU12}: - mem[1,6]
   - mem[1,5] - mem[1,4] - mem[1,7] - mem[1,13]
   + mem_tot_{CPU5,CPU24,CPU16,CPU9,CPU26,CPU8,CPU10,CPU12} = 0
 R123: - 3.8147e-06 mem_tot_{CPU5,CPU24,CPU16,CPU9,CPU26,CPU8,CPU10,CPU12}
   + normalized_mem_tot_{CPU5,CPU24,CPU16,CPU9,CPU26,CPU8,CPU10,CPU12} = 0
 rows_tot_{CPU5,CPU24,CPU16,CPU9,CPU26,CPU8,CPU10,CPU12}: - 3 frac[1,6]
   - 3 frac[1,5] - 3 frac[1,4] - 3 frac[1,7] - 3 frac[1,13]
   + rows_tot_{CPU5,CPU24,CPU16,CPU9,CPU26,CPU8,CPU10,CPU12} = 0
 R125:
   - 0.0104166666666667 rows_tot_{CPU5,CPU24,CPU16,CPU9,CPU26,CPU8,CPU10,CPU12}
   + normalized_rows_tot_{CPU5,CPU24,CPU16,CPU9,CPU26,CPU8,CPU10,CPU12}
   = 0
 mem_tot_{CPU7,CPU22,CPU14,CPU17,CPU19,CPU18,CPU2,CPU20}: - mem[2,5]
   - mem[2,11] - mem[2,8] - mem[2,9] - mem[2,10]
   + mem_tot_{CPU7,CPU22,CPU14,CPU17,CPU19,CPU18,CPU2,CPU20} = 0
 R127: - 3.8147e-06 mem_tot_{CPU7,CPU22,CPU14,CPU17,CPU19,CPU18,CPU2,CPU20}
   + normalized_mem_tot_{CPU7,CPU22,CPU14,CPU17,CPU19,CPU18,CPU2,CPU20}
   = 0
 rows_tot_{CPU7,CPU22,CPU14,CPU17,CPU19,CPU18,CPU2,CPU20}: - 3 frac[2,5]
   - 3 frac[2,11] - 3 frac[2,8] - 3 frac[2,9] - 3 frac[2,10]
   + rows_tot_{CPU7,CPU22,CPU14,CPU17,CPU19,CPU18,CPU2,CPU20} = 0
 R129:
   - 0.0104166666666667 rows_tot_{CPU7,CPU22,CPU14,CPU17,CPU19,CPU18,CPU2,CPU20}
   + normalized_rows_tot_{CPU7,CPU22,CPU14,CPU17,CPU19,CPU18,CPU2,CPU20}
   = 0
 mem_tot_{CPU29,CPU1,CPU4,CPU21,CPU23,CPU32,CPU27,CPU6}: - mem[3,12]
   - mem[3,15] - mem[3,14] - mem[3,13] - mem[3,8]
   + mem_tot_{CPU29,CPU1,CPU4,CPU21,CPU23,CPU32,CPU27,CPU6} = 0
 R131: - 3.8147e-06 mem_tot_{CPU29,CPU1,CPU4,CPU21,CPU23,CPU32,CPU27,CPU6}
   + normalized_mem_tot_{CPU29,CPU1,CPU4,CPU21,CPU23,CPU32,CPU27,CPU6} = 0
 rows_tot_{CPU29,CPU1,CPU4,CPU21,CPU23,CPU32,CPU27,CPU6}: - 3 frac[3,12]
   - 3 frac[3,15] - 3 frac[3,14] - 3 frac[3,13] - 3 frac[3,8]
   + rows_tot_{CPU29,CPU1,CPU4,CPU21,CPU23,CPU32,CPU27,CPU6} = 0
 R133:
   - 0.0104166666666667 rows_tot_{CPU29,CPU1,CPU4,CPU21,CPU23,CPU32,CPU27,CPU6}
   + normalized_rows_tot_{CPU29,CPU1,CPU4,CPU21,CPU23,CPU32,CPU27,CPU6}
   = 0
 mem_tot_tor_P41: - mem[4,0] - mem[4,10] - mem[4,15] - mem[4,12]
   - mem[4,11] - mem[4,6] - mem[4,8] - mem[4,9] - mem[4,14]
   + mem_tot_tor_P41 = 0
 R135: - 1.6276041666666666e-04 mem_tot_tor_P41
   + normalized_mem_tot_tor_P41 = 0
 rows_tot_tor_P41: - 3 frac[4,0] - 3 frac[4,10] - 3 frac[4,15]
   - 3 frac[4,12] - 3 frac[4,11] - 3 frac[4,6] - 3 frac[4,8] - 3 frac[4,9]
   - 3 frac[4,14] + rows_tot_tor_P41 = 0
 R137: - 0.0208333333333333 rows_tot_tor_P41 + normalized_rows_tot_tor_P41
   = 0
 mem_tot_tor_P42: - mem[5,5] - mem[5,4] - mem[5,10] - mem[5,7] - mem[5,13]
   - mem[5,0] - mem[5,6] + mem_tot_tor_P42 = 0
 R139: - 1.6276041666666666e-04 mem_tot_tor_P42
   + normalized_mem_tot_tor_P42 = 0
 rows_tot_tor_P42: - 3 frac[5,5] - 3 frac[5,4] - 3 frac[5,10] - 3 frac[5,7]
   - 3 frac[5,13] - 3 frac[5,0] - 3 frac[5,6] + rows_tot_tor_P42 = 0
 R141: - 0.0208333333333333 rows_tot_tor_P42 + normalized_rows_tot_tor_P42
   = 0
 mem_tot_tor_P43: - mem[6,9] - mem[6,10] - mem[6,13] - mem[6,14]
   - mem[6,11] - mem[6,15] - mem[6,5] - mem[6,6] - mem[6,12]
   + mem_tot_tor_P43 = 0
 R143: - 1.6276041666666666e-04 mem_tot_tor_P43
   + normalized_mem_tot_tor_P43 = 0
 rows_tot_tor_P43: - 3 frac[6,9] - 3 frac[6,10] - 3 frac[6,13]
   - 3 frac[6,14] - 3 frac[6,11] - 3 frac[6,15] - 3 frac[6,5] - 3 frac[6,6]
   - 3 frac[6,12] + rows_tot_tor_P43 = 0
 R145: - 0.0208333333333333 rows_tot_tor_P43 + normalized_rows_tot_tor_P43
   = 0
 mem_tot_tor_P44: - mem[7,12] - mem[7,5] - mem[7,8] - mem[7,2] - mem[7,15]
   - mem[7,1] - mem[7,0] - mem[7,13] - mem[7,7] + mem_tot_tor_P44 = 0
 R147: - 1.6276041666666666e-04 mem_tot_tor_P44
   + normalized_mem_tot_tor_P44 = 0
 rows_tot_tor_P44: - 3 frac[7,12] - 3 frac[7,5] - 3 frac[7,8] - 3 frac[7,2]
   - 3 frac[7,15] - 3 frac[7,1] - 3 frac[7,0] - 3 frac[7,13] - 3 frac[7,7]
   + rows_tot_tor_P44 = 0
 R149: - 0.0208333333333333 rows_tot_tor_P44 + normalized_rows_tot_tor_P44
   = 0
 mem_tot_l1_P41: - mem[8,15] - mem[8,12] - mem[8,9] - mem[8,6] - mem[8,13]
   - mem[8,0] - mem[8,10] - mem[8,7] - mem[8,14] - mem[8,11] - mem[8,8]
   + mem_tot_l1_P41 = 0
 R151: - 1.6276041666666666e-04 mem_tot_l1_P41 + normalized_mem_tot_l1_P41
   = 0
 rows_tot_l1_P41: - 3 frac[8,15] - 3 frac[8,12] - 3 frac[8,9] - 3 frac[8,6]
   - 3 frac[8,13] - 3 frac[8,0] - 3 frac[8,10] - 3 frac[8,7] - 3 frac[8,14]
   - 3 frac[8,11] - 3 frac[8,8] + rows_tot_l1_P41 = 0
 R153: - 0.0208333333333333 rows_tot_l1_P41 + normalized_rows_tot_l1_P41
   = 0
 mem_tot_l1_P42: - mem[9,14] - mem[9,9] - mem[9,7] - mem[9,13] - mem[9,8]
   - mem[9,11] - mem[9,6] - mem[9,12] - mem[9,15] - mem[9,10] - mem[9,5]
   + mem_tot_l1_P42 = 0
 R155: - 1.6276041666666666e-04 mem_tot_l1_P42 + normalized_mem_tot_l1_P42
   = 0
 rows_tot_l1_P42: - 3 frac[9,14] - 3 frac[9,9] - 3 frac[9,7] - 3 frac[9,13]
   - 3 frac[9,8] - 3 frac[9,11] - 3 frac[9,6] - 3 frac[9,12] - 3 frac[9,15]
   - 3 frac[9,10] - 3 frac[9,5] + rows_tot_l1_P42 = 0
 R157: - 0.0208333333333333 rows_tot_l1_P42 + normalized_rows_tot_l1_P42
   = 0
 mem_tot_l2_P4: - mem[10,8] - mem[10,7] - mem[10,13] - mem[10,14]
   - mem[10,9] - mem[10,10] - mem[10,15] - mem[10,11] - mem[10,6]
   - mem[10,12] + mem_tot_l2_P4 = 0
 R159: - 1.6276041666666666e-04 mem_tot_l2_P4 + normalized_mem_tot_l2_P4
   = 0
 rows_tot_l2_P4: - 3 frac[10,8] - 3 frac[10,7] - 3 frac[10,13]
   - 3 frac[10,14] - 3 frac[10,9] - 3 frac[10,10] - 3 frac[10,15]
   - 3 frac[10,11] - 3 frac[10,6] - 3 frac[10,12] + rows_tot_l2_P4 = 0
 R161: - 0.0208333333333333 rows_tot_l2_P4 + normalized_rows_tot_l2_P4 = 0
 row_capacity_{CPU15,CPU3,CPU25,CPU28,CPU30,CPU31,CPU13,CPU11}:
   rows_tot_{CPU15,CPU3,CPU25,CPU28,CPU30,CPU31,CPU13,CPU11} <= 96
 row_capacity_{CPU5,CPU24,CPU16,CPU9,CPU26,CPU8,CPU10,CPU12}:
   rows_tot_{CPU5,CPU24,CPU16,CPU9,CPU26,CPU8,CPU10,CPU12} <= 96
 row_capacity_{CPU7,CPU22,CPU14,CPU17,CPU19,CPU18,CPU2,CPU20}:
   rows_tot_{CPU7,CPU22,CPU14,CPU17,CPU19,CPU18,CPU2,CPU20} <= 96
 row_capacity_{CPU29,CPU1,CPU4,CPU21,CPU23,CPU32,CPU27,CPU6}:
   rows_tot_{CPU29,CPU1,CPU4,CPU21,CPU23,CPU32,CPU27,CPU6} <= 96
 row_capacity_tor_P41: rows_tot_tor_P41 <= 48
 capacity_mem_par_tor_P41: mem[4,0] <= 1536
 capacity_mem_par_tor_P41: mem[4,10] <= 1536
 capacity_mem_par_tor_P41: mem[4,15] <= 1536
 capacity_mem_par_tor_P41: mem[4,12] <= 1536
 capacity_mem_par_tor_P41: mem[4,11] <= 1536
 capacity_mem_par_tor_P41: mem[4,6] <= 1536
 capacity_mem_par_tor_P41: mem[4,8] <= 1536
 capacity_mem_par_tor_P41: mem[4,9] <= 1536
 capacity_mem_par_tor_P41: mem[4,14] <= 1536
 row_capacity_tor_P42: rows_tot_tor_P42 <= 48
 capacity_mem_par_tor_P42: mem[5,5] <= 1536
 capacity_mem_par_tor_P42: mem[5,4] <= 1536
 capacity_mem_par_tor_P42: mem[5,10] <= 1536
 capacity_mem_par_tor_P42: mem[5,7] <= 1536
 capacity_mem_par_tor_P42: mem[5,13] <= 1536
 capacity_mem_par_tor_P42: mem[5,0] <= 1536
 capacity_mem_par_tor_P42: mem[5,6] <= 1536
 row_capacity_tor_P43: rows_tot_tor_P43 <= 48
 capacity_mem_par_tor_P43: mem[6,9] <= 1536
 capacity_mem_par_tor_P43: mem[6,10] <= 1536
 capacity_mem_par_tor_P43: mem[6,13] <= 1536
 capacity_mem_par_tor_P43: mem[6,14] <= 1536
 capacity_mem_par_tor_P43: mem[6,11] <= 1536
 capacity_mem_par_tor_P43: mem[6,15] <= 1536
 capacity_mem_par_tor_P43: mem[6,5] <= 1536
 capacity_mem_par_tor_P43: mem[6,6] <= 1536
 capacity_mem_par_tor_P43: mem[6,12] <= 1536
 row_capacity_tor_P44: rows_tot_tor_P44 <= 48
 capacity_mem_par_tor_P44: mem[7,12] <= 1536
 capacity_mem_par_tor_P44: mem[7,5] <= 1536
 capacity_mem_par_tor_P44: mem[7,8] <= 1536
 capacity_mem_par_tor_P44: mem[7,2] <= 1536
 capacity_mem_par_tor_P44: mem[7,15] <= 1536
 capacity_mem_par_tor_P44: mem[7,1] <= 1536
 capacity_mem_par_tor_P44: mem[7,0] <= 1536
 capacity_mem_par_tor_P44: mem[7,13] <= 1536
 capacity_mem_par_tor_P44: mem[7,7] <= 1536
 row_capacity_l1_P41: rows_tot_l1_P41 <= 48
 capacity_mem_par_l1_P41: mem[8,15] <= 1536
 capacity_mem_par_l1_P41: mem[8,12] <= 1536
 capacity_mem_par_l1_P41: mem[8,9] <= 1536
 capacity_mem_par_l1_P41: mem[8,6] <= 1536
 capacity_mem_par_l1_P41: mem[8,13] <= 1536
 capacity_mem_par_l1_P41: mem[8,0] <= 1536
 capacity_mem_par_l1_P41: mem[8,10] <= 1536
 capacity_mem_par_l1_P41: mem[8,7] <= 1536
 capacity_mem_par_l1_P41: mem[8,14] <= 1536
 capacity_mem_par_l1_P41: mem[8,11] <= 1536
 capacity_mem_par_l1_P41: mem[8,8] <= 1536
 row_capacity_l1_P42: rows_tot_l1_P42 <= 48
 capacity_mem_par_l1_P42: mem[9,14] <= 1536
 capacity_mem_par_l1_P42: mem[9,9] <= 1536
 capacity_mem_par_l1_P42: mem[9,7] <= 1536
 capacity_mem_par_l1_P42: mem[9,13] <= 1536
 capacity_mem_par_l1_P42: mem[9,8] <= 1536
 capacity_mem_par_l1_P42: mem[9,11] <= 1536
 capacity_mem_par_l1_P42: mem[9,6] <= 1536
 capacity_mem_par_l1_P42: mem[9,12] <= 1536
 capacity_mem_par_l1_P42: mem[9,15] <= 1536
 capacity_mem_par_l1_P42: mem[9,10] <= 1536
 capacity_mem_par_l1_P42: mem[9,5] <= 1536
 row_capacity_l2_P4: rows_tot_l2_P4 <= 48
 capacity_mem_par_l2_P4: mem[10,8] <= 1536
 capacity_mem_par_l2_P4: mem[10,7] <= 1536
 capacity_mem_par_l2_P4: mem[10,13] <= 1536
 capacity_mem_par_l2_P4: mem[10,14] <= 1536
 capacity_mem_par_l2_P4: mem[10,9] <= 1536
 capacity_mem_par_l2_P4: mem[10,10] <= 1536
 capacity_mem_par_l2_P4: mem[10,15] <= 1536
 capacity_mem_par_l2_P4: mem[10,11] <= 1536
 capacity_mem_par_l2_P4: mem[10,6] <= 1536
 capacity_mem_par_l2_P4: mem[10,12] <= 1536
Bounds
 frac[0,10] <= 1
 frac[0,4] <= 1
 frac[0,1] <= 1
 frac[0,2] <= 1
 frac[0,0] <= 1
 frac[1,6] <= 1
 frac[1,5] <= 1
 frac[1,4] <= 1
 frac[1,7] <= 1
 frac[1,13] <= 1
 frac[2,5] <= 1
 frac[2,11] <= 1
 frac[2,8] <= 1
 frac[2,9] <= 1
 frac[2,10] <= 1
 frac[3,12] <= 1
 frac[3,15] <= 1
 frac[3,14] <= 1
 frac[3,13] <= 1
 frac[3,8] <= 1
 mem_tot_{CPU15,CPU3,CPU25,CPU28,CPU30,CPU31,CPU13,CPU11} <= 262144
 normalized_mem_tot_{CPU15,CPU3,CPU25,CPU28,CPU30,CPU31,CPU13,CPU11} <= 1
 normalized_rows_tot_{CPU15,CPU3,CPU25,CPU28,CPU30,CPU31,CPU13,CPU11} <= 1
 mem_tot_{CPU5,CPU24,CPU16,CPU9,CPU26,CPU8,CPU10,CPU12} <= 262144
 normalized_mem_tot_{CPU5,CPU24,CPU16,CPU9,CPU26,CPU8,CPU10,CPU12} <= 1
 normalized_rows_tot_{CPU5,CPU24,CPU16,CPU9,CPU26,CPU8,CPU10,CPU12} <= 1
 mem_tot_{CPU7,CPU22,CPU14,CPU17,CPU19,CPU18,CPU2,CPU20} <= 262144
 normalized_mem_tot_{CPU7,CPU22,CPU14,CPU17,CPU19,CPU18,CPU2,CPU20} <= 1
 normalized_rows_tot_{CPU7,CPU22,CPU14,CPU17,CPU19,CPU18,CPU2,CPU20} <= 1
 mem_tot_{CPU29,CPU1,CPU4,CPU21,CPU23,CPU32,CPU27,CPU6} <= 262144
 normalized_mem_tot_{CPU29,CPU1,CPU4,CPU21,CPU23,CPU32,CPU27,CPU6} <= 1
 normalized_rows_tot_{CPU29,CPU1,CPU4,CPU21,CPU23,CPU32,CPU27,CPU6} <= 1
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
Binaries
 frac[4,0] frac[4,10] frac[4,15] frac[4,12] frac[4,11] frac[4,6] frac[4,8]
 frac[4,9] frac[4,14] frac[5,5] frac[5,4] frac[5,10] frac[5,7] frac[5,13]
 frac[5,0] frac[5,6] frac[6,9] frac[6,10] frac[6,13] frac[6,14] frac[6,11]
 frac[6,15] frac[6,5] frac[6,6] frac[6,12] frac[7,12] frac[7,5] frac[7,8]
 frac[7,2] frac[7,15] frac[7,1] frac[7,0] frac[7,13] frac[7,7] frac[8,15]
 frac[8,12] frac[8,9] frac[8,6] frac[8,13] frac[8,0] frac[8,10] frac[8,7]
 frac[8,14] frac[8,11] frac[8,8] frac[9,14] frac[9,9] frac[9,7] frac[9,13]
 frac[9,8] frac[9,11] frac[9,6] frac[9,12] frac[9,15] frac[9,10] frac[9,5]
 frac[10,8] frac[10,7] frac[10,13] frac[10,14] frac[10,9] frac[10,10]
 frac[10,15] frac[10,11] frac[10,6] frac[10,12]
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_tor_P41 , mem_tot_tor_P42 ,
   mem_tot_tor_P43 , mem_tot_tor_P44 , mem_tot_l1_P41 , mem_tot_l1_P42 ,
   mem_tot_l2_P4 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_{CPU15,CPU3,CPU25,CPU28,CPU30,CPU31,CPU13,CPU11} ,
   normalized_mem_tot_{CPU5,CPU24,CPU16,CPU9,CPU26,CPU8,CPU10,CPU12} ,
   normalized_mem_tot_{CPU7,CPU22,CPU14,CPU17,CPU19,CPU18,CPU2,CPU20} ,
   normalized_mem_tot_{CPU29,CPU1,CPU4,CPU21,CPU23,CPU32,CPU27,CPU6} )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_tor_P41 , rows_tot_tor_P42 ,
   rows_tot_tor_P43 , rows_tot_tor_P44 , rows_tot_l1_P41 ,
   rows_tot_l1_P42 , rows_tot_l2_P4 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX (
   normalized_rows_tot_{CPU15,CPU3,CPU25,CPU28,CPU30,CPU31,CPU13,CPU11} ,
   normalized_rows_tot_{CPU5,CPU24,CPU16,CPU9,CPU26,CPU8,CPU10,CPU12} ,
   normalized_rows_tot_{CPU7,CPU22,CPU14,CPU17,CPU19,CPU18,CPU2,CPU20} ,
   normalized_rows_tot_{CPU29,CPU1,CPU4,CPU21,CPU23,CPU32,CPU27,CPU6} )
End

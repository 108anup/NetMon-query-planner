\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU_Cluster: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_{CPU13,CPU4,CPU6,CPU30,CPU9,CPU16,CPU22,CPU32}
   + rows_tot_{CPU29,CPU8,CPU7,CPU21,CPU2,CPU3,CPU18,CPU5}
   + rows_tot_{CPU28,CPU19,CPU25,CPU1,CPU10,CPU17,CPU11,CPU14}
   + rows_tot_{CPU31,CPU12,CPU24,CPU27,CPU26,CPU23,CPU20,CPU15}
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
   mem_tot_{CPU13,CPU4,CPU6,CPU30,CPU9,CPU16,CPU22,CPU32}
   + mem_tot_{CPU29,CPU8,CPU7,CPU21,CPU2,CPU3,CPU18,CPU5}
   + mem_tot_{CPU28,CPU19,CPU25,CPU1,CPU10,CPU17,CPU11,CPU14}
   + mem_tot_{CPU31,CPU12,CPU24,CPU27,CPU26,CPU23,CPU20,CPU15}
  CPU_Cluster_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
  tot_mem_P4: Priority=40 Weight=1 AbsTol=0 RelTol=0
   mem_tot_tor_P41 + mem_tot_tor_P42 + mem_tot_tor_P43 + mem_tot_tor_P44
   + mem_tot_l1_P41 + mem_tot_l1_P42 + mem_tot_l2_P4
  P4_load_mem: Priority=30 Weight=1 AbsTol=0 RelTol=0
   max_mem_P4
Subject To
 cov_0_3: frac[7,3] + frac[9,3] + frac[10,3] + frac[8,3] + frac[5,3]
   + frac[0,3] >= 0.8
 cov_1_2: frac[8,2] + frac[7,2] + frac[9,2] + frac[0,2] + frac[5,2]
   + frac[10,2] >= 0.8
 cov_2_2: frac[8,2] + frac[7,2] + frac[9,2] + frac[0,2] + frac[5,2]
   + frac[10,2] >= 0.7
 cov_3_2: frac[3,2] + frac[8,2] + frac[9,2] + frac[0,2] + frac[6,2]
   + frac[5,2] + frac[10,2] >= 0.9
 cov_4_3: frac[8,3] + frac[5,3] + frac[0,3] + frac[4,3] >= 0.8
 cov_5_3: frac[9,3] + frac[10,3] + frac[8,3] + frac[0,3] + frac[3,3]
   + frac[6,3] + frac[4,3] >= 0.7
 cov_6_2: frac[0,2] + frac[5,2] >= 0.8
 cov_7_2: frac[8,2] + frac[7,2] + frac[9,2] + frac[0,2] + frac[5,2]
   + frac[10,2] >= 0.9
 cov_8_5: frac[8,5] + frac[4,5] + frac[1,5] + frac[6,5] + frac[10,5]
   + frac[9,5] >= 0.9
 cov_9_4: frac[9,4] + frac[10,4] + frac[7,4] + frac[1,4] + frac[8,4]
   + frac[4,4] >= 0.8
 cov_10_5: frac[4,5] + frac[1,5] >= 1
 cov_11_6: frac[1,6] + frac[8,6] + frac[4,6] + frac[9,6] + frac[6,6]
   + frac[10,6] >= 0.7
 cov_12_5: frac[4,5] + frac[1,5] >= 0.8
 cov_13_4: frac[9,4] + frac[6,4] + frac[7,4] + frac[1,4] + frac[3,4] >= 1
 cov_14_4: frac[1,4] + frac[4,4] >= 0.9
 cov_15_6: frac[1,6] + frac[4,6] >= 0.7
 cov_16_9: frac[6,9] + frac[9,9] + frac[2,9] + frac[7,9] >= 1
 cov_17_8: frac[10,8] + frac[4,8] + frac[2,8] + frac[9,8] + frac[6,8]
   + frac[8,8] >= 0.7
 cov_18_9: frac[8,9] + frac[5,9] + frac[4,9] + frac[2,9] >= 0.7
 cov_19_11: frac[5,11] + frac[2,11] + frac[6,11] + frac[9,11] + frac[10,11]
   + frac[8,11] >= 0.7
 cov_20_9: frac[6,9] + frac[9,9] + frac[8,9] + frac[10,9] + frac[5,9]
   + frac[2,9] >= 0.8
 cov_21_8: frac[5,8] + frac[10,8] + frac[7,8] + frac[2,8] + frac[9,8]
   + frac[8,8] >= 0.9
 cov_22_10: frac[7,10] + frac[2,10] >= 0.9
 cov_23_8: frac[5,8] + frac[4,8] + frac[2,8] + frac[3,8] + frac[8,8]
   >= 0.7
 cov_24_14: frac[9,14] + frac[10,14] + frac[0,14] + frac[6,14] + frac[3,14]
   + frac[8,14] + frac[4,14] >= 0.9
 cov_25_15: frac[7,15] + frac[3,15] + frac[6,15] + frac[9,15] >= 0.7
 cov_26_15: frac[8,15] + frac[7,15] + frac[3,15] + frac[10,15] + frac[9,15]
   + frac[5,15] >= 0.8
 cov_27_14: frac[9,14] + frac[5,14] + frac[10,14] + frac[6,14] + frac[3,14]
   + frac[8,14] >= 0.9
 cov_28_14: frac[9,14] + frac[6,14] + frac[7,14] + frac[3,14] >= 1
 cov_29_12: frac[3,12] + frac[8,12] + frac[9,12] + frac[5,12] + frac[10,12]
   + frac[6,12] >= 0.7
 cov_30_12: frac[7,12] + frac[3,12] + frac[8,12] + frac[9,12] + frac[5,12]
   + frac[10,12] >= 0.9
 cov_31_13: frac[6,13] + frac[9,13] + frac[7,13] + frac[3,13] >= 0.9
 cols_helper_7_3: cols_helper_7_3[0] + cols_helper_7_3[1]
   + cols_helper_7_3[2] + cols_helper_7_3[3] + cols_helper_7_3[4]
   + cols_helper_7_3[5] + cols_helper_7_3[6] + cols_helper_7_3[7]
   + cols_helper_7_3[8] + cols_helper_7_3[9] + cols_helper_7_3[10]
   + cols_helper_7_3[11] + cols_helper_7_3[12] + cols_helper_7_3[13]
   + cols_helper_7_3[14] + cols_helper_7_3[15] + cols_helper_7_3[16] <= 1
 cols_7_3: cols_7_3 - 2 cols_helper_7_3[0] - 4 cols_helper_7_3[1]
   - 8 cols_helper_7_3[2] - 16 cols_helper_7_3[3] - 32 cols_helper_7_3[4]
   - 64 cols_helper_7_3[5] - 128 cols_helper_7_3[6]
   - 256 cols_helper_7_3[7] - 512 cols_helper_7_3[8]
   - 1024 cols_helper_7_3[9] - 2048 cols_helper_7_3[10]
   - 4096 cols_helper_7_3[11] - 8192 cols_helper_7_3[12]
   - 16384 cols_helper_7_3[13] - 32768 cols_helper_7_3[14]
   - 65536 cols_helper_7_3[15] - 131072 cols_helper_7_3[16] = 0
 cols_acc_7_3: - 43493 frac[7,3] + cols_7_3 >= 0
 accuracy_7_3: mem[7,3] - 0.01171875 cols_7_3 = 0
 cols_helper_6_9: cols_helper_6_9[0] + cols_helper_6_9[1]
   + cols_helper_6_9[2] + cols_helper_6_9[3] + cols_helper_6_9[4]
   + cols_helper_6_9[5] + cols_helper_6_9[6] + cols_helper_6_9[7]
   + cols_helper_6_9[8] + cols_helper_6_9[9] + cols_helper_6_9[10]
   + cols_helper_6_9[11] + cols_helper_6_9[12] + cols_helper_6_9[13]
   + cols_helper_6_9[14] + cols_helper_6_9[15] + cols_helper_6_9[16] <= 1
 cols_6_9: cols_6_9 - 2 cols_helper_6_9[0] - 4 cols_helper_6_9[1]
   - 8 cols_helper_6_9[2] - 16 cols_helper_6_9[3] - 32 cols_helper_6_9[4]
   - 64 cols_helper_6_9[5] - 128 cols_helper_6_9[6]
   - 256 cols_helper_6_9[7] - 512 cols_helper_6_9[8]
   - 1024 cols_helper_6_9[9] - 2048 cols_helper_6_9[10]
   - 4096 cols_helper_6_9[11] - 8192 cols_helper_6_9[12]
   - 16384 cols_helper_6_9[13] - 32768 cols_helper_6_9[14]
   - 65536 cols_helper_6_9[15] - 131072 cols_helper_6_9[16] = 0
 cols_acc_6_9: - 43493 frac[6,9] + cols_6_9 >= 0
 accuracy_6_9: mem[6,9] - 0.01171875 cols_6_9 = 0
 cols_helper_7_12: cols_helper_7_12[0] + cols_helper_7_12[1]
   + cols_helper_7_12[2] + cols_helper_7_12[3] + cols_helper_7_12[4]
   + cols_helper_7_12[5] + cols_helper_7_12[6] + cols_helper_7_12[7]
   + cols_helper_7_12[8] + cols_helper_7_12[9] + cols_helper_7_12[10]
   + cols_helper_7_12[11] + cols_helper_7_12[12] + cols_helper_7_12[13]
   + cols_helper_7_12[14] + cols_helper_7_12[15] + cols_helper_7_12[16]
   <= 1
 cols_7_12: cols_7_12 - 2 cols_helper_7_12[0] - 4 cols_helper_7_12[1]
   - 8 cols_helper_7_12[2] - 16 cols_helper_7_12[3]
   - 32 cols_helper_7_12[4] - 64 cols_helper_7_12[5]
   - 128 cols_helper_7_12[6] - 256 cols_helper_7_12[7]
   - 512 cols_helper_7_12[8] - 1024 cols_helper_7_12[9]
   - 2048 cols_helper_7_12[10] - 4096 cols_helper_7_12[11]
   - 8192 cols_helper_7_12[12] - 16384 cols_helper_7_12[13]
   - 32768 cols_helper_7_12[14] - 65536 cols_helper_7_12[15]
   - 131072 cols_helper_7_12[16] = 0
 cols_acc_7_12: - 43493 frac[7,12] + cols_7_12 >= 0
 accuracy_7_12: mem[7,12] - 0.01171875 cols_7_12 = 0
 accuracy_1_6: - 509.68359375 frac[1,6] + mem[1,6] = 0
 cols_helper_8_5: cols_helper_8_5[0] + cols_helper_8_5[1]
   + cols_helper_8_5[2] + cols_helper_8_5[3] + cols_helper_8_5[4]
   + cols_helper_8_5[5] + cols_helper_8_5[6] + cols_helper_8_5[7]
   + cols_helper_8_5[8] + cols_helper_8_5[9] + cols_helper_8_5[10]
   + cols_helper_8_5[11] + cols_helper_8_5[12] + cols_helper_8_5[13]
   + cols_helper_8_5[14] + cols_helper_8_5[15] + cols_helper_8_5[16] <= 1
 cols_8_5: cols_8_5 - 2 cols_helper_8_5[0] - 4 cols_helper_8_5[1]
   - 8 cols_helper_8_5[2] - 16 cols_helper_8_5[3] - 32 cols_helper_8_5[4]
   - 64 cols_helper_8_5[5] - 128 cols_helper_8_5[6]
   - 256 cols_helper_8_5[7] - 512 cols_helper_8_5[8]
   - 1024 cols_helper_8_5[9] - 2048 cols_helper_8_5[10]
   - 4096 cols_helper_8_5[11] - 8192 cols_helper_8_5[12]
   - 16384 cols_helper_8_5[13] - 32768 cols_helper_8_5[14]
   - 65536 cols_helper_8_5[15] - 131072 cols_helper_8_5[16] = 0
 cols_acc_8_5: - 43493 frac[8,5] + cols_8_5 >= 0
 accuracy_8_5: mem[8,5] - 0.01171875 cols_8_5 = 0
 cols_helper_5_8: cols_helper_5_8[0] + cols_helper_5_8[1]
   + cols_helper_5_8[2] + cols_helper_5_8[3] + cols_helper_5_8[4]
   + cols_helper_5_8[5] + cols_helper_5_8[6] + cols_helper_5_8[7]
   + cols_helper_5_8[8] + cols_helper_5_8[9] + cols_helper_5_8[10]
   + cols_helper_5_8[11] + cols_helper_5_8[12] + cols_helper_5_8[13]
   + cols_helper_5_8[14] + cols_helper_5_8[15] + cols_helper_5_8[16] <= 1
 cols_5_8: cols_5_8 - 2 cols_helper_5_8[0] - 4 cols_helper_5_8[1]
   - 8 cols_helper_5_8[2] - 16 cols_helper_5_8[3] - 32 cols_helper_5_8[4]
   - 64 cols_helper_5_8[5] - 128 cols_helper_5_8[6]
   - 256 cols_helper_5_8[7] - 512 cols_helper_5_8[8]
   - 1024 cols_helper_5_8[9] - 2048 cols_helper_5_8[10]
   - 4096 cols_helper_5_8[11] - 8192 cols_helper_5_8[12]
   - 16384 cols_helper_5_8[13] - 32768 cols_helper_5_8[14]
   - 65536 cols_helper_5_8[15] - 131072 cols_helper_5_8[16] = 0
 cols_acc_5_8: - 43493 frac[5,8] + cols_5_8 >= 0
 accuracy_5_8: mem[5,8] - 0.01171875 cols_5_8 = 0
 cols_helper_10_8: cols_helper_10_8[0] + cols_helper_10_8[1]
   + cols_helper_10_8[2] + cols_helper_10_8[3] + cols_helper_10_8[4]
   + cols_helper_10_8[5] + cols_helper_10_8[6] + cols_helper_10_8[7]
   + cols_helper_10_8[8] + cols_helper_10_8[9] + cols_helper_10_8[10]
   + cols_helper_10_8[11] + cols_helper_10_8[12] + cols_helper_10_8[13]
   + cols_helper_10_8[14] + cols_helper_10_8[15] + cols_helper_10_8[16]
   <= 1
 cols_10_8: cols_10_8 - 2 cols_helper_10_8[0] - 4 cols_helper_10_8[1]
   - 8 cols_helper_10_8[2] - 16 cols_helper_10_8[3]
   - 32 cols_helper_10_8[4] - 64 cols_helper_10_8[5]
   - 128 cols_helper_10_8[6] - 256 cols_helper_10_8[7]
   - 512 cols_helper_10_8[8] - 1024 cols_helper_10_8[9]
   - 2048 cols_helper_10_8[10] - 4096 cols_helper_10_8[11]
   - 8192 cols_helper_10_8[12] - 16384 cols_helper_10_8[13]
   - 32768 cols_helper_10_8[14] - 65536 cols_helper_10_8[15]
   - 131072 cols_helper_10_8[16] = 0
 cols_acc_10_8: - 43493 frac[10,8] + cols_10_8 >= 0
 accuracy_10_8: mem[10,8] - 0.01171875 cols_10_8 = 0
 cols_helper_8_15: cols_helper_8_15[0] + cols_helper_8_15[1]
   + cols_helper_8_15[2] + cols_helper_8_15[3] + cols_helper_8_15[4]
   + cols_helper_8_15[5] + cols_helper_8_15[6] + cols_helper_8_15[7]
   + cols_helper_8_15[8] + cols_helper_8_15[9] + cols_helper_8_15[10]
   + cols_helper_8_15[11] + cols_helper_8_15[12] + cols_helper_8_15[13]
   + cols_helper_8_15[14] + cols_helper_8_15[15] + cols_helper_8_15[16]
   <= 1
 cols_8_15: cols_8_15 - 2 cols_helper_8_15[0] - 4 cols_helper_8_15[1]
   - 8 cols_helper_8_15[2] - 16 cols_helper_8_15[3]
   - 32 cols_helper_8_15[4] - 64 cols_helper_8_15[5]
   - 128 cols_helper_8_15[6] - 256 cols_helper_8_15[7]
   - 512 cols_helper_8_15[8] - 1024 cols_helper_8_15[9]
   - 2048 cols_helper_8_15[10] - 4096 cols_helper_8_15[11]
   - 8192 cols_helper_8_15[12] - 16384 cols_helper_8_15[13]
   - 32768 cols_helper_8_15[14] - 65536 cols_helper_8_15[15]
   - 131072 cols_helper_8_15[16] = 0
 cols_acc_8_15: - 43493 frac[8,15] + cols_8_15 >= 0
 accuracy_8_15: mem[8,15] - 0.01171875 cols_8_15 = 0
 accuracy_3_2: - 509.68359375 frac[3,2] + mem[3,2] = 0
 cols_helper_9_14: cols_helper_9_14[0] + cols_helper_9_14[1]
   + cols_helper_9_14[2] + cols_helper_9_14[3] + cols_helper_9_14[4]
   + cols_helper_9_14[5] + cols_helper_9_14[6] + cols_helper_9_14[7]
   + cols_helper_9_14[8] + cols_helper_9_14[9] + cols_helper_9_14[10]
   + cols_helper_9_14[11] + cols_helper_9_14[12] + cols_helper_9_14[13]
   + cols_helper_9_14[14] + cols_helper_9_14[15] + cols_helper_9_14[16]
   <= 1
 cols_9_14: cols_9_14 - 2 cols_helper_9_14[0] - 4 cols_helper_9_14[1]
   - 8 cols_helper_9_14[2] - 16 cols_helper_9_14[3]
   - 32 cols_helper_9_14[4] - 64 cols_helper_9_14[5]
   - 128 cols_helper_9_14[6] - 256 cols_helper_9_14[7]
   - 512 cols_helper_9_14[8] - 1024 cols_helper_9_14[9]
   - 2048 cols_helper_9_14[10] - 4096 cols_helper_9_14[11]
   - 8192 cols_helper_9_14[12] - 16384 cols_helper_9_14[13]
   - 32768 cols_helper_9_14[14] - 65536 cols_helper_9_14[15]
   - 131072 cols_helper_9_14[16] = 0
 cols_acc_9_14: - 43493 frac[9,14] + cols_9_14 >= 0
 accuracy_9_14: mem[9,14] - 0.01171875 cols_9_14 = 0
 cols_helper_8_2: cols_helper_8_2[0] + cols_helper_8_2[1]
   + cols_helper_8_2[2] + cols_helper_8_2[3] + cols_helper_8_2[4]
   + cols_helper_8_2[5] + cols_helper_8_2[6] + cols_helper_8_2[7]
   + cols_helper_8_2[8] + cols_helper_8_2[9] + cols_helper_8_2[10]
   + cols_helper_8_2[11] + cols_helper_8_2[12] + cols_helper_8_2[13]
   + cols_helper_8_2[14] + cols_helper_8_2[15] + cols_helper_8_2[16] <= 1
 cols_8_2: cols_8_2 - 2 cols_helper_8_2[0] - 4 cols_helper_8_2[1]
   - 8 cols_helper_8_2[2] - 16 cols_helper_8_2[3] - 32 cols_helper_8_2[4]
   - 64 cols_helper_8_2[5] - 128 cols_helper_8_2[6]
   - 256 cols_helper_8_2[7] - 512 cols_helper_8_2[8]
   - 1024 cols_helper_8_2[9] - 2048 cols_helper_8_2[10]
   - 4096 cols_helper_8_2[11] - 8192 cols_helper_8_2[12]
   - 16384 cols_helper_8_2[13] - 32768 cols_helper_8_2[14]
   - 65536 cols_helper_8_2[15] - 131072 cols_helper_8_2[16] = 0
 cols_acc_8_2: - 43493 frac[8,2] + cols_8_2 >= 0
 accuracy_8_2: mem[8,2] - 0.01171875 cols_8_2 = 0
 cols_helper_5_11: cols_helper_5_11[0] + cols_helper_5_11[1]
   + cols_helper_5_11[2] + cols_helper_5_11[3] + cols_helper_5_11[4]
   + cols_helper_5_11[5] + cols_helper_5_11[6] + cols_helper_5_11[7]
   + cols_helper_5_11[8] + cols_helper_5_11[9] + cols_helper_5_11[10]
   + cols_helper_5_11[11] + cols_helper_5_11[12] + cols_helper_5_11[13]
   + cols_helper_5_11[14] + cols_helper_5_11[15] + cols_helper_5_11[16]
   <= 1
 cols_5_11: cols_5_11 - 2 cols_helper_5_11[0] - 4 cols_helper_5_11[1]
   - 8 cols_helper_5_11[2] - 16 cols_helper_5_11[3]
   - 32 cols_helper_5_11[4] - 64 cols_helper_5_11[5]
   - 128 cols_helper_5_11[6] - 256 cols_helper_5_11[7]
   - 512 cols_helper_5_11[8] - 1024 cols_helper_5_11[9]
   - 2048 cols_helper_5_11[10] - 4096 cols_helper_5_11[11]
   - 8192 cols_helper_5_11[12] - 16384 cols_helper_5_11[13]
   - 32768 cols_helper_5_11[14] - 65536 cols_helper_5_11[15]
   - 131072 cols_helper_5_11[16] = 0
 cols_acc_5_11: - 43493 frac[5,11] + cols_5_11 >= 0
 accuracy_5_11: mem[5,11] - 0.01171875 cols_5_11 = 0
 cols_helper_4_5: cols_helper_4_5[0] + cols_helper_4_5[1]
   + cols_helper_4_5[2] + cols_helper_4_5[3] + cols_helper_4_5[4]
   + cols_helper_4_5[5] + cols_helper_4_5[6] + cols_helper_4_5[7]
   + cols_helper_4_5[8] + cols_helper_4_5[9] + cols_helper_4_5[10]
   + cols_helper_4_5[11] + cols_helper_4_5[12] + cols_helper_4_5[13]
   + cols_helper_4_5[14] + cols_helper_4_5[15] + cols_helper_4_5[16] <= 1
 cols_4_5: cols_4_5 - 2 cols_helper_4_5[0] - 4 cols_helper_4_5[1]
   - 8 cols_helper_4_5[2] - 16 cols_helper_4_5[3] - 32 cols_helper_4_5[4]
   - 64 cols_helper_4_5[5] - 128 cols_helper_4_5[6]
   - 256 cols_helper_4_5[7] - 512 cols_helper_4_5[8]
   - 1024 cols_helper_4_5[9] - 2048 cols_helper_4_5[10]
   - 4096 cols_helper_4_5[11] - 8192 cols_helper_4_5[12]
   - 16384 cols_helper_4_5[13] - 32768 cols_helper_4_5[14]
   - 65536 cols_helper_4_5[15] - 131072 cols_helper_4_5[16] = 0
 cols_acc_4_5: - 43493 frac[4,5] + cols_4_5 >= 0
 accuracy_4_5: mem[4,5] - 0.01171875 cols_4_5 = 0
 cols_helper_9_3: cols_helper_9_3[0] + cols_helper_9_3[1]
   + cols_helper_9_3[2] + cols_helper_9_3[3] + cols_helper_9_3[4]
   + cols_helper_9_3[5] + cols_helper_9_3[6] + cols_helper_9_3[7]
   + cols_helper_9_3[8] + cols_helper_9_3[9] + cols_helper_9_3[10]
   + cols_helper_9_3[11] + cols_helper_9_3[12] + cols_helper_9_3[13]
   + cols_helper_9_3[14] + cols_helper_9_3[15] + cols_helper_9_3[16] <= 1
 cols_9_3: cols_9_3 - 2 cols_helper_9_3[0] - 4 cols_helper_9_3[1]
   - 8 cols_helper_9_3[2] - 16 cols_helper_9_3[3] - 32 cols_helper_9_3[4]
   - 64 cols_helper_9_3[5] - 128 cols_helper_9_3[6]
   - 256 cols_helper_9_3[7] - 512 cols_helper_9_3[8]
   - 1024 cols_helper_9_3[9] - 2048 cols_helper_9_3[10]
   - 4096 cols_helper_9_3[11] - 8192 cols_helper_9_3[12]
   - 16384 cols_helper_9_3[13] - 32768 cols_helper_9_3[14]
   - 65536 cols_helper_9_3[15] - 131072 cols_helper_9_3[16] = 0
 cols_acc_9_3: - 43493 frac[9,3] + cols_9_3 >= 0
 accuracy_9_3: mem[9,3] - 0.01171875 cols_9_3 = 0
 accuracy_3_12: - 509.68359375 frac[3,12] + mem[3,12] = 0
 cols_helper_8_12: cols_helper_8_12[0] + cols_helper_8_12[1]
   + cols_helper_8_12[2] + cols_helper_8_12[3] + cols_helper_8_12[4]
   + cols_helper_8_12[5] + cols_helper_8_12[6] + cols_helper_8_12[7]
   + cols_helper_8_12[8] + cols_helper_8_12[9] + cols_helper_8_12[10]
   + cols_helper_8_12[11] + cols_helper_8_12[12] + cols_helper_8_12[13]
   + cols_helper_8_12[14] + cols_helper_8_12[15] + cols_helper_8_12[16]
   <= 1
 cols_8_12: cols_8_12 - 2 cols_helper_8_12[0] - 4 cols_helper_8_12[1]
   - 8 cols_helper_8_12[2] - 16 cols_helper_8_12[3]
   - 32 cols_helper_8_12[4] - 64 cols_helper_8_12[5]
   - 128 cols_helper_8_12[6] - 256 cols_helper_8_12[7]
   - 512 cols_helper_8_12[8] - 1024 cols_helper_8_12[9]
   - 2048 cols_helper_8_12[10] - 4096 cols_helper_8_12[11]
   - 8192 cols_helper_8_12[12] - 16384 cols_helper_8_12[13]
   - 32768 cols_helper_8_12[14] - 65536 cols_helper_8_12[15]
   - 131072 cols_helper_8_12[16] = 0
 cols_acc_8_12: - 43493 frac[8,12] + cols_8_12 >= 0
 accuracy_8_12: mem[8,12] - 0.01171875 cols_8_12 = 0
 accuracy_2_11: - 509.68359375 frac[2,11] + mem[2,11] = 0
 cols_helper_9_9: cols_helper_9_9[0] + cols_helper_9_9[1]
   + cols_helper_9_9[2] + cols_helper_9_9[3] + cols_helper_9_9[4]
   + cols_helper_9_9[5] + cols_helper_9_9[6] + cols_helper_9_9[7]
   + cols_helper_9_9[8] + cols_helper_9_9[9] + cols_helper_9_9[10]
   + cols_helper_9_9[11] + cols_helper_9_9[12] + cols_helper_9_9[13]
   + cols_helper_9_9[14] + cols_helper_9_9[15] + cols_helper_9_9[16] <= 1
 cols_9_9: cols_9_9 - 2 cols_helper_9_9[0] - 4 cols_helper_9_9[1]
   - 8 cols_helper_9_9[2] - 16 cols_helper_9_9[3] - 32 cols_helper_9_9[4]
   - 64 cols_helper_9_9[5] - 128 cols_helper_9_9[6]
   - 256 cols_helper_9_9[7] - 512 cols_helper_9_9[8]
   - 1024 cols_helper_9_9[9] - 2048 cols_helper_9_9[10]
   - 4096 cols_helper_9_9[11] - 8192 cols_helper_9_9[12]
   - 16384 cols_helper_9_9[13] - 32768 cols_helper_9_9[14]
   - 65536 cols_helper_9_9[15] - 131072 cols_helper_9_9[16] = 0
 cols_acc_9_9: - 43493 frac[9,9] + cols_9_9 >= 0
 accuracy_9_9: mem[9,9] - 0.01171875 cols_9_9 = 0
 cols_helper_5_14: cols_helper_5_14[0] + cols_helper_5_14[1]
   + cols_helper_5_14[2] + cols_helper_5_14[3] + cols_helper_5_14[4]
   + cols_helper_5_14[5] + cols_helper_5_14[6] + cols_helper_5_14[7]
   + cols_helper_5_14[8] + cols_helper_5_14[9] + cols_helper_5_14[10]
   + cols_helper_5_14[11] + cols_helper_5_14[12] + cols_helper_5_14[13]
   + cols_helper_5_14[14] + cols_helper_5_14[15] + cols_helper_5_14[16]
   <= 1
 cols_5_14: cols_5_14 - 2 cols_helper_5_14[0] - 4 cols_helper_5_14[1]
   - 8 cols_helper_5_14[2] - 16 cols_helper_5_14[3]
   - 32 cols_helper_5_14[4] - 64 cols_helper_5_14[5]
   - 128 cols_helper_5_14[6] - 256 cols_helper_5_14[7]
   - 512 cols_helper_5_14[8] - 1024 cols_helper_5_14[9]
   - 2048 cols_helper_5_14[10] - 4096 cols_helper_5_14[11]
   - 8192 cols_helper_5_14[12] - 16384 cols_helper_5_14[13]
   - 32768 cols_helper_5_14[14] - 65536 cols_helper_5_14[15]
   - 131072 cols_helper_5_14[16] = 0
 cols_acc_5_14: - 43493 frac[5,14] + cols_5_14 >= 0
 accuracy_5_14: mem[5,14] - 0.01171875 cols_5_14 = 0
 cols_helper_10_14: cols_helper_10_14[0] + cols_helper_10_14[1]
   + cols_helper_10_14[2] + cols_helper_10_14[3] + cols_helper_10_14[4]
   + cols_helper_10_14[5] + cols_helper_10_14[6] + cols_helper_10_14[7]
   + cols_helper_10_14[8] + cols_helper_10_14[9] + cols_helper_10_14[10]
   + cols_helper_10_14[11] + cols_helper_10_14[12] + cols_helper_10_14[13]
   + cols_helper_10_14[14] + cols_helper_10_14[15] + cols_helper_10_14[16]
   <= 1
 cols_10_14: cols_10_14 - 2 cols_helper_10_14[0] - 4 cols_helper_10_14[1]
   - 8 cols_helper_10_14[2] - 16 cols_helper_10_14[3]
   - 32 cols_helper_10_14[4] - 64 cols_helper_10_14[5]
   - 128 cols_helper_10_14[6] - 256 cols_helper_10_14[7]
   - 512 cols_helper_10_14[8] - 1024 cols_helper_10_14[9]
   - 2048 cols_helper_10_14[10] - 4096 cols_helper_10_14[11]
   - 8192 cols_helper_10_14[12] - 16384 cols_helper_10_14[13]
   - 32768 cols_helper_10_14[14] - 65536 cols_helper_10_14[15]
   - 131072 cols_helper_10_14[16] = 0
 cols_acc_10_14: - 43493 frac[10,14] + cols_10_14 >= 0
 accuracy_10_14: mem[10,14] - 0.01171875 cols_10_14 = 0
 cols_helper_6_13: cols_helper_6_13[0] + cols_helper_6_13[1]
   + cols_helper_6_13[2] + cols_helper_6_13[3] + cols_helper_6_13[4]
   + cols_helper_6_13[5] + cols_helper_6_13[6] + cols_helper_6_13[7]
   + cols_helper_6_13[8] + cols_helper_6_13[9] + cols_helper_6_13[10]
   + cols_helper_6_13[11] + cols_helper_6_13[12] + cols_helper_6_13[13]
   + cols_helper_6_13[14] + cols_helper_6_13[15] + cols_helper_6_13[16]
   <= 1
 cols_6_13: cols_6_13 - 2 cols_helper_6_13[0] - 4 cols_helper_6_13[1]
   - 8 cols_helper_6_13[2] - 16 cols_helper_6_13[3]
   - 32 cols_helper_6_13[4] - 64 cols_helper_6_13[5]
   - 128 cols_helper_6_13[6] - 256 cols_helper_6_13[7]
   - 512 cols_helper_6_13[8] - 1024 cols_helper_6_13[9]
   - 2048 cols_helper_6_13[10] - 4096 cols_helper_6_13[11]
   - 8192 cols_helper_6_13[12] - 16384 cols_helper_6_13[13]
   - 32768 cols_helper_6_13[14] - 65536 cols_helper_6_13[15]
   - 131072 cols_helper_6_13[16] = 0
 cols_acc_6_13: - 43493 frac[6,13] + cols_6_13 >= 0
 accuracy_6_13: mem[6,13] - 0.01171875 cols_6_13 = 0
 cols_helper_7_8: cols_helper_7_8[0] + cols_helper_7_8[1]
   + cols_helper_7_8[2] + cols_helper_7_8[3] + cols_helper_7_8[4]
   + cols_helper_7_8[5] + cols_helper_7_8[6] + cols_helper_7_8[7]
   + cols_helper_7_8[8] + cols_helper_7_8[9] + cols_helper_7_8[10]
   + cols_helper_7_8[11] + cols_helper_7_8[12] + cols_helper_7_8[13]
   + cols_helper_7_8[14] + cols_helper_7_8[15] + cols_helper_7_8[16] <= 1
 cols_7_8: cols_7_8 - 2 cols_helper_7_8[0] - 4 cols_helper_7_8[1]
   - 8 cols_helper_7_8[2] - 16 cols_helper_7_8[3] - 32 cols_helper_7_8[4]
   - 64 cols_helper_7_8[5] - 128 cols_helper_7_8[6]
   - 256 cols_helper_7_8[7] - 512 cols_helper_7_8[8]
   - 1024 cols_helper_7_8[9] - 2048 cols_helper_7_8[10]
   - 4096 cols_helper_7_8[11] - 8192 cols_helper_7_8[12]
   - 16384 cols_helper_7_8[13] - 32768 cols_helper_7_8[14]
   - 65536 cols_helper_7_8[15] - 131072 cols_helper_7_8[16] = 0
 cols_acc_7_8: - 43493 frac[7,8] + cols_7_8 >= 0
 accuracy_7_8: mem[7,8] - 0.01171875 cols_7_8 = 0
 accuracy_0_14: - 509.68359375 frac[0,14] + mem[0,14] = 0
 cols_helper_8_9: cols_helper_8_9[0] + cols_helper_8_9[1]
   + cols_helper_8_9[2] + cols_helper_8_9[3] + cols_helper_8_9[4]
   + cols_helper_8_9[5] + cols_helper_8_9[6] + cols_helper_8_9[7]
   + cols_helper_8_9[8] + cols_helper_8_9[9] + cols_helper_8_9[10]
   + cols_helper_8_9[11] + cols_helper_8_9[12] + cols_helper_8_9[13]
   + cols_helper_8_9[14] + cols_helper_8_9[15] + cols_helper_8_9[16] <= 1
 cols_8_9: cols_8_9 - 2 cols_helper_8_9[0] - 4 cols_helper_8_9[1]
   - 8 cols_helper_8_9[2] - 16 cols_helper_8_9[3] - 32 cols_helper_8_9[4]
   - 64 cols_helper_8_9[5] - 128 cols_helper_8_9[6]
   - 256 cols_helper_8_9[7] - 512 cols_helper_8_9[8]
   - 1024 cols_helper_8_9[9] - 2048 cols_helper_8_9[10]
   - 4096 cols_helper_8_9[11] - 8192 cols_helper_8_9[12]
   - 16384 cols_helper_8_9[13] - 32768 cols_helper_8_9[14]
   - 65536 cols_helper_8_9[15] - 131072 cols_helper_8_9[16] = 0
 cols_acc_8_9: - 43493 frac[8,9] + cols_8_9 >= 0
 accuracy_8_9: mem[8,9] - 0.01171875 cols_8_9 = 0
 cols_helper_9_4: cols_helper_9_4[0] + cols_helper_9_4[1]
   + cols_helper_9_4[2] + cols_helper_9_4[3] + cols_helper_9_4[4]
   + cols_helper_9_4[5] + cols_helper_9_4[6] + cols_helper_9_4[7]
   + cols_helper_9_4[8] + cols_helper_9_4[9] + cols_helper_9_4[10]
   + cols_helper_9_4[11] + cols_helper_9_4[12] + cols_helper_9_4[13]
   + cols_helper_9_4[14] + cols_helper_9_4[15] + cols_helper_9_4[16] <= 1
 cols_9_4: cols_9_4 - 2 cols_helper_9_4[0] - 4 cols_helper_9_4[1]
   - 8 cols_helper_9_4[2] - 16 cols_helper_9_4[3] - 32 cols_helper_9_4[4]
   - 64 cols_helper_9_4[5] - 128 cols_helper_9_4[6]
   - 256 cols_helper_9_4[7] - 512 cols_helper_9_4[8]
   - 1024 cols_helper_9_4[9] - 2048 cols_helper_9_4[10]
   - 4096 cols_helper_9_4[11] - 8192 cols_helper_9_4[12]
   - 16384 cols_helper_9_4[13] - 32768 cols_helper_9_4[14]
   - 65536 cols_helper_9_4[15] - 131072 cols_helper_9_4[16] = 0
 cols_acc_9_4: - 43493 frac[9,4] + cols_9_4 >= 0
 accuracy_9_4: mem[9,4] - 0.01171875 cols_9_4 = 0
 cols_helper_10_3: cols_helper_10_3[0] + cols_helper_10_3[1]
   + cols_helper_10_3[2] + cols_helper_10_3[3] + cols_helper_10_3[4]
   + cols_helper_10_3[5] + cols_helper_10_3[6] + cols_helper_10_3[7]
   + cols_helper_10_3[8] + cols_helper_10_3[9] + cols_helper_10_3[10]
   + cols_helper_10_3[11] + cols_helper_10_3[12] + cols_helper_10_3[13]
   + cols_helper_10_3[14] + cols_helper_10_3[15] + cols_helper_10_3[16]
   <= 1
 cols_10_3: cols_10_3 - 2 cols_helper_10_3[0] - 4 cols_helper_10_3[1]
   - 8 cols_helper_10_3[2] - 16 cols_helper_10_3[3]
   - 32 cols_helper_10_3[4] - 64 cols_helper_10_3[5]
   - 128 cols_helper_10_3[6] - 256 cols_helper_10_3[7]
   - 512 cols_helper_10_3[8] - 1024 cols_helper_10_3[9]
   - 2048 cols_helper_10_3[10] - 4096 cols_helper_10_3[11]
   - 8192 cols_helper_10_3[12] - 16384 cols_helper_10_3[13]
   - 32768 cols_helper_10_3[14] - 65536 cols_helper_10_3[15]
   - 131072 cols_helper_10_3[16] = 0
 cols_acc_10_3: - 43493 frac[10,3] + cols_10_3 >= 0
 accuracy_10_3: mem[10,3] - 0.01171875 cols_10_3 = 0
 cols_helper_7_2: cols_helper_7_2[0] + cols_helper_7_2[1]
   + cols_helper_7_2[2] + cols_helper_7_2[3] + cols_helper_7_2[4]
   + cols_helper_7_2[5] + cols_helper_7_2[6] + cols_helper_7_2[7]
   + cols_helper_7_2[8] + cols_helper_7_2[9] + cols_helper_7_2[10]
   + cols_helper_7_2[11] + cols_helper_7_2[12] + cols_helper_7_2[13]
   + cols_helper_7_2[14] + cols_helper_7_2[15] + cols_helper_7_2[16] <= 1
 cols_7_2: cols_7_2 - 2 cols_helper_7_2[0] - 4 cols_helper_7_2[1]
   - 8 cols_helper_7_2[2] - 16 cols_helper_7_2[3] - 32 cols_helper_7_2[4]
   - 64 cols_helper_7_2[5] - 128 cols_helper_7_2[6]
   - 256 cols_helper_7_2[7] - 512 cols_helper_7_2[8]
   - 1024 cols_helper_7_2[9] - 2048 cols_helper_7_2[10]
   - 4096 cols_helper_7_2[11] - 8192 cols_helper_7_2[12]
   - 16384 cols_helper_7_2[13] - 32768 cols_helper_7_2[14]
   - 65536 cols_helper_7_2[15] - 131072 cols_helper_7_2[16] = 0
 cols_acc_7_2: - 43493 frac[7,2] + cols_7_2 >= 0
 accuracy_7_2: mem[7,2] - 0.01171875 cols_7_2 = 0
 cols_helper_6_14: cols_helper_6_14[0] + cols_helper_6_14[1]
   + cols_helper_6_14[2] + cols_helper_6_14[3] + cols_helper_6_14[4]
   + cols_helper_6_14[5] + cols_helper_6_14[6] + cols_helper_6_14[7]
   + cols_helper_6_14[8] + cols_helper_6_14[9] + cols_helper_6_14[10]
   + cols_helper_6_14[11] + cols_helper_6_14[12] + cols_helper_6_14[13]
   + cols_helper_6_14[14] + cols_helper_6_14[15] + cols_helper_6_14[16]
   <= 1
 cols_6_14: cols_6_14 - 2 cols_helper_6_14[0] - 4 cols_helper_6_14[1]
   - 8 cols_helper_6_14[2] - 16 cols_helper_6_14[3]
   - 32 cols_helper_6_14[4] - 64 cols_helper_6_14[5]
   - 128 cols_helper_6_14[6] - 256 cols_helper_6_14[7]
   - 512 cols_helper_6_14[8] - 1024 cols_helper_6_14[9]
   - 2048 cols_helper_6_14[10] - 4096 cols_helper_6_14[11]
   - 8192 cols_helper_6_14[12] - 16384 cols_helper_6_14[13]
   - 32768 cols_helper_6_14[14] - 65536 cols_helper_6_14[15]
   - 131072 cols_helper_6_14[16] = 0
 cols_acc_6_14: - 43493 frac[6,14] + cols_6_14 >= 0
 accuracy_6_14: mem[6,14] - 0.01171875 cols_6_14 = 0
 cols_helper_7_15: cols_helper_7_15[0] + cols_helper_7_15[1]
   + cols_helper_7_15[2] + cols_helper_7_15[3] + cols_helper_7_15[4]
   + cols_helper_7_15[5] + cols_helper_7_15[6] + cols_helper_7_15[7]
   + cols_helper_7_15[8] + cols_helper_7_15[9] + cols_helper_7_15[10]
   + cols_helper_7_15[11] + cols_helper_7_15[12] + cols_helper_7_15[13]
   + cols_helper_7_15[14] + cols_helper_7_15[15] + cols_helper_7_15[16]
   <= 1
 cols_7_15: cols_7_15 - 2 cols_helper_7_15[0] - 4 cols_helper_7_15[1]
   - 8 cols_helper_7_15[2] - 16 cols_helper_7_15[3]
   - 32 cols_helper_7_15[4] - 64 cols_helper_7_15[5]
   - 128 cols_helper_7_15[6] - 256 cols_helper_7_15[7]
   - 512 cols_helper_7_15[8] - 1024 cols_helper_7_15[9]
   - 2048 cols_helper_7_15[10] - 4096 cols_helper_7_15[11]
   - 8192 cols_helper_7_15[12] - 16384 cols_helper_7_15[13]
   - 32768 cols_helper_7_15[14] - 65536 cols_helper_7_15[15]
   - 131072 cols_helper_7_15[16] = 0
 cols_acc_7_15: - 43493 frac[7,15] + cols_7_15 >= 0
 accuracy_7_15: mem[7,15] - 0.01171875 cols_7_15 = 0
 accuracy_1_5: - 509.68359375 frac[1,5] + mem[1,5] = 0
 cols_helper_8_6: cols_helper_8_6[0] + cols_helper_8_6[1]
   + cols_helper_8_6[2] + cols_helper_8_6[3] + cols_helper_8_6[4]
   + cols_helper_8_6[5] + cols_helper_8_6[6] + cols_helper_8_6[7]
   + cols_helper_8_6[8] + cols_helper_8_6[9] + cols_helper_8_6[10]
   + cols_helper_8_6[11] + cols_helper_8_6[12] + cols_helper_8_6[13]
   + cols_helper_8_6[14] + cols_helper_8_6[15] + cols_helper_8_6[16] <= 1
 cols_8_6: cols_8_6 - 2 cols_helper_8_6[0] - 4 cols_helper_8_6[1]
   - 8 cols_helper_8_6[2] - 16 cols_helper_8_6[3] - 32 cols_helper_8_6[4]
   - 64 cols_helper_8_6[5] - 128 cols_helper_8_6[6]
   - 256 cols_helper_8_6[7] - 512 cols_helper_8_6[8]
   - 1024 cols_helper_8_6[9] - 2048 cols_helper_8_6[10]
   - 4096 cols_helper_8_6[11] - 8192 cols_helper_8_6[12]
   - 16384 cols_helper_8_6[13] - 32768 cols_helper_8_6[14]
   - 65536 cols_helper_8_6[15] - 131072 cols_helper_8_6[16] = 0
 cols_acc_8_6: - 43493 frac[8,6] + cols_8_6 >= 0
 accuracy_8_6: mem[8,6] - 0.01171875 cols_8_6 = 0
 cols_helper_10_9: cols_helper_10_9[0] + cols_helper_10_9[1]
   + cols_helper_10_9[2] + cols_helper_10_9[3] + cols_helper_10_9[4]
   + cols_helper_10_9[5] + cols_helper_10_9[6] + cols_helper_10_9[7]
   + cols_helper_10_9[8] + cols_helper_10_9[9] + cols_helper_10_9[10]
   + cols_helper_10_9[11] + cols_helper_10_9[12] + cols_helper_10_9[13]
   + cols_helper_10_9[14] + cols_helper_10_9[15] + cols_helper_10_9[16]
   <= 1
 cols_10_9: cols_10_9 - 2 cols_helper_10_9[0] - 4 cols_helper_10_9[1]
   - 8 cols_helper_10_9[2] - 16 cols_helper_10_9[3]
   - 32 cols_helper_10_9[4] - 64 cols_helper_10_9[5]
   - 128 cols_helper_10_9[6] - 256 cols_helper_10_9[7]
   - 512 cols_helper_10_9[8] - 1024 cols_helper_10_9[9]
   - 2048 cols_helper_10_9[10] - 4096 cols_helper_10_9[11]
   - 8192 cols_helper_10_9[12] - 16384 cols_helper_10_9[13]
   - 32768 cols_helper_10_9[14] - 65536 cols_helper_10_9[15]
   - 131072 cols_helper_10_9[16] = 0
 cols_acc_10_9: - 43493 frac[10,9] + cols_10_9 >= 0
 accuracy_10_9: mem[10,9] - 0.01171875 cols_10_9 = 0
 cols_helper_6_4: cols_helper_6_4[0] + cols_helper_6_4[1]
   + cols_helper_6_4[2] + cols_helper_6_4[3] + cols_helper_6_4[4]
   + cols_helper_6_4[5] + cols_helper_6_4[6] + cols_helper_6_4[7]
   + cols_helper_6_4[8] + cols_helper_6_4[9] + cols_helper_6_4[10]
   + cols_helper_6_4[11] + cols_helper_6_4[12] + cols_helper_6_4[13]
   + cols_helper_6_4[14] + cols_helper_6_4[15] + cols_helper_6_4[16] <= 1
 cols_6_4: cols_6_4 - 2 cols_helper_6_4[0] - 4 cols_helper_6_4[1]
   - 8 cols_helper_6_4[2] - 16 cols_helper_6_4[3] - 32 cols_helper_6_4[4]
   - 64 cols_helper_6_4[5] - 128 cols_helper_6_4[6]
   - 256 cols_helper_6_4[7] - 512 cols_helper_6_4[8]
   - 1024 cols_helper_6_4[9] - 2048 cols_helper_6_4[10]
   - 4096 cols_helper_6_4[11] - 8192 cols_helper_6_4[12]
   - 16384 cols_helper_6_4[13] - 32768 cols_helper_6_4[14]
   - 65536 cols_helper_6_4[15] - 131072 cols_helper_6_4[16] = 0
 cols_acc_6_4: - 43493 frac[6,4] + cols_6_4 >= 0
 accuracy_6_4: mem[6,4] - 0.01171875 cols_6_4 = 0
 cols_helper_10_4: cols_helper_10_4[0] + cols_helper_10_4[1]
   + cols_helper_10_4[2] + cols_helper_10_4[3] + cols_helper_10_4[4]
   + cols_helper_10_4[5] + cols_helper_10_4[6] + cols_helper_10_4[7]
   + cols_helper_10_4[8] + cols_helper_10_4[9] + cols_helper_10_4[10]
   + cols_helper_10_4[11] + cols_helper_10_4[12] + cols_helper_10_4[13]
   + cols_helper_10_4[14] + cols_helper_10_4[15] + cols_helper_10_4[16]
   <= 1
 cols_10_4: cols_10_4 - 2 cols_helper_10_4[0] - 4 cols_helper_10_4[1]
   - 8 cols_helper_10_4[2] - 16 cols_helper_10_4[3]
   - 32 cols_helper_10_4[4] - 64 cols_helper_10_4[5]
   - 128 cols_helper_10_4[6] - 256 cols_helper_10_4[7]
   - 512 cols_helper_10_4[8] - 1024 cols_helper_10_4[9]
   - 2048 cols_helper_10_4[10] - 4096 cols_helper_10_4[11]
   - 8192 cols_helper_10_4[12] - 16384 cols_helper_10_4[13]
   - 32768 cols_helper_10_4[14] - 65536 cols_helper_10_4[15]
   - 131072 cols_helper_10_4[16] = 0
 cols_acc_10_4: - 43493 frac[10,4] + cols_10_4 >= 0
 accuracy_10_4: mem[10,4] - 0.01171875 cols_10_4 = 0
 cols_helper_6_11: cols_helper_6_11[0] + cols_helper_6_11[1]
   + cols_helper_6_11[2] + cols_helper_6_11[3] + cols_helper_6_11[4]
   + cols_helper_6_11[5] + cols_helper_6_11[6] + cols_helper_6_11[7]
   + cols_helper_6_11[8] + cols_helper_6_11[9] + cols_helper_6_11[10]
   + cols_helper_6_11[11] + cols_helper_6_11[12] + cols_helper_6_11[13]
   + cols_helper_6_11[14] + cols_helper_6_11[15] + cols_helper_6_11[16]
   <= 1
 cols_6_11: cols_6_11 - 2 cols_helper_6_11[0] - 4 cols_helper_6_11[1]
   - 8 cols_helper_6_11[2] - 16 cols_helper_6_11[3]
   - 32 cols_helper_6_11[4] - 64 cols_helper_6_11[5]
   - 128 cols_helper_6_11[6] - 256 cols_helper_6_11[7]
   - 512 cols_helper_6_11[8] - 1024 cols_helper_6_11[9]
   - 2048 cols_helper_6_11[10] - 4096 cols_helper_6_11[11]
   - 8192 cols_helper_6_11[12] - 16384 cols_helper_6_11[13]
   - 32768 cols_helper_6_11[14] - 65536 cols_helper_6_11[15]
   - 131072 cols_helper_6_11[16] = 0
 cols_acc_6_11: - 43493 frac[6,11] + cols_6_11 >= 0
 accuracy_6_11: mem[6,11] - 0.01171875 cols_6_11 = 0
 cols_helper_9_13: cols_helper_9_13[0] + cols_helper_9_13[1]
   + cols_helper_9_13[2] + cols_helper_9_13[3] + cols_helper_9_13[4]
   + cols_helper_9_13[5] + cols_helper_9_13[6] + cols_helper_9_13[7]
   + cols_helper_9_13[8] + cols_helper_9_13[9] + cols_helper_9_13[10]
   + cols_helper_9_13[11] + cols_helper_9_13[12] + cols_helper_9_13[13]
   + cols_helper_9_13[14] + cols_helper_9_13[15] + cols_helper_9_13[16]
   <= 1
 cols_9_13: cols_9_13 - 2 cols_helper_9_13[0] - 4 cols_helper_9_13[1]
   - 8 cols_helper_9_13[2] - 16 cols_helper_9_13[3]
   - 32 cols_helper_9_13[4] - 64 cols_helper_9_13[5]
   - 128 cols_helper_9_13[6] - 256 cols_helper_9_13[7]
   - 512 cols_helper_9_13[8] - 1024 cols_helper_9_13[9]
   - 2048 cols_helper_9_13[10] - 4096 cols_helper_9_13[11]
   - 8192 cols_helper_9_13[12] - 16384 cols_helper_9_13[13]
   - 32768 cols_helper_9_13[14] - 65536 cols_helper_9_13[15]
   - 131072 cols_helper_9_13[16] = 0
 cols_acc_9_13: - 43493 frac[9,13] + cols_9_13 >= 0
 accuracy_9_13: mem[9,13] - 0.01171875 cols_9_13 = 0
 cols_helper_8_3: cols_helper_8_3[0] + cols_helper_8_3[1]
   + cols_helper_8_3[2] + cols_helper_8_3[3] + cols_helper_8_3[4]
   + cols_helper_8_3[5] + cols_helper_8_3[6] + cols_helper_8_3[7]
   + cols_helper_8_3[8] + cols_helper_8_3[9] + cols_helper_8_3[10]
   + cols_helper_8_3[11] + cols_helper_8_3[12] + cols_helper_8_3[13]
   + cols_helper_8_3[14] + cols_helper_8_3[15] + cols_helper_8_3[16] <= 1
 cols_8_3: cols_8_3 - 2 cols_helper_8_3[0] - 4 cols_helper_8_3[1]
   - 8 cols_helper_8_3[2] - 16 cols_helper_8_3[3] - 32 cols_helper_8_3[4]
   - 64 cols_helper_8_3[5] - 128 cols_helper_8_3[6]
   - 256 cols_helper_8_3[7] - 512 cols_helper_8_3[8]
   - 1024 cols_helper_8_3[9] - 2048 cols_helper_8_3[10]
   - 4096 cols_helper_8_3[11] - 8192 cols_helper_8_3[12]
   - 16384 cols_helper_8_3[13] - 32768 cols_helper_8_3[14]
   - 65536 cols_helper_8_3[15] - 131072 cols_helper_8_3[16] = 0
 cols_acc_8_3: - 43493 frac[8,3] + cols_8_3 >= 0
 accuracy_8_3: mem[8,3] - 0.01171875 cols_8_3 = 0
 cols_helper_4_6: cols_helper_4_6[0] + cols_helper_4_6[1]
   + cols_helper_4_6[2] + cols_helper_4_6[3] + cols_helper_4_6[4]
   + cols_helper_4_6[5] + cols_helper_4_6[6] + cols_helper_4_6[7]
   + cols_helper_4_6[8] + cols_helper_4_6[9] + cols_helper_4_6[10]
   + cols_helper_4_6[11] + cols_helper_4_6[12] + cols_helper_4_6[13]
   + cols_helper_4_6[14] + cols_helper_4_6[15] + cols_helper_4_6[16] <= 1
 cols_4_6: cols_4_6 - 2 cols_helper_4_6[0] - 4 cols_helper_4_6[1]
   - 8 cols_helper_4_6[2] - 16 cols_helper_4_6[3] - 32 cols_helper_4_6[4]
   - 64 cols_helper_4_6[5] - 128 cols_helper_4_6[6]
   - 256 cols_helper_4_6[7] - 512 cols_helper_4_6[8]
   - 1024 cols_helper_4_6[9] - 2048 cols_helper_4_6[10]
   - 4096 cols_helper_4_6[11] - 8192 cols_helper_4_6[12]
   - 16384 cols_helper_4_6[13] - 32768 cols_helper_4_6[14]
   - 65536 cols_helper_4_6[15] - 131072 cols_helper_4_6[16] = 0
 cols_acc_4_6: - 43493 frac[4,6] + cols_4_6 >= 0
 accuracy_4_6: mem[4,6] - 0.01171875 cols_4_6 = 0
 cols_helper_9_2: cols_helper_9_2[0] + cols_helper_9_2[1]
   + cols_helper_9_2[2] + cols_helper_9_2[3] + cols_helper_9_2[4]
   + cols_helper_9_2[5] + cols_helper_9_2[6] + cols_helper_9_2[7]
   + cols_helper_9_2[8] + cols_helper_9_2[9] + cols_helper_9_2[10]
   + cols_helper_9_2[11] + cols_helper_9_2[12] + cols_helper_9_2[13]
   + cols_helper_9_2[14] + cols_helper_9_2[15] + cols_helper_9_2[16] <= 1
 cols_9_2: cols_9_2 - 2 cols_helper_9_2[0] - 4 cols_helper_9_2[1]
   - 8 cols_helper_9_2[2] - 16 cols_helper_9_2[3] - 32 cols_helper_9_2[4]
   - 64 cols_helper_9_2[5] - 128 cols_helper_9_2[6]
   - 256 cols_helper_9_2[7] - 512 cols_helper_9_2[8]
   - 1024 cols_helper_9_2[9] - 2048 cols_helper_9_2[10]
   - 4096 cols_helper_9_2[11] - 8192 cols_helper_9_2[12]
   - 16384 cols_helper_9_2[13] - 32768 cols_helper_9_2[14]
   - 65536 cols_helper_9_2[15] - 131072 cols_helper_9_2[16] = 0
 cols_acc_9_2: - 43493 frac[9,2] + cols_9_2 >= 0
 accuracy_9_2: mem[9,2] - 0.01171875 cols_9_2 = 0
 cols_helper_7_4: cols_helper_7_4[0] + cols_helper_7_4[1]
   + cols_helper_7_4[2] + cols_helper_7_4[3] + cols_helper_7_4[4]
   + cols_helper_7_4[5] + cols_helper_7_4[6] + cols_helper_7_4[7]
   + cols_helper_7_4[8] + cols_helper_7_4[9] + cols_helper_7_4[10]
   + cols_helper_7_4[11] + cols_helper_7_4[12] + cols_helper_7_4[13]
   + cols_helper_7_4[14] + cols_helper_7_4[15] + cols_helper_7_4[16] <= 1
 cols_7_4: cols_7_4 - 2 cols_helper_7_4[0] - 4 cols_helper_7_4[1]
   - 8 cols_helper_7_4[2] - 16 cols_helper_7_4[3] - 32 cols_helper_7_4[4]
   - 64 cols_helper_7_4[5] - 128 cols_helper_7_4[6]
   - 256 cols_helper_7_4[7] - 512 cols_helper_7_4[8]
   - 1024 cols_helper_7_4[9] - 2048 cols_helper_7_4[10]
   - 4096 cols_helper_7_4[11] - 8192 cols_helper_7_4[12]
   - 16384 cols_helper_7_4[13] - 32768 cols_helper_7_4[14]
   - 65536 cols_helper_7_4[15] - 131072 cols_helper_7_4[16] = 0
 cols_acc_7_4: - 43493 frac[7,4] + cols_7_4 >= 0
 accuracy_7_4: mem[7,4] - 0.01171875 cols_7_4 = 0
 accuracy_0_2: - 509.68359375 frac[0,2] + mem[0,2] = 0
 accuracy_3_15: - 509.68359375 frac[3,15] + mem[3,15] = 0
 cols_helper_4_8: cols_helper_4_8[0] + cols_helper_4_8[1]
   + cols_helper_4_8[2] + cols_helper_4_8[3] + cols_helper_4_8[4]
   + cols_helper_4_8[5] + cols_helper_4_8[6] + cols_helper_4_8[7]
   + cols_helper_4_8[8] + cols_helper_4_8[9] + cols_helper_4_8[10]
   + cols_helper_4_8[11] + cols_helper_4_8[12] + cols_helper_4_8[13]
   + cols_helper_4_8[14] + cols_helper_4_8[15] + cols_helper_4_8[16] <= 1
 cols_4_8: cols_4_8 - 2 cols_helper_4_8[0] - 4 cols_helper_4_8[1]
   - 8 cols_helper_4_8[2] - 16 cols_helper_4_8[3] - 32 cols_helper_4_8[4]
   - 64 cols_helper_4_8[5] - 128 cols_helper_4_8[6]
   - 256 cols_helper_4_8[7] - 512 cols_helper_4_8[8]
   - 1024 cols_helper_4_8[9] - 2048 cols_helper_4_8[10]
   - 4096 cols_helper_4_8[11] - 8192 cols_helper_4_8[12]
   - 16384 cols_helper_4_8[13] - 32768 cols_helper_4_8[14]
   - 65536 cols_helper_4_8[15] - 131072 cols_helper_4_8[16] = 0
 cols_acc_4_8: - 43493 frac[4,8] + cols_4_8 >= 0
 accuracy_4_8: mem[4,8] - 0.01171875 cols_4_8 = 0
 accuracy_2_8: - 509.68359375 frac[2,8] + mem[2,8] = 0
 cols_helper_9_8: cols_helper_9_8[0] + cols_helper_9_8[1]
   + cols_helper_9_8[2] + cols_helper_9_8[3] + cols_helper_9_8[4]
   + cols_helper_9_8[5] + cols_helper_9_8[6] + cols_helper_9_8[7]
   + cols_helper_9_8[8] + cols_helper_9_8[9] + cols_helper_9_8[10]
   + cols_helper_9_8[11] + cols_helper_9_8[12] + cols_helper_9_8[13]
   + cols_helper_9_8[14] + cols_helper_9_8[15] + cols_helper_9_8[16] <= 1
 cols_9_8: cols_9_8 - 2 cols_helper_9_8[0] - 4 cols_helper_9_8[1]
   - 8 cols_helper_9_8[2] - 16 cols_helper_9_8[3] - 32 cols_helper_9_8[4]
   - 64 cols_helper_9_8[5] - 128 cols_helper_9_8[6]
   - 256 cols_helper_9_8[7] - 512 cols_helper_9_8[8]
   - 1024 cols_helper_9_8[9] - 2048 cols_helper_9_8[10]
   - 4096 cols_helper_9_8[11] - 8192 cols_helper_9_8[12]
   - 16384 cols_helper_9_8[13] - 32768 cols_helper_9_8[14]
   - 65536 cols_helper_9_8[15] - 131072 cols_helper_9_8[16] = 0
 cols_acc_9_8: - 43493 frac[9,8] + cols_9_8 >= 0
 accuracy_9_8: mem[9,8] - 0.01171875 cols_9_8 = 0
 cols_helper_10_15: cols_helper_10_15[0] + cols_helper_10_15[1]
   + cols_helper_10_15[2] + cols_helper_10_15[3] + cols_helper_10_15[4]
   + cols_helper_10_15[5] + cols_helper_10_15[6] + cols_helper_10_15[7]
   + cols_helper_10_15[8] + cols_helper_10_15[9] + cols_helper_10_15[10]
   + cols_helper_10_15[11] + cols_helper_10_15[12] + cols_helper_10_15[13]
   + cols_helper_10_15[14] + cols_helper_10_15[15] + cols_helper_10_15[16]
   <= 1
 cols_10_15: cols_10_15 - 2 cols_helper_10_15[0] - 4 cols_helper_10_15[1]
   - 8 cols_helper_10_15[2] - 16 cols_helper_10_15[3]
   - 32 cols_helper_10_15[4] - 64 cols_helper_10_15[5]
   - 128 cols_helper_10_15[6] - 256 cols_helper_10_15[7]
   - 512 cols_helper_10_15[8] - 1024 cols_helper_10_15[9]
   - 2048 cols_helper_10_15[10] - 4096 cols_helper_10_15[11]
   - 8192 cols_helper_10_15[12] - 16384 cols_helper_10_15[13]
   - 32768 cols_helper_10_15[14] - 65536 cols_helper_10_15[15]
   - 131072 cols_helper_10_15[16] = 0
 cols_acc_10_15: - 43493 frac[10,15] + cols_10_15 >= 0
 accuracy_10_15: mem[10,15] - 0.01171875 cols_10_15 = 0
 cols_helper_6_2: cols_helper_6_2[0] + cols_helper_6_2[1]
   + cols_helper_6_2[2] + cols_helper_6_2[3] + cols_helper_6_2[4]
   + cols_helper_6_2[5] + cols_helper_6_2[6] + cols_helper_6_2[7]
   + cols_helper_6_2[8] + cols_helper_6_2[9] + cols_helper_6_2[10]
   + cols_helper_6_2[11] + cols_helper_6_2[12] + cols_helper_6_2[13]
   + cols_helper_6_2[14] + cols_helper_6_2[15] + cols_helper_6_2[16] <= 1
 cols_6_2: cols_6_2 - 2 cols_helper_6_2[0] - 4 cols_helper_6_2[1]
   - 8 cols_helper_6_2[2] - 16 cols_helper_6_2[3] - 32 cols_helper_6_2[4]
   - 64 cols_helper_6_2[5] - 128 cols_helper_6_2[6]
   - 256 cols_helper_6_2[7] - 512 cols_helper_6_2[8]
   - 1024 cols_helper_6_2[9] - 2048 cols_helper_6_2[10]
   - 4096 cols_helper_6_2[11] - 8192 cols_helper_6_2[12]
   - 16384 cols_helper_6_2[13] - 32768 cols_helper_6_2[14]
   - 65536 cols_helper_6_2[15] - 131072 cols_helper_6_2[16] = 0
 cols_acc_6_2: - 43493 frac[6,2] + cols_6_2 >= 0
 accuracy_6_2: mem[6,2] - 0.01171875 cols_6_2 = 0
 cols_helper_9_11: cols_helper_9_11[0] + cols_helper_9_11[1]
   + cols_helper_9_11[2] + cols_helper_9_11[3] + cols_helper_9_11[4]
   + cols_helper_9_11[5] + cols_helper_9_11[6] + cols_helper_9_11[7]
   + cols_helper_9_11[8] + cols_helper_9_11[9] + cols_helper_9_11[10]
   + cols_helper_9_11[11] + cols_helper_9_11[12] + cols_helper_9_11[13]
   + cols_helper_9_11[14] + cols_helper_9_11[15] + cols_helper_9_11[16]
   <= 1
 cols_9_11: cols_9_11 - 2 cols_helper_9_11[0] - 4 cols_helper_9_11[1]
   - 8 cols_helper_9_11[2] - 16 cols_helper_9_11[3]
   - 32 cols_helper_9_11[4] - 64 cols_helper_9_11[5]
   - 128 cols_helper_9_11[6] - 256 cols_helper_9_11[7]
   - 512 cols_helper_9_11[8] - 1024 cols_helper_9_11[9]
   - 2048 cols_helper_9_11[10] - 4096 cols_helper_9_11[11]
   - 8192 cols_helper_9_11[12] - 16384 cols_helper_9_11[13]
   - 32768 cols_helper_9_11[14] - 65536 cols_helper_9_11[15]
   - 131072 cols_helper_9_11[16] = 0
 cols_acc_9_11: - 43493 frac[9,11] + cols_9_11 >= 0
 accuracy_9_11: mem[9,11] - 0.01171875 cols_9_11 = 0
 cols_helper_6_15: cols_helper_6_15[0] + cols_helper_6_15[1]
   + cols_helper_6_15[2] + cols_helper_6_15[3] + cols_helper_6_15[4]
   + cols_helper_6_15[5] + cols_helper_6_15[6] + cols_helper_6_15[7]
   + cols_helper_6_15[8] + cols_helper_6_15[9] + cols_helper_6_15[10]
   + cols_helper_6_15[11] + cols_helper_6_15[12] + cols_helper_6_15[13]
   + cols_helper_6_15[14] + cols_helper_6_15[15] + cols_helper_6_15[16]
   <= 1
 cols_6_15: cols_6_15 - 2 cols_helper_6_15[0] - 4 cols_helper_6_15[1]
   - 8 cols_helper_6_15[2] - 16 cols_helper_6_15[3]
   - 32 cols_helper_6_15[4] - 64 cols_helper_6_15[5]
   - 128 cols_helper_6_15[6] - 256 cols_helper_6_15[7]
   - 512 cols_helper_6_15[8] - 1024 cols_helper_6_15[9]
   - 2048 cols_helper_6_15[10] - 4096 cols_helper_6_15[11]
   - 8192 cols_helper_6_15[12] - 16384 cols_helper_6_15[13]
   - 32768 cols_helper_6_15[14] - 65536 cols_helper_6_15[15]
   - 131072 cols_helper_6_15[16] = 0
 cols_acc_6_15: - 43493 frac[6,15] + cols_6_15 >= 0
 accuracy_6_15: mem[6,15] - 0.01171875 cols_6_15 = 0
 cols_helper_7_14: cols_helper_7_14[0] + cols_helper_7_14[1]
   + cols_helper_7_14[2] + cols_helper_7_14[3] + cols_helper_7_14[4]
   + cols_helper_7_14[5] + cols_helper_7_14[6] + cols_helper_7_14[7]
   + cols_helper_7_14[8] + cols_helper_7_14[9] + cols_helper_7_14[10]
   + cols_helper_7_14[11] + cols_helper_7_14[12] + cols_helper_7_14[13]
   + cols_helper_7_14[14] + cols_helper_7_14[15] + cols_helper_7_14[16]
   <= 1
 cols_7_14: cols_7_14 - 2 cols_helper_7_14[0] - 4 cols_helper_7_14[1]
   - 8 cols_helper_7_14[2] - 16 cols_helper_7_14[3]
   - 32 cols_helper_7_14[4] - 64 cols_helper_7_14[5]
   - 128 cols_helper_7_14[6] - 256 cols_helper_7_14[7]
   - 512 cols_helper_7_14[8] - 1024 cols_helper_7_14[9]
   - 2048 cols_helper_7_14[10] - 4096 cols_helper_7_14[11]
   - 8192 cols_helper_7_14[12] - 16384 cols_helper_7_14[13]
   - 32768 cols_helper_7_14[14] - 65536 cols_helper_7_14[15]
   - 131072 cols_helper_7_14[16] = 0
 cols_acc_7_14: - 43493 frac[7,14] + cols_7_14 >= 0
 accuracy_7_14: mem[7,14] - 0.01171875 cols_7_14 = 0
 accuracy_1_4: - 509.68359375 frac[1,4] + mem[1,4] = 0
 cols_helper_9_6: cols_helper_9_6[0] + cols_helper_9_6[1]
   + cols_helper_9_6[2] + cols_helper_9_6[3] + cols_helper_9_6[4]
   + cols_helper_9_6[5] + cols_helper_9_6[6] + cols_helper_9_6[7]
   + cols_helper_9_6[8] + cols_helper_9_6[9] + cols_helper_9_6[10]
   + cols_helper_9_6[11] + cols_helper_9_6[12] + cols_helper_9_6[13]
   + cols_helper_9_6[14] + cols_helper_9_6[15] + cols_helper_9_6[16] <= 1
 cols_9_6: cols_9_6 - 2 cols_helper_9_6[0] - 4 cols_helper_9_6[1]
   - 8 cols_helper_9_6[2] - 16 cols_helper_9_6[3] - 32 cols_helper_9_6[4]
   - 64 cols_helper_9_6[5] - 128 cols_helper_9_6[6]
   - 256 cols_helper_9_6[7] - 512 cols_helper_9_6[8]
   - 1024 cols_helper_9_6[9] - 2048 cols_helper_9_6[10]
   - 4096 cols_helper_9_6[11] - 8192 cols_helper_9_6[12]
   - 16384 cols_helper_9_6[13] - 32768 cols_helper_9_6[14]
   - 65536 cols_helper_9_6[15] - 131072 cols_helper_9_6[16] = 0
 cols_acc_9_6: - 43493 frac[9,6] + cols_9_6 >= 0
 accuracy_9_6: mem[9,6] - 0.01171875 cols_9_6 = 0
 cols_helper_6_5: cols_helper_6_5[0] + cols_helper_6_5[1]
   + cols_helper_6_5[2] + cols_helper_6_5[3] + cols_helper_6_5[4]
   + cols_helper_6_5[5] + cols_helper_6_5[6] + cols_helper_6_5[7]
   + cols_helper_6_5[8] + cols_helper_6_5[9] + cols_helper_6_5[10]
   + cols_helper_6_5[11] + cols_helper_6_5[12] + cols_helper_6_5[13]
   + cols_helper_6_5[14] + cols_helper_6_5[15] + cols_helper_6_5[16] <= 1
 cols_6_5: cols_6_5 - 2 cols_helper_6_5[0] - 4 cols_helper_6_5[1]
   - 8 cols_helper_6_5[2] - 16 cols_helper_6_5[3] - 32 cols_helper_6_5[4]
   - 64 cols_helper_6_5[5] - 128 cols_helper_6_5[6]
   - 256 cols_helper_6_5[7] - 512 cols_helper_6_5[8]
   - 1024 cols_helper_6_5[9] - 2048 cols_helper_6_5[10]
   - 4096 cols_helper_6_5[11] - 8192 cols_helper_6_5[12]
   - 16384 cols_helper_6_5[13] - 32768 cols_helper_6_5[14]
   - 65536 cols_helper_6_5[15] - 131072 cols_helper_6_5[16] = 0
 cols_acc_6_5: - 43493 frac[6,5] + cols_6_5 >= 0
 accuracy_6_5: mem[6,5] - 0.01171875 cols_6_5 = 0
 cols_helper_5_3: cols_helper_5_3[0] + cols_helper_5_3[1]
   + cols_helper_5_3[2] + cols_helper_5_3[3] + cols_helper_5_3[4]
   + cols_helper_5_3[5] + cols_helper_5_3[6] + cols_helper_5_3[7]
   + cols_helper_5_3[8] + cols_helper_5_3[9] + cols_helper_5_3[10]
   + cols_helper_5_3[11] + cols_helper_5_3[12] + cols_helper_5_3[13]
   + cols_helper_5_3[14] + cols_helper_5_3[15] + cols_helper_5_3[16] <= 1
 cols_5_3: cols_5_3 - 2 cols_helper_5_3[0] - 4 cols_helper_5_3[1]
   - 8 cols_helper_5_3[2] - 16 cols_helper_5_3[3] - 32 cols_helper_5_3[4]
   - 64 cols_helper_5_3[5] - 128 cols_helper_5_3[6]
   - 256 cols_helper_5_3[7] - 512 cols_helper_5_3[8]
   - 1024 cols_helper_5_3[9] - 2048 cols_helper_5_3[10]
   - 4096 cols_helper_5_3[11] - 8192 cols_helper_5_3[12]
   - 16384 cols_helper_5_3[13] - 32768 cols_helper_5_3[14]
   - 65536 cols_helper_5_3[15] - 131072 cols_helper_5_3[16] = 0
 cols_acc_5_3: - 43493 frac[5,3] + cols_5_3 >= 0
 accuracy_5_3: mem[5,3] - 0.01171875 cols_5_3 = 0
 cols_helper_10_5: cols_helper_10_5[0] + cols_helper_10_5[1]
   + cols_helper_10_5[2] + cols_helper_10_5[3] + cols_helper_10_5[4]
   + cols_helper_10_5[5] + cols_helper_10_5[6] + cols_helper_10_5[7]
   + cols_helper_10_5[8] + cols_helper_10_5[9] + cols_helper_10_5[10]
   + cols_helper_10_5[11] + cols_helper_10_5[12] + cols_helper_10_5[13]
   + cols_helper_10_5[14] + cols_helper_10_5[15] + cols_helper_10_5[16]
   <= 1
 cols_10_5: cols_10_5 - 2 cols_helper_10_5[0] - 4 cols_helper_10_5[1]
   - 8 cols_helper_10_5[2] - 16 cols_helper_10_5[3]
   - 32 cols_helper_10_5[4] - 64 cols_helper_10_5[5]
   - 128 cols_helper_10_5[6] - 256 cols_helper_10_5[7]
   - 512 cols_helper_10_5[8] - 1024 cols_helper_10_5[9]
   - 2048 cols_helper_10_5[10] - 4096 cols_helper_10_5[11]
   - 8192 cols_helper_10_5[12] - 16384 cols_helper_10_5[13]
   - 32768 cols_helper_10_5[14] - 65536 cols_helper_10_5[15]
   - 131072 cols_helper_10_5[16] = 0
 cols_acc_10_5: - 43493 frac[10,5] + cols_10_5 >= 0
 accuracy_10_5: mem[10,5] - 0.01171875 cols_10_5 = 0
 cols_helper_6_8: cols_helper_6_8[0] + cols_helper_6_8[1]
   + cols_helper_6_8[2] + cols_helper_6_8[3] + cols_helper_6_8[4]
   + cols_helper_6_8[5] + cols_helper_6_8[6] + cols_helper_6_8[7]
   + cols_helper_6_8[8] + cols_helper_6_8[9] + cols_helper_6_8[10]
   + cols_helper_6_8[11] + cols_helper_6_8[12] + cols_helper_6_8[13]
   + cols_helper_6_8[14] + cols_helper_6_8[15] + cols_helper_6_8[16] <= 1
 cols_6_8: cols_6_8 - 2 cols_helper_6_8[0] - 4 cols_helper_6_8[1]
   - 8 cols_helper_6_8[2] - 16 cols_helper_6_8[3] - 32 cols_helper_6_8[4]
   - 64 cols_helper_6_8[5] - 128 cols_helper_6_8[6]
   - 256 cols_helper_6_8[7] - 512 cols_helper_6_8[8]
   - 1024 cols_helper_6_8[9] - 2048 cols_helper_6_8[10]
   - 4096 cols_helper_6_8[11] - 8192 cols_helper_6_8[12]
   - 16384 cols_helper_6_8[13] - 32768 cols_helper_6_8[14]
   - 65536 cols_helper_6_8[15] - 131072 cols_helper_6_8[16] = 0
 cols_acc_6_8: - 43493 frac[6,8] + cols_6_8 >= 0
 accuracy_6_8: mem[6,8] - 0.01171875 cols_6_8 = 0
 cols_helper_7_13: cols_helper_7_13[0] + cols_helper_7_13[1]
   + cols_helper_7_13[2] + cols_helper_7_13[3] + cols_helper_7_13[4]
   + cols_helper_7_13[5] + cols_helper_7_13[6] + cols_helper_7_13[7]
   + cols_helper_7_13[8] + cols_helper_7_13[9] + cols_helper_7_13[10]
   + cols_helper_7_13[11] + cols_helper_7_13[12] + cols_helper_7_13[13]
   + cols_helper_7_13[14] + cols_helper_7_13[15] + cols_helper_7_13[16]
   <= 1
 cols_7_13: cols_7_13 - 2 cols_helper_7_13[0] - 4 cols_helper_7_13[1]
   - 8 cols_helper_7_13[2] - 16 cols_helper_7_13[3]
   - 32 cols_helper_7_13[4] - 64 cols_helper_7_13[5]
   - 128 cols_helper_7_13[6] - 256 cols_helper_7_13[7]
   - 512 cols_helper_7_13[8] - 1024 cols_helper_7_13[9]
   - 2048 cols_helper_7_13[10] - 4096 cols_helper_7_13[11]
   - 8192 cols_helper_7_13[12] - 16384 cols_helper_7_13[13]
   - 32768 cols_helper_7_13[14] - 65536 cols_helper_7_13[15]
   - 131072 cols_helper_7_13[16] = 0
 cols_acc_7_13: - 43493 frac[7,13] + cols_7_13 >= 0
 accuracy_7_13: mem[7,13] - 0.01171875 cols_7_13 = 0
 accuracy_3_4: - 509.68359375 frac[3,4] + mem[3,4] = 0
 cols_helper_9_12: cols_helper_9_12[0] + cols_helper_9_12[1]
   + cols_helper_9_12[2] + cols_helper_9_12[3] + cols_helper_9_12[4]
   + cols_helper_9_12[5] + cols_helper_9_12[6] + cols_helper_9_12[7]
   + cols_helper_9_12[8] + cols_helper_9_12[9] + cols_helper_9_12[10]
   + cols_helper_9_12[11] + cols_helper_9_12[12] + cols_helper_9_12[13]
   + cols_helper_9_12[14] + cols_helper_9_12[15] + cols_helper_9_12[16]
   <= 1
 cols_9_12: cols_9_12 - 2 cols_helper_9_12[0] - 4 cols_helper_9_12[1]
   - 8 cols_helper_9_12[2] - 16 cols_helper_9_12[3]
   - 32 cols_helper_9_12[4] - 64 cols_helper_9_12[5]
   - 128 cols_helper_9_12[6] - 256 cols_helper_9_12[7]
   - 512 cols_helper_9_12[8] - 1024 cols_helper_9_12[9]
   - 2048 cols_helper_9_12[10] - 4096 cols_helper_9_12[11]
   - 8192 cols_helper_9_12[12] - 16384 cols_helper_9_12[13]
   - 32768 cols_helper_9_12[14] - 65536 cols_helper_9_12[15]
   - 131072 cols_helper_9_12[16] = 0
 cols_acc_9_12: - 43493 frac[9,12] + cols_9_12 >= 0
 accuracy_9_12: mem[9,12] - 0.01171875 cols_9_12 = 0
 cols_helper_8_4: cols_helper_8_4[0] + cols_helper_8_4[1]
   + cols_helper_8_4[2] + cols_helper_8_4[3] + cols_helper_8_4[4]
   + cols_helper_8_4[5] + cols_helper_8_4[6] + cols_helper_8_4[7]
   + cols_helper_8_4[8] + cols_helper_8_4[9] + cols_helper_8_4[10]
   + cols_helper_8_4[11] + cols_helper_8_4[12] + cols_helper_8_4[13]
   + cols_helper_8_4[14] + cols_helper_8_4[15] + cols_helper_8_4[16] <= 1
 cols_8_4: cols_8_4 - 2 cols_helper_8_4[0] - 4 cols_helper_8_4[1]
   - 8 cols_helper_8_4[2] - 16 cols_helper_8_4[3] - 32 cols_helper_8_4[4]
   - 64 cols_helper_8_4[5] - 128 cols_helper_8_4[6]
   - 256 cols_helper_8_4[7] - 512 cols_helper_8_4[8]
   - 1024 cols_helper_8_4[9] - 2048 cols_helper_8_4[10]
   - 4096 cols_helper_8_4[11] - 8192 cols_helper_8_4[12]
   - 16384 cols_helper_8_4[13] - 32768 cols_helper_8_4[14]
   - 65536 cols_helper_8_4[15] - 131072 cols_helper_8_4[16] = 0
 cols_acc_8_4: - 43493 frac[8,4] + cols_8_4 >= 0
 accuracy_8_4: mem[8,4] - 0.01171875 cols_8_4 = 0
 cols_helper_5_9: cols_helper_5_9[0] + cols_helper_5_9[1]
   + cols_helper_5_9[2] + cols_helper_5_9[3] + cols_helper_5_9[4]
   + cols_helper_5_9[5] + cols_helper_5_9[6] + cols_helper_5_9[7]
   + cols_helper_5_9[8] + cols_helper_5_9[9] + cols_helper_5_9[10]
   + cols_helper_5_9[11] + cols_helper_5_9[12] + cols_helper_5_9[13]
   + cols_helper_5_9[14] + cols_helper_5_9[15] + cols_helper_5_9[16] <= 1
 cols_5_9: cols_5_9 - 2 cols_helper_5_9[0] - 4 cols_helper_5_9[1]
   - 8 cols_helper_5_9[2] - 16 cols_helper_5_9[3] - 32 cols_helper_5_9[4]
   - 64 cols_helper_5_9[5] - 128 cols_helper_5_9[6]
   - 256 cols_helper_5_9[7] - 512 cols_helper_5_9[8]
   - 1024 cols_helper_5_9[9] - 2048 cols_helper_5_9[10]
   - 4096 cols_helper_5_9[11] - 8192 cols_helper_5_9[12]
   - 16384 cols_helper_5_9[13] - 32768 cols_helper_5_9[14]
   - 65536 cols_helper_5_9[15] - 131072 cols_helper_5_9[16] = 0
 cols_acc_5_9: - 43493 frac[5,9] + cols_5_9 >= 0
 accuracy_5_9: mem[5,9] - 0.01171875 cols_5_9 = 0
 cols_helper_10_11: cols_helper_10_11[0] + cols_helper_10_11[1]
   + cols_helper_10_11[2] + cols_helper_10_11[3] + cols_helper_10_11[4]
   + cols_helper_10_11[5] + cols_helper_10_11[6] + cols_helper_10_11[7]
   + cols_helper_10_11[8] + cols_helper_10_11[9] + cols_helper_10_11[10]
   + cols_helper_10_11[11] + cols_helper_10_11[12] + cols_helper_10_11[13]
   + cols_helper_10_11[14] + cols_helper_10_11[15] + cols_helper_10_11[16]
   <= 1
 cols_10_11: cols_10_11 - 2 cols_helper_10_11[0] - 4 cols_helper_10_11[1]
   - 8 cols_helper_10_11[2] - 16 cols_helper_10_11[3]
   - 32 cols_helper_10_11[4] - 64 cols_helper_10_11[5]
   - 128 cols_helper_10_11[6] - 256 cols_helper_10_11[7]
   - 512 cols_helper_10_11[8] - 1024 cols_helper_10_11[9]
   - 2048 cols_helper_10_11[10] - 4096 cols_helper_10_11[11]
   - 8192 cols_helper_10_11[12] - 16384 cols_helper_10_11[13]
   - 32768 cols_helper_10_11[14] - 65536 cols_helper_10_11[15]
   - 131072 cols_helper_10_11[16] = 0
 cols_acc_10_11: - 43493 frac[10,11] + cols_10_11 >= 0
 accuracy_10_11: mem[10,11] - 0.01171875 cols_10_11 = 0
 cols_helper_6_6: cols_helper_6_6[0] + cols_helper_6_6[1]
   + cols_helper_6_6[2] + cols_helper_6_6[3] + cols_helper_6_6[4]
   + cols_helper_6_6[5] + cols_helper_6_6[6] + cols_helper_6_6[7]
   + cols_helper_6_6[8] + cols_helper_6_6[9] + cols_helper_6_6[10]
   + cols_helper_6_6[11] + cols_helper_6_6[12] + cols_helper_6_6[13]
   + cols_helper_6_6[14] + cols_helper_6_6[15] + cols_helper_6_6[16] <= 1
 cols_6_6: cols_6_6 - 2 cols_helper_6_6[0] - 4 cols_helper_6_6[1]
   - 8 cols_helper_6_6[2] - 16 cols_helper_6_6[3] - 32 cols_helper_6_6[4]
   - 64 cols_helper_6_6[5] - 128 cols_helper_6_6[6]
   - 256 cols_helper_6_6[7] - 512 cols_helper_6_6[8]
   - 1024 cols_helper_6_6[9] - 2048 cols_helper_6_6[10]
   - 4096 cols_helper_6_6[11] - 8192 cols_helper_6_6[12]
   - 16384 cols_helper_6_6[13] - 32768 cols_helper_6_6[14]
   - 65536 cols_helper_6_6[15] - 131072 cols_helper_6_6[16] = 0
 cols_acc_6_6: - 43493 frac[6,6] + cols_6_6 >= 0
 accuracy_6_6: mem[6,6] - 0.01171875 cols_6_6 = 0
 cols_helper_10_6: cols_helper_10_6[0] + cols_helper_10_6[1]
   + cols_helper_10_6[2] + cols_helper_10_6[3] + cols_helper_10_6[4]
   + cols_helper_10_6[5] + cols_helper_10_6[6] + cols_helper_10_6[7]
   + cols_helper_10_6[8] + cols_helper_10_6[9] + cols_helper_10_6[10]
   + cols_helper_10_6[11] + cols_helper_10_6[12] + cols_helper_10_6[13]
   + cols_helper_10_6[14] + cols_helper_10_6[15] + cols_helper_10_6[16]
   <= 1
 cols_10_6: cols_10_6 - 2 cols_helper_10_6[0] - 4 cols_helper_10_6[1]
   - 8 cols_helper_10_6[2] - 16 cols_helper_10_6[3]
   - 32 cols_helper_10_6[4] - 64 cols_helper_10_6[5]
   - 128 cols_helper_10_6[6] - 256 cols_helper_10_6[7]
   - 512 cols_helper_10_6[8] - 1024 cols_helper_10_6[9]
   - 2048 cols_helper_10_6[10] - 4096 cols_helper_10_6[11]
   - 8192 cols_helper_10_6[12] - 16384 cols_helper_10_6[13]
   - 32768 cols_helper_10_6[14] - 65536 cols_helper_10_6[15]
   - 131072 cols_helper_10_6[16] = 0
 cols_acc_10_6: - 43493 frac[10,6] + cols_10_6 >= 0
 accuracy_10_6: mem[10,6] - 0.01171875 cols_10_6 = 0
 accuracy_0_3: - 509.68359375 frac[0,3] + mem[0,3] = 0
 accuracy_3_14: - 509.68359375 frac[3,14] + mem[3,14] = 0
 cols_helper_8_14: cols_helper_8_14[0] + cols_helper_8_14[1]
   + cols_helper_8_14[2] + cols_helper_8_14[3] + cols_helper_8_14[4]
   + cols_helper_8_14[5] + cols_helper_8_14[6] + cols_helper_8_14[7]
   + cols_helper_8_14[8] + cols_helper_8_14[9] + cols_helper_8_14[10]
   + cols_helper_8_14[11] + cols_helper_8_14[12] + cols_helper_8_14[13]
   + cols_helper_8_14[14] + cols_helper_8_14[15] + cols_helper_8_14[16]
   <= 1
 cols_8_14: cols_8_14 - 2 cols_helper_8_14[0] - 4 cols_helper_8_14[1]
   - 8 cols_helper_8_14[2] - 16 cols_helper_8_14[3]
   - 32 cols_helper_8_14[4] - 64 cols_helper_8_14[5]
   - 128 cols_helper_8_14[6] - 256 cols_helper_8_14[7]
   - 512 cols_helper_8_14[8] - 1024 cols_helper_8_14[9]
   - 2048 cols_helper_8_14[10] - 4096 cols_helper_8_14[11]
   - 8192 cols_helper_8_14[12] - 16384 cols_helper_8_14[13]
   - 32768 cols_helper_8_14[14] - 65536 cols_helper_8_14[15]
   - 131072 cols_helper_8_14[16] = 0
 cols_acc_8_14: - 43493 frac[8,14] + cols_8_14 >= 0
 accuracy_8_14: mem[8,14] - 0.01171875 cols_8_14 = 0
 cols_helper_4_9: cols_helper_4_9[0] + cols_helper_4_9[1]
   + cols_helper_4_9[2] + cols_helper_4_9[3] + cols_helper_4_9[4]
   + cols_helper_4_9[5] + cols_helper_4_9[6] + cols_helper_4_9[7]
   + cols_helper_4_9[8] + cols_helper_4_9[9] + cols_helper_4_9[10]
   + cols_helper_4_9[11] + cols_helper_4_9[12] + cols_helper_4_9[13]
   + cols_helper_4_9[14] + cols_helper_4_9[15] + cols_helper_4_9[16] <= 1
 cols_4_9: cols_4_9 - 2 cols_helper_4_9[0] - 4 cols_helper_4_9[1]
   - 8 cols_helper_4_9[2] - 16 cols_helper_4_9[3] - 32 cols_helper_4_9[4]
   - 64 cols_helper_4_9[5] - 128 cols_helper_4_9[6]
   - 256 cols_helper_4_9[7] - 512 cols_helper_4_9[8]
   - 1024 cols_helper_4_9[9] - 2048 cols_helper_4_9[10]
   - 4096 cols_helper_4_9[11] - 8192 cols_helper_4_9[12]
   - 16384 cols_helper_4_9[13] - 32768 cols_helper_4_9[14]
   - 65536 cols_helper_4_9[15] - 131072 cols_helper_4_9[16] = 0
 cols_acc_4_9: - 43493 frac[4,9] + cols_4_9 >= 0
 accuracy_4_9: mem[4,9] - 0.01171875 cols_4_9 = 0
 accuracy_3_3: - 509.68359375 frac[3,3] + mem[3,3] = 0
 accuracy_2_9: - 509.68359375 frac[2,9] + mem[2,9] = 0
 cols_helper_9_15: cols_helper_9_15[0] + cols_helper_9_15[1]
   + cols_helper_9_15[2] + cols_helper_9_15[3] + cols_helper_9_15[4]
   + cols_helper_9_15[5] + cols_helper_9_15[6] + cols_helper_9_15[7]
   + cols_helper_9_15[8] + cols_helper_9_15[9] + cols_helper_9_15[10]
   + cols_helper_9_15[11] + cols_helper_9_15[12] + cols_helper_9_15[13]
   + cols_helper_9_15[14] + cols_helper_9_15[15] + cols_helper_9_15[16]
   <= 1
 cols_9_15: cols_9_15 - 2 cols_helper_9_15[0] - 4 cols_helper_9_15[1]
   - 8 cols_helper_9_15[2] - 16 cols_helper_9_15[3]
   - 32 cols_helper_9_15[4] - 64 cols_helper_9_15[5]
   - 128 cols_helper_9_15[6] - 256 cols_helper_9_15[7]
   - 512 cols_helper_9_15[8] - 1024 cols_helper_9_15[9]
   - 2048 cols_helper_9_15[10] - 4096 cols_helper_9_15[11]
   - 8192 cols_helper_9_15[12] - 16384 cols_helper_9_15[13]
   - 32768 cols_helper_9_15[14] - 65536 cols_helper_9_15[15]
   - 131072 cols_helper_9_15[16] = 0
 cols_acc_9_15: - 43493 frac[9,15] + cols_9_15 >= 0
 accuracy_9_15: mem[9,15] - 0.01171875 cols_9_15 = 0
 cols_helper_5_12: cols_helper_5_12[0] + cols_helper_5_12[1]
   + cols_helper_5_12[2] + cols_helper_5_12[3] + cols_helper_5_12[4]
   + cols_helper_5_12[5] + cols_helper_5_12[6] + cols_helper_5_12[7]
   + cols_helper_5_12[8] + cols_helper_5_12[9] + cols_helper_5_12[10]
   + cols_helper_5_12[11] + cols_helper_5_12[12] + cols_helper_5_12[13]
   + cols_helper_5_12[14] + cols_helper_5_12[15] + cols_helper_5_12[16]
   <= 1
 cols_5_12: cols_5_12 - 2 cols_helper_5_12[0] - 4 cols_helper_5_12[1]
   - 8 cols_helper_5_12[2] - 16 cols_helper_5_12[3]
   - 32 cols_helper_5_12[4] - 64 cols_helper_5_12[5]
   - 128 cols_helper_5_12[6] - 256 cols_helper_5_12[7]
   - 512 cols_helper_5_12[8] - 1024 cols_helper_5_12[9]
   - 2048 cols_helper_5_12[10] - 4096 cols_helper_5_12[11]
   - 8192 cols_helper_5_12[12] - 16384 cols_helper_5_12[13]
   - 32768 cols_helper_5_12[14] - 65536 cols_helper_5_12[15]
   - 131072 cols_helper_5_12[16] = 0
 cols_acc_5_12: - 43493 frac[5,12] + cols_5_12 >= 0
 accuracy_5_12: mem[5,12] - 0.01171875 cols_5_12 = 0
 cols_helper_4_4: cols_helper_4_4[0] + cols_helper_4_4[1]
   + cols_helper_4_4[2] + cols_helper_4_4[3] + cols_helper_4_4[4]
   + cols_helper_4_4[5] + cols_helper_4_4[6] + cols_helper_4_4[7]
   + cols_helper_4_4[8] + cols_helper_4_4[9] + cols_helper_4_4[10]
   + cols_helper_4_4[11] + cols_helper_4_4[12] + cols_helper_4_4[13]
   + cols_helper_4_4[14] + cols_helper_4_4[15] + cols_helper_4_4[16] <= 1
 cols_4_4: cols_4_4 - 2 cols_helper_4_4[0] - 4 cols_helper_4_4[1]
   - 8 cols_helper_4_4[2] - 16 cols_helper_4_4[3] - 32 cols_helper_4_4[4]
   - 64 cols_helper_4_4[5] - 128 cols_helper_4_4[6]
   - 256 cols_helper_4_4[7] - 512 cols_helper_4_4[8]
   - 1024 cols_helper_4_4[9] - 2048 cols_helper_4_4[10]
   - 4096 cols_helper_4_4[11] - 8192 cols_helper_4_4[12]
   - 16384 cols_helper_4_4[13] - 32768 cols_helper_4_4[14]
   - 65536 cols_helper_4_4[15] - 131072 cols_helper_4_4[16] = 0
 cols_acc_4_4: - 43493 frac[4,4] + cols_4_4 >= 0
 accuracy_4_4: mem[4,4] - 0.01171875 cols_4_4 = 0
 cols_helper_10_12: cols_helper_10_12[0] + cols_helper_10_12[1]
   + cols_helper_10_12[2] + cols_helper_10_12[3] + cols_helper_10_12[4]
   + cols_helper_10_12[5] + cols_helper_10_12[6] + cols_helper_10_12[7]
   + cols_helper_10_12[8] + cols_helper_10_12[9] + cols_helper_10_12[10]
   + cols_helper_10_12[11] + cols_helper_10_12[12] + cols_helper_10_12[13]
   + cols_helper_10_12[14] + cols_helper_10_12[15] + cols_helper_10_12[16]
   <= 1
 cols_10_12: cols_10_12 - 2 cols_helper_10_12[0] - 4 cols_helper_10_12[1]
   - 8 cols_helper_10_12[2] - 16 cols_helper_10_12[3]
   - 32 cols_helper_10_12[4] - 64 cols_helper_10_12[5]
   - 128 cols_helper_10_12[6] - 256 cols_helper_10_12[7]
   - 512 cols_helper_10_12[8] - 1024 cols_helper_10_12[9]
   - 2048 cols_helper_10_12[10] - 4096 cols_helper_10_12[11]
   - 8192 cols_helper_10_12[12] - 16384 cols_helper_10_12[13]
   - 32768 cols_helper_10_12[14] - 65536 cols_helper_10_12[15]
   - 131072 cols_helper_10_12[16] = 0
 cols_acc_10_12: - 43493 frac[10,12] + cols_10_12 >= 0
 accuracy_10_12: mem[10,12] - 0.01171875 cols_10_12 = 0
 cols_helper_6_3: cols_helper_6_3[0] + cols_helper_6_3[1]
   + cols_helper_6_3[2] + cols_helper_6_3[3] + cols_helper_6_3[4]
   + cols_helper_6_3[5] + cols_helper_6_3[6] + cols_helper_6_3[7]
   + cols_helper_6_3[8] + cols_helper_6_3[9] + cols_helper_6_3[10]
   + cols_helper_6_3[11] + cols_helper_6_3[12] + cols_helper_6_3[13]
   + cols_helper_6_3[14] + cols_helper_6_3[15] + cols_helper_6_3[16] <= 1
 cols_6_3: cols_6_3 - 2 cols_helper_6_3[0] - 4 cols_helper_6_3[1]
   - 8 cols_helper_6_3[2] - 16 cols_helper_6_3[3] - 32 cols_helper_6_3[4]
   - 64 cols_helper_6_3[5] - 128 cols_helper_6_3[6]
   - 256 cols_helper_6_3[7] - 512 cols_helper_6_3[8]
   - 1024 cols_helper_6_3[9] - 2048 cols_helper_6_3[10]
   - 4096 cols_helper_6_3[11] - 8192 cols_helper_6_3[12]
   - 16384 cols_helper_6_3[13] - 32768 cols_helper_6_3[14]
   - 65536 cols_helper_6_3[15] - 131072 cols_helper_6_3[16] = 0
 cols_acc_6_3: - 43493 frac[6,3] + cols_6_3 >= 0
 accuracy_6_3: mem[6,3] - 0.01171875 cols_6_3 = 0
 cols_helper_7_10: cols_helper_7_10[0] + cols_helper_7_10[1]
   + cols_helper_7_10[2] + cols_helper_7_10[3] + cols_helper_7_10[4]
   + cols_helper_7_10[5] + cols_helper_7_10[6] + cols_helper_7_10[7]
   + cols_helper_7_10[8] + cols_helper_7_10[9] + cols_helper_7_10[10]
   + cols_helper_7_10[11] + cols_helper_7_10[12] + cols_helper_7_10[13]
   + cols_helper_7_10[14] + cols_helper_7_10[15] + cols_helper_7_10[16]
   <= 1
 cols_7_10: cols_7_10 - 2 cols_helper_7_10[0] - 4 cols_helper_7_10[1]
   - 8 cols_helper_7_10[2] - 16 cols_helper_7_10[3]
   - 32 cols_helper_7_10[4] - 64 cols_helper_7_10[5]
   - 128 cols_helper_7_10[6] - 256 cols_helper_7_10[7]
   - 512 cols_helper_7_10[8] - 1024 cols_helper_7_10[9]
   - 2048 cols_helper_7_10[10] - 4096 cols_helper_7_10[11]
   - 8192 cols_helper_7_10[12] - 16384 cols_helper_7_10[13]
   - 32768 cols_helper_7_10[14] - 65536 cols_helper_7_10[15]
   - 131072 cols_helper_7_10[16] = 0
 cols_acc_7_10: - 43493 frac[7,10] + cols_7_10 >= 0
 accuracy_7_10: mem[7,10] - 0.01171875 cols_7_10 = 0
 accuracy_3_13: - 509.68359375 frac[3,13] + mem[3,13] = 0
 cols_helper_8_11: cols_helper_8_11[0] + cols_helper_8_11[1]
   + cols_helper_8_11[2] + cols_helper_8_11[3] + cols_helper_8_11[4]
   + cols_helper_8_11[5] + cols_helper_8_11[6] + cols_helper_8_11[7]
   + cols_helper_8_11[8] + cols_helper_8_11[9] + cols_helper_8_11[10]
   + cols_helper_8_11[11] + cols_helper_8_11[12] + cols_helper_8_11[13]
   + cols_helper_8_11[14] + cols_helper_8_11[15] + cols_helper_8_11[16]
   <= 1
 cols_8_11: cols_8_11 - 2 cols_helper_8_11[0] - 4 cols_helper_8_11[1]
   - 8 cols_helper_8_11[2] - 16 cols_helper_8_11[3]
   - 32 cols_helper_8_11[4] - 64 cols_helper_8_11[5]
   - 128 cols_helper_8_11[6] - 256 cols_helper_8_11[7]
   - 512 cols_helper_8_11[8] - 1024 cols_helper_8_11[9]
   - 2048 cols_helper_8_11[10] - 4096 cols_helper_8_11[11]
   - 8192 cols_helper_8_11[12] - 16384 cols_helper_8_11[13]
   - 32768 cols_helper_8_11[14] - 65536 cols_helper_8_11[15]
   - 131072 cols_helper_8_11[16] = 0
 cols_acc_8_11: - 43493 frac[8,11] + cols_8_11 >= 0
 accuracy_8_11: mem[8,11] - 0.01171875 cols_8_11 = 0
 cols_helper_4_14: cols_helper_4_14[0] + cols_helper_4_14[1]
   + cols_helper_4_14[2] + cols_helper_4_14[3] + cols_helper_4_14[4]
   + cols_helper_4_14[5] + cols_helper_4_14[6] + cols_helper_4_14[7]
   + cols_helper_4_14[8] + cols_helper_4_14[9] + cols_helper_4_14[10]
   + cols_helper_4_14[11] + cols_helper_4_14[12] + cols_helper_4_14[13]
   + cols_helper_4_14[14] + cols_helper_4_14[15] + cols_helper_4_14[16]
   <= 1
 cols_4_14: cols_4_14 - 2 cols_helper_4_14[0] - 4 cols_helper_4_14[1]
   - 8 cols_helper_4_14[2] - 16 cols_helper_4_14[3]
   - 32 cols_helper_4_14[4] - 64 cols_helper_4_14[5]
   - 128 cols_helper_4_14[6] - 256 cols_helper_4_14[7]
   - 512 cols_helper_4_14[8] - 1024 cols_helper_4_14[9]
   - 2048 cols_helper_4_14[10] - 4096 cols_helper_4_14[11]
   - 8192 cols_helper_4_14[12] - 16384 cols_helper_4_14[13]
   - 32768 cols_helper_4_14[14] - 65536 cols_helper_4_14[15]
   - 131072 cols_helper_4_14[16] = 0
 cols_acc_4_14: - 43493 frac[4,14] + cols_4_14 >= 0
 accuracy_4_14: mem[4,14] - 0.01171875 cols_4_14 = 0
 accuracy_2_10: - 509.68359375 frac[2,10] + mem[2,10] = 0
 cols_helper_5_15: cols_helper_5_15[0] + cols_helper_5_15[1]
   + cols_helper_5_15[2] + cols_helper_5_15[3] + cols_helper_5_15[4]
   + cols_helper_5_15[5] + cols_helper_5_15[6] + cols_helper_5_15[7]
   + cols_helper_5_15[8] + cols_helper_5_15[9] + cols_helper_5_15[10]
   + cols_helper_5_15[11] + cols_helper_5_15[12] + cols_helper_5_15[13]
   + cols_helper_5_15[14] + cols_helper_5_15[15] + cols_helper_5_15[16]
   <= 1
 cols_5_15: cols_5_15 - 2 cols_helper_5_15[0] - 4 cols_helper_5_15[1]
   - 8 cols_helper_5_15[2] - 16 cols_helper_5_15[3]
   - 32 cols_helper_5_15[4] - 64 cols_helper_5_15[5]
   - 128 cols_helper_5_15[6] - 256 cols_helper_5_15[7]
   - 512 cols_helper_5_15[8] - 1024 cols_helper_5_15[9]
   - 2048 cols_helper_5_15[10] - 4096 cols_helper_5_15[11]
   - 8192 cols_helper_5_15[12] - 16384 cols_helper_5_15[13]
   - 32768 cols_helper_5_15[14] - 65536 cols_helper_5_15[15]
   - 131072 cols_helper_5_15[16] = 0
 cols_acc_5_15: - 43493 frac[5,15] + cols_5_15 >= 0
 accuracy_5_15: mem[5,15] - 0.01171875 cols_5_15 = 0
 cols_helper_6_12: cols_helper_6_12[0] + cols_helper_6_12[1]
   + cols_helper_6_12[2] + cols_helper_6_12[3] + cols_helper_6_12[4]
   + cols_helper_6_12[5] + cols_helper_6_12[6] + cols_helper_6_12[7]
   + cols_helper_6_12[8] + cols_helper_6_12[9] + cols_helper_6_12[10]
   + cols_helper_6_12[11] + cols_helper_6_12[12] + cols_helper_6_12[13]
   + cols_helper_6_12[14] + cols_helper_6_12[15] + cols_helper_6_12[16]
   <= 1
 cols_6_12: cols_6_12 - 2 cols_helper_6_12[0] - 4 cols_helper_6_12[1]
   - 8 cols_helper_6_12[2] - 16 cols_helper_6_12[3]
   - 32 cols_helper_6_12[4] - 64 cols_helper_6_12[5]
   - 128 cols_helper_6_12[6] - 256 cols_helper_6_12[7]
   - 512 cols_helper_6_12[8] - 1024 cols_helper_6_12[9]
   - 2048 cols_helper_6_12[10] - 4096 cols_helper_6_12[11]
   - 8192 cols_helper_6_12[12] - 16384 cols_helper_6_12[13]
   - 32768 cols_helper_6_12[14] - 65536 cols_helper_6_12[15]
   - 131072 cols_helper_6_12[16] = 0
 cols_acc_6_12: - 43493 frac[6,12] + cols_6_12 >= 0
 accuracy_6_12: mem[6,12] - 0.01171875 cols_6_12 = 0
 cols_helper_7_9: cols_helper_7_9[0] + cols_helper_7_9[1]
   + cols_helper_7_9[2] + cols_helper_7_9[3] + cols_helper_7_9[4]
   + cols_helper_7_9[5] + cols_helper_7_9[6] + cols_helper_7_9[7]
   + cols_helper_7_9[8] + cols_helper_7_9[9] + cols_helper_7_9[10]
   + cols_helper_7_9[11] + cols_helper_7_9[12] + cols_helper_7_9[13]
   + cols_helper_7_9[14] + cols_helper_7_9[15] + cols_helper_7_9[16] <= 1
 cols_7_9: cols_7_9 - 2 cols_helper_7_9[0] - 4 cols_helper_7_9[1]
   - 8 cols_helper_7_9[2] - 16 cols_helper_7_9[3] - 32 cols_helper_7_9[4]
   - 64 cols_helper_7_9[5] - 128 cols_helper_7_9[6]
   - 256 cols_helper_7_9[7] - 512 cols_helper_7_9[8]
   - 1024 cols_helper_7_9[9] - 2048 cols_helper_7_9[10]
   - 4096 cols_helper_7_9[11] - 8192 cols_helper_7_9[12]
   - 16384 cols_helper_7_9[13] - 32768 cols_helper_7_9[14]
   - 65536 cols_helper_7_9[15] - 131072 cols_helper_7_9[16] = 0
 cols_acc_7_9: - 43493 frac[7,9] + cols_7_9 >= 0
 accuracy_7_9: mem[7,9] - 0.01171875 cols_7_9 = 0
 accuracy_3_8: - 509.68359375 frac[3,8] + mem[3,8] = 0
 cols_helper_8_8: cols_helper_8_8[0] + cols_helper_8_8[1]
   + cols_helper_8_8[2] + cols_helper_8_8[3] + cols_helper_8_8[4]
   + cols_helper_8_8[5] + cols_helper_8_8[6] + cols_helper_8_8[7]
   + cols_helper_8_8[8] + cols_helper_8_8[9] + cols_helper_8_8[10]
   + cols_helper_8_8[11] + cols_helper_8_8[12] + cols_helper_8_8[13]
   + cols_helper_8_8[14] + cols_helper_8_8[15] + cols_helper_8_8[16] <= 1
 cols_8_8: cols_8_8 - 2 cols_helper_8_8[0] - 4 cols_helper_8_8[1]
   - 8 cols_helper_8_8[2] - 16 cols_helper_8_8[3] - 32 cols_helper_8_8[4]
   - 64 cols_helper_8_8[5] - 128 cols_helper_8_8[6]
   - 256 cols_helper_8_8[7] - 512 cols_helper_8_8[8]
   - 1024 cols_helper_8_8[9] - 2048 cols_helper_8_8[10]
   - 4096 cols_helper_8_8[11] - 8192 cols_helper_8_8[12]
   - 16384 cols_helper_8_8[13] - 32768 cols_helper_8_8[14]
   - 65536 cols_helper_8_8[15] - 131072 cols_helper_8_8[16] = 0
 cols_acc_8_8: - 43493 frac[8,8] + cols_8_8 >= 0
 accuracy_8_8: mem[8,8] - 0.01171875 cols_8_8 = 0
 cols_helper_4_3: cols_helper_4_3[0] + cols_helper_4_3[1]
   + cols_helper_4_3[2] + cols_helper_4_3[3] + cols_helper_4_3[4]
   + cols_helper_4_3[5] + cols_helper_4_3[6] + cols_helper_4_3[7]
   + cols_helper_4_3[8] + cols_helper_4_3[9] + cols_helper_4_3[10]
   + cols_helper_4_3[11] + cols_helper_4_3[12] + cols_helper_4_3[13]
   + cols_helper_4_3[14] + cols_helper_4_3[15] + cols_helper_4_3[16] <= 1
 cols_4_3: cols_4_3 - 2 cols_helper_4_3[0] - 4 cols_helper_4_3[1]
   - 8 cols_helper_4_3[2] - 16 cols_helper_4_3[3] - 32 cols_helper_4_3[4]
   - 64 cols_helper_4_3[5] - 128 cols_helper_4_3[6]
   - 256 cols_helper_4_3[7] - 512 cols_helper_4_3[8]
   - 1024 cols_helper_4_3[9] - 2048 cols_helper_4_3[10]
   - 4096 cols_helper_4_3[11] - 8192 cols_helper_4_3[12]
   - 16384 cols_helper_4_3[13] - 32768 cols_helper_4_3[14]
   - 65536 cols_helper_4_3[15] - 131072 cols_helper_4_3[16] = 0
 cols_acc_4_3: - 43493 frac[4,3] + cols_4_3 >= 0
 accuracy_4_3: mem[4,3] - 0.01171875 cols_4_3 = 0
 cols_helper_9_5: cols_helper_9_5[0] + cols_helper_9_5[1]
   + cols_helper_9_5[2] + cols_helper_9_5[3] + cols_helper_9_5[4]
   + cols_helper_9_5[5] + cols_helper_9_5[6] + cols_helper_9_5[7]
   + cols_helper_9_5[8] + cols_helper_9_5[9] + cols_helper_9_5[10]
   + cols_helper_9_5[11] + cols_helper_9_5[12] + cols_helper_9_5[13]
   + cols_helper_9_5[14] + cols_helper_9_5[15] + cols_helper_9_5[16] <= 1
 cols_9_5: cols_9_5 - 2 cols_helper_9_5[0] - 4 cols_helper_9_5[1]
   - 8 cols_helper_9_5[2] - 16 cols_helper_9_5[3] - 32 cols_helper_9_5[4]
   - 64 cols_helper_9_5[5] - 128 cols_helper_9_5[6]
   - 256 cols_helper_9_5[7] - 512 cols_helper_9_5[8]
   - 1024 cols_helper_9_5[9] - 2048 cols_helper_9_5[10]
   - 4096 cols_helper_9_5[11] - 8192 cols_helper_9_5[12]
   - 16384 cols_helper_9_5[13] - 32768 cols_helper_9_5[14]
   - 65536 cols_helper_9_5[15] - 131072 cols_helper_9_5[16] = 0
 cols_acc_9_5: - 43493 frac[9,5] + cols_9_5 >= 0
 accuracy_9_5: mem[9,5] - 0.01171875 cols_9_5 = 0
 cols_helper_5_2: cols_helper_5_2[0] + cols_helper_5_2[1]
   + cols_helper_5_2[2] + cols_helper_5_2[3] + cols_helper_5_2[4]
   + cols_helper_5_2[5] + cols_helper_5_2[6] + cols_helper_5_2[7]
   + cols_helper_5_2[8] + cols_helper_5_2[9] + cols_helper_5_2[10]
   + cols_helper_5_2[11] + cols_helper_5_2[12] + cols_helper_5_2[13]
   + cols_helper_5_2[14] + cols_helper_5_2[15] + cols_helper_5_2[16] <= 1
 cols_5_2: cols_5_2 - 2 cols_helper_5_2[0] - 4 cols_helper_5_2[1]
   - 8 cols_helper_5_2[2] - 16 cols_helper_5_2[3] - 32 cols_helper_5_2[4]
   - 64 cols_helper_5_2[5] - 128 cols_helper_5_2[6]
   - 256 cols_helper_5_2[7] - 512 cols_helper_5_2[8]
   - 1024 cols_helper_5_2[9] - 2048 cols_helper_5_2[10]
   - 4096 cols_helper_5_2[11] - 8192 cols_helper_5_2[12]
   - 16384 cols_helper_5_2[13] - 32768 cols_helper_5_2[14]
   - 65536 cols_helper_5_2[15] - 131072 cols_helper_5_2[16] = 0
 cols_acc_5_2: - 43493 frac[5,2] + cols_5_2 >= 0
 accuracy_5_2: mem[5,2] - 0.01171875 cols_5_2 = 0
 cols_helper_10_2: cols_helper_10_2[0] + cols_helper_10_2[1]
   + cols_helper_10_2[2] + cols_helper_10_2[3] + cols_helper_10_2[4]
   + cols_helper_10_2[5] + cols_helper_10_2[6] + cols_helper_10_2[7]
   + cols_helper_10_2[8] + cols_helper_10_2[9] + cols_helper_10_2[10]
   + cols_helper_10_2[11] + cols_helper_10_2[12] + cols_helper_10_2[13]
   + cols_helper_10_2[14] + cols_helper_10_2[15] + cols_helper_10_2[16]
   <= 1
 cols_10_2: cols_10_2 - 2 cols_helper_10_2[0] - 4 cols_helper_10_2[1]
   - 8 cols_helper_10_2[2] - 16 cols_helper_10_2[3]
   - 32 cols_helper_10_2[4] - 64 cols_helper_10_2[5]
   - 128 cols_helper_10_2[6] - 256 cols_helper_10_2[7]
   - 512 cols_helper_10_2[8] - 1024 cols_helper_10_2[9]
   - 2048 cols_helper_10_2[10] - 4096 cols_helper_10_2[11]
   - 8192 cols_helper_10_2[12] - 16384 cols_helper_10_2[13]
   - 32768 cols_helper_10_2[14] - 65536 cols_helper_10_2[15]
   - 131072 cols_helper_10_2[16] = 0
 cols_acc_10_2: - 43493 frac[10,2] + cols_10_2 >= 0
 accuracy_10_2: mem[10,2] - 0.01171875 cols_10_2 = 0
 mem_tot_{CPU13,CPU4,CPU6,CPU30,CPU9,CPU16,CPU22,CPU32}: - mem[0,14]
   - mem[0,2] - mem[0,3]
   + mem_tot_{CPU13,CPU4,CPU6,CPU30,CPU9,CPU16,CPU22,CPU32} = 0
 R335: - 3.8147e-06 mem_tot_{CPU13,CPU4,CPU6,CPU30,CPU9,CPU16,CPU22,CPU32}
   + normalized_mem_tot_{CPU13,CPU4,CPU6,CPU30,CPU9,CPU16,CPU22,CPU32} = 0
 rows_tot_{CPU13,CPU4,CPU6,CPU30,CPU9,CPU16,CPU22,CPU32}: - 3 frac[0,14]
   - 3 frac[0,2] - 3 frac[0,3]
   + rows_tot_{CPU13,CPU4,CPU6,CPU30,CPU9,CPU16,CPU22,CPU32} = 0
 R337:
   - 0.0104166666666667 rows_tot_{CPU13,CPU4,CPU6,CPU30,CPU9,CPU16,CPU22,CPU32}
   + normalized_rows_tot_{CPU13,CPU4,CPU6,CPU30,CPU9,CPU16,CPU22,CPU32}
   = 0
 mem_tot_{CPU29,CPU8,CPU7,CPU21,CPU2,CPU3,CPU18,CPU5}: - mem[1,6]
   - mem[1,5] - mem[1,4]
   + mem_tot_{CPU29,CPU8,CPU7,CPU21,CPU2,CPU3,CPU18,CPU5} = 0
 R339: - 3.8147e-06 mem_tot_{CPU29,CPU8,CPU7,CPU21,CPU2,CPU3,CPU18,CPU5}
   + normalized_mem_tot_{CPU29,CPU8,CPU7,CPU21,CPU2,CPU3,CPU18,CPU5} = 0
 rows_tot_{CPU29,CPU8,CPU7,CPU21,CPU2,CPU3,CPU18,CPU5}: - 3 frac[1,6]
   - 3 frac[1,5] - 3 frac[1,4]
   + rows_tot_{CPU29,CPU8,CPU7,CPU21,CPU2,CPU3,CPU18,CPU5} = 0
 R341:
   - 0.0104166666666667 rows_tot_{CPU29,CPU8,CPU7,CPU21,CPU2,CPU3,CPU18,CPU5}
   + normalized_rows_tot_{CPU29,CPU8,CPU7,CPU21,CPU2,CPU3,CPU18,CPU5} = 0
 mem_tot_{CPU28,CPU19,CPU25,CPU1,CPU10,CPU17,CPU11,CPU14}: - mem[2,11]
   - mem[2,8] - mem[2,9] - mem[2,10]
   + mem_tot_{CPU28,CPU19,CPU25,CPU1,CPU10,CPU17,CPU11,CPU14} = 0
 R343:
   - 3.8147e-06 mem_tot_{CPU28,CPU19,CPU25,CPU1,CPU10,CPU17,CPU11,CPU14}
   + normalized_mem_tot_{CPU28,CPU19,CPU25,CPU1,CPU10,CPU17,CPU11,CPU14}
   = 0
 rows_tot_{CPU28,CPU19,CPU25,CPU1,CPU10,CPU17,CPU11,CPU14}: - 3 frac[2,11]
   - 3 frac[2,8] - 3 frac[2,9] - 3 frac[2,10]
   + rows_tot_{CPU28,CPU19,CPU25,CPU1,CPU10,CPU17,CPU11,CPU14} = 0
 R345:
   - 0.0104166666666667 rows_tot_{CPU28,CPU19,CPU25,CPU1,CPU10,CPU17,CPU11,CPU14}
   + normalized_rows_tot_{CPU28,CPU19,CPU25,CPU1,CPU10,CPU17,CPU11,CPU14}
   = 0
 mem_tot_{CPU31,CPU12,CPU24,CPU27,CPU26,CPU23,CPU20,CPU15}: - mem[3,2]
   - mem[3,12] - mem[3,15] - mem[3,4] - mem[3,14] - mem[3,3] - mem[3,13]
   - mem[3,8] + mem_tot_{CPU31,CPU12,CPU24,CPU27,CPU26,CPU23,CPU20,CPU15}
   = 0
 R347:
   - 3.8147e-06 mem_tot_{CPU31,CPU12,CPU24,CPU27,CPU26,CPU23,CPU20,CPU15}
   + normalized_mem_tot_{CPU31,CPU12,CPU24,CPU27,CPU26,CPU23,CPU20,CPU15}
   = 0
 rows_tot_{CPU31,CPU12,CPU24,CPU27,CPU26,CPU23,CPU20,CPU15}: - 3 frac[3,2]
   - 3 frac[3,12] - 3 frac[3,15] - 3 frac[3,4] - 3 frac[3,14] - 3 frac[3,3]
   - 3 frac[3,13] - 3 frac[3,8]
   + rows_tot_{CPU31,CPU12,CPU24,CPU27,CPU26,CPU23,CPU20,CPU15} = 0
 R349:
   - 0.0104166666666667 rows_tot_{CPU31,CPU12,CPU24,CPU27,CPU26,CPU23,CPU20,CPU15}
   + normalized_rows_tot_{CPU31,CPU12,CPU24,CPU27,CPU26,CPU23,CPU20,CPU15}
   = 0
 mem_tot_tor_P41: - mem[4,5] - mem[4,6] - mem[4,8] - mem[4,9] - mem[4,4]
   - mem[4,14] - mem[4,3] + mem_tot_tor_P41 = 0
 R351: - 1.6276041666666666e-04 mem_tot_tor_P41
   + normalized_mem_tot_tor_P41 = 0
 rows_tot_tor_P41: - 3 frac[4,5] - 3 frac[4,6] - 3 frac[4,8] - 3 frac[4,9]
   - 3 frac[4,4] - 3 frac[4,14] - 3 frac[4,3] + rows_tot_tor_P41 = 0
 R353: - 0.0208333333333333 rows_tot_tor_P41 + normalized_rows_tot_tor_P41
   = 0
 mem_tot_tor_P42: - mem[5,8] - mem[5,11] - mem[5,14] - mem[5,3] - mem[5,9]
   - mem[5,12] - mem[5,15] - mem[5,2] + mem_tot_tor_P42 = 0
 R355: - 1.6276041666666666e-04 mem_tot_tor_P42
   + normalized_mem_tot_tor_P42 = 0
 rows_tot_tor_P42: - 3 frac[5,8] - 3 frac[5,11] - 3 frac[5,14]
   - 3 frac[5,3] - 3 frac[5,9] - 3 frac[5,12] - 3 frac[5,15] - 3 frac[5,2]
   + rows_tot_tor_P42 = 0
 R357: - 0.0208333333333333 rows_tot_tor_P42 + normalized_rows_tot_tor_P42
   = 0
 mem_tot_tor_P43: - mem[6,9] - mem[6,13] - mem[6,14] - mem[6,4] - mem[6,11]
   - mem[6,2] - mem[6,15] - mem[6,5] - mem[6,8] - mem[6,6] - mem[6,3]
   - mem[6,12] + mem_tot_tor_P43 = 0
 R359: - 1.6276041666666666e-04 mem_tot_tor_P43
   + normalized_mem_tot_tor_P43 = 0
 rows_tot_tor_P43: - 3 frac[6,9] - 3 frac[6,13] - 3 frac[6,14]
   - 3 frac[6,4] - 3 frac[6,11] - 3 frac[6,2] - 3 frac[6,15] - 3 frac[6,5]
   - 3 frac[6,8] - 3 frac[6,6] - 3 frac[6,3] - 3 frac[6,12]
   + rows_tot_tor_P43 = 0
 R361: - 0.0208333333333333 rows_tot_tor_P43 + normalized_rows_tot_tor_P43
   = 0
 mem_tot_tor_P44: - mem[7,3] - mem[7,12] - mem[7,8] - mem[7,2] - mem[7,15]
   - mem[7,4] - mem[7,14] - mem[7,13] - mem[7,10] - mem[7,9]
   + mem_tot_tor_P44 = 0
 R363: - 1.6276041666666666e-04 mem_tot_tor_P44
   + normalized_mem_tot_tor_P44 = 0
 rows_tot_tor_P44: - 3 frac[7,3] - 3 frac[7,12] - 3 frac[7,8] - 3 frac[7,2]
   - 3 frac[7,15] - 3 frac[7,4] - 3 frac[7,14] - 3 frac[7,13]
   - 3 frac[7,10] - 3 frac[7,9] + rows_tot_tor_P44 = 0
 R365: - 0.0208333333333333 rows_tot_tor_P44 + normalized_rows_tot_tor_P44
   = 0
 mem_tot_l1_P41: - mem[8,5] - mem[8,15] - mem[8,2] - mem[8,12] - mem[8,9]
   - mem[8,6] - mem[8,3] - mem[8,4] - mem[8,14] - mem[8,11] - mem[8,8]
   + mem_tot_l1_P41 = 0
 R367: - 1.6276041666666666e-04 mem_tot_l1_P41 + normalized_mem_tot_l1_P41
   = 0
 rows_tot_l1_P41: - 3 frac[8,5] - 3 frac[8,15] - 3 frac[8,2] - 3 frac[8,12]
   - 3 frac[8,9] - 3 frac[8,6] - 3 frac[8,3] - 3 frac[8,4] - 3 frac[8,14]
   - 3 frac[8,11] - 3 frac[8,8] + rows_tot_l1_P41 = 0
 R369: - 0.0208333333333333 rows_tot_l1_P41 + normalized_rows_tot_l1_P41
   = 0
 mem_tot_l1_P42: - mem[9,14] - mem[9,3] - mem[9,9] - mem[9,4] - mem[9,13]
   - mem[9,2] - mem[9,8] - mem[9,11] - mem[9,6] - mem[9,12] - mem[9,15]
   - mem[9,5] + mem_tot_l1_P42 = 0
 R371: - 1.6276041666666666e-04 mem_tot_l1_P42 + normalized_mem_tot_l1_P42
   = 0
 rows_tot_l1_P42: - 3 frac[9,14] - 3 frac[9,3] - 3 frac[9,9] - 3 frac[9,4]
   - 3 frac[9,13] - 3 frac[9,2] - 3 frac[9,8] - 3 frac[9,11] - 3 frac[9,6]
   - 3 frac[9,12] - 3 frac[9,15] - 3 frac[9,5] + rows_tot_l1_P42 = 0
 R373: - 0.0208333333333333 rows_tot_l1_P42 + normalized_rows_tot_l1_P42
   = 0
 mem_tot_l2_P4: - mem[10,8] - mem[10,14] - mem[10,3] - mem[10,9]
   - mem[10,4] - mem[10,15] - mem[10,5] - mem[10,11] - mem[10,6]
   - mem[10,12] - mem[10,2] + mem_tot_l2_P4 = 0
 R375: - 1.6276041666666666e-04 mem_tot_l2_P4 + normalized_mem_tot_l2_P4
   = 0
 rows_tot_l2_P4: - 3 frac[10,8] - 3 frac[10,14] - 3 frac[10,3]
   - 3 frac[10,9] - 3 frac[10,4] - 3 frac[10,15] - 3 frac[10,5]
   - 3 frac[10,11] - 3 frac[10,6] - 3 frac[10,12] - 3 frac[10,2]
   + rows_tot_l2_P4 = 0
 R377: - 0.0208333333333333 rows_tot_l2_P4 + normalized_rows_tot_l2_P4 = 0
 row_capacity_{CPU13,CPU4,CPU6,CPU30,CPU9,CPU16,CPU22,CPU32}:
   rows_tot_{CPU13,CPU4,CPU6,CPU30,CPU9,CPU16,CPU22,CPU32} <= 96
 row_capacity_{CPU29,CPU8,CPU7,CPU21,CPU2,CPU3,CPU18,CPU5}:
   rows_tot_{CPU29,CPU8,CPU7,CPU21,CPU2,CPU3,CPU18,CPU5} <= 96
 row_capacity_{CPU28,CPU19,CPU25,CPU1,CPU10,CPU17,CPU11,CPU14}:
   rows_tot_{CPU28,CPU19,CPU25,CPU1,CPU10,CPU17,CPU11,CPU14} <= 96
 row_capacity_{CPU31,CPU12,CPU24,CPU27,CPU26,CPU23,CPU20,CPU15}:
   rows_tot_{CPU31,CPU12,CPU24,CPU27,CPU26,CPU23,CPU20,CPU15} <= 96
 row_capacity_tor_P41: rows_tot_tor_P41 <= 48
 capacity_mem_par_tor_P41: mem[4,5] <= 1536
 capacity_mem_par_tor_P41: mem[4,6] <= 1536
 capacity_mem_par_tor_P41: mem[4,8] <= 1536
 capacity_mem_par_tor_P41: mem[4,9] <= 1536
 capacity_mem_par_tor_P41: mem[4,4] <= 1536
 capacity_mem_par_tor_P41: mem[4,14] <= 1536
 capacity_mem_par_tor_P41: mem[4,3] <= 1536
 row_capacity_tor_P42: rows_tot_tor_P42 <= 48
 capacity_mem_par_tor_P42: mem[5,8] <= 1536
 capacity_mem_par_tor_P42: mem[5,11] <= 1536
 capacity_mem_par_tor_P42: mem[5,14] <= 1536
 capacity_mem_par_tor_P42: mem[5,3] <= 1536
 capacity_mem_par_tor_P42: mem[5,9] <= 1536
 capacity_mem_par_tor_P42: mem[5,12] <= 1536
 capacity_mem_par_tor_P42: mem[5,15] <= 1536
 capacity_mem_par_tor_P42: mem[5,2] <= 1536
 row_capacity_tor_P43: rows_tot_tor_P43 <= 48
 capacity_mem_par_tor_P43: mem[6,9] <= 1536
 capacity_mem_par_tor_P43: mem[6,13] <= 1536
 capacity_mem_par_tor_P43: mem[6,14] <= 1536
 capacity_mem_par_tor_P43: mem[6,4] <= 1536
 capacity_mem_par_tor_P43: mem[6,11] <= 1536
 capacity_mem_par_tor_P43: mem[6,2] <= 1536
 capacity_mem_par_tor_P43: mem[6,15] <= 1536
 capacity_mem_par_tor_P43: mem[6,5] <= 1536
 capacity_mem_par_tor_P43: mem[6,8] <= 1536
 capacity_mem_par_tor_P43: mem[6,6] <= 1536
 capacity_mem_par_tor_P43: mem[6,3] <= 1536
 capacity_mem_par_tor_P43: mem[6,12] <= 1536
 row_capacity_tor_P44: rows_tot_tor_P44 <= 48
 capacity_mem_par_tor_P44: mem[7,3] <= 1536
 capacity_mem_par_tor_P44: mem[7,12] <= 1536
 capacity_mem_par_tor_P44: mem[7,8] <= 1536
 capacity_mem_par_tor_P44: mem[7,2] <= 1536
 capacity_mem_par_tor_P44: mem[7,15] <= 1536
 capacity_mem_par_tor_P44: mem[7,4] <= 1536
 capacity_mem_par_tor_P44: mem[7,14] <= 1536
 capacity_mem_par_tor_P44: mem[7,13] <= 1536
 capacity_mem_par_tor_P44: mem[7,10] <= 1536
 capacity_mem_par_tor_P44: mem[7,9] <= 1536
 row_capacity_l1_P41: rows_tot_l1_P41 <= 48
 capacity_mem_par_l1_P41: mem[8,5] <= 1536
 capacity_mem_par_l1_P41: mem[8,15] <= 1536
 capacity_mem_par_l1_P41: mem[8,2] <= 1536
 capacity_mem_par_l1_P41: mem[8,12] <= 1536
 capacity_mem_par_l1_P41: mem[8,9] <= 1536
 capacity_mem_par_l1_P41: mem[8,6] <= 1536
 capacity_mem_par_l1_P41: mem[8,3] <= 1536
 capacity_mem_par_l1_P41: mem[8,4] <= 1536
 capacity_mem_par_l1_P41: mem[8,14] <= 1536
 capacity_mem_par_l1_P41: mem[8,11] <= 1536
 capacity_mem_par_l1_P41: mem[8,8] <= 1536
 row_capacity_l1_P42: rows_tot_l1_P42 <= 48
 capacity_mem_par_l1_P42: mem[9,14] <= 1536
 capacity_mem_par_l1_P42: mem[9,3] <= 1536
 capacity_mem_par_l1_P42: mem[9,9] <= 1536
 capacity_mem_par_l1_P42: mem[9,4] <= 1536
 capacity_mem_par_l1_P42: mem[9,13] <= 1536
 capacity_mem_par_l1_P42: mem[9,2] <= 1536
 capacity_mem_par_l1_P42: mem[9,8] <= 1536
 capacity_mem_par_l1_P42: mem[9,11] <= 1536
 capacity_mem_par_l1_P42: mem[9,6] <= 1536
 capacity_mem_par_l1_P42: mem[9,12] <= 1536
 capacity_mem_par_l1_P42: mem[9,15] <= 1536
 capacity_mem_par_l1_P42: mem[9,5] <= 1536
 row_capacity_l2_P4: rows_tot_l2_P4 <= 48
 capacity_mem_par_l2_P4: mem[10,8] <= 1536
 capacity_mem_par_l2_P4: mem[10,14] <= 1536
 capacity_mem_par_l2_P4: mem[10,3] <= 1536
 capacity_mem_par_l2_P4: mem[10,9] <= 1536
 capacity_mem_par_l2_P4: mem[10,4] <= 1536
 capacity_mem_par_l2_P4: mem[10,15] <= 1536
 capacity_mem_par_l2_P4: mem[10,5] <= 1536
 capacity_mem_par_l2_P4: mem[10,11] <= 1536
 capacity_mem_par_l2_P4: mem[10,6] <= 1536
 capacity_mem_par_l2_P4: mem[10,12] <= 1536
 capacity_mem_par_l2_P4: mem[10,2] <= 1536
Bounds
 frac[7,3] <= 1
 frac[6,9] <= 1
 frac[7,12] <= 1
 frac[1,6] <= 1
 frac[8,5] <= 1
 frac[5,8] <= 1
 frac[10,8] <= 1
 frac[8,15] <= 1
 frac[3,2] <= 1
 frac[9,14] <= 1
 frac[8,2] <= 1
 frac[5,11] <= 1
 frac[4,5] <= 1
 frac[9,3] <= 1
 frac[3,12] <= 1
 frac[8,12] <= 1
 frac[2,11] <= 1
 frac[9,9] <= 1
 frac[5,14] <= 1
 frac[10,14] <= 1
 frac[6,13] <= 1
 frac[7,8] <= 1
 frac[0,14] <= 1
 frac[8,9] <= 1
 frac[9,4] <= 1
 frac[10,3] <= 1
 frac[7,2] <= 1
 frac[6,14] <= 1
 frac[7,15] <= 1
 frac[1,5] <= 1
 frac[8,6] <= 1
 frac[10,9] <= 1
 frac[6,4] <= 1
 frac[10,4] <= 1
 frac[6,11] <= 1
 frac[9,13] <= 1
 frac[8,3] <= 1
 frac[4,6] <= 1
 frac[9,2] <= 1
 frac[7,4] <= 1
 frac[0,2] <= 1
 frac[3,15] <= 1
 frac[4,8] <= 1
 frac[2,8] <= 1
 frac[9,8] <= 1
 frac[10,15] <= 1
 frac[6,2] <= 1
 frac[9,11] <= 1
 frac[6,15] <= 1
 frac[7,14] <= 1
 frac[1,4] <= 1
 frac[9,6] <= 1
 frac[6,5] <= 1
 frac[5,3] <= 1
 frac[10,5] <= 1
 frac[6,8] <= 1
 frac[7,13] <= 1
 frac[3,4] <= 1
 frac[9,12] <= 1
 frac[8,4] <= 1
 frac[5,9] <= 1
 frac[10,11] <= 1
 frac[6,6] <= 1
 frac[10,6] <= 1
 frac[0,3] <= 1
 frac[3,14] <= 1
 frac[8,14] <= 1
 frac[4,9] <= 1
 frac[3,3] <= 1
 frac[2,9] <= 1
 frac[9,15] <= 1
 frac[5,12] <= 1
 frac[4,4] <= 1
 frac[10,12] <= 1
 frac[6,3] <= 1
 frac[7,10] <= 1
 frac[3,13] <= 1
 frac[8,11] <= 1
 frac[4,14] <= 1
 frac[2,10] <= 1
 frac[5,15] <= 1
 frac[6,12] <= 1
 frac[7,9] <= 1
 frac[3,8] <= 1
 frac[8,8] <= 1
 frac[4,3] <= 1
 frac[9,5] <= 1
 frac[5,2] <= 1
 frac[10,2] <= 1
 cols_7_3 <= 131072
 cols_6_9 <= 131072
 cols_7_12 <= 131072
 cols_8_5 <= 131072
 cols_5_8 <= 131072
 cols_10_8 <= 131072
 cols_8_15 <= 131072
 cols_9_14 <= 131072
 cols_8_2 <= 131072
 cols_5_11 <= 131072
 cols_4_5 <= 131072
 cols_9_3 <= 131072
 cols_8_12 <= 131072
 cols_9_9 <= 131072
 cols_5_14 <= 131072
 cols_10_14 <= 131072
 cols_6_13 <= 131072
 cols_7_8 <= 131072
 cols_8_9 <= 131072
 cols_9_4 <= 131072
 cols_10_3 <= 131072
 cols_7_2 <= 131072
 cols_6_14 <= 131072
 cols_7_15 <= 131072
 cols_8_6 <= 131072
 cols_10_9 <= 131072
 cols_6_4 <= 131072
 cols_10_4 <= 131072
 cols_6_11 <= 131072
 cols_9_13 <= 131072
 cols_8_3 <= 131072
 cols_4_6 <= 131072
 cols_9_2 <= 131072
 cols_7_4 <= 131072
 cols_4_8 <= 131072
 cols_9_8 <= 131072
 cols_10_15 <= 131072
 cols_6_2 <= 131072
 cols_9_11 <= 131072
 cols_6_15 <= 131072
 cols_7_14 <= 131072
 cols_9_6 <= 131072
 cols_6_5 <= 131072
 cols_5_3 <= 131072
 cols_10_5 <= 131072
 cols_6_8 <= 131072
 cols_7_13 <= 131072
 cols_9_12 <= 131072
 cols_8_4 <= 131072
 cols_5_9 <= 131072
 cols_10_11 <= 131072
 cols_6_6 <= 131072
 cols_10_6 <= 131072
 cols_8_14 <= 131072
 cols_4_9 <= 131072
 cols_9_15 <= 131072
 cols_5_12 <= 131072
 cols_4_4 <= 131072
 cols_10_12 <= 131072
 cols_6_3 <= 131072
 cols_7_10 <= 131072
 cols_8_11 <= 131072
 cols_4_14 <= 131072
 cols_5_15 <= 131072
 cols_6_12 <= 131072
 cols_7_9 <= 131072
 cols_8_8 <= 131072
 cols_4_3 <= 131072
 cols_9_5 <= 131072
 cols_5_2 <= 131072
 cols_10_2 <= 131072
 mem_tot_{CPU13,CPU4,CPU6,CPU30,CPU9,CPU16,CPU22,CPU32} <= 262144
 normalized_mem_tot_{CPU13,CPU4,CPU6,CPU30,CPU9,CPU16,CPU22,CPU32} <= 1
 normalized_rows_tot_{CPU13,CPU4,CPU6,CPU30,CPU9,CPU16,CPU22,CPU32} <= 1
 mem_tot_{CPU29,CPU8,CPU7,CPU21,CPU2,CPU3,CPU18,CPU5} <= 262144
 normalized_mem_tot_{CPU29,CPU8,CPU7,CPU21,CPU2,CPU3,CPU18,CPU5} <= 1
 normalized_rows_tot_{CPU29,CPU8,CPU7,CPU21,CPU2,CPU3,CPU18,CPU5} <= 1
 mem_tot_{CPU28,CPU19,CPU25,CPU1,CPU10,CPU17,CPU11,CPU14} <= 262144
 normalized_mem_tot_{CPU28,CPU19,CPU25,CPU1,CPU10,CPU17,CPU11,CPU14} <= 1
 normalized_rows_tot_{CPU28,CPU19,CPU25,CPU1,CPU10,CPU17,CPU11,CPU14} <= 1
 mem_tot_{CPU31,CPU12,CPU24,CPU27,CPU26,CPU23,CPU20,CPU15} <= 262144
 normalized_mem_tot_{CPU31,CPU12,CPU24,CPU27,CPU26,CPU23,CPU20,CPU15} <= 1
 normalized_rows_tot_{CPU31,CPU12,CPU24,CPU27,CPU26,CPU23,CPU20,CPU15} <= 1
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
 cols_helper_7_3[0] cols_helper_7_3[1] cols_helper_7_3[2]
 cols_helper_7_3[3] cols_helper_7_3[4] cols_helper_7_3[5]
 cols_helper_7_3[6] cols_helper_7_3[7] cols_helper_7_3[8]
 cols_helper_7_3[9] cols_helper_7_3[10] cols_helper_7_3[11]
 cols_helper_7_3[12] cols_helper_7_3[13] cols_helper_7_3[14]
 cols_helper_7_3[15] cols_helper_7_3[16] cols_helper_6_9[0]
 cols_helper_6_9[1] cols_helper_6_9[2] cols_helper_6_9[3]
 cols_helper_6_9[4] cols_helper_6_9[5] cols_helper_6_9[6]
 cols_helper_6_9[7] cols_helper_6_9[8] cols_helper_6_9[9]
 cols_helper_6_9[10] cols_helper_6_9[11] cols_helper_6_9[12]
 cols_helper_6_9[13] cols_helper_6_9[14] cols_helper_6_9[15]
 cols_helper_6_9[16] cols_helper_7_12[0] cols_helper_7_12[1]
 cols_helper_7_12[2] cols_helper_7_12[3] cols_helper_7_12[4]
 cols_helper_7_12[5] cols_helper_7_12[6] cols_helper_7_12[7]
 cols_helper_7_12[8] cols_helper_7_12[9] cols_helper_7_12[10]
 cols_helper_7_12[11] cols_helper_7_12[12] cols_helper_7_12[13]
 cols_helper_7_12[14] cols_helper_7_12[15] cols_helper_7_12[16]
 cols_helper_8_5[0] cols_helper_8_5[1] cols_helper_8_5[2]
 cols_helper_8_5[3] cols_helper_8_5[4] cols_helper_8_5[5]
 cols_helper_8_5[6] cols_helper_8_5[7] cols_helper_8_5[8]
 cols_helper_8_5[9] cols_helper_8_5[10] cols_helper_8_5[11]
 cols_helper_8_5[12] cols_helper_8_5[13] cols_helper_8_5[14]
 cols_helper_8_5[15] cols_helper_8_5[16] cols_helper_5_8[0]
 cols_helper_5_8[1] cols_helper_5_8[2] cols_helper_5_8[3]
 cols_helper_5_8[4] cols_helper_5_8[5] cols_helper_5_8[6]
 cols_helper_5_8[7] cols_helper_5_8[8] cols_helper_5_8[9]
 cols_helper_5_8[10] cols_helper_5_8[11] cols_helper_5_8[12]
 cols_helper_5_8[13] cols_helper_5_8[14] cols_helper_5_8[15]
 cols_helper_5_8[16] cols_helper_10_8[0] cols_helper_10_8[1]
 cols_helper_10_8[2] cols_helper_10_8[3] cols_helper_10_8[4]
 cols_helper_10_8[5] cols_helper_10_8[6] cols_helper_10_8[7]
 cols_helper_10_8[8] cols_helper_10_8[9] cols_helper_10_8[10]
 cols_helper_10_8[11] cols_helper_10_8[12] cols_helper_10_8[13]
 cols_helper_10_8[14] cols_helper_10_8[15] cols_helper_10_8[16]
 cols_helper_8_15[0] cols_helper_8_15[1] cols_helper_8_15[2]
 cols_helper_8_15[3] cols_helper_8_15[4] cols_helper_8_15[5]
 cols_helper_8_15[6] cols_helper_8_15[7] cols_helper_8_15[8]
 cols_helper_8_15[9] cols_helper_8_15[10] cols_helper_8_15[11]
 cols_helper_8_15[12] cols_helper_8_15[13] cols_helper_8_15[14]
 cols_helper_8_15[15] cols_helper_8_15[16] cols_helper_9_14[0]
 cols_helper_9_14[1] cols_helper_9_14[2] cols_helper_9_14[3]
 cols_helper_9_14[4] cols_helper_9_14[5] cols_helper_9_14[6]
 cols_helper_9_14[7] cols_helper_9_14[8] cols_helper_9_14[9]
 cols_helper_9_14[10] cols_helper_9_14[11] cols_helper_9_14[12]
 cols_helper_9_14[13] cols_helper_9_14[14] cols_helper_9_14[15]
 cols_helper_9_14[16] cols_helper_8_2[0] cols_helper_8_2[1]
 cols_helper_8_2[2] cols_helper_8_2[3] cols_helper_8_2[4]
 cols_helper_8_2[5] cols_helper_8_2[6] cols_helper_8_2[7]
 cols_helper_8_2[8] cols_helper_8_2[9] cols_helper_8_2[10]
 cols_helper_8_2[11] cols_helper_8_2[12] cols_helper_8_2[13]
 cols_helper_8_2[14] cols_helper_8_2[15] cols_helper_8_2[16]
 cols_helper_5_11[0] cols_helper_5_11[1] cols_helper_5_11[2]
 cols_helper_5_11[3] cols_helper_5_11[4] cols_helper_5_11[5]
 cols_helper_5_11[6] cols_helper_5_11[7] cols_helper_5_11[8]
 cols_helper_5_11[9] cols_helper_5_11[10] cols_helper_5_11[11]
 cols_helper_5_11[12] cols_helper_5_11[13] cols_helper_5_11[14]
 cols_helper_5_11[15] cols_helper_5_11[16] cols_helper_4_5[0]
 cols_helper_4_5[1] cols_helper_4_5[2] cols_helper_4_5[3]
 cols_helper_4_5[4] cols_helper_4_5[5] cols_helper_4_5[6]
 cols_helper_4_5[7] cols_helper_4_5[8] cols_helper_4_5[9]
 cols_helper_4_5[10] cols_helper_4_5[11] cols_helper_4_5[12]
 cols_helper_4_5[13] cols_helper_4_5[14] cols_helper_4_5[15]
 cols_helper_4_5[16] cols_helper_9_3[0] cols_helper_9_3[1]
 cols_helper_9_3[2] cols_helper_9_3[3] cols_helper_9_3[4]
 cols_helper_9_3[5] cols_helper_9_3[6] cols_helper_9_3[7]
 cols_helper_9_3[8] cols_helper_9_3[9] cols_helper_9_3[10]
 cols_helper_9_3[11] cols_helper_9_3[12] cols_helper_9_3[13]
 cols_helper_9_3[14] cols_helper_9_3[15] cols_helper_9_3[16]
 cols_helper_8_12[0] cols_helper_8_12[1] cols_helper_8_12[2]
 cols_helper_8_12[3] cols_helper_8_12[4] cols_helper_8_12[5]
 cols_helper_8_12[6] cols_helper_8_12[7] cols_helper_8_12[8]
 cols_helper_8_12[9] cols_helper_8_12[10] cols_helper_8_12[11]
 cols_helper_8_12[12] cols_helper_8_12[13] cols_helper_8_12[14]
 cols_helper_8_12[15] cols_helper_8_12[16] cols_helper_9_9[0]
 cols_helper_9_9[1] cols_helper_9_9[2] cols_helper_9_9[3]
 cols_helper_9_9[4] cols_helper_9_9[5] cols_helper_9_9[6]
 cols_helper_9_9[7] cols_helper_9_9[8] cols_helper_9_9[9]
 cols_helper_9_9[10] cols_helper_9_9[11] cols_helper_9_9[12]
 cols_helper_9_9[13] cols_helper_9_9[14] cols_helper_9_9[15]
 cols_helper_9_9[16] cols_helper_5_14[0] cols_helper_5_14[1]
 cols_helper_5_14[2] cols_helper_5_14[3] cols_helper_5_14[4]
 cols_helper_5_14[5] cols_helper_5_14[6] cols_helper_5_14[7]
 cols_helper_5_14[8] cols_helper_5_14[9] cols_helper_5_14[10]
 cols_helper_5_14[11] cols_helper_5_14[12] cols_helper_5_14[13]
 cols_helper_5_14[14] cols_helper_5_14[15] cols_helper_5_14[16]
 cols_helper_10_14[0] cols_helper_10_14[1] cols_helper_10_14[2]
 cols_helper_10_14[3] cols_helper_10_14[4] cols_helper_10_14[5]
 cols_helper_10_14[6] cols_helper_10_14[7] cols_helper_10_14[8]
 cols_helper_10_14[9] cols_helper_10_14[10] cols_helper_10_14[11]
 cols_helper_10_14[12] cols_helper_10_14[13] cols_helper_10_14[14]
 cols_helper_10_14[15] cols_helper_10_14[16] cols_helper_6_13[0]
 cols_helper_6_13[1] cols_helper_6_13[2] cols_helper_6_13[3]
 cols_helper_6_13[4] cols_helper_6_13[5] cols_helper_6_13[6]
 cols_helper_6_13[7] cols_helper_6_13[8] cols_helper_6_13[9]
 cols_helper_6_13[10] cols_helper_6_13[11] cols_helper_6_13[12]
 cols_helper_6_13[13] cols_helper_6_13[14] cols_helper_6_13[15]
 cols_helper_6_13[16] cols_helper_7_8[0] cols_helper_7_8[1]
 cols_helper_7_8[2] cols_helper_7_8[3] cols_helper_7_8[4]
 cols_helper_7_8[5] cols_helper_7_8[6] cols_helper_7_8[7]
 cols_helper_7_8[8] cols_helper_7_8[9] cols_helper_7_8[10]
 cols_helper_7_8[11] cols_helper_7_8[12] cols_helper_7_8[13]
 cols_helper_7_8[14] cols_helper_7_8[15] cols_helper_7_8[16]
 cols_helper_8_9[0] cols_helper_8_9[1] cols_helper_8_9[2]
 cols_helper_8_9[3] cols_helper_8_9[4] cols_helper_8_9[5]
 cols_helper_8_9[6] cols_helper_8_9[7] cols_helper_8_9[8]
 cols_helper_8_9[9] cols_helper_8_9[10] cols_helper_8_9[11]
 cols_helper_8_9[12] cols_helper_8_9[13] cols_helper_8_9[14]
 cols_helper_8_9[15] cols_helper_8_9[16] cols_helper_9_4[0]
 cols_helper_9_4[1] cols_helper_9_4[2] cols_helper_9_4[3]
 cols_helper_9_4[4] cols_helper_9_4[5] cols_helper_9_4[6]
 cols_helper_9_4[7] cols_helper_9_4[8] cols_helper_9_4[9]
 cols_helper_9_4[10] cols_helper_9_4[11] cols_helper_9_4[12]
 cols_helper_9_4[13] cols_helper_9_4[14] cols_helper_9_4[15]
 cols_helper_9_4[16] cols_helper_10_3[0] cols_helper_10_3[1]
 cols_helper_10_3[2] cols_helper_10_3[3] cols_helper_10_3[4]
 cols_helper_10_3[5] cols_helper_10_3[6] cols_helper_10_3[7]
 cols_helper_10_3[8] cols_helper_10_3[9] cols_helper_10_3[10]
 cols_helper_10_3[11] cols_helper_10_3[12] cols_helper_10_3[13]
 cols_helper_10_3[14] cols_helper_10_3[15] cols_helper_10_3[16]
 cols_helper_7_2[0] cols_helper_7_2[1] cols_helper_7_2[2]
 cols_helper_7_2[3] cols_helper_7_2[4] cols_helper_7_2[5]
 cols_helper_7_2[6] cols_helper_7_2[7] cols_helper_7_2[8]
 cols_helper_7_2[9] cols_helper_7_2[10] cols_helper_7_2[11]
 cols_helper_7_2[12] cols_helper_7_2[13] cols_helper_7_2[14]
 cols_helper_7_2[15] cols_helper_7_2[16] cols_helper_6_14[0]
 cols_helper_6_14[1] cols_helper_6_14[2] cols_helper_6_14[3]
 cols_helper_6_14[4] cols_helper_6_14[5] cols_helper_6_14[6]
 cols_helper_6_14[7] cols_helper_6_14[8] cols_helper_6_14[9]
 cols_helper_6_14[10] cols_helper_6_14[11] cols_helper_6_14[12]
 cols_helper_6_14[13] cols_helper_6_14[14] cols_helper_6_14[15]
 cols_helper_6_14[16] cols_helper_7_15[0] cols_helper_7_15[1]
 cols_helper_7_15[2] cols_helper_7_15[3] cols_helper_7_15[4]
 cols_helper_7_15[5] cols_helper_7_15[6] cols_helper_7_15[7]
 cols_helper_7_15[8] cols_helper_7_15[9] cols_helper_7_15[10]
 cols_helper_7_15[11] cols_helper_7_15[12] cols_helper_7_15[13]
 cols_helper_7_15[14] cols_helper_7_15[15] cols_helper_7_15[16]
 cols_helper_8_6[0] cols_helper_8_6[1] cols_helper_8_6[2]
 cols_helper_8_6[3] cols_helper_8_6[4] cols_helper_8_6[5]
 cols_helper_8_6[6] cols_helper_8_6[7] cols_helper_8_6[8]
 cols_helper_8_6[9] cols_helper_8_6[10] cols_helper_8_6[11]
 cols_helper_8_6[12] cols_helper_8_6[13] cols_helper_8_6[14]
 cols_helper_8_6[15] cols_helper_8_6[16] cols_helper_10_9[0]
 cols_helper_10_9[1] cols_helper_10_9[2] cols_helper_10_9[3]
 cols_helper_10_9[4] cols_helper_10_9[5] cols_helper_10_9[6]
 cols_helper_10_9[7] cols_helper_10_9[8] cols_helper_10_9[9]
 cols_helper_10_9[10] cols_helper_10_9[11] cols_helper_10_9[12]
 cols_helper_10_9[13] cols_helper_10_9[14] cols_helper_10_9[15]
 cols_helper_10_9[16] cols_helper_6_4[0] cols_helper_6_4[1]
 cols_helper_6_4[2] cols_helper_6_4[3] cols_helper_6_4[4]
 cols_helper_6_4[5] cols_helper_6_4[6] cols_helper_6_4[7]
 cols_helper_6_4[8] cols_helper_6_4[9] cols_helper_6_4[10]
 cols_helper_6_4[11] cols_helper_6_4[12] cols_helper_6_4[13]
 cols_helper_6_4[14] cols_helper_6_4[15] cols_helper_6_4[16]
 cols_helper_10_4[0] cols_helper_10_4[1] cols_helper_10_4[2]
 cols_helper_10_4[3] cols_helper_10_4[4] cols_helper_10_4[5]
 cols_helper_10_4[6] cols_helper_10_4[7] cols_helper_10_4[8]
 cols_helper_10_4[9] cols_helper_10_4[10] cols_helper_10_4[11]
 cols_helper_10_4[12] cols_helper_10_4[13] cols_helper_10_4[14]
 cols_helper_10_4[15] cols_helper_10_4[16] cols_helper_6_11[0]
 cols_helper_6_11[1] cols_helper_6_11[2] cols_helper_6_11[3]
 cols_helper_6_11[4] cols_helper_6_11[5] cols_helper_6_11[6]
 cols_helper_6_11[7] cols_helper_6_11[8] cols_helper_6_11[9]
 cols_helper_6_11[10] cols_helper_6_11[11] cols_helper_6_11[12]
 cols_helper_6_11[13] cols_helper_6_11[14] cols_helper_6_11[15]
 cols_helper_6_11[16] cols_helper_9_13[0] cols_helper_9_13[1]
 cols_helper_9_13[2] cols_helper_9_13[3] cols_helper_9_13[4]
 cols_helper_9_13[5] cols_helper_9_13[6] cols_helper_9_13[7]
 cols_helper_9_13[8] cols_helper_9_13[9] cols_helper_9_13[10]
 cols_helper_9_13[11] cols_helper_9_13[12] cols_helper_9_13[13]
 cols_helper_9_13[14] cols_helper_9_13[15] cols_helper_9_13[16]
 cols_helper_8_3[0] cols_helper_8_3[1] cols_helper_8_3[2]
 cols_helper_8_3[3] cols_helper_8_3[4] cols_helper_8_3[5]
 cols_helper_8_3[6] cols_helper_8_3[7] cols_helper_8_3[8]
 cols_helper_8_3[9] cols_helper_8_3[10] cols_helper_8_3[11]
 cols_helper_8_3[12] cols_helper_8_3[13] cols_helper_8_3[14]
 cols_helper_8_3[15] cols_helper_8_3[16] cols_helper_4_6[0]
 cols_helper_4_6[1] cols_helper_4_6[2] cols_helper_4_6[3]
 cols_helper_4_6[4] cols_helper_4_6[5] cols_helper_4_6[6]
 cols_helper_4_6[7] cols_helper_4_6[8] cols_helper_4_6[9]
 cols_helper_4_6[10] cols_helper_4_6[11] cols_helper_4_6[12]
 cols_helper_4_6[13] cols_helper_4_6[14] cols_helper_4_6[15]
 cols_helper_4_6[16] cols_helper_9_2[0] cols_helper_9_2[1]
 cols_helper_9_2[2] cols_helper_9_2[3] cols_helper_9_2[4]
 cols_helper_9_2[5] cols_helper_9_2[6] cols_helper_9_2[7]
 cols_helper_9_2[8] cols_helper_9_2[9] cols_helper_9_2[10]
 cols_helper_9_2[11] cols_helper_9_2[12] cols_helper_9_2[13]
 cols_helper_9_2[14] cols_helper_9_2[15] cols_helper_9_2[16]
 cols_helper_7_4[0] cols_helper_7_4[1] cols_helper_7_4[2]
 cols_helper_7_4[3] cols_helper_7_4[4] cols_helper_7_4[5]
 cols_helper_7_4[6] cols_helper_7_4[7] cols_helper_7_4[8]
 cols_helper_7_4[9] cols_helper_7_4[10] cols_helper_7_4[11]
 cols_helper_7_4[12] cols_helper_7_4[13] cols_helper_7_4[14]
 cols_helper_7_4[15] cols_helper_7_4[16] cols_helper_4_8[0]
 cols_helper_4_8[1] cols_helper_4_8[2] cols_helper_4_8[3]
 cols_helper_4_8[4] cols_helper_4_8[5] cols_helper_4_8[6]
 cols_helper_4_8[7] cols_helper_4_8[8] cols_helper_4_8[9]
 cols_helper_4_8[10] cols_helper_4_8[11] cols_helper_4_8[12]
 cols_helper_4_8[13] cols_helper_4_8[14] cols_helper_4_8[15]
 cols_helper_4_8[16] cols_helper_9_8[0] cols_helper_9_8[1]
 cols_helper_9_8[2] cols_helper_9_8[3] cols_helper_9_8[4]
 cols_helper_9_8[5] cols_helper_9_8[6] cols_helper_9_8[7]
 cols_helper_9_8[8] cols_helper_9_8[9] cols_helper_9_8[10]
 cols_helper_9_8[11] cols_helper_9_8[12] cols_helper_9_8[13]
 cols_helper_9_8[14] cols_helper_9_8[15] cols_helper_9_8[16]
 cols_helper_10_15[0] cols_helper_10_15[1] cols_helper_10_15[2]
 cols_helper_10_15[3] cols_helper_10_15[4] cols_helper_10_15[5]
 cols_helper_10_15[6] cols_helper_10_15[7] cols_helper_10_15[8]
 cols_helper_10_15[9] cols_helper_10_15[10] cols_helper_10_15[11]
 cols_helper_10_15[12] cols_helper_10_15[13] cols_helper_10_15[14]
 cols_helper_10_15[15] cols_helper_10_15[16] cols_helper_6_2[0]
 cols_helper_6_2[1] cols_helper_6_2[2] cols_helper_6_2[3]
 cols_helper_6_2[4] cols_helper_6_2[5] cols_helper_6_2[6]
 cols_helper_6_2[7] cols_helper_6_2[8] cols_helper_6_2[9]
 cols_helper_6_2[10] cols_helper_6_2[11] cols_helper_6_2[12]
 cols_helper_6_2[13] cols_helper_6_2[14] cols_helper_6_2[15]
 cols_helper_6_2[16] cols_helper_9_11[0] cols_helper_9_11[1]
 cols_helper_9_11[2] cols_helper_9_11[3] cols_helper_9_11[4]
 cols_helper_9_11[5] cols_helper_9_11[6] cols_helper_9_11[7]
 cols_helper_9_11[8] cols_helper_9_11[9] cols_helper_9_11[10]
 cols_helper_9_11[11] cols_helper_9_11[12] cols_helper_9_11[13]
 cols_helper_9_11[14] cols_helper_9_11[15] cols_helper_9_11[16]
 cols_helper_6_15[0] cols_helper_6_15[1] cols_helper_6_15[2]
 cols_helper_6_15[3] cols_helper_6_15[4] cols_helper_6_15[5]
 cols_helper_6_15[6] cols_helper_6_15[7] cols_helper_6_15[8]
 cols_helper_6_15[9] cols_helper_6_15[10] cols_helper_6_15[11]
 cols_helper_6_15[12] cols_helper_6_15[13] cols_helper_6_15[14]
 cols_helper_6_15[15] cols_helper_6_15[16] cols_helper_7_14[0]
 cols_helper_7_14[1] cols_helper_7_14[2] cols_helper_7_14[3]
 cols_helper_7_14[4] cols_helper_7_14[5] cols_helper_7_14[6]
 cols_helper_7_14[7] cols_helper_7_14[8] cols_helper_7_14[9]
 cols_helper_7_14[10] cols_helper_7_14[11] cols_helper_7_14[12]
 cols_helper_7_14[13] cols_helper_7_14[14] cols_helper_7_14[15]
 cols_helper_7_14[16] cols_helper_9_6[0] cols_helper_9_6[1]
 cols_helper_9_6[2] cols_helper_9_6[3] cols_helper_9_6[4]
 cols_helper_9_6[5] cols_helper_9_6[6] cols_helper_9_6[7]
 cols_helper_9_6[8] cols_helper_9_6[9] cols_helper_9_6[10]
 cols_helper_9_6[11] cols_helper_9_6[12] cols_helper_9_6[13]
 cols_helper_9_6[14] cols_helper_9_6[15] cols_helper_9_6[16]
 cols_helper_6_5[0] cols_helper_6_5[1] cols_helper_6_5[2]
 cols_helper_6_5[3] cols_helper_6_5[4] cols_helper_6_5[5]
 cols_helper_6_5[6] cols_helper_6_5[7] cols_helper_6_5[8]
 cols_helper_6_5[9] cols_helper_6_5[10] cols_helper_6_5[11]
 cols_helper_6_5[12] cols_helper_6_5[13] cols_helper_6_5[14]
 cols_helper_6_5[15] cols_helper_6_5[16] cols_helper_5_3[0]
 cols_helper_5_3[1] cols_helper_5_3[2] cols_helper_5_3[3]
 cols_helper_5_3[4] cols_helper_5_3[5] cols_helper_5_3[6]
 cols_helper_5_3[7] cols_helper_5_3[8] cols_helper_5_3[9]
 cols_helper_5_3[10] cols_helper_5_3[11] cols_helper_5_3[12]
 cols_helper_5_3[13] cols_helper_5_3[14] cols_helper_5_3[15]
 cols_helper_5_3[16] cols_helper_10_5[0] cols_helper_10_5[1]
 cols_helper_10_5[2] cols_helper_10_5[3] cols_helper_10_5[4]
 cols_helper_10_5[5] cols_helper_10_5[6] cols_helper_10_5[7]
 cols_helper_10_5[8] cols_helper_10_5[9] cols_helper_10_5[10]
 cols_helper_10_5[11] cols_helper_10_5[12] cols_helper_10_5[13]
 cols_helper_10_5[14] cols_helper_10_5[15] cols_helper_10_5[16]
 cols_helper_6_8[0] cols_helper_6_8[1] cols_helper_6_8[2]
 cols_helper_6_8[3] cols_helper_6_8[4] cols_helper_6_8[5]
 cols_helper_6_8[6] cols_helper_6_8[7] cols_helper_6_8[8]
 cols_helper_6_8[9] cols_helper_6_8[10] cols_helper_6_8[11]
 cols_helper_6_8[12] cols_helper_6_8[13] cols_helper_6_8[14]
 cols_helper_6_8[15] cols_helper_6_8[16] cols_helper_7_13[0]
 cols_helper_7_13[1] cols_helper_7_13[2] cols_helper_7_13[3]
 cols_helper_7_13[4] cols_helper_7_13[5] cols_helper_7_13[6]
 cols_helper_7_13[7] cols_helper_7_13[8] cols_helper_7_13[9]
 cols_helper_7_13[10] cols_helper_7_13[11] cols_helper_7_13[12]
 cols_helper_7_13[13] cols_helper_7_13[14] cols_helper_7_13[15]
 cols_helper_7_13[16] cols_helper_9_12[0] cols_helper_9_12[1]
 cols_helper_9_12[2] cols_helper_9_12[3] cols_helper_9_12[4]
 cols_helper_9_12[5] cols_helper_9_12[6] cols_helper_9_12[7]
 cols_helper_9_12[8] cols_helper_9_12[9] cols_helper_9_12[10]
 cols_helper_9_12[11] cols_helper_9_12[12] cols_helper_9_12[13]
 cols_helper_9_12[14] cols_helper_9_12[15] cols_helper_9_12[16]
 cols_helper_8_4[0] cols_helper_8_4[1] cols_helper_8_4[2]
 cols_helper_8_4[3] cols_helper_8_4[4] cols_helper_8_4[5]
 cols_helper_8_4[6] cols_helper_8_4[7] cols_helper_8_4[8]
 cols_helper_8_4[9] cols_helper_8_4[10] cols_helper_8_4[11]
 cols_helper_8_4[12] cols_helper_8_4[13] cols_helper_8_4[14]
 cols_helper_8_4[15] cols_helper_8_4[16] cols_helper_5_9[0]
 cols_helper_5_9[1] cols_helper_5_9[2] cols_helper_5_9[3]
 cols_helper_5_9[4] cols_helper_5_9[5] cols_helper_5_9[6]
 cols_helper_5_9[7] cols_helper_5_9[8] cols_helper_5_9[9]
 cols_helper_5_9[10] cols_helper_5_9[11] cols_helper_5_9[12]
 cols_helper_5_9[13] cols_helper_5_9[14] cols_helper_5_9[15]
 cols_helper_5_9[16] cols_helper_10_11[0] cols_helper_10_11[1]
 cols_helper_10_11[2] cols_helper_10_11[3] cols_helper_10_11[4]
 cols_helper_10_11[5] cols_helper_10_11[6] cols_helper_10_11[7]
 cols_helper_10_11[8] cols_helper_10_11[9] cols_helper_10_11[10]
 cols_helper_10_11[11] cols_helper_10_11[12] cols_helper_10_11[13]
 cols_helper_10_11[14] cols_helper_10_11[15] cols_helper_10_11[16]
 cols_helper_6_6[0] cols_helper_6_6[1] cols_helper_6_6[2]
 cols_helper_6_6[3] cols_helper_6_6[4] cols_helper_6_6[5]
 cols_helper_6_6[6] cols_helper_6_6[7] cols_helper_6_6[8]
 cols_helper_6_6[9] cols_helper_6_6[10] cols_helper_6_6[11]
 cols_helper_6_6[12] cols_helper_6_6[13] cols_helper_6_6[14]
 cols_helper_6_6[15] cols_helper_6_6[16] cols_helper_10_6[0]
 cols_helper_10_6[1] cols_helper_10_6[2] cols_helper_10_6[3]
 cols_helper_10_6[4] cols_helper_10_6[5] cols_helper_10_6[6]
 cols_helper_10_6[7] cols_helper_10_6[8] cols_helper_10_6[9]
 cols_helper_10_6[10] cols_helper_10_6[11] cols_helper_10_6[12]
 cols_helper_10_6[13] cols_helper_10_6[14] cols_helper_10_6[15]
 cols_helper_10_6[16] cols_helper_8_14[0] cols_helper_8_14[1]
 cols_helper_8_14[2] cols_helper_8_14[3] cols_helper_8_14[4]
 cols_helper_8_14[5] cols_helper_8_14[6] cols_helper_8_14[7]
 cols_helper_8_14[8] cols_helper_8_14[9] cols_helper_8_14[10]
 cols_helper_8_14[11] cols_helper_8_14[12] cols_helper_8_14[13]
 cols_helper_8_14[14] cols_helper_8_14[15] cols_helper_8_14[16]
 cols_helper_4_9[0] cols_helper_4_9[1] cols_helper_4_9[2]
 cols_helper_4_9[3] cols_helper_4_9[4] cols_helper_4_9[5]
 cols_helper_4_9[6] cols_helper_4_9[7] cols_helper_4_9[8]
 cols_helper_4_9[9] cols_helper_4_9[10] cols_helper_4_9[11]
 cols_helper_4_9[12] cols_helper_4_9[13] cols_helper_4_9[14]
 cols_helper_4_9[15] cols_helper_4_9[16] cols_helper_9_15[0]
 cols_helper_9_15[1] cols_helper_9_15[2] cols_helper_9_15[3]
 cols_helper_9_15[4] cols_helper_9_15[5] cols_helper_9_15[6]
 cols_helper_9_15[7] cols_helper_9_15[8] cols_helper_9_15[9]
 cols_helper_9_15[10] cols_helper_9_15[11] cols_helper_9_15[12]
 cols_helper_9_15[13] cols_helper_9_15[14] cols_helper_9_15[15]
 cols_helper_9_15[16] cols_helper_5_12[0] cols_helper_5_12[1]
 cols_helper_5_12[2] cols_helper_5_12[3] cols_helper_5_12[4]
 cols_helper_5_12[5] cols_helper_5_12[6] cols_helper_5_12[7]
 cols_helper_5_12[8] cols_helper_5_12[9] cols_helper_5_12[10]
 cols_helper_5_12[11] cols_helper_5_12[12] cols_helper_5_12[13]
 cols_helper_5_12[14] cols_helper_5_12[15] cols_helper_5_12[16]
 cols_helper_4_4[0] cols_helper_4_4[1] cols_helper_4_4[2]
 cols_helper_4_4[3] cols_helper_4_4[4] cols_helper_4_4[5]
 cols_helper_4_4[6] cols_helper_4_4[7] cols_helper_4_4[8]
 cols_helper_4_4[9] cols_helper_4_4[10] cols_helper_4_4[11]
 cols_helper_4_4[12] cols_helper_4_4[13] cols_helper_4_4[14]
 cols_helper_4_4[15] cols_helper_4_4[16] cols_helper_10_12[0]
 cols_helper_10_12[1] cols_helper_10_12[2] cols_helper_10_12[3]
 cols_helper_10_12[4] cols_helper_10_12[5] cols_helper_10_12[6]
 cols_helper_10_12[7] cols_helper_10_12[8] cols_helper_10_12[9]
 cols_helper_10_12[10] cols_helper_10_12[11] cols_helper_10_12[12]
 cols_helper_10_12[13] cols_helper_10_12[14] cols_helper_10_12[15]
 cols_helper_10_12[16] cols_helper_6_3[0] cols_helper_6_3[1]
 cols_helper_6_3[2] cols_helper_6_3[3] cols_helper_6_3[4]
 cols_helper_6_3[5] cols_helper_6_3[6] cols_helper_6_3[7]
 cols_helper_6_3[8] cols_helper_6_3[9] cols_helper_6_3[10]
 cols_helper_6_3[11] cols_helper_6_3[12] cols_helper_6_3[13]
 cols_helper_6_3[14] cols_helper_6_3[15] cols_helper_6_3[16]
 cols_helper_7_10[0] cols_helper_7_10[1] cols_helper_7_10[2]
 cols_helper_7_10[3] cols_helper_7_10[4] cols_helper_7_10[5]
 cols_helper_7_10[6] cols_helper_7_10[7] cols_helper_7_10[8]
 cols_helper_7_10[9] cols_helper_7_10[10] cols_helper_7_10[11]
 cols_helper_7_10[12] cols_helper_7_10[13] cols_helper_7_10[14]
 cols_helper_7_10[15] cols_helper_7_10[16] cols_helper_8_11[0]
 cols_helper_8_11[1] cols_helper_8_11[2] cols_helper_8_11[3]
 cols_helper_8_11[4] cols_helper_8_11[5] cols_helper_8_11[6]
 cols_helper_8_11[7] cols_helper_8_11[8] cols_helper_8_11[9]
 cols_helper_8_11[10] cols_helper_8_11[11] cols_helper_8_11[12]
 cols_helper_8_11[13] cols_helper_8_11[14] cols_helper_8_11[15]
 cols_helper_8_11[16] cols_helper_4_14[0] cols_helper_4_14[1]
 cols_helper_4_14[2] cols_helper_4_14[3] cols_helper_4_14[4]
 cols_helper_4_14[5] cols_helper_4_14[6] cols_helper_4_14[7]
 cols_helper_4_14[8] cols_helper_4_14[9] cols_helper_4_14[10]
 cols_helper_4_14[11] cols_helper_4_14[12] cols_helper_4_14[13]
 cols_helper_4_14[14] cols_helper_4_14[15] cols_helper_4_14[16]
 cols_helper_5_15[0] cols_helper_5_15[1] cols_helper_5_15[2]
 cols_helper_5_15[3] cols_helper_5_15[4] cols_helper_5_15[5]
 cols_helper_5_15[6] cols_helper_5_15[7] cols_helper_5_15[8]
 cols_helper_5_15[9] cols_helper_5_15[10] cols_helper_5_15[11]
 cols_helper_5_15[12] cols_helper_5_15[13] cols_helper_5_15[14]
 cols_helper_5_15[15] cols_helper_5_15[16] cols_helper_6_12[0]
 cols_helper_6_12[1] cols_helper_6_12[2] cols_helper_6_12[3]
 cols_helper_6_12[4] cols_helper_6_12[5] cols_helper_6_12[6]
 cols_helper_6_12[7] cols_helper_6_12[8] cols_helper_6_12[9]
 cols_helper_6_12[10] cols_helper_6_12[11] cols_helper_6_12[12]
 cols_helper_6_12[13] cols_helper_6_12[14] cols_helper_6_12[15]
 cols_helper_6_12[16] cols_helper_7_9[0] cols_helper_7_9[1]
 cols_helper_7_9[2] cols_helper_7_9[3] cols_helper_7_9[4]
 cols_helper_7_9[5] cols_helper_7_9[6] cols_helper_7_9[7]
 cols_helper_7_9[8] cols_helper_7_9[9] cols_helper_7_9[10]
 cols_helper_7_9[11] cols_helper_7_9[12] cols_helper_7_9[13]
 cols_helper_7_9[14] cols_helper_7_9[15] cols_helper_7_9[16]
 cols_helper_8_8[0] cols_helper_8_8[1] cols_helper_8_8[2]
 cols_helper_8_8[3] cols_helper_8_8[4] cols_helper_8_8[5]
 cols_helper_8_8[6] cols_helper_8_8[7] cols_helper_8_8[8]
 cols_helper_8_8[9] cols_helper_8_8[10] cols_helper_8_8[11]
 cols_helper_8_8[12] cols_helper_8_8[13] cols_helper_8_8[14]
 cols_helper_8_8[15] cols_helper_8_8[16] cols_helper_4_3[0]
 cols_helper_4_3[1] cols_helper_4_3[2] cols_helper_4_3[3]
 cols_helper_4_3[4] cols_helper_4_3[5] cols_helper_4_3[6]
 cols_helper_4_3[7] cols_helper_4_3[8] cols_helper_4_3[9]
 cols_helper_4_3[10] cols_helper_4_3[11] cols_helper_4_3[12]
 cols_helper_4_3[13] cols_helper_4_3[14] cols_helper_4_3[15]
 cols_helper_4_3[16] cols_helper_9_5[0] cols_helper_9_5[1]
 cols_helper_9_5[2] cols_helper_9_5[3] cols_helper_9_5[4]
 cols_helper_9_5[5] cols_helper_9_5[6] cols_helper_9_5[7]
 cols_helper_9_5[8] cols_helper_9_5[9] cols_helper_9_5[10]
 cols_helper_9_5[11] cols_helper_9_5[12] cols_helper_9_5[13]
 cols_helper_9_5[14] cols_helper_9_5[15] cols_helper_9_5[16]
 cols_helper_5_2[0] cols_helper_5_2[1] cols_helper_5_2[2]
 cols_helper_5_2[3] cols_helper_5_2[4] cols_helper_5_2[5]
 cols_helper_5_2[6] cols_helper_5_2[7] cols_helper_5_2[8]
 cols_helper_5_2[9] cols_helper_5_2[10] cols_helper_5_2[11]
 cols_helper_5_2[12] cols_helper_5_2[13] cols_helper_5_2[14]
 cols_helper_5_2[15] cols_helper_5_2[16] cols_helper_10_2[0]
 cols_helper_10_2[1] cols_helper_10_2[2] cols_helper_10_2[3]
 cols_helper_10_2[4] cols_helper_10_2[5] cols_helper_10_2[6]
 cols_helper_10_2[7] cols_helper_10_2[8] cols_helper_10_2[9]
 cols_helper_10_2[10] cols_helper_10_2[11] cols_helper_10_2[12]
 cols_helper_10_2[13] cols_helper_10_2[14] cols_helper_10_2[15]
 cols_helper_10_2[16]
Generals
 cols_7_3 cols_6_9 cols_7_12 cols_8_5 cols_5_8 cols_10_8 cols_8_15
 cols_9_14 cols_8_2 cols_5_11 cols_4_5 cols_9_3 cols_8_12 cols_9_9
 cols_5_14 cols_10_14 cols_6_13 cols_7_8 cols_8_9 cols_9_4 cols_10_3
 cols_7_2 cols_6_14 cols_7_15 cols_8_6 cols_10_9 cols_6_4 cols_10_4
 cols_6_11 cols_9_13 cols_8_3 cols_4_6 cols_9_2 cols_7_4 cols_4_8 cols_9_8
 cols_10_15 cols_6_2 cols_9_11 cols_6_15 cols_7_14 cols_9_6 cols_6_5
 cols_5_3 cols_10_5 cols_6_8 cols_7_13 cols_9_12 cols_8_4 cols_5_9
 cols_10_11 cols_6_6 cols_10_6 cols_8_14 cols_4_9 cols_9_15 cols_5_12
 cols_4_4 cols_10_12 cols_6_3 cols_7_10 cols_8_11 cols_4_14 cols_5_15
 cols_6_12 cols_7_9 cols_8_8 cols_4_3 cols_9_5 cols_5_2 cols_10_2
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_tor_P41 , mem_tot_tor_P42 ,
   mem_tot_tor_P43 , mem_tot_tor_P44 , mem_tot_l1_P41 , mem_tot_l1_P42 ,
   mem_tot_l2_P4 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_{CPU13,CPU4,CPU6,CPU30,CPU9,CPU16,CPU22,CPU32} ,
   normalized_mem_tot_{CPU29,CPU8,CPU7,CPU21,CPU2,CPU3,CPU18,CPU5} ,
   normalized_mem_tot_{CPU28,CPU19,CPU25,CPU1,CPU10,CPU17,CPU11,CPU14} ,
   normalized_mem_tot_{CPU31,CPU12,CPU24,CPU27,CPU26,CPU23,CPU20,CPU15} )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_tor_P41 , rows_tot_tor_P42 ,
   rows_tot_tor_P43 , rows_tot_tor_P44 , rows_tot_l1_P41 ,
   rows_tot_l1_P42 , rows_tot_l2_P4 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX (
   normalized_rows_tot_{CPU13,CPU4,CPU6,CPU30,CPU9,CPU16,CPU22,CPU32} ,
   normalized_rows_tot_{CPU29,CPU8,CPU7,CPU21,CPU2,CPU3,CPU18,CPU5} ,
   normalized_rows_tot_{CPU28,CPU19,CPU25,CPU1,CPU10,CPU17,CPU11,CPU14} ,
   normalized_rows_tot_{CPU31,CPU12,CPU24,CPU27,CPU26,CPU23,CPU20,CPU15} )
End

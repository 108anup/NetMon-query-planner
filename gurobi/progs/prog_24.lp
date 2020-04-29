\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU_Cluster: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU1 + rows_tot_CPU6 + rows_tot_CPU9 + rows_tot_CPU10
   + rows_tot_CPU11 + rows_tot_CPU18 + rows_tot_CPU24 + rows_tot_CPU28
  CPU_Cluster_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU_Cluster
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_rows_P4: Priority=80 Weight=1 AbsTol=0 RelTol=0
   rows_tot_tor_P42
  P4_rows_load: Priority=70 Weight=1 AbsTol=0 RelTol=0
   max_rows_P4
  tot_mem_CPU_Cluster: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU1 + mem_tot_CPU6 + mem_tot_CPU9 + mem_tot_CPU10 + mem_tot_CPU11
   + mem_tot_CPU18 + mem_tot_CPU24 + mem_tot_CPU28
  CPU_Cluster_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
  tot_mem_P4: Priority=40 Weight=1 AbsTol=0 RelTol=0
   mem_tot_tor_P42
  P4_load_mem: Priority=30 Weight=1 AbsTol=0 RelTol=0
   max_mem_P4
Subject To
 cov_0_10: frac[5,10] + frac[8,10] + frac[2,10] >= 0.9
 cov_1_32: frac[6,32] + frac[1,32] >= 0.7
 cov_2_24: frac[3,24] + frac[8,24] + frac[4,24] >= 0.7
 cov_3_27: frac[0,27] + frac[6,27] >= 0.9
 cov_4_33: frac[1,33] + frac[6,33] >= 0.7
 cov_5_21: frac[8,21] + frac[4,21] + frac[7,21] >= 0.9
 cov_6_12: frac[7,12] + frac[0,12] >= 1
 cov_7_33: frac[3,33] + frac[4,33] + frac[8,33] >= 0.8
 cov_8_3: frac[8,3] + frac[3,3] + frac[6,3] >= 0.7
 cov_9_0: frac[5,0] + frac[0,0] >= 0.7
 cov_10_16: frac[1,16] + frac[5,16] >= 0.9
 cov_11_29: frac[2,29] + frac[8,29] + frac[3,29] >= 1
 cov_12_12: frac[7,12] + frac[3,12] + frac[8,12] >= 0.7
 cov_13_27: frac[5,27] + frac[6,27] >= 1
 cov_14_34: frac[5,34] >= 0.9
 cov_15_30: frac[1,30] + frac[3,30] + frac[8,30] >= 0.7
 cov_16_17: frac[0,17] + frac[6,17] >= 0.7
 cov_17_17: frac[3,17] + frac[6,17] + frac[8,17] >= 0.9
 cov_18_17: frac[1,17] + frac[4,17] + frac[8,17] >= 1
 cov_19_19: frac[8,19] + frac[2,19] + frac[3,19] >= 0.7
 cov_20_4: frac[5,4] + frac[8,4] + frac[4,4] >= 0.7
 cov_21_3: frac[8,3] + frac[3,3] + frac[4,3] >= 0.7
 cov_22_6: frac[6,6] + frac[5,6] >= 0.057488
 cov_23_2: frac[8,2] + frac[2,2] + frac[1,2] >= 0.8
 cov_24_3: frac[8,3] + frac[1,3] + frac[3,3] >= 0.7
 cov_25_5: frac[1,5] + frac[0,5] >= 0.8
 cov_26_17: frac[0,17] + frac[4,17] + frac[8,17] >= 0.8
 cov_27_9: frac[8,9] + frac[3,9] + frac[2,9] >= 0.8
 cov_28_10: frac[6,10] + frac[1,10] >= 0.8
 cov_29_13: frac[5,13] + frac[7,13] >= 0.9
 cov_30_29: frac[8,29] + frac[0,29] + frac[3,29] >= 0.8
 cov_31_13: frac[8,13] + frac[4,13] + frac[7,13] >= 0.9
 cov_32_2: frac[3,2] + frac[8,2] >= 0.8
 cov_33_19: frac[1,19] + frac[8,19] + frac[2,19] >= 0.7
 cov_34_31: frac[8,31] + frac[0,31] + frac[3,31] >= 0.592512
 cov_35_12: frac[8,12] + frac[4,12] + frac[2,12] >= 1
 cov_36_0: frac[4,0] + frac[1,0] + frac[8,0] >= 0.7
 cov_37_5: frac[2,5] + frac[8,5] + frac[3,5] >= 0.8
 cov_38_4: frac[0,4] + frac[3,4] + frac[8,4] >= 0.7
 cov_39_33: frac[5,33] + frac[6,33] >= 0.8
 cov_40_16: frac[5,16] + frac[7,16] >= 0.9
 cov_41_5: frac[8,5] + frac[0,5] + frac[3,5] >= 0.7
 cov_42_16: frac[5,16] + frac[0,16] >= 0.9
 cov_43_5: frac[1,5] + frac[6,5] >= 0.8
 cov_44_32: frac[4,32] + frac[8,32] + frac[3,32] >= 0.7
 cov_45_2: frac[6,2] + frac[5,2] >= 0.7
 cov_46_17: frac[7,17] + frac[6,17] >= 0.8
 cov_47_30: frac[6,30] + frac[1,30] >= 0.7
 cov_48_28: frac[2,28] + frac[8,28] + frac[4,28] >= 0.9
 cov_49_34: frac[2,34] + frac[0,34] + frac[8,34] >= 0.9
 cov_50_21: frac[8,21] + frac[1,21] + frac[3,21] >= 0.7
 cov_51_24: frac[8,24] + frac[7,24] + frac[2,24] >= 0.7
 cov_52_32: frac[2,32] + frac[4,32] + frac[8,32] >= 0.7
 cov_53_0: frac[8,0] + frac[0,0] + frac[2,0] >= 0.7
 cov_54_33: frac[3,33] + frac[8,33] + frac[6,33] >= 0.8
 cov_55_10: frac[5,10] + frac[7,10] >= 0.9
 cov_56_4: frac[2,4] + frac[8,4] + frac[4,4] >= 1
 cov_57_25: frac[8,25] >= 0.2
 cov_58_23: frac[8,23] >= 1
 cov_59_7: frac[8,7] >= 0.9
 cov_60_7: frac[8,7] >= 1
 cov_61_26: frac[8,26] >= 0.9
 cov_62_15: frac[8,15] >= 0.592512
 cov_63_22: frac[8,22] >= 0.8
 cov_64_1: frac[8,1] >= 0.3
 cov_65_8: frac[8,8] >= 0.7
 cov_66_8: frac[8,8] >= 0.7
 cov_67_18: frac[8,18] >= 0.3
 cov_68_20: frac[8,20] >= 0.1
 cov_69_8: frac[8,8] >= 0.7
 cov_70_1: frac[8,1] >= 0.3
 cov_71_11: frac[8,11] >= 3.33067e-16
 cov_72_14: frac[8,14] >= 0.3
 cov_73_11: frac[8,11] >= 3.33067e-16
 accuracy_7_12: - 5.09765625 frac[7,12] + mem[7,12] = 0
 accuracy_1_17: - 5.09765625 frac[1,17] + mem[1,17] = 0
 accuracy_2_5: - 5.09765625 frac[2,5] + mem[2,5] = 0
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
 cols_acc_8_5: - 435 frac[8,5] + cols_8_5 >= 0
 accuracy_8_5: mem[8,5] - 0.01171875 cols_8_5 = 0
 accuracy_4_0: - 5.09765625 frac[4,0] + mem[4,0] = 0
 accuracy_3_24: - 5.09765625 frac[3,24] + mem[3,24] = 0
 cols_helper_8_24: cols_helper_8_24[0] + cols_helper_8_24[1]
   + cols_helper_8_24[2] + cols_helper_8_24[3] + cols_helper_8_24[4]
   + cols_helper_8_24[5] + cols_helper_8_24[6] + cols_helper_8_24[7]
   + cols_helper_8_24[8] + cols_helper_8_24[9] + cols_helper_8_24[10]
   + cols_helper_8_24[11] + cols_helper_8_24[12] + cols_helper_8_24[13]
   + cols_helper_8_24[14] + cols_helper_8_24[15] + cols_helper_8_24[16]
   <= 1
 cols_8_24: cols_8_24 - 2 cols_helper_8_24[0] - 4 cols_helper_8_24[1]
   - 8 cols_helper_8_24[2] - 16 cols_helper_8_24[3]
   - 32 cols_helper_8_24[4] - 64 cols_helper_8_24[5]
   - 128 cols_helper_8_24[6] - 256 cols_helper_8_24[7]
   - 512 cols_helper_8_24[8] - 1024 cols_helper_8_24[9]
   - 2048 cols_helper_8_24[10] - 4096 cols_helper_8_24[11]
   - 8192 cols_helper_8_24[12] - 16384 cols_helper_8_24[13]
   - 32768 cols_helper_8_24[14] - 65536 cols_helper_8_24[15]
   - 131072 cols_helper_8_24[16] = 0
 cols_acc_8_24: - 435 frac[8,24] + cols_8_24 >= 0
 accuracy_8_24: mem[8,24] - 0.01171875 cols_8_24 = 0
 accuracy_6_10: - 5.09765625 frac[6,10] + mem[6,10] = 0
 accuracy_0_17: - 5.09765625 frac[0,17] + mem[0,17] = 0
 accuracy_0_4: - 5.09765625 frac[0,4] + mem[0,4] = 0
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
 cols_acc_8_15: - 435 frac[8,15] + cols_8_15 >= 0
 accuracy_8_15: mem[8,15] - 0.01171875 cols_8_15 = 0
 accuracy_3_2: - 5.09765625 frac[3,2] + mem[3,2] = 0
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
 cols_acc_8_2: - 435 frac[8,2] + cols_8_2 >= 0
 accuracy_8_2: mem[8,2] - 0.01171875 cols_8_2 = 0
 cols_helper_8_21: cols_helper_8_21[0] + cols_helper_8_21[1]
   + cols_helper_8_21[2] + cols_helper_8_21[3] + cols_helper_8_21[4]
   + cols_helper_8_21[5] + cols_helper_8_21[6] + cols_helper_8_21[7]
   + cols_helper_8_21[8] + cols_helper_8_21[9] + cols_helper_8_21[10]
   + cols_helper_8_21[11] + cols_helper_8_21[12] + cols_helper_8_21[13]
   + cols_helper_8_21[14] + cols_helper_8_21[15] + cols_helper_8_21[16]
   <= 1
 cols_8_21: cols_8_21 - 2 cols_helper_8_21[0] - 4 cols_helper_8_21[1]
   - 8 cols_helper_8_21[2] - 16 cols_helper_8_21[3]
   - 32 cols_helper_8_21[4] - 64 cols_helper_8_21[5]
   - 128 cols_helper_8_21[6] - 256 cols_helper_8_21[7]
   - 512 cols_helper_8_21[8] - 1024 cols_helper_8_21[9]
   - 2048 cols_helper_8_21[10] - 4096 cols_helper_8_21[11]
   - 8192 cols_helper_8_21[12] - 16384 cols_helper_8_21[13]
   - 32768 cols_helper_8_21[14] - 65536 cols_helper_8_21[15]
   - 131072 cols_helper_8_21[16] = 0
 cols_acc_8_21: - 435 frac[8,21] + cols_8_21 >= 0
 accuracy_8_21: mem[8,21] - 0.01171875 cols_8_21 = 0
 accuracy_2_32: - 5.09765625 frac[2,32] + mem[2,32] = 0
 accuracy_5_34: - 5.09765625 frac[5,34] + mem[5,34] = 0
 accuracy_3_12: - 5.09765625 frac[3,12] + mem[3,12] = 0
 accuracy_2_28: - 5.09765625 frac[2,28] + mem[2,28] = 0
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
 cols_acc_8_12: - 435 frac[8,12] + cols_8_12 >= 0
 accuracy_8_12: mem[8,12] - 0.01171875 cols_8_12 = 0
 cols_helper_8_31: cols_helper_8_31[0] + cols_helper_8_31[1]
   + cols_helper_8_31[2] + cols_helper_8_31[3] + cols_helper_8_31[4]
   + cols_helper_8_31[5] + cols_helper_8_31[6] + cols_helper_8_31[7]
   + cols_helper_8_31[8] + cols_helper_8_31[9] + cols_helper_8_31[10]
   + cols_helper_8_31[11] + cols_helper_8_31[12] + cols_helper_8_31[13]
   + cols_helper_8_31[14] + cols_helper_8_31[15] + cols_helper_8_31[16]
   <= 1
 cols_8_31: cols_8_31 - 2 cols_helper_8_31[0] - 4 cols_helper_8_31[1]
   - 8 cols_helper_8_31[2] - 16 cols_helper_8_31[3]
   - 32 cols_helper_8_31[4] - 64 cols_helper_8_31[5]
   - 128 cols_helper_8_31[6] - 256 cols_helper_8_31[7]
   - 512 cols_helper_8_31[8] - 1024 cols_helper_8_31[9]
   - 2048 cols_helper_8_31[10] - 4096 cols_helper_8_31[11]
   - 8192 cols_helper_8_31[12] - 16384 cols_helper_8_31[13]
   - 32768 cols_helper_8_31[14] - 65536 cols_helper_8_31[15]
   - 131072 cols_helper_8_31[16] = 0
 cols_acc_8_31: - 435 frac[8,31] + cols_8_31 >= 0
 accuracy_8_31: mem[8,31] - 0.01171875 cols_8_31 = 0
 cols_helper_8_18: cols_helper_8_18[0] + cols_helper_8_18[1]
   + cols_helper_8_18[2] + cols_helper_8_18[3] + cols_helper_8_18[4]
   + cols_helper_8_18[5] + cols_helper_8_18[6] + cols_helper_8_18[7]
   + cols_helper_8_18[8] + cols_helper_8_18[9] + cols_helper_8_18[10]
   + cols_helper_8_18[11] + cols_helper_8_18[12] + cols_helper_8_18[13]
   + cols_helper_8_18[14] + cols_helper_8_18[15] + cols_helper_8_18[16]
   <= 1
 cols_8_18: cols_8_18 - 2 cols_helper_8_18[0] - 4 cols_helper_8_18[1]
   - 8 cols_helper_8_18[2] - 16 cols_helper_8_18[3]
   - 32 cols_helper_8_18[4] - 64 cols_helper_8_18[5]
   - 128 cols_helper_8_18[6] - 256 cols_helper_8_18[7]
   - 512 cols_helper_8_18[8] - 1024 cols_helper_8_18[9]
   - 2048 cols_helper_8_18[10] - 4096 cols_helper_8_18[11]
   - 8192 cols_helper_8_18[12] - 16384 cols_helper_8_18[13]
   - 32768 cols_helper_8_18[14] - 65536 cols_helper_8_18[15]
   - 131072 cols_helper_8_18[16] = 0
 cols_acc_8_18: - 435 frac[8,18] + cols_8_18 >= 0
 accuracy_8_18: mem[8,18] - 0.01171875 cols_8_18 = 0
 accuracy_5_27: - 5.09765625 frac[5,27] + mem[5,27] = 0
 accuracy_4_21: - 5.09765625 frac[4,21] + mem[4,21] = 0
 accuracy_4_32: - 5.09765625 frac[4,32] + mem[4,32] = 0
 accuracy_1_21: - 5.09765625 frac[1,21] + mem[1,21] = 0
 accuracy_0_27: - 5.09765625 frac[0,27] + mem[0,27] = 0
 accuracy_7_21: - 5.09765625 frac[7,21] + mem[7,21] = 0
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
 cols_acc_8_9: - 435 frac[8,9] + cols_8_9 >= 0
 accuracy_8_9: mem[8,9] - 0.01171875 cols_8_9 = 0
 accuracy_4_12: - 5.09765625 frac[4,12] + mem[4,12] = 0
 accuracy_2_12: - 5.09765625 frac[2,12] + mem[2,12] = 0
 cols_helper_8_28: cols_helper_8_28[0] + cols_helper_8_28[1]
   + cols_helper_8_28[2] + cols_helper_8_28[3] + cols_helper_8_28[4]
   + cols_helper_8_28[5] + cols_helper_8_28[6] + cols_helper_8_28[7]
   + cols_helper_8_28[8] + cols_helper_8_28[9] + cols_helper_8_28[10]
   + cols_helper_8_28[11] + cols_helper_8_28[12] + cols_helper_8_28[13]
   + cols_helper_8_28[14] + cols_helper_8_28[15] + cols_helper_8_28[16]
   <= 1
 cols_8_28: cols_8_28 - 2 cols_helper_8_28[0] - 4 cols_helper_8_28[1]
   - 8 cols_helper_8_28[2] - 16 cols_helper_8_28[3]
   - 32 cols_helper_8_28[4] - 64 cols_helper_8_28[5]
   - 128 cols_helper_8_28[6] - 256 cols_helper_8_28[7]
   - 512 cols_helper_8_28[8] - 1024 cols_helper_8_28[9]
   - 2048 cols_helper_8_28[10] - 4096 cols_helper_8_28[11]
   - 8192 cols_helper_8_28[12] - 16384 cols_helper_8_28[13]
   - 32768 cols_helper_8_28[14] - 65536 cols_helper_8_28[15]
   - 131072 cols_helper_8_28[16] = 0
 cols_acc_8_28: - 435 frac[8,28] + cols_8_28 >= 0
 accuracy_8_28: mem[8,28] - 0.01171875 cols_8_28 = 0
 accuracy_3_17: - 5.09765625 frac[3,17] + mem[3,17] = 0
 accuracy_1_33: - 5.09765625 frac[1,33] + mem[1,33] = 0
 accuracy_1_16: - 5.09765625 frac[1,16] + mem[1,16] = 0
 cols_helper_8_32: cols_helper_8_32[0] + cols_helper_8_32[1]
   + cols_helper_8_32[2] + cols_helper_8_32[3] + cols_helper_8_32[4]
   + cols_helper_8_32[5] + cols_helper_8_32[6] + cols_helper_8_32[7]
   + cols_helper_8_32[8] + cols_helper_8_32[9] + cols_helper_8_32[10]
   + cols_helper_8_32[11] + cols_helper_8_32[12] + cols_helper_8_32[13]
   + cols_helper_8_32[14] + cols_helper_8_32[15] + cols_helper_8_32[16]
   <= 1
 cols_8_32: cols_8_32 - 2 cols_helper_8_32[0] - 4 cols_helper_8_32[1]
   - 8 cols_helper_8_32[2] - 16 cols_helper_8_32[3]
   - 32 cols_helper_8_32[4] - 64 cols_helper_8_32[5]
   - 128 cols_helper_8_32[6] - 256 cols_helper_8_32[7]
   - 512 cols_helper_8_32[8] - 1024 cols_helper_8_32[9]
   - 2048 cols_helper_8_32[10] - 4096 cols_helper_8_32[11]
   - 8192 cols_helper_8_32[12] - 16384 cols_helper_8_32[13]
   - 32768 cols_helper_8_32[14] - 65536 cols_helper_8_32[15]
   - 131072 cols_helper_8_32[16] = 0
 cols_acc_8_32: - 435 frac[8,32] + cols_8_32 >= 0
 accuracy_8_32: mem[8,32] - 0.01171875 cols_8_32 = 0
 accuracy_7_24: - 5.09765625 frac[7,24] + mem[7,24] = 0
 accuracy_6_32: - 5.09765625 frac[6,32] + mem[6,32] = 0
 accuracy_1_5: - 5.09765625 frac[1,5] + mem[1,5] = 0
 accuracy_2_2: - 5.09765625 frac[2,2] + mem[2,2] = 0
 accuracy_1_10: - 5.09765625 frac[1,10] + mem[1,10] = 0
 cols_helper_8_25: cols_helper_8_25[0] + cols_helper_8_25[1]
   + cols_helper_8_25[2] + cols_helper_8_25[3] + cols_helper_8_25[4]
   + cols_helper_8_25[5] + cols_helper_8_25[6] + cols_helper_8_25[7]
   + cols_helper_8_25[8] + cols_helper_8_25[9] + cols_helper_8_25[10]
   + cols_helper_8_25[11] + cols_helper_8_25[12] + cols_helper_8_25[13]
   + cols_helper_8_25[14] + cols_helper_8_25[15] + cols_helper_8_25[16]
   <= 1
 cols_8_25: cols_8_25 - 2 cols_helper_8_25[0] - 4 cols_helper_8_25[1]
   - 8 cols_helper_8_25[2] - 16 cols_helper_8_25[3]
   - 32 cols_helper_8_25[4] - 64 cols_helper_8_25[5]
   - 128 cols_helper_8_25[6] - 256 cols_helper_8_25[7]
   - 512 cols_helper_8_25[8] - 1024 cols_helper_8_25[9]
   - 2048 cols_helper_8_25[10] - 4096 cols_helper_8_25[11]
   - 8192 cols_helper_8_25[12] - 16384 cols_helper_8_25[13]
   - 32768 cols_helper_8_25[14] - 65536 cols_helper_8_25[15]
   - 131072 cols_helper_8_25[16] = 0
 cols_acc_8_25: - 435 frac[8,25] + cols_8_25 >= 0
 accuracy_8_25: mem[8,25] - 0.01171875 cols_8_25 = 0
 accuracy_5_4: - 5.09765625 frac[5,4] + mem[5,4] = 0
 accuracy_4_28: - 5.09765625 frac[4,28] + mem[4,28] = 0
 accuracy_3_33: - 5.09765625 frac[3,33] + mem[3,33] = 0
 accuracy_6_30: - 5.09765625 frac[6,30] + mem[6,30] = 0
 accuracy_1_19: - 5.09765625 frac[1,19] + mem[1,19] = 0
 accuracy_0_5: - 5.09765625 frac[0,5] + mem[0,5] = 0
 accuracy_2_24: - 5.09765625 frac[2,24] + mem[2,24] = 0
 accuracy_1_0: - 5.09765625 frac[1,0] + mem[1,0] = 0
 accuracy_3_5: - 5.09765625 frac[3,5] + mem[3,5] = 0
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
 cols_acc_8_3: - 435 frac[8,3] + cols_8_3 >= 0
 accuracy_8_3: mem[8,3] - 0.01171875 cols_8_3 = 0
 accuracy_5_10: - 5.09765625 frac[5,10] + mem[5,10] = 0
 cols_helper_8_22: cols_helper_8_22[0] + cols_helper_8_22[1]
   + cols_helper_8_22[2] + cols_helper_8_22[3] + cols_helper_8_22[4]
   + cols_helper_8_22[5] + cols_helper_8_22[6] + cols_helper_8_22[7]
   + cols_helper_8_22[8] + cols_helper_8_22[9] + cols_helper_8_22[10]
   + cols_helper_8_22[11] + cols_helper_8_22[12] + cols_helper_8_22[13]
   + cols_helper_8_22[14] + cols_helper_8_22[15] + cols_helper_8_22[16]
   <= 1
 cols_8_22: cols_8_22 - 2 cols_helper_8_22[0] - 4 cols_helper_8_22[1]
   - 8 cols_helper_8_22[2] - 16 cols_helper_8_22[3]
   - 32 cols_helper_8_22[4] - 64 cols_helper_8_22[5]
   - 128 cols_helper_8_22[6] - 256 cols_helper_8_22[7]
   - 512 cols_helper_8_22[8] - 1024 cols_helper_8_22[9]
   - 2048 cols_helper_8_22[10] - 4096 cols_helper_8_22[11]
   - 8192 cols_helper_8_22[12] - 16384 cols_helper_8_22[13]
   - 32768 cols_helper_8_22[14] - 65536 cols_helper_8_22[15]
   - 131072 cols_helper_8_22[16] = 0
 cols_acc_8_22: - 435 frac[8,22] + cols_8_22 >= 0
 accuracy_8_22: mem[8,22] - 0.01171875 cols_8_22 = 0
 accuracy_4_17: - 5.09765625 frac[4,17] + mem[4,17] = 0
 accuracy_0_31: - 5.09765625 frac[0,31] + mem[0,31] = 0
 accuracy_7_17: - 5.09765625 frac[7,17] + mem[7,17] = 0
 accuracy_6_27: - 5.09765625 frac[6,27] + mem[6,27] = 0
 accuracy_5_33: - 5.09765625 frac[5,33] + mem[5,33] = 0
 accuracy_1_30: - 5.09765625 frac[1,30] + mem[1,30] = 0
 accuracy_2_29: - 5.09765625 frac[2,29] + mem[2,29] = 0
 accuracy_1_3: - 5.09765625 frac[1,3] + mem[1,3] = 0
 cols_helper_8_13: cols_helper_8_13[0] + cols_helper_8_13[1]
   + cols_helper_8_13[2] + cols_helper_8_13[3] + cols_helper_8_13[4]
   + cols_helper_8_13[5] + cols_helper_8_13[6] + cols_helper_8_13[7]
   + cols_helper_8_13[8] + cols_helper_8_13[9] + cols_helper_8_13[10]
   + cols_helper_8_13[11] + cols_helper_8_13[12] + cols_helper_8_13[13]
   + cols_helper_8_13[14] + cols_helper_8_13[15] + cols_helper_8_13[16]
   <= 1
 cols_8_13: cols_8_13 - 2 cols_helper_8_13[0] - 4 cols_helper_8_13[1]
   - 8 cols_helper_8_13[2] - 16 cols_helper_8_13[3]
   - 32 cols_helper_8_13[4] - 64 cols_helper_8_13[5]
   - 128 cols_helper_8_13[6] - 256 cols_helper_8_13[7]
   - 512 cols_helper_8_13[8] - 1024 cols_helper_8_13[9]
   - 2048 cols_helper_8_13[10] - 4096 cols_helper_8_13[11]
   - 8192 cols_helper_8_13[12] - 16384 cols_helper_8_13[13]
   - 32768 cols_helper_8_13[14] - 65536 cols_helper_8_13[15]
   - 131072 cols_helper_8_13[16] = 0
 cols_acc_8_13: - 435 frac[8,13] + cols_8_13 >= 0
 accuracy_8_13: mem[8,13] - 0.01171875 cols_8_13 = 0
 cols_helper_8_0: cols_helper_8_0[0] + cols_helper_8_0[1]
   + cols_helper_8_0[2] + cols_helper_8_0[3] + cols_helper_8_0[4]
   + cols_helper_8_0[5] + cols_helper_8_0[6] + cols_helper_8_0[7]
   + cols_helper_8_0[8] + cols_helper_8_0[9] + cols_helper_8_0[10]
   + cols_helper_8_0[11] + cols_helper_8_0[12] + cols_helper_8_0[13]
   + cols_helper_8_0[14] + cols_helper_8_0[15] + cols_helper_8_0[16] <= 1
 cols_8_0: cols_8_0 - 2 cols_helper_8_0[0] - 4 cols_helper_8_0[1]
   - 8 cols_helper_8_0[2] - 16 cols_helper_8_0[3] - 32 cols_helper_8_0[4]
   - 64 cols_helper_8_0[5] - 128 cols_helper_8_0[6]
   - 256 cols_helper_8_0[7] - 512 cols_helper_8_0[8]
   - 1024 cols_helper_8_0[9] - 2048 cols_helper_8_0[10]
   - 4096 cols_helper_8_0[11] - 8192 cols_helper_8_0[12]
   - 16384 cols_helper_8_0[13] - 32768 cols_helper_8_0[14]
   - 65536 cols_helper_8_0[15] - 131072 cols_helper_8_0[16] = 0
 cols_acc_8_0: - 435 frac[8,0] + cols_8_0 >= 0
 accuracy_8_0: mem[8,0] - 0.01171875 cols_8_0 = 0
 accuracy_5_13: - 5.09765625 frac[5,13] + mem[5,13] = 0
 accuracy_3_21: - 5.09765625 frac[3,21] + mem[3,21] = 0
 cols_helper_8_19: cols_helper_8_19[0] + cols_helper_8_19[1]
   + cols_helper_8_19[2] + cols_helper_8_19[3] + cols_helper_8_19[4]
   + cols_helper_8_19[5] + cols_helper_8_19[6] + cols_helper_8_19[7]
   + cols_helper_8_19[8] + cols_helper_8_19[9] + cols_helper_8_19[10]
   + cols_helper_8_19[11] + cols_helper_8_19[12] + cols_helper_8_19[13]
   + cols_helper_8_19[14] + cols_helper_8_19[15] + cols_helper_8_19[16]
   <= 1
 cols_8_19: cols_8_19 - 2 cols_helper_8_19[0] - 4 cols_helper_8_19[1]
   - 8 cols_helper_8_19[2] - 16 cols_helper_8_19[3]
   - 32 cols_helper_8_19[4] - 64 cols_helper_8_19[5]
   - 128 cols_helper_8_19[6] - 256 cols_helper_8_19[7]
   - 512 cols_helper_8_19[8] - 1024 cols_helper_8_19[9]
   - 2048 cols_helper_8_19[10] - 4096 cols_helper_8_19[11]
   - 8192 cols_helper_8_19[12] - 16384 cols_helper_8_19[13]
   - 32768 cols_helper_8_19[14] - 65536 cols_helper_8_19[15]
   - 131072 cols_helper_8_19[16] = 0
 cols_acc_8_19: - 435 frac[8,19] + cols_8_19 >= 0
 accuracy_8_19: mem[8,19] - 0.01171875 cols_8_19 = 0
 accuracy_6_2: - 5.09765625 frac[6,2] + mem[6,2] = 0
 accuracy_2_34: - 5.09765625 frac[2,34] + mem[2,34] = 0
 accuracy_6_17: - 5.09765625 frac[6,17] + mem[6,17] = 0
 accuracy_4_33: - 5.09765625 frac[4,33] + mem[4,33] = 0
 accuracy_2_19: - 5.09765625 frac[2,19] + mem[2,19] = 0
 cols_helper_8_10: cols_helper_8_10[0] + cols_helper_8_10[1]
   + cols_helper_8_10[2] + cols_helper_8_10[3] + cols_helper_8_10[4]
   + cols_helper_8_10[5] + cols_helper_8_10[6] + cols_helper_8_10[7]
   + cols_helper_8_10[8] + cols_helper_8_10[9] + cols_helper_8_10[10]
   + cols_helper_8_10[11] + cols_helper_8_10[12] + cols_helper_8_10[13]
   + cols_helper_8_10[14] + cols_helper_8_10[15] + cols_helper_8_10[16]
   <= 1
 cols_8_10: cols_8_10 - 2 cols_helper_8_10[0] - 4 cols_helper_8_10[1]
   - 8 cols_helper_8_10[2] - 16 cols_helper_8_10[3]
   - 32 cols_helper_8_10[4] - 64 cols_helper_8_10[5]
   - 128 cols_helper_8_10[6] - 256 cols_helper_8_10[7]
   - 512 cols_helper_8_10[8] - 1024 cols_helper_8_10[9]
   - 2048 cols_helper_8_10[10] - 4096 cols_helper_8_10[11]
   - 8192 cols_helper_8_10[12] - 16384 cols_helper_8_10[13]
   - 32768 cols_helper_8_10[14] - 65536 cols_helper_8_10[15]
   - 131072 cols_helper_8_10[16] = 0
 cols_acc_8_10: - 435 frac[8,10] + cols_8_10 >= 0
 accuracy_8_10: mem[8,10] - 0.01171875 cols_8_10 = 0
 accuracy_4_13: - 5.09765625 frac[4,13] + mem[4,13] = 0
 accuracy_3_31: - 5.09765625 frac[3,31] + mem[3,31] = 0
 cols_helper_8_29: cols_helper_8_29[0] + cols_helper_8_29[1]
   + cols_helper_8_29[2] + cols_helper_8_29[3] + cols_helper_8_29[4]
   + cols_helper_8_29[5] + cols_helper_8_29[6] + cols_helper_8_29[7]
   + cols_helper_8_29[8] + cols_helper_8_29[9] + cols_helper_8_29[10]
   + cols_helper_8_29[11] + cols_helper_8_29[12] + cols_helper_8_29[13]
   + cols_helper_8_29[14] + cols_helper_8_29[15] + cols_helper_8_29[16]
   <= 1
 cols_8_29: cols_8_29 - 2 cols_helper_8_29[0] - 4 cols_helper_8_29[1]
   - 8 cols_helper_8_29[2] - 16 cols_helper_8_29[3]
   - 32 cols_helper_8_29[4] - 64 cols_helper_8_29[5]
   - 128 cols_helper_8_29[6] - 256 cols_helper_8_29[7]
   - 512 cols_helper_8_29[8] - 1024 cols_helper_8_29[9]
   - 2048 cols_helper_8_29[10] - 4096 cols_helper_8_29[11]
   - 8192 cols_helper_8_29[12] - 16384 cols_helper_8_29[13]
   - 32768 cols_helper_8_29[14] - 65536 cols_helper_8_29[15]
   - 131072 cols_helper_8_29[16] = 0
 cols_acc_8_29: - 435 frac[8,29] + cols_8_29 >= 0
 accuracy_8_29: mem[8,29] - 0.01171875 cols_8_29 = 0
 accuracy_5_0: - 5.09765625 frac[5,0] + mem[5,0] = 0
 accuracy_4_24: - 5.09765625 frac[4,24] + mem[4,24] = 0
 accuracy_1_32: - 5.09765625 frac[1,32] + mem[1,32] = 0
 cols_helper_8_33: cols_helper_8_33[0] + cols_helper_8_33[1]
   + cols_helper_8_33[2] + cols_helper_8_33[3] + cols_helper_8_33[4]
   + cols_helper_8_33[5] + cols_helper_8_33[6] + cols_helper_8_33[7]
   + cols_helper_8_33[8] + cols_helper_8_33[9] + cols_helper_8_33[10]
   + cols_helper_8_33[11] + cols_helper_8_33[12] + cols_helper_8_33[13]
   + cols_helper_8_33[14] + cols_helper_8_33[15] + cols_helper_8_33[16]
   <= 1
 cols_8_33: cols_8_33 - 2 cols_helper_8_33[0] - 4 cols_helper_8_33[1]
   - 8 cols_helper_8_33[2] - 16 cols_helper_8_33[3]
   - 32 cols_helper_8_33[4] - 64 cols_helper_8_33[5]
   - 128 cols_helper_8_33[6] - 256 cols_helper_8_33[7]
   - 512 cols_helper_8_33[8] - 1024 cols_helper_8_33[9]
   - 2048 cols_helper_8_33[10] - 4096 cols_helper_8_33[11]
   - 8192 cols_helper_8_33[12] - 16384 cols_helper_8_33[13]
   - 32768 cols_helper_8_33[14] - 65536 cols_helper_8_33[15]
   - 131072 cols_helper_8_33[16] = 0
 cols_acc_8_33: - 435 frac[8,33] + cols_8_33 >= 0
 accuracy_8_33: mem[8,33] - 0.01171875 cols_8_33 = 0
 accuracy_6_33: - 5.09765625 frac[6,33] + mem[6,33] = 0
 accuracy_0_12: - 5.09765625 frac[0,12] + mem[0,12] = 0
 accuracy_3_9: - 5.09765625 frac[3,9] + mem[3,9] = 0
 cols_helper_8_7: cols_helper_8_7[0] + cols_helper_8_7[1]
   + cols_helper_8_7[2] + cols_helper_8_7[3] + cols_helper_8_7[4]
   + cols_helper_8_7[5] + cols_helper_8_7[6] + cols_helper_8_7[7]
   + cols_helper_8_7[8] + cols_helper_8_7[9] + cols_helper_8_7[10]
   + cols_helper_8_7[11] + cols_helper_8_7[12] + cols_helper_8_7[13]
   + cols_helper_8_7[14] + cols_helper_8_7[15] + cols_helper_8_7[16] <= 1
 cols_8_7: cols_8_7 - 2 cols_helper_8_7[0] - 4 cols_helper_8_7[1]
   - 8 cols_helper_8_7[2] - 16 cols_helper_8_7[3] - 32 cols_helper_8_7[4]
   - 64 cols_helper_8_7[5] - 128 cols_helper_8_7[6]
   - 256 cols_helper_8_7[7] - 512 cols_helper_8_7[8]
   - 1024 cols_helper_8_7[9] - 2048 cols_helper_8_7[10]
   - 4096 cols_helper_8_7[11] - 8192 cols_helper_8_7[12]
   - 16384 cols_helper_8_7[13] - 32768 cols_helper_8_7[14]
   - 65536 cols_helper_8_7[15] - 131072 cols_helper_8_7[16] = 0
 cols_acc_8_7: - 435 frac[8,7] + cols_8_7 >= 0
 accuracy_8_7: mem[8,7] - 0.01171875 cols_8_7 = 0
 accuracy_0_34: - 5.09765625 frac[0,34] + mem[0,34] = 0
 accuracy_6_5: - 5.09765625 frac[6,5] + mem[6,5] = 0
 cols_helper_8_26: cols_helper_8_26[0] + cols_helper_8_26[1]
   + cols_helper_8_26[2] + cols_helper_8_26[3] + cols_helper_8_26[4]
   + cols_helper_8_26[5] + cols_helper_8_26[6] + cols_helper_8_26[7]
   + cols_helper_8_26[8] + cols_helper_8_26[9] + cols_helper_8_26[10]
   + cols_helper_8_26[11] + cols_helper_8_26[12] + cols_helper_8_26[13]
   + cols_helper_8_26[14] + cols_helper_8_26[15] + cols_helper_8_26[16]
   <= 1
 cols_8_26: cols_8_26 - 2 cols_helper_8_26[0] - 4 cols_helper_8_26[1]
   - 8 cols_helper_8_26[2] - 16 cols_helper_8_26[3]
   - 32 cols_helper_8_26[4] - 64 cols_helper_8_26[5]
   - 128 cols_helper_8_26[6] - 256 cols_helper_8_26[7]
   - 512 cols_helper_8_26[8] - 1024 cols_helper_8_26[9]
   - 2048 cols_helper_8_26[10] - 4096 cols_helper_8_26[11]
   - 8192 cols_helper_8_26[12] - 16384 cols_helper_8_26[13]
   - 32768 cols_helper_8_26[14] - 65536 cols_helper_8_26[15]
   - 131072 cols_helper_8_26[16] = 0
 cols_acc_8_26: - 435 frac[8,26] + cols_8_26 >= 0
 accuracy_8_26: mem[8,26] - 0.01171875 cols_8_26 = 0
 accuracy_5_16: - 5.09765625 frac[5,16] + mem[5,16] = 0
 accuracy_3_32: - 5.09765625 frac[3,32] + mem[3,32] = 0
 accuracy_7_13: - 5.09765625 frac[7,13] + mem[7,13] = 0
 accuracy_3_4: - 5.09765625 frac[3,4] + mem[3,4] = 0
 accuracy_2_4: - 5.09765625 frac[2,4] + mem[2,4] = 0
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
 cols_acc_8_4: - 435 frac[8,4] + cols_8_4 >= 0
 accuracy_8_4: mem[8,4] - 0.01171875 cols_8_4 = 0
 cols_helper_8_23: cols_helper_8_23[0] + cols_helper_8_23[1]
   + cols_helper_8_23[2] + cols_helper_8_23[3] + cols_helper_8_23[4]
   + cols_helper_8_23[5] + cols_helper_8_23[6] + cols_helper_8_23[7]
   + cols_helper_8_23[8] + cols_helper_8_23[9] + cols_helper_8_23[10]
   + cols_helper_8_23[11] + cols_helper_8_23[12] + cols_helper_8_23[13]
   + cols_helper_8_23[14] + cols_helper_8_23[15] + cols_helper_8_23[16]
   <= 1
 cols_8_23: cols_8_23 - 2 cols_helper_8_23[0] - 4 cols_helper_8_23[1]
   - 8 cols_helper_8_23[2] - 16 cols_helper_8_23[3]
   - 32 cols_helper_8_23[4] - 64 cols_helper_8_23[5]
   - 128 cols_helper_8_23[6] - 256 cols_helper_8_23[7]
   - 512 cols_helper_8_23[8] - 1024 cols_helper_8_23[9]
   - 2048 cols_helper_8_23[10] - 4096 cols_helper_8_23[11]
   - 8192 cols_helper_8_23[12] - 16384 cols_helper_8_23[13]
   - 32768 cols_helper_8_23[14] - 65536 cols_helper_8_23[15]
   - 131072 cols_helper_8_23[16] = 0
 cols_acc_8_23: - 435 frac[8,23] + cols_8_23 >= 0
 accuracy_8_23: mem[8,23] - 0.01171875 cols_8_23 = 0
 accuracy_6_6: - 5.09765625 frac[6,6] + mem[6,6] = 0
 accuracy_5_6: - 5.09765625 frac[5,6] + mem[5,6] = 0
 accuracy_0_16: - 5.09765625 frac[0,16] + mem[0,16] = 0
 accuracy_7_16: - 5.09765625 frac[7,16] + mem[7,16] = 0
 accuracy_1_2: - 5.09765625 frac[1,2] + mem[1,2] = 0
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
 cols_acc_8_14: - 435 frac[8,14] + cols_8_14 >= 0
 accuracy_8_14: mem[8,14] - 0.01171875 cols_8_14 = 0
 accuracy_3_3: - 5.09765625 frac[3,3] + mem[3,3] = 0
 accuracy_2_9: - 5.09765625 frac[2,9] + mem[2,9] = 0
 cols_helper_8_1: cols_helper_8_1[0] + cols_helper_8_1[1]
   + cols_helper_8_1[2] + cols_helper_8_1[3] + cols_helper_8_1[4]
   + cols_helper_8_1[5] + cols_helper_8_1[6] + cols_helper_8_1[7]
   + cols_helper_8_1[8] + cols_helper_8_1[9] + cols_helper_8_1[10]
   + cols_helper_8_1[11] + cols_helper_8_1[12] + cols_helper_8_1[13]
   + cols_helper_8_1[14] + cols_helper_8_1[15] + cols_helper_8_1[16] <= 1
 cols_8_1: cols_8_1 - 2 cols_helper_8_1[0] - 4 cols_helper_8_1[1]
   - 8 cols_helper_8_1[2] - 16 cols_helper_8_1[3] - 32 cols_helper_8_1[4]
   - 64 cols_helper_8_1[5] - 128 cols_helper_8_1[6]
   - 256 cols_helper_8_1[7] - 512 cols_helper_8_1[8]
   - 1024 cols_helper_8_1[9] - 2048 cols_helper_8_1[10]
   - 4096 cols_helper_8_1[11] - 8192 cols_helper_8_1[12]
   - 16384 cols_helper_8_1[13] - 32768 cols_helper_8_1[14]
   - 65536 cols_helper_8_1[15] - 131072 cols_helper_8_1[16] = 0
 cols_acc_8_1: - 435 frac[8,1] + cols_8_1 >= 0
 accuracy_8_1: mem[8,1] - 0.01171875 cols_8_1 = 0
 accuracy_4_4: - 5.09765625 frac[4,4] + mem[4,4] = 0
 cols_helper_8_20: cols_helper_8_20[0] + cols_helper_8_20[1]
   + cols_helper_8_20[2] + cols_helper_8_20[3] + cols_helper_8_20[4]
   + cols_helper_8_20[5] + cols_helper_8_20[6] + cols_helper_8_20[7]
   + cols_helper_8_20[8] + cols_helper_8_20[9] + cols_helper_8_20[10]
   + cols_helper_8_20[11] + cols_helper_8_20[12] + cols_helper_8_20[13]
   + cols_helper_8_20[14] + cols_helper_8_20[15] + cols_helper_8_20[16]
   <= 1
 cols_8_20: cols_8_20 - 2 cols_helper_8_20[0] - 4 cols_helper_8_20[1]
   - 8 cols_helper_8_20[2] - 16 cols_helper_8_20[3]
   - 32 cols_helper_8_20[4] - 64 cols_helper_8_20[5]
   - 128 cols_helper_8_20[6] - 256 cols_helper_8_20[7]
   - 512 cols_helper_8_20[8] - 1024 cols_helper_8_20[9]
   - 2048 cols_helper_8_20[10] - 4096 cols_helper_8_20[11]
   - 8192 cols_helper_8_20[12] - 16384 cols_helper_8_20[13]
   - 32768 cols_helper_8_20[14] - 65536 cols_helper_8_20[15]
   - 131072 cols_helper_8_20[16] = 0
 cols_acc_8_20: - 435 frac[8,20] + cols_8_20 >= 0
 accuracy_8_20: mem[8,20] - 0.01171875 cols_8_20 = 0
 accuracy_6_3: - 5.09765625 frac[6,3] + mem[6,3] = 0
 accuracy_7_10: - 5.09765625 frac[7,10] + mem[7,10] = 0
 accuracy_0_29: - 5.09765625 frac[0,29] + mem[0,29] = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
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
 cols_acc_8_11: - 435 frac[8,11] + cols_8_11 >= 0
 accuracy_8_11: mem[8,11] - 0.01171875 cols_8_11 = 0
 accuracy_3_30: - 5.09765625 frac[3,30] + mem[3,30] = 0
 accuracy_2_10: - 5.09765625 frac[2,10] + mem[2,10] = 0
 cols_helper_8_30: cols_helper_8_30[0] + cols_helper_8_30[1]
   + cols_helper_8_30[2] + cols_helper_8_30[3] + cols_helper_8_30[4]
   + cols_helper_8_30[5] + cols_helper_8_30[6] + cols_helper_8_30[7]
   + cols_helper_8_30[8] + cols_helper_8_30[9] + cols_helper_8_30[10]
   + cols_helper_8_30[11] + cols_helper_8_30[12] + cols_helper_8_30[13]
   + cols_helper_8_30[14] + cols_helper_8_30[15] + cols_helper_8_30[16]
   <= 1
 cols_8_30: cols_8_30 - 2 cols_helper_8_30[0] - 4 cols_helper_8_30[1]
   - 8 cols_helper_8_30[2] - 16 cols_helper_8_30[3]
   - 32 cols_helper_8_30[4] - 64 cols_helper_8_30[5]
   - 128 cols_helper_8_30[6] - 256 cols_helper_8_30[7]
   - 512 cols_helper_8_30[8] - 1024 cols_helper_8_30[9]
   - 2048 cols_helper_8_30[10] - 4096 cols_helper_8_30[11]
   - 8192 cols_helper_8_30[12] - 16384 cols_helper_8_30[13]
   - 32768 cols_helper_8_30[14] - 65536 cols_helper_8_30[15]
   - 131072 cols_helper_8_30[16] = 0
 cols_acc_8_30: - 435 frac[8,30] + cols_8_30 >= 0
 accuracy_8_30: mem[8,30] - 0.01171875 cols_8_30 = 0
 accuracy_3_19: - 5.09765625 frac[3,19] + mem[3,19] = 0
 cols_helper_8_17: cols_helper_8_17[0] + cols_helper_8_17[1]
   + cols_helper_8_17[2] + cols_helper_8_17[3] + cols_helper_8_17[4]
   + cols_helper_8_17[5] + cols_helper_8_17[6] + cols_helper_8_17[7]
   + cols_helper_8_17[8] + cols_helper_8_17[9] + cols_helper_8_17[10]
   + cols_helper_8_17[11] + cols_helper_8_17[12] + cols_helper_8_17[13]
   + cols_helper_8_17[14] + cols_helper_8_17[15] + cols_helper_8_17[16]
   <= 1
 cols_8_17: cols_8_17 - 2 cols_helper_8_17[0] - 4 cols_helper_8_17[1]
   - 8 cols_helper_8_17[2] - 16 cols_helper_8_17[3]
   - 32 cols_helper_8_17[4] - 64 cols_helper_8_17[5]
   - 128 cols_helper_8_17[6] - 256 cols_helper_8_17[7]
   - 512 cols_helper_8_17[8] - 1024 cols_helper_8_17[9]
   - 2048 cols_helper_8_17[10] - 4096 cols_helper_8_17[11]
   - 8192 cols_helper_8_17[12] - 16384 cols_helper_8_17[13]
   - 32768 cols_helper_8_17[14] - 65536 cols_helper_8_17[15]
   - 131072 cols_helper_8_17[16] = 0
 cols_acc_8_17: - 435 frac[8,17] + cols_8_17 >= 0
 accuracy_8_17: mem[8,17] - 0.01171875 cols_8_17 = 0
 cols_helper_8_34: cols_helper_8_34[0] + cols_helper_8_34[1]
   + cols_helper_8_34[2] + cols_helper_8_34[3] + cols_helper_8_34[4]
   + cols_helper_8_34[5] + cols_helper_8_34[6] + cols_helper_8_34[7]
   + cols_helper_8_34[8] + cols_helper_8_34[9] + cols_helper_8_34[10]
   + cols_helper_8_34[11] + cols_helper_8_34[12] + cols_helper_8_34[13]
   + cols_helper_8_34[14] + cols_helper_8_34[15] + cols_helper_8_34[16]
   <= 1
 cols_8_34: cols_8_34 - 2 cols_helper_8_34[0] - 4 cols_helper_8_34[1]
   - 8 cols_helper_8_34[2] - 16 cols_helper_8_34[3]
   - 32 cols_helper_8_34[4] - 64 cols_helper_8_34[5]
   - 128 cols_helper_8_34[6] - 256 cols_helper_8_34[7]
   - 512 cols_helper_8_34[8] - 1024 cols_helper_8_34[9]
   - 2048 cols_helper_8_34[10] - 4096 cols_helper_8_34[11]
   - 8192 cols_helper_8_34[12] - 16384 cols_helper_8_34[13]
   - 32768 cols_helper_8_34[14] - 65536 cols_helper_8_34[15]
   - 131072 cols_helper_8_34[16] = 0
 cols_acc_8_34: - 435 frac[8,34] + cols_8_34 >= 0
 accuracy_8_34: mem[8,34] - 0.01171875 cols_8_34 = 0
 accuracy_2_0: - 5.09765625 frac[2,0] + mem[2,0] = 0
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
 cols_acc_8_8: - 435 frac[8,8] + cols_8_8 >= 0
 accuracy_8_8: mem[8,8] - 0.01171875 cols_8_8 = 0
 accuracy_4_3: - 5.09765625 frac[4,3] + mem[4,3] = 0
 accuracy_3_29: - 5.09765625 frac[3,29] + mem[3,29] = 0
 accuracy_5_2: - 5.09765625 frac[5,2] + mem[5,2] = 0
 mem_tot_CPU1: - mem[0,17] - mem[0,4] - mem[0,27] - mem[0,5] - mem[0,31]
   - mem[0,12] - mem[0,34] - mem[0,16] - mem[0,29] - mem[0,0]
   + mem_tot_CPU1 = 0
 R291: - 3.05176e-05 mem_tot_CPU1 + normalized_mem_tot_CPU1 = 0
 rows_tot_CPU1: - 3 frac[0,17] - 3 frac[0,4] - 3 frac[0,27] - 3 frac[0,5]
   - 3 frac[0,31] - 3 frac[0,12] - 3 frac[0,34] - 3 frac[0,16]
   - 3 frac[0,29] - 3 frac[0,0] + rows_tot_CPU1 = 0
 R293: - 0.0833333333333333 rows_tot_CPU1 + normalized_rows_tot_CPU1 = 0
 mem_tot_CPU6: - mem[1,17] - mem[1,21] - mem[1,33] - mem[1,16] - mem[1,5]
   - mem[1,10] - mem[1,19] - mem[1,0] - mem[1,30] - mem[1,3] - mem[1,32]
   - mem[1,2] + mem_tot_CPU6 = 0
 R295: - 3.05176e-05 mem_tot_CPU6 + normalized_mem_tot_CPU6 = 0
 rows_tot_CPU6: - 3 frac[1,17] - 3 frac[1,21] - 3 frac[1,33] - 3 frac[1,16]
   - 3 frac[1,5] - 3 frac[1,10] - 3 frac[1,19] - 3 frac[1,0] - 3 frac[1,30]
   - 3 frac[1,3] - 3 frac[1,32] - 3 frac[1,2] + rows_tot_CPU6 = 0
 R297: - 0.0833333333333333 rows_tot_CPU6 + normalized_rows_tot_CPU6 = 0
 mem_tot_CPU9: - mem[2,5] - mem[2,32] - mem[2,28] - mem[2,12] - mem[2,2]
   - mem[2,24] - mem[2,29] - mem[2,34] - mem[2,19] - mem[2,4] - mem[2,9]
   - mem[2,10] - mem[2,0] + mem_tot_CPU9 = 0
 R299: - 3.05176e-05 mem_tot_CPU9 + normalized_mem_tot_CPU9 = 0
 rows_tot_CPU9: - 3 frac[2,5] - 3 frac[2,32] - 3 frac[2,28] - 3 frac[2,12]
   - 3 frac[2,2] - 3 frac[2,24] - 3 frac[2,29] - 3 frac[2,34]
   - 3 frac[2,19] - 3 frac[2,4] - 3 frac[2,9] - 3 frac[2,10] - 3 frac[2,0]
   + rows_tot_CPU9 = 0
 R301: - 0.0833333333333333 rows_tot_CPU9 + normalized_rows_tot_CPU9 = 0
 mem_tot_CPU10: - mem[3,24] - mem[3,2] - mem[3,12] - mem[3,17] - mem[3,33]
   - mem[3,5] - mem[3,21] - mem[3,31] - mem[3,9] - mem[3,32] - mem[3,4]
   - mem[3,3] - mem[3,30] - mem[3,19] - mem[3,29] + mem_tot_CPU10 = 0
 R303: - 3.05176e-05 mem_tot_CPU10 + normalized_mem_tot_CPU10 = 0
 rows_tot_CPU10: - 3 frac[3,24] - 3 frac[3,2] - 3 frac[3,12] - 3 frac[3,17]
   - 3 frac[3,33] - 3 frac[3,5] - 3 frac[3,21] - 3 frac[3,31] - 3 frac[3,9]
   - 3 frac[3,32] - 3 frac[3,4] - 3 frac[3,3] - 3 frac[3,30] - 3 frac[3,19]
   - 3 frac[3,29] + rows_tot_CPU10 = 0
 R305: - 0.0833333333333333 rows_tot_CPU10 + normalized_rows_tot_CPU10 = 0
 mem_tot_CPU11: - mem[4,0] - mem[4,21] - mem[4,32] - mem[4,12] - mem[4,28]
   - mem[4,17] - mem[4,33] - mem[4,13] - mem[4,24] - mem[4,4] - mem[4,3]
   + mem_tot_CPU11 = 0
 R307: - 3.05176e-05 mem_tot_CPU11 + normalized_mem_tot_CPU11 = 0
 rows_tot_CPU11: - 3 frac[4,0] - 3 frac[4,21] - 3 frac[4,32] - 3 frac[4,12]
   - 3 frac[4,28] - 3 frac[4,17] - 3 frac[4,33] - 3 frac[4,13]
   - 3 frac[4,24] - 3 frac[4,4] - 3 frac[4,3] + rows_tot_CPU11 = 0
 R309: - 0.0833333333333333 rows_tot_CPU11 + normalized_rows_tot_CPU11 = 0
 mem_tot_CPU18: - mem[5,34] - mem[5,27] - mem[5,4] - mem[5,10] - mem[5,33]
   - mem[5,13] - mem[5,0] - mem[5,16] - mem[5,6] - mem[5,2] + mem_tot_CPU18
   = 0
 R311: - 3.05176e-05 mem_tot_CPU18 + normalized_mem_tot_CPU18 = 0
 rows_tot_CPU18: - 3 frac[5,34] - 3 frac[5,27] - 3 frac[5,4] - 3 frac[5,10]
   - 3 frac[5,33] - 3 frac[5,13] - 3 frac[5,0] - 3 frac[5,16] - 3 frac[5,6]
   - 3 frac[5,2] + rows_tot_CPU18 = 0
 R313: - 0.0833333333333333 rows_tot_CPU18 + normalized_rows_tot_CPU18 = 0
 mem_tot_CPU24: - mem[6,10] - mem[6,32] - mem[6,30] - mem[6,27] - mem[6,2]
   - mem[6,17] - mem[6,33] - mem[6,5] - mem[6,6] - mem[6,3] + mem_tot_CPU24
   = 0
 R315: - 3.05176e-05 mem_tot_CPU24 + normalized_mem_tot_CPU24 = 0
 rows_tot_CPU24: - 3 frac[6,10] - 3 frac[6,32] - 3 frac[6,30]
   - 3 frac[6,27] - 3 frac[6,2] - 3 frac[6,17] - 3 frac[6,33] - 3 frac[6,5]
   - 3 frac[6,6] - 3 frac[6,3] + rows_tot_CPU24 = 0
 R317: - 0.0833333333333333 rows_tot_CPU24 + normalized_rows_tot_CPU24 = 0
 mem_tot_CPU28: - mem[7,12] - mem[7,21] - mem[7,24] - mem[7,17] - mem[7,13]
   - mem[7,16] - mem[7,10] + mem_tot_CPU28 = 0
 R319: - 3.05176e-05 mem_tot_CPU28 + normalized_mem_tot_CPU28 = 0
 rows_tot_CPU28: - 3 frac[7,12] - 3 frac[7,21] - 3 frac[7,24]
   - 3 frac[7,17] - 3 frac[7,13] - 3 frac[7,16] - 3 frac[7,10]
   + rows_tot_CPU28 = 0
 R321: - 0.0833333333333333 rows_tot_CPU28 + normalized_rows_tot_CPU28 = 0
 mem_tot_tor_P42: - mem[8,5] - mem[8,24] - mem[8,15] - mem[8,2] - mem[8,21]
   - mem[8,12] - mem[8,31] - mem[8,18] - mem[8,9] - mem[8,28] - mem[8,32]
   - mem[8,25] - mem[8,3] - mem[8,22] - mem[8,13] - mem[8,0] - mem[8,19]
   - mem[8,10] - mem[8,29] - mem[8,33] - mem[8,7] - mem[8,26] - mem[8,4]
   - mem[8,23] - mem[8,14] - mem[8,1] - mem[8,20] - mem[8,11] - mem[8,30]
   - mem[8,17] - mem[8,34] - mem[8,8] + mem_tot_tor_P42 = 0
 R323: - 1.6276041666666666e-04 mem_tot_tor_P42
   + normalized_mem_tot_tor_P42 = 0
 rows_tot_tor_P42: - 3 frac[8,5] - 3 frac[8,24] - 3 frac[8,15]
   - 3 frac[8,2] - 3 frac[8,21] - 3 frac[8,12] - 3 frac[8,31]
   - 3 frac[8,18] - 3 frac[8,9] - 3 frac[8,28] - 3 frac[8,32]
   - 3 frac[8,25] - 3 frac[8,3] - 3 frac[8,22] - 3 frac[8,13] - 3 frac[8,0]
   - 3 frac[8,19] - 3 frac[8,10] - 3 frac[8,29] - 3 frac[8,33]
   - 3 frac[8,7] - 3 frac[8,26] - 3 frac[8,4] - 3 frac[8,23] - 3 frac[8,14]
   - 3 frac[8,1] - 3 frac[8,20] - 3 frac[8,11] - 3 frac[8,30]
   - 3 frac[8,17] - 3 frac[8,34] - 3 frac[8,8] + rows_tot_tor_P42 = 0
 R325: - 0.0208333333333333 rows_tot_tor_P42 + normalized_rows_tot_tor_P42
   = 0
 row_capacity_CPU1: rows_tot_CPU1 <= 12
 row_capacity_CPU6: rows_tot_CPU6 <= 12
 row_capacity_CPU9: rows_tot_CPU9 <= 12
 row_capacity_CPU10: rows_tot_CPU10 <= 12
 row_capacity_CPU11: rows_tot_CPU11 <= 12
 row_capacity_CPU18: rows_tot_CPU18 <= 12
 row_capacity_CPU24: rows_tot_CPU24 <= 12
 row_capacity_CPU28: rows_tot_CPU28 <= 12
 row_capacity_tor_P42: rows_tot_tor_P42 <= 48
 capacity_mem_par_tor_P42: mem[8,5] <= 1536
 capacity_mem_par_tor_P42: mem[8,24] <= 1536
 capacity_mem_par_tor_P42: mem[8,15] <= 1536
 capacity_mem_par_tor_P42: mem[8,2] <= 1536
 capacity_mem_par_tor_P42: mem[8,21] <= 1536
 capacity_mem_par_tor_P42: mem[8,12] <= 1536
 capacity_mem_par_tor_P42: mem[8,31] <= 1536
 capacity_mem_par_tor_P42: mem[8,18] <= 1536
 capacity_mem_par_tor_P42: mem[8,9] <= 1536
 capacity_mem_par_tor_P42: mem[8,28] <= 1536
 capacity_mem_par_tor_P42: mem[8,32] <= 1536
 capacity_mem_par_tor_P42: mem[8,25] <= 1536
 capacity_mem_par_tor_P42: mem[8,3] <= 1536
 capacity_mem_par_tor_P42: mem[8,22] <= 1536
 capacity_mem_par_tor_P42: mem[8,13] <= 1536
 capacity_mem_par_tor_P42: mem[8,0] <= 1536
 capacity_mem_par_tor_P42: mem[8,19] <= 1536
 capacity_mem_par_tor_P42: mem[8,10] <= 1536
 capacity_mem_par_tor_P42: mem[8,29] <= 1536
 capacity_mem_par_tor_P42: mem[8,33] <= 1536
 capacity_mem_par_tor_P42: mem[8,7] <= 1536
 capacity_mem_par_tor_P42: mem[8,26] <= 1536
 capacity_mem_par_tor_P42: mem[8,4] <= 1536
 capacity_mem_par_tor_P42: mem[8,23] <= 1536
 capacity_mem_par_tor_P42: mem[8,14] <= 1536
 capacity_mem_par_tor_P42: mem[8,1] <= 1536
 capacity_mem_par_tor_P42: mem[8,20] <= 1536
 capacity_mem_par_tor_P42: mem[8,11] <= 1536
 capacity_mem_par_tor_P42: mem[8,30] <= 1536
 capacity_mem_par_tor_P42: mem[8,17] <= 1536
 capacity_mem_par_tor_P42: mem[8,34] <= 1536
 capacity_mem_par_tor_P42: mem[8,8] <= 1536
Bounds
 frac[7,12] <= 1
 frac[1,17] <= 1
 frac[2,5] <= 1
 frac[8,5] <= 1
 frac[4,0] <= 1
 frac[3,24] <= 1
 frac[8,24] <= 1
 frac[6,10] <= 1
 frac[0,17] <= 1
 frac[0,4] <= 1
 frac[8,15] <= 1
 frac[3,2] <= 1
 frac[8,2] <= 1
 frac[8,21] <= 1
 frac[2,32] <= 1
 frac[5,34] <= 1
 frac[3,12] <= 1
 frac[2,28] <= 1
 frac[8,12] <= 1
 frac[8,31] <= 1
 frac[8,18] <= 1
 frac[5,27] <= 1
 frac[4,21] <= 1
 frac[4,32] <= 1
 frac[1,21] <= 1
 frac[0,27] <= 1
 frac[7,21] <= 1
 frac[8,9] <= 1
 frac[4,12] <= 1
 frac[2,12] <= 1
 frac[8,28] <= 1
 frac[3,17] <= 1
 frac[1,33] <= 1
 frac[1,16] <= 1
 frac[8,32] <= 1
 frac[7,24] <= 1
 frac[6,32] <= 1
 frac[1,5] <= 1
 frac[2,2] <= 1
 frac[1,10] <= 1
 frac[8,25] <= 1
 frac[5,4] <= 1
 frac[4,28] <= 1
 frac[3,33] <= 1
 frac[6,30] <= 1
 frac[1,19] <= 1
 frac[0,5] <= 1
 frac[2,24] <= 1
 frac[1,0] <= 1
 frac[3,5] <= 1
 frac[8,3] <= 1
 frac[5,10] <= 1
 frac[8,22] <= 1
 frac[4,17] <= 1
 frac[0,31] <= 1
 frac[7,17] <= 1
 frac[6,27] <= 1
 frac[5,33] <= 1
 frac[1,30] <= 1
 frac[2,29] <= 1
 frac[1,3] <= 1
 frac[8,13] <= 1
 frac[8,0] <= 1
 frac[5,13] <= 1
 frac[3,21] <= 1
 frac[8,19] <= 1
 frac[6,2] <= 1
 frac[2,34] <= 1
 frac[6,17] <= 1
 frac[4,33] <= 1
 frac[2,19] <= 1
 frac[8,10] <= 1
 frac[4,13] <= 1
 frac[3,31] <= 1
 frac[8,29] <= 1
 frac[5,0] <= 1
 frac[4,24] <= 1
 frac[1,32] <= 1
 frac[8,33] <= 1
 frac[6,33] <= 1
 frac[0,12] <= 1
 frac[3,9] <= 1
 frac[8,7] <= 1
 frac[0,34] <= 1
 frac[6,5] <= 1
 frac[8,26] <= 1
 frac[5,16] <= 1
 frac[3,32] <= 1
 frac[7,13] <= 1
 frac[3,4] <= 1
 frac[2,4] <= 1
 frac[8,4] <= 1
 frac[8,23] <= 1
 frac[6,6] <= 1
 frac[5,6] <= 1
 frac[0,16] <= 1
 frac[7,16] <= 1
 frac[1,2] <= 1
 frac[8,14] <= 1
 frac[3,3] <= 1
 frac[2,9] <= 1
 frac[8,1] <= 1
 frac[4,4] <= 1
 frac[8,20] <= 1
 frac[6,3] <= 1
 frac[7,10] <= 1
 frac[0,29] <= 1
 frac[0,0] <= 1
 frac[8,11] <= 1
 frac[3,30] <= 1
 frac[2,10] <= 1
 frac[8,30] <= 1
 frac[3,19] <= 1
 frac[8,17] <= 1
 frac[8,34] <= 1
 frac[2,0] <= 1
 frac[8,8] <= 1
 frac[4,3] <= 1
 frac[3,29] <= 1
 frac[5,2] <= 1
 cols_8_5 <= 131072
 cols_8_24 <= 131072
 cols_8_15 <= 131072
 cols_8_2 <= 131072
 cols_8_21 <= 131072
 cols_8_12 <= 131072
 cols_8_31 <= 131072
 cols_8_18 <= 131072
 cols_8_9 <= 131072
 cols_8_28 <= 131072
 cols_8_32 <= 131072
 cols_8_25 <= 131072
 cols_8_3 <= 131072
 cols_8_22 <= 131072
 cols_8_13 <= 131072
 cols_8_0 <= 131072
 cols_8_19 <= 131072
 cols_8_10 <= 131072
 cols_8_29 <= 131072
 cols_8_33 <= 131072
 cols_8_7 <= 131072
 cols_8_26 <= 131072
 cols_8_4 <= 131072
 cols_8_23 <= 131072
 cols_8_14 <= 131072
 cols_8_1 <= 131072
 cols_8_20 <= 131072
 cols_8_11 <= 131072
 cols_8_30 <= 131072
 cols_8_17 <= 131072
 cols_8_34 <= 131072
 cols_8_8 <= 131072
 mem_tot_CPU1 <= 32768
 normalized_mem_tot_CPU1 <= 1
 normalized_rows_tot_CPU1 <= 1
 mem_tot_CPU6 <= 32768
 normalized_mem_tot_CPU6 <= 1
 normalized_rows_tot_CPU6 <= 1
 mem_tot_CPU9 <= 32768
 normalized_mem_tot_CPU9 <= 1
 normalized_rows_tot_CPU9 <= 1
 mem_tot_CPU10 <= 32768
 normalized_mem_tot_CPU10 <= 1
 normalized_rows_tot_CPU10 <= 1
 mem_tot_CPU11 <= 32768
 normalized_mem_tot_CPU11 <= 1
 normalized_rows_tot_CPU11 <= 1
 mem_tot_CPU18 <= 32768
 normalized_mem_tot_CPU18 <= 1
 normalized_rows_tot_CPU18 <= 1
 mem_tot_CPU24 <= 32768
 normalized_mem_tot_CPU24 <= 1
 normalized_rows_tot_CPU24 <= 1
 mem_tot_CPU28 <= 32768
 normalized_mem_tot_CPU28 <= 1
 normalized_rows_tot_CPU28 <= 1
 mem_tot_tor_P42 <= 6144
 normalized_mem_tot_tor_P42 <= 1
 normalized_rows_tot_tor_P42 <= 1
Binaries
 cols_helper_8_5[0] cols_helper_8_5[1] cols_helper_8_5[2]
 cols_helper_8_5[3] cols_helper_8_5[4] cols_helper_8_5[5]
 cols_helper_8_5[6] cols_helper_8_5[7] cols_helper_8_5[8]
 cols_helper_8_5[9] cols_helper_8_5[10] cols_helper_8_5[11]
 cols_helper_8_5[12] cols_helper_8_5[13] cols_helper_8_5[14]
 cols_helper_8_5[15] cols_helper_8_5[16] cols_helper_8_24[0]
 cols_helper_8_24[1] cols_helper_8_24[2] cols_helper_8_24[3]
 cols_helper_8_24[4] cols_helper_8_24[5] cols_helper_8_24[6]
 cols_helper_8_24[7] cols_helper_8_24[8] cols_helper_8_24[9]
 cols_helper_8_24[10] cols_helper_8_24[11] cols_helper_8_24[12]
 cols_helper_8_24[13] cols_helper_8_24[14] cols_helper_8_24[15]
 cols_helper_8_24[16] cols_helper_8_15[0] cols_helper_8_15[1]
 cols_helper_8_15[2] cols_helper_8_15[3] cols_helper_8_15[4]
 cols_helper_8_15[5] cols_helper_8_15[6] cols_helper_8_15[7]
 cols_helper_8_15[8] cols_helper_8_15[9] cols_helper_8_15[10]
 cols_helper_8_15[11] cols_helper_8_15[12] cols_helper_8_15[13]
 cols_helper_8_15[14] cols_helper_8_15[15] cols_helper_8_15[16]
 cols_helper_8_2[0] cols_helper_8_2[1] cols_helper_8_2[2]
 cols_helper_8_2[3] cols_helper_8_2[4] cols_helper_8_2[5]
 cols_helper_8_2[6] cols_helper_8_2[7] cols_helper_8_2[8]
 cols_helper_8_2[9] cols_helper_8_2[10] cols_helper_8_2[11]
 cols_helper_8_2[12] cols_helper_8_2[13] cols_helper_8_2[14]
 cols_helper_8_2[15] cols_helper_8_2[16] cols_helper_8_21[0]
 cols_helper_8_21[1] cols_helper_8_21[2] cols_helper_8_21[3]
 cols_helper_8_21[4] cols_helper_8_21[5] cols_helper_8_21[6]
 cols_helper_8_21[7] cols_helper_8_21[8] cols_helper_8_21[9]
 cols_helper_8_21[10] cols_helper_8_21[11] cols_helper_8_21[12]
 cols_helper_8_21[13] cols_helper_8_21[14] cols_helper_8_21[15]
 cols_helper_8_21[16] cols_helper_8_12[0] cols_helper_8_12[1]
 cols_helper_8_12[2] cols_helper_8_12[3] cols_helper_8_12[4]
 cols_helper_8_12[5] cols_helper_8_12[6] cols_helper_8_12[7]
 cols_helper_8_12[8] cols_helper_8_12[9] cols_helper_8_12[10]
 cols_helper_8_12[11] cols_helper_8_12[12] cols_helper_8_12[13]
 cols_helper_8_12[14] cols_helper_8_12[15] cols_helper_8_12[16]
 cols_helper_8_31[0] cols_helper_8_31[1] cols_helper_8_31[2]
 cols_helper_8_31[3] cols_helper_8_31[4] cols_helper_8_31[5]
 cols_helper_8_31[6] cols_helper_8_31[7] cols_helper_8_31[8]
 cols_helper_8_31[9] cols_helper_8_31[10] cols_helper_8_31[11]
 cols_helper_8_31[12] cols_helper_8_31[13] cols_helper_8_31[14]
 cols_helper_8_31[15] cols_helper_8_31[16] cols_helper_8_18[0]
 cols_helper_8_18[1] cols_helper_8_18[2] cols_helper_8_18[3]
 cols_helper_8_18[4] cols_helper_8_18[5] cols_helper_8_18[6]
 cols_helper_8_18[7] cols_helper_8_18[8] cols_helper_8_18[9]
 cols_helper_8_18[10] cols_helper_8_18[11] cols_helper_8_18[12]
 cols_helper_8_18[13] cols_helper_8_18[14] cols_helper_8_18[15]
 cols_helper_8_18[16] cols_helper_8_9[0] cols_helper_8_9[1]
 cols_helper_8_9[2] cols_helper_8_9[3] cols_helper_8_9[4]
 cols_helper_8_9[5] cols_helper_8_9[6] cols_helper_8_9[7]
 cols_helper_8_9[8] cols_helper_8_9[9] cols_helper_8_9[10]
 cols_helper_8_9[11] cols_helper_8_9[12] cols_helper_8_9[13]
 cols_helper_8_9[14] cols_helper_8_9[15] cols_helper_8_9[16]
 cols_helper_8_28[0] cols_helper_8_28[1] cols_helper_8_28[2]
 cols_helper_8_28[3] cols_helper_8_28[4] cols_helper_8_28[5]
 cols_helper_8_28[6] cols_helper_8_28[7] cols_helper_8_28[8]
 cols_helper_8_28[9] cols_helper_8_28[10] cols_helper_8_28[11]
 cols_helper_8_28[12] cols_helper_8_28[13] cols_helper_8_28[14]
 cols_helper_8_28[15] cols_helper_8_28[16] cols_helper_8_32[0]
 cols_helper_8_32[1] cols_helper_8_32[2] cols_helper_8_32[3]
 cols_helper_8_32[4] cols_helper_8_32[5] cols_helper_8_32[6]
 cols_helper_8_32[7] cols_helper_8_32[8] cols_helper_8_32[9]
 cols_helper_8_32[10] cols_helper_8_32[11] cols_helper_8_32[12]
 cols_helper_8_32[13] cols_helper_8_32[14] cols_helper_8_32[15]
 cols_helper_8_32[16] cols_helper_8_25[0] cols_helper_8_25[1]
 cols_helper_8_25[2] cols_helper_8_25[3] cols_helper_8_25[4]
 cols_helper_8_25[5] cols_helper_8_25[6] cols_helper_8_25[7]
 cols_helper_8_25[8] cols_helper_8_25[9] cols_helper_8_25[10]
 cols_helper_8_25[11] cols_helper_8_25[12] cols_helper_8_25[13]
 cols_helper_8_25[14] cols_helper_8_25[15] cols_helper_8_25[16]
 cols_helper_8_3[0] cols_helper_8_3[1] cols_helper_8_3[2]
 cols_helper_8_3[3] cols_helper_8_3[4] cols_helper_8_3[5]
 cols_helper_8_3[6] cols_helper_8_3[7] cols_helper_8_3[8]
 cols_helper_8_3[9] cols_helper_8_3[10] cols_helper_8_3[11]
 cols_helper_8_3[12] cols_helper_8_3[13] cols_helper_8_3[14]
 cols_helper_8_3[15] cols_helper_8_3[16] cols_helper_8_22[0]
 cols_helper_8_22[1] cols_helper_8_22[2] cols_helper_8_22[3]
 cols_helper_8_22[4] cols_helper_8_22[5] cols_helper_8_22[6]
 cols_helper_8_22[7] cols_helper_8_22[8] cols_helper_8_22[9]
 cols_helper_8_22[10] cols_helper_8_22[11] cols_helper_8_22[12]
 cols_helper_8_22[13] cols_helper_8_22[14] cols_helper_8_22[15]
 cols_helper_8_22[16] cols_helper_8_13[0] cols_helper_8_13[1]
 cols_helper_8_13[2] cols_helper_8_13[3] cols_helper_8_13[4]
 cols_helper_8_13[5] cols_helper_8_13[6] cols_helper_8_13[7]
 cols_helper_8_13[8] cols_helper_8_13[9] cols_helper_8_13[10]
 cols_helper_8_13[11] cols_helper_8_13[12] cols_helper_8_13[13]
 cols_helper_8_13[14] cols_helper_8_13[15] cols_helper_8_13[16]
 cols_helper_8_0[0] cols_helper_8_0[1] cols_helper_8_0[2]
 cols_helper_8_0[3] cols_helper_8_0[4] cols_helper_8_0[5]
 cols_helper_8_0[6] cols_helper_8_0[7] cols_helper_8_0[8]
 cols_helper_8_0[9] cols_helper_8_0[10] cols_helper_8_0[11]
 cols_helper_8_0[12] cols_helper_8_0[13] cols_helper_8_0[14]
 cols_helper_8_0[15] cols_helper_8_0[16] cols_helper_8_19[0]
 cols_helper_8_19[1] cols_helper_8_19[2] cols_helper_8_19[3]
 cols_helper_8_19[4] cols_helper_8_19[5] cols_helper_8_19[6]
 cols_helper_8_19[7] cols_helper_8_19[8] cols_helper_8_19[9]
 cols_helper_8_19[10] cols_helper_8_19[11] cols_helper_8_19[12]
 cols_helper_8_19[13] cols_helper_8_19[14] cols_helper_8_19[15]
 cols_helper_8_19[16] cols_helper_8_10[0] cols_helper_8_10[1]
 cols_helper_8_10[2] cols_helper_8_10[3] cols_helper_8_10[4]
 cols_helper_8_10[5] cols_helper_8_10[6] cols_helper_8_10[7]
 cols_helper_8_10[8] cols_helper_8_10[9] cols_helper_8_10[10]
 cols_helper_8_10[11] cols_helper_8_10[12] cols_helper_8_10[13]
 cols_helper_8_10[14] cols_helper_8_10[15] cols_helper_8_10[16]
 cols_helper_8_29[0] cols_helper_8_29[1] cols_helper_8_29[2]
 cols_helper_8_29[3] cols_helper_8_29[4] cols_helper_8_29[5]
 cols_helper_8_29[6] cols_helper_8_29[7] cols_helper_8_29[8]
 cols_helper_8_29[9] cols_helper_8_29[10] cols_helper_8_29[11]
 cols_helper_8_29[12] cols_helper_8_29[13] cols_helper_8_29[14]
 cols_helper_8_29[15] cols_helper_8_29[16] cols_helper_8_33[0]
 cols_helper_8_33[1] cols_helper_8_33[2] cols_helper_8_33[3]
 cols_helper_8_33[4] cols_helper_8_33[5] cols_helper_8_33[6]
 cols_helper_8_33[7] cols_helper_8_33[8] cols_helper_8_33[9]
 cols_helper_8_33[10] cols_helper_8_33[11] cols_helper_8_33[12]
 cols_helper_8_33[13] cols_helper_8_33[14] cols_helper_8_33[15]
 cols_helper_8_33[16] cols_helper_8_7[0] cols_helper_8_7[1]
 cols_helper_8_7[2] cols_helper_8_7[3] cols_helper_8_7[4]
 cols_helper_8_7[5] cols_helper_8_7[6] cols_helper_8_7[7]
 cols_helper_8_7[8] cols_helper_8_7[9] cols_helper_8_7[10]
 cols_helper_8_7[11] cols_helper_8_7[12] cols_helper_8_7[13]
 cols_helper_8_7[14] cols_helper_8_7[15] cols_helper_8_7[16]
 cols_helper_8_26[0] cols_helper_8_26[1] cols_helper_8_26[2]
 cols_helper_8_26[3] cols_helper_8_26[4] cols_helper_8_26[5]
 cols_helper_8_26[6] cols_helper_8_26[7] cols_helper_8_26[8]
 cols_helper_8_26[9] cols_helper_8_26[10] cols_helper_8_26[11]
 cols_helper_8_26[12] cols_helper_8_26[13] cols_helper_8_26[14]
 cols_helper_8_26[15] cols_helper_8_26[16] cols_helper_8_4[0]
 cols_helper_8_4[1] cols_helper_8_4[2] cols_helper_8_4[3]
 cols_helper_8_4[4] cols_helper_8_4[5] cols_helper_8_4[6]
 cols_helper_8_4[7] cols_helper_8_4[8] cols_helper_8_4[9]
 cols_helper_8_4[10] cols_helper_8_4[11] cols_helper_8_4[12]
 cols_helper_8_4[13] cols_helper_8_4[14] cols_helper_8_4[15]
 cols_helper_8_4[16] cols_helper_8_23[0] cols_helper_8_23[1]
 cols_helper_8_23[2] cols_helper_8_23[3] cols_helper_8_23[4]
 cols_helper_8_23[5] cols_helper_8_23[6] cols_helper_8_23[7]
 cols_helper_8_23[8] cols_helper_8_23[9] cols_helper_8_23[10]
 cols_helper_8_23[11] cols_helper_8_23[12] cols_helper_8_23[13]
 cols_helper_8_23[14] cols_helper_8_23[15] cols_helper_8_23[16]
 cols_helper_8_14[0] cols_helper_8_14[1] cols_helper_8_14[2]
 cols_helper_8_14[3] cols_helper_8_14[4] cols_helper_8_14[5]
 cols_helper_8_14[6] cols_helper_8_14[7] cols_helper_8_14[8]
 cols_helper_8_14[9] cols_helper_8_14[10] cols_helper_8_14[11]
 cols_helper_8_14[12] cols_helper_8_14[13] cols_helper_8_14[14]
 cols_helper_8_14[15] cols_helper_8_14[16] cols_helper_8_1[0]
 cols_helper_8_1[1] cols_helper_8_1[2] cols_helper_8_1[3]
 cols_helper_8_1[4] cols_helper_8_1[5] cols_helper_8_1[6]
 cols_helper_8_1[7] cols_helper_8_1[8] cols_helper_8_1[9]
 cols_helper_8_1[10] cols_helper_8_1[11] cols_helper_8_1[12]
 cols_helper_8_1[13] cols_helper_8_1[14] cols_helper_8_1[15]
 cols_helper_8_1[16] cols_helper_8_20[0] cols_helper_8_20[1]
 cols_helper_8_20[2] cols_helper_8_20[3] cols_helper_8_20[4]
 cols_helper_8_20[5] cols_helper_8_20[6] cols_helper_8_20[7]
 cols_helper_8_20[8] cols_helper_8_20[9] cols_helper_8_20[10]
 cols_helper_8_20[11] cols_helper_8_20[12] cols_helper_8_20[13]
 cols_helper_8_20[14] cols_helper_8_20[15] cols_helper_8_20[16]
 cols_helper_8_11[0] cols_helper_8_11[1] cols_helper_8_11[2]
 cols_helper_8_11[3] cols_helper_8_11[4] cols_helper_8_11[5]
 cols_helper_8_11[6] cols_helper_8_11[7] cols_helper_8_11[8]
 cols_helper_8_11[9] cols_helper_8_11[10] cols_helper_8_11[11]
 cols_helper_8_11[12] cols_helper_8_11[13] cols_helper_8_11[14]
 cols_helper_8_11[15] cols_helper_8_11[16] cols_helper_8_30[0]
 cols_helper_8_30[1] cols_helper_8_30[2] cols_helper_8_30[3]
 cols_helper_8_30[4] cols_helper_8_30[5] cols_helper_8_30[6]
 cols_helper_8_30[7] cols_helper_8_30[8] cols_helper_8_30[9]
 cols_helper_8_30[10] cols_helper_8_30[11] cols_helper_8_30[12]
 cols_helper_8_30[13] cols_helper_8_30[14] cols_helper_8_30[15]
 cols_helper_8_30[16] cols_helper_8_17[0] cols_helper_8_17[1]
 cols_helper_8_17[2] cols_helper_8_17[3] cols_helper_8_17[4]
 cols_helper_8_17[5] cols_helper_8_17[6] cols_helper_8_17[7]
 cols_helper_8_17[8] cols_helper_8_17[9] cols_helper_8_17[10]
 cols_helper_8_17[11] cols_helper_8_17[12] cols_helper_8_17[13]
 cols_helper_8_17[14] cols_helper_8_17[15] cols_helper_8_17[16]
 cols_helper_8_34[0] cols_helper_8_34[1] cols_helper_8_34[2]
 cols_helper_8_34[3] cols_helper_8_34[4] cols_helper_8_34[5]
 cols_helper_8_34[6] cols_helper_8_34[7] cols_helper_8_34[8]
 cols_helper_8_34[9] cols_helper_8_34[10] cols_helper_8_34[11]
 cols_helper_8_34[12] cols_helper_8_34[13] cols_helper_8_34[14]
 cols_helper_8_34[15] cols_helper_8_34[16] cols_helper_8_8[0]
 cols_helper_8_8[1] cols_helper_8_8[2] cols_helper_8_8[3]
 cols_helper_8_8[4] cols_helper_8_8[5] cols_helper_8_8[6]
 cols_helper_8_8[7] cols_helper_8_8[8] cols_helper_8_8[9]
 cols_helper_8_8[10] cols_helper_8_8[11] cols_helper_8_8[12]
 cols_helper_8_8[13] cols_helper_8_8[14] cols_helper_8_8[15]
 cols_helper_8_8[16]
Generals
 cols_8_5 cols_8_24 cols_8_15 cols_8_2 cols_8_21 cols_8_12 cols_8_31
 cols_8_18 cols_8_9 cols_8_28 cols_8_32 cols_8_25 cols_8_3 cols_8_22
 cols_8_13 cols_8_0 cols_8_19 cols_8_10 cols_8_29 cols_8_33 cols_8_7
 cols_8_26 cols_8_4 cols_8_23 cols_8_14 cols_8_1 cols_8_20 cols_8_11
 cols_8_30 cols_8_17 cols_8_34 cols_8_8
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_tor_P42 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU1 , normalized_mem_tot_CPU6 ,
   normalized_mem_tot_CPU9 , normalized_mem_tot_CPU10 ,
   normalized_mem_tot_CPU11 , normalized_mem_tot_CPU18 ,
   normalized_mem_tot_CPU24 , normalized_mem_tot_CPU28 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_tor_P42 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU1 ,
   normalized_rows_tot_CPU6 , normalized_rows_tot_CPU9 ,
   normalized_rows_tot_CPU10 , normalized_rows_tot_CPU11 ,
   normalized_rows_tot_CPU18 , normalized_rows_tot_CPU24 ,
   normalized_rows_tot_CPU28 )
End

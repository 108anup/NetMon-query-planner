\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU_Cluster: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU2 + rows_tot_CPU4 + rows_tot_CPU13 + rows_tot_CPU18
   + rows_tot_CPU21 + rows_tot_CPU24 + rows_tot_CPU26 + rows_tot_CPU32
  CPU_Cluster_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU_Cluster
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_rows_P4: Priority=80 Weight=1 AbsTol=0 RelTol=0
   rows_tot_tor_P43 + rows_tot_l1_P42
  P4_rows_load: Priority=70 Weight=1 AbsTol=0 RelTol=0
   max_rows_P4
  tot_mem_CPU_Cluster: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU2 + mem_tot_CPU4 + mem_tot_CPU13 + mem_tot_CPU18
   + mem_tot_CPU21 + mem_tot_CPU24 + mem_tot_CPU26 + mem_tot_CPU32
  CPU_Cluster_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
  tot_mem_P4: Priority=40 Weight=1 AbsTol=0 RelTol=0
   mem_tot_tor_P43 + mem_tot_l1_P42
  P4_load_mem: Priority=30 Weight=1 AbsTol=0 RelTol=0
   max_mem_P4
Subject To
 cov_0_8: frac[8,8] + frac[9,8] >= 0.8
 cov_1_30: frac[8,30] + frac[9,30] >= 0.7
 cov_2_27: frac[9,27] >= 0.46333333333334
 cov_3_3: frac[8,3] + frac[9,3] >= 0.1
 cov_4_22: frac[8,22] + frac[9,22] >= 0.1
 cov_5_25: frac[8,25] + frac[9,25] >= 0.8
 cov_6_19: frac[4,19] + frac[8,19] + frac[9,19] >= 0.7
 cov_7_8: frac[8,8] + frac[9,8] >= 0.8
 cov_8_25: frac[8,25] + frac[9,25] >= 0.8
 cov_9_36: frac[8,36] + frac[9,36] >= 0.2
 cov_10_35: frac[8,35] + frac[9,35] >= 0.3
 cov_11_22: frac[8,22] + frac[9,22] >= 0.1
 cov_12_10: frac[8,10] >= 0.8
 cov_13_27: frac[8,27] + frac[9,27] >= 0.46333333333334
 cov_14_1: frac[9,1] >= 0.8
 cov_15_36: frac[8,36] + frac[9,36] >= 0.2
 cov_16_3: frac[9,3] >= 0.1
 cov_17_1: frac[8,1] + frac[9,1] >= 1
 cov_18_35: frac[8,35] + frac[9,35] >= 0.3
 cov_19_1: frac[9,1] >= 0.8
 cov_20_27: frac[8,27] + frac[9,27] >= 0.46333333333334
 cov_21_19: frac[8,19] >= 1
 cov_22_15: frac[8,15] + frac[9,15] >= 0.8
 cov_23_1: frac[8,1] + frac[9,1] >= 0.7
 cov_24_30: frac[8,30] + frac[9,30] >= 0.7
 cov_25_13: frac[8,13] + frac[9,13] >= 0.1
 cov_26_13: frac[8,13] + frac[9,13] >= 0.1
 cov_27_13: frac[2,13] >= 0.1
 cov_28_13: frac[9,13] >= 0.1
 cov_29_33: frac[8,33] + frac[9,33] >= 0.8
 cov_30_33: frac[9,33] >= 0.8
 cov_31_0: frac[8,0] + frac[9,0] >= 0.9
 cov_32_32: frac[8,32] + frac[9,32] >= 0.9
 cov_33_23: frac[8,23] + frac[9,23] >= 0.1
 cov_34_16: frac[9,16] >= 0.2
 cov_35_20: frac[8,20] + frac[9,20] >= 0.1
 cov_36_16: frac[8,16] + frac[9,16] >= 0.2
 cov_37_0: frac[8,0] + frac[9,0] >= 0.9
 cov_38_16: frac[8,16] + frac[9,16] >= 0.2
 cov_39_12: frac[1,12] + frac[3,12] + frac[8,12] + frac[9,12] >= 0.8
 cov_40_17: frac[2,17] + frac[6,17] + frac[9,17] >= 0.9
 cov_41_17: frac[5,17] + frac[8,17] + frac[9,17] >= 0.7
 cov_42_2: frac[4,2] + frac[7,2] + frac[8,2] + frac[9,2] >= 0.8
 cov_43_2: frac[2,2] + frac[4,2] + frac[8,2] + frac[9,2] >= 0.7
 cov_44_28: frac[0,28] + frac[1,28] >= 1
 cov_45_21: frac[6,21] + frac[7,21] >= 0.7
 cov_46_31: frac[0,31] + frac[4,31] + frac[8,31] + frac[9,31] >= 0.9
 cov_47_17: frac[0,17] + frac[2,17] >= 0.8
 cov_48_14: frac[1,14] + frac[2,14] >= 0.8
 cov_49_7: frac[0,7] + frac[1,7] >= 0.9
 cov_50_28: frac[3,28] + frac[7,28] + frac[8,28] + frac[9,28] >= 0.8
 cov_51_29: frac[2,29] + frac[3,29] + frac[8,29] + frac[9,29]
   >= 0.46333333333334
 cov_52_28: frac[2,28] + frac[4,28] + frac[8,28] + frac[9,28] >= 0.9
 cov_53_9: frac[6,9] >= 0.8
 cov_54_31: frac[0,31] + frac[1,31] >= 0.9
 cov_55_6: frac[0,6] + frac[1,6] >= 0.9
 cov_56_11: frac[0,11] + frac[1,11] >= 1
 cov_57_6: frac[2,6] + frac[3,6] + frac[8,6] + frac[9,6] >= 0.9
 cov_58_5: frac[3,5] + frac[7,5] + frac[8,5] + frac[9,5] >= 0.7
 cov_59_14: frac[2,14] + frac[5,14] + frac[8,14] + frac[9,14] >= 0.8
 cov_60_2: frac[4,2] + frac[7,2] + frac[8,2] + frac[9,2] >= 0.9
 cov_61_34: frac[0,34] + frac[1,34] >= 1
 cov_62_29: frac[1,29] + frac[3,29] + frac[8,29] + frac[9,29]
   >= 0.46333333333334
 cov_63_26: frac[1,26] + frac[3,26] + frac[8,26] + frac[9,26] >= 0.8
 cov_64_2: frac[4,2] + frac[7,2] + frac[8,2] + frac[9,2] >= 0.7
 cov_65_29: frac[1,29] + frac[6,29] + frac[9,29] >= 0.46333333333334
 cov_66_18: frac[0,18] + frac[5,18] + frac[8,18] + frac[9,18] >= 0.9
 cov_67_7: frac[1,7] + frac[4,7] + frac[8,7] + frac[9,7] >= 0.7
 cov_68_28: frac[1,28] + frac[4,28] + frac[8,28] + frac[9,28] >= 1
 cov_69_2: frac[2,2] + frac[7,2] + frac[9,2] >= 0.9
 cov_70_24: frac[2,24] + frac[5,24] + frac[8,24] + frac[9,24] >= 0.7
 cov_71_28: frac[0,28] + frac[5,28] + frac[8,28] + frac[9,28] >= 0.8
 cov_72_5: frac[2,5] + frac[4,5] + frac[8,5] + frac[9,5] >= 1
 cov_73_21: frac[4,21] + frac[7,21] + frac[8,21] + frac[9,21] >= 0.8
 cov_74_34: frac[2,34] + frac[3,34] + frac[8,34] + frac[9,34] >= 0.7
 cov_75_5: frac[4,5] + frac[8,5] >= 1
 cov_76_4: frac[2,4] + frac[3,4] + frac[8,4] + frac[9,4] >= 0.8
 cov_77_24: frac[0,24] + frac[5,24] + frac[8,24] + frac[9,24] >= 1
 cov_78_31: frac[3,31] + frac[6,31] + frac[8,31] + frac[9,31] >= 0.7
 cov_79_24: frac[3,24] + frac[5,24] + frac[8,24] >= 1
 cov_80_12: frac[0,12] + frac[3,12] + frac[8,12] + frac[9,12] >= 0.8
 cov_81_31: frac[2,31] + frac[7,31] + frac[9,31] >= 0.9
 cov_82_34: frac[3,34] + frac[4,34] + frac[8,34] >= 0.8
 cov_83_12: frac[2,12] + frac[9,12] >= 0.8
 cov_84_14: frac[4,14] + frac[7,14] + frac[8,14] + frac[9,14] >= 0.7
 cov_85_21: frac[2,21] + frac[3,21] + frac[8,21] + frac[9,21] >= 1
 cov_86_14: frac[4,14] + frac[5,14] + frac[8,14] >= 0.8
 cov_87_17: frac[5,17] + frac[8,17] + frac[9,17] >= 0.9
 cov_88_12: frac[2,12] + frac[4,12] + frac[8,12] + frac[9,12] >= 0.8
 cov_89_4: frac[5,4] + frac[8,4] + frac[9,4] >= 0.8
 cov_90_24: frac[1,24] + frac[3,24] + frac[8,24] + frac[9,24] >= 1
 cov_91_5: frac[2,5] + frac[8,5] + frac[9,5] >= 0.8
 accuracy_1_34: - 5.09765625 frac[1,34] + mem[1,34] = 0
 accuracy_9_26: - 6 frac[9,26] + mem[9,26] = 0
 accuracy_6_9: - 5.09765625 frac[6,9] + mem[6,9] = 0
 accuracy_0_7: - 5.09765625 frac[0,7] + mem[0,7] = 0
 accuracy_1_6: - 5.09765625 frac[1,6] + mem[1,6] = 0
 accuracy_2_5: - 5.09765625 frac[2,5] + mem[2,5] = 0
 accuracy_1_11: - 5.09765625 frac[1,11] + mem[1,11] = 0
 accuracy_8_5: - 6 frac[8,5] + mem[8,5] = 0
 accuracy_3_24: - 5.09765625 frac[3,24] + mem[3,24] = 0
 accuracy_9_0: - 6 frac[9,0] + mem[9,0] = 0
 accuracy_8_24: - 6 frac[8,24] + mem[8,24] = 0
 accuracy_4_19: - 5.09765625 frac[4,19] + mem[4,19] = 0
 accuracy_9_21: - 6 frac[9,21] + mem[9,21] = 0
 accuracy_5_18: - 5.09765625 frac[5,18] + mem[5,18] = 0
 accuracy_0_17: - 5.09765625 frac[0,17] + mem[0,17] = 0
 accuracy_9_36: - 6 frac[9,36] + mem[9,36] = 0
 accuracy_1_28: - 5.09765625 frac[1,28] + mem[1,28] = 0
 accuracy_7_28: - 5.09765625 frac[7,28] + mem[7,28] = 0
 accuracy_8_15: - 6 frac[8,15] + mem[8,15] = 0
 accuracy_2_6: - 5.09765625 frac[2,6] + mem[2,6] = 0
 accuracy_9_14: - 6 frac[9,14] + mem[9,14] = 0
 accuracy_8_2: - 6 frac[8,2] + mem[8,2] = 0
 accuracy_4_5: - 5.09765625 frac[4,5] + mem[4,5] = 0
 accuracy_9_3: - 6 frac[9,3] + mem[9,3] = 0
 accuracy_8_21: - 6 frac[8,21] + mem[8,21] = 0
 accuracy_5_24: - 5.09765625 frac[5,24] + mem[5,24] = 0
 accuracy_9_16: - 6 frac[9,16] + mem[9,16] = 0
 accuracy_7_5: - 5.09765625 frac[7,5] + mem[7,5] = 0
 accuracy_2_17: - 5.09765625 frac[2,17] + mem[2,17] = 0
 accuracy_1_31: - 5.09765625 frac[1,31] + mem[1,31] = 0
 accuracy_3_12: - 5.09765625 frac[3,12] + mem[3,12] = 0
 accuracy_2_28: - 5.09765625 frac[2,28] + mem[2,28] = 0
 accuracy_1_12: - 5.09765625 frac[1,12] + mem[1,12] = 0
 accuracy_8_12: - 6 frac[8,12] + mem[8,12] = 0
 accuracy_8_31: - 6 frac[8,31] + mem[8,31] = 0
 accuracy_5_14: - 5.09765625 frac[5,14] + mem[5,14] = 0
 accuracy_9_30: - 6 frac[9,30] + mem[9,30] = 0
 accuracy_8_18: - 6 frac[8,18] + mem[8,18] = 0
 accuracy_4_21: - 5.09765625 frac[4,21] + mem[4,21] = 0
 accuracy_9_19: - 6 frac[9,19] + mem[9,19] = 0
 accuracy_8_35: - 6 frac[8,35] + mem[8,35] = 0
 accuracy_7_21: - 5.09765625 frac[7,21] + mem[7,21] = 0
 accuracy_1_26: - 5.09765625 frac[1,26] + mem[1,26] = 0
 accuracy_9_34: - 6 frac[9,34] + mem[9,34] = 0
 accuracy_4_12: - 5.09765625 frac[4,12] + mem[4,12] = 0
 accuracy_3_28: - 5.09765625 frac[3,28] + mem[3,28] = 0
 accuracy_2_12: - 5.09765625 frac[2,12] + mem[2,12] = 0
 accuracy_9_4: - 6 frac[9,4] + mem[9,4] = 0
 accuracy_8_28: - 6 frac[8,28] + mem[8,28] = 0
 accuracy_4_31: - 5.09765625 frac[4,31] + mem[4,31] = 0
 accuracy_9_25: - 6 frac[9,25] + mem[9,25] = 0
 accuracy_7_2: - 5.09765625 frac[7,2] + mem[7,2] = 0
 accuracy_3_34: - 5.09765625 frac[3,34] + mem[3,34] = 0
 accuracy_6_29: - 5.09765625 frac[6,29] + mem[6,29] = 0
 accuracy_8_32: - 6 frac[8,32] + mem[8,32] = 0
 accuracy_0_24: - 5.09765625 frac[0,24] + mem[0,24] = 0
 accuracy_0_11: - 5.09765625 frac[0,11] + mem[0,11] = 0
 accuracy_3_6: - 5.09765625 frac[3,6] + mem[3,6] = 0
 accuracy_2_2: - 5.09765625 frac[2,2] + mem[2,2] = 0
 accuracy_8_6: - 6 frac[8,6] + mem[8,6] = 0
 accuracy_9_7: - 6 frac[9,7] + mem[9,7] = 0
 accuracy_8_25: - 6 frac[8,25] + mem[8,25] = 0
 accuracy_5_4: - 5.09765625 frac[5,4] + mem[5,4] = 0
 accuracy_4_28: - 5.09765625 frac[4,28] + mem[4,28] = 0
 accuracy_9_20: - 6 frac[9,20] + mem[9,20] = 0
 accuracy_5_17: - 5.09765625 frac[5,17] + mem[5,17] = 0
 accuracy_0_18: - 5.09765625 frac[0,18] + mem[0,18] = 0
 accuracy_7_31: - 5.09765625 frac[7,31] + mem[7,31] = 0
 accuracy_2_24: - 5.09765625 frac[2,24] + mem[2,24] = 0
 accuracy_3_5: - 5.09765625 frac[3,5] + mem[3,5] = 0
 accuracy_9_13: - 6 frac[9,13] + mem[9,13] = 0
 accuracy_8_3: - 6 frac[8,3] + mem[8,3] = 0
 accuracy_9_2: - 6 frac[9,2] + mem[9,2] = 0
 accuracy_8_22: - 6 frac[8,22] + mem[8,22] = 0
 accuracy_9_23: - 6 frac[9,23] + mem[9,23] = 0
 accuracy_0_31: - 5.09765625 frac[0,31] + mem[0,31] = 0
 accuracy_2_29: - 5.09765625 frac[2,29] + mem[2,29] = 0
 accuracy_8_13: - 6 frac[8,13] + mem[8,13] = 0
 accuracy_9_8: - 6 frac[9,8] + mem[9,8] = 0
 accuracy_8_0: - 6 frac[8,0] + mem[8,0] = 0
 accuracy_3_21: - 5.09765625 frac[3,21] + mem[3,21] = 0
 accuracy_9_29: - 6 frac[9,29] + mem[9,29] = 0
 accuracy_8_19: - 6 frac[8,19] + mem[8,19] = 0
 accuracy_2_34: - 5.09765625 frac[2,34] + mem[2,34] = 0
 accuracy_9_18: - 6 frac[9,18] + mem[9,18] = 0
 accuracy_6_17: - 5.09765625 frac[6,17] + mem[6,17] = 0
 accuracy_8_36: - 6 frac[8,36] + mem[8,36] = 0
 accuracy_0_28: - 5.09765625 frac[0,28] + mem[0,28] = 0
 accuracy_9_33: - 6 frac[9,33] + mem[9,33] = 0
 accuracy_1_14: - 5.09765625 frac[1,14] + mem[1,14] = 0
 accuracy_8_10: - 6 frac[8,10] + mem[8,10] = 0
 accuracy_3_31: - 5.09765625 frac[3,31] + mem[3,31] = 0
 accuracy_2_13: - 5.09765625 frac[2,13] + mem[2,13] = 0
 accuracy_8_29: - 6 frac[8,29] + mem[8,29] = 0
 accuracy_9_24: - 6 frac[9,24] + mem[9,24] = 0
 accuracy_8_16: - 6 frac[8,16] + mem[8,16] = 0
 accuracy_7_14: - 5.09765625 frac[7,14] + mem[7,14] = 0
 accuracy_8_33: - 6 frac[8,33] + mem[8,33] = 0
 accuracy_0_12: - 5.09765625 frac[0,12] + mem[0,12] = 0
 accuracy_8_7: - 6 frac[8,7] + mem[8,7] = 0
 accuracy_4_2: - 5.09765625 frac[4,2] + mem[4,2] = 0
 accuracy_3_26: - 5.09765625 frac[3,26] + mem[3,26] = 0
 accuracy_2_14: - 5.09765625 frac[2,14] + mem[2,14] = 0
 accuracy_9_6: - 6 frac[9,6] + mem[9,6] = 0
 accuracy_0_34: - 5.09765625 frac[0,34] + mem[0,34] = 0
 accuracy_8_26: - 6 frac[8,26] + mem[8,26] = 0
 accuracy_9_27: - 6 frac[9,27] + mem[9,27] = 0
 accuracy_6_31: - 5.09765625 frac[6,31] + mem[6,31] = 0
 accuracy_0_6: - 5.09765625 frac[0,6] + mem[0,6] = 0
 accuracy_1_7: - 5.09765625 frac[1,7] + mem[1,7] = 0
 accuracy_3_4: - 5.09765625 frac[3,4] + mem[3,4] = 0
 accuracy_2_4: - 5.09765625 frac[2,4] + mem[2,4] = 0
 accuracy_9_12: - 6 frac[9,12] + mem[9,12] = 0
 accuracy_8_4: - 6 frac[8,4] + mem[8,4] = 0
 accuracy_4_7: - 5.09765625 frac[4,7] + mem[4,7] = 0
 accuracy_9_1: - 6 frac[9,1] + mem[9,1] = 0
 accuracy_8_23: - 6 frac[8,23] + mem[8,23] = 0
 accuracy_9_22: - 6 frac[9,22] + mem[9,22] = 0
 accuracy_6_21: - 5.09765625 frac[6,21] + mem[6,21] = 0
 accuracy_1_29: - 5.09765625 frac[1,29] + mem[1,29] = 0
 accuracy_8_14: - 6 frac[8,14] + mem[8,14] = 0
 accuracy_9_15: - 6 frac[9,15] + mem[9,15] = 0
 accuracy_8_1: - 6 frac[8,1] + mem[8,1] = 0
 accuracy_9_28: - 6 frac[9,28] + mem[9,28] = 0
 accuracy_8_20: - 6 frac[8,20] + mem[8,20] = 0
 accuracy_9_17: - 6 frac[9,17] + mem[9,17] = 0
 accuracy_4_34: - 5.09765625 frac[4,34] + mem[4,34] = 0
 accuracy_9_32: - 6 frac[9,32] + mem[9,32] = 0
 accuracy_1_24: - 5.09765625 frac[1,24] + mem[1,24] = 0
 accuracy_2_31: - 5.09765625 frac[2,31] + mem[2,31] = 0
 accuracy_4_14: - 5.09765625 frac[4,14] + mem[4,14] = 0
 accuracy_8_30: - 6 frac[8,30] + mem[8,30] = 0
 accuracy_9_31: - 6 frac[9,31] + mem[9,31] = 0
 accuracy_8_17: - 6 frac[8,17] + mem[8,17] = 0
 accuracy_5_28: - 5.09765625 frac[5,28] + mem[5,28] = 0
 accuracy_8_34: - 6 frac[8,34] + mem[8,34] = 0
 accuracy_2_21: - 5.09765625 frac[2,21] + mem[2,21] = 0
 accuracy_9_35: - 6 frac[9,35] + mem[9,35] = 0
 accuracy_8_8: - 6 frac[8,8] + mem[8,8] = 0
 accuracy_3_29: - 5.09765625 frac[3,29] + mem[3,29] = 0
 accuracy_9_5: - 6 frac[9,5] + mem[9,5] = 0
 accuracy_8_27: - 6 frac[8,27] + mem[8,27] = 0
 mem_tot_CPU2: - mem[0,7] - mem[0,17] - mem[0,24] - mem[0,11] - mem[0,18]
   - mem[0,31] - mem[0,28] - mem[0,12] - mem[0,34] - mem[0,6]
   + mem_tot_CPU2 = 0
 R235: - 3.05176e-05 mem_tot_CPU2 + normalized_mem_tot_CPU2 = 0
 rows_tot_CPU2: - 3 frac[0,7] - 3 frac[0,17] - 3 frac[0,24] - 3 frac[0,11]
   - 3 frac[0,18] - 3 frac[0,31] - 3 frac[0,28] - 3 frac[0,12]
   - 3 frac[0,34] - 3 frac[0,6] + rows_tot_CPU2 = 0
 R237: - 0.0833333333333333 rows_tot_CPU2 + normalized_rows_tot_CPU2 = 0
 mem_tot_CPU4: - mem[1,34] - mem[1,6] - mem[1,11] - mem[1,28] - mem[1,31]
   - mem[1,12] - mem[1,26] - mem[1,14] - mem[1,7] - mem[1,29] - mem[1,24]
   + mem_tot_CPU4 = 0
 R239: - 3.05176e-05 mem_tot_CPU4 + normalized_mem_tot_CPU4 = 0
 rows_tot_CPU4: - 3 frac[1,34] - 3 frac[1,6] - 3 frac[1,11] - 3 frac[1,28]
   - 3 frac[1,31] - 3 frac[1,12] - 3 frac[1,26] - 3 frac[1,14]
   - 3 frac[1,7] - 3 frac[1,29] - 3 frac[1,24] + rows_tot_CPU4 = 0
 R241: - 0.0833333333333333 rows_tot_CPU4 + normalized_rows_tot_CPU4 = 0
 mem_tot_CPU13: - mem[2,5] - mem[2,6] - mem[2,17] - mem[2,28] - mem[2,12]
   - mem[2,2] - mem[2,24] - mem[2,29] - mem[2,34] - mem[2,13] - mem[2,14]
   - mem[2,4] - mem[2,31] - mem[2,21] + mem_tot_CPU13 = 0
 R243: - 3.05176e-05 mem_tot_CPU13 + normalized_mem_tot_CPU13 = 0
 rows_tot_CPU13: - 3 frac[2,5] - 3 frac[2,6] - 3 frac[2,17] - 3 frac[2,28]
   - 3 frac[2,12] - 3 frac[2,2] - 3 frac[2,24] - 3 frac[2,29]
   - 3 frac[2,34] - 3 frac[2,13] - 3 frac[2,14] - 3 frac[2,4]
   - 3 frac[2,31] - 3 frac[2,21] + rows_tot_CPU13 = 0
 R245: - 0.0833333333333333 rows_tot_CPU13 + normalized_rows_tot_CPU13 = 0
 mem_tot_CPU18: - mem[3,24] - mem[3,12] - mem[3,28] - mem[3,34] - mem[3,6]
   - mem[3,5] - mem[3,21] - mem[3,31] - mem[3,26] - mem[3,4] - mem[3,29]
   + mem_tot_CPU18 = 0
 R247: - 3.05176e-05 mem_tot_CPU18 + normalized_mem_tot_CPU18 = 0
 rows_tot_CPU18: - 3 frac[3,24] - 3 frac[3,12] - 3 frac[3,28]
   - 3 frac[3,34] - 3 frac[3,6] - 3 frac[3,5] - 3 frac[3,21] - 3 frac[3,31]
   - 3 frac[3,26] - 3 frac[3,4] - 3 frac[3,29] + rows_tot_CPU18 = 0
 R249: - 0.0833333333333333 rows_tot_CPU18 + normalized_rows_tot_CPU18 = 0
 mem_tot_CPU21: - mem[4,19] - mem[4,5] - mem[4,21] - mem[4,12] - mem[4,31]
   - mem[4,28] - mem[4,2] - mem[4,7] - mem[4,34] - mem[4,14]
   + mem_tot_CPU21 = 0
 R251: - 3.05176e-05 mem_tot_CPU21 + normalized_mem_tot_CPU21 = 0
 rows_tot_CPU21: - 3 frac[4,19] - 3 frac[4,5] - 3 frac[4,21] - 3 frac[4,12]
   - 3 frac[4,31] - 3 frac[4,28] - 3 frac[4,2] - 3 frac[4,7] - 3 frac[4,34]
   - 3 frac[4,14] + rows_tot_CPU21 = 0
 R253: - 0.0833333333333333 rows_tot_CPU21 + normalized_rows_tot_CPU21 = 0
 mem_tot_CPU24: - mem[5,18] - mem[5,24] - mem[5,14] - mem[5,4] - mem[5,17]
   - mem[5,28] + mem_tot_CPU24 = 0
 R255: - 3.05176e-05 mem_tot_CPU24 + normalized_mem_tot_CPU24 = 0
 rows_tot_CPU24: - 3 frac[5,18] - 3 frac[5,24] - 3 frac[5,14] - 3 frac[5,4]
   - 3 frac[5,17] - 3 frac[5,28] + rows_tot_CPU24 = 0
 R257: - 0.0833333333333333 rows_tot_CPU24 + normalized_rows_tot_CPU24 = 0
 mem_tot_CPU26: - mem[6,9] - mem[6,29] - mem[6,17] - mem[6,31] - mem[6,21]
   + mem_tot_CPU26 = 0
 R259: - 3.05176e-05 mem_tot_CPU26 + normalized_mem_tot_CPU26 = 0
 rows_tot_CPU26: - 3 frac[6,9] - 3 frac[6,29] - 3 frac[6,17] - 3 frac[6,31]
   - 3 frac[6,21] + rows_tot_CPU26 = 0
 R261: - 0.0833333333333333 rows_tot_CPU26 + normalized_rows_tot_CPU26 = 0
 mem_tot_CPU32: - mem[7,28] - mem[7,5] - mem[7,21] - mem[7,2] - mem[7,31]
   - mem[7,14] + mem_tot_CPU32 = 0
 R263: - 3.05176e-05 mem_tot_CPU32 + normalized_mem_tot_CPU32 = 0
 rows_tot_CPU32: - 3 frac[7,28] - 3 frac[7,5] - 3 frac[7,21] - 3 frac[7,2]
   - 3 frac[7,31] - 3 frac[7,14] + rows_tot_CPU32 = 0
 R265: - 0.0833333333333333 rows_tot_CPU32 + normalized_rows_tot_CPU32 = 0
 mem_tot_tor_P43: - mem[8,5] - mem[8,24] - mem[8,15] - mem[8,2] - mem[8,21]
   - mem[8,12] - mem[8,31] - mem[8,18] - mem[8,35] - mem[8,28] - mem[8,32]
   - mem[8,6] - mem[8,25] - mem[8,3] - mem[8,22] - mem[8,13] - mem[8,0]
   - mem[8,19] - mem[8,36] - mem[8,10] - mem[8,29] - mem[8,16] - mem[8,33]
   - mem[8,7] - mem[8,26] - mem[8,4] - mem[8,23] - mem[8,14] - mem[8,1]
   - mem[8,20] - mem[8,30] - mem[8,17] - mem[8,34] - mem[8,8] - mem[8,27]
   + mem_tot_tor_P43 = 0
 R267: - 1.6276041666666666e-04 mem_tot_tor_P43
   + normalized_mem_tot_tor_P43 = 0
 rows_tot_tor_P43: - 3 frac[8,5] - 3 frac[8,24] - 3 frac[8,15]
   - 3 frac[8,2] - 3 frac[8,21] - 3 frac[8,12] - 3 frac[8,31]
   - 3 frac[8,18] - 3 frac[8,35] - 3 frac[8,28] - 3 frac[8,32]
   - 3 frac[8,6] - 3 frac[8,25] - 3 frac[8,3] - 3 frac[8,22] - 3 frac[8,13]
   - 3 frac[8,0] - 3 frac[8,19] - 3 frac[8,36] - 3 frac[8,10]
   - 3 frac[8,29] - 3 frac[8,16] - 3 frac[8,33] - 3 frac[8,7]
   - 3 frac[8,26] - 3 frac[8,4] - 3 frac[8,23] - 3 frac[8,14] - 3 frac[8,1]
   - 3 frac[8,20] - 3 frac[8,30] - 3 frac[8,17] - 3 frac[8,34]
   - 3 frac[8,8] - 3 frac[8,27] + rows_tot_tor_P43 = 0
 R269: - 0.0208333333333333 rows_tot_tor_P43 + normalized_rows_tot_tor_P43
   = 0
 mem_tot_l1_P42: - mem[9,26] - mem[9,0] - mem[9,21] - mem[9,36] - mem[9,14]
   - mem[9,3] - mem[9,16] - mem[9,30] - mem[9,19] - mem[9,34] - mem[9,4]
   - mem[9,25] - mem[9,7] - mem[9,20] - mem[9,13] - mem[9,2] - mem[9,23]
   - mem[9,8] - mem[9,29] - mem[9,18] - mem[9,33] - mem[9,24] - mem[9,6]
   - mem[9,27] - mem[9,12] - mem[9,1] - mem[9,22] - mem[9,15] - mem[9,28]
   - mem[9,17] - mem[9,32] - mem[9,31] - mem[9,35] - mem[9,5]
   + mem_tot_l1_P42 = 0
 R271: - 1.6276041666666666e-04 mem_tot_l1_P42 + normalized_mem_tot_l1_P42
   = 0
 rows_tot_l1_P42: - 3 frac[9,26] - 3 frac[9,0] - 3 frac[9,21]
   - 3 frac[9,36] - 3 frac[9,14] - 3 frac[9,3] - 3 frac[9,16]
   - 3 frac[9,30] - 3 frac[9,19] - 3 frac[9,34] - 3 frac[9,4]
   - 3 frac[9,25] - 3 frac[9,7] - 3 frac[9,20] - 3 frac[9,13] - 3 frac[9,2]
   - 3 frac[9,23] - 3 frac[9,8] - 3 frac[9,29] - 3 frac[9,18]
   - 3 frac[9,33] - 3 frac[9,24] - 3 frac[9,6] - 3 frac[9,27]
   - 3 frac[9,12] - 3 frac[9,1] - 3 frac[9,22] - 3 frac[9,15]
   - 3 frac[9,28] - 3 frac[9,17] - 3 frac[9,32] - 3 frac[9,31]
   - 3 frac[9,35] - 3 frac[9,5] + rows_tot_l1_P42 = 0
 R273: - 0.0208333333333333 rows_tot_l1_P42 + normalized_rows_tot_l1_P42
   = 0
 row_capacity_CPU2: rows_tot_CPU2 <= 12
 row_capacity_CPU4: rows_tot_CPU4 <= 12
 row_capacity_CPU13: rows_tot_CPU13 <= 12
 row_capacity_CPU18: rows_tot_CPU18 <= 12
 row_capacity_CPU21: rows_tot_CPU21 <= 12
 row_capacity_CPU24: rows_tot_CPU24 <= 12
 row_capacity_CPU26: rows_tot_CPU26 <= 12
 row_capacity_CPU32: rows_tot_CPU32 <= 12
 row_capacity_tor_P43: rows_tot_tor_P43 <= 48
 capacity_mem_par_tor_P43: mem[8,5] <= 1536
 capacity_mem_par_tor_P43: mem[8,24] <= 1536
 capacity_mem_par_tor_P43: mem[8,15] <= 1536
 capacity_mem_par_tor_P43: mem[8,2] <= 1536
 capacity_mem_par_tor_P43: mem[8,21] <= 1536
 capacity_mem_par_tor_P43: mem[8,12] <= 1536
 capacity_mem_par_tor_P43: mem[8,31] <= 1536
 capacity_mem_par_tor_P43: mem[8,18] <= 1536
 capacity_mem_par_tor_P43: mem[8,35] <= 1536
 capacity_mem_par_tor_P43: mem[8,28] <= 1536
 capacity_mem_par_tor_P43: mem[8,32] <= 1536
 capacity_mem_par_tor_P43: mem[8,6] <= 1536
 capacity_mem_par_tor_P43: mem[8,25] <= 1536
 capacity_mem_par_tor_P43: mem[8,3] <= 1536
 capacity_mem_par_tor_P43: mem[8,22] <= 1536
 capacity_mem_par_tor_P43: mem[8,13] <= 1536
 capacity_mem_par_tor_P43: mem[8,0] <= 1536
 capacity_mem_par_tor_P43: mem[8,19] <= 1536
 capacity_mem_par_tor_P43: mem[8,36] <= 1536
 capacity_mem_par_tor_P43: mem[8,10] <= 1536
 capacity_mem_par_tor_P43: mem[8,29] <= 1536
 capacity_mem_par_tor_P43: mem[8,16] <= 1536
 capacity_mem_par_tor_P43: mem[8,33] <= 1536
 capacity_mem_par_tor_P43: mem[8,7] <= 1536
 capacity_mem_par_tor_P43: mem[8,26] <= 1536
 capacity_mem_par_tor_P43: mem[8,4] <= 1536
 capacity_mem_par_tor_P43: mem[8,23] <= 1536
 capacity_mem_par_tor_P43: mem[8,14] <= 1536
 capacity_mem_par_tor_P43: mem[8,1] <= 1536
 capacity_mem_par_tor_P43: mem[8,20] <= 1536
 capacity_mem_par_tor_P43: mem[8,30] <= 1536
 capacity_mem_par_tor_P43: mem[8,17] <= 1536
 capacity_mem_par_tor_P43: mem[8,34] <= 1536
 capacity_mem_par_tor_P43: mem[8,8] <= 1536
 capacity_mem_par_tor_P43: mem[8,27] <= 1536
 row_capacity_l1_P42: rows_tot_l1_P42 <= 48
 capacity_mem_par_l1_P42: mem[9,26] <= 1536
 capacity_mem_par_l1_P42: mem[9,0] <= 1536
 capacity_mem_par_l1_P42: mem[9,21] <= 1536
 capacity_mem_par_l1_P42: mem[9,36] <= 1536
 capacity_mem_par_l1_P42: mem[9,14] <= 1536
 capacity_mem_par_l1_P42: mem[9,3] <= 1536
 capacity_mem_par_l1_P42: mem[9,16] <= 1536
 capacity_mem_par_l1_P42: mem[9,30] <= 1536
 capacity_mem_par_l1_P42: mem[9,19] <= 1536
 capacity_mem_par_l1_P42: mem[9,34] <= 1536
 capacity_mem_par_l1_P42: mem[9,4] <= 1536
 capacity_mem_par_l1_P42: mem[9,25] <= 1536
 capacity_mem_par_l1_P42: mem[9,7] <= 1536
 capacity_mem_par_l1_P42: mem[9,20] <= 1536
 capacity_mem_par_l1_P42: mem[9,13] <= 1536
 capacity_mem_par_l1_P42: mem[9,2] <= 1536
 capacity_mem_par_l1_P42: mem[9,23] <= 1536
 capacity_mem_par_l1_P42: mem[9,8] <= 1536
 capacity_mem_par_l1_P42: mem[9,29] <= 1536
 capacity_mem_par_l1_P42: mem[9,18] <= 1536
 capacity_mem_par_l1_P42: mem[9,33] <= 1536
 capacity_mem_par_l1_P42: mem[9,24] <= 1536
 capacity_mem_par_l1_P42: mem[9,6] <= 1536
 capacity_mem_par_l1_P42: mem[9,27] <= 1536
 capacity_mem_par_l1_P42: mem[9,12] <= 1536
 capacity_mem_par_l1_P42: mem[9,1] <= 1536
 capacity_mem_par_l1_P42: mem[9,22] <= 1536
 capacity_mem_par_l1_P42: mem[9,15] <= 1536
 capacity_mem_par_l1_P42: mem[9,28] <= 1536
 capacity_mem_par_l1_P42: mem[9,17] <= 1536
 capacity_mem_par_l1_P42: mem[9,32] <= 1536
 capacity_mem_par_l1_P42: mem[9,31] <= 1536
 capacity_mem_par_l1_P42: mem[9,35] <= 1536
 capacity_mem_par_l1_P42: mem[9,5] <= 1536
Bounds
 frac[0,7] <= 1
 frac[0,17] <= 1
 frac[0,24] <= 1
 frac[0,11] <= 1
 frac[0,18] <= 1
 frac[0,31] <= 1
 frac[0,28] <= 1
 frac[0,12] <= 1
 frac[0,34] <= 1
 frac[0,6] <= 1
 frac[1,34] <= 1
 frac[1,6] <= 1
 frac[1,11] <= 1
 frac[1,28] <= 1
 frac[1,31] <= 1
 frac[1,12] <= 1
 frac[1,26] <= 1
 frac[1,14] <= 1
 frac[1,7] <= 1
 frac[1,29] <= 1
 frac[1,24] <= 1
 frac[2,5] <= 1
 frac[2,6] <= 1
 frac[2,17] <= 1
 frac[2,28] <= 1
 frac[2,12] <= 1
 frac[2,2] <= 1
 frac[2,24] <= 1
 frac[2,29] <= 1
 frac[2,34] <= 1
 frac[2,13] <= 1
 frac[2,14] <= 1
 frac[2,4] <= 1
 frac[2,31] <= 1
 frac[2,21] <= 1
 frac[3,24] <= 1
 frac[3,12] <= 1
 frac[3,28] <= 1
 frac[3,34] <= 1
 frac[3,6] <= 1
 frac[3,5] <= 1
 frac[3,21] <= 1
 frac[3,31] <= 1
 frac[3,26] <= 1
 frac[3,4] <= 1
 frac[3,29] <= 1
 frac[4,19] <= 1
 frac[4,5] <= 1
 frac[4,21] <= 1
 frac[4,12] <= 1
 frac[4,31] <= 1
 frac[4,28] <= 1
 frac[4,2] <= 1
 frac[4,7] <= 1
 frac[4,34] <= 1
 frac[4,14] <= 1
 frac[5,18] <= 1
 frac[5,24] <= 1
 frac[5,14] <= 1
 frac[5,4] <= 1
 frac[5,17] <= 1
 frac[5,28] <= 1
 frac[6,9] <= 1
 frac[6,29] <= 1
 frac[6,17] <= 1
 frac[6,31] <= 1
 frac[6,21] <= 1
 frac[7,28] <= 1
 frac[7,5] <= 1
 frac[7,21] <= 1
 frac[7,2] <= 1
 frac[7,31] <= 1
 frac[7,14] <= 1
 mem_tot_CPU2 <= 32768
 normalized_mem_tot_CPU2 <= 1
 normalized_rows_tot_CPU2 <= 1
 mem_tot_CPU4 <= 32768
 normalized_mem_tot_CPU4 <= 1
 normalized_rows_tot_CPU4 <= 1
 mem_tot_CPU13 <= 32768
 normalized_mem_tot_CPU13 <= 1
 normalized_rows_tot_CPU13 <= 1
 mem_tot_CPU18 <= 32768
 normalized_mem_tot_CPU18 <= 1
 normalized_rows_tot_CPU18 <= 1
 mem_tot_CPU21 <= 32768
 normalized_mem_tot_CPU21 <= 1
 normalized_rows_tot_CPU21 <= 1
 mem_tot_CPU24 <= 32768
 normalized_mem_tot_CPU24 <= 1
 normalized_rows_tot_CPU24 <= 1
 mem_tot_CPU26 <= 32768
 normalized_mem_tot_CPU26 <= 1
 normalized_rows_tot_CPU26 <= 1
 mem_tot_CPU32 <= 32768
 normalized_mem_tot_CPU32 <= 1
 normalized_rows_tot_CPU32 <= 1
 mem_tot_tor_P43 <= 6144
 normalized_mem_tot_tor_P43 <= 1
 normalized_rows_tot_tor_P43 <= 1
 mem_tot_l1_P42 <= 6144
 normalized_mem_tot_l1_P42 <= 1
 normalized_rows_tot_l1_P42 <= 1
Binaries
 frac[8,5] frac[8,24] frac[8,15] frac[8,2] frac[8,21] frac[8,12] frac[8,31]
 frac[8,18] frac[8,35] frac[8,28] frac[8,32] frac[8,6] frac[8,25] frac[8,3]
 frac[8,22] frac[8,13] frac[8,0] frac[8,19] frac[8,36] frac[8,10]
 frac[8,29] frac[8,16] frac[8,33] frac[8,7] frac[8,26] frac[8,4] frac[8,23]
 frac[8,14] frac[8,1] frac[8,20] frac[8,30] frac[8,17] frac[8,34] frac[8,8]
 frac[8,27] frac[9,26] frac[9,0] frac[9,21] frac[9,36] frac[9,14] frac[9,3]
 frac[9,16] frac[9,30] frac[9,19] frac[9,34] frac[9,4] frac[9,25] frac[9,7]
 frac[9,20] frac[9,13] frac[9,2] frac[9,23] frac[9,8] frac[9,29] frac[9,18]
 frac[9,33] frac[9,24] frac[9,6] frac[9,27] frac[9,12] frac[9,1] frac[9,22]
 frac[9,15] frac[9,28] frac[9,17] frac[9,32] frac[9,31] frac[9,35]
 frac[9,5]
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_tor_P43 , mem_tot_l1_P42 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU2 , normalized_mem_tot_CPU4 ,
   normalized_mem_tot_CPU13 , normalized_mem_tot_CPU18 ,
   normalized_mem_tot_CPU21 , normalized_mem_tot_CPU24 ,
   normalized_mem_tot_CPU26 , normalized_mem_tot_CPU32 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_tor_P43 , rows_tot_l1_P42 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU2 ,
   normalized_rows_tot_CPU4 , normalized_rows_tot_CPU13 ,
   normalized_rows_tot_CPU18 , normalized_rows_tot_CPU21 ,
   normalized_rows_tot_CPU24 , normalized_rows_tot_CPU26 ,
   normalized_rows_tot_CPU32 )
End

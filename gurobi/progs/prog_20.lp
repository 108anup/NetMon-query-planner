\ Model UnivmonGreedy
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_mem_CPU_Cluster: Priority=20 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU1 + mem_tot_CPU2
  Cluster_CPU_mem_load: Priority=15 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
Subject To
 cov_0_1: frac[1,1] >= 0.8
 cov_1_6: frac[0,6] >= 1
 cov_1_9: frac[0,9] >= 1
 cov_2_7: frac[1,7] >= 0.8
 cov_3_16: frac[0,16] >= 0.9
 cov_4_16: frac[0,16] + frac[1,16] >= 0.7
 cov_5_7: frac[1,7] >= 0.8
 cov_6_11: frac[1,11] + frac[0,11] >= 0.7
 cov_7_18: frac[0,18] + frac[1,18] >= 0.7
 cov_8_11: frac[0,11] >= 1
 cov_9_13: frac[1,13] + frac[0,13] >= 1
 cov_9_17: frac[1,17] + frac[0,17] >= 1
 cov_10_2: frac[1,2] >= 1
 cov_11_4: frac[0,4] >= 0.9
 cov_11_14: frac[0,14] >= 0.9
 cov_12_5: frac[0,5] >= 0.9
 cov_13_8: frac[0,8] + frac[1,8] >= 1
 cov_14_5: frac[1,5] + frac[0,5] >= 1
 cov_15_5: frac[1,5] >= 0.8
 cov_16_6: frac[1,6] >= 1
 cov_16_9: frac[1,9] >= 1
 cov_17_3: frac[0,3] >= 1
 cov_18_0: frac[1,0] >= 0.9
 cov_19_7: frac[0,7] >= 1
 cov_20_8: frac[0,8] >= 0.9
 cov_21_12: frac[0,12] + frac[1,12] >= 0.7
 cov_22_13: frac[0,13] >= 0.9
 cov_22_17: frac[0,17] >= 0.9
 cov_23_12: frac[1,12] >= 0.8
 cov_24_3: frac[0,3] >= 0.8
 cov_25_6: frac[1,6] >= 0.7
 cov_25_9: frac[1,9] >= 0.7
 cov_26_10: frac[0,10] >= 1
 cov_27_2: frac[1,2] + frac[0,2] >= 1
 cov_28_11: frac[0,11] >= 1
 cov_29_0: frac[0,0] + frac[1,0] >= 1
 cov_30_10: frac[0,10] + frac[1,10] >= 1
 cov_31_12: frac[1,12] >= 0.8
 cov_32_8: frac[0,8] >= 0.7
 cov_33_1: frac[0,1] >= 0.9
 cov_34_16: frac[0,16] >= 0.7
 cov_35_4: frac[0,4] + frac[1,4] >= 0.9
 cov_35_14: frac[0,14] + frac[1,14] >= 0.9
 cov_36_10: frac[0,10] >= 0.9
 cov_37_1: frac[1,1] + frac[0,1] >= 0.9
 cov_38_10: frac[0,10] >= 0.7
 cov_39_5: frac[0,5] >= 0.8
 cov_40_7: frac[1,7] >= 0.7
 cov_41_15: frac[1,15] + frac[0,15] >= 0.8
 cov_42_8: frac[0,8] >= 0.8
 cov_43_16: frac[0,16] >= 1
 cov_44_7: frac[0,7] >= 0.8
 cov_45_0: frac[0,0] >= 0.9
 accuracy_0_14: - 1.69921875 frac[0,14] + mem[0,14] = 0
 accuracy_1_17: - 1.69921875 frac[1,17] + mem[1,17] = 0
 accuracy_0_7: - 1.69921875 frac[0,7] + mem[0,7] = 0
 accuracy_1_15: - 1.69921875 frac[1,15] + mem[1,15] = 0
 accuracy_0_16: - 1.69921875 frac[0,16] + mem[0,16] = 0
 accuracy_1_6: - 1.69921875 frac[1,6] + mem[1,6] = 0
 accuracy_0_10: - 1.69921875 frac[0,10] + mem[0,10] = 0
 accuracy_0_3: - 1.69921875 frac[0,3] + mem[0,3] = 0
 accuracy_1_11: - 1.69921875 frac[1,11] + mem[1,11] = 0
 accuracy_1_2: - 1.69921875 frac[1,2] + mem[1,2] = 0
 accuracy_0_15: - 1.69921875 frac[0,15] + mem[0,15] = 0
 accuracy_1_16: - 1.69921875 frac[1,16] + mem[1,16] = 0
 accuracy_1_14: - 1.69921875 frac[1,14] + mem[1,14] = 0
 accuracy_0_17: - 1.69921875 frac[0,17] + mem[0,17] = 0
 accuracy_1_5: - 1.69921875 frac[1,5] + mem[1,5] = 0
 accuracy_0_11: - 1.69921875 frac[0,11] + mem[0,11] = 0
 accuracy_0_4: - 1.69921875 frac[0,4] + mem[0,4] = 0
 accuracy_1_10: - 1.69921875 frac[1,10] + mem[1,10] = 0
 accuracy_1_1: - 1.69921875 frac[1,1] + mem[1,1] = 0
 accuracy_0_0: - 1.69921875 frac[0,0] + mem[0,0] = 0
 accuracy_1_13: - 1.69921875 frac[1,13] + mem[1,13] = 0
 accuracy_0_18: - 1.69921875 frac[0,18] + mem[0,18] = 0
 accuracy_1_4: - 1.69921875 frac[1,4] + mem[1,4] = 0
 accuracy_0_12: - 1.69921875 frac[0,12] + mem[0,12] = 0
 accuracy_0_5: - 1.69921875 frac[0,5] + mem[0,5] = 0
 accuracy_1_9: - 1.69921875 frac[1,9] + mem[1,9] = 0
 accuracy_1_0: - 1.69921875 frac[1,0] + mem[1,0] = 0
 accuracy_0_8: - 1.69921875 frac[0,8] + mem[0,8] = 0
 accuracy_0_1: - 1.69921875 frac[0,1] + mem[0,1] = 0
 accuracy_1_12: - 1.69921875 frac[1,12] + mem[1,12] = 0
 accuracy_0_13: - 1.69921875 frac[0,13] + mem[0,13] = 0
 accuracy_1_18: - 1.69921875 frac[1,18] + mem[1,18] = 0
 accuracy_0_6: - 1.69921875 frac[0,6] + mem[0,6] = 0
 accuracy_1_8: - 1.69921875 frac[1,8] + mem[1,8] = 0
 accuracy_1_7: - 1.69921875 frac[1,7] + mem[1,7] = 0
 accuracy_0_9: - 1.69921875 frac[0,9] + mem[0,9] = 0
 accuracy_0_2: - 1.69921875 frac[0,2] + mem[0,2] = 0
 mem_tot_CPU1: - mem[0,14] - mem[0,7] - mem[0,16] - mem[0,10] - mem[0,3]
   - mem[0,15] - mem[0,17] - mem[0,11] - mem[0,4] - mem[0,0] - mem[0,18]
   - mem[0,12] - mem[0,5] - mem[0,8] - mem[0,1] - mem[0,13] - mem[0,6]
   - mem[0,9] - mem[0,2] + mem_tot_CPU1 = 0
 R91: - 3.05176e-05 mem_tot_CPU1 + normalized_mem_tot_CPU1 = 0
 rows_tot_CPU1: - frac[0,14] - frac[0,7] - frac[0,16] - frac[0,10]
   - frac[0,3] - frac[0,15] - frac[0,17] - frac[0,11] - frac[0,4]
   - frac[0,0] - frac[0,18] - frac[0,12] - frac[0,5] - frac[0,8]
   - frac[0,1] - frac[0,13] - frac[0,6] - frac[0,9] - frac[0,2]
   + rows_tot_CPU1 = 0
 R93: - 0.0833333333333333 rows_tot_CPU1 + normalized_rows_tot_CPU1 = 0
 mem_tot_CPU2: - mem[1,17] - mem[1,15] - mem[1,6] - mem[1,11] - mem[1,2]
   - mem[1,16] - mem[1,14] - mem[1,5] - mem[1,10] - mem[1,1] - mem[1,13]
   - mem[1,4] - mem[1,9] - mem[1,0] - mem[1,12] - mem[1,18] - mem[1,8]
   - mem[1,7] + mem_tot_CPU2 = 0
 R95: - 3.05176e-05 mem_tot_CPU2 + normalized_mem_tot_CPU2 = 0
 rows_tot_CPU2: - frac[1,17] - frac[1,15] - frac[1,6] - frac[1,11]
   - frac[1,2] - frac[1,16] - frac[1,14] - frac[1,5] - frac[1,10]
   - frac[1,1] - frac[1,13] - frac[1,4] - frac[1,9] - frac[1,0]
   - frac[1,12] - frac[1,18] - frac[1,8] - frac[1,7] + rows_tot_CPU2 = 0
 R97: - 0.0833333333333333 rows_tot_CPU2 + normalized_rows_tot_CPU2 = 0
 row_capacity_CPU1: rows_tot_CPU1 <= 12
 row_capacity_CPU2: rows_tot_CPU2 <= 12
Bounds
 mem_tot_CPU1 <= 32768
 normalized_mem_tot_CPU1 <= 1
 normalized_rows_tot_CPU1 <= 1
 mem_tot_CPU2 <= 32768
 normalized_mem_tot_CPU2 <= 1
 normalized_rows_tot_CPU2 <= 1
Binaries
 frac[0,14] frac[1,17] frac[0,7] frac[1,15] frac[0,16] frac[1,6] frac[0,10]
 frac[0,3] frac[1,11] frac[1,2] frac[0,15] frac[1,16] frac[1,14] frac[0,17]
 frac[1,5] frac[0,11] frac[0,4] frac[1,10] frac[1,1] frac[0,0] frac[1,13]
 frac[0,18] frac[1,4] frac[0,12] frac[0,5] frac[1,9] frac[1,0] frac[0,8]
 frac[0,1] frac[1,12] frac[0,13] frac[1,18] frac[0,6] frac[1,8] frac[1,7]
 frac[0,9] frac[0,2]
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU1 , normalized_mem_tot_CPU2 )
End

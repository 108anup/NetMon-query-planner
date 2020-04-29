\ Model UnivmonGreedy
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_mem_CPU_Cluster: Priority=20 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU_1
  Cluster_CPU_mem_load: Priority=15 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_mem_P4: Priority=10 Weight=1 AbsTol=0 RelTol=0
   mem_tot_P4_1
  P4_mem_load: Priority=5 Weight=1 AbsTol=0 RelTol=0
   max_mem_P4
Subject To
 cov_0_0: frac[0,0] + frac[1,0] >= 1
 cov_0_1: frac[0,1] + frac[1,1] >= 1
 cov_0_2: frac[1,2] + frac[0,2] >= 1
 cov_0_3: frac[1,3] + frac[0,3] >= 1
 cov_0_4: frac[1,4] + frac[0,4] >= 1
 cov_0_5: frac[1,5] + frac[0,5] >= 1
 cov_0_6: frac[0,6] + frac[1,6] >= 1
 cov_0_7: frac[0,7] + frac[1,7] >= 1
 cov_0_8: frac[0,8] + frac[1,8] >= 1
 accuracy_0_1: - 1698.9296875 frac[0,1] + mem[0,1] = 0
 cols_helper_1_2: cols_helper_1_2[0] + cols_helper_1_2[1]
   + cols_helper_1_2[2] + cols_helper_1_2[3] + cols_helper_1_2[4]
   + cols_helper_1_2[5] + cols_helper_1_2[6] + cols_helper_1_2[7]
   + cols_helper_1_2[8] + cols_helper_1_2[9] + cols_helper_1_2[10]
   + cols_helper_1_2[11] + cols_helper_1_2[12] + cols_helper_1_2[13]
   + cols_helper_1_2[14] + cols_helper_1_2[15] + cols_helper_1_2[16] <= 1
 cols_1_2: cols_1_2 - 2 cols_helper_1_2[0] - 4 cols_helper_1_2[1]
   - 8 cols_helper_1_2[2] - 16 cols_helper_1_2[3] - 32 cols_helper_1_2[4]
   - 64 cols_helper_1_2[5] - 128 cols_helper_1_2[6]
   - 256 cols_helper_1_2[7] - 512 cols_helper_1_2[8]
   - 1024 cols_helper_1_2[9] - 2048 cols_helper_1_2[10]
   - 4096 cols_helper_1_2[11] - 8192 cols_helper_1_2[12]
   - 16384 cols_helper_1_2[13] - 32768 cols_helper_1_2[14]
   - 65536 cols_helper_1_2[15] - 131072 cols_helper_1_2[16] = 0
 cols_acc_1_2: - 434926 frac[1,2] + cols_1_2 >= 0
 accuracy_1_2: mem[1,2] - 0.00390625 cols_1_2 = 0
 accuracy_0_0: - 1698.9296875 frac[0,0] + mem[0,0] = 0
 cols_helper_1_3: cols_helper_1_3[0] + cols_helper_1_3[1]
   + cols_helper_1_3[2] + cols_helper_1_3[3] + cols_helper_1_3[4]
   + cols_helper_1_3[5] + cols_helper_1_3[6] + cols_helper_1_3[7]
   + cols_helper_1_3[8] + cols_helper_1_3[9] + cols_helper_1_3[10]
   + cols_helper_1_3[11] + cols_helper_1_3[12] + cols_helper_1_3[13]
   + cols_helper_1_3[14] + cols_helper_1_3[15] + cols_helper_1_3[16] <= 1
 cols_1_3: cols_1_3 - 2 cols_helper_1_3[0] - 4 cols_helper_1_3[1]
   - 8 cols_helper_1_3[2] - 16 cols_helper_1_3[3] - 32 cols_helper_1_3[4]
   - 64 cols_helper_1_3[5] - 128 cols_helper_1_3[6]
   - 256 cols_helper_1_3[7] - 512 cols_helper_1_3[8]
   - 1024 cols_helper_1_3[9] - 2048 cols_helper_1_3[10]
   - 4096 cols_helper_1_3[11] - 8192 cols_helper_1_3[12]
   - 16384 cols_helper_1_3[13] - 32768 cols_helper_1_3[14]
   - 65536 cols_helper_1_3[15] - 131072 cols_helper_1_3[16] = 0
 cols_acc_1_3: - 4350 frac[1,3] + cols_1_3 >= 0
 accuracy_1_3: mem[1,3] - 0.00390625 cols_1_3 = 0
 cols_helper_1_4: cols_helper_1_4[0] + cols_helper_1_4[1]
   + cols_helper_1_4[2] + cols_helper_1_4[3] + cols_helper_1_4[4]
   + cols_helper_1_4[5] + cols_helper_1_4[6] + cols_helper_1_4[7]
   + cols_helper_1_4[8] + cols_helper_1_4[9] + cols_helper_1_4[10]
   + cols_helper_1_4[11] + cols_helper_1_4[12] + cols_helper_1_4[13]
   + cols_helper_1_4[14] + cols_helper_1_4[15] + cols_helper_1_4[16] <= 1
 cols_1_4: cols_1_4 - 2 cols_helper_1_4[0] - 4 cols_helper_1_4[1]
   - 8 cols_helper_1_4[2] - 16 cols_helper_1_4[3] - 32 cols_helper_1_4[4]
   - 64 cols_helper_1_4[5] - 128 cols_helper_1_4[6]
   - 256 cols_helper_1_4[7] - 512 cols_helper_1_4[8]
   - 1024 cols_helper_1_4[9] - 2048 cols_helper_1_4[10]
   - 4096 cols_helper_1_4[11] - 8192 cols_helper_1_4[12]
   - 16384 cols_helper_1_4[13] - 32768 cols_helper_1_4[14]
   - 65536 cols_helper_1_4[15] - 131072 cols_helper_1_4[16] = 0
 cols_acc_1_4: - 4350 frac[1,4] + cols_1_4 >= 0
 accuracy_1_4: mem[1,4] - 0.00390625 cols_1_4 = 0
 accuracy_0_7: - 16.9921875 frac[0,7] + mem[0,7] = 0
 accuracy_0_8: - 16.9921875 frac[0,8] + mem[0,8] = 0
 cols_helper_1_5: cols_helper_1_5[0] + cols_helper_1_5[1]
   + cols_helper_1_5[2] + cols_helper_1_5[3] + cols_helper_1_5[4]
   + cols_helper_1_5[5] + cols_helper_1_5[6] + cols_helper_1_5[7]
   + cols_helper_1_5[8] + cols_helper_1_5[9] + cols_helper_1_5[10]
   + cols_helper_1_5[11] + cols_helper_1_5[12] + cols_helper_1_5[13]
   + cols_helper_1_5[14] + cols_helper_1_5[15] + cols_helper_1_5[16] <= 1
 cols_1_5: cols_1_5 - 2 cols_helper_1_5[0] - 4 cols_helper_1_5[1]
   - 8 cols_helper_1_5[2] - 16 cols_helper_1_5[3] - 32 cols_helper_1_5[4]
   - 64 cols_helper_1_5[5] - 128 cols_helper_1_5[6]
   - 256 cols_helper_1_5[7] - 512 cols_helper_1_5[8]
   - 1024 cols_helper_1_5[9] - 2048 cols_helper_1_5[10]
   - 4096 cols_helper_1_5[11] - 8192 cols_helper_1_5[12]
   - 16384 cols_helper_1_5[13] - 32768 cols_helper_1_5[14]
   - 65536 cols_helper_1_5[15] - 131072 cols_helper_1_5[16] = 0
 cols_acc_1_5: - 4350 frac[1,5] + cols_1_5 >= 0
 accuracy_1_5: mem[1,5] - 0.00390625 cols_1_5 = 0
 accuracy_0_6: - 16.9921875 frac[0,6] + mem[0,6] = 0
 cols_helper_1_1: cols_helper_1_1[0] + cols_helper_1_1[1]
   + cols_helper_1_1[2] + cols_helper_1_1[3] + cols_helper_1_1[4]
   + cols_helper_1_1[5] + cols_helper_1_1[6] + cols_helper_1_1[7]
   + cols_helper_1_1[8] + cols_helper_1_1[9] + cols_helper_1_1[10]
   + cols_helper_1_1[11] + cols_helper_1_1[12] + cols_helper_1_1[13]
   + cols_helper_1_1[14] + cols_helper_1_1[15] + cols_helper_1_1[16] <= 1
 cols_1_1: cols_1_1 - 2 cols_helper_1_1[0] - 4 cols_helper_1_1[1]
   - 8 cols_helper_1_1[2] - 16 cols_helper_1_1[3] - 32 cols_helper_1_1[4]
   - 64 cols_helper_1_1[5] - 128 cols_helper_1_1[6]
   - 256 cols_helper_1_1[7] - 512 cols_helper_1_1[8]
   - 1024 cols_helper_1_1[9] - 2048 cols_helper_1_1[10]
   - 4096 cols_helper_1_1[11] - 8192 cols_helper_1_1[12]
   - 16384 cols_helper_1_1[13] - 32768 cols_helper_1_1[14]
   - 65536 cols_helper_1_1[15] - 131072 cols_helper_1_1[16] = 0
 cols_acc_1_1: - 434926 frac[1,1] + cols_1_1 >= 0
 accuracy_1_1: mem[1,1] - 0.00390625 cols_1_1 = 0
 accuracy_0_5: - 16.9921875 frac[0,5] + mem[0,5] = 0
 cols_helper_1_6: cols_helper_1_6[0] + cols_helper_1_6[1]
   + cols_helper_1_6[2] + cols_helper_1_6[3] + cols_helper_1_6[4]
   + cols_helper_1_6[5] + cols_helper_1_6[6] + cols_helper_1_6[7]
   + cols_helper_1_6[8] + cols_helper_1_6[9] + cols_helper_1_6[10]
   + cols_helper_1_6[11] + cols_helper_1_6[12] + cols_helper_1_6[13]
   + cols_helper_1_6[14] + cols_helper_1_6[15] + cols_helper_1_6[16] <= 1
 cols_1_6: cols_1_6 - 2 cols_helper_1_6[0] - 4 cols_helper_1_6[1]
   - 8 cols_helper_1_6[2] - 16 cols_helper_1_6[3] - 32 cols_helper_1_6[4]
   - 64 cols_helper_1_6[5] - 128 cols_helper_1_6[6]
   - 256 cols_helper_1_6[7] - 512 cols_helper_1_6[8]
   - 1024 cols_helper_1_6[9] - 2048 cols_helper_1_6[10]
   - 4096 cols_helper_1_6[11] - 8192 cols_helper_1_6[12]
   - 16384 cols_helper_1_6[13] - 32768 cols_helper_1_6[14]
   - 65536 cols_helper_1_6[15] - 131072 cols_helper_1_6[16] = 0
 cols_acc_1_6: - 4350 frac[1,6] + cols_1_6 >= 0
 accuracy_1_6: mem[1,6] - 0.00390625 cols_1_6 = 0
 cols_helper_1_8: cols_helper_1_8[0] + cols_helper_1_8[1]
   + cols_helper_1_8[2] + cols_helper_1_8[3] + cols_helper_1_8[4]
   + cols_helper_1_8[5] + cols_helper_1_8[6] + cols_helper_1_8[7]
   + cols_helper_1_8[8] + cols_helper_1_8[9] + cols_helper_1_8[10]
   + cols_helper_1_8[11] + cols_helper_1_8[12] + cols_helper_1_8[13]
   + cols_helper_1_8[14] + cols_helper_1_8[15] + cols_helper_1_8[16] <= 1
 cols_1_8: cols_1_8 - 2 cols_helper_1_8[0] - 4 cols_helper_1_8[1]
   - 8 cols_helper_1_8[2] - 16 cols_helper_1_8[3] - 32 cols_helper_1_8[4]
   - 64 cols_helper_1_8[5] - 128 cols_helper_1_8[6]
   - 256 cols_helper_1_8[7] - 512 cols_helper_1_8[8]
   - 1024 cols_helper_1_8[9] - 2048 cols_helper_1_8[10]
   - 4096 cols_helper_1_8[11] - 8192 cols_helper_1_8[12]
   - 16384 cols_helper_1_8[13] - 32768 cols_helper_1_8[14]
   - 65536 cols_helper_1_8[15] - 131072 cols_helper_1_8[16] = 0
 cols_acc_1_8: - 4350 frac[1,8] + cols_1_8 >= 0
 accuracy_1_8: mem[1,8] - 0.00390625 cols_1_8 = 0
 accuracy_0_4: - 16.9921875 frac[0,4] + mem[0,4] = 0
 cols_helper_1_0: cols_helper_1_0[0] + cols_helper_1_0[1]
   + cols_helper_1_0[2] + cols_helper_1_0[3] + cols_helper_1_0[4]
   + cols_helper_1_0[5] + cols_helper_1_0[6] + cols_helper_1_0[7]
   + cols_helper_1_0[8] + cols_helper_1_0[9] + cols_helper_1_0[10]
   + cols_helper_1_0[11] + cols_helper_1_0[12] + cols_helper_1_0[13]
   + cols_helper_1_0[14] + cols_helper_1_0[15] + cols_helper_1_0[16] <= 1
 cols_1_0: cols_1_0 - 2 cols_helper_1_0[0] - 4 cols_helper_1_0[1]
   - 8 cols_helper_1_0[2] - 16 cols_helper_1_0[3] - 32 cols_helper_1_0[4]
   - 64 cols_helper_1_0[5] - 128 cols_helper_1_0[6]
   - 256 cols_helper_1_0[7] - 512 cols_helper_1_0[8]
   - 1024 cols_helper_1_0[9] - 2048 cols_helper_1_0[10]
   - 4096 cols_helper_1_0[11] - 8192 cols_helper_1_0[12]
   - 16384 cols_helper_1_0[13] - 32768 cols_helper_1_0[14]
   - 65536 cols_helper_1_0[15] - 131072 cols_helper_1_0[16] = 0
 cols_acc_1_0: - 434926 frac[1,0] + cols_1_0 >= 0
 accuracy_1_0: mem[1,0] - 0.00390625 cols_1_0 = 0
 cols_helper_1_7: cols_helper_1_7[0] + cols_helper_1_7[1]
   + cols_helper_1_7[2] + cols_helper_1_7[3] + cols_helper_1_7[4]
   + cols_helper_1_7[5] + cols_helper_1_7[6] + cols_helper_1_7[7]
   + cols_helper_1_7[8] + cols_helper_1_7[9] + cols_helper_1_7[10]
   + cols_helper_1_7[11] + cols_helper_1_7[12] + cols_helper_1_7[13]
   + cols_helper_1_7[14] + cols_helper_1_7[15] + cols_helper_1_7[16] <= 1
 cols_1_7: cols_1_7 - 2 cols_helper_1_7[0] - 4 cols_helper_1_7[1]
   - 8 cols_helper_1_7[2] - 16 cols_helper_1_7[3] - 32 cols_helper_1_7[4]
   - 64 cols_helper_1_7[5] - 128 cols_helper_1_7[6]
   - 256 cols_helper_1_7[7] - 512 cols_helper_1_7[8]
   - 1024 cols_helper_1_7[9] - 2048 cols_helper_1_7[10]
   - 4096 cols_helper_1_7[11] - 8192 cols_helper_1_7[12]
   - 16384 cols_helper_1_7[13] - 32768 cols_helper_1_7[14]
   - 65536 cols_helper_1_7[15] - 131072 cols_helper_1_7[16] = 0
 cols_acc_1_7: - 4350 frac[1,7] + cols_1_7 >= 0
 accuracy_1_7: mem[1,7] - 0.00390625 cols_1_7 = 0
 accuracy_0_2: - 1698.9296875 frac[0,2] + mem[0,2] = 0
 accuracy_0_3: - 16.9921875 frac[0,3] + mem[0,3] = 0
 mem_tot_CPU_1: - mem[0,1] - mem[0,0] - mem[0,7] - mem[0,8] - mem[0,6]
   - mem[0,5] - mem[0,4] - mem[0,2] - mem[0,3] + mem_tot_CPU_1 = 0
 R55: - 3.05176e-05 mem_tot_CPU_1 + normalized_mem_tot_CPU_1 = 0
 rows_tot_CPU_1: - frac[0,1] - frac[0,0] - frac[0,7] - frac[0,8]
   - frac[0,6] - frac[0,5] - frac[0,4] - frac[0,2] - frac[0,3]
   + rows_tot_CPU_1 = 0
 R57: - 0.0833333333333333 rows_tot_CPU_1 + normalized_rows_tot_CPU_1 = 0
 mem_tot_P4_1: - mem[1,2] - mem[1,3] - mem[1,4] - mem[1,5] - mem[1,1]
   - mem[1,6] - mem[1,8] - mem[1,0] - mem[1,7] + mem_tot_P4_1 = 0
 R59: - 0.00173611111111111 mem_tot_P4_1 + normalized_mem_tot_P4_1 = 0
 rows_tot_P4_1: - frac[1,2] - frac[1,3] - frac[1,4] - frac[1,5] - frac[1,1]
   - frac[1,6] - frac[1,8] - frac[1,0] - frac[1,7] + rows_tot_P4_1 = 0
 R61: - 0.0208333333333333 rows_tot_P4_1 + normalized_rows_tot_P4_1 = 0
 row_capacity_CPU_1: rows_tot_CPU_1 <= 12
 row_capacity_P4_1: rows_tot_P4_1 <= 48
 capacity_mem_par_P4_1: mem[1,0] <= 48
 capacity_mem_par_P4_1: mem[1,1] <= 48
 capacity_mem_par_P4_1: mem[1,2] <= 48
 capacity_mem_par_P4_1: mem[1,3] <= 48
 capacity_mem_par_P4_1: mem[1,4] <= 48
 capacity_mem_par_P4_1: mem[1,5] <= 48
 capacity_mem_par_P4_1: mem[1,6] <= 48
 capacity_mem_par_P4_1: mem[1,7] <= 48
 capacity_mem_par_P4_1: mem[1,8] <= 48
Bounds
 frac[0,1] <= 1
 frac[1,2] <= 1
 frac[0,0] <= 1
 frac[1,3] <= 1
 frac[1,4] <= 1
 frac[0,7] <= 1
 frac[0,8] <= 1
 frac[1,5] <= 1
 frac[0,6] <= 1
 frac[1,1] <= 1
 frac[0,5] <= 1
 frac[1,6] <= 1
 frac[1,8] <= 1
 frac[0,4] <= 1
 frac[1,0] <= 1
 frac[1,7] <= 1
 frac[0,2] <= 1
 frac[0,3] <= 1
 cols_1_2 <= 131072
 cols_1_3 <= 131072
 cols_1_4 <= 131072
 cols_1_5 <= 131072
 cols_1_1 <= 131072
 cols_1_6 <= 131072
 cols_1_8 <= 131072
 cols_1_0 <= 131072
 cols_1_7 <= 131072
 mem_tot_CPU_1 <= 32768
 normalized_mem_tot_CPU_1 <= 1
 normalized_rows_tot_CPU_1 <= 1
 mem_tot_P4_1 <= 576
 normalized_mem_tot_P4_1 <= 1
 normalized_rows_tot_P4_1 <= 1
Binaries
 cols_helper_1_2[0] cols_helper_1_2[1] cols_helper_1_2[2]
 cols_helper_1_2[3] cols_helper_1_2[4] cols_helper_1_2[5]
 cols_helper_1_2[6] cols_helper_1_2[7] cols_helper_1_2[8]
 cols_helper_1_2[9] cols_helper_1_2[10] cols_helper_1_2[11]
 cols_helper_1_2[12] cols_helper_1_2[13] cols_helper_1_2[14]
 cols_helper_1_2[15] cols_helper_1_2[16] cols_helper_1_3[0]
 cols_helper_1_3[1] cols_helper_1_3[2] cols_helper_1_3[3]
 cols_helper_1_3[4] cols_helper_1_3[5] cols_helper_1_3[6]
 cols_helper_1_3[7] cols_helper_1_3[8] cols_helper_1_3[9]
 cols_helper_1_3[10] cols_helper_1_3[11] cols_helper_1_3[12]
 cols_helper_1_3[13] cols_helper_1_3[14] cols_helper_1_3[15]
 cols_helper_1_3[16] cols_helper_1_4[0] cols_helper_1_4[1]
 cols_helper_1_4[2] cols_helper_1_4[3] cols_helper_1_4[4]
 cols_helper_1_4[5] cols_helper_1_4[6] cols_helper_1_4[7]
 cols_helper_1_4[8] cols_helper_1_4[9] cols_helper_1_4[10]
 cols_helper_1_4[11] cols_helper_1_4[12] cols_helper_1_4[13]
 cols_helper_1_4[14] cols_helper_1_4[15] cols_helper_1_4[16]
 cols_helper_1_5[0] cols_helper_1_5[1] cols_helper_1_5[2]
 cols_helper_1_5[3] cols_helper_1_5[4] cols_helper_1_5[5]
 cols_helper_1_5[6] cols_helper_1_5[7] cols_helper_1_5[8]
 cols_helper_1_5[9] cols_helper_1_5[10] cols_helper_1_5[11]
 cols_helper_1_5[12] cols_helper_1_5[13] cols_helper_1_5[14]
 cols_helper_1_5[15] cols_helper_1_5[16] cols_helper_1_1[0]
 cols_helper_1_1[1] cols_helper_1_1[2] cols_helper_1_1[3]
 cols_helper_1_1[4] cols_helper_1_1[5] cols_helper_1_1[6]
 cols_helper_1_1[7] cols_helper_1_1[8] cols_helper_1_1[9]
 cols_helper_1_1[10] cols_helper_1_1[11] cols_helper_1_1[12]
 cols_helper_1_1[13] cols_helper_1_1[14] cols_helper_1_1[15]
 cols_helper_1_1[16] cols_helper_1_6[0] cols_helper_1_6[1]
 cols_helper_1_6[2] cols_helper_1_6[3] cols_helper_1_6[4]
 cols_helper_1_6[5] cols_helper_1_6[6] cols_helper_1_6[7]
 cols_helper_1_6[8] cols_helper_1_6[9] cols_helper_1_6[10]
 cols_helper_1_6[11] cols_helper_1_6[12] cols_helper_1_6[13]
 cols_helper_1_6[14] cols_helper_1_6[15] cols_helper_1_6[16]
 cols_helper_1_8[0] cols_helper_1_8[1] cols_helper_1_8[2]
 cols_helper_1_8[3] cols_helper_1_8[4] cols_helper_1_8[5]
 cols_helper_1_8[6] cols_helper_1_8[7] cols_helper_1_8[8]
 cols_helper_1_8[9] cols_helper_1_8[10] cols_helper_1_8[11]
 cols_helper_1_8[12] cols_helper_1_8[13] cols_helper_1_8[14]
 cols_helper_1_8[15] cols_helper_1_8[16] cols_helper_1_0[0]
 cols_helper_1_0[1] cols_helper_1_0[2] cols_helper_1_0[3]
 cols_helper_1_0[4] cols_helper_1_0[5] cols_helper_1_0[6]
 cols_helper_1_0[7] cols_helper_1_0[8] cols_helper_1_0[9]
 cols_helper_1_0[10] cols_helper_1_0[11] cols_helper_1_0[12]
 cols_helper_1_0[13] cols_helper_1_0[14] cols_helper_1_0[15]
 cols_helper_1_0[16] cols_helper_1_7[0] cols_helper_1_7[1]
 cols_helper_1_7[2] cols_helper_1_7[3] cols_helper_1_7[4]
 cols_helper_1_7[5] cols_helper_1_7[6] cols_helper_1_7[7]
 cols_helper_1_7[8] cols_helper_1_7[9] cols_helper_1_7[10]
 cols_helper_1_7[11] cols_helper_1_7[12] cols_helper_1_7[13]
 cols_helper_1_7[14] cols_helper_1_7[15] cols_helper_1_7[16]
Generals
 cols_1_2 cols_1_3 cols_1_4 cols_1_5 cols_1_1 cols_1_6 cols_1_8 cols_1_0
 cols_1_7
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_P4_1 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU_1 )
End

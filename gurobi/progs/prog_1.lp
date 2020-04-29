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
 accuracy_0_1: - 50.9765625 frac[0,1] + mem[0,1] = 0
 cols_helper_1_2: cols_helper_1_2[0] + cols_helper_1_2[1]
   + cols_helper_1_2[2] + cols_helper_1_2[3] + cols_helper_1_2[4]
   + cols_helper_1_2[5] + cols_helper_1_2[6] + cols_helper_1_2[7]
   + cols_helper_1_2[8] + cols_helper_1_2[9] + cols_helper_1_2[10]
   + cols_helper_1_2[11] + cols_helper_1_2[12] <= 1
 cols_1_2: cols_1_2 - 2 cols_helper_1_2[0] - 4 cols_helper_1_2[1]
   - 8 cols_helper_1_2[2] - 16 cols_helper_1_2[3] - 32 cols_helper_1_2[4]
   - 64 cols_helper_1_2[5] - 128 cols_helper_1_2[6]
   - 256 cols_helper_1_2[7] - 512 cols_helper_1_2[8]
   - 1024 cols_helper_1_2[9] - 2048 cols_helper_1_2[10]
   - 4096 cols_helper_1_2[11] - 8192 cols_helper_1_2[12] = 0
 cols_acc_1_2: - 4350 frac[1,2] + cols_1_2 >= 0
 accuracy_1_2: mem[1,2] - 0.01171875 cols_1_2 = 0
 accuracy_0_0: - 5096.7890625 frac[0,0] + mem[0,0] = 0
 cols_helper_1_1: cols_helper_1_1[0] + cols_helper_1_1[1]
   + cols_helper_1_1[2] + cols_helper_1_1[3] + cols_helper_1_1[4]
   + cols_helper_1_1[5] + cols_helper_1_1[6] + cols_helper_1_1[7]
   + cols_helper_1_1[8] + cols_helper_1_1[9] + cols_helper_1_1[10]
   + cols_helper_1_1[11] + cols_helper_1_1[12] <= 1
 cols_1_1: cols_1_1 - 2 cols_helper_1_1[0] - 4 cols_helper_1_1[1]
   - 8 cols_helper_1_1[2] - 16 cols_helper_1_1[3] - 32 cols_helper_1_1[4]
   - 64 cols_helper_1_1[5] - 128 cols_helper_1_1[6]
   - 256 cols_helper_1_1[7] - 512 cols_helper_1_1[8]
   - 1024 cols_helper_1_1[9] - 2048 cols_helper_1_1[10]
   - 4096 cols_helper_1_1[11] - 8192 cols_helper_1_1[12] = 0
 cols_acc_1_1: - 4350 frac[1,1] + cols_1_1 >= 0
 accuracy_1_1: mem[1,1] - 0.01171875 cols_1_1 = 0
 cols_helper_1_0: cols_helper_1_0[0] + cols_helper_1_0[1]
   + cols_helper_1_0[2] + cols_helper_1_0[3] + cols_helper_1_0[4]
   + cols_helper_1_0[5] + cols_helper_1_0[6] + cols_helper_1_0[7]
   + cols_helper_1_0[8] + cols_helper_1_0[9] + cols_helper_1_0[10]
   + cols_helper_1_0[11] + cols_helper_1_0[12] <= 1
 cols_1_0: cols_1_0 - 2 cols_helper_1_0[0] - 4 cols_helper_1_0[1]
   - 8 cols_helper_1_0[2] - 16 cols_helper_1_0[3] - 32 cols_helper_1_0[4]
   - 64 cols_helper_1_0[5] - 128 cols_helper_1_0[6]
   - 256 cols_helper_1_0[7] - 512 cols_helper_1_0[8]
   - 1024 cols_helper_1_0[9] - 2048 cols_helper_1_0[10]
   - 4096 cols_helper_1_0[11] - 8192 cols_helper_1_0[12] = 0
 cols_acc_1_0: - 434926 frac[1,0] + cols_1_0 >= 0
 accuracy_1_0: mem[1,0] - 0.01171875 cols_1_0 = 0
 accuracy_0_2: - 50.9765625 frac[0,2] + mem[0,2] = 0
 mem_tot_CPU_1: - mem[0,1] - mem[0,0] - mem[0,2] + mem_tot_CPU_1 = 0
 R19: - 3.05176e-05 mem_tot_CPU_1 + normalized_mem_tot_CPU_1 = 0
 rows_tot_CPU_1: - 3 frac[0,1] - 3 frac[0,0] - 3 frac[0,2] + rows_tot_CPU_1
   = 0
 R21: - 0.0833333333333333 rows_tot_CPU_1 + normalized_rows_tot_CPU_1 = 0
 mem_tot_P4_1: - mem[1,2] - mem[1,1] - mem[1,0] + mem_tot_P4_1 = 0
 R23: - 0.00173611111111111 mem_tot_P4_1 + normalized_mem_tot_P4_1 = 0
 rows_tot_P4_1: - 3 frac[1,2] - 3 frac[1,1] - 3 frac[1,0] + rows_tot_P4_1
   = 0
 R25: - 0.0208333333333333 rows_tot_P4_1 + normalized_rows_tot_P4_1 = 0
 row_capacity_CPU_1: rows_tot_CPU_1 <= 12
 row_capacity_P4_1: rows_tot_P4_1 <= 48
Bounds
 frac[0,1] <= 1
 frac[1,2] <= 1
 frac[0,0] <= 1
 frac[1,1] <= 1
 frac[1,0] <= 1
 frac[0,2] <= 1
 cols_1_2 <= 8192
 cols_1_1 <= 8192
 cols_1_0 <= 8192
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
 cols_helper_1_2[12] cols_helper_1_1[0] cols_helper_1_1[1]
 cols_helper_1_1[2] cols_helper_1_1[3] cols_helper_1_1[4]
 cols_helper_1_1[5] cols_helper_1_1[6] cols_helper_1_1[7]
 cols_helper_1_1[8] cols_helper_1_1[9] cols_helper_1_1[10]
 cols_helper_1_1[11] cols_helper_1_1[12] cols_helper_1_0[0]
 cols_helper_1_0[1] cols_helper_1_0[2] cols_helper_1_0[3]
 cols_helper_1_0[4] cols_helper_1_0[5] cols_helper_1_0[6]
 cols_helper_1_0[7] cols_helper_1_0[8] cols_helper_1_0[9]
 cols_helper_1_0[10] cols_helper_1_0[11] cols_helper_1_0[12]
Generals
 cols_1_2 cols_1_1 cols_1_0
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_P4_1 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU_1 )
End

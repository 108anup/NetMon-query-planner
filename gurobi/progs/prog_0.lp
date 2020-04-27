\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU_Cluster: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU_1
  CPU_Cluster_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU_Cluster
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_rows_P4: Priority=80 Weight=1 AbsTol=0 RelTol=0
   rows_tot_P4_1
  P4_rows_load: Priority=70 Weight=1 AbsTol=0 RelTol=0
   max_rows_P4
  tot_mem_CPU_Cluster: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU_1
  CPU_Cluster_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
  tot_mem_P4: Priority=40 Weight=1 AbsTol=0 RelTol=0
   mem_tot_P4_1
  P4_load_mem: Priority=30 Weight=1 AbsTol=0 RelTol=0
   max_mem_P4
Subject To
 cov_0_0: frac[1,0] + frac[0,0] >= 1
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
 cols_acc_1_0: - 435 frac[1,0] + cols_1_0 >= 0
 accuracy_1_0: mem[1,0] - 0.01171875 cols_1_0 = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 mem_tot_CPU_1: - mem[0,0] + mem_tot_CPU_1 = 0
 R7: - 3.05176e-05 mem_tot_CPU_1 + normalized_mem_tot_CPU_1 = 0
 rows_tot_CPU_1: - 3 frac[0,0] + rows_tot_CPU_1 = 0
 R9: - 0.0833333333333333 rows_tot_CPU_1 + normalized_rows_tot_CPU_1 = 0
 mem_tot_P4_1: - mem[1,0] + mem_tot_P4_1 = 0
 R11: - 1.6276041666666666e-04 mem_tot_P4_1 + normalized_mem_tot_P4_1 = 0
 rows_tot_P4_1: - 3 frac[1,0] + rows_tot_P4_1 = 0
 R13: - 0.0208333333333333 rows_tot_P4_1 + normalized_rows_tot_P4_1 = 0
 row_capacity_CPU_1: rows_tot_CPU_1 <= 12
 row_capacity_P4_1: rows_tot_P4_1 <= 48
Bounds
 frac[1,0] <= 1
 frac[0,0] <= 1
 cols_1_0 <= 131072
 mem_tot_CPU_1 <= 32768
 normalized_mem_tot_CPU_1 <= 1
 normalized_rows_tot_CPU_1 <= 1
 mem_tot_P4_1 <= 6144
 normalized_mem_tot_P4_1 <= 1
 normalized_rows_tot_P4_1 <= 1
Binaries
 cols_helper_1_0[0] cols_helper_1_0[1] cols_helper_1_0[2]
 cols_helper_1_0[3] cols_helper_1_0[4] cols_helper_1_0[5]
 cols_helper_1_0[6] cols_helper_1_0[7] cols_helper_1_0[8]
 cols_helper_1_0[9] cols_helper_1_0[10] cols_helper_1_0[11]
 cols_helper_1_0[12] cols_helper_1_0[13] cols_helper_1_0[14]
 cols_helper_1_0[15] cols_helper_1_0[16]
Generals
 cols_1_0
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_P4_1 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU_1 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_P4_1 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU_1 )
End

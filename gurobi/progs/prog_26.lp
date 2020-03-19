\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU_Cluster: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU_0 + rows_tot_CPU_1 + rows_tot_CPU_2
  CPU_Cluster_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU_Cluster
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ4: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ5: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_mem_CPU_Cluster: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU_0 + mem_tot_CPU_1 + mem_tot_CPU_2
  CPU_Cluster_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
Subject To
 cov_0_0: frac[0,0] >= 1
 cov_1_0: frac[1,0] >= 1
 cov_2_0: frac[2,0] >= 1
 accuracy_2_0: - 5.09765625 frac[2,0] + mem[2,0] = 0
 accuracy_1_0: - 5.09765625 frac[1,0] + mem[1,0] = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 mem_tot_CPU_0: - mem[0,0] + mem_tot_CPU_0 = 0
 R7: - 3.05176e-05 mem_tot_CPU_0 + normalized_mem_tot_CPU_0 = 0
 rows_tot_CPU_0: - 3 frac[0,0] + rows_tot_CPU_0 = 0
 R9: - 0.0833333333333333 rows_tot_CPU_0 + normalized_rows_tot_CPU_0 = 0
 mem_tot_CPU_1: - mem[1,0] + mem_tot_CPU_1 = 0
 R11: - 3.05176e-05 mem_tot_CPU_1 + normalized_mem_tot_CPU_1 = 0
 rows_tot_CPU_1: - 3 frac[1,0] + rows_tot_CPU_1 = 0
 R13: - 0.0833333333333333 rows_tot_CPU_1 + normalized_rows_tot_CPU_1 = 0
 mem_tot_CPU_2: - mem[2,0] + mem_tot_CPU_2 = 0
 R15: - 3.05176e-05 mem_tot_CPU_2 + normalized_mem_tot_CPU_2 = 0
 rows_tot_CPU_2: - 3 frac[2,0] + rows_tot_CPU_2 = 0
 R17: - 0.0833333333333333 rows_tot_CPU_2 + normalized_rows_tot_CPU_2 = 0
 row_capacity_CPU_0: rows_tot_CPU_0 <= 12
 row_capacity_CPU_1: rows_tot_CPU_1 <= 12
 row_capacity_CPU_2: rows_tot_CPU_2 <= 12
Bounds
 mem_tot_CPU_0 <= 32768
 normalized_mem_tot_CPU_0 <= 1
 normalized_rows_tot_CPU_0 <= 1
 mem_tot_CPU_1 <= 32768
 normalized_mem_tot_CPU_1 <= 1
 normalized_rows_tot_CPU_1 <= 1
 mem_tot_CPU_2 <= 32768
 normalized_mem_tot_CPU_2 <= 1
 normalized_rows_tot_CPU_2 <= 1
Binaries
 frac[2,0] frac[1,0] frac[0,0]
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU_0 , normalized_mem_tot_CPU_1 ,
   normalized_mem_tot_CPU_2 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU_0 ,
   normalized_rows_tot_CPU_1 , normalized_rows_tot_CPU_2 )
End

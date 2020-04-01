\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU_Cluster: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU6 + rows_tot_CPU19 + rows_tot_CPU30 + rows_tot_CPU31
  CPU_Cluster_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU_Cluster
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ4: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ5: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_mem_CPU_Cluster: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU6 + mem_tot_CPU19 + mem_tot_CPU30 + mem_tot_CPU31
  CPU_Cluster_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
Subject To
 cov_0_1: frac[2,1] + frac[1,1] >= 0.2182996363636366
 cov_1_0: frac[0,0] + frac[2,0] >= 0.8
 cov_2_1: frac[3,1] + frac[2,1] >= 0.2182996363636366
 cov_3_0: frac[3,0] >= 0.7
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 accuracy_3_0: - 5.09765625 frac[3,0] + mem[3,0] = 0
 accuracy_3_1: - 5.09765625 frac[3,1] + mem[3,1] = 0
 accuracy_2_1: - 5.09765625 frac[2,1] + mem[2,1] = 0
 accuracy_2_0: - 5.09765625 frac[2,0] + mem[2,0] = 0
 accuracy_1_1: - 5.09765625 frac[1,1] + mem[1,1] = 0
 mem_tot_CPU6: - mem[0,0] + mem_tot_CPU6 = 0
 R11: - 3.05176e-05 mem_tot_CPU6 + normalized_mem_tot_CPU6 = 0
 rows_tot_CPU6: - 3 frac[0,0] + rows_tot_CPU6 = 0
 R13: - 0.0833333333333333 rows_tot_CPU6 + normalized_rows_tot_CPU6 = 0
 mem_tot_CPU19: - mem[1,1] + mem_tot_CPU19 = 0
 R15: - 3.05176e-05 mem_tot_CPU19 + normalized_mem_tot_CPU19 = 0
 rows_tot_CPU19: - 3 frac[1,1] + rows_tot_CPU19 = 0
 R17: - 0.0833333333333333 rows_tot_CPU19 + normalized_rows_tot_CPU19 = 0
 mem_tot_CPU30: - mem[2,1] - mem[2,0] + mem_tot_CPU30 = 0
 R19: - 3.05176e-05 mem_tot_CPU30 + normalized_mem_tot_CPU30 = 0
 rows_tot_CPU30: - 3 frac[2,1] - 3 frac[2,0] + rows_tot_CPU30 = 0
 R21: - 0.0833333333333333 rows_tot_CPU30 + normalized_rows_tot_CPU30 = 0
 mem_tot_CPU31: - mem[3,0] - mem[3,1] + mem_tot_CPU31 = 0
 R23: - 3.05176e-05 mem_tot_CPU31 + normalized_mem_tot_CPU31 = 0
 rows_tot_CPU31: - 3 frac[3,0] - 3 frac[3,1] + rows_tot_CPU31 = 0
 R25: - 0.0833333333333333 rows_tot_CPU31 + normalized_rows_tot_CPU31 = 0
 row_capacity_CPU6: rows_tot_CPU6 <= 12
 row_capacity_CPU19: rows_tot_CPU19 <= 12
 row_capacity_CPU30: rows_tot_CPU30 <= 12
 row_capacity_CPU31: rows_tot_CPU31 <= 12
Bounds
 frac[0,0] <= 1
 frac[3,0] <= 1
 frac[3,1] <= 1
 frac[2,1] <= 1
 frac[2,0] <= 1
 frac[1,1] <= 1
 mem_tot_CPU6 <= 32768
 normalized_mem_tot_CPU6 <= 1
 normalized_rows_tot_CPU6 <= 1
 mem_tot_CPU19 <= 32768
 normalized_mem_tot_CPU19 <= 1
 normalized_rows_tot_CPU19 <= 1
 mem_tot_CPU30 <= 32768
 normalized_mem_tot_CPU30 <= 1
 normalized_rows_tot_CPU30 <= 1
 mem_tot_CPU31 <= 32768
 normalized_mem_tot_CPU31 <= 1
 normalized_rows_tot_CPU31 <= 1
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU6 , normalized_mem_tot_CPU19 ,
   normalized_mem_tot_CPU30 , normalized_mem_tot_CPU31 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU6 ,
   normalized_rows_tot_CPU19 , normalized_rows_tot_CPU30 ,
   normalized_rows_tot_CPU31 )
End

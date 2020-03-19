\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU_Cluster: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU6 + rows_tot_CPU7
  CPU_Cluster_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU_Cluster
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ4: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ5: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_mem_CPU_Cluster: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU6 + mem_tot_CPU7
  CPU_Cluster_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
Subject To
 cov_0_1: frac[1,1] >= 0.7
 cov_1_0: frac[0,0] >= 0.9
 cov_2_1: frac[0,1] >= 0.8
 cov_3_0: frac[1,0] >= 1
 cov_4_2: frac[0,2] >= 0.9
 cov_5_1: frac[0,1] + frac[1,1] >= 1
 cov_6_0: frac[0,0] >= 0.7
 cov_7_0: frac[1,0] >= 0.7
 cov_8_0: frac[0,0] >= 0.7
 cov_9_1: frac[1,1] >= 0.7
 cov_10_2: frac[1,2] >= 0.9
 accuracy_0_1: - 5.09765625 frac[0,1] + mem[0,1] = 0
 accuracy_1_2: - 5.09765625 frac[1,2] + mem[1,2] = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 accuracy_1_1: - 5.09765625 frac[1,1] + mem[1,1] = 0
 accuracy_1_0: - 5.09765625 frac[1,0] + mem[1,0] = 0
 accuracy_0_2: - 5.09765625 frac[0,2] + mem[0,2] = 0
 mem_tot_CPU6: - mem[0,1] - mem[0,0] - mem[0,2] + mem_tot_CPU6 = 0
 R18: - 3.05176e-05 mem_tot_CPU6 + normalized_mem_tot_CPU6 = 0
 rows_tot_CPU6: - 3 frac[0,1] - 3 frac[0,0] - 3 frac[0,2] + rows_tot_CPU6
   = 0
 R20: - 0.0833333333333333 rows_tot_CPU6 + normalized_rows_tot_CPU6 = 0
 mem_tot_CPU7: - mem[1,2] - mem[1,1] - mem[1,0] + mem_tot_CPU7 = 0
 R22: - 3.05176e-05 mem_tot_CPU7 + normalized_mem_tot_CPU7 = 0
 rows_tot_CPU7: - 3 frac[1,2] - 3 frac[1,1] - 3 frac[1,0] + rows_tot_CPU7
   = 0
 R24: - 0.0833333333333333 rows_tot_CPU7 + normalized_rows_tot_CPU7 = 0
 row_capacity_CPU6: rows_tot_CPU6 <= 12
 row_capacity_CPU7: rows_tot_CPU7 <= 12
Bounds
 mem_tot_CPU6 <= 32768
 normalized_mem_tot_CPU6 <= 1
 normalized_rows_tot_CPU6 <= 1
 mem_tot_CPU7 <= 32768
 normalized_mem_tot_CPU7 <= 1
 normalized_rows_tot_CPU7 <= 1
Binaries
 frac[0,1] frac[1,2] frac[0,0] frac[1,1] frac[1,0] frac[0,2]
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU6 , normalized_mem_tot_CPU7 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU6 ,
   normalized_rows_tot_CPU7 )
End

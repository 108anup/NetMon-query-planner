\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU_Cluster: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU4328 + rows_tot_CPU1340 + rows_tot_CPU1271 + rows_tot_CPU2450
   + rows_tot_CPU1117 + rows_tot_CPU957 + rows_tot_CPU196 + rows_tot_CPU46
  CPU_Cluster_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU_Cluster
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ4: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ5: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_mem_CPU_Cluster: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU4328 + mem_tot_CPU1340 + mem_tot_CPU1271 + mem_tot_CPU2450
   + mem_tot_CPU1117 + mem_tot_CPU957 + mem_tot_CPU196 + mem_tot_CPU46
  CPU_Cluster_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
Subject To
 cov_0_1: frac[5,1] >= 0.9
 cov_1_0: frac[3,0] >= 0.9
 cov_2_1: frac[0,1] + frac[1,1] >= 0.7
 cov_3_1: frac[7,1] + frac[1,1] >= 0.7
 cov_4_1: frac[0,1] + frac[7,1] >= 0.9
 accuracy_0_1: - 509.68359375 frac[0,1] + mem[0,1] = 0
 accuracy_7_1: - 509.68359375 frac[7,1] + mem[7,1] = 0
 accuracy_3_0: - 509.68359375 frac[3,0] + mem[3,0] = 0
 accuracy_5_1: - 509.68359375 frac[5,1] + mem[5,1] = 0
 accuracy_1_1: - 509.68359375 frac[1,1] + mem[1,1] = 0
 mem_tot_CPU4328: - mem[0,1] + mem_tot_CPU4328 = 0
 R11: - 3.05176e-05 mem_tot_CPU4328 + normalized_mem_tot_CPU4328 = 0
 rows_tot_CPU4328: - 3 frac[0,1] + rows_tot_CPU4328 = 0
 R13: - 0.0833333333333333 rows_tot_CPU4328 + normalized_rows_tot_CPU4328
   = 0
 mem_tot_CPU1340: - mem[1,1] + mem_tot_CPU1340 = 0
 R15: - 3.05176e-05 mem_tot_CPU1340 + normalized_mem_tot_CPU1340 = 0
 rows_tot_CPU1340: - 3 frac[1,1] + rows_tot_CPU1340 = 0
 R17: - 0.0833333333333333 rows_tot_CPU1340 + normalized_rows_tot_CPU1340
   = 0
 mem_tot_CPU1271: mem_tot_CPU1271 = 0
 R19: - 3.05176e-05 mem_tot_CPU1271 + normalized_mem_tot_CPU1271 = 0
 rows_tot_CPU1271: rows_tot_CPU1271 = 0
 R21: - 0.0833333333333333 rows_tot_CPU1271 + normalized_rows_tot_CPU1271
   = 0
 mem_tot_CPU2450: - mem[3,0] + mem_tot_CPU2450 = 0
 R23: - 3.05176e-05 mem_tot_CPU2450 + normalized_mem_tot_CPU2450 = 0
 rows_tot_CPU2450: - 3 frac[3,0] + rows_tot_CPU2450 = 0
 R25: - 0.0833333333333333 rows_tot_CPU2450 + normalized_rows_tot_CPU2450
   = 0
 mem_tot_CPU1117: mem_tot_CPU1117 = 0
 R27: - 3.05176e-05 mem_tot_CPU1117 + normalized_mem_tot_CPU1117 = 0
 rows_tot_CPU1117: rows_tot_CPU1117 = 0
 R29: - 0.0833333333333333 rows_tot_CPU1117 + normalized_rows_tot_CPU1117
   = 0
 mem_tot_CPU957: - mem[5,1] + mem_tot_CPU957 = 0
 R31: - 3.05176e-05 mem_tot_CPU957 + normalized_mem_tot_CPU957 = 0
 rows_tot_CPU957: - 3 frac[5,1] + rows_tot_CPU957 = 0
 R33: - 0.0833333333333333 rows_tot_CPU957 + normalized_rows_tot_CPU957
   = 0
 mem_tot_CPU196: mem_tot_CPU196 = 0
 R35: - 3.05176e-05 mem_tot_CPU196 + normalized_mem_tot_CPU196 = 0
 rows_tot_CPU196: rows_tot_CPU196 = 0
 R37: - 0.0833333333333333 rows_tot_CPU196 + normalized_rows_tot_CPU196
   = 0
 mem_tot_CPU46: - mem[7,1] + mem_tot_CPU46 = 0
 R39: - 3.05176e-05 mem_tot_CPU46 + normalized_mem_tot_CPU46 = 0
 rows_tot_CPU46: - 3 frac[7,1] + rows_tot_CPU46 = 0
 R41: - 0.0833333333333333 rows_tot_CPU46 + normalized_rows_tot_CPU46 = 0
 row_capacity_CPU4328: rows_tot_CPU4328 <= 12
 row_capacity_CPU1340: rows_tot_CPU1340 <= 12
 row_capacity_CPU1271: rows_tot_CPU1271 <= 12
 row_capacity_CPU2450: rows_tot_CPU2450 <= 12
 row_capacity_CPU1117: rows_tot_CPU1117 <= 12
 row_capacity_CPU957: rows_tot_CPU957 <= 12
 row_capacity_CPU196: rows_tot_CPU196 <= 12
 row_capacity_CPU46: rows_tot_CPU46 <= 12
Bounds
 mem_tot_CPU4328 <= 32768
 normalized_mem_tot_CPU4328 <= 1
 normalized_rows_tot_CPU4328 <= 1
 mem_tot_CPU1340 <= 32768
 normalized_mem_tot_CPU1340 <= 1
 normalized_rows_tot_CPU1340 <= 1
 mem_tot_CPU1271 <= 32768
 normalized_mem_tot_CPU1271 <= 1
 normalized_rows_tot_CPU1271 <= 1
 mem_tot_CPU2450 <= 32768
 normalized_mem_tot_CPU2450 <= 1
 normalized_rows_tot_CPU2450 <= 1
 mem_tot_CPU1117 <= 32768
 normalized_mem_tot_CPU1117 <= 1
 normalized_rows_tot_CPU1117 <= 1
 mem_tot_CPU957 <= 32768
 normalized_mem_tot_CPU957 <= 1
 normalized_rows_tot_CPU957 <= 1
 mem_tot_CPU196 <= 32768
 normalized_mem_tot_CPU196 <= 1
 normalized_rows_tot_CPU196 <= 1
 mem_tot_CPU46 <= 32768
 normalized_mem_tot_CPU46 <= 1
 normalized_rows_tot_CPU46 <= 1
Binaries
 frac[0,1] frac[7,1] frac[3,0] frac[5,1] frac[1,1]
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU4328 , normalized_mem_tot_CPU1340 ,
   normalized_mem_tot_CPU1271 , normalized_mem_tot_CPU2450 ,
   normalized_mem_tot_CPU1117 , normalized_mem_tot_CPU957 ,
   normalized_mem_tot_CPU196 , normalized_mem_tot_CPU46 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX (
   normalized_rows_tot_CPU4328 , normalized_rows_tot_CPU1340 ,
   normalized_rows_tot_CPU1271 , normalized_rows_tot_CPU2450 ,
   normalized_rows_tot_CPU1117 , normalized_rows_tot_CPU957 ,
   normalized_rows_tot_CPU196 , normalized_rows_tot_CPU46 )
End

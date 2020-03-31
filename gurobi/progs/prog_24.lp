\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU_Cluster: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU24 + rows_tot_CPU13 + rows_tot_CPU14 + rows_tot_CPU15
   + rows_tot_CPU12 + rows_tot_CPU10 + rows_tot_CPU22 + rows_tot_CPU8
  CPU_Cluster_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU_Cluster
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ4: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ5: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_mem_CPU_Cluster: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU24 + mem_tot_CPU13 + mem_tot_CPU14 + mem_tot_CPU15
   + mem_tot_CPU12 + mem_tot_CPU10 + mem_tot_CPU22 + mem_tot_CPU8
  CPU_Cluster_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
Subject To
 cov_0_0: frac[3,0] >= 2.64698e-23
 cov_1_1: frac[7,1] >= 1.96188e-07
 cov_2_1: frac[5,1] + frac[4,1] >= 1.96188e-07
 cov_3_1: frac[6,1] + frac[1,1] >= 1.96188e-07
 cov_4_1: frac[3,1] + frac[4,1] >= 1.96188e-07
 accuracy_7_1: - 5.09765625 frac[7,1] + mem[7,1] = 0
 accuracy_3_0: - 5.09765625 frac[3,0] + mem[3,0] = 0
 accuracy_6_1: - 5.09765625 frac[6,1] + mem[6,1] = 0
 accuracy_3_1: - 5.09765625 frac[3,1] + mem[3,1] = 0
 accuracy_5_1: - 5.09765625 frac[5,1] + mem[5,1] = 0
 accuracy_4_1: - 5.09765625 frac[4,1] + mem[4,1] = 0
 accuracy_1_1: - 5.09765625 frac[1,1] + mem[1,1] = 0
 mem_tot_CPU24: mem_tot_CPU24 = 0
 R13: - 3.05176e-05 mem_tot_CPU24 + normalized_mem_tot_CPU24 = 0
 rows_tot_CPU24: rows_tot_CPU24 = 0
 R15: - 0.0833333333333333 rows_tot_CPU24 + normalized_rows_tot_CPU24 = 0
 mem_tot_CPU13: - mem[1,1] + mem_tot_CPU13 = 0
 R17: - 3.05176e-05 mem_tot_CPU13 + normalized_mem_tot_CPU13 = 0
 rows_tot_CPU13: - 3 frac[1,1] + rows_tot_CPU13 = 0
 R19: - 0.0833333333333333 rows_tot_CPU13 + normalized_rows_tot_CPU13 = 0
 mem_tot_CPU14: mem_tot_CPU14 = 0
 R21: - 3.05176e-05 mem_tot_CPU14 + normalized_mem_tot_CPU14 = 0
 rows_tot_CPU14: rows_tot_CPU14 = 0
 R23: - 0.0833333333333333 rows_tot_CPU14 + normalized_rows_tot_CPU14 = 0
 mem_tot_CPU15: - mem[3,0] - mem[3,1] + mem_tot_CPU15 = 0
 R25: - 3.05176e-05 mem_tot_CPU15 + normalized_mem_tot_CPU15 = 0
 rows_tot_CPU15: - 3 frac[3,0] - 3 frac[3,1] + rows_tot_CPU15 = 0
 R27: - 0.0833333333333333 rows_tot_CPU15 + normalized_rows_tot_CPU15 = 0
 mem_tot_CPU12: - mem[4,1] + mem_tot_CPU12 = 0
 R29: - 3.05176e-05 mem_tot_CPU12 + normalized_mem_tot_CPU12 = 0
 rows_tot_CPU12: - 3 frac[4,1] + rows_tot_CPU12 = 0
 R31: - 0.0833333333333333 rows_tot_CPU12 + normalized_rows_tot_CPU12 = 0
 mem_tot_CPU10: - mem[5,1] + mem_tot_CPU10 = 0
 R33: - 3.05176e-05 mem_tot_CPU10 + normalized_mem_tot_CPU10 = 0
 rows_tot_CPU10: - 3 frac[5,1] + rows_tot_CPU10 = 0
 R35: - 0.0833333333333333 rows_tot_CPU10 + normalized_rows_tot_CPU10 = 0
 mem_tot_CPU22: - mem[6,1] + mem_tot_CPU22 = 0
 R37: - 3.05176e-05 mem_tot_CPU22 + normalized_mem_tot_CPU22 = 0
 rows_tot_CPU22: - 3 frac[6,1] + rows_tot_CPU22 = 0
 R39: - 0.0833333333333333 rows_tot_CPU22 + normalized_rows_tot_CPU22 = 0
 mem_tot_CPU8: - mem[7,1] + mem_tot_CPU8 = 0
 R41: - 3.05176e-05 mem_tot_CPU8 + normalized_mem_tot_CPU8 = 0
 rows_tot_CPU8: - 3 frac[7,1] + rows_tot_CPU8 = 0
 R43: - 0.0833333333333333 rows_tot_CPU8 + normalized_rows_tot_CPU8 = 0
 row_capacity_CPU24: rows_tot_CPU24 <= 12
 row_capacity_CPU13: rows_tot_CPU13 <= 12
 row_capacity_CPU14: rows_tot_CPU14 <= 12
 row_capacity_CPU15: rows_tot_CPU15 <= 12
 row_capacity_CPU12: rows_tot_CPU12 <= 12
 row_capacity_CPU10: rows_tot_CPU10 <= 12
 row_capacity_CPU22: rows_tot_CPU22 <= 12
 row_capacity_CPU8: rows_tot_CPU8 <= 12
Bounds
 frac[7,1] <= 1
 frac[3,0] <= 1
 frac[6,1] <= 1
 frac[3,1] <= 1
 frac[5,1] <= 1
 frac[4,1] <= 1
 frac[1,1] <= 1
 mem_tot_CPU24 <= 32768
 normalized_mem_tot_CPU24 <= 1
 normalized_rows_tot_CPU24 <= 1
 mem_tot_CPU13 <= 32768
 normalized_mem_tot_CPU13 <= 1
 normalized_rows_tot_CPU13 <= 1
 mem_tot_CPU14 <= 32768
 normalized_mem_tot_CPU14 <= 1
 normalized_rows_tot_CPU14 <= 1
 mem_tot_CPU15 <= 32768
 normalized_mem_tot_CPU15 <= 1
 normalized_rows_tot_CPU15 <= 1
 mem_tot_CPU12 <= 32768
 normalized_mem_tot_CPU12 <= 1
 normalized_rows_tot_CPU12 <= 1
 mem_tot_CPU10 <= 32768
 normalized_mem_tot_CPU10 <= 1
 normalized_rows_tot_CPU10 <= 1
 mem_tot_CPU22 <= 32768
 normalized_mem_tot_CPU22 <= 1
 normalized_rows_tot_CPU22 <= 1
 mem_tot_CPU8 <= 32768
 normalized_mem_tot_CPU8 <= 1
 normalized_rows_tot_CPU8 <= 1
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU24 , normalized_mem_tot_CPU13 ,
   normalized_mem_tot_CPU14 , normalized_mem_tot_CPU15 ,
   normalized_mem_tot_CPU12 , normalized_mem_tot_CPU10 ,
   normalized_mem_tot_CPU22 , normalized_mem_tot_CPU8 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU24 ,
   normalized_rows_tot_CPU13 , normalized_rows_tot_CPU14 ,
   normalized_rows_tot_CPU15 , normalized_rows_tot_CPU12 ,
   normalized_rows_tot_CPU10 , normalized_rows_tot_CPU22 ,
   normalized_rows_tot_CPU8 )
End

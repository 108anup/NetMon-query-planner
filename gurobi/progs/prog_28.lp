\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU_Cluster: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU468 + rows_tot_CPU810 + rows_tot_CPU671 + rows_tot_CPU617
   + rows_tot_CPU386 + rows_tot_CPU664 + rows_tot_CPU3 + rows_tot_CPU123
  CPU_Cluster_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU_Cluster
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ4: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ5: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_mem_CPU_Cluster: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU468 + mem_tot_CPU810 + mem_tot_CPU671 + mem_tot_CPU617
   + mem_tot_CPU386 + mem_tot_CPU664 + mem_tot_CPU3 + mem_tot_CPU123
  CPU_Cluster_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
Subject To
 cov_0_0: frac[0,0] + frac[2,0] >= 1
 cov_1_0: frac[6,0] + frac[4,0] >= 0.9
 cov_2_0: frac[1,0] + frac[4,0] >= 0.8
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 accuracy_6_0: - 5.09765625 frac[6,0] + mem[6,0] = 0
 accuracy_2_0: - 5.09765625 frac[2,0] + mem[2,0] = 0
 accuracy_1_0: - 5.09765625 frac[1,0] + mem[1,0] = 0
 accuracy_4_0: - 5.09765625 frac[4,0] + mem[4,0] = 0
 mem_tot_CPU468: - mem[0,0] + mem_tot_CPU468 = 0
 R9: - 3.05176e-05 mem_tot_CPU468 + normalized_mem_tot_CPU468 = 0
 rows_tot_CPU468: - 3 frac[0,0] + rows_tot_CPU468 = 0
 R11: - 0.0833333333333333 rows_tot_CPU468 + normalized_rows_tot_CPU468
   = 0
 mem_tot_CPU810: - mem[1,0] + mem_tot_CPU810 = 0
 R13: - 3.05176e-05 mem_tot_CPU810 + normalized_mem_tot_CPU810 = 0
 rows_tot_CPU810: - 3 frac[1,0] + rows_tot_CPU810 = 0
 R15: - 0.0833333333333333 rows_tot_CPU810 + normalized_rows_tot_CPU810
   = 0
 mem_tot_CPU671: - mem[2,0] + mem_tot_CPU671 = 0
 R17: - 3.05176e-05 mem_tot_CPU671 + normalized_mem_tot_CPU671 = 0
 rows_tot_CPU671: - 3 frac[2,0] + rows_tot_CPU671 = 0
 R19: - 0.0833333333333333 rows_tot_CPU671 + normalized_rows_tot_CPU671
   = 0
 mem_tot_CPU617: mem_tot_CPU617 = 0
 R21: - 3.05176e-05 mem_tot_CPU617 + normalized_mem_tot_CPU617 = 0
 rows_tot_CPU617: rows_tot_CPU617 = 0
 R23: - 0.0833333333333333 rows_tot_CPU617 + normalized_rows_tot_CPU617
   = 0
 mem_tot_CPU386: - mem[4,0] + mem_tot_CPU386 = 0
 R25: - 3.05176e-05 mem_tot_CPU386 + normalized_mem_tot_CPU386 = 0
 rows_tot_CPU386: - 3 frac[4,0] + rows_tot_CPU386 = 0
 R27: - 0.0833333333333333 rows_tot_CPU386 + normalized_rows_tot_CPU386
   = 0
 mem_tot_CPU664: mem_tot_CPU664 = 0
 R29: - 3.05176e-05 mem_tot_CPU664 + normalized_mem_tot_CPU664 = 0
 rows_tot_CPU664: rows_tot_CPU664 = 0
 R31: - 0.0833333333333333 rows_tot_CPU664 + normalized_rows_tot_CPU664
   = 0
 mem_tot_CPU3: - mem[6,0] + mem_tot_CPU3 = 0
 R33: - 3.05176e-05 mem_tot_CPU3 + normalized_mem_tot_CPU3 = 0
 rows_tot_CPU3: - 3 frac[6,0] + rows_tot_CPU3 = 0
 R35: - 0.0833333333333333 rows_tot_CPU3 + normalized_rows_tot_CPU3 = 0
 mem_tot_CPU123: mem_tot_CPU123 = 0
 R37: - 3.05176e-05 mem_tot_CPU123 + normalized_mem_tot_CPU123 = 0
 rows_tot_CPU123: rows_tot_CPU123 = 0
 R39: - 0.0833333333333333 rows_tot_CPU123 + normalized_rows_tot_CPU123
   = 0
 row_capacity_CPU468: rows_tot_CPU468 <= 12
 row_capacity_CPU810: rows_tot_CPU810 <= 12
 row_capacity_CPU671: rows_tot_CPU671 <= 12
 row_capacity_CPU617: rows_tot_CPU617 <= 12
 row_capacity_CPU386: rows_tot_CPU386 <= 12
 row_capacity_CPU664: rows_tot_CPU664 <= 12
 row_capacity_CPU3: rows_tot_CPU3 <= 12
 row_capacity_CPU123: rows_tot_CPU123 <= 12
Bounds
 mem_tot_CPU468 <= 32768
 normalized_mem_tot_CPU468 <= 1
 normalized_rows_tot_CPU468 <= 1
 mem_tot_CPU810 <= 32768
 normalized_mem_tot_CPU810 <= 1
 normalized_rows_tot_CPU810 <= 1
 mem_tot_CPU671 <= 32768
 normalized_mem_tot_CPU671 <= 1
 normalized_rows_tot_CPU671 <= 1
 mem_tot_CPU617 <= 32768
 normalized_mem_tot_CPU617 <= 1
 normalized_rows_tot_CPU617 <= 1
 mem_tot_CPU386 <= 32768
 normalized_mem_tot_CPU386 <= 1
 normalized_rows_tot_CPU386 <= 1
 mem_tot_CPU664 <= 32768
 normalized_mem_tot_CPU664 <= 1
 normalized_rows_tot_CPU664 <= 1
 mem_tot_CPU3 <= 32768
 normalized_mem_tot_CPU3 <= 1
 normalized_rows_tot_CPU3 <= 1
 mem_tot_CPU123 <= 32768
 normalized_mem_tot_CPU123 <= 1
 normalized_rows_tot_CPU123 <= 1
Binaries
 frac[0,0] frac[6,0] frac[2,0] frac[1,0] frac[4,0]
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU468 , normalized_mem_tot_CPU810 ,
   normalized_mem_tot_CPU671 , normalized_mem_tot_CPU617 ,
   normalized_mem_tot_CPU386 , normalized_mem_tot_CPU664 ,
   normalized_mem_tot_CPU3 , normalized_mem_tot_CPU123 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX (
   normalized_rows_tot_CPU468 , normalized_rows_tot_CPU810 ,
   normalized_rows_tot_CPU671 , normalized_rows_tot_CPU617 ,
   normalized_rows_tot_CPU386 , normalized_rows_tot_CPU664 ,
   normalized_rows_tot_CPU3 , normalized_rows_tot_CPU123 )
End

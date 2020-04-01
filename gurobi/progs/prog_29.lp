\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU_Cluster: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU46807 + rows_tot_CPU17007 + rows_tot_CPU11763
   + rows_tot_CPU24839 + rows_tot_CPU27912 + rows_tot_CPU30482
   + rows_tot_CPU34930 + rows_tot_CPU41011 + rows_tot_CPU23947
   + rows_tot_CPU33917 + rows_tot_CPU42963 + rows_tot_CPU23631
   + rows_tot_CPU31016 + rows_tot_CPU9692 + rows_tot_CPU4786
   + rows_tot_CPU32990
  CPU_Cluster_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU_Cluster
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ4: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ5: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_mem_CPU_Cluster: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU46807 + mem_tot_CPU17007 + mem_tot_CPU11763 + mem_tot_CPU24839
   + mem_tot_CPU27912 + mem_tot_CPU30482 + mem_tot_CPU34930
   + mem_tot_CPU41011 + mem_tot_CPU23947 + mem_tot_CPU33917
   + mem_tot_CPU42963 + mem_tot_CPU23631 + mem_tot_CPU31016
   + mem_tot_CPU9692 + mem_tot_CPU4786 + mem_tot_CPU32990
  CPU_Cluster_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
Subject To
 cov_0_5: frac[0,5] >= 0.2
 cov_1_4: frac[4,4] + frac[6,4] >= 0.8
 cov_2_1: frac[5,1] + frac[4,1] >= 0.9
 cov_3_4: frac[0,4] + frac[2,4] >= 0.8
 cov_4_4: frac[4,4] + frac[1,4] >= 0.8
 cov_5_4: frac[0,4] + frac[7,4] >= 0.7
 cov_6_0: frac[15,0] + frac[10,0] >= 1
 cov_7_3: frac[10,3] + frac[9,3] >= 0.8
 cov_8_0: frac[10,0] + frac[13,0] >= 0.8
 cov_9_3: frac[15,3] + frac[11,3] >= 0.8
 cov_10_3: frac[10,3] >= 0.7
 cov_11_2: frac[15,2] + frac[9,2] >= 0.9
 cov_12_2: frac[11,2] + frac[9,2] >= 0.9
 accuracy_11_2: - 5.09765625 frac[11,2] + mem[11,2] = 0
 accuracy_5_1: - 5.09765625 frac[5,1] + mem[5,1] = 0
 accuracy_10_3: - 5.09765625 frac[10,3] + mem[10,3] = 0
 accuracy_4_4: - 5.09765625 frac[4,4] + mem[4,4] = 0
 accuracy_15_0: - 5.09765625 frac[15,0] + mem[15,0] = 0
 accuracy_0_4: - 5.09765625 frac[0,4] + mem[0,4] = 0
 accuracy_10_0: - 5.09765625 frac[10,0] + mem[10,0] = 0
 accuracy_4_1: - 5.09765625 frac[4,1] + mem[4,1] = 0
 accuracy_6_4: - 5.09765625 frac[6,4] + mem[6,4] = 0
 accuracy_15_3: - 5.09765625 frac[15,3] + mem[15,3] = 0
 accuracy_9_3: - 5.09765625 frac[9,3] + mem[9,3] = 0
 accuracy_1_4: - 5.09765625 frac[1,4] + mem[1,4] = 0
 accuracy_0_5: - 5.09765625 frac[0,5] + mem[0,5] = 0
 accuracy_15_2: - 5.09765625 frac[15,2] + mem[15,2] = 0
 accuracy_9_2: - 5.09765625 frac[9,2] + mem[9,2] = 0
 accuracy_11_3: - 5.09765625 frac[11,3] + mem[11,3] = 0
 accuracy_7_4: - 5.09765625 frac[7,4] + mem[7,4] = 0
 accuracy_2_4: - 5.09765625 frac[2,4] + mem[2,4] = 0
 accuracy_13_0: - 5.09765625 frac[13,0] + mem[13,0] = 0
 mem_tot_CPU46807: - mem[0,4] - mem[0,5] + mem_tot_CPU46807 = 0
 R33: - 3.05176e-05 mem_tot_CPU46807 + normalized_mem_tot_CPU46807 = 0
 rows_tot_CPU46807: - 3 frac[0,4] - 3 frac[0,5] + rows_tot_CPU46807 = 0
 R35: - 0.0833333333333333 rows_tot_CPU46807 + normalized_rows_tot_CPU46807
   = 0
 mem_tot_CPU17007: - mem[1,4] + mem_tot_CPU17007 = 0
 R37: - 3.05176e-05 mem_tot_CPU17007 + normalized_mem_tot_CPU17007 = 0
 rows_tot_CPU17007: - 3 frac[1,4] + rows_tot_CPU17007 = 0
 R39: - 0.0833333333333333 rows_tot_CPU17007 + normalized_rows_tot_CPU17007
   = 0
 mem_tot_CPU11763: - mem[2,4] + mem_tot_CPU11763 = 0
 R41: - 3.05176e-05 mem_tot_CPU11763 + normalized_mem_tot_CPU11763 = 0
 rows_tot_CPU11763: - 3 frac[2,4] + rows_tot_CPU11763 = 0
 R43: - 0.0833333333333333 rows_tot_CPU11763 + normalized_rows_tot_CPU11763
   = 0
 mem_tot_CPU24839: mem_tot_CPU24839 = 0
 R45: - 3.05176e-05 mem_tot_CPU24839 + normalized_mem_tot_CPU24839 = 0
 rows_tot_CPU24839: rows_tot_CPU24839 = 0
 R47: - 0.0833333333333333 rows_tot_CPU24839 + normalized_rows_tot_CPU24839
   = 0
 mem_tot_CPU27912: - mem[4,4] - mem[4,1] + mem_tot_CPU27912 = 0
 R49: - 3.05176e-05 mem_tot_CPU27912 + normalized_mem_tot_CPU27912 = 0
 rows_tot_CPU27912: - 3 frac[4,4] - 3 frac[4,1] + rows_tot_CPU27912 = 0
 R51: - 0.0833333333333333 rows_tot_CPU27912 + normalized_rows_tot_CPU27912
   = 0
 mem_tot_CPU30482: - mem[5,1] + mem_tot_CPU30482 = 0
 R53: - 3.05176e-05 mem_tot_CPU30482 + normalized_mem_tot_CPU30482 = 0
 rows_tot_CPU30482: - 3 frac[5,1] + rows_tot_CPU30482 = 0
 R55: - 0.0833333333333333 rows_tot_CPU30482 + normalized_rows_tot_CPU30482
   = 0
 mem_tot_CPU34930: - mem[6,4] + mem_tot_CPU34930 = 0
 R57: - 3.05176e-05 mem_tot_CPU34930 + normalized_mem_tot_CPU34930 = 0
 rows_tot_CPU34930: - 3 frac[6,4] + rows_tot_CPU34930 = 0
 R59: - 0.0833333333333333 rows_tot_CPU34930 + normalized_rows_tot_CPU34930
   = 0
 mem_tot_CPU41011: - mem[7,4] + mem_tot_CPU41011 = 0
 R61: - 3.05176e-05 mem_tot_CPU41011 + normalized_mem_tot_CPU41011 = 0
 rows_tot_CPU41011: - 3 frac[7,4] + rows_tot_CPU41011 = 0
 R63: - 0.0833333333333333 rows_tot_CPU41011 + normalized_rows_tot_CPU41011
   = 0
 mem_tot_CPU23947: mem_tot_CPU23947 = 0
 R65: - 3.05176e-05 mem_tot_CPU23947 + normalized_mem_tot_CPU23947 = 0
 rows_tot_CPU23947: rows_tot_CPU23947 = 0
 R67: - 0.0833333333333333 rows_tot_CPU23947 + normalized_rows_tot_CPU23947
   = 0
 mem_tot_CPU33917: - mem[9,3] - mem[9,2] + mem_tot_CPU33917 = 0
 R69: - 3.05176e-05 mem_tot_CPU33917 + normalized_mem_tot_CPU33917 = 0
 rows_tot_CPU33917: - 3 frac[9,3] - 3 frac[9,2] + rows_tot_CPU33917 = 0
 R71: - 0.0833333333333333 rows_tot_CPU33917 + normalized_rows_tot_CPU33917
   = 0
 mem_tot_CPU42963: - mem[10,3] - mem[10,0] + mem_tot_CPU42963 = 0
 R73: - 3.05176e-05 mem_tot_CPU42963 + normalized_mem_tot_CPU42963 = 0
 rows_tot_CPU42963: - 3 frac[10,3] - 3 frac[10,0] + rows_tot_CPU42963 = 0
 R75: - 0.0833333333333333 rows_tot_CPU42963 + normalized_rows_tot_CPU42963
   = 0
 mem_tot_CPU23631: - mem[11,2] - mem[11,3] + mem_tot_CPU23631 = 0
 R77: - 3.05176e-05 mem_tot_CPU23631 + normalized_mem_tot_CPU23631 = 0
 rows_tot_CPU23631: - 3 frac[11,2] - 3 frac[11,3] + rows_tot_CPU23631 = 0
 R79: - 0.0833333333333333 rows_tot_CPU23631 + normalized_rows_tot_CPU23631
   = 0
 mem_tot_CPU31016: mem_tot_CPU31016 = 0
 R81: - 3.05176e-05 mem_tot_CPU31016 + normalized_mem_tot_CPU31016 = 0
 rows_tot_CPU31016: rows_tot_CPU31016 = 0
 R83: - 0.0833333333333333 rows_tot_CPU31016 + normalized_rows_tot_CPU31016
   = 0
 mem_tot_CPU9692: - mem[13,0] + mem_tot_CPU9692 = 0
 R85: - 3.05176e-05 mem_tot_CPU9692 + normalized_mem_tot_CPU9692 = 0
 rows_tot_CPU9692: - 3 frac[13,0] + rows_tot_CPU9692 = 0
 R87: - 0.0833333333333333 rows_tot_CPU9692 + normalized_rows_tot_CPU9692
   = 0
 mem_tot_CPU4786: mem_tot_CPU4786 = 0
 R89: - 3.05176e-05 mem_tot_CPU4786 + normalized_mem_tot_CPU4786 = 0
 rows_tot_CPU4786: rows_tot_CPU4786 = 0
 R91: - 0.0833333333333333 rows_tot_CPU4786 + normalized_rows_tot_CPU4786
   = 0
 mem_tot_CPU32990: - mem[15,0] - mem[15,3] - mem[15,2] + mem_tot_CPU32990
   = 0
 R93: - 3.05176e-05 mem_tot_CPU32990 + normalized_mem_tot_CPU32990 = 0
 rows_tot_CPU32990: - 3 frac[15,0] - 3 frac[15,3] - 3 frac[15,2]
   + rows_tot_CPU32990 = 0
 R95: - 0.0833333333333333 rows_tot_CPU32990 + normalized_rows_tot_CPU32990
   = 0
 row_capacity_CPU46807: rows_tot_CPU46807 <= 12
 row_capacity_CPU17007: rows_tot_CPU17007 <= 12
 row_capacity_CPU11763: rows_tot_CPU11763 <= 12
 row_capacity_CPU24839: rows_tot_CPU24839 <= 12
 row_capacity_CPU27912: rows_tot_CPU27912 <= 12
 row_capacity_CPU30482: rows_tot_CPU30482 <= 12
 row_capacity_CPU34930: rows_tot_CPU34930 <= 12
 row_capacity_CPU41011: rows_tot_CPU41011 <= 12
 row_capacity_CPU23947: rows_tot_CPU23947 <= 12
 row_capacity_CPU33917: rows_tot_CPU33917 <= 12
 row_capacity_CPU42963: rows_tot_CPU42963 <= 12
 row_capacity_CPU23631: rows_tot_CPU23631 <= 12
 row_capacity_CPU31016: rows_tot_CPU31016 <= 12
 row_capacity_CPU9692: rows_tot_CPU9692 <= 12
 row_capacity_CPU4786: rows_tot_CPU4786 <= 12
 row_capacity_CPU32990: rows_tot_CPU32990 <= 12
Bounds
 frac[11,2] <= 1
 frac[5,1] <= 1
 frac[10,3] <= 1
 frac[4,4] <= 1
 frac[15,0] <= 1
 frac[0,4] <= 1
 frac[10,0] <= 1
 frac[4,1] <= 1
 frac[6,4] <= 1
 frac[15,3] <= 1
 frac[9,3] <= 1
 frac[1,4] <= 1
 frac[0,5] <= 1
 frac[15,2] <= 1
 frac[9,2] <= 1
 frac[11,3] <= 1
 frac[7,4] <= 1
 frac[2,4] <= 1
 frac[13,0] <= 1
 mem_tot_CPU46807 <= 32768
 normalized_mem_tot_CPU46807 <= 1
 normalized_rows_tot_CPU46807 <= 1
 mem_tot_CPU17007 <= 32768
 normalized_mem_tot_CPU17007 <= 1
 normalized_rows_tot_CPU17007 <= 1
 mem_tot_CPU11763 <= 32768
 normalized_mem_tot_CPU11763 <= 1
 normalized_rows_tot_CPU11763 <= 1
 mem_tot_CPU24839 <= 32768
 normalized_mem_tot_CPU24839 <= 1
 normalized_rows_tot_CPU24839 <= 1
 mem_tot_CPU27912 <= 32768
 normalized_mem_tot_CPU27912 <= 1
 normalized_rows_tot_CPU27912 <= 1
 mem_tot_CPU30482 <= 32768
 normalized_mem_tot_CPU30482 <= 1
 normalized_rows_tot_CPU30482 <= 1
 mem_tot_CPU34930 <= 32768
 normalized_mem_tot_CPU34930 <= 1
 normalized_rows_tot_CPU34930 <= 1
 mem_tot_CPU41011 <= 32768
 normalized_mem_tot_CPU41011 <= 1
 normalized_rows_tot_CPU41011 <= 1
 mem_tot_CPU23947 <= 32768
 normalized_mem_tot_CPU23947 <= 1
 normalized_rows_tot_CPU23947 <= 1
 mem_tot_CPU33917 <= 32768
 normalized_mem_tot_CPU33917 <= 1
 normalized_rows_tot_CPU33917 <= 1
 mem_tot_CPU42963 <= 32768
 normalized_mem_tot_CPU42963 <= 1
 normalized_rows_tot_CPU42963 <= 1
 mem_tot_CPU23631 <= 32768
 normalized_mem_tot_CPU23631 <= 1
 normalized_rows_tot_CPU23631 <= 1
 mem_tot_CPU31016 <= 32768
 normalized_mem_tot_CPU31016 <= 1
 normalized_rows_tot_CPU31016 <= 1
 mem_tot_CPU9692 <= 32768
 normalized_mem_tot_CPU9692 <= 1
 normalized_rows_tot_CPU9692 <= 1
 mem_tot_CPU4786 <= 32768
 normalized_mem_tot_CPU4786 <= 1
 normalized_rows_tot_CPU4786 <= 1
 mem_tot_CPU32990 <= 32768
 normalized_mem_tot_CPU32990 <= 1
 normalized_rows_tot_CPU32990 <= 1
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU46807 , normalized_mem_tot_CPU17007 ,
   normalized_mem_tot_CPU11763 , normalized_mem_tot_CPU24839 ,
   normalized_mem_tot_CPU27912 , normalized_mem_tot_CPU30482 ,
   normalized_mem_tot_CPU34930 , normalized_mem_tot_CPU41011 ,
   normalized_mem_tot_CPU23947 , normalized_mem_tot_CPU33917 ,
   normalized_mem_tot_CPU42963 , normalized_mem_tot_CPU23631 ,
   normalized_mem_tot_CPU31016 , normalized_mem_tot_CPU9692 ,
   normalized_mem_tot_CPU4786 , normalized_mem_tot_CPU32990 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX (
   normalized_rows_tot_CPU46807 , normalized_rows_tot_CPU17007 ,
   normalized_rows_tot_CPU11763 , normalized_rows_tot_CPU24839 ,
   normalized_rows_tot_CPU27912 , normalized_rows_tot_CPU30482 ,
   normalized_rows_tot_CPU34930 , normalized_rows_tot_CPU41011 ,
   normalized_rows_tot_CPU23947 , normalized_rows_tot_CPU33917 ,
   normalized_rows_tot_CPU42963 , normalized_rows_tot_CPU23631 ,
   normalized_rows_tot_CPU31016 , normalized_rows_tot_CPU9692 ,
   normalized_rows_tot_CPU4786 , normalized_rows_tot_CPU32990 )
End

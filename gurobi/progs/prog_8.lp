\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  OBJ0: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ1: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_rows_fixed_thr: Priority=80 Weight=1 AbsTol=0 RelTol=0
   rows_tot_P4_1 + rows_tot_P4_2
  rows_load_fixed_thr: Priority=70 Weight=1 AbsTol=0 RelTol=0
   max_rows_fixed_thr
  OBJ6: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ7: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_mem_fixed_thr: Priority=40 Weight=1 AbsTol=0 RelTol=0
   mem_tot_P4_1 + mem_tot_P4_2
  mem_load_fixed_thr: Priority=30 Weight=1 AbsTol=0 RelTol=0
   max_mem_fixed_thr
Subject To
 cov_0_0: frac[0,0] + frac[1,0] >= 1
 accuracy_1_0: - 192 frac[1,0] + mem[1,0] = 0
 accuracy_0_0: - 192 frac[0,0] + mem[0,0] = 0
 mem_tot_P4_1: - mem[0,0] + mem_tot_P4_1 = 0
 R4: - 1.6276041666666666e-04 mem_tot_P4_1 + normalized_mem_tot_P4_1 = 0
 rows_tot_P4_1: - 3 frac[0,0] + rows_tot_P4_1 = 0
 R6: - 0.0208333333333333 rows_tot_P4_1 + normalized_rows_tot_P4_1 = 0
 mem_tot_P4_2: - mem[1,0] + mem_tot_P4_2 = 0
 R8: - 1.6276041666666666e-04 mem_tot_P4_2 + normalized_mem_tot_P4_2 = 0
 rows_tot_P4_2: - 3 frac[1,0] + rows_tot_P4_2 = 0
 R10: - 0.0208333333333333 rows_tot_P4_2 + normalized_rows_tot_P4_2 = 0
 row_capacity_P4_1: rows_tot_P4_1 <= 48
 capacity_mem_par_P4_1: mem[0,0] <= 1536
 row_capacity_P4_2: rows_tot_P4_2 <= 48
 capacity_mem_par_P4_2: mem[1,0] <= 1536
Bounds
 mem_tot_P4_1 <= 6144
 normalized_mem_tot_P4_1 <= 1
 normalized_rows_tot_P4_1 <= 1
 mem_tot_P4_2 <= 6144
 normalized_mem_tot_P4_2 <= 1
 normalized_rows_tot_P4_2 <= 1
Binaries
 frac[0,0] frac[1,0]
General Constraints
 mem_overall_fixed_thr: max_mem_fixed_thr = MAX ( mem_tot_P4_1 ,
   mem_tot_P4_2 )
 rows_overall_fixed_thr: max_rows_fixed_thr = MAX ( rows_tot_P4_1 ,
   rows_tot_P4_2 )
End

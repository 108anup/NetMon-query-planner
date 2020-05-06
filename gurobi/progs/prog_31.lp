\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_others: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU_1 + rows_tot_Netronome_1
  others_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_others
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_rows_fixed_thr: Priority=80 Weight=1 AbsTol=0 RelTol=0
   rows_tot_P4_1
  rows_load_fixed_thr: Priority=70 Weight=1 AbsTol=0 RelTol=0
   max_rows_fixed_thr
  tot_mem_others: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU_1 + mem_tot_Netronome_1
  others_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_others
  tot_mem_fixed_thr: Priority=40 Weight=1 AbsTol=0 RelTol=0
   mem_tot_P4_1
  mem_load_fixed_thr: Priority=30 Weight=1 AbsTol=0 RelTol=0
   max_mem_fixed_thr
Subject To
 cov_0_0: frac[0,0] + frac[1,0] + frac[2,0] >= 1
 accuracy_2_0: - 131072 frac[2,0] + mem[2,0] = 0
 accuracy_1_0: - 131072 frac[1,0] + mem[1,0] = 0
 accuracy_0_0: - 67957.046875 frac[0,0] + mem[0,0] = 0
 mem_tot_CPU_1: - mem[0,0] + mem_tot_CPU_1 = 0
 R5: - 3.05176e-05 mem_tot_CPU_1 + normalized_mem_tot_CPU_1 = 0
 rows_tot_CPU_1: - 4 frac[0,0] + rows_tot_CPU_1 = 0
 R7: - 0.0833333333333333 rows_tot_CPU_1 + normalized_rows_tot_CPU_1 = 0
 mem_tot_Netronome_1: - mem[1,0] + mem_tot_Netronome_1 = 0
 R9: - 5e-06 mem_tot_Netronome_1 + normalized_mem_tot_Netronome_1 = 0
 rows_tot_Netronome_1: - 4 frac[1,0] + rows_tot_Netronome_1 = 0
 R11: - 0.0833333333333333 rows_tot_Netronome_1
   + normalized_rows_tot_Netronome_1 = 0
 mem_tot_P4_1: - mem[2,0] + mem_tot_P4_1 = 0
 R13: - 1.6276041666666666e-04 mem_tot_P4_1 + normalized_mem_tot_P4_1 = 0
 rows_tot_P4_1: - 4 frac[2,0] + rows_tot_P4_1 = 0
 R15: - 0.0208333333333333 rows_tot_P4_1 + normalized_rows_tot_P4_1 = 0
 row_capacity_CPU_1: rows_tot_CPU_1 <= 12
 row_capacity_Netronome_1: rows_tot_Netronome_1 <= 12
 row_capacity_P4_1: rows_tot_P4_1 <= 48
 capacity_mem_par_P4_1: mem[2,0] <= 2048
Bounds
 frac[0,0] <= 1
 mem_tot_CPU_1 <= 32768
 normalized_mem_tot_CPU_1 <= 1
 normalized_rows_tot_CPU_1 <= 1
 mem_tot_Netronome_1 <= 200000
 normalized_mem_tot_Netronome_1 <= 1
 normalized_rows_tot_Netronome_1 <= 1
 mem_tot_P4_1 <= 6144
 normalized_mem_tot_P4_1 <= 1
 normalized_rows_tot_P4_1 <= 1
Binaries
 frac[1,0] frac[2,0]
General Constraints
 mem_overall_fixed_thr: max_mem_fixed_thr = MAX ( mem_tot_P4_1 )
 mem_overall_others: max_mem_others = MAX ( normalized_mem_tot_CPU_1 ,
   normalized_mem_tot_Netronome_1 )
 rows_overall_fixed_thr: max_rows_fixed_thr = MAX ( rows_tot_P4_1 )
 rows_overall_others: max_rows_others = MAX ( normalized_rows_tot_CPU_1 ,
   normalized_rows_tot_Netronome_1 )
End

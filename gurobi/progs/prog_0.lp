\ Model UnivmonGreedy
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_mem_CPU: Priority=20 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU_1
  CPU_mem_load: Priority=15 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU
  tot_mem_P4: Priority=10 Weight=1 AbsTol=0 RelTol=0
   mem_tot_P4_1
  P4_mem_load: Priority=5 Weight=1 AbsTol=0 RelTol=0
   max_mem_P4
Subject To
 cov_0_0: frac[1,0] + frac[0,0] >= 1
 accuracy_1_0: - 5.09765625 frac[1,0] + mem[1,0] = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 mem_tot_CPU_1: - mem[0,0] + mem_tot_CPU_1 = 0
 rows_tot_CPU_1: - 3 frac[0,0] + rows_tot_CPU_1 = 0
 mem_tot_P4_1: - mem[1,0] + mem_tot_P4_1 = 0
 rows_tot_P4_1: - 3 frac[1,0] + rows_tot_P4_1 = 0
 row_capacity_CPU_1: rows_tot_CPU_1 <= 12
 row_capacity_P4_1: rows_tot_P4_1 <= 48
 tot_mem: - mem_tot_CPU_1 - mem_tot_P4_1 + tot_mem = 0
Bounds
 frac[1,0] <= 1
 frac[0,0] <= 1
 mem_tot_CPU_1 <= 32768
 mem_tot_P4_1 <= 576
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_P4_1 )
 mem_overall_CPU: max_mem_CPU = MAX ( mem_tot_CPU_1 )
End

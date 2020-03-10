\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU_1
  CPU_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU
  tot_rows_P4: Priority=80 Weight=1 AbsTol=0 RelTol=0
   rows_tot_P4_1
  P4_rows_load: Priority=70 Weight=1 AbsTol=0 RelTol=0
   max_rows_P4
  tot_mem_CPU: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU_1
  CPU_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU
  tot_mem_P4: Priority=40 Weight=1 AbsTol=0 RelTol=0
   mem_tot_P4_1
  P4_load_mem: Priority=30 Weight=1 AbsTol=0 RelTol=0
   max_mem_P4
Subject To
 cov_0_0: frac[1,0] + frac[0,0] >= 1
 accuracy_1_0: - 5.09765625 frac[1,0] + mem[1,0] = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 mem_tot_CPU_1: - mem[0,0] + mem_tot_CPU_1 = 0
 rows_tot_CPU_1: - 3 frac[0,0] + rows_tot_CPU_1 = 0
 mem_tot_P4_1: - mem[1,0] + mem_tot_P4_1 = 0
 rows_tot_P4_1: - 3 frac[1,0] + rows_tot_P4_1 = 0
 tot_mem: - mem_tot_CPU_1 - mem_tot_P4_1 + tot_mem = 0
 tot_rows: - rows_tot_CPU_1 - rows_tot_P4_1 + tot_rows = 0
Bounds
 mem_tot_CPU_1 <= 32768
 mem_tot_P4_1 <= 576
Binaries
 frac[1,0] frac[0,0]
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_P4_1 )
 mem_overall_CPU: max_mem_CPU = MAX ( mem_tot_CPU_1 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_P4_1 )
 rows_overall_CPU: max_rows_CPU = MAX ( rows_tot_CPU_1 )
End

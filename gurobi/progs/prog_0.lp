\ Model netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_cpu: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_cpu_1
  cpu_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_cpu
  tot_rows_p4: Priority=80 Weight=1 AbsTol=0 RelTol=0
   rows_tot_p4_1
  p4_rows_load: Priority=70 Weight=1 AbsTol=0 RelTol=0
   max_rows_p4
  tot_mem_cpu: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_cpu_1
  cpu_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_cpu
  tot_mem_p4: Priority=40 Weight=1 AbsTol=0 RelTol=0
   mem_tot_p4_1
  p4_load_mem: Priority=30 Weight=1 AbsTol=0 RelTol=0
   max_mem_p4
Subject To
 cov_0_0: frac[0,0] + frac[1,0] >= 1
 accuracy_0_sk_0[0]: - 5.09765625 frac[0,0] + mem[0,0] = 0
 accuracy_0_sk_0[1]: - 5.09765625 frac[1,0] + mem[1,0] = 0
 mem_tot_cpu_1: - mem[0,0] + mem_tot_cpu_1 = 0
 rows_tot_cpu_1: - 3 frac[0,0] + rows_tot_cpu_1 = 0
 mem_tot_p4_1: - mem[1,0] + mem_tot_p4_1 = 0
 rows_tot_p4_1: - 3 frac[1,0] + rows_tot_p4_1 = 0
 row_capacity_cpu_1: rows_tot_cpu_1 <= 12
 row_capacity_p4_1: rows_tot_p4_1 <= 48
 tot_mem: - mem_tot_cpu_1 - mem_tot_p4_1 + tot_mem = 0
 tot_rows: - rows_tot_cpu_1 - rows_tot_p4_1 + tot_rows = 0
Bounds
 frac[0,0] <= 1
 frac[1,0] <= 1
 mem_tot_cpu_1 <= 32768
 mem_tot_p4_1 <= 576
General Constraints
 mem_overall_p4: max_mem_p4 = MAX ( mem_tot_p4_1 )
 mem_overall_cpu: max_mem_cpu = MAX ( mem_tot_cpu_1 )
 rows_overall_p4: max_rows_p4 = MAX ( rows_tot_p4_1 )
 rows_overall_cpu: max_rows_cpu = MAX ( rows_tot_cpu_1 )
End

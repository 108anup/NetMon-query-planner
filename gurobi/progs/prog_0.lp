\ Model netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  max_mem: Priority=10 Weight=1 AbsTol=0 RelTol=0
   max_mem
  tot_mem: Priority=5 Weight=1 AbsTol=0 RelTol=0
   tot_mem
Subject To
 cov_0_0: frac[1,0] + frac[0,0] >= 1
 accuracy_1_0: - 5.09765625 frac[1,0] + mem[1,0] = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 mem_tot_cpu_1: - mem[0,0] + mem_tot_cpu_1 = 0
 rows_tot_cpu_1: - 3 frac[0,0] + rows_tot_cpu_1 = 0
 mem_tot_p4_1: - mem[1,0] + mem_tot_p4_1 = 0
 rows_tot_p4_1: - 3 frac[1,0] + rows_tot_p4_1 = 0
 tot_mem: - mem_tot_cpu_1 - mem_tot_p4_1 + tot_mem = 0
Bounds
 mem_tot_cpu_1 <= 32768
 mem_tot_p4_1 <= 576
Binaries
 frac[1,0] frac[0,0]
General Constraints
 mem_overall: max_mem = MAX ( mem_tot_cpu_1 , mem_tot_p4_1 )
End

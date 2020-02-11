\ Model netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_mem_cpu: Priority=20 Weight=1 AbsTol=0 RelTol=0
   mem_tot_cpu_1
  cpu_mem_load: Priority=15 Weight=1 AbsTol=0 RelTol=0
   max_mem_cpu
  tot_mem_p4: Priority=10 Weight=1 AbsTol=0 RelTol=0
   mem_tot_p4_1
  p4_mem_load: Priority=5 Weight=1 AbsTol=0 RelTol=0
   max_mem_p4
Subject To
 cov_0_0: frac[0,0] + frac[1,0] >= 1
 cov_0_1: frac[0,1] + frac[1,1] >= 1
 cov_0_2: frac[0,2] + frac[1,2] >= 1
 accuracy_0_sk_0[0]: - 5096.7890625 frac[0,0] + mem[0,0] = 0
 accuracy_0_sk_0[1]: - 5096.7890625 frac[1,0] + mem[1,0] = 0
 accuracy_1_sk_1[0]: - 50.9765625 frac[0,1] + mem[0,1] = 0
 accuracy_1_sk_1[1]: - 50.9765625 frac[1,1] + mem[1,1] = 0
 accuracy_2_sk_2[0]: - 50.9765625 frac[0,2] + mem[0,2] = 0
 accuracy_2_sk_2[1]: - 50.9765625 frac[1,2] + mem[1,2] = 0
 mem_tot_cpu_1: - mem[0,0] - mem[0,1] - mem[0,2] + mem_tot_cpu_1 = 0
 rows_tot_cpu_1: - 3 frac[0,0] - 3 frac[0,1] - 3 frac[0,2] + rows_tot_cpu_1
   = 0
 mem_tot_p4_1: - mem[1,0] - mem[1,1] - mem[1,2] + mem_tot_p4_1 = 0
 rows_tot_p4_1: - 3 frac[1,0] - 3 frac[1,1] - 3 frac[1,2] + rows_tot_p4_1
   = 0
 row_capacity_cpu_1: rows_tot_cpu_1 <= 12
 row_capacity_p4_1: rows_tot_p4_1 <= 48
 tot_mem: - mem_tot_cpu_1 - mem_tot_p4_1 + tot_mem = 0
Bounds
 frac[0,0] <= 1
 frac[0,1] <= 1
 frac[0,2] <= 1
 frac[1,0] <= 1
 frac[1,1] <= 1
 frac[1,2] <= 1
 mem_tot_cpu_1 <= 32768
 mem_tot_p4_1 <= 576
General Constraints
 mem_overall_p4: max_mem_p4 = MAX ( mem_tot_p4_1 )
 mem_overall_cpu: max_mem_cpu = MAX ( mem_tot_cpu_1 )
End

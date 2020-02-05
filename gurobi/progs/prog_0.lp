\ Model netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  cpu_rows_load: Priority=30 Weight=1 AbsTol=0 RelTol=0
   max_rows_cpu
  p4_rows_load: Priority=25 Weight=1 AbsTol=0 RelTol=0
   max_rows_p4
  rows_load: Priority=20 Weight=1 AbsTol=0 RelTol=0
   tot_rows
  cpu_load_mem: Priority=15 Weight=1 AbsTol=0 RelTol=0
   max_mem_cpu
  p4_load_mem: Priority=10 Weight=1 AbsTol=0 RelTol=0
   max_mem_p4
  mem_load: Priority=5 Weight=1 AbsTol=0 RelTol=0
   tot_mem
Subject To
 cov_0_1: frac[0,1] + frac[1,1] + frac[2,1] >= 1
 cov_0_2: frac[0,2] + frac[1,2] + frac[2,2] >= 1
 cov_1_2: frac[0,2] + frac[2,2] >= 0.9
 cov_1_0: frac[0,0] + frac[2,0] >= 1
 cov_2_1: frac[0,1] + frac[1,1] + frac[3,1] >= 0.8
 cov_2_2: frac[0,2] + frac[1,2] + frac[3,2] >= 0.5
 accuracy_0_sk_0[0]: - 0.10546875 frac[0,0] + mem[0,0] >= 0
 accuracy_0_sk_0[1]: - 0.10546875 frac[1,0] + mem[1,0] >= 0
 accuracy_0_sk_0[2]: - 0.10546875 frac[2,0] + mem[2,0] >= 0
 accuracy_0_sk_0[3]: - 0.10546875 frac[3,0] + mem[3,0] >= 0
 accuracy_1_sk_1[0]: - 25.48828125 frac[0,1] + mem[0,1] >= 0
 accuracy_1_sk_1[1]: - 25.48828125 frac[1,1] + mem[1,1] >= 0
 accuracy_1_sk_1[2]: - 25.48828125 frac[2,1] + mem[2,1] >= 0
 accuracy_1_sk_1[3]: - 25.48828125 frac[3,1] + mem[3,1] >= 0
 accuracy_2_sk_2[0]: - 0.078125 frac[0,2] + mem[0,2] >= 0
 accuracy_2_sk_2[1]: - 0.078125 frac[1,2] + mem[1,2] >= 0
 accuracy_2_sk_2[2]: - 0.078125 frac[2,2] + mem[2,2] >= 0
 accuracy_2_sk_2[3]: - 0.078125 frac[3,2] + mem[3,2] >= 0
 mem_tot_cpu_1: - mem[0,0] - mem[0,1] - mem[0,2] + mem_tot_cpu_1 = 0
 rows_tot_cpu_1: - 3 frac[0,0] - 3 frac[0,1] - 4 frac[0,2] + rows_tot_cpu_1
   = 0
 mem_tot_cpu_2: - mem[1,0] - mem[1,1] - mem[1,2] + mem_tot_cpu_2 = 0
 rows_tot_cpu_2: - 3 frac[1,0] - 3 frac[1,1] - 4 frac[1,2] + rows_tot_cpu_2
   = 0
 mem_tot_p4_1: - mem[2,0] - mem[2,1] - mem[2,2] + mem_tot_p4_1 = 0
 rows_tot_p4_1: - 3 frac[2,0] - 3 frac[2,1] - 4 frac[2,2] + rows_tot_p4_1
   = 0
 mem_tot_p4_2: - mem[3,0] - mem[3,1] - mem[3,2] + mem_tot_p4_2 = 0
 rows_tot_p4_2: - 3 frac[3,0] - 3 frac[3,1] - 4 frac[3,2] + rows_tot_p4_2
   = 0
 row_capacity_cpu_1: rows_tot_cpu_1 <= 12
 row_capacity_cpu_2: rows_tot_cpu_2 <= 12
 row_capacity_p4_1: rows_tot_p4_1 <= 48
 row_capacity_p4_2: rows_tot_p4_2 <= 48
 tot_mem: - mem_tot_cpu_1 - mem_tot_cpu_2 - mem_tot_p4_1 - mem_tot_p4_2
   + tot_mem = 0
 tot_rows: - rows_tot_cpu_1 - rows_tot_cpu_2 - rows_tot_p4_1
   - rows_tot_p4_2 + tot_rows = 0
Bounds
 frac[0,0] <= 1
 frac[0,1] <= 1
 frac[0,2] <= 1
 frac[1,0] <= 1
 frac[1,1] <= 1
 frac[1,2] <= 1
 frac[2,0] <= 1
 frac[2,1] <= 1
 frac[2,2] <= 1
 frac[3,0] <= 1
 frac[3,1] <= 1
 frac[3,2] <= 1
 mem_tot_cpu_1 <= 32768
 mem_tot_cpu_2 <= 32768
 mem_tot_p4_1 <= 576
 mem_tot_p4_2 <= 576
General Constraints
 mem_overall_p4: max_mem_p4 = MAX ( mem_tot_p4_1 , mem_tot_p4_2 )
 mem_overall_cpu: max_mem_cpu = MAX ( mem_tot_cpu_1 , mem_tot_cpu_2 )
 rows_overall_p4: max_rows_p4 = MAX ( rows_tot_p4_1 , rows_tot_p4_2 )
 rows_overall_cpu: max_rows_cpu = MAX ( rows_tot_cpu_1 , rows_tot_cpu_2 )
End

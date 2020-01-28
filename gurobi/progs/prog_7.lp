\ Model netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0.05
   ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0.05
   res_overall
Subject To
 cov_0: frac[0,0] + frac[1,0] + frac[2,0] = 1
 cov_1: frac[0,1] + frac[1,1] + frac[2,1] = 1
 cov_2: frac[0,2] + frac[1,2] + frac[2,2] = 1
 cov_3: frac[0,3] + frac[1,3] + frac[2,3] = 1
 accuracy_(1,_sk_1)[0]: - 88.48828125 frac[0,0] + mem[0,0] >= 0
 accuracy_(1,_sk_1)[1]: - 88.48828125 frac[1,0] + mem[1,0] >= 0
 accuracy_(1,_sk_1)[2]: - 88.48828125 frac[2,0] + mem[2,0] >= 0
 accuracy_(2,_sk_1)[0]: - 88.48828125 frac[0,1] + mem[0,1] >= 0
 accuracy_(2,_sk_1)[1]: - 88.48828125 frac[1,1] + mem[1,1] >= 0
 accuracy_(2,_sk_1)[2]: - 88.48828125 frac[2,1] + mem[2,1] >= 0
 accuracy_(3,_sk_1)[0]: - 88.48828125 frac[0,2] + mem[0,2] >= 0
 accuracy_(3,_sk_1)[1]: - 88.48828125 frac[1,2] + mem[1,2] >= 0
 accuracy_(3,_sk_1)[2]: - 88.48828125 frac[2,2] + mem[2,2] >= 0
 accuracy_(4,_sk_1)[0]: - 88.48828125 frac[0,3] + mem[0,3] >= 0
 accuracy_(4,_sk_1)[1]: - 88.48828125 frac[1,3] + mem[1,3] >= 0
 accuracy_(4,_sk_1)[2]: - 88.48828125 frac[2,3] + mem[2,3] >= 0
 rows_tot_p4_1: - frac[0,0] - frac[0,1] - frac[0,2] - frac[0,3]
   + rows_tot_p4_1 = 0
 mem_tot_p4_1: - mem[0,0] - mem[0,1] - mem[0,2] - mem[0,3] + mem_tot_p4_1
   = 0
 capacity_mem_par_p4_1: mem[0,0] <= 48
 capacity_mem_par_p4_1: mem[0,1] <= 48
 capacity_mem_par_p4_1: mem[0,2] <= 48
 capacity_mem_par_p4_1: mem[0,3] <= 48
 ns_p4_1: ns_p4_1 = 6.756756756756757
 rows_tot_p4_2: - frac[1,0] - frac[1,1] - frac[1,2] - frac[1,3]
   + rows_tot_p4_2 = 0
 mem_tot_p4_2: - mem[1,0] - mem[1,1] - mem[1,2] - mem[1,3] + mem_tot_p4_2
   = 0
 capacity_mem_par_p4_2: mem[1,0] <= 48
 capacity_mem_par_p4_2: mem[1,1] <= 48
 capacity_mem_par_p4_2: mem[1,2] <= 48
 capacity_mem_par_p4_2: mem[1,3] <= 48
 ns_p4_2: ns_p4_2 = 6.756756756756757
 rows_tot_p4_3: - frac[2,0] - frac[2,1] - frac[2,2] - frac[2,3]
   + rows_tot_p4_3 = 0
 mem_tot_p4_3: - mem[2,0] - mem[2,1] - mem[2,2] - mem[2,3] + mem_tot_p4_3
   = 0
 capacity_mem_par_p4_3: mem[2,0] <= 48
 capacity_mem_par_p4_3: mem[2,1] <= 48
 capacity_mem_par_p4_3: mem[2,2] <= 48
 capacity_mem_par_p4_3: mem[2,3] <= 48
 ns_p4_3: ns_p4_3 = 6.756756756756757
 res: - 0.25 rows_tot_p4_1 - 0.0208333333333333 mem_tot_p4_1
   - 0.25 rows_tot_p4_2 - 0.0208333333333333 mem_tot_p4_2
   - 0.25 rows_tot_p4_3 - 0.0208333333333333 mem_tot_p4_3 + res_overall
   = 0
Bounds
 frac[0,0] <= 1
 frac[0,1] <= 1
 frac[0,2] <= 1
 frac[0,3] <= 1
 frac[1,0] <= 1
 frac[1,1] <= 1
 frac[1,2] <= 1
 frac[1,3] <= 1
 frac[2,0] <= 1
 frac[2,1] <= 1
 frac[2,2] <= 1
 frac[2,3] <= 1
 rows_tot_p4_1 <= 12
 mem_tot_p4_1 <= 576
 rows_tot_p4_2 <= 12
 mem_tot_p4_2 <= 576
 rows_tot_p4_3 <= 12
 mem_tot_p4_3 <= 576
General Constraints
 ns_overall: ns = MAX ( ns_p4_1 , ns_p4_2 , ns_p4_3 )
End

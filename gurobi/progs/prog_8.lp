\ Model netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0.1
   ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0.05
   res
Subject To
 cov_0_0: frac[0,0] + frac[1,0] >= 1
 cov_0_1: frac[0,1] + frac[1,1] >= 1
 cov_0_2: frac[0,2] + frac[1,2] >= 1
 accuracy_0_sk_0[0]: - 33.98046875 frac[0,0] + mem[0,0] = 0
 accuracy_0_sk_0[1]: - 33.98046875 frac[1,0] + mem[1,0] = 0
 accuracy_1_sk_0[0]: - 33.98046875 frac[0,1] + mem[0,1] = 0
 accuracy_1_sk_0[1]: - 33.98046875 frac[1,1] + mem[1,1] = 0
 accuracy_2_sk_0[0]: - 33.98046875 frac[0,2] + mem[0,2] = 0
 accuracy_2_sk_0[1]: - 33.98046875 frac[1,2] + mem[1,2] = 0
 mem_tot_p4_1: - mem[0,0] - mem[0,1] - mem[0,2] + mem_tot_p4_1 = 0
 rows_tot_p4_1: - frac[0,0] - frac[0,1] - frac[0,2] + rows_tot_p4_1 = 0
 mem_tot_p4_2: - mem[1,0] - mem[1,1] - mem[1,2] + mem_tot_p4_2 = 0
 rows_tot_p4_2: - frac[1,0] - frac[1,1] - frac[1,2] + rows_tot_p4_2 = 0
 row_capacity_p4_1: rows_tot_p4_1 <= 48
 row_capacity_p4_2: rows_tot_p4_2 <= 48
 ns_p4_1: ns_p4_1 = 6.756756756756757
 ns_p4_2: ns_p4_2 = 6.756756756756757
 res_acc: 0.0208333333333333 mem_tot_p4_1 + 0.25 rows_tot_p4_1
   + 0.0208333333333333 mem_tot_p4_2 + 0.25 rows_tot_p4_2 - res = 0
Bounds
 frac[0,0] <= 1
 frac[0,1] <= 1
 frac[0,2] <= 1
 frac[1,0] <= 1
 frac[1,1] <= 1
 frac[1,2] <= 1
 mem_tot_p4_1 <= 576
 mem_tot_p4_2 <= 576
General Constraints
 ns_overall: ns = MAX ( ns_p4_1 , ns_p4_2 )
End

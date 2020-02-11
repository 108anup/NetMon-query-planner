\ Model netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0
   ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0
   res
Subject To
 cov_0_0: frac[0,0] + frac[1,0] >= 1
 accuracy_0_sk_0[0]: - 101.94140625 frac[0,0] + mem[0,0] >= 0
 accuracy_0_sk_0[1]: - 101.94140625 frac[1,0] + mem[1,0] >= 0
 mem_tot_p4_1: - mem[0,0] + mem_tot_p4_1 = 0
 rows_tot_p4_1: - 3 frac[0,0] + rows_tot_p4_1 = 0
 mem_tot_p4_2: - mem[1,0] + mem_tot_p4_2 = 0
 rows_tot_p4_2: - 3 frac[1,0] + rows_tot_p4_2 = 0
 row_capacity_p4_1: rows_tot_p4_1 <= 48
 row_capacity_p4_2: rows_tot_p4_2 <= 48
 ns_p4_1: ns_p4_1 = 6.756756756756757
 ns_p4_2: ns_p4_2 = 6.756756756756757
 res_acc: 0.0208333333333333 mem_tot_p4_1 + 0.25 rows_tot_p4_1
   + 0.0208333333333333 mem_tot_p4_2 + 0.25 rows_tot_p4_2 - res = 0
Bounds
 mem_tot_p4_1 <= 576
 mem_tot_p4_2 <= 576
Binaries
 frac[0,0] frac[1,0]
General Constraints
 ns_overall: ns = MAX ( ns_p4_1 , ns_p4_2 )
End

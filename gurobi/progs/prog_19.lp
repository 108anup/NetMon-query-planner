\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_P4 + 0 max_rows_P4 + res
Subject To
 cov_0_0: frac[0,0] >= 2.5e-05
 cov_1_0: frac[1,0] + frac[0,0] >= 2.5e-05
 cov_2_0: frac[2,0] + frac[1,0] >= 2.5e-05
 cov_3_0: frac[3,0] + frac[2,0] >= 2.5e-05
 cov_4_0: frac[3,0] >= 2.5e-05
 accuracy_3_0: - 5096.7890625 frac[3,0] + mem[3,0] = 0
 accuracy_2_0: - 5096.7890625 frac[2,0] + mem[2,0] = 0
 accuracy_1_0: - 5096.7890625 frac[1,0] + mem[1,0] = 0
 accuracy_0_0: - 5096.7890625 frac[0,0] + mem[0,0] = 0
 mem_tot_P4_16: - mem[0,0] + mem_tot_P4_16 = 0
 R10: - 0.00173611111111111 mem_tot_P4_16 + normalized_mem_tot_P4_16 = 0
 rows_tot_P4_16: - 3 frac[0,0] + rows_tot_P4_16 = 0
 R12: - 0.0208333333333333 rows_tot_P4_16 + normalized_rows_tot_P4_16 = 0
 mem_tot_P4_17: - mem[1,0] + mem_tot_P4_17 = 0
 R14: - 0.00173611111111111 mem_tot_P4_17 + normalized_mem_tot_P4_17 = 0
 rows_tot_P4_17: - 3 frac[1,0] + rows_tot_P4_17 = 0
 R16: - 0.0208333333333333 rows_tot_P4_17 + normalized_rows_tot_P4_17 = 0
 mem_tot_P4_18: - mem[2,0] + mem_tot_P4_18 = 0
 R18: - 0.00173611111111111 mem_tot_P4_18 + normalized_mem_tot_P4_18 = 0
 rows_tot_P4_18: - 3 frac[2,0] + rows_tot_P4_18 = 0
 R20: - 0.0208333333333333 rows_tot_P4_18 + normalized_rows_tot_P4_18 = 0
 mem_tot_P4_19: - mem[3,0] + mem_tot_P4_19 = 0
 R22: - 0.00173611111111111 mem_tot_P4_19 + normalized_mem_tot_P4_19 = 0
 rows_tot_P4_19: - 3 frac[3,0] + rows_tot_P4_19 = 0
 R24: - 0.0208333333333333 rows_tot_P4_19 + normalized_rows_tot_P4_19 = 0
 row_capacity_P4_16: rows_tot_P4_16 <= 48
 row_capacity_P4_17: rows_tot_P4_17 <= 48
 row_capacity_P4_18: rows_tot_P4_18 <= 48
 row_capacity_P4_19: rows_tot_P4_19 <= 48
 ns_P4_16: ns_P4_16 = 6.756756756756757
 max_ns_P4_16: ns_P4_16 <= 6.756756756756757
 ns_P4_17: ns_P4_17 = 6.756756756756757
 max_ns_P4_17: ns_P4_17 <= 6.756756756756757
 ns_P4_18: ns_P4_18 = 6.756756756756757
 max_ns_P4_18: ns_P4_18 <= 6.756756756756757
 ns_P4_19: ns_P4_19 = 6.756756756756757
 max_ns_P4_19: ns_P4_19 <= 6.756756756756757
 res_acc: 0.0208333333333333 mem_tot_P4_16 + 0.25 rows_tot_P4_16
   + 0.0208333333333333 mem_tot_P4_17 + 0.25 rows_tot_P4_17
   + 0.0208333333333333 mem_tot_P4_18 + 0.25 rows_tot_P4_18
   + 0.0208333333333333 mem_tot_P4_19 + 0.25 rows_tot_P4_19 - res = 0
Bounds
 frac[3,0] <= 1
 frac[2,0] <= 1
 frac[1,0] <= 1
 frac[0,0] <= 1
 mem_tot_P4_16 <= 576
 normalized_mem_tot_P4_16 <= 1
 normalized_rows_tot_P4_16 <= 1
 mem_tot_P4_17 <= 576
 normalized_mem_tot_P4_17 <= 1
 normalized_rows_tot_P4_17 <= 1
 mem_tot_P4_18 <= 576
 normalized_mem_tot_P4_18 <= 1
 normalized_rows_tot_P4_18 <= 1
 mem_tot_P4_19 <= 576
 normalized_mem_tot_P4_19 <= 1
 normalized_rows_tot_P4_19 <= 1
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_P4_16 , mem_tot_P4_17 ,
   mem_tot_P4_18 , mem_tot_P4_19 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_P4_16 , rows_tot_P4_17 ,
   rows_tot_P4_18 , rows_tot_P4_19 )
End

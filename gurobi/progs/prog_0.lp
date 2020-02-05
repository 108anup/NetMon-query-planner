\ Model netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  cpu_rows_load: Priority=20 Weight=1 AbsTol=0 RelTol=0
   max_rows_cpu
  p4_rows_load: Priority=15 Weight=1 AbsTol=0 RelTol=0
   max_rows_p4
  cpu_load_mem: Priority=10 Weight=1 AbsTol=0 RelTol=0
   max_mem_cpu
  p4_load_mem: Priority=5 Weight=1 AbsTol=0 RelTol=0
   max_mem_p4
Subject To
 cov_0_3: frac[0,3] + frac[1,3] + frac[2,3] >= 1
 cov_0_4: frac[0,4] + frac[1,4] + frac[2,4] >= 1
 cov_0_5: frac[0,5] + frac[1,5] + frac[2,5] >= 1
 cov_0_6: frac[0,6] + frac[1,6] + frac[2,6] >= 1
 cov_0_7: frac[0,7] + frac[1,7] + frac[2,7] >= 1
 cov_0_8: frac[0,8] + frac[1,8] + frac[2,8] >= 1
 cov_0_9: frac[0,9] + frac[1,9] + frac[2,9] >= 1
 cov_1_6: frac[0,6] + frac[2,6] >= 0.9
 cov_1_7: frac[0,7] + frac[2,7] >= 0.9
 cov_1_8: frac[0,8] + frac[2,8] >= 0.9
 cov_1_9: frac[0,9] + frac[2,9] >= 0.9
 cov_1_0: frac[0,0] + frac[2,0] >= 1
 cov_1_1: frac[0,1] + frac[2,1] >= 1
 cov_1_2: frac[0,2] + frac[2,2] >= 1
 cov_2_3: frac[0,3] + frac[1,3] + frac[3,3] >= 0.8
 cov_2_4: frac[0,4] + frac[1,4] + frac[3,4] >= 0.8
 cov_2_5: frac[0,5] + frac[1,5] + frac[3,5] >= 0.8
 cov_2_6: frac[0,6] + frac[1,6] + frac[3,6] >= 0.5
 cov_2_7: frac[0,7] + frac[1,7] + frac[3,7] >= 0.5
 cov_2_8: frac[0,8] + frac[1,8] + frac[3,8] >= 0.5
 cov_2_9: frac[0,9] + frac[1,9] + frac[3,9] >= 0.5
 accuracy_0_sk_0[0]: - 0.03515625 frac[0,0] + mem[0,0] >= 0
 accuracy_0_sk_0[1]: - 0.03515625 frac[1,0] + mem[1,0] >= 0
 accuracy_0_sk_0[2]: - 0.03515625 frac[2,0] + mem[2,0] >= 0
 accuracy_0_sk_0[3]: - 0.03515625 frac[3,0] + mem[3,0] >= 0
 accuracy_1_sk_0[0]: - 0.03515625 frac[0,1] + mem[0,1] >= 0
 accuracy_1_sk_0[1]: - 0.03515625 frac[1,1] + mem[1,1] >= 0
 accuracy_1_sk_0[2]: - 0.03515625 frac[2,1] + mem[2,1] >= 0
 accuracy_1_sk_0[3]: - 0.03515625 frac[3,1] + mem[3,1] >= 0
 accuracy_2_sk_0[0]: - 0.03515625 frac[0,2] + mem[0,2] >= 0
 accuracy_2_sk_0[1]: - 0.03515625 frac[1,2] + mem[1,2] >= 0
 accuracy_2_sk_0[2]: - 0.03515625 frac[2,2] + mem[2,2] >= 0
 accuracy_2_sk_0[3]: - 0.03515625 frac[3,2] + mem[3,2] >= 0
 accuracy_3_sk_1[0]: - 8.49609375 frac[0,3] + mem[0,3] >= 0
 accuracy_3_sk_1[1]: - 8.49609375 frac[1,3] + mem[1,3] >= 0
 accuracy_3_sk_1[2]: - 8.49609375 frac[2,3] + mem[2,3] >= 0
 accuracy_3_sk_1[3]: - 8.49609375 frac[3,3] + mem[3,3] >= 0
 accuracy_4_sk_1[0]: - 8.49609375 frac[0,4] + mem[0,4] >= 0
 accuracy_4_sk_1[1]: - 8.49609375 frac[1,4] + mem[1,4] >= 0
 accuracy_4_sk_1[2]: - 8.49609375 frac[2,4] + mem[2,4] >= 0
 accuracy_4_sk_1[3]: - 8.49609375 frac[3,4] + mem[3,4] >= 0
 accuracy_5_sk_1[0]: - 8.49609375 frac[0,5] + mem[0,5] >= 0
 accuracy_5_sk_1[1]: - 8.49609375 frac[1,5] + mem[1,5] >= 0
 accuracy_5_sk_1[2]: - 8.49609375 frac[2,5] + mem[2,5] >= 0
 accuracy_5_sk_1[3]: - 8.49609375 frac[3,5] + mem[3,5] >= 0
 accuracy_6_sk_2[0]: - 0.01953125 frac[0,6] + mem[0,6] >= 0
 accuracy_6_sk_2[1]: - 0.01953125 frac[1,6] + mem[1,6] >= 0
 accuracy_6_sk_2[2]: - 0.01953125 frac[2,6] + mem[2,6] >= 0
 accuracy_6_sk_2[3]: - 0.01953125 frac[3,6] + mem[3,6] >= 0
 accuracy_7_sk_2[0]: - 0.01953125 frac[0,7] + mem[0,7] >= 0
 accuracy_7_sk_2[1]: - 0.01953125 frac[1,7] + mem[1,7] >= 0
 accuracy_7_sk_2[2]: - 0.01953125 frac[2,7] + mem[2,7] >= 0
 accuracy_7_sk_2[3]: - 0.01953125 frac[3,7] + mem[3,7] >= 0
 accuracy_8_sk_2[0]: - 0.01953125 frac[0,8] + mem[0,8] >= 0
 accuracy_8_sk_2[1]: - 0.01953125 frac[1,8] + mem[1,8] >= 0
 accuracy_8_sk_2[2]: - 0.01953125 frac[2,8] + mem[2,8] >= 0
 accuracy_8_sk_2[3]: - 0.01953125 frac[3,8] + mem[3,8] >= 0
 accuracy_9_sk_2[0]: - 0.01953125 frac[0,9] + mem[0,9] >= 0
 accuracy_9_sk_2[1]: - 0.01953125 frac[1,9] + mem[1,9] >= 0
 accuracy_9_sk_2[2]: - 0.01953125 frac[2,9] + mem[2,9] >= 0
 accuracy_9_sk_2[3]: - 0.01953125 frac[3,9] + mem[3,9] >= 0
 mem_tot_cpu_1: - mem[0,0] - mem[0,1] - mem[0,2] - mem[0,3] - mem[0,4]
   - mem[0,5] - mem[0,6] - mem[0,7] - mem[0,8] - mem[0,9] + mem_tot_cpu_1
   = 0
 mem_tot_cpu_2: - mem[1,0] - mem[1,1] - mem[1,2] - mem[1,3] - mem[1,4]
   - mem[1,5] - mem[1,6] - mem[1,7] - mem[1,8] - mem[1,9] + mem_tot_cpu_2
   = 0
 mem_tot_p4_1: - mem[2,0] - mem[2,1] - mem[2,2] - mem[2,3] - mem[2,4]
   - mem[2,5] - mem[2,6] - mem[2,7] - mem[2,8] - mem[2,9] + mem_tot_p4_1
   = 0
 mem_tot_p4_2: - mem[3,0] - mem[3,1] - mem[3,2] - mem[3,3] - mem[3,4]
   - mem[3,5] - mem[3,6] - mem[3,7] - mem[3,8] - mem[3,9] + mem_tot_p4_2
   = 0
 rows_tot_cpu_1: - frac[0,0] - frac[0,1] - frac[0,2] - frac[0,3]
   - frac[0,4] - frac[0,5] - frac[0,6] - frac[0,7] - frac[0,8] - frac[0,9]
   + rows_tot_cpu_1 = 0
 rows_tot_cpu_2: - frac[1,0] - frac[1,1] - frac[1,2] - frac[1,3]
   - frac[1,4] - frac[1,5] - frac[1,6] - frac[1,7] - frac[1,8] - frac[1,9]
   + rows_tot_cpu_2 = 0
 rows_tot_p4_1: - frac[2,0] - frac[2,1] - frac[2,2] - frac[2,3] - frac[2,4]
   - frac[2,5] - frac[2,6] - frac[2,7] - frac[2,8] - frac[2,9]
   + rows_tot_p4_1 = 0
 rows_tot_p4_2: - frac[3,0] - frac[3,1] - frac[3,2] - frac[3,3] - frac[3,4]
   - frac[3,5] - frac[3,6] - frac[3,7] - frac[3,8] - frac[3,9]
   + rows_tot_p4_2 = 0
Bounds
 frac[0,0] <= 1
 frac[0,1] <= 1
 frac[0,2] <= 1
 frac[0,3] <= 1
 frac[0,4] <= 1
 frac[0,5] <= 1
 frac[0,6] <= 1
 frac[0,7] <= 1
 frac[0,8] <= 1
 frac[0,9] <= 1
 frac[1,0] <= 1
 frac[1,1] <= 1
 frac[1,2] <= 1
 frac[1,3] <= 1
 frac[1,4] <= 1
 frac[1,5] <= 1
 frac[1,6] <= 1
 frac[1,7] <= 1
 frac[1,8] <= 1
 frac[1,9] <= 1
 frac[2,0] <= 1
 frac[2,1] <= 1
 frac[2,2] <= 1
 frac[2,3] <= 1
 frac[2,4] <= 1
 frac[2,5] <= 1
 frac[2,6] <= 1
 frac[2,7] <= 1
 frac[2,8] <= 1
 frac[2,9] <= 1
 frac[3,0] <= 1
 frac[3,1] <= 1
 frac[3,2] <= 1
 frac[3,3] <= 1
 frac[3,4] <= 1
 frac[3,5] <= 1
 frac[3,6] <= 1
 frac[3,7] <= 1
 frac[3,8] <= 1
 frac[3,9] <= 1
 mem_tot_cpu_1 <= 32768
 mem_tot_cpu_2 <= 32768
 mem_tot_p4_1 <= 576
 mem_tot_p4_2 <= 576
 rows_tot_cpu_1 <= 12
 rows_tot_cpu_2 <= 12
 rows_tot_p4_1 <= 48
 rows_tot_p4_2 <= 48
General Constraints
 mem_overall_p4: max_mem_p4 = MAX ( mem_tot_p4_1 , mem_tot_p4_2 )
 mem_overall_cpu: max_mem_cpu = MAX ( mem_tot_cpu_1 , mem_tot_cpu_2 )
 rows_overall_p4: max_rows_p4 = MAX ( rows_tot_p4_1 , rows_tot_p4_2 )
 rows_overall_cpu: max_rows_cpu = MAX ( rows_tot_cpu_1 , rows_tot_cpu_2 )
End

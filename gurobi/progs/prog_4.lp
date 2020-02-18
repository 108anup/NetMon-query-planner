\ Model netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0.1
   0 m_access_time_cpu_1 + 0 ns_sketch_cpu_1 + 0 ns_dpdk_cpu_1 + 0 ns_cpu_1
   + ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0.05
   res
Subject To
 cov_0_0: frac[0,0] + frac[1,0] >= 1
 cov_0_1: frac[0,1] + frac[1,1] >= 1
 cov_0_2: frac[0,2] + frac[1,2] >= 1
 cov_0_3: frac[0,3] + frac[1,3] >= 1
 cov_0_4: frac[0,4] + frac[1,4] >= 1
 cov_0_5: frac[0,5] + frac[1,5] >= 1
 cov_0_6: frac[0,6] + frac[1,6] >= 1
 cov_0_7: frac[0,7] + frac[1,7] >= 1
 cov_0_8: frac[0,8] + frac[1,8] >= 1
 cov_0_9: frac[0,9] + frac[1,9] >= 1
 accuracy_0_sk_0[0]: - 0.03515625 frac[0,0] + mem[0,0] = 0
 accuracy_0_sk_0[1]: - 0.03515625 frac[1,0] + mem[1,0] = 0
 accuracy_1_sk_0[0]: - 0.03515625 frac[0,1] + mem[0,1] = 0
 accuracy_1_sk_0[1]: - 0.03515625 frac[1,1] + mem[1,1] = 0
 accuracy_2_sk_0[0]: - 0.03515625 frac[0,2] + mem[0,2] = 0
 accuracy_2_sk_0[1]: - 0.03515625 frac[1,2] + mem[1,2] = 0
 accuracy_3_sk_1[0]: - 8.49609375 frac[0,3] + mem[0,3] = 0
 accuracy_3_sk_1[1]: - 8.49609375 frac[1,3] + mem[1,3] = 0
 accuracy_4_sk_1[0]: - 8.49609375 frac[0,4] + mem[0,4] = 0
 accuracy_4_sk_1[1]: - 8.49609375 frac[1,4] + mem[1,4] = 0
 accuracy_5_sk_1[0]: - 8.49609375 frac[0,5] + mem[0,5] = 0
 accuracy_5_sk_1[1]: - 8.49609375 frac[1,5] + mem[1,5] = 0
 accuracy_6_sk_2[0]: - 0.01953125 frac[0,6] + mem[0,6] = 0
 accuracy_6_sk_2[1]: - 0.01953125 frac[1,6] + mem[1,6] = 0
 accuracy_7_sk_2[0]: - 0.01953125 frac[0,7] + mem[0,7] = 0
 accuracy_7_sk_2[1]: - 0.01953125 frac[1,7] + mem[1,7] = 0
 accuracy_8_sk_2[0]: - 0.01953125 frac[0,8] + mem[0,8] = 0
 accuracy_8_sk_2[1]: - 0.01953125 frac[1,8] + mem[1,8] = 0
 accuracy_9_sk_2[0]: - 0.01953125 frac[0,9] + mem[0,9] = 0
 accuracy_9_sk_2[1]: - 0.01953125 frac[1,9] + mem[1,9] = 0
 mem_tot_cpu_1: - mem[0,0] - mem[0,1] - mem[0,2] - mem[0,3] - mem[0,4]
   - mem[0,5] - mem[0,6] - mem[0,7] - mem[0,8] - mem[0,9] + mem_tot_cpu_1
   = 0
 rows_tot_cpu_1: - frac[0,0] - frac[0,1] - frac[0,2] - frac[0,3]
   - frac[0,4] - frac[0,5] - frac[0,6] - frac[0,7] - frac[0,8] - frac[0,9]
   + rows_tot_cpu_1 = 0
 mem_tot_p4_1: - mem[1,0] - mem[1,1] - mem[1,2] - mem[1,3] - mem[1,4]
   - mem[1,5] - mem[1,6] - mem[1,7] - mem[1,8] - mem[1,9] + mem_tot_p4_1
   = 0
 rows_tot_p4_1: - frac[1,0] - frac[1,1] - frac[1,2] - frac[1,3] - frac[1,4]
   - frac[1,5] - frac[1,6] - frac[1,7] - frac[1,8] - frac[1,9]
   + rows_tot_p4_1 = 0
 row_capacity_cpu_1: rows_tot_cpu_1 <= 12
 row_capacity_p4_1: rows_tot_p4_1 <= 48
 ns_single_cpu_1: 3.5 rows_tot_cpu_1 - ns_single_cpu_1 + pdt_m_rows_cpu_1
   = 0
 capacity_cores_cpu_1: cores_sketch_cpu_1 + cores_dpdk_cpu_1 <= 7
 ns_sketch_cpu_1: - ns_single_cpu_1 + pdt_nsk_c_cpu_1 = 0
 ns_dpdk_cpu_1: 7.142857142857143 cores_dpdk_cpu_1 - pdt_nsc_dpdk_cpu_1
   = -21.42857142857143
 ns_p4_1: ns_p4_1 = 6.756756756756757
 res_acc: 1.220703125e-04 mem_tot_cpu_1 + 0.0208333333333333 mem_tot_p4_1
   + 0.25 rows_tot_p4_1 + 10 cores_sketch_cpu_1 + 10 cores_dpdk_cpu_1 - res
   = 0
 pdt_m_rows_cpu_1: pdt_m_rows_cpu_1 + [
   - rows_tot_cpu_1 * m_access_time_cpu_1 ] = 0
 pdt_nsk_c_cpu_1: pdt_nsk_c_cpu_1 + [
   - cores_sketch_cpu_1 * ns_sketch_cpu_1 ] = 0
 pdt_nsc_dpdk_cpu_1: pdt_nsc_dpdk_cpu_1 + [
   - cores_dpdk_cpu_1 * ns_dpdk_cpu_1 ] = 0
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
 mem_tot_cpu_1 <= 32768
 mem_tot_p4_1 <= 576
 cores_sketch_cpu_1 <= 7
 1 <= cores_dpdk_cpu_1 <= 7
Generals
 cores_sketch_cpu_1 cores_dpdk_cpu_1
General Constraints
 mem_access_time_cpu_1: m_access_time_cpu_1 = PWL ( mem_tot_cpu_1 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_1: ns_cpu_1 = MAX ( ns_dpdk_cpu_1 , ns_sketch_cpu_1 )
 ns_overall: ns = MAX ( ns_cpu_1 , ns_p4_1 )
End

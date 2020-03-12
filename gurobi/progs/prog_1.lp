\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 m_access_time_CPU_1 + 0 ns_sketch_CPU_1 + 0 ns_dpdk_CPU_1 + 0 ns_CPU_1
   + ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0
   res
Subject To
 cov_0_0: frac[0,0] + frac[1,0] >= 1
 cov_0_1: frac[0,1] + frac[1,1] >= 1
 cov_0_2: frac[1,2] + frac[0,2] >= 1
 cov_0_3: frac[1,3] + frac[0,3] >= 1
 cov_0_4: frac[1,4] + frac[0,4] >= 1
 cov_0_5: frac[1,5] + frac[0,5] >= 1
 cov_0_6: frac[0,6] + frac[1,6] >= 1
 cov_0_7: frac[0,7] + frac[1,7] >= 1
 cov_0_8: frac[0,8] + frac[1,8] >= 1
 accuracy_0_1: - 1698.9296875 frac[0,1] + mem[0,1] = 0
 accuracy_1_2: - 1698.9296875 frac[1,2] + mem[1,2] = 0
 accuracy_0_0: - 1698.9296875 frac[0,0] + mem[0,0] = 0
 accuracy_1_3: - 16.9921875 frac[1,3] + mem[1,3] = 0
 accuracy_1_4: - 16.9921875 frac[1,4] + mem[1,4] = 0
 accuracy_0_7: - 16.9921875 frac[0,7] + mem[0,7] = 0
 accuracy_0_8: - 16.9921875 frac[0,8] + mem[0,8] = 0
 accuracy_1_5: - 16.9921875 frac[1,5] + mem[1,5] = 0
 accuracy_0_6: - 16.9921875 frac[0,6] + mem[0,6] = 0
 accuracy_1_1: - 1698.9296875 frac[1,1] + mem[1,1] = 0
 accuracy_0_5: - 16.9921875 frac[0,5] + mem[0,5] = 0
 accuracy_1_6: - 16.9921875 frac[1,6] + mem[1,6] = 0
 accuracy_1_8: - 16.9921875 frac[1,8] + mem[1,8] = 0
 accuracy_0_4: - 16.9921875 frac[0,4] + mem[0,4] = 0
 accuracy_1_0: - 1698.9296875 frac[1,0] + mem[1,0] = 0
 accuracy_1_7: - 16.9921875 frac[1,7] + mem[1,7] = 0
 accuracy_0_2: - 1698.9296875 frac[0,2] + mem[0,2] = 0
 accuracy_0_3: - 16.9921875 frac[0,3] + mem[0,3] = 0
 mem_tot_CPU_1: - mem[0,1] - mem[0,0] - mem[0,7] - mem[0,8] - mem[0,6]
   - mem[0,5] - mem[0,4] - mem[0,2] - mem[0,3] + mem_tot_CPU_1 = 0
 rows_tot_CPU_1: - frac[0,1] - frac[0,0] - frac[0,7] - frac[0,8]
   - frac[0,6] - frac[0,5] - frac[0,4] - frac[0,2] - frac[0,3]
   + rows_tot_CPU_1 = 0
 mem_tot_P4_1: - mem[1,2] - mem[1,3] - mem[1,4] - mem[1,5] - mem[1,1]
   - mem[1,6] - mem[1,8] - mem[1,0] - mem[1,7] + mem_tot_P4_1 = 0
 rows_tot_P4_1: - frac[1,2] - frac[1,3] - frac[1,4] - frac[1,5] - frac[1,1]
   - frac[1,6] - frac[1,8] - frac[1,0] - frac[1,7] + rows_tot_P4_1 = 0
 ns_single_CPU_1: 3.5 rows_tot_CPU_1 - ns_single_CPU_1 + pdt_m_rows_CPU_1
   = 0
 capacity_cores_CPU_1: cores_sketch_CPU_1 + cores_dpdk_CPU_1 <= 7
 ns_sketch_CPU_1: - ns_single_CPU_1 + pdt_nsk_c_CPU_1 = 0
 ns_dpdk_CPU_1: 7.142857142857143 cores_dpdk_CPU_1 - pdt_nsc_dpdk_CPU_1
   = -21.42857142857143
 ns_P4_1: ns_P4_1 = 6.756756756756757
 res_acc: 1.220703125e-04 mem_tot_CPU_1 + 0.0208333333333333 mem_tot_P4_1
   + 0.25 rows_tot_P4_1 + 10 cores_sketch_CPU_1 + 10 cores_dpdk_CPU_1 - res
   = 0
 pdt_m_rows_CPU_1: pdt_m_rows_CPU_1 + [
   - rows_tot_CPU_1 * m_access_time_CPU_1 ] = 0
 pdt_nsk_c_CPU_1: pdt_nsk_c_CPU_1 + [
   - cores_sketch_CPU_1 * ns_sketch_CPU_1 ] = 0
 pdt_nsc_dpdk_CPU_1: pdt_nsc_dpdk_CPU_1 + [
   - cores_dpdk_CPU_1 * ns_dpdk_CPU_1 ] = 0
Bounds
 frac[0,1] <= 1
 frac[1,2] <= 1
 frac[0,0] <= 1
 frac[1,3] <= 1
 frac[1,4] <= 1
 frac[0,7] <= 1
 frac[0,8] <= 1
 frac[1,5] <= 1
 frac[0,6] <= 1
 frac[1,1] <= 1
 frac[0,5] <= 1
 frac[1,6] <= 1
 frac[1,8] <= 1
 frac[0,4] <= 1
 frac[1,0] <= 1
 frac[1,7] <= 1
 frac[0,2] <= 1
 frac[0,3] <= 1
 mem_tot_CPU_1 <= 32768
 mem_tot_P4_1 <= 576
 cores_sketch_CPU_1 <= 7
 1 <= cores_dpdk_CPU_1 <= 7
Generals
 cores_sketch_CPU_1 cores_dpdk_CPU_1
General Constraints
 mem_access_time_CPU_1: m_access_time_CPU_1 = PWL ( mem_tot_CPU_1 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_1: ns_CPU_1 = MAX ( ns_dpdk_CPU_1 , ns_sketch_CPU_1 )
 ns_overall: ns = MAX ( ns_CPU_1 , ns_P4_1 )
End

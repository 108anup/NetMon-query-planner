\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_P4 + 0 max_mem_CPU_Cluster + 0 max_rows_P4
   + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU_1 + 0 ns_sketch_CPU_1
   + 0 ns_dpdk_CPU_1 + 0 ns_CPU_1 + ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0
   res
Subject To
 cov_0_1: frac[1,1] + frac[2,1] >= 0.6
 cov_1_0: frac[0,0] >= 0.9
 cov_2_1: frac[1,1] + frac[2,1] >= 1
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 accuracy_1_1: - 5.09765625 frac[1,1] + mem[1,1] = 0
 accuracy_2_1: - 5.09765625 frac[2,1] + mem[2,1] = 0
 mem_tot_P4_4: - mem[0,0] + mem_tot_P4_4 = 0
 R7: - 0.00173611111111111 mem_tot_P4_4 + normalized_mem_tot_P4_4 = 0
 rows_tot_P4_4: - 3 frac[0,0] + rows_tot_P4_4 = 0
 R9: - 0.0208333333333333 rows_tot_P4_4 + normalized_rows_tot_P4_4 = 0
 mem_tot_P4_5: - mem[1,1] + mem_tot_P4_5 = 0
 R11: - 0.00173611111111111 mem_tot_P4_5 + normalized_mem_tot_P4_5 = 0
 rows_tot_P4_5: - 3 frac[1,1] + rows_tot_P4_5 = 0
 R13: - 0.0208333333333333 rows_tot_P4_5 + normalized_rows_tot_P4_5 = 0
 mem_tot_CPU_1: - mem[2,1] + mem_tot_CPU_1 = 0
 R15: - 3.05176e-05 mem_tot_CPU_1 + normalized_mem_tot_CPU_1 = 0
 rows_tot_CPU_1: - 3 frac[2,1] + rows_tot_CPU_1 = 0
 R17: - 0.0833333333333333 rows_tot_CPU_1 + normalized_rows_tot_CPU_1 = 0
 row_capacity_P4_4: rows_tot_P4_4 <= 48
 row_capacity_P4_5: rows_tot_P4_5 <= 48
 row_capacity_CPU_1: rows_tot_CPU_1 <= 12
 ns_P4_4: ns_P4_4 = 6.756756756756757
 ns_P4_5: ns_P4_5 = 6.756756756756757
 ns_single_CPU_1: 3.5 rows_tot_CPU_1 - ns_single_CPU_1 + pdt_m_rows_CPU_1
   = 0
 capacity_cores_CPU_1: cores_sketch_CPU_1 + cores_dpdk_CPU_1 <= 7
 ns_sketch_CPU_1: - ns_single_CPU_1 + pdt_nsk_c_CPU_1 = 0
 ns_dpdk_CPU_1: 7.142857142857143 cores_dpdk_CPU_1 - pdt_nsc_dpdk_CPU_1
   = -21.42857142857143
 res_acc: 0.0208333333333333 mem_tot_P4_4 + 0.25 rows_tot_P4_4
   + 0.0208333333333333 mem_tot_P4_5 + 0.25 rows_tot_P4_5
   + 1.220703125e-04 mem_tot_CPU_1 + 10 cores_sketch_CPU_1
   + 10 cores_dpdk_CPU_1 - res = 0
 pdt_m_rows_CPU_1: pdt_m_rows_CPU_1 + [
   - rows_tot_CPU_1 * m_access_time_CPU_1 ] = 0
 pdt_nsk_c_CPU_1: pdt_nsk_c_CPU_1 + [
   - cores_sketch_CPU_1 * ns_sketch_CPU_1 ] = 0
 pdt_nsc_dpdk_CPU_1: pdt_nsc_dpdk_CPU_1 + [
   - cores_dpdk_CPU_1 * ns_dpdk_CPU_1 ] = 0
Bounds
 mem_tot_P4_4 <= 576
 normalized_mem_tot_P4_4 <= 1
 normalized_rows_tot_P4_4 <= 1
 mem_tot_P4_5 <= 576
 normalized_mem_tot_P4_5 <= 1
 normalized_rows_tot_P4_5 <= 1
 mem_tot_CPU_1 <= 32768
 normalized_mem_tot_CPU_1 <= 1
 normalized_rows_tot_CPU_1 <= 1
 cores_sketch_CPU_1 <= 7
 1 <= cores_dpdk_CPU_1 <= 7
Binaries
 frac[0,0] frac[1,1] frac[2,1]
Generals
 cores_sketch_CPU_1 cores_dpdk_CPU_1
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_P4_4 , mem_tot_P4_5 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU_1 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_P4_4 , rows_tot_P4_5 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU_1 )
 mem_access_time_CPU_1: m_access_time_CPU_1 = PWL ( mem_tot_CPU_1 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_1: ns_CPU_1 = MAX ( ns_dpdk_CPU_1 , ns_sketch_CPU_1 )
 ns_overall: ns = MAX ( ns_P4_4 , ns_P4_5 , ns_CPU_1 )
End

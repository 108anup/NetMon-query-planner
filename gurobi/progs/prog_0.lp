\ Model netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 m_access_time_cpu_1 + 0 ns_sketch_cpu_1 + 0 ns_dpdk_cpu_1 + 0 ns_cpu_1
   + ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0
   res
Subject To
 cov_0_0: frac[1,0] + frac[0,0] >= 1
 accuracy_1_0: - 5.09765625 frac[1,0] + mem[1,0] = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 mem_tot_cpu_1: - mem[0,0] + mem_tot_cpu_1 = 0
 rows_tot_cpu_1: - 3 frac[0,0] + rows_tot_cpu_1 = 0
 mem_tot_p4_1: - mem[1,0] + mem_tot_p4_1 = 0
 rows_tot_p4_1: - 3 frac[1,0] + rows_tot_p4_1 = 0
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
 mem_tot_cpu_1 <= 32768
 mem_tot_p4_1 <= 576
 cores_sketch_cpu_1 <= 7
 1 <= cores_dpdk_cpu_1 <= 7
Binaries
 frac[1,0] frac[0,0]
Generals
 cores_sketch_cpu_1 cores_dpdk_cpu_1
General Constraints
 mem_access_time_cpu_1: m_access_time_cpu_1 = PWL ( mem_tot_cpu_1 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_1: ns_cpu_1 = MAX ( ns_dpdk_cpu_1 , ns_sketch_cpu_1 )
 ns_overall: ns = MAX ( ns_cpu_1 , ns_p4_1 )
End

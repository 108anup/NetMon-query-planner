\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_CPU + 0 max_rows_CPU + 0 m_access_time_CPU_2 + 0 ns_sketch_CPU_2
   + 0 ns_dpdk_CPU_2 + 0 ns_CPU_2 + 0 m_access_time_CPU_3
   + 0 ns_sketch_CPU_3 + 0 ns_dpdk_CPU_3 + 0 ns_CPU_3 + ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0
   res
Subject To
 cov_0_0: frac[1,0] + frac[0,0] >= 1
 cov_1_1: frac[1,1] >= 1
 accuracy_1_0: - 5.09765625 frac[1,0] + mem[1,0] = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 accuracy_1_1: - 5.09765625 frac[1,1] + mem[1,1] = 0
 mem_tot_CPU_2: - mem[0,0] + mem_tot_CPU_2 = 0
 rows_tot_CPU_2: - 3 frac[0,0] + rows_tot_CPU_2 = 0
 mem_tot_CPU_3: - mem[1,0] - mem[1,1] + mem_tot_CPU_3 = 0
 rows_tot_CPU_3: - 3 frac[1,0] - 3 frac[1,1] + rows_tot_CPU_3 = 0
 row_capacity_CPU_2: rows_tot_CPU_2 <= 12
 row_capacity_CPU_3: rows_tot_CPU_3 <= 12
 tot_mem: - mem_tot_CPU_2 - mem_tot_CPU_3 + tot_mem = 0
 tot_rows: - rows_tot_CPU_2 - rows_tot_CPU_3 + tot_rows = 0
 ns_single_CPU_2: 3.5 rows_tot_CPU_2 - ns_single_CPU_2 + pdt_m_rows_CPU_2
   = 0
 capacity_cores_CPU_2: cores_sketch_CPU_2 + cores_dpdk_CPU_2 <= 7
 ns_sketch_CPU_2: - ns_single_CPU_2 + pdt_nsk_c_CPU_2 = 0
 ns_dpdk_CPU_2: 7.142857142857143 cores_dpdk_CPU_2 - pdt_nsc_dpdk_CPU_2
   = -21.42857142857143
 ns_single_CPU_3: 3.5 rows_tot_CPU_3 - ns_single_CPU_3 + pdt_m_rows_CPU_3
   = 0
 capacity_cores_CPU_3: cores_sketch_CPU_3 + cores_dpdk_CPU_3 <= 7
 ns_sketch_CPU_3: - ns_single_CPU_3 + pdt_nsk_c_CPU_3 = 0
 ns_dpdk_CPU_3: 7.142857142857143 cores_dpdk_CPU_3 - pdt_nsc_dpdk_CPU_3
   = -21.42857142857143
 res_acc: 1.220703125e-04 mem_tot_CPU_2 + 1.220703125e-04 mem_tot_CPU_3
   + 10 cores_sketch_CPU_2 + 10 cores_dpdk_CPU_2 + 10 cores_sketch_CPU_3
   + 10 cores_dpdk_CPU_3 - res = 0
 pdt_m_rows_CPU_2: pdt_m_rows_CPU_2 + [
   - rows_tot_CPU_2 * m_access_time_CPU_2 ] = 0
 pdt_nsk_c_CPU_2: pdt_nsk_c_CPU_2 + [
   - cores_sketch_CPU_2 * ns_sketch_CPU_2 ] = 0
 pdt_nsc_dpdk_CPU_2: pdt_nsc_dpdk_CPU_2 + [
   - cores_dpdk_CPU_2 * ns_dpdk_CPU_2 ] = 0
 pdt_m_rows_CPU_3: pdt_m_rows_CPU_3 + [
   - rows_tot_CPU_3 * m_access_time_CPU_3 ] = 0
 pdt_nsk_c_CPU_3: pdt_nsk_c_CPU_3 + [
   - cores_sketch_CPU_3 * ns_sketch_CPU_3 ] = 0
 pdt_nsc_dpdk_CPU_3: pdt_nsc_dpdk_CPU_3 + [
   - cores_dpdk_CPU_3 * ns_dpdk_CPU_3 ] = 0
Bounds
 mem_tot_CPU_2 <= 32768
 mem_tot_CPU_3 <= 32768
 cores_sketch_CPU_2 <= 7
 1 <= cores_dpdk_CPU_2 <= 7
 cores_sketch_CPU_3 <= 7
 1 <= cores_dpdk_CPU_3 <= 7
Binaries
 frac[1,0] frac[0,0] frac[1,1]
Generals
 cores_sketch_CPU_2 cores_dpdk_CPU_2 cores_sketch_CPU_3 cores_dpdk_CPU_3
General Constraints
 mem_overall_CPU: max_mem_CPU = MAX ( mem_tot_CPU_2 , mem_tot_CPU_3 )
 rows_overall_CPU: max_rows_CPU = MAX ( rows_tot_CPU_2 , rows_tot_CPU_3 )
 mem_access_time_CPU_2: m_access_time_CPU_2 = PWL ( mem_tot_CPU_2 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_2: ns_CPU_2 = MAX ( ns_dpdk_CPU_2 , ns_sketch_CPU_2 )
 mem_access_time_CPU_3: m_access_time_CPU_3 = PWL ( mem_tot_CPU_3 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_3: ns_CPU_3 = MAX ( ns_dpdk_CPU_3 , ns_sketch_CPU_3 )
 ns_overall: ns = MAX ( ns_CPU_2 , ns_CPU_3 )
End

\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 m_access_time_CPU_16 + 0 ns_sketch_CPU_16 + 0 ns_dpdk_CPU_16
   + 0 ns_CPU_16 + 0 m_access_time_CPU_17 + 0 ns_sketch_CPU_17
   + 0 ns_dpdk_CPU_17 + 0 ns_CPU_17 + 0 m_access_time_CPU_18
   + 0 ns_sketch_CPU_18 + 0 ns_dpdk_CPU_18 + 0 ns_CPU_18
   + 0 m_access_time_CPU_19 + 0 ns_sketch_CPU_19 + 0 ns_dpdk_CPU_19
   + 0 ns_CPU_19 + ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0
   res
Subject To
 cov_0_0: frac[1,0] + frac[0,0] >= 1
 accuracy_1_0: - 5.09765625 frac[1,0] + mem[1,0] = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 mem_tot_CPU_16: - mem[0,0] + mem_tot_CPU_16 = 0
 rows_tot_CPU_16: - 3 frac[0,0] + rows_tot_CPU_16 = 0
 mem_tot_CPU_17: - mem[1,0] + mem_tot_CPU_17 = 0
 rows_tot_CPU_17: - 3 frac[1,0] + rows_tot_CPU_17 = 0
 mem_tot_CPU_18: mem_tot_CPU_18 = 0
 rows_tot_CPU_18: rows_tot_CPU_18 = 0
 mem_tot_CPU_19: mem_tot_CPU_19 = 0
 rows_tot_CPU_19: rows_tot_CPU_19 = 0
 row_capacity_CPU_16: rows_tot_CPU_16 <= 12
 row_capacity_CPU_17: rows_tot_CPU_17 <= 12
 row_capacity_CPU_18: rows_tot_CPU_18 <= 12
 row_capacity_CPU_19: rows_tot_CPU_19 <= 12
 ns_single_CPU_16: 3.5 rows_tot_CPU_16 - ns_single_CPU_16
   + pdt_m_rows_CPU_16 = 0
 capacity_cores_CPU_16: cores_sketch_CPU_16 + cores_dpdk_CPU_16 <= 7
 ns_sketch_CPU_16: - ns_single_CPU_16 + pdt_nsk_c_CPU_16 = 0
 ns_dpdk_CPU_16: 7.142857142857143 cores_dpdk_CPU_16 - pdt_nsc_dpdk_CPU_16
   = -21.42857142857143
 ns_single_CPU_17: 3.5 rows_tot_CPU_17 - ns_single_CPU_17
   + pdt_m_rows_CPU_17 = 0
 capacity_cores_CPU_17: cores_sketch_CPU_17 + cores_dpdk_CPU_17 <= 7
 ns_sketch_CPU_17: - ns_single_CPU_17 + pdt_nsk_c_CPU_17 = 0
 ns_dpdk_CPU_17: 7.142857142857143 cores_dpdk_CPU_17 - pdt_nsc_dpdk_CPU_17
   = -21.42857142857143
 ns_single_CPU_18: 3.5 rows_tot_CPU_18 - ns_single_CPU_18
   + pdt_m_rows_CPU_18 = 0
 capacity_cores_CPU_18: cores_sketch_CPU_18 + cores_dpdk_CPU_18 <= 7
 ns_sketch_CPU_18: - ns_single_CPU_18 + pdt_nsk_c_CPU_18 = 0
 ns_dpdk_CPU_18: 7.142857142857143 cores_dpdk_CPU_18 - pdt_nsc_dpdk_CPU_18
   = -21.42857142857143
 ns_single_CPU_19: 3.5 rows_tot_CPU_19 - ns_single_CPU_19
   + pdt_m_rows_CPU_19 = 0
 capacity_cores_CPU_19: cores_sketch_CPU_19 + cores_dpdk_CPU_19 <= 7
 ns_sketch_CPU_19: - ns_single_CPU_19 + pdt_nsk_c_CPU_19 = 0
 ns_dpdk_CPU_19: 7.142857142857143 cores_dpdk_CPU_19 - pdt_nsc_dpdk_CPU_19
   = -21.42857142857143
 res_acc: 1.220703125e-04 mem_tot_CPU_16 + 1.220703125e-04 mem_tot_CPU_17
   + 1.220703125e-04 mem_tot_CPU_18 + 1.220703125e-04 mem_tot_CPU_19
   + 10 cores_sketch_CPU_16 + 10 cores_dpdk_CPU_16 + 10 cores_sketch_CPU_17
   + 10 cores_dpdk_CPU_17 + 10 cores_sketch_CPU_18 + 10 cores_dpdk_CPU_18
   + 10 cores_sketch_CPU_19 + 10 cores_dpdk_CPU_19 - res = 0
 pdt_m_rows_CPU_16: pdt_m_rows_CPU_16 + [
   - rows_tot_CPU_16 * m_access_time_CPU_16 ] = 0
 pdt_nsk_c_CPU_16: pdt_nsk_c_CPU_16 + [
   - cores_sketch_CPU_16 * ns_sketch_CPU_16 ] = 0
 pdt_nsc_dpdk_CPU_16: pdt_nsc_dpdk_CPU_16 + [
   - cores_dpdk_CPU_16 * ns_dpdk_CPU_16 ] = 0
 pdt_m_rows_CPU_17: pdt_m_rows_CPU_17 + [
   - rows_tot_CPU_17 * m_access_time_CPU_17 ] = 0
 pdt_nsk_c_CPU_17: pdt_nsk_c_CPU_17 + [
   - cores_sketch_CPU_17 * ns_sketch_CPU_17 ] = 0
 pdt_nsc_dpdk_CPU_17: pdt_nsc_dpdk_CPU_17 + [
   - cores_dpdk_CPU_17 * ns_dpdk_CPU_17 ] = 0
 pdt_m_rows_CPU_18: pdt_m_rows_CPU_18 + [
   - rows_tot_CPU_18 * m_access_time_CPU_18 ] = 0
 pdt_nsk_c_CPU_18: pdt_nsk_c_CPU_18 + [
   - cores_sketch_CPU_18 * ns_sketch_CPU_18 ] = 0
 pdt_nsc_dpdk_CPU_18: pdt_nsc_dpdk_CPU_18 + [
   - cores_dpdk_CPU_18 * ns_dpdk_CPU_18 ] = 0
 pdt_m_rows_CPU_19: pdt_m_rows_CPU_19 + [
   - rows_tot_CPU_19 * m_access_time_CPU_19 ] = 0
 pdt_nsk_c_CPU_19: pdt_nsk_c_CPU_19 + [
   - cores_sketch_CPU_19 * ns_sketch_CPU_19 ] = 0
 pdt_nsc_dpdk_CPU_19: pdt_nsc_dpdk_CPU_19 + [
   - cores_dpdk_CPU_19 * ns_dpdk_CPU_19 ] = 0
Bounds
 mem_tot_CPU_16 <= 32768
 mem_tot_CPU_17 <= 32768
 mem_tot_CPU_18 <= 32768
 mem_tot_CPU_19 <= 32768
 cores_sketch_CPU_16 <= 7
 1 <= cores_dpdk_CPU_16 <= 7
 cores_sketch_CPU_17 <= 7
 1 <= cores_dpdk_CPU_17 <= 7
 cores_sketch_CPU_18 <= 7
 1 <= cores_dpdk_CPU_18 <= 7
 cores_sketch_CPU_19 <= 7
 1 <= cores_dpdk_CPU_19 <= 7
Binaries
 frac[1,0] frac[0,0]
Generals
 cores_sketch_CPU_16 cores_dpdk_CPU_16 cores_sketch_CPU_17
 cores_dpdk_CPU_17 cores_sketch_CPU_18 cores_dpdk_CPU_18
 cores_sketch_CPU_19 cores_dpdk_CPU_19
General Constraints
 mem_access_time_CPU_16: m_access_time_CPU_16 = PWL ( mem_tot_CPU_16 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_16: ns_CPU_16 = MAX ( ns_dpdk_CPU_16 , ns_sketch_CPU_16 )
 mem_access_time_CPU_17: m_access_time_CPU_17 = PWL ( mem_tot_CPU_17 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_17: ns_CPU_17 = MAX ( ns_dpdk_CPU_17 , ns_sketch_CPU_17 )
 mem_access_time_CPU_18: m_access_time_CPU_18 = PWL ( mem_tot_CPU_18 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_18: ns_CPU_18 = MAX ( ns_dpdk_CPU_18 , ns_sketch_CPU_18 )
 mem_access_time_CPU_19: m_access_time_CPU_19 = PWL ( mem_tot_CPU_19 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_19: ns_CPU_19 = MAX ( ns_dpdk_CPU_19 , ns_sketch_CPU_19 )
 ns_overall: ns = MAX ( ns_CPU_16 , ns_CPU_17 , ns_CPU_18 , ns_CPU_19 )
End

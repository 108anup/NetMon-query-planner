\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_CPU_Cluster + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU2156
   + 0 m_access_time_CPU3964 + res
Subject To
 cov_0_2167: frac[0,2167] + frac[1,2167] >= 0.5392877447520098
 accuracy_0_2167: - 5.09765625 frac[0,2167] + mem[0,2167] = 0
 accuracy_1_2167: - 5.09765625 frac[1,2167] + mem[1,2167] = 0
 mem_tot_CPU2156: - mem[0,2167] + mem_tot_CPU2156 = 0
 R4: - 3.05176e-05 mem_tot_CPU2156 + normalized_mem_tot_CPU2156 = 0
 rows_tot_CPU2156: - 3 frac[0,2167] + rows_tot_CPU2156 = 0
 R6: - 0.0833333333333333 rows_tot_CPU2156 + normalized_rows_tot_CPU2156
   = 0
 mem_tot_CPU3964: - mem[1,2167] + mem_tot_CPU3964 = 0
 R8: - 3.05176e-05 mem_tot_CPU3964 + normalized_mem_tot_CPU3964 = 0
 rows_tot_CPU3964: - 3 frac[1,2167] + rows_tot_CPU3964 = 0
 R10: - 0.0833333333333333 rows_tot_CPU3964 + normalized_rows_tot_CPU3964
   = 0
 row_capacity_CPU2156: rows_tot_CPU2156 <= 12
 row_capacity_CPU3964: rows_tot_CPU3964 <= 12
 ns_single_CPU2156: 3.5 rows_tot_CPU2156 - ns_single_CPU2156
   + pdt_m_rows_CPU2156 = 0
 capacity_cores_CPU2156: cores_sketch_CPU2156 + cores_dpdk_CPU2156 <= 7
 ns_sketch_CPU2156: - ns_single_CPU2156
   + 10.71428571428572 cores_sketch_CPU2156 >= 0
 ns_dpdk_CPU2156: - 3.571428571428572 cores_dpdk_CPU2156
   <= -21.42857142857143
 ns_single_CPU3964: 3.5 rows_tot_CPU3964 - ns_single_CPU3964
   + pdt_m_rows_CPU3964 = 0
 capacity_cores_CPU3964: cores_sketch_CPU3964 + cores_dpdk_CPU3964 <= 7
 ns_sketch_CPU3964: - ns_single_CPU3964
   + 10.71428571428572 cores_sketch_CPU3964 >= 0
 ns_dpdk_CPU3964: - 3.571428571428572 cores_dpdk_CPU3964
   <= -21.42857142857143
 res_acc: 1.220703125e-04 mem_tot_CPU2156 + 1.220703125e-04 mem_tot_CPU3964
   + 10 cores_sketch_CPU2156 + 10 cores_dpdk_CPU2156
   + 10 cores_sketch_CPU3964 + 10 cores_dpdk_CPU3964 - res = 0
 pdt_m_rows_CPU2156: pdt_m_rows_CPU2156 + [
   - rows_tot_CPU2156 * m_access_time_CPU2156 ] = 0
 pdt_m_rows_CPU3964: pdt_m_rows_CPU3964 + [
   - rows_tot_CPU3964 * m_access_time_CPU3964 ] = 0
Bounds
 frac[0,2167] <= 1
 frac[1,2167] <= 1
 mem_tot_CPU2156 <= 32768
 normalized_mem_tot_CPU2156 <= 1
 normalized_rows_tot_CPU2156 <= 1
 mem_tot_CPU3964 <= 32768
 normalized_mem_tot_CPU3964 <= 1
 normalized_rows_tot_CPU3964 <= 1
 cores_sketch_CPU2156 <= 7
 1 <= cores_dpdk_CPU2156 <= 7
 cores_sketch_CPU3964 <= 7
 1 <= cores_dpdk_CPU3964 <= 7
Generals
 cores_sketch_CPU2156 cores_dpdk_CPU2156 cores_sketch_CPU3964
 cores_dpdk_CPU3964
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU2156 , normalized_mem_tot_CPU3964 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX (
   normalized_rows_tot_CPU2156 , normalized_rows_tot_CPU3964 )
 mem_access_time_CPU2156:
   m_access_time_CPU2156 = PWL ( mem_tot_CPU2156 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU3964:
   m_access_time_CPU3964 = PWL ( mem_tot_CPU3964 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

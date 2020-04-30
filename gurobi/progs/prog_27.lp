\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_CPU_Cluster + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU_2
   + res
Subject To
 cov_0_0: frac[0,0] >= 1
 cov_1_0: frac[0,0] >= 1
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 mem_tot_CPU_2: - mem[0,0] + mem_tot_CPU_2 = 0
 R4: - 3.05176e-05 mem_tot_CPU_2 + normalized_mem_tot_CPU_2 = 0
 rows_tot_CPU_2: - 3 frac[0,0] + rows_tot_CPU_2 = 0
 R6: - 0.0833333333333333 rows_tot_CPU_2 + normalized_rows_tot_CPU_2 = 0
 row_capacity_CPU_2: rows_tot_CPU_2 <= 12
 ns_single_CPU_2: 3.5 rows_tot_CPU_2 - ns_single_CPU_2 + pdt_m_rows_CPU_2
   = 0
 capacity_cores_CPU_2: cores_sketch_CPU_2 + cores_dpdk_CPU_2 <= 7
 ns_sketch_CPU_2: - ns_single_CPU_2 + 11.42857142857143 cores_sketch_CPU_2
   >= 0
 ns_dpdk_CPU_2: - 4.285714285714287 cores_dpdk_CPU_2 <= -21.42857142857143
 res_acc: 1.220703125e-04 mem_tot_CPU_2 + 10 cores_sketch_CPU_2
   + 10 cores_dpdk_CPU_2 - res = 0
 pdt_m_rows_CPU_2: pdt_m_rows_CPU_2 + [
   - rows_tot_CPU_2 * m_access_time_CPU_2 ] = 0
Bounds
 mem_tot_CPU_2 <= 32768
 normalized_mem_tot_CPU_2 <= 1
 normalized_rows_tot_CPU_2 <= 1
 cores_sketch_CPU_2 <= 7
 1 <= cores_dpdk_CPU_2 <= 7
Binaries
 frac[0,0]
Generals
 cores_sketch_CPU_2 cores_dpdk_CPU_2
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU_2 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU_2 )
 mem_access_time_CPU_2: m_access_time_CPU_2 = PWL ( mem_tot_CPU_2 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

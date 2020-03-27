\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_CPU_Cluster + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU7798
   + 0 m_access_time_CPU7817 + 0 m_access_time_CPU7820
   + 0 m_access_time_CPU7824 + res
Subject To
 cov_0_2: frac[0,2] >= 0.9
 cov_1_0: frac[3,0] >= 0.7
 cov_2_0: frac[3,0] >= 0.7
 cov_3_3: frac[1,3] >= 0.1699292277795224
 cov_4_1: frac[2,1] >= 0.7
 accuracy_3_0: - 5.09765625 frac[3,0] + mem[3,0] = 0
 accuracy_1_3: - 5.09765625 frac[1,3] + mem[1,3] = 0
 accuracy_0_2: - 5.09765625 frac[0,2] + mem[0,2] = 0
 accuracy_2_1: - 5.09765625 frac[2,1] + mem[2,1] = 0
 mem_tot_CPU7798: - mem[0,2] + mem_tot_CPU7798 = 0
 R10: - 3.05176e-05 mem_tot_CPU7798 + normalized_mem_tot_CPU7798 = 0
 rows_tot_CPU7798: - 3 frac[0,2] + rows_tot_CPU7798 = 0
 R12: - 0.0833333333333333 rows_tot_CPU7798 + normalized_rows_tot_CPU7798
   = 0
 mem_tot_CPU7817: - mem[1,3] + mem_tot_CPU7817 = 0
 R14: - 3.05176e-05 mem_tot_CPU7817 + normalized_mem_tot_CPU7817 = 0
 rows_tot_CPU7817: - 3 frac[1,3] + rows_tot_CPU7817 = 0
 R16: - 0.0833333333333333 rows_tot_CPU7817 + normalized_rows_tot_CPU7817
   = 0
 mem_tot_CPU7820: - mem[2,1] + mem_tot_CPU7820 = 0
 R18: - 3.05176e-05 mem_tot_CPU7820 + normalized_mem_tot_CPU7820 = 0
 rows_tot_CPU7820: - 3 frac[2,1] + rows_tot_CPU7820 = 0
 R20: - 0.0833333333333333 rows_tot_CPU7820 + normalized_rows_tot_CPU7820
   = 0
 mem_tot_CPU7824: - mem[3,0] + mem_tot_CPU7824 = 0
 R22: - 3.05176e-05 mem_tot_CPU7824 + normalized_mem_tot_CPU7824 = 0
 rows_tot_CPU7824: - 3 frac[3,0] + rows_tot_CPU7824 = 0
 R24: - 0.0833333333333333 rows_tot_CPU7824 + normalized_rows_tot_CPU7824
   = 0
 row_capacity_CPU7798: rows_tot_CPU7798 <= 12
 row_capacity_CPU7817: rows_tot_CPU7817 <= 12
 row_capacity_CPU7820: rows_tot_CPU7820 <= 12
 row_capacity_CPU7824: rows_tot_CPU7824 <= 12
 ns_single_CPU7798: 3.5 rows_tot_CPU7798 - ns_single_CPU7798
   + pdt_m_rows_CPU7798 = 0
 capacity_cores_CPU7798: cores_sketch_CPU7798 + cores_dpdk_CPU7798 <= 7
 ns_sketch_CPU7798: - ns_single_CPU7798
   + 11.42857142857143 cores_sketch_CPU7798 >= 0
 ns_dpdk_CPU7798: - 4.285714285714286 cores_dpdk_CPU7798
   <= -21.42857142857143
 ns_single_CPU7817: 3.5 rows_tot_CPU7817 - ns_single_CPU7817
   + pdt_m_rows_CPU7817 = 0
 capacity_cores_CPU7817: cores_sketch_CPU7817 + cores_dpdk_CPU7817 <= 7
 ns_sketch_CPU7817: - ns_single_CPU7817
   + 11.42857142857143 cores_sketch_CPU7817 >= 0
 ns_dpdk_CPU7817: - 4.285714285714286 cores_dpdk_CPU7817
   <= -21.42857142857143
 ns_single_CPU7820: 3.5 rows_tot_CPU7820 - ns_single_CPU7820
   + pdt_m_rows_CPU7820 = 0
 capacity_cores_CPU7820: cores_sketch_CPU7820 + cores_dpdk_CPU7820 <= 7
 ns_sketch_CPU7820: - ns_single_CPU7820
   + 11.42857142857143 cores_sketch_CPU7820 >= 0
 ns_dpdk_CPU7820: - 4.285714285714286 cores_dpdk_CPU7820
   <= -21.42857142857143
 ns_single_CPU7824: 3.5 rows_tot_CPU7824 - ns_single_CPU7824
   + pdt_m_rows_CPU7824 = 0
 capacity_cores_CPU7824: cores_sketch_CPU7824 + cores_dpdk_CPU7824 <= 7
 ns_sketch_CPU7824: - ns_single_CPU7824
   + 11.42857142857143 cores_sketch_CPU7824 >= 0
 ns_dpdk_CPU7824: - 4.285714285714286 cores_dpdk_CPU7824
   <= -21.42857142857143
 res_acc: 1.220703125e-04 mem_tot_CPU7798 + 1.220703125e-04 mem_tot_CPU7817
   + 1.220703125e-04 mem_tot_CPU7820 + 1.220703125e-04 mem_tot_CPU7824
   + 10 cores_sketch_CPU7798 + 10 cores_dpdk_CPU7798
   + 10 cores_sketch_CPU7817 + 10 cores_dpdk_CPU7817
   + 10 cores_sketch_CPU7820 + 10 cores_dpdk_CPU7820
   + 10 cores_sketch_CPU7824 + 10 cores_dpdk_CPU7824 - res = 0
 pdt_m_rows_CPU7798: pdt_m_rows_CPU7798 + [
   - rows_tot_CPU7798 * m_access_time_CPU7798 ] = 0
 pdt_m_rows_CPU7817: pdt_m_rows_CPU7817 + [
   - rows_tot_CPU7817 * m_access_time_CPU7817 ] = 0
 pdt_m_rows_CPU7820: pdt_m_rows_CPU7820 + [
   - rows_tot_CPU7820 * m_access_time_CPU7820 ] = 0
 pdt_m_rows_CPU7824: pdt_m_rows_CPU7824 + [
   - rows_tot_CPU7824 * m_access_time_CPU7824 ] = 0
Bounds
 frac[3,0] <= 1
 frac[1,3] <= 1
 frac[0,2] <= 1
 frac[2,1] <= 1
 mem_tot_CPU7798 <= 32768
 normalized_mem_tot_CPU7798 <= 1
 normalized_rows_tot_CPU7798 <= 1
 mem_tot_CPU7817 <= 32768
 normalized_mem_tot_CPU7817 <= 1
 normalized_rows_tot_CPU7817 <= 1
 mem_tot_CPU7820 <= 32768
 normalized_mem_tot_CPU7820 <= 1
 normalized_rows_tot_CPU7820 <= 1
 mem_tot_CPU7824 <= 32768
 normalized_mem_tot_CPU7824 <= 1
 normalized_rows_tot_CPU7824 <= 1
 cores_sketch_CPU7798 <= 7
 1 <= cores_dpdk_CPU7798 <= 7
 cores_sketch_CPU7817 <= 7
 1 <= cores_dpdk_CPU7817 <= 7
 cores_sketch_CPU7820 <= 7
 1 <= cores_dpdk_CPU7820 <= 7
 cores_sketch_CPU7824 <= 7
 1 <= cores_dpdk_CPU7824 <= 7
Generals
 cores_sketch_CPU7798 cores_dpdk_CPU7798 cores_sketch_CPU7817
 cores_dpdk_CPU7817 cores_sketch_CPU7820 cores_dpdk_CPU7820
 cores_sketch_CPU7824 cores_dpdk_CPU7824
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU7798 , normalized_mem_tot_CPU7817 ,
   normalized_mem_tot_CPU7820 , normalized_mem_tot_CPU7824 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX (
   normalized_rows_tot_CPU7798 , normalized_rows_tot_CPU7817 ,
   normalized_rows_tot_CPU7820 , normalized_rows_tot_CPU7824 )
 mem_access_time_CPU7798:
   m_access_time_CPU7798 = PWL ( mem_tot_CPU7798 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU7817:
   m_access_time_CPU7817 = PWL ( mem_tot_CPU7817 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU7820:
   m_access_time_CPU7820 = PWL ( mem_tot_CPU7820 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU7824:
   m_access_time_CPU7824 = PWL ( mem_tot_CPU7824 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

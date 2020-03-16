\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_CPU_Cluster + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU_0
   + 0 ns_sketch_CPU_0 + 0 ns_dpdk_CPU_0 + 0 ns_CPU_0
   + 0 m_access_time_CPU_1 + 0 ns_sketch_CPU_1 + 0 ns_dpdk_CPU_1
   + 0 ns_CPU_1 + 0 m_access_time_CPU_2 + 0 ns_sketch_CPU_2
   + 0 ns_dpdk_CPU_2 + 0 ns_CPU_2 + 0 m_access_time_CPU_3
   + 0 ns_sketch_CPU_3 + 0 ns_dpdk_CPU_3 + 0 ns_CPU_3
   + 0 m_access_time_CPU_4 + 0 ns_sketch_CPU_4 + 0 ns_dpdk_CPU_4
   + 0 ns_CPU_4 + ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0
   res
Subject To
 cov_0_0: frac[0,0] + frac[1,0] + frac[4,0] >= 1
 cov_1_1: frac[2,1] + frac[4,1] + frac[1,1] >= 1
 cov_2_2: frac[3,2] + frac[2,2] + frac[4,2] >= 1
 cov_3_3: frac[3,3] + frac[0,3] + frac[4,3] >= 1
 accuracy_3_2: - 5.09765625 frac[3,2] + mem[3,2] = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 accuracy_3_3: - 5.09765625 frac[3,3] + mem[3,3] = 0
 accuracy_2_1: - 5.09765625 frac[2,1] + mem[2,1] = 0
 accuracy_0_3: - 5.09765625 frac[0,3] + mem[0,3] = 0
 accuracy_4_3: - 5.09765625 frac[4,3] + mem[4,3] = 0
 accuracy_2_2: - 5.09765625 frac[2,2] + mem[2,2] = 0
 accuracy_4_2: - 5.09765625 frac[4,2] + mem[4,2] = 0
 accuracy_1_0: - 5.09765625 frac[1,0] + mem[1,0] = 0
 accuracy_4_1: - 5.09765625 frac[4,1] + mem[4,1] = 0
 accuracy_1_1: - 5.09765625 frac[1,1] + mem[1,1] = 0
 accuracy_4_0: - 5.09765625 frac[4,0] + mem[4,0] = 0
 mem_tot_CPU_0: - mem[0,0] - mem[0,3] + mem_tot_CPU_0 = 0
 R17: - 3.05176e-05 mem_tot_CPU_0 + normalized_mem_tot_CPU_0 = 0
 rows_tot_CPU_0: - 3 frac[0,0] - 3 frac[0,3] + rows_tot_CPU_0 = 0
 R19: - 0.0833333333333333 rows_tot_CPU_0 + normalized_rows_tot_CPU_0 = 0
 mem_tot_CPU_1: - mem[1,0] - mem[1,1] + mem_tot_CPU_1 = 0
 R21: - 3.05176e-05 mem_tot_CPU_1 + normalized_mem_tot_CPU_1 = 0
 rows_tot_CPU_1: - 3 frac[1,0] - 3 frac[1,1] + rows_tot_CPU_1 = 0
 R23: - 0.0833333333333333 rows_tot_CPU_1 + normalized_rows_tot_CPU_1 = 0
 mem_tot_CPU_2: - mem[2,1] - mem[2,2] + mem_tot_CPU_2 = 0
 R25: - 3.05176e-05 mem_tot_CPU_2 + normalized_mem_tot_CPU_2 = 0
 rows_tot_CPU_2: - 3 frac[2,1] - 3 frac[2,2] + rows_tot_CPU_2 = 0
 R27: - 0.0833333333333333 rows_tot_CPU_2 + normalized_rows_tot_CPU_2 = 0
 mem_tot_CPU_3: - mem[3,2] - mem[3,3] + mem_tot_CPU_3 = 0
 R29: - 3.05176e-05 mem_tot_CPU_3 + normalized_mem_tot_CPU_3 = 0
 rows_tot_CPU_3: - 3 frac[3,2] - 3 frac[3,3] + rows_tot_CPU_3 = 0
 R31: - 0.0833333333333333 rows_tot_CPU_3 + normalized_rows_tot_CPU_3 = 0
 mem_tot_CPU_4: - mem[4,3] - mem[4,2] - mem[4,1] - mem[4,0] + mem_tot_CPU_4
   = 0
 R33: - 3.05176e-05 mem_tot_CPU_4 + normalized_mem_tot_CPU_4 = 0
 rows_tot_CPU_4: - 3 frac[4,3] - 3 frac[4,2] - 3 frac[4,1] - 3 frac[4,0]
   + rows_tot_CPU_4 = 0
 R35: - 0.0833333333333333 rows_tot_CPU_4 + normalized_rows_tot_CPU_4 = 0
 row_capacity_CPU_0: rows_tot_CPU_0 <= 12
 row_capacity_CPU_1: rows_tot_CPU_1 <= 12
 row_capacity_CPU_2: rows_tot_CPU_2 <= 12
 row_capacity_CPU_3: rows_tot_CPU_3 <= 12
 row_capacity_CPU_4: rows_tot_CPU_4 <= 12
 ns_single_CPU_0: 3.5 rows_tot_CPU_0 - ns_single_CPU_0 + pdt_m_rows_CPU_0
   = 0
 capacity_cores_CPU_0: cores_sketch_CPU_0 + cores_dpdk_CPU_0 <= 7
 ns_sketch_CPU_0: - ns_single_CPU_0 + pdt_nsk_c_CPU_0 = 0
 ns_dpdk_CPU_0: 7.142857142857143 cores_dpdk_CPU_0 - pdt_nsc_dpdk_CPU_0
   = -21.42857142857143
 ns_single_CPU_1: 3.5 rows_tot_CPU_1 - ns_single_CPU_1 + pdt_m_rows_CPU_1
   = 0
 capacity_cores_CPU_1: cores_sketch_CPU_1 + cores_dpdk_CPU_1 <= 7
 ns_sketch_CPU_1: - ns_single_CPU_1 + pdt_nsk_c_CPU_1 = 0
 ns_dpdk_CPU_1: 7.142857142857143 cores_dpdk_CPU_1 - pdt_nsc_dpdk_CPU_1
   = -21.42857142857143
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
 ns_single_CPU_4: 3.5 rows_tot_CPU_4 - ns_single_CPU_4 + pdt_m_rows_CPU_4
   = 0
 capacity_cores_CPU_4: cores_sketch_CPU_4 + cores_dpdk_CPU_4 <= 7
 ns_sketch_CPU_4: - ns_single_CPU_4 + pdt_nsk_c_CPU_4 = 0
 ns_dpdk_CPU_4: 7.142857142857143 cores_dpdk_CPU_4 - pdt_nsc_dpdk_CPU_4
   = -21.42857142857143
 res_acc: 1.220703125e-04 mem_tot_CPU_0 + 1.220703125e-04 mem_tot_CPU_1
   + 1.220703125e-04 mem_tot_CPU_2 + 1.220703125e-04 mem_tot_CPU_3
   + 1.220703125e-04 mem_tot_CPU_4 + 10 cores_sketch_CPU_0
   + 10 cores_dpdk_CPU_0 + 10 cores_sketch_CPU_1 + 10 cores_dpdk_CPU_1
   + 10 cores_sketch_CPU_2 + 10 cores_dpdk_CPU_2 + 10 cores_sketch_CPU_3
   + 10 cores_dpdk_CPU_3 + 10 cores_sketch_CPU_4 + 10 cores_dpdk_CPU_4
   - res = 0
 pdt_m_rows_CPU_0: pdt_m_rows_CPU_0 + [
   - rows_tot_CPU_0 * m_access_time_CPU_0 ] = 0
 pdt_nsk_c_CPU_0: pdt_nsk_c_CPU_0 + [
   - cores_sketch_CPU_0 * ns_sketch_CPU_0 ] = 0
 pdt_nsc_dpdk_CPU_0: pdt_nsc_dpdk_CPU_0 + [
   - cores_dpdk_CPU_0 * ns_dpdk_CPU_0 ] = 0
 pdt_m_rows_CPU_1: pdt_m_rows_CPU_1 + [
   - rows_tot_CPU_1 * m_access_time_CPU_1 ] = 0
 pdt_nsk_c_CPU_1: pdt_nsk_c_CPU_1 + [
   - cores_sketch_CPU_1 * ns_sketch_CPU_1 ] = 0
 pdt_nsc_dpdk_CPU_1: pdt_nsc_dpdk_CPU_1 + [
   - cores_dpdk_CPU_1 * ns_dpdk_CPU_1 ] = 0
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
 pdt_m_rows_CPU_4: pdt_m_rows_CPU_4 + [
   - rows_tot_CPU_4 * m_access_time_CPU_4 ] = 0
 pdt_nsk_c_CPU_4: pdt_nsk_c_CPU_4 + [
   - cores_sketch_CPU_4 * ns_sketch_CPU_4 ] = 0
 pdt_nsc_dpdk_CPU_4: pdt_nsc_dpdk_CPU_4 + [
   - cores_dpdk_CPU_4 * ns_dpdk_CPU_4 ] = 0
Bounds
 mem_tot_CPU_0 <= 32768
 normalized_mem_tot_CPU_0 <= 1
 normalized_rows_tot_CPU_0 <= 1
 mem_tot_CPU_1 <= 32768
 normalized_mem_tot_CPU_1 <= 1
 normalized_rows_tot_CPU_1 <= 1
 mem_tot_CPU_2 <= 32768
 normalized_mem_tot_CPU_2 <= 1
 normalized_rows_tot_CPU_2 <= 1
 mem_tot_CPU_3 <= 32768
 normalized_mem_tot_CPU_3 <= 1
 normalized_rows_tot_CPU_3 <= 1
 mem_tot_CPU_4 <= 32768
 normalized_mem_tot_CPU_4 <= 1
 normalized_rows_tot_CPU_4 <= 1
 cores_sketch_CPU_0 <= 7
 1 <= cores_dpdk_CPU_0 <= 7
 cores_sketch_CPU_1 <= 7
 1 <= cores_dpdk_CPU_1 <= 7
 cores_sketch_CPU_2 <= 7
 1 <= cores_dpdk_CPU_2 <= 7
 cores_sketch_CPU_3 <= 7
 1 <= cores_dpdk_CPU_3 <= 7
 cores_sketch_CPU_4 <= 7
 1 <= cores_dpdk_CPU_4 <= 7
Binaries
 frac[3,2] frac[0,0] frac[3,3] frac[2,1] frac[0,3] frac[4,3] frac[2,2]
 frac[4,2] frac[1,0] frac[4,1] frac[1,1] frac[4,0]
Generals
 cores_sketch_CPU_0 cores_dpdk_CPU_0 cores_sketch_CPU_1 cores_dpdk_CPU_1
 cores_sketch_CPU_2 cores_dpdk_CPU_2 cores_sketch_CPU_3 cores_dpdk_CPU_3
 cores_sketch_CPU_4 cores_dpdk_CPU_4
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU_0 , normalized_mem_tot_CPU_1 ,
   normalized_mem_tot_CPU_2 , normalized_mem_tot_CPU_3 ,
   normalized_mem_tot_CPU_4 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU_0 ,
   normalized_rows_tot_CPU_1 , normalized_rows_tot_CPU_2 ,
   normalized_rows_tot_CPU_3 , normalized_rows_tot_CPU_4 )
 mem_access_time_CPU_0: m_access_time_CPU_0 = PWL ( mem_tot_CPU_0 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_0: ns_CPU_0 = MAX ( ns_dpdk_CPU_0 , ns_sketch_CPU_0 )
 mem_access_time_CPU_1: m_access_time_CPU_1 = PWL ( mem_tot_CPU_1 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_1: ns_CPU_1 = MAX ( ns_dpdk_CPU_1 , ns_sketch_CPU_1 )
 mem_access_time_CPU_2: m_access_time_CPU_2 = PWL ( mem_tot_CPU_2 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_2: ns_CPU_2 = MAX ( ns_dpdk_CPU_2 , ns_sketch_CPU_2 )
 mem_access_time_CPU_3: m_access_time_CPU_3 = PWL ( mem_tot_CPU_3 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_3: ns_CPU_3 = MAX ( ns_dpdk_CPU_3 , ns_sketch_CPU_3 )
 mem_access_time_CPU_4: m_access_time_CPU_4 = PWL ( mem_tot_CPU_4 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_4: ns_CPU_4 = MAX ( ns_dpdk_CPU_4 , ns_sketch_CPU_4 )
 ns_overall: ns = MAX ( ns_CPU_0 , ns_CPU_1 , ns_CPU_2 , ns_CPU_3 ,
   ns_CPU_4 )
End

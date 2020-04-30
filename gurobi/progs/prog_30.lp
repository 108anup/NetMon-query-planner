\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_CPU_Cluster + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU29
   + 0 m_access_time_CPU8 + 0 m_access_time_CPU7 + 0 m_access_time_CPU21
   + 0 m_access_time_CPU2 + 0 m_access_time_CPU3 + 0 m_access_time_CPU18
   + 0 m_access_time_CPU5 + res
Subject To
 cov_0_0: frac[0,0] + frac[4,0] >= 1.9622e-09
 cov_1_0: frac[0,0] >= 1.9622e-09
 cov_2_0: frac[1,0] + frac[5,0] >= 1.9622e-09
 accuracy_1_0: - 509.68359375 frac[1,0] + mem[1,0] = 0
 accuracy_0_0: - 509.68359375 frac[0,0] + mem[0,0] = 0
 accuracy_5_0: - 509.68359375 frac[5,0] + mem[5,0] = 0
 accuracy_4_0: - 509.68359375 frac[4,0] + mem[4,0] = 0
 mem_tot_CPU29: - mem[0,0] + mem_tot_CPU29 = 0
 R8: - 3.05176e-05 mem_tot_CPU29 + normalized_mem_tot_CPU29 = 0
 rows_tot_CPU29: - 3 frac[0,0] + rows_tot_CPU29 = 0
 R10: - 0.0833333333333333 rows_tot_CPU29 + normalized_rows_tot_CPU29 = 0
 mem_tot_CPU8: - mem[1,0] + mem_tot_CPU8 = 0
 R12: - 3.05176e-05 mem_tot_CPU8 + normalized_mem_tot_CPU8 = 0
 rows_tot_CPU8: - 3 frac[1,0] + rows_tot_CPU8 = 0
 R14: - 0.0833333333333333 rows_tot_CPU8 + normalized_rows_tot_CPU8 = 0
 mem_tot_CPU7: mem_tot_CPU7 = 0
 R16: - 3.05176e-05 mem_tot_CPU7 + normalized_mem_tot_CPU7 = 0
 rows_tot_CPU7: rows_tot_CPU7 = 0
 R18: - 0.0833333333333333 rows_tot_CPU7 + normalized_rows_tot_CPU7 = 0
 mem_tot_CPU21: mem_tot_CPU21 = 0
 R20: - 3.05176e-05 mem_tot_CPU21 + normalized_mem_tot_CPU21 = 0
 rows_tot_CPU21: rows_tot_CPU21 = 0
 R22: - 0.0833333333333333 rows_tot_CPU21 + normalized_rows_tot_CPU21 = 0
 mem_tot_CPU2: - mem[4,0] + mem_tot_CPU2 = 0
 R24: - 3.05176e-05 mem_tot_CPU2 + normalized_mem_tot_CPU2 = 0
 rows_tot_CPU2: - 3 frac[4,0] + rows_tot_CPU2 = 0
 R26: - 0.0833333333333333 rows_tot_CPU2 + normalized_rows_tot_CPU2 = 0
 mem_tot_CPU3: - mem[5,0] + mem_tot_CPU3 = 0
 R28: - 3.05176e-05 mem_tot_CPU3 + normalized_mem_tot_CPU3 = 0
 rows_tot_CPU3: - 3 frac[5,0] + rows_tot_CPU3 = 0
 R30: - 0.0833333333333333 rows_tot_CPU3 + normalized_rows_tot_CPU3 = 0
 mem_tot_CPU18: mem_tot_CPU18 = 0
 R32: - 3.05176e-05 mem_tot_CPU18 + normalized_mem_tot_CPU18 = 0
 rows_tot_CPU18: rows_tot_CPU18 = 0
 R34: - 0.0833333333333333 rows_tot_CPU18 + normalized_rows_tot_CPU18 = 0
 mem_tot_CPU5: mem_tot_CPU5 = 0
 R36: - 3.05176e-05 mem_tot_CPU5 + normalized_mem_tot_CPU5 = 0
 rows_tot_CPU5: rows_tot_CPU5 = 0
 R38: - 0.0833333333333333 rows_tot_CPU5 + normalized_rows_tot_CPU5 = 0
 row_capacity_CPU29: rows_tot_CPU29 <= 12
 row_capacity_CPU8: rows_tot_CPU8 <= 12
 row_capacity_CPU7: rows_tot_CPU7 <= 12
 row_capacity_CPU21: rows_tot_CPU21 <= 12
 row_capacity_CPU2: rows_tot_CPU2 <= 12
 row_capacity_CPU3: rows_tot_CPU3 <= 12
 row_capacity_CPU18: rows_tot_CPU18 <= 12
 row_capacity_CPU5: rows_tot_CPU5 <= 12
 ns_single_CPU29: 3.5 rows_tot_CPU29 - ns_single_CPU29 + pdt_m_rows_CPU29
   = 0
 capacity_cores_CPU29: cores_sketch_CPU29 + cores_dpdk_CPU29 <= 7
 ns_sketch_CPU29: - ns_single_CPU29 + 10.20408163265306 cores_sketch_CPU29
   >= 0
 ns_dpdk_CPU29: - 3.06122448979592 cores_dpdk_CPU29 <= -21.42857142857143
 ns_single_CPU8: 3.5 rows_tot_CPU8 - ns_single_CPU8 + pdt_m_rows_CPU8 = 0
 capacity_cores_CPU8: cores_sketch_CPU8 + cores_dpdk_CPU8 <= 7
 ns_sketch_CPU8: - ns_single_CPU8 + 10.20408163265306 cores_sketch_CPU8
   >= 0
 ns_dpdk_CPU8: - 3.06122448979592 cores_dpdk_CPU8 <= -21.42857142857143
 ns_single_CPU7: 3.5 rows_tot_CPU7 - ns_single_CPU7 + pdt_m_rows_CPU7 = 0
 capacity_cores_CPU7: cores_sketch_CPU7 + cores_dpdk_CPU7 <= 7
 ns_sketch_CPU7: - ns_single_CPU7 + 10.20408163265306 cores_sketch_CPU7
   >= 0
 ns_dpdk_CPU7: - 3.06122448979592 cores_dpdk_CPU7 <= -21.42857142857143
 ns_single_CPU21: 3.5 rows_tot_CPU21 - ns_single_CPU21 + pdt_m_rows_CPU21
   = 0
 capacity_cores_CPU21: cores_sketch_CPU21 + cores_dpdk_CPU21 <= 7
 ns_sketch_CPU21: - ns_single_CPU21 + 10.20408163265306 cores_sketch_CPU21
   >= 0
 ns_dpdk_CPU21: - 3.06122448979592 cores_dpdk_CPU21 <= -21.42857142857143
 ns_single_CPU2: 3.5 rows_tot_CPU2 - ns_single_CPU2 + pdt_m_rows_CPU2 = 0
 capacity_cores_CPU2: cores_sketch_CPU2 + cores_dpdk_CPU2 <= 7
 ns_sketch_CPU2: - ns_single_CPU2 + 10.20408163265306 cores_sketch_CPU2
   >= 0
 ns_dpdk_CPU2: - 3.06122448979592 cores_dpdk_CPU2 <= -21.42857142857143
 ns_single_CPU3: 3.5 rows_tot_CPU3 - ns_single_CPU3 + pdt_m_rows_CPU3 = 0
 capacity_cores_CPU3: cores_sketch_CPU3 + cores_dpdk_CPU3 <= 7
 ns_sketch_CPU3: - ns_single_CPU3 + 10.20408163265306 cores_sketch_CPU3
   >= 0
 ns_dpdk_CPU3: - 3.06122448979592 cores_dpdk_CPU3 <= -21.42857142857143
 ns_single_CPU18: 3.5 rows_tot_CPU18 - ns_single_CPU18 + pdt_m_rows_CPU18
   = 0
 capacity_cores_CPU18: cores_sketch_CPU18 + cores_dpdk_CPU18 <= 7
 ns_sketch_CPU18: - ns_single_CPU18 + 10.20408163265306 cores_sketch_CPU18
   >= 0
 ns_dpdk_CPU18: - 3.06122448979592 cores_dpdk_CPU18 <= -21.42857142857143
 ns_single_CPU5: 3.5 rows_tot_CPU5 - ns_single_CPU5 + pdt_m_rows_CPU5 = 0
 capacity_cores_CPU5: cores_sketch_CPU5 + cores_dpdk_CPU5 <= 7
 ns_sketch_CPU5: - ns_single_CPU5 + 10.20408163265306 cores_sketch_CPU5
   >= 0
 ns_dpdk_CPU5: - 3.06122448979592 cores_dpdk_CPU5 <= -21.42857142857143
 res_acc: 1.220703125e-04 mem_tot_CPU29 + 1.220703125e-04 mem_tot_CPU8
   + 1.220703125e-04 mem_tot_CPU7 + 1.220703125e-04 mem_tot_CPU21
   + 1.220703125e-04 mem_tot_CPU2 + 1.220703125e-04 mem_tot_CPU3
   + 1.220703125e-04 mem_tot_CPU18 + 1.220703125e-04 mem_tot_CPU5
   + 10 cores_sketch_CPU29 + 10 cores_dpdk_CPU29 + 10 cores_sketch_CPU8
   + 10 cores_dpdk_CPU8 + 10 cores_sketch_CPU7 + 10 cores_dpdk_CPU7
   + 10 cores_sketch_CPU21 + 10 cores_dpdk_CPU21 + 10 cores_sketch_CPU2
   + 10 cores_dpdk_CPU2 + 10 cores_sketch_CPU3 + 10 cores_dpdk_CPU3
   + 10 cores_sketch_CPU18 + 10 cores_dpdk_CPU18 + 10 cores_sketch_CPU5
   + 10 cores_dpdk_CPU5 - res = 0
 pdt_m_rows_CPU29: pdt_m_rows_CPU29 + [
   - rows_tot_CPU29 * m_access_time_CPU29 ] = 0
 pdt_m_rows_CPU8: pdt_m_rows_CPU8 + [ - rows_tot_CPU8 * m_access_time_CPU8
   ] = 0
 pdt_m_rows_CPU7: pdt_m_rows_CPU7 + [ - rows_tot_CPU7 * m_access_time_CPU7
   ] = 0
 pdt_m_rows_CPU21: pdt_m_rows_CPU21 + [
   - rows_tot_CPU21 * m_access_time_CPU21 ] = 0
 pdt_m_rows_CPU2: pdt_m_rows_CPU2 + [ - rows_tot_CPU2 * m_access_time_CPU2
   ] = 0
 pdt_m_rows_CPU3: pdt_m_rows_CPU3 + [ - rows_tot_CPU3 * m_access_time_CPU3
   ] = 0
 pdt_m_rows_CPU18: pdt_m_rows_CPU18 + [
   - rows_tot_CPU18 * m_access_time_CPU18 ] = 0
 pdt_m_rows_CPU5: pdt_m_rows_CPU5 + [ - rows_tot_CPU5 * m_access_time_CPU5
   ] = 0
Bounds
 frac[1,0] <= 1
 frac[0,0] <= 1
 frac[5,0] <= 1
 frac[4,0] <= 1
 mem_tot_CPU29 <= 32768
 normalized_mem_tot_CPU29 <= 1
 normalized_rows_tot_CPU29 <= 1
 mem_tot_CPU8 <= 32768
 normalized_mem_tot_CPU8 <= 1
 normalized_rows_tot_CPU8 <= 1
 mem_tot_CPU7 <= 32768
 normalized_mem_tot_CPU7 <= 1
 normalized_rows_tot_CPU7 <= 1
 mem_tot_CPU21 <= 32768
 normalized_mem_tot_CPU21 <= 1
 normalized_rows_tot_CPU21 <= 1
 mem_tot_CPU2 <= 32768
 normalized_mem_tot_CPU2 <= 1
 normalized_rows_tot_CPU2 <= 1
 mem_tot_CPU3 <= 32768
 normalized_mem_tot_CPU3 <= 1
 normalized_rows_tot_CPU3 <= 1
 mem_tot_CPU18 <= 32768
 normalized_mem_tot_CPU18 <= 1
 normalized_rows_tot_CPU18 <= 1
 mem_tot_CPU5 <= 32768
 normalized_mem_tot_CPU5 <= 1
 normalized_rows_tot_CPU5 <= 1
 cores_sketch_CPU29 <= 7
 1 <= cores_dpdk_CPU29 <= 7
 cores_sketch_CPU8 <= 7
 1 <= cores_dpdk_CPU8 <= 7
 cores_sketch_CPU7 <= 7
 1 <= cores_dpdk_CPU7 <= 7
 cores_sketch_CPU21 <= 7
 1 <= cores_dpdk_CPU21 <= 7
 cores_sketch_CPU2 <= 7
 1 <= cores_dpdk_CPU2 <= 7
 cores_sketch_CPU3 <= 7
 1 <= cores_dpdk_CPU3 <= 7
 cores_sketch_CPU18 <= 7
 1 <= cores_dpdk_CPU18 <= 7
 cores_sketch_CPU5 <= 7
 1 <= cores_dpdk_CPU5 <= 7
Generals
 cores_sketch_CPU29 cores_dpdk_CPU29 cores_sketch_CPU8 cores_dpdk_CPU8
 cores_sketch_CPU7 cores_dpdk_CPU7 cores_sketch_CPU21 cores_dpdk_CPU21
 cores_sketch_CPU2 cores_dpdk_CPU2 cores_sketch_CPU3 cores_dpdk_CPU3
 cores_sketch_CPU18 cores_dpdk_CPU18 cores_sketch_CPU5 cores_dpdk_CPU5
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU29 , normalized_mem_tot_CPU8 ,
   normalized_mem_tot_CPU7 , normalized_mem_tot_CPU21 ,
   normalized_mem_tot_CPU2 , normalized_mem_tot_CPU3 ,
   normalized_mem_tot_CPU18 , normalized_mem_tot_CPU5 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU29 ,
   normalized_rows_tot_CPU8 , normalized_rows_tot_CPU7 ,
   normalized_rows_tot_CPU21 , normalized_rows_tot_CPU2 ,
   normalized_rows_tot_CPU3 , normalized_rows_tot_CPU18 ,
   normalized_rows_tot_CPU5 )
 mem_access_time_CPU29: m_access_time_CPU29 = PWL ( mem_tot_CPU29 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU8: m_access_time_CPU8 = PWL ( mem_tot_CPU8 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU7: m_access_time_CPU7 = PWL ( mem_tot_CPU7 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU21: m_access_time_CPU21 = PWL ( mem_tot_CPU21 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU2: m_access_time_CPU2 = PWL ( mem_tot_CPU2 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU3: m_access_time_CPU3 = PWL ( mem_tot_CPU3 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU18: m_access_time_CPU18 = PWL ( mem_tot_CPU18 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU5: m_access_time_CPU5 = PWL ( mem_tot_CPU5 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_CPU_Cluster + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU7823
   + 0 m_access_time_CPU8428 + 0 m_access_time_CPU4047
   + 0 m_access_time_CPU7500 + 0 m_access_time_CPU7933
   + 0 m_access_time_CPU1986 + 0 m_access_time_CPU2620
   + 0 m_access_time_CPU84 + res
Subject To
 cov_0_2: frac[0,2] + frac[2,2] >= 0.2465914055135254
 cov_1_3: frac[2,3] + frac[0,3] >= 0.8
 cov_2_1: frac[1,1] >= 0.7
 cov_3_0: frac[0,0] + frac[7,0] >= 0.8
 cov_4_3: frac[0,3] + frac[5,3] >= 0.8
 cov_5_1: frac[3,1] + frac[5,1] >= 0.8
 cov_6_3: frac[1,3] + frac[6,3] >= 0.8
 cov_7_0: frac[5,0] >= 0.8116478513783827
 accuracy_0_0: - 509.68359375 frac[0,0] + mem[0,0] = 0
 accuracy_7_0: - 509.68359375 frac[7,0] + mem[7,0] = 0
 accuracy_1_3: - 509.68359375 frac[1,3] + mem[1,3] = 0
 accuracy_0_2: - 509.68359375 frac[0,2] + mem[0,2] = 0
 accuracy_3_1: - 509.68359375 frac[3,1] + mem[3,1] = 0
 accuracy_6_3: - 509.68359375 frac[6,3] + mem[6,3] = 0
 accuracy_2_3: - 509.68359375 frac[2,3] + mem[2,3] = 0
 accuracy_5_0: - 509.68359375 frac[5,0] + mem[5,0] = 0
 accuracy_2_2: - 509.68359375 frac[2,2] + mem[2,2] = 0
 accuracy_5_1: - 509.68359375 frac[5,1] + mem[5,1] = 0
 accuracy_0_3: - 509.68359375 frac[0,3] + mem[0,3] = 0
 accuracy_1_1: - 509.68359375 frac[1,1] + mem[1,1] = 0
 accuracy_5_3: - 509.68359375 frac[5,3] + mem[5,3] = 0
 mem_tot_CPU7823: - mem[0,0] - mem[0,2] - mem[0,3] + mem_tot_CPU7823 = 0
 R22: - 3.05176e-05 mem_tot_CPU7823 + normalized_mem_tot_CPU7823 = 0
 rows_tot_CPU7823: - 3 frac[0,0] - 3 frac[0,2] - 3 frac[0,3]
   + rows_tot_CPU7823 = 0
 R24: - 0.0833333333333333 rows_tot_CPU7823 + normalized_rows_tot_CPU7823
   = 0
 mem_tot_CPU8428: - mem[1,3] - mem[1,1] + mem_tot_CPU8428 = 0
 R26: - 3.05176e-05 mem_tot_CPU8428 + normalized_mem_tot_CPU8428 = 0
 rows_tot_CPU8428: - 3 frac[1,3] - 3 frac[1,1] + rows_tot_CPU8428 = 0
 R28: - 0.0833333333333333 rows_tot_CPU8428 + normalized_rows_tot_CPU8428
   = 0
 mem_tot_CPU4047: - mem[2,3] - mem[2,2] + mem_tot_CPU4047 = 0
 R30: - 3.05176e-05 mem_tot_CPU4047 + normalized_mem_tot_CPU4047 = 0
 rows_tot_CPU4047: - 3 frac[2,3] - 3 frac[2,2] + rows_tot_CPU4047 = 0
 R32: - 0.0833333333333333 rows_tot_CPU4047 + normalized_rows_tot_CPU4047
   = 0
 mem_tot_CPU7500: - mem[3,1] + mem_tot_CPU7500 = 0
 R34: - 3.05176e-05 mem_tot_CPU7500 + normalized_mem_tot_CPU7500 = 0
 rows_tot_CPU7500: - 3 frac[3,1] + rows_tot_CPU7500 = 0
 R36: - 0.0833333333333333 rows_tot_CPU7500 + normalized_rows_tot_CPU7500
   = 0
 mem_tot_CPU7933: mem_tot_CPU7933 = 0
 R38: - 3.05176e-05 mem_tot_CPU7933 + normalized_mem_tot_CPU7933 = 0
 rows_tot_CPU7933: rows_tot_CPU7933 = 0
 R40: - 0.0833333333333333 rows_tot_CPU7933 + normalized_rows_tot_CPU7933
   = 0
 mem_tot_CPU1986: - mem[5,0] - mem[5,1] - mem[5,3] + mem_tot_CPU1986 = 0
 R42: - 3.05176e-05 mem_tot_CPU1986 + normalized_mem_tot_CPU1986 = 0
 rows_tot_CPU1986: - 3 frac[5,0] - 3 frac[5,1] - 3 frac[5,3]
   + rows_tot_CPU1986 = 0
 R44: - 0.0833333333333333 rows_tot_CPU1986 + normalized_rows_tot_CPU1986
   = 0
 mem_tot_CPU2620: - mem[6,3] + mem_tot_CPU2620 = 0
 R46: - 3.05176e-05 mem_tot_CPU2620 + normalized_mem_tot_CPU2620 = 0
 rows_tot_CPU2620: - 3 frac[6,3] + rows_tot_CPU2620 = 0
 R48: - 0.0833333333333333 rows_tot_CPU2620 + normalized_rows_tot_CPU2620
   = 0
 mem_tot_CPU84: - mem[7,0] + mem_tot_CPU84 = 0
 R50: - 3.05176e-05 mem_tot_CPU84 + normalized_mem_tot_CPU84 = 0
 rows_tot_CPU84: - 3 frac[7,0] + rows_tot_CPU84 = 0
 R52: - 0.0833333333333333 rows_tot_CPU84 + normalized_rows_tot_CPU84 = 0
 row_capacity_CPU7823: rows_tot_CPU7823 <= 12
 row_capacity_CPU8428: rows_tot_CPU8428 <= 12
 row_capacity_CPU4047: rows_tot_CPU4047 <= 12
 row_capacity_CPU7500: rows_tot_CPU7500 <= 12
 row_capacity_CPU7933: rows_tot_CPU7933 <= 12
 row_capacity_CPU1986: rows_tot_CPU1986 <= 12
 row_capacity_CPU2620: rows_tot_CPU2620 <= 12
 row_capacity_CPU84: rows_tot_CPU84 <= 12
 ns_single_CPU7823: 3.5 rows_tot_CPU7823 - ns_single_CPU7823
   + pdt_m_rows_CPU7823 = 0
 capacity_cores_CPU7823: cores_sketch_CPU7823 + cores_dpdk_CPU7823 <= 7
 ns_sketch_CPU7823: - ns_single_CPU7823
   + 10.71428571428572 cores_sketch_CPU7823 >= 0
 ns_dpdk_CPU7823: - 3.571428571428572 cores_dpdk_CPU7823
   <= -21.42857142857143
 ns_single_CPU8428: 3.5 rows_tot_CPU8428 - ns_single_CPU8428
   + pdt_m_rows_CPU8428 = 0
 capacity_cores_CPU8428: cores_sketch_CPU8428 + cores_dpdk_CPU8428 <= 7
 ns_sketch_CPU8428: - ns_single_CPU8428
   + 10.71428571428572 cores_sketch_CPU8428 >= 0
 ns_dpdk_CPU8428: - 3.571428571428572 cores_dpdk_CPU8428
   <= -21.42857142857143
 ns_single_CPU4047: 3.5 rows_tot_CPU4047 - ns_single_CPU4047
   + pdt_m_rows_CPU4047 = 0
 capacity_cores_CPU4047: cores_sketch_CPU4047 + cores_dpdk_CPU4047 <= 7
 ns_sketch_CPU4047: - ns_single_CPU4047
   + 10.71428571428572 cores_sketch_CPU4047 >= 0
 ns_dpdk_CPU4047: - 3.571428571428572 cores_dpdk_CPU4047
   <= -21.42857142857143
 ns_single_CPU7500: 3.5 rows_tot_CPU7500 - ns_single_CPU7500
   + pdt_m_rows_CPU7500 = 0
 capacity_cores_CPU7500: cores_sketch_CPU7500 + cores_dpdk_CPU7500 <= 7
 ns_sketch_CPU7500: - ns_single_CPU7500
   + 10.71428571428572 cores_sketch_CPU7500 >= 0
 ns_dpdk_CPU7500: - 3.571428571428572 cores_dpdk_CPU7500
   <= -21.42857142857143
 ns_single_CPU7933: 3.5 rows_tot_CPU7933 - ns_single_CPU7933
   + pdt_m_rows_CPU7933 = 0
 capacity_cores_CPU7933: cores_sketch_CPU7933 + cores_dpdk_CPU7933 <= 7
 ns_sketch_CPU7933: - ns_single_CPU7933
   + 10.71428571428572 cores_sketch_CPU7933 >= 0
 ns_dpdk_CPU7933: - 3.571428571428572 cores_dpdk_CPU7933
   <= -21.42857142857143
 ns_single_CPU1986: 3.5 rows_tot_CPU1986 - ns_single_CPU1986
   + pdt_m_rows_CPU1986 = 0
 capacity_cores_CPU1986: cores_sketch_CPU1986 + cores_dpdk_CPU1986 <= 7
 ns_sketch_CPU1986: - ns_single_CPU1986
   + 10.71428571428572 cores_sketch_CPU1986 >= 0
 ns_dpdk_CPU1986: - 3.571428571428572 cores_dpdk_CPU1986
   <= -21.42857142857143
 ns_single_CPU2620: 3.5 rows_tot_CPU2620 - ns_single_CPU2620
   + pdt_m_rows_CPU2620 = 0
 capacity_cores_CPU2620: cores_sketch_CPU2620 + cores_dpdk_CPU2620 <= 7
 ns_sketch_CPU2620: - ns_single_CPU2620
   + 10.71428571428572 cores_sketch_CPU2620 >= 0
 ns_dpdk_CPU2620: - 3.571428571428572 cores_dpdk_CPU2620
   <= -21.42857142857143
 ns_single_CPU84: 3.5 rows_tot_CPU84 - ns_single_CPU84 + pdt_m_rows_CPU84
   = 0
 capacity_cores_CPU84: cores_sketch_CPU84 + cores_dpdk_CPU84 <= 7
 ns_sketch_CPU84: - ns_single_CPU84 + 10.71428571428572 cores_sketch_CPU84
   >= 0
 ns_dpdk_CPU84: - 3.571428571428572 cores_dpdk_CPU84 <= -21.42857142857143
 res_acc: 1.220703125e-04 mem_tot_CPU7823 + 1.220703125e-04 mem_tot_CPU8428
   + 1.220703125e-04 mem_tot_CPU4047 + 1.220703125e-04 mem_tot_CPU7500
   + 1.220703125e-04 mem_tot_CPU7933 + 1.220703125e-04 mem_tot_CPU1986
   + 1.220703125e-04 mem_tot_CPU2620 + 1.220703125e-04 mem_tot_CPU84
   + 10 cores_sketch_CPU7823 + 10 cores_dpdk_CPU7823
   + 10 cores_sketch_CPU8428 + 10 cores_dpdk_CPU8428
   + 10 cores_sketch_CPU4047 + 10 cores_dpdk_CPU4047
   + 10 cores_sketch_CPU7500 + 10 cores_dpdk_CPU7500
   + 10 cores_sketch_CPU7933 + 10 cores_dpdk_CPU7933
   + 10 cores_sketch_CPU1986 + 10 cores_dpdk_CPU1986
   + 10 cores_sketch_CPU2620 + 10 cores_dpdk_CPU2620
   + 10 cores_sketch_CPU84 + 10 cores_dpdk_CPU84 - res = 0
 pdt_m_rows_CPU7823: pdt_m_rows_CPU7823 + [
   - rows_tot_CPU7823 * m_access_time_CPU7823 ] = 0
 pdt_m_rows_CPU8428: pdt_m_rows_CPU8428 + [
   - rows_tot_CPU8428 * m_access_time_CPU8428 ] = 0
 pdt_m_rows_CPU4047: pdt_m_rows_CPU4047 + [
   - rows_tot_CPU4047 * m_access_time_CPU4047 ] = 0
 pdt_m_rows_CPU7500: pdt_m_rows_CPU7500 + [
   - rows_tot_CPU7500 * m_access_time_CPU7500 ] = 0
 pdt_m_rows_CPU7933: pdt_m_rows_CPU7933 + [
   - rows_tot_CPU7933 * m_access_time_CPU7933 ] = 0
 pdt_m_rows_CPU1986: pdt_m_rows_CPU1986 + [
   - rows_tot_CPU1986 * m_access_time_CPU1986 ] = 0
 pdt_m_rows_CPU2620: pdt_m_rows_CPU2620 + [
   - rows_tot_CPU2620 * m_access_time_CPU2620 ] = 0
 pdt_m_rows_CPU84: pdt_m_rows_CPU84 + [
   - rows_tot_CPU84 * m_access_time_CPU84 ] = 0
Bounds
 frac[0,0] <= 1
 frac[7,0] <= 1
 frac[1,3] <= 1
 frac[0,2] <= 1
 frac[3,1] <= 1
 frac[6,3] <= 1
 frac[2,3] <= 1
 frac[5,0] <= 1
 frac[2,2] <= 1
 frac[5,1] <= 1
 frac[0,3] <= 1
 frac[1,1] <= 1
 frac[5,3] <= 1
 mem_tot_CPU7823 <= 32768
 normalized_mem_tot_CPU7823 <= 1
 normalized_rows_tot_CPU7823 <= 1
 mem_tot_CPU8428 <= 32768
 normalized_mem_tot_CPU8428 <= 1
 normalized_rows_tot_CPU8428 <= 1
 mem_tot_CPU4047 <= 32768
 normalized_mem_tot_CPU4047 <= 1
 normalized_rows_tot_CPU4047 <= 1
 mem_tot_CPU7500 <= 32768
 normalized_mem_tot_CPU7500 <= 1
 normalized_rows_tot_CPU7500 <= 1
 mem_tot_CPU7933 <= 32768
 normalized_mem_tot_CPU7933 <= 1
 normalized_rows_tot_CPU7933 <= 1
 mem_tot_CPU1986 <= 32768
 normalized_mem_tot_CPU1986 <= 1
 normalized_rows_tot_CPU1986 <= 1
 mem_tot_CPU2620 <= 32768
 normalized_mem_tot_CPU2620 <= 1
 normalized_rows_tot_CPU2620 <= 1
 mem_tot_CPU84 <= 32768
 normalized_mem_tot_CPU84 <= 1
 normalized_rows_tot_CPU84 <= 1
 cores_sketch_CPU7823 <= 7
 1 <= cores_dpdk_CPU7823 <= 7
 cores_sketch_CPU8428 <= 7
 1 <= cores_dpdk_CPU8428 <= 7
 cores_sketch_CPU4047 <= 7
 1 <= cores_dpdk_CPU4047 <= 7
 cores_sketch_CPU7500 <= 7
 1 <= cores_dpdk_CPU7500 <= 7
 cores_sketch_CPU7933 <= 7
 1 <= cores_dpdk_CPU7933 <= 7
 cores_sketch_CPU1986 <= 7
 1 <= cores_dpdk_CPU1986 <= 7
 cores_sketch_CPU2620 <= 7
 1 <= cores_dpdk_CPU2620 <= 7
 cores_sketch_CPU84 <= 7
 1 <= cores_dpdk_CPU84 <= 7
Generals
 cores_sketch_CPU7823 cores_dpdk_CPU7823 cores_sketch_CPU8428
 cores_dpdk_CPU8428 cores_sketch_CPU4047 cores_dpdk_CPU4047
 cores_sketch_CPU7500 cores_dpdk_CPU7500 cores_sketch_CPU7933
 cores_dpdk_CPU7933 cores_sketch_CPU1986 cores_dpdk_CPU1986
 cores_sketch_CPU2620 cores_dpdk_CPU2620 cores_sketch_CPU84
 cores_dpdk_CPU84
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU7823 , normalized_mem_tot_CPU8428 ,
   normalized_mem_tot_CPU4047 , normalized_mem_tot_CPU7500 ,
   normalized_mem_tot_CPU7933 , normalized_mem_tot_CPU1986 ,
   normalized_mem_tot_CPU2620 , normalized_mem_tot_CPU84 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX (
   normalized_rows_tot_CPU7823 , normalized_rows_tot_CPU8428 ,
   normalized_rows_tot_CPU4047 , normalized_rows_tot_CPU7500 ,
   normalized_rows_tot_CPU7933 , normalized_rows_tot_CPU1986 ,
   normalized_rows_tot_CPU2620 , normalized_rows_tot_CPU84 )
 mem_access_time_CPU7823:
   m_access_time_CPU7823 = PWL ( mem_tot_CPU7823 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU8428:
   m_access_time_CPU8428 = PWL ( mem_tot_CPU8428 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU4047:
   m_access_time_CPU4047 = PWL ( mem_tot_CPU4047 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU7500:
   m_access_time_CPU7500 = PWL ( mem_tot_CPU7500 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU7933:
   m_access_time_CPU7933 = PWL ( mem_tot_CPU7933 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1986:
   m_access_time_CPU1986 = PWL ( mem_tot_CPU1986 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU2620:
   m_access_time_CPU2620 = PWL ( mem_tot_CPU2620 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU84: m_access_time_CPU84 = PWL ( mem_tot_CPU84 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

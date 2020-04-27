\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_CPU_Cluster + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU757
   + 0 m_access_time_CPU750 + 0 m_access_time_CPU2138
   + 0 m_access_time_CPU6861 + 0 m_access_time_CPU7530
   + 0 m_access_time_CPU2031 + 0 m_access_time_CPU1968
   + 0 m_access_time_CPU1203 + res
Subject To
 cov_0_2: frac[0,2] + frac[7,2] >= 1
 cov_1_1: frac[6,1] + frac[3,1] >= 0.9
 cov_2_3: frac[1,3] + frac[5,3] >= 1
 cov_3_0: frac[4,0] >= 0.5
 cov_4_3: frac[7,3] + frac[1,3] >= 0.8
 cov_5_2: frac[2,2] + frac[7,2] >= 0.8
 cov_6_1: frac[6,1] + frac[3,1] >= 0.8
 cov_7_1: frac[2,1] + frac[4,1] >= 0.8
 accuracy_7_3: - 509.68359375 frac[7,3] + mem[7,3] = 0
 accuracy_1_3: - 509.68359375 frac[1,3] + mem[1,3] = 0
 accuracy_6_1: - 509.68359375 frac[6,1] + mem[6,1] = 0
 accuracy_3_1: - 509.68359375 frac[3,1] + mem[3,1] = 0
 accuracy_4_0: - 509.68359375 frac[4,0] + mem[4,0] = 0
 accuracy_2_1: - 509.68359375 frac[2,1] + mem[2,1] = 0
 accuracy_2_2: - 509.68359375 frac[2,2] + mem[2,2] = 0
 accuracy_4_1: - 509.68359375 frac[4,1] + mem[4,1] = 0
 accuracy_0_2: - 509.68359375 frac[0,2] + mem[0,2] = 0
 accuracy_7_2: - 509.68359375 frac[7,2] + mem[7,2] = 0
 accuracy_5_3: - 509.68359375 frac[5,3] + mem[5,3] = 0
 mem_tot_CPU757: - mem[0,2] + mem_tot_CPU757 = 0
 R20: - 3.05176e-05 mem_tot_CPU757 + normalized_mem_tot_CPU757 = 0
 rows_tot_CPU757: - 3 frac[0,2] + rows_tot_CPU757 = 0
 R22: - 0.0833333333333333 rows_tot_CPU757 + normalized_rows_tot_CPU757
   = 0
 mem_tot_CPU750: - mem[1,3] + mem_tot_CPU750 = 0
 R24: - 3.05176e-05 mem_tot_CPU750 + normalized_mem_tot_CPU750 = 0
 rows_tot_CPU750: - 3 frac[1,3] + rows_tot_CPU750 = 0
 R26: - 0.0833333333333333 rows_tot_CPU750 + normalized_rows_tot_CPU750
   = 0
 mem_tot_CPU2138: - mem[2,1] - mem[2,2] + mem_tot_CPU2138 = 0
 R28: - 3.05176e-05 mem_tot_CPU2138 + normalized_mem_tot_CPU2138 = 0
 rows_tot_CPU2138: - 3 frac[2,1] - 3 frac[2,2] + rows_tot_CPU2138 = 0
 R30: - 0.0833333333333333 rows_tot_CPU2138 + normalized_rows_tot_CPU2138
   = 0
 mem_tot_CPU6861: - mem[3,1] + mem_tot_CPU6861 = 0
 R32: - 3.05176e-05 mem_tot_CPU6861 + normalized_mem_tot_CPU6861 = 0
 rows_tot_CPU6861: - 3 frac[3,1] + rows_tot_CPU6861 = 0
 R34: - 0.0833333333333333 rows_tot_CPU6861 + normalized_rows_tot_CPU6861
   = 0
 mem_tot_CPU7530: - mem[4,0] - mem[4,1] + mem_tot_CPU7530 = 0
 R36: - 3.05176e-05 mem_tot_CPU7530 + normalized_mem_tot_CPU7530 = 0
 rows_tot_CPU7530: - 3 frac[4,0] - 3 frac[4,1] + rows_tot_CPU7530 = 0
 R38: - 0.0833333333333333 rows_tot_CPU7530 + normalized_rows_tot_CPU7530
   = 0
 mem_tot_CPU2031: - mem[5,3] + mem_tot_CPU2031 = 0
 R40: - 3.05176e-05 mem_tot_CPU2031 + normalized_mem_tot_CPU2031 = 0
 rows_tot_CPU2031: - 3 frac[5,3] + rows_tot_CPU2031 = 0
 R42: - 0.0833333333333333 rows_tot_CPU2031 + normalized_rows_tot_CPU2031
   = 0
 mem_tot_CPU1968: - mem[6,1] + mem_tot_CPU1968 = 0
 R44: - 3.05176e-05 mem_tot_CPU1968 + normalized_mem_tot_CPU1968 = 0
 rows_tot_CPU1968: - 3 frac[6,1] + rows_tot_CPU1968 = 0
 R46: - 0.0833333333333333 rows_tot_CPU1968 + normalized_rows_tot_CPU1968
   = 0
 mem_tot_CPU1203: - mem[7,3] - mem[7,2] + mem_tot_CPU1203 = 0
 R48: - 3.05176e-05 mem_tot_CPU1203 + normalized_mem_tot_CPU1203 = 0
 rows_tot_CPU1203: - 3 frac[7,3] - 3 frac[7,2] + rows_tot_CPU1203 = 0
 R50: - 0.0833333333333333 rows_tot_CPU1203 + normalized_rows_tot_CPU1203
   = 0
 row_capacity_CPU757: rows_tot_CPU757 <= 12
 row_capacity_CPU750: rows_tot_CPU750 <= 12
 row_capacity_CPU2138: rows_tot_CPU2138 <= 12
 row_capacity_CPU6861: rows_tot_CPU6861 <= 12
 row_capacity_CPU7530: rows_tot_CPU7530 <= 12
 row_capacity_CPU2031: rows_tot_CPU2031 <= 12
 row_capacity_CPU1968: rows_tot_CPU1968 <= 12
 row_capacity_CPU1203: rows_tot_CPU1203 <= 12
 ns_single_CPU757: 3.5 rows_tot_CPU757 - ns_single_CPU757
   + pdt_m_rows_CPU757 = 0
 capacity_cores_CPU757: cores_sketch_CPU757 + cores_dpdk_CPU757 <= 7
 ns_sketch_CPU757: - ns_single_CPU757
   + 11.42857142857143 cores_sketch_CPU757 >= 0
 ns_dpdk_CPU757: - 4.285714285714287 cores_dpdk_CPU757
   <= -21.42857142857143
 ns_single_CPU750: 3.5 rows_tot_CPU750 - ns_single_CPU750
   + pdt_m_rows_CPU750 = 0
 capacity_cores_CPU750: cores_sketch_CPU750 + cores_dpdk_CPU750 <= 7
 ns_sketch_CPU750: - ns_single_CPU750
   + 11.42857142857143 cores_sketch_CPU750 >= 0
 ns_dpdk_CPU750: - 4.285714285714287 cores_dpdk_CPU750
   <= -21.42857142857143
 ns_single_CPU2138: 3.5 rows_tot_CPU2138 - ns_single_CPU2138
   + pdt_m_rows_CPU2138 = 0
 capacity_cores_CPU2138: cores_sketch_CPU2138 + cores_dpdk_CPU2138 <= 7
 ns_sketch_CPU2138: - ns_single_CPU2138
   + 11.42857142857143 cores_sketch_CPU2138 >= 0
 ns_dpdk_CPU2138: - 4.285714285714287 cores_dpdk_CPU2138
   <= -21.42857142857143
 ns_single_CPU6861: 3.5 rows_tot_CPU6861 - ns_single_CPU6861
   + pdt_m_rows_CPU6861 = 0
 capacity_cores_CPU6861: cores_sketch_CPU6861 + cores_dpdk_CPU6861 <= 7
 ns_sketch_CPU6861: - ns_single_CPU6861
   + 11.42857142857143 cores_sketch_CPU6861 >= 0
 ns_dpdk_CPU6861: - 4.285714285714287 cores_dpdk_CPU6861
   <= -21.42857142857143
 ns_single_CPU7530: 3.5 rows_tot_CPU7530 - ns_single_CPU7530
   + pdt_m_rows_CPU7530 = 0
 capacity_cores_CPU7530: cores_sketch_CPU7530 + cores_dpdk_CPU7530 <= 7
 ns_sketch_CPU7530: - ns_single_CPU7530
   + 11.42857142857143 cores_sketch_CPU7530 >= 0
 ns_dpdk_CPU7530: - 4.285714285714287 cores_dpdk_CPU7530
   <= -21.42857142857143
 ns_single_CPU2031: 3.5 rows_tot_CPU2031 - ns_single_CPU2031
   + pdt_m_rows_CPU2031 = 0
 capacity_cores_CPU2031: cores_sketch_CPU2031 + cores_dpdk_CPU2031 <= 7
 ns_sketch_CPU2031: - ns_single_CPU2031
   + 11.42857142857143 cores_sketch_CPU2031 >= 0
 ns_dpdk_CPU2031: - 4.285714285714287 cores_dpdk_CPU2031
   <= -21.42857142857143
 ns_single_CPU1968: 3.5 rows_tot_CPU1968 - ns_single_CPU1968
   + pdt_m_rows_CPU1968 = 0
 capacity_cores_CPU1968: cores_sketch_CPU1968 + cores_dpdk_CPU1968 <= 7
 ns_sketch_CPU1968: - ns_single_CPU1968
   + 11.42857142857143 cores_sketch_CPU1968 >= 0
 ns_dpdk_CPU1968: - 4.285714285714287 cores_dpdk_CPU1968
   <= -21.42857142857143
 ns_single_CPU1203: 3.5 rows_tot_CPU1203 - ns_single_CPU1203
   + pdt_m_rows_CPU1203 = 0
 capacity_cores_CPU1203: cores_sketch_CPU1203 + cores_dpdk_CPU1203 <= 7
 ns_sketch_CPU1203: - ns_single_CPU1203
   + 11.42857142857143 cores_sketch_CPU1203 >= 0
 ns_dpdk_CPU1203: - 4.285714285714287 cores_dpdk_CPU1203
   <= -21.42857142857143
 res_acc: 1.220703125e-04 mem_tot_CPU757 + 1.220703125e-04 mem_tot_CPU750
   + 1.220703125e-04 mem_tot_CPU2138 + 1.220703125e-04 mem_tot_CPU6861
   + 1.220703125e-04 mem_tot_CPU7530 + 1.220703125e-04 mem_tot_CPU2031
   + 1.220703125e-04 mem_tot_CPU1968 + 1.220703125e-04 mem_tot_CPU1203
   + 10 cores_sketch_CPU757 + 10 cores_dpdk_CPU757 + 10 cores_sketch_CPU750
   + 10 cores_dpdk_CPU750 + 10 cores_sketch_CPU2138 + 10 cores_dpdk_CPU2138
   + 10 cores_sketch_CPU6861 + 10 cores_dpdk_CPU6861
   + 10 cores_sketch_CPU7530 + 10 cores_dpdk_CPU7530
   + 10 cores_sketch_CPU2031 + 10 cores_dpdk_CPU2031
   + 10 cores_sketch_CPU1968 + 10 cores_dpdk_CPU1968
   + 10 cores_sketch_CPU1203 + 10 cores_dpdk_CPU1203 - res = 0
 pdt_m_rows_CPU757: pdt_m_rows_CPU757 + [
   - rows_tot_CPU757 * m_access_time_CPU757 ] = 0
 pdt_m_rows_CPU750: pdt_m_rows_CPU750 + [
   - rows_tot_CPU750 * m_access_time_CPU750 ] = 0
 pdt_m_rows_CPU2138: pdt_m_rows_CPU2138 + [
   - rows_tot_CPU2138 * m_access_time_CPU2138 ] = 0
 pdt_m_rows_CPU6861: pdt_m_rows_CPU6861 + [
   - rows_tot_CPU6861 * m_access_time_CPU6861 ] = 0
 pdt_m_rows_CPU7530: pdt_m_rows_CPU7530 + [
   - rows_tot_CPU7530 * m_access_time_CPU7530 ] = 0
 pdt_m_rows_CPU2031: pdt_m_rows_CPU2031 + [
   - rows_tot_CPU2031 * m_access_time_CPU2031 ] = 0
 pdt_m_rows_CPU1968: pdt_m_rows_CPU1968 + [
   - rows_tot_CPU1968 * m_access_time_CPU1968 ] = 0
 pdt_m_rows_CPU1203: pdt_m_rows_CPU1203 + [
   - rows_tot_CPU1203 * m_access_time_CPU1203 ] = 0
Bounds
 frac[7,3] <= 1
 frac[1,3] <= 1
 frac[6,1] <= 1
 frac[3,1] <= 1
 frac[4,0] <= 1
 frac[2,1] <= 1
 frac[2,2] <= 1
 frac[4,1] <= 1
 frac[0,2] <= 1
 frac[7,2] <= 1
 frac[5,3] <= 1
 mem_tot_CPU757 <= 32768
 normalized_mem_tot_CPU757 <= 1
 normalized_rows_tot_CPU757 <= 1
 mem_tot_CPU750 <= 32768
 normalized_mem_tot_CPU750 <= 1
 normalized_rows_tot_CPU750 <= 1
 mem_tot_CPU2138 <= 32768
 normalized_mem_tot_CPU2138 <= 1
 normalized_rows_tot_CPU2138 <= 1
 mem_tot_CPU6861 <= 32768
 normalized_mem_tot_CPU6861 <= 1
 normalized_rows_tot_CPU6861 <= 1
 mem_tot_CPU7530 <= 32768
 normalized_mem_tot_CPU7530 <= 1
 normalized_rows_tot_CPU7530 <= 1
 mem_tot_CPU2031 <= 32768
 normalized_mem_tot_CPU2031 <= 1
 normalized_rows_tot_CPU2031 <= 1
 mem_tot_CPU1968 <= 32768
 normalized_mem_tot_CPU1968 <= 1
 normalized_rows_tot_CPU1968 <= 1
 mem_tot_CPU1203 <= 32768
 normalized_mem_tot_CPU1203 <= 1
 normalized_rows_tot_CPU1203 <= 1
 cores_sketch_CPU757 <= 7
 1 <= cores_dpdk_CPU757 <= 7
 cores_sketch_CPU750 <= 7
 1 <= cores_dpdk_CPU750 <= 7
 cores_sketch_CPU2138 <= 7
 1 <= cores_dpdk_CPU2138 <= 7
 cores_sketch_CPU6861 <= 7
 1 <= cores_dpdk_CPU6861 <= 7
 cores_sketch_CPU7530 <= 7
 1 <= cores_dpdk_CPU7530 <= 7
 cores_sketch_CPU2031 <= 7
 1 <= cores_dpdk_CPU2031 <= 7
 cores_sketch_CPU1968 <= 7
 1 <= cores_dpdk_CPU1968 <= 7
 cores_sketch_CPU1203 <= 7
 1 <= cores_dpdk_CPU1203 <= 7
Generals
 cores_sketch_CPU757 cores_dpdk_CPU757 cores_sketch_CPU750
 cores_dpdk_CPU750 cores_sketch_CPU2138 cores_dpdk_CPU2138
 cores_sketch_CPU6861 cores_dpdk_CPU6861 cores_sketch_CPU7530
 cores_dpdk_CPU7530 cores_sketch_CPU2031 cores_dpdk_CPU2031
 cores_sketch_CPU1968 cores_dpdk_CPU1968 cores_sketch_CPU1203
 cores_dpdk_CPU1203
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU757 , normalized_mem_tot_CPU750 ,
   normalized_mem_tot_CPU2138 , normalized_mem_tot_CPU6861 ,
   normalized_mem_tot_CPU7530 , normalized_mem_tot_CPU2031 ,
   normalized_mem_tot_CPU1968 , normalized_mem_tot_CPU1203 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX (
   normalized_rows_tot_CPU757 , normalized_rows_tot_CPU750 ,
   normalized_rows_tot_CPU2138 , normalized_rows_tot_CPU6861 ,
   normalized_rows_tot_CPU7530 , normalized_rows_tot_CPU2031 ,
   normalized_rows_tot_CPU1968 , normalized_rows_tot_CPU1203 )
 mem_access_time_CPU757: m_access_time_CPU757 = PWL ( mem_tot_CPU757 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU750: m_access_time_CPU750 = PWL ( mem_tot_CPU750 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU2138:
   m_access_time_CPU2138 = PWL ( mem_tot_CPU2138 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU6861:
   m_access_time_CPU6861 = PWL ( mem_tot_CPU6861 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU7530:
   m_access_time_CPU7530 = PWL ( mem_tot_CPU7530 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU2031:
   m_access_time_CPU2031 = PWL ( mem_tot_CPU2031 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1968:
   m_access_time_CPU1968 = PWL ( mem_tot_CPU1968 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1203:
   m_access_time_CPU1203 = PWL ( mem_tot_CPU1203 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

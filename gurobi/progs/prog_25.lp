\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_CPU_Cluster + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU9515
   + 0 m_access_time_CPU1779 + 0 m_access_time_CPU35
   + 0 m_access_time_CPU1079 + 0 m_access_time_CPU8799
   + 0 m_access_time_CPU2019 + 0 m_access_time_CPU1731
   + 0 m_access_time_CPU1699 + res
Subject To
 cov_0_2: frac[6,2] + frac[4,2] >= 0.9
 cov_1_3: frac[3,3] >= 0.8
 cov_2_0: frac[0,0] + frac[3,0] >= 0.8
 cov_3_2: frac[1,2] + frac[3,2] >= 0.8
 cov_4_1: frac[7,1] + frac[2,1] >= 0.9
 cov_5_3: frac[1,3] + frac[5,3] >= 0.8
 cov_6_1: frac[0,1] + frac[5,1] >= 0.9
 cov_7_2: frac[4,2] + frac[0,2] >= 0.9
 accuracy_1_2: - 509.68359375 frac[1,2] + mem[1,2] = 0
 accuracy_0_1: - 509.68359375 frac[0,1] + mem[0,1] = 0
 accuracy_3_2: - 509.68359375 frac[3,2] + mem[3,2] = 0
 accuracy_0_0: - 509.68359375 frac[0,0] + mem[0,0] = 0
 accuracy_1_3: - 509.68359375 frac[1,3] + mem[1,3] = 0
 accuracy_3_3: - 509.68359375 frac[3,3] + mem[3,3] = 0
 accuracy_5_1: - 509.68359375 frac[5,1] + mem[5,1] = 0
 accuracy_7_1: - 509.68359375 frac[7,1] + mem[7,1] = 0
 accuracy_3_0: - 509.68359375 frac[3,0] + mem[3,0] = 0
 accuracy_2_1: - 509.68359375 frac[2,1] + mem[2,1] = 0
 accuracy_6_2: - 509.68359375 frac[6,2] + mem[6,2] = 0
 accuracy_4_2: - 509.68359375 frac[4,2] + mem[4,2] = 0
 accuracy_0_2: - 509.68359375 frac[0,2] + mem[0,2] = 0
 accuracy_5_3: - 509.68359375 frac[5,3] + mem[5,3] = 0
 mem_tot_CPU9515: - mem[0,1] - mem[0,0] - mem[0,2] + mem_tot_CPU9515 = 0
 R23: - 3.05176e-05 mem_tot_CPU9515 + normalized_mem_tot_CPU9515 = 0
 rows_tot_CPU9515: - 3 frac[0,1] - 3 frac[0,0] - 3 frac[0,2]
   + rows_tot_CPU9515 = 0
 R25: - 0.0833333333333333 rows_tot_CPU9515 + normalized_rows_tot_CPU9515
   = 0
 mem_tot_CPU1779: - mem[1,2] - mem[1,3] + mem_tot_CPU1779 = 0
 R27: - 3.05176e-05 mem_tot_CPU1779 + normalized_mem_tot_CPU1779 = 0
 rows_tot_CPU1779: - 3 frac[1,2] - 3 frac[1,3] + rows_tot_CPU1779 = 0
 R29: - 0.0833333333333333 rows_tot_CPU1779 + normalized_rows_tot_CPU1779
   = 0
 mem_tot_CPU35: - mem[2,1] + mem_tot_CPU35 = 0
 R31: - 3.05176e-05 mem_tot_CPU35 + normalized_mem_tot_CPU35 = 0
 rows_tot_CPU35: - 3 frac[2,1] + rows_tot_CPU35 = 0
 R33: - 0.0833333333333333 rows_tot_CPU35 + normalized_rows_tot_CPU35 = 0
 mem_tot_CPU1079: - mem[3,2] - mem[3,3] - mem[3,0] + mem_tot_CPU1079 = 0
 R35: - 3.05176e-05 mem_tot_CPU1079 + normalized_mem_tot_CPU1079 = 0
 rows_tot_CPU1079: - 3 frac[3,2] - 3 frac[3,3] - 3 frac[3,0]
   + rows_tot_CPU1079 = 0
 R37: - 0.0833333333333333 rows_tot_CPU1079 + normalized_rows_tot_CPU1079
   = 0
 mem_tot_CPU8799: - mem[4,2] + mem_tot_CPU8799 = 0
 R39: - 3.05176e-05 mem_tot_CPU8799 + normalized_mem_tot_CPU8799 = 0
 rows_tot_CPU8799: - 3 frac[4,2] + rows_tot_CPU8799 = 0
 R41: - 0.0833333333333333 rows_tot_CPU8799 + normalized_rows_tot_CPU8799
   = 0
 mem_tot_CPU2019: - mem[5,1] - mem[5,3] + mem_tot_CPU2019 = 0
 R43: - 3.05176e-05 mem_tot_CPU2019 + normalized_mem_tot_CPU2019 = 0
 rows_tot_CPU2019: - 3 frac[5,1] - 3 frac[5,3] + rows_tot_CPU2019 = 0
 R45: - 0.0833333333333333 rows_tot_CPU2019 + normalized_rows_tot_CPU2019
   = 0
 mem_tot_CPU1731: - mem[6,2] + mem_tot_CPU1731 = 0
 R47: - 3.05176e-05 mem_tot_CPU1731 + normalized_mem_tot_CPU1731 = 0
 rows_tot_CPU1731: - 3 frac[6,2] + rows_tot_CPU1731 = 0
 R49: - 0.0833333333333333 rows_tot_CPU1731 + normalized_rows_tot_CPU1731
   = 0
 mem_tot_CPU1699: - mem[7,1] + mem_tot_CPU1699 = 0
 R51: - 3.05176e-05 mem_tot_CPU1699 + normalized_mem_tot_CPU1699 = 0
 rows_tot_CPU1699: - 3 frac[7,1] + rows_tot_CPU1699 = 0
 R53: - 0.0833333333333333 rows_tot_CPU1699 + normalized_rows_tot_CPU1699
   = 0
 row_capacity_CPU9515: rows_tot_CPU9515 <= 12
 row_capacity_CPU1779: rows_tot_CPU1779 <= 12
 row_capacity_CPU35: rows_tot_CPU35 <= 12
 row_capacity_CPU1079: rows_tot_CPU1079 <= 12
 row_capacity_CPU8799: rows_tot_CPU8799 <= 12
 row_capacity_CPU2019: rows_tot_CPU2019 <= 12
 row_capacity_CPU1731: rows_tot_CPU1731 <= 12
 row_capacity_CPU1699: rows_tot_CPU1699 <= 12
 ns_single_CPU9515: 3.5 rows_tot_CPU9515 - ns_single_CPU9515
   + pdt_m_rows_CPU9515 = 0
 capacity_cores_CPU9515: cores_sketch_CPU9515 + cores_dpdk_CPU9515 <= 7
 ns_sketch_CPU9515: - ns_single_CPU9515
   + 11.42857142857143 cores_sketch_CPU9515 >= 0
 ns_dpdk_CPU9515: - 4.285714285714287 cores_dpdk_CPU9515
   <= -21.42857142857143
 ns_single_CPU1779: 3.5 rows_tot_CPU1779 - ns_single_CPU1779
   + pdt_m_rows_CPU1779 = 0
 capacity_cores_CPU1779: cores_sketch_CPU1779 + cores_dpdk_CPU1779 <= 7
 ns_sketch_CPU1779: - ns_single_CPU1779
   + 11.42857142857143 cores_sketch_CPU1779 >= 0
 ns_dpdk_CPU1779: - 4.285714285714287 cores_dpdk_CPU1779
   <= -21.42857142857143
 ns_single_CPU35: 3.5 rows_tot_CPU35 - ns_single_CPU35 + pdt_m_rows_CPU35
   = 0
 capacity_cores_CPU35: cores_sketch_CPU35 + cores_dpdk_CPU35 <= 7
 ns_sketch_CPU35: - ns_single_CPU35 + 11.42857142857143 cores_sketch_CPU35
   >= 0
 ns_dpdk_CPU35: - 4.285714285714287 cores_dpdk_CPU35 <= -21.42857142857143
 ns_single_CPU1079: 3.5 rows_tot_CPU1079 - ns_single_CPU1079
   + pdt_m_rows_CPU1079 = 0
 capacity_cores_CPU1079: cores_sketch_CPU1079 + cores_dpdk_CPU1079 <= 7
 ns_sketch_CPU1079: - ns_single_CPU1079
   + 11.42857142857143 cores_sketch_CPU1079 >= 0
 ns_dpdk_CPU1079: - 4.285714285714287 cores_dpdk_CPU1079
   <= -21.42857142857143
 ns_single_CPU8799: 3.5 rows_tot_CPU8799 - ns_single_CPU8799
   + pdt_m_rows_CPU8799 = 0
 capacity_cores_CPU8799: cores_sketch_CPU8799 + cores_dpdk_CPU8799 <= 7
 ns_sketch_CPU8799: - ns_single_CPU8799
   + 11.42857142857143 cores_sketch_CPU8799 >= 0
 ns_dpdk_CPU8799: - 4.285714285714287 cores_dpdk_CPU8799
   <= -21.42857142857143
 ns_single_CPU2019: 3.5 rows_tot_CPU2019 - ns_single_CPU2019
   + pdt_m_rows_CPU2019 = 0
 capacity_cores_CPU2019: cores_sketch_CPU2019 + cores_dpdk_CPU2019 <= 7
 ns_sketch_CPU2019: - ns_single_CPU2019
   + 11.42857142857143 cores_sketch_CPU2019 >= 0
 ns_dpdk_CPU2019: - 4.285714285714287 cores_dpdk_CPU2019
   <= -21.42857142857143
 ns_single_CPU1731: 3.5 rows_tot_CPU1731 - ns_single_CPU1731
   + pdt_m_rows_CPU1731 = 0
 capacity_cores_CPU1731: cores_sketch_CPU1731 + cores_dpdk_CPU1731 <= 7
 ns_sketch_CPU1731: - ns_single_CPU1731
   + 11.42857142857143 cores_sketch_CPU1731 >= 0
 ns_dpdk_CPU1731: - 4.285714285714287 cores_dpdk_CPU1731
   <= -21.42857142857143
 ns_single_CPU1699: 3.5 rows_tot_CPU1699 - ns_single_CPU1699
   + pdt_m_rows_CPU1699 = 0
 capacity_cores_CPU1699: cores_sketch_CPU1699 + cores_dpdk_CPU1699 <= 7
 ns_sketch_CPU1699: - ns_single_CPU1699
   + 11.42857142857143 cores_sketch_CPU1699 >= 0
 ns_dpdk_CPU1699: - 4.285714285714287 cores_dpdk_CPU1699
   <= -21.42857142857143
 res_acc: 1.220703125e-04 mem_tot_CPU9515 + 1.220703125e-04 mem_tot_CPU1779
   + 1.220703125e-04 mem_tot_CPU35 + 1.220703125e-04 mem_tot_CPU1079
   + 1.220703125e-04 mem_tot_CPU8799 + 1.220703125e-04 mem_tot_CPU2019
   + 1.220703125e-04 mem_tot_CPU1731 + 1.220703125e-04 mem_tot_CPU1699
   + 10 cores_sketch_CPU9515 + 10 cores_dpdk_CPU9515
   + 10 cores_sketch_CPU1779 + 10 cores_dpdk_CPU1779
   + 10 cores_sketch_CPU35 + 10 cores_dpdk_CPU35 + 10 cores_sketch_CPU1079
   + 10 cores_dpdk_CPU1079 + 10 cores_sketch_CPU8799
   + 10 cores_dpdk_CPU8799 + 10 cores_sketch_CPU2019
   + 10 cores_dpdk_CPU2019 + 10 cores_sketch_CPU1731
   + 10 cores_dpdk_CPU1731 + 10 cores_sketch_CPU1699
   + 10 cores_dpdk_CPU1699 - res = 0
 pdt_m_rows_CPU9515: pdt_m_rows_CPU9515 + [
   - rows_tot_CPU9515 * m_access_time_CPU9515 ] = 0
 pdt_m_rows_CPU1779: pdt_m_rows_CPU1779 + [
   - rows_tot_CPU1779 * m_access_time_CPU1779 ] = 0
 pdt_m_rows_CPU35: pdt_m_rows_CPU35 + [
   - rows_tot_CPU35 * m_access_time_CPU35 ] = 0
 pdt_m_rows_CPU1079: pdt_m_rows_CPU1079 + [
   - rows_tot_CPU1079 * m_access_time_CPU1079 ] = 0
 pdt_m_rows_CPU8799: pdt_m_rows_CPU8799 + [
   - rows_tot_CPU8799 * m_access_time_CPU8799 ] = 0
 pdt_m_rows_CPU2019: pdt_m_rows_CPU2019 + [
   - rows_tot_CPU2019 * m_access_time_CPU2019 ] = 0
 pdt_m_rows_CPU1731: pdt_m_rows_CPU1731 + [
   - rows_tot_CPU1731 * m_access_time_CPU1731 ] = 0
 pdt_m_rows_CPU1699: pdt_m_rows_CPU1699 + [
   - rows_tot_CPU1699 * m_access_time_CPU1699 ] = 0
Bounds
 frac[1,2] <= 1
 frac[0,1] <= 1
 frac[3,2] <= 1
 frac[0,0] <= 1
 frac[1,3] <= 1
 frac[3,3] <= 1
 frac[5,1] <= 1
 frac[7,1] <= 1
 frac[3,0] <= 1
 frac[2,1] <= 1
 frac[6,2] <= 1
 frac[4,2] <= 1
 frac[0,2] <= 1
 frac[5,3] <= 1
 mem_tot_CPU9515 <= 32768
 normalized_mem_tot_CPU9515 <= 1
 normalized_rows_tot_CPU9515 <= 1
 mem_tot_CPU1779 <= 32768
 normalized_mem_tot_CPU1779 <= 1
 normalized_rows_tot_CPU1779 <= 1
 mem_tot_CPU35 <= 32768
 normalized_mem_tot_CPU35 <= 1
 normalized_rows_tot_CPU35 <= 1
 mem_tot_CPU1079 <= 32768
 normalized_mem_tot_CPU1079 <= 1
 normalized_rows_tot_CPU1079 <= 1
 mem_tot_CPU8799 <= 32768
 normalized_mem_tot_CPU8799 <= 1
 normalized_rows_tot_CPU8799 <= 1
 mem_tot_CPU2019 <= 32768
 normalized_mem_tot_CPU2019 <= 1
 normalized_rows_tot_CPU2019 <= 1
 mem_tot_CPU1731 <= 32768
 normalized_mem_tot_CPU1731 <= 1
 normalized_rows_tot_CPU1731 <= 1
 mem_tot_CPU1699 <= 32768
 normalized_mem_tot_CPU1699 <= 1
 normalized_rows_tot_CPU1699 <= 1
 cores_sketch_CPU9515 <= 7
 1 <= cores_dpdk_CPU9515 <= 7
 cores_sketch_CPU1779 <= 7
 1 <= cores_dpdk_CPU1779 <= 7
 cores_sketch_CPU35 <= 7
 1 <= cores_dpdk_CPU35 <= 7
 cores_sketch_CPU1079 <= 7
 1 <= cores_dpdk_CPU1079 <= 7
 cores_sketch_CPU8799 <= 7
 1 <= cores_dpdk_CPU8799 <= 7
 cores_sketch_CPU2019 <= 7
 1 <= cores_dpdk_CPU2019 <= 7
 cores_sketch_CPU1731 <= 7
 1 <= cores_dpdk_CPU1731 <= 7
 cores_sketch_CPU1699 <= 7
 1 <= cores_dpdk_CPU1699 <= 7
Generals
 cores_sketch_CPU9515 cores_dpdk_CPU9515 cores_sketch_CPU1779
 cores_dpdk_CPU1779 cores_sketch_CPU35 cores_dpdk_CPU35
 cores_sketch_CPU1079 cores_dpdk_CPU1079 cores_sketch_CPU8799
 cores_dpdk_CPU8799 cores_sketch_CPU2019 cores_dpdk_CPU2019
 cores_sketch_CPU1731 cores_dpdk_CPU1731 cores_sketch_CPU1699
 cores_dpdk_CPU1699
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU9515 , normalized_mem_tot_CPU1779 ,
   normalized_mem_tot_CPU35 , normalized_mem_tot_CPU1079 ,
   normalized_mem_tot_CPU8799 , normalized_mem_tot_CPU2019 ,
   normalized_mem_tot_CPU1731 , normalized_mem_tot_CPU1699 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX (
   normalized_rows_tot_CPU9515 , normalized_rows_tot_CPU1779 ,
   normalized_rows_tot_CPU35 , normalized_rows_tot_CPU1079 ,
   normalized_rows_tot_CPU8799 , normalized_rows_tot_CPU2019 ,
   normalized_rows_tot_CPU1731 , normalized_rows_tot_CPU1699 )
 mem_access_time_CPU9515:
   m_access_time_CPU9515 = PWL ( mem_tot_CPU9515 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1779:
   m_access_time_CPU1779 = PWL ( mem_tot_CPU1779 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU35: m_access_time_CPU35 = PWL ( mem_tot_CPU35 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1079:
   m_access_time_CPU1079 = PWL ( mem_tot_CPU1079 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU8799:
   m_access_time_CPU8799 = PWL ( mem_tot_CPU8799 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU2019:
   m_access_time_CPU2019 = PWL ( mem_tot_CPU2019 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1731:
   m_access_time_CPU1731 = PWL ( mem_tot_CPU1731 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1699:
   m_access_time_CPU1699 = PWL ( mem_tot_CPU1699 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

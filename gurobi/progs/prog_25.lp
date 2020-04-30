\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_CPU_Cluster + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU1116
   + 0 m_access_time_CPU1711 + 0 m_access_time_CPU272
   + 0 m_access_time_CPU3103 + 0 m_access_time_CPU3240
   + 0 m_access_time_CPU1720 + 0 m_access_time_CPU2412
   + 0 m_access_time_CPU2491 + res
Subject To
 cov_0_1: frac[1,1] + frac[6,1] >= 0.7
 cov_1_0: frac[3,0] + frac[7,0] >= 0.8
 cov_2_0: frac[4,0] >= 0.7
 cov_3_1: frac[2,1] + frac[4,1] >= 1
 cov_4_1: frac[0,1] + frac[3,1] >= 0.7
 accuracy_0_1: - 509.68359375 frac[0,1] + mem[0,1] = 0
 accuracy_7_0: - 509.68359375 frac[7,0] + mem[7,0] = 0
 accuracy_3_0: - 509.68359375 frac[3,0] + mem[3,0] = 0
 accuracy_6_1: - 509.68359375 frac[6,1] + mem[6,1] = 0
 accuracy_3_1: - 509.68359375 frac[3,1] + mem[3,1] = 0
 accuracy_2_1: - 509.68359375 frac[2,1] + mem[2,1] = 0
 accuracy_4_1: - 509.68359375 frac[4,1] + mem[4,1] = 0
 accuracy_1_1: - 509.68359375 frac[1,1] + mem[1,1] = 0
 accuracy_4_0: - 509.68359375 frac[4,0] + mem[4,0] = 0
 mem_tot_CPU1116: - mem[0,1] + mem_tot_CPU1116 = 0
 R15: - 3.05176e-05 mem_tot_CPU1116 + normalized_mem_tot_CPU1116 = 0
 rows_tot_CPU1116: - 3 frac[0,1] + rows_tot_CPU1116 = 0
 R17: - 0.0833333333333333 rows_tot_CPU1116 + normalized_rows_tot_CPU1116
   = 0
 mem_tot_CPU1711: - mem[1,1] + mem_tot_CPU1711 = 0
 R19: - 3.05176e-05 mem_tot_CPU1711 + normalized_mem_tot_CPU1711 = 0
 rows_tot_CPU1711: - 3 frac[1,1] + rows_tot_CPU1711 = 0
 R21: - 0.0833333333333333 rows_tot_CPU1711 + normalized_rows_tot_CPU1711
   = 0
 mem_tot_CPU272: - mem[2,1] + mem_tot_CPU272 = 0
 R23: - 3.05176e-05 mem_tot_CPU272 + normalized_mem_tot_CPU272 = 0
 rows_tot_CPU272: - 3 frac[2,1] + rows_tot_CPU272 = 0
 R25: - 0.0833333333333333 rows_tot_CPU272 + normalized_rows_tot_CPU272
   = 0
 mem_tot_CPU3103: - mem[3,0] - mem[3,1] + mem_tot_CPU3103 = 0
 R27: - 3.05176e-05 mem_tot_CPU3103 + normalized_mem_tot_CPU3103 = 0
 rows_tot_CPU3103: - 3 frac[3,0] - 3 frac[3,1] + rows_tot_CPU3103 = 0
 R29: - 0.0833333333333333 rows_tot_CPU3103 + normalized_rows_tot_CPU3103
   = 0
 mem_tot_CPU3240: - mem[4,1] - mem[4,0] + mem_tot_CPU3240 = 0
 R31: - 3.05176e-05 mem_tot_CPU3240 + normalized_mem_tot_CPU3240 = 0
 rows_tot_CPU3240: - 3 frac[4,1] - 3 frac[4,0] + rows_tot_CPU3240 = 0
 R33: - 0.0833333333333333 rows_tot_CPU3240 + normalized_rows_tot_CPU3240
   = 0
 mem_tot_CPU1720: mem_tot_CPU1720 = 0
 R35: - 3.05176e-05 mem_tot_CPU1720 + normalized_mem_tot_CPU1720 = 0
 rows_tot_CPU1720: rows_tot_CPU1720 = 0
 R37: - 0.0833333333333333 rows_tot_CPU1720 + normalized_rows_tot_CPU1720
   = 0
 mem_tot_CPU2412: - mem[6,1] + mem_tot_CPU2412 = 0
 R39: - 3.05176e-05 mem_tot_CPU2412 + normalized_mem_tot_CPU2412 = 0
 rows_tot_CPU2412: - 3 frac[6,1] + rows_tot_CPU2412 = 0
 R41: - 0.0833333333333333 rows_tot_CPU2412 + normalized_rows_tot_CPU2412
   = 0
 mem_tot_CPU2491: - mem[7,0] + mem_tot_CPU2491 = 0
 R43: - 3.05176e-05 mem_tot_CPU2491 + normalized_mem_tot_CPU2491 = 0
 rows_tot_CPU2491: - 3 frac[7,0] + rows_tot_CPU2491 = 0
 R45: - 0.0833333333333333 rows_tot_CPU2491 + normalized_rows_tot_CPU2491
   = 0
 row_capacity_CPU1116: rows_tot_CPU1116 <= 12
 row_capacity_CPU1711: rows_tot_CPU1711 <= 12
 row_capacity_CPU272: rows_tot_CPU272 <= 12
 row_capacity_CPU3103: rows_tot_CPU3103 <= 12
 row_capacity_CPU3240: rows_tot_CPU3240 <= 12
 row_capacity_CPU1720: rows_tot_CPU1720 <= 12
 row_capacity_CPU2412: rows_tot_CPU2412 <= 12
 row_capacity_CPU2491: rows_tot_CPU2491 <= 12
 ns_single_CPU1116: 3.5 rows_tot_CPU1116 - ns_single_CPU1116
   + pdt_m_rows_CPU1116 = 0
 ns_sketch_CPU1116: - ns_single_CPU1116
   + 11.42863142857143 cores_sketch_CPU1116 >= 0
 capacity_cores_CPU1116: cores_sketch_CPU1116 <= 2
 ns_single_CPU1711: 3.5 rows_tot_CPU1711 - ns_single_CPU1711
   + pdt_m_rows_CPU1711 = 0
 ns_sketch_CPU1711: - ns_single_CPU1711
   + 11.42863142857143 cores_sketch_CPU1711 >= 0
 capacity_cores_CPU1711: cores_sketch_CPU1711 <= 2
 ns_single_CPU272: 3.5 rows_tot_CPU272 - ns_single_CPU272
   + pdt_m_rows_CPU272 = 0
 ns_sketch_CPU272: - ns_single_CPU272
   + 11.42863142857143 cores_sketch_CPU272 >= 0
 capacity_cores_CPU272: cores_sketch_CPU272 <= 2
 ns_single_CPU3103: 3.5 rows_tot_CPU3103 - ns_single_CPU3103
   + pdt_m_rows_CPU3103 = 0
 ns_sketch_CPU3103: - ns_single_CPU3103
   + 11.42863142857143 cores_sketch_CPU3103 >= 0
 capacity_cores_CPU3103: cores_sketch_CPU3103 <= 2
 ns_single_CPU3240: 3.5 rows_tot_CPU3240 - ns_single_CPU3240
   + pdt_m_rows_CPU3240 = 0
 ns_sketch_CPU3240: - ns_single_CPU3240
   + 11.42863142857143 cores_sketch_CPU3240 >= 0
 capacity_cores_CPU3240: cores_sketch_CPU3240 <= 2
 ns_single_CPU1720: 3.5 rows_tot_CPU1720 - ns_single_CPU1720
   + pdt_m_rows_CPU1720 = 0
 ns_sketch_CPU1720: - ns_single_CPU1720
   + 11.42863142857143 cores_sketch_CPU1720 >= 0
 capacity_cores_CPU1720: cores_sketch_CPU1720 <= 2
 ns_single_CPU2412: 3.5 rows_tot_CPU2412 - ns_single_CPU2412
   + pdt_m_rows_CPU2412 = 0
 ns_sketch_CPU2412: - ns_single_CPU2412
   + 11.42863142857143 cores_sketch_CPU2412 >= 0
 capacity_cores_CPU2412: cores_sketch_CPU2412 <= 2
 ns_single_CPU2491: 3.5 rows_tot_CPU2491 - ns_single_CPU2491
   + pdt_m_rows_CPU2491 = 0
 ns_sketch_CPU2491: - ns_single_CPU2491
   + 11.42863142857143 cores_sketch_CPU2491 >= 0
 capacity_cores_CPU2491: cores_sketch_CPU2491 <= 2
 res_acc: 1.220703125e-04 mem_tot_CPU1116 + 1.220703125e-04 mem_tot_CPU1711
   + 1.220703125e-04 mem_tot_CPU272 + 1.220703125e-04 mem_tot_CPU3103
   + 1.220703125e-04 mem_tot_CPU3240 + 1.220703125e-04 mem_tot_CPU1720
   + 1.220703125e-04 mem_tot_CPU2412 + 1.220703125e-04 mem_tot_CPU2491
   + 10 cores_sketch_CPU1116 + 10 cores_sketch_CPU1711
   + 10 cores_sketch_CPU272 + 10 cores_sketch_CPU3103
   + 10 cores_sketch_CPU3240 + 10 cores_sketch_CPU1720
   + 10 cores_sketch_CPU2412 + 10 cores_sketch_CPU2491 - res = -400
 pdt_m_rows_CPU1116: pdt_m_rows_CPU1116 + [
   - rows_tot_CPU1116 * m_access_time_CPU1116 ] = 0
 pdt_m_rows_CPU1711: pdt_m_rows_CPU1711 + [
   - rows_tot_CPU1711 * m_access_time_CPU1711 ] = 0
 pdt_m_rows_CPU272: pdt_m_rows_CPU272 + [
   - rows_tot_CPU272 * m_access_time_CPU272 ] = 0
 pdt_m_rows_CPU3103: pdt_m_rows_CPU3103 + [
   - rows_tot_CPU3103 * m_access_time_CPU3103 ] = 0
 pdt_m_rows_CPU3240: pdt_m_rows_CPU3240 + [
   - rows_tot_CPU3240 * m_access_time_CPU3240 ] = 0
 pdt_m_rows_CPU1720: pdt_m_rows_CPU1720 + [
   - rows_tot_CPU1720 * m_access_time_CPU1720 ] = 0
 pdt_m_rows_CPU2412: pdt_m_rows_CPU2412 + [
   - rows_tot_CPU2412 * m_access_time_CPU2412 ] = 0
 pdt_m_rows_CPU2491: pdt_m_rows_CPU2491 + [
   - rows_tot_CPU2491 * m_access_time_CPU2491 ] = 0
Bounds
 frac[0,1] <= 1
 frac[1,1] <= 1
 frac[2,1] <= 1
 frac[3,0] <= 1
 frac[3,1] <= 1
 frac[4,1] <= 1
 frac[4,0] <= 1
 frac[6,1] <= 1
 frac[7,0] <= 1
 mem_tot_CPU1116 <= 32768
 normalized_mem_tot_CPU1116 <= 1
 normalized_rows_tot_CPU1116 <= 1
 mem_tot_CPU1711 <= 32768
 normalized_mem_tot_CPU1711 <= 1
 normalized_rows_tot_CPU1711 <= 1
 mem_tot_CPU272 <= 32768
 normalized_mem_tot_CPU272 <= 1
 normalized_rows_tot_CPU272 <= 1
 mem_tot_CPU3103 <= 32768
 normalized_mem_tot_CPU3103 <= 1
 normalized_rows_tot_CPU3103 <= 1
 mem_tot_CPU3240 <= 32768
 normalized_mem_tot_CPU3240 <= 1
 normalized_rows_tot_CPU3240 <= 1
 mem_tot_CPU1720 <= 32768
 normalized_mem_tot_CPU1720 <= 1
 normalized_rows_tot_CPU1720 <= 1
 mem_tot_CPU2412 <= 32768
 normalized_mem_tot_CPU2412 <= 1
 normalized_rows_tot_CPU2412 <= 1
 mem_tot_CPU2491 <= 32768
 normalized_mem_tot_CPU2491 <= 1
 normalized_rows_tot_CPU2491 <= 1
 cores_sketch_CPU1116 <= 7
 cores_sketch_CPU1711 <= 7
 cores_sketch_CPU272 <= 7
 cores_sketch_CPU3103 <= 7
 cores_sketch_CPU3240 <= 7
 cores_sketch_CPU1720 <= 7
 cores_sketch_CPU2412 <= 7
 cores_sketch_CPU2491 <= 7
Generals
 cores_sketch_CPU1116 cores_sketch_CPU1711 cores_sketch_CPU272
 cores_sketch_CPU3103 cores_sketch_CPU3240 cores_sketch_CPU1720
 cores_sketch_CPU2412 cores_sketch_CPU2491
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU1116 , normalized_mem_tot_CPU1711 ,
   normalized_mem_tot_CPU272 , normalized_mem_tot_CPU3103 ,
   normalized_mem_tot_CPU3240 , normalized_mem_tot_CPU1720 ,
   normalized_mem_tot_CPU2412 , normalized_mem_tot_CPU2491 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX (
   normalized_rows_tot_CPU1116 , normalized_rows_tot_CPU1711 ,
   normalized_rows_tot_CPU272 , normalized_rows_tot_CPU3103 ,
   normalized_rows_tot_CPU3240 , normalized_rows_tot_CPU1720 ,
   normalized_rows_tot_CPU2412 , normalized_rows_tot_CPU2491 )
 mem_access_time_CPU1116:
   m_access_time_CPU1116 = PWL ( mem_tot_CPU1116 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1711:
   m_access_time_CPU1711 = PWL ( mem_tot_CPU1711 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU272: m_access_time_CPU272 = PWL ( mem_tot_CPU272 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU3103:
   m_access_time_CPU3103 = PWL ( mem_tot_CPU3103 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU3240:
   m_access_time_CPU3240 = PWL ( mem_tot_CPU3240 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1720:
   m_access_time_CPU1720 = PWL ( mem_tot_CPU1720 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU2412:
   m_access_time_CPU2412 = PWL ( mem_tot_CPU2412 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU2491:
   m_access_time_CPU2491 = PWL ( mem_tot_CPU2491 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

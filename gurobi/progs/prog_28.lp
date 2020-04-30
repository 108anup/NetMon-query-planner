\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_CPU_Cluster + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU358
   + 0 m_access_time_CPU879 + 0 m_access_time_CPU1300
   + 0 m_access_time_CPU515 + 0 m_access_time_CPU946
   + 0 m_access_time_CPU1463 + 0 m_access_time_CPU822
   + 0 m_access_time_CPU685 + res
Subject To
 cov_0_1: frac[0,1] + frac[7,1] >= 0.8
 cov_1_0: frac[4,0] >= 0.8
 cov_2_0: frac[5,0] + frac[7,0] >= 0.8
 cov_3_0: frac[0,0] + frac[3,0] >= 0.8
 cov_4_0: frac[1,0] + frac[2,0] >= 0.7
 cov_5_1: frac[1,1] >= 0.7
 accuracy_0_1: - 509.68359375 frac[0,1] + mem[0,1] = 0
 accuracy_0_0: - 509.68359375 frac[0,0] + mem[0,0] = 0
 accuracy_7_0: - 509.68359375 frac[7,0] + mem[7,0] = 0
 accuracy_7_1: - 509.68359375 frac[7,1] + mem[7,1] = 0
 accuracy_3_0: - 509.68359375 frac[3,0] + mem[3,0] = 0
 accuracy_2_0: - 509.68359375 frac[2,0] + mem[2,0] = 0
 accuracy_5_0: - 509.68359375 frac[5,0] + mem[5,0] = 0
 accuracy_1_0: - 509.68359375 frac[1,0] + mem[1,0] = 0
 accuracy_1_1: - 509.68359375 frac[1,1] + mem[1,1] = 0
 accuracy_4_0: - 509.68359375 frac[4,0] + mem[4,0] = 0
 mem_tot_CPU358: - mem[0,1] - mem[0,0] + mem_tot_CPU358 = 0
 R17: - 3.05176e-05 mem_tot_CPU358 + normalized_mem_tot_CPU358 = 0
 rows_tot_CPU358: - 3 frac[0,1] - 3 frac[0,0] + rows_tot_CPU358 = 0
 R19: - 0.0833333333333333 rows_tot_CPU358 + normalized_rows_tot_CPU358
   = 0
 mem_tot_CPU879: - mem[1,0] - mem[1,1] + mem_tot_CPU879 = 0
 R21: - 3.05176e-05 mem_tot_CPU879 + normalized_mem_tot_CPU879 = 0
 rows_tot_CPU879: - 3 frac[1,0] - 3 frac[1,1] + rows_tot_CPU879 = 0
 R23: - 0.0833333333333333 rows_tot_CPU879 + normalized_rows_tot_CPU879
   = 0
 mem_tot_CPU1300: - mem[2,0] + mem_tot_CPU1300 = 0
 R25: - 3.05176e-05 mem_tot_CPU1300 + normalized_mem_tot_CPU1300 = 0
 rows_tot_CPU1300: - 3 frac[2,0] + rows_tot_CPU1300 = 0
 R27: - 0.0833333333333333 rows_tot_CPU1300 + normalized_rows_tot_CPU1300
   = 0
 mem_tot_CPU515: - mem[3,0] + mem_tot_CPU515 = 0
 R29: - 3.05176e-05 mem_tot_CPU515 + normalized_mem_tot_CPU515 = 0
 rows_tot_CPU515: - 3 frac[3,0] + rows_tot_CPU515 = 0
 R31: - 0.0833333333333333 rows_tot_CPU515 + normalized_rows_tot_CPU515
   = 0
 mem_tot_CPU946: - mem[4,0] + mem_tot_CPU946 = 0
 R33: - 3.05176e-05 mem_tot_CPU946 + normalized_mem_tot_CPU946 = 0
 rows_tot_CPU946: - 3 frac[4,0] + rows_tot_CPU946 = 0
 R35: - 0.0833333333333333 rows_tot_CPU946 + normalized_rows_tot_CPU946
   = 0
 mem_tot_CPU1463: - mem[5,0] + mem_tot_CPU1463 = 0
 R37: - 3.05176e-05 mem_tot_CPU1463 + normalized_mem_tot_CPU1463 = 0
 rows_tot_CPU1463: - 3 frac[5,0] + rows_tot_CPU1463 = 0
 R39: - 0.0833333333333333 rows_tot_CPU1463 + normalized_rows_tot_CPU1463
   = 0
 mem_tot_CPU822: mem_tot_CPU822 = 0
 R41: - 3.05176e-05 mem_tot_CPU822 + normalized_mem_tot_CPU822 = 0
 rows_tot_CPU822: rows_tot_CPU822 = 0
 R43: - 0.0833333333333333 rows_tot_CPU822 + normalized_rows_tot_CPU822
   = 0
 mem_tot_CPU685: - mem[7,0] - mem[7,1] + mem_tot_CPU685 = 0
 R45: - 3.05176e-05 mem_tot_CPU685 + normalized_mem_tot_CPU685 = 0
 rows_tot_CPU685: - 3 frac[7,0] - 3 frac[7,1] + rows_tot_CPU685 = 0
 R47: - 0.0833333333333333 rows_tot_CPU685 + normalized_rows_tot_CPU685
   = 0
 row_capacity_CPU358: rows_tot_CPU358 <= 12
 row_capacity_CPU879: rows_tot_CPU879 <= 12
 row_capacity_CPU1300: rows_tot_CPU1300 <= 12
 row_capacity_CPU515: rows_tot_CPU515 <= 12
 row_capacity_CPU946: rows_tot_CPU946 <= 12
 row_capacity_CPU1463: rows_tot_CPU1463 <= 12
 row_capacity_CPU822: rows_tot_CPU822 <= 12
 row_capacity_CPU685: rows_tot_CPU685 <= 12
 ns_single_CPU358: 3.5 rows_tot_CPU358 - ns_single_CPU358
   + pdt_m_rows_CPU358 = 0
 ns_sketch_CPU358: - ns_single_CPU358
   + 11.42857142857143 cores_sketch_CPU358 >= 0
 capacity_cores_CPU358: cores_sketch_CPU358 <= 2
 ns_single_CPU879: 3.5 rows_tot_CPU879 - ns_single_CPU879
   + pdt_m_rows_CPU879 = 0
 ns_sketch_CPU879: - ns_single_CPU879
   + 11.42857142857143 cores_sketch_CPU879 >= 0
 capacity_cores_CPU879: cores_sketch_CPU879 <= 2
 ns_single_CPU1300: 3.5 rows_tot_CPU1300 - ns_single_CPU1300
   + pdt_m_rows_CPU1300 = 0
 ns_sketch_CPU1300: - ns_single_CPU1300
   + 11.42857142857143 cores_sketch_CPU1300 >= 0
 capacity_cores_CPU1300: cores_sketch_CPU1300 <= 2
 ns_single_CPU515: 3.5 rows_tot_CPU515 - ns_single_CPU515
   + pdt_m_rows_CPU515 = 0
 ns_sketch_CPU515: - ns_single_CPU515
   + 11.42857142857143 cores_sketch_CPU515 >= 0
 capacity_cores_CPU515: cores_sketch_CPU515 <= 2
 ns_single_CPU946: 3.5 rows_tot_CPU946 - ns_single_CPU946
   + pdt_m_rows_CPU946 = 0
 ns_sketch_CPU946: - ns_single_CPU946
   + 11.42857142857143 cores_sketch_CPU946 >= 0
 capacity_cores_CPU946: cores_sketch_CPU946 <= 2
 ns_single_CPU1463: 3.5 rows_tot_CPU1463 - ns_single_CPU1463
   + pdt_m_rows_CPU1463 = 0
 ns_sketch_CPU1463: - ns_single_CPU1463
   + 11.42857142857143 cores_sketch_CPU1463 >= 0
 capacity_cores_CPU1463: cores_sketch_CPU1463 <= 2
 ns_single_CPU822: 3.5 rows_tot_CPU822 - ns_single_CPU822
   + pdt_m_rows_CPU822 = 0
 ns_sketch_CPU822: - ns_single_CPU822
   + 11.42857142857143 cores_sketch_CPU822 >= 0
 capacity_cores_CPU822: cores_sketch_CPU822 <= 2
 ns_single_CPU685: 3.5 rows_tot_CPU685 - ns_single_CPU685
   + pdt_m_rows_CPU685 = 0
 ns_sketch_CPU685: - ns_single_CPU685
   + 11.42857142857143 cores_sketch_CPU685 >= 0
 capacity_cores_CPU685: cores_sketch_CPU685 <= 2
 res_acc: 1.220703125e-04 mem_tot_CPU358 + 1.220703125e-04 mem_tot_CPU879
   + 1.220703125e-04 mem_tot_CPU1300 + 1.220703125e-04 mem_tot_CPU515
   + 1.220703125e-04 mem_tot_CPU946 + 1.220703125e-04 mem_tot_CPU1463
   + 1.220703125e-04 mem_tot_CPU822 + 1.220703125e-04 mem_tot_CPU685
   + 10 cores_sketch_CPU358 + 10 cores_sketch_CPU879
   + 10 cores_sketch_CPU1300 + 10 cores_sketch_CPU515
   + 10 cores_sketch_CPU946 + 10 cores_sketch_CPU1463
   + 10 cores_sketch_CPU822 + 10 cores_sketch_CPU685 - res = -400
 pdt_m_rows_CPU358: pdt_m_rows_CPU358 + [
   - rows_tot_CPU358 * m_access_time_CPU358 ] = 0
 pdt_m_rows_CPU879: pdt_m_rows_CPU879 + [
   - rows_tot_CPU879 * m_access_time_CPU879 ] = 0
 pdt_m_rows_CPU1300: pdt_m_rows_CPU1300 + [
   - rows_tot_CPU1300 * m_access_time_CPU1300 ] = 0
 pdt_m_rows_CPU515: pdt_m_rows_CPU515 + [
   - rows_tot_CPU515 * m_access_time_CPU515 ] = 0
 pdt_m_rows_CPU946: pdt_m_rows_CPU946 + [
   - rows_tot_CPU946 * m_access_time_CPU946 ] = 0
 pdt_m_rows_CPU1463: pdt_m_rows_CPU1463 + [
   - rows_tot_CPU1463 * m_access_time_CPU1463 ] = 0
 pdt_m_rows_CPU822: pdt_m_rows_CPU822 + [
   - rows_tot_CPU822 * m_access_time_CPU822 ] = 0
 pdt_m_rows_CPU685: pdt_m_rows_CPU685 + [
   - rows_tot_CPU685 * m_access_time_CPU685 ] = 0
Bounds
 frac[0,1] <= 1
 frac[0,0] <= 1
 frac[1,0] <= 1
 frac[1,1] <= 1
 frac[2,0] <= 1
 frac[3,0] <= 1
 frac[4,0] <= 1
 frac[5,0] <= 1
 frac[7,0] <= 1
 frac[7,1] <= 1
 mem_tot_CPU358 <= 32768
 normalized_mem_tot_CPU358 <= 1
 normalized_rows_tot_CPU358 <= 1
 mem_tot_CPU879 <= 32768
 normalized_mem_tot_CPU879 <= 1
 normalized_rows_tot_CPU879 <= 1
 mem_tot_CPU1300 <= 32768
 normalized_mem_tot_CPU1300 <= 1
 normalized_rows_tot_CPU1300 <= 1
 mem_tot_CPU515 <= 32768
 normalized_mem_tot_CPU515 <= 1
 normalized_rows_tot_CPU515 <= 1
 mem_tot_CPU946 <= 32768
 normalized_mem_tot_CPU946 <= 1
 normalized_rows_tot_CPU946 <= 1
 mem_tot_CPU1463 <= 32768
 normalized_mem_tot_CPU1463 <= 1
 normalized_rows_tot_CPU1463 <= 1
 mem_tot_CPU822 <= 32768
 normalized_mem_tot_CPU822 <= 1
 normalized_rows_tot_CPU822 <= 1
 mem_tot_CPU685 <= 32768
 normalized_mem_tot_CPU685 <= 1
 normalized_rows_tot_CPU685 <= 1
 cores_sketch_CPU358 <= 7
 cores_sketch_CPU879 <= 7
 cores_sketch_CPU1300 <= 7
 cores_sketch_CPU515 <= 7
 cores_sketch_CPU946 <= 7
 cores_sketch_CPU1463 <= 7
 cores_sketch_CPU822 <= 7
 cores_sketch_CPU685 <= 7
Generals
 cores_sketch_CPU358 cores_sketch_CPU879 cores_sketch_CPU1300
 cores_sketch_CPU515 cores_sketch_CPU946 cores_sketch_CPU1463
 cores_sketch_CPU822 cores_sketch_CPU685
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU358 , normalized_mem_tot_CPU879 ,
   normalized_mem_tot_CPU1300 , normalized_mem_tot_CPU515 ,
   normalized_mem_tot_CPU946 , normalized_mem_tot_CPU1463 ,
   normalized_mem_tot_CPU822 , normalized_mem_tot_CPU685 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX (
   normalized_rows_tot_CPU358 , normalized_rows_tot_CPU879 ,
   normalized_rows_tot_CPU1300 , normalized_rows_tot_CPU515 ,
   normalized_rows_tot_CPU946 , normalized_rows_tot_CPU1463 ,
   normalized_rows_tot_CPU822 , normalized_rows_tot_CPU685 )
 mem_access_time_CPU358: m_access_time_CPU358 = PWL ( mem_tot_CPU358 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU879: m_access_time_CPU879 = PWL ( mem_tot_CPU879 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1300:
   m_access_time_CPU1300 = PWL ( mem_tot_CPU1300 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU515: m_access_time_CPU515 = PWL ( mem_tot_CPU515 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU946: m_access_time_CPU946 = PWL ( mem_tot_CPU946 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1463:
   m_access_time_CPU1463 = PWL ( mem_tot_CPU1463 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU822: m_access_time_CPU822 = PWL ( mem_tot_CPU822 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU685: m_access_time_CPU685 = PWL ( mem_tot_CPU685 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

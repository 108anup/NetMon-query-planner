\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_CPU_Cluster + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU1770
   + 0 m_access_time_CPU1582 + 0 m_access_time_CPU1472
   + 0 m_access_time_CPU1642 + 0 m_access_time_CPU826
   + 0 m_access_time_CPU313 + 0 m_access_time_CPU214
   + 0 m_access_time_CPU208 + res
Subject To
 cov_0_1: frac[0,1] + frac[2,1] >= 0.0111512197365087
 cov_1_0: frac[3,0] + frac[4,0] >= 0.0582392568919133
 cov_2_0: frac[6,0] >= 0.0582392568919133
 cov_3_1: frac[3,1] + frac[2,1] >= 0.0111512197365087
 cov_4_1: frac[6,1] + frac[2,1] >= 0.0111512197365087
 cov_5_0: frac[7,0] + frac[1,0] >= 0.0582392568919133
 cov_6_2: frac[2,2] >= 0.00582392568919154
 accuracy_0_1: - 509.68359375 frac[0,1] + mem[0,1] = 0
 accuracy_7_0: - 509.68359375 frac[7,0] + mem[7,0] = 0
 accuracy_3_0: - 509.68359375 frac[3,0] + mem[3,0] = 0
 accuracy_6_1: - 509.68359375 frac[6,1] + mem[6,1] = 0
 accuracy_3_1: - 509.68359375 frac[3,1] + mem[3,1] = 0
 accuracy_6_0: - 509.68359375 frac[6,0] + mem[6,0] = 0
 accuracy_2_1: - 509.68359375 frac[2,1] + mem[2,1] = 0
 accuracy_2_2: - 509.68359375 frac[2,2] + mem[2,2] = 0
 accuracy_1_0: - 509.68359375 frac[1,0] + mem[1,0] = 0
 accuracy_4_0: - 509.68359375 frac[4,0] + mem[4,0] = 0
 mem_tot_CPU1770: - mem[0,1] + mem_tot_CPU1770 = 0
 R18: - 3.05176e-05 mem_tot_CPU1770 + normalized_mem_tot_CPU1770 = 0
 rows_tot_CPU1770: - 3 frac[0,1] + rows_tot_CPU1770 = 0
 R20: - 0.0833333333333333 rows_tot_CPU1770 + normalized_rows_tot_CPU1770
   = 0
 mem_tot_CPU1582: - mem[1,0] + mem_tot_CPU1582 = 0
 R22: - 3.05176e-05 mem_tot_CPU1582 + normalized_mem_tot_CPU1582 = 0
 rows_tot_CPU1582: - 3 frac[1,0] + rows_tot_CPU1582 = 0
 R24: - 0.0833333333333333 rows_tot_CPU1582 + normalized_rows_tot_CPU1582
   = 0
 mem_tot_CPU1472: - mem[2,1] - mem[2,2] + mem_tot_CPU1472 = 0
 R26: - 3.05176e-05 mem_tot_CPU1472 + normalized_mem_tot_CPU1472 = 0
 rows_tot_CPU1472: - 3 frac[2,1] - 3 frac[2,2] + rows_tot_CPU1472 = 0
 R28: - 0.0833333333333333 rows_tot_CPU1472 + normalized_rows_tot_CPU1472
   = 0
 mem_tot_CPU1642: - mem[3,0] - mem[3,1] + mem_tot_CPU1642 = 0
 R30: - 3.05176e-05 mem_tot_CPU1642 + normalized_mem_tot_CPU1642 = 0
 rows_tot_CPU1642: - 3 frac[3,0] - 3 frac[3,1] + rows_tot_CPU1642 = 0
 R32: - 0.0833333333333333 rows_tot_CPU1642 + normalized_rows_tot_CPU1642
   = 0
 mem_tot_CPU826: - mem[4,0] + mem_tot_CPU826 = 0
 R34: - 3.05176e-05 mem_tot_CPU826 + normalized_mem_tot_CPU826 = 0
 rows_tot_CPU826: - 3 frac[4,0] + rows_tot_CPU826 = 0
 R36: - 0.0833333333333333 rows_tot_CPU826 + normalized_rows_tot_CPU826
   = 0
 mem_tot_CPU313: mem_tot_CPU313 = 0
 R38: - 3.05176e-05 mem_tot_CPU313 + normalized_mem_tot_CPU313 = 0
 rows_tot_CPU313: rows_tot_CPU313 = 0
 R40: - 0.0833333333333333 rows_tot_CPU313 + normalized_rows_tot_CPU313
   = 0
 mem_tot_CPU214: - mem[6,1] - mem[6,0] + mem_tot_CPU214 = 0
 R42: - 3.05176e-05 mem_tot_CPU214 + normalized_mem_tot_CPU214 = 0
 rows_tot_CPU214: - 3 frac[6,1] - 3 frac[6,0] + rows_tot_CPU214 = 0
 R44: - 0.0833333333333333 rows_tot_CPU214 + normalized_rows_tot_CPU214
   = 0
 mem_tot_CPU208: - mem[7,0] + mem_tot_CPU208 = 0
 R46: - 3.05176e-05 mem_tot_CPU208 + normalized_mem_tot_CPU208 = 0
 rows_tot_CPU208: - 3 frac[7,0] + rows_tot_CPU208 = 0
 R48: - 0.0833333333333333 rows_tot_CPU208 + normalized_rows_tot_CPU208
   = 0
 row_capacity_CPU1770: rows_tot_CPU1770 <= 12
 row_capacity_CPU1582: rows_tot_CPU1582 <= 12
 row_capacity_CPU1472: rows_tot_CPU1472 <= 12
 row_capacity_CPU1642: rows_tot_CPU1642 <= 12
 row_capacity_CPU826: rows_tot_CPU826 <= 12
 row_capacity_CPU313: rows_tot_CPU313 <= 12
 row_capacity_CPU214: rows_tot_CPU214 <= 12
 row_capacity_CPU208: rows_tot_CPU208 <= 12
 ns_single_CPU1770: 3.5 rows_tot_CPU1770 - ns_single_CPU1770
   + pdt_m_rows_CPU1770 = 0
 capacity_cores_CPU1770: cores_sketch_CPU1770 + cores_dpdk_CPU1770 <= 7
 ns_sketch_CPU1770: - ns_single_CPU1770
   + 10.20408163265306 cores_sketch_CPU1770 >= 0
 ns_dpdk_CPU1770: - 3.06122448979592 cores_dpdk_CPU1770
   <= -21.42857142857143
 ns_single_CPU1582: 3.5 rows_tot_CPU1582 - ns_single_CPU1582
   + pdt_m_rows_CPU1582 = 0
 capacity_cores_CPU1582: cores_sketch_CPU1582 + cores_dpdk_CPU1582 <= 7
 ns_sketch_CPU1582: - ns_single_CPU1582
   + 10.20408163265306 cores_sketch_CPU1582 >= 0
 ns_dpdk_CPU1582: - 3.06122448979592 cores_dpdk_CPU1582
   <= -21.42857142857143
 ns_single_CPU1472: 3.5 rows_tot_CPU1472 - ns_single_CPU1472
   + pdt_m_rows_CPU1472 = 0
 capacity_cores_CPU1472: cores_sketch_CPU1472 + cores_dpdk_CPU1472 <= 7
 ns_sketch_CPU1472: - ns_single_CPU1472
   + 10.20408163265306 cores_sketch_CPU1472 >= 0
 ns_dpdk_CPU1472: - 3.06122448979592 cores_dpdk_CPU1472
   <= -21.42857142857143
 ns_single_CPU1642: 3.5 rows_tot_CPU1642 - ns_single_CPU1642
   + pdt_m_rows_CPU1642 = 0
 capacity_cores_CPU1642: cores_sketch_CPU1642 + cores_dpdk_CPU1642 <= 7
 ns_sketch_CPU1642: - ns_single_CPU1642
   + 10.20408163265306 cores_sketch_CPU1642 >= 0
 ns_dpdk_CPU1642: - 3.06122448979592 cores_dpdk_CPU1642
   <= -21.42857142857143
 ns_single_CPU826: 3.5 rows_tot_CPU826 - ns_single_CPU826
   + pdt_m_rows_CPU826 = 0
 capacity_cores_CPU826: cores_sketch_CPU826 + cores_dpdk_CPU826 <= 7
 ns_sketch_CPU826: - ns_single_CPU826
   + 10.20408163265306 cores_sketch_CPU826 >= 0
 ns_dpdk_CPU826: - 3.06122448979592 cores_dpdk_CPU826
   <= -21.42857142857143
 ns_single_CPU313: 3.5 rows_tot_CPU313 - ns_single_CPU313
   + pdt_m_rows_CPU313 = 0
 capacity_cores_CPU313: cores_sketch_CPU313 + cores_dpdk_CPU313 <= 7
 ns_sketch_CPU313: - ns_single_CPU313
   + 10.20408163265306 cores_sketch_CPU313 >= 0
 ns_dpdk_CPU313: - 3.06122448979592 cores_dpdk_CPU313
   <= -21.42857142857143
 ns_single_CPU214: 3.5 rows_tot_CPU214 - ns_single_CPU214
   + pdt_m_rows_CPU214 = 0
 capacity_cores_CPU214: cores_sketch_CPU214 + cores_dpdk_CPU214 <= 7
 ns_sketch_CPU214: - ns_single_CPU214
   + 10.20408163265306 cores_sketch_CPU214 >= 0
 ns_dpdk_CPU214: - 3.06122448979592 cores_dpdk_CPU214
   <= -21.42857142857143
 ns_single_CPU208: 3.5 rows_tot_CPU208 - ns_single_CPU208
   + pdt_m_rows_CPU208 = 0
 capacity_cores_CPU208: cores_sketch_CPU208 + cores_dpdk_CPU208 <= 7
 ns_sketch_CPU208: - ns_single_CPU208
   + 10.20408163265306 cores_sketch_CPU208 >= 0
 ns_dpdk_CPU208: - 3.06122448979592 cores_dpdk_CPU208
   <= -21.42857142857143
 res_acc: 1.220703125e-04 mem_tot_CPU1770 + 1.220703125e-04 mem_tot_CPU1582
   + 1.220703125e-04 mem_tot_CPU1472 + 1.220703125e-04 mem_tot_CPU1642
   + 1.220703125e-04 mem_tot_CPU826 + 1.220703125e-04 mem_tot_CPU313
   + 1.220703125e-04 mem_tot_CPU214 + 1.220703125e-04 mem_tot_CPU208
   + 10 cores_sketch_CPU1770 + 10 cores_dpdk_CPU1770
   + 10 cores_sketch_CPU1582 + 10 cores_dpdk_CPU1582
   + 10 cores_sketch_CPU1472 + 10 cores_dpdk_CPU1472
   + 10 cores_sketch_CPU1642 + 10 cores_dpdk_CPU1642
   + 10 cores_sketch_CPU826 + 10 cores_dpdk_CPU826 + 10 cores_sketch_CPU313
   + 10 cores_dpdk_CPU313 + 10 cores_sketch_CPU214 + 10 cores_dpdk_CPU214
   + 10 cores_sketch_CPU208 + 10 cores_dpdk_CPU208 - res = 0
 pdt_m_rows_CPU1770: pdt_m_rows_CPU1770 + [
   - rows_tot_CPU1770 * m_access_time_CPU1770 ] = 0
 pdt_m_rows_CPU1582: pdt_m_rows_CPU1582 + [
   - rows_tot_CPU1582 * m_access_time_CPU1582 ] = 0
 pdt_m_rows_CPU1472: pdt_m_rows_CPU1472 + [
   - rows_tot_CPU1472 * m_access_time_CPU1472 ] = 0
 pdt_m_rows_CPU1642: pdt_m_rows_CPU1642 + [
   - rows_tot_CPU1642 * m_access_time_CPU1642 ] = 0
 pdt_m_rows_CPU826: pdt_m_rows_CPU826 + [
   - rows_tot_CPU826 * m_access_time_CPU826 ] = 0
 pdt_m_rows_CPU313: pdt_m_rows_CPU313 + [
   - rows_tot_CPU313 * m_access_time_CPU313 ] = 0
 pdt_m_rows_CPU214: pdt_m_rows_CPU214 + [
   - rows_tot_CPU214 * m_access_time_CPU214 ] = 0
 pdt_m_rows_CPU208: pdt_m_rows_CPU208 + [
   - rows_tot_CPU208 * m_access_time_CPU208 ] = 0
Bounds
 frac[0,1] <= 1
 frac[7,0] <= 1
 frac[3,0] <= 1
 frac[6,1] <= 1
 frac[3,1] <= 1
 frac[6,0] <= 1
 frac[2,1] <= 1
 frac[2,2] <= 1
 frac[1,0] <= 1
 frac[4,0] <= 1
 mem_tot_CPU1770 <= 32768
 normalized_mem_tot_CPU1770 <= 1
 normalized_rows_tot_CPU1770 <= 1
 mem_tot_CPU1582 <= 32768
 normalized_mem_tot_CPU1582 <= 1
 normalized_rows_tot_CPU1582 <= 1
 mem_tot_CPU1472 <= 32768
 normalized_mem_tot_CPU1472 <= 1
 normalized_rows_tot_CPU1472 <= 1
 mem_tot_CPU1642 <= 32768
 normalized_mem_tot_CPU1642 <= 1
 normalized_rows_tot_CPU1642 <= 1
 mem_tot_CPU826 <= 32768
 normalized_mem_tot_CPU826 <= 1
 normalized_rows_tot_CPU826 <= 1
 mem_tot_CPU313 <= 32768
 normalized_mem_tot_CPU313 <= 1
 normalized_rows_tot_CPU313 <= 1
 mem_tot_CPU214 <= 32768
 normalized_mem_tot_CPU214 <= 1
 normalized_rows_tot_CPU214 <= 1
 mem_tot_CPU208 <= 32768
 normalized_mem_tot_CPU208 <= 1
 normalized_rows_tot_CPU208 <= 1
 cores_sketch_CPU1770 <= 7
 1 <= cores_dpdk_CPU1770 <= 7
 cores_sketch_CPU1582 <= 7
 1 <= cores_dpdk_CPU1582 <= 7
 cores_sketch_CPU1472 <= 7
 1 <= cores_dpdk_CPU1472 <= 7
 cores_sketch_CPU1642 <= 7
 1 <= cores_dpdk_CPU1642 <= 7
 cores_sketch_CPU826 <= 7
 1 <= cores_dpdk_CPU826 <= 7
 cores_sketch_CPU313 <= 7
 1 <= cores_dpdk_CPU313 <= 7
 cores_sketch_CPU214 <= 7
 1 <= cores_dpdk_CPU214 <= 7
 cores_sketch_CPU208 <= 7
 1 <= cores_dpdk_CPU208 <= 7
Generals
 cores_sketch_CPU1770 cores_dpdk_CPU1770 cores_sketch_CPU1582
 cores_dpdk_CPU1582 cores_sketch_CPU1472 cores_dpdk_CPU1472
 cores_sketch_CPU1642 cores_dpdk_CPU1642 cores_sketch_CPU826
 cores_dpdk_CPU826 cores_sketch_CPU313 cores_dpdk_CPU313
 cores_sketch_CPU214 cores_dpdk_CPU214 cores_sketch_CPU208
 cores_dpdk_CPU208
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU1770 , normalized_mem_tot_CPU1582 ,
   normalized_mem_tot_CPU1472 , normalized_mem_tot_CPU1642 ,
   normalized_mem_tot_CPU826 , normalized_mem_tot_CPU313 ,
   normalized_mem_tot_CPU214 , normalized_mem_tot_CPU208 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX (
   normalized_rows_tot_CPU1770 , normalized_rows_tot_CPU1582 ,
   normalized_rows_tot_CPU1472 , normalized_rows_tot_CPU1642 ,
   normalized_rows_tot_CPU826 , normalized_rows_tot_CPU313 ,
   normalized_rows_tot_CPU214 , normalized_rows_tot_CPU208 )
 mem_access_time_CPU1770:
   m_access_time_CPU1770 = PWL ( mem_tot_CPU1770 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1582:
   m_access_time_CPU1582 = PWL ( mem_tot_CPU1582 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1472:
   m_access_time_CPU1472 = PWL ( mem_tot_CPU1472 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1642:
   m_access_time_CPU1642 = PWL ( mem_tot_CPU1642 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU826: m_access_time_CPU826 = PWL ( mem_tot_CPU826 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU313: m_access_time_CPU313 = PWL ( mem_tot_CPU313 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU214: m_access_time_CPU214 = PWL ( mem_tot_CPU214 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU208: m_access_time_CPU208 = PWL ( mem_tot_CPU208 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

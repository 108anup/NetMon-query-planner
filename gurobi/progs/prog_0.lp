\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_CPU_Cluster + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU1652
   + 0 m_access_time_CPU413 + 0 m_access_time_CPU2470
   + 0 m_access_time_CPU3929 + 0 m_access_time_CPU777
   + 0 m_access_time_CPU2819 + 0 m_access_time_CPU837
   + 0 m_access_time_CPU1288 + res
Subject To
 cov_0_2372: frac[2,2372] + frac[6,2372] >= 0.7
 cov_1_2375: frac[0,2375] + frac[1,2375] >= 0.7
 cov_2_2372: frac[1,2372] + frac[7,2372] >= 0.9
 cov_3_2372: frac[6,2372] + frac[7,2372] >= 0.9
 cov_4_2373: frac[1,2373] + frac[3,2373] >= 0.8
 accuracy_6_2372: - 50.9765625 frac[6,2372] + mem[6,2372] = 0
 accuracy_1_2372: - 50.9765625 frac[1,2372] + mem[1,2372] = 0
 accuracy_0_2375: - 50.9765625 frac[0,2375] + mem[0,2375] = 0
 accuracy_7_2372: - 50.9765625 frac[7,2372] + mem[7,2372] = 0
 accuracy_1_2373: - 50.9765625 frac[1,2373] + mem[1,2373] = 0
 accuracy_1_2375: - 50.9765625 frac[1,2375] + mem[1,2375] = 0
 accuracy_2_2372: - 50.9765625 frac[2,2372] + mem[2,2372] = 0
 accuracy_3_2373: - 50.9765625 frac[3,2373] + mem[3,2373] = 0
 mem_tot_CPU1652: - mem[0,2375] + mem_tot_CPU1652 = 0
 R14: - 3.05176e-05 mem_tot_CPU1652 + normalized_mem_tot_CPU1652 = 0
 rows_tot_CPU1652: - 3 frac[0,2375] + rows_tot_CPU1652 = 0
 R16: - 0.0833333333333333 rows_tot_CPU1652 + normalized_rows_tot_CPU1652
   = 0
 mem_tot_CPU413: - mem[1,2372] - mem[1,2373] - mem[1,2375] + mem_tot_CPU413
   = 0
 R18: - 3.05176e-05 mem_tot_CPU413 + normalized_mem_tot_CPU413 = 0
 rows_tot_CPU413: - 3 frac[1,2372] - 3 frac[1,2373] - 3 frac[1,2375]
   + rows_tot_CPU413 = 0
 R20: - 0.0833333333333333 rows_tot_CPU413 + normalized_rows_tot_CPU413
   = 0
 mem_tot_CPU2470: - mem[2,2372] + mem_tot_CPU2470 = 0
 R22: - 3.05176e-05 mem_tot_CPU2470 + normalized_mem_tot_CPU2470 = 0
 rows_tot_CPU2470: - 3 frac[2,2372] + rows_tot_CPU2470 = 0
 R24: - 0.0833333333333333 rows_tot_CPU2470 + normalized_rows_tot_CPU2470
   = 0
 mem_tot_CPU3929: - mem[3,2373] + mem_tot_CPU3929 = 0
 R26: - 3.05176e-05 mem_tot_CPU3929 + normalized_mem_tot_CPU3929 = 0
 rows_tot_CPU3929: - 3 frac[3,2373] + rows_tot_CPU3929 = 0
 R28: - 0.0833333333333333 rows_tot_CPU3929 + normalized_rows_tot_CPU3929
   = 0
 mem_tot_CPU777: mem_tot_CPU777 = 0
 R30: - 3.05176e-05 mem_tot_CPU777 + normalized_mem_tot_CPU777 = 0
 rows_tot_CPU777: rows_tot_CPU777 = 0
 R32: - 0.0833333333333333 rows_tot_CPU777 + normalized_rows_tot_CPU777
   = 0
 mem_tot_CPU2819: mem_tot_CPU2819 = 0
 R34: - 3.05176e-05 mem_tot_CPU2819 + normalized_mem_tot_CPU2819 = 0
 rows_tot_CPU2819: rows_tot_CPU2819 = 0
 R36: - 0.0833333333333333 rows_tot_CPU2819 + normalized_rows_tot_CPU2819
   = 0
 mem_tot_CPU837: - mem[6,2372] + mem_tot_CPU837 = 0
 R38: - 3.05176e-05 mem_tot_CPU837 + normalized_mem_tot_CPU837 = 0
 rows_tot_CPU837: - 3 frac[6,2372] + rows_tot_CPU837 = 0
 R40: - 0.0833333333333333 rows_tot_CPU837 + normalized_rows_tot_CPU837
   = 0
 mem_tot_CPU1288: - mem[7,2372] + mem_tot_CPU1288 = 0
 R42: - 3.05176e-05 mem_tot_CPU1288 + normalized_mem_tot_CPU1288 = 0
 rows_tot_CPU1288: - 3 frac[7,2372] + rows_tot_CPU1288 = 0
 R44: - 0.0833333333333333 rows_tot_CPU1288 + normalized_rows_tot_CPU1288
   = 0
 row_capacity_CPU1652: rows_tot_CPU1652 <= 12
 row_capacity_CPU413: rows_tot_CPU413 <= 12
 row_capacity_CPU2470: rows_tot_CPU2470 <= 12
 row_capacity_CPU3929: rows_tot_CPU3929 <= 12
 row_capacity_CPU777: rows_tot_CPU777 <= 12
 row_capacity_CPU2819: rows_tot_CPU2819 <= 12
 row_capacity_CPU837: rows_tot_CPU837 <= 12
 row_capacity_CPU1288: rows_tot_CPU1288 <= 12
 ns_single_CPU1652: 3.5 rows_tot_CPU1652 - ns_single_CPU1652
   + pdt_m_rows_CPU1652 = 0
 ns_sketch_CPU1652: - ns_single_CPU1652
   + 10.71434571428571 cores_sketch_CPU1652 >= 0
 capacity_cores_CPU1652: cores_sketch_CPU1652 <= 1
 ns_single_CPU413: 3.5 rows_tot_CPU413 - ns_single_CPU413
   + pdt_m_rows_CPU413 = 0
 ns_sketch_CPU413: - ns_single_CPU413
   + 10.71434571428571 cores_sketch_CPU413 >= 0
 capacity_cores_CPU413: cores_sketch_CPU413 <= 1
 ns_single_CPU2470: 3.5 rows_tot_CPU2470 - ns_single_CPU2470
   + pdt_m_rows_CPU2470 = 0
 ns_sketch_CPU2470: - ns_single_CPU2470
   + 10.71434571428571 cores_sketch_CPU2470 >= 0
 capacity_cores_CPU2470: cores_sketch_CPU2470 <= 1
 ns_single_CPU3929: 3.5 rows_tot_CPU3929 - ns_single_CPU3929
   + pdt_m_rows_CPU3929 = 0
 ns_sketch_CPU3929: - ns_single_CPU3929
   + 10.71434571428571 cores_sketch_CPU3929 >= 0
 capacity_cores_CPU3929: cores_sketch_CPU3929 <= 1
 ns_single_CPU777: 3.5 rows_tot_CPU777 - ns_single_CPU777
   + pdt_m_rows_CPU777 = 0
 ns_sketch_CPU777: - ns_single_CPU777
   + 10.71434571428571 cores_sketch_CPU777 >= 0
 capacity_cores_CPU777: cores_sketch_CPU777 <= 1
 ns_single_CPU2819: 3.5 rows_tot_CPU2819 - ns_single_CPU2819
   + pdt_m_rows_CPU2819 = 0
 ns_sketch_CPU2819: - ns_single_CPU2819
   + 10.71434571428571 cores_sketch_CPU2819 >= 0
 capacity_cores_CPU2819: cores_sketch_CPU2819 <= 1
 ns_single_CPU837: 3.5 rows_tot_CPU837 - ns_single_CPU837
   + pdt_m_rows_CPU837 = 0
 ns_sketch_CPU837: - ns_single_CPU837
   + 10.71434571428571 cores_sketch_CPU837 >= 0
 capacity_cores_CPU837: cores_sketch_CPU837 <= 1
 ns_single_CPU1288: 3.5 rows_tot_CPU1288 - ns_single_CPU1288
   + pdt_m_rows_CPU1288 = 0
 ns_sketch_CPU1288: - ns_single_CPU1288
   + 10.71434571428571 cores_sketch_CPU1288 >= 0
 capacity_cores_CPU1288: cores_sketch_CPU1288 <= 1
 res_acc: 1.220703125e-04 mem_tot_CPU1652 + 1.220703125e-04 mem_tot_CPU413
   + 1.220703125e-04 mem_tot_CPU2470 + 1.220703125e-04 mem_tot_CPU3929
   + 1.220703125e-04 mem_tot_CPU777 + 1.220703125e-04 mem_tot_CPU2819
   + 1.220703125e-04 mem_tot_CPU837 + 1.220703125e-04 mem_tot_CPU1288
   + 10 cores_sketch_CPU1652 + 10 cores_sketch_CPU413
   + 10 cores_sketch_CPU2470 + 10 cores_sketch_CPU3929
   + 10 cores_sketch_CPU777 + 10 cores_sketch_CPU2819
   + 10 cores_sketch_CPU837 + 10 cores_sketch_CPU1288 - res = -480
 pdt_m_rows_CPU1652: pdt_m_rows_CPU1652 + [
   - rows_tot_CPU1652 * m_access_time_CPU1652 ] = 0
 pdt_m_rows_CPU413: pdt_m_rows_CPU413 + [
   - rows_tot_CPU413 * m_access_time_CPU413 ] = 0
 pdt_m_rows_CPU2470: pdt_m_rows_CPU2470 + [
   - rows_tot_CPU2470 * m_access_time_CPU2470 ] = 0
 pdt_m_rows_CPU3929: pdt_m_rows_CPU3929 + [
   - rows_tot_CPU3929 * m_access_time_CPU3929 ] = 0
 pdt_m_rows_CPU777: pdt_m_rows_CPU777 + [
   - rows_tot_CPU777 * m_access_time_CPU777 ] = 0
 pdt_m_rows_CPU2819: pdt_m_rows_CPU2819 + [
   - rows_tot_CPU2819 * m_access_time_CPU2819 ] = 0
 pdt_m_rows_CPU837: pdt_m_rows_CPU837 + [
   - rows_tot_CPU837 * m_access_time_CPU837 ] = 0
 pdt_m_rows_CPU1288: pdt_m_rows_CPU1288 + [
   - rows_tot_CPU1288 * m_access_time_CPU1288 ] = 0
Bounds
 frac[0,2375] <= 1
 frac[1,2372] <= 1
 frac[1,2373] <= 1
 frac[1,2375] <= 1
 frac[2,2372] <= 1
 frac[3,2373] <= 1
 frac[6,2372] <= 1
 frac[7,2372] <= 1
 mem_tot_CPU1652 <= 32768
 normalized_mem_tot_CPU1652 <= 1
 normalized_rows_tot_CPU1652 <= 1
 mem_tot_CPU413 <= 32768
 normalized_mem_tot_CPU413 <= 1
 normalized_rows_tot_CPU413 <= 1
 mem_tot_CPU2470 <= 32768
 normalized_mem_tot_CPU2470 <= 1
 normalized_rows_tot_CPU2470 <= 1
 mem_tot_CPU3929 <= 32768
 normalized_mem_tot_CPU3929 <= 1
 normalized_rows_tot_CPU3929 <= 1
 mem_tot_CPU777 <= 32768
 normalized_mem_tot_CPU777 <= 1
 normalized_rows_tot_CPU777 <= 1
 mem_tot_CPU2819 <= 32768
 normalized_mem_tot_CPU2819 <= 1
 normalized_rows_tot_CPU2819 <= 1
 mem_tot_CPU837 <= 32768
 normalized_mem_tot_CPU837 <= 1
 normalized_rows_tot_CPU837 <= 1
 mem_tot_CPU1288 <= 32768
 normalized_mem_tot_CPU1288 <= 1
 normalized_rows_tot_CPU1288 <= 1
 cores_sketch_CPU1652 <= 7
 cores_sketch_CPU413 <= 7
 cores_sketch_CPU2470 <= 7
 cores_sketch_CPU3929 <= 7
 cores_sketch_CPU777 <= 7
 cores_sketch_CPU2819 <= 7
 cores_sketch_CPU837 <= 7
 cores_sketch_CPU1288 <= 7
Generals
 cores_sketch_CPU1652 cores_sketch_CPU413 cores_sketch_CPU2470
 cores_sketch_CPU3929 cores_sketch_CPU777 cores_sketch_CPU2819
 cores_sketch_CPU837 cores_sketch_CPU1288
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU1652 , normalized_mem_tot_CPU413 ,
   normalized_mem_tot_CPU2470 , normalized_mem_tot_CPU3929 ,
   normalized_mem_tot_CPU777 , normalized_mem_tot_CPU2819 ,
   normalized_mem_tot_CPU837 , normalized_mem_tot_CPU1288 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX (
   normalized_rows_tot_CPU1652 , normalized_rows_tot_CPU413 ,
   normalized_rows_tot_CPU2470 , normalized_rows_tot_CPU3929 ,
   normalized_rows_tot_CPU777 , normalized_rows_tot_CPU2819 ,
   normalized_rows_tot_CPU837 , normalized_rows_tot_CPU1288 )
 mem_access_time_CPU1652:
   m_access_time_CPU1652 = PWL ( mem_tot_CPU1652 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU413: m_access_time_CPU413 = PWL ( mem_tot_CPU413 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU2470:
   m_access_time_CPU2470 = PWL ( mem_tot_CPU2470 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU3929:
   m_access_time_CPU3929 = PWL ( mem_tot_CPU3929 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU777: m_access_time_CPU777 = PWL ( mem_tot_CPU777 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU2819:
   m_access_time_CPU2819 = PWL ( mem_tot_CPU2819 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU837: m_access_time_CPU837 = PWL ( mem_tot_CPU837 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1288:
   m_access_time_CPU1288 = PWL ( mem_tot_CPU1288 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

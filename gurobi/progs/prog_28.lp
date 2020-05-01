\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 m_access_time_CPU711 + 0 m_access_time_CPU110 + 0 m_access_time_CPU338
   + 0 m_access_time_CPU1266 + 0 m_access_time_CPU695
   + 0 m_access_time_CPU147 + 0 m_access_time_CPU159
   + 0 m_access_time_CPU852 + res
Subject To
 cov_0_957: frac[3,957] >= 0.7863817742403144
 cov_1_956: frac[3,956] + frac[7,956] >= 0.7
 cov_2_958: frac[0,958] + frac[3,958] >= 0.7
 cov_3_956: frac[1,956] >= 0.7
 accuracy_3_958: - 50.9765625 frac[3,958] + mem[3,958] = 0
 accuracy_3_956: - 50.9765625 frac[3,956] + mem[3,956] = 0
 accuracy_1_956: - 50.9765625 frac[1,956] + mem[1,956] = 0
 accuracy_3_957: - 50.9765625 frac[3,957] + mem[3,957] = 0
 accuracy_7_956: - 50.9765625 frac[7,956] + mem[7,956] = 0
 accuracy_0_958: - 50.9765625 frac[0,958] + mem[0,958] = 0
 mem_tot_CPU711: - mem[0,958] + mem_tot_CPU711 = 0
 R11: - 3.05176e-05 mem_tot_CPU711 + normalized_mem_tot_CPU711 = 0
 rows_tot_CPU711: - 3 frac[0,958] + rows_tot_CPU711 = 0
 R13: - 0.0833333333333333 rows_tot_CPU711 + normalized_rows_tot_CPU711
   = 0
 mem_tot_CPU110: - mem[1,956] + mem_tot_CPU110 = 0
 R15: - 3.05176e-05 mem_tot_CPU110 + normalized_mem_tot_CPU110 = 0
 rows_tot_CPU110: - 3 frac[1,956] + rows_tot_CPU110 = 0
 R17: - 0.0833333333333333 rows_tot_CPU110 + normalized_rows_tot_CPU110
   = 0
 mem_tot_CPU338: mem_tot_CPU338 = 0
 R19: - 3.05176e-05 mem_tot_CPU338 + normalized_mem_tot_CPU338 = 0
 rows_tot_CPU338: rows_tot_CPU338 = 0
 R21: - 0.0833333333333333 rows_tot_CPU338 + normalized_rows_tot_CPU338
   = 0
 mem_tot_CPU1266: - mem[3,958] - mem[3,956] - mem[3,957] + mem_tot_CPU1266
   = 0
 R23: - 3.05176e-05 mem_tot_CPU1266 + normalized_mem_tot_CPU1266 = 0
 rows_tot_CPU1266: - 3 frac[3,958] - 3 frac[3,956] - 3 frac[3,957]
   + rows_tot_CPU1266 = 0
 R25: - 0.0833333333333333 rows_tot_CPU1266 + normalized_rows_tot_CPU1266
   = 0
 mem_tot_CPU695: mem_tot_CPU695 = 0
 R27: - 3.05176e-05 mem_tot_CPU695 + normalized_mem_tot_CPU695 = 0
 rows_tot_CPU695: rows_tot_CPU695 = 0
 R29: - 0.0833333333333333 rows_tot_CPU695 + normalized_rows_tot_CPU695
   = 0
 mem_tot_CPU147: mem_tot_CPU147 = 0
 R31: - 3.05176e-05 mem_tot_CPU147 + normalized_mem_tot_CPU147 = 0
 rows_tot_CPU147: rows_tot_CPU147 = 0
 R33: - 0.0833333333333333 rows_tot_CPU147 + normalized_rows_tot_CPU147
   = 0
 mem_tot_CPU159: mem_tot_CPU159 = 0
 R35: - 3.05176e-05 mem_tot_CPU159 + normalized_mem_tot_CPU159 = 0
 rows_tot_CPU159: rows_tot_CPU159 = 0
 R37: - 0.0833333333333333 rows_tot_CPU159 + normalized_rows_tot_CPU159
   = 0
 mem_tot_CPU852: - mem[7,956] + mem_tot_CPU852 = 0
 R39: - 3.05176e-05 mem_tot_CPU852 + normalized_mem_tot_CPU852 = 0
 rows_tot_CPU852: - 3 frac[7,956] + rows_tot_CPU852 = 0
 R41: - 0.0833333333333333 rows_tot_CPU852 + normalized_rows_tot_CPU852
   = 0
 row_capacity_CPU711: rows_tot_CPU711 <= 12
 row_capacity_CPU110: rows_tot_CPU110 <= 12
 row_capacity_CPU338: rows_tot_CPU338 <= 12
 row_capacity_CPU1266: rows_tot_CPU1266 <= 12
 row_capacity_CPU695: rows_tot_CPU695 <= 12
 row_capacity_CPU147: rows_tot_CPU147 <= 12
 row_capacity_CPU159: rows_tot_CPU159 <= 12
 row_capacity_CPU852: rows_tot_CPU852 <= 12
 ns_single_CPU711: 3.5 rows_tot_CPU711 - ns_single_CPU711
   + pdt_m_rows_CPU711 = 0
 ns_sketch_CPU711: - ns_single_CPU711
   + 10.71428571428572 cores_sketch_CPU711 >= 0
 capacity_cores_CPU711: cores_sketch_CPU711 <= 1
 ns_single_CPU110: 3.5 rows_tot_CPU110 - ns_single_CPU110
   + pdt_m_rows_CPU110 = 0
 ns_sketch_CPU110: - ns_single_CPU110
   + 10.71428571428572 cores_sketch_CPU110 >= 0
 capacity_cores_CPU110: cores_sketch_CPU110 <= 1
 ns_single_CPU338: 3.5 rows_tot_CPU338 - ns_single_CPU338
   + pdt_m_rows_CPU338 = 0
 ns_sketch_CPU338: - ns_single_CPU338
   + 10.71428571428572 cores_sketch_CPU338 >= 0
 capacity_cores_CPU338: cores_sketch_CPU338 <= 1
 ns_single_CPU1266: 3.5 rows_tot_CPU1266 - ns_single_CPU1266
   + pdt_m_rows_CPU1266 = 0
 ns_sketch_CPU1266: - ns_single_CPU1266
   + 10.71428571428572 cores_sketch_CPU1266 >= 0
 capacity_cores_CPU1266: cores_sketch_CPU1266 <= 1
 ns_single_CPU695: 3.5 rows_tot_CPU695 - ns_single_CPU695
   + pdt_m_rows_CPU695 = 0
 ns_sketch_CPU695: - ns_single_CPU695
   + 10.71428571428572 cores_sketch_CPU695 >= 0
 capacity_cores_CPU695: cores_sketch_CPU695 <= 1
 ns_single_CPU147: 3.5 rows_tot_CPU147 - ns_single_CPU147
   + pdt_m_rows_CPU147 = 0
 ns_sketch_CPU147: - ns_single_CPU147
   + 10.71428571428572 cores_sketch_CPU147 >= 0
 capacity_cores_CPU147: cores_sketch_CPU147 <= 1
 ns_single_CPU159: 3.5 rows_tot_CPU159 - ns_single_CPU159
   + pdt_m_rows_CPU159 = 0
 ns_sketch_CPU159: - ns_single_CPU159
   + 10.71428571428572 cores_sketch_CPU159 >= 0
 capacity_cores_CPU159: cores_sketch_CPU159 <= 1
 ns_single_CPU852: 3.5 rows_tot_CPU852 - ns_single_CPU852
   + pdt_m_rows_CPU852 = 0
 ns_sketch_CPU852: - ns_single_CPU852
   + 10.71428571428572 cores_sketch_CPU852 >= 0
 capacity_cores_CPU852: cores_sketch_CPU852 <= 1
 res_acc: 1.220703125e-04 mem_tot_CPU711 + 1.220703125e-04 mem_tot_CPU110
   + 1.220703125e-04 mem_tot_CPU338 + 1.220703125e-04 mem_tot_CPU1266
   + 1.220703125e-04 mem_tot_CPU695 + 1.220703125e-04 mem_tot_CPU147
   + 1.220703125e-04 mem_tot_CPU159 + 1.220703125e-04 mem_tot_CPU852
   + 10 cores_sketch_CPU711 + 10 cores_sketch_CPU110
   + 10 cores_sketch_CPU338 + 10 cores_sketch_CPU1266
   + 10 cores_sketch_CPU695 + 10 cores_sketch_CPU147
   + 10 cores_sketch_CPU159 + 10 cores_sketch_CPU852 - res = -480
 pdt_m_rows_CPU711: pdt_m_rows_CPU711 + [
   - rows_tot_CPU711 * m_access_time_CPU711 ] = 0
 pdt_m_rows_CPU110: pdt_m_rows_CPU110 + [
   - rows_tot_CPU110 * m_access_time_CPU110 ] = 0
 pdt_m_rows_CPU338: pdt_m_rows_CPU338 + [
   - rows_tot_CPU338 * m_access_time_CPU338 ] = 0
 pdt_m_rows_CPU1266: pdt_m_rows_CPU1266 + [
   - rows_tot_CPU1266 * m_access_time_CPU1266 ] = 0
 pdt_m_rows_CPU695: pdt_m_rows_CPU695 + [
   - rows_tot_CPU695 * m_access_time_CPU695 ] = 0
 pdt_m_rows_CPU147: pdt_m_rows_CPU147 + [
   - rows_tot_CPU147 * m_access_time_CPU147 ] = 0
 pdt_m_rows_CPU159: pdt_m_rows_CPU159 + [
   - rows_tot_CPU159 * m_access_time_CPU159 ] = 0
 pdt_m_rows_CPU852: pdt_m_rows_CPU852 + [
   - rows_tot_CPU852 * m_access_time_CPU852 ] = 0
Bounds
 frac[0,958] <= 1
 frac[1,956] <= 1
 frac[3,958] <= 1
 frac[3,956] <= 1
 frac[3,957] <= 1
 frac[7,956] <= 1
 mem_tot_CPU711 <= 32768
 normalized_mem_tot_CPU711 <= 1
 normalized_rows_tot_CPU711 <= 1
 mem_tot_CPU110 <= 32768
 normalized_mem_tot_CPU110 <= 1
 normalized_rows_tot_CPU110 <= 1
 mem_tot_CPU338 <= 32768
 normalized_mem_tot_CPU338 <= 1
 normalized_rows_tot_CPU338 <= 1
 mem_tot_CPU1266 <= 32768
 normalized_mem_tot_CPU1266 <= 1
 normalized_rows_tot_CPU1266 <= 1
 mem_tot_CPU695 <= 32768
 normalized_mem_tot_CPU695 <= 1
 normalized_rows_tot_CPU695 <= 1
 mem_tot_CPU147 <= 32768
 normalized_mem_tot_CPU147 <= 1
 normalized_rows_tot_CPU147 <= 1
 mem_tot_CPU159 <= 32768
 normalized_mem_tot_CPU159 <= 1
 normalized_rows_tot_CPU159 <= 1
 mem_tot_CPU852 <= 32768
 normalized_mem_tot_CPU852 <= 1
 normalized_rows_tot_CPU852 <= 1
 cores_sketch_CPU711 <= 7
 cores_sketch_CPU110 <= 7
 cores_sketch_CPU338 <= 7
 cores_sketch_CPU1266 <= 7
 cores_sketch_CPU695 <= 7
 cores_sketch_CPU147 <= 7
 cores_sketch_CPU159 <= 7
 cores_sketch_CPU852 <= 7
Generals
 cores_sketch_CPU711 cores_sketch_CPU110 cores_sketch_CPU338
 cores_sketch_CPU1266 cores_sketch_CPU695 cores_sketch_CPU147
 cores_sketch_CPU159 cores_sketch_CPU852
General Constraints
 mem_access_time_CPU711: m_access_time_CPU711 = PWL ( mem_tot_CPU711 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU110: m_access_time_CPU110 = PWL ( mem_tot_CPU110 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU338: m_access_time_CPU338 = PWL ( mem_tot_CPU338 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1266:
   m_access_time_CPU1266 = PWL ( mem_tot_CPU1266 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU695: m_access_time_CPU695 = PWL ( mem_tot_CPU695 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU147: m_access_time_CPU147 = PWL ( mem_tot_CPU147 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU159: m_access_time_CPU159 = PWL ( mem_tot_CPU159 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU852: m_access_time_CPU852 = PWL ( mem_tot_CPU852 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 m_access_time_CPU4261 + 0 m_access_time_CPU6715
   + 0 m_access_time_CPU1522 + 0 m_access_time_CPU1018
   + 0 m_access_time_CPU1612 + 0 m_access_time_CPU9194
   + 0 m_access_time_CPU4085 + 0 m_access_time_CPU7795 + res
Subject To
 cov_0_1524: frac[4,1524] + frac[7,1524] >= 0.8
 cov_1_1527: frac[0,1527] + frac[5,1527] >= 0.7
 cov_2_1527: frac[4,1527] + frac[7,1527] >= 0.7
 cov_3_1524: frac[6,1524] + frac[7,1524] >= 0.7
 cov_4_1525: frac[1,1525] + frac[2,1525] >= 0.8
 cov_5_1526: frac[1,1526] + frac[5,1526] >= 0.8
 cov_6_1527: frac[3,1527] + frac[5,1527] >= 0.8
 accuracy_7_1524: - 50.9765625 frac[7,1524] + mem[7,1524] = 0
 accuracy_1_1525: - 50.9765625 frac[1,1525] + mem[1,1525] = 0
 accuracy_1_1526: - 50.9765625 frac[1,1526] + mem[1,1526] = 0
 accuracy_3_1527: - 50.9765625 frac[3,1527] + mem[3,1527] = 0
 accuracy_0_1527: - 50.9765625 frac[0,1527] + mem[0,1527] = 0
 accuracy_2_1525: - 50.9765625 frac[2,1525] + mem[2,1525] = 0
 accuracy_6_1524: - 50.9765625 frac[6,1524] + mem[6,1524] = 0
 accuracy_4_1527: - 50.9765625 frac[4,1527] + mem[4,1527] = 0
 accuracy_5_1526: - 50.9765625 frac[5,1526] + mem[5,1526] = 0
 accuracy_5_1527: - 50.9765625 frac[5,1527] + mem[5,1527] = 0
 accuracy_7_1527: - 50.9765625 frac[7,1527] + mem[7,1527] = 0
 accuracy_4_1524: - 50.9765625 frac[4,1524] + mem[4,1524] = 0
 mem_tot_CPU4261: - mem[0,1527] + mem_tot_CPU4261 = 0
 R20: - 3.05176e-05 mem_tot_CPU4261 + normalized_mem_tot_CPU4261 = 0
 rows_tot_CPU4261: - 3 frac[0,1527] + rows_tot_CPU4261 = 0
 R22: - 0.0833333333333333 rows_tot_CPU4261 + normalized_rows_tot_CPU4261
   = 0
 mem_tot_CPU6715: - mem[1,1525] - mem[1,1526] + mem_tot_CPU6715 = 0
 R24: - 3.05176e-05 mem_tot_CPU6715 + normalized_mem_tot_CPU6715 = 0
 rows_tot_CPU6715: - 3 frac[1,1525] - 3 frac[1,1526] + rows_tot_CPU6715
   = 0
 R26: - 0.0833333333333333 rows_tot_CPU6715 + normalized_rows_tot_CPU6715
   = 0
 mem_tot_CPU1522: - mem[2,1525] + mem_tot_CPU1522 = 0
 R28: - 3.05176e-05 mem_tot_CPU1522 + normalized_mem_tot_CPU1522 = 0
 rows_tot_CPU1522: - 3 frac[2,1525] + rows_tot_CPU1522 = 0
 R30: - 0.0833333333333333 rows_tot_CPU1522 + normalized_rows_tot_CPU1522
   = 0
 mem_tot_CPU1018: - mem[3,1527] + mem_tot_CPU1018 = 0
 R32: - 3.05176e-05 mem_tot_CPU1018 + normalized_mem_tot_CPU1018 = 0
 rows_tot_CPU1018: - 3 frac[3,1527] + rows_tot_CPU1018 = 0
 R34: - 0.0833333333333333 rows_tot_CPU1018 + normalized_rows_tot_CPU1018
   = 0
 mem_tot_CPU1612: - mem[4,1527] - mem[4,1524] + mem_tot_CPU1612 = 0
 R36: - 3.05176e-05 mem_tot_CPU1612 + normalized_mem_tot_CPU1612 = 0
 rows_tot_CPU1612: - 3 frac[4,1527] - 3 frac[4,1524] + rows_tot_CPU1612
   = 0
 R38: - 0.0833333333333333 rows_tot_CPU1612 + normalized_rows_tot_CPU1612
   = 0
 mem_tot_CPU9194: - mem[5,1526] - mem[5,1527] + mem_tot_CPU9194 = 0
 R40: - 3.05176e-05 mem_tot_CPU9194 + normalized_mem_tot_CPU9194 = 0
 rows_tot_CPU9194: - 3 frac[5,1526] - 3 frac[5,1527] + rows_tot_CPU9194
   = 0
 R42: - 0.0833333333333333 rows_tot_CPU9194 + normalized_rows_tot_CPU9194
   = 0
 mem_tot_CPU4085: - mem[6,1524] + mem_tot_CPU4085 = 0
 R44: - 3.05176e-05 mem_tot_CPU4085 + normalized_mem_tot_CPU4085 = 0
 rows_tot_CPU4085: - 3 frac[6,1524] + rows_tot_CPU4085 = 0
 R46: - 0.0833333333333333 rows_tot_CPU4085 + normalized_rows_tot_CPU4085
   = 0
 mem_tot_CPU7795: - mem[7,1524] - mem[7,1527] + mem_tot_CPU7795 = 0
 R48: - 3.05176e-05 mem_tot_CPU7795 + normalized_mem_tot_CPU7795 = 0
 rows_tot_CPU7795: - 3 frac[7,1524] - 3 frac[7,1527] + rows_tot_CPU7795
   = 0
 R50: - 0.0833333333333333 rows_tot_CPU7795 + normalized_rows_tot_CPU7795
   = 0
 row_capacity_CPU4261: rows_tot_CPU4261 <= 12
 row_capacity_CPU6715: rows_tot_CPU6715 <= 12
 row_capacity_CPU1522: rows_tot_CPU1522 <= 12
 row_capacity_CPU1018: rows_tot_CPU1018 <= 12
 row_capacity_CPU1612: rows_tot_CPU1612 <= 12
 row_capacity_CPU9194: rows_tot_CPU9194 <= 12
 row_capacity_CPU4085: rows_tot_CPU4085 <= 12
 row_capacity_CPU7795: rows_tot_CPU7795 <= 12
 ns_single_CPU4261: 3.5 rows_tot_CPU4261 - ns_single_CPU4261
   + pdt_m_rows_CPU4261 = 0
 ns_sketch_CPU4261: - ns_single_CPU4261
   + 11.42857142857143 cores_sketch_CPU4261 >= 0
 capacity_cores_CPU4261: cores_sketch_CPU4261 <= 2
 ns_single_CPU6715: 3.5 rows_tot_CPU6715 - ns_single_CPU6715
   + pdt_m_rows_CPU6715 = 0
 ns_sketch_CPU6715: - ns_single_CPU6715
   + 11.42857142857143 cores_sketch_CPU6715 >= 0
 capacity_cores_CPU6715: cores_sketch_CPU6715 <= 2
 ns_single_CPU1522: 3.5 rows_tot_CPU1522 - ns_single_CPU1522
   + pdt_m_rows_CPU1522 = 0
 ns_sketch_CPU1522: - ns_single_CPU1522
   + 11.42857142857143 cores_sketch_CPU1522 >= 0
 capacity_cores_CPU1522: cores_sketch_CPU1522 <= 2
 ns_single_CPU1018: 3.5 rows_tot_CPU1018 - ns_single_CPU1018
   + pdt_m_rows_CPU1018 = 0
 ns_sketch_CPU1018: - ns_single_CPU1018
   + 11.42857142857143 cores_sketch_CPU1018 >= 0
 capacity_cores_CPU1018: cores_sketch_CPU1018 <= 2
 ns_single_CPU1612: 3.5 rows_tot_CPU1612 - ns_single_CPU1612
   + pdt_m_rows_CPU1612 = 0
 ns_sketch_CPU1612: - ns_single_CPU1612
   + 11.42857142857143 cores_sketch_CPU1612 >= 0
 capacity_cores_CPU1612: cores_sketch_CPU1612 <= 2
 ns_single_CPU9194: 3.5 rows_tot_CPU9194 - ns_single_CPU9194
   + pdt_m_rows_CPU9194 = 0
 ns_sketch_CPU9194: - ns_single_CPU9194
   + 11.42857142857143 cores_sketch_CPU9194 >= 0
 capacity_cores_CPU9194: cores_sketch_CPU9194 <= 2
 ns_single_CPU4085: 3.5 rows_tot_CPU4085 - ns_single_CPU4085
   + pdt_m_rows_CPU4085 = 0
 ns_sketch_CPU4085: - ns_single_CPU4085
   + 11.42857142857143 cores_sketch_CPU4085 >= 0
 capacity_cores_CPU4085: cores_sketch_CPU4085 <= 2
 ns_single_CPU7795: 3.5 rows_tot_CPU7795 - ns_single_CPU7795
   + pdt_m_rows_CPU7795 = 0
 ns_sketch_CPU7795: - ns_single_CPU7795
   + 11.42857142857143 cores_sketch_CPU7795 >= 0
 capacity_cores_CPU7795: cores_sketch_CPU7795 <= 2
 res_acc: 1.220703125e-04 mem_tot_CPU4261 + 1.220703125e-04 mem_tot_CPU6715
   + 1.220703125e-04 mem_tot_CPU1522 + 1.220703125e-04 mem_tot_CPU1018
   + 1.220703125e-04 mem_tot_CPU1612 + 1.220703125e-04 mem_tot_CPU9194
   + 1.220703125e-04 mem_tot_CPU4085 + 1.220703125e-04 mem_tot_CPU7795
   + 10 cores_sketch_CPU4261 + 10 cores_sketch_CPU6715
   + 10 cores_sketch_CPU1522 + 10 cores_sketch_CPU1018
   + 10 cores_sketch_CPU1612 + 10 cores_sketch_CPU9194
   + 10 cores_sketch_CPU4085 + 10 cores_sketch_CPU7795 - res = -400
 pdt_m_rows_CPU4261: pdt_m_rows_CPU4261 + [
   - rows_tot_CPU4261 * m_access_time_CPU4261 ] = 0
 pdt_m_rows_CPU6715: pdt_m_rows_CPU6715 + [
   - rows_tot_CPU6715 * m_access_time_CPU6715 ] = 0
 pdt_m_rows_CPU1522: pdt_m_rows_CPU1522 + [
   - rows_tot_CPU1522 * m_access_time_CPU1522 ] = 0
 pdt_m_rows_CPU1018: pdt_m_rows_CPU1018 + [
   - rows_tot_CPU1018 * m_access_time_CPU1018 ] = 0
 pdt_m_rows_CPU1612: pdt_m_rows_CPU1612 + [
   - rows_tot_CPU1612 * m_access_time_CPU1612 ] = 0
 pdt_m_rows_CPU9194: pdt_m_rows_CPU9194 + [
   - rows_tot_CPU9194 * m_access_time_CPU9194 ] = 0
 pdt_m_rows_CPU4085: pdt_m_rows_CPU4085 + [
   - rows_tot_CPU4085 * m_access_time_CPU4085 ] = 0
 pdt_m_rows_CPU7795: pdt_m_rows_CPU7795 + [
   - rows_tot_CPU7795 * m_access_time_CPU7795 ] = 0
Bounds
 frac[0,1527] <= 1
 frac[1,1525] <= 1
 frac[1,1526] <= 1
 frac[2,1525] <= 1
 frac[3,1527] <= 1
 frac[4,1527] <= 1
 frac[4,1524] <= 1
 frac[5,1526] <= 1
 frac[5,1527] <= 1
 frac[6,1524] <= 1
 frac[7,1524] <= 1
 frac[7,1527] <= 1
 mem_tot_CPU4261 <= 32768
 normalized_mem_tot_CPU4261 <= 1
 normalized_rows_tot_CPU4261 <= 1
 mem_tot_CPU6715 <= 32768
 normalized_mem_tot_CPU6715 <= 1
 normalized_rows_tot_CPU6715 <= 1
 mem_tot_CPU1522 <= 32768
 normalized_mem_tot_CPU1522 <= 1
 normalized_rows_tot_CPU1522 <= 1
 mem_tot_CPU1018 <= 32768
 normalized_mem_tot_CPU1018 <= 1
 normalized_rows_tot_CPU1018 <= 1
 mem_tot_CPU1612 <= 32768
 normalized_mem_tot_CPU1612 <= 1
 normalized_rows_tot_CPU1612 <= 1
 mem_tot_CPU9194 <= 32768
 normalized_mem_tot_CPU9194 <= 1
 normalized_rows_tot_CPU9194 <= 1
 mem_tot_CPU4085 <= 32768
 normalized_mem_tot_CPU4085 <= 1
 normalized_rows_tot_CPU4085 <= 1
 mem_tot_CPU7795 <= 32768
 normalized_mem_tot_CPU7795 <= 1
 normalized_rows_tot_CPU7795 <= 1
 cores_sketch_CPU4261 <= 7
 cores_sketch_CPU6715 <= 7
 cores_sketch_CPU1522 <= 7
 cores_sketch_CPU1018 <= 7
 cores_sketch_CPU1612 <= 7
 cores_sketch_CPU9194 <= 7
 cores_sketch_CPU4085 <= 7
 cores_sketch_CPU7795 <= 7
Generals
 cores_sketch_CPU4261 cores_sketch_CPU6715 cores_sketch_CPU1522
 cores_sketch_CPU1018 cores_sketch_CPU1612 cores_sketch_CPU9194
 cores_sketch_CPU4085 cores_sketch_CPU7795
General Constraints
 mem_access_time_CPU4261:
   m_access_time_CPU4261 = PWL ( mem_tot_CPU4261 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU6715:
   m_access_time_CPU6715 = PWL ( mem_tot_CPU6715 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1522:
   m_access_time_CPU1522 = PWL ( mem_tot_CPU1522 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1018:
   m_access_time_CPU1018 = PWL ( mem_tot_CPU1018 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1612:
   m_access_time_CPU1612 = PWL ( mem_tot_CPU1612 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU9194:
   m_access_time_CPU9194 = PWL ( mem_tot_CPU9194 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU4085:
   m_access_time_CPU4085 = PWL ( mem_tot_CPU4085 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU7795:
   m_access_time_CPU7795 = PWL ( mem_tot_CPU7795 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

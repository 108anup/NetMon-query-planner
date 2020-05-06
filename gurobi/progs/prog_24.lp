\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 m_access_time_CPU1 + 0 m_access_time_CPU19 + 0 m_access_time_CPU22
   + 0 m_access_time_Netro1 + 0 m_access_time_Netro14 + res
Subject To
 cov_0_41: frac[0,41] + frac[1,41] + frac[3,41] + frac[5,41] >= 0.7
 cov_1_40: frac[5,40] >= 0.9
 cov_2_53: frac[5,53] >= 0.7
 cov_3_51: frac[2,51] + frac[4,51] + frac[5,51] >= 0.8
 cov_4_38: frac[0,38] >= 0.7
 cov_5_63: frac[5,63] >= 1
 cov_6_41: frac[0,41] + frac[5,41] >= 0.8
 cov_7_59: frac[1,59] + frac[3,59] + frac[5,59] >= 0.8
 cov_8_59: frac[2,59] + frac[4,59] + frac[5,59] >= 1
 cov_9_53: frac[5,53] >= 0.8
 cov_10_63: frac[2,63] + frac[4,63] + frac[5,63] >= 0.8
 cov_11_51: frac[5,51] >= 0.8
 cov_12_34: frac[1,34] + frac[2,34] + frac[3,34] + frac[4,34] + frac[5,34]
   >= 1
 cov_13_59: frac[0,59] + frac[1,59] + frac[3,59] + frac[5,59] >= 0.9
 cov_14_45: frac[1,45] + frac[2,45] + frac[3,45] + frac[4,45] + frac[5,45]
   >= 0.9
 cov_15_41: frac[2,41] + frac[4,41] + frac[5,41] >= 1
 cov_16_60: frac[1,60] + frac[2,60] + frac[3,60] + frac[4,60] + frac[5,60]
   >= 0.9
 cov_17_37: frac[0,37] >= 0.1
 cov_18_58: frac[2,58] + frac[4,58] + frac[5,58] >= 0.9
 cov_19_37: frac[5,37] >= 1
 cov_20_54: frac[1,54] + frac[3,54] + frac[5,54] >= 1
 cov_21_34: frac[2,34] + frac[4,34] + frac[5,34] >= 0.8
 cov_22_58: frac[2,58] + frac[4,58] + frac[5,58] >= 0.8
 cov_23_32: frac[5,32] >= 0.9
 cov_24_53: frac[5,53] >= 0.8
 cov_25_40: frac[2,40] + frac[4,40] + frac[5,40] >= 0.7
 cov_26_32: frac[1,32] + frac[3,32] + frac[5,32] >= 0.8
 cov_27_56: frac[1,56] + frac[2,56] + frac[3,56] + frac[4,56] + frac[5,56]
   >= 0.8
 cov_28_38: frac[1,38] + frac[2,38] + frac[3,38] + frac[4,38] + frac[5,38]
   >= 0.8
 cov_29_95: frac[5,95] >= 0.8
 cov_30_94: frac[5,94] >= 0.7
 cov_31_94: frac[5,94] >= 1
 cov_32_65: frac[5,65] >= 1
 cov_33_95: frac[5,95] >= 0.9
 accuracy_1_34: - 5.09765625 frac[1,34] + mem[1,34] = 0
 accuracy_3_56: - 6 frac[3,56] + mem[3,56] = 0
 accuracy_2_51: - 5.09765625 frac[2,51] + mem[2,51] = 0
 accuracy_4_41: - 6 frac[4,41] + mem[4,41] = 0
 accuracy_4_51: - 6 frac[4,51] + mem[4,51] = 0
 accuracy_2_41: - 5.09765625 frac[2,41] + mem[2,41] = 0
 accuracy_5_37: - 6 frac[5,37] + mem[5,37] = 0
 accuracy_4_45: - 6 frac[4,45] + mem[4,45] = 0
 accuracy_2_45: - 5.09765625 frac[2,45] + mem[2,45] = 0
 accuracy_1_60: - 5.09765625 frac[1,60] + mem[1,60] = 0
 accuracy_2_34: - 5.09765625 frac[2,34] + mem[2,34] = 0
 accuracy_5_54: - 6 frac[5,54] + mem[5,54] = 0
 accuracy_5_32: - 6 frac[5,32] + mem[5,32] = 0
 accuracy_4_56: - 6 frac[4,56] + mem[4,56] = 0
 accuracy_3_38: - 6 frac[3,38] + mem[3,38] = 0
 accuracy_2_59: - 5.09765625 frac[2,59] + mem[2,59] = 0
 accuracy_3_59: - 6 frac[3,59] + mem[3,59] = 0
 accuracy_3_34: - 6 frac[3,34] + mem[3,34] = 0
 accuracy_4_60: - 6 frac[4,60] + mem[4,60] = 0
 accuracy_1_56: - 5.09765625 frac[1,56] + mem[1,56] = 0
 accuracy_2_38: - 5.09765625 frac[2,38] + mem[2,38] = 0
 accuracy_3_45: - 6 frac[3,45] + mem[3,45] = 0
 accuracy_2_63: - 5.09765625 frac[2,63] + mem[2,63] = 0
 accuracy_1_45: - 5.09765625 frac[1,45] + mem[1,45] = 0
 accuracy_5_56: - 6 frac[5,56] + mem[5,56] = 0
 accuracy_3_41: - 6 frac[3,41] + mem[3,41] = 0
 accuracy_0_37: - 5.09765625 frac[0,37] + mem[0,37] = 0
 accuracy_1_41: - 5.09765625 frac[1,41] + mem[1,41] = 0
 accuracy_5_53: - 6 frac[5,53] + mem[5,53] = 0
 accuracy_4_34: - 6 frac[4,34] + mem[4,34] = 0
 accuracy_5_60: - 6 frac[5,60] + mem[5,60] = 0
 accuracy_1_32: - 5.09765625 frac[1,32] + mem[1,32] = 0
 accuracy_2_56: - 5.09765625 frac[2,56] + mem[2,56] = 0
 accuracy_5_95: - 6 frac[5,95] + mem[5,95] = 0
 accuracy_1_54: - 5.09765625 frac[1,54] + mem[1,54] = 0
 accuracy_2_60: - 5.09765625 frac[2,60] + mem[2,60] = 0
 accuracy_3_54: - 6 frac[3,54] + mem[3,54] = 0
 accuracy_4_38: - 6 frac[4,38] + mem[4,38] = 0
 accuracy_1_59: - 5.09765625 frac[1,59] + mem[1,59] = 0
 accuracy_0_38: - 5.09765625 frac[0,38] + mem[0,38] = 0
 accuracy_4_58: - 6 frac[4,58] + mem[4,58] = 0
 accuracy_5_59: - 6 frac[5,59] + mem[5,59] = 0
 accuracy_0_41: - 5.09765625 frac[0,41] + mem[0,41] = 0
 accuracy_5_65: - 6 frac[5,65] + mem[5,65] = 0
 accuracy_5_34: - 6 frac[5,34] + mem[5,34] = 0
 accuracy_5_94: - 6 frac[5,94] + mem[5,94] = 0
 accuracy_5_41: - 6 frac[5,41] + mem[5,41] = 0
 accuracy_0_59: - 5.09765625 frac[0,59] + mem[0,59] = 0
 accuracy_5_63: - 6 frac[5,63] + mem[5,63] = 0
 accuracy_4_40: - 6 frac[4,40] + mem[4,40] = 0
 accuracy_3_32: - 6 frac[3,32] + mem[3,32] = 0
 accuracy_2_40: - 5.09765625 frac[2,40] + mem[2,40] = 0
 accuracy_5_38: - 6 frac[5,38] + mem[5,38] = 0
 accuracy_5_45: - 6 frac[5,45] + mem[5,45] = 0
 accuracy_4_59: - 6 frac[4,59] + mem[4,59] = 0
 accuracy_5_51: - 6 frac[5,51] + mem[5,51] = 0
 accuracy_3_60: - 6 frac[3,60] + mem[3,60] = 0
 accuracy_5_58: - 6 frac[5,58] + mem[5,58] = 0
 accuracy_1_38: - 5.09765625 frac[1,38] + mem[1,38] = 0
 accuracy_4_63: - 6 frac[4,63] + mem[4,63] = 0
 accuracy_5_40: - 6 frac[5,40] + mem[5,40] = 0
 accuracy_2_58: - 5.09765625 frac[2,58] + mem[2,58] = 0
 mem_tot_CPU1: - mem[0,37] - mem[0,38] - mem[0,41] - mem[0,59]
   + mem_tot_CPU1 = 0
 R97: - 3.05176e-05 mem_tot_CPU1 + normalized_mem_tot_CPU1 = 0
 rows_tot_CPU1: - 3 frac[0,37] - 3 frac[0,38] - 3 frac[0,41] - 3 frac[0,59]
   + rows_tot_CPU1 = 0
 R99: - 0.0833333333333333 rows_tot_CPU1 + normalized_rows_tot_CPU1 = 0
 mem_tot_CPU19: - mem[1,34] - mem[1,60] - mem[1,56] - mem[1,45] - mem[1,41]
   - mem[1,32] - mem[1,54] - mem[1,59] - mem[1,38] + mem_tot_CPU19 = 0
 R101: - 3.05176e-05 mem_tot_CPU19 + normalized_mem_tot_CPU19 = 0
 rows_tot_CPU19: - 3 frac[1,34] - 3 frac[1,60] - 3 frac[1,56]
   - 3 frac[1,45] - 3 frac[1,41] - 3 frac[1,32] - 3 frac[1,54]
   - 3 frac[1,59] - 3 frac[1,38] + rows_tot_CPU19 = 0
 R103: - 0.0833333333333333 rows_tot_CPU19 + normalized_rows_tot_CPU19 = 0
 mem_tot_CPU22: - mem[2,51] - mem[2,41] - mem[2,45] - mem[2,34] - mem[2,59]
   - mem[2,38] - mem[2,63] - mem[2,56] - mem[2,60] - mem[2,40] - mem[2,58]
   + mem_tot_CPU22 = 0
 R105: - 3.05176e-05 mem_tot_CPU22 + normalized_mem_tot_CPU22 = 0
 rows_tot_CPU22: - 3 frac[2,51] - 3 frac[2,41] - 3 frac[2,45]
   - 3 frac[2,34] - 3 frac[2,59] - 3 frac[2,38] - 3 frac[2,63]
   - 3 frac[2,56] - 3 frac[2,60] - 3 frac[2,40] - 3 frac[2,58]
   + rows_tot_CPU22 = 0
 R107: - 0.0833333333333333 rows_tot_CPU22 + normalized_rows_tot_CPU22 = 0
 mem_tot_Netro1: - mem[3,56] - mem[3,38] - mem[3,59] - mem[3,34]
   - mem[3,45] - mem[3,41] - mem[3,54] - mem[3,32] - mem[3,60]
   + mem_tot_Netro1 = 0
 R109: - 5e-06 mem_tot_Netro1 + normalized_mem_tot_Netro1 = 0
 rows_tot_Netro1: - 3 frac[3,56] - 3 frac[3,38] - 3 frac[3,59]
   - 3 frac[3,34] - 3 frac[3,45] - 3 frac[3,41] - 3 frac[3,54]
   - 3 frac[3,32] - 3 frac[3,60] + rows_tot_Netro1 = 0
 R111: - 0.0833333333333333 rows_tot_Netro1 + normalized_rows_tot_Netro1
   = 0
 mem_tot_Netro14: - mem[4,41] - mem[4,51] - mem[4,45] - mem[4,56]
   - mem[4,60] - mem[4,34] - mem[4,38] - mem[4,58] - mem[4,40] - mem[4,59]
   - mem[4,63] + mem_tot_Netro14 = 0
 R113: - 5e-06 mem_tot_Netro14 + normalized_mem_tot_Netro14 = 0
 rows_tot_Netro14: - 3 frac[4,41] - 3 frac[4,51] - 3 frac[4,45]
   - 3 frac[4,56] - 3 frac[4,60] - 3 frac[4,34] - 3 frac[4,38]
   - 3 frac[4,58] - 3 frac[4,40] - 3 frac[4,59] - 3 frac[4,63]
   + rows_tot_Netro14 = 0
 R115: - 0.0833333333333333 rows_tot_Netro14 + normalized_rows_tot_Netro14
   = 0
 mem_tot_tor_P43: - mem[5,37] - mem[5,54] - mem[5,32] - mem[5,56]
   - mem[5,53] - mem[5,60] - mem[5,95] - mem[5,59] - mem[5,65] - mem[5,34]
   - mem[5,94] - mem[5,41] - mem[5,63] - mem[5,38] - mem[5,45] - mem[5,51]
   - mem[5,58] - mem[5,40] + mem_tot_tor_P43 = 0
 R117: - 1.6276041666666666e-04 mem_tot_tor_P43
   + normalized_mem_tot_tor_P43 = 0
 rows_tot_tor_P43: - 3 frac[5,37] - 3 frac[5,54] - 3 frac[5,32]
   - 3 frac[5,56] - 3 frac[5,53] - 3 frac[5,60] - 3 frac[5,95]
   - 3 frac[5,59] - 3 frac[5,65] - 3 frac[5,34] - 3 frac[5,94]
   - 3 frac[5,41] - 3 frac[5,63] - 3 frac[5,38] - 3 frac[5,45]
   - 3 frac[5,51] - 3 frac[5,58] - 3 frac[5,40] + rows_tot_tor_P43 = 0
 R119: - 0.0208333333333333 rows_tot_tor_P43 + normalized_rows_tot_tor_P43
   = 0
 row_capacity_CPU1: rows_tot_CPU1 <= 12
 row_capacity_CPU19: rows_tot_CPU19 <= 12
 row_capacity_CPU22: rows_tot_CPU22 <= 12
 row_capacity_Netro1: rows_tot_Netro1 <= 12
 row_capacity_Netro14: rows_tot_Netro14 <= 12
 row_capacity_tor_P43: rows_tot_tor_P43 <= 48
 capacity_mem_par_tor_P43: mem[5,37] <= 1536
 capacity_mem_par_tor_P43: mem[5,54] <= 1536
 capacity_mem_par_tor_P43: mem[5,32] <= 1536
 capacity_mem_par_tor_P43: mem[5,56] <= 1536
 capacity_mem_par_tor_P43: mem[5,53] <= 1536
 capacity_mem_par_tor_P43: mem[5,60] <= 1536
 capacity_mem_par_tor_P43: mem[5,95] <= 1536
 capacity_mem_par_tor_P43: mem[5,59] <= 1536
 capacity_mem_par_tor_P43: mem[5,65] <= 1536
 capacity_mem_par_tor_P43: mem[5,34] <= 1536
 capacity_mem_par_tor_P43: mem[5,94] <= 1536
 capacity_mem_par_tor_P43: mem[5,41] <= 1536
 capacity_mem_par_tor_P43: mem[5,63] <= 1536
 capacity_mem_par_tor_P43: mem[5,38] <= 1536
 capacity_mem_par_tor_P43: mem[5,45] <= 1536
 capacity_mem_par_tor_P43: mem[5,51] <= 1536
 capacity_mem_par_tor_P43: mem[5,58] <= 1536
 capacity_mem_par_tor_P43: mem[5,40] <= 1536
 ns_single_CPU1: 3.5 rows_tot_CPU1 - ns_single_CPU1 + pdt_m_rows_CPU1 = 0
 ns_sketch_CPU1: - ns_single_CPU1 + 36.29764065335753 cores_sketch_CPU1
   >= 0
 capacity_cores_CPU1: cores_sketch_CPU1 <= 6
 ns_single_CPU19: 3.5 rows_tot_CPU19 - ns_single_CPU19 + pdt_m_rows_CPU19
   = 0
 ns_sketch_CPU19: - ns_single_CPU19 + 36.29764065335753 cores_sketch_CPU19
   >= 0
 capacity_cores_CPU19: cores_sketch_CPU19 <= 6
 ns_single_CPU22: 3.5 rows_tot_CPU22 - ns_single_CPU22 + pdt_m_rows_CPU22
   = 0
 ns_sketch_CPU22: - ns_single_CPU22 + 36.29764065335753 cores_sketch_CPU22
   >= 0
 capacity_cores_CPU22: cores_sketch_CPU22 <= 6
 ns_mem_max_Netro1: - ns_mem_max_Netro1 + pdt_m_rows_Netro1
   = -13.40036284744041
 ns_hash_max_Netro1: - 1.01427115498035 rows_tot_Netro1
   + ns_hash_max_Netro1 = 33.22322438313901
 ns_req_hash_Netro1: - 54 ns_hash_max_Netro1
   + 36.29764065335753 micro_engines_Netro1 >= 0
 ns_req_mem_Netro1: ns_mem_max_Netro1 <= 36.29764065335753
 ns_req_fwd_Netro1: 36.29764065335753 micro_engines_Netro1
   >= 1960.072595281307
 ns_mem_max_Netro14: - ns_mem_max_Netro14 + pdt_m_rows_Netro14
   = -13.40036284744041
 ns_hash_max_Netro14: - 1.01427115498035 rows_tot_Netro14
   + ns_hash_max_Netro14 = 33.22322438313901
 ns_req_hash_Netro14: - 54 ns_hash_max_Netro14
   + 36.29764065335753 micro_engines_Netro14 >= 0
 ns_req_mem_Netro14: ns_mem_max_Netro14 <= 36.29764065335753
 ns_req_fwd_Netro14: 36.29764065335753 micro_engines_Netro14
   >= 1960.072595281307
 res_acc: 1.220703125e-04 mem_tot_CPU1 + 1.220703125e-04 mem_tot_CPU19
   + 1.220703125e-04 mem_tot_CPU22 + 3.2552083333333332e-04 mem_tot_Netro1
   + 3.2552083333333332e-04 mem_tot_Netro14 + 0.001953125 mem_tot_tor_P43
   + 0.25 rows_tot_tor_P43 + 10 cores_sketch_CPU1 + 10 cores_sketch_CPU19
   + 10 cores_sketch_CPU22 + 5 micro_engines_Netro1
   + 5 micro_engines_Netro14 - res = -30
 pdt_m_rows_CPU1: pdt_m_rows_CPU1 + [ - rows_tot_CPU1 * m_access_time_CPU1
   ] = 0
 pdt_m_rows_CPU19: pdt_m_rows_CPU19 + [
   - rows_tot_CPU19 * m_access_time_CPU19 ] = 0
 pdt_m_rows_CPU22: pdt_m_rows_CPU22 + [
   - rows_tot_CPU22 * m_access_time_CPU22 ] = 0
 pdt_m_rows_Netro1: pdt_m_rows_Netro1 + [
   - rows_tot_Netro1 * m_access_time_Netro1 ] = 0
 pdt_m_rows_Netro14: pdt_m_rows_Netro14 + [
   - rows_tot_Netro14 * m_access_time_Netro14 ] = 0
Bounds
 frac[0,37] <= 1
 frac[0,38] <= 1
 frac[0,41] <= 1
 frac[0,59] <= 1
 frac[1,34] <= 1
 frac[1,60] <= 1
 frac[1,56] <= 1
 frac[1,45] <= 1
 frac[1,41] <= 1
 frac[1,32] <= 1
 frac[1,54] <= 1
 frac[1,59] <= 1
 frac[1,38] <= 1
 frac[2,51] <= 1
 frac[2,41] <= 1
 frac[2,45] <= 1
 frac[2,34] <= 1
 frac[2,59] <= 1
 frac[2,38] <= 1
 frac[2,63] <= 1
 frac[2,56] <= 1
 frac[2,60] <= 1
 frac[2,40] <= 1
 frac[2,58] <= 1
 mem_tot_CPU1 <= 32768
 normalized_mem_tot_CPU1 <= 1
 normalized_rows_tot_CPU1 <= 1
 mem_tot_CPU19 <= 32768
 normalized_mem_tot_CPU19 <= 1
 normalized_rows_tot_CPU19 <= 1
 mem_tot_CPU22 <= 32768
 normalized_mem_tot_CPU22 <= 1
 normalized_rows_tot_CPU22 <= 1
 mem_tot_Netro1 <= 200000
 normalized_mem_tot_Netro1 <= 1
 normalized_rows_tot_Netro1 <= 1
 mem_tot_Netro14 <= 200000
 normalized_mem_tot_Netro14 <= 1
 normalized_rows_tot_Netro14 <= 1
 mem_tot_tor_P43 <= 6144
 normalized_mem_tot_tor_P43 <= 1
 normalized_rows_tot_tor_P43 <= 1
 cores_sketch_CPU1 <= 7
 cores_sketch_CPU19 <= 7
 cores_sketch_CPU22 <= 7
 micro_engines_Netro1 <= 54
 micro_engines_Netro14 <= 54
Binaries
 frac[3,56] frac[3,38] frac[3,59] frac[3,34] frac[3,45] frac[3,41]
 frac[3,54] frac[3,32] frac[3,60] frac[4,41] frac[4,51] frac[4,45]
 frac[4,56] frac[4,60] frac[4,34] frac[4,38] frac[4,58] frac[4,40]
 frac[4,59] frac[4,63] frac[5,37] frac[5,54] frac[5,32] frac[5,56]
 frac[5,53] frac[5,60] frac[5,95] frac[5,59] frac[5,65] frac[5,34]
 frac[5,94] frac[5,41] frac[5,63] frac[5,38] frac[5,45] frac[5,51]
 frac[5,58] frac[5,40]
Generals
 cores_sketch_CPU1 cores_sketch_CPU19 cores_sketch_CPU22
 micro_engines_Netro1 micro_engines_Netro14
General Constraints
 mem_access_time_CPU1: m_access_time_CPU1 = PWL ( mem_tot_CPU1 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU19: m_access_time_CPU19 = PWL ( mem_tot_CPU19 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU22: m_access_time_CPU22 = PWL ( mem_tot_CPU22 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_Netro1: m_access_time_Netro1 = PWL ( mem_tot_Netro1 ) : 
  (0, 2.437806463533801) (0.0390625, 2.437806463533801) 
  (0.078125, 3.001397701320998) (0.625, 3.001397701320998) 
  (1.25, 3.419718427960497) (1280, 3.419718427960497) 
  (5120, 5.797905875083051) (10240, 8.557025366725194) 
  (20480, 10.33170622821247) (40960, 11.79222979807959) 
  (163840, 13.76708909669199) (200000, 14.34823258821074)
 mem_access_time_Netro14:
   m_access_time_Netro14 = PWL ( mem_tot_Netro14 ) : 
  (0, 2.437806463533801) (0.0390625, 2.437806463533801) 
  (0.078125, 3.001397701320998) (0.625, 3.001397701320998) 
  (1.25, 3.419718427960497) (1280, 3.419718427960497) 
  (5120, 5.797905875083051) (10240, 8.557025366725194) 
  (20480, 10.33170622821247) (40960, 11.79222979807959) 
  (163840, 13.76708909669199) (200000, 14.34823258821074)
End

\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_others + 0 max_rows_others + 0 m_access_time_CPU609
   + 0 m_access_time_CPU568 + 0 m_access_time_CPU687
   + 0 m_access_time_CPU445 + 0 m_access_time_CPU157
   + 0 m_access_time_CPU151 + 0 m_access_time_CPU743
   + 0 m_access_time_CPU507 + 0 m_access_time_CPU670
   + 0 m_access_time_CPU33 + 0 m_access_time_CPU489
   + 0 m_access_time_CPU455 + 0 m_access_time_CPU379
   + 0 m_access_time_CPU488 + 0 m_access_time_CPU420
   + 0 m_access_time_CPU433 + res
Subject To
 cov_0_6: frac[2,6] + frac[6,6] >= 0.7
 cov_1_0: frac[0,0] + frac[6,0] >= 0.9
 cov_2_4: frac[3,4] + frac[6,4] >= 0.9
 cov_3_14: frac[1,14] + frac[4,14] >= 0.5961910383419688
 cov_4_14: frac[1,14] + frac[5,14] >= 0.5961910383419688
 cov_5_7: frac[1,7] + frac[6,7] >= 0.5961910383419688
 cov_6_4: frac[4,4] + frac[7,4] >= 0.8
 cov_7_12: frac[0,12] + frac[1,12] >= 0.9
 cov_8_7: frac[2,7] + frac[4,7] >= 0.5961910383419688
 cov_9_7: frac[6,7] + frac[7,7] >= 0.5961910383419688
 cov_10_4: frac[4,4] + frac[5,4] >= 0.9
 cov_11_0: frac[2,0] + frac[7,0] >= 0.8
 cov_12_2: frac[4,2] + frac[6,2] >= 0.7
 cov_13_4: frac[1,4] + frac[2,4] >= 0.9
 cov_14_9: frac[2,9] + frac[5,9] >= 0.7
 cov_15_14: frac[5,14] + frac[6,14] >= 0.5961910383419688
 cov_16_8: frac[11,8] + frac[14,8] >= 0.7
 cov_17_10: frac[12,10] + frac[13,10] >= 0.5961910383419688
 cov_18_3: frac[8,3] + frac[14,3] >= 0.5961910383419688
 cov_19_15: frac[11,15] + frac[15,15] >= 0.7
 cov_20_13: frac[12,13] + frac[15,13] >= 0.9
 cov_21_1: frac[9,1] + frac[11,1] >= 0.7
 cov_22_5: frac[12,5] + frac[15,5] >= 0.9
 cov_23_11: frac[13,11] + frac[14,11] >= 0.5961910383419688
 cov_24_13: frac[9,13] + frac[12,13] >= 0.8
 cov_25_5: frac[12,5] + frac[14,5] >= 0.8
 cov_26_15: frac[13,15] + frac[15,15] >= 0.8
 cov_27_11: frac[10,11] + frac[12,11] >= 0.5961910383419688
 cov_28_15: frac[10,15] + frac[13,15] >= 0.8
 cov_29_11: frac[11,11] + frac[12,11] >= 0.5961910383419688
 cov_30_5: frac[8,5] + frac[15,5] >= 0.8
 cov_31_11: frac[8,11] + frac[12,11] >= 0.5961910383419688
 accuracy_5_9: - 50.9765625 frac[5,9] + mem[5,9] = 0
 accuracy_4_7: - 50.9765625 frac[4,7] + mem[4,7] = 0
 accuracy_15_13: - 50.9765625 frac[15,13] + mem[15,13] = 0
 accuracy_10_11: - 50.9765625 frac[10,11] + mem[10,11] = 0
 accuracy_9_1: - 50.9765625 frac[9,1] + mem[9,1] = 0
 accuracy_11_11: - 50.9765625 frac[11,11] + mem[11,11] = 0
 accuracy_6_6: - 50.9765625 frac[6,6] + mem[6,6] = 0
 accuracy_7_7: - 50.9765625 frac[7,7] + mem[7,7] = 0
 accuracy_12_5: - 50.9765625 frac[12,5] + mem[12,5] = 0
 accuracy_10_15: - 50.9765625 frac[10,15] + mem[10,15] = 0
 accuracy_6_2: - 50.9765625 frac[6,2] + mem[6,2] = 0
 accuracy_15_5: - 50.9765625 frac[15,5] + mem[15,5] = 0
 accuracy_8_5: - 50.9765625 frac[8,5] + mem[8,5] = 0
 accuracy_6_14: - 50.9765625 frac[6,14] + mem[6,14] = 0
 accuracy_6_7: - 50.9765625 frac[6,7] + mem[6,7] = 0
 accuracy_2_9: - 50.9765625 frac[2,9] + mem[2,9] = 0
 accuracy_14_5: - 50.9765625 frac[14,5] + mem[14,5] = 0
 accuracy_12_13: - 50.9765625 frac[12,13] + mem[12,13] = 0
 accuracy_4_4: - 50.9765625 frac[4,4] + mem[4,4] = 0
 accuracy_1_14: - 50.9765625 frac[1,14] + mem[1,14] = 0
 accuracy_11_1: - 50.9765625 frac[11,1] + mem[11,1] = 0
 accuracy_12_10: - 50.9765625 frac[12,10] + mem[12,10] = 0
 accuracy_15_15: - 50.9765625 frac[15,15] + mem[15,15] = 0
 accuracy_13_11: - 50.9765625 frac[13,11] + mem[13,11] = 0
 accuracy_6_4: - 50.9765625 frac[6,4] + mem[6,4] = 0
 accuracy_5_4: - 50.9765625 frac[5,4] + mem[5,4] = 0
 accuracy_2_6: - 50.9765625 frac[2,6] + mem[2,6] = 0
 accuracy_0_0: - 50.9765625 frac[0,0] + mem[0,0] = 0
 accuracy_8_11: - 50.9765625 frac[8,11] + mem[8,11] = 0
 accuracy_13_15: - 50.9765625 frac[13,15] + mem[13,15] = 0
 accuracy_6_0: - 50.9765625 frac[6,0] + mem[6,0] = 0
 accuracy_4_14: - 50.9765625 frac[4,14] + mem[4,14] = 0
 accuracy_0_12: - 50.9765625 frac[0,12] + mem[0,12] = 0
 accuracy_1_4: - 50.9765625 frac[1,4] + mem[1,4] = 0
 accuracy_12_11: - 50.9765625 frac[12,11] + mem[12,11] = 0
 accuracy_4_2: - 50.9765625 frac[4,2] + mem[4,2] = 0
 accuracy_13_10: - 50.9765625 frac[13,10] + mem[13,10] = 0
 accuracy_2_7: - 50.9765625 frac[2,7] + mem[2,7] = 0
 accuracy_14_11: - 50.9765625 frac[14,11] + mem[14,11] = 0
 accuracy_8_3: - 50.9765625 frac[8,3] + mem[8,3] = 0
 accuracy_9_13: - 50.9765625 frac[9,13] + mem[9,13] = 0
 accuracy_7_0: - 50.9765625 frac[7,0] + mem[7,0] = 0
 accuracy_1_12: - 50.9765625 frac[1,12] + mem[1,12] = 0
 accuracy_11_8: - 50.9765625 frac[11,8] + mem[11,8] = 0
 accuracy_5_14: - 50.9765625 frac[5,14] + mem[5,14] = 0
 accuracy_7_4: - 50.9765625 frac[7,4] + mem[7,4] = 0
 accuracy_2_0: - 50.9765625 frac[2,0] + mem[2,0] = 0
 accuracy_14_8: - 50.9765625 frac[14,8] + mem[14,8] = 0
 accuracy_1_7: - 50.9765625 frac[1,7] + mem[1,7] = 0
 accuracy_14_3: - 50.9765625 frac[14,3] + mem[14,3] = 0
 accuracy_11_15: - 50.9765625 frac[11,15] + mem[11,15] = 0
 accuracy_3_4: - 50.9765625 frac[3,4] + mem[3,4] = 0
 accuracy_2_4: - 50.9765625 frac[2,4] + mem[2,4] = 0
 mem_tot_CPU609: - mem[0,0] - mem[0,12] + mem_tot_CPU609 = 0
 R86: - 3.05176e-05 mem_tot_CPU609 + normalized_mem_tot_CPU609 = 0
 rows_tot_CPU609: - 3 frac[0,0] - 3 frac[0,12] + rows_tot_CPU609 = 0
 R88: - 0.0833333333333333 rows_tot_CPU609 + normalized_rows_tot_CPU609
   = 0
 mem_tot_CPU568: - mem[1,14] - mem[1,4] - mem[1,12] - mem[1,7]
   + mem_tot_CPU568 = 0
 R90: - 3.05176e-05 mem_tot_CPU568 + normalized_mem_tot_CPU568 = 0
 rows_tot_CPU568: - 3 frac[1,14] - 3 frac[1,4] - 3 frac[1,12] - 3 frac[1,7]
   + rows_tot_CPU568 = 0
 R92: - 0.0833333333333333 rows_tot_CPU568 + normalized_rows_tot_CPU568
   = 0
 mem_tot_CPU687: - mem[2,9] - mem[2,6] - mem[2,7] - mem[2,0] - mem[2,4]
   + mem_tot_CPU687 = 0
 R94: - 3.05176e-05 mem_tot_CPU687 + normalized_mem_tot_CPU687 = 0
 rows_tot_CPU687: - 3 frac[2,9] - 3 frac[2,6] - 3 frac[2,7] - 3 frac[2,0]
   - 3 frac[2,4] + rows_tot_CPU687 = 0
 R96: - 0.0833333333333333 rows_tot_CPU687 + normalized_rows_tot_CPU687
   = 0
 mem_tot_CPU445: - mem[3,4] + mem_tot_CPU445 = 0
 R98: - 3.05176e-05 mem_tot_CPU445 + normalized_mem_tot_CPU445 = 0
 rows_tot_CPU445: - 3 frac[3,4] + rows_tot_CPU445 = 0
 R100: - 0.0833333333333333 rows_tot_CPU445 + normalized_rows_tot_CPU445
   = 0
 mem_tot_CPU157: - mem[4,7] - mem[4,4] - mem[4,14] - mem[4,2]
   + mem_tot_CPU157 = 0
 R102: - 3.05176e-05 mem_tot_CPU157 + normalized_mem_tot_CPU157 = 0
 rows_tot_CPU157: - 3 frac[4,7] - 3 frac[4,4] - 3 frac[4,14] - 3 frac[4,2]
   + rows_tot_CPU157 = 0
 R104: - 0.0833333333333333 rows_tot_CPU157 + normalized_rows_tot_CPU157
   = 0
 mem_tot_CPU151: - mem[5,9] - mem[5,4] - mem[5,14] + mem_tot_CPU151 = 0
 R106: - 3.05176e-05 mem_tot_CPU151 + normalized_mem_tot_CPU151 = 0
 rows_tot_CPU151: - 3 frac[5,9] - 3 frac[5,4] - 3 frac[5,14]
   + rows_tot_CPU151 = 0
 R108: - 0.0833333333333333 rows_tot_CPU151 + normalized_rows_tot_CPU151
   = 0
 mem_tot_CPU743: - mem[6,6] - mem[6,2] - mem[6,14] - mem[6,7] - mem[6,4]
   - mem[6,0] + mem_tot_CPU743 = 0
 R110: - 3.05176e-05 mem_tot_CPU743 + normalized_mem_tot_CPU743 = 0
 rows_tot_CPU743: - 3 frac[6,6] - 3 frac[6,2] - 3 frac[6,14] - 3 frac[6,7]
   - 3 frac[6,4] - 3 frac[6,0] + rows_tot_CPU743 = 0
 R112: - 0.0833333333333333 rows_tot_CPU743 + normalized_rows_tot_CPU743
   = 0
 mem_tot_CPU507: - mem[7,7] - mem[7,0] - mem[7,4] + mem_tot_CPU507 = 0
 R114: - 3.05176e-05 mem_tot_CPU507 + normalized_mem_tot_CPU507 = 0
 rows_tot_CPU507: - 3 frac[7,7] - 3 frac[7,0] - 3 frac[7,4]
   + rows_tot_CPU507 = 0
 R116: - 0.0833333333333333 rows_tot_CPU507 + normalized_rows_tot_CPU507
   = 0
 mem_tot_CPU670: - mem[8,5] - mem[8,11] - mem[8,3] + mem_tot_CPU670 = 0
 R118: - 3.05176e-05 mem_tot_CPU670 + normalized_mem_tot_CPU670 = 0
 rows_tot_CPU670: - 3 frac[8,5] - 3 frac[8,11] - 3 frac[8,3]
   + rows_tot_CPU670 = 0
 R120: - 0.0833333333333333 rows_tot_CPU670 + normalized_rows_tot_CPU670
   = 0
 mem_tot_CPU33: - mem[9,1] - mem[9,13] + mem_tot_CPU33 = 0
 R122: - 3.05176e-05 mem_tot_CPU33 + normalized_mem_tot_CPU33 = 0
 rows_tot_CPU33: - 3 frac[9,1] - 3 frac[9,13] + rows_tot_CPU33 = 0
 R124: - 0.0833333333333333 rows_tot_CPU33 + normalized_rows_tot_CPU33 = 0
 mem_tot_CPU489: - mem[10,11] - mem[10,15] + mem_tot_CPU489 = 0
 R126: - 3.05176e-05 mem_tot_CPU489 + normalized_mem_tot_CPU489 = 0
 rows_tot_CPU489: - 3 frac[10,11] - 3 frac[10,15] + rows_tot_CPU489 = 0
 R128: - 0.0833333333333333 rows_tot_CPU489 + normalized_rows_tot_CPU489
   = 0
 mem_tot_CPU455: - mem[11,11] - mem[11,1] - mem[11,8] - mem[11,15]
   + mem_tot_CPU455 = 0
 R130: - 3.05176e-05 mem_tot_CPU455 + normalized_mem_tot_CPU455 = 0
 rows_tot_CPU455: - 3 frac[11,11] - 3 frac[11,1] - 3 frac[11,8]
   - 3 frac[11,15] + rows_tot_CPU455 = 0
 R132: - 0.0833333333333333 rows_tot_CPU455 + normalized_rows_tot_CPU455
   = 0
 mem_tot_CPU379: - mem[12,5] - mem[12,13] - mem[12,10] - mem[12,11]
   + mem_tot_CPU379 = 0
 R134: - 3.05176e-05 mem_tot_CPU379 + normalized_mem_tot_CPU379 = 0
 rows_tot_CPU379: - 3 frac[12,5] - 3 frac[12,13] - 3 frac[12,10]
   - 3 frac[12,11] + rows_tot_CPU379 = 0
 R136: - 0.0833333333333333 rows_tot_CPU379 + normalized_rows_tot_CPU379
   = 0
 mem_tot_CPU488: - mem[13,11] - mem[13,15] - mem[13,10] + mem_tot_CPU488
   = 0
 R138: - 3.05176e-05 mem_tot_CPU488 + normalized_mem_tot_CPU488 = 0
 rows_tot_CPU488: - 3 frac[13,11] - 3 frac[13,15] - 3 frac[13,10]
   + rows_tot_CPU488 = 0
 R140: - 0.0833333333333333 rows_tot_CPU488 + normalized_rows_tot_CPU488
   = 0
 mem_tot_CPU420: - mem[14,5] - mem[14,11] - mem[14,8] - mem[14,3]
   + mem_tot_CPU420 = 0
 R142: - 3.05176e-05 mem_tot_CPU420 + normalized_mem_tot_CPU420 = 0
 rows_tot_CPU420: - 3 frac[14,5] - 3 frac[14,11] - 3 frac[14,8]
   - 3 frac[14,3] + rows_tot_CPU420 = 0
 R144: - 0.0833333333333333 rows_tot_CPU420 + normalized_rows_tot_CPU420
   = 0
 mem_tot_CPU433: - mem[15,13] - mem[15,5] - mem[15,15] + mem_tot_CPU433
   = 0
 R146: - 3.05176e-05 mem_tot_CPU433 + normalized_mem_tot_CPU433 = 0
 rows_tot_CPU433: - 3 frac[15,13] - 3 frac[15,5] - 3 frac[15,15]
   + rows_tot_CPU433 = 0
 R148: - 0.0833333333333333 rows_tot_CPU433 + normalized_rows_tot_CPU433
   = 0
 row_capacity_CPU609: rows_tot_CPU609 <= 12
 row_capacity_CPU568: rows_tot_CPU568 <= 12
 row_capacity_CPU687: rows_tot_CPU687 <= 12
 row_capacity_CPU445: rows_tot_CPU445 <= 12
 row_capacity_CPU157: rows_tot_CPU157 <= 12
 row_capacity_CPU151: rows_tot_CPU151 <= 12
 row_capacity_CPU743: rows_tot_CPU743 <= 12
 row_capacity_CPU507: rows_tot_CPU507 <= 12
 row_capacity_CPU670: rows_tot_CPU670 <= 12
 row_capacity_CPU33: rows_tot_CPU33 <= 12
 row_capacity_CPU489: rows_tot_CPU489 <= 12
 row_capacity_CPU455: rows_tot_CPU455 <= 12
 row_capacity_CPU379: rows_tot_CPU379 <= 12
 row_capacity_CPU488: rows_tot_CPU488 <= 12
 row_capacity_CPU420: rows_tot_CPU420 <= 12
 row_capacity_CPU433: rows_tot_CPU433 <= 12
 ns_single_CPU609: 3.5 rows_tot_CPU609 - ns_single_CPU609
   + pdt_m_rows_CPU609 = 0
 ns_sketch_CPU609: - ns_single_CPU609
   + 11.42863142857143 cores_sketch_CPU609 >= 0
 capacity_cores_CPU609: cores_sketch_CPU609 <= 2
 ns_single_CPU568: 3.5 rows_tot_CPU568 - ns_single_CPU568
   + pdt_m_rows_CPU568 = 0
 ns_sketch_CPU568: - ns_single_CPU568
   + 11.42863142857143 cores_sketch_CPU568 >= 0
 capacity_cores_CPU568: cores_sketch_CPU568 <= 2
 ns_single_CPU687: 3.5 rows_tot_CPU687 - ns_single_CPU687
   + pdt_m_rows_CPU687 = 0
 ns_sketch_CPU687: - ns_single_CPU687
   + 11.42863142857143 cores_sketch_CPU687 >= 0
 capacity_cores_CPU687: cores_sketch_CPU687 <= 2
 ns_single_CPU445: 3.5 rows_tot_CPU445 - ns_single_CPU445
   + pdt_m_rows_CPU445 = 0
 ns_sketch_CPU445: - ns_single_CPU445
   + 11.42863142857143 cores_sketch_CPU445 >= 0
 capacity_cores_CPU445: cores_sketch_CPU445 <= 2
 ns_single_CPU157: 3.5 rows_tot_CPU157 - ns_single_CPU157
   + pdt_m_rows_CPU157 = 0
 ns_sketch_CPU157: - ns_single_CPU157
   + 11.42863142857143 cores_sketch_CPU157 >= 0
 capacity_cores_CPU157: cores_sketch_CPU157 <= 2
 ns_single_CPU151: 3.5 rows_tot_CPU151 - ns_single_CPU151
   + pdt_m_rows_CPU151 = 0
 ns_sketch_CPU151: - ns_single_CPU151
   + 11.42863142857143 cores_sketch_CPU151 >= 0
 capacity_cores_CPU151: cores_sketch_CPU151 <= 2
 ns_single_CPU743: 3.5 rows_tot_CPU743 - ns_single_CPU743
   + pdt_m_rows_CPU743 = 0
 ns_sketch_CPU743: - ns_single_CPU743
   + 11.42863142857143 cores_sketch_CPU743 >= 0
 capacity_cores_CPU743: cores_sketch_CPU743 <= 2
 ns_single_CPU507: 3.5 rows_tot_CPU507 - ns_single_CPU507
   + pdt_m_rows_CPU507 = 0
 ns_sketch_CPU507: - ns_single_CPU507
   + 11.42863142857143 cores_sketch_CPU507 >= 0
 capacity_cores_CPU507: cores_sketch_CPU507 <= 2
 ns_single_CPU670: 3.5 rows_tot_CPU670 - ns_single_CPU670
   + pdt_m_rows_CPU670 = 0
 ns_sketch_CPU670: - ns_single_CPU670
   + 11.42863142857143 cores_sketch_CPU670 >= 0
 capacity_cores_CPU670: cores_sketch_CPU670 <= 2
 ns_single_CPU33: 3.5 rows_tot_CPU33 - ns_single_CPU33 + pdt_m_rows_CPU33
   = 0
 ns_sketch_CPU33: - ns_single_CPU33 + 11.42863142857143 cores_sketch_CPU33
   >= 0
 capacity_cores_CPU33: cores_sketch_CPU33 <= 2
 ns_single_CPU489: 3.5 rows_tot_CPU489 - ns_single_CPU489
   + pdt_m_rows_CPU489 = 0
 ns_sketch_CPU489: - ns_single_CPU489
   + 11.42863142857143 cores_sketch_CPU489 >= 0
 capacity_cores_CPU489: cores_sketch_CPU489 <= 2
 ns_single_CPU455: 3.5 rows_tot_CPU455 - ns_single_CPU455
   + pdt_m_rows_CPU455 = 0
 ns_sketch_CPU455: - ns_single_CPU455
   + 11.42863142857143 cores_sketch_CPU455 >= 0
 capacity_cores_CPU455: cores_sketch_CPU455 <= 2
 ns_single_CPU379: 3.5 rows_tot_CPU379 - ns_single_CPU379
   + pdt_m_rows_CPU379 = 0
 ns_sketch_CPU379: - ns_single_CPU379
   + 11.42863142857143 cores_sketch_CPU379 >= 0
 capacity_cores_CPU379: cores_sketch_CPU379 <= 2
 ns_single_CPU488: 3.5 rows_tot_CPU488 - ns_single_CPU488
   + pdt_m_rows_CPU488 = 0
 ns_sketch_CPU488: - ns_single_CPU488
   + 11.42863142857143 cores_sketch_CPU488 >= 0
 capacity_cores_CPU488: cores_sketch_CPU488 <= 2
 ns_single_CPU420: 3.5 rows_tot_CPU420 - ns_single_CPU420
   + pdt_m_rows_CPU420 = 0
 ns_sketch_CPU420: - ns_single_CPU420
   + 11.42863142857143 cores_sketch_CPU420 >= 0
 capacity_cores_CPU420: cores_sketch_CPU420 <= 2
 ns_single_CPU433: 3.5 rows_tot_CPU433 - ns_single_CPU433
   + pdt_m_rows_CPU433 = 0
 ns_sketch_CPU433: - ns_single_CPU433
   + 11.42863142857143 cores_sketch_CPU433 >= 0
 capacity_cores_CPU433: cores_sketch_CPU433 <= 2
 res_acc: 1.220703125e-04 mem_tot_CPU609 + 1.220703125e-04 mem_tot_CPU568
   + 1.220703125e-04 mem_tot_CPU687 + 1.220703125e-04 mem_tot_CPU445
   + 1.220703125e-04 mem_tot_CPU157 + 1.220703125e-04 mem_tot_CPU151
   + 1.220703125e-04 mem_tot_CPU743 + 1.220703125e-04 mem_tot_CPU507
   + 1.220703125e-04 mem_tot_CPU670 + 1.220703125e-04 mem_tot_CPU33
   + 1.220703125e-04 mem_tot_CPU489 + 1.220703125e-04 mem_tot_CPU455
   + 1.220703125e-04 mem_tot_CPU379 + 1.220703125e-04 mem_tot_CPU488
   + 1.220703125e-04 mem_tot_CPU420 + 1.220703125e-04 mem_tot_CPU433
   + 10 cores_sketch_CPU609 + 10 cores_sketch_CPU568
   + 10 cores_sketch_CPU687 + 10 cores_sketch_CPU445
   + 10 cores_sketch_CPU157 + 10 cores_sketch_CPU151
   + 10 cores_sketch_CPU743 + 10 cores_sketch_CPU507
   + 10 cores_sketch_CPU670 + 10 cores_sketch_CPU33
   + 10 cores_sketch_CPU489 + 10 cores_sketch_CPU455
   + 10 cores_sketch_CPU379 + 10 cores_sketch_CPU488
   + 10 cores_sketch_CPU420 + 10 cores_sketch_CPU433 - res = -800
 pdt_m_rows_CPU609: pdt_m_rows_CPU609 + [
   - rows_tot_CPU609 * m_access_time_CPU609 ] = 0
 pdt_m_rows_CPU568: pdt_m_rows_CPU568 + [
   - rows_tot_CPU568 * m_access_time_CPU568 ] = 0
 pdt_m_rows_CPU687: pdt_m_rows_CPU687 + [
   - rows_tot_CPU687 * m_access_time_CPU687 ] = 0
 pdt_m_rows_CPU445: pdt_m_rows_CPU445 + [
   - rows_tot_CPU445 * m_access_time_CPU445 ] = 0
 pdt_m_rows_CPU157: pdt_m_rows_CPU157 + [
   - rows_tot_CPU157 * m_access_time_CPU157 ] = 0
 pdt_m_rows_CPU151: pdt_m_rows_CPU151 + [
   - rows_tot_CPU151 * m_access_time_CPU151 ] = 0
 pdt_m_rows_CPU743: pdt_m_rows_CPU743 + [
   - rows_tot_CPU743 * m_access_time_CPU743 ] = 0
 pdt_m_rows_CPU507: pdt_m_rows_CPU507 + [
   - rows_tot_CPU507 * m_access_time_CPU507 ] = 0
 pdt_m_rows_CPU670: pdt_m_rows_CPU670 + [
   - rows_tot_CPU670 * m_access_time_CPU670 ] = 0
 pdt_m_rows_CPU33: pdt_m_rows_CPU33 + [
   - rows_tot_CPU33 * m_access_time_CPU33 ] = 0
 pdt_m_rows_CPU489: pdt_m_rows_CPU489 + [
   - rows_tot_CPU489 * m_access_time_CPU489 ] = 0
 pdt_m_rows_CPU455: pdt_m_rows_CPU455 + [
   - rows_tot_CPU455 * m_access_time_CPU455 ] = 0
 pdt_m_rows_CPU379: pdt_m_rows_CPU379 + [
   - rows_tot_CPU379 * m_access_time_CPU379 ] = 0
 pdt_m_rows_CPU488: pdt_m_rows_CPU488 + [
   - rows_tot_CPU488 * m_access_time_CPU488 ] = 0
 pdt_m_rows_CPU420: pdt_m_rows_CPU420 + [
   - rows_tot_CPU420 * m_access_time_CPU420 ] = 0
 pdt_m_rows_CPU433: pdt_m_rows_CPU433 + [
   - rows_tot_CPU433 * m_access_time_CPU433 ] = 0
Bounds
 frac[0,0] <= 1
 frac[0,12] <= 1
 frac[1,14] <= 1
 frac[1,4] <= 1
 frac[1,12] <= 1
 frac[1,7] <= 1
 frac[2,9] <= 1
 frac[2,6] <= 1
 frac[2,7] <= 1
 frac[2,0] <= 1
 frac[2,4] <= 1
 frac[3,4] <= 1
 frac[4,7] <= 1
 frac[4,4] <= 1
 frac[4,14] <= 1
 frac[4,2] <= 1
 frac[5,9] <= 1
 frac[5,4] <= 1
 frac[5,14] <= 1
 frac[6,6] <= 1
 frac[6,2] <= 1
 frac[6,14] <= 1
 frac[6,7] <= 1
 frac[6,4] <= 1
 frac[6,0] <= 1
 frac[7,7] <= 1
 frac[7,0] <= 1
 frac[7,4] <= 1
 frac[8,5] <= 1
 frac[8,11] <= 1
 frac[8,3] <= 1
 frac[9,1] <= 1
 frac[9,13] <= 1
 frac[10,11] <= 1
 frac[10,15] <= 1
 frac[11,11] <= 1
 frac[11,1] <= 1
 frac[11,8] <= 1
 frac[11,15] <= 1
 frac[12,5] <= 1
 frac[12,13] <= 1
 frac[12,10] <= 1
 frac[12,11] <= 1
 frac[13,11] <= 1
 frac[13,15] <= 1
 frac[13,10] <= 1
 frac[14,5] <= 1
 frac[14,11] <= 1
 frac[14,8] <= 1
 frac[14,3] <= 1
 frac[15,13] <= 1
 frac[15,5] <= 1
 frac[15,15] <= 1
 mem_tot_CPU609 <= 32768
 normalized_mem_tot_CPU609 <= 1
 normalized_rows_tot_CPU609 <= 1
 mem_tot_CPU568 <= 32768
 normalized_mem_tot_CPU568 <= 1
 normalized_rows_tot_CPU568 <= 1
 mem_tot_CPU687 <= 32768
 normalized_mem_tot_CPU687 <= 1
 normalized_rows_tot_CPU687 <= 1
 mem_tot_CPU445 <= 32768
 normalized_mem_tot_CPU445 <= 1
 normalized_rows_tot_CPU445 <= 1
 mem_tot_CPU157 <= 32768
 normalized_mem_tot_CPU157 <= 1
 normalized_rows_tot_CPU157 <= 1
 mem_tot_CPU151 <= 32768
 normalized_mem_tot_CPU151 <= 1
 normalized_rows_tot_CPU151 <= 1
 mem_tot_CPU743 <= 32768
 normalized_mem_tot_CPU743 <= 1
 normalized_rows_tot_CPU743 <= 1
 mem_tot_CPU507 <= 32768
 normalized_mem_tot_CPU507 <= 1
 normalized_rows_tot_CPU507 <= 1
 mem_tot_CPU670 <= 32768
 normalized_mem_tot_CPU670 <= 1
 normalized_rows_tot_CPU670 <= 1
 mem_tot_CPU33 <= 32768
 normalized_mem_tot_CPU33 <= 1
 normalized_rows_tot_CPU33 <= 1
 mem_tot_CPU489 <= 32768
 normalized_mem_tot_CPU489 <= 1
 normalized_rows_tot_CPU489 <= 1
 mem_tot_CPU455 <= 32768
 normalized_mem_tot_CPU455 <= 1
 normalized_rows_tot_CPU455 <= 1
 mem_tot_CPU379 <= 32768
 normalized_mem_tot_CPU379 <= 1
 normalized_rows_tot_CPU379 <= 1
 mem_tot_CPU488 <= 32768
 normalized_mem_tot_CPU488 <= 1
 normalized_rows_tot_CPU488 <= 1
 mem_tot_CPU420 <= 32768
 normalized_mem_tot_CPU420 <= 1
 normalized_rows_tot_CPU420 <= 1
 mem_tot_CPU433 <= 32768
 normalized_mem_tot_CPU433 <= 1
 normalized_rows_tot_CPU433 <= 1
 cores_sketch_CPU609 <= 7
 cores_sketch_CPU568 <= 7
 cores_sketch_CPU687 <= 7
 cores_sketch_CPU445 <= 7
 cores_sketch_CPU157 <= 7
 cores_sketch_CPU151 <= 7
 cores_sketch_CPU743 <= 7
 cores_sketch_CPU507 <= 7
 cores_sketch_CPU670 <= 7
 cores_sketch_CPU33 <= 7
 cores_sketch_CPU489 <= 7
 cores_sketch_CPU455 <= 7
 cores_sketch_CPU379 <= 7
 cores_sketch_CPU488 <= 7
 cores_sketch_CPU420 <= 7
 cores_sketch_CPU433 <= 7
Generals
 cores_sketch_CPU609 cores_sketch_CPU568 cores_sketch_CPU687
 cores_sketch_CPU445 cores_sketch_CPU157 cores_sketch_CPU151
 cores_sketch_CPU743 cores_sketch_CPU507 cores_sketch_CPU670
 cores_sketch_CPU33 cores_sketch_CPU489 cores_sketch_CPU455
 cores_sketch_CPU379 cores_sketch_CPU488 cores_sketch_CPU420
 cores_sketch_CPU433
General Constraints
 mem_overall_others: max_mem_others = MAX ( normalized_mem_tot_CPU609 ,
   normalized_mem_tot_CPU568 , normalized_mem_tot_CPU687 ,
   normalized_mem_tot_CPU445 , normalized_mem_tot_CPU157 ,
   normalized_mem_tot_CPU151 , normalized_mem_tot_CPU743 ,
   normalized_mem_tot_CPU507 , normalized_mem_tot_CPU670 ,
   normalized_mem_tot_CPU33 , normalized_mem_tot_CPU489 ,
   normalized_mem_tot_CPU455 , normalized_mem_tot_CPU379 ,
   normalized_mem_tot_CPU488 , normalized_mem_tot_CPU420 ,
   normalized_mem_tot_CPU433 )
 rows_overall_others: max_rows_others = MAX ( normalized_rows_tot_CPU609 ,
   normalized_rows_tot_CPU568 , normalized_rows_tot_CPU687 ,
   normalized_rows_tot_CPU445 , normalized_rows_tot_CPU157 ,
   normalized_rows_tot_CPU151 , normalized_rows_tot_CPU743 ,
   normalized_rows_tot_CPU507 , normalized_rows_tot_CPU670 ,
   normalized_rows_tot_CPU33 , normalized_rows_tot_CPU489 ,
   normalized_rows_tot_CPU455 , normalized_rows_tot_CPU379 ,
   normalized_rows_tot_CPU488 , normalized_rows_tot_CPU420 ,
   normalized_rows_tot_CPU433 )
 mem_access_time_CPU609: m_access_time_CPU609 = PWL ( mem_tot_CPU609 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU568: m_access_time_CPU568 = PWL ( mem_tot_CPU568 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU687: m_access_time_CPU687 = PWL ( mem_tot_CPU687 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU445: m_access_time_CPU445 = PWL ( mem_tot_CPU445 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU157: m_access_time_CPU157 = PWL ( mem_tot_CPU157 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU151: m_access_time_CPU151 = PWL ( mem_tot_CPU151 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU743: m_access_time_CPU743 = PWL ( mem_tot_CPU743 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU507: m_access_time_CPU507 = PWL ( mem_tot_CPU507 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU670: m_access_time_CPU670 = PWL ( mem_tot_CPU670 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU33: m_access_time_CPU33 = PWL ( mem_tot_CPU33 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU489: m_access_time_CPU489 = PWL ( mem_tot_CPU489 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU455: m_access_time_CPU455 = PWL ( mem_tot_CPU455 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU379: m_access_time_CPU379 = PWL ( mem_tot_CPU379 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU488: m_access_time_CPU488 = PWL ( mem_tot_CPU488 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU420: m_access_time_CPU420 = PWL ( mem_tot_CPU420 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU433: m_access_time_CPU433 = PWL ( mem_tot_CPU433 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_P4 + 0 max_mem_CPU_Cluster + 0 max_rows_P4
   + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU_1 + res
Subject To
 cov_0_0: frac[20,0] + frac[0,0] + frac[1,0] >= 1
 cov_1_0: frac[20,0] + frac[1,0] + frac[2,0] >= 1
 cov_2_0: frac[3,0] + frac[20,0] + frac[2,0] >= 1
 cov_3_0: frac[3,0] + frac[4,0] + frac[20,0] >= 1
 cov_4_0: frac[4,0] + frac[20,0] + frac[5,0] >= 1
 cov_5_0: frac[20,0] + frac[5,0] + frac[6,0] >= 1
 cov_6_0: frac[20,0] + frac[6,0] + frac[7,0] >= 1
 cov_7_0: frac[8,0] + frac[20,0] + frac[7,0] >= 1
 cov_8_0: frac[8,0] + frac[9,0] + frac[20,0] >= 1
 cov_9_0: frac[9,0] + frac[20,0] + frac[10,0] >= 1
 cov_10_0: frac[20,0] + frac[10,0] + frac[11,0] >= 1
 cov_11_0: frac[20,0] + frac[11,0] + frac[12,0] >= 1
 cov_12_0: frac[20,0] + frac[12,0] + frac[13,0] >= 1
 cov_13_0: frac[14,0] + frac[20,0] + frac[13,0] >= 1
 cov_14_0: frac[14,0] + frac[15,0] + frac[20,0] >= 1
 cov_15_0: frac[15,0] + frac[20,0] + frac[16,0] >= 1
 cov_16_0: frac[20,0] + frac[16,0] + frac[17,0] >= 1
 cov_17_0: frac[18,0] + frac[20,0] + frac[17,0] >= 1
 cov_18_0: frac[18,0] + frac[19,0] + frac[20,0] >= 1
 cov_19_0: frac[19,0] + frac[20,0] + frac[0,0] >= 1
 accuracy_18_0: - 5096.7890625 frac[18,0] + mem[18,0] = 0
 accuracy_3_0: - 5096.7890625 frac[3,0] + mem[3,0] = 0
 accuracy_8_0: - 5096.7890625 frac[8,0] + mem[8,0] = 0
 accuracy_14_0: - 5096.7890625 frac[14,0] + mem[14,0] = 0
 accuracy_19_0: - 5096.7890625 frac[19,0] + mem[19,0] = 0
 accuracy_4_0: - 5096.7890625 frac[4,0] + mem[4,0] = 0
 accuracy_9_0: - 5096.7890625 frac[9,0] + mem[9,0] = 0
 accuracy_15_0: - 5096.7890625 frac[15,0] + mem[15,0] = 0
 accuracy_20_0: - 5096.7890625 frac[20,0] + mem[20,0] = 0
 accuracy_5_0: - 5096.7890625 frac[5,0] + mem[5,0] = 0
 accuracy_10_0: - 5096.7890625 frac[10,0] + mem[10,0] = 0
 accuracy_0_0: - 5096.7890625 frac[0,0] + mem[0,0] = 0
 accuracy_6_0: - 5096.7890625 frac[6,0] + mem[6,0] = 0
 accuracy_11_0: - 5096.7890625 frac[11,0] + mem[11,0] = 0
 accuracy_16_0: - 5096.7890625 frac[16,0] + mem[16,0] = 0
 accuracy_1_0: - 5096.7890625 frac[1,0] + mem[1,0] = 0
 accuracy_7_0: - 5096.7890625 frac[7,0] + mem[7,0] = 0
 accuracy_12_0: - 5096.7890625 frac[12,0] + mem[12,0] = 0
 accuracy_17_0: - 5096.7890625 frac[17,0] + mem[17,0] = 0
 accuracy_2_0: - 5096.7890625 frac[2,0] + mem[2,0] = 0
 accuracy_13_0: - 5096.7890625 frac[13,0] + mem[13,0] = 0
 mem_tot_P4_0: - mem[0,0] + mem_tot_P4_0 = 0
 R42: - 0.00173611111111111 mem_tot_P4_0 + normalized_mem_tot_P4_0 = 0
 rows_tot_P4_0: - 3 frac[0,0] + rows_tot_P4_0 = 0
 R44: - 0.0208333333333333 rows_tot_P4_0 + normalized_rows_tot_P4_0 = 0
 mem_tot_P4_1: - mem[1,0] + mem_tot_P4_1 = 0
 R46: - 0.00173611111111111 mem_tot_P4_1 + normalized_mem_tot_P4_1 = 0
 rows_tot_P4_1: - 3 frac[1,0] + rows_tot_P4_1 = 0
 R48: - 0.0208333333333333 rows_tot_P4_1 + normalized_rows_tot_P4_1 = 0
 mem_tot_P4_2: - mem[2,0] + mem_tot_P4_2 = 0
 R50: - 0.00173611111111111 mem_tot_P4_2 + normalized_mem_tot_P4_2 = 0
 rows_tot_P4_2: - 3 frac[2,0] + rows_tot_P4_2 = 0
 R52: - 0.0208333333333333 rows_tot_P4_2 + normalized_rows_tot_P4_2 = 0
 mem_tot_P4_3: - mem[3,0] + mem_tot_P4_3 = 0
 R54: - 0.00173611111111111 mem_tot_P4_3 + normalized_mem_tot_P4_3 = 0
 rows_tot_P4_3: - 3 frac[3,0] + rows_tot_P4_3 = 0
 R56: - 0.0208333333333333 rows_tot_P4_3 + normalized_rows_tot_P4_3 = 0
 mem_tot_P4_4: - mem[4,0] + mem_tot_P4_4 = 0
 R58: - 0.00173611111111111 mem_tot_P4_4 + normalized_mem_tot_P4_4 = 0
 rows_tot_P4_4: - 3 frac[4,0] + rows_tot_P4_4 = 0
 R60: - 0.0208333333333333 rows_tot_P4_4 + normalized_rows_tot_P4_4 = 0
 mem_tot_P4_5: - mem[5,0] + mem_tot_P4_5 = 0
 R62: - 0.00173611111111111 mem_tot_P4_5 + normalized_mem_tot_P4_5 = 0
 rows_tot_P4_5: - 3 frac[5,0] + rows_tot_P4_5 = 0
 R64: - 0.0208333333333333 rows_tot_P4_5 + normalized_rows_tot_P4_5 = 0
 mem_tot_P4_6: - mem[6,0] + mem_tot_P4_6 = 0
 R66: - 0.00173611111111111 mem_tot_P4_6 + normalized_mem_tot_P4_6 = 0
 rows_tot_P4_6: - 3 frac[6,0] + rows_tot_P4_6 = 0
 R68: - 0.0208333333333333 rows_tot_P4_6 + normalized_rows_tot_P4_6 = 0
 mem_tot_P4_7: - mem[7,0] + mem_tot_P4_7 = 0
 R70: - 0.00173611111111111 mem_tot_P4_7 + normalized_mem_tot_P4_7 = 0
 rows_tot_P4_7: - 3 frac[7,0] + rows_tot_P4_7 = 0
 R72: - 0.0208333333333333 rows_tot_P4_7 + normalized_rows_tot_P4_7 = 0
 mem_tot_P4_8: - mem[8,0] + mem_tot_P4_8 = 0
 R74: - 0.00173611111111111 mem_tot_P4_8 + normalized_mem_tot_P4_8 = 0
 rows_tot_P4_8: - 3 frac[8,0] + rows_tot_P4_8 = 0
 R76: - 0.0208333333333333 rows_tot_P4_8 + normalized_rows_tot_P4_8 = 0
 mem_tot_P4_9: - mem[9,0] + mem_tot_P4_9 = 0
 R78: - 0.00173611111111111 mem_tot_P4_9 + normalized_mem_tot_P4_9 = 0
 rows_tot_P4_9: - 3 frac[9,0] + rows_tot_P4_9 = 0
 R80: - 0.0208333333333333 rows_tot_P4_9 + normalized_rows_tot_P4_9 = 0
 mem_tot_P4_10: - mem[10,0] + mem_tot_P4_10 = 0
 R82: - 0.00173611111111111 mem_tot_P4_10 + normalized_mem_tot_P4_10 = 0
 rows_tot_P4_10: - 3 frac[10,0] + rows_tot_P4_10 = 0
 R84: - 0.0208333333333333 rows_tot_P4_10 + normalized_rows_tot_P4_10 = 0
 mem_tot_P4_11: - mem[11,0] + mem_tot_P4_11 = 0
 R86: - 0.00173611111111111 mem_tot_P4_11 + normalized_mem_tot_P4_11 = 0
 rows_tot_P4_11: - 3 frac[11,0] + rows_tot_P4_11 = 0
 R88: - 0.0208333333333333 rows_tot_P4_11 + normalized_rows_tot_P4_11 = 0
 mem_tot_P4_12: - mem[12,0] + mem_tot_P4_12 = 0
 R90: - 0.00173611111111111 mem_tot_P4_12 + normalized_mem_tot_P4_12 = 0
 rows_tot_P4_12: - 3 frac[12,0] + rows_tot_P4_12 = 0
 R92: - 0.0208333333333333 rows_tot_P4_12 + normalized_rows_tot_P4_12 = 0
 mem_tot_P4_13: - mem[13,0] + mem_tot_P4_13 = 0
 R94: - 0.00173611111111111 mem_tot_P4_13 + normalized_mem_tot_P4_13 = 0
 rows_tot_P4_13: - 3 frac[13,0] + rows_tot_P4_13 = 0
 R96: - 0.0208333333333333 rows_tot_P4_13 + normalized_rows_tot_P4_13 = 0
 mem_tot_P4_14: - mem[14,0] + mem_tot_P4_14 = 0
 R98: - 0.00173611111111111 mem_tot_P4_14 + normalized_mem_tot_P4_14 = 0
 rows_tot_P4_14: - 3 frac[14,0] + rows_tot_P4_14 = 0
 R100: - 0.0208333333333333 rows_tot_P4_14 + normalized_rows_tot_P4_14 = 0
 mem_tot_P4_15: - mem[15,0] + mem_tot_P4_15 = 0
 R102: - 0.00173611111111111 mem_tot_P4_15 + normalized_mem_tot_P4_15 = 0
 rows_tot_P4_15: - 3 frac[15,0] + rows_tot_P4_15 = 0
 R104: - 0.0208333333333333 rows_tot_P4_15 + normalized_rows_tot_P4_15 = 0
 mem_tot_P4_16: - mem[16,0] + mem_tot_P4_16 = 0
 R106: - 0.00173611111111111 mem_tot_P4_16 + normalized_mem_tot_P4_16 = 0
 rows_tot_P4_16: - 3 frac[16,0] + rows_tot_P4_16 = 0
 R108: - 0.0208333333333333 rows_tot_P4_16 + normalized_rows_tot_P4_16 = 0
 mem_tot_P4_17: - mem[17,0] + mem_tot_P4_17 = 0
 R110: - 0.00173611111111111 mem_tot_P4_17 + normalized_mem_tot_P4_17 = 0
 rows_tot_P4_17: - 3 frac[17,0] + rows_tot_P4_17 = 0
 R112: - 0.0208333333333333 rows_tot_P4_17 + normalized_rows_tot_P4_17 = 0
 mem_tot_P4_18: - mem[18,0] + mem_tot_P4_18 = 0
 R114: - 0.00173611111111111 mem_tot_P4_18 + normalized_mem_tot_P4_18 = 0
 rows_tot_P4_18: - 3 frac[18,0] + rows_tot_P4_18 = 0
 R116: - 0.0208333333333333 rows_tot_P4_18 + normalized_rows_tot_P4_18 = 0
 mem_tot_P4_19: - mem[19,0] + mem_tot_P4_19 = 0
 R118: - 0.00173611111111111 mem_tot_P4_19 + normalized_mem_tot_P4_19 = 0
 rows_tot_P4_19: - 3 frac[19,0] + rows_tot_P4_19 = 0
 R120: - 0.0208333333333333 rows_tot_P4_19 + normalized_rows_tot_P4_19 = 0
 mem_tot_CPU_1: - mem[20,0] + mem_tot_CPU_1 = 0
 R122: - 3.05176e-05 mem_tot_CPU_1 + normalized_mem_tot_CPU_1 = 0
 rows_tot_CPU_1: - 3 frac[20,0] + rows_tot_CPU_1 = 0
 R124: - 0.0833333333333333 rows_tot_CPU_1 + normalized_rows_tot_CPU_1 = 0
 row_capacity_P4_0: rows_tot_P4_0 <= 48
 row_capacity_P4_1: rows_tot_P4_1 <= 48
 row_capacity_P4_2: rows_tot_P4_2 <= 48
 row_capacity_P4_3: rows_tot_P4_3 <= 48
 row_capacity_P4_4: rows_tot_P4_4 <= 48
 row_capacity_P4_5: rows_tot_P4_5 <= 48
 row_capacity_P4_6: rows_tot_P4_6 <= 48
 row_capacity_P4_7: rows_tot_P4_7 <= 48
 row_capacity_P4_8: rows_tot_P4_8 <= 48
 row_capacity_P4_9: rows_tot_P4_9 <= 48
 row_capacity_P4_10: rows_tot_P4_10 <= 48
 row_capacity_P4_11: rows_tot_P4_11 <= 48
 row_capacity_P4_12: rows_tot_P4_12 <= 48
 row_capacity_P4_13: rows_tot_P4_13 <= 48
 row_capacity_P4_14: rows_tot_P4_14 <= 48
 row_capacity_P4_15: rows_tot_P4_15 <= 48
 row_capacity_P4_16: rows_tot_P4_16 <= 48
 row_capacity_P4_17: rows_tot_P4_17 <= 48
 row_capacity_P4_18: rows_tot_P4_18 <= 48
 row_capacity_P4_19: rows_tot_P4_19 <= 48
 row_capacity_CPU_1: rows_tot_CPU_1 <= 12
 ns_P4_0: ns_P4_0 = 6.756756756756757
 max_ns_P4_0: ns_P4_0 <= 11.42857142857143
 ns_P4_1: ns_P4_1 = 6.756756756756757
 max_ns_P4_1: ns_P4_1 <= 11.42857142857143
 ns_P4_2: ns_P4_2 = 6.756756756756757
 max_ns_P4_2: ns_P4_2 <= 11.42857142857143
 ns_P4_3: ns_P4_3 = 6.756756756756757
 max_ns_P4_3: ns_P4_3 <= 11.42857142857143
 ns_P4_4: ns_P4_4 = 6.756756756756757
 max_ns_P4_4: ns_P4_4 <= 11.42857142857143
 ns_P4_5: ns_P4_5 = 6.756756756756757
 max_ns_P4_5: ns_P4_5 <= 11.42857142857143
 ns_P4_6: ns_P4_6 = 6.756756756756757
 max_ns_P4_6: ns_P4_6 <= 11.42857142857143
 ns_P4_7: ns_P4_7 = 6.756756756756757
 max_ns_P4_7: ns_P4_7 <= 11.42857142857143
 ns_P4_8: ns_P4_8 = 6.756756756756757
 max_ns_P4_8: ns_P4_8 <= 11.42857142857143
 ns_P4_9: ns_P4_9 = 6.756756756756757
 max_ns_P4_9: ns_P4_9 <= 11.42857142857143
 ns_P4_10: ns_P4_10 = 6.756756756756757
 max_ns_P4_10: ns_P4_10 <= 11.42857142857143
 ns_P4_11: ns_P4_11 = 6.756756756756757
 max_ns_P4_11: ns_P4_11 <= 11.42857142857143
 ns_P4_12: ns_P4_12 = 6.756756756756757
 max_ns_P4_12: ns_P4_12 <= 11.42857142857143
 ns_P4_13: ns_P4_13 = 6.756756756756757
 max_ns_P4_13: ns_P4_13 <= 11.42857142857143
 ns_P4_14: ns_P4_14 = 6.756756756756757
 max_ns_P4_14: ns_P4_14 <= 11.42857142857143
 ns_P4_15: ns_P4_15 = 6.756756756756757
 max_ns_P4_15: ns_P4_15 <= 11.42857142857143
 ns_P4_16: ns_P4_16 = 6.756756756756757
 max_ns_P4_16: ns_P4_16 <= 11.42857142857143
 ns_P4_17: ns_P4_17 = 6.756756756756757
 max_ns_P4_17: ns_P4_17 <= 11.42857142857143
 ns_P4_18: ns_P4_18 = 6.756756756756757
 max_ns_P4_18: ns_P4_18 <= 11.42857142857143
 ns_P4_19: ns_P4_19 = 6.756756756756757
 max_ns_P4_19: ns_P4_19 <= 11.42857142857143
 ns_single_CPU_1: 3.5 rows_tot_CPU_1 - ns_single_CPU_1 + pdt_m_rows_CPU_1
   = 0
 capacity_cores_CPU_1: cores_sketch_CPU_1 + cores_dpdk_CPU_1 <= 7
 ns_sketch_CPU_1: - ns_single_CPU_1 + 11.42857142857143 cores_sketch_CPU_1
   >= 0
 ns_dpdk_CPU_1: - 4.285714285714287 cores_dpdk_CPU_1 <= -21.42857142857143
 res_acc: 0.0208333333333333 mem_tot_P4_0 + 0.25 rows_tot_P4_0
   + 0.0208333333333333 mem_tot_P4_1 + 0.25 rows_tot_P4_1
   + 0.0208333333333333 mem_tot_P4_2 + 0.25 rows_tot_P4_2
   + 0.0208333333333333 mem_tot_P4_3 + 0.25 rows_tot_P4_3
   + 0.0208333333333333 mem_tot_P4_4 + 0.25 rows_tot_P4_4
   + 0.0208333333333333 mem_tot_P4_5 + 0.25 rows_tot_P4_5
   + 0.0208333333333333 mem_tot_P4_6 + 0.25 rows_tot_P4_6
   + 0.0208333333333333 mem_tot_P4_7 + 0.25 rows_tot_P4_7
   + 0.0208333333333333 mem_tot_P4_8 + 0.25 rows_tot_P4_8
   + 0.0208333333333333 mem_tot_P4_9 + 0.25 rows_tot_P4_9
   + 0.0208333333333333 mem_tot_P4_10 + 0.25 rows_tot_P4_10
   + 0.0208333333333333 mem_tot_P4_11 + 0.25 rows_tot_P4_11
   + 0.0208333333333333 mem_tot_P4_12 + 0.25 rows_tot_P4_12
   + 0.0208333333333333 mem_tot_P4_13 + 0.25 rows_tot_P4_13
   + 0.0208333333333333 mem_tot_P4_14 + 0.25 rows_tot_P4_14
   + 0.0208333333333333 mem_tot_P4_15 + 0.25 rows_tot_P4_15
   + 0.0208333333333333 mem_tot_P4_16 + 0.25 rows_tot_P4_16
   + 0.0208333333333333 mem_tot_P4_17 + 0.25 rows_tot_P4_17
   + 0.0208333333333333 mem_tot_P4_18 + 0.25 rows_tot_P4_18
   + 0.0208333333333333 mem_tot_P4_19 + 0.25 rows_tot_P4_19
   + 1.220703125e-04 mem_tot_CPU_1 + 10 cores_sketch_CPU_1
   + 10 cores_dpdk_CPU_1 - res = 0
 pdt_m_rows_CPU_1: pdt_m_rows_CPU_1 + [
   - rows_tot_CPU_1 * m_access_time_CPU_1 ] = 0
Bounds
 frac[18,0] <= 1
 frac[3,0] <= 1
 frac[8,0] <= 1
 frac[14,0] <= 1
 frac[19,0] <= 1
 frac[4,0] <= 1
 frac[9,0] <= 1
 frac[15,0] <= 1
 frac[20,0] <= 1
 frac[5,0] <= 1
 frac[10,0] <= 1
 frac[0,0] <= 1
 frac[6,0] <= 1
 frac[11,0] <= 1
 frac[16,0] <= 1
 frac[1,0] <= 1
 frac[7,0] <= 1
 frac[12,0] <= 1
 frac[17,0] <= 1
 frac[2,0] <= 1
 frac[13,0] <= 1
 mem_tot_P4_0 <= 576
 normalized_mem_tot_P4_0 <= 1
 normalized_rows_tot_P4_0 <= 1
 mem_tot_P4_1 <= 576
 normalized_mem_tot_P4_1 <= 1
 normalized_rows_tot_P4_1 <= 1
 mem_tot_P4_2 <= 576
 normalized_mem_tot_P4_2 <= 1
 normalized_rows_tot_P4_2 <= 1
 mem_tot_P4_3 <= 576
 normalized_mem_tot_P4_3 <= 1
 normalized_rows_tot_P4_3 <= 1
 mem_tot_P4_4 <= 576
 normalized_mem_tot_P4_4 <= 1
 normalized_rows_tot_P4_4 <= 1
 mem_tot_P4_5 <= 576
 normalized_mem_tot_P4_5 <= 1
 normalized_rows_tot_P4_5 <= 1
 mem_tot_P4_6 <= 576
 normalized_mem_tot_P4_6 <= 1
 normalized_rows_tot_P4_6 <= 1
 mem_tot_P4_7 <= 576
 normalized_mem_tot_P4_7 <= 1
 normalized_rows_tot_P4_7 <= 1
 mem_tot_P4_8 <= 576
 normalized_mem_tot_P4_8 <= 1
 normalized_rows_tot_P4_8 <= 1
 mem_tot_P4_9 <= 576
 normalized_mem_tot_P4_9 <= 1
 normalized_rows_tot_P4_9 <= 1
 mem_tot_P4_10 <= 576
 normalized_mem_tot_P4_10 <= 1
 normalized_rows_tot_P4_10 <= 1
 mem_tot_P4_11 <= 576
 normalized_mem_tot_P4_11 <= 1
 normalized_rows_tot_P4_11 <= 1
 mem_tot_P4_12 <= 576
 normalized_mem_tot_P4_12 <= 1
 normalized_rows_tot_P4_12 <= 1
 mem_tot_P4_13 <= 576
 normalized_mem_tot_P4_13 <= 1
 normalized_rows_tot_P4_13 <= 1
 mem_tot_P4_14 <= 576
 normalized_mem_tot_P4_14 <= 1
 normalized_rows_tot_P4_14 <= 1
 mem_tot_P4_15 <= 576
 normalized_mem_tot_P4_15 <= 1
 normalized_rows_tot_P4_15 <= 1
 mem_tot_P4_16 <= 576
 normalized_mem_tot_P4_16 <= 1
 normalized_rows_tot_P4_16 <= 1
 mem_tot_P4_17 <= 576
 normalized_mem_tot_P4_17 <= 1
 normalized_rows_tot_P4_17 <= 1
 mem_tot_P4_18 <= 576
 normalized_mem_tot_P4_18 <= 1
 normalized_rows_tot_P4_18 <= 1
 mem_tot_P4_19 <= 576
 normalized_mem_tot_P4_19 <= 1
 normalized_rows_tot_P4_19 <= 1
 mem_tot_CPU_1 <= 32768
 normalized_mem_tot_CPU_1 <= 1
 normalized_rows_tot_CPU_1 <= 1
 cores_sketch_CPU_1 <= 7
 1 <= cores_dpdk_CPU_1 <= 7
Generals
 cores_sketch_CPU_1 cores_dpdk_CPU_1
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_P4_0 , mem_tot_P4_1 ,
   mem_tot_P4_2 , mem_tot_P4_3 , mem_tot_P4_4 , mem_tot_P4_5 ,
   mem_tot_P4_6 , mem_tot_P4_7 , mem_tot_P4_8 , mem_tot_P4_9 ,
   mem_tot_P4_10 , mem_tot_P4_11 , mem_tot_P4_12 , mem_tot_P4_13 ,
   mem_tot_P4_14 , mem_tot_P4_15 , mem_tot_P4_16 , mem_tot_P4_17 ,
   mem_tot_P4_18 , mem_tot_P4_19 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU_1 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_P4_0 , rows_tot_P4_1 ,
   rows_tot_P4_2 , rows_tot_P4_3 , rows_tot_P4_4 , rows_tot_P4_5 ,
   rows_tot_P4_6 , rows_tot_P4_7 , rows_tot_P4_8 , rows_tot_P4_9 ,
   rows_tot_P4_10 , rows_tot_P4_11 , rows_tot_P4_12 , rows_tot_P4_13 ,
   rows_tot_P4_14 , rows_tot_P4_15 , rows_tot_P4_16 , rows_tot_P4_17 ,
   rows_tot_P4_18 , rows_tot_P4_19 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU_1 )
 mem_access_time_CPU_1: m_access_time_CPU_1 = PWL ( mem_tot_CPU_1 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

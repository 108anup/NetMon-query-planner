\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 m_access_time_CPU_0 + 0 ns_sketch_CPU_0 + 0 ns_dpdk_CPU_0 + 0 ns_CPU_0
   + 0 m_access_time_CPU_1 + 0 ns_sketch_CPU_1 + 0 ns_dpdk_CPU_1
   + 0 ns_CPU_1 + 0 m_access_time_CPU_2 + 0 ns_sketch_CPU_2
   + 0 ns_dpdk_CPU_2 + 0 ns_CPU_2 + 0 m_access_time_CPU_3
   + 0 ns_sketch_CPU_3 + 0 ns_dpdk_CPU_3 + 0 ns_CPU_3
   + 0 m_access_time_CPU_4 + 0 ns_sketch_CPU_4 + 0 ns_dpdk_CPU_4
   + 0 ns_CPU_4 + 0 m_access_time_CPU_5 + 0 ns_sketch_CPU_5
   + 0 ns_dpdk_CPU_5 + 0 ns_CPU_5 + 0 m_access_time_CPU_6
   + 0 ns_sketch_CPU_6 + 0 ns_dpdk_CPU_6 + 0 ns_CPU_6
   + 0 m_access_time_CPU_7 + 0 ns_sketch_CPU_7 + 0 ns_dpdk_CPU_7
   + 0 ns_CPU_7 + 0 m_access_time_CPU_8 + 0 ns_sketch_CPU_8
   + 0 ns_dpdk_CPU_8 + 0 ns_CPU_8 + 0 m_access_time_CPU_9
   + 0 ns_sketch_CPU_9 + 0 ns_dpdk_CPU_9 + 0 ns_CPU_9
   + 0 m_access_time_CPU_10 + 0 ns_sketch_CPU_10 + 0 ns_dpdk_CPU_10
   + 0 ns_CPU_10 + 0 m_access_time_CPU_11 + 0 ns_sketch_CPU_11
   + 0 ns_dpdk_CPU_11 + 0 ns_CPU_11 + 0 m_access_time_CPU_12
   + 0 ns_sketch_CPU_12 + 0 ns_dpdk_CPU_12 + 0 ns_CPU_12
   + 0 m_access_time_CPU_13 + 0 ns_sketch_CPU_13 + 0 ns_dpdk_CPU_13
   + 0 ns_CPU_13 + 0 m_access_time_CPU_14 + 0 ns_sketch_CPU_14
   + 0 ns_dpdk_CPU_14 + 0 ns_CPU_14 + 0 m_access_time_CPU_15
   + 0 ns_sketch_CPU_15 + 0 ns_dpdk_CPU_15 + 0 ns_CPU_15
   + 0 m_access_time_CPU_16 + 0 ns_sketch_CPU_16 + 0 ns_dpdk_CPU_16
   + 0 ns_CPU_16 + 0 m_access_time_CPU_17 + 0 ns_sketch_CPU_17
   + 0 ns_dpdk_CPU_17 + 0 ns_CPU_17 + 0 m_access_time_CPU_18
   + 0 ns_sketch_CPU_18 + 0 ns_dpdk_CPU_18 + 0 ns_CPU_18
   + 0 m_access_time_CPU_19 + 0 ns_sketch_CPU_19 + 0 ns_dpdk_CPU_19
   + 0 ns_CPU_19 + ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0
   res
Subject To
 cov_0_0: frac[20,0] + frac[0,0] + frac[1,0] >= 1
 cov_1_1: frac[2,1] + frac[1,1] + frac[20,1] >= 1
 cov_2_2: frac[2,2] + frac[3,2] + frac[20,2] >= 1
 cov_3_3: frac[20,3] + frac[3,3] + frac[4,3] >= 1
 cov_4_4: frac[4,4] + frac[20,4] + frac[5,4] >= 1
 cov_5_5: frac[5,5] + frac[20,5] + frac[6,5] >= 1
 cov_6_6: frac[6,6] + frac[7,6] + frac[20,6] >= 1
 cov_7_7: frac[7,7] + frac[20,7] + frac[8,7] >= 1
 cov_8_8: frac[9,8] + frac[20,8] + frac[8,8] >= 1
 cov_9_9: frac[20,9] + frac[10,9] + frac[9,9] >= 1
 cov_10_10: frac[11,10] + frac[20,10] + frac[10,10] >= 1
 cov_11_11: frac[11,11] + frac[12,11] + frac[20,11] >= 1
 cov_12_12: frac[12,12] + frac[20,12] + frac[13,12] >= 1
 cov_13_13: frac[13,13] + frac[14,13] + frac[20,13] >= 1
 cov_14_14: frac[14,14] + frac[15,14] + frac[20,14] >= 1
 cov_15_15: frac[16,15] + frac[15,15] + frac[20,15] >= 1
 cov_16_16: frac[16,16] + frac[17,16] + frac[20,16] >= 1
 cov_17_17: frac[20,17] + frac[18,17] + frac[17,17] >= 1
 cov_18_18: frac[19,18] + frac[20,18] + frac[18,18] >= 1
 cov_19_19: frac[19,19] + frac[20,19] + frac[0,19] >= 1
 accuracy_16_16: - 5.09765625 frac[16,16] + mem[16,16] = 0
 accuracy_11_11: - 5.09765625 frac[11,11] + mem[11,11] = 0
 accuracy_20_3: - 5.09765625 frac[20,3] + mem[20,3] = 0
 accuracy_6_6: - 5.09765625 frac[6,6] + mem[6,6] = 0
 accuracy_13_13: - 5.09765625 frac[13,13] + mem[13,13] = 0
 accuracy_19_19: - 5.09765625 frac[19,19] + mem[19,19] = 0
 accuracy_9_8: - 5.09765625 frac[9,8] + mem[9,8] = 0
 accuracy_12_12: - 5.09765625 frac[12,12] + mem[12,12] = 0
 accuracy_7_7: - 5.09765625 frac[7,7] + mem[7,7] = 0
 accuracy_2_1: - 5.09765625 frac[2,1] + mem[2,1] = 0
 accuracy_14_13: - 5.09765625 frac[14,13] + mem[14,13] = 0
 accuracy_20_7: - 5.09765625 frac[20,7] + mem[20,7] = 0
 accuracy_20_8: - 5.09765625 frac[20,8] + mem[20,8] = 0
 accuracy_17_16: - 5.09765625 frac[17,16] + mem[17,16] = 0
 accuracy_20_17: - 5.09765625 frac[20,17] + mem[20,17] = 0
 accuracy_11_10: - 5.09765625 frac[11,10] + mem[11,10] = 0
 accuracy_20_12: - 5.09765625 frac[20,12] + mem[20,12] = 0
 accuracy_13_12: - 5.09765625 frac[13,12] + mem[13,12] = 0
 accuracy_3_3: - 5.09765625 frac[3,3] + mem[3,3] = 0
 accuracy_5_5: - 5.09765625 frac[5,5] + mem[5,5] = 0
 accuracy_19_18: - 5.09765625 frac[19,18] + mem[19,18] = 0
 accuracy_7_6: - 5.09765625 frac[7,6] + mem[7,6] = 0
 accuracy_4_4: - 5.09765625 frac[4,4] + mem[4,4] = 0
 accuracy_20_0: - 5.09765625 frac[20,0] + mem[20,0] = 0
 accuracy_20_9: - 5.09765625 frac[20,9] + mem[20,9] = 0
 accuracy_16_15: - 5.09765625 frac[16,15] + mem[16,15] = 0
 accuracy_2_2: - 5.09765625 frac[2,2] + mem[2,2] = 0
 accuracy_14_14: - 5.09765625 frac[14,14] + mem[14,14] = 0
 accuracy_20_18: - 5.09765625 frac[20,18] + mem[20,18] = 0
 accuracy_20_4: - 5.09765625 frac[20,4] + mem[20,4] = 0
 accuracy_1_1: - 5.09765625 frac[1,1] + mem[1,1] = 0
 accuracy_10_9: - 5.09765625 frac[10,9] + mem[10,9] = 0
 accuracy_15_15: - 5.09765625 frac[15,15] + mem[15,15] = 0
 accuracy_20_13: - 5.09765625 frac[20,13] + mem[20,13] = 0
 accuracy_3_2: - 5.09765625 frac[3,2] + mem[3,2] = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 accuracy_5_4: - 5.09765625 frac[5,4] + mem[5,4] = 0
 accuracy_20_1: - 5.09765625 frac[20,1] + mem[20,1] = 0
 accuracy_18_17: - 5.09765625 frac[18,17] + mem[18,17] = 0
 accuracy_20_10: - 5.09765625 frac[20,10] + mem[20,10] = 0
 accuracy_8_7: - 5.09765625 frac[8,7] + mem[8,7] = 0
 accuracy_12_11: - 5.09765625 frac[12,11] + mem[12,11] = 0
 accuracy_20_5: - 5.09765625 frac[20,5] + mem[20,5] = 0
 accuracy_20_19: - 5.09765625 frac[20,19] + mem[20,19] = 0
 accuracy_1_0: - 5.09765625 frac[1,0] + mem[1,0] = 0
 accuracy_15_14: - 5.09765625 frac[15,14] + mem[15,14] = 0
 accuracy_20_14: - 5.09765625 frac[20,14] + mem[20,14] = 0
 accuracy_6_5: - 5.09765625 frac[6,5] + mem[6,5] = 0
 accuracy_10_10: - 5.09765625 frac[10,10] + mem[10,10] = 0
 accuracy_20_2: - 5.09765625 frac[20,2] + mem[20,2] = 0
 accuracy_0_19: - 5.09765625 frac[0,19] + mem[0,19] = 0
 accuracy_9_9: - 5.09765625 frac[9,9] + mem[9,9] = 0
 accuracy_20_11: - 5.09765625 frac[20,11] + mem[20,11] = 0
 accuracy_8_8: - 5.09765625 frac[8,8] + mem[8,8] = 0
 accuracy_20_6: - 5.09765625 frac[20,6] + mem[20,6] = 0
 accuracy_4_3: - 5.09765625 frac[4,3] + mem[4,3] = 0
 accuracy_18_18: - 5.09765625 frac[18,18] + mem[18,18] = 0
 accuracy_20_15: - 5.09765625 frac[20,15] + mem[20,15] = 0
 accuracy_17_17: - 5.09765625 frac[17,17] + mem[17,17] = 0
 accuracy_20_16: - 5.09765625 frac[20,16] + mem[20,16] = 0
 mem_tot_CPU_0: - mem[0,0] - mem[0,19] + mem_tot_CPU_0 = 0
 rows_tot_CPU_0: - 3 frac[0,0] - 3 frac[0,19] + rows_tot_CPU_0 = 0
 mem_tot_CPU_1: - mem[1,1] - mem[1,0] + mem_tot_CPU_1 = 0
 rows_tot_CPU_1: - 3 frac[1,1] - 3 frac[1,0] + rows_tot_CPU_1 = 0
 mem_tot_CPU_2: - mem[2,1] - mem[2,2] + mem_tot_CPU_2 = 0
 rows_tot_CPU_2: - 3 frac[2,1] - 3 frac[2,2] + rows_tot_CPU_2 = 0
 mem_tot_CPU_3: - mem[3,3] - mem[3,2] + mem_tot_CPU_3 = 0
 rows_tot_CPU_3: - 3 frac[3,3] - 3 frac[3,2] + rows_tot_CPU_3 = 0
 mem_tot_CPU_4: - mem[4,4] - mem[4,3] + mem_tot_CPU_4 = 0
 rows_tot_CPU_4: - 3 frac[4,4] - 3 frac[4,3] + rows_tot_CPU_4 = 0
 mem_tot_CPU_5: - mem[5,5] - mem[5,4] + mem_tot_CPU_5 = 0
 rows_tot_CPU_5: - 3 frac[5,5] - 3 frac[5,4] + rows_tot_CPU_5 = 0
 mem_tot_CPU_6: - mem[6,6] - mem[6,5] + mem_tot_CPU_6 = 0
 rows_tot_CPU_6: - 3 frac[6,6] - 3 frac[6,5] + rows_tot_CPU_6 = 0
 mem_tot_CPU_7: - mem[7,7] - mem[7,6] + mem_tot_CPU_7 = 0
 rows_tot_CPU_7: - 3 frac[7,7] - 3 frac[7,6] + rows_tot_CPU_7 = 0
 mem_tot_CPU_8: - mem[8,7] - mem[8,8] + mem_tot_CPU_8 = 0
 rows_tot_CPU_8: - 3 frac[8,7] - 3 frac[8,8] + rows_tot_CPU_8 = 0
 mem_tot_CPU_9: - mem[9,8] - mem[9,9] + mem_tot_CPU_9 = 0
 rows_tot_CPU_9: - 3 frac[9,8] - 3 frac[9,9] + rows_tot_CPU_9 = 0
 mem_tot_CPU_10: - mem[10,9] - mem[10,10] + mem_tot_CPU_10 = 0
 rows_tot_CPU_10: - 3 frac[10,9] - 3 frac[10,10] + rows_tot_CPU_10 = 0
 mem_tot_CPU_11: - mem[11,11] - mem[11,10] + mem_tot_CPU_11 = 0
 rows_tot_CPU_11: - 3 frac[11,11] - 3 frac[11,10] + rows_tot_CPU_11 = 0
 mem_tot_CPU_12: - mem[12,12] - mem[12,11] + mem_tot_CPU_12 = 0
 rows_tot_CPU_12: - 3 frac[12,12] - 3 frac[12,11] + rows_tot_CPU_12 = 0
 mem_tot_CPU_13: - mem[13,13] - mem[13,12] + mem_tot_CPU_13 = 0
 rows_tot_CPU_13: - 3 frac[13,13] - 3 frac[13,12] + rows_tot_CPU_13 = 0
 mem_tot_CPU_14: - mem[14,13] - mem[14,14] + mem_tot_CPU_14 = 0
 rows_tot_CPU_14: - 3 frac[14,13] - 3 frac[14,14] + rows_tot_CPU_14 = 0
 mem_tot_CPU_15: - mem[15,15] - mem[15,14] + mem_tot_CPU_15 = 0
 rows_tot_CPU_15: - 3 frac[15,15] - 3 frac[15,14] + rows_tot_CPU_15 = 0
 mem_tot_CPU_16: - mem[16,16] - mem[16,15] + mem_tot_CPU_16 = 0
 rows_tot_CPU_16: - 3 frac[16,16] - 3 frac[16,15] + rows_tot_CPU_16 = 0
 mem_tot_CPU_17: - mem[17,16] - mem[17,17] + mem_tot_CPU_17 = 0
 rows_tot_CPU_17: - 3 frac[17,16] - 3 frac[17,17] + rows_tot_CPU_17 = 0
 mem_tot_CPU_18: - mem[18,17] - mem[18,18] + mem_tot_CPU_18 = 0
 rows_tot_CPU_18: - 3 frac[18,17] - 3 frac[18,18] + rows_tot_CPU_18 = 0
 mem_tot_CPU_19: - mem[19,19] - mem[19,18] + mem_tot_CPU_19 = 0
 rows_tot_CPU_19: - 3 frac[19,19] - 3 frac[19,18] + rows_tot_CPU_19 = 0
 mem_tot_P4_1: - mem[20,3] - mem[20,7] - mem[20,8] - mem[20,17]
   - mem[20,12] - mem[20,0] - mem[20,9] - mem[20,18] - mem[20,4]
   - mem[20,13] - mem[20,1] - mem[20,10] - mem[20,5] - mem[20,19]
   - mem[20,14] - mem[20,2] - mem[20,11] - mem[20,6] - mem[20,15]
   - mem[20,16] + mem_tot_P4_1 = 0
 rows_tot_P4_1: - 3 frac[20,3] - 3 frac[20,7] - 3 frac[20,8]
   - 3 frac[20,17] - 3 frac[20,12] - 3 frac[20,0] - 3 frac[20,9]
   - 3 frac[20,18] - 3 frac[20,4] - 3 frac[20,13] - 3 frac[20,1]
   - 3 frac[20,10] - 3 frac[20,5] - 3 frac[20,19] - 3 frac[20,14]
   - 3 frac[20,2] - 3 frac[20,11] - 3 frac[20,6] - 3 frac[20,15]
   - 3 frac[20,16] + rows_tot_P4_1 = 0
 ns_single_CPU_0: 3.5 rows_tot_CPU_0 - ns_single_CPU_0 + pdt_m_rows_CPU_0
   = 0
 capacity_cores_CPU_0: cores_sketch_CPU_0 + cores_dpdk_CPU_0 <= 7
 ns_sketch_CPU_0: - ns_single_CPU_0 + pdt_nsk_c_CPU_0 = 0
 ns_dpdk_CPU_0: 7.142857142857143 cores_dpdk_CPU_0 - pdt_nsc_dpdk_CPU_0
   = -21.42857142857143
 ns_single_CPU_1: 3.5 rows_tot_CPU_1 - ns_single_CPU_1 + pdt_m_rows_CPU_1
   = 0
 capacity_cores_CPU_1: cores_sketch_CPU_1 + cores_dpdk_CPU_1 <= 7
 ns_sketch_CPU_1: - ns_single_CPU_1 + pdt_nsk_c_CPU_1 = 0
 ns_dpdk_CPU_1: 7.142857142857143 cores_dpdk_CPU_1 - pdt_nsc_dpdk_CPU_1
   = -21.42857142857143
 ns_single_CPU_2: 3.5 rows_tot_CPU_2 - ns_single_CPU_2 + pdt_m_rows_CPU_2
   = 0
 capacity_cores_CPU_2: cores_sketch_CPU_2 + cores_dpdk_CPU_2 <= 7
 ns_sketch_CPU_2: - ns_single_CPU_2 + pdt_nsk_c_CPU_2 = 0
 ns_dpdk_CPU_2: 7.142857142857143 cores_dpdk_CPU_2 - pdt_nsc_dpdk_CPU_2
   = -21.42857142857143
 ns_single_CPU_3: 3.5 rows_tot_CPU_3 - ns_single_CPU_3 + pdt_m_rows_CPU_3
   = 0
 capacity_cores_CPU_3: cores_sketch_CPU_3 + cores_dpdk_CPU_3 <= 7
 ns_sketch_CPU_3: - ns_single_CPU_3 + pdt_nsk_c_CPU_3 = 0
 ns_dpdk_CPU_3: 7.142857142857143 cores_dpdk_CPU_3 - pdt_nsc_dpdk_CPU_3
   = -21.42857142857143
 ns_single_CPU_4: 3.5 rows_tot_CPU_4 - ns_single_CPU_4 + pdt_m_rows_CPU_4
   = 0
 capacity_cores_CPU_4: cores_sketch_CPU_4 + cores_dpdk_CPU_4 <= 7
 ns_sketch_CPU_4: - ns_single_CPU_4 + pdt_nsk_c_CPU_4 = 0
 ns_dpdk_CPU_4: 7.142857142857143 cores_dpdk_CPU_4 - pdt_nsc_dpdk_CPU_4
   = -21.42857142857143
 ns_single_CPU_5: 3.5 rows_tot_CPU_5 - ns_single_CPU_5 + pdt_m_rows_CPU_5
   = 0
 capacity_cores_CPU_5: cores_sketch_CPU_5 + cores_dpdk_CPU_5 <= 7
 ns_sketch_CPU_5: - ns_single_CPU_5 + pdt_nsk_c_CPU_5 = 0
 ns_dpdk_CPU_5: 7.142857142857143 cores_dpdk_CPU_5 - pdt_nsc_dpdk_CPU_5
   = -21.42857142857143
 ns_single_CPU_6: 3.5 rows_tot_CPU_6 - ns_single_CPU_6 + pdt_m_rows_CPU_6
   = 0
 capacity_cores_CPU_6: cores_sketch_CPU_6 + cores_dpdk_CPU_6 <= 7
 ns_sketch_CPU_6: - ns_single_CPU_6 + pdt_nsk_c_CPU_6 = 0
 ns_dpdk_CPU_6: 7.142857142857143 cores_dpdk_CPU_6 - pdt_nsc_dpdk_CPU_6
   = -21.42857142857143
 ns_single_CPU_7: 3.5 rows_tot_CPU_7 - ns_single_CPU_7 + pdt_m_rows_CPU_7
   = 0
 capacity_cores_CPU_7: cores_sketch_CPU_7 + cores_dpdk_CPU_7 <= 7
 ns_sketch_CPU_7: - ns_single_CPU_7 + pdt_nsk_c_CPU_7 = 0
 ns_dpdk_CPU_7: 7.142857142857143 cores_dpdk_CPU_7 - pdt_nsc_dpdk_CPU_7
   = -21.42857142857143
 ns_single_CPU_8: 3.5 rows_tot_CPU_8 - ns_single_CPU_8 + pdt_m_rows_CPU_8
   = 0
 capacity_cores_CPU_8: cores_sketch_CPU_8 + cores_dpdk_CPU_8 <= 7
 ns_sketch_CPU_8: - ns_single_CPU_8 + pdt_nsk_c_CPU_8 = 0
 ns_dpdk_CPU_8: 7.142857142857143 cores_dpdk_CPU_8 - pdt_nsc_dpdk_CPU_8
   = -21.42857142857143
 ns_single_CPU_9: 3.5 rows_tot_CPU_9 - ns_single_CPU_9 + pdt_m_rows_CPU_9
   = 0
 capacity_cores_CPU_9: cores_sketch_CPU_9 + cores_dpdk_CPU_9 <= 7
 ns_sketch_CPU_9: - ns_single_CPU_9 + pdt_nsk_c_CPU_9 = 0
 ns_dpdk_CPU_9: 7.142857142857143 cores_dpdk_CPU_9 - pdt_nsc_dpdk_CPU_9
   = -21.42857142857143
 ns_single_CPU_10: 3.5 rows_tot_CPU_10 - ns_single_CPU_10
   + pdt_m_rows_CPU_10 = 0
 capacity_cores_CPU_10: cores_sketch_CPU_10 + cores_dpdk_CPU_10 <= 7
 ns_sketch_CPU_10: - ns_single_CPU_10 + pdt_nsk_c_CPU_10 = 0
 ns_dpdk_CPU_10: 7.142857142857143 cores_dpdk_CPU_10 - pdt_nsc_dpdk_CPU_10
   = -21.42857142857143
 ns_single_CPU_11: 3.5 rows_tot_CPU_11 - ns_single_CPU_11
   + pdt_m_rows_CPU_11 = 0
 capacity_cores_CPU_11: cores_sketch_CPU_11 + cores_dpdk_CPU_11 <= 7
 ns_sketch_CPU_11: - ns_single_CPU_11 + pdt_nsk_c_CPU_11 = 0
 ns_dpdk_CPU_11: 7.142857142857143 cores_dpdk_CPU_11 - pdt_nsc_dpdk_CPU_11
   = -21.42857142857143
 ns_single_CPU_12: 3.5 rows_tot_CPU_12 - ns_single_CPU_12
   + pdt_m_rows_CPU_12 = 0
 capacity_cores_CPU_12: cores_sketch_CPU_12 + cores_dpdk_CPU_12 <= 7
 ns_sketch_CPU_12: - ns_single_CPU_12 + pdt_nsk_c_CPU_12 = 0
 ns_dpdk_CPU_12: 7.142857142857143 cores_dpdk_CPU_12 - pdt_nsc_dpdk_CPU_12
   = -21.42857142857143
 ns_single_CPU_13: 3.5 rows_tot_CPU_13 - ns_single_CPU_13
   + pdt_m_rows_CPU_13 = 0
 capacity_cores_CPU_13: cores_sketch_CPU_13 + cores_dpdk_CPU_13 <= 7
 ns_sketch_CPU_13: - ns_single_CPU_13 + pdt_nsk_c_CPU_13 = 0
 ns_dpdk_CPU_13: 7.142857142857143 cores_dpdk_CPU_13 - pdt_nsc_dpdk_CPU_13
   = -21.42857142857143
 ns_single_CPU_14: 3.5 rows_tot_CPU_14 - ns_single_CPU_14
   + pdt_m_rows_CPU_14 = 0
 capacity_cores_CPU_14: cores_sketch_CPU_14 + cores_dpdk_CPU_14 <= 7
 ns_sketch_CPU_14: - ns_single_CPU_14 + pdt_nsk_c_CPU_14 = 0
 ns_dpdk_CPU_14: 7.142857142857143 cores_dpdk_CPU_14 - pdt_nsc_dpdk_CPU_14
   = -21.42857142857143
 ns_single_CPU_15: 3.5 rows_tot_CPU_15 - ns_single_CPU_15
   + pdt_m_rows_CPU_15 = 0
 capacity_cores_CPU_15: cores_sketch_CPU_15 + cores_dpdk_CPU_15 <= 7
 ns_sketch_CPU_15: - ns_single_CPU_15 + pdt_nsk_c_CPU_15 = 0
 ns_dpdk_CPU_15: 7.142857142857143 cores_dpdk_CPU_15 - pdt_nsc_dpdk_CPU_15
   = -21.42857142857143
 ns_single_CPU_16: 3.5 rows_tot_CPU_16 - ns_single_CPU_16
   + pdt_m_rows_CPU_16 = 0
 capacity_cores_CPU_16: cores_sketch_CPU_16 + cores_dpdk_CPU_16 <= 7
 ns_sketch_CPU_16: - ns_single_CPU_16 + pdt_nsk_c_CPU_16 = 0
 ns_dpdk_CPU_16: 7.142857142857143 cores_dpdk_CPU_16 - pdt_nsc_dpdk_CPU_16
   = -21.42857142857143
 ns_single_CPU_17: 3.5 rows_tot_CPU_17 - ns_single_CPU_17
   + pdt_m_rows_CPU_17 = 0
 capacity_cores_CPU_17: cores_sketch_CPU_17 + cores_dpdk_CPU_17 <= 7
 ns_sketch_CPU_17: - ns_single_CPU_17 + pdt_nsk_c_CPU_17 = 0
 ns_dpdk_CPU_17: 7.142857142857143 cores_dpdk_CPU_17 - pdt_nsc_dpdk_CPU_17
   = -21.42857142857143
 ns_single_CPU_18: 3.5 rows_tot_CPU_18 - ns_single_CPU_18
   + pdt_m_rows_CPU_18 = 0
 capacity_cores_CPU_18: cores_sketch_CPU_18 + cores_dpdk_CPU_18 <= 7
 ns_sketch_CPU_18: - ns_single_CPU_18 + pdt_nsk_c_CPU_18 = 0
 ns_dpdk_CPU_18: 7.142857142857143 cores_dpdk_CPU_18 - pdt_nsc_dpdk_CPU_18
   = -21.42857142857143
 ns_single_CPU_19: 3.5 rows_tot_CPU_19 - ns_single_CPU_19
   + pdt_m_rows_CPU_19 = 0
 capacity_cores_CPU_19: cores_sketch_CPU_19 + cores_dpdk_CPU_19 <= 7
 ns_sketch_CPU_19: - ns_single_CPU_19 + pdt_nsk_c_CPU_19 = 0
 ns_dpdk_CPU_19: 7.142857142857143 cores_dpdk_CPU_19 - pdt_nsc_dpdk_CPU_19
   = -21.42857142857143
 ns_P4_1: ns_P4_1 = 6.756756756756757
 res_acc: 1.220703125e-04 mem_tot_CPU_0 + 1.220703125e-04 mem_tot_CPU_1
   + 1.220703125e-04 mem_tot_CPU_2 + 1.220703125e-04 mem_tot_CPU_3
   + 1.220703125e-04 mem_tot_CPU_4 + 1.220703125e-04 mem_tot_CPU_5
   + 1.220703125e-04 mem_tot_CPU_6 + 1.220703125e-04 mem_tot_CPU_7
   + 1.220703125e-04 mem_tot_CPU_8 + 1.220703125e-04 mem_tot_CPU_9
   + 1.220703125e-04 mem_tot_CPU_10 + 1.220703125e-04 mem_tot_CPU_11
   + 1.220703125e-04 mem_tot_CPU_12 + 1.220703125e-04 mem_tot_CPU_13
   + 1.220703125e-04 mem_tot_CPU_14 + 1.220703125e-04 mem_tot_CPU_15
   + 1.220703125e-04 mem_tot_CPU_16 + 1.220703125e-04 mem_tot_CPU_17
   + 1.220703125e-04 mem_tot_CPU_18 + 1.220703125e-04 mem_tot_CPU_19
   + 0.0208333333333333 mem_tot_P4_1 + 0.25 rows_tot_P4_1
   + 10 cores_sketch_CPU_0 + 10 cores_dpdk_CPU_0 + 10 cores_sketch_CPU_1
   + 10 cores_dpdk_CPU_1 + 10 cores_sketch_CPU_2 + 10 cores_dpdk_CPU_2
   + 10 cores_sketch_CPU_3 + 10 cores_dpdk_CPU_3 + 10 cores_sketch_CPU_4
   + 10 cores_dpdk_CPU_4 + 10 cores_sketch_CPU_5 + 10 cores_dpdk_CPU_5
   + 10 cores_sketch_CPU_6 + 10 cores_dpdk_CPU_6 + 10 cores_sketch_CPU_7
   + 10 cores_dpdk_CPU_7 + 10 cores_sketch_CPU_8 + 10 cores_dpdk_CPU_8
   + 10 cores_sketch_CPU_9 + 10 cores_dpdk_CPU_9 + 10 cores_sketch_CPU_10
   + 10 cores_dpdk_CPU_10 + 10 cores_sketch_CPU_11 + 10 cores_dpdk_CPU_11
   + 10 cores_sketch_CPU_12 + 10 cores_dpdk_CPU_12 + 10 cores_sketch_CPU_13
   + 10 cores_dpdk_CPU_13 + 10 cores_sketch_CPU_14 + 10 cores_dpdk_CPU_14
   + 10 cores_sketch_CPU_15 + 10 cores_dpdk_CPU_15 + 10 cores_sketch_CPU_16
   + 10 cores_dpdk_CPU_16 + 10 cores_sketch_CPU_17 + 10 cores_dpdk_CPU_17
   + 10 cores_sketch_CPU_18 + 10 cores_dpdk_CPU_18 + 10 cores_sketch_CPU_19
   + 10 cores_dpdk_CPU_19 - res = 0
 pdt_m_rows_CPU_0: pdt_m_rows_CPU_0 + [
   - rows_tot_CPU_0 * m_access_time_CPU_0 ] = 0
 pdt_nsk_c_CPU_0: pdt_nsk_c_CPU_0 + [
   - cores_sketch_CPU_0 * ns_sketch_CPU_0 ] = 0
 pdt_nsc_dpdk_CPU_0: pdt_nsc_dpdk_CPU_0 + [
   - cores_dpdk_CPU_0 * ns_dpdk_CPU_0 ] = 0
 pdt_m_rows_CPU_1: pdt_m_rows_CPU_1 + [
   - rows_tot_CPU_1 * m_access_time_CPU_1 ] = 0
 pdt_nsk_c_CPU_1: pdt_nsk_c_CPU_1 + [
   - cores_sketch_CPU_1 * ns_sketch_CPU_1 ] = 0
 pdt_nsc_dpdk_CPU_1: pdt_nsc_dpdk_CPU_1 + [
   - cores_dpdk_CPU_1 * ns_dpdk_CPU_1 ] = 0
 pdt_m_rows_CPU_2: pdt_m_rows_CPU_2 + [
   - rows_tot_CPU_2 * m_access_time_CPU_2 ] = 0
 pdt_nsk_c_CPU_2: pdt_nsk_c_CPU_2 + [
   - cores_sketch_CPU_2 * ns_sketch_CPU_2 ] = 0
 pdt_nsc_dpdk_CPU_2: pdt_nsc_dpdk_CPU_2 + [
   - cores_dpdk_CPU_2 * ns_dpdk_CPU_2 ] = 0
 pdt_m_rows_CPU_3: pdt_m_rows_CPU_3 + [
   - rows_tot_CPU_3 * m_access_time_CPU_3 ] = 0
 pdt_nsk_c_CPU_3: pdt_nsk_c_CPU_3 + [
   - cores_sketch_CPU_3 * ns_sketch_CPU_3 ] = 0
 pdt_nsc_dpdk_CPU_3: pdt_nsc_dpdk_CPU_3 + [
   - cores_dpdk_CPU_3 * ns_dpdk_CPU_3 ] = 0
 pdt_m_rows_CPU_4: pdt_m_rows_CPU_4 + [
   - rows_tot_CPU_4 * m_access_time_CPU_4 ] = 0
 pdt_nsk_c_CPU_4: pdt_nsk_c_CPU_4 + [
   - cores_sketch_CPU_4 * ns_sketch_CPU_4 ] = 0
 pdt_nsc_dpdk_CPU_4: pdt_nsc_dpdk_CPU_4 + [
   - cores_dpdk_CPU_4 * ns_dpdk_CPU_4 ] = 0
 pdt_m_rows_CPU_5: pdt_m_rows_CPU_5 + [
   - rows_tot_CPU_5 * m_access_time_CPU_5 ] = 0
 pdt_nsk_c_CPU_5: pdt_nsk_c_CPU_5 + [
   - cores_sketch_CPU_5 * ns_sketch_CPU_5 ] = 0
 pdt_nsc_dpdk_CPU_5: pdt_nsc_dpdk_CPU_5 + [
   - cores_dpdk_CPU_5 * ns_dpdk_CPU_5 ] = 0
 pdt_m_rows_CPU_6: pdt_m_rows_CPU_6 + [
   - rows_tot_CPU_6 * m_access_time_CPU_6 ] = 0
 pdt_nsk_c_CPU_6: pdt_nsk_c_CPU_6 + [
   - cores_sketch_CPU_6 * ns_sketch_CPU_6 ] = 0
 pdt_nsc_dpdk_CPU_6: pdt_nsc_dpdk_CPU_6 + [
   - cores_dpdk_CPU_6 * ns_dpdk_CPU_6 ] = 0
 pdt_m_rows_CPU_7: pdt_m_rows_CPU_7 + [
   - rows_tot_CPU_7 * m_access_time_CPU_7 ] = 0
 pdt_nsk_c_CPU_7: pdt_nsk_c_CPU_7 + [
   - cores_sketch_CPU_7 * ns_sketch_CPU_7 ] = 0
 pdt_nsc_dpdk_CPU_7: pdt_nsc_dpdk_CPU_7 + [
   - cores_dpdk_CPU_7 * ns_dpdk_CPU_7 ] = 0
 pdt_m_rows_CPU_8: pdt_m_rows_CPU_8 + [
   - rows_tot_CPU_8 * m_access_time_CPU_8 ] = 0
 pdt_nsk_c_CPU_8: pdt_nsk_c_CPU_8 + [
   - cores_sketch_CPU_8 * ns_sketch_CPU_8 ] = 0
 pdt_nsc_dpdk_CPU_8: pdt_nsc_dpdk_CPU_8 + [
   - cores_dpdk_CPU_8 * ns_dpdk_CPU_8 ] = 0
 pdt_m_rows_CPU_9: pdt_m_rows_CPU_9 + [
   - rows_tot_CPU_9 * m_access_time_CPU_9 ] = 0
 pdt_nsk_c_CPU_9: pdt_nsk_c_CPU_9 + [
   - cores_sketch_CPU_9 * ns_sketch_CPU_9 ] = 0
 pdt_nsc_dpdk_CPU_9: pdt_nsc_dpdk_CPU_9 + [
   - cores_dpdk_CPU_9 * ns_dpdk_CPU_9 ] = 0
 pdt_m_rows_CPU_10: pdt_m_rows_CPU_10 + [
   - rows_tot_CPU_10 * m_access_time_CPU_10 ] = 0
 pdt_nsk_c_CPU_10: pdt_nsk_c_CPU_10 + [
   - cores_sketch_CPU_10 * ns_sketch_CPU_10 ] = 0
 pdt_nsc_dpdk_CPU_10: pdt_nsc_dpdk_CPU_10 + [
   - cores_dpdk_CPU_10 * ns_dpdk_CPU_10 ] = 0
 pdt_m_rows_CPU_11: pdt_m_rows_CPU_11 + [
   - rows_tot_CPU_11 * m_access_time_CPU_11 ] = 0
 pdt_nsk_c_CPU_11: pdt_nsk_c_CPU_11 + [
   - cores_sketch_CPU_11 * ns_sketch_CPU_11 ] = 0
 pdt_nsc_dpdk_CPU_11: pdt_nsc_dpdk_CPU_11 + [
   - cores_dpdk_CPU_11 * ns_dpdk_CPU_11 ] = 0
 pdt_m_rows_CPU_12: pdt_m_rows_CPU_12 + [
   - rows_tot_CPU_12 * m_access_time_CPU_12 ] = 0
 pdt_nsk_c_CPU_12: pdt_nsk_c_CPU_12 + [
   - cores_sketch_CPU_12 * ns_sketch_CPU_12 ] = 0
 pdt_nsc_dpdk_CPU_12: pdt_nsc_dpdk_CPU_12 + [
   - cores_dpdk_CPU_12 * ns_dpdk_CPU_12 ] = 0
 pdt_m_rows_CPU_13: pdt_m_rows_CPU_13 + [
   - rows_tot_CPU_13 * m_access_time_CPU_13 ] = 0
 pdt_nsk_c_CPU_13: pdt_nsk_c_CPU_13 + [
   - cores_sketch_CPU_13 * ns_sketch_CPU_13 ] = 0
 pdt_nsc_dpdk_CPU_13: pdt_nsc_dpdk_CPU_13 + [
   - cores_dpdk_CPU_13 * ns_dpdk_CPU_13 ] = 0
 pdt_m_rows_CPU_14: pdt_m_rows_CPU_14 + [
   - rows_tot_CPU_14 * m_access_time_CPU_14 ] = 0
 pdt_nsk_c_CPU_14: pdt_nsk_c_CPU_14 + [
   - cores_sketch_CPU_14 * ns_sketch_CPU_14 ] = 0
 pdt_nsc_dpdk_CPU_14: pdt_nsc_dpdk_CPU_14 + [
   - cores_dpdk_CPU_14 * ns_dpdk_CPU_14 ] = 0
 pdt_m_rows_CPU_15: pdt_m_rows_CPU_15 + [
   - rows_tot_CPU_15 * m_access_time_CPU_15 ] = 0
 pdt_nsk_c_CPU_15: pdt_nsk_c_CPU_15 + [
   - cores_sketch_CPU_15 * ns_sketch_CPU_15 ] = 0
 pdt_nsc_dpdk_CPU_15: pdt_nsc_dpdk_CPU_15 + [
   - cores_dpdk_CPU_15 * ns_dpdk_CPU_15 ] = 0
 pdt_m_rows_CPU_16: pdt_m_rows_CPU_16 + [
   - rows_tot_CPU_16 * m_access_time_CPU_16 ] = 0
 pdt_nsk_c_CPU_16: pdt_nsk_c_CPU_16 + [
   - cores_sketch_CPU_16 * ns_sketch_CPU_16 ] = 0
 pdt_nsc_dpdk_CPU_16: pdt_nsc_dpdk_CPU_16 + [
   - cores_dpdk_CPU_16 * ns_dpdk_CPU_16 ] = 0
 pdt_m_rows_CPU_17: pdt_m_rows_CPU_17 + [
   - rows_tot_CPU_17 * m_access_time_CPU_17 ] = 0
 pdt_nsk_c_CPU_17: pdt_nsk_c_CPU_17 + [
   - cores_sketch_CPU_17 * ns_sketch_CPU_17 ] = 0
 pdt_nsc_dpdk_CPU_17: pdt_nsc_dpdk_CPU_17 + [
   - cores_dpdk_CPU_17 * ns_dpdk_CPU_17 ] = 0
 pdt_m_rows_CPU_18: pdt_m_rows_CPU_18 + [
   - rows_tot_CPU_18 * m_access_time_CPU_18 ] = 0
 pdt_nsk_c_CPU_18: pdt_nsk_c_CPU_18 + [
   - cores_sketch_CPU_18 * ns_sketch_CPU_18 ] = 0
 pdt_nsc_dpdk_CPU_18: pdt_nsc_dpdk_CPU_18 + [
   - cores_dpdk_CPU_18 * ns_dpdk_CPU_18 ] = 0
 pdt_m_rows_CPU_19: pdt_m_rows_CPU_19 + [
   - rows_tot_CPU_19 * m_access_time_CPU_19 ] = 0
 pdt_nsk_c_CPU_19: pdt_nsk_c_CPU_19 + [
   - cores_sketch_CPU_19 * ns_sketch_CPU_19 ] = 0
 pdt_nsc_dpdk_CPU_19: pdt_nsc_dpdk_CPU_19 + [
   - cores_dpdk_CPU_19 * ns_dpdk_CPU_19 ] = 0
Bounds
 mem_tot_CPU_0 <= 32768
 mem_tot_CPU_1 <= 32768
 mem_tot_CPU_2 <= 32768
 mem_tot_CPU_3 <= 32768
 mem_tot_CPU_4 <= 32768
 mem_tot_CPU_5 <= 32768
 mem_tot_CPU_6 <= 32768
 mem_tot_CPU_7 <= 32768
 mem_tot_CPU_8 <= 32768
 mem_tot_CPU_9 <= 32768
 mem_tot_CPU_10 <= 32768
 mem_tot_CPU_11 <= 32768
 mem_tot_CPU_12 <= 32768
 mem_tot_CPU_13 <= 32768
 mem_tot_CPU_14 <= 32768
 mem_tot_CPU_15 <= 32768
 mem_tot_CPU_16 <= 32768
 mem_tot_CPU_17 <= 32768
 mem_tot_CPU_18 <= 32768
 mem_tot_CPU_19 <= 32768
 mem_tot_P4_1 <= 576
 cores_sketch_CPU_0 <= 7
 1 <= cores_dpdk_CPU_0 <= 7
 cores_sketch_CPU_1 <= 7
 1 <= cores_dpdk_CPU_1 <= 7
 cores_sketch_CPU_2 <= 7
 1 <= cores_dpdk_CPU_2 <= 7
 cores_sketch_CPU_3 <= 7
 1 <= cores_dpdk_CPU_3 <= 7
 cores_sketch_CPU_4 <= 7
 1 <= cores_dpdk_CPU_4 <= 7
 cores_sketch_CPU_5 <= 7
 1 <= cores_dpdk_CPU_5 <= 7
 cores_sketch_CPU_6 <= 7
 1 <= cores_dpdk_CPU_6 <= 7
 cores_sketch_CPU_7 <= 7
 1 <= cores_dpdk_CPU_7 <= 7
 cores_sketch_CPU_8 <= 7
 1 <= cores_dpdk_CPU_8 <= 7
 cores_sketch_CPU_9 <= 7
 1 <= cores_dpdk_CPU_9 <= 7
 cores_sketch_CPU_10 <= 7
 1 <= cores_dpdk_CPU_10 <= 7
 cores_sketch_CPU_11 <= 7
 1 <= cores_dpdk_CPU_11 <= 7
 cores_sketch_CPU_12 <= 7
 1 <= cores_dpdk_CPU_12 <= 7
 cores_sketch_CPU_13 <= 7
 1 <= cores_dpdk_CPU_13 <= 7
 cores_sketch_CPU_14 <= 7
 1 <= cores_dpdk_CPU_14 <= 7
 cores_sketch_CPU_15 <= 7
 1 <= cores_dpdk_CPU_15 <= 7
 cores_sketch_CPU_16 <= 7
 1 <= cores_dpdk_CPU_16 <= 7
 cores_sketch_CPU_17 <= 7
 1 <= cores_dpdk_CPU_17 <= 7
 cores_sketch_CPU_18 <= 7
 1 <= cores_dpdk_CPU_18 <= 7
 cores_sketch_CPU_19 <= 7
 1 <= cores_dpdk_CPU_19 <= 7
Binaries
 frac[16,16] frac[11,11] frac[20,3] frac[6,6] frac[13,13] frac[19,19]
 frac[9,8] frac[12,12] frac[7,7] frac[2,1] frac[14,13] frac[20,7]
 frac[20,8] frac[17,16] frac[20,17] frac[11,10] frac[20,12] frac[13,12]
 frac[3,3] frac[5,5] frac[19,18] frac[7,6] frac[4,4] frac[20,0] frac[20,9]
 frac[16,15] frac[2,2] frac[14,14] frac[20,18] frac[20,4] frac[1,1]
 frac[10,9] frac[15,15] frac[20,13] frac[3,2] frac[0,0] frac[5,4]
 frac[20,1] frac[18,17] frac[20,10] frac[8,7] frac[12,11] frac[20,5]
 frac[20,19] frac[1,0] frac[15,14] frac[20,14] frac[6,5] frac[10,10]
 frac[20,2] frac[0,19] frac[9,9] frac[20,11] frac[8,8] frac[20,6] frac[4,3]
 frac[18,18] frac[20,15] frac[17,17] frac[20,16]
Generals
 cores_sketch_CPU_0 cores_dpdk_CPU_0 cores_sketch_CPU_1 cores_dpdk_CPU_1
 cores_sketch_CPU_2 cores_dpdk_CPU_2 cores_sketch_CPU_3 cores_dpdk_CPU_3
 cores_sketch_CPU_4 cores_dpdk_CPU_4 cores_sketch_CPU_5 cores_dpdk_CPU_5
 cores_sketch_CPU_6 cores_dpdk_CPU_6 cores_sketch_CPU_7 cores_dpdk_CPU_7
 cores_sketch_CPU_8 cores_dpdk_CPU_8 cores_sketch_CPU_9 cores_dpdk_CPU_9
 cores_sketch_CPU_10 cores_dpdk_CPU_10 cores_sketch_CPU_11
 cores_dpdk_CPU_11 cores_sketch_CPU_12 cores_dpdk_CPU_12
 cores_sketch_CPU_13 cores_dpdk_CPU_13 cores_sketch_CPU_14
 cores_dpdk_CPU_14 cores_sketch_CPU_15 cores_dpdk_CPU_15
 cores_sketch_CPU_16 cores_dpdk_CPU_16 cores_sketch_CPU_17
 cores_dpdk_CPU_17 cores_sketch_CPU_18 cores_dpdk_CPU_18
 cores_sketch_CPU_19 cores_dpdk_CPU_19
General Constraints
 mem_access_time_CPU_0: m_access_time_CPU_0 = PWL ( mem_tot_CPU_0 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_0: ns_CPU_0 = MAX ( ns_dpdk_CPU_0 , ns_sketch_CPU_0 )
 mem_access_time_CPU_1: m_access_time_CPU_1 = PWL ( mem_tot_CPU_1 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_1: ns_CPU_1 = MAX ( ns_dpdk_CPU_1 , ns_sketch_CPU_1 )
 mem_access_time_CPU_2: m_access_time_CPU_2 = PWL ( mem_tot_CPU_2 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_2: ns_CPU_2 = MAX ( ns_dpdk_CPU_2 , ns_sketch_CPU_2 )
 mem_access_time_CPU_3: m_access_time_CPU_3 = PWL ( mem_tot_CPU_3 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_3: ns_CPU_3 = MAX ( ns_dpdk_CPU_3 , ns_sketch_CPU_3 )
 mem_access_time_CPU_4: m_access_time_CPU_4 = PWL ( mem_tot_CPU_4 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_4: ns_CPU_4 = MAX ( ns_dpdk_CPU_4 , ns_sketch_CPU_4 )
 mem_access_time_CPU_5: m_access_time_CPU_5 = PWL ( mem_tot_CPU_5 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_5: ns_CPU_5 = MAX ( ns_dpdk_CPU_5 , ns_sketch_CPU_5 )
 mem_access_time_CPU_6: m_access_time_CPU_6 = PWL ( mem_tot_CPU_6 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_6: ns_CPU_6 = MAX ( ns_dpdk_CPU_6 , ns_sketch_CPU_6 )
 mem_access_time_CPU_7: m_access_time_CPU_7 = PWL ( mem_tot_CPU_7 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_7: ns_CPU_7 = MAX ( ns_dpdk_CPU_7 , ns_sketch_CPU_7 )
 mem_access_time_CPU_8: m_access_time_CPU_8 = PWL ( mem_tot_CPU_8 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_8: ns_CPU_8 = MAX ( ns_dpdk_CPU_8 , ns_sketch_CPU_8 )
 mem_access_time_CPU_9: m_access_time_CPU_9 = PWL ( mem_tot_CPU_9 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_9: ns_CPU_9 = MAX ( ns_dpdk_CPU_9 , ns_sketch_CPU_9 )
 mem_access_time_CPU_10: m_access_time_CPU_10 = PWL ( mem_tot_CPU_10 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_10: ns_CPU_10 = MAX ( ns_dpdk_CPU_10 , ns_sketch_CPU_10 )
 mem_access_time_CPU_11: m_access_time_CPU_11 = PWL ( mem_tot_CPU_11 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_11: ns_CPU_11 = MAX ( ns_dpdk_CPU_11 , ns_sketch_CPU_11 )
 mem_access_time_CPU_12: m_access_time_CPU_12 = PWL ( mem_tot_CPU_12 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_12: ns_CPU_12 = MAX ( ns_dpdk_CPU_12 , ns_sketch_CPU_12 )
 mem_access_time_CPU_13: m_access_time_CPU_13 = PWL ( mem_tot_CPU_13 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_13: ns_CPU_13 = MAX ( ns_dpdk_CPU_13 , ns_sketch_CPU_13 )
 mem_access_time_CPU_14: m_access_time_CPU_14 = PWL ( mem_tot_CPU_14 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_14: ns_CPU_14 = MAX ( ns_dpdk_CPU_14 , ns_sketch_CPU_14 )
 mem_access_time_CPU_15: m_access_time_CPU_15 = PWL ( mem_tot_CPU_15 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_15: ns_CPU_15 = MAX ( ns_dpdk_CPU_15 , ns_sketch_CPU_15 )
 mem_access_time_CPU_16: m_access_time_CPU_16 = PWL ( mem_tot_CPU_16 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_16: ns_CPU_16 = MAX ( ns_dpdk_CPU_16 , ns_sketch_CPU_16 )
 mem_access_time_CPU_17: m_access_time_CPU_17 = PWL ( mem_tot_CPU_17 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_17: ns_CPU_17 = MAX ( ns_dpdk_CPU_17 , ns_sketch_CPU_17 )
 mem_access_time_CPU_18: m_access_time_CPU_18 = PWL ( mem_tot_CPU_18 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_18: ns_CPU_18 = MAX ( ns_dpdk_CPU_18 , ns_sketch_CPU_18 )
 mem_access_time_CPU_19: m_access_time_CPU_19 = PWL ( mem_tot_CPU_19 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_19: ns_CPU_19 = MAX ( ns_dpdk_CPU_19 , ns_sketch_CPU_19 )
 ns_overall: ns = MAX ( ns_CPU_0 , ns_CPU_1 , ns_CPU_2 , ns_CPU_3 ,
   ns_CPU_4 , ns_CPU_5 , ns_CPU_6 , ns_CPU_7 , ns_CPU_8 , ns_CPU_9 ,
   ns_CPU_10 , ns_CPU_11 , ns_CPU_12 , ns_CPU_13 , ns_CPU_14 , ns_CPU_15 ,
   ns_CPU_16 , ns_CPU_17 , ns_CPU_18 , ns_CPU_19 , ns_P4_1 )
End

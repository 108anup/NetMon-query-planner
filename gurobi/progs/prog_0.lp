\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_others + 0 max_rows_others + 0 m_access_time_CPU7865
   + 0 m_access_time_CPU3360 + 0 m_access_time_CPU6779
   + 0 m_access_time_CPU9205 + 0 m_access_time_CPU5603
   + 0 m_access_time_CPU3189 + 0 m_access_time_CPU1861
   + 0 m_access_time_CPU2730 + 0 m_access_time_CPU7358
   + 0 m_access_time_CPU6183 + 0 m_access_time_CPU6929
   + 0 m_access_time_CPU9110 + 0 m_access_time_CPU6689
   + 0 m_access_time_CPU5970 + 0 m_access_time_CPU1900
   + 0 m_access_time_CPU6364 + 0 m_access_time_CPU1757
   + 0 m_access_time_CPU7398 + 0 m_access_time_CPU1680
   + 0 m_access_time_CPU1756 + 0 m_access_time_CPU6503
   + 0 m_access_time_CPU3649 + 0 m_access_time_CPU4236
   + 0 m_access_time_CPU9147 + res
Subject To
 cov_0_4: frac[1,4] + frac[6,4] >= 0.7
 cov_1_4: frac[0,4] + frac[3,4] >= 0.9
 cov_2_5: frac[2,5] + frac[6,5] >= 0.9
 cov_3_5: frac[3,5] + frac[7,5] >= 1
 cov_4_1: frac[0,1] + frac[5,1] >= 0.9
 cov_5_5: frac[1,5] + frac[2,5] >= 0.9
 cov_6_5: frac[4,5] + frac[6,5] >= 0.8
 cov_7_5: frac[1,5] + frac[7,5] >= 0.7
 cov_8_7: frac[9,7] + frac[14,7] >= 0.7
 cov_9_2: frac[13,2] + frac[15,2] >= 0.7
 cov_10_7: frac[10,7] + frac[15,7] >= 0.9
 cov_11_0: frac[9,0] + frac[12,0] >= 0.9
 cov_12_7: frac[9,7] + frac[10,7] >= 0.9
 cov_13_2: frac[11,2] + frac[15,2] >= 0.9
 cov_14_0: frac[8,0] + frac[14,0] >= 0.8
 cov_15_0: frac[8,0] + frac[13,0] >= 0.8
 cov_16_3: frac[20,3] + frac[23,3] >= 1
 cov_17_3: frac[19,3] + frac[23,3] >= 0.8
 cov_18_3: frac[17,3] + frac[19,3] >= 0.9
 cov_19_3: frac[20,3] + frac[22,3] >= 0.7
 cov_20_3: frac[17,3] + frac[21,3] >= 1
 cov_21_6: frac[21,6] + frac[22,6] >= 1
 cov_22_6: frac[17,6] + frac[18,6] >= 0.8
 cov_23_8: frac[17,8] + frac[20,8] >= 1
 accuracy_17_8: - 50.9765625 frac[17,8] + mem[17,8] = 0
 accuracy_20_3: - 50.9765625 frac[20,3] + mem[20,3] = 0
 accuracy_11_2: - 50.9765625 frac[11,2] + mem[11,2] = 0
 accuracy_8_0: - 50.9765625 frac[8,0] + mem[8,0] = 0
 accuracy_23_3: - 50.9765625 frac[23,3] + mem[23,3] = 0
 accuracy_17_3: - 50.9765625 frac[17,3] + mem[17,3] = 0
 accuracy_14_0: - 50.9765625 frac[14,0] + mem[14,0] = 0
 accuracy_22_6: - 50.9765625 frac[22,6] + mem[22,6] = 0
 accuracy_21_6: - 50.9765625 frac[21,6] + mem[21,6] = 0
 accuracy_5_1: - 50.9765625 frac[5,1] + mem[5,1] = 0
 accuracy_2_5: - 50.9765625 frac[2,5] + mem[2,5] = 0
 accuracy_20_8: - 50.9765625 frac[20,8] + mem[20,8] = 0
 accuracy_9_0: - 50.9765625 frac[9,0] + mem[9,0] = 0
 accuracy_17_6: - 50.9765625 frac[17,6] + mem[17,6] = 0
 accuracy_10_7: - 50.9765625 frac[10,7] + mem[10,7] = 0
 accuracy_1_5: - 50.9765625 frac[1,5] + mem[1,5] = 0
 accuracy_19_3: - 50.9765625 frac[19,3] + mem[19,3] = 0
 accuracy_0_4: - 50.9765625 frac[0,4] + mem[0,4] = 0
 accuracy_9_7: - 50.9765625 frac[9,7] + mem[9,7] = 0
 accuracy_22_3: - 50.9765625 frac[22,3] + mem[22,3] = 0
 accuracy_6_4: - 50.9765625 frac[6,4] + mem[6,4] = 0
 accuracy_13_2: - 50.9765625 frac[13,2] + mem[13,2] = 0
 accuracy_4_5: - 50.9765625 frac[4,5] + mem[4,5] = 0
 accuracy_1_4: - 50.9765625 frac[1,4] + mem[1,4] = 0
 accuracy_7_5: - 50.9765625 frac[7,5] + mem[7,5] = 0
 accuracy_18_6: - 50.9765625 frac[18,6] + mem[18,6] = 0
 accuracy_15_7: - 50.9765625 frac[15,7] + mem[15,7] = 0
 accuracy_6_5: - 50.9765625 frac[6,5] + mem[6,5] = 0
 accuracy_3_5: - 50.9765625 frac[3,5] + mem[3,5] = 0
 accuracy_0_1: - 50.9765625 frac[0,1] + mem[0,1] = 0
 accuracy_15_2: - 50.9765625 frac[15,2] + mem[15,2] = 0
 accuracy_12_0: - 50.9765625 frac[12,0] + mem[12,0] = 0
 accuracy_14_7: - 50.9765625 frac[14,7] + mem[14,7] = 0
 accuracy_21_3: - 50.9765625 frac[21,3] + mem[21,3] = 0
 accuracy_3_4: - 50.9765625 frac[3,4] + mem[3,4] = 0
 accuracy_13_0: - 50.9765625 frac[13,0] + mem[13,0] = 0
 mem_tot_CPU7865: - mem[0,4] - mem[0,1] + mem_tot_CPU7865 = 0
 R61: - 3.05176e-05 mem_tot_CPU7865 + normalized_mem_tot_CPU7865 = 0
 rows_tot_CPU7865: - 3 frac[0,4] - 3 frac[0,1] + rows_tot_CPU7865 = 0
 R63: - 0.0833333333333333 rows_tot_CPU7865 + normalized_rows_tot_CPU7865
   = 0
 mem_tot_CPU3360: - mem[1,5] - mem[1,4] + mem_tot_CPU3360 = 0
 R65: - 3.05176e-05 mem_tot_CPU3360 + normalized_mem_tot_CPU3360 = 0
 rows_tot_CPU3360: - 3 frac[1,5] - 3 frac[1,4] + rows_tot_CPU3360 = 0
 R67: - 0.0833333333333333 rows_tot_CPU3360 + normalized_rows_tot_CPU3360
   = 0
 mem_tot_CPU6779: - mem[2,5] + mem_tot_CPU6779 = 0
 R69: - 3.05176e-05 mem_tot_CPU6779 + normalized_mem_tot_CPU6779 = 0
 rows_tot_CPU6779: - 3 frac[2,5] + rows_tot_CPU6779 = 0
 R71: - 0.0833333333333333 rows_tot_CPU6779 + normalized_rows_tot_CPU6779
   = 0
 mem_tot_CPU9205: - mem[3,5] - mem[3,4] + mem_tot_CPU9205 = 0
 R73: - 3.05176e-05 mem_tot_CPU9205 + normalized_mem_tot_CPU9205 = 0
 rows_tot_CPU9205: - 3 frac[3,5] - 3 frac[3,4] + rows_tot_CPU9205 = 0
 R75: - 0.0833333333333333 rows_tot_CPU9205 + normalized_rows_tot_CPU9205
   = 0
 mem_tot_CPU5603: - mem[4,5] + mem_tot_CPU5603 = 0
 R77: - 3.05176e-05 mem_tot_CPU5603 + normalized_mem_tot_CPU5603 = 0
 rows_tot_CPU5603: - 3 frac[4,5] + rows_tot_CPU5603 = 0
 R79: - 0.0833333333333333 rows_tot_CPU5603 + normalized_rows_tot_CPU5603
   = 0
 mem_tot_CPU3189: - mem[5,1] + mem_tot_CPU3189 = 0
 R81: - 3.05176e-05 mem_tot_CPU3189 + normalized_mem_tot_CPU3189 = 0
 rows_tot_CPU3189: - 3 frac[5,1] + rows_tot_CPU3189 = 0
 R83: - 0.0833333333333333 rows_tot_CPU3189 + normalized_rows_tot_CPU3189
   = 0
 mem_tot_CPU1861: - mem[6,4] - mem[6,5] + mem_tot_CPU1861 = 0
 R85: - 3.05176e-05 mem_tot_CPU1861 + normalized_mem_tot_CPU1861 = 0
 rows_tot_CPU1861: - 3 frac[6,4] - 3 frac[6,5] + rows_tot_CPU1861 = 0
 R87: - 0.0833333333333333 rows_tot_CPU1861 + normalized_rows_tot_CPU1861
   = 0
 mem_tot_CPU2730: - mem[7,5] + mem_tot_CPU2730 = 0
 R89: - 3.05176e-05 mem_tot_CPU2730 + normalized_mem_tot_CPU2730 = 0
 rows_tot_CPU2730: - 3 frac[7,5] + rows_tot_CPU2730 = 0
 R91: - 0.0833333333333333 rows_tot_CPU2730 + normalized_rows_tot_CPU2730
   = 0
 mem_tot_CPU7358: - mem[8,0] + mem_tot_CPU7358 = 0
 R93: - 3.05176e-05 mem_tot_CPU7358 + normalized_mem_tot_CPU7358 = 0
 rows_tot_CPU7358: - 3 frac[8,0] + rows_tot_CPU7358 = 0
 R95: - 0.0833333333333333 rows_tot_CPU7358 + normalized_rows_tot_CPU7358
   = 0
 mem_tot_CPU6183: - mem[9,0] - mem[9,7] + mem_tot_CPU6183 = 0
 R97: - 3.05176e-05 mem_tot_CPU6183 + normalized_mem_tot_CPU6183 = 0
 rows_tot_CPU6183: - 3 frac[9,0] - 3 frac[9,7] + rows_tot_CPU6183 = 0
 R99: - 0.0833333333333333 rows_tot_CPU6183 + normalized_rows_tot_CPU6183
   = 0
 mem_tot_CPU6929: - mem[10,7] + mem_tot_CPU6929 = 0
 R101: - 3.05176e-05 mem_tot_CPU6929 + normalized_mem_tot_CPU6929 = 0
 rows_tot_CPU6929: - 3 frac[10,7] + rows_tot_CPU6929 = 0
 R103: - 0.0833333333333333 rows_tot_CPU6929 + normalized_rows_tot_CPU6929
   = 0
 mem_tot_CPU9110: - mem[11,2] + mem_tot_CPU9110 = 0
 R105: - 3.05176e-05 mem_tot_CPU9110 + normalized_mem_tot_CPU9110 = 0
 rows_tot_CPU9110: - 3 frac[11,2] + rows_tot_CPU9110 = 0
 R107: - 0.0833333333333333 rows_tot_CPU9110 + normalized_rows_tot_CPU9110
   = 0
 mem_tot_CPU6689: - mem[12,0] + mem_tot_CPU6689 = 0
 R109: - 3.05176e-05 mem_tot_CPU6689 + normalized_mem_tot_CPU6689 = 0
 rows_tot_CPU6689: - 3 frac[12,0] + rows_tot_CPU6689 = 0
 R111: - 0.0833333333333333 rows_tot_CPU6689 + normalized_rows_tot_CPU6689
   = 0
 mem_tot_CPU5970: - mem[13,2] - mem[13,0] + mem_tot_CPU5970 = 0
 R113: - 3.05176e-05 mem_tot_CPU5970 + normalized_mem_tot_CPU5970 = 0
 rows_tot_CPU5970: - 3 frac[13,2] - 3 frac[13,0] + rows_tot_CPU5970 = 0
 R115: - 0.0833333333333333 rows_tot_CPU5970 + normalized_rows_tot_CPU5970
   = 0
 mem_tot_CPU1900: - mem[14,0] - mem[14,7] + mem_tot_CPU1900 = 0
 R117: - 3.05176e-05 mem_tot_CPU1900 + normalized_mem_tot_CPU1900 = 0
 rows_tot_CPU1900: - 3 frac[14,0] - 3 frac[14,7] + rows_tot_CPU1900 = 0
 R119: - 0.0833333333333333 rows_tot_CPU1900 + normalized_rows_tot_CPU1900
   = 0
 mem_tot_CPU6364: - mem[15,7] - mem[15,2] + mem_tot_CPU6364 = 0
 R121: - 3.05176e-05 mem_tot_CPU6364 + normalized_mem_tot_CPU6364 = 0
 rows_tot_CPU6364: - 3 frac[15,7] - 3 frac[15,2] + rows_tot_CPU6364 = 0
 R123: - 0.0833333333333333 rows_tot_CPU6364 + normalized_rows_tot_CPU6364
   = 0
 mem_tot_CPU1757: mem_tot_CPU1757 = 0
 R125: - 3.05176e-05 mem_tot_CPU1757 + normalized_mem_tot_CPU1757 = 0
 rows_tot_CPU1757: rows_tot_CPU1757 = 0
 R127: - 0.0833333333333333 rows_tot_CPU1757 + normalized_rows_tot_CPU1757
   = 0
 mem_tot_CPU7398: - mem[17,8] - mem[17,3] - mem[17,6] + mem_tot_CPU7398
   = 0
 R129: - 3.05176e-05 mem_tot_CPU7398 + normalized_mem_tot_CPU7398 = 0
 rows_tot_CPU7398: - 3 frac[17,8] - 3 frac[17,3] - 3 frac[17,6]
   + rows_tot_CPU7398 = 0
 R131: - 0.0833333333333333 rows_tot_CPU7398 + normalized_rows_tot_CPU7398
   = 0
 mem_tot_CPU1680: - mem[18,6] + mem_tot_CPU1680 = 0
 R133: - 3.05176e-05 mem_tot_CPU1680 + normalized_mem_tot_CPU1680 = 0
 rows_tot_CPU1680: - 3 frac[18,6] + rows_tot_CPU1680 = 0
 R135: - 0.0833333333333333 rows_tot_CPU1680 + normalized_rows_tot_CPU1680
   = 0
 mem_tot_CPU1756: - mem[19,3] + mem_tot_CPU1756 = 0
 R137: - 3.05176e-05 mem_tot_CPU1756 + normalized_mem_tot_CPU1756 = 0
 rows_tot_CPU1756: - 3 frac[19,3] + rows_tot_CPU1756 = 0
 R139: - 0.0833333333333333 rows_tot_CPU1756 + normalized_rows_tot_CPU1756
   = 0
 mem_tot_CPU6503: - mem[20,3] - mem[20,8] + mem_tot_CPU6503 = 0
 R141: - 3.05176e-05 mem_tot_CPU6503 + normalized_mem_tot_CPU6503 = 0
 rows_tot_CPU6503: - 3 frac[20,3] - 3 frac[20,8] + rows_tot_CPU6503 = 0
 R143: - 0.0833333333333333 rows_tot_CPU6503 + normalized_rows_tot_CPU6503
   = 0
 mem_tot_CPU3649: - mem[21,6] - mem[21,3] + mem_tot_CPU3649 = 0
 R145: - 3.05176e-05 mem_tot_CPU3649 + normalized_mem_tot_CPU3649 = 0
 rows_tot_CPU3649: - 3 frac[21,6] - 3 frac[21,3] + rows_tot_CPU3649 = 0
 R147: - 0.0833333333333333 rows_tot_CPU3649 + normalized_rows_tot_CPU3649
   = 0
 mem_tot_CPU4236: - mem[22,6] - mem[22,3] + mem_tot_CPU4236 = 0
 R149: - 3.05176e-05 mem_tot_CPU4236 + normalized_mem_tot_CPU4236 = 0
 rows_tot_CPU4236: - 3 frac[22,6] - 3 frac[22,3] + rows_tot_CPU4236 = 0
 R151: - 0.0833333333333333 rows_tot_CPU4236 + normalized_rows_tot_CPU4236
   = 0
 mem_tot_CPU9147: - mem[23,3] + mem_tot_CPU9147 = 0
 R153: - 3.05176e-05 mem_tot_CPU9147 + normalized_mem_tot_CPU9147 = 0
 rows_tot_CPU9147: - 3 frac[23,3] + rows_tot_CPU9147 = 0
 R155: - 0.0833333333333333 rows_tot_CPU9147 + normalized_rows_tot_CPU9147
   = 0
 row_capacity_CPU7865: rows_tot_CPU7865 <= 12
 row_capacity_CPU3360: rows_tot_CPU3360 <= 12
 row_capacity_CPU6779: rows_tot_CPU6779 <= 12
 row_capacity_CPU9205: rows_tot_CPU9205 <= 12
 row_capacity_CPU5603: rows_tot_CPU5603 <= 12
 row_capacity_CPU3189: rows_tot_CPU3189 <= 12
 row_capacity_CPU1861: rows_tot_CPU1861 <= 12
 row_capacity_CPU2730: rows_tot_CPU2730 <= 12
 row_capacity_CPU7358: rows_tot_CPU7358 <= 12
 row_capacity_CPU6183: rows_tot_CPU6183 <= 12
 row_capacity_CPU6929: rows_tot_CPU6929 <= 12
 row_capacity_CPU9110: rows_tot_CPU9110 <= 12
 row_capacity_CPU6689: rows_tot_CPU6689 <= 12
 row_capacity_CPU5970: rows_tot_CPU5970 <= 12
 row_capacity_CPU1900: rows_tot_CPU1900 <= 12
 row_capacity_CPU6364: rows_tot_CPU6364 <= 12
 row_capacity_CPU1757: rows_tot_CPU1757 <= 12
 row_capacity_CPU7398: rows_tot_CPU7398 <= 12
 row_capacity_CPU1680: rows_tot_CPU1680 <= 12
 row_capacity_CPU1756: rows_tot_CPU1756 <= 12
 row_capacity_CPU6503: rows_tot_CPU6503 <= 12
 row_capacity_CPU3649: rows_tot_CPU3649 <= 12
 row_capacity_CPU4236: rows_tot_CPU4236 <= 12
 row_capacity_CPU9147: rows_tot_CPU9147 <= 12
 ns_single_CPU7865: 3.5 rows_tot_CPU7865 - ns_single_CPU7865
   + pdt_m_rows_CPU7865 = 0
 ns_sketch_CPU7865: - ns_single_CPU7865
   + 11.42863142857143 cores_sketch_CPU7865 >= 0
 capacity_cores_CPU7865: cores_sketch_CPU7865 <= 2
 ns_single_CPU3360: 3.5 rows_tot_CPU3360 - ns_single_CPU3360
   + pdt_m_rows_CPU3360 = 0
 ns_sketch_CPU3360: - ns_single_CPU3360
   + 11.42863142857143 cores_sketch_CPU3360 >= 0
 capacity_cores_CPU3360: cores_sketch_CPU3360 <= 2
 ns_single_CPU6779: 3.5 rows_tot_CPU6779 - ns_single_CPU6779
   + pdt_m_rows_CPU6779 = 0
 ns_sketch_CPU6779: - ns_single_CPU6779
   + 11.42863142857143 cores_sketch_CPU6779 >= 0
 capacity_cores_CPU6779: cores_sketch_CPU6779 <= 2
 ns_single_CPU9205: 3.5 rows_tot_CPU9205 - ns_single_CPU9205
   + pdt_m_rows_CPU9205 = 0
 ns_sketch_CPU9205: - ns_single_CPU9205
   + 11.42863142857143 cores_sketch_CPU9205 >= 0
 capacity_cores_CPU9205: cores_sketch_CPU9205 <= 2
 ns_single_CPU5603: 3.5 rows_tot_CPU5603 - ns_single_CPU5603
   + pdt_m_rows_CPU5603 = 0
 ns_sketch_CPU5603: - ns_single_CPU5603
   + 11.42863142857143 cores_sketch_CPU5603 >= 0
 capacity_cores_CPU5603: cores_sketch_CPU5603 <= 2
 ns_single_CPU3189: 3.5 rows_tot_CPU3189 - ns_single_CPU3189
   + pdt_m_rows_CPU3189 = 0
 ns_sketch_CPU3189: - ns_single_CPU3189
   + 11.42863142857143 cores_sketch_CPU3189 >= 0
 capacity_cores_CPU3189: cores_sketch_CPU3189 <= 2
 ns_single_CPU1861: 3.5 rows_tot_CPU1861 - ns_single_CPU1861
   + pdt_m_rows_CPU1861 = 0
 ns_sketch_CPU1861: - ns_single_CPU1861
   + 11.42863142857143 cores_sketch_CPU1861 >= 0
 capacity_cores_CPU1861: cores_sketch_CPU1861 <= 2
 ns_single_CPU2730: 3.5 rows_tot_CPU2730 - ns_single_CPU2730
   + pdt_m_rows_CPU2730 = 0
 ns_sketch_CPU2730: - ns_single_CPU2730
   + 11.42863142857143 cores_sketch_CPU2730 >= 0
 capacity_cores_CPU2730: cores_sketch_CPU2730 <= 2
 ns_single_CPU7358: 3.5 rows_tot_CPU7358 - ns_single_CPU7358
   + pdt_m_rows_CPU7358 = 0
 ns_sketch_CPU7358: - ns_single_CPU7358
   + 11.42863142857143 cores_sketch_CPU7358 >= 0
 capacity_cores_CPU7358: cores_sketch_CPU7358 <= 2
 ns_single_CPU6183: 3.5 rows_tot_CPU6183 - ns_single_CPU6183
   + pdt_m_rows_CPU6183 = 0
 ns_sketch_CPU6183: - ns_single_CPU6183
   + 11.42863142857143 cores_sketch_CPU6183 >= 0
 capacity_cores_CPU6183: cores_sketch_CPU6183 <= 2
 ns_single_CPU6929: 3.5 rows_tot_CPU6929 - ns_single_CPU6929
   + pdt_m_rows_CPU6929 = 0
 ns_sketch_CPU6929: - ns_single_CPU6929
   + 11.42863142857143 cores_sketch_CPU6929 >= 0
 capacity_cores_CPU6929: cores_sketch_CPU6929 <= 2
 ns_single_CPU9110: 3.5 rows_tot_CPU9110 - ns_single_CPU9110
   + pdt_m_rows_CPU9110 = 0
 ns_sketch_CPU9110: - ns_single_CPU9110
   + 11.42863142857143 cores_sketch_CPU9110 >= 0
 capacity_cores_CPU9110: cores_sketch_CPU9110 <= 2
 ns_single_CPU6689: 3.5 rows_tot_CPU6689 - ns_single_CPU6689
   + pdt_m_rows_CPU6689 = 0
 ns_sketch_CPU6689: - ns_single_CPU6689
   + 11.42863142857143 cores_sketch_CPU6689 >= 0
 capacity_cores_CPU6689: cores_sketch_CPU6689 <= 2
 ns_single_CPU5970: 3.5 rows_tot_CPU5970 - ns_single_CPU5970
   + pdt_m_rows_CPU5970 = 0
 ns_sketch_CPU5970: - ns_single_CPU5970
   + 11.42863142857143 cores_sketch_CPU5970 >= 0
 capacity_cores_CPU5970: cores_sketch_CPU5970 <= 2
 ns_single_CPU1900: 3.5 rows_tot_CPU1900 - ns_single_CPU1900
   + pdt_m_rows_CPU1900 = 0
 ns_sketch_CPU1900: - ns_single_CPU1900
   + 11.42863142857143 cores_sketch_CPU1900 >= 0
 capacity_cores_CPU1900: cores_sketch_CPU1900 <= 2
 ns_single_CPU6364: 3.5 rows_tot_CPU6364 - ns_single_CPU6364
   + pdt_m_rows_CPU6364 = 0
 ns_sketch_CPU6364: - ns_single_CPU6364
   + 11.42863142857143 cores_sketch_CPU6364 >= 0
 capacity_cores_CPU6364: cores_sketch_CPU6364 <= 2
 ns_single_CPU1757: 3.5 rows_tot_CPU1757 - ns_single_CPU1757
   + pdt_m_rows_CPU1757 = 0
 ns_sketch_CPU1757: - ns_single_CPU1757
   + 11.42863142857143 cores_sketch_CPU1757 >= 0
 capacity_cores_CPU1757: cores_sketch_CPU1757 <= 2
 ns_single_CPU7398: 3.5 rows_tot_CPU7398 - ns_single_CPU7398
   + pdt_m_rows_CPU7398 = 0
 ns_sketch_CPU7398: - ns_single_CPU7398
   + 11.42863142857143 cores_sketch_CPU7398 >= 0
 capacity_cores_CPU7398: cores_sketch_CPU7398 <= 2
 ns_single_CPU1680: 3.5 rows_tot_CPU1680 - ns_single_CPU1680
   + pdt_m_rows_CPU1680 = 0
 ns_sketch_CPU1680: - ns_single_CPU1680
   + 11.42863142857143 cores_sketch_CPU1680 >= 0
 capacity_cores_CPU1680: cores_sketch_CPU1680 <= 2
 ns_single_CPU1756: 3.5 rows_tot_CPU1756 - ns_single_CPU1756
   + pdt_m_rows_CPU1756 = 0
 ns_sketch_CPU1756: - ns_single_CPU1756
   + 11.42863142857143 cores_sketch_CPU1756 >= 0
 capacity_cores_CPU1756: cores_sketch_CPU1756 <= 2
 ns_single_CPU6503: 3.5 rows_tot_CPU6503 - ns_single_CPU6503
   + pdt_m_rows_CPU6503 = 0
 ns_sketch_CPU6503: - ns_single_CPU6503
   + 11.42863142857143 cores_sketch_CPU6503 >= 0
 capacity_cores_CPU6503: cores_sketch_CPU6503 <= 2
 ns_single_CPU3649: 3.5 rows_tot_CPU3649 - ns_single_CPU3649
   + pdt_m_rows_CPU3649 = 0
 ns_sketch_CPU3649: - ns_single_CPU3649
   + 11.42863142857143 cores_sketch_CPU3649 >= 0
 capacity_cores_CPU3649: cores_sketch_CPU3649 <= 2
 ns_single_CPU4236: 3.5 rows_tot_CPU4236 - ns_single_CPU4236
   + pdt_m_rows_CPU4236 = 0
 ns_sketch_CPU4236: - ns_single_CPU4236
   + 11.42863142857143 cores_sketch_CPU4236 >= 0
 capacity_cores_CPU4236: cores_sketch_CPU4236 <= 2
 ns_single_CPU9147: 3.5 rows_tot_CPU9147 - ns_single_CPU9147
   + pdt_m_rows_CPU9147 = 0
 ns_sketch_CPU9147: - ns_single_CPU9147
   + 11.42863142857143 cores_sketch_CPU9147 >= 0
 capacity_cores_CPU9147: cores_sketch_CPU9147 <= 2
 res_acc: 1.220703125e-04 mem_tot_CPU7865 + 1.220703125e-04 mem_tot_CPU3360
   + 1.220703125e-04 mem_tot_CPU6779 + 1.220703125e-04 mem_tot_CPU9205
   + 1.220703125e-04 mem_tot_CPU5603 + 1.220703125e-04 mem_tot_CPU3189
   + 1.220703125e-04 mem_tot_CPU1861 + 1.220703125e-04 mem_tot_CPU2730
   + 1.220703125e-04 mem_tot_CPU7358 + 1.220703125e-04 mem_tot_CPU6183
   + 1.220703125e-04 mem_tot_CPU6929 + 1.220703125e-04 mem_tot_CPU9110
   + 1.220703125e-04 mem_tot_CPU6689 + 1.220703125e-04 mem_tot_CPU5970
   + 1.220703125e-04 mem_tot_CPU1900 + 1.220703125e-04 mem_tot_CPU6364
   + 1.220703125e-04 mem_tot_CPU1757 + 1.220703125e-04 mem_tot_CPU7398
   + 1.220703125e-04 mem_tot_CPU1680 + 1.220703125e-04 mem_tot_CPU1756
   + 1.220703125e-04 mem_tot_CPU6503 + 1.220703125e-04 mem_tot_CPU3649
   + 1.220703125e-04 mem_tot_CPU4236 + 1.220703125e-04 mem_tot_CPU9147
   + 10 cores_sketch_CPU7865 + 10 cores_sketch_CPU3360
   + 10 cores_sketch_CPU6779 + 10 cores_sketch_CPU9205
   + 10 cores_sketch_CPU5603 + 10 cores_sketch_CPU3189
   + 10 cores_sketch_CPU1861 + 10 cores_sketch_CPU2730
   + 10 cores_sketch_CPU7358 + 10 cores_sketch_CPU6183
   + 10 cores_sketch_CPU6929 + 10 cores_sketch_CPU9110
   + 10 cores_sketch_CPU6689 + 10 cores_sketch_CPU5970
   + 10 cores_sketch_CPU1900 + 10 cores_sketch_CPU6364
   + 10 cores_sketch_CPU1757 + 10 cores_sketch_CPU7398
   + 10 cores_sketch_CPU1680 + 10 cores_sketch_CPU1756
   + 10 cores_sketch_CPU6503 + 10 cores_sketch_CPU3649
   + 10 cores_sketch_CPU4236 + 10 cores_sketch_CPU9147 - res = -1200
 pdt_m_rows_CPU7865: pdt_m_rows_CPU7865 + [
   - rows_tot_CPU7865 * m_access_time_CPU7865 ] = 0
 pdt_m_rows_CPU3360: pdt_m_rows_CPU3360 + [
   - rows_tot_CPU3360 * m_access_time_CPU3360 ] = 0
 pdt_m_rows_CPU6779: pdt_m_rows_CPU6779 + [
   - rows_tot_CPU6779 * m_access_time_CPU6779 ] = 0
 pdt_m_rows_CPU9205: pdt_m_rows_CPU9205 + [
   - rows_tot_CPU9205 * m_access_time_CPU9205 ] = 0
 pdt_m_rows_CPU5603: pdt_m_rows_CPU5603 + [
   - rows_tot_CPU5603 * m_access_time_CPU5603 ] = 0
 pdt_m_rows_CPU3189: pdt_m_rows_CPU3189 + [
   - rows_tot_CPU3189 * m_access_time_CPU3189 ] = 0
 pdt_m_rows_CPU1861: pdt_m_rows_CPU1861 + [
   - rows_tot_CPU1861 * m_access_time_CPU1861 ] = 0
 pdt_m_rows_CPU2730: pdt_m_rows_CPU2730 + [
   - rows_tot_CPU2730 * m_access_time_CPU2730 ] = 0
 pdt_m_rows_CPU7358: pdt_m_rows_CPU7358 + [
   - rows_tot_CPU7358 * m_access_time_CPU7358 ] = 0
 pdt_m_rows_CPU6183: pdt_m_rows_CPU6183 + [
   - rows_tot_CPU6183 * m_access_time_CPU6183 ] = 0
 pdt_m_rows_CPU6929: pdt_m_rows_CPU6929 + [
   - rows_tot_CPU6929 * m_access_time_CPU6929 ] = 0
 pdt_m_rows_CPU9110: pdt_m_rows_CPU9110 + [
   - rows_tot_CPU9110 * m_access_time_CPU9110 ] = 0
 pdt_m_rows_CPU6689: pdt_m_rows_CPU6689 + [
   - rows_tot_CPU6689 * m_access_time_CPU6689 ] = 0
 pdt_m_rows_CPU5970: pdt_m_rows_CPU5970 + [
   - rows_tot_CPU5970 * m_access_time_CPU5970 ] = 0
 pdt_m_rows_CPU1900: pdt_m_rows_CPU1900 + [
   - rows_tot_CPU1900 * m_access_time_CPU1900 ] = 0
 pdt_m_rows_CPU6364: pdt_m_rows_CPU6364 + [
   - rows_tot_CPU6364 * m_access_time_CPU6364 ] = 0
 pdt_m_rows_CPU1757: pdt_m_rows_CPU1757 + [
   - rows_tot_CPU1757 * m_access_time_CPU1757 ] = 0
 pdt_m_rows_CPU7398: pdt_m_rows_CPU7398 + [
   - rows_tot_CPU7398 * m_access_time_CPU7398 ] = 0
 pdt_m_rows_CPU1680: pdt_m_rows_CPU1680 + [
   - rows_tot_CPU1680 * m_access_time_CPU1680 ] = 0
 pdt_m_rows_CPU1756: pdt_m_rows_CPU1756 + [
   - rows_tot_CPU1756 * m_access_time_CPU1756 ] = 0
 pdt_m_rows_CPU6503: pdt_m_rows_CPU6503 + [
   - rows_tot_CPU6503 * m_access_time_CPU6503 ] = 0
 pdt_m_rows_CPU3649: pdt_m_rows_CPU3649 + [
   - rows_tot_CPU3649 * m_access_time_CPU3649 ] = 0
 pdt_m_rows_CPU4236: pdt_m_rows_CPU4236 + [
   - rows_tot_CPU4236 * m_access_time_CPU4236 ] = 0
 pdt_m_rows_CPU9147: pdt_m_rows_CPU9147 + [
   - rows_tot_CPU9147 * m_access_time_CPU9147 ] = 0
Bounds
 frac[0,4] <= 1
 frac[0,1] <= 1
 frac[1,5] <= 1
 frac[1,4] <= 1
 frac[2,5] <= 1
 frac[3,5] <= 1
 frac[3,4] <= 1
 frac[4,5] <= 1
 frac[5,1] <= 1
 frac[6,4] <= 1
 frac[6,5] <= 1
 frac[7,5] <= 1
 frac[8,0] <= 1
 frac[9,0] <= 1
 frac[9,7] <= 1
 frac[10,7] <= 1
 frac[11,2] <= 1
 frac[12,0] <= 1
 frac[13,2] <= 1
 frac[13,0] <= 1
 frac[14,0] <= 1
 frac[14,7] <= 1
 frac[15,7] <= 1
 frac[15,2] <= 1
 frac[17,8] <= 1
 frac[17,3] <= 1
 frac[17,6] <= 1
 frac[18,6] <= 1
 frac[19,3] <= 1
 frac[20,3] <= 1
 frac[20,8] <= 1
 frac[21,6] <= 1
 frac[21,3] <= 1
 frac[22,6] <= 1
 frac[22,3] <= 1
 frac[23,3] <= 1
 mem_tot_CPU7865 <= 32768
 normalized_mem_tot_CPU7865 <= 1
 normalized_rows_tot_CPU7865 <= 1
 mem_tot_CPU3360 <= 32768
 normalized_mem_tot_CPU3360 <= 1
 normalized_rows_tot_CPU3360 <= 1
 mem_tot_CPU6779 <= 32768
 normalized_mem_tot_CPU6779 <= 1
 normalized_rows_tot_CPU6779 <= 1
 mem_tot_CPU9205 <= 32768
 normalized_mem_tot_CPU9205 <= 1
 normalized_rows_tot_CPU9205 <= 1
 mem_tot_CPU5603 <= 32768
 normalized_mem_tot_CPU5603 <= 1
 normalized_rows_tot_CPU5603 <= 1
 mem_tot_CPU3189 <= 32768
 normalized_mem_tot_CPU3189 <= 1
 normalized_rows_tot_CPU3189 <= 1
 mem_tot_CPU1861 <= 32768
 normalized_mem_tot_CPU1861 <= 1
 normalized_rows_tot_CPU1861 <= 1
 mem_tot_CPU2730 <= 32768
 normalized_mem_tot_CPU2730 <= 1
 normalized_rows_tot_CPU2730 <= 1
 mem_tot_CPU7358 <= 32768
 normalized_mem_tot_CPU7358 <= 1
 normalized_rows_tot_CPU7358 <= 1
 mem_tot_CPU6183 <= 32768
 normalized_mem_tot_CPU6183 <= 1
 normalized_rows_tot_CPU6183 <= 1
 mem_tot_CPU6929 <= 32768
 normalized_mem_tot_CPU6929 <= 1
 normalized_rows_tot_CPU6929 <= 1
 mem_tot_CPU9110 <= 32768
 normalized_mem_tot_CPU9110 <= 1
 normalized_rows_tot_CPU9110 <= 1
 mem_tot_CPU6689 <= 32768
 normalized_mem_tot_CPU6689 <= 1
 normalized_rows_tot_CPU6689 <= 1
 mem_tot_CPU5970 <= 32768
 normalized_mem_tot_CPU5970 <= 1
 normalized_rows_tot_CPU5970 <= 1
 mem_tot_CPU1900 <= 32768
 normalized_mem_tot_CPU1900 <= 1
 normalized_rows_tot_CPU1900 <= 1
 mem_tot_CPU6364 <= 32768
 normalized_mem_tot_CPU6364 <= 1
 normalized_rows_tot_CPU6364 <= 1
 mem_tot_CPU1757 <= 32768
 normalized_mem_tot_CPU1757 <= 1
 normalized_rows_tot_CPU1757 <= 1
 mem_tot_CPU7398 <= 32768
 normalized_mem_tot_CPU7398 <= 1
 normalized_rows_tot_CPU7398 <= 1
 mem_tot_CPU1680 <= 32768
 normalized_mem_tot_CPU1680 <= 1
 normalized_rows_tot_CPU1680 <= 1
 mem_tot_CPU1756 <= 32768
 normalized_mem_tot_CPU1756 <= 1
 normalized_rows_tot_CPU1756 <= 1
 mem_tot_CPU6503 <= 32768
 normalized_mem_tot_CPU6503 <= 1
 normalized_rows_tot_CPU6503 <= 1
 mem_tot_CPU3649 <= 32768
 normalized_mem_tot_CPU3649 <= 1
 normalized_rows_tot_CPU3649 <= 1
 mem_tot_CPU4236 <= 32768
 normalized_mem_tot_CPU4236 <= 1
 normalized_rows_tot_CPU4236 <= 1
 mem_tot_CPU9147 <= 32768
 normalized_mem_tot_CPU9147 <= 1
 normalized_rows_tot_CPU9147 <= 1
 cores_sketch_CPU7865 <= 7
 cores_sketch_CPU3360 <= 7
 cores_sketch_CPU6779 <= 7
 cores_sketch_CPU9205 <= 7
 cores_sketch_CPU5603 <= 7
 cores_sketch_CPU3189 <= 7
 cores_sketch_CPU1861 <= 7
 cores_sketch_CPU2730 <= 7
 cores_sketch_CPU7358 <= 7
 cores_sketch_CPU6183 <= 7
 cores_sketch_CPU6929 <= 7
 cores_sketch_CPU9110 <= 7
 cores_sketch_CPU6689 <= 7
 cores_sketch_CPU5970 <= 7
 cores_sketch_CPU1900 <= 7
 cores_sketch_CPU6364 <= 7
 cores_sketch_CPU1757 <= 7
 cores_sketch_CPU7398 <= 7
 cores_sketch_CPU1680 <= 7
 cores_sketch_CPU1756 <= 7
 cores_sketch_CPU6503 <= 7
 cores_sketch_CPU3649 <= 7
 cores_sketch_CPU4236 <= 7
 cores_sketch_CPU9147 <= 7
Generals
 cores_sketch_CPU7865 cores_sketch_CPU3360 cores_sketch_CPU6779
 cores_sketch_CPU9205 cores_sketch_CPU5603 cores_sketch_CPU3189
 cores_sketch_CPU1861 cores_sketch_CPU2730 cores_sketch_CPU7358
 cores_sketch_CPU6183 cores_sketch_CPU6929 cores_sketch_CPU9110
 cores_sketch_CPU6689 cores_sketch_CPU5970 cores_sketch_CPU1900
 cores_sketch_CPU6364 cores_sketch_CPU1757 cores_sketch_CPU7398
 cores_sketch_CPU1680 cores_sketch_CPU1756 cores_sketch_CPU6503
 cores_sketch_CPU3649 cores_sketch_CPU4236 cores_sketch_CPU9147
General Constraints
 mem_overall_others: max_mem_others = MAX ( normalized_mem_tot_CPU7865 ,
   normalized_mem_tot_CPU3360 , normalized_mem_tot_CPU6779 ,
   normalized_mem_tot_CPU9205 , normalized_mem_tot_CPU5603 ,
   normalized_mem_tot_CPU3189 , normalized_mem_tot_CPU1861 ,
   normalized_mem_tot_CPU2730 , normalized_mem_tot_CPU7358 ,
   normalized_mem_tot_CPU6183 , normalized_mem_tot_CPU6929 ,
   normalized_mem_tot_CPU9110 , normalized_mem_tot_CPU6689 ,
   normalized_mem_tot_CPU5970 , normalized_mem_tot_CPU1900 ,
   normalized_mem_tot_CPU6364 , normalized_mem_tot_CPU1757 ,
   normalized_mem_tot_CPU7398 , normalized_mem_tot_CPU1680 ,
   normalized_mem_tot_CPU1756 , normalized_mem_tot_CPU6503 ,
   normalized_mem_tot_CPU3649 , normalized_mem_tot_CPU4236 ,
   normalized_mem_tot_CPU9147 )
 rows_overall_others: max_rows_others = MAX ( normalized_rows_tot_CPU7865 ,
   normalized_rows_tot_CPU3360 , normalized_rows_tot_CPU6779 ,
   normalized_rows_tot_CPU9205 , normalized_rows_tot_CPU5603 ,
   normalized_rows_tot_CPU3189 , normalized_rows_tot_CPU1861 ,
   normalized_rows_tot_CPU2730 , normalized_rows_tot_CPU7358 ,
   normalized_rows_tot_CPU6183 , normalized_rows_tot_CPU6929 ,
   normalized_rows_tot_CPU9110 , normalized_rows_tot_CPU6689 ,
   normalized_rows_tot_CPU5970 , normalized_rows_tot_CPU1900 ,
   normalized_rows_tot_CPU6364 , normalized_rows_tot_CPU1757 ,
   normalized_rows_tot_CPU7398 , normalized_rows_tot_CPU1680 ,
   normalized_rows_tot_CPU1756 , normalized_rows_tot_CPU6503 ,
   normalized_rows_tot_CPU3649 , normalized_rows_tot_CPU4236 ,
   normalized_rows_tot_CPU9147 )
 mem_access_time_CPU7865:
   m_access_time_CPU7865 = PWL ( mem_tot_CPU7865 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU3360:
   m_access_time_CPU3360 = PWL ( mem_tot_CPU3360 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU6779:
   m_access_time_CPU6779 = PWL ( mem_tot_CPU6779 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU9205:
   m_access_time_CPU9205 = PWL ( mem_tot_CPU9205 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU5603:
   m_access_time_CPU5603 = PWL ( mem_tot_CPU5603 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU3189:
   m_access_time_CPU3189 = PWL ( mem_tot_CPU3189 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1861:
   m_access_time_CPU1861 = PWL ( mem_tot_CPU1861 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU2730:
   m_access_time_CPU2730 = PWL ( mem_tot_CPU2730 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU7358:
   m_access_time_CPU7358 = PWL ( mem_tot_CPU7358 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU6183:
   m_access_time_CPU6183 = PWL ( mem_tot_CPU6183 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU6929:
   m_access_time_CPU6929 = PWL ( mem_tot_CPU6929 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU9110:
   m_access_time_CPU9110 = PWL ( mem_tot_CPU9110 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU6689:
   m_access_time_CPU6689 = PWL ( mem_tot_CPU6689 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU5970:
   m_access_time_CPU5970 = PWL ( mem_tot_CPU5970 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1900:
   m_access_time_CPU1900 = PWL ( mem_tot_CPU1900 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU6364:
   m_access_time_CPU6364 = PWL ( mem_tot_CPU6364 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1757:
   m_access_time_CPU1757 = PWL ( mem_tot_CPU1757 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU7398:
   m_access_time_CPU7398 = PWL ( mem_tot_CPU7398 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1680:
   m_access_time_CPU1680 = PWL ( mem_tot_CPU1680 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1756:
   m_access_time_CPU1756 = PWL ( mem_tot_CPU1756 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU6503:
   m_access_time_CPU6503 = PWL ( mem_tot_CPU6503 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU3649:
   m_access_time_CPU3649 = PWL ( mem_tot_CPU3649 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU4236:
   m_access_time_CPU4236 = PWL ( mem_tot_CPU4236 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU9147:
   m_access_time_CPU9147 = PWL ( mem_tot_CPU9147 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

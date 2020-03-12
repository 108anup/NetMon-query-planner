\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU_1
  CPU_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU
  tot_rows_P4: Priority=80 Weight=1 AbsTol=0 RelTol=0
   rows_tot_P4_0 + rows_tot_P4_1 + rows_tot_P4_2 + rows_tot_P4_3
   + rows_tot_P4_4 + rows_tot_P4_5 + rows_tot_P4_6 + rows_tot_P4_7
   + rows_tot_P4_8 + rows_tot_P4_9 + rows_tot_P4_10 + rows_tot_P4_11
   + rows_tot_P4_12 + rows_tot_P4_13 + rows_tot_P4_14 + rows_tot_P4_15
   + rows_tot_P4_16 + rows_tot_P4_17 + rows_tot_P4_18 + rows_tot_P4_19
  P4_rows_load: Priority=70 Weight=1 AbsTol=0 RelTol=0
   max_rows_P4
  tot_mem_CPU: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU_1
  CPU_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU
  tot_mem_P4: Priority=40 Weight=1 AbsTol=0 RelTol=0
   mem_tot_P4_0 + mem_tot_P4_1 + mem_tot_P4_2 + mem_tot_P4_3 + mem_tot_P4_4
   + mem_tot_P4_5 + mem_tot_P4_6 + mem_tot_P4_7 + mem_tot_P4_8
   + mem_tot_P4_9 + mem_tot_P4_10 + mem_tot_P4_11 + mem_tot_P4_12
   + mem_tot_P4_13 + mem_tot_P4_14 + mem_tot_P4_15 + mem_tot_P4_16
   + mem_tot_P4_17 + mem_tot_P4_18 + mem_tot_P4_19
  P4_load_mem: Priority=30 Weight=1 AbsTol=0 RelTol=0
   max_mem_P4
Subject To
 cov_0_0: frac[20,0] + frac[0,0] + frac[1,0] >= 1
 cov_0_1: frac[1,1] + frac[20,1] + frac[0,1] >= 1
 cov_0_2: frac[1,2] + frac[20,2] + frac[0,2] >= 1
 cov_1_0: frac[20,0] + frac[1,0] + frac[2,0] >= 1
 cov_1_1: frac[2,1] + frac[1,1] + frac[20,1] >= 1
 cov_1_2: frac[1,2] + frac[2,2] + frac[20,2] >= 1
 cov_2_0: frac[3,0] + frac[20,0] + frac[2,0] >= 1
 cov_2_1: frac[2,1] + frac[20,1] + frac[3,1] >= 1
 cov_2_2: frac[2,2] + frac[3,2] + frac[20,2] >= 1
 cov_3_0: frac[3,0] + frac[4,0] + frac[20,0] >= 1
 cov_3_1: frac[4,1] + frac[20,1] + frac[3,1] >= 1
 cov_3_2: frac[3,2] + frac[4,2] + frac[20,2] >= 1
 cov_4_0: frac[4,0] + frac[20,0] + frac[5,0] >= 1
 cov_4_1: frac[5,1] + frac[4,1] + frac[20,1] >= 1
 cov_4_2: frac[4,2] + frac[20,2] + frac[5,2] >= 1
 cov_5_0: frac[20,0] + frac[5,0] + frac[6,0] >= 1
 cov_5_1: frac[5,1] + frac[20,1] + frac[6,1] >= 1
 cov_5_2: frac[6,2] + frac[20,2] + frac[5,2] >= 1
 cov_6_0: frac[20,0] + frac[6,0] + frac[7,0] >= 1
 cov_6_1: frac[7,1] + frac[20,1] + frac[6,1] >= 1
 cov_6_2: frac[6,2] + frac[7,2] + frac[20,2] >= 1
 cov_7_0: frac[8,0] + frac[20,0] + frac[7,0] >= 1
 cov_7_1: frac[8,1] + frac[7,1] + frac[20,1] >= 1
 cov_7_2: frac[7,2] + frac[8,2] + frac[20,2] >= 1
 cov_8_0: frac[8,0] + frac[9,0] + frac[20,0] >= 1
 cov_8_1: frac[9,1] + frac[8,1] + frac[20,1] >= 1
 cov_8_2: frac[8,2] + frac[9,2] + frac[20,2] >= 1
 cov_9_0: frac[9,0] + frac[20,0] + frac[10,0] >= 1
 cov_9_1: frac[9,1] + frac[20,1] + frac[10,1] >= 1
 cov_9_2: frac[9,2] + frac[20,2] + frac[10,2] >= 1
 cov_10_0: frac[20,0] + frac[10,0] + frac[11,0] >= 1
 cov_10_1: frac[11,1] + frac[20,1] + frac[10,1] >= 1
 cov_10_2: frac[11,2] + frac[20,2] + frac[10,2] >= 1
 cov_11_0: frac[20,0] + frac[11,0] + frac[12,0] >= 1
 cov_11_1: frac[12,1] + frac[11,1] + frac[20,1] >= 1
 cov_11_2: frac[11,2] + frac[12,2] + frac[20,2] >= 1
 cov_12_0: frac[20,0] + frac[12,0] + frac[13,0] >= 1
 cov_12_1: frac[12,1] + frac[20,1] + frac[13,1] >= 1
 cov_12_2: frac[12,2] + frac[13,2] + frac[20,2] >= 1
 cov_13_0: frac[14,0] + frac[20,0] + frac[13,0] >= 1
 cov_13_1: frac[14,1] + frac[20,1] + frac[13,1] >= 1
 cov_13_2: frac[13,2] + frac[14,2] + frac[20,2] >= 1
 cov_14_0: frac[14,0] + frac[15,0] + frac[20,0] >= 1
 cov_14_1: frac[15,1] + frac[14,1] + frac[20,1] >= 1
 cov_14_2: frac[14,2] + frac[15,2] + frac[20,2] >= 1
 cov_15_0: frac[15,0] + frac[20,0] + frac[16,0] >= 1
 cov_15_1: frac[15,1] + frac[20,1] + frac[16,1] >= 1
 cov_15_2: frac[16,2] + frac[15,2] + frac[20,2] >= 1
 cov_16_0: frac[20,0] + frac[16,0] + frac[17,0] >= 1
 cov_16_1: frac[20,1] + frac[17,1] + frac[16,1] >= 1
 cov_16_2: frac[16,2] + frac[17,2] + frac[20,2] >= 1
 cov_17_0: frac[18,0] + frac[20,0] + frac[17,0] >= 1
 cov_17_1: frac[18,1] + frac[20,1] + frac[17,1] >= 1
 cov_17_2: frac[17,2] + frac[18,2] + frac[20,2] >= 1
 cov_18_0: frac[18,0] + frac[19,0] + frac[20,0] >= 1
 cov_18_1: frac[18,1] + frac[20,1] + frac[19,1] >= 1
 cov_18_2: frac[18,2] + frac[19,2] + frac[20,2] >= 1
 cov_19_0: frac[19,0] + frac[20,0] + frac[0,0] >= 1
 cov_19_1: frac[20,1] + frac[0,1] + frac[19,1] >= 1
 cov_19_2: frac[19,2] + frac[20,2] + frac[0,2] >= 1
 accuracy_18_0: - 1698.9296875 frac[18,0] + mem[18,0] = 0
 accuracy_12_1: - 1698.9296875 frac[12,1] + mem[12,1] = 0
 accuracy_9_1: - 1698.9296875 frac[9,1] + mem[9,1] = 0
 accuracy_3_0: - 1698.9296875 frac[3,0] + mem[3,0] = 0
 accuracy_11_2: - 1698.9296875 frac[11,2] + mem[11,2] = 0
 accuracy_8_0: - 1698.9296875 frac[8,0] + mem[8,0] = 0
 accuracy_16_2: - 1698.9296875 frac[16,2] + mem[16,2] = 0
 accuracy_2_1: - 1698.9296875 frac[2,1] + mem[2,1] = 0
 accuracy_15_1: - 1698.9296875 frac[15,1] + mem[15,1] = 0
 accuracy_6_2: - 1698.9296875 frac[6,2] + mem[6,2] = 0
 accuracy_14_0: - 1698.9296875 frac[14,0] + mem[14,0] = 0
 accuracy_19_0: - 1698.9296875 frac[19,0] + mem[19,0] = 0
 accuracy_5_1: - 1698.9296875 frac[5,1] + mem[5,1] = 0
 accuracy_7_2: - 1698.9296875 frac[7,2] + mem[7,2] = 0
 accuracy_4_0: - 1698.9296875 frac[4,0] + mem[4,0] = 0
 accuracy_1_2: - 1698.9296875 frac[1,2] + mem[1,2] = 0
 accuracy_12_2: - 1698.9296875 frac[12,2] + mem[12,2] = 0
 accuracy_9_0: - 1698.9296875 frac[9,0] + mem[9,0] = 0
 accuracy_18_1: - 1698.9296875 frac[18,1] + mem[18,1] = 0
 accuracy_8_1: - 1698.9296875 frac[8,1] + mem[8,1] = 0
 accuracy_15_0: - 1698.9296875 frac[15,0] + mem[15,0] = 0
 accuracy_20_0: - 1698.9296875 frac[20,0] + mem[20,0] = 0
 accuracy_17_2: - 1698.9296875 frac[17,2] + mem[17,2] = 0
 accuracy_14_1: - 1698.9296875 frac[14,1] + mem[14,1] = 0
 accuracy_11_1: - 1698.9296875 frac[11,1] + mem[11,1] = 0
 accuracy_5_0: - 1698.9296875 frac[5,0] + mem[5,0] = 0
 accuracy_2_2: - 1698.9296875 frac[2,2] + mem[2,2] = 0
 accuracy_10_0: - 1698.9296875 frac[10,0] + mem[10,0] = 0
 accuracy_4_1: - 1698.9296875 frac[4,1] + mem[4,1] = 0
 accuracy_1_1: - 1698.9296875 frac[1,1] + mem[1,1] = 0
 accuracy_3_2: - 1698.9296875 frac[3,2] + mem[3,2] = 0
 accuracy_0_0: - 1698.9296875 frac[0,0] + mem[0,0] = 0
 accuracy_13_2: - 1698.9296875 frac[13,2] + mem[13,2] = 0
 accuracy_8_2: - 1698.9296875 frac[8,2] + mem[8,2] = 0
 accuracy_7_1: - 1698.9296875 frac[7,1] + mem[7,1] = 0
 accuracy_18_2: - 1698.9296875 frac[18,2] + mem[18,2] = 0
 accuracy_20_1: - 1698.9296875 frac[20,1] + mem[20,1] = 0
 accuracy_6_0: - 1698.9296875 frac[6,0] + mem[6,0] = 0
 accuracy_17_1: - 1698.9296875 frac[17,1] + mem[17,1] = 0
 accuracy_11_0: - 1698.9296875 frac[11,0] + mem[11,0] = 0
 accuracy_19_2: - 1698.9296875 frac[19,2] + mem[19,2] = 0
 accuracy_16_0: - 1698.9296875 frac[16,0] + mem[16,0] = 0
 accuracy_10_1: - 1698.9296875 frac[10,1] + mem[10,1] = 0
 accuracy_4_2: - 1698.9296875 frac[4,2] + mem[4,2] = 0
 accuracy_1_0: - 1698.9296875 frac[1,0] + mem[1,0] = 0
 accuracy_14_2: - 1698.9296875 frac[14,2] + mem[14,2] = 0
 accuracy_0_1: - 1698.9296875 frac[0,1] + mem[0,1] = 0
 accuracy_13_1: - 1698.9296875 frac[13,1] + mem[13,1] = 0
 accuracy_7_0: - 1698.9296875 frac[7,0] + mem[7,0] = 0
 accuracy_15_2: - 1698.9296875 frac[15,2] + mem[15,2] = 0
 accuracy_12_0: - 1698.9296875 frac[12,0] + mem[12,0] = 0
 accuracy_9_2: - 1698.9296875 frac[9,2] + mem[9,2] = 0
 accuracy_20_2: - 1698.9296875 frac[20,2] + mem[20,2] = 0
 accuracy_6_1: - 1698.9296875 frac[6,1] + mem[6,1] = 0
 accuracy_3_1: - 1698.9296875 frac[3,1] + mem[3,1] = 0
 accuracy_17_0: - 1698.9296875 frac[17,0] + mem[17,0] = 0
 accuracy_16_1: - 1698.9296875 frac[16,1] + mem[16,1] = 0
 accuracy_2_0: - 1698.9296875 frac[2,0] + mem[2,0] = 0
 accuracy_19_1: - 1698.9296875 frac[19,1] + mem[19,1] = 0
 accuracy_5_2: - 1698.9296875 frac[5,2] + mem[5,2] = 0
 accuracy_0_2: - 1698.9296875 frac[0,2] + mem[0,2] = 0
 accuracy_13_0: - 1698.9296875 frac[13,0] + mem[13,0] = 0
 accuracy_10_2: - 1698.9296875 frac[10,2] + mem[10,2] = 0
 mem_tot_P4_0: - mem[0,0] - mem[0,1] - mem[0,2] + mem_tot_P4_0 = 0
 rows_tot_P4_0: - frac[0,0] - frac[0,1] - frac[0,2] + rows_tot_P4_0 = 0
 mem_tot_P4_1: - mem[1,2] - mem[1,1] - mem[1,0] + mem_tot_P4_1 = 0
 rows_tot_P4_1: - frac[1,2] - frac[1,1] - frac[1,0] + rows_tot_P4_1 = 0
 mem_tot_P4_2: - mem[2,1] - mem[2,2] - mem[2,0] + mem_tot_P4_2 = 0
 rows_tot_P4_2: - frac[2,1] - frac[2,2] - frac[2,0] + rows_tot_P4_2 = 0
 mem_tot_P4_3: - mem[3,0] - mem[3,2] - mem[3,1] + mem_tot_P4_3 = 0
 rows_tot_P4_3: - frac[3,0] - frac[3,2] - frac[3,1] + rows_tot_P4_3 = 0
 mem_tot_P4_4: - mem[4,0] - mem[4,1] - mem[4,2] + mem_tot_P4_4 = 0
 rows_tot_P4_4: - frac[4,0] - frac[4,1] - frac[4,2] + rows_tot_P4_4 = 0
 mem_tot_P4_5: - mem[5,1] - mem[5,0] - mem[5,2] + mem_tot_P4_5 = 0
 rows_tot_P4_5: - frac[5,1] - frac[5,0] - frac[5,2] + rows_tot_P4_5 = 0
 mem_tot_P4_6: - mem[6,2] - mem[6,0] - mem[6,1] + mem_tot_P4_6 = 0
 rows_tot_P4_6: - frac[6,2] - frac[6,0] - frac[6,1] + rows_tot_P4_6 = 0
 mem_tot_P4_7: - mem[7,2] - mem[7,1] - mem[7,0] + mem_tot_P4_7 = 0
 rows_tot_P4_7: - frac[7,2] - frac[7,1] - frac[7,0] + rows_tot_P4_7 = 0
 mem_tot_P4_8: - mem[8,0] - mem[8,1] - mem[8,2] + mem_tot_P4_8 = 0
 rows_tot_P4_8: - frac[8,0] - frac[8,1] - frac[8,2] + rows_tot_P4_8 = 0
 mem_tot_P4_9: - mem[9,1] - mem[9,0] - mem[9,2] + mem_tot_P4_9 = 0
 rows_tot_P4_9: - frac[9,1] - frac[9,0] - frac[9,2] + rows_tot_P4_9 = 0
 mem_tot_P4_10: - mem[10,0] - mem[10,1] - mem[10,2] + mem_tot_P4_10 = 0
 rows_tot_P4_10: - frac[10,0] - frac[10,1] - frac[10,2] + rows_tot_P4_10
   = 0
 mem_tot_P4_11: - mem[11,2] - mem[11,1] - mem[11,0] + mem_tot_P4_11 = 0
 rows_tot_P4_11: - frac[11,2] - frac[11,1] - frac[11,0] + rows_tot_P4_11
   = 0
 mem_tot_P4_12: - mem[12,1] - mem[12,2] - mem[12,0] + mem_tot_P4_12 = 0
 rows_tot_P4_12: - frac[12,1] - frac[12,2] - frac[12,0] + rows_tot_P4_12
   = 0
 mem_tot_P4_13: - mem[13,2] - mem[13,1] - mem[13,0] + mem_tot_P4_13 = 0
 rows_tot_P4_13: - frac[13,2] - frac[13,1] - frac[13,0] + rows_tot_P4_13
   = 0
 mem_tot_P4_14: - mem[14,0] - mem[14,1] - mem[14,2] + mem_tot_P4_14 = 0
 rows_tot_P4_14: - frac[14,0] - frac[14,1] - frac[14,2] + rows_tot_P4_14
   = 0
 mem_tot_P4_15: - mem[15,1] - mem[15,0] - mem[15,2] + mem_tot_P4_15 = 0
 rows_tot_P4_15: - frac[15,1] - frac[15,0] - frac[15,2] + rows_tot_P4_15
   = 0
 mem_tot_P4_16: - mem[16,2] - mem[16,0] - mem[16,1] + mem_tot_P4_16 = 0
 rows_tot_P4_16: - frac[16,2] - frac[16,0] - frac[16,1] + rows_tot_P4_16
   = 0
 mem_tot_P4_17: - mem[17,2] - mem[17,1] - mem[17,0] + mem_tot_P4_17 = 0
 rows_tot_P4_17: - frac[17,2] - frac[17,1] - frac[17,0] + rows_tot_P4_17
   = 0
 mem_tot_P4_18: - mem[18,0] - mem[18,1] - mem[18,2] + mem_tot_P4_18 = 0
 rows_tot_P4_18: - frac[18,0] - frac[18,1] - frac[18,2] + rows_tot_P4_18
   = 0
 mem_tot_P4_19: - mem[19,0] - mem[19,2] - mem[19,1] + mem_tot_P4_19 = 0
 rows_tot_P4_19: - frac[19,0] - frac[19,2] - frac[19,1] + rows_tot_P4_19
   = 0
 mem_tot_CPU_1: - mem[20,0] - mem[20,1] - mem[20,2] + mem_tot_CPU_1 = 0
 rows_tot_CPU_1: - frac[20,0] - frac[20,1] - frac[20,2] + rows_tot_CPU_1
   = 0
 tot_mem: - mem_tot_P4_0 - mem_tot_P4_1 - mem_tot_P4_2 - mem_tot_P4_3
   - mem_tot_P4_4 - mem_tot_P4_5 - mem_tot_P4_6 - mem_tot_P4_7
   - mem_tot_P4_8 - mem_tot_P4_9 - mem_tot_P4_10 - mem_tot_P4_11
   - mem_tot_P4_12 - mem_tot_P4_13 - mem_tot_P4_14 - mem_tot_P4_15
   - mem_tot_P4_16 - mem_tot_P4_17 - mem_tot_P4_18 - mem_tot_P4_19
   - mem_tot_CPU_1 + tot_mem = 0
 tot_rows: - rows_tot_P4_0 - rows_tot_P4_1 - rows_tot_P4_2 - rows_tot_P4_3
   - rows_tot_P4_4 - rows_tot_P4_5 - rows_tot_P4_6 - rows_tot_P4_7
   - rows_tot_P4_8 - rows_tot_P4_9 - rows_tot_P4_10 - rows_tot_P4_11
   - rows_tot_P4_12 - rows_tot_P4_13 - rows_tot_P4_14 - rows_tot_P4_15
   - rows_tot_P4_16 - rows_tot_P4_17 - rows_tot_P4_18 - rows_tot_P4_19
   - rows_tot_CPU_1 + tot_rows = 0
Bounds
 frac[18,0] <= 1
 frac[12,1] <= 1
 frac[9,1] <= 1
 frac[3,0] <= 1
 frac[11,2] <= 1
 frac[8,0] <= 1
 frac[16,2] <= 1
 frac[2,1] <= 1
 frac[15,1] <= 1
 frac[6,2] <= 1
 frac[14,0] <= 1
 frac[19,0] <= 1
 frac[5,1] <= 1
 frac[7,2] <= 1
 frac[4,0] <= 1
 frac[1,2] <= 1
 frac[12,2] <= 1
 frac[9,0] <= 1
 frac[18,1] <= 1
 frac[8,1] <= 1
 frac[15,0] <= 1
 frac[20,0] <= 1
 frac[17,2] <= 1
 frac[14,1] <= 1
 frac[11,1] <= 1
 frac[5,0] <= 1
 frac[2,2] <= 1
 frac[10,0] <= 1
 frac[4,1] <= 1
 frac[1,1] <= 1
 frac[3,2] <= 1
 frac[0,0] <= 1
 frac[13,2] <= 1
 frac[8,2] <= 1
 frac[7,1] <= 1
 frac[18,2] <= 1
 frac[20,1] <= 1
 frac[6,0] <= 1
 frac[17,1] <= 1
 frac[11,0] <= 1
 frac[19,2] <= 1
 frac[16,0] <= 1
 frac[10,1] <= 1
 frac[4,2] <= 1
 frac[1,0] <= 1
 frac[14,2] <= 1
 frac[0,1] <= 1
 frac[13,1] <= 1
 frac[7,0] <= 1
 frac[15,2] <= 1
 frac[12,0] <= 1
 frac[9,2] <= 1
 frac[20,2] <= 1
 frac[6,1] <= 1
 frac[3,1] <= 1
 frac[17,0] <= 1
 frac[16,1] <= 1
 frac[2,0] <= 1
 frac[19,1] <= 1
 frac[5,2] <= 1
 frac[0,2] <= 1
 frac[13,0] <= 1
 frac[10,2] <= 1
 mem_tot_P4_0 <= 576
 mem_tot_P4_1 <= 576
 mem_tot_P4_2 <= 576
 mem_tot_P4_3 <= 576
 mem_tot_P4_4 <= 576
 mem_tot_P4_5 <= 576
 mem_tot_P4_6 <= 576
 mem_tot_P4_7 <= 576
 mem_tot_P4_8 <= 576
 mem_tot_P4_9 <= 576
 mem_tot_P4_10 <= 576
 mem_tot_P4_11 <= 576
 mem_tot_P4_12 <= 576
 mem_tot_P4_13 <= 576
 mem_tot_P4_14 <= 576
 mem_tot_P4_15 <= 576
 mem_tot_P4_16 <= 576
 mem_tot_P4_17 <= 576
 mem_tot_P4_18 <= 576
 mem_tot_P4_19 <= 576
 mem_tot_CPU_1 <= 32768
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_P4_0 , mem_tot_P4_1 ,
   mem_tot_P4_2 , mem_tot_P4_3 , mem_tot_P4_4 , mem_tot_P4_5 ,
   mem_tot_P4_6 , mem_tot_P4_7 , mem_tot_P4_8 , mem_tot_P4_9 ,
   mem_tot_P4_10 , mem_tot_P4_11 , mem_tot_P4_12 , mem_tot_P4_13 ,
   mem_tot_P4_14 , mem_tot_P4_15 , mem_tot_P4_16 , mem_tot_P4_17 ,
   mem_tot_P4_18 , mem_tot_P4_19 )
 mem_overall_CPU: max_mem_CPU = MAX ( mem_tot_CPU_1 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_P4_0 , rows_tot_P4_1 ,
   rows_tot_P4_2 , rows_tot_P4_3 , rows_tot_P4_4 , rows_tot_P4_5 ,
   rows_tot_P4_6 , rows_tot_P4_7 , rows_tot_P4_8 , rows_tot_P4_9 ,
   rows_tot_P4_10 , rows_tot_P4_11 , rows_tot_P4_12 , rows_tot_P4_13 ,
   rows_tot_P4_14 , rows_tot_P4_15 , rows_tot_P4_16 , rows_tot_P4_17 ,
   rows_tot_P4_18 , rows_tot_P4_19 )
 rows_overall_CPU: max_rows_CPU = MAX ( rows_tot_CPU_1 )
End

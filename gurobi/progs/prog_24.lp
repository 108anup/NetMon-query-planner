\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_P4 + 0 max_mem_CPU_Cluster + 0 max_rows_P4
   + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU2 + 0 m_access_time_CPU4
   + 0 m_access_time_CPU13 + 0 m_access_time_CPU18 + 0 m_access_time_CPU21
   + 0 m_access_time_CPU24 + 0 m_access_time_CPU26 + 0 m_access_time_CPU32
   + res
Subject To
 cov_0_8: frac[8,8] + frac[9,8] >= 0.8
 cov_1_30: frac[8,30] + frac[9,30] >= 0.7
 cov_2_25: frac[9,25] >= 0.7
 cov_3_3: frac[8,3] + frac[9,3] >= 0.1
 cov_4_21: frac[8,21] + frac[9,21] >= 0.1
 cov_5_24: frac[8,24] + frac[9,24] >= 0.8
 cov_6_17: frac[4,17] + frac[8,17] + frac[9,17] >= 0.7
 cov_7_8: frac[8,8] + frac[9,8] >= 0.8
 cov_8_24: frac[8,24] + frac[9,24] >= 0.8
 cov_9_36: frac[8,36] + frac[9,36] >= 0.2
 cov_10_21: frac[8,21] + frac[9,21] >= 0.1
 cov_11_10: frac[8,10] >= 0.726
 cov_12_25: frac[8,25] + frac[9,25] >= 0.7
 cov_13_1: frac[9,1] >= 0.8
 cov_14_36: frac[8,36] + frac[9,36] >= 0.2
 cov_15_3: frac[9,3] >= 0.1
 cov_16_1: frac[8,1] + frac[9,1] >= 1
 cov_17_1: frac[9,1] >= 0.8
 cov_18_25: frac[8,25] + frac[9,25] >= 0.7
 cov_19_17: frac[8,17] >= 1
 cov_20_1: frac[8,1] + frac[9,1] >= 0.7
 cov_21_30: frac[8,30] + frac[9,30] >= 0.9
 cov_22_11: frac[8,11] + frac[9,11] >= 0.1
 cov_23_11: frac[8,11] + frac[9,11] >= 0.1
 cov_24_11: frac[2,11] >= 0.1
 cov_25_11: frac[9,11] >= 0.1
 cov_26_34: frac[8,34] + frac[9,34] >= 0.8
 cov_27_34: frac[9,34] >= 0.8
 cov_28_18: frac[8,18] + frac[9,18] >= 0.2
 cov_29_22: frac[8,22] + frac[9,22] >= 0.1
 cov_30_13: frac[9,13] >= 0.2
 cov_31_19: frac[8,19] + frac[9,19] >= 0.1
 cov_32_31: frac[9,31] >= 0.8
 cov_33_26: frac[8,26] + frac[9,26] >= 0.8
 cov_34_26: frac[9,26] >= 0.7
 cov_35_13: frac[8,13] + frac[9,13] >= 0.2
 cov_36_31: frac[8,31] + frac[9,31] >= 0.8
 cov_37_13: frac[8,13] + frac[9,13] >= 0.2
 cov_38_15: frac[8,15] + frac[9,15] >= 0.8
 cov_39_28: frac[4,28] + frac[7,28] + frac[8,28] + frac[9,28] >= 1
 cov_40_16: frac[2,16] + frac[6,16] + frac[9,16] >= 0.9
 cov_41_16: frac[5,16] + frac[8,16] + frac[9,16] >= 0.7
 cov_42_2: frac[4,2] + frac[7,2] + frac[8,2] + frac[9,2] >= 0.8
 cov_43_2: frac[2,2] + frac[4,2] + frac[8,2] + frac[9,2] >= 0.7
 cov_44_27: frac[0,27] + frac[1,27] >= 0.9
 cov_45_20: frac[6,20] + frac[7,20] >= 0.7
 cov_46_32: frac[0,32] + frac[4,32] + frac[8,32] + frac[9,32] >= 0.9
 cov_47_16: frac[0,16] + frac[2,16] >= 0.8
 cov_48_14: frac[1,14] + frac[7,14] + frac[9,14] >= 1
 cov_49_33: frac[0,33] >= 0.9
 cov_50_12: frac[1,12] + frac[2,12] >= 0.8
 cov_51_7: frac[0,7] + frac[1,7] >= 0.9
 cov_52_27: frac[3,27] + frac[7,27] + frac[8,27] + frac[9,27] >= 0.8
 cov_53_29: frac[2,29] + frac[3,29] + frac[8,29] + frac[9,29] >= 0.7
 cov_54_33: frac[3,33] + frac[7,33] + frac[8,33] + frac[9,33] >= 0.7
 cov_55_27: frac[2,27] + frac[4,27] + frac[8,27] + frac[9,27] >= 0.9
 cov_56_9: frac[6,9] >= 0.8
 cov_57_32: frac[0,32] + frac[1,32] >= 0.9
 cov_58_6: frac[0,6] + frac[1,6] >= 0.9
 cov_59_6: frac[2,6] + frac[3,6] + frac[8,6] + frac[9,6] >= 0.9
 cov_60_5: frac[3,5] + frac[7,5] + frac[8,5] + frac[9,5] >= 0.7
 cov_61_12: frac[2,12] + frac[5,12] + frac[8,12] + frac[9,12] >= 0.8
 cov_62_2: frac[4,2] + frac[7,2] + frac[8,2] + frac[9,2] >= 0.9
 cov_63_35: frac[0,35] + frac[1,35] >= 0.8
 cov_64_29: frac[1,29] + frac[3,29] + frac[8,29] + frac[9,29] >= 0.7
 cov_65_2: frac[4,2] + frac[7,2] + frac[8,2] + frac[9,2] >= 0.7
 cov_66_29: frac[1,29] + frac[6,29] + frac[9,29] >= 0.7
 cov_67_0: frac[0,0] >= 0.8
 cov_68_7: frac[1,7] + frac[4,7] + frac[8,7] + frac[9,7] >= 0.7
 cov_69_27: frac[1,27] + frac[4,27] + frac[8,27] + frac[9,27] >= 0.9
 cov_70_2: frac[2,2] + frac[7,2] + frac[9,2] >= 0.9
 cov_71_23: frac[2,23] + frac[5,23] + frac[8,23] + frac[9,23] >= 0.7
 cov_72_27: frac[0,27] + frac[5,27] + frac[8,27] + frac[9,27] >= 0.8
 cov_73_5: frac[2,5] + frac[4,5] + frac[8,5] + frac[9,5] >= 1
 cov_74_20: frac[4,20] + frac[7,20] + frac[8,20] + frac[9,20] >= 0.8
 cov_75_14: frac[0,14] + frac[5,14] + frac[8,14] + frac[9,14] >= 0.7
 cov_76_35: frac[2,35] + frac[3,35] + frac[8,35] + frac[9,35] >= 0.7
 cov_77_5: frac[4,5] + frac[8,5] >= 1
 cov_78_4: frac[2,4] + frac[3,4] + frac[8,4] + frac[9,4] >= 0.8
 cov_79_23: frac[0,23] + frac[5,23] + frac[8,23] + frac[9,23] >= 1
 cov_80_32: frac[3,32] + frac[6,32] + frac[8,32] + frac[9,32] >= 0.7
 cov_81_23: frac[3,23] + frac[5,23] + frac[8,23] >= 1
 cov_82_0: frac[3,0] + frac[7,0] + frac[8,0] + frac[9,0] >= 0.7
 cov_83_32: frac[2,32] + frac[7,32] + frac[9,32] >= 0.9
 cov_84_35: frac[3,35] + frac[4,35] + frac[8,35] >= 0.8
 cov_85_12: frac[4,12] + frac[7,12] + frac[8,12] + frac[9,12] >= 0.7
 cov_86_20: frac[2,20] + frac[3,20] + frac[8,20] + frac[9,20] >= 0.8
 cov_87_12: frac[4,12] + frac[5,12] + frac[8,12] >= 0.8
 cov_88_16: frac[5,16] + frac[8,16] + frac[9,16] >= 0.9
 cov_89_14: frac[1,14] + frac[3,14] + frac[8,14] + frac[9,14] >= 0.9
 cov_90_4: frac[5,4] + frac[8,4] + frac[9,4] >= 0.8
 cov_91_23: frac[1,23] + frac[3,23] + frac[8,23] + frac[9,23] >= 1
 cov_92_5: frac[2,5] + frac[8,5] + frac[9,5] >= 0.8
 accuracy_9_26: - 6 frac[9,26] + mem[9,26] = 0
 accuracy_6_9: - 5.09765625 frac[6,9] + mem[6,9] = 0
 accuracy_0_35: - 5.09765625 frac[0,35] + mem[0,35] = 0
 accuracy_3_35: - 5.09765625 frac[3,35] + mem[3,35] = 0
 accuracy_7_12: - 5.09765625 frac[7,12] + mem[7,12] = 0
 accuracy_0_7: - 5.09765625 frac[0,7] + mem[0,7] = 0
 accuracy_1_6: - 5.09765625 frac[1,6] + mem[1,6] = 0
 accuracy_2_5: - 5.09765625 frac[2,5] + mem[2,5] = 0
 accuracy_8_5: - 6 frac[8,5] + mem[8,5] = 0
 accuracy_9_0: - 6 frac[9,0] + mem[9,0] = 0
 accuracy_8_24: - 6 frac[8,24] + mem[8,24] = 0
 accuracy_0_32: - 5.09765625 frac[0,32] + mem[0,32] = 0
 accuracy_9_21: - 6 frac[9,21] + mem[9,21] = 0
 accuracy_9_36: - 6 frac[9,36] + mem[9,36] = 0
 accuracy_7_28: - 5.09765625 frac[7,28] + mem[7,28] = 0
 accuracy_2_27: - 5.09765625 frac[2,27] + mem[2,27] = 0
 accuracy_8_15: - 6 frac[8,15] + mem[8,15] = 0
 accuracy_2_6: - 5.09765625 frac[2,6] + mem[2,6] = 0
 accuracy_9_14: - 6 frac[9,14] + mem[9,14] = 0
 accuracy_8_2: - 6 frac[8,2] + mem[8,2] = 0
 accuracy_4_5: - 5.09765625 frac[4,5] + mem[4,5] = 0
 accuracy_3_23: - 5.09765625 frac[3,23] + mem[3,23] = 0
 accuracy_9_3: - 6 frac[9,3] + mem[9,3] = 0
 accuracy_8_21: - 6 frac[8,21] + mem[8,21] = 0
 accuracy_2_32: - 5.09765625 frac[2,32] + mem[2,32] = 0
 accuracy_9_16: - 6 frac[9,16] + mem[9,16] = 0
 accuracy_7_5: - 5.09765625 frac[7,5] + mem[7,5] = 0
 accuracy_4_35: - 5.09765625 frac[4,35] + mem[4,35] = 0
 accuracy_1_12: - 5.09765625 frac[1,12] + mem[1,12] = 0
 accuracy_8_12: - 6 frac[8,12] + mem[8,12] = 0
 accuracy_2_11: - 5.09765625 frac[2,11] + mem[2,11] = 0
 accuracy_8_31: - 6 frac[8,31] + mem[8,31] = 0
 accuracy_5_14: - 5.09765625 frac[5,14] + mem[5,14] = 0
 accuracy_9_30: - 6 frac[9,30] + mem[9,30] = 0
 accuracy_8_18: - 6 frac[8,18] + mem[8,18] = 0
 accuracy_5_27: - 5.09765625 frac[5,27] + mem[5,27] = 0
 accuracy_9_19: - 6 frac[9,19] + mem[9,19] = 0
 accuracy_6_16: - 5.09765625 frac[6,16] + mem[6,16] = 0
 accuracy_4_32: - 5.09765625 frac[4,32] + mem[4,32] = 0
 accuracy_0_27: - 5.09765625 frac[0,27] + mem[0,27] = 0
 accuracy_8_35: - 6 frac[8,35] + mem[8,35] = 0
 accuracy_9_34: - 6 frac[9,34] + mem[9,34] = 0
 accuracy_0_14: - 5.09765625 frac[0,14] + mem[0,14] = 0
 accuracy_4_12: - 5.09765625 frac[4,12] + mem[4,12] = 0
 accuracy_2_12: - 5.09765625 frac[2,12] + mem[2,12] = 0
 accuracy_9_4: - 6 frac[9,4] + mem[9,4] = 0
 accuracy_8_28: - 6 frac[8,28] + mem[8,28] = 0
 accuracy_9_25: - 6 frac[9,25] + mem[9,25] = 0
 accuracy_7_2: - 5.09765625 frac[7,2] + mem[7,2] = 0
 accuracy_6_29: - 5.09765625 frac[6,29] + mem[6,29] = 0
 accuracy_8_32: - 6 frac[8,32] + mem[8,32] = 0
 accuracy_6_32: - 5.09765625 frac[6,32] + mem[6,32] = 0
 accuracy_2_23: - 5.09765625 frac[2,23] + mem[2,23] = 0
 accuracy_3_6: - 5.09765625 frac[3,6] + mem[3,6] = 0
 accuracy_2_2: - 5.09765625 frac[2,2] + mem[2,2] = 0
 accuracy_8_6: - 6 frac[8,6] + mem[8,6] = 0
 accuracy_3_27: - 5.09765625 frac[3,27] + mem[3,27] = 0
 accuracy_9_7: - 6 frac[9,7] + mem[9,7] = 0
 accuracy_8_25: - 6 frac[8,25] + mem[8,25] = 0
 accuracy_0_33: - 5.09765625 frac[0,33] + mem[0,33] = 0
 accuracy_5_4: - 5.09765625 frac[5,4] + mem[5,4] = 0
 accuracy_4_28: - 5.09765625 frac[4,28] + mem[4,28] = 0
 accuracy_9_20: - 6 frac[9,20] + mem[9,20] = 0
 accuracy_3_33: - 5.09765625 frac[3,33] + mem[3,33] = 0
 accuracy_3_5: - 5.09765625 frac[3,5] + mem[3,5] = 0
 accuracy_9_13: - 6 frac[9,13] + mem[9,13] = 0
 accuracy_8_3: - 6 frac[8,3] + mem[8,3] = 0
 accuracy_9_2: - 6 frac[9,2] + mem[9,2] = 0
 accuracy_8_22: - 6 frac[8,22] + mem[8,22] = 0
 accuracy_4_17: - 5.09765625 frac[4,17] + mem[4,17] = 0
 accuracy_9_23: - 6 frac[9,23] + mem[9,23] = 0
 accuracy_6_20: - 5.09765625 frac[6,20] + mem[6,20] = 0
 accuracy_2_29: - 5.09765625 frac[2,29] + mem[2,29] = 0
 accuracy_8_13: - 6 frac[8,13] + mem[8,13] = 0
 accuracy_3_0: - 5.09765625 frac[3,0] + mem[3,0] = 0
 accuracy_9_8: - 6 frac[9,8] + mem[9,8] = 0
 accuracy_8_0: - 6 frac[8,0] + mem[8,0] = 0
 accuracy_4_27: - 5.09765625 frac[4,27] + mem[4,27] = 0
 accuracy_9_29: - 6 frac[9,29] + mem[9,29] = 0
 accuracy_8_19: - 6 frac[8,19] + mem[8,19] = 0
 accuracy_9_18: - 6 frac[9,18] + mem[9,18] = 0
 accuracy_5_23: - 5.09765625 frac[5,23] + mem[5,23] = 0
 accuracy_8_36: - 6 frac[8,36] + mem[8,36] = 0
 accuracy_7_20: - 5.09765625 frac[7,20] + mem[7,20] = 0
 accuracy_9_33: - 6 frac[9,33] + mem[9,33] = 0
 accuracy_7_33: - 5.09765625 frac[7,33] + mem[7,33] = 0
 accuracy_1_14: - 5.09765625 frac[1,14] + mem[1,14] = 0
 accuracy_8_10: - 6 frac[8,10] + mem[8,10] = 0
 accuracy_9_11: - 6 frac[9,11] + mem[9,11] = 0
 accuracy_8_29: - 6 frac[8,29] + mem[8,29] = 0
 accuracy_9_24: - 6 frac[9,24] + mem[9,24] = 0
 accuracy_8_16: - 6 frac[8,16] + mem[8,16] = 0
 accuracy_1_32: - 5.09765625 frac[1,32] + mem[1,32] = 0
 accuracy_7_14: - 5.09765625 frac[7,14] + mem[7,14] = 0
 accuracy_1_23: - 5.09765625 frac[1,23] + mem[1,23] = 0
 accuracy_8_33: - 6 frac[8,33] + mem[8,33] = 0
 accuracy_7_27: - 5.09765625 frac[7,27] + mem[7,27] = 0
 accuracy_2_20: - 5.09765625 frac[2,20] + mem[2,20] = 0
 accuracy_8_7: - 6 frac[8,7] + mem[8,7] = 0
 accuracy_4_2: - 5.09765625 frac[4,2] + mem[4,2] = 0
 accuracy_9_6: - 6 frac[9,6] + mem[9,6] = 0
 accuracy_8_26: - 6 frac[8,26] + mem[8,26] = 0
 accuracy_9_27: - 6 frac[9,27] + mem[9,27] = 0
 accuracy_1_35: - 5.09765625 frac[1,35] + mem[1,35] = 0
 accuracy_7_0: - 5.09765625 frac[7,0] + mem[7,0] = 0
 accuracy_5_16: - 5.09765625 frac[5,16] + mem[5,16] = 0
 accuracy_3_32: - 5.09765625 frac[3,32] + mem[3,32] = 0
 accuracy_0_6: - 5.09765625 frac[0,6] + mem[0,6] = 0
 accuracy_1_7: - 5.09765625 frac[1,7] + mem[1,7] = 0
 accuracy_3_4: - 5.09765625 frac[3,4] + mem[3,4] = 0
 accuracy_2_4: - 5.09765625 frac[2,4] + mem[2,4] = 0
 accuracy_9_12: - 6 frac[9,12] + mem[9,12] = 0
 accuracy_8_4: - 6 frac[8,4] + mem[8,4] = 0
 accuracy_4_7: - 5.09765625 frac[4,7] + mem[4,7] = 0
 accuracy_9_1: - 6 frac[9,1] + mem[9,1] = 0
 accuracy_8_23: - 6 frac[8,23] + mem[8,23] = 0
 accuracy_9_22: - 6 frac[9,22] + mem[9,22] = 0
 accuracy_0_16: - 5.09765625 frac[0,16] + mem[0,16] = 0
 accuracy_1_29: - 5.09765625 frac[1,29] + mem[1,29] = 0
 accuracy_3_14: - 5.09765625 frac[3,14] + mem[3,14] = 0
 accuracy_8_14: - 6 frac[8,14] + mem[8,14] = 0
 accuracy_9_15: - 6 frac[9,15] + mem[9,15] = 0
 accuracy_8_1: - 6 frac[8,1] + mem[8,1] = 0
 accuracy_5_12: - 5.09765625 frac[5,12] + mem[5,12] = 0
 accuracy_3_20: - 5.09765625 frac[3,20] + mem[3,20] = 0
 accuracy_9_28: - 6 frac[9,28] + mem[9,28] = 0
 accuracy_8_20: - 6 frac[8,20] + mem[8,20] = 0
 accuracy_2_35: - 5.09765625 frac[2,35] + mem[2,35] = 0
 accuracy_9_17: - 6 frac[9,17] + mem[9,17] = 0
 accuracy_2_16: - 5.09765625 frac[2,16] + mem[2,16] = 0
 accuracy_9_32: - 6 frac[9,32] + mem[9,32] = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 accuracy_7_32: - 5.09765625 frac[7,32] + mem[7,32] = 0
 accuracy_8_11: - 6 frac[8,11] + mem[8,11] = 0
 accuracy_8_30: - 6 frac[8,30] + mem[8,30] = 0
 accuracy_9_31: - 6 frac[9,31] + mem[9,31] = 0
 accuracy_8_17: - 6 frac[8,17] + mem[8,17] = 0
 accuracy_4_20: - 5.09765625 frac[4,20] + mem[4,20] = 0
 accuracy_0_23: - 5.09765625 frac[0,23] + mem[0,23] = 0
 accuracy_8_34: - 6 frac[8,34] + mem[8,34] = 0
 accuracy_1_27: - 5.09765625 frac[1,27] + mem[1,27] = 0
 accuracy_9_35: - 6 frac[9,35] + mem[9,35] = 0
 accuracy_8_8: - 6 frac[8,8] + mem[8,8] = 0
 accuracy_3_29: - 5.09765625 frac[3,29] + mem[3,29] = 0
 accuracy_9_5: - 6 frac[9,5] + mem[9,5] = 0
 accuracy_8_27: - 6 frac[8,27] + mem[8,27] = 0
 mem_tot_CPU2: - mem[0,35] - mem[0,7] - mem[0,32] - mem[0,27] - mem[0,14]
   - mem[0,33] - mem[0,6] - mem[0,16] - mem[0,0] - mem[0,23] + mem_tot_CPU2
   = 0
 R240: - 3.05176e-05 mem_tot_CPU2 + normalized_mem_tot_CPU2 = 0
 rows_tot_CPU2: - 3 frac[0,35] - 3 frac[0,7] - 3 frac[0,32] - 3 frac[0,27]
   - 3 frac[0,14] - 3 frac[0,33] - 3 frac[0,6] - 3 frac[0,16] - 3 frac[0,0]
   - 3 frac[0,23] + rows_tot_CPU2 = 0
 R242: - 0.0833333333333333 rows_tot_CPU2 + normalized_rows_tot_CPU2 = 0
 mem_tot_CPU4: - mem[1,6] - mem[1,12] - mem[1,14] - mem[1,32] - mem[1,23]
   - mem[1,35] - mem[1,7] - mem[1,29] - mem[1,27] + mem_tot_CPU4 = 0
 R244: - 3.05176e-05 mem_tot_CPU4 + normalized_mem_tot_CPU4 = 0
 rows_tot_CPU4: - 3 frac[1,6] - 3 frac[1,12] - 3 frac[1,14] - 3 frac[1,32]
   - 3 frac[1,23] - 3 frac[1,35] - 3 frac[1,7] - 3 frac[1,29]
   - 3 frac[1,27] + rows_tot_CPU4 = 0
 R246: - 0.0833333333333333 rows_tot_CPU4 + normalized_rows_tot_CPU4 = 0
 mem_tot_CPU13: - mem[2,5] - mem[2,27] - mem[2,6] - mem[2,32] - mem[2,11]
   - mem[2,12] - mem[2,23] - mem[2,2] - mem[2,29] - mem[2,20] - mem[2,4]
   - mem[2,35] - mem[2,16] + mem_tot_CPU13 = 0
 R248: - 3.05176e-05 mem_tot_CPU13 + normalized_mem_tot_CPU13 = 0
 rows_tot_CPU13: - 3 frac[2,5] - 3 frac[2,27] - 3 frac[2,6] - 3 frac[2,32]
   - 3 frac[2,11] - 3 frac[2,12] - 3 frac[2,23] - 3 frac[2,2]
   - 3 frac[2,29] - 3 frac[2,20] - 3 frac[2,4] - 3 frac[2,35]
   - 3 frac[2,16] + rows_tot_CPU13 = 0
 R250: - 0.0833333333333333 rows_tot_CPU13 + normalized_rows_tot_CPU13 = 0
 mem_tot_CPU18: - mem[3,35] - mem[3,23] - mem[3,6] - mem[3,27] - mem[3,33]
   - mem[3,5] - mem[3,0] - mem[3,32] - mem[3,4] - mem[3,14] - mem[3,20]
   - mem[3,29] + mem_tot_CPU18 = 0
 R252: - 3.05176e-05 mem_tot_CPU18 + normalized_mem_tot_CPU18 = 0
 rows_tot_CPU18: - 3 frac[3,35] - 3 frac[3,23] - 3 frac[3,6] - 3 frac[3,27]
   - 3 frac[3,33] - 3 frac[3,5] - 3 frac[3,0] - 3 frac[3,32] - 3 frac[3,4]
   - 3 frac[3,14] - 3 frac[3,20] - 3 frac[3,29] + rows_tot_CPU18 = 0
 R254: - 0.0833333333333333 rows_tot_CPU18 + normalized_rows_tot_CPU18 = 0
 mem_tot_CPU21: - mem[4,5] - mem[4,35] - mem[4,32] - mem[4,12] - mem[4,28]
   - mem[4,17] - mem[4,27] - mem[4,2] - mem[4,7] - mem[4,20]
   + mem_tot_CPU21 = 0
 R256: - 3.05176e-05 mem_tot_CPU21 + normalized_mem_tot_CPU21 = 0
 rows_tot_CPU21: - 3 frac[4,5] - 3 frac[4,35] - 3 frac[4,32] - 3 frac[4,12]
   - 3 frac[4,28] - 3 frac[4,17] - 3 frac[4,27] - 3 frac[4,2] - 3 frac[4,7]
   - 3 frac[4,20] + rows_tot_CPU21 = 0
 R258: - 0.0833333333333333 rows_tot_CPU21 + normalized_rows_tot_CPU21 = 0
 mem_tot_CPU24: - mem[5,14] - mem[5,27] - mem[5,4] - mem[5,23] - mem[5,16]
   - mem[5,12] + mem_tot_CPU24 = 0
 R260: - 3.05176e-05 mem_tot_CPU24 + normalized_mem_tot_CPU24 = 0
 rows_tot_CPU24: - 3 frac[5,14] - 3 frac[5,27] - 3 frac[5,4] - 3 frac[5,23]
   - 3 frac[5,16] - 3 frac[5,12] + rows_tot_CPU24 = 0
 R262: - 0.0833333333333333 rows_tot_CPU24 + normalized_rows_tot_CPU24 = 0
 mem_tot_CPU26: - mem[6,9] - mem[6,16] - mem[6,29] - mem[6,32] - mem[6,20]
   + mem_tot_CPU26 = 0
 R264: - 3.05176e-05 mem_tot_CPU26 + normalized_mem_tot_CPU26 = 0
 rows_tot_CPU26: - 3 frac[6,9] - 3 frac[6,16] - 3 frac[6,29] - 3 frac[6,32]
   - 3 frac[6,20] + rows_tot_CPU26 = 0
 R266: - 0.0833333333333333 rows_tot_CPU26 + normalized_rows_tot_CPU26 = 0
 mem_tot_CPU32: - mem[7,12] - mem[7,28] - mem[7,5] - mem[7,2] - mem[7,20]
   - mem[7,33] - mem[7,14] - mem[7,27] - mem[7,0] - mem[7,32]
   + mem_tot_CPU32 = 0
 R268: - 3.05176e-05 mem_tot_CPU32 + normalized_mem_tot_CPU32 = 0
 rows_tot_CPU32: - 3 frac[7,12] - 3 frac[7,28] - 3 frac[7,5] - 3 frac[7,2]
   - 3 frac[7,20] - 3 frac[7,33] - 3 frac[7,14] - 3 frac[7,27]
   - 3 frac[7,0] - 3 frac[7,32] + rows_tot_CPU32 = 0
 R270: - 0.0833333333333333 rows_tot_CPU32 + normalized_rows_tot_CPU32 = 0
 mem_tot_tor_P43: - mem[8,5] - mem[8,24] - mem[8,15] - mem[8,2] - mem[8,21]
   - mem[8,12] - mem[8,31] - mem[8,18] - mem[8,35] - mem[8,28] - mem[8,32]
   - mem[8,6] - mem[8,25] - mem[8,3] - mem[8,22] - mem[8,13] - mem[8,0]
   - mem[8,19] - mem[8,36] - mem[8,10] - mem[8,29] - mem[8,16] - mem[8,33]
   - mem[8,7] - mem[8,26] - mem[8,4] - mem[8,23] - mem[8,14] - mem[8,1]
   - mem[8,20] - mem[8,11] - mem[8,30] - mem[8,17] - mem[8,34] - mem[8,8]
   - mem[8,27] + mem_tot_tor_P43 = 0
 R272: - 1.6276041666666666e-04 mem_tot_tor_P43
   + normalized_mem_tot_tor_P43 = 0
 rows_tot_tor_P43: - 3 frac[8,5] - 3 frac[8,24] - 3 frac[8,15]
   - 3 frac[8,2] - 3 frac[8,21] - 3 frac[8,12] - 3 frac[8,31]
   - 3 frac[8,18] - 3 frac[8,35] - 3 frac[8,28] - 3 frac[8,32]
   - 3 frac[8,6] - 3 frac[8,25] - 3 frac[8,3] - 3 frac[8,22] - 3 frac[8,13]
   - 3 frac[8,0] - 3 frac[8,19] - 3 frac[8,36] - 3 frac[8,10]
   - 3 frac[8,29] - 3 frac[8,16] - 3 frac[8,33] - 3 frac[8,7]
   - 3 frac[8,26] - 3 frac[8,4] - 3 frac[8,23] - 3 frac[8,14] - 3 frac[8,1]
   - 3 frac[8,20] - 3 frac[8,11] - 3 frac[8,30] - 3 frac[8,17]
   - 3 frac[8,34] - 3 frac[8,8] - 3 frac[8,27] + rows_tot_tor_P43 = 0
 R274: - 0.0208333333333333 rows_tot_tor_P43 + normalized_rows_tot_tor_P43
   = 0
 mem_tot_l1_P42: - mem[9,26] - mem[9,0] - mem[9,21] - mem[9,36] - mem[9,14]
   - mem[9,3] - mem[9,16] - mem[9,30] - mem[9,19] - mem[9,34] - mem[9,4]
   - mem[9,25] - mem[9,7] - mem[9,20] - mem[9,13] - mem[9,2] - mem[9,23]
   - mem[9,8] - mem[9,29] - mem[9,18] - mem[9,33] - mem[9,11] - mem[9,24]
   - mem[9,6] - mem[9,27] - mem[9,12] - mem[9,1] - mem[9,22] - mem[9,15]
   - mem[9,28] - mem[9,17] - mem[9,32] - mem[9,31] - mem[9,35] - mem[9,5]
   + mem_tot_l1_P42 = 0
 R276: - 1.6276041666666666e-04 mem_tot_l1_P42 + normalized_mem_tot_l1_P42
   = 0
 rows_tot_l1_P42: - 3 frac[9,26] - 3 frac[9,0] - 3 frac[9,21]
   - 3 frac[9,36] - 3 frac[9,14] - 3 frac[9,3] - 3 frac[9,16]
   - 3 frac[9,30] - 3 frac[9,19] - 3 frac[9,34] - 3 frac[9,4]
   - 3 frac[9,25] - 3 frac[9,7] - 3 frac[9,20] - 3 frac[9,13] - 3 frac[9,2]
   - 3 frac[9,23] - 3 frac[9,8] - 3 frac[9,29] - 3 frac[9,18]
   - 3 frac[9,33] - 3 frac[9,11] - 3 frac[9,24] - 3 frac[9,6]
   - 3 frac[9,27] - 3 frac[9,12] - 3 frac[9,1] - 3 frac[9,22]
   - 3 frac[9,15] - 3 frac[9,28] - 3 frac[9,17] - 3 frac[9,32]
   - 3 frac[9,31] - 3 frac[9,35] - 3 frac[9,5] + rows_tot_l1_P42 = 0
 R278: - 0.0208333333333333 rows_tot_l1_P42 + normalized_rows_tot_l1_P42
   = 0
 row_capacity_CPU2: rows_tot_CPU2 <= 12
 row_capacity_CPU4: rows_tot_CPU4 <= 12
 row_capacity_CPU13: rows_tot_CPU13 <= 12
 row_capacity_CPU18: rows_tot_CPU18 <= 12
 row_capacity_CPU21: rows_tot_CPU21 <= 12
 row_capacity_CPU24: rows_tot_CPU24 <= 12
 row_capacity_CPU26: rows_tot_CPU26 <= 12
 row_capacity_CPU32: rows_tot_CPU32 <= 12
 row_capacity_tor_P43: rows_tot_tor_P43 <= 48
 capacity_mem_par_tor_P43: mem[8,5] <= 1536
 capacity_mem_par_tor_P43: mem[8,24] <= 1536
 capacity_mem_par_tor_P43: mem[8,15] <= 1536
 capacity_mem_par_tor_P43: mem[8,2] <= 1536
 capacity_mem_par_tor_P43: mem[8,21] <= 1536
 capacity_mem_par_tor_P43: mem[8,12] <= 1536
 capacity_mem_par_tor_P43: mem[8,31] <= 1536
 capacity_mem_par_tor_P43: mem[8,18] <= 1536
 capacity_mem_par_tor_P43: mem[8,35] <= 1536
 capacity_mem_par_tor_P43: mem[8,28] <= 1536
 capacity_mem_par_tor_P43: mem[8,32] <= 1536
 capacity_mem_par_tor_P43: mem[8,6] <= 1536
 capacity_mem_par_tor_P43: mem[8,25] <= 1536
 capacity_mem_par_tor_P43: mem[8,3] <= 1536
 capacity_mem_par_tor_P43: mem[8,22] <= 1536
 capacity_mem_par_tor_P43: mem[8,13] <= 1536
 capacity_mem_par_tor_P43: mem[8,0] <= 1536
 capacity_mem_par_tor_P43: mem[8,19] <= 1536
 capacity_mem_par_tor_P43: mem[8,36] <= 1536
 capacity_mem_par_tor_P43: mem[8,10] <= 1536
 capacity_mem_par_tor_P43: mem[8,29] <= 1536
 capacity_mem_par_tor_P43: mem[8,16] <= 1536
 capacity_mem_par_tor_P43: mem[8,33] <= 1536
 capacity_mem_par_tor_P43: mem[8,7] <= 1536
 capacity_mem_par_tor_P43: mem[8,26] <= 1536
 capacity_mem_par_tor_P43: mem[8,4] <= 1536
 capacity_mem_par_tor_P43: mem[8,23] <= 1536
 capacity_mem_par_tor_P43: mem[8,14] <= 1536
 capacity_mem_par_tor_P43: mem[8,1] <= 1536
 capacity_mem_par_tor_P43: mem[8,20] <= 1536
 capacity_mem_par_tor_P43: mem[8,11] <= 1536
 capacity_mem_par_tor_P43: mem[8,30] <= 1536
 capacity_mem_par_tor_P43: mem[8,17] <= 1536
 capacity_mem_par_tor_P43: mem[8,34] <= 1536
 capacity_mem_par_tor_P43: mem[8,8] <= 1536
 capacity_mem_par_tor_P43: mem[8,27] <= 1536
 row_capacity_l1_P42: rows_tot_l1_P42 <= 48
 capacity_mem_par_l1_P42: mem[9,26] <= 1536
 capacity_mem_par_l1_P42: mem[9,0] <= 1536
 capacity_mem_par_l1_P42: mem[9,21] <= 1536
 capacity_mem_par_l1_P42: mem[9,36] <= 1536
 capacity_mem_par_l1_P42: mem[9,14] <= 1536
 capacity_mem_par_l1_P42: mem[9,3] <= 1536
 capacity_mem_par_l1_P42: mem[9,16] <= 1536
 capacity_mem_par_l1_P42: mem[9,30] <= 1536
 capacity_mem_par_l1_P42: mem[9,19] <= 1536
 capacity_mem_par_l1_P42: mem[9,34] <= 1536
 capacity_mem_par_l1_P42: mem[9,4] <= 1536
 capacity_mem_par_l1_P42: mem[9,25] <= 1536
 capacity_mem_par_l1_P42: mem[9,7] <= 1536
 capacity_mem_par_l1_P42: mem[9,20] <= 1536
 capacity_mem_par_l1_P42: mem[9,13] <= 1536
 capacity_mem_par_l1_P42: mem[9,2] <= 1536
 capacity_mem_par_l1_P42: mem[9,23] <= 1536
 capacity_mem_par_l1_P42: mem[9,8] <= 1536
 capacity_mem_par_l1_P42: mem[9,29] <= 1536
 capacity_mem_par_l1_P42: mem[9,18] <= 1536
 capacity_mem_par_l1_P42: mem[9,33] <= 1536
 capacity_mem_par_l1_P42: mem[9,11] <= 1536
 capacity_mem_par_l1_P42: mem[9,24] <= 1536
 capacity_mem_par_l1_P42: mem[9,6] <= 1536
 capacity_mem_par_l1_P42: mem[9,27] <= 1536
 capacity_mem_par_l1_P42: mem[9,12] <= 1536
 capacity_mem_par_l1_P42: mem[9,1] <= 1536
 capacity_mem_par_l1_P42: mem[9,22] <= 1536
 capacity_mem_par_l1_P42: mem[9,15] <= 1536
 capacity_mem_par_l1_P42: mem[9,28] <= 1536
 capacity_mem_par_l1_P42: mem[9,17] <= 1536
 capacity_mem_par_l1_P42: mem[9,32] <= 1536
 capacity_mem_par_l1_P42: mem[9,31] <= 1536
 capacity_mem_par_l1_P42: mem[9,35] <= 1536
 capacity_mem_par_l1_P42: mem[9,5] <= 1536
 ns_single_CPU2: 3.5 rows_tot_CPU2 - ns_single_CPU2 + pdt_m_rows_CPU2 = 0
 ns_sketch_CPU2: - ns_single_CPU2 + 13.41966586883029 cores_sketch_CPU2
   >= 0
 capacity_cores_CPU2: cores_sketch_CPU2 <= 3
 ns_single_CPU4: 3.5 rows_tot_CPU4 - ns_single_CPU4 + pdt_m_rows_CPU4 = 0
 ns_sketch_CPU4: - ns_single_CPU4 + 13.41966586883029 cores_sketch_CPU4
   >= 0
 capacity_cores_CPU4: cores_sketch_CPU4 <= 3
 ns_single_CPU13: 3.5 rows_tot_CPU13 - ns_single_CPU13 + pdt_m_rows_CPU13
   = 0
 ns_sketch_CPU13: - ns_single_CPU13 + 13.41966586883029 cores_sketch_CPU13
   >= 0
 capacity_cores_CPU13: cores_sketch_CPU13 <= 3
 ns_single_CPU18: 3.5 rows_tot_CPU18 - ns_single_CPU18 + pdt_m_rows_CPU18
   = 0
 ns_sketch_CPU18: - ns_single_CPU18 + 13.41966586883029 cores_sketch_CPU18
   >= 0
 capacity_cores_CPU18: cores_sketch_CPU18 <= 3
 ns_single_CPU21: 3.5 rows_tot_CPU21 - ns_single_CPU21 + pdt_m_rows_CPU21
   = 0
 ns_sketch_CPU21: - ns_single_CPU21 + 13.41966586883029 cores_sketch_CPU21
   >= 0
 capacity_cores_CPU21: cores_sketch_CPU21 <= 3
 ns_single_CPU24: 3.5 rows_tot_CPU24 - ns_single_CPU24 + pdt_m_rows_CPU24
   = 0
 ns_sketch_CPU24: - ns_single_CPU24 + 13.41966586883029 cores_sketch_CPU24
   >= 0
 capacity_cores_CPU24: cores_sketch_CPU24 <= 3
 ns_single_CPU26: 3.5 rows_tot_CPU26 - ns_single_CPU26 + pdt_m_rows_CPU26
   = 0
 ns_sketch_CPU26: - ns_single_CPU26 + 13.41966586883029 cores_sketch_CPU26
   >= 0
 capacity_cores_CPU26: cores_sketch_CPU26 <= 3
 ns_single_CPU32: 3.5 rows_tot_CPU32 - ns_single_CPU32 + pdt_m_rows_CPU32
   = 0
 ns_sketch_CPU32: - ns_single_CPU32 + 13.41966586883029 cores_sketch_CPU32
   >= 0
 capacity_cores_CPU32: cores_sketch_CPU32 <= 3
 res_acc: 1.220703125e-04 mem_tot_CPU2 + 1.220703125e-04 mem_tot_CPU4
   + 1.220703125e-04 mem_tot_CPU13 + 1.220703125e-04 mem_tot_CPU18
   + 1.220703125e-04 mem_tot_CPU21 + 1.220703125e-04 mem_tot_CPU24
   + 1.220703125e-04 mem_tot_CPU26 + 1.220703125e-04 mem_tot_CPU32
   + 0.001953125 mem_tot_tor_P43 + 0.25 rows_tot_tor_P43
   + 0.001953125 mem_tot_l1_P42 + 0.25 rows_tot_l1_P42
   + 10 cores_sketch_CPU2 + 10 cores_sketch_CPU4 + 10 cores_sketch_CPU13
   + 10 cores_sketch_CPU18 + 10 cores_sketch_CPU21 + 10 cores_sketch_CPU24
   + 10 cores_sketch_CPU26 + 10 cores_sketch_CPU32 - res = -320
 pdt_m_rows_CPU2: pdt_m_rows_CPU2 + [ - rows_tot_CPU2 * m_access_time_CPU2
   ] = 0
 pdt_m_rows_CPU4: pdt_m_rows_CPU4 + [ - rows_tot_CPU4 * m_access_time_CPU4
   ] = 0
 pdt_m_rows_CPU13: pdt_m_rows_CPU13 + [
   - rows_tot_CPU13 * m_access_time_CPU13 ] = 0
 pdt_m_rows_CPU18: pdt_m_rows_CPU18 + [
   - rows_tot_CPU18 * m_access_time_CPU18 ] = 0
 pdt_m_rows_CPU21: pdt_m_rows_CPU21 + [
   - rows_tot_CPU21 * m_access_time_CPU21 ] = 0
 pdt_m_rows_CPU24: pdt_m_rows_CPU24 + [
   - rows_tot_CPU24 * m_access_time_CPU24 ] = 0
 pdt_m_rows_CPU26: pdt_m_rows_CPU26 + [
   - rows_tot_CPU26 * m_access_time_CPU26 ] = 0
 pdt_m_rows_CPU32: pdt_m_rows_CPU32 + [
   - rows_tot_CPU32 * m_access_time_CPU32 ] = 0
Bounds
 frac[0,35] <= 1
 frac[0,7] <= 1
 frac[0,32] <= 1
 frac[0,27] <= 1
 frac[0,14] <= 1
 frac[0,33] <= 1
 frac[0,6] <= 1
 frac[0,16] <= 1
 frac[0,0] <= 1
 frac[0,23] <= 1
 frac[1,6] <= 1
 frac[1,12] <= 1
 frac[1,14] <= 1
 frac[1,32] <= 1
 frac[1,23] <= 1
 frac[1,35] <= 1
 frac[1,7] <= 1
 frac[1,29] <= 1
 frac[1,27] <= 1
 frac[2,5] <= 1
 frac[2,27] <= 1
 frac[2,6] <= 1
 frac[2,32] <= 1
 frac[2,11] <= 1
 frac[2,12] <= 1
 frac[2,23] <= 1
 frac[2,2] <= 1
 frac[2,29] <= 1
 frac[2,20] <= 1
 frac[2,4] <= 1
 frac[2,35] <= 1
 frac[2,16] <= 1
 frac[3,35] <= 1
 frac[3,23] <= 1
 frac[3,6] <= 1
 frac[3,27] <= 1
 frac[3,33] <= 1
 frac[3,5] <= 1
 frac[3,0] <= 1
 frac[3,32] <= 1
 frac[3,4] <= 1
 frac[3,14] <= 1
 frac[3,20] <= 1
 frac[3,29] <= 1
 frac[4,5] <= 1
 frac[4,35] <= 1
 frac[4,32] <= 1
 frac[4,12] <= 1
 frac[4,28] <= 1
 frac[4,17] <= 1
 frac[4,27] <= 1
 frac[4,2] <= 1
 frac[4,7] <= 1
 frac[4,20] <= 1
 frac[5,14] <= 1
 frac[5,27] <= 1
 frac[5,4] <= 1
 frac[5,23] <= 1
 frac[5,16] <= 1
 frac[5,12] <= 1
 frac[6,9] <= 1
 frac[6,16] <= 1
 frac[6,29] <= 1
 frac[6,32] <= 1
 frac[6,20] <= 1
 frac[7,12] <= 1
 frac[7,28] <= 1
 frac[7,5] <= 1
 frac[7,2] <= 1
 frac[7,20] <= 1
 frac[7,33] <= 1
 frac[7,14] <= 1
 frac[7,27] <= 1
 frac[7,0] <= 1
 frac[7,32] <= 1
 mem_tot_CPU2 <= 32768
 normalized_mem_tot_CPU2 <= 1
 normalized_rows_tot_CPU2 <= 1
 mem_tot_CPU4 <= 32768
 normalized_mem_tot_CPU4 <= 1
 normalized_rows_tot_CPU4 <= 1
 mem_tot_CPU13 <= 32768
 normalized_mem_tot_CPU13 <= 1
 normalized_rows_tot_CPU13 <= 1
 mem_tot_CPU18 <= 32768
 normalized_mem_tot_CPU18 <= 1
 normalized_rows_tot_CPU18 <= 1
 mem_tot_CPU21 <= 32768
 normalized_mem_tot_CPU21 <= 1
 normalized_rows_tot_CPU21 <= 1
 mem_tot_CPU24 <= 32768
 normalized_mem_tot_CPU24 <= 1
 normalized_rows_tot_CPU24 <= 1
 mem_tot_CPU26 <= 32768
 normalized_mem_tot_CPU26 <= 1
 normalized_rows_tot_CPU26 <= 1
 mem_tot_CPU32 <= 32768
 normalized_mem_tot_CPU32 <= 1
 normalized_rows_tot_CPU32 <= 1
 mem_tot_tor_P43 <= 6144
 normalized_mem_tot_tor_P43 <= 1
 normalized_rows_tot_tor_P43 <= 1
 mem_tot_l1_P42 <= 6144
 normalized_mem_tot_l1_P42 <= 1
 normalized_rows_tot_l1_P42 <= 1
 cores_sketch_CPU2 <= 7
 cores_sketch_CPU4 <= 7
 cores_sketch_CPU13 <= 7
 cores_sketch_CPU18 <= 7
 cores_sketch_CPU21 <= 7
 cores_sketch_CPU24 <= 7
 cores_sketch_CPU26 <= 7
 cores_sketch_CPU32 <= 7
Binaries
 frac[8,5] frac[8,24] frac[8,15] frac[8,2] frac[8,21] frac[8,12] frac[8,31]
 frac[8,18] frac[8,35] frac[8,28] frac[8,32] frac[8,6] frac[8,25] frac[8,3]
 frac[8,22] frac[8,13] frac[8,0] frac[8,19] frac[8,36] frac[8,10]
 frac[8,29] frac[8,16] frac[8,33] frac[8,7] frac[8,26] frac[8,4] frac[8,23]
 frac[8,14] frac[8,1] frac[8,20] frac[8,11] frac[8,30] frac[8,17]
 frac[8,34] frac[8,8] frac[8,27] frac[9,26] frac[9,0] frac[9,21] frac[9,36]
 frac[9,14] frac[9,3] frac[9,16] frac[9,30] frac[9,19] frac[9,34] frac[9,4]
 frac[9,25] frac[9,7] frac[9,20] frac[9,13] frac[9,2] frac[9,23] frac[9,8]
 frac[9,29] frac[9,18] frac[9,33] frac[9,11] frac[9,24] frac[9,6]
 frac[9,27] frac[9,12] frac[9,1] frac[9,22] frac[9,15] frac[9,28]
 frac[9,17] frac[9,32] frac[9,31] frac[9,35] frac[9,5]
Generals
 cores_sketch_CPU2 cores_sketch_CPU4 cores_sketch_CPU13 cores_sketch_CPU18
 cores_sketch_CPU21 cores_sketch_CPU24 cores_sketch_CPU26
 cores_sketch_CPU32
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_tor_P43 , mem_tot_l1_P42 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU2 , normalized_mem_tot_CPU4 ,
   normalized_mem_tot_CPU13 , normalized_mem_tot_CPU18 ,
   normalized_mem_tot_CPU21 , normalized_mem_tot_CPU24 ,
   normalized_mem_tot_CPU26 , normalized_mem_tot_CPU32 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_tor_P43 , rows_tot_l1_P42 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU2 ,
   normalized_rows_tot_CPU4 , normalized_rows_tot_CPU13 ,
   normalized_rows_tot_CPU18 , normalized_rows_tot_CPU21 ,
   normalized_rows_tot_CPU24 , normalized_rows_tot_CPU26 ,
   normalized_rows_tot_CPU32 )
 mem_access_time_CPU2: m_access_time_CPU2 = PWL ( mem_tot_CPU2 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU4: m_access_time_CPU4 = PWL ( mem_tot_CPU4 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU13: m_access_time_CPU13 = PWL ( mem_tot_CPU13 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU18: m_access_time_CPU18 = PWL ( mem_tot_CPU18 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU21: m_access_time_CPU21 = PWL ( mem_tot_CPU21 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU24: m_access_time_CPU24 = PWL ( mem_tot_CPU24 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU26: m_access_time_CPU26 = PWL ( mem_tot_CPU26 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU32: m_access_time_CPU32 = PWL ( mem_tot_CPU32 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

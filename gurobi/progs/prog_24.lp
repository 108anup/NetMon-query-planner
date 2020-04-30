\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_P4 + 0 max_mem_CPU_Cluster + 0 max_rows_P4
   + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU1 + 0 m_access_time_CPU6
   + 0 m_access_time_CPU9 + 0 m_access_time_CPU10 + 0 m_access_time_CPU11
   + 0 m_access_time_CPU18 + 0 m_access_time_CPU24 + res
Subject To
 cov_0_9: frac[5,9] + frac[2,9] + frac[7,9] >= 0.9
 cov_1_27: frac[6,27] + frac[1,27] >= 0.7
 cov_2_21: frac[4,21] + frac[7,21] + frac[3,21] >= 0.7
 cov_3_28: frac[6,28] + frac[1,28] >= 0.7
 cov_4_10: frac[0,10] >= 1
 cov_5_18: frac[7,18] + frac[4,18] >= 1
 cov_6_28: frac[7,28] + frac[3,28] + frac[4,28] >= 0.8
 cov_7_2: frac[3,2] + frac[7,2] + frac[6,2] >= 0.7
 cov_8_13: frac[5,13] + frac[1,13] >= 0.9
 cov_9_18: frac[5,18] + frac[7,18] + frac[4,18] >= 0.8
 cov_10_24: frac[3,24] + frac[7,24] + frac[2,24] >= 1
 cov_11_10: frac[3,10] + frac[7,10] >= 0.7
 cov_12_5: frac[0,5] + frac[6,5] >= 0.8
 cov_13_29: frac[5,29] >= 1
 cov_14_15: frac[0,15] + frac[6,15] >= 0.7
 cov_15_15: frac[7,15] + frac[3,15] + frac[6,15] >= 0.8
 cov_16_15: frac[4,15] + frac[1,15] + frac[7,15] >= 0.8
 cov_17_17: frac[2,17] + frac[3,17] + frac[7,17] >= 0.7
 cov_18_3: frac[7,3] + frac[5,3] + frac[4,3] >= 0.7
 cov_19_2: frac[3,2] + frac[7,2] + frac[4,2] >= 0.7
 cov_20_1: frac[1,1] + frac[2,1] + frac[7,1] >= 0.7
 cov_21_2: frac[3,2] + frac[7,2] + frac[1,2] >= 0.7
 cov_22_4: frac[0,4] + frac[1,4] >= 1
 cov_23_15: frac[4,15] + frac[7,15] + frac[0,15] >= 0.8
 cov_24_8: frac[7,8] + frac[2,8] + frac[3,8] >= 0.8
 cov_25_9: frac[6,9] + frac[1,9] >= 0.8
 cov_26_24: frac[3,24] + frac[0,24] + frac[7,24] >= 0.8
 cov_27_1: frac[3,1] + frac[7,1] >= 0.7
 cov_28_17: frac[1,17] + frac[2,17] + frac[7,17] >= 0.7
 cov_29_25: frac[7,25] + frac[0,25] + frac[3,25] >= 0.7
 cov_30_10: frac[4,10] + frac[7,10] + frac[2,10] >= 1
 cov_31_4: frac[7,4] + frac[3,4] + frac[2,4] >= 0.8
 cov_32_3: frac[7,3] + frac[0,3] + frac[3,3] >= 0.7
 cov_33_18: frac[1,18] >= 0.7
 cov_34_28: frac[6,28] + frac[5,28] >= 0.8
 cov_35_13: frac[5,13] >= 0.9
 cov_36_4: frac[0,4] + frac[7,4] + frac[3,4] >= 0.7
 cov_37_13: frac[5,13] + frac[0,13] >= 0.9
 cov_38_5: frac[5,5] + frac[6,5] >= 0.9
 cov_39_4: frac[6,4] + frac[1,4] >= 1
 cov_40_27: frac[3,27] + frac[4,27] + frac[7,27] >= 0.7
 cov_41_1: frac[5,1] + frac[6,1] >= 0.7
 cov_42_15: frac[6,15] >= 0.8
 cov_43_20: frac[6,20] + frac[7,20] + frac[3,20] >= 0.8
 cov_44_23: frac[2,23] + frac[4,23] + frac[7,23] >= 0.9
 cov_45_29: frac[2,29] + frac[7,29] + frac[0,29] >= 0.9
 cov_46_21: frac[7,21] + frac[2,21] >= 0.7
 cov_47_27: frac[2,27] + frac[4,27] + frac[7,27] >= 0.7
 cov_48_28: frac[6,28] + frac[7,28] + frac[3,28] >= 0.8
 cov_49_9: frac[5,9] >= 0.9
 cov_50_3: frac[7,3] + frac[2,3] + frac[4,3] >= 1
 cov_51_22: frac[7,22] >= 0.6444846666666639
 cov_52_12: frac[7,12] >= 0.7
 cov_53_0: frac[7,0] >= 0.3
 cov_54_7: frac[7,7] >= 0.7
 cov_55_14: frac[7,14] >= 0.6444846666666639
 cov_56_7: frac[7,7] >= 0.7
 cov_57_16: frac[7,16] >= 0.3
 cov_58_19: frac[7,19] >= 0.1
 cov_59_7: frac[7,7] >= 0.7
 cov_60_22: frac[7,22] >= 0.6444846666666639
 cov_61_0: frac[7,0] >= 0.3
 cov_62_14: frac[7,14] >= 0.6444846666666639
 cov_63_26: frac[7,26] >= 0.9
 cov_64_26: frac[7,26] >= 0.7
 cov_65_11: frac[7,11] >= 0.3
 cov_66_6: frac[7,6] >= 1
 cov_67_6: frac[7,6] >= 1
 cols_helper_7_3: cols_helper_7_3[0] + cols_helper_7_3[1]
   + cols_helper_7_3[2] + cols_helper_7_3[3] + cols_helper_7_3[4]
   + cols_helper_7_3[5] + cols_helper_7_3[6] + cols_helper_7_3[7]
   + cols_helper_7_3[8] + cols_helper_7_3[9] + cols_helper_7_3[10]
   + cols_helper_7_3[11] + cols_helper_7_3[12] + cols_helper_7_3[13]
   + cols_helper_7_3[14] + cols_helper_7_3[15] + cols_helper_7_3[16] <= 1
 cols_7_3: cols_7_3 - 2 cols_helper_7_3[0] - 4 cols_helper_7_3[1]
   - 8 cols_helper_7_3[2] - 16 cols_helper_7_3[3] - 32 cols_helper_7_3[4]
   - 64 cols_helper_7_3[5] - 128 cols_helper_7_3[6]
   - 256 cols_helper_7_3[7] - 512 cols_helper_7_3[8]
   - 1024 cols_helper_7_3[9] - 2048 cols_helper_7_3[10]
   - 4096 cols_helper_7_3[11] - 8192 cols_helper_7_3[12]
   - 16384 cols_helper_7_3[13] - 32768 cols_helper_7_3[14]
   - 65536 cols_helper_7_3[15] - 131072 cols_helper_7_3[16] = 0
 cols_acc_7_3: - 435 frac[7,3] + cols_7_3 >= 0
 accuracy_7_3: mem[7,3] - 0.01171875 cols_7_3 = 0
 accuracy_6_9: - 5.09765625 frac[6,9] + mem[6,9] = 0
 cols_helper_7_12: cols_helper_7_12[0] + cols_helper_7_12[1]
   + cols_helper_7_12[2] + cols_helper_7_12[3] + cols_helper_7_12[4]
   + cols_helper_7_12[5] + cols_helper_7_12[6] + cols_helper_7_12[7]
   + cols_helper_7_12[8] + cols_helper_7_12[9] + cols_helper_7_12[10]
   + cols_helper_7_12[11] + cols_helper_7_12[12] + cols_helper_7_12[13]
   + cols_helper_7_12[14] + cols_helper_7_12[15] + cols_helper_7_12[16]
   <= 1
 cols_7_12: cols_7_12 - 2 cols_helper_7_12[0] - 4 cols_helper_7_12[1]
   - 8 cols_helper_7_12[2] - 16 cols_helper_7_12[3]
   - 32 cols_helper_7_12[4] - 64 cols_helper_7_12[5]
   - 128 cols_helper_7_12[6] - 256 cols_helper_7_12[7]
   - 512 cols_helper_7_12[8] - 1024 cols_helper_7_12[9]
   - 2048 cols_helper_7_12[10] - 4096 cols_helper_7_12[11]
   - 8192 cols_helper_7_12[12] - 16384 cols_helper_7_12[13]
   - 32768 cols_helper_7_12[14] - 65536 cols_helper_7_12[15]
   - 131072 cols_helper_7_12[16] = 0
 cols_acc_7_12: - 435 frac[7,12] + cols_7_12 >= 0
 accuracy_7_12: mem[7,12] - 0.01171875 cols_7_12 = 0
 accuracy_6_28: - 5.09765625 frac[6,28] + mem[6,28] = 0
 accuracy_1_17: - 5.09765625 frac[1,17] + mem[1,17] = 0
 cols_helper_7_25: cols_helper_7_25[0] + cols_helper_7_25[1]
   + cols_helper_7_25[2] + cols_helper_7_25[3] + cols_helper_7_25[4]
   + cols_helper_7_25[5] + cols_helper_7_25[6] + cols_helper_7_25[7]
   + cols_helper_7_25[8] + cols_helper_7_25[9] + cols_helper_7_25[10]
   + cols_helper_7_25[11] + cols_helper_7_25[12] + cols_helper_7_25[13]
   + cols_helper_7_25[14] + cols_helper_7_25[15] + cols_helper_7_25[16]
   <= 1
 cols_7_25: cols_7_25 - 2 cols_helper_7_25[0] - 4 cols_helper_7_25[1]
   - 8 cols_helper_7_25[2] - 16 cols_helper_7_25[3]
   - 32 cols_helper_7_25[4] - 64 cols_helper_7_25[5]
   - 128 cols_helper_7_25[6] - 256 cols_helper_7_25[7]
   - 512 cols_helper_7_25[8] - 1024 cols_helper_7_25[9]
   - 2048 cols_helper_7_25[10] - 4096 cols_helper_7_25[11]
   - 8192 cols_helper_7_25[12] - 16384 cols_helper_7_25[13]
   - 32768 cols_helper_7_25[14] - 65536 cols_helper_7_25[15]
   - 131072 cols_helper_7_25[16] = 0
 cols_acc_7_25: - 435 frac[7,25] + cols_7_25 >= 0
 accuracy_7_25: mem[7,25] - 0.01171875 cols_7_25 = 0
 accuracy_0_10: - 5.09765625 frac[0,10] + mem[0,10] = 0
 accuracy_3_24: - 5.09765625 frac[3,24] + mem[3,24] = 0
 accuracy_5_5: - 5.09765625 frac[5,5] + mem[5,5] = 0
 cols_helper_7_6: cols_helper_7_6[0] + cols_helper_7_6[1]
   + cols_helper_7_6[2] + cols_helper_7_6[3] + cols_helper_7_6[4]
   + cols_helper_7_6[5] + cols_helper_7_6[6] + cols_helper_7_6[7]
   + cols_helper_7_6[8] + cols_helper_7_6[9] + cols_helper_7_6[10]
   + cols_helper_7_6[11] + cols_helper_7_6[12] + cols_helper_7_6[13]
   + cols_helper_7_6[14] + cols_helper_7_6[15] + cols_helper_7_6[16] <= 1
 cols_7_6: cols_7_6 - 2 cols_helper_7_6[0] - 4 cols_helper_7_6[1]
   - 8 cols_helper_7_6[2] - 16 cols_helper_7_6[3] - 32 cols_helper_7_6[4]
   - 64 cols_helper_7_6[5] - 128 cols_helper_7_6[6]
   - 256 cols_helper_7_6[7] - 512 cols_helper_7_6[8]
   - 1024 cols_helper_7_6[9] - 2048 cols_helper_7_6[10]
   - 4096 cols_helper_7_6[11] - 8192 cols_helper_7_6[12]
   - 16384 cols_helper_7_6[13] - 32768 cols_helper_7_6[14]
   - 65536 cols_helper_7_6[15] - 131072 cols_helper_7_6[16] = 0
 cols_acc_7_6: - 435 frac[7,6] + cols_7_6 >= 0
 accuracy_7_6: mem[7,6] - 0.01171875 cols_7_6 = 0
 accuracy_5_18: - 5.09765625 frac[5,18] + mem[5,18] = 0
 cols_helper_7_19: cols_helper_7_19[0] + cols_helper_7_19[1]
   + cols_helper_7_19[2] + cols_helper_7_19[3] + cols_helper_7_19[4]
   + cols_helper_7_19[5] + cols_helper_7_19[6] + cols_helper_7_19[7]
   + cols_helper_7_19[8] + cols_helper_7_19[9] + cols_helper_7_19[10]
   + cols_helper_7_19[11] + cols_helper_7_19[12] + cols_helper_7_19[13]
   + cols_helper_7_19[14] + cols_helper_7_19[15] + cols_helper_7_19[16]
   <= 1
 cols_7_19: cols_7_19 - 2 cols_helper_7_19[0] - 4 cols_helper_7_19[1]
   - 8 cols_helper_7_19[2] - 16 cols_helper_7_19[3]
   - 32 cols_helper_7_19[4] - 64 cols_helper_7_19[5]
   - 128 cols_helper_7_19[6] - 256 cols_helper_7_19[7]
   - 512 cols_helper_7_19[8] - 1024 cols_helper_7_19[9]
   - 2048 cols_helper_7_19[10] - 4096 cols_helper_7_19[11]
   - 8192 cols_helper_7_19[12] - 16384 cols_helper_7_19[13]
   - 32768 cols_helper_7_19[14] - 65536 cols_helper_7_19[15]
   - 131072 cols_helper_7_19[16] = 0
 cols_acc_7_19: - 435 frac[7,19] + cols_7_19 >= 0
 accuracy_7_19: mem[7,19] - 0.01171875 cols_7_19 = 0
 accuracy_1_28: - 5.09765625 frac[1,28] + mem[1,28] = 0
 accuracy_0_4: - 5.09765625 frac[0,4] + mem[0,4] = 0
 cols_helper_7_28: cols_helper_7_28[0] + cols_helper_7_28[1]
   + cols_helper_7_28[2] + cols_helper_7_28[3] + cols_helper_7_28[4]
   + cols_helper_7_28[5] + cols_helper_7_28[6] + cols_helper_7_28[7]
   + cols_helper_7_28[8] + cols_helper_7_28[9] + cols_helper_7_28[10]
   + cols_helper_7_28[11] + cols_helper_7_28[12] + cols_helper_7_28[13]
   + cols_helper_7_28[14] + cols_helper_7_28[15] + cols_helper_7_28[16]
   <= 1
 cols_7_28: cols_7_28 - 2 cols_helper_7_28[0] - 4 cols_helper_7_28[1]
   - 8 cols_helper_7_28[2] - 16 cols_helper_7_28[3]
   - 32 cols_helper_7_28[4] - 64 cols_helper_7_28[5]
   - 128 cols_helper_7_28[6] - 256 cols_helper_7_28[7]
   - 512 cols_helper_7_28[8] - 1024 cols_helper_7_28[9]
   - 2048 cols_helper_7_28[10] - 4096 cols_helper_7_28[11]
   - 8192 cols_helper_7_28[12] - 16384 cols_helper_7_28[13]
   - 32768 cols_helper_7_28[14] - 65536 cols_helper_7_28[15]
   - 131072 cols_helper_7_28[16] = 0
 cols_acc_7_28: - 435 frac[7,28] + cols_7_28 >= 0
 accuracy_7_28: mem[7,28] - 0.01171875 cols_7_28 = 0
 accuracy_2_27: - 5.09765625 frac[2,27] + mem[2,27] = 0
 accuracy_1_1: - 5.09765625 frac[1,1] + mem[1,1] = 0
 accuracy_4_10: - 5.09765625 frac[4,10] + mem[4,10] = 0
 accuracy_3_2: - 5.09765625 frac[3,2] + mem[3,2] = 0
 cols_helper_7_22: cols_helper_7_22[0] + cols_helper_7_22[1]
   + cols_helper_7_22[2] + cols_helper_7_22[3] + cols_helper_7_22[4]
   + cols_helper_7_22[5] + cols_helper_7_22[6] + cols_helper_7_22[7]
   + cols_helper_7_22[8] + cols_helper_7_22[9] + cols_helper_7_22[10]
   + cols_helper_7_22[11] + cols_helper_7_22[12] + cols_helper_7_22[13]
   + cols_helper_7_22[14] + cols_helper_7_22[15] + cols_helper_7_22[16]
   <= 1
 cols_7_22: cols_7_22 - 2 cols_helper_7_22[0] - 4 cols_helper_7_22[1]
   - 8 cols_helper_7_22[2] - 16 cols_helper_7_22[3]
   - 32 cols_helper_7_22[4] - 64 cols_helper_7_22[5]
   - 128 cols_helper_7_22[6] - 256 cols_helper_7_22[7]
   - 512 cols_helper_7_22[8] - 1024 cols_helper_7_22[9]
   - 2048 cols_helper_7_22[10] - 4096 cols_helper_7_22[11]
   - 8192 cols_helper_7_22[12] - 16384 cols_helper_7_22[13]
   - 32768 cols_helper_7_22[14] - 65536 cols_helper_7_22[15]
   - 131072 cols_helper_7_22[16] = 0
 cols_acc_7_22: - 435 frac[7,22] + cols_7_22 >= 0
 accuracy_7_22: mem[7,22] - 0.01171875 cols_7_22 = 0
 accuracy_2_17: - 5.09765625 frac[2,17] + mem[2,17] = 0
 accuracy_4_15: - 5.09765625 frac[4,15] + mem[4,15] = 0
 accuracy_3_1: - 5.09765625 frac[3,1] + mem[3,1] = 0
 accuracy_4_21: - 5.09765625 frac[4,21] + mem[4,21] = 0
 cols_helper_7_8: cols_helper_7_8[0] + cols_helper_7_8[1]
   + cols_helper_7_8[2] + cols_helper_7_8[3] + cols_helper_7_8[4]
   + cols_helper_7_8[5] + cols_helper_7_8[6] + cols_helper_7_8[7]
   + cols_helper_7_8[8] + cols_helper_7_8[9] + cols_helper_7_8[10]
   + cols_helper_7_8[11] + cols_helper_7_8[12] + cols_helper_7_8[13]
   + cols_helper_7_8[14] + cols_helper_7_8[15] + cols_helper_7_8[16] <= 1
 cols_7_8: cols_7_8 - 2 cols_helper_7_8[0] - 4 cols_helper_7_8[1]
   - 8 cols_helper_7_8[2] - 16 cols_helper_7_8[3] - 32 cols_helper_7_8[4]
   - 64 cols_helper_7_8[5] - 128 cols_helper_7_8[6]
   - 256 cols_helper_7_8[7] - 512 cols_helper_7_8[8]
   - 1024 cols_helper_7_8[9] - 2048 cols_helper_7_8[10]
   - 4096 cols_helper_7_8[11] - 8192 cols_helper_7_8[12]
   - 16384 cols_helper_7_8[13] - 32768 cols_helper_7_8[14]
   - 65536 cols_helper_7_8[15] - 131072 cols_helper_7_8[16] = 0
 cols_acc_7_8: - 435 frac[7,8] + cols_7_8 >= 0
 accuracy_7_8: mem[7,8] - 0.01171875 cols_7_8 = 0
 cols_helper_7_21: cols_helper_7_21[0] + cols_helper_7_21[1]
   + cols_helper_7_21[2] + cols_helper_7_21[3] + cols_helper_7_21[4]
   + cols_helper_7_21[5] + cols_helper_7_21[6] + cols_helper_7_21[7]
   + cols_helper_7_21[8] + cols_helper_7_21[9] + cols_helper_7_21[10]
   + cols_helper_7_21[11] + cols_helper_7_21[12] + cols_helper_7_21[13]
   + cols_helper_7_21[14] + cols_helper_7_21[15] + cols_helper_7_21[16]
   <= 1
 cols_7_21: cols_7_21 - 2 cols_helper_7_21[0] - 4 cols_helper_7_21[1]
   - 8 cols_helper_7_21[2] - 16 cols_helper_7_21[3]
   - 32 cols_helper_7_21[4] - 64 cols_helper_7_21[5]
   - 128 cols_helper_7_21[6] - 256 cols_helper_7_21[7]
   - 512 cols_helper_7_21[8] - 1024 cols_helper_7_21[9]
   - 2048 cols_helper_7_21[10] - 4096 cols_helper_7_21[11]
   - 8192 cols_helper_7_21[12] - 16384 cols_helper_7_21[13]
   - 32768 cols_helper_7_21[14] - 65536 cols_helper_7_21[15]
   - 131072 cols_helper_7_21[16] = 0
 cols_acc_7_21: - 435 frac[7,21] + cols_7_21 >= 0
 accuracy_7_21: mem[7,21] - 0.01171875 cols_7_21 = 0
 accuracy_2_1: - 5.09765625 frac[2,1] + mem[2,1] = 0
 accuracy_1_15: - 5.09765625 frac[1,15] + mem[1,15] = 0
 accuracy_3_28: - 5.09765625 frac[3,28] + mem[3,28] = 0
 accuracy_5_1: - 5.09765625 frac[5,1] + mem[5,1] = 0
 accuracy_3_17: - 5.09765625 frac[3,17] + mem[3,17] = 0
 cols_helper_7_2: cols_helper_7_2[0] + cols_helper_7_2[1]
   + cols_helper_7_2[2] + cols_helper_7_2[3] + cols_helper_7_2[4]
   + cols_helper_7_2[5] + cols_helper_7_2[6] + cols_helper_7_2[7]
   + cols_helper_7_2[8] + cols_helper_7_2[9] + cols_helper_7_2[10]
   + cols_helper_7_2[11] + cols_helper_7_2[12] + cols_helper_7_2[13]
   + cols_helper_7_2[14] + cols_helper_7_2[15] + cols_helper_7_2[16] <= 1
 cols_7_2: cols_7_2 - 2 cols_helper_7_2[0] - 4 cols_helper_7_2[1]
   - 8 cols_helper_7_2[2] - 16 cols_helper_7_2[3] - 32 cols_helper_7_2[4]
   - 64 cols_helper_7_2[5] - 128 cols_helper_7_2[6]
   - 256 cols_helper_7_2[7] - 512 cols_helper_7_2[8]
   - 1024 cols_helper_7_2[9] - 2048 cols_helper_7_2[10]
   - 4096 cols_helper_7_2[11] - 8192 cols_helper_7_2[12]
   - 16384 cols_helper_7_2[13] - 32768 cols_helper_7_2[14]
   - 65536 cols_helper_7_2[15] - 131072 cols_helper_7_2[16] = 0
 cols_acc_7_2: - 435 frac[7,2] + cols_7_2 >= 0
 accuracy_7_2: mem[7,2] - 0.01171875 cols_7_2 = 0
 cols_helper_7_15: cols_helper_7_15[0] + cols_helper_7_15[1]
   + cols_helper_7_15[2] + cols_helper_7_15[3] + cols_helper_7_15[4]
   + cols_helper_7_15[5] + cols_helper_7_15[6] + cols_helper_7_15[7]
   + cols_helper_7_15[8] + cols_helper_7_15[9] + cols_helper_7_15[10]
   + cols_helper_7_15[11] + cols_helper_7_15[12] + cols_helper_7_15[13]
   + cols_helper_7_15[14] + cols_helper_7_15[15] + cols_helper_7_15[16]
   <= 1
 cols_7_15: cols_7_15 - 2 cols_helper_7_15[0] - 4 cols_helper_7_15[1]
   - 8 cols_helper_7_15[2] - 16 cols_helper_7_15[3]
   - 32 cols_helper_7_15[4] - 64 cols_helper_7_15[5]
   - 128 cols_helper_7_15[6] - 256 cols_helper_7_15[7]
   - 512 cols_helper_7_15[8] - 1024 cols_helper_7_15[9]
   - 2048 cols_helper_7_15[10] - 4096 cols_helper_7_15[11]
   - 8192 cols_helper_7_15[12] - 16384 cols_helper_7_15[13]
   - 32768 cols_helper_7_15[14] - 65536 cols_helper_7_15[15]
   - 131072 cols_helper_7_15[16] = 0
 cols_acc_7_15: - 435 frac[7,15] + cols_7_15 >= 0
 accuracy_7_15: mem[7,15] - 0.01171875 cols_7_15 = 0
 accuracy_0_24: - 5.09765625 frac[0,24] + mem[0,24] = 0
 cols_helper_7_24: cols_helper_7_24[0] + cols_helper_7_24[1]
   + cols_helper_7_24[2] + cols_helper_7_24[3] + cols_helper_7_24[4]
   + cols_helper_7_24[5] + cols_helper_7_24[6] + cols_helper_7_24[7]
   + cols_helper_7_24[8] + cols_helper_7_24[9] + cols_helper_7_24[10]
   + cols_helper_7_24[11] + cols_helper_7_24[12] + cols_helper_7_24[13]
   + cols_helper_7_24[14] + cols_helper_7_24[15] + cols_helper_7_24[16]
   <= 1
 cols_7_24: cols_7_24 - 2 cols_helper_7_24[0] - 4 cols_helper_7_24[1]
   - 8 cols_helper_7_24[2] - 16 cols_helper_7_24[3]
   - 32 cols_helper_7_24[4] - 64 cols_helper_7_24[5]
   - 128 cols_helper_7_24[6] - 256 cols_helper_7_24[7]
   - 512 cols_helper_7_24[8] - 1024 cols_helper_7_24[9]
   - 2048 cols_helper_7_24[10] - 4096 cols_helper_7_24[11]
   - 8192 cols_helper_7_24[12] - 16384 cols_helper_7_24[13]
   - 32768 cols_helper_7_24[14] - 65536 cols_helper_7_24[15]
   - 131072 cols_helper_7_24[16] = 0
 cols_acc_7_24: - 435 frac[7,24] + cols_7_24 >= 0
 accuracy_7_24: mem[7,24] - 0.01171875 cols_7_24 = 0
 accuracy_2_23: - 5.09765625 frac[2,23] + mem[2,23] = 0
 accuracy_3_27: - 5.09765625 frac[3,27] + mem[3,27] = 0
 accuracy_6_4: - 5.09765625 frac[6,4] + mem[6,4] = 0
 accuracy_4_28: - 5.09765625 frac[4,28] + mem[4,28] = 0
 cols_helper_7_1: cols_helper_7_1[0] + cols_helper_7_1[1]
   + cols_helper_7_1[2] + cols_helper_7_1[3] + cols_helper_7_1[4]
   + cols_helper_7_1[5] + cols_helper_7_1[6] + cols_helper_7_1[7]
   + cols_helper_7_1[8] + cols_helper_7_1[9] + cols_helper_7_1[10]
   + cols_helper_7_1[11] + cols_helper_7_1[12] + cols_helper_7_1[13]
   + cols_helper_7_1[14] + cols_helper_7_1[15] + cols_helper_7_1[16] <= 1
 cols_7_1: cols_7_1 - 2 cols_helper_7_1[0] - 4 cols_helper_7_1[1]
   - 8 cols_helper_7_1[2] - 16 cols_helper_7_1[3] - 32 cols_helper_7_1[4]
   - 64 cols_helper_7_1[5] - 128 cols_helper_7_1[6]
   - 256 cols_helper_7_1[7] - 512 cols_helper_7_1[8]
   - 1024 cols_helper_7_1[9] - 2048 cols_helper_7_1[10]
   - 4096 cols_helper_7_1[11] - 8192 cols_helper_7_1[12]
   - 16384 cols_helper_7_1[13] - 32768 cols_helper_7_1[14]
   - 65536 cols_helper_7_1[15] - 131072 cols_helper_7_1[16] = 0
 cols_acc_7_1: - 435 frac[7,1] + cols_7_1 >= 0
 accuracy_7_1: mem[7,1] - 0.01171875 cols_7_1 = 0
 cols_helper_7_18: cols_helper_7_18[0] + cols_helper_7_18[1]
   + cols_helper_7_18[2] + cols_helper_7_18[3] + cols_helper_7_18[4]
   + cols_helper_7_18[5] + cols_helper_7_18[6] + cols_helper_7_18[7]
   + cols_helper_7_18[8] + cols_helper_7_18[9] + cols_helper_7_18[10]
   + cols_helper_7_18[11] + cols_helper_7_18[12] + cols_helper_7_18[13]
   + cols_helper_7_18[14] + cols_helper_7_18[15] + cols_helper_7_18[16]
   <= 1
 cols_7_18: cols_7_18 - 2 cols_helper_7_18[0] - 4 cols_helper_7_18[1]
   - 8 cols_helper_7_18[2] - 16 cols_helper_7_18[3]
   - 32 cols_helper_7_18[4] - 64 cols_helper_7_18[5]
   - 128 cols_helper_7_18[6] - 256 cols_helper_7_18[7]
   - 512 cols_helper_7_18[8] - 1024 cols_helper_7_18[9]
   - 2048 cols_helper_7_18[10] - 4096 cols_helper_7_18[11]
   - 8192 cols_helper_7_18[12] - 16384 cols_helper_7_18[13]
   - 32768 cols_helper_7_18[14] - 65536 cols_helper_7_18[15]
   - 131072 cols_helper_7_18[16] = 0
 cols_acc_7_18: - 435 frac[7,18] + cols_7_18 >= 0
 accuracy_7_18: mem[7,18] - 0.01171875 cols_7_18 = 0
 accuracy_0_5: - 5.09765625 frac[0,5] + mem[0,5] = 0
 accuracy_2_24: - 5.09765625 frac[2,24] + mem[2,24] = 0
 accuracy_6_1: - 5.09765625 frac[6,1] + mem[6,1] = 0
 cols_helper_7_4: cols_helper_7_4[0] + cols_helper_7_4[1]
   + cols_helper_7_4[2] + cols_helper_7_4[3] + cols_helper_7_4[4]
   + cols_helper_7_4[5] + cols_helper_7_4[6] + cols_helper_7_4[7]
   + cols_helper_7_4[8] + cols_helper_7_4[9] + cols_helper_7_4[10]
   + cols_helper_7_4[11] + cols_helper_7_4[12] + cols_helper_7_4[13]
   + cols_helper_7_4[14] + cols_helper_7_4[15] + cols_helper_7_4[16] <= 1
 cols_7_4: cols_7_4 - 2 cols_helper_7_4[0] - 4 cols_helper_7_4[1]
   - 8 cols_helper_7_4[2] - 16 cols_helper_7_4[3] - 32 cols_helper_7_4[4]
   - 64 cols_helper_7_4[5] - 128 cols_helper_7_4[6]
   - 256 cols_helper_7_4[7] - 512 cols_helper_7_4[8]
   - 1024 cols_helper_7_4[9] - 2048 cols_helper_7_4[10]
   - 4096 cols_helper_7_4[11] - 8192 cols_helper_7_4[12]
   - 16384 cols_helper_7_4[13] - 32768 cols_helper_7_4[14]
   - 65536 cols_helper_7_4[15] - 131072 cols_helper_7_4[16] = 0
 cols_acc_7_4: - 435 frac[7,4] + cols_7_4 >= 0
 accuracy_7_4: mem[7,4] - 0.01171875 cols_7_4 = 0
 accuracy_6_20: - 5.09765625 frac[6,20] + mem[6,20] = 0
 cols_helper_7_17: cols_helper_7_17[0] + cols_helper_7_17[1]
   + cols_helper_7_17[2] + cols_helper_7_17[3] + cols_helper_7_17[4]
   + cols_helper_7_17[5] + cols_helper_7_17[6] + cols_helper_7_17[7]
   + cols_helper_7_17[8] + cols_helper_7_17[9] + cols_helper_7_17[10]
   + cols_helper_7_17[11] + cols_helper_7_17[12] + cols_helper_7_17[13]
   + cols_helper_7_17[14] + cols_helper_7_17[15] + cols_helper_7_17[16]
   <= 1
 cols_7_17: cols_7_17 - 2 cols_helper_7_17[0] - 4 cols_helper_7_17[1]
   - 8 cols_helper_7_17[2] - 16 cols_helper_7_17[3]
   - 32 cols_helper_7_17[4] - 64 cols_helper_7_17[5]
   - 128 cols_helper_7_17[6] - 256 cols_helper_7_17[7]
   - 512 cols_helper_7_17[8] - 1024 cols_helper_7_17[9]
   - 2048 cols_helper_7_17[10] - 4096 cols_helper_7_17[11]
   - 8192 cols_helper_7_17[12] - 16384 cols_helper_7_17[13]
   - 32768 cols_helper_7_17[14] - 65536 cols_helper_7_17[15]
   - 131072 cols_helper_7_17[16] = 0
 cols_acc_7_17: - 435 frac[7,17] + cols_7_17 >= 0
 accuracy_7_17: mem[7,17] - 0.01171875 cols_7_17 = 0
 accuracy_6_27: - 5.09765625 frac[6,27] + mem[6,27] = 0
 accuracy_3_15: - 5.09765625 frac[3,15] + mem[3,15] = 0
 accuracy_2_29: - 5.09765625 frac[2,29] + mem[2,29] = 0
 accuracy_2_8: - 5.09765625 frac[2,8] + mem[2,8] = 0
 accuracy_5_13: - 5.09765625 frac[5,13] + mem[5,13] = 0
 accuracy_4_27: - 5.09765625 frac[4,27] + mem[4,27] = 0
 accuracy_3_21: - 5.09765625 frac[3,21] + mem[3,21] = 0
 accuracy_6_2: - 5.09765625 frac[6,2] + mem[6,2] = 0
 cols_helper_7_11: cols_helper_7_11[0] + cols_helper_7_11[1]
   + cols_helper_7_11[2] + cols_helper_7_11[3] + cols_helper_7_11[4]
   + cols_helper_7_11[5] + cols_helper_7_11[6] + cols_helper_7_11[7]
   + cols_helper_7_11[8] + cols_helper_7_11[9] + cols_helper_7_11[10]
   + cols_helper_7_11[11] + cols_helper_7_11[12] + cols_helper_7_11[13]
   + cols_helper_7_11[14] + cols_helper_7_11[15] + cols_helper_7_11[16]
   <= 1
 cols_7_11: cols_7_11 - 2 cols_helper_7_11[0] - 4 cols_helper_7_11[1]
   - 8 cols_helper_7_11[2] - 16 cols_helper_7_11[3]
   - 32 cols_helper_7_11[4] - 64 cols_helper_7_11[5]
   - 128 cols_helper_7_11[6] - 256 cols_helper_7_11[7]
   - 512 cols_helper_7_11[8] - 1024 cols_helper_7_11[9]
   - 2048 cols_helper_7_11[10] - 4096 cols_helper_7_11[11]
   - 8192 cols_helper_7_11[12] - 16384 cols_helper_7_11[13]
   - 32768 cols_helper_7_11[14] - 65536 cols_helper_7_11[15]
   - 131072 cols_helper_7_11[16] = 0
 cols_acc_7_11: - 435 frac[7,11] + cols_7_11 >= 0
 accuracy_7_11: mem[7,11] - 0.01171875 cols_7_11 = 0
 cols_helper_7_20: cols_helper_7_20[0] + cols_helper_7_20[1]
   + cols_helper_7_20[2] + cols_helper_7_20[3] + cols_helper_7_20[4]
   + cols_helper_7_20[5] + cols_helper_7_20[6] + cols_helper_7_20[7]
   + cols_helper_7_20[8] + cols_helper_7_20[9] + cols_helper_7_20[10]
   + cols_helper_7_20[11] + cols_helper_7_20[12] + cols_helper_7_20[13]
   + cols_helper_7_20[14] + cols_helper_7_20[15] + cols_helper_7_20[16]
   <= 1
 cols_7_20: cols_7_20 - 2 cols_helper_7_20[0] - 4 cols_helper_7_20[1]
   - 8 cols_helper_7_20[2] - 16 cols_helper_7_20[3]
   - 32 cols_helper_7_20[4] - 64 cols_helper_7_20[5]
   - 128 cols_helper_7_20[6] - 256 cols_helper_7_20[7]
   - 512 cols_helper_7_20[8] - 1024 cols_helper_7_20[9]
   - 2048 cols_helper_7_20[10] - 4096 cols_helper_7_20[11]
   - 8192 cols_helper_7_20[12] - 16384 cols_helper_7_20[13]
   - 32768 cols_helper_7_20[14] - 65536 cols_helper_7_20[15]
   - 131072 cols_helper_7_20[16] = 0
 cols_acc_7_20: - 435 frac[7,20] + cols_7_20 >= 0
 accuracy_7_20: mem[7,20] - 0.01171875 cols_7_20 = 0
 accuracy_0_15: - 5.09765625 frac[0,15] + mem[0,15] = 0
 accuracy_3_10: - 5.09765625 frac[3,10] + mem[3,10] = 0
 accuracy_6_15: - 5.09765625 frac[6,15] + mem[6,15] = 0
 accuracy_5_29: - 5.09765625 frac[5,29] + mem[5,29] = 0
 cols_helper_7_14: cols_helper_7_14[0] + cols_helper_7_14[1]
   + cols_helper_7_14[2] + cols_helper_7_14[3] + cols_helper_7_14[4]
   + cols_helper_7_14[5] + cols_helper_7_14[6] + cols_helper_7_14[7]
   + cols_helper_7_14[8] + cols_helper_7_14[9] + cols_helper_7_14[10]
   + cols_helper_7_14[11] + cols_helper_7_14[12] + cols_helper_7_14[13]
   + cols_helper_7_14[14] + cols_helper_7_14[15] + cols_helper_7_14[16]
   <= 1
 cols_7_14: cols_7_14 - 2 cols_helper_7_14[0] - 4 cols_helper_7_14[1]
   - 8 cols_helper_7_14[2] - 16 cols_helper_7_14[3]
   - 32 cols_helper_7_14[4] - 64 cols_helper_7_14[5]
   - 128 cols_helper_7_14[6] - 256 cols_helper_7_14[7]
   - 512 cols_helper_7_14[8] - 1024 cols_helper_7_14[9]
   - 2048 cols_helper_7_14[10] - 4096 cols_helper_7_14[11]
   - 8192 cols_helper_7_14[12] - 16384 cols_helper_7_14[13]
   - 32768 cols_helper_7_14[14] - 65536 cols_helper_7_14[15]
   - 131072 cols_helper_7_14[16] = 0
 cols_acc_7_14: - 435 frac[7,14] + cols_7_14 >= 0
 accuracy_7_14: mem[7,14] - 0.01171875 cols_7_14 = 0
 accuracy_0_25: - 5.09765625 frac[0,25] + mem[0,25] = 0
 cols_helper_7_27: cols_helper_7_27[0] + cols_helper_7_27[1]
   + cols_helper_7_27[2] + cols_helper_7_27[3] + cols_helper_7_27[4]
   + cols_helper_7_27[5] + cols_helper_7_27[6] + cols_helper_7_27[7]
   + cols_helper_7_27[8] + cols_helper_7_27[9] + cols_helper_7_27[10]
   + cols_helper_7_27[11] + cols_helper_7_27[12] + cols_helper_7_27[13]
   + cols_helper_7_27[14] + cols_helper_7_27[15] + cols_helper_7_27[16]
   <= 1
 cols_7_27: cols_7_27 - 2 cols_helper_7_27[0] - 4 cols_helper_7_27[1]
   - 8 cols_helper_7_27[2] - 16 cols_helper_7_27[3]
   - 32 cols_helper_7_27[4] - 64 cols_helper_7_27[5]
   - 128 cols_helper_7_27[6] - 256 cols_helper_7_27[7]
   - 512 cols_helper_7_27[8] - 1024 cols_helper_7_27[9]
   - 2048 cols_helper_7_27[10] - 4096 cols_helper_7_27[11]
   - 8192 cols_helper_7_27[12] - 16384 cols_helper_7_27[13]
   - 32768 cols_helper_7_27[14] - 65536 cols_helper_7_27[15]
   - 131072 cols_helper_7_27[16] = 0
 cols_acc_7_27: - 435 frac[7,27] + cols_7_27 >= 0
 accuracy_7_27: mem[7,27] - 0.01171875 cols_7_27 = 0
 accuracy_1_4: - 5.09765625 frac[1,4] + mem[1,4] = 0
 accuracy_2_3: - 5.09765625 frac[2,3] + mem[2,3] = 0
 accuracy_1_9: - 5.09765625 frac[1,9] + mem[1,9] = 0
 accuracy_4_2: - 5.09765625 frac[4,2] + mem[4,2] = 0
 accuracy_6_5: - 5.09765625 frac[6,5] + mem[6,5] = 0
 accuracy_5_3: - 5.09765625 frac[5,3] + mem[5,3] = 0
 cols_helper_7_0: cols_helper_7_0[0] + cols_helper_7_0[1]
   + cols_helper_7_0[2] + cols_helper_7_0[3] + cols_helper_7_0[4]
   + cols_helper_7_0[5] + cols_helper_7_0[6] + cols_helper_7_0[7]
   + cols_helper_7_0[8] + cols_helper_7_0[9] + cols_helper_7_0[10]
   + cols_helper_7_0[11] + cols_helper_7_0[12] + cols_helper_7_0[13]
   + cols_helper_7_0[14] + cols_helper_7_0[15] + cols_helper_7_0[16] <= 1
 cols_7_0: cols_7_0 - 2 cols_helper_7_0[0] - 4 cols_helper_7_0[1]
   - 8 cols_helper_7_0[2] - 16 cols_helper_7_0[3] - 32 cols_helper_7_0[4]
   - 64 cols_helper_7_0[5] - 128 cols_helper_7_0[6]
   - 256 cols_helper_7_0[7] - 512 cols_helper_7_0[8]
   - 1024 cols_helper_7_0[9] - 2048 cols_helper_7_0[10]
   - 4096 cols_helper_7_0[11] - 8192 cols_helper_7_0[12]
   - 16384 cols_helper_7_0[13] - 32768 cols_helper_7_0[14]
   - 65536 cols_helper_7_0[15] - 131072 cols_helper_7_0[16] = 0
 cols_acc_7_0: - 435 frac[7,0] + cols_7_0 >= 0
 accuracy_7_0: mem[7,0] - 0.01171875 cols_7_0 = 0
 accuracy_1_18: - 5.09765625 frac[1,18] + mem[1,18] = 0
 accuracy_3_4: - 5.09765625 frac[3,4] + mem[3,4] = 0
 accuracy_2_4: - 5.09765625 frac[2,4] + mem[2,4] = 0
 accuracy_5_9: - 5.09765625 frac[5,9] + mem[5,9] = 0
 accuracy_3_25: - 5.09765625 frac[3,25] + mem[3,25] = 0
 accuracy_4_18: - 5.09765625 frac[4,18] + mem[4,18] = 0
 cols_helper_7_7: cols_helper_7_7[0] + cols_helper_7_7[1]
   + cols_helper_7_7[2] + cols_helper_7_7[3] + cols_helper_7_7[4]
   + cols_helper_7_7[5] + cols_helper_7_7[6] + cols_helper_7_7[7]
   + cols_helper_7_7[8] + cols_helper_7_7[9] + cols_helper_7_7[10]
   + cols_helper_7_7[11] + cols_helper_7_7[12] + cols_helper_7_7[13]
   + cols_helper_7_7[14] + cols_helper_7_7[15] + cols_helper_7_7[16] <= 1
 cols_7_7: cols_7_7 - 2 cols_helper_7_7[0] - 4 cols_helper_7_7[1]
   - 8 cols_helper_7_7[2] - 16 cols_helper_7_7[3] - 32 cols_helper_7_7[4]
   - 64 cols_helper_7_7[5] - 128 cols_helper_7_7[6]
   - 256 cols_helper_7_7[7] - 512 cols_helper_7_7[8]
   - 1024 cols_helper_7_7[9] - 2048 cols_helper_7_7[10]
   - 4096 cols_helper_7_7[11] - 8192 cols_helper_7_7[12]
   - 16384 cols_helper_7_7[13] - 32768 cols_helper_7_7[14]
   - 65536 cols_helper_7_7[15] - 131072 cols_helper_7_7[16] = 0
 cols_acc_7_7: - 435 frac[7,7] + cols_7_7 >= 0
 accuracy_7_7: mem[7,7] - 0.01171875 cols_7_7 = 0
 cols_helper_7_16: cols_helper_7_16[0] + cols_helper_7_16[1]
   + cols_helper_7_16[2] + cols_helper_7_16[3] + cols_helper_7_16[4]
   + cols_helper_7_16[5] + cols_helper_7_16[6] + cols_helper_7_16[7]
   + cols_helper_7_16[8] + cols_helper_7_16[9] + cols_helper_7_16[10]
   + cols_helper_7_16[11] + cols_helper_7_16[12] + cols_helper_7_16[13]
   + cols_helper_7_16[14] + cols_helper_7_16[15] + cols_helper_7_16[16]
   <= 1
 cols_7_16: cols_7_16 - 2 cols_helper_7_16[0] - 4 cols_helper_7_16[1]
   - 8 cols_helper_7_16[2] - 16 cols_helper_7_16[3]
   - 32 cols_helper_7_16[4] - 64 cols_helper_7_16[5]
   - 128 cols_helper_7_16[6] - 256 cols_helper_7_16[7]
   - 512 cols_helper_7_16[8] - 1024 cols_helper_7_16[9]
   - 2048 cols_helper_7_16[10] - 4096 cols_helper_7_16[11]
   - 8192 cols_helper_7_16[12] - 16384 cols_helper_7_16[13]
   - 32768 cols_helper_7_16[14] - 65536 cols_helper_7_16[15]
   - 131072 cols_helper_7_16[16] = 0
 cols_acc_7_16: - 435 frac[7,16] + cols_7_16 >= 0
 accuracy_7_16: mem[7,16] - 0.01171875 cols_7_16 = 0
 accuracy_0_3: - 5.09765625 frac[0,3] + mem[0,3] = 0
 cols_helper_7_29: cols_helper_7_29[0] + cols_helper_7_29[1]
   + cols_helper_7_29[2] + cols_helper_7_29[3] + cols_helper_7_29[4]
   + cols_helper_7_29[5] + cols_helper_7_29[6] + cols_helper_7_29[7]
   + cols_helper_7_29[8] + cols_helper_7_29[9] + cols_helper_7_29[10]
   + cols_helper_7_29[11] + cols_helper_7_29[12] + cols_helper_7_29[13]
   + cols_helper_7_29[14] + cols_helper_7_29[15] + cols_helper_7_29[16]
   <= 1
 cols_7_29: cols_7_29 - 2 cols_helper_7_29[0] - 4 cols_helper_7_29[1]
   - 8 cols_helper_7_29[2] - 16 cols_helper_7_29[3]
   - 32 cols_helper_7_29[4] - 64 cols_helper_7_29[5]
   - 128 cols_helper_7_29[6] - 256 cols_helper_7_29[7]
   - 512 cols_helper_7_29[8] - 1024 cols_helper_7_29[9]
   - 2048 cols_helper_7_29[10] - 4096 cols_helper_7_29[11]
   - 8192 cols_helper_7_29[12] - 16384 cols_helper_7_29[13]
   - 32768 cols_helper_7_29[14] - 65536 cols_helper_7_29[15]
   - 131072 cols_helper_7_29[16] = 0
 cols_acc_7_29: - 435 frac[7,29] + cols_7_29 >= 0
 accuracy_7_29: mem[7,29] - 0.01171875 cols_7_29 = 0
 accuracy_1_2: - 5.09765625 frac[1,2] + mem[1,2] = 0
 accuracy_3_3: - 5.09765625 frac[3,3] + mem[3,3] = 0
 accuracy_2_9: - 5.09765625 frac[2,9] + mem[2,9] = 0
 accuracy_3_20: - 5.09765625 frac[3,20] + mem[3,20] = 0
 accuracy_4_23: - 5.09765625 frac[4,23] + mem[4,23] = 0
 cols_helper_7_10: cols_helper_7_10[0] + cols_helper_7_10[1]
   + cols_helper_7_10[2] + cols_helper_7_10[3] + cols_helper_7_10[4]
   + cols_helper_7_10[5] + cols_helper_7_10[6] + cols_helper_7_10[7]
   + cols_helper_7_10[8] + cols_helper_7_10[9] + cols_helper_7_10[10]
   + cols_helper_7_10[11] + cols_helper_7_10[12] + cols_helper_7_10[13]
   + cols_helper_7_10[14] + cols_helper_7_10[15] + cols_helper_7_10[16]
   <= 1
 cols_7_10: cols_7_10 - 2 cols_helper_7_10[0] - 4 cols_helper_7_10[1]
   - 8 cols_helper_7_10[2] - 16 cols_helper_7_10[3]
   - 32 cols_helper_7_10[4] - 64 cols_helper_7_10[5]
   - 128 cols_helper_7_10[6] - 256 cols_helper_7_10[7]
   - 512 cols_helper_7_10[8] - 1024 cols_helper_7_10[9]
   - 2048 cols_helper_7_10[10] - 4096 cols_helper_7_10[11]
   - 8192 cols_helper_7_10[12] - 16384 cols_helper_7_10[13]
   - 32768 cols_helper_7_10[14] - 65536 cols_helper_7_10[15]
   - 131072 cols_helper_7_10[16] = 0
 cols_acc_7_10: - 435 frac[7,10] + cols_7_10 >= 0
 accuracy_7_10: mem[7,10] - 0.01171875 cols_7_10 = 0
 accuracy_0_29: - 5.09765625 frac[0,29] + mem[0,29] = 0
 cols_helper_7_23: cols_helper_7_23[0] + cols_helper_7_23[1]
   + cols_helper_7_23[2] + cols_helper_7_23[3] + cols_helper_7_23[4]
   + cols_helper_7_23[5] + cols_helper_7_23[6] + cols_helper_7_23[7]
   + cols_helper_7_23[8] + cols_helper_7_23[9] + cols_helper_7_23[10]
   + cols_helper_7_23[11] + cols_helper_7_23[12] + cols_helper_7_23[13]
   + cols_helper_7_23[14] + cols_helper_7_23[15] + cols_helper_7_23[16]
   <= 1
 cols_7_23: cols_7_23 - 2 cols_helper_7_23[0] - 4 cols_helper_7_23[1]
   - 8 cols_helper_7_23[2] - 16 cols_helper_7_23[3]
   - 32 cols_helper_7_23[4] - 64 cols_helper_7_23[5]
   - 128 cols_helper_7_23[6] - 256 cols_helper_7_23[7]
   - 512 cols_helper_7_23[8] - 1024 cols_helper_7_23[9]
   - 2048 cols_helper_7_23[10] - 4096 cols_helper_7_23[11]
   - 8192 cols_helper_7_23[12] - 16384 cols_helper_7_23[13]
   - 32768 cols_helper_7_23[14] - 65536 cols_helper_7_23[15]
   - 131072 cols_helper_7_23[16] = 0
 cols_acc_7_23: - 435 frac[7,23] + cols_7_23 >= 0
 accuracy_7_23: mem[7,23] - 0.01171875 cols_7_23 = 0
 accuracy_1_13: - 5.09765625 frac[1,13] + mem[1,13] = 0
 accuracy_2_10: - 5.09765625 frac[2,10] + mem[2,10] = 0
 accuracy_5_28: - 5.09765625 frac[5,28] + mem[5,28] = 0
 cols_helper_7_9: cols_helper_7_9[0] + cols_helper_7_9[1]
   + cols_helper_7_9[2] + cols_helper_7_9[3] + cols_helper_7_9[4]
   + cols_helper_7_9[5] + cols_helper_7_9[6] + cols_helper_7_9[7]
   + cols_helper_7_9[8] + cols_helper_7_9[9] + cols_helper_7_9[10]
   + cols_helper_7_9[11] + cols_helper_7_9[12] + cols_helper_7_9[13]
   + cols_helper_7_9[14] + cols_helper_7_9[15] + cols_helper_7_9[16] <= 1
 cols_7_9: cols_7_9 - 2 cols_helper_7_9[0] - 4 cols_helper_7_9[1]
   - 8 cols_helper_7_9[2] - 16 cols_helper_7_9[3] - 32 cols_helper_7_9[4]
   - 64 cols_helper_7_9[5] - 128 cols_helper_7_9[6]
   - 256 cols_helper_7_9[7] - 512 cols_helper_7_9[8]
   - 1024 cols_helper_7_9[9] - 2048 cols_helper_7_9[10]
   - 4096 cols_helper_7_9[11] - 8192 cols_helper_7_9[12]
   - 16384 cols_helper_7_9[13] - 32768 cols_helper_7_9[14]
   - 65536 cols_helper_7_9[15] - 131072 cols_helper_7_9[16] = 0
 cols_acc_7_9: - 435 frac[7,9] + cols_7_9 >= 0
 accuracy_7_9: mem[7,9] - 0.01171875 cols_7_9 = 0
 cols_helper_7_26: cols_helper_7_26[0] + cols_helper_7_26[1]
   + cols_helper_7_26[2] + cols_helper_7_26[3] + cols_helper_7_26[4]
   + cols_helper_7_26[5] + cols_helper_7_26[6] + cols_helper_7_26[7]
   + cols_helper_7_26[8] + cols_helper_7_26[9] + cols_helper_7_26[10]
   + cols_helper_7_26[11] + cols_helper_7_26[12] + cols_helper_7_26[13]
   + cols_helper_7_26[14] + cols_helper_7_26[15] + cols_helper_7_26[16]
   <= 1
 cols_7_26: cols_7_26 - 2 cols_helper_7_26[0] - 4 cols_helper_7_26[1]
   - 8 cols_helper_7_26[2] - 16 cols_helper_7_26[3]
   - 32 cols_helper_7_26[4] - 64 cols_helper_7_26[5]
   - 128 cols_helper_7_26[6] - 256 cols_helper_7_26[7]
   - 512 cols_helper_7_26[8] - 1024 cols_helper_7_26[9]
   - 2048 cols_helper_7_26[10] - 4096 cols_helper_7_26[11]
   - 8192 cols_helper_7_26[12] - 16384 cols_helper_7_26[13]
   - 32768 cols_helper_7_26[14] - 65536 cols_helper_7_26[15]
   - 131072 cols_helper_7_26[16] = 0
 cols_acc_7_26: - 435 frac[7,26] + cols_7_26 >= 0
 accuracy_7_26: mem[7,26] - 0.01171875 cols_7_26 = 0
 accuracy_2_21: - 5.09765625 frac[2,21] + mem[2,21] = 0
 accuracy_1_27: - 5.09765625 frac[1,27] + mem[1,27] = 0
 accuracy_0_13: - 5.09765625 frac[0,13] + mem[0,13] = 0
 accuracy_3_8: - 5.09765625 frac[3,8] + mem[3,8] = 0
 accuracy_4_3: - 5.09765625 frac[4,3] + mem[4,3] = 0
 mem_tot_CPU1: - mem[0,10] - mem[0,4] - mem[0,24] - mem[0,5] - mem[0,15]
   - mem[0,25] - mem[0,3] - mem[0,29] - mem[0,13] + mem_tot_CPU1 = 0
 R252: - 3.05176e-05 mem_tot_CPU1 + normalized_mem_tot_CPU1 = 0
 rows_tot_CPU1: - 3 frac[0,10] - 3 frac[0,4] - 3 frac[0,24] - 3 frac[0,5]
   - 3 frac[0,15] - 3 frac[0,25] - 3 frac[0,3] - 3 frac[0,29]
   - 3 frac[0,13] + rows_tot_CPU1 = 0
 R254: - 0.0833333333333333 rows_tot_CPU1 + normalized_rows_tot_CPU1 = 0
 mem_tot_CPU6: - mem[1,17] - mem[1,28] - mem[1,1] - mem[1,15] - mem[1,4]
   - mem[1,9] - mem[1,18] - mem[1,2] - mem[1,13] - mem[1,27] + mem_tot_CPU6
   = 0
 R256: - 3.05176e-05 mem_tot_CPU6 + normalized_mem_tot_CPU6 = 0
 rows_tot_CPU6: - 3 frac[1,17] - 3 frac[1,28] - 3 frac[1,1] - 3 frac[1,15]
   - 3 frac[1,4] - 3 frac[1,9] - 3 frac[1,18] - 3 frac[1,2] - 3 frac[1,13]
   - 3 frac[1,27] + rows_tot_CPU6 = 0
 R258: - 0.0833333333333333 rows_tot_CPU6 + normalized_rows_tot_CPU6 = 0
 mem_tot_CPU9: - mem[2,27] - mem[2,17] - mem[2,1] - mem[2,23] - mem[2,24]
   - mem[2,29] - mem[2,8] - mem[2,3] - mem[2,4] - mem[2,9] - mem[2,10]
   - mem[2,21] + mem_tot_CPU9 = 0
 R260: - 3.05176e-05 mem_tot_CPU9 + normalized_mem_tot_CPU9 = 0
 rows_tot_CPU9: - 3 frac[2,27] - 3 frac[2,17] - 3 frac[2,1] - 3 frac[2,23]
   - 3 frac[2,24] - 3 frac[2,29] - 3 frac[2,8] - 3 frac[2,3] - 3 frac[2,4]
   - 3 frac[2,9] - 3 frac[2,10] - 3 frac[2,21] + rows_tot_CPU9 = 0
 R262: - 0.0833333333333333 rows_tot_CPU9 + normalized_rows_tot_CPU9 = 0
 mem_tot_CPU10: - mem[3,24] - mem[3,2] - mem[3,1] - mem[3,28] - mem[3,17]
   - mem[3,27] - mem[3,15] - mem[3,21] - mem[3,10] - mem[3,4] - mem[3,25]
   - mem[3,3] - mem[3,20] - mem[3,8] + mem_tot_CPU10 = 0
 R264: - 3.05176e-05 mem_tot_CPU10 + normalized_mem_tot_CPU10 = 0
 rows_tot_CPU10: - 3 frac[3,24] - 3 frac[3,2] - 3 frac[3,1] - 3 frac[3,28]
   - 3 frac[3,17] - 3 frac[3,27] - 3 frac[3,15] - 3 frac[3,21]
   - 3 frac[3,10] - 3 frac[3,4] - 3 frac[3,25] - 3 frac[3,3] - 3 frac[3,20]
   - 3 frac[3,8] + rows_tot_CPU10 = 0
 R266: - 0.0833333333333333 rows_tot_CPU10 + normalized_rows_tot_CPU10 = 0
 mem_tot_CPU11: - mem[4,10] - mem[4,15] - mem[4,21] - mem[4,28] - mem[4,27]
   - mem[4,2] - mem[4,18] - mem[4,23] - mem[4,3] + mem_tot_CPU11 = 0
 R268: - 3.05176e-05 mem_tot_CPU11 + normalized_mem_tot_CPU11 = 0
 rows_tot_CPU11: - 3 frac[4,10] - 3 frac[4,15] - 3 frac[4,21]
   - 3 frac[4,28] - 3 frac[4,27] - 3 frac[4,2] - 3 frac[4,18]
   - 3 frac[4,23] - 3 frac[4,3] + rows_tot_CPU11 = 0
 R270: - 0.0833333333333333 rows_tot_CPU11 + normalized_rows_tot_CPU11 = 0
 mem_tot_CPU18: - mem[5,5] - mem[5,18] - mem[5,1] - mem[5,13] - mem[5,29]
   - mem[5,3] - mem[5,9] - mem[5,28] + mem_tot_CPU18 = 0
 R272: - 3.05176e-05 mem_tot_CPU18 + normalized_mem_tot_CPU18 = 0
 rows_tot_CPU18: - 3 frac[5,5] - 3 frac[5,18] - 3 frac[5,1] - 3 frac[5,13]
   - 3 frac[5,29] - 3 frac[5,3] - 3 frac[5,9] - 3 frac[5,28]
   + rows_tot_CPU18 = 0
 R274: - 0.0833333333333333 rows_tot_CPU18 + normalized_rows_tot_CPU18 = 0
 mem_tot_CPU24: - mem[6,9] - mem[6,28] - mem[6,4] - mem[6,1] - mem[6,20]
   - mem[6,27] - mem[6,2] - mem[6,15] - mem[6,5] + mem_tot_CPU24 = 0
 R276: - 3.05176e-05 mem_tot_CPU24 + normalized_mem_tot_CPU24 = 0
 rows_tot_CPU24: - 3 frac[6,9] - 3 frac[6,28] - 3 frac[6,4] - 3 frac[6,1]
   - 3 frac[6,20] - 3 frac[6,27] - 3 frac[6,2] - 3 frac[6,15] - 3 frac[6,5]
   + rows_tot_CPU24 = 0
 R278: - 0.0833333333333333 rows_tot_CPU24 + normalized_rows_tot_CPU24 = 0
 mem_tot_tor_P42: - mem[7,3] - mem[7,12] - mem[7,25] - mem[7,6] - mem[7,19]
   - mem[7,28] - mem[7,22] - mem[7,8] - mem[7,21] - mem[7,2] - mem[7,15]
   - mem[7,24] - mem[7,1] - mem[7,18] - mem[7,4] - mem[7,17] - mem[7,11]
   - mem[7,20] - mem[7,14] - mem[7,27] - mem[7,0] - mem[7,7] - mem[7,16]
   - mem[7,29] - mem[7,10] - mem[7,23] - mem[7,9] - mem[7,26]
   + mem_tot_tor_P42 = 0
 R280: - 1.6276041666666666e-04 mem_tot_tor_P42
   + normalized_mem_tot_tor_P42 = 0
 rows_tot_tor_P42: - 3 frac[7,3] - 3 frac[7,12] - 3 frac[7,25]
   - 3 frac[7,6] - 3 frac[7,19] - 3 frac[7,28] - 3 frac[7,22] - 3 frac[7,8]
   - 3 frac[7,21] - 3 frac[7,2] - 3 frac[7,15] - 3 frac[7,24] - 3 frac[7,1]
   - 3 frac[7,18] - 3 frac[7,4] - 3 frac[7,17] - 3 frac[7,11]
   - 3 frac[7,20] - 3 frac[7,14] - 3 frac[7,27] - 3 frac[7,0] - 3 frac[7,7]
   - 3 frac[7,16] - 3 frac[7,29] - 3 frac[7,10] - 3 frac[7,23]
   - 3 frac[7,9] - 3 frac[7,26] + rows_tot_tor_P42 = 0
 R282: - 0.0208333333333333 rows_tot_tor_P42 + normalized_rows_tot_tor_P42
   = 0
 row_capacity_CPU1: rows_tot_CPU1 <= 12
 row_capacity_CPU6: rows_tot_CPU6 <= 12
 row_capacity_CPU9: rows_tot_CPU9 <= 12
 row_capacity_CPU10: rows_tot_CPU10 <= 12
 row_capacity_CPU11: rows_tot_CPU11 <= 12
 row_capacity_CPU18: rows_tot_CPU18 <= 12
 row_capacity_CPU24: rows_tot_CPU24 <= 12
 row_capacity_tor_P42: rows_tot_tor_P42 <= 48
 capacity_mem_par_tor_P42: mem[7,3] <= 1536
 capacity_mem_par_tor_P42: mem[7,12] <= 1536
 capacity_mem_par_tor_P42: mem[7,25] <= 1536
 capacity_mem_par_tor_P42: mem[7,6] <= 1536
 capacity_mem_par_tor_P42: mem[7,19] <= 1536
 capacity_mem_par_tor_P42: mem[7,28] <= 1536
 capacity_mem_par_tor_P42: mem[7,22] <= 1536
 capacity_mem_par_tor_P42: mem[7,8] <= 1536
 capacity_mem_par_tor_P42: mem[7,21] <= 1536
 capacity_mem_par_tor_P42: mem[7,2] <= 1536
 capacity_mem_par_tor_P42: mem[7,15] <= 1536
 capacity_mem_par_tor_P42: mem[7,24] <= 1536
 capacity_mem_par_tor_P42: mem[7,1] <= 1536
 capacity_mem_par_tor_P42: mem[7,18] <= 1536
 capacity_mem_par_tor_P42: mem[7,4] <= 1536
 capacity_mem_par_tor_P42: mem[7,17] <= 1536
 capacity_mem_par_tor_P42: mem[7,11] <= 1536
 capacity_mem_par_tor_P42: mem[7,20] <= 1536
 capacity_mem_par_tor_P42: mem[7,14] <= 1536
 capacity_mem_par_tor_P42: mem[7,27] <= 1536
 capacity_mem_par_tor_P42: mem[7,0] <= 1536
 capacity_mem_par_tor_P42: mem[7,7] <= 1536
 capacity_mem_par_tor_P42: mem[7,16] <= 1536
 capacity_mem_par_tor_P42: mem[7,29] <= 1536
 capacity_mem_par_tor_P42: mem[7,10] <= 1536
 capacity_mem_par_tor_P42: mem[7,23] <= 1536
 capacity_mem_par_tor_P42: mem[7,9] <= 1536
 capacity_mem_par_tor_P42: mem[7,26] <= 1536
 ns_single_CPU1: 3.5 rows_tot_CPU1 - ns_single_CPU1 + pdt_m_rows_CPU1 = 0
 capacity_cores_CPU1: cores_sketch_CPU1 + cores_dpdk_CPU1 <= 7
 ns_sketch_CPU1: - ns_single_CPU1 + 12.34709234276538 cores_sketch_CPU1
   >= 0
 ns_dpdk_CPU1: - 5.204235199908235 cores_dpdk_CPU1 <= -21.42857142857143
 ns_single_CPU6: 3.5 rows_tot_CPU6 - ns_single_CPU6 + pdt_m_rows_CPU6 = 0
 capacity_cores_CPU6: cores_sketch_CPU6 + cores_dpdk_CPU6 <= 7
 ns_sketch_CPU6: - ns_single_CPU6 + 12.34709234276538 cores_sketch_CPU6
   >= 0
 ns_dpdk_CPU6: - 5.204235199908235 cores_dpdk_CPU6 <= -21.42857142857143
 ns_single_CPU9: 3.5 rows_tot_CPU9 - ns_single_CPU9 + pdt_m_rows_CPU9 = 0
 capacity_cores_CPU9: cores_sketch_CPU9 + cores_dpdk_CPU9 <= 7
 ns_sketch_CPU9: - ns_single_CPU9 + 12.34709234276538 cores_sketch_CPU9
   >= 0
 ns_dpdk_CPU9: - 5.204235199908235 cores_dpdk_CPU9 <= -21.42857142857143
 ns_single_CPU10: 3.5 rows_tot_CPU10 - ns_single_CPU10 + pdt_m_rows_CPU10
   = 0
 capacity_cores_CPU10: cores_sketch_CPU10 + cores_dpdk_CPU10 <= 7
 ns_sketch_CPU10: - ns_single_CPU10 + 12.34709234276538 cores_sketch_CPU10
   >= 0
 ns_dpdk_CPU10: - 5.204235199908235 cores_dpdk_CPU10 <= -21.42857142857143
 ns_single_CPU11: 3.5 rows_tot_CPU11 - ns_single_CPU11 + pdt_m_rows_CPU11
   = 0
 capacity_cores_CPU11: cores_sketch_CPU11 + cores_dpdk_CPU11 <= 7
 ns_sketch_CPU11: - ns_single_CPU11 + 12.34709234276538 cores_sketch_CPU11
   >= 0
 ns_dpdk_CPU11: - 5.204235199908235 cores_dpdk_CPU11 <= -21.42857142857143
 ns_single_CPU18: 3.5 rows_tot_CPU18 - ns_single_CPU18 + pdt_m_rows_CPU18
   = 0
 capacity_cores_CPU18: cores_sketch_CPU18 + cores_dpdk_CPU18 <= 7
 ns_sketch_CPU18: - ns_single_CPU18 + 12.34709234276538 cores_sketch_CPU18
   >= 0
 ns_dpdk_CPU18: - 5.204235199908235 cores_dpdk_CPU18 <= -21.42857142857143
 ns_single_CPU24: 3.5 rows_tot_CPU24 - ns_single_CPU24 + pdt_m_rows_CPU24
   = 0
 capacity_cores_CPU24: cores_sketch_CPU24 + cores_dpdk_CPU24 <= 7
 ns_sketch_CPU24: - ns_single_CPU24 + 12.34709234276538 cores_sketch_CPU24
   >= 0
 ns_dpdk_CPU24: - 5.204235199908235 cores_dpdk_CPU24 <= -21.42857142857143
 res_acc: 1.220703125e-04 mem_tot_CPU1 + 1.220703125e-04 mem_tot_CPU6
   + 1.220703125e-04 mem_tot_CPU9 + 1.220703125e-04 mem_tot_CPU10
   + 1.220703125e-04 mem_tot_CPU11 + 1.220703125e-04 mem_tot_CPU18
   + 1.220703125e-04 mem_tot_CPU24 + 0.001953125 mem_tot_tor_P42
   + 0.25 rows_tot_tor_P42 + 10 cores_sketch_CPU1 + 10 cores_dpdk_CPU1
   + 10 cores_sketch_CPU6 + 10 cores_dpdk_CPU6 + 10 cores_sketch_CPU9
   + 10 cores_dpdk_CPU9 + 10 cores_sketch_CPU10 + 10 cores_dpdk_CPU10
   + 10 cores_sketch_CPU11 + 10 cores_dpdk_CPU11 + 10 cores_sketch_CPU18
   + 10 cores_dpdk_CPU18 + 10 cores_sketch_CPU24 + 10 cores_dpdk_CPU24
   - res = 0
 pdt_m_rows_CPU1: pdt_m_rows_CPU1 + [ - rows_tot_CPU1 * m_access_time_CPU1
   ] = 0
 pdt_m_rows_CPU6: pdt_m_rows_CPU6 + [ - rows_tot_CPU6 * m_access_time_CPU6
   ] = 0
 pdt_m_rows_CPU9: pdt_m_rows_CPU9 + [ - rows_tot_CPU9 * m_access_time_CPU9
   ] = 0
 pdt_m_rows_CPU10: pdt_m_rows_CPU10 + [
   - rows_tot_CPU10 * m_access_time_CPU10 ] = 0
 pdt_m_rows_CPU11: pdt_m_rows_CPU11 + [
   - rows_tot_CPU11 * m_access_time_CPU11 ] = 0
 pdt_m_rows_CPU18: pdt_m_rows_CPU18 + [
   - rows_tot_CPU18 * m_access_time_CPU18 ] = 0
 pdt_m_rows_CPU24: pdt_m_rows_CPU24 + [
   - rows_tot_CPU24 * m_access_time_CPU24 ] = 0
Bounds
 frac[7,3] <= 1
 frac[6,9] <= 1
 frac[7,12] <= 1
 frac[6,28] <= 1
 frac[1,17] <= 1
 frac[7,25] <= 1
 frac[0,10] <= 1
 frac[3,24] <= 1
 frac[5,5] <= 1
 frac[7,6] <= 1
 frac[5,18] <= 1
 frac[7,19] <= 1
 frac[1,28] <= 1
 frac[0,4] <= 1
 frac[7,28] <= 1
 frac[2,27] <= 1
 frac[1,1] <= 1
 frac[4,10] <= 1
 frac[3,2] <= 1
 frac[7,22] <= 1
 frac[2,17] <= 1
 frac[4,15] <= 1
 frac[3,1] <= 1
 frac[4,21] <= 1
 frac[7,8] <= 1
 frac[7,21] <= 1
 frac[2,1] <= 1
 frac[1,15] <= 1
 frac[3,28] <= 1
 frac[5,1] <= 1
 frac[3,17] <= 1
 frac[7,2] <= 1
 frac[7,15] <= 1
 frac[0,24] <= 1
 frac[7,24] <= 1
 frac[2,23] <= 1
 frac[3,27] <= 1
 frac[6,4] <= 1
 frac[4,28] <= 1
 frac[7,1] <= 1
 frac[7,18] <= 1
 frac[0,5] <= 1
 frac[2,24] <= 1
 frac[6,1] <= 1
 frac[7,4] <= 1
 frac[6,20] <= 1
 frac[7,17] <= 1
 frac[6,27] <= 1
 frac[3,15] <= 1
 frac[2,29] <= 1
 frac[2,8] <= 1
 frac[5,13] <= 1
 frac[4,27] <= 1
 frac[3,21] <= 1
 frac[6,2] <= 1
 frac[7,11] <= 1
 frac[7,20] <= 1
 frac[0,15] <= 1
 frac[3,10] <= 1
 frac[6,15] <= 1
 frac[5,29] <= 1
 frac[7,14] <= 1
 frac[0,25] <= 1
 frac[7,27] <= 1
 frac[1,4] <= 1
 frac[2,3] <= 1
 frac[1,9] <= 1
 frac[4,2] <= 1
 frac[6,5] <= 1
 frac[5,3] <= 1
 frac[7,0] <= 1
 frac[1,18] <= 1
 frac[3,4] <= 1
 frac[2,4] <= 1
 frac[5,9] <= 1
 frac[3,25] <= 1
 frac[4,18] <= 1
 frac[7,7] <= 1
 frac[7,16] <= 1
 frac[0,3] <= 1
 frac[7,29] <= 1
 frac[1,2] <= 1
 frac[3,3] <= 1
 frac[2,9] <= 1
 frac[3,20] <= 1
 frac[4,23] <= 1
 frac[7,10] <= 1
 frac[0,29] <= 1
 frac[7,23] <= 1
 frac[1,13] <= 1
 frac[2,10] <= 1
 frac[5,28] <= 1
 frac[7,9] <= 1
 frac[7,26] <= 1
 frac[2,21] <= 1
 frac[1,27] <= 1
 frac[0,13] <= 1
 frac[3,8] <= 1
 frac[4,3] <= 1
 cols_7_3 <= 131072
 cols_7_12 <= 131072
 cols_7_25 <= 131072
 cols_7_6 <= 131072
 cols_7_19 <= 131072
 cols_7_28 <= 131072
 cols_7_22 <= 131072
 cols_7_8 <= 131072
 cols_7_21 <= 131072
 cols_7_2 <= 131072
 cols_7_15 <= 131072
 cols_7_24 <= 131072
 cols_7_1 <= 131072
 cols_7_18 <= 131072
 cols_7_4 <= 131072
 cols_7_17 <= 131072
 cols_7_11 <= 131072
 cols_7_20 <= 131072
 cols_7_14 <= 131072
 cols_7_27 <= 131072
 cols_7_0 <= 131072
 cols_7_7 <= 131072
 cols_7_16 <= 131072
 cols_7_29 <= 131072
 cols_7_10 <= 131072
 cols_7_23 <= 131072
 cols_7_9 <= 131072
 cols_7_26 <= 131072
 mem_tot_CPU1 <= 32768
 normalized_mem_tot_CPU1 <= 1
 normalized_rows_tot_CPU1 <= 1
 mem_tot_CPU6 <= 32768
 normalized_mem_tot_CPU6 <= 1
 normalized_rows_tot_CPU6 <= 1
 mem_tot_CPU9 <= 32768
 normalized_mem_tot_CPU9 <= 1
 normalized_rows_tot_CPU9 <= 1
 mem_tot_CPU10 <= 32768
 normalized_mem_tot_CPU10 <= 1
 normalized_rows_tot_CPU10 <= 1
 mem_tot_CPU11 <= 32768
 normalized_mem_tot_CPU11 <= 1
 normalized_rows_tot_CPU11 <= 1
 mem_tot_CPU18 <= 32768
 normalized_mem_tot_CPU18 <= 1
 normalized_rows_tot_CPU18 <= 1
 mem_tot_CPU24 <= 32768
 normalized_mem_tot_CPU24 <= 1
 normalized_rows_tot_CPU24 <= 1
 mem_tot_tor_P42 <= 6144
 normalized_mem_tot_tor_P42 <= 1
 normalized_rows_tot_tor_P42 <= 1
 cores_sketch_CPU1 <= 7
 1 <= cores_dpdk_CPU1 <= 7
 cores_sketch_CPU6 <= 7
 1 <= cores_dpdk_CPU6 <= 7
 cores_sketch_CPU9 <= 7
 1 <= cores_dpdk_CPU9 <= 7
 cores_sketch_CPU10 <= 7
 1 <= cores_dpdk_CPU10 <= 7
 cores_sketch_CPU11 <= 7
 1 <= cores_dpdk_CPU11 <= 7
 cores_sketch_CPU18 <= 7
 1 <= cores_dpdk_CPU18 <= 7
 cores_sketch_CPU24 <= 7
 1 <= cores_dpdk_CPU24 <= 7
Binaries
 cols_helper_7_3[0] cols_helper_7_3[1] cols_helper_7_3[2]
 cols_helper_7_3[3] cols_helper_7_3[4] cols_helper_7_3[5]
 cols_helper_7_3[6] cols_helper_7_3[7] cols_helper_7_3[8]
 cols_helper_7_3[9] cols_helper_7_3[10] cols_helper_7_3[11]
 cols_helper_7_3[12] cols_helper_7_3[13] cols_helper_7_3[14]
 cols_helper_7_3[15] cols_helper_7_3[16] cols_helper_7_12[0]
 cols_helper_7_12[1] cols_helper_7_12[2] cols_helper_7_12[3]
 cols_helper_7_12[4] cols_helper_7_12[5] cols_helper_7_12[6]
 cols_helper_7_12[7] cols_helper_7_12[8] cols_helper_7_12[9]
 cols_helper_7_12[10] cols_helper_7_12[11] cols_helper_7_12[12]
 cols_helper_7_12[13] cols_helper_7_12[14] cols_helper_7_12[15]
 cols_helper_7_12[16] cols_helper_7_25[0] cols_helper_7_25[1]
 cols_helper_7_25[2] cols_helper_7_25[3] cols_helper_7_25[4]
 cols_helper_7_25[5] cols_helper_7_25[6] cols_helper_7_25[7]
 cols_helper_7_25[8] cols_helper_7_25[9] cols_helper_7_25[10]
 cols_helper_7_25[11] cols_helper_7_25[12] cols_helper_7_25[13]
 cols_helper_7_25[14] cols_helper_7_25[15] cols_helper_7_25[16]
 cols_helper_7_6[0] cols_helper_7_6[1] cols_helper_7_6[2]
 cols_helper_7_6[3] cols_helper_7_6[4] cols_helper_7_6[5]
 cols_helper_7_6[6] cols_helper_7_6[7] cols_helper_7_6[8]
 cols_helper_7_6[9] cols_helper_7_6[10] cols_helper_7_6[11]
 cols_helper_7_6[12] cols_helper_7_6[13] cols_helper_7_6[14]
 cols_helper_7_6[15] cols_helper_7_6[16] cols_helper_7_19[0]
 cols_helper_7_19[1] cols_helper_7_19[2] cols_helper_7_19[3]
 cols_helper_7_19[4] cols_helper_7_19[5] cols_helper_7_19[6]
 cols_helper_7_19[7] cols_helper_7_19[8] cols_helper_7_19[9]
 cols_helper_7_19[10] cols_helper_7_19[11] cols_helper_7_19[12]
 cols_helper_7_19[13] cols_helper_7_19[14] cols_helper_7_19[15]
 cols_helper_7_19[16] cols_helper_7_28[0] cols_helper_7_28[1]
 cols_helper_7_28[2] cols_helper_7_28[3] cols_helper_7_28[4]
 cols_helper_7_28[5] cols_helper_7_28[6] cols_helper_7_28[7]
 cols_helper_7_28[8] cols_helper_7_28[9] cols_helper_7_28[10]
 cols_helper_7_28[11] cols_helper_7_28[12] cols_helper_7_28[13]
 cols_helper_7_28[14] cols_helper_7_28[15] cols_helper_7_28[16]
 cols_helper_7_22[0] cols_helper_7_22[1] cols_helper_7_22[2]
 cols_helper_7_22[3] cols_helper_7_22[4] cols_helper_7_22[5]
 cols_helper_7_22[6] cols_helper_7_22[7] cols_helper_7_22[8]
 cols_helper_7_22[9] cols_helper_7_22[10] cols_helper_7_22[11]
 cols_helper_7_22[12] cols_helper_7_22[13] cols_helper_7_22[14]
 cols_helper_7_22[15] cols_helper_7_22[16] cols_helper_7_8[0]
 cols_helper_7_8[1] cols_helper_7_8[2] cols_helper_7_8[3]
 cols_helper_7_8[4] cols_helper_7_8[5] cols_helper_7_8[6]
 cols_helper_7_8[7] cols_helper_7_8[8] cols_helper_7_8[9]
 cols_helper_7_8[10] cols_helper_7_8[11] cols_helper_7_8[12]
 cols_helper_7_8[13] cols_helper_7_8[14] cols_helper_7_8[15]
 cols_helper_7_8[16] cols_helper_7_21[0] cols_helper_7_21[1]
 cols_helper_7_21[2] cols_helper_7_21[3] cols_helper_7_21[4]
 cols_helper_7_21[5] cols_helper_7_21[6] cols_helper_7_21[7]
 cols_helper_7_21[8] cols_helper_7_21[9] cols_helper_7_21[10]
 cols_helper_7_21[11] cols_helper_7_21[12] cols_helper_7_21[13]
 cols_helper_7_21[14] cols_helper_7_21[15] cols_helper_7_21[16]
 cols_helper_7_2[0] cols_helper_7_2[1] cols_helper_7_2[2]
 cols_helper_7_2[3] cols_helper_7_2[4] cols_helper_7_2[5]
 cols_helper_7_2[6] cols_helper_7_2[7] cols_helper_7_2[8]
 cols_helper_7_2[9] cols_helper_7_2[10] cols_helper_7_2[11]
 cols_helper_7_2[12] cols_helper_7_2[13] cols_helper_7_2[14]
 cols_helper_7_2[15] cols_helper_7_2[16] cols_helper_7_15[0]
 cols_helper_7_15[1] cols_helper_7_15[2] cols_helper_7_15[3]
 cols_helper_7_15[4] cols_helper_7_15[5] cols_helper_7_15[6]
 cols_helper_7_15[7] cols_helper_7_15[8] cols_helper_7_15[9]
 cols_helper_7_15[10] cols_helper_7_15[11] cols_helper_7_15[12]
 cols_helper_7_15[13] cols_helper_7_15[14] cols_helper_7_15[15]
 cols_helper_7_15[16] cols_helper_7_24[0] cols_helper_7_24[1]
 cols_helper_7_24[2] cols_helper_7_24[3] cols_helper_7_24[4]
 cols_helper_7_24[5] cols_helper_7_24[6] cols_helper_7_24[7]
 cols_helper_7_24[8] cols_helper_7_24[9] cols_helper_7_24[10]
 cols_helper_7_24[11] cols_helper_7_24[12] cols_helper_7_24[13]
 cols_helper_7_24[14] cols_helper_7_24[15] cols_helper_7_24[16]
 cols_helper_7_1[0] cols_helper_7_1[1] cols_helper_7_1[2]
 cols_helper_7_1[3] cols_helper_7_1[4] cols_helper_7_1[5]
 cols_helper_7_1[6] cols_helper_7_1[7] cols_helper_7_1[8]
 cols_helper_7_1[9] cols_helper_7_1[10] cols_helper_7_1[11]
 cols_helper_7_1[12] cols_helper_7_1[13] cols_helper_7_1[14]
 cols_helper_7_1[15] cols_helper_7_1[16] cols_helper_7_18[0]
 cols_helper_7_18[1] cols_helper_7_18[2] cols_helper_7_18[3]
 cols_helper_7_18[4] cols_helper_7_18[5] cols_helper_7_18[6]
 cols_helper_7_18[7] cols_helper_7_18[8] cols_helper_7_18[9]
 cols_helper_7_18[10] cols_helper_7_18[11] cols_helper_7_18[12]
 cols_helper_7_18[13] cols_helper_7_18[14] cols_helper_7_18[15]
 cols_helper_7_18[16] cols_helper_7_4[0] cols_helper_7_4[1]
 cols_helper_7_4[2] cols_helper_7_4[3] cols_helper_7_4[4]
 cols_helper_7_4[5] cols_helper_7_4[6] cols_helper_7_4[7]
 cols_helper_7_4[8] cols_helper_7_4[9] cols_helper_7_4[10]
 cols_helper_7_4[11] cols_helper_7_4[12] cols_helper_7_4[13]
 cols_helper_7_4[14] cols_helper_7_4[15] cols_helper_7_4[16]
 cols_helper_7_17[0] cols_helper_7_17[1] cols_helper_7_17[2]
 cols_helper_7_17[3] cols_helper_7_17[4] cols_helper_7_17[5]
 cols_helper_7_17[6] cols_helper_7_17[7] cols_helper_7_17[8]
 cols_helper_7_17[9] cols_helper_7_17[10] cols_helper_7_17[11]
 cols_helper_7_17[12] cols_helper_7_17[13] cols_helper_7_17[14]
 cols_helper_7_17[15] cols_helper_7_17[16] cols_helper_7_11[0]
 cols_helper_7_11[1] cols_helper_7_11[2] cols_helper_7_11[3]
 cols_helper_7_11[4] cols_helper_7_11[5] cols_helper_7_11[6]
 cols_helper_7_11[7] cols_helper_7_11[8] cols_helper_7_11[9]
 cols_helper_7_11[10] cols_helper_7_11[11] cols_helper_7_11[12]
 cols_helper_7_11[13] cols_helper_7_11[14] cols_helper_7_11[15]
 cols_helper_7_11[16] cols_helper_7_20[0] cols_helper_7_20[1]
 cols_helper_7_20[2] cols_helper_7_20[3] cols_helper_7_20[4]
 cols_helper_7_20[5] cols_helper_7_20[6] cols_helper_7_20[7]
 cols_helper_7_20[8] cols_helper_7_20[9] cols_helper_7_20[10]
 cols_helper_7_20[11] cols_helper_7_20[12] cols_helper_7_20[13]
 cols_helper_7_20[14] cols_helper_7_20[15] cols_helper_7_20[16]
 cols_helper_7_14[0] cols_helper_7_14[1] cols_helper_7_14[2]
 cols_helper_7_14[3] cols_helper_7_14[4] cols_helper_7_14[5]
 cols_helper_7_14[6] cols_helper_7_14[7] cols_helper_7_14[8]
 cols_helper_7_14[9] cols_helper_7_14[10] cols_helper_7_14[11]
 cols_helper_7_14[12] cols_helper_7_14[13] cols_helper_7_14[14]
 cols_helper_7_14[15] cols_helper_7_14[16] cols_helper_7_27[0]
 cols_helper_7_27[1] cols_helper_7_27[2] cols_helper_7_27[3]
 cols_helper_7_27[4] cols_helper_7_27[5] cols_helper_7_27[6]
 cols_helper_7_27[7] cols_helper_7_27[8] cols_helper_7_27[9]
 cols_helper_7_27[10] cols_helper_7_27[11] cols_helper_7_27[12]
 cols_helper_7_27[13] cols_helper_7_27[14] cols_helper_7_27[15]
 cols_helper_7_27[16] cols_helper_7_0[0] cols_helper_7_0[1]
 cols_helper_7_0[2] cols_helper_7_0[3] cols_helper_7_0[4]
 cols_helper_7_0[5] cols_helper_7_0[6] cols_helper_7_0[7]
 cols_helper_7_0[8] cols_helper_7_0[9] cols_helper_7_0[10]
 cols_helper_7_0[11] cols_helper_7_0[12] cols_helper_7_0[13]
 cols_helper_7_0[14] cols_helper_7_0[15] cols_helper_7_0[16]
 cols_helper_7_7[0] cols_helper_7_7[1] cols_helper_7_7[2]
 cols_helper_7_7[3] cols_helper_7_7[4] cols_helper_7_7[5]
 cols_helper_7_7[6] cols_helper_7_7[7] cols_helper_7_7[8]
 cols_helper_7_7[9] cols_helper_7_7[10] cols_helper_7_7[11]
 cols_helper_7_7[12] cols_helper_7_7[13] cols_helper_7_7[14]
 cols_helper_7_7[15] cols_helper_7_7[16] cols_helper_7_16[0]
 cols_helper_7_16[1] cols_helper_7_16[2] cols_helper_7_16[3]
 cols_helper_7_16[4] cols_helper_7_16[5] cols_helper_7_16[6]
 cols_helper_7_16[7] cols_helper_7_16[8] cols_helper_7_16[9]
 cols_helper_7_16[10] cols_helper_7_16[11] cols_helper_7_16[12]
 cols_helper_7_16[13] cols_helper_7_16[14] cols_helper_7_16[15]
 cols_helper_7_16[16] cols_helper_7_29[0] cols_helper_7_29[1]
 cols_helper_7_29[2] cols_helper_7_29[3] cols_helper_7_29[4]
 cols_helper_7_29[5] cols_helper_7_29[6] cols_helper_7_29[7]
 cols_helper_7_29[8] cols_helper_7_29[9] cols_helper_7_29[10]
 cols_helper_7_29[11] cols_helper_7_29[12] cols_helper_7_29[13]
 cols_helper_7_29[14] cols_helper_7_29[15] cols_helper_7_29[16]
 cols_helper_7_10[0] cols_helper_7_10[1] cols_helper_7_10[2]
 cols_helper_7_10[3] cols_helper_7_10[4] cols_helper_7_10[5]
 cols_helper_7_10[6] cols_helper_7_10[7] cols_helper_7_10[8]
 cols_helper_7_10[9] cols_helper_7_10[10] cols_helper_7_10[11]
 cols_helper_7_10[12] cols_helper_7_10[13] cols_helper_7_10[14]
 cols_helper_7_10[15] cols_helper_7_10[16] cols_helper_7_23[0]
 cols_helper_7_23[1] cols_helper_7_23[2] cols_helper_7_23[3]
 cols_helper_7_23[4] cols_helper_7_23[5] cols_helper_7_23[6]
 cols_helper_7_23[7] cols_helper_7_23[8] cols_helper_7_23[9]
 cols_helper_7_23[10] cols_helper_7_23[11] cols_helper_7_23[12]
 cols_helper_7_23[13] cols_helper_7_23[14] cols_helper_7_23[15]
 cols_helper_7_23[16] cols_helper_7_9[0] cols_helper_7_9[1]
 cols_helper_7_9[2] cols_helper_7_9[3] cols_helper_7_9[4]
 cols_helper_7_9[5] cols_helper_7_9[6] cols_helper_7_9[7]
 cols_helper_7_9[8] cols_helper_7_9[9] cols_helper_7_9[10]
 cols_helper_7_9[11] cols_helper_7_9[12] cols_helper_7_9[13]
 cols_helper_7_9[14] cols_helper_7_9[15] cols_helper_7_9[16]
 cols_helper_7_26[0] cols_helper_7_26[1] cols_helper_7_26[2]
 cols_helper_7_26[3] cols_helper_7_26[4] cols_helper_7_26[5]
 cols_helper_7_26[6] cols_helper_7_26[7] cols_helper_7_26[8]
 cols_helper_7_26[9] cols_helper_7_26[10] cols_helper_7_26[11]
 cols_helper_7_26[12] cols_helper_7_26[13] cols_helper_7_26[14]
 cols_helper_7_26[15] cols_helper_7_26[16]
Generals
 cols_7_3 cols_7_12 cols_7_25 cols_7_6 cols_7_19 cols_7_28 cols_7_22
 cols_7_8 cols_7_21 cols_7_2 cols_7_15 cols_7_24 cols_7_1 cols_7_18
 cols_7_4 cols_7_17 cols_7_11 cols_7_20 cols_7_14 cols_7_27 cols_7_0
 cols_7_7 cols_7_16 cols_7_29 cols_7_10 cols_7_23 cols_7_9 cols_7_26
 cores_sketch_CPU1 cores_dpdk_CPU1 cores_sketch_CPU6 cores_dpdk_CPU6
 cores_sketch_CPU9 cores_dpdk_CPU9 cores_sketch_CPU10 cores_dpdk_CPU10
 cores_sketch_CPU11 cores_dpdk_CPU11 cores_sketch_CPU18 cores_dpdk_CPU18
 cores_sketch_CPU24 cores_dpdk_CPU24
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_tor_P42 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU1 , normalized_mem_tot_CPU6 ,
   normalized_mem_tot_CPU9 , normalized_mem_tot_CPU10 ,
   normalized_mem_tot_CPU11 , normalized_mem_tot_CPU18 ,
   normalized_mem_tot_CPU24 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_tor_P42 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU1 ,
   normalized_rows_tot_CPU6 , normalized_rows_tot_CPU9 ,
   normalized_rows_tot_CPU10 , normalized_rows_tot_CPU11 ,
   normalized_rows_tot_CPU18 , normalized_rows_tot_CPU24 )
 mem_access_time_CPU1: m_access_time_CPU1 = PWL ( mem_tot_CPU1 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU6: m_access_time_CPU6 = PWL ( mem_tot_CPU6 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU9: m_access_time_CPU9 = PWL ( mem_tot_CPU9 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU10: m_access_time_CPU10 = PWL ( mem_tot_CPU10 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU11: m_access_time_CPU11 = PWL ( mem_tot_CPU11 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU18: m_access_time_CPU18 = PWL ( mem_tot_CPU18 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU24: m_access_time_CPU24 = PWL ( mem_tot_CPU24 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

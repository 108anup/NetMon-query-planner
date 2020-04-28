\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_P4 + 0 max_mem_CPU_Cluster + 0 max_rows_P4
   + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU2 + 0 m_access_time_CPU3
   + 0 m_access_time_CPU4 + 0 m_access_time_CPU7 + 0 m_access_time_CPU17
   + 0 m_access_time_CPU18 + 0 m_access_time_CPU24 + 0 m_access_time_CPU26
   + res
Subject To
 cov_0_70: frac[10,70] + frac[8,70] >= 0.482903666666666
 cov_1_63: frac[8,63] + frac[10,63] >= 0.482903666666666
 cov_2_24: frac[9,24] >= 0.8
 cov_3_53: frac[10,53] + frac[11,53] + frac[9,53] >= 0.482903666666666
 cov_4_46: frac[11,46] + frac[9,46] + frac[10,46] >= 1
 cov_5_28: frac[8,28] >= 0.1
 cov_6_8: frac[11,8] + frac[10,8] >= 1
 cov_7_59: frac[10,59] + frac[8,59] >= 0.482903666666666
 cov_8_60: frac[9,60] + frac[10,60] + frac[11,60] >= 0.1
 cov_9_60: frac[9,60] + frac[8,60] + frac[10,60] + frac[11,60] >= 0.1
 cov_10_35: frac[8,35] + frac[9,35] + frac[10,35] + frac[11,35] >= 0.1
 cov_11_13: frac[9,13] + frac[10,13] + frac[11,13] >= 1
 cov_12_55: frac[10,55] + frac[8,55] >= 0.8
 cov_13_50: frac[9,50] + frac[11,50] + frac[10,50] >= 0.482903666666666
 cov_14_55: frac[9,55] >= 0.7
 cov_15_35: frac[8,35] + frac[10,35] >= 0.1
 cov_16_1: frac[11,1] + frac[9,1] + frac[10,1] >= 0.8
 cov_17_53: frac[8,53] >= 0.482903666666666
 cov_18_59: frac[10,59] + frac[8,59] >= 0.482903666666666
 cov_19_50: frac[9,50] + frac[11,50] + frac[10,50] >= 0.482903666666666
 cov_20_59: frac[9,59] + frac[10,59] + frac[11,59] >= 0.482903666666666
 cov_21_50: frac[8,50] + frac[10,50] >= 0.482903666666666
 cov_22_28: frac[11,28] + frac[9,28] + frac[8,28] + frac[10,28] >= 0.1
 cov_23_4: frac[9,4] >= 0.482903666666666
 cov_24_60: frac[8,60] + frac[10,60] >= 0.1
 cov_25_15: frac[10,15] + frac[8,15] >= 0.8
 cov_26_37: frac[8,37] + frac[10,37] + frac[11,37] >= 0.3
 cov_27_20: frac[9,20] >= 0.7
 cov_28_68: frac[9,68] >= 0.1
 cov_29_17: frac[9,17] >= 0.2
 cov_30_26: frac[11,26] + frac[10,26] + frac[8,26] >= 0.0329036666666662
 cov_31_17: frac[10,17] + frac[8,17] + frac[11,17] >= 0.2
 cov_32_48: frac[11,48] + frac[10,48] + frac[8,48] >= 0.8
 cov_33_37: frac[10,37] + frac[11,37] >= 0.3
 cov_34_44: frac[9,44] >= 0.1
 cov_35_11: frac[11,11] + frac[10,11] >= 0.8
 cov_36_40: frac[8,40] + frac[11,40] + frac[10,40] >= 0.1
 cov_37_37: frac[9,37] + frac[10,37] + frac[11,37] >= 0.3
 cov_38_44: frac[10,44] + frac[11,44] >= 0.1
 cov_39_37: frac[9,37] >= 0.3
 cov_40_72: frac[11,72] + frac[9,72] + frac[10,72] + frac[8,72] >= 0.9
 cov_41_10: frac[11,10] + frac[10,10] >= 0.7
 cov_42_10: frac[9,10] >= 0.7
 cov_43_37: frac[8,37] + frac[10,37] + frac[11,37] >= 0.3
 cov_44_44: frac[9,44] >= 0.1
 cov_45_62: frac[10,62] + frac[8,62] + frac[11,62] >= 0.0329036666666662
 cov_46_38: frac[9,38] + frac[11,38] + frac[10,38] >= 1
 cov_47_52: frac[11,52] + frac[10,52] >= 1
 cov_48_41: frac[10,41] + frac[8,41] >= 0.7
 cov_49_23: frac[11,23] + frac[10,23] + frac[8,23] >= 0.0329036666666662
 cov_50_67: frac[10,67] + frac[8,67] >= 0.9
 cov_51_73: frac[10,73] + frac[8,73] + frac[11,73] >= 1
 cov_52_27: frac[8,27] + frac[10,27] >= 0.2
 cov_53_7: frac[10,7] + frac[8,7] >= 0.1
 cov_54_73: frac[10,73] + frac[11,73] >= 0.8
 cov_55_27: frac[8,27] + frac[10,27] >= 0.2
 cov_56_31: frac[10,31] + frac[11,31] >= 1
 cov_57_3: frac[10,3] + frac[8,3] >= 0.2
 cov_58_67: frac[9,67] >= 0.9
 cov_59_56: frac[1,56] + frac[11,56] + frac[10,56] + frac[8,56] >= 0.3
 cov_60_49: frac[10,49] + frac[8,49] >= 0.2
 cov_61_41: frac[11,41] + frac[9,41] + frac[10,41] + frac[8,41] >= 0.7
 cov_62_6: frac[11,6] + frac[10,6] >= 0.9
 cov_63_58: frac[11,58] + frac[10,58] + frac[9,58] >= 0.9
 cov_64_67: frac[9,67] + frac[11,67] + frac[10,67] + frac[8,67] >= 0.8
 cov_65_67: frac[9,67] + frac[11,67] + frac[10,67] + frac[2,67]
   + frac[8,67] >= 0.9
 cov_66_45: frac[10,45] + frac[11,45] >= 0.0329036666666662
 cov_67_23: frac[9,23] >= 0.0329036666666662
 cov_68_67: frac[10,67] + frac[8,67] >= 0.9
 cov_69_73: frac[10,73] + frac[8,73] + frac[11,73] + frac[9,73] >= 0.7
 cov_70_62: frac[9,62] >= 0.0329036666666662
 cov_71_41: frac[11,41] + frac[10,41] + frac[8,41] >= 0.7
 cov_72_7: frac[9,7] >= 0.1
 cov_73_73: frac[10,73] + frac[8,73] + frac[11,73] >= 0.8
 cov_74_73: frac[10,73] + frac[8,73] >= 1
 cov_75_52: frac[11,52] + frac[10,52] >= 1
 cov_76_45: frac[4,45] + frac[9,45] >= 0.0329036666666662
 cov_77_16: frac[1,16] + frac[0,16] + frac[8,16] >= 0.8
 cov_78_19: frac[9,19] + frac[10,19] + frac[11,19] + frac[8,19]
   + frac[2,19] + frac[5,19] >= 0.8
 cov_79_69: frac[8,69] + frac[10,69] + frac[2,69] + frac[11,69]
   + frac[7,69] >= 0.9
 cov_80_36: frac[0,36] + frac[3,36] + frac[8,36] >= 1
 cov_81_32: frac[8,32] + frac[1,32] + frac[0,32] >= 0.9
 cov_82_43: frac[5,43] + frac[9,43] + frac[7,43] >= 0.8
 cov_83_54: frac[6,54] + frac[2,54] + frac[9,54] + frac[11,54]
   + frac[10,54] + frac[8,54] >= 0.9
 cov_84_71: frac[8,71] + frac[0,71] + frac[1,71] >= 0.9
 cov_85_21: frac[6,21] + frac[9,21] + frac[4,21] >= 0.9
 cov_86_65: frac[1,65] + frac[8,65] + frac[0,65] >= 1
 cov_87_69: frac[8,69] + frac[4,69] + frac[0,69] + frac[10,69]
   + frac[11,69] + frac[9,69] >= 0.7
 cov_88_71: frac[2,71] + frac[8,71] + frac[0,71] >= 0.8
 cov_89_36: frac[10,36] + frac[3,36] + frac[8,36] >= 0.9
 cov_90_25: frac[8,25] + frac[11,25] + frac[0,25] + frac[5,25] + frac[9,25]
   + frac[10,25] >= 0.9
 cov_91_9: frac[0,9] + frac[8,9] + frac[1,9] >= 1
 cov_92_2: frac[8,2] + frac[10,2] + frac[2,2] + frac[9,2] + frac[4,2]
   + frac[11,2] >= 0.8
 cov_93_61: frac[10,61] + frac[11,61] + frac[3,61] + frac[9,61]
   + frac[8,61] + frac[4,61] >= 0.8
 cov_94_64: frac[2,64] + frac[8,64] + frac[11,64] + frac[4,64] + frac[9,64]
   + frac[10,64] >= 0.7
 cov_95_32: frac[6,32] + frac[11,32] + frac[10,32] + frac[3,32]
   + frac[9,32] + frac[8,32] >= 0.7
 cov_96_71: frac[6,71] + frac[9,71] + frac[7,71] >= 0.8
 cov_97_21: frac[11,21] + frac[0,21] + frac[9,21] + frac[5,21] + frac[8,21]
   + frac[10,21] >= 0.8
 cov_98_33: frac[2,33] + frac[8,33] + frac[7,33] + frac[11,33]
   + frac[10,33] >= 0.7
 cov_99_2: frac[8,2] + frac[0,2] + frac[2,2] >= 1
 cov_100_71: frac[2,71] + frac[8,71] + frac[0,71] >= 1
 cov_101_22: frac[11,22] + frac[7,22] + frac[10,22] + frac[8,22]
   + frac[0,22] >= 0.8
 cov_102_18: frac[8,18] + frac[7,18] + frac[10,18] + frac[11,18] >= 0.9
 cov_103_21: frac[6,21] + frac[11,21] + frac[2,21] + frac[9,21]
   + frac[8,21] + frac[10,21] >= 0.8
 cov_104_19: frac[1,19] + frac[8,19] + frac[2,19] >= 0.7
 cov_105_47: frac[10,47] + frac[8,47] + frac[11,47] + frac[7,47]
   + frac[3,47] >= 0.9632293333333302
 cov_106_5: frac[4,5] + frac[6,5] + frac[9,5] >= 0.7
 cov_107_51: frac[8,51] + frac[2,51] + frac[7,51] + frac[10,51]
   + frac[11,51] >= 0.7
 cov_108_66: frac[8,66] + frac[0,66] + frac[3,66] >= 0.9
 cov_109_29: frac[9,29] + frac[8,29] + frac[4,29] + frac[10,29]
   + frac[11,29] + frac[0,29] >= 0.9
 cov_110_66: frac[4,66] + frac[8,66] + frac[9,66] + frac[10,66]
   + frac[11,66] + frac[3,66] >= 0.9
 cov_111_16: frac[1,16] + frac[3,16] + frac[8,16] >= 0.7
 cov_112_32: frac[3,32] + frac[8,32] + frac[1,32] >= 1
 cov_113_43: frac[3,43] + frac[0,43] + frac[8,43] >= 0.8
 cov_114_0: frac[9,0] + frac[8,0] + frac[0,0] + frac[4,0] + frac[11,0]
   + frac[10,0] >= 0.7
 cov_115_21: frac[1,21] + frac[11,21] + frac[7,21] + frac[8,21]
   + frac[10,21] >= 0.9
 cov_116_51: frac[8,51] + frac[2,51] + frac[7,51] + frac[10,51]
   + frac[11,51] >= 0.8
 cov_117_66: frac[8,66] + frac[1,66] + frac[2,66] >= 0.8
 cov_118_5: frac[4,5] + frac[9,5] + frac[5,5] >= 0.9
 cov_119_14: frac[10,14] + frac[8,14] + frac[4,14] + frac[11,14]
   + frac[9,14] + frac[3,14] >= 0.9
 cov_120_42: frac[6,42] + frac[9,42] >= 0.7
 cov_121_25: frac[8,25] + frac[1,25] + frac[11,25] + frac[4,25]
   + frac[9,25] + frac[10,25] >= 0.9
 cov_122_69: frac[8,69] + frac[0,69] + frac[1,69] >= 1
 cov_123_12: frac[8,12] + frac[2,12] + frac[3,12] >= 0.9
 cov_124_65: frac[9,65] + frac[4,65] + frac[5,65] >= 1
 cov_125_30: frac[9,30] + frac[5,30] + frac[4,30] >= 0.9
 cov_126_51: frac[8,51] + frac[1,51] + frac[6,51] + frac[9,51]
   + frac[10,51] + frac[11,51] >= 0.8
 cov_127_66: frac[8,66] + frac[5,66] + frac[0,66] + frac[9,66]
   + frac[10,66] + frac[11,66] >= 0.7
 cov_128_21: frac[11,21] + frac[7,21] + frac[3,21] + frac[8,21]
   + frac[10,21] >= 0.9
 cov_129_18: frac[8,18] + frac[6,18] + frac[10,18] + frac[0,18]
   + frac[9,18] + frac[11,18] >= 0.7
 cov_130_57: frac[4,57] + frac[9,57] + frac[10,57] + frac[8,57]
   + frac[2,57] + frac[11,57] >= 0.9
 cov_131_25: frac[8,25] + frac[6,25] + frac[11,25] + frac[0,25]
   + frac[9,25] + frac[10,25] >= 0.8
 cov_132_5: frac[8,5] + frac[2,5] + frac[1,5] >= 0.7
 cov_133_74: frac[1,74] + frac[2,74] + frac[8,74] >= 0.7
 cov_134_39: frac[9,39] + frac[11,39] + frac[5,39] + frac[3,39]
   + frac[10,39] + frac[8,39] >= 0.8
 cov_135_74: frac[11,74] + frac[9,74] + frac[10,74] + frac[0,74]
   + frac[6,74] + frac[8,74] >= 0.7
 cov_136_19: frac[0,19] + frac[1,19] + frac[8,19] >= 0.8
 cov_137_30: frac[0,30] + frac[11,30] + frac[9,30] + frac[10,30]
   + frac[8,30] >= 0.8
 cov_138_34: frac[1,34] + frac[3,34] + frac[8,34] >= 1
 cov_139_74: frac[5,74] + frac[11,74] + frac[9,74] + frac[10,74]
   + frac[3,74] + frac[8,74] >= 0.7
 cov_140_33: frac[8,33] + frac[3,33] + frac[10,33] >= 0.7
 cov_141_25: frac[8,25] + frac[1,25] + frac[2,25] >= 0.9
 cov_142_39: frac[11,39] + frac[7,39] + frac[10,39] + frac[8,39]
   + frac[2,39] >= 0.8
 cov_143_34: frac[2,34] + frac[1,34] + frac[8,34] >= 0.8
 cov_144_2: frac[7,2] + frac[10,2] + frac[11,2] >= 0.9
 cov_145_29: frac[9,29] + frac[8,29] + frac[4,29] + frac[1,29]
   + frac[10,29] + frac[11,29] >= 0.9
 cov_146_5: frac[8,5] + frac[10,5] + frac[11,5] + frac[3,5] + frac[9,5]
   + frac[5,5] >= 0.9
 cov_147_65: frac[2,65] + frac[8,65] + frac[0,65] >= 0.8
 cov_148_32: frac[5,32] + frac[11,32] + frac[10,32] + frac[3,32]
   + frac[9,32] + frac[8,32] >= 0.8
 cov_149_30: frac[11,30] + frac[9,30] + frac[5,30] + frac[10,30]
   + frac[1,30] + frac[8,30] >= 0.8
 cov_150_32: frac[2,32] + frac[8,32] + frac[0,32] >= 0.7
 cov_151_39: frac[9,39] + frac[4,39] + frac[6,39] >= 0.8
 cov_152_65: frac[11,65] + frac[10,65] + frac[2,65] + frac[7,65]
   + frac[8,65] >= 0.9
 cov_153_12: frac[8,12] + frac[10,12] + frac[11,12] + frac[7,12]
   + frac[0,12] >= 0.8
 cols_helper_8_63: cols_helper_8_63[0] + cols_helper_8_63[1]
   + cols_helper_8_63[2] + cols_helper_8_63[3] + cols_helper_8_63[4]
   + cols_helper_8_63[5] + cols_helper_8_63[6] + cols_helper_8_63[7]
   + cols_helper_8_63[8] + cols_helper_8_63[9] + cols_helper_8_63[10]
   + cols_helper_8_63[11] + cols_helper_8_63[12] + cols_helper_8_63[13]
   + cols_helper_8_63[14] + cols_helper_8_63[15] + cols_helper_8_63[16]
   <= 1
 cols_8_63: cols_8_63 - 2 cols_helper_8_63[0] - 4 cols_helper_8_63[1]
   - 8 cols_helper_8_63[2] - 16 cols_helper_8_63[3]
   - 32 cols_helper_8_63[4] - 64 cols_helper_8_63[5]
   - 128 cols_helper_8_63[6] - 256 cols_helper_8_63[7]
   - 512 cols_helper_8_63[8] - 1024 cols_helper_8_63[9]
   - 2048 cols_helper_8_63[10] - 4096 cols_helper_8_63[11]
   - 8192 cols_helper_8_63[12] - 16384 cols_helper_8_63[13]
   - 32768 cols_helper_8_63[14] - 65536 cols_helper_8_63[15]
   - 131072 cols_helper_8_63[16] = 0
 cols_acc_8_63: - 435 frac[8,63] + cols_8_63 >= 0
 accuracy_8_63: mem[8,63] - 0.01171875 cols_8_63 = 0
 cols_helper_9_62: cols_helper_9_62[0] + cols_helper_9_62[1]
   + cols_helper_9_62[2] + cols_helper_9_62[3] + cols_helper_9_62[4]
   + cols_helper_9_62[5] + cols_helper_9_62[6] + cols_helper_9_62[7]
   + cols_helper_9_62[8] + cols_helper_9_62[9] + cols_helper_9_62[10]
   + cols_helper_9_62[11] + cols_helper_9_62[12] + cols_helper_9_62[13]
   + cols_helper_9_62[14] + cols_helper_9_62[15] + cols_helper_9_62[16]
   <= 1
 cols_9_62: cols_9_62 - 2 cols_helper_9_62[0] - 4 cols_helper_9_62[1]
   - 8 cols_helper_9_62[2] - 16 cols_helper_9_62[3]
   - 32 cols_helper_9_62[4] - 64 cols_helper_9_62[5]
   - 128 cols_helper_9_62[6] - 256 cols_helper_9_62[7]
   - 512 cols_helper_9_62[8] - 1024 cols_helper_9_62[9]
   - 2048 cols_helper_9_62[10] - 4096 cols_helper_9_62[11]
   - 8192 cols_helper_9_62[12] - 16384 cols_helper_9_62[13]
   - 32768 cols_helper_9_62[14] - 65536 cols_helper_9_62[15]
   - 131072 cols_helper_9_62[16] = 0
 cols_acc_9_62: - 435 frac[9,62] + cols_9_62 >= 0
 accuracy_9_62: mem[9,62] - 0.01171875 cols_9_62 = 0
 accuracy_6_54: - 5.09765625 frac[6,54] + mem[6,54] = 0
 accuracy_4_66: - 5.09765625 frac[4,66] + mem[4,66] = 0
 cols_helper_10_61: cols_helper_10_61[0] + cols_helper_10_61[1]
   + cols_helper_10_61[2] + cols_helper_10_61[3] + cols_helper_10_61[4]
   + cols_helper_10_61[5] + cols_helper_10_61[6] + cols_helper_10_61[7]
   + cols_helper_10_61[8] + cols_helper_10_61[9] + cols_helper_10_61[10]
   + cols_helper_10_61[11] + cols_helper_10_61[12] + cols_helper_10_61[13]
   + cols_helper_10_61[14] + cols_helper_10_61[15] + cols_helper_10_61[16]
   <= 1
 cols_10_61: cols_10_61 - 2 cols_helper_10_61[0] - 4 cols_helper_10_61[1]
   - 8 cols_helper_10_61[2] - 16 cols_helper_10_61[3]
   - 32 cols_helper_10_61[4] - 64 cols_helper_10_61[5]
   - 128 cols_helper_10_61[6] - 256 cols_helper_10_61[7]
   - 512 cols_helper_10_61[8] - 1024 cols_helper_10_61[9]
   - 2048 cols_helper_10_61[10] - 4096 cols_helper_10_61[11]
   - 8192 cols_helper_10_61[12] - 16384 cols_helper_10_61[13]
   - 32768 cols_helper_10_61[14] - 65536 cols_helper_10_61[15]
   - 131072 cols_helper_10_61[16] = 0
 cols_acc_10_61: - 435 frac[10,61] + cols_10_61 >= 0
 accuracy_10_61: mem[10,61] - 0.01171875 cols_10_61 = 0
 cols_helper_8_5: cols_helper_8_5[0] + cols_helper_8_5[1]
   + cols_helper_8_5[2] + cols_helper_8_5[3] + cols_helper_8_5[4]
   + cols_helper_8_5[5] + cols_helper_8_5[6] + cols_helper_8_5[7]
   + cols_helper_8_5[8] + cols_helper_8_5[9] + cols_helper_8_5[10]
   + cols_helper_8_5[11] + cols_helper_8_5[12] + cols_helper_8_5[13]
   + cols_helper_8_5[14] + cols_helper_8_5[15] + cols_helper_8_5[16] <= 1
 cols_8_5: cols_8_5 - 2 cols_helper_8_5[0] - 4 cols_helper_8_5[1]
   - 8 cols_helper_8_5[2] - 16 cols_helper_8_5[3] - 32 cols_helper_8_5[4]
   - 64 cols_helper_8_5[5] - 128 cols_helper_8_5[6]
   - 256 cols_helper_8_5[7] - 512 cols_helper_8_5[8]
   - 1024 cols_helper_8_5[9] - 2048 cols_helper_8_5[10]
   - 4096 cols_helper_8_5[11] - 8192 cols_helper_8_5[12]
   - 16384 cols_helper_8_5[13] - 32768 cols_helper_8_5[14]
   - 65536 cols_helper_8_5[15] - 131072 cols_helper_8_5[16] = 0
 cols_acc_8_5: - 435 frac[8,5] + cols_8_5 >= 0
 accuracy_8_5: mem[8,5] - 0.01171875 cols_8_5 = 0
 cols_helper_11_48: cols_helper_11_48[0] + cols_helper_11_48[1]
   + cols_helper_11_48[2] + cols_helper_11_48[3] + cols_helper_11_48[4]
   + cols_helper_11_48[5] + cols_helper_11_48[6] + cols_helper_11_48[7]
   + cols_helper_11_48[8] + cols_helper_11_48[9] + cols_helper_11_48[10]
   + cols_helper_11_48[11] + cols_helper_11_48[12] + cols_helper_11_48[13]
   + cols_helper_11_48[14] + cols_helper_11_48[15] + cols_helper_11_48[16]
   <= 1
 cols_11_48: cols_11_48 - 2 cols_helper_11_48[0] - 4 cols_helper_11_48[1]
   - 8 cols_helper_11_48[2] - 16 cols_helper_11_48[3]
   - 32 cols_helper_11_48[4] - 64 cols_helper_11_48[5]
   - 128 cols_helper_11_48[6] - 256 cols_helper_11_48[7]
   - 512 cols_helper_11_48[8] - 1024 cols_helper_11_48[9]
   - 2048 cols_helper_11_48[10] - 4096 cols_helper_11_48[11]
   - 8192 cols_helper_11_48[12] - 16384 cols_helper_11_48[13]
   - 32768 cols_helper_11_48[14] - 65536 cols_helper_11_48[15]
   - 131072 cols_helper_11_48[16] = 0
 cols_acc_11_48: - 435 frac[11,48] + cols_11_48 >= 0
 accuracy_11_48: mem[11,48] - 0.01171875 cols_11_48 = 0
 cols_helper_9_0: cols_helper_9_0[0] + cols_helper_9_0[1]
   + cols_helper_9_0[2] + cols_helper_9_0[3] + cols_helper_9_0[4]
   + cols_helper_9_0[5] + cols_helper_9_0[6] + cols_helper_9_0[7]
   + cols_helper_9_0[8] + cols_helper_9_0[9] + cols_helper_9_0[10]
   + cols_helper_9_0[11] + cols_helper_9_0[12] + cols_helper_9_0[13]
   + cols_helper_9_0[14] + cols_helper_9_0[15] + cols_helper_9_0[16] <= 1
 cols_9_0: cols_9_0 - 2 cols_helper_9_0[0] - 4 cols_helper_9_0[1]
   - 8 cols_helper_9_0[2] - 16 cols_helper_9_0[3] - 32 cols_helper_9_0[4]
   - 64 cols_helper_9_0[5] - 128 cols_helper_9_0[6]
   - 256 cols_helper_9_0[7] - 512 cols_helper_9_0[8]
   - 1024 cols_helper_9_0[9] - 2048 cols_helper_9_0[10]
   - 4096 cols_helper_9_0[11] - 8192 cols_helper_9_0[12]
   - 16384 cols_helper_9_0[13] - 32768 cols_helper_9_0[14]
   - 65536 cols_helper_9_0[15] - 131072 cols_helper_9_0[16] = 0
 cols_acc_9_0: - 435 frac[9,0] + cols_9_0 >= 0
 accuracy_9_0: mem[9,0] - 0.01171875 cols_9_0 = 0
 accuracy_5_74: - 5.09765625 frac[5,74] + mem[5,74] = 0
 cols_helper_10_7: cols_helper_10_7[0] + cols_helper_10_7[1]
   + cols_helper_10_7[2] + cols_helper_10_7[3] + cols_helper_10_7[4]
   + cols_helper_10_7[5] + cols_helper_10_7[6] + cols_helper_10_7[7]
   + cols_helper_10_7[8] + cols_helper_10_7[9] + cols_helper_10_7[10]
   + cols_helper_10_7[11] + cols_helper_10_7[12] + cols_helper_10_7[13]
   + cols_helper_10_7[14] + cols_helper_10_7[15] + cols_helper_10_7[16]
   <= 1
 cols_10_7: cols_10_7 - 2 cols_helper_10_7[0] - 4 cols_helper_10_7[1]
   - 8 cols_helper_10_7[2] - 16 cols_helper_10_7[3]
   - 32 cols_helper_10_7[4] - 64 cols_helper_10_7[5]
   - 128 cols_helper_10_7[6] - 256 cols_helper_10_7[7]
   - 512 cols_helper_10_7[8] - 1024 cols_helper_10_7[9]
   - 2048 cols_helper_10_7[10] - 4096 cols_helper_10_7[11]
   - 8192 cols_helper_10_7[12] - 16384 cols_helper_10_7[13]
   - 32768 cols_helper_10_7[14] - 65536 cols_helper_10_7[15]
   - 131072 cols_helper_10_7[16] = 0
 cols_acc_10_7: - 435 frac[10,7] + cols_10_7 >= 0
 accuracy_10_7: mem[10,7] - 0.01171875 cols_10_7 = 0
 cols_helper_11_22: cols_helper_11_22[0] + cols_helper_11_22[1]
   + cols_helper_11_22[2] + cols_helper_11_22[3] + cols_helper_11_22[4]
   + cols_helper_11_22[5] + cols_helper_11_22[6] + cols_helper_11_22[7]
   + cols_helper_11_22[8] + cols_helper_11_22[9] + cols_helper_11_22[10]
   + cols_helper_11_22[11] + cols_helper_11_22[12] + cols_helper_11_22[13]
   + cols_helper_11_22[14] + cols_helper_11_22[15] + cols_helper_11_22[16]
   <= 1
 cols_11_22: cols_11_22 - 2 cols_helper_11_22[0] - 4 cols_helper_11_22[1]
   - 8 cols_helper_11_22[2] - 16 cols_helper_11_22[3]
   - 32 cols_helper_11_22[4] - 64 cols_helper_11_22[5]
   - 128 cols_helper_11_22[6] - 256 cols_helper_11_22[7]
   - 512 cols_helper_11_22[8] - 1024 cols_helper_11_22[9]
   - 2048 cols_helper_11_22[10] - 4096 cols_helper_11_22[11]
   - 8192 cols_helper_11_22[12] - 16384 cols_helper_11_22[13]
   - 32768 cols_helper_11_22[14] - 65536 cols_helper_11_22[15]
   - 131072 cols_helper_11_22[16] = 0
 cols_acc_11_22: - 435 frac[11,22] + cols_11_22 >= 0
 accuracy_11_22: mem[11,22] - 0.01171875 cols_11_22 = 0
 accuracy_4_5: - 5.09765625 frac[4,5] + mem[4,5] = 0
 accuracy_7_22: - 5.09765625 frac[7,22] + mem[7,22] = 0
 cols_helper_9_39: cols_helper_9_39[0] + cols_helper_9_39[1]
   + cols_helper_9_39[2] + cols_helper_9_39[3] + cols_helper_9_39[4]
   + cols_helper_9_39[5] + cols_helper_9_39[6] + cols_helper_9_39[7]
   + cols_helper_9_39[8] + cols_helper_9_39[9] + cols_helper_9_39[10]
   + cols_helper_9_39[11] + cols_helper_9_39[12] + cols_helper_9_39[13]
   + cols_helper_9_39[14] + cols_helper_9_39[15] + cols_helper_9_39[16]
   <= 1
 cols_9_39: cols_9_39 - 2 cols_helper_9_39[0] - 4 cols_helper_9_39[1]
   - 8 cols_helper_9_39[2] - 16 cols_helper_9_39[3]
   - 32 cols_helper_9_39[4] - 64 cols_helper_9_39[5]
   - 128 cols_helper_9_39[6] - 256 cols_helper_9_39[7]
   - 512 cols_helper_9_39[8] - 1024 cols_helper_9_39[9]
   - 2048 cols_helper_9_39[10] - 4096 cols_helper_9_39[11]
   - 8192 cols_helper_9_39[12] - 16384 cols_helper_9_39[13]
   - 32768 cols_helper_9_39[14] - 65536 cols_helper_9_39[15]
   - 131072 cols_helper_9_39[16] = 0
 cols_acc_9_39: - 435 frac[9,39] + cols_9_39 >= 0
 accuracy_9_39: mem[9,39] - 0.01171875 cols_9_39 = 0
 cols_helper_10_36: cols_helper_10_36[0] + cols_helper_10_36[1]
   + cols_helper_10_36[2] + cols_helper_10_36[3] + cols_helper_10_36[4]
   + cols_helper_10_36[5] + cols_helper_10_36[6] + cols_helper_10_36[7]
   + cols_helper_10_36[8] + cols_helper_10_36[9] + cols_helper_10_36[10]
   + cols_helper_10_36[11] + cols_helper_10_36[12] + cols_helper_10_36[13]
   + cols_helper_10_36[14] + cols_helper_10_36[15] + cols_helper_10_36[16]
   <= 1
 cols_10_36: cols_10_36 - 2 cols_helper_10_36[0] - 4 cols_helper_10_36[1]
   - 8 cols_helper_10_36[2] - 16 cols_helper_10_36[3]
   - 32 cols_helper_10_36[4] - 64 cols_helper_10_36[5]
   - 128 cols_helper_10_36[6] - 256 cols_helper_10_36[7]
   - 512 cols_helper_10_36[8] - 1024 cols_helper_10_36[9]
   - 2048 cols_helper_10_36[10] - 4096 cols_helper_10_36[11]
   - 8192 cols_helper_10_36[12] - 16384 cols_helper_10_36[13]
   - 32768 cols_helper_10_36[14] - 65536 cols_helper_10_36[15]
   - 131072 cols_helper_10_36[16] = 0
 cols_acc_10_36: - 435 frac[10,36] + cols_10_36 >= 0
 accuracy_10_36: mem[10,36] - 0.01171875 cols_10_36 = 0
 cols_helper_8_12: cols_helper_8_12[0] + cols_helper_8_12[1]
   + cols_helper_8_12[2] + cols_helper_8_12[3] + cols_helper_8_12[4]
   + cols_helper_8_12[5] + cols_helper_8_12[6] + cols_helper_8_12[7]
   + cols_helper_8_12[8] + cols_helper_8_12[9] + cols_helper_8_12[10]
   + cols_helper_8_12[11] + cols_helper_8_12[12] + cols_helper_8_12[13]
   + cols_helper_8_12[14] + cols_helper_8_12[15] + cols_helper_8_12[16]
   <= 1
 cols_8_12: cols_8_12 - 2 cols_helper_8_12[0] - 4 cols_helper_8_12[1]
   - 8 cols_helper_8_12[2] - 16 cols_helper_8_12[3]
   - 32 cols_helper_8_12[4] - 64 cols_helper_8_12[5]
   - 128 cols_helper_8_12[6] - 256 cols_helper_8_12[7]
   - 512 cols_helper_8_12[8] - 1024 cols_helper_8_12[9]
   - 2048 cols_helper_8_12[10] - 4096 cols_helper_8_12[11]
   - 8192 cols_helper_8_12[12] - 16384 cols_helper_8_12[13]
   - 32768 cols_helper_8_12[14] - 65536 cols_helper_8_12[15]
   - 131072 cols_helper_8_12[16] = 0
 cols_acc_8_12: - 435 frac[8,12] + cols_8_12 >= 0
 accuracy_8_12: mem[8,12] - 0.01171875 cols_8_12 = 0
 cols_helper_11_41: cols_helper_11_41[0] + cols_helper_11_41[1]
   + cols_helper_11_41[2] + cols_helper_11_41[3] + cols_helper_11_41[4]
   + cols_helper_11_41[5] + cols_helper_11_41[6] + cols_helper_11_41[7]
   + cols_helper_11_41[8] + cols_helper_11_41[9] + cols_helper_11_41[10]
   + cols_helper_11_41[11] + cols_helper_11_41[12] + cols_helper_11_41[13]
   + cols_helper_11_41[14] + cols_helper_11_41[15] + cols_helper_11_41[16]
   <= 1
 cols_11_41: cols_11_41 - 2 cols_helper_11_41[0] - 4 cols_helper_11_41[1]
   - 8 cols_helper_11_41[2] - 16 cols_helper_11_41[3]
   - 32 cols_helper_11_41[4] - 64 cols_helper_11_41[5]
   - 128 cols_helper_11_41[6] - 256 cols_helper_11_41[7]
   - 512 cols_helper_11_41[8] - 1024 cols_helper_11_41[9]
   - 2048 cols_helper_11_41[10] - 4096 cols_helper_11_41[11]
   - 8192 cols_helper_11_41[12] - 16384 cols_helper_11_41[13]
   - 32768 cols_helper_11_41[14] - 65536 cols_helper_11_41[15]
   - 131072 cols_helper_11_41[16] = 0
 cols_acc_11_41: - 435 frac[11,41] + cols_11_41 >= 0
 accuracy_11_41: mem[11,41] - 0.01171875 cols_11_41 = 0
 cols_helper_10_14: cols_helper_10_14[0] + cols_helper_10_14[1]
   + cols_helper_10_14[2] + cols_helper_10_14[3] + cols_helper_10_14[4]
   + cols_helper_10_14[5] + cols_helper_10_14[6] + cols_helper_10_14[7]
   + cols_helper_10_14[8] + cols_helper_10_14[9] + cols_helper_10_14[10]
   + cols_helper_10_14[11] + cols_helper_10_14[12] + cols_helper_10_14[13]
   + cols_helper_10_14[14] + cols_helper_10_14[15] + cols_helper_10_14[16]
   <= 1
 cols_10_14: cols_10_14 - 2 cols_helper_10_14[0] - 4 cols_helper_10_14[1]
   - 8 cols_helper_10_14[2] - 16 cols_helper_10_14[3]
   - 32 cols_helper_10_14[4] - 64 cols_helper_10_14[5]
   - 128 cols_helper_10_14[6] - 256 cols_helper_10_14[7]
   - 512 cols_helper_10_14[8] - 1024 cols_helper_10_14[9]
   - 2048 cols_helper_10_14[10] - 4096 cols_helper_10_14[11]
   - 8192 cols_helper_10_14[12] - 16384 cols_helper_10_14[13]
   - 32768 cols_helper_10_14[14] - 65536 cols_helper_10_14[15]
   - 131072 cols_helper_10_14[16] = 0
 cols_acc_10_14: - 435 frac[10,14] + cols_10_14 >= 0
 accuracy_10_14: mem[10,14] - 0.01171875 cols_10_14 = 0
 cols_helper_8_18: cols_helper_8_18[0] + cols_helper_8_18[1]
   + cols_helper_8_18[2] + cols_helper_8_18[3] + cols_helper_8_18[4]
   + cols_helper_8_18[5] + cols_helper_8_18[6] + cols_helper_8_18[7]
   + cols_helper_8_18[8] + cols_helper_8_18[9] + cols_helper_8_18[10]
   + cols_helper_8_18[11] + cols_helper_8_18[12] + cols_helper_8_18[13]
   + cols_helper_8_18[14] + cols_helper_8_18[15] + cols_helper_8_18[16]
   <= 1
 cols_8_18: cols_8_18 - 2 cols_helper_8_18[0] - 4 cols_helper_8_18[1]
   - 8 cols_helper_8_18[2] - 16 cols_helper_8_18[3]
   - 32 cols_helper_8_18[4] - 64 cols_helper_8_18[5]
   - 128 cols_helper_8_18[6] - 256 cols_helper_8_18[7]
   - 512 cols_helper_8_18[8] - 1024 cols_helper_8_18[9]
   - 2048 cols_helper_8_18[10] - 4096 cols_helper_8_18[11]
   - 8192 cols_helper_8_18[12] - 16384 cols_helper_8_18[13]
   - 32768 cols_helper_8_18[14] - 65536 cols_helper_8_18[15]
   - 131072 cols_helper_8_18[16] = 0
 cols_acc_8_18: - 435 frac[8,18] + cols_8_18 >= 0
 accuracy_8_18: mem[8,18] - 0.01171875 cols_8_18 = 0
 cols_helper_9_19: cols_helper_9_19[0] + cols_helper_9_19[1]
   + cols_helper_9_19[2] + cols_helper_9_19[3] + cols_helper_9_19[4]
   + cols_helper_9_19[5] + cols_helper_9_19[6] + cols_helper_9_19[7]
   + cols_helper_9_19[8] + cols_helper_9_19[9] + cols_helper_9_19[10]
   + cols_helper_9_19[11] + cols_helper_9_19[12] + cols_helper_9_19[13]
   + cols_helper_9_19[14] + cols_helper_9_19[15] + cols_helper_9_19[16]
   <= 1
 cols_9_19: cols_9_19 - 2 cols_helper_9_19[0] - 4 cols_helper_9_19[1]
   - 8 cols_helper_9_19[2] - 16 cols_helper_9_19[3]
   - 32 cols_helper_9_19[4] - 64 cols_helper_9_19[5]
   - 128 cols_helper_9_19[6] - 256 cols_helper_9_19[7]
   - 512 cols_helper_9_19[8] - 1024 cols_helper_9_19[9]
   - 2048 cols_helper_9_19[10] - 4096 cols_helper_9_19[11]
   - 8192 cols_helper_9_19[12] - 16384 cols_helper_9_19[13]
   - 32768 cols_helper_9_19[14] - 65536 cols_helper_9_19[15]
   - 131072 cols_helper_9_19[16] = 0
 cols_acc_9_19: - 435 frac[9,19] + cols_9_19 >= 0
 accuracy_9_19: mem[9,19] - 0.01171875 cols_9_19 = 0
 accuracy_1_21: - 5.09765625 frac[1,21] + mem[1,21] = 0
 accuracy_2_12: - 5.09765625 frac[2,12] + mem[2,12] = 0
 accuracy_0_36: - 5.09765625 frac[0,36] + mem[0,36] = 0
 accuracy_5_43: - 5.09765625 frac[5,43] + mem[5,43] = 0
 cols_helper_11_72: cols_helper_11_72[0] + cols_helper_11_72[1]
   + cols_helper_11_72[2] + cols_helper_11_72[3] + cols_helper_11_72[4]
   + cols_helper_11_72[5] + cols_helper_11_72[6] + cols_helper_11_72[7]
   + cols_helper_11_72[8] + cols_helper_11_72[9] + cols_helper_11_72[10]
   + cols_helper_11_72[11] + cols_helper_11_72[12] + cols_helper_11_72[13]
   + cols_helper_11_72[14] + cols_helper_11_72[15] + cols_helper_11_72[16]
   <= 1
 cols_11_72: cols_11_72 - 2 cols_helper_11_72[0] - 4 cols_helper_11_72[1]
   - 8 cols_helper_11_72[2] - 16 cols_helper_11_72[3]
   - 32 cols_helper_11_72[4] - 64 cols_helper_11_72[5]
   - 128 cols_helper_11_72[6] - 256 cols_helper_11_72[7]
   - 512 cols_helper_11_72[8] - 1024 cols_helper_11_72[9]
   - 2048 cols_helper_11_72[10] - 4096 cols_helper_11_72[11]
   - 8192 cols_helper_11_72[12] - 16384 cols_helper_11_72[13]
   - 32768 cols_helper_11_72[14] - 65536 cols_helper_11_72[15]
   - 131072 cols_helper_11_72[16] = 0
 cols_acc_11_72: - 435 frac[11,72] + cols_11_72 >= 0
 accuracy_11_72: mem[11,72] - 0.01171875 cols_11_72 = 0
 accuracy_6_32: - 5.09765625 frac[6,32] + mem[6,32] = 0
 cols_helper_10_47: cols_helper_10_47[0] + cols_helper_10_47[1]
   + cols_helper_10_47[2] + cols_helper_10_47[3] + cols_helper_10_47[4]
   + cols_helper_10_47[5] + cols_helper_10_47[6] + cols_helper_10_47[7]
   + cols_helper_10_47[8] + cols_helper_10_47[9] + cols_helper_10_47[10]
   + cols_helper_10_47[11] + cols_helper_10_47[12] + cols_helper_10_47[13]
   + cols_helper_10_47[14] + cols_helper_10_47[15] + cols_helper_10_47[16]
   <= 1
 cols_10_47: cols_10_47 - 2 cols_helper_10_47[0] - 4 cols_helper_10_47[1]
   - 8 cols_helper_10_47[2] - 16 cols_helper_10_47[3]
   - 32 cols_helper_10_47[4] - 64 cols_helper_10_47[5]
   - 128 cols_helper_10_47[6] - 256 cols_helper_10_47[7]
   - 512 cols_helper_10_47[8] - 1024 cols_helper_10_47[9]
   - 2048 cols_helper_10_47[10] - 4096 cols_helper_10_47[11]
   - 8192 cols_helper_10_47[12] - 16384 cols_helper_10_47[13]
   - 32768 cols_helper_10_47[14] - 65536 cols_helper_10_47[15]
   - 131072 cols_helper_10_47[16] = 0
 cols_acc_10_47: - 435 frac[10,47] + cols_10_47 >= 0
 accuracy_10_47: mem[10,47] - 0.01171875 cols_10_47 = 0
 accuracy_2_64: - 5.09765625 frac[2,64] + mem[2,64] = 0
 cols_helper_8_51: cols_helper_8_51[0] + cols_helper_8_51[1]
   + cols_helper_8_51[2] + cols_helper_8_51[3] + cols_helper_8_51[4]
   + cols_helper_8_51[5] + cols_helper_8_51[6] + cols_helper_8_51[7]
   + cols_helper_8_51[8] + cols_helper_8_51[9] + cols_helper_8_51[10]
   + cols_helper_8_51[11] + cols_helper_8_51[12] + cols_helper_8_51[13]
   + cols_helper_8_51[14] + cols_helper_8_51[15] + cols_helper_8_51[16]
   <= 1
 cols_8_51: cols_8_51 - 2 cols_helper_8_51[0] - 4 cols_helper_8_51[1]
   - 8 cols_helper_8_51[2] - 16 cols_helper_8_51[3]
   - 32 cols_helper_8_51[4] - 64 cols_helper_8_51[5]
   - 128 cols_helper_8_51[6] - 256 cols_helper_8_51[7]
   - 512 cols_helper_8_51[8] - 1024 cols_helper_8_51[9]
   - 2048 cols_helper_8_51[10] - 4096 cols_helper_8_51[11]
   - 8192 cols_helper_8_51[12] - 16384 cols_helper_8_51[13]
   - 32768 cols_helper_8_51[14] - 65536 cols_helper_8_51[15]
   - 131072 cols_helper_8_51[16] = 0
 cols_acc_8_51: - 435 frac[8,51] + cols_8_51 >= 0
 accuracy_8_51: mem[8,51] - 0.01171875 cols_8_51 = 0
 cols_helper_11_46: cols_helper_11_46[0] + cols_helper_11_46[1]
   + cols_helper_11_46[2] + cols_helper_11_46[3] + cols_helper_11_46[4]
   + cols_helper_11_46[5] + cols_helper_11_46[6] + cols_helper_11_46[7]
   + cols_helper_11_46[8] + cols_helper_11_46[9] + cols_helper_11_46[10]
   + cols_helper_11_46[11] + cols_helper_11_46[12] + cols_helper_11_46[13]
   + cols_helper_11_46[14] + cols_helper_11_46[15] + cols_helper_11_46[16]
   <= 1
 cols_11_46: cols_11_46 - 2 cols_helper_11_46[0] - 4 cols_helper_11_46[1]
   - 8 cols_helper_11_46[2] - 16 cols_helper_11_46[3]
   - 32 cols_helper_11_46[4] - 64 cols_helper_11_46[5]
   - 128 cols_helper_11_46[6] - 256 cols_helper_11_46[7]
   - 512 cols_helper_11_46[8] - 1024 cols_helper_11_46[9]
   - 2048 cols_helper_11_46[10] - 4096 cols_helper_11_46[11]
   - 8192 cols_helper_11_46[12] - 16384 cols_helper_11_46[13]
   - 32768 cols_helper_11_46[14] - 65536 cols_helper_11_46[15]
   - 131072 cols_helper_11_46[16] = 0
 cols_acc_11_46: - 435 frac[11,46] + cols_11_46 >= 0
 accuracy_11_46: mem[11,46] - 0.01171875 cols_11_46 = 0
 cols_helper_9_50: cols_helper_9_50[0] + cols_helper_9_50[1]
   + cols_helper_9_50[2] + cols_helper_9_50[3] + cols_helper_9_50[4]
   + cols_helper_9_50[5] + cols_helper_9_50[6] + cols_helper_9_50[7]
   + cols_helper_9_50[8] + cols_helper_9_50[9] + cols_helper_9_50[10]
   + cols_helper_9_50[11] + cols_helper_9_50[12] + cols_helper_9_50[13]
   + cols_helper_9_50[14] + cols_helper_9_50[15] + cols_helper_9_50[16]
   <= 1
 cols_9_50: cols_9_50 - 2 cols_helper_9_50[0] - 4 cols_helper_9_50[1]
   - 8 cols_helper_9_50[2] - 16 cols_helper_9_50[3]
   - 32 cols_helper_9_50[4] - 64 cols_helper_9_50[5]
   - 128 cols_helper_9_50[6] - 256 cols_helper_9_50[7]
   - 512 cols_helper_9_50[8] - 1024 cols_helper_9_50[9]
   - 2048 cols_helper_9_50[10] - 4096 cols_helper_9_50[11]
   - 8192 cols_helper_9_50[12] - 16384 cols_helper_9_50[13]
   - 32768 cols_helper_9_50[14] - 65536 cols_helper_9_50[15]
   - 131072 cols_helper_9_50[16] = 0
 cols_acc_9_50: - 435 frac[9,50] + cols_9_50 >= 0
 accuracy_9_50: mem[9,50] - 0.01171875 cols_9_50 = 0
 cols_helper_8_25: cols_helper_8_25[0] + cols_helper_8_25[1]
   + cols_helper_8_25[2] + cols_helper_8_25[3] + cols_helper_8_25[4]
   + cols_helper_8_25[5] + cols_helper_8_25[6] + cols_helper_8_25[7]
   + cols_helper_8_25[8] + cols_helper_8_25[9] + cols_helper_8_25[10]
   + cols_helper_8_25[11] + cols_helper_8_25[12] + cols_helper_8_25[13]
   + cols_helper_8_25[14] + cols_helper_8_25[15] + cols_helper_8_25[16]
   <= 1
 cols_8_25: cols_8_25 - 2 cols_helper_8_25[0] - 4 cols_helper_8_25[1]
   - 8 cols_helper_8_25[2] - 16 cols_helper_8_25[3]
   - 32 cols_helper_8_25[4] - 64 cols_helper_8_25[5]
   - 128 cols_helper_8_25[6] - 256 cols_helper_8_25[7]
   - 512 cols_helper_8_25[8] - 1024 cols_helper_8_25[9]
   - 2048 cols_helper_8_25[10] - 4096 cols_helper_8_25[11]
   - 8192 cols_helper_8_25[12] - 16384 cols_helper_8_25[13]
   - 32768 cols_helper_8_25[14] - 65536 cols_helper_8_25[15]
   - 131072 cols_helper_8_25[16] = 0
 cols_acc_8_25: - 435 frac[8,25] + cols_8_25 >= 0
 accuracy_8_25: mem[8,25] - 0.01171875 cols_8_25 = 0
 cols_helper_9_20: cols_helper_9_20[0] + cols_helper_9_20[1]
   + cols_helper_9_20[2] + cols_helper_9_20[3] + cols_helper_9_20[4]
   + cols_helper_9_20[5] + cols_helper_9_20[6] + cols_helper_9_20[7]
   + cols_helper_9_20[8] + cols_helper_9_20[9] + cols_helper_9_20[10]
   + cols_helper_9_20[11] + cols_helper_9_20[12] + cols_helper_9_20[13]
   + cols_helper_9_20[14] + cols_helper_9_20[15] + cols_helper_9_20[16]
   <= 1
 cols_9_20: cols_9_20 - 2 cols_helper_9_20[0] - 4 cols_helper_9_20[1]
   - 8 cols_helper_9_20[2] - 16 cols_helper_9_20[3]
   - 32 cols_helper_9_20[4] - 64 cols_helper_9_20[5]
   - 128 cols_helper_9_20[6] - 256 cols_helper_9_20[7]
   - 512 cols_helper_9_20[8] - 1024 cols_helper_9_20[9]
   - 2048 cols_helper_9_20[10] - 4096 cols_helper_9_20[11]
   - 8192 cols_helper_9_20[12] - 16384 cols_helper_9_20[13]
   - 32768 cols_helper_9_20[14] - 65536 cols_helper_9_20[15]
   - 131072 cols_helper_9_20[16] = 0
 cols_acc_9_20: - 435 frac[9,20] + cols_9_20 >= 0
 accuracy_9_20: mem[9,20] - 0.01171875 cols_9_20 = 0
 cols_helper_10_19: cols_helper_10_19[0] + cols_helper_10_19[1]
   + cols_helper_10_19[2] + cols_helper_10_19[3] + cols_helper_10_19[4]
   + cols_helper_10_19[5] + cols_helper_10_19[6] + cols_helper_10_19[7]
   + cols_helper_10_19[8] + cols_helper_10_19[9] + cols_helper_10_19[10]
   + cols_helper_10_19[11] + cols_helper_10_19[12] + cols_helper_10_19[13]
   + cols_helper_10_19[14] + cols_helper_10_19[15] + cols_helper_10_19[16]
   <= 1
 cols_10_19: cols_10_19 - 2 cols_helper_10_19[0] - 4 cols_helper_10_19[1]
   - 8 cols_helper_10_19[2] - 16 cols_helper_10_19[3]
   - 32 cols_helper_10_19[4] - 64 cols_helper_10_19[5]
   - 128 cols_helper_10_19[6] - 256 cols_helper_10_19[7]
   - 512 cols_helper_10_19[8] - 1024 cols_helper_10_19[9]
   - 2048 cols_helper_10_19[10] - 4096 cols_helper_10_19[11]
   - 8192 cols_helper_10_19[12] - 16384 cols_helper_10_19[13]
   - 32768 cols_helper_10_19[14] - 65536 cols_helper_10_19[15]
   - 131072 cols_helper_10_19[16] = 0
 cols_acc_10_19: - 435 frac[10,19] + cols_10_19 >= 0
 accuracy_10_19: mem[10,19] - 0.01171875 cols_10_19 = 0
 cols_helper_11_26: cols_helper_11_26[0] + cols_helper_11_26[1]
   + cols_helper_11_26[2] + cols_helper_11_26[3] + cols_helper_11_26[4]
   + cols_helper_11_26[5] + cols_helper_11_26[6] + cols_helper_11_26[7]
   + cols_helper_11_26[8] + cols_helper_11_26[9] + cols_helper_11_26[10]
   + cols_helper_11_26[11] + cols_helper_11_26[12] + cols_helper_11_26[13]
   + cols_helper_11_26[14] + cols_helper_11_26[15] + cols_helper_11_26[16]
   <= 1
 cols_11_26: cols_11_26 - 2 cols_helper_11_26[0] - 4 cols_helper_11_26[1]
   - 8 cols_helper_11_26[2] - 16 cols_helper_11_26[3]
   - 32 cols_helper_11_26[4] - 64 cols_helper_11_26[5]
   - 128 cols_helper_11_26[6] - 256 cols_helper_11_26[7]
   - 512 cols_helper_11_26[8] - 1024 cols_helper_11_26[9]
   - 2048 cols_helper_11_26[10] - 4096 cols_helper_11_26[11]
   - 8192 cols_helper_11_26[12] - 16384 cols_helper_11_26[13]
   - 32768 cols_helper_11_26[14] - 65536 cols_helper_11_26[15]
   - 131072 cols_helper_11_26[16] = 0
 cols_acc_11_26: - 435 frac[11,26] + cols_11_26 >= 0
 accuracy_11_26: mem[11,26] - 0.01171875 cols_11_26 = 0
 cols_helper_8_69: cols_helper_8_69[0] + cols_helper_8_69[1]
   + cols_helper_8_69[2] + cols_helper_8_69[3] + cols_helper_8_69[4]
   + cols_helper_8_69[5] + cols_helper_8_69[6] + cols_helper_8_69[7]
   + cols_helper_8_69[8] + cols_helper_8_69[9] + cols_helper_8_69[10]
   + cols_helper_8_69[11] + cols_helper_8_69[12] + cols_helper_8_69[13]
   + cols_helper_8_69[14] + cols_helper_8_69[15] + cols_helper_8_69[16]
   <= 1
 cols_8_69: cols_8_69 - 2 cols_helper_8_69[0] - 4 cols_helper_8_69[1]
   - 8 cols_helper_8_69[2] - 16 cols_helper_8_69[3]
   - 32 cols_helper_8_69[4] - 64 cols_helper_8_69[5]
   - 128 cols_helper_8_69[6] - 256 cols_helper_8_69[7]
   - 512 cols_helper_8_69[8] - 1024 cols_helper_8_69[9]
   - 2048 cols_helper_8_69[10] - 4096 cols_helper_8_69[11]
   - 8192 cols_helper_8_69[12] - 16384 cols_helper_8_69[13]
   - 32768 cols_helper_8_69[14] - 65536 cols_helper_8_69[15]
   - 131072 cols_helper_8_69[16] = 0
 cols_acc_8_69: - 435 frac[8,69] + cols_8_69 >= 0
 accuracy_8_69: mem[8,69] - 0.01171875 cols_8_69 = 0
 accuracy_2_33: - 5.09765625 frac[2,33] + mem[2,33] = 0
 cols_helper_11_65: cols_helper_11_65[0] + cols_helper_11_65[1]
   + cols_helper_11_65[2] + cols_helper_11_65[3] + cols_helper_11_65[4]
   + cols_helper_11_65[5] + cols_helper_11_65[6] + cols_helper_11_65[7]
   + cols_helper_11_65[8] + cols_helper_11_65[9] + cols_helper_11_65[10]
   + cols_helper_11_65[11] + cols_helper_11_65[12] + cols_helper_11_65[13]
   + cols_helper_11_65[14] + cols_helper_11_65[15] + cols_helper_11_65[16]
   <= 1
 cols_11_65: cols_11_65 - 2 cols_helper_11_65[0] - 4 cols_helper_11_65[1]
   - 8 cols_helper_11_65[2] - 16 cols_helper_11_65[3]
   - 32 cols_helper_11_65[4] - 64 cols_helper_11_65[5]
   - 128 cols_helper_11_65[6] - 256 cols_helper_11_65[7]
   - 512 cols_helper_11_65[8] - 1024 cols_helper_11_65[9]
   - 2048 cols_helper_11_65[10] - 4096 cols_helper_11_65[11]
   - 8192 cols_helper_11_65[12] - 16384 cols_helper_11_65[13]
   - 32768 cols_helper_11_65[14] - 65536 cols_helper_11_65[15]
   - 131072 cols_helper_11_65[16] = 0
 cols_acc_11_65: - 435 frac[11,65] + cols_11_65 >= 0
 accuracy_11_65: mem[11,65] - 0.01171875 cols_11_65 = 0
 cols_helper_11_39: cols_helper_11_39[0] + cols_helper_11_39[1]
   + cols_helper_11_39[2] + cols_helper_11_39[3] + cols_helper_11_39[4]
   + cols_helper_11_39[5] + cols_helper_11_39[6] + cols_helper_11_39[7]
   + cols_helper_11_39[8] + cols_helper_11_39[9] + cols_helper_11_39[10]
   + cols_helper_11_39[11] + cols_helper_11_39[12] + cols_helper_11_39[13]
   + cols_helper_11_39[14] + cols_helper_11_39[15] + cols_helper_11_39[16]
   <= 1
 cols_11_39: cols_11_39 - 2 cols_helper_11_39[0] - 4 cols_helper_11_39[1]
   - 8 cols_helper_11_39[2] - 16 cols_helper_11_39[3]
   - 32 cols_helper_11_39[4] - 64 cols_helper_11_39[5]
   - 128 cols_helper_11_39[6] - 256 cols_helper_11_39[7]
   - 512 cols_helper_11_39[8] - 1024 cols_helper_11_39[9]
   - 2048 cols_helper_11_39[10] - 4096 cols_helper_11_39[11]
   - 8192 cols_helper_11_39[12] - 16384 cols_helper_11_39[13]
   - 32768 cols_helper_11_39[14] - 65536 cols_helper_11_39[15]
   - 131072 cols_helper_11_39[16] = 0
 cols_acc_11_39: - 435 frac[11,39] + cols_11_39 >= 0
 accuracy_11_39: mem[11,39] - 0.01171875 cols_11_39 = 0
 cols_helper_9_59: cols_helper_9_59[0] + cols_helper_9_59[1]
   + cols_helper_9_59[2] + cols_helper_9_59[3] + cols_helper_9_59[4]
   + cols_helper_9_59[5] + cols_helper_9_59[6] + cols_helper_9_59[7]
   + cols_helper_9_59[8] + cols_helper_9_59[9] + cols_helper_9_59[10]
   + cols_helper_9_59[11] + cols_helper_9_59[12] + cols_helper_9_59[13]
   + cols_helper_9_59[14] + cols_helper_9_59[15] + cols_helper_9_59[16]
   <= 1
 cols_9_59: cols_9_59 - 2 cols_helper_9_59[0] - 4 cols_helper_9_59[1]
   - 8 cols_helper_9_59[2] - 16 cols_helper_9_59[3]
   - 32 cols_helper_9_59[4] - 64 cols_helper_9_59[5]
   - 128 cols_helper_9_59[6] - 256 cols_helper_9_59[7]
   - 512 cols_helper_9_59[8] - 1024 cols_helper_9_59[9]
   - 2048 cols_helper_9_59[10] - 4096 cols_helper_9_59[11]
   - 8192 cols_helper_9_59[12] - 16384 cols_helper_9_59[13]
   - 32768 cols_helper_9_59[14] - 65536 cols_helper_9_59[15]
   - 131072 cols_helper_9_59[16] = 0
 cols_acc_9_59: - 435 frac[9,59] + cols_9_59 >= 0
 accuracy_9_59: mem[9,59] - 0.01171875 cols_9_59 = 0
 accuracy_4_69: - 5.09765625 frac[4,69] + mem[4,69] = 0
 cols_helper_10_48: cols_helper_10_48[0] + cols_helper_10_48[1]
   + cols_helper_10_48[2] + cols_helper_10_48[3] + cols_helper_10_48[4]
   + cols_helper_10_48[5] + cols_helper_10_48[6] + cols_helper_10_48[7]
   + cols_helper_10_48[8] + cols_helper_10_48[9] + cols_helper_10_48[10]
   + cols_helper_10_48[11] + cols_helper_10_48[12] + cols_helper_10_48[13]
   + cols_helper_10_48[14] + cols_helper_10_48[15] + cols_helper_10_48[16]
   <= 1
 cols_10_48: cols_10_48 - 2 cols_helper_10_48[0] - 4 cols_helper_10_48[1]
   - 8 cols_helper_10_48[2] - 16 cols_helper_10_48[3]
   - 32 cols_helper_10_48[4] - 64 cols_helper_10_48[5]
   - 128 cols_helper_10_48[6] - 256 cols_helper_10_48[7]
   - 512 cols_helper_10_48[8] - 1024 cols_helper_10_48[9]
   - 2048 cols_helper_10_48[10] - 4096 cols_helper_10_48[11]
   - 8192 cols_helper_10_48[12] - 16384 cols_helper_10_48[13]
   - 32768 cols_helper_10_48[14] - 65536 cols_helper_10_48[15]
   - 131072 cols_helper_10_48[16] = 0
 cols_acc_10_48: - 435 frac[10,48] + cols_10_48 >= 0
 accuracy_10_48: mem[10,48] - 0.01171875 cols_10_48 = 0
 cols_helper_8_0: cols_helper_8_0[0] + cols_helper_8_0[1]
   + cols_helper_8_0[2] + cols_helper_8_0[3] + cols_helper_8_0[4]
   + cols_helper_8_0[5] + cols_helper_8_0[6] + cols_helper_8_0[7]
   + cols_helper_8_0[8] + cols_helper_8_0[9] + cols_helper_8_0[10]
   + cols_helper_8_0[11] + cols_helper_8_0[12] + cols_helper_8_0[13]
   + cols_helper_8_0[14] + cols_helper_8_0[15] + cols_helper_8_0[16] <= 1
 cols_8_0: cols_8_0 - 2 cols_helper_8_0[0] - 4 cols_helper_8_0[1]
   - 8 cols_helper_8_0[2] - 16 cols_helper_8_0[3] - 32 cols_helper_8_0[4]
   - 64 cols_helper_8_0[5] - 128 cols_helper_8_0[6]
   - 256 cols_helper_8_0[7] - 512 cols_helper_8_0[8]
   - 1024 cols_helper_8_0[9] - 2048 cols_helper_8_0[10]
   - 4096 cols_helper_8_0[11] - 8192 cols_helper_8_0[12]
   - 16384 cols_helper_8_0[13] - 32768 cols_helper_8_0[14]
   - 65536 cols_helper_8_0[15] - 131072 cols_helper_8_0[16] = 0
 cols_acc_8_0: - 435 frac[8,0] + cols_8_0 >= 0
 accuracy_8_0: mem[8,0] - 0.01171875 cols_8_0 = 0
 cols_helper_11_61: cols_helper_11_61[0] + cols_helper_11_61[1]
   + cols_helper_11_61[2] + cols_helper_11_61[3] + cols_helper_11_61[4]
   + cols_helper_11_61[5] + cols_helper_11_61[6] + cols_helper_11_61[7]
   + cols_helper_11_61[8] + cols_helper_11_61[9] + cols_helper_11_61[10]
   + cols_helper_11_61[11] + cols_helper_11_61[12] + cols_helper_11_61[13]
   + cols_helper_11_61[14] + cols_helper_11_61[15] + cols_helper_11_61[16]
   <= 1
 cols_11_61: cols_11_61 - 2 cols_helper_11_61[0] - 4 cols_helper_11_61[1]
   - 8 cols_helper_11_61[2] - 16 cols_helper_11_61[3]
   - 32 cols_helper_11_61[4] - 64 cols_helper_11_61[5]
   - 128 cols_helper_11_61[6] - 256 cols_helper_11_61[7]
   - 512 cols_helper_11_61[8] - 1024 cols_helper_11_61[9]
   - 2048 cols_helper_11_61[10] - 4096 cols_helper_11_61[11]
   - 8192 cols_helper_11_61[12] - 16384 cols_helper_11_61[13]
   - 32768 cols_helper_11_61[14] - 65536 cols_helper_11_61[15]
   - 131072 cols_helper_11_61[16] = 0
 cols_acc_11_61: - 435 frac[11,61] + cols_11_61 >= 0
 accuracy_11_61: mem[11,61] - 0.01171875 cols_11_61 = 0
 cols_helper_9_29: cols_helper_9_29[0] + cols_helper_9_29[1]
   + cols_helper_9_29[2] + cols_helper_9_29[3] + cols_helper_9_29[4]
   + cols_helper_9_29[5] + cols_helper_9_29[6] + cols_helper_9_29[7]
   + cols_helper_9_29[8] + cols_helper_9_29[9] + cols_helper_9_29[10]
   + cols_helper_9_29[11] + cols_helper_9_29[12] + cols_helper_9_29[13]
   + cols_helper_9_29[14] + cols_helper_9_29[15] + cols_helper_9_29[16]
   <= 1
 cols_9_29: cols_9_29 - 2 cols_helper_9_29[0] - 4 cols_helper_9_29[1]
   - 8 cols_helper_9_29[2] - 16 cols_helper_9_29[3]
   - 32 cols_helper_9_29[4] - 64 cols_helper_9_29[5]
   - 128 cols_helper_9_29[6] - 256 cols_helper_9_29[7]
   - 512 cols_helper_9_29[8] - 1024 cols_helper_9_29[9]
   - 2048 cols_helper_9_29[10] - 4096 cols_helper_9_29[11]
   - 8192 cols_helper_9_29[12] - 16384 cols_helper_9_29[13]
   - 32768 cols_helper_9_29[14] - 65536 cols_helper_9_29[15]
   - 131072 cols_helper_9_29[16] = 0
 cols_acc_9_29: - 435 frac[9,29] + cols_9_29 >= 0
 accuracy_9_29: mem[9,29] - 0.01171875 cols_9_29 = 0
 cols_helper_10_26: cols_helper_10_26[0] + cols_helper_10_26[1]
   + cols_helper_10_26[2] + cols_helper_10_26[3] + cols_helper_10_26[4]
   + cols_helper_10_26[5] + cols_helper_10_26[6] + cols_helper_10_26[7]
   + cols_helper_10_26[8] + cols_helper_10_26[9] + cols_helper_10_26[10]
   + cols_helper_10_26[11] + cols_helper_10_26[12] + cols_helper_10_26[13]
   + cols_helper_10_26[14] + cols_helper_10_26[15] + cols_helper_10_26[16]
   <= 1
 cols_10_26: cols_10_26 - 2 cols_helper_10_26[0] - 4 cols_helper_10_26[1]
   - 8 cols_helper_10_26[2] - 16 cols_helper_10_26[3]
   - 32 cols_helper_10_26[4] - 64 cols_helper_10_26[5]
   - 128 cols_helper_10_26[6] - 256 cols_helper_10_26[7]
   - 512 cols_helper_10_26[8] - 1024 cols_helper_10_26[9]
   - 2048 cols_helper_10_26[10] - 4096 cols_helper_10_26[11]
   - 8192 cols_helper_10_26[12] - 16384 cols_helper_10_26[13]
   - 32768 cols_helper_10_26[14] - 65536 cols_helper_10_26[15]
   - 131072 cols_helper_10_26[16] = 0
 cols_acc_10_26: - 435 frac[10,26] + cols_10_26 >= 0
 accuracy_10_26: mem[10,26] - 0.01171875 cols_10_26 = 0
 cols_helper_11_19: cols_helper_11_19[0] + cols_helper_11_19[1]
   + cols_helper_11_19[2] + cols_helper_11_19[3] + cols_helper_11_19[4]
   + cols_helper_11_19[5] + cols_helper_11_19[6] + cols_helper_11_19[7]
   + cols_helper_11_19[8] + cols_helper_11_19[9] + cols_helper_11_19[10]
   + cols_helper_11_19[11] + cols_helper_11_19[12] + cols_helper_11_19[13]
   + cols_helper_11_19[14] + cols_helper_11_19[15] + cols_helper_11_19[16]
   <= 1
 cols_11_19: cols_11_19 - 2 cols_helper_11_19[0] - 4 cols_helper_11_19[1]
   - 8 cols_helper_11_19[2] - 16 cols_helper_11_19[3]
   - 32 cols_helper_11_19[4] - 64 cols_helper_11_19[5]
   - 128 cols_helper_11_19[6] - 256 cols_helper_11_19[7]
   - 512 cols_helper_11_19[8] - 1024 cols_helper_11_19[9]
   - 2048 cols_helper_11_19[10] - 4096 cols_helper_11_19[11]
   - 8192 cols_helper_11_19[12] - 16384 cols_helper_11_19[13]
   - 32768 cols_helper_11_19[14] - 65536 cols_helper_11_19[15]
   - 131072 cols_helper_11_19[16] = 0
 cols_acc_11_19: - 435 frac[11,19] + cols_11_19 >= 0
 accuracy_11_19: mem[11,19] - 0.01171875 cols_11_19 = 0
 accuracy_1_25: - 5.09765625 frac[1,25] + mem[1,25] = 0
 accuracy_1_51: - 5.09765625 frac[1,51] + mem[1,51] = 0
 accuracy_6_18: - 5.09765625 frac[6,18] + mem[6,18] = 0
 cols_helper_8_33: cols_helper_8_33[0] + cols_helper_8_33[1]
   + cols_helper_8_33[2] + cols_helper_8_33[3] + cols_helper_8_33[4]
   + cols_helper_8_33[5] + cols_helper_8_33[6] + cols_helper_8_33[7]
   + cols_helper_8_33[8] + cols_helper_8_33[9] + cols_helper_8_33[10]
   + cols_helper_8_33[11] + cols_helper_8_33[12] + cols_helper_8_33[13]
   + cols_helper_8_33[14] + cols_helper_8_33[15] + cols_helper_8_33[16]
   <= 1
 cols_8_33: cols_8_33 - 2 cols_helper_8_33[0] - 4 cols_helper_8_33[1]
   - 8 cols_helper_8_33[2] - 16 cols_helper_8_33[3]
   - 32 cols_helper_8_33[4] - 64 cols_helper_8_33[5]
   - 128 cols_helper_8_33[6] - 256 cols_helper_8_33[7]
   - 512 cols_helper_8_33[8] - 1024 cols_helper_8_33[9]
   - 2048 cols_helper_8_33[10] - 4096 cols_helper_8_33[11]
   - 8192 cols_helper_8_33[12] - 16384 cols_helper_8_33[13]
   - 32768 cols_helper_8_33[14] - 65536 cols_helper_8_33[15]
   - 131072 cols_helper_8_33[16] = 0
 cols_acc_8_33: - 435 frac[8,33] + cols_8_33 >= 0
 accuracy_8_33: mem[8,33] - 0.01171875 cols_8_33 = 0
 accuracy_5_39: - 5.09765625 frac[5,39] + mem[5,39] = 0
 cols_helper_9_60: cols_helper_9_60[0] + cols_helper_9_60[1]
   + cols_helper_9_60[2] + cols_helper_9_60[3] + cols_helper_9_60[4]
   + cols_helper_9_60[5] + cols_helper_9_60[6] + cols_helper_9_60[7]
   + cols_helper_9_60[8] + cols_helper_9_60[9] + cols_helper_9_60[10]
   + cols_helper_9_60[11] + cols_helper_9_60[12] + cols_helper_9_60[13]
   + cols_helper_9_60[14] + cols_helper_9_60[15] + cols_helper_9_60[16]
   <= 1
 cols_9_60: cols_9_60 - 2 cols_helper_9_60[0] - 4 cols_helper_9_60[1]
   - 8 cols_helper_9_60[2] - 16 cols_helper_9_60[3]
   - 32 cols_helper_9_60[4] - 64 cols_helper_9_60[5]
   - 128 cols_helper_9_60[6] - 256 cols_helper_9_60[7]
   - 512 cols_helper_9_60[8] - 1024 cols_helper_9_60[9]
   - 2048 cols_helper_9_60[10] - 4096 cols_helper_9_60[11]
   - 8192 cols_helper_9_60[12] - 16384 cols_helper_9_60[13]
   - 32768 cols_helper_9_60[14] - 65536 cols_helper_9_60[15]
   - 131072 cols_helper_9_60[16] = 0
 cols_acc_9_60: - 435 frac[9,60] + cols_9_60 >= 0
 accuracy_9_60: mem[9,60] - 0.01171875 cols_9_60 = 0
 cols_helper_10_59: cols_helper_10_59[0] + cols_helper_10_59[1]
   + cols_helper_10_59[2] + cols_helper_10_59[3] + cols_helper_10_59[4]
   + cols_helper_10_59[5] + cols_helper_10_59[6] + cols_helper_10_59[7]
   + cols_helper_10_59[8] + cols_helper_10_59[9] + cols_helper_10_59[10]
   + cols_helper_10_59[11] + cols_helper_10_59[12] + cols_helper_10_59[13]
   + cols_helper_10_59[14] + cols_helper_10_59[15] + cols_helper_10_59[16]
   <= 1
 cols_10_59: cols_10_59 - 2 cols_helper_10_59[0] - 4 cols_helper_10_59[1]
   - 8 cols_helper_10_59[2] - 16 cols_helper_10_59[3]
   - 32 cols_helper_10_59[4] - 64 cols_helper_10_59[5]
   - 128 cols_helper_10_59[6] - 256 cols_helper_10_59[7]
   - 512 cols_helper_10_59[8] - 1024 cols_helper_10_59[9]
   - 2048 cols_helper_10_59[10] - 4096 cols_helper_10_59[11]
   - 8192 cols_helper_10_59[12] - 16384 cols_helper_10_59[13]
   - 32768 cols_helper_10_59[14] - 65536 cols_helper_10_59[15]
   - 131072 cols_helper_10_59[16] = 0
 cols_acc_10_59: - 435 frac[10,59] + cols_10_59 >= 0
 accuracy_10_59: mem[10,59] - 0.01171875 cols_10_59 = 0
 cols_helper_8_7: cols_helper_8_7[0] + cols_helper_8_7[1]
   + cols_helper_8_7[2] + cols_helper_8_7[3] + cols_helper_8_7[4]
   + cols_helper_8_7[5] + cols_helper_8_7[6] + cols_helper_8_7[7]
   + cols_helper_8_7[8] + cols_helper_8_7[9] + cols_helper_8_7[10]
   + cols_helper_8_7[11] + cols_helper_8_7[12] + cols_helper_8_7[13]
   + cols_helper_8_7[14] + cols_helper_8_7[15] + cols_helper_8_7[16] <= 1
 cols_8_7: cols_8_7 - 2 cols_helper_8_7[0] - 4 cols_helper_8_7[1]
   - 8 cols_helper_8_7[2] - 16 cols_helper_8_7[3] - 32 cols_helper_8_7[4]
   - 64 cols_helper_8_7[5] - 128 cols_helper_8_7[6]
   - 256 cols_helper_8_7[7] - 512 cols_helper_8_7[8]
   - 1024 cols_helper_8_7[9] - 2048 cols_helper_8_7[10]
   - 4096 cols_helper_8_7[11] - 8192 cols_helper_8_7[12]
   - 16384 cols_helper_8_7[13] - 32768 cols_helper_8_7[14]
   - 65536 cols_helper_8_7[15] - 131072 cols_helper_8_7[16] = 0
 cols_acc_8_7: - 435 frac[8,7] + cols_8_7 >= 0
 accuracy_8_7: mem[8,7] - 0.01171875 cols_8_7 = 0
 cols_helper_11_50: cols_helper_11_50[0] + cols_helper_11_50[1]
   + cols_helper_11_50[2] + cols_helper_11_50[3] + cols_helper_11_50[4]
   + cols_helper_11_50[5] + cols_helper_11_50[6] + cols_helper_11_50[7]
   + cols_helper_11_50[8] + cols_helper_11_50[9] + cols_helper_11_50[10]
   + cols_helper_11_50[11] + cols_helper_11_50[12] + cols_helper_11_50[13]
   + cols_helper_11_50[14] + cols_helper_11_50[15] + cols_helper_11_50[16]
   <= 1
 cols_11_50: cols_11_50 - 2 cols_helper_11_50[0] - 4 cols_helper_11_50[1]
   - 8 cols_helper_11_50[2] - 16 cols_helper_11_50[3]
   - 32 cols_helper_11_50[4] - 64 cols_helper_11_50[5]
   - 128 cols_helper_11_50[6] - 256 cols_helper_11_50[7]
   - 512 cols_helper_11_50[8] - 1024 cols_helper_11_50[9]
   - 2048 cols_helper_11_50[10] - 4096 cols_helper_11_50[11]
   - 8192 cols_helper_11_50[12] - 16384 cols_helper_11_50[13]
   - 32768 cols_helper_11_50[14] - 65536 cols_helper_11_50[15]
   - 131072 cols_helper_11_50[16] = 0
 cols_acc_11_50: - 435 frac[11,50] + cols_11_50 >= 0
 accuracy_11_50: mem[11,50] - 0.01171875 cols_11_50 = 0
 cols_helper_10_5: cols_helper_10_5[0] + cols_helper_10_5[1]
   + cols_helper_10_5[2] + cols_helper_10_5[3] + cols_helper_10_5[4]
   + cols_helper_10_5[5] + cols_helper_10_5[6] + cols_helper_10_5[7]
   + cols_helper_10_5[8] + cols_helper_10_5[9] + cols_helper_10_5[10]
   + cols_helper_10_5[11] + cols_helper_10_5[12] + cols_helper_10_5[13]
   + cols_helper_10_5[14] + cols_helper_10_5[15] + cols_helper_10_5[16]
   <= 1
 cols_10_5: cols_10_5 - 2 cols_helper_10_5[0] - 4 cols_helper_10_5[1]
   - 8 cols_helper_10_5[2] - 16 cols_helper_10_5[3]
   - 32 cols_helper_10_5[4] - 64 cols_helper_10_5[5]
   - 128 cols_helper_10_5[6] - 256 cols_helper_10_5[7]
   - 512 cols_helper_10_5[8] - 1024 cols_helper_10_5[9]
   - 2048 cols_helper_10_5[10] - 4096 cols_helper_10_5[11]
   - 8192 cols_helper_10_5[12] - 16384 cols_helper_10_5[13]
   - 32768 cols_helper_10_5[14] - 65536 cols_helper_10_5[15]
   - 131072 cols_helper_10_5[16] = 0
 cols_acc_10_5: - 435 frac[10,5] + cols_10_5 >= 0
 accuracy_10_5: mem[10,5] - 0.01171875 cols_10_5 = 0
 cols_helper_11_8: cols_helper_11_8[0] + cols_helper_11_8[1]
   + cols_helper_11_8[2] + cols_helper_11_8[3] + cols_helper_11_8[4]
   + cols_helper_11_8[5] + cols_helper_11_8[6] + cols_helper_11_8[7]
   + cols_helper_11_8[8] + cols_helper_11_8[9] + cols_helper_11_8[10]
   + cols_helper_11_8[11] + cols_helper_11_8[12] + cols_helper_11_8[13]
   + cols_helper_11_8[14] + cols_helper_11_8[15] + cols_helper_11_8[16]
   <= 1
 cols_11_8: cols_11_8 - 2 cols_helper_11_8[0] - 4 cols_helper_11_8[1]
   - 8 cols_helper_11_8[2] - 16 cols_helper_11_8[3]
   - 32 cols_helper_11_8[4] - 64 cols_helper_11_8[5]
   - 128 cols_helper_11_8[6] - 256 cols_helper_11_8[7]
   - 512 cols_helper_11_8[8] - 1024 cols_helper_11_8[9]
   - 2048 cols_helper_11_8[10] - 4096 cols_helper_11_8[11]
   - 8192 cols_helper_11_8[12] - 16384 cols_helper_11_8[13]
   - 32768 cols_helper_11_8[14] - 65536 cols_helper_11_8[15]
   - 131072 cols_helper_11_8[16] = 0
 cols_acc_11_8: - 435 frac[11,8] + cols_11_8 >= 0
 accuracy_11_8: mem[11,8] - 0.01171875 cols_11_8 = 0
 accuracy_0_19: - 5.09765625 frac[0,19] + mem[0,19] = 0
 accuracy_2_25: - 5.09765625 frac[2,25] + mem[2,25] = 0
 accuracy_0_9: - 5.09765625 frac[0,9] + mem[0,9] = 0
 cols_helper_10_73: cols_helper_10_73[0] + cols_helper_10_73[1]
   + cols_helper_10_73[2] + cols_helper_10_73[3] + cols_helper_10_73[4]
   + cols_helper_10_73[5] + cols_helper_10_73[6] + cols_helper_10_73[7]
   + cols_helper_10_73[8] + cols_helper_10_73[9] + cols_helper_10_73[10]
   + cols_helper_10_73[11] + cols_helper_10_73[12] + cols_helper_10_73[13]
   + cols_helper_10_73[14] + cols_helper_10_73[15] + cols_helper_10_73[16]
   <= 1
 cols_10_73: cols_10_73 - 2 cols_helper_10_73[0] - 4 cols_helper_10_73[1]
   - 8 cols_helper_10_73[2] - 16 cols_helper_10_73[3]
   - 32 cols_helper_10_73[4] - 64 cols_helper_10_73[5]
   - 128 cols_helper_10_73[6] - 256 cols_helper_10_73[7]
   - 512 cols_helper_10_73[8] - 1024 cols_helper_10_73[9]
   - 2048 cols_helper_10_73[10] - 4096 cols_helper_10_73[11]
   - 8192 cols_helper_10_73[12] - 16384 cols_helper_10_73[13]
   - 32768 cols_helper_10_73[14] - 65536 cols_helper_10_73[15]
   - 131072 cols_helper_10_73[16] = 0
 cols_acc_10_73: - 435 frac[10,73] + cols_10_73 >= 0
 accuracy_10_73: mem[10,73] - 0.01171875 cols_10_73 = 0
 accuracy_2_51: - 5.09765625 frac[2,51] + mem[2,51] = 0
 cols_helper_9_67: cols_helper_9_67[0] + cols_helper_9_67[1]
   + cols_helper_9_67[2] + cols_helper_9_67[3] + cols_helper_9_67[4]
   + cols_helper_9_67[5] + cols_helper_9_67[6] + cols_helper_9_67[7]
   + cols_helper_9_67[8] + cols_helper_9_67[9] + cols_helper_9_67[10]
   + cols_helper_9_67[11] + cols_helper_9_67[12] + cols_helper_9_67[13]
   + cols_helper_9_67[14] + cols_helper_9_67[15] + cols_helper_9_67[16]
   <= 1
 cols_9_67: cols_9_67 - 2 cols_helper_9_67[0] - 4 cols_helper_9_67[1]
   - 8 cols_helper_9_67[2] - 16 cols_helper_9_67[3]
   - 32 cols_helper_9_67[4] - 64 cols_helper_9_67[5]
   - 128 cols_helper_9_67[6] - 256 cols_helper_9_67[7]
   - 512 cols_helper_9_67[8] - 1024 cols_helper_9_67[9]
   - 2048 cols_helper_9_67[10] - 4096 cols_helper_9_67[11]
   - 8192 cols_helper_9_67[12] - 16384 cols_helper_9_67[13]
   - 32768 cols_helper_9_67[14] - 65536 cols_helper_9_67[15]
   - 131072 cols_helper_9_67[16] = 0
 cols_acc_9_67: - 435 frac[9,67] + cols_9_67 >= 0
 accuracy_9_67: mem[9,67] - 0.01171875 cols_9_67 = 0
 accuracy_6_21: - 5.09765625 frac[6,21] + mem[6,21] = 0
 accuracy_4_45: - 5.09765625 frac[4,45] + mem[4,45] = 0
 cols_helper_8_40: cols_helper_8_40[0] + cols_helper_8_40[1]
   + cols_helper_8_40[2] + cols_helper_8_40[3] + cols_helper_8_40[4]
   + cols_helper_8_40[5] + cols_helper_8_40[6] + cols_helper_8_40[7]
   + cols_helper_8_40[8] + cols_helper_8_40[9] + cols_helper_8_40[10]
   + cols_helper_8_40[11] + cols_helper_8_40[12] + cols_helper_8_40[13]
   + cols_helper_8_40[14] + cols_helper_8_40[15] + cols_helper_8_40[16]
   <= 1
 cols_8_40: cols_8_40 - 2 cols_helper_8_40[0] - 4 cols_helper_8_40[1]
   - 8 cols_helper_8_40[2] - 16 cols_helper_8_40[3]
   - 32 cols_helper_8_40[4] - 64 cols_helper_8_40[5]
   - 128 cols_helper_8_40[6] - 256 cols_helper_8_40[7]
   - 512 cols_helper_8_40[8] - 1024 cols_helper_8_40[9]
   - 2048 cols_helper_8_40[10] - 4096 cols_helper_8_40[11]
   - 8192 cols_helper_8_40[12] - 16384 cols_helper_8_40[13]
   - 32768 cols_helper_8_40[14] - 65536 cols_helper_8_40[15]
   - 131072 cols_helper_8_40[16] = 0
 cols_acc_8_40: - 435 frac[8,40] + cols_8_40 >= 0
 accuracy_8_40: mem[8,40] - 0.01171875 cols_8_40 = 0
 accuracy_0_69: - 5.09765625 frac[0,69] + mem[0,69] = 0
 accuracy_5_32: - 5.09765625 frac[5,32] + mem[5,32] = 0
 cols_helper_9_37: cols_helper_9_37[0] + cols_helper_9_37[1]
   + cols_helper_9_37[2] + cols_helper_9_37[3] + cols_helper_9_37[4]
   + cols_helper_9_37[5] + cols_helper_9_37[6] + cols_helper_9_37[7]
   + cols_helper_9_37[8] + cols_helper_9_37[9] + cols_helper_9_37[10]
   + cols_helper_9_37[11] + cols_helper_9_37[12] + cols_helper_9_37[13]
   + cols_helper_9_37[14] + cols_helper_9_37[15] + cols_helper_9_37[16]
   <= 1
 cols_9_37: cols_9_37 - 2 cols_helper_9_37[0] - 4 cols_helper_9_37[1]
   - 8 cols_helper_9_37[2] - 16 cols_helper_9_37[3]
   - 32 cols_helper_9_37[4] - 64 cols_helper_9_37[5]
   - 128 cols_helper_9_37[6] - 256 cols_helper_9_37[7]
   - 512 cols_helper_9_37[8] - 1024 cols_helper_9_37[9]
   - 2048 cols_helper_9_37[10] - 4096 cols_helper_9_37[11]
   - 8192 cols_helper_9_37[12] - 16384 cols_helper_9_37[13]
   - 32768 cols_helper_9_37[14] - 65536 cols_helper_9_37[15]
   - 131072 cols_helper_9_37[16] = 0
 cols_acc_9_37: - 435 frac[9,37] + cols_9_37 >= 0
 accuracy_9_37: mem[9,37] - 0.01171875 cols_9_37 = 0
 accuracy_2_71: - 5.09765625 frac[2,71] + mem[2,71] = 0
 cols_helper_8_14: cols_helper_8_14[0] + cols_helper_8_14[1]
   + cols_helper_8_14[2] + cols_helper_8_14[3] + cols_helper_8_14[4]
   + cols_helper_8_14[5] + cols_helper_8_14[6] + cols_helper_8_14[7]
   + cols_helper_8_14[8] + cols_helper_8_14[9] + cols_helper_8_14[10]
   + cols_helper_8_14[11] + cols_helper_8_14[12] + cols_helper_8_14[13]
   + cols_helper_8_14[14] + cols_helper_8_14[15] + cols_helper_8_14[16]
   <= 1
 cols_8_14: cols_8_14 - 2 cols_helper_8_14[0] - 4 cols_helper_8_14[1]
   - 8 cols_helper_8_14[2] - 16 cols_helper_8_14[3]
   - 32 cols_helper_8_14[4] - 64 cols_helper_8_14[5]
   - 128 cols_helper_8_14[6] - 256 cols_helper_8_14[7]
   - 512 cols_helper_8_14[8] - 1024 cols_helper_8_14[9]
   - 2048 cols_helper_8_14[10] - 4096 cols_helper_8_14[11]
   - 8192 cols_helper_8_14[12] - 16384 cols_helper_8_14[13]
   - 32768 cols_helper_8_14[14] - 65536 cols_helper_8_14[15]
   - 131072 cols_helper_8_14[16] = 0
 cols_acc_8_14: - 435 frac[8,14] + cols_8_14 >= 0
 accuracy_8_14: mem[8,14] - 0.01171875 cols_8_14 = 0
 cols_helper_10_12: cols_helper_10_12[0] + cols_helper_10_12[1]
   + cols_helper_10_12[2] + cols_helper_10_12[3] + cols_helper_10_12[4]
   + cols_helper_10_12[5] + cols_helper_10_12[6] + cols_helper_10_12[7]
   + cols_helper_10_12[8] + cols_helper_10_12[9] + cols_helper_10_12[10]
   + cols_helper_10_12[11] + cols_helper_10_12[12] + cols_helper_10_12[13]
   + cols_helper_10_12[14] + cols_helper_10_12[15] + cols_helper_10_12[16]
   <= 1
 cols_10_12: cols_10_12 - 2 cols_helper_10_12[0] - 4 cols_helper_10_12[1]
   - 8 cols_helper_10_12[2] - 16 cols_helper_10_12[3]
   - 32 cols_helper_10_12[4] - 64 cols_helper_10_12[5]
   - 128 cols_helper_10_12[6] - 256 cols_helper_10_12[7]
   - 512 cols_helper_10_12[8] - 1024 cols_helper_10_12[9]
   - 2048 cols_helper_10_12[10] - 4096 cols_helper_10_12[11]
   - 8192 cols_helper_10_12[12] - 16384 cols_helper_10_12[13]
   - 32768 cols_helper_10_12[14] - 65536 cols_helper_10_12[15]
   - 131072 cols_helper_10_12[16] = 0
 cols_acc_10_12: - 435 frac[10,12] + cols_10_12 >= 0
 accuracy_10_12: mem[10,12] - 0.01171875 cols_10_12 = 0
 cols_helper_11_1: cols_helper_11_1[0] + cols_helper_11_1[1]
   + cols_helper_11_1[2] + cols_helper_11_1[3] + cols_helper_11_1[4]
   + cols_helper_11_1[5] + cols_helper_11_1[6] + cols_helper_11_1[7]
   + cols_helper_11_1[8] + cols_helper_11_1[9] + cols_helper_11_1[10]
   + cols_helper_11_1[11] + cols_helper_11_1[12] + cols_helper_11_1[13]
   + cols_helper_11_1[14] + cols_helper_11_1[15] + cols_helper_11_1[16]
   <= 1
 cols_11_1: cols_11_1 - 2 cols_helper_11_1[0] - 4 cols_helper_11_1[1]
   - 8 cols_helper_11_1[2] - 16 cols_helper_11_1[3]
   - 32 cols_helper_11_1[4] - 64 cols_helper_11_1[5]
   - 128 cols_helper_11_1[6] - 256 cols_helper_11_1[7]
   - 512 cols_helper_11_1[8] - 1024 cols_helper_11_1[9]
   - 2048 cols_helper_11_1[10] - 4096 cols_helper_11_1[11]
   - 8192 cols_helper_11_1[12] - 16384 cols_helper_11_1[13]
   - 32768 cols_helper_11_1[14] - 65536 cols_helper_11_1[15]
   - 131072 cols_helper_11_1[16] = 0
 cols_acc_11_1: - 435 frac[11,1] + cols_11_1 >= 0
 accuracy_11_1: mem[11,1] - 0.01171875 cols_11_1 = 0
 cols_helper_9_17: cols_helper_9_17[0] + cols_helper_9_17[1]
   + cols_helper_9_17[2] + cols_helper_9_17[3] + cols_helper_9_17[4]
   + cols_helper_9_17[5] + cols_helper_9_17[6] + cols_helper_9_17[7]
   + cols_helper_9_17[8] + cols_helper_9_17[9] + cols_helper_9_17[10]
   + cols_helper_9_17[11] + cols_helper_9_17[12] + cols_helper_9_17[13]
   + cols_helper_9_17[14] + cols_helper_9_17[15] + cols_helper_9_17[16]
   <= 1
 cols_9_17: cols_9_17 - 2 cols_helper_9_17[0] - 4 cols_helper_9_17[1]
   - 8 cols_helper_9_17[2] - 16 cols_helper_9_17[3]
   - 32 cols_helper_9_17[4] - 64 cols_helper_9_17[5]
   - 128 cols_helper_9_17[6] - 256 cols_helper_9_17[7]
   - 512 cols_helper_9_17[8] - 1024 cols_helper_9_17[9]
   - 2048 cols_helper_9_17[10] - 4096 cols_helper_9_17[11]
   - 8192 cols_helper_9_17[12] - 16384 cols_helper_9_17[13]
   - 32768 cols_helper_9_17[14] - 65536 cols_helper_9_17[15]
   - 131072 cols_helper_9_17[16] = 0
 cols_acc_9_17: - 435 frac[9,17] + cols_9_17 >= 0
 accuracy_9_17: mem[9,17] - 0.01171875 cols_9_17 = 0
 cols_helper_10_22: cols_helper_10_22[0] + cols_helper_10_22[1]
   + cols_helper_10_22[2] + cols_helper_10_22[3] + cols_helper_10_22[4]
   + cols_helper_10_22[5] + cols_helper_10_22[6] + cols_helper_10_22[7]
   + cols_helper_10_22[8] + cols_helper_10_22[9] + cols_helper_10_22[10]
   + cols_helper_10_22[11] + cols_helper_10_22[12] + cols_helper_10_22[13]
   + cols_helper_10_22[14] + cols_helper_10_22[15] + cols_helper_10_22[16]
   <= 1
 cols_10_22: cols_10_22 - 2 cols_helper_10_22[0] - 4 cols_helper_10_22[1]
   - 8 cols_helper_10_22[2] - 16 cols_helper_10_22[3]
   - 32 cols_helper_10_22[4] - 64 cols_helper_10_22[5]
   - 128 cols_helper_10_22[6] - 256 cols_helper_10_22[7]
   - 512 cols_helper_10_22[8] - 1024 cols_helper_10_22[9]
   - 2048 cols_helper_10_22[10] - 4096 cols_helper_10_22[11]
   - 8192 cols_helper_10_22[12] - 16384 cols_helper_10_22[13]
   - 32768 cols_helper_10_22[14] - 65536 cols_helper_10_22[15]
   - 131072 cols_helper_10_22[16] = 0
 cols_acc_10_22: - 435 frac[10,22] + cols_10_22 >= 0
 accuracy_10_22: mem[10,22] - 0.01171875 cols_10_22 = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 cols_helper_8_64: cols_helper_8_64[0] + cols_helper_8_64[1]
   + cols_helper_8_64[2] + cols_helper_8_64[3] + cols_helper_8_64[4]
   + cols_helper_8_64[5] + cols_helper_8_64[6] + cols_helper_8_64[7]
   + cols_helper_8_64[8] + cols_helper_8_64[9] + cols_helper_8_64[10]
   + cols_helper_8_64[11] + cols_helper_8_64[12] + cols_helper_8_64[13]
   + cols_helper_8_64[14] + cols_helper_8_64[15] + cols_helper_8_64[16]
   <= 1
 cols_8_64: cols_8_64 - 2 cols_helper_8_64[0] - 4 cols_helper_8_64[1]
   - 8 cols_helper_8_64[2] - 16 cols_helper_8_64[3]
   - 32 cols_helper_8_64[4] - 64 cols_helper_8_64[5]
   - 128 cols_helper_8_64[6] - 256 cols_helper_8_64[7]
   - 512 cols_helper_8_64[8] - 1024 cols_helper_8_64[9]
   - 2048 cols_helper_8_64[10] - 4096 cols_helper_8_64[11]
   - 8192 cols_helper_8_64[12] - 16384 cols_helper_8_64[13]
   - 32768 cols_helper_8_64[14] - 65536 cols_helper_8_64[15]
   - 131072 cols_helper_8_64[16] = 0
 cols_acc_8_64: - 435 frac[8,64] + cols_8_64 >= 0
 accuracy_8_64: mem[8,64] - 0.01171875 cols_8_64 = 0
 accuracy_4_14: - 5.09765625 frac[4,14] + mem[4,14] = 0
 cols_helper_9_68: cols_helper_9_68[0] + cols_helper_9_68[1]
   + cols_helper_9_68[2] + cols_helper_9_68[3] + cols_helper_9_68[4]
   + cols_helper_9_68[5] + cols_helper_9_68[6] + cols_helper_9_68[7]
   + cols_helper_9_68[8] + cols_helper_9_68[9] + cols_helper_9_68[10]
   + cols_helper_9_68[11] + cols_helper_9_68[12] + cols_helper_9_68[13]
   + cols_helper_9_68[14] + cols_helper_9_68[15] + cols_helper_9_68[16]
   <= 1
 cols_9_68: cols_9_68 - 2 cols_helper_9_68[0] - 4 cols_helper_9_68[1]
   - 8 cols_helper_9_68[2] - 16 cols_helper_9_68[3]
   - 32 cols_helper_9_68[4] - 64 cols_helper_9_68[5]
   - 128 cols_helper_9_68[6] - 256 cols_helper_9_68[7]
   - 512 cols_helper_9_68[8] - 1024 cols_helper_9_68[9]
   - 2048 cols_helper_9_68[10] - 4096 cols_helper_9_68[11]
   - 8192 cols_helper_9_68[12] - 16384 cols_helper_9_68[13]
   - 32768 cols_helper_9_68[14] - 65536 cols_helper_9_68[15]
   - 131072 cols_helper_9_68[16] = 0
 cols_acc_9_68: - 435 frac[9,68] + cols_9_68 >= 0
 accuracy_9_68: mem[9,68] - 0.01171875 cols_9_68 = 0
 cols_helper_8_47: cols_helper_8_47[0] + cols_helper_8_47[1]
   + cols_helper_8_47[2] + cols_helper_8_47[3] + cols_helper_8_47[4]
   + cols_helper_8_47[5] + cols_helper_8_47[6] + cols_helper_8_47[7]
   + cols_helper_8_47[8] + cols_helper_8_47[9] + cols_helper_8_47[10]
   + cols_helper_8_47[11] + cols_helper_8_47[12] + cols_helper_8_47[13]
   + cols_helper_8_47[14] + cols_helper_8_47[15] + cols_helper_8_47[16]
   <= 1
 cols_8_47: cols_8_47 - 2 cols_helper_8_47[0] - 4 cols_helper_8_47[1]
   - 8 cols_helper_8_47[2] - 16 cols_helper_8_47[3]
   - 32 cols_helper_8_47[4] - 64 cols_helper_8_47[5]
   - 128 cols_helper_8_47[6] - 256 cols_helper_8_47[7]
   - 512 cols_helper_8_47[8] - 1024 cols_helper_8_47[9]
   - 2048 cols_helper_8_47[10] - 4096 cols_helper_8_47[11]
   - 8192 cols_helper_8_47[12] - 16384 cols_helper_8_47[13]
   - 32768 cols_helper_8_47[14] - 65536 cols_helper_8_47[15]
   - 131072 cols_helper_8_47[16] = 0
 cols_acc_8_47: - 435 frac[8,47] + cols_8_47 >= 0
 accuracy_8_47: mem[8,47] - 0.01171875 cols_8_47 = 0
 cols_helper_11_74: cols_helper_11_74[0] + cols_helper_11_74[1]
   + cols_helper_11_74[2] + cols_helper_11_74[3] + cols_helper_11_74[4]
   + cols_helper_11_74[5] + cols_helper_11_74[6] + cols_helper_11_74[7]
   + cols_helper_11_74[8] + cols_helper_11_74[9] + cols_helper_11_74[10]
   + cols_helper_11_74[11] + cols_helper_11_74[12] + cols_helper_11_74[13]
   + cols_helper_11_74[14] + cols_helper_11_74[15] + cols_helper_11_74[16]
   <= 1
 cols_11_74: cols_11_74 - 2 cols_helper_11_74[0] - 4 cols_helper_11_74[1]
   - 8 cols_helper_11_74[2] - 16 cols_helper_11_74[3]
   - 32 cols_helper_11_74[4] - 64 cols_helper_11_74[5]
   - 128 cols_helper_11_74[6] - 256 cols_helper_11_74[7]
   - 512 cols_helper_11_74[8] - 1024 cols_helper_11_74[9]
   - 2048 cols_helper_11_74[10] - 4096 cols_helper_11_74[11]
   - 8192 cols_helper_11_74[12] - 16384 cols_helper_11_74[13]
   - 32768 cols_helper_11_74[14] - 65536 cols_helper_11_74[15]
   - 131072 cols_helper_11_74[16] = 0
 cols_acc_11_74: - 435 frac[11,74] + cols_11_74 >= 0
 accuracy_11_74: mem[11,74] - 0.01171875 cols_11_74 = 0
 cols_helper_9_46: cols_helper_9_46[0] + cols_helper_9_46[1]
   + cols_helper_9_46[2] + cols_helper_9_46[3] + cols_helper_9_46[4]
   + cols_helper_9_46[5] + cols_helper_9_46[6] + cols_helper_9_46[7]
   + cols_helper_9_46[8] + cols_helper_9_46[9] + cols_helper_9_46[10]
   + cols_helper_9_46[11] + cols_helper_9_46[12] + cols_helper_9_46[13]
   + cols_helper_9_46[14] + cols_helper_9_46[15] + cols_helper_9_46[16]
   <= 1
 cols_9_46: cols_9_46 - 2 cols_helper_9_46[0] - 4 cols_helper_9_46[1]
   - 8 cols_helper_9_46[2] - 16 cols_helper_9_46[3]
   - 32 cols_helper_9_46[4] - 64 cols_helper_9_46[5]
   - 128 cols_helper_9_46[6] - 256 cols_helper_9_46[7]
   - 512 cols_helper_9_46[8] - 1024 cols_helper_9_46[9]
   - 2048 cols_helper_9_46[10] - 4096 cols_helper_9_46[11]
   - 8192 cols_helper_9_46[12] - 16384 cols_helper_9_46[13]
   - 32768 cols_helper_9_46[14] - 65536 cols_helper_9_46[15]
   - 131072 cols_helper_9_46[16] = 0
 cols_acc_9_46: - 435 frac[9,46] + cols_9_46 >= 0
 accuracy_9_46: mem[9,46] - 0.01171875 cols_9_46 = 0
 cols_helper_10_45: cols_helper_10_45[0] + cols_helper_10_45[1]
   + cols_helper_10_45[2] + cols_helper_10_45[3] + cols_helper_10_45[4]
   + cols_helper_10_45[5] + cols_helper_10_45[6] + cols_helper_10_45[7]
   + cols_helper_10_45[8] + cols_helper_10_45[9] + cols_helper_10_45[10]
   + cols_helper_10_45[11] + cols_helper_10_45[12] + cols_helper_10_45[13]
   + cols_helper_10_45[14] + cols_helper_10_45[15] + cols_helper_10_45[16]
   <= 1
 cols_10_45: cols_10_45 - 2 cols_helper_10_45[0] - 4 cols_helper_10_45[1]
   - 8 cols_helper_10_45[2] - 16 cols_helper_10_45[3]
   - 32 cols_helper_10_45[4] - 64 cols_helper_10_45[5]
   - 128 cols_helper_10_45[6] - 256 cols_helper_10_45[7]
   - 512 cols_helper_10_45[8] - 1024 cols_helper_10_45[9]
   - 2048 cols_helper_10_45[10] - 4096 cols_helper_10_45[11]
   - 8192 cols_helper_10_45[12] - 16384 cols_helper_10_45[13]
   - 32768 cols_helper_10_45[14] - 65536 cols_helper_10_45[15]
   - 131072 cols_helper_10_45[16] = 0
 cols_acc_10_45: - 435 frac[10,45] + cols_10_45 >= 0
 accuracy_10_45: mem[10,45] - 0.01171875 cols_10_45 = 0
 cols_helper_8_53: cols_helper_8_53[0] + cols_helper_8_53[1]
   + cols_helper_8_53[2] + cols_helper_8_53[3] + cols_helper_8_53[4]
   + cols_helper_8_53[5] + cols_helper_8_53[6] + cols_helper_8_53[7]
   + cols_helper_8_53[8] + cols_helper_8_53[9] + cols_helper_8_53[10]
   + cols_helper_8_53[11] + cols_helper_8_53[12] + cols_helper_8_53[13]
   + cols_helper_8_53[14] + cols_helper_8_53[15] + cols_helper_8_53[16]
   <= 1
 cols_8_53: cols_8_53 - 2 cols_helper_8_53[0] - 4 cols_helper_8_53[1]
   - 8 cols_helper_8_53[2] - 16 cols_helper_8_53[3]
   - 32 cols_helper_8_53[4] - 64 cols_helper_8_53[5]
   - 128 cols_helper_8_53[6] - 256 cols_helper_8_53[7]
   - 512 cols_helper_8_53[8] - 1024 cols_helper_8_53[9]
   - 2048 cols_helper_8_53[10] - 4096 cols_helper_8_53[11]
   - 8192 cols_helper_8_53[12] - 16384 cols_helper_8_53[13]
   - 32768 cols_helper_8_53[14] - 65536 cols_helper_8_53[15]
   - 131072 cols_helper_8_53[16] = 0
 cols_acc_8_53: - 435 frac[8,53] + cols_8_53 >= 0
 accuracy_8_53: mem[8,53] - 0.01171875 cols_8_53 = 0
 accuracy_7_39: - 5.09765625 frac[7,39] + mem[7,39] = 0
 cols_helper_11_32: cols_helper_11_32[0] + cols_helper_11_32[1]
   + cols_helper_11_32[2] + cols_helper_11_32[3] + cols_helper_11_32[4]
   + cols_helper_11_32[5] + cols_helper_11_32[6] + cols_helper_11_32[7]
   + cols_helper_11_32[8] + cols_helper_11_32[9] + cols_helper_11_32[10]
   + cols_helper_11_32[11] + cols_helper_11_32[12] + cols_helper_11_32[13]
   + cols_helper_11_32[14] + cols_helper_11_32[15] + cols_helper_11_32[16]
   <= 1
 cols_11_32: cols_11_32 - 2 cols_helper_11_32[0] - 4 cols_helper_11_32[1]
   - 8 cols_helper_11_32[2] - 16 cols_helper_11_32[3]
   - 32 cols_helper_11_32[4] - 64 cols_helper_11_32[5]
   - 128 cols_helper_11_32[6] - 256 cols_helper_11_32[7]
   - 512 cols_helper_11_32[8] - 1024 cols_helper_11_32[9]
   - 2048 cols_helper_11_32[10] - 4096 cols_helper_11_32[11]
   - 8192 cols_helper_11_32[12] - 16384 cols_helper_11_32[13]
   - 32768 cols_helper_11_32[14] - 65536 cols_helper_11_32[15]
   - 131072 cols_helper_11_32[16] = 0
 cols_acc_11_32: - 435 frac[11,32] + cols_11_32 >= 0
 accuracy_11_32: mem[11,32] - 0.01171875 cols_11_32 = 0
 cols_helper_10_55: cols_helper_10_55[0] + cols_helper_10_55[1]
   + cols_helper_10_55[2] + cols_helper_10_55[3] + cols_helper_10_55[4]
   + cols_helper_10_55[5] + cols_helper_10_55[6] + cols_helper_10_55[7]
   + cols_helper_10_55[8] + cols_helper_10_55[9] + cols_helper_10_55[10]
   + cols_helper_10_55[11] + cols_helper_10_55[12] + cols_helper_10_55[13]
   + cols_helper_10_55[14] + cols_helper_10_55[15] + cols_helper_10_55[16]
   <= 1
 cols_10_55: cols_10_55 - 2 cols_helper_10_55[0] - 4 cols_helper_10_55[1]
   - 8 cols_helper_10_55[2] - 16 cols_helper_10_55[3]
   - 32 cols_helper_10_55[4] - 64 cols_helper_10_55[5]
   - 128 cols_helper_10_55[6] - 256 cols_helper_10_55[7]
   - 512 cols_helper_10_55[8] - 1024 cols_helper_10_55[9]
   - 2048 cols_helper_10_55[10] - 4096 cols_helper_10_55[11]
   - 8192 cols_helper_10_55[12] - 16384 cols_helper_10_55[13]
   - 32768 cols_helper_10_55[14] - 65536 cols_helper_10_55[15]
   - 131072 cols_helper_10_55[16] = 0
 cols_acc_10_55: - 435 frac[10,55] + cols_10_55 >= 0
 accuracy_10_55: mem[10,55] - 0.01171875 cols_10_55 = 0
 accuracy_1_74: - 5.09765625 frac[1,74] + mem[1,74] = 0
 cols_helper_8_27: cols_helper_8_27[0] + cols_helper_8_27[1]
   + cols_helper_8_27[2] + cols_helper_8_27[3] + cols_helper_8_27[4]
   + cols_helper_8_27[5] + cols_helper_8_27[6] + cols_helper_8_27[7]
   + cols_helper_8_27[8] + cols_helper_8_27[9] + cols_helper_8_27[10]
   + cols_helper_8_27[11] + cols_helper_8_27[12] + cols_helper_8_27[13]
   + cols_helper_8_27[14] + cols_helper_8_27[15] + cols_helper_8_27[16]
   <= 1
 cols_8_27: cols_8_27 - 2 cols_helper_8_27[0] - 4 cols_helper_8_27[1]
   - 8 cols_helper_8_27[2] - 16 cols_helper_8_27[3]
   - 32 cols_helper_8_27[4] - 64 cols_helper_8_27[5]
   - 128 cols_helper_8_27[6] - 256 cols_helper_8_27[7]
   - 512 cols_helper_8_27[8] - 1024 cols_helper_8_27[9]
   - 2048 cols_helper_8_27[10] - 4096 cols_helper_8_27[11]
   - 8192 cols_helper_8_27[12] - 16384 cols_helper_8_27[13]
   - 32768 cols_helper_8_27[14] - 65536 cols_helper_8_27[15]
   - 131072 cols_helper_8_27[16] = 0
 cols_acc_8_27: - 435 frac[8,27] + cols_8_27 >= 0
 accuracy_8_27: mem[8,27] - 0.01171875 cols_8_27 = 0
 cols_helper_11_6: cols_helper_11_6[0] + cols_helper_11_6[1]
   + cols_helper_11_6[2] + cols_helper_11_6[3] + cols_helper_11_6[4]
   + cols_helper_11_6[5] + cols_helper_11_6[6] + cols_helper_11_6[7]
   + cols_helper_11_6[8] + cols_helper_11_6[9] + cols_helper_11_6[10]
   + cols_helper_11_6[11] + cols_helper_11_6[12] + cols_helper_11_6[13]
   + cols_helper_11_6[14] + cols_helper_11_6[15] + cols_helper_11_6[16]
   <= 1
 cols_11_6: cols_11_6 - 2 cols_helper_11_6[0] - 4 cols_helper_11_6[1]
   - 8 cols_helper_11_6[2] - 16 cols_helper_11_6[3]
   - 32 cols_helper_11_6[4] - 64 cols_helper_11_6[5]
   - 128 cols_helper_11_6[6] - 256 cols_helper_11_6[7]
   - 512 cols_helper_11_6[8] - 1024 cols_helper_11_6[9]
   - 2048 cols_helper_11_6[10] - 4096 cols_helper_11_6[11]
   - 8192 cols_helper_11_6[12] - 16384 cols_helper_11_6[13]
   - 32768 cols_helper_11_6[14] - 65536 cols_helper_11_6[15]
   - 131072 cols_helper_11_6[16] = 0
 cols_acc_11_6: - 435 frac[11,6] + cols_11_6 >= 0
 accuracy_11_6: mem[11,6] - 0.01171875 cols_11_6 = 0
 cols_helper_10_17: cols_helper_10_17[0] + cols_helper_10_17[1]
   + cols_helper_10_17[2] + cols_helper_10_17[3] + cols_helper_10_17[4]
   + cols_helper_10_17[5] + cols_helper_10_17[6] + cols_helper_10_17[7]
   + cols_helper_10_17[8] + cols_helper_10_17[9] + cols_helper_10_17[10]
   + cols_helper_10_17[11] + cols_helper_10_17[12] + cols_helper_10_17[13]
   + cols_helper_10_17[14] + cols_helper_10_17[15] + cols_helper_10_17[16]
   <= 1
 cols_10_17: cols_10_17 - 2 cols_helper_10_17[0] - 4 cols_helper_10_17[1]
   - 8 cols_helper_10_17[2] - 16 cols_helper_10_17[3]
   - 32 cols_helper_10_17[4] - 64 cols_helper_10_17[5]
   - 128 cols_helper_10_17[6] - 256 cols_helper_10_17[7]
   - 512 cols_helper_10_17[8] - 1024 cols_helper_10_17[9]
   - 2048 cols_helper_10_17[10] - 4096 cols_helper_10_17[11]
   - 8192 cols_helper_10_17[12] - 16384 cols_helper_10_17[13]
   - 32768 cols_helper_10_17[14] - 65536 cols_helper_10_17[15]
   - 131072 cols_helper_10_17[16] = 0
 cols_acc_10_17: - 435 frac[10,17] + cols_10_17 >= 0
 accuracy_10_17: mem[10,17] - 0.01171875 cols_10_17 = 0
 cols_helper_11_28: cols_helper_11_28[0] + cols_helper_11_28[1]
   + cols_helper_11_28[2] + cols_helper_11_28[3] + cols_helper_11_28[4]
   + cols_helper_11_28[5] + cols_helper_11_28[6] + cols_helper_11_28[7]
   + cols_helper_11_28[8] + cols_helper_11_28[9] + cols_helper_11_28[10]
   + cols_helper_11_28[11] + cols_helper_11_28[12] + cols_helper_11_28[13]
   + cols_helper_11_28[14] + cols_helper_11_28[15] + cols_helper_11_28[16]
   <= 1
 cols_11_28: cols_11_28 - 2 cols_helper_11_28[0] - 4 cols_helper_11_28[1]
   - 8 cols_helper_11_28[2] - 16 cols_helper_11_28[3]
   - 32 cols_helper_11_28[4] - 64 cols_helper_11_28[5]
   - 128 cols_helper_11_28[6] - 256 cols_helper_11_28[7]
   - 512 cols_helper_11_28[8] - 1024 cols_helper_11_28[9]
   - 2048 cols_helper_11_28[10] - 4096 cols_helper_11_28[11]
   - 8192 cols_helper_11_28[12] - 16384 cols_helper_11_28[13]
   - 32768 cols_helper_11_28[14] - 65536 cols_helper_11_28[15]
   - 131072 cols_helper_11_28[16] = 0
 cols_acc_11_28: - 435 frac[11,28] + cols_11_28 >= 0
 accuracy_11_28: mem[11,28] - 0.01171875 cols_11_28 = 0
 cols_helper_8_71: cols_helper_8_71[0] + cols_helper_8_71[1]
   + cols_helper_8_71[2] + cols_helper_8_71[3] + cols_helper_8_71[4]
   + cols_helper_8_71[5] + cols_helper_8_71[6] + cols_helper_8_71[7]
   + cols_helper_8_71[8] + cols_helper_8_71[9] + cols_helper_8_71[10]
   + cols_helper_8_71[11] + cols_helper_8_71[12] + cols_helper_8_71[13]
   + cols_helper_8_71[14] + cols_helper_8_71[15] + cols_helper_8_71[16]
   <= 1
 cols_8_71: cols_8_71 - 2 cols_helper_8_71[0] - 4 cols_helper_8_71[1]
   - 8 cols_helper_8_71[2] - 16 cols_helper_8_71[3]
   - 32 cols_helper_8_71[4] - 64 cols_helper_8_71[5]
   - 128 cols_helper_8_71[6] - 256 cols_helper_8_71[7]
   - 512 cols_helper_8_71[8] - 1024 cols_helper_8_71[9]
   - 2048 cols_helper_8_71[10] - 4096 cols_helper_8_71[11]
   - 8192 cols_helper_8_71[12] - 16384 cols_helper_8_71[13]
   - 32768 cols_helper_8_71[14] - 65536 cols_helper_8_71[15]
   - 131072 cols_helper_8_71[16] = 0
 cols_acc_8_71: - 435 frac[8,71] + cols_8_71 >= 0
 accuracy_8_71: mem[8,71] - 0.01171875 cols_8_71 = 0
 accuracy_2_5: - 5.09765625 frac[2,5] + mem[2,5] = 0
 accuracy_1_56: - 5.09765625 frac[1,56] + mem[1,56] = 0
 cols_helper_10_69: cols_helper_10_69[0] + cols_helper_10_69[1]
   + cols_helper_10_69[2] + cols_helper_10_69[3] + cols_helper_10_69[4]
   + cols_helper_10_69[5] + cols_helper_10_69[6] + cols_helper_10_69[7]
   + cols_helper_10_69[8] + cols_helper_10_69[9] + cols_helper_10_69[10]
   + cols_helper_10_69[11] + cols_helper_10_69[12] + cols_helper_10_69[13]
   + cols_helper_10_69[14] + cols_helper_10_69[15] + cols_helper_10_69[16]
   <= 1
 cols_10_69: cols_10_69 - 2 cols_helper_10_69[0] - 4 cols_helper_10_69[1]
   - 8 cols_helper_10_69[2] - 16 cols_helper_10_69[3]
   - 32 cols_helper_10_69[4] - 64 cols_helper_10_69[5]
   - 128 cols_helper_10_69[6] - 256 cols_helper_10_69[7]
   - 512 cols_helper_10_69[8] - 1024 cols_helper_10_69[9]
   - 2048 cols_helper_10_69[10] - 4096 cols_helper_10_69[11]
   - 8192 cols_helper_10_69[12] - 16384 cols_helper_10_69[13]
   - 32768 cols_helper_10_69[14] - 65536 cols_helper_10_69[15]
   - 131072 cols_helper_10_69[16] = 0
 cols_acc_10_69: - 435 frac[10,69] + cols_10_69 >= 0
 accuracy_10_69: mem[10,69] - 0.01171875 cols_10_69 = 0
 cols_helper_11_67: cols_helper_11_67[0] + cols_helper_11_67[1]
   + cols_helper_11_67[2] + cols_helper_11_67[3] + cols_helper_11_67[4]
   + cols_helper_11_67[5] + cols_helper_11_67[6] + cols_helper_11_67[7]
   + cols_helper_11_67[8] + cols_helper_11_67[9] + cols_helper_11_67[10]
   + cols_helper_11_67[11] + cols_helper_11_67[12] + cols_helper_11_67[13]
   + cols_helper_11_67[14] + cols_helper_11_67[15] + cols_helper_11_67[16]
   <= 1
 cols_11_67: cols_11_67 - 2 cols_helper_11_67[0] - 4 cols_helper_11_67[1]
   - 8 cols_helper_11_67[2] - 16 cols_helper_11_67[3]
   - 32 cols_helper_11_67[4] - 64 cols_helper_11_67[5]
   - 128 cols_helper_11_67[6] - 256 cols_helper_11_67[7]
   - 512 cols_helper_11_67[8] - 1024 cols_helper_11_67[9]
   - 2048 cols_helper_11_67[10] - 4096 cols_helper_11_67[11]
   - 8192 cols_helper_11_67[12] - 16384 cols_helper_11_67[13]
   - 32768 cols_helper_11_67[14] - 65536 cols_helper_11_67[15]
   - 131072 cols_helper_11_67[16] = 0
 cols_acc_11_67: - 435 frac[11,67] + cols_11_67 >= 0
 accuracy_11_67: mem[11,67] - 0.01171875 cols_11_67 = 0
 accuracy_6_25: - 5.09765625 frac[6,25] + mem[6,25] = 0
 accuracy_4_57: - 5.09765625 frac[4,57] + mem[4,57] = 0
 cols_helper_8_60: cols_helper_8_60[0] + cols_helper_8_60[1]
   + cols_helper_8_60[2] + cols_helper_8_60[3] + cols_helper_8_60[4]
   + cols_helper_8_60[5] + cols_helper_8_60[6] + cols_helper_8_60[7]
   + cols_helper_8_60[8] + cols_helper_8_60[9] + cols_helper_8_60[10]
   + cols_helper_8_60[11] + cols_helper_8_60[12] + cols_helper_8_60[13]
   + cols_helper_8_60[14] + cols_helper_8_60[15] + cols_helper_8_60[16]
   <= 1
 cols_8_60: cols_8_60 - 2 cols_helper_8_60[0] - 4 cols_helper_8_60[1]
   - 8 cols_helper_8_60[2] - 16 cols_helper_8_60[3]
   - 32 cols_helper_8_60[4] - 64 cols_helper_8_60[5]
   - 128 cols_helper_8_60[6] - 256 cols_helper_8_60[7]
   - 512 cols_helper_8_60[8] - 1024 cols_helper_8_60[9]
   - 2048 cols_helper_8_60[10] - 4096 cols_helper_8_60[11]
   - 8192 cols_helper_8_60[12] - 16384 cols_helper_8_60[13]
   - 32768 cols_helper_8_60[14] - 65536 cols_helper_8_60[15]
   - 131072 cols_helper_8_60[16] = 0
 cols_acc_8_60: - 435 frac[8,60] + cols_8_60 >= 0
 accuracy_8_60: mem[8,60] - 0.01171875 cols_8_60 = 0
 cols_helper_9_57: cols_helper_9_57[0] + cols_helper_9_57[1]
   + cols_helper_9_57[2] + cols_helper_9_57[3] + cols_helper_9_57[4]
   + cols_helper_9_57[5] + cols_helper_9_57[6] + cols_helper_9_57[7]
   + cols_helper_9_57[8] + cols_helper_9_57[9] + cols_helper_9_57[10]
   + cols_helper_9_57[11] + cols_helper_9_57[12] + cols_helper_9_57[13]
   + cols_helper_9_57[14] + cols_helper_9_57[15] + cols_helper_9_57[16]
   <= 1
 cols_9_57: cols_9_57 - 2 cols_helper_9_57[0] - 4 cols_helper_9_57[1]
   - 8 cols_helper_9_57[2] - 16 cols_helper_9_57[3]
   - 32 cols_helper_9_57[4] - 64 cols_helper_9_57[5]
   - 128 cols_helper_9_57[6] - 256 cols_helper_9_57[7]
   - 512 cols_helper_9_57[8] - 1024 cols_helper_9_57[9]
   - 2048 cols_helper_9_57[10] - 4096 cols_helper_9_57[11]
   - 8192 cols_helper_9_57[12] - 16384 cols_helper_9_57[13]
   - 32768 cols_helper_9_57[14] - 65536 cols_helper_9_57[15]
   - 131072 cols_helper_9_57[16] = 0
 cols_acc_9_57: - 435 frac[9,57] + cols_9_57 >= 0
 accuracy_9_57: mem[9,57] - 0.01171875 cols_9_57 = 0
 accuracy_6_51: - 5.09765625 frac[6,51] + mem[6,51] = 0
 cols_helper_10_62: cols_helper_10_62[0] + cols_helper_10_62[1]
   + cols_helper_10_62[2] + cols_helper_10_62[3] + cols_helper_10_62[4]
   + cols_helper_10_62[5] + cols_helper_10_62[6] + cols_helper_10_62[7]
   + cols_helper_10_62[8] + cols_helper_10_62[9] + cols_helper_10_62[10]
   + cols_helper_10_62[11] + cols_helper_10_62[12] + cols_helper_10_62[13]
   + cols_helper_10_62[14] + cols_helper_10_62[15] + cols_helper_10_62[16]
   <= 1
 cols_10_62: cols_10_62 - 2 cols_helper_10_62[0] - 4 cols_helper_10_62[1]
   - 8 cols_helper_10_62[2] - 16 cols_helper_10_62[3]
   - 32 cols_helper_10_62[4] - 64 cols_helper_10_62[5]
   - 128 cols_helper_10_62[6] - 256 cols_helper_10_62[7]
   - 512 cols_helper_10_62[8] - 1024 cols_helper_10_62[9]
   - 2048 cols_helper_10_62[10] - 4096 cols_helper_10_62[11]
   - 8192 cols_helper_10_62[12] - 16384 cols_helper_10_62[13]
   - 32768 cols_helper_10_62[14] - 65536 cols_helper_10_62[15]
   - 131072 cols_helper_10_62[16] = 0
 cols_acc_10_62: - 435 frac[10,62] + cols_10_62 >= 0
 accuracy_10_62: mem[10,62] - 0.01171875 cols_10_62 = 0
 cols_helper_8_2: cols_helper_8_2[0] + cols_helper_8_2[1]
   + cols_helper_8_2[2] + cols_helper_8_2[3] + cols_helper_8_2[4]
   + cols_helper_8_2[5] + cols_helper_8_2[6] + cols_helper_8_2[7]
   + cols_helper_8_2[8] + cols_helper_8_2[9] + cols_helper_8_2[10]
   + cols_helper_8_2[11] + cols_helper_8_2[12] + cols_helper_8_2[13]
   + cols_helper_8_2[14] + cols_helper_8_2[15] + cols_helper_8_2[16] <= 1
 cols_8_2: cols_8_2 - 2 cols_helper_8_2[0] - 4 cols_helper_8_2[1]
   - 8 cols_helper_8_2[2] - 16 cols_helper_8_2[3] - 32 cols_helper_8_2[4]
   - 64 cols_helper_8_2[5] - 128 cols_helper_8_2[6]
   - 256 cols_helper_8_2[7] - 512 cols_helper_8_2[8]
   - 1024 cols_helper_8_2[9] - 2048 cols_helper_8_2[10]
   - 4096 cols_helper_8_2[11] - 8192 cols_helper_8_2[12]
   - 16384 cols_helper_8_2[13] - 32768 cols_helper_8_2[14]
   - 65536 cols_helper_8_2[15] - 131072 cols_helper_8_2[16] = 0
 cols_acc_8_2: - 435 frac[8,2] + cols_8_2 >= 0
 accuracy_8_2: mem[8,2] - 0.01171875 cols_8_2 = 0
 cols_helper_11_21: cols_helper_11_21[0] + cols_helper_11_21[1]
   + cols_helper_11_21[2] + cols_helper_11_21[3] + cols_helper_11_21[4]
   + cols_helper_11_21[5] + cols_helper_11_21[6] + cols_helper_11_21[7]
   + cols_helper_11_21[8] + cols_helper_11_21[9] + cols_helper_11_21[10]
   + cols_helper_11_21[11] + cols_helper_11_21[12] + cols_helper_11_21[13]
   + cols_helper_11_21[14] + cols_helper_11_21[15] + cols_helper_11_21[16]
   <= 1
 cols_11_21: cols_11_21 - 2 cols_helper_11_21[0] - 4 cols_helper_11_21[1]
   - 8 cols_helper_11_21[2] - 16 cols_helper_11_21[3]
   - 32 cols_helper_11_21[4] - 64 cols_helper_11_21[5]
   - 128 cols_helper_11_21[6] - 256 cols_helper_11_21[7]
   - 512 cols_helper_11_21[8] - 1024 cols_helper_11_21[9]
   - 2048 cols_helper_11_21[10] - 4096 cols_helper_11_21[11]
   - 8192 cols_helper_11_21[12] - 16384 cols_helper_11_21[13]
   - 32768 cols_helper_11_21[14] - 65536 cols_helper_11_21[15]
   - 131072 cols_helper_11_21[16] = 0
 cols_acc_11_21: - 435 frac[11,21] + cols_11_21 >= 0
 accuracy_11_21: mem[11,21] - 0.01171875 cols_11_21 = 0
 accuracy_0_30: - 5.09765625 frac[0,30] + mem[0,30] = 0
 accuracy_2_54: - 5.09765625 frac[2,54] + mem[2,54] = 0
 accuracy_3_39: - 5.09765625 frac[3,39] + mem[3,39] = 0
 cols_helper_8_35: cols_helper_8_35[0] + cols_helper_8_35[1]
   + cols_helper_8_35[2] + cols_helper_8_35[3] + cols_helper_8_35[4]
   + cols_helper_8_35[5] + cols_helper_8_35[6] + cols_helper_8_35[7]
   + cols_helper_8_35[8] + cols_helper_8_35[9] + cols_helper_8_35[10]
   + cols_helper_8_35[11] + cols_helper_8_35[12] + cols_helper_8_35[13]
   + cols_helper_8_35[14] + cols_helper_8_35[15] + cols_helper_8_35[16]
   <= 1
 cols_8_35: cols_8_35 - 2 cols_helper_8_35[0] - 4 cols_helper_8_35[1]
   - 8 cols_helper_8_35[2] - 16 cols_helper_8_35[3]
   - 32 cols_helper_8_35[4] - 64 cols_helper_8_35[5]
   - 128 cols_helper_8_35[6] - 256 cols_helper_8_35[7]
   - 512 cols_helper_8_35[8] - 1024 cols_helper_8_35[9]
   - 2048 cols_helper_8_35[10] - 4096 cols_helper_8_35[11]
   - 8192 cols_helper_8_35[12] - 16384 cols_helper_8_35[13]
   - 32768 cols_helper_8_35[14] - 65536 cols_helper_8_35[15]
   - 131072 cols_helper_8_35[16] = 0
 cols_acc_8_35: - 435 frac[8,35] + cols_8_35 >= 0
 accuracy_8_35: mem[8,35] - 0.01171875 cols_8_35 = 0
 accuracy_7_21: - 5.09765625 frac[7,21] + mem[7,21] = 0
 accuracy_6_42: - 5.09765625 frac[6,42] + mem[6,42] = 0
 cols_helper_10_57: cols_helper_10_57[0] + cols_helper_10_57[1]
   + cols_helper_10_57[2] + cols_helper_10_57[3] + cols_helper_10_57[4]
   + cols_helper_10_57[5] + cols_helper_10_57[6] + cols_helper_10_57[7]
   + cols_helper_10_57[8] + cols_helper_10_57[9] + cols_helper_10_57[10]
   + cols_helper_10_57[11] + cols_helper_10_57[12] + cols_helper_10_57[13]
   + cols_helper_10_57[14] + cols_helper_10_57[15] + cols_helper_10_57[16]
   <= 1
 cols_10_57: cols_10_57 - 2 cols_helper_10_57[0] - 4 cols_helper_10_57[1]
   - 8 cols_helper_10_57[2] - 16 cols_helper_10_57[3]
   - 32 cols_helper_10_57[4] - 64 cols_helper_10_57[5]
   - 128 cols_helper_10_57[6] - 256 cols_helper_10_57[7]
   - 512 cols_helper_10_57[8] - 1024 cols_helper_10_57[9]
   - 2048 cols_helper_10_57[10] - 4096 cols_helper_10_57[11]
   - 8192 cols_helper_10_57[12] - 16384 cols_helper_10_57[13]
   - 32768 cols_helper_10_57[14] - 65536 cols_helper_10_57[15]
   - 131072 cols_helper_10_57[16] = 0
 cols_acc_10_57: - 435 frac[10,57] + cols_10_57 >= 0
 accuracy_10_57: mem[10,57] - 0.01171875 cols_10_57 = 0
 accuracy_2_74: - 5.09765625 frac[2,74] + mem[2,74] = 0
 cols_helper_8_9: cols_helper_8_9[0] + cols_helper_8_9[1]
   + cols_helper_8_9[2] + cols_helper_8_9[3] + cols_helper_8_9[4]
   + cols_helper_8_9[5] + cols_helper_8_9[6] + cols_helper_8_9[7]
   + cols_helper_8_9[8] + cols_helper_8_9[9] + cols_helper_8_9[10]
   + cols_helper_8_9[11] + cols_helper_8_9[12] + cols_helper_8_9[13]
   + cols_helper_8_9[14] + cols_helper_8_9[15] + cols_helper_8_9[16] <= 1
 cols_8_9: cols_8_9 - 2 cols_helper_8_9[0] - 4 cols_helper_8_9[1]
   - 8 cols_helper_8_9[2] - 16 cols_helper_8_9[3] - 32 cols_helper_8_9[4]
   - 64 cols_helper_8_9[5] - 128 cols_helper_8_9[6]
   - 256 cols_helper_8_9[7] - 512 cols_helper_8_9[8]
   - 1024 cols_helper_8_9[9] - 2048 cols_helper_8_9[10]
   - 4096 cols_helper_8_9[11] - 8192 cols_helper_8_9[12]
   - 16384 cols_helper_8_9[13] - 32768 cols_helper_8_9[14]
   - 65536 cols_helper_8_9[15] - 131072 cols_helper_8_9[16] = 0
 cols_acc_8_9: - 435 frac[8,9] + cols_8_9 >= 0
 accuracy_8_9: mem[8,9] - 0.01171875 cols_8_9 = 0
 accuracy_7_51: - 5.09765625 frac[7,51] + mem[7,51] = 0
 cols_helper_11_52: cols_helper_11_52[0] + cols_helper_11_52[1]
   + cols_helper_11_52[2] + cols_helper_11_52[3] + cols_helper_11_52[4]
   + cols_helper_11_52[5] + cols_helper_11_52[6] + cols_helper_11_52[7]
   + cols_helper_11_52[8] + cols_helper_11_52[9] + cols_helper_11_52[10]
   + cols_helper_11_52[11] + cols_helper_11_52[12] + cols_helper_11_52[13]
   + cols_helper_11_52[14] + cols_helper_11_52[15] + cols_helper_11_52[16]
   <= 1
 cols_11_52: cols_11_52 - 2 cols_helper_11_52[0] - 4 cols_helper_11_52[1]
   - 8 cols_helper_11_52[2] - 16 cols_helper_11_52[3]
   - 32 cols_helper_11_52[4] - 64 cols_helper_11_52[5]
   - 128 cols_helper_11_52[6] - 256 cols_helper_11_52[7]
   - 512 cols_helper_11_52[8] - 1024 cols_helper_11_52[9]
   - 2048 cols_helper_11_52[10] - 4096 cols_helper_11_52[11]
   - 8192 cols_helper_11_52[12] - 16384 cols_helper_11_52[13]
   - 32768 cols_helper_11_52[14] - 65536 cols_helper_11_52[15]
   - 131072 cols_helper_11_52[16] = 0
 cols_acc_11_52: - 435 frac[11,52] + cols_11_52 >= 0
 accuracy_11_52: mem[11,52] - 0.01171875 cols_11_52 = 0
 cols_helper_9_4: cols_helper_9_4[0] + cols_helper_9_4[1]
   + cols_helper_9_4[2] + cols_helper_9_4[3] + cols_helper_9_4[4]
   + cols_helper_9_4[5] + cols_helper_9_4[6] + cols_helper_9_4[7]
   + cols_helper_9_4[8] + cols_helper_9_4[9] + cols_helper_9_4[10]
   + cols_helper_9_4[11] + cols_helper_9_4[12] + cols_helper_9_4[13]
   + cols_helper_9_4[14] + cols_helper_9_4[15] + cols_helper_9_4[16] <= 1
 cols_9_4: cols_9_4 - 2 cols_helper_9_4[0] - 4 cols_helper_9_4[1]
   - 8 cols_helper_9_4[2] - 16 cols_helper_9_4[3] - 32 cols_helper_9_4[4]
   - 64 cols_helper_9_4[5] - 128 cols_helper_9_4[6]
   - 256 cols_helper_9_4[7] - 512 cols_helper_9_4[8]
   - 1024 cols_helper_9_4[9] - 2048 cols_helper_9_4[10]
   - 4096 cols_helper_9_4[11] - 8192 cols_helper_9_4[12]
   - 16384 cols_helper_9_4[13] - 32768 cols_helper_9_4[14]
   - 65536 cols_helper_9_4[15] - 131072 cols_helper_9_4[16] = 0
 cols_acc_9_4: - 435 frac[9,4] + cols_9_4 >= 0
 accuracy_9_4: mem[9,4] - 0.01171875 cols_9_4 = 0
 cols_helper_10_3: cols_helper_10_3[0] + cols_helper_10_3[1]
   + cols_helper_10_3[2] + cols_helper_10_3[3] + cols_helper_10_3[4]
   + cols_helper_10_3[5] + cols_helper_10_3[6] + cols_helper_10_3[7]
   + cols_helper_10_3[8] + cols_helper_10_3[9] + cols_helper_10_3[10]
   + cols_helper_10_3[11] + cols_helper_10_3[12] + cols_helper_10_3[13]
   + cols_helper_10_3[14] + cols_helper_10_3[15] + cols_helper_10_3[16]
   <= 1
 cols_10_3: cols_10_3 - 2 cols_helper_10_3[0] - 4 cols_helper_10_3[1]
   - 8 cols_helper_10_3[2] - 16 cols_helper_10_3[3]
   - 32 cols_helper_10_3[4] - 64 cols_helper_10_3[5]
   - 128 cols_helper_10_3[6] - 256 cols_helper_10_3[7]
   - 512 cols_helper_10_3[8] - 1024 cols_helper_10_3[9]
   - 2048 cols_helper_10_3[10] - 4096 cols_helper_10_3[11]
   - 8192 cols_helper_10_3[12] - 16384 cols_helper_10_3[13]
   - 32768 cols_helper_10_3[14] - 65536 cols_helper_10_3[15]
   - 131072 cols_helper_10_3[16] = 0
 cols_acc_10_3: - 435 frac[10,3] + cols_10_3 >= 0
 accuracy_10_3: mem[10,3] - 0.01171875 cols_10_3 = 0
 cols_helper_11_10: cols_helper_11_10[0] + cols_helper_11_10[1]
   + cols_helper_11_10[2] + cols_helper_11_10[3] + cols_helper_11_10[4]
   + cols_helper_11_10[5] + cols_helper_11_10[6] + cols_helper_11_10[7]
   + cols_helper_11_10[8] + cols_helper_11_10[9] + cols_helper_11_10[10]
   + cols_helper_11_10[11] + cols_helper_11_10[12] + cols_helper_11_10[13]
   + cols_helper_11_10[14] + cols_helper_11_10[15] + cols_helper_11_10[16]
   <= 1
 cols_11_10: cols_11_10 - 2 cols_helper_11_10[0] - 4 cols_helper_11_10[1]
   - 8 cols_helper_11_10[2] - 16 cols_helper_11_10[3]
   - 32 cols_helper_11_10[4] - 64 cols_helper_11_10[5]
   - 128 cols_helper_11_10[6] - 256 cols_helper_11_10[7]
   - 512 cols_helper_11_10[8] - 1024 cols_helper_11_10[9]
   - 2048 cols_helper_11_10[10] - 4096 cols_helper_11_10[11]
   - 8192 cols_helper_11_10[12] - 16384 cols_helper_11_10[13]
   - 32768 cols_helper_11_10[14] - 65536 cols_helper_11_10[15]
   - 131072 cols_helper_11_10[16] = 0
 cols_acc_11_10: - 435 frac[11,10] + cols_11_10 >= 0
 accuracy_11_10: mem[11,10] - 0.01171875 cols_11_10 = 0
 accuracy_0_21: - 5.09765625 frac[0,21] + mem[0,21] = 0
 accuracy_1_16: - 5.09765625 frac[1,16] + mem[1,16] = 0
 cols_helper_9_65: cols_helper_9_65[0] + cols_helper_9_65[1]
   + cols_helper_9_65[2] + cols_helper_9_65[3] + cols_helper_9_65[4]
   + cols_helper_9_65[5] + cols_helper_9_65[6] + cols_helper_9_65[7]
   + cols_helper_9_65[8] + cols_helper_9_65[9] + cols_helper_9_65[10]
   + cols_helper_9_65[11] + cols_helper_9_65[12] + cols_helper_9_65[13]
   + cols_helper_9_65[14] + cols_helper_9_65[15] + cols_helper_9_65[16]
   <= 1
 cols_9_65: cols_9_65 - 2 cols_helper_9_65[0] - 4 cols_helper_9_65[1]
   - 8 cols_helper_9_65[2] - 16 cols_helper_9_65[3]
   - 32 cols_helper_9_65[4] - 64 cols_helper_9_65[5]
   - 128 cols_helper_9_65[6] - 256 cols_helper_9_65[7]
   - 512 cols_helper_9_65[8] - 1024 cols_helper_9_65[9]
   - 2048 cols_helper_9_65[10] - 4096 cols_helper_9_65[11]
   - 8192 cols_helper_9_65[12] - 16384 cols_helper_9_65[13]
   - 32768 cols_helper_9_65[14] - 65536 cols_helper_9_65[15]
   - 131072 cols_helper_9_65[16] = 0
 cols_acc_9_65: - 435 frac[9,65] + cols_9_65 >= 0
 accuracy_9_65: mem[9,65] - 0.01171875 cols_9_65 = 0
 accuracy_0_71: - 5.09765625 frac[0,71] + mem[0,71] = 0
 accuracy_7_18: - 5.09765625 frac[7,18] + mem[7,18] = 0
 cols_helper_9_43: cols_helper_9_43[0] + cols_helper_9_43[1]
   + cols_helper_9_43[2] + cols_helper_9_43[3] + cols_helper_9_43[4]
   + cols_helper_9_43[5] + cols_helper_9_43[6] + cols_helper_9_43[7]
   + cols_helper_9_43[8] + cols_helper_9_43[9] + cols_helper_9_43[10]
   + cols_helper_9_43[11] + cols_helper_9_43[12] + cols_helper_9_43[13]
   + cols_helper_9_43[14] + cols_helper_9_43[15] + cols_helper_9_43[16]
   <= 1
 cols_9_43: cols_9_43 - 2 cols_helper_9_43[0] - 4 cols_helper_9_43[1]
   - 8 cols_helper_9_43[2] - 16 cols_helper_9_43[3]
   - 32 cols_helper_9_43[4] - 64 cols_helper_9_43[5]
   - 128 cols_helper_9_43[6] - 256 cols_helper_9_43[7]
   - 512 cols_helper_9_43[8] - 1024 cols_helper_9_43[9]
   - 2048 cols_helper_9_43[10] - 4096 cols_helper_9_43[11]
   - 8192 cols_helper_9_43[12] - 16384 cols_helper_9_43[13]
   - 32768 cols_helper_9_43[14] - 65536 cols_helper_9_43[15]
   - 131072 cols_helper_9_43[16] = 0
 cols_acc_9_43: - 435 frac[9,43] + cols_9_43 >= 0
 accuracy_9_43: mem[9,43] - 0.01171875 cols_9_43 = 0
 cols_helper_10_32: cols_helper_10_32[0] + cols_helper_10_32[1]
   + cols_helper_10_32[2] + cols_helper_10_32[3] + cols_helper_10_32[4]
   + cols_helper_10_32[5] + cols_helper_10_32[6] + cols_helper_10_32[7]
   + cols_helper_10_32[8] + cols_helper_10_32[9] + cols_helper_10_32[10]
   + cols_helper_10_32[11] + cols_helper_10_32[12] + cols_helper_10_32[13]
   + cols_helper_10_32[14] + cols_helper_10_32[15] + cols_helper_10_32[16]
   <= 1
 cols_10_32: cols_10_32 - 2 cols_helper_10_32[0] - 4 cols_helper_10_32[1]
   - 8 cols_helper_10_32[2] - 16 cols_helper_10_32[3]
   - 32 cols_helper_10_32[4] - 64 cols_helper_10_32[5]
   - 128 cols_helper_10_32[6] - 256 cols_helper_10_32[7]
   - 512 cols_helper_10_32[8] - 1024 cols_helper_10_32[9]
   - 2048 cols_helper_10_32[10] - 4096 cols_helper_10_32[11]
   - 8192 cols_helper_10_32[12] - 16384 cols_helper_10_32[13]
   - 32768 cols_helper_10_32[14] - 65536 cols_helper_10_32[15]
   - 131072 cols_helper_10_32[16] = 0
 cols_acc_10_32: - 435 frac[10,32] + cols_10_32 >= 0
 accuracy_10_32: mem[10,32] - 0.01171875 cols_10_32 = 0
 accuracy_2_69: - 5.09765625 frac[2,69] + mem[2,69] = 0
 cols_helper_8_48: cols_helper_8_48[0] + cols_helper_8_48[1]
   + cols_helper_8_48[2] + cols_helper_8_48[3] + cols_helper_8_48[4]
   + cols_helper_8_48[5] + cols_helper_8_48[6] + cols_helper_8_48[7]
   + cols_helper_8_48[8] + cols_helper_8_48[9] + cols_helper_8_48[10]
   + cols_helper_8_48[11] + cols_helper_8_48[12] + cols_helper_8_48[13]
   + cols_helper_8_48[14] + cols_helper_8_48[15] + cols_helper_8_48[16]
   <= 1
 cols_8_48: cols_8_48 - 2 cols_helper_8_48[0] - 4 cols_helper_8_48[1]
   - 8 cols_helper_8_48[2] - 16 cols_helper_8_48[3]
   - 32 cols_helper_8_48[4] - 64 cols_helper_8_48[5]
   - 128 cols_helper_8_48[6] - 256 cols_helper_8_48[7]
   - 512 cols_helper_8_48[8] - 1024 cols_helper_8_48[9]
   - 2048 cols_helper_8_48[10] - 4096 cols_helper_8_48[11]
   - 8192 cols_helper_8_48[12] - 16384 cols_helper_8_48[13]
   - 32768 cols_helper_8_48[14] - 65536 cols_helper_8_48[15]
   - 131072 cols_helper_8_48[16] = 0
 cols_acc_8_48: - 435 frac[8,48] + cols_8_48 >= 0
 accuracy_8_48: mem[8,48] - 0.01171875 cols_8_48 = 0
 cols_helper_11_45: cols_helper_11_45[0] + cols_helper_11_45[1]
   + cols_helper_11_45[2] + cols_helper_11_45[3] + cols_helper_11_45[4]
   + cols_helper_11_45[5] + cols_helper_11_45[6] + cols_helper_11_45[7]
   + cols_helper_11_45[8] + cols_helper_11_45[9] + cols_helper_11_45[10]
   + cols_helper_11_45[11] + cols_helper_11_45[12] + cols_helper_11_45[13]
   + cols_helper_11_45[14] + cols_helper_11_45[15] + cols_helper_11_45[16]
   <= 1
 cols_11_45: cols_11_45 - 2 cols_helper_11_45[0] - 4 cols_helper_11_45[1]
   - 8 cols_helper_11_45[2] - 16 cols_helper_11_45[3]
   - 32 cols_helper_11_45[4] - 64 cols_helper_11_45[5]
   - 128 cols_helper_11_45[6] - 256 cols_helper_11_45[7]
   - 512 cols_helper_11_45[8] - 1024 cols_helper_11_45[9]
   - 2048 cols_helper_11_45[10] - 4096 cols_helper_11_45[11]
   - 8192 cols_helper_11_45[12] - 16384 cols_helper_11_45[13]
   - 32768 cols_helper_11_45[14] - 65536 cols_helper_11_45[15]
   - 131072 cols_helper_11_45[16] = 0
 cols_acc_11_45: - 435 frac[11,45] + cols_11_45 >= 0
 accuracy_11_45: mem[11,45] - 0.01171875 cols_11_45 = 0
 cols_helper_9_13: cols_helper_9_13[0] + cols_helper_9_13[1]
   + cols_helper_9_13[2] + cols_helper_9_13[3] + cols_helper_9_13[4]
   + cols_helper_9_13[5] + cols_helper_9_13[6] + cols_helper_9_13[7]
   + cols_helper_9_13[8] + cols_helper_9_13[9] + cols_helper_9_13[10]
   + cols_helper_9_13[11] + cols_helper_9_13[12] + cols_helper_9_13[13]
   + cols_helper_9_13[14] + cols_helper_9_13[15] + cols_helper_9_13[16]
   <= 1
 cols_9_13: cols_9_13 - 2 cols_helper_9_13[0] - 4 cols_helper_9_13[1]
   - 8 cols_helper_9_13[2] - 16 cols_helper_9_13[3]
   - 32 cols_helper_9_13[4] - 64 cols_helper_9_13[5]
   - 128 cols_helper_9_13[6] - 256 cols_helper_9_13[7]
   - 512 cols_helper_9_13[8] - 1024 cols_helper_9_13[9]
   - 2048 cols_helper_9_13[10] - 4096 cols_helper_9_13[11]
   - 8192 cols_helper_9_13[12] - 16384 cols_helper_9_13[13]
   - 32768 cols_helper_9_13[14] - 65536 cols_helper_9_13[15]
   - 131072 cols_helper_9_13[16] = 0
 cols_acc_9_13: - 435 frac[9,13] + cols_9_13 >= 0
 accuracy_9_13: mem[9,13] - 0.01171875 cols_9_13 = 0
 accuracy_6_71: - 5.09765625 frac[6,71] + mem[6,71] = 0
 cols_helper_10_10: cols_helper_10_10[0] + cols_helper_10_10[1]
   + cols_helper_10_10[2] + cols_helper_10_10[3] + cols_helper_10_10[4]
   + cols_helper_10_10[5] + cols_helper_10_10[6] + cols_helper_10_10[7]
   + cols_helper_10_10[8] + cols_helper_10_10[9] + cols_helper_10_10[10]
   + cols_helper_10_10[11] + cols_helper_10_10[12] + cols_helper_10_10[13]
   + cols_helper_10_10[14] + cols_helper_10_10[15] + cols_helper_10_10[16]
   <= 1
 cols_10_10: cols_10_10 - 2 cols_helper_10_10[0] - 4 cols_helper_10_10[1]
   - 8 cols_helper_10_10[2] - 16 cols_helper_10_10[3]
   - 32 cols_helper_10_10[4] - 64 cols_helper_10_10[5]
   - 128 cols_helper_10_10[6] - 256 cols_helper_10_10[7]
   - 512 cols_helper_10_10[8] - 1024 cols_helper_10_10[9]
   - 2048 cols_helper_10_10[10] - 4096 cols_helper_10_10[11]
   - 8192 cols_helper_10_10[12] - 16384 cols_helper_10_10[13]
   - 32768 cols_helper_10_10[14] - 65536 cols_helper_10_10[15]
   - 131072 cols_helper_10_10[16] = 0
 cols_acc_10_10: - 435 frac[10,10] + cols_10_10 >= 0
 accuracy_10_10: mem[10,10] - 0.01171875 cols_10_10 = 0
 cols_helper_8_22: cols_helper_8_22[0] + cols_helper_8_22[1]
   + cols_helper_8_22[2] + cols_helper_8_22[3] + cols_helper_8_22[4]
   + cols_helper_8_22[5] + cols_helper_8_22[6] + cols_helper_8_22[7]
   + cols_helper_8_22[8] + cols_helper_8_22[9] + cols_helper_8_22[10]
   + cols_helper_8_22[11] + cols_helper_8_22[12] + cols_helper_8_22[13]
   + cols_helper_8_22[14] + cols_helper_8_22[15] + cols_helper_8_22[16]
   <= 1
 cols_8_22: cols_8_22 - 2 cols_helper_8_22[0] - 4 cols_helper_8_22[1]
   - 8 cols_helper_8_22[2] - 16 cols_helper_8_22[3]
   - 32 cols_helper_8_22[4] - 64 cols_helper_8_22[5]
   - 128 cols_helper_8_22[6] - 256 cols_helper_8_22[7]
   - 512 cols_helper_8_22[8] - 1024 cols_helper_8_22[9]
   - 2048 cols_helper_8_22[10] - 4096 cols_helper_8_22[11]
   - 8192 cols_helper_8_22[12] - 16384 cols_helper_8_22[13]
   - 32768 cols_helper_8_22[14] - 65536 cols_helper_8_22[15]
   - 131072 cols_helper_8_22[16] = 0
 cols_acc_8_22: - 435 frac[8,22] + cols_8_22 >= 0
 accuracy_8_22: mem[8,22] - 0.01171875 cols_8_22 = 0
 cols_helper_9_23: cols_helper_9_23[0] + cols_helper_9_23[1]
   + cols_helper_9_23[2] + cols_helper_9_23[3] + cols_helper_9_23[4]
   + cols_helper_9_23[5] + cols_helper_9_23[6] + cols_helper_9_23[7]
   + cols_helper_9_23[8] + cols_helper_9_23[9] + cols_helper_9_23[10]
   + cols_helper_9_23[11] + cols_helper_9_23[12] + cols_helper_9_23[13]
   + cols_helper_9_23[14] + cols_helper_9_23[15] + cols_helper_9_23[16]
   <= 1
 cols_9_23: cols_9_23 - 2 cols_helper_9_23[0] - 4 cols_helper_9_23[1]
   - 8 cols_helper_9_23[2] - 16 cols_helper_9_23[3]
   - 32 cols_helper_9_23[4] - 64 cols_helper_9_23[5]
   - 128 cols_helper_9_23[6] - 256 cols_helper_9_23[7]
   - 512 cols_helper_9_23[8] - 1024 cols_helper_9_23[9]
   - 2048 cols_helper_9_23[10] - 4096 cols_helper_9_23[11]
   - 8192 cols_helper_9_23[12] - 16384 cols_helper_9_23[13]
   - 32768 cols_helper_9_23[14] - 65536 cols_helper_9_23[15]
   - 131072 cols_helper_9_23[16] = 0
 cols_acc_9_23: - 435 frac[9,23] + cols_9_23 >= 0
 accuracy_9_23: mem[9,23] - 0.01171875 cols_9_23 = 0
 cols_helper_11_25: cols_helper_11_25[0] + cols_helper_11_25[1]
   + cols_helper_11_25[2] + cols_helper_11_25[3] + cols_helper_11_25[4]
   + cols_helper_11_25[5] + cols_helper_11_25[6] + cols_helper_11_25[7]
   + cols_helper_11_25[8] + cols_helper_11_25[9] + cols_helper_11_25[10]
   + cols_helper_11_25[11] + cols_helper_11_25[12] + cols_helper_11_25[13]
   + cols_helper_11_25[14] + cols_helper_11_25[15] + cols_helper_11_25[16]
   <= 1
 cols_11_25: cols_11_25 - 2 cols_helper_11_25[0] - 4 cols_helper_11_25[1]
   - 8 cols_helper_11_25[2] - 16 cols_helper_11_25[3]
   - 32 cols_helper_11_25[4] - 64 cols_helper_11_25[5]
   - 128 cols_helper_11_25[6] - 256 cols_helper_11_25[7]
   - 512 cols_helper_11_25[8] - 1024 cols_helper_11_25[9]
   - 2048 cols_helper_11_25[10] - 4096 cols_helper_11_25[11]
   - 8192 cols_helper_11_25[12] - 16384 cols_helper_11_25[13]
   - 32768 cols_helper_11_25[14] - 65536 cols_helper_11_25[15]
   - 131072 cols_helper_11_25[16] = 0
 cols_acc_11_25: - 435 frac[11,25] + cols_11_25 >= 0
 accuracy_11_25: mem[11,25] - 0.01171875 cols_11_25 = 0
 accuracy_0_2: - 5.09765625 frac[0,2] + mem[0,2] = 0
 cols_helper_8_66: cols_helper_8_66[0] + cols_helper_8_66[1]
   + cols_helper_8_66[2] + cols_helper_8_66[3] + cols_helper_8_66[4]
   + cols_helper_8_66[5] + cols_helper_8_66[6] + cols_helper_8_66[7]
   + cols_helper_8_66[8] + cols_helper_8_66[9] + cols_helper_8_66[10]
   + cols_helper_8_66[11] + cols_helper_8_66[12] + cols_helper_8_66[13]
   + cols_helper_8_66[14] + cols_helper_8_66[15] + cols_helper_8_66[16]
   <= 1
 cols_8_66: cols_8_66 - 2 cols_helper_8_66[0] - 4 cols_helper_8_66[1]
   - 8 cols_helper_8_66[2] - 16 cols_helper_8_66[3]
   - 32 cols_helper_8_66[4] - 64 cols_helper_8_66[5]
   - 128 cols_helper_8_66[6] - 256 cols_helper_8_66[7]
   - 512 cols_helper_8_66[8] - 1024 cols_helper_8_66[9]
   - 2048 cols_helper_8_66[10] - 4096 cols_helper_8_66[11]
   - 8192 cols_helper_8_66[12] - 16384 cols_helper_8_66[13]
   - 32768 cols_helper_8_66[14] - 65536 cols_helper_8_66[15]
   - 131072 cols_helper_8_66[16] = 0
 cols_acc_8_66: - 435 frac[8,66] + cols_8_66 >= 0
 accuracy_8_66: mem[8,66] - 0.01171875 cols_8_66 = 0
 accuracy_3_21: - 5.09765625 frac[3,21] + mem[3,21] = 0
 cols_helper_9_74: cols_helper_9_74[0] + cols_helper_9_74[1]
   + cols_helper_9_74[2] + cols_helper_9_74[3] + cols_helper_9_74[4]
   + cols_helper_9_74[5] + cols_helper_9_74[6] + cols_helper_9_74[7]
   + cols_helper_9_74[8] + cols_helper_9_74[9] + cols_helper_9_74[10]
   + cols_helper_9_74[11] + cols_helper_9_74[12] + cols_helper_9_74[13]
   + cols_helper_9_74[14] + cols_helper_9_74[15] + cols_helper_9_74[16]
   <= 1
 cols_9_74: cols_9_74 - 2 cols_helper_9_74[0] - 4 cols_helper_9_74[1]
   - 8 cols_helper_9_74[2] - 16 cols_helper_9_74[3]
   - 32 cols_helper_9_74[4] - 64 cols_helper_9_74[5]
   - 128 cols_helper_9_74[6] - 256 cols_helper_9_74[7]
   - 512 cols_helper_9_74[8] - 1024 cols_helper_9_74[9]
   - 2048 cols_helper_9_74[10] - 4096 cols_helper_9_74[11]
   - 8192 cols_helper_9_74[12] - 16384 cols_helper_9_74[13]
   - 32768 cols_helper_9_74[14] - 65536 cols_helper_9_74[15]
   - 131072 cols_helper_9_74[16] = 0
 cols_acc_9_74: - 435 frac[9,74] + cols_9_74 >= 0
 accuracy_9_74: mem[9,74] - 0.01171875 cols_9_74 = 0
 accuracy_2_34: - 5.09765625 frac[2,34] + mem[2,34] = 0
 cols_helper_9_44: cols_helper_9_44[0] + cols_helper_9_44[1]
   + cols_helper_9_44[2] + cols_helper_9_44[3] + cols_helper_9_44[4]
   + cols_helper_9_44[5] + cols_helper_9_44[6] + cols_helper_9_44[7]
   + cols_helper_9_44[8] + cols_helper_9_44[9] + cols_helper_9_44[10]
   + cols_helper_9_44[11] + cols_helper_9_44[12] + cols_helper_9_44[13]
   + cols_helper_9_44[14] + cols_helper_9_44[15] + cols_helper_9_44[16]
   <= 1
 cols_9_44: cols_9_44 - 2 cols_helper_9_44[0] - 4 cols_helper_9_44[1]
   - 8 cols_helper_9_44[2] - 16 cols_helper_9_44[3]
   - 32 cols_helper_9_44[4] - 64 cols_helper_9_44[5]
   - 128 cols_helper_9_44[6] - 256 cols_helper_9_44[7]
   - 512 cols_helper_9_44[8] - 1024 cols_helper_9_44[9]
   - 2048 cols_helper_9_44[10] - 4096 cols_helper_9_44[11]
   - 8192 cols_helper_9_44[12] - 16384 cols_helper_9_44[13]
   - 32768 cols_helper_9_44[14] - 65536 cols_helper_9_44[15]
   - 131072 cols_helper_9_44[16] = 0
 cols_acc_9_44: - 435 frac[9,44] + cols_9_44 >= 0
 accuracy_9_44: mem[9,44] - 0.01171875 cols_9_44 = 0
 cols_helper_8_55: cols_helper_8_55[0] + cols_helper_8_55[1]
   + cols_helper_8_55[2] + cols_helper_8_55[3] + cols_helper_8_55[4]
   + cols_helper_8_55[5] + cols_helper_8_55[6] + cols_helper_8_55[7]
   + cols_helper_8_55[8] + cols_helper_8_55[9] + cols_helper_8_55[10]
   + cols_helper_8_55[11] + cols_helper_8_55[12] + cols_helper_8_55[13]
   + cols_helper_8_55[14] + cols_helper_8_55[15] + cols_helper_8_55[16]
   <= 1
 cols_8_55: cols_8_55 - 2 cols_helper_8_55[0] - 4 cols_helper_8_55[1]
   - 8 cols_helper_8_55[2] - 16 cols_helper_8_55[3]
   - 32 cols_helper_8_55[4] - 64 cols_helper_8_55[5]
   - 128 cols_helper_8_55[6] - 256 cols_helper_8_55[7]
   - 512 cols_helper_8_55[8] - 1024 cols_helper_8_55[9]
   - 2048 cols_helper_8_55[10] - 4096 cols_helper_8_55[11]
   - 8192 cols_helper_8_55[12] - 16384 cols_helper_8_55[13]
   - 32768 cols_helper_8_55[14] - 65536 cols_helper_8_55[15]
   - 131072 cols_helper_8_55[16] = 0
 cols_acc_8_55: - 435 frac[8,55] + cols_8_55 >= 0
 accuracy_8_55: mem[8,55] - 0.01171875 cols_8_55 = 0
 accuracy_7_33: - 5.09765625 frac[7,33] + mem[7,33] = 0
 cols_helper_9_54: cols_helper_9_54[0] + cols_helper_9_54[1]
   + cols_helper_9_54[2] + cols_helper_9_54[3] + cols_helper_9_54[4]
   + cols_helper_9_54[5] + cols_helper_9_54[6] + cols_helper_9_54[7]
   + cols_helper_9_54[8] + cols_helper_9_54[9] + cols_helper_9_54[10]
   + cols_helper_9_54[11] + cols_helper_9_54[12] + cols_helper_9_54[13]
   + cols_helper_9_54[14] + cols_helper_9_54[15] + cols_helper_9_54[16]
   <= 1
 cols_9_54: cols_9_54 - 2 cols_helper_9_54[0] - 4 cols_helper_9_54[1]
   - 8 cols_helper_9_54[2] - 16 cols_helper_9_54[3]
   - 32 cols_helper_9_54[4] - 64 cols_helper_9_54[5]
   - 128 cols_helper_9_54[6] - 256 cols_helper_9_54[7]
   - 512 cols_helper_9_54[8] - 1024 cols_helper_9_54[9]
   - 2048 cols_helper_9_54[10] - 4096 cols_helper_9_54[11]
   - 8192 cols_helper_9_54[12] - 16384 cols_helper_9_54[13]
   - 32768 cols_helper_9_54[14] - 65536 cols_helper_9_54[15]
   - 131072 cols_helper_9_54[16] = 0
 cols_acc_9_54: - 435 frac[9,54] + cols_9_54 >= 0
 accuracy_9_54: mem[9,54] - 0.01171875 cols_9_54 = 0
 cols_helper_10_53: cols_helper_10_53[0] + cols_helper_10_53[1]
   + cols_helper_10_53[2] + cols_helper_10_53[3] + cols_helper_10_53[4]
   + cols_helper_10_53[5] + cols_helper_10_53[6] + cols_helper_10_53[7]
   + cols_helper_10_53[8] + cols_helper_10_53[9] + cols_helper_10_53[10]
   + cols_helper_10_53[11] + cols_helper_10_53[12] + cols_helper_10_53[13]
   + cols_helper_10_53[14] + cols_helper_10_53[15] + cols_helper_10_53[16]
   <= 1
 cols_10_53: cols_10_53 - 2 cols_helper_10_53[0] - 4 cols_helper_10_53[1]
   - 8 cols_helper_10_53[2] - 16 cols_helper_10_53[3]
   - 32 cols_helper_10_53[4] - 64 cols_helper_10_53[5]
   - 128 cols_helper_10_53[6] - 256 cols_helper_10_53[7]
   - 512 cols_helper_10_53[8] - 1024 cols_helper_10_53[9]
   - 2048 cols_helper_10_53[10] - 4096 cols_helper_10_53[11]
   - 8192 cols_helper_10_53[12] - 16384 cols_helper_10_53[13]
   - 32768 cols_helper_10_53[14] - 65536 cols_helper_10_53[15]
   - 131072 cols_helper_10_53[16] = 0
 cols_acc_10_53: - 435 frac[10,53] + cols_10_53 >= 0
 accuracy_10_53: mem[10,53] - 0.01171875 cols_10_53 = 0
 cols_helper_8_29: cols_helper_8_29[0] + cols_helper_8_29[1]
   + cols_helper_8_29[2] + cols_helper_8_29[3] + cols_helper_8_29[4]
   + cols_helper_8_29[5] + cols_helper_8_29[6] + cols_helper_8_29[7]
   + cols_helper_8_29[8] + cols_helper_8_29[9] + cols_helper_8_29[10]
   + cols_helper_8_29[11] + cols_helper_8_29[12] + cols_helper_8_29[13]
   + cols_helper_8_29[14] + cols_helper_8_29[15] + cols_helper_8_29[16]
   <= 1
 cols_8_29: cols_8_29 - 2 cols_helper_8_29[0] - 4 cols_helper_8_29[1]
   - 8 cols_helper_8_29[2] - 16 cols_helper_8_29[3]
   - 32 cols_helper_8_29[4] - 64 cols_helper_8_29[5]
   - 128 cols_helper_8_29[6] - 256 cols_helper_8_29[7]
   - 512 cols_helper_8_29[8] - 1024 cols_helper_8_29[9]
   - 2048 cols_helper_8_29[10] - 4096 cols_helper_8_29[11]
   - 8192 cols_helper_8_29[12] - 16384 cols_helper_8_29[13]
   - 32768 cols_helper_8_29[14] - 65536 cols_helper_8_29[15]
   - 131072 cols_helper_8_29[16] = 0
 cols_acc_8_29: - 435 frac[8,29] + cols_8_29 >= 0
 accuracy_8_29: mem[8,29] - 0.01171875 cols_8_29 = 0
 cols_helper_11_56: cols_helper_11_56[0] + cols_helper_11_56[1]
   + cols_helper_11_56[2] + cols_helper_11_56[3] + cols_helper_11_56[4]
   + cols_helper_11_56[5] + cols_helper_11_56[6] + cols_helper_11_56[7]
   + cols_helper_11_56[8] + cols_helper_11_56[9] + cols_helper_11_56[10]
   + cols_helper_11_56[11] + cols_helper_11_56[12] + cols_helper_11_56[13]
   + cols_helper_11_56[14] + cols_helper_11_56[15] + cols_helper_11_56[16]
   <= 1
 cols_11_56: cols_11_56 - 2 cols_helper_11_56[0] - 4 cols_helper_11_56[1]
   - 8 cols_helper_11_56[2] - 16 cols_helper_11_56[3]
   - 32 cols_helper_11_56[4] - 64 cols_helper_11_56[5]
   - 128 cols_helper_11_56[6] - 256 cols_helper_11_56[7]
   - 512 cols_helper_11_56[8] - 1024 cols_helper_11_56[9]
   - 2048 cols_helper_11_56[10] - 4096 cols_helper_11_56[11]
   - 8192 cols_helper_11_56[12] - 16384 cols_helper_11_56[13]
   - 32768 cols_helper_11_56[14] - 65536 cols_helper_11_56[15]
   - 131072 cols_helper_11_56[16] = 0
 cols_acc_11_56: - 435 frac[11,56] + cols_11_56 >= 0
 accuracy_11_56: mem[11,56] - 0.01171875 cols_11_56 = 0
 cols_helper_9_24: cols_helper_9_24[0] + cols_helper_9_24[1]
   + cols_helper_9_24[2] + cols_helper_9_24[3] + cols_helper_9_24[4]
   + cols_helper_9_24[5] + cols_helper_9_24[6] + cols_helper_9_24[7]
   + cols_helper_9_24[8] + cols_helper_9_24[9] + cols_helper_9_24[10]
   + cols_helper_9_24[11] + cols_helper_9_24[12] + cols_helper_9_24[13]
   + cols_helper_9_24[14] + cols_helper_9_24[15] + cols_helper_9_24[16]
   <= 1
 cols_9_24: cols_9_24 - 2 cols_helper_9_24[0] - 4 cols_helper_9_24[1]
   - 8 cols_helper_9_24[2] - 16 cols_helper_9_24[3]
   - 32 cols_helper_9_24[4] - 64 cols_helper_9_24[5]
   - 128 cols_helper_9_24[6] - 256 cols_helper_9_24[7]
   - 512 cols_helper_9_24[8] - 1024 cols_helper_9_24[9]
   - 2048 cols_helper_9_24[10] - 4096 cols_helper_9_24[11]
   - 8192 cols_helper_9_24[12] - 16384 cols_helper_9_24[13]
   - 32768 cols_helper_9_24[14] - 65536 cols_helper_9_24[15]
   - 131072 cols_helper_9_24[16] = 0
 cols_acc_9_24: - 435 frac[9,24] + cols_9_24 >= 0
 accuracy_9_24: mem[9,24] - 0.01171875 cols_9_24 = 0
 accuracy_5_66: - 5.09765625 frac[5,66] + mem[5,66] = 0
 cols_helper_10_31: cols_helper_10_31[0] + cols_helper_10_31[1]
   + cols_helper_10_31[2] + cols_helper_10_31[3] + cols_helper_10_31[4]
   + cols_helper_10_31[5] + cols_helper_10_31[6] + cols_helper_10_31[7]
   + cols_helper_10_31[8] + cols_helper_10_31[9] + cols_helper_10_31[10]
   + cols_helper_10_31[11] + cols_helper_10_31[12] + cols_helper_10_31[13]
   + cols_helper_10_31[14] + cols_helper_10_31[15] + cols_helper_10_31[16]
   <= 1
 cols_10_31: cols_10_31 - 2 cols_helper_10_31[0] - 4 cols_helper_10_31[1]
   - 8 cols_helper_10_31[2] - 16 cols_helper_10_31[3]
   - 32 cols_helper_10_31[4] - 64 cols_helper_10_31[5]
   - 128 cols_helper_10_31[6] - 256 cols_helper_10_31[7]
   - 512 cols_helper_10_31[8] - 1024 cols_helper_10_31[9]
   - 2048 cols_helper_10_31[10] - 4096 cols_helper_10_31[11]
   - 8192 cols_helper_10_31[12] - 16384 cols_helper_10_31[13]
   - 32768 cols_helper_10_31[14] - 65536 cols_helper_10_31[15]
   - 131072 cols_helper_10_31[16] = 0
 cols_acc_10_31: - 435 frac[10,31] + cols_10_31 >= 0
 accuracy_10_31: mem[10,31] - 0.01171875 cols_10_31 = 0
 cols_helper_11_30: cols_helper_11_30[0] + cols_helper_11_30[1]
   + cols_helper_11_30[2] + cols_helper_11_30[3] + cols_helper_11_30[4]
   + cols_helper_11_30[5] + cols_helper_11_30[6] + cols_helper_11_30[7]
   + cols_helper_11_30[8] + cols_helper_11_30[9] + cols_helper_11_30[10]
   + cols_helper_11_30[11] + cols_helper_11_30[12] + cols_helper_11_30[13]
   + cols_helper_11_30[14] + cols_helper_11_30[15] + cols_helper_11_30[16]
   <= 1
 cols_11_30: cols_11_30 - 2 cols_helper_11_30[0] - 4 cols_helper_11_30[1]
   - 8 cols_helper_11_30[2] - 16 cols_helper_11_30[3]
   - 32 cols_helper_11_30[4] - 64 cols_helper_11_30[5]
   - 128 cols_helper_11_30[6] - 256 cols_helper_11_30[7]
   - 512 cols_helper_11_30[8] - 1024 cols_helper_11_30[9]
   - 2048 cols_helper_11_30[10] - 4096 cols_helper_11_30[11]
   - 8192 cols_helper_11_30[12] - 16384 cols_helper_11_30[13]
   - 32768 cols_helper_11_30[14] - 65536 cols_helper_11_30[15]
   - 131072 cols_helper_11_30[16] = 0
 cols_acc_11_30: - 435 frac[11,30] + cols_11_30 >= 0
 accuracy_11_30: mem[11,30] - 0.01171875 cols_11_30 = 0
 accuracy_0_25: - 5.09765625 frac[0,25] + mem[0,25] = 0
 cols_helper_8_73: cols_helper_8_73[0] + cols_helper_8_73[1]
   + cols_helper_8_73[2] + cols_helper_8_73[3] + cols_helper_8_73[4]
   + cols_helper_8_73[5] + cols_helper_8_73[6] + cols_helper_8_73[7]
   + cols_helper_8_73[8] + cols_helper_8_73[9] + cols_helper_8_73[10]
   + cols_helper_8_73[11] + cols_helper_8_73[12] + cols_helper_8_73[13]
   + cols_helper_8_73[14] + cols_helper_8_73[15] + cols_helper_8_73[16]
   <= 1
 cols_8_73: cols_8_73 - 2 cols_helper_8_73[0] - 4 cols_helper_8_73[1]
   - 8 cols_helper_8_73[2] - 16 cols_helper_8_73[3]
   - 32 cols_helper_8_73[4] - 64 cols_helper_8_73[5]
   - 128 cols_helper_8_73[6] - 256 cols_helper_8_73[7]
   - 512 cols_helper_8_73[8] - 1024 cols_helper_8_73[9]
   - 2048 cols_helper_8_73[10] - 4096 cols_helper_8_73[11]
   - 8192 cols_helper_8_73[12] - 16384 cols_helper_8_73[13]
   - 32768 cols_helper_8_73[14] - 65536 cols_helper_8_73[15]
   - 131072 cols_helper_8_73[16] = 0
 cols_acc_8_73: - 435 frac[8,73] + cols_8_73 >= 0
 accuracy_8_73: mem[8,73] - 0.01171875 cols_8_73 = 0
 cols_helper_10_67: cols_helper_10_67[0] + cols_helper_10_67[1]
   + cols_helper_10_67[2] + cols_helper_10_67[3] + cols_helper_10_67[4]
   + cols_helper_10_67[5] + cols_helper_10_67[6] + cols_helper_10_67[7]
   + cols_helper_10_67[8] + cols_helper_10_67[9] + cols_helper_10_67[10]
   + cols_helper_10_67[11] + cols_helper_10_67[12] + cols_helper_10_67[13]
   + cols_helper_10_67[14] + cols_helper_10_67[15] + cols_helper_10_67[16]
   <= 1
 cols_10_67: cols_10_67 - 2 cols_helper_10_67[0] - 4 cols_helper_10_67[1]
   - 8 cols_helper_10_67[2] - 16 cols_helper_10_67[3]
   - 32 cols_helper_10_67[4] - 64 cols_helper_10_67[5]
   - 128 cols_helper_10_67[6] - 256 cols_helper_10_67[7]
   - 512 cols_helper_10_67[8] - 1024 cols_helper_10_67[9]
   - 2048 cols_helper_10_67[10] - 4096 cols_helper_10_67[11]
   - 8192 cols_helper_10_67[12] - 16384 cols_helper_10_67[13]
   - 32768 cols_helper_10_67[14] - 65536 cols_helper_10_67[15]
   - 131072 cols_helper_10_67[16] = 0
 cols_acc_10_67: - 435 frac[10,67] + cols_10_67 >= 0
 accuracy_10_67: mem[10,67] - 0.01171875 cols_10_67 = 0
 accuracy_6_5: - 5.09765625 frac[6,5] + mem[6,5] = 0
 accuracy_4_29: - 5.09765625 frac[4,29] + mem[4,29] = 0
 cols_helper_11_69: cols_helper_11_69[0] + cols_helper_11_69[1]
   + cols_helper_11_69[2] + cols_helper_11_69[3] + cols_helper_11_69[4]
   + cols_helper_11_69[5] + cols_helper_11_69[6] + cols_helper_11_69[7]
   + cols_helper_11_69[8] + cols_helper_11_69[9] + cols_helper_11_69[10]
   + cols_helper_11_69[11] + cols_helper_11_69[12] + cols_helper_11_69[13]
   + cols_helper_11_69[14] + cols_helper_11_69[15] + cols_helper_11_69[16]
   <= 1
 cols_11_69: cols_11_69 - 2 cols_helper_11_69[0] - 4 cols_helper_11_69[1]
   - 8 cols_helper_11_69[2] - 16 cols_helper_11_69[3]
   - 32 cols_helper_11_69[4] - 64 cols_helper_11_69[5]
   - 128 cols_helper_11_69[6] - 256 cols_helper_11_69[7]
   - 512 cols_helper_11_69[8] - 1024 cols_helper_11_69[9]
   - 2048 cols_helper_11_69[10] - 4096 cols_helper_11_69[11]
   - 8192 cols_helper_11_69[12] - 16384 cols_helper_11_69[13]
   - 32768 cols_helper_11_69[14] - 65536 cols_helper_11_69[15]
   - 131072 cols_helper_11_69[16] = 0
 cols_acc_11_69: - 435 frac[11,69] + cols_11_69 >= 0
 accuracy_11_69: mem[11,69] - 0.01171875 cols_11_69 = 0
 accuracy_3_32: - 5.09765625 frac[3,32] + mem[3,32] = 0
 cols_helper_8_62: cols_helper_8_62[0] + cols_helper_8_62[1]
   + cols_helper_8_62[2] + cols_helper_8_62[3] + cols_helper_8_62[4]
   + cols_helper_8_62[5] + cols_helper_8_62[6] + cols_helper_8_62[7]
   + cols_helper_8_62[8] + cols_helper_8_62[9] + cols_helper_8_62[10]
   + cols_helper_8_62[11] + cols_helper_8_62[12] + cols_helper_8_62[13]
   + cols_helper_8_62[14] + cols_helper_8_62[15] + cols_helper_8_62[16]
   <= 1
 cols_8_62: cols_8_62 - 2 cols_helper_8_62[0] - 4 cols_helper_8_62[1]
   - 8 cols_helper_8_62[2] - 16 cols_helper_8_62[3]
   - 32 cols_helper_8_62[4] - 64 cols_helper_8_62[5]
   - 128 cols_helper_8_62[6] - 256 cols_helper_8_62[7]
   - 512 cols_helper_8_62[8] - 1024 cols_helper_8_62[9]
   - 2048 cols_helper_8_62[10] - 4096 cols_helper_8_62[11]
   - 8192 cols_helper_8_62[12] - 16384 cols_helper_8_62[13]
   - 32768 cols_helper_8_62[14] - 65536 cols_helper_8_62[15]
   - 131072 cols_helper_8_62[16] = 0
 cols_acc_8_62: - 435 frac[8,62] + cols_8_62 >= 0
 accuracy_8_62: mem[8,62] - 0.01171875 cols_8_62 = 0
 accuracy_4_65: - 5.09765625 frac[4,65] + mem[4,65] = 0
 cols_helper_10_60: cols_helper_10_60[0] + cols_helper_10_60[1]
   + cols_helper_10_60[2] + cols_helper_10_60[3] + cols_helper_10_60[4]
   + cols_helper_10_60[5] + cols_helper_10_60[6] + cols_helper_10_60[7]
   + cols_helper_10_60[8] + cols_helper_10_60[9] + cols_helper_10_60[10]
   + cols_helper_10_60[11] + cols_helper_10_60[12] + cols_helper_10_60[13]
   + cols_helper_10_60[14] + cols_helper_10_60[15] + cols_helper_10_60[16]
   <= 1
 cols_10_60: cols_10_60 - 2 cols_helper_10_60[0] - 4 cols_helper_10_60[1]
   - 8 cols_helper_10_60[2] - 16 cols_helper_10_60[3]
   - 32 cols_helper_10_60[4] - 64 cols_helper_10_60[5]
   - 128 cols_helper_10_60[6] - 256 cols_helper_10_60[7]
   - 512 cols_helper_10_60[8] - 1024 cols_helper_10_60[9]
   - 2048 cols_helper_10_60[10] - 4096 cols_helper_10_60[11]
   - 8192 cols_helper_10_60[12] - 16384 cols_helper_10_60[13]
   - 32768 cols_helper_10_60[14] - 65536 cols_helper_10_60[15]
   - 131072 cols_helper_10_60[16] = 0
 cols_acc_10_60: - 435 frac[10,60] + cols_10_60 >= 0
 accuracy_10_60: mem[10,60] - 0.01171875 cols_10_60 = 0
 accuracy_1_65: - 5.09765625 frac[1,65] + mem[1,65] = 0
 cols_helper_9_1: cols_helper_9_1[0] + cols_helper_9_1[1]
   + cols_helper_9_1[2] + cols_helper_9_1[3] + cols_helper_9_1[4]
   + cols_helper_9_1[5] + cols_helper_9_1[6] + cols_helper_9_1[7]
   + cols_helper_9_1[8] + cols_helper_9_1[9] + cols_helper_9_1[10]
   + cols_helper_9_1[11] + cols_helper_9_1[12] + cols_helper_9_1[13]
   + cols_helper_9_1[14] + cols_helper_9_1[15] + cols_helper_9_1[16] <= 1
 cols_9_1: cols_9_1 - 2 cols_helper_9_1[0] - 4 cols_helper_9_1[1]
   - 8 cols_helper_9_1[2] - 16 cols_helper_9_1[3] - 32 cols_helper_9_1[4]
   - 64 cols_helper_9_1[5] - 128 cols_helper_9_1[6]
   - 256 cols_helper_9_1[7] - 512 cols_helper_9_1[8]
   - 1024 cols_helper_9_1[9] - 2048 cols_helper_9_1[10]
   - 4096 cols_helper_9_1[11] - 8192 cols_helper_9_1[12]
   - 16384 cols_helper_9_1[13] - 32768 cols_helper_9_1[14]
   - 65536 cols_helper_9_1[15] - 131072 cols_helper_9_1[16] = 0
 cols_acc_9_1: - 435 frac[9,1] + cols_9_1 >= 0
 accuracy_9_1: mem[9,1] - 0.01171875 cols_9_1 = 0
 cols_helper_10_6: cols_helper_10_6[0] + cols_helper_10_6[1]
   + cols_helper_10_6[2] + cols_helper_10_6[3] + cols_helper_10_6[4]
   + cols_helper_10_6[5] + cols_helper_10_6[6] + cols_helper_10_6[7]
   + cols_helper_10_6[8] + cols_helper_10_6[9] + cols_helper_10_6[10]
   + cols_helper_10_6[11] + cols_helper_10_6[12] + cols_helper_10_6[13]
   + cols_helper_10_6[14] + cols_helper_10_6[15] + cols_helper_10_6[16]
   <= 1
 cols_10_6: cols_10_6 - 2 cols_helper_10_6[0] - 4 cols_helper_10_6[1]
   - 8 cols_helper_10_6[2] - 16 cols_helper_10_6[3]
   - 32 cols_helper_10_6[4] - 64 cols_helper_10_6[5]
   - 128 cols_helper_10_6[6] - 256 cols_helper_10_6[7]
   - 512 cols_helper_10_6[8] - 1024 cols_helper_10_6[9]
   - 2048 cols_helper_10_6[10] - 4096 cols_helper_10_6[11]
   - 8192 cols_helper_10_6[12] - 16384 cols_helper_10_6[13]
   - 32768 cols_helper_10_6[14] - 65536 cols_helper_10_6[15]
   - 131072 cols_helper_10_6[16] = 0
 cols_acc_10_6: - 435 frac[10,6] + cols_10_6 >= 0
 accuracy_10_6: mem[10,6] - 0.01171875 cols_10_6 = 0
 cols_helper_11_23: cols_helper_11_23[0] + cols_helper_11_23[1]
   + cols_helper_11_23[2] + cols_helper_11_23[3] + cols_helper_11_23[4]
   + cols_helper_11_23[5] + cols_helper_11_23[6] + cols_helper_11_23[7]
   + cols_helper_11_23[8] + cols_helper_11_23[9] + cols_helper_11_23[10]
   + cols_helper_11_23[11] + cols_helper_11_23[12] + cols_helper_11_23[13]
   + cols_helper_11_23[14] + cols_helper_11_23[15] + cols_helper_11_23[16]
   <= 1
 cols_11_23: cols_11_23 - 2 cols_helper_11_23[0] - 4 cols_helper_11_23[1]
   - 8 cols_helper_11_23[2] - 16 cols_helper_11_23[3]
   - 32 cols_helper_11_23[4] - 64 cols_helper_11_23[5]
   - 128 cols_helper_11_23[6] - 256 cols_helper_11_23[7]
   - 512 cols_helper_11_23[8] - 1024 cols_helper_11_23[9]
   - 2048 cols_helper_11_23[10] - 4096 cols_helper_11_23[11]
   - 8192 cols_helper_11_23[12] - 16384 cols_helper_11_23[13]
   - 32768 cols_helper_11_23[14] - 65536 cols_helper_11_23[15]
   - 131072 cols_helper_11_23[16] = 0
 cols_acc_11_23: - 435 frac[11,23] + cols_11_23 >= 0
 accuracy_11_23: mem[11,23] - 0.01171875 cols_11_23 = 0
 accuracy_0_16: - 5.09765625 frac[0,16] + mem[0,16] = 0
 accuracy_7_69: - 5.09765625 frac[7,69] + mem[7,69] = 0
 accuracy_1_29: - 5.09765625 frac[1,29] + mem[1,29] = 0
 cols_helper_10_74: cols_helper_10_74[0] + cols_helper_10_74[1]
   + cols_helper_10_74[2] + cols_helper_10_74[3] + cols_helper_10_74[4]
   + cols_helper_10_74[5] + cols_helper_10_74[6] + cols_helper_10_74[7]
   + cols_helper_10_74[8] + cols_helper_10_74[9] + cols_helper_10_74[10]
   + cols_helper_10_74[11] + cols_helper_10_74[12] + cols_helper_10_74[13]
   + cols_helper_10_74[14] + cols_helper_10_74[15] + cols_helper_10_74[16]
   <= 1
 cols_10_74: cols_10_74 - 2 cols_helper_10_74[0] - 4 cols_helper_10_74[1]
   - 8 cols_helper_10_74[2] - 16 cols_helper_10_74[3]
   - 32 cols_helper_10_74[4] - 64 cols_helper_10_74[5]
   - 128 cols_helper_10_74[6] - 256 cols_helper_10_74[7]
   - 512 cols_helper_10_74[8] - 1024 cols_helper_10_74[9]
   - 2048 cols_helper_10_74[10] - 4096 cols_helper_10_74[11]
   - 8192 cols_helper_10_74[12] - 16384 cols_helper_10_74[13]
   - 32768 cols_helper_10_74[14] - 65536 cols_helper_10_74[15]
   - 131072 cols_helper_10_74[16] = 0
 cols_acc_10_74: - 435 frac[10,74] + cols_10_74 >= 0
 accuracy_10_74: mem[10,74] - 0.01171875 cols_10_74 = 0
 accuracy_5_25: - 5.09765625 frac[5,25] + mem[5,25] = 0
 cols_helper_8_37: cols_helper_8_37[0] + cols_helper_8_37[1]
   + cols_helper_8_37[2] + cols_helper_8_37[3] + cols_helper_8_37[4]
   + cols_helper_8_37[5] + cols_helper_8_37[6] + cols_helper_8_37[7]
   + cols_helper_8_37[8] + cols_helper_8_37[9] + cols_helper_8_37[10]
   + cols_helper_8_37[11] + cols_helper_8_37[12] + cols_helper_8_37[13]
   + cols_helper_8_37[14] + cols_helper_8_37[15] + cols_helper_8_37[16]
   <= 1
 cols_8_37: cols_8_37 - 2 cols_helper_8_37[0] - 4 cols_helper_8_37[1]
   - 8 cols_helper_8_37[2] - 16 cols_helper_8_37[3]
   - 32 cols_helper_8_37[4] - 64 cols_helper_8_37[5]
   - 128 cols_helper_8_37[6] - 256 cols_helper_8_37[7]
   - 512 cols_helper_8_37[8] - 1024 cols_helper_8_37[9]
   - 2048 cols_helper_8_37[10] - 4096 cols_helper_8_37[11]
   - 8192 cols_helper_8_37[12] - 16384 cols_helper_8_37[13]
   - 32768 cols_helper_8_37[14] - 65536 cols_helper_8_37[15]
   - 131072 cols_helper_8_37[16] = 0
 cols_acc_8_37: - 435 frac[8,37] + cols_8_37 >= 0
 accuracy_8_37: mem[8,37] - 0.01171875 cols_8_37 = 0
 accuracy_0_66: - 5.09765625 frac[0,66] + mem[0,66] = 0
 cols_helper_9_32: cols_helper_9_32[0] + cols_helper_9_32[1]
   + cols_helper_9_32[2] + cols_helper_9_32[3] + cols_helper_9_32[4]
   + cols_helper_9_32[5] + cols_helper_9_32[6] + cols_helper_9_32[7]
   + cols_helper_9_32[8] + cols_helper_9_32[9] + cols_helper_9_32[10]
   + cols_helper_9_32[11] + cols_helper_9_32[12] + cols_helper_9_32[13]
   + cols_helper_9_32[14] + cols_helper_9_32[15] + cols_helper_9_32[16]
   <= 1
 cols_9_32: cols_9_32 - 2 cols_helper_9_32[0] - 4 cols_helper_9_32[1]
   - 8 cols_helper_9_32[2] - 16 cols_helper_9_32[3]
   - 32 cols_helper_9_32[4] - 64 cols_helper_9_32[5]
   - 128 cols_helper_9_32[6] - 256 cols_helper_9_32[7]
   - 512 cols_helper_9_32[8] - 1024 cols_helper_9_32[9]
   - 2048 cols_helper_9_32[10] - 4096 cols_helper_9_32[11]
   - 8192 cols_helper_9_32[12] - 16384 cols_helper_9_32[13]
   - 32768 cols_helper_9_32[14] - 65536 cols_helper_9_32[15]
   - 131072 cols_helper_9_32[16] = 0
 cols_acc_9_32: - 435 frac[9,32] + cols_9_32 >= 0
 accuracy_9_32: mem[9,32] - 0.01171875 cols_9_32 = 0
 cols_helper_10_39: cols_helper_10_39[0] + cols_helper_10_39[1]
   + cols_helper_10_39[2] + cols_helper_10_39[3] + cols_helper_10_39[4]
   + cols_helper_10_39[5] + cols_helper_10_39[6] + cols_helper_10_39[7]
   + cols_helper_10_39[8] + cols_helper_10_39[9] + cols_helper_10_39[10]
   + cols_helper_10_39[11] + cols_helper_10_39[12] + cols_helper_10_39[13]
   + cols_helper_10_39[14] + cols_helper_10_39[15] + cols_helper_10_39[16]
   <= 1
 cols_10_39: cols_10_39 - 2 cols_helper_10_39[0] - 4 cols_helper_10_39[1]
   - 8 cols_helper_10_39[2] - 16 cols_helper_10_39[3]
   - 32 cols_helper_10_39[4] - 64 cols_helper_10_39[5]
   - 128 cols_helper_10_39[6] - 256 cols_helper_10_39[7]
   - 512 cols_helper_10_39[8] - 1024 cols_helper_10_39[9]
   - 2048 cols_helper_10_39[10] - 4096 cols_helper_10_39[11]
   - 8192 cols_helper_10_39[12] - 16384 cols_helper_10_39[13]
   - 32768 cols_helper_10_39[14] - 65536 cols_helper_10_39[15]
   - 131072 cols_helper_10_39[16] = 0
 cols_acc_10_39: - 435 frac[10,39] + cols_10_39 >= 0
 accuracy_10_39: mem[10,39] - 0.01171875 cols_10_39 = 0
 cols_helper_11_54: cols_helper_11_54[0] + cols_helper_11_54[1]
   + cols_helper_11_54[2] + cols_helper_11_54[3] + cols_helper_11_54[4]
   + cols_helper_11_54[5] + cols_helper_11_54[6] + cols_helper_11_54[7]
   + cols_helper_11_54[8] + cols_helper_11_54[9] + cols_helper_11_54[10]
   + cols_helper_11_54[11] + cols_helper_11_54[12] + cols_helper_11_54[13]
   + cols_helper_11_54[14] + cols_helper_11_54[15] + cols_helper_11_54[16]
   <= 1
 cols_11_54: cols_11_54 - 2 cols_helper_11_54[0] - 4 cols_helper_11_54[1]
   - 8 cols_helper_11_54[2] - 16 cols_helper_11_54[3]
   - 32 cols_helper_11_54[4] - 64 cols_helper_11_54[5]
   - 128 cols_helper_11_54[6] - 256 cols_helper_11_54[7]
   - 512 cols_helper_11_54[8] - 1024 cols_helper_11_54[9]
   - 2048 cols_helper_11_54[10] - 4096 cols_helper_11_54[11]
   - 8192 cols_helper_11_54[12] - 16384 cols_helper_11_54[13]
   - 32768 cols_helper_11_54[14] - 65536 cols_helper_11_54[15]
   - 131072 cols_helper_11_54[16] = 0
 cols_acc_11_54: - 435 frac[11,54] + cols_11_54 >= 0
 accuracy_11_54: mem[11,54] - 0.01171875 cols_11_54 = 0
 cols_helper_9_10: cols_helper_9_10[0] + cols_helper_9_10[1]
   + cols_helper_9_10[2] + cols_helper_9_10[3] + cols_helper_9_10[4]
   + cols_helper_9_10[5] + cols_helper_9_10[6] + cols_helper_9_10[7]
   + cols_helper_9_10[8] + cols_helper_9_10[9] + cols_helper_9_10[10]
   + cols_helper_9_10[11] + cols_helper_9_10[12] + cols_helper_9_10[13]
   + cols_helper_9_10[14] + cols_helper_9_10[15] + cols_helper_9_10[16]
   <= 1
 cols_9_10: cols_9_10 - 2 cols_helper_9_10[0] - 4 cols_helper_9_10[1]
   - 8 cols_helper_9_10[2] - 16 cols_helper_9_10[3]
   - 32 cols_helper_9_10[4] - 64 cols_helper_9_10[5]
   - 128 cols_helper_9_10[6] - 256 cols_helper_9_10[7]
   - 512 cols_helper_9_10[8] - 1024 cols_helper_9_10[9]
   - 2048 cols_helper_9_10[10] - 4096 cols_helper_9_10[11]
   - 8192 cols_helper_9_10[12] - 16384 cols_helper_9_10[13]
   - 32768 cols_helper_9_10[14] - 65536 cols_helper_9_10[15]
   - 131072 cols_helper_9_10[16] = 0
 cols_acc_9_10: - 435 frac[9,10] + cols_9_10 >= 0
 accuracy_9_10: mem[9,10] - 0.01171875 cols_9_10 = 0
 cols_helper_10_1: cols_helper_10_1[0] + cols_helper_10_1[1]
   + cols_helper_10_1[2] + cols_helper_10_1[3] + cols_helper_10_1[4]
   + cols_helper_10_1[5] + cols_helper_10_1[6] + cols_helper_10_1[7]
   + cols_helper_10_1[8] + cols_helper_10_1[9] + cols_helper_10_1[10]
   + cols_helper_10_1[11] + cols_helper_10_1[12] + cols_helper_10_1[13]
   + cols_helper_10_1[14] + cols_helper_10_1[15] + cols_helper_10_1[16]
   <= 1
 cols_10_1: cols_10_1 - 2 cols_helper_10_1[0] - 4 cols_helper_10_1[1]
   - 8 cols_helper_10_1[2] - 16 cols_helper_10_1[3]
   - 32 cols_helper_10_1[4] - 64 cols_helper_10_1[5]
   - 128 cols_helper_10_1[6] - 256 cols_helper_10_1[7]
   - 512 cols_helper_10_1[8] - 1024 cols_helper_10_1[9]
   - 2048 cols_helper_10_1[10] - 4096 cols_helper_10_1[11]
   - 8192 cols_helper_10_1[12] - 16384 cols_helper_10_1[13]
   - 32768 cols_helper_10_1[14] - 65536 cols_helper_10_1[15]
   - 131072 cols_helper_10_1[16] = 0
 cols_acc_10_1: - 435 frac[10,1] + cols_10_1 >= 0
 accuracy_10_1: mem[10,1] - 0.01171875 cols_10_1 = 0
 cols_helper_8_17: cols_helper_8_17[0] + cols_helper_8_17[1]
   + cols_helper_8_17[2] + cols_helper_8_17[3] + cols_helper_8_17[4]
   + cols_helper_8_17[5] + cols_helper_8_17[6] + cols_helper_8_17[7]
   + cols_helper_8_17[8] + cols_helper_8_17[9] + cols_helper_8_17[10]
   + cols_helper_8_17[11] + cols_helper_8_17[12] + cols_helper_8_17[13]
   + cols_helper_8_17[14] + cols_helper_8_17[15] + cols_helper_8_17[16]
   <= 1
 cols_8_17: cols_8_17 - 2 cols_helper_8_17[0] - 4 cols_helper_8_17[1]
   - 8 cols_helper_8_17[2] - 16 cols_helper_8_17[3]
   - 32 cols_helper_8_17[4] - 64 cols_helper_8_17[5]
   - 128 cols_helper_8_17[6] - 256 cols_helper_8_17[7]
   - 512 cols_helper_8_17[8] - 1024 cols_helper_8_17[9]
   - 2048 cols_helper_8_17[10] - 4096 cols_helper_8_17[11]
   - 8192 cols_helper_8_17[12] - 16384 cols_helper_8_17[13]
   - 32768 cols_helper_8_17[14] - 65536 cols_helper_8_17[15]
   - 131072 cols_helper_8_17[16] = 0
 cols_acc_8_17: - 435 frac[8,17] + cols_8_17 >= 0
 accuracy_8_17: mem[8,17] - 0.01171875 cols_8_17 = 0
 cols_helper_11_12: cols_helper_11_12[0] + cols_helper_11_12[1]
   + cols_helper_11_12[2] + cols_helper_11_12[3] + cols_helper_11_12[4]
   + cols_helper_11_12[5] + cols_helper_11_12[6] + cols_helper_11_12[7]
   + cols_helper_11_12[8] + cols_helper_11_12[9] + cols_helper_11_12[10]
   + cols_helper_11_12[11] + cols_helper_11_12[12] + cols_helper_11_12[13]
   + cols_helper_11_12[14] + cols_helper_11_12[15] + cols_helper_11_12[16]
   <= 1
 cols_11_12: cols_11_12 - 2 cols_helper_11_12[0] - 4 cols_helper_11_12[1]
   - 8 cols_helper_11_12[2] - 16 cols_helper_11_12[3]
   - 32 cols_helper_11_12[4] - 64 cols_helper_11_12[5]
   - 128 cols_helper_11_12[6] - 256 cols_helper_11_12[7]
   - 512 cols_helper_11_12[8] - 1024 cols_helper_11_12[9]
   - 2048 cols_helper_11_12[10] - 4096 cols_helper_11_12[11]
   - 8192 cols_helper_11_12[12] - 16384 cols_helper_11_12[13]
   - 32768 cols_helper_11_12[14] - 65536 cols_helper_11_12[15]
   - 131072 cols_helper_11_12[16] = 0
 cols_acc_11_12: - 435 frac[11,12] + cols_11_12 >= 0
 accuracy_11_12: mem[11,12] - 0.01171875 cols_11_12 = 0
 accuracy_2_21: - 5.09765625 frac[2,21] + mem[2,21] = 0
 cols_helper_9_71: cols_helper_9_71[0] + cols_helper_9_71[1]
   + cols_helper_9_71[2] + cols_helper_9_71[3] + cols_helper_9_71[4]
   + cols_helper_9_71[5] + cols_helper_9_71[6] + cols_helper_9_71[7]
   + cols_helper_9_71[8] + cols_helper_9_71[9] + cols_helper_9_71[10]
   + cols_helper_9_71[11] + cols_helper_9_71[12] + cols_helper_9_71[13]
   + cols_helper_9_71[14] + cols_helper_9_71[15] + cols_helper_9_71[16]
   <= 1
 cols_9_71: cols_9_71 - 2 cols_helper_9_71[0] - 4 cols_helper_9_71[1]
   - 8 cols_helper_9_71[2] - 16 cols_helper_9_71[3]
   - 32 cols_helper_9_71[4] - 64 cols_helper_9_71[5]
   - 128 cols_helper_9_71[6] - 256 cols_helper_9_71[7]
   - 512 cols_helper_9_71[8] - 1024 cols_helper_9_71[9]
   - 2048 cols_helper_9_71[10] - 4096 cols_helper_9_71[11]
   - 8192 cols_helper_9_71[12] - 16384 cols_helper_9_71[13]
   - 32768 cols_helper_9_71[14] - 65536 cols_helper_9_71[15]
   - 131072 cols_helper_9_71[16] = 0
 cols_acc_9_71: - 435 frac[9,71] + cols_9_71 >= 0
 accuracy_9_71: mem[9,71] - 0.01171875 cols_9_71 = 0
 accuracy_1_34: - 5.09765625 frac[1,34] + mem[1,34] = 0
 accuracy_7_12: - 5.09765625 frac[7,12] + mem[7,12] = 0
 cols_helper_11_73: cols_helper_11_73[0] + cols_helper_11_73[1]
   + cols_helper_11_73[2] + cols_helper_11_73[3] + cols_helper_11_73[4]
   + cols_helper_11_73[5] + cols_helper_11_73[6] + cols_helper_11_73[7]
   + cols_helper_11_73[8] + cols_helper_11_73[9] + cols_helper_11_73[10]
   + cols_helper_11_73[11] + cols_helper_11_73[12] + cols_helper_11_73[13]
   + cols_helper_11_73[14] + cols_helper_11_73[15] + cols_helper_11_73[16]
   <= 1
 cols_11_73: cols_11_73 - 2 cols_helper_11_73[0] - 4 cols_helper_11_73[1]
   - 8 cols_helper_11_73[2] - 16 cols_helper_11_73[3]
   - 32 cols_helper_11_73[4] - 64 cols_helper_11_73[5]
   - 128 cols_helper_11_73[6] - 256 cols_helper_11_73[7]
   - 512 cols_helper_11_73[8] - 1024 cols_helper_11_73[9]
   - 2048 cols_helper_11_73[10] - 4096 cols_helper_11_73[11]
   - 8192 cols_helper_11_73[12] - 16384 cols_helper_11_73[13]
   - 32768 cols_helper_11_73[14] - 65536 cols_helper_11_73[15]
   - 131072 cols_helper_11_73[16] = 0
 cols_acc_11_73: - 435 frac[11,73] + cols_11_73 >= 0
 accuracy_11_73: mem[11,73] - 0.01171875 cols_11_73 = 0
 cols_helper_9_41: cols_helper_9_41[0] + cols_helper_9_41[1]
   + cols_helper_9_41[2] + cols_helper_9_41[3] + cols_helper_9_41[4]
   + cols_helper_9_41[5] + cols_helper_9_41[6] + cols_helper_9_41[7]
   + cols_helper_9_41[8] + cols_helper_9_41[9] + cols_helper_9_41[10]
   + cols_helper_9_41[11] + cols_helper_9_41[12] + cols_helper_9_41[13]
   + cols_helper_9_41[14] + cols_helper_9_41[15] + cols_helper_9_41[16]
   <= 1
 cols_9_41: cols_9_41 - 2 cols_helper_9_41[0] - 4 cols_helper_9_41[1]
   - 8 cols_helper_9_41[2] - 16 cols_helper_9_41[3]
   - 32 cols_helper_9_41[4] - 64 cols_helper_9_41[5]
   - 128 cols_helper_9_41[6] - 256 cols_helper_9_41[7]
   - 512 cols_helper_9_41[8] - 1024 cols_helper_9_41[9]
   - 2048 cols_helper_9_41[10] - 4096 cols_helper_9_41[11]
   - 8192 cols_helper_9_41[12] - 16384 cols_helper_9_41[13]
   - 32768 cols_helper_9_41[14] - 65536 cols_helper_9_41[15]
   - 131072 cols_helper_9_41[16] = 0
 cols_acc_9_41: - 435 frac[9,41] + cols_9_41 >= 0
 accuracy_9_41: mem[9,41] - 0.01171875 cols_9_41 = 0
 cols_helper_10_46: cols_helper_10_46[0] + cols_helper_10_46[1]
   + cols_helper_10_46[2] + cols_helper_10_46[3] + cols_helper_10_46[4]
   + cols_helper_10_46[5] + cols_helper_10_46[6] + cols_helper_10_46[7]
   + cols_helper_10_46[8] + cols_helper_10_46[9] + cols_helper_10_46[10]
   + cols_helper_10_46[11] + cols_helper_10_46[12] + cols_helper_10_46[13]
   + cols_helper_10_46[14] + cols_helper_10_46[15] + cols_helper_10_46[16]
   <= 1
 cols_10_46: cols_10_46 - 2 cols_helper_10_46[0] - 4 cols_helper_10_46[1]
   - 8 cols_helper_10_46[2] - 16 cols_helper_10_46[3]
   - 32 cols_helper_10_46[4] - 64 cols_helper_10_46[5]
   - 128 cols_helper_10_46[6] - 256 cols_helper_10_46[7]
   - 512 cols_helper_10_46[8] - 1024 cols_helper_10_46[9]
   - 2048 cols_helper_10_46[10] - 4096 cols_helper_10_46[11]
   - 8192 cols_helper_10_46[12] - 16384 cols_helper_10_46[13]
   - 32768 cols_helper_10_46[14] - 65536 cols_helper_10_46[15]
   - 131072 cols_helper_10_46[16] = 0
 cols_acc_10_46: - 435 frac[10,46] + cols_10_46 >= 0
 accuracy_10_46: mem[10,46] - 0.01171875 cols_10_46 = 0
 accuracy_2_67: - 5.09765625 frac[2,67] + mem[2,67] = 0
 cols_helper_8_50: cols_helper_8_50[0] + cols_helper_8_50[1]
   + cols_helper_8_50[2] + cols_helper_8_50[3] + cols_helper_8_50[4]
   + cols_helper_8_50[5] + cols_helper_8_50[6] + cols_helper_8_50[7]
   + cols_helper_8_50[8] + cols_helper_8_50[9] + cols_helper_8_50[10]
   + cols_helper_8_50[11] + cols_helper_8_50[12] + cols_helper_8_50[13]
   + cols_helper_8_50[14] + cols_helper_8_50[15] + cols_helper_8_50[16]
   <= 1
 cols_8_50: cols_8_50 - 2 cols_helper_8_50[0] - 4 cols_helper_8_50[1]
   - 8 cols_helper_8_50[2] - 16 cols_helper_8_50[3]
   - 32 cols_helper_8_50[4] - 64 cols_helper_8_50[5]
   - 128 cols_helper_8_50[6] - 256 cols_helper_8_50[7]
   - 512 cols_helper_8_50[8] - 1024 cols_helper_8_50[9]
   - 2048 cols_helper_8_50[10] - 4096 cols_helper_8_50[11]
   - 8192 cols_helper_8_50[12] - 16384 cols_helper_8_50[13]
   - 32768 cols_helper_8_50[14] - 65536 cols_helper_8_50[15]
   - 131072 cols_helper_8_50[16] = 0
 cols_acc_8_50: - 435 frac[8,50] + cols_8_50 >= 0
 accuracy_8_50: mem[8,50] - 0.01171875 cols_8_50 = 0
 cols_helper_11_47: cols_helper_11_47[0] + cols_helper_11_47[1]
   + cols_helper_11_47[2] + cols_helper_11_47[3] + cols_helper_11_47[4]
   + cols_helper_11_47[5] + cols_helper_11_47[6] + cols_helper_11_47[7]
   + cols_helper_11_47[8] + cols_helper_11_47[9] + cols_helper_11_47[10]
   + cols_helper_11_47[11] + cols_helper_11_47[12] + cols_helper_11_47[13]
   + cols_helper_11_47[14] + cols_helper_11_47[15] + cols_helper_11_47[16]
   <= 1
 cols_11_47: cols_11_47 - 2 cols_helper_11_47[0] - 4 cols_helper_11_47[1]
   - 8 cols_helper_11_47[2] - 16 cols_helper_11_47[3]
   - 32 cols_helper_11_47[4] - 64 cols_helper_11_47[5]
   - 128 cols_helper_11_47[6] - 256 cols_helper_11_47[7]
   - 512 cols_helper_11_47[8] - 1024 cols_helper_11_47[9]
   - 2048 cols_helper_11_47[10] - 4096 cols_helper_11_47[11]
   - 8192 cols_helper_11_47[12] - 16384 cols_helper_11_47[13]
   - 32768 cols_helper_11_47[14] - 65536 cols_helper_11_47[15]
   - 131072 cols_helper_11_47[16] = 0
 cols_acc_11_47: - 435 frac[11,47] + cols_11_47 >= 0
 accuracy_11_47: mem[11,47] - 0.01171875 cols_11_47 = 0
 cols_helper_9_51: cols_helper_9_51[0] + cols_helper_9_51[1]
   + cols_helper_9_51[2] + cols_helper_9_51[3] + cols_helper_9_51[4]
   + cols_helper_9_51[5] + cols_helper_9_51[6] + cols_helper_9_51[7]
   + cols_helper_9_51[8] + cols_helper_9_51[9] + cols_helper_9_51[10]
   + cols_helper_9_51[11] + cols_helper_9_51[12] + cols_helper_9_51[13]
   + cols_helper_9_51[14] + cols_helper_9_51[15] + cols_helper_9_51[16]
   <= 1
 cols_9_51: cols_9_51 - 2 cols_helper_9_51[0] - 4 cols_helper_9_51[1]
   - 8 cols_helper_9_51[2] - 16 cols_helper_9_51[3]
   - 32 cols_helper_9_51[4] - 64 cols_helper_9_51[5]
   - 128 cols_helper_9_51[6] - 256 cols_helper_9_51[7]
   - 512 cols_helper_9_51[8] - 1024 cols_helper_9_51[9]
   - 2048 cols_helper_9_51[10] - 4096 cols_helper_9_51[11]
   - 8192 cols_helper_9_51[12] - 16384 cols_helper_9_51[13]
   - 32768 cols_helper_9_51[14] - 65536 cols_helper_9_51[15]
   - 131072 cols_helper_9_51[16] = 0
 cols_acc_9_51: - 435 frac[9,51] + cols_9_51 >= 0
 accuracy_9_51: mem[9,51] - 0.01171875 cols_9_51 = 0
 cols_helper_10_8: cols_helper_10_8[0] + cols_helper_10_8[1]
   + cols_helper_10_8[2] + cols_helper_10_8[3] + cols_helper_10_8[4]
   + cols_helper_10_8[5] + cols_helper_10_8[6] + cols_helper_10_8[7]
   + cols_helper_10_8[8] + cols_helper_10_8[9] + cols_helper_10_8[10]
   + cols_helper_10_8[11] + cols_helper_10_8[12] + cols_helper_10_8[13]
   + cols_helper_10_8[14] + cols_helper_10_8[15] + cols_helper_10_8[16]
   <= 1
 cols_10_8: cols_10_8 - 2 cols_helper_10_8[0] - 4 cols_helper_10_8[1]
   - 8 cols_helper_10_8[2] - 16 cols_helper_10_8[3]
   - 32 cols_helper_10_8[4] - 64 cols_helper_10_8[5]
   - 128 cols_helper_10_8[6] - 256 cols_helper_10_8[7]
   - 512 cols_helper_10_8[8] - 1024 cols_helper_10_8[9]
   - 2048 cols_helper_10_8[10] - 4096 cols_helper_10_8[11]
   - 8192 cols_helper_10_8[12] - 16384 cols_helper_10_8[13]
   - 32768 cols_helper_10_8[14] - 65536 cols_helper_10_8[15]
   - 131072 cols_helper_10_8[16] = 0
 cols_acc_10_8: - 435 frac[10,8] + cols_10_8 >= 0
 accuracy_10_8: mem[10,8] - 0.01171875 cols_10_8 = 0
 cols_helper_11_5: cols_helper_11_5[0] + cols_helper_11_5[1]
   + cols_helper_11_5[2] + cols_helper_11_5[3] + cols_helper_11_5[4]
   + cols_helper_11_5[5] + cols_helper_11_5[6] + cols_helper_11_5[7]
   + cols_helper_11_5[8] + cols_helper_11_5[9] + cols_helper_11_5[10]
   + cols_helper_11_5[11] + cols_helper_11_5[12] + cols_helper_11_5[13]
   + cols_helper_11_5[14] + cols_helper_11_5[15] + cols_helper_11_5[16]
   <= 1
 cols_11_5: cols_11_5 - 2 cols_helper_11_5[0] - 4 cols_helper_11_5[1]
   - 8 cols_helper_11_5[2] - 16 cols_helper_11_5[3]
   - 32 cols_helper_11_5[4] - 64 cols_helper_11_5[5]
   - 128 cols_helper_11_5[6] - 256 cols_helper_11_5[7]
   - 512 cols_helper_11_5[8] - 1024 cols_helper_11_5[9]
   - 2048 cols_helper_11_5[10] - 4096 cols_helper_11_5[11]
   - 8192 cols_helper_11_5[12] - 16384 cols_helper_11_5[13]
   - 32768 cols_helper_11_5[14] - 65536 cols_helper_11_5[15]
   - 131072 cols_helper_11_5[16] = 0
 cols_acc_11_5: - 435 frac[11,5] + cols_11_5 >= 0
 accuracy_11_5: mem[11,5] - 0.01171875 cols_11_5 = 0
 cols_helper_9_21: cols_helper_9_21[0] + cols_helper_9_21[1]
   + cols_helper_9_21[2] + cols_helper_9_21[3] + cols_helper_9_21[4]
   + cols_helper_9_21[5] + cols_helper_9_21[6] + cols_helper_9_21[7]
   + cols_helper_9_21[8] + cols_helper_9_21[9] + cols_helper_9_21[10]
   + cols_helper_9_21[11] + cols_helper_9_21[12] + cols_helper_9_21[13]
   + cols_helper_9_21[14] + cols_helper_9_21[15] + cols_helper_9_21[16]
   <= 1
 cols_9_21: cols_9_21 - 2 cols_helper_9_21[0] - 4 cols_helper_9_21[1]
   - 8 cols_helper_9_21[2] - 16 cols_helper_9_21[3]
   - 32 cols_helper_9_21[4] - 64 cols_helper_9_21[5]
   - 128 cols_helper_9_21[6] - 256 cols_helper_9_21[7]
   - 512 cols_helper_9_21[8] - 1024 cols_helper_9_21[9]
   - 2048 cols_helper_9_21[10] - 4096 cols_helper_9_21[11]
   - 8192 cols_helper_9_21[12] - 16384 cols_helper_9_21[13]
   - 32768 cols_helper_9_21[14] - 65536 cols_helper_9_21[15]
   - 131072 cols_helper_9_21[16] = 0
 cols_acc_9_21: - 435 frac[9,21] + cols_9_21 >= 0
 accuracy_9_21: mem[9,21] - 0.01171875 cols_9_21 = 0
 cols_helper_10_18: cols_helper_10_18[0] + cols_helper_10_18[1]
   + cols_helper_10_18[2] + cols_helper_10_18[3] + cols_helper_10_18[4]
   + cols_helper_10_18[5] + cols_helper_10_18[6] + cols_helper_10_18[7]
   + cols_helper_10_18[8] + cols_helper_10_18[9] + cols_helper_10_18[10]
   + cols_helper_10_18[11] + cols_helper_10_18[12] + cols_helper_10_18[13]
   + cols_helper_10_18[14] + cols_helper_10_18[15] + cols_helper_10_18[16]
   <= 1
 cols_10_18: cols_10_18 - 2 cols_helper_10_18[0] - 4 cols_helper_10_18[1]
   - 8 cols_helper_10_18[2] - 16 cols_helper_10_18[3]
   - 32 cols_helper_10_18[4] - 64 cols_helper_10_18[5]
   - 128 cols_helper_10_18[6] - 256 cols_helper_10_18[7]
   - 512 cols_helper_10_18[8] - 1024 cols_helper_10_18[9]
   - 2048 cols_helper_10_18[10] - 4096 cols_helper_10_18[11]
   - 8192 cols_helper_10_18[12] - 16384 cols_helper_10_18[13]
   - 32768 cols_helper_10_18[14] - 65536 cols_helper_10_18[15]
   - 131072 cols_helper_10_18[16] = 0
 cols_acc_10_18: - 435 frac[10,18] + cols_10_18 >= 0
 accuracy_10_18: mem[10,18] - 0.01171875 cols_10_18 = 0
 cols_helper_9_72: cols_helper_9_72[0] + cols_helper_9_72[1]
   + cols_helper_9_72[2] + cols_helper_9_72[3] + cols_helper_9_72[4]
   + cols_helper_9_72[5] + cols_helper_9_72[6] + cols_helper_9_72[7]
   + cols_helper_9_72[8] + cols_helper_9_72[9] + cols_helper_9_72[10]
   + cols_helper_9_72[11] + cols_helper_9_72[12] + cols_helper_9_72[13]
   + cols_helper_9_72[14] + cols_helper_9_72[15] + cols_helper_9_72[16]
   <= 1
 cols_9_72: cols_9_72 - 2 cols_helper_9_72[0] - 4 cols_helper_9_72[1]
   - 8 cols_helper_9_72[2] - 16 cols_helper_9_72[3]
   - 32 cols_helper_9_72[4] - 64 cols_helper_9_72[5]
   - 128 cols_helper_9_72[6] - 256 cols_helper_9_72[7]
   - 512 cols_helper_9_72[8] - 1024 cols_helper_9_72[9]
   - 2048 cols_helper_9_72[10] - 4096 cols_helper_9_72[11]
   - 8192 cols_helper_9_72[12] - 16384 cols_helper_9_72[13]
   - 32768 cols_helper_9_72[14] - 65536 cols_helper_9_72[15]
   - 131072 cols_helper_9_72[16] = 0
 cols_acc_9_72: - 435 frac[9,72] + cols_9_72 >= 0
 accuracy_9_72: mem[9,72] - 0.01171875 cols_9_72 = 0
 cols_helper_10_70: cols_helper_10_70[0] + cols_helper_10_70[1]
   + cols_helper_10_70[2] + cols_helper_10_70[3] + cols_helper_10_70[4]
   + cols_helper_10_70[5] + cols_helper_10_70[6] + cols_helper_10_70[7]
   + cols_helper_10_70[8] + cols_helper_10_70[9] + cols_helper_10_70[10]
   + cols_helper_10_70[11] + cols_helper_10_70[12] + cols_helper_10_70[13]
   + cols_helper_10_70[14] + cols_helper_10_70[15] + cols_helper_10_70[16]
   <= 1
 cols_10_70: cols_10_70 - 2 cols_helper_10_70[0] - 4 cols_helper_10_70[1]
   - 8 cols_helper_10_70[2] - 16 cols_helper_10_70[3]
   - 32 cols_helper_10_70[4] - 64 cols_helper_10_70[5]
   - 128 cols_helper_10_70[6] - 256 cols_helper_10_70[7]
   - 512 cols_helper_10_70[8] - 1024 cols_helper_10_70[9]
   - 2048 cols_helper_10_70[10] - 4096 cols_helper_10_70[11]
   - 8192 cols_helper_10_70[12] - 16384 cols_helper_10_70[13]
   - 32768 cols_helper_10_70[14] - 65536 cols_helper_10_70[15]
   - 131072 cols_helper_10_70[16] = 0
 cols_acc_10_70: - 435 frac[10,70] + cols_10_70 >= 0
 accuracy_10_70: mem[10,70] - 0.01171875 cols_10_70 = 0
 accuracy_2_32: - 5.09765625 frac[2,32] + mem[2,32] = 0
 accuracy_5_21: - 5.09765625 frac[5,21] + mem[5,21] = 0
 accuracy_3_61: - 5.09765625 frac[3,61] + mem[3,61] = 0
 cols_helper_10_41: cols_helper_10_41[0] + cols_helper_10_41[1]
   + cols_helper_10_41[2] + cols_helper_10_41[3] + cols_helper_10_41[4]
   + cols_helper_10_41[5] + cols_helper_10_41[6] + cols_helper_10_41[7]
   + cols_helper_10_41[8] + cols_helper_10_41[9] + cols_helper_10_41[10]
   + cols_helper_10_41[11] + cols_helper_10_41[12] + cols_helper_10_41[13]
   + cols_helper_10_41[14] + cols_helper_10_41[15] + cols_helper_10_41[16]
   <= 1
 cols_10_41: cols_10_41 - 2 cols_helper_10_41[0] - 4 cols_helper_10_41[1]
   - 8 cols_helper_10_41[2] - 16 cols_helper_10_41[3]
   - 32 cols_helper_10_41[4] - 64 cols_helper_10_41[5]
   - 128 cols_helper_10_41[6] - 256 cols_helper_10_41[7]
   - 512 cols_helper_10_41[8] - 1024 cols_helper_10_41[9]
   - 2048 cols_helper_10_41[10] - 4096 cols_helper_10_41[11]
   - 8192 cols_helper_10_41[12] - 16384 cols_helper_10_41[13]
   - 32768 cols_helper_10_41[14] - 65536 cols_helper_10_41[15]
   - 131072 cols_helper_10_41[16] = 0
 cols_acc_10_41: - 435 frac[10,41] + cols_10_41 >= 0
 accuracy_10_41: mem[10,41] - 0.01171875 cols_10_41 = 0
 cols_helper_8_57: cols_helper_8_57[0] + cols_helper_8_57[1]
   + cols_helper_8_57[2] + cols_helper_8_57[3] + cols_helper_8_57[4]
   + cols_helper_8_57[5] + cols_helper_8_57[6] + cols_helper_8_57[7]
   + cols_helper_8_57[8] + cols_helper_8_57[9] + cols_helper_8_57[10]
   + cols_helper_8_57[11] + cols_helper_8_57[12] + cols_helper_8_57[13]
   + cols_helper_8_57[14] + cols_helper_8_57[15] + cols_helper_8_57[16]
   <= 1
 cols_8_57: cols_8_57 - 2 cols_helper_8_57[0] - 4 cols_helper_8_57[1]
   - 8 cols_helper_8_57[2] - 16 cols_helper_8_57[3]
   - 32 cols_helper_8_57[4] - 64 cols_helper_8_57[5]
   - 128 cols_helper_8_57[6] - 256 cols_helper_8_57[7]
   - 512 cols_helper_8_57[8] - 1024 cols_helper_8_57[9]
   - 2048 cols_helper_8_57[10] - 4096 cols_helper_8_57[11]
   - 8192 cols_helper_8_57[12] - 16384 cols_helper_8_57[13]
   - 32768 cols_helper_8_57[14] - 65536 cols_helper_8_57[15]
   - 131072 cols_helper_8_57[16] = 0
 cols_acc_8_57: - 435 frac[8,57] + cols_8_57 >= 0
 accuracy_8_57: mem[8,57] - 0.01171875 cols_8_57 = 0
 accuracy_3_74: - 5.09765625 frac[3,74] + mem[3,74] = 0
 cols_helper_10_51: cols_helper_10_51[0] + cols_helper_10_51[1]
   + cols_helper_10_51[2] + cols_helper_10_51[3] + cols_helper_10_51[4]
   + cols_helper_10_51[5] + cols_helper_10_51[6] + cols_helper_10_51[7]
   + cols_helper_10_51[8] + cols_helper_10_51[9] + cols_helper_10_51[10]
   + cols_helper_10_51[11] + cols_helper_10_51[12] + cols_helper_10_51[13]
   + cols_helper_10_51[14] + cols_helper_10_51[15] + cols_helper_10_51[16]
   <= 1
 cols_10_51: cols_10_51 - 2 cols_helper_10_51[0] - 4 cols_helper_10_51[1]
   - 8 cols_helper_10_51[2] - 16 cols_helper_10_51[3]
   - 32 cols_helper_10_51[4] - 64 cols_helper_10_51[5]
   - 128 cols_helper_10_51[6] - 256 cols_helper_10_51[7]
   - 512 cols_helper_10_51[8] - 1024 cols_helper_10_51[9]
   - 2048 cols_helper_10_51[10] - 4096 cols_helper_10_51[11]
   - 8192 cols_helper_10_51[12] - 16384 cols_helper_10_51[13]
   - 32768 cols_helper_10_51[14] - 65536 cols_helper_10_51[15]
   - 131072 cols_helper_10_51[16] = 0
 cols_acc_10_51: - 435 frac[10,51] + cols_10_51 >= 0
 accuracy_10_51: mem[10,51] - 0.01171875 cols_10_51 = 0
 cols_helper_11_58: cols_helper_11_58[0] + cols_helper_11_58[1]
   + cols_helper_11_58[2] + cols_helper_11_58[3] + cols_helper_11_58[4]
   + cols_helper_11_58[5] + cols_helper_11_58[6] + cols_helper_11_58[7]
   + cols_helper_11_58[8] + cols_helper_11_58[9] + cols_helper_11_58[10]
   + cols_helper_11_58[11] + cols_helper_11_58[12] + cols_helper_11_58[13]
   + cols_helper_11_58[14] + cols_helper_11_58[15] + cols_helper_11_58[16]
   <= 1
 cols_11_58: cols_11_58 - 2 cols_helper_11_58[0] - 4 cols_helper_11_58[1]
   - 8 cols_helper_11_58[2] - 16 cols_helper_11_58[3]
   - 32 cols_helper_11_58[4] - 64 cols_helper_11_58[5]
   - 128 cols_helper_11_58[6] - 256 cols_helper_11_58[7]
   - 512 cols_helper_11_58[8] - 1024 cols_helper_11_58[9]
   - 2048 cols_helper_11_58[10] - 4096 cols_helper_11_58[11]
   - 8192 cols_helper_11_58[12] - 16384 cols_helper_11_58[13]
   - 32768 cols_helper_11_58[14] - 65536 cols_helper_11_58[15]
   - 131072 cols_helper_11_58[16] = 0
 cols_acc_11_58: - 435 frac[11,58] + cols_11_58 >= 0
 accuracy_11_58: mem[11,58] - 0.01171875 cols_11_58 = 0
 cols_helper_9_30: cols_helper_9_30[0] + cols_helper_9_30[1]
   + cols_helper_9_30[2] + cols_helper_9_30[3] + cols_helper_9_30[4]
   + cols_helper_9_30[5] + cols_helper_9_30[6] + cols_helper_9_30[7]
   + cols_helper_9_30[8] + cols_helper_9_30[9] + cols_helper_9_30[10]
   + cols_helper_9_30[11] + cols_helper_9_30[12] + cols_helper_9_30[13]
   + cols_helper_9_30[14] + cols_helper_9_30[15] + cols_helper_9_30[16]
   <= 1
 cols_9_30: cols_9_30 - 2 cols_helper_9_30[0] - 4 cols_helper_9_30[1]
   - 8 cols_helper_9_30[2] - 16 cols_helper_9_30[3]
   - 32 cols_helper_9_30[4] - 64 cols_helper_9_30[5]
   - 128 cols_helper_9_30[6] - 256 cols_helper_9_30[7]
   - 512 cols_helper_9_30[8] - 1024 cols_helper_9_30[9]
   - 2048 cols_helper_9_30[10] - 4096 cols_helper_9_30[11]
   - 8192 cols_helper_9_30[12] - 16384 cols_helper_9_30[13]
   - 32768 cols_helper_9_30[14] - 65536 cols_helper_9_30[15]
   - 131072 cols_helper_9_30[16] = 0
 cols_acc_9_30: - 435 frac[9,30] + cols_9_30 >= 0
 accuracy_9_30: mem[9,30] - 0.01171875 cols_9_30 = 0
 cols_helper_10_29: cols_helper_10_29[0] + cols_helper_10_29[1]
   + cols_helper_10_29[2] + cols_helper_10_29[3] + cols_helper_10_29[4]
   + cols_helper_10_29[5] + cols_helper_10_29[6] + cols_helper_10_29[7]
   + cols_helper_10_29[8] + cols_helper_10_29[9] + cols_helper_10_29[10]
   + cols_helper_10_29[11] + cols_helper_10_29[12] + cols_helper_10_29[13]
   + cols_helper_10_29[14] + cols_helper_10_29[15] + cols_helper_10_29[16]
   <= 1
 cols_10_29: cols_10_29 - 2 cols_helper_10_29[0] - 4 cols_helper_10_29[1]
   - 8 cols_helper_10_29[2] - 16 cols_helper_10_29[3]
   - 32 cols_helper_10_29[4] - 64 cols_helper_10_29[5]
   - 128 cols_helper_10_29[6] - 256 cols_helper_10_29[7]
   - 512 cols_helper_10_29[8] - 1024 cols_helper_10_29[9]
   - 2048 cols_helper_10_29[10] - 4096 cols_helper_10_29[11]
   - 8192 cols_helper_10_29[12] - 16384 cols_helper_10_29[13]
   - 32768 cols_helper_10_29[14] - 65536 cols_helper_10_29[15]
   - 131072 cols_helper_10_29[16] = 0
 cols_acc_10_29: - 435 frac[10,29] + cols_10_29 >= 0
 accuracy_10_29: mem[10,29] - 0.01171875 cols_10_29 = 0
 cols_helper_10_65: cols_helper_10_65[0] + cols_helper_10_65[1]
   + cols_helper_10_65[2] + cols_helper_10_65[3] + cols_helper_10_65[4]
   + cols_helper_10_65[5] + cols_helper_10_65[6] + cols_helper_10_65[7]
   + cols_helper_10_65[8] + cols_helper_10_65[9] + cols_helper_10_65[10]
   + cols_helper_10_65[11] + cols_helper_10_65[12] + cols_helper_10_65[13]
   + cols_helper_10_65[14] + cols_helper_10_65[15] + cols_helper_10_65[16]
   <= 1
 cols_10_65: cols_10_65 - 2 cols_helper_10_65[0] - 4 cols_helper_10_65[1]
   - 8 cols_helper_10_65[2] - 16 cols_helper_10_65[3]
   - 32 cols_helper_10_65[4] - 64 cols_helper_10_65[5]
   - 128 cols_helper_10_65[6] - 256 cols_helper_10_65[7]
   - 512 cols_helper_10_65[8] - 1024 cols_helper_10_65[9]
   - 2048 cols_helper_10_65[10] - 4096 cols_helper_10_65[11]
   - 8192 cols_helper_10_65[12] - 16384 cols_helper_10_65[13]
   - 32768 cols_helper_10_65[14] - 65536 cols_helper_10_65[15]
   - 131072 cols_helper_10_65[16] = 0
 cols_acc_10_65: - 435 frac[10,65] + cols_10_65 >= 0
 accuracy_10_65: mem[10,65] - 0.01171875 cols_10_65 = 0
 accuracy_7_2: - 5.09765625 frac[7,2] + mem[7,2] = 0
 accuracy_5_30: - 5.09765625 frac[5,30] + mem[5,30] = 0
 accuracy_3_34: - 5.09765625 frac[3,34] + mem[3,34] = 0
 cols_helper_8_32: cols_helper_8_32[0] + cols_helper_8_32[1]
   + cols_helper_8_32[2] + cols_helper_8_32[3] + cols_helper_8_32[4]
   + cols_helper_8_32[5] + cols_helper_8_32[6] + cols_helper_8_32[7]
   + cols_helper_8_32[8] + cols_helper_8_32[9] + cols_helper_8_32[10]
   + cols_helper_8_32[11] + cols_helper_8_32[12] + cols_helper_8_32[13]
   + cols_helper_8_32[14] + cols_helper_8_32[15] + cols_helper_8_32[16]
   <= 1
 cols_8_32: cols_8_32 - 2 cols_helper_8_32[0] - 4 cols_helper_8_32[1]
   - 8 cols_helper_8_32[2] - 16 cols_helper_8_32[3]
   - 32 cols_helper_8_32[4] - 64 cols_helper_8_32[5]
   - 128 cols_helper_8_32[6] - 256 cols_helper_8_32[7]
   - 512 cols_helper_8_32[8] - 1024 cols_helper_8_32[9]
   - 2048 cols_helper_8_32[10] - 4096 cols_helper_8_32[11]
   - 8192 cols_helper_8_32[12] - 16384 cols_helper_8_32[13]
   - 32768 cols_helper_8_32[14] - 65536 cols_helper_8_32[15]
   - 131072 cols_helper_8_32[16] = 0
 cols_acc_8_32: - 435 frac[8,32] + cols_8_32 >= 0
 accuracy_8_32: mem[8,32] - 0.01171875 cols_8_32 = 0
 cols_helper_9_61: cols_helper_9_61[0] + cols_helper_9_61[1]
   + cols_helper_9_61[2] + cols_helper_9_61[3] + cols_helper_9_61[4]
   + cols_helper_9_61[5] + cols_helper_9_61[6] + cols_helper_9_61[7]
   + cols_helper_9_61[8] + cols_helper_9_61[9] + cols_helper_9_61[10]
   + cols_helper_9_61[11] + cols_helper_9_61[12] + cols_helper_9_61[13]
   + cols_helper_9_61[14] + cols_helper_9_61[15] + cols_helper_9_61[16]
   <= 1
 cols_9_61: cols_9_61 - 2 cols_helper_9_61[0] - 4 cols_helper_9_61[1]
   - 8 cols_helper_9_61[2] - 16 cols_helper_9_61[3]
   - 32 cols_helper_9_61[4] - 64 cols_helper_9_61[5]
   - 128 cols_helper_9_61[6] - 256 cols_helper_9_61[7]
   - 512 cols_helper_9_61[8] - 1024 cols_helper_9_61[9]
   - 2048 cols_helper_9_61[10] - 4096 cols_helper_9_61[11]
   - 8192 cols_helper_9_61[12] - 16384 cols_helper_9_61[13]
   - 32768 cols_helper_9_61[14] - 65536 cols_helper_9_61[15]
   - 131072 cols_helper_9_61[16] = 0
 cols_acc_9_61: - 435 frac[9,61] + cols_9_61 >= 0
 accuracy_9_61: mem[9,61] - 0.01171875 cols_9_61 = 0
 cols_helper_10_58: cols_helper_10_58[0] + cols_helper_10_58[1]
   + cols_helper_10_58[2] + cols_helper_10_58[3] + cols_helper_10_58[4]
   + cols_helper_10_58[5] + cols_helper_10_58[6] + cols_helper_10_58[7]
   + cols_helper_10_58[8] + cols_helper_10_58[9] + cols_helper_10_58[10]
   + cols_helper_10_58[11] + cols_helper_10_58[12] + cols_helper_10_58[13]
   + cols_helper_10_58[14] + cols_helper_10_58[15] + cols_helper_10_58[16]
   <= 1
 cols_10_58: cols_10_58 - 2 cols_helper_10_58[0] - 4 cols_helper_10_58[1]
   - 8 cols_helper_10_58[2] - 16 cols_helper_10_58[3]
   - 32 cols_helper_10_58[4] - 64 cols_helper_10_58[5]
   - 128 cols_helper_10_58[6] - 256 cols_helper_10_58[7]
   - 512 cols_helper_10_58[8] - 1024 cols_helper_10_58[9]
   - 2048 cols_helper_10_58[10] - 4096 cols_helper_10_58[11]
   - 8192 cols_helper_10_58[12] - 16384 cols_helper_10_58[13]
   - 32768 cols_helper_10_58[14] - 65536 cols_helper_10_58[15]
   - 131072 cols_helper_10_58[16] = 0
 cols_acc_10_58: - 435 frac[10,58] + cols_10_58 >= 0
 accuracy_10_58: mem[10,58] - 0.01171875 cols_10_58 = 0
 accuracy_1_71: - 5.09765625 frac[1,71] + mem[1,71] = 0
 cols_helper_11_51: cols_helper_11_51[0] + cols_helper_11_51[1]
   + cols_helper_11_51[2] + cols_helper_11_51[3] + cols_helper_11_51[4]
   + cols_helper_11_51[5] + cols_helper_11_51[6] + cols_helper_11_51[7]
   + cols_helper_11_51[8] + cols_helper_11_51[9] + cols_helper_11_51[10]
   + cols_helper_11_51[11] + cols_helper_11_51[12] + cols_helper_11_51[13]
   + cols_helper_11_51[14] + cols_helper_11_51[15] + cols_helper_11_51[16]
   <= 1
 cols_11_51: cols_11_51 - 2 cols_helper_11_51[0] - 4 cols_helper_11_51[1]
   - 8 cols_helper_11_51[2] - 16 cols_helper_11_51[3]
   - 32 cols_helper_11_51[4] - 64 cols_helper_11_51[5]
   - 128 cols_helper_11_51[6] - 256 cols_helper_11_51[7]
   - 512 cols_helper_11_51[8] - 1024 cols_helper_11_51[9]
   - 2048 cols_helper_11_51[10] - 4096 cols_helper_11_51[11]
   - 8192 cols_helper_11_51[12] - 16384 cols_helper_11_51[13]
   - 32768 cols_helper_11_51[14] - 65536 cols_helper_11_51[15]
   - 131072 cols_helper_11_51[16] = 0
 cols_acc_11_51: - 435 frac[11,51] + cols_11_51 >= 0
 accuracy_11_51: mem[11,51] - 0.01171875 cols_11_51 = 0
 cols_helper_9_7: cols_helper_9_7[0] + cols_helper_9_7[1]
   + cols_helper_9_7[2] + cols_helper_9_7[3] + cols_helper_9_7[4]
   + cols_helper_9_7[5] + cols_helper_9_7[6] + cols_helper_9_7[7]
   + cols_helper_9_7[8] + cols_helper_9_7[9] + cols_helper_9_7[10]
   + cols_helper_9_7[11] + cols_helper_9_7[12] + cols_helper_9_7[13]
   + cols_helper_9_7[14] + cols_helper_9_7[15] + cols_helper_9_7[16] <= 1
 cols_9_7: cols_9_7 - 2 cols_helper_9_7[0] - 4 cols_helper_9_7[1]
   - 8 cols_helper_9_7[2] - 16 cols_helper_9_7[3] - 32 cols_helper_9_7[4]
   - 64 cols_helper_9_7[5] - 128 cols_helper_9_7[6]
   - 256 cols_helper_9_7[7] - 512 cols_helper_9_7[8]
   - 1024 cols_helper_9_7[9] - 2048 cols_helper_9_7[10]
   - 4096 cols_helper_9_7[11] - 8192 cols_helper_9_7[12]
   - 16384 cols_helper_9_7[13] - 32768 cols_helper_9_7[14]
   - 65536 cols_helper_9_7[15] - 131072 cols_helper_9_7[16] = 0
 cols_acc_9_7: - 435 frac[9,7] + cols_9_7 >= 0
 accuracy_9_7: mem[9,7] - 0.01171875 cols_9_7 = 0
 accuracy_7_71: - 5.09765625 frac[7,71] + mem[7,71] = 0
 accuracy_0_18: - 5.09765625 frac[0,18] + mem[0,18] = 0
 accuracy_1_19: - 5.09765625 frac[1,19] + mem[1,19] = 0
 accuracy_3_5: - 5.09765625 frac[3,5] + mem[3,5] = 0
 cols_helper_10_72: cols_helper_10_72[0] + cols_helper_10_72[1]
   + cols_helper_10_72[2] + cols_helper_10_72[3] + cols_helper_10_72[4]
   + cols_helper_10_72[5] + cols_helper_10_72[6] + cols_helper_10_72[7]
   + cols_helper_10_72[8] + cols_helper_10_72[9] + cols_helper_10_72[10]
   + cols_helper_10_72[11] + cols_helper_10_72[12] + cols_helper_10_72[13]
   + cols_helper_10_72[14] + cols_helper_10_72[15] + cols_helper_10_72[16]
   <= 1
 cols_10_72: cols_10_72 - 2 cols_helper_10_72[0] - 4 cols_helper_10_72[1]
   - 8 cols_helper_10_72[2] - 16 cols_helper_10_72[3]
   - 32 cols_helper_10_72[4] - 64 cols_helper_10_72[5]
   - 128 cols_helper_10_72[6] - 256 cols_helper_10_72[7]
   - 512 cols_helper_10_72[8] - 1024 cols_helper_10_72[9]
   - 2048 cols_helper_10_72[10] - 4096 cols_helper_10_72[11]
   - 8192 cols_helper_10_72[12] - 16384 cols_helper_10_72[13]
   - 32768 cols_helper_10_72[14] - 65536 cols_helper_10_72[15]
   - 131072 cols_helper_10_72[16] = 0
 cols_acc_10_72: - 435 frac[10,72] + cols_10_72 >= 0
 accuracy_10_72: mem[10,72] - 0.01171875 cols_10_72 = 0
 accuracy_3_43: - 5.09765625 frac[3,43] + mem[3,43] = 0
 cols_helper_8_39: cols_helper_8_39[0] + cols_helper_8_39[1]
   + cols_helper_8_39[2] + cols_helper_8_39[3] + cols_helper_8_39[4]
   + cols_helper_8_39[5] + cols_helper_8_39[6] + cols_helper_8_39[7]
   + cols_helper_8_39[8] + cols_helper_8_39[9] + cols_helper_8_39[10]
   + cols_helper_8_39[11] + cols_helper_8_39[12] + cols_helper_8_39[13]
   + cols_helper_8_39[14] + cols_helper_8_39[15] + cols_helper_8_39[16]
   <= 1
 cols_8_39: cols_8_39 - 2 cols_helper_8_39[0] - 4 cols_helper_8_39[1]
   - 8 cols_helper_8_39[2] - 16 cols_helper_8_39[3]
   - 32 cols_helper_8_39[4] - 64 cols_helper_8_39[5]
   - 128 cols_helper_8_39[6] - 256 cols_helper_8_39[7]
   - 512 cols_helper_8_39[8] - 1024 cols_helper_8_39[9]
   - 2048 cols_helper_8_39[10] - 4096 cols_helper_8_39[11]
   - 8192 cols_helper_8_39[12] - 16384 cols_helper_8_39[13]
   - 32768 cols_helper_8_39[14] - 65536 cols_helper_8_39[15]
   - 131072 cols_helper_8_39[16] = 0
 cols_acc_8_39: - 435 frac[8,39] + cols_8_39 >= 0
 accuracy_8_39: mem[8,39] - 0.01171875 cols_8_39 = 0
 cols_helper_9_38: cols_helper_9_38[0] + cols_helper_9_38[1]
   + cols_helper_9_38[2] + cols_helper_9_38[3] + cols_helper_9_38[4]
   + cols_helper_9_38[5] + cols_helper_9_38[6] + cols_helper_9_38[7]
   + cols_helper_9_38[8] + cols_helper_9_38[9] + cols_helper_9_38[10]
   + cols_helper_9_38[11] + cols_helper_9_38[12] + cols_helper_9_38[13]
   + cols_helper_9_38[14] + cols_helper_9_38[15] + cols_helper_9_38[16]
   <= 1
 cols_9_38: cols_9_38 - 2 cols_helper_9_38[0] - 4 cols_helper_9_38[1]
   - 8 cols_helper_9_38[2] - 16 cols_helper_9_38[3]
   - 32 cols_helper_9_38[4] - 64 cols_helper_9_38[5]
   - 128 cols_helper_9_38[6] - 256 cols_helper_9_38[7]
   - 512 cols_helper_9_38[8] - 1024 cols_helper_9_38[9]
   - 2048 cols_helper_9_38[10] - 4096 cols_helper_9_38[11]
   - 8192 cols_helper_9_38[12] - 16384 cols_helper_9_38[13]
   - 32768 cols_helper_9_38[14] - 65536 cols_helper_9_38[15]
   - 131072 cols_helper_9_38[16] = 0
 cols_acc_9_38: - 435 frac[9,38] + cols_9_38 >= 0
 accuracy_9_38: mem[9,38] - 0.01171875 cols_9_38 = 0
 cols_helper_10_37: cols_helper_10_37[0] + cols_helper_10_37[1]
   + cols_helper_10_37[2] + cols_helper_10_37[3] + cols_helper_10_37[4]
   + cols_helper_10_37[5] + cols_helper_10_37[6] + cols_helper_10_37[7]
   + cols_helper_10_37[8] + cols_helper_10_37[9] + cols_helper_10_37[10]
   + cols_helper_10_37[11] + cols_helper_10_37[12] + cols_helper_10_37[13]
   + cols_helper_10_37[14] + cols_helper_10_37[15] + cols_helper_10_37[16]
   <= 1
 cols_10_37: cols_10_37 - 2 cols_helper_10_37[0] - 4 cols_helper_10_37[1]
   - 8 cols_helper_10_37[2] - 16 cols_helper_10_37[3]
   - 32 cols_helper_10_37[4] - 64 cols_helper_10_37[5]
   - 128 cols_helper_10_37[6] - 256 cols_helper_10_37[7]
   - 512 cols_helper_10_37[8] - 1024 cols_helper_10_37[9]
   - 2048 cols_helper_10_37[10] - 4096 cols_helper_10_37[11]
   - 8192 cols_helper_10_37[12] - 16384 cols_helper_10_37[13]
   - 32768 cols_helper_10_37[14] - 65536 cols_helper_10_37[15]
   - 131072 cols_helper_10_37[16] = 0
 cols_acc_10_37: - 435 frac[10,37] + cols_10_37 >= 0
 accuracy_10_37: mem[10,37] - 0.01171875 cols_10_37 = 0
 accuracy_7_47: - 5.09765625 frac[7,47] + mem[7,47] = 0
 cols_helper_11_40: cols_helper_11_40[0] + cols_helper_11_40[1]
   + cols_helper_11_40[2] + cols_helper_11_40[3] + cols_helper_11_40[4]
   + cols_helper_11_40[5] + cols_helper_11_40[6] + cols_helper_11_40[7]
   + cols_helper_11_40[8] + cols_helper_11_40[9] + cols_helper_11_40[10]
   + cols_helper_11_40[11] + cols_helper_11_40[12] + cols_helper_11_40[13]
   + cols_helper_11_40[14] + cols_helper_11_40[15] + cols_helper_11_40[16]
   <= 1
 cols_11_40: cols_11_40 - 2 cols_helper_11_40[0] - 4 cols_helper_11_40[1]
   - 8 cols_helper_11_40[2] - 16 cols_helper_11_40[3]
   - 32 cols_helper_11_40[4] - 64 cols_helper_11_40[5]
   - 128 cols_helper_11_40[6] - 256 cols_helper_11_40[7]
   - 512 cols_helper_11_40[8] - 1024 cols_helper_11_40[9]
   - 2048 cols_helper_11_40[10] - 4096 cols_helper_11_40[11]
   - 8192 cols_helper_11_40[12] - 16384 cols_helper_11_40[13]
   - 32768 cols_helper_11_40[14] - 65536 cols_helper_11_40[15]
   - 131072 cols_helper_11_40[16] = 0
 cols_acc_11_40: - 435 frac[11,40] + cols_11_40 >= 0
 accuracy_11_40: mem[11,40] - 0.01171875 cols_11_40 = 0
 cols_helper_10_15: cols_helper_10_15[0] + cols_helper_10_15[1]
   + cols_helper_10_15[2] + cols_helper_10_15[3] + cols_helper_10_15[4]
   + cols_helper_10_15[5] + cols_helper_10_15[6] + cols_helper_10_15[7]
   + cols_helper_10_15[8] + cols_helper_10_15[9] + cols_helper_10_15[10]
   + cols_helper_10_15[11] + cols_helper_10_15[12] + cols_helper_10_15[13]
   + cols_helper_10_15[14] + cols_helper_10_15[15] + cols_helper_10_15[16]
   <= 1
 cols_10_15: cols_10_15 - 2 cols_helper_10_15[0] - 4 cols_helper_10_15[1]
   - 8 cols_helper_10_15[2] - 16 cols_helper_10_15[3]
   - 32 cols_helper_10_15[4] - 64 cols_helper_10_15[5]
   - 128 cols_helper_10_15[6] - 256 cols_helper_10_15[7]
   - 512 cols_helper_10_15[8] - 1024 cols_helper_10_15[9]
   - 2048 cols_helper_10_15[10] - 4096 cols_helper_10_15[11]
   - 8192 cols_helper_10_15[12] - 16384 cols_helper_10_15[13]
   - 32768 cols_helper_10_15[14] - 65536 cols_helper_10_15[15]
   - 131072 cols_helper_10_15[16] = 0
 cols_acc_10_15: - 435 frac[10,15] + cols_10_15 >= 0
 accuracy_10_15: mem[10,15] - 0.01171875 cols_10_15 = 0
 cols_helper_8_19: cols_helper_8_19[0] + cols_helper_8_19[1]
   + cols_helper_8_19[2] + cols_helper_8_19[3] + cols_helper_8_19[4]
   + cols_helper_8_19[5] + cols_helper_8_19[6] + cols_helper_8_19[7]
   + cols_helper_8_19[8] + cols_helper_8_19[9] + cols_helper_8_19[10]
   + cols_helper_8_19[11] + cols_helper_8_19[12] + cols_helper_8_19[13]
   + cols_helper_8_19[14] + cols_helper_8_19[15] + cols_helper_8_19[16]
   <= 1
 cols_8_19: cols_8_19 - 2 cols_helper_8_19[0] - 4 cols_helper_8_19[1]
   - 8 cols_helper_8_19[2] - 16 cols_helper_8_19[3]
   - 32 cols_helper_8_19[4] - 64 cols_helper_8_19[5]
   - 128 cols_helper_8_19[6] - 256 cols_helper_8_19[7]
   - 512 cols_helper_8_19[8] - 1024 cols_helper_8_19[9]
   - 2048 cols_helper_8_19[10] - 4096 cols_helper_8_19[11]
   - 8192 cols_helper_8_19[12] - 16384 cols_helper_8_19[13]
   - 32768 cols_helper_8_19[14] - 65536 cols_helper_8_19[15]
   - 131072 cols_helper_8_19[16] = 0
 cols_acc_8_19: - 435 frac[8,19] + cols_8_19 >= 0
 accuracy_8_19: mem[8,19] - 0.01171875 cols_8_19 = 0
 cols_helper_11_14: cols_helper_11_14[0] + cols_helper_11_14[1]
   + cols_helper_11_14[2] + cols_helper_11_14[3] + cols_helper_11_14[4]
   + cols_helper_11_14[5] + cols_helper_11_14[6] + cols_helper_11_14[7]
   + cols_helper_11_14[8] + cols_helper_11_14[9] + cols_helper_11_14[10]
   + cols_helper_11_14[11] + cols_helper_11_14[12] + cols_helper_11_14[13]
   + cols_helper_11_14[14] + cols_helper_11_14[15] + cols_helper_11_14[16]
   <= 1
 cols_11_14: cols_11_14 - 2 cols_helper_11_14[0] - 4 cols_helper_11_14[1]
   - 8 cols_helper_11_14[2] - 16 cols_helper_11_14[3]
   - 32 cols_helper_11_14[4] - 64 cols_helper_11_14[5]
   - 128 cols_helper_11_14[6] - 256 cols_helper_11_14[7]
   - 512 cols_helper_11_14[8] - 1024 cols_helper_11_14[9]
   - 2048 cols_helper_11_14[10] - 4096 cols_helper_11_14[11]
   - 8192 cols_helper_11_14[12] - 16384 cols_helper_11_14[13]
   - 32768 cols_helper_11_14[14] - 65536 cols_helper_11_14[15]
   - 131072 cols_helper_11_14[16] = 0
 cols_acc_11_14: - 435 frac[11,14] + cols_11_14 >= 0
 accuracy_11_14: mem[11,14] - 0.01171875 cols_11_14 = 0
 cols_helper_9_18: cols_helper_9_18[0] + cols_helper_9_18[1]
   + cols_helper_9_18[2] + cols_helper_9_18[3] + cols_helper_9_18[4]
   + cols_helper_9_18[5] + cols_helper_9_18[6] + cols_helper_9_18[7]
   + cols_helper_9_18[8] + cols_helper_9_18[9] + cols_helper_9_18[10]
   + cols_helper_9_18[11] + cols_helper_9_18[12] + cols_helper_9_18[13]
   + cols_helper_9_18[14] + cols_helper_9_18[15] + cols_helper_9_18[16]
   <= 1
 cols_9_18: cols_9_18 - 2 cols_helper_9_18[0] - 4 cols_helper_9_18[1]
   - 8 cols_helper_9_18[2] - 16 cols_helper_9_18[3]
   - 32 cols_helper_9_18[4] - 64 cols_helper_9_18[5]
   - 128 cols_helper_9_18[6] - 256 cols_helper_9_18[7]
   - 512 cols_helper_9_18[8] - 1024 cols_helper_9_18[9]
   - 2048 cols_helper_9_18[10] - 4096 cols_helper_9_18[11]
   - 8192 cols_helper_9_18[12] - 16384 cols_helper_9_18[13]
   - 32768 cols_helper_9_18[14] - 65536 cols_helper_9_18[15]
   - 131072 cols_helper_9_18[16] = 0
 cols_acc_9_18: - 435 frac[9,18] + cols_9_18 >= 0
 accuracy_9_18: mem[9,18] - 0.01171875 cols_9_18 = 0
 accuracy_2_19: - 5.09765625 frac[2,19] + mem[2,19] = 0
 accuracy_3_16: - 5.09765625 frac[3,16] + mem[3,16] = 0
 cols_helper_9_69: cols_helper_9_69[0] + cols_helper_9_69[1]
   + cols_helper_9_69[2] + cols_helper_9_69[3] + cols_helper_9_69[4]
   + cols_helper_9_69[5] + cols_helper_9_69[6] + cols_helper_9_69[7]
   + cols_helper_9_69[8] + cols_helper_9_69[9] + cols_helper_9_69[10]
   + cols_helper_9_69[11] + cols_helper_9_69[12] + cols_helper_9_69[13]
   + cols_helper_9_69[14] + cols_helper_9_69[15] + cols_helper_9_69[16]
   <= 1
 cols_9_69: cols_9_69 - 2 cols_helper_9_69[0] - 4 cols_helper_9_69[1]
   - 8 cols_helper_9_69[2] - 16 cols_helper_9_69[3]
   - 32 cols_helper_9_69[4] - 64 cols_helper_9_69[5]
   - 128 cols_helper_9_69[6] - 256 cols_helper_9_69[7]
   - 512 cols_helper_9_69[8] - 1024 cols_helper_9_69[9]
   - 2048 cols_helper_9_69[10] - 4096 cols_helper_9_69[11]
   - 8192 cols_helper_9_69[12] - 16384 cols_helper_9_69[13]
   - 32768 cols_helper_9_69[14] - 65536 cols_helper_9_69[15]
   - 131072 cols_helper_9_69[16] = 0
 cols_acc_9_69: - 435 frac[9,69] + cols_9_69 >= 0
 accuracy_9_69: mem[9,69] - 0.01171875 cols_9_69 = 0
 accuracy_1_32: - 5.09765625 frac[1,32] + mem[1,32] = 0
 accuracy_2_39: - 5.09765625 frac[2,39] + mem[2,39] = 0
 cols_helper_10_44: cols_helper_10_44[0] + cols_helper_10_44[1]
   + cols_helper_10_44[2] + cols_helper_10_44[3] + cols_helper_10_44[4]
   + cols_helper_10_44[5] + cols_helper_10_44[6] + cols_helper_10_44[7]
   + cols_helper_10_44[8] + cols_helper_10_44[9] + cols_helper_10_44[10]
   + cols_helper_10_44[11] + cols_helper_10_44[12] + cols_helper_10_44[13]
   + cols_helper_10_44[14] + cols_helper_10_44[15] + cols_helper_10_44[16]
   <= 1
 cols_10_44: cols_10_44 - 2 cols_helper_10_44[0] - 4 cols_helper_10_44[1]
   - 8 cols_helper_10_44[2] - 16 cols_helper_10_44[3]
   - 32 cols_helper_10_44[4] - 64 cols_helper_10_44[5]
   - 128 cols_helper_10_44[6] - 256 cols_helper_10_44[7]
   - 512 cols_helper_10_44[8] - 1024 cols_helper_10_44[9]
   - 2048 cols_helper_10_44[10] - 4096 cols_helper_10_44[11]
   - 8192 cols_helper_10_44[12] - 16384 cols_helper_10_44[13]
   - 32768 cols_helper_10_44[14] - 65536 cols_helper_10_44[15]
   - 131072 cols_helper_10_44[16] = 0
 cols_acc_10_44: - 435 frac[10,44] + cols_10_44 >= 0
 accuracy_10_44: mem[10,44] - 0.01171875 cols_10_44 = 0
 accuracy_2_65: - 5.09765625 frac[2,65] + mem[2,65] = 0
 cols_helper_11_33: cols_helper_11_33[0] + cols_helper_11_33[1]
   + cols_helper_11_33[2] + cols_helper_11_33[3] + cols_helper_11_33[4]
   + cols_helper_11_33[5] + cols_helper_11_33[6] + cols_helper_11_33[7]
   + cols_helper_11_33[8] + cols_helper_11_33[9] + cols_helper_11_33[10]
   + cols_helper_11_33[11] + cols_helper_11_33[12] + cols_helper_11_33[13]
   + cols_helper_11_33[14] + cols_helper_11_33[15] + cols_helper_11_33[16]
   <= 1
 cols_11_33: cols_11_33 - 2 cols_helper_11_33[0] - 4 cols_helper_11_33[1]
   - 8 cols_helper_11_33[2] - 16 cols_helper_11_33[3]
   - 32 cols_helper_11_33[4] - 64 cols_helper_11_33[5]
   - 128 cols_helper_11_33[6] - 256 cols_helper_11_33[7]
   - 512 cols_helper_11_33[8] - 1024 cols_helper_11_33[9]
   - 2048 cols_helper_11_33[10] - 4096 cols_helper_11_33[11]
   - 8192 cols_helper_11_33[12] - 16384 cols_helper_11_33[13]
   - 32768 cols_helper_11_33[14] - 65536 cols_helper_11_33[15]
   - 131072 cols_helper_11_33[16] = 0
 cols_acc_11_33: - 435 frac[11,33] + cols_11_33 >= 0
 accuracy_11_33: mem[11,33] - 0.01171875 cols_11_33 = 0
 cols_helper_10_54: cols_helper_10_54[0] + cols_helper_10_54[1]
   + cols_helper_10_54[2] + cols_helper_10_54[3] + cols_helper_10_54[4]
   + cols_helper_10_54[5] + cols_helper_10_54[6] + cols_helper_10_54[7]
   + cols_helper_10_54[8] + cols_helper_10_54[9] + cols_helper_10_54[10]
   + cols_helper_10_54[11] + cols_helper_10_54[12] + cols_helper_10_54[13]
   + cols_helper_10_54[14] + cols_helper_10_54[15] + cols_helper_10_54[16]
   <= 1
 cols_10_54: cols_10_54 - 2 cols_helper_10_54[0] - 4 cols_helper_10_54[1]
   - 8 cols_helper_10_54[2] - 16 cols_helper_10_54[3]
   - 32 cols_helper_10_54[4] - 64 cols_helper_10_54[5]
   - 128 cols_helper_10_54[6] - 256 cols_helper_10_54[7]
   - 512 cols_helper_10_54[8] - 1024 cols_helper_10_54[9]
   - 2048 cols_helper_10_54[10] - 4096 cols_helper_10_54[11]
   - 8192 cols_helper_10_54[12] - 16384 cols_helper_10_54[13]
   - 32768 cols_helper_10_54[14] - 65536 cols_helper_10_54[15]
   - 131072 cols_helper_10_54[16] = 0
 cols_acc_10_54: - 435 frac[10,54] + cols_10_54 >= 0
 accuracy_10_54: mem[10,54] - 0.01171875 cols_10_54 = 0
 cols_helper_8_26: cols_helper_8_26[0] + cols_helper_8_26[1]
   + cols_helper_8_26[2] + cols_helper_8_26[3] + cols_helper_8_26[4]
   + cols_helper_8_26[5] + cols_helper_8_26[6] + cols_helper_8_26[7]
   + cols_helper_8_26[8] + cols_helper_8_26[9] + cols_helper_8_26[10]
   + cols_helper_8_26[11] + cols_helper_8_26[12] + cols_helper_8_26[13]
   + cols_helper_8_26[14] + cols_helper_8_26[15] + cols_helper_8_26[16]
   <= 1
 cols_8_26: cols_8_26 - 2 cols_helper_8_26[0] - 4 cols_helper_8_26[1]
   - 8 cols_helper_8_26[2] - 16 cols_helper_8_26[3]
   - 32 cols_helper_8_26[4] - 64 cols_helper_8_26[5]
   - 128 cols_helper_8_26[6] - 256 cols_helper_8_26[7]
   - 512 cols_helper_8_26[8] - 1024 cols_helper_8_26[9]
   - 2048 cols_helper_8_26[10] - 4096 cols_helper_8_26[11]
   - 8192 cols_helper_8_26[12] - 16384 cols_helper_8_26[13]
   - 32768 cols_helper_8_26[14] - 65536 cols_helper_8_26[15]
   - 131072 cols_helper_8_26[16] = 0
 cols_acc_8_26: - 435 frac[8,26] + cols_8_26 >= 0
 accuracy_8_26: mem[8,26] - 0.01171875 cols_8_26 = 0
 cols_helper_11_29: cols_helper_11_29[0] + cols_helper_11_29[1]
   + cols_helper_11_29[2] + cols_helper_11_29[3] + cols_helper_11_29[4]
   + cols_helper_11_29[5] + cols_helper_11_29[6] + cols_helper_11_29[7]
   + cols_helper_11_29[8] + cols_helper_11_29[9] + cols_helper_11_29[10]
   + cols_helper_11_29[11] + cols_helper_11_29[12] + cols_helper_11_29[13]
   + cols_helper_11_29[14] + cols_helper_11_29[15] + cols_helper_11_29[16]
   <= 1
 cols_11_29: cols_11_29 - 2 cols_helper_11_29[0] - 4 cols_helper_11_29[1]
   - 8 cols_helper_11_29[2] - 16 cols_helper_11_29[3]
   - 32 cols_helper_11_29[4] - 64 cols_helper_11_29[5]
   - 128 cols_helper_11_29[6] - 256 cols_helper_11_29[7]
   - 512 cols_helper_11_29[8] - 1024 cols_helper_11_29[9]
   - 2048 cols_helper_11_29[10] - 4096 cols_helper_11_29[11]
   - 8192 cols_helper_11_29[12] - 16384 cols_helper_11_29[13]
   - 32768 cols_helper_11_29[14] - 65536 cols_helper_11_29[15]
   - 131072 cols_helper_11_29[16] = 0
 cols_acc_11_29: - 435 frac[11,29] + cols_11_29 >= 0
 accuracy_11_29: mem[11,29] - 0.01171875 cols_11_29 = 0
 cols_helper_8_70: cols_helper_8_70[0] + cols_helper_8_70[1]
   + cols_helper_8_70[2] + cols_helper_8_70[3] + cols_helper_8_70[4]
   + cols_helper_8_70[5] + cols_helper_8_70[6] + cols_helper_8_70[7]
   + cols_helper_8_70[8] + cols_helper_8_70[9] + cols_helper_8_70[10]
   + cols_helper_8_70[11] + cols_helper_8_70[12] + cols_helper_8_70[13]
   + cols_helper_8_70[14] + cols_helper_8_70[15] + cols_helper_8_70[16]
   <= 1
 cols_8_70: cols_8_70 - 2 cols_helper_8_70[0] - 4 cols_helper_8_70[1]
   - 8 cols_helper_8_70[2] - 16 cols_helper_8_70[3]
   - 32 cols_helper_8_70[4] - 64 cols_helper_8_70[5]
   - 128 cols_helper_8_70[6] - 256 cols_helper_8_70[7]
   - 512 cols_helper_8_70[8] - 1024 cols_helper_8_70[9]
   - 2048 cols_helper_8_70[10] - 4096 cols_helper_8_70[11]
   - 8192 cols_helper_8_70[12] - 16384 cols_helper_8_70[13]
   - 32768 cols_helper_8_70[14] - 65536 cols_helper_8_70[15]
   - 131072 cols_helper_8_70[16] = 0
 cols_acc_8_70: - 435 frac[8,70] + cols_8_70 >= 0
 accuracy_8_70: mem[8,70] - 0.01171875 cols_8_70 = 0
 accuracy_5_19: - 5.09765625 frac[5,19] + mem[5,19] = 0
 cols_helper_11_64: cols_helper_11_64[0] + cols_helper_11_64[1]
   + cols_helper_11_64[2] + cols_helper_11_64[3] + cols_helper_11_64[4]
   + cols_helper_11_64[5] + cols_helper_11_64[6] + cols_helper_11_64[7]
   + cols_helper_11_64[8] + cols_helper_11_64[9] + cols_helper_11_64[10]
   + cols_helper_11_64[11] + cols_helper_11_64[12] + cols_helper_11_64[13]
   + cols_helper_11_64[14] + cols_helper_11_64[15] + cols_helper_11_64[16]
   <= 1
 cols_11_64: cols_11_64 - 2 cols_helper_11_64[0] - 4 cols_helper_11_64[1]
   - 8 cols_helper_11_64[2] - 16 cols_helper_11_64[3]
   - 32 cols_helper_11_64[4] - 64 cols_helper_11_64[5]
   - 128 cols_helper_11_64[6] - 256 cols_helper_11_64[7]
   - 512 cols_helper_11_64[8] - 1024 cols_helper_11_64[9]
   - 2048 cols_helper_11_64[10] - 4096 cols_helper_11_64[11]
   - 8192 cols_helper_11_64[12] - 16384 cols_helper_11_64[13]
   - 32768 cols_helper_11_64[14] - 65536 cols_helper_11_64[15]
   - 131072 cols_helper_11_64[16] = 0
 cols_acc_11_64: - 435 frac[11,64] + cols_11_64 >= 0
 accuracy_11_64: mem[11,64] - 0.01171875 cols_11_64 = 0
 cols_helper_8_59: cols_helper_8_59[0] + cols_helper_8_59[1]
   + cols_helper_8_59[2] + cols_helper_8_59[3] + cols_helper_8_59[4]
   + cols_helper_8_59[5] + cols_helper_8_59[6] + cols_helper_8_59[7]
   + cols_helper_8_59[8] + cols_helper_8_59[9] + cols_helper_8_59[10]
   + cols_helper_8_59[11] + cols_helper_8_59[12] + cols_helper_8_59[13]
   + cols_helper_8_59[14] + cols_helper_8_59[15] + cols_helper_8_59[16]
   <= 1
 cols_8_59: cols_8_59 - 2 cols_helper_8_59[0] - 4 cols_helper_8_59[1]
   - 8 cols_helper_8_59[2] - 16 cols_helper_8_59[3]
   - 32 cols_helper_8_59[4] - 64 cols_helper_8_59[5]
   - 128 cols_helper_8_59[6] - 256 cols_helper_8_59[7]
   - 512 cols_helper_8_59[8] - 1024 cols_helper_8_59[9]
   - 2048 cols_helper_8_59[10] - 4096 cols_helper_8_59[11]
   - 8192 cols_helper_8_59[12] - 16384 cols_helper_8_59[13]
   - 32768 cols_helper_8_59[14] - 65536 cols_helper_8_59[15]
   - 131072 cols_helper_8_59[16] = 0
 cols_acc_8_59: - 435 frac[8,59] + cols_8_59 >= 0
 accuracy_8_59: mem[8,59] - 0.01171875 cols_8_59 = 0
 cols_helper_11_38: cols_helper_11_38[0] + cols_helper_11_38[1]
   + cols_helper_11_38[2] + cols_helper_11_38[3] + cols_helper_11_38[4]
   + cols_helper_11_38[5] + cols_helper_11_38[6] + cols_helper_11_38[7]
   + cols_helper_11_38[8] + cols_helper_11_38[9] + cols_helper_11_38[10]
   + cols_helper_11_38[11] + cols_helper_11_38[12] + cols_helper_11_38[13]
   + cols_helper_11_38[14] + cols_helper_11_38[15] + cols_helper_11_38[16]
   <= 1
 cols_11_38: cols_11_38 - 2 cols_helper_11_38[0] - 4 cols_helper_11_38[1]
   - 8 cols_helper_11_38[2] - 16 cols_helper_11_38[3]
   - 32 cols_helper_11_38[4] - 64 cols_helper_11_38[5]
   - 128 cols_helper_11_38[6] - 256 cols_helper_11_38[7]
   - 512 cols_helper_11_38[8] - 1024 cols_helper_11_38[9]
   - 2048 cols_helper_11_38[10] - 4096 cols_helper_11_38[11]
   - 8192 cols_helper_11_38[12] - 16384 cols_helper_11_38[13]
   - 32768 cols_helper_11_38[14] - 65536 cols_helper_11_38[15]
   - 131072 cols_helper_11_38[16] = 0
 cols_acc_11_38: - 435 frac[11,38] + cols_11_38 >= 0
 accuracy_11_38: mem[11,38] - 0.01171875 cols_11_38 = 0
 cols_helper_9_58: cols_helper_9_58[0] + cols_helper_9_58[1]
   + cols_helper_9_58[2] + cols_helper_9_58[3] + cols_helper_9_58[4]
   + cols_helper_9_58[5] + cols_helper_9_58[6] + cols_helper_9_58[7]
   + cols_helper_9_58[8] + cols_helper_9_58[9] + cols_helper_9_58[10]
   + cols_helper_9_58[11] + cols_helper_9_58[12] + cols_helper_9_58[13]
   + cols_helper_9_58[14] + cols_helper_9_58[15] + cols_helper_9_58[16]
   <= 1
 cols_9_58: cols_9_58 - 2 cols_helper_9_58[0] - 4 cols_helper_9_58[1]
   - 8 cols_helper_9_58[2] - 16 cols_helper_9_58[3]
   - 32 cols_helper_9_58[4] - 64 cols_helper_9_58[5]
   - 128 cols_helper_9_58[6] - 256 cols_helper_9_58[7]
   - 512 cols_helper_9_58[8] - 1024 cols_helper_9_58[9]
   - 2048 cols_helper_9_58[10] - 4096 cols_helper_9_58[11]
   - 8192 cols_helper_9_58[12] - 16384 cols_helper_9_58[13]
   - 32768 cols_helper_9_58[14] - 65536 cols_helper_9_58[15]
   - 131072 cols_helper_9_58[16] = 0
 cols_acc_9_58: - 435 frac[9,58] + cols_9_58 >= 0
 accuracy_9_58: mem[9,58] - 0.01171875 cols_9_58 = 0
 cols_helper_10_49: cols_helper_10_49[0] + cols_helper_10_49[1]
   + cols_helper_10_49[2] + cols_helper_10_49[3] + cols_helper_10_49[4]
   + cols_helper_10_49[5] + cols_helper_10_49[6] + cols_helper_10_49[7]
   + cols_helper_10_49[8] + cols_helper_10_49[9] + cols_helper_10_49[10]
   + cols_helper_10_49[11] + cols_helper_10_49[12] + cols_helper_10_49[13]
   + cols_helper_10_49[14] + cols_helper_10_49[15] + cols_helper_10_49[16]
   <= 1
 cols_10_49: cols_10_49 - 2 cols_helper_10_49[0] - 4 cols_helper_10_49[1]
   - 8 cols_helper_10_49[2] - 16 cols_helper_10_49[3]
   - 32 cols_helper_10_49[4] - 64 cols_helper_10_49[5]
   - 128 cols_helper_10_49[6] - 256 cols_helper_10_49[7]
   - 512 cols_helper_10_49[8] - 1024 cols_helper_10_49[9]
   - 2048 cols_helper_10_49[10] - 4096 cols_helper_10_49[11]
   - 8192 cols_helper_10_49[12] - 16384 cols_helper_10_49[13]
   - 32768 cols_helper_10_49[14] - 65536 cols_helper_10_49[15]
   - 131072 cols_helper_10_49[16] = 0
 cols_acc_10_49: - 435 frac[10,49] + cols_10_49 >= 0
 accuracy_10_49: mem[10,49] - 0.01171875 cols_10_49 = 0
 cols_helper_11_60: cols_helper_11_60[0] + cols_helper_11_60[1]
   + cols_helper_11_60[2] + cols_helper_11_60[3] + cols_helper_11_60[4]
   + cols_helper_11_60[5] + cols_helper_11_60[6] + cols_helper_11_60[7]
   + cols_helper_11_60[8] + cols_helper_11_60[9] + cols_helper_11_60[10]
   + cols_helper_11_60[11] + cols_helper_11_60[12] + cols_helper_11_60[13]
   + cols_helper_11_60[14] + cols_helper_11_60[15] + cols_helper_11_60[16]
   <= 1
 cols_11_60: cols_11_60 - 2 cols_helper_11_60[0] - 4 cols_helper_11_60[1]
   - 8 cols_helper_11_60[2] - 16 cols_helper_11_60[3]
   - 32 cols_helper_11_60[4] - 64 cols_helper_11_60[5]
   - 128 cols_helper_11_60[6] - 256 cols_helper_11_60[7]
   - 512 cols_helper_11_60[8] - 1024 cols_helper_11_60[9]
   - 2048 cols_helper_11_60[10] - 4096 cols_helper_11_60[11]
   - 8192 cols_helper_11_60[12] - 16384 cols_helper_11_60[13]
   - 32768 cols_helper_11_60[14] - 65536 cols_helper_11_60[15]
   - 131072 cols_helper_11_60[16] = 0
 cols_acc_11_60: - 435 frac[11,60] + cols_11_60 >= 0
 accuracy_11_60: mem[11,60] - 0.01171875 cols_11_60 = 0
 cols_helper_9_28: cols_helper_9_28[0] + cols_helper_9_28[1]
   + cols_helper_9_28[2] + cols_helper_9_28[3] + cols_helper_9_28[4]
   + cols_helper_9_28[5] + cols_helper_9_28[6] + cols_helper_9_28[7]
   + cols_helper_9_28[8] + cols_helper_9_28[9] + cols_helper_9_28[10]
   + cols_helper_9_28[11] + cols_helper_9_28[12] + cols_helper_9_28[13]
   + cols_helper_9_28[14] + cols_helper_9_28[15] + cols_helper_9_28[16]
   <= 1
 cols_9_28: cols_9_28 - 2 cols_helper_9_28[0] - 4 cols_helper_9_28[1]
   - 8 cols_helper_9_28[2] - 16 cols_helper_9_28[3]
   - 32 cols_helper_9_28[4] - 64 cols_helper_9_28[5]
   - 128 cols_helper_9_28[6] - 256 cols_helper_9_28[7]
   - 512 cols_helper_9_28[8] - 1024 cols_helper_9_28[9]
   - 2048 cols_helper_9_28[10] - 4096 cols_helper_9_28[11]
   - 8192 cols_helper_9_28[12] - 16384 cols_helper_9_28[13]
   - 32768 cols_helper_9_28[14] - 65536 cols_helper_9_28[15]
   - 131072 cols_helper_9_28[16] = 0
 cols_acc_9_28: - 435 frac[9,28] + cols_9_28 >= 0
 accuracy_9_28: mem[9,28] - 0.01171875 cols_9_28 = 0
 cols_helper_10_27: cols_helper_10_27[0] + cols_helper_10_27[1]
   + cols_helper_10_27[2] + cols_helper_10_27[3] + cols_helper_10_27[4]
   + cols_helper_10_27[5] + cols_helper_10_27[6] + cols_helper_10_27[7]
   + cols_helper_10_27[8] + cols_helper_10_27[9] + cols_helper_10_27[10]
   + cols_helper_10_27[11] + cols_helper_10_27[12] + cols_helper_10_27[13]
   + cols_helper_10_27[14] + cols_helper_10_27[15] + cols_helper_10_27[16]
   <= 1
 cols_10_27: cols_10_27 - 2 cols_helper_10_27[0] - 4 cols_helper_10_27[1]
   - 8 cols_helper_10_27[2] - 16 cols_helper_10_27[3]
   - 32 cols_helper_10_27[4] - 64 cols_helper_10_27[5]
   - 128 cols_helper_10_27[6] - 256 cols_helper_10_27[7]
   - 512 cols_helper_10_27[8] - 1024 cols_helper_10_27[9]
   - 2048 cols_helper_10_27[10] - 4096 cols_helper_10_27[11]
   - 8192 cols_helper_10_27[12] - 16384 cols_helper_10_27[13]
   - 32768 cols_helper_10_27[14] - 65536 cols_helper_10_27[15]
   - 131072 cols_helper_10_27[16] = 0
 cols_acc_10_27: - 435 frac[10,27] + cols_10_27 >= 0
 accuracy_10_27: mem[10,27] - 0.01171875 cols_10_27 = 0
 cols_helper_11_18: cols_helper_11_18[0] + cols_helper_11_18[1]
   + cols_helper_11_18[2] + cols_helper_11_18[3] + cols_helper_11_18[4]
   + cols_helper_11_18[5] + cols_helper_11_18[6] + cols_helper_11_18[7]
   + cols_helper_11_18[8] + cols_helper_11_18[9] + cols_helper_11_18[10]
   + cols_helper_11_18[11] + cols_helper_11_18[12] + cols_helper_11_18[13]
   + cols_helper_11_18[14] + cols_helper_11_18[15] + cols_helper_11_18[16]
   <= 1
 cols_11_18: cols_11_18 - 2 cols_helper_11_18[0] - 4 cols_helper_11_18[1]
   - 8 cols_helper_11_18[2] - 16 cols_helper_11_18[3]
   - 32 cols_helper_11_18[4] - 64 cols_helper_11_18[5]
   - 128 cols_helper_11_18[6] - 256 cols_helper_11_18[7]
   - 512 cols_helper_11_18[8] - 1024 cols_helper_11_18[9]
   - 2048 cols_helper_11_18[10] - 4096 cols_helper_11_18[11]
   - 8192 cols_helper_11_18[12] - 16384 cols_helper_11_18[13]
   - 32768 cols_helper_11_18[14] - 65536 cols_helper_11_18[15]
   - 131072 cols_helper_11_18[16] = 0
 cols_acc_11_18: - 435 frac[11,18] + cols_11_18 >= 0
 accuracy_11_18: mem[11,18] - 0.01171875 cols_11_18 = 0
 accuracy_0_29: - 5.09765625 frac[0,29] + mem[0,29] = 0
 accuracy_4_25: - 5.09765625 frac[4,25] + mem[4,25] = 0
 accuracy_2_57: - 5.09765625 frac[2,57] + mem[2,57] = 0
 accuracy_3_36: - 5.09765625 frac[3,36] + mem[3,36] = 0
 accuracy_4_39: - 5.09765625 frac[4,39] + mem[4,39] = 0
 cols_helper_8_34: cols_helper_8_34[0] + cols_helper_8_34[1]
   + cols_helper_8_34[2] + cols_helper_8_34[3] + cols_helper_8_34[4]
   + cols_helper_8_34[5] + cols_helper_8_34[6] + cols_helper_8_34[7]
   + cols_helper_8_34[8] + cols_helper_8_34[9] + cols_helper_8_34[10]
   + cols_helper_8_34[11] + cols_helper_8_34[12] + cols_helper_8_34[13]
   + cols_helper_8_34[14] + cols_helper_8_34[15] + cols_helper_8_34[16]
   <= 1
 cols_8_34: cols_8_34 - 2 cols_helper_8_34[0] - 4 cols_helper_8_34[1]
   - 8 cols_helper_8_34[2] - 16 cols_helper_8_34[3]
   - 32 cols_helper_8_34[4] - 64 cols_helper_8_34[5]
   - 128 cols_helper_8_34[6] - 256 cols_helper_8_34[7]
   - 512 cols_helper_8_34[8] - 1024 cols_helper_8_34[9]
   - 2048 cols_helper_8_34[10] - 4096 cols_helper_8_34[11]
   - 8192 cols_helper_8_34[12] - 16384 cols_helper_8_34[13]
   - 32768 cols_helper_8_34[14] - 65536 cols_helper_8_34[15]
   - 131072 cols_helper_8_34[16] = 0
 cols_acc_8_34: - 435 frac[8,34] + cols_8_34 >= 0
 accuracy_8_34: mem[8,34] - 0.01171875 cols_8_34 = 0
 cols_helper_9_35: cols_helper_9_35[0] + cols_helper_9_35[1]
   + cols_helper_9_35[2] + cols_helper_9_35[3] + cols_helper_9_35[4]
   + cols_helper_9_35[5] + cols_helper_9_35[6] + cols_helper_9_35[7]
   + cols_helper_9_35[8] + cols_helper_9_35[9] + cols_helper_9_35[10]
   + cols_helper_9_35[11] + cols_helper_9_35[12] + cols_helper_9_35[13]
   + cols_helper_9_35[14] + cols_helper_9_35[15] + cols_helper_9_35[16]
   <= 1
 cols_9_35: cols_9_35 - 2 cols_helper_9_35[0] - 4 cols_helper_9_35[1]
   - 8 cols_helper_9_35[2] - 16 cols_helper_9_35[3]
   - 32 cols_helper_9_35[4] - 64 cols_helper_9_35[5]
   - 128 cols_helper_9_35[6] - 256 cols_helper_9_35[7]
   - 512 cols_helper_9_35[8] - 1024 cols_helper_9_35[9]
   - 2048 cols_helper_9_35[10] - 4096 cols_helper_9_35[11]
   - 8192 cols_helper_9_35[12] - 16384 cols_helper_9_35[13]
   - 32768 cols_helper_9_35[14] - 65536 cols_helper_9_35[15]
   - 131072 cols_helper_9_35[16] = 0
 cols_acc_9_35: - 435 frac[9,35] + cols_9_35 >= 0
 accuracy_9_35: mem[9,35] - 0.01171875 cols_9_35 = 0
 cols_helper_10_56: cols_helper_10_56[0] + cols_helper_10_56[1]
   + cols_helper_10_56[2] + cols_helper_10_56[3] + cols_helper_10_56[4]
   + cols_helper_10_56[5] + cols_helper_10_56[6] + cols_helper_10_56[7]
   + cols_helper_10_56[8] + cols_helper_10_56[9] + cols_helper_10_56[10]
   + cols_helper_10_56[11] + cols_helper_10_56[12] + cols_helper_10_56[13]
   + cols_helper_10_56[14] + cols_helper_10_56[15] + cols_helper_10_56[16]
   <= 1
 cols_10_56: cols_10_56 - 2 cols_helper_10_56[0] - 4 cols_helper_10_56[1]
   - 8 cols_helper_10_56[2] - 16 cols_helper_10_56[3]
   - 32 cols_helper_10_56[4] - 64 cols_helper_10_56[5]
   - 128 cols_helper_10_56[6] - 256 cols_helper_10_56[7]
   - 512 cols_helper_10_56[8] - 1024 cols_helper_10_56[9]
   - 2048 cols_helper_10_56[10] - 4096 cols_helper_10_56[11]
   - 8192 cols_helper_10_56[12] - 16384 cols_helper_10_56[13]
   - 32768 cols_helper_10_56[14] - 65536 cols_helper_10_56[15]
   - 131072 cols_helper_10_56[16] = 0
 cols_acc_10_56: - 435 frac[10,56] + cols_10_56 >= 0
 accuracy_10_56: mem[10,56] - 0.01171875 cols_10_56 = 0
 accuracy_1_69: - 5.09765625 frac[1,69] + mem[1,69] = 0
 cols_helper_11_53: cols_helper_11_53[0] + cols_helper_11_53[1]
   + cols_helper_11_53[2] + cols_helper_11_53[3] + cols_helper_11_53[4]
   + cols_helper_11_53[5] + cols_helper_11_53[6] + cols_helper_11_53[7]
   + cols_helper_11_53[8] + cols_helper_11_53[9] + cols_helper_11_53[10]
   + cols_helper_11_53[11] + cols_helper_11_53[12] + cols_helper_11_53[13]
   + cols_helper_11_53[14] + cols_helper_11_53[15] + cols_helper_11_53[16]
   <= 1
 cols_11_53: cols_11_53 - 2 cols_helper_11_53[0] - 4 cols_helper_11_53[1]
   - 8 cols_helper_11_53[2] - 16 cols_helper_11_53[3]
   - 32 cols_helper_11_53[4] - 64 cols_helper_11_53[5]
   - 128 cols_helper_11_53[6] - 256 cols_helper_11_53[7]
   - 512 cols_helper_11_53[8] - 1024 cols_helper_11_53[9]
   - 2048 cols_helper_11_53[10] - 4096 cols_helper_11_53[11]
   - 8192 cols_helper_11_53[12] - 16384 cols_helper_11_53[13]
   - 32768 cols_helper_11_53[14] - 65536 cols_helper_11_53[15]
   - 131072 cols_helper_11_53[16] = 0
 cols_acc_11_53: - 435 frac[11,53] + cols_11_53 >= 0
 accuracy_11_53: mem[11,53] - 0.01171875 cols_11_53 = 0
 cols_helper_9_5: cols_helper_9_5[0] + cols_helper_9_5[1]
   + cols_helper_9_5[2] + cols_helper_9_5[3] + cols_helper_9_5[4]
   + cols_helper_9_5[5] + cols_helper_9_5[6] + cols_helper_9_5[7]
   + cols_helper_9_5[8] + cols_helper_9_5[9] + cols_helper_9_5[10]
   + cols_helper_9_5[11] + cols_helper_9_5[12] + cols_helper_9_5[13]
   + cols_helper_9_5[14] + cols_helper_9_5[15] + cols_helper_9_5[16] <= 1
 cols_9_5: cols_9_5 - 2 cols_helper_9_5[0] - 4 cols_helper_9_5[1]
   - 8 cols_helper_9_5[2] - 16 cols_helper_9_5[3] - 32 cols_helper_9_5[4]
   - 64 cols_helper_9_5[5] - 128 cols_helper_9_5[6]
   - 256 cols_helper_9_5[7] - 512 cols_helper_9_5[8]
   - 1024 cols_helper_9_5[9] - 2048 cols_helper_9_5[10]
   - 4096 cols_helper_9_5[11] - 8192 cols_helper_9_5[12]
   - 16384 cols_helper_9_5[13] - 32768 cols_helper_9_5[14]
   - 65536 cols_helper_9_5[15] - 131072 cols_helper_9_5[16] = 0
 cols_acc_9_5: - 435 frac[9,5] + cols_9_5 >= 0
 accuracy_9_5: mem[9,5] - 0.01171875 cols_9_5 = 0
 cols_helper_10_2: cols_helper_10_2[0] + cols_helper_10_2[1]
   + cols_helper_10_2[2] + cols_helper_10_2[3] + cols_helper_10_2[4]
   + cols_helper_10_2[5] + cols_helper_10_2[6] + cols_helper_10_2[7]
   + cols_helper_10_2[8] + cols_helper_10_2[9] + cols_helper_10_2[10]
   + cols_helper_10_2[11] + cols_helper_10_2[12] + cols_helper_10_2[13]
   + cols_helper_10_2[14] + cols_helper_10_2[15] + cols_helper_10_2[16]
   <= 1
 cols_10_2: cols_10_2 - 2 cols_helper_10_2[0] - 4 cols_helper_10_2[1]
   - 8 cols_helper_10_2[2] - 16 cols_helper_10_2[3]
   - 32 cols_helper_10_2[4] - 64 cols_helper_10_2[5]
   - 128 cols_helper_10_2[6] - 256 cols_helper_10_2[7]
   - 512 cols_helper_10_2[8] - 1024 cols_helper_10_2[9]
   - 2048 cols_helper_10_2[10] - 4096 cols_helper_10_2[11]
   - 8192 cols_helper_10_2[12] - 16384 cols_helper_10_2[13]
   - 32768 cols_helper_10_2[14] - 65536 cols_helper_10_2[15]
   - 131072 cols_helper_10_2[16] = 0
 cols_acc_10_2: - 435 frac[10,2] + cols_10_2 >= 0
 accuracy_10_2: mem[10,2] - 0.01171875 cols_10_2 = 0
 cols_helper_11_11: cols_helper_11_11[0] + cols_helper_11_11[1]
   + cols_helper_11_11[2] + cols_helper_11_11[3] + cols_helper_11_11[4]
   + cols_helper_11_11[5] + cols_helper_11_11[6] + cols_helper_11_11[7]
   + cols_helper_11_11[8] + cols_helper_11_11[9] + cols_helper_11_11[10]
   + cols_helper_11_11[11] + cols_helper_11_11[12] + cols_helper_11_11[13]
   + cols_helper_11_11[14] + cols_helper_11_11[15] + cols_helper_11_11[16]
   <= 1
 cols_11_11: cols_11_11 - 2 cols_helper_11_11[0] - 4 cols_helper_11_11[1]
   - 8 cols_helper_11_11[2] - 16 cols_helper_11_11[3]
   - 32 cols_helper_11_11[4] - 64 cols_helper_11_11[5]
   - 128 cols_helper_11_11[6] - 256 cols_helper_11_11[7]
   - 512 cols_helper_11_11[8] - 1024 cols_helper_11_11[9]
   - 2048 cols_helper_11_11[10] - 4096 cols_helper_11_11[11]
   - 8192 cols_helper_11_11[12] - 16384 cols_helper_11_11[13]
   - 32768 cols_helper_11_11[14] - 65536 cols_helper_11_11[15]
   - 131072 cols_helper_11_11[16] = 0
 cols_acc_11_11: - 435 frac[11,11] + cols_11_11 >= 0
 accuracy_11_11: mem[11,11] - 0.01171875 cols_11_11 = 0
 accuracy_7_65: - 5.09765625 frac[7,65] + mem[7,65] = 0
 accuracy_4_0: - 5.09765625 frac[4,0] + mem[4,0] = 0
 accuracy_0_32: - 5.09765625 frac[0,32] + mem[0,32] = 0
 accuracy_5_5: - 5.09765625 frac[5,5] + mem[5,5] = 0
 cols_helper_9_66: cols_helper_9_66[0] + cols_helper_9_66[1]
   + cols_helper_9_66[2] + cols_helper_9_66[3] + cols_helper_9_66[4]
   + cols_helper_9_66[5] + cols_helper_9_66[6] + cols_helper_9_66[7]
   + cols_helper_9_66[8] + cols_helper_9_66[9] + cols_helper_9_66[10]
   + cols_helper_9_66[11] + cols_helper_9_66[12] + cols_helper_9_66[13]
   + cols_helper_9_66[14] + cols_helper_9_66[15] + cols_helper_9_66[16]
   <= 1
 cols_9_66: cols_9_66 - 2 cols_helper_9_66[0] - 4 cols_helper_9_66[1]
   - 8 cols_helper_9_66[2] - 16 cols_helper_9_66[3]
   - 32 cols_helper_9_66[4] - 64 cols_helper_9_66[5]
   - 128 cols_helper_9_66[6] - 256 cols_helper_9_66[7]
   - 512 cols_helper_9_66[8] - 1024 cols_helper_9_66[9]
   - 2048 cols_helper_9_66[10] - 4096 cols_helper_9_66[11]
   - 8192 cols_helper_9_66[12] - 16384 cols_helper_9_66[13]
   - 32768 cols_helper_9_66[14] - 65536 cols_helper_9_66[15]
   - 131072 cols_helper_9_66[16] = 0
 cols_acc_9_66: - 435 frac[9,66] + cols_9_66 >= 0
 accuracy_9_66: mem[9,66] - 0.01171875 cols_9_66 = 0
 cols_helper_8_41: cols_helper_8_41[0] + cols_helper_8_41[1]
   + cols_helper_8_41[2] + cols_helper_8_41[3] + cols_helper_8_41[4]
   + cols_helper_8_41[5] + cols_helper_8_41[6] + cols_helper_8_41[7]
   + cols_helper_8_41[8] + cols_helper_8_41[9] + cols_helper_8_41[10]
   + cols_helper_8_41[11] + cols_helper_8_41[12] + cols_helper_8_41[13]
   + cols_helper_8_41[14] + cols_helper_8_41[15] + cols_helper_8_41[16]
   <= 1
 cols_8_41: cols_8_41 - 2 cols_helper_8_41[0] - 4 cols_helper_8_41[1]
   - 8 cols_helper_8_41[2] - 16 cols_helper_8_41[3]
   - 32 cols_helper_8_41[4] - 64 cols_helper_8_41[5]
   - 128 cols_helper_8_41[6] - 256 cols_helper_8_41[7]
   - 512 cols_helper_8_41[8] - 1024 cols_helper_8_41[9]
   - 2048 cols_helper_8_41[10] - 4096 cols_helper_8_41[11]
   - 8192 cols_helper_8_41[12] - 16384 cols_helper_8_41[13]
   - 32768 cols_helper_8_41[14] - 65536 cols_helper_8_41[15]
   - 131072 cols_helper_8_41[16] = 0
 cols_acc_8_41: - 435 frac[8,41] + cols_8_41 >= 0
 accuracy_8_41: mem[8,41] - 0.01171875 cols_8_41 = 0
 cols_helper_10_35: cols_helper_10_35[0] + cols_helper_10_35[1]
   + cols_helper_10_35[2] + cols_helper_10_35[3] + cols_helper_10_35[4]
   + cols_helper_10_35[5] + cols_helper_10_35[6] + cols_helper_10_35[7]
   + cols_helper_10_35[8] + cols_helper_10_35[9] + cols_helper_10_35[10]
   + cols_helper_10_35[11] + cols_helper_10_35[12] + cols_helper_10_35[13]
   + cols_helper_10_35[14] + cols_helper_10_35[15] + cols_helper_10_35[16]
   <= 1
 cols_10_35: cols_10_35 - 2 cols_helper_10_35[0] - 4 cols_helper_10_35[1]
   - 8 cols_helper_10_35[2] - 16 cols_helper_10_35[3]
   - 32 cols_helper_10_35[4] - 64 cols_helper_10_35[5]
   - 128 cols_helper_10_35[6] - 256 cols_helper_10_35[7]
   - 512 cols_helper_10_35[8] - 1024 cols_helper_10_35[9]
   - 2048 cols_helper_10_35[10] - 4096 cols_helper_10_35[11]
   - 8192 cols_helper_10_35[12] - 16384 cols_helper_10_35[13]
   - 32768 cols_helper_10_35[14] - 65536 cols_helper_10_35[15]
   - 131072 cols_helper_10_35[16] = 0
 cols_acc_10_35: - 435 frac[10,35] + cols_10_35 >= 0
 accuracy_10_35: mem[10,35] - 0.01171875 cols_10_35 = 0
 cols_helper_8_15: cols_helper_8_15[0] + cols_helper_8_15[1]
   + cols_helper_8_15[2] + cols_helper_8_15[3] + cols_helper_8_15[4]
   + cols_helper_8_15[5] + cols_helper_8_15[6] + cols_helper_8_15[7]
   + cols_helper_8_15[8] + cols_helper_8_15[9] + cols_helper_8_15[10]
   + cols_helper_8_15[11] + cols_helper_8_15[12] + cols_helper_8_15[13]
   + cols_helper_8_15[14] + cols_helper_8_15[15] + cols_helper_8_15[16]
   <= 1
 cols_8_15: cols_8_15 - 2 cols_helper_8_15[0] - 4 cols_helper_8_15[1]
   - 8 cols_helper_8_15[2] - 16 cols_helper_8_15[3]
   - 32 cols_helper_8_15[4] - 64 cols_helper_8_15[5]
   - 128 cols_helper_8_15[6] - 256 cols_helper_8_15[7]
   - 512 cols_helper_8_15[8] - 1024 cols_helper_8_15[9]
   - 2048 cols_helper_8_15[10] - 4096 cols_helper_8_15[11]
   - 8192 cols_helper_8_15[12] - 16384 cols_helper_8_15[13]
   - 32768 cols_helper_8_15[14] - 65536 cols_helper_8_15[15]
   - 131072 cols_helper_8_15[16] = 0
 cols_acc_8_15: - 435 frac[8,15] + cols_8_15 >= 0
 accuracy_8_15: mem[8,15] - 0.01171875 cols_8_15 = 0
 cols_helper_9_14: cols_helper_9_14[0] + cols_helper_9_14[1]
   + cols_helper_9_14[2] + cols_helper_9_14[3] + cols_helper_9_14[4]
   + cols_helper_9_14[5] + cols_helper_9_14[6] + cols_helper_9_14[7]
   + cols_helper_9_14[8] + cols_helper_9_14[9] + cols_helper_9_14[10]
   + cols_helper_9_14[11] + cols_helper_9_14[12] + cols_helper_9_14[13]
   + cols_helper_9_14[14] + cols_helper_9_14[15] + cols_helper_9_14[16]
   <= 1
 cols_9_14: cols_9_14 - 2 cols_helper_9_14[0] - 4 cols_helper_9_14[1]
   - 8 cols_helper_9_14[2] - 16 cols_helper_9_14[3]
   - 32 cols_helper_9_14[4] - 64 cols_helper_9_14[5]
   - 128 cols_helper_9_14[6] - 256 cols_helper_9_14[7]
   - 512 cols_helper_9_14[8] - 1024 cols_helper_9_14[9]
   - 2048 cols_helper_9_14[10] - 4096 cols_helper_9_14[11]
   - 8192 cols_helper_9_14[12] - 16384 cols_helper_9_14[13]
   - 32768 cols_helper_9_14[14] - 65536 cols_helper_9_14[15]
   - 131072 cols_helper_9_14[16] = 0
 cols_acc_9_14: - 435 frac[9,14] + cols_9_14 >= 0
 accuracy_9_14: mem[9,14] - 0.01171875 cols_9_14 = 0
 cols_helper_10_13: cols_helper_10_13[0] + cols_helper_10_13[1]
   + cols_helper_10_13[2] + cols_helper_10_13[3] + cols_helper_10_13[4]
   + cols_helper_10_13[5] + cols_helper_10_13[6] + cols_helper_10_13[7]
   + cols_helper_10_13[8] + cols_helper_10_13[9] + cols_helper_10_13[10]
   + cols_helper_10_13[11] + cols_helper_10_13[12] + cols_helper_10_13[13]
   + cols_helper_10_13[14] + cols_helper_10_13[15] + cols_helper_10_13[16]
   <= 1
 cols_10_13: cols_10_13 - 2 cols_helper_10_13[0] - 4 cols_helper_10_13[1]
   - 8 cols_helper_10_13[2] - 16 cols_helper_10_13[3]
   - 32 cols_helper_10_13[4] - 64 cols_helper_10_13[5]
   - 128 cols_helper_10_13[6] - 256 cols_helper_10_13[7]
   - 512 cols_helper_10_13[8] - 1024 cols_helper_10_13[9]
   - 2048 cols_helper_10_13[10] - 4096 cols_helper_10_13[11]
   - 8192 cols_helper_10_13[12] - 16384 cols_helper_10_13[13]
   - 32768 cols_helper_10_13[14] - 65536 cols_helper_10_13[15]
   - 131072 cols_helper_10_13[16] = 0
 cols_acc_10_13: - 435 frac[10,13] + cols_10_13 >= 0
 accuracy_10_13: mem[10,13] - 0.01171875 cols_10_13 = 0
 cols_helper_8_21: cols_helper_8_21[0] + cols_helper_8_21[1]
   + cols_helper_8_21[2] + cols_helper_8_21[3] + cols_helper_8_21[4]
   + cols_helper_8_21[5] + cols_helper_8_21[6] + cols_helper_8_21[7]
   + cols_helper_8_21[8] + cols_helper_8_21[9] + cols_helper_8_21[10]
   + cols_helper_8_21[11] + cols_helper_8_21[12] + cols_helper_8_21[13]
   + cols_helper_8_21[14] + cols_helper_8_21[15] + cols_helper_8_21[16]
   <= 1
 cols_8_21: cols_8_21 - 2 cols_helper_8_21[0] - 4 cols_helper_8_21[1]
   - 8 cols_helper_8_21[2] - 16 cols_helper_8_21[3]
   - 32 cols_helper_8_21[4] - 64 cols_helper_8_21[5]
   - 128 cols_helper_8_21[6] - 256 cols_helper_8_21[7]
   - 512 cols_helper_8_21[8] - 1024 cols_helper_8_21[9]
   - 2048 cols_helper_8_21[10] - 4096 cols_helper_8_21[11]
   - 8192 cols_helper_8_21[12] - 16384 cols_helper_8_21[13]
   - 32768 cols_helper_8_21[14] - 65536 cols_helper_8_21[15]
   - 131072 cols_helper_8_21[16] = 0
 cols_acc_8_21: - 435 frac[8,21] + cols_8_21 >= 0
 accuracy_8_21: mem[8,21] - 0.01171875 cols_8_21 = 0
 cols_helper_11_0: cols_helper_11_0[0] + cols_helper_11_0[1]
   + cols_helper_11_0[2] + cols_helper_11_0[3] + cols_helper_11_0[4]
   + cols_helper_11_0[5] + cols_helper_11_0[6] + cols_helper_11_0[7]
   + cols_helper_11_0[8] + cols_helper_11_0[9] + cols_helper_11_0[10]
   + cols_helper_11_0[11] + cols_helper_11_0[12] + cols_helper_11_0[13]
   + cols_helper_11_0[14] + cols_helper_11_0[15] + cols_helper_11_0[16]
   <= 1
 cols_11_0: cols_11_0 - 2 cols_helper_11_0[0] - 4 cols_helper_11_0[1]
   - 8 cols_helper_11_0[2] - 16 cols_helper_11_0[3]
   - 32 cols_helper_11_0[4] - 64 cols_helper_11_0[5]
   - 128 cols_helper_11_0[6] - 256 cols_helper_11_0[7]
   - 512 cols_helper_11_0[8] - 1024 cols_helper_11_0[9]
   - 2048 cols_helper_11_0[10] - 4096 cols_helper_11_0[11]
   - 8192 cols_helper_11_0[12] - 16384 cols_helper_11_0[13]
   - 32768 cols_helper_11_0[14] - 65536 cols_helper_11_0[15]
   - 131072 cols_helper_11_0[16] = 0
 cols_acc_11_0: - 435 frac[11,0] + cols_11_0 >= 0
 accuracy_11_0: mem[11,0] - 0.01171875 cols_11_0 = 0
 cols_helper_10_23: cols_helper_10_23[0] + cols_helper_10_23[1]
   + cols_helper_10_23[2] + cols_helper_10_23[3] + cols_helper_10_23[4]
   + cols_helper_10_23[5] + cols_helper_10_23[6] + cols_helper_10_23[7]
   + cols_helper_10_23[8] + cols_helper_10_23[9] + cols_helper_10_23[10]
   + cols_helper_10_23[11] + cols_helper_10_23[12] + cols_helper_10_23[13]
   + cols_helper_10_23[14] + cols_helper_10_23[15] + cols_helper_10_23[16]
   <= 1
 cols_10_23: cols_10_23 - 2 cols_helper_10_23[0] - 4 cols_helper_10_23[1]
   - 8 cols_helper_10_23[2] - 16 cols_helper_10_23[3]
   - 32 cols_helper_10_23[4] - 64 cols_helper_10_23[5]
   - 128 cols_helper_10_23[6] - 256 cols_helper_10_23[7]
   - 512 cols_helper_10_23[8] - 1024 cols_helper_10_23[9]
   - 2048 cols_helper_10_23[10] - 4096 cols_helper_10_23[11]
   - 8192 cols_helper_10_23[12] - 16384 cols_helper_10_23[13]
   - 32768 cols_helper_10_23[14] - 65536 cols_helper_10_23[15]
   - 131072 cols_helper_10_23[16] = 0
 cols_acc_10_23: - 435 frac[10,23] + cols_10_23 >= 0
 accuracy_10_23: mem[10,23] - 0.01171875 cols_10_23 = 0
 accuracy_3_12: - 5.09765625 frac[3,12] + mem[3,12] = 0
 cols_helper_8_65: cols_helper_8_65[0] + cols_helper_8_65[1]
   + cols_helper_8_65[2] + cols_helper_8_65[3] + cols_helper_8_65[4]
   + cols_helper_8_65[5] + cols_helper_8_65[6] + cols_helper_8_65[7]
   + cols_helper_8_65[8] + cols_helper_8_65[9] + cols_helper_8_65[10]
   + cols_helper_8_65[11] + cols_helper_8_65[12] + cols_helper_8_65[13]
   + cols_helper_8_65[14] + cols_helper_8_65[15] + cols_helper_8_65[16]
   <= 1
 cols_8_65: cols_8_65 - 2 cols_helper_8_65[0] - 4 cols_helper_8_65[1]
   - 8 cols_helper_8_65[2] - 16 cols_helper_8_65[3]
   - 32 cols_helper_8_65[4] - 64 cols_helper_8_65[5]
   - 128 cols_helper_8_65[6] - 256 cols_helper_8_65[7]
   - 512 cols_helper_8_65[8] - 1024 cols_helper_8_65[9]
   - 2048 cols_helper_8_65[10] - 4096 cols_helper_8_65[11]
   - 8192 cols_helper_8_65[12] - 16384 cols_helper_8_65[13]
   - 32768 cols_helper_8_65[14] - 65536 cols_helper_8_65[15]
   - 131072 cols_helper_8_65[16] = 0
 cols_acc_8_65: - 435 frac[8,65] + cols_8_65 >= 0
 accuracy_8_65: mem[8,65] - 0.01171875 cols_8_65 = 0
 accuracy_4_21: - 5.09765625 frac[4,21] + mem[4,21] = 0
 cols_helper_9_45: cols_helper_9_45[0] + cols_helper_9_45[1]
   + cols_helper_9_45[2] + cols_helper_9_45[3] + cols_helper_9_45[4]
   + cols_helper_9_45[5] + cols_helper_9_45[6] + cols_helper_9_45[7]
   + cols_helper_9_45[8] + cols_helper_9_45[9] + cols_helper_9_45[10]
   + cols_helper_9_45[11] + cols_helper_9_45[12] + cols_helper_9_45[13]
   + cols_helper_9_45[14] + cols_helper_9_45[15] + cols_helper_9_45[16]
   <= 1
 cols_9_45: cols_9_45 - 2 cols_helper_9_45[0] - 4 cols_helper_9_45[1]
   - 8 cols_helper_9_45[2] - 16 cols_helper_9_45[3]
   - 32 cols_helper_9_45[4] - 64 cols_helper_9_45[5]
   - 128 cols_helper_9_45[6] - 256 cols_helper_9_45[7]
   - 512 cols_helper_9_45[8] - 1024 cols_helper_9_45[9]
   - 2048 cols_helper_9_45[10] - 4096 cols_helper_9_45[11]
   - 8192 cols_helper_9_45[12] - 16384 cols_helper_9_45[13]
   - 32768 cols_helper_9_45[14] - 65536 cols_helper_9_45[15]
   - 131072 cols_helper_9_45[16] = 0
 cols_acc_9_45: - 435 frac[9,45] + cols_9_45 >= 0
 accuracy_9_45: mem[9,45] - 0.01171875 cols_9_45 = 0
 accuracy_6_39: - 5.09765625 frac[6,39] + mem[6,39] = 0
 cols_helper_8_54: cols_helper_8_54[0] + cols_helper_8_54[1]
   + cols_helper_8_54[2] + cols_helper_8_54[3] + cols_helper_8_54[4]
   + cols_helper_8_54[5] + cols_helper_8_54[6] + cols_helper_8_54[7]
   + cols_helper_8_54[8] + cols_helper_8_54[9] + cols_helper_8_54[10]
   + cols_helper_8_54[11] + cols_helper_8_54[12] + cols_helper_8_54[13]
   + cols_helper_8_54[14] + cols_helper_8_54[15] + cols_helper_8_54[16]
   <= 1
 cols_8_54: cols_8_54 - 2 cols_helper_8_54[0] - 4 cols_helper_8_54[1]
   - 8 cols_helper_8_54[2] - 16 cols_helper_8_54[3]
   - 32 cols_helper_8_54[4] - 64 cols_helper_8_54[5]
   - 128 cols_helper_8_54[6] - 256 cols_helper_8_54[7]
   - 512 cols_helper_8_54[8] - 1024 cols_helper_8_54[9]
   - 2048 cols_helper_8_54[10] - 4096 cols_helper_8_54[11]
   - 8192 cols_helper_8_54[12] - 16384 cols_helper_8_54[13]
   - 32768 cols_helper_8_54[14] - 65536 cols_helper_8_54[15]
   - 131072 cols_helper_8_54[16] = 0
 cols_acc_8_54: - 435 frac[8,54] + cols_8_54 >= 0
 accuracy_8_54: mem[8,54] - 0.01171875 cols_8_54 = 0
 cols_helper_11_35: cols_helper_11_35[0] + cols_helper_11_35[1]
   + cols_helper_11_35[2] + cols_helper_11_35[3] + cols_helper_11_35[4]
   + cols_helper_11_35[5] + cols_helper_11_35[6] + cols_helper_11_35[7]
   + cols_helper_11_35[8] + cols_helper_11_35[9] + cols_helper_11_35[10]
   + cols_helper_11_35[11] + cols_helper_11_35[12] + cols_helper_11_35[13]
   + cols_helper_11_35[14] + cols_helper_11_35[15] + cols_helper_11_35[16]
   <= 1
 cols_11_35: cols_11_35 - 2 cols_helper_11_35[0] - 4 cols_helper_11_35[1]
   - 8 cols_helper_11_35[2] - 16 cols_helper_11_35[3]
   - 32 cols_helper_11_35[4] - 64 cols_helper_11_35[5]
   - 128 cols_helper_11_35[6] - 256 cols_helper_11_35[7]
   - 512 cols_helper_11_35[8] - 1024 cols_helper_11_35[9]
   - 2048 cols_helper_11_35[10] - 4096 cols_helper_11_35[11]
   - 8192 cols_helper_11_35[12] - 16384 cols_helper_11_35[13]
   - 32768 cols_helper_11_35[14] - 65536 cols_helper_11_35[15]
   - 131072 cols_helper_11_35[16] = 0
 cols_acc_11_35: - 435 frac[11,35] + cols_11_35 >= 0
 accuracy_11_35: mem[11,35] - 0.01171875 cols_11_35 = 0
 cols_helper_9_55: cols_helper_9_55[0] + cols_helper_9_55[1]
   + cols_helper_9_55[2] + cols_helper_9_55[3] + cols_helper_9_55[4]
   + cols_helper_9_55[5] + cols_helper_9_55[6] + cols_helper_9_55[7]
   + cols_helper_9_55[8] + cols_helper_9_55[9] + cols_helper_9_55[10]
   + cols_helper_9_55[11] + cols_helper_9_55[12] + cols_helper_9_55[13]
   + cols_helper_9_55[14] + cols_helper_9_55[15] + cols_helper_9_55[16]
   <= 1
 cols_9_55: cols_9_55 - 2 cols_helper_9_55[0] - 4 cols_helper_9_55[1]
   - 8 cols_helper_9_55[2] - 16 cols_helper_9_55[3]
   - 32 cols_helper_9_55[4] - 64 cols_helper_9_55[5]
   - 128 cols_helper_9_55[6] - 256 cols_helper_9_55[7]
   - 512 cols_helper_9_55[8] - 1024 cols_helper_9_55[9]
   - 2048 cols_helper_9_55[10] - 4096 cols_helper_9_55[11]
   - 8192 cols_helper_9_55[12] - 16384 cols_helper_9_55[13]
   - 32768 cols_helper_9_55[14] - 65536 cols_helper_9_55[15]
   - 131072 cols_helper_9_55[16] = 0
 cols_acc_9_55: - 435 frac[9,55] + cols_9_55 >= 0
 accuracy_9_55: mem[9,55] - 0.01171875 cols_9_55 = 0
 cols_helper_10_52: cols_helper_10_52[0] + cols_helper_10_52[1]
   + cols_helper_10_52[2] + cols_helper_10_52[3] + cols_helper_10_52[4]
   + cols_helper_10_52[5] + cols_helper_10_52[6] + cols_helper_10_52[7]
   + cols_helper_10_52[8] + cols_helper_10_52[9] + cols_helper_10_52[10]
   + cols_helper_10_52[11] + cols_helper_10_52[12] + cols_helper_10_52[13]
   + cols_helper_10_52[14] + cols_helper_10_52[15] + cols_helper_10_52[16]
   <= 1
 cols_10_52: cols_10_52 - 2 cols_helper_10_52[0] - 4 cols_helper_10_52[1]
   - 8 cols_helper_10_52[2] - 16 cols_helper_10_52[3]
   - 32 cols_helper_10_52[4] - 64 cols_helper_10_52[5]
   - 128 cols_helper_10_52[6] - 256 cols_helper_10_52[7]
   - 512 cols_helper_10_52[8] - 1024 cols_helper_10_52[9]
   - 2048 cols_helper_10_52[10] - 4096 cols_helper_10_52[11]
   - 8192 cols_helper_10_52[12] - 16384 cols_helper_10_52[13]
   - 32768 cols_helper_10_52[14] - 65536 cols_helper_10_52[15]
   - 131072 cols_helper_10_52[16] = 0
 cols_acc_10_52: - 435 frac[10,52] + cols_10_52 >= 0
 accuracy_10_52: mem[10,52] - 0.01171875 cols_10_52 = 0
 cols_helper_8_28: cols_helper_8_28[0] + cols_helper_8_28[1]
   + cols_helper_8_28[2] + cols_helper_8_28[3] + cols_helper_8_28[4]
   + cols_helper_8_28[5] + cols_helper_8_28[6] + cols_helper_8_28[7]
   + cols_helper_8_28[8] + cols_helper_8_28[9] + cols_helper_8_28[10]
   + cols_helper_8_28[11] + cols_helper_8_28[12] + cols_helper_8_28[13]
   + cols_helper_8_28[14] + cols_helper_8_28[15] + cols_helper_8_28[16]
   <= 1
 cols_8_28: cols_8_28 - 2 cols_helper_8_28[0] - 4 cols_helper_8_28[1]
   - 8 cols_helper_8_28[2] - 16 cols_helper_8_28[3]
   - 32 cols_helper_8_28[4] - 64 cols_helper_8_28[5]
   - 128 cols_helper_8_28[6] - 256 cols_helper_8_28[7]
   - 512 cols_helper_8_28[8] - 1024 cols_helper_8_28[9]
   - 2048 cols_helper_8_28[10] - 4096 cols_helper_8_28[11]
   - 8192 cols_helper_8_28[12] - 16384 cols_helper_8_28[13]
   - 32768 cols_helper_8_28[14] - 65536 cols_helper_8_28[15]
   - 131072 cols_helper_8_28[16] = 0
 cols_acc_8_28: - 435 frac[8,28] + cols_8_28 >= 0
 accuracy_8_28: mem[8,28] - 0.01171875 cols_8_28 = 0
 cols_helper_11_57: cols_helper_11_57[0] + cols_helper_11_57[1]
   + cols_helper_11_57[2] + cols_helper_11_57[3] + cols_helper_11_57[4]
   + cols_helper_11_57[5] + cols_helper_11_57[6] + cols_helper_11_57[7]
   + cols_helper_11_57[8] + cols_helper_11_57[9] + cols_helper_11_57[10]
   + cols_helper_11_57[11] + cols_helper_11_57[12] + cols_helper_11_57[13]
   + cols_helper_11_57[14] + cols_helper_11_57[15] + cols_helper_11_57[16]
   <= 1
 cols_11_57: cols_11_57 - 2 cols_helper_11_57[0] - 4 cols_helper_11_57[1]
   - 8 cols_helper_11_57[2] - 16 cols_helper_11_57[3]
   - 32 cols_helper_11_57[4] - 64 cols_helper_11_57[5]
   - 128 cols_helper_11_57[6] - 256 cols_helper_11_57[7]
   - 512 cols_helper_11_57[8] - 1024 cols_helper_11_57[9]
   - 2048 cols_helper_11_57[10] - 4096 cols_helper_11_57[11]
   - 8192 cols_helper_11_57[12] - 16384 cols_helper_11_57[13]
   - 32768 cols_helper_11_57[14] - 65536 cols_helper_11_57[15]
   - 131072 cols_helper_11_57[16] = 0
 cols_acc_11_57: - 435 frac[11,57] + cols_11_57 >= 0
 accuracy_11_57: mem[11,57] - 0.01171875 cols_11_57 = 0
 cols_helper_9_25: cols_helper_9_25[0] + cols_helper_9_25[1]
   + cols_helper_9_25[2] + cols_helper_9_25[3] + cols_helper_9_25[4]
   + cols_helper_9_25[5] + cols_helper_9_25[6] + cols_helper_9_25[7]
   + cols_helper_9_25[8] + cols_helper_9_25[9] + cols_helper_9_25[10]
   + cols_helper_9_25[11] + cols_helper_9_25[12] + cols_helper_9_25[13]
   + cols_helper_9_25[14] + cols_helper_9_25[15] + cols_helper_9_25[16]
   <= 1
 cols_9_25: cols_9_25 - 2 cols_helper_9_25[0] - 4 cols_helper_9_25[1]
   - 8 cols_helper_9_25[2] - 16 cols_helper_9_25[3]
   - 32 cols_helper_9_25[4] - 64 cols_helper_9_25[5]
   - 128 cols_helper_9_25[6] - 256 cols_helper_9_25[7]
   - 512 cols_helper_9_25[8] - 1024 cols_helper_9_25[9]
   - 2048 cols_helper_9_25[10] - 4096 cols_helper_9_25[11]
   - 8192 cols_helper_9_25[12] - 16384 cols_helper_9_25[13]
   - 32768 cols_helper_9_25[14] - 65536 cols_helper_9_25[15]
   - 131072 cols_helper_9_25[16] = 0
 cols_acc_9_25: - 435 frac[9,25] + cols_9_25 >= 0
 accuracy_9_25: mem[9,25] - 0.01171875 cols_9_25 = 0
 cols_helper_10_30: cols_helper_10_30[0] + cols_helper_10_30[1]
   + cols_helper_10_30[2] + cols_helper_10_30[3] + cols_helper_10_30[4]
   + cols_helper_10_30[5] + cols_helper_10_30[6] + cols_helper_10_30[7]
   + cols_helper_10_30[8] + cols_helper_10_30[9] + cols_helper_10_30[10]
   + cols_helper_10_30[11] + cols_helper_10_30[12] + cols_helper_10_30[13]
   + cols_helper_10_30[14] + cols_helper_10_30[15] + cols_helper_10_30[16]
   <= 1
 cols_10_30: cols_10_30 - 2 cols_helper_10_30[0] - 4 cols_helper_10_30[1]
   - 8 cols_helper_10_30[2] - 16 cols_helper_10_30[3]
   - 32 cols_helper_10_30[4] - 64 cols_helper_10_30[5]
   - 128 cols_helper_10_30[6] - 256 cols_helper_10_30[7]
   - 512 cols_helper_10_30[8] - 1024 cols_helper_10_30[9]
   - 2048 cols_helper_10_30[10] - 4096 cols_helper_10_30[11]
   - 8192 cols_helper_10_30[12] - 16384 cols_helper_10_30[13]
   - 32768 cols_helper_10_30[14] - 65536 cols_helper_10_30[15]
   - 131072 cols_helper_10_30[16] = 0
 cols_acc_10_30: - 435 frac[10,30] + cols_10_30 >= 0
 accuracy_10_30: mem[10,30] - 0.01171875 cols_10_30 = 0
 cols_helper_11_31: cols_helper_11_31[0] + cols_helper_11_31[1]
   + cols_helper_11_31[2] + cols_helper_11_31[3] + cols_helper_11_31[4]
   + cols_helper_11_31[5] + cols_helper_11_31[6] + cols_helper_11_31[7]
   + cols_helper_11_31[8] + cols_helper_11_31[9] + cols_helper_11_31[10]
   + cols_helper_11_31[11] + cols_helper_11_31[12] + cols_helper_11_31[13]
   + cols_helper_11_31[14] + cols_helper_11_31[15] + cols_helper_11_31[16]
   <= 1
 cols_11_31: cols_11_31 - 2 cols_helper_11_31[0] - 4 cols_helper_11_31[1]
   - 8 cols_helper_11_31[2] - 16 cols_helper_11_31[3]
   - 32 cols_helper_11_31[4] - 64 cols_helper_11_31[5]
   - 128 cols_helper_11_31[6] - 256 cols_helper_11_31[7]
   - 512 cols_helper_11_31[8] - 1024 cols_helper_11_31[9]
   - 2048 cols_helper_11_31[10] - 4096 cols_helper_11_31[11]
   - 8192 cols_helper_11_31[12] - 16384 cols_helper_11_31[13]
   - 32768 cols_helper_11_31[14] - 65536 cols_helper_11_31[15]
   - 131072 cols_helper_11_31[16] = 0
 cols_acc_11_31: - 435 frac[11,31] + cols_11_31 >= 0
 accuracy_11_31: mem[11,31] - 0.01171875 cols_11_31 = 0
 accuracy_1_5: - 5.09765625 frac[1,5] + mem[1,5] = 0
 cols_helper_8_72: cols_helper_8_72[0] + cols_helper_8_72[1]
   + cols_helper_8_72[2] + cols_helper_8_72[3] + cols_helper_8_72[4]
   + cols_helper_8_72[5] + cols_helper_8_72[6] + cols_helper_8_72[7]
   + cols_helper_8_72[8] + cols_helper_8_72[9] + cols_helper_8_72[10]
   + cols_helper_8_72[11] + cols_helper_8_72[12] + cols_helper_8_72[13]
   + cols_helper_8_72[14] + cols_helper_8_72[15] + cols_helper_8_72[16]
   <= 1
 cols_8_72: cols_8_72 - 2 cols_helper_8_72[0] - 4 cols_helper_8_72[1]
   - 8 cols_helper_8_72[2] - 16 cols_helper_8_72[3]
   - 32 cols_helper_8_72[4] - 64 cols_helper_8_72[5]
   - 128 cols_helper_8_72[6] - 256 cols_helper_8_72[7]
   - 512 cols_helper_8_72[8] - 1024 cols_helper_8_72[9]
   - 2048 cols_helper_8_72[10] - 4096 cols_helper_8_72[11]
   - 8192 cols_helper_8_72[12] - 16384 cols_helper_8_72[13]
   - 32768 cols_helper_8_72[14] - 65536 cols_helper_8_72[15]
   - 131072 cols_helper_8_72[16] = 0
 cols_acc_8_72: - 435 frac[8,72] + cols_8_72 >= 0
 accuracy_8_72: mem[8,72] - 0.01171875 cols_8_72 = 0
 accuracy_2_2: - 5.09765625 frac[2,2] + mem[2,2] = 0
 cols_helper_10_66: cols_helper_10_66[0] + cols_helper_10_66[1]
   + cols_helper_10_66[2] + cols_helper_10_66[3] + cols_helper_10_66[4]
   + cols_helper_10_66[5] + cols_helper_10_66[6] + cols_helper_10_66[7]
   + cols_helper_10_66[8] + cols_helper_10_66[9] + cols_helper_10_66[10]
   + cols_helper_10_66[11] + cols_helper_10_66[12] + cols_helper_10_66[13]
   + cols_helper_10_66[14] + cols_helper_10_66[15] + cols_helper_10_66[16]
   <= 1
 cols_10_66: cols_10_66 - 2 cols_helper_10_66[0] - 4 cols_helper_10_66[1]
   - 8 cols_helper_10_66[2] - 16 cols_helper_10_66[3]
   - 32 cols_helper_10_66[4] - 64 cols_helper_10_66[5]
   - 128 cols_helper_10_66[6] - 256 cols_helper_10_66[7]
   - 512 cols_helper_10_66[8] - 1024 cols_helper_10_66[9]
   - 2048 cols_helper_10_66[10] - 4096 cols_helper_10_66[11]
   - 8192 cols_helper_10_66[12] - 16384 cols_helper_10_66[13]
   - 32768 cols_helper_10_66[14] - 65536 cols_helper_10_66[15]
   - 131072 cols_helper_10_66[16] = 0
 cols_acc_10_66: - 435 frac[10,66] + cols_10_66 >= 0
 accuracy_10_66: mem[10,66] - 0.01171875 cols_10_66 = 0
 cols_helper_11_66: cols_helper_11_66[0] + cols_helper_11_66[1]
   + cols_helper_11_66[2] + cols_helper_11_66[3] + cols_helper_11_66[4]
   + cols_helper_11_66[5] + cols_helper_11_66[6] + cols_helper_11_66[7]
   + cols_helper_11_66[8] + cols_helper_11_66[9] + cols_helper_11_66[10]
   + cols_helper_11_66[11] + cols_helper_11_66[12] + cols_helper_11_66[13]
   + cols_helper_11_66[14] + cols_helper_11_66[15] + cols_helper_11_66[16]
   <= 1
 cols_11_66: cols_11_66 - 2 cols_helper_11_66[0] - 4 cols_helper_11_66[1]
   - 8 cols_helper_11_66[2] - 16 cols_helper_11_66[3]
   - 32 cols_helper_11_66[4] - 64 cols_helper_11_66[5]
   - 128 cols_helper_11_66[6] - 256 cols_helper_11_66[7]
   - 512 cols_helper_11_66[8] - 1024 cols_helper_11_66[9]
   - 2048 cols_helper_11_66[10] - 4096 cols_helper_11_66[11]
   - 8192 cols_helper_11_66[12] - 16384 cols_helper_11_66[13]
   - 32768 cols_helper_11_66[14] - 65536 cols_helper_11_66[15]
   - 131072 cols_helper_11_66[16] = 0
 cols_acc_11_66: - 435 frac[11,66] + cols_11_66 >= 0
 accuracy_11_66: mem[11,66] - 0.01171875 cols_11_66 = 0
 accuracy_3_33: - 5.09765625 frac[3,33] + mem[3,33] = 0
 cols_helper_8_61: cols_helper_8_61[0] + cols_helper_8_61[1]
   + cols_helper_8_61[2] + cols_helper_8_61[3] + cols_helper_8_61[4]
   + cols_helper_8_61[5] + cols_helper_8_61[6] + cols_helper_8_61[7]
   + cols_helper_8_61[8] + cols_helper_8_61[9] + cols_helper_8_61[10]
   + cols_helper_8_61[11] + cols_helper_8_61[12] + cols_helper_8_61[13]
   + cols_helper_8_61[14] + cols_helper_8_61[15] + cols_helper_8_61[16]
   <= 1
 cols_8_61: cols_8_61 - 2 cols_helper_8_61[0] - 4 cols_helper_8_61[1]
   - 8 cols_helper_8_61[2] - 16 cols_helper_8_61[3]
   - 32 cols_helper_8_61[4] - 64 cols_helper_8_61[5]
   - 128 cols_helper_8_61[6] - 256 cols_helper_8_61[7]
   - 512 cols_helper_8_61[8] - 1024 cols_helper_8_61[9]
   - 2048 cols_helper_8_61[10] - 4096 cols_helper_8_61[11]
   - 8192 cols_helper_8_61[12] - 16384 cols_helper_8_61[13]
   - 32768 cols_helper_8_61[14] - 65536 cols_helper_8_61[15]
   - 131072 cols_helper_8_61[16] = 0
 cols_acc_8_61: - 435 frac[8,61] + cols_8_61 >= 0
 accuracy_8_61: mem[8,61] - 0.01171875 cols_8_61 = 0
 accuracy_0_74: - 5.09765625 frac[0,74] + mem[0,74] = 0
 accuracy_4_64: - 5.09765625 frac[4,64] + mem[4,64] = 0
 cols_helper_10_63: cols_helper_10_63[0] + cols_helper_10_63[1]
   + cols_helper_10_63[2] + cols_helper_10_63[3] + cols_helper_10_63[4]
   + cols_helper_10_63[5] + cols_helper_10_63[6] + cols_helper_10_63[7]
   + cols_helper_10_63[8] + cols_helper_10_63[9] + cols_helper_10_63[10]
   + cols_helper_10_63[11] + cols_helper_10_63[12] + cols_helper_10_63[13]
   + cols_helper_10_63[14] + cols_helper_10_63[15] + cols_helper_10_63[16]
   <= 1
 cols_10_63: cols_10_63 - 2 cols_helper_10_63[0] - 4 cols_helper_10_63[1]
   - 8 cols_helper_10_63[2] - 16 cols_helper_10_63[3]
   - 32 cols_helper_10_63[4] - 64 cols_helper_10_63[5]
   - 128 cols_helper_10_63[6] - 256 cols_helper_10_63[7]
   - 512 cols_helper_10_63[8] - 1024 cols_helper_10_63[9]
   - 2048 cols_helper_10_63[10] - 4096 cols_helper_10_63[11]
   - 8192 cols_helper_10_63[12] - 16384 cols_helper_10_63[13]
   - 32768 cols_helper_10_63[14] - 65536 cols_helper_10_63[15]
   - 131072 cols_helper_10_63[16] = 0
 cols_acc_10_63: - 435 frac[10,63] + cols_10_63 >= 0
 accuracy_10_63: mem[10,63] - 0.01171875 cols_10_63 = 0
 accuracy_1_66: - 5.09765625 frac[1,66] + mem[1,66] = 0
 cols_helper_8_3: cols_helper_8_3[0] + cols_helper_8_3[1]
   + cols_helper_8_3[2] + cols_helper_8_3[3] + cols_helper_8_3[4]
   + cols_helper_8_3[5] + cols_helper_8_3[6] + cols_helper_8_3[7]
   + cols_helper_8_3[8] + cols_helper_8_3[9] + cols_helper_8_3[10]
   + cols_helper_8_3[11] + cols_helper_8_3[12] + cols_helper_8_3[13]
   + cols_helper_8_3[14] + cols_helper_8_3[15] + cols_helper_8_3[16] <= 1
 cols_8_3: cols_8_3 - 2 cols_helper_8_3[0] - 4 cols_helper_8_3[1]
   - 8 cols_helper_8_3[2] - 16 cols_helper_8_3[3] - 32 cols_helper_8_3[4]
   - 64 cols_helper_8_3[5] - 128 cols_helper_8_3[6]
   - 256 cols_helper_8_3[7] - 512 cols_helper_8_3[8]
   - 1024 cols_helper_8_3[9] - 2048 cols_helper_8_3[10]
   - 4096 cols_helper_8_3[11] - 8192 cols_helper_8_3[12]
   - 16384 cols_helper_8_3[13] - 32768 cols_helper_8_3[14]
   - 65536 cols_helper_8_3[15] - 131072 cols_helper_8_3[16] = 0
 cols_acc_8_3: - 435 frac[8,3] + cols_8_3 >= 0
 accuracy_8_3: mem[8,3] - 0.01171875 cols_8_3 = 0
 cols_helper_11_62: cols_helper_11_62[0] + cols_helper_11_62[1]
   + cols_helper_11_62[2] + cols_helper_11_62[3] + cols_helper_11_62[4]
   + cols_helper_11_62[5] + cols_helper_11_62[6] + cols_helper_11_62[7]
   + cols_helper_11_62[8] + cols_helper_11_62[9] + cols_helper_11_62[10]
   + cols_helper_11_62[11] + cols_helper_11_62[12] + cols_helper_11_62[13]
   + cols_helper_11_62[14] + cols_helper_11_62[15] + cols_helper_11_62[16]
   <= 1
 cols_11_62: cols_11_62 - 2 cols_helper_11_62[0] - 4 cols_helper_11_62[1]
   - 8 cols_helper_11_62[2] - 16 cols_helper_11_62[3]
   - 32 cols_helper_11_62[4] - 64 cols_helper_11_62[5]
   - 128 cols_helper_11_62[6] - 256 cols_helper_11_62[7]
   - 512 cols_helper_11_62[8] - 1024 cols_helper_11_62[9]
   - 2048 cols_helper_11_62[10] - 4096 cols_helper_11_62[11]
   - 8192 cols_helper_11_62[12] - 16384 cols_helper_11_62[13]
   - 32768 cols_helper_11_62[14] - 65536 cols_helper_11_62[15]
   - 131072 cols_helper_11_62[16] = 0
 cols_acc_11_62: - 435 frac[11,62] + cols_11_62 >= 0
 accuracy_11_62: mem[11,62] - 0.01171875 cols_11_62 = 0
 cols_helper_9_2: cols_helper_9_2[0] + cols_helper_9_2[1]
   + cols_helper_9_2[2] + cols_helper_9_2[3] + cols_helper_9_2[4]
   + cols_helper_9_2[5] + cols_helper_9_2[6] + cols_helper_9_2[7]
   + cols_helper_9_2[8] + cols_helper_9_2[9] + cols_helper_9_2[10]
   + cols_helper_9_2[11] + cols_helper_9_2[12] + cols_helper_9_2[13]
   + cols_helper_9_2[14] + cols_helper_9_2[15] + cols_helper_9_2[16] <= 1
 cols_9_2: cols_9_2 - 2 cols_helper_9_2[0] - 4 cols_helper_9_2[1]
   - 8 cols_helper_9_2[2] - 16 cols_helper_9_2[3] - 32 cols_helper_9_2[4]
   - 64 cols_helper_9_2[5] - 128 cols_helper_9_2[6]
   - 256 cols_helper_9_2[7] - 512 cols_helper_9_2[8]
   - 1024 cols_helper_9_2[9] - 2048 cols_helper_9_2[10]
   - 4096 cols_helper_9_2[11] - 8192 cols_helper_9_2[12]
   - 16384 cols_helper_9_2[13] - 32768 cols_helper_9_2[14]
   - 65536 cols_helper_9_2[15] - 131072 cols_helper_9_2[16] = 0
 cols_acc_9_2: - 435 frac[9,2] + cols_9_2 >= 0
 accuracy_9_2: mem[9,2] - 0.01171875 cols_9_2 = 0
 accuracy_6_74: - 5.09765625 frac[6,74] + mem[6,74] = 0
 cols_helper_10_25: cols_helper_10_25[0] + cols_helper_10_25[1]
   + cols_helper_10_25[2] + cols_helper_10_25[3] + cols_helper_10_25[4]
   + cols_helper_10_25[5] + cols_helper_10_25[6] + cols_helper_10_25[7]
   + cols_helper_10_25[8] + cols_helper_10_25[9] + cols_helper_10_25[10]
   + cols_helper_10_25[11] + cols_helper_10_25[12] + cols_helper_10_25[13]
   + cols_helper_10_25[14] + cols_helper_10_25[15] + cols_helper_10_25[16]
   <= 1
 cols_10_25: cols_10_25 - 2 cols_helper_10_25[0] - 4 cols_helper_10_25[1]
   - 8 cols_helper_10_25[2] - 16 cols_helper_10_25[3]
   - 32 cols_helper_10_25[4] - 64 cols_helper_10_25[5]
   - 128 cols_helper_10_25[6] - 256 cols_helper_10_25[7]
   - 512 cols_helper_10_25[8] - 1024 cols_helper_10_25[9]
   - 2048 cols_helper_10_25[10] - 4096 cols_helper_10_25[11]
   - 8192 cols_helper_10_25[12] - 16384 cols_helper_10_25[13]
   - 32768 cols_helper_10_25[14] - 65536 cols_helper_10_25[15]
   - 131072 cols_helper_10_25[16] = 0
 cols_acc_10_25: - 435 frac[10,25] + cols_10_25 >= 0
 accuracy_10_25: mem[10,25] - 0.01171875 cols_10_25 = 0
 accuracy_1_30: - 5.09765625 frac[1,30] + mem[1,30] = 0
 accuracy_0_43: - 5.09765625 frac[0,43] + mem[0,43] = 0
 cols_helper_8_36: cols_helper_8_36[0] + cols_helper_8_36[1]
   + cols_helper_8_36[2] + cols_helper_8_36[3] + cols_helper_8_36[4]
   + cols_helper_8_36[5] + cols_helper_8_36[6] + cols_helper_8_36[7]
   + cols_helper_8_36[8] + cols_helper_8_36[9] + cols_helper_8_36[10]
   + cols_helper_8_36[11] + cols_helper_8_36[12] + cols_helper_8_36[13]
   + cols_helper_8_36[14] + cols_helper_8_36[15] + cols_helper_8_36[16]
   <= 1
 cols_8_36: cols_8_36 - 2 cols_helper_8_36[0] - 4 cols_helper_8_36[1]
   - 8 cols_helper_8_36[2] - 16 cols_helper_8_36[3]
   - 32 cols_helper_8_36[4] - 64 cols_helper_8_36[5]
   - 128 cols_helper_8_36[6] - 256 cols_helper_8_36[7]
   - 512 cols_helper_8_36[8] - 1024 cols_helper_8_36[9]
   - 2048 cols_helper_8_36[10] - 4096 cols_helper_8_36[11]
   - 8192 cols_helper_8_36[12] - 16384 cols_helper_8_36[13]
   - 32768 cols_helper_8_36[14] - 65536 cols_helper_8_36[15]
   - 131072 cols_helper_8_36[16] = 0
 cols_acc_8_36: - 435 frac[8,36] + cols_8_36 >= 0
 accuracy_8_36: mem[8,36] - 0.01171875 cols_8_36 = 0
 accuracy_0_65: - 5.09765625 frac[0,65] + mem[0,65] = 0
 cols_helper_10_38: cols_helper_10_38[0] + cols_helper_10_38[1]
   + cols_helper_10_38[2] + cols_helper_10_38[3] + cols_helper_10_38[4]
   + cols_helper_10_38[5] + cols_helper_10_38[6] + cols_helper_10_38[7]
   + cols_helper_10_38[8] + cols_helper_10_38[9] + cols_helper_10_38[10]
   + cols_helper_10_38[11] + cols_helper_10_38[12] + cols_helper_10_38[13]
   + cols_helper_10_38[14] + cols_helper_10_38[15] + cols_helper_10_38[16]
   <= 1
 cols_10_38: cols_10_38 - 2 cols_helper_10_38[0] - 4 cols_helper_10_38[1]
   - 8 cols_helper_10_38[2] - 16 cols_helper_10_38[3]
   - 32 cols_helper_10_38[4] - 64 cols_helper_10_38[5]
   - 128 cols_helper_10_38[6] - 256 cols_helper_10_38[7]
   - 512 cols_helper_10_38[8] - 1024 cols_helper_10_38[9]
   - 2048 cols_helper_10_38[10] - 4096 cols_helper_10_38[11]
   - 8192 cols_helper_10_38[12] - 16384 cols_helper_10_38[13]
   - 32768 cols_helper_10_38[14] - 65536 cols_helper_10_38[15]
   - 131072 cols_helper_10_38[16] = 0
 cols_acc_10_38: - 435 frac[10,38] + cols_10_38 >= 0
 accuracy_10_38: mem[10,38] - 0.01171875 cols_10_38 = 0
 cols_helper_10_0: cols_helper_10_0[0] + cols_helper_10_0[1]
   + cols_helper_10_0[2] + cols_helper_10_0[3] + cols_helper_10_0[4]
   + cols_helper_10_0[5] + cols_helper_10_0[6] + cols_helper_10_0[7]
   + cols_helper_10_0[8] + cols_helper_10_0[9] + cols_helper_10_0[10]
   + cols_helper_10_0[11] + cols_helper_10_0[12] + cols_helper_10_0[13]
   + cols_helper_10_0[14] + cols_helper_10_0[15] + cols_helper_10_0[16]
   <= 1
 cols_10_0: cols_10_0 - 2 cols_helper_10_0[0] - 4 cols_helper_10_0[1]
   - 8 cols_helper_10_0[2] - 16 cols_helper_10_0[3]
   - 32 cols_helper_10_0[4] - 64 cols_helper_10_0[5]
   - 128 cols_helper_10_0[6] - 256 cols_helper_10_0[7]
   - 512 cols_helper_10_0[8] - 1024 cols_helper_10_0[9]
   - 2048 cols_helper_10_0[10] - 4096 cols_helper_10_0[11]
   - 8192 cols_helper_10_0[12] - 16384 cols_helper_10_0[13]
   - 32768 cols_helper_10_0[14] - 65536 cols_helper_10_0[15]
   - 131072 cols_helper_10_0[16] = 0
 cols_acc_10_0: - 435 frac[10,0] + cols_10_0 >= 0
 accuracy_10_0: mem[10,0] - 0.01171875 cols_10_0 = 0
 cols_helper_8_16: cols_helper_8_16[0] + cols_helper_8_16[1]
   + cols_helper_8_16[2] + cols_helper_8_16[3] + cols_helper_8_16[4]
   + cols_helper_8_16[5] + cols_helper_8_16[6] + cols_helper_8_16[7]
   + cols_helper_8_16[8] + cols_helper_8_16[9] + cols_helper_8_16[10]
   + cols_helper_8_16[11] + cols_helper_8_16[12] + cols_helper_8_16[13]
   + cols_helper_8_16[14] + cols_helper_8_16[15] + cols_helper_8_16[16]
   <= 1
 cols_8_16: cols_8_16 - 2 cols_helper_8_16[0] - 4 cols_helper_8_16[1]
   - 8 cols_helper_8_16[2] - 16 cols_helper_8_16[3]
   - 32 cols_helper_8_16[4] - 64 cols_helper_8_16[5]
   - 128 cols_helper_8_16[6] - 256 cols_helper_8_16[7]
   - 512 cols_helper_8_16[8] - 1024 cols_helper_8_16[9]
   - 2048 cols_helper_8_16[10] - 4096 cols_helper_8_16[11]
   - 8192 cols_helper_8_16[12] - 16384 cols_helper_8_16[13]
   - 32768 cols_helper_8_16[14] - 65536 cols_helper_8_16[15]
   - 131072 cols_helper_8_16[16] = 0
 cols_acc_8_16: - 435 frac[8,16] + cols_8_16 >= 0
 accuracy_8_16: mem[8,16] - 0.01171875 cols_8_16 = 0
 cols_helper_11_13: cols_helper_11_13[0] + cols_helper_11_13[1]
   + cols_helper_11_13[2] + cols_helper_11_13[3] + cols_helper_11_13[4]
   + cols_helper_11_13[5] + cols_helper_11_13[6] + cols_helper_11_13[7]
   + cols_helper_11_13[8] + cols_helper_11_13[9] + cols_helper_11_13[10]
   + cols_helper_11_13[11] + cols_helper_11_13[12] + cols_helper_11_13[13]
   + cols_helper_11_13[14] + cols_helper_11_13[15] + cols_helper_11_13[16]
   <= 1
 cols_11_13: cols_11_13 - 2 cols_helper_11_13[0] - 4 cols_helper_11_13[1]
   - 8 cols_helper_11_13[2] - 16 cols_helper_11_13[3]
   - 32 cols_helper_11_13[4] - 64 cols_helper_11_13[5]
   - 128 cols_helper_11_13[6] - 256 cols_helper_11_13[7]
   - 512 cols_helper_11_13[8] - 1024 cols_helper_11_13[9]
   - 2048 cols_helper_11_13[10] - 4096 cols_helper_11_13[11]
   - 8192 cols_helper_11_13[12] - 16384 cols_helper_11_13[13]
   - 32768 cols_helper_11_13[14] - 65536 cols_helper_11_13[15]
   - 131072 cols_helper_11_13[16] = 0
 cols_acc_11_13: - 435 frac[11,13] + cols_11_13 >= 0
 accuracy_11_13: mem[11,13] - 0.01171875 cols_11_13 = 0
 accuracy_0_22: - 5.09765625 frac[0,22] + mem[0,22] = 0
 accuracy_0_12: - 5.09765625 frac[0,12] + mem[0,12] = 0
 accuracy_1_9: - 5.09765625 frac[1,9] + mem[1,9] = 0
 accuracy_4_2: - 5.09765625 frac[4,2] + mem[4,2] = 0
 accuracy_3_47: - 5.09765625 frac[3,47] + mem[3,47] = 0
 cols_helper_9_64: cols_helper_9_64[0] + cols_helper_9_64[1]
   + cols_helper_9_64[2] + cols_helper_9_64[3] + cols_helper_9_64[4]
   + cols_helper_9_64[5] + cols_helper_9_64[6] + cols_helper_9_64[7]
   + cols_helper_9_64[8] + cols_helper_9_64[9] + cols_helper_9_64[10]
   + cols_helper_9_64[11] + cols_helper_9_64[12] + cols_helper_9_64[13]
   + cols_helper_9_64[14] + cols_helper_9_64[15] + cols_helper_9_64[16]
   <= 1
 cols_9_64: cols_9_64 - 2 cols_helper_9_64[0] - 4 cols_helper_9_64[1]
   - 8 cols_helper_9_64[2] - 16 cols_helper_9_64[3]
   - 32 cols_helper_9_64[4] - 64 cols_helper_9_64[5]
   - 128 cols_helper_9_64[6] - 256 cols_helper_9_64[7]
   - 512 cols_helper_9_64[8] - 1024 cols_helper_9_64[9]
   - 2048 cols_helper_9_64[10] - 4096 cols_helper_9_64[11]
   - 8192 cols_helper_9_64[12] - 16384 cols_helper_9_64[13]
   - 32768 cols_helper_9_64[14] - 65536 cols_helper_9_64[15]
   - 131072 cols_helper_9_64[16] = 0
 cols_acc_9_64: - 435 frac[9,64] + cols_9_64 >= 0
 accuracy_9_64: mem[9,64] - 0.01171875 cols_9_64 = 0
 cols_helper_8_43: cols_helper_8_43[0] + cols_helper_8_43[1]
   + cols_helper_8_43[2] + cols_helper_8_43[3] + cols_helper_8_43[4]
   + cols_helper_8_43[5] + cols_helper_8_43[6] + cols_helper_8_43[7]
   + cols_helper_8_43[8] + cols_helper_8_43[9] + cols_helper_8_43[10]
   + cols_helper_8_43[11] + cols_helper_8_43[12] + cols_helper_8_43[13]
   + cols_helper_8_43[14] + cols_helper_8_43[15] + cols_helper_8_43[16]
   <= 1
 cols_8_43: cols_8_43 - 2 cols_helper_8_43[0] - 4 cols_helper_8_43[1]
   - 8 cols_helper_8_43[2] - 16 cols_helper_8_43[3]
   - 32 cols_helper_8_43[4] - 64 cols_helper_8_43[5]
   - 128 cols_helper_8_43[6] - 256 cols_helper_8_43[7]
   - 512 cols_helper_8_43[8] - 1024 cols_helper_8_43[9]
   - 2048 cols_helper_8_43[10] - 4096 cols_helper_8_43[11]
   - 8192 cols_helper_8_43[12] - 16384 cols_helper_8_43[13]
   - 32768 cols_helper_8_43[14] - 65536 cols_helper_8_43[15]
   - 131072 cols_helper_8_43[16] = 0
 cols_acc_8_43: - 435 frac[8,43] + cols_8_43 >= 0
 accuracy_8_43: mem[8,43] - 0.01171875 cols_8_43 = 0
 cols_helper_9_42: cols_helper_9_42[0] + cols_helper_9_42[1]
   + cols_helper_9_42[2] + cols_helper_9_42[3] + cols_helper_9_42[4]
   + cols_helper_9_42[5] + cols_helper_9_42[6] + cols_helper_9_42[7]
   + cols_helper_9_42[8] + cols_helper_9_42[9] + cols_helper_9_42[10]
   + cols_helper_9_42[11] + cols_helper_9_42[12] + cols_helper_9_42[13]
   + cols_helper_9_42[14] + cols_helper_9_42[15] + cols_helper_9_42[16]
   <= 1
 cols_9_42: cols_9_42 - 2 cols_helper_9_42[0] - 4 cols_helper_9_42[1]
   - 8 cols_helper_9_42[2] - 16 cols_helper_9_42[3]
   - 32 cols_helper_9_42[4] - 64 cols_helper_9_42[5]
   - 128 cols_helper_9_42[6] - 256 cols_helper_9_42[7]
   - 512 cols_helper_9_42[8] - 1024 cols_helper_9_42[9]
   - 2048 cols_helper_9_42[10] - 4096 cols_helper_9_42[11]
   - 8192 cols_helper_9_42[12] - 16384 cols_helper_9_42[13]
   - 32768 cols_helper_9_42[14] - 65536 cols_helper_9_42[15]
   - 131072 cols_helper_9_42[16] = 0
 cols_acc_9_42: - 435 frac[9,42] + cols_9_42 >= 0
 accuracy_9_42: mem[9,42] - 0.01171875 cols_9_42 = 0
 cols_helper_10_33: cols_helper_10_33[0] + cols_helper_10_33[1]
   + cols_helper_10_33[2] + cols_helper_10_33[3] + cols_helper_10_33[4]
   + cols_helper_10_33[5] + cols_helper_10_33[6] + cols_helper_10_33[7]
   + cols_helper_10_33[8] + cols_helper_10_33[9] + cols_helper_10_33[10]
   + cols_helper_10_33[11] + cols_helper_10_33[12] + cols_helper_10_33[13]
   + cols_helper_10_33[14] + cols_helper_10_33[15] + cols_helper_10_33[16]
   <= 1
 cols_10_33: cols_10_33 - 2 cols_helper_10_33[0] - 4 cols_helper_10_33[1]
   - 8 cols_helper_10_33[2] - 16 cols_helper_10_33[3]
   - 32 cols_helper_10_33[4] - 64 cols_helper_10_33[5]
   - 128 cols_helper_10_33[6] - 256 cols_helper_10_33[7]
   - 512 cols_helper_10_33[8] - 1024 cols_helper_10_33[9]
   - 2048 cols_helper_10_33[10] - 4096 cols_helper_10_33[11]
   - 8192 cols_helper_10_33[12] - 16384 cols_helper_10_33[13]
   - 32768 cols_helper_10_33[14] - 65536 cols_helper_10_33[15]
   - 131072 cols_helper_10_33[16] = 0
 cols_acc_10_33: - 435 frac[10,33] + cols_10_33 >= 0
 accuracy_10_33: mem[10,33] - 0.01171875 cols_10_33 = 0
 accuracy_2_66: - 5.09765625 frac[2,66] + mem[2,66] = 0
 cols_helper_8_49: cols_helper_8_49[0] + cols_helper_8_49[1]
   + cols_helper_8_49[2] + cols_helper_8_49[3] + cols_helper_8_49[4]
   + cols_helper_8_49[5] + cols_helper_8_49[6] + cols_helper_8_49[7]
   + cols_helper_8_49[8] + cols_helper_8_49[9] + cols_helper_8_49[10]
   + cols_helper_8_49[11] + cols_helper_8_49[12] + cols_helper_8_49[13]
   + cols_helper_8_49[14] + cols_helper_8_49[15] + cols_helper_8_49[16]
   <= 1
 cols_8_49: cols_8_49 - 2 cols_helper_8_49[0] - 4 cols_helper_8_49[1]
   - 8 cols_helper_8_49[2] - 16 cols_helper_8_49[3]
   - 32 cols_helper_8_49[4] - 64 cols_helper_8_49[5]
   - 128 cols_helper_8_49[6] - 256 cols_helper_8_49[7]
   - 512 cols_helper_8_49[8] - 1024 cols_helper_8_49[9]
   - 2048 cols_helper_8_49[10] - 4096 cols_helper_8_49[11]
   - 8192 cols_helper_8_49[12] - 16384 cols_helper_8_49[13]
   - 32768 cols_helper_8_49[14] - 65536 cols_helper_8_49[15]
   - 131072 cols_helper_8_49[16] = 0
 cols_acc_8_49: - 435 frac[8,49] + cols_8_49 >= 0
 accuracy_8_49: mem[8,49] - 0.01171875 cols_8_49 = 0
 accuracy_7_43: - 5.09765625 frac[7,43] + mem[7,43] = 0
 cols_helper_11_44: cols_helper_11_44[0] + cols_helper_11_44[1]
   + cols_helper_11_44[2] + cols_helper_11_44[3] + cols_helper_11_44[4]
   + cols_helper_11_44[5] + cols_helper_11_44[6] + cols_helper_11_44[7]
   + cols_helper_11_44[8] + cols_helper_11_44[9] + cols_helper_11_44[10]
   + cols_helper_11_44[11] + cols_helper_11_44[12] + cols_helper_11_44[13]
   + cols_helper_11_44[14] + cols_helper_11_44[15] + cols_helper_11_44[16]
   <= 1
 cols_11_44: cols_11_44 - 2 cols_helper_11_44[0] - 4 cols_helper_11_44[1]
   - 8 cols_helper_11_44[2] - 16 cols_helper_11_44[3]
   - 32 cols_helper_11_44[4] - 64 cols_helper_11_44[5]
   - 128 cols_helper_11_44[6] - 256 cols_helper_11_44[7]
   - 512 cols_helper_11_44[8] - 1024 cols_helper_11_44[9]
   - 2048 cols_helper_11_44[10] - 4096 cols_helper_11_44[11]
   - 8192 cols_helper_11_44[12] - 16384 cols_helper_11_44[13]
   - 32768 cols_helper_11_44[14] - 65536 cols_helper_11_44[15]
   - 131072 cols_helper_11_44[16] = 0
 cols_acc_11_44: - 435 frac[11,44] + cols_11_44 >= 0
 accuracy_11_44: mem[11,44] - 0.01171875 cols_11_44 = 0
 accuracy_3_66: - 5.09765625 frac[3,66] + mem[3,66] = 0
 cols_helper_10_11: cols_helper_10_11[0] + cols_helper_10_11[1]
   + cols_helper_10_11[2] + cols_helper_10_11[3] + cols_helper_10_11[4]
   + cols_helper_10_11[5] + cols_helper_10_11[6] + cols_helper_10_11[7]
   + cols_helper_10_11[8] + cols_helper_10_11[9] + cols_helper_10_11[10]
   + cols_helper_10_11[11] + cols_helper_10_11[12] + cols_helper_10_11[13]
   + cols_helper_10_11[14] + cols_helper_10_11[15] + cols_helper_10_11[16]
   <= 1
 cols_10_11: cols_10_11 - 2 cols_helper_10_11[0] - 4 cols_helper_10_11[1]
   - 8 cols_helper_10_11[2] - 16 cols_helper_10_11[3]
   - 32 cols_helper_10_11[4] - 64 cols_helper_10_11[5]
   - 128 cols_helper_10_11[6] - 256 cols_helper_10_11[7]
   - 512 cols_helper_10_11[8] - 1024 cols_helper_10_11[9]
   - 2048 cols_helper_10_11[10] - 4096 cols_helper_10_11[11]
   - 8192 cols_helper_10_11[12] - 16384 cols_helper_10_11[13]
   - 32768 cols_helper_10_11[14] - 65536 cols_helper_10_11[15]
   - 131072 cols_helper_10_11[16] = 0
 cols_acc_10_11: - 435 frac[10,11] + cols_10_11 >= 0
 accuracy_10_11: mem[10,11] - 0.01171875 cols_10_11 = 0
 cols_helper_8_23: cols_helper_8_23[0] + cols_helper_8_23[1]
   + cols_helper_8_23[2] + cols_helper_8_23[3] + cols_helper_8_23[4]
   + cols_helper_8_23[5] + cols_helper_8_23[6] + cols_helper_8_23[7]
   + cols_helper_8_23[8] + cols_helper_8_23[9] + cols_helper_8_23[10]
   + cols_helper_8_23[11] + cols_helper_8_23[12] + cols_helper_8_23[13]
   + cols_helper_8_23[14] + cols_helper_8_23[15] + cols_helper_8_23[16]
   <= 1
 cols_8_23: cols_8_23 - 2 cols_helper_8_23[0] - 4 cols_helper_8_23[1]
   - 8 cols_helper_8_23[2] - 16 cols_helper_8_23[3]
   - 32 cols_helper_8_23[4] - 64 cols_helper_8_23[5]
   - 128 cols_helper_8_23[6] - 256 cols_helper_8_23[7]
   - 512 cols_helper_8_23[8] - 1024 cols_helper_8_23[9]
   - 2048 cols_helper_8_23[10] - 4096 cols_helper_8_23[11]
   - 8192 cols_helper_8_23[12] - 16384 cols_helper_8_23[13]
   - 32768 cols_helper_8_23[14] - 65536 cols_helper_8_23[15]
   - 131072 cols_helper_8_23[16] = 0
 cols_acc_8_23: - 435 frac[8,23] + cols_8_23 >= 0
 accuracy_8_23: mem[8,23] - 0.01171875 cols_8_23 = 0
 cols_helper_11_2: cols_helper_11_2[0] + cols_helper_11_2[1]
   + cols_helper_11_2[2] + cols_helper_11_2[3] + cols_helper_11_2[4]
   + cols_helper_11_2[5] + cols_helper_11_2[6] + cols_helper_11_2[7]
   + cols_helper_11_2[8] + cols_helper_11_2[9] + cols_helper_11_2[10]
   + cols_helper_11_2[11] + cols_helper_11_2[12] + cols_helper_11_2[13]
   + cols_helper_11_2[14] + cols_helper_11_2[15] + cols_helper_11_2[16]
   <= 1
 cols_11_2: cols_11_2 - 2 cols_helper_11_2[0] - 4 cols_helper_11_2[1]
   - 8 cols_helper_11_2[2] - 16 cols_helper_11_2[3]
   - 32 cols_helper_11_2[4] - 64 cols_helper_11_2[5]
   - 128 cols_helper_11_2[6] - 256 cols_helper_11_2[7]
   - 512 cols_helper_11_2[8] - 1024 cols_helper_11_2[9]
   - 2048 cols_helper_11_2[10] - 4096 cols_helper_11_2[11]
   - 8192 cols_helper_11_2[12] - 16384 cols_helper_11_2[13]
   - 32768 cols_helper_11_2[14] - 65536 cols_helper_11_2[15]
   - 131072 cols_helper_11_2[16] = 0
 cols_acc_11_2: - 435 frac[11,2] + cols_11_2 >= 0
 accuracy_11_2: mem[11,2] - 0.01171875 cols_11_2 = 0
 cols_helper_10_21: cols_helper_10_21[0] + cols_helper_10_21[1]
   + cols_helper_10_21[2] + cols_helper_10_21[3] + cols_helper_10_21[4]
   + cols_helper_10_21[5] + cols_helper_10_21[6] + cols_helper_10_21[7]
   + cols_helper_10_21[8] + cols_helper_10_21[9] + cols_helper_10_21[10]
   + cols_helper_10_21[11] + cols_helper_10_21[12] + cols_helper_10_21[13]
   + cols_helper_10_21[14] + cols_helper_10_21[15] + cols_helper_10_21[16]
   <= 1
 cols_10_21: cols_10_21 - 2 cols_helper_10_21[0] - 4 cols_helper_10_21[1]
   - 8 cols_helper_10_21[2] - 16 cols_helper_10_21[3]
   - 32 cols_helper_10_21[4] - 64 cols_helper_10_21[5]
   - 128 cols_helper_10_21[6] - 256 cols_helper_10_21[7]
   - 512 cols_helper_10_21[8] - 1024 cols_helper_10_21[9]
   - 2048 cols_helper_10_21[10] - 4096 cols_helper_10_21[11]
   - 8192 cols_helper_10_21[12] - 16384 cols_helper_10_21[13]
   - 32768 cols_helper_10_21[14] - 65536 cols_helper_10_21[15]
   - 131072 cols_helper_10_21[16] = 0
 cols_acc_10_21: - 435 frac[10,21] + cols_10_21 >= 0
 accuracy_10_21: mem[10,21] - 0.01171875 cols_10_21 = 0
 accuracy_3_14: - 5.09765625 frac[3,14] + mem[3,14] = 0
 cols_helper_8_67: cols_helper_8_67[0] + cols_helper_8_67[1]
   + cols_helper_8_67[2] + cols_helper_8_67[3] + cols_helper_8_67[4]
   + cols_helper_8_67[5] + cols_helper_8_67[6] + cols_helper_8_67[7]
   + cols_helper_8_67[8] + cols_helper_8_67[9] + cols_helper_8_67[10]
   + cols_helper_8_67[11] + cols_helper_8_67[12] + cols_helper_8_67[13]
   + cols_helper_8_67[14] + cols_helper_8_67[15] + cols_helper_8_67[16]
   <= 1
 cols_8_67: cols_8_67 - 2 cols_helper_8_67[0] - 4 cols_helper_8_67[1]
   - 8 cols_helper_8_67[2] - 16 cols_helper_8_67[3]
   - 32 cols_helper_8_67[4] - 64 cols_helper_8_67[5]
   - 128 cols_helper_8_67[6] - 256 cols_helper_8_67[7]
   - 512 cols_helper_8_67[8] - 1024 cols_helper_8_67[9]
   - 2048 cols_helper_8_67[10] - 4096 cols_helper_8_67[11]
   - 8192 cols_helper_8_67[12] - 16384 cols_helper_8_67[13]
   - 32768 cols_helper_8_67[14] - 65536 cols_helper_8_67[15]
   - 131072 cols_helper_8_67[16] = 0
 cols_acc_8_67: - 435 frac[8,67] + cols_8_67 >= 0
 accuracy_8_67: mem[8,67] - 0.01171875 cols_8_67 = 0
 cols_helper_9_73: cols_helper_9_73[0] + cols_helper_9_73[1]
   + cols_helper_9_73[2] + cols_helper_9_73[3] + cols_helper_9_73[4]
   + cols_helper_9_73[5] + cols_helper_9_73[6] + cols_helper_9_73[7]
   + cols_helper_9_73[8] + cols_helper_9_73[9] + cols_helper_9_73[10]
   + cols_helper_9_73[11] + cols_helper_9_73[12] + cols_helper_9_73[13]
   + cols_helper_9_73[14] + cols_helper_9_73[15] + cols_helper_9_73[16]
   <= 1
 cols_9_73: cols_9_73 - 2 cols_helper_9_73[0] - 4 cols_helper_9_73[1]
   - 8 cols_helper_9_73[2] - 16 cols_helper_9_73[3]
   - 32 cols_helper_9_73[4] - 64 cols_helper_9_73[5]
   - 128 cols_helper_9_73[6] - 256 cols_helper_9_73[7]
   - 512 cols_helper_9_73[8] - 1024 cols_helper_9_73[9]
   - 2048 cols_helper_9_73[10] - 4096 cols_helper_9_73[11]
   - 8192 cols_helper_9_73[12] - 16384 cols_helper_9_73[13]
   - 32768 cols_helper_9_73[14] - 65536 cols_helper_9_73[15]
   - 131072 cols_helper_9_73[16] = 0
 cols_acc_9_73: - 435 frac[9,73] + cols_9_73 >= 0
 accuracy_9_73: mem[9,73] - 0.01171875 cols_9_73 = 0
 accuracy_4_61: - 5.09765625 frac[4,61] + mem[4,61] = 0
 cols_helper_10_40: cols_helper_10_40[0] + cols_helper_10_40[1]
   + cols_helper_10_40[2] + cols_helper_10_40[3] + cols_helper_10_40[4]
   + cols_helper_10_40[5] + cols_helper_10_40[6] + cols_helper_10_40[7]
   + cols_helper_10_40[8] + cols_helper_10_40[9] + cols_helper_10_40[10]
   + cols_helper_10_40[11] + cols_helper_10_40[12] + cols_helper_10_40[13]
   + cols_helper_10_40[14] + cols_helper_10_40[15] + cols_helper_10_40[16]
   <= 1
 cols_10_40: cols_10_40 - 2 cols_helper_10_40[0] - 4 cols_helper_10_40[1]
   - 8 cols_helper_10_40[2] - 16 cols_helper_10_40[3]
   - 32 cols_helper_10_40[4] - 64 cols_helper_10_40[5]
   - 128 cols_helper_10_40[6] - 256 cols_helper_10_40[7]
   - 512 cols_helper_10_40[8] - 1024 cols_helper_10_40[9]
   - 2048 cols_helper_10_40[10] - 4096 cols_helper_10_40[11]
   - 8192 cols_helper_10_40[12] - 16384 cols_helper_10_40[13]
   - 32768 cols_helper_10_40[14] - 65536 cols_helper_10_40[15]
   - 131072 cols_helper_10_40[16] = 0
 cols_acc_10_40: - 435 frac[10,40] + cols_10_40 >= 0
 accuracy_10_40: mem[10,40] - 0.01171875 cols_10_40 = 0
 cols_helper_8_56: cols_helper_8_56[0] + cols_helper_8_56[1]
   + cols_helper_8_56[2] + cols_helper_8_56[3] + cols_helper_8_56[4]
   + cols_helper_8_56[5] + cols_helper_8_56[6] + cols_helper_8_56[7]
   + cols_helper_8_56[8] + cols_helper_8_56[9] + cols_helper_8_56[10]
   + cols_helper_8_56[11] + cols_helper_8_56[12] + cols_helper_8_56[13]
   + cols_helper_8_56[14] + cols_helper_8_56[15] + cols_helper_8_56[16]
   <= 1
 cols_8_56: cols_8_56 - 2 cols_helper_8_56[0] - 4 cols_helper_8_56[1]
   - 8 cols_helper_8_56[2] - 16 cols_helper_8_56[3]
   - 32 cols_helper_8_56[4] - 64 cols_helper_8_56[5]
   - 128 cols_helper_8_56[6] - 256 cols_helper_8_56[7]
   - 512 cols_helper_8_56[8] - 1024 cols_helper_8_56[9]
   - 2048 cols_helper_8_56[10] - 4096 cols_helper_8_56[11]
   - 8192 cols_helper_8_56[12] - 16384 cols_helper_8_56[13]
   - 32768 cols_helper_8_56[14] - 65536 cols_helper_8_56[15]
   - 131072 cols_helper_8_56[16] = 0
 cols_acc_8_56: - 435 frac[8,56] + cols_8_56 >= 0
 accuracy_8_56: mem[8,56] - 0.01171875 cols_8_56 = 0
 cols_helper_11_37: cols_helper_11_37[0] + cols_helper_11_37[1]
   + cols_helper_11_37[2] + cols_helper_11_37[3] + cols_helper_11_37[4]
   + cols_helper_11_37[5] + cols_helper_11_37[6] + cols_helper_11_37[7]
   + cols_helper_11_37[8] + cols_helper_11_37[9] + cols_helper_11_37[10]
   + cols_helper_11_37[11] + cols_helper_11_37[12] + cols_helper_11_37[13]
   + cols_helper_11_37[14] + cols_helper_11_37[15] + cols_helper_11_37[16]
   <= 1
 cols_11_37: cols_11_37 - 2 cols_helper_11_37[0] - 4 cols_helper_11_37[1]
   - 8 cols_helper_11_37[2] - 16 cols_helper_11_37[3]
   - 32 cols_helper_11_37[4] - 64 cols_helper_11_37[5]
   - 128 cols_helper_11_37[6] - 256 cols_helper_11_37[7]
   - 512 cols_helper_11_37[8] - 1024 cols_helper_11_37[9]
   - 2048 cols_helper_11_37[10] - 4096 cols_helper_11_37[11]
   - 8192 cols_helper_11_37[12] - 16384 cols_helper_11_37[13]
   - 32768 cols_helper_11_37[14] - 65536 cols_helper_11_37[15]
   - 131072 cols_helper_11_37[16] = 0
 cols_acc_11_37: - 435 frac[11,37] + cols_11_37 >= 0
 accuracy_11_37: mem[11,37] - 0.01171875 cols_11_37 = 0
 cols_helper_9_53: cols_helper_9_53[0] + cols_helper_9_53[1]
   + cols_helper_9_53[2] + cols_helper_9_53[3] + cols_helper_9_53[4]
   + cols_helper_9_53[5] + cols_helper_9_53[6] + cols_helper_9_53[7]
   + cols_helper_9_53[8] + cols_helper_9_53[9] + cols_helper_9_53[10]
   + cols_helper_9_53[11] + cols_helper_9_53[12] + cols_helper_9_53[13]
   + cols_helper_9_53[14] + cols_helper_9_53[15] + cols_helper_9_53[16]
   <= 1
 cols_9_53: cols_9_53 - 2 cols_helper_9_53[0] - 4 cols_helper_9_53[1]
   - 8 cols_helper_9_53[2] - 16 cols_helper_9_53[3]
   - 32 cols_helper_9_53[4] - 64 cols_helper_9_53[5]
   - 128 cols_helper_9_53[6] - 256 cols_helper_9_53[7]
   - 512 cols_helper_9_53[8] - 1024 cols_helper_9_53[9]
   - 2048 cols_helper_9_53[10] - 4096 cols_helper_9_53[11]
   - 8192 cols_helper_9_53[12] - 16384 cols_helper_9_53[13]
   - 32768 cols_helper_9_53[14] - 65536 cols_helper_9_53[15]
   - 131072 cols_helper_9_53[16] = 0
 cols_acc_9_53: - 435 frac[9,53] + cols_9_53 >= 0
 accuracy_9_53: mem[9,53] - 0.01171875 cols_9_53 = 0
 cols_helper_10_50: cols_helper_10_50[0] + cols_helper_10_50[1]
   + cols_helper_10_50[2] + cols_helper_10_50[3] + cols_helper_10_50[4]
   + cols_helper_10_50[5] + cols_helper_10_50[6] + cols_helper_10_50[7]
   + cols_helper_10_50[8] + cols_helper_10_50[9] + cols_helper_10_50[10]
   + cols_helper_10_50[11] + cols_helper_10_50[12] + cols_helper_10_50[13]
   + cols_helper_10_50[14] + cols_helper_10_50[15] + cols_helper_10_50[16]
   <= 1
 cols_10_50: cols_10_50 - 2 cols_helper_10_50[0] - 4 cols_helper_10_50[1]
   - 8 cols_helper_10_50[2] - 16 cols_helper_10_50[3]
   - 32 cols_helper_10_50[4] - 64 cols_helper_10_50[5]
   - 128 cols_helper_10_50[6] - 256 cols_helper_10_50[7]
   - 512 cols_helper_10_50[8] - 1024 cols_helper_10_50[9]
   - 2048 cols_helper_10_50[10] - 4096 cols_helper_10_50[11]
   - 8192 cols_helper_10_50[12] - 16384 cols_helper_10_50[13]
   - 32768 cols_helper_10_50[14] - 65536 cols_helper_10_50[15]
   - 131072 cols_helper_10_50[16] = 0
 cols_acc_10_50: - 435 frac[10,50] + cols_10_50 >= 0
 accuracy_10_50: mem[10,50] - 0.01171875 cols_10_50 = 0
 cols_helper_8_30: cols_helper_8_30[0] + cols_helper_8_30[1]
   + cols_helper_8_30[2] + cols_helper_8_30[3] + cols_helper_8_30[4]
   + cols_helper_8_30[5] + cols_helper_8_30[6] + cols_helper_8_30[7]
   + cols_helper_8_30[8] + cols_helper_8_30[9] + cols_helper_8_30[10]
   + cols_helper_8_30[11] + cols_helper_8_30[12] + cols_helper_8_30[13]
   + cols_helper_8_30[14] + cols_helper_8_30[15] + cols_helper_8_30[16]
   <= 1
 cols_8_30: cols_8_30 - 2 cols_helper_8_30[0] - 4 cols_helper_8_30[1]
   - 8 cols_helper_8_30[2] - 16 cols_helper_8_30[3]
   - 32 cols_helper_8_30[4] - 64 cols_helper_8_30[5]
   - 128 cols_helper_8_30[6] - 256 cols_helper_8_30[7]
   - 512 cols_helper_8_30[8] - 1024 cols_helper_8_30[9]
   - 2048 cols_helper_8_30[10] - 4096 cols_helper_8_30[11]
   - 8192 cols_helper_8_30[12] - 16384 cols_helper_8_30[13]
   - 32768 cols_helper_8_30[14] - 65536 cols_helper_8_30[15]
   - 131072 cols_helper_8_30[16] = 0
 cols_acc_8_30: - 435 frac[8,30] + cols_8_30 >= 0
 accuracy_8_30: mem[8,30] - 0.01171875 cols_8_30 = 0
 cols_helper_11_59: cols_helper_11_59[0] + cols_helper_11_59[1]
   + cols_helper_11_59[2] + cols_helper_11_59[3] + cols_helper_11_59[4]
   + cols_helper_11_59[5] + cols_helper_11_59[6] + cols_helper_11_59[7]
   + cols_helper_11_59[8] + cols_helper_11_59[9] + cols_helper_11_59[10]
   + cols_helper_11_59[11] + cols_helper_11_59[12] + cols_helper_11_59[13]
   + cols_helper_11_59[14] + cols_helper_11_59[15] + cols_helper_11_59[16]
   <= 1
 cols_11_59: cols_11_59 - 2 cols_helper_11_59[0] - 4 cols_helper_11_59[1]
   - 8 cols_helper_11_59[2] - 16 cols_helper_11_59[3]
   - 32 cols_helper_11_59[4] - 64 cols_helper_11_59[5]
   - 128 cols_helper_11_59[6] - 256 cols_helper_11_59[7]
   - 512 cols_helper_11_59[8] - 1024 cols_helper_11_59[9]
   - 2048 cols_helper_11_59[10] - 4096 cols_helper_11_59[11]
   - 8192 cols_helper_11_59[12] - 16384 cols_helper_11_59[13]
   - 32768 cols_helper_11_59[14] - 65536 cols_helper_11_59[15]
   - 131072 cols_helper_11_59[16] = 0
 cols_acc_11_59: - 435 frac[11,59] + cols_11_59 >= 0
 accuracy_11_59: mem[11,59] - 0.01171875 cols_11_59 = 0
 accuracy_5_65: - 5.09765625 frac[5,65] + mem[5,65] = 0
 cols_helper_10_28: cols_helper_10_28[0] + cols_helper_10_28[1]
   + cols_helper_10_28[2] + cols_helper_10_28[3] + cols_helper_10_28[4]
   + cols_helper_10_28[5] + cols_helper_10_28[6] + cols_helper_10_28[7]
   + cols_helper_10_28[8] + cols_helper_10_28[9] + cols_helper_10_28[10]
   + cols_helper_10_28[11] + cols_helper_10_28[12] + cols_helper_10_28[13]
   + cols_helper_10_28[14] + cols_helper_10_28[15] + cols_helper_10_28[16]
   <= 1
 cols_10_28: cols_10_28 - 2 cols_helper_10_28[0] - 4 cols_helper_10_28[1]
   - 8 cols_helper_10_28[2] - 16 cols_helper_10_28[3]
   - 32 cols_helper_10_28[4] - 64 cols_helper_10_28[5]
   - 128 cols_helper_10_28[6] - 256 cols_helper_10_28[7]
   - 512 cols_helper_10_28[8] - 1024 cols_helper_10_28[9]
   - 2048 cols_helper_10_28[10] - 4096 cols_helper_10_28[11]
   - 8192 cols_helper_10_28[12] - 16384 cols_helper_10_28[13]
   - 32768 cols_helper_10_28[14] - 65536 cols_helper_10_28[15]
   - 131072 cols_helper_10_28[16] = 0
 cols_acc_10_28: - 435 frac[10,28] + cols_10_28 >= 0
 accuracy_10_28: mem[10,28] - 0.01171875 cols_10_28 = 0
 cols_helper_11_17: cols_helper_11_17[0] + cols_helper_11_17[1]
   + cols_helper_11_17[2] + cols_helper_11_17[3] + cols_helper_11_17[4]
   + cols_helper_11_17[5] + cols_helper_11_17[6] + cols_helper_11_17[7]
   + cols_helper_11_17[8] + cols_helper_11_17[9] + cols_helper_11_17[10]
   + cols_helper_11_17[11] + cols_helper_11_17[12] + cols_helper_11_17[13]
   + cols_helper_11_17[14] + cols_helper_11_17[15] + cols_helper_11_17[16]
   <= 1
 cols_11_17: cols_11_17 - 2 cols_helper_11_17[0] - 4 cols_helper_11_17[1]
   - 8 cols_helper_11_17[2] - 16 cols_helper_11_17[3]
   - 32 cols_helper_11_17[4] - 64 cols_helper_11_17[5]
   - 128 cols_helper_11_17[6] - 256 cols_helper_11_17[7]
   - 512 cols_helper_11_17[8] - 1024 cols_helper_11_17[9]
   - 2048 cols_helper_11_17[10] - 4096 cols_helper_11_17[11]
   - 8192 cols_helper_11_17[12] - 16384 cols_helper_11_17[13]
   - 32768 cols_helper_11_17[14] - 65536 cols_helper_11_17[15]
   - 131072 cols_helper_11_17[16] = 0
 cols_acc_11_17: - 435 frac[11,17] + cols_11_17 >= 0
 accuracy_11_17: mem[11,17] - 0.01171875 cols_11_17 = 0
 cols_helper_8_74: cols_helper_8_74[0] + cols_helper_8_74[1]
   + cols_helper_8_74[2] + cols_helper_8_74[3] + cols_helper_8_74[4]
   + cols_helper_8_74[5] + cols_helper_8_74[6] + cols_helper_8_74[7]
   + cols_helper_8_74[8] + cols_helper_8_74[9] + cols_helper_8_74[10]
   + cols_helper_8_74[11] + cols_helper_8_74[12] + cols_helper_8_74[13]
   + cols_helper_8_74[14] + cols_helper_8_74[15] + cols_helper_8_74[16]
   <= 1
 cols_8_74: cols_8_74 - 2 cols_helper_8_74[0] - 4 cols_helper_8_74[1]
   - 8 cols_helper_8_74[2] - 16 cols_helper_8_74[3]
   - 32 cols_helper_8_74[4] - 64 cols_helper_8_74[5]
   - 128 cols_helper_8_74[6] - 256 cols_helper_8_74[7]
   - 512 cols_helper_8_74[8] - 1024 cols_helper_8_74[9]
   - 2048 cols_helper_8_74[10] - 4096 cols_helper_8_74[11]
   - 8192 cols_helper_8_74[12] - 16384 cols_helper_8_74[13]
   - 32768 cols_helper_8_74[14] - 65536 cols_helper_8_74[15]
   - 131072 cols_helper_8_74[16] = 0
 cols_acc_8_74: - 435 frac[8,74] + cols_8_74 >= 0
 accuracy_8_74: mem[8,74] - 0.01171875 cols_8_74 = 0
 cols_helper_10_64: cols_helper_10_64[0] + cols_helper_10_64[1]
   + cols_helper_10_64[2] + cols_helper_10_64[3] + cols_helper_10_64[4]
   + cols_helper_10_64[5] + cols_helper_10_64[6] + cols_helper_10_64[7]
   + cols_helper_10_64[8] + cols_helper_10_64[9] + cols_helper_10_64[10]
   + cols_helper_10_64[11] + cols_helper_10_64[12] + cols_helper_10_64[13]
   + cols_helper_10_64[14] + cols_helper_10_64[15] + cols_helper_10_64[16]
   <= 1
 cols_10_64: cols_10_64 - 2 cols_helper_10_64[0] - 4 cols_helper_10_64[1]
   - 8 cols_helper_10_64[2] - 16 cols_helper_10_64[3]
   - 32 cols_helper_10_64[4] - 64 cols_helper_10_64[5]
   - 128 cols_helper_10_64[6] - 256 cols_helper_10_64[7]
   - 512 cols_helper_10_64[8] - 1024 cols_helper_10_64[9]
   - 2048 cols_helper_10_64[10] - 4096 cols_helper_10_64[11]
   - 8192 cols_helper_10_64[12] - 16384 cols_helper_10_64[13]
   - 32768 cols_helper_10_64[14] - 65536 cols_helper_10_64[15]
   - 131072 cols_helper_10_64[16] = 0
 cols_acc_10_64: - 435 frac[10,64] + cols_10_64 >= 0
 accuracy_10_64: mem[10,64] - 0.01171875 cols_10_64 = 0
 accuracy_4_30: - 5.09765625 frac[4,30] + mem[4,30] = 0
 mem_tot_CPU2: - mem[0,36] - mem[0,19] - mem[0,9] - mem[0,69] - mem[0,0]
   - mem[0,30] - mem[0,21] - mem[0,71] - mem[0,2] - mem[0,25] - mem[0,16]
   - mem[0,66] - mem[0,18] - mem[0,29] - mem[0,32] - mem[0,74] - mem[0,43]
   - mem[0,65] - mem[0,22] - mem[0,12] + mem_tot_CPU2 = 0
 R1198: - 3.05176e-05 mem_tot_CPU2 + normalized_mem_tot_CPU2 = 0
 rows_tot_CPU2: - 3 frac[0,36] - 3 frac[0,19] - 3 frac[0,9] - 3 frac[0,69]
   - 3 frac[0,0] - 3 frac[0,30] - 3 frac[0,21] - 3 frac[0,71] - 3 frac[0,2]
   - 3 frac[0,25] - 3 frac[0,16] - 3 frac[0,66] - 3 frac[0,18]
   - 3 frac[0,29] - 3 frac[0,32] - 3 frac[0,74] - 3 frac[0,43]
   - 3 frac[0,65] - 3 frac[0,22] - 3 frac[0,12] + rows_tot_CPU2 = 0
 R1200: - 0.0833333333333333 rows_tot_CPU2 + normalized_rows_tot_CPU2 = 0
 mem_tot_CPU3: - mem[1,21] - mem[1,25] - mem[1,51] - mem[1,74] - mem[1,56]
   - mem[1,16] - mem[1,65] - mem[1,29] - mem[1,34] - mem[1,71] - mem[1,19]
   - mem[1,32] - mem[1,69] - mem[1,5] - mem[1,66] - mem[1,30] - mem[1,9]
   + mem_tot_CPU3 = 0
 R1202: - 3.05176e-05 mem_tot_CPU3 + normalized_mem_tot_CPU3 = 0
 rows_tot_CPU3: - 3 frac[1,21] - 3 frac[1,25] - 3 frac[1,51] - 3 frac[1,74]
   - 3 frac[1,56] - 3 frac[1,16] - 3 frac[1,65] - 3 frac[1,29]
   - 3 frac[1,34] - 3 frac[1,71] - 3 frac[1,19] - 3 frac[1,32]
   - 3 frac[1,69] - 3 frac[1,5] - 3 frac[1,66] - 3 frac[1,30] - 3 frac[1,9]
   + rows_tot_CPU3 = 0
 R1204: - 0.0833333333333333 rows_tot_CPU3 + normalized_rows_tot_CPU3 = 0
 mem_tot_CPU4: - mem[2,12] - mem[2,64] - mem[2,33] - mem[2,25] - mem[2,51]
   - mem[2,71] - mem[2,5] - mem[2,54] - mem[2,74] - mem[2,69] - mem[2,34]
   - mem[2,21] - mem[2,67] - mem[2,32] - mem[2,19] - mem[2,39] - mem[2,65]
   - mem[2,57] - mem[2,2] - mem[2,66] + mem_tot_CPU4 = 0
 R1206: - 3.05176e-05 mem_tot_CPU4 + normalized_mem_tot_CPU4 = 0
 rows_tot_CPU4: - 3 frac[2,12] - 3 frac[2,64] - 3 frac[2,33] - 3 frac[2,25]
   - 3 frac[2,51] - 3 frac[2,71] - 3 frac[2,5] - 3 frac[2,54]
   - 3 frac[2,74] - 3 frac[2,69] - 3 frac[2,34] - 3 frac[2,21]
   - 3 frac[2,67] - 3 frac[2,32] - 3 frac[2,19] - 3 frac[2,39]
   - 3 frac[2,65] - 3 frac[2,57] - 3 frac[2,2] - 3 frac[2,66]
   + rows_tot_CPU4 = 0
 R1208: - 0.0833333333333333 rows_tot_CPU4 + normalized_rows_tot_CPU4 = 0
 mem_tot_CPU7: - mem[3,39] - mem[3,21] - mem[3,32] - mem[3,61] - mem[3,74]
   - mem[3,34] - mem[3,5] - mem[3,43] - mem[3,16] - mem[3,36] - mem[3,12]
   - mem[3,33] - mem[3,47] - mem[3,66] - mem[3,14] + mem_tot_CPU7 = 0
 R1210: - 3.05176e-05 mem_tot_CPU7 + normalized_mem_tot_CPU7 = 0
 rows_tot_CPU7: - 3 frac[3,39] - 3 frac[3,21] - 3 frac[3,32] - 3 frac[3,61]
   - 3 frac[3,74] - 3 frac[3,34] - 3 frac[3,5] - 3 frac[3,43]
   - 3 frac[3,16] - 3 frac[3,36] - 3 frac[3,12] - 3 frac[3,33]
   - 3 frac[3,47] - 3 frac[3,66] - 3 frac[3,14] + rows_tot_CPU7 = 0
 R1212: - 0.0833333333333333 rows_tot_CPU7 + normalized_rows_tot_CPU7 = 0
 mem_tot_CPU17: - mem[4,66] - mem[4,5] - mem[4,69] - mem[4,45] - mem[4,14]
   - mem[4,57] - mem[4,29] - mem[4,65] - mem[4,25] - mem[4,39] - mem[4,0]
   - mem[4,21] - mem[4,64] - mem[4,2] - mem[4,61] - mem[4,30]
   + mem_tot_CPU17 = 0
 R1214: - 3.05176e-05 mem_tot_CPU17 + normalized_mem_tot_CPU17 = 0
 rows_tot_CPU17: - 3 frac[4,66] - 3 frac[4,5] - 3 frac[4,69] - 3 frac[4,45]
   - 3 frac[4,14] - 3 frac[4,57] - 3 frac[4,29] - 3 frac[4,65]
   - 3 frac[4,25] - 3 frac[4,39] - 3 frac[4,0] - 3 frac[4,21]
   - 3 frac[4,64] - 3 frac[4,2] - 3 frac[4,61] - 3 frac[4,30]
   + rows_tot_CPU17 = 0
 R1216: - 0.0833333333333333 rows_tot_CPU17 + normalized_rows_tot_CPU17
   = 0
 mem_tot_CPU18: - mem[5,74] - mem[5,43] - mem[5,39] - mem[5,32] - mem[5,66]
   - mem[5,25] - mem[5,21] - mem[5,30] - mem[5,19] - mem[5,5] - mem[5,65]
   + mem_tot_CPU18 = 0
 R1218: - 3.05176e-05 mem_tot_CPU18 + normalized_mem_tot_CPU18 = 0
 rows_tot_CPU18: - 3 frac[5,74] - 3 frac[5,43] - 3 frac[5,39]
   - 3 frac[5,32] - 3 frac[5,66] - 3 frac[5,25] - 3 frac[5,21]
   - 3 frac[5,30] - 3 frac[5,19] - 3 frac[5,5] - 3 frac[5,65]
   + rows_tot_CPU18 = 0
 R1220: - 0.0833333333333333 rows_tot_CPU18 + normalized_rows_tot_CPU18
   = 0
 mem_tot_CPU24: - mem[6,54] - mem[6,32] - mem[6,18] - mem[6,21] - mem[6,25]
   - mem[6,51] - mem[6,42] - mem[6,71] - mem[6,5] - mem[6,39] - mem[6,74]
   + mem_tot_CPU24 = 0
 R1222: - 3.05176e-05 mem_tot_CPU24 + normalized_mem_tot_CPU24 = 0
 rows_tot_CPU24: - 3 frac[6,54] - 3 frac[6,32] - 3 frac[6,18]
   - 3 frac[6,21] - 3 frac[6,25] - 3 frac[6,51] - 3 frac[6,42]
   - 3 frac[6,71] - 3 frac[6,5] - 3 frac[6,39] - 3 frac[6,74]
   + rows_tot_CPU24 = 0
 R1224: - 0.0833333333333333 rows_tot_CPU24 + normalized_rows_tot_CPU24
   = 0
 mem_tot_CPU26: - mem[7,22] - mem[7,39] - mem[7,21] - mem[7,51] - mem[7,18]
   - mem[7,33] - mem[7,69] - mem[7,12] - mem[7,2] - mem[7,71] - mem[7,47]
   - mem[7,65] - mem[7,43] + mem_tot_CPU26 = 0
 R1226: - 3.05176e-05 mem_tot_CPU26 + normalized_mem_tot_CPU26 = 0
 rows_tot_CPU26: - 3 frac[7,22] - 3 frac[7,39] - 3 frac[7,21]
   - 3 frac[7,51] - 3 frac[7,18] - 3 frac[7,33] - 3 frac[7,69]
   - 3 frac[7,12] - 3 frac[7,2] - 3 frac[7,71] - 3 frac[7,47]
   - 3 frac[7,65] - 3 frac[7,43] + rows_tot_CPU26 = 0
 R1228: - 0.0833333333333333 rows_tot_CPU26 + normalized_rows_tot_CPU26
   = 0
 mem_tot_tor_P41: - mem[8,63] - mem[8,5] - mem[8,12] - mem[8,18]
   - mem[8,51] - mem[8,25] - mem[8,69] - mem[8,0] - mem[8,33] - mem[8,7]
   - mem[8,40] - mem[8,14] - mem[8,64] - mem[8,47] - mem[8,53] - mem[8,27]
   - mem[8,71] - mem[8,60] - mem[8,2] - mem[8,35] - mem[8,9] - mem[8,48]
   - mem[8,22] - mem[8,66] - mem[8,55] - mem[8,29] - mem[8,73] - mem[8,62]
   - mem[8,37] - mem[8,17] - mem[8,50] - mem[8,57] - mem[8,32] - mem[8,39]
   - mem[8,19] - mem[8,26] - mem[8,70] - mem[8,59] - mem[8,34] - mem[8,41]
   - mem[8,15] - mem[8,21] - mem[8,65] - mem[8,54] - mem[8,28] - mem[8,72]
   - mem[8,61] - mem[8,3] - mem[8,36] - mem[8,16] - mem[8,43] - mem[8,49]
   - mem[8,23] - mem[8,67] - mem[8,56] - mem[8,30] - mem[8,74]
   + mem_tot_tor_P41 = 0
 R1230: - 1.6276041666666666e-04 mem_tot_tor_P41
   + normalized_mem_tot_tor_P41 = 0
 rows_tot_tor_P41: - 3 frac[8,63] - 3 frac[8,5] - 3 frac[8,12]
   - 3 frac[8,18] - 3 frac[8,51] - 3 frac[8,25] - 3 frac[8,69]
   - 3 frac[8,0] - 3 frac[8,33] - 3 frac[8,7] - 3 frac[8,40] - 3 frac[8,14]
   - 3 frac[8,64] - 3 frac[8,47] - 3 frac[8,53] - 3 frac[8,27]
   - 3 frac[8,71] - 3 frac[8,60] - 3 frac[8,2] - 3 frac[8,35] - 3 frac[8,9]
   - 3 frac[8,48] - 3 frac[8,22] - 3 frac[8,66] - 3 frac[8,55]
   - 3 frac[8,29] - 3 frac[8,73] - 3 frac[8,62] - 3 frac[8,37]
   - 3 frac[8,17] - 3 frac[8,50] - 3 frac[8,57] - 3 frac[8,32]
   - 3 frac[8,39] - 3 frac[8,19] - 3 frac[8,26] - 3 frac[8,70]
   - 3 frac[8,59] - 3 frac[8,34] - 3 frac[8,41] - 3 frac[8,15]
   - 3 frac[8,21] - 3 frac[8,65] - 3 frac[8,54] - 3 frac[8,28]
   - 3 frac[8,72] - 3 frac[8,61] - 3 frac[8,3] - 3 frac[8,36]
   - 3 frac[8,16] - 3 frac[8,43] - 3 frac[8,49] - 3 frac[8,23]
   - 3 frac[8,67] - 3 frac[8,56] - 3 frac[8,30] - 3 frac[8,74]
   + rows_tot_tor_P41 = 0
 R1232: - 0.0208333333333333 rows_tot_tor_P41 + normalized_rows_tot_tor_P41
   = 0
 mem_tot_tor_P43: - mem[9,62] - mem[9,0] - mem[9,39] - mem[9,19]
   - mem[9,50] - mem[9,20] - mem[9,59] - mem[9,29] - mem[9,60] - mem[9,67]
   - mem[9,37] - mem[9,17] - mem[9,68] - mem[9,46] - mem[9,57] - mem[9,4]
   - mem[9,65] - mem[9,43] - mem[9,13] - mem[9,23] - mem[9,74] - mem[9,44]
   - mem[9,54] - mem[9,24] - mem[9,1] - mem[9,32] - mem[9,10] - mem[9,71]
   - mem[9,41] - mem[9,51] - mem[9,21] - mem[9,72] - mem[9,30] - mem[9,61]
   - mem[9,7] - mem[9,38] - mem[9,18] - mem[9,69] - mem[9,58] - mem[9,28]
   - mem[9,35] - mem[9,5] - mem[9,66] - mem[9,14] - mem[9,45] - mem[9,55]
   - mem[9,25] - mem[9,2] - mem[9,64] - mem[9,42] - mem[9,73] - mem[9,53]
   + mem_tot_tor_P43 = 0
 R1234: - 1.6276041666666666e-04 mem_tot_tor_P43
   + normalized_mem_tot_tor_P43 = 0
 rows_tot_tor_P43: - 3 frac[9,62] - 3 frac[9,0] - 3 frac[9,39]
   - 3 frac[9,19] - 3 frac[9,50] - 3 frac[9,20] - 3 frac[9,59]
   - 3 frac[9,29] - 3 frac[9,60] - 3 frac[9,67] - 3 frac[9,37]
   - 3 frac[9,17] - 3 frac[9,68] - 3 frac[9,46] - 3 frac[9,57]
   - 3 frac[9,4] - 3 frac[9,65] - 3 frac[9,43] - 3 frac[9,13]
   - 3 frac[9,23] - 3 frac[9,74] - 3 frac[9,44] - 3 frac[9,54]
   - 3 frac[9,24] - 3 frac[9,1] - 3 frac[9,32] - 3 frac[9,10]
   - 3 frac[9,71] - 3 frac[9,41] - 3 frac[9,51] - 3 frac[9,21]
   - 3 frac[9,72] - 3 frac[9,30] - 3 frac[9,61] - 3 frac[9,7]
   - 3 frac[9,38] - 3 frac[9,18] - 3 frac[9,69] - 3 frac[9,58]
   - 3 frac[9,28] - 3 frac[9,35] - 3 frac[9,5] - 3 frac[9,66]
   - 3 frac[9,14] - 3 frac[9,45] - 3 frac[9,55] - 3 frac[9,25]
   - 3 frac[9,2] - 3 frac[9,64] - 3 frac[9,42] - 3 frac[9,73]
   - 3 frac[9,53] + rows_tot_tor_P43 = 0
 R1236: - 0.0208333333333333 rows_tot_tor_P43 + normalized_rows_tot_tor_P43
   = 0
 mem_tot_l1_P41: - mem[10,61] - mem[10,7] - mem[10,36] - mem[10,14]
   - mem[10,47] - mem[10,19] - mem[10,48] - mem[10,26] - mem[10,59]
   - mem[10,5] - mem[10,73] - mem[10,12] - mem[10,22] - mem[10,45]
   - mem[10,55] - mem[10,17] - mem[10,69] - mem[10,62] - mem[10,57]
   - mem[10,3] - mem[10,32] - mem[10,10] - mem[10,53] - mem[10,31]
   - mem[10,67] - mem[10,60] - mem[10,6] - mem[10,74] - mem[10,39]
   - mem[10,1] - mem[10,46] - mem[10,8] - mem[10,18] - mem[10,70]
   - mem[10,41] - mem[10,51] - mem[10,29] - mem[10,65] - mem[10,58]
   - mem[10,72] - mem[10,37] - mem[10,15] - mem[10,44] - mem[10,54]
   - mem[10,49] - mem[10,27] - mem[10,56] - mem[10,2] - mem[10,35]
   - mem[10,13] - mem[10,23] - mem[10,52] - mem[10,30] - mem[10,66]
   - mem[10,63] - mem[10,25] - mem[10,38] - mem[10,0] - mem[10,33]
   - mem[10,11] - mem[10,21] - mem[10,40] - mem[10,50] - mem[10,28]
   - mem[10,64] + mem_tot_l1_P41 = 0
 R1238: - 1.6276041666666666e-04 mem_tot_l1_P41 + normalized_mem_tot_l1_P41
   = 0
 rows_tot_l1_P41: - 3 frac[10,61] - 3 frac[10,7] - 3 frac[10,36]
   - 3 frac[10,14] - 3 frac[10,47] - 3 frac[10,19] - 3 frac[10,48]
   - 3 frac[10,26] - 3 frac[10,59] - 3 frac[10,5] - 3 frac[10,73]
   - 3 frac[10,12] - 3 frac[10,22] - 3 frac[10,45] - 3 frac[10,55]
   - 3 frac[10,17] - 3 frac[10,69] - 3 frac[10,62] - 3 frac[10,57]
   - 3 frac[10,3] - 3 frac[10,32] - 3 frac[10,10] - 3 frac[10,53]
   - 3 frac[10,31] - 3 frac[10,67] - 3 frac[10,60] - 3 frac[10,6]
   - 3 frac[10,74] - 3 frac[10,39] - 3 frac[10,1] - 3 frac[10,46]
   - 3 frac[10,8] - 3 frac[10,18] - 3 frac[10,70] - 3 frac[10,41]
   - 3 frac[10,51] - 3 frac[10,29] - 3 frac[10,65] - 3 frac[10,58]
   - 3 frac[10,72] - 3 frac[10,37] - 3 frac[10,15] - 3 frac[10,44]
   - 3 frac[10,54] - 3 frac[10,49] - 3 frac[10,27] - 3 frac[10,56]
   - 3 frac[10,2] - 3 frac[10,35] - 3 frac[10,13] - 3 frac[10,23]
   - 3 frac[10,52] - 3 frac[10,30] - 3 frac[10,66] - 3 frac[10,63]
   - 3 frac[10,25] - 3 frac[10,38] - 3 frac[10,0] - 3 frac[10,33]
   - 3 frac[10,11] - 3 frac[10,21] - 3 frac[10,40] - 3 frac[10,50]
   - 3 frac[10,28] - 3 frac[10,64] + rows_tot_l1_P41 = 0
 R1240: - 0.0208333333333333 rows_tot_l1_P41 + normalized_rows_tot_l1_P41
   = 0
 mem_tot_l2_P4: - mem[11,48] - mem[11,22] - mem[11,41] - mem[11,72]
   - mem[11,46] - mem[11,26] - mem[11,65] - mem[11,39] - mem[11,61]
   - mem[11,19] - mem[11,50] - mem[11,8] - mem[11,1] - mem[11,74]
   - mem[11,32] - mem[11,6] - mem[11,28] - mem[11,67] - mem[11,21]
   - mem[11,52] - mem[11,10] - mem[11,45] - mem[11,25] - mem[11,56]
   - mem[11,30] - mem[11,69] - mem[11,23] - mem[11,54] - mem[11,12]
   - mem[11,73] - mem[11,47] - mem[11,5] - mem[11,58] - mem[11,51]
   - mem[11,40] - mem[11,14] - mem[11,33] - mem[11,29] - mem[11,64]
   - mem[11,38] - mem[11,60] - mem[11,18] - mem[11,53] - mem[11,11]
   - mem[11,0] - mem[11,35] - mem[11,57] - mem[11,31] - mem[11,66]
   - mem[11,62] - mem[11,13] - mem[11,44] - mem[11,2] - mem[11,37]
   - mem[11,59] - mem[11,17] + mem_tot_l2_P4 = 0
 R1242: - 1.6276041666666666e-04 mem_tot_l2_P4 + normalized_mem_tot_l2_P4
   = 0
 rows_tot_l2_P4: - 3 frac[11,48] - 3 frac[11,22] - 3 frac[11,41]
   - 3 frac[11,72] - 3 frac[11,46] - 3 frac[11,26] - 3 frac[11,65]
   - 3 frac[11,39] - 3 frac[11,61] - 3 frac[11,19] - 3 frac[11,50]
   - 3 frac[11,8] - 3 frac[11,1] - 3 frac[11,74] - 3 frac[11,32]
   - 3 frac[11,6] - 3 frac[11,28] - 3 frac[11,67] - 3 frac[11,21]
   - 3 frac[11,52] - 3 frac[11,10] - 3 frac[11,45] - 3 frac[11,25]
   - 3 frac[11,56] - 3 frac[11,30] - 3 frac[11,69] - 3 frac[11,23]
   - 3 frac[11,54] - 3 frac[11,12] - 3 frac[11,73] - 3 frac[11,47]
   - 3 frac[11,5] - 3 frac[11,58] - 3 frac[11,51] - 3 frac[11,40]
   - 3 frac[11,14] - 3 frac[11,33] - 3 frac[11,29] - 3 frac[11,64]
   - 3 frac[11,38] - 3 frac[11,60] - 3 frac[11,18] - 3 frac[11,53]
   - 3 frac[11,11] - 3 frac[11,0] - 3 frac[11,35] - 3 frac[11,57]
   - 3 frac[11,31] - 3 frac[11,66] - 3 frac[11,62] - 3 frac[11,13]
   - 3 frac[11,44] - 3 frac[11,2] - 3 frac[11,37] - 3 frac[11,59]
   - 3 frac[11,17] + rows_tot_l2_P4 = 0
 R1244: - 0.0208333333333333 rows_tot_l2_P4 + normalized_rows_tot_l2_P4
   = 0
 row_capacity_CPU2: rows_tot_CPU2 <= 12
 row_capacity_CPU3: rows_tot_CPU3 <= 12
 row_capacity_CPU4: rows_tot_CPU4 <= 12
 row_capacity_CPU7: rows_tot_CPU7 <= 12
 row_capacity_CPU17: rows_tot_CPU17 <= 12
 row_capacity_CPU18: rows_tot_CPU18 <= 12
 row_capacity_CPU24: rows_tot_CPU24 <= 12
 row_capacity_CPU26: rows_tot_CPU26 <= 12
 row_capacity_tor_P41: rows_tot_tor_P41 <= 48
 row_capacity_tor_P43: rows_tot_tor_P43 <= 48
 row_capacity_l1_P41: rows_tot_l1_P41 <= 48
 row_capacity_l2_P4: rows_tot_l2_P4 <= 48
 ns_single_CPU2: 3.5 rows_tot_CPU2 - ns_single_CPU2 + pdt_m_rows_CPU2 = 0
 capacity_cores_CPU2: cores_sketch_CPU2 + cores_dpdk_CPU2 <= 7
 ns_sketch_CPU2: - ns_single_CPU2 + 10.20408163265306 cores_sketch_CPU2
   >= 0
 ns_dpdk_CPU2: - 3.06122448979592 cores_dpdk_CPU2 <= -21.42857142857143
 ns_single_CPU3: 3.5 rows_tot_CPU3 - ns_single_CPU3 + pdt_m_rows_CPU3 = 0
 capacity_cores_CPU3: cores_sketch_CPU3 + cores_dpdk_CPU3 <= 7
 ns_sketch_CPU3: - ns_single_CPU3 + 10.20408163265306 cores_sketch_CPU3
   >= 0
 ns_dpdk_CPU3: - 3.06122448979592 cores_dpdk_CPU3 <= -21.42857142857143
 ns_single_CPU4: 3.5 rows_tot_CPU4 - ns_single_CPU4 + pdt_m_rows_CPU4 = 0
 capacity_cores_CPU4: cores_sketch_CPU4 + cores_dpdk_CPU4 <= 7
 ns_sketch_CPU4: - ns_single_CPU4 + 10.20408163265306 cores_sketch_CPU4
   >= 0
 ns_dpdk_CPU4: - 3.06122448979592 cores_dpdk_CPU4 <= -21.42857142857143
 ns_single_CPU7: 3.5 rows_tot_CPU7 - ns_single_CPU7 + pdt_m_rows_CPU7 = 0
 capacity_cores_CPU7: cores_sketch_CPU7 + cores_dpdk_CPU7 <= 7
 ns_sketch_CPU7: - ns_single_CPU7 + 10.20408163265306 cores_sketch_CPU7
   >= 0
 ns_dpdk_CPU7: - 3.06122448979592 cores_dpdk_CPU7 <= -21.42857142857143
 ns_single_CPU17: 3.5 rows_tot_CPU17 - ns_single_CPU17 + pdt_m_rows_CPU17
   = 0
 capacity_cores_CPU17: cores_sketch_CPU17 + cores_dpdk_CPU17 <= 7
 ns_sketch_CPU17: - ns_single_CPU17 + 10.20408163265306 cores_sketch_CPU17
   >= 0
 ns_dpdk_CPU17: - 3.06122448979592 cores_dpdk_CPU17 <= -21.42857142857143
 ns_single_CPU18: 3.5 rows_tot_CPU18 - ns_single_CPU18 + pdt_m_rows_CPU18
   = 0
 capacity_cores_CPU18: cores_sketch_CPU18 + cores_dpdk_CPU18 <= 7
 ns_sketch_CPU18: - ns_single_CPU18 + 10.20408163265306 cores_sketch_CPU18
   >= 0
 ns_dpdk_CPU18: - 3.06122448979592 cores_dpdk_CPU18 <= -21.42857142857143
 ns_single_CPU24: 3.5 rows_tot_CPU24 - ns_single_CPU24 + pdt_m_rows_CPU24
   = 0
 capacity_cores_CPU24: cores_sketch_CPU24 + cores_dpdk_CPU24 <= 7
 ns_sketch_CPU24: - ns_single_CPU24 + 10.20408163265306 cores_sketch_CPU24
   >= 0
 ns_dpdk_CPU24: - 3.06122448979592 cores_dpdk_CPU24 <= -21.42857142857143
 ns_single_CPU26: 3.5 rows_tot_CPU26 - ns_single_CPU26 + pdt_m_rows_CPU26
   = 0
 capacity_cores_CPU26: cores_sketch_CPU26 + cores_dpdk_CPU26 <= 7
 ns_sketch_CPU26: - ns_single_CPU26 + 10.20408163265306 cores_sketch_CPU26
   >= 0
 ns_dpdk_CPU26: - 3.06122448979592 cores_dpdk_CPU26 <= -21.42857142857143
 ns_tor_P41: ns_tor_P41 = 6.756756756756757
 max_ns_tor_P41: ns_tor_P41 <= 10.20408163265306
 ns_tor_P43: ns_tor_P43 = 6.756756756756757
 max_ns_tor_P43: ns_tor_P43 <= 10.20408163265306
 ns_l1_P41: ns_l1_P41 = 6.756756756756757
 max_ns_l1_P41: ns_l1_P41 <= 10.20408163265306
 ns_l2_P4: ns_l2_P4 = 6.756756756756757
 max_ns_l2_P4: ns_l2_P4 <= 10.20408163265306
 res_acc: 1.220703125e-04 mem_tot_CPU2 + 1.220703125e-04 mem_tot_CPU3
   + 1.220703125e-04 mem_tot_CPU4 + 1.220703125e-04 mem_tot_CPU7
   + 1.220703125e-04 mem_tot_CPU17 + 1.220703125e-04 mem_tot_CPU18
   + 1.220703125e-04 mem_tot_CPU24 + 1.220703125e-04 mem_tot_CPU26
   + 0.001953125 mem_tot_tor_P41 + 0.25 rows_tot_tor_P41
   + 0.001953125 mem_tot_tor_P43 + 0.25 rows_tot_tor_P43
   + 0.001953125 mem_tot_l1_P41 + 0.25 rows_tot_l1_P41
   + 0.001953125 mem_tot_l2_P4 + 0.25 rows_tot_l2_P4 + 10 cores_sketch_CPU2
   + 10 cores_dpdk_CPU2 + 10 cores_sketch_CPU3 + 10 cores_dpdk_CPU3
   + 10 cores_sketch_CPU4 + 10 cores_dpdk_CPU4 + 10 cores_sketch_CPU7
   + 10 cores_dpdk_CPU7 + 10 cores_sketch_CPU17 + 10 cores_dpdk_CPU17
   + 10 cores_sketch_CPU18 + 10 cores_dpdk_CPU18 + 10 cores_sketch_CPU24
   + 10 cores_dpdk_CPU24 + 10 cores_sketch_CPU26 + 10 cores_dpdk_CPU26
   - res = 0
 pdt_m_rows_CPU2: pdt_m_rows_CPU2 + [ - rows_tot_CPU2 * m_access_time_CPU2
   ] = 0
 pdt_m_rows_CPU3: pdt_m_rows_CPU3 + [ - rows_tot_CPU3 * m_access_time_CPU3
   ] = 0
 pdt_m_rows_CPU4: pdt_m_rows_CPU4 + [ - rows_tot_CPU4 * m_access_time_CPU4
   ] = 0
 pdt_m_rows_CPU7: pdt_m_rows_CPU7 + [ - rows_tot_CPU7 * m_access_time_CPU7
   ] = 0
 pdt_m_rows_CPU17: pdt_m_rows_CPU17 + [
   - rows_tot_CPU17 * m_access_time_CPU17 ] = 0
 pdt_m_rows_CPU18: pdt_m_rows_CPU18 + [
   - rows_tot_CPU18 * m_access_time_CPU18 ] = 0
 pdt_m_rows_CPU24: pdt_m_rows_CPU24 + [
   - rows_tot_CPU24 * m_access_time_CPU24 ] = 0
 pdt_m_rows_CPU26: pdt_m_rows_CPU26 + [
   - rows_tot_CPU26 * m_access_time_CPU26 ] = 0
Bounds
 frac[8,63] <= 1
 frac[9,62] <= 1
 frac[6,54] <= 1
 frac[4,66] <= 1
 frac[10,61] <= 1
 frac[8,5] <= 1
 frac[11,48] <= 1
 frac[9,0] <= 1
 frac[5,74] <= 1
 frac[10,7] <= 1
 frac[11,22] <= 1
 frac[4,5] <= 1
 frac[7,22] <= 1
 frac[9,39] <= 1
 frac[10,36] <= 1
 frac[8,12] <= 1
 frac[11,41] <= 1
 frac[10,14] <= 1
 frac[8,18] <= 1
 frac[9,19] <= 1
 frac[1,21] <= 1
 frac[2,12] <= 1
 frac[0,36] <= 1
 frac[5,43] <= 1
 frac[11,72] <= 1
 frac[6,32] <= 1
 frac[10,47] <= 1
 frac[2,64] <= 1
 frac[8,51] <= 1
 frac[11,46] <= 1
 frac[9,50] <= 1
 frac[8,25] <= 1
 frac[9,20] <= 1
 frac[10,19] <= 1
 frac[11,26] <= 1
 frac[8,69] <= 1
 frac[2,33] <= 1
 frac[11,65] <= 1
 frac[11,39] <= 1
 frac[9,59] <= 1
 frac[4,69] <= 1
 frac[10,48] <= 1
 frac[8,0] <= 1
 frac[11,61] <= 1
 frac[9,29] <= 1
 frac[10,26] <= 1
 frac[11,19] <= 1
 frac[1,25] <= 1
 frac[1,51] <= 1
 frac[6,18] <= 1
 frac[8,33] <= 1
 frac[5,39] <= 1
 frac[9,60] <= 1
 frac[10,59] <= 1
 frac[8,7] <= 1
 frac[11,50] <= 1
 frac[10,5] <= 1
 frac[11,8] <= 1
 frac[0,19] <= 1
 frac[2,25] <= 1
 frac[0,9] <= 1
 frac[10,73] <= 1
 frac[2,51] <= 1
 frac[9,67] <= 1
 frac[6,21] <= 1
 frac[4,45] <= 1
 frac[8,40] <= 1
 frac[0,69] <= 1
 frac[5,32] <= 1
 frac[9,37] <= 1
 frac[2,71] <= 1
 frac[8,14] <= 1
 frac[10,12] <= 1
 frac[11,1] <= 1
 frac[9,17] <= 1
 frac[10,22] <= 1
 frac[0,0] <= 1
 frac[8,64] <= 1
 frac[4,14] <= 1
 frac[9,68] <= 1
 frac[8,47] <= 1
 frac[11,74] <= 1
 frac[9,46] <= 1
 frac[10,45] <= 1
 frac[8,53] <= 1
 frac[7,39] <= 1
 frac[11,32] <= 1
 frac[10,55] <= 1
 frac[1,74] <= 1
 frac[8,27] <= 1
 frac[11,6] <= 1
 frac[10,17] <= 1
 frac[11,28] <= 1
 frac[8,71] <= 1
 frac[2,5] <= 1
 frac[1,56] <= 1
 frac[10,69] <= 1
 frac[11,67] <= 1
 frac[6,25] <= 1
 frac[4,57] <= 1
 frac[8,60] <= 1
 frac[9,57] <= 1
 frac[6,51] <= 1
 frac[10,62] <= 1
 frac[8,2] <= 1
 frac[11,21] <= 1
 frac[0,30] <= 1
 frac[2,54] <= 1
 frac[3,39] <= 1
 frac[8,35] <= 1
 frac[7,21] <= 1
 frac[6,42] <= 1
 frac[10,57] <= 1
 frac[2,74] <= 1
 frac[8,9] <= 1
 frac[7,51] <= 1
 frac[11,52] <= 1
 frac[9,4] <= 1
 frac[10,3] <= 1
 frac[11,10] <= 1
 frac[0,21] <= 1
 frac[1,16] <= 1
 frac[9,65] <= 1
 frac[0,71] <= 1
 frac[7,18] <= 1
 frac[9,43] <= 1
 frac[10,32] <= 1
 frac[2,69] <= 1
 frac[8,48] <= 1
 frac[11,45] <= 1
 frac[9,13] <= 1
 frac[6,71] <= 1
 frac[10,10] <= 1
 frac[8,22] <= 1
 frac[9,23] <= 1
 frac[11,25] <= 1
 frac[0,2] <= 1
 frac[8,66] <= 1
 frac[3,21] <= 1
 frac[9,74] <= 1
 frac[2,34] <= 1
 frac[9,44] <= 1
 frac[8,55] <= 1
 frac[7,33] <= 1
 frac[9,54] <= 1
 frac[10,53] <= 1
 frac[8,29] <= 1
 frac[11,56] <= 1
 frac[9,24] <= 1
 frac[5,66] <= 1
 frac[10,31] <= 1
 frac[11,30] <= 1
 frac[0,25] <= 1
 frac[8,73] <= 1
 frac[10,67] <= 1
 frac[6,5] <= 1
 frac[4,29] <= 1
 frac[11,69] <= 1
 frac[3,32] <= 1
 frac[8,62] <= 1
 frac[4,65] <= 1
 frac[10,60] <= 1
 frac[1,65] <= 1
 frac[9,1] <= 1
 frac[10,6] <= 1
 frac[11,23] <= 1
 frac[0,16] <= 1
 frac[7,69] <= 1
 frac[1,29] <= 1
 frac[10,74] <= 1
 frac[5,25] <= 1
 frac[8,37] <= 1
 frac[0,66] <= 1
 frac[9,32] <= 1
 frac[10,39] <= 1
 frac[11,54] <= 1
 frac[9,10] <= 1
 frac[10,1] <= 1
 frac[8,17] <= 1
 frac[11,12] <= 1
 frac[2,21] <= 1
 frac[9,71] <= 1
 frac[1,34] <= 1
 frac[7,12] <= 1
 frac[11,73] <= 1
 frac[9,41] <= 1
 frac[10,46] <= 1
 frac[2,67] <= 1
 frac[8,50] <= 1
 frac[11,47] <= 1
 frac[9,51] <= 1
 frac[10,8] <= 1
 frac[11,5] <= 1
 frac[9,21] <= 1
 frac[10,18] <= 1
 frac[9,72] <= 1
 frac[10,70] <= 1
 frac[2,32] <= 1
 frac[5,21] <= 1
 frac[3,61] <= 1
 frac[10,41] <= 1
 frac[8,57] <= 1
 frac[3,74] <= 1
 frac[10,51] <= 1
 frac[11,58] <= 1
 frac[9,30] <= 1
 frac[10,29] <= 1
 frac[10,65] <= 1
 frac[7,2] <= 1
 frac[5,30] <= 1
 frac[3,34] <= 1
 frac[8,32] <= 1
 frac[9,61] <= 1
 frac[10,58] <= 1
 frac[1,71] <= 1
 frac[11,51] <= 1
 frac[9,7] <= 1
 frac[7,71] <= 1
 frac[0,18] <= 1
 frac[1,19] <= 1
 frac[3,5] <= 1
 frac[10,72] <= 1
 frac[3,43] <= 1
 frac[8,39] <= 1
 frac[9,38] <= 1
 frac[10,37] <= 1
 frac[7,47] <= 1
 frac[11,40] <= 1
 frac[10,15] <= 1
 frac[8,19] <= 1
 frac[11,14] <= 1
 frac[9,18] <= 1
 frac[2,19] <= 1
 frac[3,16] <= 1
 frac[9,69] <= 1
 frac[1,32] <= 1
 frac[2,39] <= 1
 frac[10,44] <= 1
 frac[2,65] <= 1
 frac[11,33] <= 1
 frac[10,54] <= 1
 frac[8,26] <= 1
 frac[11,29] <= 1
 frac[8,70] <= 1
 frac[5,19] <= 1
 frac[11,64] <= 1
 frac[8,59] <= 1
 frac[11,38] <= 1
 frac[9,58] <= 1
 frac[10,49] <= 1
 frac[11,60] <= 1
 frac[9,28] <= 1
 frac[10,27] <= 1
 frac[11,18] <= 1
 frac[0,29] <= 1
 frac[4,25] <= 1
 frac[2,57] <= 1
 frac[3,36] <= 1
 frac[4,39] <= 1
 frac[8,34] <= 1
 frac[9,35] <= 1
 frac[10,56] <= 1
 frac[1,69] <= 1
 frac[11,53] <= 1
 frac[9,5] <= 1
 frac[10,2] <= 1
 frac[11,11] <= 1
 frac[7,65] <= 1
 frac[4,0] <= 1
 frac[0,32] <= 1
 frac[5,5] <= 1
 frac[9,66] <= 1
 frac[8,41] <= 1
 frac[10,35] <= 1
 frac[8,15] <= 1
 frac[9,14] <= 1
 frac[10,13] <= 1
 frac[8,21] <= 1
 frac[11,0] <= 1
 frac[10,23] <= 1
 frac[3,12] <= 1
 frac[8,65] <= 1
 frac[4,21] <= 1
 frac[9,45] <= 1
 frac[6,39] <= 1
 frac[8,54] <= 1
 frac[11,35] <= 1
 frac[9,55] <= 1
 frac[10,52] <= 1
 frac[8,28] <= 1
 frac[11,57] <= 1
 frac[9,25] <= 1
 frac[10,30] <= 1
 frac[11,31] <= 1
 frac[1,5] <= 1
 frac[8,72] <= 1
 frac[2,2] <= 1
 frac[10,66] <= 1
 frac[11,66] <= 1
 frac[3,33] <= 1
 frac[8,61] <= 1
 frac[0,74] <= 1
 frac[4,64] <= 1
 frac[10,63] <= 1
 frac[1,66] <= 1
 frac[8,3] <= 1
 frac[11,62] <= 1
 frac[9,2] <= 1
 frac[6,74] <= 1
 frac[10,25] <= 1
 frac[1,30] <= 1
 frac[0,43] <= 1
 frac[8,36] <= 1
 frac[0,65] <= 1
 frac[10,38] <= 1
 frac[10,0] <= 1
 frac[8,16] <= 1
 frac[11,13] <= 1
 frac[0,22] <= 1
 frac[0,12] <= 1
 frac[1,9] <= 1
 frac[4,2] <= 1
 frac[3,47] <= 1
 frac[9,64] <= 1
 frac[8,43] <= 1
 frac[9,42] <= 1
 frac[10,33] <= 1
 frac[2,66] <= 1
 frac[8,49] <= 1
 frac[7,43] <= 1
 frac[11,44] <= 1
 frac[3,66] <= 1
 frac[10,11] <= 1
 frac[8,23] <= 1
 frac[11,2] <= 1
 frac[10,21] <= 1
 frac[3,14] <= 1
 frac[8,67] <= 1
 frac[9,73] <= 1
 frac[4,61] <= 1
 frac[10,40] <= 1
 frac[8,56] <= 1
 frac[11,37] <= 1
 frac[9,53] <= 1
 frac[10,50] <= 1
 frac[8,30] <= 1
 frac[11,59] <= 1
 frac[5,65] <= 1
 frac[10,28] <= 1
 frac[11,17] <= 1
 frac[8,74] <= 1
 frac[10,64] <= 1
 frac[4,30] <= 1
 cols_8_63 <= 131072
 cols_9_62 <= 131072
 cols_10_61 <= 131072
 cols_8_5 <= 131072
 cols_11_48 <= 131072
 cols_9_0 <= 131072
 cols_10_7 <= 131072
 cols_11_22 <= 131072
 cols_9_39 <= 131072
 cols_10_36 <= 131072
 cols_8_12 <= 131072
 cols_11_41 <= 131072
 cols_10_14 <= 131072
 cols_8_18 <= 131072
 cols_9_19 <= 131072
 cols_11_72 <= 131072
 cols_10_47 <= 131072
 cols_8_51 <= 131072
 cols_11_46 <= 131072
 cols_9_50 <= 131072
 cols_8_25 <= 131072
 cols_9_20 <= 131072
 cols_10_19 <= 131072
 cols_11_26 <= 131072
 cols_8_69 <= 131072
 cols_11_65 <= 131072
 cols_11_39 <= 131072
 cols_9_59 <= 131072
 cols_10_48 <= 131072
 cols_8_0 <= 131072
 cols_11_61 <= 131072
 cols_9_29 <= 131072
 cols_10_26 <= 131072
 cols_11_19 <= 131072
 cols_8_33 <= 131072
 cols_9_60 <= 131072
 cols_10_59 <= 131072
 cols_8_7 <= 131072
 cols_11_50 <= 131072
 cols_10_5 <= 131072
 cols_11_8 <= 131072
 cols_10_73 <= 131072
 cols_9_67 <= 131072
 cols_8_40 <= 131072
 cols_9_37 <= 131072
 cols_8_14 <= 131072
 cols_10_12 <= 131072
 cols_11_1 <= 131072
 cols_9_17 <= 131072
 cols_10_22 <= 131072
 cols_8_64 <= 131072
 cols_9_68 <= 131072
 cols_8_47 <= 131072
 cols_11_74 <= 131072
 cols_9_46 <= 131072
 cols_10_45 <= 131072
 cols_8_53 <= 131072
 cols_11_32 <= 131072
 cols_10_55 <= 131072
 cols_8_27 <= 131072
 cols_11_6 <= 131072
 cols_10_17 <= 131072
 cols_11_28 <= 131072
 cols_8_71 <= 131072
 cols_10_69 <= 131072
 cols_11_67 <= 131072
 cols_8_60 <= 131072
 cols_9_57 <= 131072
 cols_10_62 <= 131072
 cols_8_2 <= 131072
 cols_11_21 <= 131072
 cols_8_35 <= 131072
 cols_10_57 <= 131072
 cols_8_9 <= 131072
 cols_11_52 <= 131072
 cols_9_4 <= 131072
 cols_10_3 <= 131072
 cols_11_10 <= 131072
 cols_9_65 <= 131072
 cols_9_43 <= 131072
 cols_10_32 <= 131072
 cols_8_48 <= 131072
 cols_11_45 <= 131072
 cols_9_13 <= 131072
 cols_10_10 <= 131072
 cols_8_22 <= 131072
 cols_9_23 <= 131072
 cols_11_25 <= 131072
 cols_8_66 <= 131072
 cols_9_74 <= 131072
 cols_9_44 <= 131072
 cols_8_55 <= 131072
 cols_9_54 <= 131072
 cols_10_53 <= 131072
 cols_8_29 <= 131072
 cols_11_56 <= 131072
 cols_9_24 <= 131072
 cols_10_31 <= 131072
 cols_11_30 <= 131072
 cols_8_73 <= 131072
 cols_10_67 <= 131072
 cols_11_69 <= 131072
 cols_8_62 <= 131072
 cols_10_60 <= 131072
 cols_9_1 <= 131072
 cols_10_6 <= 131072
 cols_11_23 <= 131072
 cols_10_74 <= 131072
 cols_8_37 <= 131072
 cols_9_32 <= 131072
 cols_10_39 <= 131072
 cols_11_54 <= 131072
 cols_9_10 <= 131072
 cols_10_1 <= 131072
 cols_8_17 <= 131072
 cols_11_12 <= 131072
 cols_9_71 <= 131072
 cols_11_73 <= 131072
 cols_9_41 <= 131072
 cols_10_46 <= 131072
 cols_8_50 <= 131072
 cols_11_47 <= 131072
 cols_9_51 <= 131072
 cols_10_8 <= 131072
 cols_11_5 <= 131072
 cols_9_21 <= 131072
 cols_10_18 <= 131072
 cols_9_72 <= 131072
 cols_10_70 <= 131072
 cols_10_41 <= 131072
 cols_8_57 <= 131072
 cols_10_51 <= 131072
 cols_11_58 <= 131072
 cols_9_30 <= 131072
 cols_10_29 <= 131072
 cols_10_65 <= 131072
 cols_8_32 <= 131072
 cols_9_61 <= 131072
 cols_10_58 <= 131072
 cols_11_51 <= 131072
 cols_9_7 <= 131072
 cols_10_72 <= 131072
 cols_8_39 <= 131072
 cols_9_38 <= 131072
 cols_10_37 <= 131072
 cols_11_40 <= 131072
 cols_10_15 <= 131072
 cols_8_19 <= 131072
 cols_11_14 <= 131072
 cols_9_18 <= 131072
 cols_9_69 <= 131072
 cols_10_44 <= 131072
 cols_11_33 <= 131072
 cols_10_54 <= 131072
 cols_8_26 <= 131072
 cols_11_29 <= 131072
 cols_8_70 <= 131072
 cols_11_64 <= 131072
 cols_8_59 <= 131072
 cols_11_38 <= 131072
 cols_9_58 <= 131072
 cols_10_49 <= 131072
 cols_11_60 <= 131072
 cols_9_28 <= 131072
 cols_10_27 <= 131072
 cols_11_18 <= 131072
 cols_8_34 <= 131072
 cols_9_35 <= 131072
 cols_10_56 <= 131072
 cols_11_53 <= 131072
 cols_9_5 <= 131072
 cols_10_2 <= 131072
 cols_11_11 <= 131072
 cols_9_66 <= 131072
 cols_8_41 <= 131072
 cols_10_35 <= 131072
 cols_8_15 <= 131072
 cols_9_14 <= 131072
 cols_10_13 <= 131072
 cols_8_21 <= 131072
 cols_11_0 <= 131072
 cols_10_23 <= 131072
 cols_8_65 <= 131072
 cols_9_45 <= 131072
 cols_8_54 <= 131072
 cols_11_35 <= 131072
 cols_9_55 <= 131072
 cols_10_52 <= 131072
 cols_8_28 <= 131072
 cols_11_57 <= 131072
 cols_9_25 <= 131072
 cols_10_30 <= 131072
 cols_11_31 <= 131072
 cols_8_72 <= 131072
 cols_10_66 <= 131072
 cols_11_66 <= 131072
 cols_8_61 <= 131072
 cols_10_63 <= 131072
 cols_8_3 <= 131072
 cols_11_62 <= 131072
 cols_9_2 <= 131072
 cols_10_25 <= 131072
 cols_8_36 <= 131072
 cols_10_38 <= 131072
 cols_10_0 <= 131072
 cols_8_16 <= 131072
 cols_11_13 <= 131072
 cols_9_64 <= 131072
 cols_8_43 <= 131072
 cols_9_42 <= 131072
 cols_10_33 <= 131072
 cols_8_49 <= 131072
 cols_11_44 <= 131072
 cols_10_11 <= 131072
 cols_8_23 <= 131072
 cols_11_2 <= 131072
 cols_10_21 <= 131072
 cols_8_67 <= 131072
 cols_9_73 <= 131072
 cols_10_40 <= 131072
 cols_8_56 <= 131072
 cols_11_37 <= 131072
 cols_9_53 <= 131072
 cols_10_50 <= 131072
 cols_8_30 <= 131072
 cols_11_59 <= 131072
 cols_10_28 <= 131072
 cols_11_17 <= 131072
 cols_8_74 <= 131072
 cols_10_64 <= 131072
 mem_tot_CPU2 <= 32768
 normalized_mem_tot_CPU2 <= 1
 normalized_rows_tot_CPU2 <= 1
 mem_tot_CPU3 <= 32768
 normalized_mem_tot_CPU3 <= 1
 normalized_rows_tot_CPU3 <= 1
 mem_tot_CPU4 <= 32768
 normalized_mem_tot_CPU4 <= 1
 normalized_rows_tot_CPU4 <= 1
 mem_tot_CPU7 <= 32768
 normalized_mem_tot_CPU7 <= 1
 normalized_rows_tot_CPU7 <= 1
 mem_tot_CPU17 <= 32768
 normalized_mem_tot_CPU17 <= 1
 normalized_rows_tot_CPU17 <= 1
 mem_tot_CPU18 <= 32768
 normalized_mem_tot_CPU18 <= 1
 normalized_rows_tot_CPU18 <= 1
 mem_tot_CPU24 <= 32768
 normalized_mem_tot_CPU24 <= 1
 normalized_rows_tot_CPU24 <= 1
 mem_tot_CPU26 <= 32768
 normalized_mem_tot_CPU26 <= 1
 normalized_rows_tot_CPU26 <= 1
 mem_tot_tor_P41 <= 6144
 normalized_mem_tot_tor_P41 <= 1
 normalized_rows_tot_tor_P41 <= 1
 mem_tot_tor_P43 <= 6144
 normalized_mem_tot_tor_P43 <= 1
 normalized_rows_tot_tor_P43 <= 1
 mem_tot_l1_P41 <= 6144
 normalized_mem_tot_l1_P41 <= 1
 normalized_rows_tot_l1_P41 <= 1
 mem_tot_l2_P4 <= 6144
 normalized_mem_tot_l2_P4 <= 1
 normalized_rows_tot_l2_P4 <= 1
 cores_sketch_CPU2 <= 7
 1 <= cores_dpdk_CPU2 <= 7
 cores_sketch_CPU3 <= 7
 1 <= cores_dpdk_CPU3 <= 7
 cores_sketch_CPU4 <= 7
 1 <= cores_dpdk_CPU4 <= 7
 cores_sketch_CPU7 <= 7
 1 <= cores_dpdk_CPU7 <= 7
 cores_sketch_CPU17 <= 7
 1 <= cores_dpdk_CPU17 <= 7
 cores_sketch_CPU18 <= 7
 1 <= cores_dpdk_CPU18 <= 7
 cores_sketch_CPU24 <= 7
 1 <= cores_dpdk_CPU24 <= 7
 cores_sketch_CPU26 <= 7
 1 <= cores_dpdk_CPU26 <= 7
Binaries
 cols_helper_8_63[0] cols_helper_8_63[1] cols_helper_8_63[2]
 cols_helper_8_63[3] cols_helper_8_63[4] cols_helper_8_63[5]
 cols_helper_8_63[6] cols_helper_8_63[7] cols_helper_8_63[8]
 cols_helper_8_63[9] cols_helper_8_63[10] cols_helper_8_63[11]
 cols_helper_8_63[12] cols_helper_8_63[13] cols_helper_8_63[14]
 cols_helper_8_63[15] cols_helper_8_63[16] cols_helper_9_62[0]
 cols_helper_9_62[1] cols_helper_9_62[2] cols_helper_9_62[3]
 cols_helper_9_62[4] cols_helper_9_62[5] cols_helper_9_62[6]
 cols_helper_9_62[7] cols_helper_9_62[8] cols_helper_9_62[9]
 cols_helper_9_62[10] cols_helper_9_62[11] cols_helper_9_62[12]
 cols_helper_9_62[13] cols_helper_9_62[14] cols_helper_9_62[15]
 cols_helper_9_62[16] cols_helper_10_61[0] cols_helper_10_61[1]
 cols_helper_10_61[2] cols_helper_10_61[3] cols_helper_10_61[4]
 cols_helper_10_61[5] cols_helper_10_61[6] cols_helper_10_61[7]
 cols_helper_10_61[8] cols_helper_10_61[9] cols_helper_10_61[10]
 cols_helper_10_61[11] cols_helper_10_61[12] cols_helper_10_61[13]
 cols_helper_10_61[14] cols_helper_10_61[15] cols_helper_10_61[16]
 cols_helper_8_5[0] cols_helper_8_5[1] cols_helper_8_5[2]
 cols_helper_8_5[3] cols_helper_8_5[4] cols_helper_8_5[5]
 cols_helper_8_5[6] cols_helper_8_5[7] cols_helper_8_5[8]
 cols_helper_8_5[9] cols_helper_8_5[10] cols_helper_8_5[11]
 cols_helper_8_5[12] cols_helper_8_5[13] cols_helper_8_5[14]
 cols_helper_8_5[15] cols_helper_8_5[16] cols_helper_11_48[0]
 cols_helper_11_48[1] cols_helper_11_48[2] cols_helper_11_48[3]
 cols_helper_11_48[4] cols_helper_11_48[5] cols_helper_11_48[6]
 cols_helper_11_48[7] cols_helper_11_48[8] cols_helper_11_48[9]
 cols_helper_11_48[10] cols_helper_11_48[11] cols_helper_11_48[12]
 cols_helper_11_48[13] cols_helper_11_48[14] cols_helper_11_48[15]
 cols_helper_11_48[16] cols_helper_9_0[0] cols_helper_9_0[1]
 cols_helper_9_0[2] cols_helper_9_0[3] cols_helper_9_0[4]
 cols_helper_9_0[5] cols_helper_9_0[6] cols_helper_9_0[7]
 cols_helper_9_0[8] cols_helper_9_0[9] cols_helper_9_0[10]
 cols_helper_9_0[11] cols_helper_9_0[12] cols_helper_9_0[13]
 cols_helper_9_0[14] cols_helper_9_0[15] cols_helper_9_0[16]
 cols_helper_10_7[0] cols_helper_10_7[1] cols_helper_10_7[2]
 cols_helper_10_7[3] cols_helper_10_7[4] cols_helper_10_7[5]
 cols_helper_10_7[6] cols_helper_10_7[7] cols_helper_10_7[8]
 cols_helper_10_7[9] cols_helper_10_7[10] cols_helper_10_7[11]
 cols_helper_10_7[12] cols_helper_10_7[13] cols_helper_10_7[14]
 cols_helper_10_7[15] cols_helper_10_7[16] cols_helper_11_22[0]
 cols_helper_11_22[1] cols_helper_11_22[2] cols_helper_11_22[3]
 cols_helper_11_22[4] cols_helper_11_22[5] cols_helper_11_22[6]
 cols_helper_11_22[7] cols_helper_11_22[8] cols_helper_11_22[9]
 cols_helper_11_22[10] cols_helper_11_22[11] cols_helper_11_22[12]
 cols_helper_11_22[13] cols_helper_11_22[14] cols_helper_11_22[15]
 cols_helper_11_22[16] cols_helper_9_39[0] cols_helper_9_39[1]
 cols_helper_9_39[2] cols_helper_9_39[3] cols_helper_9_39[4]
 cols_helper_9_39[5] cols_helper_9_39[6] cols_helper_9_39[7]
 cols_helper_9_39[8] cols_helper_9_39[9] cols_helper_9_39[10]
 cols_helper_9_39[11] cols_helper_9_39[12] cols_helper_9_39[13]
 cols_helper_9_39[14] cols_helper_9_39[15] cols_helper_9_39[16]
 cols_helper_10_36[0] cols_helper_10_36[1] cols_helper_10_36[2]
 cols_helper_10_36[3] cols_helper_10_36[4] cols_helper_10_36[5]
 cols_helper_10_36[6] cols_helper_10_36[7] cols_helper_10_36[8]
 cols_helper_10_36[9] cols_helper_10_36[10] cols_helper_10_36[11]
 cols_helper_10_36[12] cols_helper_10_36[13] cols_helper_10_36[14]
 cols_helper_10_36[15] cols_helper_10_36[16] cols_helper_8_12[0]
 cols_helper_8_12[1] cols_helper_8_12[2] cols_helper_8_12[3]
 cols_helper_8_12[4] cols_helper_8_12[5] cols_helper_8_12[6]
 cols_helper_8_12[7] cols_helper_8_12[8] cols_helper_8_12[9]
 cols_helper_8_12[10] cols_helper_8_12[11] cols_helper_8_12[12]
 cols_helper_8_12[13] cols_helper_8_12[14] cols_helper_8_12[15]
 cols_helper_8_12[16] cols_helper_11_41[0] cols_helper_11_41[1]
 cols_helper_11_41[2] cols_helper_11_41[3] cols_helper_11_41[4]
 cols_helper_11_41[5] cols_helper_11_41[6] cols_helper_11_41[7]
 cols_helper_11_41[8] cols_helper_11_41[9] cols_helper_11_41[10]
 cols_helper_11_41[11] cols_helper_11_41[12] cols_helper_11_41[13]
 cols_helper_11_41[14] cols_helper_11_41[15] cols_helper_11_41[16]
 cols_helper_10_14[0] cols_helper_10_14[1] cols_helper_10_14[2]
 cols_helper_10_14[3] cols_helper_10_14[4] cols_helper_10_14[5]
 cols_helper_10_14[6] cols_helper_10_14[7] cols_helper_10_14[8]
 cols_helper_10_14[9] cols_helper_10_14[10] cols_helper_10_14[11]
 cols_helper_10_14[12] cols_helper_10_14[13] cols_helper_10_14[14]
 cols_helper_10_14[15] cols_helper_10_14[16] cols_helper_8_18[0]
 cols_helper_8_18[1] cols_helper_8_18[2] cols_helper_8_18[3]
 cols_helper_8_18[4] cols_helper_8_18[5] cols_helper_8_18[6]
 cols_helper_8_18[7] cols_helper_8_18[8] cols_helper_8_18[9]
 cols_helper_8_18[10] cols_helper_8_18[11] cols_helper_8_18[12]
 cols_helper_8_18[13] cols_helper_8_18[14] cols_helper_8_18[15]
 cols_helper_8_18[16] cols_helper_9_19[0] cols_helper_9_19[1]
 cols_helper_9_19[2] cols_helper_9_19[3] cols_helper_9_19[4]
 cols_helper_9_19[5] cols_helper_9_19[6] cols_helper_9_19[7]
 cols_helper_9_19[8] cols_helper_9_19[9] cols_helper_9_19[10]
 cols_helper_9_19[11] cols_helper_9_19[12] cols_helper_9_19[13]
 cols_helper_9_19[14] cols_helper_9_19[15] cols_helper_9_19[16]
 cols_helper_11_72[0] cols_helper_11_72[1] cols_helper_11_72[2]
 cols_helper_11_72[3] cols_helper_11_72[4] cols_helper_11_72[5]
 cols_helper_11_72[6] cols_helper_11_72[7] cols_helper_11_72[8]
 cols_helper_11_72[9] cols_helper_11_72[10] cols_helper_11_72[11]
 cols_helper_11_72[12] cols_helper_11_72[13] cols_helper_11_72[14]
 cols_helper_11_72[15] cols_helper_11_72[16] cols_helper_10_47[0]
 cols_helper_10_47[1] cols_helper_10_47[2] cols_helper_10_47[3]
 cols_helper_10_47[4] cols_helper_10_47[5] cols_helper_10_47[6]
 cols_helper_10_47[7] cols_helper_10_47[8] cols_helper_10_47[9]
 cols_helper_10_47[10] cols_helper_10_47[11] cols_helper_10_47[12]
 cols_helper_10_47[13] cols_helper_10_47[14] cols_helper_10_47[15]
 cols_helper_10_47[16] cols_helper_8_51[0] cols_helper_8_51[1]
 cols_helper_8_51[2] cols_helper_8_51[3] cols_helper_8_51[4]
 cols_helper_8_51[5] cols_helper_8_51[6] cols_helper_8_51[7]
 cols_helper_8_51[8] cols_helper_8_51[9] cols_helper_8_51[10]
 cols_helper_8_51[11] cols_helper_8_51[12] cols_helper_8_51[13]
 cols_helper_8_51[14] cols_helper_8_51[15] cols_helper_8_51[16]
 cols_helper_11_46[0] cols_helper_11_46[1] cols_helper_11_46[2]
 cols_helper_11_46[3] cols_helper_11_46[4] cols_helper_11_46[5]
 cols_helper_11_46[6] cols_helper_11_46[7] cols_helper_11_46[8]
 cols_helper_11_46[9] cols_helper_11_46[10] cols_helper_11_46[11]
 cols_helper_11_46[12] cols_helper_11_46[13] cols_helper_11_46[14]
 cols_helper_11_46[15] cols_helper_11_46[16] cols_helper_9_50[0]
 cols_helper_9_50[1] cols_helper_9_50[2] cols_helper_9_50[3]
 cols_helper_9_50[4] cols_helper_9_50[5] cols_helper_9_50[6]
 cols_helper_9_50[7] cols_helper_9_50[8] cols_helper_9_50[9]
 cols_helper_9_50[10] cols_helper_9_50[11] cols_helper_9_50[12]
 cols_helper_9_50[13] cols_helper_9_50[14] cols_helper_9_50[15]
 cols_helper_9_50[16] cols_helper_8_25[0] cols_helper_8_25[1]
 cols_helper_8_25[2] cols_helper_8_25[3] cols_helper_8_25[4]
 cols_helper_8_25[5] cols_helper_8_25[6] cols_helper_8_25[7]
 cols_helper_8_25[8] cols_helper_8_25[9] cols_helper_8_25[10]
 cols_helper_8_25[11] cols_helper_8_25[12] cols_helper_8_25[13]
 cols_helper_8_25[14] cols_helper_8_25[15] cols_helper_8_25[16]
 cols_helper_9_20[0] cols_helper_9_20[1] cols_helper_9_20[2]
 cols_helper_9_20[3] cols_helper_9_20[4] cols_helper_9_20[5]
 cols_helper_9_20[6] cols_helper_9_20[7] cols_helper_9_20[8]
 cols_helper_9_20[9] cols_helper_9_20[10] cols_helper_9_20[11]
 cols_helper_9_20[12] cols_helper_9_20[13] cols_helper_9_20[14]
 cols_helper_9_20[15] cols_helper_9_20[16] cols_helper_10_19[0]
 cols_helper_10_19[1] cols_helper_10_19[2] cols_helper_10_19[3]
 cols_helper_10_19[4] cols_helper_10_19[5] cols_helper_10_19[6]
 cols_helper_10_19[7] cols_helper_10_19[8] cols_helper_10_19[9]
 cols_helper_10_19[10] cols_helper_10_19[11] cols_helper_10_19[12]
 cols_helper_10_19[13] cols_helper_10_19[14] cols_helper_10_19[15]
 cols_helper_10_19[16] cols_helper_11_26[0] cols_helper_11_26[1]
 cols_helper_11_26[2] cols_helper_11_26[3] cols_helper_11_26[4]
 cols_helper_11_26[5] cols_helper_11_26[6] cols_helper_11_26[7]
 cols_helper_11_26[8] cols_helper_11_26[9] cols_helper_11_26[10]
 cols_helper_11_26[11] cols_helper_11_26[12] cols_helper_11_26[13]
 cols_helper_11_26[14] cols_helper_11_26[15] cols_helper_11_26[16]
 cols_helper_8_69[0] cols_helper_8_69[1] cols_helper_8_69[2]
 cols_helper_8_69[3] cols_helper_8_69[4] cols_helper_8_69[5]
 cols_helper_8_69[6] cols_helper_8_69[7] cols_helper_8_69[8]
 cols_helper_8_69[9] cols_helper_8_69[10] cols_helper_8_69[11]
 cols_helper_8_69[12] cols_helper_8_69[13] cols_helper_8_69[14]
 cols_helper_8_69[15] cols_helper_8_69[16] cols_helper_11_65[0]
 cols_helper_11_65[1] cols_helper_11_65[2] cols_helper_11_65[3]
 cols_helper_11_65[4] cols_helper_11_65[5] cols_helper_11_65[6]
 cols_helper_11_65[7] cols_helper_11_65[8] cols_helper_11_65[9]
 cols_helper_11_65[10] cols_helper_11_65[11] cols_helper_11_65[12]
 cols_helper_11_65[13] cols_helper_11_65[14] cols_helper_11_65[15]
 cols_helper_11_65[16] cols_helper_11_39[0] cols_helper_11_39[1]
 cols_helper_11_39[2] cols_helper_11_39[3] cols_helper_11_39[4]
 cols_helper_11_39[5] cols_helper_11_39[6] cols_helper_11_39[7]
 cols_helper_11_39[8] cols_helper_11_39[9] cols_helper_11_39[10]
 cols_helper_11_39[11] cols_helper_11_39[12] cols_helper_11_39[13]
 cols_helper_11_39[14] cols_helper_11_39[15] cols_helper_11_39[16]
 cols_helper_9_59[0] cols_helper_9_59[1] cols_helper_9_59[2]
 cols_helper_9_59[3] cols_helper_9_59[4] cols_helper_9_59[5]
 cols_helper_9_59[6] cols_helper_9_59[7] cols_helper_9_59[8]
 cols_helper_9_59[9] cols_helper_9_59[10] cols_helper_9_59[11]
 cols_helper_9_59[12] cols_helper_9_59[13] cols_helper_9_59[14]
 cols_helper_9_59[15] cols_helper_9_59[16] cols_helper_10_48[0]
 cols_helper_10_48[1] cols_helper_10_48[2] cols_helper_10_48[3]
 cols_helper_10_48[4] cols_helper_10_48[5] cols_helper_10_48[6]
 cols_helper_10_48[7] cols_helper_10_48[8] cols_helper_10_48[9]
 cols_helper_10_48[10] cols_helper_10_48[11] cols_helper_10_48[12]
 cols_helper_10_48[13] cols_helper_10_48[14] cols_helper_10_48[15]
 cols_helper_10_48[16] cols_helper_8_0[0] cols_helper_8_0[1]
 cols_helper_8_0[2] cols_helper_8_0[3] cols_helper_8_0[4]
 cols_helper_8_0[5] cols_helper_8_0[6] cols_helper_8_0[7]
 cols_helper_8_0[8] cols_helper_8_0[9] cols_helper_8_0[10]
 cols_helper_8_0[11] cols_helper_8_0[12] cols_helper_8_0[13]
 cols_helper_8_0[14] cols_helper_8_0[15] cols_helper_8_0[16]
 cols_helper_11_61[0] cols_helper_11_61[1] cols_helper_11_61[2]
 cols_helper_11_61[3] cols_helper_11_61[4] cols_helper_11_61[5]
 cols_helper_11_61[6] cols_helper_11_61[7] cols_helper_11_61[8]
 cols_helper_11_61[9] cols_helper_11_61[10] cols_helper_11_61[11]
 cols_helper_11_61[12] cols_helper_11_61[13] cols_helper_11_61[14]
 cols_helper_11_61[15] cols_helper_11_61[16] cols_helper_9_29[0]
 cols_helper_9_29[1] cols_helper_9_29[2] cols_helper_9_29[3]
 cols_helper_9_29[4] cols_helper_9_29[5] cols_helper_9_29[6]
 cols_helper_9_29[7] cols_helper_9_29[8] cols_helper_9_29[9]
 cols_helper_9_29[10] cols_helper_9_29[11] cols_helper_9_29[12]
 cols_helper_9_29[13] cols_helper_9_29[14] cols_helper_9_29[15]
 cols_helper_9_29[16] cols_helper_10_26[0] cols_helper_10_26[1]
 cols_helper_10_26[2] cols_helper_10_26[3] cols_helper_10_26[4]
 cols_helper_10_26[5] cols_helper_10_26[6] cols_helper_10_26[7]
 cols_helper_10_26[8] cols_helper_10_26[9] cols_helper_10_26[10]
 cols_helper_10_26[11] cols_helper_10_26[12] cols_helper_10_26[13]
 cols_helper_10_26[14] cols_helper_10_26[15] cols_helper_10_26[16]
 cols_helper_11_19[0] cols_helper_11_19[1] cols_helper_11_19[2]
 cols_helper_11_19[3] cols_helper_11_19[4] cols_helper_11_19[5]
 cols_helper_11_19[6] cols_helper_11_19[7] cols_helper_11_19[8]
 cols_helper_11_19[9] cols_helper_11_19[10] cols_helper_11_19[11]
 cols_helper_11_19[12] cols_helper_11_19[13] cols_helper_11_19[14]
 cols_helper_11_19[15] cols_helper_11_19[16] cols_helper_8_33[0]
 cols_helper_8_33[1] cols_helper_8_33[2] cols_helper_8_33[3]
 cols_helper_8_33[4] cols_helper_8_33[5] cols_helper_8_33[6]
 cols_helper_8_33[7] cols_helper_8_33[8] cols_helper_8_33[9]
 cols_helper_8_33[10] cols_helper_8_33[11] cols_helper_8_33[12]
 cols_helper_8_33[13] cols_helper_8_33[14] cols_helper_8_33[15]
 cols_helper_8_33[16] cols_helper_9_60[0] cols_helper_9_60[1]
 cols_helper_9_60[2] cols_helper_9_60[3] cols_helper_9_60[4]
 cols_helper_9_60[5] cols_helper_9_60[6] cols_helper_9_60[7]
 cols_helper_9_60[8] cols_helper_9_60[9] cols_helper_9_60[10]
 cols_helper_9_60[11] cols_helper_9_60[12] cols_helper_9_60[13]
 cols_helper_9_60[14] cols_helper_9_60[15] cols_helper_9_60[16]
 cols_helper_10_59[0] cols_helper_10_59[1] cols_helper_10_59[2]
 cols_helper_10_59[3] cols_helper_10_59[4] cols_helper_10_59[5]
 cols_helper_10_59[6] cols_helper_10_59[7] cols_helper_10_59[8]
 cols_helper_10_59[9] cols_helper_10_59[10] cols_helper_10_59[11]
 cols_helper_10_59[12] cols_helper_10_59[13] cols_helper_10_59[14]
 cols_helper_10_59[15] cols_helper_10_59[16] cols_helper_8_7[0]
 cols_helper_8_7[1] cols_helper_8_7[2] cols_helper_8_7[3]
 cols_helper_8_7[4] cols_helper_8_7[5] cols_helper_8_7[6]
 cols_helper_8_7[7] cols_helper_8_7[8] cols_helper_8_7[9]
 cols_helper_8_7[10] cols_helper_8_7[11] cols_helper_8_7[12]
 cols_helper_8_7[13] cols_helper_8_7[14] cols_helper_8_7[15]
 cols_helper_8_7[16] cols_helper_11_50[0] cols_helper_11_50[1]
 cols_helper_11_50[2] cols_helper_11_50[3] cols_helper_11_50[4]
 cols_helper_11_50[5] cols_helper_11_50[6] cols_helper_11_50[7]
 cols_helper_11_50[8] cols_helper_11_50[9] cols_helper_11_50[10]
 cols_helper_11_50[11] cols_helper_11_50[12] cols_helper_11_50[13]
 cols_helper_11_50[14] cols_helper_11_50[15] cols_helper_11_50[16]
 cols_helper_10_5[0] cols_helper_10_5[1] cols_helper_10_5[2]
 cols_helper_10_5[3] cols_helper_10_5[4] cols_helper_10_5[5]
 cols_helper_10_5[6] cols_helper_10_5[7] cols_helper_10_5[8]
 cols_helper_10_5[9] cols_helper_10_5[10] cols_helper_10_5[11]
 cols_helper_10_5[12] cols_helper_10_5[13] cols_helper_10_5[14]
 cols_helper_10_5[15] cols_helper_10_5[16] cols_helper_11_8[0]
 cols_helper_11_8[1] cols_helper_11_8[2] cols_helper_11_8[3]
 cols_helper_11_8[4] cols_helper_11_8[5] cols_helper_11_8[6]
 cols_helper_11_8[7] cols_helper_11_8[8] cols_helper_11_8[9]
 cols_helper_11_8[10] cols_helper_11_8[11] cols_helper_11_8[12]
 cols_helper_11_8[13] cols_helper_11_8[14] cols_helper_11_8[15]
 cols_helper_11_8[16] cols_helper_10_73[0] cols_helper_10_73[1]
 cols_helper_10_73[2] cols_helper_10_73[3] cols_helper_10_73[4]
 cols_helper_10_73[5] cols_helper_10_73[6] cols_helper_10_73[7]
 cols_helper_10_73[8] cols_helper_10_73[9] cols_helper_10_73[10]
 cols_helper_10_73[11] cols_helper_10_73[12] cols_helper_10_73[13]
 cols_helper_10_73[14] cols_helper_10_73[15] cols_helper_10_73[16]
 cols_helper_9_67[0] cols_helper_9_67[1] cols_helper_9_67[2]
 cols_helper_9_67[3] cols_helper_9_67[4] cols_helper_9_67[5]
 cols_helper_9_67[6] cols_helper_9_67[7] cols_helper_9_67[8]
 cols_helper_9_67[9] cols_helper_9_67[10] cols_helper_9_67[11]
 cols_helper_9_67[12] cols_helper_9_67[13] cols_helper_9_67[14]
 cols_helper_9_67[15] cols_helper_9_67[16] cols_helper_8_40[0]
 cols_helper_8_40[1] cols_helper_8_40[2] cols_helper_8_40[3]
 cols_helper_8_40[4] cols_helper_8_40[5] cols_helper_8_40[6]
 cols_helper_8_40[7] cols_helper_8_40[8] cols_helper_8_40[9]
 cols_helper_8_40[10] cols_helper_8_40[11] cols_helper_8_40[12]
 cols_helper_8_40[13] cols_helper_8_40[14] cols_helper_8_40[15]
 cols_helper_8_40[16] cols_helper_9_37[0] cols_helper_9_37[1]
 cols_helper_9_37[2] cols_helper_9_37[3] cols_helper_9_37[4]
 cols_helper_9_37[5] cols_helper_9_37[6] cols_helper_9_37[7]
 cols_helper_9_37[8] cols_helper_9_37[9] cols_helper_9_37[10]
 cols_helper_9_37[11] cols_helper_9_37[12] cols_helper_9_37[13]
 cols_helper_9_37[14] cols_helper_9_37[15] cols_helper_9_37[16]
 cols_helper_8_14[0] cols_helper_8_14[1] cols_helper_8_14[2]
 cols_helper_8_14[3] cols_helper_8_14[4] cols_helper_8_14[5]
 cols_helper_8_14[6] cols_helper_8_14[7] cols_helper_8_14[8]
 cols_helper_8_14[9] cols_helper_8_14[10] cols_helper_8_14[11]
 cols_helper_8_14[12] cols_helper_8_14[13] cols_helper_8_14[14]
 cols_helper_8_14[15] cols_helper_8_14[16] cols_helper_10_12[0]
 cols_helper_10_12[1] cols_helper_10_12[2] cols_helper_10_12[3]
 cols_helper_10_12[4] cols_helper_10_12[5] cols_helper_10_12[6]
 cols_helper_10_12[7] cols_helper_10_12[8] cols_helper_10_12[9]
 cols_helper_10_12[10] cols_helper_10_12[11] cols_helper_10_12[12]
 cols_helper_10_12[13] cols_helper_10_12[14] cols_helper_10_12[15]
 cols_helper_10_12[16] cols_helper_11_1[0] cols_helper_11_1[1]
 cols_helper_11_1[2] cols_helper_11_1[3] cols_helper_11_1[4]
 cols_helper_11_1[5] cols_helper_11_1[6] cols_helper_11_1[7]
 cols_helper_11_1[8] cols_helper_11_1[9] cols_helper_11_1[10]
 cols_helper_11_1[11] cols_helper_11_1[12] cols_helper_11_1[13]
 cols_helper_11_1[14] cols_helper_11_1[15] cols_helper_11_1[16]
 cols_helper_9_17[0] cols_helper_9_17[1] cols_helper_9_17[2]
 cols_helper_9_17[3] cols_helper_9_17[4] cols_helper_9_17[5]
 cols_helper_9_17[6] cols_helper_9_17[7] cols_helper_9_17[8]
 cols_helper_9_17[9] cols_helper_9_17[10] cols_helper_9_17[11]
 cols_helper_9_17[12] cols_helper_9_17[13] cols_helper_9_17[14]
 cols_helper_9_17[15] cols_helper_9_17[16] cols_helper_10_22[0]
 cols_helper_10_22[1] cols_helper_10_22[2] cols_helper_10_22[3]
 cols_helper_10_22[4] cols_helper_10_22[5] cols_helper_10_22[6]
 cols_helper_10_22[7] cols_helper_10_22[8] cols_helper_10_22[9]
 cols_helper_10_22[10] cols_helper_10_22[11] cols_helper_10_22[12]
 cols_helper_10_22[13] cols_helper_10_22[14] cols_helper_10_22[15]
 cols_helper_10_22[16] cols_helper_8_64[0] cols_helper_8_64[1]
 cols_helper_8_64[2] cols_helper_8_64[3] cols_helper_8_64[4]
 cols_helper_8_64[5] cols_helper_8_64[6] cols_helper_8_64[7]
 cols_helper_8_64[8] cols_helper_8_64[9] cols_helper_8_64[10]
 cols_helper_8_64[11] cols_helper_8_64[12] cols_helper_8_64[13]
 cols_helper_8_64[14] cols_helper_8_64[15] cols_helper_8_64[16]
 cols_helper_9_68[0] cols_helper_9_68[1] cols_helper_9_68[2]
 cols_helper_9_68[3] cols_helper_9_68[4] cols_helper_9_68[5]
 cols_helper_9_68[6] cols_helper_9_68[7] cols_helper_9_68[8]
 cols_helper_9_68[9] cols_helper_9_68[10] cols_helper_9_68[11]
 cols_helper_9_68[12] cols_helper_9_68[13] cols_helper_9_68[14]
 cols_helper_9_68[15] cols_helper_9_68[16] cols_helper_8_47[0]
 cols_helper_8_47[1] cols_helper_8_47[2] cols_helper_8_47[3]
 cols_helper_8_47[4] cols_helper_8_47[5] cols_helper_8_47[6]
 cols_helper_8_47[7] cols_helper_8_47[8] cols_helper_8_47[9]
 cols_helper_8_47[10] cols_helper_8_47[11] cols_helper_8_47[12]
 cols_helper_8_47[13] cols_helper_8_47[14] cols_helper_8_47[15]
 cols_helper_8_47[16] cols_helper_11_74[0] cols_helper_11_74[1]
 cols_helper_11_74[2] cols_helper_11_74[3] cols_helper_11_74[4]
 cols_helper_11_74[5] cols_helper_11_74[6] cols_helper_11_74[7]
 cols_helper_11_74[8] cols_helper_11_74[9] cols_helper_11_74[10]
 cols_helper_11_74[11] cols_helper_11_74[12] cols_helper_11_74[13]
 cols_helper_11_74[14] cols_helper_11_74[15] cols_helper_11_74[16]
 cols_helper_9_46[0] cols_helper_9_46[1] cols_helper_9_46[2]
 cols_helper_9_46[3] cols_helper_9_46[4] cols_helper_9_46[5]
 cols_helper_9_46[6] cols_helper_9_46[7] cols_helper_9_46[8]
 cols_helper_9_46[9] cols_helper_9_46[10] cols_helper_9_46[11]
 cols_helper_9_46[12] cols_helper_9_46[13] cols_helper_9_46[14]
 cols_helper_9_46[15] cols_helper_9_46[16] cols_helper_10_45[0]
 cols_helper_10_45[1] cols_helper_10_45[2] cols_helper_10_45[3]
 cols_helper_10_45[4] cols_helper_10_45[5] cols_helper_10_45[6]
 cols_helper_10_45[7] cols_helper_10_45[8] cols_helper_10_45[9]
 cols_helper_10_45[10] cols_helper_10_45[11] cols_helper_10_45[12]
 cols_helper_10_45[13] cols_helper_10_45[14] cols_helper_10_45[15]
 cols_helper_10_45[16] cols_helper_8_53[0] cols_helper_8_53[1]
 cols_helper_8_53[2] cols_helper_8_53[3] cols_helper_8_53[4]
 cols_helper_8_53[5] cols_helper_8_53[6] cols_helper_8_53[7]
 cols_helper_8_53[8] cols_helper_8_53[9] cols_helper_8_53[10]
 cols_helper_8_53[11] cols_helper_8_53[12] cols_helper_8_53[13]
 cols_helper_8_53[14] cols_helper_8_53[15] cols_helper_8_53[16]
 cols_helper_11_32[0] cols_helper_11_32[1] cols_helper_11_32[2]
 cols_helper_11_32[3] cols_helper_11_32[4] cols_helper_11_32[5]
 cols_helper_11_32[6] cols_helper_11_32[7] cols_helper_11_32[8]
 cols_helper_11_32[9] cols_helper_11_32[10] cols_helper_11_32[11]
 cols_helper_11_32[12] cols_helper_11_32[13] cols_helper_11_32[14]
 cols_helper_11_32[15] cols_helper_11_32[16] cols_helper_10_55[0]
 cols_helper_10_55[1] cols_helper_10_55[2] cols_helper_10_55[3]
 cols_helper_10_55[4] cols_helper_10_55[5] cols_helper_10_55[6]
 cols_helper_10_55[7] cols_helper_10_55[8] cols_helper_10_55[9]
 cols_helper_10_55[10] cols_helper_10_55[11] cols_helper_10_55[12]
 cols_helper_10_55[13] cols_helper_10_55[14] cols_helper_10_55[15]
 cols_helper_10_55[16] cols_helper_8_27[0] cols_helper_8_27[1]
 cols_helper_8_27[2] cols_helper_8_27[3] cols_helper_8_27[4]
 cols_helper_8_27[5] cols_helper_8_27[6] cols_helper_8_27[7]
 cols_helper_8_27[8] cols_helper_8_27[9] cols_helper_8_27[10]
 cols_helper_8_27[11] cols_helper_8_27[12] cols_helper_8_27[13]
 cols_helper_8_27[14] cols_helper_8_27[15] cols_helper_8_27[16]
 cols_helper_11_6[0] cols_helper_11_6[1] cols_helper_11_6[2]
 cols_helper_11_6[3] cols_helper_11_6[4] cols_helper_11_6[5]
 cols_helper_11_6[6] cols_helper_11_6[7] cols_helper_11_6[8]
 cols_helper_11_6[9] cols_helper_11_6[10] cols_helper_11_6[11]
 cols_helper_11_6[12] cols_helper_11_6[13] cols_helper_11_6[14]
 cols_helper_11_6[15] cols_helper_11_6[16] cols_helper_10_17[0]
 cols_helper_10_17[1] cols_helper_10_17[2] cols_helper_10_17[3]
 cols_helper_10_17[4] cols_helper_10_17[5] cols_helper_10_17[6]
 cols_helper_10_17[7] cols_helper_10_17[8] cols_helper_10_17[9]
 cols_helper_10_17[10] cols_helper_10_17[11] cols_helper_10_17[12]
 cols_helper_10_17[13] cols_helper_10_17[14] cols_helper_10_17[15]
 cols_helper_10_17[16] cols_helper_11_28[0] cols_helper_11_28[1]
 cols_helper_11_28[2] cols_helper_11_28[3] cols_helper_11_28[4]
 cols_helper_11_28[5] cols_helper_11_28[6] cols_helper_11_28[7]
 cols_helper_11_28[8] cols_helper_11_28[9] cols_helper_11_28[10]
 cols_helper_11_28[11] cols_helper_11_28[12] cols_helper_11_28[13]
 cols_helper_11_28[14] cols_helper_11_28[15] cols_helper_11_28[16]
 cols_helper_8_71[0] cols_helper_8_71[1] cols_helper_8_71[2]
 cols_helper_8_71[3] cols_helper_8_71[4] cols_helper_8_71[5]
 cols_helper_8_71[6] cols_helper_8_71[7] cols_helper_8_71[8]
 cols_helper_8_71[9] cols_helper_8_71[10] cols_helper_8_71[11]
 cols_helper_8_71[12] cols_helper_8_71[13] cols_helper_8_71[14]
 cols_helper_8_71[15] cols_helper_8_71[16] cols_helper_10_69[0]
 cols_helper_10_69[1] cols_helper_10_69[2] cols_helper_10_69[3]
 cols_helper_10_69[4] cols_helper_10_69[5] cols_helper_10_69[6]
 cols_helper_10_69[7] cols_helper_10_69[8] cols_helper_10_69[9]
 cols_helper_10_69[10] cols_helper_10_69[11] cols_helper_10_69[12]
 cols_helper_10_69[13] cols_helper_10_69[14] cols_helper_10_69[15]
 cols_helper_10_69[16] cols_helper_11_67[0] cols_helper_11_67[1]
 cols_helper_11_67[2] cols_helper_11_67[3] cols_helper_11_67[4]
 cols_helper_11_67[5] cols_helper_11_67[6] cols_helper_11_67[7]
 cols_helper_11_67[8] cols_helper_11_67[9] cols_helper_11_67[10]
 cols_helper_11_67[11] cols_helper_11_67[12] cols_helper_11_67[13]
 cols_helper_11_67[14] cols_helper_11_67[15] cols_helper_11_67[16]
 cols_helper_8_60[0] cols_helper_8_60[1] cols_helper_8_60[2]
 cols_helper_8_60[3] cols_helper_8_60[4] cols_helper_8_60[5]
 cols_helper_8_60[6] cols_helper_8_60[7] cols_helper_8_60[8]
 cols_helper_8_60[9] cols_helper_8_60[10] cols_helper_8_60[11]
 cols_helper_8_60[12] cols_helper_8_60[13] cols_helper_8_60[14]
 cols_helper_8_60[15] cols_helper_8_60[16] cols_helper_9_57[0]
 cols_helper_9_57[1] cols_helper_9_57[2] cols_helper_9_57[3]
 cols_helper_9_57[4] cols_helper_9_57[5] cols_helper_9_57[6]
 cols_helper_9_57[7] cols_helper_9_57[8] cols_helper_9_57[9]
 cols_helper_9_57[10] cols_helper_9_57[11] cols_helper_9_57[12]
 cols_helper_9_57[13] cols_helper_9_57[14] cols_helper_9_57[15]
 cols_helper_9_57[16] cols_helper_10_62[0] cols_helper_10_62[1]
 cols_helper_10_62[2] cols_helper_10_62[3] cols_helper_10_62[4]
 cols_helper_10_62[5] cols_helper_10_62[6] cols_helper_10_62[7]
 cols_helper_10_62[8] cols_helper_10_62[9] cols_helper_10_62[10]
 cols_helper_10_62[11] cols_helper_10_62[12] cols_helper_10_62[13]
 cols_helper_10_62[14] cols_helper_10_62[15] cols_helper_10_62[16]
 cols_helper_8_2[0] cols_helper_8_2[1] cols_helper_8_2[2]
 cols_helper_8_2[3] cols_helper_8_2[4] cols_helper_8_2[5]
 cols_helper_8_2[6] cols_helper_8_2[7] cols_helper_8_2[8]
 cols_helper_8_2[9] cols_helper_8_2[10] cols_helper_8_2[11]
 cols_helper_8_2[12] cols_helper_8_2[13] cols_helper_8_2[14]
 cols_helper_8_2[15] cols_helper_8_2[16] cols_helper_11_21[0]
 cols_helper_11_21[1] cols_helper_11_21[2] cols_helper_11_21[3]
 cols_helper_11_21[4] cols_helper_11_21[5] cols_helper_11_21[6]
 cols_helper_11_21[7] cols_helper_11_21[8] cols_helper_11_21[9]
 cols_helper_11_21[10] cols_helper_11_21[11] cols_helper_11_21[12]
 cols_helper_11_21[13] cols_helper_11_21[14] cols_helper_11_21[15]
 cols_helper_11_21[16] cols_helper_8_35[0] cols_helper_8_35[1]
 cols_helper_8_35[2] cols_helper_8_35[3] cols_helper_8_35[4]
 cols_helper_8_35[5] cols_helper_8_35[6] cols_helper_8_35[7]
 cols_helper_8_35[8] cols_helper_8_35[9] cols_helper_8_35[10]
 cols_helper_8_35[11] cols_helper_8_35[12] cols_helper_8_35[13]
 cols_helper_8_35[14] cols_helper_8_35[15] cols_helper_8_35[16]
 cols_helper_10_57[0] cols_helper_10_57[1] cols_helper_10_57[2]
 cols_helper_10_57[3] cols_helper_10_57[4] cols_helper_10_57[5]
 cols_helper_10_57[6] cols_helper_10_57[7] cols_helper_10_57[8]
 cols_helper_10_57[9] cols_helper_10_57[10] cols_helper_10_57[11]
 cols_helper_10_57[12] cols_helper_10_57[13] cols_helper_10_57[14]
 cols_helper_10_57[15] cols_helper_10_57[16] cols_helper_8_9[0]
 cols_helper_8_9[1] cols_helper_8_9[2] cols_helper_8_9[3]
 cols_helper_8_9[4] cols_helper_8_9[5] cols_helper_8_9[6]
 cols_helper_8_9[7] cols_helper_8_9[8] cols_helper_8_9[9]
 cols_helper_8_9[10] cols_helper_8_9[11] cols_helper_8_9[12]
 cols_helper_8_9[13] cols_helper_8_9[14] cols_helper_8_9[15]
 cols_helper_8_9[16] cols_helper_11_52[0] cols_helper_11_52[1]
 cols_helper_11_52[2] cols_helper_11_52[3] cols_helper_11_52[4]
 cols_helper_11_52[5] cols_helper_11_52[6] cols_helper_11_52[7]
 cols_helper_11_52[8] cols_helper_11_52[9] cols_helper_11_52[10]
 cols_helper_11_52[11] cols_helper_11_52[12] cols_helper_11_52[13]
 cols_helper_11_52[14] cols_helper_11_52[15] cols_helper_11_52[16]
 cols_helper_9_4[0] cols_helper_9_4[1] cols_helper_9_4[2]
 cols_helper_9_4[3] cols_helper_9_4[4] cols_helper_9_4[5]
 cols_helper_9_4[6] cols_helper_9_4[7] cols_helper_9_4[8]
 cols_helper_9_4[9] cols_helper_9_4[10] cols_helper_9_4[11]
 cols_helper_9_4[12] cols_helper_9_4[13] cols_helper_9_4[14]
 cols_helper_9_4[15] cols_helper_9_4[16] cols_helper_10_3[0]
 cols_helper_10_3[1] cols_helper_10_3[2] cols_helper_10_3[3]
 cols_helper_10_3[4] cols_helper_10_3[5] cols_helper_10_3[6]
 cols_helper_10_3[7] cols_helper_10_3[8] cols_helper_10_3[9]
 cols_helper_10_3[10] cols_helper_10_3[11] cols_helper_10_3[12]
 cols_helper_10_3[13] cols_helper_10_3[14] cols_helper_10_3[15]
 cols_helper_10_3[16] cols_helper_11_10[0] cols_helper_11_10[1]
 cols_helper_11_10[2] cols_helper_11_10[3] cols_helper_11_10[4]
 cols_helper_11_10[5] cols_helper_11_10[6] cols_helper_11_10[7]
 cols_helper_11_10[8] cols_helper_11_10[9] cols_helper_11_10[10]
 cols_helper_11_10[11] cols_helper_11_10[12] cols_helper_11_10[13]
 cols_helper_11_10[14] cols_helper_11_10[15] cols_helper_11_10[16]
 cols_helper_9_65[0] cols_helper_9_65[1] cols_helper_9_65[2]
 cols_helper_9_65[3] cols_helper_9_65[4] cols_helper_9_65[5]
 cols_helper_9_65[6] cols_helper_9_65[7] cols_helper_9_65[8]
 cols_helper_9_65[9] cols_helper_9_65[10] cols_helper_9_65[11]
 cols_helper_9_65[12] cols_helper_9_65[13] cols_helper_9_65[14]
 cols_helper_9_65[15] cols_helper_9_65[16] cols_helper_9_43[0]
 cols_helper_9_43[1] cols_helper_9_43[2] cols_helper_9_43[3]
 cols_helper_9_43[4] cols_helper_9_43[5] cols_helper_9_43[6]
 cols_helper_9_43[7] cols_helper_9_43[8] cols_helper_9_43[9]
 cols_helper_9_43[10] cols_helper_9_43[11] cols_helper_9_43[12]
 cols_helper_9_43[13] cols_helper_9_43[14] cols_helper_9_43[15]
 cols_helper_9_43[16] cols_helper_10_32[0] cols_helper_10_32[1]
 cols_helper_10_32[2] cols_helper_10_32[3] cols_helper_10_32[4]
 cols_helper_10_32[5] cols_helper_10_32[6] cols_helper_10_32[7]
 cols_helper_10_32[8] cols_helper_10_32[9] cols_helper_10_32[10]
 cols_helper_10_32[11] cols_helper_10_32[12] cols_helper_10_32[13]
 cols_helper_10_32[14] cols_helper_10_32[15] cols_helper_10_32[16]
 cols_helper_8_48[0] cols_helper_8_48[1] cols_helper_8_48[2]
 cols_helper_8_48[3] cols_helper_8_48[4] cols_helper_8_48[5]
 cols_helper_8_48[6] cols_helper_8_48[7] cols_helper_8_48[8]
 cols_helper_8_48[9] cols_helper_8_48[10] cols_helper_8_48[11]
 cols_helper_8_48[12] cols_helper_8_48[13] cols_helper_8_48[14]
 cols_helper_8_48[15] cols_helper_8_48[16] cols_helper_11_45[0]
 cols_helper_11_45[1] cols_helper_11_45[2] cols_helper_11_45[3]
 cols_helper_11_45[4] cols_helper_11_45[5] cols_helper_11_45[6]
 cols_helper_11_45[7] cols_helper_11_45[8] cols_helper_11_45[9]
 cols_helper_11_45[10] cols_helper_11_45[11] cols_helper_11_45[12]
 cols_helper_11_45[13] cols_helper_11_45[14] cols_helper_11_45[15]
 cols_helper_11_45[16] cols_helper_9_13[0] cols_helper_9_13[1]
 cols_helper_9_13[2] cols_helper_9_13[3] cols_helper_9_13[4]
 cols_helper_9_13[5] cols_helper_9_13[6] cols_helper_9_13[7]
 cols_helper_9_13[8] cols_helper_9_13[9] cols_helper_9_13[10]
 cols_helper_9_13[11] cols_helper_9_13[12] cols_helper_9_13[13]
 cols_helper_9_13[14] cols_helper_9_13[15] cols_helper_9_13[16]
 cols_helper_10_10[0] cols_helper_10_10[1] cols_helper_10_10[2]
 cols_helper_10_10[3] cols_helper_10_10[4] cols_helper_10_10[5]
 cols_helper_10_10[6] cols_helper_10_10[7] cols_helper_10_10[8]
 cols_helper_10_10[9] cols_helper_10_10[10] cols_helper_10_10[11]
 cols_helper_10_10[12] cols_helper_10_10[13] cols_helper_10_10[14]
 cols_helper_10_10[15] cols_helper_10_10[16] cols_helper_8_22[0]
 cols_helper_8_22[1] cols_helper_8_22[2] cols_helper_8_22[3]
 cols_helper_8_22[4] cols_helper_8_22[5] cols_helper_8_22[6]
 cols_helper_8_22[7] cols_helper_8_22[8] cols_helper_8_22[9]
 cols_helper_8_22[10] cols_helper_8_22[11] cols_helper_8_22[12]
 cols_helper_8_22[13] cols_helper_8_22[14] cols_helper_8_22[15]
 cols_helper_8_22[16] cols_helper_9_23[0] cols_helper_9_23[1]
 cols_helper_9_23[2] cols_helper_9_23[3] cols_helper_9_23[4]
 cols_helper_9_23[5] cols_helper_9_23[6] cols_helper_9_23[7]
 cols_helper_9_23[8] cols_helper_9_23[9] cols_helper_9_23[10]
 cols_helper_9_23[11] cols_helper_9_23[12] cols_helper_9_23[13]
 cols_helper_9_23[14] cols_helper_9_23[15] cols_helper_9_23[16]
 cols_helper_11_25[0] cols_helper_11_25[1] cols_helper_11_25[2]
 cols_helper_11_25[3] cols_helper_11_25[4] cols_helper_11_25[5]
 cols_helper_11_25[6] cols_helper_11_25[7] cols_helper_11_25[8]
 cols_helper_11_25[9] cols_helper_11_25[10] cols_helper_11_25[11]
 cols_helper_11_25[12] cols_helper_11_25[13] cols_helper_11_25[14]
 cols_helper_11_25[15] cols_helper_11_25[16] cols_helper_8_66[0]
 cols_helper_8_66[1] cols_helper_8_66[2] cols_helper_8_66[3]
 cols_helper_8_66[4] cols_helper_8_66[5] cols_helper_8_66[6]
 cols_helper_8_66[7] cols_helper_8_66[8] cols_helper_8_66[9]
 cols_helper_8_66[10] cols_helper_8_66[11] cols_helper_8_66[12]
 cols_helper_8_66[13] cols_helper_8_66[14] cols_helper_8_66[15]
 cols_helper_8_66[16] cols_helper_9_74[0] cols_helper_9_74[1]
 cols_helper_9_74[2] cols_helper_9_74[3] cols_helper_9_74[4]
 cols_helper_9_74[5] cols_helper_9_74[6] cols_helper_9_74[7]
 cols_helper_9_74[8] cols_helper_9_74[9] cols_helper_9_74[10]
 cols_helper_9_74[11] cols_helper_9_74[12] cols_helper_9_74[13]
 cols_helper_9_74[14] cols_helper_9_74[15] cols_helper_9_74[16]
 cols_helper_9_44[0] cols_helper_9_44[1] cols_helper_9_44[2]
 cols_helper_9_44[3] cols_helper_9_44[4] cols_helper_9_44[5]
 cols_helper_9_44[6] cols_helper_9_44[7] cols_helper_9_44[8]
 cols_helper_9_44[9] cols_helper_9_44[10] cols_helper_9_44[11]
 cols_helper_9_44[12] cols_helper_9_44[13] cols_helper_9_44[14]
 cols_helper_9_44[15] cols_helper_9_44[16] cols_helper_8_55[0]
 cols_helper_8_55[1] cols_helper_8_55[2] cols_helper_8_55[3]
 cols_helper_8_55[4] cols_helper_8_55[5] cols_helper_8_55[6]
 cols_helper_8_55[7] cols_helper_8_55[8] cols_helper_8_55[9]
 cols_helper_8_55[10] cols_helper_8_55[11] cols_helper_8_55[12]
 cols_helper_8_55[13] cols_helper_8_55[14] cols_helper_8_55[15]
 cols_helper_8_55[16] cols_helper_9_54[0] cols_helper_9_54[1]
 cols_helper_9_54[2] cols_helper_9_54[3] cols_helper_9_54[4]
 cols_helper_9_54[5] cols_helper_9_54[6] cols_helper_9_54[7]
 cols_helper_9_54[8] cols_helper_9_54[9] cols_helper_9_54[10]
 cols_helper_9_54[11] cols_helper_9_54[12] cols_helper_9_54[13]
 cols_helper_9_54[14] cols_helper_9_54[15] cols_helper_9_54[16]
 cols_helper_10_53[0] cols_helper_10_53[1] cols_helper_10_53[2]
 cols_helper_10_53[3] cols_helper_10_53[4] cols_helper_10_53[5]
 cols_helper_10_53[6] cols_helper_10_53[7] cols_helper_10_53[8]
 cols_helper_10_53[9] cols_helper_10_53[10] cols_helper_10_53[11]
 cols_helper_10_53[12] cols_helper_10_53[13] cols_helper_10_53[14]
 cols_helper_10_53[15] cols_helper_10_53[16] cols_helper_8_29[0]
 cols_helper_8_29[1] cols_helper_8_29[2] cols_helper_8_29[3]
 cols_helper_8_29[4] cols_helper_8_29[5] cols_helper_8_29[6]
 cols_helper_8_29[7] cols_helper_8_29[8] cols_helper_8_29[9]
 cols_helper_8_29[10] cols_helper_8_29[11] cols_helper_8_29[12]
 cols_helper_8_29[13] cols_helper_8_29[14] cols_helper_8_29[15]
 cols_helper_8_29[16] cols_helper_11_56[0] cols_helper_11_56[1]
 cols_helper_11_56[2] cols_helper_11_56[3] cols_helper_11_56[4]
 cols_helper_11_56[5] cols_helper_11_56[6] cols_helper_11_56[7]
 cols_helper_11_56[8] cols_helper_11_56[9] cols_helper_11_56[10]
 cols_helper_11_56[11] cols_helper_11_56[12] cols_helper_11_56[13]
 cols_helper_11_56[14] cols_helper_11_56[15] cols_helper_11_56[16]
 cols_helper_9_24[0] cols_helper_9_24[1] cols_helper_9_24[2]
 cols_helper_9_24[3] cols_helper_9_24[4] cols_helper_9_24[5]
 cols_helper_9_24[6] cols_helper_9_24[7] cols_helper_9_24[8]
 cols_helper_9_24[9] cols_helper_9_24[10] cols_helper_9_24[11]
 cols_helper_9_24[12] cols_helper_9_24[13] cols_helper_9_24[14]
 cols_helper_9_24[15] cols_helper_9_24[16] cols_helper_10_31[0]
 cols_helper_10_31[1] cols_helper_10_31[2] cols_helper_10_31[3]
 cols_helper_10_31[4] cols_helper_10_31[5] cols_helper_10_31[6]
 cols_helper_10_31[7] cols_helper_10_31[8] cols_helper_10_31[9]
 cols_helper_10_31[10] cols_helper_10_31[11] cols_helper_10_31[12]
 cols_helper_10_31[13] cols_helper_10_31[14] cols_helper_10_31[15]
 cols_helper_10_31[16] cols_helper_11_30[0] cols_helper_11_30[1]
 cols_helper_11_30[2] cols_helper_11_30[3] cols_helper_11_30[4]
 cols_helper_11_30[5] cols_helper_11_30[6] cols_helper_11_30[7]
 cols_helper_11_30[8] cols_helper_11_30[9] cols_helper_11_30[10]
 cols_helper_11_30[11] cols_helper_11_30[12] cols_helper_11_30[13]
 cols_helper_11_30[14] cols_helper_11_30[15] cols_helper_11_30[16]
 cols_helper_8_73[0] cols_helper_8_73[1] cols_helper_8_73[2]
 cols_helper_8_73[3] cols_helper_8_73[4] cols_helper_8_73[5]
 cols_helper_8_73[6] cols_helper_8_73[7] cols_helper_8_73[8]
 cols_helper_8_73[9] cols_helper_8_73[10] cols_helper_8_73[11]
 cols_helper_8_73[12] cols_helper_8_73[13] cols_helper_8_73[14]
 cols_helper_8_73[15] cols_helper_8_73[16] cols_helper_10_67[0]
 cols_helper_10_67[1] cols_helper_10_67[2] cols_helper_10_67[3]
 cols_helper_10_67[4] cols_helper_10_67[5] cols_helper_10_67[6]
 cols_helper_10_67[7] cols_helper_10_67[8] cols_helper_10_67[9]
 cols_helper_10_67[10] cols_helper_10_67[11] cols_helper_10_67[12]
 cols_helper_10_67[13] cols_helper_10_67[14] cols_helper_10_67[15]
 cols_helper_10_67[16] cols_helper_11_69[0] cols_helper_11_69[1]
 cols_helper_11_69[2] cols_helper_11_69[3] cols_helper_11_69[4]
 cols_helper_11_69[5] cols_helper_11_69[6] cols_helper_11_69[7]
 cols_helper_11_69[8] cols_helper_11_69[9] cols_helper_11_69[10]
 cols_helper_11_69[11] cols_helper_11_69[12] cols_helper_11_69[13]
 cols_helper_11_69[14] cols_helper_11_69[15] cols_helper_11_69[16]
 cols_helper_8_62[0] cols_helper_8_62[1] cols_helper_8_62[2]
 cols_helper_8_62[3] cols_helper_8_62[4] cols_helper_8_62[5]
 cols_helper_8_62[6] cols_helper_8_62[7] cols_helper_8_62[8]
 cols_helper_8_62[9] cols_helper_8_62[10] cols_helper_8_62[11]
 cols_helper_8_62[12] cols_helper_8_62[13] cols_helper_8_62[14]
 cols_helper_8_62[15] cols_helper_8_62[16] cols_helper_10_60[0]
 cols_helper_10_60[1] cols_helper_10_60[2] cols_helper_10_60[3]
 cols_helper_10_60[4] cols_helper_10_60[5] cols_helper_10_60[6]
 cols_helper_10_60[7] cols_helper_10_60[8] cols_helper_10_60[9]
 cols_helper_10_60[10] cols_helper_10_60[11] cols_helper_10_60[12]
 cols_helper_10_60[13] cols_helper_10_60[14] cols_helper_10_60[15]
 cols_helper_10_60[16] cols_helper_9_1[0] cols_helper_9_1[1]
 cols_helper_9_1[2] cols_helper_9_1[3] cols_helper_9_1[4]
 cols_helper_9_1[5] cols_helper_9_1[6] cols_helper_9_1[7]
 cols_helper_9_1[8] cols_helper_9_1[9] cols_helper_9_1[10]
 cols_helper_9_1[11] cols_helper_9_1[12] cols_helper_9_1[13]
 cols_helper_9_1[14] cols_helper_9_1[15] cols_helper_9_1[16]
 cols_helper_10_6[0] cols_helper_10_6[1] cols_helper_10_6[2]
 cols_helper_10_6[3] cols_helper_10_6[4] cols_helper_10_6[5]
 cols_helper_10_6[6] cols_helper_10_6[7] cols_helper_10_6[8]
 cols_helper_10_6[9] cols_helper_10_6[10] cols_helper_10_6[11]
 cols_helper_10_6[12] cols_helper_10_6[13] cols_helper_10_6[14]
 cols_helper_10_6[15] cols_helper_10_6[16] cols_helper_11_23[0]
 cols_helper_11_23[1] cols_helper_11_23[2] cols_helper_11_23[3]
 cols_helper_11_23[4] cols_helper_11_23[5] cols_helper_11_23[6]
 cols_helper_11_23[7] cols_helper_11_23[8] cols_helper_11_23[9]
 cols_helper_11_23[10] cols_helper_11_23[11] cols_helper_11_23[12]
 cols_helper_11_23[13] cols_helper_11_23[14] cols_helper_11_23[15]
 cols_helper_11_23[16] cols_helper_10_74[0] cols_helper_10_74[1]
 cols_helper_10_74[2] cols_helper_10_74[3] cols_helper_10_74[4]
 cols_helper_10_74[5] cols_helper_10_74[6] cols_helper_10_74[7]
 cols_helper_10_74[8] cols_helper_10_74[9] cols_helper_10_74[10]
 cols_helper_10_74[11] cols_helper_10_74[12] cols_helper_10_74[13]
 cols_helper_10_74[14] cols_helper_10_74[15] cols_helper_10_74[16]
 cols_helper_8_37[0] cols_helper_8_37[1] cols_helper_8_37[2]
 cols_helper_8_37[3] cols_helper_8_37[4] cols_helper_8_37[5]
 cols_helper_8_37[6] cols_helper_8_37[7] cols_helper_8_37[8]
 cols_helper_8_37[9] cols_helper_8_37[10] cols_helper_8_37[11]
 cols_helper_8_37[12] cols_helper_8_37[13] cols_helper_8_37[14]
 cols_helper_8_37[15] cols_helper_8_37[16] cols_helper_9_32[0]
 cols_helper_9_32[1] cols_helper_9_32[2] cols_helper_9_32[3]
 cols_helper_9_32[4] cols_helper_9_32[5] cols_helper_9_32[6]
 cols_helper_9_32[7] cols_helper_9_32[8] cols_helper_9_32[9]
 cols_helper_9_32[10] cols_helper_9_32[11] cols_helper_9_32[12]
 cols_helper_9_32[13] cols_helper_9_32[14] cols_helper_9_32[15]
 cols_helper_9_32[16] cols_helper_10_39[0] cols_helper_10_39[1]
 cols_helper_10_39[2] cols_helper_10_39[3] cols_helper_10_39[4]
 cols_helper_10_39[5] cols_helper_10_39[6] cols_helper_10_39[7]
 cols_helper_10_39[8] cols_helper_10_39[9] cols_helper_10_39[10]
 cols_helper_10_39[11] cols_helper_10_39[12] cols_helper_10_39[13]
 cols_helper_10_39[14] cols_helper_10_39[15] cols_helper_10_39[16]
 cols_helper_11_54[0] cols_helper_11_54[1] cols_helper_11_54[2]
 cols_helper_11_54[3] cols_helper_11_54[4] cols_helper_11_54[5]
 cols_helper_11_54[6] cols_helper_11_54[7] cols_helper_11_54[8]
 cols_helper_11_54[9] cols_helper_11_54[10] cols_helper_11_54[11]
 cols_helper_11_54[12] cols_helper_11_54[13] cols_helper_11_54[14]
 cols_helper_11_54[15] cols_helper_11_54[16] cols_helper_9_10[0]
 cols_helper_9_10[1] cols_helper_9_10[2] cols_helper_9_10[3]
 cols_helper_9_10[4] cols_helper_9_10[5] cols_helper_9_10[6]
 cols_helper_9_10[7] cols_helper_9_10[8] cols_helper_9_10[9]
 cols_helper_9_10[10] cols_helper_9_10[11] cols_helper_9_10[12]
 cols_helper_9_10[13] cols_helper_9_10[14] cols_helper_9_10[15]
 cols_helper_9_10[16] cols_helper_10_1[0] cols_helper_10_1[1]
 cols_helper_10_1[2] cols_helper_10_1[3] cols_helper_10_1[4]
 cols_helper_10_1[5] cols_helper_10_1[6] cols_helper_10_1[7]
 cols_helper_10_1[8] cols_helper_10_1[9] cols_helper_10_1[10]
 cols_helper_10_1[11] cols_helper_10_1[12] cols_helper_10_1[13]
 cols_helper_10_1[14] cols_helper_10_1[15] cols_helper_10_1[16]
 cols_helper_8_17[0] cols_helper_8_17[1] cols_helper_8_17[2]
 cols_helper_8_17[3] cols_helper_8_17[4] cols_helper_8_17[5]
 cols_helper_8_17[6] cols_helper_8_17[7] cols_helper_8_17[8]
 cols_helper_8_17[9] cols_helper_8_17[10] cols_helper_8_17[11]
 cols_helper_8_17[12] cols_helper_8_17[13] cols_helper_8_17[14]
 cols_helper_8_17[15] cols_helper_8_17[16] cols_helper_11_12[0]
 cols_helper_11_12[1] cols_helper_11_12[2] cols_helper_11_12[3]
 cols_helper_11_12[4] cols_helper_11_12[5] cols_helper_11_12[6]
 cols_helper_11_12[7] cols_helper_11_12[8] cols_helper_11_12[9]
 cols_helper_11_12[10] cols_helper_11_12[11] cols_helper_11_12[12]
 cols_helper_11_12[13] cols_helper_11_12[14] cols_helper_11_12[15]
 cols_helper_11_12[16] cols_helper_9_71[0] cols_helper_9_71[1]
 cols_helper_9_71[2] cols_helper_9_71[3] cols_helper_9_71[4]
 cols_helper_9_71[5] cols_helper_9_71[6] cols_helper_9_71[7]
 cols_helper_9_71[8] cols_helper_9_71[9] cols_helper_9_71[10]
 cols_helper_9_71[11] cols_helper_9_71[12] cols_helper_9_71[13]
 cols_helper_9_71[14] cols_helper_9_71[15] cols_helper_9_71[16]
 cols_helper_11_73[0] cols_helper_11_73[1] cols_helper_11_73[2]
 cols_helper_11_73[3] cols_helper_11_73[4] cols_helper_11_73[5]
 cols_helper_11_73[6] cols_helper_11_73[7] cols_helper_11_73[8]
 cols_helper_11_73[9] cols_helper_11_73[10] cols_helper_11_73[11]
 cols_helper_11_73[12] cols_helper_11_73[13] cols_helper_11_73[14]
 cols_helper_11_73[15] cols_helper_11_73[16] cols_helper_9_41[0]
 cols_helper_9_41[1] cols_helper_9_41[2] cols_helper_9_41[3]
 cols_helper_9_41[4] cols_helper_9_41[5] cols_helper_9_41[6]
 cols_helper_9_41[7] cols_helper_9_41[8] cols_helper_9_41[9]
 cols_helper_9_41[10] cols_helper_9_41[11] cols_helper_9_41[12]
 cols_helper_9_41[13] cols_helper_9_41[14] cols_helper_9_41[15]
 cols_helper_9_41[16] cols_helper_10_46[0] cols_helper_10_46[1]
 cols_helper_10_46[2] cols_helper_10_46[3] cols_helper_10_46[4]
 cols_helper_10_46[5] cols_helper_10_46[6] cols_helper_10_46[7]
 cols_helper_10_46[8] cols_helper_10_46[9] cols_helper_10_46[10]
 cols_helper_10_46[11] cols_helper_10_46[12] cols_helper_10_46[13]
 cols_helper_10_46[14] cols_helper_10_46[15] cols_helper_10_46[16]
 cols_helper_8_50[0] cols_helper_8_50[1] cols_helper_8_50[2]
 cols_helper_8_50[3] cols_helper_8_50[4] cols_helper_8_50[5]
 cols_helper_8_50[6] cols_helper_8_50[7] cols_helper_8_50[8]
 cols_helper_8_50[9] cols_helper_8_50[10] cols_helper_8_50[11]
 cols_helper_8_50[12] cols_helper_8_50[13] cols_helper_8_50[14]
 cols_helper_8_50[15] cols_helper_8_50[16] cols_helper_11_47[0]
 cols_helper_11_47[1] cols_helper_11_47[2] cols_helper_11_47[3]
 cols_helper_11_47[4] cols_helper_11_47[5] cols_helper_11_47[6]
 cols_helper_11_47[7] cols_helper_11_47[8] cols_helper_11_47[9]
 cols_helper_11_47[10] cols_helper_11_47[11] cols_helper_11_47[12]
 cols_helper_11_47[13] cols_helper_11_47[14] cols_helper_11_47[15]
 cols_helper_11_47[16] cols_helper_9_51[0] cols_helper_9_51[1]
 cols_helper_9_51[2] cols_helper_9_51[3] cols_helper_9_51[4]
 cols_helper_9_51[5] cols_helper_9_51[6] cols_helper_9_51[7]
 cols_helper_9_51[8] cols_helper_9_51[9] cols_helper_9_51[10]
 cols_helper_9_51[11] cols_helper_9_51[12] cols_helper_9_51[13]
 cols_helper_9_51[14] cols_helper_9_51[15] cols_helper_9_51[16]
 cols_helper_10_8[0] cols_helper_10_8[1] cols_helper_10_8[2]
 cols_helper_10_8[3] cols_helper_10_8[4] cols_helper_10_8[5]
 cols_helper_10_8[6] cols_helper_10_8[7] cols_helper_10_8[8]
 cols_helper_10_8[9] cols_helper_10_8[10] cols_helper_10_8[11]
 cols_helper_10_8[12] cols_helper_10_8[13] cols_helper_10_8[14]
 cols_helper_10_8[15] cols_helper_10_8[16] cols_helper_11_5[0]
 cols_helper_11_5[1] cols_helper_11_5[2] cols_helper_11_5[3]
 cols_helper_11_5[4] cols_helper_11_5[5] cols_helper_11_5[6]
 cols_helper_11_5[7] cols_helper_11_5[8] cols_helper_11_5[9]
 cols_helper_11_5[10] cols_helper_11_5[11] cols_helper_11_5[12]
 cols_helper_11_5[13] cols_helper_11_5[14] cols_helper_11_5[15]
 cols_helper_11_5[16] cols_helper_9_21[0] cols_helper_9_21[1]
 cols_helper_9_21[2] cols_helper_9_21[3] cols_helper_9_21[4]
 cols_helper_9_21[5] cols_helper_9_21[6] cols_helper_9_21[7]
 cols_helper_9_21[8] cols_helper_9_21[9] cols_helper_9_21[10]
 cols_helper_9_21[11] cols_helper_9_21[12] cols_helper_9_21[13]
 cols_helper_9_21[14] cols_helper_9_21[15] cols_helper_9_21[16]
 cols_helper_10_18[0] cols_helper_10_18[1] cols_helper_10_18[2]
 cols_helper_10_18[3] cols_helper_10_18[4] cols_helper_10_18[5]
 cols_helper_10_18[6] cols_helper_10_18[7] cols_helper_10_18[8]
 cols_helper_10_18[9] cols_helper_10_18[10] cols_helper_10_18[11]
 cols_helper_10_18[12] cols_helper_10_18[13] cols_helper_10_18[14]
 cols_helper_10_18[15] cols_helper_10_18[16] cols_helper_9_72[0]
 cols_helper_9_72[1] cols_helper_9_72[2] cols_helper_9_72[3]
 cols_helper_9_72[4] cols_helper_9_72[5] cols_helper_9_72[6]
 cols_helper_9_72[7] cols_helper_9_72[8] cols_helper_9_72[9]
 cols_helper_9_72[10] cols_helper_9_72[11] cols_helper_9_72[12]
 cols_helper_9_72[13] cols_helper_9_72[14] cols_helper_9_72[15]
 cols_helper_9_72[16] cols_helper_10_70[0] cols_helper_10_70[1]
 cols_helper_10_70[2] cols_helper_10_70[3] cols_helper_10_70[4]
 cols_helper_10_70[5] cols_helper_10_70[6] cols_helper_10_70[7]
 cols_helper_10_70[8] cols_helper_10_70[9] cols_helper_10_70[10]
 cols_helper_10_70[11] cols_helper_10_70[12] cols_helper_10_70[13]
 cols_helper_10_70[14] cols_helper_10_70[15] cols_helper_10_70[16]
 cols_helper_10_41[0] cols_helper_10_41[1] cols_helper_10_41[2]
 cols_helper_10_41[3] cols_helper_10_41[4] cols_helper_10_41[5]
 cols_helper_10_41[6] cols_helper_10_41[7] cols_helper_10_41[8]
 cols_helper_10_41[9] cols_helper_10_41[10] cols_helper_10_41[11]
 cols_helper_10_41[12] cols_helper_10_41[13] cols_helper_10_41[14]
 cols_helper_10_41[15] cols_helper_10_41[16] cols_helper_8_57[0]
 cols_helper_8_57[1] cols_helper_8_57[2] cols_helper_8_57[3]
 cols_helper_8_57[4] cols_helper_8_57[5] cols_helper_8_57[6]
 cols_helper_8_57[7] cols_helper_8_57[8] cols_helper_8_57[9]
 cols_helper_8_57[10] cols_helper_8_57[11] cols_helper_8_57[12]
 cols_helper_8_57[13] cols_helper_8_57[14] cols_helper_8_57[15]
 cols_helper_8_57[16] cols_helper_10_51[0] cols_helper_10_51[1]
 cols_helper_10_51[2] cols_helper_10_51[3] cols_helper_10_51[4]
 cols_helper_10_51[5] cols_helper_10_51[6] cols_helper_10_51[7]
 cols_helper_10_51[8] cols_helper_10_51[9] cols_helper_10_51[10]
 cols_helper_10_51[11] cols_helper_10_51[12] cols_helper_10_51[13]
 cols_helper_10_51[14] cols_helper_10_51[15] cols_helper_10_51[16]
 cols_helper_11_58[0] cols_helper_11_58[1] cols_helper_11_58[2]
 cols_helper_11_58[3] cols_helper_11_58[4] cols_helper_11_58[5]
 cols_helper_11_58[6] cols_helper_11_58[7] cols_helper_11_58[8]
 cols_helper_11_58[9] cols_helper_11_58[10] cols_helper_11_58[11]
 cols_helper_11_58[12] cols_helper_11_58[13] cols_helper_11_58[14]
 cols_helper_11_58[15] cols_helper_11_58[16] cols_helper_9_30[0]
 cols_helper_9_30[1] cols_helper_9_30[2] cols_helper_9_30[3]
 cols_helper_9_30[4] cols_helper_9_30[5] cols_helper_9_30[6]
 cols_helper_9_30[7] cols_helper_9_30[8] cols_helper_9_30[9]
 cols_helper_9_30[10] cols_helper_9_30[11] cols_helper_9_30[12]
 cols_helper_9_30[13] cols_helper_9_30[14] cols_helper_9_30[15]
 cols_helper_9_30[16] cols_helper_10_29[0] cols_helper_10_29[1]
 cols_helper_10_29[2] cols_helper_10_29[3] cols_helper_10_29[4]
 cols_helper_10_29[5] cols_helper_10_29[6] cols_helper_10_29[7]
 cols_helper_10_29[8] cols_helper_10_29[9] cols_helper_10_29[10]
 cols_helper_10_29[11] cols_helper_10_29[12] cols_helper_10_29[13]
 cols_helper_10_29[14] cols_helper_10_29[15] cols_helper_10_29[16]
 cols_helper_10_65[0] cols_helper_10_65[1] cols_helper_10_65[2]
 cols_helper_10_65[3] cols_helper_10_65[4] cols_helper_10_65[5]
 cols_helper_10_65[6] cols_helper_10_65[7] cols_helper_10_65[8]
 cols_helper_10_65[9] cols_helper_10_65[10] cols_helper_10_65[11]
 cols_helper_10_65[12] cols_helper_10_65[13] cols_helper_10_65[14]
 cols_helper_10_65[15] cols_helper_10_65[16] cols_helper_8_32[0]
 cols_helper_8_32[1] cols_helper_8_32[2] cols_helper_8_32[3]
 cols_helper_8_32[4] cols_helper_8_32[5] cols_helper_8_32[6]
 cols_helper_8_32[7] cols_helper_8_32[8] cols_helper_8_32[9]
 cols_helper_8_32[10] cols_helper_8_32[11] cols_helper_8_32[12]
 cols_helper_8_32[13] cols_helper_8_32[14] cols_helper_8_32[15]
 cols_helper_8_32[16] cols_helper_9_61[0] cols_helper_9_61[1]
 cols_helper_9_61[2] cols_helper_9_61[3] cols_helper_9_61[4]
 cols_helper_9_61[5] cols_helper_9_61[6] cols_helper_9_61[7]
 cols_helper_9_61[8] cols_helper_9_61[9] cols_helper_9_61[10]
 cols_helper_9_61[11] cols_helper_9_61[12] cols_helper_9_61[13]
 cols_helper_9_61[14] cols_helper_9_61[15] cols_helper_9_61[16]
 cols_helper_10_58[0] cols_helper_10_58[1] cols_helper_10_58[2]
 cols_helper_10_58[3] cols_helper_10_58[4] cols_helper_10_58[5]
 cols_helper_10_58[6] cols_helper_10_58[7] cols_helper_10_58[8]
 cols_helper_10_58[9] cols_helper_10_58[10] cols_helper_10_58[11]
 cols_helper_10_58[12] cols_helper_10_58[13] cols_helper_10_58[14]
 cols_helper_10_58[15] cols_helper_10_58[16] cols_helper_11_51[0]
 cols_helper_11_51[1] cols_helper_11_51[2] cols_helper_11_51[3]
 cols_helper_11_51[4] cols_helper_11_51[5] cols_helper_11_51[6]
 cols_helper_11_51[7] cols_helper_11_51[8] cols_helper_11_51[9]
 cols_helper_11_51[10] cols_helper_11_51[11] cols_helper_11_51[12]
 cols_helper_11_51[13] cols_helper_11_51[14] cols_helper_11_51[15]
 cols_helper_11_51[16] cols_helper_9_7[0] cols_helper_9_7[1]
 cols_helper_9_7[2] cols_helper_9_7[3] cols_helper_9_7[4]
 cols_helper_9_7[5] cols_helper_9_7[6] cols_helper_9_7[7]
 cols_helper_9_7[8] cols_helper_9_7[9] cols_helper_9_7[10]
 cols_helper_9_7[11] cols_helper_9_7[12] cols_helper_9_7[13]
 cols_helper_9_7[14] cols_helper_9_7[15] cols_helper_9_7[16]
 cols_helper_10_72[0] cols_helper_10_72[1] cols_helper_10_72[2]
 cols_helper_10_72[3] cols_helper_10_72[4] cols_helper_10_72[5]
 cols_helper_10_72[6] cols_helper_10_72[7] cols_helper_10_72[8]
 cols_helper_10_72[9] cols_helper_10_72[10] cols_helper_10_72[11]
 cols_helper_10_72[12] cols_helper_10_72[13] cols_helper_10_72[14]
 cols_helper_10_72[15] cols_helper_10_72[16] cols_helper_8_39[0]
 cols_helper_8_39[1] cols_helper_8_39[2] cols_helper_8_39[3]
 cols_helper_8_39[4] cols_helper_8_39[5] cols_helper_8_39[6]
 cols_helper_8_39[7] cols_helper_8_39[8] cols_helper_8_39[9]
 cols_helper_8_39[10] cols_helper_8_39[11] cols_helper_8_39[12]
 cols_helper_8_39[13] cols_helper_8_39[14] cols_helper_8_39[15]
 cols_helper_8_39[16] cols_helper_9_38[0] cols_helper_9_38[1]
 cols_helper_9_38[2] cols_helper_9_38[3] cols_helper_9_38[4]
 cols_helper_9_38[5] cols_helper_9_38[6] cols_helper_9_38[7]
 cols_helper_9_38[8] cols_helper_9_38[9] cols_helper_9_38[10]
 cols_helper_9_38[11] cols_helper_9_38[12] cols_helper_9_38[13]
 cols_helper_9_38[14] cols_helper_9_38[15] cols_helper_9_38[16]
 cols_helper_10_37[0] cols_helper_10_37[1] cols_helper_10_37[2]
 cols_helper_10_37[3] cols_helper_10_37[4] cols_helper_10_37[5]
 cols_helper_10_37[6] cols_helper_10_37[7] cols_helper_10_37[8]
 cols_helper_10_37[9] cols_helper_10_37[10] cols_helper_10_37[11]
 cols_helper_10_37[12] cols_helper_10_37[13] cols_helper_10_37[14]
 cols_helper_10_37[15] cols_helper_10_37[16] cols_helper_11_40[0]
 cols_helper_11_40[1] cols_helper_11_40[2] cols_helper_11_40[3]
 cols_helper_11_40[4] cols_helper_11_40[5] cols_helper_11_40[6]
 cols_helper_11_40[7] cols_helper_11_40[8] cols_helper_11_40[9]
 cols_helper_11_40[10] cols_helper_11_40[11] cols_helper_11_40[12]
 cols_helper_11_40[13] cols_helper_11_40[14] cols_helper_11_40[15]
 cols_helper_11_40[16] cols_helper_10_15[0] cols_helper_10_15[1]
 cols_helper_10_15[2] cols_helper_10_15[3] cols_helper_10_15[4]
 cols_helper_10_15[5] cols_helper_10_15[6] cols_helper_10_15[7]
 cols_helper_10_15[8] cols_helper_10_15[9] cols_helper_10_15[10]
 cols_helper_10_15[11] cols_helper_10_15[12] cols_helper_10_15[13]
 cols_helper_10_15[14] cols_helper_10_15[15] cols_helper_10_15[16]
 cols_helper_8_19[0] cols_helper_8_19[1] cols_helper_8_19[2]
 cols_helper_8_19[3] cols_helper_8_19[4] cols_helper_8_19[5]
 cols_helper_8_19[6] cols_helper_8_19[7] cols_helper_8_19[8]
 cols_helper_8_19[9] cols_helper_8_19[10] cols_helper_8_19[11]
 cols_helper_8_19[12] cols_helper_8_19[13] cols_helper_8_19[14]
 cols_helper_8_19[15] cols_helper_8_19[16] cols_helper_11_14[0]
 cols_helper_11_14[1] cols_helper_11_14[2] cols_helper_11_14[3]
 cols_helper_11_14[4] cols_helper_11_14[5] cols_helper_11_14[6]
 cols_helper_11_14[7] cols_helper_11_14[8] cols_helper_11_14[9]
 cols_helper_11_14[10] cols_helper_11_14[11] cols_helper_11_14[12]
 cols_helper_11_14[13] cols_helper_11_14[14] cols_helper_11_14[15]
 cols_helper_11_14[16] cols_helper_9_18[0] cols_helper_9_18[1]
 cols_helper_9_18[2] cols_helper_9_18[3] cols_helper_9_18[4]
 cols_helper_9_18[5] cols_helper_9_18[6] cols_helper_9_18[7]
 cols_helper_9_18[8] cols_helper_9_18[9] cols_helper_9_18[10]
 cols_helper_9_18[11] cols_helper_9_18[12] cols_helper_9_18[13]
 cols_helper_9_18[14] cols_helper_9_18[15] cols_helper_9_18[16]
 cols_helper_9_69[0] cols_helper_9_69[1] cols_helper_9_69[2]
 cols_helper_9_69[3] cols_helper_9_69[4] cols_helper_9_69[5]
 cols_helper_9_69[6] cols_helper_9_69[7] cols_helper_9_69[8]
 cols_helper_9_69[9] cols_helper_9_69[10] cols_helper_9_69[11]
 cols_helper_9_69[12] cols_helper_9_69[13] cols_helper_9_69[14]
 cols_helper_9_69[15] cols_helper_9_69[16] cols_helper_10_44[0]
 cols_helper_10_44[1] cols_helper_10_44[2] cols_helper_10_44[3]
 cols_helper_10_44[4] cols_helper_10_44[5] cols_helper_10_44[6]
 cols_helper_10_44[7] cols_helper_10_44[8] cols_helper_10_44[9]
 cols_helper_10_44[10] cols_helper_10_44[11] cols_helper_10_44[12]
 cols_helper_10_44[13] cols_helper_10_44[14] cols_helper_10_44[15]
 cols_helper_10_44[16] cols_helper_11_33[0] cols_helper_11_33[1]
 cols_helper_11_33[2] cols_helper_11_33[3] cols_helper_11_33[4]
 cols_helper_11_33[5] cols_helper_11_33[6] cols_helper_11_33[7]
 cols_helper_11_33[8] cols_helper_11_33[9] cols_helper_11_33[10]
 cols_helper_11_33[11] cols_helper_11_33[12] cols_helper_11_33[13]
 cols_helper_11_33[14] cols_helper_11_33[15] cols_helper_11_33[16]
 cols_helper_10_54[0] cols_helper_10_54[1] cols_helper_10_54[2]
 cols_helper_10_54[3] cols_helper_10_54[4] cols_helper_10_54[5]
 cols_helper_10_54[6] cols_helper_10_54[7] cols_helper_10_54[8]
 cols_helper_10_54[9] cols_helper_10_54[10] cols_helper_10_54[11]
 cols_helper_10_54[12] cols_helper_10_54[13] cols_helper_10_54[14]
 cols_helper_10_54[15] cols_helper_10_54[16] cols_helper_8_26[0]
 cols_helper_8_26[1] cols_helper_8_26[2] cols_helper_8_26[3]
 cols_helper_8_26[4] cols_helper_8_26[5] cols_helper_8_26[6]
 cols_helper_8_26[7] cols_helper_8_26[8] cols_helper_8_26[9]
 cols_helper_8_26[10] cols_helper_8_26[11] cols_helper_8_26[12]
 cols_helper_8_26[13] cols_helper_8_26[14] cols_helper_8_26[15]
 cols_helper_8_26[16] cols_helper_11_29[0] cols_helper_11_29[1]
 cols_helper_11_29[2] cols_helper_11_29[3] cols_helper_11_29[4]
 cols_helper_11_29[5] cols_helper_11_29[6] cols_helper_11_29[7]
 cols_helper_11_29[8] cols_helper_11_29[9] cols_helper_11_29[10]
 cols_helper_11_29[11] cols_helper_11_29[12] cols_helper_11_29[13]
 cols_helper_11_29[14] cols_helper_11_29[15] cols_helper_11_29[16]
 cols_helper_8_70[0] cols_helper_8_70[1] cols_helper_8_70[2]
 cols_helper_8_70[3] cols_helper_8_70[4] cols_helper_8_70[5]
 cols_helper_8_70[6] cols_helper_8_70[7] cols_helper_8_70[8]
 cols_helper_8_70[9] cols_helper_8_70[10] cols_helper_8_70[11]
 cols_helper_8_70[12] cols_helper_8_70[13] cols_helper_8_70[14]
 cols_helper_8_70[15] cols_helper_8_70[16] cols_helper_11_64[0]
 cols_helper_11_64[1] cols_helper_11_64[2] cols_helper_11_64[3]
 cols_helper_11_64[4] cols_helper_11_64[5] cols_helper_11_64[6]
 cols_helper_11_64[7] cols_helper_11_64[8] cols_helper_11_64[9]
 cols_helper_11_64[10] cols_helper_11_64[11] cols_helper_11_64[12]
 cols_helper_11_64[13] cols_helper_11_64[14] cols_helper_11_64[15]
 cols_helper_11_64[16] cols_helper_8_59[0] cols_helper_8_59[1]
 cols_helper_8_59[2] cols_helper_8_59[3] cols_helper_8_59[4]
 cols_helper_8_59[5] cols_helper_8_59[6] cols_helper_8_59[7]
 cols_helper_8_59[8] cols_helper_8_59[9] cols_helper_8_59[10]
 cols_helper_8_59[11] cols_helper_8_59[12] cols_helper_8_59[13]
 cols_helper_8_59[14] cols_helper_8_59[15] cols_helper_8_59[16]
 cols_helper_11_38[0] cols_helper_11_38[1] cols_helper_11_38[2]
 cols_helper_11_38[3] cols_helper_11_38[4] cols_helper_11_38[5]
 cols_helper_11_38[6] cols_helper_11_38[7] cols_helper_11_38[8]
 cols_helper_11_38[9] cols_helper_11_38[10] cols_helper_11_38[11]
 cols_helper_11_38[12] cols_helper_11_38[13] cols_helper_11_38[14]
 cols_helper_11_38[15] cols_helper_11_38[16] cols_helper_9_58[0]
 cols_helper_9_58[1] cols_helper_9_58[2] cols_helper_9_58[3]
 cols_helper_9_58[4] cols_helper_9_58[5] cols_helper_9_58[6]
 cols_helper_9_58[7] cols_helper_9_58[8] cols_helper_9_58[9]
 cols_helper_9_58[10] cols_helper_9_58[11] cols_helper_9_58[12]
 cols_helper_9_58[13] cols_helper_9_58[14] cols_helper_9_58[15]
 cols_helper_9_58[16] cols_helper_10_49[0] cols_helper_10_49[1]
 cols_helper_10_49[2] cols_helper_10_49[3] cols_helper_10_49[4]
 cols_helper_10_49[5] cols_helper_10_49[6] cols_helper_10_49[7]
 cols_helper_10_49[8] cols_helper_10_49[9] cols_helper_10_49[10]
 cols_helper_10_49[11] cols_helper_10_49[12] cols_helper_10_49[13]
 cols_helper_10_49[14] cols_helper_10_49[15] cols_helper_10_49[16]
 cols_helper_11_60[0] cols_helper_11_60[1] cols_helper_11_60[2]
 cols_helper_11_60[3] cols_helper_11_60[4] cols_helper_11_60[5]
 cols_helper_11_60[6] cols_helper_11_60[7] cols_helper_11_60[8]
 cols_helper_11_60[9] cols_helper_11_60[10] cols_helper_11_60[11]
 cols_helper_11_60[12] cols_helper_11_60[13] cols_helper_11_60[14]
 cols_helper_11_60[15] cols_helper_11_60[16] cols_helper_9_28[0]
 cols_helper_9_28[1] cols_helper_9_28[2] cols_helper_9_28[3]
 cols_helper_9_28[4] cols_helper_9_28[5] cols_helper_9_28[6]
 cols_helper_9_28[7] cols_helper_9_28[8] cols_helper_9_28[9]
 cols_helper_9_28[10] cols_helper_9_28[11] cols_helper_9_28[12]
 cols_helper_9_28[13] cols_helper_9_28[14] cols_helper_9_28[15]
 cols_helper_9_28[16] cols_helper_10_27[0] cols_helper_10_27[1]
 cols_helper_10_27[2] cols_helper_10_27[3] cols_helper_10_27[4]
 cols_helper_10_27[5] cols_helper_10_27[6] cols_helper_10_27[7]
 cols_helper_10_27[8] cols_helper_10_27[9] cols_helper_10_27[10]
 cols_helper_10_27[11] cols_helper_10_27[12] cols_helper_10_27[13]
 cols_helper_10_27[14] cols_helper_10_27[15] cols_helper_10_27[16]
 cols_helper_11_18[0] cols_helper_11_18[1] cols_helper_11_18[2]
 cols_helper_11_18[3] cols_helper_11_18[4] cols_helper_11_18[5]
 cols_helper_11_18[6] cols_helper_11_18[7] cols_helper_11_18[8]
 cols_helper_11_18[9] cols_helper_11_18[10] cols_helper_11_18[11]
 cols_helper_11_18[12] cols_helper_11_18[13] cols_helper_11_18[14]
 cols_helper_11_18[15] cols_helper_11_18[16] cols_helper_8_34[0]
 cols_helper_8_34[1] cols_helper_8_34[2] cols_helper_8_34[3]
 cols_helper_8_34[4] cols_helper_8_34[5] cols_helper_8_34[6]
 cols_helper_8_34[7] cols_helper_8_34[8] cols_helper_8_34[9]
 cols_helper_8_34[10] cols_helper_8_34[11] cols_helper_8_34[12]
 cols_helper_8_34[13] cols_helper_8_34[14] cols_helper_8_34[15]
 cols_helper_8_34[16] cols_helper_9_35[0] cols_helper_9_35[1]
 cols_helper_9_35[2] cols_helper_9_35[3] cols_helper_9_35[4]
 cols_helper_9_35[5] cols_helper_9_35[6] cols_helper_9_35[7]
 cols_helper_9_35[8] cols_helper_9_35[9] cols_helper_9_35[10]
 cols_helper_9_35[11] cols_helper_9_35[12] cols_helper_9_35[13]
 cols_helper_9_35[14] cols_helper_9_35[15] cols_helper_9_35[16]
 cols_helper_10_56[0] cols_helper_10_56[1] cols_helper_10_56[2]
 cols_helper_10_56[3] cols_helper_10_56[4] cols_helper_10_56[5]
 cols_helper_10_56[6] cols_helper_10_56[7] cols_helper_10_56[8]
 cols_helper_10_56[9] cols_helper_10_56[10] cols_helper_10_56[11]
 cols_helper_10_56[12] cols_helper_10_56[13] cols_helper_10_56[14]
 cols_helper_10_56[15] cols_helper_10_56[16] cols_helper_11_53[0]
 cols_helper_11_53[1] cols_helper_11_53[2] cols_helper_11_53[3]
 cols_helper_11_53[4] cols_helper_11_53[5] cols_helper_11_53[6]
 cols_helper_11_53[7] cols_helper_11_53[8] cols_helper_11_53[9]
 cols_helper_11_53[10] cols_helper_11_53[11] cols_helper_11_53[12]
 cols_helper_11_53[13] cols_helper_11_53[14] cols_helper_11_53[15]
 cols_helper_11_53[16] cols_helper_9_5[0] cols_helper_9_5[1]
 cols_helper_9_5[2] cols_helper_9_5[3] cols_helper_9_5[4]
 cols_helper_9_5[5] cols_helper_9_5[6] cols_helper_9_5[7]
 cols_helper_9_5[8] cols_helper_9_5[9] cols_helper_9_5[10]
 cols_helper_9_5[11] cols_helper_9_5[12] cols_helper_9_5[13]
 cols_helper_9_5[14] cols_helper_9_5[15] cols_helper_9_5[16]
 cols_helper_10_2[0] cols_helper_10_2[1] cols_helper_10_2[2]
 cols_helper_10_2[3] cols_helper_10_2[4] cols_helper_10_2[5]
 cols_helper_10_2[6] cols_helper_10_2[7] cols_helper_10_2[8]
 cols_helper_10_2[9] cols_helper_10_2[10] cols_helper_10_2[11]
 cols_helper_10_2[12] cols_helper_10_2[13] cols_helper_10_2[14]
 cols_helper_10_2[15] cols_helper_10_2[16] cols_helper_11_11[0]
 cols_helper_11_11[1] cols_helper_11_11[2] cols_helper_11_11[3]
 cols_helper_11_11[4] cols_helper_11_11[5] cols_helper_11_11[6]
 cols_helper_11_11[7] cols_helper_11_11[8] cols_helper_11_11[9]
 cols_helper_11_11[10] cols_helper_11_11[11] cols_helper_11_11[12]
 cols_helper_11_11[13] cols_helper_11_11[14] cols_helper_11_11[15]
 cols_helper_11_11[16] cols_helper_9_66[0] cols_helper_9_66[1]
 cols_helper_9_66[2] cols_helper_9_66[3] cols_helper_9_66[4]
 cols_helper_9_66[5] cols_helper_9_66[6] cols_helper_9_66[7]
 cols_helper_9_66[8] cols_helper_9_66[9] cols_helper_9_66[10]
 cols_helper_9_66[11] cols_helper_9_66[12] cols_helper_9_66[13]
 cols_helper_9_66[14] cols_helper_9_66[15] cols_helper_9_66[16]
 cols_helper_8_41[0] cols_helper_8_41[1] cols_helper_8_41[2]
 cols_helper_8_41[3] cols_helper_8_41[4] cols_helper_8_41[5]
 cols_helper_8_41[6] cols_helper_8_41[7] cols_helper_8_41[8]
 cols_helper_8_41[9] cols_helper_8_41[10] cols_helper_8_41[11]
 cols_helper_8_41[12] cols_helper_8_41[13] cols_helper_8_41[14]
 cols_helper_8_41[15] cols_helper_8_41[16] cols_helper_10_35[0]
 cols_helper_10_35[1] cols_helper_10_35[2] cols_helper_10_35[3]
 cols_helper_10_35[4] cols_helper_10_35[5] cols_helper_10_35[6]
 cols_helper_10_35[7] cols_helper_10_35[8] cols_helper_10_35[9]
 cols_helper_10_35[10] cols_helper_10_35[11] cols_helper_10_35[12]
 cols_helper_10_35[13] cols_helper_10_35[14] cols_helper_10_35[15]
 cols_helper_10_35[16] cols_helper_8_15[0] cols_helper_8_15[1]
 cols_helper_8_15[2] cols_helper_8_15[3] cols_helper_8_15[4]
 cols_helper_8_15[5] cols_helper_8_15[6] cols_helper_8_15[7]
 cols_helper_8_15[8] cols_helper_8_15[9] cols_helper_8_15[10]
 cols_helper_8_15[11] cols_helper_8_15[12] cols_helper_8_15[13]
 cols_helper_8_15[14] cols_helper_8_15[15] cols_helper_8_15[16]
 cols_helper_9_14[0] cols_helper_9_14[1] cols_helper_9_14[2]
 cols_helper_9_14[3] cols_helper_9_14[4] cols_helper_9_14[5]
 cols_helper_9_14[6] cols_helper_9_14[7] cols_helper_9_14[8]
 cols_helper_9_14[9] cols_helper_9_14[10] cols_helper_9_14[11]
 cols_helper_9_14[12] cols_helper_9_14[13] cols_helper_9_14[14]
 cols_helper_9_14[15] cols_helper_9_14[16] cols_helper_10_13[0]
 cols_helper_10_13[1] cols_helper_10_13[2] cols_helper_10_13[3]
 cols_helper_10_13[4] cols_helper_10_13[5] cols_helper_10_13[6]
 cols_helper_10_13[7] cols_helper_10_13[8] cols_helper_10_13[9]
 cols_helper_10_13[10] cols_helper_10_13[11] cols_helper_10_13[12]
 cols_helper_10_13[13] cols_helper_10_13[14] cols_helper_10_13[15]
 cols_helper_10_13[16] cols_helper_8_21[0] cols_helper_8_21[1]
 cols_helper_8_21[2] cols_helper_8_21[3] cols_helper_8_21[4]
 cols_helper_8_21[5] cols_helper_8_21[6] cols_helper_8_21[7]
 cols_helper_8_21[8] cols_helper_8_21[9] cols_helper_8_21[10]
 cols_helper_8_21[11] cols_helper_8_21[12] cols_helper_8_21[13]
 cols_helper_8_21[14] cols_helper_8_21[15] cols_helper_8_21[16]
 cols_helper_11_0[0] cols_helper_11_0[1] cols_helper_11_0[2]
 cols_helper_11_0[3] cols_helper_11_0[4] cols_helper_11_0[5]
 cols_helper_11_0[6] cols_helper_11_0[7] cols_helper_11_0[8]
 cols_helper_11_0[9] cols_helper_11_0[10] cols_helper_11_0[11]
 cols_helper_11_0[12] cols_helper_11_0[13] cols_helper_11_0[14]
 cols_helper_11_0[15] cols_helper_11_0[16] cols_helper_10_23[0]
 cols_helper_10_23[1] cols_helper_10_23[2] cols_helper_10_23[3]
 cols_helper_10_23[4] cols_helper_10_23[5] cols_helper_10_23[6]
 cols_helper_10_23[7] cols_helper_10_23[8] cols_helper_10_23[9]
 cols_helper_10_23[10] cols_helper_10_23[11] cols_helper_10_23[12]
 cols_helper_10_23[13] cols_helper_10_23[14] cols_helper_10_23[15]
 cols_helper_10_23[16] cols_helper_8_65[0] cols_helper_8_65[1]
 cols_helper_8_65[2] cols_helper_8_65[3] cols_helper_8_65[4]
 cols_helper_8_65[5] cols_helper_8_65[6] cols_helper_8_65[7]
 cols_helper_8_65[8] cols_helper_8_65[9] cols_helper_8_65[10]
 cols_helper_8_65[11] cols_helper_8_65[12] cols_helper_8_65[13]
 cols_helper_8_65[14] cols_helper_8_65[15] cols_helper_8_65[16]
 cols_helper_9_45[0] cols_helper_9_45[1] cols_helper_9_45[2]
 cols_helper_9_45[3] cols_helper_9_45[4] cols_helper_9_45[5]
 cols_helper_9_45[6] cols_helper_9_45[7] cols_helper_9_45[8]
 cols_helper_9_45[9] cols_helper_9_45[10] cols_helper_9_45[11]
 cols_helper_9_45[12] cols_helper_9_45[13] cols_helper_9_45[14]
 cols_helper_9_45[15] cols_helper_9_45[16] cols_helper_8_54[0]
 cols_helper_8_54[1] cols_helper_8_54[2] cols_helper_8_54[3]
 cols_helper_8_54[4] cols_helper_8_54[5] cols_helper_8_54[6]
 cols_helper_8_54[7] cols_helper_8_54[8] cols_helper_8_54[9]
 cols_helper_8_54[10] cols_helper_8_54[11] cols_helper_8_54[12]
 cols_helper_8_54[13] cols_helper_8_54[14] cols_helper_8_54[15]
 cols_helper_8_54[16] cols_helper_11_35[0] cols_helper_11_35[1]
 cols_helper_11_35[2] cols_helper_11_35[3] cols_helper_11_35[4]
 cols_helper_11_35[5] cols_helper_11_35[6] cols_helper_11_35[7]
 cols_helper_11_35[8] cols_helper_11_35[9] cols_helper_11_35[10]
 cols_helper_11_35[11] cols_helper_11_35[12] cols_helper_11_35[13]
 cols_helper_11_35[14] cols_helper_11_35[15] cols_helper_11_35[16]
 cols_helper_9_55[0] cols_helper_9_55[1] cols_helper_9_55[2]
 cols_helper_9_55[3] cols_helper_9_55[4] cols_helper_9_55[5]
 cols_helper_9_55[6] cols_helper_9_55[7] cols_helper_9_55[8]
 cols_helper_9_55[9] cols_helper_9_55[10] cols_helper_9_55[11]
 cols_helper_9_55[12] cols_helper_9_55[13] cols_helper_9_55[14]
 cols_helper_9_55[15] cols_helper_9_55[16] cols_helper_10_52[0]
 cols_helper_10_52[1] cols_helper_10_52[2] cols_helper_10_52[3]
 cols_helper_10_52[4] cols_helper_10_52[5] cols_helper_10_52[6]
 cols_helper_10_52[7] cols_helper_10_52[8] cols_helper_10_52[9]
 cols_helper_10_52[10] cols_helper_10_52[11] cols_helper_10_52[12]
 cols_helper_10_52[13] cols_helper_10_52[14] cols_helper_10_52[15]
 cols_helper_10_52[16] cols_helper_8_28[0] cols_helper_8_28[1]
 cols_helper_8_28[2] cols_helper_8_28[3] cols_helper_8_28[4]
 cols_helper_8_28[5] cols_helper_8_28[6] cols_helper_8_28[7]
 cols_helper_8_28[8] cols_helper_8_28[9] cols_helper_8_28[10]
 cols_helper_8_28[11] cols_helper_8_28[12] cols_helper_8_28[13]
 cols_helper_8_28[14] cols_helper_8_28[15] cols_helper_8_28[16]
 cols_helper_11_57[0] cols_helper_11_57[1] cols_helper_11_57[2]
 cols_helper_11_57[3] cols_helper_11_57[4] cols_helper_11_57[5]
 cols_helper_11_57[6] cols_helper_11_57[7] cols_helper_11_57[8]
 cols_helper_11_57[9] cols_helper_11_57[10] cols_helper_11_57[11]
 cols_helper_11_57[12] cols_helper_11_57[13] cols_helper_11_57[14]
 cols_helper_11_57[15] cols_helper_11_57[16] cols_helper_9_25[0]
 cols_helper_9_25[1] cols_helper_9_25[2] cols_helper_9_25[3]
 cols_helper_9_25[4] cols_helper_9_25[5] cols_helper_9_25[6]
 cols_helper_9_25[7] cols_helper_9_25[8] cols_helper_9_25[9]
 cols_helper_9_25[10] cols_helper_9_25[11] cols_helper_9_25[12]
 cols_helper_9_25[13] cols_helper_9_25[14] cols_helper_9_25[15]
 cols_helper_9_25[16] cols_helper_10_30[0] cols_helper_10_30[1]
 cols_helper_10_30[2] cols_helper_10_30[3] cols_helper_10_30[4]
 cols_helper_10_30[5] cols_helper_10_30[6] cols_helper_10_30[7]
 cols_helper_10_30[8] cols_helper_10_30[9] cols_helper_10_30[10]
 cols_helper_10_30[11] cols_helper_10_30[12] cols_helper_10_30[13]
 cols_helper_10_30[14] cols_helper_10_30[15] cols_helper_10_30[16]
 cols_helper_11_31[0] cols_helper_11_31[1] cols_helper_11_31[2]
 cols_helper_11_31[3] cols_helper_11_31[4] cols_helper_11_31[5]
 cols_helper_11_31[6] cols_helper_11_31[7] cols_helper_11_31[8]
 cols_helper_11_31[9] cols_helper_11_31[10] cols_helper_11_31[11]
 cols_helper_11_31[12] cols_helper_11_31[13] cols_helper_11_31[14]
 cols_helper_11_31[15] cols_helper_11_31[16] cols_helper_8_72[0]
 cols_helper_8_72[1] cols_helper_8_72[2] cols_helper_8_72[3]
 cols_helper_8_72[4] cols_helper_8_72[5] cols_helper_8_72[6]
 cols_helper_8_72[7] cols_helper_8_72[8] cols_helper_8_72[9]
 cols_helper_8_72[10] cols_helper_8_72[11] cols_helper_8_72[12]
 cols_helper_8_72[13] cols_helper_8_72[14] cols_helper_8_72[15]
 cols_helper_8_72[16] cols_helper_10_66[0] cols_helper_10_66[1]
 cols_helper_10_66[2] cols_helper_10_66[3] cols_helper_10_66[4]
 cols_helper_10_66[5] cols_helper_10_66[6] cols_helper_10_66[7]
 cols_helper_10_66[8] cols_helper_10_66[9] cols_helper_10_66[10]
 cols_helper_10_66[11] cols_helper_10_66[12] cols_helper_10_66[13]
 cols_helper_10_66[14] cols_helper_10_66[15] cols_helper_10_66[16]
 cols_helper_11_66[0] cols_helper_11_66[1] cols_helper_11_66[2]
 cols_helper_11_66[3] cols_helper_11_66[4] cols_helper_11_66[5]
 cols_helper_11_66[6] cols_helper_11_66[7] cols_helper_11_66[8]
 cols_helper_11_66[9] cols_helper_11_66[10] cols_helper_11_66[11]
 cols_helper_11_66[12] cols_helper_11_66[13] cols_helper_11_66[14]
 cols_helper_11_66[15] cols_helper_11_66[16] cols_helper_8_61[0]
 cols_helper_8_61[1] cols_helper_8_61[2] cols_helper_8_61[3]
 cols_helper_8_61[4] cols_helper_8_61[5] cols_helper_8_61[6]
 cols_helper_8_61[7] cols_helper_8_61[8] cols_helper_8_61[9]
 cols_helper_8_61[10] cols_helper_8_61[11] cols_helper_8_61[12]
 cols_helper_8_61[13] cols_helper_8_61[14] cols_helper_8_61[15]
 cols_helper_8_61[16] cols_helper_10_63[0] cols_helper_10_63[1]
 cols_helper_10_63[2] cols_helper_10_63[3] cols_helper_10_63[4]
 cols_helper_10_63[5] cols_helper_10_63[6] cols_helper_10_63[7]
 cols_helper_10_63[8] cols_helper_10_63[9] cols_helper_10_63[10]
 cols_helper_10_63[11] cols_helper_10_63[12] cols_helper_10_63[13]
 cols_helper_10_63[14] cols_helper_10_63[15] cols_helper_10_63[16]
 cols_helper_8_3[0] cols_helper_8_3[1] cols_helper_8_3[2]
 cols_helper_8_3[3] cols_helper_8_3[4] cols_helper_8_3[5]
 cols_helper_8_3[6] cols_helper_8_3[7] cols_helper_8_3[8]
 cols_helper_8_3[9] cols_helper_8_3[10] cols_helper_8_3[11]
 cols_helper_8_3[12] cols_helper_8_3[13] cols_helper_8_3[14]
 cols_helper_8_3[15] cols_helper_8_3[16] cols_helper_11_62[0]
 cols_helper_11_62[1] cols_helper_11_62[2] cols_helper_11_62[3]
 cols_helper_11_62[4] cols_helper_11_62[5] cols_helper_11_62[6]
 cols_helper_11_62[7] cols_helper_11_62[8] cols_helper_11_62[9]
 cols_helper_11_62[10] cols_helper_11_62[11] cols_helper_11_62[12]
 cols_helper_11_62[13] cols_helper_11_62[14] cols_helper_11_62[15]
 cols_helper_11_62[16] cols_helper_9_2[0] cols_helper_9_2[1]
 cols_helper_9_2[2] cols_helper_9_2[3] cols_helper_9_2[4]
 cols_helper_9_2[5] cols_helper_9_2[6] cols_helper_9_2[7]
 cols_helper_9_2[8] cols_helper_9_2[9] cols_helper_9_2[10]
 cols_helper_9_2[11] cols_helper_9_2[12] cols_helper_9_2[13]
 cols_helper_9_2[14] cols_helper_9_2[15] cols_helper_9_2[16]
 cols_helper_10_25[0] cols_helper_10_25[1] cols_helper_10_25[2]
 cols_helper_10_25[3] cols_helper_10_25[4] cols_helper_10_25[5]
 cols_helper_10_25[6] cols_helper_10_25[7] cols_helper_10_25[8]
 cols_helper_10_25[9] cols_helper_10_25[10] cols_helper_10_25[11]
 cols_helper_10_25[12] cols_helper_10_25[13] cols_helper_10_25[14]
 cols_helper_10_25[15] cols_helper_10_25[16] cols_helper_8_36[0]
 cols_helper_8_36[1] cols_helper_8_36[2] cols_helper_8_36[3]
 cols_helper_8_36[4] cols_helper_8_36[5] cols_helper_8_36[6]
 cols_helper_8_36[7] cols_helper_8_36[8] cols_helper_8_36[9]
 cols_helper_8_36[10] cols_helper_8_36[11] cols_helper_8_36[12]
 cols_helper_8_36[13] cols_helper_8_36[14] cols_helper_8_36[15]
 cols_helper_8_36[16] cols_helper_10_38[0] cols_helper_10_38[1]
 cols_helper_10_38[2] cols_helper_10_38[3] cols_helper_10_38[4]
 cols_helper_10_38[5] cols_helper_10_38[6] cols_helper_10_38[7]
 cols_helper_10_38[8] cols_helper_10_38[9] cols_helper_10_38[10]
 cols_helper_10_38[11] cols_helper_10_38[12] cols_helper_10_38[13]
 cols_helper_10_38[14] cols_helper_10_38[15] cols_helper_10_38[16]
 cols_helper_10_0[0] cols_helper_10_0[1] cols_helper_10_0[2]
 cols_helper_10_0[3] cols_helper_10_0[4] cols_helper_10_0[5]
 cols_helper_10_0[6] cols_helper_10_0[7] cols_helper_10_0[8]
 cols_helper_10_0[9] cols_helper_10_0[10] cols_helper_10_0[11]
 cols_helper_10_0[12] cols_helper_10_0[13] cols_helper_10_0[14]
 cols_helper_10_0[15] cols_helper_10_0[16] cols_helper_8_16[0]
 cols_helper_8_16[1] cols_helper_8_16[2] cols_helper_8_16[3]
 cols_helper_8_16[4] cols_helper_8_16[5] cols_helper_8_16[6]
 cols_helper_8_16[7] cols_helper_8_16[8] cols_helper_8_16[9]
 cols_helper_8_16[10] cols_helper_8_16[11] cols_helper_8_16[12]
 cols_helper_8_16[13] cols_helper_8_16[14] cols_helper_8_16[15]
 cols_helper_8_16[16] cols_helper_11_13[0] cols_helper_11_13[1]
 cols_helper_11_13[2] cols_helper_11_13[3] cols_helper_11_13[4]
 cols_helper_11_13[5] cols_helper_11_13[6] cols_helper_11_13[7]
 cols_helper_11_13[8] cols_helper_11_13[9] cols_helper_11_13[10]
 cols_helper_11_13[11] cols_helper_11_13[12] cols_helper_11_13[13]
 cols_helper_11_13[14] cols_helper_11_13[15] cols_helper_11_13[16]
 cols_helper_9_64[0] cols_helper_9_64[1] cols_helper_9_64[2]
 cols_helper_9_64[3] cols_helper_9_64[4] cols_helper_9_64[5]
 cols_helper_9_64[6] cols_helper_9_64[7] cols_helper_9_64[8]
 cols_helper_9_64[9] cols_helper_9_64[10] cols_helper_9_64[11]
 cols_helper_9_64[12] cols_helper_9_64[13] cols_helper_9_64[14]
 cols_helper_9_64[15] cols_helper_9_64[16] cols_helper_8_43[0]
 cols_helper_8_43[1] cols_helper_8_43[2] cols_helper_8_43[3]
 cols_helper_8_43[4] cols_helper_8_43[5] cols_helper_8_43[6]
 cols_helper_8_43[7] cols_helper_8_43[8] cols_helper_8_43[9]
 cols_helper_8_43[10] cols_helper_8_43[11] cols_helper_8_43[12]
 cols_helper_8_43[13] cols_helper_8_43[14] cols_helper_8_43[15]
 cols_helper_8_43[16] cols_helper_9_42[0] cols_helper_9_42[1]
 cols_helper_9_42[2] cols_helper_9_42[3] cols_helper_9_42[4]
 cols_helper_9_42[5] cols_helper_9_42[6] cols_helper_9_42[7]
 cols_helper_9_42[8] cols_helper_9_42[9] cols_helper_9_42[10]
 cols_helper_9_42[11] cols_helper_9_42[12] cols_helper_9_42[13]
 cols_helper_9_42[14] cols_helper_9_42[15] cols_helper_9_42[16]
 cols_helper_10_33[0] cols_helper_10_33[1] cols_helper_10_33[2]
 cols_helper_10_33[3] cols_helper_10_33[4] cols_helper_10_33[5]
 cols_helper_10_33[6] cols_helper_10_33[7] cols_helper_10_33[8]
 cols_helper_10_33[9] cols_helper_10_33[10] cols_helper_10_33[11]
 cols_helper_10_33[12] cols_helper_10_33[13] cols_helper_10_33[14]
 cols_helper_10_33[15] cols_helper_10_33[16] cols_helper_8_49[0]
 cols_helper_8_49[1] cols_helper_8_49[2] cols_helper_8_49[3]
 cols_helper_8_49[4] cols_helper_8_49[5] cols_helper_8_49[6]
 cols_helper_8_49[7] cols_helper_8_49[8] cols_helper_8_49[9]
 cols_helper_8_49[10] cols_helper_8_49[11] cols_helper_8_49[12]
 cols_helper_8_49[13] cols_helper_8_49[14] cols_helper_8_49[15]
 cols_helper_8_49[16] cols_helper_11_44[0] cols_helper_11_44[1]
 cols_helper_11_44[2] cols_helper_11_44[3] cols_helper_11_44[4]
 cols_helper_11_44[5] cols_helper_11_44[6] cols_helper_11_44[7]
 cols_helper_11_44[8] cols_helper_11_44[9] cols_helper_11_44[10]
 cols_helper_11_44[11] cols_helper_11_44[12] cols_helper_11_44[13]
 cols_helper_11_44[14] cols_helper_11_44[15] cols_helper_11_44[16]
 cols_helper_10_11[0] cols_helper_10_11[1] cols_helper_10_11[2]
 cols_helper_10_11[3] cols_helper_10_11[4] cols_helper_10_11[5]
 cols_helper_10_11[6] cols_helper_10_11[7] cols_helper_10_11[8]
 cols_helper_10_11[9] cols_helper_10_11[10] cols_helper_10_11[11]
 cols_helper_10_11[12] cols_helper_10_11[13] cols_helper_10_11[14]
 cols_helper_10_11[15] cols_helper_10_11[16] cols_helper_8_23[0]
 cols_helper_8_23[1] cols_helper_8_23[2] cols_helper_8_23[3]
 cols_helper_8_23[4] cols_helper_8_23[5] cols_helper_8_23[6]
 cols_helper_8_23[7] cols_helper_8_23[8] cols_helper_8_23[9]
 cols_helper_8_23[10] cols_helper_8_23[11] cols_helper_8_23[12]
 cols_helper_8_23[13] cols_helper_8_23[14] cols_helper_8_23[15]
 cols_helper_8_23[16] cols_helper_11_2[0] cols_helper_11_2[1]
 cols_helper_11_2[2] cols_helper_11_2[3] cols_helper_11_2[4]
 cols_helper_11_2[5] cols_helper_11_2[6] cols_helper_11_2[7]
 cols_helper_11_2[8] cols_helper_11_2[9] cols_helper_11_2[10]
 cols_helper_11_2[11] cols_helper_11_2[12] cols_helper_11_2[13]
 cols_helper_11_2[14] cols_helper_11_2[15] cols_helper_11_2[16]
 cols_helper_10_21[0] cols_helper_10_21[1] cols_helper_10_21[2]
 cols_helper_10_21[3] cols_helper_10_21[4] cols_helper_10_21[5]
 cols_helper_10_21[6] cols_helper_10_21[7] cols_helper_10_21[8]
 cols_helper_10_21[9] cols_helper_10_21[10] cols_helper_10_21[11]
 cols_helper_10_21[12] cols_helper_10_21[13] cols_helper_10_21[14]
 cols_helper_10_21[15] cols_helper_10_21[16] cols_helper_8_67[0]
 cols_helper_8_67[1] cols_helper_8_67[2] cols_helper_8_67[3]
 cols_helper_8_67[4] cols_helper_8_67[5] cols_helper_8_67[6]
 cols_helper_8_67[7] cols_helper_8_67[8] cols_helper_8_67[9]
 cols_helper_8_67[10] cols_helper_8_67[11] cols_helper_8_67[12]
 cols_helper_8_67[13] cols_helper_8_67[14] cols_helper_8_67[15]
 cols_helper_8_67[16] cols_helper_9_73[0] cols_helper_9_73[1]
 cols_helper_9_73[2] cols_helper_9_73[3] cols_helper_9_73[4]
 cols_helper_9_73[5] cols_helper_9_73[6] cols_helper_9_73[7]
 cols_helper_9_73[8] cols_helper_9_73[9] cols_helper_9_73[10]
 cols_helper_9_73[11] cols_helper_9_73[12] cols_helper_9_73[13]
 cols_helper_9_73[14] cols_helper_9_73[15] cols_helper_9_73[16]
 cols_helper_10_40[0] cols_helper_10_40[1] cols_helper_10_40[2]
 cols_helper_10_40[3] cols_helper_10_40[4] cols_helper_10_40[5]
 cols_helper_10_40[6] cols_helper_10_40[7] cols_helper_10_40[8]
 cols_helper_10_40[9] cols_helper_10_40[10] cols_helper_10_40[11]
 cols_helper_10_40[12] cols_helper_10_40[13] cols_helper_10_40[14]
 cols_helper_10_40[15] cols_helper_10_40[16] cols_helper_8_56[0]
 cols_helper_8_56[1] cols_helper_8_56[2] cols_helper_8_56[3]
 cols_helper_8_56[4] cols_helper_8_56[5] cols_helper_8_56[6]
 cols_helper_8_56[7] cols_helper_8_56[8] cols_helper_8_56[9]
 cols_helper_8_56[10] cols_helper_8_56[11] cols_helper_8_56[12]
 cols_helper_8_56[13] cols_helper_8_56[14] cols_helper_8_56[15]
 cols_helper_8_56[16] cols_helper_11_37[0] cols_helper_11_37[1]
 cols_helper_11_37[2] cols_helper_11_37[3] cols_helper_11_37[4]
 cols_helper_11_37[5] cols_helper_11_37[6] cols_helper_11_37[7]
 cols_helper_11_37[8] cols_helper_11_37[9] cols_helper_11_37[10]
 cols_helper_11_37[11] cols_helper_11_37[12] cols_helper_11_37[13]
 cols_helper_11_37[14] cols_helper_11_37[15] cols_helper_11_37[16]
 cols_helper_9_53[0] cols_helper_9_53[1] cols_helper_9_53[2]
 cols_helper_9_53[3] cols_helper_9_53[4] cols_helper_9_53[5]
 cols_helper_9_53[6] cols_helper_9_53[7] cols_helper_9_53[8]
 cols_helper_9_53[9] cols_helper_9_53[10] cols_helper_9_53[11]
 cols_helper_9_53[12] cols_helper_9_53[13] cols_helper_9_53[14]
 cols_helper_9_53[15] cols_helper_9_53[16] cols_helper_10_50[0]
 cols_helper_10_50[1] cols_helper_10_50[2] cols_helper_10_50[3]
 cols_helper_10_50[4] cols_helper_10_50[5] cols_helper_10_50[6]
 cols_helper_10_50[7] cols_helper_10_50[8] cols_helper_10_50[9]
 cols_helper_10_50[10] cols_helper_10_50[11] cols_helper_10_50[12]
 cols_helper_10_50[13] cols_helper_10_50[14] cols_helper_10_50[15]
 cols_helper_10_50[16] cols_helper_8_30[0] cols_helper_8_30[1]
 cols_helper_8_30[2] cols_helper_8_30[3] cols_helper_8_30[4]
 cols_helper_8_30[5] cols_helper_8_30[6] cols_helper_8_30[7]
 cols_helper_8_30[8] cols_helper_8_30[9] cols_helper_8_30[10]
 cols_helper_8_30[11] cols_helper_8_30[12] cols_helper_8_30[13]
 cols_helper_8_30[14] cols_helper_8_30[15] cols_helper_8_30[16]
 cols_helper_11_59[0] cols_helper_11_59[1] cols_helper_11_59[2]
 cols_helper_11_59[3] cols_helper_11_59[4] cols_helper_11_59[5]
 cols_helper_11_59[6] cols_helper_11_59[7] cols_helper_11_59[8]
 cols_helper_11_59[9] cols_helper_11_59[10] cols_helper_11_59[11]
 cols_helper_11_59[12] cols_helper_11_59[13] cols_helper_11_59[14]
 cols_helper_11_59[15] cols_helper_11_59[16] cols_helper_10_28[0]
 cols_helper_10_28[1] cols_helper_10_28[2] cols_helper_10_28[3]
 cols_helper_10_28[4] cols_helper_10_28[5] cols_helper_10_28[6]
 cols_helper_10_28[7] cols_helper_10_28[8] cols_helper_10_28[9]
 cols_helper_10_28[10] cols_helper_10_28[11] cols_helper_10_28[12]
 cols_helper_10_28[13] cols_helper_10_28[14] cols_helper_10_28[15]
 cols_helper_10_28[16] cols_helper_11_17[0] cols_helper_11_17[1]
 cols_helper_11_17[2] cols_helper_11_17[3] cols_helper_11_17[4]
 cols_helper_11_17[5] cols_helper_11_17[6] cols_helper_11_17[7]
 cols_helper_11_17[8] cols_helper_11_17[9] cols_helper_11_17[10]
 cols_helper_11_17[11] cols_helper_11_17[12] cols_helper_11_17[13]
 cols_helper_11_17[14] cols_helper_11_17[15] cols_helper_11_17[16]
 cols_helper_8_74[0] cols_helper_8_74[1] cols_helper_8_74[2]
 cols_helper_8_74[3] cols_helper_8_74[4] cols_helper_8_74[5]
 cols_helper_8_74[6] cols_helper_8_74[7] cols_helper_8_74[8]
 cols_helper_8_74[9] cols_helper_8_74[10] cols_helper_8_74[11]
 cols_helper_8_74[12] cols_helper_8_74[13] cols_helper_8_74[14]
 cols_helper_8_74[15] cols_helper_8_74[16] cols_helper_10_64[0]
 cols_helper_10_64[1] cols_helper_10_64[2] cols_helper_10_64[3]
 cols_helper_10_64[4] cols_helper_10_64[5] cols_helper_10_64[6]
 cols_helper_10_64[7] cols_helper_10_64[8] cols_helper_10_64[9]
 cols_helper_10_64[10] cols_helper_10_64[11] cols_helper_10_64[12]
 cols_helper_10_64[13] cols_helper_10_64[14] cols_helper_10_64[15]
 cols_helper_10_64[16]
Generals
 cols_8_63 cols_9_62 cols_10_61 cols_8_5 cols_11_48 cols_9_0 cols_10_7
 cols_11_22 cols_9_39 cols_10_36 cols_8_12 cols_11_41 cols_10_14 cols_8_18
 cols_9_19 cols_11_72 cols_10_47 cols_8_51 cols_11_46 cols_9_50 cols_8_25
 cols_9_20 cols_10_19 cols_11_26 cols_8_69 cols_11_65 cols_11_39 cols_9_59
 cols_10_48 cols_8_0 cols_11_61 cols_9_29 cols_10_26 cols_11_19 cols_8_33
 cols_9_60 cols_10_59 cols_8_7 cols_11_50 cols_10_5 cols_11_8 cols_10_73
 cols_9_67 cols_8_40 cols_9_37 cols_8_14 cols_10_12 cols_11_1 cols_9_17
 cols_10_22 cols_8_64 cols_9_68 cols_8_47 cols_11_74 cols_9_46 cols_10_45
 cols_8_53 cols_11_32 cols_10_55 cols_8_27 cols_11_6 cols_10_17 cols_11_28
 cols_8_71 cols_10_69 cols_11_67 cols_8_60 cols_9_57 cols_10_62 cols_8_2
 cols_11_21 cols_8_35 cols_10_57 cols_8_9 cols_11_52 cols_9_4 cols_10_3
 cols_11_10 cols_9_65 cols_9_43 cols_10_32 cols_8_48 cols_11_45 cols_9_13
 cols_10_10 cols_8_22 cols_9_23 cols_11_25 cols_8_66 cols_9_74 cols_9_44
 cols_8_55 cols_9_54 cols_10_53 cols_8_29 cols_11_56 cols_9_24 cols_10_31
 cols_11_30 cols_8_73 cols_10_67 cols_11_69 cols_8_62 cols_10_60 cols_9_1
 cols_10_6 cols_11_23 cols_10_74 cols_8_37 cols_9_32 cols_10_39 cols_11_54
 cols_9_10 cols_10_1 cols_8_17 cols_11_12 cols_9_71 cols_11_73 cols_9_41
 cols_10_46 cols_8_50 cols_11_47 cols_9_51 cols_10_8 cols_11_5 cols_9_21
 cols_10_18 cols_9_72 cols_10_70 cols_10_41 cols_8_57 cols_10_51 cols_11_58
 cols_9_30 cols_10_29 cols_10_65 cols_8_32 cols_9_61 cols_10_58 cols_11_51
 cols_9_7 cols_10_72 cols_8_39 cols_9_38 cols_10_37 cols_11_40 cols_10_15
 cols_8_19 cols_11_14 cols_9_18 cols_9_69 cols_10_44 cols_11_33 cols_10_54
 cols_8_26 cols_11_29 cols_8_70 cols_11_64 cols_8_59 cols_11_38 cols_9_58
 cols_10_49 cols_11_60 cols_9_28 cols_10_27 cols_11_18 cols_8_34 cols_9_35
 cols_10_56 cols_11_53 cols_9_5 cols_10_2 cols_11_11 cols_9_66 cols_8_41
 cols_10_35 cols_8_15 cols_9_14 cols_10_13 cols_8_21 cols_11_0 cols_10_23
 cols_8_65 cols_9_45 cols_8_54 cols_11_35 cols_9_55 cols_10_52 cols_8_28
 cols_11_57 cols_9_25 cols_10_30 cols_11_31 cols_8_72 cols_10_66 cols_11_66
 cols_8_61 cols_10_63 cols_8_3 cols_11_62 cols_9_2 cols_10_25 cols_8_36
 cols_10_38 cols_10_0 cols_8_16 cols_11_13 cols_9_64 cols_8_43 cols_9_42
 cols_10_33 cols_8_49 cols_11_44 cols_10_11 cols_8_23 cols_11_2 cols_10_21
 cols_8_67 cols_9_73 cols_10_40 cols_8_56 cols_11_37 cols_9_53 cols_10_50
 cols_8_30 cols_11_59 cols_10_28 cols_11_17 cols_8_74 cols_10_64
 cores_sketch_CPU2 cores_dpdk_CPU2 cores_sketch_CPU3 cores_dpdk_CPU3
 cores_sketch_CPU4 cores_dpdk_CPU4 cores_sketch_CPU7 cores_dpdk_CPU7
 cores_sketch_CPU17 cores_dpdk_CPU17 cores_sketch_CPU18 cores_dpdk_CPU18
 cores_sketch_CPU24 cores_dpdk_CPU24 cores_sketch_CPU26 cores_dpdk_CPU26
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_tor_P41 , mem_tot_tor_P43 ,
   mem_tot_l1_P41 , mem_tot_l2_P4 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU2 , normalized_mem_tot_CPU3 ,
   normalized_mem_tot_CPU4 , normalized_mem_tot_CPU7 ,
   normalized_mem_tot_CPU17 , normalized_mem_tot_CPU18 ,
   normalized_mem_tot_CPU24 , normalized_mem_tot_CPU26 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_tor_P41 , rows_tot_tor_P43 ,
   rows_tot_l1_P41 , rows_tot_l2_P4 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU2 ,
   normalized_rows_tot_CPU3 , normalized_rows_tot_CPU4 ,
   normalized_rows_tot_CPU7 , normalized_rows_tot_CPU17 ,
   normalized_rows_tot_CPU18 , normalized_rows_tot_CPU24 ,
   normalized_rows_tot_CPU26 )
 mem_access_time_CPU2: m_access_time_CPU2 = PWL ( mem_tot_CPU2 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU3: m_access_time_CPU3 = PWL ( mem_tot_CPU3 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU4: m_access_time_CPU4 = PWL ( mem_tot_CPU4 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU7: m_access_time_CPU7 = PWL ( mem_tot_CPU7 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU17: m_access_time_CPU17 = PWL ( mem_tot_CPU17 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU18: m_access_time_CPU18 = PWL ( mem_tot_CPU18 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU24: m_access_time_CPU24 = PWL ( mem_tot_CPU24 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU26: m_access_time_CPU26 = PWL ( mem_tot_CPU26 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

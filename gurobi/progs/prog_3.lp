\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU_Cluster: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU1 + rows_tot_CPU2 + rows_tot_CPU3 + rows_tot_CPU4
   + rows_tot_CPU5 + rows_tot_CPU6 + rows_tot_CPU7 + rows_tot_CPU8
  CPU_Cluster_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU_Cluster
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_rows_P4: Priority=80 Weight=1 AbsTol=0 RelTol=0
   rows_tot_tor_P41 + rows_tot_tor_P42 + rows_tot_tor_P43 + rows_tot_tor_P44
   + rows_tot_l1_P41 + rows_tot_l1_P42 + rows_tot_l2_P4
  P4_rows_load: Priority=70 Weight=1 AbsTol=0 RelTol=0
   max_rows_P4
  tot_mem_CPU_Cluster: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU1 + mem_tot_CPU2 + mem_tot_CPU3 + mem_tot_CPU4 + mem_tot_CPU5
   + mem_tot_CPU6 + mem_tot_CPU7 + mem_tot_CPU8
  CPU_Cluster_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
  tot_mem_P4: Priority=40 Weight=1 AbsTol=0 RelTol=0
   mem_tot_tor_P41 + mem_tot_tor_P42 + mem_tot_tor_P43 + mem_tot_tor_P44
   + mem_tot_l1_P41 + mem_tot_l1_P42 + mem_tot_l2_P4
  P4_load_mem: Priority=30 Weight=1 AbsTol=0 RelTol=0
   max_mem_P4
Subject To
 cov_0_12: frac[8,12] + frac[4,12] + frac[10,12] + frac[14,12]
   + frac[13,12] + frac[12,12] + frac[0,12] >= 0.8
 cov_1_46: frac[9,46] + frac[12,46] + frac[2,46] + frac[0,46] + frac[8,46]
   >= 0.9
 cov_2_34: frac[1,34] + frac[8,34] + frac[0,34] >= 0.7
 cov_3_20: frac[13,20] + frac[9,20] + frac[12,20] + frac[6,20]
   + frac[11,20] + frac[3,20] + frac[14,20] >= 0.8
 cov_4_16: frac[13,16] + frac[5,16] + frac[12,16] + frac[3,16]
   + frac[14,16] + frac[10,16] + frac[9,16] >= 1
 cov_5_17: frac[0,17] + frac[8,17] + frac[1,17] >= 0.9
 cov_6_55: frac[0,55] + frac[10,55] + frac[12,55] + frac[8,55] + frac[4,55]
   + frac[13,55] + frac[14,55] >= 0.7
 cov_7_34: frac[5,34] + frac[10,34] + frac[12,34] + frac[1,34] + frac[8,34]
   + frac[14,34] + frac[13,34] >= 0.9
 cov_8_40: frac[0,40] + frac[8,40] + frac[13,40] + frac[12,40]
   + frac[14,40] + frac[4,40] + frac[10,40] >= 0.7
 cov_9_65: frac[11,65] + frac[6,65] + frac[4,65] + frac[10,65]
   + frac[13,65] >= 0.7
 cov_10_34: frac[9,34] + frac[2,34] + frac[3,34] >= 0.8
 cov_11_57: frac[5,57] + frac[4,57] + frac[10,57] >= 0.8
 cov_12_38: frac[2,38] + frac[14,38] + frac[4,38] + frac[13,38]
   + frac[9,38] + frac[10,38] + frac[12,38] >= 0.8
 cov_13_40: frac[0,40] + frac[8,40] + frac[13,40] + frac[12,40]
   + frac[14,40] + frac[4,40] + frac[10,40] >= 0.9
 cov_14_28: frac[11,28] + frac[7,28] + frac[13,28] + frac[4,28]
   + frac[10,28] >= 0.7
 cov_15_25: frac[8,25] + frac[13,25] + frac[0,25] + frac[12,25]
   + frac[5,25] + frac[14,25] + frac[10,25] >= 0.7
 cov_16_69: frac[4,69] + frac[10,69] + frac[11,69] + frac[6,69]
   + frac[13,69] >= 1
 cov_17_37: frac[13,37] + frac[3,37] + frac[9,37] + frac[12,37]
   + frac[4,37] + frac[10,37] + frac[14,37] >= 0.8
 cov_18_18: frac[2,18] + frac[6,18] + frac[13,18] + frac[14,18]
   + frac[12,18] + frac[9,18] + frac[11,18] >= 0.7
 cov_19_42: frac[4,42] + frac[5,42] + frac[10,42] >= 0.9
 cov_20_59: frac[12,59] + frac[0,59] + frac[9,59] + frac[2,59] + frac[8,59]
   >= 0.9
 cov_21_3: frac[9,3] + frac[10,3] + frac[12,3] + frac[3,3] + frac[13,3]
   + frac[14,3] + frac[5,3] >= 0.9
 cov_22_13: frac[13,13] + frac[3,13] + frac[12,13] + frac[9,13]
   + frac[4,13] + frac[10,13] + frac[14,13] >= 1
 cov_23_41: frac[12,41] + frac[14,41] + frac[1,41] + frac[4,41]
   + frac[10,41] + frac[8,41] + frac[13,41] >= 0.9
 cov_24_19: frac[10,19] + frac[11,19] + frac[4,19] + frac[13,19]
   + frac[6,19] >= 1
 cov_25_62: frac[9,62] + frac[12,62] + frac[10,62] + frac[4,62]
   + frac[2,62] + frac[13,62] + frac[14,62] >= 0.7
 cov_26_43: frac[14,43] + frac[5,43] + frac[12,43] + frac[2,43]
   + frac[9,43] + frac[10,43] + frac[13,43] >= 0.9
 cov_27_33: frac[2,33] + frac[7,33] + frac[13,33] + frac[14,33]
   + frac[11,33] + frac[12,33] + frac[9,33] >= 1
 cov_28_22: frac[10,22] + frac[12,22] + frac[2,22] + frac[14,22]
   + frac[13,22] + frac[9,22] + frac[5,22] >= 0.7
 cov_29_15: frac[5,15] + frac[10,15] + frac[4,15] >= 0.8
 cov_30_70: frac[13,70] + frac[11,70] + frac[1,70] + frac[12,70]
   + frac[8,70] + frac[14,70] + frac[6,70] >= 0.8
 cov_31_12: frac[8,12] + frac[6,12] + frac[14,12] + frac[11,12]
   + frac[13,12] + frac[12,12] + frac[1,12] >= 0.7
 cov_32_45: frac[11,45] + frac[7,45] + frac[14,45] + frac[13,45]
   + frac[3,45] + frac[12,45] + frac[9,45] >= 1
 cov_33_4: frac[9,4] + frac[12,4] + frac[8,4] + frac[0,4] + frac[2,4]
   >= 0.8
 cov_34_31: frac[14,31] + frac[12,31] + frac[8,31] + frac[13,31]
   + frac[11,31] + frac[7,31] + frac[0,31] >= 0.9
 cov_35_56: frac[1,56] + frac[0,56] + frac[8,56] >= 0.8
 cov_36_54: frac[2,54] + frac[13,54] + frac[14,54] + frac[9,54]
   + frac[10,54] + frac[12,54] + frac[4,54] >= 1
 cov_37_13: frac[1,13] + frac[12,13] + frac[9,13] + frac[8,13] + frac[2,13]
   >= 1
 cov_38_3: frac[7,3] + frac[11,3] + frac[12,3] + frac[13,3] + frac[8,3]
   + frac[14,3] + frac[0,3] >= 0.7
 cov_39_41: frac[12,41] + frac[14,41] + frac[1,41] + frac[4,41]
   + frac[10,41] + frac[8,41] + frac[13,41] >= 1
 cov_40_78: frac[12,78] + frac[2,78] + frac[10,78] + frac[14,78]
   + frac[13,78] + frac[9,78] + frac[5,78] >= 0.7
 cov_41_46: frac[9,46] + frac[5,46] + frac[12,46] + frac[2,46]
   + frac[10,46] + frac[14,46] + frac[13,46] >= 0.8
 cov_42_56: frac[13,56] + frac[6,56] + frac[11,56] + frac[5,56]
   + frac[10,56] >= 0.7
 cov_43_7: frac[10,7] + frac[4,7] + frac[6,7] + frac[13,7] + frac[11,7]
   >= 0.9
 cov_44_29: frac[4,29] + frac[10,29] + frac[6,29] + frac[11,29]
   + frac[13,29] >= 0.7
 cov_45_52: frac[5,52] + frac[4,52] + frac[10,52] >= 0.9
 cov_46_66: frac[4,66] + frac[13,66] + frac[6,66] + frac[10,66]
   + frac[11,66] >= 0.9
 cov_47_11: frac[4,11] + frac[13,11] + frac[8,11] + frac[14,11]
   + frac[1,11] + frac[12,11] + frac[10,11] >= 0.8
 cov_48_33: frac[13,33] + frac[14,33] + frac[12,33] + frac[3,33]
   + frac[4,33] + frac[9,33] + frac[10,33] >= 0.8
 cov_49_29: frac[4,29] + frac[10,29] + frac[6,29] + frac[11,29]
   + frac[13,29] >= 1
 cov_50_48: frac[10,48] + frac[14,48] + frac[8,48] + frac[13,48]
   + frac[12,48] + frac[5,48] + frac[0,48] >= 1
 cov_51_71: frac[10,71] + frac[6,71] + frac[13,71] + frac[5,71]
   + frac[11,71] >= 0.7
 cov_52_29: frac[5,29] + frac[10,29] + frac[11,29] + frac[7,29]
   + frac[13,29] >= 0.7
 cov_53_20: frac[13,20] + frac[4,20] + frac[10,20] + frac[6,20]
   + frac[11,20] >= 0.9
 cov_54_26: frac[14,26] + frac[10,26] + frac[9,26] + frac[4,26]
   + frac[3,26] + frac[13,26] + frac[12,26] >= 1
 cov_55_77: frac[4,77] + frac[11,77] + frac[10,77] + frac[6,77]
   + frac[13,77] >= 0.9
 cov_56_64: frac[7,64] + frac[13,64] + frac[14,64] + frac[11,64]
   + frac[12,64] + frac[3,64] + frac[9,64] >= 0.9
 cov_57_59: frac[10,59] + frac[4,59] + frac[13,59] + frac[6,59]
   + frac[11,59] >= 1
 cov_58_8: frac[11,8] + frac[5,8] + frac[10,8] + frac[6,8] + frac[13,8]
   >= 1
 cov_59_61: frac[10,61] + frac[0,61] + frac[13,61] + frac[12,61]
   + frac[14,61] + frac[8,61] + frac[4,61] >= 0.7
 cov_60_13: frac[13,13] + frac[3,13] + frac[12,13] + frac[9,13]
   + frac[6,13] + frac[11,13] + frac[14,13] >= 1
 cov_61_65: frac[11,65] + frac[10,65] + frac[7,65] + frac[13,65]
   + frac[5,65] >= 0.8
 cov_62_14: frac[10,14] + frac[6,14] + frac[4,14] + frac[13,14]
   + frac[11,14] >= 0.8
 cov_63_74: frac[14,74] + frac[11,74] + frac[12,74] + frac[9,74]
   + frac[3,74] + frac[13,74] + frac[6,74] >= 0.9
 cov_64_49: frac[13,49] + frac[6,49] + frac[11,49] + frac[12,49]
   + frac[3,49] + frac[9,49] + frac[14,49] >= 0.7
 cov_65_26: frac[11,26] + frac[14,26] + frac[9,26] + frac[2,26]
   + frac[13,26] + frac[7,26] + frac[12,26] >= 1
 cov_66_66: frac[13,66] + frac[14,66] + frac[5,66] + frac[9,66]
   + frac[10,66] + frac[12,66] + frac[2,66] >= 0.8
 cov_67_62: frac[12,62] + frac[10,62] + frac[8,62] + frac[4,62]
   + frac[13,62] + frac[0,62] + frac[14,62] >= 0.9
 cov_68_58: frac[13,58] + frac[6,58] + frac[1,58] + frac[8,58]
   + frac[12,58] + frac[11,58] + frac[14,58] >= 0.8
 cov_69_37: frac[13,37] + frac[5,37] + frac[12,37] + frac[8,37]
   + frac[10,37] + frac[0,37] + frac[14,37] >= 0.7
 cov_70_73: frac[13,73] + frac[0,73] + frac[8,73] + frac[11,73]
   + frac[6,73] + frac[14,73] + frac[12,73] >= 0.8
 cov_71_78: frac[12,78] + frac[10,78] + frac[0,78] + frac[8,78]
   + frac[14,78] + frac[4,78] + frac[13,78] >= 0.8
 cov_72_33: frac[8,33] + frac[0,33] + frac[13,33] + frac[14,33]
   + frac[12,33] + frac[4,33] + frac[10,33] >= 0.7
 cov_73_0: frac[1,0] + frac[8,0] + frac[0,0] >= 0.7
 cov_74_62: frac[10,62] + frac[4,62] + frac[13,62] + frac[11,62]
   + frac[7,62] >= 0.8
 cov_75_19: frac[14,19] + frac[11,19] + frac[12,19] + frac[1,19]
   + frac[13,19] + frac[8,19] + frac[7,19] >= 0.7
 cov_76_43: frac[14,43] + frac[11,43] + frac[12,43] + frac[1,43]
   + frac[13,43] + frac[6,43] + frac[8,43] >= 0.8
 cov_77_17: frac[12,17] + frac[4,17] + frac[14,17] + frac[10,17]
   + frac[8,17] + frac[1,17] + frac[13,17] >= 0.9
 cov_78_71: frac[12,71] + frac[14,71] + frac[13,71] + frac[9,71]
   + frac[11,71] + frac[7,71] + frac[3,71] >= 1
 cov_79_43: frac[14,43] + frac[5,43] + frac[12,43] + frac[10,43]
   + frac[13,43] + frac[0,43] + frac[8,43] >= 0.9
 cov_80_30: frac[9,30] + frac[12,30] + frac[3,30] + frac[1,30] + frac[8,30]
   >= 0.7
 cov_81_40: frac[9,40] + frac[0,40] + frac[8,40] + frac[12,40] + frac[2,40]
   >= 0.8
 cov_82_43: frac[14,43] + frac[11,43] + frac[12,43] + frac[13,43]
   + frac[0,43] + frac[8,43] + frac[7,43] >= 0.8
 cov_83_14: frac[6,14] + frac[11,14] + frac[7,14] >= 1
 cov_84_13: frac[13,13] + frac[12,13] + frac[5,13] + frac[0,13]
   + frac[8,13] + frac[10,13] + frac[14,13] >= 1
 cov_85_27: frac[5,27] + frac[10,27] + frac[4,27] >= 0.7
 cov_86_29: frac[9,29] + frac[2,29] + frac[3,29] >= 1
 cov_87_39: frac[9,39] + frac[12,39] + frac[8,39] + frac[2,39] + frac[0,39]
   >= 0.8
 cov_88_78: frac[10,78] + frac[6,78] + frac[13,78] + frac[11,78]
   + frac[5,78] >= 0.9
 cov_89_9: frac[7,9] + frac[6,9] + frac[11,9] >= 1
 cov_90_43: frac[11,43] + frac[10,43] + frac[4,43] + frac[13,43]
   + frac[6,43] >= 0.9
 cov_91_4: frac[3,4] + frac[9,4] + frac[12,4] + frac[8,4] + frac[0,4]
   >= 0.8
 cov_92_61: frac[10,61] + frac[13,61] + frac[12,61] + frac[5,61]
   + frac[14,61] + frac[8,61] + frac[1,61] >= 0.7
 cov_93_60: frac[3,60] + frac[9,60] + frac[8,60] + frac[1,60] + frac[12,60]
   >= 0.8
 cov_94_9: frac[9,9] + frac[13,9] + frac[6,9] + frac[11,9] + frac[14,9]
   + frac[12,9] + frac[3,9] >= 0.8
 cov_95_12: frac[4,12] + frac[2,12] + frac[10,12] + frac[14,12]
   + frac[13,12] + frac[12,12] + frac[9,12] >= 1
 cov_96_16: frac[6,16] + frac[13,16] + frac[4,16] + frac[11,16]
   + frac[10,16] >= 0.8
 cov_97_73: frac[8,73] + frac[12,73] + frac[3,73] + frac[1,73] + frac[9,73]
   >= 0.9
 cov_98_31: frac[14,31] + frac[12,31] + frac[8,31] + frac[13,31]
   + frac[11,31] + frac[7,31] + frac[0,31] >= 0.7
 cov_99_63: frac[8,63] + frac[13,63] + frac[0,63] + frac[5,63]
   + frac[14,63] + frac[12,63] + frac[10,63] >= 0.7
 cov_100_16: frac[2,16] + frac[6,16] + frac[13,16] + frac[12,16]
   + frac[11,16] + frac[14,16] + frac[9,16] >= 0.8
 cov_101_72: frac[12,72] + frac[2,72] + frac[9,72] + frac[0,72]
   + frac[8,72] >= 1
 cov_102_2: frac[13,2] + frac[7,2] + frac[10,2] + frac[4,2] + frac[11,2]
   >= 0.8
 cov_103_12: frac[8,12] + frac[14,12] + frac[11,12] + frac[7,12]
   + frac[13,12] + frac[12,12] + frac[0,12] >= 0.8
 cov_104_40: frac[0,40] + frac[8,40] + frac[7,40] + frac[13,40]
   + frac[12,40] + frac[14,40] + frac[11,40] >= 0.7
 cov_105_39: frac[5,39] + frac[1,39] + frac[10,39] + frac[12,39]
   + frac[8,39] + frac[14,39] + frac[13,39] >= 0.9
 cov_106_4: frac[13,4] + frac[12,4] + frac[8,4] + frac[4,4] + frac[0,4]
   + frac[10,4] + frac[14,4] >= 0.8
 cov_107_49: frac[13,49] + frac[2,49] + frac[12,49] + frac[4,49]
   + frac[9,49] + frac[10,49] + frac[14,49] >= 1
 cov_108_55: frac[7,55] + frac[10,55] + frac[13,55] + frac[11,55]
   + frac[5,55] >= 0.8
 cov_109_9: frac[10,9] + frac[7,9] + frac[13,9] + frac[11,9] + frac[5,9]
   >= 0.7
 cov_110_70: frac[13,70] + frac[5,70] + frac[11,70] + frac[10,70]
   + frac[6,70] >= 0.7
 cov_111_52: frac[1,52] + frac[13,52] + frac[14,52] + frac[12,52]
   + frac[8,52] + frac[4,52] + frac[10,52] >= 1
 cov_112_62: frac[9,62] + frac[12,62] + frac[3,62] + frac[10,62]
   + frac[4,62] + frac[13,62] + frac[14,62] >= 0.9
 cov_113_24: frac[3,24] + frac[14,24] + frac[9,24] + frac[6,24]
   + frac[13,24] + frac[12,24] + frac[11,24] >= 0.9
 cov_114_7: frac[10,7] + frac[6,7] + frac[13,7] + frac[5,7] + frac[11,7]
   >= 0.9
 cov_115_56: frac[7,56] + frac[13,56] + frac[11,56] + frac[5,56]
   + frac[10,56] >= 0.9
 cov_116_5: frac[8,5] + frac[4,5] + frac[0,5] + frac[10,5] + frac[14,5]
   + frac[13,5] + frac[12,5] >= 0.7
 cov_117_8: frac[14,8] + frac[12,8] + frac[5,8] + frac[2,8] + frac[10,8]
   + frac[9,8] + frac[13,8] >= 1
 cov_118_54: frac[13,54] + frac[14,54] + frac[0,54] + frac[5,54]
   + frac[10,54] + frac[12,54] + frac[8,54] >= 0.9
 cov_119_33: frac[13,33] + frac[5,33] + frac[6,33] + frac[11,33]
   + frac[10,33] >= 1
 cov_120_26: frac[11,26] + frac[14,26] + frac[6,26] + frac[1,26]
   + frac[13,26] + frac[8,26] + frac[12,26] >= 0.9
 cov_121_15: frac[9,15] + frac[3,15] + frac[12,15] + frac[0,15]
   + frac[8,15] >= 0.9
 cov_122_3: frac[1,3] + frac[8,3] + frac[0,3] >= 1
 cov_123_11: frac[3,11] + frac[0,11] + frac[8,11] + frac[12,11]
   + frac[9,11] >= 0.8
 cov_124_26: frac[11,26] + frac[14,26] + frac[9,26] + frac[3,26]
   + frac[6,26] + frac[13,26] + frac[12,26] >= 0.8
 cov_125_28: frac[2,28] + frac[3,28] + frac[9,28] >= 0.7
 cov_126_24: frac[4,24] + frac[10,24] + frac[7,24] + frac[13,24]
   + frac[11,24] >= 0.8
 cov_127_7: frac[12,7] + frac[8,7] + frac[14,7] + frac[6,7] + frac[13,7]
   + frac[11,7] + frac[1,7] >= 1
 cov_128_5: frac[4,5] + frac[10,5] + frac[6,5] + frac[11,5] + frac[13,5]
   >= 0.7
 cov_129_37: frac[13,37] + frac[3,37] + frac[9,37] + frac[12,37]
   + frac[4,37] + frac[10,37] + frac[14,37] >= 0.9
 cov_130_7: frac[12,7] + frac[8,7] + frac[9,7] + frac[1,7] + frac[3,7]
   >= 1
 cov_131_18: frac[8,18] + frac[6,18] + frac[13,18] + frac[14,18]
   + frac[0,18] + frac[12,18] + frac[11,18] >= 0.7
 cov_132_47: frac[2,47] + frac[10,47] + frac[14,47] + frac[13,47]
   + frac[9,47] + frac[5,47] + frac[12,47] >= 1
 cov_133_41: frac[12,41] + frac[3,41] + frac[9,41] + frac[0,41]
   + frac[8,41] >= 1
 cov_134_15: frac[1,15] + frac[9,15] + frac[12,15] + frac[2,15]
   + frac[8,15] >= 1
 cov_135_1: frac[14,1] + frac[11,1] + frac[12,1] + frac[8,1] + frac[0,1]
   + frac[7,1] + frac[13,1] >= 0.7
 cov_136_29: frac[9,29] + frac[2,29] + frac[3,29] >= 0.8
 cov_137_38: frac[8,38] + frac[14,38] + frac[13,38] + frac[5,38]
   + frac[1,38] + frac[10,38] + frac[12,38] >= 0.9
 cov_138_0: frac[8,0] + frac[0,0] + frac[14,0] + frac[6,0] + frac[13,0]
   + frac[12,0] + frac[11,0] >= 0.8
 cov_139_3: frac[10,3] + frac[1,3] + frac[12,3] + frac[4,3] + frac[13,3]
   + frac[8,3] + frac[14,3] >= 0.7
 cov_140_31: frac[14,31] + frac[12,31] + frac[8,31] + frac[13,31]
   + frac[11,31] + frac[7,31] + frac[0,31] >= 0.9
 cov_141_63: frac[8,63] + frac[0,63] + frac[9,63] + frac[3,63]
   + frac[12,63] >= 0.7
 cov_142_61: frac[10,61] + frac[6,61] + frac[11,61] + frac[13,61]
   + frac[5,61] >= 0.8
 cov_143_1: frac[14,1] + frac[11,1] + frac[12,1] + frac[8,1] + frac[0,1]
   + frac[7,1] + frac[13,1] >= 1
 cov_144_25: frac[7,25] + frac[8,25] + frac[1,25] + frac[13,25]
   + frac[11,25] + frac[12,25] + frac[14,25] >= 0.8
 cov_145_21: frac[13,21] + frac[2,21] + frac[9,21] + frac[14,21]
   + frac[12,21] + frac[4,21] + frac[10,21] >= 0.7
 cov_146_30: frac[13,30] + frac[11,30] + frac[7,30] + frac[5,30]
   + frac[10,30] >= 0.8
 cov_147_32: frac[13,32] + frac[12,32] + frac[5,32] + frac[10,32]
   + frac[8,32] + frac[0,32] + frac[14,32] >= 0.7
 cov_148_2: frac[8,2] + frac[14,2] + frac[0,2] + frac[13,2] + frac[10,2]
   + frac[12,2] + frac[4,2] >= 0.8
 cov_149_11: frac[13,11] + frac[7,11] + frac[8,11] + frac[14,11]
   + frac[11,11] + frac[1,11] + frac[12,11] >= 0.8
 cov_150_77: frac[14,77] + frac[5,77] + frac[12,77] + frac[0,77]
   + frac[8,77] + frac[10,77] + frac[13,77] >= 1
 cov_151_68: frac[3,68] + frac[9,68] + frac[12,68] + frac[8,68]
   + frac[0,68] >= 0.9
 cov_152_1: frac[5,1] + frac[4,1] + frac[10,1] >= 1
 cov_153_17: frac[12,17] + frac[14,17] + frac[8,17] + frac[1,17]
   + frac[13,17] + frac[6,17] + frac[11,17] >= 0.7
 cov_154_5: frac[10,5] + frac[11,5] + frac[7,5] + frac[13,5] + frac[5,5]
   >= 1
 cov_155_56: frac[2,56] + frac[13,56] + frac[12,56] + frac[5,56]
   + frac[10,56] + frac[14,56] + frac[9,56] >= 0.8
 cov_156_13: frac[13,13] + frac[3,13] + frac[12,13] + frac[9,13]
   + frac[11,13] + frac[7,13] + frac[14,13] >= 0.7
 cov_157_79: frac[10,79] + frac[13,79] + frac[4,79] + frac[11,79]
   + frac[7,79] >= 0.7
 cov_158_73: frac[2,73] + frac[0,73] + frac[8,73] + frac[12,73]
   + frac[9,73] >= 0.7
 cov_159_46: frac[1,46] + frac[5,46] + frac[12,46] + frac[10,46]
   + frac[8,46] + frac[14,46] + frac[13,46] >= 0.8
 cov_160_50: frac[11,50] + frac[13,50] + frac[6,50] + frac[5,50]
   + frac[10,50] >= 1
 cov_161_0: frac[9,0] + frac[14,0] + frac[13,0] + frac[12,0] + frac[4,0]
   + frac[10,0] + frac[2,0] >= 1
 cov_162_15: frac[9,15] + frac[5,15] + frac[12,15] + frac[2,15]
   + frac[10,15] + frac[14,15] + frac[13,15] >= 0.9
 cov_163_53: frac[2,53] + frac[14,53] + frac[12,53] + frac[11,53]
   + frac[7,53] + frac[13,53] + frac[9,53] >= 0.7
 cov_164_6: frac[13,6] + frac[5,6] + frac[11,6] + frac[7,6] + frac[10,6]
   >= 0.8
 cov_165_34: frac[10,34] + frac[12,34] + frac[9,34] + frac[2,34]
   + frac[4,34] + frac[14,34] + frac[13,34] >= 0.7
 cov_166_1: frac[6,1] + frac[11,1] + frac[7,1] >= 1
 cov_167_34: frac[5,34] + frac[10,34] + frac[12,34] + frac[9,34]
   + frac[3,34] + frac[14,34] + frac[13,34] >= 0.8
 cov_168_19: frac[9,19] + frac[0,19] + frac[12,19] + frac[8,19]
   + frac[2,19] >= 0.7
 cov_169_62: frac[5,62] + frac[10,62] + frac[4,62] >= 1
 cov_170_63: frac[8,63] + frac[0,63] + frac[1,63] >= 0.8
 cov_171_71: frac[10,71] + frac[12,71] + frac[8,71] + frac[14,71]
   + frac[0,71] + frac[13,71] + frac[5,71] >= 1
 cov_172_75: frac[10,75] + frac[4,75] + frac[5,75] >= 0.8
 cov_173_57: frac[14,57] + frac[8,57] + frac[7,57] + frac[13,57]
   + frac[11,57] + frac[12,57] + frac[0,57] >= 0.8
 cov_174_22: frac[11,22] + frac[10,22] + frac[4,22] + frac[6,22]
   + frac[13,22] >= 1
 cov_175_37: frac[13,37] + frac[9,37] + frac[12,37] + frac[2,37]
   + frac[14,37] + frac[6,37] + frac[11,37] >= 1
 cov_176_22: frac[10,22] + frac[4,22] + frac[5,22] >= 1
 cov_177_3: frac[11,3] + frac[1,3] + frac[12,3] + frac[13,3] + frac[8,3]
   + frac[14,3] + frac[6,3] >= 0.8
 cov_178_0: frac[9,0] + frac[7,0] + frac[14,0] + frac[13,0] + frac[12,0]
   + frac[11,0] + frac[2,0] >= 1
 cov_179_50: frac[12,50] + frac[14,50] + frac[11,50] + frac[8,50]
   + frac[13,50] + frac[6,50] + frac[1,50] >= 0.9
 cov_180_48: frac[11,48] + frac[10,48] + frac[13,48] + frac[6,48]
   + frac[5,48] >= 0.8
 cov_181_6: frac[13,6] + frac[11,6] + frac[7,6] + frac[10,6] + frac[4,6]
   >= 1
 cov_182_53: frac[14,53] + frac[12,53] + frac[8,53] + frac[1,53]
   + frac[6,53] + frac[11,53] + frac[13,53] >= 1
 cov_183_77: frac[4,77] + frac[11,77] + frac[10,77] + frac[6,77]
   + frac[13,77] >= 1
 cov_184_27: frac[6,27] + frac[8,27] + frac[11,27] + frac[12,27]
   + frac[14,27] + frac[1,27] + frac[13,27] >= 1
 cov_185_27: frac[6,27] + frac[7,27] + frac[11,27] >= 1
 cov_186_53: frac[12,53] + frac[8,53] + frac[0,53] + frac[3,53]
   + frac[9,53] >= 1
 cov_187_63: frac[11,63] + frac[7,63] + frac[6,63] >= 0.9
 cov_188_14: frac[0,14] + frac[8,14] + frac[9,14] + frac[12,14]
   + frac[3,14] >= 0.9
 cov_189_53: frac[4,53] + frac[10,53] + frac[11,53] + frac[7,53]
   + frac[13,53] >= 0.8
 cov_190_27: frac[2,27] + frac[12,27] + frac[9,27] + frac[14,27]
   + frac[10,27] + frac[4,27] + frac[13,27] >= 1
 cov_191_79: frac[9,79] + frac[3,79] + frac[12,79] + frac[8,79]
   + frac[1,79] >= 0.8
 cov_192_71: frac[10,71] + frac[4,71] + frac[5,71] >= 0.7
 cov_193_38: frac[2,38] + frac[14,38] + frac[4,38] + frac[13,38]
   + frac[9,38] + frac[10,38] + frac[12,38] >= 1
 cov_194_34: frac[12,34] + frac[11,34] + frac[8,34] + frac[14,34]
   + frac[13,34] + frac[0,34] + frac[6,34] >= 0.9
 cov_195_70: frac[13,70] + frac[9,70] + frac[11,70] + frac[7,70]
   + frac[3,70] + frac[12,70] + frac[14,70] >= 0.8
 cov_196_71: frac[10,71] + frac[4,71] + frac[5,71] >= 1
 cov_197_65: frac[14,65] + frac[11,65] + frac[3,65] + frac[12,65]
   + frac[9,65] + frac[7,65] + frac[13,65] >= 0.7
 cov_198_73: frac[2,73] + frac[0,73] + frac[8,73] + frac[12,73]
   + frac[9,73] >= 0.8
 cov_199_35: frac[3,35] + frac[9,35] + frac[2,35] >= 0.9
 cov_200_55: frac[10,55] + frac[4,55] + frac[5,55] >= 1
 cov_201_29: frac[12,29] + frac[9,29] + frac[5,29] + frac[14,29]
   + frac[10,29] + frac[13,29] + frac[3,29] >= 0.8
 cov_202_2: frac[14,2] + frac[6,2] + frac[13,2] + frac[12,2] + frac[2,2]
   + frac[9,2] + frac[11,2] >= 0.8
 cov_203_68: frac[11,68] + frac[13,68] + frac[9,68] + frac[12,68]
   + frac[2,68] + frac[14,68] + frac[6,68] >= 0.9
 cov_204_76: frac[11,76] + frac[7,76] + frac[14,76] + frac[13,76]
   + frac[3,76] + frac[9,76] + frac[12,76] >= 0.7
 cov_205_50: frac[12,50] + frac[9,50] + frac[8,50] + frac[2,50]
   + frac[1,50] >= 0.7
 cov_206_11: frac[13,11] + frac[7,11] + frac[8,11] + frac[14,11]
   + frac[11,11] + frac[1,11] + frac[12,11] >= 1
 cov_207_23: frac[0,23] + frac[1,23] + frac[8,23] >= 1
 cov_208_16: frac[2,16] + frac[0,16] + frac[12,16] + frac[9,16]
   + frac[8,16] >= 0.8
 cov_209_36: frac[2,36] + frac[14,36] + frac[6,36] + frac[11,36]
   + frac[13,36] + frac[9,36] + frac[12,36] >= 0.9
 cov_210_57: frac[3,57] + frac[9,57] + frac[14,57] + frac[7,57]
   + frac[13,57] + frac[11,57] + frac[12,57] >= 0.9
 cov_211_67: frac[9,67] + frac[1,67] + frac[12,67] + frac[2,67]
   + frac[8,67] >= 1
 cov_212_54: frac[1,54] + frac[0,54] + frac[8,54] >= 0.8
 cov_213_56: frac[7,56] + frac[13,56] + frac[1,56] + frac[12,56]
   + frac[11,56] + frac[14,56] + frac[8,56] >= 1
 cov_214_58: frac[8,58] + frac[12,58] + frac[0,58] + frac[9,58]
   + frac[3,58] >= 0.8
 cov_215_3: frac[7,3] + frac[9,3] + frac[11,3] + frac[12,3] + frac[2,3]
   + frac[13,3] + frac[14,3] >= 0.8
 cov_216_56: frac[2,56] + frac[12,56] + frac[0,56] + frac[9,56]
   + frac[8,56] >= 0.7
 cov_217_67: frac[0,67] + frac[10,67] + frac[12,67] + frac[4,67]
   + frac[13,67] + frac[8,67] + frac[14,67] >= 0.9
 cov_218_67: frac[0,67] + frac[1,67] + frac[8,67] >= 0.9
 cov_219_25: frac[8,25] + frac[1,25] + frac[13,25] + frac[12,25]
   + frac[5,25] + frac[14,25] + frac[10,25] >= 1
 cov_220_6: frac[13,6] + frac[11,6] + frac[1,6] + frac[12,6] + frac[8,6]
   + frac[14,6] + frac[6,6] >= 1
 cov_221_72: frac[12,72] + frac[1,72] + frac[2,72] + frac[9,72]
   + frac[8,72] >= 0.7
 cov_222_27: frac[8,27] + frac[0,27] + frac[1,27] >= 0.8
 cov_223_65: frac[11,65] + frac[10,65] + frac[7,65] + frac[13,65]
   + frac[5,65] >= 0.8
 cov_224_22: frac[11,22] + frac[7,22] + frac[10,22] + frac[4,22]
   + frac[13,22] >= 0.7
 cov_225_59: frac[12,59] + frac[9,59] + frac[10,59] + frac[14,59]
   + frac[13,59] + frac[2,59] + frac[5,59] >= 0.7
 cov_226_70: frac[13,70] + frac[9,70] + frac[5,70] + frac[10,70]
   + frac[12,70] + frac[2,70] + frac[14,70] >= 1
 cov_227_30: frac[13,30] + frac[0,30] + frac[11,30] + frac[14,30]
   + frac[12,30] + frac[6,30] + frac[8,30] >= 0.8
 cov_228_25: frac[13,25] + frac[2,25] + frac[12,25] + frac[5,25]
   + frac[9,25] + frac[14,25] + frac[10,25] >= 1
 cov_229_45: frac[10,45] + frac[6,45] + frac[11,45] + frac[13,45]
   + frac[5,45] >= 0.9
 cov_230_42: frac[3,42] + frac[0,42] + frac[8,42] + frac[12,42]
   + frac[9,42] >= 0.9
 cov_231_42: frac[4,42] + frac[13,42] + frac[0,42] + frac[8,42]
   + frac[14,42] + frac[12,42] + frac[10,42] >= 0.7
 cov_232_22: frac[12,22] + frac[8,22] + frac[1,22] + frac[2,22]
   + frac[9,22] >= 1
 cov_233_30: frac[13,30] + frac[11,30] + frac[7,30] + frac[14,30]
   + frac[9,30] + frac[12,30] + frac[3,30] >= 0.8
 cov_234_19: frac[14,19] + frac[10,19] + frac[0,19] + frac[12,19]
   + frac[4,19] + frac[13,19] + frac[8,19] >= 1
 cov_235_55: frac[0,55] + frac[8,55] + frac[1,55] >= 0.7
 cov_236_19: frac[14,19] + frac[10,19] + frac[12,19] + frac[4,19]
   + frac[1,19] + frac[13,19] + frac[8,19] >= 0.7
 cov_237_43: frac[5,43] + frac[10,43] + frac[4,43] >= 0.8
 cov_238_26: frac[11,26] + frac[14,26] + frac[6,26] + frac[1,26]
   + frac[13,26] + frac[8,26] + frac[12,26] >= 0.8
 cov_239_11: frac[3,11] + frac[2,11] + frac[9,11] >= 0.8
 cov_240_33: frac[2,33] + frac[8,33] + frac[0,33] + frac[12,33]
   + frac[9,33] >= 0.7
 cov_241_55: frac[12,55] + frac[8,55] + frac[1,55] + frac[9,55]
   + frac[2,55] >= 0.7
 cov_242_38: frac[2,38] + frac[14,38] + frac[13,38] + frac[9,38]
   + frac[5,38] + frac[10,38] + frac[12,38] >= 0.7
 cov_243_32: frac[13,32] + frac[6,32] + frac[12,32] + frac[11,32]
   + frac[8,32] + frac[0,32] + frac[14,32] >= 0.7
 cov_244_11: frac[13,11] + frac[0,11] + frac[7,11] + frac[8,11]
   + frac[14,11] + frac[11,11] + frac[12,11] >= 0.9
 cov_245_14: frac[10,14] + frac[8,14] + frac[14,14] + frac[4,14]
   + frac[13,14] + frac[1,14] + frac[12,14] >= 0.9
 cov_246_68: frac[11,68] + frac[7,68] + frac[13,68] + frac[4,68]
   + frac[10,68] >= 0.9
 cov_247_17: frac[12,17] + frac[14,17] + frac[8,17] + frac[7,17]
   + frac[1,17] + frac[13,17] + frac[11,17] >= 1
 cov_248_43: frac[11,43] + frac[10,43] + frac[4,43] + frac[13,43]
   + frac[7,43] >= 0.7
 cov_249_75: frac[13,75] + frac[7,75] + frac[14,75] + frac[11,75]
   + frac[9,75] + frac[12,75] + frac[3,75] >= 0.7
 cov_250_34: frac[1,34] + frac[8,34] + frac[0,34] >= 1
 cov_251_30: frac[13,30] + frac[0,30] + frac[11,30] + frac[7,30]
   + frac[14,30] + frac[12,30] + frac[8,30] >= 0.7
 cov_252_57: frac[9,57] + frac[8,57] + frac[1,57] + frac[2,57]
   + frac[12,57] >= 0.8
 cov_253_67: frac[6,67] + frac[11,67] + frac[10,67] + frac[4,67]
   + frac[13,67] >= 0.8
 cov_254_58: frac[6,58] + frac[7,58] + frac[11,58] >= 0.9
 cov_255_64: frac[2,64] + frac[3,64] + frac[9,64] >= 0.8
 cov_256_6: frac[11,6] + frac[7,6] + frac[6,6] >= 0.8
 cov_257_18: frac[2,18] + frac[9,18] + frac[3,18] >= 0.8
 cov_258_49: frac[2,49] + frac[3,49] + frac[9,49] >= 0.7
 cov_259_3: frac[1,3] + frac[8,3] + frac[0,3] >= 0.9
 cov_260_65: frac[11,65] + frac[10,65] + frac[7,65] + frac[13,65]
   + frac[5,65] >= 0.7
 cov_261_54: frac[13,54] + frac[11,54] + frac[7,54] + frac[10,54]
   + frac[4,54] >= 0.9
 cov_262_67: frac[0,67] + frac[9,67] + frac[12,67] + frac[2,67]
   + frac[8,67] >= 0.8
 cov_263_7: frac[12,7] + frac[8,7] + frac[0,7] + frac[9,7] + frac[3,7]
   >= 0.8
 cov_264_59: frac[12,59] + frac[0,59] + frac[9,59] + frac[3,59]
   + frac[8,59] >= 0.7
 cov_265_70: frac[13,70] + frac[11,70] + frac[1,70] + frac[12,70]
   + frac[8,70] + frac[14,70] + frac[6,70] >= 1
 cov_266_38: frac[14,38] + frac[13,38] + frac[9,38] + frac[11,38]
   + frac[7,38] + frac[3,38] + frac[12,38] >= 0.9
 cov_267_46: frac[12,46] + frac[10,46] + frac[0,46] + frac[8,46]
   + frac[14,46] + frac[4,46] + frac[13,46] >= 1
 cov_268_65: frac[11,65] + frac[4,65] + frac[10,65] + frac[7,65]
   + frac[13,65] >= 0.7
 cov_269_1: frac[12,1] + frac[3,1] + frac[9,1] + frac[8,1] + frac[0,1]
   >= 0.7
 cov_270_39: frac[1,39] + frac[8,39] + frac[0,39] >= 1
 cov_271_76: frac[4,76] + frac[2,76] + frac[10,76] + frac[14,76]
   + frac[13,76] + frac[9,76] + frac[12,76] >= 0.8
 cov_272_44: frac[7,44] + frac[13,44] + frac[12,44] + frac[3,44]
   + frac[9,44] + frac[14,44] + frac[11,44] >= 0.8
 cov_273_49: frac[7,49] + frac[13,49] + frac[2,49] + frac[11,49]
   + frac[12,49] + frac[9,49] + frac[14,49] >= 0.7
 cov_274_19: frac[10,19] + frac[11,19] + frac[4,19] + frac[13,19]
   + frac[6,19] >= 0.9
 cov_275_73: frac[2,73] + frac[7,73] + frac[13,73] + frac[11,73]
   + frac[14,73] + frac[12,73] + frac[9,73] >= 0.7
 cov_276_42: frac[3,42] + frac[13,42] + frac[14,42] + frac[5,42]
   + frac[12,42] + frac[10,42] + frac[9,42] >= 0.8
 cov_277_20: frac[8,20] + frac[1,20] + frac[0,20] >= 0.8
 cov_278_10: frac[5,10] + frac[10,10] + frac[4,10] >= 0.9
 cov_279_68: frac[11,68] + frac[7,68] + frac[13,68] + frac[12,68]
   + frac[8,68] + frac[0,68] + frac[14,68] >= 0.7
 cov_280_29: frac[5,29] + frac[10,29] + frac[11,29] + frac[7,29]
   + frac[13,29] >= 0.9
 cov_281_33: frac[1,33] + frac[8,33] + frac[13,33] + frac[14,33]
   + frac[5,33] + frac[12,33] + frac[10,33] >= 0.8
 cov_282_57: frac[5,57] + frac[10,57] + frac[7,57] + frac[13,57]
   + frac[11,57] >= 0.9
 cov_283_63: frac[8,63] + frac[2,63] + frac[9,63] + frac[12,63]
   + frac[1,63] >= 0.9
 cov_284_67: frac[9,67] + frac[11,67] + frac[12,67] + frac[2,67]
   + frac[13,67] + frac[7,67] + frac[14,67] >= 0.8
 cov_285_36: frac[4,36] + frac[10,36] + frac[11,36] + frac[7,36]
   + frac[13,36] >= 1
 cov_286_20: frac[13,20] + frac[4,20] + frac[10,20] + frac[6,20]
   + frac[11,20] >= 0.8
 cov_287_39: frac[5,39] + frac[1,39] + frac[10,39] + frac[12,39]
   + frac[8,39] + frac[14,39] + frac[13,39] >= 1
 cov_288_46: frac[11,46] + frac[7,46] + frac[9,46] + frac[12,46]
   + frac[3,46] + frac[14,46] + frac[13,46] >= 1
 cov_289_9: frac[10,9] + frac[13,9] + frac[6,9] + frac[11,9] + frac[4,9]
   >= 0.9
 cov_290_77: frac[14,77] + frac[12,77] + frac[0,77] + frac[8,77]
   + frac[11,77] + frac[6,77] + frac[13,77] >= 1
 cov_291_26: frac[11,26] + frac[14,26] + frac[1,26] + frac[13,26]
   + frac[8,26] + frac[7,26] + frac[12,26] >= 1
 cov_292_12: frac[8,12] + frac[14,12] + frac[11,12] + frac[7,12]
   + frac[13,12] + frac[12,12] + frac[1,12] >= 0.8
 cov_293_64: frac[7,64] + frac[6,64] + frac[11,64] >= 0.8
 cov_294_74: frac[11,74] + frac[6,74] + frac[7,74] >= 0.7
 cov_295_3: frac[7,3] + frac[11,3] + frac[12,3] + frac[13,3] + frac[8,3]
   + frac[14,3] + frac[0,3] >= 0.9
 cov_296_62: frac[10,62] + frac[4,62] + frac[13,62] + frac[11,62]
   + frac[7,62] >= 0.9
 cov_297_7: frac[12,7] + frac[2,7] + frac[14,7] + frac[13,7] + frac[9,7]
   + frac[11,7] + frac[7,7] >= 0.9
 cov_298_24: frac[14,24] + frac[8,24] + frac[6,24] + frac[13,24]
   + frac[0,24] + frac[12,24] + frac[11,24] >= 0.7
 cov_299_19: frac[14,19] + frac[10,19] + frac[12,19] + frac[1,19]
   + frac[13,19] + frac[8,19] + frac[5,19] >= 0.7
 cov_300_35: frac[4,35] + frac[13,35] + frac[7,35] + frac[10,35]
   + frac[11,35] >= 0.9
 cov_301_4: frac[11,4] + frac[7,4] + frac[13,4] + frac[12,4] + frac[8,4]
   + frac[0,4] + frac[14,4] >= 1
 cov_302_40: frac[9,40] + frac[13,40] + frac[12,40] + frac[14,40]
   + frac[5,40] + frac[2,40] + frac[10,40] >= 0.9
 cov_303_41: frac[11,41] + frac[4,41] + frac[10,41] + frac[7,41]
   + frac[13,41] >= 0.8
 cov_304_66: frac[13,66] + frac[14,66] + frac[7,66] + frac[9,66]
   + frac[11,66] + frac[12,66] + frac[3,66] >= 0.9
 cov_305_3: frac[9,3] + frac[12,3] + frac[3,3] + frac[8,3] + frac[0,3]
   >= 0.8
 cov_306_75: frac[0,75] + frac[8,75] + frac[9,75] + frac[12,75]
   + frac[3,75] >= 0.7
 cov_307_55: frac[3,55] + frac[9,55] + frac[2,55] >= 0.8
 cov_308_77: frac[14,77] + frac[9,77] + frac[12,77] + frac[2,77]
   + frac[11,77] + frac[7,77] + frac[13,77] >= 0.9
 cov_309_74: frac[14,74] + frac[11,74] + frac[1,74] + frac[12,74]
   + frac[13,74] + frac[6,74] + frac[8,74] >= 0.9
 cov_310_74: frac[1,74] + frac[0,74] + frac[8,74] >= 0.7
 cov_311_67: frac[6,67] + frac[9,67] + frac[11,67] + frac[3,67]
   + frac[12,67] + frac[13,67] + frac[14,67] >= 0.9
 cov_312_2: frac[3,2] + frac[2,2] + frac[9,2] >= 1
 cov_313_72: frac[12,72] + frac[3,72] + frac[9,72] + frac[0,72]
   + frac[8,72] >= 0.8
 cov_314_56: frac[2,56] + frac[13,56] + frac[6,56] + frac[12,56]
   + frac[11,56] + frac[14,56] + frac[9,56] >= 1
 cov_315_54: frac[13,54] + frac[14,54] + frac[0,54] + frac[5,54]
   + frac[10,54] + frac[12,54] + frac[8,54] >= 1
 cov_316_42: frac[4,42] + frac[13,42] + frac[14,42] + frac[12,42]
   + frac[2,42] + frac[10,42] + frac[9,42] >= 0.8
 cov_317_44: frac[13,44] + frac[12,44] + frac[1,44] + frac[8,44]
   + frac[5,44] + frac[10,44] + frac[14,44] >= 0.8
 cov_318_60: frac[8,60] + frac[1,60] + frac[0,60] >= 1
 cov_319_12: frac[10,12] + frac[11,12] + frac[7,12] + frac[13,12]
   + frac[5,12] >= 0.9
 cov_320_15: frac[11,15] + frac[7,15] + frac[6,15] >= 0.8
 cov_321_17: frac[0,17] + frac[12,17] + frac[14,17] + frac[8,17]
   + frac[7,17] + frac[13,17] + frac[11,17] >= 0.9
 cov_322_20: frac[13,20] + frac[9,20] + frac[12,20] + frac[6,20]
   + frac[11,20] + frac[3,20] + frac[14,20] >= 0.7
 cov_323_37: frac[7,37] + frac[13,37] + frac[5,37] + frac[10,37]
   + frac[11,37] >= 0.8
 cov_324_64: frac[1,64] + frac[10,64] + frac[8,64] + frac[13,64]
   + frac[14,64] + frac[5,64] + frac[12,64] >= 1
 cov_325_12: frac[10,12] + frac[6,12] + frac[11,12] + frac[13,12]
   + frac[5,12] >= 0.9
 cov_326_49: frac[2,49] + frac[3,49] + frac[9,49] >= 0.9
 cov_327_62: frac[9,62] + frac[12,62] + frac[3,62] + frac[8,62]
   + frac[0,62] >= 0.7
 cov_328_51: frac[13,51] + frac[9,51] + frac[10,51] + frac[12,51]
   + frac[14,51] + frac[3,51] + frac[4,51] >= 1
 cov_329_38: frac[14,38] + frac[4,38] + frac[13,38] + frac[9,38]
   + frac[3,38] + frac[10,38] + frac[12,38] >= 0.9
 cov_330_69: frac[12,69] + frac[4,69] + frac[10,69] + frac[2,69]
   + frac[14,69] + frac[13,69] + frac[9,69] >= 0.9
 cov_331_34: frac[7,34] + frac[12,34] + frac[9,34] + frac[11,34]
   + frac[3,34] + frac[14,34] + frac[13,34] >= 0.7
 cov_332_63: frac[13,63] + frac[11,63] + frac[5,63] + frac[10,63]
   + frac[6,63] >= 0.8
 cov_333_20: frac[5,20] + frac[4,20] + frac[10,20] >= 1
 cov_334_10: frac[5,10] + frac[11,10] + frac[10,10] + frac[13,10]
   + frac[7,10] >= 0.9
 cov_335_66: frac[13,66] + frac[8,66] + frac[14,66] + frac[7,66]
   + frac[11,66] + frac[1,66] + frac[12,66] >= 0.9
 cov_336_61: frac[6,61] + frac[11,61] + frac[0,61] + frac[13,61]
   + frac[12,61] + frac[14,61] + frac[8,61] >= 0.9
 cov_337_38: frac[8,38] + frac[0,38] + frac[1,38] >= 0.8
 cov_338_37: frac[13,37] + frac[1,37] + frac[12,37] + frac[8,37]
   + frac[14,37] + frac[6,37] + frac[11,37] >= 1
 cov_339_10: frac[12,10] + frac[10,10] + frac[4,10] + frac[0,10]
   + frac[13,10] + frac[8,10] + frac[14,10] >= 0.7
 cov_340_15: frac[1,15] + frac[12,15] + frac[10,15] + frac[8,15]
   + frac[14,15] + frac[4,15] + frac[13,15] >= 0.8
 cov_341_78: frac[12,78] + frac[0,78] + frac[8,78] + frac[14,78]
   + frac[13,78] + frac[11,78] + frac[7,78] >= 0.8
 cov_342_14: frac[0,14] + frac[8,14] + frac[14,14] + frac[13,14]
   + frac[11,14] + frac[7,14] + frac[12,14] >= 0.7
 cov_343_77: frac[14,77] + frac[1,77] + frac[12,77] + frac[8,77]
   + frac[4,77] + frac[10,77] + frac[13,77] >= 0.7
 cov_344_2: frac[5,2] + frac[10,2] + frac[4,2] >= 0.8
 cov_345_68: frac[11,68] + frac[13,68] + frac[1,68] + frac[12,68]
   + frac[8,68] + frac[14,68] + frac[6,68] >= 0.7
 cov_346_37: frac[13,37] + frac[9,37] + frac[12,37] + frac[4,37]
   + frac[10,37] + frac[2,37] + frac[14,37] >= 0.7
 cov_347_42: frac[13,42] + frac[14,42] + frac[5,42] + frac[12,42]
   + frac[2,42] + frac[10,42] + frac[9,42] >= 1
 cov_348_18: frac[8,18] + frac[1,18] + frac[0,18] >= 0.9
 cov_349_1: frac[14,1] + frac[5,1] + frac[12,1] + frac[10,1] + frac[8,1]
   + frac[0,1] + frac[13,1] >= 0.8
 cov_350_7: frac[12,7] + frac[8,7] + frac[14,7] + frac[0,7] + frac[13,7]
   + frac[11,7] + frac[7,7] >= 0.8
 cov_351_23: frac[6,23] + frac[13,23] + frac[14,23] + frac[11,23]
   + frac[0,23] + frac[12,23] + frac[8,23] >= 0.8
 cov_352_33: frac[7,33] + frac[13,33] + frac[5,33] + frac[11,33]
   + frac[10,33] >= 0.8
 cov_353_9: frac[0,9] + frac[7,9] + frac[8,9] + frac[13,9] + frac[11,9]
   + frac[14,9] + frac[12,9] >= 0.9
 cov_354_43: frac[2,43] + frac[9,43] + frac[3,43] >= 1
 cov_355_1: frac[14,1] + frac[6,1] + frac[11,1] + frac[12,1] + frac[8,1]
   + frac[0,1] + frac[13,1] >= 0.7
 cov_356_3: frac[1,3] + frac[8,3] + frac[0,3] >= 0.7
 cov_357_19: frac[14,19] + frac[11,19] + frac[12,19] + frac[1,19]
   + frac[13,19] + frac[8,19] + frac[6,19] >= 0.8
 cov_358_4: frac[11,4] + frac[7,4] + frac[3,4] + frac[13,4] + frac[9,4]
   + frac[12,4] + frac[14,4] >= 1
 cov_359_64: frac[10,64] + frac[13,64] + frac[6,64] + frac[11,64]
   + frac[4,64] >= 1
 cov_360_35: frac[3,35] + frac[8,35] + frac[0,35] + frac[9,35]
   + frac[12,35] >= 1
 cov_361_7: frac[10,7] + frac[12,7] + frac[2,7] + frac[14,7] + frac[13,7]
   + frac[9,7] + frac[5,7] >= 0.7
 cov_362_2: frac[8,2] + frac[14,2] + frac[13,2] + frac[7,2] + frac[12,2]
   + frac[11,2] + frac[1,2] >= 0.7
 cov_363_4: frac[11,4] + frac[7,4] + frac[13,4] + frac[12,4] + frac[8,4]
   + frac[0,4] + frac[14,4] >= 0.8
 cov_364_41: frac[11,41] + frac[12,41] + frac[14,41] + frac[1,41]
   + frac[8,41] + frac[7,41] + frac[13,41] >= 1
 cov_365_0: frac[9,0] + frac[8,0] + frac[0,0] + frac[12,0] + frac[3,0]
   >= 0.7
 cov_366_27: frac[8,27] + frac[12,27] + frac[14,27] + frac[10,27]
   + frac[4,27] + frac[1,27] + frac[13,27] >= 0.9
 cov_367_65: frac[14,65] + frac[12,65] + frac[1,65] + frac[10,65]
   + frac[8,65] + frac[13,65] + frac[5,65] >= 0.8
 cov_368_11: frac[13,11] + frac[8,11] + frac[14,11] + frac[5,11]
   + frac[1,11] + frac[12,11] + frac[10,11] >= 0.9
 cov_369_15: frac[11,15] + frac[5,15] + frac[10,15] + frac[6,15]
   + frac[13,15] >= 1
 cov_370_63: frac[8,63] + frac[13,63] + frac[5,63] + frac[14,63]
   + frac[12,63] + frac[1,63] + frac[10,63] >= 0.9
 cov_371_43: frac[14,43] + frac[12,43] + frac[2,43] + frac[9,43]
   + frac[10,43] + frac[4,43] + frac[13,43] >= 0.8
 cov_372_61: frac[3,61] + frac[12,61] + frac[9,61] + frac[8,61]
   + frac[1,61] >= 1
 cov_373_50: frac[12,50] + frac[14,50] + frac[9,50] + frac[11,50]
   + frac[3,50] + frac[13,50] + frac[6,50] >= 1
 cov_374_9: frac[9,9] + frac[10,9] + frac[13,9] + frac[14,9] + frac[12,9]
   + frac[4,9] + frac[2,9] >= 0.8
 cov_375_43: frac[14,43] + frac[11,43] + frac[12,43] + frac[13,43]
   + frac[0,43] + frac[6,43] + frac[8,43] >= 1
 cov_376_55: frac[7,55] + frac[0,55] + frac[12,55] + frac[8,55]
   + frac[13,55] + frac[11,55] + frac[14,55] >= 0.7
 cov_377_47: frac[2,47] + frac[9,47] + frac[3,47] >= 0.7
 cov_378_78: frac[12,78] + frac[0,78] + frac[8,78] + frac[14,78]
   + frac[13,78] + frac[11,78] + frac[7,78] >= 0.8
 cov_379_69: frac[4,69] + frac[10,69] + frac[11,69] + frac[6,69]
   + frac[13,69] >= 0.7
 cov_380_27: frac[2,27] + frac[6,27] + frac[11,27] + frac[12,27]
   + frac[9,27] + frac[14,27] + frac[13,27] >= 0.8
 cov_381_49: frac[7,49] + frac[13,49] + frac[11,49] + frac[4,49]
   + frac[10,49] >= 0.9
 cov_382_67: frac[1,67] + frac[10,67] + frac[12,67] + frac[13,67]
   + frac[5,67] + frac[8,67] + frac[14,67] >= 0.9
 cov_383_61: frac[3,61] + frac[12,61] + frac[9,61] + frac[8,61]
   + frac[1,61] >= 0.7
 cov_384_11: frac[13,11] + frac[7,11] + frac[5,11] + frac[11,11]
   + frac[10,11] >= 1
 cov_385_28: frac[1,28] + frac[0,28] + frac[8,28] >= 0.8
 cov_386_27: frac[6,27] + frac[11,27] + frac[10,27] + frac[4,27]
   + frac[13,27] >= 1
 cov_387_60: frac[8,60] + frac[1,60] + frac[0,60] >= 0.7
 cov_388_6: frac[13,6] + frac[9,6] + frac[11,6] + frac[3,6] + frac[12,6]
   + frac[14,6] + frac[6,6] >= 1
 cov_389_11: frac[0,11] + frac[8,11] + frac[12,11] + frac[2,11]
   + frac[9,11] >= 1
 cov_390_0: frac[8,0] + frac[0,0] + frac[14,0] + frac[6,0] + frac[13,0]
   + frac[12,0] + frac[11,0] >= 0.7
 cov_391_40: frac[0,40] + frac[8,40] + frac[7,40] + frac[13,40]
   + frac[12,40] + frac[14,40] + frac[11,40] >= 0.8
 cov_392_18: frac[8,18] + frac[6,18] + frac[1,18] + frac[13,18]
   + frac[14,18] + frac[12,18] + frac[11,18] >= 0.7
 cov_393_47: frac[10,47] + frac[8,47] + frac[14,47] + frac[4,47]
   + frac[13,47] + frac[1,47] + frac[12,47] >= 0.7
 cov_394_48: frac[11,48] + frac[14,48] + frac[8,48] + frac[7,48]
   + frac[13,48] + frac[1,48] + frac[12,48] >= 0.8
 cov_395_19: frac[9,19] + frac[14,19] + frac[11,19] + frac[12,19]
   + frac[13,19] + frac[2,19] + frac[7,19] >= 0.8
 cov_396_60: frac[14,60] + frac[9,60] + frac[6,60] + frac[11,60]
   + frac[2,60] + frac[13,60] + frac[12,60] >= 0.7
 cov_397_22: frac[10,22] + frac[12,22] + frac[8,22] + frac[4,22]
   + frac[1,22] + frac[14,22] + frac[13,22] >= 1
 cov_398_79: frac[9,79] + frac[12,79] + frac[2,79] + frac[13,79]
   + frac[14,79] + frac[11,79] + frac[7,79] >= 0.9
 cov_399_45: frac[4,45] + frac[10,45] + frac[6,45] + frac[11,45]
   + frac[13,45] >= 1
 accuracy_7_3: - 5.09765625 frac[7,3] + mem[7,3] = 0
 accuracy_14_74: - 5.09765625 frac[14,74] + mem[14,74] = 0
 accuracy_11_68: - 5.09765625 frac[11,68] + mem[11,68] = 0
 accuracy_12_78: - 5.09765625 frac[12,78] + mem[12,78] = 0
 accuracy_3_35: - 5.09765625 frac[3,35] + mem[3,35] = 0
 accuracy_4_36: - 5.09765625 frac[4,36] + mem[4,36] = 0
 accuracy_8_63: - 5.09765625 frac[8,63] + mem[8,63] = 0
 accuracy_7_25: - 5.09765625 frac[7,25] + mem[7,25] = 0
 accuracy_13_70: - 5.09765625 frac[13,70] + mem[13,70] = 0
 accuracy_5_57: - 5.09765625 frac[5,57] + mem[5,57] = 0
 accuracy_9_62: - 5.09765625 frac[9,62] + mem[9,62] = 0
 accuracy_4_66: - 5.09765625 frac[4,66] + mem[4,66] = 0
 accuracy_10_61: - 5.09765625 frac[10,61] + mem[10,61] = 0
 accuracy_2_78: - 5.09765625 frac[2,78] + mem[2,78] = 0
 accuracy_8_5: - 5.09765625 frac[8,5] + mem[8,5] = 0
 accuracy_7_55: - 5.09765625 frac[7,55] + mem[7,55] = 0
 accuracy_13_32: - 5.09765625 frac[13,32] + mem[13,32] = 0
 accuracy_1_64: - 5.09765625 frac[1,64] + mem[1,64] = 0
 accuracy_11_48: - 5.09765625 frac[11,48] + mem[11,48] = 0
 accuracy_10_64: - 5.09765625 frac[10,64] + mem[10,64] = 0
 accuracy_9_0: - 5.09765625 frac[9,0] + mem[9,0] = 0
 accuracy_14_31: - 5.09765625 frac[14,31] + mem[14,31] = 0
 accuracy_12_59: - 5.09765625 frac[12,59] + mem[12,59] = 0
 accuracy_10_7: - 5.09765625 frac[10,7] + mem[10,7] = 0
 accuracy_13_58: - 5.09765625 frac[13,58] + mem[13,58] = 0
 accuracy_11_22: - 5.09765625 frac[11,22] + mem[11,22] = 0
 accuracy_0_17: - 5.09765625 frac[0,17] + mem[0,17] = 0
 accuracy_7_68: - 5.09765625 frac[7,68] + mem[7,68] = 0
 accuracy_14_1: - 5.09765625 frac[14,1] + mem[14,1] = 0
 accuracy_12_17: - 5.09765625 frac[12,17] + mem[12,17] = 0
 accuracy_1_28: - 5.09765625 frac[1,28] + mem[1,28] = 0
 accuracy_13_20: - 5.09765625 frac[13,20] + mem[13,20] = 0
 accuracy_2_27: - 5.09765625 frac[2,27] + mem[2,27] = 0
 accuracy_0_55: - 5.09765625 frac[0,55] + mem[0,55] = 0
 accuracy_14_43: - 5.09765625 frac[14,43] + mem[14,43] = 0
 accuracy_3_2: - 5.09765625 frac[3,2] + mem[3,2] = 0
 accuracy_10_75: - 5.09765625 frac[10,75] + mem[10,75] = 0
 accuracy_1_54: - 5.09765625 frac[1,54] + mem[1,54] = 0
 accuracy_4_5: - 5.09765625 frac[4,5] + mem[4,5] = 0
 accuracy_2_53: - 5.09765625 frac[2,53] + mem[2,53] = 0
 accuracy_14_77: - 5.09765625 frac[14,77] + mem[14,77] = 0
 accuracy_12_69: - 5.09765625 frac[12,69] + mem[12,69] = 0
 accuracy_6_23: - 5.09765625 frac[6,23] + mem[6,23] = 0
 accuracy_4_35: - 5.09765625 frac[4,35] + mem[4,35] = 0
 accuracy_2_47: - 5.09765625 frac[2,47] + mem[2,47] = 0
 accuracy_8_38: - 5.09765625 frac[8,38] + mem[8,38] = 0
 accuracy_0_67: - 5.09765625 frac[0,67] + mem[0,67] = 0
 accuracy_7_22: - 5.09765625 frac[7,22] + mem[7,22] = 0
 accuracy_5_34: - 5.09765625 frac[5,34] + mem[5,34] = 0
 accuracy_9_39: - 5.09765625 frac[9,39] + mem[9,39] = 0
 accuracy_10_36: - 5.09765625 frac[10,36] + mem[10,36] = 0
 accuracy_2_73: - 5.09765625 frac[2,73] + mem[2,73] = 0
 accuracy_8_12: - 5.09765625 frac[8,12] + mem[8,12] = 0
 accuracy_7_44: - 5.09765625 frac[7,44] + mem[7,44] = 0
 accuracy_11_41: - 5.09765625 frac[11,41] + mem[11,41] = 0
 accuracy_9_9: - 5.09765625 frac[9,9] + mem[9,9] = 0
 accuracy_6_67: - 5.09765625 frac[6,67] + mem[6,67] = 0
 accuracy_12_50: - 5.09765625 frac[12,50] + mem[12,50] = 0
 accuracy_10_14: - 5.09765625 frac[10,14] + mem[10,14] = 0
 accuracy_8_18: - 5.09765625 frac[8,18] + mem[8,18] = 0
 accuracy_13_51: - 5.09765625 frac[13,51] + mem[13,51] = 0
 accuracy_11_15: - 5.09765625 frac[11,15] + mem[11,15] = 0
 accuracy_9_19: - 5.09765625 frac[9,19] + mem[9,19] = 0
 accuracy_14_8: - 5.09765625 frac[14,8] + mem[14,8] = 0
 accuracy_12_8: - 5.09765625 frac[12,8] + mem[12,8] = 0
 accuracy_13_13: - 5.09765625 frac[13,13] + mem[13,13] = 0
 accuracy_2_18: - 5.09765625 frac[2,18] + mem[2,18] = 0
 accuracy_0_14: - 5.09765625 frac[0,14] + mem[0,14] = 0
 accuracy_14_50: - 5.09765625 frac[14,50] + mem[14,50] = 0
 accuracy_3_11: - 5.09765625 frac[3,11] + mem[3,11] = 0
 accuracy_1_15: - 5.09765625 frac[1,15] + mem[1,15] = 0
 accuracy_4_12: - 5.09765625 frac[4,12] + mem[4,12] = 0
 accuracy_2_12: - 5.09765625 frac[2,12] + mem[2,12] = 0
 accuracy_5_1: - 5.09765625 frac[5,1] + mem[5,1] = 0
 accuracy_9_70: - 5.09765625 frac[9,70] + mem[9,70] = 0
 accuracy_1_33: - 5.09765625 frac[1,33] + mem[1,33] = 0
 accuracy_6_14: - 5.09765625 frac[6,14] + mem[6,14] = 0
 accuracy_4_42: - 5.09765625 frac[4,42] + mem[4,42] = 0
 accuracy_2_38: - 5.09765625 frac[2,38] + mem[2,38] = 0
 accuracy_7_15: - 5.09765625 frac[7,15] + mem[7,15] = 0
 accuracy_5_43: - 5.09765625 frac[5,43] + mem[5,43] = 0
 accuracy_3_55: - 5.09765625 frac[3,55] + mem[3,55] = 0
 accuracy_9_40: - 5.09765625 frac[9,40] + mem[9,40] = 0
 accuracy_6_32: - 5.09765625 frac[6,32] + mem[6,32] = 0
 accuracy_10_47: - 5.09765625 frac[10,47] + mem[10,47] = 0
 accuracy_2_64: - 5.09765625 frac[2,64] + mem[2,64] = 0
 accuracy_7_37: - 5.09765625 frac[7,37] + mem[7,37] = 0
 accuracy_11_46: - 5.09765625 frac[11,46] + mem[11,46] = 0
 accuracy_9_50: - 5.09765625 frac[9,50] + mem[9,50] = 0
 accuracy_6_58: - 5.09765625 frac[6,58] + mem[6,58] = 0
 accuracy_12_41: - 5.09765625 frac[12,41] + mem[12,41] = 0
 accuracy_3_68: - 5.09765625 frac[3,68] + mem[3,68] = 0
 accuracy_10_9: - 5.09765625 frac[10,9] + mem[10,9] = 0
 accuracy_8_25: - 5.09765625 frac[8,25] + mem[8,25] = 0
 accuracy_13_44: - 5.09765625 frac[13,44] + mem[13,44] = 0
 accuracy_11_4: - 5.09765625 frac[11,4] + mem[11,4] = 0
 accuracy_9_20: - 5.09765625 frac[9,20] + mem[9,20] = 0
 accuracy_14_19: - 5.09765625 frac[14,19] + mem[14,19] = 0
 accuracy_5_70: - 5.09765625 frac[5,70] + mem[5,70] = 0
 accuracy_12_7: - 5.09765625 frac[12,7] + mem[12,7] = 0
 accuracy_10_19: - 5.09765625 frac[10,19] + mem[10,19] = 0
 accuracy_13_6: - 5.09765625 frac[13,6] + mem[13,6] = 0
 accuracy_11_26: - 5.09765625 frac[11,26] + mem[11,26] = 0
 accuracy_0_5: - 5.09765625 frac[0,5] + mem[0,5] = 0
 accuracy_14_53: - 5.09765625 frac[14,53] + mem[14,53] = 0
 accuracy_12_29: - 5.09765625 frac[12,29] + mem[12,29] = 0
 accuracy_1_0: - 5.09765625 frac[1,0] + mem[1,0] = 0
 accuracy_4_11: - 5.09765625 frac[4,11] + mem[4,11] = 0
 accuracy_2_7: - 5.09765625 frac[2,7] + mem[2,7] = 0
 accuracy_0_59: - 5.09765625 frac[0,59] + mem[0,59] = 0
 accuracy_5_10: - 5.09765625 frac[5,10] + mem[5,10] = 0
 accuracy_9_79: - 5.09765625 frac[9,79] + mem[9,79] = 0
 accuracy_10_71: - 5.09765625 frac[10,71] + mem[10,71] = 0
 accuracy_1_58: - 5.09765625 frac[1,58] + mem[1,58] = 0
 accuracy_6_1: - 5.09765625 frac[6,1] + mem[6,1] = 0
 accuracy_4_17: - 5.09765625 frac[4,17] + mem[4,17] = 0
 accuracy_2_33: - 5.09765625 frac[2,33] + mem[2,33] = 0
 accuracy_7_4: - 5.09765625 frac[7,4] + mem[7,4] = 0
 accuracy_14_65: - 5.09765625 frac[14,65] + mem[14,65] = 0
 accuracy_11_65: - 5.09765625 frac[11,65] + mem[11,65] = 0
 accuracy_3_60: - 5.09765625 frac[3,60] + mem[3,60] = 0
 accuracy_5_20: - 5.09765625 frac[5,20] + mem[5,20] = 0
 accuracy_6_27: - 5.09765625 frac[6,27] + mem[6,27] = 0
 accuracy_8_58: - 5.09765625 frac[8,58] + mem[8,58] = 0
 accuracy_7_34: - 5.09765625 frac[7,34] + mem[7,34] = 0
 accuracy_13_75: - 5.09765625 frac[13,75] + mem[13,75] = 0
 accuracy_5_62: - 5.09765625 frac[5,62] + mem[5,62] = 0
 accuracy_9_59: - 5.09765625 frac[9,59] + mem[9,59] = 0
 accuracy_6_61: - 5.09765625 frac[6,61] + mem[6,61] = 0
 accuracy_12_32: - 5.09765625 frac[12,32] + mem[12,32] = 0
 accuracy_4_69: - 5.09765625 frac[4,69] + mem[4,69] = 0
 accuracy_10_48: - 5.09765625 frac[10,48] + mem[10,48] = 0
 accuracy_1_77: - 5.09765625 frac[1,77] + mem[1,77] = 0
 accuracy_8_0: - 5.09765625 frac[8,0] + mem[8,0] = 0
 accuracy_7_56: - 5.09765625 frac[7,56] + mem[7,56] = 0
 accuracy_13_37: - 5.09765625 frac[13,37] + mem[13,37] = 0
 accuracy_11_61: - 5.09765625 frac[11,61] + mem[11,61] = 0
 accuracy_9_29: - 5.09765625 frac[9,29] + mem[9,29] = 0
 accuracy_14_26: - 5.09765625 frac[14,26] + mem[14,26] = 0
 accuracy_12_62: - 5.09765625 frac[12,62] + mem[12,62] = 0
 accuracy_10_26: - 5.09765625 frac[10,26] + mem[10,26] = 0
 accuracy_13_63: - 5.09765625 frac[13,63] + mem[13,63] = 0
 accuracy_11_19: - 5.09765625 frac[11,19] + mem[11,19] = 0
 accuracy_7_73: - 5.09765625 frac[7,73] + mem[7,73] = 0
 accuracy_0_28: - 5.09765625 frac[0,28] + mem[0,28] = 0
 accuracy_14_60: - 5.09765625 frac[14,60] + mem[14,60] = 0
 accuracy_12_20: - 5.09765625 frac[12,20] + mem[12,20] = 0
 accuracy_1_25: - 5.09765625 frac[1,25] + mem[1,25] = 0
 accuracy_13_25: - 5.09765625 frac[13,25] + mem[13,25] = 0
 accuracy_14_38: - 5.09765625 frac[14,38] + mem[14,38] = 0
 accuracy_10_78: - 5.09765625 frac[10,78] + mem[10,78] = 0
 accuracy_4_24: - 5.09765625 frac[4,24] + mem[4,24] = 0
 accuracy_2_56: - 5.09765625 frac[2,56] + mem[2,56] = 0
 accuracy_0_40: - 5.09765625 frac[0,40] + mem[0,40] = 0
 accuracy_5_29: - 5.09765625 frac[5,29] + mem[5,29] = 0
 accuracy_11_70: - 5.09765625 frac[11,70] + mem[11,70] = 0
 accuracy_3_37: - 5.09765625 frac[3,37] + mem[3,37] = 0
 accuracy_12_72: - 5.09765625 frac[12,72] + mem[12,72] = 0
 accuracy_6_18: - 5.09765625 frac[6,18] + mem[6,18] = 0
 accuracy_4_38: - 5.09765625 frac[4,38] + mem[4,38] = 0
 accuracy_8_33: - 5.09765625 frac[8,33] + mem[8,33] = 0
 accuracy_0_78: - 5.09765625 frac[0,78] + mem[0,78] = 0
 accuracy_7_27: - 5.09765625 frac[7,27] + mem[7,27] = 0
 accuracy_5_39: - 5.09765625 frac[5,39] + mem[5,39] = 0
 accuracy_13_68: - 5.09765625 frac[13,68] + mem[13,68] = 0
 accuracy_9_60: - 5.09765625 frac[9,60] + mem[9,60] = 0
 accuracy_4_76: - 5.09765625 frac[4,76] + mem[4,76] = 0
 accuracy_1_70: - 5.09765625 frac[1,70] + mem[1,70] = 0
 accuracy_10_59: - 5.09765625 frac[10,59] + mem[10,59] = 0
 accuracy_8_7: - 5.09765625 frac[8,7] + mem[8,7] = 0
 accuracy_2_76: - 5.09765625 frac[2,76] + mem[2,76] = 0
 accuracy_7_49: - 5.09765625 frac[7,49] + mem[7,49] = 0
 accuracy_11_50: - 5.09765625 frac[11,50] + mem[11,50] = 0
 accuracy_9_6: - 5.09765625 frac[9,6] + mem[9,6] = 0
 accuracy_14_29: - 5.09765625 frac[14,29] + mem[14,29] = 0
 accuracy_6_78: - 5.09765625 frac[6,78] + mem[6,78] = 0
 accuracy_12_53: - 5.09765625 frac[12,53] + mem[12,53] = 0
 accuracy_10_5: - 5.09765625 frac[10,5] + mem[10,5] = 0
 accuracy_13_56: - 5.09765625 frac[13,56] + mem[13,56] = 0
 accuracy_11_8: - 5.09765625 frac[11,8] + mem[11,8] = 0
 accuracy_0_19: - 5.09765625 frac[0,19] + mem[0,19] = 0
 accuracy_7_70: - 5.09765625 frac[7,70] + mem[7,70] = 0
 accuracy_14_7: - 5.09765625 frac[14,7] + mem[14,7] = 0
 accuracy_12_19: - 5.09765625 frac[12,19] + mem[12,19] = 0
 accuracy_1_18: - 5.09765625 frac[1,18] + mem[1,18] = 0
 accuracy_13_18: - 5.09765625 frac[13,18] + mem[13,18] = 0
 accuracy_2_25: - 5.09765625 frac[2,25] + mem[2,25] = 0
 accuracy_0_9: - 5.09765625 frac[0,9] + mem[0,9] = 0
 accuracy_14_41: - 5.09765625 frac[14,41] + mem[14,41] = 0
 accuracy_3_4: - 5.09765625 frac[3,4] + mem[3,4] = 0
 accuracy_1_52: - 5.09765625 frac[1,52] + mem[1,52] = 0
 accuracy_4_7: - 5.09765625 frac[4,7] + mem[4,7] = 0
 accuracy_5_6: - 5.09765625 frac[5,6] + mem[5,6] = 0
 accuracy_12_71: - 5.09765625 frac[12,71] + mem[12,71] = 0
 accuracy_9_67: - 5.09765625 frac[9,67] + mem[9,67] = 0
 accuracy_1_46: - 5.09765625 frac[1,46] + mem[1,46] = 0
 accuracy_3_42: - 5.09765625 frac[3,42] + mem[3,42] = 0
 accuracy_4_45: - 5.09765625 frac[4,45] + mem[4,45] = 0
 accuracy_8_40: - 5.09765625 frac[8,40] + mem[8,40] = 0
 accuracy_5_32: - 5.09765625 frac[5,32] + mem[5,32] = 0
 accuracy_9_37: - 5.09765625 frac[9,37] + mem[9,37] = 0
 accuracy_4_75: - 5.09765625 frac[4,75] + mem[4,75] = 0
 accuracy_10_34: - 5.09765625 frac[10,34] + mem[10,34] = 0
 accuracy_8_14: - 5.09765625 frac[8,14] + mem[8,14] = 0
 accuracy_7_46: - 5.09765625 frac[7,46] + mem[7,46] = 0
 accuracy_11_43: - 5.09765625 frac[11,43] + mem[11,43] = 0
 accuracy_9_15: - 5.09765625 frac[9,15] + mem[9,15] = 0
 accuracy_6_65: - 5.09765625 frac[6,65] + mem[6,65] = 0
 accuracy_3_65: - 5.09765625 frac[3,65] + mem[3,65] = 0
 accuracy_12_44: - 5.09765625 frac[12,44] + mem[12,44] = 0
 accuracy_10_12: - 5.09765625 frac[10,12] + mem[10,12] = 0
 accuracy_8_20: - 5.09765625 frac[8,20] + mem[8,20] = 0
 accuracy_13_49: - 5.09765625 frac[13,49] + mem[13,49] = 0
 accuracy_11_1: - 5.09765625 frac[11,1] + mem[11,1] = 0
 accuracy_14_14: - 5.09765625 frac[14,14] + mem[14,14] = 0
 accuracy_12_10: - 5.09765625 frac[12,10] + mem[12,10] = 0
 accuracy_10_22: - 5.09765625 frac[10,22] + mem[10,22] = 0
 accuracy_13_11: - 5.09765625 frac[13,11] + mem[13,11] = 0
 accuracy_2_16: - 5.09765625 frac[2,16] + mem[2,16] = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 accuracy_14_48: - 5.09765625 frac[14,48] + mem[14,48] = 0
 accuracy_3_13: - 5.09765625 frac[3,13] + mem[3,13] = 0
 accuracy_1_13: - 5.09765625 frac[1,13] + mem[1,13] = 0
 accuracy_8_64: - 5.09765625 frac[8,64] + mem[8,64] = 0
 accuracy_4_14: - 5.09765625 frac[4,14] + mem[4,14] = 0
 accuracy_0_38: - 5.09765625 frac[0,38] + mem[0,38] = 0
 accuracy_5_15: - 5.09765625 frac[5,15] + mem[5,15] = 0
 accuracy_9_68: - 5.09765625 frac[9,68] + mem[9,68] = 0
 accuracy_1_39: - 5.09765625 frac[1,39] + mem[1,39] = 0
 accuracy_6_12: - 5.09765625 frac[6,12] + mem[6,12] = 0
 accuracy_4_20: - 5.09765625 frac[4,20] + mem[4,20] = 0
 accuracy_2_36: - 5.09765625 frac[2,36] + mem[2,36] = 0
 accuracy_8_47: - 5.09765625 frac[8,47] + mem[8,47] = 0
 accuracy_7_9: - 5.09765625 frac[7,9] + mem[7,9] = 0
 accuracy_11_74: - 5.09765625 frac[11,74] + mem[11,74] = 0
 accuracy_3_57: - 5.09765625 frac[3,57] + mem[3,57] = 0
 accuracy_9_46: - 5.09765625 frac[9,46] + mem[9,46] = 0
 accuracy_10_45: - 5.09765625 frac[10,45] + mem[10,45] = 0
 accuracy_8_53: - 5.09765625 frac[8,53] + mem[8,53] = 0
 accuracy_11_32: - 5.09765625 frac[11,32] + mem[11,32] = 0
 accuracy_6_56: - 5.09765625 frac[6,56] + mem[6,56] = 0
 accuracy_12_43: - 5.09765625 frac[12,43] + mem[12,43] = 0
 accuracy_3_70: - 5.09765625 frac[3,70] + mem[3,70] = 0
 accuracy_10_55: - 5.09765625 frac[10,55] + mem[10,55] = 0
 accuracy_1_74: - 5.09765625 frac[1,74] + mem[1,74] = 0
 accuracy_8_27: - 5.09765625 frac[8,27] + mem[8,27] = 0
 accuracy_13_42: - 5.09765625 frac[13,42] + mem[13,42] = 0
 accuracy_11_6: - 5.09765625 frac[11,6] + mem[11,6] = 0
 accuracy_9_26: - 5.09765625 frac[9,26] + mem[9,26] = 0
 accuracy_14_17: - 5.09765625 frac[14,17] + mem[14,17] = 0
 accuracy_12_1: - 5.09765625 frac[12,1] + mem[12,1] = 0
 accuracy_10_17: - 5.09765625 frac[10,17] + mem[10,17] = 0
 accuracy_13_4: - 5.09765625 frac[13,4] + mem[13,4] = 0
 accuracy_11_28: - 5.09765625 frac[11,28] + mem[11,28] = 0
 accuracy_0_7: - 5.09765625 frac[0,7] + mem[0,7] = 0
 accuracy_14_59: - 5.09765625 frac[14,59] + mem[14,59] = 0
 accuracy_12_31: - 5.09765625 frac[12,31] + mem[12,31] = 0
 accuracy_1_6: - 5.09765625 frac[1,6] + mem[1,6] = 0
 accuracy_8_71: - 5.09765625 frac[8,71] + mem[8,71] = 0
 accuracy_13_30: - 5.09765625 frac[13,30] + mem[13,30] = 0
 accuracy_0_61: - 5.09765625 frac[0,61] + mem[0,61] = 0
 accuracy_5_8: - 5.09765625 frac[5,8] + mem[5,8] = 0
 accuracy_3_24: - 5.09765625 frac[3,24] + mem[3,24] = 0
 accuracy_9_77: - 5.09765625 frac[9,77] + mem[9,77] = 0
 accuracy_10_69: - 5.09765625 frac[10,69] + mem[10,69] = 0
 accuracy_1_56: - 5.09765625 frac[1,56] + mem[1,56] = 0
 accuracy_6_7: - 5.09765625 frac[6,7] + mem[6,7] = 0
 accuracy_4_19: - 5.09765625 frac[4,19] + mem[4,19] = 0
 accuracy_2_63: - 5.09765625 frac[2,63] + mem[2,63] = 0
 accuracy_7_6: - 5.09765625 frac[7,6] + mem[7,6] = 0
 accuracy_14_71: - 5.09765625 frac[14,71] + mem[14,71] = 0
 accuracy_11_67: - 5.09765625 frac[11,67] + mem[11,67] = 0
 accuracy_3_62: - 5.09765625 frac[3,62] + mem[3,62] = 0
 accuracy_4_57: - 5.09765625 frac[4,57] + mem[4,57] = 0
 accuracy_8_60: - 5.09765625 frac[8,60] + mem[8,60] = 0
 accuracy_7_28: - 5.09765625 frac[7,28] + mem[7,28] = 0
 accuracy_13_73: - 5.09765625 frac[13,73] + mem[13,73] = 0
 accuracy_0_73: - 5.09765625 frac[0,73] + mem[0,73] = 0
 accuracy_9_57: - 5.09765625 frac[9,57] + mem[9,57] = 0
 accuracy_12_34: - 5.09765625 frac[12,34] + mem[12,34] = 0
 accuracy_3_79: - 5.09765625 frac[3,79] + mem[3,79] = 0
 accuracy_10_62: - 5.09765625 frac[10,62] + mem[10,62] = 0
 accuracy_4_71: - 5.09765625 frac[4,71] + mem[4,71] = 0
 accuracy_8_2: - 5.09765625 frac[8,2] + mem[8,2] = 0
 accuracy_1_67: - 5.09765625 frac[1,67] + mem[1,67] = 0
 accuracy_7_58: - 5.09765625 frac[7,58] + mem[7,58] = 0
 accuracy_13_35: - 5.09765625 frac[13,35] + mem[13,35] = 0
 accuracy_11_63: - 5.09765625 frac[11,63] + mem[11,63] = 0
 accuracy_9_3: - 5.09765625 frac[9,3] + mem[9,3] = 0
 accuracy_14_24: - 5.09765625 frac[14,24] + mem[14,24] = 0
 accuracy_5_77: - 5.09765625 frac[5,77] + mem[5,77] = 0
 accuracy_12_56: - 5.09765625 frac[12,56] + mem[12,56] = 0
 accuracy_10_24: - 5.09765625 frac[10,24] + mem[10,24] = 0
 accuracy_13_61: - 5.09765625 frac[13,61] + mem[13,61] = 0
 accuracy_0_30: - 5.09765625 frac[0,30] + mem[0,30] = 0
 accuracy_7_75: - 5.09765625 frac[7,75] + mem[7,75] = 0
 accuracy_14_2: - 5.09765625 frac[14,2] + mem[14,2] = 0
 accuracy_12_22: - 5.09765625 frac[12,22] + mem[12,22] = 0
 accuracy_8_78: - 5.09765625 frac[8,78] + mem[8,78] = 0
 accuracy_13_23: - 5.09765625 frac[13,23] + mem[13,23] = 0
 accuracy_2_28: - 5.09765625 frac[2,28] + mem[2,28] = 0
 accuracy_14_36: - 5.09765625 frac[14,36] + mem[14,36] = 0
 accuracy_3_1: - 5.09765625 frac[3,1] + mem[3,1] = 0
 accuracy_10_76: - 5.09765625 frac[10,76] + mem[10,76] = 0
 accuracy_4_26: - 5.09765625 frac[4,26] + mem[4,26] = 0
 accuracy_2_54: - 5.09765625 frac[2,54] + mem[2,54] = 0
 accuracy_0_42: - 5.09765625 frac[0,42] + mem[0,42] = 0
 accuracy_14_78: - 5.09765625 frac[14,78] + mem[14,78] = 0
 accuracy_5_27: - 5.09765625 frac[5,27] + mem[5,27] = 0
 accuracy_12_74: - 5.09765625 frac[12,74] + mem[12,74] = 0
 accuracy_1_43: - 5.09765625 frac[1,43] + mem[1,43] = 0
 accuracy_6_16: - 5.09765625 frac[6,16] + mem[6,16] = 0
 accuracy_8_35: - 5.09765625 frac[8,35] + mem[8,35] = 0
 accuracy_13_66: - 5.09765625 frac[13,66] + mem[13,66] = 0
 accuracy_5_37: - 5.09765625 frac[5,37] + mem[5,37] = 0
 accuracy_9_34: - 5.09765625 frac[9,34] + mem[9,34] = 0
 accuracy_4_78: - 5.09765625 frac[4,78] + mem[4,78] = 0
 accuracy_10_57: - 5.09765625 frac[10,57] + mem[10,57] = 0
 accuracy_1_68: - 5.09765625 frac[1,68] + mem[1,68] = 0
 accuracy_8_9: - 5.09765625 frac[8,9] + mem[8,9] = 0
 accuracy_9_4: - 5.09765625 frac[9,4] + mem[9,4] = 0
 accuracy_12_55: - 5.09765625 frac[12,55] + mem[12,55] = 0
 accuracy_10_3: - 5.09765625 frac[10,3] + mem[10,3] = 0
 accuracy_13_54: - 5.09765625 frac[13,54] + mem[13,54] = 0
 accuracy_11_10: - 5.09765625 frac[11,10] + mem[11,10] = 0
 accuracy_7_64: - 5.09765625 frac[7,64] + mem[7,64] = 0
 accuracy_14_5: - 5.09765625 frac[14,5] + mem[14,5] = 0
 accuracy_12_13: - 5.09765625 frac[12,13] + mem[12,13] = 0
 accuracy_13_16: - 5.09765625 frac[13,16] + mem[13,16] = 0
 accuracy_0_11: - 5.09765625 frac[0,11] + mem[0,11] = 0
 accuracy_14_47: - 5.09765625 frac[14,47] + mem[14,47] = 0
 accuracy_3_6: - 5.09765625 frac[3,6] + mem[3,6] = 0
 accuracy_4_1: - 5.09765625 frac[4,1] + mem[4,1] = 0
 accuracy_2_49: - 5.09765625 frac[2,49] + mem[2,49] = 0
 accuracy_0_33: - 5.09765625 frac[0,33] + mem[0,33] = 0
 accuracy_12_65: - 5.09765625 frac[12,65] + mem[12,65] = 0
 accuracy_9_65: - 5.09765625 frac[9,65] + mem[9,65] = 0
 accuracy_3_44: - 5.09765625 frac[3,44] + mem[3,44] = 0
 accuracy_1_44: - 5.09765625 frac[1,44] + mem[1,44] = 0
 accuracy_4_47: - 5.09765625 frac[4,47] + mem[4,47] = 0
 accuracy_2_43: - 5.09765625 frac[2,43] + mem[2,43] = 0
 accuracy_8_42: - 5.09765625 frac[8,42] + mem[8,42] = 0
 accuracy_0_71: - 5.09765625 frac[0,71] + mem[0,71] = 0
 accuracy_5_46: - 5.09765625 frac[5,46] + mem[5,46] = 0
 accuracy_3_50: - 5.09765625 frac[3,50] + mem[3,50] = 0
 accuracy_9_43: - 5.09765625 frac[9,43] + mem[9,43] = 0
 accuracy_6_45: - 5.09765625 frac[6,45] + mem[6,45] = 0
 accuracy_4_53: - 5.09765625 frac[4,53] + mem[4,53] = 0
 accuracy_10_32: - 5.09765625 frac[10,32] + mem[10,32] = 0
 accuracy_2_69: - 5.09765625 frac[2,69] + mem[2,69] = 0
 accuracy_8_48: - 5.09765625 frac[8,48] + mem[8,48] = 0
 accuracy_7_40: - 5.09765625 frac[7,40] + mem[7,40] = 0
 accuracy_11_45: - 5.09765625 frac[11,45] + mem[11,45] = 0
 accuracy_9_13: - 5.09765625 frac[9,13] + mem[9,13] = 0
 accuracy_6_71: - 5.09765625 frac[6,71] + mem[6,71] = 0
 accuracy_12_46: - 5.09765625 frac[12,46] + mem[12,46] = 0
 accuracy_3_67: - 5.09765625 frac[3,67] + mem[3,67] = 0
 accuracy_10_10: - 5.09765625 frac[10,10] + mem[10,10] = 0
 accuracy_8_22: - 5.09765625 frac[8,22] + mem[8,22] = 0
 accuracy_13_47: - 5.09765625 frac[13,47] + mem[13,47] = 0
 accuracy_11_3: - 5.09765625 frac[11,3] + mem[11,3] = 0
 accuracy_14_12: - 5.09765625 frac[14,12] + mem[14,12] = 0
 accuracy_12_4: - 5.09765625 frac[12,4] + mem[12,4] = 0
 accuracy_10_20: - 5.09765625 frac[10,20] + mem[10,20] = 0
 accuracy_13_9: - 5.09765625 frac[13,9] + mem[13,9] = 0
 accuracy_11_25: - 5.09765625 frac[11,25] + mem[11,25] = 0
 accuracy_0_2: - 5.09765625 frac[0,2] + mem[0,2] = 0
 accuracy_14_54: - 5.09765625 frac[14,54] + mem[14,54] = 0
 accuracy_3_15: - 5.09765625 frac[3,15] + mem[3,15] = 0
 accuracy_1_3: - 5.09765625 frac[1,3] + mem[1,3] = 0
 accuracy_8_66: - 5.09765625 frac[8,66] + mem[8,66] = 0
 accuracy_2_8: - 5.09765625 frac[2,8] + mem[2,8] = 0
 accuracy_0_56: - 5.09765625 frac[0,56] + mem[0,56] = 0
 accuracy_5_13: - 5.09765625 frac[5,13] + mem[5,13] = 0
 accuracy_9_74: - 5.09765625 frac[9,74] + mem[9,74] = 0
 accuracy_1_37: - 5.09765625 frac[1,37] + mem[1,37] = 0
 accuracy_6_2: - 5.09765625 frac[6,2] + mem[6,2] = 0
 accuracy_4_22: - 5.09765625 frac[4,22] + mem[4,22] = 0
 accuracy_2_34: - 5.09765625 frac[2,34] + mem[2,34] = 0
 accuracy_7_11: - 5.09765625 frac[7,11] + mem[7,11] = 0
 accuracy_14_66: - 5.09765625 frac[14,66] + mem[14,66] = 0
 accuracy_11_76: - 5.09765625 frac[11,76] + mem[11,76] = 0
 accuracy_3_59: - 5.09765625 frac[3,59] + mem[3,59] = 0
 accuracy_9_44: - 5.09765625 frac[9,44] + mem[9,44] = 0
 accuracy_6_36: - 5.09765625 frac[6,36] + mem[6,36] = 0
 accuracy_10_43: - 5.09765625 frac[10,43] + mem[10,43] = 0
 accuracy_8_55: - 5.09765625 frac[8,55] + mem[8,55] = 0
 accuracy_7_33: - 5.09765625 frac[7,33] + mem[7,33] = 0
 accuracy_13_78: - 5.09765625 frac[13,78] + mem[13,78] = 0
 accuracy_11_34: - 5.09765625 frac[11,34] + mem[11,34] = 0
 accuracy_9_54: - 5.09765625 frac[9,54] + mem[9,54] = 0
 accuracy_12_37: - 5.09765625 frac[12,37] + mem[12,37] = 0
 accuracy_3_72: - 5.09765625 frac[3,72] + mem[3,72] = 0
 accuracy_10_53: - 5.09765625 frac[10,53] + mem[10,53] = 0
 accuracy_1_72: - 5.09765625 frac[1,72] + mem[1,72] = 0
 accuracy_7_63: - 5.09765625 frac[7,63] + mem[7,63] = 0
 accuracy_13_40: - 5.09765625 frac[13,40] + mem[13,40] = 0
 accuracy_11_56: - 5.09765625 frac[11,56] + mem[11,56] = 0
 accuracy_9_24: - 5.09765625 frac[9,24] + mem[9,24] = 0
 accuracy_5_66: - 5.09765625 frac[5,66] + mem[5,66] = 0
 accuracy_14_23: - 5.09765625 frac[14,23] + mem[14,23] = 0
 accuracy_12_3: - 5.09765625 frac[12,3] + mem[12,3] = 0
 accuracy_13_2: - 5.09765625 frac[13,2] + mem[13,2] = 0
 accuracy_11_30: - 5.09765625 frac[11,30] + mem[11,30] = 0
 accuracy_0_25: - 5.09765625 frac[0,25] + mem[0,25] = 0
 accuracy_7_76: - 5.09765625 frac[7,76] + mem[7,76] = 0
 accuracy_14_57: - 5.09765625 frac[14,57] + mem[14,57] = 0
 accuracy_12_25: - 5.09765625 frac[12,25] + mem[12,25] = 0
 accuracy_8_73: - 5.09765625 frac[8,73] + mem[8,73] = 0
 accuracy_13_28: - 5.09765625 frac[13,28] + mem[13,28] = 0
 accuracy_2_3: - 5.09765625 frac[2,3] + mem[2,3] = 0
 accuracy_0_63: - 5.09765625 frac[0,63] + mem[0,63] = 0
 accuracy_3_26: - 5.09765625 frac[3,26] + mem[3,26] = 0
 accuracy_10_67: - 5.09765625 frac[10,67] + mem[10,67] = 0
 accuracy_6_5: - 5.09765625 frac[6,5] + mem[6,5] = 0
 accuracy_4_29: - 5.09765625 frac[4,29] + mem[4,29] = 0
 accuracy_7_0: - 5.09765625 frac[7,0] + mem[7,0] = 0
 accuracy_5_16: - 5.09765625 frac[5,16] + mem[5,16] = 0
 accuracy_11_69: - 5.09765625 frac[11,69] + mem[11,69] = 0
 accuracy_12_77: - 5.09765625 frac[12,77] + mem[12,77] = 0
 accuracy_14_69: - 5.09765625 frac[14,69] + mem[14,69] = 0
 accuracy_4_59: - 5.09765625 frac[4,59] + mem[4,59] = 0
 accuracy_8_62: - 5.09765625 frac[8,62] + mem[8,62] = 0
 accuracy_7_30: - 5.09765625 frac[7,30] + mem[7,30] = 0
 accuracy_13_71: - 5.09765625 frac[13,71] + mem[13,71] = 0
 accuracy_0_75: - 5.09765625 frac[0,75] + mem[0,75] = 0
 accuracy_9_63: - 5.09765625 frac[9,63] + mem[9,63] = 0
 accuracy_6_49: - 5.09765625 frac[6,49] + mem[6,49] = 0
 accuracy_4_65: - 5.09765625 frac[4,65] + mem[4,65] = 0
 accuracy_1_65: - 5.09765625 frac[1,65] + mem[1,65] = 0
 accuracy_8_4: - 5.09765625 frac[8,4] + mem[8,4] = 0
 accuracy_13_33: - 5.09765625 frac[13,33] + mem[13,33] = 0
 accuracy_11_49: - 5.09765625 frac[11,49] + mem[11,49] = 0
 accuracy_9_1: - 5.09765625 frac[9,1] + mem[9,1] = 0
 accuracy_5_75: - 5.09765625 frac[5,75] + mem[5,75] = 0
 accuracy_14_30: - 5.09765625 frac[14,30] + mem[14,30] = 0
 accuracy_12_58: - 5.09765625 frac[12,58] + mem[12,58] = 0
 accuracy_10_6: - 5.09765625 frac[10,6] + mem[10,6] = 0
 accuracy_13_59: - 5.09765625 frac[13,59] + mem[13,59] = 0
 accuracy_11_23: - 5.09765625 frac[11,23] + mem[11,23] = 0
 accuracy_0_16: - 5.09765625 frac[0,16] + mem[0,16] = 0
 accuracy_14_0: - 5.09765625 frac[14,0] + mem[14,0] = 0
 accuracy_12_16: - 5.09765625 frac[12,16] + mem[12,16] = 0
 accuracy_13_21: - 5.09765625 frac[13,21] + mem[13,21] = 0
 accuracy_2_26: - 5.09765625 frac[2,26] + mem[2,26] = 0
 accuracy_0_54: - 5.09765625 frac[0,54] + mem[0,54] = 0
 accuracy_14_42: - 5.09765625 frac[14,42] + mem[14,42] = 0
 accuracy_3_3: - 5.09765625 frac[3,3] + mem[3,3] = 0
 accuracy_1_55: - 5.09765625 frac[1,55] + mem[1,55] = 0
 accuracy_4_4: - 5.09765625 frac[4,4] + mem[4,4] = 0
 accuracy_5_25: - 5.09765625 frac[5,25] + mem[5,25] = 0
 accuracy_14_76: - 5.09765625 frac[14,76] + mem[14,76] = 0
 accuracy_3_41: - 5.09765625 frac[3,41] + mem[3,41] = 0
 accuracy_12_68: - 5.09765625 frac[12,68] + mem[12,68] = 0
 accuracy_1_41: - 5.09765625 frac[1,41] + mem[1,41] = 0
 accuracy_6_22: - 5.09765625 frac[6,22] + mem[6,22] = 0
 accuracy_4_34: - 5.09765625 frac[4,34] + mem[4,34] = 0
 accuracy_2_46: - 5.09765625 frac[2,46] + mem[2,46] = 0
 accuracy_8_37: - 5.09765625 frac[8,37] + mem[8,37] = 0
 accuracy_13_64: - 5.09765625 frac[13,64] + mem[13,64] = 0
 accuracy_10_39: - 5.09765625 frac[10,39] + mem[10,39] = 0
 accuracy_2_72: - 5.09765625 frac[2,72] + mem[2,72] = 0
 accuracy_8_11: - 5.09765625 frac[8,11] + mem[8,11] = 0
 accuracy_7_45: - 5.09765625 frac[7,45] + mem[7,45] = 0
 accuracy_11_54: - 5.09765625 frac[11,54] + mem[11,54] = 0
 accuracy_6_66: - 5.09765625 frac[6,66] + mem[6,66] = 0
 accuracy_12_49: - 5.09765625 frac[12,49] + mem[12,49] = 0
 accuracy_10_1: - 5.09765625 frac[10,1] + mem[10,1] = 0
 accuracy_8_17: - 5.09765625 frac[8,17] + mem[8,17] = 0
 accuracy_13_52: - 5.09765625 frac[13,52] + mem[13,52] = 0
 accuracy_11_12: - 5.09765625 frac[11,12] + mem[11,12] = 0
 accuracy_0_23: - 5.09765625 frac[0,23] + mem[0,23] = 0
 accuracy_7_66: - 5.09765625 frac[7,66] + mem[7,66] = 0
 accuracy_14_11: - 5.09765625 frac[14,11] + mem[14,11] = 0
 accuracy_12_15: - 5.09765625 frac[12,15] + mem[12,15] = 0
 accuracy_1_22: - 5.09765625 frac[1,22] + mem[1,22] = 0
 accuracy_13_14: - 5.09765625 frac[13,14] + mem[13,14] = 0
 accuracy_2_21: - 5.09765625 frac[2,21] + mem[2,21] = 0
 accuracy_0_13: - 5.09765625 frac[0,13] + mem[0,13] = 0
 accuracy_14_45: - 5.09765625 frac[14,45] + mem[14,45] = 0
 accuracy_4_3: - 5.09765625 frac[4,3] + mem[4,3] = 0
 accuracy_2_15: - 5.09765625 frac[2,15] + mem[2,15] = 0
 accuracy_0_35: - 5.09765625 frac[0,35] + mem[0,35] = 0
 accuracy_5_2: - 5.09765625 frac[5,2] + mem[5,2] = 0
 accuracy_12_67: - 5.09765625 frac[12,67] + mem[12,67] = 0
 accuracy_9_71: - 5.09765625 frac[9,71] + mem[9,71] = 0
 accuracy_1_34: - 5.09765625 frac[1,34] + mem[1,34] = 0
 accuracy_3_46: - 5.09765625 frac[3,46] + mem[3,46] = 0
 accuracy_6_9: - 5.09765625 frac[6,9] + mem[6,9] = 0
 accuracy_4_41: - 5.09765625 frac[4,41] + mem[4,41] = 0
 accuracy_8_44: - 5.09765625 frac[8,44] + mem[8,44] = 0
 accuracy_7_12: - 5.09765625 frac[7,12] + mem[7,12] = 0
 accuracy_5_44: - 5.09765625 frac[5,44] + mem[5,44] = 0
 accuracy_11_73: - 5.09765625 frac[11,73] + mem[11,73] = 0
 accuracy_9_41: - 5.09765625 frac[9,41] + mem[9,41] = 0
 accuracy_4_55: - 5.09765625 frac[4,55] + mem[4,55] = 0
 accuracy_10_46: - 5.09765625 frac[10,46] + mem[10,46] = 0
 accuracy_2_67: - 5.09765625 frac[2,67] + mem[2,67] = 0
 accuracy_8_50: - 5.09765625 frac[8,50] + mem[8,50] = 0
 accuracy_5_54: - 5.09765625 frac[5,54] + mem[5,54] = 0
 accuracy_9_51: - 5.09765625 frac[9,51] + mem[9,51] = 0
 accuracy_6_69: - 5.09765625 frac[6,69] + mem[6,69] = 0
 accuracy_12_40: - 5.09765625 frac[12,40] + mem[12,40] = 0
 accuracy_10_8: - 5.09765625 frac[10,8] + mem[10,8] = 0
 accuracy_8_24: - 5.09765625 frac[8,24] + mem[8,24] = 0
 accuracy_13_45: - 5.09765625 frac[13,45] + mem[13,45] = 0
 accuracy_11_5: - 5.09765625 frac[11,5] + mem[11,5] = 0
 accuracy_9_21: - 5.09765625 frac[9,21] + mem[9,21] = 0
 accuracy_14_18: - 5.09765625 frac[14,18] + mem[14,18] = 0
 accuracy_5_71: - 5.09765625 frac[5,71] + mem[5,71] = 0
 accuracy_12_6: - 5.09765625 frac[12,6] + mem[12,6] = 0
 accuracy_13_7: - 5.09765625 frac[13,7] + mem[13,7] = 0
 accuracy_11_27: - 5.09765625 frac[11,27] + mem[11,27] = 0
 accuracy_0_4: - 5.09765625 frac[0,4] + mem[0,4] = 0
 accuracy_14_52: - 5.09765625 frac[14,52] + mem[14,52] = 0
 accuracy_8_68: - 5.09765625 frac[8,68] + mem[8,68] = 0
 accuracy_4_10: - 5.09765625 frac[4,10] + mem[4,10] = 0
 accuracy_0_58: - 5.09765625 frac[0,58] + mem[0,58] = 0
 accuracy_5_11: - 5.09765625 frac[5,11] + mem[5,11] = 0
 accuracy_9_72: - 5.09765625 frac[9,72] + mem[9,72] = 0
 accuracy_10_70: - 5.09765625 frac[10,70] + mem[10,70] = 0
 accuracy_6_0: - 5.09765625 frac[6,0] + mem[6,0] = 0
 accuracy_4_16: - 5.09765625 frac[4,16] + mem[4,16] = 0
 accuracy_7_5: - 5.09765625 frac[7,5] + mem[7,5] = 0
 accuracy_14_64: - 5.09765625 frac[14,64] + mem[14,64] = 0
 accuracy_11_78: - 5.09765625 frac[11,78] + mem[11,78] = 0
 accuracy_3_61: - 5.09765625 frac[3,61] + mem[3,61] = 0
 accuracy_6_26: - 5.09765625 frac[6,26] + mem[6,26] = 0
 accuracy_4_62: - 5.09765625 frac[4,62] + mem[4,62] = 0
 accuracy_10_41: - 5.09765625 frac[10,41] + mem[10,41] = 0
 accuracy_8_57: - 5.09765625 frac[8,57] + mem[8,57] = 0
 accuracy_7_35: - 5.09765625 frac[7,35] + mem[7,35] = 0
 accuracy_13_76: - 5.09765625 frac[13,76] + mem[13,76] = 0
 accuracy_5_63: - 5.09765625 frac[5,63] + mem[5,63] = 0
 accuracy_11_36: - 5.09765625 frac[11,36] + mem[11,36] = 0
 accuracy_6_60: - 5.09765625 frac[6,60] + mem[6,60] = 0
 accuracy_3_74: - 5.09765625 frac[3,74] + mem[3,74] = 0
 accuracy_12_39: - 5.09765625 frac[12,39] + mem[12,39] = 0
 accuracy_4_68: - 5.09765625 frac[4,68] + mem[4,68] = 0
 accuracy_10_51: - 5.09765625 frac[10,51] + mem[10,51] = 0
 accuracy_8_31: - 5.09765625 frac[8,31] + mem[8,31] = 0
 accuracy_7_57: - 5.09765625 frac[7,57] + mem[7,57] = 0
 accuracy_13_38: - 5.09765625 frac[13,38] + mem[13,38] = 0
 accuracy_11_58: - 5.09765625 frac[11,58] + mem[11,58] = 0
 accuracy_9_30: - 5.09765625 frac[9,30] + mem[9,30] = 0
 accuracy_14_21: - 5.09765625 frac[14,21] + mem[14,21] = 0
 accuracy_5_64: - 5.09765625 frac[5,64] + mem[5,64] = 0
 accuracy_12_61: - 5.09765625 frac[12,61] + mem[12,61] = 0
 accuracy_10_29: - 5.09765625 frac[10,29] + mem[10,29] = 0
 accuracy_13_0: - 5.09765625 frac[13,0] + mem[13,0] = 0
 accuracy_11_16: - 5.09765625 frac[11,16] + mem[11,16] = 0
 accuracy_0_27: - 5.09765625 frac[0,27] + mem[0,27] = 0
 accuracy_7_78: - 5.09765625 frac[7,78] + mem[7,78] = 0
 accuracy_14_63: - 5.09765625 frac[14,63] + mem[14,63] = 0
 accuracy_12_27: - 5.09765625 frac[12,27] + mem[12,27] = 0
 accuracy_1_26: - 5.09765625 frac[1,26] + mem[1,26] = 0
 accuracy_8_75: - 5.09765625 frac[8,75] + mem[8,75] = 0
 accuracy_13_26: - 5.09765625 frac[13,26] + mem[13,26] = 0
 accuracy_14_33: - 5.09765625 frac[14,33] + mem[14,33] = 0
 accuracy_3_28: - 5.09765625 frac[3,28] + mem[3,28] = 0
 accuracy_10_65: - 5.09765625 frac[10,65] + mem[10,65] = 0
 accuracy_1_60: - 5.09765625 frac[1,60] + mem[1,60] = 0
 accuracy_2_59: - 5.09765625 frac[2,59] + mem[2,59] = 0
 accuracy_7_2: - 5.09765625 frac[7,2] + mem[7,2] = 0
 accuracy_5_30: - 5.09765625 frac[5,30] + mem[5,30] = 0
 accuracy_11_71: - 5.09765625 frac[11,71] + mem[11,71] = 0
 accuracy_3_34: - 5.09765625 frac[3,34] + mem[3,34] = 0
 accuracy_12_79: - 5.09765625 frac[12,79] + mem[12,79] = 0
 accuracy_14_75: - 5.09765625 frac[14,75] + mem[14,75] = 0
 accuracy_6_29: - 5.09765625 frac[6,29] + mem[6,29] = 0
 accuracy_4_37: - 5.09765625 frac[4,37] + mem[4,37] = 0
 accuracy_8_32: - 5.09765625 frac[8,32] + mem[8,32] = 0
 accuracy_7_24: - 5.09765625 frac[7,24] + mem[7,24] = 0
 accuracy_13_69: - 5.09765625 frac[13,69] + mem[13,69] = 0
 accuracy_5_56: - 5.09765625 frac[5,56] + mem[5,56] = 0
 accuracy_0_77: - 5.09765625 frac[0,77] + mem[0,77] = 0
 accuracy_9_61: - 5.09765625 frac[9,61] + mem[9,61] = 0
 accuracy_4_67: - 5.09765625 frac[4,67] + mem[4,67] = 0
 accuracy_2_79: - 5.09765625 frac[2,79] + mem[2,79] = 0
 accuracy_8_6: - 5.09765625 frac[8,6] + mem[8,6] = 0
 accuracy_7_54: - 5.09765625 frac[7,54] + mem[7,54] = 0
 accuracy_9_7: - 5.09765625 frac[9,7] + mem[9,7] = 0
 accuracy_6_73: - 5.09765625 frac[6,73] + mem[6,73] = 0
 accuracy_12_52: - 5.09765625 frac[12,52] + mem[12,52] = 0
 accuracy_10_4: - 5.09765625 frac[10,4] + mem[10,4] = 0
 accuracy_13_57: - 5.09765625 frac[13,57] + mem[13,57] = 0
 accuracy_11_9: - 5.09765625 frac[11,9] + mem[11,9] = 0
 accuracy_7_71: - 5.09765625 frac[7,71] + mem[7,71] = 0
 accuracy_0_18: - 5.09765625 frac[0,18] + mem[0,18] = 0
 accuracy_14_6: - 5.09765625 frac[14,6] + mem[14,6] = 0
 accuracy_12_18: - 5.09765625 frac[12,18] + mem[12,18] = 0
 accuracy_1_19: - 5.09765625 frac[1,19] + mem[1,19] = 0
 accuracy_13_19: - 5.09765625 frac[13,19] + mem[13,19] = 0
 accuracy_14_40: - 5.09765625 frac[14,40] + mem[14,40] = 0
 accuracy_1_53: - 5.09765625 frac[1,53] + mem[1,53] = 0
 accuracy_4_6: - 5.09765625 frac[4,6] + mem[4,6] = 0
 accuracy_2_50: - 5.09765625 frac[2,50] + mem[2,50] = 0
 accuracy_0_46: - 5.09765625 frac[0,46] + mem[0,46] = 0
 accuracy_5_7: - 5.09765625 frac[5,7] + mem[5,7] = 0
 accuracy_12_70: - 5.09765625 frac[12,70] + mem[12,70] = 0
 accuracy_3_43: - 5.09765625 frac[3,43] + mem[3,43] = 0
 accuracy_1_47: - 5.09765625 frac[1,47] + mem[1,47] = 0
 accuracy_6_20: - 5.09765625 frac[6,20] + mem[6,20] = 0
 accuracy_8_39: - 5.09765625 frac[8,39] + mem[8,39] = 0
 accuracy_0_68: - 5.09765625 frac[0,68] + mem[0,68] = 0
 accuracy_7_17: - 5.09765625 frac[7,17] + mem[7,17] = 0
 accuracy_5_33: - 5.09765625 frac[5,33] + mem[5,33] = 0
 accuracy_3_49: - 5.09765625 frac[3,49] + mem[3,49] = 0
 accuracy_9_38: - 5.09765625 frac[9,38] + mem[9,38] = 0
 accuracy_10_37: - 5.09765625 frac[10,37] + mem[10,37] = 0
 accuracy_2_70: - 5.09765625 frac[2,70] + mem[2,70] = 0
 accuracy_8_13: - 5.09765625 frac[8,13] + mem[8,13] = 0
 accuracy_11_40: - 5.09765625 frac[11,40] + mem[11,40] = 0
 accuracy_9_8: - 5.09765625 frac[9,8] + mem[9,8] = 0
 accuracy_6_64: - 5.09765625 frac[6,64] + mem[6,64] = 0
 accuracy_12_51: - 5.09765625 frac[12,51] + mem[12,51] = 0
 accuracy_10_15: - 5.09765625 frac[10,15] + mem[10,15] = 0
 accuracy_8_19: - 5.09765625 frac[8,19] + mem[8,19] = 0
 accuracy_13_50: - 5.09765625 frac[13,50] + mem[13,50] = 0
 accuracy_11_14: - 5.09765625 frac[11,14] + mem[11,14] = 0
 accuracy_9_18: - 5.09765625 frac[9,18] + mem[9,18] = 0
 accuracy_14_9: - 5.09765625 frac[14,9] + mem[14,9] = 0
 accuracy_12_9: - 5.09765625 frac[12,9] + mem[12,9] = 0
 accuracy_1_20: - 5.09765625 frac[1,20] + mem[1,20] = 0
 accuracy_13_12: - 5.09765625 frac[13,12] + mem[13,12] = 0
 accuracy_2_19: - 5.09765625 frac[2,19] + mem[2,19] = 0
 accuracy_0_15: - 5.09765625 frac[0,15] + mem[0,15] = 0
 accuracy_14_51: - 5.09765625 frac[14,51] + mem[14,51] = 0
 accuracy_1_14: - 5.09765625 frac[1,14] + mem[1,14] = 0
 accuracy_4_13: - 5.09765625 frac[4,13] + mem[4,13] = 0
 accuracy_2_13: - 5.09765625 frac[2,13] + mem[2,13] = 0
 accuracy_0_37: - 5.09765625 frac[0,37] + mem[0,37] = 0
 accuracy_3_16: - 5.09765625 frac[3,16] + mem[3,16] = 0
 accuracy_9_69: - 5.09765625 frac[9,69] + mem[9,69] = 0
 accuracy_6_15: - 5.09765625 frac[6,15] + mem[6,15] = 0
 accuracy_4_43: - 5.09765625 frac[4,43] + mem[4,43] = 0
 accuracy_2_39: - 5.09765625 frac[2,39] + mem[2,39] = 0
 accuracy_8_46: - 5.09765625 frac[8,46] + mem[8,46] = 0
 accuracy_7_14: - 5.09765625 frac[7,14] + mem[7,14] = 0
 accuracy_5_42: - 5.09765625 frac[5,42] + mem[5,42] = 0
 accuracy_11_75: - 5.09765625 frac[11,75] + mem[11,75] = 0
 accuracy_9_47: - 5.09765625 frac[9,47] + mem[9,47] = 0
 accuracy_6_33: - 5.09765625 frac[6,33] + mem[6,33] = 0
 accuracy_4_49: - 5.09765625 frac[4,49] + mem[4,49] = 0
 accuracy_10_44: - 5.09765625 frac[10,44] + mem[10,44] = 0
 accuracy_8_52: - 5.09765625 frac[8,52] + mem[8,52] = 0
 accuracy_7_36: - 5.09765625 frac[7,36] + mem[7,36] = 0
 accuracy_5_52: - 5.09765625 frac[5,52] + mem[5,52] = 0
 accuracy_11_33: - 5.09765625 frac[11,33] + mem[11,33] = 0
 accuracy_9_49: - 5.09765625 frac[9,49] + mem[9,49] = 0
 accuracy_6_59: - 5.09765625 frac[6,59] + mem[6,59] = 0
 accuracy_3_71: - 5.09765625 frac[3,71] + mem[3,71] = 0
 accuracy_12_42: - 5.09765625 frac[12,42] + mem[12,42] = 0
 accuracy_10_54: - 5.09765625 frac[10,54] + mem[10,54] = 0
 accuracy_8_26: - 5.09765625 frac[8,26] + mem[8,26] = 0
 accuracy_13_43: - 5.09765625 frac[13,43] + mem[13,43] = 0
 accuracy_11_7: - 5.09765625 frac[11,7] + mem[11,7] = 0
 accuracy_9_27: - 5.09765625 frac[9,27] + mem[9,27] = 0
 accuracy_14_16: - 5.09765625 frac[14,16] + mem[14,16] = 0
 accuracy_12_0: - 5.09765625 frac[12,0] + mem[12,0] = 0
 accuracy_10_16: - 5.09765625 frac[10,16] + mem[10,16] = 0
 accuracy_13_5: - 5.09765625 frac[13,5] + mem[13,5] = 0
 accuracy_11_29: - 5.09765625 frac[11,29] + mem[11,29] = 0
 accuracy_14_58: - 5.09765625 frac[14,58] + mem[14,58] = 0
 accuracy_12_30: - 5.09765625 frac[12,30] + mem[12,30] = 0
 accuracy_1_7: - 5.09765625 frac[1,7] + mem[1,7] = 0
 accuracy_8_70: - 5.09765625 frac[8,70] + mem[8,70] = 0
 accuracy_13_31: - 5.09765625 frac[13,31] + mem[13,31] = 0
 accuracy_2_4: - 5.09765625 frac[2,4] + mem[2,4] = 0
 accuracy_0_60: - 5.09765625 frac[0,60] + mem[0,60] = 0
 accuracy_5_9: - 5.09765625 frac[5,9] + mem[5,9] = 0
 accuracy_9_78: - 5.09765625 frac[9,78] + mem[9,78] = 0
 accuracy_10_68: - 5.09765625 frac[10,68] + mem[10,68] = 0
 accuracy_1_57: - 5.09765625 frac[1,57] + mem[1,57] = 0
 accuracy_6_6: - 5.09765625 frac[6,6] + mem[6,6] = 0
 accuracy_2_62: - 5.09765625 frac[2,62] + mem[2,62] = 0
 accuracy_7_7: - 5.09765625 frac[7,7] + mem[7,7] = 0
 accuracy_14_70: - 5.09765625 frac[14,70] + mem[14,70] = 0
 accuracy_11_64: - 5.09765625 frac[11,64] + mem[11,64] = 0
 accuracy_3_63: - 5.09765625 frac[3,63] + mem[3,63] = 0
 accuracy_5_19: - 5.09765625 frac[5,19] + mem[5,19] = 0
 accuracy_6_24: - 5.09765625 frac[6,24] + mem[6,24] = 0
 accuracy_8_59: - 5.09765625 frac[8,59] + mem[8,59] = 0
 accuracy_7_29: - 5.09765625 frac[7,29] + mem[7,29] = 0
 accuracy_13_74: - 5.09765625 frac[13,74] + mem[13,74] = 0
 accuracy_5_61: - 5.09765625 frac[5,61] + mem[5,61] = 0
 accuracy_0_72: - 5.09765625 frac[0,72] + mem[0,72] = 0
 accuracy_11_38: - 5.09765625 frac[11,38] + mem[11,38] = 0
 accuracy_9_58: - 5.09765625 frac[9,58] + mem[9,58] = 0
 accuracy_6_50: - 5.09765625 frac[6,50] + mem[6,50] = 0
 accuracy_12_33: - 5.09765625 frac[12,33] + mem[12,33] = 0
 accuracy_3_76: - 5.09765625 frac[3,76] + mem[3,76] = 0
 accuracy_10_49: - 5.09765625 frac[10,49] + mem[10,49] = 0
 accuracy_8_1: - 5.09765625 frac[8,1] + mem[8,1] = 0
 accuracy_13_36: - 5.09765625 frac[13,36] + mem[13,36] = 0
 accuracy_11_60: - 5.09765625 frac[11,60] + mem[11,60] = 0
 accuracy_9_28: - 5.09765625 frac[9,28] + mem[9,28] = 0
 accuracy_5_78: - 5.09765625 frac[5,78] + mem[5,78] = 0
 accuracy_14_27: - 5.09765625 frac[14,27] + mem[14,27] = 0
 accuracy_12_63: - 5.09765625 frac[12,63] + mem[12,63] = 0
 accuracy_10_27: - 5.09765625 frac[10,27] + mem[10,27] = 0
 accuracy_13_62: - 5.09765625 frac[13,62] + mem[13,62] = 0
 accuracy_11_18: - 5.09765625 frac[11,18] + mem[11,18] = 0
 accuracy_14_61: - 5.09765625 frac[14,61] + mem[14,61] = 0
 accuracy_12_21: - 5.09765625 frac[12,21] + mem[12,21] = 0
 accuracy_8_77: - 5.09765625 frac[8,77] + mem[8,77] = 0
 accuracy_13_24: - 5.09765625 frac[13,24] + mem[13,24] = 0
 accuracy_14_39: - 5.09765625 frac[14,39] + mem[14,39] = 0
 accuracy_3_30: - 5.09765625 frac[3,30] + mem[3,30] = 0
 accuracy_10_79: - 5.09765625 frac[10,79] + mem[10,79] = 0
 accuracy_1_50: - 5.09765625 frac[1,50] + mem[1,50] = 0
 accuracy_2_57: - 5.09765625 frac[2,57] + mem[2,57] = 0
 accuracy_0_41: - 5.09765625 frac[0,41] + mem[0,41] = 0
 accuracy_14_73: - 5.09765625 frac[14,73] + mem[14,73] = 0
 accuracy_12_73: - 5.09765625 frac[12,73] + mem[12,73] = 0
 accuracy_6_19: - 5.09765625 frac[6,19] + mem[6,19] = 0
 accuracy_8_34: - 5.09765625 frac[8,34] + mem[8,34] = 0
 accuracy_7_26: - 5.09765625 frac[7,26] + mem[7,26] = 0
 accuracy_13_67: - 5.09765625 frac[13,67] + mem[13,67] = 0
 accuracy_5_38: - 5.09765625 frac[5,38] + mem[5,38] = 0
 accuracy_9_35: - 5.09765625 frac[9,35] + mem[9,35] = 0
 accuracy_6_53: - 5.09765625 frac[6,53] + mem[6,53] = 0
 accuracy_4_77: - 5.09765625 frac[4,77] + mem[4,77] = 0
 accuracy_10_56: - 5.09765625 frac[10,56] + mem[10,56] = 0
 accuracy_2_77: - 5.09765625 frac[2,77] + mem[2,77] = 0
 accuracy_7_48: - 5.09765625 frac[7,48] + mem[7,48] = 0
 accuracy_11_53: - 5.09765625 frac[11,53] + mem[11,53] = 0
 accuracy_12_54: - 5.09765625 frac[12,54] + mem[12,54] = 0
 accuracy_10_2: - 5.09765625 frac[10,2] + mem[10,2] = 0
 accuracy_13_55: - 5.09765625 frac[13,55] + mem[13,55] = 0
 accuracy_11_11: - 5.09765625 frac[11,11] + mem[11,11] = 0
 accuracy_7_65: - 5.09765625 frac[7,65] + mem[7,65] = 0
 accuracy_0_20: - 5.09765625 frac[0,20] + mem[0,20] = 0
 accuracy_14_4: - 5.09765625 frac[14,4] + mem[14,4] = 0
 accuracy_12_12: - 5.09765625 frac[12,12] + mem[12,12] = 0
 accuracy_1_17: - 5.09765625 frac[1,17] + mem[1,17] = 0
 accuracy_13_17: - 5.09765625 frac[13,17] + mem[13,17] = 0
 accuracy_2_22: - 5.09765625 frac[2,22] + mem[2,22] = 0
 accuracy_0_10: - 5.09765625 frac[0,10] + mem[0,10] = 0
 accuracy_14_46: - 5.09765625 frac[14,46] + mem[14,46] = 0
 accuracy_3_7: - 5.09765625 frac[3,7] + mem[3,7] = 0
 accuracy_1_11: - 5.09765625 frac[1,11] + mem[1,11] = 0
 accuracy_4_0: - 5.09765625 frac[4,0] + mem[4,0] = 0
 accuracy_0_32: - 5.09765625 frac[0,32] + mem[0,32] = 0
 accuracy_5_5: - 5.09765625 frac[5,5] + mem[5,5] = 0
 accuracy_3_45: - 5.09765625 frac[3,45] + mem[3,45] = 0
 accuracy_12_64: - 5.09765625 frac[12,64] + mem[12,64] = 0
 accuracy_9_66: - 5.09765625 frac[9,66] + mem[9,66] = 0
 accuracy_4_46: - 5.09765625 frac[4,46] + mem[4,46] = 0
 accuracy_2_42: - 5.09765625 frac[2,42] + mem[2,42] = 0
 accuracy_8_41: - 5.09765625 frac[8,41] + mem[8,41] = 0
 accuracy_7_19: - 5.09765625 frac[7,19] + mem[7,19] = 0
 accuracy_5_47: - 5.09765625 frac[5,47] + mem[5,47] = 0
 accuracy_3_51: - 5.09765625 frac[3,51] + mem[3,51] = 0
 accuracy_9_36: - 5.09765625 frac[9,36] + mem[9,36] = 0
 accuracy_4_52: - 5.09765625 frac[4,52] + mem[4,52] = 0
 accuracy_10_35: - 5.09765625 frac[10,35] + mem[10,35] = 0
 accuracy_2_68: - 5.09765625 frac[2,68] + mem[2,68] = 0
 accuracy_8_15: - 5.09765625 frac[8,15] + mem[8,15] = 0
 accuracy_7_41: - 5.09765625 frac[7,41] + mem[7,41] = 0
 accuracy_9_14: - 5.09765625 frac[9,14] + mem[9,14] = 0
 accuracy_6_70: - 5.09765625 frac[6,70] + mem[6,70] = 0
 accuracy_12_45: - 5.09765625 frac[12,45] + mem[12,45] = 0
 accuracy_3_64: - 5.09765625 frac[3,64] + mem[3,64] = 0
 accuracy_10_13: - 5.09765625 frac[10,13] + mem[10,13] = 0
 accuracy_13_48: - 5.09765625 frac[13,48] + mem[13,48] = 0
 accuracy_11_0: - 5.09765625 frac[11,0] + mem[11,0] = 0
 accuracy_9_16: - 5.09765625 frac[9,16] + mem[9,16] = 0
 accuracy_14_15: - 5.09765625 frac[14,15] + mem[14,15] = 0
 accuracy_12_11: - 5.09765625 frac[12,11] + mem[12,11] = 0
 accuracy_13_10: - 5.09765625 frac[13,10] + mem[13,10] = 0
 accuracy_0_1: - 5.09765625 frac[0,1] + mem[0,1] = 0
 accuracy_14_49: - 5.09765625 frac[14,49] + mem[14,49] = 0
 accuracy_1_12: - 5.09765625 frac[1,12] + mem[1,12] = 0
 accuracy_8_65: - 5.09765625 frac[8,65] + mem[8,65] = 0
 accuracy_4_15: - 5.09765625 frac[4,15] + mem[4,15] = 0
 accuracy_2_11: - 5.09765625 frac[2,11] + mem[2,11] = 0
 accuracy_0_39: - 5.09765625 frac[0,39] + mem[0,39] = 0
 accuracy_3_18: - 5.09765625 frac[3,18] + mem[3,18] = 0
 accuracy_9_75: - 5.09765625 frac[9,75] + mem[9,75] = 0
 accuracy_1_38: - 5.09765625 frac[1,38] + mem[1,38] = 0
 accuracy_6_13: - 5.09765625 frac[6,13] + mem[6,13] = 0
 accuracy_4_21: - 5.09765625 frac[4,21] + mem[4,21] = 0
 accuracy_2_37: - 5.09765625 frac[2,37] + mem[2,37] = 0
 accuracy_5_40: - 5.09765625 frac[5,40] + mem[5,40] = 0
 accuracy_11_77: - 5.09765625 frac[11,77] + mem[11,77] = 0
 accuracy_9_45: - 5.09765625 frac[9,45] + mem[9,45] = 0
 accuracy_4_51: - 5.09765625 frac[4,51] + mem[4,51] = 0
 accuracy_10_42: - 5.09765625 frac[10,42] + mem[10,42] = 0
 accuracy_8_54: - 5.09765625 frac[8,54] + mem[8,54] = 0
 accuracy_7_38: - 5.09765625 frac[7,38] + mem[7,38] = 0
 accuracy_13_79: - 5.09765625 frac[13,79] + mem[13,79] = 0
 accuracy_5_50: - 5.09765625 frac[5,50] + mem[5,50] = 0
 accuracy_11_35: - 5.09765625 frac[11,35] + mem[11,35] = 0
 accuracy_9_55: - 5.09765625 frac[9,55] + mem[9,55] = 0
 accuracy_3_73: - 5.09765625 frac[3,73] + mem[3,73] = 0
 accuracy_12_36: - 5.09765625 frac[12,36] + mem[12,36] = 0
 accuracy_10_52: - 5.09765625 frac[10,52] + mem[10,52] = 0
 accuracy_1_73: - 5.09765625 frac[1,73] + mem[1,73] = 0
 accuracy_8_28: - 5.09765625 frac[8,28] + mem[8,28] = 0
 accuracy_13_41: - 5.09765625 frac[13,41] + mem[13,41] = 0
 accuracy_11_57: - 5.09765625 frac[11,57] + mem[11,57] = 0
 accuracy_9_25: - 5.09765625 frac[9,25] + mem[9,25] = 0
 accuracy_14_22: - 5.09765625 frac[14,22] + mem[14,22] = 0
 accuracy_5_67: - 5.09765625 frac[5,67] + mem[5,67] = 0
 accuracy_12_2: - 5.09765625 frac[12,2] + mem[12,2] = 0
 accuracy_10_30: - 5.09765625 frac[10,30] + mem[10,30] = 0
 accuracy_13_3: - 5.09765625 frac[13,3] + mem[13,3] = 0
 accuracy_11_31: - 5.09765625 frac[11,31] + mem[11,31] = 0
 accuracy_0_24: - 5.09765625 frac[0,24] + mem[0,24] = 0
 accuracy_7_77: - 5.09765625 frac[7,77] + mem[7,77] = 0
 accuracy_14_56: - 5.09765625 frac[14,56] + mem[14,56] = 0
 accuracy_12_24: - 5.09765625 frac[12,24] + mem[12,24] = 0
 accuracy_8_72: - 5.09765625 frac[8,72] + mem[8,72] = 0
 accuracy_13_29: - 5.09765625 frac[13,29] + mem[13,29] = 0
 accuracy_2_2: - 5.09765625 frac[2,2] + mem[2,2] = 0
 accuracy_0_62: - 5.09765625 frac[0,62] + mem[0,62] = 0
 accuracy_14_34: - 5.09765625 frac[14,34] + mem[14,34] = 0
 accuracy_9_76: - 5.09765625 frac[9,76] + mem[9,76] = 0
 accuracy_10_66: - 5.09765625 frac[10,66] + mem[10,66] = 0
 accuracy_1_63: - 5.09765625 frac[1,63] + mem[1,63] = 0
 accuracy_4_28: - 5.09765625 frac[4,28] + mem[4,28] = 0
 accuracy_2_60: - 5.09765625 frac[2,60] + mem[2,60] = 0
 accuracy_7_1: - 5.09765625 frac[7,1] + mem[7,1] = 0
 accuracy_14_68: - 5.09765625 frac[14,68] + mem[14,68] = 0
 accuracy_11_66: - 5.09765625 frac[11,66] + mem[11,66] = 0
 accuracy_3_33: - 5.09765625 frac[3,33] + mem[3,33] = 0
 accuracy_12_76: - 5.09765625 frac[12,76] + mem[12,76] = 0
 accuracy_6_30: - 5.09765625 frac[6,30] + mem[6,30] = 0
 accuracy_8_61: - 5.09765625 frac[8,61] + mem[8,61] = 0
 accuracy_7_31: - 5.09765625 frac[7,31] + mem[7,31] = 0
 accuracy_0_74: - 5.09765625 frac[0,74] + mem[0,74] = 0
 accuracy_5_59: - 5.09765625 frac[5,59] + mem[5,59] = 0
 accuracy_9_56: - 5.09765625 frac[9,56] + mem[9,56] = 0
 accuracy_6_48: - 5.09765625 frac[6,48] + mem[6,48] = 0
 accuracy_12_35: - 5.09765625 frac[12,35] + mem[12,35] = 0
 accuracy_4_64: - 5.09765625 frac[4,64] + mem[4,64] = 0
 accuracy_10_63: - 5.09765625 frac[10,63] + mem[10,63] = 0
 accuracy_1_66: - 5.09765625 frac[1,66] + mem[1,66] = 0
 accuracy_8_3: - 5.09765625 frac[8,3] + mem[8,3] = 0
 accuracy_7_53: - 5.09765625 frac[7,53] + mem[7,53] = 0
 accuracy_13_34: - 5.09765625 frac[13,34] + mem[13,34] = 0
 accuracy_11_62: - 5.09765625 frac[11,62] + mem[11,62] = 0
 accuracy_9_2: - 5.09765625 frac[9,2] + mem[9,2] = 0
 accuracy_14_25: - 5.09765625 frac[14,25] + mem[14,25] = 0
 accuracy_6_74: - 5.09765625 frac[6,74] + mem[6,74] = 0
 accuracy_12_57: - 5.09765625 frac[12,57] + mem[12,57] = 0
 accuracy_10_25: - 5.09765625 frac[10,25] + mem[10,25] = 0
 accuracy_13_60: - 5.09765625 frac[13,60] + mem[13,60] = 0
 accuracy_11_20: - 5.09765625 frac[11,20] + mem[11,20] = 0
 accuracy_0_31: - 5.09765625 frac[0,31] + mem[0,31] = 0
 accuracy_7_74: - 5.09765625 frac[7,74] + mem[7,74] = 0
 accuracy_14_3: - 5.09765625 frac[14,3] + mem[14,3] = 0
 accuracy_12_23: - 5.09765625 frac[12,23] + mem[12,23] = 0
 accuracy_1_30: - 5.09765625 frac[1,30] + mem[1,30] = 0
 accuracy_8_79: - 5.09765625 frac[8,79] + mem[8,79] = 0
 accuracy_13_22: - 5.09765625 frac[13,22] + mem[13,22] = 0
 accuracy_2_29: - 5.09765625 frac[2,29] + mem[2,29] = 0
 accuracy_0_53: - 5.09765625 frac[0,53] + mem[0,53] = 0
 accuracy_14_37: - 5.09765625 frac[14,37] + mem[14,37] = 0
 accuracy_3_0: - 5.09765625 frac[3,0] + mem[3,0] = 0
 accuracy_10_77: - 5.09765625 frac[10,77] + mem[10,77] = 0
 accuracy_1_48: - 5.09765625 frac[1,48] + mem[1,48] = 0
 accuracy_4_27: - 5.09765625 frac[4,27] + mem[4,27] = 0
 accuracy_2_55: - 5.09765625 frac[2,55] + mem[2,55] = 0
 accuracy_0_43: - 5.09765625 frac[0,43] + mem[0,43] = 0
 accuracy_14_79: - 5.09765625 frac[14,79] + mem[14,79] = 0
 accuracy_12_75: - 5.09765625 frac[12,75] + mem[12,75] = 0
 accuracy_3_38: - 5.09765625 frac[3,38] + mem[3,38] = 0
 accuracy_6_17: - 5.09765625 frac[6,17] + mem[6,17] = 0
 accuracy_4_33: - 5.09765625 frac[4,33] + mem[4,33] = 0
 accuracy_13_65: - 5.09765625 frac[13,65] + mem[13,65] = 0
 accuracy_9_33: - 5.09765625 frac[9,33] + mem[9,33] = 0
 accuracy_6_43: - 5.09765625 frac[6,43] + mem[6,43] = 0
 accuracy_4_79: - 5.09765625 frac[4,79] + mem[4,79] = 0
 accuracy_10_38: - 5.09765625 frac[10,38] + mem[10,38] = 0
 accuracy_8_10: - 5.09765625 frac[8,10] + mem[8,10] = 0
 accuracy_11_55: - 5.09765625 frac[11,55] + mem[11,55] = 0
 accuracy_9_11: - 5.09765625 frac[9,11] + mem[9,11] = 0
 accuracy_6_77: - 5.09765625 frac[6,77] + mem[6,77] = 0
 accuracy_12_48: - 5.09765625 frac[12,48] + mem[12,48] = 0
 accuracy_10_0: - 5.09765625 frac[10,0] + mem[10,0] = 0
 accuracy_8_16: - 5.09765625 frac[8,16] + mem[8,16] = 0
 accuracy_13_53: - 5.09765625 frac[13,53] + mem[13,53] = 0
 accuracy_11_13: - 5.09765625 frac[11,13] + mem[11,13] = 0
 accuracy_7_67: - 5.09765625 frac[7,67] + mem[7,67] = 0
 accuracy_14_10: - 5.09765625 frac[14,10] + mem[14,10] = 0
 accuracy_12_14: - 5.09765625 frac[12,14] + mem[12,14] = 0
 accuracy_1_23: - 5.09765625 frac[1,23] + mem[1,23] = 0
 accuracy_13_15: - 5.09765625 frac[13,15] + mem[13,15] = 0
 accuracy_0_12: - 5.09765625 frac[0,12] + mem[0,12] = 0
 accuracy_14_44: - 5.09765625 frac[14,44] + mem[14,44] = 0
 accuracy_3_9: - 5.09765625 frac[3,9] + mem[3,9] = 0
 accuracy_4_2: - 5.09765625 frac[4,2] + mem[4,2] = 0
 accuracy_0_34: - 5.09765625 frac[0,34] + mem[0,34] = 0
 accuracy_5_3: - 5.09765625 frac[5,3] + mem[5,3] = 0
 accuracy_12_66: - 5.09765625 frac[12,66] + mem[12,66] = 0
 accuracy_9_64: - 5.09765625 frac[9,64] + mem[9,64] = 0
 accuracy_3_47: - 5.09765625 frac[3,47] + mem[3,47] = 0
 accuracy_6_8: - 5.09765625 frac[6,8] + mem[6,8] = 0
 accuracy_4_40: - 5.09765625 frac[4,40] + mem[4,40] = 0
 accuracy_2_40: - 5.09765625 frac[2,40] + mem[2,40] = 0
 accuracy_8_43: - 5.09765625 frac[8,43] + mem[8,43] = 0
 accuracy_7_13: - 5.09765625 frac[7,13] + mem[7,13] = 0
 accuracy_5_45: - 5.09765625 frac[5,45] + mem[5,45] = 0
 accuracy_3_53: - 5.09765625 frac[3,53] + mem[3,53] = 0
 accuracy_9_42: - 5.09765625 frac[9,42] + mem[9,42] = 0
 accuracy_6_34: - 5.09765625 frac[6,34] + mem[6,34] = 0
 accuracy_4_54: - 5.09765625 frac[4,54] + mem[4,54] = 0
 accuracy_10_33: - 5.09765625 frac[10,33] + mem[10,33] = 0
 accuracy_2_66: - 5.09765625 frac[2,66] + mem[2,66] = 0
 accuracy_7_43: - 5.09765625 frac[7,43] + mem[7,43] = 0
 accuracy_5_55: - 5.09765625 frac[5,55] + mem[5,55] = 0
 accuracy_11_44: - 5.09765625 frac[11,44] + mem[11,44] = 0
 accuracy_9_12: - 5.09765625 frac[9,12] + mem[9,12] = 0
 accuracy_6_68: - 5.09765625 frac[6,68] + mem[6,68] = 0
 accuracy_12_47: - 5.09765625 frac[12,47] + mem[12,47] = 0
 accuracy_3_66: - 5.09765625 frac[3,66] + mem[3,66] = 0
 accuracy_10_11: - 5.09765625 frac[10,11] + mem[10,11] = 0
 accuracy_8_23: - 5.09765625 frac[8,23] + mem[8,23] = 0
 accuracy_13_46: - 5.09765625 frac[13,46] + mem[13,46] = 0
 accuracy_11_2: - 5.09765625 frac[11,2] + mem[11,2] = 0
 accuracy_9_22: - 5.09765625 frac[9,22] + mem[9,22] = 0
 accuracy_14_13: - 5.09765625 frac[14,13] + mem[14,13] = 0
 accuracy_12_5: - 5.09765625 frac[12,5] + mem[12,5] = 0
 accuracy_10_21: - 5.09765625 frac[10,21] + mem[10,21] = 0
 accuracy_13_8: - 5.09765625 frac[13,8] + mem[13,8] = 0
 accuracy_11_24: - 5.09765625 frac[11,24] + mem[11,24] = 0
 accuracy_0_3: - 5.09765625 frac[0,3] + mem[0,3] = 0
 accuracy_14_55: - 5.09765625 frac[14,55] + mem[14,55] = 0
 accuracy_3_14: - 5.09765625 frac[3,14] + mem[3,14] = 0
 accuracy_1_2: - 5.09765625 frac[1,2] + mem[1,2] = 0
 accuracy_8_67: - 5.09765625 frac[8,67] + mem[8,67] = 0
 accuracy_4_9: - 5.09765625 frac[4,9] + mem[4,9] = 0
 accuracy_2_9: - 5.09765625 frac[2,9] + mem[2,9] = 0
 accuracy_0_57: - 5.09765625 frac[0,57] + mem[0,57] = 0
 accuracy_5_12: - 5.09765625 frac[5,12] + mem[5,12] = 0
 accuracy_3_20: - 5.09765625 frac[3,20] + mem[3,20] = 0
 accuracy_9_73: - 5.09765625 frac[9,73] + mem[9,73] = 0
 accuracy_6_3: - 5.09765625 frac[6,3] + mem[6,3] = 0
 accuracy_2_35: - 5.09765625 frac[2,35] + mem[2,35] = 0
 accuracy_7_10: - 5.09765625 frac[7,10] + mem[7,10] = 0
 accuracy_5_22: - 5.09765625 frac[5,22] + mem[5,22] = 0
 accuracy_11_79: - 5.09765625 frac[11,79] + mem[11,79] = 0
 accuracy_3_58: - 5.09765625 frac[3,58] + mem[3,58] = 0
 accuracy_14_67: - 5.09765625 frac[14,67] + mem[14,67] = 0
 accuracy_6_37: - 5.09765625 frac[6,37] + mem[6,37] = 0
 accuracy_4_61: - 5.09765625 frac[4,61] + mem[4,61] = 0
 accuracy_10_40: - 5.09765625 frac[10,40] + mem[10,40] = 0
 accuracy_8_56: - 5.09765625 frac[8,56] + mem[8,56] = 0
 accuracy_13_77: - 5.09765625 frac[13,77] + mem[13,77] = 0
 accuracy_5_48: - 5.09765625 frac[5,48] + mem[5,48] = 0
 accuracy_11_37: - 5.09765625 frac[11,37] + mem[11,37] = 0
 accuracy_9_53: - 5.09765625 frac[9,53] + mem[9,53] = 0
 accuracy_6_63: - 5.09765625 frac[6,63] + mem[6,63] = 0
 accuracy_12_38: - 5.09765625 frac[12,38] + mem[12,38] = 0
 accuracy_3_75: - 5.09765625 frac[3,75] + mem[3,75] = 0
 accuracy_10_50: - 5.09765625 frac[10,50] + mem[10,50] = 0
 accuracy_1_79: - 5.09765625 frac[1,79] + mem[1,79] = 0
 accuracy_8_30: - 5.09765625 frac[8,30] + mem[8,30] = 0
 accuracy_7_62: - 5.09765625 frac[7,62] + mem[7,62] = 0
 accuracy_13_39: - 5.09765625 frac[13,39] + mem[13,39] = 0
 accuracy_11_59: - 5.09765625 frac[11,59] + mem[11,59] = 0
 accuracy_14_20: - 5.09765625 frac[14,20] + mem[14,20] = 0
 accuracy_5_65: - 5.09765625 frac[5,65] + mem[5,65] = 0
 accuracy_12_60: - 5.09765625 frac[12,60] + mem[12,60] = 0
 accuracy_10_28: - 5.09765625 frac[10,28] + mem[10,28] = 0
 accuracy_13_1: - 5.09765625 frac[13,1] + mem[13,1] = 0
 accuracy_11_17: - 5.09765625 frac[11,17] + mem[11,17] = 0
 accuracy_7_79: - 5.09765625 frac[7,79] + mem[7,79] = 0
 accuracy_14_62: - 5.09765625 frac[14,62] + mem[14,62] = 0
 accuracy_12_26: - 5.09765625 frac[12,26] + mem[12,26] = 0
 accuracy_1_27: - 5.09765625 frac[1,27] + mem[1,27] = 0
 accuracy_8_74: - 5.09765625 frac[8,74] + mem[8,74] = 0
 accuracy_13_27: - 5.09765625 frac[13,27] + mem[13,27] = 0
 accuracy_2_0: - 5.09765625 frac[2,0] + mem[2,0] = 0
 accuracy_0_48: - 5.09765625 frac[0,48] + mem[0,48] = 0
 accuracy_14_32: - 5.09765625 frac[14,32] + mem[14,32] = 0
 accuracy_3_29: - 5.09765625 frac[3,29] + mem[3,29] = 0
 accuracy_1_61: - 5.09765625 frac[1,61] + mem[1,61] = 0
 mem_tot_CPU1: - mem[0,17] - mem[0,55] - mem[0,67] - mem[0,14] - mem[0,5]
   - mem[0,59] - mem[0,28] - mem[0,40] - mem[0,78] - mem[0,19] - mem[0,9]
   - mem[0,0] - mem[0,38] - mem[0,7] - mem[0,61] - mem[0,73] - mem[0,30]
   - mem[0,42] - mem[0,11] - mem[0,33] - mem[0,71] - mem[0,2] - mem[0,56]
   - mem[0,25] - mem[0,63] - mem[0,75] - mem[0,16] - mem[0,54] - mem[0,23]
   - mem[0,13] - mem[0,35] - mem[0,4] - mem[0,58] - mem[0,27] - mem[0,77]
   - mem[0,18] - mem[0,46] - mem[0,68] - mem[0,15] - mem[0,37] - mem[0,60]
   - mem[0,72] - mem[0,41] - mem[0,20] - mem[0,10] - mem[0,32] - mem[0,1]
   - mem[0,39] - mem[0,24] - mem[0,62] - mem[0,74] - mem[0,31] - mem[0,53]
   - mem[0,43] - mem[0,12] - mem[0,34] - mem[0,3] - mem[0,57] - mem[0,48]
   + mem_tot_CPU1 = 0
 R1381: - 3.05176e-05 mem_tot_CPU1 + normalized_mem_tot_CPU1 = 0
 rows_tot_CPU1: - 3 frac[0,17] - 3 frac[0,55] - 3 frac[0,67] - 3 frac[0,14]
   - 3 frac[0,5] - 3 frac[0,59] - 3 frac[0,28] - 3 frac[0,40]
   - 3 frac[0,78] - 3 frac[0,19] - 3 frac[0,9] - 3 frac[0,0] - 3 frac[0,38]
   - 3 frac[0,7] - 3 frac[0,61] - 3 frac[0,73] - 3 frac[0,30]
   - 3 frac[0,42] - 3 frac[0,11] - 3 frac[0,33] - 3 frac[0,71]
   - 3 frac[0,2] - 3 frac[0,56] - 3 frac[0,25] - 3 frac[0,63]
   - 3 frac[0,75] - 3 frac[0,16] - 3 frac[0,54] - 3 frac[0,23]
   - 3 frac[0,13] - 3 frac[0,35] - 3 frac[0,4] - 3 frac[0,58]
   - 3 frac[0,27] - 3 frac[0,77] - 3 frac[0,18] - 3 frac[0,46]
   - 3 frac[0,68] - 3 frac[0,15] - 3 frac[0,37] - 3 frac[0,60]
   - 3 frac[0,72] - 3 frac[0,41] - 3 frac[0,20] - 3 frac[0,10]
   - 3 frac[0,32] - 3 frac[0,1] - 3 frac[0,39] - 3 frac[0,24]
   - 3 frac[0,62] - 3 frac[0,74] - 3 frac[0,31] - 3 frac[0,53]
   - 3 frac[0,43] - 3 frac[0,12] - 3 frac[0,34] - 3 frac[0,3]
   - 3 frac[0,57] - 3 frac[0,48] + rows_tot_CPU1 = 0
 R1383: - 0.0833333333333333 rows_tot_CPU1 + normalized_rows_tot_CPU1 = 0
 mem_tot_CPU2: - mem[1,64] - mem[1,28] - mem[1,54] - mem[1,15] - mem[1,33]
   - mem[1,0] - mem[1,58] - mem[1,77] - mem[1,25] - mem[1,70] - mem[1,18]
   - mem[1,52] - mem[1,46] - mem[1,13] - mem[1,39] - mem[1,74] - mem[1,6]
   - mem[1,56] - mem[1,67] - mem[1,43] - mem[1,68] - mem[1,44] - mem[1,3]
   - mem[1,37] - mem[1,72] - mem[1,65] - mem[1,55] - mem[1,41] - mem[1,22]
   - mem[1,34] - mem[1,26] - mem[1,60] - mem[1,19] - mem[1,53] - mem[1,47]
   - mem[1,20] - mem[1,14] - mem[1,7] - mem[1,57] - mem[1,50] - mem[1,17]
   - mem[1,11] - mem[1,12] - mem[1,38] - mem[1,73] - mem[1,63] - mem[1,66]
   - mem[1,30] - mem[1,48] - mem[1,23] - mem[1,2] - mem[1,79] - mem[1,27]
   - mem[1,61] + mem_tot_CPU2 = 0
 R1385: - 3.05176e-05 mem_tot_CPU2 + normalized_mem_tot_CPU2 = 0
 rows_tot_CPU2: - 3 frac[1,64] - 3 frac[1,28] - 3 frac[1,54] - 3 frac[1,15]
   - 3 frac[1,33] - 3 frac[1,0] - 3 frac[1,58] - 3 frac[1,77]
   - 3 frac[1,25] - 3 frac[1,70] - 3 frac[1,18] - 3 frac[1,52]
   - 3 frac[1,46] - 3 frac[1,13] - 3 frac[1,39] - 3 frac[1,74]
   - 3 frac[1,6] - 3 frac[1,56] - 3 frac[1,67] - 3 frac[1,43]
   - 3 frac[1,68] - 3 frac[1,44] - 3 frac[1,3] - 3 frac[1,37]
   - 3 frac[1,72] - 3 frac[1,65] - 3 frac[1,55] - 3 frac[1,41]
   - 3 frac[1,22] - 3 frac[1,34] - 3 frac[1,26] - 3 frac[1,60]
   - 3 frac[1,19] - 3 frac[1,53] - 3 frac[1,47] - 3 frac[1,20]
   - 3 frac[1,14] - 3 frac[1,7] - 3 frac[1,57] - 3 frac[1,50]
   - 3 frac[1,17] - 3 frac[1,11] - 3 frac[1,12] - 3 frac[1,38]
   - 3 frac[1,73] - 3 frac[1,63] - 3 frac[1,66] - 3 frac[1,30]
   - 3 frac[1,48] - 3 frac[1,23] - 3 frac[1,2] - 3 frac[1,79]
   - 3 frac[1,27] - 3 frac[1,61] + rows_tot_CPU2 = 0
 R1387: - 0.0833333333333333 rows_tot_CPU2 + normalized_rows_tot_CPU2 = 0
 mem_tot_CPU3: - mem[2,78] - mem[2,27] - mem[2,53] - mem[2,47] - mem[2,73]
   - mem[2,18] - mem[2,12] - mem[2,38] - mem[2,64] - mem[2,7] - mem[2,33]
   - mem[2,56] - mem[2,76] - mem[2,25] - mem[2,16] - mem[2,36] - mem[2,63]
   - mem[2,28] - mem[2,54] - mem[2,49] - mem[2,43] - mem[2,69] - mem[2,8]
   - mem[2,34] - mem[2,3] - mem[2,26] - mem[2,46] - mem[2,72] - mem[2,21]
   - mem[2,15] - mem[2,67] - mem[2,59] - mem[2,79] - mem[2,50] - mem[2,70]
   - mem[2,19] - mem[2,13] - mem[2,39] - mem[2,4] - mem[2,62] - mem[2,57]
   - mem[2,77] - mem[2,22] - mem[2,42] - mem[2,68] - mem[2,11] - mem[2,37]
   - mem[2,2] - mem[2,60] - mem[2,29] - mem[2,55] - mem[2,40] - mem[2,66]
   - mem[2,9] - mem[2,35] - mem[2,0] + mem_tot_CPU3 = 0
 R1389: - 3.05176e-05 mem_tot_CPU3 + normalized_mem_tot_CPU3 = 0
 rows_tot_CPU3: - 3 frac[2,78] - 3 frac[2,27] - 3 frac[2,53] - 3 frac[2,47]
   - 3 frac[2,73] - 3 frac[2,18] - 3 frac[2,12] - 3 frac[2,38]
   - 3 frac[2,64] - 3 frac[2,7] - 3 frac[2,33] - 3 frac[2,56]
   - 3 frac[2,76] - 3 frac[2,25] - 3 frac[2,16] - 3 frac[2,36]
   - 3 frac[2,63] - 3 frac[2,28] - 3 frac[2,54] - 3 frac[2,49]
   - 3 frac[2,43] - 3 frac[2,69] - 3 frac[2,8] - 3 frac[2,34] - 3 frac[2,3]
   - 3 frac[2,26] - 3 frac[2,46] - 3 frac[2,72] - 3 frac[2,21]
   - 3 frac[2,15] - 3 frac[2,67] - 3 frac[2,59] - 3 frac[2,79]
   - 3 frac[2,50] - 3 frac[2,70] - 3 frac[2,19] - 3 frac[2,13]
   - 3 frac[2,39] - 3 frac[2,4] - 3 frac[2,62] - 3 frac[2,57]
   - 3 frac[2,77] - 3 frac[2,22] - 3 frac[2,42] - 3 frac[2,68]
   - 3 frac[2,11] - 3 frac[2,37] - 3 frac[2,2] - 3 frac[2,60]
   - 3 frac[2,29] - 3 frac[2,55] - 3 frac[2,40] - 3 frac[2,66]
   - 3 frac[2,9] - 3 frac[2,35] - 3 frac[2,0] + rows_tot_CPU3 = 0
 R1391: - 0.0833333333333333 rows_tot_CPU3 + normalized_rows_tot_CPU3 = 0
 mem_tot_CPU4: - mem[3,35] - mem[3,2] - mem[3,11] - mem[3,55] - mem[3,68]
   - mem[3,60] - mem[3,37] - mem[3,4] - mem[3,42] - mem[3,65] - mem[3,13]
   - mem[3,57] - mem[3,70] - mem[3,24] - mem[3,62] - mem[3,79] - mem[3,1]
   - mem[3,6] - mem[3,44] - mem[3,50] - mem[3,67] - mem[3,15] - mem[3,59]
   - mem[3,72] - mem[3,26] - mem[3,3] - mem[3,41] - mem[3,46] - mem[3,61]
   - mem[3,74] - mem[3,28] - mem[3,34] - mem[3,43] - mem[3,49] - mem[3,16]
   - mem[3,71] - mem[3,63] - mem[3,76] - mem[3,30] - mem[3,7] - mem[3,45]
   - mem[3,51] - mem[3,64] - mem[3,18] - mem[3,73] - mem[3,33] - mem[3,0]
   - mem[3,38] - mem[3,9] - mem[3,47] - mem[3,53] - mem[3,66] - mem[3,14]
   - mem[3,20] - mem[3,58] - mem[3,75] - mem[3,29] + mem_tot_CPU4 = 0
 R1393: - 3.05176e-05 mem_tot_CPU4 + normalized_mem_tot_CPU4 = 0
 rows_tot_CPU4: - 3 frac[3,35] - 3 frac[3,2] - 3 frac[3,11] - 3 frac[3,55]
   - 3 frac[3,68] - 3 frac[3,60] - 3 frac[3,37] - 3 frac[3,4]
   - 3 frac[3,42] - 3 frac[3,65] - 3 frac[3,13] - 3 frac[3,57]
   - 3 frac[3,70] - 3 frac[3,24] - 3 frac[3,62] - 3 frac[3,79]
   - 3 frac[3,1] - 3 frac[3,6] - 3 frac[3,44] - 3 frac[3,50] - 3 frac[3,67]
   - 3 frac[3,15] - 3 frac[3,59] - 3 frac[3,72] - 3 frac[3,26]
   - 3 frac[3,3] - 3 frac[3,41] - 3 frac[3,46] - 3 frac[3,61]
   - 3 frac[3,74] - 3 frac[3,28] - 3 frac[3,34] - 3 frac[3,43]
   - 3 frac[3,49] - 3 frac[3,16] - 3 frac[3,71] - 3 frac[3,63]
   - 3 frac[3,76] - 3 frac[3,30] - 3 frac[3,7] - 3 frac[3,45]
   - 3 frac[3,51] - 3 frac[3,64] - 3 frac[3,18] - 3 frac[3,73]
   - 3 frac[3,33] - 3 frac[3,0] - 3 frac[3,38] - 3 frac[3,9] - 3 frac[3,47]
   - 3 frac[3,53] - 3 frac[3,66] - 3 frac[3,14] - 3 frac[3,20]
   - 3 frac[3,58] - 3 frac[3,75] - 3 frac[3,29] + rows_tot_CPU4 = 0
 R1395: - 0.0833333333333333 rows_tot_CPU4 + normalized_rows_tot_CPU4 = 0
 mem_tot_CPU5: - mem[4,36] - mem[4,66] - mem[4,5] - mem[4,35] - mem[4,12]
   - mem[4,42] - mem[4,11] - mem[4,17] - mem[4,69] - mem[4,24] - mem[4,38]
   - mem[4,76] - mem[4,7] - mem[4,45] - mem[4,75] - mem[4,14] - mem[4,20]
   - mem[4,19] - mem[4,57] - mem[4,71] - mem[4,26] - mem[4,78] - mem[4,1]
   - mem[4,47] - mem[4,53] - mem[4,22] - mem[4,29] - mem[4,59] - mem[4,65]
   - mem[4,4] - mem[4,34] - mem[4,3] - mem[4,41] - mem[4,55] - mem[4,10]
   - mem[4,16] - mem[4,62] - mem[4,68] - mem[4,37] - mem[4,67] - mem[4,6]
   - mem[4,13] - mem[4,43] - mem[4,49] - mem[4,77] - mem[4,0] - mem[4,46]
   - mem[4,52] - mem[4,15] - mem[4,21] - mem[4,51] - mem[4,28] - mem[4,64]
   - mem[4,27] - mem[4,33] - mem[4,79] - mem[4,2] - mem[4,40] - mem[4,54]
   - mem[4,9] - mem[4,61] + mem_tot_CPU5 = 0
 R1397: - 3.05176e-05 mem_tot_CPU5 + normalized_mem_tot_CPU5 = 0
 rows_tot_CPU5: - 3 frac[4,36] - 3 frac[4,66] - 3 frac[4,5] - 3 frac[4,35]
   - 3 frac[4,12] - 3 frac[4,42] - 3 frac[4,11] - 3 frac[4,17]
   - 3 frac[4,69] - 3 frac[4,24] - 3 frac[4,38] - 3 frac[4,76]
   - 3 frac[4,7] - 3 frac[4,45] - 3 frac[4,75] - 3 frac[4,14]
   - 3 frac[4,20] - 3 frac[4,19] - 3 frac[4,57] - 3 frac[4,71]
   - 3 frac[4,26] - 3 frac[4,78] - 3 frac[4,1] - 3 frac[4,47]
   - 3 frac[4,53] - 3 frac[4,22] - 3 frac[4,29] - 3 frac[4,59]
   - 3 frac[4,65] - 3 frac[4,4] - 3 frac[4,34] - 3 frac[4,3] - 3 frac[4,41]
   - 3 frac[4,55] - 3 frac[4,10] - 3 frac[4,16] - 3 frac[4,62]
   - 3 frac[4,68] - 3 frac[4,37] - 3 frac[4,67] - 3 frac[4,6]
   - 3 frac[4,13] - 3 frac[4,43] - 3 frac[4,49] - 3 frac[4,77]
   - 3 frac[4,0] - 3 frac[4,46] - 3 frac[4,52] - 3 frac[4,15]
   - 3 frac[4,21] - 3 frac[4,51] - 3 frac[4,28] - 3 frac[4,64]
   - 3 frac[4,27] - 3 frac[4,33] - 3 frac[4,79] - 3 frac[4,2]
   - 3 frac[4,40] - 3 frac[4,54] - 3 frac[4,9] - 3 frac[4,61]
   + rows_tot_CPU5 = 0
 R1399: - 0.0833333333333333 rows_tot_CPU5 + normalized_rows_tot_CPU5 = 0
 mem_tot_CPU6: - mem[5,57] - mem[5,34] - mem[5,1] - mem[5,43] - mem[5,70]
   - mem[5,10] - mem[5,20] - mem[5,62] - mem[5,29] - mem[5,39] - mem[5,6]
   - mem[5,32] - mem[5,15] - mem[5,8] - mem[5,77] - mem[5,27] - mem[5,37]
   - mem[5,46] - mem[5,13] - mem[5,66] - mem[5,16] - mem[5,75] - mem[5,25]
   - mem[5,2] - mem[5,44] - mem[5,54] - mem[5,71] - mem[5,11] - mem[5,63]
   - mem[5,64] - mem[5,30] - mem[5,56] - mem[5,7] - mem[5,33] - mem[5,42]
   - mem[5,52] - mem[5,9] - mem[5,19] - mem[5,61] - mem[5,78] - mem[5,38]
   - mem[5,5] - mem[5,47] - mem[5,40] - mem[5,50] - mem[5,67] - mem[5,59]
   - mem[5,3] - mem[5,45] - mem[5,55] - mem[5,12] - mem[5,22] - mem[5,48]
   - mem[5,65] + mem_tot_CPU6 = 0
 R1401: - 3.05176e-05 mem_tot_CPU6 + normalized_mem_tot_CPU6 = 0
 rows_tot_CPU6: - 3 frac[5,57] - 3 frac[5,34] - 3 frac[5,1] - 3 frac[5,43]
   - 3 frac[5,70] - 3 frac[5,10] - 3 frac[5,20] - 3 frac[5,62]
   - 3 frac[5,29] - 3 frac[5,39] - 3 frac[5,6] - 3 frac[5,32]
   - 3 frac[5,15] - 3 frac[5,8] - 3 frac[5,77] - 3 frac[5,27]
   - 3 frac[5,37] - 3 frac[5,46] - 3 frac[5,13] - 3 frac[5,66]
   - 3 frac[5,16] - 3 frac[5,75] - 3 frac[5,25] - 3 frac[5,2]
   - 3 frac[5,44] - 3 frac[5,54] - 3 frac[5,71] - 3 frac[5,11]
   - 3 frac[5,63] - 3 frac[5,64] - 3 frac[5,30] - 3 frac[5,56]
   - 3 frac[5,7] - 3 frac[5,33] - 3 frac[5,42] - 3 frac[5,52] - 3 frac[5,9]
   - 3 frac[5,19] - 3 frac[5,61] - 3 frac[5,78] - 3 frac[5,38]
   - 3 frac[5,5] - 3 frac[5,47] - 3 frac[5,40] - 3 frac[5,50]
   - 3 frac[5,67] - 3 frac[5,59] - 3 frac[5,3] - 3 frac[5,45]
   - 3 frac[5,55] - 3 frac[5,12] - 3 frac[5,22] - 3 frac[5,48]
   - 3 frac[5,65] + rows_tot_CPU6 = 0
 R1403: - 0.0833333333333333 rows_tot_CPU6 + normalized_rows_tot_CPU6 = 0
 mem_tot_CPU7: - mem[6,23] - mem[6,67] - mem[6,14] - mem[6,32] - mem[6,58]
   - mem[6,1] - mem[6,27] - mem[6,61] - mem[6,18] - mem[6,78] - mem[6,65]
   - mem[6,12] - mem[6,56] - mem[6,7] - mem[6,16] - mem[6,45] - mem[6,71]
   - mem[6,2] - mem[6,36] - mem[6,5] - mem[6,49] - mem[6,22] - mem[6,66]
   - mem[6,9] - mem[6,69] - mem[6,0] - mem[6,26] - mem[6,60] - mem[6,29]
   - mem[6,73] - mem[6,20] - mem[6,64] - mem[6,15] - mem[6,33] - mem[6,59]
   - mem[6,6] - mem[6,24] - mem[6,50] - mem[6,19] - mem[6,53] - mem[6,70]
   - mem[6,13] - mem[6,30] - mem[6,48] - mem[6,74] - mem[6,17] - mem[6,43]
   - mem[6,77] - mem[6,8] - mem[6,34] - mem[6,68] - mem[6,3] - mem[6,37]
   - mem[6,63] + mem_tot_CPU7 = 0
 R1405: - 3.05176e-05 mem_tot_CPU7 + normalized_mem_tot_CPU7 = 0
 rows_tot_CPU7: - 3 frac[6,23] - 3 frac[6,67] - 3 frac[6,14] - 3 frac[6,32]
   - 3 frac[6,58] - 3 frac[6,1] - 3 frac[6,27] - 3 frac[6,61]
   - 3 frac[6,18] - 3 frac[6,78] - 3 frac[6,65] - 3 frac[6,12]
   - 3 frac[6,56] - 3 frac[6,7] - 3 frac[6,16] - 3 frac[6,45]
   - 3 frac[6,71] - 3 frac[6,2] - 3 frac[6,36] - 3 frac[6,5] - 3 frac[6,49]
   - 3 frac[6,22] - 3 frac[6,66] - 3 frac[6,9] - 3 frac[6,69] - 3 frac[6,0]
   - 3 frac[6,26] - 3 frac[6,60] - 3 frac[6,29] - 3 frac[6,73]
   - 3 frac[6,20] - 3 frac[6,64] - 3 frac[6,15] - 3 frac[6,33]
   - 3 frac[6,59] - 3 frac[6,6] - 3 frac[6,24] - 3 frac[6,50]
   - 3 frac[6,19] - 3 frac[6,53] - 3 frac[6,70] - 3 frac[6,13]
   - 3 frac[6,30] - 3 frac[6,48] - 3 frac[6,74] - 3 frac[6,17]
   - 3 frac[6,43] - 3 frac[6,77] - 3 frac[6,8] - 3 frac[6,34]
   - 3 frac[6,68] - 3 frac[6,3] - 3 frac[6,37] - 3 frac[6,63]
   + rows_tot_CPU7 = 0
 R1407: - 0.0833333333333333 rows_tot_CPU7 + normalized_rows_tot_CPU7 = 0
 mem_tot_CPU8: - mem[7,3] - mem[7,25] - mem[7,55] - mem[7,68] - mem[7,22]
   - mem[7,44] - mem[7,15] - mem[7,37] - mem[7,4] - mem[7,34] - mem[7,56]
   - mem[7,73] - mem[7,27] - mem[7,49] - mem[7,70] - mem[7,46] - mem[7,9]
   - mem[7,6] - mem[7,28] - mem[7,58] - mem[7,75] - mem[7,64] - mem[7,40]
   - mem[7,11] - mem[7,33] - mem[7,63] - mem[7,76] - mem[7,0] - mem[7,30]
   - mem[7,45] - mem[7,66] - mem[7,12] - mem[7,5] - mem[7,35] - mem[7,57]
   - mem[7,78] - mem[7,2] - mem[7,24] - mem[7,54] - mem[7,71] - mem[7,17]
   - mem[7,14] - mem[7,36] - mem[7,7] - mem[7,29] - mem[7,26] - mem[7,48]
   - mem[7,65] - mem[7,19] - mem[7,41] - mem[7,38] - mem[7,77] - mem[7,1]
   - mem[7,31] - mem[7,53] - mem[7,74] - mem[7,67] - mem[7,13] - mem[7,43]
   - mem[7,10] - mem[7,62] - mem[7,79] + mem_tot_CPU8 = 0
 R1409: - 3.05176e-05 mem_tot_CPU8 + normalized_mem_tot_CPU8 = 0
 rows_tot_CPU8: - 3 frac[7,3] - 3 frac[7,25] - 3 frac[7,55] - 3 frac[7,68]
   - 3 frac[7,22] - 3 frac[7,44] - 3 frac[7,15] - 3 frac[7,37]
   - 3 frac[7,4] - 3 frac[7,34] - 3 frac[7,56] - 3 frac[7,73]
   - 3 frac[7,27] - 3 frac[7,49] - 3 frac[7,70] - 3 frac[7,46]
   - 3 frac[7,9] - 3 frac[7,6] - 3 frac[7,28] - 3 frac[7,58] - 3 frac[7,75]
   - 3 frac[7,64] - 3 frac[7,40] - 3 frac[7,11] - 3 frac[7,33]
   - 3 frac[7,63] - 3 frac[7,76] - 3 frac[7,0] - 3 frac[7,30]
   - 3 frac[7,45] - 3 frac[7,66] - 3 frac[7,12] - 3 frac[7,5]
   - 3 frac[7,35] - 3 frac[7,57] - 3 frac[7,78] - 3 frac[7,2]
   - 3 frac[7,24] - 3 frac[7,54] - 3 frac[7,71] - 3 frac[7,17]
   - 3 frac[7,14] - 3 frac[7,36] - 3 frac[7,7] - 3 frac[7,29]
   - 3 frac[7,26] - 3 frac[7,48] - 3 frac[7,65] - 3 frac[7,19]
   - 3 frac[7,41] - 3 frac[7,38] - 3 frac[7,77] - 3 frac[7,1]
   - 3 frac[7,31] - 3 frac[7,53] - 3 frac[7,74] - 3 frac[7,67]
   - 3 frac[7,13] - 3 frac[7,43] - 3 frac[7,10] - 3 frac[7,62]
   - 3 frac[7,79] + rows_tot_CPU8 = 0
 R1411: - 0.0833333333333333 rows_tot_CPU8 + normalized_rows_tot_CPU8 = 0
 mem_tot_tor_P41: - mem[8,63] - mem[8,5] - mem[8,38] - mem[8,12]
   - mem[8,18] - mem[8,25] - mem[8,58] - mem[8,0] - mem[8,33] - mem[8,7]
   - mem[8,40] - mem[8,14] - mem[8,20] - mem[8,64] - mem[8,47] - mem[8,53]
   - mem[8,27] - mem[8,71] - mem[8,60] - mem[8,2] - mem[8,78] - mem[8,35]
   - mem[8,9] - mem[8,42] - mem[8,48] - mem[8,22] - mem[8,66] - mem[8,55]
   - mem[8,73] - mem[8,62] - mem[8,4] - mem[8,37] - mem[8,11] - mem[8,17]
   - mem[8,44] - mem[8,50] - mem[8,24] - mem[8,68] - mem[8,57] - mem[8,31]
   - mem[8,75] - mem[8,32] - mem[8,6] - mem[8,39] - mem[8,13] - mem[8,19]
   - mem[8,46] - mem[8,52] - mem[8,26] - mem[8,70] - mem[8,59] - mem[8,1]
   - mem[8,77] - mem[8,34] - mem[8,41] - mem[8,15] - mem[8,65] - mem[8,54]
   - mem[8,28] - mem[8,72] - mem[8,61] - mem[8,3] - mem[8,79] - mem[8,10]
   - mem[8,16] - mem[8,43] - mem[8,23] - mem[8,67] - mem[8,56] - mem[8,30]
   - mem[8,74] + mem_tot_tor_P41 = 0
 R1413: - 0.00173611111111111 mem_tot_tor_P41 + normalized_mem_tot_tor_P41
   = 0
 rows_tot_tor_P41: - 3 frac[8,63] - 3 frac[8,5] - 3 frac[8,38]
   - 3 frac[8,12] - 3 frac[8,18] - 3 frac[8,25] - 3 frac[8,58]
   - 3 frac[8,0] - 3 frac[8,33] - 3 frac[8,7] - 3 frac[8,40] - 3 frac[8,14]
   - 3 frac[8,20] - 3 frac[8,64] - 3 frac[8,47] - 3 frac[8,53]
   - 3 frac[8,27] - 3 frac[8,71] - 3 frac[8,60] - 3 frac[8,2]
   - 3 frac[8,78] - 3 frac[8,35] - 3 frac[8,9] - 3 frac[8,42]
   - 3 frac[8,48] - 3 frac[8,22] - 3 frac[8,66] - 3 frac[8,55]
   - 3 frac[8,73] - 3 frac[8,62] - 3 frac[8,4] - 3 frac[8,37]
   - 3 frac[8,11] - 3 frac[8,17] - 3 frac[8,44] - 3 frac[8,50]
   - 3 frac[8,24] - 3 frac[8,68] - 3 frac[8,57] - 3 frac[8,31]
   - 3 frac[8,75] - 3 frac[8,32] - 3 frac[8,6] - 3 frac[8,39]
   - 3 frac[8,13] - 3 frac[8,19] - 3 frac[8,46] - 3 frac[8,52]
   - 3 frac[8,26] - 3 frac[8,70] - 3 frac[8,59] - 3 frac[8,1]
   - 3 frac[8,77] - 3 frac[8,34] - 3 frac[8,41] - 3 frac[8,15]
   - 3 frac[8,65] - 3 frac[8,54] - 3 frac[8,28] - 3 frac[8,72]
   - 3 frac[8,61] - 3 frac[8,3] - 3 frac[8,79] - 3 frac[8,10]
   - 3 frac[8,16] - 3 frac[8,43] - 3 frac[8,23] - 3 frac[8,67]
   - 3 frac[8,56] - 3 frac[8,30] - 3 frac[8,74] + rows_tot_tor_P41 = 0
 R1415: - 0.0208333333333333 rows_tot_tor_P41 + normalized_rows_tot_tor_P41
   = 0
 mem_tot_tor_P42: - mem[9,62] - mem[9,0] - mem[9,39] - mem[9,9] - mem[9,19]
   - mem[9,70] - mem[9,40] - mem[9,50] - mem[9,20] - mem[9,79] - mem[9,59]
   - mem[9,29] - mem[9,60] - mem[9,6] - mem[9,67] - mem[9,37] - mem[9,15]
   - mem[9,68] - mem[9,46] - mem[9,26] - mem[9,77] - mem[9,57] - mem[9,3]
   - mem[9,34] - mem[9,4] - mem[9,65] - mem[9,43] - mem[9,13] - mem[9,74]
   - mem[9,44] - mem[9,54] - mem[9,24] - mem[9,63] - mem[9,1] - mem[9,71]
   - mem[9,41] - mem[9,51] - mem[9,21] - mem[9,72] - mem[9,30] - mem[9,61]
   - mem[9,7] - mem[9,38] - mem[9,8] - mem[9,18] - mem[9,69] - mem[9,47]
   - mem[9,49] - mem[9,27] - mem[9,78] - mem[9,58] - mem[9,28] - mem[9,35]
   - mem[9,66] - mem[9,36] - mem[9,14] - mem[9,16] - mem[9,75] - mem[9,45]
   - mem[9,55] - mem[9,25] - mem[9,76] - mem[9,56] - mem[9,2] - mem[9,33]
   - mem[9,11] - mem[9,64] - mem[9,42] - mem[9,12] - mem[9,22] - mem[9,73]
   - mem[9,53] + mem_tot_tor_P42 = 0
 R1417: - 0.00173611111111111 mem_tot_tor_P42 + normalized_mem_tot_tor_P42
   = 0
 rows_tot_tor_P42: - 3 frac[9,62] - 3 frac[9,0] - 3 frac[9,39]
   - 3 frac[9,9] - 3 frac[9,19] - 3 frac[9,70] - 3 frac[9,40]
   - 3 frac[9,50] - 3 frac[9,20] - 3 frac[9,79] - 3 frac[9,59]
   - 3 frac[9,29] - 3 frac[9,60] - 3 frac[9,6] - 3 frac[9,67]
   - 3 frac[9,37] - 3 frac[9,15] - 3 frac[9,68] - 3 frac[9,46]
   - 3 frac[9,26] - 3 frac[9,77] - 3 frac[9,57] - 3 frac[9,3]
   - 3 frac[9,34] - 3 frac[9,4] - 3 frac[9,65] - 3 frac[9,43]
   - 3 frac[9,13] - 3 frac[9,74] - 3 frac[9,44] - 3 frac[9,54]
   - 3 frac[9,24] - 3 frac[9,63] - 3 frac[9,1] - 3 frac[9,71]
   - 3 frac[9,41] - 3 frac[9,51] - 3 frac[9,21] - 3 frac[9,72]
   - 3 frac[9,30] - 3 frac[9,61] - 3 frac[9,7] - 3 frac[9,38] - 3 frac[9,8]
   - 3 frac[9,18] - 3 frac[9,69] - 3 frac[9,47] - 3 frac[9,49]
   - 3 frac[9,27] - 3 frac[9,78] - 3 frac[9,58] - 3 frac[9,28]
   - 3 frac[9,35] - 3 frac[9,66] - 3 frac[9,36] - 3 frac[9,14]
   - 3 frac[9,16] - 3 frac[9,75] - 3 frac[9,45] - 3 frac[9,55]
   - 3 frac[9,25] - 3 frac[9,76] - 3 frac[9,56] - 3 frac[9,2]
   - 3 frac[9,33] - 3 frac[9,11] - 3 frac[9,64] - 3 frac[9,42]
   - 3 frac[9,12] - 3 frac[9,22] - 3 frac[9,73] - 3 frac[9,53]
   + rows_tot_tor_P42 = 0
 R1419: - 0.0208333333333333 rows_tot_tor_P42 + normalized_rows_tot_tor_P42
   = 0
 mem_tot_tor_P43: - mem[10,61] - mem[10,64] - mem[10,7] - mem[10,75]
   - mem[10,36] - mem[10,14] - mem[10,47] - mem[10,9] - mem[10,19]
   - mem[10,71] - mem[10,48] - mem[10,26] - mem[10,78] - mem[10,59]
   - mem[10,5] - mem[10,34] - mem[10,12] - mem[10,22] - mem[10,45]
   - mem[10,55] - mem[10,17] - mem[10,69] - mem[10,62] - mem[10,24]
   - mem[10,76] - mem[10,57] - mem[10,3] - mem[10,32] - mem[10,10]
   - mem[10,20] - mem[10,43] - mem[10,53] - mem[10,67] - mem[10,6]
   - mem[10,39] - mem[10,1] - mem[10,46] - mem[10,8] - mem[10,70]
   - mem[10,41] - mem[10,51] - mem[10,29] - mem[10,65] - mem[10,4]
   - mem[10,37] - mem[10,15] - mem[10,44] - mem[10,54] - mem[10,16]
   - mem[10,68] - mem[10,49] - mem[10,27] - mem[10,79] - mem[10,56]
   - mem[10,2] - mem[10,35] - mem[10,13] - mem[10,42] - mem[10,52]
   - mem[10,30] - mem[10,66] - mem[10,63] - mem[10,25] - mem[10,77]
   - mem[10,38] - mem[10,0] - mem[10,33] - mem[10,11] - mem[10,21]
   - mem[10,40] - mem[10,50] - mem[10,28] + mem_tot_tor_P43 = 0
 R1421: - 0.00173611111111111 mem_tot_tor_P43 + normalized_mem_tot_tor_P43
   = 0
 rows_tot_tor_P43: - 3 frac[10,61] - 3 frac[10,64] - 3 frac[10,7]
   - 3 frac[10,75] - 3 frac[10,36] - 3 frac[10,14] - 3 frac[10,47]
   - 3 frac[10,9] - 3 frac[10,19] - 3 frac[10,71] - 3 frac[10,48]
   - 3 frac[10,26] - 3 frac[10,78] - 3 frac[10,59] - 3 frac[10,5]
   - 3 frac[10,34] - 3 frac[10,12] - 3 frac[10,22] - 3 frac[10,45]
   - 3 frac[10,55] - 3 frac[10,17] - 3 frac[10,69] - 3 frac[10,62]
   - 3 frac[10,24] - 3 frac[10,76] - 3 frac[10,57] - 3 frac[10,3]
   - 3 frac[10,32] - 3 frac[10,10] - 3 frac[10,20] - 3 frac[10,43]
   - 3 frac[10,53] - 3 frac[10,67] - 3 frac[10,6] - 3 frac[10,39]
   - 3 frac[10,1] - 3 frac[10,46] - 3 frac[10,8] - 3 frac[10,70]
   - 3 frac[10,41] - 3 frac[10,51] - 3 frac[10,29] - 3 frac[10,65]
   - 3 frac[10,4] - 3 frac[10,37] - 3 frac[10,15] - 3 frac[10,44]
   - 3 frac[10,54] - 3 frac[10,16] - 3 frac[10,68] - 3 frac[10,49]
   - 3 frac[10,27] - 3 frac[10,79] - 3 frac[10,56] - 3 frac[10,2]
   - 3 frac[10,35] - 3 frac[10,13] - 3 frac[10,42] - 3 frac[10,52]
   - 3 frac[10,30] - 3 frac[10,66] - 3 frac[10,63] - 3 frac[10,25]
   - 3 frac[10,77] - 3 frac[10,38] - 3 frac[10,0] - 3 frac[10,33]
   - 3 frac[10,11] - 3 frac[10,21] - 3 frac[10,40] - 3 frac[10,50]
   - 3 frac[10,28] + rows_tot_tor_P43 = 0
 R1423: - 0.0208333333333333 rows_tot_tor_P43 + normalized_rows_tot_tor_P43
   = 0
 mem_tot_tor_P44: - mem[11,68] - mem[11,48] - mem[11,22] - mem[11,41]
   - mem[11,15] - mem[11,46] - mem[11,4] - mem[11,26] - mem[11,65]
   - mem[11,61] - mem[11,19] - mem[11,70] - mem[11,50] - mem[11,8]
   - mem[11,43] - mem[11,1] - mem[11,74] - mem[11,32] - mem[11,6]
   - mem[11,28] - mem[11,67] - mem[11,63] - mem[11,10] - mem[11,45]
   - mem[11,3] - mem[11,25] - mem[11,76] - mem[11,34] - mem[11,56]
   - mem[11,30] - mem[11,69] - mem[11,49] - mem[11,23] - mem[11,54]
   - mem[11,12] - mem[11,73] - mem[11,5] - mem[11,27] - mem[11,78]
   - mem[11,36] - mem[11,58] - mem[11,16] - mem[11,71] - mem[11,9]
   - mem[11,40] - mem[11,14] - mem[11,75] - mem[11,33] - mem[11,7]
   - mem[11,29] - mem[11,64] - mem[11,38] - mem[11,60] - mem[11,18]
   - mem[11,53] - mem[11,11] - mem[11,0] - mem[11,77] - mem[11,35]
   - mem[11,57] - mem[11,31] - mem[11,66] - mem[11,62] - mem[11,20]
   - mem[11,55] - mem[11,13] - mem[11,44] - mem[11,2] - mem[11,24]
   - mem[11,79] - mem[11,37] - mem[11,59] - mem[11,17] + mem_tot_tor_P44
   = 0
 R1425: - 0.00173611111111111 mem_tot_tor_P44 + normalized_mem_tot_tor_P44
   = 0
 rows_tot_tor_P44: - 3 frac[11,68] - 3 frac[11,48] - 3 frac[11,22]
   - 3 frac[11,41] - 3 frac[11,15] - 3 frac[11,46] - 3 frac[11,4]
   - 3 frac[11,26] - 3 frac[11,65] - 3 frac[11,61] - 3 frac[11,19]
   - 3 frac[11,70] - 3 frac[11,50] - 3 frac[11,8] - 3 frac[11,43]
   - 3 frac[11,1] - 3 frac[11,74] - 3 frac[11,32] - 3 frac[11,6]
   - 3 frac[11,28] - 3 frac[11,67] - 3 frac[11,63] - 3 frac[11,10]
   - 3 frac[11,45] - 3 frac[11,3] - 3 frac[11,25] - 3 frac[11,76]
   - 3 frac[11,34] - 3 frac[11,56] - 3 frac[11,30] - 3 frac[11,69]
   - 3 frac[11,49] - 3 frac[11,23] - 3 frac[11,54] - 3 frac[11,12]
   - 3 frac[11,73] - 3 frac[11,5] - 3 frac[11,27] - 3 frac[11,78]
   - 3 frac[11,36] - 3 frac[11,58] - 3 frac[11,16] - 3 frac[11,71]
   - 3 frac[11,9] - 3 frac[11,40] - 3 frac[11,14] - 3 frac[11,75]
   - 3 frac[11,33] - 3 frac[11,7] - 3 frac[11,29] - 3 frac[11,64]
   - 3 frac[11,38] - 3 frac[11,60] - 3 frac[11,18] - 3 frac[11,53]
   - 3 frac[11,11] - 3 frac[11,0] - 3 frac[11,77] - 3 frac[11,35]
   - 3 frac[11,57] - 3 frac[11,31] - 3 frac[11,66] - 3 frac[11,62]
   - 3 frac[11,20] - 3 frac[11,55] - 3 frac[11,13] - 3 frac[11,44]
   - 3 frac[11,2] - 3 frac[11,24] - 3 frac[11,79] - 3 frac[11,37]
   - 3 frac[11,59] - 3 frac[11,17] + rows_tot_tor_P44 = 0
 R1427: - 0.0208333333333333 rows_tot_tor_P44 + normalized_rows_tot_tor_P44
   = 0
 mem_tot_l1_P41: - mem[12,78] - mem[12,59] - mem[12,17] - mem[12,69]
   - mem[12,50] - mem[12,8] - mem[12,41] - mem[12,7] - mem[12,29]
   - mem[12,32] - mem[12,62] - mem[12,20] - mem[12,72] - mem[12,53]
   - mem[12,19] - mem[12,71] - mem[12,44] - mem[12,10] - mem[12,43]
   - mem[12,1] - mem[12,31] - mem[12,34] - mem[12,56] - mem[12,22]
   - mem[12,74] - mem[12,55] - mem[12,13] - mem[12,65] - mem[12,46]
   - mem[12,4] - mem[12,37] - mem[12,3] - mem[12,25] - mem[12,77]
   - mem[12,58] - mem[12,16] - mem[12,68] - mem[12,49] - mem[12,15]
   - mem[12,67] - mem[12,40] - mem[12,6] - mem[12,39] - mem[12,61]
   - mem[12,27] - mem[12,79] - mem[12,52] - mem[12,18] - mem[12,70]
   - mem[12,51] - mem[12,9] - mem[12,42] - mem[12,0] - mem[12,30]
   - mem[12,33] - mem[12,63] - mem[12,21] - mem[12,73] - mem[12,54]
   - mem[12,12] - mem[12,64] - mem[12,45] - mem[12,11] - mem[12,36]
   - mem[12,2] - mem[12,24] - mem[12,76] - mem[12,35] - mem[12,57]
   - mem[12,23] - mem[12,75] - mem[12,48] - mem[12,14] - mem[12,66]
   - mem[12,47] - mem[12,5] - mem[12,38] - mem[12,60] - mem[12,26]
   + mem_tot_l1_P41 = 0
 R1429: - 0.00173611111111111 mem_tot_l1_P41 + normalized_mem_tot_l1_P41
   = 0
 rows_tot_l1_P41: - 3 frac[12,78] - 3 frac[12,59] - 3 frac[12,17]
   - 3 frac[12,69] - 3 frac[12,50] - 3 frac[12,8] - 3 frac[12,41]
   - 3 frac[12,7] - 3 frac[12,29] - 3 frac[12,32] - 3 frac[12,62]
   - 3 frac[12,20] - 3 frac[12,72] - 3 frac[12,53] - 3 frac[12,19]
   - 3 frac[12,71] - 3 frac[12,44] - 3 frac[12,10] - 3 frac[12,43]
   - 3 frac[12,1] - 3 frac[12,31] - 3 frac[12,34] - 3 frac[12,56]
   - 3 frac[12,22] - 3 frac[12,74] - 3 frac[12,55] - 3 frac[12,13]
   - 3 frac[12,65] - 3 frac[12,46] - 3 frac[12,4] - 3 frac[12,37]
   - 3 frac[12,3] - 3 frac[12,25] - 3 frac[12,77] - 3 frac[12,58]
   - 3 frac[12,16] - 3 frac[12,68] - 3 frac[12,49] - 3 frac[12,15]
   - 3 frac[12,67] - 3 frac[12,40] - 3 frac[12,6] - 3 frac[12,39]
   - 3 frac[12,61] - 3 frac[12,27] - 3 frac[12,79] - 3 frac[12,52]
   - 3 frac[12,18] - 3 frac[12,70] - 3 frac[12,51] - 3 frac[12,9]
   - 3 frac[12,42] - 3 frac[12,0] - 3 frac[12,30] - 3 frac[12,33]
   - 3 frac[12,63] - 3 frac[12,21] - 3 frac[12,73] - 3 frac[12,54]
   - 3 frac[12,12] - 3 frac[12,64] - 3 frac[12,45] - 3 frac[12,11]
   - 3 frac[12,36] - 3 frac[12,2] - 3 frac[12,24] - 3 frac[12,76]
   - 3 frac[12,35] - 3 frac[12,57] - 3 frac[12,23] - 3 frac[12,75]
   - 3 frac[12,48] - 3 frac[12,14] - 3 frac[12,66] - 3 frac[12,47]
   - 3 frac[12,5] - 3 frac[12,38] - 3 frac[12,60] - 3 frac[12,26]
   + rows_tot_l1_P41 = 0
 R1431: - 0.0208333333333333 rows_tot_l1_P41 + normalized_rows_tot_l1_P41
   = 0
 mem_tot_l1_P42: - mem[13,70] - mem[13,32] - mem[13,58] - mem[13,20]
   - mem[13,51] - mem[13,13] - mem[13,44] - mem[13,6] - mem[13,75]
   - mem[13,37] - mem[13,63] - mem[13,25] - mem[13,68] - mem[13,56]
   - mem[13,18] - mem[13,49] - mem[13,11] - mem[13,42] - mem[13,4]
   - mem[13,30] - mem[13,73] - mem[13,35] - mem[13,61] - mem[13,23]
   - mem[13,66] - mem[13,54] - mem[13,16] - mem[13,47] - mem[13,9]
   - mem[13,78] - mem[13,40] - mem[13,2] - mem[13,28] - mem[13,71]
   - mem[13,33] - mem[13,59] - mem[13,21] - mem[13,64] - mem[13,52]
   - mem[13,14] - mem[13,45] - mem[13,7] - mem[13,76] - mem[13,38]
   - mem[13,0] - mem[13,26] - mem[13,69] - mem[13,57] - mem[13,19]
   - mem[13,50] - mem[13,12] - mem[13,43] - mem[13,5] - mem[13,31]
   - mem[13,74] - mem[13,36] - mem[13,62] - mem[13,24] - mem[13,67]
   - mem[13,55] - mem[13,17] - mem[13,48] - mem[13,10] - mem[13,79]
   - mem[13,41] - mem[13,3] - mem[13,29] - mem[13,34] - mem[13,60]
   - mem[13,22] - mem[13,65] - mem[13,53] - mem[13,15] - mem[13,46]
   - mem[13,8] - mem[13,77] - mem[13,39] - mem[13,1] - mem[13,27]
   + mem_tot_l1_P42 = 0
 R1433: - 0.00173611111111111 mem_tot_l1_P42 + normalized_mem_tot_l1_P42
   = 0
 rows_tot_l1_P42: - 3 frac[13,70] - 3 frac[13,32] - 3 frac[13,58]
   - 3 frac[13,20] - 3 frac[13,51] - 3 frac[13,13] - 3 frac[13,44]
   - 3 frac[13,6] - 3 frac[13,75] - 3 frac[13,37] - 3 frac[13,63]
   - 3 frac[13,25] - 3 frac[13,68] - 3 frac[13,56] - 3 frac[13,18]
   - 3 frac[13,49] - 3 frac[13,11] - 3 frac[13,42] - 3 frac[13,4]
   - 3 frac[13,30] - 3 frac[13,73] - 3 frac[13,35] - 3 frac[13,61]
   - 3 frac[13,23] - 3 frac[13,66] - 3 frac[13,54] - 3 frac[13,16]
   - 3 frac[13,47] - 3 frac[13,9] - 3 frac[13,78] - 3 frac[13,40]
   - 3 frac[13,2] - 3 frac[13,28] - 3 frac[13,71] - 3 frac[13,33]
   - 3 frac[13,59] - 3 frac[13,21] - 3 frac[13,64] - 3 frac[13,52]
   - 3 frac[13,14] - 3 frac[13,45] - 3 frac[13,7] - 3 frac[13,76]
   - 3 frac[13,38] - 3 frac[13,0] - 3 frac[13,26] - 3 frac[13,69]
   - 3 frac[13,57] - 3 frac[13,19] - 3 frac[13,50] - 3 frac[13,12]
   - 3 frac[13,43] - 3 frac[13,5] - 3 frac[13,31] - 3 frac[13,74]
   - 3 frac[13,36] - 3 frac[13,62] - 3 frac[13,24] - 3 frac[13,67]
   - 3 frac[13,55] - 3 frac[13,17] - 3 frac[13,48] - 3 frac[13,10]
   - 3 frac[13,79] - 3 frac[13,41] - 3 frac[13,3] - 3 frac[13,29]
   - 3 frac[13,34] - 3 frac[13,60] - 3 frac[13,22] - 3 frac[13,65]
   - 3 frac[13,53] - 3 frac[13,15] - 3 frac[13,46] - 3 frac[13,8]
   - 3 frac[13,77] - 3 frac[13,39] - 3 frac[13,1] - 3 frac[13,27]
   + rows_tot_l1_P42 = 0
 R1435: - 0.0208333333333333 rows_tot_l1_P42 + normalized_rows_tot_l1_P42
   = 0
 mem_tot_l2_P4: - mem[14,74] - mem[14,31] - mem[14,1] - mem[14,43]
   - mem[14,77] - mem[14,8] - mem[14,50] - mem[14,19] - mem[14,53]
   - mem[14,65] - mem[14,26] - mem[14,60] - mem[14,38] - mem[14,29]
   - mem[14,7] - mem[14,41] - mem[14,14] - mem[14,48] - mem[14,17]
   - mem[14,59] - mem[14,71] - mem[14,24] - mem[14,2] - mem[14,36]
   - mem[14,78] - mem[14,5] - mem[14,47] - mem[14,12] - mem[14,54]
   - mem[14,66] - mem[14,23] - mem[14,57] - mem[14,69] - mem[14,30]
   - mem[14,0] - mem[14,42] - mem[14,76] - mem[14,11] - mem[14,45]
   - mem[14,18] - mem[14,52] - mem[14,64] - mem[14,21] - mem[14,63]
   - mem[14,33] - mem[14,75] - mem[14,6] - mem[14,40] - mem[14,9]
   - mem[14,51] - mem[14,16] - mem[14,58] - mem[14,70] - mem[14,27]
   - mem[14,61] - mem[14,39] - mem[14,73] - mem[14,4] - mem[14,46]
   - mem[14,15] - mem[14,49] - mem[14,22] - mem[14,56] - mem[14,34]
   - mem[14,68] - mem[14,25] - mem[14,3] - mem[14,37] - mem[14,79]
   - mem[14,10] - mem[14,44] - mem[14,13] - mem[14,55] - mem[14,67]
   - mem[14,20] - mem[14,62] - mem[14,32] + mem_tot_l2_P4 = 0
 R1437: - 0.00173611111111111 mem_tot_l2_P4 + normalized_mem_tot_l2_P4 = 0
 rows_tot_l2_P4: - 3 frac[14,74] - 3 frac[14,31] - 3 frac[14,1]
   - 3 frac[14,43] - 3 frac[14,77] - 3 frac[14,8] - 3 frac[14,50]
   - 3 frac[14,19] - 3 frac[14,53] - 3 frac[14,65] - 3 frac[14,26]
   - 3 frac[14,60] - 3 frac[14,38] - 3 frac[14,29] - 3 frac[14,7]
   - 3 frac[14,41] - 3 frac[14,14] - 3 frac[14,48] - 3 frac[14,17]
   - 3 frac[14,59] - 3 frac[14,71] - 3 frac[14,24] - 3 frac[14,2]
   - 3 frac[14,36] - 3 frac[14,78] - 3 frac[14,5] - 3 frac[14,47]
   - 3 frac[14,12] - 3 frac[14,54] - 3 frac[14,66] - 3 frac[14,23]
   - 3 frac[14,57] - 3 frac[14,69] - 3 frac[14,30] - 3 frac[14,0]
   - 3 frac[14,42] - 3 frac[14,76] - 3 frac[14,11] - 3 frac[14,45]
   - 3 frac[14,18] - 3 frac[14,52] - 3 frac[14,64] - 3 frac[14,21]
   - 3 frac[14,63] - 3 frac[14,33] - 3 frac[14,75] - 3 frac[14,6]
   - 3 frac[14,40] - 3 frac[14,9] - 3 frac[14,51] - 3 frac[14,16]
   - 3 frac[14,58] - 3 frac[14,70] - 3 frac[14,27] - 3 frac[14,61]
   - 3 frac[14,39] - 3 frac[14,73] - 3 frac[14,4] - 3 frac[14,46]
   - 3 frac[14,15] - 3 frac[14,49] - 3 frac[14,22] - 3 frac[14,56]
   - 3 frac[14,34] - 3 frac[14,68] - 3 frac[14,25] - 3 frac[14,3]
   - 3 frac[14,37] - 3 frac[14,79] - 3 frac[14,10] - 3 frac[14,44]
   - 3 frac[14,13] - 3 frac[14,55] - 3 frac[14,67] - 3 frac[14,20]
   - 3 frac[14,62] - 3 frac[14,32] + rows_tot_l2_P4 = 0
 R1439: - 0.0208333333333333 rows_tot_l2_P4 + normalized_rows_tot_l2_P4
   = 0
 row_capacity_CPU1: rows_tot_CPU1 <= 12
 row_capacity_CPU2: rows_tot_CPU2 <= 12
 row_capacity_CPU3: rows_tot_CPU3 <= 12
 row_capacity_CPU4: rows_tot_CPU4 <= 12
 row_capacity_CPU5: rows_tot_CPU5 <= 12
 row_capacity_CPU6: rows_tot_CPU6 <= 12
 row_capacity_CPU7: rows_tot_CPU7 <= 12
 row_capacity_CPU8: rows_tot_CPU8 <= 12
 row_capacity_tor_P41: rows_tot_tor_P41 <= 48
 row_capacity_tor_P42: rows_tot_tor_P42 <= 48
 row_capacity_tor_P43: rows_tot_tor_P43 <= 48
 row_capacity_tor_P44: rows_tot_tor_P44 <= 48
 row_capacity_l1_P41: rows_tot_l1_P41 <= 48
 row_capacity_l1_P42: rows_tot_l1_P42 <= 48
 row_capacity_l2_P4: rows_tot_l2_P4 <= 48
Bounds
 mem_tot_CPU1 <= 32768
 normalized_mem_tot_CPU1 <= 1
 normalized_rows_tot_CPU1 <= 1
 mem_tot_CPU2 <= 32768
 normalized_mem_tot_CPU2 <= 1
 normalized_rows_tot_CPU2 <= 1
 mem_tot_CPU3 <= 32768
 normalized_mem_tot_CPU3 <= 1
 normalized_rows_tot_CPU3 <= 1
 mem_tot_CPU4 <= 32768
 normalized_mem_tot_CPU4 <= 1
 normalized_rows_tot_CPU4 <= 1
 mem_tot_CPU5 <= 32768
 normalized_mem_tot_CPU5 <= 1
 normalized_rows_tot_CPU5 <= 1
 mem_tot_CPU6 <= 32768
 normalized_mem_tot_CPU6 <= 1
 normalized_rows_tot_CPU6 <= 1
 mem_tot_CPU7 <= 32768
 normalized_mem_tot_CPU7 <= 1
 normalized_rows_tot_CPU7 <= 1
 mem_tot_CPU8 <= 32768
 normalized_mem_tot_CPU8 <= 1
 normalized_rows_tot_CPU8 <= 1
 mem_tot_tor_P41 <= 576
 normalized_mem_tot_tor_P41 <= 1
 normalized_rows_tot_tor_P41 <= 1
 mem_tot_tor_P42 <= 576
 normalized_mem_tot_tor_P42 <= 1
 normalized_rows_tot_tor_P42 <= 1
 mem_tot_tor_P43 <= 576
 normalized_mem_tot_tor_P43 <= 1
 normalized_rows_tot_tor_P43 <= 1
 mem_tot_tor_P44 <= 576
 normalized_mem_tot_tor_P44 <= 1
 normalized_rows_tot_tor_P44 <= 1
 mem_tot_l1_P41 <= 576
 normalized_mem_tot_l1_P41 <= 1
 normalized_rows_tot_l1_P41 <= 1
 mem_tot_l1_P42 <= 576
 normalized_mem_tot_l1_P42 <= 1
 normalized_rows_tot_l1_P42 <= 1
 mem_tot_l2_P4 <= 576
 normalized_mem_tot_l2_P4 <= 1
 normalized_rows_tot_l2_P4 <= 1
Binaries
 frac[7,3] frac[14,74] frac[11,68] frac[12,78] frac[3,35] frac[4,36]
 frac[8,63] frac[7,25] frac[13,70] frac[5,57] frac[9,62] frac[4,66]
 frac[10,61] frac[2,78] frac[8,5] frac[7,55] frac[13,32] frac[1,64]
 frac[11,48] frac[10,64] frac[9,0] frac[14,31] frac[12,59] frac[10,7]
 frac[13,58] frac[11,22] frac[0,17] frac[7,68] frac[14,1] frac[12,17]
 frac[1,28] frac[13,20] frac[2,27] frac[0,55] frac[14,43] frac[3,2]
 frac[10,75] frac[1,54] frac[4,5] frac[2,53] frac[14,77] frac[12,69]
 frac[6,23] frac[4,35] frac[2,47] frac[8,38] frac[0,67] frac[7,22]
 frac[5,34] frac[9,39] frac[10,36] frac[2,73] frac[8,12] frac[7,44]
 frac[11,41] frac[9,9] frac[6,67] frac[12,50] frac[10,14] frac[8,18]
 frac[13,51] frac[11,15] frac[9,19] frac[14,8] frac[12,8] frac[13,13]
 frac[2,18] frac[0,14] frac[14,50] frac[3,11] frac[1,15] frac[4,12]
 frac[2,12] frac[5,1] frac[9,70] frac[1,33] frac[6,14] frac[4,42]
 frac[2,38] frac[7,15] frac[5,43] frac[3,55] frac[9,40] frac[6,32]
 frac[10,47] frac[2,64] frac[7,37] frac[11,46] frac[9,50] frac[6,58]
 frac[12,41] frac[3,68] frac[10,9] frac[8,25] frac[13,44] frac[11,4]
 frac[9,20] frac[14,19] frac[5,70] frac[12,7] frac[10,19] frac[13,6]
 frac[11,26] frac[0,5] frac[14,53] frac[12,29] frac[1,0] frac[4,11]
 frac[2,7] frac[0,59] frac[5,10] frac[9,79] frac[10,71] frac[1,58]
 frac[6,1] frac[4,17] frac[2,33] frac[7,4] frac[14,65] frac[11,65]
 frac[3,60] frac[5,20] frac[6,27] frac[8,58] frac[7,34] frac[13,75]
 frac[5,62] frac[9,59] frac[6,61] frac[12,32] frac[4,69] frac[10,48]
 frac[1,77] frac[8,0] frac[7,56] frac[13,37] frac[11,61] frac[9,29]
 frac[14,26] frac[12,62] frac[10,26] frac[13,63] frac[11,19] frac[7,73]
 frac[0,28] frac[14,60] frac[12,20] frac[1,25] frac[13,25] frac[14,38]
 frac[10,78] frac[4,24] frac[2,56] frac[0,40] frac[5,29] frac[11,70]
 frac[3,37] frac[12,72] frac[6,18] frac[4,38] frac[8,33] frac[0,78]
 frac[7,27] frac[5,39] frac[13,68] frac[9,60] frac[4,76] frac[1,70]
 frac[10,59] frac[8,7] frac[2,76] frac[7,49] frac[11,50] frac[9,6]
 frac[14,29] frac[6,78] frac[12,53] frac[10,5] frac[13,56] frac[11,8]
 frac[0,19] frac[7,70] frac[14,7] frac[12,19] frac[1,18] frac[13,18]
 frac[2,25] frac[0,9] frac[14,41] frac[3,4] frac[1,52] frac[4,7] frac[5,6]
 frac[12,71] frac[9,67] frac[1,46] frac[3,42] frac[4,45] frac[8,40]
 frac[5,32] frac[9,37] frac[4,75] frac[10,34] frac[8,14] frac[7,46]
 frac[11,43] frac[9,15] frac[6,65] frac[3,65] frac[12,44] frac[10,12]
 frac[8,20] frac[13,49] frac[11,1] frac[14,14] frac[12,10] frac[10,22]
 frac[13,11] frac[2,16] frac[0,0] frac[14,48] frac[3,13] frac[1,13]
 frac[8,64] frac[4,14] frac[0,38] frac[5,15] frac[9,68] frac[1,39]
 frac[6,12] frac[4,20] frac[2,36] frac[8,47] frac[7,9] frac[11,74]
 frac[3,57] frac[9,46] frac[10,45] frac[8,53] frac[11,32] frac[6,56]
 frac[12,43] frac[3,70] frac[10,55] frac[1,74] frac[8,27] frac[13,42]
 frac[11,6] frac[9,26] frac[14,17] frac[12,1] frac[10,17] frac[13,4]
 frac[11,28] frac[0,7] frac[14,59] frac[12,31] frac[1,6] frac[8,71]
 frac[13,30] frac[0,61] frac[5,8] frac[3,24] frac[9,77] frac[10,69]
 frac[1,56] frac[6,7] frac[4,19] frac[2,63] frac[7,6] frac[14,71]
 frac[11,67] frac[3,62] frac[4,57] frac[8,60] frac[7,28] frac[13,73]
 frac[0,73] frac[9,57] frac[12,34] frac[3,79] frac[10,62] frac[4,71]
 frac[8,2] frac[1,67] frac[7,58] frac[13,35] frac[11,63] frac[9,3]
 frac[14,24] frac[5,77] frac[12,56] frac[10,24] frac[13,61] frac[0,30]
 frac[7,75] frac[14,2] frac[12,22] frac[8,78] frac[13,23] frac[2,28]
 frac[14,36] frac[3,1] frac[10,76] frac[4,26] frac[2,54] frac[0,42]
 frac[14,78] frac[5,27] frac[12,74] frac[1,43] frac[6,16] frac[8,35]
 frac[13,66] frac[5,37] frac[9,34] frac[4,78] frac[10,57] frac[1,68]
 frac[8,9] frac[9,4] frac[12,55] frac[10,3] frac[13,54] frac[11,10]
 frac[7,64] frac[14,5] frac[12,13] frac[13,16] frac[0,11] frac[14,47]
 frac[3,6] frac[4,1] frac[2,49] frac[0,33] frac[12,65] frac[9,65]
 frac[3,44] frac[1,44] frac[4,47] frac[2,43] frac[8,42] frac[0,71]
 frac[5,46] frac[3,50] frac[9,43] frac[6,45] frac[4,53] frac[10,32]
 frac[2,69] frac[8,48] frac[7,40] frac[11,45] frac[9,13] frac[6,71]
 frac[12,46] frac[3,67] frac[10,10] frac[8,22] frac[13,47] frac[11,3]
 frac[14,12] frac[12,4] frac[10,20] frac[13,9] frac[11,25] frac[0,2]
 frac[14,54] frac[3,15] frac[1,3] frac[8,66] frac[2,8] frac[0,56]
 frac[5,13] frac[9,74] frac[1,37] frac[6,2] frac[4,22] frac[2,34]
 frac[7,11] frac[14,66] frac[11,76] frac[3,59] frac[9,44] frac[6,36]
 frac[10,43] frac[8,55] frac[7,33] frac[13,78] frac[11,34] frac[9,54]
 frac[12,37] frac[3,72] frac[10,53] frac[1,72] frac[7,63] frac[13,40]
 frac[11,56] frac[9,24] frac[5,66] frac[14,23] frac[12,3] frac[13,2]
 frac[11,30] frac[0,25] frac[7,76] frac[14,57] frac[12,25] frac[8,73]
 frac[13,28] frac[2,3] frac[0,63] frac[3,26] frac[10,67] frac[6,5]
 frac[4,29] frac[7,0] frac[5,16] frac[11,69] frac[12,77] frac[14,69]
 frac[4,59] frac[8,62] frac[7,30] frac[13,71] frac[0,75] frac[9,63]
 frac[6,49] frac[4,65] frac[1,65] frac[8,4] frac[13,33] frac[11,49]
 frac[9,1] frac[5,75] frac[14,30] frac[12,58] frac[10,6] frac[13,59]
 frac[11,23] frac[0,16] frac[14,0] frac[12,16] frac[13,21] frac[2,26]
 frac[0,54] frac[14,42] frac[3,3] frac[1,55] frac[4,4] frac[5,25]
 frac[14,76] frac[3,41] frac[12,68] frac[1,41] frac[6,22] frac[4,34]
 frac[2,46] frac[8,37] frac[13,64] frac[10,39] frac[2,72] frac[8,11]
 frac[7,45] frac[11,54] frac[6,66] frac[12,49] frac[10,1] frac[8,17]
 frac[13,52] frac[11,12] frac[0,23] frac[7,66] frac[14,11] frac[12,15]
 frac[1,22] frac[13,14] frac[2,21] frac[0,13] frac[14,45] frac[4,3]
 frac[2,15] frac[0,35] frac[5,2] frac[12,67] frac[9,71] frac[1,34]
 frac[3,46] frac[6,9] frac[4,41] frac[8,44] frac[7,12] frac[5,44]
 frac[11,73] frac[9,41] frac[4,55] frac[10,46] frac[2,67] frac[8,50]
 frac[5,54] frac[9,51] frac[6,69] frac[12,40] frac[10,8] frac[8,24]
 frac[13,45] frac[11,5] frac[9,21] frac[14,18] frac[5,71] frac[12,6]
 frac[13,7] frac[11,27] frac[0,4] frac[14,52] frac[8,68] frac[4,10]
 frac[0,58] frac[5,11] frac[9,72] frac[10,70] frac[6,0] frac[4,16]
 frac[7,5] frac[14,64] frac[11,78] frac[3,61] frac[6,26] frac[4,62]
 frac[10,41] frac[8,57] frac[7,35] frac[13,76] frac[5,63] frac[11,36]
 frac[6,60] frac[3,74] frac[12,39] frac[4,68] frac[10,51] frac[8,31]
 frac[7,57] frac[13,38] frac[11,58] frac[9,30] frac[14,21] frac[5,64]
 frac[12,61] frac[10,29] frac[13,0] frac[11,16] frac[0,27] frac[7,78]
 frac[14,63] frac[12,27] frac[1,26] frac[8,75] frac[13,26] frac[14,33]
 frac[3,28] frac[10,65] frac[1,60] frac[2,59] frac[7,2] frac[5,30]
 frac[11,71] frac[3,34] frac[12,79] frac[14,75] frac[6,29] frac[4,37]
 frac[8,32] frac[7,24] frac[13,69] frac[5,56] frac[0,77] frac[9,61]
 frac[4,67] frac[2,79] frac[8,6] frac[7,54] frac[9,7] frac[6,73]
 frac[12,52] frac[10,4] frac[13,57] frac[11,9] frac[7,71] frac[0,18]
 frac[14,6] frac[12,18] frac[1,19] frac[13,19] frac[14,40] frac[1,53]
 frac[4,6] frac[2,50] frac[0,46] frac[5,7] frac[12,70] frac[3,43]
 frac[1,47] frac[6,20] frac[8,39] frac[0,68] frac[7,17] frac[5,33]
 frac[3,49] frac[9,38] frac[10,37] frac[2,70] frac[8,13] frac[11,40]
 frac[9,8] frac[6,64] frac[12,51] frac[10,15] frac[8,19] frac[13,50]
 frac[11,14] frac[9,18] frac[14,9] frac[12,9] frac[1,20] frac[13,12]
 frac[2,19] frac[0,15] frac[14,51] frac[1,14] frac[4,13] frac[2,13]
 frac[0,37] frac[3,16] frac[9,69] frac[6,15] frac[4,43] frac[2,39]
 frac[8,46] frac[7,14] frac[5,42] frac[11,75] frac[9,47] frac[6,33]
 frac[4,49] frac[10,44] frac[8,52] frac[7,36] frac[5,52] frac[11,33]
 frac[9,49] frac[6,59] frac[3,71] frac[12,42] frac[10,54] frac[8,26]
 frac[13,43] frac[11,7] frac[9,27] frac[14,16] frac[12,0] frac[10,16]
 frac[13,5] frac[11,29] frac[14,58] frac[12,30] frac[1,7] frac[8,70]
 frac[13,31] frac[2,4] frac[0,60] frac[5,9] frac[9,78] frac[10,68]
 frac[1,57] frac[6,6] frac[2,62] frac[7,7] frac[14,70] frac[11,64]
 frac[3,63] frac[5,19] frac[6,24] frac[8,59] frac[7,29] frac[13,74]
 frac[5,61] frac[0,72] frac[11,38] frac[9,58] frac[6,50] frac[12,33]
 frac[3,76] frac[10,49] frac[8,1] frac[13,36] frac[11,60] frac[9,28]
 frac[5,78] frac[14,27] frac[12,63] frac[10,27] frac[13,62] frac[11,18]
 frac[14,61] frac[12,21] frac[8,77] frac[13,24] frac[14,39] frac[3,30]
 frac[10,79] frac[1,50] frac[2,57] frac[0,41] frac[14,73] frac[12,73]
 frac[6,19] frac[8,34] frac[7,26] frac[13,67] frac[5,38] frac[9,35]
 frac[6,53] frac[4,77] frac[10,56] frac[2,77] frac[7,48] frac[11,53]
 frac[12,54] frac[10,2] frac[13,55] frac[11,11] frac[7,65] frac[0,20]
 frac[14,4] frac[12,12] frac[1,17] frac[13,17] frac[2,22] frac[0,10]
 frac[14,46] frac[3,7] frac[1,11] frac[4,0] frac[0,32] frac[5,5] frac[3,45]
 frac[12,64] frac[9,66] frac[4,46] frac[2,42] frac[8,41] frac[7,19]
 frac[5,47] frac[3,51] frac[9,36] frac[4,52] frac[10,35] frac[2,68]
 frac[8,15] frac[7,41] frac[9,14] frac[6,70] frac[12,45] frac[3,64]
 frac[10,13] frac[13,48] frac[11,0] frac[9,16] frac[14,15] frac[12,11]
 frac[13,10] frac[0,1] frac[14,49] frac[1,12] frac[8,65] frac[4,15]
 frac[2,11] frac[0,39] frac[3,18] frac[9,75] frac[1,38] frac[6,13]
 frac[4,21] frac[2,37] frac[5,40] frac[11,77] frac[9,45] frac[4,51]
 frac[10,42] frac[8,54] frac[7,38] frac[13,79] frac[5,50] frac[11,35]
 frac[9,55] frac[3,73] frac[12,36] frac[10,52] frac[1,73] frac[8,28]
 frac[13,41] frac[11,57] frac[9,25] frac[14,22] frac[5,67] frac[12,2]
 frac[10,30] frac[13,3] frac[11,31] frac[0,24] frac[7,77] frac[14,56]
 frac[12,24] frac[8,72] frac[13,29] frac[2,2] frac[0,62] frac[14,34]
 frac[9,76] frac[10,66] frac[1,63] frac[4,28] frac[2,60] frac[7,1]
 frac[14,68] frac[11,66] frac[3,33] frac[12,76] frac[6,30] frac[8,61]
 frac[7,31] frac[0,74] frac[5,59] frac[9,56] frac[6,48] frac[12,35]
 frac[4,64] frac[10,63] frac[1,66] frac[8,3] frac[7,53] frac[13,34]
 frac[11,62] frac[9,2] frac[14,25] frac[6,74] frac[12,57] frac[10,25]
 frac[13,60] frac[11,20] frac[0,31] frac[7,74] frac[14,3] frac[12,23]
 frac[1,30] frac[8,79] frac[13,22] frac[2,29] frac[0,53] frac[14,37]
 frac[3,0] frac[10,77] frac[1,48] frac[4,27] frac[2,55] frac[0,43]
 frac[14,79] frac[12,75] frac[3,38] frac[6,17] frac[4,33] frac[13,65]
 frac[9,33] frac[6,43] frac[4,79] frac[10,38] frac[8,10] frac[11,55]
 frac[9,11] frac[6,77] frac[12,48] frac[10,0] frac[8,16] frac[13,53]
 frac[11,13] frac[7,67] frac[14,10] frac[12,14] frac[1,23] frac[13,15]
 frac[0,12] frac[14,44] frac[3,9] frac[4,2] frac[0,34] frac[5,3]
 frac[12,66] frac[9,64] frac[3,47] frac[6,8] frac[4,40] frac[2,40]
 frac[8,43] frac[7,13] frac[5,45] frac[3,53] frac[9,42] frac[6,34]
 frac[4,54] frac[10,33] frac[2,66] frac[7,43] frac[5,55] frac[11,44]
 frac[9,12] frac[6,68] frac[12,47] frac[3,66] frac[10,11] frac[8,23]
 frac[13,46] frac[11,2] frac[9,22] frac[14,13] frac[12,5] frac[10,21]
 frac[13,8] frac[11,24] frac[0,3] frac[14,55] frac[3,14] frac[1,2]
 frac[8,67] frac[4,9] frac[2,9] frac[0,57] frac[5,12] frac[3,20] frac[9,73]
 frac[6,3] frac[2,35] frac[7,10] frac[5,22] frac[11,79] frac[3,58]
 frac[14,67] frac[6,37] frac[4,61] frac[10,40] frac[8,56] frac[13,77]
 frac[5,48] frac[11,37] frac[9,53] frac[6,63] frac[12,38] frac[3,75]
 frac[10,50] frac[1,79] frac[8,30] frac[7,62] frac[13,39] frac[11,59]
 frac[14,20] frac[5,65] frac[12,60] frac[10,28] frac[13,1] frac[11,17]
 frac[7,79] frac[14,62] frac[12,26] frac[1,27] frac[8,74] frac[13,27]
 frac[2,0] frac[0,48] frac[14,32] frac[3,29] frac[1,61]
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_tor_P41 , mem_tot_tor_P42 ,
   mem_tot_tor_P43 , mem_tot_tor_P44 , mem_tot_l1_P41 , mem_tot_l1_P42 ,
   mem_tot_l2_P4 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU1 , normalized_mem_tot_CPU2 ,
   normalized_mem_tot_CPU3 , normalized_mem_tot_CPU4 ,
   normalized_mem_tot_CPU5 , normalized_mem_tot_CPU6 ,
   normalized_mem_tot_CPU7 , normalized_mem_tot_CPU8 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_tor_P41 , rows_tot_tor_P42 ,
   rows_tot_tor_P43 , rows_tot_tor_P44 , rows_tot_l1_P41 ,
   rows_tot_l1_P42 , rows_tot_l2_P4 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU1 ,
   normalized_rows_tot_CPU2 , normalized_rows_tot_CPU3 ,
   normalized_rows_tot_CPU4 , normalized_rows_tot_CPU5 ,
   normalized_rows_tot_CPU6 , normalized_rows_tot_CPU7 ,
   normalized_rows_tot_CPU8 )
End

\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_others: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU1 + rows_tot_CPU2 + rows_tot_CPU3 + rows_tot_CPU4
   + rows_tot_CPU5 + rows_tot_CPU6 + rows_tot_CPU7 + rows_tot_CPU8
   + rows_tot_CPU9 + rows_tot_CPU10 + rows_tot_CPU11 + rows_tot_CPU12
   + rows_tot_CPU13 + rows_tot_CPU14 + rows_tot_CPU15 + rows_tot_CPU16
   + rows_tot_CPU17 + rows_tot_CPU18 + rows_tot_CPU19 + rows_tot_CPU20
   + rows_tot_CPU21 + rows_tot_CPU22 + rows_tot_CPU23 + rows_tot_CPU24
   + rows_tot_CPU25 + rows_tot_CPU26 + rows_tot_CPU27 + rows_tot_CPU28
   + rows_tot_CPU29 + rows_tot_CPU30 + rows_tot_CPU31 + rows_tot_CPU32
  others_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_others
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_rows_fixed_thr: Priority=80 Weight=1 AbsTol=0 RelTol=0
   rows_tot_tor_P41 + rows_tot_tor_P42 + rows_tot_tor_P43 + rows_tot_tor_P44
   + rows_tot_l1_P41 + rows_tot_l1_P42 + rows_tot_l2_P4
  rows_load_fixed_thr: Priority=70 Weight=1 AbsTol=0 RelTol=0
   max_rows_fixed_thr
  tot_mem_others: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU1 + mem_tot_CPU2 + mem_tot_CPU3 + mem_tot_CPU4 + mem_tot_CPU5
   + mem_tot_CPU6 + mem_tot_CPU7 + mem_tot_CPU8 + mem_tot_CPU9
   + mem_tot_CPU10 + mem_tot_CPU11 + mem_tot_CPU12 + mem_tot_CPU13
   + mem_tot_CPU14 + mem_tot_CPU15 + mem_tot_CPU16 + mem_tot_CPU17
   + mem_tot_CPU18 + mem_tot_CPU19 + mem_tot_CPU20 + mem_tot_CPU21
   + mem_tot_CPU22 + mem_tot_CPU23 + mem_tot_CPU24 + mem_tot_CPU25
   + mem_tot_CPU26 + mem_tot_CPU27 + mem_tot_CPU28 + mem_tot_CPU29
   + mem_tot_CPU30 + mem_tot_CPU31 + mem_tot_CPU32
  others_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_others
  tot_mem_fixed_thr: Priority=40 Weight=1 AbsTol=0 RelTol=0
   mem_tot_tor_P41 + mem_tot_tor_P42 + mem_tot_tor_P43 + mem_tot_tor_P44
   + mem_tot_l1_P41 + mem_tot_l1_P42 + mem_tot_l2_P4
  mem_load_fixed_thr: Priority=30 Weight=1 AbsTol=0 RelTol=0
   max_mem_fixed_thr
Subject To
 cov_0_1: frac[6,1] + frac[25,1] + frac[32,1] + frac[35,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.7
 cov_1_10: frac[0,10] + frac[16,10] + frac[32,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.9
 cov_2_18: frac[16,18] + frac[22,18] + frac[34,18] >= 0.7
 cov_3_20: frac[18,20] + frac[22,20] + frac[34,20] >= 0.8
 cov_4_25: frac[0,25] + frac[4,25] + frac[32,25] >= 0.9
 cov_5_22: frac[16,22] + frac[22,22] + frac[34,22] >= 1
 cov_6_10: frac[18,10] + frac[22,10] + frac[34,10] >= 0.7
 cov_7_10: frac[16,10] + frac[18,10] + frac[34,10] >= 0.8
 cov_8_20: frac[0,20] + frac[4,20] + frac[32,20] >= 0.9
 cov_9_26: frac[16,26] + frac[18,26] + frac[34,26] >= 1
 cov_10_25: frac[18,25] + frac[22,25] + frac[34,25] >= 0.8
 cov_11_23: frac[16,23] + frac[22,23] + frac[34,23] >= 0.9
 cov_12_11: frac[0,11] + frac[22,11] + frac[32,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.9
 cov_13_10: frac[0,10] + frac[22,10] + frac[32,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.8
 cov_14_27: frac[0,27] + frac[22,27] + frac[32,27] + frac[34,27]
   + frac[36,27] + frac[37,27] + frac[38,27] >= 0.7
 cov_15_0: frac[15,0] + frac[22,0] + frac[33,0] + frac[34,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.8
 cov_16_26: frac[6,26] + frac[25,26] + frac[32,26] + frac[35,26]
   + frac[36,26] + frac[37,26] + frac[38,26] >= 1
 cov_17_0: frac[6,0] + frac[18,0] + frac[32,0] + frac[34,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.7
 cov_18_0: frac[6,0] + frac[22,0] + frac[32,0] + frac[34,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.7
 cov_19_26: frac[4,26] + frac[18,26] + frac[32,26] + frac[34,26]
   + frac[36,26] + frac[37,26] + frac[38,26] >= 0.8
 cov_20_9: frac[22,9] + frac[25,9] + frac[34,9] + frac[35,9] + frac[37,9]
   >= 0.7
 cov_21_19: frac[4,19] + frac[22,19] + frac[32,19] + frac[34,19]
   + frac[36,19] + frac[37,19] + frac[38,19] >= 0.7
 cov_22_13: frac[6,13] + frac[22,13] + frac[32,13] + frac[34,13]
   + frac[36,13] + frac[37,13] + frac[38,13] >= 0.9
 cov_23_4: frac[0,4] + frac[6,4] + frac[32,4] >= 0.9
 cov_24_3: frac[4,3] + frac[16,3] + frac[32,3] + frac[34,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 1
 cov_25_4: frac[18,4] + frac[22,4] + frac[34,4] >= 0.9
 cov_26_14: frac[18,14] + frac[22,14] + frac[34,14] >= 0.9
 cov_27_2: frac[16,2] + frac[25,2] + frac[34,2] + frac[35,2] + frac[37,2]
   >= 0.9
 cov_28_1: frac[0,1] + frac[22,1] + frac[32,1] + frac[34,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.8
 cov_29_3: frac[6,3] + frac[22,3] + frac[32,3] + frac[34,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 1
 cov_30_3: frac[22,3] + frac[25,3] + frac[34,3] + frac[35,3] + frac[37,3]
   >= 1
 cov_31_22: frac[4,22] + frac[16,22] + frac[32,22] + frac[34,22]
   + frac[36,22] + frac[37,22] + frac[38,22] >= 1
 cov_32_0: frac[0,0] + frac[22,0] + frac[32,0] + frac[34,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.9
 cov_33_29: frac[6,29] + frac[18,29] + frac[32,29] + frac[34,29]
   + frac[36,29] + frac[37,29] + frac[38,29] >= 0.8
 cov_34_5: frac[0,5] + frac[22,5] + frac[32,5] + frac[34,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 1
 cov_35_7: frac[0,7] + frac[25,7] + frac[32,7] + frac[35,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.7
 cov_36_24: frac[15,24] + frac[25,24] + frac[33,24] + frac[35,24]
   + frac[36,24] + frac[37,24] + frac[38,24] >= 1
 cov_37_1: frac[0,1] + frac[6,1] + frac[32,1] >= 1
 cov_38_24: frac[0,24] + frac[16,24] + frac[32,24] + frac[34,24]
   + frac[36,24] + frac[37,24] + frac[38,24] >= 0.9
 cov_39_22: frac[18,22] + frac[22,22] + frac[34,22] >= 0.9
 cov_40_7: frac[0,7] + frac[16,7] + frac[32,7] + frac[34,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 1
 cov_41_24: frac[6,24] + frac[22,24] + frac[32,24] + frac[34,24]
   + frac[36,24] + frac[37,24] + frac[38,24] >= 0.7
 cov_42_10: frac[15,10] + frac[16,10] + frac[33,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.9
 cov_43_2: frac[16,2] + frac[22,2] + frac[34,2] >= 1
 cov_44_4: frac[6,4] + frac[18,4] + frac[32,4] + frac[34,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.7
 cov_45_24: frac[22,24] + frac[25,24] + frac[34,24] + frac[35,24]
   + frac[37,24] >= 0.8
 cov_46_29: frac[16,29] + frac[18,29] + frac[34,29] >= 0.9
 cov_47_15: frac[18,15] + frac[22,15] + frac[34,15] >= 1
 cov_48_4: frac[16,4] + frac[25,4] + frac[34,4] + frac[35,4] + frac[37,4]
   >= 0.8
 cov_49_14: frac[16,14] + frac[18,14] + frac[34,14] >= 0.8
 cov_50_10: frac[0,10] + frac[16,10] + frac[32,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 1
 cov_51_17: frac[6,17] + frac[18,17] + frac[32,17] + frac[34,17]
   + frac[36,17] + frac[37,17] + frac[38,17] >= 0.7
 cov_52_22: frac[15,22] + frac[22,22] + frac[33,22] + frac[34,22]
   + frac[36,22] + frac[37,22] + frac[38,22] >= 0.8
 cov_53_23: frac[0,23] + frac[18,23] + frac[32,23] + frac[34,23]
   + frac[36,23] + frac[37,23] + frac[38,23] >= 0.7
 cov_54_13: frac[0,13] + frac[16,13] + frac[32,13] + frac[34,13]
   + frac[36,13] + frac[37,13] + frac[38,13] >= 0.8
 cov_55_6: frac[4,6] + frac[16,6] + frac[32,6] + frac[34,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 1
 cov_56_22: frac[15,22] + frac[16,22] + frac[33,22] + frac[34,22]
   + frac[36,22] + frac[37,22] + frac[38,22] >= 1
 cov_57_7: frac[6,7] + frac[22,7] + frac[32,7] + frac[34,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 1
 cov_58_6: frac[15,6] + frac[16,6] + frac[33,6] + frac[34,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.9
 cov_59_11: frac[4,11] + frac[22,11] + frac[32,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.7
 cov_60_57: frac[8,57] + frac[17,57] + frac[33,57] + frac[34,57]
   + frac[36,57] + frac[37,57] + frac[38,57] >= 0.9
 cov_61_38: frac[1,38] + frac[19,38] + frac[32,38] + frac[34,38]
   + frac[36,38] + frac[37,38] + frac[38,38] >= 1
 cov_62_38: frac[14,38] + frac[26,38] + frac[33,38] + frac[35,38]
   + frac[36,38] + frac[37,38] + frac[38,38] >= 0.8
 cov_63_38: frac[1,38] + frac[3,38] + frac[32,38] >= 0.9
 cov_64_38: frac[3,38] + frac[14,38] + frac[32,38] + frac[33,38]
   + frac[36,38] >= 0.9
 cov_65_46: frac[8,46] + frac[14,46] + frac[33,46] >= 0.9
 cov_66_32: frac[3,32] + frac[17,32] + frac[32,32] + frac[34,32]
   + frac[36,32] + frac[37,32] + frac[38,32] >= 1
 cov_67_51: frac[8,51] + frac[29,51] + frac[33,51] + frac[35,51]
   + frac[36,51] + frac[37,51] + frac[38,51] >= 0.8
 cov_68_52: frac[3,52] + frac[29,52] + frac[32,52] + frac[35,52]
   + frac[36,52] + frac[37,52] + frac[38,52] >= 1
 cov_69_32: frac[1,32] + frac[17,32] + frac[32,32] + frac[34,32]
   + frac[36,32] + frac[37,32] + frac[38,32] >= 0.9
 cov_70_44: frac[3,44] + frac[8,44] + frac[32,44] + frac[33,44]
   + frac[36,44] >= 1
 cov_71_40: frac[3,40] + frac[14,40] + frac[32,40] + frac[33,40]
   + frac[36,40] >= 0.8
 cov_72_56: frac[14,56] + frac[19,56] + frac[33,56] + frac[34,56]
   + frac[36,56] + frac[37,56] + frac[38,56] >= 1
 cov_73_58: frac[1,58] + frac[29,58] + frac[32,58] + frac[35,58]
   + frac[36,58] + frac[37,58] + frac[38,58] >= 1
 cov_74_36: frac[14,36] + frac[29,36] + frac[33,36] + frac[35,36]
   + frac[36,36] + frac[37,36] + frac[38,36] >= 0.7
 cov_75_31: frac[1,31] + frac[14,31] + frac[32,31] + frac[33,31]
   + frac[36,31] >= 1
 cov_76_34: frac[19,34] + frac[29,34] + frac[34,34] + frac[35,34]
   + frac[37,34] >= 0.9
 cov_77_55: frac[19,55] + frac[29,55] + frac[34,55] + frac[35,55]
   + frac[37,55] >= 1
 cov_78_49: frac[1,49] + frac[14,49] + frac[32,49] + frac[33,49]
   + frac[36,49] >= 0.8
 cov_79_43: frac[1,43] + frac[17,43] + frac[32,43] + frac[34,43]
   + frac[36,43] + frac[37,43] + frac[38,43] >= 1
 cov_80_46: frac[3,46] + frac[29,46] + frac[32,46] + frac[35,46]
   + frac[36,46] + frac[37,46] + frac[38,46] >= 1
 cov_81_49: frac[14,49] + frac[19,49] + frac[33,49] + frac[34,49]
   + frac[36,49] + frac[37,49] + frac[38,49] >= 0.8
 cov_82_56: frac[1,56] + frac[8,56] + frac[32,56] + frac[33,56]
   + frac[36,56] >= 0.9
 cov_83_32: frac[19,32] + frac[26,32] + frac[34,32] + frac[35,32]
   + frac[37,32] >= 0.9
 cov_84_38: frac[8,38] + frac[17,38] + frac[33,38] + frac[34,38]
   + frac[36,38] + frac[37,38] + frac[38,38] >= 0.9
 cov_85_44: frac[1,44] + frac[14,44] + frac[32,44] + frac[33,44]
   + frac[36,44] >= 1
 cov_86_35: frac[3,35] + frac[8,35] + frac[32,35] + frac[33,35]
   + frac[36,35] >= 0.9
 cov_87_44: frac[14,44] + frac[17,44] + frac[33,44] + frac[34,44]
   + frac[36,44] + frac[37,44] + frac[38,44] >= 0.7
 cov_88_47: frac[17,47] + frac[29,47] + frac[34,47] + frac[35,47]
   + frac[37,47] >= 0.8
 cov_89_50: frac[8,50] + frac[29,50] + frac[33,50] + frac[35,50]
   + frac[36,50] + frac[37,50] + frac[38,50] >= 0.8
 cov_90_47: frac[17,47] + frac[26,47] + frac[34,47] + frac[35,47]
   + frac[37,47] >= 0.7
 cov_91_46: frac[14,46] + frac[19,46] + frac[33,46] + frac[34,46]
   + frac[36,46] + frac[37,46] + frac[38,46] >= 1
 cov_92_36: frac[8,36] + frac[29,36] + frac[33,36] + frac[35,36]
   + frac[36,36] + frac[37,36] + frac[38,36] >= 0.7
 cov_93_31: frac[1,31] + frac[29,31] + frac[32,31] + frac[35,31]
   + frac[36,31] + frac[37,31] + frac[38,31] >= 0.7
 cov_94_57: frac[14,57] + frac[29,57] + frac[33,57] + frac[35,57]
   + frac[36,57] + frac[37,57] + frac[38,57] >= 1
 cov_95_37: frac[3,37] + frac[17,37] + frac[32,37] + frac[34,37]
   + frac[36,37] + frac[37,37] + frac[38,37] >= 1
 cov_96_58: frac[19,58] + frac[29,58] + frac[34,58] + frac[35,58]
   + frac[37,58] >= 0.9
 cov_97_46: frac[14,46] + frac[29,46] + frac[33,46] + frac[35,46]
   + frac[36,46] + frac[37,46] + frac[38,46] >= 1
 cov_98_42: frac[3,42] + frac[8,42] + frac[32,42] + frac[33,42]
   + frac[36,42] >= 1
 cov_99_33: frac[3,33] + frac[8,33] + frac[32,33] + frac[33,33]
   + frac[36,33] >= 0.8
 cov_100_30: frac[3,30] + frac[29,30] + frac[32,30] + frac[35,30]
   + frac[36,30] + frac[37,30] + frac[38,30] >= 0.9
 cov_101_43: frac[1,43] + frac[26,43] + frac[32,43] + frac[35,43]
   + frac[36,43] + frac[37,43] + frac[38,43] >= 0.7
 cov_102_50: frac[17,50] + frac[29,50] + frac[34,50] + frac[35,50]
   + frac[37,50] >= 0.7
 cov_103_54: frac[14,54] + frac[17,54] + frac[33,54] + frac[34,54]
   + frac[36,54] + frac[37,54] + frac[38,54] >= 0.8
 cov_104_57: frac[1,57] + frac[17,57] + frac[32,57] + frac[34,57]
   + frac[36,57] + frac[37,57] + frac[38,57] >= 0.9
 cov_105_52: frac[1,52] + frac[19,52] + frac[32,52] + frac[34,52]
   + frac[36,52] + frac[37,52] + frac[38,52] >= 0.7
 cov_106_47: frac[14,47] + frac[29,47] + frac[33,47] + frac[35,47]
   + frac[36,47] + frac[37,47] + frac[38,47] >= 0.9
 cov_107_49: frac[1,49] + frac[14,49] + frac[32,49] + frac[33,49]
   + frac[36,49] >= 0.8
 cov_108_54: frac[3,54] + frac[17,54] + frac[32,54] + frac[34,54]
   + frac[36,54] + frac[37,54] + frac[38,54] >= 0.9
 cov_109_46: frac[17,46] + frac[26,46] + frac[34,46] + frac[35,46]
   + frac[37,46] >= 0.9
 cov_110_53: frac[8,53] + frac[19,53] + frac[33,53] + frac[34,53]
   + frac[36,53] + frac[37,53] + frac[38,53] >= 1
 cov_111_52: frac[3,52] + frac[14,52] + frac[32,52] + frac[33,52]
   + frac[36,52] >= 0.9
 cov_112_32: frac[3,32] + frac[19,32] + frac[32,32] + frac[34,32]
   + frac[36,32] + frac[37,32] + frac[38,32] >= 0.8
 cov_113_32: frac[14,32] + frac[17,32] + frac[33,32] + frac[34,32]
   + frac[36,32] + frac[37,32] + frac[38,32] >= 1
 cov_114_43: frac[8,43] + frac[26,43] + frac[33,43] + frac[35,43]
   + frac[36,43] + frac[37,43] + frac[38,43] >= 0.7
 cov_115_48: frac[1,48] + frac[29,48] + frac[32,48] + frac[35,48]
   + frac[36,48] + frac[37,48] + frac[38,48] >= 0.9
 cov_116_39: frac[3,39] + frac[14,39] + frac[32,39] + frac[33,39]
   + frac[36,39] >= 1
 cov_117_34: frac[8,34] + frac[14,34] + frac[33,34] >= 0.9
 cov_118_45: frac[3,45] + frac[26,45] + frac[32,45] + frac[35,45]
   + frac[36,45] + frac[37,45] + frac[38,45] >= 1
 cov_119_40: frac[1,40] + frac[17,40] + frac[32,40] + frac[34,40]
   + frac[36,40] + frac[37,40] + frac[38,40] >= 1
 cov_120_87: frac[13,87] + frac[20,87] + frac[33,87] + frac[34,87]
   + frac[36,87] + frac[37,87] + frac[38,87] >= 0.8
 cov_121_73: frac[5,73] + frac[20,73] + frac[32,73] + frac[34,73]
   + frac[36,73] + frac[37,73] + frac[38,73] >= 0.7
 cov_122_79: frac[10,79] + frac[20,79] + frac[33,79] + frac[34,79]
   + frac[36,79] + frac[37,79] + frac[38,79] >= 1
 cov_123_63: frac[20,63] + frac[28,63] + frac[34,63] + frac[35,63]
   + frac[37,63] >= 0.9
 cov_124_74: frac[10,74] + frac[20,74] + frac[33,74] + frac[34,74]
   + frac[36,74] + frac[37,74] + frac[38,74] >= 0.9
 cov_125_64: frac[13,64] + frac[24,64] + frac[33,64] + frac[35,64]
   + frac[36,64] + frac[37,64] + frac[38,64] >= 0.8
 cov_126_75: frac[27,75] + frac[28,75] + frac[35,75] >= 0.7
 cov_127_62: frac[10,62] + frac[27,62] + frac[33,62] + frac[35,62]
   + frac[36,62] + frac[37,62] + frac[38,62] >= 0.8
 cov_128_65: frac[20,65] + frac[28,65] + frac[34,65] + frac[35,65]
   + frac[37,65] >= 0.9
 cov_129_75: frac[10,75] + frac[24,75] + frac[33,75] + frac[35,75]
   + frac[36,75] + frac[37,75] + frac[38,75] >= 0.9
 cov_130_61: frac[5,61] + frac[24,61] + frac[32,61] + frac[35,61]
   + frac[36,61] + frac[37,61] + frac[38,61] >= 0.9
 cov_131_70: frac[9,70] + frac[24,70] + frac[33,70] + frac[35,70]
   + frac[36,70] + frac[37,70] + frac[38,70] >= 0.8
 cov_132_73: frac[5,73] + frac[10,73] + frac[32,73] + frac[33,73]
   + frac[36,73] >= 0.9
 cov_133_60: frac[9,60] + frac[28,60] + frac[33,60] + frac[35,60]
   + frac[36,60] + frac[37,60] + frac[38,60] >= 0.8
 cov_134_84: frac[13,84] + frac[27,84] + frac[33,84] + frac[35,84]
   + frac[36,84] + frac[37,84] + frac[38,84] >= 1
 cov_135_62: frac[9,62] + frac[13,62] + frac[33,62] >= 0.8
 cov_136_61: frac[27,61] + frac[28,61] + frac[35,61] >= 0.9
 cov_137_65: frac[5,65] + frac[10,65] + frac[32,65] + frac[33,65]
   + frac[36,65] >= 0.8
 cov_138_70: frac[13,70] + frac[28,70] + frac[33,70] + frac[35,70]
   + frac[36,70] + frac[37,70] + frac[38,70] >= 0.7
 cov_139_74: frac[27,74] + frac[28,74] + frac[35,74] >= 0.7
 cov_140_62: frac[5,62] + frac[10,62] + frac[32,62] + frac[33,62]
   + frac[36,62] >= 0.8
 cov_141_64: frac[9,64] + frac[10,64] + frac[33,64] >= 0.8
 cov_142_71: frac[24,71] + frac[28,71] + frac[35,71] >= 0.9
 cov_143_85: frac[10,85] + frac[20,85] + frac[33,85] + frac[34,85]
   + frac[36,85] + frac[37,85] + frac[38,85] >= 1
 cov_144_70: frac[9,70] + frac[24,70] + frac[33,70] + frac[35,70]
   + frac[36,70] + frac[37,70] + frac[38,70] >= 0.7
 cov_145_73: frac[10,73] + frac[13,73] + frac[33,73] >= 0.8
 cov_146_72: frac[24,72] + frac[27,72] + frac[35,72] >= 0.9
 cov_147_82: frac[5,82] + frac[24,82] + frac[32,82] + frac[35,82]
   + frac[36,82] + frac[37,82] + frac[38,82] >= 1
 cov_148_62: frac[9,62] + frac[27,62] + frac[33,62] + frac[35,62]
   + frac[36,62] + frac[37,62] + frac[38,62] >= 0.8
 cov_149_81: frac[9,81] + frac[10,81] + frac[33,81] >= 1
 cov_150_79: frac[24,79] + frac[27,79] + frac[35,79] >= 0.8
 cov_151_61: frac[9,61] + frac[24,61] + frac[33,61] + frac[35,61]
   + frac[36,61] + frac[37,61] + frac[38,61] >= 1
 cov_152_62: frac[20,62] + frac[28,62] + frac[34,62] + frac[35,62]
   + frac[37,62] >= 0.8
 cov_153_68: frac[13,68] + frac[27,68] + frac[33,68] + frac[35,68]
   + frac[36,68] + frac[37,68] + frac[38,68] >= 0.8
 cov_154_73: frac[10,73] + frac[28,73] + frac[33,73] + frac[35,73]
   + frac[36,73] + frac[37,73] + frac[38,73] >= 0.7
 cov_155_84: frac[5,84] + frac[13,84] + frac[32,84] + frac[33,84]
   + frac[36,84] >= 0.9
 cov_156_64: frac[10,64] + frac[13,64] + frac[33,64] >= 0.8
 cov_157_71: frac[27,71] + frac[28,71] + frac[35,71] >= 1
 cov_158_64: frac[24,64] + frac[28,64] + frac[35,64] >= 0.9
 cov_159_86: frac[27,86] + frac[28,86] + frac[35,86] >= 1
 cov_160_62: frac[9,62] + frac[24,62] + frac[33,62] + frac[35,62]
   + frac[36,62] + frac[37,62] + frac[38,62] >= 1
 cov_161_68: frac[9,68] + frac[13,68] + frac[33,68] >= 0.8
 cov_162_63: frac[5,63] + frac[10,63] + frac[32,63] + frac[33,63]
   + frac[36,63] >= 0.8
 cov_163_84: frac[10,84] + frac[20,84] + frac[33,84] + frac[34,84]
   + frac[36,84] + frac[37,84] + frac[38,84] >= 0.8
 cov_164_88: frac[5,88] + frac[28,88] + frac[32,88] + frac[35,88]
   + frac[36,88] + frac[37,88] + frac[38,88] >= 0.9
 cov_165_83: frac[13,83] + frac[28,83] + frac[33,83] + frac[35,83]
   + frac[36,83] + frac[37,83] + frac[38,83] >= 0.7
 cov_166_75: frac[20,75] + frac[24,75] + frac[34,75] + frac[35,75]
   + frac[37,75] >= 0.7
 cov_167_67: frac[5,67] + frac[10,67] + frac[32,67] + frac[33,67]
   + frac[36,67] >= 0.7
 cov_168_76: frac[5,76] + frac[13,76] + frac[32,76] + frac[33,76]
   + frac[36,76] >= 0.7
 cov_169_83: frac[5,83] + frac[20,83] + frac[32,83] + frac[34,83]
   + frac[36,83] + frac[37,83] + frac[38,83] >= 1
 cov_170_65: frac[20,65] + frac[24,65] + frac[34,65] + frac[35,65]
   + frac[37,65] >= 1
 cov_171_78: frac[10,78] + frac[13,78] + frac[33,78] >= 0.7
 cov_172_82: frac[9,82] + frac[20,82] + frac[33,82] + frac[34,82]
   + frac[36,82] + frac[37,82] + frac[38,82] >= 0.9
 cov_173_85: frac[20,85] + frac[24,85] + frac[34,85] + frac[35,85]
   + frac[37,85] >= 1
 cov_174_75: frac[10,75] + frac[13,75] + frac[33,75] >= 0.7
 cov_175_65: frac[9,65] + frac[13,65] + frac[33,65] >= 0.8
 cov_176_67: frac[27,67] + frac[28,67] + frac[35,67] >= 1
 cov_177_62: frac[10,62] + frac[13,62] + frac[33,62] >= 1
 cov_178_86: frac[27,86] + frac[28,86] + frac[35,86] >= 0.8
 cov_179_77: frac[13,77] + frac[24,77] + frac[33,77] + frac[35,77]
   + frac[36,77] + frac[37,77] + frac[38,77] >= 0.8
 cov_180_112: frac[7,112] + frac[21,112] + frac[32,112] + frac[34,112]
   + frac[36,112] + frac[37,112] + frac[38,112] >= 1
 cov_181_104: frac[11,104] + frac[21,104] + frac[33,104] + frac[34,104]
   + frac[36,104] + frac[37,104] + frac[38,104] >= 1
 cov_182_98: frac[7,98] + frac[31,98] + frac[32,98] + frac[35,98]
   + frac[36,98] + frac[37,98] + frac[38,98] >= 0.9
 cov_183_94: frac[21,94] + frac[31,94] + frac[34,94] + frac[35,94]
   + frac[37,94] >= 0.8
 cov_184_114: frac[11,114] + frac[30,114] + frac[33,114] + frac[35,114]
   + frac[36,114] + frac[37,114] + frac[38,114] >= 0.8
 cov_185_100: frac[2,100] + frac[30,100] + frac[32,100] + frac[35,100]
   + frac[36,100] + frac[37,100] + frac[38,100] >= 1
 cov_186_94: frac[11,94] + frac[31,94] + frac[33,94] + frac[35,94]
   + frac[36,94] + frac[37,94] + frac[38,94] >= 1
 cov_187_100: frac[7,100] + frac[21,100] + frac[32,100] + frac[34,100]
   + frac[36,100] + frac[37,100] + frac[38,100] >= 1
 cov_188_100: frac[23,100] + frac[30,100] + frac[34,100] + frac[35,100]
   + frac[37,100] >= 0.7
 cov_189_90: frac[23,90] + frac[31,90] + frac[34,90] + frac[35,90]
   + frac[37,90] >= 0.9
 cov_190_107: frac[11,107] + frac[30,107] + frac[33,107] + frac[35,107]
   + frac[36,107] + frac[37,107] + frac[38,107] >= 0.7
 cov_191_101: frac[12,101] + frac[31,101] + frac[33,101] + frac[35,101]
   + frac[36,101] + frac[37,101] + frac[38,101] >= 1
 cov_192_106: frac[23,106] + frac[30,106] + frac[34,106] + frac[35,106]
   + frac[37,106] >= 1
 cov_193_118: frac[11,118] + frac[23,118] + frac[33,118] + frac[34,118]
   + frac[36,118] + frac[37,118] + frac[38,118] >= 0.9
 cov_194_97: frac[7,97] + frac[12,97] + frac[32,97] + frac[33,97]
   + frac[36,97] >= 1
 cov_195_115: frac[30,115] + frac[31,115] + frac[35,115] >= 0.8
 cov_196_101: frac[11,101] + frac[21,101] + frac[33,101] + frac[34,101]
   + frac[36,101] + frac[37,101] + frac[38,101] >= 0.8
 cov_197_102: frac[7,102] + frac[21,102] + frac[32,102] + frac[34,102]
   + frac[36,102] + frac[37,102] + frac[38,102] >= 0.8
 cov_198_106: frac[2,106] + frac[30,106] + frac[32,106] + frac[35,106]
   + frac[36,106] + frac[37,106] + frac[38,106] >= 0.8
 cov_199_116: frac[23,116] + frac[31,116] + frac[34,116] + frac[35,116]
   + frac[37,116] >= 0.7
 cov_200_116: frac[23,116] + frac[31,116] + frac[34,116] + frac[35,116]
   + frac[37,116] >= 0.9
 cov_201_90: frac[7,90] + frac[30,90] + frac[32,90] + frac[35,90]
   + frac[36,90] + frac[37,90] + frac[38,90] >= 0.9
 cov_202_116: frac[12,116] + frac[23,116] + frac[33,116] + frac[34,116]
   + frac[36,116] + frac[37,116] + frac[38,116] >= 1
 cov_203_111: frac[23,111] + frac[31,111] + frac[34,111] + frac[35,111]
   + frac[37,111] >= 0.9
 cov_204_116: frac[7,116] + frac[31,116] + frac[32,116] + frac[35,116]
   + frac[36,116] + frac[37,116] + frac[38,116] >= 0.9
 cov_205_91: frac[7,91] + frac[21,91] + frac[32,91] + frac[34,91]
   + frac[36,91] + frac[37,91] + frac[38,91] >= 1
 cov_206_116: frac[11,116] + frac[30,116] + frac[33,116] + frac[35,116]
   + frac[36,116] + frac[37,116] + frac[38,116] >= 1
 cov_207_97: frac[2,97] + frac[21,97] + frac[32,97] + frac[34,97]
   + frac[36,97] + frac[37,97] + frac[38,97] >= 0.7
 cov_208_99: frac[2,99] + frac[21,99] + frac[32,99] + frac[34,99]
   + frac[36,99] + frac[37,99] + frac[38,99] >= 0.9
 cov_209_92: frac[21,92] + frac[23,92] + frac[34,92] >= 0.8
 cov_210_101: frac[12,101] + frac[30,101] + frac[33,101] + frac[35,101]
   + frac[36,101] + frac[37,101] + frac[38,101] >= 0.8
 cov_211_104: frac[7,104] + frac[30,104] + frac[32,104] + frac[35,104]
   + frac[36,104] + frac[37,104] + frac[38,104] >= 0.9
 cov_212_96: frac[2,96] + frac[30,96] + frac[32,96] + frac[35,96]
   + frac[36,96] + frac[37,96] + frac[38,96] >= 1
 cov_213_92: frac[12,92] + frac[30,92] + frac[33,92] + frac[35,92]
   + frac[36,92] + frac[37,92] + frac[38,92] >= 0.9
 cov_214_113: frac[7,113] + frac[30,113] + frac[32,113] + frac[35,113]
   + frac[36,113] + frac[37,113] + frac[38,113] >= 0.7
 cov_215_93: frac[12,93] + frac[30,93] + frac[33,93] + frac[35,93]
   + frac[36,93] + frac[37,93] + frac[38,93] >= 0.9
 cov_216_102: frac[7,102] + frac[21,102] + frac[32,102] + frac[34,102]
   + frac[36,102] + frac[37,102] + frac[38,102] >= 0.7
 cov_217_116: frac[11,116] + frac[21,116] + frac[33,116] + frac[34,116]
   + frac[36,116] + frac[37,116] + frac[38,116] >= 0.9
 cov_218_115: frac[2,115] + frac[12,115] + frac[32,115] + frac[33,115]
   + frac[36,115] >= 1
 cov_219_90: frac[21,90] + frac[31,90] + frac[34,90] + frac[35,90]
   + frac[37,90] >= 0.9
 cov_220_111: frac[11,111] + frac[30,111] + frac[33,111] + frac[35,111]
   + frac[36,111] + frac[37,111] + frac[38,111] >= 0.9
 cov_221_98: frac[2,98] + frac[12,98] + frac[32,98] + frac[33,98]
   + frac[36,98] >= 1
 cov_222_100: frac[11,100] + frac[23,100] + frac[33,100] + frac[34,100]
   + frac[36,100] + frac[37,100] + frac[38,100] >= 0.8
 cov_223_113: frac[23,113] + frac[31,113] + frac[34,113] + frac[35,113]
   + frac[37,113] >= 0.9
 cov_224_95: frac[2,95] + frac[11,95] + frac[32,95] + frac[33,95]
   + frac[36,95] >= 0.8
 cov_225_119: frac[2,119] + frac[30,119] + frac[32,119] + frac[35,119]
   + frac[36,119] + frac[37,119] + frac[38,119] >= 0.9
 cov_226_97: frac[21,97] + frac[23,97] + frac[34,97] >= 1
 cov_227_119: frac[23,119] + frac[30,119] + frac[34,119] + frac[35,119]
   + frac[37,119] >= 1
 cov_228_108: frac[7,108] + frac[23,108] + frac[32,108] + frac[34,108]
   + frac[36,108] + frac[37,108] + frac[38,108] >= 0.9
 cov_229_104: frac[11,104] + frac[21,104] + frac[33,104] + frac[34,104]
   + frac[36,104] + frac[37,104] + frac[38,104] >= 0.9
 cov_230_118: frac[2,118] + frac[7,118] + frac[32,118] >= 0.7
 cov_231_107: frac[21,107] + frac[30,107] + frac[34,107] + frac[35,107]
   + frac[37,107] >= 1
 cov_232_104: frac[11,104] + frac[30,104] + frac[33,104] + frac[35,104]
   + frac[36,104] + frac[37,104] + frac[38,104] >= 1
 cov_233_105: frac[11,105] + frac[31,105] + frac[33,105] + frac[35,105]
   + frac[36,105] + frac[37,105] + frac[38,105] >= 0.9
 cov_234_95: frac[7,95] + frac[30,95] + frac[32,95] + frac[35,95]
   + frac[36,95] + frac[37,95] + frac[38,95] >= 0.8
 cov_235_101: frac[23,101] + frac[30,101] + frac[34,101] + frac[35,101]
   + frac[37,101] >= 0.9
 cov_236_114: frac[11,114] + frac[23,114] + frac[33,114] + frac[34,114]
   + frac[36,114] + frac[37,114] + frac[38,114] >= 0.7
 cov_237_117: frac[2,117] + frac[11,117] + frac[32,117] + frac[33,117]
   + frac[36,117] >= 1
 cov_238_119: frac[21,119] + frac[30,119] + frac[34,119] + frac[35,119]
   + frac[37,119] >= 0.9
 cov_239_104: frac[2,104] + frac[30,104] + frac[32,104] + frac[35,104]
   + frac[36,104] + frac[37,104] + frac[38,104] >= 0.9
 accuracy_26_47: - 5.09765625 frac[26,47] + mem[26,47] = 0
 accuracy_32_54: - 6 frac[32,54] + mem[32,54] = 0
 accuracy_35_88: - 6 frac[35,88] + mem[35,88] = 0
 accuracy_29_50: - 5.09765625 frac[29,50] + mem[29,50] = 0
 accuracy_3_35: - 5.09765625 frac[3,35] + mem[3,35] = 0
 accuracy_36_68: - 6 frac[36,68] + mem[36,68] = 0
 accuracy_37_82: - 6 frac[37,82] + mem[37,82] = 0
 accuracy_34_52: - 6 frac[34,52] + mem[34,52] = 0
 accuracy_32_44: - 6 frac[32,44] + mem[32,44] = 0
 accuracy_22_19: - 5.09765625 frac[22,19] + mem[22,19] = 0
 accuracy_13_70: - 5.09765625 frac[13,70] + mem[13,70] = 0
 accuracy_9_62: - 5.09765625 frac[9,62] + mem[9,62] = 0
 accuracy_27_68: - 5.09765625 frac[27,68] + mem[27,68] = 0
 accuracy_34_46: - 6 frac[34,46] + mem[34,46] = 0
 accuracy_17_46: - 5.09765625 frac[17,46] + mem[17,46] = 0
 accuracy_28_88: - 5.09765625 frac[28,88] + mem[28,88] = 0
 accuracy_14_31: - 5.09765625 frac[14,31] + mem[14,31] = 0
 accuracy_38_40: - 6 frac[38,40] + mem[38,40] = 0
 accuracy_2_104: - 5.09765625 frac[2,104] + mem[2,104] = 0
 accuracy_21_119: - 5.09765625 frac[21,119] + mem[21,119] = 0
 accuracy_37_13: - 6 frac[37,13] + mem[37,13] = 0
 accuracy_20_75: - 5.09765625 frac[20,75] + mem[20,75] = 0
 accuracy_32_113: - 6 frac[32,113] + mem[32,113] = 0
 accuracy_33_101: - 6 frac[33,101] + mem[33,101] = 0
 accuracy_5_84: - 5.09765625 frac[5,84] + mem[5,84] = 0
 accuracy_36_22: - 6 frac[36,22] + mem[36,22] = 0
 accuracy_23_97: - 5.09765625 frac[23,97] + mem[23,97] = 0
 accuracy_30_100: - 5.09765625 frac[30,100] + mem[30,100] = 0
 accuracy_37_23: - 6 frac[37,23] + mem[37,23] = 0
 accuracy_7_98: - 5.09765625 frac[7,98] + mem[7,98] = 0
 accuracy_33_95: - 6 frac[33,95] + mem[33,95] = 0
 accuracy_10_75: - 5.09765625 frac[10,75] + mem[10,75] = 0
 accuracy_36_101: - 6 frac[36,101] + mem[36,101] = 0
 accuracy_37_113: - 6 frac[37,113] + mem[37,113] = 0
 accuracy_34_85: - 6 frac[34,85] + mem[34,85] = 0
 accuracy_35_65: - 6 frac[35,65] + mem[35,65] = 0
 accuracy_3_40: - 5.09765625 frac[3,40] + mem[3,40] = 0
 accuracy_1_40: - 5.09765625 frac[1,40] + mem[1,40] = 0
 accuracy_10_85: - 5.09765625 frac[10,85] + mem[10,85] = 0
 accuracy_38_119: - 6 frac[38,119] + mem[38,119] = 0
 accuracy_36_67: - 6 frac[36,67] + mem[36,67] = 0
 accuracy_37_91: - 6 frac[37,91] + mem[37,91] = 0
 accuracy_34_79: - 6 frac[34,79] + mem[34,79] = 0
 accuracy_8_38: - 5.09765625 frac[8,38] + mem[8,38] = 0
 accuracy_32_35: - 6 frac[32,35] + mem[32,35] = 0
 accuracy_35_46: - 6 frac[35,46] + mem[35,46] = 0
 accuracy_18_10: - 5.09765625 frac[18,10] + mem[18,10] = 0
 accuracy_33_34: - 6 frac[33,34] + mem[33,34] = 0
 accuracy_38_73: - 6 frac[38,73] + mem[38,73] = 0
 accuracy_32_25: - 6 frac[32,25] + mem[32,25] = 0
 accuracy_37_44: - 6 frac[37,44] + mem[37,44] = 0
 accuracy_35_52: - 6 frac[35,52] + mem[35,52] = 0
 accuracy_34_3: - 6 frac[34,3] + mem[34,3] = 0
 accuracy_17_57: - 5.09765625 frac[17,57] + mem[17,57] = 0
 accuracy_21_112: - 5.09765625 frac[21,112] + mem[21,112] = 0
 accuracy_2_99: - 5.09765625 frac[2,99] + mem[2,99] = 0
 accuracy_37_54: - 6 frac[37,54] + mem[37,54] = 0
 accuracy_20_82: - 5.09765625 frac[20,82] + mem[20,82] = 0
 accuracy_35_26: - 6 frac[35,26] + mem[35,26] = 0
 accuracy_38_5: - 6 frac[38,5] + mem[38,5] = 0
 accuracy_36_29: - 6 frac[36,29] + mem[36,29] = 0
 accuracy_21_90: - 5.09765625 frac[21,90] + mem[21,90] = 0
 accuracy_30_111: - 5.09765625 frac[30,111] + mem[30,111] = 0
 accuracy_34_114: - 6 frac[34,114] + mem[34,114] = 0
 accuracy_35_96: - 6 frac[35,96] + mem[35,96] = 0
 accuracy_7_91: - 5.09765625 frac[7,91] + mem[7,91] = 0
 accuracy_38_31: - 6 frac[38,31] + mem[38,31] = 0
 accuracy_30_113: - 5.09765625 frac[30,113] + mem[30,113] = 0
 accuracy_36_108: - 6 frac[36,108] + mem[36,108] = 0
 accuracy_34_108: - 6 frac[34,108] + mem[34,108] = 0
 accuracy_35_70: - 6 frac[35,70] + mem[35,70] = 0
 accuracy_7_113: - 5.09765625 frac[7,113] + mem[7,113] = 0
 accuracy_33_74: - 6 frac[33,74] + mem[33,74] = 0
 accuracy_11_114: - 5.09765625 frac[11,114] + mem[11,114] = 0
 accuracy_9_70: - 5.09765625 frac[9,70] + mem[9,70] = 0
 accuracy_16_7: - 5.09765625 frac[16,7] + mem[16,7] = 0
 accuracy_36_74: - 6 frac[36,74] + mem[36,74] = 0
 accuracy_28_63: - 5.09765625 frac[28,63] + mem[28,63] = 0
 accuracy_37_68: - 6 frac[37,68] + mem[37,68] = 0
 accuracy_32_58: - 6 frac[32,58] + mem[32,58] = 0
 accuracy_16_29: - 5.09765625 frac[16,29] + mem[16,29] = 0
 accuracy_38_64: - 6 frac[38,64] + mem[38,64] = 0
 accuracy_34_32: - 6 frac[34,32] + mem[34,32] = 0
 accuracy_8_51: - 5.09765625 frac[8,51] + mem[8,51] = 0
 accuracy_22_7: - 5.09765625 frac[22,7] + mem[22,7] = 0
 accuracy_35_61: - 6 frac[35,61] + mem[35,61] = 0
 accuracy_27_72: - 5.09765625 frac[27,72] + mem[27,72] = 0
 accuracy_38_90: - 6 frac[38,90] + mem[38,90] = 0
 accuracy_36_62: - 6 frac[36,62] + mem[36,62] = 0
 accuracy_34_26: - 6 frac[34,26] + mem[34,26] = 0
 accuracy_17_50: - 5.09765625 frac[17,50] + mem[17,50] = 0
 accuracy_37_63: - 6 frac[37,63] + mem[37,63] = 0
 accuracy_35_3: - 6 frac[35,3] + mem[35,3] = 0
 accuracy_38_60: - 6 frac[38,60] + mem[38,60] = 0
 accuracy_36_4: - 6 frac[36,4] + mem[36,4] = 0
 accuracy_23_119: - 5.09765625 frac[23,119] + mem[23,119] = 0
 accuracy_15_10: - 5.09765625 frac[15,10] + mem[15,10] = 0
 accuracy_21_99: - 5.09765625 frac[21,99] + mem[21,99] = 0
 accuracy_35_105: - 6 frac[35,105] + mem[35,105] = 0
 accuracy_0_5: - 5.09765625 frac[0,5] + mem[0,5] = 0
 accuracy_33_105: - 6 frac[33,105] + mem[33,105] = 0
 accuracy_38_22: - 6 frac[38,22] + mem[38,22] = 0
 accuracy_25_3: - 5.09765625 frac[25,3] + mem[25,3] = 0
 accuracy_4_11: - 5.09765625 frac[4,11] + mem[4,11] = 0
 accuracy_19_53: - 5.09765625 frac[19,53] + mem[19,53] = 0
 accuracy_36_107: - 6 frac[36,107] + mem[36,107] = 0
 accuracy_35_79: - 6 frac[35,79] + mem[35,79] = 0
 accuracy_32_91: - 6 frac[32,91] + mem[32,91] = 0
 accuracy_33_67: - 6 frac[33,67] + mem[33,67] = 0
 accuracy_37_99: - 6 frac[37,99] + mem[37,99] = 0
 accuracy_11_107: - 5.09765625 frac[11,107] + mem[11,107] = 0
 accuracy_1_58: - 5.09765625 frac[1,58] + mem[1,58] = 0
 accuracy_6_1: - 5.09765625 frac[6,1] + mem[6,1] = 0
 accuracy_38_97: - 6 frac[38,97] + mem[38,97] = 0
 accuracy_36_113: - 6 frac[36,113] + mem[36,113] = 0
 accuracy_37_77: - 6 frac[37,77] + mem[37,77] = 0
 accuracy_34_65: - 6 frac[34,65] + mem[34,65] = 0
 accuracy_35_85: - 6 frac[35,85] + mem[35,85] = 0
 accuracy_32_49: - 6 frac[32,49] + mem[32,49] = 0
 accuracy_29_55: - 5.09765625 frac[29,55] + mem[29,55] = 0
 accuracy_9_81: - 5.09765625 frac[9,81] + mem[9,81] = 0
 accuracy_33_60: - 6 frac[33,60] + mem[33,60] = 0
 accuracy_36_95: - 6 frac[36,95] + mem[36,95] = 0
 accuracy_37_87: - 6 frac[37,87] + mem[37,87] = 0
 accuracy_32_23: - 6 frac[32,23] + mem[32,23] = 0
 accuracy_13_75: - 5.09765625 frac[13,75] + mem[13,75] = 0
 accuracy_5_62: - 5.09765625 frac[5,62] + mem[5,62] = 0
 accuracy_35_34: - 6 frac[35,34] + mem[35,34] = 0
 accuracy_18_22: - 5.09765625 frac[18,22] + mem[18,22] = 0
 accuracy_33_22: - 6 frac[33,22] + mem[33,22] = 0
 accuracy_38_93: - 6 frac[38,93] + mem[38,93] = 0
 accuracy_36_37: - 6 frac[36,37] + mem[36,37] = 0
 accuracy_2_117: - 5.09765625 frac[2,117] + mem[2,117] = 0
 accuracy_17_43: - 5.09765625 frac[17,43] + mem[17,43] = 0
 accuracy_32_13: - 6 frac[32,13] + mem[32,13] = 0
 accuracy_37_56: - 6 frac[37,56] + mem[37,56] = 0
 accuracy_28_67: - 5.09765625 frac[28,67] + mem[28,67] = 0
 accuracy_36_3: - 6 frac[36,3] + mem[36,3] = 0
 accuracy_34_15: - 6 frac[34,15] + mem[34,15] = 0
 accuracy_37_2: - 6 frac[37,2] + mem[37,2] = 0
 accuracy_33_98: - 6 frac[33,98] + mem[33,98] = 0
 accuracy_25_4: - 5.09765625 frac[25,4] + mem[25,4] = 0
 accuracy_23_106: - 5.09765625 frac[23,106] + mem[23,106] = 0
 accuracy_31_115: - 5.09765625 frac[31,115] + mem[31,115] = 0
 accuracy_19_58: - 5.09765625 frac[19,58] + mem[19,58] = 0
 accuracy_37_108: - 6 frac[37,108] + mem[37,108] = 0
 accuracy_35_116: - 6 frac[35,116] + mem[35,116] = 0
 accuracy_32_82: - 6 frac[32,82] + mem[32,82] = 0
 accuracy_33_68: - 6 frac[33,68] + mem[33,68] = 0
 accuracy_14_38: - 5.09765625 frac[14,38] + mem[14,38] = 0
 accuracy_10_78: - 5.09765625 frac[10,78] + mem[10,78] = 0
 accuracy_19_32: - 5.09765625 frac[19,32] + mem[19,32] = 0
 accuracy_37_118: - 6 frac[37,118] + mem[37,118] = 0
 accuracy_26_45: - 5.09765625 frac[26,45] + mem[26,45] = 0
 accuracy_35_90: - 6 frac[35,90] + mem[35,90] = 0
 accuracy_29_48: - 5.09765625 frac[29,48] + mem[29,48] = 0
 accuracy_3_37: - 5.09765625 frac[3,37] + mem[3,37] = 0
 accuracy_33_53: - 6 frac[33,53] + mem[33,53] = 0
 accuracy_38_114: - 6 frac[38,114] + mem[38,114] = 0
 accuracy_36_70: - 6 frac[36,70] + mem[36,70] = 0
 accuracy_34_50: - 6 frac[34,50] + mem[34,50] = 0
 accuracy_8_33: - 5.09765625 frac[8,33] + mem[8,33] = 0
 accuracy_32_46: - 6 frac[32,46] + mem[32,46] = 0
 accuracy_13_68: - 5.09765625 frac[13,68] + mem[13,68] = 0
 accuracy_35_43: - 6 frac[35,43] + mem[35,43] = 0
 accuracy_18_17: - 5.09765625 frac[18,17] + mem[18,17] = 0
 accuracy_33_47: - 6 frac[33,47] + mem[33,47] = 0
 accuracy_9_60: - 5.09765625 frac[9,60] + mem[9,60] = 0
 accuracy_38_84: - 6 frac[38,84] + mem[38,84] = 0
 accuracy_36_44: - 6 frac[36,44] + mem[36,44] = 0
 accuracy_34_44: - 6 frac[34,44] + mem[34,44] = 0
 accuracy_17_44: - 5.09765625 frac[17,44] + mem[17,44] = 0
 accuracy_32_4: - 6 frac[32,4] + mem[32,4] = 0
 accuracy_22_11: - 5.09765625 frac[22,11] + mem[22,11] = 0
 accuracy_38_46: - 6 frac[38,46] + mem[38,46] = 0
 accuracy_36_10: - 6 frac[36,10] + mem[36,10] = 0
 accuracy_34_6: - 6 frac[34,6] + mem[34,6] = 0
 accuracy_15_24: - 5.09765625 frac[15,24] + mem[15,24] = 0
 accuracy_37_11: - 6 frac[37,11] + mem[37,11] = 0
 accuracy_20_85: - 5.09765625 frac[20,85] + mem[20,85] = 0
 accuracy_32_115: - 6 frac[32,115] + mem[32,115] = 0
 accuracy_5_82: - 5.09765625 frac[5,82] + mem[5,82] = 0
 accuracy_38_0: - 6 frac[38,0] + mem[38,0] = 0
 accuracy_15_6: - 5.09765625 frac[15,6] + mem[15,6] = 0
 accuracy_30_106: - 5.09765625 frac[30,106] + mem[30,106] = 0
 accuracy_33_93: - 6 frac[33,93] + mem[33,93] = 0
 accuracy_38_26: - 6 frac[38,26] + mem[38,26] = 0
 accuracy_1_52: - 5.09765625 frac[1,52] + mem[1,52] = 0
 accuracy_10_73: - 5.09765625 frac[10,73] + mem[10,73] = 0
 accuracy_34_83: - 6 frac[34,83] + mem[34,83] = 0
 accuracy_35_67: - 6 frac[35,67] + mem[35,67] = 0
 accuracy_3_42: - 5.09765625 frac[3,42] + mem[3,42] = 0
 accuracy_16_2: - 5.09765625 frac[16,2] + mem[16,2] = 0
 accuracy_36_77: - 6 frac[36,77] + mem[36,77] = 0
 accuracy_32_37: - 6 frac[32,37] + mem[32,37] = 0
 accuracy_12_93: - 5.09765625 frac[12,93] + mem[12,93] = 0
 accuracy_33_32: - 6 frac[33,32] + mem[33,32] = 0
 accuracy_16_24: - 5.09765625 frac[16,24] + mem[16,24] = 0
 accuracy_38_79: - 6 frac[38,79] + mem[38,79] = 0
 accuracy_24_75: - 5.09765625 frac[24,75] + mem[24,75] = 0
 accuracy_36_43: - 6 frac[36,43] + mem[36,43] = 0
 accuracy_17_37: - 5.09765625 frac[17,37] + mem[17,37] = 0
 accuracy_32_27: - 6 frac[32,27] + mem[32,27] = 0
 accuracy_22_2: - 5.09765625 frac[22,2] + mem[22,2] = 0
 accuracy_13_87: - 5.09765625 frac[13,87] + mem[13,87] = 0
 accuracy_36_49: - 6 frac[36,49] + mem[36,49] = 0
 accuracy_34_1: - 6 frac[34,1] + mem[34,1] = 0
 accuracy_2_97: - 5.09765625 frac[2,97] + mem[2,97] = 0
 accuracy_37_52: - 6 frac[37,52] + mem[37,52] = 0
 accuracy_36_31: - 6 frac[36,31] + mem[36,31] = 0
 accuracy_37_30: - 6 frac[37,30] + mem[37,30] = 0
 accuracy_34_112: - 6 frac[34,112] + mem[34,112] = 0
 accuracy_35_98: - 6 frac[35,98] + mem[35,98] = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 accuracy_24_61: - 5.09765625 frac[24,61] + mem[24,61] = 0
 accuracy_32_96: - 6 frac[32,96] + mem[32,96] = 0
 accuracy_38_29: - 6 frac[38,29] + mem[38,29] = 0
 accuracy_30_119: - 5.09765625 frac[30,119] + mem[30,119] = 0
 accuracy_19_46: - 5.09765625 frac[19,46] + mem[19,46] = 0
 accuracy_34_106: - 6 frac[34,106] + mem[34,106] = 0
 accuracy_35_72: - 6 frac[35,72] + mem[35,72] = 0
 accuracy_11_116: - 5.09765625 frac[11,116] + mem[11,116] = 0
 accuracy_27_62: - 5.09765625 frac[27,62] + mem[27,62] = 0
 accuracy_9_68: - 5.09765625 frac[9,68] + mem[9,68] = 0
 accuracy_38_108: - 6 frac[38,108] + mem[38,108] = 0
 accuracy_4_20: - 5.09765625 frac[4,20] + mem[4,20] = 0
 accuracy_36_116: - 6 frac[36,116] + mem[36,116] = 0
 accuracy_27_84: - 5.09765625 frac[27,84] + mem[27,84] = 0
 accuracy_33_57: - 6 frac[33,57] + mem[33,57] = 0
 accuracy_24_82: - 5.09765625 frac[24,82] + mem[24,82] = 0
 accuracy_38_70: - 6 frac[38,70] + mem[38,70] = 0
 accuracy_36_82: - 6 frac[36,82] + mem[36,82] = 0
 accuracy_34_62: - 6 frac[34,62] + mem[34,62] = 0
 accuracy_8_53: - 5.09765625 frac[8,53] + mem[8,53] = 0
 accuracy_22_5: - 5.09765625 frac[22,5] + mem[22,5] = 0
 accuracy_35_63: - 6 frac[35,63] + mem[35,63] = 0
 accuracy_18_29: - 5.09765625 frac[18,29] + mem[18,29] = 0
 accuracy_27_74: - 5.09765625 frac[27,74] + mem[27,74] = 0
 accuracy_38_88: - 6 frac[38,88] + mem[38,88] = 0
 accuracy_36_56: - 6 frac[36,56] + mem[36,56] = 0
 accuracy_34_24: - 6 frac[34,24] + mem[34,24] = 0
 accuracy_37_61: - 6 frac[37,61] + mem[37,61] = 0
 accuracy_28_70: - 5.09765625 frac[28,70] + mem[28,70] = 0
 accuracy_33_117: - 6 frac[33,117] + mem[33,117] = 0
 accuracy_38_50: - 6 frac[38,50] + mem[38,50] = 0
 accuracy_36_6: - 6 frac[36,6] + mem[36,6] = 0
 accuracy_23_113: - 5.09765625 frac[23,113] + mem[23,113] = 0
 accuracy_21_97: - 5.09765625 frac[21,97] + mem[21,97] = 0
 accuracy_37_7: - 6 frac[37,7] + mem[37,7] = 0
 accuracy_20_65: - 5.09765625 frac[20,65] + mem[20,65] = 0
 accuracy_35_107: - 6 frac[35,107] + mem[35,107] = 0
 accuracy_0_7: - 5.09765625 frac[0,7] + mem[0,7] = 0
 accuracy_33_111: - 6 frac[33,111] + mem[33,111] = 0
 accuracy_25_1: - 5.09765625 frac[25,1] + mem[25,1] = 0
 accuracy_23_111: - 5.09765625 frac[23,111] + mem[23,111] = 0
 accuracy_31_116: - 5.09765625 frac[31,116] + mem[31,116] = 0
 accuracy_19_55: - 5.09765625 frac[19,55] + mem[19,55] = 0
 accuracy_34_101: - 6 frac[34,101] + mem[34,101] = 0
 accuracy_37_97: - 6 frac[37,97] + mem[37,97] = 0
 accuracy_7_104: - 5.09765625 frac[7,104] + mem[7,104] = 0
 accuracy_33_65: - 6 frac[33,65] + mem[33,65] = 0
 accuracy_35_113: - 6 frac[35,113] + mem[35,113] = 0
 accuracy_1_56: - 5.09765625 frac[1,56] + mem[1,56] = 0
 accuracy_6_7: - 5.09765625 frac[6,7] + mem[6,7] = 0
 accuracy_4_19: - 5.09765625 frac[4,19] + mem[4,19] = 0
 accuracy_37_75: - 6 frac[37,75] + mem[37,75] = 0
 accuracy_36_115: - 6 frac[36,115] + mem[36,115] = 0
 accuracy_26_32: - 5.09765625 frac[26,32] + mem[26,32] = 0
 accuracy_33_50: - 6 frac[33,50] + mem[33,50] = 0
 accuracy_16_22: - 5.09765625 frac[16,22] + mem[16,22] = 0
 accuracy_37_85: - 6 frac[37,85] + mem[37,85] = 0
 accuracy_34_57: - 6 frac[34,57] + mem[34,57] = 0
 accuracy_13_73: - 5.09765625 frac[13,73] + mem[13,73] = 0
 accuracy_29_47: - 5.09765625 frac[29,47] + mem[29,47] = 0
 accuracy_35_36: - 6 frac[35,36] + mem[35,36] = 0
 accuracy_18_20: - 5.09765625 frac[18,20] + mem[18,20] = 0
 accuracy_27_67: - 5.09765625 frac[27,67] + mem[27,67] = 0
 accuracy_38_83: - 6 frac[38,83] + mem[38,83] = 0
 accuracy_36_39: - 6 frac[36,39] + mem[36,39] = 0
 accuracy_10_62: - 5.09765625 frac[10,62] + mem[10,62] = 0
 accuracy_34_19: - 6 frac[34,19] + mem[34,19] = 0
 accuracy_2_115: - 5.09765625 frac[2,115] + mem[2,115] = 0
 accuracy_37_38: - 6 frac[37,38] + mem[37,38] = 0
 accuracy_38_53: - 6 frac[38,53] + mem[38,53] = 0
 accuracy_36_13: - 6 frac[36,13] + mem[36,13] = 0
 accuracy_34_13: - 6 frac[34,13] + mem[34,13] = 0
 accuracy_30_95: - 5.09765625 frac[30,95] + mem[30,95] = 0
 accuracy_37_0: - 6 frac[37,0] + mem[37,0] = 0
 accuracy_1_31: - 5.09765625 frac[1,31] + mem[1,31] = 0
 accuracy_23_100: - 5.09765625 frac[23,100] + mem[23,100] = 0
 accuracy_37_106: - 6 frac[37,106] + mem[37,106] = 0
 accuracy_32_84: - 6 frac[32,84] + mem[32,84] = 0
 accuracy_7_97: - 5.09765625 frac[7,97] + mem[7,97] = 0
 accuracy_14_36: - 5.09765625 frac[14,36] + mem[14,36] = 0
 accuracy_20_63: - 5.09765625 frac[20,63] + mem[20,63] = 0
 accuracy_1_49: - 5.09765625 frac[1,49] + mem[1,49] = 0
 accuracy_4_26: - 5.09765625 frac[4,26] + mem[4,26] = 0
 accuracy_19_34: - 5.09765625 frac[19,34] + mem[19,34] = 0
 accuracy_37_116: - 6 frac[37,116] + mem[37,116] = 0
 accuracy_26_43: - 5.09765625 frac[26,43] + mem[26,43] = 0
 accuracy_35_92: - 6 frac[35,92] + mem[35,92] = 0
 accuracy_3_39: - 5.09765625 frac[3,39] + mem[3,39] = 0
 accuracy_1_43: - 5.09765625 frac[1,43] + mem[1,43] = 0
 accuracy_16_13: - 5.09765625 frac[16,13] + mem[16,13] = 0
 accuracy_38_112: - 6 frac[38,112] + mem[38,112] = 0
 accuracy_36_64: - 6 frac[36,64] + mem[36,64] = 0
 accuracy_37_94: - 6 frac[37,94] + mem[37,94] = 0
 accuracy_8_35: - 5.09765625 frac[8,35] + mem[8,35] = 0
 accuracy_32_32: - 6 frac[32,32] + mem[32,32] = 0
 accuracy_22_23: - 5.09765625 frac[22,23] + mem[22,23] = 0
 accuracy_11_94: - 5.09765625 frac[11,94] + mem[11,94] = 0
 accuracy_35_45: - 6 frac[35,45] + mem[35,45] = 0
 accuracy_18_15: - 5.09765625 frac[18,15] + mem[18,15] = 0
 accuracy_24_70: - 5.09765625 frac[24,70] + mem[24,70] = 0
 accuracy_38_74: - 6 frac[38,74] + mem[38,74] = 0
 accuracy_36_46: - 6 frac[36,46] + mem[36,46] = 0
 accuracy_32_6: - 6 frac[32,6] + mem[32,6] = 0
 accuracy_22_9: - 5.09765625 frac[22,9] + mem[22,9] = 0
 accuracy_37_47: - 6 frac[37,47] + mem[37,47] = 0
 accuracy_35_51: - 6 frac[35,51] + mem[35,51] = 0
 accuracy_38_44: - 6 frac[38,44] + mem[38,44] = 0
 accuracy_36_52: - 6 frac[36,52] + mem[36,52] = 0
 accuracy_34_4: - 6 frac[34,4] + mem[34,4] = 0
 accuracy_2_100: - 5.09765625 frac[2,100] + mem[2,100] = 0
 accuracy_37_9: - 6 frac[37,9] + mem[37,9] = 0
 accuracy_20_87: - 5.09765625 frac[20,87] + mem[20,87] = 0
 accuracy_32_117: - 6 frac[32,117] + mem[32,117] = 0
 accuracy_38_6: - 6 frac[38,6] + mem[38,6] = 0
 accuracy_15_0: - 5.09765625 frac[15,0] + mem[15,0] = 0
 accuracy_30_104: - 5.09765625 frac[30,104] + mem[30,104] = 0
 accuracy_37_19: - 6 frac[37,19] + mem[37,19] = 0
 accuracy_34_119: - 6 frac[34,119] + mem[34,119] = 0
 accuracy_0_11: - 5.09765625 frac[0,11] + mem[0,11] = 0
 accuracy_33_83: - 6 frac[33,83] + mem[33,83] = 0
 accuracy_14_47: - 5.09765625 frac[14,47] + mem[14,47] = 0
 accuracy_38_24: - 6 frac[38,24] + mem[38,24] = 0
 accuracy_30_114: - 5.09765625 frac[30,114] + mem[30,114] = 0
 accuracy_36_97: - 6 frac[36,97] + mem[36,97] = 0
 accuracy_32_65: - 6 frac[32,65] + mem[32,65] = 0
 accuracy_7_116: - 5.09765625 frac[7,116] + mem[7,116] = 0
 accuracy_3_44: - 5.09765625 frac[3,44] + mem[3,44] = 0
 accuracy_9_65: - 5.09765625 frac[9,65] + mem[9,65] = 0
 accuracy_1_44: - 5.09765625 frac[1,44] + mem[1,44] = 0
 accuracy_16_4: - 5.09765625 frac[16,4] + mem[16,4] = 0
 accuracy_10_81: - 5.09765625 frac[10,81] + mem[10,81] = 0
 accuracy_38_107: - 6 frac[38,107] + mem[38,107] = 0
 accuracy_36_79: - 6 frac[36,79] + mem[36,79] = 0
 accuracy_28_60: - 5.09765625 frac[28,60] + mem[28,60] = 0
 accuracy_34_75: - 6 frac[34,75] + mem[34,75] = 0
 accuracy_8_42: - 5.09765625 frac[8,42] + mem[8,42] = 0
 accuracy_32_39: - 6 frac[32,39] + mem[32,39] = 0
 accuracy_33_38: - 6 frac[33,38] + mem[33,38] = 0
 accuracy_16_26: - 5.09765625 frac[16,26] + mem[16,26] = 0
 accuracy_38_77: - 6 frac[38,77] + mem[38,77] = 0
 accuracy_24_77: - 5.09765625 frac[24,77] + mem[24,77] = 0
 accuracy_36_85: - 6 frac[36,85] + mem[36,85] = 0
 accuracy_34_37: - 6 frac[34,37] + mem[34,37] = 0
 accuracy_32_29: - 6 frac[32,29] + mem[32,29] = 0
 accuracy_22_0: - 5.09765625 frac[22,0] + mem[22,0] = 0
 accuracy_37_40: - 6 frac[37,40] + mem[37,40] = 0
 accuracy_28_83: - 5.09765625 frac[28,83] + mem[28,83] = 0
 accuracy_33_24: - 6 frac[33,24] + mem[33,24] = 0
 accuracy_36_51: - 6 frac[36,51] + mem[36,51] = 0
 accuracy_37_50: - 6 frac[37,50] + mem[37,50] = 0
 accuracy_28_73: - 5.09765625 frac[28,73] + mem[28,73] = 0
 accuracy_35_30: - 6 frac[35,30] + mem[35,30] = 0
 accuracy_33_114: - 6 frac[33,114] + mem[33,114] = 0
 accuracy_38_57: - 6 frac[38,57] + mem[38,57] = 0
 accuracy_21_102: - 5.09765625 frac[21,102] + mem[21,102] = 0
 accuracy_35_100: - 6 frac[35,100] + mem[35,100] = 0
 accuracy_32_98: - 6 frac[32,98] + mem[32,98] = 0
 accuracy_33_84: - 6 frac[33,84] + mem[33,84] = 0
 accuracy_14_54: - 5.09765625 frac[14,54] + mem[14,54] = 0
 accuracy_38_19: - 6 frac[38,19] + mem[38,19] = 0
 accuracy_12_98: - 5.09765625 frac[12,98] + mem[12,98] = 0
 accuracy_36_104: - 6 frac[36,104] + mem[36,104] = 0
 accuracy_37_102: - 6 frac[37,102] + mem[37,102] = 0
 accuracy_34_104: - 6 frac[34,104] + mem[34,104] = 0
 accuracy_35_74: - 6 frac[35,74] + mem[35,74] = 0
 accuracy_32_88: - 6 frac[32,88] + mem[32,88] = 0
 accuracy_33_78: - 6 frac[33,78] + mem[33,78] = 0
 accuracy_11_118: - 5.09765625 frac[11,118] + mem[11,118] = 0
 accuracy_38_98: - 6 frac[38,98] + mem[38,98] = 0
 accuracy_4_22: - 5.09765625 frac[4,22] + mem[4,22] = 0
 accuracy_37_64: - 6 frac[37,64] + mem[37,64] = 0
 accuracy_36_118: - 6 frac[36,118] + mem[36,118] = 0
 accuracy_32_62: - 6 frac[32,62] + mem[32,62] = 0
 accuracy_29_58: - 5.09765625 frac[29,58] + mem[29,58] = 0
 accuracy_27_86: - 5.09765625 frac[27,86] + mem[27,86] = 0
 accuracy_33_63: - 6 frac[33,63] + mem[33,63] = 0
 accuracy_38_68: - 6 frac[38,68] + mem[38,68] = 0
 accuracy_36_92: - 6 frac[36,92] + mem[36,92] = 0
 accuracy_32_20: - 6 frac[32,20] + mem[32,20] = 0
 accuracy_22_27: - 5.09765625 frac[22,27] + mem[22,27] = 0
 accuracy_13_78: - 5.09765625 frac[13,78] + mem[13,78] = 0
 accuracy_38_94: - 6 frac[38,94] + mem[38,94] = 0
 accuracy_36_58: - 6 frac[36,58] + mem[36,58] = 0
 accuracy_34_22: - 6 frac[34,22] + mem[34,22] = 0
 accuracy_17_54: - 5.09765625 frac[17,54] + mem[17,54] = 0
 accuracy_32_10: - 6 frac[32,10] + mem[32,10] = 0
 accuracy_2_118: - 5.09765625 frac[2,118] + mem[2,118] = 0
 accuracy_28_64: - 5.09765625 frac[28,64] + mem[28,64] = 0
 accuracy_35_7: - 6 frac[35,7] + mem[35,7] = 0
 accuracy_38_48: - 6 frac[38,48] + mem[38,48] = 0
 accuracy_36_0: - 6 frac[36,0] + mem[36,0] = 0
 accuracy_15_22: - 5.09765625 frac[15,22] + mem[15,22] = 0
 accuracy_30_90: - 5.09765625 frac[30,90] + mem[30,90] = 0
 accuracy_37_5: - 6 frac[37,5] + mem[37,5] = 0
 accuracy_0_25: - 5.09765625 frac[0,25] + mem[0,25] = 0
 accuracy_14_57: - 5.09765625 frac[14,57] + mem[14,57] = 0
 accuracy_38_10: - 6 frac[38,10] + mem[38,10] = 0
 accuracy_25_7: - 5.09765625 frac[25,7] + mem[25,7] = 0
 accuracy_37_111: - 6 frac[37,111] + mem[37,111] = 0
 accuracy_34_99: - 6 frac[34,99] + mem[34,99] = 0
 accuracy_35_115: - 6 frac[35,115] + mem[35,115] = 0
 accuracy_32_95: - 6 frac[32,95] + mem[32,95] = 0
 accuracy_11_111: - 5.09765625 frac[11,111] + mem[11,111] = 0
 accuracy_10_67: - 5.09765625 frac[10,67] + mem[10,67] = 0
 accuracy_38_101: - 6 frac[38,101] + mem[38,101] = 0
 accuracy_37_73: - 6 frac[37,73] + mem[37,73] = 0
 accuracy_26_46: - 5.09765625 frac[26,46] + mem[26,46] = 0
 accuracy_29_51: - 5.09765625 frac[29,51] + mem[29,51] = 0
 accuracy_3_32: - 5.09765625 frac[3,32] + mem[3,32] = 0
 accuracy_36_91: - 6 frac[36,91] + mem[36,91] = 0
 accuracy_37_83: - 6 frac[37,83] + mem[37,83] = 0
 accuracy_34_55: - 6 frac[34,55] + mem[34,55] = 0
 accuracy_32_43: - 6 frac[32,43] + mem[32,43] = 0
 accuracy_22_18: - 5.09765625 frac[22,18] + mem[22,18] = 0
 accuracy_35_38: - 6 frac[35,38] + mem[35,38] = 0
 accuracy_33_42: - 6 frac[33,42] + mem[33,42] = 0
 accuracy_24_65: - 5.09765625 frac[24,65] + mem[24,65] = 0
 accuracy_36_33: - 6 frac[36,33] + mem[36,33] = 0
 accuracy_34_17: - 6 frac[34,17] + mem[34,17] = 0
 accuracy_17_47: - 5.09765625 frac[17,47] + mem[17,47] = 0
 accuracy_32_1: - 6 frac[32,1] + mem[32,1] = 0
 accuracy_37_36: - 6 frac[37,36] + mem[37,36] = 0
 accuracy_38_43: - 6 frac[38,43] + mem[38,43] = 0
 accuracy_34_11: - 6 frac[34,11] + mem[34,11] = 0
 accuracy_21_104: - 5.09765625 frac[21,104] + mem[21,104] = 0
 accuracy_30_93: - 5.09765625 frac[30,93] + mem[30,93] = 0
 accuracy_20_74: - 5.09765625 frac[20,74] + mem[20,74] = 0
 accuracy_32_112: - 6 frac[32,112] + mem[32,112] = 0
 accuracy_38_13: - 6 frac[38,13] + mem[38,13] = 0
 accuracy_37_104: - 6 frac[37,104] + mem[37,104] = 0
 accuracy_11_100: - 5.09765625 frac[11,100] + mem[11,100] = 0
 accuracy_10_74: - 5.09765625 frac[10,74] + mem[10,74] = 0
 accuracy_23_92: - 5.09765625 frac[23,92] + mem[23,92] = 0
 accuracy_36_100: - 6 frac[36,100] + mem[36,100] = 0
 accuracy_37_114: - 6 frac[37,114] + mem[37,114] = 0
 accuracy_34_84: - 6 frac[34,84] + mem[34,84] = 0
 accuracy_35_94: - 6 frac[35,94] + mem[35,94] = 0
 accuracy_32_76: - 6 frac[32,76] + mem[32,76] = 0
 accuracy_10_84: - 5.09765625 frac[10,84] + mem[10,84] = 0
 accuracy_38_118: - 6 frac[38,118] + mem[38,118] = 0
 accuracy_37_92: - 6 frac[37,92] + mem[37,92] = 0
 accuracy_13_64: - 5.09765625 frac[13,64] + mem[13,64] = 0
 accuracy_35_47: - 6 frac[35,47] + mem[35,47] = 0
 accuracy_33_35: - 6 frac[33,35] + mem[33,35] = 0
 accuracy_24_72: - 5.09765625 frac[24,72] + mem[24,72] = 0
 accuracy_36_40: - 6 frac[36,40] + mem[36,40] = 0
 accuracy_34_40: - 6 frac[34,40] + mem[34,40] = 0
 accuracy_17_32: - 5.09765625 frac[17,32] + mem[17,32] = 0
 accuracy_32_24: - 6 frac[32,24] + mem[32,24] = 0
 accuracy_22_15: - 5.09765625 frac[22,15] + mem[22,15] = 0
 accuracy_37_45: - 6 frac[37,45] + mem[37,45] = 0
 accuracy_28_86: - 5.09765625 frac[28,86] + mem[28,86] = 0
 accuracy_36_54: - 6 frac[36,54] + mem[36,54] = 0
 accuracy_34_2: - 6 frac[34,2] + mem[34,2] = 0
 accuracy_2_98: - 5.09765625 frac[2,98] + mem[2,98] = 0
 accuracy_31_94: - 5.09765625 frac[31,94] + mem[31,94] = 0
 accuracy_37_55: - 6 frac[37,55] + mem[37,55] = 0
 accuracy_0_23: - 5.09765625 frac[0,23] + mem[0,23] = 0
 accuracy_32_119: - 6 frac[32,119] + mem[32,119] = 0
 accuracy_38_4: - 6 frac[38,4] + mem[38,4] = 0
 accuracy_21_91: - 5.09765625 frac[21,91] + mem[21,91] = 0
 accuracy_37_17: - 6 frac[37,17] + mem[37,17] = 0
 accuracy_0_13: - 5.09765625 frac[0,13] + mem[0,13] = 0
 accuracy_33_81: - 6 frac[33,81] + mem[33,81] = 0
 accuracy_38_30: - 6 frac[38,30] + mem[38,30] = 0
 accuracy_12_101: - 5.09765625 frac[12,101] + mem[12,101] = 0
 accuracy_4_3: - 5.09765625 frac[4,3] + mem[4,3] = 0
 accuracy_36_99: - 6 frac[36,99] + mem[36,99] = 0
 accuracy_34_111: - 6 frac[34,111] + mem[34,111] = 0
 accuracy_35_71: - 6 frac[35,71] + mem[35,71] = 0
 accuracy_32_67: - 6 frac[32,67] + mem[32,67] = 0
 accuracy_33_75: - 6 frac[33,75] + mem[33,75] = 0
 accuracy_7_118: - 5.09765625 frac[7,118] + mem[7,118] = 0
 accuracy_3_46: - 5.09765625 frac[3,46] + mem[3,46] = 0
 accuracy_27_61: - 5.09765625 frac[27,61] + mem[27,61] = 0
 accuracy_16_6: - 5.09765625 frac[16,6] + mem[16,6] = 0
 accuracy_38_105: - 6 frac[38,105] + mem[38,105] = 0
 accuracy_36_73: - 6 frac[36,73] + mem[36,73] = 0
 accuracy_28_62: - 5.09765625 frac[28,62] + mem[28,62] = 0
 accuracy_34_73: - 6 frac[34,73] + mem[34,73] = 0
 accuracy_8_44: - 5.09765625 frac[8,44] + mem[8,44] = 0
 accuracy_32_57: - 6 frac[32,57] + mem[32,57] = 0
 accuracy_3_52: - 5.09765625 frac[3,52] + mem[3,52] = 0
 accuracy_18_4: - 5.09765625 frac[18,4] + mem[18,4] = 0
 accuracy_33_36: - 6 frac[33,36] + mem[33,36] = 0
 accuracy_24_79: - 5.09765625 frac[24,79] + mem[24,79] = 0
 accuracy_36_87: - 6 frac[36,87] + mem[36,87] = 0
 accuracy_8_50: - 5.09765625 frac[8,50] + mem[8,50] = 0
 accuracy_32_31: - 6 frac[32,31] + mem[32,31] = 0
 accuracy_13_83: - 5.09765625 frac[13,83] + mem[13,83] = 0
 accuracy_35_58: - 6 frac[35,58] + mem[35,58] = 0
 accuracy_38_37: - 6 frac[38,37] + mem[38,37] = 0
 accuracy_36_61: - 6 frac[36,61] + mem[36,61] = 0
 accuracy_34_29: - 6 frac[34,29] + mem[34,29] = 0
 accuracy_37_48: - 6 frac[37,48] + mem[37,48] = 0
 accuracy_28_75: - 5.09765625 frac[28,75] + mem[28,75] = 0
 accuracy_36_27: - 6 frac[36,27] + mem[36,27] = 0
 accuracy_25_26: - 5.09765625 frac[25,26] + mem[25,26] = 0
 accuracy_23_116: - 5.09765625 frac[23,116] + mem[23,116] = 0
 accuracy_21_100: - 5.09765625 frac[21,100] + mem[21,100] = 0
 accuracy_37_26: - 6 frac[37,26] + mem[37,26] = 0
 accuracy_0_4: - 5.09765625 frac[0,4] + mem[0,4] = 0
 accuracy_32_100: - 6 frac[32,100] + mem[32,100] = 0
 accuracy_14_52: - 5.09765625 frac[14,52] + mem[14,52] = 0
 accuracy_38_17: - 6 frac[38,17] + mem[38,17] = 0
 accuracy_36_106: - 6 frac[36,106] + mem[36,106] = 0
 accuracy_37_100: - 6 frac[37,100] + mem[37,100] = 0
 accuracy_34_102: - 6 frac[34,102] + mem[34,102] = 0
 accuracy_32_90: - 6 frac[32,90] + mem[32,90] = 0
 accuracy_33_76: - 6 frac[33,76] + mem[33,76] = 0
 accuracy_29_30: - 5.09765625 frac[29,30] + mem[29,30] = 0
 accuracy_11_104: - 5.09765625 frac[11,104] + mem[11,104] = 0
 accuracy_6_0: - 5.09765625 frac[6,0] + mem[6,0] = 0
 accuracy_38_96: - 6 frac[38,96] + mem[38,96] = 0
 accuracy_36_112: - 6 frac[36,112] + mem[36,112] = 0
 accuracy_35_82: - 6 frac[35,82] + mem[35,82] = 0
 accuracy_32_48: - 6 frac[32,48] + mem[32,48] = 0
 accuracy_9_82: - 5.09765625 frac[9,82] + mem[9,82] = 0
 accuracy_33_61: - 6 frac[33,61] + mem[33,61] = 0
 accuracy_6_26: - 5.09765625 frac[6,26] + mem[6,26] = 0
 accuracy_36_94: - 6 frac[36,94] + mem[36,94] = 0
 accuracy_34_58: - 6 frac[34,58] + mem[34,58] = 0
 accuracy_8_57: - 5.09765625 frac[8,57] + mem[8,57] = 0
 accuracy_32_22: - 6 frac[32,22] + mem[32,22] = 0
 accuracy_22_25: - 5.09765625 frac[22,25] + mem[22,25] = 0
 accuracy_5_63: - 5.09765625 frac[5,63] + mem[5,63] = 0
 accuracy_13_76: - 5.09765625 frac[13,76] + mem[13,76] = 0
 accuracy_18_25: - 5.09765625 frac[18,25] + mem[18,25] = 0
 accuracy_38_92: - 6 frac[38,92] + mem[38,92] = 0
 accuracy_36_36: - 6 frac[36,36] + mem[36,36] = 0
 accuracy_34_20: - 6 frac[34,20] + mem[34,20] = 0
 accuracy_37_57: - 6 frac[37,57] + mem[37,57] = 0
 accuracy_35_9: - 6 frac[35,9] + mem[35,9] = 0
 accuracy_38_54: - 6 frac[38,54] + mem[38,54] = 0
 accuracy_34_14: - 6 frac[34,14] + mem[34,14] = 0
 accuracy_37_3: - 6 frac[37,3] + mem[37,3] = 0
 accuracy_35_111: - 6 frac[35,111] + mem[35,111] = 0
 accuracy_0_27: - 5.09765625 frac[0,27] + mem[0,27] = 0
 accuracy_34_97: - 6 frac[34,97] + mem[34,97] = 0
 accuracy_7_100: - 5.09765625 frac[7,100] + mem[7,100] = 0
 accuracy_10_65: - 5.09765625 frac[10,65] + mem[10,65] = 0
 accuracy_37_119: - 6 frac[37,119] + mem[37,119] = 0
 accuracy_34_91: - 6 frac[34,91] + mem[34,91] = 0
 accuracy_33_54: - 6 frac[33,54] + mem[33,54] = 0
 accuracy_16_10: - 5.09765625 frac[16,10] + mem[16,10] = 0
 accuracy_6_29: - 5.09765625 frac[6,29] + mem[6,29] = 0
 accuracy_34_53: - 6 frac[34,53] + mem[34,53] = 0
 accuracy_32_45: - 6 frac[32,45] + mem[32,45] = 0
 accuracy_9_61: - 5.09765625 frac[9,61] + mem[9,61] = 0
 accuracy_33_40: - 6 frac[33,40] + mem[33,40] = 0
 accuracy_27_71: - 5.09765625 frac[27,71] + mem[27,71] = 0
 accuracy_38_87: - 6 frac[38,87] + mem[38,87] = 0
 accuracy_36_35: - 6 frac[36,35] + mem[36,35] = 0
 accuracy_34_47: - 6 frac[34,47] + mem[34,47] = 0
 accuracy_32_3: - 6 frac[32,3] + mem[32,3] = 0
 accuracy_22_10: - 5.09765625 frac[22,10] + mem[22,10] = 0
 accuracy_37_34: - 6 frac[37,34] + mem[37,34] = 0
 accuracy_5_73: - 5.09765625 frac[5,73] + mem[5,73] = 0
 accuracy_34_9: - 6 frac[34,9] + mem[34,9] = 0
 accuracy_20_84: - 5.09765625 frac[20,84] + mem[20,84] = 0
 accuracy_33_100: - 6 frac[33,100] + mem[33,100] = 0
 accuracy_5_83: - 5.09765625 frac[5,83] + mem[5,83] = 0
 accuracy_38_3: - 6 frac[38,3] + mem[38,3] = 0
 accuracy_36_23: - 6 frac[36,23] + mem[36,23] = 0
 accuracy_30_101: - 5.09765625 frac[30,101] + mem[30,101] = 0
 accuracy_37_22: - 6 frac[37,22] + mem[37,22] = 0
 accuracy_32_104: - 6 frac[32,104] + mem[32,104] = 0
 accuracy_33_94: - 6 frac[33,94] + mem[33,94] = 0
 accuracy_14_40: - 5.09765625 frac[14,40] + mem[14,40] = 0
 accuracy_4_6: - 5.09765625 frac[4,6] + mem[4,6] = 0
 accuracy_19_38: - 5.09765625 frac[19,38] + mem[19,38] = 0
 accuracy_34_82: - 6 frac[34,82] + mem[34,82] = 0
 accuracy_35_64: - 6 frac[35,64] + mem[35,64] = 0
 accuracy_37_112: - 6 frac[37,112] + mem[37,112] = 0
 accuracy_36_102: - 6 frac[36,102] + mem[36,102] = 0
 accuracy_38_116: - 6 frac[38,116] + mem[38,116] = 0
 accuracy_36_76: - 6 frac[36,76] + mem[36,76] = 0
 accuracy_37_90: - 6 frac[37,90] + mem[37,90] = 0
 accuracy_29_36: - 5.09765625 frac[29,36] + mem[29,36] = 0
 accuracy_12_92: - 5.09765625 frac[12,92] + mem[12,92] = 0
 accuracy_33_33: - 6 frac[33,33] + mem[33,33] = 0
 accuracy_36_42: - 6 frac[36,42] + mem[36,42] = 0
 accuracy_34_38: - 6 frac[34,38] + mem[34,38] = 0
 accuracy_17_38: - 5.09765625 frac[17,38] + mem[17,38] = 0
 accuracy_32_26: - 6 frac[32,26] + mem[32,26] = 0
 accuracy_22_13: - 5.09765625 frac[22,13] + mem[22,13] = 0
 accuracy_37_43: - 6 frac[37,43] + mem[37,43] = 0
 accuracy_35_55: - 6 frac[35,55] + mem[35,55] = 0
 accuracy_38_32: - 6 frac[38,32] + mem[38,32] = 0
 accuracy_36_48: - 6 frac[36,48] + mem[36,48] = 0
 accuracy_34_0: - 6 frac[34,0] + mem[34,0] = 0
 accuracy_2_96: - 5.09765625 frac[2,96] + mem[2,96] = 0
 accuracy_37_53: - 6 frac[37,53] + mem[37,53] = 0
 accuracy_20_83: - 5.09765625 frac[20,83] + mem[20,83] = 0
 accuracy_38_58: - 6 frac[38,58] + mem[38,58] = 0
 accuracy_36_30: - 6 frac[36,30] + mem[36,30] = 0
 accuracy_37_31: - 6 frac[37,31] + mem[37,31] = 0
 accuracy_7_90: - 5.09765625 frac[7,90] + mem[7,90] = 0
 accuracy_33_87: - 6 frac[33,87] + mem[33,87] = 0
 accuracy_35_73: - 6 frac[35,73] + mem[35,73] = 0
 accuracy_7_112: - 5.09765625 frac[7,112] + mem[7,112] = 0
 accuracy_33_73: - 6 frac[33,73] + mem[33,73] = 0
 accuracy_11_117: - 5.09765625 frac[11,117] + mem[11,117] = 0
 accuracy_1_32: - 5.09765625 frac[1,32] + mem[1,32] = 0
 accuracy_38_111: - 6 frac[38,111] + mem[38,111] = 0
 accuracy_36_75: - 6 frac[36,75] + mem[36,75] = 0
 accuracy_8_46: - 5.09765625 frac[8,46] + mem[8,46] = 0
 accuracy_3_54: - 5.09765625 frac[3,54] + mem[3,54] = 0
 accuracy_32_17: - 6 frac[32,17] + mem[32,17] = 0
 accuracy_22_4: - 5.09765625 frac[22,4] + mem[22,4] = 0
 accuracy_35_60: - 6 frac[35,60] + mem[35,60] = 0
 accuracy_27_75: - 5.09765625 frac[27,75] + mem[27,75] = 0
 accuracy_38_91: - 6 frac[38,91] + mem[38,91] = 0
 accuracy_36_63: - 6 frac[36,63] + mem[36,63] = 0
 accuracy_34_27: - 6 frac[34,27] + mem[34,27] = 0
 accuracy_37_62: - 6 frac[37,62] + mem[37,62] = 0
 accuracy_35_2: - 6 frac[35,2] + mem[35,2] = 0
 accuracy_33_118: - 6 frac[33,118] + mem[33,118] = 0
 accuracy_38_61: - 6 frac[38,61] + mem[38,61] = 0
 accuracy_36_5: - 6 frac[36,5] + mem[36,5] = 0
 accuracy_25_24: - 5.09765625 frac[25,24] + mem[25,24] = 0
 accuracy_23_118: - 5.09765625 frac[23,118] + mem[23,118] = 0
 accuracy_31_111: - 5.09765625 frac[31,111] + mem[31,111] = 0
 accuracy_37_24: - 6 frac[37,24] + mem[37,24] = 0
 accuracy_35_104: - 6 frac[35,104] + mem[35,104] = 0
 accuracy_32_102: - 6 frac[32,102] + mem[32,102] = 0
 accuracy_33_104: - 6 frac[33,104] + mem[33,104] = 0
 accuracy_38_23: - 6 frac[38,23] + mem[38,23] = 0
 accuracy_25_2: - 5.09765625 frac[25,2] + mem[25,2] = 0
 accuracy_23_108: - 5.09765625 frac[23,108] + mem[23,108] = 0
 accuracy_19_52: - 5.09765625 frac[19,52] + mem[19,52] = 0
 accuracy_37_98: - 6 frac[37,98] + mem[37,98] = 0
 accuracy_34_100: - 6 frac[34,100] + mem[34,100] = 0
 accuracy_12_116: - 5.09765625 frac[12,116] + mem[12,116] = 0
 accuracy_1_57: - 5.09765625 frac[1,57] + mem[1,57] = 0
 accuracy_38_102: - 6 frac[38,102] + mem[38,102] = 0
 accuracy_36_114: - 6 frac[36,114] + mem[36,114] = 0
 accuracy_34_94: - 6 frac[34,94] + mem[34,94] = 0
 accuracy_35_84: - 6 frac[35,84] + mem[35,84] = 0
 accuracy_33_51: - 6 frac[33,51] + mem[33,51] = 0
 accuracy_6_24: - 5.09765625 frac[6,24] + mem[6,24] = 0
 accuracy_36_88: - 6 frac[36,88] + mem[36,88] = 0
 accuracy_34_56: - 6 frac[34,56] + mem[34,56] = 0
 accuracy_32_40: - 6 frac[32,40] + mem[32,40] = 0
 accuracy_5_61: - 5.09765625 frac[5,61] + mem[5,61] = 0
 accuracy_18_23: - 5.09765625 frac[18,23] + mem[18,23] = 0
 accuracy_38_82: - 6 frac[38,82] + mem[38,82] = 0
 accuracy_36_38: - 6 frac[36,38] + mem[36,38] = 0
 accuracy_34_18: - 6 frac[34,18] + mem[34,18] = 0
 accuracy_38_52: - 6 frac[38,52] + mem[38,52] = 0
 accuracy_21_107: - 5.09765625 frac[21,107] + mem[21,107] = 0
 accuracy_13_62: - 5.09765625 frac[13,62] + mem[13,62] = 0
 accuracy_37_1: - 6 frac[37,1] + mem[37,1] = 0
 accuracy_20_79: - 5.09765625 frac[20,79] + mem[20,79] = 0
 accuracy_33_97: - 6 frac[33,97] + mem[33,97] = 0
 accuracy_23_101: - 5.09765625 frac[23,101] + mem[23,101] = 0
 accuracy_30_96: - 5.09765625 frac[30,96] + mem[30,96] = 0
 accuracy_37_107: - 6 frac[37,107] + mem[37,107] = 0
 accuracy_35_119: - 6 frac[35,119] + mem[35,119] = 0
 accuracy_32_83: - 6 frac[32,83] + mem[32,83] = 0
 accuracy_7_102: - 5.09765625 frac[7,102] + mem[7,102] = 0
 accuracy_14_39: - 5.09765625 frac[14,39] + mem[14,39] = 0
 accuracy_20_62: - 5.09765625 frac[20,62] + mem[20,62] = 0
 accuracy_3_30: - 5.09765625 frac[3,30] + mem[3,30] = 0
 accuracy_12_115: - 5.09765625 frac[12,115] + mem[12,115] = 0
 accuracy_10_79: - 5.09765625 frac[10,79] + mem[10,79] = 0
 accuracy_4_25: - 5.09765625 frac[4,25] + mem[4,25] = 0
 accuracy_35_93: - 6 frac[35,93] + mem[35,93] = 0
 accuracy_32_73: - 6 frac[32,73] + mem[32,73] = 0
 accuracy_33_52: - 6 frac[33,52] + mem[33,52] = 0
 accuracy_37_95: - 6 frac[37,95] + mem[37,95] = 0
 accuracy_8_34: - 5.09765625 frac[8,34] + mem[8,34] = 0
 accuracy_22_22: - 5.09765625 frac[22,22] + mem[22,22] = 0
 accuracy_11_95: - 5.09765625 frac[11,95] + mem[11,95] = 0
 accuracy_18_14: - 5.09765625 frac[18,14] + mem[18,14] = 0
 accuracy_33_46: - 6 frac[33,46] + mem[33,46] = 0
 accuracy_38_85: - 6 frac[38,85] + mem[38,85] = 0
 accuracy_36_45: - 6 frac[36,45] + mem[36,45] = 0
 accuracy_32_5: - 6 frac[32,5] + mem[32,5] = 0
 accuracy_37_32: - 6 frac[37,32] + mem[37,32] = 0
 accuracy_35_48: - 6 frac[35,48] + mem[35,48] = 0
 accuracy_33_0: - 6 frac[33,0] + mem[33,0] = 0
 accuracy_38_47: - 6 frac[38,47] + mem[38,47] = 0
 accuracy_36_11: - 6 frac[36,11] + mem[36,11] = 0
 accuracy_34_7: - 6 frac[34,7] + mem[34,7] = 0
 accuracy_21_116: - 5.09765625 frac[21,116] + mem[21,116] = 0
 accuracy_37_10: - 6 frac[37,10] + mem[37,10] = 0
 accuracy_0_20: - 5.09765625 frac[0,20] + mem[0,20] = 0
 accuracy_32_116: - 6 frac[32,116] + mem[32,116] = 0
 accuracy_38_1: - 6 frac[38,1] + mem[38,1] = 0
 accuracy_36_17: - 6 frac[36,17] + mem[36,17] = 0
 accuracy_21_94: - 5.09765625 frac[21,94] + mem[21,94] = 0
 accuracy_30_107: - 5.09765625 frac[30,107] + mem[30,107] = 0
 accuracy_34_118: - 6 frac[34,118] + mem[34,118] = 0
 accuracy_0_10: - 5.09765625 frac[0,10] + mem[0,10] = 0
 accuracy_32_106: - 6 frac[32,106] + mem[32,106] = 0
 accuracy_14_46: - 5.09765625 frac[14,46] + mem[14,46] = 0
 accuracy_38_27: - 6 frac[38,27] + mem[38,27] = 0
 accuracy_33_92: - 6 frac[33,92] + mem[33,92] = 0
 accuracy_7_95: - 5.09765625 frac[7,95] + mem[7,95] = 0
 accuracy_36_96: - 6 frac[36,96] + mem[36,96] = 0
 accuracy_3_45: - 5.09765625 frac[3,45] + mem[3,45] = 0
 accuracy_16_3: - 5.09765625 frac[16,3] + mem[16,3] = 0
 accuracy_38_106: - 6 frac[38,106] + mem[38,106] = 0
 accuracy_37_88: - 6 frac[37,88] + mem[37,88] = 0
 accuracy_34_74: - 6 frac[34,74] + mem[34,74] = 0
 accuracy_32_38: - 6 frac[32,38] + mem[32,38] = 0
 accuracy_29_34: - 5.09765625 frac[29,34] + mem[29,34] = 0
 accuracy_33_39: - 6 frac[33,39] + mem[33,39] = 0
 accuracy_36_84: - 6 frac[36,84] + mem[36,84] = 0
 accuracy_22_3: - 5.09765625 frac[22,3] + mem[22,3] = 0
 accuracy_35_57: - 6 frac[35,57] + mem[35,57] = 0
 accuracy_38_38: - 6 frac[38,38] + mem[38,38] = 0
 accuracy_36_50: - 6 frac[36,50] + mem[36,50] = 0
 accuracy_31_90: - 5.09765625 frac[31,90] + mem[31,90] = 0
 accuracy_37_51: - 6 frac[37,51] + mem[37,51] = 0
 accuracy_35_31: - 6 frac[35,31] + mem[35,31] = 0
 accuracy_33_115: - 6 frac[33,115] + mem[33,115] = 0
 accuracy_38_56: - 6 frac[38,56] + mem[38,56] = 0
 accuracy_36_24: - 6 frac[36,24] + mem[36,24] = 0
 accuracy_37_29: - 6 frac[37,29] + mem[37,29] = 0
 accuracy_34_113: - 6 frac[34,113] + mem[34,113] = 0
 accuracy_35_101: - 6 frac[35,101] + mem[35,101] = 0
 accuracy_0_1: - 5.09765625 frac[0,1] + mem[0,1] = 0
 accuracy_33_85: - 6 frac[33,85] + mem[33,85] = 0
 accuracy_14_49: - 5.09765625 frac[14,49] + mem[14,49] = 0
 accuracy_24_62: - 5.09765625 frac[24,62] + mem[24,62] = 0
 accuracy_32_97: - 6 frac[32,97] + mem[32,97] = 0
 accuracy_12_97: - 5.09765625 frac[12,97] + mem[12,97] = 0
 accuracy_30_116: - 5.09765625 frac[30,116] + mem[30,116] = 0
 accuracy_36_111: - 6 frac[36,111] + mem[36,111] = 0
 accuracy_19_49: - 5.09765625 frac[19,49] + mem[19,49] = 0
 accuracy_34_107: - 6 frac[34,107] + mem[34,107] = 0
 accuracy_35_75: - 6 frac[35,75] + mem[35,75] = 0
 accuracy_33_79: - 6 frac[33,79] + mem[33,79] = 0
 accuracy_1_38: - 5.09765625 frac[1,38] + mem[1,38] = 0
 accuracy_6_13: - 5.09765625 frac[6,13] + mem[6,13] = 0
 accuracy_36_117: - 6 frac[36,117] + mem[36,117] = 0
 accuracy_37_65: - 6 frac[37,65] + mem[37,65] = 0
 accuracy_26_38: - 5.09765625 frac[26,38] + mem[26,38] = 0
 accuracy_32_61: - 6 frac[32,61] + mem[32,61] = 0
 accuracy_18_0: - 5.09765625 frac[18,0] + mem[18,0] = 0
 accuracy_33_56: - 6 frac[33,56] + mem[33,56] = 0
 accuracy_36_83: - 6 frac[36,83] + mem[36,83] = 0
 accuracy_34_63: - 6 frac[34,63] + mem[34,63] = 0
 accuracy_2_95: - 5.09765625 frac[2,95] + mem[2,95] = 0
 accuracy_32_19: - 6 frac[32,19] + mem[32,19] = 0
 accuracy_35_62: - 6 frac[35,62] + mem[35,62] = 0
 accuracy_18_26: - 5.09765625 frac[18,26] + mem[18,26] = 0
 accuracy_36_57: - 6 frac[36,57] + mem[36,57] = 0
 accuracy_34_25: - 6 frac[34,25] + mem[34,25] = 0
 accuracy_37_60: - 6 frac[37,60] + mem[37,60] = 0
 accuracy_28_71: - 5.09765625 frac[28,71] + mem[28,71] = 0
 accuracy_35_4: - 6 frac[35,4] + mem[35,4] = 0
 accuracy_33_116: - 6 frac[33,116] + mem[33,116] = 0
 accuracy_5_67: - 5.09765625 frac[5,67] + mem[5,67] = 0
 accuracy_38_51: - 6 frac[38,51] + mem[38,51] = 0
 accuracy_36_7: - 6 frac[36,7] + mem[36,7] = 0
 accuracy_31_105: - 5.09765625 frac[31,105] + mem[31,105] = 0
 accuracy_37_6: - 6 frac[37,6] + mem[37,6] = 0
 accuracy_35_106: - 6 frac[35,106] + mem[35,106] = 0
 accuracy_0_24: - 5.09765625 frac[0,24] + mem[0,24] = 0
 accuracy_14_56: - 5.09765625 frac[14,56] + mem[14,56] = 0
 accuracy_37_96: - 6 frac[37,96] + mem[37,96] = 0
 accuracy_33_64: - 6 frac[33,64] + mem[33,64] = 0
 accuracy_14_34: - 5.09765625 frac[14,34] + mem[14,34] = 0
 accuracy_6_4: - 5.09765625 frac[6,4] + mem[6,4] = 0
 accuracy_38_100: - 6 frac[38,100] + mem[38,100] = 0
 accuracy_37_74: - 6 frac[37,74] + mem[37,74] = 0
 accuracy_34_92: - 6 frac[34,92] + mem[34,92] = 0
 accuracy_35_86: - 6 frac[35,86] + mem[35,86] = 0
 accuracy_32_52: - 6 frac[32,52] + mem[32,52] = 0
 accuracy_29_52: - 5.09765625 frac[29,52] + mem[29,52] = 0
 accuracy_3_33: - 5.09765625 frac[3,33] + mem[3,33] = 0
 accuracy_33_49: - 6 frac[33,49] + mem[33,49] = 0
 accuracy_16_23: - 5.09765625 frac[16,23] + mem[16,23] = 0
 accuracy_36_90: - 6 frac[36,90] + mem[36,90] = 0
 accuracy_37_84: - 6 frac[37,84] + mem[37,84] = 0
 accuracy_34_54: - 6 frac[34,54] + mem[34,54] = 0
 accuracy_32_42: - 6 frac[32,42] + mem[32,42] = 0
 accuracy_29_46: - 5.09765625 frac[29,46] + mem[29,46] = 0
 accuracy_33_43: - 6 frac[33,43] + mem[33,43] = 0
 accuracy_24_64: - 5.09765625 frac[24,64] + mem[24,64] = 0
 accuracy_36_32: - 6 frac[36,32] + mem[36,32] = 0
 accuracy_10_63: - 5.09765625 frac[10,63] + mem[10,63] = 0
 accuracy_17_40: - 5.09765625 frac[17,40] + mem[17,40] = 0
 accuracy_32_0: - 6 frac[32,0] + mem[32,0] = 0
 accuracy_37_37: - 6 frac[37,37] + mem[37,37] = 0
 accuracy_5_76: - 5.09765625 frac[5,76] + mem[5,76] = 0
 accuracy_34_10: - 6 frac[34,10] + mem[34,10] = 0
 accuracy_2_106: - 5.09765625 frac[2,106] + mem[2,106] = 0
 accuracy_30_92: - 5.09765625 frac[30,92] + mem[30,92] = 0
 accuracy_20_73: - 5.09765625 frac[20,73] + mem[20,73] = 0
 accuracy_25_9: - 5.09765625 frac[25,9] + mem[25,9] = 0
 accuracy_37_105: - 6 frac[37,105] + mem[37,105] = 0
 accuracy_11_101: - 5.09765625 frac[11,101] + mem[11,101] = 0
 accuracy_1_48: - 5.09765625 frac[1,48] + mem[1,48] = 0
 accuracy_34_87: - 6 frac[34,87] + mem[34,87] = 0
 accuracy_35_95: - 6 frac[35,95] + mem[35,95] = 0
 accuracy_3_38: - 5.09765625 frac[3,38] + mem[3,38] = 0
 accuracy_16_14: - 5.09765625 frac[16,14] + mem[16,14] = 0
 accuracy_6_17: - 5.09765625 frac[6,17] + mem[6,17] = 0
 accuracy_38_113: - 6 frac[38,113] + mem[38,113] = 0
 accuracy_36_65: - 6 frac[36,65] + mem[36,65] = 0
 accuracy_37_93: - 6 frac[37,93] + mem[37,93] = 0
 accuracy_34_49: - 6 frac[34,49] + mem[34,49] = 0
 accuracy_8_36: - 5.09765625 frac[8,36] + mem[8,36] = 0
 accuracy_32_33: - 6 frac[32,33] + mem[32,33] = 0
 accuracy_22_20: - 5.09765625 frac[22,20] + mem[22,20] = 0
 accuracy_13_65: - 5.09765625 frac[13,65] + mem[13,65] = 0
 accuracy_33_44: - 6 frac[33,44] + mem[33,44] = 0
 accuracy_24_71: - 5.09765625 frac[24,71] + mem[24,71] = 0
 accuracy_38_75: - 6 frac[38,75] + mem[38,75] = 0
 accuracy_36_47: - 6 frac[36,47] + mem[36,47] = 0
 accuracy_34_43: - 6 frac[34,43] + mem[34,43] = 0
 accuracy_32_7: - 6 frac[32,7] + mem[32,7] = 0
 accuracy_22_14: - 5.09765625 frac[22,14] + mem[22,14] = 0
 accuracy_37_46: - 6 frac[37,46] + mem[37,46] = 0
 accuracy_35_50: - 6 frac[35,50] + mem[35,50] = 0
 accuracy_33_6: - 6 frac[33,6] + mem[33,6] = 0
 accuracy_38_45: - 6 frac[38,45] + mem[38,45] = 0
 accuracy_36_53: - 6 frac[36,53] + mem[36,53] = 0
 accuracy_34_5: - 6 frac[34,5] + mem[34,5] = 0
 accuracy_35_24: - 6 frac[35,24] + mem[35,24] = 0
 accuracy_32_118: - 6 frac[32,118] + mem[32,118] = 0
 accuracy_38_7: - 6 frac[38,7] + mem[38,7] = 0
 accuracy_36_19: - 6 frac[36,19] + mem[36,19] = 0
 accuracy_21_92: - 5.09765625 frac[21,92] + mem[21,92] = 0
 accuracy_31_101: - 5.09765625 frac[31,101] + mem[31,101] = 0
 accuracy_34_116: - 6 frac[34,116] + mem[34,116] = 0
 accuracy_32_108: - 6 frac[32,108] + mem[32,108] = 0
 accuracy_33_82: - 6 frac[33,82] + mem[33,82] = 0
 accuracy_14_44: - 5.09765625 frac[14,44] + mem[14,44] = 0
 accuracy_23_90: - 5.09765625 frac[23,90] + mem[23,90] = 0
 accuracy_30_115: - 5.09765625 frac[30,115] + mem[30,115] = 0
 accuracy_36_98: - 6 frac[36,98] + mem[36,98] = 0
 accuracy_35_68: - 6 frac[35,68] + mem[35,68] = 0
 accuracy_9_64: - 5.09765625 frac[9,64] + mem[9,64] = 0
 accuracy_38_104: - 6 frac[38,104] + mem[38,104] = 0
 accuracy_37_70: - 6 frac[37,70] + mem[37,70] = 0
 accuracy_28_61: - 5.09765625 frac[28,61] + mem[28,61] = 0
 accuracy_8_43: - 5.09765625 frac[8,43] + mem[8,43] = 0
 accuracy_32_56: - 6 frac[32,56] + mem[32,56] = 0
 accuracy_34_34: - 6 frac[34,34] + mem[34,34] = 0
 accuracy_32_30: - 6 frac[32,30] + mem[32,30] = 0
 accuracy_22_1: - 5.09765625 frac[22,1] + mem[22,1] = 0
 accuracy_13_84: - 5.09765625 frac[13,84] + mem[13,84] = 0
 accuracy_33_31: - 6 frac[33,31] + mem[33,31] = 0
 accuracy_38_36: - 6 frac[38,36] + mem[38,36] = 0
 accuracy_36_60: - 6 frac[36,60] + mem[36,60] = 0
 accuracy_37_49: - 6 frac[37,49] + mem[37,49] = 0
 accuracy_28_74: - 5.09765625 frac[28,74] + mem[28,74] = 0
 accuracy_35_1: - 6 frac[35,1] + mem[35,1] = 0
 accuracy_5_88: - 5.09765625 frac[5,88] + mem[5,88] = 0
 accuracy_38_62: - 6 frac[38,62] + mem[38,62] = 0
 accuracy_36_26: - 6 frac[36,26] + mem[36,26] = 0
 accuracy_21_101: - 5.09765625 frac[21,101] + mem[21,101] = 0
 accuracy_31_98: - 5.09765625 frac[31,98] + mem[31,98] = 0
 accuracy_37_27: - 6 frac[37,27] + mem[37,27] = 0
 accuracy_32_99: - 6 frac[32,99] + mem[32,99] = 0
 accuracy_33_107: - 6 frac[33,107] + mem[33,107] = 0
 accuracy_36_105: - 6 frac[36,105] + mem[36,105] = 0
 accuracy_37_101: - 6 frac[37,101] + mem[37,101] = 0
 accuracy_35_77: - 6 frac[35,77] + mem[35,77] = 0
 accuracy_7_108: - 5.09765625 frac[7,108] + mem[7,108] = 0
 accuracy_33_77: - 6 frac[33,77] + mem[33,77] = 0
 accuracy_29_31: - 5.09765625 frac[29,31] + mem[29,31] = 0
 accuracy_11_105: - 5.09765625 frac[11,105] + mem[11,105] = 0
 accuracy_6_3: - 5.09765625 frac[6,3] + mem[6,3] = 0
 accuracy_38_99: - 6 frac[38,99] + mem[38,99] = 0
 accuracy_36_119: - 6 frac[36,119] + mem[36,119] = 0
 accuracy_37_79: - 6 frac[37,79] + mem[37,79] = 0
 accuracy_35_83: - 6 frac[35,83] + mem[35,83] = 0
 accuracy_32_63: - 6 frac[32,63] + mem[32,63] = 0
 accuracy_29_57: - 5.09765625 frac[29,57] + mem[29,57] = 0
 accuracy_33_62: - 6 frac[33,62] + mem[33,62] = 0
 accuracy_16_18: - 5.09765625 frac[16,18] + mem[16,18] = 0
 accuracy_24_85: - 5.09765625 frac[24,85] + mem[24,85] = 0
 accuracy_36_93: - 6 frac[36,93] + mem[36,93] = 0
 accuracy_8_56: - 5.09765625 frac[8,56] + mem[8,56] = 0
 accuracy_22_24: - 5.09765625 frac[22,24] + mem[22,24] = 0
 accuracy_13_77: - 5.09765625 frac[13,77] + mem[13,77] = 0
 accuracy_35_32: - 6 frac[35,32] + mem[35,32] = 0
 accuracy_27_79: - 5.09765625 frac[27,79] + mem[27,79] = 0
 accuracy_38_95: - 6 frac[38,95] + mem[38,95] = 0
 accuracy_34_23: - 6 frac[34,23] + mem[34,23] = 0
 accuracy_2_119: - 5.09765625 frac[2,119] + mem[2,119] = 0
 accuracy_32_11: - 6 frac[32,11] + mem[32,11] = 0
 accuracy_37_58: - 6 frac[37,58] + mem[37,58] = 0
 accuracy_28_65: - 5.09765625 frac[28,65] + mem[28,65] = 0
 accuracy_33_10: - 6 frac[33,10] + mem[33,10] = 0
 accuracy_5_65: - 5.09765625 frac[5,65] + mem[5,65] = 0
 accuracy_38_49: - 6 frac[38,49] + mem[38,49] = 0
 accuracy_36_1: - 6 frac[36,1] + mem[36,1] = 0
 accuracy_23_114: - 5.09765625 frac[23,114] + mem[23,114] = 0
 accuracy_37_4: - 6 frac[37,4] + mem[37,4] = 0
 accuracy_38_11: - 6 frac[38,11] + mem[38,11] = 0
 accuracy_31_113: - 5.09765625 frac[31,113] + mem[31,113] = 0
 accuracy_19_56: - 5.09765625 frac[19,56] + mem[19,56] = 0
 accuracy_35_114: - 6 frac[35,114] + mem[35,114] = 0
 accuracy_33_70: - 6 frac[33,70] + mem[33,70] = 0
 accuracy_14_32: - 5.09765625 frac[14,32] + mem[14,32] = 0
 accuracy_10_64: - 5.09765625 frac[10,64] + mem[10,64] = 0
 accuracy_34_90: - 6 frac[34,90] + mem[34,90] = 0
 mem_tot_CPU1: - mem[0,5] - mem[0,0] - mem[0,7] - mem[0,11] - mem[0,25]
   - mem[0,23] - mem[0,13] - mem[0,4] - mem[0,27] - mem[0,20] - mem[0,10]
   - mem[0,1] - mem[0,24] + mem_tot_CPU1 = 0
 R1159: - 3.05176e-05 mem_tot_CPU1 + normalized_mem_tot_CPU1 = 0
 rows_tot_CPU1: - 3 frac[0,5] - 3 frac[0,0] - 3 frac[0,7] - 3 frac[0,11]
   - 3 frac[0,25] - 3 frac[0,23] - 3 frac[0,13] - 3 frac[0,4]
   - 3 frac[0,27] - 3 frac[0,20] - 3 frac[0,10] - 3 frac[0,1]
   - 3 frac[0,24] + rows_tot_CPU1 = 0
 R1161: - 0.0833333333333333 rows_tot_CPU1 + normalized_rows_tot_CPU1 = 0
 mem_tot_CPU2: - mem[1,40] - mem[1,58] - mem[1,52] - mem[1,56] - mem[1,31]
   - mem[1,49] - mem[1,43] - mem[1,44] - mem[1,32] - mem[1,57] - mem[1,38]
   - mem[1,48] + mem_tot_CPU2 = 0
 R1163: - 3.05176e-05 mem_tot_CPU2 + normalized_mem_tot_CPU2 = 0
 rows_tot_CPU2: - 3 frac[1,40] - 3 frac[1,58] - 3 frac[1,52] - 3 frac[1,56]
   - 3 frac[1,31] - 3 frac[1,49] - 3 frac[1,43] - 3 frac[1,44]
   - 3 frac[1,32] - 3 frac[1,57] - 3 frac[1,38] - 3 frac[1,48]
   + rows_tot_CPU2 = 0
 R1165: - 0.0833333333333333 rows_tot_CPU2 + normalized_rows_tot_CPU2 = 0
 mem_tot_CPU3: - mem[2,104] - mem[2,99] - mem[2,117] - mem[2,97]
   - mem[2,115] - mem[2,100] - mem[2,118] - mem[2,98] - mem[2,96]
   - mem[2,95] - mem[2,106] - mem[2,119] + mem_tot_CPU3 = 0
 R1167: - 3.05176e-05 mem_tot_CPU3 + normalized_mem_tot_CPU3 = 0
 rows_tot_CPU3: - 3 frac[2,104] - 3 frac[2,99] - 3 frac[2,117]
   - 3 frac[2,97] - 3 frac[2,115] - 3 frac[2,100] - 3 frac[2,118]
   - 3 frac[2,98] - 3 frac[2,96] - 3 frac[2,95] - 3 frac[2,106]
   - 3 frac[2,119] + rows_tot_CPU3 = 0
 R1169: - 0.0833333333333333 rows_tot_CPU3 + normalized_rows_tot_CPU3 = 0
 mem_tot_CPU4: - mem[3,35] - mem[3,40] - mem[3,37] - mem[3,42] - mem[3,39]
   - mem[3,44] - mem[3,32] - mem[3,46] - mem[3,52] - mem[3,54] - mem[3,30]
   - mem[3,45] - mem[3,33] - mem[3,38] + mem_tot_CPU4 = 0
 R1171: - 3.05176e-05 mem_tot_CPU4 + normalized_mem_tot_CPU4 = 0
 rows_tot_CPU4: - 3 frac[3,35] - 3 frac[3,40] - 3 frac[3,37] - 3 frac[3,42]
   - 3 frac[3,39] - 3 frac[3,44] - 3 frac[3,32] - 3 frac[3,46]
   - 3 frac[3,52] - 3 frac[3,54] - 3 frac[3,30] - 3 frac[3,45]
   - 3 frac[3,33] - 3 frac[3,38] + rows_tot_CPU4 = 0
 R1173: - 0.0833333333333333 rows_tot_CPU4 + normalized_rows_tot_CPU4 = 0
 mem_tot_CPU5: - mem[4,11] - mem[4,20] - mem[4,19] - mem[4,26] - mem[4,22]
   - mem[4,3] - mem[4,6] - mem[4,25] + mem_tot_CPU5 = 0
 R1175: - 3.05176e-05 mem_tot_CPU5 + normalized_mem_tot_CPU5 = 0
 rows_tot_CPU5: - 3 frac[4,11] - 3 frac[4,20] - 3 frac[4,19] - 3 frac[4,26]
   - 3 frac[4,22] - 3 frac[4,3] - 3 frac[4,6] - 3 frac[4,25]
   + rows_tot_CPU5 = 0
 R1177: - 0.0833333333333333 rows_tot_CPU5 + normalized_rows_tot_CPU5 = 0
 mem_tot_CPU6: - mem[5,84] - mem[5,62] - mem[5,82] - mem[5,63] - mem[5,73]
   - mem[5,83] - mem[5,61] - mem[5,67] - mem[5,76] - mem[5,88] - mem[5,65]
   + mem_tot_CPU6 = 0
 R1179: - 3.05176e-05 mem_tot_CPU6 + normalized_mem_tot_CPU6 = 0
 rows_tot_CPU6: - 3 frac[5,84] - 3 frac[5,62] - 3 frac[5,82] - 3 frac[5,63]
   - 3 frac[5,73] - 3 frac[5,83] - 3 frac[5,61] - 3 frac[5,67]
   - 3 frac[5,76] - 3 frac[5,88] - 3 frac[5,65] + rows_tot_CPU6 = 0
 R1181: - 0.0833333333333333 rows_tot_CPU6 + normalized_rows_tot_CPU6 = 0
 mem_tot_CPU7: - mem[6,1] - mem[6,7] - mem[6,0] - mem[6,26] - mem[6,29]
   - mem[6,24] - mem[6,13] - mem[6,4] - mem[6,17] - mem[6,3] + mem_tot_CPU7
   = 0
 R1183: - 3.05176e-05 mem_tot_CPU7 + normalized_mem_tot_CPU7 = 0
 rows_tot_CPU7: - 3 frac[6,1] - 3 frac[6,7] - 3 frac[6,0] - 3 frac[6,26]
   - 3 frac[6,29] - 3 frac[6,24] - 3 frac[6,13] - 3 frac[6,4]
   - 3 frac[6,17] - 3 frac[6,3] + rows_tot_CPU7 = 0
 R1185: - 0.0833333333333333 rows_tot_CPU7 + normalized_rows_tot_CPU7 = 0
 mem_tot_CPU8: - mem[7,98] - mem[7,91] - mem[7,113] - mem[7,104]
   - mem[7,97] - mem[7,116] - mem[7,118] - mem[7,100] - mem[7,90]
   - mem[7,112] - mem[7,102] - mem[7,95] - mem[7,108] + mem_tot_CPU8 = 0
 R1187: - 3.05176e-05 mem_tot_CPU8 + normalized_mem_tot_CPU8 = 0
 rows_tot_CPU8: - 3 frac[7,98] - 3 frac[7,91] - 3 frac[7,113]
   - 3 frac[7,104] - 3 frac[7,97] - 3 frac[7,116] - 3 frac[7,118]
   - 3 frac[7,100] - 3 frac[7,90] - 3 frac[7,112] - 3 frac[7,102]
   - 3 frac[7,95] - 3 frac[7,108] + rows_tot_CPU8 = 0
 R1189: - 0.0833333333333333 rows_tot_CPU8 + normalized_rows_tot_CPU8 = 0
 mem_tot_CPU9: - mem[8,38] - mem[8,51] - mem[8,33] - mem[8,53] - mem[8,35]
   - mem[8,42] - mem[8,44] - mem[8,50] - mem[8,57] - mem[8,46] - mem[8,34]
   - mem[8,36] - mem[8,43] - mem[8,56] + mem_tot_CPU9 = 0
 R1191: - 3.05176e-05 mem_tot_CPU9 + normalized_mem_tot_CPU9 = 0
 rows_tot_CPU9: - 3 frac[8,38] - 3 frac[8,51] - 3 frac[8,33] - 3 frac[8,53]
   - 3 frac[8,35] - 3 frac[8,42] - 3 frac[8,44] - 3 frac[8,50]
   - 3 frac[8,57] - 3 frac[8,46] - 3 frac[8,34] - 3 frac[8,36]
   - 3 frac[8,43] - 3 frac[8,56] + rows_tot_CPU9 = 0
 R1193: - 0.0833333333333333 rows_tot_CPU9 + normalized_rows_tot_CPU9 = 0
 mem_tot_CPU10: - mem[9,62] - mem[9,70] - mem[9,81] - mem[9,60] - mem[9,68]
   - mem[9,65] - mem[9,82] - mem[9,61] - mem[9,64] + mem_tot_CPU10 = 0
 R1195: - 3.05176e-05 mem_tot_CPU10 + normalized_mem_tot_CPU10 = 0
 rows_tot_CPU10: - 3 frac[9,62] - 3 frac[9,70] - 3 frac[9,81]
   - 3 frac[9,60] - 3 frac[9,68] - 3 frac[9,65] - 3 frac[9,82]
   - 3 frac[9,61] - 3 frac[9,64] + rows_tot_CPU10 = 0
 R1197: - 0.0833333333333333 rows_tot_CPU10 + normalized_rows_tot_CPU10
   = 0
 mem_tot_CPU11: - mem[10,75] - mem[10,85] - mem[10,78] - mem[10,73]
   - mem[10,62] - mem[10,81] - mem[10,67] - mem[10,74] - mem[10,84]
   - mem[10,65] - mem[10,79] - mem[10,63] - mem[10,64] + mem_tot_CPU11 = 0
 R1199: - 3.05176e-05 mem_tot_CPU11 + normalized_mem_tot_CPU11 = 0
 rows_tot_CPU11: - 3 frac[10,75] - 3 frac[10,85] - 3 frac[10,78]
   - 3 frac[10,73] - 3 frac[10,62] - 3 frac[10,81] - 3 frac[10,67]
   - 3 frac[10,74] - 3 frac[10,84] - 3 frac[10,65] - 3 frac[10,79]
   - 3 frac[10,63] - 3 frac[10,64] + rows_tot_CPU11 = 0
 R1201: - 0.0833333333333333 rows_tot_CPU11 + normalized_rows_tot_CPU11
   = 0
 mem_tot_CPU12: - mem[11,114] - mem[11,107] - mem[11,116] - mem[11,94]
   - mem[11,118] - mem[11,111] - mem[11,100] - mem[11,104] - mem[11,117]
   - mem[11,95] - mem[11,101] - mem[11,105] + mem_tot_CPU12 = 0
 R1203: - 3.05176e-05 mem_tot_CPU12 + normalized_mem_tot_CPU12 = 0
 rows_tot_CPU12: - 3 frac[11,114] - 3 frac[11,107] - 3 frac[11,116]
   - 3 frac[11,94] - 3 frac[11,118] - 3 frac[11,111] - 3 frac[11,100]
   - 3 frac[11,104] - 3 frac[11,117] - 3 frac[11,95] - 3 frac[11,101]
   - 3 frac[11,105] + rows_tot_CPU12 = 0
 R1205: - 0.0833333333333333 rows_tot_CPU12 + normalized_rows_tot_CPU12
   = 0
 mem_tot_CPU13: - mem[12,93] - mem[12,98] - mem[12,101] - mem[12,92]
   - mem[12,116] - mem[12,115] - mem[12,97] + mem_tot_CPU13 = 0
 R1207: - 3.05176e-05 mem_tot_CPU13 + normalized_mem_tot_CPU13 = 0
 rows_tot_CPU13: - 3 frac[12,93] - 3 frac[12,98] - 3 frac[12,101]
   - 3 frac[12,92] - 3 frac[12,116] - 3 frac[12,115] - 3 frac[12,97]
   + rows_tot_CPU13 = 0
 R1209: - 0.0833333333333333 rows_tot_CPU13 + normalized_rows_tot_CPU13
   = 0
 mem_tot_CPU14: - mem[13,70] - mem[13,75] - mem[13,68] - mem[13,87]
   - mem[13,73] - mem[13,78] - mem[13,64] - mem[13,83] - mem[13,76]
   - mem[13,62] - mem[13,65] - mem[13,84] - mem[13,77] + mem_tot_CPU14 = 0
 R1211: - 3.05176e-05 mem_tot_CPU14 + normalized_mem_tot_CPU14 = 0
 rows_tot_CPU14: - 3 frac[13,70] - 3 frac[13,75] - 3 frac[13,68]
   - 3 frac[13,87] - 3 frac[13,73] - 3 frac[13,78] - 3 frac[13,64]
   - 3 frac[13,83] - 3 frac[13,76] - 3 frac[13,62] - 3 frac[13,65]
   - 3 frac[13,84] - 3 frac[13,77] + rows_tot_CPU14 = 0
 R1213: - 0.0833333333333333 rows_tot_CPU14 + normalized_rows_tot_CPU14
   = 0
 mem_tot_CPU15: - mem[14,31] - mem[14,38] - mem[14,36] - mem[14,47]
   - mem[14,54] - mem[14,57] - mem[14,52] - mem[14,40] - mem[14,39]
   - mem[14,46] - mem[14,49] - mem[14,56] - mem[14,34] - mem[14,44]
   - mem[14,32] + mem_tot_CPU15 = 0
 R1215: - 3.05176e-05 mem_tot_CPU15 + normalized_mem_tot_CPU15 = 0
 rows_tot_CPU15: - 3 frac[14,31] - 3 frac[14,38] - 3 frac[14,36]
   - 3 frac[14,47] - 3 frac[14,54] - 3 frac[14,57] - 3 frac[14,52]
   - 3 frac[14,40] - 3 frac[14,39] - 3 frac[14,46] - 3 frac[14,49]
   - 3 frac[14,56] - 3 frac[14,34] - 3 frac[14,44] - 3 frac[14,32]
   + rows_tot_CPU15 = 0
 R1217: - 0.0833333333333333 rows_tot_CPU15 + normalized_rows_tot_CPU15
   = 0
 mem_tot_CPU16: - mem[15,10] - mem[15,24] - mem[15,6] - mem[15,0]
   - mem[15,22] + mem_tot_CPU16 = 0
 R1219: - 3.05176e-05 mem_tot_CPU16 + normalized_mem_tot_CPU16 = 0
 rows_tot_CPU16: - 3 frac[15,10] - 3 frac[15,24] - 3 frac[15,6]
   - 3 frac[15,0] - 3 frac[15,22] + rows_tot_CPU16 = 0
 R1221: - 0.0833333333333333 rows_tot_CPU16 + normalized_rows_tot_CPU16
   = 0
 mem_tot_CPU17: - mem[16,7] - mem[16,29] - mem[16,2] - mem[16,24]
   - mem[16,22] - mem[16,13] - mem[16,4] - mem[16,26] - mem[16,6]
   - mem[16,10] - mem[16,3] - mem[16,23] - mem[16,14] - mem[16,18]
   + mem_tot_CPU17 = 0
 R1223: - 3.05176e-05 mem_tot_CPU17 + normalized_mem_tot_CPU17 = 0
 rows_tot_CPU17: - 3 frac[16,7] - 3 frac[16,29] - 3 frac[16,2]
   - 3 frac[16,24] - 3 frac[16,22] - 3 frac[16,13] - 3 frac[16,4]
   - 3 frac[16,26] - 3 frac[16,6] - 3 frac[16,10] - 3 frac[16,3]
   - 3 frac[16,23] - 3 frac[16,14] - 3 frac[16,18] + rows_tot_CPU17 = 0
 R1225: - 0.0833333333333333 rows_tot_CPU17 + normalized_rows_tot_CPU17
   = 0
 mem_tot_CPU18: - mem[17,46] - mem[17,57] - mem[17,50] - mem[17,43]
   - mem[17,44] - mem[17,37] - mem[17,54] - mem[17,47] - mem[17,32]
   - mem[17,38] - mem[17,40] + mem_tot_CPU18 = 0
 R1227: - 3.05176e-05 mem_tot_CPU18 + normalized_mem_tot_CPU18 = 0
 rows_tot_CPU18: - 3 frac[17,46] - 3 frac[17,57] - 3 frac[17,50]
   - 3 frac[17,43] - 3 frac[17,44] - 3 frac[17,37] - 3 frac[17,54]
   - 3 frac[17,47] - 3 frac[17,32] - 3 frac[17,38] - 3 frac[17,40]
   + rows_tot_CPU18 = 0
 R1229: - 0.0833333333333333 rows_tot_CPU18 + normalized_rows_tot_CPU18
   = 0
 mem_tot_CPU19: - mem[18,10] - mem[18,22] - mem[18,17] - mem[18,29]
   - mem[18,20] - mem[18,15] - mem[18,4] - mem[18,25] - mem[18,23]
   - mem[18,14] - mem[18,0] - mem[18,26] + mem_tot_CPU19 = 0
 R1231: - 3.05176e-05 mem_tot_CPU19 + normalized_mem_tot_CPU19 = 0
 rows_tot_CPU19: - 3 frac[18,10] - 3 frac[18,22] - 3 frac[18,17]
   - 3 frac[18,29] - 3 frac[18,20] - 3 frac[18,15] - 3 frac[18,4]
   - 3 frac[18,25] - 3 frac[18,23] - 3 frac[18,14] - 3 frac[18,0]
   - 3 frac[18,26] + rows_tot_CPU19 = 0
 R1233: - 0.0833333333333333 rows_tot_CPU19 + normalized_rows_tot_CPU19
   = 0
 mem_tot_CPU20: - mem[19,53] - mem[19,58] - mem[19,32] - mem[19,46]
   - mem[19,55] - mem[19,34] - mem[19,38] - mem[19,52] - mem[19,49]
   - mem[19,56] + mem_tot_CPU20 = 0
 R1235: - 3.05176e-05 mem_tot_CPU20 + normalized_mem_tot_CPU20 = 0
 rows_tot_CPU20: - 3 frac[19,53] - 3 frac[19,58] - 3 frac[19,32]
   - 3 frac[19,46] - 3 frac[19,55] - 3 frac[19,34] - 3 frac[19,38]
   - 3 frac[19,52] - 3 frac[19,49] - 3 frac[19,56] + rows_tot_CPU20 = 0
 R1237: - 0.0833333333333333 rows_tot_CPU20 + normalized_rows_tot_CPU20
   = 0
 mem_tot_CPU21: - mem[20,75] - mem[20,82] - mem[20,85] - mem[20,65]
   - mem[20,63] - mem[20,87] - mem[20,74] - mem[20,84] - mem[20,83]
   - mem[20,79] - mem[20,62] - mem[20,73] + mem_tot_CPU21 = 0
 R1239: - 3.05176e-05 mem_tot_CPU21 + normalized_mem_tot_CPU21 = 0
 rows_tot_CPU21: - 3 frac[20,75] - 3 frac[20,82] - 3 frac[20,85]
   - 3 frac[20,65] - 3 frac[20,63] - 3 frac[20,87] - 3 frac[20,74]
   - 3 frac[20,84] - 3 frac[20,83] - 3 frac[20,79] - 3 frac[20,62]
   - 3 frac[20,73] + rows_tot_CPU21 = 0
 R1241: - 0.0833333333333333 rows_tot_CPU21 + normalized_rows_tot_CPU21
   = 0
 mem_tot_CPU22: - mem[21,119] - mem[21,112] - mem[21,90] - mem[21,99]
   - mem[21,97] - mem[21,102] - mem[21,104] - mem[21,91] - mem[21,100]
   - mem[21,107] - mem[21,116] - mem[21,94] - mem[21,92] - mem[21,101]
   + mem_tot_CPU22 = 0
 R1243: - 3.05176e-05 mem_tot_CPU22 + normalized_mem_tot_CPU22 = 0
 rows_tot_CPU22: - 3 frac[21,119] - 3 frac[21,112] - 3 frac[21,90]
   - 3 frac[21,99] - 3 frac[21,97] - 3 frac[21,102] - 3 frac[21,104]
   - 3 frac[21,91] - 3 frac[21,100] - 3 frac[21,107] - 3 frac[21,116]
   - 3 frac[21,94] - 3 frac[21,92] - 3 frac[21,101] + rows_tot_CPU22 = 0
 R1245: - 0.0833333333333333 rows_tot_CPU22 + normalized_rows_tot_CPU22
   = 0
 mem_tot_CPU23: - mem[22,19] - mem[22,7] - mem[22,11] - mem[22,2]
   - mem[22,5] - mem[22,23] - mem[22,9] - mem[22,0] - mem[22,27]
   - mem[22,18] - mem[22,15] - mem[22,25] - mem[22,10] - mem[22,13]
   - mem[22,4] - mem[22,22] - mem[22,3] - mem[22,20] - mem[22,14]
   - mem[22,1] - mem[22,24] + mem_tot_CPU23 = 0
 R1247: - 3.05176e-05 mem_tot_CPU23 + normalized_mem_tot_CPU23 = 0
 rows_tot_CPU23: - 3 frac[22,19] - 3 frac[22,7] - 3 frac[22,11]
   - 3 frac[22,2] - 3 frac[22,5] - 3 frac[22,23] - 3 frac[22,9]
   - 3 frac[22,0] - 3 frac[22,27] - 3 frac[22,18] - 3 frac[22,15]
   - 3 frac[22,25] - 3 frac[22,10] - 3 frac[22,13] - 3 frac[22,4]
   - 3 frac[22,22] - 3 frac[22,3] - 3 frac[22,20] - 3 frac[22,14]
   - 3 frac[22,1] - 3 frac[22,24] + rows_tot_CPU23 = 0
 R1249: - 0.0833333333333333 rows_tot_CPU23 + normalized_rows_tot_CPU23
   = 0
 mem_tot_CPU24: - mem[23,97] - mem[23,119] - mem[23,106] - mem[23,113]
   - mem[23,111] - mem[23,100] - mem[23,92] - mem[23,116] - mem[23,118]
   - mem[23,108] - mem[23,101] - mem[23,90] - mem[23,114] + mem_tot_CPU24
   = 0
 R1251: - 3.05176e-05 mem_tot_CPU24 + normalized_mem_tot_CPU24 = 0
 rows_tot_CPU24: - 3 frac[23,97] - 3 frac[23,119] - 3 frac[23,106]
   - 3 frac[23,113] - 3 frac[23,111] - 3 frac[23,100] - 3 frac[23,92]
   - 3 frac[23,116] - 3 frac[23,118] - 3 frac[23,108] - 3 frac[23,101]
   - 3 frac[23,90] - 3 frac[23,114] + rows_tot_CPU24 = 0
 R1253: - 0.0833333333333333 rows_tot_CPU24 + normalized_rows_tot_CPU24
   = 0
 mem_tot_CPU25: - mem[24,75] - mem[24,61] - mem[24,82] - mem[24,70]
   - mem[24,77] - mem[24,65] - mem[24,72] - mem[24,79] - mem[24,62]
   - mem[24,64] - mem[24,71] - mem[24,85] + mem_tot_CPU25 = 0
 R1255: - 3.05176e-05 mem_tot_CPU25 + normalized_mem_tot_CPU25 = 0
 rows_tot_CPU25: - 3 frac[24,75] - 3 frac[24,61] - 3 frac[24,82]
   - 3 frac[24,70] - 3 frac[24,77] - 3 frac[24,65] - 3 frac[24,72]
   - 3 frac[24,79] - 3 frac[24,62] - 3 frac[24,64] - 3 frac[24,71]
   - 3 frac[24,85] + rows_tot_CPU25 = 0
 R1257: - 0.0833333333333333 rows_tot_CPU25 + normalized_rows_tot_CPU25
   = 0
 mem_tot_CPU26: - mem[25,3] - mem[25,4] - mem[25,1] - mem[25,7]
   - mem[25,26] - mem[25,24] - mem[25,2] - mem[25,9] + mem_tot_CPU26 = 0
 R1259: - 3.05176e-05 mem_tot_CPU26 + normalized_mem_tot_CPU26 = 0
 rows_tot_CPU26: - 3 frac[25,3] - 3 frac[25,4] - 3 frac[25,1]
   - 3 frac[25,7] - 3 frac[25,26] - 3 frac[25,24] - 3 frac[25,2]
   - 3 frac[25,9] + rows_tot_CPU26 = 0
 R1261: - 0.0833333333333333 rows_tot_CPU26 + normalized_rows_tot_CPU26
   = 0
 mem_tot_CPU27: - mem[26,47] - mem[26,45] - mem[26,32] - mem[26,43]
   - mem[26,46] - mem[26,38] + mem_tot_CPU27 = 0
 R1263: - 3.05176e-05 mem_tot_CPU27 + normalized_mem_tot_CPU27 = 0
 rows_tot_CPU27: - 3 frac[26,47] - 3 frac[26,45] - 3 frac[26,32]
   - 3 frac[26,43] - 3 frac[26,46] - 3 frac[26,38] + rows_tot_CPU27 = 0
 R1265: - 0.0833333333333333 rows_tot_CPU27 + normalized_rows_tot_CPU27
   = 0
 mem_tot_CPU28: - mem[27,68] - mem[27,72] - mem[27,62] - mem[27,84]
   - mem[27,74] - mem[27,67] - mem[27,86] - mem[27,61] - mem[27,71]
   - mem[27,75] - mem[27,79] + mem_tot_CPU28 = 0
 R1267: - 3.05176e-05 mem_tot_CPU28 + normalized_mem_tot_CPU28 = 0
 rows_tot_CPU28: - 3 frac[27,68] - 3 frac[27,72] - 3 frac[27,62]
   - 3 frac[27,84] - 3 frac[27,74] - 3 frac[27,67] - 3 frac[27,86]
   - 3 frac[27,61] - 3 frac[27,71] - 3 frac[27,75] - 3 frac[27,79]
   + rows_tot_CPU28 = 0
 R1269: - 0.0833333333333333 rows_tot_CPU28 + normalized_rows_tot_CPU28
   = 0
 mem_tot_CPU29: - mem[28,88] - mem[28,63] - mem[28,67] - mem[28,70]
   - mem[28,60] - mem[28,83] - mem[28,73] - mem[28,64] - mem[28,86]
   - mem[28,62] - mem[28,75] - mem[28,71] - mem[28,61] - mem[28,74]
   - mem[28,65] + mem_tot_CPU29 = 0
 R1271: - 3.05176e-05 mem_tot_CPU29 + normalized_mem_tot_CPU29 = 0
 rows_tot_CPU29: - 3 frac[28,88] - 3 frac[28,63] - 3 frac[28,67]
   - 3 frac[28,70] - 3 frac[28,60] - 3 frac[28,83] - 3 frac[28,73]
   - 3 frac[28,64] - 3 frac[28,86] - 3 frac[28,62] - 3 frac[28,75]
   - 3 frac[28,71] - 3 frac[28,61] - 3 frac[28,74] - 3 frac[28,65]
   + rows_tot_CPU29 = 0
 R1273: - 0.0833333333333333 rows_tot_CPU29 + normalized_rows_tot_CPU29
   = 0
 mem_tot_CPU30: - mem[29,50] - mem[29,55] - mem[29,48] - mem[29,47]
   - mem[29,58] - mem[29,51] - mem[29,30] - mem[29,36] - mem[29,34]
   - mem[29,52] - mem[29,46] - mem[29,31] - mem[29,57] + mem_tot_CPU30 = 0
 R1275: - 3.05176e-05 mem_tot_CPU30 + normalized_mem_tot_CPU30 = 0
 rows_tot_CPU30: - 3 frac[29,50] - 3 frac[29,55] - 3 frac[29,48]
   - 3 frac[29,47] - 3 frac[29,58] - 3 frac[29,51] - 3 frac[29,30]
   - 3 frac[29,36] - 3 frac[29,34] - 3 frac[29,52] - 3 frac[29,46]
   - 3 frac[29,31] - 3 frac[29,57] + rows_tot_CPU30 = 0
 R1277: - 0.0833333333333333 rows_tot_CPU30 + normalized_rows_tot_CPU30
   = 0
 mem_tot_CPU31: - mem[30,100] - mem[30,111] - mem[30,113] - mem[30,106]
   - mem[30,119] - mem[30,95] - mem[30,104] - mem[30,114] - mem[30,90]
   - mem[30,93] - mem[30,101] - mem[30,96] - mem[30,107] - mem[30,116]
   - mem[30,92] - mem[30,115] + mem_tot_CPU31 = 0
 R1279: - 3.05176e-05 mem_tot_CPU31 + normalized_mem_tot_CPU31 = 0
 rows_tot_CPU31: - 3 frac[30,100] - 3 frac[30,111] - 3 frac[30,113]
   - 3 frac[30,106] - 3 frac[30,119] - 3 frac[30,95] - 3 frac[30,104]
   - 3 frac[30,114] - 3 frac[30,90] - 3 frac[30,93] - 3 frac[30,101]
   - 3 frac[30,96] - 3 frac[30,107] - 3 frac[30,116] - 3 frac[30,92]
   - 3 frac[30,115] + rows_tot_CPU31 = 0
 R1281: - 0.0833333333333333 rows_tot_CPU31 + normalized_rows_tot_CPU31
   = 0
 mem_tot_CPU32: - mem[31,115] - mem[31,116] - mem[31,94] - mem[31,111]
   - mem[31,90] - mem[31,105] - mem[31,101] - mem[31,98] - mem[31,113]
   + mem_tot_CPU32 = 0
 R1283: - 3.05176e-05 mem_tot_CPU32 + normalized_mem_tot_CPU32 = 0
 rows_tot_CPU32: - 3 frac[31,115] - 3 frac[31,116] - 3 frac[31,94]
   - 3 frac[31,111] - 3 frac[31,90] - 3 frac[31,105] - 3 frac[31,101]
   - 3 frac[31,98] - 3 frac[31,113] + rows_tot_CPU32 = 0
 R1285: - 0.0833333333333333 rows_tot_CPU32 + normalized_rows_tot_CPU32
   = 0
 mem_tot_tor_P41: - mem[32,54] - mem[32,44] - mem[32,113] - mem[32,35]
   - mem[32,25] - mem[32,58] - mem[32,91] - mem[32,49] - mem[32,23]
   - mem[32,13] - mem[32,82] - mem[32,46] - mem[32,4] - mem[32,115]
   - mem[32,37] - mem[32,27] - mem[32,96] - mem[32,84] - mem[32,32]
   - mem[32,6] - mem[32,117] - mem[32,65] - mem[32,39] - mem[32,29]
   - mem[32,98] - mem[32,88] - mem[32,62] - mem[32,20] - mem[32,10]
   - mem[32,95] - mem[32,43] - mem[32,1] - mem[32,112] - mem[32,76]
   - mem[32,24] - mem[32,119] - mem[32,67] - mem[32,57] - mem[32,31]
   - mem[32,100] - mem[32,90] - mem[32,48] - mem[32,22] - mem[32,45]
   - mem[32,3] - mem[32,104] - mem[32,26] - mem[32,17] - mem[32,102]
   - mem[32,40] - mem[32,83] - mem[32,73] - mem[32,5] - mem[32,116]
   - mem[32,106] - mem[32,38] - mem[32,97] - mem[32,61] - mem[32,19]
   - mem[32,52] - mem[32,42] - mem[32,0] - mem[32,33] - mem[32,7]
   - mem[32,118] - mem[32,108] - mem[32,56] - mem[32,30] - mem[32,99]
   - mem[32,63] - mem[32,11] + mem_tot_tor_P41 = 0
 R1287: - 1.6276041666666666e-04 mem_tot_tor_P41
   + normalized_mem_tot_tor_P41 = 0
 rows_tot_tor_P41: - 3 frac[32,54] - 3 frac[32,44] - 3 frac[32,113]
   - 3 frac[32,35] - 3 frac[32,25] - 3 frac[32,58] - 3 frac[32,91]
   - 3 frac[32,49] - 3 frac[32,23] - 3 frac[32,13] - 3 frac[32,82]
   - 3 frac[32,46] - 3 frac[32,4] - 3 frac[32,115] - 3 frac[32,37]
   - 3 frac[32,27] - 3 frac[32,96] - 3 frac[32,84] - 3 frac[32,32]
   - 3 frac[32,6] - 3 frac[32,117] - 3 frac[32,65] - 3 frac[32,39]
   - 3 frac[32,29] - 3 frac[32,98] - 3 frac[32,88] - 3 frac[32,62]
   - 3 frac[32,20] - 3 frac[32,10] - 3 frac[32,95] - 3 frac[32,43]
   - 3 frac[32,1] - 3 frac[32,112] - 3 frac[32,76] - 3 frac[32,24]
   - 3 frac[32,119] - 3 frac[32,67] - 3 frac[32,57] - 3 frac[32,31]
   - 3 frac[32,100] - 3 frac[32,90] - 3 frac[32,48] - 3 frac[32,22]
   - 3 frac[32,45] - 3 frac[32,3] - 3 frac[32,104] - 3 frac[32,26]
   - 3 frac[32,17] - 3 frac[32,102] - 3 frac[32,40] - 3 frac[32,83]
   - 3 frac[32,73] - 3 frac[32,5] - 3 frac[32,116] - 3 frac[32,106]
   - 3 frac[32,38] - 3 frac[32,97] - 3 frac[32,61] - 3 frac[32,19]
   - 3 frac[32,52] - 3 frac[32,42] - 3 frac[32,0] - 3 frac[32,33]
   - 3 frac[32,7] - 3 frac[32,118] - 3 frac[32,108] - 3 frac[32,56]
   - 3 frac[32,30] - 3 frac[32,99] - 3 frac[32,63] - 3 frac[32,11]
   + rows_tot_tor_P41 = 0
 R1289: - 0.0208333333333333 rows_tot_tor_P41 + normalized_rows_tot_tor_P41
   = 0
 mem_tot_tor_P42: - mem[33,101] - mem[33,95] - mem[33,34] - mem[33,74]
   - mem[33,105] - mem[33,67] - mem[33,60] - mem[33,22] - mem[33,98]
   - mem[33,68] - mem[33,53] - mem[33,47] - mem[33,93] - mem[33,32]
   - mem[33,57] - mem[33,117] - mem[33,111] - mem[33,65] - mem[33,50]
   - mem[33,83] - mem[33,38] - mem[33,24] - mem[33,114] - mem[33,84]
   - mem[33,78] - mem[33,63] - mem[33,42] - mem[33,35] - mem[33,81]
   - mem[33,75] - mem[33,36] - mem[33,76] - mem[33,61] - mem[33,54]
   - mem[33,40] - mem[33,100] - mem[33,94] - mem[33,33] - mem[33,87]
   - mem[33,73] - mem[33,118] - mem[33,104] - mem[33,51] - mem[33,97]
   - mem[33,52] - mem[33,46] - mem[33,0] - mem[33,92] - mem[33,39]
   - mem[33,115] - mem[33,85] - mem[33,79] - mem[33,56] - mem[33,116]
   - mem[33,64] - mem[33,49] - mem[33,43] - mem[33,44] - mem[33,6]
   - mem[33,82] - mem[33,31] - mem[33,107] - mem[33,77] - mem[33,62]
   - mem[33,10] - mem[33,70] + mem_tot_tor_P42 = 0
 R1291: - 1.6276041666666666e-04 mem_tot_tor_P42
   + normalized_mem_tot_tor_P42 = 0
 rows_tot_tor_P42: - 3 frac[33,101] - 3 frac[33,95] - 3 frac[33,34]
   - 3 frac[33,74] - 3 frac[33,105] - 3 frac[33,67] - 3 frac[33,60]
   - 3 frac[33,22] - 3 frac[33,98] - 3 frac[33,68] - 3 frac[33,53]
   - 3 frac[33,47] - 3 frac[33,93] - 3 frac[33,32] - 3 frac[33,57]
   - 3 frac[33,117] - 3 frac[33,111] - 3 frac[33,65] - 3 frac[33,50]
   - 3 frac[33,83] - 3 frac[33,38] - 3 frac[33,24] - 3 frac[33,114]
   - 3 frac[33,84] - 3 frac[33,78] - 3 frac[33,63] - 3 frac[33,42]
   - 3 frac[33,35] - 3 frac[33,81] - 3 frac[33,75] - 3 frac[33,36]
   - 3 frac[33,76] - 3 frac[33,61] - 3 frac[33,54] - 3 frac[33,40]
   - 3 frac[33,100] - 3 frac[33,94] - 3 frac[33,33] - 3 frac[33,87]
   - 3 frac[33,73] - 3 frac[33,118] - 3 frac[33,104] - 3 frac[33,51]
   - 3 frac[33,97] - 3 frac[33,52] - 3 frac[33,46] - 3 frac[33,0]
   - 3 frac[33,92] - 3 frac[33,39] - 3 frac[33,115] - 3 frac[33,85]
   - 3 frac[33,79] - 3 frac[33,56] - 3 frac[33,116] - 3 frac[33,64]
   - 3 frac[33,49] - 3 frac[33,43] - 3 frac[33,44] - 3 frac[33,6]
   - 3 frac[33,82] - 3 frac[33,31] - 3 frac[33,107] - 3 frac[33,77]
   - 3 frac[33,62] - 3 frac[33,10] - 3 frac[33,70] + rows_tot_tor_P42 = 0
 R1293: - 0.0208333333333333 rows_tot_tor_P42 + normalized_rows_tot_tor_P42
   = 0
 mem_tot_tor_P43: - mem[34,52] - mem[34,46] - mem[34,85] - mem[34,79]
   - mem[34,3] - mem[34,114] - mem[34,108] - mem[34,32] - mem[34,26]
   - mem[34,65] - mem[34,15] - mem[34,50] - mem[34,44] - mem[34,6]
   - mem[34,83] - mem[34,1] - mem[34,112] - mem[34,106] - mem[34,62]
   - mem[34,24] - mem[34,101] - mem[34,57] - mem[34,19] - mem[34,13]
   - mem[34,4] - mem[34,119] - mem[34,75] - mem[34,37] - mem[34,104]
   - mem[34,22] - mem[34,99] - mem[34,55] - mem[34,17] - mem[34,11]
   - mem[34,84] - mem[34,40] - mem[34,2] - mem[34,111] - mem[34,73]
   - mem[34,29] - mem[34,102] - mem[34,58] - mem[34,20] - mem[34,14]
   - mem[34,97] - mem[34,91] - mem[34,53] - mem[34,47] - mem[34,9]
   - mem[34,82] - mem[34,38] - mem[34,0] - mem[34,27] - mem[34,100]
   - mem[34,94] - mem[34,56] - mem[34,18] - mem[34,7] - mem[34,118]
   - mem[34,74] - mem[34,113] - mem[34,107] - mem[34,63] - mem[34,25]
   - mem[34,92] - mem[34,54] - mem[34,10] - mem[34,87] - mem[34,49]
   - mem[34,43] - mem[34,5] - mem[34,116] - mem[34,34] - mem[34,23]
   - mem[34,90] + mem_tot_tor_P43 = 0
 R1295: - 1.6276041666666666e-04 mem_tot_tor_P43
   + normalized_mem_tot_tor_P43 = 0
 rows_tot_tor_P43: - 3 frac[34,52] - 3 frac[34,46] - 3 frac[34,85]
   - 3 frac[34,79] - 3 frac[34,3] - 3 frac[34,114] - 3 frac[34,108]
   - 3 frac[34,32] - 3 frac[34,26] - 3 frac[34,65] - 3 frac[34,15]
   - 3 frac[34,50] - 3 frac[34,44] - 3 frac[34,6] - 3 frac[34,83]
   - 3 frac[34,1] - 3 frac[34,112] - 3 frac[34,106] - 3 frac[34,62]
   - 3 frac[34,24] - 3 frac[34,101] - 3 frac[34,57] - 3 frac[34,19]
   - 3 frac[34,13] - 3 frac[34,4] - 3 frac[34,119] - 3 frac[34,75]
   - 3 frac[34,37] - 3 frac[34,104] - 3 frac[34,22] - 3 frac[34,99]
   - 3 frac[34,55] - 3 frac[34,17] - 3 frac[34,11] - 3 frac[34,84]
   - 3 frac[34,40] - 3 frac[34,2] - 3 frac[34,111] - 3 frac[34,73]
   - 3 frac[34,29] - 3 frac[34,102] - 3 frac[34,58] - 3 frac[34,20]
   - 3 frac[34,14] - 3 frac[34,97] - 3 frac[34,91] - 3 frac[34,53]
   - 3 frac[34,47] - 3 frac[34,9] - 3 frac[34,82] - 3 frac[34,38]
   - 3 frac[34,0] - 3 frac[34,27] - 3 frac[34,100] - 3 frac[34,94]
   - 3 frac[34,56] - 3 frac[34,18] - 3 frac[34,7] - 3 frac[34,118]
   - 3 frac[34,74] - 3 frac[34,113] - 3 frac[34,107] - 3 frac[34,63]
   - 3 frac[34,25] - 3 frac[34,92] - 3 frac[34,54] - 3 frac[34,10]
   - 3 frac[34,87] - 3 frac[34,49] - 3 frac[34,43] - 3 frac[34,5]
   - 3 frac[34,116] - 3 frac[34,34] - 3 frac[34,23] - 3 frac[34,90]
   + rows_tot_tor_P43 = 0
 R1297: - 0.0208333333333333 rows_tot_tor_P43 + normalized_rows_tot_tor_P43
   = 0
 mem_tot_tor_P44: - mem[35,88] - mem[35,65] - mem[35,46] - mem[35,52]
   - mem[35,26] - mem[35,96] - mem[35,70] - mem[35,61] - mem[35,3]
   - mem[35,105] - mem[35,79] - mem[35,85] - mem[35,34] - mem[35,116]
   - mem[35,90] - mem[35,43] - mem[35,67] - mem[35,98] - mem[35,72]
   - mem[35,63] - mem[35,107] - mem[35,113] - mem[35,36] - mem[35,92]
   - mem[35,45] - mem[35,51] - mem[35,30] - mem[35,100] - mem[35,74]
   - mem[35,7] - mem[35,115] - mem[35,38] - mem[35,94] - mem[35,47]
   - mem[35,71] - mem[35,58] - mem[35,82] - mem[35,9] - mem[35,111]
   - mem[35,64] - mem[35,55] - mem[35,73] - mem[35,60] - mem[35,2]
   - mem[35,104] - mem[35,84] - mem[35,119] - mem[35,93] - mem[35,48]
   - mem[35,57] - mem[35,31] - mem[35,101] - mem[35,75] - mem[35,62]
   - mem[35,4] - mem[35,106] - mem[35,86] - mem[35,95] - mem[35,50]
   - mem[35,24] - mem[35,68] - mem[35,1] - mem[35,77] - mem[35,83]
   - mem[35,32] - mem[35,114] + mem_tot_tor_P44 = 0
 R1299: - 1.6276041666666666e-04 mem_tot_tor_P44
   + normalized_mem_tot_tor_P44 = 0
 rows_tot_tor_P44: - 3 frac[35,88] - 3 frac[35,65] - 3 frac[35,46]
   - 3 frac[35,52] - 3 frac[35,26] - 3 frac[35,96] - 3 frac[35,70]
   - 3 frac[35,61] - 3 frac[35,3] - 3 frac[35,105] - 3 frac[35,79]
   - 3 frac[35,85] - 3 frac[35,34] - 3 frac[35,116] - 3 frac[35,90]
   - 3 frac[35,43] - 3 frac[35,67] - 3 frac[35,98] - 3 frac[35,72]
   - 3 frac[35,63] - 3 frac[35,107] - 3 frac[35,113] - 3 frac[35,36]
   - 3 frac[35,92] - 3 frac[35,45] - 3 frac[35,51] - 3 frac[35,30]
   - 3 frac[35,100] - 3 frac[35,74] - 3 frac[35,7] - 3 frac[35,115]
   - 3 frac[35,38] - 3 frac[35,94] - 3 frac[35,47] - 3 frac[35,71]
   - 3 frac[35,58] - 3 frac[35,82] - 3 frac[35,9] - 3 frac[35,111]
   - 3 frac[35,64] - 3 frac[35,55] - 3 frac[35,73] - 3 frac[35,60]
   - 3 frac[35,2] - 3 frac[35,104] - 3 frac[35,84] - 3 frac[35,119]
   - 3 frac[35,93] - 3 frac[35,48] - 3 frac[35,57] - 3 frac[35,31]
   - 3 frac[35,101] - 3 frac[35,75] - 3 frac[35,62] - 3 frac[35,4]
   - 3 frac[35,106] - 3 frac[35,86] - 3 frac[35,95] - 3 frac[35,50]
   - 3 frac[35,24] - 3 frac[35,68] - 3 frac[35,1] - 3 frac[35,77]
   - 3 frac[35,83] - 3 frac[35,32] - 3 frac[35,114] + rows_tot_tor_P44 = 0
 R1301: - 0.0208333333333333 rows_tot_tor_P44 + normalized_rows_tot_tor_P44
   = 0
 mem_tot_l1_P41: - mem[36,68] - mem[36,22] - mem[36,101] - mem[36,67]
   - mem[36,29] - mem[36,108] - mem[36,74] - mem[36,62] - mem[36,4]
   - mem[36,107] - mem[36,113] - mem[36,95] - mem[36,37] - mem[36,3]
   - mem[36,70] - mem[36,44] - mem[36,10] - mem[36,77] - mem[36,43]
   - mem[36,49] - mem[36,31] - mem[36,116] - mem[36,82] - mem[36,56]
   - mem[36,6] - mem[36,115] - mem[36,39] - mem[36,13] - mem[36,64]
   - mem[36,46] - mem[36,52] - mem[36,97] - mem[36,79] - mem[36,85]
   - mem[36,51] - mem[36,104] - mem[36,118] - mem[36,92] - mem[36,58]
   - mem[36,0] - mem[36,91] - mem[36,33] - mem[36,100] - mem[36,40]
   - mem[36,54] - mem[36,99] - mem[36,73] - mem[36,87] - mem[36,61]
   - mem[36,27] - mem[36,106] - mem[36,112] - mem[36,94] - mem[36,36]
   - mem[36,35] - mem[36,23] - mem[36,102] - mem[36,76] - mem[36,42]
   - mem[36,48] - mem[36,30] - mem[36,75] - mem[36,63] - mem[36,5]
   - mem[36,114] - mem[36,88] - mem[36,38] - mem[36,45] - mem[36,11]
   - mem[36,17] - mem[36,96] - mem[36,84] - mem[36,50] - mem[36,24]
   - mem[36,111] - mem[36,117] - mem[36,83] - mem[36,57] - mem[36,7]
   - mem[36,90] - mem[36,32] - mem[36,65] - mem[36,47] - mem[36,53]
   - mem[36,19] - mem[36,98] - mem[36,60] - mem[36,26] - mem[36,105]
   - mem[36,119] - mem[36,93] - mem[36,1] + mem_tot_l1_P41 = 0
 R1303: - 1.6276041666666666e-04 mem_tot_l1_P41 + normalized_mem_tot_l1_P41
   = 0
 rows_tot_l1_P41: - 3 frac[36,68] - 3 frac[36,22] - 3 frac[36,101]
   - 3 frac[36,67] - 3 frac[36,29] - 3 frac[36,108] - 3 frac[36,74]
   - 3 frac[36,62] - 3 frac[36,4] - 3 frac[36,107] - 3 frac[36,113]
   - 3 frac[36,95] - 3 frac[36,37] - 3 frac[36,3] - 3 frac[36,70]
   - 3 frac[36,44] - 3 frac[36,10] - 3 frac[36,77] - 3 frac[36,43]
   - 3 frac[36,49] - 3 frac[36,31] - 3 frac[36,116] - 3 frac[36,82]
   - 3 frac[36,56] - 3 frac[36,6] - 3 frac[36,115] - 3 frac[36,39]
   - 3 frac[36,13] - 3 frac[36,64] - 3 frac[36,46] - 3 frac[36,52]
   - 3 frac[36,97] - 3 frac[36,79] - 3 frac[36,85] - 3 frac[36,51]
   - 3 frac[36,104] - 3 frac[36,118] - 3 frac[36,92] - 3 frac[36,58]
   - 3 frac[36,0] - 3 frac[36,91] - 3 frac[36,33] - 3 frac[36,100]
   - 3 frac[36,40] - 3 frac[36,54] - 3 frac[36,99] - 3 frac[36,73]
   - 3 frac[36,87] - 3 frac[36,61] - 3 frac[36,27] - 3 frac[36,106]
   - 3 frac[36,112] - 3 frac[36,94] - 3 frac[36,36] - 3 frac[36,35]
   - 3 frac[36,23] - 3 frac[36,102] - 3 frac[36,76] - 3 frac[36,42]
   - 3 frac[36,48] - 3 frac[36,30] - 3 frac[36,75] - 3 frac[36,63]
   - 3 frac[36,5] - 3 frac[36,114] - 3 frac[36,88] - 3 frac[36,38]
   - 3 frac[36,45] - 3 frac[36,11] - 3 frac[36,17] - 3 frac[36,96]
   - 3 frac[36,84] - 3 frac[36,50] - 3 frac[36,24] - 3 frac[36,111]
   - 3 frac[36,117] - 3 frac[36,83] - 3 frac[36,57] - 3 frac[36,7]
   - 3 frac[36,90] - 3 frac[36,32] - 3 frac[36,65] - 3 frac[36,47]
   - 3 frac[36,53] - 3 frac[36,19] - 3 frac[36,98] - 3 frac[36,60]
   - 3 frac[36,26] - 3 frac[36,105] - 3 frac[36,119] - 3 frac[36,93]
   - 3 frac[36,1] + rows_tot_l1_P41 = 0
 R1305: - 0.0208333333333333 rows_tot_l1_P41 + normalized_rows_tot_l1_P41
   = 0
 mem_tot_l1_P42: - mem[37,82] - mem[37,13] - mem[37,23] - mem[37,113]
   - mem[37,91] - mem[37,44] - mem[37,54] - mem[37,68] - mem[37,63]
   - mem[37,99] - mem[37,77] - mem[37,87] - mem[37,56] - mem[37,2]
   - mem[37,108] - mem[37,118] - mem[37,11] - mem[37,52] - mem[37,30]
   - mem[37,61] - mem[37,7] - mem[37,97] - mem[37,75] - mem[37,85]
   - mem[37,38] - mem[37,0] - mem[37,106] - mem[37,116] - mem[37,94]
   - mem[37,47] - mem[37,9] - mem[37,19] - mem[37,40] - mem[37,50]
   - mem[37,102] - mem[37,64] - mem[37,5] - mem[37,111] - mem[37,73]
   - mem[37,83] - mem[37,36] - mem[37,104] - mem[37,114] - mem[37,92]
   - mem[37,45] - mem[37,55] - mem[37,17] - mem[37,48] - mem[37,26]
   - mem[37,100] - mem[37,57] - mem[37,3] - mem[37,119] - mem[37,34]
   - mem[37,22] - mem[37,112] - mem[37,90] - mem[37,43] - mem[37,53]
   - mem[37,31] - mem[37,62] - mem[37,24] - mem[37,98] - mem[37,1]
   - mem[37,107] - mem[37,95] - mem[37,32] - mem[37,10] - mem[37,88]
   - mem[37,51] - mem[37,29] - mem[37,65] - mem[37,60] - mem[37,6]
   - mem[37,96] - mem[37,74] - mem[37,84] - mem[37,37] - mem[37,105]
   - mem[37,93] - mem[37,46] - mem[37,70] - mem[37,49] - mem[37,27]
   - mem[37,101] - mem[37,79] - mem[37,58] - mem[37,4] + mem_tot_l1_P42
   = 0
 R1307: - 1.6276041666666666e-04 mem_tot_l1_P42 + normalized_mem_tot_l1_P42
   = 0
 rows_tot_l1_P42: - 3 frac[37,82] - 3 frac[37,13] - 3 frac[37,23]
   - 3 frac[37,113] - 3 frac[37,91] - 3 frac[37,44] - 3 frac[37,54]
   - 3 frac[37,68] - 3 frac[37,63] - 3 frac[37,99] - 3 frac[37,77]
   - 3 frac[37,87] - 3 frac[37,56] - 3 frac[37,2] - 3 frac[37,108]
   - 3 frac[37,118] - 3 frac[37,11] - 3 frac[37,52] - 3 frac[37,30]
   - 3 frac[37,61] - 3 frac[37,7] - 3 frac[37,97] - 3 frac[37,75]
   - 3 frac[37,85] - 3 frac[37,38] - 3 frac[37,0] - 3 frac[37,106]
   - 3 frac[37,116] - 3 frac[37,94] - 3 frac[37,47] - 3 frac[37,9]
   - 3 frac[37,19] - 3 frac[37,40] - 3 frac[37,50] - 3 frac[37,102]
   - 3 frac[37,64] - 3 frac[37,5] - 3 frac[37,111] - 3 frac[37,73]
   - 3 frac[37,83] - 3 frac[37,36] - 3 frac[37,104] - 3 frac[37,114]
   - 3 frac[37,92] - 3 frac[37,45] - 3 frac[37,55] - 3 frac[37,17]
   - 3 frac[37,48] - 3 frac[37,26] - 3 frac[37,100] - 3 frac[37,57]
   - 3 frac[37,3] - 3 frac[37,119] - 3 frac[37,34] - 3 frac[37,22]
   - 3 frac[37,112] - 3 frac[37,90] - 3 frac[37,43] - 3 frac[37,53]
   - 3 frac[37,31] - 3 frac[37,62] - 3 frac[37,24] - 3 frac[37,98]
   - 3 frac[37,1] - 3 frac[37,107] - 3 frac[37,95] - 3 frac[37,32]
   - 3 frac[37,10] - 3 frac[37,88] - 3 frac[37,51] - 3 frac[37,29]
   - 3 frac[37,65] - 3 frac[37,60] - 3 frac[37,6] - 3 frac[37,96]
   - 3 frac[37,74] - 3 frac[37,84] - 3 frac[37,37] - 3 frac[37,105]
   - 3 frac[37,93] - 3 frac[37,46] - 3 frac[37,70] - 3 frac[37,49]
   - 3 frac[37,27] - 3 frac[37,101] - 3 frac[37,79] - 3 frac[37,58]
   - 3 frac[37,4] + rows_tot_l1_P42 = 0
 R1309: - 0.0208333333333333 rows_tot_l1_P42 + normalized_rows_tot_l1_P42
   = 0
 mem_tot_l2_P4: - mem[38,40] - mem[38,119] - mem[38,73] - mem[38,5]
   - mem[38,31] - mem[38,64] - mem[38,90] - mem[38,60] - mem[38,22]
   - mem[38,97] - mem[38,93] - mem[38,114] - mem[38,84] - mem[38,46]
   - mem[38,0] - mem[38,26] - mem[38,79] - mem[38,29] - mem[38,108]
   - mem[38,70] - mem[38,88] - mem[38,50] - mem[38,83] - mem[38,53]
   - mem[38,112] - mem[38,74] - mem[38,44] - mem[38,6] - mem[38,24]
   - mem[38,107] - mem[38,77] - mem[38,57] - mem[38,19] - mem[38,98]
   - mem[38,68] - mem[38,94] - mem[38,48] - mem[38,10] - mem[38,101]
   - mem[38,43] - mem[38,13] - mem[38,118] - mem[38,4] - mem[38,30]
   - mem[38,105] - mem[38,37] - mem[38,17] - mem[38,96] - mem[38,92]
   - mem[38,54] - mem[38,87] - mem[38,3] - mem[38,116] - mem[38,32]
   - mem[38,58] - mem[38,111] - mem[38,91] - mem[38,61] - mem[38,23]
   - mem[38,102] - mem[38,82] - mem[38,52] - mem[38,85] - mem[38,47]
   - mem[38,1] - mem[38,27] - mem[38,106] - mem[38,38] - mem[38,56]
   - mem[38,51] - mem[38,100] - mem[38,113] - mem[38,75] - mem[38,45]
   - mem[38,7] - mem[38,104] - mem[38,36] - mem[38,62] - mem[38,99]
   - mem[38,95] - mem[38,49] - mem[38,11] + mem_tot_l2_P4 = 0
 R1311: - 1.6276041666666666e-04 mem_tot_l2_P4 + normalized_mem_tot_l2_P4
   = 0
 rows_tot_l2_P4: - 3 frac[38,40] - 3 frac[38,119] - 3 frac[38,73]
   - 3 frac[38,5] - 3 frac[38,31] - 3 frac[38,64] - 3 frac[38,90]
   - 3 frac[38,60] - 3 frac[38,22] - 3 frac[38,97] - 3 frac[38,93]
   - 3 frac[38,114] - 3 frac[38,84] - 3 frac[38,46] - 3 frac[38,0]
   - 3 frac[38,26] - 3 frac[38,79] - 3 frac[38,29] - 3 frac[38,108]
   - 3 frac[38,70] - 3 frac[38,88] - 3 frac[38,50] - 3 frac[38,83]
   - 3 frac[38,53] - 3 frac[38,112] - 3 frac[38,74] - 3 frac[38,44]
   - 3 frac[38,6] - 3 frac[38,24] - 3 frac[38,107] - 3 frac[38,77]
   - 3 frac[38,57] - 3 frac[38,19] - 3 frac[38,98] - 3 frac[38,68]
   - 3 frac[38,94] - 3 frac[38,48] - 3 frac[38,10] - 3 frac[38,101]
   - 3 frac[38,43] - 3 frac[38,13] - 3 frac[38,118] - 3 frac[38,4]
   - 3 frac[38,30] - 3 frac[38,105] - 3 frac[38,37] - 3 frac[38,17]
   - 3 frac[38,96] - 3 frac[38,92] - 3 frac[38,54] - 3 frac[38,87]
   - 3 frac[38,3] - 3 frac[38,116] - 3 frac[38,32] - 3 frac[38,58]
   - 3 frac[38,111] - 3 frac[38,91] - 3 frac[38,61] - 3 frac[38,23]
   - 3 frac[38,102] - 3 frac[38,82] - 3 frac[38,52] - 3 frac[38,85]
   - 3 frac[38,47] - 3 frac[38,1] - 3 frac[38,27] - 3 frac[38,106]
   - 3 frac[38,38] - 3 frac[38,56] - 3 frac[38,51] - 3 frac[38,100]
   - 3 frac[38,113] - 3 frac[38,75] - 3 frac[38,45] - 3 frac[38,7]
   - 3 frac[38,104] - 3 frac[38,36] - 3 frac[38,62] - 3 frac[38,99]
   - 3 frac[38,95] - 3 frac[38,49] - 3 frac[38,11] + rows_tot_l2_P4 = 0
 R1313: - 0.0208333333333333 rows_tot_l2_P4 + normalized_rows_tot_l2_P4
   = 0
 row_capacity_CPU1: rows_tot_CPU1 <= 12
 row_capacity_CPU2: rows_tot_CPU2 <= 12
 row_capacity_CPU3: rows_tot_CPU3 <= 12
 row_capacity_CPU4: rows_tot_CPU4 <= 12
 row_capacity_CPU5: rows_tot_CPU5 <= 12
 row_capacity_CPU6: rows_tot_CPU6 <= 12
 row_capacity_CPU7: rows_tot_CPU7 <= 12
 row_capacity_CPU8: rows_tot_CPU8 <= 12
 row_capacity_CPU9: rows_tot_CPU9 <= 12
 row_capacity_CPU10: rows_tot_CPU10 <= 12
 row_capacity_CPU11: rows_tot_CPU11 <= 12
 row_capacity_CPU12: rows_tot_CPU12 <= 12
 row_capacity_CPU13: rows_tot_CPU13 <= 12
 row_capacity_CPU14: rows_tot_CPU14 <= 12
 row_capacity_CPU15: rows_tot_CPU15 <= 12
 row_capacity_CPU16: rows_tot_CPU16 <= 12
 row_capacity_CPU17: rows_tot_CPU17 <= 12
 row_capacity_CPU18: rows_tot_CPU18 <= 12
 row_capacity_CPU19: rows_tot_CPU19 <= 12
 row_capacity_CPU20: rows_tot_CPU20 <= 12
 row_capacity_CPU21: rows_tot_CPU21 <= 12
 row_capacity_CPU22: rows_tot_CPU22 <= 12
 row_capacity_CPU23: rows_tot_CPU23 <= 12
 row_capacity_CPU24: rows_tot_CPU24 <= 12
 row_capacity_CPU25: rows_tot_CPU25 <= 12
 row_capacity_CPU26: rows_tot_CPU26 <= 12
 row_capacity_CPU27: rows_tot_CPU27 <= 12
 row_capacity_CPU28: rows_tot_CPU28 <= 12
 row_capacity_CPU29: rows_tot_CPU29 <= 12
 row_capacity_CPU30: rows_tot_CPU30 <= 12
 row_capacity_CPU31: rows_tot_CPU31 <= 12
 row_capacity_CPU32: rows_tot_CPU32 <= 12
 row_capacity_tor_P41: rows_tot_tor_P41 <= 48
 capacity_mem_par_tor_P41: mem[32,54] <= 1536
 capacity_mem_par_tor_P41: mem[32,44] <= 1536
 capacity_mem_par_tor_P41: mem[32,113] <= 1536
 capacity_mem_par_tor_P41: mem[32,35] <= 1536
 capacity_mem_par_tor_P41: mem[32,25] <= 1536
 capacity_mem_par_tor_P41: mem[32,58] <= 1536
 capacity_mem_par_tor_P41: mem[32,91] <= 1536
 capacity_mem_par_tor_P41: mem[32,49] <= 1536
 capacity_mem_par_tor_P41: mem[32,23] <= 1536
 capacity_mem_par_tor_P41: mem[32,13] <= 1536
 capacity_mem_par_tor_P41: mem[32,82] <= 1536
 capacity_mem_par_tor_P41: mem[32,46] <= 1536
 capacity_mem_par_tor_P41: mem[32,4] <= 1536
 capacity_mem_par_tor_P41: mem[32,115] <= 1536
 capacity_mem_par_tor_P41: mem[32,37] <= 1536
 capacity_mem_par_tor_P41: mem[32,27] <= 1536
 capacity_mem_par_tor_P41: mem[32,96] <= 1536
 capacity_mem_par_tor_P41: mem[32,84] <= 1536
 capacity_mem_par_tor_P41: mem[32,32] <= 1536
 capacity_mem_par_tor_P41: mem[32,6] <= 1536
 capacity_mem_par_tor_P41: mem[32,117] <= 1536
 capacity_mem_par_tor_P41: mem[32,65] <= 1536
 capacity_mem_par_tor_P41: mem[32,39] <= 1536
 capacity_mem_par_tor_P41: mem[32,29] <= 1536
 capacity_mem_par_tor_P41: mem[32,98] <= 1536
 capacity_mem_par_tor_P41: mem[32,88] <= 1536
 capacity_mem_par_tor_P41: mem[32,62] <= 1536
 capacity_mem_par_tor_P41: mem[32,20] <= 1536
 capacity_mem_par_tor_P41: mem[32,10] <= 1536
 capacity_mem_par_tor_P41: mem[32,95] <= 1536
 capacity_mem_par_tor_P41: mem[32,43] <= 1536
 capacity_mem_par_tor_P41: mem[32,1] <= 1536
 capacity_mem_par_tor_P41: mem[32,112] <= 1536
 capacity_mem_par_tor_P41: mem[32,76] <= 1536
 capacity_mem_par_tor_P41: mem[32,24] <= 1536
 capacity_mem_par_tor_P41: mem[32,119] <= 1536
 capacity_mem_par_tor_P41: mem[32,67] <= 1536
 capacity_mem_par_tor_P41: mem[32,57] <= 1536
 capacity_mem_par_tor_P41: mem[32,31] <= 1536
 capacity_mem_par_tor_P41: mem[32,100] <= 1536
 capacity_mem_par_tor_P41: mem[32,90] <= 1536
 capacity_mem_par_tor_P41: mem[32,48] <= 1536
 capacity_mem_par_tor_P41: mem[32,22] <= 1536
 capacity_mem_par_tor_P41: mem[32,45] <= 1536
 capacity_mem_par_tor_P41: mem[32,3] <= 1536
 capacity_mem_par_tor_P41: mem[32,104] <= 1536
 capacity_mem_par_tor_P41: mem[32,26] <= 1536
 capacity_mem_par_tor_P41: mem[32,17] <= 1536
 capacity_mem_par_tor_P41: mem[32,102] <= 1536
 capacity_mem_par_tor_P41: mem[32,40] <= 1536
 capacity_mem_par_tor_P41: mem[32,83] <= 1536
 capacity_mem_par_tor_P41: mem[32,73] <= 1536
 capacity_mem_par_tor_P41: mem[32,5] <= 1536
 capacity_mem_par_tor_P41: mem[32,116] <= 1536
 capacity_mem_par_tor_P41: mem[32,106] <= 1536
 capacity_mem_par_tor_P41: mem[32,38] <= 1536
 capacity_mem_par_tor_P41: mem[32,97] <= 1536
 capacity_mem_par_tor_P41: mem[32,61] <= 1536
 capacity_mem_par_tor_P41: mem[32,19] <= 1536
 capacity_mem_par_tor_P41: mem[32,52] <= 1536
 capacity_mem_par_tor_P41: mem[32,42] <= 1536
 capacity_mem_par_tor_P41: mem[32,0] <= 1536
 capacity_mem_par_tor_P41: mem[32,33] <= 1536
 capacity_mem_par_tor_P41: mem[32,7] <= 1536
 capacity_mem_par_tor_P41: mem[32,118] <= 1536
 capacity_mem_par_tor_P41: mem[32,108] <= 1536
 capacity_mem_par_tor_P41: mem[32,56] <= 1536
 capacity_mem_par_tor_P41: mem[32,30] <= 1536
 capacity_mem_par_tor_P41: mem[32,99] <= 1536
 capacity_mem_par_tor_P41: mem[32,63] <= 1536
 capacity_mem_par_tor_P41: mem[32,11] <= 1536
 row_capacity_tor_P42: rows_tot_tor_P42 <= 48
 capacity_mem_par_tor_P42: mem[33,101] <= 1536
 capacity_mem_par_tor_P42: mem[33,95] <= 1536
 capacity_mem_par_tor_P42: mem[33,34] <= 1536
 capacity_mem_par_tor_P42: mem[33,74] <= 1536
 capacity_mem_par_tor_P42: mem[33,105] <= 1536
 capacity_mem_par_tor_P42: mem[33,67] <= 1536
 capacity_mem_par_tor_P42: mem[33,60] <= 1536
 capacity_mem_par_tor_P42: mem[33,22] <= 1536
 capacity_mem_par_tor_P42: mem[33,98] <= 1536
 capacity_mem_par_tor_P42: mem[33,68] <= 1536
 capacity_mem_par_tor_P42: mem[33,53] <= 1536
 capacity_mem_par_tor_P42: mem[33,47] <= 1536
 capacity_mem_par_tor_P42: mem[33,93] <= 1536
 capacity_mem_par_tor_P42: mem[33,32] <= 1536
 capacity_mem_par_tor_P42: mem[33,57] <= 1536
 capacity_mem_par_tor_P42: mem[33,117] <= 1536
 capacity_mem_par_tor_P42: mem[33,111] <= 1536
 capacity_mem_par_tor_P42: mem[33,65] <= 1536
 capacity_mem_par_tor_P42: mem[33,50] <= 1536
 capacity_mem_par_tor_P42: mem[33,83] <= 1536
 capacity_mem_par_tor_P42: mem[33,38] <= 1536
 capacity_mem_par_tor_P42: mem[33,24] <= 1536
 capacity_mem_par_tor_P42: mem[33,114] <= 1536
 capacity_mem_par_tor_P42: mem[33,84] <= 1536
 capacity_mem_par_tor_P42: mem[33,78] <= 1536
 capacity_mem_par_tor_P42: mem[33,63] <= 1536
 capacity_mem_par_tor_P42: mem[33,42] <= 1536
 capacity_mem_par_tor_P42: mem[33,35] <= 1536
 capacity_mem_par_tor_P42: mem[33,81] <= 1536
 capacity_mem_par_tor_P42: mem[33,75] <= 1536
 capacity_mem_par_tor_P42: mem[33,36] <= 1536
 capacity_mem_par_tor_P42: mem[33,76] <= 1536
 capacity_mem_par_tor_P42: mem[33,61] <= 1536
 capacity_mem_par_tor_P42: mem[33,54] <= 1536
 capacity_mem_par_tor_P42: mem[33,40] <= 1536
 capacity_mem_par_tor_P42: mem[33,100] <= 1536
 capacity_mem_par_tor_P42: mem[33,94] <= 1536
 capacity_mem_par_tor_P42: mem[33,33] <= 1536
 capacity_mem_par_tor_P42: mem[33,87] <= 1536
 capacity_mem_par_tor_P42: mem[33,73] <= 1536
 capacity_mem_par_tor_P42: mem[33,118] <= 1536
 capacity_mem_par_tor_P42: mem[33,104] <= 1536
 capacity_mem_par_tor_P42: mem[33,51] <= 1536
 capacity_mem_par_tor_P42: mem[33,97] <= 1536
 capacity_mem_par_tor_P42: mem[33,52] <= 1536
 capacity_mem_par_tor_P42: mem[33,46] <= 1536
 capacity_mem_par_tor_P42: mem[33,0] <= 1536
 capacity_mem_par_tor_P42: mem[33,92] <= 1536
 capacity_mem_par_tor_P42: mem[33,39] <= 1536
 capacity_mem_par_tor_P42: mem[33,115] <= 1536
 capacity_mem_par_tor_P42: mem[33,85] <= 1536
 capacity_mem_par_tor_P42: mem[33,79] <= 1536
 capacity_mem_par_tor_P42: mem[33,56] <= 1536
 capacity_mem_par_tor_P42: mem[33,116] <= 1536
 capacity_mem_par_tor_P42: mem[33,64] <= 1536
 capacity_mem_par_tor_P42: mem[33,49] <= 1536
 capacity_mem_par_tor_P42: mem[33,43] <= 1536
 capacity_mem_par_tor_P42: mem[33,44] <= 1536
 capacity_mem_par_tor_P42: mem[33,6] <= 1536
 capacity_mem_par_tor_P42: mem[33,82] <= 1536
 capacity_mem_par_tor_P42: mem[33,31] <= 1536
 capacity_mem_par_tor_P42: mem[33,107] <= 1536
 capacity_mem_par_tor_P42: mem[33,77] <= 1536
 capacity_mem_par_tor_P42: mem[33,62] <= 1536
 capacity_mem_par_tor_P42: mem[33,10] <= 1536
 capacity_mem_par_tor_P42: mem[33,70] <= 1536
 row_capacity_tor_P43: rows_tot_tor_P43 <= 48
 capacity_mem_par_tor_P43: mem[34,52] <= 1536
 capacity_mem_par_tor_P43: mem[34,46] <= 1536
 capacity_mem_par_tor_P43: mem[34,85] <= 1536
 capacity_mem_par_tor_P43: mem[34,79] <= 1536
 capacity_mem_par_tor_P43: mem[34,3] <= 1536
 capacity_mem_par_tor_P43: mem[34,114] <= 1536
 capacity_mem_par_tor_P43: mem[34,108] <= 1536
 capacity_mem_par_tor_P43: mem[34,32] <= 1536
 capacity_mem_par_tor_P43: mem[34,26] <= 1536
 capacity_mem_par_tor_P43: mem[34,65] <= 1536
 capacity_mem_par_tor_P43: mem[34,15] <= 1536
 capacity_mem_par_tor_P43: mem[34,50] <= 1536
 capacity_mem_par_tor_P43: mem[34,44] <= 1536
 capacity_mem_par_tor_P43: mem[34,6] <= 1536
 capacity_mem_par_tor_P43: mem[34,83] <= 1536
 capacity_mem_par_tor_P43: mem[34,1] <= 1536
 capacity_mem_par_tor_P43: mem[34,112] <= 1536
 capacity_mem_par_tor_P43: mem[34,106] <= 1536
 capacity_mem_par_tor_P43: mem[34,62] <= 1536
 capacity_mem_par_tor_P43: mem[34,24] <= 1536
 capacity_mem_par_tor_P43: mem[34,101] <= 1536
 capacity_mem_par_tor_P43: mem[34,57] <= 1536
 capacity_mem_par_tor_P43: mem[34,19] <= 1536
 capacity_mem_par_tor_P43: mem[34,13] <= 1536
 capacity_mem_par_tor_P43: mem[34,4] <= 1536
 capacity_mem_par_tor_P43: mem[34,119] <= 1536
 capacity_mem_par_tor_P43: mem[34,75] <= 1536
 capacity_mem_par_tor_P43: mem[34,37] <= 1536
 capacity_mem_par_tor_P43: mem[34,104] <= 1536
 capacity_mem_par_tor_P43: mem[34,22] <= 1536
 capacity_mem_par_tor_P43: mem[34,99] <= 1536
 capacity_mem_par_tor_P43: mem[34,55] <= 1536
 capacity_mem_par_tor_P43: mem[34,17] <= 1536
 capacity_mem_par_tor_P43: mem[34,11] <= 1536
 capacity_mem_par_tor_P43: mem[34,84] <= 1536
 capacity_mem_par_tor_P43: mem[34,40] <= 1536
 capacity_mem_par_tor_P43: mem[34,2] <= 1536
 capacity_mem_par_tor_P43: mem[34,111] <= 1536
 capacity_mem_par_tor_P43: mem[34,73] <= 1536
 capacity_mem_par_tor_P43: mem[34,29] <= 1536
 capacity_mem_par_tor_P43: mem[34,102] <= 1536
 capacity_mem_par_tor_P43: mem[34,58] <= 1536
 capacity_mem_par_tor_P43: mem[34,20] <= 1536
 capacity_mem_par_tor_P43: mem[34,14] <= 1536
 capacity_mem_par_tor_P43: mem[34,97] <= 1536
 capacity_mem_par_tor_P43: mem[34,91] <= 1536
 capacity_mem_par_tor_P43: mem[34,53] <= 1536
 capacity_mem_par_tor_P43: mem[34,47] <= 1536
 capacity_mem_par_tor_P43: mem[34,9] <= 1536
 capacity_mem_par_tor_P43: mem[34,82] <= 1536
 capacity_mem_par_tor_P43: mem[34,38] <= 1536
 capacity_mem_par_tor_P43: mem[34,0] <= 1536
 capacity_mem_par_tor_P43: mem[34,27] <= 1536
 capacity_mem_par_tor_P43: mem[34,100] <= 1536
 capacity_mem_par_tor_P43: mem[34,94] <= 1536
 capacity_mem_par_tor_P43: mem[34,56] <= 1536
 capacity_mem_par_tor_P43: mem[34,18] <= 1536
 capacity_mem_par_tor_P43: mem[34,7] <= 1536
 capacity_mem_par_tor_P43: mem[34,118] <= 1536
 capacity_mem_par_tor_P43: mem[34,74] <= 1536
 capacity_mem_par_tor_P43: mem[34,113] <= 1536
 capacity_mem_par_tor_P43: mem[34,107] <= 1536
 capacity_mem_par_tor_P43: mem[34,63] <= 1536
 capacity_mem_par_tor_P43: mem[34,25] <= 1536
 capacity_mem_par_tor_P43: mem[34,92] <= 1536
 capacity_mem_par_tor_P43: mem[34,54] <= 1536
 capacity_mem_par_tor_P43: mem[34,10] <= 1536
 capacity_mem_par_tor_P43: mem[34,87] <= 1536
 capacity_mem_par_tor_P43: mem[34,49] <= 1536
 capacity_mem_par_tor_P43: mem[34,43] <= 1536
 capacity_mem_par_tor_P43: mem[34,5] <= 1536
 capacity_mem_par_tor_P43: mem[34,116] <= 1536
 capacity_mem_par_tor_P43: mem[34,34] <= 1536
 capacity_mem_par_tor_P43: mem[34,23] <= 1536
 capacity_mem_par_tor_P43: mem[34,90] <= 1536
 row_capacity_tor_P44: rows_tot_tor_P44 <= 48
 capacity_mem_par_tor_P44: mem[35,88] <= 1536
 capacity_mem_par_tor_P44: mem[35,65] <= 1536
 capacity_mem_par_tor_P44: mem[35,46] <= 1536
 capacity_mem_par_tor_P44: mem[35,52] <= 1536
 capacity_mem_par_tor_P44: mem[35,26] <= 1536
 capacity_mem_par_tor_P44: mem[35,96] <= 1536
 capacity_mem_par_tor_P44: mem[35,70] <= 1536
 capacity_mem_par_tor_P44: mem[35,61] <= 1536
 capacity_mem_par_tor_P44: mem[35,3] <= 1536
 capacity_mem_par_tor_P44: mem[35,105] <= 1536
 capacity_mem_par_tor_P44: mem[35,79] <= 1536
 capacity_mem_par_tor_P44: mem[35,85] <= 1536
 capacity_mem_par_tor_P44: mem[35,34] <= 1536
 capacity_mem_par_tor_P44: mem[35,116] <= 1536
 capacity_mem_par_tor_P44: mem[35,90] <= 1536
 capacity_mem_par_tor_P44: mem[35,43] <= 1536
 capacity_mem_par_tor_P44: mem[35,67] <= 1536
 capacity_mem_par_tor_P44: mem[35,98] <= 1536
 capacity_mem_par_tor_P44: mem[35,72] <= 1536
 capacity_mem_par_tor_P44: mem[35,63] <= 1536
 capacity_mem_par_tor_P44: mem[35,107] <= 1536
 capacity_mem_par_tor_P44: mem[35,113] <= 1536
 capacity_mem_par_tor_P44: mem[35,36] <= 1536
 capacity_mem_par_tor_P44: mem[35,92] <= 1536
 capacity_mem_par_tor_P44: mem[35,45] <= 1536
 capacity_mem_par_tor_P44: mem[35,51] <= 1536
 capacity_mem_par_tor_P44: mem[35,30] <= 1536
 capacity_mem_par_tor_P44: mem[35,100] <= 1536
 capacity_mem_par_tor_P44: mem[35,74] <= 1536
 capacity_mem_par_tor_P44: mem[35,7] <= 1536
 capacity_mem_par_tor_P44: mem[35,115] <= 1536
 capacity_mem_par_tor_P44: mem[35,38] <= 1536
 capacity_mem_par_tor_P44: mem[35,94] <= 1536
 capacity_mem_par_tor_P44: mem[35,47] <= 1536
 capacity_mem_par_tor_P44: mem[35,71] <= 1536
 capacity_mem_par_tor_P44: mem[35,58] <= 1536
 capacity_mem_par_tor_P44: mem[35,82] <= 1536
 capacity_mem_par_tor_P44: mem[35,9] <= 1536
 capacity_mem_par_tor_P44: mem[35,111] <= 1536
 capacity_mem_par_tor_P44: mem[35,64] <= 1536
 capacity_mem_par_tor_P44: mem[35,55] <= 1536
 capacity_mem_par_tor_P44: mem[35,73] <= 1536
 capacity_mem_par_tor_P44: mem[35,60] <= 1536
 capacity_mem_par_tor_P44: mem[35,2] <= 1536
 capacity_mem_par_tor_P44: mem[35,104] <= 1536
 capacity_mem_par_tor_P44: mem[35,84] <= 1536
 capacity_mem_par_tor_P44: mem[35,119] <= 1536
 capacity_mem_par_tor_P44: mem[35,93] <= 1536
 capacity_mem_par_tor_P44: mem[35,48] <= 1536
 capacity_mem_par_tor_P44: mem[35,57] <= 1536
 capacity_mem_par_tor_P44: mem[35,31] <= 1536
 capacity_mem_par_tor_P44: mem[35,101] <= 1536
 capacity_mem_par_tor_P44: mem[35,75] <= 1536
 capacity_mem_par_tor_P44: mem[35,62] <= 1536
 capacity_mem_par_tor_P44: mem[35,4] <= 1536
 capacity_mem_par_tor_P44: mem[35,106] <= 1536
 capacity_mem_par_tor_P44: mem[35,86] <= 1536
 capacity_mem_par_tor_P44: mem[35,95] <= 1536
 capacity_mem_par_tor_P44: mem[35,50] <= 1536
 capacity_mem_par_tor_P44: mem[35,24] <= 1536
 capacity_mem_par_tor_P44: mem[35,68] <= 1536
 capacity_mem_par_tor_P44: mem[35,1] <= 1536
 capacity_mem_par_tor_P44: mem[35,77] <= 1536
 capacity_mem_par_tor_P44: mem[35,83] <= 1536
 capacity_mem_par_tor_P44: mem[35,32] <= 1536
 capacity_mem_par_tor_P44: mem[35,114] <= 1536
 row_capacity_l1_P41: rows_tot_l1_P41 <= 48
 capacity_mem_par_l1_P41: mem[36,68] <= 1536
 capacity_mem_par_l1_P41: mem[36,22] <= 1536
 capacity_mem_par_l1_P41: mem[36,101] <= 1536
 capacity_mem_par_l1_P41: mem[36,67] <= 1536
 capacity_mem_par_l1_P41: mem[36,29] <= 1536
 capacity_mem_par_l1_P41: mem[36,108] <= 1536
 capacity_mem_par_l1_P41: mem[36,74] <= 1536
 capacity_mem_par_l1_P41: mem[36,62] <= 1536
 capacity_mem_par_l1_P41: mem[36,4] <= 1536
 capacity_mem_par_l1_P41: mem[36,107] <= 1536
 capacity_mem_par_l1_P41: mem[36,113] <= 1536
 capacity_mem_par_l1_P41: mem[36,95] <= 1536
 capacity_mem_par_l1_P41: mem[36,37] <= 1536
 capacity_mem_par_l1_P41: mem[36,3] <= 1536
 capacity_mem_par_l1_P41: mem[36,70] <= 1536
 capacity_mem_par_l1_P41: mem[36,44] <= 1536
 capacity_mem_par_l1_P41: mem[36,10] <= 1536
 capacity_mem_par_l1_P41: mem[36,77] <= 1536
 capacity_mem_par_l1_P41: mem[36,43] <= 1536
 capacity_mem_par_l1_P41: mem[36,49] <= 1536
 capacity_mem_par_l1_P41: mem[36,31] <= 1536
 capacity_mem_par_l1_P41: mem[36,116] <= 1536
 capacity_mem_par_l1_P41: mem[36,82] <= 1536
 capacity_mem_par_l1_P41: mem[36,56] <= 1536
 capacity_mem_par_l1_P41: mem[36,6] <= 1536
 capacity_mem_par_l1_P41: mem[36,115] <= 1536
 capacity_mem_par_l1_P41: mem[36,39] <= 1536
 capacity_mem_par_l1_P41: mem[36,13] <= 1536
 capacity_mem_par_l1_P41: mem[36,64] <= 1536
 capacity_mem_par_l1_P41: mem[36,46] <= 1536
 capacity_mem_par_l1_P41: mem[36,52] <= 1536
 capacity_mem_par_l1_P41: mem[36,97] <= 1536
 capacity_mem_par_l1_P41: mem[36,79] <= 1536
 capacity_mem_par_l1_P41: mem[36,85] <= 1536
 capacity_mem_par_l1_P41: mem[36,51] <= 1536
 capacity_mem_par_l1_P41: mem[36,104] <= 1536
 capacity_mem_par_l1_P41: mem[36,118] <= 1536
 capacity_mem_par_l1_P41: mem[36,92] <= 1536
 capacity_mem_par_l1_P41: mem[36,58] <= 1536
 capacity_mem_par_l1_P41: mem[36,0] <= 1536
 capacity_mem_par_l1_P41: mem[36,91] <= 1536
 capacity_mem_par_l1_P41: mem[36,33] <= 1536
 capacity_mem_par_l1_P41: mem[36,100] <= 1536
 capacity_mem_par_l1_P41: mem[36,40] <= 1536
 capacity_mem_par_l1_P41: mem[36,54] <= 1536
 capacity_mem_par_l1_P41: mem[36,99] <= 1536
 capacity_mem_par_l1_P41: mem[36,73] <= 1536
 capacity_mem_par_l1_P41: mem[36,87] <= 1536
 capacity_mem_par_l1_P41: mem[36,61] <= 1536
 capacity_mem_par_l1_P41: mem[36,27] <= 1536
 capacity_mem_par_l1_P41: mem[36,106] <= 1536
 capacity_mem_par_l1_P41: mem[36,112] <= 1536
 capacity_mem_par_l1_P41: mem[36,94] <= 1536
 capacity_mem_par_l1_P41: mem[36,36] <= 1536
 capacity_mem_par_l1_P41: mem[36,35] <= 1536
 capacity_mem_par_l1_P41: mem[36,23] <= 1536
 capacity_mem_par_l1_P41: mem[36,102] <= 1536
 capacity_mem_par_l1_P41: mem[36,76] <= 1536
 capacity_mem_par_l1_P41: mem[36,42] <= 1536
 capacity_mem_par_l1_P41: mem[36,48] <= 1536
 capacity_mem_par_l1_P41: mem[36,30] <= 1536
 capacity_mem_par_l1_P41: mem[36,75] <= 1536
 capacity_mem_par_l1_P41: mem[36,63] <= 1536
 capacity_mem_par_l1_P41: mem[36,5] <= 1536
 capacity_mem_par_l1_P41: mem[36,114] <= 1536
 capacity_mem_par_l1_P41: mem[36,88] <= 1536
 capacity_mem_par_l1_P41: mem[36,38] <= 1536
 capacity_mem_par_l1_P41: mem[36,45] <= 1536
 capacity_mem_par_l1_P41: mem[36,11] <= 1536
 capacity_mem_par_l1_P41: mem[36,17] <= 1536
 capacity_mem_par_l1_P41: mem[36,96] <= 1536
 capacity_mem_par_l1_P41: mem[36,84] <= 1536
 capacity_mem_par_l1_P41: mem[36,50] <= 1536
 capacity_mem_par_l1_P41: mem[36,24] <= 1536
 capacity_mem_par_l1_P41: mem[36,111] <= 1536
 capacity_mem_par_l1_P41: mem[36,117] <= 1536
 capacity_mem_par_l1_P41: mem[36,83] <= 1536
 capacity_mem_par_l1_P41: mem[36,57] <= 1536
 capacity_mem_par_l1_P41: mem[36,7] <= 1536
 capacity_mem_par_l1_P41: mem[36,90] <= 1536
 capacity_mem_par_l1_P41: mem[36,32] <= 1536
 capacity_mem_par_l1_P41: mem[36,65] <= 1536
 capacity_mem_par_l1_P41: mem[36,47] <= 1536
 capacity_mem_par_l1_P41: mem[36,53] <= 1536
 capacity_mem_par_l1_P41: mem[36,19] <= 1536
 capacity_mem_par_l1_P41: mem[36,98] <= 1536
 capacity_mem_par_l1_P41: mem[36,60] <= 1536
 capacity_mem_par_l1_P41: mem[36,26] <= 1536
 capacity_mem_par_l1_P41: mem[36,105] <= 1536
 capacity_mem_par_l1_P41: mem[36,119] <= 1536
 capacity_mem_par_l1_P41: mem[36,93] <= 1536
 capacity_mem_par_l1_P41: mem[36,1] <= 1536
 row_capacity_l1_P42: rows_tot_l1_P42 <= 48
 capacity_mem_par_l1_P42: mem[37,82] <= 1536
 capacity_mem_par_l1_P42: mem[37,13] <= 1536
 capacity_mem_par_l1_P42: mem[37,23] <= 1536
 capacity_mem_par_l1_P42: mem[37,113] <= 1536
 capacity_mem_par_l1_P42: mem[37,91] <= 1536
 capacity_mem_par_l1_P42: mem[37,44] <= 1536
 capacity_mem_par_l1_P42: mem[37,54] <= 1536
 capacity_mem_par_l1_P42: mem[37,68] <= 1536
 capacity_mem_par_l1_P42: mem[37,63] <= 1536
 capacity_mem_par_l1_P42: mem[37,99] <= 1536
 capacity_mem_par_l1_P42: mem[37,77] <= 1536
 capacity_mem_par_l1_P42: mem[37,87] <= 1536
 capacity_mem_par_l1_P42: mem[37,56] <= 1536
 capacity_mem_par_l1_P42: mem[37,2] <= 1536
 capacity_mem_par_l1_P42: mem[37,108] <= 1536
 capacity_mem_par_l1_P42: mem[37,118] <= 1536
 capacity_mem_par_l1_P42: mem[37,11] <= 1536
 capacity_mem_par_l1_P42: mem[37,52] <= 1536
 capacity_mem_par_l1_P42: mem[37,30] <= 1536
 capacity_mem_par_l1_P42: mem[37,61] <= 1536
 capacity_mem_par_l1_P42: mem[37,7] <= 1536
 capacity_mem_par_l1_P42: mem[37,97] <= 1536
 capacity_mem_par_l1_P42: mem[37,75] <= 1536
 capacity_mem_par_l1_P42: mem[37,85] <= 1536
 capacity_mem_par_l1_P42: mem[37,38] <= 1536
 capacity_mem_par_l1_P42: mem[37,0] <= 1536
 capacity_mem_par_l1_P42: mem[37,106] <= 1536
 capacity_mem_par_l1_P42: mem[37,116] <= 1536
 capacity_mem_par_l1_P42: mem[37,94] <= 1536
 capacity_mem_par_l1_P42: mem[37,47] <= 1536
 capacity_mem_par_l1_P42: mem[37,9] <= 1536
 capacity_mem_par_l1_P42: mem[37,19] <= 1536
 capacity_mem_par_l1_P42: mem[37,40] <= 1536
 capacity_mem_par_l1_P42: mem[37,50] <= 1536
 capacity_mem_par_l1_P42: mem[37,102] <= 1536
 capacity_mem_par_l1_P42: mem[37,64] <= 1536
 capacity_mem_par_l1_P42: mem[37,5] <= 1536
 capacity_mem_par_l1_P42: mem[37,111] <= 1536
 capacity_mem_par_l1_P42: mem[37,73] <= 1536
 capacity_mem_par_l1_P42: mem[37,83] <= 1536
 capacity_mem_par_l1_P42: mem[37,36] <= 1536
 capacity_mem_par_l1_P42: mem[37,104] <= 1536
 capacity_mem_par_l1_P42: mem[37,114] <= 1536
 capacity_mem_par_l1_P42: mem[37,92] <= 1536
 capacity_mem_par_l1_P42: mem[37,45] <= 1536
 capacity_mem_par_l1_P42: mem[37,55] <= 1536
 capacity_mem_par_l1_P42: mem[37,17] <= 1536
 capacity_mem_par_l1_P42: mem[37,48] <= 1536
 capacity_mem_par_l1_P42: mem[37,26] <= 1536
 capacity_mem_par_l1_P42: mem[37,100] <= 1536
 capacity_mem_par_l1_P42: mem[37,57] <= 1536
 capacity_mem_par_l1_P42: mem[37,3] <= 1536
 capacity_mem_par_l1_P42: mem[37,119] <= 1536
 capacity_mem_par_l1_P42: mem[37,34] <= 1536
 capacity_mem_par_l1_P42: mem[37,22] <= 1536
 capacity_mem_par_l1_P42: mem[37,112] <= 1536
 capacity_mem_par_l1_P42: mem[37,90] <= 1536
 capacity_mem_par_l1_P42: mem[37,43] <= 1536
 capacity_mem_par_l1_P42: mem[37,53] <= 1536
 capacity_mem_par_l1_P42: mem[37,31] <= 1536
 capacity_mem_par_l1_P42: mem[37,62] <= 1536
 capacity_mem_par_l1_P42: mem[37,24] <= 1536
 capacity_mem_par_l1_P42: mem[37,98] <= 1536
 capacity_mem_par_l1_P42: mem[37,1] <= 1536
 capacity_mem_par_l1_P42: mem[37,107] <= 1536
 capacity_mem_par_l1_P42: mem[37,95] <= 1536
 capacity_mem_par_l1_P42: mem[37,32] <= 1536
 capacity_mem_par_l1_P42: mem[37,10] <= 1536
 capacity_mem_par_l1_P42: mem[37,88] <= 1536
 capacity_mem_par_l1_P42: mem[37,51] <= 1536
 capacity_mem_par_l1_P42: mem[37,29] <= 1536
 capacity_mem_par_l1_P42: mem[37,65] <= 1536
 capacity_mem_par_l1_P42: mem[37,60] <= 1536
 capacity_mem_par_l1_P42: mem[37,6] <= 1536
 capacity_mem_par_l1_P42: mem[37,96] <= 1536
 capacity_mem_par_l1_P42: mem[37,74] <= 1536
 capacity_mem_par_l1_P42: mem[37,84] <= 1536
 capacity_mem_par_l1_P42: mem[37,37] <= 1536
 capacity_mem_par_l1_P42: mem[37,105] <= 1536
 capacity_mem_par_l1_P42: mem[37,93] <= 1536
 capacity_mem_par_l1_P42: mem[37,46] <= 1536
 capacity_mem_par_l1_P42: mem[37,70] <= 1536
 capacity_mem_par_l1_P42: mem[37,49] <= 1536
 capacity_mem_par_l1_P42: mem[37,27] <= 1536
 capacity_mem_par_l1_P42: mem[37,101] <= 1536
 capacity_mem_par_l1_P42: mem[37,79] <= 1536
 capacity_mem_par_l1_P42: mem[37,58] <= 1536
 capacity_mem_par_l1_P42: mem[37,4] <= 1536
 row_capacity_l2_P4: rows_tot_l2_P4 <= 48
 capacity_mem_par_l2_P4: mem[38,40] <= 1536
 capacity_mem_par_l2_P4: mem[38,119] <= 1536
 capacity_mem_par_l2_P4: mem[38,73] <= 1536
 capacity_mem_par_l2_P4: mem[38,5] <= 1536
 capacity_mem_par_l2_P4: mem[38,31] <= 1536
 capacity_mem_par_l2_P4: mem[38,64] <= 1536
 capacity_mem_par_l2_P4: mem[38,90] <= 1536
 capacity_mem_par_l2_P4: mem[38,60] <= 1536
 capacity_mem_par_l2_P4: mem[38,22] <= 1536
 capacity_mem_par_l2_P4: mem[38,97] <= 1536
 capacity_mem_par_l2_P4: mem[38,93] <= 1536
 capacity_mem_par_l2_P4: mem[38,114] <= 1536
 capacity_mem_par_l2_P4: mem[38,84] <= 1536
 capacity_mem_par_l2_P4: mem[38,46] <= 1536
 capacity_mem_par_l2_P4: mem[38,0] <= 1536
 capacity_mem_par_l2_P4: mem[38,26] <= 1536
 capacity_mem_par_l2_P4: mem[38,79] <= 1536
 capacity_mem_par_l2_P4: mem[38,29] <= 1536
 capacity_mem_par_l2_P4: mem[38,108] <= 1536
 capacity_mem_par_l2_P4: mem[38,70] <= 1536
 capacity_mem_par_l2_P4: mem[38,88] <= 1536
 capacity_mem_par_l2_P4: mem[38,50] <= 1536
 capacity_mem_par_l2_P4: mem[38,83] <= 1536
 capacity_mem_par_l2_P4: mem[38,53] <= 1536
 capacity_mem_par_l2_P4: mem[38,112] <= 1536
 capacity_mem_par_l2_P4: mem[38,74] <= 1536
 capacity_mem_par_l2_P4: mem[38,44] <= 1536
 capacity_mem_par_l2_P4: mem[38,6] <= 1536
 capacity_mem_par_l2_P4: mem[38,24] <= 1536
 capacity_mem_par_l2_P4: mem[38,107] <= 1536
 capacity_mem_par_l2_P4: mem[38,77] <= 1536
 capacity_mem_par_l2_P4: mem[38,57] <= 1536
 capacity_mem_par_l2_P4: mem[38,19] <= 1536
 capacity_mem_par_l2_P4: mem[38,98] <= 1536
 capacity_mem_par_l2_P4: mem[38,68] <= 1536
 capacity_mem_par_l2_P4: mem[38,94] <= 1536
 capacity_mem_par_l2_P4: mem[38,48] <= 1536
 capacity_mem_par_l2_P4: mem[38,10] <= 1536
 capacity_mem_par_l2_P4: mem[38,101] <= 1536
 capacity_mem_par_l2_P4: mem[38,43] <= 1536
 capacity_mem_par_l2_P4: mem[38,13] <= 1536
 capacity_mem_par_l2_P4: mem[38,118] <= 1536
 capacity_mem_par_l2_P4: mem[38,4] <= 1536
 capacity_mem_par_l2_P4: mem[38,30] <= 1536
 capacity_mem_par_l2_P4: mem[38,105] <= 1536
 capacity_mem_par_l2_P4: mem[38,37] <= 1536
 capacity_mem_par_l2_P4: mem[38,17] <= 1536
 capacity_mem_par_l2_P4: mem[38,96] <= 1536
 capacity_mem_par_l2_P4: mem[38,92] <= 1536
 capacity_mem_par_l2_P4: mem[38,54] <= 1536
 capacity_mem_par_l2_P4: mem[38,87] <= 1536
 capacity_mem_par_l2_P4: mem[38,3] <= 1536
 capacity_mem_par_l2_P4: mem[38,116] <= 1536
 capacity_mem_par_l2_P4: mem[38,32] <= 1536
 capacity_mem_par_l2_P4: mem[38,58] <= 1536
 capacity_mem_par_l2_P4: mem[38,111] <= 1536
 capacity_mem_par_l2_P4: mem[38,91] <= 1536
 capacity_mem_par_l2_P4: mem[38,61] <= 1536
 capacity_mem_par_l2_P4: mem[38,23] <= 1536
 capacity_mem_par_l2_P4: mem[38,102] <= 1536
 capacity_mem_par_l2_P4: mem[38,82] <= 1536
 capacity_mem_par_l2_P4: mem[38,52] <= 1536
 capacity_mem_par_l2_P4: mem[38,85] <= 1536
 capacity_mem_par_l2_P4: mem[38,47] <= 1536
 capacity_mem_par_l2_P4: mem[38,1] <= 1536
 capacity_mem_par_l2_P4: mem[38,27] <= 1536
 capacity_mem_par_l2_P4: mem[38,106] <= 1536
 capacity_mem_par_l2_P4: mem[38,38] <= 1536
 capacity_mem_par_l2_P4: mem[38,56] <= 1536
 capacity_mem_par_l2_P4: mem[38,51] <= 1536
 capacity_mem_par_l2_P4: mem[38,100] <= 1536
 capacity_mem_par_l2_P4: mem[38,113] <= 1536
 capacity_mem_par_l2_P4: mem[38,75] <= 1536
 capacity_mem_par_l2_P4: mem[38,45] <= 1536
 capacity_mem_par_l2_P4: mem[38,7] <= 1536
 capacity_mem_par_l2_P4: mem[38,104] <= 1536
 capacity_mem_par_l2_P4: mem[38,36] <= 1536
 capacity_mem_par_l2_P4: mem[38,62] <= 1536
 capacity_mem_par_l2_P4: mem[38,99] <= 1536
 capacity_mem_par_l2_P4: mem[38,95] <= 1536
 capacity_mem_par_l2_P4: mem[38,49] <= 1536
 capacity_mem_par_l2_P4: mem[38,11] <= 1536
Bounds
 frac[0,5] <= 1
 frac[0,0] <= 1
 frac[0,7] <= 1
 frac[0,11] <= 1
 frac[0,25] <= 1
 frac[0,23] <= 1
 frac[0,13] <= 1
 frac[0,4] <= 1
 frac[0,27] <= 1
 frac[0,20] <= 1
 frac[0,10] <= 1
 frac[0,1] <= 1
 frac[0,24] <= 1
 frac[1,40] <= 1
 frac[1,58] <= 1
 frac[1,52] <= 1
 frac[1,56] <= 1
 frac[1,31] <= 1
 frac[1,49] <= 1
 frac[1,43] <= 1
 frac[1,44] <= 1
 frac[1,32] <= 1
 frac[1,57] <= 1
 frac[1,38] <= 1
 frac[1,48] <= 1
 frac[2,104] <= 1
 frac[2,99] <= 1
 frac[2,117] <= 1
 frac[2,97] <= 1
 frac[2,115] <= 1
 frac[2,100] <= 1
 frac[2,118] <= 1
 frac[2,98] <= 1
 frac[2,96] <= 1
 frac[2,95] <= 1
 frac[2,106] <= 1
 frac[2,119] <= 1
 frac[3,35] <= 1
 frac[3,40] <= 1
 frac[3,37] <= 1
 frac[3,42] <= 1
 frac[3,39] <= 1
 frac[3,44] <= 1
 frac[3,32] <= 1
 frac[3,46] <= 1
 frac[3,52] <= 1
 frac[3,54] <= 1
 frac[3,30] <= 1
 frac[3,45] <= 1
 frac[3,33] <= 1
 frac[3,38] <= 1
 frac[4,11] <= 1
 frac[4,20] <= 1
 frac[4,19] <= 1
 frac[4,26] <= 1
 frac[4,22] <= 1
 frac[4,3] <= 1
 frac[4,6] <= 1
 frac[4,25] <= 1
 frac[5,84] <= 1
 frac[5,62] <= 1
 frac[5,82] <= 1
 frac[5,63] <= 1
 frac[5,73] <= 1
 frac[5,83] <= 1
 frac[5,61] <= 1
 frac[5,67] <= 1
 frac[5,76] <= 1
 frac[5,88] <= 1
 frac[5,65] <= 1
 frac[6,1] <= 1
 frac[6,7] <= 1
 frac[6,0] <= 1
 frac[6,26] <= 1
 frac[6,29] <= 1
 frac[6,24] <= 1
 frac[6,13] <= 1
 frac[6,4] <= 1
 frac[6,17] <= 1
 frac[6,3] <= 1
 frac[7,98] <= 1
 frac[7,91] <= 1
 frac[7,113] <= 1
 frac[7,104] <= 1
 frac[7,97] <= 1
 frac[7,116] <= 1
 frac[7,118] <= 1
 frac[7,100] <= 1
 frac[7,90] <= 1
 frac[7,112] <= 1
 frac[7,102] <= 1
 frac[7,95] <= 1
 frac[7,108] <= 1
 frac[8,38] <= 1
 frac[8,51] <= 1
 frac[8,33] <= 1
 frac[8,53] <= 1
 frac[8,35] <= 1
 frac[8,42] <= 1
 frac[8,44] <= 1
 frac[8,50] <= 1
 frac[8,57] <= 1
 frac[8,46] <= 1
 frac[8,34] <= 1
 frac[8,36] <= 1
 frac[8,43] <= 1
 frac[8,56] <= 1
 frac[9,62] <= 1
 frac[9,70] <= 1
 frac[9,81] <= 1
 frac[9,60] <= 1
 frac[9,68] <= 1
 frac[9,65] <= 1
 frac[9,82] <= 1
 frac[9,61] <= 1
 frac[9,64] <= 1
 frac[10,75] <= 1
 frac[10,85] <= 1
 frac[10,78] <= 1
 frac[10,73] <= 1
 frac[10,62] <= 1
 frac[10,81] <= 1
 frac[10,67] <= 1
 frac[10,74] <= 1
 frac[10,84] <= 1
 frac[10,65] <= 1
 frac[10,79] <= 1
 frac[10,63] <= 1
 frac[10,64] <= 1
 frac[11,114] <= 1
 frac[11,107] <= 1
 frac[11,116] <= 1
 frac[11,94] <= 1
 frac[11,118] <= 1
 frac[11,111] <= 1
 frac[11,100] <= 1
 frac[11,104] <= 1
 frac[11,117] <= 1
 frac[11,95] <= 1
 frac[11,101] <= 1
 frac[11,105] <= 1
 frac[12,93] <= 1
 frac[12,98] <= 1
 frac[12,101] <= 1
 frac[12,92] <= 1
 frac[12,116] <= 1
 frac[12,115] <= 1
 frac[12,97] <= 1
 frac[13,70] <= 1
 frac[13,75] <= 1
 frac[13,68] <= 1
 frac[13,87] <= 1
 frac[13,73] <= 1
 frac[13,78] <= 1
 frac[13,64] <= 1
 frac[13,83] <= 1
 frac[13,76] <= 1
 frac[13,62] <= 1
 frac[13,65] <= 1
 frac[13,84] <= 1
 frac[13,77] <= 1
 frac[14,31] <= 1
 frac[14,38] <= 1
 frac[14,36] <= 1
 frac[14,47] <= 1
 frac[14,54] <= 1
 frac[14,57] <= 1
 frac[14,52] <= 1
 frac[14,40] <= 1
 frac[14,39] <= 1
 frac[14,46] <= 1
 frac[14,49] <= 1
 frac[14,56] <= 1
 frac[14,34] <= 1
 frac[14,44] <= 1
 frac[14,32] <= 1
 frac[15,10] <= 1
 frac[15,24] <= 1
 frac[15,6] <= 1
 frac[15,0] <= 1
 frac[15,22] <= 1
 frac[16,7] <= 1
 frac[16,29] <= 1
 frac[16,2] <= 1
 frac[16,24] <= 1
 frac[16,22] <= 1
 frac[16,13] <= 1
 frac[16,4] <= 1
 frac[16,26] <= 1
 frac[16,6] <= 1
 frac[16,10] <= 1
 frac[16,3] <= 1
 frac[16,23] <= 1
 frac[16,14] <= 1
 frac[16,18] <= 1
 frac[17,46] <= 1
 frac[17,57] <= 1
 frac[17,50] <= 1
 frac[17,43] <= 1
 frac[17,44] <= 1
 frac[17,37] <= 1
 frac[17,54] <= 1
 frac[17,47] <= 1
 frac[17,32] <= 1
 frac[17,38] <= 1
 frac[17,40] <= 1
 frac[18,10] <= 1
 frac[18,22] <= 1
 frac[18,17] <= 1
 frac[18,29] <= 1
 frac[18,20] <= 1
 frac[18,15] <= 1
 frac[18,4] <= 1
 frac[18,25] <= 1
 frac[18,23] <= 1
 frac[18,14] <= 1
 frac[18,0] <= 1
 frac[18,26] <= 1
 frac[19,53] <= 1
 frac[19,58] <= 1
 frac[19,32] <= 1
 frac[19,46] <= 1
 frac[19,55] <= 1
 frac[19,34] <= 1
 frac[19,38] <= 1
 frac[19,52] <= 1
 frac[19,49] <= 1
 frac[19,56] <= 1
 frac[20,75] <= 1
 frac[20,82] <= 1
 frac[20,85] <= 1
 frac[20,65] <= 1
 frac[20,63] <= 1
 frac[20,87] <= 1
 frac[20,74] <= 1
 frac[20,84] <= 1
 frac[20,83] <= 1
 frac[20,79] <= 1
 frac[20,62] <= 1
 frac[20,73] <= 1
 frac[21,119] <= 1
 frac[21,112] <= 1
 frac[21,90] <= 1
 frac[21,99] <= 1
 frac[21,97] <= 1
 frac[21,102] <= 1
 frac[21,104] <= 1
 frac[21,91] <= 1
 frac[21,100] <= 1
 frac[21,107] <= 1
 frac[21,116] <= 1
 frac[21,94] <= 1
 frac[21,92] <= 1
 frac[21,101] <= 1
 frac[22,19] <= 1
 frac[22,7] <= 1
 frac[22,11] <= 1
 frac[22,2] <= 1
 frac[22,5] <= 1
 frac[22,23] <= 1
 frac[22,9] <= 1
 frac[22,0] <= 1
 frac[22,27] <= 1
 frac[22,18] <= 1
 frac[22,15] <= 1
 frac[22,25] <= 1
 frac[22,10] <= 1
 frac[22,13] <= 1
 frac[22,4] <= 1
 frac[22,22] <= 1
 frac[22,3] <= 1
 frac[22,20] <= 1
 frac[22,14] <= 1
 frac[22,1] <= 1
 frac[22,24] <= 1
 frac[23,97] <= 1
 frac[23,119] <= 1
 frac[23,106] <= 1
 frac[23,113] <= 1
 frac[23,111] <= 1
 frac[23,100] <= 1
 frac[23,92] <= 1
 frac[23,116] <= 1
 frac[23,118] <= 1
 frac[23,108] <= 1
 frac[23,101] <= 1
 frac[23,90] <= 1
 frac[23,114] <= 1
 frac[24,75] <= 1
 frac[24,61] <= 1
 frac[24,82] <= 1
 frac[24,70] <= 1
 frac[24,77] <= 1
 frac[24,65] <= 1
 frac[24,72] <= 1
 frac[24,79] <= 1
 frac[24,62] <= 1
 frac[24,64] <= 1
 frac[24,71] <= 1
 frac[24,85] <= 1
 frac[25,3] <= 1
 frac[25,4] <= 1
 frac[25,1] <= 1
 frac[25,7] <= 1
 frac[25,26] <= 1
 frac[25,24] <= 1
 frac[25,2] <= 1
 frac[25,9] <= 1
 frac[26,47] <= 1
 frac[26,45] <= 1
 frac[26,32] <= 1
 frac[26,43] <= 1
 frac[26,46] <= 1
 frac[26,38] <= 1
 frac[27,68] <= 1
 frac[27,72] <= 1
 frac[27,62] <= 1
 frac[27,84] <= 1
 frac[27,74] <= 1
 frac[27,67] <= 1
 frac[27,86] <= 1
 frac[27,61] <= 1
 frac[27,71] <= 1
 frac[27,75] <= 1
 frac[27,79] <= 1
 frac[28,88] <= 1
 frac[28,63] <= 1
 frac[28,67] <= 1
 frac[28,70] <= 1
 frac[28,60] <= 1
 frac[28,83] <= 1
 frac[28,73] <= 1
 frac[28,64] <= 1
 frac[28,86] <= 1
 frac[28,62] <= 1
 frac[28,75] <= 1
 frac[28,71] <= 1
 frac[28,61] <= 1
 frac[28,74] <= 1
 frac[28,65] <= 1
 frac[29,50] <= 1
 frac[29,55] <= 1
 frac[29,48] <= 1
 frac[29,47] <= 1
 frac[29,58] <= 1
 frac[29,51] <= 1
 frac[29,30] <= 1
 frac[29,36] <= 1
 frac[29,34] <= 1
 frac[29,52] <= 1
 frac[29,46] <= 1
 frac[29,31] <= 1
 frac[29,57] <= 1
 frac[30,100] <= 1
 frac[30,111] <= 1
 frac[30,113] <= 1
 frac[30,106] <= 1
 frac[30,119] <= 1
 frac[30,95] <= 1
 frac[30,104] <= 1
 frac[30,114] <= 1
 frac[30,90] <= 1
 frac[30,93] <= 1
 frac[30,101] <= 1
 frac[30,96] <= 1
 frac[30,107] <= 1
 frac[30,116] <= 1
 frac[30,92] <= 1
 frac[30,115] <= 1
 frac[31,115] <= 1
 frac[31,116] <= 1
 frac[31,94] <= 1
 frac[31,111] <= 1
 frac[31,90] <= 1
 frac[31,105] <= 1
 frac[31,101] <= 1
 frac[31,98] <= 1
 frac[31,113] <= 1
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
 mem_tot_CPU9 <= 32768
 normalized_mem_tot_CPU9 <= 1
 normalized_rows_tot_CPU9 <= 1
 mem_tot_CPU10 <= 32768
 normalized_mem_tot_CPU10 <= 1
 normalized_rows_tot_CPU10 <= 1
 mem_tot_CPU11 <= 32768
 normalized_mem_tot_CPU11 <= 1
 normalized_rows_tot_CPU11 <= 1
 mem_tot_CPU12 <= 32768
 normalized_mem_tot_CPU12 <= 1
 normalized_rows_tot_CPU12 <= 1
 mem_tot_CPU13 <= 32768
 normalized_mem_tot_CPU13 <= 1
 normalized_rows_tot_CPU13 <= 1
 mem_tot_CPU14 <= 32768
 normalized_mem_tot_CPU14 <= 1
 normalized_rows_tot_CPU14 <= 1
 mem_tot_CPU15 <= 32768
 normalized_mem_tot_CPU15 <= 1
 normalized_rows_tot_CPU15 <= 1
 mem_tot_CPU16 <= 32768
 normalized_mem_tot_CPU16 <= 1
 normalized_rows_tot_CPU16 <= 1
 mem_tot_CPU17 <= 32768
 normalized_mem_tot_CPU17 <= 1
 normalized_rows_tot_CPU17 <= 1
 mem_tot_CPU18 <= 32768
 normalized_mem_tot_CPU18 <= 1
 normalized_rows_tot_CPU18 <= 1
 mem_tot_CPU19 <= 32768
 normalized_mem_tot_CPU19 <= 1
 normalized_rows_tot_CPU19 <= 1
 mem_tot_CPU20 <= 32768
 normalized_mem_tot_CPU20 <= 1
 normalized_rows_tot_CPU20 <= 1
 mem_tot_CPU21 <= 32768
 normalized_mem_tot_CPU21 <= 1
 normalized_rows_tot_CPU21 <= 1
 mem_tot_CPU22 <= 32768
 normalized_mem_tot_CPU22 <= 1
 normalized_rows_tot_CPU22 <= 1
 mem_tot_CPU23 <= 32768
 normalized_mem_tot_CPU23 <= 1
 normalized_rows_tot_CPU23 <= 1
 mem_tot_CPU24 <= 32768
 normalized_mem_tot_CPU24 <= 1
 normalized_rows_tot_CPU24 <= 1
 mem_tot_CPU25 <= 32768
 normalized_mem_tot_CPU25 <= 1
 normalized_rows_tot_CPU25 <= 1
 mem_tot_CPU26 <= 32768
 normalized_mem_tot_CPU26 <= 1
 normalized_rows_tot_CPU26 <= 1
 mem_tot_CPU27 <= 32768
 normalized_mem_tot_CPU27 <= 1
 normalized_rows_tot_CPU27 <= 1
 mem_tot_CPU28 <= 32768
 normalized_mem_tot_CPU28 <= 1
 normalized_rows_tot_CPU28 <= 1
 mem_tot_CPU29 <= 32768
 normalized_mem_tot_CPU29 <= 1
 normalized_rows_tot_CPU29 <= 1
 mem_tot_CPU30 <= 32768
 normalized_mem_tot_CPU30 <= 1
 normalized_rows_tot_CPU30 <= 1
 mem_tot_CPU31 <= 32768
 normalized_mem_tot_CPU31 <= 1
 normalized_rows_tot_CPU31 <= 1
 mem_tot_CPU32 <= 32768
 normalized_mem_tot_CPU32 <= 1
 normalized_rows_tot_CPU32 <= 1
 mem_tot_tor_P41 <= 6144
 normalized_mem_tot_tor_P41 <= 1
 normalized_rows_tot_tor_P41 <= 1
 mem_tot_tor_P42 <= 6144
 normalized_mem_tot_tor_P42 <= 1
 normalized_rows_tot_tor_P42 <= 1
 mem_tot_tor_P43 <= 6144
 normalized_mem_tot_tor_P43 <= 1
 normalized_rows_tot_tor_P43 <= 1
 mem_tot_tor_P44 <= 6144
 normalized_mem_tot_tor_P44 <= 1
 normalized_rows_tot_tor_P44 <= 1
 mem_tot_l1_P41 <= 6144
 normalized_mem_tot_l1_P41 <= 1
 normalized_rows_tot_l1_P41 <= 1
 mem_tot_l1_P42 <= 6144
 normalized_mem_tot_l1_P42 <= 1
 normalized_rows_tot_l1_P42 <= 1
 mem_tot_l2_P4 <= 6144
 normalized_mem_tot_l2_P4 <= 1
 normalized_rows_tot_l2_P4 <= 1
Binaries
 frac[32,54] frac[32,44] frac[32,113] frac[32,35] frac[32,25] frac[32,58]
 frac[32,91] frac[32,49] frac[32,23] frac[32,13] frac[32,82] frac[32,46]
 frac[32,4] frac[32,115] frac[32,37] frac[32,27] frac[32,96] frac[32,84]
 frac[32,32] frac[32,6] frac[32,117] frac[32,65] frac[32,39] frac[32,29]
 frac[32,98] frac[32,88] frac[32,62] frac[32,20] frac[32,10] frac[32,95]
 frac[32,43] frac[32,1] frac[32,112] frac[32,76] frac[32,24] frac[32,119]
 frac[32,67] frac[32,57] frac[32,31] frac[32,100] frac[32,90] frac[32,48]
 frac[32,22] frac[32,45] frac[32,3] frac[32,104] frac[32,26] frac[32,17]
 frac[32,102] frac[32,40] frac[32,83] frac[32,73] frac[32,5] frac[32,116]
 frac[32,106] frac[32,38] frac[32,97] frac[32,61] frac[32,19] frac[32,52]
 frac[32,42] frac[32,0] frac[32,33] frac[32,7] frac[32,118] frac[32,108]
 frac[32,56] frac[32,30] frac[32,99] frac[32,63] frac[32,11] frac[33,101]
 frac[33,95] frac[33,34] frac[33,74] frac[33,105] frac[33,67] frac[33,60]
 frac[33,22] frac[33,98] frac[33,68] frac[33,53] frac[33,47] frac[33,93]
 frac[33,32] frac[33,57] frac[33,117] frac[33,111] frac[33,65] frac[33,50]
 frac[33,83] frac[33,38] frac[33,24] frac[33,114] frac[33,84] frac[33,78]
 frac[33,63] frac[33,42] frac[33,35] frac[33,81] frac[33,75] frac[33,36]
 frac[33,76] frac[33,61] frac[33,54] frac[33,40] frac[33,100] frac[33,94]
 frac[33,33] frac[33,87] frac[33,73] frac[33,118] frac[33,104] frac[33,51]
 frac[33,97] frac[33,52] frac[33,46] frac[33,0] frac[33,92] frac[33,39]
 frac[33,115] frac[33,85] frac[33,79] frac[33,56] frac[33,116] frac[33,64]
 frac[33,49] frac[33,43] frac[33,44] frac[33,6] frac[33,82] frac[33,31]
 frac[33,107] frac[33,77] frac[33,62] frac[33,10] frac[33,70] frac[34,52]
 frac[34,46] frac[34,85] frac[34,79] frac[34,3] frac[34,114] frac[34,108]
 frac[34,32] frac[34,26] frac[34,65] frac[34,15] frac[34,50] frac[34,44]
 frac[34,6] frac[34,83] frac[34,1] frac[34,112] frac[34,106] frac[34,62]
 frac[34,24] frac[34,101] frac[34,57] frac[34,19] frac[34,13] frac[34,4]
 frac[34,119] frac[34,75] frac[34,37] frac[34,104] frac[34,22] frac[34,99]
 frac[34,55] frac[34,17] frac[34,11] frac[34,84] frac[34,40] frac[34,2]
 frac[34,111] frac[34,73] frac[34,29] frac[34,102] frac[34,58] frac[34,20]
 frac[34,14] frac[34,97] frac[34,91] frac[34,53] frac[34,47] frac[34,9]
 frac[34,82] frac[34,38] frac[34,0] frac[34,27] frac[34,100] frac[34,94]
 frac[34,56] frac[34,18] frac[34,7] frac[34,118] frac[34,74] frac[34,113]
 frac[34,107] frac[34,63] frac[34,25] frac[34,92] frac[34,54] frac[34,10]
 frac[34,87] frac[34,49] frac[34,43] frac[34,5] frac[34,116] frac[34,34]
 frac[34,23] frac[34,90] frac[35,88] frac[35,65] frac[35,46] frac[35,52]
 frac[35,26] frac[35,96] frac[35,70] frac[35,61] frac[35,3] frac[35,105]
 frac[35,79] frac[35,85] frac[35,34] frac[35,116] frac[35,90] frac[35,43]
 frac[35,67] frac[35,98] frac[35,72] frac[35,63] frac[35,107] frac[35,113]
 frac[35,36] frac[35,92] frac[35,45] frac[35,51] frac[35,30] frac[35,100]
 frac[35,74] frac[35,7] frac[35,115] frac[35,38] frac[35,94] frac[35,47]
 frac[35,71] frac[35,58] frac[35,82] frac[35,9] frac[35,111] frac[35,64]
 frac[35,55] frac[35,73] frac[35,60] frac[35,2] frac[35,104] frac[35,84]
 frac[35,119] frac[35,93] frac[35,48] frac[35,57] frac[35,31] frac[35,101]
 frac[35,75] frac[35,62] frac[35,4] frac[35,106] frac[35,86] frac[35,95]
 frac[35,50] frac[35,24] frac[35,68] frac[35,1] frac[35,77] frac[35,83]
 frac[35,32] frac[35,114] frac[36,68] frac[36,22] frac[36,101] frac[36,67]
 frac[36,29] frac[36,108] frac[36,74] frac[36,62] frac[36,4] frac[36,107]
 frac[36,113] frac[36,95] frac[36,37] frac[36,3] frac[36,70] frac[36,44]
 frac[36,10] frac[36,77] frac[36,43] frac[36,49] frac[36,31] frac[36,116]
 frac[36,82] frac[36,56] frac[36,6] frac[36,115] frac[36,39] frac[36,13]
 frac[36,64] frac[36,46] frac[36,52] frac[36,97] frac[36,79] frac[36,85]
 frac[36,51] frac[36,104] frac[36,118] frac[36,92] frac[36,58] frac[36,0]
 frac[36,91] frac[36,33] frac[36,100] frac[36,40] frac[36,54] frac[36,99]
 frac[36,73] frac[36,87] frac[36,61] frac[36,27] frac[36,106] frac[36,112]
 frac[36,94] frac[36,36] frac[36,35] frac[36,23] frac[36,102] frac[36,76]
 frac[36,42] frac[36,48] frac[36,30] frac[36,75] frac[36,63] frac[36,5]
 frac[36,114] frac[36,88] frac[36,38] frac[36,45] frac[36,11] frac[36,17]
 frac[36,96] frac[36,84] frac[36,50] frac[36,24] frac[36,111] frac[36,117]
 frac[36,83] frac[36,57] frac[36,7] frac[36,90] frac[36,32] frac[36,65]
 frac[36,47] frac[36,53] frac[36,19] frac[36,98] frac[36,60] frac[36,26]
 frac[36,105] frac[36,119] frac[36,93] frac[36,1] frac[37,82] frac[37,13]
 frac[37,23] frac[37,113] frac[37,91] frac[37,44] frac[37,54] frac[37,68]
 frac[37,63] frac[37,99] frac[37,77] frac[37,87] frac[37,56] frac[37,2]
 frac[37,108] frac[37,118] frac[37,11] frac[37,52] frac[37,30] frac[37,61]
 frac[37,7] frac[37,97] frac[37,75] frac[37,85] frac[37,38] frac[37,0]
 frac[37,106] frac[37,116] frac[37,94] frac[37,47] frac[37,9] frac[37,19]
 frac[37,40] frac[37,50] frac[37,102] frac[37,64] frac[37,5] frac[37,111]
 frac[37,73] frac[37,83] frac[37,36] frac[37,104] frac[37,114] frac[37,92]
 frac[37,45] frac[37,55] frac[37,17] frac[37,48] frac[37,26] frac[37,100]
 frac[37,57] frac[37,3] frac[37,119] frac[37,34] frac[37,22] frac[37,112]
 frac[37,90] frac[37,43] frac[37,53] frac[37,31] frac[37,62] frac[37,24]
 frac[37,98] frac[37,1] frac[37,107] frac[37,95] frac[37,32] frac[37,10]
 frac[37,88] frac[37,51] frac[37,29] frac[37,65] frac[37,60] frac[37,6]
 frac[37,96] frac[37,74] frac[37,84] frac[37,37] frac[37,105] frac[37,93]
 frac[37,46] frac[37,70] frac[37,49] frac[37,27] frac[37,101] frac[37,79]
 frac[37,58] frac[37,4] frac[38,40] frac[38,119] frac[38,73] frac[38,5]
 frac[38,31] frac[38,64] frac[38,90] frac[38,60] frac[38,22] frac[38,97]
 frac[38,93] frac[38,114] frac[38,84] frac[38,46] frac[38,0] frac[38,26]
 frac[38,79] frac[38,29] frac[38,108] frac[38,70] frac[38,88] frac[38,50]
 frac[38,83] frac[38,53] frac[38,112] frac[38,74] frac[38,44] frac[38,6]
 frac[38,24] frac[38,107] frac[38,77] frac[38,57] frac[38,19] frac[38,98]
 frac[38,68] frac[38,94] frac[38,48] frac[38,10] frac[38,101] frac[38,43]
 frac[38,13] frac[38,118] frac[38,4] frac[38,30] frac[38,105] frac[38,37]
 frac[38,17] frac[38,96] frac[38,92] frac[38,54] frac[38,87] frac[38,3]
 frac[38,116] frac[38,32] frac[38,58] frac[38,111] frac[38,91] frac[38,61]
 frac[38,23] frac[38,102] frac[38,82] frac[38,52] frac[38,85] frac[38,47]
 frac[38,1] frac[38,27] frac[38,106] frac[38,38] frac[38,56] frac[38,51]
 frac[38,100] frac[38,113] frac[38,75] frac[38,45] frac[38,7] frac[38,104]
 frac[38,36] frac[38,62] frac[38,99] frac[38,95] frac[38,49] frac[38,11]
General Constraints
 mem_overall_fixed_thr: max_mem_fixed_thr = MAX ( mem_tot_tor_P41 ,
   mem_tot_tor_P42 , mem_tot_tor_P43 , mem_tot_tor_P44 , mem_tot_l1_P41 ,
   mem_tot_l1_P42 , mem_tot_l2_P4 )
 mem_overall_others: max_mem_others = MAX ( normalized_mem_tot_CPU1 ,
   normalized_mem_tot_CPU2 , normalized_mem_tot_CPU3 ,
   normalized_mem_tot_CPU4 , normalized_mem_tot_CPU5 ,
   normalized_mem_tot_CPU6 , normalized_mem_tot_CPU7 ,
   normalized_mem_tot_CPU8 , normalized_mem_tot_CPU9 ,
   normalized_mem_tot_CPU10 , normalized_mem_tot_CPU11 ,
   normalized_mem_tot_CPU12 , normalized_mem_tot_CPU13 ,
   normalized_mem_tot_CPU14 , normalized_mem_tot_CPU15 ,
   normalized_mem_tot_CPU16 , normalized_mem_tot_CPU17 ,
   normalized_mem_tot_CPU18 , normalized_mem_tot_CPU19 ,
   normalized_mem_tot_CPU20 , normalized_mem_tot_CPU21 ,
   normalized_mem_tot_CPU22 , normalized_mem_tot_CPU23 ,
   normalized_mem_tot_CPU24 , normalized_mem_tot_CPU25 ,
   normalized_mem_tot_CPU26 , normalized_mem_tot_CPU27 ,
   normalized_mem_tot_CPU28 , normalized_mem_tot_CPU29 ,
   normalized_mem_tot_CPU30 , normalized_mem_tot_CPU31 ,
   normalized_mem_tot_CPU32 )
 rows_overall_fixed_thr: max_rows_fixed_thr = MAX ( rows_tot_tor_P41 ,
   rows_tot_tor_P42 , rows_tot_tor_P43 , rows_tot_tor_P44 ,
   rows_tot_l1_P41 , rows_tot_l1_P42 , rows_tot_l2_P4 )
 rows_overall_others: max_rows_others = MAX ( normalized_rows_tot_CPU1 ,
   normalized_rows_tot_CPU2 , normalized_rows_tot_CPU3 ,
   normalized_rows_tot_CPU4 , normalized_rows_tot_CPU5 ,
   normalized_rows_tot_CPU6 , normalized_rows_tot_CPU7 ,
   normalized_rows_tot_CPU8 , normalized_rows_tot_CPU9 ,
   normalized_rows_tot_CPU10 , normalized_rows_tot_CPU11 ,
   normalized_rows_tot_CPU12 , normalized_rows_tot_CPU13 ,
   normalized_rows_tot_CPU14 , normalized_rows_tot_CPU15 ,
   normalized_rows_tot_CPU16 , normalized_rows_tot_CPU17 ,
   normalized_rows_tot_CPU18 , normalized_rows_tot_CPU19 ,
   normalized_rows_tot_CPU20 , normalized_rows_tot_CPU21 ,
   normalized_rows_tot_CPU22 , normalized_rows_tot_CPU23 ,
   normalized_rows_tot_CPU24 , normalized_rows_tot_CPU25 ,
   normalized_rows_tot_CPU26 , normalized_rows_tot_CPU27 ,
   normalized_rows_tot_CPU28 , normalized_rows_tot_CPU29 ,
   normalized_rows_tot_CPU30 , normalized_rows_tot_CPU31 ,
   normalized_rows_tot_CPU32 )
End

\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_P4 + 0 max_mem_CPU_Cluster + 0 max_rows_P4
   + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU4 + 0 m_access_time_CPU6
   + 0 m_access_time_CPU10 + 0 m_access_time_CPU11 + 0 m_access_time_CPU13
   + 0 m_access_time_CPU21 + 0 m_access_time_CPU27 + 0 m_access_time_CPU29
   + res
Subject To
 cov_0_4: frac[10,4] + frac[8,4] >= 0.7
 cov_1_0: frac[9,0] + frac[8,0] + frac[10,0] >= 1
 cov_2_37: frac[10,37] + frac[9,37] + frac[8,37] >= 0.9
 cov_3_21: frac[8,21] >= 0.8
 cov_4_10: frac[10,10] + frac[8,10] + frac[9,10] >= 0.7
 cov_5_36: frac[10,36] + frac[8,36] >= 0.7
 cov_6_37: frac[10,37] + frac[9,37] >= 0.7
 cov_7_32: frac[8,32] >= 0.4115185329775536
 cov_8_38: frac[8,38] >= 0.9
 cov_9_15: frac[8,15] + frac[10,15] + frac[9,15] >= 0.9
 cov_10_18: frac[8,18] >= 0.8
 cov_11_15: frac[10,15] + frac[9,15] >= 0.7
 cov_12_4: frac[10,4] + frac[8,4] >= 0.7
 cov_13_14: frac[9,14] + frac[10,14] + frac[8,14] >= 1
 cov_14_26: frac[8,26] >= 0.1
 cov_15_25: frac[8,25] + frac[10,25] >= 0.4757322170224461
 cov_16_30: frac[9,30] + frac[10,30] >= 0.5623037065955098
 cov_17_4: frac[9,4] + frac[10,4] + frac[8,4] >= 0.7
 cov_18_26: frac[10,26] + frac[8,26] >= 0.1
 cov_19_5: frac[8,5] + frac[10,5] + frac[9,5] >= 0.8
 cov_20_22: frac[8,22] + frac[10,22] >= 0.7
 cov_21_19: frac[9,19] + frac[10,19] + frac[8,19] >= 0.8
 cov_22_15: frac[8,15] + frac[10,15] >= 0.7
 cov_23_22: frac[8,22] + frac[9,22] + frac[10,22] >= 0.7
 cov_24_25: frac[8,25] >= 0.4757322170224461
 cov_25_15: frac[10,15] + frac[9,15] >= 0.9
 cov_26_5: frac[10,5] + frac[9,5] >= 0.8
 cov_27_26: frac[10,26] + frac[8,26] >= 0.1
 cov_28_35: frac[10,35] + frac[8,35] >= 1
 cov_29_10: frac[10,10] + frac[8,10] >= 0.7
 cov_30_25: frac[8,25] + frac[10,25] >= 0.4757322170224461
 cov_31_35: frac[10,35] + frac[8,35] >= 0.8
 cov_32_25: frac[9,25] + frac[10,25] >= 0.0252866736179556
 cov_33_33: frac[9,33] + frac[8,33] + frac[10,33] >= 0.9
 cov_34_36: frac[10,36] + frac[8,36] >= 0.7
 cov_35_47: frac[10,47] + frac[9,47] >= 0.2
 cov_36_57: frac[8,57] + frac[10,57] >= 0.8
 cov_37_79: frac[9,79] >= 0.8872507499999999
 cov_38_57: frac[9,57] + frac[8,57] + frac[10,57] >= 0.8936253749999998
 cov_39_75: frac[10,75] + frac[9,75] >= 0.9
 cov_40_79: frac[9,79] + frac[10,79] >= 0.8872507499999999
 cov_41_48: frac[10,48] + frac[9,48] >= 0.9
 cov_42_74: frac[10,74] + frac[8,74] >= 0.993625375
 cov_43_42: frac[10,42] + frac[8,42] + frac[9,42] >= 0.4
 cov_44_48: frac[10,48] + frac[9,48] >= 0.7
 cov_45_42: frac[10,42] + frac[9,42] >= 0.4
 cov_46_48: frac[8,48] + frac[10,48] >= 0.9
 cov_47_48: frac[8,48] + frac[10,48] >= 1
 cov_48_79: frac[9,79] >= 0.8872507499999999
 cov_49_90: frac[8,90] + frac[0,90] + frac[4,90] + frac[9,90] + frac[10,90]
   >= 0.9
 cov_50_98: frac[6,98] + frac[10,98] + frac[4,98] + frac[9,98] >= 0.7
 cov_51_103: frac[2,103] + frac[9,103] + frac[3,103] >= 1
 cov_52_116: frac[8,116] + frac[6,116] + frac[0,116] + frac[10,116] >= 1
 cov_53_118: frac[10,118] + frac[0,118] + frac[9,118] + frac[8,118] >= 0.8
 cov_54_88: frac[9,88] + frac[10,88] + frac[3,88] + frac[5,88] >= 0.1
 cov_55_89: frac[0,89] + frac[1,89] + frac[8,89] >= 0.9
 cov_56_97: frac[8,97] + frac[3,97] + frac[10,97] + frac[9,97] + frac[1,97]
   >= 0.9
 cov_57_93: frac[0,93] + frac[1,93] + frac[8,93] >= 0.8
 cov_58_91: frac[7,91] + frac[5,91] >= 0.9
 cov_59_104: frac[4,104] + frac[8,104] + frac[10,104] + frac[9,104]
   + frac[0,104] >= 0.7
 cov_60_93: frac[9,93] + frac[3,93] + frac[10,93] + frac[8,93] >= 0.9
 cov_61_115: frac[4,115] + frac[9,115] + frac[3,115] >= 0.7
 cov_62_109: frac[4,109] + frac[3,109] + frac[9,109] >= 0.8
 cov_63_103: frac[1,103] + frac[0,103] + frac[8,103] >= 1
 cov_64_84: frac[9,84] + frac[1,84] + frac[8,84] + frac[4,84] + frac[10,84]
   >= 1
 cov_65_114: frac[8,114] + frac[1,114] + frac[0,114] >= 0.7
 cov_66_94: frac[8,94] + frac[4,94] + frac[10,94] + frac[9,94] >= 0.7
 cov_67_104: frac[7,104] + frac[2,104] + frac[10,104] + frac[9,104] >= 0.9
 cov_68_105: frac[4,105] + frac[10,105] + frac[5,105] + frac[9,105] >= 0.7
 cov_69_110: frac[8,110] + frac[5,110] + frac[1,110] + frac[10,110] >= 0.8
 cov_70_93: frac[0,93] + frac[1,93] + frac[8,93] >= 0.8
 cov_71_103: frac[2,103] + frac[9,103] + frac[3,103] >= 0.7
 cov_72_81: frac[8,81] + frac[10,81] + frac[9,81] + frac[1,81] + frac[3,81]
   >= 0.9
 cov_73_112: frac[0,112] + frac[7,112] + frac[8,112] + frac[10,112] >= 0.7
 cov_74_97: frac[3,97] + frac[9,97] + frac[2,97] >= 1
 cov_75_85: frac[3,85] + frac[4,85] + frac[9,85] >= 0.7
 cov_76_99: frac[0,99] + frac[10,99] + frac[5,99] + frac[8,99] >= 0.9
 cov_77_115: frac[2,115] + frac[0,115] + frac[10,115] + frac[9,115]
   + frac[8,115] >= 0.9
 cov_78_95: frac[8,95] + frac[1,95] + frac[0,95] >= 0.7
 cov_79_110: frac[7,110] + frac[6,110] >= 0.9
 cov_80_93: frac[9,93] + frac[0,93] + frac[3,93] + frac[10,93] + frac[8,93]
   >= 1
 cov_81_116: frac[2,116] + frac[5,116] + frac[10,116] + frac[9,116] >= 0.7
 cov_82_105: frac[8,105] + frac[3,105] + frac[1,105] + frac[10,105]
   + frac[9,105] >= 0.7
 cov_83_116: frac[8,116] + frac[2,116] + frac[0,116] + frac[10,116]
   + frac[9,116] >= 0.7
 cov_84_84: frac[1,84] + frac[8,84] + frac[0,84] >= 0.8
 cov_85_116: frac[8,116] + frac[1,116] + frac[10,116] >= 0.9
 cov_86_98: frac[0,98] + frac[10,98] + frac[8,98] + frac[9,98] + frac[3,98]
   >= 0.8
 cov_87_115: frac[0,115] + frac[10,115] + frac[4,115] + frac[9,115]
   + frac[8,115] >= 0.9
 cov_88_99: frac[6,99] + frac[5,99] >= 0.5063746250000001
 cov_89_104: frac[2,104] + frac[10,104] + frac[6,104] + frac[9,104] >= 0.7
 cov_90_99: frac[10,99] + frac[9,99] + frac[3,99] + frac[7,99]
   >= 0.206374625
 cov_91_92: frac[10,92] + frac[6,92] + frac[9,92] + frac[2,92]
   >= 0.0126214461685817
 cov_92_119: frac[10,119] + frac[7,119] + frac[8,119] >= 0.7
 cov_93_85: frac[3,85] + frac[10,85] + frac[9,85] + frac[7,85] >= 0.9
 cov_94_116: frac[8,116] + frac[6,116] + frac[1,116] + frac[10,116] >= 1
 cov_95_88: frac[5,88] + frac[7,88] >= 0.1
 cov_96_118: frac[0,118] + frac[8,118] + frac[1,118] >= 0.9
 cov_97_119: frac[10,119] + frac[9,119] + frac[2,119] >= 0.9
 cov_98_112: frac[0,112] + frac[9,112] + frac[4,112] + frac[8,112]
   + frac[10,112] >= 0.7
 cov_99_95: frac[10,95] + frac[6,95] + frac[3,95] + frac[9,95] >= 0.2
 cov_100_110: frac[8,110] + frac[2,110] + frac[1,110] + frac[10,110]
   + frac[9,110] >= 0.9
 cov_101_99: frac[6,99] + frac[7,99] >= 0.3063746250000001
 cov_102_95: frac[10,95] + frac[6,95] + frac[9,95] + frac[4,95] >= 0.2
 cov_103_109: frac[4,109] + frac[10,109] + frac[0,109] + frac[8,109]
   + frac[9,109] >= 0.9
 cov_104_86: frac[3,86] + frac[9,86] + frac[4,86] >= 0.9
 cov_105_81: frac[8,81] + frac[10,81] + frac[9,81] + frac[1,81]
   + frac[3,81] >= 0.8
 cov_106_97: frac[10,97] + frac[9,97] + frac[2,97] >= 0.9
 cov_107_96: frac[5,96] + frac[7,96] >= 0.3127492500000001
 cov_108_102: frac[4,102] + frac[9,102] + frac[5,102] + frac[10,102]
   >= 0.7
 cov_109_95: frac[10,95] + frac[7,95] + frac[2,95] + frac[9,95] >= 0.2
 cov_110_159: frac[10,159] + frac[8,159] >= 0.7
 cov_111_140: frac[8,140] + frac[10,140] >= 0.8
 cov_112_127: frac[9,127] + frac[8,127] + frac[10,127] >= 0.8
 cov_113_133: frac[8,133] + frac[10,133] >= 1
 cov_114_158: frac[10,158] + frac[9,158] >= 0.9
 cov_115_141: frac[9,141] + frac[8,141] + frac[10,141] >= 1
 cov_116_126: frac[10,126] + frac[9,126] >= 0.5936253749999999
 cov_117_121: frac[10,121] + frac[9,121] >= 0.9
 cov_118_157: frac[8,157] + frac[10,157] >= 0.7
 cov_119_134: frac[9,134] + frac[10,134] + frac[8,134] >= 0.7
 cov_120_141: frac[9,141] + frac[10,141] >= 0.7
 cov_121_152: frac[9,152] + frac[10,152] >= 0.3
 cov_122_145: frac[9,145] + frac[10,145] >= 0.1
 cov_123_127: frac[9,127] + frac[8,127] + frac[10,127] >= 0.8
 cov_124_147: frac[10,147] + frac[9,147] + frac[8,147] >= 0.7
 cov_125_149: frac[8,149] >= 1
 cov_126_140: frac[8,140] + frac[10,140] >= 0.8
 cov_127_157: frac[8,157] + frac[10,157] >= 0.7
 cov_128_147: frac[10,147] + frac[8,147] >= 0.7
 cov_129_126: frac[10,126] + frac[9,126] >= 0.5936253749999999
 cov_130_141: frac[9,141] + frac[10,141] >= 1
 cov_131_141: frac[9,141] + frac[10,141] >= 1
 cov_132_125: frac[10,125] + frac[8,125] + frac[9,125] >= 0.7
 cov_133_159: frac[8,159] + frac[1,159] >= 0.8
 cov_134_133: frac[8,133] + frac[10,133] + frac[9,133] >= 1
 cov_135_126: frac[8,126] >= 0.1936253749999999
 cov_136_147: frac[10,147] + frac[8,147] >= 0.7
 cov_137_121: frac[8,121] + frac[10,121] + frac[9,121] >= 0.9
 cov_138_140: frac[8,140] + frac[9,140] + frac[10,140] >= 0.8
 cov_139_141: frac[8,141] + frac[10,141] >= 0.9
 cov_140_133: frac[8,133] + frac[10,133] + frac[9,133] >= 1
 cov_141_158: frac[10,158] + frac[8,158] >= 0.8
 cov_142_138: frac[8,138] >= 0.1936253749999999
 cov_143_134: frac[9,134] + frac[10,134] >= 1
 cov_144_135: frac[9,135] + frac[10,135] + frac[8,135]
   >= 0.5878022065955109
 cov_145_152: frac[9,152] + frac[10,152] >= 0.3
 cov_146_140: frac[8,140] + frac[9,140] + frac[10,140] >= 0.7
 cov_147_158: frac[10,158] + frac[8,158] + frac[9,158] >= 0.9
 cov_148_127: frac[9,127] + frac[10,127] >= 0.8
 cov_149_159: frac[10,159] + frac[8,159] >= 0.8
 cov_150_159: frac[8,159] >= 0.8
 cov_151_152: frac[9,152] + frac[10,152] >= 0.3
 cov_152_157: frac[9,157] >= 0.8
 cov_153_133: frac[8,133] + frac[10,133] >= 0.8
 cov_154_155: frac[9,155] + frac[10,155] >= 0.00582316840448909
 cov_155_128: frac[10,128] + frac[9,128] >= 0.5936253749999999
 accuracy_2_103: - 5.09765625 frac[2,103] + mem[2,103] = 0
 accuracy_8_110: - 5.09765625 frac[8,110] + mem[8,110] = 0
 accuracy_10_95: - 5.09765625 frac[10,95] + mem[10,95] = 0
 accuracy_4_102: - 5.09765625 frac[4,102] + mem[4,102] = 0
 accuracy_9_84: - 5.09765625 frac[9,84] + mem[9,84] = 0
 accuracy_0_89: - 5.09765625 frac[0,89] + mem[0,89] = 0
 accuracy_8_97: - 5.09765625 frac[8,97] + mem[8,97] = 0
 accuracy_3_97: - 5.09765625 frac[3,97] + mem[3,97] = 0
 accuracy_8_116: - 5.09765625 frac[8,116] + mem[8,116] = 0
 accuracy_5_110: - 5.09765625 frac[5,110] + mem[5,110] = 0
 accuracy_6_116: - 5.09765625 frac[6,116] + mem[6,116] = 0
 accuracy_7_95: - 5.09765625 frac[7,95] + mem[7,95] = 0
 accuracy_10_118: - 5.09765625 frac[10,118] + mem[10,118] = 0
 accuracy_8_5: - 5.09765625 frac[8,5] + mem[8,5] = 0
 accuracy_8_90: - 5.09765625 frac[8,90] + mem[8,90] = 0
 accuracy_7_104: - 5.09765625 frac[7,104] + mem[7,104] = 0
 accuracy_0_98: - 5.09765625 frac[0,98] + mem[0,98] = 0
 accuracy_9_0: - 5.09765625 frac[9,0] + mem[9,0] = 0
 accuracy_10_158: - 5.09765625 frac[10,158] + mem[10,158] = 0
 accuracy_3_86: - 5.09765625 frac[3,86] + mem[3,86] = 0
 accuracy_4_104: - 5.09765625 frac[4,104] + mem[4,104] = 0
 accuracy_2_104: - 5.09765625 frac[2,104] + mem[2,104] = 0
 accuracy_6_95: - 5.09765625 frac[6,95] + mem[6,95] = 0
 accuracy_8_126: - 5.09765625 frac[8,126] + mem[8,126] = 0
 accuracy_6_98: - 5.09765625 frac[6,98] + mem[6,98] = 0
 accuracy_10_97: - 5.09765625 frac[10,97] + mem[10,97] = 0
 accuracy_10_35: - 5.09765625 frac[10,35] + mem[10,35] = 0
 accuracy_9_57: - 5.09765625 frac[9,57] + mem[9,57] = 0
 accuracy_8_15: - 5.09765625 frac[8,15] + mem[8,15] = 0
 accuracy_9_102: - 5.09765625 frac[9,102] + mem[9,102] = 0
 accuracy_9_135: - 5.09765625 frac[9,135] + mem[9,135] = 0
 accuracy_9_14: - 5.09765625 frac[9,14] + mem[9,14] = 0
 accuracy_10_75: - 5.09765625 frac[10,75] + mem[10,75] = 0
 accuracy_2_115: - 5.09765625 frac[2,115] + mem[2,115] = 0
 accuracy_8_21: - 5.09765625 frac[8,21] + mem[8,21] = 0
 accuracy_10_147: - 5.09765625 frac[10,147] + mem[10,147] = 0
 accuracy_3_85: - 5.09765625 frac[3,85] + mem[3,85] = 0
 accuracy_4_109: - 5.09765625 frac[4,109] + mem[4,109] = 0
 accuracy_9_93: - 5.09765625 frac[9,93] + mem[9,93] = 0
 accuracy_10_85: - 5.09765625 frac[10,85] + mem[10,85] = 0
 accuracy_8_104: - 5.09765625 frac[8,104] + mem[8,104] = 0
 accuracy_8_158: - 5.09765625 frac[8,158] + mem[8,158] = 0
 accuracy_8_38: - 5.09765625 frac[8,38] + mem[8,38] = 0
 accuracy_1_84: - 5.09765625 frac[1,84] + mem[1,84] = 0
 accuracy_10_98: - 5.09765625 frac[10,98] + mem[10,98] = 0
 accuracy_8_57: - 5.09765625 frac[8,57] + mem[8,57] = 0
 accuracy_0_118: - 5.09765625 frac[0,118] + mem[0,118] = 0
 accuracy_9_141: - 5.09765625 frac[9,141] + mem[9,141] = 0
 accuracy_10_36: - 5.09765625 frac[10,36] + mem[10,36] = 0
 accuracy_1_89: - 5.09765625 frac[1,89] + mem[1,89] = 0
 accuracy_9_97: - 5.09765625 frac[9,97] + mem[9,97] = 0
 accuracy_9_118: - 5.09765625 frac[9,118] + mem[9,118] = 0
 accuracy_2_116: - 5.09765625 frac[2,116] + mem[2,116] = 0
 accuracy_8_84: - 5.09765625 frac[8,84] + mem[8,84] = 0
 accuracy_3_95: - 5.09765625 frac[3,95] + mem[3,95] = 0
 accuracy_10_14: - 5.09765625 frac[10,14] + mem[10,14] = 0
 accuracy_9_30: - 5.09765625 frac[9,30] + mem[9,30] = 0
 accuracy_8_18: - 5.09765625 frac[8,18] + mem[8,18] = 0
 accuracy_9_75: - 5.09765625 frac[9,75] + mem[9,75] = 0
 accuracy_4_98: - 5.09765625 frac[4,98] + mem[4,98] = 0
 accuracy_9_88: - 5.09765625 frac[9,88] + mem[9,88] = 0
 accuracy_9_19: - 5.09765625 frac[9,19] + mem[9,19] = 0
 accuracy_0_93: - 5.09765625 frac[0,93] + mem[0,93] = 0
 accuracy_2_110: - 5.09765625 frac[2,110] + mem[2,110] = 0
 accuracy_10_104: - 5.09765625 frac[10,104] + mem[10,104] = 0
 accuracy_8_35: - 5.09765625 frac[8,35] + mem[8,35] = 0
 accuracy_6_104: - 5.09765625 frac[6,104] + mem[6,104] = 0
 accuracy_10_42: - 5.09765625 frac[10,42] + mem[10,42] = 0
 accuracy_2_95: - 5.09765625 frac[2,95] + mem[2,95] = 0
 accuracy_7_91: - 5.09765625 frac[7,91] + mem[7,91] = 0
 accuracy_0_115: - 5.09765625 frac[0,115] + mem[0,115] = 0
 accuracy_10_57: - 5.09765625 frac[10,57] + mem[10,57] = 0
 accuracy_8_94: - 5.09765625 frac[8,94] + mem[8,94] = 0
 accuracy_5_116: - 5.09765625 frac[5,116] + mem[5,116] = 0
 accuracy_9_157: - 5.09765625 frac[9,157] + mem[9,157] = 0
 accuracy_9_4: - 5.09765625 frac[9,4] + mem[9,4] = 0
 accuracy_8_81: - 5.09765625 frac[8,81] + mem[8,81] = 0
 accuracy_4_84: - 5.09765625 frac[4,84] + mem[4,84] = 0
 accuracy_8_133: - 5.09765625 frac[8,133] + mem[8,133] = 0
 accuracy_9_25: - 5.09765625 frac[9,25] + mem[9,25] = 0
 accuracy_10_92: - 5.09765625 frac[10,92] + mem[10,92] = 0
 accuracy_10_30: - 5.09765625 frac[10,30] + mem[10,30] = 0
 accuracy_0_90: - 5.09765625 frac[0,90] + mem[0,90] = 0
 accuracy_8_98: - 5.09765625 frac[8,98] + mem[8,98] = 0
 accuracy_10_109: - 5.09765625 frac[10,109] + mem[10,109] = 0
 accuracy_8_32: - 5.09765625 frac[8,32] + mem[8,32] = 0
 accuracy_10_47: - 5.09765625 frac[10,47] + mem[10,47] = 0
 accuracy_0_112: - 5.09765625 frac[0,112] + mem[0,112] = 0
 accuracy_9_127: - 5.09765625 frac[9,127] + mem[9,127] = 0
 accuracy_10_119: - 5.09765625 frac[10,119] + mem[10,119] = 0
 accuracy_9_147: - 5.09765625 frac[9,147] + mem[9,147] = 0
 accuracy_0_99: - 5.09765625 frac[0,99] + mem[0,99] = 0
 accuracy_9_152: - 5.09765625 frac[9,152] + mem[9,152] = 0
 accuracy_10_128: - 5.09765625 frac[10,128] + mem[10,128] = 0
 accuracy_4_94: - 5.09765625 frac[4,94] + mem[4,94] = 0
 accuracy_1_116: - 5.09765625 frac[1,116] + mem[1,116] = 0
 accuracy_8_25: - 5.09765625 frac[8,25] + mem[8,25] = 0
 accuracy_10_159: - 5.09765625 frac[10,159] + mem[10,159] = 0
 accuracy_4_105: - 5.09765625 frac[4,105] + mem[4,105] = 0
 accuracy_10_4: - 5.09765625 frac[10,4] + mem[10,4] = 0
 accuracy_10_81: - 5.09765625 frac[10,81] + mem[10,81] = 0
 accuracy_0_84: - 5.09765625 frac[0,84] + mem[0,84] = 0
 accuracy_6_92: - 5.09765625 frac[6,92] + mem[6,92] = 0
 accuracy_10_19: - 5.09765625 frac[10,19] + mem[10,19] = 0
 accuracy_8_42: - 5.09765625 frac[8,42] + mem[8,42] = 0
 accuracy_1_110: - 5.09765625 frac[1,110] + mem[1,110] = 0
 accuracy_9_86: - 5.09765625 frac[9,86] + mem[9,86] = 0
 accuracy_6_99: - 5.09765625 frac[6,99] + mem[6,99] = 0
 accuracy_8_127: - 5.09765625 frac[8,127] + mem[8,127] = 0
 accuracy_8_149: - 5.09765625 frac[8,149] + mem[8,149] = 0
 accuracy_10_110: - 5.09765625 frac[10,110] + mem[10,110] = 0
 accuracy_8_114: - 5.09765625 frac[8,114] + mem[8,114] = 0
 accuracy_5_96: - 5.09765625 frac[5,96] + mem[5,96] = 0
 accuracy_1_93: - 5.09765625 frac[1,93] + mem[1,93] = 0
 accuracy_10_125: - 5.09765625 frac[10,125] + mem[10,125] = 0
 accuracy_8_48: - 5.09765625 frac[8,48] + mem[8,48] = 0
 accuracy_9_134: - 5.09765625 frac[9,134] + mem[9,134] = 0
 accuracy_7_110: - 5.09765625 frac[7,110] + mem[7,110] = 0
 accuracy_10_133: - 5.09765625 frac[10,133] + mem[10,133] = 0
 accuracy_9_155: - 5.09765625 frac[9,155] + mem[9,155] = 0
 accuracy_10_10: - 5.09765625 frac[10,10] + mem[10,10] = 0
 accuracy_9_79: - 5.09765625 frac[9,79] + mem[9,79] = 0
 accuracy_8_22: - 5.09765625 frac[8,22] + mem[8,22] = 0
 accuracy_8_140: - 5.09765625 frac[8,140] + mem[8,140] = 0
 accuracy_10_25: - 5.09765625 frac[10,25] + mem[10,25] = 0
 accuracy_9_92: - 5.09765625 frac[9,92] + mem[9,92] = 0
 accuracy_8_105: - 5.09765625 frac[8,105] + mem[8,105] = 0
 accuracy_8_159: - 5.09765625 frac[8,159] + mem[8,159] = 0
 accuracy_3_105: - 5.09765625 frac[3,105] + mem[3,105] = 0
 accuracy_9_81: - 5.09765625 frac[9,81] + mem[9,81] = 0
 accuracy_1_105: - 5.09765625 frac[1,105] + mem[1,105] = 0
 accuracy_10_99: - 5.09765625 frac[10,99] + mem[10,99] = 0
 accuracy_9_140: - 5.09765625 frac[9,140] + mem[9,140] = 0
 accuracy_5_99: - 5.09765625 frac[5,99] + mem[5,99] = 0
 accuracy_10_37: - 5.09765625 frac[10,37] + mem[10,37] = 0
 accuracy_10_126: - 5.09765625 frac[10,126] + mem[10,126] = 0
 accuracy_7_96: - 5.09765625 frac[7,96] + mem[7,96] = 0
 accuracy_10_48: - 5.09765625 frac[10,48] + mem[10,48] = 0
 accuracy_8_0: - 5.09765625 frac[8,0] + mem[8,0] = 0
 accuracy_0_109: - 5.09765625 frac[0,109] + mem[0,109] = 0
 accuracy_10_134: - 5.09765625 frac[10,134] + mem[10,134] = 0
 accuracy_10_15: - 5.09765625 frac[10,15] + mem[10,15] = 0
 accuracy_10_88: - 5.09765625 frac[10,88] + mem[10,88] = 0
 accuracy_8_19: - 5.09765625 frac[8,19] + mem[8,19] = 0
 accuracy_1_114: - 5.09765625 frac[1,114] + mem[1,114] = 0
 accuracy_10_26: - 5.09765625 frac[10,26] + mem[10,26] = 0
 accuracy_1_103: - 5.09765625 frac[1,103] + mem[1,103] = 0
 accuracy_9_95: - 5.09765625 frac[9,95] + mem[9,95] = 0
 accuracy_10_105: - 5.09765625 frac[10,105] + mem[10,105] = 0
 accuracy_8_36: - 5.09765625 frac[8,36] + mem[8,36] = 0
 accuracy_8_121: - 5.09765625 frac[8,121] + mem[8,121] = 0
 accuracy_5_105: - 5.09765625 frac[5,105] + mem[5,105] = 0
 accuracy_9_110: - 5.09765625 frac[9,110] + mem[9,110] = 0
 accuracy_9_33: - 5.09765625 frac[9,33] + mem[9,33] = 0
 accuracy_2_92: - 5.09765625 frac[2,92] + mem[2,92] = 0
 accuracy_0_116: - 5.09765625 frac[0,116] + mem[0,116] = 0
 accuracy_5_102: - 5.09765625 frac[5,102] + mem[5,102] = 0
 accuracy_9_99: - 5.09765625 frac[9,99] + mem[9,99] = 0
 accuracy_10_115: - 5.09765625 frac[10,115] + mem[10,115] = 0
 accuracy_8_10: - 5.09765625 frac[8,10] + mem[8,10] = 0
 accuracy_0_103: - 5.09765625 frac[0,103] + mem[0,103] = 0
 accuracy_8_95: - 5.09765625 frac[8,95] + mem[8,95] = 0
 accuracy_10_140: - 5.09765625 frac[10,140] + mem[10,140] = 0
 accuracy_4_90: - 5.09765625 frac[4,90] + mem[4,90] = 0
 accuracy_9_112: - 5.09765625 frac[9,112] + mem[9,112] = 0
 accuracy_7_112: - 5.09765625 frac[7,112] + mem[7,112] = 0
 accuracy_9_145: - 5.09765625 frac[9,145] + mem[9,145] = 0
 accuracy_10_155: - 5.09765625 frac[10,155] + mem[10,155] = 0
 accuracy_3_93: - 5.09765625 frac[3,93] + mem[3,93] = 0
 accuracy_4_85: - 5.09765625 frac[4,85] + mem[4,85] = 0
 accuracy_10_0: - 5.09765625 frac[10,0] + mem[10,0] = 0
 accuracy_10_93: - 5.09765625 frac[10,93] + mem[10,93] = 0
 accuracy_8_134: - 5.09765625 frac[8,134] + mem[8,134] = 0
 accuracy_9_90: - 5.09765625 frac[9,90] + mem[9,90] = 0
 accuracy_8_99: - 5.09765625 frac[8,99] + mem[8,99] = 0
 accuracy_3_99: - 5.09765625 frac[3,99] + mem[3,99] = 0
 accuracy_4_115: - 5.09765625 frac[4,115] + mem[4,115] = 0
 accuracy_9_47: - 5.09765625 frac[9,47] + mem[9,47] = 0
 accuracy_8_33: - 5.09765625 frac[8,33] + mem[8,33] = 0
 accuracy_8_118: - 5.09765625 frac[8,118] + mem[8,118] = 0
 accuracy_9_105: - 5.09765625 frac[9,105] + mem[9,105] = 0
 accuracy_1_81: - 5.09765625 frac[1,81] + mem[1,81] = 0
 accuracy_10_121: - 5.09765625 frac[10,121] + mem[10,121] = 0
 accuracy_9_126: - 5.09765625 frac[9,126] + mem[9,126] = 0
 accuracy_10_116: - 5.09765625 frac[10,116] + mem[10,116] = 0
 accuracy_4_95: - 5.09765625 frac[4,95] + mem[4,95] = 0
 accuracy_9_115: - 5.09765625 frac[9,115] + mem[9,115] = 0
 accuracy_8_26: - 5.09765625 frac[8,26] + mem[8,26] = 0
 accuracy_7_119: - 5.09765625 frac[7,119] + mem[7,119] = 0
 accuracy_3_88: - 5.09765625 frac[3,88] + mem[3,88] = 0
 accuracy_10_5: - 5.09765625 frac[10,5] + mem[10,5] = 0
 accuracy_10_94: - 5.09765625 frac[10,94] + mem[10,94] = 0
 accuracy_8_109: - 5.09765625 frac[8,109] + mem[8,109] = 0
 accuracy_3_109: - 5.09765625 frac[3,109] + mem[3,109] = 0
 accuracy_9_85: - 5.09765625 frac[9,85] + mem[9,85] = 0
 accuracy_4_112: - 5.09765625 frac[4,112] + mem[4,112] = 0
 accuracy_9_42: - 5.09765625 frac[9,42] + mem[9,42] = 0
 accuracy_8_115: - 5.09765625 frac[8,115] + mem[8,115] = 0
 accuracy_3_115: - 5.09765625 frac[3,115] + mem[3,115] = 0
 accuracy_10_33: - 5.09765625 frac[10,33] + mem[10,33] = 0
 accuracy_9_133: - 5.09765625 frac[9,133] + mem[9,133] = 0
 accuracy_9_121: - 5.09765625 frac[9,121] + mem[9,121] = 0
 accuracy_8_4: - 5.09765625 frac[8,4] + mem[8,4] = 0
 accuracy_8_89: - 5.09765625 frac[8,89] + mem[8,89] = 0
 accuracy_1_118: - 5.09765625 frac[1,118] + mem[1,118] = 0
 accuracy_8_141: - 5.09765625 frac[8,141] + mem[8,141] = 0
 accuracy_10_145: - 5.09765625 frac[10,145] + mem[10,145] = 0
 accuracy_9_22: - 5.09765625 frac[9,22] + mem[9,22] = 0
 accuracy_5_88: - 5.09765625 frac[5,88] + mem[5,88] = 0
 accuracy_8_147: - 5.09765625 frac[8,147] + mem[8,147] = 0
 accuracy_8_125: - 5.09765625 frac[8,125] + mem[8,125] = 0
 accuracy_9_37: - 5.09765625 frac[9,37] + mem[9,37] = 0
 accuracy_8_112: - 5.09765625 frac[8,112] + mem[8,112] = 0
 accuracy_9_103: - 5.09765625 frac[9,103] + mem[9,103] = 0
 accuracy_1_95: - 5.09765625 frac[1,95] + mem[1,95] = 0
 accuracy_8_14: - 5.09765625 frac[8,14] + mem[8,14] = 0
 accuracy_7_99: - 5.09765625 frac[7,99] + mem[7,99] = 0
 accuracy_10_127: - 5.09765625 frac[10,127] + mem[10,127] = 0
 accuracy_9_128: - 5.09765625 frac[9,128] + mem[9,128] = 0
 accuracy_9_116: - 5.09765625 frac[9,116] + mem[9,116] = 0
 accuracy_9_15: - 5.09765625 frac[9,15] + mem[9,15] = 0
 accuracy_10_74: - 5.09765625 frac[10,74] + mem[10,74] = 0
 accuracy_10_135: - 5.09765625 frac[10,135] + mem[10,135] = 0
 accuracy_8_138: - 5.09765625 frac[8,138] + mem[8,138] = 0
 accuracy_2_97: - 5.09765625 frac[2,97] + mem[2,97] = 0
 accuracy_9_94: - 5.09765625 frac[9,94] + mem[9,94] = 0
 accuracy_10_84: - 5.09765625 frac[10,84] + mem[10,84] = 0
 accuracy_8_103: - 5.09765625 frac[8,103] + mem[8,103] = 0
 accuracy_5_91: - 5.09765625 frac[5,91] + mem[5,91] = 0
 accuracy_0_95: - 5.09765625 frac[0,95] + mem[0,95] = 0
 accuracy_3_103: - 5.09765625 frac[3,103] + mem[3,103] = 0
 accuracy_8_157: - 5.09765625 frac[8,157] + mem[8,157] = 0
 accuracy_10_22: - 5.09765625 frac[10,22] + mem[10,22] = 0
 accuracy_8_37: - 5.09765625 frac[8,37] + mem[8,37] = 0
 accuracy_6_110: - 5.09765625 frac[6,110] + mem[6,110] = 0
 accuracy_9_109: - 5.09765625 frac[9,109] + mem[9,109] = 0
 accuracy_7_85: - 5.09765625 frac[7,85] + mem[7,85] = 0
 accuracy_9_98: - 5.09765625 frac[9,98] + mem[9,98] = 0
 accuracy_10_112: - 5.09765625 frac[10,112] + mem[10,112] = 0
 accuracy_10_141: - 5.09765625 frac[10,141] + mem[10,141] = 0
 accuracy_9_119: - 5.09765625 frac[9,119] + mem[9,119] = 0
 accuracy_9_10: - 5.09765625 frac[9,10] + mem[9,10] = 0
 accuracy_10_79: - 5.09765625 frac[10,79] + mem[10,79] = 0
 accuracy_2_119: - 5.09765625 frac[2,119] + mem[2,119] = 0
 accuracy_10_152: - 5.09765625 frac[10,152] + mem[10,152] = 0
 accuracy_4_86: - 5.09765625 frac[4,86] + mem[4,86] = 0
 accuracy_8_135: - 5.09765625 frac[8,135] + mem[8,135] = 0
 accuracy_10_90: - 5.09765625 frac[10,90] + mem[10,90] = 0
 accuracy_3_81: - 5.09765625 frac[3,81] + mem[3,81] = 0
 accuracy_1_97: - 5.09765625 frac[1,97] + mem[1,97] = 0
 accuracy_1_159: - 5.09765625 frac[1,159] + mem[1,159] = 0
 accuracy_3_98: - 5.09765625 frac[3,98] + mem[3,98] = 0
 accuracy_8_119: - 5.09765625 frac[8,119] + mem[8,119] = 0
 accuracy_9_104: - 5.09765625 frac[9,104] + mem[9,104] = 0
 accuracy_10_102: - 5.09765625 frac[10,102] + mem[10,102] = 0
 accuracy_7_88: - 5.09765625 frac[7,88] + mem[7,88] = 0
 accuracy_8_74: - 5.09765625 frac[8,74] + mem[8,74] = 0
 accuracy_0_114: - 5.09765625 frac[0,114] + mem[0,114] = 0
 accuracy_9_125: - 5.09765625 frac[9,125] + mem[9,125] = 0
 accuracy_9_48: - 5.09765625 frac[9,48] + mem[9,48] = 0
 accuracy_8_93: - 5.09765625 frac[8,93] + mem[8,93] = 0
 accuracy_9_158: - 5.09765625 frac[9,158] + mem[9,158] = 0
 accuracy_9_5: - 5.09765625 frac[9,5] + mem[9,5] = 0
 accuracy_10_157: - 5.09765625 frac[10,157] + mem[10,157] = 0
 accuracy_0_104: - 5.09765625 frac[0,104] + mem[0,104] = 0
 mem_tot_CPU4: - mem[0,89] - mem[0,98] - mem[0,118] - mem[0,93]
   - mem[0,115] - mem[0,90] - mem[0,112] - mem[0,99] - mem[0,84]
   - mem[0,109] - mem[0,116] - mem[0,103] - mem[0,95] - mem[0,114]
   - mem[0,104] + mem_tot_CPU4 = 0
 R422: - 3.05176e-05 mem_tot_CPU4 + normalized_mem_tot_CPU4 = 0
 rows_tot_CPU4: - 3 frac[0,89] - 3 frac[0,98] - 3 frac[0,118]
   - 3 frac[0,93] - 3 frac[0,115] - 3 frac[0,90] - 3 frac[0,112]
   - 3 frac[0,99] - 3 frac[0,84] - 3 frac[0,109] - 3 frac[0,116]
   - 3 frac[0,103] - 3 frac[0,95] - 3 frac[0,114] - 3 frac[0,104]
   + rows_tot_CPU4 = 0
 R424: - 0.0833333333333333 rows_tot_CPU4 + normalized_rows_tot_CPU4 = 0
 mem_tot_CPU6: - mem[1,84] - mem[1,89] - mem[1,116] - mem[1,110]
   - mem[1,93] - mem[1,105] - mem[1,114] - mem[1,103] - mem[1,81]
   - mem[1,118] - mem[1,95] - mem[1,97] - mem[1,159] + mem_tot_CPU6 = 0
 R426: - 3.05176e-05 mem_tot_CPU6 + normalized_mem_tot_CPU6 = 0
 rows_tot_CPU6: - 3 frac[1,84] - 3 frac[1,89] - 3 frac[1,116]
   - 3 frac[1,110] - 3 frac[1,93] - 3 frac[1,105] - 3 frac[1,114]
   - 3 frac[1,103] - 3 frac[1,81] - 3 frac[1,118] - 3 frac[1,95]
   - 3 frac[1,97] - 3 frac[1,159] + rows_tot_CPU6 = 0
 R428: - 0.0833333333333333 rows_tot_CPU6 + normalized_rows_tot_CPU6 = 0
 mem_tot_CPU10: - mem[2,103] - mem[2,104] - mem[2,115] - mem[2,116]
   - mem[2,110] - mem[2,95] - mem[2,92] - mem[2,97] - mem[2,119]
   + mem_tot_CPU10 = 0
 R430: - 3.05176e-05 mem_tot_CPU10 + normalized_mem_tot_CPU10 = 0
 rows_tot_CPU10: - 3 frac[2,103] - 3 frac[2,104] - 3 frac[2,115]
   - 3 frac[2,116] - 3 frac[2,110] - 3 frac[2,95] - 3 frac[2,92]
   - 3 frac[2,97] - 3 frac[2,119] + rows_tot_CPU10 = 0
 R432: - 0.0833333333333333 rows_tot_CPU10 + normalized_rows_tot_CPU10 = 0
 mem_tot_CPU11: - mem[3,97] - mem[3,86] - mem[3,85] - mem[3,95]
   - mem[3,105] - mem[3,93] - mem[3,99] - mem[3,88] - mem[3,109]
   - mem[3,115] - mem[3,103] - mem[3,81] - mem[3,98] + mem_tot_CPU11 = 0
 R434: - 3.05176e-05 mem_tot_CPU11 + normalized_mem_tot_CPU11 = 0
 rows_tot_CPU11: - 3 frac[3,97] - 3 frac[3,86] - 3 frac[3,85]
   - 3 frac[3,95] - 3 frac[3,105] - 3 frac[3,93] - 3 frac[3,99]
   - 3 frac[3,88] - 3 frac[3,109] - 3 frac[3,115] - 3 frac[3,103]
   - 3 frac[3,81] - 3 frac[3,98] + rows_tot_CPU11 = 0
 R436: - 0.0833333333333333 rows_tot_CPU11 + normalized_rows_tot_CPU11 = 0
 mem_tot_CPU13: - mem[4,102] - mem[4,104] - mem[4,109] - mem[4,98]
   - mem[4,84] - mem[4,94] - mem[4,105] - mem[4,90] - mem[4,85]
   - mem[4,115] - mem[4,95] - mem[4,112] - mem[4,86] + mem_tot_CPU13 = 0
 R438: - 3.05176e-05 mem_tot_CPU13 + normalized_mem_tot_CPU13 = 0
 rows_tot_CPU13: - 3 frac[4,102] - 3 frac[4,104] - 3 frac[4,109]
   - 3 frac[4,98] - 3 frac[4,84] - 3 frac[4,94] - 3 frac[4,105]
   - 3 frac[4,90] - 3 frac[4,85] - 3 frac[4,115] - 3 frac[4,95]
   - 3 frac[4,112] - 3 frac[4,86] + rows_tot_CPU13 = 0
 R440: - 0.0833333333333333 rows_tot_CPU13 + normalized_rows_tot_CPU13 = 0
 mem_tot_CPU21: - mem[5,110] - mem[5,116] - mem[5,96] - mem[5,99]
   - mem[5,105] - mem[5,102] - mem[5,88] - mem[5,91] + mem_tot_CPU21 = 0
 R442: - 3.05176e-05 mem_tot_CPU21 + normalized_mem_tot_CPU21 = 0
 rows_tot_CPU21: - 3 frac[5,110] - 3 frac[5,116] - 3 frac[5,96]
   - 3 frac[5,99] - 3 frac[5,105] - 3 frac[5,102] - 3 frac[5,88]
   - 3 frac[5,91] + rows_tot_CPU21 = 0
 R444: - 0.0833333333333333 rows_tot_CPU21 + normalized_rows_tot_CPU21 = 0
 mem_tot_CPU27: - mem[6,116] - mem[6,95] - mem[6,98] - mem[6,104]
   - mem[6,92] - mem[6,99] - mem[6,110] + mem_tot_CPU27 = 0
 R446: - 3.05176e-05 mem_tot_CPU27 + normalized_mem_tot_CPU27 = 0
 rows_tot_CPU27: - 3 frac[6,116] - 3 frac[6,95] - 3 frac[6,98]
   - 3 frac[6,104] - 3 frac[6,92] - 3 frac[6,99] - 3 frac[6,110]
   + rows_tot_CPU27 = 0
 R448: - 0.0833333333333333 rows_tot_CPU27 + normalized_rows_tot_CPU27 = 0
 mem_tot_CPU29: - mem[7,95] - mem[7,104] - mem[7,91] - mem[7,110]
   - mem[7,96] - mem[7,112] - mem[7,119] - mem[7,99] - mem[7,85]
   - mem[7,88] + mem_tot_CPU29 = 0
 R450: - 3.05176e-05 mem_tot_CPU29 + normalized_mem_tot_CPU29 = 0
 rows_tot_CPU29: - 3 frac[7,95] - 3 frac[7,104] - 3 frac[7,91]
   - 3 frac[7,110] - 3 frac[7,96] - 3 frac[7,112] - 3 frac[7,119]
   - 3 frac[7,99] - 3 frac[7,85] - 3 frac[7,88] + rows_tot_CPU29 = 0
 R452: - 0.0833333333333333 rows_tot_CPU29 + normalized_rows_tot_CPU29 = 0
 mem_tot_tor_P41: - mem[8,110] - mem[8,97] - mem[8,116] - mem[8,5]
   - mem[8,90] - mem[8,126] - mem[8,15] - mem[8,21] - mem[8,104]
   - mem[8,158] - mem[8,38] - mem[8,57] - mem[8,84] - mem[8,18] - mem[8,35]
   - mem[8,94] - mem[8,81] - mem[8,133] - mem[8,98] - mem[8,32] - mem[8,25]
   - mem[8,42] - mem[8,127] - mem[8,149] - mem[8,114] - mem[8,48]
   - mem[8,22] - mem[8,140] - mem[8,105] - mem[8,159] - mem[8,0]
   - mem[8,19] - mem[8,36] - mem[8,121] - mem[8,10] - mem[8,95]
   - mem[8,134] - mem[8,99] - mem[8,33] - mem[8,118] - mem[8,26]
   - mem[8,109] - mem[8,115] - mem[8,4] - mem[8,89] - mem[8,141]
   - mem[8,147] - mem[8,125] - mem[8,112] - mem[8,14] - mem[8,138]
   - mem[8,103] - mem[8,157] - mem[8,37] - mem[8,135] - mem[8,119]
   - mem[8,74] - mem[8,93] + mem_tot_tor_P41 = 0
 R454: - 0.00173611111111111 mem_tot_tor_P41 + normalized_mem_tot_tor_P41
   = 0
 rows_tot_tor_P41: - 3 frac[8,110] - 3 frac[8,97] - 3 frac[8,116]
   - 3 frac[8,5] - 3 frac[8,90] - 3 frac[8,126] - 3 frac[8,15]
   - 3 frac[8,21] - 3 frac[8,104] - 3 frac[8,158] - 3 frac[8,38]
   - 3 frac[8,57] - 3 frac[8,84] - 3 frac[8,18] - 3 frac[8,35]
   - 3 frac[8,94] - 3 frac[8,81] - 3 frac[8,133] - 3 frac[8,98]
   - 3 frac[8,32] - 3 frac[8,25] - 3 frac[8,42] - 3 frac[8,127]
   - 3 frac[8,149] - 3 frac[8,114] - 3 frac[8,48] - 3 frac[8,22]
   - 3 frac[8,140] - 3 frac[8,105] - 3 frac[8,159] - 3 frac[8,0]
   - 3 frac[8,19] - 3 frac[8,36] - 3 frac[8,121] - 3 frac[8,10]
   - 3 frac[8,95] - 3 frac[8,134] - 3 frac[8,99] - 3 frac[8,33]
   - 3 frac[8,118] - 3 frac[8,26] - 3 frac[8,109] - 3 frac[8,115]
   - 3 frac[8,4] - 3 frac[8,89] - 3 frac[8,141] - 3 frac[8,147]
   - 3 frac[8,125] - 3 frac[8,112] - 3 frac[8,14] - 3 frac[8,138]
   - 3 frac[8,103] - 3 frac[8,157] - 3 frac[8,37] - 3 frac[8,135]
   - 3 frac[8,119] - 3 frac[8,74] - 3 frac[8,93] + rows_tot_tor_P41 = 0
 R456: - 0.0208333333333333 rows_tot_tor_P41 + normalized_rows_tot_tor_P41
   = 0
 mem_tot_tor_P42: - mem[9,84] - mem[9,0] - mem[9,57] - mem[9,102]
   - mem[9,135] - mem[9,14] - mem[9,93] - mem[9,141] - mem[9,97]
   - mem[9,118] - mem[9,30] - mem[9,75] - mem[9,88] - mem[9,19]
   - mem[9,157] - mem[9,4] - mem[9,25] - mem[9,127] - mem[9,147]
   - mem[9,152] - mem[9,86] - mem[9,134] - mem[9,155] - mem[9,79]
   - mem[9,92] - mem[9,81] - mem[9,140] - mem[9,95] - mem[9,110]
   - mem[9,33] - mem[9,99] - mem[9,112] - mem[9,145] - mem[9,90]
   - mem[9,47] - mem[9,105] - mem[9,126] - mem[9,115] - mem[9,85]
   - mem[9,42] - mem[9,133] - mem[9,121] - mem[9,22] - mem[9,37]
   - mem[9,103] - mem[9,128] - mem[9,116] - mem[9,15] - mem[9,94]
   - mem[9,109] - mem[9,98] - mem[9,119] - mem[9,10] - mem[9,104]
   - mem[9,125] - mem[9,48] - mem[9,158] - mem[9,5] + mem_tot_tor_P42 = 0
 R458: - 0.00173611111111111 mem_tot_tor_P42 + normalized_mem_tot_tor_P42
   = 0
 rows_tot_tor_P42: - 3 frac[9,84] - 3 frac[9,0] - 3 frac[9,57]
   - 3 frac[9,102] - 3 frac[9,135] - 3 frac[9,14] - 3 frac[9,93]
   - 3 frac[9,141] - 3 frac[9,97] - 3 frac[9,118] - 3 frac[9,30]
   - 3 frac[9,75] - 3 frac[9,88] - 3 frac[9,19] - 3 frac[9,157]
   - 3 frac[9,4] - 3 frac[9,25] - 3 frac[9,127] - 3 frac[9,147]
   - 3 frac[9,152] - 3 frac[9,86] - 3 frac[9,134] - 3 frac[9,155]
   - 3 frac[9,79] - 3 frac[9,92] - 3 frac[9,81] - 3 frac[9,140]
   - 3 frac[9,95] - 3 frac[9,110] - 3 frac[9,33] - 3 frac[9,99]
   - 3 frac[9,112] - 3 frac[9,145] - 3 frac[9,90] - 3 frac[9,47]
   - 3 frac[9,105] - 3 frac[9,126] - 3 frac[9,115] - 3 frac[9,85]
   - 3 frac[9,42] - 3 frac[9,133] - 3 frac[9,121] - 3 frac[9,22]
   - 3 frac[9,37] - 3 frac[9,103] - 3 frac[9,128] - 3 frac[9,116]
   - 3 frac[9,15] - 3 frac[9,94] - 3 frac[9,109] - 3 frac[9,98]
   - 3 frac[9,119] - 3 frac[9,10] - 3 frac[9,104] - 3 frac[9,125]
   - 3 frac[9,48] - 3 frac[9,158] - 3 frac[9,5] + rows_tot_tor_P42 = 0
 R460: - 0.0208333333333333 rows_tot_tor_P42 + normalized_rows_tot_tor_P42
   = 0
 mem_tot_l1_P41: - mem[10,95] - mem[10,118] - mem[10,158] - mem[10,97]
   - mem[10,35] - mem[10,75] - mem[10,147] - mem[10,85] - mem[10,98]
   - mem[10,36] - mem[10,14] - mem[10,104] - mem[10,42] - mem[10,57]
   - mem[10,92] - mem[10,30] - mem[10,109] - mem[10,47] - mem[10,119]
   - mem[10,128] - mem[10,159] - mem[10,4] - mem[10,81] - mem[10,19]
   - mem[10,110] - mem[10,125] - mem[10,133] - mem[10,10] - mem[10,25]
   - mem[10,99] - mem[10,37] - mem[10,126] - mem[10,48] - mem[10,134]
   - mem[10,15] - mem[10,88] - mem[10,26] - mem[10,105] - mem[10,115]
   - mem[10,140] - mem[10,155] - mem[10,0] - mem[10,93] - mem[10,121]
   - mem[10,116] - mem[10,5] - mem[10,94] - mem[10,33] - mem[10,145]
   - mem[10,127] - mem[10,74] - mem[10,135] - mem[10,84] - mem[10,22]
   - mem[10,112] - mem[10,141] - mem[10,79] - mem[10,152] - mem[10,90]
   - mem[10,102] - mem[10,157] + mem_tot_l1_P41 = 0
 R462: - 0.00173611111111111 mem_tot_l1_P41 + normalized_mem_tot_l1_P41
   = 0
 rows_tot_l1_P41: - 3 frac[10,95] - 3 frac[10,118] - 3 frac[10,158]
   - 3 frac[10,97] - 3 frac[10,35] - 3 frac[10,75] - 3 frac[10,147]
   - 3 frac[10,85] - 3 frac[10,98] - 3 frac[10,36] - 3 frac[10,14]
   - 3 frac[10,104] - 3 frac[10,42] - 3 frac[10,57] - 3 frac[10,92]
   - 3 frac[10,30] - 3 frac[10,109] - 3 frac[10,47] - 3 frac[10,119]
   - 3 frac[10,128] - 3 frac[10,159] - 3 frac[10,4] - 3 frac[10,81]
   - 3 frac[10,19] - 3 frac[10,110] - 3 frac[10,125] - 3 frac[10,133]
   - 3 frac[10,10] - 3 frac[10,25] - 3 frac[10,99] - 3 frac[10,37]
   - 3 frac[10,126] - 3 frac[10,48] - 3 frac[10,134] - 3 frac[10,15]
   - 3 frac[10,88] - 3 frac[10,26] - 3 frac[10,105] - 3 frac[10,115]
   - 3 frac[10,140] - 3 frac[10,155] - 3 frac[10,0] - 3 frac[10,93]
   - 3 frac[10,121] - 3 frac[10,116] - 3 frac[10,5] - 3 frac[10,94]
   - 3 frac[10,33] - 3 frac[10,145] - 3 frac[10,127] - 3 frac[10,74]
   - 3 frac[10,135] - 3 frac[10,84] - 3 frac[10,22] - 3 frac[10,112]
   - 3 frac[10,141] - 3 frac[10,79] - 3 frac[10,152] - 3 frac[10,90]
   - 3 frac[10,102] - 3 frac[10,157] + rows_tot_l1_P41 = 0
 R464: - 0.0208333333333333 rows_tot_l1_P41 + normalized_rows_tot_l1_P41
   = 0
 row_capacity_CPU4: rows_tot_CPU4 <= 12
 row_capacity_CPU6: rows_tot_CPU6 <= 12
 row_capacity_CPU10: rows_tot_CPU10 <= 12
 row_capacity_CPU11: rows_tot_CPU11 <= 12
 row_capacity_CPU13: rows_tot_CPU13 <= 12
 row_capacity_CPU21: rows_tot_CPU21 <= 12
 row_capacity_CPU27: rows_tot_CPU27 <= 12
 row_capacity_CPU29: rows_tot_CPU29 <= 12
 row_capacity_tor_P41: rows_tot_tor_P41 <= 48
 row_capacity_tor_P42: rows_tot_tor_P42 <= 48
 row_capacity_l1_P41: rows_tot_l1_P41 <= 48
 ns_single_CPU4: 3.5 rows_tot_CPU4 - ns_single_CPU4 + pdt_m_rows_CPU4 = 0
 capacity_cores_CPU4: cores_sketch_CPU4 + cores_dpdk_CPU4 <= 7
 ns_sketch_CPU4: - ns_single_CPU4 + 11.42857142857143 cores_sketch_CPU4
   >= 0
 ns_dpdk_CPU4: - 4.285714285714287 cores_dpdk_CPU4 <= -21.42857142857143
 ns_single_CPU6: 3.5 rows_tot_CPU6 - ns_single_CPU6 + pdt_m_rows_CPU6 = 0
 capacity_cores_CPU6: cores_sketch_CPU6 + cores_dpdk_CPU6 <= 7
 ns_sketch_CPU6: - ns_single_CPU6 + 11.42857142857143 cores_sketch_CPU6
   >= 0
 ns_dpdk_CPU6: - 4.285714285714287 cores_dpdk_CPU6 <= -21.42857142857143
 ns_single_CPU10: 3.5 rows_tot_CPU10 - ns_single_CPU10 + pdt_m_rows_CPU10
   = 0
 capacity_cores_CPU10: cores_sketch_CPU10 + cores_dpdk_CPU10 <= 7
 ns_sketch_CPU10: - ns_single_CPU10 + 11.42857142857143 cores_sketch_CPU10
   >= 0
 ns_dpdk_CPU10: - 4.285714285714287 cores_dpdk_CPU10 <= -21.42857142857143
 ns_single_CPU11: 3.5 rows_tot_CPU11 - ns_single_CPU11 + pdt_m_rows_CPU11
   = 0
 capacity_cores_CPU11: cores_sketch_CPU11 + cores_dpdk_CPU11 <= 7
 ns_sketch_CPU11: - ns_single_CPU11 + 11.42857142857143 cores_sketch_CPU11
   >= 0
 ns_dpdk_CPU11: - 4.285714285714287 cores_dpdk_CPU11 <= -21.42857142857143
 ns_single_CPU13: 3.5 rows_tot_CPU13 - ns_single_CPU13 + pdt_m_rows_CPU13
   = 0
 capacity_cores_CPU13: cores_sketch_CPU13 + cores_dpdk_CPU13 <= 7
 ns_sketch_CPU13: - ns_single_CPU13 + 11.42857142857143 cores_sketch_CPU13
   >= 0
 ns_dpdk_CPU13: - 4.285714285714287 cores_dpdk_CPU13 <= -21.42857142857143
 ns_single_CPU21: 3.5 rows_tot_CPU21 - ns_single_CPU21 + pdt_m_rows_CPU21
   = 0
 capacity_cores_CPU21: cores_sketch_CPU21 + cores_dpdk_CPU21 <= 7
 ns_sketch_CPU21: - ns_single_CPU21 + 11.42857142857143 cores_sketch_CPU21
   >= 0
 ns_dpdk_CPU21: - 4.285714285714287 cores_dpdk_CPU21 <= -21.42857142857143
 ns_single_CPU27: 3.5 rows_tot_CPU27 - ns_single_CPU27 + pdt_m_rows_CPU27
   = 0
 capacity_cores_CPU27: cores_sketch_CPU27 + cores_dpdk_CPU27 <= 7
 ns_sketch_CPU27: - ns_single_CPU27 + 11.42857142857143 cores_sketch_CPU27
   >= 0
 ns_dpdk_CPU27: - 4.285714285714287 cores_dpdk_CPU27 <= -21.42857142857143
 ns_single_CPU29: 3.5 rows_tot_CPU29 - ns_single_CPU29 + pdt_m_rows_CPU29
   = 0
 capacity_cores_CPU29: cores_sketch_CPU29 + cores_dpdk_CPU29 <= 7
 ns_sketch_CPU29: - ns_single_CPU29 + 11.42857142857143 cores_sketch_CPU29
   >= 0
 ns_dpdk_CPU29: - 4.285714285714287 cores_dpdk_CPU29 <= -21.42857142857143
 ns_tor_P41: ns_tor_P41 = 6.756756756756757
 max_ns_tor_P41: ns_tor_P41 <= 11.42857142857143
 ns_tor_P42: ns_tor_P42 = 6.756756756756757
 max_ns_tor_P42: ns_tor_P42 <= 11.42857142857143
 ns_l1_P41: ns_l1_P41 = 6.756756756756757
 max_ns_l1_P41: ns_l1_P41 <= 11.42857142857143
 res_acc: 1.220703125e-04 mem_tot_CPU4 + 1.220703125e-04 mem_tot_CPU6
   + 1.220703125e-04 mem_tot_CPU10 + 1.220703125e-04 mem_tot_CPU11
   + 1.220703125e-04 mem_tot_CPU13 + 1.220703125e-04 mem_tot_CPU21
   + 1.220703125e-04 mem_tot_CPU27 + 1.220703125e-04 mem_tot_CPU29
   + 0.0208333333333333 mem_tot_tor_P41 + 0.25 rows_tot_tor_P41
   + 0.0208333333333333 mem_tot_tor_P42 + 0.25 rows_tot_tor_P42
   + 0.0208333333333333 mem_tot_l1_P41 + 0.25 rows_tot_l1_P41
   + 10 cores_sketch_CPU4 + 10 cores_dpdk_CPU4 + 10 cores_sketch_CPU6
   + 10 cores_dpdk_CPU6 + 10 cores_sketch_CPU10 + 10 cores_dpdk_CPU10
   + 10 cores_sketch_CPU11 + 10 cores_dpdk_CPU11 + 10 cores_sketch_CPU13
   + 10 cores_dpdk_CPU13 + 10 cores_sketch_CPU21 + 10 cores_dpdk_CPU21
   + 10 cores_sketch_CPU27 + 10 cores_dpdk_CPU27 + 10 cores_sketch_CPU29
   + 10 cores_dpdk_CPU29 - res = 0
 pdt_m_rows_CPU4: pdt_m_rows_CPU4 + [ - rows_tot_CPU4 * m_access_time_CPU4
   ] = 0
 pdt_m_rows_CPU6: pdt_m_rows_CPU6 + [ - rows_tot_CPU6 * m_access_time_CPU6
   ] = 0
 pdt_m_rows_CPU10: pdt_m_rows_CPU10 + [
   - rows_tot_CPU10 * m_access_time_CPU10 ] = 0
 pdt_m_rows_CPU11: pdt_m_rows_CPU11 + [
   - rows_tot_CPU11 * m_access_time_CPU11 ] = 0
 pdt_m_rows_CPU13: pdt_m_rows_CPU13 + [
   - rows_tot_CPU13 * m_access_time_CPU13 ] = 0
 pdt_m_rows_CPU21: pdt_m_rows_CPU21 + [
   - rows_tot_CPU21 * m_access_time_CPU21 ] = 0
 pdt_m_rows_CPU27: pdt_m_rows_CPU27 + [
   - rows_tot_CPU27 * m_access_time_CPU27 ] = 0
 pdt_m_rows_CPU29: pdt_m_rows_CPU29 + [
   - rows_tot_CPU29 * m_access_time_CPU29 ] = 0
Bounds
 frac[2,103] <= 1
 frac[8,110] <= 1
 frac[10,95] <= 1
 frac[4,102] <= 1
 frac[9,84] <= 1
 frac[0,89] <= 1
 frac[8,97] <= 1
 frac[3,97] <= 1
 frac[8,116] <= 1
 frac[5,110] <= 1
 frac[6,116] <= 1
 frac[7,95] <= 1
 frac[10,118] <= 1
 frac[8,5] <= 1
 frac[8,90] <= 1
 frac[7,104] <= 1
 frac[0,98] <= 1
 frac[9,0] <= 1
 frac[10,158] <= 1
 frac[3,86] <= 1
 frac[4,104] <= 1
 frac[2,104] <= 1
 frac[6,95] <= 1
 frac[8,126] <= 1
 frac[6,98] <= 1
 frac[10,97] <= 1
 frac[10,35] <= 1
 frac[9,57] <= 1
 frac[8,15] <= 1
 frac[9,102] <= 1
 frac[9,135] <= 1
 frac[9,14] <= 1
 frac[10,75] <= 1
 frac[2,115] <= 1
 frac[8,21] <= 1
 frac[10,147] <= 1
 frac[3,85] <= 1
 frac[4,109] <= 1
 frac[9,93] <= 1
 frac[10,85] <= 1
 frac[8,104] <= 1
 frac[8,158] <= 1
 frac[8,38] <= 1
 frac[1,84] <= 1
 frac[10,98] <= 1
 frac[8,57] <= 1
 frac[0,118] <= 1
 frac[9,141] <= 1
 frac[10,36] <= 1
 frac[1,89] <= 1
 frac[9,97] <= 1
 frac[9,118] <= 1
 frac[2,116] <= 1
 frac[8,84] <= 1
 frac[3,95] <= 1
 frac[10,14] <= 1
 frac[9,30] <= 1
 frac[8,18] <= 1
 frac[9,75] <= 1
 frac[4,98] <= 1
 frac[9,88] <= 1
 frac[9,19] <= 1
 frac[0,93] <= 1
 frac[2,110] <= 1
 frac[10,104] <= 1
 frac[8,35] <= 1
 frac[6,104] <= 1
 frac[10,42] <= 1
 frac[2,95] <= 1
 frac[7,91] <= 1
 frac[0,115] <= 1
 frac[10,57] <= 1
 frac[8,94] <= 1
 frac[5,116] <= 1
 frac[9,157] <= 1
 frac[9,4] <= 1
 frac[8,81] <= 1
 frac[4,84] <= 1
 frac[8,133] <= 1
 frac[9,25] <= 1
 frac[10,92] <= 1
 frac[10,30] <= 1
 frac[0,90] <= 1
 frac[8,98] <= 1
 frac[10,109] <= 1
 frac[8,32] <= 1
 frac[10,47] <= 1
 frac[0,112] <= 1
 frac[9,127] <= 1
 frac[10,119] <= 1
 frac[9,147] <= 1
 frac[0,99] <= 1
 frac[9,152] <= 1
 frac[10,128] <= 1
 frac[4,94] <= 1
 frac[1,116] <= 1
 frac[8,25] <= 1
 frac[10,159] <= 1
 frac[4,105] <= 1
 frac[10,4] <= 1
 frac[10,81] <= 1
 frac[0,84] <= 1
 frac[6,92] <= 1
 frac[10,19] <= 1
 frac[8,42] <= 1
 frac[1,110] <= 1
 frac[9,86] <= 1
 frac[6,99] <= 1
 frac[8,127] <= 1
 frac[8,149] <= 1
 frac[10,110] <= 1
 frac[8,114] <= 1
 frac[5,96] <= 1
 frac[1,93] <= 1
 frac[10,125] <= 1
 frac[8,48] <= 1
 frac[9,134] <= 1
 frac[7,110] <= 1
 frac[10,133] <= 1
 frac[9,155] <= 1
 frac[10,10] <= 1
 frac[9,79] <= 1
 frac[8,22] <= 1
 frac[8,140] <= 1
 frac[10,25] <= 1
 frac[9,92] <= 1
 frac[8,105] <= 1
 frac[8,159] <= 1
 frac[3,105] <= 1
 frac[9,81] <= 1
 frac[1,105] <= 1
 frac[10,99] <= 1
 frac[9,140] <= 1
 frac[5,99] <= 1
 frac[10,37] <= 1
 frac[10,126] <= 1
 frac[7,96] <= 1
 frac[10,48] <= 1
 frac[8,0] <= 1
 frac[0,109] <= 1
 frac[10,134] <= 1
 frac[10,15] <= 1
 frac[10,88] <= 1
 frac[8,19] <= 1
 frac[1,114] <= 1
 frac[10,26] <= 1
 frac[1,103] <= 1
 frac[9,95] <= 1
 frac[10,105] <= 1
 frac[8,36] <= 1
 frac[8,121] <= 1
 frac[5,105] <= 1
 frac[9,110] <= 1
 frac[9,33] <= 1
 frac[2,92] <= 1
 frac[0,116] <= 1
 frac[5,102] <= 1
 frac[9,99] <= 1
 frac[10,115] <= 1
 frac[8,10] <= 1
 frac[0,103] <= 1
 frac[8,95] <= 1
 frac[10,140] <= 1
 frac[4,90] <= 1
 frac[9,112] <= 1
 frac[7,112] <= 1
 frac[9,145] <= 1
 frac[10,155] <= 1
 frac[3,93] <= 1
 frac[4,85] <= 1
 frac[10,0] <= 1
 frac[10,93] <= 1
 frac[8,134] <= 1
 frac[9,90] <= 1
 frac[8,99] <= 1
 frac[3,99] <= 1
 frac[4,115] <= 1
 frac[9,47] <= 1
 frac[8,33] <= 1
 frac[8,118] <= 1
 frac[9,105] <= 1
 frac[1,81] <= 1
 frac[10,121] <= 1
 frac[9,126] <= 1
 frac[10,116] <= 1
 frac[4,95] <= 1
 frac[9,115] <= 1
 frac[8,26] <= 1
 frac[7,119] <= 1
 frac[3,88] <= 1
 frac[10,5] <= 1
 frac[10,94] <= 1
 frac[8,109] <= 1
 frac[3,109] <= 1
 frac[9,85] <= 1
 frac[4,112] <= 1
 frac[9,42] <= 1
 frac[8,115] <= 1
 frac[3,115] <= 1
 frac[10,33] <= 1
 frac[9,133] <= 1
 frac[9,121] <= 1
 frac[8,4] <= 1
 frac[8,89] <= 1
 frac[1,118] <= 1
 frac[8,141] <= 1
 frac[10,145] <= 1
 frac[9,22] <= 1
 frac[5,88] <= 1
 frac[8,147] <= 1
 frac[8,125] <= 1
 frac[9,37] <= 1
 frac[8,112] <= 1
 frac[9,103] <= 1
 frac[1,95] <= 1
 frac[8,14] <= 1
 frac[7,99] <= 1
 frac[10,127] <= 1
 frac[9,128] <= 1
 frac[9,116] <= 1
 frac[9,15] <= 1
 frac[10,74] <= 1
 frac[10,135] <= 1
 frac[8,138] <= 1
 frac[2,97] <= 1
 frac[9,94] <= 1
 frac[10,84] <= 1
 frac[8,103] <= 1
 frac[5,91] <= 1
 frac[0,95] <= 1
 frac[3,103] <= 1
 frac[8,157] <= 1
 frac[10,22] <= 1
 frac[8,37] <= 1
 frac[6,110] <= 1
 frac[9,109] <= 1
 frac[7,85] <= 1
 frac[9,98] <= 1
 frac[10,112] <= 1
 frac[10,141] <= 1
 frac[9,119] <= 1
 frac[9,10] <= 1
 frac[10,79] <= 1
 frac[2,119] <= 1
 frac[10,152] <= 1
 frac[4,86] <= 1
 frac[8,135] <= 1
 frac[10,90] <= 1
 frac[3,81] <= 1
 frac[1,97] <= 1
 frac[1,159] <= 1
 frac[3,98] <= 1
 frac[8,119] <= 1
 frac[9,104] <= 1
 frac[10,102] <= 1
 frac[7,88] <= 1
 frac[8,74] <= 1
 frac[0,114] <= 1
 frac[9,125] <= 1
 frac[9,48] <= 1
 frac[8,93] <= 1
 frac[9,158] <= 1
 frac[9,5] <= 1
 frac[10,157] <= 1
 frac[0,104] <= 1
 mem_tot_CPU4 <= 32768
 normalized_mem_tot_CPU4 <= 1
 normalized_rows_tot_CPU4 <= 1
 mem_tot_CPU6 <= 32768
 normalized_mem_tot_CPU6 <= 1
 normalized_rows_tot_CPU6 <= 1
 mem_tot_CPU10 <= 32768
 normalized_mem_tot_CPU10 <= 1
 normalized_rows_tot_CPU10 <= 1
 mem_tot_CPU11 <= 32768
 normalized_mem_tot_CPU11 <= 1
 normalized_rows_tot_CPU11 <= 1
 mem_tot_CPU13 <= 32768
 normalized_mem_tot_CPU13 <= 1
 normalized_rows_tot_CPU13 <= 1
 mem_tot_CPU21 <= 32768
 normalized_mem_tot_CPU21 <= 1
 normalized_rows_tot_CPU21 <= 1
 mem_tot_CPU27 <= 32768
 normalized_mem_tot_CPU27 <= 1
 normalized_rows_tot_CPU27 <= 1
 mem_tot_CPU29 <= 32768
 normalized_mem_tot_CPU29 <= 1
 normalized_rows_tot_CPU29 <= 1
 mem_tot_tor_P41 <= 576
 normalized_mem_tot_tor_P41 <= 1
 normalized_rows_tot_tor_P41 <= 1
 mem_tot_tor_P42 <= 576
 normalized_mem_tot_tor_P42 <= 1
 normalized_rows_tot_tor_P42 <= 1
 mem_tot_l1_P41 <= 576
 normalized_mem_tot_l1_P41 <= 1
 normalized_rows_tot_l1_P41 <= 1
 cores_sketch_CPU4 <= 7
 1 <= cores_dpdk_CPU4 <= 7
 cores_sketch_CPU6 <= 7
 1 <= cores_dpdk_CPU6 <= 7
 cores_sketch_CPU10 <= 7
 1 <= cores_dpdk_CPU10 <= 7
 cores_sketch_CPU11 <= 7
 1 <= cores_dpdk_CPU11 <= 7
 cores_sketch_CPU13 <= 7
 1 <= cores_dpdk_CPU13 <= 7
 cores_sketch_CPU21 <= 7
 1 <= cores_dpdk_CPU21 <= 7
 cores_sketch_CPU27 <= 7
 1 <= cores_dpdk_CPU27 <= 7
 cores_sketch_CPU29 <= 7
 1 <= cores_dpdk_CPU29 <= 7
Generals
 cores_sketch_CPU4 cores_dpdk_CPU4 cores_sketch_CPU6 cores_dpdk_CPU6
 cores_sketch_CPU10 cores_dpdk_CPU10 cores_sketch_CPU11 cores_dpdk_CPU11
 cores_sketch_CPU13 cores_dpdk_CPU13 cores_sketch_CPU21 cores_dpdk_CPU21
 cores_sketch_CPU27 cores_dpdk_CPU27 cores_sketch_CPU29 cores_dpdk_CPU29
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_tor_P41 , mem_tot_tor_P42 ,
   mem_tot_l1_P41 )
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU4 , normalized_mem_tot_CPU6 ,
   normalized_mem_tot_CPU10 , normalized_mem_tot_CPU11 ,
   normalized_mem_tot_CPU13 , normalized_mem_tot_CPU21 ,
   normalized_mem_tot_CPU27 , normalized_mem_tot_CPU29 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_tor_P41 , rows_tot_tor_P42 ,
   rows_tot_l1_P41 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU4 ,
   normalized_rows_tot_CPU6 , normalized_rows_tot_CPU10 ,
   normalized_rows_tot_CPU11 , normalized_rows_tot_CPU13 ,
   normalized_rows_tot_CPU21 , normalized_rows_tot_CPU27 ,
   normalized_rows_tot_CPU29 )
 mem_access_time_CPU4: m_access_time_CPU4 = PWL ( mem_tot_CPU4 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU6: m_access_time_CPU6 = PWL ( mem_tot_CPU6 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU10: m_access_time_CPU10 = PWL ( mem_tot_CPU10 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU11: m_access_time_CPU11 = PWL ( mem_tot_CPU11 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU13: m_access_time_CPU13 = PWL ( mem_tot_CPU13 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU21: m_access_time_CPU21 = PWL ( mem_tot_CPU21 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU27: m_access_time_CPU27 = PWL ( mem_tot_CPU27 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU29: m_access_time_CPU29 = PWL ( mem_tot_CPU29 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

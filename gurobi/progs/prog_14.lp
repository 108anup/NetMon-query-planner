\ Model UnivmonGreedy
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_mem_CPU: Priority=20 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU1 + mem_tot_CPU2 + mem_tot_CPU3 + mem_tot_CPU4 + mem_tot_CPU5
   + mem_tot_CPU6 + mem_tot_CPU7 + mem_tot_CPU8 + mem_tot_CPU9
   + mem_tot_CPU10 + mem_tot_CPU11 + mem_tot_CPU12 + mem_tot_CPU13
   + mem_tot_CPU14 + mem_tot_CPU15 + mem_tot_CPU16 + mem_tot_CPU17
   + mem_tot_CPU18 + mem_tot_CPU19 + mem_tot_CPU20 + mem_tot_CPU21
   + mem_tot_CPU22 + mem_tot_CPU23 + mem_tot_CPU24 + mem_tot_CPU25
   + mem_tot_CPU26 + mem_tot_CPU27 + mem_tot_CPU28 + mem_tot_CPU29
   + mem_tot_CPU30 + mem_tot_CPU31 + mem_tot_CPU32 + mem_tot_CPU33
   + mem_tot_CPU34 + mem_tot_CPU35 + mem_tot_CPU36 + mem_tot_CPU37
   + mem_tot_CPU38 + mem_tot_CPU39 + mem_tot_CPU40 + mem_tot_CPU41
   + mem_tot_CPU42 + mem_tot_CPU43 + mem_tot_CPU44 + mem_tot_CPU45
   + mem_tot_CPU46 + mem_tot_CPU47 + mem_tot_CPU48 + mem_tot_CPU49
   + mem_tot_CPU50 + mem_tot_CPU51 + mem_tot_CPU52 + mem_tot_CPU53
   + mem_tot_CPU54 + mem_tot_CPU55 + mem_tot_CPU56 + mem_tot_CPU57
   + mem_tot_CPU58 + mem_tot_CPU59 + mem_tot_CPU60 + mem_tot_CPU61
   + mem_tot_CPU62 + mem_tot_CPU63 + mem_tot_CPU64 + mem_tot_CPU65
   + mem_tot_CPU66 + mem_tot_CPU67 + mem_tot_CPU68 + mem_tot_CPU69
   + mem_tot_CPU70 + mem_tot_CPU71 + mem_tot_CPU72 + mem_tot_CPU73
   + mem_tot_CPU74 + mem_tot_CPU75 + mem_tot_CPU76 + mem_tot_CPU77
   + mem_tot_CPU78 + mem_tot_CPU79 + mem_tot_CPU80 + mem_tot_CPU81
   + mem_tot_CPU82 + mem_tot_CPU83 + mem_tot_CPU84 + mem_tot_CPU85
   + mem_tot_CPU86 + mem_tot_CPU87 + mem_tot_CPU88 + mem_tot_CPU89
   + mem_tot_CPU90 + mem_tot_CPU91 + mem_tot_CPU92 + mem_tot_CPU93
   + mem_tot_CPU94 + mem_tot_CPU95 + mem_tot_CPU96 + mem_tot_CPU97
   + mem_tot_CPU98 + mem_tot_CPU99 + mem_tot_CPU100 + mem_tot_CPU101
   + mem_tot_CPU102 + mem_tot_CPU103 + mem_tot_CPU104 + mem_tot_CPU105
   + mem_tot_CPU106 + mem_tot_CPU107 + mem_tot_CPU108 + mem_tot_CPU109
   + mem_tot_CPU110 + mem_tot_CPU111 + mem_tot_CPU112 + mem_tot_CPU113
   + mem_tot_CPU114 + mem_tot_CPU115 + mem_tot_CPU116 + mem_tot_CPU117
   + mem_tot_CPU118 + mem_tot_CPU119 + mem_tot_CPU120 + mem_tot_CPU121
   + mem_tot_CPU122 + mem_tot_CPU123 + mem_tot_CPU124 + mem_tot_CPU125
   + mem_tot_CPU126 + mem_tot_CPU127 + mem_tot_CPU128 + mem_tot_CPU129
   + mem_tot_CPU130 + mem_tot_CPU131 + mem_tot_CPU132 + mem_tot_CPU133
   + mem_tot_CPU134 + mem_tot_CPU135 + mem_tot_CPU136 + mem_tot_CPU137
   + mem_tot_CPU138 + mem_tot_CPU139 + mem_tot_CPU140 + mem_tot_CPU141
   + mem_tot_CPU142 + mem_tot_CPU143 + mem_tot_CPU144 + mem_tot_CPU145
   + mem_tot_CPU146 + mem_tot_CPU147 + mem_tot_CPU148 + mem_tot_CPU149
   + mem_tot_CPU150 + mem_tot_CPU151 + mem_tot_CPU152 + mem_tot_CPU153
   + mem_tot_CPU154 + mem_tot_CPU155 + mem_tot_CPU156 + mem_tot_CPU157
   + mem_tot_CPU158 + mem_tot_CPU159 + mem_tot_CPU160 + mem_tot_CPU161
   + mem_tot_CPU162 + mem_tot_CPU163 + mem_tot_CPU164 + mem_tot_CPU165
   + mem_tot_CPU166 + mem_tot_CPU167 + mem_tot_CPU168 + mem_tot_CPU169
   + mem_tot_CPU170 + mem_tot_CPU171 + mem_tot_CPU172 + mem_tot_CPU173
   + mem_tot_CPU174 + mem_tot_CPU175 + mem_tot_CPU176 + mem_tot_CPU177
   + mem_tot_CPU178 + mem_tot_CPU179 + mem_tot_CPU180 + mem_tot_CPU181
   + mem_tot_CPU182 + mem_tot_CPU183 + mem_tot_CPU184 + mem_tot_CPU185
   + mem_tot_CPU186 + mem_tot_CPU187 + mem_tot_CPU188 + mem_tot_CPU189
   + mem_tot_CPU190 + mem_tot_CPU191 + mem_tot_CPU192
  CPU_mem_load: Priority=15 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU
  tot_mem_P4: Priority=10 Weight=1 AbsTol=0 RelTol=0
   mem_tot_tor_P41 + mem_tot_tor_P42 + mem_tot_tor_P43 + mem_tot_tor_P44
   + mem_tot_l1_P41 + mem_tot_l1_P42 + mem_tot_l2_P4
  P4_mem_load: Priority=5 Weight=1 AbsTol=0 RelTol=0
   max_mem_P4
Subject To
 cov_0_57: frac[192,57] + frac[198,57] + frac[28,57] + frac[196,57]
   + frac[197,57] + frac[173,57] + frac[195,57] >= 0.9
 cov_1_24: frac[193,24] + frac[196,24] + frac[61,24] + frac[3,24]
   + frac[192,24] >= 1
 cov_2_201: frac[194,201] + frac[96,201] + frac[139,201] >= 0.7
 cov_3_25: frac[5,25] + frac[3,25] + frac[192,25] >= 0.8
 cov_4_2: frac[198,2] + frac[194,2] + frac[197,2] + frac[196,2]
   + frac[192,2] + frac[8,2] + frac[107,2] >= 0.8
 cov_5_237: frac[194,237] + frac[169,237] + frac[195,237] + frac[99,237]
   + frac[197,237] >= 0.9
 cov_6_94: frac[196,94] + frac[193,94] + frac[198,94] + frac[195,94]
   + frac[197,94] + frac[62,94] + frac[188,94] >= 0.8
 cov_7_183: frac[193,183] + frac[195,183] + frac[90,183] + frac[197,183]
   + frac[198,183] + frac[174,183] + frac[196,183] >= 1
 cov_8_3: frac[43,3] + frac[192,3] + frac[6,3] >= 0.9
 cov_9_24: frac[39,24] + frac[42,24] + frac[192,24] >= 0.9
 cov_10_227: frac[31,227] + frac[43,227] + frac[192,227] >= 1
 cov_11_110: frac[198,110] + frac[195,110] + frac[193,110] + frac[197,110]
   + frac[169,110] + frac[60,110] + frac[196,110] >= 1
 cov_12_120: frac[184,120] + frac[192,120] + frac[198,120] + frac[1,120]
   + frac[196,120] + frac[197,120] + frac[195,120] >= 1
 cov_13_216: frac[193,216] + frac[194,216] + frac[57,216] + frac[116,216]
   + frac[198,216] + frac[196,216] + frac[197,216] >= 0.9
 cov_14_141: frac[39,141] + frac[194,141] + frac[198,141] + frac[192,141]
   + frac[196,141] + frac[197,141] + frac[128,141] >= 1
 cov_15_87: frac[192,87] + frac[42,87] + frac[47,87] >= 0.8
 cov_16_1: frac[108,1] + frac[194,1] + frac[138,1] >= 0.7
 cov_17_92: frac[127,92] + frac[194,92] + frac[193,92] + frac[95,92]
   + frac[198,92] + frac[196,92] + frac[197,92] >= 0.9
 cov_18_147: frac[88,147] + frac[193,147] + frac[192,147] + frac[7,147]
   + frac[196,147] >= 1
 cov_19_16: frac[198,16] + frac[196,16] + frac[193,16] + frac[94,16]
   + frac[197,16] + frac[195,16] + frac[174,16] >= 1
 cov_20_253: frac[74,253] + frac[193,253] + frac[192,253] + frac[36,253]
   + frac[196,253] >= 1
 cov_21_235: frac[196,235] + frac[95,235] + frac[192,235] + frac[8,235]
   + frac[193,235] >= 0.7
 cov_22_134: frac[194,134] + frac[118,134] + frac[122,134] >= 0.8
 cov_23_254: frac[114,254] + frac[127,254] + frac[194,254] >= 1
 cov_24_255: frac[194,255] + frac[118,255] + frac[128,255] >= 0.8
 cov_25_144: frac[143,144] + frac[113,144] + frac[194,144] >= 1
 cov_26_68: frac[192,68] + frac[31,68] + frac[17,68] >= 0.9
 cov_27_246: frac[193,246] + frac[195,246] + frac[197,246] + frac[169,246]
   + frac[198,246] + frac[60,246] + frac[196,246] >= 0.8
 cov_28_90: frac[4,90] + frac[192,90] + frac[73,90] + frac[193,90]
   + frac[196,90] >= 1
 cov_29_66: frac[198,66] + frac[197,66] + frac[194,66] + frac[122,66]
   + frac[6,66] + frac[196,66] + frac[192,66] >= 0.7
 cov_30_7: frac[170,7] + frac[195,7] + frac[194,7] + frac[197,7]
   + frac[123,7] >= 1
 cov_31_62: frac[187,62] + frac[182,62] + frac[195,62] >= 0.8
 cov_32_55: frac[134,55] + frac[194,55] + frac[195,55] + frac[168,55]
   + frac[197,55] >= 0.7
 cov_33_188: frac[192,188] + frac[41,188] + frac[195,188] + frac[165,188]
   + frac[198,188] + frac[196,188] + frac[197,188] >= 0.7
 cov_34_222: frac[197,222] + frac[192,222] + frac[157,222] + frac[196,222]
   + frac[44,222] + frac[198,222] + frac[195,222] >= 1
 cov_35_216: frac[14,216] + frac[192,216] + frac[7,216] >= 0.8
 cov_36_7: frac[196,7] + frac[40,7] + frac[192,7] + frac[195,7]
   + frac[198,7] + frac[197,7] + frac[183,7] >= 0.7
 cov_37_91: frac[197,91] + frac[193,91] + frac[125,91] + frac[194,91]
   + frac[59,91] + frac[196,91] + frac[198,91] >= 0.8
 cov_38_169: frac[64,169] + frac[193,169] + frac[81,169] >= 0.8
 cov_39_135: frac[95,135] + frac[193,135] + frac[196,135] + frac[35,135]
   + frac[192,135] >= 1
 cov_40_166: frac[146,166] + frac[193,166] + frac[67,166] + frac[198,166]
   + frac[195,166] + frac[197,166] + frac[196,166] >= 1
 cov_41_212: frac[150,212] + frac[193,212] + frac[196,212] + frac[197,212]
   + frac[198,212] + frac[195,212] + frac[71,212] >= 0.7
 cov_42_57: frac[155,57] + frac[180,57] + frac[195,57] >= 1
 cov_43_229: frac[193,229] + frac[196,229] + frac[194,229] + frac[198,229]
   + frac[135,229] + frac[197,229] + frac[63,229] >= 0.9
 cov_44_152: frac[3,152] + frac[192,152] + frac[194,152] + frac[197,152]
   + frac[198,152] + frac[131,152] + frac[196,152] >= 0.7
 cov_45_81: frac[194,81] + frac[84,81] + frac[197,81] + frac[127,81]
   + frac[198,81] + frac[196,81] + frac[193,81] >= 0.8
 cov_46_56: frac[192,56] + frac[197,56] + frac[195,56] + frac[38,56]
   + frac[198,56] + frac[165,56] + frac[196,56] >= 0.7
 cov_47_83: frac[187,83] + frac[195,83] + frac[172,83] >= 0.7
 cov_48_18: frac[79,18] + frac[54,18] + frac[193,18] >= 0.8
 cov_49_157: frac[169,157] + frac[115,157] + frac[197,157] + frac[194,157]
   + frac[195,157] >= 1
 cov_50_227: frac[193,227] + frac[79,227] + frac[81,227] >= 0.7
 cov_51_14: frac[192,14] + frac[29,14] + frac[45,14] >= 0.7
 cov_52_72: frac[64,72] + frac[53,72] + frac[193,72] >= 0.7
 cov_53_55: frac[98,55] + frac[196,55] + frac[194,55] + frac[90,55]
   + frac[193,55] + frac[198,55] + frac[197,55] >= 0.8
 cov_54_237: frac[194,237] + frac[195,237] + frac[174,237] + frac[197,237]
   + frac[105,237] >= 0.8
 cov_55_34: frac[194,34] + frac[195,34] + frac[115,34] + frac[149,34]
   + frac[197,34] >= 0.7
 cov_56_8: frac[195,8] + frac[198,8] + frac[190,8] + frac[196,8]
   + frac[92,8] + frac[193,8] + frac[197,8] >= 0.9
 cov_57_77: frac[197,77] + frac[194,77] + frac[198,77] + frac[192,77]
   + frac[196,77] + frac[27,77] + frac[119,77] >= 0.8
 cov_58_248: frac[85,248] + frac[193,248] + frac[89,248] >= 0.7
 cov_59_188: frac[198,188] + frac[97,188] + frac[73,188] + frac[196,188]
   + frac[193,188] + frac[194,188] + frac[197,188] >= 0.7
 cov_60_37: frac[196,37] + frac[198,37] + frac[197,37] + frac[147,37]
   + frac[192,37] + frac[195,37] + frac[13,37] >= 0.8
 cov_61_25: frac[190,25] + frac[195,25] + frac[154,25] >= 0.8
 cov_62_102: frac[197,102] + frac[44,102] + frac[198,102] + frac[196,102]
   + frac[190,102] + frac[195,102] + frac[192,102] >= 1
 cov_63_41: frac[193,41] + frac[196,41] + frac[74,41] + frac[124,41]
   + frac[197,41] + frac[198,41] + frac[194,41] >= 0.8
 cov_64_213: frac[197,213] + frac[194,213] + frac[108,213] + frac[15,213]
   + frac[192,213] + frac[196,213] + frac[198,213] >= 0.7
 cov_65_23: frac[194,23] + frac[135,23] + frac[193,23] + frac[198,23]
   + frac[197,23] + frac[92,23] + frac[196,23] >= 0.9
 cov_66_136: frac[195,136] + frac[184,136] + frac[172,136] >= 0.9
 cov_67_149: frac[190,149] + frac[196,149] + frac[197,149] + frac[198,149]
   + frac[195,149] + frac[34,149] + frac[192,149] >= 0.9
 cov_68_235: frac[39,235] + frac[40,235] + frac[192,235] >= 0.7
 cov_69_57: frac[32,57] + frac[192,57] + frac[1,57] >= 0.9
 cov_70_169: frac[198,169] + frac[192,169] + frac[148,169] + frac[196,169]
   + frac[197,169] + frac[195,169] + frac[8,169] >= 0.7
 cov_71_88: frac[198,88] + frac[102,88] + frac[196,88] + frac[197,88]
   + frac[71,88] + frac[194,88] + frac[193,88] >= 0.9
 cov_72_254: frac[176,254] + frac[195,254] + frac[159,254] >= 0.9
 cov_73_254: frac[193,254] + frac[139,254] + frac[198,254] + frac[197,254]
   + frac[94,254] + frac[194,254] + frac[196,254] >= 1
 cov_74_230: frac[111,230] + frac[194,230] + frac[96,230] >= 0.7
 cov_75_255: frac[194,255] + frac[195,255] + frac[191,255] + frac[197,255]
   + frac[117,255] >= 0.9
 cov_76_148: frac[101,148] + frac[194,148] + frac[97,148] >= 0.8
 cov_77_89: frac[198,89] + frac[68,89] + frac[197,89] + frac[195,89]
   + frac[196,89] + frac[193,89] + frac[190,89] >= 0.8
 cov_78_119: frac[193,119] + frac[82,119] + frac[49,119] >= 0.9
 cov_79_7: frac[193,7] + frac[196,7] + frac[152,7] + frac[195,7]
   + frac[198,7] + frac[88,7] + frac[197,7] >= 0.9
 cov_80_225: frac[157,225] + frac[197,225] + frac[195,225] + frac[194,225]
   + frac[123,225] >= 1
 cov_81_232: frac[80,232] + frac[192,232] + frac[193,232] + frac[196,232]
   + frac[6,232] >= 0.7
 cov_82_231: frac[36,231] + frac[196,231] + frac[197,231] + frac[192,231]
   + frac[100,231] + frac[194,231] + frac[198,231] >= 1
 cov_83_67: frac[101,67] + frac[96,67] + frac[194,67] >= 0.9
 cov_84_185: frac[67,185] + frac[196,185] + frac[197,185] + frac[195,185]
   + frac[193,185] + frac[149,185] + frac[198,185] >= 0.7
 cov_85_250: frac[197,250] + frac[146,250] + frac[196,250] + frac[198,250]
   + frac[195,250] + frac[44,250] + frac[192,250] >= 0.9
 cov_86_101: frac[79,101] + frac[197,101] + frac[195,101] + frac[193,101]
   + frac[196,101] + frac[198,101] + frac[182,101] >= 1
 cov_87_61: frac[196,61] + frac[74,61] + frac[197,61] + frac[194,61]
   + frac[193,61] + frac[100,61] + frac[198,61] >= 0.7
 cov_88_8: frac[198,8] + frac[194,8] + frac[192,8] + frac[196,8]
   + frac[119,8] + frac[37,8] + frac[197,8] >= 0.7
 cov_89_181: frac[192,181] + frac[39,181] + frac[29,181] >= 0.9
 cov_90_41: frac[193,41] + frac[71,41] + frac[76,41] >= 0.8
 cov_91_211: frac[193,211] + frac[196,211] + frac[118,211] + frac[197,211]
   + frac[194,211] + frac[198,211] + frac[92,211] >= 0.8
 cov_92_206: frac[71,206] + frac[193,206] + frac[79,206] >= 0.8
 cov_93_168: frac[198,168] + frac[194,168] + frac[85,168] + frac[196,168]
   + frac[193,168] + frac[197,168] + frac[98,168] >= 1
 cov_94_214: frac[156,214] + frac[102,214] + frac[194,214] + frac[195,214]
   + frac[197,214] >= 0.8
 cov_95_85: frac[47,85] + frac[156,85] + frac[197,85] + frac[192,85]
   + frac[195,85] + frac[196,85] + frac[198,85] >= 0.8
 cov_96_225: frac[33,225] + frac[193,225] + frac[196,225] + frac[192,225]
   + frac[74,225] >= 0.8
 cov_97_224: frac[197,224] + frac[153,224] + frac[194,224] + frac[195,224]
   + frac[96,224] >= 0.8
 cov_98_108: frac[194,108] + frac[140,108] + frac[102,108] >= 0.8
 cov_99_65: frac[54,65] + frac[197,65] + frac[195,65] + frac[193,65]
   + frac[198,65] + frac[196,65] + frac[180,65] >= 0.9
 cov_100_211: frac[193,211] + frac[34,211] + frac[196,211] + frac[192,211]
   + frac[56,211] >= 0.7
 cov_101_123: frac[194,123] + frac[140,123] + frac[196,123] + frac[198,123]
   + frac[197,123] + frac[192,123] + frac[40,123] >= 1
 cov_102_239: frac[116,239] + frac[194,239] + frac[118,239] >= 1
 cov_103_162: frac[195,162] + frac[170,162] + frac[197,162] + frac[196,162]
   + frac[192,162] + frac[7,162] + frac[198,162] >= 0.9
 cov_104_216: frac[195,216] + frac[190,216] + frac[194,216] + frac[139,216]
   + frac[197,216] >= 1
 cov_105_171: frac[194,171] + frac[114,171] + frac[135,171] >= 0.9
 cov_106_100: frac[104,100] + frac[194,100] + frac[102,100] >= 1
 cov_107_241: frac[192,241] + frac[197,241] + frac[147,241] + frac[29,241]
   + frac[195,241] + frac[196,241] + frac[198,241] >= 0.8
 cov_108_240: frac[196,240] + frac[197,240] + frac[198,240] + frac[195,240]
   + frac[41,240] + frac[192,240] + frac[159,240] >= 1
 cov_109_73: frac[110,73] + frac[61,73] + frac[193,73] + frac[198,73]
   + frac[197,73] + frac[194,73] + frac[196,73] >= 0.8
 cov_110_177: frac[196,177] + frac[198,177] + frac[39,177] + frac[197,177]
   + frac[183,177] + frac[195,177] + frac[192,177] >= 0.7
 cov_111_3: frac[32,3] + frac[21,3] + frac[192,3] >= 0.7
 cov_112_43: frac[192,43] + frac[26,43] + frac[196,43] + frac[198,43]
   + frac[157,43] + frac[195,43] + frac[197,43] >= 0.8
 cov_113_189: frac[25,189] + frac[192,189] + frac[6,189] >= 1
 cov_114_182: frac[99,182] + frac[156,182] + frac[195,182] + frac[197,182]
   + frac[194,182] >= 0.9
 cov_115_239: frac[184,239] + frac[192,239] + frac[35,239] + frac[196,239]
   + frac[195,239] + frac[197,239] + frac[198,239] >= 0.7
 cov_116_181: frac[192,181] + frac[196,181] + frac[197,181] + frac[198,181]
   + frac[175,181] + frac[10,181] + frac[195,181] >= 0.9
 cov_117_155: frac[196,155] + frac[193,155] + frac[198,155] + frac[195,155]
   + frac[58,155] + frac[197,155] + frac[157,155] >= 0.8
 cov_118_190: frac[198,190] + frac[193,190] + frac[195,190] + frac[155,190]
   + frac[197,190] + frac[94,190] + frac[196,190] >= 0.8
 cov_119_248: frac[197,248] + frac[195,248] + frac[136,248] + frac[194,248]
   + frac[171,248] >= 0.7
 cov_120_32: frac[114,32] + frac[197,32] + frac[17,32] + frac[198,32]
   + frac[196,32] + frac[194,32] + frac[192,32] >= 0.7
 cov_121_12: frac[45,12] + frac[132,12] + frac[197,12] + frac[194,12]
   + frac[198,12] + frac[192,12] + frac[196,12] >= 1
 cov_122_175: frac[194,175] + frac[137,175] + frac[100,175] >= 1
 cov_123_174: frac[196,174] + frac[198,174] + frac[78,174] + frac[147,174]
   + frac[195,174] + frac[193,174] + frac[197,174] >= 0.8
 cov_124_84: frac[68,84] + frac[193,84] + frac[74,84] >= 0.9
 cov_125_137: frac[197,137] + frac[163,137] + frac[195,137] + frac[120,137]
   + frac[194,137] >= 0.9
 cov_126_24: frac[129,24] + frac[194,24] + frac[197,24] + frac[195,24]
   + frac[157,24] >= 0.7
 cov_127_251: frac[39,251] + frac[192,251] + frac[193,251] + frac[49,251]
   + frac[196,251] >= 0.7
 cov_128_160: frac[158,160] + frac[196,160] + frac[198,160] + frac[61,160]
   + frac[197,160] + frac[193,160] + frac[195,160] >= 0.8
 cov_129_189: frac[195,189] + frac[193,189] + frac[63,189] + frac[198,189]
   + frac[196,189] + frac[197,189] + frac[150,189] >= 1
 cov_130_9: frac[34,9] + frac[77,9] + frac[192,9] + frac[193,9]
   + frac[196,9] >= 0.7
 cov_131_167: frac[173,167] + frac[195,167] + frac[180,167] >= 0.9
 cov_132_5: frac[192,5] + frac[193,5] + frac[29,5] + frac[66,5]
   + frac[196,5] >= 0.9
 cov_133_184: frac[196,184] + frac[197,184] + frac[195,184] + frac[198,184]
   + frac[76,184] + frac[182,184] + frac[193,184] >= 1
 cov_134_49: frac[194,49] + frac[134,49] + frac[197,49] + frac[195,49]
   + frac[170,49] >= 0.8
 cov_135_72: frac[194,72] + frac[117,72] + frac[136,72] >= 1
 cov_136_190: frac[192,190] + frac[41,190] + frac[6,190] >= 0.9
 cov_137_204: frac[172,204] + frac[176,204] + frac[195,204] >= 0.8
 cov_138_37: frac[21,37] + frac[192,37] + frac[46,37] >= 0.7
 cov_139_116: frac[105,116] + frac[140,116] + frac[194,116] >= 0.8
 cov_140_90: frac[119,90] + frac[108,90] + frac[194,90] >= 0.7
 cov_141_147: frac[178,147] + frac[195,147] + frac[191,147] >= 0.9
 cov_142_133: frac[116,133] + frac[147,133] + frac[194,133] + frac[197,133]
   + frac[195,133] >= 1
 cov_143_2: frac[195,2] + frac[165,2] + frac[169,2] >= 0.9
 cov_144_77: frac[32,77] + frac[192,77] + frac[5,77] >= 0.8
 cov_145_177: frac[196,177] + frac[198,177] + frac[197,177] + frac[73,177]
   + frac[194,177] + frac[97,177] + frac[193,177] >= 0.8
 cov_146_175: frac[99,175] + frac[194,175] + frac[127,175] >= 1
 cov_147_35: frac[93,35] + frac[196,35] + frac[192,35] + frac[20,35]
   + frac[193,35] >= 1
 cov_148_55: frac[152,55] + frac[196,55] + frac[78,55] + frac[195,55]
   + frac[193,55] + frac[198,55] + frac[197,55] >= 1
 cov_149_254: frac[193,254] + frac[90,254] + frac[181,254] + frac[198,254]
   + frac[195,254] + frac[197,254] + frac[196,254] >= 0.7
 cov_150_92: frac[194,92] + frac[81,92] + frac[193,92] + frac[116,92]
   + frac[198,92] + frac[196,92] + frac[197,92] >= 0.9
 cov_151_12: frac[96,12] + frac[194,12] + frac[104,12] >= 0.9
 cov_152_244: frac[113,244] + frac[194,244] + frac[193,244] + frac[196,244]
   + frac[197,244] + frac[198,244] + frac[66,244] >= 0.9
 cov_153_245: frac[102,245] + frac[193,245] + frac[197,245] + frac[194,245]
   + frac[49,245] + frac[196,245] + frac[198,245] >= 0.7
 cov_154_8: frac[198,8] + frac[194,8] + frac[112,8] + frac[196,8]
   + frac[60,8] + frac[193,8] + frac[197,8] >= 0.9
 cov_155_166: frac[41,166] + frac[193,166] + frac[192,166] + frac[92,166]
   + frac[196,166] >= 0.7
 cov_156_150: frac[192,150] + frac[197,150] + frac[129,150] + frac[196,150]
   + frac[29,150] + frac[194,150] + frac[198,150] >= 0.8
 cov_157_117: frac[108,117] + frac[196,117] + frac[198,117] + frac[72,117]
   + frac[193,117] + frac[194,117] + frac[197,117] >= 0.7
 cov_158_174: frac[104,174] + frac[194,174] + frac[135,174] >= 0.8
 cov_159_17: frac[193,17] + frac[196,17] + frac[60,17] + frac[11,17]
   + frac[192,17] >= 0.8
 cov_160_142: frac[77,142] + frac[115,142] + frac[193,142] + frac[197,142]
   + frac[196,142] + frac[194,142] + frac[198,142] >= 0.7
 cov_161_224: frac[121,224] + frac[157,224] + frac[197,224] + frac[194,224]
   + frac[195,224] >= 0.7
 cov_162_126: frac[192,126] + frac[194,126] + frac[115,126] + frac[196,126]
   + frac[198,126] + frac[23,126] + frac[197,126] >= 1
 cov_163_17: frac[193,17] + frac[198,17] + frac[196,17] + frac[166,17]
   + frac[88,17] + frac[197,17] + frac[195,17] >= 1
 cov_164_99: frac[196,99] + frac[192,99] + frac[197,99] + frac[33,99]
   + frac[198,99] + frac[124,99] + frac[194,99] >= 1
 cov_165_158: frac[103,158] + frac[194,158] + frac[197,158] + frac[196,158]
   + frac[198,158] + frac[24,158] + frac[192,158] >= 1
 cov_166_187: frac[197,187] + frac[194,187] + frac[196,187] + frac[14,187]
   + frac[198,187] + frac[134,187] + frac[192,187] >= 0.9
 cov_167_125: frac[197,125] + frac[150,125] + frac[195,125] + frac[194,125]
   + frac[141,125] >= 0.8
 cov_168_224: frac[196,224] + frac[197,224] + frac[19,224] + frac[198,224]
   + frac[192,224] + frac[195,224] + frac[149,224] >= 0.9
 cov_169_142: frac[169,142] + frac[195,142] + frac[197,142] + frac[142,142]
   + frac[194,142] >= 0.7
 cov_170_187: frac[26,187] + frac[39,187] + frac[192,187] >= 0.8
 cov_171_14: frac[170,14] + frac[197,14] + frac[194,14] + frac[121,14]
   + frac[195,14] >= 0.7
 cov_172_229: frac[193,229] + frac[196,229] + frac[192,229] + frac[18,229]
   + frac[49,229] >= 0.9
 cov_173_231: frac[185,231] + frac[196,231] + frac[195,231] + frac[197,231]
   + frac[193,231] + frac[67,231] + frac[198,231] >= 0.8
 cov_174_47: frac[195,47] + frac[197,47] + frac[173,47] + frac[192,47]
   + frac[196,47] + frac[6,47] + frac[198,47] >= 0.9
 cov_175_170: frac[164,170] + frac[190,170] + frac[195,170] >= 0.7
 cov_176_22: frac[97,22] + frac[198,22] + frac[84,22] + frac[193,22]
   + frac[197,22] + frac[196,22] + frac[194,22] >= 0.9
 cov_177_47: frac[192,47] + frac[21,47] + frac[15,47] >= 1
 cov_178_247: frac[193,247] + frac[195,247] + frac[197,247] + frac[198,247]
   + frac[59,247] + frac[153,247] + frac[196,247] >= 0.8
 cov_179_39: frac[195,39] + frac[198,39] + frac[197,39] + frac[11,39]
   + frac[196,39] + frac[178,39] + frac[192,39] >= 0.9
 cov_180_11: frac[198,11] + frac[194,11] + frac[193,11] + frac[197,11]
   + frac[70,11] + frac[111,11] + frac[196,11] >= 0.8
 cov_181_136: frac[193,136] + frac[52,136] + frac[0,136] + frac[192,136]
   + frac[196,136] >= 1
 cov_182_142: frac[10,142] + frac[193,142] + frac[192,142] + frac[196,142]
   + frac[86,142] >= 0.7
 cov_183_202: frac[192,202] + frac[196,202] + frac[193,202] + frac[74,202]
   + frac[46,202] >= 0.9
 cov_184_253: frac[22,253] + frac[192,253] + frac[167,253] + frac[197,253]
   + frac[198,253] + frac[196,253] + frac[195,253] >= 1
 cov_185_157: frac[198,157] + frac[107,157] + frac[196,157] + frac[197,157]
   + frac[194,157] + frac[192,157] + frac[38,157] >= 0.7
 cov_186_111: frac[9,111] + frac[197,111] + frac[198,111] + frac[146,111]
   + frac[195,111] + frac[192,111] + frac[196,111] >= 0.9
 cov_187_50: frac[9,50] + frac[198,50] + frac[101,50] + frac[192,50]
   + frac[197,50] + frac[196,50] + frac[194,50] >= 0.9
 cov_188_105: frac[194,105] + frac[101,105] + frac[100,105] >= 0.7
 cov_189_190: frac[198,190] + frac[193,190] + frac[83,190] + frac[171,190]
   + frac[195,190] + frac[197,190] + frac[196,190] >= 0.9
 cov_190_73: frac[167,73] + frac[195,73] + frac[149,73] >= 0.7
 cov_191_166: frac[193,166] + frac[194,166] + frac[198,166] + frac[48,166]
   + frac[197,166] + frac[196,166] + frac[96,166] >= 0.8
 cov_192_101: frac[40,101] + frac[78,101] + frac[193,101] + frac[196,101]
   + frac[192,101] >= 1
 cov_193_202: frac[48,202] + frac[56,202] + frac[193,202] >= 1
 cov_194_48: frac[194,48] + frac[198,48] + frac[142,48] + frac[196,48]
   + frac[193,48] + frac[66,48] + frac[197,48] >= 1
 cov_195_202: frac[66,202] + frac[193,202] + frac[89,202] >= 0.9
 cov_196_242: frac[95,242] + frac[196,242] + frac[187,242] + frac[193,242]
   + frac[195,242] + frac[197,242] + frac[198,242] >= 1
 cov_197_156: frac[196,156] + frac[193,156] + frac[85,156] + frac[194,156]
   + frac[138,156] + frac[197,156] + frac[198,156] >= 0.8
 cov_198_107: frac[81,107] + frac[193,107] + frac[67,107] >= 1
 cov_199_2: frac[194,2] + frac[109,2] + frac[117,2] >= 1
 cov_200_223: frac[198,223] + frac[75,223] + frac[193,223] + frac[194,223]
   + frac[112,223] + frac[196,223] + frac[197,223] >= 0.9
 cov_201_124: frac[196,124] + frac[53,124] + frac[33,124] + frac[192,124]
   + frac[193,124] >= 1
 cov_202_183: frac[193,183] + frac[195,183] + frac[75,183] + frac[197,183]
   + frac[198,183] + frac[182,183] + frac[196,183] >= 0.7
 cov_203_21: frac[195,21] + frac[17,21] + frac[192,21] + frac[196,21]
   + frac[198,21] + frac[197,21] + frac[150,21] >= 1
 cov_204_41: frac[193,41] + frac[66,41] + frac[51,41] >= 0.7
 cov_205_153: frac[192,153] + frac[95,153] + frac[193,153] + frac[7,153]
   + frac[196,153] >= 1
 cov_206_56: frac[57,56] + frac[193,56] + frac[48,56] >= 0.9
 cov_207_107: frac[35,107] + frac[192,107] + frac[193,107] + frac[196,107]
   + frac[89,107] >= 1
 cov_208_63: frac[192,63] + frac[49,63] + frac[196,63] + frac[193,63]
   + frac[42,63] >= 0.7
 cov_209_201: frac[197,201] + frac[194,201] + frac[195,201] + frac[141,201]
   + frac[190,201] >= 0.9
 cov_210_143: frac[194,143] + frac[198,143] + frac[42,143] + frac[113,143]
   + frac[197,143] + frac[192,143] + frac[196,143] >= 1
 cov_211_184: frac[196,184] + frac[129,184] + frac[53,184] + frac[194,184]
   + frac[197,184] + frac[198,184] + frac[193,184] >= 1
 cov_212_113: frac[52,113] + frac[193,113] + frac[194,113] + frac[197,113]
   + frac[111,113] + frac[198,113] + frac[196,113] >= 0.7
 cov_213_126: frac[192,126] + frac[46,126] + frac[24,126] >= 0.9
 cov_214_74: frac[193,74] + frac[57,74] + frac[68,74] >= 0.9
 cov_215_227: frac[197,227] + frac[198,227] + frac[195,227] + frac[8,227]
   + frac[196,227] + frac[190,227] + frac[192,227] >= 0.9
 cov_216_145: frac[174,145] + frac[167,145] + frac[195,145] >= 0.8
 cov_217_189: frac[192,189] + frac[39,189] + frac[27,189] >= 0.7
 cov_218_4: frac[192,4] + frac[33,4] + frac[197,4] + frac[195,4]
   + frac[150,4] + frac[196,4] + frac[198,4] >= 0.7
 cov_219_210: frac[195,210] + frac[194,210] + frac[158,210] + frac[141,210]
   + frac[197,210] >= 0.9
 cov_220_121: frac[33,121] + frac[192,121] + frac[198,121] + frac[197,121]
   + frac[195,121] + frac[155,121] + frac[196,121] >= 0.8
 cov_221_252: frac[192,252] + frac[193,252] + frac[76,252] + frac[196,252]
   + frac[9,252] >= 0.7
 cov_222_13: frac[192,13] + frac[196,13] + frac[197,13] + frac[194,13]
   + frac[24,13] + frac[198,13] + frac[111,13] >= 0.9
 cov_223_145: frac[194,145] + frac[193,145] + frac[69,145] + frac[196,145]
   + frac[198,145] + frac[197,145] + frac[113,145] >= 1
 cov_224_8: frac[195,8] + frac[194,8] + frac[177,8] + frac[99,8]
   + frac[197,8] >= 1
 cov_225_222: frac[197,222] + frac[192,222] + frac[36,222] + frac[196,222]
   + frac[184,222] + frac[198,222] + frac[195,222] >= 0.7
 cov_226_24: frac[193,24] + frac[196,24] + frac[194,24] + frac[197,24]
   + frac[100,24] + frac[198,24] + frac[71,24] >= 1
 cov_227_23: frac[80,23] + frac[193,23] + frac[13,23] + frac[192,23]
   + frac[196,23] >= 1
 cov_228_117: frac[196,117] + frac[198,117] + frac[109,117] + frac[90,117]
   + frac[193,117] + frac[194,117] + frac[197,117] >= 1
 cov_229_149: frac[197,149] + frac[105,149] + frac[195,149] + frac[194,149]
   + frac[176,149] >= 0.9
 cov_230_111: frac[174,111] + frac[184,111] + frac[195,111] >= 0.7
 cov_231_220: frac[167,220] + frac[10,220] + frac[198,220] + frac[196,220]
   + frac[197,220] + frac[195,220] + frac[192,220] >= 0.8
 cov_232_219: frac[89,219] + frac[195,219] + frac[196,219] + frac[197,219]
   + frac[198,219] + frac[193,219] + frac[188,219] >= 0.8
 cov_233_67: frac[197,67] + frac[130,67] + frac[194,67] + frac[195,67]
   + frac[191,67] >= 0.8
 cov_234_51: frac[195,51] + frac[198,51] + frac[182,51] + frac[197,51]
   + frac[192,51] + frac[47,51] + frac[196,51] >= 0.9
 cov_235_33: frac[197,33] + frac[193,33] + frac[58,33] + frac[198,33]
   + frac[196,33] + frac[99,33] + frac[194,33] >= 1
 cov_236_116: frac[168,116] + frac[195,116] + frac[187,116] >= 0.7
 cov_237_36: frac[196,36] + frac[194,36] + frac[198,36] + frac[192,36]
   + frac[12,36] + frac[197,36] + frac[115,36] >= 0.8
 cov_238_165: frac[193,165] + frac[125,165] + frac[83,165] + frac[196,165]
   + frac[194,165] + frac[197,165] + frac[198,165] >= 1
 cov_239_115: frac[196,115] + frac[194,115] + frac[197,115] + frac[198,115]
   + frac[113,115] + frac[45,115] + frac[192,115] >= 0.9
 cov_240_122: frac[181,122] + frac[195,122] + frac[182,122] >= 0.9
 cov_241_143: frac[193,143] + frac[63,143] + frac[66,143] >= 0.9
 cov_242_21: frac[195,21] + frac[169,21] + frac[161,21] >= 0.9
 cov_243_122: frac[196,122] + frac[198,122] + frac[195,122] + frac[71,122]
   + frac[149,122] + frac[197,122] + frac[193,122] >= 0.9
 cov_244_87: frac[192,87] + frac[42,87] + frac[196,87] + frac[193,87]
   + frac[67,87] >= 1
 cov_245_82: frac[197,82] + frac[190,82] + frac[90,82] + frac[196,82]
   + frac[195,82] + frac[193,82] + frac[198,82] >= 1
 cov_246_111: frac[193,111] + frac[197,111] + frac[194,111] + frac[198,111]
   + frac[63,111] + frac[126,111] + frac[196,111] >= 1
 cov_247_165: frac[193,165] + frac[100,165] + frac[63,165] + frac[196,165]
   + frac[194,165] + frac[197,165] + frac[198,165] >= 0.8
 cov_248_75: frac[23,75] + frac[38,75] + frac[192,75] >= 1
 cov_249_215: frac[198,215] + frac[196,215] + frac[173,215] + frac[193,215]
   + frac[195,215] + frac[197,215] + frac[82,215] >= 0.8
 cov_250_6: frac[196,6] + frac[193,6] + frac[159,6] + frac[198,6]
   + frac[86,6] + frac[195,6] + frac[197,6] >= 0.7
 cov_251_124: frac[189,124] + frac[114,124] + frac[197,124] + frac[195,124]
   + frac[194,124] >= 0.8
 cov_252_64: frac[15,64] + frac[78,64] + frac[196,64] + frac[192,64]
   + frac[193,64] >= 1
 cov_253_18: frac[109,18] + frac[102,18] + frac[194,18] >= 0.9
 cov_254_169: frac[198,169] + frac[194,169] + frac[193,169] + frac[196,169]
   + frac[55,169] + frac[197,169] + frac[113,169] >= 0.9
 cov_255_153: frac[193,153] + frac[83,153] + frac[81,153] >= 0.8
 cov_256_130: frac[196,130] + frac[193,130] + frac[26,130] + frac[192,130]
   + frac[78,130] >= 0.7
 cov_257_1: frac[0,1] + frac[198,1] + frac[196,1] + frac[194,1]
   + frac[192,1] + frac[132,1] + frac[197,1] >= 0.7
 cov_258_32: frac[117,32] + frac[115,32] + frac[194,32] >= 0.9
 cov_259_81: frac[194,81] + frac[197,81] + frac[195,81] + frac[96,81]
   + frac[183,81] >= 0.7
 cov_260_65: frac[77,65] + frac[94,65] + frac[193,65] >= 0.8
 cov_261_233: frac[193,233] + frac[77,233] + frac[197,233] + frac[198,233]
   + frac[194,233] + frac[104,233] + frac[196,233] >= 0.9
 cov_262_188: frac[171,188] + frac[195,188] + frac[139,188] + frac[194,188]
   + frac[197,188] >= 1
 cov_263_95: frac[193,95] + frac[194,95] + frac[196,95] + frac[197,95]
   + frac[198,95] + frac[102,95] + frac[48,95] >= 0.8
 cov_264_160: frac[196,160] + frac[198,160] + frac[21,160] + frac[192,160]
   + frac[194,160] + frac[197,160] + frac[135,160] >= 0.9
 cov_265_67: frac[196,67] + frac[197,67] + frac[106,67] + frac[193,67]
   + frac[198,67] + frac[58,67] + frac[194,67] >= 0.9
 cov_266_89: frac[88,89] + frac[198,89] + frac[197,89] + frac[196,89]
   + frac[193,89] + frac[134,89] + frac[194,89] >= 0.7
 cov_267_7: frac[144,7] + frac[164,7] + frac[195,7] >= 1
 cov_268_111: frac[193,111] + frac[88,111] + frac[197,111] + frac[194,111]
   + frac[198,111] + frac[131,111] + frac[196,111] >= 1
 cov_269_142: frac[18,142] + frac[197,142] + frac[192,142] + frac[141,142]
   + frac[196,142] + frac[194,142] + frac[198,142] >= 0.9
 cov_270_153: frac[34,153] + frac[192,153] + frac[193,153] + frac[82,153]
   + frac[196,153] >= 0.9
 cov_271_183: frac[193,183] + frac[26,183] + frac[192,183] + frac[196,183]
   + frac[87,183] >= 0.9
 cov_272_62: frac[194,62] + frac[197,62] + frac[196,62] + frac[139,62]
   + frac[198,62] + frac[3,62] + frac[192,62] >= 0.7
 cov_273_10: frac[145,10] + frac[25,10] + frac[195,10] + frac[192,10]
   + frac[197,10] + frac[196,10] + frac[198,10] >= 0.9
 cov_274_242: frac[26,242] + frac[192,242] + frac[196,242] + frac[195,242]
   + frac[173,242] + frac[197,242] + frac[198,242] >= 0.7
 cov_275_222: frac[197,222] + frac[39,222] + frac[192,222] + frac[194,222]
   + frac[196,222] + frac[97,222] + frac[198,222] >= 0.8
 cov_276_12: frac[193,12] + frac[197,12] + frac[141,12] + frac[194,12]
   + frac[198,12] + frac[63,12] + frac[196,12] >= 0.9
 cov_277_121: frac[163,121] + frac[136,121] + frac[194,121] + frac[197,121]
   + frac[195,121] >= 0.8
 cov_278_31: frac[123,31] + frac[196,31] + frac[193,31] + frac[198,31]
   + frac[197,31] + frac[194,31] + frac[86,31] >= 0.9
 cov_279_214: frac[198,214] + frac[93,214] + frac[196,214] + frac[140,214]
   + frac[194,214] + frac[193,214] + frac[197,214] >= 0.8
 cov_280_96: frac[192,96] + frac[2,96] + frac[42,96] >= 0.9
 cov_281_243: frac[174,243] + frac[156,243] + frac[195,243] >= 1
 cov_282_247: frac[55,247] + frac[193,247] + frac[87,247] >= 0.9
 cov_283_142: frac[70,142] + frac[193,142] + frac[75,142] >= 0.7
 cov_284_44: frac[195,44] + frac[194,44] + frac[169,44] + frac[197,44]
   + frac[98,44] >= 0.7
 cov_285_47: frac[195,47] + frac[197,47] + frac[109,47] + frac[181,47]
   + frac[194,47] >= 0.9
 cov_286_78: frac[76,78] + frac[193,78] + frac[62,78] >= 1
 cov_287_189: frac[170,189] + frac[195,189] + frac[192,189] + frac[198,189]
   + frac[36,189] + frac[196,189] + frac[197,189] >= 0.8
 cov_288_0: frac[93,0] + frac[88,0] + frac[193,0] >= 0.9
 cov_289_124: frac[198,124] + frac[196,124] + frac[197,124] + frac[169,124]
   + frac[195,124] + frac[32,124] + frac[192,124] >= 0.7
 cov_290_235: frac[105,235] + frac[197,235] + frac[171,235] + frac[194,235]
   + frac[195,235] >= 0.7
 cov_291_168: frac[198,168] + frac[68,168] + frac[194,168] + frac[196,168]
   + frac[193,168] + frac[197,168] + frac[113,168] >= 1
 cov_292_3: frac[195,3] + frac[194,3] + frac[134,3] + frac[197,3]
   + frac[170,3] >= 1
 cov_293_156: frac[50,156] + frac[193,156] + frac[87,156] >= 0.9
 cov_294_129: frac[198,129] + frac[192,129] + frac[197,129] + frac[195,129]
   + frac[16,129] + frac[150,129] + frac[196,129] >= 0.7
 cov_295_126: frac[192,126] + frac[129,126] + frac[194,126] + frac[196,126]
   + frac[15,126] + frac[198,126] + frac[197,126] >= 0.9
 cov_296_149: frac[197,149] + frac[126,149] + frac[195,149] + frac[194,149]
   + frac[191,149] >= 0.9
 cov_297_12: frac[149,12] + frac[107,12] + frac[197,12] + frac[195,12]
   + frac[194,12] >= 1
 cov_298_239: frac[144,239] + frac[192,239] + frac[196,239] + frac[195,239]
   + frac[197,239] + frac[45,239] + frac[198,239] >= 0.9
 cov_299_165: frac[153,165] + frac[195,165] + frac[106,165] + frac[194,165]
   + frac[197,165] >= 0.9
 cov_300_159: frac[192,159] + frac[1,159] + frac[43,159] >= 0.9
 cov_301_162: frac[60,162] + frac[196,162] + frac[193,162] + frac[192,162]
   + frac[16,162] >= 0.8
 cov_302_247: frac[84,247] + frac[193,247] + frac[29,247] + frac[192,247]
   + frac[196,247] >= 0.8
 cov_303_188: frac[192,188] + frac[4,188] + frac[198,188] + frac[196,188]
   + frac[122,188] + frac[194,188] + frac[197,188] >= 1
 cov_304_35: frac[113,35] + frac[137,35] + frac[194,35] >= 0.9
 cov_305_91: frac[94,91] + frac[66,91] + frac[193,91] >= 0.8
 cov_306_148: frac[198,148] + frac[72,148] + frac[182,148] + frac[193,148]
   + frac[197,148] + frac[195,148] + frac[196,148] >= 0.8
 cov_307_19: frac[197,19] + frac[116,19] + frac[194,19] + frac[195,19]
   + frac[147,19] >= 0.8
 cov_308_2: frac[198,2] + frac[194,2] + frac[19,2] + frac[197,2]
   + frac[196,2] + frac[192,2] + frac[111,2] >= 0.7
 cov_309_38: frac[87,38] + frac[193,38] + frac[72,38] >= 0.7
 cov_310_200: frac[192,200] + frac[2,200] + frac[8,200] >= 1
 cov_311_153: frac[192,153] + frac[54,153] + frac[30,153] + frac[193,153]
   + frac[196,153] >= 1
 cov_312_90: frac[192,90] + frac[31,90] + frac[45,90] >= 0.8
 cov_313_246: frac[194,246] + frac[7,246] + frac[197,246] + frac[198,246]
   + frac[122,246] + frac[192,246] + frac[196,246] >= 0.9
 cov_314_1: frac[186,1] + frac[152,1] + frac[195,1] >= 0.7
 cov_315_254: frac[183,254] + frac[195,254] + frac[178,254] >= 0.9
 cov_316_70: frac[195,70] + frac[115,70] + frac[197,70] + frac[194,70]
   + frac[166,70] >= 1
 cov_317_192: frac[197,192] + frac[153,192] + frac[194,192] + frac[195,192]
   + frac[129,192] >= 1
 cov_318_252: frac[194,252] + frac[31,252] + frac[192,252] + frac[198,252]
   + frac[140,252] + frac[196,252] + frac[197,252] >= 0.8
 cov_319_99: frac[196,99] + frac[134,99] + frac[197,99] + frac[193,99]
   + frac[73,99] + frac[198,99] + frac[194,99] >= 0.9
 cov_320_240: frac[197,240] + frac[125,240] + frac[195,240] + frac[194,240]
   + frac[172,240] >= 1
 cov_321_166: frac[22,166] + frac[192,166] + frac[18,166] >= 0.9
 cov_322_213: frac[197,213] + frac[194,213] + frac[192,213] + frac[1,213]
   + frac[196,213] + frac[104,213] + frac[198,213] >= 0.8
 cov_323_38: frac[43,38] + frac[197,38] + frac[196,38] + frac[192,38]
   + frac[119,38] + frac[194,38] + frac[198,38] >= 0.9
 cov_324_148: frac[119,148] + frac[164,148] + frac[197,148] + frac[194,148]
   + frac[195,148] >= 0.8
 cov_325_85: frac[63,85] + frac[193,85] + frac[97,85] + frac[194,85]
   + frac[197,85] + frac[196,85] + frac[198,85] >= 1
 cov_326_25: frac[12,25] + frac[104,25] + frac[196,25] + frac[197,25]
   + frac[194,25] + frac[192,25] + frac[198,25] >= 0.9
 cov_327_46: frac[68,46] + frac[90,46] + frac[193,46] >= 1
 cov_328_212: frac[182,212] + frac[195,212] + frac[154,212] >= 0.9
 cov_329_31: frac[196,31] + frac[72,31] + frac[193,31] + frac[198,31]
   + frac[100,31] + frac[197,31] + frac[194,31] >= 0.7
 cov_330_140: frac[185,140] + frac[195,140] + frac[194,140] + frac[131,140]
   + frac[197,140] >= 1
 cov_331_6: frac[194,6] + frac[137,6] + frac[138,6] >= 0.7
 cov_332_34: frac[196,34] + frac[193,34] + frac[192,34] + frac[52,34]
   + frac[46,34] >= 1
 cov_333_54: frac[179,54] + frac[192,54] + frac[197,54] + frac[196,54]
   + frac[198,54] + frac[195,54] + frac[10,54] >= 0.9
 cov_334_70: frac[193,70] + frac[60,70] + frac[196,70] + frac[100,70]
   + frac[197,70] + frac[194,70] + frac[198,70] >= 0.7
 cov_335_90: frac[197,90] + frac[68,90] + frac[148,90] + frac[193,90]
   + frac[196,90] + frac[198,90] + frac[195,90] >= 0.9
 cov_336_221: frac[172,221] + frac[197,221] + frac[195,221] + frac[130,221]
   + frac[194,221] >= 0.9
 cov_337_126: frac[194,126] + frac[123,126] + frac[156,126] + frac[195,126]
   + frac[197,126] >= 0.9
 cov_338_132: frac[195,132] + frac[194,132] + frac[139,132] + frac[197,132]
   + frac[150,132] >= 0.9
 cov_339_20: frac[195,20] + frac[41,20] + frac[192,20] + frac[196,20]
   + frac[198,20] + frac[159,20] + frac[197,20] >= 0.8
 cov_340_227: frac[169,227] + frac[158,227] + frac[195,227] >= 0.7
 cov_341_182: frac[198,182] + frac[193,182] + frac[108,182] + frac[197,182]
   + frac[92,182] + frac[196,182] + frac[194,182] >= 0.7
 cov_342_46: frac[99,46] + frac[1,46] + frac[197,46] + frac[192,46]
   + frac[196,46] + frac[194,46] + frac[198,46] >= 0.9
 cov_343_16: frac[198,16] + frac[196,16] + frac[193,16] + frac[50,16]
   + frac[197,16] + frac[195,16] + frac[176,16] >= 0.9
 cov_344_228: frac[194,228] + frac[198,228] + frac[193,228] + frac[100,228]
   + frac[196,228] + frac[64,228] + frac[197,228] >= 0.9
 cov_345_52: frac[193,52] + frac[75,52] + frac[27,52] + frac[192,52]
   + frac[196,52] >= 0.7
 cov_346_191: frac[195,191] + frac[176,191] + frac[178,191] >= 1
 cov_347_90: frac[4,90] + frac[192,90] + frac[197,90] + frac[179,90]
   + frac[196,90] + frac[198,90] + frac[195,90] >= 0.7
 cov_348_159: frac[192,159] + frac[139,159] + frac[197,159] + frac[10,159]
   + frac[194,159] + frac[196,159] + frac[198,159] >= 0.7
 cov_349_150: frac[192,150] + frac[31,150] + frac[15,150] >= 0.7
 cov_350_83: frac[198,83] + frac[65,83] + frac[193,83] + frac[196,83]
   + frac[194,83] + frac[143,83] + frac[197,83] >= 1
 cov_351_189: frac[194,189] + frac[193,189] + frac[198,189] + frac[52,189]
   + frac[196,189] + frac[197,189] + frac[98,189] >= 0.8
 cov_352_100: frac[30,100] + frac[192,100] + frac[28,100] >= 0.8
 cov_353_148: frac[198,148] + frac[170,148] + frac[197,148] + frac[16,148]
   + frac[195,148] + frac[192,148] + frac[196,148] >= 1
 cov_354_119: frac[44,119] + frac[192,119] + frac[55,119] + frac[196,119]
   + frac[193,119] >= 0.8
 cov_355_248: frac[198,248] + frac[196,248] + frac[197,248] + frac[195,248]
   + frac[53,248] + frac[164,248] + frac[193,248] >= 1
 cov_356_109: frac[119,109] + frac[197,109] + frac[194,109] + frac[198,109]
   + frac[192,109] + frac[196,109] + frac[2,109] >= 1
 cov_357_142: frac[4,142] + frac[192,142] + frac[21,142] >= 0.9
 cov_358_159: frac[193,159] + frac[61,159] + frac[95,159] >= 0.7
 cov_359_121: frac[81,121] + frac[193,121] + frac[198,121] + frac[197,121]
   + frac[195,121] + frac[166,121] + frac[196,121] >= 0.9
 cov_360_36: frac[196,36] + frac[198,36] + frac[57,36] + frac[197,36]
   + frac[147,36] + frac[195,36] + frac[193,36] >= 1
 cov_361_34: frac[196,34] + frac[40,34] + frac[192,34] + frac[198,34]
   + frac[194,34] + frac[143,34] + frac[197,34] >= 0.8
 cov_362_72: frac[170,72] + frac[195,72] + frac[148,72] >= 0.8
 cov_363_3: frac[109,3] + frac[194,3] + frac[117,3] >= 1
 cov_364_134: frac[195,134] + frac[156,134] + frac[151,134] >= 0.9
 cov_365_99: frac[195,99] + frac[196,99] + frac[192,99] + frac[197,99]
   + frac[175,99] + frac[198,99] + frac[1,99] >= 0.8
 cov_366_122: frac[196,122] + frac[198,122] + frac[106,122] + frac[197,122]
   + frac[74,122] + frac[194,122] + frac[193,122] >= 0.9
 cov_367_163: frac[193,163] + frac[73,163] + frac[196,163] + frac[192,163]
   + frac[40,163] >= 0.9
 cov_368_249: frac[83,249] + frac[197,249] + frac[198,249] + frac[196,249]
   + frac[193,249] + frac[194,249] + frac[142,249] >= 0.9
 cov_369_157: frac[198,157] + frac[196,157] + frac[197,157] + frac[8,157]
   + frac[194,157] + frac[192,157] + frac[139,157] >= 1
 cov_370_128: frac[79,128] + frac[193,128] + frac[196,128] + frac[192,128]
   + frac[8,128] >= 0.9
 cov_371_175: frac[61,175] + frac[196,175] + frac[194,175] + frac[132,175]
   + frac[198,175] + frac[193,175] + frac[197,175] >= 0.8
 cov_372_113: frac[149,113] + frac[195,113] + frac[163,113] >= 0.9
 cov_373_8: frac[100,8] + frac[9,8] + frac[198,8] + frac[194,8]
   + frac[192,8] + frac[196,8] + frac[197,8] >= 0.7
 cov_374_214: frac[59,214] + frac[50,214] + frac[193,214] >= 1
 cov_375_168: frac[152,168] + frac[153,168] + frac[195,168] >= 0.9
 cov_376_185: frac[166,185] + frac[195,185] + frac[168,185] >= 1
 cov_377_7: frac[193,7] + frac[142,7] + frac[196,7] + frac[194,7]
   + frac[83,7] + frac[198,7] + frac[197,7] >= 0.8
 cov_378_16: frac[193,16] + frac[66,16] + frac[77,16] >= 0.7
 cov_379_98: frac[85,98] + frac[196,98] + frac[195,98] + frac[151,98]
   + frac[198,98] + frac[197,98] + frac[193,98] >= 0.9
 cov_380_119: frac[198,119] + frac[107,119] + frac[88,119] + frac[196,119]
   + frac[193,119] + frac[194,119] + frac[197,119] >= 0.8
 cov_381_208: frac[23,208] + frac[192,208] + frac[2,208] >= 1
 cov_382_7: frac[146,7] + frac[152,7] + frac[195,7] >= 1
 cov_383_252: frac[52,252] + frac[194,252] + frac[193,252] + frac[125,252]
   + frac[198,252] + frac[196,252] + frac[197,252] >= 1
 cov_384_240: frac[196,240] + frac[197,240] + frac[198,240] + frac[195,240]
   + frac[193,240] + frac[80,240] + frac[158,240] >= 0.9
 cov_385_152: frac[178,152] + frac[194,152] + frac[197,152] + frac[195,152]
   + frac[119,152] >= 0.7
 cov_386_167: frac[196,167] + frac[112,167] + frac[194,167] + frac[197,167]
   + frac[65,167] + frac[198,167] + frac[193,167] >= 0.8
 cov_387_163: frac[194,163] + frac[197,163] + frac[143,163] + frac[175,163]
   + frac[195,163] >= 1
 cov_388_229: frac[195,229] + frac[194,229] + frac[98,229] + frac[157,229]
   + frac[197,229] >= 0.8
 cov_389_215: frac[174,215] + frac[195,215] + frac[120,215] + frac[197,215]
   + frac[194,215] >= 1
 cov_390_19: frac[197,19] + frac[196,19] + frac[88,19] + frac[195,19]
   + frac[193,19] + frac[198,19] + frac[161,19] >= 0.7
 cov_391_34: frac[196,34] + frac[192,34] + frac[12,34] + frac[198,34]
   + frac[195,34] + frac[197,34] + frac[182,34] >= 0.8
 cov_392_246: frac[194,246] + frac[197,246] + frac[198,246] + frac[130,246]
   + frac[192,246] + frac[36,246] + frac[196,246] >= 0.9
 cov_393_246: frac[194,246] + frac[103,246] + frac[187,246] + frac[195,246]
   + frac[197,246] >= 1
 cov_394_28: frac[192,28] + frac[23,28] + frac[196,28] + frac[193,28]
   + frac[58,28] >= 0.7
 cov_395_253: frac[193,253] + frac[72,253] + frac[197,253] + frac[169,253]
   + frac[198,253] + frac[196,253] + frac[195,253] >= 0.9
 cov_396_244: frac[47,244] + frac[192,244] + frac[2,244] >= 1
 cov_397_245: frac[133,245] + frac[194,245] + frac[103,245] >= 0.9
 cov_398_158: frac[194,158] + frac[197,158] + frac[196,158] + frac[110,158]
   + frac[9,158] + frac[198,158] + frac[192,158] >= 0.8
 cov_399_73: frac[193,73] + frac[198,73] + frac[57,73] + frac[197,73]
   + frac[195,73] + frac[160,73] + frac[196,73] >= 1
 cov_400_211: frac[21,211] + frac[33,211] + frac[192,211] >= 0.7
 cov_401_153: frac[195,153] + frac[176,153] + frac[192,153] + frac[198,153]
   + frac[47,153] + frac[196,153] + frac[197,153] >= 1
 cov_402_83: frac[198,83] + frac[95,83] + frac[139,83] + frac[193,83]
   + frac[196,83] + frac[194,83] + frac[197,83] >= 0.9
 cov_403_146: frac[32,146] + frac[193,146] + frac[192,146] + frac[62,146]
   + frac[196,146] >= 0.8
 cov_404_137: frac[140,137] + frac[103,137] + frac[194,137] >= 0.8
 cov_405_221: frac[192,221] + frac[197,221] + frac[198,221] + frac[196,221]
   + frac[138,221] + frac[16,221] + frac[194,221] >= 0.8
 cov_406_203: frac[87,203] + frac[140,203] + frac[193,203] + frac[196,203]
   + frac[197,203] + frac[198,203] + frac[194,203] >= 1
 cov_407_8: frac[74,8] + frac[193,8] + frac[90,8] >= 0.7
 cov_408_217: frac[193,217] + frac[60,217] + frac[84,217] >= 0.9
 cov_409_184: frac[196,184] + frac[61,184] + frac[192,184] + frac[14,184]
   + frac[193,184] >= 0.7
 cov_410_158: frac[139,158] + frac[194,158] + frac[197,158] + frac[196,158]
   + frac[30,158] + frac[198,158] + frac[192,158] >= 0.9
 cov_411_114: frac[182,114] + frac[195,114] + frac[194,114] + frac[109,114]
   + frac[197,114] >= 0.7
 cov_412_200: frac[159,200] + frac[195,200] + frac[165,200] >= 1
 cov_413_158: frac[49,158] + frac[193,158] + frac[61,158] >= 0.8
 cov_414_130: frac[197,130] + frac[196,130] + frac[193,130] + frac[160,130]
   + frac[198,130] + frac[195,130] + frac[82,130] >= 1
 cov_415_181: frac[156,181] + frac[186,181] + frac[195,181] >= 0.7
 cov_416_134: frac[196,134] + frac[53,134] + frac[41,134] + frac[193,134]
   + frac[192,134] >= 0.8
 cov_417_2: frac[198,2] + frac[195,2] + frac[148,2] + frac[197,2]
   + frac[72,2] + frac[196,2] + frac[193,2] >= 0.8
 cov_418_71: frac[196,71] + frac[197,71] + frac[134,71] + frac[71,71]
   + frac[193,71] + frac[194,71] + frac[198,71] >= 0.8
 cov_419_70: frac[193,70] + frac[196,70] + frac[116,70] + frac[197,70]
   + frac[194,70] + frac[198,70] + frac[68,70] >= 1
 cov_420_107: frac[65,107] + frac[193,107] + frac[55,107] >= 0.7
 cov_421_207: frac[195,207] + frac[197,207] + frac[194,207] + frac[106,207]
   + frac[146,207] >= 0.9
 cov_422_93: frac[196,93] + frac[59,93] + frac[193,93] + frac[126,93]
   + frac[194,93] + frac[197,93] + frac[198,93] >= 0.8
 cov_423_48: frac[192,48] + frac[9,48] + frac[198,48] + frac[196,48]
   + frac[197,48] + frac[148,48] + frac[195,48] >= 0.8
 cov_424_59: frac[196,59] + frac[193,59] + frac[192,59] + frac[55,59]
   + frac[41,59] >= 1
 cov_425_236: frac[196,236] + frac[197,236] + frac[9,236] + frac[107,236]
   + frac[194,236] + frac[198,236] + frac[192,236] >= 0.8
 cov_426_242: frac[192,242] + frac[5,242] + frac[196,242] + frac[124,242]
   + frac[194,242] + frac[197,242] + frac[198,242] >= 0.9
 cov_427_164: frac[58,164] + frac[193,164] + frac[64,164] >= 0.9
 cov_428_174: frac[192,174] + frac[196,174] + frac[73,174] + frac[11,174]
   + frac[193,174] >= 0.9
 cov_429_6: frac[196,6] + frac[73,6] + frac[182,6] + frac[193,6]
   + frac[198,6] + frac[195,6] + frac[197,6] >= 0.8
 cov_430_89: frac[192,89] + frac[172,89] + frac[198,89] + frac[197,89]
   + frac[36,89] + frac[195,89] + frac[196,89] >= 0.7
 cov_431_232: frac[133,232] + frac[198,232] + frac[193,232] + frac[194,232]
   + frac[196,232] + frac[197,232] + frac[62,232] >= 1
 cov_432_38: frac[197,38] + frac[196,38] + frac[193,38] + frac[67,38]
   + frac[194,38] + frac[198,38] + frac[141,38] >= 0.9
 cov_433_35: frac[176,35] + frac[195,35] + frac[160,35] >= 0.7
 cov_434_68: frac[192,68] + frac[44,68] + frac[19,68] >= 0.7
 cov_435_182: frac[157,182] + frac[198,182] + frac[195,182] + frac[192,182]
   + frac[197,182] + frac[196,182] + frac[47,182] >= 0.9
 cov_436_168: frac[194,168] + frac[197,168] + frac[178,168] + frac[98,168]
   + frac[195,168] >= 0.7
 cov_437_127: frac[193,127] + frac[194,127] + frac[83,127] + frac[196,127]
   + frac[197,127] + frac[112,127] + frac[198,127] >= 0.9
 cov_438_57: frac[17,57] + frac[192,57] + frac[193,57] + frac[53,57]
   + frac[196,57] >= 1
 cov_439_10: frac[193,10] + frac[192,10] + frac[58,10] + frac[196,10]
   + frac[29,10] >= 1
 cov_440_197: frac[77,197] + frac[197,197] + frac[193,197] + frac[196,197]
   + frac[195,197] + frac[198,197] + frac[176,197] >= 0.8
 cov_441_240: frac[196,240] + frac[197,240] + frac[198,240] + frac[174,240]
   + frac[28,240] + frac[195,240] + frac[192,240] >= 0.8
 cov_442_242: frac[192,242] + frac[196,242] + frac[22,242] + frac[193,242]
   + frac[69,242] >= 0.9
 cov_443_69: frac[195,69] + frac[193,69] + frac[196,69] + frac[198,69]
   + frac[145,69] + frac[197,69] + frac[59,69] >= 1
 cov_444_122: frac[196,122] + frac[198,122] + frac[195,122] + frac[175,122]
   + frac[29,122] + frac[192,122] + frac[197,122] >= 0.7
 cov_445_246: frac[194,246] + frac[193,246] + frac[197,246] + frac[51,246]
   + frac[198,246] + frac[114,246] + frac[196,246] >= 0.9
 cov_446_20: frac[105,20] + frac[194,20] + frac[120,20] >= 0.7
 cov_447_23: frac[194,23] + frac[48,23] + frac[114,23] + frac[193,23]
   + frac[198,23] + frac[197,23] + frac[196,23] >= 0.9
 cov_448_194: frac[55,194] + frac[196,194] + frac[47,194] + frac[192,194]
   + frac[193,194] >= 0.8
 cov_449_185: frac[196,185] + frac[185,185] + frac[197,185] + frac[195,185]
   + frac[73,185] + frac[193,185] + frac[198,185] >= 0.8
 cov_450_73: frac[91,73] + frac[146,73] + frac[193,73] + frac[198,73]
   + frac[197,73] + frac[195,73] + frac[196,73] >= 0.9
 cov_451_93: frac[196,93] + frac[193,93] + frac[58,93] + frac[197,93]
   + frac[198,93] + frac[157,93] + frac[195,93] >= 0.7
 cov_452_93: frac[196,93] + frac[44,93] + frac[194,93] + frac[192,93]
   + frac[197,93] + frac[198,93] + frac[137,93] >= 0.7
 cov_453_47: frac[121,47] + frac[197,47] + frac[192,47] + frac[196,47]
   + frac[194,47] + frac[15,47] + frac[198,47] >= 1
 cov_454_247: frac[193,247] + frac[49,247] + frac[5,247] + frac[192,247]
   + frac[196,247] >= 1
 cov_455_190: frac[189,190] + frac[198,190] + frac[193,190] + frac[195,190]
   + frac[197,190] + frac[196,190] + frac[81,190] >= 0.8
 cov_456_245: frac[133,245] + frac[193,245] + frac[197,245] + frac[194,245]
   + frac[63,245] + frac[196,245] + frac[198,245] >= 0.8
 cov_457_243: frac[89,243] + frac[193,243] + frac[94,243] >= 0.7
 cov_458_141: frac[194,141] + frac[198,141] + frac[192,141] + frac[196,141]
   + frac[11,141] + frac[197,141] + frac[132,141] >= 0.9
 cov_459_199: frac[100,199] + frac[197,199] + frac[193,199] + frac[95,199]
   + frac[194,199] + frac[198,199] + frac[196,199] >= 1
 cov_460_204: frac[20,204] + frac[198,204] + frac[192,204] + frac[196,204]
   + frac[188,204] + frac[197,204] + frac[195,204] >= 0.8
 cov_461_212: frac[182,212] + frac[197,212] + frac[136,212] + frac[195,212]
   + frac[194,212] >= 1
 cov_462_207: frac[192,207] + frac[196,207] + frac[96,207] + frac[11,207]
   + frac[197,207] + frac[194,207] + frac[198,207] >= 1
 cov_463_27: frac[192,27] + frac[11,27] + frac[45,27] >= 1
 cov_464_126: frac[102,126] + frac[192,126] + frac[194,126] + frac[25,126]
   + frac[196,126] + frac[198,126] + frac[197,126] >= 1
 cov_465_190: frac[198,190] + frac[18,190] + frac[192,190] + frac[194,190]
   + frac[197,190] + frac[196,190] + frac[139,190] >= 0.7
 cov_466_164: frac[197,164] + frac[2,164] + frac[196,164] + frac[194,164]
   + frac[198,164] + frac[109,164] + frac[192,164] >= 0.7
 cov_467_254: frac[193,254] + frac[175,254] + frac[198,254] + frac[91,254]
   + frac[195,254] + frac[197,254] + frac[196,254] >= 0.7
 cov_468_216: frac[192,216] + frac[193,216] + frac[85,216] + frac[196,216]
   + frac[15,216] >= 1
 cov_469_143: frac[193,143] + frac[14,143] + frac[192,143] + frac[196,143]
   + frac[55,143] >= 0.9
 cov_470_16: frac[196,16] + frac[193,16] + frac[5,16] + frac[81,16]
   + frac[192,16] >= 0.7
 cov_471_128: frac[12,128] + frac[192,128] + frac[27,128] >= 0.9
 cov_472_195: frac[194,195] + frac[195,195] + frac[134,195] + frac[197,195]
   + frac[170,195] >= 0.7
 cov_473_2: frac[194,2] + frac[195,2] + frac[142,2] + frac[156,2]
   + frac[197,2] >= 0.7
 cov_474_195: frac[18,195] + frac[88,195] + frac[196,195] + frac[192,195]
   + frac[193,195] >= 0.7
 cov_475_196: frac[1,196] + frac[196,196] + frac[91,196] + frac[192,196]
   + frac[193,196] >= 1
 cov_476_206: frac[194,206] + frac[195,206] + frac[197,206] + frac[177,206]
   + frac[110,206] >= 0.8
 cov_477_41: frac[193,41] + frac[196,41] + frac[197,41] + frac[195,41]
   + frac[51,41] + frac[198,41] + frac[188,41] >= 1
 cov_478_61: frac[143,61] + frac[156,61] + frac[197,61] + frac[194,61]
   + frac[195,61] >= 0.8
 cov_479_114: frac[124,114] + frac[195,114] + frac[194,114] + frac[197,114]
   + frac[159,114] >= 1
 cov_480_66: frac[198,66] + frac[197,66] + frac[166,66] + frac[196,66]
   + frac[10,66] + frac[195,66] + frac[192,66] >= 0.8
 cov_481_126: frac[169,126] + frac[190,126] + frac[195,126] >= 0.9
 cov_482_24: frac[150,24] + frac[193,24] + frac[196,24] + frac[197,24]
   + frac[195,24] + frac[57,24] + frac[198,24] >= 0.9
 cov_483_92: frac[53,92] + frac[194,92] + frac[193,92] + frac[198,92]
   + frac[196,92] + frac[197,92] + frac[100,92] >= 1
 cov_484_131: frac[83,131] + frac[54,131] + frac[193,131] >= 0.7
 cov_485_154: frac[75,154] + frac[193,154] + frac[192,154] + frac[29,154]
   + frac[196,154] >= 0.7
 cov_486_227: frac[89,227] + frac[193,227] + frac[94,227] >= 0.7
 cov_487_92: frac[22,92] + frac[192,92] + frac[27,92] >= 0.9
 cov_488_102: frac[197,102] + frac[194,102] + frac[198,102] + frac[49,102]
   + frac[193,102] + frac[196,102] + frac[133,102] >= 0.8
 cov_489_148: frac[193,148] + frac[86,148] + frac[81,148] >= 0.9
 cov_490_131: frac[59,131] + frac[196,131] + frac[150,131] + frac[195,131]
   + frac[198,131] + frac[197,131] + frac[193,131] >= 0.7
 cov_491_127: frac[39,127] + frac[192,127] + frac[194,127] + frac[118,127]
   + frac[196,127] + frac[197,127] + frac[198,127] >= 0.8
 cov_492_45: frac[98,45] + frac[31,45] + frac[194,45] + frac[198,45]
   + frac[192,45] + frac[196,45] + frac[197,45] >= 0.8
 cov_493_144: frac[57,144] + frac[92,144] + frac[193,144] >= 0.9
 cov_494_58: frac[192,58] + frac[35,58] + frac[174,58] + frac[195,58]
   + frac[197,58] + frac[196,58] + frac[198,58] >= 0.8
 cov_495_30: frac[198,30] + frac[27,30] + frac[192,30] + frac[195,30]
   + frac[162,30] + frac[197,30] + frac[196,30] >= 0.7
 cov_496_211: frac[195,211] + frac[196,211] + frac[197,211] + frac[198,211]
   + frac[41,211] + frac[152,211] + frac[192,211] >= 1
 cov_497_133: frac[196,133] + frac[145,133] + frac[197,133] + frac[198,133]
   + frac[193,133] + frac[195,133] + frac[93,133] >= 0.8
 cov_498_101: frac[197,101] + frac[195,101] + frac[196,101] + frac[198,101]
   + frac[192,101] + frac[187,101] + frac[27,101] >= 0.7
 cov_499_193: frac[195,193] + frac[175,193] + frac[196,193] + frac[198,193]
   + frac[192,193] + frac[4,193] + frac[197,193] >= 0.7
 cov_500_45: frac[120,45] + frac[194,45] + frac[149,45] + frac[197,45]
   + frac[195,45] >= 0.7
 cov_501_215: frac[198,215] + frac[13,215] + frac[192,215] + frac[196,215]
   + frac[197,215] + frac[194,215] + frac[143,215] >= 0.7
 cov_502_206: frac[193,206] + frac[18,206] + frac[60,206] + frac[192,206]
   + frac[196,206] >= 1
 cov_503_64: frac[157,64] + frac[195,64] + frac[164,64] >= 0.8
 cov_504_14: frac[27,14] + frac[192,14] + frac[2,14] >= 1
 cov_505_71: frac[102,71] + frac[195,71] + frac[197,71] + frac[158,71]
   + frac[194,71] >= 0.7
 cov_506_240: frac[77,240] + frac[193,240] + frac[57,240] >= 0.8
 cov_507_179: frac[66,179] + frac[193,179] + frac[75,179] >= 0.9
 cov_508_72: frac[146,72] + frac[194,72] + frac[96,72] + frac[197,72]
   + frac[195,72] >= 0.7
 cov_509_11: frac[65,11] + frac[193,11] + frac[75,11] >= 0.7
 cov_510_170: frac[195,170] + frac[188,170] + frac[155,170] >= 1
 cov_511_59: frac[196,59] + frac[193,59] + frac[198,59] + frac[167,59]
   + frac[195,59] + frac[82,59] + frac[197,59] >= 0.9
 cov_512_186: frac[194,186] + frac[197,186] + frac[196,186] + frac[198,186]
   + frac[193,186] + frac[82,186] + frac[116,186] >= 0.8
 cov_513_120: frac[193,120] + frac[88,120] + frac[198,120] + frac[149,120]
   + frac[196,120] + frac[197,120] + frac[195,120] >= 0.7
 cov_514_220: frac[198,220] + frac[196,220] + frac[197,220] + frac[35,220]
   + frac[129,220] + frac[194,220] + frac[192,220] >= 1
 cov_515_36: frac[56,36] + frac[196,36] + frac[198,36] + frac[151,36]
   + frac[197,36] + frac[195,36] + frac[193,36] >= 0.9
 cov_516_167: frac[196,167] + frac[194,167] + frac[197,167] + frac[198,167]
   + frac[127,167] + frac[54,167] + frac[193,167] >= 0.9
 cov_517_220: frac[9,220] + frac[196,220] + frac[55,220] + frac[192,220]
   + frac[193,220] >= 1
 cov_518_84: frac[90,84] + frac[194,84] + frac[193,84] + frac[196,84]
   + frac[197,84] + frac[198,84] + frac[120,84] >= 0.7
 cov_519_16: frac[168,16] + frac[197,16] + frac[195,16] + frac[194,16]
   + frac[128,16] >= 1
 cov_520_20: frac[195,20] + frac[179,20] + frac[192,20] + frac[196,20]
   + frac[198,20] + frac[23,20] + frac[197,20] >= 1
 cov_521_143: frac[45,143] + frac[193,143] + frac[93,143] + frac[192,143]
   + frac[196,143] >= 0.8
 cov_522_231: frac[46,231] + frac[192,231] + frac[16,231] >= 0.7
 cov_523_66: frac[39,66] + frac[42,66] + frac[192,66] >= 0.9
 cov_524_227: frac[197,227] + frac[193,227] + frac[67,227] + frac[198,227]
   + frac[194,227] + frac[196,227] + frac[122,227] >= 0.8
 cov_525_146: frac[193,146] + frac[53,146] + frac[192,146] + frac[41,146]
   + frac[196,146] >= 0.7
 cov_526_78: frac[126,78] + frac[187,78] + frac[194,78] + frac[195,78]
   + frac[197,78] >= 0.8
 cov_527_132: frac[195,132] + frac[158,132] + frac[166,132] >= 1
 cov_528_75: frac[42,75] + frac[196,75] + frac[198,75] + frac[197,75]
   + frac[155,75] + frac[195,75] + frac[192,75] >= 1
 cov_529_26: frac[126,26] + frac[194,26] + frac[130,26] >= 0.9
 cov_530_179: frac[194,179] + frac[106,179] + frac[134,179] >= 0.8
 cov_531_130: frac[196,130] + frac[70,130] + frac[193,130] + frac[192,130]
   + frac[39,130] >= 0.9
 cov_532_6: frac[196,6] + frac[193,6] + frac[172,6] + frac[198,6]
   + frac[55,6] + frac[195,6] + frac[197,6] >= 0.7
 cov_533_132: frac[196,132] + frac[192,132] + frac[20,132] + frac[90,132]
   + frac[193,132] >= 0.9
 cov_534_191: frac[96,191] + frac[195,191] + frac[197,191] + frac[146,191]
   + frac[194,191] >= 1
 cov_535_147: frac[189,147] + frac[193,147] + frac[195,147] + frac[198,147]
   + frac[197,147] + frac[52,147] + frac[196,147] >= 0.8
 cov_536_165: frac[137,165] + frac[194,165] + frac[99,165] >= 1
 cov_537_213: frac[192,213] + frac[196,213] + frac[66,213] + frac[47,213]
   + frac[193,213] >= 0.9
 cov_538_216: frac[60,216] + frac[193,216] + frac[82,216] >= 0.7
 cov_539_191: frac[198,191] + frac[22,191] + frac[192,191] + frac[197,191]
   + frac[133,191] + frac[194,191] + frac[196,191] >= 0.8
 cov_540_44: frac[195,44] + frac[198,44] + frac[196,44] + frac[145,44]
   + frac[79,44] + frac[193,44] + frac[197,44] >= 0.8
 cov_541_130: frac[197,130] + frac[196,130] + frac[192,130] + frac[190,130]
   + frac[16,130] + frac[198,130] + frac[195,130] >= 0.9
 cov_542_9: frac[198,9] + frac[166,9] + frac[86,9] + frac[193,9]
   + frac[196,9] + frac[197,9] + frac[195,9] >= 1
 cov_543_73: frac[144,73] + frac[198,73] + frac[197,73] + frac[195,73]
   + frac[192,73] + frac[17,73] + frac[196,73] >= 0.9
 cov_544_116: frac[56,116] + frac[92,116] + frac[193,116] >= 0.8
 cov_545_188: frac[119,188] + frac[120,188] + frac[194,188] >= 0.7
 cov_546_87: frac[149,87] + frac[195,87] + frac[194,87] + frac[197,87]
   + frac[101,87] >= 0.8
 cov_547_222: frac[197,222] + frac[62,222] + frac[123,222] + frac[194,222]
   + frac[196,222] + frac[193,222] + frac[198,222] >= 0.9
 cov_548_27: frac[197,27] + frac[189,27] + frac[70,27] + frac[196,27]
   + frac[193,27] + frac[198,27] + frac[195,27] >= 1
 cov_549_249: frac[197,249] + frac[198,249] + frac[8,249] + frac[196,249]
   + frac[192,249] + frac[194,249] + frac[97,249] >= 0.8
 cov_550_188: frac[192,188] + frac[195,188] + frac[198,188] + frac[22,188]
   + frac[196,188] + frac[183,188] + frac[197,188] >= 1
 cov_551_53: frac[196,53] + frac[198,53] + frac[197,53] + frac[195,53]
   + frac[168,53] + frac[0,53] + frac[192,53] >= 0.7
 cov_552_82: frac[39,82] + frac[192,82] + frac[35,82] >= 0.9
 cov_553_175: frac[61,175] + frac[196,175] + frac[198,175] + frac[195,175]
   + frac[193,175] + frac[197,175] + frac[188,175] >= 0.8
 cov_554_10: frac[120,10] + frac[98,10] + frac[194,10] >= 0.9
 cov_555_226: frac[197,226] + frac[198,226] + frac[194,226] + frac[98,226]
   + frac[23,226] + frac[196,226] + frac[192,226] >= 0.9
 cov_556_176: frac[40,176] + frac[172,176] + frac[197,176] + frac[195,176]
   + frac[192,176] + frac[196,176] + frac[198,176] >= 1
 cov_557_3: frac[195,3] + frac[194,3] + frac[197,3] + frac[141,3]
   + frac[146,3] >= 0.8
 cov_558_159: frac[195,159] + frac[197,159] + frac[196,159] + frac[193,159]
   + frac[72,159] + frac[198,159] + frac[153,159] >= 0.7
 cov_559_113: frac[42,113] + frac[194,113] + frac[192,113] + frac[197,113]
   + frac[103,113] + frac[198,113] + frac[196,113] >= 0.9
 cov_560_69: frac[193,69] + frac[196,69] + frac[194,69] + frac[198,69]
   + frac[82,69] + frac[107,69] + frac[197,69] >= 0.9
 cov_561_75: frac[193,75] + frac[196,75] + frac[198,75] + frac[197,75]
   + frac[154,75] + frac[87,75] + frac[195,75] >= 0.9
 cov_562_55: frac[147,55] + frac[196,55] + frac[195,55] + frac[90,55]
   + frac[193,55] + frac[198,55] + frac[197,55] >= 0.7
 cov_563_133: frac[196,133] + frac[3,133] + frac[192,133] + frac[53,133]
   + frac[193,133] >= 0.8
 cov_564_100: frac[196,100] + frac[50,100] + frac[192,100] + frac[46,100]
   + frac[193,100] >= 0.7
 cov_565_97: frac[34,97] + frac[192,97] + frac[30,97] >= 0.9
 cov_566_7: frac[196,7] + frac[132,7] + frac[192,7] + frac[8,7]
   + frac[194,7] + frac[198,7] + frac[197,7] >= 0.7
 cov_567_79: frac[51,79] + frac[196,79] + frac[193,79] + frac[97,79]
   + frac[197,79] + frac[194,79] + frac[198,79] >= 0.8
 cov_568_7: frac[159,7] + frac[196,7] + frac[192,7] + frac[43,7]
   + frac[195,7] + frac[198,7] + frac[197,7] >= 0.7
 cov_569_45: frac[198,45] + frac[196,45] + frac[193,45] + frac[197,45]
   + frac[164,45] + frac[72,45] + frac[195,45] >= 0.7
 cov_570_96: frac[192,96] + frac[17,96] + frac[25,96] >= 0.8
 cov_571_4: frac[197,4] + frac[195,4] + frac[193,4] + frac[176,4]
   + frac[196,4] + frac[59,4] + frac[198,4] >= 0.9
 cov_572_58: frac[156,58] + frac[140,58] + frac[195,58] + frac[194,58]
   + frac[197,58] >= 0.9
 cov_573_254: frac[133,254] + frac[143,254] + frac[194,254] >= 0.9
 cov_574_35: frac[165,35] + frac[195,35] + frac[194,35] + frac[197,35]
   + frac[123,35] >= 1
 cov_575_90: frac[197,90] + frac[193,90] + frac[196,90] + frac[198,90]
   + frac[195,90] + frac[191,90] + frac[60,90] >= 0.8
 cov_576_63: frac[189,63] + frac[154,63] + frac[195,63] >= 0.8
 cov_577_253: frac[193,253] + frac[69,253] + frac[192,253] + frac[2,253]
   + frac[196,253] >= 1
 cov_578_199: frac[147,199] + frac[197,199] + frac[106,199] + frac[194,199]
   + frac[195,199] >= 1
 cov_579_102: frac[193,102] + frac[52,102] + frac[12,102] + frac[196,102]
   + frac[192,102] >= 0.8
 cov_580_120: frac[99,120] + frac[194,120] + frac[103,120] >= 1
 cov_581_215: frac[113,215] + frac[109,215] + frac[194,215] >= 0.9
 cov_582_138: frac[51,138] + frac[72,138] + frac[193,138] >= 1
 cov_583_81: frac[194,81] + frac[197,81] + frac[195,81] + frac[143,81]
   + frac[181,81] >= 1
 cov_584_223: frac[198,223] + frac[61,223] + frac[193,223] + frac[194,223]
   + frac[196,223] + frac[119,223] + frac[197,223] >= 0.8
 cov_585_225: frac[105,225] + frac[118,225] + frac[194,225] >= 0.9
 cov_586_77: frac[32,77] + frac[197,77] + frac[194,77] + frac[198,77]
   + frac[192,77] + frac[196,77] + frac[125,77] >= 0.7
 cov_587_249: frac[197,249] + frac[198,249] + frac[195,249] + frac[65,249]
   + frac[196,249] + frac[193,249] + frac[164,249] >= 1
 cov_588_117: frac[175,117] + frac[195,117] + frac[196,117] + frac[198,117]
   + frac[193,117] + frac[85,117] + frac[197,117] >= 1
 cov_589_219: frac[133,219] + frac[194,219] + frac[131,219] >= 0.8
 cov_590_151: frac[113,151] + frac[196,151] + frac[194,151] + frac[86,151]
   + frac[193,151] + frac[197,151] + frac[198,151] >= 0.8
 cov_591_155: frac[195,155] + frac[168,155] + frac[160,155] >= 0.7
 cov_592_192: frac[196,192] + frac[116,192] + frac[197,192] + frac[198,192]
   + frac[60,192] + frac[193,192] + frac[194,192] >= 0.8
 cov_593_129: frac[192,129] + frac[193,129] + frac[66,129] + frac[37,129]
   + frac[196,129] >= 0.8
 cov_594_46: frac[35,46] + frac[192,46] + frac[34,46] >= 0.7
 cov_595_230: frac[196,230] + frac[116,230] + frac[30,230] + frac[192,230]
   + frac[197,230] + frac[194,230] + frac[198,230] >= 0.8
 cov_596_133: frac[7,133] + frac[33,133] + frac[192,133] >= 0.7
 cov_597_164: frac[197,164] + frac[160,164] + frac[195,164] + frac[196,164]
   + frac[198,164] + frac[7,164] + frac[192,164] >= 0.8
 cov_598_187: frac[58,187] + frac[193,187] + frac[48,187] >= 1
 cov_599_198: frac[195,198] + frac[197,198] + frac[198,198] + frac[193,198]
   + frac[172,198] + frac[196,198] + frac[60,198] >= 0.9
 cov_600_106: frac[126,106] + frac[79,106] + frac[196,106] + frac[198,106]
   + frac[194,106] + frac[193,106] + frac[197,106] >= 0.7
 cov_601_101: frac[78,101] + frac[193,101] + frac[196,101] + frac[15,101]
   + frac[192,101] >= 0.9
 cov_602_119: frac[185,119] + frac[198,119] + frac[195,119] + frac[196,119]
   + frac[193,119] + frac[197,119] + frac[70,119] >= 0.9
 cov_603_211: frac[193,211] + frac[196,211] + frac[197,211] + frac[194,211]
   + frac[52,211] + frac[198,211] + frac[108,211] >= 0.8
 cov_604_54: frac[177,54] + frac[148,54] + frac[195,54] >= 0.7
 cov_605_149: frac[196,149] + frac[197,149] + frac[166,149] + frac[198,149]
   + frac[29,149] + frac[195,149] + frac[192,149] >= 0.9
 cov_606_91: frac[197,91] + frac[45,91] + frac[192,91] + frac[195,91]
   + frac[177,91] + frac[196,91] + frac[198,91] >= 0.7
 cov_607_155: frac[154,155] + frac[195,155] + frac[197,155] + frac[136,155]
   + frac[194,155] >= 1
 cov_608_91: frac[123,91] + frac[194,91] + frac[98,91] >= 0.9
 cov_609_202: frac[52,202] + frac[84,202] + frac[193,202] >= 1
 cov_610_214: frac[65,214] + frac[75,214] + frac[193,214] >= 0.7
 cov_611_75: frac[193,75] + frac[196,75] + frac[198,75] + frac[197,75]
   + frac[167,75] + frac[79,75] + frac[195,75] >= 0.9
 cov_612_70: frac[196,70] + frac[111,70] + frac[9,70] + frac[192,70]
   + frac[197,70] + frac[194,70] + frac[198,70] >= 0.7
 cov_613_55: frac[114,55] + frac[192,55] + frac[196,55] + frac[194,55]
   + frac[19,55] + frac[198,55] + frac[197,55] >= 0.9
 cov_614_143: frac[194,143] + frac[195,143] + frac[197,143] + frac[133,143]
   + frac[170,143] >= 0.7
 cov_615_74: frac[196,74] + frac[198,74] + frac[194,74] + frac[193,74]
   + frac[197,74] + frac[60,74] + frac[130,74] >= 0.9
 cov_616_100: frac[61,100] + frac[48,100] + frac[193,100] >= 0.9
 cov_617_230: frac[196,230] + frac[116,230] + frac[192,230] + frac[197,230]
   + frac[194,230] + frac[198,230] + frac[12,230] >= 0.9
 cov_618_154: frac[198,154] + frac[192,154] + frac[13,154] + frac[194,154]
   + frac[197,154] + frac[120,154] + frac[196,154] >= 0.8
 cov_619_52: frac[185,52] + frac[197,52] + frac[2,52] + frac[195,52]
   + frac[192,52] + frac[196,52] + frac[198,52] >= 0.8
 cov_620_97: frac[182,97] + frac[194,97] + frac[132,97] + frac[197,97]
   + frac[195,97] >= 0.9
 cov_621_235: frac[196,235] + frac[197,235] + frac[198,235] + frac[56,235]
   + frac[159,235] + frac[195,235] + frac[193,235] >= 0.7
 cov_622_182: frac[198,182] + frac[43,182] + frac[195,182] + frac[152,182]
   + frac[192,182] + frac[197,182] + frac[196,182] >= 0.9
 cov_623_89: frac[192,89] + frac[147,89] + frac[46,89] + frac[198,89]
   + frac[197,89] + frac[195,89] + frac[196,89] >= 0.8
 cov_624_43: frac[53,43] + frac[196,43] + frac[198,43] + frac[195,43]
   + frac[193,43] + frac[156,43] + frac[197,43] >= 1
 cov_625_51: frac[195,51] + frac[168,51] + frac[166,51] >= 0.7
 cov_626_132: frac[194,132] + frac[132,132] + frac[135,132] >= 0.8
 cov_627_243: frac[192,243] + frac[193,243] + frac[8,243] + frac[196,243]
   + frac[84,243] >= 1
 cov_628_148: frac[60,148] + frac[198,148] + frac[193,148] + frac[197,148]
   + frac[195,148] + frac[189,148] + frac[196,148] >= 0.9
 cov_629_213: frac[197,213] + frac[192,213] + frac[196,213] + frac[195,213]
   + frac[198,213] + frac[0,213] + frac[157,213] >= 0.9
 cov_630_57: frac[194,57] + frac[97,57] + frac[192,57] + frac[1,57]
   + frac[198,57] + frac[196,57] + frac[197,57] >= 0.7
 cov_631_18: frac[109,18] + frac[192,18] + frac[197,18] + frac[196,18]
   + frac[194,18] + frac[24,18] + frac[198,18] >= 0.7
 cov_632_72: frac[198,72] + frac[192,72] + frac[189,72] + frac[196,72]
   + frac[197,72] + frac[195,72] + frac[0,72] >= 0.9
 cov_633_171: frac[194,171] + frac[193,171] + frac[197,171] + frac[68,171]
   + frac[126,171] + frac[196,171] + frac[198,171] >= 0.9
 cov_634_94: frac[154,94] + frac[195,94] + frac[135,94] + frac[197,94]
   + frac[194,94] >= 1
 cov_635_51: frac[192,51] + frac[28,51] + frac[25,51] >= 1
 cov_636_59: frac[196,59] + frac[193,59] + frac[198,59] + frac[195,59]
   + frac[53,59] + frac[197,59] + frac[147,59] >= 0.8
 cov_637_168: frac[192,168] + frac[33,168] + frac[10,168] >= 0.8
 cov_638_190: frac[97,190] + frac[151,190] + frac[195,190] + frac[194,190]
   + frac[197,190] >= 0.8
 cov_639_204: frac[173,204] + frac[194,204] + frac[197,204] + frac[136,204]
   + frac[195,204] >= 1
 cov_640_229: frac[196,229] + frac[195,229] + frac[198,229] + frac[192,229]
   + frac[148,229] + frac[23,229] + frac[197,229] >= 0.8
 cov_641_206: frac[193,206] + frac[73,206] + frac[52,206] >= 0.9
 cov_642_89: frac[73,89] + frac[193,89] + frac[77,89] >= 1
 cov_643_54: frac[192,54] + frac[196,54] + frac[21,54] + frac[82,54]
   + frac[193,54] >= 0.7
 cov_644_153: frac[194,153] + frac[130,153] + frac[115,153] >= 0.9
 cov_645_17: frac[140,17] + frac[102,17] + frac[194,17] >= 0.9
 cov_646_66: frac[197,66] + frac[117,66] + frac[194,66] + frac[165,66]
   + frac[195,66] >= 0.9
 cov_647_62: frac[100,62] + frac[194,62] + frac[179,62] + frac[197,62]
   + frac[195,62] >= 0.9
 cov_648_70: frac[193,70] + frac[196,70] + frac[100,70] + frac[78,70]
   + frac[197,70] + frac[194,70] + frac[198,70] >= 0.9
 cov_649_225: frac[96,225] + frac[197,225] + frac[33,225] + frac[196,225]
   + frac[198,225] + frac[192,225] + frac[194,225] >= 0.9
 cov_650_13: frac[113,13] + frac[197,13] + frac[195,13] + frac[194,13]
   + frac[145,13] >= 1
 cov_651_42: frac[197,42] + frac[196,42] + frac[198,42] + frac[49,42]
   + frac[144,42] + frac[195,42] + frac[193,42] >= 0.8
 cov_652_244: frac[82,244] + frac[192,244] + frac[193,244] + frac[36,244]
   + frac[196,244] >= 1
 cov_653_170: frac[20,170] + frac[193,170] + frac[192,170] + frac[196,170]
   + frac[52,170] >= 0.9
 cov_654_30: frac[143,30] + frac[168,30] + frac[195,30] + frac[194,30]
   + frac[197,30] >= 1
 cov_655_167: frac[196,167] + frac[129,167] + frac[32,167] + frac[192,167]
   + frac[194,167] + frac[197,167] + frac[198,167] >= 0.8
 cov_656_225: frac[192,225] + frac[29,225] + frac[4,225] >= 0.9
 cov_657_217: frac[196,217] + frac[193,217] + frac[192,217] + frac[27,217]
   + frac[61,217] >= 0.7
 cov_658_203: frac[103,203] + frac[113,203] + frac[194,203] >= 0.7
 cov_659_135: frac[2,135] + frac[192,135] + frac[20,135] >= 0.9
 cov_660_84: frac[42,84] + frac[189,84] + frac[192,84] + frac[196,84]
   + frac[197,84] + frac[198,84] + frac[195,84] >= 1
 cov_661_72: frac[129,72] + frac[194,72] + frac[124,72] >= 0.7
 cov_662_233: frac[95,233] + frac[193,233] + frac[21,233] + frac[192,233]
   + frac[196,233] >= 0.8
 cov_663_93: frac[196,93] + frac[153,93] + frac[192,93] + frac[197,93]
   + frac[198,93] + frac[13,93] + frac[195,93] >= 1
 cov_664_42: frac[197,42] + frac[196,42] + frac[198,42] + frac[186,42]
   + frac[195,42] + frac[50,42] + frac[193,42] >= 0.9
 cov_665_87: frac[173,87] + frac[195,87] + frac[196,87] + frac[193,87]
   + frac[197,87] + frac[198,87] + frac[84,87] >= 0.9
 cov_666_88: frac[198,88] + frac[196,88] + frac[197,88] + frac[195,88]
   + frac[71,88] + frac[151,88] + frac[193,88] >= 0.7
 cov_667_135: frac[195,135] + frac[168,135] + frac[197,135] + frac[198,135]
   + frac[193,135] + frac[62,135] + frac[196,135] >= 0.8
 cov_668_61: frac[196,61] + frac[197,61] + frac[120,61] + frac[194,61]
   + frac[41,61] + frac[192,61] + frac[198,61] >= 0.9
 cov_669_52: frac[193,52] + frac[72,52] + frac[95,52] >= 0.8
 cov_670_209: frac[193,209] + frac[70,209] + frac[197,209] + frac[150,209]
   + frac[195,209] + frac[196,209] + frac[198,209] >= 1
 cov_671_142: frac[195,142] + frac[193,142] + frac[197,142] + frac[196,142]
   + frac[191,142] + frac[79,142] + frac[198,142] >= 0.7
 cov_672_67: frac[157,67] + frac[195,67] + frac[163,67] >= 0.8
 cov_673_123: frac[195,123] + frac[196,123] + frac[198,123] + frac[197,123]
   + frac[2,123] + frac[171,123] + frac[192,123] >= 1
 cov_674_198: frac[197,198] + frac[194,198] + frac[198,198] + frac[193,198]
   + frac[196,198] + frac[111,198] + frac[50,198] >= 0.9
 cov_675_120: frac[194,120] + frac[116,120] + frac[117,120] >= 0.8
 cov_676_25: frac[25,25] + frac[192,25] + frac[31,25] >= 0.9
 cov_677_48: frac[192,48] + frac[198,48] + frac[196,48] + frac[168,48]
   + frac[197,48] + frac[2,48] + frac[195,48] >= 1
 cov_678_169: frac[27,169] + frac[198,169] + frac[192,169] + frac[196,169]
   + frac[197,169] + frac[195,169] + frac[164,169] >= 0.7
 cov_679_235: frac[177,235] + frac[173,235] + frac[195,235] >= 0.8
 cov_680_92: frac[195,92] + frac[126,92] + frac[194,92] + frac[148,92]
   + frac[197,92] >= 1
 cov_681_151: frac[26,151] + frac[192,151] + frac[196,151] + frac[67,151]
   + frac[193,151] >= 0.7
 cov_682_18: frac[77,18] + frac[197,18] + frac[120,18] + frac[196,18]
   + frac[194,18] + frac[198,18] + frac[193,18] >= 0.8
 cov_683_20: frac[32,20] + frac[192,20] + frac[196,20] + frac[55,20]
   + frac[193,20] >= 0.7
 cov_684_122: frac[196,122] + frac[83,122] + frac[198,122] + frac[195,122]
   + frac[197,122] + frac[157,122] + frac[193,122] >= 1
 cov_685_24: frac[38,24] + frac[192,24] + frac[44,24] >= 0.9
 cov_686_58: frac[120,58] + frac[194,58] + frac[105,58] >= 0.7
 cov_687_69: frac[195,69] + frac[196,69] + frac[198,69] + frac[192,69]
   + frac[176,69] + frac[16,69] + frac[197,69] >= 0.7
 cov_688_28: frac[192,28] + frac[82,28] + frac[196,28] + frac[193,28]
   + frac[40,28] >= 0.7
 cov_689_158: frac[171,158] + frac[195,158] + frac[162,158] >= 0.9
 cov_690_229: frac[192,229] + frac[2,229] + frac[7,229] >= 1
 cov_691_123: frac[16,123] + frac[24,123] + frac[192,123] >= 0.7
 cov_692_11: frac[198,11] + frac[194,11] + frac[193,11] + frac[95,11]
   + frac[197,11] + frac[142,11] + frac[196,11] >= 0.7
 cov_693_62: frac[160,62] + frac[197,62] + frac[196,62] + frac[198,62]
   + frac[3,62] + frac[192,62] + frac[195,62] >= 1
 cov_694_117: frac[93,117] + frac[66,117] + frac[193,117] >= 0.8
 cov_695_242: frac[192,242] + frac[45,242] + frac[29,242] >= 0.8
 cov_696_76: frac[192,76] + frac[196,76] + frac[181,76] + frac[197,76]
   + frac[2,76] + frac[195,76] + frac[198,76] >= 0.7
 cov_697_91: frac[124,91] + frac[194,91] + frac[113,91] >= 0.8
 cov_698_93: frac[145,93] + frac[182,93] + frac[195,93] >= 0.7
 cov_699_46: frac[157,46] + frac[14,46] + frac[195,46] + frac[197,46]
   + frac[192,46] + frac[196,46] + frac[198,46] >= 0.9
 cov_700_136: frac[193,136] + frac[64,136] + frac[71,136] >= 1
 cov_701_246: frac[195,246] + frac[197,246] + frac[31,246] + frac[198,246]
   + frac[192,246] + frac[171,246] + frac[196,246] >= 0.7
 cov_702_128: frac[88,128] + frac[197,128] + frac[193,128] + frac[196,128]
   + frac[198,128] + frac[99,128] + frac[194,128] >= 0.9
 cov_703_209: frac[89,209] + frac[193,209] + frac[88,209] >= 1
 cov_704_123: frac[194,123] + frac[196,123] + frac[198,123] + frac[197,123]
   + frac[192,123] + frac[139,123] + frac[7,123] >= 0.9
 cov_705_190: frac[72,190] + frac[193,190] + frac[80,190] >= 1
 cov_706_0: frac[9,0] + frac[63,0] + frac[196,0] + frac[192,0]
   + frac[193,0] >= 0.9
 cov_707_104: frac[21,104] + frac[196,104] + frac[69,104] + frac[192,104]
   + frac[193,104] >= 0.9
 cov_708_255: frac[196,255] + frac[195,255] + frac[197,255] + frac[9,255]
   + frac[198,255] + frac[179,255] + frac[192,255] >= 0.7
 cov_709_78: frac[48,78] + frac[196,78] + frac[133,78] + frac[194,78]
   + frac[198,78] + frac[193,78] + frac[197,78] >= 1
 cov_710_153: frac[194,153] + frac[124,153] + frac[193,153] + frac[92,153]
   + frac[198,153] + frac[196,153] + frac[197,153] >= 0.9
 cov_711_182: frac[198,182] + frac[95,182] + frac[193,182] + frac[195,182]
   + frac[197,182] + frac[196,182] + frac[155,182] >= 0.9
 cov_712_173: frac[197,173] + frac[182,173] + frac[108,173] + frac[195,173]
   + frac[194,173] >= 0.7
 cov_713_170: frac[195,170] + frac[192,170] + frac[197,170] + frac[13,170]
   + frac[196,170] + frac[161,170] + frac[198,170] >= 1
 cov_714_143: frac[194,143] + frac[198,143] + frac[125,143] + frac[18,143]
   + frac[197,143] + frac[192,143] + frac[196,143] >= 1
 cov_715_255: frac[196,255] + frac[93,255] + frac[192,255] + frac[193,255]
   + frac[14,255] >= 1
 cov_716_12: frac[193,12] + frac[76,12] + frac[51,12] >= 0.7
 cov_717_152: frac[192,152] + frac[194,152] + frac[197,152] + frac[46,152]
   + frac[198,152] + frac[196,152] + frac[113,152] >= 0.7
 cov_718_81: frac[192,81] + frac[197,81] + frac[195,81] + frac[198,81]
   + frac[196,81] + frac[17,81] + frac[161,81] >= 0.7
 cov_719_151: frac[192,151] + frac[196,151] + frac[173,151] + frac[38,151]
   + frac[195,151] + frac[197,151] + frac[198,151] >= 0.9
 cov_720_82: frac[197,82] + frac[192,82] + frac[103,82] + frac[196,82]
   + frac[194,82] + frac[27,82] + frac[198,82] >= 0.8
 cov_721_217: frac[192,217] + frac[23,217] + frac[40,217] >= 0.9
 cov_722_100: frac[173,100] + frac[90,100] + frac[196,100] + frac[197,100]
   + frac[198,100] + frac[195,100] + frac[193,100] >= 0.8
 cov_723_11: frac[198,11] + frac[195,11] + frac[197,11] + frac[165,11]
   + frac[38,11] + frac[192,11] + frac[196,11] >= 0.9
 cov_724_39: frac[194,39] + frac[198,39] + frac[197,39] + frac[133,39]
   + frac[193,39] + frac[196,39] + frac[89,39] >= 0.9
 cov_725_184: frac[196,184] + frac[91,184] + frac[194,184] + frac[197,184]
   + frac[143,184] + frac[198,184] + frac[193,184] >= 0.8
 cov_726_119: frac[198,119] + frac[196,119] + frac[193,119] + frac[75,119]
   + frac[194,119] + frac[143,119] + frac[197,119] >= 0.8
 cov_727_140: frac[32,140] + frac[195,140] + frac[198,140] + frac[192,140]
   + frac[196,140] + frac[155,140] + frac[197,140] >= 1
 cov_728_6: frac[194,6] + frac[167,6] + frac[135,6] + frac[195,6]
   + frac[197,6] >= 0.9
 cov_729_194: frac[178,194] + frac[68,194] + frac[196,194] + frac[195,194]
   + frac[197,194] + frac[198,194] + frac[193,194] >= 0.9
 cov_730_88: frac[198,88] + frac[196,88] + frac[197,88] + frac[17,88]
   + frac[128,88] + frac[194,88] + frac[192,88] >= 0.8
 cov_731_34: frac[44,34] + frac[196,34] + frac[192,34] + frac[198,34]
   + frac[195,34] + frac[149,34] + frac[197,34] >= 0.8
 cov_732_15: frac[196,15] + frac[198,15] + frac[154,15] + frac[195,15]
   + frac[91,15] + frac[193,15] + frac[197,15] >= 0.9
 cov_733_91: frac[141,91] + frac[194,91] + frac[113,91] >= 0.9
 cov_734_146: frac[141,146] + frac[23,146] + frac[198,146] + frac[192,146]
   + frac[197,146] + frac[196,146] + frac[194,146] >= 0.8
 cov_735_208: frac[92,208] + frac[93,208] + frac[193,208] >= 1
 cov_736_240: frac[106,240] + frac[196,240] + frac[197,240] + frac[198,240]
   + frac[41,240] + frac[192,240] + frac[194,240] >= 0.9
 cov_737_47: frac[35,47] + frac[197,47] + frac[125,47] + frac[192,47]
   + frac[196,47] + frac[194,47] + frac[198,47] >= 0.8
 cov_738_192: frac[196,192] + frac[192,192] + frac[193,192] + frac[17,192]
   + frac[53,192] >= 0.9
 cov_739_244: frac[99,244] + frac[194,244] + frac[197,244] + frac[195,244]
   + frac[191,244] >= 1
 cov_740_249: frac[169,249] + frac[197,249] + frac[195,249] + frac[194,249]
   + frac[115,249] >= 1
 cov_741_190: frac[107,190] + frac[198,190] + frac[192,190] + frac[28,190]
   + frac[194,190] + frac[197,190] + frac[196,190] >= 0.7
 cov_742_144: frac[195,144] + frac[181,144] + frac[194,144] + frac[128,144]
   + frac[197,144] >= 0.7
 cov_743_233: frac[56,233] + frac[193,233] + frac[103,233] + frac[197,233]
   + frac[198,233] + frac[194,233] + frac[196,233] >= 0.8
 cov_744_161: frac[60,161] + frac[196,161] + frac[192,161] + frac[43,161]
   + frac[193,161] >= 0.9
 cov_745_137: frac[193,137] + frac[79,137] + frac[196,137] + frac[197,137]
   + frac[117,137] + frac[198,137] + frac[194,137] >= 0.9
 cov_746_3: frac[124,3] + frac[195,3] + frac[194,3] + frac[197,3]
   + frac[159,3] >= 1
 cov_747_4: frac[192,4] + frac[54,4] + frac[193,4] + frac[196,4]
   + frac[17,4] >= 1
 cov_748_86: frac[196,86] + frac[195,86] + frac[197,86] + frac[198,86]
   + frac[39,86] + frac[177,86] + frac[192,86] >= 0.9
 cov_749_50: frac[198,50] + frac[22,50] + frac[189,50] + frac[195,50]
   + frac[192,50] + frac[197,50] + frac[196,50] >= 0.9
 cov_750_85: frac[167,85] + frac[195,85] + frac[186,85] >= 0.7
 cov_751_43: frac[192,43] + frac[196,43] + frac[45,43] + frac[193,43]
   + frac[50,43] >= 0.8
 cov_752_208: frac[12,208] + frac[134,208] + frac[192,208] + frac[194,208]
   + frac[196,208] + frac[197,208] + frac[198,208] >= 1
 cov_753_230: frac[54,230] + frac[77,230] + frac[193,230] >= 0.8
 cov_754_175: frac[57,175] + frac[193,175] + frac[64,175] >= 1
 cov_755_34: frac[146,34] + frac[194,34] + frac[195,34] + frac[104,34]
   + frac[197,34] >= 0.9
 cov_756_165: frac[193,165] + frac[81,165] + frac[60,165] >= 0.8
 cov_757_184: frac[195,184] + frac[181,184] + frac[160,184] >= 0.8
 cov_758_116: frac[171,116] + frac[196,116] + frac[197,116] + frac[198,116]
   + frac[195,116] + frac[33,116] + frac[192,116] >= 1
 cov_759_53: frac[196,53] + frac[193,53] + frac[82,53] + frac[23,53]
   + frac[192,53] >= 0.9
 cov_760_79: frac[155,79] + frac[195,79] + frac[192,79] + frac[196,79]
   + frac[10,79] + frac[197,79] + frac[198,79] >= 0.7
 cov_761_255: frac[194,255] + frac[195,255] + frac[126,255] + frac[197,255]
   + frac[188,255] >= 0.7
 cov_762_183: frac[97,183] + frac[116,183] + frac[194,183] >= 0.7
 cov_763_207: frac[195,207] + frac[155,207] + frac[157,207] >= 0.9
 cov_764_84: frac[37,84] + frac[40,84] + frac[192,84] >= 1
 cov_765_117: frac[192,117] + frac[196,117] + frac[198,117] + frac[39,117]
   + frac[194,117] + frac[132,117] + frac[197,117] >= 1
 cov_766_169: frac[198,169] + frac[194,169] + frac[193,169] + frac[196,169]
   + frac[110,169] + frac[74,169] + frac[197,169] >= 0.7
 cov_767_230: frac[195,230] + frac[170,230] + frac[197,230] + frac[141,230]
   + frac[194,230] >= 1
 cov_768_110: frac[194,110] + frac[195,110] + frac[197,110] + frac[142,110]
   + frac[155,110] >= 0.7
 cov_769_96: frac[197,96] + frac[194,96] + frac[193,96] + frac[133,96]
   + frac[92,96] + frac[198,96] + frac[196,96] >= 1
 cov_770_91: frac[73,91] + frac[67,91] + frac[193,91] >= 0.9
 cov_771_73: frac[197,73] + frac[194,73] + frac[195,73] + frac[147,73]
   + frac[104,73] >= 1
 cov_772_78: frac[194,78] + frac[195,78] + frac[197,78] + frac[161,78]
   + frac[120,78] >= 0.7
 cov_773_241: frac[197,241] + frac[195,241] + frac[196,241] + frac[198,241]
   + frac[67,241] + frac[193,241] + frac[190,241] >= 0.9
 cov_774_27: frac[192,27] + frac[108,27] + frac[197,27] + frac[2,27]
   + frac[194,27] + frac[196,27] + frac[198,27] >= 0.8
 cov_775_82: frac[154,82] + frac[195,82] + frac[146,82] >= 0.7
 cov_776_12: frac[37,12] + frac[197,12] + frac[127,12] + frac[194,12]
   + frac[198,12] + frac[192,12] + frac[196,12] >= 0.7
 cov_777_149: frac[196,149] + frac[197,149] + frac[198,149] + frac[161,149]
   + frac[195,149] + frac[33,149] + frac[192,149] >= 0.8
 cov_778_6: frac[196,6] + frac[53,6] + frac[193,6] + frac[27,6]
   + frac[192,6] >= 1
 cov_779_114: frac[195,114] + frac[193,114] + frac[198,114] + frac[197,114]
   + frac[183,114] + frac[95,114] + frac[196,114] >= 0.9
 cov_780_172: frac[197,172] + frac[34,172] + frac[194,172] + frac[198,172]
   + frac[131,172] + frac[192,172] + frac[196,172] >= 0.8
 cov_781_184: frac[64,184] + frac[58,184] + frac[193,184] >= 0.7
 cov_782_58: frac[193,58] + frac[136,58] + frac[194,58] + frac[197,58]
   + frac[65,58] + frac[196,58] + frac[198,58] >= 0.9
 cov_783_161: frac[141,161] + frac[27,161] + frac[196,161] + frac[198,161]
   + frac[192,161] + frac[194,161] + frac[197,161] >= 0.7
 cov_784_129: frac[198,129] + frac[192,129] + frac[194,129] + frac[197,129]
   + frac[96,129] + frac[27,129] + frac[196,129] >= 0.9
 cov_785_18: frac[197,18] + frac[196,18] + frac[194,18] + frac[56,18]
   + frac[198,18] + frac[101,18] + frac[193,18] >= 0.7
 cov_786_210: frac[195,210] + frac[164,210] + frac[194,210] + frac[122,210]
   + frac[197,210] >= 0.9
 cov_787_64: frac[93,64] + frac[198,64] + frac[196,64] + frac[197,64]
   + frac[195,64] + frac[193,64] + frac[174,64] >= 0.9
 cov_788_148: frac[50,148] + frac[193,148] + frac[49,148] >= 0.7
 cov_789_201: frac[197,201] + frac[198,201] + frac[194,201] + frac[96,201]
   + frac[196,201] + frac[72,201] + frac[193,201] >= 0.8
 cov_790_89: frac[192,89] + frac[198,89] + frac[197,89] + frac[99,89]
   + frac[33,89] + frac[196,89] + frac[194,89] >= 0.9
 cov_791_144: frac[192,144] + frac[65,144] + frac[196,144] + frac[193,144]
   + frac[10,144] >= 1
 cov_792_100: frac[78,100] + frac[196,100] + frac[181,100] + frac[197,100]
   + frac[198,100] + frac[195,100] + frac[193,100] >= 0.8
 cov_793_133: frac[102,133] + frac[194,133] + frac[197,133] + frac[163,133]
   + frac[195,133] >= 1
 cov_794_243: frac[133,243] + frac[194,243] + frac[125,243] >= 0.7
 cov_795_169: frac[51,169] + frac[198,169] + frac[158,169] + frac[193,169]
   + frac[196,169] + frac[197,169] + frac[195,169] >= 1
 cov_796_69: frac[195,69] + frac[193,69] + frac[196,69] + frac[48,69]
   + frac[198,69] + frac[171,69] + frac[197,69] >= 0.9
 cov_797_98: frac[165,98] + frac[195,98] + frac[106,98] + frac[197,98]
   + frac[194,98] >= 0.7
 cov_798_96: frac[197,96] + frac[195,96] + frac[193,96] + frac[189,96]
   + frac[80,96] + frac[198,96] + frac[196,96] >= 0.8
 cov_799_204: frac[33,204] + frac[192,204] + frac[193,204] + frac[196,204]
   + frac[89,204] >= 0.9
 cov_800_121: frac[194,121] + frac[125,121] + frac[104,121] >= 0.8
 cov_801_161: frac[36,161] + frac[196,161] + frac[198,161] + frac[192,161]
   + frac[197,161] + frac[195,161] + frac[171,161] >= 0.8
 cov_802_216: frac[44,216] + frac[192,216] + frac[36,216] >= 0.7
 cov_803_4: frac[112,4] + frac[115,4] + frac[194,4] >= 0.7
 cov_804_66: frac[90,66] + frac[198,66] + frac[197,66] + frac[193,66]
   + frac[155,66] + frac[196,66] + frac[195,66] >= 0.9
 cov_805_170: frac[195,170] + frac[103,170] + frac[197,170] + frac[179,170]
   + frac[194,170] >= 1
 cov_806_20: frac[25,20] + frac[192,20] + frac[196,20] + frac[56,20]
   + frac[193,20] >= 1
 cov_807_217: frac[103,217] + frac[194,217] + frac[140,217] >= 0.7
 cov_808_48: frac[192,48] + frac[31,48] + frac[198,48] + frac[196,48]
   + frac[172,48] + frac[197,48] + frac[195,48] >= 0.7
 cov_809_26: frac[195,26] + frac[171,26] + frac[197,26] + frac[196,26]
   + frac[198,26] + frac[193,26] + frac[77,26] >= 1
 cov_810_208: frac[13,208] + frac[153,208] + frac[192,208] + frac[196,208]
   + frac[197,208] + frac[198,208] + frac[195,208] >= 1
 cov_811_226: frac[65,226] + frac[193,226] + frac[78,226] >= 0.7
 cov_812_30: frac[182,30] + frac[161,30] + frac[195,30] >= 0.8
 cov_813_145: frac[62,145] + frac[97,145] + frac[194,145] + frac[193,145]
   + frac[196,145] + frac[198,145] + frac[197,145] >= 1
 cov_814_152: frac[192,152] + frac[194,152] + frac[197,152] + frac[22,152]
   + frac[198,152] + frac[119,152] + frac[196,152] >= 0.9
 cov_815_204: frac[21,204] + frac[192,204] + frac[16,204] >= 0.9
 cov_816_163: frac[163,163] + frac[194,163] + frac[197,163] + frac[143,163]
   + frac[195,163] >= 0.7
 cov_817_232: frac[191,232] + frac[198,232] + frac[192,232] + frac[196,232]
   + frac[197,232] + frac[18,232] + frac[195,232] >= 0.7
 cov_818_143: frac[65,143] + frac[193,143] + frac[39,143] + frac[192,143]
   + frac[196,143] >= 0.9
 cov_819_83: frac[198,83] + frac[192,83] + frac[30,83] + frac[140,83]
   + frac[196,83] + frac[194,83] + frac[197,83] >= 0.7
 cov_820_145: frac[192,145] + frac[193,145] + frac[196,145] + frac[10,145]
   + frac[61,145] >= 0.8
 cov_821_69: frac[195,69] + frac[175,69] + frac[193,69] + frac[196,69]
   + frac[198,69] + frac[79,69] + frac[197,69] >= 0.9
 cov_822_17: frac[198,17] + frac[196,17] + frac[42,17] + frac[197,17]
   + frac[111,17] + frac[194,17] + frac[192,17] >= 0.9
 cov_823_65: frac[105,65] + frac[38,65] + frac[197,65] + frac[198,65]
   + frac[196,65] + frac[194,65] + frac[192,65] >= 0.8
 cov_824_210: frac[196,210] + frac[195,210] + frac[174,210] + frac[197,210]
   + frac[198,210] + frac[192,210] + frac[35,210] >= 1
 cov_825_15: frac[153,15] + frac[34,15] + frac[192,15] + frac[196,15]
   + frac[198,15] + frac[195,15] + frac[197,15] >= 0.9
 cov_826_126: frac[124,126] + frac[194,126] + frac[103,126] >= 0.8
 cov_827_46: frac[197,46] + frac[7,46] + frac[192,46] + frac[196,46]
   + frac[104,46] + frac[194,46] + frac[198,46] >= 0.9
 cov_828_37: frac[57,37] + frac[55,37] + frac[193,37] >= 0.7
 cov_829_72: frac[192,72] + frac[20,72] + frac[193,72] + frac[90,72]
   + frac[196,72] >= 0.7
 cov_830_93: frac[196,93] + frac[193,93] + frac[32,93] + frac[192,93]
   + frac[56,93] >= 0.9
 cov_831_102: frac[5,102] + frac[19,102] + frac[192,102] >= 0.9
 cov_832_2: frac[198,2] + frac[195,2] + frac[65,2] + frac[156,2]
   + frac[197,2] + frac[196,2] + frac[193,2] >= 0.7
 cov_833_59: frac[195,59] + frac[119,59] + frac[150,59] + frac[197,59]
   + frac[194,59] >= 0.7
 cov_834_241: frac[192,241] + frac[197,241] + frac[12,241] + frac[195,241]
   + frac[196,241] + frac[198,241] + frac[146,241] >= 0.8
 cov_835_225: frac[0,225] + frac[34,225] + frac[192,225] >= 1
 cov_836_101: frac[160,101] + frac[197,101] + frac[195,101] + frac[194,101]
   + frac[100,101] >= 0.8
 cov_837_220: frac[198,220] + frac[196,220] + frac[197,220] + frac[183,220]
   + frac[195,220] + frac[60,220] + frac[193,220] >= 0.8
 cov_838_158: frac[194,158] + frac[120,158] + frac[123,158] >= 0.9
 cov_839_75: frac[196,75] + frac[198,75] + frac[197,75] + frac[194,75]
   + frac[18,75] + frac[143,75] + frac[192,75] >= 0.7
 cov_840_44: frac[138,44] + frac[194,44] + frac[198,44] + frac[31,44]
   + frac[192,44] + frac[196,44] + frac[197,44] >= 0.8
 cov_841_250: frac[130,250] + frac[143,250] + frac[194,250] >= 0.7
 cov_842_56: frac[192,56] + frac[197,56] + frac[195,56] + frac[198,56]
   + frac[196,56] + frac[174,56] + frac[3,56] >= 0.7
 cov_843_17: frac[132,17] + frac[97,17] + frac[194,17] >= 0.9
 cov_844_152: frac[192,152] + frac[194,152] + frac[197,152] + frac[103,152]
   + frac[198,152] + frac[196,152] + frac[47,152] >= 0.8
 cov_845_240: frac[139,240] + frac[102,240] + frac[194,240] >= 0.9
 cov_846_205: frac[130,205] + frac[128,205] + frac[194,205] >= 0.7
 cov_847_243: frac[198,243] + frac[192,243] + frac[132,243] + frac[196,243]
   + frac[197,243] + frac[194,243] + frac[31,243] >= 0.8
 cov_848_39: frac[194,39] + frac[198,39] + frac[197,39] + frac[196,39]
   + frac[0,39] + frac[192,39] + frac[101,39] >= 0.8
 cov_849_79: frac[133,79] + frac[127,79] + frac[194,79] >= 1
 cov_850_5: frac[192,5] + frac[16,5] + frac[2,5] >= 1
 cov_851_201: frac[8,201] + frac[197,201] + frac[198,201] + frac[192,201]
   + frac[194,201] + frac[96,201] + frac[196,201] >= 0.8
 cov_852_32: frac[15,32] + frac[32,32] + frac[192,32] >= 1
 cov_853_145: frac[194,145] + frac[193,145] + frac[196,145] + frac[198,145]
   + frac[134,145] + frac[197,145] + frac[57,145] >= 1
 cov_854_254: frac[198,254] + frac[195,254] + frac[197,254] + frac[192,254]
   + frac[39,254] + frac[188,254] + frac[196,254] >= 0.9
 cov_855_103: frac[192,103] + frac[10,103] + frac[193,103] + frac[69,103]
   + frac[196,103] >= 0.9
 cov_856_226: frac[197,226] + frac[198,226] + frac[62,226] + frac[193,226]
   + frac[194,226] + frac[196,226] + frac[109,226] >= 1
 cov_857_101: frac[197,101] + frac[130,101] + frac[193,101] + frac[196,101]
   + frac[50,101] + frac[194,101] + frac[198,101] >= 0.9
 cov_858_70: frac[193,70] + frac[29,70] + frac[196,70] + frac[56,70]
   + frac[192,70] >= 1
 cov_859_85: frac[197,85] + frac[192,85] + frac[195,85] + frac[177,85]
   + frac[196,85] + frac[27,85] + frac[198,85] >= 0.8
 cov_860_86: frac[32,86] + frac[43,86] + frac[192,86] >= 0.7
 cov_861_66: frac[198,66] + frac[197,66] + frac[28,66] + frac[183,66]
   + frac[196,66] + frac[195,66] + frac[192,66] >= 0.8
 cov_862_97: frac[69,97] + frac[70,97] + frac[193,97] >= 1
 cov_863_31: frac[196,31] + frac[169,31] + frac[198,31] + frac[195,31]
   + frac[192,31] + frac[197,31] + frac[33,31] >= 0.9
 cov_864_85: frac[30,85] + frac[15,85] + frac[192,85] >= 0.7
 cov_865_48: frac[192,48] + frac[49,48] + frac[196,48] + frac[193,48]
   + frac[8,48] >= 0.8
 cov_866_60: frac[163,60] + frac[196,60] + frac[18,60] + frac[197,60]
   + frac[192,60] + frac[195,60] + frac[198,60] >= 0.9
 cov_867_175: frac[70,175] + frac[196,175] + frac[116,175] + frac[194,175]
   + frac[198,175] + frac[193,175] + frac[197,175] >= 0.8
 cov_868_6: frac[71,6] + frac[193,6] + frac[50,6] >= 0.7
 cov_869_215: frac[21,215] + frac[198,215] + frac[117,215] + frac[192,215]
   + frac[196,215] + frac[197,215] + frac[194,215] >= 1
 cov_870_6: frac[196,6] + frac[193,6] + frac[192,6] + frac[20,6]
   + frac[79,6] >= 1
 cov_871_114: frac[194,114] + frac[135,114] + frac[131,114] >= 0.9
 cov_872_60: frac[114,60] + frac[196,60] + frac[193,60] + frac[194,60]
   + frac[85,60] + frac[197,60] + frac[198,60] >= 0.9
 cov_873_117: frac[195,117] + frac[159,117] + frac[194,117] + frac[118,117]
   + frac[197,117] >= 1
 cov_874_186: frac[197,186] + frac[79,186] + frac[196,186] + frac[198,186]
   + frac[158,186] + frac[193,186] + frac[195,186] >= 0.9
 cov_875_48: frac[194,48] + frac[192,48] + frac[136,48] + frac[198,48]
   + frac[196,48] + frac[1,48] + frac[197,48] >= 0.7
 cov_876_205: frac[37,205] + frac[193,205] + frac[192,205] + frac[196,205]
   + frac[88,205] >= 0.9
 cov_877_156: frac[109,156] + frac[196,156] + frac[194,156] + frac[197,156]
   + frac[192,156] + frac[9,156] + frac[198,156] >= 0.7
 cov_878_94: frac[83,94] + frac[193,94] + frac[66,94] >= 0.8
 cov_879_253: frac[12,253] + frac[192,253] + frac[197,253] + frac[198,253]
   + frac[147,253] + frac[196,253] + frac[195,253] >= 0.8
 cov_880_90: frac[197,90] + frac[48,90] + frac[194,90] + frac[193,90]
   + frac[196,90] + frac[198,90] + frac[99,90] >= 0.7
 cov_881_41: frac[193,41] + frac[196,41] + frac[197,41] + frac[195,41]
   + frac[94,41] + frac[198,41] + frac[181,41] >= 0.9
 cov_882_94: frac[196,94] + frac[198,94] + frac[195,94] + frac[197,94]
   + frac[45,94] + frac[192,94] + frac[180,94] >= 0.9
 cov_883_65: frac[44,65] + frac[197,65] + frac[195,65] + frac[198,65]
   + frac[196,65] + frac[187,65] + frac[192,65] >= 0.9
 cov_884_244: frac[32,244] + frac[192,244] + frac[8,244] >= 0.7
 cov_885_226: frac[197,226] + frac[194,226] + frac[107,226] + frac[156,226]
   + frac[195,226] >= 0.8
 cov_886_152: frac[194,152] + frac[197,152] + frac[198,152] + frac[95,152]
   + frac[193,152] + frac[120,152] + frac[196,152] >= 0.7
 cov_887_198: frac[192,198] + frac[195,198] + frac[197,198] + frac[198,198]
   + frac[156,198] + frac[17,198] + frac[196,198] >= 0.7
 cov_888_58: frac[193,58] + frac[194,58] + frac[78,58] + frac[197,58]
   + frac[196,58] + frac[198,58] + frac[121,58] >= 0.7
 cov_889_226: frac[197,226] + frac[118,226] + frac[182,226] + frac[194,226]
   + frac[195,226] >= 1
 cov_890_153: frac[195,153] + frac[146,153] + frac[194,153] + frac[119,153]
   + frac[197,153] >= 0.9
 cov_891_198: frac[192,198] + frac[106,198] + frac[197,198] + frac[194,198]
   + frac[198,198] + frac[45,198] + frac[196,198] >= 1
 cov_892_158: frac[36,158] + frac[18,158] + frac[192,158] >= 0.9
 cov_893_41: frac[118,41] + frac[194,41] + frac[100,41] >= 0.8
 cov_894_153: frac[141,153] + frac[163,153] + frac[195,153] + frac[194,153]
   + frac[197,153] >= 0.8
 cov_895_71: frac[195,71] + frac[116,71] + frac[197,71] + frac[170,71]
   + frac[194,71] >= 1
 cov_896_102: frac[187,102] + frac[184,102] + frac[195,102] >= 0.7
 cov_897_174: frac[192,174] + frac[127,174] + frac[196,174] + frac[194,174]
   + frac[198,174] + frac[15,174] + frac[197,174] >= 1
 cov_898_138: frac[192,138] + frac[196,138] + frac[53,138] + frac[28,138]
   + frac[193,138] >= 0.9
 cov_899_216: frac[193,216] + frac[194,216] + frac[198,216] + frac[137,216]
   + frac[196,216] + frac[55,216] + frac[197,216] >= 0.9
 cov_900_239: frac[105,239] + frac[194,239] + frac[96,239] >= 0.8
 cov_901_238: frac[141,238] + frac[194,238] + frac[115,238] >= 0.9
 cov_902_152: frac[192,152] + frac[193,152] + frac[196,152] + frac[8,152]
   + frac[53,152] >= 1
 cov_903_47: frac[84,47] + frac[193,47] + frac[65,47] >= 0.8
 cov_904_72: frac[38,72] + frac[40,72] + frac[192,72] >= 1
 cov_905_147: frac[118,147] + frac[194,147] + frac[193,147] + frac[198,147]
   + frac[197,147] + frac[68,147] + frac[196,147] >= 0.8
 cov_906_68: frac[197,68] + frac[101,68] + frac[194,68] + frac[148,68]
   + frac[195,68] >= 0.8
 cov_907_37: frac[196,37] + frac[52,37] + frac[192,37] + frac[193,37]
   + frac[16,37] >= 0.8
 cov_908_111: frac[193,111] + frac[197,111] + frac[194,111] + frac[112,111]
   + frac[198,111] + frac[74,111] + frac[196,111] >= 0.9
 cov_909_123: frac[196,123] + frac[3,123] + frac[58,123] + frac[192,123]
   + frac[193,123] >= 0.8
 cov_910_155: frac[196,155] + frac[193,155] + frac[198,155] + frac[195,155]
   + frac[49,155] + frac[197,155] + frac[162,155] >= 0.7
 cov_911_226: frac[124,226] + frac[194,226] + frac[116,226] >= 0.8
 cov_912_220: frac[148,220] + frac[112,220] + frac[197,220] + frac[195,220]
   + frac[194,220] >= 0.8
 cov_913_26: frac[58,26] + frac[131,26] + frac[194,26] + frac[197,26]
   + frac[196,26] + frac[198,26] + frac[193,26] >= 0.9
 cov_914_93: frac[196,93] + frac[41,93] + frac[179,93] + frac[192,93]
   + frac[197,93] + frac[198,93] + frac[195,93] >= 0.8
 cov_915_244: frac[180,244] + frac[152,244] + frac[195,244] >= 1
 cov_916_104: frac[7,104] + frac[196,104] + frac[197,104] + frac[115,104]
   + frac[198,104] + frac[194,104] + frac[192,104] >= 0.8
 cov_917_19: frac[197,19] + frac[29,19] + frac[192,19] + frac[196,19]
   + frac[194,19] + frac[117,19] + frac[198,19] >= 0.8
 cov_918_206: frac[194,206] + frac[195,206] + frac[197,206] + frac[155,206]
   + frac[143,206] >= 1
 cov_919_86: frac[104,86] + frac[194,86] + frac[195,86] + frac[197,86]
   + frac[144,86] >= 0.9
 cov_920_174: frac[192,174] + frac[196,174] + frac[198,174] + frac[157,174]
   + frac[25,174] + frac[195,174] + frac[197,174] >= 0.8
 cov_921_245: frac[197,245] + frac[190,245] + frac[192,245] + frac[196,245]
   + frac[195,245] + frac[38,245] + frac[198,245] >= 0.9
 cov_922_227: frac[137,227] + frac[194,227] + frac[97,227] >= 0.9
 cov_923_161: frac[21,161] + frac[196,161] + frac[198,161] + frac[184,161]
   + frac[192,161] + frac[197,161] + frac[195,161] >= 1
 cov_924_112: frac[32,112] + frac[192,112] + frac[170,112] + frac[198,112]
   + frac[196,112] + frac[197,112] + frac[195,112] >= 0.7
 cov_925_53: frac[14,53] + frac[15,53] + frac[192,53] >= 0.7
 cov_926_167: frac[196,167] + frac[61,167] + frac[192,167] + frac[8,167]
   + frac[193,167] >= 0.8
 cov_927_43: frac[126,43] + frac[67,43] + frac[196,43] + frac[198,43]
   + frac[194,43] + frac[193,43] + frac[197,43] >= 0.7
 cov_928_218: frac[193,218] + frac[63,218] + frac[196,218] + frac[192,218]
   + frac[28,218] >= 0.9
 cov_929_204: frac[198,204] + frac[192,204] + frac[196,204] + frac[197,204]
   + frac[177,204] + frac[16,204] + frac[195,204] >= 0.7
 cov_930_173: frac[195,173] + frac[174,173] + frac[163,173] >= 0.9
 cov_931_141: frac[195,141] + frac[167,141] + frac[184,141] >= 0.7
 cov_932_136: frac[105,136] + frac[136,136] + frac[194,136] >= 0.9
 cov_933_142: frac[74,142] + frac[193,142] + frac[192,142] + frac[196,142]
   + frac[9,142] >= 0.8
 cov_934_15: frac[192,15] + frac[196,15] + frac[128,15] + frac[194,15]
   + frac[198,15] + frac[8,15] + frac[197,15] >= 0.8
 cov_935_206: frac[14,206] + frac[198,206] + frac[195,206] + frac[197,206]
   + frac[186,206] + frac[192,206] + frac[196,206] >= 1
 cov_936_213: frac[193,213] + frac[58,213] + frac[83,213] >= 1
 cov_937_183: frac[153,183] + frac[35,183] + frac[195,183] + frac[197,183]
   + frac[198,183] + frac[192,183] + frac[196,183] >= 0.9
 cov_938_185: frac[177,185] + frac[195,185] + frac[186,185] >= 0.9
 cov_939_157: frac[197,157] + frac[121,157] + frac[194,157] + frac[195,157]
   + frac[186,157] >= 0.9
 cov_940_183: frac[195,183] + frac[179,183] + frac[197,183] + frac[198,183]
   + frac[192,183] + frac[196,183] + frac[3,183] >= 0.7
 cov_941_11: frac[198,11] + frac[194,11] + frac[28,11] + frac[134,11]
   + frac[197,11] + frac[192,11] + frac[196,11] >= 0.8
 cov_942_177: frac[196,177] + frac[198,177] + frac[197,177] + frac[43,177]
   + frac[195,177] + frac[192,177] + frac[153,177] >= 0.9
 cov_943_114: frac[194,114] + frac[198,114] + frac[197,114] + frac[192,114]
   + frac[18,114] + frac[104,114] + frac[196,114] >= 0.8
 cov_944_178: frac[197,178] + frac[194,178] + frac[142,178] + frac[195,178]
   + frac[145,178] >= 0.9
 cov_945_90: frac[71,90] + frac[197,90] + frac[108,90] + frac[194,90]
   + frac[193,90] + frac[196,90] + frac[198,90] >= 0.7
 cov_946_170: frac[136,170] + frac[104,170] + frac[194,170] >= 1
 cov_947_158: frac[194,158] + frac[197,158] + frac[130,158] + frac[196,158]
   + frac[46,158] + frac[198,158] + frac[192,158] >= 0.9
 cov_948_242: frac[192,242] + frac[17,242] + frac[196,242] + frac[181,242]
   + frac[195,242] + frac[197,242] + frac[198,242] >= 1
 cov_949_229: frac[193,229] + frac[196,229] + frac[194,229] + frac[198,229]
   + frac[121,229] + frac[69,229] + frac[197,229] >= 1
 cov_950_182: frac[102,182] + frac[198,182] + frac[192,182] + frac[197,182]
   + frac[196,182] + frac[34,182] + frac[194,182] >= 1
 cov_951_129: frac[198,129] + frac[192,129] + frac[194,129] + frac[197,129]
   + frac[34,129] + frac[109,129] + frac[196,129] >= 0.8
 cov_952_238: frac[192,238] + frac[196,238] + frac[53,238] + frac[193,238]
   + frac[9,238] >= 1
 cov_953_239: frac[185,239] + frac[195,239] + frac[161,239] >= 0.7
 cov_954_78: frac[192,78] + frac[196,78] + frac[96,78] + frac[14,78]
   + frac[194,78] + frac[198,78] + frac[197,78] >= 0.9
 cov_955_107: frac[167,107] + frac[195,107] + frac[193,107] + frac[196,107]
   + frac[55,107] + frac[198,107] + frac[197,107] >= 1
 cov_956_104: frac[196,104] + frac[6,104] + frac[197,104] + frac[117,104]
   + frac[198,104] + frac[194,104] + frac[192,104] >= 1
 cov_957_32: frac[197,32] + frac[187,32] + frac[195,32] + frac[133,32]
   + frac[194,32] >= 0.9
 cov_958_244: frac[19,244] + frac[192,244] + frac[35,244] >= 0.7
 cov_959_203: frac[184,203] + frac[59,203] + frac[195,203] + frac[193,203]
   + frac[196,203] + frac[197,203] + frac[198,203] >= 0.7
 cov_960_204: frac[103,204] + frac[194,204] + frac[111,204] >= 1
 cov_961_238: frac[192,238] + frac[60,238] + frac[196,238] + frac[193,238]
   + frac[15,238] >= 1
 cov_962_153: frac[140,153] + frac[194,153] + frac[98,153] >= 0.7
 cov_963_65: frac[53,65] + frac[24,65] + frac[193,65] + frac[196,65]
   + frac[192,65] >= 0.9
 cov_964_56: frac[192,56] + frac[37,56] + frac[193,56] + frac[196,56]
   + frac[77,56] >= 0.9
 cov_965_184: frac[196,184] + frac[61,184] + frac[197,184] + frac[173,184]
   + frac[195,184] + frac[198,184] + frac[193,184] >= 0.9
 cov_966_89: frac[58,89] + frac[193,89] + frac[92,89] >= 0.7
 cov_967_225: frac[129,225] + frac[197,225] + frac[196,225] + frac[198,225]
   + frac[192,225] + frac[194,225] + frac[25,225] >= 0.7
 cov_968_208: frac[57,208] + frac[192,208] + frac[193,208] + frac[196,208]
   + frac[29,208] >= 0.8
 cov_969_181: frac[131,181] + frac[194,181] + frac[197,181] + frac[160,181]
   + frac[195,181] >= 0.8
 cov_970_205: frac[148,205] + frac[193,205] + frac[197,205] + frac[198,205]
   + frac[74,205] + frac[196,205] + frac[195,205] >= 0.7
 cov_971_59: frac[193,59] + frac[56,59] + frac[71,59] >= 0.8
 cov_972_41: frac[159,41] + frac[196,41] + frac[197,41] + frac[195,41]
   + frac[47,41] + frac[198,41] + frac[192,41] >= 0.9
 cov_973_68: frac[196,68] + frac[197,68] + frac[194,68] + frac[198,68]
   + frac[193,68] + frac[71,68] + frac[106,68] >= 1
 cov_974_214: frac[183,214] + frac[103,214] + frac[194,214] + frac[195,214]
   + frac[197,214] >= 0.9
 cov_975_254: frac[193,254] + frac[50,254] + frac[150,254] + frac[198,254]
   + frac[195,254] + frac[197,254] + frac[196,254] >= 0.9
 cov_976_190: frac[198,190] + frac[115,190] + frac[72,190] + frac[193,190]
   + frac[194,190] + frac[197,190] + frac[196,190] >= 0.9
 cov_977_227: frac[197,227] + frac[48,227] + frac[193,227] + frac[198,227]
   + frac[162,227] + frac[195,227] + frac[196,227] >= 0.7
 cov_978_33: frac[192,33] + frac[197,33] + frac[195,33] + frac[165,33]
   + frac[198,33] + frac[196,33] + frac[42,33] >= 0.8
 cov_979_39: frac[193,39] + frac[58,39] + frac[69,39] >= 1
 cov_980_34: frac[196,34] + frac[192,34] + frac[198,34] + frac[194,34]
   + frac[43,34] + frac[131,34] + frac[197,34] >= 0.7
 cov_981_90: frac[74,90] + frac[197,90] + frac[194,90] + frac[193,90]
   + frac[196,90] + frac[198,90] + frac[106,90] >= 1
 cov_982_113: frac[193,113] + frac[109,113] + frac[194,113] + frac[197,113]
   + frac[198,113] + frac[196,113] + frac[74,113] >= 0.8
 cov_983_87: frac[96,87] + frac[196,87] + frac[193,87] + frac[194,87]
   + frac[197,87] + frac[198,87] + frac[75,87] >= 0.8
 cov_984_227: frac[193,227] + frac[56,227] + frac[196,227] + frac[20,227]
   + frac[192,227] >= 1
 cov_985_100: frac[133,100] + frac[185,100] + frac[197,100] + frac[194,100]
   + frac[195,100] >= 0.7
 cov_986_79: frac[65,79] + frac[195,79] + frac[196,79] + frac[193,79]
   + frac[197,79] + frac[148,79] + frac[198,79] >= 0.9
 cov_987_89: frac[198,89] + frac[197,89] + frac[195,89] + frac[168,89]
   + frac[196,89] + frac[193,89] + frac[77,89] >= 1
 cov_988_236: frac[13,236] + frac[38,236] + frac[192,236] >= 0.8
 cov_989_182: frac[108,182] + frac[194,182] + frac[121,182] >= 1
 cov_990_205: frac[197,205] + frac[35,205] + frac[194,205] + frac[114,205]
   + frac[198,205] + frac[192,205] + frac[196,205] >= 0.8
 cov_991_161: frac[194,161] + frac[197,161] + frac[132,161] + frac[195,161]
   + frac[173,161] >= 0.9
 cov_992_84: frac[157,84] + frac[194,84] + frac[122,84] + frac[197,84]
   + frac[195,84] >= 0.8
 cov_993_226: frac[166,226] + frac[168,226] + frac[195,226] >= 0.7
 cov_994_202: frac[192,202] + frac[196,202] + frac[198,202] + frac[0,202]
   + frac[194,202] + frac[98,202] + frac[197,202] >= 0.8
 cov_995_104: frac[196,104] + frac[197,104] + frac[195,104] + frac[54,104]
   + frac[174,104] + frac[198,104] + frac[193,104] >= 1
 cov_996_129: frac[192,129] + frac[31,129] + frac[38,129] >= 0.7
 cov_997_148: frac[142,148] + frac[198,148] + frac[193,148] + frac[197,148]
   + frac[93,148] + frac[194,148] + frac[196,148] >= 0.9
 cov_998_93: frac[196,93] + frac[193,93] + frac[194,93] + frac[118,93]
   + frac[197,93] + frac[198,93] + frac[65,93] >= 0.7
 cov_999_131: frac[196,131] + frac[192,131] + frac[43,131] + frac[198,131]
   + frac[194,131] + frac[197,131] + frac[136,131] >= 1
 cov_1000_239: frac[87,239] + frac[193,239] + frac[88,239] >= 1
 cov_1001_231: frac[103,231] + frac[194,231] + frac[138,231] >= 1
 cov_1002_205: frac[194,205] + frac[98,205] + frac[109,205] >= 1
 cov_1003_144: frac[195,144] + frac[118,144] + frac[194,144]
   + frac[160,144] + frac[197,144] >= 1
 cov_1004_143: frac[154,143] + frac[163,143] + frac[195,143] >= 1
 cov_1005_219: frac[70,219] + frac[194,219] + frac[113,219] + frac[196,219]
   + frac[197,219] + frac[198,219] + frac[193,219] >= 0.9
 cov_1006_58: frac[132,58] + frac[120,58] + frac[194,58] >= 1
 cov_1007_181: frac[165,181] + frac[192,181] + frac[196,181] + frac[20,181]
   + frac[197,181] + frac[198,181] + frac[195,181] >= 0.7
 cov_1008_75: frac[196,75] + frac[198,75] + frac[197,75] + frac[7,75]
   + frac[179,75] + frac[195,75] + frac[192,75] >= 0.9
 cov_1009_56: frac[192,56] + frac[197,56] + frac[198,56] + frac[28,56]
   + frac[196,56] + frac[107,56] + frac[194,56] >= 0.8
 cov_1010_58: frac[193,58] + frac[195,58] + frac[152,58] + frac[58,58]
   + frac[197,58] + frac[196,58] + frac[198,58] >= 1
 cov_1011_111: frac[193,111] + frac[28,111] + frac[64,111] + frac[192,111]
   + frac[196,111] >= 0.7
 cov_1012_188: frac[92,188] + frac[195,188] + frac[198,188] + frac[196,188]
   + frac[193,188] + frac[190,188] + frac[197,188] >= 1
 cov_1013_195: frac[195,195] + frac[196,195] + frac[173,195]
   + frac[197,195] + frac[193,195] + frac[85,195] + frac[198,195] >= 1
 cov_1014_27: frac[192,27] + frac[10,27] + frac[78,27] + frac[196,27]
   + frac[193,27] >= 0.7
 cov_1015_74: frac[196,74] + frac[198,74] + frac[194,74] + frac[197,74]
   + frac[192,74] + frac[22,74] + frac[130,74] >= 0.7
 cov_1016_196: frac[123,196] + frac[53,196] + frac[196,196] + frac[197,196]
   + frac[194,196] + frac[198,196] + frac[193,196] >= 1
 cov_1017_27: frac[103,27] + frac[197,27] + frac[194,27] + frac[196,27]
   + frac[193,27] + frac[198,27] + frac[93,27] >= 0.7
 cov_1018_67: frac[196,67] + frac[192,67] + frac[197,67] + frac[15,67]
   + frac[198,67] + frac[195,67] + frac[147,67] >= 0.7
 cov_1019_30: frac[198,30] + frac[90,30] + frac[125,30] + frac[193,30]
   + frac[194,30] + frac[197,30] + frac[196,30] >= 0.9
 cov_1020_100: frac[25,100] + frac[79,100] + frac[196,100] + frac[192,100]
   + frac[193,100] >= 1
 cov_1021_166: frac[179,166] + frac[99,166] + frac[194,166] + frac[195,166]
   + frac[197,166] >= 0.7
 cov_1022_106: frac[194,106] + frac[114,106] + frac[108,106] >= 1
 cov_1023_251: frac[131,251] + frac[132,251] + frac[194,251] >= 0.7
 cov_1024_184: frac[142,184] + frac[194,184] + frac[136,184] >= 0.9
 cov_1025_179: frac[196,179] + frac[194,179] + frac[117,179]
   + frac[193,179] + frac[198,179] + frac[197,179] + frac[93,179] >= 0.9
 cov_1026_91: frac[192,91] + frac[193,91] + frac[84,91] + frac[196,91]
   + frac[39,91] >= 0.7
 cov_1027_39: frac[195,39] + frac[194,39] + frac[197,39] + frac[138,39]
   + frac[169,39] >= 0.9
 cov_1028_222: frac[197,222] + frac[194,222] + frac[131,222]
   + frac[196,222] + frac[78,222] + frac[193,222] + frac[198,222] >= 0.8
 cov_1029_164: frac[193,164] + frac[63,164] + frac[80,164] >= 1
 cov_1030_92: frac[195,92] + frac[183,92] + frac[192,92] + frac[2,92]
   + frac[198,92] + frac[196,92] + frac[197,92] >= 0.7
 cov_1031_243: frac[165,243] + frac[138,243] + frac[195,243]
   + frac[197,243] + frac[194,243] >= 1
 cov_1032_99: frac[144,99] + frac[195,99] + frac[196,99] + frac[192,99]
   + frac[197,99] + frac[198,99] + frac[45,99] >= 1
 cov_1033_45: frac[96,45] + frac[194,45] + frac[198,45] + frac[192,45]
   + frac[196,45] + frac[197,45] + frac[45,45] >= 0.9
 cov_1034_51: frac[43,51] + frac[193,51] + frac[64,51] + frac[192,51]
   + frac[196,51] >= 0.9
 cov_1035_15: frac[192,15] + frac[196,15] + frac[198,15] + frac[13,15]
   + frac[195,15] + frac[144,15] + frac[197,15] >= 0.7
 cov_1036_0: frac[89,0] + frac[196,0] + frac[192,0] + frac[13,0]
   + frac[193,0] >= 1
 cov_1037_177: frac[101,177] + frac[197,177] + frac[195,177]
   + frac[194,177] + frac[178,177] >= 0.7
 cov_1038_34: frac[196,34] + frac[192,34] + frac[146,34] + frac[198,34]
   + frac[195,34] + frac[18,34] + frac[197,34] >= 0.7
 cov_1039_201: frac[194,201] + frac[119,201] + frac[108,201] >= 0.8
 cov_1040_217: frac[196,217] + frac[193,217] + frac[146,217] + frac[82,217]
   + frac[197,217] + frac[198,217] + frac[195,217] >= 0.7
 cov_1041_39: frac[54,39] + frac[94,39] + frac[193,39] >= 1
 cov_1042_92: frac[194,92] + frac[135,92] + frac[121,92] >= 0.9
 cov_1043_108: frac[40,108] + frac[198,108] + frac[192,108] + frac[196,108]
   + frac[197,108] + frac[162,108] + frac[195,108] >= 1
 cov_1044_198: frac[195,198] + frac[138,198] + frac[197,198]
   + frac[194,198] + frac[152,198] >= 0.7
 cov_1045_3: frac[41,3] + frac[26,3] + frac[192,3] >= 0.7
 cov_1046_19: frac[197,19] + frac[191,19] + frac[141,19] + frac[194,19]
   + frac[195,19] >= 0.7
 cov_1047_49: frac[192,49] + frac[185,49] + frac[198,49] + frac[196,49]
   + frac[197,49] + frac[195,49] + frac[21,49] >= 0.7
 cov_1048_11: frac[59,11] + frac[193,11] + frac[192,11] + frac[34,11]
   + frac[196,11] >= 0.9
 cov_1049_194: frac[156,194] + frac[195,194] + frac[166,194] >= 1
 cov_1050_121: frac[31,121] + frac[192,121] + frac[40,121] >= 0.9
 cov_1051_149: frac[161,149] + frac[195,149] + frac[171,149] >= 0.7
 cov_1052_16: frac[196,16] + frac[15,16] + frac[56,16] + frac[193,16]
   + frac[192,16] >= 1
 cov_1053_7: frac[179,7] + frac[196,7] + frac[192,7] + frac[8,7]
   + frac[195,7] + frac[198,7] + frac[197,7] >= 1
 cov_1054_134: frac[194,134] + frac[198,134] + frac[197,134]
   + frac[196,134] + frac[193,134] + frac[102,134] + frac[67,134] >= 0.9
 cov_1055_16: frac[198,16] + frac[196,16] + frac[88,16] + frac[115,16]
   + frac[193,16] + frac[197,16] + frac[194,16] >= 0.8
 cov_1056_222: frac[197,222] + frac[192,222] + frac[194,222]
   + frac[196,222] + frac[9,222] + frac[198,222] + frac[135,222] >= 0.8
 cov_1057_11: frac[198,11] + frac[195,11] + frac[197,11] + frac[170,11]
   + frac[192,11] + frac[6,11] + frac[196,11] >= 0.7
 cov_1058_41: frac[14,41] + frac[196,41] + frac[160,41] + frac[197,41]
   + frac[195,41] + frac[198,41] + frac[192,41] >= 0.7
 cov_1059_99: frac[110,99] + frac[196,99] + frac[197,99] + frac[193,99]
   + frac[56,99] + frac[198,99] + frac[194,99] >= 0.9
 cov_1060_140: frac[192,140] + frac[196,140] + frac[37,140] + frac[193,140]
   + frac[90,140] >= 1
 cov_1061_105: frac[197,105] + frac[194,105] + frac[192,105] + frac[17,105]
   + frac[123,105] + frac[196,105] + frac[198,105] >= 0.8
 cov_1062_55: frac[192,55] + frac[196,55] + frac[194,55] + frac[33,55]
   + frac[122,55] + frac[198,55] + frac[197,55] >= 1
 cov_1063_8: frac[194,8] + frac[134,8] + frac[137,8] >= 0.8
 cov_1064_135: frac[194,135] + frac[96,135] + frac[107,135] >= 0.8
 cov_1065_180: frac[192,180] + frac[45,180] + frac[16,180] >= 0.7
 cov_1066_202: frac[196,202] + frac[77,202] + frac[198,202] + frac[163,202]
   + frac[195,202] + frac[193,202] + frac[197,202] >= 0.7
 cov_1067_90: frac[197,90] + frac[144,90] + frac[123,90] + frac[194,90]
   + frac[195,90] >= 1
 cov_1068_114: frac[193,114] + frac[57,114] + frac[192,114] + frac[6,114]
   + frac[196,114] >= 0.9
 cov_1069_240: frac[196,240] + frac[197,240] + frac[198,240] + frac[74,240]
   + frac[133,240] + frac[193,240] + frac[194,240] >= 0.9
 cov_1070_225: frac[117,225] + frac[194,225] + frac[127,225] >= 0.9
 cov_1071_172: frac[99,172] + frac[194,172] + frac[109,172] >= 1
 cov_1072_221: frac[197,221] + frac[140,221] + frac[198,221]
   + frac[196,221] + frac[65,221] + frac[193,221] + frac[194,221] >= 0.8
 cov_1073_119: frac[133,119] + frac[194,119] + frac[96,119] >= 0.8
 cov_1074_55: frac[196,55] + frac[194,55] + frac[70,55] + frac[123,55]
   + frac[193,55] + frac[198,55] + frac[197,55] >= 0.8
 cov_1075_229: frac[193,229] + frac[71,229] + frac[48,229] >= 0.7
 cov_1076_71: frac[51,71] + frac[196,71] + frac[197,71] + frac[123,71]
   + frac[193,71] + frac[194,71] + frac[198,71] >= 0.9
 cov_1077_241: frac[60,241] + frac[193,241] + frac[87,241] >= 0.8
 cov_1078_3: frac[10,3] + frac[192,3] + frac[6,3] >= 0.7
 cov_1079_185: frac[196,185] + frac[197,185] + frac[13,185] + frac[195,185]
   + frac[192,185] + frac[157,185] + frac[198,185] >= 1
 cov_1080_154: frac[107,154] + frac[198,154] + frac[193,154] + frac[93,154]
   + frac[194,154] + frac[197,154] + frac[196,154] >= 0.8
 cov_1081_47: frac[131,47] + frac[197,47] + frac[10,47] + frac[192,47]
   + frac[196,47] + frac[194,47] + frac[198,47] >= 1
 cov_1082_194: frac[195,194] + frac[124,194] + frac[170,194]
   + frac[197,194] + frac[194,194] >= 0.9
 cov_1083_214: frac[198,214] + frac[159,214] + frac[196,214]
   + frac[193,214] + frac[195,214] + frac[81,214] + frac[197,214] >= 1
 cov_1084_139: frac[197,139] + frac[62,139] + frac[196,139] + frac[198,139]
   + frac[194,139] + frac[114,139] + frac[193,139] >= 0.7
 cov_1085_108: frac[192,108] + frac[193,108] + frac[196,108] + frac[75,108]
   + frac[17,108] >= 0.7
 cov_1086_95: frac[193,95] + frac[57,95] + frac[195,95] + frac[196,95]
   + frac[164,95] + frac[197,95] + frac[198,95] >= 1
 cov_1087_169: frac[31,169] + frac[192,169] + frac[86,169] + frac[193,169]
   + frac[196,169] >= 0.9
 cov_1088_42: frac[192,42] + frac[197,42] + frac[196,42] + frac[198,42]
   + frac[147,42] + frac[11,42] + frac[195,42] >= 0.9
 cov_1089_148: frac[198,148] + frac[24,148] + frac[197,148] + frac[194,148]
   + frac[136,148] + frac[192,148] + frac[196,148] >= 0.8
 cov_1090_100: frac[159,100] + frac[196,100] + frac[197,100]
   + frac[198,100] + frac[195,100] + frac[192,100] + frac[17,100] >= 0.8
 cov_1091_160: frac[196,160] + frac[198,160] + frac[29,160] + frac[192,160]
   + frac[194,160] + frac[124,160] + frac[197,160] >= 0.8
 cov_1092_113: frac[193,113] + frac[92,113] + frac[62,113] >= 0.9
 cov_1093_40: frac[187,40] + frac[193,40] + frac[197,40] + frac[195,40]
   + frac[198,40] + frac[56,40] + frac[196,40] >= 0.7
 cov_1094_42: frac[197,42] + frac[93,42] + frac[196,42] + frac[198,42]
   + frac[194,42] + frac[115,42] + frac[193,42] >= 0.7
 cov_1095_63: frac[192,63] + frac[196,63] + frac[80,63] + frac[193,63]
   + frac[42,63] >= 0.9
 cov_1096_250: frac[197,250] + frac[188,250] + frac[196,250]
   + frac[198,250] + frac[195,250] + frac[37,250] + frac[192,250] >= 1
 cov_1097_162: frac[194,162] + frac[135,162] + frac[197,162] + frac[90,162]
   + frac[196,162] + frac[193,162] + frac[198,162] >= 0.7
 cov_1098_135: frac[193,135] + frac[196,135] + frac[67,135] + frac[35,135]
   + frac[192,135] >= 0.9
 cov_1099_29: frac[146,29] + frac[195,29] + frac[63,29] + frac[193,29]
   + frac[198,29] + frac[196,29] + frac[197,29] >= 1
 cov_1100_37: frac[196,37] + frac[192,37] + frac[54,37] + frac[193,37]
   + frac[20,37] >= 0.7
 cov_1101_89: frac[136,89] + frac[126,89] + frac[194,89] >= 0.9
 cov_1102_188: frac[192,188] + frac[198,188] + frac[127,188] + frac[46,188]
   + frac[196,188] + frac[194,188] + frac[197,188] >= 0.7
 cov_1103_202: frac[92,202] + frac[77,202] + frac[193,202] >= 1
 cov_1104_181: frac[194,181] + frac[192,181] + frac[196,181]
   + frac[197,181] + frac[198,181] + frac[10,181] + frac[112,181] >= 0.9
 cov_1105_184: frac[192,184] + frac[27,184] + frac[28,184] >= 1
 cov_1106_167: frac[196,167] + frac[192,167] + frac[194,167]
   + frac[197,167] + frac[198,167] + frac[105,167] + frac[46,167] >= 0.8
 cov_1107_192: frac[196,192] + frac[197,192] + frac[198,192]
   + frac[192,192] + frac[194,192] + frac[110,192] + frac[32,192] >= 0.7
 cov_1108_101: frac[197,101] + frac[195,101] + frac[196,101] + frac[5,101]
   + frac[198,101] + frac[192,101] + frac[187,101] >= 0.8
 cov_1109_153: frac[195,153] + frac[194,153] + frac[111,153]
   + frac[175,153] + frac[197,153] >= 0.8
 cov_1110_66: frac[94,66] + frac[90,66] + frac[193,66] >= 0.9
 cov_1111_82: frac[194,82] + frac[125,82] + frac[122,82] >= 0.8
 cov_1112_219: frac[194,219] + frac[128,219] + frac[131,219] >= 0.9
 cov_1113_180: frac[195,180] + frac[196,180] + frac[198,180]
   + frac[193,180] + frac[197,180] + frac[150,180] + frac[56,180] >= 0.8
 cov_1114_58: frac[192,58] + frac[194,58] + frac[103,58] + frac[197,58]
   + frac[196,58] + frac[198,58] + frac[40,58] >= 1
 cov_1115_176: frac[197,176] + frac[188,176] + frac[195,176]
   + frac[192,176] + frac[36,176] + frac[196,176] + frac[198,176] >= 0.7
 cov_1116_48: frac[192,48] + frac[196,48] + frac[15,48] + frac[193,48]
   + frac[78,48] >= 1
 cov_1117_54: frac[192,54] + frac[197,54] + frac[196,54] + frac[34,54]
   + frac[198,54] + frac[153,54] + frac[195,54] >= 0.9
 cov_1118_168: frac[198,168] + frac[194,168] + frac[196,168]
   + frac[137,168] + frac[193,168] + frac[197,168] + frac[93,168] >= 1
 cov_1119_142: frac[89,142] + frac[195,142] + frac[193,142] + frac[197,142]
   + frac[196,142] + frac[191,142] + frac[198,142] >= 0.9
 cov_1120_91: frac[93,91] + frac[193,91] + frac[56,91] >= 0.9
 cov_1121_16: frac[198,16] + frac[196,16] + frac[193,16] + frac[77,16]
   + frac[197,16] + frac[195,16] + frac[162,16] >= 0.8
 cov_1122_2: frac[198,2] + frac[194,2] + frac[43,2] + frac[142,2]
   + frac[197,2] + frac[196,2] + frac[192,2] >= 0.8
 cov_1123_181: frac[17,181] + frac[111,181] + frac[194,181] + frac[192,181]
   + frac[196,181] + frac[197,181] + frac[198,181] >= 1
 cov_1124_250: frac[197,250] + frac[73,250] + frac[193,250] + frac[196,250]
   + frac[147,250] + frac[198,250] + frac[195,250] >= 0.8
 cov_1125_125: frac[197,125] + frac[195,125] + frac[135,125]
   + frac[191,125] + frac[194,125] >= 0.8
 cov_1126_90: frac[197,90] + frac[159,90] + frac[193,90] + frac[196,90]
   + frac[198,90] + frac[195,90] + frac[60,90] >= 0.9
 cov_1127_36: frac[196,36] + frac[34,36] + frac[192,36] + frac[60,36]
   + frac[193,36] >= 1
 cov_1128_90: frac[192,90] + frac[6,90] + frac[10,90] >= 1
 cov_1129_91: frac[197,91] + frac[152,91] + frac[117,91] + frac[194,91]
   + frac[195,91] >= 0.9
 cov_1130_5: frac[195,5] + frac[178,5] + frac[182,5] >= 0.7
 cov_1131_63: frac[155,63] + frac[195,63] + frac[161,63] >= 0.7
 cov_1132_91: frac[79,91] + frac[192,91] + frac[193,91] + frac[196,91]
   + frac[10,91] >= 0.9
 cov_1133_23: frac[77,23] + frac[193,23] + frac[58,23] >= 0.8
 cov_1134_53: frac[196,53] + frac[193,53] + frac[198,53] + frac[74,53]
   + frac[197,53] + frac[195,53] + frac[150,53] >= 1
 cov_1135_182: frac[125,182] + frac[194,182] + frac[118,182] >= 0.7
 cov_1136_144: frac[195,144] + frac[165,144] + frac[169,144] >= 0.9
 cov_1137_171: frac[194,171] + frac[193,171] + frac[103,171]
   + frac[197,171] + frac[196,171] + frac[85,171] + frac[198,171] >= 0.7
 cov_1138_151: frac[192,151] + frac[196,151] + frac[128,151]
   + frac[194,151] + frac[3,151] + frac[197,151] + frac[198,151] >= 0.7
 cov_1139_164: frac[96,164] + frac[194,164] + frac[110,164] >= 0.7
 cov_1140_100: frac[121,100] + frac[196,100] + frac[197,100]
   + frac[194,100] + frac[198,100] + frac[61,100] + frac[193,100] >= 0.9
 cov_1141_161: frac[137,161] + frac[196,161] + frac[198,161]
   + frac[192,161] + frac[194,161] + frac[197,161] + frac[30,161] >= 0.9
 cov_1142_180: frac[195,180] + frac[192,180] + frac[196,180]
   + frac[180,180] + frac[198,180] + frac[10,180] + frac[197,180] >= 1
 cov_1143_150: frac[193,150] + frac[195,150] + frac[152,150]
   + frac[197,150] + frac[196,150] + frac[71,150] + frac[198,150] >= 1
 cov_1144_189: frac[194,189] + frac[193,189] + frac[198,189]
   + frac[196,189] + frac[90,189] + frac[197,189] + frac[112,189] >= 1
 cov_1145_119: frac[16,119] + frac[18,119] + frac[192,119] >= 0.9
 cov_1146_127: frac[192,127] + frac[26,127] + frac[2,127] >= 1
 cov_1147_242: frac[196,242] + frac[193,242] + frac[194,242] + frac[86,242]
   + frac[112,242] + frac[197,242] + frac[198,242] >= 0.9
 cov_1148_167: frac[196,167] + frac[64,167] + frac[128,167] + frac[194,167]
   + frac[197,167] + frac[198,167] + frac[193,167] >= 0.9
 cov_1149_38: frac[195,38] + frac[167,38] + frac[169,38] >= 0.8
 cov_1150_34: frac[196,34] + frac[193,34] + frac[79,34] + frac[198,34]
   + frac[194,34] + frac[96,34] + frac[197,34] >= 1
 cov_1151_123: frac[8,123] + frac[196,123] + frac[192,123] + frac[67,123]
   + frac[193,123] >= 0.8
 cov_1152_215: frac[178,215] + frac[195,215] + frac[197,215]
   + frac[194,215] + frac[137,215] >= 0.7
 cov_1153_252: frac[195,252] + frac[194,252] + frac[141,252]
   + frac[197,252] + frac[174,252] >= 1
 cov_1154_16: frac[198,16] + frac[196,16] + frac[138,16] + frac[197,16]
   + frac[41,16] + frac[194,16] + frac[192,16] >= 0.9
 cov_1155_123: frac[194,123] + frac[76,123] + frac[196,123] + frac[198,123]
   + frac[197,123] + frac[100,123] + frac[193,123] >= 0.7
 cov_1156_80: frac[168,80] + frac[197,80] + frac[195,80] + frac[119,80]
   + frac[194,80] >= 0.8
 cov_1157_60: frac[194,60] + frac[128,60] + frac[127,60] >= 0.8
 cov_1158_223: frac[198,223] + frac[192,223] + frac[147,223]
   + frac[196,223] + frac[195,223] + frac[197,223] + frac[16,223] >= 1
 cov_1159_5: frac[84,5] + frac[198,5] + frac[197,5] + frac[193,5]
   + frac[121,5] + frac[196,5] + frac[194,5] >= 0.7
 cov_1160_217: frac[107,217] + frac[139,217] + frac[194,217] >= 0.8
 cov_1161_33: frac[192,33] + frac[197,33] + frac[195,33] + frac[170,33]
   + frac[28,33] + frac[198,33] + frac[196,33] >= 0.9
 cov_1162_110: frac[194,110] + frac[112,110] + frac[128,110] >= 0.7
 cov_1163_180: frac[195,180] + frac[196,180] + frac[198,180]
   + frac[193,180] + frac[197,180] + frac[94,180] + frac[156,180] >= 0.7
 cov_1164_7: frac[107,7] + frac[194,7] + frac[117,7] >= 1
 cov_1165_112: frac[117,112] + frac[194,112] + frac[193,112]
   + frac[198,112] + frac[196,112] + frac[197,112] + frac[48,112] >= 1
 cov_1166_80: frac[193,80] + frac[173,80] + frac[48,80] + frac[198,80]
   + frac[196,80] + frac[197,80] + frac[195,80] >= 0.9
 cov_1167_50: frac[198,50] + frac[195,50] + frac[193,50] + frac[197,50]
   + frac[89,50] + frac[196,50] + frac[180,50] >= 0.7
 cov_1168_57: frac[192,57] + frac[9,57] + frac[18,57] >= 0.8
 cov_1169_252: frac[195,252] + frac[192,252] + frac[180,252]
   + frac[198,252] + frac[0,252] + frac[196,252] + frac[197,252] >= 1
 cov_1170_113: frac[24,113] + frac[33,113] + frac[192,113] >= 0.7
 cov_1171_56: frac[197,56] + frac[195,56] + frac[198,56] + frac[68,56]
   + frac[189,56] + frac[193,56] + frac[196,56] >= 1
 cov_1172_108: frac[33,108] + frac[192,108] + frac[193,108] + frac[196,108]
   + frac[71,108] >= 1
 cov_1173_43: frac[32,43] + frac[192,43] + frac[196,43] + frac[62,43]
   + frac[193,43] >= 0.8
 cov_1174_113: frac[102,113] + frac[194,113] + frac[192,113]
   + frac[197,113] + frac[1,113] + frac[198,113] + frac[196,113] >= 0.8
 cov_1175_75: frac[196,75] + frac[198,75] + frac[197,75] + frac[17,75]
   + frac[179,75] + frac[195,75] + frac[192,75] >= 0.7
 cov_1176_221: frac[197,221] + frac[102,221] + frac[157,221]
   + frac[195,221] + frac[194,221] >= 0.8
 cov_1177_194: frac[91,194] + frac[196,194] + frac[134,194] + frac[197,194]
   + frac[198,194] + frac[193,194] + frac[194,194] >= 0.8
 cov_1178_193: frac[195,193] + frac[174,193] + frac[171,193] >= 1
 cov_1179_219: frac[40,219] + frac[196,219] + frac[192,219] + frac[80,219]
   + frac[193,219] >= 0.9
 cov_1180_168: frac[198,168] + frac[192,168] + frac[163,168]
   + frac[196,168] + frac[10,168] + frac[197,168] + frac[195,168] >= 0.7
 cov_1181_31: frac[166,31] + frac[195,31] + frac[150,31] >= 0.7
 cov_1182_144: frac[195,144] + frac[192,144] + frac[196,144]
   + frac[198,144] + frac[177,144] + frac[5,144] + frac[197,144] >= 0.7
 cov_1183_229: frac[196,229] + frac[20,229] + frac[194,229] + frac[198,229]
   + frac[192,229] + frac[197,229] + frac[141,229] >= 0.9
 cov_1184_245: frac[193,245] + frac[90,245] + frac[197,245] + frac[194,245]
   + frac[196,245] + frac[198,245] + frac[140,245] >= 0.8
 cov_1185_99: frac[195,99] + frac[162,99] + frac[14,99] + frac[196,99]
   + frac[192,99] + frac[197,99] + frac[198,99] >= 0.8
 cov_1186_187: frac[197,187] + frac[113,187] + frac[194,187]
   + frac[182,187] + frac[195,187] >= 0.8
 cov_1187_213: frac[192,213] + frac[28,213] + frac[44,213] >= 0.7
 cov_1188_78: frac[192,78] + frac[196,78] + frac[194,78] + frac[111,78]
   + frac[198,78] + frac[27,78] + frac[197,78] >= 0.8
 cov_1189_134: frac[119,134] + frac[194,134] + frac[118,134] >= 0.8
 cov_1190_108: frac[31,108] + frac[155,108] + frac[198,108] + frac[192,108]
   + frac[196,108] + frac[197,108] + frac[195,108] >= 0.8
 cov_1191_131: frac[195,131] + frac[177,131] + frac[194,131]
   + frac[197,131] + frac[113,131] >= 0.9
 cov_1192_161: frac[106,161] + frac[137,161] + frac[194,161] >= 0.9
 cov_1193_172: frac[60,172] + frac[193,172] + frac[170,172] + frac[197,172]
   + frac[195,172] + frac[198,172] + frac[196,172] >= 1
 cov_1194_65: frac[197,65] + frac[22,65] + frac[179,65] + frac[195,65]
   + frac[198,65] + frac[196,65] + frac[192,65] >= 0.8
 cov_1195_108: frac[194,108] + frac[198,108] + frac[77,108] + frac[193,108]
   + frac[196,108] + frac[197,108] + frac[135,108] >= 0.9
 cov_1196_78: frac[91,78] + frac[192,78] + frac[196,78] + frac[33,78]
   + frac[193,78] >= 0.9
 cov_1197_161: frac[152,161] + frac[177,161] + frac[195,161] >= 0.9
 cov_1198_249: frac[103,249] + frac[197,249] + frac[198,249]
   + frac[196,249] + frac[16,249] + frac[192,249] + frac[194,249] >= 0.7
 cov_1199_229: frac[196,229] + frac[116,229] + frac[194,229]
   + frac[198,229] + frac[192,229] + frac[18,229] + frac[197,229] >= 0.9
 cov_1200_77: frac[197,77] + frac[198,77] + frac[195,77] + frac[192,77]
   + frac[196,77] + frac[148,77] + frac[19,77] >= 0.7
 cov_1201_6: frac[171,6] + frac[196,6] + frac[193,6] + frac[198,6]
   + frac[68,6] + frac[195,6] + frac[197,6] >= 0.8
 cov_1202_132: frac[115,132] + frac[194,132] + frac[138,132] >= 1
 cov_1203_76: frac[192,76] + frac[196,76] + frac[188,76] + frac[197,76]
   + frac[195,76] + frac[0,76] + frac[198,76] >= 0.8
 cov_1204_233: frac[136,233] + frac[30,233] + frac[197,233] + frac[198,233]
   + frac[192,233] + frac[194,233] + frac[196,233] >= 0.9
 cov_1205_184: frac[196,184] + frac[74,184] + frac[197,184] + frac[154,184]
   + frac[195,184] + frac[198,184] + frac[193,184] >= 0.8
 cov_1206_132: frac[195,132] + frac[196,132] + frac[198,132] + frac[37,132]
   + frac[192,132] + frac[172,132] + frac[197,132] >= 0.7
 cov_1207_240: frac[197,240] + frac[195,240] + frac[191,240]
   + frac[194,240] + frac[132,240] >= 0.7
 cov_1208_233: frac[133,233] + frac[197,233] + frac[198,233]
   + frac[192,233] + frac[194,233] + frac[36,233] + frac[196,233] >= 0.7
 cov_1209_142: frac[44,142] + frac[192,142] + frac[31,142] >= 1
 cov_1210_86: frac[196,86] + frac[194,86] + frac[75,86] + frac[193,86]
   + frac[197,86] + frac[198,86] + frac[103,86] >= 0.8
 cov_1211_250: frac[197,250] + frac[138,250] + frac[194,250]
   + frac[195,250] + frac[166,250] >= 0.8
 cov_1212_59: frac[110,59] + frac[196,59] + frac[198,59] + frac[192,59]
   + frac[197,59] + frac[194,59] + frac[15,59] >= 0.7
 cov_1213_231: frac[120,231] + frac[35,231] + frac[196,231] + frac[197,231]
   + frac[192,231] + frac[194,231] + frac[198,231] >= 1
 cov_1214_110: frac[194,110] + frac[112,110] + frac[198,110]
   + frac[193,110] + frac[197,110] + frac[82,110] + frac[196,110] >= 1
 cov_1215_98: frac[14,98] + frac[196,98] + frac[192,98] + frac[193,98]
   + frac[71,98] >= 0.8
 cov_1216_136: frac[197,136] + frac[6,136] + frac[195,136] + frac[166,136]
   + frac[198,136] + frac[192,136] + frac[196,136] >= 0.7
 cov_1217_193: frac[195,193] + frac[196,193] + frac[198,193]
   + frac[192,193] + frac[23,193] + frac[186,193] + frac[197,193] >= 1
 cov_1218_157: frac[198,157] + frac[112,157] + frac[196,157] + frac[18,157]
   + frac[197,157] + frac[194,157] + frac[192,157] >= 0.7
 cov_1219_173: frac[193,173] + frac[197,173] + frac[83,173] + frac[194,173]
   + frac[198,173] + frac[119,173] + frac[196,173] >= 0.7
 cov_1220_30: frac[132,30] + frac[99,30] + frac[194,30] >= 1
 cov_1221_142: frac[158,142] + frac[37,142] + frac[195,142] + frac[197,142]
   + frac[192,142] + frac[196,142] + frac[198,142] >= 0.8
 cov_1222_126: frac[59,126] + frac[194,126] + frac[103,126] + frac[196,126]
   + frac[193,126] + frac[198,126] + frac[197,126] >= 0.9
 cov_1223_90: frac[192,90] + frac[63,90] + frac[193,90] + frac[41,90]
   + frac[196,90] >= 0.7
 cov_1224_227: frac[197,227] + frac[123,227] + frac[194,227]
   + frac[195,227] + frac[173,227] >= 0.8
 cov_1225_95: frac[193,95] + frac[195,95] + frac[196,95] + frac[197,95]
   + frac[92,95] + frac[198,95] + frac[185,95] >= 0.7
 cov_1226_87: frac[193,87] + frac[87,87] + frac[61,87] >= 0.7
 cov_1227_107: frac[82,107] + frac[193,107] + frac[73,107] >= 0.8
 cov_1228_75: frac[197,75] + frac[194,75] + frac[126,75] + frac[173,75]
   + frac[195,75] >= 0.7
 cov_1229_36: frac[196,36] + frac[194,36] + frac[89,36] + frac[198,36]
   + frac[111,36] + frac[197,36] + frac[193,36] >= 0.9
 cov_1230_107: frac[64,107] + frac[192,107] + frac[193,107] + frac[196,107]
   + frac[4,107] >= 1
 cov_1231_6: frac[26,6] + frac[196,6] + frac[192,6] + frac[194,6]
   + frac[198,6] + frac[127,6] + frac[197,6] >= 0.8
 cov_1232_14: frac[193,14] + frac[197,14] + frac[196,14] + frac[49,14]
   + frac[194,14] + frac[198,14] + frac[120,14] >= 1
 cov_1233_101: frac[124,101] + frac[108,101] + frac[194,101] >= 0.8
 cov_1234_116: frac[0,116] + frac[40,116] + frac[192,116] >= 0.7
 cov_1235_5: frac[197,5] + frac[195,5] + frac[161,5] + frac[96,5]
   + frac[194,5] >= 0.7
 cov_1236_226: frac[49,226] + frac[197,226] + frac[198,226] + frac[193,226]
   + frac[196,226] + frac[146,226] + frac[195,226] >= 0.9
 cov_1237_129: frac[192,129] + frac[65,129] + frac[193,129] + frac[23,129]
   + frac[196,129] >= 0.9
 cov_1238_38: frac[180,38] + frac[195,38] + frac[197,38] + frac[196,38]
   + frac[193,38] + frac[80,38] + frac[198,38] >= 1
 cov_1239_254: frac[193,254] + frac[81,254] + frac[53,254] >= 1
 cov_1240_180: frac[195,180] + frac[192,180] + frac[196,180]
   + frac[198,180] + frac[12,180] + frac[172,180] + frac[197,180] >= 0.7
 cov_1241_167: frac[36,167] + frac[196,167] + frac[59,167] + frac[192,167]
   + frac[193,167] >= 0.9
 cov_1242_133: frac[196,133] + frac[194,133] + frac[197,133]
   + frac[198,133] + frac[55,133] + frac[193,133] + frac[140,133] >= 0.9
 cov_1243_0: frac[194,0] + frac[124,0] + frac[159,0] + frac[197,0]
   + frac[195,0] >= 0.9
 cov_1244_251: frac[164,251] + frac[134,251] + frac[194,251]
   + frac[195,251] + frac[197,251] >= 0.8
 cov_1245_229: frac[196,229] + frac[194,229] + frac[198,229]
   + frac[192,229] + frac[28,229] + frac[197,229] + frac[134,229] >= 0.7
 cov_1246_162: frac[141,162] + frac[194,162] + frac[195,162]
   + frac[157,162] + frac[197,162] >= 0.7
 cov_1247_148: frac[65,148] + frac[198,148] + frac[193,148] + frac[197,148]
   + frac[146,148] + frac[195,148] + frac[196,148] >= 0.8
 cov_1248_30: frac[193,30] + frac[51,30] + frac[78,30] >= 1
 cov_1249_55: frac[171,55] + frac[58,55] + frac[196,55] + frac[195,55]
   + frac[193,55] + frac[198,55] + frac[197,55] >= 0.8
 cov_1250_170: frac[167,170] + frac[195,170] + frac[187,170] >= 0.8
 cov_1251_119: frac[130,119] + frac[195,119] + frac[194,119]
   + frac[197,119] + frac[188,119] >= 1
 cov_1252_195: frac[151,195] + frac[195,195] + frac[196,195]
   + frac[197,195] + frac[50,195] + frac[193,195] + frac[198,195] >= 0.9
 cov_1253_247: frac[195,247] + frac[186,247] + frac[154,247] >= 0.7
 cov_1254_67: frac[26,67] + frac[196,67] + frac[136,67] + frac[192,67]
   + frac[197,67] + frac[198,67] + frac[194,67] >= 0.8
 cov_1255_145: frac[184,145] + frac[193,145] + frac[196,145]
   + frac[198,145] + frac[75,145] + frac[197,145] + frac[195,145] >= 0.8
 cov_1256_101: frac[197,101] + frac[92,101] + frac[115,101] + frac[193,101]
   + frac[196,101] + frac[194,101] + frac[198,101] >= 0.7
 cov_1257_227: frac[197,227] + frac[198,227] + frac[194,227]
   + frac[108,227] + frac[196,227] + frac[14,227] + frac[192,227] >= 0.7
 cov_1258_90: frac[197,90] + frac[129,90] + frac[194,90] + frac[181,90]
   + frac[195,90] >= 0.9
 cov_1259_37: frac[5,37] + frac[14,37] + frac[192,37] >= 0.9
 cov_1260_155: frac[172,155] + frac[190,155] + frac[195,155] >= 1
 cov_1261_2: frac[154,2] + frac[195,2] + frac[178,2] >= 1
 cov_1262_74: frac[196,74] + frac[9,74] + frac[198,74] + frac[194,74]
   + frac[197,74] + frac[192,74] + frac[130,74] >= 0.7
 cov_1263_74: frac[196,74] + frac[198,74] + frac[195,74] + frac[169,74]
   + frac[39,74] + frac[197,74] + frac[192,74] >= 0.9
 cov_1264_244: frac[194,244] + frac[151,244] + frac[197,244]
   + frac[124,244] + frac[195,244] >= 1
 cov_1265_215: frac[198,215] + frac[63,215] + frac[107,215] + frac[196,215]
   + frac[193,215] + frac[197,215] + frac[194,215] >= 0.7
 cov_1266_75: frac[174,75] + frac[197,75] + frac[194,75] + frac[139,75]
   + frac[195,75] >= 1
 cov_1267_219: frac[190,219] + frac[194,219] + frac[195,219]
   + frac[197,219] + frac[106,219] >= 0.9
 cov_1268_51: frac[195,51] + frac[193,51] + frac[198,51] + frac[197,51]
   + frac[147,51] + frac[61,51] + frac[196,51] >= 0.7
 cov_1269_73: frac[9,73] + frac[193,73] + frac[63,73] + frac[192,73]
   + frac[196,73] >= 0.8
 cov_1270_62: frac[189,62] + frac[174,62] + frac[195,62] >= 1
 cov_1271_152: frac[192,152] + frac[194,152] + frac[197,152]
   + frac[198,152] + frac[114,152] + frac[196,152] + frac[2,152] >= 0.8
 cov_1272_100: frac[196,100] + frac[50,100] + frac[192,100] + frac[193,100]
   + frac[18,100] >= 1
 cov_1273_232: frac[198,232] + frac[193,232] + frac[194,232] + frac[88,232]
   + frac[196,232] + frac[197,232] + frac[120,232] >= 0.8
 cov_1274_42: frac[192,42] + frac[197,42] + frac[196,42] + frac[198,42]
   + frac[158,42] + frac[195,42] + frac[44,42] >= 0.9
 cov_1275_138: frac[197,138] + frac[196,138] + frac[198,138] + frac[86,138]
   + frac[157,138] + frac[195,138] + frac[193,138] >= 0.8
 cov_1276_46: frac[25,46] + frac[195,46] + frac[197,46] + frac[192,46]
   + frac[163,46] + frac[196,46] + frac[198,46] >= 0.7
 cov_1277_21: frac[195,21] + frac[194,21] + frac[158,21] + frac[107,21]
   + frac[197,21] >= 0.7
 cov_1278_22: frac[198,22] + frac[100,22] + frac[193,22] + frac[197,22]
   + frac[196,22] + frac[73,22] + frac[194,22] >= 1
 cov_1279_181: frac[194,181] + frac[102,181] + frac[197,181]
   + frac[164,181] + frac[195,181] >= 0.9
 accuracy_88_147: - 5.09765625 frac[88,147] + mem[88,147] = 0
 accuracy_169_227: - 5.09765625 frac[169,227] + mem[169,227] = 0
 accuracy_149_87: - 5.09765625 frac[149,87] + mem[149,87] = 0
 accuracy_43_3: - 5.09765625 frac[43,3] + mem[43,3] = 0
 accuracy_154_2: - 5.09765625 frac[154,2] + mem[154,2] = 0
 accuracy_192_241: - 5.09765625 frac[192,241] + mem[192,241] = 0
 accuracy_193_229: - 5.09765625 frac[193,229] + mem[193,229] = 0
 accuracy_194_237: - 5.09765625 frac[194,237] + mem[194,237] = 0
 accuracy_196_86: - 5.09765625 frac[196,86] + mem[196,86] = 0
 accuracy_34_153: - 5.09765625 frac[34,153] + mem[34,153] = 0
 accuracy_179_54: - 5.09765625 frac[179,54] + mem[179,54] = 0
 accuracy_197_96: - 5.09765625 frac[197,96] + mem[197,96] = 0
 accuracy_74_253: - 5.09765625 frac[74,253] + mem[74,253] = 0
 accuracy_59_131: - 5.09765625 frac[59,131] + mem[59,131] = 0
 accuracy_44_34: - 5.09765625 frac[44,34] + mem[44,34] = 0
 accuracy_100_199: - 5.09765625 frac[100,199] + mem[100,199] = 0
 accuracy_9_0: - 5.09765625 frac[9,0] + mem[9,0] = 0
 accuracy_191_232: - 5.09765625 frac[191,232] + mem[191,232] = 0
 accuracy_193_150: - 5.09765625 frac[193,150] + mem[193,150] = 0
 accuracy_105_20: - 5.09765625 frac[105,20] + mem[105,20] = 0
 accuracy_178_194: - 5.09765625 frac[178,194] + mem[178,194] = 0
 accuracy_196_229: - 5.09765625 frac[196,229] + mem[196,229] = 0
 accuracy_197_241: - 5.09765625 frac[197,241] + mem[197,241] = 0
 accuracy_198_129: - 5.09765625 frac[198,129] + mem[198,129] = 0
 accuracy_195_150: - 5.09765625 frac[195,150] + mem[195,150] = 0
 accuracy_55_247: - 5.09765625 frac[55,247] + mem[55,247] = 0
 accuracy_185_140: - 5.09765625 frac[185,140] + mem[185,140] = 0
 accuracy_98_45: - 5.09765625 frac[98,45] + mem[98,45] = 0
 accuracy_134_55: - 5.09765625 frac[134,55] + mem[134,55] = 0
 accuracy_192_42: - 5.09765625 frac[192,42] + mem[192,42] = 0
 accuracy_119_90: - 5.09765625 frac[119,90] + mem[119,90] = 0
 accuracy_102_182: - 5.09765625 frac[102,182] + mem[102,182] = 0
 accuracy_195_39: - 5.09765625 frac[195,39] + mem[195,39] = 0
 accuracy_123_31: - 5.09765625 frac[123,31] + mem[123,31] = 0
 accuracy_30_100: - 5.09765625 frac[30,100] + mem[30,100] = 0
 accuracy_197_130: - 5.09765625 frac[197,130] + mem[197,130] = 0
 accuracy_198_16: - 5.09765625 frac[198,16] + mem[198,16] = 0
 accuracy_89_227: - 5.09765625 frac[89,227] + mem[89,227] = 0
 accuracy_103_203: - 5.09765625 frac[103,203] + mem[103,203] = 0
 accuracy_192_89: - 5.09765625 frac[192,89] + mem[192,89] = 0
 accuracy_32_77: - 5.09765625 frac[32,77] + mem[32,77] = 0
 accuracy_140_153: - 5.09765625 frac[140,153] + mem[140,153] = 0
 accuracy_21_37: - 5.09765625 frac[21,37] + mem[21,37] = 0
 accuracy_39_127: - 5.09765625 frac[39,127] + mem[39,127] = 0
 accuracy_114_60: - 5.09765625 frac[114,60] + mem[114,60] = 0
 accuracy_116_229: - 5.09765625 frac[116,229] + mem[116,229] = 0
 accuracy_192_200: - 5.09765625 frac[192,200] + mem[192,200] = 0
 accuracy_193_254: - 5.09765625 frac[193,254] + mem[193,254] = 0
 accuracy_194_228: - 5.09765625 frac[194,228] + mem[194,228] = 0
 accuracy_65_11: - 5.09765625 frac[65,11] + mem[65,11] = 0
 accuracy_196_93: - 5.09765625 frac[196,93] + mem[196,93] = 0
 accuracy_197_105: - 5.09765625 frac[197,105] + mem[197,105] = 0
 accuracy_124_101: - 5.09765625 frac[124,101] + mem[124,101] = 0
 accuracy_73_91: - 5.09765625 frac[73,91] + mem[73,91] = 0
 accuracy_58_26: - 5.09765625 frac[58,26] + mem[58,26] = 0
 accuracy_99_244: - 5.09765625 frac[99,244] + mem[99,244] = 0
 accuracy_26_67: - 5.09765625 frac[26,67] + mem[26,67] = 0
 accuracy_194_23: - 5.09765625 frac[194,23] + mem[194,23] = 0
 accuracy_195_159: - 5.09765625 frac[195,159] + mem[195,159] = 0
 accuracy_196_236: - 5.09765625 frac[196,236] + mem[196,236] = 0
 accuracy_197_250: - 5.09765625 frac[197,250] + mem[197,250] = 0
 accuracy_198_248: - 5.09765625 frac[198,248] + mem[198,248] = 0
 accuracy_17_57: - 5.09765625 frac[17,57] + mem[17,57] = 0
 accuracy_180_38: - 5.09765625 frac[180,38] + mem[180,38] = 0
 accuracy_65_226: - 5.09765625 frac[65,226] + mem[65,226] = 0
 accuracy_192_33: - 5.09765625 frac[192,33] + mem[192,33] = 0
 accuracy_84_247: - 5.09765625 frac[84,247] + mem[84,247] = 0
 accuracy_195_44: - 5.09765625 frac[195,44] + mem[195,44] = 0
 accuracy_158_160: - 5.09765625 frac[158,160] + mem[158,160] = 0
 accuracy_107_190: - 5.09765625 frac[107,190] + mem[107,190] = 0
 accuracy_197_139: - 5.09765625 frac[197,139] + mem[197,139] = 0
 accuracy_198_11: - 5.09765625 frac[198,11] + mem[198,11] = 0
 accuracy_171_55: - 5.09765625 frac[171,55] + mem[171,55] = 0
 accuracy_62_145: - 5.09765625 frac[62,145] + mem[62,145] = 0
 accuracy_64_107: - 5.09765625 frac[64,107] + mem[64,107] = 0
 accuracy_193_70: - 5.09765625 frac[193,70] + mem[193,70] = 0
 accuracy_196_53: - 5.09765625 frac[196,53] + mem[196,53] = 0
 accuracy_185_119: - 5.09765625 frac[185,119] + mem[185,119] = 0
 accuracy_3_152: - 5.09765625 frac[3,152] + mem[3,152] = 0
 accuracy_169_237: - 5.09765625 frac[169,237] + mem[169,237] = 0
 accuracy_114_55: - 5.09765625 frac[114,55] + mem[114,55] = 0
 accuracy_96_225: - 5.09765625 frac[96,225] + mem[96,225] = 0
 accuracy_192_207: - 5.09765625 frac[192,207] + mem[192,207] = 0
 accuracy_193_247: - 5.09765625 frac[193,247] + mem[193,247] = 0
 accuracy_194_255: - 5.09765625 frac[194,255] + mem[194,255] = 0
 accuracy_195_247: - 5.09765625 frac[195,247] + mem[195,247] = 0
 accuracy_103_27: - 5.09765625 frac[103,27] + mem[103,27] = 0
 accuracy_196_68: - 5.09765625 frac[196,68] + mem[196,68] = 0
 accuracy_197_82: - 5.09765625 frac[197,82] + mem[197,82] = 0
 accuracy_142_184: - 5.09765625 frac[142,184] + mem[142,184] = 0
 accuracy_120_231: - 5.09765625 frac[120,231] + mem[120,231] = 0
 accuracy_110_59: - 5.09765625 frac[110,59] + mem[110,59] = 0
 accuracy_169_126: - 5.09765625 frac[169,126] + mem[169,126] = 0
 accuracy_9_50: - 5.09765625 frac[9,50] + mem[9,50] = 0
 accuracy_172_221: - 5.09765625 frac[172,221] + mem[172,221] = 0
 accuracy_155_63: - 5.09765625 frac[155,63] + mem[155,63] = 0
 accuracy_194_110: - 5.09765625 frac[194,110] + mem[194,110] = 0
 accuracy_195_132: - 5.09765625 frac[195,132] + mem[195,132] = 0
 accuracy_196_235: - 5.09765625 frac[196,235] + mem[196,235] = 0
 accuracy_197_227: - 5.09765625 frac[197,227] + mem[197,227] = 0
 accuracy_108_117: - 5.09765625 frac[108,117] + mem[108,117] = 0
 accuracy_198_243: - 5.09765625 frac[198,243] + mem[198,243] = 0
 accuracy_36_167: - 5.09765625 frac[36,167] + mem[36,167] = 0
 accuracy_76_78: - 5.09765625 frac[76,78] + mem[76,78] = 0
 accuracy_170_7: - 5.09765625 frac[170,7] + mem[170,7] = 0
 accuracy_192_56: - 5.09765625 frac[192,56] + mem[192,56] = 0
 accuracy_156_214: - 5.09765625 frac[156,214] + mem[156,214] = 0
 accuracy_138_44: - 5.09765625 frac[138,44] + mem[138,44] = 0
 accuracy_195_21: - 5.09765625 frac[195,21] + mem[195,21] = 0
 accuracy_31_108: - 5.09765625 frac[31,108] + mem[31,108] = 0
 accuracy_103_249: - 5.09765625 frac[103,249] + mem[103,249] = 0
 accuracy_198_2: - 5.09765625 frac[198,2] + mem[198,2] = 0
 accuracy_109_18: - 5.09765625 frac[109,18] + mem[109,18] = 0
 accuracy_163_60: - 5.09765625 frac[163,60] + mem[163,60] = 0
 accuracy_113_215: - 5.09765625 frac[113,215] + mem[113,215] = 0
 accuracy_192_87: - 5.09765625 frac[192,87] + mem[192,87] = 0
 accuracy_193_95: - 5.09765625 frac[193,95] + mem[193,95] = 0
 accuracy_65_148: - 5.09765625 frac[65,148] + mem[65,148] = 0
 accuracy_51_169: - 5.09765625 frac[51,169] + mem[51,169] = 0
 accuracy_159_200: - 5.09765625 frac[159,200] + mem[159,200] = 0
 accuracy_196_60: - 5.09765625 frac[196,60] + mem[196,60] = 0
 accuracy_71_90: - 5.09765625 frac[71,90] + mem[71,90] = 0
 accuracy_141_162: - 5.09765625 frac[141,162] + mem[141,162] = 0
 accuracy_147_199: - 5.09765625 frac[147,199] + mem[147,199] = 0
 accuracy_183_214: - 5.09765625 frac[183,214] + mem[183,214] = 0
 accuracy_115_132: - 5.09765625 frac[115,132] + mem[115,132] = 0
 accuracy_118_147: - 5.09765625 frac[118,147] + mem[118,147] = 0
 accuracy_192_198: - 5.09765625 frac[192,198] + mem[192,198] = 0
 accuracy_121_224: - 5.09765625 frac[121,224] + mem[121,224] = 0
 accuracy_194_246: - 5.09765625 frac[194,246] + mem[194,246] = 0
 accuracy_123_196: - 5.09765625 frac[123,196] + mem[123,196] = 0
 accuracy_50_156: - 5.09765625 frac[50,156] + mem[50,156] = 0
 accuracy_196_67: - 5.09765625 frac[196,67] + mem[196,67] = 0
 accuracy_197_91: - 5.09765625 frac[197,91] + mem[197,91] = 0
 accuracy_70_219: - 5.09765625 frac[70,219] + mem[70,219] = 0
 accuracy_195_252: - 5.09765625 frac[195,252] + mem[195,252] = 0
 accuracy_165_98: - 5.09765625 frac[165,98] + mem[165,98] = 0
 accuracy_79_91: - 5.09765625 frac[79,91] + mem[79,91] = 0
 accuracy_189_124: - 5.09765625 frac[189,124] + mem[189,124] = 0
 accuracy_194_105: - 5.09765625 frac[194,105] + mem[194,105] = 0
 accuracy_195_141: - 5.09765625 frac[195,141] + mem[195,141] = 0
 accuracy_196_210: - 5.09765625 frac[196,210] + mem[196,210] = 0
 accuracy_197_236: - 5.09765625 frac[197,236] + mem[197,236] = 0
 accuracy_51_71: - 5.09765625 frac[51,71] + mem[51,71] = 0
 accuracy_55_194: - 5.09765625 frac[55,194] + mem[55,194] = 0
 accuracy_37_56: - 5.09765625 frac[37,56] + mem[37,56] = 0
 accuracy_170_14: - 5.09765625 frac[170,14] + mem[170,14] = 0
 accuracy_192_63: - 5.09765625 frac[192,63] + mem[192,63] = 0
 accuracy_84_5: - 5.09765625 frac[84,5] + mem[84,5] = 0
 accuracy_103_246: - 5.09765625 frac[103,246] + mem[103,246] = 0
 accuracy_195_26: - 5.09765625 frac[195,26] + mem[195,26] = 0
 accuracy_198_5: - 5.09765625 frac[198,5] + mem[198,5] = 0
 accuracy_189_147: - 5.09765625 frac[189,147] + mem[189,147] = 0
 accuracy_192_174: - 5.09765625 frac[192,174] + mem[192,174] = 0
 accuracy_48_227: - 5.09765625 frac[48,227] + mem[48,227] = 0
 accuracy_193_80: - 5.09765625 frac[193,80] + mem[193,80] = 0
 accuracy_85_98: - 5.09765625 frac[85,98] + mem[85,98] = 0
 accuracy_196_59: - 5.09765625 frac[196,59] + mem[196,59] = 0
 accuracy_124_3: - 5.09765625 frac[124,3] + mem[124,3] = 0
 accuracy_109_156: - 5.09765625 frac[109,156] + mem[109,156] = 0
 accuracy_146_166: - 5.09765625 frac[146,166] + mem[146,166] = 0
 accuracy_182_114: - 5.09765625 frac[182,114] + mem[182,114] = 0
 accuracy_74_142: - 5.09765625 frac[74,142] + mem[74,142] = 0
 accuracy_22_166: - 5.09765625 frac[22,166] + mem[22,166] = 0
 accuracy_44_119: - 5.09765625 frac[44,119] + mem[44,119] = 0
 accuracy_27_169: - 5.09765625 frac[27,169] + mem[27,169] = 0
 accuracy_192_221: - 5.09765625 frac[192,221] + mem[192,221] = 0
 accuracy_195_229: - 5.09765625 frac[195,229] + mem[195,229] = 0
 accuracy_124_114: - 5.09765625 frac[124,114] + mem[124,114] = 0
 accuracy_196_74: - 5.09765625 frac[196,74] + mem[196,74] = 0
 accuracy_197_68: - 5.09765625 frac[197,68] + mem[197,68] = 0
 accuracy_58_55: - 5.09765625 frac[58,55] + mem[58,55] = 0
 accuracy_80_232: - 5.09765625 frac[80,232] + mem[80,232] = 0
 accuracy_81_254: - 5.09765625 frac[81,254] + mem[81,254] = 0
 accuracy_156_181: - 5.09765625 frac[156,181] + mem[156,181] = 0
 accuracy_194_96: - 5.09765625 frac[194,96] + mem[194,96] = 0
 accuracy_103_158: - 5.09765625 frac[103,158] + mem[103,158] = 0
 accuracy_195_114: - 5.09765625 frac[195,114] + mem[195,114] = 0
 accuracy_196_217: - 5.09765625 frac[196,217] + mem[196,217] = 0
 accuracy_197_213: - 5.09765625 frac[197,213] + mem[197,213] = 0
 accuracy_112_110: - 5.09765625 frac[112,110] + mem[112,110] = 0
 accuracy_40_219: - 5.09765625 frac[40,219] + mem[40,219] = 0
 accuracy_41_166: - 5.09765625 frac[41,166] + mem[41,166] = 0
 accuracy_192_54: - 5.09765625 frac[192,54] + mem[192,54] = 0
 accuracy_101_68: - 5.09765625 frac[101,68] + mem[101,68] = 0
 accuracy_195_3: - 5.09765625 frac[195,3] + mem[195,3] = 0
 accuracy_159_41: - 5.09765625 frac[159,41] + mem[159,41] = 0
 accuracy_198_124: - 5.09765625 frac[198,124] + mem[198,124] = 0
 accuracy_127_174: - 5.09765625 frac[127,174] + mem[127,174] = 0
 accuracy_171_26: - 5.09765625 frac[171,26] + mem[171,26] = 0
 accuracy_193_41: - 5.09765625 frac[193,41] + mem[193,41] = 0
 accuracy_157_67: - 5.09765625 frac[157,67] + mem[157,67] = 0
 accuracy_196_34: - 5.09765625 frac[196,34] + mem[196,34] = 0
 accuracy_14_41: - 5.09765625 frac[14,41] + mem[14,41] = 0
 accuracy_143_144: - 5.09765625 frac[143,144] + mem[143,144] = 0
 accuracy_148_205: - 5.09765625 frac[148,205] + mem[148,205] = 0
 accuracy_114_32: - 5.09765625 frac[114,32] + mem[114,32] = 0
 accuracy_23_208: - 5.09765625 frac[23,208] + mem[23,208] = 0
 accuracy_190_149: - 5.09765625 frac[190,149] + mem[190,149] = 0
 accuracy_193_218: - 5.09765625 frac[193,218] + mem[193,218] = 0
 accuracy_104_86: - 5.09765625 frac[104,86] + mem[104,86] = 0
 accuracy_196_177: - 5.09765625 frac[196,177] + mem[196,177] = 0
 accuracy_197_77: - 5.09765625 frac[197,77] + mem[197,77] = 0
 accuracy_75_154: - 5.09765625 frac[75,154] + mem[75,154] = 0
 accuracy_152_55: - 5.09765625 frac[152,55] + mem[152,55] = 0
 accuracy_64_169: - 5.09765625 frac[64,169] + mem[64,169] = 0
 accuracy_119_134: - 5.09765625 frac[119,134] + mem[119,134] = 0
 accuracy_194_123: - 5.09765625 frac[194,123] + mem[194,123] = 0
 accuracy_174_215: - 5.09765625 frac[174,215] + mem[174,215] = 0
 accuracy_195_123: - 5.09765625 frac[195,123] + mem[195,123] = 0
 accuracy_196_192: - 5.09765625 frac[196,192] + mem[196,192] = 0
 accuracy_197_222: - 5.09765625 frac[197,222] + mem[197,222] = 0
 accuracy_198_228: - 5.09765625 frac[198,228] + mem[198,228] = 0
 accuracy_76_123: - 5.09765625 frac[76,123] + mem[76,123] = 0
 accuracy_24_113: - 5.09765625 frac[24,113] + mem[24,113] = 0
 accuracy_192_13: - 5.09765625 frac[192,13] + mem[192,13] = 0
 accuracy_195_8: - 5.09765625 frac[195,8] + mem[195,8] = 0
 accuracy_107_154: - 5.09765625 frac[107,154] + mem[107,154] = 0
 accuracy_90_254: - 5.09765625 frac[90,254] + mem[90,254] = 0
 accuracy_198_119: - 5.09765625 frac[198,119] + mem[198,119] = 0
 accuracy_18_60: - 5.09765625 frac[18,60] + mem[18,60] = 0
 accuracy_20_229: - 5.09765625 frac[20,229] + mem[20,229] = 0
 accuracy_149_12: - 5.09765625 frac[149,12] + mem[149,12] = 0
 accuracy_192_188: - 5.09765625 frac[192,188] + mem[192,188] = 0
 accuracy_193_34: - 5.09765625 frac[193,34] + mem[193,34] = 0
 accuracy_196_41: - 5.09765625 frac[196,41] + mem[196,41] = 0
 accuracy_92_202: - 5.09765625 frac[92,202] + mem[92,202] = 0
 accuracy_39_24: - 5.09765625 frac[39,24] + mem[39,24] = 0
 accuracy_116_192: - 5.09765625 frac[116,192] + mem[116,192] = 0
 accuracy_8_201: - 5.09765625 frac[8,201] + mem[8,201] = 0
 accuracy_101_148: - 5.09765625 frac[101,148] + mem[101,148] = 0
 accuracy_193_211: - 5.09765625 frac[193,211] + mem[193,211] = 0
 accuracy_174_111: - 5.09765625 frac[174,111] + mem[174,111] = 0
 accuracy_194_195: - 5.09765625 frac[194,195] + mem[194,195] = 0
 accuracy_195_211: - 5.09765625 frac[195,211] + mem[195,211] = 0
 accuracy_196_184: - 5.09765625 frac[196,184] + mem[196,184] = 0
 accuracy_197_54: - 5.09765625 frac[197,54] + mem[197,54] = 0
 accuracy_177_185: - 5.09765625 frac[177,185] + mem[177,185] = 0
 accuracy_142_148: - 5.09765625 frac[142,148] + mem[142,148] = 0
 accuracy_129_225: - 5.09765625 frac[129,225] + mem[129,225] = 0
 accuracy_150_212: - 5.09765625 frac[150,212] + mem[150,212] = 0
 accuracy_133_100: - 5.09765625 frac[133,100] + mem[133,100] = 0
 accuracy_154_155: - 5.09765625 frac[154,155] + mem[154,155] = 0
 accuracy_194_114: - 5.09765625 frac[194,114] + mem[194,114] = 0
 accuracy_195_96: - 5.09765625 frac[195,96] + mem[195,96] = 0
 accuracy_178_152: - 5.09765625 frac[178,152] + mem[178,152] = 0
 accuracy_196_207: - 5.09765625 frac[196,207] + mem[196,207] = 0
 accuracy_197_199: - 5.09765625 frac[197,199] + mem[197,199] = 0
 accuracy_198_223: - 5.09765625 frac[198,223] + mem[198,223] = 0
 accuracy_15_64: - 5.09765625 frac[15,64] + mem[15,64] = 0
 accuracy_106_240: - 5.09765625 frac[106,240] + mem[106,240] = 0
 accuracy_60_148: - 5.09765625 frac[60,148] + mem[60,148] = 0
 accuracy_81_121: - 5.09765625 frac[81,121] + mem[81,121] = 0
 accuracy_133_245: - 5.09765625 frac[133,245] + mem[133,245] = 0
 accuracy_192_4: - 5.09765625 frac[192,4] + mem[192,4] = 0
 accuracy_156_58: - 5.09765625 frac[156,58] + mem[156,58] = 0
 accuracy_198_110: - 5.09765625 frac[198,110] + mem[198,110] = 0
 accuracy_127_92: - 5.09765625 frac[127,92] + mem[127,92] = 0
 accuracy_164_170: - 5.09765625 frac[164,170] + mem[164,170] = 0
 accuracy_182_212: - 5.09765625 frac[182,212] + mem[182,212] = 0
 accuracy_0_225: - 5.09765625 frac[0,225] + mem[0,225] = 0
 accuracy_189_190: - 5.09765625 frac[189,190] + mem[189,190] = 0
 accuracy_26_151: - 5.09765625 frac[26,151] + mem[26,151] = 0
 accuracy_193_59: - 5.09765625 frac[193,59] + mem[193,59] = 0
 accuracy_194_171: - 5.09765625 frac[194,171] + mem[194,171] = 0
 accuracy_196_16: - 5.09765625 frac[196,16] + mem[196,16] = 0
 accuracy_165_181: - 5.09765625 frac[165,181] + mem[165,181] = 0
 accuracy_167_220: - 5.09765625 frac[167,220] + mem[167,220] = 0
 accuracy_152_150: - 5.09765625 frac[152,150] + mem[152,150] = 0
 accuracy_26_6: - 5.09765625 frac[26,6] + mem[26,6] = 0
 accuracy_118_255: - 5.09765625 frac[118,255] + mem[118,255] = 0
 accuracy_10_220: - 5.09765625 frac[10,220] + mem[10,220] = 0
 accuracy_193_212: - 5.09765625 frac[193,212] + mem[193,212] = 0
 accuracy_7_104: - 5.09765625 frac[7,104] + mem[7,104] = 0
 accuracy_195_216: - 5.09765625 frac[195,216] + mem[195,216] = 0
 accuracy_196_167: - 5.09765625 frac[196,167] + mem[196,167] = 0
 accuracy_112_4: - 5.09765625 frac[112,4] + mem[112,4] = 0
 accuracy_64_167: - 5.09765625 frac[64,167] + mem[64,167] = 0
 accuracy_119_148: - 5.09765625 frac[119,148] + mem[119,148] = 0
 accuracy_194_77: - 5.09765625 frac[194,77] + mem[194,77] = 0
 accuracy_105_235: - 5.09765625 frac[105,235] + mem[105,235] = 0
 accuracy_178_147: - 5.09765625 frac[178,147] + mem[178,147] = 0
 accuracy_197_192: - 5.09765625 frac[197,192] + mem[197,192] = 0
 accuracy_198_214: - 5.09765625 frac[198,214] + mem[198,214] = 0
 accuracy_184_203: - 5.09765625 frac[184,203] + mem[184,203] = 0
 accuracy_40_176: - 5.09765625 frac[40,176] + mem[40,176] = 0
 accuracy_133_254: - 5.09765625 frac[133,254] + mem[133,254] = 0
 accuracy_45_12: - 5.09765625 frac[45,12] + mem[45,12] = 0
 accuracy_192_27: - 5.09765625 frac[192,27] + mem[192,27] = 0
 accuracy_102_71: - 5.09765625 frac[102,71] + mem[102,71] = 0
 accuracy_66_179: - 5.09765625 frac[66,179] + mem[66,179] = 0
 accuracy_173_87: - 5.09765625 frac[173,87] + mem[173,87] = 0
 accuracy_158_142: - 5.09765625 frac[158,142] + mem[158,142] = 0
 accuracy_113_244: - 5.09765625 frac[113,244] + mem[113,244] = 0
 accuracy_132_58: - 5.09765625 frac[132,58] + mem[132,58] = 0
 accuracy_192_138: - 5.09765625 frac[192,138] + mem[192,138] = 0
 accuracy_193_60: - 5.09765625 frac[193,60] + mem[193,60] = 0
 accuracy_194_162: - 5.09765625 frac[194,162] + mem[194,162] = 0
 accuracy_157_46: - 5.09765625 frac[157,46] + mem[157,46] = 0
 accuracy_68_194: - 5.09765625 frac[68,194] + mem[68,194] = 0
 accuracy_196_31: - 5.09765625 frac[196,31] + mem[196,31] = 0
 accuracy_51_138: - 5.09765625 frac[51,138] + mem[51,138] = 0
 accuracy_70_175: - 5.09765625 frac[70,175] + mem[70,175] = 0
 accuracy_2_135: - 5.09765625 frac[2,135] + mem[2,135] = 0
 accuracy_99_182: - 5.09765625 frac[99,182] + mem[99,182] = 0
 accuracy_100_8: - 5.09765625 frac[100,8] + mem[100,8] = 0
 accuracy_171_158: - 5.09765625 frac[171,158] + mem[171,158] = 0
 accuracy_193_173: - 5.09765625 frac[193,173] + mem[193,173] = 0
 accuracy_194_213: - 5.09765625 frac[194,213] + mem[194,213] = 0
 accuracy_176_35: - 5.09765625 frac[176,35] + mem[176,35] = 0
 accuracy_195_193: - 5.09765625 frac[195,193] + mem[195,193] = 0
 accuracy_196_174: - 5.09765625 frac[196,174] + mem[196,174] = 0
 accuracy_197_56: - 5.09765625 frac[197,56] + mem[197,56] = 0
 accuracy_198_190: - 5.09765625 frac[198,190] + mem[198,190] = 0
 accuracy_175_117: - 5.09765625 frac[175,117] + mem[175,117] = 0
 accuracy_54_4: - 5.09765625 frac[54,4] + mem[54,4] = 0
 accuracy_16_119: - 5.09765625 frac[16,119] + mem[16,119] = 0
 accuracy_147_89: - 5.09765625 frac[147,89] + mem[147,89] = 0
 accuracy_59_203: - 5.09765625 frac[59,203] + mem[59,203] = 0
 accuracy_167_170: - 5.09765625 frac[167,170] + mem[167,170] = 0
 accuracy_190_25: - 5.09765625 frac[190,25] + mem[190,25] = 0
 accuracy_7_246: - 5.09765625 frac[7,246] + mem[7,246] = 0
 accuracy_119_109: - 5.09765625 frac[119,109] + mem[119,109] = 0
 accuracy_194_68: - 5.09765625 frac[194,68] + mem[194,68] = 0
 accuracy_195_110: - 5.09765625 frac[195,110] + mem[195,110] = 0
 accuracy_197_201: - 5.09765625 frac[197,201] + mem[197,201] = 0
 accuracy_198_201: - 5.09765625 frac[198,201] + mem[198,201] = 0
 accuracy_163_121: - 5.09765625 frac[163,121] + mem[163,121] = 0
 accuracy_81_107: - 5.09765625 frac[81,107] + mem[81,107] = 0
 accuracy_61_223: - 5.09765625 frac[61,223] + mem[61,223] = 0
 accuracy_192_18: - 5.09765625 frac[192,18] + mem[192,18] = 0
 accuracy_120_58: - 5.09765625 frac[120,58] + mem[120,58] = 0
 accuracy_102_126: - 5.09765625 frac[102,126] + mem[102,126] = 0
 accuracy_157_182: - 5.09765625 frac[157,182] + mem[157,182] = 0
 accuracy_173_80: - 5.09765625 frac[173,80] + mem[173,80] = 0
 accuracy_89_219: - 5.09765625 frac[89,219] + mem[89,219] = 0
 accuracy_198_88: - 5.09765625 frac[198,88] + mem[198,88] = 0
 accuracy_74_90: - 5.09765625 frac[74,90] + mem[74,90] = 0
 accuracy_43_51: - 5.09765625 frac[43,51] + mem[43,51] = 0
 accuracy_99_46: - 5.09765625 frac[99,46] + mem[99,46] = 0
 accuracy_171_6: - 5.09765625 frac[171,6] + mem[171,6] = 0
 accuracy_192_129: - 5.09765625 frac[192,129] + mem[192,129] = 0
 accuracy_193_53: - 5.09765625 frac[193,53] + mem[193,53] = 0
 accuracy_194_189: - 5.09765625 frac[194,189] + mem[194,189] = 0
 accuracy_67_185: - 5.09765625 frac[67,185] + mem[67,185] = 0
 accuracy_33_121: - 5.09765625 frac[33,121] + mem[33,121] = 0
 accuracy_196_6: - 5.09765625 frac[196,6] + mem[196,6] = 0
 accuracy_147_241: - 5.09765625 frac[147,241] + mem[147,241] = 0
 accuracy_173_167: - 5.09765625 frac[173,167] + mem[173,167] = 0
 accuracy_193_166: - 5.09765625 frac[193,166] + mem[193,166] = 0
 accuracy_194_44: - 5.09765625 frac[194,44] + mem[194,44] = 0
 accuracy_195_198: - 5.09765625 frac[195,198] + mem[195,198] = 0
 accuracy_196_149: - 5.09765625 frac[196,149] + mem[196,149] = 0
 accuracy_197_33: - 5.09765625 frac[197,33] + mem[197,33] = 0
 accuracy_198_177: - 5.09765625 frac[198,177] + mem[198,177] = 0
 accuracy_130_250: - 5.09765625 frac[130,250] + mem[130,250] = 0
 accuracy_194_95: - 5.09765625 frac[194,95] + mem[194,95] = 0
 accuracy_174_243: - 5.09765625 frac[174,243] + mem[174,243] = 0
 accuracy_195_87: - 5.09765625 frac[195,87] + mem[195,87] = 0
 accuracy_197_178: - 5.09765625 frac[197,178] + mem[197,178] = 0
 accuracy_198_192: - 5.09765625 frac[198,192] + mem[198,192] = 0
 accuracy_56_36: - 5.09765625 frac[56,36] + mem[56,36] = 0
 accuracy_39_222: - 5.09765625 frac[39,222] + mem[39,222] = 0
 accuracy_60_161: - 5.09765625 frac[60,161] + mem[60,161] = 0
 accuracy_62_222: - 5.09765625 frac[62,222] + mem[62,222] = 0
 accuracy_192_105: - 5.09765625 frac[192,105] + mem[192,105] = 0
 accuracy_84_47: - 5.09765625 frac[84,47] + mem[84,47] = 0
 accuracy_102_113: - 5.09765625 frac[102,113] + mem[102,113] = 0
 accuracy_108_213: - 5.09765625 frac[108,213] + mem[108,213] = 0
 accuracy_198_83: - 5.09765625 frac[198,83] + mem[198,83] = 0
 accuracy_192_152: - 5.09765625 frac[192,152] + mem[192,152] = 0
 accuracy_45_143: - 5.09765625 frac[45,143] + mem[45,143] = 0
 accuracy_46_231: - 5.09765625 frac[46,231] + mem[46,231] = 0
 accuracy_193_14: - 5.09765625 frac[193,14] + mem[193,14] = 0
 accuracy_32_140: - 5.09765625 frac[32,140] + mem[32,140] = 0
 accuracy_196_13: - 5.09765625 frac[196,13] + mem[196,13] = 0
 accuracy_22_92: - 5.09765625 frac[22,92] + mem[22,92] = 0
 accuracy_170_189: - 5.09765625 frac[170,189] + mem[170,189] = 0
 accuracy_79_101: - 5.09765625 frac[79,101] + mem[79,101] = 0
 accuracy_100_62: - 5.09765625 frac[100,62] + mem[100,62] = 0
 accuracy_83_94: - 5.09765625 frac[83,94] + mem[83,94] = 0
 accuracy_194_39: - 5.09765625 frac[194,39] + mem[194,39] = 0
 accuracy_195_207: - 5.09765625 frac[195,207] + mem[195,207] = 0
 accuracy_196_156: - 5.09765625 frac[196,156] + mem[196,156] = 0
 accuracy_34_211: - 5.09765625 frac[34,211] + mem[34,211] = 0
 accuracy_198_168: - 5.09765625 frac[198,168] + mem[198,168] = 0
 accuracy_197_42: - 5.09765625 frac[197,42] + mem[197,42] = 0
 accuracy_17_105: - 5.09765625 frac[17,105] + mem[17,105] = 0
 accuracy_9_74: - 5.09765625 frac[9,74] + mem[9,74] = 0
 accuracy_194_86: - 5.09765625 frac[194,86] + mem[194,86] = 0
 accuracy_177_54: - 5.09765625 frac[177,54] + mem[177,54] = 0
 accuracy_195_92: - 5.09765625 frac[195,92] + mem[195,92] = 0
 accuracy_12_241: - 5.09765625 frac[12,241] + mem[12,241] = 0
 accuracy_197_187: - 5.09765625 frac[197,187] + mem[197,187] = 0
 accuracy_198_59: - 5.09765625 frac[198,59] + mem[198,59] = 0
 accuracy_133_233: - 5.09765625 frac[133,233] + mem[133,233] = 0
 accuracy_136_67: - 5.09765625 frac[136,67] + mem[136,67] = 0
 accuracy_192_96: - 5.09765625 frac[192,96] + mem[192,96] = 0
 accuracy_139_254: - 5.09765625 frac[139,254] + mem[139,254] = 0
 accuracy_68_168: - 5.09765625 frac[68,168] + mem[68,168] = 0
 accuracy_158_227: - 5.09765625 frac[158,227] + mem[158,227] = 0
 accuracy_87_203: - 5.09765625 frac[87,203] + mem[87,203] = 0
 accuracy_32_20: - 5.09765625 frac[32,20] + mem[32,20] = 0
 accuracy_15_213: - 5.09765625 frac[15,213] + mem[15,213] = 0
 accuracy_198_74: - 5.09765625 frac[198,74] + mem[198,74] = 0
 accuracy_141_153: - 5.09765625 frac[141,153] + mem[141,153] = 0
 accuracy_75_86: - 5.09765625 frac[75,86] + mem[75,86] = 0
 accuracy_4_90: - 5.09765625 frac[4,90] + mem[4,90] = 0
 accuracy_169_157: - 5.09765625 frac[169,157] + mem[169,157] = 0
 accuracy_171_116: - 5.09765625 frac[171,116] + mem[171,116] = 0
 accuracy_192_159: - 5.09765625 frac[192,159] + mem[192,159] = 0
 accuracy_193_7: - 5.09765625 frac[193,7] + mem[193,7] = 0
 accuracy_194_143: - 5.09765625 frac[194,143] + mem[194,143] = 0
 accuracy_121_47: - 5.09765625 frac[121,47] + mem[121,47] = 0
 accuracy_196_116: - 5.09765625 frac[196,116] + mem[196,116] = 0
 accuracy_107_12: - 5.09765625 frac[107,12] + mem[107,12] = 0
 accuracy_143_250: - 5.09765625 frac[143,250] + mem[143,250] = 0
 accuracy_97_22: - 5.09765625 frac[97,22] + mem[97,22] = 0
 accuracy_150_24: - 5.09765625 frac[150,24] + mem[150,24] = 0
 accuracy_155_79: - 5.09765625 frac[155,79] + mem[155,79] = 0
 accuracy_46_89: - 5.09765625 frac[46,89] + mem[46,89] = 0
 accuracy_47_85: - 5.09765625 frac[47,85] + mem[47,85] = 0
 accuracy_194_62: - 5.09765625 frac[194,62] + mem[194,62] = 0
 accuracy_12_25: - 5.09765625 frac[12,25] + mem[12,25] = 0
 accuracy_196_155: - 5.09765625 frac[196,155] + mem[196,155] = 0
 accuracy_197_19: - 5.09765625 frac[197,19] + mem[197,19] = 0
 accuracy_101_177: - 5.09765625 frac[101,177] + mem[101,177] = 0
 accuracy_195_180: - 5.09765625 frac[195,180] + mem[195,180] = 0
 accuracy_184_120: - 5.09765625 frac[184,120] + mem[184,120] = 0
 accuracy_60_192: - 5.09765625 frac[60,192] + mem[60,192] = 0
 accuracy_194_81: - 5.09765625 frac[194,81] + mem[194,81] = 0
 accuracy_195_69: - 5.09765625 frac[195,69] + mem[195,69] = 0
 accuracy_197_164: - 5.09765625 frac[197,164] + mem[197,164] = 0
 accuracy_198_50: - 5.09765625 frac[198,50] + mem[198,50] = 0
 accuracy_126_26: - 5.09765625 frac[126,26] + mem[126,26] = 0
 accuracy_79_128: - 5.09765625 frac[79,128] + mem[79,128] = 0
 accuracy_192_103: - 5.09765625 frac[192,103] + mem[192,103] = 0
 accuracy_193_111: - 5.09765625 frac[193,111] + mem[193,111] = 0
 accuracy_63_218: - 5.09765625 frac[63,218] + mem[63,218] = 0
 accuracy_32_43: - 5.09765625 frac[32,43] + mem[32,43] = 0
 accuracy_31_45: - 5.09765625 frac[31,45] + mem[31,45] = 0
 accuracy_141_146: - 5.09765625 frac[141,146] + mem[141,146] = 0
 accuracy_198_77: - 5.09765625 frac[198,77] + mem[198,77] = 0
 accuracy_113_144: - 5.09765625 frac[113,144] + mem[113,144] = 0
 accuracy_97_190: - 5.09765625 frac[97,190] + mem[97,190] = 0
 accuracy_132_30: - 5.09765625 frac[132,30] + mem[132,30] = 0
 accuracy_43_38: - 5.09765625 frac[43,38] + mem[43,38] = 0
 accuracy_26_242: - 5.09765625 frac[26,242] + mem[26,242] = 0
 accuracy_23_146: - 5.09765625 frac[23,146] + mem[23,146] = 0
 accuracy_192_150: - 5.09765625 frac[192,150] + mem[192,150] = 0
 accuracy_193_24: - 5.09765625 frac[193,24] + mem[193,24] = 0
 accuracy_194_134: - 5.09765625 frac[194,134] + mem[194,134] = 0
 accuracy_190_219: - 5.09765625 frac[190,219] + mem[190,219] = 0
 accuracy_196_115: - 5.09765625 frac[196,115] + mem[196,115] = 0
 accuracy_53_65: - 5.09765625 frac[53,65] + mem[53,65] = 0
 accuracy_92_188: - 5.09765625 frac[92,188] + mem[92,188] = 0
 accuracy_93_42: - 5.09765625 frac[93,42] + mem[93,42] = 0
 accuracy_21_104: - 5.09765625 frac[21,104] + mem[21,104] = 0
 accuracy_190_170: - 5.09765625 frac[190,170] + mem[190,170] = 0
 accuracy_154_94: - 5.09765625 frac[154,94] + mem[154,94] = 0
 accuracy_193_137: - 5.09765625 frac[193,137] + mem[193,137] = 0
 accuracy_194_57: - 5.09765625 frac[194,57] + mem[194,57] = 0
 accuracy_195_189: - 5.09765625 frac[195,189] + mem[195,189] = 0
 accuracy_196_130: - 5.09765625 frac[196,130] + mem[196,130] = 0
 accuracy_198_154: - 5.09765625 frac[198,154] + mem[198,154] = 0
 accuracy_73_6: - 5.09765625 frac[73,6] + mem[73,6] = 0
 accuracy_36_158: - 5.09765625 frac[36,158] + mem[36,158] = 0
 accuracy_165_35: - 5.09765625 frac[165,35] + mem[165,35] = 0
 accuracy_94_91: - 5.09765625 frac[94,91] + mem[94,91] = 0
 accuracy_93_91: - 5.09765625 frac[93,91] + mem[93,91] = 0
 accuracy_27_14: - 5.09765625 frac[27,14] + mem[27,14] = 0
 accuracy_172_155: - 5.09765625 frac[172,155] + mem[172,155] = 0
 accuracy_139_159: - 5.09765625 frac[139,159] + mem[139,159] = 0
 accuracy_175_254: - 5.09765625 frac[175,254] + mem[175,254] = 0
 accuracy_106_198: - 5.09765625 frac[106,198] + mem[106,198] = 0
 accuracy_88_16: - 5.09765625 frac[88,16] + mem[88,16] = 0
 accuracy_197_173: - 5.09765625 frac[197,173] + mem[197,173] = 0
 accuracy_198_53: - 5.09765625 frac[198,53] + mem[198,53] = 0
 accuracy_195_74: - 5.09765625 frac[195,74] + mem[195,74] = 0
 accuracy_77_18: - 5.09765625 frac[77,18] + mem[77,18] = 0
 accuracy_116_19: - 5.09765625 frac[116,19] + mem[116,19] = 0
 accuracy_133_219: - 5.09765625 frac[133,219] + mem[133,219] = 0
 accuracy_79_137: - 5.09765625 frac[79,137] + mem[79,137] = 0
 accuracy_192_126: - 5.09765625 frac[192,126] + mem[192,126] = 0
 accuracy_193_96: - 5.09765625 frac[193,96] + mem[193,96] = 0
 accuracy_153_15: - 5.09765625 frac[153,15] + mem[153,15] = 0
 accuracy_35_47: - 5.09765625 frac[35,47] + mem[35,47] = 0
 accuracy_198_68: - 5.09765625 frac[198,68] + mem[198,68] = 0
 accuracy_40_34: - 5.09765625 frac[40,34] + mem[40,34] = 0
 accuracy_96_207: - 5.09765625 frac[96,207] + mem[96,207] = 0
 accuracy_97_183: - 5.09765625 frac[97,183] + mem[97,183] = 0
 accuracy_38_72: - 5.09765625 frac[38,72] + mem[38,72] = 0
 accuracy_193_17: - 5.09765625 frac[193,17] + mem[193,17] = 0
 accuracy_194_129: - 5.09765625 frac[194,129] + mem[194,129] = 0
 accuracy_196_122: - 5.09765625 frac[196,122] + mem[196,122] = 0
 accuracy_197_116: - 5.09765625 frac[197,116] + mem[197,116] = 0
 accuracy_70_130: - 5.09765625 frac[70,130] + mem[70,130] = 0
 accuracy_93_35: - 5.09765625 frac[93,35] + mem[93,35] = 0
 accuracy_129_167: - 5.09765625 frac[129,167] + mem[129,167] = 0
 accuracy_9_236: - 5.09765625 frac[9,236] + mem[9,236] = 0
 accuracy_193_130: - 5.09765625 frac[193,130] + mem[193,130] = 0
 accuracy_29_241: - 5.09765625 frac[29,241] + mem[29,241] = 0
 accuracy_195_162: - 5.09765625 frac[195,162] + mem[195,162] = 0
 accuracy_104_174: - 5.09765625 frac[104,174] + mem[104,174] = 0
 accuracy_194_48: - 5.09765625 frac[194,48] + mem[194,48] = 0
 accuracy_196_137: - 5.09765625 frac[196,137] + mem[196,137] = 0
 accuracy_198_157: - 5.09765625 frac[198,157] + mem[198,157] = 0
 accuracy_108_27: - 5.09765625 frac[108,27] + mem[108,27] = 0
 accuracy_197_5: - 5.09765625 frac[197,5] + mem[197,5] = 0
 accuracy_94_66: - 5.09765625 frac[94,66] + mem[94,66] = 0
 accuracy_136_233: - 5.09765625 frac[136,233] + mem[136,233] = 0
 accuracy_195_51: - 5.09765625 frac[195,51] + mem[195,51] = 0
 accuracy_14_206: - 5.09765625 frac[14,206] + mem[14,206] = 0
 accuracy_197_150: - 5.09765625 frac[197,150] + mem[197,150] = 0
 accuracy_198_44: - 5.09765625 frac[198,44] + mem[198,44] = 0
 accuracy_164_148: - 5.09765625 frac[164,148] + mem[164,148] = 0
 accuracy_96_87: - 5.09765625 frac[96,87] + mem[96,87] = 0
 accuracy_59_93: - 5.09765625 frac[59,93] + mem[59,93] = 0
 accuracy_192_117: - 5.09765625 frac[192,117] + mem[192,117] = 0
 accuracy_48_202: - 5.09765625 frac[48,202] + mem[48,202] = 0
 accuracy_193_121: - 5.09765625 frac[193,121] + mem[193,121] = 0
 accuracy_32_57: - 5.09765625 frac[32,57] + mem[32,57] = 0
 accuracy_1_196: - 5.09765625 frac[1,196] + mem[1,196] = 0
 accuracy_165_243: - 5.09765625 frac[165,243] + mem[165,243] = 0
 accuracy_164_251: - 5.09765625 frac[164,251] + mem[164,251] = 0
 accuracy_132_12: - 5.09765625 frac[132,12] + mem[132,12] = 0
 accuracy_194_152: - 5.09765625 frac[194,152] + mem[194,152] = 0
 accuracy_50_254: - 5.09765625 frac[50,254] + mem[50,254] = 0
 accuracy_197_125: - 5.09765625 frac[197,125] + mem[197,125] = 0
 accuracy_144_7: - 5.09765625 frac[144,7] + mem[144,7] = 0
 accuracy_56_233: - 5.09765625 frac[56,233] + mem[56,233] = 0
 accuracy_129_184: - 5.09765625 frac[129,184] + mem[129,184] = 0
 accuracy_57_144: - 5.09765625 frac[57,144] + mem[57,144] = 0
 accuracy_95_135: - 5.09765625 frac[95,135] + mem[95,135] = 0
 accuracy_97_57: - 5.09765625 frac[97,57] + mem[97,57] = 0
 accuracy_83_122: - 5.09765625 frac[83,122] + mem[83,122] = 0
 accuracy_193_155: - 5.09765625 frac[193,155] + mem[193,155] = 0
 accuracy_194_11: - 5.09765625 frac[194,11] + mem[194,11] = 0
 accuracy_88_111: - 5.09765625 frac[88,111] + mem[88,111] = 0
 accuracy_196_240: - 5.09765625 frac[196,240] + mem[196,240] = 0
 accuracy_197_14: - 5.09765625 frac[197,14] + mem[197,14] = 0
 accuracy_198_148: - 5.09765625 frac[198,148] + mem[198,148] = 0
 accuracy_53_196: - 5.09765625 frac[53,196] + mem[53,196] = 0
 accuracy_42_87: - 5.09765625 frac[42,87] + mem[42,87] = 0
 accuracy_187_40: - 5.09765625 frac[187,40] + mem[187,40] = 0
 accuracy_156_243: - 5.09765625 frac[156,243] + mem[156,243] = 0
 accuracy_102_133: - 5.09765625 frac[102,133] + mem[102,133] = 0
 accuracy_195_56: - 5.09765625 frac[195,56] + mem[195,56] = 0
 accuracy_197_159: - 5.09765625 frac[197,159] + mem[197,159] = 0
 accuracy_198_39: - 5.09765625 frac[198,39] + mem[198,39] = 0
 accuracy_183_92: - 5.09765625 frac[183,92] + mem[183,92] = 0
 accuracy_39_251: - 5.09765625 frac[39,251] + mem[39,251] = 0
 accuracy_21_161: - 5.09765625 frac[21,161] + mem[21,161] = 0
 accuracy_45_91: - 5.09765625 frac[45,91] + mem[45,91] = 0
 accuracy_192_76: - 5.09765625 frac[192,76] + mem[192,76] = 0
 accuracy_193_114: - 5.09765625 frac[193,114] + mem[193,114] = 0
 accuracy_65_143: - 5.09765625 frac[65,143] + mem[65,143] = 0
 accuracy_160_101: - 5.09765625 frac[160,101] + mem[160,101] = 0
 accuracy_87_239: - 5.09765625 frac[87,239] + mem[87,239] = 0
 accuracy_144_239: - 5.09765625 frac[144,239] + mem[144,239] = 0
 accuracy_53_43: - 5.09765625 frac[53,43] + mem[53,43] = 0
 accuracy_146_72: - 5.09765625 frac[146,72] + mem[146,72] = 0
 accuracy_113_187: - 5.09765625 frac[113,187] + mem[113,187] = 0
 accuracy_169_249: - 5.09765625 frac[169,249] + mem[169,249] = 0
 accuracy_187_246: - 5.09765625 frac[187,246] + mem[187,246] = 0
 accuracy_99_120: - 5.09765625 frac[99,120] + mem[99,120] = 0
 accuracy_192_251: - 5.09765625 frac[192,251] + mem[192,251] = 0
 accuracy_193_227: - 5.09765625 frac[193,227] + mem[193,227] = 0
 accuracy_194_147: - 5.09765625 frac[194,147] + mem[194,147] = 0
 accuracy_48_80: - 5.09765625 frac[48,80] + mem[48,80] = 0
 accuracy_140_123: - 5.09765625 frac[140,123] + mem[140,123] = 0
 accuracy_196_104: - 5.09765625 frac[196,104] + mem[196,104] = 0
 accuracy_197_102: - 5.09765625 frac[197,102] + mem[197,102] = 0
 accuracy_53_92: - 5.09765625 frac[53,92] + mem[53,92] = 0
 accuracy_71_6: - 5.09765625 frac[71,6] + mem[71,6] = 0
 accuracy_131_181: - 5.09765625 frac[131,181] + mem[131,181] = 0
 accuracy_153_192: - 5.09765625 frac[153,192] + mem[153,192] = 0
 accuracy_193_156: - 5.09765625 frac[193,156] + mem[193,156] = 0
 accuracy_194_2: - 5.09765625 frac[194,2] + mem[194,2] = 0
 accuracy_195_144: - 5.09765625 frac[195,144] + mem[195,144] = 0
 accuracy_31_227: - 5.09765625 frac[31,227] + mem[31,227] = 0
 accuracy_196_255: - 5.09765625 frac[196,255] + mem[196,255] = 0
 accuracy_197_247: - 5.09765625 frac[197,247] + mem[197,247] = 0
 accuracy_198_143: - 5.09765625 frac[198,143] + mem[198,143] = 0
 accuracy_15_16: - 5.09765625 frac[15,16] + mem[15,16] = 0
 accuracy_6_104: - 5.09765625 frac[6,104] + mem[6,104] = 0
 accuracy_9_111: - 5.09765625 frac[9,111] + mem[9,111] = 0
 accuracy_10_103: - 5.09765625 frac[10,103] + mem[10,103] = 0
 accuracy_195_33: - 5.09765625 frac[195,33] + mem[195,33] = 0
 accuracy_14_216: - 5.09765625 frac[14,216] + mem[14,216] = 0
 accuracy_197_152: - 5.09765625 frac[197,152] + mem[197,152] = 0
 accuracy_198_30: - 5.09765625 frac[198,30] + mem[198,30] = 0
 accuracy_181_254: - 5.09765625 frac[181,254] + mem[181,254] = 0
 accuracy_34_97: - 5.09765625 frac[34,97] + mem[34,97] = 0
 accuracy_113_35: - 5.09765625 frac[113,35] + mem[113,35] = 0
 accuracy_192_67: - 5.09765625 frac[192,67] + mem[192,67] = 0
 accuracy_193_75: - 5.09765625 frac[193,75] + mem[193,75] = 0
 accuracy_84_81: - 5.09765625 frac[84,81] + mem[84,81] = 0
 accuracy_163_153: - 5.09765625 frac[163,153] + mem[163,153] = 0
 accuracy_185_100: - 5.09765625 frac[185,100] + mem[185,100] = 0
 accuracy_43_2: - 5.09765625 frac[43,2] + mem[43,2] = 0
 accuracy_172_89: - 5.09765625 frac[172,89] + mem[172,89] = 0
 accuracy_192_242: - 5.09765625 frac[192,242] + mem[192,242] = 0
 accuracy_193_228: - 5.09765625 frac[193,228] + mem[193,228] = 0
 accuracy_49_226: - 5.09765625 frac[49,226] + mem[49,226] = 0
 accuracy_196_87: - 5.09765625 frac[196,87] + mem[196,87] = 0
 accuracy_197_111: - 5.09765625 frac[197,111] + mem[197,111] = 0
 accuracy_52_252: - 5.09765625 frac[52,252] + mem[52,252] = 0
 accuracy_182_6: - 5.09765625 frac[182,6] + mem[182,6] = 0
 accuracy_10_159: - 5.09765625 frac[10,159] + mem[10,159] = 0
 accuracy_195_153: - 5.09765625 frac[195,153] + mem[195,153] = 0
 accuracy_196_230: - 5.09765625 frac[196,230] + mem[196,230] = 0
 accuracy_197_240: - 5.09765625 frac[197,240] + mem[197,240] = 0
 accuracy_34_9: - 5.09765625 frac[34,9] + mem[34,9] = 0
 accuracy_145_10: - 5.09765625 frac[145,10] + mem[145,10] = 0
 accuracy_179_166: - 5.09765625 frac[179,166] + mem[179,166] = 0
 accuracy_37_12: - 5.09765625 frac[37,12] + mem[37,12] = 0
 accuracy_198_134: - 5.09765625 frac[198,134] + mem[198,134] = 0
 accuracy_192_43: - 5.09765625 frac[192,43] + mem[192,43] = 0
 accuracy_83_249: - 5.09765625 frac[83,249] + mem[83,249] = 0
 accuracy_195_38: - 5.09765625 frac[195,38] + mem[195,38] = 0
 accuracy_31_121: - 5.09765625 frac[31,121] + mem[31,121] = 0
 accuracy_197_129: - 5.09765625 frac[197,129] + mem[197,129] = 0
 accuracy_198_17: - 5.09765625 frac[198,17] + mem[198,17] = 0
 accuracy_91_73: - 5.09765625 frac[91,73] + mem[91,73] = 0
 accuracy_192_90: - 5.09765625 frac[192,90] + mem[192,90] = 0
 accuracy_28_11: - 5.09765625 frac[28,11] + mem[28,11] = 0
 accuracy_121_100: - 5.09765625 frac[121,100] + mem[121,100] = 0
 accuracy_142_7: - 5.09765625 frac[142,7] + mem[142,7] = 0
 accuracy_20_170: - 5.09765625 frac[20,170] + mem[20,170] = 0
 accuracy_166_185: - 5.09765625 frac[166,185] + mem[166,185] = 0
 accuracy_116_230: - 5.09765625 frac[116,230] + mem[116,230] = 0
 accuracy_118_134: - 5.09765625 frac[118,134] + mem[118,134] = 0
 accuracy_192_201: - 5.09765625 frac[192,201] + mem[192,201] = 0
 accuracy_193_253: - 5.09765625 frac[193,253] + mem[193,253] = 0
 accuracy_194_229: - 5.09765625 frac[194,229] + mem[194,229] = 0
 accuracy_195_241: - 5.09765625 frac[195,241] + mem[195,241] = 0
 accuracy_48_78: - 5.09765625 frac[48,78] + mem[48,78] = 0
 accuracy_196_94: - 5.09765625 frac[196,94] + mem[196,94] = 0
 accuracy_179_62: - 5.09765625 frac[179,62] + mem[179,62] = 0
 accuracy_175_69: - 5.09765625 frac[175,69] + mem[175,69] = 0
 accuracy_73_90: - 5.09765625 frac[73,90] + mem[73,90] = 0
 accuracy_197_104: - 5.09765625 frac[197,104] + mem[197,104] = 0
 accuracy_95_242: - 5.09765625 frac[95,242] + mem[95,242] = 0
 accuracy_9_8: - 5.09765625 frac[9,8] + mem[9,8] = 0
 accuracy_155_57: - 5.09765625 frac[155,57] + mem[155,57] = 0
 accuracy_194_20: - 5.09765625 frac[194,20] + mem[194,20] = 0
 accuracy_195_158: - 5.09765625 frac[195,158] + mem[195,158] = 0
 accuracy_197_249: - 5.09765625 frac[197,249] + mem[197,249] = 0
 accuracy_198_249: - 5.09765625 frac[198,249] + mem[198,249] = 0
 accuracy_61_175: - 5.09765625 frac[61,175] + mem[61,175] = 0
 accuracy_6_90: - 5.09765625 frac[6,90] + mem[6,90] = 0
 accuracy_192_34: - 5.09765625 frac[192,34] + mem[192,34] = 0
 accuracy_120_10: - 5.09765625 frac[120,10] + mem[120,10] = 0
 accuracy_195_47: - 5.09765625 frac[195,47] + mem[195,47] = 0
 accuracy_197_138: - 5.09765625 frac[197,138] + mem[197,138] = 0
 accuracy_198_8: - 5.09765625 frac[198,8] + mem[198,8] = 0
 accuracy_91_78: - 5.09765625 frac[91,78] + mem[91,78] = 0
 accuracy_146_34: - 5.09765625 frac[146,34] + mem[146,34] = 0
 accuracy_184_145: - 5.09765625 frac[184,145] + mem[184,145] = 0
 accuracy_43_227: - 5.09765625 frac[43,227] + mem[43,227] = 0
 accuracy_117_104: - 5.09765625 frac[117,104] + mem[117,104] = 0
 accuracy_99_30: - 5.09765625 frac[99,30] + mem[99,30] = 0
 accuracy_192_81: - 5.09765625 frac[192,81] + mem[192,81] = 0
 accuracy_193_69: - 5.09765625 frac[193,69] + mem[193,69] = 0
 accuracy_196_54: - 5.09765625 frac[196,54] + mem[196,54] = 0
 accuracy_53_6: - 5.09765625 frac[53,6] + mem[53,6] = 0
 accuracy_110_73: - 5.09765625 frac[110,73] + mem[110,73] = 0
 accuracy_182_97: - 5.09765625 frac[182,97] + mem[182,97] = 0
 accuracy_115_190: - 5.09765625 frac[115,190] + mem[115,190] = 0
 accuracy_192_192: - 5.09765625 frac[192,192] + mem[192,192] = 0
 accuracy_193_246: - 5.09765625 frac[193,246] + mem[193,246] = 0
 accuracy_194_252: - 5.09765625 frac[194,252] + mem[194,252] = 0
 accuracy_195_246: - 5.09765625 frac[195,246] + mem[195,246] = 0
 accuracy_105_116: - 5.09765625 frac[105,116] + mem[105,116] = 0
 accuracy_196_69: - 5.09765625 frac[196,69] + mem[196,69] = 0
 accuracy_197_81: - 5.09765625 frac[197,81] + mem[197,81] = 0
 accuracy_48_69: - 5.09765625 frac[48,69] + mem[48,69] = 0
 accuracy_144_99: - 5.09765625 frac[144,99] + mem[144,99] = 0
 accuracy_179_7: - 5.09765625 frac[179,7] + mem[179,7] = 0
 accuracy_93_0: - 5.09765625 frac[93,0] + mem[93,0] = 0
 accuracy_102_214: - 5.09765625 frac[102,214] + mem[102,214] = 0
 accuracy_194_111: - 5.09765625 frac[194,111] + mem[194,111] = 0
 accuracy_176_153: - 5.09765625 frac[176,153] + mem[176,153] = 0
 accuracy_195_135: - 5.09765625 frac[195,135] + mem[195,135] = 0
 accuracy_196_212: - 5.09765625 frac[196,212] + mem[196,212] = 0
 accuracy_52_113: - 5.09765625 frac[52,113] + mem[52,113] = 0
 accuracy_198_240: - 5.09765625 frac[198,240] + mem[198,240] = 0
 accuracy_197_226: - 5.09765625 frac[197,226] + mem[197,226] = 0
 accuracy_56_116: - 5.09765625 frac[56,116] + mem[56,116] = 0
 accuracy_42_75: - 5.09765625 frac[42,75] + mem[42,75] = 0
 accuracy_60_241: - 5.09765625 frac[60,241] + mem[60,241] = 0
 accuracy_192_57: - 5.09765625 frac[192,57] + mem[192,57] = 0
 accuracy_195_20: - 5.09765625 frac[195,20] + mem[195,20] = 0
 accuracy_72_148: - 5.09765625 frac[72,148] + mem[72,148] = 0
 accuracy_126_43: - 5.09765625 frac[126,43] + mem[126,43] = 0
 accuracy_146_29: - 5.09765625 frac[146,29] + mem[146,29] = 0
 accuracy_74_53: - 5.09765625 frac[74,53] + mem[74,53] = 0
 accuracy_59_126: - 5.09765625 frac[59,126] + mem[59,126] = 0
 accuracy_82_53: - 5.09765625 frac[82,53] + mem[82,53] = 0
 accuracy_65_107: - 5.09765625 frac[65,107] + mem[65,107] = 0
 accuracy_193_94: - 5.09765625 frac[193,94] + mem[193,94] = 0
 accuracy_192_168: - 5.09765625 frac[192,168] + mem[192,168] = 0
 accuracy_196_61: - 5.09765625 frac[196,61] + mem[196,61] = 0
 accuracy_141_161: - 5.09765625 frac[141,161] + mem[141,161] = 0
 accuracy_181_122: - 5.09765625 frac[181,122] + mem[181,122] = 0
 accuracy_1_57: - 5.09765625 frac[1,57] + mem[1,57] = 0
 accuracy_165_144: - 5.09765625 frac[165,144] + mem[165,144] = 0
 accuracy_21_54: - 5.09765625 frac[21,54] + mem[21,54] = 0
 accuracy_95_83: - 5.09765625 frac[95,83] + mem[95,83] = 0
 accuracy_167_59: - 5.09765625 frac[167,59] + mem[167,59] = 0
 accuracy_171_188: - 5.09765625 frac[171,188] + mem[171,188] = 0
 accuracy_118_144: - 5.09765625 frac[118,144] + mem[118,144] = 0
 accuracy_195_255: - 5.09765625 frac[195,255] + mem[195,255] = 0
 accuracy_196_76: - 5.09765625 frac[196,76] + mem[196,76] = 0
 accuracy_197_90: - 5.09765625 frac[197,90] + mem[197,90] = 0
 accuracy_144_90: - 5.09765625 frac[144,90] + mem[144,90] = 0
 accuracy_147_55: - 5.09765625 frac[147,55] + mem[147,55] = 0
 accuracy_174_75: - 5.09765625 frac[174,75] + mem[174,75] = 0
 accuracy_154_143: - 5.09765625 frac[154,143] + mem[154,143] = 0
 accuracy_194_102: - 5.09765625 frac[194,102] + mem[194,102] = 0
 accuracy_195_140: - 5.09765625 frac[195,140] + mem[195,140] = 0
 accuracy_196_211: - 5.09765625 frac[196,211] + mem[196,211] = 0
 accuracy_197_235: - 5.09765625 frac[197,235] + mem[197,235] = 0
 accuracy_198_235: - 5.09765625 frac[198,235] + mem[198,235] = 0
 accuracy_41_188: - 5.09765625 frac[41,188] + mem[41,188] = 0
 accuracy_98_55: - 5.09765625 frac[98,55] + mem[98,55] = 0
 accuracy_192_48: - 5.09765625 frac[192,48] + mem[192,48] = 0
 accuracy_10_27: - 5.09765625 frac[10,27] + mem[10,27] = 0
 accuracy_102_88: - 5.09765625 frac[102,88] + mem[102,88] = 0
 accuracy_195_29: - 5.09765625 frac[195,29] + mem[195,29] = 0
 accuracy_160_41: - 5.09765625 frac[160,41] + mem[160,41] = 0
 accuracy_198_122: - 5.09765625 frac[198,122] + mem[198,122] = 0
 accuracy_26_187: - 5.09765625 frac[26,187] + mem[26,187] = 0
 accuracy_193_87: - 5.09765625 frac[193,87] + mem[193,87] = 0
 accuracy_196_36: - 5.09765625 frac[196,36] + mem[196,36] = 0
 accuracy_192_222: - 5.09765625 frac[192,222] + mem[192,222] = 0
 accuracy_193_192: - 5.09765625 frac[193,192] + mem[193,192] = 0
 accuracy_194_206: - 5.09765625 frac[194,206] + mem[194,206] = 0
 accuracy_50_148: - 5.09765625 frac[50,148] + mem[50,148] = 0
 accuracy_196_75: - 5.09765625 frac[196,75] + mem[196,75] = 0
 accuracy_197_67: - 5.09765625 frac[197,67] + mem[197,67] = 0
 accuracy_129_150: - 5.09765625 frac[129,150] + mem[129,150] = 0
 accuracy_41_20: - 5.09765625 frac[41,20] + mem[41,20] = 0
 accuracy_95_233: - 5.09765625 frac[95,233] + mem[95,233] = 0
 accuracy_61_73: - 5.09765625 frac[61,73] + mem[61,73] = 0
 accuracy_153_165: - 5.09765625 frac[153,165] + mem[153,165] = 0
 accuracy_172_204: - 5.09765625 frac[172,204] + mem[172,204] = 0
 accuracy_100_228: - 5.09765625 frac[100,228] + mem[100,228] = 0
 accuracy_156_182: - 5.09765625 frac[156,182] + mem[156,182] = 0
 accuracy_194_97: - 5.09765625 frac[194,97] + mem[194,97] = 0
 accuracy_175_193: - 5.09765625 frac[175,193] + mem[175,193] = 0
 accuracy_88_89: - 5.09765625 frac[88,89] + mem[88,89] = 0
 accuracy_51_79: - 5.09765625 frac[51,79] + mem[51,79] = 0
 accuracy_197_212: - 5.09765625 frac[197,212] + mem[197,212] = 0
 accuracy_198_226: - 5.09765625 frac[198,226] + mem[198,226] = 0
 accuracy_109_114: - 5.09765625 frac[109,114] + mem[109,114] = 0
 accuracy_195_117: - 5.09765625 frac[195,117] + mem[195,117] = 0
 accuracy_112_111: - 5.09765625 frac[112,111] + mem[112,111] = 0
 accuracy_124_226: - 5.09765625 frac[124,226] + mem[124,226] = 0
 accuracy_196_218: - 5.09765625 frac[196,218] + mem[196,218] = 0
 accuracy_188_76: - 5.09765625 frac[188,76] + mem[188,76] = 0
 accuracy_192_55: - 5.09765625 frac[192,55] + mem[192,55] = 0
 accuracy_101_67: - 5.09765625 frac[101,67] + mem[101,67] = 0
 accuracy_135_162: - 5.09765625 frac[135,162] + mem[135,162] = 0
 accuracy_195_2: - 5.09765625 frac[195,2] + mem[195,2] = 0
 accuracy_109_3: - 5.09765625 frac[109,3] + mem[109,3] = 0
 accuracy_39_141: - 5.09765625 frac[39,141] + mem[39,141] = 0
 accuracy_21_215: - 5.09765625 frac[21,215] + mem[21,215] = 0
 accuracy_26_130: - 5.09765625 frac[26,130] + mem[26,130] = 0
 accuracy_44_216: - 5.09765625 frac[44,216] + mem[44,216] = 0
 accuracy_192_166: - 5.09765625 frac[192,166] + mem[192,166] = 0
 accuracy_28_111: - 5.09765625 frac[28,111] + mem[28,111] = 0
 accuracy_62_139: - 5.09765625 frac[62,139] + mem[62,139] = 0
 accuracy_193_40: - 5.09765625 frac[193,40] + mem[193,40] = 0
 accuracy_196_35: - 5.09765625 frac[196,35] + mem[196,35] = 0
 accuracy_14_46: - 5.09765625 frac[14,46] + mem[14,46] = 0
 accuracy_181_76: - 5.09765625 frac[181,76] + mem[181,76] = 0
 accuracy_91_194: - 5.09765625 frac[91,194] + mem[91,194] = 0
 accuracy_129_126: - 5.09765625 frac[129,126] + mem[129,126] = 0
 accuracy_25_189: - 5.09765625 frac[25,189] + mem[25,189] = 0
 accuracy_27_161: - 5.09765625 frac[27,161] + mem[27,161] = 0
 accuracy_192_213: - 5.09765625 frac[192,213] + mem[192,213] = 0
 accuracy_193_217: - 5.09765625 frac[193,217] + mem[193,217] = 0
 accuracy_194_201: - 5.09765625 frac[194,201] + mem[194,201] = 0
 accuracy_195_237: - 5.09765625 frac[195,237] + mem[195,237] = 0
 accuracy_118_226: - 5.09765625 frac[118,226] + mem[118,226] = 0
 accuracy_197_76: - 5.09765625 frac[197,76] + mem[197,76] = 0
 accuracy_59_167: - 5.09765625 frac[59,167] + mem[59,167] = 0
 accuracy_11_207: - 5.09765625 frac[11,207] + mem[11,207] = 0
 accuracy_195_122: - 5.09765625 frac[195,122] + mem[195,122] = 0
 accuracy_194_120: - 5.09765625 frac[194,120] + mem[194,120] = 0
 accuracy_196_193: - 5.09765625 frac[196,193] + mem[196,193] = 0
 accuracy_197_221: - 5.09765625 frac[197,221] + mem[197,221] = 0
 accuracy_198_229: - 5.09765625 frac[198,229] + mem[198,229] = 0
 accuracy_34_36: - 5.09765625 frac[34,36] + mem[34,36] = 0
 accuracy_25_100: - 5.09765625 frac[25,100] + mem[25,100] = 0
 accuracy_134_11: - 5.09765625 frac[134,11] + mem[134,11] = 0
 accuracy_192_14: - 5.09765625 frac[192,14] + mem[192,14] = 0
 accuracy_173_100: - 5.09765625 frac[173,100] + mem[173,100] = 0
 accuracy_138_198: - 5.09765625 frac[138,198] + mem[138,198] = 0
 accuracy_195_11: - 5.09765625 frac[195,11] + mem[195,11] = 0
 accuracy_31_90: - 5.09765625 frac[31,90] + mem[31,90] = 0
 accuracy_107_157: - 5.09765625 frac[107,157] + mem[107,157] = 0
 accuracy_103_231: - 5.09765625 frac[103,231] + mem[103,231] = 0
 accuracy_126_92: - 5.09765625 frac[126,92] + mem[126,92] = 0
 accuracy_198_116: - 5.09765625 frac[198,116] + mem[198,116] = 0
 accuracy_17_181: - 5.09765625 frac[17,181] + mem[17,181] = 0
 accuracy_38_56: - 5.09765625 frac[38,56] + mem[38,56] = 0
 accuracy_8_123: - 5.09765625 frac[8,123] + mem[8,123] = 0
 accuracy_64_72: - 5.09765625 frac[64,72] + mem[64,72] = 0
 accuracy_192_189: - 5.09765625 frac[192,189] + mem[192,189] = 0
 accuracy_193_33: - 5.09765625 frac[193,33] + mem[193,33] = 0
 accuracy_63_0: - 5.09765625 frac[63,0] + mem[63,0] = 0
 accuracy_136_48: - 5.09765625 frac[136,48] + mem[136,48] = 0
 accuracy_196_42: - 5.09765625 frac[196,42] + mem[196,42] = 0
 accuracy_163_163: - 5.09765625 frac[163,163] + mem[163,163] = 0
 accuracy_146_153: - 5.09765625 frac[146,153] + mem[146,153] = 0
 accuracy_57_95: - 5.09765625 frac[57,95] + mem[57,95] = 0
 accuracy_94_227: - 5.09765625 frac[94,227] + mem[94,227] = 0
 accuracy_167_38: - 5.09765625 frac[167,38] + mem[167,38] = 0
 accuracy_44_102: - 5.09765625 frac[44,102] + mem[44,102] = 0
 accuracy_9_220: - 5.09765625 frac[9,220] + mem[9,220] = 0
 accuracy_194_192: - 5.09765625 frac[194,192] + mem[194,192] = 0
 accuracy_195_210: - 5.09765625 frac[195,210] + mem[195,210] = 0
 accuracy_196_185: - 5.09765625 frac[196,185] + mem[196,185] = 0
 accuracy_197_53: - 5.09765625 frac[197,53] + mem[197,53] = 0
 accuracy_148_68: - 5.09765625 frac[148,68] + mem[148,68] = 0
 accuracy_153_183: - 5.09765625 frac[153,183] + mem[153,183] = 0
 accuracy_173_204: - 5.09765625 frac[173,204] + mem[173,204] = 0
 accuracy_194_115: - 5.09765625 frac[194,115] + mem[194,115] = 0
 accuracy_195_99: - 5.09765625 frac[195,99] + mem[195,99] = 0
 accuracy_105_225: - 5.09765625 frac[105,225] + mem[105,225] = 0
 accuracy_123_91: - 5.09765625 frac[123,91] + mem[123,91] = 0
 accuracy_197_198: - 5.09765625 frac[197,198] + mem[197,198] = 0
 accuracy_108_90: - 5.09765625 frac[108,90] + mem[108,90] = 0
 accuracy_198_220: - 5.09765625 frac[198,220] + mem[198,220] = 0
 accuracy_35_183: - 5.09765625 frac[35,183] + mem[35,183] = 0
 accuracy_15_67: - 5.09765625 frac[15,67] + mem[15,67] = 0
 accuracy_185_185: - 5.09765625 frac[185,185] + mem[185,185] = 0
 accuracy_56_16: - 5.09765625 frac[56,16] + mem[56,16] = 0
 accuracy_112_157: - 5.09765625 frac[112,157] + mem[112,157] = 0
 accuracy_192_5: - 5.09765625 frac[192,5] + mem[192,5] = 0
 accuracy_120_45: - 5.09765625 frac[120,45] + mem[120,45] = 0
 accuracy_140_221: - 5.09765625 frac[140,221] + mem[140,221] = 0
 accuracy_158_132: - 5.09765625 frac[158,132] + mem[158,132] = 0
 accuracy_87_38: - 5.09765625 frac[87,38] + mem[87,38] = 0
 accuracy_160_62: - 5.09765625 frac[160,62] + mem[160,62] = 0
 accuracy_198_111: - 5.09765625 frac[198,111] + mem[198,111] = 0
 accuracy_74_41: - 5.09765625 frac[74,41] + mem[74,41] = 0
 accuracy_192_180: - 5.09765625 frac[192,180] + mem[192,180] = 0
 accuracy_193_58: - 5.09765625 frac[193,58] + mem[193,58] = 0
 accuracy_194_168: - 5.09765625 frac[194,168] + mem[194,168] = 0
 accuracy_157_84: - 5.09765625 frac[157,84] + mem[157,84] = 0
 accuracy_124_41: - 5.09765625 frac[124,41] + mem[124,41] = 0
 accuracy_196_17: - 5.09765625 frac[196,17] + mem[196,17] = 0
 accuracy_72_31: - 5.09765625 frac[72,31] + mem[72,31] = 0
 accuracy_163_168: - 5.09765625 frac[163,168] + mem[163,168] = 0
 accuracy_74_184: - 5.09765625 frac[74,184] + mem[74,184] = 0
 accuracy_129_72: - 5.09765625 frac[129,72] + mem[129,72] = 0
 accuracy_148_220: - 5.09765625 frac[148,220] + mem[148,220] = 0
 accuracy_150_125: - 5.09765625 frac[150,125] + mem[150,125] = 0
 accuracy_44_93: - 5.09765625 frac[44,93] + mem[44,93] = 0
 accuracy_193_171: - 5.09765625 frac[193,171] + mem[193,171] = 0
 accuracy_194_219: - 5.09765625 frac[194,219] + mem[194,219] = 0
 accuracy_195_219: - 5.09765625 frac[195,219] + mem[195,219] = 0
 accuracy_123_227: - 5.09765625 frac[123,227] + mem[123,227] = 0
 accuracy_196_160: - 5.09765625 frac[196,160] + mem[196,160] = 0
 accuracy_197_62: - 5.09765625 frac[197,62] + mem[197,62] = 0
 accuracy_71_206: - 5.09765625 frac[71,206] + mem[71,206] = 0
 accuracy_115_16: - 5.09765625 frac[115,16] + mem[115,16] = 0
 accuracy_64_184: - 5.09765625 frac[64,184] + mem[64,184] = 0
 accuracy_102_245: - 5.09765625 frac[102,245] + mem[102,245] = 0
 accuracy_194_74: - 5.09765625 frac[194,74] + mem[194,74] = 0
 accuracy_195_104: - 5.09765625 frac[195,104] + mem[195,104] = 0
 accuracy_197_207: - 5.09765625 frac[197,207] + mem[197,207] = 0
 accuracy_198_215: - 5.09765625 frac[198,215] + mem[198,215] = 0
 accuracy_126_255: - 5.09765625 frac[126,255] + mem[126,255] = 0
 accuracy_34_54: - 5.09765625 frac[34,54] + mem[34,54] = 0
 accuracy_162_99: - 5.09765625 frac[162,99] + mem[162,99] = 0
 accuracy_131_26: - 5.09765625 frac[131,26] + mem[131,26] = 0
 accuracy_60_172: - 5.09765625 frac[60,172] + mem[60,172] = 0
 accuracy_192_28: - 5.09765625 frac[192,28] + mem[192,28] = 0
 accuracy_159_7: - 5.09765625 frac[159,7] + mem[159,7] = 0
 accuracy_89_209: - 5.09765625 frac[89,209] + mem[89,209] = 0
 accuracy_198_102: - 5.09765625 frac[198,102] + mem[198,102] = 0
 accuracy_126_78: - 5.09765625 frac[126,78] + mem[126,78] = 0
 accuracy_91_184: - 5.09765625 frac[91,184] + mem[91,184] = 0
 accuracy_164_210: - 5.09765625 frac[164,210] + mem[164,210] = 0
 accuracy_111_181: - 5.09765625 frac[111,181] + mem[111,181] = 0
 accuracy_96_45: - 5.09765625 frac[96,45] + mem[96,45] = 0
 accuracy_97_145: - 5.09765625 frac[97,145] + mem[97,145] = 0
 accuracy_65_129: - 5.09765625 frac[65,129] + mem[65,129] = 0
 accuracy_193_51: - 5.09765625 frac[193,51] + mem[193,51] = 0
 accuracy_194_163: - 5.09765625 frac[194,163] + mem[194,163] = 0
 accuracy_196_24: - 5.09765625 frac[196,24] + mem[196,24] = 0
 accuracy_58_89: - 5.09765625 frac[58,89] + mem[58,89] = 0
 accuracy_115_104: - 5.09765625 frac[115,104] + mem[115,104] = 0
 accuracy_193_172: - 5.09765625 frac[193,172] + mem[193,172] = 0
 accuracy_194_210: - 5.09765625 frac[194,210] + mem[194,210] = 0
 accuracy_195_192: - 5.09765625 frac[195,192] + mem[195,192] = 0
 accuracy_140_58: - 5.09765625 frac[140,58] + mem[140,58] = 0
 accuracy_196_175: - 5.09765625 frac[196,175] + mem[196,175] = 0
 accuracy_197_39: - 5.09765625 frac[197,39] + mem[197,39] = 0
 accuracy_198_191: - 5.09765625 frac[198,191] + mem[198,191] = 0
 accuracy_15_32: - 5.09765625 frac[15,32] + mem[15,32] = 0
 accuracy_29_19: - 5.09765625 frac[29,19] + mem[29,19] = 0
 accuracy_22_50: - 5.09765625 frac[22,50] + mem[22,50] = 0
 accuracy_172_176: - 5.09765625 frac[172,176] + mem[172,176] = 0
 accuracy_194_69: - 5.09765625 frac[194,69] + mem[194,69] = 0
 accuracy_195_81: - 5.09765625 frac[195,81] + mem[195,81] = 0
 accuracy_123_105: - 5.09765625 frac[123,105] + mem[123,105] = 0
 accuracy_125_240: - 5.09765625 frac[125,240] + mem[125,240] = 0
 accuracy_198_206: - 5.09765625 frac[198,206] + mem[198,206] = 0
 accuracy_58_187: - 5.09765625 frac[58,187] + mem[58,187] = 0
 accuracy_42_17: - 5.09765625 frac[42,17] + mem[42,17] = 0
 accuracy_192_19: - 5.09765625 frac[192,19] + mem[192,19] = 0
 accuracy_137_227: - 5.09765625 frac[137,227] + mem[137,227] = 0
 accuracy_138_243: - 5.09765625 frac[138,243] + mem[138,243] = 0
 accuracy_67_43: - 5.09765625 frac[67,43] + mem[67,43] = 0
 accuracy_72_190: - 5.09765625 frac[72,190] + mem[72,190] = 0
 accuracy_198_89: - 5.09765625 frac[198,89] + mem[198,89] = 0
 accuracy_185_52: - 5.09765625 frac[185,52] + mem[185,52] = 0
 accuracy_39_177: - 5.09765625 frac[39,177] + mem[39,177] = 0
 accuracy_116_175: - 5.09765625 frac[116,175] + mem[116,175] = 0
 accuracy_135_23: - 5.09765625 frac[135,23] + mem[135,23] = 0
 accuracy_192_130: - 5.09765625 frac[192,130] + mem[192,130] = 0
 accuracy_193_52: - 5.09765625 frac[193,52] + mem[193,52] = 0
 accuracy_194_186: - 5.09765625 frac[194,186] + mem[194,186] = 0
 accuracy_196_7: - 5.09765625 frac[196,7] + mem[196,7] = 0
 accuracy_92_208: - 5.09765625 frac[92,208] + mem[92,208] = 0
 accuracy_57_114: - 5.09765625 frac[57,114] + mem[57,114] = 0
 accuracy_129_90: - 5.09765625 frac[129,90] + mem[129,90] = 0
 accuracy_5_242: - 5.09765625 frac[5,242] + mem[5,242] = 0
 accuracy_193_165: - 5.09765625 frac[193,165] + mem[193,165] = 0
 accuracy_194_45: - 5.09765625 frac[194,45] + mem[194,45] = 0
 accuracy_195_201: - 5.09765625 frac[195,201] + mem[195,201] = 0
 accuracy_68_89: - 5.09765625 frac[68,89] + mem[68,89] = 0
 accuracy_196_150: - 5.09765625 frac[196,150] + mem[196,150] = 0
 accuracy_197_32: - 5.09765625 frac[197,32] + mem[197,32] = 0
 accuracy_198_182: - 5.09765625 frac[198,182] + mem[198,182] = 0
 accuracy_2_14: - 5.09765625 frac[2,14] + mem[2,14] = 0
 accuracy_131_251: - 5.09765625 frac[131,251] + mem[131,251] = 0
 accuracy_150_254: - 5.09765625 frac[150,254] + mem[150,254] = 0
 accuracy_194_92: - 5.09765625 frac[194,92] + mem[194,92] = 0
 accuracy_174_240: - 5.09765625 frac[174,240] + mem[174,240] = 0
 accuracy_195_86: - 5.09765625 frac[195,86] + mem[195,86] = 0
 accuracy_31_169: - 5.09765625 frac[31,169] + mem[31,169] = 0
 accuracy_197_177: - 5.09765625 frac[197,177] + mem[197,177] = 0
 accuracy_52_34: - 5.09765625 frac[52,34] + mem[52,34] = 0
 accuracy_18_142: - 5.09765625 frac[18,142] + mem[18,142] = 0
 accuracy_198_193: - 5.09765625 frac[198,193] + mem[198,193] = 0
 accuracy_37_205: - 5.09765625 frac[37,205] + mem[37,205] = 0
 accuracy_159_100: - 5.09765625 frac[159,100] + mem[159,100] = 0
 accuracy_168_116: - 5.09765625 frac[168,116] + mem[168,116] = 0
 accuracy_42_24: - 5.09765625 frac[42,24] + mem[42,24] = 0
 accuracy_60_162: - 5.09765625 frac[60,162] + mem[60,162] = 0
 accuracy_63_215: - 5.09765625 frac[63,215] + mem[63,215] = 0
 accuracy_138_250: - 5.09765625 frac[138,250] + mem[138,250] = 0
 accuracy_14_98: - 5.09765625 frac[14,98] + mem[14,98] = 0
 accuracy_198_80: - 5.09765625 frac[198,80] + mem[198,80] = 0
 accuracy_186_181: - 5.09765625 frac[186,181] + mem[186,181] = 0
 accuracy_60_17: - 5.09765625 frac[60,17] + mem[60,17] = 0
 accuracy_117_32: - 5.09765625 frac[117,32] + mem[117,32] = 0
 accuracy_192_153: - 5.09765625 frac[192,153] + mem[192,153] = 0
 accuracy_194_181: - 5.09765625 frac[194,181] + mem[194,181] = 0
 accuracy_33_113: - 5.09765625 frac[33,113] + mem[33,113] = 0
 accuracy_196_14: - 5.09765625 frac[196,14] + mem[196,14] = 0
 accuracy_72_52: - 5.09765625 frac[72,52] + mem[72,52] = 0
 accuracy_90_66: - 5.09765625 frac[90,66] + mem[90,66] = 0
 accuracy_130_67: - 5.09765625 frac[130,67] + mem[130,67] = 0
 accuracy_114_124: - 5.09765625 frac[114,124] + mem[114,124] = 0
 accuracy_79_100: - 5.09765625 frac[79,100] + mem[79,100] = 0
 accuracy_193_190: - 5.09765625 frac[193,190] + mem[193,190] = 0
 accuracy_194_36: - 5.09765625 frac[194,36] + mem[194,36] = 0
 accuracy_175_122: - 5.09765625 frac[175,122] + mem[175,122] = 0
 accuracy_195_206: - 5.09765625 frac[195,206] + mem[195,206] = 0
 accuracy_196_157: - 5.09765625 frac[196,157] + mem[196,157] = 0
 accuracy_197_41: - 5.09765625 frac[197,41] + mem[197,41] = 0
 accuracy_198_169: - 5.09765625 frac[198,169] + mem[198,169] = 0
 accuracy_77_142: - 5.09765625 frac[77,142] + mem[77,142] = 0
 accuracy_190_8: - 5.09765625 frac[190,8] + mem[190,8] = 0
 accuracy_9_73: - 5.09765625 frac[9,73] + mem[9,73] = 0
 accuracy_29_150: - 5.09765625 frac[29,150] + mem[29,150] = 0
 accuracy_194_87: - 5.09765625 frac[194,87] + mem[194,87] = 0
 accuracy_31_150: - 5.09765625 frac[31,150] + mem[31,150] = 0
 accuracy_195_95: - 5.09765625 frac[195,95] + mem[195,95] = 0
 accuracy_197_186: - 5.09765625 frac[197,186] + mem[197,186] = 0
 accuracy_198_56: - 5.09765625 frac[198,56] + mem[198,56] = 0
 accuracy_181_144: - 5.09765625 frac[181,144] + mem[181,144] = 0
 accuracy_113_13: - 5.09765625 frac[113,13] + mem[113,13] = 0
 accuracy_96_67: - 5.09765625 frac[96,67] + mem[96,67] = 0
 accuracy_166_17: - 5.09765625 frac[166,17] + mem[166,17] = 0
 accuracy_133_232: - 5.09765625 frac[133,232] + mem[133,232] = 0
 accuracy_79_186: - 5.09765625 frac[79,186] + mem[79,186] = 0
 accuracy_192_97: - 5.09765625 frac[192,97] + mem[192,97] = 0
 accuracy_101_105: - 5.09765625 frac[101,105] + mem[101,105] = 0
 accuracy_89_36: - 5.09765625 frac[89,36] + mem[89,36] = 0
 accuracy_198_75: - 5.09765625 frac[198,75] + mem[198,75] = 0
 accuracy_65_83: - 5.09765625 frac[65,83] + mem[65,83] = 0
 accuracy_193_6: - 5.09765625 frac[193,6] + mem[193,6] = 0
 accuracy_194_140: - 5.09765625 frac[194,140] + mem[194,140] = 0
 accuracy_67_166: - 5.09765625 frac[67,166] + mem[67,166] = 0
 accuracy_192_144: - 5.09765625 frac[192,144] + mem[192,144] = 0
 accuracy_196_117: - 5.09765625 frac[196,117] + mem[196,117] = 0
 accuracy_68_56: - 5.09765625 frac[68,56] + mem[68,56] = 0
 accuracy_92_166: - 5.09765625 frac[92,166] + mem[92,166] = 0
 accuracy_93_208: - 5.09765625 frac[93,208] + mem[93,208] = 0
 accuracy_189_50: - 5.09765625 frac[189,50] + mem[189,50] = 0
 accuracy_44_65: - 5.09765625 frac[44,65] + mem[44,65] = 0
 accuracy_99_172: - 5.09765625 frac[99,172] + mem[99,172] = 0
 accuracy_193_183: - 5.09765625 frac[193,183] + mem[193,183] = 0
 accuracy_28_240: - 5.09765625 frac[28,240] + mem[28,240] = 0
 accuracy_195_183: - 5.09765625 frac[195,183] + mem[195,183] = 0
 accuracy_196_132: - 5.09765625 frac[196,132] + mem[196,132] = 0
 accuracy_197_18: - 5.09765625 frac[197,18] + mem[197,18] = 0
 accuracy_198_160: - 5.09765625 frac[198,160] + mem[198,160] = 0
 accuracy_75_183: - 5.09765625 frac[75,183] + mem[75,183] = 0
 accuracy_130_205: - 5.09765625 frac[130,205] + mem[130,205] = 0
 accuracy_79_18: - 5.09765625 frac[79,18] + mem[79,18] = 0
 accuracy_195_68: - 5.09765625 frac[195,68] + mem[195,68] = 0
 accuracy_197_163: - 5.09765625 frac[197,163] + mem[197,163] = 0
 accuracy_198_51: - 5.09765625 frac[198,51] + mem[198,51] = 0
 accuracy_17_242: - 5.09765625 frac[17,242] + mem[17,242] = 0
 accuracy_36_231: - 5.09765625 frac[36,231] + mem[36,231] = 0
 accuracy_54_153: - 5.09765625 frac[54,153] + mem[54,153] = 0
 accuracy_64_51: - 5.09765625 frac[64,51] + mem[64,51] = 0
 accuracy_192_120: - 5.09765625 frac[192,120] + mem[192,120] = 0
 accuracy_193_110: - 5.09765625 frac[193,110] + mem[193,110] = 0
 accuracy_85_168: - 5.09765625 frac[85,168] + mem[85,168] = 0
 accuracy_31_44: - 5.09765625 frac[31,44] + mem[31,44] = 0
 accuracy_107_119: - 5.09765625 frac[107,119] + mem[107,119] = 0
 accuracy_198_66: - 5.09765625 frac[198,66] + mem[198,66] = 0
 accuracy_126_106: - 5.09765625 frac[126,106] + mem[126,106] = 0
 accuracy_113_151: - 5.09765625 frac[113,151] + mem[113,151] = 0
 accuracy_96_201: - 5.09765625 frac[96,201] + mem[96,201] = 0
 accuracy_167_107: - 5.09765625 frac[167,107] + mem[167,107] = 0
 accuracy_190_216: - 5.09765625 frac[190,216] + mem[190,216] = 0
 accuracy_192_151: - 5.09765625 frac[192,151] + mem[192,151] = 0
 accuracy_193_31: - 5.09765625 frac[193,31] + mem[193,31] = 0
 accuracy_104_25: - 5.09765625 frac[104,25] + mem[104,25] = 0
 accuracy_29_70: - 5.09765625 frac[29,70] + mem[29,70] = 0
 accuracy_194_135: - 5.09765625 frac[194,135] + mem[194,135] = 0
 accuracy_196_124: - 5.09765625 frac[196,124] + mem[196,124] = 0
 accuracy_90_84: - 5.09765625 frac[90,84] + mem[90,84] = 0
 accuracy_75_223: - 5.09765625 frac[75,223] + mem[75,223] = 0
 accuracy_170_172: - 5.09765625 frac[170,172] + mem[170,172] = 0
 accuracy_79_106: - 5.09765625 frac[79,106] + mem[79,106] = 0
 accuracy_118_211: - 5.09765625 frac[118,211] + mem[118,211] = 0
 accuracy_193_136: - 5.09765625 frac[193,136] + mem[193,136] = 0
 accuracy_29_247: - 5.09765625 frac[29,247] + mem[29,247] = 0
 accuracy_195_188: - 5.09765625 frac[195,188] + mem[195,188] = 0
 accuracy_196_131: - 5.09765625 frac[196,131] + mem[196,131] = 0
 accuracy_52_136: - 5.09765625 frac[52,136] + mem[52,136] = 0
 accuracy_198_155: - 5.09765625 frac[198,155] + mem[198,155] = 0
 accuracy_197_27: - 5.09765625 frac[197,27] + mem[197,27] = 0
 accuracy_2_27: - 5.09765625 frac[2,27] + mem[2,27] = 0
 accuracy_59_214: - 5.09765625 frac[59,214] + mem[59,214] = 0
 accuracy_60_216: - 5.09765625 frac[60,216] + mem[60,216] = 0
 accuracy_150_131: - 5.09765625 frac[150,131] + mem[150,131] = 0
 accuracy_139_158: - 5.09765625 frac[139,158] + mem[139,158] = 0
 accuracy_195_77: - 5.09765625 frac[195,77] + mem[195,77] = 0
 accuracy_88_17: - 5.09765625 frac[88,17] + mem[88,17] = 0
 accuracy_106_199: - 5.09765625 frac[106,199] + mem[106,199] = 0
 accuracy_197_172: - 5.09765625 frac[197,172] + mem[197,172] = 0
 accuracy_198_42: - 5.09765625 frac[198,42] + mem[198,42] = 0
 accuracy_162_158: - 5.09765625 frac[162,158] + mem[162,158] = 0
 accuracy_180_244: - 5.09765625 frac[180,244] + mem[180,244] = 0
 accuracy_184_239: - 5.09765625 frac[184,239] + mem[184,239] = 0
 accuracy_185_231: - 5.09765625 frac[185,231] + mem[185,231] = 0
 accuracy_96_81: - 5.09765625 frac[96,81] + mem[96,81] = 0
 accuracy_23_53: - 5.09765625 frac[23,53] + mem[23,53] = 0
 accuracy_192_127: - 5.09765625 frac[192,127] + mem[192,127] = 0
 accuracy_193_103: - 5.09765625 frac[193,103] + mem[193,103] = 0
 accuracy_158_210: - 5.09765625 frac[158,210] + mem[158,210] = 0
 accuracy_88_128: - 5.09765625 frac[88,128] + mem[88,128] = 0
 accuracy_198_69: - 5.09765625 frac[198,69] + mem[198,69] = 0
 accuracy_35_46: - 5.09765625 frac[35,46] + mem[35,46] = 0
 accuracy_183_254: - 5.09765625 frac[183,254] + mem[183,254] = 0
 accuracy_169_142: - 5.09765625 frac[169,142] + mem[169,142] = 0
 accuracy_192_238: - 5.09765625 frac[192,238] + mem[192,238] = 0
 accuracy_193_16: - 5.09765625 frac[193,16] + mem[193,16] = 0
 accuracy_194_158: - 5.09765625 frac[194,158] + mem[194,158] = 0
 accuracy_68_46: - 5.09765625 frac[68,46] + mem[68,46] = 0
 accuracy_196_123: - 5.09765625 frac[196,123] + mem[196,123] = 0
 accuracy_33_4: - 5.09765625 frac[33,4] + mem[33,4] = 0
 accuracy_197_115: - 5.09765625 frac[197,115] + mem[197,115] = 0
 accuracy_32_146: - 5.09765625 frac[32,146] + mem[32,146] = 0
 accuracy_56_227: - 5.09765625 frac[56,227] + mem[56,227] = 0
 accuracy_96_191: - 5.09765625 frac[96,191] + mem[96,191] = 0
 accuracy_95_153: - 5.09765625 frac[95,153] + mem[95,153] = 0
 accuracy_168_135: - 5.09765625 frac[168,135] + mem[168,135] = 0
 accuracy_169_31: - 5.09765625 frac[169,31] + mem[169,31] = 0
 accuracy_190_82: - 5.09765625 frac[190,82] + mem[190,82] = 0
 accuracy_191_255: - 5.09765625 frac[191,255] + mem[191,255] = 0
 accuracy_194_49: - 5.09765625 frac[194,49] + mem[194,49] = 0
 accuracy_195_165: - 5.09765625 frac[195,165] + mem[195,165] = 0
 accuracy_31_252: - 5.09765625 frac[31,252] + mem[31,252] = 0
 accuracy_193_129: - 5.09765625 frac[193,129] + mem[193,129] = 0
 accuracy_197_4: - 5.09765625 frac[197,4] + mem[197,4] = 0
 accuracy_30_230: - 5.09765625 frac[30,230] + mem[30,230] = 0
 accuracy_198_146: - 5.09765625 frac[198,146] + mem[198,146] = 0
 accuracy_196_138: - 5.09765625 frac[196,138] + mem[196,138] = 0
 accuracy_178_215: - 5.09765625 frac[178,215] + mem[178,215] = 0
 accuracy_27_6: - 5.09765625 frac[27,6] + mem[27,6] = 0
 accuracy_157_225: - 5.09765625 frac[157,225] + mem[157,225] = 0
 accuracy_195_50: - 5.09765625 frac[195,50] + mem[195,50] = 0
 accuracy_89_142: - 5.09765625 frac[89,142] + mem[89,142] = 0
 accuracy_197_149: - 5.09765625 frac[197,149] + mem[197,149] = 0
 accuracy_198_45: - 5.09765625 frac[198,45] + mem[198,45] = 0
 accuracy_168_80: - 5.09765625 frac[168,80] + mem[168,80] = 0
 accuracy_60_70: - 5.09765625 frac[60,70] + mem[60,70] = 0
 accuracy_136_121: - 5.09765625 frac[136,121] + mem[136,121] = 0
 accuracy_193_120: - 5.09765625 frac[193,120] + mem[193,120] = 0
 accuracy_102_18: - 5.09765625 frac[102,18] + mem[102,18] = 0
 accuracy_49_102: - 5.09765625 frac[49,102] + mem[49,102] = 0
 accuracy_90_183: - 5.09765625 frac[90,183] + mem[90,183] = 0
 accuracy_143_163: - 5.09765625 frac[143,163] + mem[143,163] = 0
 accuracy_39_82: - 5.09765625 frac[39,82] + mem[39,82] = 0
 accuracy_166_149: - 5.09765625 frac[166,149] + mem[166,149] = 0
 accuracy_42_143: - 5.09765625 frac[42,143] + mem[42,143] = 0
 accuracy_192_229: - 5.09765625 frac[192,229] + mem[192,229] = 0
 accuracy_193_233: - 5.09765625 frac[193,233] + mem[193,233] = 0
 accuracy_194_153: - 5.09765625 frac[194,153] + mem[194,153] = 0
 accuracy_49_247: - 5.09765625 frac[49,247] + mem[49,247] = 0
 accuracy_45_242: - 5.09765625 frac[45,242] + mem[45,242] = 0
 accuracy_196_98: - 5.09765625 frac[196,98] + mem[196,98] = 0
 accuracy_197_124: - 5.09765625 frac[197,124] + mem[197,124] = 0
 accuracy_48_90: - 5.09765625 frac[48,90] + mem[48,90] = 0
 accuracy_171_235: - 5.09765625 frac[171,235] + mem[171,235] = 0
 accuracy_193_154: - 5.09765625 frac[193,154] + mem[193,154] = 0
 accuracy_176_254: - 5.09765625 frac[176,254] + mem[176,254] = 0
 accuracy_194_8: - 5.09765625 frac[194,8] + mem[194,8] = 0
 accuracy_195_170: - 5.09765625 frac[195,170] + mem[195,170] = 0
 accuracy_196_241: - 5.09765625 frac[196,241] + mem[196,241] = 0
 accuracy_197_13: - 5.09765625 frac[197,13] + mem[197,13] = 0
 accuracy_198_149: - 5.09765625 frac[198,149] + mem[198,149] = 0
 accuracy_30_233: - 5.09765625 frac[30,233] + mem[30,233] = 0
 accuracy_42_84: - 5.09765625 frac[42,84] + mem[42,84] = 0
 accuracy_60_238: - 5.09765625 frac[60,238] + mem[60,238] = 0
 accuracy_170_33: - 5.09765625 frac[170,33] + mem[170,33] = 0
 accuracy_49_14: - 5.09765625 frac[49,14] + mem[49,14] = 0
 accuracy_195_59: - 5.09765625 frac[195,59] + mem[195,59] = 0
 accuracy_197_158: - 5.09765625 frac[197,158] + mem[197,158] = 0
 accuracy_198_36: - 5.09765625 frac[198,36] + mem[198,36] = 0
 accuracy_182_226: - 5.09765625 frac[182,226] + mem[182,226] = 0
 accuracy_21_160: - 5.09765625 frac[21,160] + mem[21,160] = 0
 accuracy_23_75: - 5.09765625 frac[23,75] + mem[23,75] = 0
 accuracy_62_226: - 5.09765625 frac[62,226] + mem[62,226] = 0
 accuracy_192_77: - 5.09765625 frac[192,77] + mem[192,77] = 0
 accuracy_193_113: - 5.09765625 frac[193,113] + mem[193,113] = 0
 accuracy_45_90: - 5.09765625 frac[45,90] + mem[45,90] = 0
 accuracy_46_152: - 5.09765625 frac[46,152] + mem[46,152] = 0
 accuracy_89_0: - 5.09765625 frac[89,0] + mem[89,0] = 0
 accuracy_159_214: - 5.09765625 frac[159,214] + mem[159,214] = 0
 accuracy_146_73: - 5.09765625 frac[146,73] + mem[146,73] = 0
 accuracy_192_252: - 5.09765625 frac[192,252] + mem[192,252] = 0
 accuracy_193_226: - 5.09765625 frac[193,226] + mem[193,226] = 0
 accuracy_194_144: - 5.09765625 frac[194,144] + mem[194,144] = 0
 accuracy_11_39: - 5.09765625 frac[11,39] + mem[11,39] = 0
 accuracy_140_116: - 5.09765625 frac[140,116] + mem[140,116] = 0
 accuracy_196_105: - 5.09765625 frac[196,105] + mem[196,105] = 0
 accuracy_197_101: - 5.09765625 frac[197,101] + mem[197,101] = 0
 accuracy_74_240: - 5.09765625 frac[74,240] + mem[74,240] = 0
 accuracy_80_23: - 5.09765625 frac[80,23] + mem[80,23] = 0
 accuracy_26_127: - 5.09765625 frac[26,127] + mem[26,127] = 0
 accuracy_193_147: - 5.09765625 frac[193,147] + mem[193,147] = 0
 accuracy_194_3: - 5.09765625 frac[194,3] + mem[194,3] = 0
 accuracy_195_147: - 5.09765625 frac[195,147] + mem[195,147] = 0
 accuracy_88_119: - 5.09765625 frac[88,119] + mem[88,119] = 0
 accuracy_196_248: - 5.09765625 frac[196,248] + mem[196,248] = 0
 accuracy_197_246: - 5.09765625 frac[197,246] + mem[197,246] = 0
 accuracy_198_140: - 5.09765625 frac[198,140] + mem[198,140] = 0
 accuracy_34_15: - 5.09765625 frac[34,15] + mem[34,15] = 0
 accuracy_71_150: - 5.09765625 frac[71,150] + mem[71,150] = 0
 accuracy_106_161: - 5.09765625 frac[106,161] + mem[106,161] = 0
 accuracy_35_231: - 5.09765625 frac[35,231] + mem[35,231] = 0
 accuracy_187_32: - 5.09765625 frac[187,32] + mem[187,32] = 0
 accuracy_137_161: - 5.09765625 frac[137,161] + mem[137,161] = 0
 accuracy_195_32: - 5.09765625 frac[195,32] + mem[195,32] = 0
 accuracy_103_204: - 5.09765625 frac[103,204] + mem[103,204] = 0
 accuracy_197_135: - 5.09765625 frac[197,135] + mem[197,135] = 0
 accuracy_198_31: - 5.09765625 frac[198,31] + mem[198,31] = 0
 accuracy_112_220: - 5.09765625 frac[112,220] + mem[112,220] = 0
 accuracy_25_46: - 5.09765625 frac[25,46] + mem[25,46] = 0
 accuracy_192_68: - 5.09765625 frac[192,68] + mem[192,68] = 0
 accuracy_193_74: - 5.09765625 frac[193,74] + mem[193,74] = 0
 accuracy_106_67: - 5.09765625 frac[106,67] + mem[106,67] = 0
 accuracy_18_119: - 5.09765625 frac[18,119] + mem[18,119] = 0
 accuracy_182_148: - 5.09765625 frac[182,148] + mem[182,148] = 0
 accuracy_129_24: - 5.09765625 frac[129,24] + mem[129,24] = 0
 accuracy_1_213: - 5.09765625 frac[1,213] + mem[1,213] = 0
 accuracy_22_152: - 5.09765625 frac[22,152] + mem[22,152] = 0
 accuracy_192_243: - 5.09765625 frac[192,243] + mem[192,243] = 0
 accuracy_193_251: - 5.09765625 frac[193,251] + mem[193,251] = 0
 accuracy_194_235: - 5.09765625 frac[194,235] + mem[194,235] = 0
 accuracy_32_167: - 5.09765625 frac[32,167] + mem[32,167] = 0
 accuracy_196_80: - 5.09765625 frac[196,80] + mem[196,80] = 0
 accuracy_197_110: - 5.09765625 frac[197,110] + mem[197,110] = 0
 accuracy_2_76: - 5.09765625 frac[2,76] + mem[2,76] = 0
 accuracy_115_32: - 5.09765625 frac[115,32] + mem[115,32] = 0
 accuracy_193_148: - 5.09765625 frac[193,148] + mem[193,148] = 0
 accuracy_194_26: - 5.09765625 frac[194,26] + mem[194,26] = 0
 accuracy_195_152: - 5.09765625 frac[195,152] + mem[195,152] = 0
 accuracy_196_231: - 5.09765625 frac[196,231] + mem[196,231] = 0
 accuracy_197_255: - 5.09765625 frac[197,255] + mem[197,255] = 0
 accuracy_108_1: - 5.09765625 frac[108,1] + mem[108,1] = 0
 accuracy_198_135: - 5.09765625 frac[198,135] + mem[198,135] = 0
 accuracy_148_2: - 5.09765625 frac[148,2] + mem[148,2] = 0
 accuracy_192_44: - 5.09765625 frac[192,44] + mem[192,44] = 0
 accuracy_156_194: - 5.09765625 frac[156,194] + mem[156,194] = 0
 accuracy_195_41: - 5.09765625 frac[195,41] + mem[195,41] = 0
 accuracy_197_128: - 5.09765625 frac[197,128] + mem[197,128] = 0
 accuracy_198_22: - 5.09765625 frac[198,22] + mem[198,22] = 0
 accuracy_113_219: - 5.09765625 frac[113,219] + mem[113,219] = 0
 accuracy_114_171: - 5.09765625 frac[114,171] + mem[114,171] = 0
 accuracy_192_91: - 5.09765625 frac[192,91] + mem[192,91] = 0
 accuracy_193_67: - 5.09765625 frac[193,67] + mem[193,67] = 0
 accuracy_67_227: - 5.09765625 frac[67,227] + mem[67,227] = 0
 accuracy_1_46: - 5.09765625 frac[1,46] + mem[1,46] = 0
 accuracy_192_202: - 5.09765625 frac[192,202] + mem[192,202] = 0
 accuracy_193_252: - 5.09765625 frac[193,252] + mem[193,252] = 0
 accuracy_194_226: - 5.09765625 frac[194,226] + mem[194,226] = 0
 accuracy_195_240: - 5.09765625 frac[195,240] + mem[195,240] = 0
 accuracy_196_95: - 5.09765625 frac[196,95] + mem[196,95] = 0
 accuracy_197_87: - 5.09765625 frac[197,87] + mem[197,87] = 0
 accuracy_73_89: - 5.09765625 frac[73,89] + mem[73,89] = 0
 accuracy_182_30: - 5.09765625 frac[182,30] + mem[182,30] = 0
 accuracy_4_188: - 5.09765625 frac[4,188] + mem[4,188] = 0
 accuracy_119_188: - 5.09765625 frac[119,188] + mem[119,188] = 0
 accuracy_194_21: - 5.09765625 frac[194,21] + mem[194,21] = 0
 accuracy_195_129: - 5.09765625 frac[195,129] + mem[195,129] = 0
 accuracy_177_235: - 5.09765625 frac[177,235] + mem[177,235] = 0
 accuracy_196_238: - 5.09765625 frac[196,238] + mem[196,238] = 0
 accuracy_197_248: - 5.09765625 frac[197,248] + mem[197,248] = 0
 accuracy_198_254: - 5.09765625 frac[198,254] + mem[198,254] = 0
 accuracy_59_11: - 5.09765625 frac[59,11] + mem[59,11] = 0
 accuracy_187_62: - 5.09765625 frac[187,62] + mem[187,62] = 0
 accuracy_192_35: - 5.09765625 frac[192,35] + mem[192,35] = 0
 accuracy_101_87: - 5.09765625 frac[101,87] + mem[101,87] = 0
 accuracy_66_91: - 5.09765625 frac[66,91] + mem[66,91] = 0
 accuracy_67_91: - 5.09765625 frac[67,91] + mem[67,91] = 0
 accuracy_195_46: - 5.09765625 frac[195,46] + mem[195,46] = 0
 accuracy_197_137: - 5.09765625 frac[197,137] + mem[197,137] = 0
 accuracy_15_150: - 5.09765625 frac[15,150] + mem[15,150] = 0
 accuracy_198_9: - 5.09765625 frac[198,9] + mem[198,9] = 0
 accuracy_92_113: - 5.09765625 frac[92,113] + mem[92,113] = 0
 accuracy_82_59: - 5.09765625 frac[82,59] + mem[82,59] = 0
 accuracy_192_82: - 5.09765625 frac[192,82] + mem[192,82] = 0
 accuracy_193_68: - 5.09765625 frac[193,68] + mem[193,68] = 0
 accuracy_66_202: - 5.09765625 frac[66,202] + mem[66,202] = 0
 accuracy_50_16: - 5.09765625 frac[50,16] + mem[50,16] = 0
 accuracy_196_55: - 5.09765625 frac[196,55] + mem[196,55] = 0
 accuracy_72_253: - 5.09765625 frac[72,253] + mem[72,253] = 0
 accuracy_91_254: - 5.09765625 frac[91,254] + mem[91,254] = 0
 accuracy_29_160: - 5.09765625 frac[29,160] + mem[29,160] = 0
 accuracy_78_55: - 5.09765625 frac[78,55] + mem[78,55] = 0
 accuracy_192_193: - 5.09765625 frac[192,193] + mem[192,193] = 0
 accuracy_193_245: - 5.09765625 frac[193,245] + mem[193,245] = 0
 accuracy_65_2: - 5.09765625 frac[65,2] + mem[65,2] = 0
 accuracy_195_249: - 5.09765625 frac[195,249] + mem[195,249] = 0
 accuracy_196_70: - 5.09765625 frac[196,70] + mem[196,70] = 0
 accuracy_124_126: - 5.09765625 frac[124,126] + mem[124,126] = 0
 accuracy_197_80: - 5.09765625 frac[197,80] + mem[197,80] = 0
 accuracy_77_197: - 5.09765625 frac[77,197] + mem[77,197] = 0
 accuracy_169_124: - 5.09765625 frac[169,124] + mem[169,124] = 0
 accuracy_117_137: - 5.09765625 frac[117,137] + mem[117,137] = 0
 accuracy_9_48: - 5.09765625 frac[9,48] + mem[9,48] = 0
 accuracy_10_142: - 5.09765625 frac[10,142] + mem[10,142] = 0
 accuracy_194_108: - 5.09765625 frac[194,108] + mem[194,108] = 0
 accuracy_195_134: - 5.09765625 frac[195,134] + mem[195,134] = 0
 accuracy_160_164: - 5.09765625 frac[160,164] + mem[160,164] = 0
 accuracy_196_213: - 5.09765625 frac[196,213] + mem[196,213] = 0
 accuracy_197_225: - 5.09765625 frac[197,225] + mem[197,225] = 0
 accuracy_198_241: - 5.09765625 frac[198,241] + mem[198,241] = 0
 accuracy_18_190: - 5.09765625 frac[18,190] + mem[18,190] = 0
 accuracy_36_161: - 5.09765625 frac[36,161] + mem[36,161] = 0
 accuracy_179_183: - 5.09765625 frac[179,183] + mem[179,183] = 0
 accuracy_61_167: - 5.09765625 frac[61,167] + mem[61,167] = 0
 accuracy_192_58: - 5.09765625 frac[192,58] + mem[192,58] = 0
 accuracy_65_249: - 5.09765625 frac[65,249] + mem[65,249] = 0
 accuracy_120_18: - 5.09765625 frac[120,18] + mem[120,18] = 0
 accuracy_157_222: - 5.09765625 frac[157,222] + mem[157,222] = 0
 accuracy_105_149: - 5.09765625 frac[105,149] + mem[105,149] = 0
 accuracy_175_163: - 5.09765625 frac[175,163] + mem[175,163] = 0
 accuracy_89_243: - 5.09765625 frac[89,243] + mem[89,243] = 0
 accuracy_49_42: - 5.09765625 frac[49,42] + mem[49,42] = 0
 accuracy_35_107: - 5.09765625 frac[35,107] + mem[35,107] = 0
 accuracy_158_169: - 5.09765625 frac[158,169] + mem[158,169] = 0
 accuracy_37_142: - 5.09765625 frac[37,142] + mem[37,142] = 0
 accuracy_168_51: - 5.09765625 frac[168,51] + mem[168,51] = 0
 accuracy_116_70: - 5.09765625 frac[116,70] + mem[116,70] = 0
 accuracy_117_112: - 5.09765625 frac[117,112] + mem[117,112] = 0
 accuracy_192_169: - 5.09765625 frac[192,169] + mem[192,169] = 0
 accuracy_193_93: - 5.09765625 frac[193,93] + mem[193,93] = 0
 accuracy_140_137: - 5.09765625 frac[140,137] + mem[140,137] = 0
 accuracy_196_62: - 5.09765625 frac[196,62] + mem[196,62] = 0
 accuracy_69_253: - 5.09765625 frac[69,253] + mem[69,253] = 0
 accuracy_32_93: - 5.09765625 frac[32,93] + mem[32,93] = 0
 accuracy_73_250: - 5.09765625 frac[73,250] + mem[73,250] = 0
 accuracy_163_143: - 5.09765625 frac[163,143] + mem[163,143] = 0
 accuracy_164_7: - 5.09765625 frac[164,7] + mem[164,7] = 0
 accuracy_117_225: - 5.09765625 frac[117,225] + mem[117,225] = 0
 accuracy_192_216: - 5.09765625 frac[192,216] + mem[192,216] = 0
 accuracy_193_206: - 5.09765625 frac[193,206] + mem[193,206] = 0
 accuracy_194_244: - 5.09765625 frac[194,244] + mem[194,244] = 0
 accuracy_195_254: - 5.09765625 frac[195,254] + mem[195,254] = 0
 accuracy_196_77: - 5.09765625 frac[196,77] + mem[196,77] = 0
 accuracy_197_89: - 5.09765625 frac[197,89] + mem[197,89] = 0
 accuracy_52_202: - 5.09765625 frac[52,202] + mem[52,202] = 0
 accuracy_143_61: - 5.09765625 frac[143,61] + mem[143,61] = 0
 accuracy_36_89: - 5.09765625 frac[36,89] + mem[36,89] = 0
 accuracy_54_39: - 5.09765625 frac[54,39] + mem[54,39] = 0
 accuracy_71_229: - 5.09765625 frac[71,229] + mem[71,229] = 0
 accuracy_133_119: - 5.09765625 frac[133,119] + mem[133,119] = 0
 accuracy_9_57: - 5.09765625 frac[9,57] + mem[9,57] = 0
 accuracy_12_34: - 5.09765625 frac[12,34] + mem[12,34] = 0
 accuracy_195_143: - 5.09765625 frac[195,143] + mem[195,143] = 0
 accuracy_196_220: - 5.09765625 frac[196,220] + mem[196,220] = 0
 accuracy_198_232: - 5.09765625 frac[198,232] + mem[198,232] = 0
 accuracy_55_220: - 5.09765625 frac[55,220] + mem[55,220] = 0
 accuracy_94_41: - 5.09765625 frac[94,41] + mem[94,41] = 0
 accuracy_5_77: - 5.09765625 frac[5,77] + mem[5,77] = 0
 accuracy_61_160: - 5.09765625 frac[61,160] + mem[61,160] = 0
 accuracy_192_49: - 5.09765625 frac[192,49] + mem[192,49] = 0
 accuracy_142_110: - 5.09765625 frac[142,110] + mem[142,110] = 0
 accuracy_198_123: - 5.09765625 frac[198,123] + mem[198,123] = 0
 accuracy_74_61: - 5.09765625 frac[74,61] + mem[74,61] = 0
 accuracy_40_101: - 5.09765625 frac[40,101] + mem[40,101] = 0
 accuracy_3_123: - 5.09765625 frac[3,123] + mem[3,123] = 0
 accuracy_24_148: - 5.09765625 frac[24,148] + mem[24,148] = 0
 accuracy_25_10: - 5.09765625 frac[25,10] + mem[25,10] = 0
 accuracy_192_160: - 5.09765625 frac[192,160] + mem[192,160] = 0
 accuracy_48_229: - 5.09765625 frac[48,229] + mem[48,229] = 0
 accuracy_63_29: - 5.09765625 frac[63,29] + mem[63,29] = 0
 accuracy_139_62: - 5.09765625 frac[139,62] + mem[139,62] = 0
 accuracy_193_86: - 5.09765625 frac[193,86] + mem[193,86] = 0
 accuracy_196_37: - 5.09765625 frac[196,37] + mem[196,37] = 0
 accuracy_26_43: - 5.09765625 frac[26,43] + mem[26,43] = 0
 accuracy_100_22: - 5.09765625 frac[100,22] + mem[100,22] = 0
 accuracy_192_223: - 5.09765625 frac[192,223] + mem[192,223] = 0
 accuracy_193_199: - 5.09765625 frac[193,199] + mem[193,199] = 0
 accuracy_29_14: - 5.09765625 frac[29,14] + mem[29,14] = 0
 accuracy_195_231: - 5.09765625 frac[195,231] + mem[195,231] = 0
 accuracy_194_207: - 5.09765625 frac[194,207] + mem[194,207] = 0
 accuracy_196_180: - 5.09765625 frac[196,180] + mem[196,180] = 0
 accuracy_197_66: - 5.09765625 frac[197,66] + mem[197,66] = 0
 accuracy_179_20: - 5.09765625 frac[179,20] + mem[179,20] = 0
 accuracy_53_184: - 5.09765625 frac[53,184] + mem[53,184] = 0
 accuracy_169_110: - 5.09765625 frac[169,110] + mem[169,110] = 0
 accuracy_5_37: - 5.09765625 frac[5,37] + mem[5,37] = 0
 accuracy_6_189: - 5.09765625 frac[6,189] + mem[6,189] = 0
 accuracy_99_237: - 5.09765625 frac[99,237] + mem[99,237] = 0
 accuracy_79_34: - 5.09765625 frac[79,34] + mem[79,34] = 0
 accuracy_194_126: - 5.09765625 frac[194,126] + mem[194,126] = 0
 accuracy_195_116: - 5.09765625 frac[195,116] + mem[195,116] = 0
 accuracy_174_210: - 5.09765625 frac[174,210] + mem[174,210] = 0
 accuracy_196_219: - 5.09765625 frac[196,219] + mem[196,219] = 0
 accuracy_197_211: - 5.09765625 frac[197,211] + mem[197,211] = 0
 accuracy_198_227: - 5.09765625 frac[198,227] + mem[198,227] = 0
 accuracy_103_152: - 5.09765625 frac[103,152] + mem[103,152] = 0
 accuracy_109_113: - 5.09765625 frac[109,113] + mem[109,113] = 0
 accuracy_108_101: - 5.09765625 frac[108,101] + mem[108,101] = 0
 accuracy_94_16: - 5.09765625 frac[94,16] + mem[94,16] = 0
 accuracy_58_164: - 5.09765625 frac[58,164] + mem[58,164] = 0
 accuracy_192_8: - 5.09765625 frac[192,8] + mem[192,8] = 0
 accuracy_10_3: - 5.09765625 frac[10,3] + mem[10,3] = 0
 accuracy_85_248: - 5.09765625 frac[85,248] + mem[85,248] = 0
 accuracy_103_233: - 5.09765625 frac[103,233] + mem[103,233] = 0
 accuracy_195_5: - 5.09765625 frac[195,5] + mem[195,5] = 0
 accuracy_161_149: - 5.09765625 frac[161,149] + mem[161,149] = 0
 accuracy_90_245: - 5.09765625 frac[90,245] + mem[90,245] = 0
 accuracy_198_114: - 5.09765625 frac[198,114] + mem[198,114] = 0
 accuracy_109_2: - 5.09765625 frac[109,2] + mem[109,2] = 0
 accuracy_18_195: - 5.09765625 frac[18,195] + mem[18,195] = 0
 accuracy_183_177: - 5.09765625 frac[183,177] + mem[183,177] = 0
 accuracy_40_108: - 5.09765625 frac[40,108] + mem[40,108] = 0
 accuracy_111_153: - 5.09765625 frac[111,153] + mem[111,153] = 0
 accuracy_117_66: - 5.09765625 frac[117,66] + mem[117,66] = 0
 accuracy_192_167: - 5.09765625 frac[192,167] + mem[192,167] = 0
 accuracy_193_47: - 5.09765625 frac[193,47] + mem[193,47] = 0
 accuracy_156_85: - 5.09765625 frac[156,85] + mem[156,85] = 0
 accuracy_196_44: - 5.09765625 frac[196,44] + mem[196,44] = 0
 accuracy_141_210: - 5.09765625 frac[141,210] + mem[141,210] = 0
 accuracy_90_100: - 5.09765625 frac[90,100] + mem[90,100] = 0
 accuracy_110_99: - 5.09765625 frac[110,99] + mem[110,99] = 0
 accuracy_130_101: - 5.09765625 frac[130,101] + mem[130,101] = 0
 accuracy_22_191: - 5.09765625 frac[22,191] + mem[22,191] = 0
 accuracy_189_27: - 5.09765625 frac[189,27] + mem[189,27] = 0
 accuracy_190_155: - 5.09765625 frac[190,155] + mem[190,155] = 0
 accuracy_193_216: - 5.09765625 frac[193,216] + mem[193,216] = 0
 accuracy_194_198: - 5.09765625 frac[194,198] + mem[194,198] = 0
 accuracy_33_204: - 5.09765625 frac[33,204] + mem[33,204] = 0
 accuracy_197_75: - 5.09765625 frac[197,75] + mem[197,75] = 0
 accuracy_144_73: - 5.09765625 frac[144,73] + mem[144,73] = 0
 accuracy_145_93: - 5.09765625 frac[145,93] + mem[145,93] = 0
 accuracy_196_179: - 5.09765625 frac[196,179] + mem[196,179] = 0
 accuracy_112_8: - 5.09765625 frac[112,8] + mem[112,8] = 0
 accuracy_77_240: - 5.09765625 frac[77,240] + mem[77,240] = 0
 accuracy_151_195: - 5.09765625 frac[151,195] + mem[151,195] = 0
 accuracy_7_133: - 5.09765625 frac[7,133] + mem[7,133] = 0
 accuracy_194_121: - 5.09765625 frac[194,121] + mem[194,121] = 0
 accuracy_195_125: - 5.09765625 frac[195,125] + mem[195,125] = 0
 accuracy_12_208: - 5.09765625 frac[12,208] + mem[12,208] = 0
 accuracy_196_194: - 5.09765625 frac[196,194] + mem[196,194] = 0
 accuracy_197_220: - 5.09765625 frac[197,220] + mem[197,220] = 0
 accuracy_36_222: - 5.09765625 frac[36,222] + mem[36,222] = 0
 accuracy_77_65: - 5.09765625 frac[77,65] + mem[77,65] = 0
 accuracy_131_47: - 5.09765625 frac[131,47] + mem[131,47] = 0
 accuracy_43_177: - 5.09765625 frac[43,177] + mem[43,177] = 0
 accuracy_134_8: - 5.09765625 frac[134,8] + mem[134,8] = 0
 accuracy_192_15: - 5.09765625 frac[192,15] + mem[192,15] = 0
 accuracy_156_61: - 5.09765625 frac[156,61] + mem[156,61] = 0
 accuracy_195_10: - 5.09765625 frac[195,10] + mem[195,10] = 0
 accuracy_105_136: - 5.09765625 frac[105,136] + mem[105,136] = 0
 accuracy_72_138: - 5.09765625 frac[72,138] + mem[72,138] = 0
 accuracy_198_117: - 5.09765625 frac[198,117] + mem[198,117] = 0
 accuracy_127_81: - 5.09765625 frac[127,81] + mem[127,81] = 0
 accuracy_146_7: - 5.09765625 frac[146,7] + mem[146,7] = 0
 accuracy_126_93: - 5.09765625 frac[126,93] + mem[126,93] = 0
 accuracy_166_66: - 5.09765625 frac[166,66] + mem[166,66] = 0
 accuracy_25_20: - 5.09765625 frac[25,20] + mem[25,20] = 0
 accuracy_192_190: - 5.09765625 frac[192,190] + mem[192,190] = 0
 accuracy_122_134: - 5.09765625 frac[122,134] + mem[122,134] = 0
 accuracy_194_174: - 5.09765625 frac[194,174] + mem[194,174] = 0
 accuracy_196_43: - 5.09765625 frac[196,43] + mem[196,43] = 0
 accuracy_186_1: - 5.09765625 frac[186,1] + mem[186,1] = 0
 accuracy_115_157: - 5.09765625 frac[115,157] + mem[115,157] = 0
 accuracy_188_170: - 5.09765625 frac[188,170] + mem[188,170] = 0
 accuracy_190_130: - 5.09765625 frac[190,130] + mem[190,130] = 0
 accuracy_155_108: - 5.09765625 frac[155,108] + mem[155,108] = 0
 accuracy_193_209: - 5.09765625 frac[193,209] + mem[193,209] = 0
 accuracy_195_213: - 5.09765625 frac[195,213] + mem[195,213] = 0
 accuracy_196_186: - 5.09765625 frac[196,186] + mem[196,186] = 0
 accuracy_197_52: - 5.09765625 frac[197,52] + mem[197,52] = 0
 accuracy_16_123: - 5.09765625 frac[16,123] + mem[16,123] = 0
 accuracy_53_138: - 5.09765625 frac[53,138] + mem[53,138] = 0
 accuracy_90_30: - 5.09765625 frac[90,30] + mem[90,30] = 0
 accuracy_77_233: - 5.09765625 frac[77,233] + mem[77,233] = 0
 accuracy_78_130: - 5.09765625 frac[78,130] + mem[78,130] = 0
 accuracy_82_244: - 5.09765625 frac[82,244] + mem[82,244] = 0
 accuracy_119_153: - 5.09765625 frac[119,153] + mem[119,153] = 0
 accuracy_194_112: - 5.09765625 frac[194,112] + mem[194,112] = 0
 accuracy_177_8: - 5.09765625 frac[177,8] + mem[177,8] = 0
 accuracy_195_98: - 5.09765625 frac[195,98] + mem[195,98] = 0
 accuracy_13_215: - 5.09765625 frac[13,215] + mem[13,215] = 0
 accuracy_197_197: - 5.09765625 frac[197,197] + mem[197,197] = 0
 accuracy_198_221: - 5.09765625 frac[198,221] + mem[198,221] = 0
 accuracy_196_201: - 5.09765625 frac[196,201] + mem[196,201] = 0
 accuracy_123_90: - 5.09765625 frac[123,90] + mem[123,90] = 0
 accuracy_0_116: - 5.09765625 frac[0,116] + mem[0,116] = 0
 accuracy_130_158: - 5.09765625 frac[130,158] + mem[130,158] = 0
 accuracy_187_83: - 5.09765625 frac[187,83] + mem[187,83] = 0
 accuracy_43_182: - 5.09765625 frac[43,182] + mem[43,182] = 0
 accuracy_133_243: - 5.09765625 frac[133,243] + mem[133,243] = 0
 accuracy_134_3: - 5.09765625 frac[134,3] + mem[134,3] = 0
 accuracy_192_6: - 5.09765625 frac[192,6] + mem[192,6] = 0
 accuracy_62_43: - 5.09765625 frac[62,43] + mem[62,43] = 0
 accuracy_189_62: - 5.09765625 frac[189,62] + mem[189,62] = 0
 accuracy_157_162: - 5.09765625 frac[157,162] + mem[157,162] = 0
 accuracy_70_27: - 5.09765625 frac[70,27] + mem[70,27] = 0
 accuracy_198_108: - 5.09765625 frac[198,108] + mem[198,108] = 0
 accuracy_19_244: - 5.09765625 frac[19,244] + mem[19,244] = 0
 accuracy_39_130: - 5.09765625 frac[39,130] + mem[39,130] = 0
 accuracy_167_73: - 5.09765625 frac[167,73] + mem[167,73] = 0
 accuracy_192_181: - 5.09765625 frac[192,181] + mem[192,181] = 0
 accuracy_193_57: - 5.09765625 frac[193,57] + mem[193,57] = 0
 accuracy_194_169: - 5.09765625 frac[194,169] + mem[194,169] = 0
 accuracy_47_194: - 5.09765625 frac[47,194] + mem[47,194] = 0
 accuracy_196_18: - 5.09765625 frac[196,18] + mem[196,18] = 0
 accuracy_130_119: - 5.09765625 frac[130,119] + mem[130,119] = 0
 accuracy_95_182: - 5.09765625 frac[95,182] + mem[95,182] = 0
 accuracy_193_170: - 5.09765625 frac[193,170] + mem[193,170] = 0
 accuracy_194_216: - 5.09765625 frac[194,216] + mem[194,216] = 0
 accuracy_30_153: - 5.09765625 frac[30,153] + mem[30,153] = 0
 accuracy_68_84: - 5.09765625 frac[68,84] + mem[68,84] = 0
 accuracy_124_153: - 5.09765625 frac[124,153] + mem[124,153] = 0
 accuracy_197_61: - 5.09765625 frac[197,61] + mem[197,61] = 0
 accuracy_196_161: - 5.09765625 frac[196,161] + mem[196,161] = 0
 accuracy_57_208: - 5.09765625 frac[57,208] + mem[57,208] = 0
 accuracy_131_140: - 5.09765625 frac[131,140] + mem[131,140] = 0
 accuracy_95_199: - 5.09765625 frac[95,199] + mem[95,199] = 0
 accuracy_152_7: - 5.09765625 frac[152,7] + mem[152,7] = 0
 accuracy_5_16: - 5.09765625 frac[5,16] + mem[5,16] = 0
 accuracy_104_213: - 5.09765625 frac[104,213] + mem[104,213] = 0
 accuracy_194_75: - 5.09765625 frac[194,75] + mem[194,75] = 0
 accuracy_13_208: - 5.09765625 frac[13,208] + mem[13,208] = 0
 accuracy_195_107: - 5.09765625 frac[195,107] + mem[195,107] = 0
 accuracy_16_129: - 5.09765625 frac[16,129] + mem[16,129] = 0
 accuracy_198_212: - 5.09765625 frac[198,212] + mem[198,212] = 0
 accuracy_17_21: - 5.09765625 frac[17,21] + mem[17,21] = 0
 accuracy_197_206: - 5.09765625 frac[197,206] + mem[197,206] = 0
 accuracy_18_157: - 5.09765625 frac[18,157] + mem[18,157] = 0
 accuracy_20_6: - 5.09765625 frac[20,6] + mem[20,6] = 0
 accuracy_130_153: - 5.09765625 frac[130,153] + mem[130,153] = 0
 accuracy_24_65: - 5.09765625 frac[24,65] + mem[24,65] = 0
 accuracy_159_6: - 5.09765625 frac[159,6] + mem[159,6] = 0
 accuracy_182_173: - 5.09765625 frac[182,173] + mem[182,173] = 0
 accuracy_39_187: - 5.09765625 frac[39,187] + mem[39,187] = 0
 accuracy_192_140: - 5.09765625 frac[192,140] + mem[192,140] = 0
 accuracy_65_79: - 5.09765625 frac[65,79] + mem[65,79] = 0
 accuracy_194_160: - 5.09765625 frac[194,160] + mem[194,160] = 0
 accuracy_193_50: - 5.09765625 frac[193,50] + mem[193,50] = 0
 accuracy_32_112: - 5.09765625 frac[32,112] + mem[32,112] = 0
 accuracy_196_25: - 5.09765625 frac[196,25] + mem[196,25] = 0
 accuracy_165_188: - 5.09765625 frac[165,188] + mem[165,188] = 0
 accuracy_8_249: - 5.09765625 frac[8,249] + mem[8,249] = 0
 accuracy_193_163: - 5.09765625 frac[193,163] + mem[193,163] = 0
 accuracy_194_211: - 5.09765625 frac[194,211] + mem[194,211] = 0
 accuracy_195_195: - 5.09765625 frac[195,195] + mem[195,195] = 0
 accuracy_105_65: - 5.09765625 frac[105,65] + mem[105,65] = 0
 accuracy_196_168: - 5.09765625 frac[196,168] + mem[196,168] = 0
 accuracy_197_38: - 5.09765625 frac[197,38] + mem[197,38] = 0
 accuracy_198_188: - 5.09765625 frac[198,188] + mem[198,188] = 0
 accuracy_143_30: - 5.09765625 frac[143,30] + mem[143,30] = 0
 accuracy_125_30: - 5.09765625 frac[125,30] + mem[125,30] = 0
 accuracy_2_52: - 5.09765625 frac[2,52] + mem[2,52] = 0
 accuracy_93_117: - 5.09765625 frac[93,117] + mem[93,117] = 0
 accuracy_169_74: - 5.09765625 frac[169,74] + mem[169,74] = 0
 accuracy_5_25: - 5.09765625 frac[5,25] + mem[5,25] = 0
 accuracy_79_6: - 5.09765625 frac[79,6] + mem[79,6] = 0
 accuracy_194_66: - 5.09765625 frac[194,66] + mem[194,66] = 0
 accuracy_12_253: - 5.09765625 frac[12,253] + mem[12,253] = 0
 accuracy_47_41: - 5.09765625 frac[47,41] + mem[47,41] = 0
 accuracy_195_80: - 5.09765625 frac[195,80] + mem[195,80] = 0
 accuracy_197_183: - 5.09765625 frac[197,183] + mem[197,183] = 0
 accuracy_198_207: - 5.09765625 frac[198,207] + mem[198,207] = 0
 accuracy_125_143: - 5.09765625 frac[125,143] + mem[125,143] = 0
 accuracy_58_184: - 5.09765625 frac[58,184] + mem[58,184] = 0
 accuracy_77_108: - 5.09765625 frac[77,108] + mem[77,108] = 0
 accuracy_25_126: - 5.09765625 frac[25,126] + mem[25,126] = 0
 accuracy_192_20: - 5.09765625 frac[192,20] + mem[192,20] = 0
 accuracy_85_156: - 5.09765625 frac[85,156] + mem[85,156] = 0
 accuracy_31_48: - 5.09765625 frac[31,48] + mem[31,48] = 0
 accuracy_198_94: - 5.09765625 frac[198,94] + mem[198,94] = 0
 accuracy_40_72: - 5.09765625 frac[40,72] + mem[40,72] = 0
 accuracy_23_129: - 5.09765625 frac[23,129] + mem[23,129] = 0
 accuracy_192_131: - 5.09765625 frac[192,131] + mem[192,131] = 0
 accuracy_193_11: - 5.09765625 frac[193,11] + mem[193,11] = 0
 accuracy_194_187: - 5.09765625 frac[194,187] + mem[194,187] = 0
 accuracy_139_83: - 5.09765625 frac[139,83] + mem[139,83] = 0
 accuracy_29_122: - 5.09765625 frac[29,122] + mem[29,122] = 0
 accuracy_196_0: - 5.09765625 frac[196,0] + mem[196,0] = 0
 accuracy_78_100: - 5.09765625 frac[78,100] + mem[78,100] = 0
 accuracy_189_63: - 5.09765625 frac[189,63] + mem[189,63] = 0
 accuracy_117_215: - 5.09765625 frac[117,215] + mem[117,215] = 0
 accuracy_193_164: - 5.09765625 frac[193,164] + mem[193,164] = 0
 accuracy_48_23: - 5.09765625 frac[48,23] + mem[48,23] = 0
 accuracy_195_200: - 5.09765625 frac[195,200] + mem[195,200] = 0
 accuracy_68_90: - 5.09765625 frac[68,90] + mem[68,90] = 0
 accuracy_196_151: - 5.09765625 frac[196,151] + mem[196,151] = 0
 accuracy_197_47: - 5.09765625 frac[197,47] + mem[197,47] = 0
 accuracy_198_183: - 5.09765625 frac[198,183] + mem[198,183] = 0
 accuracy_35_220: - 5.09765625 frac[35,220] + mem[35,220] = 0
 accuracy_194_42: - 5.09765625 frac[194,42] + mem[194,42] = 0
 accuracy_66_16: - 5.09765625 frac[66,16] + mem[66,16] = 0
 accuracy_194_93: - 5.09765625 frac[194,93] + mem[194,93] = 0
 accuracy_195_89: - 5.09765625 frac[195,89] + mem[195,89] = 0
 accuracy_197_176: - 5.09765625 frac[197,176] + mem[197,176] = 0
 accuracy_198_198: - 5.09765625 frac[198,198] + mem[198,198] = 0
 accuracy_18_143: - 5.09765625 frac[18,143] + mem[18,143] = 0
 accuracy_166_31: - 5.09765625 frac[166,31] + mem[166,31] = 0
 accuracy_187_102: - 5.09765625 frac[187,102] + mem[187,102] = 0
 accuracy_192_107: - 5.09765625 frac[192,107] + mem[192,107] = 0
 accuracy_140_203: - 5.09765625 frac[140,203] + mem[140,203] = 0
 accuracy_106_122: - 5.09765625 frac[106,122] + mem[106,122] = 0
 accuracy_14_99: - 5.09765625 frac[14,99] + mem[14,99] = 0
 accuracy_198_81: - 5.09765625 frac[198,81] + mem[198,81] = 0
 accuracy_35_58: - 5.09765625 frac[35,58] + mem[35,58] = 0
 accuracy_116_183: - 5.09765625 frac[116,183] + mem[116,183] = 0
 accuracy_99_89: - 5.09765625 frac[99,89] + mem[99,89] = 0
 accuracy_192_154: - 5.09765625 frac[192,154] + mem[192,154] = 0
 accuracy_193_12: - 5.09765625 frac[193,12] + mem[193,12] = 0
 accuracy_194_178: - 5.09765625 frac[194,178] + mem[194,178] = 0
 accuracy_196_15: - 5.09765625 frac[196,15] + mem[196,15] = 0
 accuracy_146_250: - 5.09765625 frac[146,250] + mem[146,250] = 0
 accuracy_93_214: - 5.09765625 frac[93,214] + mem[93,214] = 0
 accuracy_99_166: - 5.09765625 frac[99,166] + mem[99,166] = 0
 accuracy_9_255: - 5.09765625 frac[9,255] + mem[9,255] = 0
 accuracy_189_56: - 5.09765625 frac[189,56] + mem[189,56] = 0
 accuracy_193_189: - 5.09765625 frac[193,189] + mem[193,189] = 0
 accuracy_195_177: - 5.09765625 frac[195,177] + mem[195,177] = 0
 accuracy_33_225: - 5.09765625 frac[33,225] + mem[33,225] = 0
 accuracy_196_158: - 5.09765625 frac[196,158] + mem[196,158] = 0
 accuracy_198_174: - 5.09765625 frac[198,174] + mem[198,174] = 0
 accuracy_53_238: - 5.09765625 frac[53,238] + mem[53,238] = 0
 accuracy_197_40: - 5.09765625 frac[197,40] + mem[197,40] = 0
 accuracy_42_113: - 5.09765625 frac[42,113] + mem[42,113] = 0
 accuracy_132_243: - 5.09765625 frac[132,243] + mem[132,243] = 0
 accuracy_82_216: - 5.09765625 frac[82,216] + mem[82,216] = 0
 accuracy_173_47: - 5.09765625 frac[173,47] + mem[173,47] = 0
 accuracy_194_84: - 5.09765625 frac[194,84] + mem[194,84] = 0
 accuracy_195_94: - 5.09765625 frac[195,94] + mem[195,94] = 0
 accuracy_29_149: - 5.09765625 frac[29,149] + mem[29,149] = 0
 accuracy_197_185: - 5.09765625 frac[197,185] + mem[197,185] = 0
 accuracy_198_57: - 5.09765625 frac[198,57] + mem[198,57] = 0
 accuracy_35_82: - 5.09765625 frac[35,82] + mem[35,82] = 0
 accuracy_82_107: - 5.09765625 frac[82,107] + mem[82,107] = 0
 accuracy_192_98: - 5.09765625 frac[192,98] + mem[192,98] = 0
 accuracy_83_190: - 5.09765625 frac[83,190] + mem[83,190] = 0
 accuracy_139_240: - 5.09765625 frac[139,240] + mem[139,240] = 0
 accuracy_198_72: - 5.09765625 frac[198,72] + mem[198,72] = 0
 accuracy_110_158: - 5.09765625 frac[110,158] + mem[110,158] = 0
 accuracy_132_17: - 5.09765625 frac[132,17] + mem[132,17] = 0
 accuracy_192_145: - 5.09765625 frac[192,145] + mem[192,145] = 0
 accuracy_193_5: - 5.09765625 frac[193,5] + mem[193,5] = 0
 accuracy_194_141: - 5.09765625 frac[194,141] + mem[194,141] = 0
 accuracy_28_66: - 5.09765625 frac[28,66] + mem[28,66] = 0
 accuracy_164_95: - 5.09765625 frac[164,95] + mem[164,95] = 0
 accuracy_115_126: - 5.09765625 frac[115,126] + mem[115,126] = 0
 accuracy_170_162: - 5.09765625 frac[170,162] + mem[170,162] = 0
 accuracy_169_44: - 5.09765625 frac[169,44] + mem[169,44] = 0
 accuracy_99_175: - 5.09765625 frac[99,175] + mem[99,175] = 0
 accuracy_152_244: - 5.09765625 frac[152,244] + mem[152,244] = 0
 accuracy_173_151: - 5.09765625 frac[173,151] + mem[173,151] = 0
 accuracy_193_182: - 5.09765625 frac[193,182] + mem[193,182] = 0
 accuracy_47_87: - 5.09765625 frac[47,87] + mem[47,87] = 0
 accuracy_195_182: - 5.09765625 frac[195,182] + mem[195,182] = 0
 accuracy_194_60: - 5.09765625 frac[194,60] + mem[194,60] = 0
 accuracy_196_133: - 5.09765625 frac[196,133] + mem[196,133] = 0
 accuracy_197_17: - 5.09765625 frac[197,17] + mem[197,17] = 0
 accuracy_17_96: - 5.09765625 frac[17,96] + mem[17,96] = 0
 accuracy_198_161: - 5.09765625 frac[198,161] + mem[198,161] = 0
 accuracy_32_244: - 5.09765625 frac[32,244] + mem[32,244] = 0
 accuracy_165_56: - 5.09765625 frac[165,56] + mem[165,56] = 0
 accuracy_93_64: - 5.09765625 frac[93,64] + mem[93,64] = 0
 accuracy_133_79: - 5.09765625 frac[133,79] + mem[133,79] = 0
 accuracy_195_71: - 5.09765625 frac[195,71] + mem[195,71] = 0
 accuracy_103_171: - 5.09765625 frac[103,171] + mem[103,171] = 0
 accuracy_107_217: - 5.09765625 frac[107,217] + mem[107,217] = 0
 accuracy_197_162: - 5.09765625 frac[197,162] + mem[197,162] = 0
 accuracy_198_48: - 5.09765625 frac[198,48] + mem[198,48] = 0
 accuracy_108_182: - 5.09765625 frac[108,182] + mem[108,182] = 0
 accuracy_166_9: - 5.09765625 frac[166,9] + mem[166,9] = 0
 accuracy_77_23: - 5.09765625 frac[77,23] + mem[77,23] = 0
 accuracy_187_116: - 5.09765625 frac[187,116] + mem[187,116] = 0
 accuracy_81_92: - 5.09765625 frac[81,92] + mem[81,92] = 0
 accuracy_192_121: - 5.09765625 frac[192,121] + mem[192,121] = 0
 accuracy_191_19: - 5.09765625 frac[191,19] + mem[191,19] = 0
 accuracy_90_162: - 5.09765625 frac[90,162] + mem[90,162] = 0
 accuracy_198_67: - 5.09765625 frac[198,67] + mem[198,67] = 0
 accuracy_97_188: - 5.09765625 frac[97,188] + mem[97,188] = 0
 accuracy_22_253: - 5.09765625 frac[22,253] + mem[22,253] = 0
 accuracy_116_133: - 5.09765625 frac[116,133] + mem[116,133] = 0
 accuracy_192_232: - 5.09765625 frac[192,232] + mem[192,232] = 0
 accuracy_63_85: - 5.09765625 frac[63,85] + mem[63,85] = 0
 accuracy_194_132: - 5.09765625 frac[194,132] + mem[194,132] = 0
 accuracy_193_30: - 5.09765625 frac[193,30] + mem[193,30] = 0
 accuracy_135_125: - 5.09765625 frac[135,125] + mem[135,125] = 0
 accuracy_107_7: - 5.09765625 frac[107,7] + mem[107,7] = 0
 accuracy_169_21: - 5.09765625 frac[169,21] + mem[169,21] = 0
 accuracy_133_39: - 5.09765625 frac[133,39] + mem[133,39] = 0
 accuracy_172_6: - 5.09765625 frac[172,6] + mem[172,6] = 0
 accuracy_193_143: - 5.09765625 frac[193,143] + mem[193,143] = 0
 accuracy_194_55: - 5.09765625 frac[194,55] + mem[194,55] = 0
 accuracy_195_191: - 5.09765625 frac[195,191] + mem[195,191] = 0
 accuracy_86_9: - 5.09765625 frac[86,9] + mem[86,9] = 0
 accuracy_31_246: - 5.09765625 frac[31,246] + mem[31,246] = 0
 accuracy_196_140: - 5.09765625 frac[196,140] + mem[196,140] = 0
 accuracy_198_152: - 5.09765625 frac[198,152] + mem[198,152] = 0
 accuracy_145_44: - 5.09765625 frac[145,44] + mem[145,44] = 0
 accuracy_54_230: - 5.09765625 frac[54,230] + mem[54,230] = 0
 accuracy_197_26: - 5.09765625 frac[197,26] + mem[197,26] = 0
 accuracy_165_33: - 5.09765625 frac[165,33] + mem[165,33] = 0
 accuracy_60_217: - 5.09765625 frac[60,217] + mem[60,217] = 0
 accuracy_195_76: - 5.09765625 frac[195,76] + mem[195,76] = 0
 accuracy_197_171: - 5.09765625 frac[197,171] + mem[197,171] = 0
 accuracy_198_43: - 5.09765625 frac[198,43] + mem[198,43] = 0
 accuracy_108_173: - 5.09765625 frac[108,173] + mem[108,173] = 0
 accuracy_56_59: - 5.09765625 frac[56,59] + mem[56,59] = 0
 accuracy_77_16: - 5.09765625 frac[77,16] + mem[77,16] = 0
 accuracy_135_229: - 5.09765625 frac[135,229] + mem[135,229] = 0
 accuracy_193_102: - 5.09765625 frac[193,102] + mem[193,102] = 0
 accuracy_192_112: - 5.09765625 frac[192,112] + mem[192,112] = 0
 accuracy_28_33: - 5.09765625 frac[28,33] + mem[28,33] = 0
 accuracy_165_200: - 5.09765625 frac[165,200] + mem[165,200] = 0
 accuracy_113_143: - 5.09765625 frac[113,143] + mem[113,143] = 0
 accuracy_114_23: - 5.09765625 frac[114,23] + mem[114,23] = 0
 accuracy_95_11: - 5.09765625 frac[95,11] + mem[95,11] = 0
 accuracy_132_7: - 5.09765625 frac[132,7] + mem[132,7] = 0
 accuracy_100_70: - 5.09765625 frac[100,70] + mem[100,70] = 0
 accuracy_192_239: - 5.09765625 frac[192,239] + mem[192,239] = 0
 accuracy_193_23: - 5.09765625 frac[193,23] + mem[193,23] = 0
 accuracy_194_159: - 5.09765625 frac[194,159] + mem[194,159] = 0
 accuracy_47_244: - 5.09765625 frac[47,244] + mem[47,244] = 0
 accuracy_67_151: - 5.09765625 frac[67,151] + mem[67,151] = 0
 accuracy_196_100: - 5.09765625 frac[196,100] + mem[196,100] = 0
 accuracy_197_114: - 5.09765625 frac[197,114] + mem[197,114] = 0
 accuracy_53_72: - 5.09765625 frac[53,72] + mem[53,72] = 0
 accuracy_73_188: - 5.09765625 frac[73,188] + mem[73,188] = 0
 accuracy_182_51: - 5.09765625 frac[182,51] + mem[182,51] = 0
 accuracy_2_96: - 5.09765625 frac[2,96] + mem[2,96] = 0
 accuracy_166_226: - 5.09765625 frac[166,226] + mem[166,226] = 0
 accuracy_95_152: - 5.09765625 frac[95,152] + mem[95,152] = 0
 accuracy_78_64: - 5.09765625 frac[78,64] + mem[78,64] = 0
 accuracy_61_24: - 5.09765625 frac[61,24] + mem[61,24] = 0
 accuracy_170_148: - 5.09765625 frac[170,148] + mem[170,148] = 0
 accuracy_133_32: - 5.09765625 frac[133,32] + mem[133,32] = 0
 accuracy_193_128: - 5.09765625 frac[193,128] + mem[193,128] = 0
 accuracy_194_14: - 5.09765625 frac[194,14] + mem[194,14] = 0
 accuracy_177_206: - 5.09765625 frac[177,206] + mem[177,206] = 0
 accuracy_160_130: - 5.09765625 frac[160,130] + mem[160,130] = 0
 accuracy_195_164: - 5.09765625 frac[195,164] + mem[195,164] = 0
 accuracy_197_3: - 5.09765625 frac[197,3] + mem[197,3] = 0
 accuracy_198_147: - 5.09765625 frac[198,147] + mem[198,147] = 0
 accuracy_196_139: - 5.09765625 frac[196,139] + mem[196,139] = 0
 accuracy_51_30: - 5.09765625 frac[51,30] + mem[51,30] = 0
 accuracy_5_102: - 5.09765625 frac[5,102] + mem[5,102] = 0
 accuracy_79_227: - 5.09765625 frac[79,227] + mem[79,227] = 0
 accuracy_101_50: - 5.09765625 frac[101,50] + mem[101,50] = 0
 accuracy_157_224: - 5.09765625 frac[157,224] + mem[157,224] = 0
 accuracy_195_53: - 5.09765625 frac[195,53] + mem[195,53] = 0
 accuracy_106_207: - 5.09765625 frac[106,207] + mem[106,207] = 0
 accuracy_103_217: - 5.09765625 frac[103,217] + mem[103,217] = 0
 accuracy_197_148: - 5.09765625 frac[197,148] + mem[197,148] = 0
 accuracy_198_34: - 5.09765625 frac[198,34] + mem[198,34] = 0
 accuracy_180_252: - 5.09765625 frac[180,252] + mem[180,252] = 0
 accuracy_110_192: - 5.09765625 frac[110,192] + mem[110,192] = 0
 accuracy_107_215: - 5.09765625 frac[107,215] + mem[107,215] = 0
 accuracy_185_239: - 5.09765625 frac[185,239] + mem[185,239] = 0
 accuracy_77_9: - 5.09765625 frac[77,9] + mem[77,9] = 0
 accuracy_134_208: - 5.09765625 frac[134,208] + mem[134,208] = 0
 accuracy_192_119: - 5.09765625 frac[192,119] + mem[192,119] = 0
 accuracy_193_127: - 5.09765625 frac[193,127] + mem[193,127] = 0
 accuracy_28_56: - 5.09765625 frac[28,56] + mem[28,56] = 0
 accuracy_142_48: - 5.09765625 frac[142,48] + mem[142,48] = 0
 accuracy_71_122: - 5.09765625 frac[71,122] + mem[71,122] = 0
 accuracy_2_200: - 5.09765625 frac[2,200] + mem[2,200] = 0
 accuracy_38_65: - 5.09765625 frac[38,65] + mem[38,65] = 0
 accuracy_57_37: - 5.09765625 frac[57,37] + mem[57,37] = 0
 accuracy_169_246: - 5.09765625 frac[169,246] + mem[169,246] = 0
 accuracy_117_3: - 5.09765625 frac[117,3] + mem[117,3] = 0
 accuracy_154_63: - 5.09765625 frac[154,63] + mem[154,63] = 0
 accuracy_192_230: - 5.09765625 frac[192,230] + mem[192,230] = 0
 accuracy_193_232: - 5.09765625 frac[193,232] + mem[193,232] = 0
 accuracy_194_150: - 5.09765625 frac[194,150] + mem[194,150] = 0
 accuracy_174_58: - 5.09765625 frac[174,58] + mem[174,58] = 0
 accuracy_196_99: - 5.09765625 frac[196,99] + mem[196,99] = 0
 accuracy_197_123: - 5.09765625 frac[197,123] + mem[197,123] = 0
 accuracy_51_246: - 5.09765625 frac[51,246] + mem[51,246] = 0
 accuracy_124_91: - 5.09765625 frac[124,91] + mem[124,91] = 0
 accuracy_54_65: - 5.09765625 frac[54,65] + mem[54,65] = 0
 accuracy_56_235: - 5.09765625 frac[56,235] + mem[56,235] = 0
 accuracy_165_66: - 5.09765625 frac[165,66] + mem[165,66] = 0
 accuracy_2_123: - 5.09765625 frac[2,123] + mem[2,123] = 0
 accuracy_81_165: - 5.09765625 frac[81,165] + mem[81,165] = 0
 accuracy_193_153: - 5.09765625 frac[193,153] + mem[193,153] = 0
 accuracy_195_173: - 5.09765625 frac[195,173] + mem[195,173] = 0
 accuracy_196_242: - 5.09765625 frac[196,242] + mem[196,242] = 0
 accuracy_197_12: - 5.09765625 frac[197,12] + mem[197,12] = 0
 accuracy_179_90: - 5.09765625 frac[179,90] + mem[179,90] = 0
 accuracy_17_75: - 5.09765625 frac[17,75] + mem[17,75] = 0
 accuracy_198_138: - 5.09765625 frac[198,138] + mem[198,138] = 0
 accuracy_41_93: - 5.09765625 frac[41,93] + mem[41,93] = 0
 accuracy_27_30: - 5.09765625 frac[27,30] + mem[27,30] = 0
 accuracy_118_93: - 5.09765625 frac[118,93] + mem[118,93] = 0
 accuracy_195_58: - 5.09765625 frac[195,58] + mem[195,58] = 0
 accuracy_103_214: - 5.09765625 frac[103,214] + mem[103,214] = 0
 accuracy_86_138: - 5.09765625 frac[86,138] + mem[86,138] = 0
 accuracy_197_157: - 5.09765625 frac[197,157] + mem[197,157] = 0
 accuracy_198_37: - 5.09765625 frac[198,37] + mem[198,37] = 0
 accuracy_125_165: - 5.09765625 frac[125,165] + mem[125,165] = 0
 accuracy_92_101: - 5.09765625 frac[92,101] + mem[92,101] = 0
 accuracy_3_25: - 5.09765625 frac[3,25] + mem[3,25] = 0
 accuracy_192_78: - 5.09765625 frac[192,78] + mem[192,78] = 0
 accuracy_193_112: - 5.09765625 frac[193,112] + mem[193,112] = 0
 accuracy_49_158: - 5.09765625 frac[49,158] + mem[49,158] = 0
 accuracy_50_100: - 5.09765625 frac[50,100] + mem[50,100] = 0
 accuracy_72_201: - 5.09765625 frac[72,201] + mem[72,201] = 0
 accuracy_181_100: - 5.09765625 frac[181,100] + mem[181,100] = 0
 accuracy_39_74: - 5.09765625 frac[39,74] + mem[39,74] = 0
 accuracy_152_161: - 5.09765625 frac[152,161] + mem[152,161] = 0
 accuracy_188_250: - 5.09765625 frac[188,250] + mem[188,250] = 0
 accuracy_192_253: - 5.09765625 frac[192,253] + mem[192,253] = 0
 accuracy_193_225: - 5.09765625 frac[193,225] + mem[193,225] = 0
 accuracy_194_145: - 5.09765625 frac[194,145] + mem[194,145] = 0
 accuracy_196_106: - 5.09765625 frac[196,106] + mem[196,106] = 0
 accuracy_197_100: - 5.09765625 frac[197,100] + mem[197,100] = 0
 accuracy_90_46: - 5.09765625 frac[90,46] + mem[90,46] = 0
 accuracy_73_174: - 5.09765625 frac[73,174] + mem[73,174] = 0
 accuracy_146_217: - 5.09765625 frac[146,217] + mem[146,217] = 0
 accuracy_58_23: - 5.09765625 frac[58,23] + mem[58,23] = 0
 accuracy_193_146: - 5.09765625 frac[193,146] + mem[193,146] = 0
 accuracy_28_213: - 5.09765625 frac[28,213] + mem[28,213] = 0
 accuracy_103_126: - 5.09765625 frac[103,126] + mem[103,126] = 0
 accuracy_88_120: - 5.09765625 frac[88,120] + mem[88,120] = 0
 accuracy_196_249: - 5.09765625 frac[196,249] + mem[196,249] = 0
 accuracy_197_245: - 5.09765625 frac[197,245] + mem[197,245] = 0
 accuracy_198_141: - 5.09765625 frac[198,141] + mem[198,141] = 0
 accuracy_52_102: - 5.09765625 frac[52,102] + mem[52,102] = 0
 accuracy_160_144: - 5.09765625 frac[160,144] + mem[160,144] = 0
 accuracy_194_0: - 5.09765625 frac[194,0] + mem[194,0] = 0
 accuracy_158_42: - 5.09765625 frac[158,42] + mem[158,42] = 0
 accuracy_139_132: - 5.09765625 frac[139,132] + mem[139,132] = 0
 accuracy_195_35: - 5.09765625 frac[195,35] + mem[195,35] = 0
 accuracy_197_134: - 5.09765625 frac[197,134] + mem[197,134] = 0
 accuracy_18_229: - 5.09765625 frac[18,229] + mem[18,229] = 0
 accuracy_134_194: - 5.09765625 frac[134,194] + mem[134,194] = 0
 accuracy_192_69: - 5.09765625 frac[192,69] + mem[192,69] = 0
 accuracy_193_73: - 5.09765625 frac[193,73] + mem[193,73] = 0
 accuracy_142_2: - 5.09765625 frac[142,2] + mem[142,2] = 0
 accuracy_166_132: - 5.09765625 frac[166,132] + mem[166,132] = 0
 accuracy_152_168: - 5.09765625 frac[152,168] + mem[152,168] = 0
 accuracy_44_142: - 5.09765625 frac[44,142] + mem[44,142] = 0
 accuracy_190_245: - 5.09765625 frac[190,245] + mem[190,245] = 0
 accuracy_192_244: - 5.09765625 frac[192,244] + mem[192,244] = 0
 accuracy_193_250: - 5.09765625 frac[193,250] + mem[193,250] = 0
 accuracy_194_232: - 5.09765625 frac[194,232] + mem[194,232] = 0
 accuracy_85_60: - 5.09765625 frac[85,60] + mem[85,60] = 0
 accuracy_140_108: - 5.09765625 frac[140,108] + mem[140,108] = 0
 accuracy_196_81: - 5.09765625 frac[196,81] + mem[196,81] = 0
 accuracy_197_109: - 5.09765625 frac[197,109] + mem[197,109] = 0
 accuracy_63_73: - 5.09765625 frac[63,73] + mem[63,73] = 0
 accuracy_174_183: - 5.09765625 frac[174,183] + mem[174,183] = 0
 accuracy_195_155: - 5.09765625 frac[195,155] + mem[195,155] = 0
 accuracy_194_27: - 5.09765625 frac[194,27] + mem[194,27] = 0
 accuracy_196_224: - 5.09765625 frac[196,224] + mem[196,224] = 0
 accuracy_197_254: - 5.09765625 frac[197,254] + mem[197,254] = 0
 accuracy_35_239: - 5.09765625 frac[35,239] + mem[35,239] = 0
 accuracy_198_132: - 5.09765625 frac[198,132] + mem[198,132] = 0
 accuracy_192_45: - 5.09765625 frac[192,45] + mem[192,45] = 0
 accuracy_102_181: - 5.09765625 frac[102,181] + mem[102,181] = 0
 accuracy_49_63: - 5.09765625 frac[49,63] + mem[49,63] = 0
 accuracy_195_40: - 5.09765625 frac[195,40] + mem[195,40] = 0
 accuracy_197_143: - 5.09765625 frac[197,143] + mem[197,143] = 0
 accuracy_198_23: - 5.09765625 frac[198,23] + mem[198,23] = 0
 accuracy_39_235: - 5.09765625 frac[39,235] + mem[39,235] = 0
 accuracy_192_92: - 5.09765625 frac[192,92] + mem[192,92] = 0
 accuracy_193_66: - 5.09765625 frac[193,66] + mem[193,66] = 0
 accuracy_53_59: - 5.09765625 frac[53,59] + mem[53,59] = 0
 accuracy_57_56: - 5.09765625 frac[57,56] + mem[57,56] = 0
 accuracy_93_148: - 5.09765625 frac[93,148] + mem[93,148] = 0
 accuracy_135_94: - 5.09765625 frac[135,94] + mem[135,94] = 0
 accuracy_193_243: - 5.09765625 frac[193,243] + mem[193,243] = 0
 accuracy_194_227: - 5.09765625 frac[194,227] + mem[194,227] = 0
 accuracy_195_243: - 5.09765625 frac[195,243] + mem[195,243] = 0
 accuracy_196_88: - 5.09765625 frac[196,88] + mem[196,88] = 0
 accuracy_197_86: - 5.09765625 frac[197,86] + mem[197,86] = 0
 accuracy_92_153: - 5.09765625 frac[92,153] + mem[92,153] = 0
 accuracy_151_190: - 5.09765625 frac[151,190] + mem[151,190] = 0
 accuracy_153_208: - 5.09765625 frac[153,208] + mem[153,208] = 0
 accuracy_66_41: - 5.09765625 frac[66,41] + mem[66,41] = 0
 accuracy_194_18: - 5.09765625 frac[194,18] + mem[194,18] = 0
 accuracy_102_221: - 5.09765625 frac[102,221] + mem[102,221] = 0
 accuracy_13_185: - 5.09765625 frac[13,185] + mem[13,185] = 0
 accuracy_196_239: - 5.09765625 frac[196,239] + mem[196,239] = 0
 accuracy_197_231: - 5.09765625 frac[197,231] + mem[197,231] = 0
 accuracy_198_255: - 5.09765625 frac[198,255] + mem[198,255] = 0
 accuracy_192_36: - 5.09765625 frac[192,36] + mem[192,36] = 0
 accuracy_49_48: - 5.09765625 frac[49,48] + mem[49,48] = 0
 accuracy_195_17: - 5.09765625 frac[195,17] + mem[195,17] = 0
 accuracy_197_136: - 5.09765625 frac[197,136] + mem[197,136] = 0
 accuracy_198_14: - 5.09765625 frac[198,14] + mem[198,14] = 0
 accuracy_0_1: - 5.09765625 frac[0,1] + mem[0,1] = 0
 accuracy_74_8: - 5.09765625 frac[74,8] + mem[74,8] = 0
 accuracy_127_188: - 5.09765625 frac[127,188] + mem[127,188] = 0
 accuracy_20_204: - 5.09765625 frac[20,204] + mem[20,204] = 0
 accuracy_37_132: - 5.09765625 frac[37,132] + mem[37,132] = 0
 accuracy_116_120: - 5.09765625 frac[116,120] + mem[116,120] = 0
 accuracy_26_183: - 5.09765625 frac[26,183] + mem[26,183] = 0
 accuracy_65_144: - 5.09765625 frac[65,144] + mem[65,144] = 0
 accuracy_193_91: - 5.09765625 frac[193,91] + mem[193,91] = 0
 accuracy_192_83: - 5.09765625 frac[192,83] + mem[192,83] = 0
 accuracy_196_48: - 5.09765625 frac[196,48] + mem[196,48] = 0
 accuracy_163_137: - 5.09765625 frac[163,137] + mem[163,137] = 0
 accuracy_40_7: - 5.09765625 frac[40,7] + mem[40,7] = 0
 accuracy_152_182: - 5.09765625 frac[152,182] + mem[152,182] = 0
 accuracy_116_239: - 5.09765625 frac[116,239] + mem[116,239] = 0
 accuracy_192_194: - 5.09765625 frac[192,194] + mem[192,194] = 0
 accuracy_193_244: - 5.09765625 frac[193,244] + mem[193,244] = 0
 accuracy_104_100: - 5.09765625 frac[104,100] + mem[104,100] = 0
 accuracy_195_248: - 5.09765625 frac[195,248] + mem[195,248] = 0
 accuracy_122_210: - 5.09765625 frac[122,210] + mem[122,210] = 0
 accuracy_196_71: - 5.09765625 frac[196,71] + mem[196,71] = 0
 accuracy_197_95: - 5.09765625 frac[197,95] + mem[197,95] = 0
 accuracy_194_250: - 5.09765625 frac[194,250] + mem[194,250] = 0
 accuracy_92_144: - 5.09765625 frac[92,144] + mem[92,144] = 0
 accuracy_167_141: - 5.09765625 frac[167,141] + mem[167,141] = 0
 accuracy_189_96: - 5.09765625 frac[189,96] + mem[189,96] = 0
 accuracy_27_77: - 5.09765625 frac[27,77] + mem[27,77] = 0
 accuracy_7_153: - 5.09765625 frac[7,153] + mem[7,153] = 0
 accuracy_194_109: - 5.09765625 frac[194,109] + mem[194,109] = 0
 accuracy_195_137: - 5.09765625 frac[195,137] + mem[195,137] = 0
 accuracy_12_36: - 5.09765625 frac[12,36] + mem[12,36] = 0
 accuracy_196_214: - 5.09765625 frac[196,214] + mem[196,214] = 0
 accuracy_197_224: - 5.09765625 frac[197,224] + mem[197,224] = 0
 accuracy_198_246: - 5.09765625 frac[198,246] + mem[198,246] = 0
 accuracy_174_193: - 5.09765625 frac[174,193] + mem[174,193] = 0
 accuracy_94_39: - 5.09765625 frac[94,39] + mem[94,39] = 0
 accuracy_151_36: - 5.09765625 frac[151,36] + mem[151,36] = 0
 accuracy_135_174: - 5.09765625 frac[135,174] + mem[135,174] = 0
 accuracy_192_59: - 5.09765625 frac[192,59] + mem[192,59] = 0
 accuracy_157_221: - 5.09765625 frac[157,221] + mem[157,221] = 0
 accuracy_30_85: - 5.09765625 frac[30,85] + mem[30,85] = 0
 accuracy_198_1: - 5.09765625 frac[198,1] + mem[198,1] = 0
 accuracy_18_206: - 5.09765625 frac[18,206] + mem[18,206] = 0
 accuracy_17_192: - 5.09765625 frac[17,192] + mem[17,192] = 0
 accuracy_55_119: - 5.09765625 frac[55,119] + mem[55,119] = 0
 accuracy_96_12: - 5.09765625 frac[96,12] + mem[96,12] = 0
 accuracy_116_71: - 5.09765625 frac[116,71] + mem[116,71] = 0
 accuracy_192_170: - 5.09765625 frac[192,170] + mem[192,170] = 0
 accuracy_193_92: - 5.09765625 frac[193,92] + mem[193,92] = 0
 accuracy_196_63: - 5.09765625 frac[196,63] + mem[196,63] = 0
 accuracy_57_74: - 5.09765625 frac[57,74] + mem[57,74] = 0
 accuracy_97_227: - 5.09765625 frac[97,227] + mem[97,227] = 0
 accuracy_171_190: - 5.09765625 frac[171,190] + mem[171,190] = 0
 accuracy_192_217: - 5.09765625 frac[192,217] + mem[192,217] = 0
 accuracy_193_205: - 5.09765625 frac[193,205] + mem[193,205] = 0
 accuracy_194_245: - 5.09765625 frac[194,245] + mem[194,245] = 0
 accuracy_195_225: - 5.09765625 frac[195,225] + mem[195,225] = 0
 accuracy_28_138: - 5.09765625 frac[28,138] + mem[28,138] = 0
 accuracy_196_78: - 5.09765625 frac[196,78] + mem[196,78] = 0
 accuracy_197_88: - 5.09765625 frac[197,88] + mem[197,88] = 0
 accuracy_121_229: - 5.09765625 frac[121,229] + mem[121,229] = 0
 accuracy_19_2: - 5.09765625 frac[19,2] + mem[19,2] = 0
 accuracy_37_84: - 5.09765625 frac[37,84] + mem[37,84] = 0
 accuracy_148_169: - 5.09765625 frac[148,169] + mem[148,169] = 0
 accuracy_78_174: - 5.09765625 frac[78,174] + mem[78,174] = 0
 accuracy_194_100: - 5.09765625 frac[194,100] + mem[194,100] = 0
 accuracy_195_142: - 5.09765625 frac[195,142] + mem[195,142] = 0
 accuracy_196_221: - 5.09765625 frac[196,221] + mem[196,221] = 0
 accuracy_197_233: - 5.09765625 frac[197,233] + mem[197,233] = 0
 accuracy_198_233: - 5.09765625 frac[198,233] + mem[198,233] = 0
 accuracy_152_91: - 5.09765625 frac[152,91] + mem[152,91] = 0
 accuracy_192_50: - 5.09765625 frac[192,50] + mem[192,50] = 0
 accuracy_63_143: - 5.09765625 frac[63,143] + mem[63,143] = 0
 accuracy_195_31: - 5.09765625 frac[195,31] + mem[195,31] = 0
 accuracy_88_195: - 5.09765625 frac[88,195] + mem[88,195] = 0
 accuracy_86_169: - 5.09765625 frac[86,169] + mem[86,169] = 0
 accuracy_70_55: - 5.09765625 frac[70,55] + mem[70,55] = 0
 accuracy_198_120: - 5.09765625 frac[198,120] + mem[198,120] = 0
 accuracy_19_224: - 5.09765625 frac[19,224] + mem[19,224] = 0
 accuracy_184_161: - 5.09765625 frac[184,161] + mem[184,161] = 0
 accuracy_21_204: - 5.09765625 frac[21,204] + mem[21,204] = 0
 accuracy_117_120: - 5.09765625 frac[117,120] + mem[117,120] = 0
 accuracy_120_137: - 5.09765625 frac[120,137] + mem[120,137] = 0
 accuracy_193_85: - 5.09765625 frac[193,85] + mem[193,85] = 0
 accuracy_192_161: - 5.09765625 frac[192,161] + mem[192,161] = 0
 accuracy_135_92: - 5.09765625 frac[135,92] + mem[135,92] = 0
 accuracy_46_188: - 5.09765625 frac[46,188] + mem[46,188] = 0
 accuracy_196_38: - 5.09765625 frac[196,38] + mem[196,38] = 0
 accuracy_33_89: - 5.09765625 frac[33,89] + mem[33,89] = 0
 accuracy_69_229: - 5.09765625 frac[69,229] + mem[69,229] = 0
 accuracy_14_37: - 5.09765625 frac[14,37] + mem[14,37] = 0
 accuracy_1_48: - 5.09765625 frac[1,48] + mem[1,48] = 0
 accuracy_93_255: - 5.09765625 frac[93,255] + mem[93,255] = 0
 accuracy_58_67: - 5.09765625 frac[58,67] + mem[58,67] = 0
 accuracy_115_142: - 5.09765625 frac[115,142] + mem[115,142] = 0
 accuracy_192_208: - 5.09765625 frac[192,208] + mem[192,208] = 0
 accuracy_193_198: - 5.09765625 frac[193,198] + mem[193,198] = 0
 accuracy_194_204: - 5.09765625 frac[194,204] + mem[194,204] = 0
 accuracy_123_222: - 5.09765625 frac[123,222] + mem[123,222] = 0
 accuracy_195_230: - 5.09765625 frac[195,230] + mem[195,230] = 0
 accuracy_196_181: - 5.09765625 frac[196,181] + mem[196,181] = 0
 accuracy_197_65: - 5.09765625 frac[197,65] + mem[197,65] = 0
 accuracy_70_209: - 5.09765625 frac[70,209] + mem[70,209] = 0
 accuracy_125_121: - 5.09765625 frac[125,121] + mem[125,121] = 0
 accuracy_130_26: - 5.09765625 frac[130,26] + mem[130,26] = 0
 accuracy_131_152: - 5.09765625 frac[131,152] + mem[131,152] = 0
 accuracy_95_235: - 5.09765625 frac[95,235] + mem[95,235] = 0
 accuracy_97_85: - 5.09765625 frac[97,85] + mem[97,85] = 0
 accuracy_194_127: - 5.09765625 frac[194,127] + mem[194,127] = 0
 accuracy_195_119: - 5.09765625 frac[195,119] + mem[195,119] = 0
 accuracy_196_196: - 5.09765625 frac[196,196] + mem[196,196] = 0
 accuracy_197_210: - 5.09765625 frac[197,210] + mem[197,210] = 0
 accuracy_198_224: - 5.09765625 frac[198,224] + mem[198,224] = 0
 accuracy_61_184: - 5.09765625 frac[61,184] + mem[61,184] = 0
 accuracy_192_9: - 5.09765625 frac[192,9] + mem[192,9] = 0
 accuracy_119_59: - 5.09765625 frac[119,59] + mem[119,59] = 0
 accuracy_138_221: - 5.09765625 frac[138,221] + mem[138,221] = 0
 accuracy_139_217: - 5.09765625 frac[139,217] + mem[139,217] = 0
 accuracy_195_4: - 5.09765625 frac[195,4] + mem[195,4] = 0
 accuracy_198_115: - 5.09765625 frac[198,115] + mem[198,115] = 0
 accuracy_92_95: - 5.09765625 frac[92,95] + mem[92,95] = 0
 accuracy_39_143: - 5.09765625 frac[39,143] + mem[39,143] = 0
 accuracy_192_184: - 5.09765625 frac[192,184] + mem[192,184] = 0
 accuracy_193_46: - 5.09765625 frac[193,46] + mem[193,46] = 0
 accuracy_157_64: - 5.09765625 frac[157,64] + mem[157,64] = 0
 accuracy_196_45: - 5.09765625 frac[196,45] + mem[196,45] = 0
 accuracy_161_5: - 5.09765625 frac[161,5] + mem[161,5] = 0
 accuracy_91_196: - 5.09765625 frac[91,196] + mem[91,196] = 0
 accuracy_94_254: - 5.09765625 frac[94,254] + mem[94,254] = 0
 accuracy_149_113: - 5.09765625 frac[149,113] + mem[149,113] = 0
 accuracy_98_229: - 5.09765625 frac[98,229] + mem[98,229] = 0
 accuracy_22_188: - 5.09765625 frac[22,188] + mem[22,188] = 0
 accuracy_192_215: - 5.09765625 frac[192,215] + mem[192,215] = 0
 accuracy_193_223: - 5.09765625 frac[193,223] + mem[193,223] = 0
 accuracy_194_199: - 5.09765625 frac[194,199] + mem[194,199] = 0
 accuracy_195_239: - 5.09765625 frac[195,239] + mem[195,239] = 0
 accuracy_196_188: - 5.09765625 frac[196,188] + mem[196,188] = 0
 accuracy_51_41: - 5.09765625 frac[51,41] + mem[51,41] = 0
 accuracy_197_74: - 5.09765625 frac[197,74] + mem[197,74] = 0
 accuracy_16_5: - 5.09765625 frac[16,5] + mem[16,5] = 0
 accuracy_41_3: - 5.09765625 frac[41,3] + mem[41,3] = 0
 accuracy_115_4: - 5.09765625 frac[115,4] + mem[115,4] = 0
 accuracy_152_58: - 5.09765625 frac[152,58] + mem[152,58] = 0
 accuracy_136_212: - 5.09765625 frac[136,212] + mem[136,212] = 0
 accuracy_195_124: - 5.09765625 frac[195,124] + mem[195,124] = 0
 accuracy_196_195: - 5.09765625 frac[196,195] + mem[196,195] = 0
 accuracy_197_219: - 5.09765625 frac[197,219] + mem[197,219] = 0
 accuracy_198_219: - 5.09765625 frac[198,219] + mem[198,219] = 0
 accuracy_16_148: - 5.09765625 frac[16,148] + mem[16,148] = 0
 accuracy_153_93: - 5.09765625 frac[153,93] + mem[153,93] = 0
 accuracy_63_189: - 5.09765625 frac[63,189] + mem[63,189] = 0
 accuracy_192_0: - 5.09765625 frac[192,0] + mem[192,0] = 0
 accuracy_84_22: - 5.09765625 frac[84,22] + mem[84,22] = 0
 accuracy_195_13: - 5.09765625 frac[195,13] + mem[195,13] = 0
 accuracy_31_68: - 5.09765625 frac[31,68] + mem[31,68] = 0
 accuracy_88_209: - 5.09765625 frac[88,209] + mem[88,209] = 0
 accuracy_198_106: - 5.09765625 frac[198,106] + mem[198,106] = 0
 accuracy_180_180: - 5.09765625 frac[180,180] + mem[180,180] = 0
 accuracy_40_84: - 5.09765625 frac[40,84] + mem[40,84] = 0
 accuracy_116_92: - 5.09765625 frac[116,92] + mem[116,92] = 0
 accuracy_82_28: - 5.09765625 frac[82,28] + mem[82,28] = 0
 accuracy_192_191: - 5.09765625 frac[192,191] + mem[192,191] = 0
 accuracy_193_39: - 5.09765625 frac[193,39] + mem[193,39] = 0
 accuracy_194_175: - 5.09765625 frac[194,175] + mem[194,175] = 0
 accuracy_196_20: - 5.09765625 frac[196,20] + mem[196,20] = 0
 accuracy_161_30: - 5.09765625 frac[161,30] + mem[161,30] = 0
 accuracy_147_223: - 5.09765625 frac[147,223] + mem[147,223] = 0
 accuracy_149_122: - 5.09765625 frac[149,122] + mem[149,122] = 0
 accuracy_27_184: - 5.09765625 frac[27,184] + mem[27,184] = 0
 accuracy_193_208: - 5.09765625 frac[193,208] + mem[193,208] = 0
 accuracy_194_222: - 5.09765625 frac[194,222] + mem[194,222] = 0
 accuracy_195_212: - 5.09765625 frac[195,212] + mem[195,212] = 0
 accuracy_122_246: - 5.09765625 frac[122,246] + mem[122,246] = 0
 accuracy_196_187: - 5.09765625 frac[196,187] + mem[196,187] = 0
 accuracy_197_51: - 5.09765625 frac[197,51] + mem[197,51] = 0
 accuracy_152_1: - 5.09765625 frac[152,1] + mem[152,1] = 0
 accuracy_119_152: - 5.09765625 frac[119,152] + mem[119,152] = 0
 accuracy_156_134: - 5.09765625 frac[156,134] + mem[156,134] = 0
 accuracy_194_113: - 5.09765625 frac[194,113] + mem[194,113] = 0
 accuracy_195_101: - 5.09765625 frac[195,101] + mem[195,101] = 0
 accuracy_103_137: - 5.09765625 frac[103,137] + mem[103,137] = 0
 accuracy_196_202: - 5.09765625 frac[196,202] + mem[196,202] = 0
 accuracy_125_252: - 5.09765625 frac[125,252] + mem[125,252] = 0
 accuracy_124_242: - 5.09765625 frac[124,242] + mem[124,242] = 0
 accuracy_198_210: - 5.09765625 frac[198,210] + mem[198,210] = 0
 accuracy_56_18: - 5.09765625 frac[56,18] + mem[56,18] = 0
 accuracy_102_240: - 5.09765625 frac[102,240] + mem[102,240] = 0
 accuracy_197_196: - 5.09765625 frac[197,196] + mem[197,196] = 0
 accuracy_132_175: - 5.09765625 frac[132,175] + mem[132,175] = 0
 accuracy_24_123: - 5.09765625 frac[24,123] + mem[24,123] = 0
 accuracy_136_170: - 5.09765625 frac[136,170] + mem[136,170] = 0
 accuracy_192_7: - 5.09765625 frac[192,7] + mem[192,7] = 0
 accuracy_83_165: - 5.09765625 frac[83,165] + mem[83,165] = 0
 accuracy_14_143: - 5.09765625 frac[14,143] + mem[14,143] = 0
 accuracy_198_109: - 5.09765625 frac[198,109] + mem[198,109] = 0
 accuracy_39_189: - 5.09765625 frac[39,189] + mem[39,189] = 0
 accuracy_114_254: - 5.09765625 frac[114,254] + mem[114,254] = 0
 accuracy_168_16: - 5.09765625 frac[168,16] + mem[168,16] = 0
 accuracy_192_182: - 5.09765625 frac[192,182] + mem[192,182] = 0
 accuracy_193_56: - 5.09765625 frac[193,56] + mem[193,56] = 0
 accuracy_11_17: - 5.09765625 frac[11,17] + mem[11,17] = 0
 accuracy_194_166: - 5.09765625 frac[194,166] + mem[194,166] = 0
 accuracy_120_158: - 5.09765625 frac[120,158] + mem[120,158] = 0
 accuracy_196_19: - 5.09765625 frac[196,19] + mem[196,19] = 0
 accuracy_33_108: - 5.09765625 frac[33,108] + mem[33,108] = 0
 accuracy_71_59: - 5.09765625 frac[71,59] + mem[71,59] = 0
 accuracy_96_135: - 5.09765625 frac[96,135] + mem[96,135] = 0
 accuracy_115_101: - 5.09765625 frac[115,101] + mem[115,101] = 0
 accuracy_78_27: - 5.09765625 frac[78,27] + mem[78,27] = 0
 accuracy_8_243: - 5.09765625 frac[8,243] + mem[8,243] = 0
 accuracy_193_169: - 5.09765625 frac[193,169] + mem[193,169] = 0
 accuracy_194_217: - 5.09765625 frac[194,217] + mem[194,217] = 0
 accuracy_195_221: - 5.09765625 frac[195,221] + mem[195,221] = 0
 accuracy_30_158: - 5.09765625 frac[30,158] + mem[30,158] = 0
 accuracy_196_162: - 5.09765625 frac[196,162] + mem[196,162] = 0
 accuracy_69_97: - 5.09765625 frac[69,97] + mem[69,97] = 0
 accuracy_198_186: - 5.09765625 frac[198,186] + mem[198,186] = 0
 accuracy_53_146: - 5.09765625 frac[53,146] + mem[53,146] = 0
 accuracy_197_60: - 5.09765625 frac[197,60] + mem[197,60] = 0
 accuracy_148_77: - 5.09765625 frac[148,77] + mem[148,77] = 0
 accuracy_173_195: - 5.09765625 frac[173,195] + mem[173,195] = 0
 accuracy_194_72: - 5.09765625 frac[194,72] + mem[194,72] = 0
 accuracy_16_130: - 5.09765625 frac[16,130] + mem[16,130] = 0
 accuracy_198_213: - 5.09765625 frac[198,213] + mem[198,213] = 0
 accuracy_197_205: - 5.09765625 frac[197,205] + mem[197,205] = 0
 accuracy_0_76: - 5.09765625 frac[0,76] + mem[0,76] = 0
 accuracy_111_70: - 5.09765625 frac[111,70] + mem[111,70] = 0
 accuracy_134_251: - 5.09765625 frac[134,251] + mem[134,251] = 0
 accuracy_192_30: - 5.09765625 frac[192,30] + mem[192,30] = 0
 accuracy_65_221: - 5.09765625 frac[65,221] + mem[65,221] = 0
 accuracy_140_214: - 5.09765625 frac[140,214] + mem[140,214] = 0
 accuracy_198_100: - 5.09765625 frac[198,100] + mem[198,100] = 0
 accuracy_185_49: - 5.09765625 frac[185,49] + mem[185,49] = 0
 accuracy_192_141: - 5.09765625 frac[192,141] + mem[192,141] = 0
 accuracy_194_161: - 5.09765625 frac[194,161] + mem[194,161] = 0
 accuracy_157_43: - 5.09765625 frac[157,43] + mem[157,43] = 0
 accuracy_196_26: - 5.09765625 frac[196,26] + mem[196,26] = 0
 accuracy_141_252: - 5.09765625 frac[141,252] + mem[141,252] = 0
 accuracy_54_104: - 5.09765625 frac[54,104] + mem[54,104] = 0
 accuracy_182_93: - 5.09765625 frac[182,93] + mem[182,93] = 0
 accuracy_148_229: - 5.09765625 frac[148,229] + mem[148,229] = 0
 accuracy_22_65: - 5.09765625 frac[22,65] + mem[22,65] = 0
 accuracy_5_247: - 5.09765625 frac[5,247] + mem[5,247] = 0
 accuracy_193_162: - 5.09765625 frac[193,162] + mem[193,162] = 0
 accuracy_194_208: - 5.09765625 frac[194,208] + mem[194,208] = 0
 accuracy_195_194: - 5.09765625 frac[195,194] + mem[195,194] = 0
 accuracy_28_229: - 5.09765625 frac[28,229] + mem[28,229] = 0
 accuracy_196_169: - 5.09765625 frac[196,169] + mem[196,169] = 0
 accuracy_197_37: - 5.09765625 frac[197,37] + mem[197,37] = 0
 accuracy_198_189: - 5.09765625 frac[198,189] + mem[198,189] = 0
 accuracy_126_149: - 5.09765625 frac[126,149] + mem[126,149] = 0
 accuracy_184_102: - 5.09765625 frac[184,102] + mem[184,102] = 0
 accuracy_57_216: - 5.09765625 frac[57,216] + mem[57,216] = 0
 accuracy_76_12: - 5.09765625 frac[76,12] + mem[76,12] = 0
 accuracy_4_142: - 5.09765625 frac[4,142] + mem[4,142] = 0
 accuracy_134_99: - 5.09765625 frac[134,99] + mem[134,99] = 0
 accuracy_194_67: - 5.09765625 frac[194,67] + mem[194,67] = 0
 accuracy_195_83: - 5.09765625 frac[195,83] + mem[195,83] = 0
 accuracy_141_19: - 5.09765625 frac[141,19] + mem[141,19] = 0
 accuracy_197_182: - 5.09765625 frac[197,182] + mem[197,182] = 0
 accuracy_198_204: - 5.09765625 frac[198,204] + mem[198,204] = 0
 accuracy_127_254: - 5.09765625 frac[127,254] + mem[127,254] = 0
 accuracy_36_244: - 5.09765625 frac[36,244] + mem[36,244] = 0
 accuracy_52_37: - 5.09765625 frac[52,37] + mem[52,37] = 0
 accuracy_16_249: - 5.09765625 frac[16,249] + mem[16,249] = 0
 accuracy_60_165: - 5.09765625 frac[60,165] + mem[60,165] = 0
 accuracy_192_21: - 5.09765625 frac[192,21] + mem[192,21] = 0
 accuracy_65_214: - 5.09765625 frac[65,214] + mem[65,214] = 0
 accuracy_120_61: - 5.09765625 frac[120,61] + mem[120,61] = 0
 accuracy_198_95: - 5.09765625 frac[198,95] + mem[198,95] = 0
 accuracy_127_79: - 5.09765625 frac[127,79] + mem[127,79] = 0
 accuracy_147_133: - 5.09765625 frac[147,133] + mem[147,133] = 0
 accuracy_168_30: - 5.09765625 frac[168,30] + mem[168,30] = 0
 accuracy_21_233: - 5.09765625 frac[21,233] + mem[21,233] = 0
 accuracy_169_144: - 5.09765625 frac[169,144] + mem[169,144] = 0
 accuracy_171_123: - 5.09765625 frac[171,123] + mem[171,123] = 0
 accuracy_192_132: - 5.09765625 frac[192,132] + mem[192,132] = 0
 accuracy_193_10: - 5.09765625 frac[193,10] + mem[193,10] = 0
 accuracy_194_184: - 5.09765625 frac[194,184] + mem[194,184] = 0
 accuracy_196_1: - 5.09765625 frac[196,1] + mem[196,1] = 0
 accuracy_71_41: - 5.09765625 frac[71,41] + mem[71,41] = 0
 accuracy_78_101: - 5.09765625 frac[78,101] + mem[78,101] = 0
 accuracy_116_216: - 5.09765625 frac[116,216] + mem[116,216] = 0
 accuracy_80_63: - 5.09765625 frac[80,63] + mem[80,63] = 0
 accuracy_188_176: - 5.09765625 frac[188,176] + mem[188,176] = 0
 accuracy_137_8: - 5.09765625 frac[137,8] + mem[137,8] = 0
 accuracy_193_187: - 5.09765625 frac[193,187] + mem[193,187] = 0
 accuracy_194_43: - 5.09765625 frac[194,43] + mem[194,43] = 0
 accuracy_12_102: - 5.09765625 frac[12,102] + mem[12,102] = 0
 accuracy_195_203: - 5.09765625 frac[195,203] + mem[195,203] = 0
 accuracy_196_144: - 5.09765625 frac[196,144] + mem[196,144] = 0
 accuracy_197_46: - 5.09765625 frac[197,46] + mem[197,46] = 0
 accuracy_52_189: - 5.09765625 frac[52,189] + mem[52,189] = 0
 accuracy_198_180: - 5.09765625 frac[198,180] + mem[198,180] = 0
 accuracy_54_18: - 5.09765625 frac[54,18] + mem[54,18] = 0
 accuracy_128_144: - 5.09765625 frac[128,144] + mem[128,144] = 0
 accuracy_177_161: - 5.09765625 frac[177,161] + mem[177,161] = 0
 accuracy_61_100: - 5.09765625 frac[61,100] + mem[61,100] = 0
 accuracy_64_136: - 5.09765625 frac[64,136] + mem[64,136] = 0
 accuracy_118_127: - 5.09765625 frac[118,127] + mem[118,127] = 0
 accuracy_9_70: - 5.09765625 frac[9,70] + mem[9,70] = 0
 accuracy_194_90: - 5.09765625 frac[194,90] + mem[194,90] = 0
 accuracy_195_88: - 5.09765625 frac[195,88] + mem[195,88] = 0
 accuracy_141_12: - 5.09765625 frac[141,12] + mem[141,12] = 0
 accuracy_197_191: - 5.09765625 frac[197,191] + mem[197,191] = 0
 accuracy_198_199: - 5.09765625 frac[198,199] + mem[198,199] = 0
 accuracy_128_15: - 5.09765625 frac[128,15] + mem[128,15] = 0
 accuracy_96_78: - 5.09765625 frac[96,78] + mem[96,78] = 0
 accuracy_114_152: - 5.09765625 frac[114,152] + mem[114,152] = 0
 accuracy_132_132: - 5.09765625 frac[132,132] + mem[132,132] = 0
 accuracy_192_108: - 5.09765625 frac[192,108] + mem[192,108] = 0
 accuracy_10_47: - 5.09765625 frac[10,47] + mem[10,47] = 0
 accuracy_156_2: - 5.09765625 frac[156,2] + mem[156,2] = 0
 accuracy_198_86: - 5.09765625 frac[198,86] + mem[198,86] = 0
 accuracy_2_229: - 5.09765625 frac[2,229] + mem[2,229] = 0
 accuracy_20_132: - 5.09765625 frac[20,132] + mem[20,132] = 0
 accuracy_149_45: - 5.09765625 frac[149,45] + mem[149,45] = 0
 accuracy_191_125: - 5.09765625 frac[191,125] + mem[191,125] = 0
 accuracy_194_179: - 5.09765625 frac[194,179] + mem[194,179] = 0
 accuracy_176_69: - 5.09765625 frac[176,69] + mem[176,69] = 0
 accuracy_139_75: - 5.09765625 frac[139,75] + mem[139,75] = 0
 accuracy_196_8: - 5.09765625 frac[196,8] + mem[196,8] = 0
 accuracy_141_238: - 5.09765625 frac[141,238] + mem[141,238] = 0
 accuracy_53_124: - 5.09765625 frac[53,124] + mem[53,124] = 0
 accuracy_143_254: - 5.09765625 frac[143,254] + mem[143,254] = 0
 accuracy_150_4: - 5.09765625 frac[150,4] + mem[150,4] = 0
 accuracy_153_224: - 5.09765625 frac[153,224] + mem[153,224] = 0
 accuracy_44_68: - 5.09765625 frac[44,68] + mem[44,68] = 0
 accuracy_154_75: - 5.09765625 frac[154,75] + mem[154,75] = 0
 accuracy_155_75: - 5.09765625 frac[155,75] + mem[155,75] = 0
 accuracy_193_188: - 5.09765625 frac[193,188] + mem[193,188] = 0
 accuracy_194_34: - 5.09765625 frac[194,34] + mem[194,34] = 0
 accuracy_195_176: - 5.09765625 frac[195,176] + mem[195,176] = 0
 accuracy_196_159: - 5.09765625 frac[196,159] + mem[196,159] = 0
 accuracy_197_23: - 5.09765625 frac[197,23] + mem[197,23] = 0
 accuracy_198_175: - 5.09765625 frac[198,175] + mem[198,175] = 0
 accuracy_125_47: - 5.09765625 frac[125,47] + mem[125,47] = 0
 accuracy_15_48: - 5.09765625 frac[15,48] + mem[15,48] = 0
 accuracy_128_151: - 5.09765625 frac[128,151] + mem[128,151] = 0
 accuracy_179_65: - 5.09765625 frac[179,65] + mem[179,65] = 0
 accuracy_148_90: - 5.09765625 frac[148,90] + mem[148,90] = 0
 accuracy_6_136: - 5.09765625 frac[6,136] + mem[6,136] = 0
 accuracy_82_217: - 5.09765625 frac[82,217] + mem[82,217] = 0
 accuracy_138_16: - 5.09765625 frac[138,16] + mem[138,16] = 0
 accuracy_194_85: - 5.09765625 frac[194,85] + mem[194,85] = 0
 accuracy_195_65: - 5.09765625 frac[195,65] + mem[195,65] = 0
 accuracy_87_87: - 5.09765625 frac[87,87] + mem[87,87] = 0
 accuracy_197_184: - 5.09765625 frac[197,184] + mem[197,184] = 0
 accuracy_198_62: - 5.09765625 frac[198,62] + mem[198,62] = 0
 accuracy_192_99: - 5.09765625 frac[192,99] + mem[192,99] = 0
 accuracy_193_107: - 5.09765625 frac[193,107] + mem[193,107] = 0
 accuracy_68_171: - 5.09765625 frac[68,171] + mem[68,171] = 0
 accuracy_106_98: - 5.09765625 frac[106,98] + mem[106,98] = 0
 accuracy_198_73: - 5.09765625 frac[198,73] + mem[198,73] = 0
 accuracy_109_215: - 5.09765625 frac[109,215] + mem[109,215] = 0
 accuracy_182_183: - 5.09765625 frac[182,183] + mem[182,183] = 0
 accuracy_43_34: - 5.09765625 frac[43,34] + mem[43,34] = 0
 accuracy_192_146: - 5.09765625 frac[192,146] + mem[192,146] = 0
 accuracy_193_4: - 5.09765625 frac[193,4] + mem[193,4] = 0
 accuracy_196_119: - 5.09765625 frac[196,119] + mem[196,119] = 0
 accuracy_73_177: - 5.09765625 frac[73,177] + mem[73,177] = 0
 accuracy_128_255: - 5.09765625 frac[128,255] + mem[128,255] = 0
 accuracy_168_155: - 5.09765625 frac[168,155] + mem[168,155] = 0
 accuracy_167_253: - 5.09765625 frac[167,253] + mem[167,253] = 0
 accuracy_150_31: - 5.09765625 frac[150,31] + mem[150,31] = 0
 accuracy_154_82: - 5.09765625 frac[154,82] + mem[154,82] = 0
 accuracy_121_157: - 5.09765625 frac[121,157] + mem[121,157] = 0
 accuracy_194_61: - 5.09765625 frac[194,61] + mem[194,61] = 0
 accuracy_195_185: - 5.09765625 frac[195,185] + mem[195,185] = 0
 accuracy_174_145: - 5.09765625 frac[174,145] + mem[174,145] = 0
 accuracy_196_134: - 5.09765625 frac[196,134] + mem[196,134] = 0
 accuracy_197_16: - 5.09765625 frac[197,16] + mem[197,16] = 0
 accuracy_198_166: - 5.09765625 frac[198,166] + mem[198,166] = 0
 accuracy_35_205: - 5.09765625 frac[35,205] + mem[35,205] = 0
 accuracy_56_70: - 5.09765625 frac[56,70] + mem[56,70] = 0
 accuracy_128_110: - 5.09765625 frac[128,110] + mem[128,110] = 0
 accuracy_132_251: - 5.09765625 frac[132,251] + mem[132,251] = 0
 accuracy_98_10: - 5.09765625 frac[98,10] + mem[98,10] = 0
 accuracy_133_78: - 5.09765625 frac[133,78] + mem[133,78] = 0
 accuracy_151_244: - 5.09765625 frac[151,244] + mem[151,244] = 0
 accuracy_195_70: - 5.09765625 frac[195,70] + mem[195,70] = 0
 accuracy_123_126: - 5.09765625 frac[123,126] + mem[123,126] = 0
 accuracy_103_170: - 5.09765625 frac[103,170] + mem[103,170] = 0
 accuracy_197_161: - 5.09765625 frac[197,161] + mem[197,161] = 0
 accuracy_15_174: - 5.09765625 frac[15,174] + mem[15,174] = 0
 accuracy_198_49: - 5.09765625 frac[198,49] + mem[198,49] = 0
 accuracy_183_114: - 5.09765625 frac[183,114] + mem[183,114] = 0
 accuracy_192_122: - 5.09765625 frac[192,122] + mem[192,122] = 0
 accuracy_193_108: - 5.09765625 frac[193,108] + mem[193,108] = 0
 accuracy_178_5: - 5.09765625 frac[178,5] + mem[178,5] = 0
 accuracy_198_64: - 5.09765625 frac[198,64] + mem[198,64] = 0
 accuracy_92_8: - 5.09765625 frac[92,8] + mem[92,8] = 0
 accuracy_22_242: - 5.09765625 frac[22,242] + mem[22,242] = 0
 accuracy_8_7: - 5.09765625 frac[8,7] + mem[8,7] = 0
 accuracy_192_233: - 5.09765625 frac[192,233] + mem[192,233] = 0
 accuracy_193_29: - 5.09765625 frac[193,29] + mem[193,29] = 0
 accuracy_194_133: - 5.09765625 frac[194,133] + mem[194,133] = 0
 accuracy_49_251: - 5.09765625 frac[49,251] + mem[49,251] = 0
 accuracy_196_126: - 5.09765625 frac[196,126] + mem[196,126] = 0
 accuracy_90_82: - 5.09765625 frac[90,82] + mem[90,82] = 0
 accuracy_70_142: - 5.09765625 frac[70,142] + mem[70,142] = 0
 accuracy_1_120: - 5.09765625 frac[1,120] + mem[1,120] = 0
 accuracy_58_123: - 5.09765625 frac[58,123] + mem[58,123] = 0
 accuracy_115_70: - 5.09765625 frac[115,70] + mem[115,70] = 0
 accuracy_137_35: - 5.09765625 frac[137,35] + mem[137,35] = 0
 accuracy_193_142: - 5.09765625 frac[193,142] + mem[193,142] = 0
 accuracy_104_170: - 5.09765625 frac[104,170] + mem[104,170] = 0
 accuracy_195_190: - 5.09765625 frac[195,190] + mem[195,190] = 0
 accuracy_103_82: - 5.09765625 frac[103,82] + mem[103,82] = 0
 accuracy_196_141: - 5.09765625 frac[196,141] + mem[196,141] = 0
 accuracy_197_25: - 5.09765625 frac[197,25] + mem[197,25] = 0
 accuracy_198_153: - 5.09765625 frac[198,153] + mem[198,153] = 0
 accuracy_17_88: - 5.09765625 frac[17,88] + mem[17,88] = 0
 accuracy_191_240: - 5.09765625 frac[191,240] + mem[191,240] = 0
 accuracy_129_220: - 5.09765625 frac[129,220] + mem[129,220] = 0
 accuracy_42_96: - 5.09765625 frac[42,96] + mem[42,96] = 0
 accuracy_150_129: - 5.09765625 frac[150,129] + mem[150,129] = 0
 accuracy_195_79: - 5.09765625 frac[195,79] + mem[195,79] = 0
 accuracy_88_19: - 5.09765625 frac[88,19] + mem[88,19] = 0
 accuracy_13_236: - 5.09765625 frac[13,236] + mem[13,236] = 0
 accuracy_197_170: - 5.09765625 frac[197,170] + mem[197,170] = 0
 accuracy_198_40: - 5.09765625 frac[198,40] + mem[198,40] = 0
 accuracy_43_131: - 5.09765625 frac[43,131] + mem[43,131] = 0
 accuracy_151_98: - 5.09765625 frac[151,98] + mem[151,98] = 0
 accuracy_192_113: - 5.09765625 frac[192,113] + mem[192,113] = 0
 accuracy_193_101: - 5.09765625 frac[193,101] + mem[193,101] = 0
 accuracy_110_169: - 5.09765625 frac[110,169] + mem[110,169] = 0
 accuracy_93_143: - 5.09765625 frac[93,143] + mem[93,143] = 0
 accuracy_7_46: - 5.09765625 frac[7,46] + mem[7,46] = 0
 accuracy_192_224: - 5.09765625 frac[192,224] + mem[192,224] = 0
 accuracy_193_22: - 5.09765625 frac[193,22] + mem[193,22] = 0
 accuracy_194_156: - 5.09765625 frac[194,156] + mem[194,156] = 0
 accuracy_196_101: - 5.09765625 frac[196,101] + mem[196,101] = 0
 accuracy_197_113: - 5.09765625 frac[197,113] + mem[197,113] = 0
 accuracy_73_163: - 5.09765625 frac[73,163] + mem[73,163] = 0
 accuracy_128_205: - 5.09765625 frac[128,205] + mem[128,205] = 0
 accuracy_92_182: - 5.09765625 frac[92,182] + mem[92,182] = 0
 accuracy_75_214: - 5.09765625 frac[75,214] + mem[75,214] = 0
 accuracy_1_113: - 5.09765625 frac[1,113] + mem[1,113] = 0
 accuracy_8_157: - 5.09765625 frac[8,157] + mem[8,157] = 0
 accuracy_193_135: - 5.09765625 frac[193,135] + mem[193,135] = 0
 accuracy_194_15: - 5.09765625 frac[194,15] + mem[194,15] = 0
 accuracy_195_167: - 5.09765625 frac[195,167] + mem[195,167] = 0
 accuracy_196_244: - 5.09765625 frac[196,244] + mem[196,244] = 0
 accuracy_197_2: - 5.09765625 frac[197,2] + mem[197,2] = 0
 accuracy_17_81: - 5.09765625 frac[17,81] + mem[17,81] = 0
 accuracy_53_248: - 5.09765625 frac[53,248] + mem[53,248] = 0
 accuracy_198_144: - 5.09765625 frac[198,144] + mem[198,144] = 0
 accuracy_94_65: - 5.09765625 frac[94,65] + mem[94,65] = 0
 accuracy_130_221: - 5.09765625 frac[130,221] + mem[130,221] = 0
 accuracy_5_101: - 5.09765625 frac[5,101] + mem[5,101] = 0
 accuracy_100_165: - 5.09765625 frac[100,165] + mem[100,165] = 0
 accuracy_155_207: - 5.09765625 frac[155,207] + mem[155,207] = 0
 accuracy_66_117: - 5.09765625 frac[66,117] + mem[66,117] = 0
 accuracy_195_52: - 5.09765625 frac[195,52] + mem[195,52] = 0
 accuracy_159_90: - 5.09765625 frac[159,90] + mem[159,90] = 0
 accuracy_197_147: - 5.09765625 frac[197,147] + mem[197,147] = 0
 accuracy_63_245: - 5.09765625 frac[63,245] + mem[63,245] = 0
 accuracy_28_57: - 5.09765625 frac[28,57] + mem[28,57] = 0
 accuracy_138_156: - 5.09765625 frac[138,156] + mem[138,156] = 0
 accuracy_192_72: - 5.09765625 frac[192,72] + mem[192,72] = 0
 accuracy_193_126: - 5.09765625 frac[193,126] + mem[193,126] = 0
 accuracy_159_235: - 5.09765625 frac[159,235] + mem[159,235] = 0
 accuracy_0_213: - 5.09765625 frac[0,213] + mem[0,213] = 0
 accuracy_57_36: - 5.09765625 frac[57,36] + mem[57,36] = 0
 accuracy_185_95: - 5.09765625 frac[185,95] + mem[185,95] = 0
 accuracy_187_242: - 5.09765625 frac[187,242] + mem[187,242] = 0
 accuracy_117_2: - 5.09765625 frac[117,2] + mem[117,2] = 0
 accuracy_192_231: - 5.09765625 frac[192,231] + mem[192,231] = 0
 accuracy_135_114: - 5.09765625 frac[135,114] + mem[135,114] = 0
 accuracy_194_151: - 5.09765625 frac[194,151] + mem[194,151] = 0
 accuracy_49_245: - 5.09765625 frac[49,245] + mem[49,245] = 0
 accuracy_193_239: - 5.09765625 frac[193,239] + mem[193,239] = 0
 accuracy_196_108: - 5.09765625 frac[196,108] + mem[196,108] = 0
 accuracy_197_122: - 5.09765625 frac[197,122] + mem[197,122] = 0
 accuracy_63_90: - 5.09765625 frac[63,90] + mem[63,90] = 0
 accuracy_4_193: - 5.09765625 frac[4,193] + mem[4,193] = 0
 accuracy_117_179: - 5.09765625 frac[117,179] + mem[117,179] = 0
 accuracy_83_127: - 5.09765625 frac[83,127] + mem[83,127] = 0
 accuracy_193_152: - 5.09765625 frac[193,152] + mem[193,152] = 0
 accuracy_194_6: - 5.09765625 frac[194,6] + mem[194,6] = 0
 accuracy_195_172: - 5.09765625 frac[195,172] + mem[195,172] = 0
 accuracy_196_243: - 5.09765625 frac[196,243] + mem[196,243] = 0
 accuracy_197_11: - 5.09765625 frac[197,11] + mem[197,11] = 0
 accuracy_179_93: - 5.09765625 frac[179,93] + mem[179,93] = 0
 accuracy_198_139: - 5.09765625 frac[198,139] + mem[198,139] = 0
 accuracy_56_91: - 5.09765625 frac[56,91] + mem[56,91] = 0
 accuracy_131_222: - 5.09765625 frac[131,222] + mem[131,222] = 0
 accuracy_172_132: - 5.09765625 frac[172,132] + mem[172,132] = 0
 accuracy_195_61: - 5.09765625 frac[195,61] + mem[195,61] = 0
 accuracy_197_156: - 5.09765625 frac[197,156] + mem[197,156] = 0
 accuracy_198_26: - 5.09765625 frac[198,26] + mem[198,26] = 0
 accuracy_181_242: - 5.09765625 frac[181,242] + mem[181,242] = 0
 accuracy_112_167: - 5.09765625 frac[112,167] + mem[112,167] = 0
 accuracy_111_113: - 5.09765625 frac[111,113] + mem[111,113] = 0
 accuracy_3_24: - 5.09765625 frac[3,24] + mem[3,24] = 0
 accuracy_131_111: - 5.09765625 frac[131,111] + mem[131,111] = 0
 accuracy_166_51: - 5.09765625 frac[166,51] + mem[166,51] = 0
 accuracy_168_89: - 5.09765625 frac[168,89] + mem[168,89] = 0
 accuracy_192_79: - 5.09765625 frac[192,79] + mem[192,79] = 0
 accuracy_193_119: - 5.09765625 frac[193,119] + mem[193,119] = 0
 accuracy_46_158: - 5.09765625 frac[46,158] + mem[46,158] = 0
 accuracy_50_101: - 5.09765625 frac[50,101] + mem[50,101] = 0
 accuracy_74_111: - 5.09765625 frac[74,111] + mem[74,111] = 0
 accuracy_75_119: - 5.09765625 frac[75,119] + mem[75,119] = 0
 accuracy_111_230: - 5.09765625 frac[111,230] + mem[111,230] = 0
 accuracy_192_254: - 5.09765625 frac[192,254] + mem[192,254] = 0
 accuracy_194_238: - 5.09765625 frac[194,238] + mem[194,238] = 0
 accuracy_196_107: - 5.09765625 frac[196,107] + mem[196,107] = 0
 accuracy_125_91: - 5.09765625 frac[125,91] + mem[125,91] = 0
 accuracy_197_99: - 5.09765625 frac[197,99] + mem[197,99] = 0
 accuracy_145_133: - 5.09765625 frac[145,133] + mem[145,133] = 0
 accuracy_128_219: - 5.09765625 frac[128,219] + mem[128,219] = 0
 accuracy_150_59: - 5.09765625 frac[150,59] + mem[150,59] = 0
 accuracy_189_84: - 5.09765625 frac[189,84] + mem[189,84] = 0
 accuracy_193_145: - 5.09765625 frac[193,145] + mem[193,145] = 0
 accuracy_194_1: - 5.09765625 frac[194,1] + mem[194,1] = 0
 accuracy_195_149: - 5.09765625 frac[195,149] + mem[195,149] = 0
 accuracy_174_173: - 5.09765625 frac[174,173] + mem[174,173] = 0
 accuracy_196_250: - 5.09765625 frac[196,250] + mem[196,250] = 0
 accuracy_197_244: - 5.09765625 frac[197,244] + mem[197,244] = 0
 accuracy_198_130: - 5.09765625 frac[198,130] + mem[198,130] = 0
 accuracy_33_133: - 5.09765625 frac[33,133] + mem[33,133] = 0
 accuracy_124_194: - 5.09765625 frac[124,194] + mem[124,194] = 0
 accuracy_20_72: - 5.09765625 frac[20,72] + mem[20,72] = 0
 accuracy_195_34: - 5.09765625 frac[195,34] + mem[195,34] = 0
 accuracy_13_23: - 5.09765625 frac[13,23] + mem[13,23] = 0
 accuracy_30_97: - 5.09765625 frac[30,97] + mem[30,97] = 0
 accuracy_197_133: - 5.09765625 frac[197,133] + mem[197,133] = 0
 accuracy_158_186: - 5.09765625 frac[158,186] + mem[158,186] = 0
 accuracy_198_29: - 5.09765625 frac[198,29] + mem[198,29] = 0
 accuracy_88_232: - 5.09765625 frac[88,232] + mem[88,232] = 0
 accuracy_55_107: - 5.09765625 frac[55,107] + mem[55,107] = 0
 accuracy_134_195: - 5.09765625 frac[134,195] + mem[134,195] = 0
 accuracy_192_70: - 5.09765625 frac[192,70] + mem[192,70] = 0
 accuracy_193_72: - 5.09765625 frac[193,72] + mem[193,72] = 0
 accuracy_14_78: - 5.09765625 frac[14,78] + mem[14,78] = 0
 accuracy_93_154: - 5.09765625 frac[93,154] + mem[93,154] = 0
 accuracy_75_108: - 5.09765625 frac[75,108] + mem[75,108] = 0
 accuracy_39_66: - 5.09765625 frac[39,66] + mem[39,66] = 0
 accuracy_43_7: - 5.09765625 frac[43,7] + mem[43,7] = 0
 accuracy_192_245: - 5.09765625 frac[192,245] + mem[192,245] = 0
 accuracy_193_249: - 5.09765625 frac[193,249] + mem[193,249] = 0
 accuracy_194_233: - 5.09765625 frac[194,233] + mem[194,233] = 0
 accuracy_28_190: - 5.09765625 frac[28,190] + mem[28,190] = 0
 accuracy_196_82: - 5.09765625 frac[196,82] + mem[196,82] = 0
 accuracy_69_145: - 5.09765625 frac[69,145] + mem[69,145] = 0
 accuracy_34_149: - 5.09765625 frac[34,149] + mem[34,149] = 0
 accuracy_142_178: - 5.09765625 frac[142,178] + mem[142,178] = 0
 accuracy_197_108: - 5.09765625 frac[197,108] + mem[197,108] = 0
 accuracy_182_5: - 5.09765625 frac[182,5] + mem[182,5] = 0
 accuracy_147_37: - 5.09765625 frac[147,37] + mem[147,37] = 0
 accuracy_41_61: - 5.09765625 frac[41,61] + mem[41,61] = 0
 accuracy_115_34: - 5.09765625 frac[115,34] + mem[115,34] = 0
 accuracy_194_24: - 5.09765625 frac[194,24] + mem[194,24] = 0
 accuracy_178_254: - 5.09765625 frac[178,254] + mem[178,254] = 0
 accuracy_196_225: - 5.09765625 frac[196,225] + mem[196,225] = 0
 accuracy_197_253: - 5.09765625 frac[197,253] + mem[197,253] = 0
 accuracy_17_68: - 5.09765625 frac[17,68] + mem[17,68] = 0
 accuracy_198_133: - 5.09765625 frac[198,133] + mem[198,133] = 0
 accuracy_36_189: - 5.09765625 frac[36,189] + mem[36,189] = 0
 accuracy_143_81: - 5.09765625 frac[143,81] + mem[143,81] = 0
 accuracy_24_18: - 5.09765625 frac[24,18] + mem[24,18] = 0
 accuracy_192_46: - 5.09765625 frac[192,46] + mem[192,46] = 0
 accuracy_137_168: - 5.09765625 frac[137,168] + mem[137,168] = 0
 accuracy_66_94: - 5.09765625 frac[66,94] + mem[66,94] = 0
 accuracy_139_188: - 5.09765625 frac[139,188] + mem[139,188] = 0
 accuracy_195_43: - 5.09765625 frac[195,43] + mem[195,43] = 0
 accuracy_88_239: - 5.09765625 frac[88,239] + mem[88,239] = 0
 accuracy_141_91: - 5.09765625 frac[141,91] + mem[141,91] = 0
 accuracy_197_142: - 5.09765625 frac[197,142] + mem[197,142] = 0
 accuracy_198_20: - 5.09765625 frac[198,20] + mem[198,20] = 0
 accuracy_125_182: - 5.09765625 frac[125,182] + mem[125,182] = 0
 accuracy_92_116: - 5.09765625 frac[92,116] + mem[92,116] = 0
 accuracy_184_141: - 5.09765625 frac[184,141] + mem[184,141] = 0
 accuracy_38_24: - 5.09765625 frac[38,24] + mem[38,24] = 0
 accuracy_62_146: - 5.09765625 frac[62,146] + mem[62,146] = 0
 accuracy_192_93: - 5.09765625 frac[192,93] + mem[192,93] = 0
 accuracy_193_65: - 5.09765625 frac[193,65] + mem[193,65] = 0
 accuracy_84_91: - 5.09765625 frac[84,91] + mem[84,91] = 0
 accuracy_73_206: - 5.09765625 frac[73,206] + mem[73,206] = 0
 accuracy_143_184: - 5.09765625 frac[143,184] + mem[143,184] = 0
 accuracy_183_220: - 5.09765625 frac[183,220] + mem[183,220] = 0
 accuracy_20_181: - 5.09765625 frac[20,181] + mem[20,181] = 0
 accuracy_167_6: - 5.09765625 frac[167,6] + mem[167,6] = 0
 accuracy_170_230: - 5.09765625 frac[170,230] + mem[170,230] = 0
 accuracy_3_151: - 5.09765625 frac[3,151] + mem[3,151] = 0
 accuracy_172_83: - 5.09765625 frac[172,83] + mem[172,83] = 0
 accuracy_192_204: - 5.09765625 frac[192,204] + mem[192,204] = 0
 accuracy_193_242: - 5.09765625 frac[193,242] + mem[193,242] = 0
 accuracy_194_224: - 5.09765625 frac[194,224] + mem[194,224] = 0
 accuracy_195_242: - 5.09765625 frac[195,242] + mem[195,242] = 0
 accuracy_100_123: - 5.09765625 frac[100,123] + mem[100,123] = 0
 accuracy_196_89: - 5.09765625 frac[196,89] + mem[196,89] = 0
 accuracy_197_85: - 5.09765625 frac[197,85] + mem[197,85] = 0
 accuracy_34_172: - 5.09765625 frac[34,172] + mem[34,172] = 0
 accuracy_147_42: - 5.09765625 frac[147,42] + mem[147,42] = 0
 accuracy_76_252: - 5.09765625 frac[76,252] + mem[76,252] = 0
 accuracy_77_202: - 5.09765625 frac[77,202] + mem[77,202] = 0
 accuracy_150_53: - 5.09765625 frac[150,53] + mem[150,53] = 0
 accuracy_154_184: - 5.09765625 frac[154,184] + mem[154,184] = 0
 accuracy_194_19: - 5.09765625 frac[194,19] + mem[194,19] = 0
 accuracy_195_131: - 5.09765625 frac[195,131] + mem[195,131] = 0
 accuracy_158_21: - 5.09765625 frac[158,21] + mem[158,21] = 0
 accuracy_196_232: - 5.09765625 frac[196,232] + mem[196,232] = 0
 accuracy_197_230: - 5.09765625 frac[197,230] + mem[197,230] = 0
 accuracy_198_252: - 5.09765625 frac[198,252] + mem[198,252] = 0
 accuracy_128_88: - 5.09765625 frac[128,88] + mem[128,88] = 0
 accuracy_136_136: - 5.09765625 frac[136,136] + mem[136,136] = 0
 accuracy_192_37: - 5.09765625 frac[192,37] + mem[192,37] = 0
 accuracy_83_131: - 5.09765625 frac[83,131] + mem[83,131] = 0
 accuracy_84_243: - 5.09765625 frac[84,243] + mem[84,243] = 0
 accuracy_195_16: - 5.09765625 frac[195,16] + mem[195,16] = 0
 accuracy_30_83: - 5.09765625 frac[30,83] + mem[30,83] = 0
 accuracy_198_15: - 5.09765625 frac[198,15] + mem[198,15] = 0
 accuracy_40_121: - 5.09765625 frac[40,121] + mem[40,121] = 0
 accuracy_64_111: - 5.09765625 frac[64,111] + mem[64,111] = 0
 accuracy_192_84: - 5.09765625 frac[192,84] + mem[192,84] = 0
 accuracy_193_90: - 5.09765625 frac[193,90] + mem[193,90] = 0
 accuracy_69_242: - 5.09765625 frac[69,242] + mem[69,242] = 0
 accuracy_87_247: - 5.09765625 frac[87,247] + mem[87,247] = 0
 accuracy_196_49: - 5.09765625 frac[196,49] + mem[196,49] = 0
 accuracy_33_78: - 5.09765625 frac[33,78] + mem[33,78] = 0
 accuracy_127_12: - 5.09765625 frac[127,12] + mem[127,12] = 0
 accuracy_23_193: - 5.09765625 frac[23,193] + mem[23,193] = 0
 accuracy_192_195: - 5.09765625 frac[192,195] + mem[192,195] = 0
 accuracy_46_34: - 5.09765625 frac[46,34] + mem[46,34] = 0
 accuracy_193_203: - 5.09765625 frac[193,203] + mem[193,203] = 0
 accuracy_194_251: - 5.09765625 frac[194,251] + mem[194,251] = 0
 accuracy_195_251: - 5.09765625 frac[195,251] + mem[195,251] = 0
 accuracy_196_64: - 5.09765625 frac[196,64] + mem[196,64] = 0
 accuracy_197_94: - 5.09765625 frac[197,94] + mem[197,94] = 0
 accuracy_2_92: - 5.09765625 frac[2,92] + mem[2,92] = 0
 accuracy_58_33: - 5.09765625 frac[58,33] + mem[58,33] = 0
 accuracy_147_51: - 5.09765625 frac[147,51] + mem[147,51] = 0
 accuracy_170_112: - 5.09765625 frac[170,112] + mem[170,112] = 0
 accuracy_122_66: - 5.09765625 frac[122,66] + mem[122,66] = 0
 accuracy_195_136: - 5.09765625 frac[195,136] + mem[195,136] = 0
 accuracy_194_106: - 5.09765625 frac[194,106] + mem[194,106] = 0
 accuracy_196_215: - 5.09765625 frac[196,215] + mem[196,215] = 0
 accuracy_197_239: - 5.09765625 frac[197,239] + mem[197,239] = 0
 accuracy_198_247: - 5.09765625 frac[198,247] + mem[198,247] = 0
 accuracy_33_168: - 5.09765625 frac[33,168] + mem[33,168] = 0
 accuracy_111_36: - 5.09765625 frac[111,36] + mem[111,36] = 0
 accuracy_170_3: - 5.09765625 frac[170,3] + mem[170,3] = 0
 accuracy_120_20: - 5.09765625 frac[120,20] + mem[120,20] = 0
 accuracy_45_43: - 5.09765625 frac[45,43] + mem[45,43] = 0
 accuracy_192_60: - 5.09765625 frac[192,60] + mem[192,60] = 0
 accuracy_195_25: - 5.09765625 frac[195,25] + mem[195,25] = 0
 accuracy_103_245: - 5.09765625 frac[103,245] + mem[103,245] = 0
 accuracy_86_151: - 5.09765625 frac[86,151] + mem[86,151] = 0
 accuracy_12_180: - 5.09765625 frac[12,180] + mem[12,180] = 0
 accuracy_198_6: - 5.09765625 frac[198,6] + mem[198,6] = 0
 accuracy_37_140: - 5.09765625 frac[37,140] + mem[37,140] = 0
 accuracy_113_203: - 5.09765625 frac[113,203] + mem[113,203] = 0
 accuracy_168_53: - 5.09765625 frac[168,53] + mem[168,53] = 0
 accuracy_99_8: - 5.09765625 frac[99,8] + mem[99,8] = 0
 accuracy_193_83: - 5.09765625 frac[193,83] + mem[193,83] = 0
 accuracy_196_56: - 5.09765625 frac[196,56] + mem[196,56] = 0
 accuracy_107_56: - 5.09765625 frac[107,56] + mem[107,56] = 0
 accuracy_124_0: - 5.09765625 frac[124,0] + mem[124,0] = 0
 accuracy_0_136: - 5.09765625 frac[0,136] + mem[0,136] = 0
 accuracy_57_73: - 5.09765625 frac[57,73] + mem[57,73] = 0
 accuracy_2_164: - 5.09765625 frac[2,164] + mem[2,164] = 0
 accuracy_3_133: - 5.09765625 frac[3,133] + mem[3,133] = 0
 accuracy_95_92: - 5.09765625 frac[95,92] + mem[95,92] = 0
 accuracy_100_105: - 5.09765625 frac[100,105] + mem[100,105] = 0
 accuracy_192_218: - 5.09765625 frac[192,218] + mem[192,218] = 0
 accuracy_46_37: - 5.09765625 frac[46,37] + mem[46,37] = 0
 accuracy_47_153: - 5.09765625 frac[47,153] + mem[47,153] = 0
 accuracy_195_224: - 5.09765625 frac[195,224] + mem[195,224] = 0
 accuracy_194_242: - 5.09765625 frac[194,242] + mem[194,242] = 0
 accuracy_196_79: - 5.09765625 frac[196,79] + mem[196,79] = 0
 accuracy_197_71: - 5.09765625 frac[197,71] + mem[197,71] = 0
 accuracy_72_117: - 5.09765625 frac[72,117] + mem[72,117] = 0
 accuracy_176_4: - 5.09765625 frac[176,4] + mem[176,4] = 0
 accuracy_193_204: - 5.09765625 frac[193,204] + mem[193,204] = 0
 accuracy_55_169: - 5.09765625 frac[55,169] + mem[55,169] = 0
 accuracy_54_37: - 5.09765625 frac[54,37] + mem[54,37] = 0
 accuracy_128_167: - 5.09765625 frac[128,167] + mem[128,167] = 0
 accuracy_41_16: - 5.09765625 frac[41,16] + mem[41,16] = 0
 accuracy_8_167: - 5.09765625 frac[8,167] + mem[8,167] = 0
 accuracy_190_126: - 5.09765625 frac[190,126] + mem[190,126] = 0
 accuracy_194_101: - 5.09765625 frac[194,101] + mem[194,101] = 0
 accuracy_195_113: - 5.09765625 frac[195,113] + mem[195,113] = 0
 accuracy_13_170: - 5.09765625 frac[13,170] + mem[13,170] = 0
 accuracy_196_222: - 5.09765625 frac[196,222] + mem[196,222] = 0
 accuracy_197_232: - 5.09765625 frac[197,232] + mem[197,232] = 0
 accuracy_113_115: - 5.09765625 frac[113,115] + mem[113,115] = 0
 accuracy_58_155: - 5.09765625 frac[58,155] + mem[58,155] = 0
 accuracy_187_78: - 5.09765625 frac[187,78] + mem[187,78] = 0
 accuracy_78_222: - 5.09765625 frac[78,222] + mem[78,222] = 0
 accuracy_192_51: - 5.09765625 frac[192,51] + mem[192,51] = 0
 accuracy_102_95: - 5.09765625 frac[102,95] + mem[102,95] = 0
 accuracy_157_213: - 5.09765625 frac[157,213] + mem[157,213] = 0
 accuracy_195_30: - 5.09765625 frac[195,30] + mem[195,30] = 0
 accuracy_14_187: - 5.09765625 frac[14,187] + mem[14,187] = 0
 accuracy_180_167: - 5.09765625 frac[180,167] + mem[180,167] = 0
 accuracy_198_121: - 5.09765625 frac[198,121] + mem[198,121] = 0
 accuracy_95_52: - 5.09765625 frac[95,52] + mem[95,52] = 0
 accuracy_62_135: - 5.09765625 frac[62,135] + mem[62,135] = 0
 accuracy_192_162: - 5.09765625 frac[192,162] + mem[192,162] = 0
 accuracy_193_84: - 5.09765625 frac[193,84] + mem[193,84] = 0
 accuracy_32_86: - 5.09765625 frac[32,86] + mem[32,86] = 0
 accuracy_196_39: - 5.09765625 frac[196,39] + mem[196,39] = 0
 accuracy_89_107: - 5.09765625 frac[89,107] + mem[89,107] = 0
 accuracy_45_198: - 5.09765625 frac[45,198] + mem[45,198] = 0
 accuracy_193_197: - 5.09765625 frac[193,197] + mem[193,197] = 0
 accuracy_194_205: - 5.09765625 frac[194,205] + mem[194,205] = 0
 accuracy_177_131: - 5.09765625 frac[177,131] + mem[177,131] = 0
 accuracy_140_17: - 5.09765625 frac[140,17] + mem[140,17] = 0
 accuracy_196_182: - 5.09765625 frac[196,182] + mem[196,182] = 0
 accuracy_52_211: - 5.09765625 frac[52,211] + mem[52,211] = 0
 accuracy_142_142: - 5.09765625 frac[142,142] + mem[142,142] = 0
 accuracy_53_134: - 5.09765625 frac[53,134] + mem[53,134] = 0
 accuracy_197_64: - 5.09765625 frac[197,64] + mem[197,64] = 0
 accuracy_79_44: - 5.09765625 frac[79,44] + mem[79,44] = 0
 accuracy_100_231: - 5.09765625 frac[100,231] + mem[100,231] = 0
 accuracy_173_215: - 5.09765625 frac[173,215] + mem[173,215] = 0
 accuracy_194_124: - 5.09765625 frac[194,124] + mem[194,124] = 0
 accuracy_122_84: - 5.09765625 frac[122,84] + mem[122,84] = 0
 accuracy_196_197: - 5.09765625 frac[196,197] + mem[196,197] = 0
 accuracy_197_209: - 5.09765625 frac[197,209] + mem[197,209] = 0
 accuracy_17_32: - 5.09765625 frac[17,32] + mem[17,32] = 0
 accuracy_15_126: - 5.09765625 frac[15,126] + mem[15,126] = 0
 accuracy_198_225: - 5.09765625 frac[198,225] + mem[198,225] = 0
 accuracy_25_96: - 5.09765625 frac[25,96] + mem[25,96] = 0
 accuracy_6_66: - 5.09765625 frac[6,66] + mem[6,66] = 0
 accuracy_133_143: - 5.09765625 frac[133,143] + mem[133,143] = 0
 accuracy_192_10: - 5.09765625 frac[192,10] + mem[192,10] = 0
 accuracy_155_190: - 5.09765625 frac[155,190] + mem[155,190] = 0
 accuracy_139_216: - 5.09765625 frac[139,216] + mem[139,216] = 0
 accuracy_195_7: - 5.09765625 frac[195,7] + mem[195,7] = 0
 accuracy_160_35: - 5.09765625 frac[160,35] + mem[160,35] = 0
 accuracy_157_174: - 5.09765625 frac[157,174] + mem[157,174] = 0
 accuracy_198_112: - 5.09765625 frac[198,112] + mem[198,112] = 0
 accuracy_163_46: - 5.09765625 frac[163,46] + mem[163,46] = 0
 accuracy_1_159: - 5.09765625 frac[1,159] + mem[1,159] = 0
 accuracy_114_205: - 5.09765625 frac[114,205] + mem[114,205] = 0
 accuracy_167_85: - 5.09765625 frac[167,85] + mem[167,85] = 0
 accuracy_192_185: - 5.09765625 frac[192,185] + mem[192,185] = 0
 accuracy_193_45: - 5.09765625 frac[193,45] + mem[193,45] = 0
 accuracy_66_213: - 5.09765625 frac[66,213] + mem[66,213] = 0
 accuracy_121_5: - 5.09765625 frac[121,5] + mem[121,5] = 0
 accuracy_196_46: - 5.09765625 frac[196,46] + mem[196,46] = 0
 accuracy_71_68: - 5.09765625 frac[71,68] + mem[71,68] = 0
 accuracy_98_226: - 5.09765625 frac[98,226] + mem[98,226] = 0
 accuracy_100_31: - 5.09765625 frac[100,31] + mem[100,31] = 0
 accuracy_118_225: - 5.09765625 frac[118,225] + mem[118,225] = 0
 accuracy_193_222: - 5.09765625 frac[193,222] + mem[193,222] = 0
 accuracy_29_5: - 5.09765625 frac[29,5] + mem[29,5] = 0
 accuracy_174_104: - 5.09765625 frac[174,104] + mem[174,104] = 0
 accuracy_194_196: - 5.09765625 frac[194,196] + mem[194,196] = 0
 accuracy_196_189: - 5.09765625 frac[196,189] + mem[196,189] = 0
 accuracy_197_73: - 5.09765625 frac[197,73] + mem[197,73] = 0
 accuracy_58_58: - 5.09765625 frac[58,58] + mem[58,58] = 0
 accuracy_150_209: - 5.09765625 frac[150,209] + mem[150,209] = 0
 accuracy_172_198: - 5.09765625 frac[172,198] + mem[172,198] = 0
 accuracy_104_233: - 5.09765625 frac[104,233] + mem[104,233] = 0
 accuracy_194_119: - 5.09765625 frac[194,119] + mem[194,119] = 0
 accuracy_123_71: - 5.09765625 frac[123,71] + mem[123,71] = 0
 accuracy_196_204: - 5.09765625 frac[196,204] + mem[196,204] = 0
 accuracy_124_244: - 5.09765625 frac[124,244] + mem[124,244] = 0
 accuracy_198_216: - 5.09765625 frac[198,216] + mem[198,216] = 0
 accuracy_162_108: - 5.09765625 frac[162,108] + mem[162,108] = 0
 accuracy_36_216: - 5.09765625 frac[36,216] + mem[36,216] = 0
 accuracy_132_161: - 5.09765625 frac[132,161] + mem[132,161] = 0
 accuracy_192_1: - 5.09765625 frac[192,1] + mem[192,1] = 0
 accuracy_195_12: - 5.09765625 frac[195,12] + mem[195,12] = 0
 accuracy_198_107: - 5.09765625 frac[198,107] + mem[198,107] = 0
 accuracy_115_238: - 5.09765625 frac[115,238] + mem[115,238] = 0
 accuracy_60_8: - 5.09765625 frac[60,8] + mem[60,8] = 0
 accuracy_192_176: - 5.09765625 frac[192,176] + mem[192,176] = 0
 accuracy_121_14: - 5.09765625 frac[121,14] + mem[121,14] = 0
 accuracy_193_38: - 5.09765625 frac[193,38] + mem[193,38] = 0
 accuracy_11_27: - 5.09765625 frac[11,27] + mem[11,27] = 0
 accuracy_194_172: - 5.09765625 frac[194,172] + mem[194,172] = 0
 accuracy_196_21: - 5.09765625 frac[196,21] + mem[196,21] = 0
 accuracy_141_201: - 5.09765625 frac[141,201] + mem[141,201] = 0
 accuracy_120_152: - 5.09765625 frac[120,152] + mem[120,152] = 0
 accuracy_18_75: - 5.09765625 frac[18,75] + mem[18,75] = 0
 accuracy_146_148: - 5.09765625 frac[146,148] + mem[146,148] = 0
 accuracy_168_185: - 5.09765625 frac[168,185] + mem[168,185] = 0
 accuracy_96_129: - 5.09765625 frac[96,129] + mem[96,129] = 0
 accuracy_23_229: - 5.09765625 frac[23,229] + mem[23,229] = 0
 accuracy_155_110: - 5.09765625 frac[155,110] + mem[155,110] = 0
 accuracy_193_215: - 5.09765625 frac[193,215] + mem[193,215] = 0
 accuracy_194_223: - 5.09765625 frac[194,223] + mem[194,223] = 0
 accuracy_195_215: - 5.09765625 frac[195,215] + mem[195,215] = 0
 accuracy_175_99: - 5.09765625 frac[175,99] + mem[175,99] = 0
 accuracy_196_164: - 5.09765625 frac[196,164] + mem[196,164] = 0
 accuracy_197_50: - 5.09765625 frac[197,50] + mem[197,50] = 0
 accuracy_48_100: - 5.09765625 frac[48,100] + mem[48,100] = 0
 accuracy_69_103: - 5.09765625 frac[69,103] + mem[69,103] = 0
 accuracy_173_184: - 5.09765625 frac[173,184] + mem[173,184] = 0
 accuracy_133_96: - 5.09765625 frac[133,96] + mem[133,96] = 0
 accuracy_62_94: - 5.09765625 frac[62,94] + mem[62,94] = 0
 accuracy_64_164: - 5.09765625 frac[64,164] + mem[64,164] = 0
 accuracy_66_5: - 5.09765625 frac[66,5] + mem[66,5] = 0
 accuracy_194_78: - 5.09765625 frac[194,78] + mem[194,78] = 0
 accuracy_195_100: - 5.09765625 frac[195,100] + mem[195,100] = 0
 accuracy_196_203: - 5.09765625 frac[196,203] + mem[196,203] = 0
 accuracy_197_195: - 5.09765625 frac[197,195] + mem[197,195] = 0
 accuracy_198_211: - 5.09765625 frac[198,211] + mem[198,211] = 0
 accuracy_55_37: - 5.09765625 frac[55,37] + mem[55,37] = 0
 accuracy_148_54: - 5.09765625 frac[148,54] + mem[148,54] = 0
 accuracy_23_28: - 5.09765625 frac[23,28] + mem[23,28] = 0
 accuracy_192_24: - 5.09765625 frac[192,24] + mem[192,24] = 0
 accuracy_119_8: - 5.09765625 frac[119,8] + mem[119,8] = 0
 accuracy_63_165: - 5.09765625 frac[63,165] + mem[63,165] = 0
 accuracy_155_140: - 5.09765625 frac[155,140] + mem[155,140] = 0
 accuracy_198_98: - 5.09765625 frac[198,98] + mem[198,98] = 0
 accuracy_74_84: - 5.09765625 frac[74,84] + mem[74,84] = 0
 accuracy_167_75: - 5.09765625 frac[167,75] + mem[167,75] = 0
 accuracy_188_204: - 5.09765625 frac[188,204] + mem[188,204] = 0
 accuracy_136_58: - 5.09765625 frac[136,58] + mem[136,58] = 0
 accuracy_192_183: - 5.09765625 frac[192,183] + mem[192,183] = 0
 accuracy_193_63: - 5.09765625 frac[193,63] + mem[193,63] = 0
 accuracy_194_167: - 5.09765625 frac[194,167] + mem[194,167] = 0
 accuracy_33_99: - 5.09765625 frac[33,99] + mem[33,99] = 0
 accuracy_196_28: - 5.09765625 frac[196,28] + mem[196,28] = 0
 accuracy_72_2: - 5.09765625 frac[72,2] + mem[72,2] = 0
 accuracy_144_42: - 5.09765625 frac[144,42] + mem[144,42] = 0
 accuracy_163_173: - 5.09765625 frac[163,173] + mem[163,173] = 0
 accuracy_58_93: - 5.09765625 frac[58,93] + mem[58,93] = 0
 accuracy_23_226: - 5.09765625 frac[23,226] + mem[23,226] = 0
 accuracy_43_86: - 5.09765625 frac[43,86] + mem[43,86] = 0
 accuracy_8_244: - 5.09765625 frac[8,244] + mem[8,244] = 0
 accuracy_193_168: - 5.09765625 frac[193,168] + mem[193,168] = 0
 accuracy_194_214: - 5.09765625 frac[194,214] + mem[194,214] = 0
 accuracy_195_220: - 5.09765625 frac[195,220] + mem[195,220] = 0
 accuracy_196_163: - 5.09765625 frac[196,163] + mem[196,163] = 0
 accuracy_197_59: - 5.09765625 frac[197,59] + mem[197,59] = 0
 accuracy_198_187: - 5.09765625 frac[198,187] + mem[198,187] = 0
 accuracy_165_2: - 5.09765625 frac[165,2] + mem[165,2] = 0
 accuracy_172_180: - 5.09765625 frac[172,180] + mem[172,180] = 0
 accuracy_194_73: - 5.09765625 frac[194,73] + mem[194,73] = 0
 accuracy_176_191: - 5.09765625 frac[176,191] + mem[176,191] = 0
 accuracy_105_239: - 5.09765625 frac[105,239] + mem[105,239] = 0
 accuracy_87_75: - 5.09765625 frac[87,75] + mem[87,75] = 0
 accuracy_197_204: - 5.09765625 frac[197,204] + mem[197,204] = 0
 accuracy_15_85: - 5.09765625 frac[15,85] + mem[15,85] = 0
 accuracy_198_202: - 5.09765625 frac[198,202] + mem[198,202] = 0
 accuracy_43_161: - 5.09765625 frac[43,161] + mem[43,161] = 0
 accuracy_192_31: - 5.09765625 frac[192,31] + mem[192,31] = 0
 accuracy_83_173: - 5.09765625 frac[83,173] + mem[83,173] = 0
 accuracy_66_143: - 5.09765625 frac[66,143] + mem[66,143] = 0
 accuracy_157_185: - 5.09765625 frac[157,185] + mem[157,185] = 0
 accuracy_32_3: - 5.09765625 frac[32,3] + mem[32,3] = 0
 accuracy_159_0: - 5.09765625 frac[159,0] + mem[159,0] = 0
 accuracy_198_101: - 5.09765625 frac[198,101] + mem[198,101] = 0
 accuracy_0_252: - 5.09765625 frac[0,252] + mem[0,252] = 0
 accuracy_108_227: - 5.09765625 frac[108,227] + mem[108,227] = 0
 accuracy_39_181: - 5.09765625 frac[39,181] + mem[39,181] = 0
 accuracy_114_246: - 5.09765625 frac[114,246] + mem[114,246] = 0
 accuracy_117_91: - 5.09765625 frac[117,91] + mem[117,91] = 0
 accuracy_192_142: - 5.09765625 frac[192,142] + mem[192,142] = 0
 accuracy_193_48: - 5.09765625 frac[193,48] + mem[193,48] = 0
 accuracy_194_190: - 5.09765625 frac[194,190] + mem[194,190] = 0
 accuracy_47_213: - 5.09765625 frac[47,213] + mem[47,213] = 0
 accuracy_196_27: - 5.09765625 frac[196,27] + mem[196,27] = 0
 accuracy_75_52: - 5.09765625 frac[75,52] + mem[75,52] = 0
 accuracy_137_6: - 5.09765625 frac[137,6] + mem[137,6] = 0
 accuracy_193_161: - 5.09765625 frac[193,161] + mem[193,161] = 0
 accuracy_195_197: - 5.09765625 frac[195,197] + mem[195,197] = 0
 accuracy_196_170: - 5.09765625 frac[196,170] + mem[196,170] = 0
 accuracy_197_36: - 5.09765625 frac[197,36] + mem[197,36] = 0
 accuracy_165_11: - 5.09765625 frac[165,11] + mem[165,11] = 0
 accuracy_195_82: - 5.09765625 frac[195,82] + mem[195,82] = 0
 accuracy_197_181: - 5.09765625 frac[197,181] + mem[197,181] = 0
 accuracy_198_205: - 5.09765625 frac[198,205] + mem[198,205] = 0
 accuracy_55_59: - 5.09765625 frac[55,59] + mem[55,59] = 0
 accuracy_111_78: - 5.09765625 frac[111,78] + mem[111,78] = 0
 accuracy_96_72: - 5.09765625 frac[96,72] + mem[96,72] = 0
 accuracy_41_134: - 5.09765625 frac[41,134] + mem[41,134] = 0
 accuracy_82_119: - 5.09765625 frac[82,119] + mem[82,119] = 0
 accuracy_136_89: - 5.09765625 frac[136,89] + mem[136,89] = 0
 accuracy_70_11: - 5.09765625 frac[70,11] + mem[70,11] = 0
 accuracy_198_92: - 5.09765625 frac[198,92] + mem[198,92] = 0
 accuracy_163_202: - 5.09765625 frac[163,202] + mem[163,202] = 0
 accuracy_192_133: - 5.09765625 frac[192,133] + mem[192,133] = 0
 accuracy_193_9: - 5.09765625 frac[193,9] + mem[193,9] = 0
 accuracy_196_2: - 5.09765625 frac[196,2] + mem[196,2] = 0
 accuracy_56_202: - 5.09765625 frac[56,202] + mem[56,202] = 0
 accuracy_92_211: - 5.09765625 frac[92,211] + mem[92,211] = 0
 accuracy_74_169: - 5.09765625 frac[74,169] + mem[74,169] = 0
 accuracy_193_186: - 5.09765625 frac[193,186] + mem[193,186] = 0
 accuracy_195_202: - 5.09765625 frac[195,202] + mem[195,202] = 0
 accuracy_196_145: - 5.09765625 frac[196,145] + mem[196,145] = 0
 accuracy_197_45: - 5.09765625 frac[197,45] + mem[197,45] = 0
 accuracy_198_181: - 5.09765625 frac[198,181] + mem[198,181] = 0
 accuracy_130_246: - 5.09765625 frac[130,246] + mem[130,246] = 0
 accuracy_148_92: - 5.09765625 frac[148,92] + mem[148,92] = 0
 accuracy_60_206: - 5.09765625 frac[60,206] + mem[60,206] = 0
 accuracy_194_91: - 5.09765625 frac[194,91] + mem[194,91] = 0
 accuracy_29_154: - 5.09765625 frac[29,154] + mem[29,154] = 0
 accuracy_195_91: - 5.09765625 frac[195,91] + mem[195,91] = 0
 accuracy_197_190: - 5.09765625 frac[197,190] + mem[197,190] = 0
 accuracy_198_196: - 5.09765625 frac[198,196] + mem[198,196] = 0
 accuracy_56_40: - 5.09765625 frac[56,40] + mem[56,40] = 0
 accuracy_186_193: - 5.09765625 frac[186,193] + mem[186,193] = 0
 accuracy_43_159: - 5.09765625 frac[43,159] + mem[43,159] = 0
 accuracy_82_110: - 5.09765625 frac[82,110] + mem[82,110] = 0
 accuracy_192_109: - 5.09765625 frac[192,109] + mem[192,109] = 0
 accuracy_66_129: - 5.09765625 frac[66,129] + mem[66,129] = 0
 accuracy_198_87: - 5.09765625 frac[198,87] + mem[198,87] = 0
 accuracy_162_227: - 5.09765625 frac[162,227] + mem[162,227] = 0
 accuracy_0_202: - 5.09765625 frac[0,202] + mem[0,202] = 0
 accuracy_192_156: - 5.09765625 frac[192,156] + mem[192,156] = 0
 accuracy_193_2: - 5.09765625 frac[193,2] + mem[193,2] = 0
 accuracy_104_46: - 5.09765625 frac[104,46] + mem[104,46] = 0
 accuracy_50_214: - 5.09765625 frac[50,214] + mem[50,214] = 0
 accuracy_196_9: - 5.09765625 frac[196,9] + mem[196,9] = 0
 accuracy_147_250: - 5.09765625 frac[147,250] + mem[147,250] = 0
 accuracy_97_17: - 5.09765625 frac[97,17] + mem[97,17] = 0
 accuracy_190_188: - 5.09765625 frac[190,188] + mem[190,188] = 0
 accuracy_193_179: - 5.09765625 frac[193,179] + mem[193,179] = 0
 accuracy_194_35: - 5.09765625 frac[194,35] + mem[194,35] = 0
 accuracy_196_152: - 5.09765625 frac[196,152] + mem[196,152] = 0
 accuracy_197_22: - 5.09765625 frac[197,22] + mem[197,22] = 0
 accuracy_198_172: - 5.09765625 frac[198,172] + mem[198,172] = 0
 accuracy_124_160: - 5.09765625 frac[124,160] + mem[124,160] = 0
 accuracy_120_232: - 5.09765625 frac[120,232] + mem[120,232] = 0
 accuracy_75_179: - 5.09765625 frac[75,179] + mem[75,179] = 0
 accuracy_170_72: - 5.09765625 frac[170,72] + mem[170,72] = 0
 accuracy_136_248: - 5.09765625 frac[136,248] + mem[136,248] = 0
 accuracy_27_52: - 5.09765625 frac[27,52] + mem[27,52] = 0
 accuracy_194_82: - 5.09765625 frac[194,82] + mem[194,82] = 0
 accuracy_195_64: - 5.09765625 frac[195,64] + mem[195,64] = 0
 accuracy_107_226: - 5.09765625 frac[107,226] + mem[107,226] = 0
 accuracy_197_167: - 5.09765625 frac[197,167] + mem[197,167] = 0
 accuracy_41_240: - 5.09765625 frac[41,240] + mem[41,240] = 0
 accuracy_131_114: - 5.09765625 frac[131,114] + mem[131,114] = 0
 accuracy_45_115: - 5.09765625 frac[45,115] + mem[45,115] = 0
 accuracy_192_100: - 5.09765625 frac[192,100] + mem[192,100] = 0
 accuracy_102_108: - 5.09765625 frac[102,108] + mem[102,108] = 0
 accuracy_65_167: - 5.09765625 frac[65,167] + mem[65,167] = 0
 accuracy_140_252: - 5.09765625 frac[140,252] + mem[140,252] = 0
 accuracy_193_106: - 5.09765625 frac[193,106] + mem[193,106] = 0
 accuracy_198_78: - 5.09765625 frac[198,78] + mem[198,78] = 0
 accuracy_2_253: - 5.09765625 frac[2,253] + mem[2,253] = 0
 accuracy_192_147: - 5.09765625 frac[192,147] + mem[192,147] = 0
 accuracy_193_27: - 5.09765625 frac[193,27] + mem[193,27] = 0
 accuracy_135_6: - 5.09765625 frac[135,6] + mem[135,6] = 0
 accuracy_194_139: - 5.09765625 frac[194,139] + mem[194,139] = 0
 accuracy_140_83: - 5.09765625 frac[140,83] + mem[140,83] = 0
 accuracy_124_72: - 5.09765625 frac[124,72] + mem[124,72] = 0
 accuracy_141_230: - 5.09765625 frac[141,230] + mem[141,230] = 0
 accuracy_90_72: - 5.09765625 frac[90,72] + mem[90,72] = 0
 accuracy_196_112: - 5.09765625 frac[196,112] + mem[196,112] = 0
 accuracy_186_42: - 5.09765625 frac[186,42] + mem[186,42] = 0
 accuracy_96_164: - 5.09765625 frac[96,164] + mem[96,164] = 0
 accuracy_151_134: - 5.09765625 frac[151,134] + mem[151,134] = 0
 accuracy_193_180: - 5.09765625 frac[193,180] + mem[193,180] = 0
 accuracy_176_204: - 5.09765625 frac[176,204] + mem[176,204] = 0
 accuracy_195_184: - 5.09765625 frac[195,184] + mem[195,184] = 0
 accuracy_68_74: - 5.09765625 frac[68,74] + mem[68,74] = 0
 accuracy_196_135: - 5.09765625 frac[196,135] + mem[196,135] = 0
 accuracy_197_31: - 5.09765625 frac[197,31] + mem[197,31] = 0
 accuracy_198_167: - 5.09765625 frac[198,167] + mem[198,167] = 0
 accuracy_11_141: - 5.09765625 frac[11,141] + mem[11,141] = 0
 accuracy_194_58: - 5.09765625 frac[194,58] + mem[194,58] = 0
 accuracy_105_58: - 5.09765625 frac[105,58] + mem[105,58] = 0
 accuracy_143_119: - 5.09765625 frac[143,119] + mem[143,119] = 0
 accuracy_60_220: - 5.09765625 frac[60,220] + mem[60,220] = 0
 accuracy_10_79: - 5.09765625 frac[10,79] + mem[10,79] = 0
 accuracy_84_202: - 5.09765625 frac[84,202] + mem[84,202] = 0
 accuracy_156_226: - 5.09765625 frac[156,226] + mem[156,226] = 0
 accuracy_195_73: - 5.09765625 frac[195,73] + mem[195,73] = 0
 accuracy_106_219: - 5.09765625 frac[106,219] + mem[106,219] = 0
 accuracy_197_160: - 5.09765625 frac[197,160] + mem[197,160] = 0
 accuracy_198_54: - 5.09765625 frac[198,54] + mem[198,54] = 0
 accuracy_55_6: - 5.09765625 frac[55,6] + mem[55,6] = 0
 accuracy_114_139: - 5.09765625 frac[114,139] + mem[114,139] = 0
 accuracy_192_123: - 5.09765625 frac[192,123] + mem[192,123] = 0
 accuracy_193_99: - 5.09765625 frac[193,99] + mem[193,99] = 0
 accuracy_68_147: - 5.09765625 frac[68,147] + mem[68,147] = 0
 accuracy_178_2: - 5.09765625 frac[178,2] + mem[178,2] = 0
 accuracy_141_142: - 5.09765625 frac[141,142] + mem[141,142] = 0
 accuracy_89_50: - 5.09765625 frac[89,50] + mem[89,50] = 0
 accuracy_198_65: - 5.09765625 frac[198,65] + mem[198,65] = 0
 accuracy_2_244: - 5.09765625 frac[2,244] + mem[2,244] = 0
 accuracy_21_3: - 5.09765625 frac[21,3] + mem[21,3] = 0
 accuracy_9_158: - 5.09765625 frac[9,158] + mem[9,158] = 0
 accuracy_193_28: - 5.09765625 frac[193,28] + mem[193,28] = 0
 accuracy_121_92: - 5.09765625 frac[121,92] + mem[121,92] = 0
 accuracy_196_127: - 5.09765625 frac[196,127] + mem[196,127] = 0
 accuracy_197_119: - 5.09765625 frac[197,119] + mem[197,119] = 0
 accuracy_73_185: - 5.09765625 frac[73,185] + mem[73,185] = 0
 accuracy_182_62: - 5.09765625 frac[182,62] + mem[182,62] = 0
 accuracy_164_64: - 5.09765625 frac[164,64] + mem[164,64] = 0
 accuracy_81_169: - 5.09765625 frac[81,169] + mem[81,169] = 0
 accuracy_6_232: - 5.09765625 frac[6,232] + mem[6,232] = 0
 accuracy_189_72: - 5.09765625 frac[189,72] + mem[189,72] = 0
 accuracy_46_100: - 5.09765625 frac[46,100] + mem[46,100] = 0
 accuracy_195_161: - 5.09765625 frac[195,161] + mem[195,161] = 0
 accuracy_106_179: - 5.09765625 frac[106,179] + mem[106,179] = 0
 accuracy_196_142: - 5.09765625 frac[196,142] + mem[196,142] = 0
 accuracy_197_24: - 5.09765625 frac[197,24] + mem[197,24] = 0
 accuracy_198_158: - 5.09765625 frac[198,158] + mem[198,158] = 0
 accuracy_13_154: - 5.09765625 frac[13,154] + mem[13,154] = 0
 accuracy_34_225: - 5.09765625 frac[34,225] + mem[34,225] = 0
 accuracy_53_254: - 5.09765625 frac[53,254] + mem[53,254] = 0
 accuracy_119_77: - 5.09765625 frac[119,77] + mem[119,77] = 0
 accuracy_157_229: - 5.09765625 frac[157,229] + mem[157,229] = 0
 accuracy_195_78: - 5.09765625 frac[195,78] + mem[195,78] = 0
 accuracy_31_129: - 5.09765625 frac[31,129] + mem[31,129] = 0
 accuracy_67_123: - 5.09765625 frac[67,123] + mem[67,123] = 0
 accuracy_197_169: - 5.09765625 frac[197,169] + mem[197,169] = 0
 accuracy_198_41: - 5.09765625 frac[198,41] + mem[198,41] = 0
 accuracy_179_255: - 5.09765625 frac[179,255] + mem[179,255] = 0
 accuracy_36_233: - 5.09765625 frac[36,233] + mem[36,233] = 0
 accuracy_20_35: - 5.09765625 frac[20,35] + mem[20,35] = 0
 accuracy_60_74: - 5.09765625 frac[60,74] + mem[60,74] = 0
 accuracy_188_119: - 5.09765625 frac[188,119] + mem[188,119] = 0
 accuracy_192_114: - 5.09765625 frac[192,114] + mem[192,114] = 0
 accuracy_193_100: - 5.09765625 frac[193,100] + mem[193,100] = 0
 accuracy_70_119: - 5.09765625 frac[70,119] + mem[70,119] = 0
 accuracy_164_248: - 5.09765625 frac[164,248] + mem[164,248] = 0
 accuracy_74_122: - 5.09765625 frac[74,122] + mem[74,122] = 0
 accuracy_146_82: - 5.09765625 frac[146,82] + mem[146,82] = 0
 accuracy_39_86: - 5.09765625 frac[39,86] + mem[39,86] = 0
 accuracy_132_1: - 5.09765625 frac[132,1] + mem[132,1] = 0
 accuracy_8_15: - 5.09765625 frac[8,15] + mem[8,15] = 0
 accuracy_192_225: - 5.09765625 frac[192,225] + mem[192,225] = 0
 accuracy_48_166: - 5.09765625 frac[48,166] + mem[48,166] = 0
 accuracy_194_157: - 5.09765625 frac[194,157] + mem[194,157] = 0
 accuracy_11_42: - 5.09765625 frac[11,42] + mem[11,42] = 0
 accuracy_50_195: - 5.09765625 frac[50,195] + mem[50,195] = 0
 accuracy_196_102: - 5.09765625 frac[196,102] + mem[196,102] = 0
 accuracy_197_112: - 5.09765625 frac[197,112] + mem[197,112] = 0
 accuracy_74_205: - 5.09765625 frac[74,205] + mem[74,205] = 0
 accuracy_78_70: - 5.09765625 frac[78,70] + mem[78,70] = 0
 accuracy_118_41: - 5.09765625 frac[118,41] + mem[118,41] = 0
 accuracy_193_134: - 5.09765625 frac[193,134] + mem[193,134] = 0
 accuracy_194_12: - 5.09765625 frac[194,12] + mem[194,12] = 0
 accuracy_195_166: - 5.09765625 frac[195,166] + mem[195,166] = 0
 accuracy_86_6: - 5.09765625 frac[86,6] + mem[86,6] = 0
 accuracy_196_245: - 5.09765625 frac[196,245] + mem[196,245] = 0
 accuracy_197_1: - 5.09765625 frac[197,1] + mem[197,1] = 0
 accuracy_198_145: - 5.09765625 frac[198,145] + mem[198,145] = 0
 accuracy_123_158: - 5.09765625 frac[123,158] + mem[123,158] = 0
 accuracy_177_204: - 5.09765625 frac[177,204] + mem[177,204] = 0
 accuracy_41_90: - 5.09765625 frac[41,90] + mem[41,90] = 0
 accuracy_133_191: - 5.09765625 frac[133,191] + mem[133,191] = 0
 accuracy_134_71: - 5.09765625 frac[134,71] + mem[134,71] = 0
 accuracy_154_212: - 5.09765625 frac[154,212] + mem[154,212] = 0
 accuracy_155_206: - 5.09765625 frac[155,206] + mem[155,206] = 0
 accuracy_102_134: - 5.09765625 frac[102,134] + mem[102,134] = 0
 accuracy_6_114: - 5.09765625 frac[6,114] + mem[6,114] = 0
 accuracy_195_55: - 5.09765625 frac[195,55] + mem[195,55] = 0
 accuracy_31_142: - 5.09765625 frac[31,142] + mem[31,142] = 0
 accuracy_16_221: - 5.09765625 frac[16,221] + mem[16,221] = 0
 accuracy_198_32: - 5.09765625 frac[198,32] + mem[198,32] = 0
 accuracy_197_146: - 5.09765625 frac[197,146] + mem[197,146] = 0
 accuracy_0_39: - 5.09765625 frac[0,39] + mem[0,39] = 0
 accuracy_55_20: - 5.09765625 frac[55,20] + mem[55,20] = 0
 accuracy_39_254: - 5.09765625 frac[39,254] + mem[39,254] = 0
 accuracy_192_73: - 5.09765625 frac[192,73] + mem[192,73] = 0
 accuracy_45_94: - 5.09765625 frac[45,94] + mem[45,94] = 0
 accuracy_102_17: - 5.09765625 frac[102,17] + mem[102,17] = 0
 accuracy_67_241: - 5.09765625 frac[67,241] + mem[67,241] = 0
 accuracy_49_155: - 5.09765625 frac[49,155] + mem[49,155] = 0
 accuracy_168_226: - 5.09765625 frac[168,226] + mem[168,226] = 0
 accuracy_95_114: - 5.09765625 frac[95,114] + mem[95,114] = 0
 accuracy_166_136: - 5.09765625 frac[166,136] + mem[166,136] = 0
 accuracy_190_201: - 5.09765625 frac[190,201] + mem[190,201] = 0
 accuracy_45_239: - 5.09765625 frac[45,239] + mem[45,239] = 0
 accuracy_193_238: - 5.09765625 frac[193,238] + mem[193,238] = 0
 accuracy_194_148: - 5.09765625 frac[194,148] + mem[194,148] = 0
 accuracy_122_188: - 5.09765625 frac[122,188] + mem[122,188] = 0
 accuracy_174_56: - 5.09765625 frac[174,56] + mem[174,56] = 0
 accuracy_196_109: - 5.09765625 frac[196,109] + mem[196,109] = 0
 accuracy_197_121: - 5.09765625 frac[197,121] + mem[197,121] = 0
 accuracy_94_190: - 5.09765625 frac[94,190] + mem[94,190] = 0
 accuracy_58_10: - 5.09765625 frac[58,10] + mem[58,10] = 0
 accuracy_79_69: - 5.09765625 frac[79,69] + mem[79,69] = 0
 accuracy_193_159: - 5.09765625 frac[193,159] + mem[193,159] = 0
 accuracy_194_7: - 5.09765625 frac[194,7] + mem[194,7] = 0
 accuracy_195_175: - 5.09765625 frac[195,175] + mem[195,175] = 0
 accuracy_106_165: - 5.09765625 frac[106,165] + mem[106,165] = 0
 accuracy_196_252: - 5.09765625 frac[196,252] + mem[196,252] = 0
 accuracy_197_10: - 5.09765625 frac[197,10] + mem[197,10] = 0
 accuracy_17_73: - 5.09765625 frac[17,73] + mem[17,73] = 0
 accuracy_160_155: - 5.09765625 frac[160,155] + mem[160,155] = 0
 accuracy_16_69: - 5.09765625 frac[16,69] + mem[16,69] = 0
 accuracy_53_192: - 5.09765625 frac[53,192] + mem[53,192] = 0
 accuracy_198_136: - 5.09765625 frac[198,136] + mem[198,136] = 0
 accuracy_82_186: - 5.09765625 frac[82,186] + mem[82,186] = 0
 accuracy_136_148: - 5.09765625 frac[136,148] + mem[136,148] = 0
 accuracy_177_86: - 5.09765625 frac[177,86] + mem[177,86] = 0
 accuracy_195_60: - 5.09765625 frac[195,60] + mem[195,60] = 0
 accuracy_197_155: - 5.09765625 frac[197,155] + mem[197,155] = 0
 accuracy_198_27: - 5.09765625 frac[198,27] + mem[198,27] = 0
 accuracy_18_232: - 5.09765625 frac[18,232] + mem[18,232] = 0
 accuracy_192_64: - 5.09765625 frac[192,64] + mem[192,64] = 0
 accuracy_156_126: - 5.09765625 frac[156,126] + mem[156,126] = 0
 accuracy_138_132: - 5.09765625 frac[138,132] + mem[138,132] = 0
 accuracy_70_97: - 5.09765625 frac[70,97] + mem[70,97] = 0
 accuracy_90_189: - 5.09765625 frac[90,189] + mem[90,189] = 0
 accuracy_147_174: - 5.09765625 frac[147,174] + mem[147,174] = 0
 accuracy_149_73: - 5.09765625 frac[149,73] + mem[149,73] = 0
 accuracy_169_253: - 5.09765625 frac[169,253] + mem[169,253] = 0
 accuracy_98_205: - 5.09765625 frac[98,205] + mem[98,205] = 0
 accuracy_120_215: - 5.09765625 frac[120,215] + mem[120,215] = 0
 accuracy_193_231: - 5.09765625 frac[193,231] + mem[193,231] = 0
 accuracy_194_239: - 5.09765625 frac[194,239] + mem[194,239] = 0
 accuracy_192_255: - 5.09765625 frac[192,255] + mem[192,255] = 0
 accuracy_67_135: - 5.09765625 frac[67,135] + mem[67,135] = 0
 accuracy_196_84: - 5.09765625 frac[196,84] + mem[196,84] = 0
 accuracy_197_98: - 5.09765625 frac[197,98] + mem[197,98] = 0
 accuracy_41_59: - 5.09765625 frac[41,59] + mem[41,59] = 0
 accuracy_83_7: - 5.09765625 frac[83,7] + mem[83,7] = 0
 accuracy_193_144: - 5.09765625 frac[193,144] + mem[193,144] = 0
 accuracy_194_30: - 5.09765625 frac[194,30] + mem[194,30] = 0
 accuracy_195_148: - 5.09765625 frac[195,148] + mem[195,148] = 0
 accuracy_103_120: - 5.09765625 frac[103,120] + mem[103,120] = 0
 accuracy_196_251: - 5.09765625 frac[196,251] + mem[196,251] = 0
 accuracy_10_144: - 5.09765625 frac[10,144] + mem[10,144] = 0
 accuracy_198_131: - 5.09765625 frac[198,131] + mem[198,131] = 0
 accuracy_126_171: - 5.09765625 frac[126,171] + mem[126,171] = 0
 accuracy_143_75: - 5.09765625 frac[143,75] + mem[143,75] = 0
 accuracy_197_243: - 5.09765625 frac[197,243] + mem[197,243] = 0
 accuracy_38_151: - 5.09765625 frac[38,151] + mem[38,151] = 0
 accuracy_56_99: - 5.09765625 frac[56,99] + mem[56,99] = 0
 accuracy_28_51: - 5.09765625 frac[28,51] + mem[28,51] = 0
 accuracy_134_49: - 5.09765625 frac[134,49] + mem[134,49] = 0
 accuracy_136_155: - 5.09765625 frac[136,155] + mem[136,155] = 0
 accuracy_156_198: - 5.09765625 frac[156,198] + mem[156,198] = 0
 accuracy_85_216: - 5.09765625 frac[85,216] + mem[85,216] = 0
 accuracy_195_37: - 5.09765625 frac[195,37] + mem[195,37] = 0
 accuracy_105_167: - 5.09765625 frac[105,167] + mem[105,167] = 0
 accuracy_197_132: - 5.09765625 frac[197,132] + mem[197,132] = 0
 accuracy_198_18: - 5.09765625 frac[198,18] + mem[198,18] = 0
 accuracy_0_53: - 5.09765625 frac[0,53] + mem[0,53] = 0
 accuracy_112_223: - 5.09765625 frac[112,223] + mem[112,223] = 0
 accuracy_183_81: - 5.09765625 frac[183,81] + mem[183,81] = 0
 accuracy_186_247: - 5.09765625 frac[186,247] + mem[186,247] = 0
 accuracy_25_51: - 5.09765625 frac[25,51] + mem[25,51] = 0
 accuracy_62_232: - 5.09765625 frac[62,232] + mem[62,232] = 0
 accuracy_193_79: - 5.09765625 frac[193,79] + mem[193,79] = 0
 accuracy_86_242: - 5.09765625 frac[86,242] + mem[86,242] = 0
 accuracy_90_132: - 5.09765625 frac[90,132] + mem[90,132] = 0
 accuracy_18_114: - 5.09765625 frac[18,114] + mem[18,114] = 0
 accuracy_146_191: - 5.09765625 frac[146,191] + mem[146,191] = 0
 accuracy_97_222: - 5.09765625 frac[97,222] + mem[97,222] = 0
 accuracy_117_19: - 5.09765625 frac[117,19] + mem[117,19] = 0
 accuracy_154_15: - 5.09765625 frac[154,15] + mem[154,15] = 0
 accuracy_192_246: - 5.09765625 frac[192,246] + mem[192,246] = 0
 accuracy_193_248: - 5.09765625 frac[193,248] + mem[193,248] = 0
 accuracy_194_230: - 5.09765625 frac[194,230] + mem[194,230] = 0
 accuracy_176_16: - 5.09765625 frac[176,16] + mem[176,16] = 0
 accuracy_68_6: - 5.09765625 frac[68,6] + mem[68,6] = 0
 accuracy_196_83: - 5.09765625 frac[196,83] + mem[196,83] = 0
 accuracy_197_107: - 5.09765625 frac[197,107] + mem[197,107] = 0
 accuracy_90_55: - 5.09765625 frac[90,55] + mem[90,55] = 0
 accuracy_147_36: - 5.09765625 frac[147,36] + mem[147,36] = 0
 accuracy_58_28: - 5.09765625 frac[58,28] + mem[58,28] = 0
 accuracy_97_79: - 5.09765625 frac[97,79] + mem[97,79] = 0
 accuracy_170_143: - 5.09765625 frac[170,143] + mem[170,143] = 0
 accuracy_79_75: - 5.09765625 frac[79,75] + mem[79,75] = 0
 accuracy_173_242: - 5.09765625 frac[173,242] + mem[173,242] = 0
 accuracy_194_25: - 5.09765625 frac[194,25] + mem[194,25] = 0
 accuracy_195_157: - 5.09765625 frac[195,157] + mem[195,157] = 0
 accuracy_11_174: - 5.09765625 frac[11,174] + mem[11,174] = 0
 accuracy_196_226: - 5.09765625 frac[196,226] + mem[196,226] = 0
 accuracy_197_252: - 5.09765625 frac[197,252] + mem[197,252] = 0
 accuracy_198_250: - 5.09765625 frac[198,250] + mem[198,250] = 0
 accuracy_103_113: - 5.09765625 frac[103,113] + mem[103,113] = 0
 accuracy_15_101: - 5.09765625 frac[15,101] + mem[15,101] = 0
 accuracy_37_8: - 5.09765625 frac[37,8] + mem[37,8] = 0
 accuracy_71_136: - 5.09765625 frac[71,136] + mem[71,136] = 0
 accuracy_179_170: - 5.09765625 frac[179,170] + mem[179,170] = 0
 accuracy_59_247: - 5.09765625 frac[59,247] + mem[59,247] = 0
 accuracy_111_17: - 5.09765625 frac[111,17] + mem[111,17] = 0
 accuracy_192_47: - 5.09765625 frac[192,47] + mem[192,47] = 0
 accuracy_137_175: - 5.09765625 frac[137,175] + mem[137,175] = 0
 accuracy_138_39: - 5.09765625 frac[138,39] + mem[138,39] = 0
 accuracy_67_87: - 5.09765625 frac[67,87] + mem[67,87] = 0
 accuracy_195_42: - 5.09765625 frac[195,42] + mem[195,42] = 0
 accuracy_13_15: - 5.09765625 frac[13,15] + mem[13,15] = 0
 accuracy_197_141: - 5.09765625 frac[197,141] + mem[197,141] = 0
 accuracy_198_21: - 5.09765625 frac[198,21] + mem[198,21] = 0
 accuracy_60_110: - 5.09765625 frac[60,110] + mem[60,110] = 0
 accuracy_134_187: - 5.09765625 frac[134,187] + mem[134,187] = 0
 accuracy_192_94: - 5.09765625 frac[192,94] + mem[192,94] = 0
 accuracy_193_64: - 5.09765625 frac[193,64] + mem[193,64] = 0
 accuracy_157_122: - 5.09765625 frac[157,122] + mem[157,122] = 0
 accuracy_142_11: - 5.09765625 frac[142,11] + mem[142,11] = 0
 accuracy_53_57: - 5.09765625 frac[53,57] + mem[53,57] = 0
 accuracy_109_172: - 5.09765625 frac[109,172] + mem[109,172] = 0
 accuracy_127_6: - 5.09765625 frac[127,6] + mem[127,6] = 0
 accuracy_113_169: - 5.09765625 frac[113,169] + mem[113,169] = 0
 accuracy_96_239: - 5.09765625 frac[96,239] + mem[96,239] = 0
 accuracy_116_226: - 5.09765625 frac[116,226] + mem[116,226] = 0
 accuracy_27_217: - 5.09765625 frac[27,217] + mem[27,217] = 0
 accuracy_192_205: - 5.09765625 frac[192,205] + mem[192,205] = 0
 accuracy_193_241: - 5.09765625 frac[193,241] + mem[193,241] = 0
 accuracy_194_225: - 5.09765625 frac[194,225] + mem[194,225] = 0
 accuracy_195_245: - 5.09765625 frac[195,245] + mem[195,245] = 0
 accuracy_196_90: - 5.09765625 frac[196,90] + mem[196,90] = 0
 accuracy_197_84: - 5.09765625 frac[197,84] + mem[197,84] = 0
 accuracy_74_225: - 5.09765625 frac[74,225] + mem[74,225] = 0
 accuracy_57_175: - 5.09765625 frac[57,175] + mem[57,175] = 0
 accuracy_58_39: - 5.09765625 frac[58,39] + mem[58,39] = 0
 accuracy_115_42: - 5.09765625 frac[115,42] + mem[115,42] = 0
 accuracy_7_162: - 5.09765625 frac[7,162] + mem[7,162] = 0
 accuracy_173_235: - 5.09765625 frac[173,235] + mem[173,235] = 0
 accuracy_194_16: - 5.09765625 frac[194,16] + mem[194,16] = 0
 accuracy_195_130: - 5.09765625 frac[195,130] + mem[195,130] = 0
 accuracy_196_233: - 5.09765625 frac[196,233] + mem[196,233] = 0
 accuracy_197_229: - 5.09765625 frac[197,229] + mem[197,229] = 0
 accuracy_198_253: - 5.09765625 frac[198,253] + mem[198,253] = 0
 accuracy_40_235: - 5.09765625 frac[40,235] + mem[40,235] = 0
 accuracy_38_129: - 5.09765625 frac[38,129] + mem[38,129] = 0
 accuracy_60_246: - 5.09765625 frac[60,246] + mem[60,246] = 0
 accuracy_192_38: - 5.09765625 frac[192,38] + mem[192,38] = 0
 accuracy_155_170: - 5.09765625 frac[155,170] + mem[155,170] = 0
 accuracy_120_14: - 5.09765625 frac[120,14] + mem[120,14] = 0
 accuracy_195_19: - 5.09765625 frac[195,19] + mem[195,19] = 0
 accuracy_198_12: - 5.09765625 frac[198,12] + mem[198,12] = 0
 accuracy_96_119: - 5.09765625 frac[96,119] + mem[96,119] = 0
 accuracy_82_54: - 5.09765625 frac[82,54] + mem[82,54] = 0
 accuracy_192_85: - 5.09765625 frac[192,85] + mem[192,85] = 0
 accuracy_193_89: - 5.09765625 frac[193,89] + mem[193,89] = 0
 accuracy_196_50: - 5.09765625 frac[196,50] + mem[196,50] = 0
 accuracy_71_88: - 5.09765625 frac[71,88] + mem[71,88] = 0
 accuracy_18_100: - 5.09765625 frac[18,100] + mem[18,100] = 0
 accuracy_182_101: - 5.09765625 frac[182,101] + mem[182,101] = 0
 accuracy_96_230: - 5.09765625 frac[96,230] + mem[96,230] = 0
 accuracy_80_96: - 5.09765625 frac[80,96] + mem[80,96] = 0
 accuracy_78_58: - 5.09765625 frac[78,58] + mem[78,58] = 0
 accuracy_25_174: - 5.09765625 frac[25,174] + mem[25,174] = 0
 accuracy_192_196: - 5.09765625 frac[192,196] + mem[192,196] = 0
 accuracy_193_202: - 5.09765625 frac[193,202] + mem[193,202] = 0
 accuracy_194_248: - 5.09765625 frac[194,248] + mem[194,248] = 0
 accuracy_195_250: - 5.09765625 frac[195,250] + mem[195,250] = 0
 accuracy_177_144: - 5.09765625 frac[177,144] + mem[177,144] = 0
 accuracy_196_65: - 5.09765625 frac[196,65] + mem[196,65] = 0
 accuracy_197_93: - 5.09765625 frac[197,93] + mem[197,93] = 0
 accuracy_52_206: - 5.09765625 frac[52,206] + mem[52,206] = 0
 accuracy_131_172: - 5.09765625 frac[131,172] + mem[131,172] = 0
 accuracy_61_51: - 5.09765625 frac[61,51] + mem[61,51] = 0
 accuracy_196_208: - 5.09765625 frac[196,208] + mem[196,208] = 0
 accuracy_198_244: - 5.09765625 frac[198,244] + mem[198,244] = 0
 accuracy_79_206: - 5.09765625 frac[79,206] + mem[79,206] = 0
 accuracy_81_16: - 5.09765625 frac[81,16] + mem[81,16] = 0
 accuracy_192_61: - 5.09765625 frac[192,61] + mem[192,61] = 0
 accuracy_195_24: - 5.09765625 frac[195,24] + mem[195,24] = 0
 accuracy_86_148: - 5.09765625 frac[86,148] + mem[86,148] = 0
 accuracy_198_7: - 5.09765625 frac[198,7] + mem[198,7] = 0
 accuracy_127_167: - 5.09765625 frac[127,167] + mem[127,167] = 0
 accuracy_17_198: - 5.09765625 frac[17,198] + mem[17,198] = 0
 accuracy_183_188: - 5.09765625 frac[183,188] + mem[183,188] = 0
 accuracy_38_11: - 5.09765625 frac[38,11] + mem[38,11] = 0
 accuracy_188_41: - 5.09765625 frac[188,41] + mem[188,41] = 0
 accuracy_192_172: - 5.09765625 frac[192,172] + mem[192,172] = 0
 accuracy_193_82: - 5.09765625 frac[193,82] + mem[193,82] = 0
 accuracy_50_6: - 5.09765625 frac[50,6] + mem[50,6] = 0
 accuracy_196_57: - 5.09765625 frac[196,57] + mem[196,57] = 0
 accuracy_23_217: - 5.09765625 frac[23,217] + mem[23,217] = 0
 accuracy_192_219: - 5.09765625 frac[192,219] + mem[192,219] = 0
 accuracy_193_195: - 5.09765625 frac[193,195] + mem[193,195] = 0
 accuracy_194_243: - 5.09765625 frac[194,243] + mem[194,243] = 0
 accuracy_195_227: - 5.09765625 frac[195,227] + mem[195,227] = 0
 accuracy_47_152: - 5.09765625 frac[47,152] + mem[47,152] = 0
 accuracy_196_72: - 5.09765625 frac[196,72] + mem[196,72] = 0
 accuracy_197_70: - 5.09765625 frac[197,70] + mem[197,70] = 0
 accuracy_33_55: - 5.09765625 frac[33,55] + mem[33,55] = 0
 accuracy_48_112: - 5.09765625 frac[48,112] + mem[48,112] = 0
 accuracy_147_59: - 5.09765625 frac[147,59] + mem[147,59] = 0
 accuracy_171_193: - 5.09765625 frac[171,193] + mem[171,193] = 0
 accuracy_10_180: - 5.09765625 frac[10,180] + mem[10,180] = 0
 accuracy_194_98: - 5.09765625 frac[194,98] + mem[194,98] = 0
 accuracy_195_112: - 5.09765625 frac[195,112] + mem[195,112] = 0
 accuracy_196_223: - 5.09765625 frac[196,223] + mem[196,223] = 0
 accuracy_197_215: - 5.09765625 frac[197,215] + mem[197,215] = 0
 accuracy_198_239: - 5.09765625 frac[198,239] + mem[198,239] = 0
 accuracy_109_117: - 5.09765625 frac[109,117] + mem[109,117] = 0
 accuracy_34_46: - 5.09765625 frac[34,46] + mem[34,46] = 0
 accuracy_40_217: - 5.09765625 frac[40,217] + mem[40,217] = 0
 accuracy_131_34: - 5.09765625 frac[131,34] + mem[131,34] = 0
 accuracy_187_65: - 5.09765625 frac[187,65] + mem[187,65] = 0
 accuracy_170_11: - 5.09765625 frac[170,11] + mem[170,11] = 0
 accuracy_82_153: - 5.09765625 frac[82,153] + mem[82,153] = 0
 accuracy_192_52: - 5.09765625 frac[192,52] + mem[192,52] = 0
 accuracy_175_181: - 5.09765625 frac[175,181] + mem[175,181] = 0
 accuracy_195_1: - 5.09765625 frac[195,1] + mem[195,1] = 0
 accuracy_14_184: - 5.09765625 frac[14,184] + mem[14,184] = 0
 accuracy_72_159: - 5.09765625 frac[72,159] + mem[72,159] = 0
 accuracy_198_126: - 5.09765625 frac[198,126] + mem[198,126] = 0
 accuracy_96_5: - 5.09765625 frac[96,5] + mem[96,5] = 0
 accuracy_192_163: - 5.09765625 frac[192,163] + mem[192,163] = 0
 accuracy_193_43: - 5.09765625 frac[193,43] + mem[193,43] = 0
 accuracy_45_180: - 5.09765625 frac[45,180] + mem[45,180] = 0
 accuracy_196_32: - 5.09765625 frac[196,32] + mem[196,32] = 0
 accuracy_75_11: - 5.09765625 frac[75,11] + mem[75,11] = 0
 accuracy_118_239: - 5.09765625 frac[118,239] + mem[118,239] = 0
 accuracy_63_111: - 5.09765625 frac[63,111] + mem[63,111] = 0
 accuracy_193_196: - 5.09765625 frac[193,196] + mem[193,196] = 0
 accuracy_192_210: - 5.09765625 frac[192,210] + mem[192,210] = 0
 accuracy_195_232: - 5.09765625 frac[195,232] + mem[195,232] = 0
 accuracy_194_202: - 5.09765625 frac[194,202] + mem[194,202] = 0
 accuracy_196_183: - 5.09765625 frac[196,183] + mem[196,183] = 0
 accuracy_197_79: - 5.09765625 frac[197,79] + mem[197,79] = 0
 accuracy_34_182: - 5.09765625 frac[34,182] + mem[34,182] = 0
 accuracy_53_133: - 5.09765625 frac[53,133] + mem[53,133] = 0
 accuracy_98_91: - 5.09765625 frac[98,91] + mem[98,91] = 0
 accuracy_190_102: - 5.09765625 frac[190,102] + mem[190,102] = 0
 accuracy_66_48: - 5.09765625 frac[66,48] + mem[66,48] = 0
 accuracy_194_125: - 5.09765625 frac[194,125] + mem[194,125] = 0
 accuracy_195_121: - 5.09765625 frac[195,121] + mem[195,121] = 0
 accuracy_160_181: - 5.09765625 frac[160,181] + mem[160,181] = 0
 accuracy_196_198: - 5.09765625 frac[196,198] + mem[196,198] = 0
 accuracy_197_208: - 5.09765625 frac[197,208] + mem[197,208] = 0
 accuracy_198_230: - 5.09765625 frac[198,230] + mem[198,230] = 0
 accuracy_192_11: - 5.09765625 frac[192,11] + mem[192,11] = 0
 accuracy_83_153: - 5.09765625 frac[83,153] + mem[83,153] = 0
 accuracy_195_6: - 5.09765625 frac[195,6] + mem[195,6] = 0
 accuracy_159_20: - 5.09765625 frac[159,20] + mem[159,20] = 0
 accuracy_161_170: - 5.09765625 frac[161,170] + mem[161,170] = 0
 accuracy_198_113: - 5.09765625 frac[198,113] + mem[198,113] = 0
 accuracy_15_238: - 5.09765625 frac[15,238] + mem[15,238] = 0
 accuracy_92_89: - 5.09765625 frac[92,89] + mem[92,89] = 0
 accuracy_126_89: - 5.09765625 frac[126,89] + mem[126,89] = 0
 accuracy_112_242: - 5.09765625 frac[112,242] + mem[112,242] = 0
 accuracy_21_211: - 5.09765625 frac[21,211] + mem[21,211] = 0
 accuracy_24_158: - 5.09765625 frac[24,158] + mem[24,158] = 0
 accuracy_23_126: - 5.09765625 frac[23,126] + mem[23,126] = 0
 accuracy_122_55: - 5.09765625 frac[122,55] + mem[122,55] = 0
 accuracy_193_44: - 5.09765625 frac[193,44] + mem[193,44] = 0
 accuracy_196_47: - 5.09765625 frac[196,47] + mem[196,47] = 0
 accuracy_71_71: - 5.09765625 frac[71,71] + mem[71,71] = 0
 accuracy_115_153: - 5.09765625 frac[115,153] + mem[115,153] = 0
 accuracy_100_24: - 5.09765625 frac[100,24] + mem[100,24] = 0
 accuracy_172_48: - 5.09765625 frac[172,48] + mem[172,48] = 0
 accuracy_193_221: - 5.09765625 frac[193,221] + mem[193,221] = 0
 accuracy_195_209: - 5.09765625 frac[195,209] + mem[195,209] = 0
 accuracy_196_190: - 5.09765625 frac[196,190] + mem[196,190] = 0
 accuracy_197_72: - 5.09765625 frac[197,72] + mem[197,72] = 0
 accuracy_71_212: - 5.09765625 frac[71,212] + mem[71,212] = 0
 accuracy_147_73: - 5.09765625 frac[147,73] + mem[147,73] = 0
 accuracy_75_145: - 5.09765625 frac[75,145] + mem[75,145] = 0
 accuracy_149_224: - 5.09765625 frac[149,224] + mem[149,224] = 0
 accuracy_133_102: - 5.09765625 frac[133,102] + mem[133,102] = 0
 accuracy_27_82: - 5.09765625 frac[27,82] + mem[27,82] = 0
 accuracy_29_181: - 5.09765625 frac[29,181] + mem[29,181] = 0
 accuracy_194_116: - 5.09765625 frac[194,116] + mem[194,116] = 0
 accuracy_195_126: - 5.09765625 frac[195,126] + mem[195,126] = 0
 accuracy_196_205: - 5.09765625 frac[196,205] + mem[196,205] = 0
 accuracy_197_217: - 5.09765625 frac[197,217] + mem[197,217] = 0
 accuracy_198_217: - 5.09765625 frac[198,217] + mem[198,217] = 0
 accuracy_18_166: - 5.09765625 frac[18,166] + mem[18,166] = 0
 accuracy_54_167: - 5.09765625 frac[54,167] + mem[54,167] = 0
 accuracy_41_146: - 5.09765625 frac[41,146] + mem[41,146] = 0
 accuracy_192_2: - 5.09765625 frac[192,2] + mem[192,2] = 0
 accuracy_155_182: - 5.09765625 frac[155,182] + mem[155,182] = 0
 accuracy_195_15: - 5.09765625 frac[195,15] + mem[195,15] = 0
 accuracy_123_55: - 5.09765625 frac[123,55] + mem[123,55] = 0
 accuracy_198_104: - 5.09765625 frac[198,104] + mem[198,104] = 0
 accuracy_18_57: - 5.09765625 frac[18,57] + mem[18,57] = 0
 accuracy_186_157: - 5.09765625 frac[186,157] + mem[186,157] = 0
 accuracy_25_25: - 5.09765625 frac[25,25] + mem[25,25] = 0
 accuracy_117_72: - 5.09765625 frac[117,72] + mem[117,72] = 0
 accuracy_192_177: - 5.09765625 frac[192,177] + mem[192,177] = 0
 accuracy_63_12: - 5.09765625 frac[63,12] + mem[63,12] = 0
 accuracy_194_173: - 5.09765625 frac[194,173] + mem[194,173] = 0
 accuracy_193_37: - 5.09765625 frac[193,37] + mem[193,37] = 0
 accuracy_196_22: - 5.09765625 frac[196,22] + mem[196,22] = 0
 accuracy_14_53: - 5.09765625 frac[14,53] + mem[14,53] = 0
 accuracy_181_81: - 5.09765625 frac[181,81] + mem[181,81] = 0
 accuracy_149_120: - 5.09765625 frac[149,120] + mem[149,120] = 0
 accuracy_170_194: - 5.09765625 frac[170,194] + mem[170,194] = 0
 accuracy_193_214: - 5.09765625 frac[193,214] + mem[193,214] = 0
 accuracy_194_220: - 5.09765625 frac[194,220] + mem[194,220] = 0
 accuracy_195_214: - 5.09765625 frac[195,214] + mem[195,214] = 0
 accuracy_103_58: - 5.09765625 frac[103,58] + mem[103,58] = 0
 accuracy_196_165: - 5.09765625 frac[196,165] + mem[196,165] = 0
 accuracy_197_49: - 5.09765625 frac[197,49] + mem[197,49] = 0
 accuracy_73_99: - 5.09765625 frac[73,99] + mem[73,99] = 0
 accuracy_128_141: - 5.09765625 frac[128,141] + mem[128,141] = 0
 accuracy_77_230: - 5.09765625 frac[77,230] + mem[77,230] = 0
 accuracy_80_219: - 5.09765625 frac[80,219] + mem[80,219] = 0
 accuracy_81_227: - 5.09765625 frac[81,227] + mem[81,227] = 0
 accuracy_61_87: - 5.09765625 frac[61,87] + mem[61,87] = 0
 accuracy_194_79: - 5.09765625 frac[194,79] + mem[194,79] = 0
 accuracy_159_117: - 5.09765625 frac[159,117] + mem[159,117] = 0
 accuracy_197_194: - 5.09765625 frac[197,194] + mem[197,194] = 0
 accuracy_198_208: - 5.09765625 frac[198,208] + mem[198,208] = 0
 accuracy_181_184: - 5.09765625 frac[181,184] + mem[181,184] = 0
 accuracy_56_20: - 5.09765625 frac[56,20] + mem[56,20] = 0
 accuracy_128_60: - 5.09765625 frac[128,60] + mem[128,60] = 0
 accuracy_188_94: - 5.09765625 frac[188,94] + mem[188,94] = 0
 accuracy_133_240: - 5.09765625 frac[133,240] + mem[133,240] = 0
 accuracy_82_130: - 5.09765625 frac[82,130] + mem[82,130] = 0
 accuracy_192_25: - 5.09765625 frac[192,25] + mem[192,25] = 0
 accuracy_45_14: - 5.09765625 frac[45,14] + mem[45,14] = 0
 accuracy_63_164: - 5.09765625 frac[63,164] + mem[63,164] = 0
 accuracy_139_201: - 5.09765625 frac[139,201] + mem[139,201] = 0
 accuracy_140_217: - 5.09765625 frac[140,217] + mem[140,217] = 0
 accuracy_13_37: - 5.09765625 frac[13,37] + mem[13,37] = 0
 accuracy_89_204: - 5.09765625 frac[89,204] + mem[89,204] = 0
 accuracy_198_99: - 5.09765625 frac[198,99] + mem[198,99] = 0
 accuracy_181_41: - 5.09765625 frac[181,41] + mem[181,41] = 0
 accuracy_126_75: - 5.09765625 frac[126,75] + mem[126,75] = 0
 accuracy_134_145: - 5.09765625 frac[134,145] + mem[134,145] = 0
 accuracy_192_136: - 5.09765625 frac[192,136] + mem[192,136] = 0
 accuracy_194_164: - 5.09765625 frac[194,164] + mem[194,164] = 0
 accuracy_32_124: - 5.09765625 frac[32,124] + mem[32,124] = 0
 accuracy_50_42: - 5.09765625 frac[50,42] + mem[50,42] = 0
 accuracy_161_21: - 5.09765625 frac[161,21] + mem[161,21] = 0
 accuracy_90_117: - 5.09765625 frac[90,117] + mem[90,117] = 0
 accuracy_196_29: - 5.09765625 frac[196,29] + mem[196,29] = 0
 accuracy_181_90: - 5.09765625 frac[181,90] + mem[181,90] = 0
 accuracy_26_3: - 5.09765625 frac[26,3] + mem[26,3] = 0
 accuracy_193_175: - 5.09765625 frac[193,175] + mem[193,175] = 0
 accuracy_194_215: - 5.09765625 frac[194,215] + mem[194,215] = 0
 accuracy_104_73: - 5.09765625 frac[104,73] + mem[104,73] = 0
 accuracy_195_223: - 5.09765625 frac[195,223] + mem[195,223] = 0
 accuracy_33_211: - 5.09765625 frac[33,211] + mem[33,211] = 0
 accuracy_197_58: - 5.09765625 frac[197,58] + mem[197,58] = 0
 accuracy_198_184: - 5.09765625 frac[198,184] + mem[198,184] = 0
 accuracy_196_172: - 5.09765625 frac[196,172] + mem[196,172] = 0
 accuracy_148_79: - 5.09765625 frac[148,79] + mem[148,79] = 0
 accuracy_10_168: - 5.09765625 frac[10,168] + mem[10,168] = 0
 accuracy_194_70: - 5.09765625 frac[194,70] + mem[194,70] = 0
 accuracy_195_108: - 5.09765625 frac[195,108] + mem[195,108] = 0
 accuracy_159_114: - 5.09765625 frac[159,114] + mem[159,114] = 0
 accuracy_197_203: - 5.09765625 frac[197,203] + mem[197,203] = 0
 accuracy_198_203: - 5.09765625 frac[198,203] + mem[198,203] = 0
 accuracy_91_15: - 5.09765625 frac[91,15] + mem[91,15] = 0
 accuracy_125_243: - 5.09765625 frac[125,243] + mem[125,243] = 0
 accuracy_23_20: - 5.09765625 frac[23,20] + mem[23,20] = 0
 accuracy_192_16: - 5.09765625 frac[192,16] + mem[192,16] = 0
 accuracy_67_38: - 5.09765625 frac[67,38] + mem[67,38] = 0
 accuracy_160_73: - 5.09765625 frac[160,73] + mem[160,73] = 0
 accuracy_159_3: - 5.09765625 frac[159,3] + mem[159,3] = 0
 accuracy_198_90: - 5.09765625 frac[198,90] + mem[198,90] = 0
 accuracy_192_143: - 5.09765625 frac[192,143] + mem[192,143] = 0
 accuracy_193_55: - 5.09765625 frac[193,55] + mem[193,55] = 0
 accuracy_194_191: - 5.09765625 frac[194,191] + mem[194,191] = 0
 accuracy_196_4: - 5.09765625 frac[196,4] + mem[196,4] = 0
 accuracy_164_45: - 5.09765625 frac[164,45] + mem[164,45] = 0
 accuracy_166_194: - 5.09765625 frac[166,194] + mem[166,194] = 0
 accuracy_171_149: - 5.09765625 frac[171,149] + mem[171,149] = 0
 accuracy_193_160: - 5.09765625 frac[193,160] + mem[193,160] = 0
 accuracy_194_46: - 5.09765625 frac[194,46] + mem[194,46] = 0
 accuracy_196_171: - 5.09765625 frac[196,171] + mem[196,171] = 0
 accuracy_197_35: - 5.09765625 frac[197,35] + mem[197,35] = 0
 accuracy_69_104: - 5.09765625 frac[69,104] + mem[69,104] = 0
 accuracy_145_69: - 5.09765625 frac[145,69] + mem[145,69] = 0
 accuracy_198_179: - 5.09765625 frac[198,179] + mem[198,179] = 0
 accuracy_55_133: - 5.09765625 frac[55,133] + mem[55,133] = 0
 accuracy_170_71: - 5.09765625 frac[170,71] + mem[170,71] = 0
 accuracy_101_18: - 5.09765625 frac[101,18] + mem[101,18] = 0
 accuracy_174_237: - 5.09765625 frac[174,237] + mem[174,237] = 0
 accuracy_195_85: - 5.09765625 frac[195,85] + mem[195,85] = 0
 accuracy_194_65: - 5.09765625 frac[194,65] + mem[194,65] = 0
 accuracy_197_180: - 5.09765625 frac[197,180] + mem[197,180] = 0
 accuracy_198_194: - 5.09765625 frac[198,194] + mem[198,194] = 0
 accuracy_36_246: - 5.09765625 frac[36,246] + mem[36,246] = 0
 accuracy_6_47: - 5.09765625 frac[6,47] + mem[6,47] = 0
 accuracy_151_88: - 5.09765625 frac[151,88] + mem[151,88] = 0
 accuracy_192_23: - 5.09765625 frac[192,23] + mem[192,23] = 0
 accuracy_153_54: - 5.09765625 frac[153,54] + mem[153,54] = 0
 accuracy_173_75: - 5.09765625 frac[173,75] + mem[173,75] = 0
 accuracy_198_93: - 5.09765625 frac[198,93] + mem[198,93] = 0
 accuracy_18_34: - 5.09765625 frac[18,34] + mem[18,34] = 0
 accuracy_146_111: - 5.09765625 frac[146,111] + mem[146,111] = 0
 accuracy_182_187: - 5.09765625 frac[182,187] + mem[182,187] = 0
 accuracy_149_34: - 5.09765625 frac[149,34] + mem[149,34] = 0
 accuracy_192_134: - 5.09765625 frac[192,134] + mem[192,134] = 0
 accuracy_193_8: - 5.09765625 frac[193,8] + mem[193,8] = 0
 accuracy_194_182: - 5.09765625 frac[194,182] + mem[194,182] = 0
 accuracy_48_187: - 5.09765625 frac[48,187] + mem[48,187] = 0
 accuracy_33_124: - 5.09765625 frac[33,124] + mem[33,124] = 0
 accuracy_107_21: - 5.09765625 frac[107,21] + mem[107,21] = 0
 accuracy_169_39: - 5.09765625 frac[169,39] + mem[169,39] = 0
 accuracy_8_227: - 5.09765625 frac[8,227] + mem[8,227] = 0
 accuracy_27_129: - 5.09765625 frac[27,129] + mem[27,129] = 0
 accuracy_193_185: - 5.09765625 frac[193,185] + mem[193,185] = 0
 accuracy_194_41: - 5.09765625 frac[194,41] + mem[194,41] = 0
 accuracy_195_205: - 5.09765625 frac[195,205] + mem[195,205] = 0
 accuracy_196_146: - 5.09765625 frac[196,146] + mem[196,146] = 0
 accuracy_197_44: - 5.09765625 frac[197,44] + mem[197,44] = 0
 accuracy_162_30: - 5.09765625 frac[162,30] + mem[162,30] = 0
 accuracy_198_170: - 5.09765625 frac[198,170] + mem[198,170] = 0
 accuracy_15_53: - 5.09765625 frac[15,53] + mem[15,53] = 0
 accuracy_184_111: - 5.09765625 frac[184,111] + mem[184,111] = 0
 accuracy_73_22: - 5.09765625 frac[73,22] + mem[73,22] = 0
 accuracy_81_214: - 5.09765625 frac[81,214] + mem[81,214] = 0
 accuracy_10_90: - 5.09765625 frac[10,90] + mem[10,90] = 0
 accuracy_194_88: - 5.09765625 frac[194,88] + mem[194,88] = 0
 accuracy_47_51: - 5.09765625 frac[47,51] + mem[47,51] = 0
 accuracy_88_0: - 5.09765625 frac[88,0] + mem[88,0] = 0
 accuracy_195_90: - 5.09765625 frac[195,90] + mem[195,90] = 0
 accuracy_197_189: - 5.09765625 frac[197,189] + mem[197,189] = 0
 accuracy_107_236: - 5.09765625 frac[107,236] + mem[107,236] = 0
 accuracy_198_197: - 5.09765625 frac[198,197] + mem[198,197] = 0
 accuracy_36_253: - 5.09765625 frac[36,253] + mem[36,253] = 0
 accuracy_184_222: - 5.09765625 frac[184,222] + mem[184,222] = 0
 accuracy_40_163: - 5.09765625 frac[40,163] + mem[40,163] = 0
 accuracy_54_131: - 5.09765625 frac[54,131] + mem[54,131] = 0
 accuracy_17_4: - 5.09765625 frac[17,4] + mem[17,4] = 0
 accuracy_186_206: - 5.09765625 frac[186,206] + mem[186,206] = 0
 accuracy_127_225: - 5.09765625 frac[127,225] + mem[127,225] = 0
 accuracy_157_138: - 5.09765625 frac[157,138] + mem[157,138] = 0
 accuracy_89_39: - 5.09765625 frac[89,39] + mem[89,39] = 0
 accuracy_198_84: - 5.09765625 frac[198,84] + mem[198,84] = 0
 accuracy_99_90: - 5.09765625 frac[99,90] + mem[99,90] = 0
 accuracy_192_157: - 5.09765625 frac[192,157] + mem[192,157] = 0
 accuracy_194_177: - 5.09765625 frac[194,177] + mem[194,177] = 0
 accuracy_196_10: - 5.09765625 frac[196,10] + mem[196,10] = 0
 accuracy_147_253: - 5.09765625 frac[147,253] + mem[147,253] = 0
 accuracy_81_190: - 5.09765625 frac[81,190] + mem[81,190] = 0
 accuracy_9_252: - 5.09765625 frac[9,252] + mem[9,252] = 0
 accuracy_119_201: - 5.09765625 frac[119,201] + mem[119,201] = 0
 accuracy_194_32: - 5.09765625 frac[194,32] + mem[194,32] = 0
 accuracy_29_225: - 5.09765625 frac[29,225] + mem[29,225] = 0
 accuracy_195_178: - 5.09765625 frac[195,178] + mem[195,178] = 0
 accuracy_196_153: - 5.09765625 frac[196,153] + mem[196,153] = 0
 accuracy_197_21: - 5.09765625 frac[197,21] + mem[197,21] = 0
 accuracy_51_12: - 5.09765625 frac[51,12] + mem[51,12] = 0
 accuracy_17_108: - 5.09765625 frac[17,108] + mem[17,108] = 0
 accuracy_198_173: - 5.09765625 frac[198,173] + mem[198,173] = 0
 accuracy_19_55: - 5.09765625 frac[19,55] + mem[19,55] = 0
 accuracy_129_192: - 5.09765625 frac[129,192] + mem[129,192] = 0
 accuracy_2_5: - 5.09765625 frac[2,5] + mem[2,5] = 0
 accuracy_60_198: - 5.09765625 frac[60,198] + mem[60,198] = 0
 accuracy_82_215: - 5.09765625 frac[82,215] + mem[82,215] = 0
 accuracy_194_83: - 5.09765625 frac[194,83] + mem[194,83] = 0
 accuracy_195_67: - 5.09765625 frac[195,67] + mem[195,67] = 0
 accuracy_88_7: - 5.09765625 frac[88,7] + mem[88,7] = 0
 accuracy_141_3: - 5.09765625 frac[141,3] + mem[141,3] = 0
 accuracy_197_166: - 5.09765625 frac[197,166] + mem[197,166] = 0
 accuracy_198_60: - 5.09765625 frac[198,60] + mem[198,60] = 0
 accuracy_112_189: - 5.09765625 frac[112,189] + mem[112,189] = 0
 accuracy_3_62: - 5.09765625 frac[3,62] + mem[3,62] = 0
 accuracy_132_141: - 5.09765625 frac[132,141] + mem[132,141] = 0
 accuracy_136_72: - 5.09765625 frac[136,72] + mem[136,72] = 0
 accuracy_192_101: - 5.09765625 frac[192,101] + mem[192,101] = 0
 accuracy_49_119: - 5.09765625 frac[49,119] + mem[49,119] = 0
 accuracy_85_171: - 5.09765625 frac[85,171] + mem[85,171] = 0
 accuracy_159_254: - 5.09765625 frac[159,254] + mem[159,254] = 0
 accuracy_198_79: - 5.09765625 frac[198,79] + mem[198,79] = 0
 accuracy_108_201: - 5.09765625 frac[108,201] + mem[108,201] = 0
 accuracy_111_204: - 5.09765625 frac[111,204] + mem[111,204] = 0
 accuracy_60_36: - 5.09765625 frac[60,36] + mem[60,36] = 0
 accuracy_8_2: - 5.09765625 frac[8,2] + mem[8,2] = 0
 accuracy_192_148: - 5.09765625 frac[192,148] + mem[192,148] = 0
 accuracy_120_188: - 5.09765625 frac[120,188] + mem[120,188] = 0
 accuracy_193_26: - 5.09765625 frac[193,26] + mem[193,26] = 0
 accuracy_194_136: - 5.09765625 frac[194,136] + mem[194,136] = 0
 accuracy_196_113: - 5.09765625 frac[196,113] + mem[196,113] = 0
 accuracy_87_183: - 5.09765625 frac[87,183] + mem[87,183] = 0
 accuracy_141_229: - 5.09765625 frac[141,229] + mem[141,229] = 0
 accuracy_2_109: - 5.09765625 frac[2,109] + mem[2,109] = 0
 accuracy_98_153: - 5.09765625 frac[98,153] + mem[98,153] = 0
 accuracy_171_248: - 5.09765625 frac[171,248] + mem[171,248] = 0
 accuracy_7_75: - 5.09765625 frac[7,75] + mem[7,75] = 0
 accuracy_155_66: - 5.09765625 frac[155,66] + mem[155,66] = 0
 accuracy_48_56: - 5.09765625 frac[48,56] + mem[48,56] = 0
 accuracy_194_59: - 5.09765625 frac[194,59] + mem[194,59] = 0
 accuracy_193_139: - 5.09765625 frac[193,139] + mem[193,139] = 0
 accuracy_195_187: - 5.09765625 frac[195,187] + mem[195,187] = 0
 accuracy_196_128: - 5.09765625 frac[196,128] + mem[196,128] = 0
 accuracy_197_30: - 5.09765625 frac[197,30] + mem[197,30] = 0
 accuracy_198_164: - 5.09765625 frac[198,164] + mem[198,164] = 0
 accuracy_162_16: - 5.09765625 frac[162,16] + mem[162,16] = 0
 accuracy_15_59: - 5.09765625 frac[15,59] + mem[15,59] = 0
 accuracy_80_190: - 5.09765625 frac[80,190] + mem[80,190] = 0
 accuracy_7_216: - 5.09765625 frac[7,216] + mem[7,216] = 0
 accuracy_139_157: - 5.09765625 frac[139,157] + mem[139,157] = 0
 accuracy_195_72: - 5.09765625 frac[195,72] + mem[195,72] = 0
 accuracy_85_195: - 5.09765625 frac[85,195] + mem[85,195] = 0
 accuracy_197_175: - 5.09765625 frac[197,175] + mem[197,175] = 0
 accuracy_198_55: - 5.09765625 frac[198,55] + mem[198,55] = 0
 accuracy_20_37: - 5.09765625 frac[20,37] + mem[20,37] = 0
 accuracy_134_229: - 5.09765625 frac[134,229] + mem[134,229] = 0
 accuracy_192_124: - 5.09765625 frac[192,124] + mem[192,124] = 0
 accuracy_193_98: - 5.09765625 frac[193,98] + mem[193,98] = 0
 accuracy_102_100: - 5.09765625 frac[102,100] + mem[102,100] = 0
 accuracy_120_84: - 5.09765625 frac[120,84] + mem[120,84] = 0
 accuracy_32_32: - 5.09765625 frac[32,32] + mem[32,32] = 0
 accuracy_198_70: - 5.09765625 frac[198,70] + mem[198,70] = 0
 accuracy_110_164: - 5.09765625 frac[110,164] + mem[110,164] = 0
 accuracy_57_24: - 5.09765625 frac[57,24] + mem[57,24] = 0
 accuracy_97_177: - 5.09765625 frac[97,177] + mem[97,177] = 0
 accuracy_192_235: - 5.09765625 frac[192,235] + mem[192,235] = 0
 accuracy_193_19: - 5.09765625 frac[193,19] + mem[193,19] = 0
 accuracy_194_131: - 5.09765625 frac[194,131] + mem[194,131] = 0
 accuracy_139_123: - 5.09765625 frac[139,123] + mem[139,123] = 0
 accuracy_196_120: - 5.09765625 frac[196,120] + mem[196,120] = 0
 accuracy_72_38: - 5.09765625 frac[72,38] + mem[72,38] = 0
 accuracy_143_206: - 5.09765625 frac[143,206] + mem[143,206] = 0
 accuracy_114_106: - 5.09765625 frac[114,106] + mem[114,106] = 0
 accuracy_80_38: - 5.09765625 frac[80,38] + mem[80,38] = 0
 accuracy_8_152: - 5.09765625 frac[8,152] + mem[8,152] = 0
 accuracy_100_41: - 5.09765625 frac[100,41] + mem[100,41] = 0
 accuracy_119_223: - 5.09765625 frac[119,223] + mem[119,223] = 0
 accuracy_9_238: - 5.09765625 frac[9,238] + mem[9,238] = 0
 accuracy_194_50: - 5.09765625 frac[194,50] + mem[194,50] = 0
 accuracy_195_160: - 5.09765625 frac[195,160] + mem[195,160] = 0
 accuracy_31_243: - 5.09765625 frac[31,243] + mem[31,243] = 0
 accuracy_196_143: - 5.09765625 frac[196,143] + mem[196,143] = 0
 accuracy_197_7: - 5.09765625 frac[197,7] + mem[197,7] = 0
 accuracy_180_65: - 5.09765625 frac[180,65] + mem[180,65] = 0
 accuracy_198_159: - 5.09765625 frac[198,159] + mem[198,159] = 0
 accuracy_35_244: - 5.09765625 frac[35,244] + mem[35,244] = 0
 accuracy_193_140: - 5.09765625 frac[193,140] + mem[193,140] = 0
 accuracy_195_49: - 5.09765625 frac[195,49] + mem[195,49] = 0
 accuracy_158_71: - 5.09765625 frac[158,71] + mem[158,71] = 0
 accuracy_177_91: - 5.09765625 frac[177,91] + mem[177,91] = 0
 accuracy_197_168: - 5.09765625 frac[197,168] + mem[197,168] = 0
 accuracy_198_46: - 5.09765625 frac[198,46] + mem[198,46] = 0
 accuracy_16_231: - 5.09765625 frac[16,231] + mem[16,231] = 0
 accuracy_59_91: - 5.09765625 frac[59,91] + mem[59,91] = 0
 accuracy_6_11: - 5.09765625 frac[6,11] + mem[6,11] = 0
 accuracy_192_115: - 5.09765625 frac[192,115] + mem[192,115] = 0
 accuracy_193_123: - 5.09765625 frac[193,123] + mem[193,123] = 0
 accuracy_164_249: - 5.09765625 frac[164,249] + mem[164,249] = 0
 accuracy_8_48: - 5.09765625 frac[8,48] + mem[8,48] = 0
 accuracy_117_7: - 5.09765625 frac[117,7] + mem[117,7] = 0
 accuracy_192_226: - 5.09765625 frac[192,226] + mem[192,226] = 0
 accuracy_193_20: - 5.09765625 frac[193,20] + mem[193,20] = 0
 accuracy_194_154: - 5.09765625 frac[194,154] + mem[194,154] = 0
 accuracy_196_103: - 5.09765625 frac[196,103] + mem[196,103] = 0
 accuracy_197_127: - 5.09765625 frac[197,127] + mem[197,127] = 0
 accuracy_72_45: - 5.09765625 frac[72,45] + mem[72,45] = 0
 accuracy_143_215: - 5.09765625 frac[143,215] + mem[143,215] = 0
 accuracy_74_202: - 5.09765625 frac[74,202] + mem[74,202] = 0
 accuracy_2_127: - 5.09765625 frac[2,127] + mem[2,127] = 0
 accuracy_94_180: - 5.09765625 frac[94,180] + mem[94,180] = 0
 accuracy_146_226: - 5.09765625 frac[146,226] + mem[146,226] = 0
 accuracy_171_246: - 5.09765625 frac[171,246] + mem[171,246] = 0
 accuracy_193_133: - 5.09765625 frac[193,133] + mem[193,133] = 0
 accuracy_194_13: - 5.09765625 frac[194,13] + mem[194,13] = 0
 accuracy_195_169: - 5.09765625 frac[195,169] + mem[195,169] = 0
 accuracy_196_246: - 5.09765625 frac[196,246] + mem[196,246] = 0
 accuracy_52_147: - 5.09765625 frac[52,147] + mem[52,147] = 0
 accuracy_198_150: - 5.09765625 frac[198,150] + mem[198,150] = 0
 accuracy_197_0: - 5.09765625 frac[197,0] + mem[197,0] = 0
 accuracy_113_91: - 5.09765625 frac[113,91] + mem[113,91] = 0
 accuracy_76_41: - 5.09765625 frac[76,41] + mem[76,41] = 0
 accuracy_131_219: - 5.09765625 frac[131,219] + mem[131,219] = 0
 accuracy_84_217: - 5.09765625 frac[84,217] + mem[84,217] = 0
 accuracy_195_54: - 5.09765625 frac[195,54] + mem[195,54] = 0
 accuracy_197_145: - 5.09765625 frac[197,145] + mem[197,145] = 0
 accuracy_198_33: - 5.09765625 frac[198,33] + mem[198,33] = 0
 accuracy_109_47: - 5.09765625 frac[109,47] + mem[109,47] = 0
 accuracy_183_66: - 5.09765625 frac[183,66] + mem[183,66] = 0
 accuracy_38_245: - 5.09765625 frac[38,245] + mem[38,245] = 0
 accuracy_192_74: - 5.09765625 frac[192,74] + mem[192,74] = 0
 accuracy_193_124: - 5.09765625 frac[193,124] + mem[193,124] = 0
 accuracy_186_85: - 5.09765625 frac[186,85] + mem[186,85] = 0
 accuracy_188_254: - 5.09765625 frac[188,254] + mem[188,254] = 0
 accuracy_118_182: - 5.09765625 frac[118,182] + mem[118,182] = 0
 accuracy_192_249: - 5.09765625 frac[192,249] + mem[192,249] = 0
 accuracy_191_147: - 5.09765625 frac[191,147] + mem[191,147] = 0
 accuracy_194_149: - 5.09765625 frac[194,149] + mem[194,149] = 0
 accuracy_65_58: - 5.09765625 frac[65,58] + mem[65,58] = 0
 accuracy_135_108: - 5.09765625 frac[135,108] + mem[135,108] = 0
 accuracy_196_110: - 5.09765625 frac[196,110] + mem[196,110] = 0
 accuracy_197_120: - 5.09765625 frac[197,120] + mem[197,120] = 0
 accuracy_34_129: - 5.09765625 frac[34,129] + mem[34,129] = 0
 accuracy_44_42: - 5.09765625 frac[44,42] + mem[44,42] = 0
 accuracy_190_89: - 5.09765625 frac[190,89] + mem[190,89] = 0
 accuracy_119_173: - 5.09765625 frac[119,173] + mem[119,173] = 0
 accuracy_193_158: - 5.09765625 frac[193,158] + mem[193,158] = 0
 accuracy_194_4: - 5.09765625 frac[194,4] + mem[194,4] = 0
 accuracy_195_174: - 5.09765625 frac[195,174] + mem[195,174] = 0
 accuracy_121_182: - 5.09765625 frac[121,182] + mem[121,182] = 0
 accuracy_196_253: - 5.09765625 frac[196,253] + mem[196,253] = 0
 accuracy_197_9: - 5.09765625 frac[197,9] + mem[197,9] = 0
 accuracy_198_137: - 5.09765625 frac[198,137] + mem[198,137] = 0
 accuracy_56_93: - 5.09765625 frac[56,93] + mem[56,93] = 0
 accuracy_38_157: - 5.09765625 frac[38,157] + mem[38,157] = 0
 accuracy_111_2: - 5.09765625 frac[111,2] + mem[111,2] = 0
 accuracy_61_159: - 5.09765625 frac[61,159] + mem[61,159] = 0
 accuracy_123_7: - 5.09765625 frac[123,7] + mem[123,7] = 0
 accuracy_195_63: - 5.09765625 frac[195,63] + mem[195,63] = 0
 accuracy_177_85: - 5.09765625 frac[177,85] + mem[177,85] = 0
 accuracy_197_154: - 5.09765625 frac[197,154] + mem[197,154] = 0
 accuracy_198_24: - 5.09765625 frac[198,24] + mem[198,24] = 0
 accuracy_110_206: - 5.09765625 frac[110,206] + mem[110,206] = 0
 accuracy_92_96: - 5.09765625 frac[92,96] + mem[92,96] = 0
 accuracy_38_236: - 5.09765625 frac[38,236] + mem[38,236] = 0
 accuracy_41_211: - 5.09765625 frac[41,211] + mem[41,211] = 0
 accuracy_132_97: - 5.09765625 frac[132,97] + mem[132,97] = 0
 accuracy_192_65: - 5.09765625 frac[192,65] + mem[192,65] = 0
 accuracy_193_117: - 5.09765625 frac[193,117] + mem[193,117] = 0
 accuracy_84_87: - 5.09765625 frac[84,87] + mem[84,87] = 0
 accuracy_106_68: - 5.09765625 frac[106,68] + mem[106,68] = 0
 accuracy_71_108: - 5.09765625 frac[71,108] + mem[71,108] = 0
 accuracy_92_23: - 5.09765625 frac[92,23] + mem[92,23] = 0
 accuracy_98_202: - 5.09765625 frac[98,202] + mem[98,202] = 0
 accuracy_190_241: - 5.09765625 frac[190,241] + mem[190,241] = 0
 accuracy_192_240: - 5.09765625 frac[192,240] + mem[192,240] = 0
 accuracy_193_230: - 5.09765625 frac[193,230] + mem[193,230] = 0
 accuracy_194_236: - 5.09765625 frac[194,236] + mem[194,236] = 0
 accuracy_157_24: - 5.09765625 frac[157,24] + mem[157,24] = 0
 accuracy_174_64: - 5.09765625 frac[174,64] + mem[174,64] = 0
 accuracy_196_85: - 5.09765625 frac[196,85] + mem[196,85] = 0
 accuracy_197_97: - 5.09765625 frac[197,97] + mem[197,97] = 0
 accuracy_104_114: - 5.09765625 frac[104,114] + mem[104,114] = 0
 accuracy_67_134: - 5.09765625 frac[67,134] + mem[67,134] = 0
 accuracy_149_185: - 5.09765625 frac[149,185] + mem[149,185] = 0
 accuracy_98_189: - 5.09765625 frac[98,189] + mem[98,189] = 0
 accuracy_152_211: - 5.09765625 frac[152,211] + mem[152,211] = 0
 accuracy_10_145: - 5.09765625 frac[10,145] + mem[10,145] = 0
 accuracy_46_126: - 5.09765625 frac[46,126] + mem[46,126] = 0
 accuracy_194_31: - 5.09765625 frac[194,31] + mem[194,31] = 0
 accuracy_193_151: - 5.09765625 frac[193,151] + mem[193,151] = 0
 accuracy_195_151: - 5.09765625 frac[195,151] + mem[195,151] = 0
 accuracy_196_228: - 5.09765625 frac[196,228] + mem[196,228] = 0
 accuracy_197_242: - 5.09765625 frac[197,242] + mem[197,242] = 0
 accuracy_198_128: - 5.09765625 frac[198,128] + mem[198,128] = 0
 accuracy_180_94: - 5.09765625 frac[180,94] + mem[180,94] = 0
 accuracy_69_39: - 5.09765625 frac[69,39] + mem[69,39] = 0
 accuracy_34_11: - 5.09765625 frac[34,11] + mem[34,11] = 0
 accuracy_111_11: - 5.09765625 frac[111,11] + mem[111,11] = 0
 accuracy_36_176: - 5.09765625 frac[36,176] + mem[36,176] = 0
 accuracy_24_13: - 5.09765625 frac[24,13] + mem[24,13] = 0
 accuracy_98_44: - 5.09765625 frac[98,44] + mem[98,44] = 0
 accuracy_64_228: - 5.09765625 frac[64,228] + mem[64,228] = 0
 accuracy_192_41: - 5.09765625 frac[192,41] + mem[192,41] = 0
 accuracy_137_165: - 5.09765625 frac[137,165] + mem[137,165] = 0
 accuracy_157_207: - 5.09765625 frac[157,207] + mem[157,207] = 0
 accuracy_195_36: - 5.09765625 frac[195,36] + mem[195,36] = 0
 accuracy_197_131: - 5.09765625 frac[197,131] + mem[197,131] = 0
 accuracy_198_19: - 5.09765625 frac[198,19] + mem[198,19] = 0
 accuracy_16_204: - 5.09765625 frac[16,204] + mem[16,204] = 0
 accuracy_184_136: - 5.09765625 frac[184,136] + mem[184,136] = 0
 accuracy_77_56: - 5.09765625 frac[77,56] + mem[77,56] = 0
 accuracy_44_250: - 5.09765625 frac[44,250] + mem[44,250] = 0
 accuracy_82_69: - 5.09765625 frac[82,69] + mem[82,69] = 0
 accuracy_63_229: - 5.09765625 frac[63,229] + mem[63,229] = 0
 accuracy_193_78: - 5.09765625 frac[193,78] + mem[193,78] = 0
 accuracy_66_244: - 5.09765625 frac[66,244] + mem[66,244] = 0
 accuracy_192_88: - 5.09765625 frac[192,88] + mem[192,88] = 0
 accuracy_49_148: - 5.09765625 frac[49,148] + mem[49,148] = 0
 accuracy_178_39: - 5.09765625 frac[178,39] + mem[178,39] = 0
 accuracy_46_167: - 5.09765625 frac[46,167] + mem[46,167] = 0
 accuracy_40_28: - 5.09765625 frac[40,28] + mem[40,28] = 0
 accuracy_192_247: - 5.09765625 frac[192,247] + mem[192,247] = 0
 accuracy_193_255: - 5.09765625 frac[193,255] + mem[193,255] = 0
 accuracy_194_231: - 5.09765625 frac[194,231] + mem[194,231] = 0
 accuracy_49_229: - 5.09765625 frac[49,229] + mem[49,229] = 0
 accuracy_104_121: - 5.09765625 frac[104,121] + mem[104,121] = 0
 accuracy_196_92: - 5.09765625 frac[196,92] + mem[196,92] = 0
 accuracy_197_106: - 5.09765625 frac[197,106] + mem[197,106] = 0
 accuracy_16_37: - 5.09765625 frac[16,37] + mem[16,37] = 0
 accuracy_143_34: - 5.09765625 frac[143,34] + mem[143,34] = 0
 accuracy_28_184: - 5.09765625 frac[28,184] + mem[28,184] = 0
 accuracy_146_207: - 5.09765625 frac[146,207] + mem[146,207] = 0
 accuracy_125_82: - 5.09765625 frac[125,82] + mem[125,82] = 0
 accuracy_115_36: - 5.09765625 frac[115,36] + mem[115,36] = 0
 accuracy_81_148: - 5.09765625 frac[81,148] + mem[81,148] = 0
 accuracy_44_24: - 5.09765625 frac[44,24] + mem[44,24] = 0
 accuracy_7_164: - 5.09765625 frac[7,164] + mem[7,164] = 0
 accuracy_137_93: - 5.09765625 frac[137,93] + mem[137,93] = 0
 accuracy_194_22: - 5.09765625 frac[194,22] + mem[194,22] = 0
 accuracy_196_227: - 5.09765625 frac[196,227] + mem[196,227] = 0
 accuracy_16_180: - 5.09765625 frac[16,180] + mem[16,180] = 0
 accuracy_197_251: - 5.09765625 frac[197,251] + mem[197,251] = 0
 accuracy_143_83: - 5.09765625 frac[143,83] + mem[143,83] = 0
 accuracy_145_13: - 5.09765625 frac[145,13] + mem[145,13] = 0
 accuracy_42_66: - 5.09765625 frac[42,66] + mem[42,66] = 0
 accuracy_61_145: - 5.09765625 frac[61,145] + mem[61,145] = 0
 accuracy_136_131: - 5.09765625 frac[136,131] + mem[136,131] = 0
 accuracy_192_32: - 5.09765625 frac[192,32] + mem[192,32] = 0
 accuracy_119_80: - 5.09765625 frac[119,80] + mem[119,80] = 0
 accuracy_139_190: - 5.09765625 frac[139,190] + mem[139,190] = 0
 accuracy_12_128: - 5.09765625 frac[12,128] + mem[12,128] = 0
 accuracy_195_45: - 5.09765625 frac[195,45] + mem[195,45] = 0
 accuracy_175_153: - 5.09765625 frac[175,153] + mem[175,153] = 0
 accuracy_197_140: - 5.09765625 frac[197,140] + mem[197,140] = 0
 accuracy_198_10: - 5.09765625 frac[198,10] + mem[198,10] = 0
 accuracy_40_116: - 5.09765625 frac[40,116] + mem[40,116] = 0
 accuracy_187_170: - 5.09765625 frac[187,170] + mem[187,170] = 0
 accuracy_193_71: - 5.09765625 frac[193,71] + mem[193,71] = 0
 accuracy_67_231: - 5.09765625 frac[67,231] + mem[67,231] = 0
 accuracy_196_52: - 5.09765625 frac[196,52] + mem[196,52] = 0
 accuracy_90_140: - 5.09765625 frac[90,140] + mem[90,140] = 0
 accuracy_71_98: - 5.09765625 frac[71,98] + mem[71,98] = 0
 accuracy_163_133: - 5.09765625 frac[163,133] + mem[163,133] = 0
 accuracy_2_208: - 5.09765625 frac[2,208] + mem[2,208] = 0
 accuracy_113_168: - 5.09765625 frac[113,168] + mem[113,168] = 0
 accuracy_96_224: - 5.09765625 frac[96,224] + mem[96,224] = 0
 accuracy_21_47: - 5.09765625 frac[21,47] + mem[21,47] = 0
 accuracy_39_117: - 5.09765625 frac[39,117] + mem[39,117] = 0
 accuracy_78_48: - 5.09765625 frac[78,48] + mem[78,48] = 0
 accuracy_171_69: - 5.09765625 frac[171,69] + mem[171,69] = 0
 accuracy_190_227: - 5.09765625 frac[190,227] + mem[190,227] = 0
 accuracy_192_206: - 5.09765625 frac[192,206] + mem[192,206] = 0
 accuracy_193_240: - 5.09765625 frac[193,240] + mem[193,240] = 0
 accuracy_194_254: - 5.09765625 frac[194,254] + mem[194,254] = 0
 accuracy_191_142: - 5.09765625 frac[191,142] + mem[191,142] = 0
 accuracy_195_244: - 5.09765625 frac[195,244] + mem[195,244] = 0
 accuracy_196_91: - 5.09765625 frac[196,91] + mem[196,91] = 0
 accuracy_124_99: - 5.09765625 frac[124,99] + mem[124,99] = 0
 accuracy_197_83: - 5.09765625 frac[197,83] + mem[197,83] = 0
 accuracy_100_101: - 5.09765625 frac[100,101] + mem[100,101] = 0
 accuracy_167_145: - 5.09765625 frac[167,145] + mem[167,145] = 0
 accuracy_194_17: - 5.09765625 frac[194,17] + mem[194,17] = 0
 accuracy_195_133: - 5.09765625 frac[195,133] + mem[195,133] = 0
 accuracy_29_208: - 5.09765625 frac[29,208] + mem[29,208] = 0
 accuracy_33_149: - 5.09765625 frac[33,149] + mem[33,149] = 0
 accuracy_197_228: - 5.09765625 frac[197,228] + mem[197,228] = 0
 accuracy_198_242: - 5.09765625 frac[198,242] + mem[198,242] = 0
 accuracy_112_127: - 5.09765625 frac[112,127] + mem[112,127] = 0
 accuracy_19_102: - 5.09765625 frac[19,102] + mem[19,102] = 0
 accuracy_77_89: - 5.09765625 frac[77,89] + mem[77,89] = 0
 accuracy_192_39: - 5.09765625 frac[192,39] + mem[192,39] = 0
 accuracy_137_215: - 5.09765625 frac[137,215] + mem[137,215] = 0
 accuracy_198_13: - 5.09765625 frac[198,13] + mem[198,13] = 0
 accuracy_164_181: - 5.09765625 frac[164,181] + mem[164,181] = 0
 accuracy_37_129: - 5.09765625 frac[37,129] + mem[37,129] = 0
 accuracy_40_123: - 5.09765625 frac[40,123] + mem[40,123] = 0
 accuracy_168_48: - 5.09765625 frac[168,48] + mem[168,48] = 0
 accuracy_134_179: - 5.09765625 frac[134,179] + mem[134,179] = 0
 accuracy_192_86: - 5.09765625 frac[192,86] + mem[192,86] = 0
 accuracy_193_88: - 5.09765625 frac[193,88] + mem[193,88] = 0
 accuracy_196_51: - 5.09765625 frac[196,51] + mem[196,51] = 0
 accuracy_107_69: - 5.09765625 frac[107,69] + mem[107,69] = 0
 accuracy_87_241: - 5.09765625 frac[87,241] + mem[87,241] = 0
 accuracy_109_164: - 5.09765625 frac[109,164] + mem[109,164] = 0
 accuracy_182_122: - 5.09765625 frac[182,122] + mem[182,122] = 0
 accuracy_193_201: - 5.09765625 frac[193,201] + mem[193,201] = 0
 accuracy_194_249: - 5.09765625 frac[194,249] + mem[194,249] = 0
 accuracy_195_253: - 5.09765625 frac[195,253] + mem[195,253] = 0
 accuracy_196_66: - 5.09765625 frac[196,66] + mem[196,66] = 0
 accuracy_197_92: - 5.09765625 frac[197,92] + mem[197,92] = 0
 accuracy_93_27: - 5.09765625 frac[93,27] + mem[93,27] = 0
 accuracy_80_240: - 5.09765625 frac[80,240] + mem[80,240] = 0
 accuracy_27_78: - 5.09765625 frac[27,78] + mem[27,78] = 0
 accuracy_173_227: - 5.09765625 frac[173,227] + mem[173,227] = 0
 accuracy_194_104: - 5.09765625 frac[194,104] + mem[194,104] = 0
 accuracy_195_138: - 5.09765625 frac[195,138] + mem[195,138] = 0
 accuracy_160_184: - 5.09765625 frac[160,184] + mem[160,184] = 0
 accuracy_196_209: - 5.09765625 frac[196,209] + mem[196,209] = 0
 accuracy_197_237: - 5.09765625 frac[197,237] + mem[197,237] = 0
 accuracy_198_245: - 5.09765625 frac[198,245] + mem[198,245] = 0
 accuracy_16_162: - 5.09765625 frac[16,162] + mem[16,162] = 0
 accuracy_41_190: - 5.09765625 frac[41,190] + mem[41,190] = 0
 accuracy_59_4: - 5.09765625 frac[59,4] + mem[59,4] = 0
 accuracy_150_189: - 5.09765625 frac[150,189] + mem[150,189] = 0
 accuracy_135_171: - 5.09765625 frac[135,171] + mem[135,171] = 0
 accuracy_192_62: - 5.09765625 frac[192,62] + mem[192,62] = 0
 accuracy_119_38: - 5.09765625 frac[119,38] + mem[119,38] = 0
 accuracy_137_216: - 5.09765625 frac[137,216] + mem[137,216] = 0
 accuracy_195_27: - 5.09765625 frac[195,27] + mem[195,27] = 0
 accuracy_123_35: - 5.09765625 frac[123,35] + mem[123,35] = 0
 accuracy_13_0: - 5.09765625 frac[13,0] + mem[13,0] = 0
 accuracy_198_4: - 5.09765625 frac[198,4] + mem[198,4] = 0
 accuracy_168_55: - 5.09765625 frac[168,55] + mem[168,55] = 0
 accuracy_189_148: - 5.09765625 frac[189,148] + mem[189,148] = 0
 accuracy_193_81: - 5.09765625 frac[193,81] + mem[193,81] = 0
 accuracy_140_133: - 5.09765625 frac[140,133] + mem[140,133] = 0
 accuracy_196_58: - 5.09765625 frac[196,58] + mem[196,58] = 0
 accuracy_94_243: - 5.09765625 frac[94,243] + mem[94,243] = 0
 accuracy_21_49: - 5.09765625 frac[21,49] + mem[21,49] = 0
 accuracy_154_25: - 5.09765625 frac[154,25] + mem[154,25] = 0
 accuracy_192_220: - 5.09765625 frac[192,220] + mem[192,220] = 0
 accuracy_193_194: - 5.09765625 frac[193,194] + mem[193,194] = 0
 accuracy_194_240: - 5.09765625 frac[194,240] + mem[194,240] = 0
 accuracy_195_226: - 5.09765625 frac[195,226] + mem[195,226] = 0
 accuracy_32_192: - 5.09765625 frac[32,192] + mem[32,192] = 0
 accuracy_196_73: - 5.09765625 frac[196,73] + mem[196,73] = 0
 accuracy_197_69: - 5.09765625 frac[197,69] + mem[197,69] = 0
 accuracy_30_161: - 5.09765625 frac[30,161] + mem[30,161] = 0
 accuracy_6_190: - 5.09765625 frac[6,190] + mem[6,190] = 0
 accuracy_8_169: - 5.09765625 frac[8,169] + mem[8,169] = 0
 accuracy_7_147: - 5.09765625 frac[7,147] + mem[7,147] = 0
 accuracy_10_181: - 5.09765625 frac[10,181] + mem[10,181] = 0
 accuracy_156_180: - 5.09765625 frac[156,180] + mem[156,180] = 0
 accuracy_194_99: - 5.09765625 frac[194,99] + mem[194,99] = 0
 accuracy_176_149: - 5.09765625 frac[176,149] + mem[176,149] = 0
 accuracy_196_216: - 5.09765625 frac[196,216] + mem[196,216] = 0
 accuracy_197_214: - 5.09765625 frac[197,214] + mem[197,214] = 0
 accuracy_35_135: - 5.09765625 frac[35,135] + mem[35,135] = 0
 accuracy_198_236: - 5.09765625 frac[198,236] + mem[198,236] = 0
 accuracy_108_106: - 5.09765625 frac[108,106] + mem[108,106] = 0
 accuracy_55_216: - 5.09765625 frac[55,216] + mem[55,216] = 0
 accuracy_183_7: - 5.09765625 frac[183,7] + mem[183,7] = 0
 accuracy_180_50: - 5.09765625 frac[180,50] + mem[180,50] = 0
 accuracy_42_63: - 5.09765625 frac[42,63] + mem[42,63] = 0
 accuracy_150_180: - 5.09765625 frac[150,180] + mem[150,180] = 0
 accuracy_136_184: - 5.09765625 frac[136,184] + mem[136,184] = 0
 accuracy_135_160: - 5.09765625 frac[135,160] + mem[135,160] = 0
 accuracy_192_53: - 5.09765625 frac[192,53] + mem[192,53] = 0
 accuracy_156_43: - 5.09765625 frac[156,43] + mem[156,43] = 0
 accuracy_195_0: - 5.09765625 frac[195,0] + mem[195,0] = 0
 accuracy_89_248: - 5.09765625 frac[89,248] + mem[89,248] = 0
 accuracy_198_127: - 5.09765625 frac[198,127] + mem[198,127] = 0
 accuracy_127_175: - 5.09765625 frac[127,175] + mem[127,175] = 0
 accuracy_44_222: - 5.09765625 frac[44,222] + mem[44,222] = 0
 accuracy_192_164: - 5.09765625 frac[192,164] + mem[192,164] = 0
 accuracy_193_42: - 5.09765625 frac[193,42] + mem[193,42] = 0
 accuracy_196_33: - 5.09765625 frac[196,33] + mem[196,33] = 0
 accuracy_127_60: - 5.09765625 frac[127,60] + mem[127,60] = 0
 accuracy_97_249: - 5.09765625 frac[97,249] + mem[97,249] = 0
 accuracy_5_144: - 5.09765625 frac[5,144] + mem[5,144] = 0
 accuracy_99_128: - 5.09765625 frac[99,128] + mem[99,128] = 0
 accuracy_7_123: - 5.09765625 frac[7,123] + mem[7,123] = 0
 accuracy_192_211: - 5.09765625 frac[192,211] + mem[192,211] = 0
 accuracy_193_219: - 5.09765625 frac[193,219] + mem[193,219] = 0
 accuracy_194_203: - 5.09765625 frac[194,203] + mem[194,203] = 0
 accuracy_195_235: - 5.09765625 frac[195,235] + mem[195,235] = 0
 accuracy_29_10: - 5.09765625 frac[29,10] + mem[29,10] = 0
 accuracy_122_227: - 5.09765625 frac[122,227] + mem[122,227] = 0
 accuracy_197_78: - 5.09765625 frac[197,78] + mem[197,78] = 0
 accuracy_90_8: - 5.09765625 frac[90,8] + mem[90,8] = 0
 accuracy_196_176: - 5.09765625 frac[196,176] + mem[196,176] = 0
 accuracy_144_86: - 5.09765625 frac[144,86] + mem[144,86] = 0
 accuracy_147_67: - 5.09765625 frac[147,67] + mem[147,67] = 0
 accuracy_153_168: - 5.09765625 frac[153,168] + mem[153,168] = 0
 accuracy_27_92: - 5.09765625 frac[27,92] + mem[27,92] = 0
 accuracy_194_122: - 5.09765625 frac[194,122] + mem[194,122] = 0
 accuracy_195_120: - 5.09765625 frac[195,120] + mem[195,120] = 0
 accuracy_122_82: - 5.09765625 frac[122,82] + mem[122,82] = 0
 accuracy_196_199: - 5.09765625 frac[196,199] + mem[196,199] = 0
 accuracy_197_223: - 5.09765625 frac[197,223] + mem[197,223] = 0
 accuracy_180_57: - 5.09765625 frac[180,57] + mem[180,57] = 0
 accuracy_198_231: - 5.09765625 frac[198,231] + mem[198,231] = 0
 accuracy_163_67: - 5.09765625 frac[163,67] + mem[163,67] = 0
 accuracy_192_12: - 5.09765625 frac[192,12] + mem[192,12] = 0
 accuracy_45_27: - 5.09765625 frac[45,27] + mem[45,27] = 0
 accuracy_195_9: - 5.09765625 frac[195,9] + mem[195,9] = 0
 accuracy_88_205: - 5.09765625 frac[88,205] + mem[88,205] = 0
 accuracy_141_125: - 5.09765625 frac[141,125] + mem[141,125] = 0
 accuracy_44_213: - 5.09765625 frac[44,213] + mem[44,213] = 0
 accuracy_192_187: - 5.09765625 frac[192,187] + mem[192,187] = 0
 accuracy_193_35: - 5.09765625 frac[193,35] + mem[193,35] = 0
 accuracy_46_202: - 5.09765625 frac[46,202] + mem[46,202] = 0
 accuracy_28_100: - 5.09765625 frac[28,100] + mem[28,100] = 0
 accuracy_157_93: - 5.09765625 frac[157,93] + mem[157,93] = 0
 accuracy_85_117: - 5.09765625 frac[85,117] + mem[85,117] = 0
 accuracy_87_156: - 5.09765625 frac[87,156] + mem[87,156] = 0
 accuracy_196_40: - 5.09765625 frac[196,40] + mem[196,40] = 0
 accuracy_188_175: - 5.09765625 frac[188,175] + mem[188,175] = 0
 accuracy_117_255: - 5.09765625 frac[117,255] + mem[117,255] = 0
 accuracy_8_200: - 5.09765625 frac[8,200] + mem[8,200] = 0
 accuracy_171_161: - 5.09765625 frac[171,161] + mem[171,161] = 0
 accuracy_193_220: - 5.09765625 frac[193,220] + mem[193,220] = 0
 accuracy_9_222: - 5.09765625 frac[9,222] + mem[9,222] = 0
 accuracy_195_208: - 5.09765625 frac[195,208] + mem[195,208] = 0
 accuracy_194_194: - 5.09765625 frac[194,194] + mem[194,194] = 0
 accuracy_196_191: - 5.09765625 frac[196,191] + mem[196,191] = 0
 accuracy_197_55: - 5.09765625 frac[197,55] + mem[197,55] = 0
 accuracy_153_177: - 5.09765625 frac[153,177] + mem[153,177] = 0
 accuracy_64_175: - 5.09765625 frac[64,175] + mem[64,175] = 0
 accuracy_27_85: - 5.09765625 frac[27,85] + mem[27,85] = 0
 accuracy_194_117: - 5.09765625 frac[194,117] + mem[194,117] = 0
 accuracy_195_97: - 5.09765625 frac[195,97] + mem[195,97] = 0
 accuracy_196_206: - 5.09765625 frac[196,206] + mem[196,206] = 0
 accuracy_197_216: - 5.09765625 frac[197,216] + mem[197,216] = 0
 accuracy_198_222: - 5.09765625 frac[198,222] + mem[198,222] = 0
 accuracy_42_33: - 5.09765625 frac[42,33] + mem[42,33] = 0
 accuracy_192_3: - 5.09765625 frac[192,3] + mem[192,3] = 0
 accuracy_10_54: - 5.09765625 frac[10,54] + mem[10,54] = 0
 accuracy_195_14: - 5.09765625 frac[195,14] + mem[195,14] = 0
 accuracy_89_202: - 5.09765625 frac[89,202] + mem[89,202] = 0
 accuracy_198_105: - 5.09765625 frac[198,105] + mem[198,105] = 0
 accuracy_181_47: - 5.09765625 frac[181,47] + mem[181,47] = 0
 accuracy_146_3: - 5.09765625 frac[146,3] + mem[146,3] = 0
 accuracy_164_169: - 5.09765625 frac[164,169] + mem[164,169] = 0
 accuracy_20_227: - 5.09765625 frac[20,227] + mem[20,227] = 0
 accuracy_166_70: - 5.09765625 frac[166,70] + mem[166,70] = 0
 accuracy_99_33: - 5.09765625 frac[99,33] + mem[99,33] = 0
 accuracy_120_154: - 5.09765625 frac[120,154] + mem[120,154] = 0
 accuracy_193_36: - 5.09765625 frac[193,36] + mem[193,36] = 0
 accuracy_191_90: - 5.09765625 frac[191,90] + mem[191,90] = 0
 accuracy_194_170: - 5.09765625 frac[194,170] + mem[194,170] = 0
 accuracy_196_23: - 5.09765625 frac[196,23] + mem[196,23] = 0
 accuracy_161_19: - 5.09765625 frac[161,19] + mem[161,19] = 0
 accuracy_22_74: - 5.09765625 frac[22,74] + mem[22,74] = 0
 accuracy_170_195: - 5.09765625 frac[170,195] + mem[170,195] = 0
 accuracy_27_189: - 5.09765625 frac[27,189] + mem[27,189] = 0
 accuracy_193_213: - 5.09765625 frac[193,213] + mem[193,213] = 0
 accuracy_194_221: - 5.09765625 frac[194,221] + mem[194,221] = 0
 accuracy_123_225: - 5.09765625 frac[123,225] + mem[123,225] = 0
 accuracy_47_182: - 5.09765625 frac[47,182] + mem[47,182] = 0
 accuracy_196_166: - 5.09765625 frac[196,166] + mem[196,166] = 0
 accuracy_197_48: - 5.09765625 frac[197,48] + mem[197,48] = 0
 accuracy_195_217: - 5.09765625 frac[195,217] + mem[195,217] = 0
 accuracy_195_102: - 5.09765625 frac[195,102] + mem[195,102] = 0
 accuracy_14_227: - 5.09765625 frac[14,227] + mem[14,227] = 0
 accuracy_197_193: - 5.09765625 frac[197,193] + mem[197,193] = 0
 accuracy_198_209: - 5.09765625 frac[198,209] + mem[198,209] = 0
 accuracy_163_113: - 5.09765625 frac[163,113] + mem[163,113] = 0
 accuracy_0_72: - 5.09765625 frac[0,72] + mem[0,72] = 0
 accuracy_18_158: - 5.09765625 frac[18,158] + mem[18,158] = 0
 accuracy_148_48: - 5.09765625 frac[148,48] + mem[148,48] = 0
 accuracy_24_126: - 5.09765625 frac[24,126] + mem[24,126] = 0
 accuracy_198_96: - 5.09765625 frac[198,96] + mem[198,96] = 0
 accuracy_166_121: - 5.09765625 frac[166,121] + mem[166,121] = 0
 accuracy_115_249: - 5.09765625 frac[115,249] + mem[115,249] = 0
 accuracy_25_225: - 5.09765625 frac[25,225] + mem[25,225] = 0
 accuracy_193_61: - 5.09765625 frac[193,61] + mem[193,61] = 0
 accuracy_191_67: - 5.09765625 frac[191,67] + mem[191,67] = 0
 accuracy_194_165: - 5.09765625 frac[194,165] + mem[194,165] = 0
 accuracy_50_43: - 5.09765625 frac[50,43] + mem[50,43] = 0
 accuracy_196_30: - 5.09765625 frac[196,30] + mem[196,30] = 0
 accuracy_145_178: - 5.09765625 frac[145,178] + mem[145,178] = 0
 accuracy_109_129: - 5.09765625 frac[109,129] + mem[109,129] = 0
 accuracy_78_30: - 5.09765625 frac[78,30] + mem[78,30] = 0
 accuracy_155_121: - 5.09765625 frac[155,121] + mem[155,121] = 0
 accuracy_193_174: - 5.09765625 frac[193,174] + mem[193,174] = 0
 accuracy_194_212: - 5.09765625 frac[194,212] + mem[194,212] = 0
 accuracy_195_222: - 5.09765625 frac[195,222] + mem[195,222] = 0
 accuracy_196_173: - 5.09765625 frac[196,173] + mem[196,173] = 0
 accuracy_197_57: - 5.09765625 frac[197,57] + mem[197,57] = 0
 accuracy_198_185: - 5.09765625 frac[198,185] + mem[198,185] = 0
 accuracy_52_170: - 5.09765625 frac[52,170] + mem[52,170] = 0
 accuracy_35_210: - 5.09765625 frac[35,210] + mem[35,210] = 0
 accuracy_55_143: - 5.09765625 frac[55,143] + mem[55,143] = 0
 accuracy_75_142: - 5.09765625 frac[75,142] + mem[75,142] = 0
 accuracy_148_72: - 5.09765625 frac[148,72] + mem[148,72] = 0
 accuracy_73_107: - 5.09765625 frac[73,107] + mem[73,107] = 0
 accuracy_194_71: - 5.09765625 frac[194,71] + mem[194,71] = 0
 accuracy_105_237: - 5.09765625 frac[105,237] + mem[105,237] = 0
 accuracy_195_111: - 5.09765625 frac[195,111] + mem[195,111] = 0
 accuracy_197_202: - 5.09765625 frac[197,202] + mem[197,202] = 0
 accuracy_192_17: - 5.09765625 frac[192,17] + mem[192,17] = 0
 accuracy_135_132: - 5.09765625 frac[135,132] + mem[135,132] = 0
 accuracy_158_240: - 5.09765625 frac[158,240] + mem[158,240] = 0
 accuracy_13_93: - 5.09765625 frac[13,93] + mem[13,93] = 0
 accuracy_198_91: - 5.09765625 frac[198,91] + mem[198,91] = 0
 accuracy_96_34: - 5.09765625 frac[96,34] + mem[96,34] = 0
 accuracy_97_148: - 5.09765625 frac[97,148] + mem[97,148] = 0
 accuracy_4_107: - 5.09765625 frac[4,107] + mem[4,107] = 0
 accuracy_192_128: - 5.09765625 frac[192,128] + mem[192,128] = 0
 accuracy_193_54: - 5.09765625 frac[193,54] + mem[193,54] = 0
 accuracy_174_16: - 5.09765625 frac[174,16] + mem[174,16] = 0
 accuracy_194_188: - 5.09765625 frac[194,188] + mem[194,188] = 0
 accuracy_104_34: - 5.09765625 frac[104,34] + mem[104,34] = 0
 accuracy_196_5: - 5.09765625 frac[196,5] + mem[196,5] = 0
 accuracy_142_249: - 5.09765625 frac[142,249] + mem[142,249] = 0
 accuracy_76_184: - 5.09765625 frac[76,184] + mem[76,184] = 0
 accuracy_130_74: - 5.09765625 frac[130,74] + mem[130,74] = 0
 accuracy_193_167: - 5.09765625 frac[193,167] + mem[193,167] = 0
 accuracy_194_47: - 5.09765625 frac[194,47] + mem[194,47] = 0
 accuracy_195_199: - 5.09765625 frac[195,199] + mem[195,199] = 0
 accuracy_196_148: - 5.09765625 frac[196,148] + mem[196,148] = 0
 accuracy_197_34: - 5.09765625 frac[197,34] + mem[197,34] = 0
 accuracy_15_47: - 5.09765625 frac[15,47] + mem[15,47] = 0
 accuracy_198_176: - 5.09765625 frac[198,176] + mem[198,176] = 0
 accuracy_53_152: - 5.09765625 frac[53,152] + mem[53,152] = 0
 accuracy_56_180: - 5.09765625 frac[56,180] + mem[56,180] = 0
 accuracy_2_48: - 5.09765625 frac[2,48] + mem[2,48] = 0
 accuracy_58_213: - 5.09765625 frac[58,213] + mem[58,213] = 0
 accuracy_62_78: - 5.09765625 frac[62,78] + mem[62,78] = 0
 accuracy_136_204: - 5.09765625 frac[136,204] + mem[136,204] = 0
 accuracy_173_57: - 5.09765625 frac[173,57] + mem[173,57] = 0
 accuracy_154_247: - 5.09765625 frac[154,247] + mem[154,247] = 0
 accuracy_194_94: - 5.09765625 frac[194,94] + mem[194,94] = 0
 accuracy_195_84: - 5.09765625 frac[195,84] + mem[195,84] = 0
 accuracy_197_179: - 5.09765625 frac[197,179] + mem[197,179] = 0
 accuracy_198_195: - 5.09765625 frac[198,195] + mem[198,195] = 0
 accuracy_187_101: - 5.09765625 frac[187,101] + mem[187,101] = 0
 accuracy_61_217: - 5.09765625 frac[61,217] + mem[61,217] = 0
 accuracy_192_104: - 5.09765625 frac[192,104] + mem[192,104] = 0
 accuracy_107_135: - 5.09765625 frac[107,135] + mem[107,135] = 0
 accuracy_198_82: - 5.09765625 frac[198,82] + mem[198,82] = 0
 accuracy_109_226: - 5.09765625 frac[109,226] + mem[109,226] = 0
 accuracy_182_184: - 5.09765625 frac[182,184] + mem[182,184] = 0
 accuracy_93_168: - 5.09765625 frac[93,168] + mem[93,168] = 0
 accuracy_192_135: - 5.09765625 frac[192,135] + mem[192,135] = 0
 accuracy_193_15: - 5.09765625 frac[193,15] + mem[193,15] = 0
 accuracy_194_183: - 5.09765625 frac[194,183] + mem[194,183] = 0
 accuracy_196_12: - 5.09765625 frac[196,12] + mem[196,12] = 0
 accuracy_2_152: - 5.09765625 frac[2,152] + mem[2,152] = 0
 accuracy_166_250: - 5.09765625 frac[166,250] + mem[166,250] = 0
 accuracy_4_225: - 5.09765625 frac[4,225] + mem[4,225] = 0
 accuracy_169_38: - 5.09765625 frac[169,38] + mem[169,38] = 0
 accuracy_99_165: - 5.09765625 frac[99,165] + mem[99,165] = 0
 accuracy_100_61: - 5.09765625 frac[100,61] + mem[100,61] = 0
 accuracy_27_128: - 5.09765625 frac[27,128] + mem[27,128] = 0
 accuracy_193_184: - 5.09765625 frac[193,184] + mem[193,184] = 0
 accuracy_194_38: - 5.09765625 frac[194,38] + mem[194,38] = 0
 accuracy_195_204: - 5.09765625 frac[195,204] + mem[195,204] = 0
 accuracy_68_70: - 5.09765625 frac[68,70] + mem[68,70] = 0
 accuracy_196_147: - 5.09765625 frac[196,147] + mem[196,147] = 0
 accuracy_197_43: - 5.09765625 frac[197,43] + mem[197,43] = 0
 accuracy_198_171: - 5.09765625 frac[198,171] + mem[198,171] = 0
 accuracy_173_161: - 5.09765625 frac[173,161] + mem[173,161] = 0
 accuracy_45_45: - 5.09765625 frac[45,45] + mem[45,45] = 0
 accuracy_132_240: - 5.09765625 frac[132,240] + mem[132,240] = 0
 accuracy_134_89: - 5.09765625 frac[134,89] + mem[134,89] = 0
 accuracy_7_229: - 5.09765625 frac[7,229] + mem[7,229] = 0
 accuracy_62_113: - 5.09765625 frac[62,113] + mem[62,113] = 0
 accuracy_10_91: - 5.09765625 frac[10,91] + mem[10,91] = 0
 accuracy_194_89: - 5.09765625 frac[194,89] + mem[194,89] = 0
 accuracy_195_93: - 5.09765625 frac[195,93] + mem[195,93] = 0
 accuracy_178_191: - 5.09765625 frac[178,191] + mem[178,191] = 0
 accuracy_197_188: - 5.09765625 frac[197,188] + mem[197,188] = 0
 accuracy_198_58: - 5.09765625 frac[198,58] + mem[198,58] = 0
 accuracy_21_142: - 5.09765625 frac[21,142] + mem[21,142] = 0
 accuracy_3_56: - 5.09765625 frac[3,56] + mem[3,56] = 0
 accuracy_192_111: - 5.09765625 frac[192,111] + mem[192,111] = 0
 accuracy_138_231: - 5.09765625 frac[138,231] + mem[138,231] = 0
 accuracy_159_240: - 5.09765625 frac[159,240] + mem[159,240] = 0
 accuracy_161_78: - 5.09765625 frac[161,78] + mem[161,78] = 0
 accuracy_198_85: - 5.09765625 frac[198,85] + mem[198,85] = 0
 accuracy_108_211: - 5.09765625 frac[108,211] + mem[108,211] = 0
 accuracy_75_87: - 5.09765625 frac[75,87] + mem[75,87] = 0
 accuracy_20_135: - 5.09765625 frac[20,135] + mem[20,135] = 0
 accuracy_111_198: - 5.09765625 frac[111,198] + mem[111,198] = 0
 accuracy_113_152: - 5.09765625 frac[113,152] + mem[113,152] = 0
 accuracy_188_219: - 5.09765625 frac[188,219] + mem[188,219] = 0
 accuracy_192_158: - 5.09765625 frac[192,158] + mem[192,158] = 0
 accuracy_193_0: - 5.09765625 frac[193,0] + mem[193,0] = 0
 accuracy_194_142: - 5.09765625 frac[194,142] + mem[194,142] = 0
 accuracy_65_93: - 5.09765625 frac[65,93] + mem[65,93] = 0
 accuracy_196_11: - 5.09765625 frac[196,11] + mem[196,11] = 0
 accuracy_33_116: - 5.09765625 frac[33,116] + mem[33,116] = 0
 accuracy_161_63: - 5.09765625 frac[161,63] + mem[161,63] = 0
 accuracy_56_211: - 5.09765625 frac[56,211] + mem[56,211] = 0
 accuracy_1_99: - 5.09765625 frac[1,99] + mem[1,99] = 0
 accuracy_182_34: - 5.09765625 frac[182,34] + mem[182,34] = 0
 accuracy_8_235: - 5.09765625 frac[8,235] + mem[8,235] = 0
 accuracy_193_177: - 5.09765625 frac[193,177] + mem[193,177] = 0
 accuracy_194_33: - 5.09765625 frac[194,33] + mem[194,33] = 0
 accuracy_195_181: - 5.09765625 frac[195,181] + mem[195,181] = 0
 accuracy_196_154: - 5.09765625 frac[196,154] + mem[196,154] = 0
 accuracy_197_20: - 5.09765625 frac[197,20] + mem[197,20] = 0
 accuracy_198_162: - 5.09765625 frac[198,162] + mem[198,162] = 0
 accuracy_78_226: - 5.09765625 frac[78,226] + mem[78,226] = 0
 accuracy_118_117: - 5.09765625 frac[118,117] + mem[118,117] = 0
 accuracy_10_66: - 5.09765625 frac[10,66] + mem[10,66] = 0
 accuracy_83_213: - 5.09765625 frac[83,213] + mem[83,213] = 0
 accuracy_174_252: - 5.09765625 frac[174,252] + mem[174,252] = 0
 accuracy_195_66: - 5.09765625 frac[195,66] + mem[195,66] = 0
 accuracy_194_80: - 5.09765625 frac[194,80] + mem[194,80] = 0
 accuracy_14_255: - 5.09765625 frac[14,255] + mem[14,255] = 0
 accuracy_197_165: - 5.09765625 frac[197,165] + mem[197,165] = 0
 accuracy_198_61: - 5.09765625 frac[198,61] + mem[198,61] = 0
 accuracy_77_26: - 5.09765625 frac[77,26] + mem[77,26] = 0
 accuracy_192_102: - 5.09765625 frac[192,102] + mem[192,102] = 0
 accuracy_193_104: - 5.09765625 frac[193,104] + mem[193,104] = 0
 accuracy_120_78: - 5.09765625 frac[120,78] + mem[120,78] = 0
 accuracy_198_76: - 5.09765625 frac[198,76] + mem[198,76] = 0
 accuracy_113_145: - 5.09765625 frac[113,145] + mem[113,145] = 0
 accuracy_186_185: - 5.09765625 frac[186,185] + mem[186,185] = 0
 accuracy_40_58: - 5.09765625 frac[40,58] + mem[40,58] = 0
 accuracy_116_186: - 5.09765625 frac[116,186] + mem[116,186] = 0
 accuracy_100_92: - 5.09765625 frac[100,92] + mem[100,92] = 0
 accuracy_192_149: - 5.09765625 frac[192,149] + mem[192,149] = 0
 accuracy_194_137: - 5.09765625 frac[194,137] + mem[194,137] = 0
 accuracy_196_114: - 5.09765625 frac[196,114] + mem[196,114] = 0
 accuracy_107_2: - 5.09765625 frac[107,2] + mem[107,2] = 0
 accuracy_71_24: - 5.09765625 frac[71,24] + mem[71,24] = 0
 accuracy_146_241: - 5.09765625 frac[146,241] + mem[146,241] = 0
 accuracy_96_166: - 5.09765625 frac[96,166] + mem[96,166] = 0
 accuracy_193_138: - 5.09765625 frac[193,138] + mem[193,138] = 0
 accuracy_191_244: - 5.09765625 frac[191,244] + mem[191,244] = 0
 accuracy_195_186: - 5.09765625 frac[195,186] + mem[195,186] = 0
 accuracy_194_56: - 5.09765625 frac[194,56] + mem[194,56] = 0
 accuracy_196_129: - 5.09765625 frac[196,129] + mem[196,129] = 0
 accuracy_179_75: - 5.09765625 frac[179,75] + mem[179,75] = 0
 accuracy_198_165: - 5.09765625 frac[198,165] + mem[198,165] = 0
 accuracy_17_100: - 5.09765625 frac[17,100] + mem[17,100] = 0
 accuracy_197_29: - 5.09765625 frac[197,29] + mem[197,29] = 0
 accuracy_103_86: - 5.09765625 frac[103,86] + mem[103,86] = 0
 accuracy_57_240: - 5.09765625 frac[57,240] + mem[57,240] = 0
 accuracy_170_49: - 5.09765625 frac[170,49] + mem[170,49] = 0
 accuracy_153_159: - 5.09765625 frac[153,159] + mem[153,159] = 0
 accuracy_138_6: - 5.09765625 frac[138,6] + mem[138,6] = 0
 accuracy_195_75: - 5.09765625 frac[195,75] + mem[195,75] = 0
 accuracy_12_230: - 5.09765625 frac[12,230] + mem[12,230] = 0
 accuracy_161_239: - 5.09765625 frac[161,239] + mem[161,239] = 0
 accuracy_197_174: - 5.09765625 frac[197,174] + mem[197,174] = 0
 accuracy_198_52: - 5.09765625 frac[198,52] + mem[198,52] = 0
 accuracy_178_177: - 5.09765625 frac[178,177] + mem[178,177] = 0
 accuracy_128_16: - 5.09765625 frac[128,16] + mem[128,16] = 0
 accuracy_37_250: - 5.09765625 frac[37,250] + mem[37,250] = 0
 accuracy_112_181: - 5.09765625 frac[112,181] + mem[112,181] = 0
 accuracy_59_69: - 5.09765625 frac[59,69] + mem[59,69] = 0
 accuracy_132_117: - 5.09765625 frac[132,117] + mem[132,117] = 0
 accuracy_79_142: - 5.09765625 frac[79,142] + mem[79,142] = 0
 accuracy_193_97: - 5.09765625 frac[193,97] + mem[193,97] = 0
 accuracy_157_155: - 5.09765625 frac[157,155] + mem[157,155] = 0
 accuracy_140_245: - 5.09765625 frac[140,245] + mem[140,245] = 0
 accuracy_126_111: - 5.09765625 frac[126,111] + mem[126,111] = 0
 accuracy_198_71: - 5.09765625 frac[198,71] + mem[198,71] = 0
 accuracy_15_216: - 5.09765625 frac[15,216] + mem[15,216] = 0
 accuracy_74_113: - 5.09765625 frac[74,113] + mem[74,113] = 0
 accuracy_38_75: - 5.09765625 frac[38,75] + mem[38,75] = 0
 accuracy_3_183: - 5.09765625 frac[3,183] + mem[3,183] = 0
 accuracy_109_205: - 5.09765625 frac[109,205] + mem[109,205] = 0
 accuracy_93_179: - 5.09765625 frac[93,179] + mem[93,179] = 0
 accuracy_39_91: - 5.09765625 frac[39,91] + mem[39,91] = 0
 accuracy_9_156: - 5.09765625 frac[9,156] + mem[9,156] = 0
 accuracy_192_236: - 5.09765625 frac[192,236] + mem[192,236] = 0
 accuracy_193_18: - 5.09765625 frac[193,18] + mem[193,18] = 0
 accuracy_194_128: - 5.09765625 frac[194,128] + mem[194,128] = 0
 accuracy_121_58: - 5.09765625 frac[121,58] + mem[121,58] = 0
 accuracy_50_198: - 5.09765625 frac[50,198] + mem[50,198] = 0
 accuracy_196_121: - 5.09765625 frac[196,121] + mem[196,121] = 0
 accuracy_197_117: - 5.09765625 frac[197,117] + mem[197,117] = 0
 accuracy_125_77: - 5.09765625 frac[125,77] + mem[125,77] = 0
 accuracy_65_47: - 5.09765625 frac[65,47] + mem[65,47] = 0
 accuracy_144_15: - 5.09765625 frac[144,15] + mem[144,15] = 0
 accuracy_95_159: - 5.09765625 frac[95,159] + mem[95,159] = 0
 accuracy_150_21: - 5.09765625 frac[150,21] + mem[150,21] = 0
 accuracy_153_247: - 5.09765625 frac[153,247] + mem[153,247] = 0
 accuracy_193_131: - 5.09765625 frac[193,131] + mem[193,131] = 0
 accuracy_176_197: - 5.09765625 frac[176,197] + mem[176,197] = 0
 accuracy_195_163: - 5.09765625 frac[195,163] + mem[195,163] = 0
 accuracy_29_242: - 5.09765625 frac[29,242] + mem[29,242] = 0
 accuracy_196_136: - 5.09765625 frac[196,136] + mem[196,136] = 0
 accuracy_197_6: - 5.09765625 frac[197,6] + mem[197,6] = 0
 accuracy_198_156: - 5.09765625 frac[198,156] + mem[198,156] = 0
 accuracy_19_68: - 5.09765625 frac[19,68] + mem[19,68] = 0
 accuracy_150_132: - 5.09765625 frac[150,132] + mem[150,132] = 0
 accuracy_138_1: - 5.09765625 frac[138,1] + mem[138,1] = 0
 accuracy_195_48: - 5.09765625 frac[195,48] + mem[195,48] = 0
 accuracy_178_168: - 5.09765625 frac[178,168] + mem[178,168] = 0
 accuracy_197_151: - 5.09765625 frac[197,151] + mem[197,151] = 0
 accuracy_198_47: - 5.09765625 frac[198,47] + mem[198,47] = 0
 accuracy_162_155: - 5.09765625 frac[162,155] + mem[162,155] = 0
 accuracy_192_116: - 5.09765625 frac[192,116] + mem[192,116] = 0
 accuracy_193_122: - 5.09765625 frac[193,122] + mem[193,122] = 0
 accuracy_45_99: - 5.09765625 frac[45,99] + mem[45,99] = 0
 accuracy_161_81: - 5.09765625 frac[161,81] + mem[161,81] = 0
 accuracy_113_131: - 5.09765625 frac[113,131] + mem[113,131] = 0
 accuracy_192_227: - 5.09765625 frac[192,227] + mem[192,227] = 0
 accuracy_193_235: - 5.09765625 frac[193,235] + mem[193,235] = 0
 accuracy_191_149: - 5.09765625 frac[191,149] + mem[191,149] = 0
 accuracy_194_155: - 5.09765625 frac[194,155] + mem[194,155] = 0
 accuracy_196_96: - 5.09765625 frac[196,96] + mem[196,96] = 0
 accuracy_197_126: - 5.09765625 frac[197,126] + mem[197,126] = 0
 accuracy_33_31: - 5.09765625 frac[33,31] + mem[33,31] = 0
 accuracy_147_19: - 5.09765625 frac[147,19] + mem[147,19] = 0
 accuracy_57_145: - 5.09765625 frac[57,145] + mem[57,145] = 0
 accuracy_169_2: - 5.09765625 frac[169,2] + mem[169,2] = 0
 accuracy_98_168: - 5.09765625 frac[98,168] + mem[98,168] = 0
 accuracy_152_198: - 5.09765625 frac[152,198] + mem[152,198] = 0
 accuracy_8_128: - 5.09765625 frac[8,128] + mem[8,128] = 0
 accuracy_193_132: - 5.09765625 frac[193,132] + mem[193,132] = 0
 accuracy_194_10: - 5.09765625 frac[194,10] + mem[194,10] = 0
 accuracy_195_168: - 5.09765625 frac[195,168] + mem[195,168] = 0
 accuracy_196_247: - 5.09765625 frac[196,247] + mem[196,247] = 0
 accuracy_197_15: - 5.09765625 frac[197,15] + mem[197,15] = 0
 accuracy_198_151: - 5.09765625 frac[198,151] + mem[198,151] = 0
 accuracy_19_77: - 5.09765625 frac[19,77] + mem[19,77] = 0
 accuracy_172_136: - 5.09765625 frac[172,136] + mem[172,136] = 0
 accuracy_195_57: - 5.09765625 frac[195,57] + mem[195,57] = 0
 accuracy_197_144: - 5.09765625 frac[197,144] + mem[197,144] = 0
 accuracy_198_38: - 5.09765625 frac[198,38] + mem[198,38] = 0
 accuracy_16_223: - 5.09765625 frac[16,223] + mem[16,223] = 0
 accuracy_6_3: - 5.09765625 frac[6,3] + mem[6,3] = 0
 accuracy_192_75: - 5.09765625 frac[192,75] + mem[192,75] = 0
 accuracy_135_222: - 5.09765625 frac[135,222] + mem[135,222] = 0
 accuracy_31_25: - 5.09765625 frac[31,25] + mem[31,25] = 0
 accuracy_106_90: - 5.09765625 frac[106,90] + mem[106,90] = 0
 accuracy_93_133: - 5.09765625 frac[93,133] + mem[93,133] = 0
 accuracy_188_255: - 5.09765625 frac[188,255] + mem[188,255] = 0
 accuracy_9_142: - 5.09765625 frac[9,142] + mem[9,142] = 0
 accuracy_192_250: - 5.09765625 frac[192,250] + mem[192,250] = 0
 accuracy_48_95: - 5.09765625 frac[48,95] + mem[48,95] = 0
 accuracy_104_12: - 5.09765625 frac[104,12] + mem[104,12] = 0
 accuracy_194_146: - 5.09765625 frac[194,146] + mem[194,146] = 0
 accuracy_174_62: - 5.09765625 frac[174,62] + mem[174,62] = 0
 accuracy_196_111: - 5.09765625 frac[196,111] + mem[196,111] = 0
 accuracy_81_153: - 5.09765625 frac[81,153] + mem[81,153] = 0
 accuracy_172_240: - 5.09765625 frac[172,240] + mem[172,240] = 0
 accuracy_27_101: - 5.09765625 frac[27,101] + mem[27,101] = 0
 accuracy_28_218: - 5.09765625 frac[28,218] + mem[28,218] = 0
 accuracy_194_5: - 5.09765625 frac[194,5] + mem[194,5] = 0
 accuracy_195_145: - 5.09765625 frac[195,145] + mem[195,145] = 0
 accuracy_86_31: - 5.09765625 frac[86,31] + mem[86,31] = 0
 accuracy_196_254: - 5.09765625 frac[196,254] + mem[196,254] = 0
 accuracy_197_8: - 5.09765625 frac[197,8] + mem[197,8] = 0
 accuracy_198_142: - 5.09765625 frac[198,142] + mem[198,142] = 0
 accuracy_111_13: - 5.09765625 frac[111,13] + mem[111,13] = 0
 accuracy_80_164: - 5.09765625 frac[80,164] + mem[80,164] = 0
 accuracy_61_158: - 5.09765625 frac[61,158] + mem[61,158] = 0
 accuracy_100_175: - 5.09765625 frac[100,175] + mem[100,175] = 0
 accuracy_101_39: - 5.09765625 frac[101,39] + mem[101,39] = 0
 accuracy_67_107: - 5.09765625 frac[67,107] + mem[67,107] = 0
 accuracy_195_62: - 5.09765625 frac[195,62] + mem[195,62] = 0
 accuracy_86_142: - 5.09765625 frac[86,142] + mem[86,142] = 0
 accuracy_141_38: - 5.09765625 frac[141,38] + mem[141,38] = 0
 accuracy_197_153: - 5.09765625 frac[197,153] + mem[197,153] = 0
 accuracy_198_25: - 5.09765625 frac[198,25] + mem[198,25] = 0
 accuracy_60_90: - 5.09765625 frac[60,90] + mem[60,90] = 0
 accuracy_192_66: - 5.09765625 frac[192,66] + mem[192,66] = 0
 accuracy_193_116: - 5.09765625 frac[193,116] + mem[193,116] = 0
 mem_tot_CPU1: - mem[0,225] - mem[0,116] - mem[0,1] - mem[0,76]
   - mem[0,213] - mem[0,136] - mem[0,252] - mem[0,202] - mem[0,39]
   - mem[0,53] - mem[0,72] + mem_tot_CPU1 = 0
 rows_tot_CPU1: - 3 frac[0,225] - 3 frac[0,116] - 3 frac[0,1]
   - 3 frac[0,76] - 3 frac[0,213] - 3 frac[0,136] - 3 frac[0,252]
   - 3 frac[0,202] - 3 frac[0,39] - 3 frac[0,53] - 3 frac[0,72]
   + rows_tot_CPU1 = 0
 mem_tot_CPU2: - mem[1,196] - mem[1,57] - mem[1,213] - mem[1,46]
   - mem[1,48] - mem[1,120] - mem[1,113] - mem[1,159] - mem[1,99]
   + mem_tot_CPU2 = 0
 rows_tot_CPU2: - 3 frac[1,196] - 3 frac[1,57] - 3 frac[1,213]
   - 3 frac[1,46] - 3 frac[1,48] - 3 frac[1,120] - 3 frac[1,113]
   - 3 frac[1,159] - 3 frac[1,99] + rows_tot_CPU2 = 0
 mem_tot_CPU3: - mem[2,135] - mem[2,14] - mem[2,27] - mem[2,76] - mem[2,52]
   - mem[2,96] - mem[2,200] - mem[2,123] - mem[2,229] - mem[2,92]
   - mem[2,164] - mem[2,253] - mem[2,244] - mem[2,5] - mem[2,109]
   - mem[2,127] - mem[2,208] - mem[2,48] - mem[2,152] + mem_tot_CPU3 = 0
 rows_tot_CPU3: - 3 frac[2,135] - 3 frac[2,14] - 3 frac[2,27]
   - 3 frac[2,76] - 3 frac[2,52] - 3 frac[2,96] - 3 frac[2,200]
   - 3 frac[2,123] - 3 frac[2,229] - 3 frac[2,92] - 3 frac[2,164]
   - 3 frac[2,253] - 3 frac[2,244] - 3 frac[2,5] - 3 frac[2,109]
   - 3 frac[2,127] - 3 frac[2,208] - 3 frac[2,48] - 3 frac[2,152]
   + rows_tot_CPU3 = 0
 mem_tot_CPU4: - mem[3,152] - mem[3,123] - mem[3,25] - mem[3,24]
   - mem[3,151] - mem[3,133] - mem[3,62] - mem[3,56] - mem[3,183]
   + mem_tot_CPU4 = 0
 rows_tot_CPU4: - 3 frac[3,152] - 3 frac[3,123] - 3 frac[3,25]
   - 3 frac[3,24] - 3 frac[3,151] - 3 frac[3,133] - 3 frac[3,62]
   - 3 frac[3,56] - 3 frac[3,183] + rows_tot_CPU4 = 0
 mem_tot_CPU5: - mem[4,90] - mem[4,188] - mem[4,142] - mem[4,193]
   - mem[4,107] - mem[4,225] + mem_tot_CPU5 = 0
 rows_tot_CPU5: - 3 frac[4,90] - 3 frac[4,188] - 3 frac[4,142]
   - 3 frac[4,193] - 3 frac[4,107] - 3 frac[4,225] + rows_tot_CPU5 = 0
 mem_tot_CPU6: - mem[5,242] - mem[5,77] - mem[5,37] - mem[5,16] - mem[5,25]
   - mem[5,102] - mem[5,247] - mem[5,101] - mem[5,144] + mem_tot_CPU6 = 0
 rows_tot_CPU6: - 3 frac[5,242] - 3 frac[5,77] - 3 frac[5,37]
   - 3 frac[5,16] - 3 frac[5,25] - 3 frac[5,102] - 3 frac[5,247]
   - 3 frac[5,101] - 3 frac[5,144] + rows_tot_CPU6 = 0
 mem_tot_CPU7: - mem[6,104] - mem[6,90] - mem[6,189] - mem[6,136]
   - mem[6,66] - mem[6,232] - mem[6,114] - mem[6,47] - mem[6,11]
   - mem[6,190] - mem[6,3] + mem_tot_CPU7 = 0
 rows_tot_CPU7: - 3 frac[6,104] - 3 frac[6,90] - 3 frac[6,189]
   - 3 frac[6,136] - 3 frac[6,66] - 3 frac[6,232] - 3 frac[6,114]
   - 3 frac[6,47] - 3 frac[6,11] - 3 frac[6,190] - 3 frac[6,3]
   + rows_tot_CPU7 = 0
 mem_tot_CPU8: - mem[7,104] - mem[7,246] - mem[7,133] - mem[7,153]
   - mem[7,46] - mem[7,162] - mem[7,75] - mem[7,216] - mem[7,164]
   - mem[7,147] - mem[7,123] - mem[7,229] + mem_tot_CPU8 = 0
 rows_tot_CPU8: - 3 frac[7,104] - 3 frac[7,246] - 3 frac[7,133]
   - 3 frac[7,153] - 3 frac[7,46] - 3 frac[7,162] - 3 frac[7,75]
   - 3 frac[7,216] - 3 frac[7,164] - 3 frac[7,147] - 3 frac[7,123]
   - 3 frac[7,229] + rows_tot_CPU8 = 0
 mem_tot_CPU9: - mem[8,201] - mem[8,123] - mem[8,249] - mem[8,243]
   - mem[8,7] - mem[8,157] - mem[8,167] - mem[8,244] - mem[8,15]
   - mem[8,227] - mem[8,2] - mem[8,152] - mem[8,48] - mem[8,169]
   - mem[8,200] - mem[8,235] - mem[8,128] + mem_tot_CPU9 = 0
 rows_tot_CPU9: - 3 frac[8,201] - 3 frac[8,123] - 3 frac[8,249]
   - 3 frac[8,243] - 3 frac[8,7] - 3 frac[8,157] - 3 frac[8,167]
   - 3 frac[8,244] - 3 frac[8,15] - 3 frac[8,227] - 3 frac[8,2]
   - 3 frac[8,152] - 3 frac[8,48] - 3 frac[8,169] - 3 frac[8,200]
   - 3 frac[8,235] - 3 frac[8,128] + rows_tot_CPU9 = 0
 mem_tot_CPU10: - mem[9,0] - mem[9,50] - mem[9,74] - mem[9,236]
   - mem[9,111] - mem[9,8] - mem[9,220] - mem[9,73] - mem[9,48] - mem[9,57]
   - mem[9,255] - mem[9,70] - mem[9,158] - mem[9,252] - mem[9,238]
   - mem[9,222] - mem[9,156] - mem[9,142] + mem_tot_CPU10 = 0
 rows_tot_CPU10: - 3 frac[9,0] - 3 frac[9,50] - 3 frac[9,74]
   - 3 frac[9,236] - 3 frac[9,111] - 3 frac[9,8] - 3 frac[9,220]
   - 3 frac[9,73] - 3 frac[9,48] - 3 frac[9,57] - 3 frac[9,255]
   - 3 frac[9,70] - 3 frac[9,158] - 3 frac[9,252] - 3 frac[9,238]
   - 3 frac[9,222] - 3 frac[9,156] - 3 frac[9,142] + rows_tot_CPU10 = 0
 mem_tot_CPU11: - mem[10,220] - mem[10,103] - mem[10,159] - mem[10,27]
   - mem[10,142] - mem[10,3] - mem[10,47] - mem[10,79] - mem[10,144]
   - mem[10,180] - mem[10,168] - mem[10,90] - mem[10,145] - mem[10,181]
   - mem[10,54] - mem[10,91] - mem[10,66] + mem_tot_CPU11 = 0
 rows_tot_CPU11: - 3 frac[10,220] - 3 frac[10,103] - 3 frac[10,159]
   - 3 frac[10,27] - 3 frac[10,142] - 3 frac[10,3] - 3 frac[10,47]
   - 3 frac[10,79] - 3 frac[10,144] - 3 frac[10,180] - 3 frac[10,168]
   - 3 frac[10,90] - 3 frac[10,145] - 3 frac[10,181] - 3 frac[10,54]
   - 3 frac[10,91] - 3 frac[10,66] + rows_tot_CPU11 = 0
 mem_tot_CPU12: - mem[11,207] - mem[11,39] - mem[11,17] - mem[11,27]
   - mem[11,141] - mem[11,42] - mem[11,174] + mem_tot_CPU12 = 0
 rows_tot_CPU12: - 3 frac[11,207] - 3 frac[11,39] - 3 frac[11,17]
   - 3 frac[11,27] - 3 frac[11,141] - 3 frac[11,42] - 3 frac[11,174]
   + rows_tot_CPU12 = 0
 mem_tot_CPU13: - mem[12,241] - mem[12,25] - mem[12,34] - mem[12,208]
   - mem[12,253] - mem[12,36] - mem[12,102] - mem[12,180] - mem[12,128]
   - mem[12,230] + mem_tot_CPU13 = 0
 rows_tot_CPU13: - 3 frac[12,241] - 3 frac[12,25] - 3 frac[12,34]
   - 3 frac[12,208] - 3 frac[12,253] - 3 frac[12,36] - 3 frac[12,102]
   - 3 frac[12,180] - 3 frac[12,128] - 3 frac[12,230] + rows_tot_CPU13 = 0
 mem_tot_CPU14: - mem[13,215] - mem[13,208] - mem[13,185] - mem[13,236]
   - mem[13,23] - mem[13,170] - mem[13,154] - mem[13,15] - mem[13,37]
   - mem[13,0] - mem[13,93] + mem_tot_CPU14 = 0
 rows_tot_CPU14: - 3 frac[13,215] - 3 frac[13,208] - 3 frac[13,185]
   - 3 frac[13,236] - 3 frac[13,23] - 3 frac[13,170] - 3 frac[13,154]
   - 3 frac[13,15] - 3 frac[13,37] - 3 frac[13,0] - 3 frac[13,93]
   + rows_tot_CPU14 = 0
 mem_tot_CPU15: - mem[14,41] - mem[14,206] - mem[14,216] - mem[14,46]
   - mem[14,98] - mem[14,99] - mem[14,37] - mem[14,143] - mem[14,78]
   - mem[14,187] - mem[14,184] - mem[14,53] - mem[14,227] - mem[14,255]
   + mem_tot_CPU15 = 0
 rows_tot_CPU15: - 3 frac[14,41] - 3 frac[14,206] - 3 frac[14,216]
   - 3 frac[14,46] - 3 frac[14,98] - 3 frac[14,99] - 3 frac[14,37]
   - 3 frac[14,143] - 3 frac[14,78] - 3 frac[14,187] - 3 frac[14,184]
   - 3 frac[14,53] - 3 frac[14,227] - 3 frac[14,255] + rows_tot_CPU15 = 0
 mem_tot_CPU16: - mem[15,64] - mem[15,213] - mem[15,16] - mem[15,67]
   - mem[15,32] - mem[15,150] - mem[15,48] - mem[15,174] - mem[15,126]
   - mem[15,85] - mem[15,101] - mem[15,238] - mem[15,53] - mem[15,59]
   - mem[15,47] - mem[15,216] + mem_tot_CPU16 = 0
 rows_tot_CPU16: - 3 frac[15,64] - 3 frac[15,213] - 3 frac[15,16]
   - 3 frac[15,67] - 3 frac[15,32] - 3 frac[15,150] - 3 frac[15,48]
   - 3 frac[15,174] - 3 frac[15,126] - 3 frac[15,85] - 3 frac[15,101]
   - 3 frac[15,238] - 3 frac[15,53] - 3 frac[15,59] - 3 frac[15,47]
   - 3 frac[15,216] + rows_tot_CPU16 = 0
 mem_tot_CPU17: - mem[16,119] - mem[16,123] - mem[16,129] - mem[16,5]
   - mem[16,148] - mem[16,130] - mem[16,249] - mem[16,221] - mem[16,69]
   - mem[16,231] - mem[16,204] - mem[16,37] - mem[16,180] - mem[16,162]
   - mem[16,223] + mem_tot_CPU17 = 0
 rows_tot_CPU17: - 3 frac[16,119] - 3 frac[16,123] - 3 frac[16,129]
   - 3 frac[16,5] - 3 frac[16,148] - 3 frac[16,130] - 3 frac[16,249]
   - 3 frac[16,221] - 3 frac[16,69] - 3 frac[16,231] - 3 frac[16,204]
   - 3 frac[16,37] - 3 frac[16,180] - 3 frac[16,162] - 3 frac[16,223]
   + rows_tot_CPU17 = 0
 mem_tot_CPU18: - mem[17,57] - mem[17,105] - mem[17,181] - mem[17,242]
   - mem[17,21] - mem[17,96] - mem[17,75] - mem[17,192] - mem[17,88]
   - mem[17,81] - mem[17,68] - mem[17,32] - mem[17,73] - mem[17,198]
   - mem[17,4] - mem[17,108] - mem[17,100] + mem_tot_CPU18 = 0
 rows_tot_CPU18: - 3 frac[17,57] - 3 frac[17,105] - 3 frac[17,181]
   - 3 frac[17,242] - 3 frac[17,21] - 3 frac[17,96] - 3 frac[17,75]
   - 3 frac[17,192] - 3 frac[17,88] - 3 frac[17,81] - 3 frac[17,68]
   - 3 frac[17,32] - 3 frac[17,73] - 3 frac[17,198] - 3 frac[17,4]
   - 3 frac[17,108] - 3 frac[17,100] + rows_tot_CPU18 = 0
 mem_tot_CPU19: - mem[18,60] - mem[18,142] - mem[18,119] - mem[18,190]
   - mem[18,195] - mem[18,157] - mem[18,143] - mem[18,229] - mem[18,206]
   - mem[18,75] - mem[18,232] - mem[18,114] - mem[18,100] - mem[18,166]
   - mem[18,57] - mem[18,34] - mem[18,158] + mem_tot_CPU19 = 0
 rows_tot_CPU19: - 3 frac[18,60] - 3 frac[18,142] - 3 frac[18,119]
   - 3 frac[18,190] - 3 frac[18,195] - 3 frac[18,157] - 3 frac[18,143]
   - 3 frac[18,229] - 3 frac[18,206] - 3 frac[18,75] - 3 frac[18,232]
   - 3 frac[18,114] - 3 frac[18,100] - 3 frac[18,166] - 3 frac[18,57]
   - 3 frac[18,34] - 3 frac[18,158] + rows_tot_CPU19 = 0
 mem_tot_CPU20: - mem[19,244] - mem[19,2] - mem[19,224] - mem[19,55]
   - mem[19,102] - mem[19,68] - mem[19,77] + mem_tot_CPU20 = 0
 rows_tot_CPU20: - 3 frac[19,244] - 3 frac[19,2] - 3 frac[19,224]
   - 3 frac[19,55] - 3 frac[19,102] - 3 frac[19,68] - 3 frac[19,77]
   + rows_tot_CPU20 = 0
 mem_tot_CPU21: - mem[20,229] - mem[20,170] - mem[20,6] - mem[20,204]
   - mem[20,132] - mem[20,72] - mem[20,181] - mem[20,35] - mem[20,37]
   - mem[20,227] - mem[20,135] + mem_tot_CPU21 = 0
 rows_tot_CPU21: - 3 frac[20,229] - 3 frac[20,170] - 3 frac[20,6]
   - 3 frac[20,204] - 3 frac[20,132] - 3 frac[20,72] - 3 frac[20,181]
   - 3 frac[20,35] - 3 frac[20,37] - 3 frac[20,227] - 3 frac[20,135]
   + rows_tot_CPU21 = 0
 mem_tot_CPU22: - mem[21,37] - mem[21,104] - mem[21,161] - mem[21,54]
   - mem[21,215] - mem[21,160] - mem[21,204] - mem[21,233] - mem[21,3]
   - mem[21,211] - mem[21,47] - mem[21,49] - mem[21,142] + mem_tot_CPU22
   = 0
 rows_tot_CPU22: - 3 frac[21,37] - 3 frac[21,104] - 3 frac[21,161]
   - 3 frac[21,54] - 3 frac[21,215] - 3 frac[21,160] - 3 frac[21,204]
   - 3 frac[21,233] - 3 frac[21,3] - 3 frac[21,211] - 3 frac[21,47]
   - 3 frac[21,49] - 3 frac[21,142] + rows_tot_CPU22 = 0
 mem_tot_CPU23: - mem[22,166] - mem[22,92] - mem[22,50] - mem[22,152]
   - mem[22,191] - mem[22,253] - mem[22,188] - mem[22,65] - mem[22,242]
   - mem[22,74] + mem_tot_CPU23 = 0
 rows_tot_CPU23: - 3 frac[22,166] - 3 frac[22,92] - 3 frac[22,50]
   - 3 frac[22,152] - 3 frac[22,191] - 3 frac[22,253] - 3 frac[22,188]
   - 3 frac[22,65] - 3 frac[22,242] - 3 frac[22,74] + rows_tot_CPU23 = 0
 mem_tot_CPU24: - mem[23,208] - mem[23,146] - mem[23,53] - mem[23,75]
   - mem[23,129] - mem[23,193] - mem[23,229] - mem[23,28] - mem[23,226]
   - mem[23,217] - mem[23,126] - mem[23,20] + mem_tot_CPU24 = 0
 rows_tot_CPU24: - 3 frac[23,208] - 3 frac[23,146] - 3 frac[23,53]
   - 3 frac[23,75] - 3 frac[23,129] - 3 frac[23,193] - 3 frac[23,229]
   - 3 frac[23,28] - 3 frac[23,226] - 3 frac[23,217] - 3 frac[23,126]
   - 3 frac[23,20] + rows_tot_CPU24 = 0
 mem_tot_CPU25: - mem[24,113] - mem[24,148] - mem[24,65] - mem[24,123]
   - mem[24,18] - mem[24,158] - mem[24,13] - mem[24,126] + mem_tot_CPU25
   = 0
 rows_tot_CPU25: - 3 frac[24,113] - 3 frac[24,148] - 3 frac[24,65]
   - 3 frac[24,123] - 3 frac[24,18] - 3 frac[24,158] - 3 frac[24,13]
   - 3 frac[24,126] + rows_tot_CPU25 = 0
 mem_tot_CPU26: - mem[25,189] - mem[25,100] - mem[25,46] - mem[25,10]
   - mem[25,20] - mem[25,126] - mem[25,96] - mem[25,51] - mem[25,174]
   - mem[25,25] - mem[25,225] + mem_tot_CPU26 = 0
 rows_tot_CPU26: - 3 frac[25,189] - 3 frac[25,100] - 3 frac[25,46]
   - 3 frac[25,10] - 3 frac[25,20] - 3 frac[25,126] - 3 frac[25,96]
   - 3 frac[25,51] - 3 frac[25,174] - 3 frac[25,25] - 3 frac[25,225]
   + rows_tot_CPU26 = 0
 mem_tot_CPU27: - mem[26,67] - mem[26,151] - mem[26,6] - mem[26,242]
   - mem[26,187] - mem[26,130] - mem[26,127] - mem[26,43] - mem[26,183]
   - mem[26,3] + mem_tot_CPU27 = 0
 rows_tot_CPU27: - 3 frac[26,67] - 3 frac[26,151] - 3 frac[26,6]
   - 3 frac[26,242] - 3 frac[26,187] - 3 frac[26,130] - 3 frac[26,127]
   - 3 frac[26,43] - 3 frac[26,183] - 3 frac[26,3] + rows_tot_CPU27 = 0
 mem_tot_CPU28: - mem[27,169] - mem[27,14] - mem[27,161] - mem[27,6]
   - mem[27,30] - mem[27,77] - mem[27,184] - mem[27,52] - mem[27,217]
   - mem[27,82] - mem[27,129] - mem[27,78] - mem[27,92] - mem[27,85]
   - mem[27,189] - mem[27,128] - mem[27,101] + mem_tot_CPU28 = 0
 rows_tot_CPU28: - 3 frac[27,169] - 3 frac[27,14] - 3 frac[27,161]
   - 3 frac[27,6] - 3 frac[27,30] - 3 frac[27,77] - 3 frac[27,184]
   - 3 frac[27,52] - 3 frac[27,217] - 3 frac[27,82] - 3 frac[27,129]
   - 3 frac[27,78] - 3 frac[27,92] - 3 frac[27,85] - 3 frac[27,189]
   - 3 frac[27,128] - 3 frac[27,101] + rows_tot_CPU28 = 0
 mem_tot_CPU29: - mem[28,11] - mem[28,111] - mem[28,240] - mem[28,66]
   - mem[28,33] - mem[28,56] - mem[28,213] - mem[28,138] - mem[28,229]
   - mem[28,57] - mem[28,190] - mem[28,51] - mem[28,184] - mem[28,100]
   - mem[28,218] + mem_tot_CPU29 = 0
 rows_tot_CPU29: - 3 frac[28,11] - 3 frac[28,111] - 3 frac[28,240]
   - 3 frac[28,66] - 3 frac[28,33] - 3 frac[28,56] - 3 frac[28,213]
   - 3 frac[28,138] - 3 frac[28,229] - 3 frac[28,57] - 3 frac[28,190]
   - 3 frac[28,51] - 3 frac[28,184] - 3 frac[28,100] - 3 frac[28,218]
   + rows_tot_CPU29 = 0
 mem_tot_CPU30: - mem[29,241] - mem[29,19] - mem[29,150] - mem[29,70]
   - mem[29,247] - mem[29,160] - mem[29,14] - mem[29,122] - mem[29,149]
   - mem[29,5] - mem[29,154] - mem[29,181] - mem[29,225] - mem[29,208]
   - mem[29,10] - mem[29,242] + mem_tot_CPU30 = 0
 rows_tot_CPU30: - 3 frac[29,241] - 3 frac[29,19] - 3 frac[29,150]
   - 3 frac[29,70] - 3 frac[29,247] - 3 frac[29,160] - 3 frac[29,14]
   - 3 frac[29,122] - 3 frac[29,149] - 3 frac[29,5] - 3 frac[29,154]
   - 3 frac[29,181] - 3 frac[29,225] - 3 frac[29,208] - 3 frac[29,10]
   - 3 frac[29,242] + rows_tot_CPU30 = 0
 mem_tot_CPU31: - mem[30,100] - mem[30,230] - mem[30,233] - mem[30,153]
   - mem[30,85] - mem[30,158] - mem[30,97] - mem[30,83] - mem[30,161]
   + mem_tot_CPU31 = 0
 rows_tot_CPU31: - 3 frac[30,100] - 3 frac[30,230] - 3 frac[30,233]
   - 3 frac[30,153] - 3 frac[30,85] - 3 frac[30,158] - 3 frac[30,97]
   - 3 frac[30,83] - 3 frac[30,161] + rows_tot_CPU31 = 0
 mem_tot_CPU32: - mem[31,108] - mem[31,45] - mem[31,227] - mem[31,121]
   - mem[31,90] - mem[31,169] - mem[31,150] - mem[31,44] - mem[31,252]
   - mem[31,48] - mem[31,246] - mem[31,68] - mem[31,129] - mem[31,142]
   - mem[31,243] - mem[31,25] + mem_tot_CPU32 = 0
 rows_tot_CPU32: - 3 frac[31,108] - 3 frac[31,45] - 3 frac[31,227]
   - 3 frac[31,121] - 3 frac[31,90] - 3 frac[31,169] - 3 frac[31,150]
   - 3 frac[31,44] - 3 frac[31,252] - 3 frac[31,48] - 3 frac[31,246]
   - 3 frac[31,68] - 3 frac[31,129] - 3 frac[31,142] - 3 frac[31,243]
   - 3 frac[31,25] + rows_tot_CPU32 = 0
 mem_tot_CPU33: - mem[32,77] - mem[32,140] - mem[32,20] - mem[32,43]
   - mem[32,57] - mem[32,146] - mem[32,167] - mem[32,93] - mem[32,112]
   - mem[32,244] - mem[32,86] - mem[32,3] - mem[32,124] - mem[32,32]
   - mem[32,192] + mem_tot_CPU33 = 0
 rows_tot_CPU33: - 3 frac[32,77] - 3 frac[32,140] - 3 frac[32,20]
   - 3 frac[32,43] - 3 frac[32,57] - 3 frac[32,146] - 3 frac[32,167]
   - 3 frac[32,93] - 3 frac[32,112] - 3 frac[32,244] - 3 frac[32,86]
   - 3 frac[32,3] - 3 frac[32,124] - 3 frac[32,32] - 3 frac[32,192]
   + rows_tot_CPU33 = 0
 mem_tot_CPU34: - mem[33,121] - mem[33,113] - mem[33,4] - mem[33,204]
   - mem[33,225] - mem[33,89] - mem[33,108] - mem[33,133] - mem[33,78]
   - mem[33,168] - mem[33,99] - mem[33,55] - mem[33,211] - mem[33,124]
   - mem[33,149] - mem[33,116] - mem[33,31] + mem_tot_CPU34 = 0
 rows_tot_CPU34: - 3 frac[33,121] - 3 frac[33,113] - 3 frac[33,4]
   - 3 frac[33,204] - 3 frac[33,225] - 3 frac[33,89] - 3 frac[33,108]
   - 3 frac[33,133] - 3 frac[33,78] - 3 frac[33,168] - 3 frac[33,99]
   - 3 frac[33,55] - 3 frac[33,211] - 3 frac[33,124] - 3 frac[33,149]
   - 3 frac[33,116] - 3 frac[33,31] + rows_tot_CPU34 = 0
 mem_tot_CPU35: - mem[34,153] - mem[34,211] - mem[34,97] - mem[34,9]
   - mem[34,36] - mem[34,54] - mem[34,15] - mem[34,149] - mem[34,172]
   - mem[34,225] - mem[34,46] - mem[34,182] - mem[34,129] - mem[34,11]
   + mem_tot_CPU35 = 0
 rows_tot_CPU35: - 3 frac[34,153] - 3 frac[34,211] - 3 frac[34,97]
   - 3 frac[34,9] - 3 frac[34,36] - 3 frac[34,54] - 3 frac[34,15]
   - 3 frac[34,149] - 3 frac[34,172] - 3 frac[34,225] - 3 frac[34,46]
   - 3 frac[34,182] - 3 frac[34,129] - 3 frac[34,11] + rows_tot_CPU35 = 0
 mem_tot_CPU36: - mem[35,47] - mem[35,183] - mem[35,46] - mem[35,231]
   - mem[35,107] - mem[35,220] - mem[35,58] - mem[35,82] - mem[35,239]
   - mem[35,205] - mem[35,244] - mem[35,135] - mem[35,210] + mem_tot_CPU36
   = 0
 rows_tot_CPU36: - 3 frac[35,47] - 3 frac[35,183] - 3 frac[35,46]
   - 3 frac[35,231] - 3 frac[35,107] - 3 frac[35,220] - 3 frac[35,58]
   - 3 frac[35,82] - 3 frac[35,239] - 3 frac[35,205] - 3 frac[35,244]
   - 3 frac[35,135] - 3 frac[35,210] + rows_tot_CPU36 = 0
 mem_tot_CPU37: - mem[36,167] - mem[36,158] - mem[36,231] - mem[36,161]
   - mem[36,89] - mem[36,222] - mem[36,244] - mem[36,189] - mem[36,216]
   - mem[36,233] - mem[36,246] - mem[36,253] - mem[36,176] + mem_tot_CPU37
   = 0
 rows_tot_CPU37: - 3 frac[36,167] - 3 frac[36,158] - 3 frac[36,231]
   - 3 frac[36,161] - 3 frac[36,89] - 3 frac[36,222] - 3 frac[36,244]
   - 3 frac[36,189] - 3 frac[36,216] - 3 frac[36,233] - 3 frac[36,246]
   - 3 frac[36,253] - 3 frac[36,176] + rows_tot_CPU37 = 0
 mem_tot_CPU38: - mem[37,56] - mem[37,12] - mem[37,205] - mem[37,142]
   - mem[37,132] - mem[37,84] - mem[37,140] - mem[37,8] - mem[37,129]
   - mem[37,250] + mem_tot_CPU38 = 0
 rows_tot_CPU38: - 3 frac[37,56] - 3 frac[37,12] - 3 frac[37,205]
   - 3 frac[37,142] - 3 frac[37,132] - 3 frac[37,84] - 3 frac[37,140]
   - 3 frac[37,8] - 3 frac[37,129] - 3 frac[37,250] + rows_tot_CPU38 = 0
 mem_tot_CPU39: - mem[38,72] - mem[38,56] - mem[38,65] - mem[38,24]
   - mem[38,151] - mem[38,129] - mem[38,11] - mem[38,245] - mem[38,157]
   - mem[38,236] - mem[38,75] + mem_tot_CPU39 = 0
 rows_tot_CPU39: - 3 frac[38,72] - 3 frac[38,56] - 3 frac[38,65]
   - 3 frac[38,24] - 3 frac[38,151] - 3 frac[38,129] - 3 frac[38,11]
   - 3 frac[38,245] - 3 frac[38,157] - 3 frac[38,236] - 3 frac[38,75]
   + rows_tot_CPU39 = 0
 mem_tot_CPU40: - mem[39,127] - mem[39,24] - mem[39,222] - mem[39,251]
   - mem[39,141] - mem[39,177] - mem[39,82] - mem[39,130] - mem[39,187]
   - mem[39,74] - mem[39,235] - mem[39,143] - mem[39,189] - mem[39,66]
   - mem[39,181] - mem[39,86] - mem[39,254] - mem[39,117] - mem[39,91]
   + mem_tot_CPU40 = 0
 rows_tot_CPU40: - 3 frac[39,127] - 3 frac[39,24] - 3 frac[39,222]
   - 3 frac[39,251] - 3 frac[39,141] - 3 frac[39,177] - 3 frac[39,82]
   - 3 frac[39,130] - 3 frac[39,187] - 3 frac[39,74] - 3 frac[39,235]
   - 3 frac[39,143] - 3 frac[39,189] - 3 frac[39,66] - 3 frac[39,181]
   - 3 frac[39,86] - 3 frac[39,254] - 3 frac[39,117] - 3 frac[39,91]
   + rows_tot_CPU40 = 0
 mem_tot_CPU41: - mem[40,219] - mem[40,176] - mem[40,34] - mem[40,101]
   - mem[40,108] - mem[40,72] - mem[40,7] - mem[40,84] - mem[40,121]
   - mem[40,235] - mem[40,217] - mem[40,163] - mem[40,28] - mem[40,116]
   - mem[40,123] - mem[40,58] + mem_tot_CPU41 = 0
 rows_tot_CPU41: - 3 frac[40,219] - 3 frac[40,176] - 3 frac[40,34]
   - 3 frac[40,101] - 3 frac[40,108] - 3 frac[40,72] - 3 frac[40,7]
   - 3 frac[40,84] - 3 frac[40,121] - 3 frac[40,235] - 3 frac[40,217]
   - 3 frac[40,163] - 3 frac[40,28] - 3 frac[40,116] - 3 frac[40,123]
   - 3 frac[40,58] + rows_tot_CPU41 = 0
 mem_tot_CPU42: - mem[41,166] - mem[41,188] - mem[41,20] - mem[41,93]
   - mem[41,3] - mem[41,61] - mem[41,16] - mem[41,134] - mem[41,240]
   - mem[41,90] - mem[41,59] - mem[41,146] - mem[41,211] - mem[41,190]
   + mem_tot_CPU42 = 0
 rows_tot_CPU42: - 3 frac[41,166] - 3 frac[41,188] - 3 frac[41,20]
   - 3 frac[41,93] - 3 frac[41,3] - 3 frac[41,61] - 3 frac[41,16]
   - 3 frac[41,134] - 3 frac[41,240] - 3 frac[41,90] - 3 frac[41,59]
   - 3 frac[41,146] - 3 frac[41,211] - 3 frac[41,190] + rows_tot_CPU42 = 0
 mem_tot_CPU43: - mem[42,87] - mem[42,75] - mem[42,17] - mem[42,24]
   - mem[42,143] - mem[42,84] - mem[42,113] - mem[42,96] - mem[42,66]
   - mem[42,63] - mem[42,33] + mem_tot_CPU43 = 0
 rows_tot_CPU43: - 3 frac[42,87] - 3 frac[42,75] - 3 frac[42,17]
   - 3 frac[42,24] - 3 frac[42,143] - 3 frac[42,84] - 3 frac[42,113]
   - 3 frac[42,96] - 3 frac[42,66] - 3 frac[42,63] - 3 frac[42,33]
   + rows_tot_CPU43 = 0
 mem_tot_CPU44: - mem[43,3] - mem[43,51] - mem[43,38] - mem[43,2]
   - mem[43,227] - mem[43,177] - mem[43,182] - mem[43,34] - mem[43,131]
   - mem[43,7] - mem[43,86] - mem[43,161] - mem[43,159] + mem_tot_CPU44
   = 0
 rows_tot_CPU44: - 3 frac[43,3] - 3 frac[43,51] - 3 frac[43,38]
   - 3 frac[43,2] - 3 frac[43,227] - 3 frac[43,177] - 3 frac[43,182]
   - 3 frac[43,34] - 3 frac[43,131] - 3 frac[43,7] - 3 frac[43,86]
   - 3 frac[43,161] - 3 frac[43,159] + rows_tot_CPU44 = 0
 mem_tot_CPU45: - mem[44,34] - mem[44,119] - mem[44,216] - mem[44,102]
   - mem[44,93] - mem[44,65] - mem[44,142] - mem[44,68] - mem[44,42]
   - mem[44,250] - mem[44,24] - mem[44,222] - mem[44,213] + mem_tot_CPU45
   = 0
 rows_tot_CPU45: - 3 frac[44,34] - 3 frac[44,119] - 3 frac[44,216]
   - 3 frac[44,102] - 3 frac[44,93] - 3 frac[44,65] - 3 frac[44,142]
   - 3 frac[44,68] - 3 frac[44,42] - 3 frac[44,250] - 3 frac[44,24]
   - 3 frac[44,222] - 3 frac[44,213] + rows_tot_CPU45 = 0
 mem_tot_CPU46: - mem[45,12] - mem[45,143] - mem[45,91] - mem[45,242]
   - mem[45,90] - mem[45,43] - mem[45,198] - mem[45,115] - mem[45,94]
   - mem[45,239] - mem[45,180] - mem[45,14] - mem[45,27] - mem[45,45]
   - mem[45,99] + mem_tot_CPU46 = 0
 rows_tot_CPU46: - 3 frac[45,12] - 3 frac[45,143] - 3 frac[45,91]
   - 3 frac[45,242] - 3 frac[45,90] - 3 frac[45,43] - 3 frac[45,198]
   - 3 frac[45,115] - 3 frac[45,94] - 3 frac[45,239] - 3 frac[45,180]
   - 3 frac[45,14] - 3 frac[45,27] - 3 frac[45,45] - 3 frac[45,99]
   + rows_tot_CPU46 = 0
 mem_tot_CPU47: - mem[46,231] - mem[46,89] - mem[46,152] - mem[46,188]
   - mem[46,158] - mem[46,34] - mem[46,37] - mem[46,100] - mem[46,126]
   - mem[46,167] - mem[46,202] + mem_tot_CPU47 = 0
 rows_tot_CPU47: - 3 frac[46,231] - 3 frac[46,89] - 3 frac[46,152]
   - 3 frac[46,188] - 3 frac[46,158] - 3 frac[46,34] - 3 frac[46,37]
   - 3 frac[46,100] - 3 frac[46,126] - 3 frac[46,167] - 3 frac[46,202]
   + rows_tot_CPU47 = 0
 mem_tot_CPU48: - mem[47,85] - mem[47,194] - mem[47,41] - mem[47,87]
   - mem[47,244] - mem[47,153] - mem[47,213] - mem[47,152] - mem[47,51]
   - mem[47,182] + mem_tot_CPU48 = 0
 rows_tot_CPU48: - 3 frac[47,85] - 3 frac[47,194] - 3 frac[47,41]
   - 3 frac[47,87] - 3 frac[47,244] - 3 frac[47,153] - 3 frac[47,213]
   - 3 frac[47,152] - 3 frac[47,51] - 3 frac[47,182] + rows_tot_CPU48 = 0
 mem_tot_CPU49: - mem[48,227] - mem[48,202] - mem[48,80] - mem[48,78]
   - mem[48,69] - mem[48,90] - mem[48,229] - mem[48,23] - mem[48,100]
   - mem[48,166] - mem[48,112] - mem[48,187] - mem[48,56] - mem[48,95]
   + mem_tot_CPU49 = 0
 rows_tot_CPU49: - 3 frac[48,227] - 3 frac[48,202] - 3 frac[48,80]
   - 3 frac[48,78] - 3 frac[48,69] - 3 frac[48,90] - 3 frac[48,229]
   - 3 frac[48,23] - 3 frac[48,100] - 3 frac[48,166] - 3 frac[48,112]
   - 3 frac[48,187] - 3 frac[48,56] - 3 frac[48,95] + rows_tot_CPU49 = 0
 mem_tot_CPU50: - mem[49,226] - mem[49,102] - mem[49,247] - mem[49,14]
   - mem[49,42] - mem[49,158] - mem[49,63] - mem[49,48] - mem[49,251]
   - mem[49,245] - mem[49,155] - mem[49,119] - mem[49,148] - mem[49,229]
   + mem_tot_CPU50 = 0
 rows_tot_CPU50: - 3 frac[49,226] - 3 frac[49,102] - 3 frac[49,247]
   - 3 frac[49,14] - 3 frac[49,42] - 3 frac[49,158] - 3 frac[49,63]
   - 3 frac[49,48] - 3 frac[49,251] - 3 frac[49,245] - 3 frac[49,155]
   - 3 frac[49,119] - 3 frac[49,148] - 3 frac[49,229] + rows_tot_CPU50 = 0
 mem_tot_CPU51: - mem[50,156] - mem[50,254] - mem[50,148] - mem[50,16]
   - mem[50,100] - mem[50,101] - mem[50,214] - mem[50,195] - mem[50,6]
   - mem[50,42] - mem[50,43] - mem[50,198] + mem_tot_CPU51 = 0
 rows_tot_CPU51: - 3 frac[50,156] - 3 frac[50,254] - 3 frac[50,148]
   - 3 frac[50,16] - 3 frac[50,100] - 3 frac[50,101] - 3 frac[50,214]
   - 3 frac[50,195] - 3 frac[50,6] - 3 frac[50,42] - 3 frac[50,43]
   - 3 frac[50,198] + rows_tot_CPU51 = 0
 mem_tot_CPU52: - mem[51,169] - mem[51,71] - mem[51,138] - mem[51,79]
   - mem[51,30] - mem[51,246] - mem[51,41] - mem[51,12] + mem_tot_CPU52
   = 0
 rows_tot_CPU52: - 3 frac[51,169] - 3 frac[51,71] - 3 frac[51,138]
   - 3 frac[51,79] - 3 frac[51,30] - 3 frac[51,246] - 3 frac[51,41]
   - 3 frac[51,12] + rows_tot_CPU52 = 0
 mem_tot_CPU53: - mem[52,252] - mem[52,113] - mem[52,34] - mem[52,136]
   - mem[52,202] - mem[52,102] - mem[52,37] - mem[52,189] - mem[52,211]
   - mem[52,206] - mem[52,147] - mem[52,170] + mem_tot_CPU53 = 0
 rows_tot_CPU53: - 3 frac[52,252] - 3 frac[52,113] - 3 frac[52,34]
   - 3 frac[52,136] - 3 frac[52,202] - 3 frac[52,102] - 3 frac[52,37]
   - 3 frac[52,189] - 3 frac[52,211] - 3 frac[52,206] - 3 frac[52,147]
   - 3 frac[52,170] + rows_tot_CPU53 = 0
 mem_tot_CPU54: - mem[53,65] - mem[53,196] - mem[53,43] - mem[53,92]
   - mem[53,6] - mem[53,184] - mem[53,138] - mem[53,238] - mem[53,72]
   - mem[53,59] - mem[53,146] - mem[53,124] - mem[53,248] - mem[53,134]
   - mem[53,254] - mem[53,192] - mem[53,57] - mem[53,133] - mem[53,152]
   + mem_tot_CPU54 = 0
 rows_tot_CPU54: - 3 frac[53,65] - 3 frac[53,196] - 3 frac[53,43]
   - 3 frac[53,92] - 3 frac[53,6] - 3 frac[53,184] - 3 frac[53,138]
   - 3 frac[53,238] - 3 frac[53,72] - 3 frac[53,59] - 3 frac[53,146]
   - 3 frac[53,124] - 3 frac[53,248] - 3 frac[53,134] - 3 frac[53,254]
   - 3 frac[53,192] - 3 frac[53,57] - 3 frac[53,133] - 3 frac[53,152]
   + rows_tot_CPU54 = 0
 mem_tot_CPU55: - mem[54,4] - mem[54,153] - mem[54,39] - mem[54,230]
   - mem[54,65] - mem[54,104] - mem[54,18] - mem[54,37] - mem[54,167]
   - mem[54,131] + mem_tot_CPU55 = 0
 rows_tot_CPU55: - 3 frac[54,4] - 3 frac[54,153] - 3 frac[54,39]
   - 3 frac[54,230] - 3 frac[54,65] - 3 frac[54,104] - 3 frac[54,18]
   - 3 frac[54,37] - 3 frac[54,167] - 3 frac[54,131] + rows_tot_CPU55 = 0
 mem_tot_CPU56: - mem[55,247] - mem[55,194] - mem[55,220] - mem[55,119]
   - mem[55,107] - mem[55,169] - mem[55,37] - mem[55,59] - mem[55,6]
   - mem[55,20] - mem[55,133] - mem[55,216] - mem[55,143] + mem_tot_CPU56
   = 0
 rows_tot_CPU56: - 3 frac[55,247] - 3 frac[55,194] - 3 frac[55,220]
   - 3 frac[55,119] - 3 frac[55,107] - 3 frac[55,169] - 3 frac[55,37]
   - 3 frac[55,59] - 3 frac[55,6] - 3 frac[55,20] - 3 frac[55,133]
   - 3 frac[55,216] - 3 frac[55,143] + rows_tot_CPU56 = 0
 mem_tot_CPU57: - mem[56,36] - mem[56,233] - mem[56,116] - mem[56,16]
   - mem[56,227] - mem[56,59] - mem[56,235] - mem[56,18] - mem[56,70]
   - mem[56,91] - mem[56,202] - mem[56,40] - mem[56,99] - mem[56,20]
   - mem[56,93] - mem[56,180] - mem[56,211] + mem_tot_CPU57 = 0
 rows_tot_CPU57: - 3 frac[56,36] - 3 frac[56,233] - 3 frac[56,116]
   - 3 frac[56,16] - 3 frac[56,227] - 3 frac[56,59] - 3 frac[56,235]
   - 3 frac[56,18] - 3 frac[56,70] - 3 frac[56,91] - 3 frac[56,202]
   - 3 frac[56,40] - 3 frac[56,99] - 3 frac[56,20] - 3 frac[56,93]
   - 3 frac[56,180] - 3 frac[56,211] + rows_tot_CPU57 = 0
 mem_tot_CPU58: - mem[57,144] - mem[57,95] - mem[57,114] - mem[57,208]
   - mem[57,37] - mem[57,56] - mem[57,74] - mem[57,216] - mem[57,36]
   - mem[57,73] - mem[57,175] - mem[57,24] - mem[57,240] - mem[57,145]
   + mem_tot_CPU58 = 0
 rows_tot_CPU58: - 3 frac[57,144] - 3 frac[57,95] - 3 frac[57,114]
   - 3 frac[57,208] - 3 frac[57,37] - 3 frac[57,56] - 3 frac[57,74]
   - 3 frac[57,216] - 3 frac[57,36] - 3 frac[57,73] - 3 frac[57,175]
   - 3 frac[57,24] - 3 frac[57,240] - 3 frac[57,145] + rows_tot_CPU58 = 0
 mem_tot_CPU59: - mem[58,26] - mem[58,55] - mem[58,89] - mem[58,187]
   - mem[58,164] - mem[58,184] - mem[58,23] - mem[58,67] - mem[58,123]
   - mem[58,33] - mem[58,155] - mem[58,58] - mem[58,93] - mem[58,10]
   - mem[58,28] - mem[58,39] - mem[58,213] + mem_tot_CPU59 = 0
 rows_tot_CPU59: - 3 frac[58,26] - 3 frac[58,55] - 3 frac[58,89]
   - 3 frac[58,187] - 3 frac[58,164] - 3 frac[58,184] - 3 frac[58,23]
   - 3 frac[58,67] - 3 frac[58,123] - 3 frac[58,33] - 3 frac[58,155]
   - 3 frac[58,58] - 3 frac[58,93] - 3 frac[58,10] - 3 frac[58,28]
   - 3 frac[58,39] - 3 frac[58,213] + rows_tot_CPU59 = 0
 mem_tot_CPU60: - mem[59,131] - mem[59,203] - mem[59,93] - mem[59,126]
   - mem[59,167] - mem[59,214] - mem[59,11] - mem[59,247] - mem[59,91]
   - mem[59,4] - mem[59,69] + mem_tot_CPU60 = 0
 rows_tot_CPU60: - 3 frac[59,131] - 3 frac[59,203] - 3 frac[59,93]
   - 3 frac[59,126] - 3 frac[59,167] - 3 frac[59,214] - 3 frac[59,11]
   - 3 frac[59,247] - 3 frac[59,91] - 3 frac[59,4] - 3 frac[59,69]
   + rows_tot_CPU60 = 0
 mem_tot_CPU61: - mem[60,148] - mem[60,161] - mem[60,192] - mem[60,241]
   - mem[60,172] - mem[60,162] - mem[60,17] - mem[60,216] - mem[60,70]
   - mem[60,238] - mem[60,217] - mem[60,165] - mem[60,8] - mem[60,206]
   - mem[60,220] - mem[60,74] - mem[60,110] - mem[60,246] - mem[60,198]
   - mem[60,36] - mem[60,90] + mem_tot_CPU61 = 0
 rows_tot_CPU61: - 3 frac[60,148] - 3 frac[60,161] - 3 frac[60,192]
   - 3 frac[60,241] - 3 frac[60,172] - 3 frac[60,162] - 3 frac[60,17]
   - 3 frac[60,216] - 3 frac[60,70] - 3 frac[60,238] - 3 frac[60,217]
   - 3 frac[60,165] - 3 frac[60,8] - 3 frac[60,206] - 3 frac[60,220]
   - 3 frac[60,74] - 3 frac[60,110] - 3 frac[60,246] - 3 frac[60,198]
   - 3 frac[60,36] - 3 frac[60,90] + rows_tot_CPU61 = 0
 mem_tot_CPU62: - mem[61,223] - mem[61,175] - mem[61,73] - mem[61,167]
   - mem[61,160] - mem[61,24] - mem[61,184] - mem[61,100] - mem[61,51]
   - mem[61,87] - mem[61,159] - mem[61,145] - mem[61,217] - mem[61,158]
   + mem_tot_CPU62 = 0
 rows_tot_CPU62: - 3 frac[61,223] - 3 frac[61,175] - 3 frac[61,73]
   - 3 frac[61,167] - 3 frac[61,160] - 3 frac[61,24] - 3 frac[61,184]
   - 3 frac[61,100] - 3 frac[61,51] - 3 frac[61,87] - 3 frac[61,159]
   - 3 frac[61,145] - 3 frac[61,217] - 3 frac[61,158] + rows_tot_CPU62 = 0
 mem_tot_CPU63: - mem[62,145] - mem[62,222] - mem[62,139] - mem[62,226]
   - mem[62,43] - mem[62,146] - mem[62,135] - mem[62,94] - mem[62,232]
   - mem[62,78] - mem[62,113] + mem_tot_CPU63 = 0
 rows_tot_CPU63: - 3 frac[62,145] - 3 frac[62,222] - 3 frac[62,139]
   - 3 frac[62,226] - 3 frac[62,43] - 3 frac[62,146] - 3 frac[62,135]
   - 3 frac[62,94] - 3 frac[62,232] - 3 frac[62,78] - 3 frac[62,113]
   + rows_tot_CPU63 = 0
 mem_tot_CPU64: - mem[63,218] - mem[63,0] - mem[63,215] - mem[63,29]
   - mem[63,85] - mem[63,73] - mem[63,143] - mem[63,189] - mem[63,245]
   - mem[63,90] - mem[63,165] - mem[63,111] - mem[63,12] - mem[63,164]
   - mem[63,229] + mem_tot_CPU64 = 0
 rows_tot_CPU64: - 3 frac[63,218] - 3 frac[63,0] - 3 frac[63,215]
   - 3 frac[63,29] - 3 frac[63,85] - 3 frac[63,73] - 3 frac[63,143]
   - 3 frac[63,189] - 3 frac[63,245] - 3 frac[63,90] - 3 frac[63,165]
   - 3 frac[63,111] - 3 frac[63,12] - 3 frac[63,164] - 3 frac[63,229]
   + rows_tot_CPU64 = 0
 mem_tot_CPU65: - mem[64,107] - mem[64,169] - mem[64,167] - mem[64,72]
   - mem[64,184] - mem[64,51] - mem[64,136] - mem[64,111] - mem[64,164]
   - mem[64,228] - mem[64,175] + mem_tot_CPU65 = 0
 rows_tot_CPU65: - 3 frac[64,107] - 3 frac[64,169] - 3 frac[64,167]
   - 3 frac[64,72] - 3 frac[64,184] - 3 frac[64,51] - 3 frac[64,136]
   - 3 frac[64,111] - 3 frac[64,164] - 3 frac[64,228] - 3 frac[64,175]
   + rows_tot_CPU65 = 0
 mem_tot_CPU66: - mem[65,11] - mem[65,226] - mem[65,148] - mem[65,143]
   - mem[65,107] - mem[65,129] - mem[65,83] - mem[65,2] - mem[65,249]
   - mem[65,79] - mem[65,144] - mem[65,221] - mem[65,214] - mem[65,167]
   - mem[65,58] - mem[65,93] - mem[65,47] + mem_tot_CPU66 = 0
 rows_tot_CPU66: - 3 frac[65,11] - 3 frac[65,226] - 3 frac[65,148]
   - 3 frac[65,143] - 3 frac[65,107] - 3 frac[65,129] - 3 frac[65,83]
   - 3 frac[65,2] - 3 frac[65,249] - 3 frac[65,79] - 3 frac[65,144]
   - 3 frac[65,221] - 3 frac[65,214] - 3 frac[65,167] - 3 frac[65,58]
   - 3 frac[65,93] - 3 frac[65,47] + rows_tot_CPU66 = 0
 mem_tot_CPU67: - mem[66,179] - mem[66,91] - mem[66,202] - mem[66,16]
   - mem[66,41] - mem[66,117] - mem[66,94] - mem[66,213] - mem[66,5]
   - mem[66,143] - mem[66,129] - mem[66,48] - mem[66,244] + mem_tot_CPU67
   = 0
 rows_tot_CPU67: - 3 frac[66,179] - 3 frac[66,91] - 3 frac[66,202]
   - 3 frac[66,16] - 3 frac[66,41] - 3 frac[66,117] - 3 frac[66,94]
   - 3 frac[66,213] - 3 frac[66,5] - 3 frac[66,143] - 3 frac[66,129]
   - 3 frac[66,48] - 3 frac[66,244] + rows_tot_CPU67 = 0
 mem_tot_CPU68: - mem[67,185] - mem[67,43] - mem[67,166] - mem[67,227]
   - mem[67,91] - mem[67,151] - mem[67,123] - mem[67,241] - mem[67,135]
   - mem[67,87] - mem[67,38] - mem[67,134] - mem[67,231] - mem[67,107]
   + mem_tot_CPU68 = 0
 rows_tot_CPU68: - 3 frac[67,185] - 3 frac[67,43] - 3 frac[67,166]
   - 3 frac[67,227] - 3 frac[67,91] - 3 frac[67,151] - 3 frac[67,123]
   - 3 frac[67,241] - 3 frac[67,135] - 3 frac[67,87] - 3 frac[67,38]
   - 3 frac[67,134] - 3 frac[67,231] - 3 frac[67,107] + rows_tot_CPU68 = 0
 mem_tot_CPU69: - mem[68,194] - mem[68,168] - mem[68,89] - mem[68,56]
   - mem[68,46] - mem[68,84] - mem[68,90] - mem[68,171] - mem[68,74]
   - mem[68,147] - mem[68,6] - mem[68,70] + mem_tot_CPU69 = 0
 rows_tot_CPU69: - 3 frac[68,194] - 3 frac[68,168] - 3 frac[68,89]
   - 3 frac[68,56] - 3 frac[68,46] - 3 frac[68,84] - 3 frac[68,90]
   - 3 frac[68,171] - 3 frac[68,74] - 3 frac[68,147] - 3 frac[68,6]
   - 3 frac[68,70] + rows_tot_CPU69 = 0
 mem_tot_CPU70: - mem[69,253] - mem[69,229] - mem[69,97] - mem[69,145]
   - mem[69,242] - mem[69,103] - mem[69,104] - mem[69,39] + mem_tot_CPU70
   = 0
 rows_tot_CPU70: - 3 frac[69,253] - 3 frac[69,229] - 3 frac[69,97]
   - 3 frac[69,145] - 3 frac[69,242] - 3 frac[69,103] - 3 frac[69,104]
   - 3 frac[69,39] + rows_tot_CPU70 = 0
 mem_tot_CPU71: - mem[70,219] - mem[70,175] - mem[70,130] - mem[70,27]
   - mem[70,55] - mem[70,209] - mem[70,142] - mem[70,11] - mem[70,119]
   - mem[70,97] + mem_tot_CPU71 = 0
 rows_tot_CPU71: - 3 frac[70,219] - 3 frac[70,175] - 3 frac[70,130]
   - 3 frac[70,27] - 3 frac[70,55] - 3 frac[70,209] - 3 frac[70,142]
   - 3 frac[70,11] - 3 frac[70,119] - 3 frac[70,97] + rows_tot_CPU71 = 0
 mem_tot_CPU72: - mem[71,90] - mem[71,6] - mem[71,206] - mem[71,150]
   - mem[71,229] - mem[71,122] - mem[71,59] - mem[71,41] - mem[71,68]
   - mem[71,136] - mem[71,88] - mem[71,71] - mem[71,212] - mem[71,108]
   - mem[71,98] - mem[71,24] + mem_tot_CPU72 = 0
 rows_tot_CPU72: - 3 frac[71,90] - 3 frac[71,6] - 3 frac[71,206]
   - 3 frac[71,150] - 3 frac[71,229] - 3 frac[71,122] - 3 frac[71,59]
   - 3 frac[71,41] - 3 frac[71,68] - 3 frac[71,136] - 3 frac[71,88]
   - 3 frac[71,71] - 3 frac[71,212] - 3 frac[71,108] - 3 frac[71,98]
   - 3 frac[71,24] + rows_tot_CPU72 = 0
 mem_tot_CPU73: - mem[72,148] - mem[72,31] - mem[72,190] - mem[72,52]
   - mem[72,253] - mem[72,138] - mem[72,201] - mem[72,117] - mem[72,2]
   - mem[72,159] - mem[72,38] - mem[72,45] + mem_tot_CPU73 = 0
 rows_tot_CPU73: - 3 frac[72,148] - 3 frac[72,31] - 3 frac[72,190]
   - 3 frac[72,52] - 3 frac[72,253] - 3 frac[72,138] - 3 frac[72,201]
   - 3 frac[72,117] - 3 frac[72,2] - 3 frac[72,159] - 3 frac[72,38]
   - 3 frac[72,45] + rows_tot_CPU73 = 0
 mem_tot_CPU74: - mem[73,91] - mem[73,6] - mem[73,90] - mem[73,89]
   - mem[73,250] - mem[73,188] - mem[73,174] - mem[73,177] - mem[73,163]
   - mem[73,206] - mem[73,185] - mem[73,99] - mem[73,22] - mem[73,107]
   + mem_tot_CPU74 = 0
 rows_tot_CPU74: - 3 frac[73,91] - 3 frac[73,6] - 3 frac[73,90]
   - 3 frac[73,89] - 3 frac[73,250] - 3 frac[73,188] - 3 frac[73,174]
   - 3 frac[73,177] - 3 frac[73,163] - 3 frac[73,206] - 3 frac[73,185]
   - 3 frac[73,99] - 3 frac[73,22] - 3 frac[73,107] + rows_tot_CPU74 = 0
 mem_tot_CPU75: - mem[74,253] - mem[74,142] - mem[74,90] - mem[74,53]
   - mem[74,41] - mem[74,184] - mem[74,240] - mem[74,61] - mem[74,8]
   - mem[74,111] - mem[74,84] - mem[74,169] - mem[74,122] - mem[74,205]
   - mem[74,225] - mem[74,202] - mem[74,113] + mem_tot_CPU75 = 0
 rows_tot_CPU75: - 3 frac[74,253] - 3 frac[74,142] - 3 frac[74,90]
   - 3 frac[74,53] - 3 frac[74,41] - 3 frac[74,184] - 3 frac[74,240]
   - 3 frac[74,61] - 3 frac[74,8] - 3 frac[74,111] - 3 frac[74,84]
   - 3 frac[74,169] - 3 frac[74,122] - 3 frac[74,205] - 3 frac[74,225]
   - 3 frac[74,202] - 3 frac[74,113] + rows_tot_CPU75 = 0
 mem_tot_CPU76: - mem[75,154] - mem[75,86] - mem[75,183] - mem[75,223]
   - mem[75,214] - mem[75,119] - mem[75,108] - mem[75,52] - mem[75,179]
   - mem[75,11] - mem[75,145] - mem[75,142] - mem[75,87] + mem_tot_CPU76
   = 0
 rows_tot_CPU76: - 3 frac[75,154] - 3 frac[75,86] - 3 frac[75,183]
   - 3 frac[75,223] - 3 frac[75,214] - 3 frac[75,119] - 3 frac[75,108]
   - 3 frac[75,52] - 3 frac[75,179] - 3 frac[75,11] - 3 frac[75,145]
   - 3 frac[75,142] - 3 frac[75,87] + rows_tot_CPU76 = 0
 mem_tot_CPU77: - mem[76,78] - mem[76,123] - mem[76,12] - mem[76,252]
   - mem[76,41] - mem[76,184] + mem_tot_CPU77 = 0
 rows_tot_CPU77: - 3 frac[76,78] - 3 frac[76,123] - 3 frac[76,12]
   - 3 frac[76,252] - 3 frac[76,41] - 3 frac[76,184] + rows_tot_CPU77 = 0
 mem_tot_CPU78: - mem[77,18] - mem[77,142] - mem[77,197] - mem[77,240]
   - mem[77,65] - mem[77,233] - mem[77,108] - mem[77,23] - mem[77,16]
   - mem[77,9] - mem[77,202] - mem[77,230] - mem[77,56] - mem[77,89]
   - mem[77,26] + mem_tot_CPU78 = 0
 rows_tot_CPU78: - 3 frac[77,18] - 3 frac[77,142] - 3 frac[77,197]
   - 3 frac[77,240] - 3 frac[77,65] - 3 frac[77,233] - 3 frac[77,108]
   - 3 frac[77,23] - 3 frac[77,16] - 3 frac[77,9] - 3 frac[77,202]
   - 3 frac[77,230] - 3 frac[77,56] - 3 frac[77,89] - 3 frac[77,26]
   + rows_tot_CPU78 = 0
 mem_tot_CPU79: - mem[78,55] - mem[78,130] - mem[78,100] - mem[78,64]
   - mem[78,174] - mem[78,27] - mem[78,101] - mem[78,222] - mem[78,70]
   - mem[78,58] - mem[78,48] - mem[78,30] - mem[78,226] + mem_tot_CPU79
   = 0
 rows_tot_CPU79: - 3 frac[78,55] - 3 frac[78,130] - 3 frac[78,100]
   - 3 frac[78,64] - 3 frac[78,174] - 3 frac[78,27] - 3 frac[78,101]
   - 3 frac[78,222] - 3 frac[78,70] - 3 frac[78,58] - 3 frac[78,48]
   - 3 frac[78,30] - 3 frac[78,226] + rows_tot_CPU79 = 0
 mem_tot_CPU80: - mem[79,91] - mem[79,101] - mem[79,128] - mem[79,137]
   - mem[79,100] - mem[79,186] - mem[79,18] - mem[79,106] - mem[79,34]
   - mem[79,6] - mem[79,227] - mem[79,44] - mem[79,69] - mem[79,75]
   - mem[79,206] - mem[79,142] + mem_tot_CPU80 = 0
 rows_tot_CPU80: - 3 frac[79,91] - 3 frac[79,101] - 3 frac[79,128]
   - 3 frac[79,137] - 3 frac[79,100] - 3 frac[79,186] - 3 frac[79,18]
   - 3 frac[79,106] - 3 frac[79,34] - 3 frac[79,6] - 3 frac[79,227]
   - 3 frac[79,44] - 3 frac[79,69] - 3 frac[79,75] - 3 frac[79,206]
   - 3 frac[79,142] + rows_tot_CPU80 = 0
 mem_tot_CPU81: - mem[80,232] - mem[80,23] - mem[80,63] - mem[80,96]
   - mem[80,219] - mem[80,190] - mem[80,38] - mem[80,240] - mem[80,164]
   + mem_tot_CPU81 = 0
 rows_tot_CPU81: - 3 frac[80,232] - 3 frac[80,23] - 3 frac[80,63]
   - 3 frac[80,96] - 3 frac[80,219] - 3 frac[80,190] - 3 frac[80,38]
   - 3 frac[80,240] - 3 frac[80,164] + rows_tot_CPU81 = 0
 mem_tot_CPU82: - mem[81,254] - mem[81,121] - mem[81,107] - mem[81,92]
   - mem[81,165] - mem[81,169] - mem[81,16] - mem[81,227] - mem[81,214]
   - mem[81,190] - mem[81,148] - mem[81,153] + mem_tot_CPU82 = 0
 rows_tot_CPU82: - 3 frac[81,254] - 3 frac[81,121] - 3 frac[81,107]
   - 3 frac[81,92] - 3 frac[81,165] - 3 frac[81,169] - 3 frac[81,16]
   - 3 frac[81,227] - 3 frac[81,214] - 3 frac[81,190] - 3 frac[81,148]
   - 3 frac[81,153] + rows_tot_CPU82 = 0
 mem_tot_CPU83: - mem[82,53] - mem[82,59] - mem[82,244] - mem[82,216]
   - mem[82,107] - mem[82,28] - mem[82,217] - mem[82,119] - mem[82,110]
   - mem[82,186] - mem[82,54] - mem[82,153] - mem[82,130] - mem[82,215]
   - mem[82,69] + mem_tot_CPU83 = 0
 rows_tot_CPU83: - 3 frac[82,53] - 3 frac[82,59] - 3 frac[82,244]
   - 3 frac[82,216] - 3 frac[82,107] - 3 frac[82,28] - 3 frac[82,217]
   - 3 frac[82,119] - 3 frac[82,110] - 3 frac[82,186] - 3 frac[82,54]
   - 3 frac[82,153] - 3 frac[82,130] - 3 frac[82,215] - 3 frac[82,69]
   + rows_tot_CPU83 = 0
 mem_tot_CPU84: - mem[83,94] - mem[83,122] - mem[83,249] - mem[83,190]
   - mem[83,165] - mem[83,127] - mem[83,131] - mem[83,173] - mem[83,7]
   - mem[83,153] - mem[83,213] + mem_tot_CPU84 = 0
 rows_tot_CPU84: - 3 frac[83,94] - 3 frac[83,122] - 3 frac[83,249]
   - 3 frac[83,190] - 3 frac[83,165] - 3 frac[83,127] - 3 frac[83,131]
   - 3 frac[83,173] - 3 frac[83,7] - 3 frac[83,153] - 3 frac[83,213]
   + rows_tot_CPU84 = 0
 mem_tot_CPU85: - mem[84,247] - mem[84,5] - mem[84,47] - mem[84,81]
   - mem[84,22] - mem[84,91] - mem[84,243] - mem[84,202] - mem[84,217]
   - mem[84,87] + mem_tot_CPU85 = 0
 rows_tot_CPU85: - 3 frac[84,247] - 3 frac[84,5] - 3 frac[84,47]
   - 3 frac[84,81] - 3 frac[84,22] - 3 frac[84,91] - 3 frac[84,243]
   - 3 frac[84,202] - 3 frac[84,217] - 3 frac[84,87] + rows_tot_CPU85 = 0
 mem_tot_CPU86: - mem[85,98] - mem[85,168] - mem[85,248] - mem[85,156]
   - mem[85,60] - mem[85,216] - mem[85,171] - mem[85,195] - mem[85,117]
   + mem_tot_CPU86 = 0
 rows_tot_CPU86: - 3 frac[85,98] - 3 frac[85,168] - 3 frac[85,248]
   - 3 frac[85,156] - 3 frac[85,60] - 3 frac[85,216] - 3 frac[85,171]
   - 3 frac[85,195] - 3 frac[85,117] + rows_tot_CPU86 = 0
 mem_tot_CPU87: - mem[86,9] - mem[86,138] - mem[86,169] - mem[86,151]
   - mem[86,6] - mem[86,242] - mem[86,148] - mem[86,31] - mem[86,142]
   + mem_tot_CPU87 = 0
 rows_tot_CPU87: - 3 frac[86,9] - 3 frac[86,138] - 3 frac[86,169]
   - 3 frac[86,151] - 3 frac[86,6] - 3 frac[86,242] - 3 frac[86,148]
   - 3 frac[86,31] - 3 frac[86,142] + rows_tot_CPU87 = 0
 mem_tot_CPU88: - mem[87,203] - mem[87,239] - mem[87,38] - mem[87,87]
   - mem[87,247] - mem[87,75] - mem[87,183] - mem[87,241] - mem[87,156]
   + mem_tot_CPU88 = 0
 rows_tot_CPU88: - 3 frac[87,203] - 3 frac[87,239] - 3 frac[87,38]
   - 3 frac[87,87] - 3 frac[87,247] - 3 frac[87,75] - 3 frac[87,183]
   - 3 frac[87,241] - 3 frac[87,156] + rows_tot_CPU88 = 0
 mem_tot_CPU89: - mem[88,147] - mem[88,16] - mem[88,111] - mem[88,89]
   - mem[88,17] - mem[88,128] - mem[88,119] - mem[88,120] - mem[88,195]
   - mem[88,209] - mem[88,19] - mem[88,232] - mem[88,239] - mem[88,0]
   - mem[88,7] - mem[88,205] + mem_tot_CPU89 = 0
 rows_tot_CPU89: - 3 frac[88,147] - 3 frac[88,16] - 3 frac[88,111]
   - 3 frac[88,89] - 3 frac[88,17] - 3 frac[88,128] - 3 frac[88,119]
   - 3 frac[88,120] - 3 frac[88,195] - 3 frac[88,209] - 3 frac[88,19]
   - 3 frac[88,232] - 3 frac[88,239] - 3 frac[88,0] - 3 frac[88,7]
   - 3 frac[88,205] + rows_tot_CPU89 = 0
 mem_tot_CPU90: - mem[89,227] - mem[89,219] - mem[89,209] - mem[89,36]
   - mem[89,142] - mem[89,0] - mem[89,243] - mem[89,107] - mem[89,50]
   - mem[89,204] - mem[89,39] - mem[89,248] - mem[89,202] + mem_tot_CPU90
   = 0
 rows_tot_CPU90: - 3 frac[89,227] - 3 frac[89,219] - 3 frac[89,209]
   - 3 frac[89,36] - 3 frac[89,142] - 3 frac[89,0] - 3 frac[89,243]
   - 3 frac[89,107] - 3 frac[89,50] - 3 frac[89,204] - 3 frac[89,39]
   - 3 frac[89,248] - 3 frac[89,202] + rows_tot_CPU90 = 0
 mem_tot_CPU91: - mem[90,254] - mem[90,66] - mem[90,84] - mem[90,183]
   - mem[90,245] - mem[90,100] - mem[90,30] - mem[90,162] - mem[90,46]
   - mem[90,82] - mem[90,72] - mem[90,189] - mem[90,132] - mem[90,55]
   - mem[90,117] - mem[90,140] - mem[90,8] + mem_tot_CPU91 = 0
 rows_tot_CPU91: - 3 frac[90,254] - 3 frac[90,66] - 3 frac[90,84]
   - 3 frac[90,183] - 3 frac[90,245] - 3 frac[90,100] - 3 frac[90,30]
   - 3 frac[90,162] - 3 frac[90,46] - 3 frac[90,82] - 3 frac[90,72]
   - 3 frac[90,189] - 3 frac[90,132] - 3 frac[90,55] - 3 frac[90,117]
   - 3 frac[90,140] - 3 frac[90,8] + rows_tot_CPU91 = 0
 mem_tot_CPU92: - mem[91,73] - mem[91,78] - mem[91,194] - mem[91,184]
   - mem[91,254] - mem[91,196] - mem[91,15] + mem_tot_CPU92 = 0
 rows_tot_CPU92: - 3 frac[91,73] - 3 frac[91,78] - 3 frac[91,194]
   - 3 frac[91,184] - 3 frac[91,254] - 3 frac[91,196] - 3 frac[91,15]
   + rows_tot_CPU92 = 0
 mem_tot_CPU93: - mem[92,202] - mem[92,188] - mem[92,208] - mem[92,166]
   - mem[92,113] - mem[92,101] - mem[92,153] - mem[92,144] - mem[92,95]
   - mem[92,8] - mem[92,182] - mem[92,116] - mem[92,211] - mem[92,89]
   - mem[92,96] - mem[92,23] + mem_tot_CPU93 = 0
 rows_tot_CPU93: - 3 frac[92,202] - 3 frac[92,188] - 3 frac[92,208]
   - 3 frac[92,166] - 3 frac[92,113] - 3 frac[92,101] - 3 frac[92,153]
   - 3 frac[92,144] - 3 frac[92,95] - 3 frac[92,8] - 3 frac[92,182]
   - 3 frac[92,116] - 3 frac[92,211] - 3 frac[92,89] - 3 frac[92,96]
   - 3 frac[92,23] + rows_tot_CPU93 = 0
 mem_tot_CPU94: - mem[93,42] - mem[93,91] - mem[93,35] - mem[93,0]
   - mem[93,208] - mem[93,117] - mem[93,214] - mem[93,64] - mem[93,148]
   - mem[93,255] - mem[93,143] - mem[93,154] - mem[93,27] - mem[93,168]
   - mem[93,179] - mem[93,133] + mem_tot_CPU94 = 0
 rows_tot_CPU94: - 3 frac[93,42] - 3 frac[93,91] - 3 frac[93,35]
   - 3 frac[93,0] - 3 frac[93,208] - 3 frac[93,117] - 3 frac[93,214]
   - 3 frac[93,64] - 3 frac[93,148] - 3 frac[93,255] - 3 frac[93,143]
   - 3 frac[93,154] - 3 frac[93,27] - 3 frac[93,168] - 3 frac[93,179]
   - 3 frac[93,133] + rows_tot_CPU94 = 0
 mem_tot_CPU95: - mem[94,91] - mem[94,66] - mem[94,227] - mem[94,41]
   - mem[94,16] - mem[94,39] - mem[94,254] - mem[94,65] - mem[94,190]
   - mem[94,180] - mem[94,243] + mem_tot_CPU95 = 0
 rows_tot_CPU95: - 3 frac[94,91] - 3 frac[94,66] - 3 frac[94,227]
   - 3 frac[94,41] - 3 frac[94,16] - 3 frac[94,39] - 3 frac[94,254]
   - 3 frac[94,65] - 3 frac[94,190] - 3 frac[94,180] - 3 frac[94,243]
   + rows_tot_CPU95 = 0
 mem_tot_CPU96: - mem[95,135] - mem[95,242] - mem[95,83] - mem[95,233]
   - mem[95,153] - mem[95,182] - mem[95,199] - mem[95,11] - mem[95,152]
   - mem[95,235] - mem[95,92] - mem[95,52] - mem[95,114] - mem[95,159]
   + mem_tot_CPU96 = 0
 rows_tot_CPU96: - 3 frac[95,135] - 3 frac[95,242] - 3 frac[95,83]
   - 3 frac[95,233] - 3 frac[95,153] - 3 frac[95,182] - 3 frac[95,199]
   - 3 frac[95,11] - 3 frac[95,152] - 3 frac[95,235] - 3 frac[95,92]
   - 3 frac[95,52] - 3 frac[95,114] - 3 frac[95,159] + rows_tot_CPU96 = 0
 mem_tot_CPU97: - mem[96,225] - mem[96,207] - mem[96,87] - mem[96,45]
   - mem[96,67] - mem[96,201] - mem[96,81] - mem[96,191] - mem[96,12]
   - mem[96,135] - mem[96,78] - mem[96,129] - mem[96,72] - mem[96,164]
   - mem[96,239] - mem[96,119] - mem[96,230] - mem[96,5] - mem[96,224]
   - mem[96,34] - mem[96,166] + mem_tot_CPU97 = 0
 rows_tot_CPU97: - 3 frac[96,225] - 3 frac[96,207] - 3 frac[96,87]
   - 3 frac[96,45] - 3 frac[96,67] - 3 frac[96,201] - 3 frac[96,81]
   - 3 frac[96,191] - 3 frac[96,12] - 3 frac[96,135] - 3 frac[96,78]
   - 3 frac[96,129] - 3 frac[96,72] - 3 frac[96,164] - 3 frac[96,239]
   - 3 frac[96,119] - 3 frac[96,230] - 3 frac[96,5] - 3 frac[96,224]
   - 3 frac[96,34] - 3 frac[96,166] + rows_tot_CPU97 = 0
 mem_tot_CPU98: - mem[97,22] - mem[97,190] - mem[97,183] - mem[97,57]
   - mem[97,145] - mem[97,188] - mem[97,227] - mem[97,85] - mem[97,17]
   - mem[97,222] - mem[97,79] - mem[97,177] - mem[97,249] - mem[97,148]
   + mem_tot_CPU98 = 0
 rows_tot_CPU98: - 3 frac[97,22] - 3 frac[97,190] - 3 frac[97,183]
   - 3 frac[97,57] - 3 frac[97,145] - 3 frac[97,188] - 3 frac[97,227]
   - 3 frac[97,85] - 3 frac[97,17] - 3 frac[97,222] - 3 frac[97,79]
   - 3 frac[97,177] - 3 frac[97,249] - 3 frac[97,148] + rows_tot_CPU98 = 0
 mem_tot_CPU99: - mem[98,45] - mem[98,55] - mem[98,229] - mem[98,10]
   - mem[98,226] - mem[98,205] - mem[98,91] - mem[98,153] - mem[98,202]
   - mem[98,189] - mem[98,44] - mem[98,168] + mem_tot_CPU99 = 0
 rows_tot_CPU99: - 3 frac[98,45] - 3 frac[98,55] - 3 frac[98,229]
   - 3 frac[98,10] - 3 frac[98,226] - 3 frac[98,205] - 3 frac[98,91]
   - 3 frac[98,153] - 3 frac[98,202] - 3 frac[98,189] - 3 frac[98,44]
   - 3 frac[98,168] + rows_tot_CPU99 = 0
 mem_tot_CPU100: - mem[99,244] - mem[99,182] - mem[99,46] - mem[99,120]
   - mem[99,30] - mem[99,172] - mem[99,237] - mem[99,89] - mem[99,166]
   - mem[99,175] - mem[99,8] - mem[99,90] - mem[99,128] - mem[99,33]
   - mem[99,165] + mem_tot_CPU100 = 0
 rows_tot_CPU100: - 3 frac[99,244] - 3 frac[99,182] - 3 frac[99,46]
   - 3 frac[99,120] - 3 frac[99,30] - 3 frac[99,172] - 3 frac[99,237]
   - 3 frac[99,89] - 3 frac[99,166] - 3 frac[99,175] - 3 frac[99,8]
   - 3 frac[99,90] - 3 frac[99,128] - 3 frac[99,33] - 3 frac[99,165]
   + rows_tot_CPU100 = 0
 mem_tot_CPU101: - mem[100,199] - mem[100,8] - mem[100,62] - mem[100,228]
   - mem[100,22] - mem[100,70] - mem[100,165] - mem[100,123] - mem[100,105]
   - mem[100,231] - mem[100,31] - mem[100,24] - mem[100,41] - mem[100,101]
   - mem[100,61] - mem[100,92] - mem[100,175] + mem_tot_CPU101 = 0
 rows_tot_CPU101: - 3 frac[100,199] - 3 frac[100,8] - 3 frac[100,62]
   - 3 frac[100,228] - 3 frac[100,22] - 3 frac[100,70] - 3 frac[100,165]
   - 3 frac[100,123] - 3 frac[100,105] - 3 frac[100,231] - 3 frac[100,31]
   - 3 frac[100,24] - 3 frac[100,41] - 3 frac[100,101] - 3 frac[100,61]
   - 3 frac[100,92] - 3 frac[100,175] + rows_tot_CPU101 = 0
 mem_tot_CPU102: - mem[101,68] - mem[101,148] - mem[101,177] - mem[101,67]
   - mem[101,105] - mem[101,87] - mem[101,50] - mem[101,18] - mem[101,39]
   + mem_tot_CPU102 = 0
 rows_tot_CPU102: - 3 frac[101,68] - 3 frac[101,148] - 3 frac[101,177]
   - 3 frac[101,67] - 3 frac[101,105] - 3 frac[101,87] - 3 frac[101,50]
   - 3 frac[101,18] - 3 frac[101,39] + rows_tot_CPU102 = 0
 mem_tot_CPU103: - mem[102,182] - mem[102,71] - mem[102,126] - mem[102,113]
   - mem[102,133] - mem[102,214] - mem[102,88] - mem[102,245] - mem[102,18]
   - mem[102,181] - mem[102,221] - mem[102,240] - mem[102,95]
   - mem[102,108] - mem[102,134] - mem[102,17] - mem[102,100]
   + mem_tot_CPU103 = 0
 rows_tot_CPU103: - 3 frac[102,182] - 3 frac[102,71] - 3 frac[102,126]
   - 3 frac[102,113] - 3 frac[102,133] - 3 frac[102,214] - 3 frac[102,88]
   - 3 frac[102,245] - 3 frac[102,18] - 3 frac[102,181] - 3 frac[102,221]
   - 3 frac[102,240] - 3 frac[102,95] - 3 frac[102,108] - 3 frac[102,134]
   - 3 frac[102,17] - 3 frac[102,100] + rows_tot_CPU103 = 0
 mem_tot_CPU104: - mem[103,203] - mem[103,27] - mem[103,249] - mem[103,246]
   - mem[103,158] - mem[103,231] - mem[103,204] - mem[103,152]
   - mem[103,233] - mem[103,171] - mem[103,217] - mem[103,214]
   - mem[103,126] - mem[103,137] - mem[103,170] - mem[103,82]
   - mem[103,245] - mem[103,120] - mem[103,113] - mem[103,58] - mem[103,86]
   + mem_tot_CPU104 = 0
 rows_tot_CPU104: - 3 frac[103,203] - 3 frac[103,27] - 3 frac[103,249]
   - 3 frac[103,246] - 3 frac[103,158] - 3 frac[103,231] - 3 frac[103,204]
   - 3 frac[103,152] - 3 frac[103,233] - 3 frac[103,171] - 3 frac[103,217]
   - 3 frac[103,214] - 3 frac[103,126] - 3 frac[103,137] - 3 frac[103,170]
   - 3 frac[103,82] - 3 frac[103,245] - 3 frac[103,120] - 3 frac[103,113]
   - 3 frac[103,58] - 3 frac[103,86] + rows_tot_CPU104 = 0
 mem_tot_CPU105: - mem[104,86] - mem[104,174] - mem[104,25] - mem[104,213]
   - mem[104,100] - mem[104,170] - mem[104,233] - mem[104,46] - mem[104,73]
   - mem[104,114] - mem[104,121] - mem[104,34] - mem[104,12]
   + mem_tot_CPU105 = 0
 rows_tot_CPU105: - 3 frac[104,86] - 3 frac[104,174] - 3 frac[104,25]
   - 3 frac[104,213] - 3 frac[104,100] - 3 frac[104,170] - 3 frac[104,233]
   - 3 frac[104,46] - 3 frac[104,73] - 3 frac[104,114] - 3 frac[104,121]
   - 3 frac[104,34] - 3 frac[104,12] + rows_tot_CPU105 = 0
 mem_tot_CPU106: - mem[105,20] - mem[105,235] - mem[105,116] - mem[105,225]
   - mem[105,149] - mem[105,136] - mem[105,65] - mem[105,239] - mem[105,58]
   - mem[105,167] - mem[105,237] + mem_tot_CPU106 = 0
 rows_tot_CPU106: - 3 frac[105,20] - 3 frac[105,235] - 3 frac[105,116]
   - 3 frac[105,225] - 3 frac[105,149] - 3 frac[105,136] - 3 frac[105,65]
   - 3 frac[105,239] - 3 frac[105,58] - 3 frac[105,167] - 3 frac[105,237]
   + rows_tot_CPU106 = 0
 mem_tot_CPU107: - mem[106,240] - mem[106,198] - mem[106,199]
   - mem[106,161] - mem[106,67] - mem[106,122] - mem[106,207] - mem[106,98]
   - mem[106,219] - mem[106,179] - mem[106,165] - mem[106,68] - mem[106,90]
   + mem_tot_CPU107 = 0
 rows_tot_CPU107: - 3 frac[106,240] - 3 frac[106,198] - 3 frac[106,199]
   - 3 frac[106,161] - 3 frac[106,67] - 3 frac[106,122] - 3 frac[106,207]
   - 3 frac[106,98] - 3 frac[106,219] - 3 frac[106,179] - 3 frac[106,165]
   - 3 frac[106,68] - 3 frac[106,90] + rows_tot_CPU107 = 0
 mem_tot_CPU108: - mem[107,190] - mem[107,154] - mem[107,12] - mem[107,157]
   - mem[107,119] - mem[107,217] - mem[107,7] - mem[107,215] - mem[107,56]
   - mem[107,226] - mem[107,21] - mem[107,236] - mem[107,69] - mem[107,135]
   - mem[107,2] + mem_tot_CPU108 = 0
 rows_tot_CPU108: - 3 frac[107,190] - 3 frac[107,154] - 3 frac[107,12]
   - 3 frac[107,157] - 3 frac[107,119] - 3 frac[107,217] - 3 frac[107,7]
   - 3 frac[107,215] - 3 frac[107,56] - 3 frac[107,226] - 3 frac[107,21]
   - 3 frac[107,236] - 3 frac[107,69] - 3 frac[107,135] - 3 frac[107,2]
   + rows_tot_CPU108 = 0
 mem_tot_CPU109: - mem[108,117] - mem[108,213] - mem[108,27] - mem[108,90]
   - mem[108,1] - mem[108,101] - mem[108,182] - mem[108,173] - mem[108,227]
   - mem[108,201] - mem[108,106] - mem[108,211] + mem_tot_CPU109 = 0
 rows_tot_CPU109: - 3 frac[108,117] - 3 frac[108,213] - 3 frac[108,27]
   - 3 frac[108,90] - 3 frac[108,1] - 3 frac[108,101] - 3 frac[108,182]
   - 3 frac[108,173] - 3 frac[108,227] - 3 frac[108,201] - 3 frac[108,106]
   - 3 frac[108,211] + rows_tot_CPU109 = 0
 mem_tot_CPU110: - mem[109,18] - mem[109,156] - mem[109,114] - mem[109,3]
   - mem[109,113] - mem[109,2] - mem[109,215] - mem[109,172] - mem[109,117]
   - mem[109,47] - mem[109,164] - mem[109,129] - mem[109,226]
   - mem[109,205] + mem_tot_CPU110 = 0
 rows_tot_CPU110: - 3 frac[109,18] - 3 frac[109,156] - 3 frac[109,114]
   - 3 frac[109,3] - 3 frac[109,113] - 3 frac[109,2] - 3 frac[109,215]
   - 3 frac[109,172] - 3 frac[109,117] - 3 frac[109,47] - 3 frac[109,164]
   - 3 frac[109,129] - 3 frac[109,226] - 3 frac[109,205] + rows_tot_CPU110
   = 0
 mem_tot_CPU111: - mem[110,59] - mem[110,73] - mem[110,99] - mem[110,158]
   - mem[110,192] - mem[110,169] - mem[110,164] - mem[110,206]
   + mem_tot_CPU111 = 0
 rows_tot_CPU111: - 3 frac[110,59] - 3 frac[110,73] - 3 frac[110,99]
   - 3 frac[110,158] - 3 frac[110,192] - 3 frac[110,169] - 3 frac[110,164]
   - 3 frac[110,206] + rows_tot_CPU111 = 0
 mem_tot_CPU112: - mem[111,181] - mem[111,153] - mem[111,70] - mem[111,113]
   - mem[111,230] - mem[111,36] - mem[111,78] - mem[111,17] - mem[111,204]
   - mem[111,2] - mem[111,11] - mem[111,198] - mem[111,13] + mem_tot_CPU112
   = 0
 rows_tot_CPU112: - 3 frac[111,181] - 3 frac[111,153] - 3 frac[111,70]
   - 3 frac[111,113] - 3 frac[111,230] - 3 frac[111,36] - 3 frac[111,78]
   - 3 frac[111,17] - 3 frac[111,204] - 3 frac[111,2] - 3 frac[111,11]
   - 3 frac[111,198] - 3 frac[111,13] + rows_tot_CPU112 = 0
 mem_tot_CPU113: - mem[112,110] - mem[112,4] - mem[112,111] - mem[112,157]
   - mem[112,220] - mem[112,8] - mem[112,167] - mem[112,223] - mem[112,242]
   - mem[112,189] - mem[112,127] - mem[112,181] + mem_tot_CPU113 = 0
 rows_tot_CPU113: - 3 frac[112,110] - 3 frac[112,4] - 3 frac[112,111]
   - 3 frac[112,157] - 3 frac[112,220] - 3 frac[112,8] - 3 frac[112,167]
   - 3 frac[112,223] - 3 frac[112,242] - 3 frac[112,189] - 3 frac[112,127]
   - 3 frac[112,181] + rows_tot_CPU113 = 0
 mem_tot_CPU114: - mem[113,215] - mem[113,244] - mem[113,144]
   - mem[113,187] - mem[113,35] - mem[113,13] - mem[113,151] - mem[113,219]
   - mem[113,143] - mem[113,203] - mem[113,115] - mem[113,169]
   - mem[113,91] - mem[113,168] - mem[113,152] - mem[113,145]
   - mem[113,131] + mem_tot_CPU114 = 0
 rows_tot_CPU114: - 3 frac[113,215] - 3 frac[113,244] - 3 frac[113,144]
   - 3 frac[113,187] - 3 frac[113,35] - 3 frac[113,13] - 3 frac[113,151]
   - 3 frac[113,219] - 3 frac[113,143] - 3 frac[113,203] - 3 frac[113,115]
   - 3 frac[113,169] - 3 frac[113,91] - 3 frac[113,168] - 3 frac[113,152]
   - 3 frac[113,145] - 3 frac[113,131] + rows_tot_CPU114 = 0
 mem_tot_CPU115: - mem[114,60] - mem[114,55] - mem[114,32] - mem[114,124]
   - mem[114,171] - mem[114,23] - mem[114,254] - mem[114,152]
   - mem[114,205] - mem[114,246] - mem[114,139] - mem[114,106]
   + mem_tot_CPU115 = 0
 rows_tot_CPU115: - 3 frac[114,60] - 3 frac[114,55] - 3 frac[114,32]
   - 3 frac[114,124] - 3 frac[114,171] - 3 frac[114,23] - 3 frac[114,254]
   - 3 frac[114,152] - 3 frac[114,205] - 3 frac[114,246] - 3 frac[114,139]
   - 3 frac[114,106] + rows_tot_CPU115 = 0
 mem_tot_CPU116: - mem[115,132] - mem[115,190] - mem[115,16] - mem[115,104]
   - mem[115,32] - mem[115,157] - mem[115,126] - mem[115,142] - mem[115,4]
   - mem[115,101] - mem[115,70] - mem[115,34] - mem[115,238] - mem[115,42]
   - mem[115,153] - mem[115,36] - mem[115,249] + mem_tot_CPU116 = 0
 rows_tot_CPU116: - 3 frac[115,132] - 3 frac[115,190] - 3 frac[115,16]
   - 3 frac[115,104] - 3 frac[115,32] - 3 frac[115,157] - 3 frac[115,126]
   - 3 frac[115,142] - 3 frac[115,4] - 3 frac[115,101] - 3 frac[115,70]
   - 3 frac[115,34] - 3 frac[115,238] - 3 frac[115,42] - 3 frac[115,153]
   - 3 frac[115,36] - 3 frac[115,249] + rows_tot_CPU116 = 0
 mem_tot_CPU117: - mem[116,229] - mem[116,192] - mem[116,19] - mem[116,230]
   - mem[116,175] - mem[116,70] - mem[116,183] - mem[116,133]
   - mem[116,120] - mem[116,239] - mem[116,71] - mem[116,92] - mem[116,216]
   - mem[116,226] - mem[116,186] + mem_tot_CPU117 = 0
 rows_tot_CPU117: - 3 frac[116,229] - 3 frac[116,192] - 3 frac[116,19]
   - 3 frac[116,230] - 3 frac[116,175] - 3 frac[116,70] - 3 frac[116,183]
   - 3 frac[116,133] - 3 frac[116,120] - 3 frac[116,239] - 3 frac[116,71]
   - 3 frac[116,92] - 3 frac[116,216] - 3 frac[116,226] - 3 frac[116,186]
   + rows_tot_CPU117 = 0
 mem_tot_CPU118: - mem[117,104] - mem[117,32] - mem[117,137] - mem[117,112]
   - mem[117,225] - mem[117,66] - mem[117,215] - mem[117,3] - mem[117,120]
   - mem[117,2] - mem[117,179] - mem[117,91] - mem[117,19] - mem[117,72]
   - mem[117,7] - mem[117,255] + mem_tot_CPU118 = 0
 rows_tot_CPU118: - 3 frac[117,104] - 3 frac[117,32] - 3 frac[117,137]
   - 3 frac[117,112] - 3 frac[117,225] - 3 frac[117,66] - 3 frac[117,215]
   - 3 frac[117,3] - 3 frac[117,120] - 3 frac[117,2] - 3 frac[117,179]
   - 3 frac[117,91] - 3 frac[117,19] - 3 frac[117,72] - 3 frac[117,7]
   - 3 frac[117,255] + rows_tot_CPU118 = 0
 mem_tot_CPU119: - mem[118,147] - mem[118,255] - mem[118,134]
   - mem[118,144] - mem[118,226] - mem[118,211] - mem[118,93]
   - mem[118,127] - mem[118,225] - mem[118,41] - mem[118,239]
   - mem[118,182] - mem[118,117] + mem_tot_CPU119 = 0
 rows_tot_CPU119: - 3 frac[118,147] - 3 frac[118,255] - 3 frac[118,134]
   - 3 frac[118,144] - 3 frac[118,226] - 3 frac[118,211] - 3 frac[118,93]
   - 3 frac[118,127] - 3 frac[118,225] - 3 frac[118,41] - 3 frac[118,239]
   - 3 frac[118,182] - 3 frac[118,117] + rows_tot_CPU119 = 0
 mem_tot_CPU120: - mem[119,90] - mem[119,134] - mem[119,148] - mem[119,109]
   - mem[119,188] - mem[119,153] - mem[119,59] - mem[119,152] - mem[119,8]
   - mem[119,77] - mem[119,201] - mem[119,223] - mem[119,173] - mem[119,80]
   - mem[119,38] + mem_tot_CPU120 = 0
 rows_tot_CPU120: - 3 frac[119,90] - 3 frac[119,134] - 3 frac[119,148]
   - 3 frac[119,109] - 3 frac[119,188] - 3 frac[119,153] - 3 frac[119,59]
   - 3 frac[119,152] - 3 frac[119,8] - 3 frac[119,77] - 3 frac[119,201]
   - 3 frac[119,223] - 3 frac[119,173] - 3 frac[119,80] - 3 frac[119,38]
   + rows_tot_CPU120 = 0
 mem_tot_CPU121: - mem[120,231] - mem[120,58] - mem[120,10] - mem[120,45]
   - mem[120,18] - mem[120,137] - mem[120,158] - mem[120,61] - mem[120,20]
   - mem[120,152] - mem[120,232] - mem[120,215] - mem[120,14]
   - mem[120,188] - mem[120,84] - mem[120,154] - mem[120,78]
   + mem_tot_CPU121 = 0
 rows_tot_CPU121: - 3 frac[120,231] - 3 frac[120,58] - 3 frac[120,10]
   - 3 frac[120,45] - 3 frac[120,18] - 3 frac[120,137] - 3 frac[120,158]
   - 3 frac[120,61] - 3 frac[120,20] - 3 frac[120,152] - 3 frac[120,232]
   - 3 frac[120,215] - 3 frac[120,14] - 3 frac[120,188] - 3 frac[120,84]
   - 3 frac[120,154] - 3 frac[120,78] + rows_tot_CPU121 = 0
 mem_tot_CPU122: - mem[121,224] - mem[121,47] - mem[121,100] - mem[121,229]
   - mem[121,157] - mem[121,5] - mem[121,14] - mem[121,92] - mem[121,182]
   - mem[121,58] + mem_tot_CPU122 = 0
 rows_tot_CPU122: - 3 frac[121,224] - 3 frac[121,47] - 3 frac[121,100]
   - 3 frac[121,229] - 3 frac[121,157] - 3 frac[121,5] - 3 frac[121,14]
   - 3 frac[121,92] - 3 frac[121,182] - 3 frac[121,58] + rows_tot_CPU122
   = 0
 mem_tot_CPU123: - mem[122,134] - mem[122,210] - mem[122,246] - mem[122,66]
   - mem[122,84] - mem[122,188] - mem[122,55] - mem[122,227] - mem[122,82]
   + mem_tot_CPU123 = 0
 rows_tot_CPU123: - 3 frac[122,134] - 3 frac[122,210] - 3 frac[122,246]
   - 3 frac[122,66] - 3 frac[122,84] - 3 frac[122,188] - 3 frac[122,55]
   - 3 frac[122,227] - 3 frac[122,82] + rows_tot_CPU123 = 0
 mem_tot_CPU124: - mem[123,31] - mem[123,196] - mem[123,91] - mem[123,227]
   - mem[123,105] - mem[123,90] - mem[123,222] - mem[123,126] - mem[123,71]
   - mem[123,158] - mem[123,55] - mem[123,7] - mem[123,35] - mem[123,225]
   + mem_tot_CPU124 = 0
 rows_tot_CPU124: - 3 frac[123,31] - 3 frac[123,196] - 3 frac[123,91]
   - 3 frac[123,227] - 3 frac[123,105] - 3 frac[123,90] - 3 frac[123,222]
   - 3 frac[123,126] - 3 frac[123,71] - 3 frac[123,158] - 3 frac[123,55]
   - 3 frac[123,7] - 3 frac[123,35] - 3 frac[123,225] + rows_tot_CPU124
   = 0
 mem_tot_CPU125: - mem[124,101] - mem[124,3] - mem[124,114] - mem[124,226]
   - mem[124,41] - mem[124,126] - mem[124,153] - mem[124,91] - mem[124,242]
   - mem[124,194] - mem[124,0] - mem[124,244] - mem[124,160] - mem[124,72]
   - mem[124,99] + mem_tot_CPU125 = 0
 rows_tot_CPU125: - 3 frac[124,101] - 3 frac[124,3] - 3 frac[124,114]
   - 3 frac[124,226] - 3 frac[124,41] - 3 frac[124,126] - 3 frac[124,153]
   - 3 frac[124,91] - 3 frac[124,242] - 3 frac[124,194] - 3 frac[124,0]
   - 3 frac[124,244] - 3 frac[124,160] - 3 frac[124,72] - 3 frac[124,99]
   + rows_tot_CPU125 = 0
 mem_tot_CPU126: - mem[125,240] - mem[125,30] - mem[125,143] - mem[125,165]
   - mem[125,121] - mem[125,252] - mem[125,47] - mem[125,91] - mem[125,182]
   - mem[125,243] - mem[125,82] - mem[125,77] + mem_tot_CPU126 = 0
 rows_tot_CPU126: - 3 frac[125,240] - 3 frac[125,30] - 3 frac[125,143]
   - 3 frac[125,165] - 3 frac[125,121] - 3 frac[125,252] - 3 frac[125,47]
   - 3 frac[125,91] - 3 frac[125,182] - 3 frac[125,243] - 3 frac[125,82]
   - 3 frac[125,77] + rows_tot_CPU126 = 0
 mem_tot_CPU127: - mem[126,26] - mem[126,43] - mem[126,92] - mem[126,255]
   - mem[126,78] - mem[126,106] - mem[126,93] - mem[126,149] - mem[126,171]
   - mem[126,89] - mem[126,75] - mem[126,111] + mem_tot_CPU127 = 0
 rows_tot_CPU127: - 3 frac[126,26] - 3 frac[126,43] - 3 frac[126,92]
   - 3 frac[126,255] - 3 frac[126,78] - 3 frac[126,106] - 3 frac[126,93]
   - 3 frac[126,149] - 3 frac[126,171] - 3 frac[126,89] - 3 frac[126,75]
   - 3 frac[126,111] + rows_tot_CPU127 = 0
 mem_tot_CPU128: - mem[127,174] - mem[127,92] - mem[127,81] - mem[127,188]
   - mem[127,254] - mem[127,79] - mem[127,12] - mem[127,6] - mem[127,167]
   - mem[127,225] - mem[127,175] - mem[127,60] + mem_tot_CPU128 = 0
 rows_tot_CPU128: - 3 frac[127,174] - 3 frac[127,92] - 3 frac[127,81]
   - 3 frac[127,188] - 3 frac[127,254] - 3 frac[127,79] - 3 frac[127,12]
   - 3 frac[127,6] - 3 frac[127,167] - 3 frac[127,225] - 3 frac[127,175]
   - 3 frac[127,60] + rows_tot_CPU128 = 0
 mem_tot_CPU129: - mem[128,144] - mem[128,15] - mem[128,151] - mem[128,255]
   - mem[128,110] - mem[128,205] - mem[128,219] - mem[128,88]
   - mem[128,167] - mem[128,141] - mem[128,60] - mem[128,16]
   + mem_tot_CPU129 = 0
 rows_tot_CPU129: - 3 frac[128,144] - 3 frac[128,15] - 3 frac[128,151]
   - 3 frac[128,255] - 3 frac[128,110] - 3 frac[128,205] - 3 frac[128,219]
   - 3 frac[128,88] - 3 frac[128,167] - 3 frac[128,141] - 3 frac[128,60]
   - 3 frac[128,16] + rows_tot_CPU129 = 0
 mem_tot_CPU130: - mem[129,225] - mem[129,167] - mem[129,184]
   - mem[129,150] - mem[129,126] - mem[129,72] - mem[129,90] - mem[129,24]
   - mem[129,220] - mem[129,192] + mem_tot_CPU130 = 0
 rows_tot_CPU130: - 3 frac[129,225] - 3 frac[129,167] - 3 frac[129,184]
   - 3 frac[129,150] - 3 frac[129,126] - 3 frac[129,72] - 3 frac[129,90]
   - 3 frac[129,24] - 3 frac[129,220] - 3 frac[129,192] + rows_tot_CPU130
   = 0
 mem_tot_CPU131: - mem[130,250] - mem[130,67] - mem[130,205] - mem[130,101]
   - mem[130,158] - mem[130,119] - mem[130,153] - mem[130,26]
   - mem[130,221] - mem[130,246] - mem[130,74] + mem_tot_CPU131 = 0
 rows_tot_CPU131: - 3 frac[130,250] - 3 frac[130,67] - 3 frac[130,205]
   - 3 frac[130,101] - 3 frac[130,158] - 3 frac[130,119] - 3 frac[130,153]
   - 3 frac[130,26] - 3 frac[130,221] - 3 frac[130,246] - 3 frac[130,74]
   + rows_tot_CPU131 = 0
 mem_tot_CPU132: - mem[131,181] - mem[131,26] - mem[131,251] - mem[131,47]
   - mem[131,140] - mem[131,152] - mem[131,222] - mem[131,111]
   - mem[131,114] - mem[131,172] - mem[131,34] - mem[131,219]
   + mem_tot_CPU132 = 0
 rows_tot_CPU132: - 3 frac[131,181] - 3 frac[131,26] - 3 frac[131,251]
   - 3 frac[131,47] - 3 frac[131,140] - 3 frac[131,152] - 3 frac[131,222]
   - 3 frac[131,111] - 3 frac[131,114] - 3 frac[131,172] - 3 frac[131,34]
   - 3 frac[131,219] + rows_tot_CPU132 = 0
 mem_tot_CPU133: - mem[132,58] - mem[132,30] - mem[132,12] - mem[132,243]
   - mem[132,17] - mem[132,7] - mem[132,175] - mem[132,132] - mem[132,251]
   - mem[132,161] - mem[132,1] - mem[132,141] - mem[132,97] - mem[132,240]
   - mem[132,117] + mem_tot_CPU133 = 0
 rows_tot_CPU133: - 3 frac[132,58] - 3 frac[132,30] - 3 frac[132,12]
   - 3 frac[132,243] - 3 frac[132,17] - 3 frac[132,7] - 3 frac[132,175]
   - 3 frac[132,132] - 3 frac[132,251] - 3 frac[132,161] - 3 frac[132,1]
   - 3 frac[132,141] - 3 frac[132,97] - 3 frac[132,240] - 3 frac[132,117]
   + rows_tot_CPU133 = 0
 mem_tot_CPU134: - mem[133,100] - mem[133,245] - mem[133,254]
   - mem[133,233] - mem[133,219] - mem[133,232] - mem[133,119]
   - mem[133,243] - mem[133,79] - mem[133,39] - mem[133,32] - mem[133,78]
   - mem[133,143] - mem[133,96] - mem[133,191] - mem[133,102]
   - mem[133,240] + mem_tot_CPU134 = 0
 rows_tot_CPU134: - 3 frac[133,100] - 3 frac[133,245] - 3 frac[133,254]
   - 3 frac[133,233] - 3 frac[133,219] - 3 frac[133,232] - 3 frac[133,119]
   - 3 frac[133,243] - 3 frac[133,79] - 3 frac[133,39] - 3 frac[133,32]
   - 3 frac[133,78] - 3 frac[133,143] - 3 frac[133,96] - 3 frac[133,191]
   - 3 frac[133,102] - 3 frac[133,240] + rows_tot_CPU134 = 0
 mem_tot_CPU135: - mem[134,55] - mem[134,11] - mem[134,8] - mem[134,3]
   - mem[134,208] - mem[134,194] - mem[134,251] - mem[134,99]
   - mem[134,195] - mem[134,71] - mem[134,49] - mem[134,187] - mem[134,145]
   - mem[134,229] - mem[134,179] - mem[134,89] + mem_tot_CPU135 = 0
 rows_tot_CPU135: - 3 frac[134,55] - 3 frac[134,11] - 3 frac[134,8]
   - 3 frac[134,3] - 3 frac[134,208] - 3 frac[134,194] - 3 frac[134,251]
   - 3 frac[134,99] - 3 frac[134,195] - 3 frac[134,71] - 3 frac[134,49]
   - 3 frac[134,187] - 3 frac[134,145] - 3 frac[134,229] - 3 frac[134,179]
   - 3 frac[134,89] + rows_tot_CPU135 = 0
 mem_tot_CPU136: - mem[135,162] - mem[135,23] - mem[135,125] - mem[135,229]
   - mem[135,94] - mem[135,174] - mem[135,92] - mem[135,114] - mem[135,6]
   - mem[135,108] - mem[135,171] - mem[135,160] - mem[135,132]
   - mem[135,222] + mem_tot_CPU136 = 0
 rows_tot_CPU136: - 3 frac[135,162] - 3 frac[135,23] - 3 frac[135,125]
   - 3 frac[135,229] - 3 frac[135,94] - 3 frac[135,174] - 3 frac[135,92]
   - 3 frac[135,114] - 3 frac[135,6] - 3 frac[135,108] - 3 frac[135,171]
   - 3 frac[135,160] - 3 frac[135,132] - 3 frac[135,222] + rows_tot_CPU136
   = 0
 mem_tot_CPU137: - mem[136,67] - mem[136,233] - mem[136,48] - mem[136,121]
   - mem[136,212] - mem[136,170] - mem[136,136] - mem[136,58] - mem[136,89]
   - mem[136,248] - mem[136,148] - mem[136,155] - mem[136,72]
   - mem[136,131] - mem[136,184] - mem[136,204] + mem_tot_CPU137 = 0
 rows_tot_CPU137: - 3 frac[136,67] - 3 frac[136,233] - 3 frac[136,48]
   - 3 frac[136,121] - 3 frac[136,212] - 3 frac[136,170] - 3 frac[136,136]
   - 3 frac[136,58] - 3 frac[136,89] - 3 frac[136,248] - 3 frac[136,148]
   - 3 frac[136,155] - 3 frac[136,72] - 3 frac[136,131] - 3 frac[136,184]
   - 3 frac[136,204] + rows_tot_CPU137 = 0
 mem_tot_CPU138: - mem[137,227] - mem[137,161] - mem[137,8] - mem[137,35]
   - mem[137,168] - mem[137,6] - mem[137,175] - mem[137,165] - mem[137,93]
   - mem[137,215] - mem[137,216] + mem_tot_CPU138 = 0
 rows_tot_CPU138: - 3 frac[137,227] - 3 frac[137,161] - 3 frac[137,8]
   - 3 frac[137,35] - 3 frac[137,168] - 3 frac[137,6] - 3 frac[137,175]
   - 3 frac[137,165] - 3 frac[137,93] - 3 frac[137,215] - 3 frac[137,216]
   + rows_tot_CPU138 = 0
 mem_tot_CPU139: - mem[138,44] - mem[138,198] - mem[138,243] - mem[138,250]
   - mem[138,221] - mem[138,16] - mem[138,156] - mem[138,132] - mem[138,39]
   - mem[138,231] - mem[138,6] - mem[138,1] + mem_tot_CPU139 = 0
 rows_tot_CPU139: - 3 frac[138,44] - 3 frac[138,198] - 3 frac[138,243]
   - 3 frac[138,250] - 3 frac[138,221] - 3 frac[138,16] - 3 frac[138,156]
   - 3 frac[138,132] - 3 frac[138,39] - 3 frac[138,231] - 3 frac[138,6]
   - 3 frac[138,1] + rows_tot_CPU139 = 0
 mem_tot_CPU140: - mem[139,254] - mem[139,159] - mem[139,158] - mem[139,62]
   - mem[139,83] - mem[139,240] - mem[139,132] - mem[139,217] - mem[139,75]
   - mem[139,188] - mem[139,216] - mem[139,201] - mem[139,157]
   - mem[139,123] - mem[139,190] + mem_tot_CPU140 = 0
 rows_tot_CPU140: - 3 frac[139,254] - 3 frac[139,159] - 3 frac[139,158]
   - 3 frac[139,62] - 3 frac[139,83] - 3 frac[139,240] - 3 frac[139,132]
   - 3 frac[139,217] - 3 frac[139,75] - 3 frac[139,188] - 3 frac[139,216]
   - 3 frac[139,201] - 3 frac[139,157] - 3 frac[139,123] - 3 frac[139,190]
   + rows_tot_CPU140 = 0
 mem_tot_CPU141: - mem[140,153] - mem[140,123] - mem[140,221] - mem[140,58]
   - mem[140,116] - mem[140,137] - mem[140,203] - mem[140,108]
   - mem[140,214] - mem[140,17] - mem[140,252] - mem[140,83] - mem[140,217]
   - mem[140,133] - mem[140,245] + mem_tot_CPU141 = 0
 rows_tot_CPU141: - 3 frac[140,153] - 3 frac[140,123] - 3 frac[140,221]
   - 3 frac[140,58] - 3 frac[140,116] - 3 frac[140,137] - 3 frac[140,203]
   - 3 frac[140,108] - 3 frac[140,214] - 3 frac[140,17] - 3 frac[140,252]
   - 3 frac[140,83] - 3 frac[140,217] - 3 frac[140,133] - 3 frac[140,245]
   + rows_tot_CPU141 = 0
 mem_tot_CPU142: - mem[141,162] - mem[141,153] - mem[141,146]
   - mem[141,161] - mem[141,210] - mem[141,252] - mem[141,19] - mem[141,12]
   - mem[141,238] - mem[141,91] - mem[141,201] - mem[141,230]
   - mem[141,142] - mem[141,3] - mem[141,229] - mem[141,125] - mem[141,38]
   + mem_tot_CPU142 = 0
 rows_tot_CPU142: - 3 frac[141,162] - 3 frac[141,153] - 3 frac[141,146]
   - 3 frac[141,161] - 3 frac[141,210] - 3 frac[141,252] - 3 frac[141,19]
   - 3 frac[141,12] - 3 frac[141,238] - 3 frac[141,91] - 3 frac[141,201]
   - 3 frac[141,230] - 3 frac[141,142] - 3 frac[141,3] - 3 frac[141,229]
   - 3 frac[141,125] - 3 frac[141,38] + rows_tot_CPU142 = 0
 mem_tot_CPU143: - mem[142,184] - mem[142,148] - mem[142,7] - mem[142,110]
   - mem[142,48] - mem[142,2] - mem[142,178] - mem[142,142] - mem[142,11]
   - mem[142,249] + mem_tot_CPU143 = 0
 rows_tot_CPU143: - 3 frac[142,184] - 3 frac[142,148] - 3 frac[142,7]
   - 3 frac[142,110] - 3 frac[142,48] - 3 frac[142,2] - 3 frac[142,178]
   - 3 frac[142,142] - 3 frac[142,11] - 3 frac[142,249] + rows_tot_CPU143
   = 0
 mem_tot_CPU144: - mem[143,144] - mem[143,250] - mem[143,163] - mem[143,61]
   - mem[143,30] - mem[143,254] - mem[143,81] - mem[143,184] - mem[143,119]
   - mem[143,75] - mem[143,206] - mem[143,215] - mem[143,34] - mem[143,83]
   + mem_tot_CPU144 = 0
 rows_tot_CPU144: - 3 frac[143,144] - 3 frac[143,250] - 3 frac[143,163]
   - 3 frac[143,61] - 3 frac[143,30] - 3 frac[143,254] - 3 frac[143,81]
   - 3 frac[143,184] - 3 frac[143,119] - 3 frac[143,75] - 3 frac[143,206]
   - 3 frac[143,215] - 3 frac[143,34] - 3 frac[143,83] + rows_tot_CPU144
   = 0
 mem_tot_CPU145: - mem[144,7] - mem[144,239] - mem[144,99] - mem[144,90]
   - mem[144,73] - mem[144,42] - mem[144,86] - mem[144,15] + mem_tot_CPU145
   = 0
 rows_tot_CPU145: - 3 frac[144,7] - 3 frac[144,239] - 3 frac[144,99]
   - 3 frac[144,90] - 3 frac[144,73] - 3 frac[144,42] - 3 frac[144,86]
   - 3 frac[144,15] + rows_tot_CPU145 = 0
 mem_tot_CPU146: - mem[145,10] - mem[145,93] - mem[145,44] - mem[145,133]
   - mem[145,69] - mem[145,13] - mem[145,178] + mem_tot_CPU146 = 0
 rows_tot_CPU146: - 3 frac[145,10] - 3 frac[145,93] - 3 frac[145,44]
   - 3 frac[145,133] - 3 frac[145,69] - 3 frac[145,13] - 3 frac[145,178]
   + rows_tot_CPU146 = 0
 mem_tot_CPU147: - mem[146,166] - mem[146,72] - mem[146,34] - mem[146,29]
   - mem[146,153] - mem[146,73] - mem[146,7] - mem[146,250] - mem[146,217]
   - mem[146,148] - mem[146,82] - mem[146,191] - mem[146,111]
   - mem[146,226] - mem[146,207] - mem[146,3] - mem[146,241]
   + mem_tot_CPU147 = 0
 rows_tot_CPU147: - 3 frac[146,166] - 3 frac[146,72] - 3 frac[146,34]
   - 3 frac[146,29] - 3 frac[146,153] - 3 frac[146,73] - 3 frac[146,7]
   - 3 frac[146,250] - 3 frac[146,217] - 3 frac[146,148] - 3 frac[146,82]
   - 3 frac[146,191] - 3 frac[146,111] - 3 frac[146,226] - 3 frac[146,207]
   - 3 frac[146,3] - 3 frac[146,241] + rows_tot_CPU147 = 0
 mem_tot_CPU148: - mem[147,199] - mem[147,89] - mem[147,241] - mem[147,55]
   - mem[147,223] - mem[147,133] - mem[147,37] - mem[147,42] - mem[147,51]
   - mem[147,250] - mem[147,174] - mem[147,36] - mem[147,59] - mem[147,73]
   - mem[147,253] - mem[147,67] - mem[147,19] + mem_tot_CPU148 = 0
 rows_tot_CPU148: - 3 frac[147,199] - 3 frac[147,89] - 3 frac[147,241]
   - 3 frac[147,55] - 3 frac[147,223] - 3 frac[147,133] - 3 frac[147,37]
   - 3 frac[147,42] - 3 frac[147,51] - 3 frac[147,250] - 3 frac[147,174]
   - 3 frac[147,36] - 3 frac[147,59] - 3 frac[147,73] - 3 frac[147,253]
   - 3 frac[147,67] - 3 frac[147,19] + rows_tot_CPU148 = 0
 mem_tot_CPU149: - mem[148,205] - mem[148,68] - mem[148,220] - mem[148,2]
   - mem[148,169] - mem[148,77] - mem[148,229] - mem[148,90] - mem[148,54]
   - mem[148,92] - mem[148,79] - mem[148,48] - mem[148,72] + mem_tot_CPU149
   = 0
 rows_tot_CPU149: - 3 frac[148,205] - 3 frac[148,68] - 3 frac[148,220]
   - 3 frac[148,2] - 3 frac[148,169] - 3 frac[148,77] - 3 frac[148,229]
   - 3 frac[148,90] - 3 frac[148,54] - 3 frac[148,92] - 3 frac[148,79]
   - 3 frac[148,48] - 3 frac[148,72] + rows_tot_CPU149 = 0
 mem_tot_CPU150: - mem[149,87] - mem[149,12] - mem[149,113] - mem[149,122]
   - mem[149,45] - mem[149,73] - mem[149,224] - mem[149,120] - mem[149,34]
   - mem[149,185] + mem_tot_CPU150 = 0
 rows_tot_CPU150: - 3 frac[149,87] - 3 frac[149,12] - 3 frac[149,113]
   - 3 frac[149,122] - 3 frac[149,45] - 3 frac[149,73] - 3 frac[149,224]
   - 3 frac[149,120] - 3 frac[149,34] - 3 frac[149,185] + rows_tot_CPU150
   = 0
 mem_tot_CPU151: - mem[150,212] - mem[150,24] - mem[150,125] - mem[150,254]
   - mem[150,131] - mem[150,4] - mem[150,31] - mem[150,129] - mem[150,59]
   - mem[150,53] - mem[150,209] - mem[150,189] - mem[150,180] - mem[150,21]
   - mem[150,132] + mem_tot_CPU151 = 0
 rows_tot_CPU151: - 3 frac[150,212] - 3 frac[150,24] - 3 frac[150,125]
   - 3 frac[150,254] - 3 frac[150,131] - 3 frac[150,4] - 3 frac[150,31]
   - 3 frac[150,129] - 3 frac[150,59] - 3 frac[150,53] - 3 frac[150,209]
   - 3 frac[150,189] - 3 frac[150,180] - 3 frac[150,21] - 3 frac[150,132]
   + rows_tot_CPU151 = 0
 mem_tot_CPU152: - mem[151,195] - mem[151,190] - mem[151,36] - mem[151,244]
   - mem[151,98] - mem[151,134] - mem[151,88] + mem_tot_CPU152 = 0
 rows_tot_CPU152: - 3 frac[151,195] - 3 frac[151,190] - 3 frac[151,36]
   - 3 frac[151,244] - 3 frac[151,98] - 3 frac[151,134] - 3 frac[151,88]
   + rows_tot_CPU152 = 0
 mem_tot_CPU153: - mem[152,55] - mem[152,150] - mem[152,7] - mem[152,244]
   - mem[152,161] - mem[152,168] - mem[152,182] - mem[152,91] - mem[152,58]
   - mem[152,1] - mem[152,211] - mem[152,198] + mem_tot_CPU153 = 0
 rows_tot_CPU153: - 3 frac[152,55] - 3 frac[152,150] - 3 frac[152,7]
   - 3 frac[152,244] - 3 frac[152,161] - 3 frac[152,168] - 3 frac[152,182]
   - 3 frac[152,91] - 3 frac[152,58] - 3 frac[152,1] - 3 frac[152,211]
   - 3 frac[152,198] + rows_tot_CPU153 = 0
 mem_tot_CPU154: - mem[153,15] - mem[153,192] - mem[153,165] - mem[153,183]
   - mem[153,208] - mem[153,93] - mem[153,224] - mem[153,54] - mem[153,168]
   - mem[153,177] - mem[153,159] - mem[153,247] + mem_tot_CPU154 = 0
 rows_tot_CPU154: - 3 frac[153,15] - 3 frac[153,192] - 3 frac[153,165]
   - 3 frac[153,183] - 3 frac[153,208] - 3 frac[153,93] - 3 frac[153,224]
   - 3 frac[153,54] - 3 frac[153,168] - 3 frac[153,177] - 3 frac[153,159]
   - 3 frac[153,247] + rows_tot_CPU154 = 0
 mem_tot_CPU155: - mem[154,2] - mem[154,155] - mem[154,94] - mem[154,143]
   - mem[154,63] - mem[154,75] - mem[154,82] - mem[154,184] - mem[154,212]
   - mem[154,15] - mem[154,25] - mem[154,247] + mem_tot_CPU155 = 0
 rows_tot_CPU155: - 3 frac[154,2] - 3 frac[154,155] - 3 frac[154,94]
   - 3 frac[154,143] - 3 frac[154,63] - 3 frac[154,75] - 3 frac[154,82]
   - 3 frac[154,184] - 3 frac[154,212] - 3 frac[154,15] - 3 frac[154,25]
   - 3 frac[154,247] + rows_tot_CPU155 = 0
 mem_tot_CPU156: - mem[155,63] - mem[155,79] - mem[155,57] - mem[155,108]
   - mem[155,75] - mem[155,207] - mem[155,190] - mem[155,110]
   - mem[155,140] - mem[155,206] - mem[155,170] - mem[155,182]
   - mem[155,66] - mem[155,121] + mem_tot_CPU156 = 0
 rows_tot_CPU156: - 3 frac[155,63] - 3 frac[155,79] - 3 frac[155,57]
   - 3 frac[155,108] - 3 frac[155,75] - 3 frac[155,207] - 3 frac[155,190]
   - 3 frac[155,110] - 3 frac[155,140] - 3 frac[155,206] - 3 frac[155,170]
   - 3 frac[155,182] - 3 frac[155,66] - 3 frac[155,121] + rows_tot_CPU156
   = 0
 mem_tot_CPU157: - mem[156,214] - mem[156,181] - mem[156,58] - mem[156,243]
   - mem[156,182] - mem[156,194] - mem[156,85] - mem[156,61] - mem[156,134]
   - mem[156,2] - mem[156,226] - mem[156,126] - mem[156,198] - mem[156,180]
   - mem[156,43] + mem_tot_CPU157 = 0
 rows_tot_CPU157: - 3 frac[156,214] - 3 frac[156,181] - 3 frac[156,58]
   - 3 frac[156,243] - 3 frac[156,182] - 3 frac[156,194] - 3 frac[156,85]
   - 3 frac[156,61] - 3 frac[156,134] - 3 frac[156,2] - 3 frac[156,226]
   - 3 frac[156,126] - 3 frac[156,198] - 3 frac[156,180] - 3 frac[156,43]
   + rows_tot_CPU157 = 0
 mem_tot_CPU158: - mem[157,67] - mem[157,46] - mem[157,182] - mem[157,84]
   - mem[157,225] - mem[157,222] - mem[157,162] - mem[157,224]
   - mem[157,221] - mem[157,64] - mem[157,43] - mem[157,213] - mem[157,174]
   - mem[157,185] - mem[157,229] - mem[157,122] - mem[157,138]
   - mem[157,24] - mem[157,207] - mem[157,93] - mem[157,155]
   + mem_tot_CPU158 = 0
 rows_tot_CPU158: - 3 frac[157,67] - 3 frac[157,46] - 3 frac[157,182]
   - 3 frac[157,84] - 3 frac[157,225] - 3 frac[157,222] - 3 frac[157,162]
   - 3 frac[157,224] - 3 frac[157,221] - 3 frac[157,64] - 3 frac[157,43]
   - 3 frac[157,213] - 3 frac[157,174] - 3 frac[157,185] - 3 frac[157,229]
   - 3 frac[157,122] - 3 frac[157,138] - 3 frac[157,24] - 3 frac[157,207]
   - 3 frac[157,93] - 3 frac[157,155] + rows_tot_CPU158 = 0
 mem_tot_CPU159: - mem[158,160] - mem[158,142] - mem[158,227]
   - mem[158,132] - mem[158,210] - mem[158,169] - mem[158,42]
   - mem[158,186] - mem[158,21] - mem[158,71] - mem[158,240]
   + mem_tot_CPU159 = 0
 rows_tot_CPU159: - 3 frac[158,160] - 3 frac[158,142] - 3 frac[158,227]
   - 3 frac[158,132] - 3 frac[158,210] - 3 frac[158,169] - 3 frac[158,42]
   - 3 frac[158,186] - 3 frac[158,21] - 3 frac[158,71] - 3 frac[158,240]
   + rows_tot_CPU159 = 0
 mem_tot_CPU160: - mem[159,200] - mem[159,41] - mem[159,7] - mem[159,100]
   - mem[159,214] - mem[159,6] - mem[159,90] - mem[159,235] - mem[159,0]
   - mem[159,20] - mem[159,117] - mem[159,114] - mem[159,3] - mem[159,254]
   - mem[159,240] + mem_tot_CPU160 = 0
 rows_tot_CPU160: - 3 frac[159,200] - 3 frac[159,41] - 3 frac[159,7]
   - 3 frac[159,100] - 3 frac[159,214] - 3 frac[159,6] - 3 frac[159,90]
   - 3 frac[159,235] - 3 frac[159,0] - 3 frac[159,20] - 3 frac[159,117]
   - 3 frac[159,114] - 3 frac[159,3] - 3 frac[159,254] - 3 frac[159,240]
   + rows_tot_CPU160 = 0
 mem_tot_CPU161: - mem[160,101] - mem[160,41] - mem[160,62] - mem[160,164]
   - mem[160,130] - mem[160,144] - mem[160,35] - mem[160,155]
   - mem[160,181] - mem[160,73] - mem[160,184] + mem_tot_CPU161 = 0
 rows_tot_CPU161: - 3 frac[160,101] - 3 frac[160,41] - 3 frac[160,62]
   - 3 frac[160,164] - 3 frac[160,130] - 3 frac[160,144] - 3 frac[160,35]
   - 3 frac[160,155] - 3 frac[160,181] - 3 frac[160,73] - 3 frac[160,184]
   + rows_tot_CPU161 = 0
 mem_tot_CPU162: - mem[161,149] - mem[161,5] - mem[161,30] - mem[161,170]
   - mem[161,21] - mem[161,19] - mem[161,78] - mem[161,63] - mem[161,239]
   - mem[161,81] + mem_tot_CPU162 = 0
 rows_tot_CPU162: - 3 frac[161,149] - 3 frac[161,5] - 3 frac[161,30]
   - 3 frac[161,170] - 3 frac[161,21] - 3 frac[161,19] - 3 frac[161,78]
   - 3 frac[161,63] - 3 frac[161,239] - 3 frac[161,81] + rows_tot_CPU162
   = 0
 mem_tot_CPU163: - mem[162,99] - mem[162,158] - mem[162,108] - mem[162,227]
   - mem[162,30] - mem[162,16] - mem[162,155] + mem_tot_CPU163 = 0
 rows_tot_CPU163: - 3 frac[162,99] - 3 frac[162,158] - 3 frac[162,108]
   - 3 frac[162,227] - 3 frac[162,30] - 3 frac[162,16] - 3 frac[162,155]
   + rows_tot_CPU163 = 0
 mem_tot_CPU164: - mem[163,60] - mem[163,121] - mem[163,153] - mem[163,163]
   - mem[163,168] - mem[163,143] - mem[163,137] - mem[163,46]
   - mem[163,173] - mem[163,202] - mem[163,133] - mem[163,67]
   - mem[163,113] + mem_tot_CPU164 = 0
 rows_tot_CPU164: - 3 frac[163,60] - 3 frac[163,121] - 3 frac[163,153]
   - 3 frac[163,163] - 3 frac[163,168] - 3 frac[163,143] - 3 frac[163,137]
   - 3 frac[163,46] - 3 frac[163,173] - 3 frac[163,202] - 3 frac[163,133]
   - 3 frac[163,67] - 3 frac[163,113] + rows_tot_CPU164 = 0
 mem_tot_CPU165: - mem[164,170] - mem[164,148] - mem[164,251]
   - mem[164,210] - mem[164,7] - mem[164,95] - mem[164,64] - mem[164,248]
   - mem[164,45] - mem[164,249] - mem[164,181] - mem[164,169]
   + mem_tot_CPU165 = 0
 rows_tot_CPU165: - 3 frac[164,170] - 3 frac[164,148] - 3 frac[164,251]
   - 3 frac[164,210] - 3 frac[164,7] - 3 frac[164,95] - 3 frac[164,64]
   - 3 frac[164,248] - 3 frac[164,45] - 3 frac[164,249] - 3 frac[164,181]
   - 3 frac[164,169] + rows_tot_CPU165 = 0
 mem_tot_CPU166: - mem[165,98] - mem[165,181] - mem[165,35] - mem[165,243]
   - mem[165,144] - mem[165,188] - mem[165,56] - mem[165,33] - mem[165,200]
   - mem[165,66] - mem[165,2] - mem[165,11] + mem_tot_CPU166 = 0
 rows_tot_CPU166: - 3 frac[165,98] - 3 frac[165,181] - 3 frac[165,35]
   - 3 frac[165,243] - 3 frac[165,144] - 3 frac[165,188] - 3 frac[165,56]
   - 3 frac[165,33] - 3 frac[165,200] - 3 frac[165,66] - 3 frac[165,2]
   - 3 frac[165,11] + rows_tot_CPU166 = 0
 mem_tot_CPU167: - mem[166,185] - mem[166,17] - mem[166,149] - mem[166,66]
   - mem[166,31] - mem[166,9] - mem[166,226] - mem[166,132] - mem[166,51]
   - mem[166,136] - mem[166,194] - mem[166,70] - mem[166,121]
   - mem[166,250] + mem_tot_CPU167 = 0
 rows_tot_CPU167: - 3 frac[166,185] - 3 frac[166,17] - 3 frac[166,149]
   - 3 frac[166,66] - 3 frac[166,31] - 3 frac[166,9] - 3 frac[166,226]
   - 3 frac[166,132] - 3 frac[166,51] - 3 frac[166,136] - 3 frac[166,194]
   - 3 frac[166,70] - 3 frac[166,121] - 3 frac[166,250] + rows_tot_CPU167
   = 0
 mem_tot_CPU168: - mem[167,220] - mem[167,170] - mem[167,59] - mem[167,38]
   - mem[167,107] - mem[167,73] - mem[167,141] - mem[167,253] - mem[167,6]
   - mem[167,85] - mem[167,75] - mem[167,145] + mem_tot_CPU168 = 0
 rows_tot_CPU168: - 3 frac[167,220] - 3 frac[167,170] - 3 frac[167,59]
   - 3 frac[167,38] - 3 frac[167,107] - 3 frac[167,73] - 3 frac[167,141]
   - 3 frac[167,253] - 3 frac[167,6] - 3 frac[167,85] - 3 frac[167,75]
   - 3 frac[167,145] + rows_tot_CPU168 = 0
 mem_tot_CPU169: - mem[168,116] - mem[168,135] - mem[168,80] - mem[168,51]
   - mem[168,16] - mem[168,30] - mem[168,155] - mem[168,89] - mem[168,53]
   - mem[168,185] - mem[168,226] - mem[168,48] - mem[168,55]
   + mem_tot_CPU169 = 0
 rows_tot_CPU169: - 3 frac[168,116] - 3 frac[168,135] - 3 frac[168,80]
   - 3 frac[168,51] - 3 frac[168,16] - 3 frac[168,30] - 3 frac[168,155]
   - 3 frac[168,89] - 3 frac[168,53] - 3 frac[168,185] - 3 frac[168,226]
   - 3 frac[168,48] - 3 frac[168,55] + rows_tot_CPU169 = 0
 mem_tot_CPU170: - mem[169,227] - mem[169,237] - mem[169,126]
   - mem[169,157] - mem[169,249] - mem[169,142] - mem[169,31]
   - mem[169,124] - mem[169,110] - mem[169,74] - mem[169,44] - mem[169,21]
   - mem[169,246] - mem[169,144] - mem[169,253] - mem[169,39] - mem[169,38]
   - mem[169,2] + mem_tot_CPU170 = 0
 rows_tot_CPU170: - 3 frac[169,227] - 3 frac[169,237] - 3 frac[169,126]
   - 3 frac[169,157] - 3 frac[169,249] - 3 frac[169,142] - 3 frac[169,31]
   - 3 frac[169,124] - 3 frac[169,110] - 3 frac[169,74] - 3 frac[169,44]
   - 3 frac[169,21] - 3 frac[169,246] - 3 frac[169,144] - 3 frac[169,253]
   - 3 frac[169,39] - 3 frac[169,38] - 3 frac[169,2] + rows_tot_CPU170 = 0
 mem_tot_CPU171: - mem[170,7] - mem[170,14] - mem[170,189] - mem[170,172]
   - mem[170,33] - mem[170,162] - mem[170,148] - mem[170,230]
   - mem[170,112] - mem[170,3] - mem[170,72] - mem[170,143] - mem[170,11]
   - mem[170,194] - mem[170,71] - mem[170,195] - mem[170,49]
   + mem_tot_CPU171 = 0
 rows_tot_CPU171: - 3 frac[170,7] - 3 frac[170,14] - 3 frac[170,189]
   - 3 frac[170,172] - 3 frac[170,33] - 3 frac[170,162] - 3 frac[170,148]
   - 3 frac[170,230] - 3 frac[170,112] - 3 frac[170,3] - 3 frac[170,72]
   - 3 frac[170,143] - 3 frac[170,11] - 3 frac[170,194] - 3 frac[170,71]
   - 3 frac[170,195] - 3 frac[170,49] + rows_tot_CPU171 = 0
 mem_tot_CPU172: - mem[171,55] - mem[171,26] - mem[171,158] - mem[171,6]
   - mem[171,116] - mem[171,188] - mem[171,235] - mem[171,190]
   - mem[171,123] - mem[171,193] - mem[171,149] - mem[171,248]
   - mem[171,246] - mem[171,69] - mem[171,161] + mem_tot_CPU172 = 0
 rows_tot_CPU172: - 3 frac[171,55] - 3 frac[171,26] - 3 frac[171,158]
   - 3 frac[171,6] - 3 frac[171,116] - 3 frac[171,188] - 3 frac[171,235]
   - 3 frac[171,190] - 3 frac[171,123] - 3 frac[171,193] - 3 frac[171,149]
   - 3 frac[171,248] - 3 frac[171,246] - 3 frac[171,69] - 3 frac[171,161]
   + rows_tot_CPU172 = 0
 mem_tot_CPU173: - mem[172,221] - mem[172,155] - mem[172,89] - mem[172,204]
   - mem[172,176] - mem[172,6] - mem[172,132] - mem[172,83] - mem[172,198]
   - mem[172,180] - mem[172,48] - mem[172,136] - mem[172,240]
   + mem_tot_CPU173 = 0
 rows_tot_CPU173: - 3 frac[172,221] - 3 frac[172,155] - 3 frac[172,89]
   - 3 frac[172,204] - 3 frac[172,176] - 3 frac[172,6] - 3 frac[172,132]
   - 3 frac[172,83] - 3 frac[172,198] - 3 frac[172,180] - 3 frac[172,48]
   - 3 frac[172,136] - 3 frac[172,240] + rows_tot_CPU173 = 0
 mem_tot_CPU174: - mem[173,87] - mem[173,80] - mem[173,167] - mem[173,100]
   - mem[173,204] - mem[173,47] - mem[173,151] - mem[173,195]
   - mem[173,215] - mem[173,184] - mem[173,242] - mem[173,235]
   - mem[173,75] - mem[173,227] - mem[173,57] - mem[173,161]
   + mem_tot_CPU174 = 0
 rows_tot_CPU174: - 3 frac[173,87] - 3 frac[173,80] - 3 frac[173,167]
   - 3 frac[173,100] - 3 frac[173,204] - 3 frac[173,47] - 3 frac[173,151]
   - 3 frac[173,195] - 3 frac[173,215] - 3 frac[173,184] - 3 frac[173,242]
   - 3 frac[173,235] - 3 frac[173,75] - 3 frac[173,227] - 3 frac[173,57]
   - 3 frac[173,161] + rows_tot_CPU174 = 0
 mem_tot_CPU175: - mem[174,215] - mem[174,111] - mem[174,243] - mem[174,75]
   - mem[174,240] - mem[174,210] - mem[174,58] - mem[174,183]
   - mem[174,193] - mem[174,145] - mem[174,173] - mem[174,104]
   - mem[174,56] - mem[174,237] - mem[174,64] - mem[174,16] - mem[174,252]
   - mem[174,62] + mem_tot_CPU175 = 0
 rows_tot_CPU175: - 3 frac[174,215] - 3 frac[174,111] - 3 frac[174,243]
   - 3 frac[174,75] - 3 frac[174,240] - 3 frac[174,210] - 3 frac[174,58]
   - 3 frac[174,183] - 3 frac[174,193] - 3 frac[174,145] - 3 frac[174,173]
   - 3 frac[174,104] - 3 frac[174,56] - 3 frac[174,237] - 3 frac[174,64]
   - 3 frac[174,16] - 3 frac[174,252] - 3 frac[174,62] + rows_tot_CPU175
   = 0
 mem_tot_CPU176: - mem[175,117] - mem[175,254] - mem[175,69] - mem[175,193]
   - mem[175,122] - mem[175,163] - mem[175,99] - mem[175,181]
   - mem[175,153] + mem_tot_CPU176 = 0
 rows_tot_CPU176: - 3 frac[175,117] - 3 frac[175,254] - 3 frac[175,69]
   - 3 frac[175,193] - 3 frac[175,122] - 3 frac[175,163] - 3 frac[175,99]
   - 3 frac[175,181] - 3 frac[175,153] + rows_tot_CPU176 = 0
 mem_tot_CPU177: - mem[176,35] - mem[176,153] - mem[176,254] - mem[176,69]
   - mem[176,4] - mem[176,191] - mem[176,204] - mem[176,16] - mem[176,149]
   - mem[176,197] + mem_tot_CPU177 = 0
 rows_tot_CPU177: - 3 frac[176,35] - 3 frac[176,153] - 3 frac[176,254]
   - 3 frac[176,69] - 3 frac[176,4] - 3 frac[176,191] - 3 frac[176,204]
   - 3 frac[176,16] - 3 frac[176,149] - 3 frac[176,197] + rows_tot_CPU177
   = 0
 mem_tot_CPU178: - mem[177,185] - mem[177,54] - mem[177,235] - mem[177,8]
   - mem[177,206] - mem[177,161] - mem[177,131] - mem[177,204]
   - mem[177,86] - mem[177,144] - mem[177,91] - mem[177,85]
   + mem_tot_CPU178 = 0
 rows_tot_CPU178: - 3 frac[177,185] - 3 frac[177,54] - 3 frac[177,235]
   - 3 frac[177,8] - 3 frac[177,206] - 3 frac[177,161] - 3 frac[177,131]
   - 3 frac[177,204] - 3 frac[177,86] - 3 frac[177,144] - 3 frac[177,91]
   - 3 frac[177,85] + rows_tot_CPU178 = 0
 mem_tot_CPU179: - mem[178,194] - mem[178,152] - mem[178,147]
   - mem[178,215] - mem[178,5] - mem[178,254] - mem[178,2] - mem[178,39]
   - mem[178,191] - mem[178,177] - mem[178,168] + mem_tot_CPU179 = 0
 rows_tot_CPU179: - 3 frac[178,194] - 3 frac[178,152] - 3 frac[178,147]
   - 3 frac[178,215] - 3 frac[178,5] - 3 frac[178,254] - 3 frac[178,2]
   - 3 frac[178,39] - 3 frac[178,191] - 3 frac[178,177] - 3 frac[178,168]
   + rows_tot_CPU179 = 0
 mem_tot_CPU180: - mem[179,54] - mem[179,166] - mem[179,62] - mem[179,7]
   - mem[179,183] - mem[179,20] - mem[179,90] - mem[179,65] - mem[179,93]
   - mem[179,255] - mem[179,170] - mem[179,75] + mem_tot_CPU180 = 0
 rows_tot_CPU180: - 3 frac[179,54] - 3 frac[179,166] - 3 frac[179,62]
   - 3 frac[179,7] - 3 frac[179,183] - 3 frac[179,20] - 3 frac[179,90]
   - 3 frac[179,65] - 3 frac[179,93] - 3 frac[179,255] - 3 frac[179,170]
   - 3 frac[179,75] + rows_tot_CPU180 = 0
 mem_tot_CPU181: - mem[180,38] - mem[180,244] - mem[180,252] - mem[180,180]
   - mem[180,167] - mem[180,65] - mem[180,94] - mem[180,50] - mem[180,57]
   + mem_tot_CPU181 = 0
 rows_tot_CPU181: - 3 frac[180,38] - 3 frac[180,244] - 3 frac[180,252]
   - 3 frac[180,180] - 3 frac[180,167] - 3 frac[180,65] - 3 frac[180,94]
   - 3 frac[180,50] - 3 frac[180,57] + rows_tot_CPU181 = 0
 mem_tot_CPU182: - mem[181,254] - mem[181,122] - mem[181,76] - mem[181,144]
   - mem[181,100] - mem[181,242] - mem[181,81] - mem[181,184] - mem[181,41]
   - mem[181,90] - mem[181,47] + mem_tot_CPU182 = 0
 rows_tot_CPU182: - 3 frac[181,254] - 3 frac[181,122] - 3 frac[181,76]
   - 3 frac[181,144] - 3 frac[181,100] - 3 frac[181,242] - 3 frac[181,81]
   - 3 frac[181,184] - 3 frac[181,41] - 3 frac[181,90] - 3 frac[181,47]
   + rows_tot_CPU182 = 0
 mem_tot_CPU183: - mem[182,114] - mem[182,212] - mem[182,6] - mem[182,97]
   - mem[182,226] - mem[182,148] - mem[182,30] - mem[182,173] - mem[182,51]
   - mem[182,93] - mem[182,183] - mem[182,5] - mem[182,62] - mem[182,101]
   - mem[182,187] - mem[182,122] - mem[182,184] - mem[182,34]
   + mem_tot_CPU183 = 0
 rows_tot_CPU183: - 3 frac[182,114] - 3 frac[182,212] - 3 frac[182,6]
   - 3 frac[182,97] - 3 frac[182,226] - 3 frac[182,148] - 3 frac[182,30]
   - 3 frac[182,173] - 3 frac[182,51] - 3 frac[182,93] - 3 frac[182,183]
   - 3 frac[182,5] - 3 frac[182,62] - 3 frac[182,101] - 3 frac[182,187]
   - 3 frac[182,122] - 3 frac[182,184] - 3 frac[182,34] + rows_tot_CPU183
   = 0
 mem_tot_CPU184: - mem[183,214] - mem[183,92] - mem[183,254] - mem[183,177]
   - mem[183,114] - mem[183,220] - mem[183,81] - mem[183,188] - mem[183,66]
   - mem[183,7] + mem_tot_CPU184 = 0
 rows_tot_CPU184: - 3 frac[183,214] - 3 frac[183,92] - 3 frac[183,254]
   - 3 frac[183,177] - 3 frac[183,114] - 3 frac[183,220] - 3 frac[183,81]
   - 3 frac[183,188] - 3 frac[183,66] - 3 frac[183,7] + rows_tot_CPU184
   = 0
 mem_tot_CPU185: - mem[184,203] - mem[184,120] - mem[184,145]
   - mem[184,239] - mem[184,161] - mem[184,102] - mem[184,141]
   - mem[184,111] - mem[184,222] - mem[184,136] + mem_tot_CPU185 = 0
 rows_tot_CPU185: - 3 frac[184,203] - 3 frac[184,120] - 3 frac[184,145]
   - 3 frac[184,239] - 3 frac[184,161] - 3 frac[184,102] - 3 frac[184,141]
   - 3 frac[184,111] - 3 frac[184,222] - 3 frac[184,136] + rows_tot_CPU185
   = 0
 mem_tot_CPU186: - mem[185,140] - mem[185,119] - mem[185,100]
   - mem[185,185] - mem[185,52] - mem[185,231] - mem[185,239] - mem[185,49]
   - mem[185,95] + mem_tot_CPU186 = 0
 rows_tot_CPU186: - 3 frac[185,140] - 3 frac[185,119] - 3 frac[185,100]
   - 3 frac[185,185] - 3 frac[185,52] - 3 frac[185,231] - 3 frac[185,239]
   - 3 frac[185,49] - 3 frac[185,95] + rows_tot_CPU186 = 0
 mem_tot_CPU187: - mem[186,181] - mem[186,1] - mem[186,193] - mem[186,42]
   - mem[186,247] - mem[186,157] - mem[186,206] - mem[186,85]
   - mem[186,185] + mem_tot_CPU187 = 0
 rows_tot_CPU187: - 3 frac[186,181] - 3 frac[186,1] - 3 frac[186,193]
   - 3 frac[186,42] - 3 frac[186,247] - 3 frac[186,157] - 3 frac[186,206]
   - 3 frac[186,85] - 3 frac[186,185] + rows_tot_CPU187 = 0
 mem_tot_CPU188: - mem[187,40] - mem[187,246] - mem[187,32] - mem[187,62]
   - mem[187,83] - mem[187,102] - mem[187,116] - mem[187,242] - mem[187,78]
   - mem[187,65] - mem[187,170] - mem[187,101] + mem_tot_CPU188 = 0
 rows_tot_CPU188: - 3 frac[187,40] - 3 frac[187,246] - 3 frac[187,32]
   - 3 frac[187,62] - 3 frac[187,83] - 3 frac[187,102] - 3 frac[187,116]
   - 3 frac[187,242] - 3 frac[187,78] - 3 frac[187,65] - 3 frac[187,170]
   - 3 frac[187,101] + rows_tot_CPU188 = 0
 mem_tot_CPU189: - mem[188,76] - mem[188,170] - mem[188,250] - mem[188,176]
   - mem[188,204] - mem[188,119] - mem[188,41] - mem[188,94] - mem[188,254]
   - mem[188,175] - mem[188,219] - mem[188,255] + mem_tot_CPU189 = 0
 rows_tot_CPU189: - 3 frac[188,76] - 3 frac[188,170] - 3 frac[188,250]
   - 3 frac[188,176] - 3 frac[188,204] - 3 frac[188,119] - 3 frac[188,41]
   - 3 frac[188,94] - 3 frac[188,254] - 3 frac[188,175] - 3 frac[188,219]
   - 3 frac[188,255] + rows_tot_CPU189 = 0
 mem_tot_CPU190: - mem[189,124] - mem[189,147] - mem[189,190] - mem[189,50]
   - mem[189,27] - mem[189,62] - mem[189,63] - mem[189,56] - mem[189,96]
   - mem[189,84] - mem[189,72] - mem[189,148] + mem_tot_CPU190 = 0
 rows_tot_CPU190: - 3 frac[189,124] - 3 frac[189,147] - 3 frac[189,190]
   - 3 frac[189,50] - 3 frac[189,27] - 3 frac[189,62] - 3 frac[189,63]
   - 3 frac[189,56] - 3 frac[189,96] - 3 frac[189,84] - 3 frac[189,72]
   - 3 frac[189,148] + rows_tot_CPU190 = 0
 mem_tot_CPU191: - mem[190,149] - mem[190,25] - mem[190,219] - mem[190,170]
   - mem[190,8] - mem[190,216] - mem[190,82] - mem[190,155] - mem[190,130]
   - mem[190,245] - mem[190,126] - mem[190,188] - mem[190,201]
   - mem[190,102] - mem[190,89] - mem[190,241] - mem[190,227]
   + mem_tot_CPU191 = 0
 rows_tot_CPU191: - 3 frac[190,149] - 3 frac[190,25] - 3 frac[190,219]
   - 3 frac[190,170] - 3 frac[190,8] - 3 frac[190,216] - 3 frac[190,82]
   - 3 frac[190,155] - 3 frac[190,130] - 3 frac[190,245] - 3 frac[190,126]
   - 3 frac[190,188] - 3 frac[190,201] - 3 frac[190,102] - 3 frac[190,89]
   - 3 frac[190,241] - 3 frac[190,227] + rows_tot_CPU191 = 0
 mem_tot_CPU192: - mem[191,232] - mem[191,255] - mem[191,19] - mem[191,125]
   - mem[191,240] - mem[191,147] - mem[191,142] - mem[191,90] - mem[191,67]
   - mem[191,244] - mem[191,149] + mem_tot_CPU192 = 0
 rows_tot_CPU192: - 3 frac[191,232] - 3 frac[191,255] - 3 frac[191,19]
   - 3 frac[191,125] - 3 frac[191,240] - 3 frac[191,147] - 3 frac[191,142]
   - 3 frac[191,90] - 3 frac[191,67] - 3 frac[191,244] - 3 frac[191,149]
   + rows_tot_CPU192 = 0
 mem_tot_tor_P41: - mem[192,241] - mem[192,42] - mem[192,89] - mem[192,200]
   - mem[192,33] - mem[192,207] - mem[192,56] - mem[192,87] - mem[192,198]
   - mem[192,63] - mem[192,174] - mem[192,221] - mem[192,54] - mem[192,13]
   - mem[192,188] - mem[192,4] - mem[192,27] - mem[192,138] - mem[192,18]
   - mem[192,129] - mem[192,105] - mem[192,152] - mem[192,96]
   - mem[192,159] - mem[192,103] - mem[192,150] - mem[192,126]
   - mem[192,117] - mem[192,76] - mem[192,251] - mem[192,67] - mem[192,242]
   - mem[192,43] - mem[192,90] - mem[192,201] - mem[192,34] - mem[192,81]
   - mem[192,192] - mem[192,57] - mem[192,168] - mem[192,48] - mem[192,222]
   - mem[192,55] - mem[192,166] - mem[192,213] - mem[192,14] - mem[192,189]
   - mem[192,5] - mem[192,180] - mem[192,28] - mem[192,19] - mem[192,130]
   - mem[192,153] - mem[192,97] - mem[192,144] - mem[192,120]
   - mem[192,151] - mem[192,127] - mem[192,238] - mem[192,229]
   - mem[192,77] - mem[192,252] - mem[192,68] - mem[192,243] - mem[192,44]
   - mem[192,91] - mem[192,202] - mem[192,35] - mem[192,82] - mem[192,193]
   - mem[192,58] - mem[192,169] - mem[192,216] - mem[192,49] - mem[192,160]
   - mem[192,223] - mem[192,8] - mem[192,167] - mem[192,15] - mem[192,190]
   - mem[192,6] - mem[192,181] - mem[192,140] - mem[192,20] - mem[192,131]
   - mem[192,107] - mem[192,154] - mem[192,98] - mem[192,145]
   - mem[192,121] - mem[192,232] - mem[192,112] - mem[192,239]
   - mem[192,119] - mem[192,230] - mem[192,78] - mem[192,253] - mem[192,69]
   - mem[192,244] - mem[192,45] - mem[192,92] - mem[192,36] - mem[192,83]
   - mem[192,194] - mem[192,59] - mem[192,170] - mem[192,217] - mem[192,50]
   - mem[192,161] - mem[192,208] - mem[192,9] - mem[192,184] - mem[192,215]
   - mem[192,0] - mem[192,191] - mem[192,7] - mem[192,182] - mem[192,30]
   - mem[192,141] - mem[192,21] - mem[192,132] - mem[192,108] - mem[192,99]
   - mem[192,146] - mem[192,122] - mem[192,233] - mem[192,113]
   - mem[192,224] - mem[192,72] - mem[192,231] - mem[192,79] - mem[192,254]
   - mem[192,70] - mem[192,245] - mem[192,46] - mem[192,93] - mem[192,204]
   - mem[192,37] - mem[192,84] - mem[192,195] - mem[192,60] - mem[192,218]
   - mem[192,51] - mem[192,162] - mem[192,10] - mem[192,185] - mem[192,1]
   - mem[192,176] - mem[192,24] - mem[192,183] - mem[192,31] - mem[192,142]
   - mem[192,133] - mem[192,109] - mem[192,156] - mem[192,100]
   - mem[192,147] - mem[192,123] - mem[192,114] - mem[192,225]
   - mem[192,73] - mem[192,64] - mem[192,255] - mem[192,246] - mem[192,47]
   - mem[192,94] - mem[192,205] - mem[192,38] - mem[192,85] - mem[192,196]
   - mem[192,61] - mem[192,172] - mem[192,219] - mem[192,52] - mem[192,163]
   - mem[192,210] - mem[192,11] - mem[192,2] - mem[192,177] - mem[192,25]
   - mem[192,136] - mem[192,16] - mem[192,143] - mem[192,23] - mem[192,134]
   - mem[192,157] - mem[192,101] - mem[192,148] - mem[192,124]
   - mem[192,235] - mem[192,115] - mem[192,226] - mem[192,74]
   - mem[192,249] - mem[192,65] - mem[192,240] - mem[192,41] - mem[192,88]
   - mem[192,247] - mem[192,32] - mem[192,206] - mem[192,39] - mem[192,86]
   - mem[192,62] - mem[192,220] - mem[192,53] - mem[192,164] - mem[192,211]
   - mem[192,12] - mem[192,187] - mem[192,3] - mem[192,17] - mem[192,128]
   - mem[192,104] - mem[192,135] - mem[192,111] - mem[192,158]
   - mem[192,102] - mem[192,149] - mem[192,236] - mem[192,116]
   - mem[192,227] - mem[192,75] - mem[192,250] - mem[192,66]
   + mem_tot_tor_P41 = 0
 rows_tot_tor_P41: - 3 frac[192,241] - 3 frac[192,42] - 3 frac[192,89]
   - 3 frac[192,200] - 3 frac[192,33] - 3 frac[192,207] - 3 frac[192,56]
   - 3 frac[192,87] - 3 frac[192,198] - 3 frac[192,63] - 3 frac[192,174]
   - 3 frac[192,221] - 3 frac[192,54] - 3 frac[192,13] - 3 frac[192,188]
   - 3 frac[192,4] - 3 frac[192,27] - 3 frac[192,138] - 3 frac[192,18]
   - 3 frac[192,129] - 3 frac[192,105] - 3 frac[192,152] - 3 frac[192,96]
   - 3 frac[192,159] - 3 frac[192,103] - 3 frac[192,150] - 3 frac[192,126]
   - 3 frac[192,117] - 3 frac[192,76] - 3 frac[192,251] - 3 frac[192,67]
   - 3 frac[192,242] - 3 frac[192,43] - 3 frac[192,90] - 3 frac[192,201]
   - 3 frac[192,34] - 3 frac[192,81] - 3 frac[192,192] - 3 frac[192,57]
   - 3 frac[192,168] - 3 frac[192,48] - 3 frac[192,222] - 3 frac[192,55]
   - 3 frac[192,166] - 3 frac[192,213] - 3 frac[192,14] - 3 frac[192,189]
   - 3 frac[192,5] - 3 frac[192,180] - 3 frac[192,28] - 3 frac[192,19]
   - 3 frac[192,130] - 3 frac[192,153] - 3 frac[192,97] - 3 frac[192,144]
   - 3 frac[192,120] - 3 frac[192,151] - 3 frac[192,127] - 3 frac[192,238]
   - 3 frac[192,229] - 3 frac[192,77] - 3 frac[192,252] - 3 frac[192,68]
   - 3 frac[192,243] - 3 frac[192,44] - 3 frac[192,91] - 3 frac[192,202]
   - 3 frac[192,35] - 3 frac[192,82] - 3 frac[192,193] - 3 frac[192,58]
   - 3 frac[192,169] - 3 frac[192,216] - 3 frac[192,49] - 3 frac[192,160]
   - 3 frac[192,223] - 3 frac[192,8] - 3 frac[192,167] - 3 frac[192,15]
   - 3 frac[192,190] - 3 frac[192,6] - 3 frac[192,181] - 3 frac[192,140]
   - 3 frac[192,20] - 3 frac[192,131] - 3 frac[192,107] - 3 frac[192,154]
   - 3 frac[192,98] - 3 frac[192,145] - 3 frac[192,121] - 3 frac[192,232]
   - 3 frac[192,112] - 3 frac[192,239] - 3 frac[192,119] - 3 frac[192,230]
   - 3 frac[192,78] - 3 frac[192,253] - 3 frac[192,69] - 3 frac[192,244]
   - 3 frac[192,45] - 3 frac[192,92] - 3 frac[192,36] - 3 frac[192,83]
   - 3 frac[192,194] - 3 frac[192,59] - 3 frac[192,170] - 3 frac[192,217]
   - 3 frac[192,50] - 3 frac[192,161] - 3 frac[192,208] - 3 frac[192,9]
   - 3 frac[192,184] - 3 frac[192,215] - 3 frac[192,0] - 3 frac[192,191]
   - 3 frac[192,7] - 3 frac[192,182] - 3 frac[192,30] - 3 frac[192,141]
   - 3 frac[192,21] - 3 frac[192,132] - 3 frac[192,108] - 3 frac[192,99]
   - 3 frac[192,146] - 3 frac[192,122] - 3 frac[192,233] - 3 frac[192,113]
   - 3 frac[192,224] - 3 frac[192,72] - 3 frac[192,231] - 3 frac[192,79]
   - 3 frac[192,254] - 3 frac[192,70] - 3 frac[192,245] - 3 frac[192,46]
   - 3 frac[192,93] - 3 frac[192,204] - 3 frac[192,37] - 3 frac[192,84]
   - 3 frac[192,195] - 3 frac[192,60] - 3 frac[192,218] - 3 frac[192,51]
   - 3 frac[192,162] - 3 frac[192,10] - 3 frac[192,185] - 3 frac[192,1]
   - 3 frac[192,176] - 3 frac[192,24] - 3 frac[192,183] - 3 frac[192,31]
   - 3 frac[192,142] - 3 frac[192,133] - 3 frac[192,109] - 3 frac[192,156]
   - 3 frac[192,100] - 3 frac[192,147] - 3 frac[192,123] - 3 frac[192,114]
   - 3 frac[192,225] - 3 frac[192,73] - 3 frac[192,64] - 3 frac[192,255]
   - 3 frac[192,246] - 3 frac[192,47] - 3 frac[192,94] - 3 frac[192,205]
   - 3 frac[192,38] - 3 frac[192,85] - 3 frac[192,196] - 3 frac[192,61]
   - 3 frac[192,172] - 3 frac[192,219] - 3 frac[192,52] - 3 frac[192,163]
   - 3 frac[192,210] - 3 frac[192,11] - 3 frac[192,2] - 3 frac[192,177]
   - 3 frac[192,25] - 3 frac[192,136] - 3 frac[192,16] - 3 frac[192,143]
   - 3 frac[192,23] - 3 frac[192,134] - 3 frac[192,157] - 3 frac[192,101]
   - 3 frac[192,148] - 3 frac[192,124] - 3 frac[192,235] - 3 frac[192,115]
   - 3 frac[192,226] - 3 frac[192,74] - 3 frac[192,249] - 3 frac[192,65]
   - 3 frac[192,240] - 3 frac[192,41] - 3 frac[192,88] - 3 frac[192,247]
   - 3 frac[192,32] - 3 frac[192,206] - 3 frac[192,39] - 3 frac[192,86]
   - 3 frac[192,62] - 3 frac[192,220] - 3 frac[192,53] - 3 frac[192,164]
   - 3 frac[192,211] - 3 frac[192,12] - 3 frac[192,187] - 3 frac[192,3]
   - 3 frac[192,17] - 3 frac[192,128] - 3 frac[192,104] - 3 frac[192,135]
   - 3 frac[192,111] - 3 frac[192,158] - 3 frac[192,102] - 3 frac[192,149]
   - 3 frac[192,236] - 3 frac[192,116] - 3 frac[192,227] - 3 frac[192,75]
   - 3 frac[192,250] - 3 frac[192,66] + rows_tot_tor_P41 = 0
 mem_tot_tor_P42: - mem[193,229] - mem[193,150] - mem[193,254]
   - mem[193,70] - mem[193,247] - mem[193,95] - mem[193,80] - mem[193,41]
   - mem[193,218] - mem[193,34] - mem[193,211] - mem[193,59] - mem[193,212]
   - mem[193,60] - mem[193,173] - mem[193,53] - mem[193,166] - mem[193,14]
   - mem[193,7] - mem[193,111] - mem[193,24] - mem[193,137] - mem[193,96]
   - mem[193,17] - mem[193,130] - mem[193,121] - mem[193,155]
   - mem[193,114] - mem[193,227] - mem[193,156] - mem[193,75]
   - mem[193,228] - mem[193,253] - mem[193,69] - mem[193,246] - mem[193,94]
   - mem[193,87] - mem[193,192] - mem[193,40] - mem[193,217] - mem[193,33]
   - mem[193,58] - mem[193,171] - mem[193,51] - mem[193,172] - mem[193,52]
   - mem[193,165] - mem[193,190] - mem[193,6] - mem[193,183] - mem[193,110]
   - mem[193,31] - mem[193,136] - mem[193,103] - mem[193,16] - mem[193,129]
   - mem[193,120] - mem[193,233] - mem[193,154] - mem[193,113]
   - mem[193,226] - mem[193,147] - mem[193,74] - mem[193,251]
   - mem[193,148] - mem[193,67] - mem[193,252] - mem[193,68] - mem[193,245]
   - mem[193,93] - mem[193,206] - mem[193,86] - mem[193,199] - mem[193,47]
   - mem[193,216] - mem[193,209] - mem[193,57] - mem[193,170] - mem[193,50]
   - mem[193,163] - mem[193,11] - mem[193,164] - mem[193,12] - mem[193,189]
   - mem[193,5] - mem[193,182] - mem[193,30] - mem[193,143] - mem[193,102]
   - mem[193,23] - mem[193,128] - mem[193,127] - mem[193,232]
   - mem[193,153] - mem[193,112] - mem[193,225] - mem[193,146]
   - mem[193,73] - mem[193,250] - mem[193,66] - mem[193,243] - mem[193,91]
   - mem[193,244] - mem[193,92] - mem[193,205] - mem[193,85] - mem[193,198]
   - mem[193,46] - mem[193,223] - mem[193,39] - mem[193,208] - mem[193,56]
   - mem[193,169] - mem[193,162] - mem[193,10] - mem[193,187]
   - mem[193,188] - mem[193,107] - mem[193,4] - mem[193,108] - mem[193,29]
   - mem[193,142] - mem[193,101] - mem[193,22] - mem[193,135]
   - mem[193,126] - mem[193,239] - mem[193,152] - mem[193,119]
   - mem[193,145] - mem[193,72] - mem[193,249] - mem[193,65] - mem[193,242]
   - mem[193,90] - mem[193,203] - mem[193,83] - mem[193,204] - mem[193,84]
   - mem[193,197] - mem[193,45] - mem[193,222] - mem[193,38] - mem[193,215]
   - mem[193,63] - mem[193,168] - mem[193,48] - mem[193,161] - mem[193,9]
   - mem[193,186] - mem[193,2] - mem[193,179] - mem[193,106] - mem[193,27]
   - mem[193,180] - mem[193,99] - mem[193,28] - mem[193,100] - mem[193,134]
   - mem[193,238] - mem[193,159] - mem[193,231] - mem[193,144]
   - mem[193,79] - mem[193,248] - mem[193,64] - mem[193,241] - mem[193,89]
   - mem[193,202] - mem[193,82] - mem[193,195] - mem[193,43] - mem[193,196]
   - mem[193,44] - mem[193,221] - mem[193,37] - mem[193,214] - mem[193,175]
   - mem[193,55] - mem[193,160] - mem[193,8] - mem[193,185] - mem[193,26]
   - mem[193,139] - mem[193,98] - mem[193,19] - mem[193,140] - mem[193,123]
   - mem[193,20] - mem[193,133] - mem[193,124] - mem[193,158]
   - mem[193,117] - mem[193,230] - mem[193,151] - mem[193,78]
   - mem[193,255] - mem[193,71] - mem[193,240] - mem[193,88] - mem[193,201]
   - mem[193,81] - mem[193,194] - mem[193,42] - mem[193,219] - mem[193,35]
   - mem[193,220] - mem[193,36] - mem[193,213] - mem[193,61] - mem[193,174]
   - mem[193,54] - mem[193,167] - mem[193,15] - mem[193,184] - mem[193,0]
   - mem[193,177] - mem[193,104] - mem[193,138] - mem[193,97] - mem[193,18]
   - mem[193,131] - mem[193,122] - mem[193,235] - mem[193,132]
   - mem[193,116] + mem_tot_tor_P42 = 0
 rows_tot_tor_P42: - 3 frac[193,229] - 3 frac[193,150] - 3 frac[193,254]
   - 3 frac[193,70] - 3 frac[193,247] - 3 frac[193,95] - 3 frac[193,80]
   - 3 frac[193,41] - 3 frac[193,218] - 3 frac[193,34] - 3 frac[193,211]
   - 3 frac[193,59] - 3 frac[193,212] - 3 frac[193,60] - 3 frac[193,173]
   - 3 frac[193,53] - 3 frac[193,166] - 3 frac[193,14] - 3 frac[193,7]
   - 3 frac[193,111] - 3 frac[193,24] - 3 frac[193,137] - 3 frac[193,96]
   - 3 frac[193,17] - 3 frac[193,130] - 3 frac[193,121] - 3 frac[193,155]
   - 3 frac[193,114] - 3 frac[193,227] - 3 frac[193,156] - 3 frac[193,75]
   - 3 frac[193,228] - 3 frac[193,253] - 3 frac[193,69] - 3 frac[193,246]
   - 3 frac[193,94] - 3 frac[193,87] - 3 frac[193,192] - 3 frac[193,40]
   - 3 frac[193,217] - 3 frac[193,33] - 3 frac[193,58] - 3 frac[193,171]
   - 3 frac[193,51] - 3 frac[193,172] - 3 frac[193,52] - 3 frac[193,165]
   - 3 frac[193,190] - 3 frac[193,6] - 3 frac[193,183] - 3 frac[193,110]
   - 3 frac[193,31] - 3 frac[193,136] - 3 frac[193,103] - 3 frac[193,16]
   - 3 frac[193,129] - 3 frac[193,120] - 3 frac[193,233] - 3 frac[193,154]
   - 3 frac[193,113] - 3 frac[193,226] - 3 frac[193,147] - 3 frac[193,74]
   - 3 frac[193,251] - 3 frac[193,148] - 3 frac[193,67] - 3 frac[193,252]
   - 3 frac[193,68] - 3 frac[193,245] - 3 frac[193,93] - 3 frac[193,206]
   - 3 frac[193,86] - 3 frac[193,199] - 3 frac[193,47] - 3 frac[193,216]
   - 3 frac[193,209] - 3 frac[193,57] - 3 frac[193,170] - 3 frac[193,50]
   - 3 frac[193,163] - 3 frac[193,11] - 3 frac[193,164] - 3 frac[193,12]
   - 3 frac[193,189] - 3 frac[193,5] - 3 frac[193,182] - 3 frac[193,30]
   - 3 frac[193,143] - 3 frac[193,102] - 3 frac[193,23] - 3 frac[193,128]
   - 3 frac[193,127] - 3 frac[193,232] - 3 frac[193,153] - 3 frac[193,112]
   - 3 frac[193,225] - 3 frac[193,146] - 3 frac[193,73] - 3 frac[193,250]
   - 3 frac[193,66] - 3 frac[193,243] - 3 frac[193,91] - 3 frac[193,244]
   - 3 frac[193,92] - 3 frac[193,205] - 3 frac[193,85] - 3 frac[193,198]
   - 3 frac[193,46] - 3 frac[193,223] - 3 frac[193,39] - 3 frac[193,208]
   - 3 frac[193,56] - 3 frac[193,169] - 3 frac[193,162] - 3 frac[193,10]
   - 3 frac[193,187] - 3 frac[193,188] - 3 frac[193,107] - 3 frac[193,4]
   - 3 frac[193,108] - 3 frac[193,29] - 3 frac[193,142] - 3 frac[193,101]
   - 3 frac[193,22] - 3 frac[193,135] - 3 frac[193,126] - 3 frac[193,239]
   - 3 frac[193,152] - 3 frac[193,119] - 3 frac[193,145] - 3 frac[193,72]
   - 3 frac[193,249] - 3 frac[193,65] - 3 frac[193,242] - 3 frac[193,90]
   - 3 frac[193,203] - 3 frac[193,83] - 3 frac[193,204] - 3 frac[193,84]
   - 3 frac[193,197] - 3 frac[193,45] - 3 frac[193,222] - 3 frac[193,38]
   - 3 frac[193,215] - 3 frac[193,63] - 3 frac[193,168] - 3 frac[193,48]
   - 3 frac[193,161] - 3 frac[193,9] - 3 frac[193,186] - 3 frac[193,2]
   - 3 frac[193,179] - 3 frac[193,106] - 3 frac[193,27] - 3 frac[193,180]
   - 3 frac[193,99] - 3 frac[193,28] - 3 frac[193,100] - 3 frac[193,134]
   - 3 frac[193,238] - 3 frac[193,159] - 3 frac[193,231] - 3 frac[193,144]
   - 3 frac[193,79] - 3 frac[193,248] - 3 frac[193,64] - 3 frac[193,241]
   - 3 frac[193,89] - 3 frac[193,202] - 3 frac[193,82] - 3 frac[193,195]
   - 3 frac[193,43] - 3 frac[193,196] - 3 frac[193,44] - 3 frac[193,221]
   - 3 frac[193,37] - 3 frac[193,214] - 3 frac[193,175] - 3 frac[193,55]
   - 3 frac[193,160] - 3 frac[193,8] - 3 frac[193,185] - 3 frac[193,26]
   - 3 frac[193,139] - 3 frac[193,98] - 3 frac[193,19] - 3 frac[193,140]
   - 3 frac[193,123] - 3 frac[193,20] - 3 frac[193,133] - 3 frac[193,124]
   - 3 frac[193,158] - 3 frac[193,117] - 3 frac[193,230] - 3 frac[193,151]
   - 3 frac[193,78] - 3 frac[193,255] - 3 frac[193,71] - 3 frac[193,240]
   - 3 frac[193,88] - 3 frac[193,201] - 3 frac[193,81] - 3 frac[193,194]
   - 3 frac[193,42] - 3 frac[193,219] - 3 frac[193,35] - 3 frac[193,220]
   - 3 frac[193,36] - 3 frac[193,213] - 3 frac[193,61] - 3 frac[193,174]
   - 3 frac[193,54] - 3 frac[193,167] - 3 frac[193,15] - 3 frac[193,184]
   - 3 frac[193,0] - 3 frac[193,177] - 3 frac[193,104] - 3 frac[193,138]
   - 3 frac[193,97] - 3 frac[193,18] - 3 frac[193,131] - 3 frac[193,122]
   - 3 frac[193,235] - 3 frac[193,132] - 3 frac[193,116] + rows_tot_tor_P42
   = 0
 mem_tot_tor_P43: - mem[194,237] - mem[194,228] - mem[194,23]
   - mem[194,255] - mem[194,110] - mem[194,246] - mem[194,105]
   - mem[194,96] - mem[194,123] - mem[194,195] - mem[194,114]
   - mem[194,171] - mem[194,77] - mem[194,162] - mem[194,213] - mem[194,68]
   - mem[194,189] - mem[194,44] - mem[194,95] - mem[194,39] - mem[194,86]
   - mem[194,143] - mem[194,62] - mem[194,81] - mem[194,134] - mem[194,57]
   - mem[194,129] - mem[194,48] - mem[194,152] - mem[194,11] - mem[194,147]
   - mem[194,2] - mem[194,229] - mem[194,20] - mem[194,252] - mem[194,111]
   - mem[194,102] - mem[194,206] - mem[194,97] - mem[194,201]
   - mem[194,120] - mem[194,192] - mem[194,115] - mem[194,168]
   - mem[194,219] - mem[194,74] - mem[194,163] - mem[194,210] - mem[194,69]
   - mem[194,186] - mem[194,45] - mem[194,92] - mem[194,181] - mem[194,36]
   - mem[194,87] - mem[194,140] - mem[194,135] - mem[194,158] - mem[194,49]
   - mem[194,153] - mem[194,8] - mem[194,144] - mem[194,3] - mem[194,235]
   - mem[194,26] - mem[194,226] - mem[194,21] - mem[194,108] - mem[194,244]
   - mem[194,207] - mem[194,126] - mem[194,198] - mem[194,121]
   - mem[194,174] - mem[194,112] - mem[194,169] - mem[194,216]
   - mem[194,75] - mem[194,160] - mem[194,211] - mem[194,66] - mem[194,187]
   - mem[194,42] - mem[194,93] - mem[194,178] - mem[194,84] - mem[194,141]
   - mem[194,60] - mem[194,132] - mem[194,55] - mem[194,159] - mem[194,14]
   - mem[194,150] - mem[194,145] - mem[194,0] - mem[194,232] - mem[194,27]
   - mem[194,227] - mem[194,18] - mem[194,250] - mem[194,109]
   - mem[194,245] - mem[194,100] - mem[194,204] - mem[194,127]
   - mem[194,199] - mem[194,175] - mem[194,222] - mem[194,113]
   - mem[194,166] - mem[194,217] - mem[194,72] - mem[194,161]
   - mem[194,208] - mem[194,67] - mem[194,184] - mem[194,43] - mem[194,90]
   - mem[194,179] - mem[194,34] - mem[194,85] - mem[194,61] - mem[194,133]
   - mem[194,156] - mem[194,15] - mem[194,151] - mem[194,6] - mem[194,238]
   - mem[194,1] - mem[194,233] - mem[194,24] - mem[194,224] - mem[194,19]
   - mem[194,251] - mem[194,106] - mem[194,242] - mem[194,101]
   - mem[194,205] - mem[194,124] - mem[194,196] - mem[194,119]
   - mem[194,172] - mem[194,223] - mem[194,78] - mem[194,167]
   - mem[194,214] - mem[194,73] - mem[194,190] - mem[194,91] - mem[194,35]
   - mem[194,82] - mem[194,139] - mem[194,58] - mem[194,157] - mem[194,12]
   - mem[194,148] - mem[194,7] - mem[194,239] - mem[194,30] - mem[194,230]
   - mem[194,25] - mem[194,225] - mem[194,16] - mem[194,248] - mem[194,243]
   - mem[194,98] - mem[194,202] - mem[194,125] - mem[194,116]
   - mem[194,173] - mem[194,220] - mem[194,79] - mem[194,164]
   - mem[194,215] - mem[194,70] - mem[194,191] - mem[194,46] - mem[194,65]
   - mem[194,182] - mem[194,41] - mem[194,88] - mem[194,177] - mem[194,32]
   - mem[194,83] - mem[194,136] - mem[194,59] - mem[194,131] - mem[194,50]
   - mem[194,154] - mem[194,13] - mem[194,149] - mem[194,4] - mem[194,236]
   - mem[194,31] - mem[194,231] - mem[194,22] - mem[194,254] - mem[194,17]
   - mem[194,249] - mem[194,104] - mem[194,240] - mem[194,99]
   - mem[194,203] - mem[194,122] - mem[194,194] - mem[194,117]
   - mem[194,170] - mem[194,221] - mem[194,165] - mem[194,212]
   - mem[194,71] - mem[194,188] - mem[194,47] - mem[194,94] - mem[194,183]
   - mem[194,38] - mem[194,89] - mem[194,142] - mem[194,33] - mem[194,80]
   - mem[194,137] - mem[194,56] - mem[194,128] - mem[194,155] - mem[194,10]
   - mem[194,146] - mem[194,5] + mem_tot_tor_P43 = 0
 rows_tot_tor_P43: - 3 frac[194,237] - 3 frac[194,228] - 3 frac[194,23]
   - 3 frac[194,255] - 3 frac[194,110] - 3 frac[194,246] - 3 frac[194,105]
   - 3 frac[194,96] - 3 frac[194,123] - 3 frac[194,195] - 3 frac[194,114]
   - 3 frac[194,171] - 3 frac[194,77] - 3 frac[194,162] - 3 frac[194,213]
   - 3 frac[194,68] - 3 frac[194,189] - 3 frac[194,44] - 3 frac[194,95]
   - 3 frac[194,39] - 3 frac[194,86] - 3 frac[194,143] - 3 frac[194,62]
   - 3 frac[194,81] - 3 frac[194,134] - 3 frac[194,57] - 3 frac[194,129]
   - 3 frac[194,48] - 3 frac[194,152] - 3 frac[194,11] - 3 frac[194,147]
   - 3 frac[194,2] - 3 frac[194,229] - 3 frac[194,20] - 3 frac[194,252]
   - 3 frac[194,111] - 3 frac[194,102] - 3 frac[194,206] - 3 frac[194,97]
   - 3 frac[194,201] - 3 frac[194,120] - 3 frac[194,192] - 3 frac[194,115]
   - 3 frac[194,168] - 3 frac[194,219] - 3 frac[194,74] - 3 frac[194,163]
   - 3 frac[194,210] - 3 frac[194,69] - 3 frac[194,186] - 3 frac[194,45]
   - 3 frac[194,92] - 3 frac[194,181] - 3 frac[194,36] - 3 frac[194,87]
   - 3 frac[194,140] - 3 frac[194,135] - 3 frac[194,158] - 3 frac[194,49]
   - 3 frac[194,153] - 3 frac[194,8] - 3 frac[194,144] - 3 frac[194,3]
   - 3 frac[194,235] - 3 frac[194,26] - 3 frac[194,226] - 3 frac[194,21]
   - 3 frac[194,108] - 3 frac[194,244] - 3 frac[194,207] - 3 frac[194,126]
   - 3 frac[194,198] - 3 frac[194,121] - 3 frac[194,174] - 3 frac[194,112]
   - 3 frac[194,169] - 3 frac[194,216] - 3 frac[194,75] - 3 frac[194,160]
   - 3 frac[194,211] - 3 frac[194,66] - 3 frac[194,187] - 3 frac[194,42]
   - 3 frac[194,93] - 3 frac[194,178] - 3 frac[194,84] - 3 frac[194,141]
   - 3 frac[194,60] - 3 frac[194,132] - 3 frac[194,55] - 3 frac[194,159]
   - 3 frac[194,14] - 3 frac[194,150] - 3 frac[194,145] - 3 frac[194,0]
   - 3 frac[194,232] - 3 frac[194,27] - 3 frac[194,227] - 3 frac[194,18]
   - 3 frac[194,250] - 3 frac[194,109] - 3 frac[194,245] - 3 frac[194,100]
   - 3 frac[194,204] - 3 frac[194,127] - 3 frac[194,199] - 3 frac[194,175]
   - 3 frac[194,222] - 3 frac[194,113] - 3 frac[194,166] - 3 frac[194,217]
   - 3 frac[194,72] - 3 frac[194,161] - 3 frac[194,208] - 3 frac[194,67]
   - 3 frac[194,184] - 3 frac[194,43] - 3 frac[194,90] - 3 frac[194,179]
   - 3 frac[194,34] - 3 frac[194,85] - 3 frac[194,61] - 3 frac[194,133]
   - 3 frac[194,156] - 3 frac[194,15] - 3 frac[194,151] - 3 frac[194,6]
   - 3 frac[194,238] - 3 frac[194,1] - 3 frac[194,233] - 3 frac[194,24]
   - 3 frac[194,224] - 3 frac[194,19] - 3 frac[194,251] - 3 frac[194,106]
   - 3 frac[194,242] - 3 frac[194,101] - 3 frac[194,205] - 3 frac[194,124]
   - 3 frac[194,196] - 3 frac[194,119] - 3 frac[194,172] - 3 frac[194,223]
   - 3 frac[194,78] - 3 frac[194,167] - 3 frac[194,214] - 3 frac[194,73]
   - 3 frac[194,190] - 3 frac[194,91] - 3 frac[194,35] - 3 frac[194,82]
   - 3 frac[194,139] - 3 frac[194,58] - 3 frac[194,157] - 3 frac[194,12]
   - 3 frac[194,148] - 3 frac[194,7] - 3 frac[194,239] - 3 frac[194,30]
   - 3 frac[194,230] - 3 frac[194,25] - 3 frac[194,225] - 3 frac[194,16]
   - 3 frac[194,248] - 3 frac[194,243] - 3 frac[194,98] - 3 frac[194,202]
   - 3 frac[194,125] - 3 frac[194,116] - 3 frac[194,173] - 3 frac[194,220]
   - 3 frac[194,79] - 3 frac[194,164] - 3 frac[194,215] - 3 frac[194,70]
   - 3 frac[194,191] - 3 frac[194,46] - 3 frac[194,65] - 3 frac[194,182]
   - 3 frac[194,41] - 3 frac[194,88] - 3 frac[194,177] - 3 frac[194,32]
   - 3 frac[194,83] - 3 frac[194,136] - 3 frac[194,59] - 3 frac[194,131]
   - 3 frac[194,50] - 3 frac[194,154] - 3 frac[194,13] - 3 frac[194,149]
   - 3 frac[194,4] - 3 frac[194,236] - 3 frac[194,31] - 3 frac[194,231]
   - 3 frac[194,22] - 3 frac[194,254] - 3 frac[194,17] - 3 frac[194,249]
   - 3 frac[194,104] - 3 frac[194,240] - 3 frac[194,99] - 3 frac[194,203]
   - 3 frac[194,122] - 3 frac[194,194] - 3 frac[194,117] - 3 frac[194,170]
   - 3 frac[194,221] - 3 frac[194,165] - 3 frac[194,212] - 3 frac[194,71]
   - 3 frac[194,188] - 3 frac[194,47] - 3 frac[194,94] - 3 frac[194,183]
   - 3 frac[194,38] - 3 frac[194,89] - 3 frac[194,142] - 3 frac[194,33]
   - 3 frac[194,80] - 3 frac[194,137] - 3 frac[194,56] - 3 frac[194,128]
   - 3 frac[194,155] - 3 frac[194,10] - 3 frac[194,146] - 3 frac[194,5]
   + rows_tot_tor_P43 = 0
 mem_tot_tor_P44: - mem[195,150] - mem[195,39] - mem[195,159] - mem[195,44]
   - mem[195,247] - mem[195,132] - mem[195,21] - mem[195,252]
   - mem[195,141] - mem[195,26] - mem[195,229] - mem[195,114] - mem[195,3]
   - mem[195,123] - mem[195,8] - mem[195,211] - mem[195,96] - mem[195,216]
   - mem[195,193] - mem[195,110] - mem[195,198] - mem[195,87]
   - mem[195,207] - mem[195,92] - mem[195,180] - mem[195,69] - mem[195,189]
   - mem[195,74] - mem[195,162] - mem[195,51] - mem[195,56] - mem[195,144]
   - mem[195,33] - mem[195,153] - mem[195,38] - mem[195,241] - mem[195,158]
   - mem[195,47] - mem[195,246] - mem[195,135] - mem[195,20] - mem[195,255]
   - mem[195,140] - mem[195,29] - mem[195,117] - mem[195,2] - mem[195,237]
   - mem[195,122] - mem[195,11] - mem[195,210] - mem[195,99] - mem[195,219]
   - mem[195,104] - mem[195,192] - mem[195,81] - mem[195,201] - mem[195,86]
   - mem[195,206] - mem[195,95] - mem[195,183] - mem[195,68] - mem[195,188]
   - mem[195,77] - mem[195,165] - mem[195,50] - mem[195,170] - mem[195,59]
   - mem[195,147] - mem[195,32] - mem[195,152] - mem[195,41] - mem[195,240]
   - mem[195,129] - mem[195,46] - mem[195,249] - mem[195,134]
   - mem[195,254] - mem[195,143] - mem[195,231] - mem[195,116] - mem[195,5]
   - mem[195,125] - mem[195,10] - mem[195,213] - mem[195,98] - mem[195,107]
   - mem[195,195] - mem[195,80] - mem[195,200] - mem[195,89] - mem[195,177]
   - mem[195,94] - mem[195,182] - mem[195,71] - mem[195,191] - mem[195,76]
   - mem[195,164] - mem[195,53] - mem[195,173] - mem[195,58] - mem[195,35]
   - mem[195,155] - mem[195,40] - mem[195,243] - mem[195,17] - mem[195,248]
   - mem[195,137] - mem[195,225] - mem[195,142] - mem[195,31]
   - mem[195,230] - mem[195,119] - mem[195,4] - mem[195,239] - mem[195,124]
   - mem[195,13] - mem[195,212] - mem[195,101] - mem[195,221]
   - mem[195,194] - mem[195,83] - mem[195,203] - mem[195,88] - mem[195,176]
   - mem[195,65] - mem[195,185] - mem[195,70] - mem[195,190] - mem[195,79]
   - mem[195,167] - mem[195,52] - mem[195,172] - mem[195,61] - mem[195,149]
   - mem[195,34] - mem[195,43] - mem[195,242] - mem[195,131] - mem[195,16]
   - mem[195,251] - mem[195,136] - mem[195,25] - mem[195,224]
   - mem[195,113] - mem[195,30] - mem[195,7] - mem[195,12] - mem[195,215]
   - mem[195,100] - mem[195,220] - mem[195,197] - mem[195,82]
   - mem[195,202] - mem[195,91] - mem[195,64] - mem[195,184] - mem[195,73]
   - mem[195,161] - mem[195,78] - mem[195,166] - mem[195,55] - mem[195,175]
   - mem[195,60] - mem[195,148] - mem[195,37] - mem[195,157] - mem[195,42]
   - mem[195,245] - mem[195,130] - mem[195,19] - mem[195,250] - mem[195,24]
   - mem[195,227] - mem[195,112] - mem[195,1] - mem[195,232] - mem[195,121]
   - mem[195,6] - mem[195,209] - mem[195,126] - mem[195,15] - mem[195,214]
   - mem[195,223] - mem[195,108] - mem[195,85] - mem[195,205] - mem[195,90]
   - mem[195,178] - mem[195,67] - mem[195,187] - mem[195,72] - mem[195,160]
   - mem[195,49] - mem[195,169] - mem[195,54] - mem[195,174] - mem[195,63]
   - mem[195,151] - mem[195,36] - mem[195,45] - mem[195,244] - mem[195,133]
   - mem[195,253] - mem[195,138] - mem[195,27] - mem[195,226] - mem[195,0]
   - mem[195,235] - mem[195,120] - mem[195,9] - mem[195,208] - mem[195,97]
   - mem[195,14] - mem[195,217] - mem[195,102] - mem[195,222]
   - mem[195,111] - mem[195,199] - mem[195,84] - mem[195,204] - mem[195,93]
   - mem[195,181] - mem[195,66] - mem[195,186] - mem[195,75] - mem[195,163]
   - mem[195,48] - mem[195,168] - mem[195,57] - mem[195,145] - mem[195,62]
   + mem_tot_tor_P44 = 0
 rows_tot_tor_P44: - 3 frac[195,150] - 3 frac[195,39] - 3 frac[195,159]
   - 3 frac[195,44] - 3 frac[195,247] - 3 frac[195,132] - 3 frac[195,21]
   - 3 frac[195,252] - 3 frac[195,141] - 3 frac[195,26] - 3 frac[195,229]
   - 3 frac[195,114] - 3 frac[195,3] - 3 frac[195,123] - 3 frac[195,8]
   - 3 frac[195,211] - 3 frac[195,96] - 3 frac[195,216] - 3 frac[195,193]
   - 3 frac[195,110] - 3 frac[195,198] - 3 frac[195,87] - 3 frac[195,207]
   - 3 frac[195,92] - 3 frac[195,180] - 3 frac[195,69] - 3 frac[195,189]
   - 3 frac[195,74] - 3 frac[195,162] - 3 frac[195,51] - 3 frac[195,56]
   - 3 frac[195,144] - 3 frac[195,33] - 3 frac[195,153] - 3 frac[195,38]
   - 3 frac[195,241] - 3 frac[195,158] - 3 frac[195,47] - 3 frac[195,246]
   - 3 frac[195,135] - 3 frac[195,20] - 3 frac[195,255] - 3 frac[195,140]
   - 3 frac[195,29] - 3 frac[195,117] - 3 frac[195,2] - 3 frac[195,237]
   - 3 frac[195,122] - 3 frac[195,11] - 3 frac[195,210] - 3 frac[195,99]
   - 3 frac[195,219] - 3 frac[195,104] - 3 frac[195,192] - 3 frac[195,81]
   - 3 frac[195,201] - 3 frac[195,86] - 3 frac[195,206] - 3 frac[195,95]
   - 3 frac[195,183] - 3 frac[195,68] - 3 frac[195,188] - 3 frac[195,77]
   - 3 frac[195,165] - 3 frac[195,50] - 3 frac[195,170] - 3 frac[195,59]
   - 3 frac[195,147] - 3 frac[195,32] - 3 frac[195,152] - 3 frac[195,41]
   - 3 frac[195,240] - 3 frac[195,129] - 3 frac[195,46] - 3 frac[195,249]
   - 3 frac[195,134] - 3 frac[195,254] - 3 frac[195,143] - 3 frac[195,231]
   - 3 frac[195,116] - 3 frac[195,5] - 3 frac[195,125] - 3 frac[195,10]
   - 3 frac[195,213] - 3 frac[195,98] - 3 frac[195,107] - 3 frac[195,195]
   - 3 frac[195,80] - 3 frac[195,200] - 3 frac[195,89] - 3 frac[195,177]
   - 3 frac[195,94] - 3 frac[195,182] - 3 frac[195,71] - 3 frac[195,191]
   - 3 frac[195,76] - 3 frac[195,164] - 3 frac[195,53] - 3 frac[195,173]
   - 3 frac[195,58] - 3 frac[195,35] - 3 frac[195,155] - 3 frac[195,40]
   - 3 frac[195,243] - 3 frac[195,17] - 3 frac[195,248] - 3 frac[195,137]
   - 3 frac[195,225] - 3 frac[195,142] - 3 frac[195,31] - 3 frac[195,230]
   - 3 frac[195,119] - 3 frac[195,4] - 3 frac[195,239] - 3 frac[195,124]
   - 3 frac[195,13] - 3 frac[195,212] - 3 frac[195,101] - 3 frac[195,221]
   - 3 frac[195,194] - 3 frac[195,83] - 3 frac[195,203] - 3 frac[195,88]
   - 3 frac[195,176] - 3 frac[195,65] - 3 frac[195,185] - 3 frac[195,70]
   - 3 frac[195,190] - 3 frac[195,79] - 3 frac[195,167] - 3 frac[195,52]
   - 3 frac[195,172] - 3 frac[195,61] - 3 frac[195,149] - 3 frac[195,34]
   - 3 frac[195,43] - 3 frac[195,242] - 3 frac[195,131] - 3 frac[195,16]
   - 3 frac[195,251] - 3 frac[195,136] - 3 frac[195,25] - 3 frac[195,224]
   - 3 frac[195,113] - 3 frac[195,30] - 3 frac[195,7] - 3 frac[195,12]
   - 3 frac[195,215] - 3 frac[195,100] - 3 frac[195,220] - 3 frac[195,197]
   - 3 frac[195,82] - 3 frac[195,202] - 3 frac[195,91] - 3 frac[195,64]
   - 3 frac[195,184] - 3 frac[195,73] - 3 frac[195,161] - 3 frac[195,78]
   - 3 frac[195,166] - 3 frac[195,55] - 3 frac[195,175] - 3 frac[195,60]
   - 3 frac[195,148] - 3 frac[195,37] - 3 frac[195,157] - 3 frac[195,42]
   - 3 frac[195,245] - 3 frac[195,130] - 3 frac[195,19] - 3 frac[195,250]
   - 3 frac[195,24] - 3 frac[195,227] - 3 frac[195,112] - 3 frac[195,1]
   - 3 frac[195,232] - 3 frac[195,121] - 3 frac[195,6] - 3 frac[195,209]
   - 3 frac[195,126] - 3 frac[195,15] - 3 frac[195,214] - 3 frac[195,223]
   - 3 frac[195,108] - 3 frac[195,85] - 3 frac[195,205] - 3 frac[195,90]
   - 3 frac[195,178] - 3 frac[195,67] - 3 frac[195,187] - 3 frac[195,72]
   - 3 frac[195,160] - 3 frac[195,49] - 3 frac[195,169] - 3 frac[195,54]
   - 3 frac[195,174] - 3 frac[195,63] - 3 frac[195,151] - 3 frac[195,36]
   - 3 frac[195,45] - 3 frac[195,244] - 3 frac[195,133] - 3 frac[195,253]
   - 3 frac[195,138] - 3 frac[195,27] - 3 frac[195,226] - 3 frac[195,0]
   - 3 frac[195,235] - 3 frac[195,120] - 3 frac[195,9] - 3 frac[195,208]
   - 3 frac[195,97] - 3 frac[195,14] - 3 frac[195,217] - 3 frac[195,102]
   - 3 frac[195,222] - 3 frac[195,111] - 3 frac[195,199] - 3 frac[195,84]
   - 3 frac[195,204] - 3 frac[195,93] - 3 frac[195,181] - 3 frac[195,66]
   - 3 frac[195,186] - 3 frac[195,75] - 3 frac[195,163] - 3 frac[195,48]
   - 3 frac[195,168] - 3 frac[195,57] - 3 frac[195,145] - 3 frac[195,62]
   + rows_tot_tor_P44 = 0
 mem_tot_l1_P41: - mem[196,86] - mem[196,229] - mem[196,93] - mem[196,236]
   - mem[196,53] - mem[196,68] - mem[196,235] - mem[196,60] - mem[196,67]
   - mem[196,210] - mem[196,59] - mem[196,74] - mem[196,217] - mem[196,34]
   - mem[196,177] - mem[196,192] - mem[196,41] - mem[196,184]
   - mem[196,207] - mem[196,16] - mem[196,167] - mem[196,31] - mem[196,174]
   - mem[196,6] - mem[196,149] - mem[196,13] - mem[196,156] - mem[196,116]
   - mem[196,155] - mem[196,115] - mem[196,130] - mem[196,122]
   - mem[196,137] - mem[196,240] - mem[196,104] - mem[196,255]
   - mem[196,87] - mem[196,230] - mem[196,94] - mem[196,54] - mem[196,69]
   - mem[196,212] - mem[196,61] - mem[196,76] - mem[196,211] - mem[196,36]
   - mem[196,75] - mem[196,218] - mem[196,35] - mem[196,193] - mem[196,42]
   - mem[196,185] - mem[196,17] - mem[196,160] - mem[196,24] - mem[196,175]
   - mem[196,7] - mem[196,150] - mem[196,14] - mem[196,157] - mem[196,117]
   - mem[196,132] - mem[196,124] - mem[196,131] - mem[196,123]
   - mem[196,138] - mem[196,98] - mem[196,241] - mem[196,105]
   - mem[196,248] - mem[196,80] - mem[196,231] - mem[196,95] - mem[196,238]
   - mem[196,55] - mem[196,70] - mem[196,213] - mem[196,62] - mem[196,77]
   - mem[196,220] - mem[196,37] - mem[196,180] - mem[196,219] - mem[196,44]
   - mem[196,179] - mem[196,194] - mem[196,43] - mem[196,186]
   - mem[196,201] - mem[196,18] - mem[196,161] - mem[196,25] - mem[196,168]
   - mem[196,0] - mem[196,151] - mem[196,15] - mem[196,158] - mem[196,133]
   - mem[196,140] - mem[196,100] - mem[196,139] - mem[196,99]
   - mem[196,242] - mem[196,106] - mem[196,249] - mem[196,81]
   - mem[196,224] - mem[196,88] - mem[196,239] - mem[196,48] - mem[196,71]
   - mem[196,214] - mem[196,63] - mem[196,78] - mem[196,221] - mem[196,38]
   - mem[196,181] - mem[196,196] - mem[196,45] - mem[196,188]
   - mem[196,195] - mem[196,20] - mem[196,187] - mem[196,202] - mem[196,19]
   - mem[196,162] - mem[196,26] - mem[196,169] - mem[196,1] - mem[196,144]
   - mem[196,8] - mem[196,159] - mem[196,119] - mem[196,134] - mem[196,126]
   - mem[196,141] - mem[196,101] - mem[196,244] - mem[196,108]
   - mem[196,243] - mem[196,107] - mem[196,250] - mem[196,82]
   - mem[196,225] - mem[196,89] - mem[196,232] - mem[196,49] - mem[196,64]
   - mem[196,215] - mem[196,56] - mem[196,79] - mem[196,222] - mem[196,39]
   - mem[196,182] - mem[196,197] - mem[196,46] - mem[196,189]
   - mem[196,204] - mem[196,21] - mem[196,164] - mem[196,203] - mem[196,28]
   - mem[196,163] - mem[196,27] - mem[196,170] - mem[196,2] - mem[196,145]
   - mem[196,9] - mem[196,152] - mem[196,112] - mem[196,135] - mem[196,127]
   - mem[196,142] - mem[196,102] - mem[196,245] - mem[196,109]
   - mem[196,252] - mem[196,84] - mem[196,251] - mem[196,83] - mem[196,226]
   - mem[196,90] - mem[196,233] - mem[196,50] - mem[196,65] - mem[196,208]
   - mem[196,57] - mem[196,72] - mem[196,223] - mem[196,32] - mem[196,183]
   - mem[196,198] - mem[196,47] - mem[196,190] - mem[196,205] - mem[196,22]
   - mem[196,165] - mem[196,29] - mem[196,172] - mem[196,4] - mem[196,171]
   - mem[196,146] - mem[196,10] - mem[196,153] - mem[196,113]
   - mem[196,128] - mem[196,120] - mem[196,143] - mem[196,103]
   - mem[196,246] - mem[196,110] - mem[196,253] - mem[196,85]
   - mem[196,228] - mem[196,92] - mem[196,227] - mem[196,52] - mem[196,91]
   - mem[196,51] - mem[196,66] - mem[196,209] - mem[196,58] - mem[196,73]
   - mem[196,216] - mem[196,33] - mem[196,176] - mem[196,199] - mem[196,40]
   - mem[196,191] - mem[196,206] - mem[196,23] - mem[196,166] - mem[196,30]
   - mem[196,173] - mem[196,5] - mem[196,148] - mem[196,12] - mem[196,147]
   - mem[196,11] - mem[196,154] - mem[196,114] - mem[196,129]
   - mem[196,121] - mem[196,136] - mem[196,96] - mem[196,247]
   - mem[196,111] - mem[196,254] + mem_tot_l1_P41 = 0
 rows_tot_l1_P41: - 3 frac[196,86] - 3 frac[196,229] - 3 frac[196,93]
   - 3 frac[196,236] - 3 frac[196,53] - 3 frac[196,68] - 3 frac[196,235]
   - 3 frac[196,60] - 3 frac[196,67] - 3 frac[196,210] - 3 frac[196,59]
   - 3 frac[196,74] - 3 frac[196,217] - 3 frac[196,34] - 3 frac[196,177]
   - 3 frac[196,192] - 3 frac[196,41] - 3 frac[196,184] - 3 frac[196,207]
   - 3 frac[196,16] - 3 frac[196,167] - 3 frac[196,31] - 3 frac[196,174]
   - 3 frac[196,6] - 3 frac[196,149] - 3 frac[196,13] - 3 frac[196,156]
   - 3 frac[196,116] - 3 frac[196,155] - 3 frac[196,115] - 3 frac[196,130]
   - 3 frac[196,122] - 3 frac[196,137] - 3 frac[196,240] - 3 frac[196,104]
   - 3 frac[196,255] - 3 frac[196,87] - 3 frac[196,230] - 3 frac[196,94]
   - 3 frac[196,54] - 3 frac[196,69] - 3 frac[196,212] - 3 frac[196,61]
   - 3 frac[196,76] - 3 frac[196,211] - 3 frac[196,36] - 3 frac[196,75]
   - 3 frac[196,218] - 3 frac[196,35] - 3 frac[196,193] - 3 frac[196,42]
   - 3 frac[196,185] - 3 frac[196,17] - 3 frac[196,160] - 3 frac[196,24]
   - 3 frac[196,175] - 3 frac[196,7] - 3 frac[196,150] - 3 frac[196,14]
   - 3 frac[196,157] - 3 frac[196,117] - 3 frac[196,132] - 3 frac[196,124]
   - 3 frac[196,131] - 3 frac[196,123] - 3 frac[196,138] - 3 frac[196,98]
   - 3 frac[196,241] - 3 frac[196,105] - 3 frac[196,248] - 3 frac[196,80]
   - 3 frac[196,231] - 3 frac[196,95] - 3 frac[196,238] - 3 frac[196,55]
   - 3 frac[196,70] - 3 frac[196,213] - 3 frac[196,62] - 3 frac[196,77]
   - 3 frac[196,220] - 3 frac[196,37] - 3 frac[196,180] - 3 frac[196,219]
   - 3 frac[196,44] - 3 frac[196,179] - 3 frac[196,194] - 3 frac[196,43]
   - 3 frac[196,186] - 3 frac[196,201] - 3 frac[196,18] - 3 frac[196,161]
   - 3 frac[196,25] - 3 frac[196,168] - 3 frac[196,0] - 3 frac[196,151]
   - 3 frac[196,15] - 3 frac[196,158] - 3 frac[196,133] - 3 frac[196,140]
   - 3 frac[196,100] - 3 frac[196,139] - 3 frac[196,99] - 3 frac[196,242]
   - 3 frac[196,106] - 3 frac[196,249] - 3 frac[196,81] - 3 frac[196,224]
   - 3 frac[196,88] - 3 frac[196,239] - 3 frac[196,48] - 3 frac[196,71]
   - 3 frac[196,214] - 3 frac[196,63] - 3 frac[196,78] - 3 frac[196,221]
   - 3 frac[196,38] - 3 frac[196,181] - 3 frac[196,196] - 3 frac[196,45]
   - 3 frac[196,188] - 3 frac[196,195] - 3 frac[196,20] - 3 frac[196,187]
   - 3 frac[196,202] - 3 frac[196,19] - 3 frac[196,162] - 3 frac[196,26]
   - 3 frac[196,169] - 3 frac[196,1] - 3 frac[196,144] - 3 frac[196,8]
   - 3 frac[196,159] - 3 frac[196,119] - 3 frac[196,134] - 3 frac[196,126]
   - 3 frac[196,141] - 3 frac[196,101] - 3 frac[196,244] - 3 frac[196,108]
   - 3 frac[196,243] - 3 frac[196,107] - 3 frac[196,250] - 3 frac[196,82]
   - 3 frac[196,225] - 3 frac[196,89] - 3 frac[196,232] - 3 frac[196,49]
   - 3 frac[196,64] - 3 frac[196,215] - 3 frac[196,56] - 3 frac[196,79]
   - 3 frac[196,222] - 3 frac[196,39] - 3 frac[196,182] - 3 frac[196,197]
   - 3 frac[196,46] - 3 frac[196,189] - 3 frac[196,204] - 3 frac[196,21]
   - 3 frac[196,164] - 3 frac[196,203] - 3 frac[196,28] - 3 frac[196,163]
   - 3 frac[196,27] - 3 frac[196,170] - 3 frac[196,2] - 3 frac[196,145]
   - 3 frac[196,9] - 3 frac[196,152] - 3 frac[196,112] - 3 frac[196,135]
   - 3 frac[196,127] - 3 frac[196,142] - 3 frac[196,102] - 3 frac[196,245]
   - 3 frac[196,109] - 3 frac[196,252] - 3 frac[196,84] - 3 frac[196,251]
   - 3 frac[196,83] - 3 frac[196,226] - 3 frac[196,90] - 3 frac[196,233]
   - 3 frac[196,50] - 3 frac[196,65] - 3 frac[196,208] - 3 frac[196,57]
   - 3 frac[196,72] - 3 frac[196,223] - 3 frac[196,32] - 3 frac[196,183]
   - 3 frac[196,198] - 3 frac[196,47] - 3 frac[196,190] - 3 frac[196,205]
   - 3 frac[196,22] - 3 frac[196,165] - 3 frac[196,29] - 3 frac[196,172]
   - 3 frac[196,4] - 3 frac[196,171] - 3 frac[196,146] - 3 frac[196,10]
   - 3 frac[196,153] - 3 frac[196,113] - 3 frac[196,128] - 3 frac[196,120]
   - 3 frac[196,143] - 3 frac[196,103] - 3 frac[196,246] - 3 frac[196,110]
   - 3 frac[196,253] - 3 frac[196,85] - 3 frac[196,228] - 3 frac[196,92]
   - 3 frac[196,227] - 3 frac[196,52] - 3 frac[196,91] - 3 frac[196,51]
   - 3 frac[196,66] - 3 frac[196,209] - 3 frac[196,58] - 3 frac[196,73]
   - 3 frac[196,216] - 3 frac[196,33] - 3 frac[196,176] - 3 frac[196,199]
   - 3 frac[196,40] - 3 frac[196,191] - 3 frac[196,206] - 3 frac[196,23]
   - 3 frac[196,166] - 3 frac[196,30] - 3 frac[196,173] - 3 frac[196,5]
   - 3 frac[196,148] - 3 frac[196,12] - 3 frac[196,147] - 3 frac[196,11]
   - 3 frac[196,154] - 3 frac[196,114] - 3 frac[196,129] - 3 frac[196,121]
   - 3 frac[196,136] - 3 frac[196,96] - 3 frac[196,247] - 3 frac[196,111]
   - 3 frac[196,254] + rows_tot_l1_P41 = 0
 mem_tot_l1_P42: - mem[197,96] - mem[197,241] - mem[197,130] - mem[197,105]
   - mem[197,250] - mem[197,139] - mem[197,82] - mem[197,227] - mem[197,91]
   - mem[197,236] - mem[197,68] - mem[197,213] - mem[197,77] - mem[197,222]
   - mem[197,54] - mem[197,199] - mem[197,192] - mem[197,56] - mem[197,201]
   - mem[197,33] - mem[197,178] - mem[197,42] - mem[197,187] - mem[197,19]
   - mem[197,164] - mem[197,173] - mem[197,116] - mem[197,5] - mem[197,150]
   - mem[197,125] - mem[197,14] - mem[197,159] - mem[197,102]
   - mem[197,247] - mem[197,152] - mem[197,111] - mem[197,240]
   - mem[197,129] - mem[197,104] - mem[197,249] - mem[197,138]
   - mem[197,81] - mem[197,226] - mem[197,90] - mem[197,235] - mem[197,67]
   - mem[197,212] - mem[197,76] - mem[197,221] - mem[197,53] - mem[197,198]
   - mem[197,62] - mem[197,207] - mem[197,39] - mem[197,32] - mem[197,177]
   - mem[197,41] - mem[197,186] - mem[197,18] - mem[197,163] - mem[197,27]
   - mem[197,172] - mem[197,115] - mem[197,4] - mem[197,149] - mem[197,124]
   - mem[197,13] - mem[197,158] - mem[197,101] - mem[197,246]
   - mem[197,135] - mem[197,110] - mem[197,255] - mem[197,128]
   - mem[197,87] - mem[197,248] - mem[197,137] - mem[197,80] - mem[197,225]
   - mem[197,89] - mem[197,66] - mem[197,211] - mem[197,75] - mem[197,220]
   - mem[197,52] - mem[197,197] - mem[197,61] - mem[197,206] - mem[197,38]
   - mem[197,183] - mem[197,47] - mem[197,176] - mem[197,40] - mem[197,185]
   - mem[197,17] - mem[197,162] - mem[197,26] - mem[197,171] - mem[197,114]
   - mem[197,3] - mem[197,148] - mem[197,123] - mem[197,12] - mem[197,157]
   - mem[197,100] - mem[197,245] - mem[197,134] - mem[197,109]
   - mem[197,254] - mem[197,143] - mem[197,86] - mem[197,231]
   - mem[197,136] - mem[197,95] - mem[197,224] - mem[197,88] - mem[197,233]
   - mem[197,65] - mem[197,210] - mem[197,74] - mem[197,219] - mem[197,51]
   - mem[197,196] - mem[197,60] - mem[197,205] - mem[197,37] - mem[197,182]
   - mem[197,46] - mem[197,191] - mem[197,23] - mem[197,184] - mem[197,16]
   - mem[197,161] - mem[197,25] - mem[197,170] - mem[197,113] - mem[197,2]
   - mem[197,147] - mem[197,122] - mem[197,11] - mem[197,156] - mem[197,99]
   - mem[197,244] - mem[197,133] - mem[197,108] - mem[197,253]
   - mem[197,142] - mem[197,85] - mem[197,230] - mem[197,94] - mem[197,239]
   - mem[197,71] - mem[197,232] - mem[197,64] - mem[197,209] - mem[197,73]
   - mem[197,50] - mem[197,195] - mem[197,59] - mem[197,204] - mem[197,36]
   - mem[197,181] - mem[197,45] - mem[197,190] - mem[197,22] - mem[197,167]
   - mem[197,31] - mem[197,160] - mem[197,119] - mem[197,24] - mem[197,169]
   - mem[197,112] - mem[197,1] - mem[197,146] - mem[197,121] - mem[197,10]
   - mem[197,155] - mem[197,98] - mem[197,243] - mem[197,132]
   - mem[197,107] - mem[197,252] - mem[197,141] - mem[197,84]
   - mem[197,229] - mem[197,93] - mem[197,70] - mem[197,215] - mem[197,79]
   - mem[197,208] - mem[197,72] - mem[197,217] - mem[197,49] - mem[197,194]
   - mem[197,58] - mem[197,203] - mem[197,35] - mem[197,180] - mem[197,44]
   - mem[197,189] - mem[197,21] - mem[197,166] - mem[197,30] - mem[197,175]
   - mem[197,7] - mem[197,168] - mem[197,127] - mem[197,0] - mem[197,145]
   - mem[197,120] - mem[197,9] - mem[197,154] - mem[197,97] - mem[197,242]
   - mem[197,131] - mem[197,106] - mem[197,251] - mem[197,140]
   - mem[197,83] - mem[197,228] - mem[197,92] - mem[197,237] - mem[197,69]
   - mem[197,214] - mem[197,78] - mem[197,223] - mem[197,55] - mem[197,216]
   - mem[197,48] - mem[197,193] - mem[197,57] - mem[197,202] - mem[197,34]
   - mem[197,179] - mem[197,43] - mem[197,188] - mem[197,20] - mem[197,165]
   - mem[197,29] - mem[197,174] - mem[197,117] - mem[197,6] - mem[197,151]
   - mem[197,126] - mem[197,15] - mem[197,144] - mem[197,8] - mem[197,153]
   + mem_tot_l1_P42 = 0
 rows_tot_l1_P42: - 3 frac[197,96] - 3 frac[197,241] - 3 frac[197,130]
   - 3 frac[197,105] - 3 frac[197,250] - 3 frac[197,139] - 3 frac[197,82]
   - 3 frac[197,227] - 3 frac[197,91] - 3 frac[197,236] - 3 frac[197,68]
   - 3 frac[197,213] - 3 frac[197,77] - 3 frac[197,222] - 3 frac[197,54]
   - 3 frac[197,199] - 3 frac[197,192] - 3 frac[197,56] - 3 frac[197,201]
   - 3 frac[197,33] - 3 frac[197,178] - 3 frac[197,42] - 3 frac[197,187]
   - 3 frac[197,19] - 3 frac[197,164] - 3 frac[197,173] - 3 frac[197,116]
   - 3 frac[197,5] - 3 frac[197,150] - 3 frac[197,125] - 3 frac[197,14]
   - 3 frac[197,159] - 3 frac[197,102] - 3 frac[197,247] - 3 frac[197,152]
   - 3 frac[197,111] - 3 frac[197,240] - 3 frac[197,129] - 3 frac[197,104]
   - 3 frac[197,249] - 3 frac[197,138] - 3 frac[197,81] - 3 frac[197,226]
   - 3 frac[197,90] - 3 frac[197,235] - 3 frac[197,67] - 3 frac[197,212]
   - 3 frac[197,76] - 3 frac[197,221] - 3 frac[197,53] - 3 frac[197,198]
   - 3 frac[197,62] - 3 frac[197,207] - 3 frac[197,39] - 3 frac[197,32]
   - 3 frac[197,177] - 3 frac[197,41] - 3 frac[197,186] - 3 frac[197,18]
   - 3 frac[197,163] - 3 frac[197,27] - 3 frac[197,172] - 3 frac[197,115]
   - 3 frac[197,4] - 3 frac[197,149] - 3 frac[197,124] - 3 frac[197,13]
   - 3 frac[197,158] - 3 frac[197,101] - 3 frac[197,246] - 3 frac[197,135]
   - 3 frac[197,110] - 3 frac[197,255] - 3 frac[197,128] - 3 frac[197,87]
   - 3 frac[197,248] - 3 frac[197,137] - 3 frac[197,80] - 3 frac[197,225]
   - 3 frac[197,89] - 3 frac[197,66] - 3 frac[197,211] - 3 frac[197,75]
   - 3 frac[197,220] - 3 frac[197,52] - 3 frac[197,197] - 3 frac[197,61]
   - 3 frac[197,206] - 3 frac[197,38] - 3 frac[197,183] - 3 frac[197,47]
   - 3 frac[197,176] - 3 frac[197,40] - 3 frac[197,185] - 3 frac[197,17]
   - 3 frac[197,162] - 3 frac[197,26] - 3 frac[197,171] - 3 frac[197,114]
   - 3 frac[197,3] - 3 frac[197,148] - 3 frac[197,123] - 3 frac[197,12]
   - 3 frac[197,157] - 3 frac[197,100] - 3 frac[197,245] - 3 frac[197,134]
   - 3 frac[197,109] - 3 frac[197,254] - 3 frac[197,143] - 3 frac[197,86]
   - 3 frac[197,231] - 3 frac[197,136] - 3 frac[197,95] - 3 frac[197,224]
   - 3 frac[197,88] - 3 frac[197,233] - 3 frac[197,65] - 3 frac[197,210]
   - 3 frac[197,74] - 3 frac[197,219] - 3 frac[197,51] - 3 frac[197,196]
   - 3 frac[197,60] - 3 frac[197,205] - 3 frac[197,37] - 3 frac[197,182]
   - 3 frac[197,46] - 3 frac[197,191] - 3 frac[197,23] - 3 frac[197,184]
   - 3 frac[197,16] - 3 frac[197,161] - 3 frac[197,25] - 3 frac[197,170]
   - 3 frac[197,113] - 3 frac[197,2] - 3 frac[197,147] - 3 frac[197,122]
   - 3 frac[197,11] - 3 frac[197,156] - 3 frac[197,99] - 3 frac[197,244]
   - 3 frac[197,133] - 3 frac[197,108] - 3 frac[197,253] - 3 frac[197,142]
   - 3 frac[197,85] - 3 frac[197,230] - 3 frac[197,94] - 3 frac[197,239]
   - 3 frac[197,71] - 3 frac[197,232] - 3 frac[197,64] - 3 frac[197,209]
   - 3 frac[197,73] - 3 frac[197,50] - 3 frac[197,195] - 3 frac[197,59]
   - 3 frac[197,204] - 3 frac[197,36] - 3 frac[197,181] - 3 frac[197,45]
   - 3 frac[197,190] - 3 frac[197,22] - 3 frac[197,167] - 3 frac[197,31]
   - 3 frac[197,160] - 3 frac[197,119] - 3 frac[197,24] - 3 frac[197,169]
   - 3 frac[197,112] - 3 frac[197,1] - 3 frac[197,146] - 3 frac[197,121]
   - 3 frac[197,10] - 3 frac[197,155] - 3 frac[197,98] - 3 frac[197,243]
   - 3 frac[197,132] - 3 frac[197,107] - 3 frac[197,252] - 3 frac[197,141]
   - 3 frac[197,84] - 3 frac[197,229] - 3 frac[197,93] - 3 frac[197,70]
   - 3 frac[197,215] - 3 frac[197,79] - 3 frac[197,208] - 3 frac[197,72]
   - 3 frac[197,217] - 3 frac[197,49] - 3 frac[197,194] - 3 frac[197,58]
   - 3 frac[197,203] - 3 frac[197,35] - 3 frac[197,180] - 3 frac[197,44]
   - 3 frac[197,189] - 3 frac[197,21] - 3 frac[197,166] - 3 frac[197,30]
   - 3 frac[197,175] - 3 frac[197,7] - 3 frac[197,168] - 3 frac[197,127]
   - 3 frac[197,0] - 3 frac[197,145] - 3 frac[197,120] - 3 frac[197,9]
   - 3 frac[197,154] - 3 frac[197,97] - 3 frac[197,242] - 3 frac[197,131]
   - 3 frac[197,106] - 3 frac[197,251] - 3 frac[197,140] - 3 frac[197,83]
   - 3 frac[197,228] - 3 frac[197,92] - 3 frac[197,237] - 3 frac[197,69]
   - 3 frac[197,214] - 3 frac[197,78] - 3 frac[197,223] - 3 frac[197,55]
   - 3 frac[197,216] - 3 frac[197,48] - 3 frac[197,193] - 3 frac[197,57]
   - 3 frac[197,202] - 3 frac[197,34] - 3 frac[197,179] - 3 frac[197,43]
   - 3 frac[197,188] - 3 frac[197,20] - 3 frac[197,165] - 3 frac[197,29]
   - 3 frac[197,174] - 3 frac[197,117] - 3 frac[197,6] - 3 frac[197,151]
   - 3 frac[197,126] - 3 frac[197,15] - 3 frac[197,144] - 3 frac[197,8]
   - 3 frac[197,153] + rows_tot_l1_P42 = 0
 mem_tot_l2_P4: - mem[198,129] - mem[198,16] - mem[198,248] - mem[198,11]
   - mem[198,243] - mem[198,2] - mem[198,5] - mem[198,124] - mem[198,228]
   - mem[198,119] - mem[198,223] - mem[198,110] - mem[198,214]
   - mem[198,190] - mem[198,201] - mem[198,88] - mem[198,177]
   - mem[198,192] - mem[198,83] - mem[198,168] - mem[198,59] - mem[198,74]
   - mem[198,50] - mem[198,77] - mem[198,154] - mem[198,53] - mem[198,68]
   - mem[198,157] - mem[198,44] - mem[198,148] - mem[198,39] - mem[198,143]
   - mem[198,30] - mem[198,134] - mem[198,17] - mem[198,249] - mem[198,8]
   - mem[198,240] - mem[198,235] - mem[198,122] - mem[198,226]
   - mem[198,229] - mem[198,116] - mem[198,220] - mem[198,111]
   - mem[198,215] - mem[198,102] - mem[198,191] - mem[198,206]
   - mem[198,89] - mem[198,182] - mem[198,193] - mem[198,80] - mem[198,169]
   - mem[198,56] - mem[198,75] - mem[198,160] - mem[198,51] - mem[198,66]
   - mem[198,155] - mem[198,42] - mem[198,69] - mem[198,146] - mem[198,45]
   - mem[198,149] - mem[198,36] - mem[198,140] - mem[198,31] - mem[198,135]
   - mem[198,22] - mem[198,254] - mem[198,9] - mem[198,241] - mem[198,232]
   - mem[198,123] - mem[198,227] - mem[198,114] - mem[198,117]
   - mem[198,221] - mem[198,108] - mem[198,212] - mem[198,188]
   - mem[198,207] - mem[198,94] - mem[198,183] - mem[198,198] - mem[198,81]
   - mem[198,174] - mem[198,57] - mem[198,72] - mem[198,161] - mem[198,48]
   - mem[198,67] - mem[198,152] - mem[198,43] - mem[198,147] - mem[198,34]
   - mem[198,138] - mem[198,37] - mem[198,141] - mem[198,132] - mem[198,23]
   - mem[198,255] - mem[198,14] - mem[198,246] - mem[198,1] - mem[198,233]
   - mem[198,120] - mem[198,224] - mem[198,115] - mem[198,219]
   - mem[198,106] - mem[198,210] - mem[198,109] - mem[198,186]
   - mem[198,213] - mem[198,100] - mem[198,189] - mem[198,204]
   - mem[198,95] - mem[198,180] - mem[198,199] - mem[198,86] - mem[198,175]
   - mem[198,62] - mem[198,73] - mem[198,166] - mem[198,49] - mem[198,64]
   - mem[198,153] - mem[198,40] - mem[198,144] - mem[198,139] - mem[198,26]
   - mem[198,130] - mem[198,29] - mem[198,133] - mem[198,20] - mem[198,252]
   - mem[198,15] - mem[198,247] - mem[198,6] - mem[198,121] - mem[198,225]
   - mem[198,112] - mem[198,216] - mem[198,107] - mem[198,211]
   - mem[198,98] - mem[198,187] - mem[198,202] - mem[198,101]
   - mem[198,205] - mem[198,92] - mem[198,181] - mem[198,196] - mem[198,87]
   - mem[198,172] - mem[198,78] - mem[198,167] - mem[198,54] - mem[198,65]
   - mem[198,158] - mem[198,41] - mem[198,145] - mem[198,32] - mem[198,136]
   - mem[198,27] - mem[198,131] - mem[198,18] - mem[198,250] - mem[198,21]
   - mem[198,253] - mem[198,12] - mem[198,244] - mem[198,7] - mem[198,239]
   - mem[198,126] - mem[198,230] - mem[198,113] - mem[198,217]
   - mem[198,104] - mem[198,208] - mem[198,99] - mem[198,184]
   - mem[198,203] - mem[198,90] - mem[198,179] - mem[198,194] - mem[198,93]
   - mem[198,170] - mem[198,197] - mem[198,84] - mem[198,173] - mem[198,60]
   - mem[198,79] - mem[198,164] - mem[198,55] - mem[198,70] - mem[198,159]
   - mem[198,46] - mem[198,150] - mem[198,33] - mem[198,137] - mem[198,24]
   - mem[198,128] - mem[198,19] - mem[198,10] - mem[198,242] - mem[198,13]
   - mem[198,245] - mem[198,4] - mem[198,236] - mem[198,127] - mem[198,231]
   - mem[198,222] - mem[198,105] - mem[198,209] - mem[198,96]
   - mem[198,185] - mem[198,91] - mem[198,176] - mem[198,195] - mem[198,82]
   - mem[198,171] - mem[198,58] - mem[198,85] - mem[198,162] - mem[198,61]
   - mem[198,76] - mem[198,165] - mem[198,52] - mem[198,71] - mem[198,156]
   - mem[198,47] - mem[198,151] - mem[198,38] - mem[198,142] - mem[198,25]
   + mem_tot_l2_P4 = 0
 rows_tot_l2_P4: - 3 frac[198,129] - 3 frac[198,16] - 3 frac[198,248]
   - 3 frac[198,11] - 3 frac[198,243] - 3 frac[198,2] - 3 frac[198,5]
   - 3 frac[198,124] - 3 frac[198,228] - 3 frac[198,119] - 3 frac[198,223]
   - 3 frac[198,110] - 3 frac[198,214] - 3 frac[198,190] - 3 frac[198,201]
   - 3 frac[198,88] - 3 frac[198,177] - 3 frac[198,192] - 3 frac[198,83]
   - 3 frac[198,168] - 3 frac[198,59] - 3 frac[198,74] - 3 frac[198,50]
   - 3 frac[198,77] - 3 frac[198,154] - 3 frac[198,53] - 3 frac[198,68]
   - 3 frac[198,157] - 3 frac[198,44] - 3 frac[198,148] - 3 frac[198,39]
   - 3 frac[198,143] - 3 frac[198,30] - 3 frac[198,134] - 3 frac[198,17]
   - 3 frac[198,249] - 3 frac[198,8] - 3 frac[198,240] - 3 frac[198,235]
   - 3 frac[198,122] - 3 frac[198,226] - 3 frac[198,229] - 3 frac[198,116]
   - 3 frac[198,220] - 3 frac[198,111] - 3 frac[198,215] - 3 frac[198,102]
   - 3 frac[198,191] - 3 frac[198,206] - 3 frac[198,89] - 3 frac[198,182]
   - 3 frac[198,193] - 3 frac[198,80] - 3 frac[198,169] - 3 frac[198,56]
   - 3 frac[198,75] - 3 frac[198,160] - 3 frac[198,51] - 3 frac[198,66]
   - 3 frac[198,155] - 3 frac[198,42] - 3 frac[198,69] - 3 frac[198,146]
   - 3 frac[198,45] - 3 frac[198,149] - 3 frac[198,36] - 3 frac[198,140]
   - 3 frac[198,31] - 3 frac[198,135] - 3 frac[198,22] - 3 frac[198,254]
   - 3 frac[198,9] - 3 frac[198,241] - 3 frac[198,232] - 3 frac[198,123]
   - 3 frac[198,227] - 3 frac[198,114] - 3 frac[198,117] - 3 frac[198,221]
   - 3 frac[198,108] - 3 frac[198,212] - 3 frac[198,188] - 3 frac[198,207]
   - 3 frac[198,94] - 3 frac[198,183] - 3 frac[198,198] - 3 frac[198,81]
   - 3 frac[198,174] - 3 frac[198,57] - 3 frac[198,72] - 3 frac[198,161]
   - 3 frac[198,48] - 3 frac[198,67] - 3 frac[198,152] - 3 frac[198,43]
   - 3 frac[198,147] - 3 frac[198,34] - 3 frac[198,138] - 3 frac[198,37]
   - 3 frac[198,141] - 3 frac[198,132] - 3 frac[198,23] - 3 frac[198,255]
   - 3 frac[198,14] - 3 frac[198,246] - 3 frac[198,1] - 3 frac[198,233]
   - 3 frac[198,120] - 3 frac[198,224] - 3 frac[198,115] - 3 frac[198,219]
   - 3 frac[198,106] - 3 frac[198,210] - 3 frac[198,109] - 3 frac[198,186]
   - 3 frac[198,213] - 3 frac[198,100] - 3 frac[198,189] - 3 frac[198,204]
   - 3 frac[198,95] - 3 frac[198,180] - 3 frac[198,199] - 3 frac[198,86]
   - 3 frac[198,175] - 3 frac[198,62] - 3 frac[198,73] - 3 frac[198,166]
   - 3 frac[198,49] - 3 frac[198,64] - 3 frac[198,153] - 3 frac[198,40]
   - 3 frac[198,144] - 3 frac[198,139] - 3 frac[198,26] - 3 frac[198,130]
   - 3 frac[198,29] - 3 frac[198,133] - 3 frac[198,20] - 3 frac[198,252]
   - 3 frac[198,15] - 3 frac[198,247] - 3 frac[198,6] - 3 frac[198,121]
   - 3 frac[198,225] - 3 frac[198,112] - 3 frac[198,216] - 3 frac[198,107]
   - 3 frac[198,211] - 3 frac[198,98] - 3 frac[198,187] - 3 frac[198,202]
   - 3 frac[198,101] - 3 frac[198,205] - 3 frac[198,92] - 3 frac[198,181]
   - 3 frac[198,196] - 3 frac[198,87] - 3 frac[198,172] - 3 frac[198,78]
   - 3 frac[198,167] - 3 frac[198,54] - 3 frac[198,65] - 3 frac[198,158]
   - 3 frac[198,41] - 3 frac[198,145] - 3 frac[198,32] - 3 frac[198,136]
   - 3 frac[198,27] - 3 frac[198,131] - 3 frac[198,18] - 3 frac[198,250]
   - 3 frac[198,21] - 3 frac[198,253] - 3 frac[198,12] - 3 frac[198,244]
   - 3 frac[198,7] - 3 frac[198,239] - 3 frac[198,126] - 3 frac[198,230]
   - 3 frac[198,113] - 3 frac[198,217] - 3 frac[198,104] - 3 frac[198,208]
   - 3 frac[198,99] - 3 frac[198,184] - 3 frac[198,203] - 3 frac[198,90]
   - 3 frac[198,179] - 3 frac[198,194] - 3 frac[198,93] - 3 frac[198,170]
   - 3 frac[198,197] - 3 frac[198,84] - 3 frac[198,173] - 3 frac[198,60]
   - 3 frac[198,79] - 3 frac[198,164] - 3 frac[198,55] - 3 frac[198,70]
   - 3 frac[198,159] - 3 frac[198,46] - 3 frac[198,150] - 3 frac[198,33]
   - 3 frac[198,137] - 3 frac[198,24] - 3 frac[198,128] - 3 frac[198,19]
   - 3 frac[198,10] - 3 frac[198,242] - 3 frac[198,13] - 3 frac[198,245]
   - 3 frac[198,4] - 3 frac[198,236] - 3 frac[198,127] - 3 frac[198,231]
   - 3 frac[198,222] - 3 frac[198,105] - 3 frac[198,209] - 3 frac[198,96]
   - 3 frac[198,185] - 3 frac[198,91] - 3 frac[198,176] - 3 frac[198,195]
   - 3 frac[198,82] - 3 frac[198,171] - 3 frac[198,58] - 3 frac[198,85]
   - 3 frac[198,162] - 3 frac[198,61] - 3 frac[198,76] - 3 frac[198,165]
   - 3 frac[198,52] - 3 frac[198,71] - 3 frac[198,156] - 3 frac[198,47]
   - 3 frac[198,151] - 3 frac[198,38] - 3 frac[198,142] - 3 frac[198,25]
   + rows_tot_l2_P4 = 0
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
 row_capacity_CPU33: rows_tot_CPU33 <= 12
 row_capacity_CPU34: rows_tot_CPU34 <= 12
 row_capacity_CPU35: rows_tot_CPU35 <= 12
 row_capacity_CPU36: rows_tot_CPU36 <= 12
 row_capacity_CPU37: rows_tot_CPU37 <= 12
 row_capacity_CPU38: rows_tot_CPU38 <= 12
 row_capacity_CPU39: rows_tot_CPU39 <= 12
 row_capacity_CPU40: rows_tot_CPU40 <= 12
 row_capacity_CPU41: rows_tot_CPU41 <= 12
 row_capacity_CPU42: rows_tot_CPU42 <= 12
 row_capacity_CPU43: rows_tot_CPU43 <= 12
 row_capacity_CPU44: rows_tot_CPU44 <= 12
 row_capacity_CPU45: rows_tot_CPU45 <= 12
 row_capacity_CPU46: rows_tot_CPU46 <= 12
 row_capacity_CPU47: rows_tot_CPU47 <= 12
 row_capacity_CPU48: rows_tot_CPU48 <= 12
 row_capacity_CPU49: rows_tot_CPU49 <= 12
 row_capacity_CPU50: rows_tot_CPU50 <= 12
 row_capacity_CPU51: rows_tot_CPU51 <= 12
 row_capacity_CPU52: rows_tot_CPU52 <= 12
 row_capacity_CPU53: rows_tot_CPU53 <= 12
 row_capacity_CPU54: rows_tot_CPU54 <= 12
 row_capacity_CPU55: rows_tot_CPU55 <= 12
 row_capacity_CPU56: rows_tot_CPU56 <= 12
 row_capacity_CPU57: rows_tot_CPU57 <= 12
 row_capacity_CPU58: rows_tot_CPU58 <= 12
 row_capacity_CPU59: rows_tot_CPU59 <= 12
 row_capacity_CPU60: rows_tot_CPU60 <= 12
 row_capacity_CPU61: rows_tot_CPU61 <= 12
 row_capacity_CPU62: rows_tot_CPU62 <= 12
 row_capacity_CPU63: rows_tot_CPU63 <= 12
 row_capacity_CPU64: rows_tot_CPU64 <= 12
 row_capacity_CPU65: rows_tot_CPU65 <= 12
 row_capacity_CPU66: rows_tot_CPU66 <= 12
 row_capacity_CPU67: rows_tot_CPU67 <= 12
 row_capacity_CPU68: rows_tot_CPU68 <= 12
 row_capacity_CPU69: rows_tot_CPU69 <= 12
 row_capacity_CPU70: rows_tot_CPU70 <= 12
 row_capacity_CPU71: rows_tot_CPU71 <= 12
 row_capacity_CPU72: rows_tot_CPU72 <= 12
 row_capacity_CPU73: rows_tot_CPU73 <= 12
 row_capacity_CPU74: rows_tot_CPU74 <= 12
 row_capacity_CPU75: rows_tot_CPU75 <= 12
 row_capacity_CPU76: rows_tot_CPU76 <= 12
 row_capacity_CPU77: rows_tot_CPU77 <= 12
 row_capacity_CPU78: rows_tot_CPU78 <= 12
 row_capacity_CPU79: rows_tot_CPU79 <= 12
 row_capacity_CPU80: rows_tot_CPU80 <= 12
 row_capacity_CPU81: rows_tot_CPU81 <= 12
 row_capacity_CPU82: rows_tot_CPU82 <= 12
 row_capacity_CPU83: rows_tot_CPU83 <= 12
 row_capacity_CPU84: rows_tot_CPU84 <= 12
 row_capacity_CPU85: rows_tot_CPU85 <= 12
 row_capacity_CPU86: rows_tot_CPU86 <= 12
 row_capacity_CPU87: rows_tot_CPU87 <= 12
 row_capacity_CPU88: rows_tot_CPU88 <= 12
 row_capacity_CPU89: rows_tot_CPU89 <= 12
 row_capacity_CPU90: rows_tot_CPU90 <= 12
 row_capacity_CPU91: rows_tot_CPU91 <= 12
 row_capacity_CPU92: rows_tot_CPU92 <= 12
 row_capacity_CPU93: rows_tot_CPU93 <= 12
 row_capacity_CPU94: rows_tot_CPU94 <= 12
 row_capacity_CPU95: rows_tot_CPU95 <= 12
 row_capacity_CPU96: rows_tot_CPU96 <= 12
 row_capacity_CPU97: rows_tot_CPU97 <= 12
 row_capacity_CPU98: rows_tot_CPU98 <= 12
 row_capacity_CPU99: rows_tot_CPU99 <= 12
 row_capacity_CPU100: rows_tot_CPU100 <= 12
 row_capacity_CPU101: rows_tot_CPU101 <= 12
 row_capacity_CPU102: rows_tot_CPU102 <= 12
 row_capacity_CPU103: rows_tot_CPU103 <= 12
 row_capacity_CPU104: rows_tot_CPU104 <= 12
 row_capacity_CPU105: rows_tot_CPU105 <= 12
 row_capacity_CPU106: rows_tot_CPU106 <= 12
 row_capacity_CPU107: rows_tot_CPU107 <= 12
 row_capacity_CPU108: rows_tot_CPU108 <= 12
 row_capacity_CPU109: rows_tot_CPU109 <= 12
 row_capacity_CPU110: rows_tot_CPU110 <= 12
 row_capacity_CPU111: rows_tot_CPU111 <= 12
 row_capacity_CPU112: rows_tot_CPU112 <= 12
 row_capacity_CPU113: rows_tot_CPU113 <= 12
 row_capacity_CPU114: rows_tot_CPU114 <= 12
 row_capacity_CPU115: rows_tot_CPU115 <= 12
 row_capacity_CPU116: rows_tot_CPU116 <= 12
 row_capacity_CPU117: rows_tot_CPU117 <= 12
 row_capacity_CPU118: rows_tot_CPU118 <= 12
 row_capacity_CPU119: rows_tot_CPU119 <= 12
 row_capacity_CPU120: rows_tot_CPU120 <= 12
 row_capacity_CPU121: rows_tot_CPU121 <= 12
 row_capacity_CPU122: rows_tot_CPU122 <= 12
 row_capacity_CPU123: rows_tot_CPU123 <= 12
 row_capacity_CPU124: rows_tot_CPU124 <= 12
 row_capacity_CPU125: rows_tot_CPU125 <= 12
 row_capacity_CPU126: rows_tot_CPU126 <= 12
 row_capacity_CPU127: rows_tot_CPU127 <= 12
 row_capacity_CPU128: rows_tot_CPU128 <= 12
 row_capacity_CPU129: rows_tot_CPU129 <= 12
 row_capacity_CPU130: rows_tot_CPU130 <= 12
 row_capacity_CPU131: rows_tot_CPU131 <= 12
 row_capacity_CPU132: rows_tot_CPU132 <= 12
 row_capacity_CPU133: rows_tot_CPU133 <= 12
 row_capacity_CPU134: rows_tot_CPU134 <= 12
 row_capacity_CPU135: rows_tot_CPU135 <= 12
 row_capacity_CPU136: rows_tot_CPU136 <= 12
 row_capacity_CPU137: rows_tot_CPU137 <= 12
 row_capacity_CPU138: rows_tot_CPU138 <= 12
 row_capacity_CPU139: rows_tot_CPU139 <= 12
 row_capacity_CPU140: rows_tot_CPU140 <= 12
 row_capacity_CPU141: rows_tot_CPU141 <= 12
 row_capacity_CPU142: rows_tot_CPU142 <= 12
 row_capacity_CPU143: rows_tot_CPU143 <= 12
 row_capacity_CPU144: rows_tot_CPU144 <= 12
 row_capacity_CPU145: rows_tot_CPU145 <= 12
 row_capacity_CPU146: rows_tot_CPU146 <= 12
 row_capacity_CPU147: rows_tot_CPU147 <= 12
 row_capacity_CPU148: rows_tot_CPU148 <= 12
 row_capacity_CPU149: rows_tot_CPU149 <= 12
 row_capacity_CPU150: rows_tot_CPU150 <= 12
 row_capacity_CPU151: rows_tot_CPU151 <= 12
 row_capacity_CPU152: rows_tot_CPU152 <= 12
 row_capacity_CPU153: rows_tot_CPU153 <= 12
 row_capacity_CPU154: rows_tot_CPU154 <= 12
 row_capacity_CPU155: rows_tot_CPU155 <= 12
 row_capacity_CPU156: rows_tot_CPU156 <= 12
 row_capacity_CPU157: rows_tot_CPU157 <= 12
 row_capacity_CPU158: rows_tot_CPU158 <= 12
 row_capacity_CPU159: rows_tot_CPU159 <= 12
 row_capacity_CPU160: rows_tot_CPU160 <= 12
 row_capacity_CPU161: rows_tot_CPU161 <= 12
 row_capacity_CPU162: rows_tot_CPU162 <= 12
 row_capacity_CPU163: rows_tot_CPU163 <= 12
 row_capacity_CPU164: rows_tot_CPU164 <= 12
 row_capacity_CPU165: rows_tot_CPU165 <= 12
 row_capacity_CPU166: rows_tot_CPU166 <= 12
 row_capacity_CPU167: rows_tot_CPU167 <= 12
 row_capacity_CPU168: rows_tot_CPU168 <= 12
 row_capacity_CPU169: rows_tot_CPU169 <= 12
 row_capacity_CPU170: rows_tot_CPU170 <= 12
 row_capacity_CPU171: rows_tot_CPU171 <= 12
 row_capacity_CPU172: rows_tot_CPU172 <= 12
 row_capacity_CPU173: rows_tot_CPU173 <= 12
 row_capacity_CPU174: rows_tot_CPU174 <= 12
 row_capacity_CPU175: rows_tot_CPU175 <= 12
 row_capacity_CPU176: rows_tot_CPU176 <= 12
 row_capacity_CPU177: rows_tot_CPU177 <= 12
 row_capacity_CPU178: rows_tot_CPU178 <= 12
 row_capacity_CPU179: rows_tot_CPU179 <= 12
 row_capacity_CPU180: rows_tot_CPU180 <= 12
 row_capacity_CPU181: rows_tot_CPU181 <= 12
 row_capacity_CPU182: rows_tot_CPU182 <= 12
 row_capacity_CPU183: rows_tot_CPU183 <= 12
 row_capacity_CPU184: rows_tot_CPU184 <= 12
 row_capacity_CPU185: rows_tot_CPU185 <= 12
 row_capacity_CPU186: rows_tot_CPU186 <= 12
 row_capacity_CPU187: rows_tot_CPU187 <= 12
 row_capacity_CPU188: rows_tot_CPU188 <= 12
 row_capacity_CPU189: rows_tot_CPU189 <= 12
 row_capacity_CPU190: rows_tot_CPU190 <= 12
 row_capacity_CPU191: rows_tot_CPU191 <= 12
 row_capacity_CPU192: rows_tot_CPU192 <= 12
 row_capacity_tor_P41: rows_tot_tor_P41 <= 48
 row_capacity_tor_P42: rows_tot_tor_P42 <= 48
 row_capacity_tor_P43: rows_tot_tor_P43 <= 48
 row_capacity_tor_P44: rows_tot_tor_P44 <= 48
 row_capacity_l1_P41: rows_tot_l1_P41 <= 48
 row_capacity_l1_P42: rows_tot_l1_P42 <= 48
 row_capacity_l2_P4: rows_tot_l2_P4 <= 48
 tot_mem: - mem_tot_CPU1 - mem_tot_CPU2 - mem_tot_CPU3 - mem_tot_CPU4
   - mem_tot_CPU5 - mem_tot_CPU6 - mem_tot_CPU7 - mem_tot_CPU8
   - mem_tot_CPU9 - mem_tot_CPU10 - mem_tot_CPU11 - mem_tot_CPU12
   - mem_tot_CPU13 - mem_tot_CPU14 - mem_tot_CPU15 - mem_tot_CPU16
   - mem_tot_CPU17 - mem_tot_CPU18 - mem_tot_CPU19 - mem_tot_CPU20
   - mem_tot_CPU21 - mem_tot_CPU22 - mem_tot_CPU23 - mem_tot_CPU24
   - mem_tot_CPU25 - mem_tot_CPU26 - mem_tot_CPU27 - mem_tot_CPU28
   - mem_tot_CPU29 - mem_tot_CPU30 - mem_tot_CPU31 - mem_tot_CPU32
   - mem_tot_CPU33 - mem_tot_CPU34 - mem_tot_CPU35 - mem_tot_CPU36
   - mem_tot_CPU37 - mem_tot_CPU38 - mem_tot_CPU39 - mem_tot_CPU40
   - mem_tot_CPU41 - mem_tot_CPU42 - mem_tot_CPU43 - mem_tot_CPU44
   - mem_tot_CPU45 - mem_tot_CPU46 - mem_tot_CPU47 - mem_tot_CPU48
   - mem_tot_CPU49 - mem_tot_CPU50 - mem_tot_CPU51 - mem_tot_CPU52
   - mem_tot_CPU53 - mem_tot_CPU54 - mem_tot_CPU55 - mem_tot_CPU56
   - mem_tot_CPU57 - mem_tot_CPU58 - mem_tot_CPU59 - mem_tot_CPU60
   - mem_tot_CPU61 - mem_tot_CPU62 - mem_tot_CPU63 - mem_tot_CPU64
   - mem_tot_CPU65 - mem_tot_CPU66 - mem_tot_CPU67 - mem_tot_CPU68
   - mem_tot_CPU69 - mem_tot_CPU70 - mem_tot_CPU71 - mem_tot_CPU72
   - mem_tot_CPU73 - mem_tot_CPU74 - mem_tot_CPU75 - mem_tot_CPU76
   - mem_tot_CPU77 - mem_tot_CPU78 - mem_tot_CPU79 - mem_tot_CPU80
   - mem_tot_CPU81 - mem_tot_CPU82 - mem_tot_CPU83 - mem_tot_CPU84
   - mem_tot_CPU85 - mem_tot_CPU86 - mem_tot_CPU87 - mem_tot_CPU88
   - mem_tot_CPU89 - mem_tot_CPU90 - mem_tot_CPU91 - mem_tot_CPU92
   - mem_tot_CPU93 - mem_tot_CPU94 - mem_tot_CPU95 - mem_tot_CPU96
   - mem_tot_CPU97 - mem_tot_CPU98 - mem_tot_CPU99 - mem_tot_CPU100
   - mem_tot_CPU101 - mem_tot_CPU102 - mem_tot_CPU103 - mem_tot_CPU104
   - mem_tot_CPU105 - mem_tot_CPU106 - mem_tot_CPU107 - mem_tot_CPU108
   - mem_tot_CPU109 - mem_tot_CPU110 - mem_tot_CPU111 - mem_tot_CPU112
   - mem_tot_CPU113 - mem_tot_CPU114 - mem_tot_CPU115 - mem_tot_CPU116
   - mem_tot_CPU117 - mem_tot_CPU118 - mem_tot_CPU119 - mem_tot_CPU120
   - mem_tot_CPU121 - mem_tot_CPU122 - mem_tot_CPU123 - mem_tot_CPU124
   - mem_tot_CPU125 - mem_tot_CPU126 - mem_tot_CPU127 - mem_tot_CPU128
   - mem_tot_CPU129 - mem_tot_CPU130 - mem_tot_CPU131 - mem_tot_CPU132
   - mem_tot_CPU133 - mem_tot_CPU134 - mem_tot_CPU135 - mem_tot_CPU136
   - mem_tot_CPU137 - mem_tot_CPU138 - mem_tot_CPU139 - mem_tot_CPU140
   - mem_tot_CPU141 - mem_tot_CPU142 - mem_tot_CPU143 - mem_tot_CPU144
   - mem_tot_CPU145 - mem_tot_CPU146 - mem_tot_CPU147 - mem_tot_CPU148
   - mem_tot_CPU149 - mem_tot_CPU150 - mem_tot_CPU151 - mem_tot_CPU152
   - mem_tot_CPU153 - mem_tot_CPU154 - mem_tot_CPU155 - mem_tot_CPU156
   - mem_tot_CPU157 - mem_tot_CPU158 - mem_tot_CPU159 - mem_tot_CPU160
   - mem_tot_CPU161 - mem_tot_CPU162 - mem_tot_CPU163 - mem_tot_CPU164
   - mem_tot_CPU165 - mem_tot_CPU166 - mem_tot_CPU167 - mem_tot_CPU168
   - mem_tot_CPU169 - mem_tot_CPU170 - mem_tot_CPU171 - mem_tot_CPU172
   - mem_tot_CPU173 - mem_tot_CPU174 - mem_tot_CPU175 - mem_tot_CPU176
   - mem_tot_CPU177 - mem_tot_CPU178 - mem_tot_CPU179 - mem_tot_CPU180
   - mem_tot_CPU181 - mem_tot_CPU182 - mem_tot_CPU183 - mem_tot_CPU184
   - mem_tot_CPU185 - mem_tot_CPU186 - mem_tot_CPU187 - mem_tot_CPU188
   - mem_tot_CPU189 - mem_tot_CPU190 - mem_tot_CPU191 - mem_tot_CPU192
   - mem_tot_tor_P41 - mem_tot_tor_P42 - mem_tot_tor_P43 - mem_tot_tor_P44
   - mem_tot_l1_P41 - mem_tot_l1_P42 - mem_tot_l2_P4 + tot_mem = 0
Bounds
 frac[88,147] <= 1
 frac[169,227] <= 1
 frac[149,87] <= 1
 frac[43,3] <= 1
 frac[154,2] <= 1
 frac[192,241] <= 1
 frac[193,229] <= 1
 frac[194,237] <= 1
 frac[196,86] <= 1
 frac[34,153] <= 1
 frac[179,54] <= 1
 frac[197,96] <= 1
 frac[74,253] <= 1
 frac[59,131] <= 1
 frac[44,34] <= 1
 frac[100,199] <= 1
 frac[9,0] <= 1
 frac[191,232] <= 1
 frac[193,150] <= 1
 frac[105,20] <= 1
 frac[178,194] <= 1
 frac[196,229] <= 1
 frac[197,241] <= 1
 frac[198,129] <= 1
 frac[195,150] <= 1
 frac[55,247] <= 1
 frac[185,140] <= 1
 frac[98,45] <= 1
 frac[134,55] <= 1
 frac[192,42] <= 1
 frac[119,90] <= 1
 frac[102,182] <= 1
 frac[195,39] <= 1
 frac[123,31] <= 1
 frac[30,100] <= 1
 frac[197,130] <= 1
 frac[198,16] <= 1
 frac[89,227] <= 1
 frac[103,203] <= 1
 frac[192,89] <= 1
 frac[32,77] <= 1
 frac[140,153] <= 1
 frac[21,37] <= 1
 frac[39,127] <= 1
 frac[114,60] <= 1
 frac[116,229] <= 1
 frac[192,200] <= 1
 frac[193,254] <= 1
 frac[194,228] <= 1
 frac[65,11] <= 1
 frac[196,93] <= 1
 frac[197,105] <= 1
 frac[124,101] <= 1
 frac[73,91] <= 1
 frac[58,26] <= 1
 frac[99,244] <= 1
 frac[26,67] <= 1
 frac[194,23] <= 1
 frac[195,159] <= 1
 frac[196,236] <= 1
 frac[197,250] <= 1
 frac[198,248] <= 1
 frac[17,57] <= 1
 frac[180,38] <= 1
 frac[65,226] <= 1
 frac[192,33] <= 1
 frac[84,247] <= 1
 frac[195,44] <= 1
 frac[158,160] <= 1
 frac[107,190] <= 1
 frac[197,139] <= 1
 frac[198,11] <= 1
 frac[171,55] <= 1
 frac[62,145] <= 1
 frac[64,107] <= 1
 frac[193,70] <= 1
 frac[196,53] <= 1
 frac[185,119] <= 1
 frac[3,152] <= 1
 frac[169,237] <= 1
 frac[114,55] <= 1
 frac[96,225] <= 1
 frac[192,207] <= 1
 frac[193,247] <= 1
 frac[194,255] <= 1
 frac[195,247] <= 1
 frac[103,27] <= 1
 frac[196,68] <= 1
 frac[197,82] <= 1
 frac[142,184] <= 1
 frac[120,231] <= 1
 frac[110,59] <= 1
 frac[169,126] <= 1
 frac[9,50] <= 1
 frac[172,221] <= 1
 frac[155,63] <= 1
 frac[194,110] <= 1
 frac[195,132] <= 1
 frac[196,235] <= 1
 frac[197,227] <= 1
 frac[108,117] <= 1
 frac[198,243] <= 1
 frac[36,167] <= 1
 frac[76,78] <= 1
 frac[170,7] <= 1
 frac[192,56] <= 1
 frac[156,214] <= 1
 frac[138,44] <= 1
 frac[195,21] <= 1
 frac[31,108] <= 1
 frac[103,249] <= 1
 frac[198,2] <= 1
 frac[109,18] <= 1
 frac[163,60] <= 1
 frac[113,215] <= 1
 frac[192,87] <= 1
 frac[193,95] <= 1
 frac[65,148] <= 1
 frac[51,169] <= 1
 frac[159,200] <= 1
 frac[196,60] <= 1
 frac[71,90] <= 1
 frac[141,162] <= 1
 frac[147,199] <= 1
 frac[183,214] <= 1
 frac[115,132] <= 1
 frac[118,147] <= 1
 frac[192,198] <= 1
 frac[121,224] <= 1
 frac[194,246] <= 1
 frac[123,196] <= 1
 frac[50,156] <= 1
 frac[196,67] <= 1
 frac[197,91] <= 1
 frac[70,219] <= 1
 frac[195,252] <= 1
 frac[165,98] <= 1
 frac[79,91] <= 1
 frac[189,124] <= 1
 frac[194,105] <= 1
 frac[195,141] <= 1
 frac[196,210] <= 1
 frac[197,236] <= 1
 frac[51,71] <= 1
 frac[55,194] <= 1
 frac[37,56] <= 1
 frac[170,14] <= 1
 frac[192,63] <= 1
 frac[84,5] <= 1
 frac[103,246] <= 1
 frac[195,26] <= 1
 frac[198,5] <= 1
 frac[189,147] <= 1
 frac[192,174] <= 1
 frac[48,227] <= 1
 frac[193,80] <= 1
 frac[85,98] <= 1
 frac[196,59] <= 1
 frac[124,3] <= 1
 frac[109,156] <= 1
 frac[146,166] <= 1
 frac[182,114] <= 1
 frac[74,142] <= 1
 frac[22,166] <= 1
 frac[44,119] <= 1
 frac[27,169] <= 1
 frac[192,221] <= 1
 frac[195,229] <= 1
 frac[124,114] <= 1
 frac[196,74] <= 1
 frac[197,68] <= 1
 frac[58,55] <= 1
 frac[80,232] <= 1
 frac[81,254] <= 1
 frac[156,181] <= 1
 frac[194,96] <= 1
 frac[103,158] <= 1
 frac[195,114] <= 1
 frac[196,217] <= 1
 frac[197,213] <= 1
 frac[112,110] <= 1
 frac[40,219] <= 1
 frac[41,166] <= 1
 frac[192,54] <= 1
 frac[101,68] <= 1
 frac[195,3] <= 1
 frac[159,41] <= 1
 frac[198,124] <= 1
 frac[127,174] <= 1
 frac[171,26] <= 1
 frac[193,41] <= 1
 frac[157,67] <= 1
 frac[196,34] <= 1
 frac[14,41] <= 1
 frac[143,144] <= 1
 frac[148,205] <= 1
 frac[114,32] <= 1
 frac[23,208] <= 1
 frac[190,149] <= 1
 frac[193,218] <= 1
 frac[104,86] <= 1
 frac[196,177] <= 1
 frac[197,77] <= 1
 frac[75,154] <= 1
 frac[152,55] <= 1
 frac[64,169] <= 1
 frac[119,134] <= 1
 frac[194,123] <= 1
 frac[174,215] <= 1
 frac[195,123] <= 1
 frac[196,192] <= 1
 frac[197,222] <= 1
 frac[198,228] <= 1
 frac[76,123] <= 1
 frac[24,113] <= 1
 frac[192,13] <= 1
 frac[195,8] <= 1
 frac[107,154] <= 1
 frac[90,254] <= 1
 frac[198,119] <= 1
 frac[18,60] <= 1
 frac[20,229] <= 1
 frac[149,12] <= 1
 frac[192,188] <= 1
 frac[193,34] <= 1
 frac[196,41] <= 1
 frac[92,202] <= 1
 frac[39,24] <= 1
 frac[116,192] <= 1
 frac[8,201] <= 1
 frac[101,148] <= 1
 frac[193,211] <= 1
 frac[174,111] <= 1
 frac[194,195] <= 1
 frac[195,211] <= 1
 frac[196,184] <= 1
 frac[197,54] <= 1
 frac[177,185] <= 1
 frac[142,148] <= 1
 frac[129,225] <= 1
 frac[150,212] <= 1
 frac[133,100] <= 1
 frac[154,155] <= 1
 frac[194,114] <= 1
 frac[195,96] <= 1
 frac[178,152] <= 1
 frac[196,207] <= 1
 frac[197,199] <= 1
 frac[198,223] <= 1
 frac[15,64] <= 1
 frac[106,240] <= 1
 frac[60,148] <= 1
 frac[81,121] <= 1
 frac[133,245] <= 1
 frac[192,4] <= 1
 frac[156,58] <= 1
 frac[198,110] <= 1
 frac[127,92] <= 1
 frac[164,170] <= 1
 frac[182,212] <= 1
 frac[0,225] <= 1
 frac[189,190] <= 1
 frac[26,151] <= 1
 frac[193,59] <= 1
 frac[194,171] <= 1
 frac[196,16] <= 1
 frac[165,181] <= 1
 frac[167,220] <= 1
 frac[152,150] <= 1
 frac[26,6] <= 1
 frac[118,255] <= 1
 frac[10,220] <= 1
 frac[193,212] <= 1
 frac[7,104] <= 1
 frac[195,216] <= 1
 frac[196,167] <= 1
 frac[112,4] <= 1
 frac[64,167] <= 1
 frac[119,148] <= 1
 frac[194,77] <= 1
 frac[105,235] <= 1
 frac[178,147] <= 1
 frac[197,192] <= 1
 frac[198,214] <= 1
 frac[184,203] <= 1
 frac[40,176] <= 1
 frac[133,254] <= 1
 frac[45,12] <= 1
 frac[192,27] <= 1
 frac[102,71] <= 1
 frac[66,179] <= 1
 frac[173,87] <= 1
 frac[158,142] <= 1
 frac[113,244] <= 1
 frac[132,58] <= 1
 frac[192,138] <= 1
 frac[193,60] <= 1
 frac[194,162] <= 1
 frac[157,46] <= 1
 frac[68,194] <= 1
 frac[196,31] <= 1
 frac[51,138] <= 1
 frac[70,175] <= 1
 frac[2,135] <= 1
 frac[99,182] <= 1
 frac[100,8] <= 1
 frac[171,158] <= 1
 frac[193,173] <= 1
 frac[194,213] <= 1
 frac[176,35] <= 1
 frac[195,193] <= 1
 frac[196,174] <= 1
 frac[197,56] <= 1
 frac[198,190] <= 1
 frac[175,117] <= 1
 frac[54,4] <= 1
 frac[16,119] <= 1
 frac[147,89] <= 1
 frac[59,203] <= 1
 frac[167,170] <= 1
 frac[190,25] <= 1
 frac[7,246] <= 1
 frac[119,109] <= 1
 frac[194,68] <= 1
 frac[195,110] <= 1
 frac[197,201] <= 1
 frac[198,201] <= 1
 frac[163,121] <= 1
 frac[81,107] <= 1
 frac[61,223] <= 1
 frac[192,18] <= 1
 frac[120,58] <= 1
 frac[102,126] <= 1
 frac[157,182] <= 1
 frac[173,80] <= 1
 frac[89,219] <= 1
 frac[198,88] <= 1
 frac[74,90] <= 1
 frac[43,51] <= 1
 frac[99,46] <= 1
 frac[171,6] <= 1
 frac[192,129] <= 1
 frac[193,53] <= 1
 frac[194,189] <= 1
 frac[67,185] <= 1
 frac[33,121] <= 1
 frac[196,6] <= 1
 frac[147,241] <= 1
 frac[173,167] <= 1
 frac[193,166] <= 1
 frac[194,44] <= 1
 frac[195,198] <= 1
 frac[196,149] <= 1
 frac[197,33] <= 1
 frac[198,177] <= 1
 frac[130,250] <= 1
 frac[194,95] <= 1
 frac[174,243] <= 1
 frac[195,87] <= 1
 frac[197,178] <= 1
 frac[198,192] <= 1
 frac[56,36] <= 1
 frac[39,222] <= 1
 frac[60,161] <= 1
 frac[62,222] <= 1
 frac[192,105] <= 1
 frac[84,47] <= 1
 frac[102,113] <= 1
 frac[108,213] <= 1
 frac[198,83] <= 1
 frac[192,152] <= 1
 frac[45,143] <= 1
 frac[46,231] <= 1
 frac[193,14] <= 1
 frac[32,140] <= 1
 frac[196,13] <= 1
 frac[22,92] <= 1
 frac[170,189] <= 1
 frac[79,101] <= 1
 frac[100,62] <= 1
 frac[83,94] <= 1
 frac[194,39] <= 1
 frac[195,207] <= 1
 frac[196,156] <= 1
 frac[34,211] <= 1
 frac[198,168] <= 1
 frac[197,42] <= 1
 frac[17,105] <= 1
 frac[9,74] <= 1
 frac[194,86] <= 1
 frac[177,54] <= 1
 frac[195,92] <= 1
 frac[12,241] <= 1
 frac[197,187] <= 1
 frac[198,59] <= 1
 frac[133,233] <= 1
 frac[136,67] <= 1
 frac[192,96] <= 1
 frac[139,254] <= 1
 frac[68,168] <= 1
 frac[158,227] <= 1
 frac[87,203] <= 1
 frac[32,20] <= 1
 frac[15,213] <= 1
 frac[198,74] <= 1
 frac[141,153] <= 1
 frac[75,86] <= 1
 frac[4,90] <= 1
 frac[169,157] <= 1
 frac[171,116] <= 1
 frac[192,159] <= 1
 frac[193,7] <= 1
 frac[194,143] <= 1
 frac[121,47] <= 1
 frac[196,116] <= 1
 frac[107,12] <= 1
 frac[143,250] <= 1
 frac[97,22] <= 1
 frac[150,24] <= 1
 frac[155,79] <= 1
 frac[46,89] <= 1
 frac[47,85] <= 1
 frac[194,62] <= 1
 frac[12,25] <= 1
 frac[196,155] <= 1
 frac[197,19] <= 1
 frac[101,177] <= 1
 frac[195,180] <= 1
 frac[184,120] <= 1
 frac[60,192] <= 1
 frac[194,81] <= 1
 frac[195,69] <= 1
 frac[197,164] <= 1
 frac[198,50] <= 1
 frac[126,26] <= 1
 frac[79,128] <= 1
 frac[192,103] <= 1
 frac[193,111] <= 1
 frac[63,218] <= 1
 frac[32,43] <= 1
 frac[31,45] <= 1
 frac[141,146] <= 1
 frac[198,77] <= 1
 frac[113,144] <= 1
 frac[97,190] <= 1
 frac[132,30] <= 1
 frac[43,38] <= 1
 frac[26,242] <= 1
 frac[23,146] <= 1
 frac[192,150] <= 1
 frac[193,24] <= 1
 frac[194,134] <= 1
 frac[190,219] <= 1
 frac[196,115] <= 1
 frac[53,65] <= 1
 frac[92,188] <= 1
 frac[93,42] <= 1
 frac[21,104] <= 1
 frac[190,170] <= 1
 frac[154,94] <= 1
 frac[193,137] <= 1
 frac[194,57] <= 1
 frac[195,189] <= 1
 frac[196,130] <= 1
 frac[198,154] <= 1
 frac[73,6] <= 1
 frac[36,158] <= 1
 frac[165,35] <= 1
 frac[94,91] <= 1
 frac[93,91] <= 1
 frac[27,14] <= 1
 frac[172,155] <= 1
 frac[139,159] <= 1
 frac[175,254] <= 1
 frac[106,198] <= 1
 frac[88,16] <= 1
 frac[197,173] <= 1
 frac[198,53] <= 1
 frac[195,74] <= 1
 frac[77,18] <= 1
 frac[116,19] <= 1
 frac[133,219] <= 1
 frac[79,137] <= 1
 frac[192,126] <= 1
 frac[193,96] <= 1
 frac[153,15] <= 1
 frac[35,47] <= 1
 frac[198,68] <= 1
 frac[40,34] <= 1
 frac[96,207] <= 1
 frac[97,183] <= 1
 frac[38,72] <= 1
 frac[193,17] <= 1
 frac[194,129] <= 1
 frac[196,122] <= 1
 frac[197,116] <= 1
 frac[70,130] <= 1
 frac[93,35] <= 1
 frac[129,167] <= 1
 frac[9,236] <= 1
 frac[193,130] <= 1
 frac[29,241] <= 1
 frac[195,162] <= 1
 frac[104,174] <= 1
 frac[194,48] <= 1
 frac[196,137] <= 1
 frac[198,157] <= 1
 frac[108,27] <= 1
 frac[197,5] <= 1
 frac[94,66] <= 1
 frac[136,233] <= 1
 frac[195,51] <= 1
 frac[14,206] <= 1
 frac[197,150] <= 1
 frac[198,44] <= 1
 frac[164,148] <= 1
 frac[96,87] <= 1
 frac[59,93] <= 1
 frac[192,117] <= 1
 frac[48,202] <= 1
 frac[193,121] <= 1
 frac[32,57] <= 1
 frac[1,196] <= 1
 frac[165,243] <= 1
 frac[164,251] <= 1
 frac[132,12] <= 1
 frac[194,152] <= 1
 frac[50,254] <= 1
 frac[197,125] <= 1
 frac[144,7] <= 1
 frac[56,233] <= 1
 frac[129,184] <= 1
 frac[57,144] <= 1
 frac[95,135] <= 1
 frac[97,57] <= 1
 frac[83,122] <= 1
 frac[193,155] <= 1
 frac[194,11] <= 1
 frac[88,111] <= 1
 frac[196,240] <= 1
 frac[197,14] <= 1
 frac[198,148] <= 1
 frac[53,196] <= 1
 frac[42,87] <= 1
 frac[187,40] <= 1
 frac[156,243] <= 1
 frac[102,133] <= 1
 frac[195,56] <= 1
 frac[197,159] <= 1
 frac[198,39] <= 1
 frac[183,92] <= 1
 frac[39,251] <= 1
 frac[21,161] <= 1
 frac[45,91] <= 1
 frac[192,76] <= 1
 frac[193,114] <= 1
 frac[65,143] <= 1
 frac[160,101] <= 1
 frac[87,239] <= 1
 frac[144,239] <= 1
 frac[53,43] <= 1
 frac[146,72] <= 1
 frac[113,187] <= 1
 frac[169,249] <= 1
 frac[187,246] <= 1
 frac[99,120] <= 1
 frac[192,251] <= 1
 frac[193,227] <= 1
 frac[194,147] <= 1
 frac[48,80] <= 1
 frac[140,123] <= 1
 frac[196,104] <= 1
 frac[197,102] <= 1
 frac[53,92] <= 1
 frac[71,6] <= 1
 frac[131,181] <= 1
 frac[153,192] <= 1
 frac[193,156] <= 1
 frac[194,2] <= 1
 frac[195,144] <= 1
 frac[31,227] <= 1
 frac[196,255] <= 1
 frac[197,247] <= 1
 frac[198,143] <= 1
 frac[15,16] <= 1
 frac[6,104] <= 1
 frac[9,111] <= 1
 frac[10,103] <= 1
 frac[195,33] <= 1
 frac[14,216] <= 1
 frac[197,152] <= 1
 frac[198,30] <= 1
 frac[181,254] <= 1
 frac[34,97] <= 1
 frac[113,35] <= 1
 frac[192,67] <= 1
 frac[193,75] <= 1
 frac[84,81] <= 1
 frac[163,153] <= 1
 frac[185,100] <= 1
 frac[43,2] <= 1
 frac[172,89] <= 1
 frac[192,242] <= 1
 frac[193,228] <= 1
 frac[49,226] <= 1
 frac[196,87] <= 1
 frac[197,111] <= 1
 frac[52,252] <= 1
 frac[182,6] <= 1
 frac[10,159] <= 1
 frac[195,153] <= 1
 frac[196,230] <= 1
 frac[197,240] <= 1
 frac[34,9] <= 1
 frac[145,10] <= 1
 frac[179,166] <= 1
 frac[37,12] <= 1
 frac[198,134] <= 1
 frac[192,43] <= 1
 frac[83,249] <= 1
 frac[195,38] <= 1
 frac[31,121] <= 1
 frac[197,129] <= 1
 frac[198,17] <= 1
 frac[91,73] <= 1
 frac[192,90] <= 1
 frac[28,11] <= 1
 frac[121,100] <= 1
 frac[142,7] <= 1
 frac[20,170] <= 1
 frac[166,185] <= 1
 frac[116,230] <= 1
 frac[118,134] <= 1
 frac[192,201] <= 1
 frac[193,253] <= 1
 frac[194,229] <= 1
 frac[195,241] <= 1
 frac[48,78] <= 1
 frac[196,94] <= 1
 frac[179,62] <= 1
 frac[175,69] <= 1
 frac[73,90] <= 1
 frac[197,104] <= 1
 frac[95,242] <= 1
 frac[9,8] <= 1
 frac[155,57] <= 1
 frac[194,20] <= 1
 frac[195,158] <= 1
 frac[197,249] <= 1
 frac[198,249] <= 1
 frac[61,175] <= 1
 frac[6,90] <= 1
 frac[192,34] <= 1
 frac[120,10] <= 1
 frac[195,47] <= 1
 frac[197,138] <= 1
 frac[198,8] <= 1
 frac[91,78] <= 1
 frac[146,34] <= 1
 frac[184,145] <= 1
 frac[43,227] <= 1
 frac[117,104] <= 1
 frac[99,30] <= 1
 frac[192,81] <= 1
 frac[193,69] <= 1
 frac[196,54] <= 1
 frac[53,6] <= 1
 frac[110,73] <= 1
 frac[182,97] <= 1
 frac[115,190] <= 1
 frac[192,192] <= 1
 frac[193,246] <= 1
 frac[194,252] <= 1
 frac[195,246] <= 1
 frac[105,116] <= 1
 frac[196,69] <= 1
 frac[197,81] <= 1
 frac[48,69] <= 1
 frac[144,99] <= 1
 frac[179,7] <= 1
 frac[93,0] <= 1
 frac[102,214] <= 1
 frac[194,111] <= 1
 frac[176,153] <= 1
 frac[195,135] <= 1
 frac[196,212] <= 1
 frac[52,113] <= 1
 frac[198,240] <= 1
 frac[197,226] <= 1
 frac[56,116] <= 1
 frac[42,75] <= 1
 frac[60,241] <= 1
 frac[192,57] <= 1
 frac[195,20] <= 1
 frac[72,148] <= 1
 frac[126,43] <= 1
 frac[146,29] <= 1
 frac[74,53] <= 1
 frac[59,126] <= 1
 frac[82,53] <= 1
 frac[65,107] <= 1
 frac[193,94] <= 1
 frac[192,168] <= 1
 frac[196,61] <= 1
 frac[141,161] <= 1
 frac[181,122] <= 1
 frac[1,57] <= 1
 frac[165,144] <= 1
 frac[21,54] <= 1
 frac[95,83] <= 1
 frac[167,59] <= 1
 frac[171,188] <= 1
 frac[118,144] <= 1
 frac[195,255] <= 1
 frac[196,76] <= 1
 frac[197,90] <= 1
 frac[144,90] <= 1
 frac[147,55] <= 1
 frac[174,75] <= 1
 frac[154,143] <= 1
 frac[194,102] <= 1
 frac[195,140] <= 1
 frac[196,211] <= 1
 frac[197,235] <= 1
 frac[198,235] <= 1
 frac[41,188] <= 1
 frac[98,55] <= 1
 frac[192,48] <= 1
 frac[10,27] <= 1
 frac[102,88] <= 1
 frac[195,29] <= 1
 frac[160,41] <= 1
 frac[198,122] <= 1
 frac[26,187] <= 1
 frac[193,87] <= 1
 frac[196,36] <= 1
 frac[192,222] <= 1
 frac[193,192] <= 1
 frac[194,206] <= 1
 frac[50,148] <= 1
 frac[196,75] <= 1
 frac[197,67] <= 1
 frac[129,150] <= 1
 frac[41,20] <= 1
 frac[95,233] <= 1
 frac[61,73] <= 1
 frac[153,165] <= 1
 frac[172,204] <= 1
 frac[100,228] <= 1
 frac[156,182] <= 1
 frac[194,97] <= 1
 frac[175,193] <= 1
 frac[88,89] <= 1
 frac[51,79] <= 1
 frac[197,212] <= 1
 frac[198,226] <= 1
 frac[109,114] <= 1
 frac[195,117] <= 1
 frac[112,111] <= 1
 frac[124,226] <= 1
 frac[196,218] <= 1
 frac[188,76] <= 1
 frac[192,55] <= 1
 frac[101,67] <= 1
 frac[135,162] <= 1
 frac[195,2] <= 1
 frac[109,3] <= 1
 frac[39,141] <= 1
 frac[21,215] <= 1
 frac[26,130] <= 1
 frac[44,216] <= 1
 frac[192,166] <= 1
 frac[28,111] <= 1
 frac[62,139] <= 1
 frac[193,40] <= 1
 frac[196,35] <= 1
 frac[14,46] <= 1
 frac[181,76] <= 1
 frac[91,194] <= 1
 frac[129,126] <= 1
 frac[25,189] <= 1
 frac[27,161] <= 1
 frac[192,213] <= 1
 frac[193,217] <= 1
 frac[194,201] <= 1
 frac[195,237] <= 1
 frac[118,226] <= 1
 frac[197,76] <= 1
 frac[59,167] <= 1
 frac[11,207] <= 1
 frac[195,122] <= 1
 frac[194,120] <= 1
 frac[196,193] <= 1
 frac[197,221] <= 1
 frac[198,229] <= 1
 frac[34,36] <= 1
 frac[25,100] <= 1
 frac[134,11] <= 1
 frac[192,14] <= 1
 frac[173,100] <= 1
 frac[138,198] <= 1
 frac[195,11] <= 1
 frac[31,90] <= 1
 frac[107,157] <= 1
 frac[103,231] <= 1
 frac[126,92] <= 1
 frac[198,116] <= 1
 frac[17,181] <= 1
 frac[38,56] <= 1
 frac[8,123] <= 1
 frac[64,72] <= 1
 frac[192,189] <= 1
 frac[193,33] <= 1
 frac[63,0] <= 1
 frac[136,48] <= 1
 frac[196,42] <= 1
 frac[163,163] <= 1
 frac[146,153] <= 1
 frac[57,95] <= 1
 frac[94,227] <= 1
 frac[167,38] <= 1
 frac[44,102] <= 1
 frac[9,220] <= 1
 frac[194,192] <= 1
 frac[195,210] <= 1
 frac[196,185] <= 1
 frac[197,53] <= 1
 frac[148,68] <= 1
 frac[153,183] <= 1
 frac[173,204] <= 1
 frac[194,115] <= 1
 frac[195,99] <= 1
 frac[105,225] <= 1
 frac[123,91] <= 1
 frac[197,198] <= 1
 frac[108,90] <= 1
 frac[198,220] <= 1
 frac[35,183] <= 1
 frac[15,67] <= 1
 frac[185,185] <= 1
 frac[56,16] <= 1
 frac[112,157] <= 1
 frac[192,5] <= 1
 frac[120,45] <= 1
 frac[140,221] <= 1
 frac[158,132] <= 1
 frac[87,38] <= 1
 frac[160,62] <= 1
 frac[198,111] <= 1
 frac[74,41] <= 1
 frac[192,180] <= 1
 frac[193,58] <= 1
 frac[194,168] <= 1
 frac[157,84] <= 1
 frac[124,41] <= 1
 frac[196,17] <= 1
 frac[72,31] <= 1
 frac[163,168] <= 1
 frac[74,184] <= 1
 frac[129,72] <= 1
 frac[148,220] <= 1
 frac[150,125] <= 1
 frac[44,93] <= 1
 frac[193,171] <= 1
 frac[194,219] <= 1
 frac[195,219] <= 1
 frac[123,227] <= 1
 frac[196,160] <= 1
 frac[197,62] <= 1
 frac[71,206] <= 1
 frac[115,16] <= 1
 frac[64,184] <= 1
 frac[102,245] <= 1
 frac[194,74] <= 1
 frac[195,104] <= 1
 frac[197,207] <= 1
 frac[198,215] <= 1
 frac[126,255] <= 1
 frac[34,54] <= 1
 frac[162,99] <= 1
 frac[131,26] <= 1
 frac[60,172] <= 1
 frac[192,28] <= 1
 frac[159,7] <= 1
 frac[89,209] <= 1
 frac[198,102] <= 1
 frac[126,78] <= 1
 frac[91,184] <= 1
 frac[164,210] <= 1
 frac[111,181] <= 1
 frac[96,45] <= 1
 frac[97,145] <= 1
 frac[65,129] <= 1
 frac[193,51] <= 1
 frac[194,163] <= 1
 frac[196,24] <= 1
 frac[58,89] <= 1
 frac[115,104] <= 1
 frac[193,172] <= 1
 frac[194,210] <= 1
 frac[195,192] <= 1
 frac[140,58] <= 1
 frac[196,175] <= 1
 frac[197,39] <= 1
 frac[198,191] <= 1
 frac[15,32] <= 1
 frac[29,19] <= 1
 frac[22,50] <= 1
 frac[172,176] <= 1
 frac[194,69] <= 1
 frac[195,81] <= 1
 frac[123,105] <= 1
 frac[125,240] <= 1
 frac[198,206] <= 1
 frac[58,187] <= 1
 frac[42,17] <= 1
 frac[192,19] <= 1
 frac[137,227] <= 1
 frac[138,243] <= 1
 frac[67,43] <= 1
 frac[72,190] <= 1
 frac[198,89] <= 1
 frac[185,52] <= 1
 frac[39,177] <= 1
 frac[116,175] <= 1
 frac[135,23] <= 1
 frac[192,130] <= 1
 frac[193,52] <= 1
 frac[194,186] <= 1
 frac[196,7] <= 1
 frac[92,208] <= 1
 frac[57,114] <= 1
 frac[129,90] <= 1
 frac[5,242] <= 1
 frac[193,165] <= 1
 frac[194,45] <= 1
 frac[195,201] <= 1
 frac[68,89] <= 1
 frac[196,150] <= 1
 frac[197,32] <= 1
 frac[198,182] <= 1
 frac[2,14] <= 1
 frac[131,251] <= 1
 frac[150,254] <= 1
 frac[194,92] <= 1
 frac[174,240] <= 1
 frac[195,86] <= 1
 frac[31,169] <= 1
 frac[197,177] <= 1
 frac[52,34] <= 1
 frac[18,142] <= 1
 frac[198,193] <= 1
 frac[37,205] <= 1
 frac[159,100] <= 1
 frac[168,116] <= 1
 frac[42,24] <= 1
 frac[60,162] <= 1
 frac[63,215] <= 1
 frac[138,250] <= 1
 frac[14,98] <= 1
 frac[198,80] <= 1
 frac[186,181] <= 1
 frac[60,17] <= 1
 frac[117,32] <= 1
 frac[192,153] <= 1
 frac[194,181] <= 1
 frac[33,113] <= 1
 frac[196,14] <= 1
 frac[72,52] <= 1
 frac[90,66] <= 1
 frac[130,67] <= 1
 frac[114,124] <= 1
 frac[79,100] <= 1
 frac[193,190] <= 1
 frac[194,36] <= 1
 frac[175,122] <= 1
 frac[195,206] <= 1
 frac[196,157] <= 1
 frac[197,41] <= 1
 frac[198,169] <= 1
 frac[77,142] <= 1
 frac[190,8] <= 1
 frac[9,73] <= 1
 frac[29,150] <= 1
 frac[194,87] <= 1
 frac[31,150] <= 1
 frac[195,95] <= 1
 frac[197,186] <= 1
 frac[198,56] <= 1
 frac[181,144] <= 1
 frac[113,13] <= 1
 frac[96,67] <= 1
 frac[166,17] <= 1
 frac[133,232] <= 1
 frac[79,186] <= 1
 frac[192,97] <= 1
 frac[101,105] <= 1
 frac[89,36] <= 1
 frac[198,75] <= 1
 frac[65,83] <= 1
 frac[193,6] <= 1
 frac[194,140] <= 1
 frac[67,166] <= 1
 frac[192,144] <= 1
 frac[196,117] <= 1
 frac[68,56] <= 1
 frac[92,166] <= 1
 frac[93,208] <= 1
 frac[189,50] <= 1
 frac[44,65] <= 1
 frac[99,172] <= 1
 frac[193,183] <= 1
 frac[28,240] <= 1
 frac[195,183] <= 1
 frac[196,132] <= 1
 frac[197,18] <= 1
 frac[198,160] <= 1
 frac[75,183] <= 1
 frac[130,205] <= 1
 frac[79,18] <= 1
 frac[195,68] <= 1
 frac[197,163] <= 1
 frac[198,51] <= 1
 frac[17,242] <= 1
 frac[36,231] <= 1
 frac[54,153] <= 1
 frac[64,51] <= 1
 frac[192,120] <= 1
 frac[193,110] <= 1
 frac[85,168] <= 1
 frac[31,44] <= 1
 frac[107,119] <= 1
 frac[198,66] <= 1
 frac[126,106] <= 1
 frac[113,151] <= 1
 frac[96,201] <= 1
 frac[167,107] <= 1
 frac[190,216] <= 1
 frac[192,151] <= 1
 frac[193,31] <= 1
 frac[104,25] <= 1
 frac[29,70] <= 1
 frac[194,135] <= 1
 frac[196,124] <= 1
 frac[90,84] <= 1
 frac[75,223] <= 1
 frac[170,172] <= 1
 frac[79,106] <= 1
 frac[118,211] <= 1
 frac[193,136] <= 1
 frac[29,247] <= 1
 frac[195,188] <= 1
 frac[196,131] <= 1
 frac[52,136] <= 1
 frac[198,155] <= 1
 frac[197,27] <= 1
 frac[2,27] <= 1
 frac[59,214] <= 1
 frac[60,216] <= 1
 frac[150,131] <= 1
 frac[139,158] <= 1
 frac[195,77] <= 1
 frac[88,17] <= 1
 frac[106,199] <= 1
 frac[197,172] <= 1
 frac[198,42] <= 1
 frac[162,158] <= 1
 frac[180,244] <= 1
 frac[184,239] <= 1
 frac[185,231] <= 1
 frac[96,81] <= 1
 frac[23,53] <= 1
 frac[192,127] <= 1
 frac[193,103] <= 1
 frac[158,210] <= 1
 frac[88,128] <= 1
 frac[198,69] <= 1
 frac[35,46] <= 1
 frac[183,254] <= 1
 frac[169,142] <= 1
 frac[192,238] <= 1
 frac[193,16] <= 1
 frac[194,158] <= 1
 frac[68,46] <= 1
 frac[196,123] <= 1
 frac[33,4] <= 1
 frac[197,115] <= 1
 frac[32,146] <= 1
 frac[56,227] <= 1
 frac[96,191] <= 1
 frac[95,153] <= 1
 frac[168,135] <= 1
 frac[169,31] <= 1
 frac[190,82] <= 1
 frac[191,255] <= 1
 frac[194,49] <= 1
 frac[195,165] <= 1
 frac[31,252] <= 1
 frac[193,129] <= 1
 frac[197,4] <= 1
 frac[30,230] <= 1
 frac[198,146] <= 1
 frac[196,138] <= 1
 frac[178,215] <= 1
 frac[27,6] <= 1
 frac[157,225] <= 1
 frac[195,50] <= 1
 frac[89,142] <= 1
 frac[197,149] <= 1
 frac[198,45] <= 1
 frac[168,80] <= 1
 frac[60,70] <= 1
 frac[136,121] <= 1
 frac[193,120] <= 1
 frac[102,18] <= 1
 frac[49,102] <= 1
 frac[90,183] <= 1
 frac[143,163] <= 1
 frac[39,82] <= 1
 frac[166,149] <= 1
 frac[42,143] <= 1
 frac[192,229] <= 1
 frac[193,233] <= 1
 frac[194,153] <= 1
 frac[49,247] <= 1
 frac[45,242] <= 1
 frac[196,98] <= 1
 frac[197,124] <= 1
 frac[48,90] <= 1
 frac[171,235] <= 1
 frac[193,154] <= 1
 frac[176,254] <= 1
 frac[194,8] <= 1
 frac[195,170] <= 1
 frac[196,241] <= 1
 frac[197,13] <= 1
 frac[198,149] <= 1
 frac[30,233] <= 1
 frac[42,84] <= 1
 frac[60,238] <= 1
 frac[170,33] <= 1
 frac[49,14] <= 1
 frac[195,59] <= 1
 frac[197,158] <= 1
 frac[198,36] <= 1
 frac[182,226] <= 1
 frac[21,160] <= 1
 frac[23,75] <= 1
 frac[62,226] <= 1
 frac[192,77] <= 1
 frac[193,113] <= 1
 frac[45,90] <= 1
 frac[46,152] <= 1
 frac[89,0] <= 1
 frac[159,214] <= 1
 frac[146,73] <= 1
 frac[192,252] <= 1
 frac[193,226] <= 1
 frac[194,144] <= 1
 frac[11,39] <= 1
 frac[140,116] <= 1
 frac[196,105] <= 1
 frac[197,101] <= 1
 frac[74,240] <= 1
 frac[80,23] <= 1
 frac[26,127] <= 1
 frac[193,147] <= 1
 frac[194,3] <= 1
 frac[195,147] <= 1
 frac[88,119] <= 1
 frac[196,248] <= 1
 frac[197,246] <= 1
 frac[198,140] <= 1
 frac[34,15] <= 1
 frac[71,150] <= 1
 frac[106,161] <= 1
 frac[35,231] <= 1
 frac[187,32] <= 1
 frac[137,161] <= 1
 frac[195,32] <= 1
 frac[103,204] <= 1
 frac[197,135] <= 1
 frac[198,31] <= 1
 frac[112,220] <= 1
 frac[25,46] <= 1
 frac[192,68] <= 1
 frac[193,74] <= 1
 frac[106,67] <= 1
 frac[18,119] <= 1
 frac[182,148] <= 1
 frac[129,24] <= 1
 frac[1,213] <= 1
 frac[22,152] <= 1
 frac[192,243] <= 1
 frac[193,251] <= 1
 frac[194,235] <= 1
 frac[32,167] <= 1
 frac[196,80] <= 1
 frac[197,110] <= 1
 frac[2,76] <= 1
 frac[115,32] <= 1
 frac[193,148] <= 1
 frac[194,26] <= 1
 frac[195,152] <= 1
 frac[196,231] <= 1
 frac[197,255] <= 1
 frac[108,1] <= 1
 frac[198,135] <= 1
 frac[148,2] <= 1
 frac[192,44] <= 1
 frac[156,194] <= 1
 frac[195,41] <= 1
 frac[197,128] <= 1
 frac[198,22] <= 1
 frac[113,219] <= 1
 frac[114,171] <= 1
 frac[192,91] <= 1
 frac[193,67] <= 1
 frac[67,227] <= 1
 frac[1,46] <= 1
 frac[192,202] <= 1
 frac[193,252] <= 1
 frac[194,226] <= 1
 frac[195,240] <= 1
 frac[196,95] <= 1
 frac[197,87] <= 1
 frac[73,89] <= 1
 frac[182,30] <= 1
 frac[4,188] <= 1
 frac[119,188] <= 1
 frac[194,21] <= 1
 frac[195,129] <= 1
 frac[177,235] <= 1
 frac[196,238] <= 1
 frac[197,248] <= 1
 frac[198,254] <= 1
 frac[59,11] <= 1
 frac[187,62] <= 1
 frac[192,35] <= 1
 frac[101,87] <= 1
 frac[66,91] <= 1
 frac[67,91] <= 1
 frac[195,46] <= 1
 frac[197,137] <= 1
 frac[15,150] <= 1
 frac[198,9] <= 1
 frac[92,113] <= 1
 frac[82,59] <= 1
 frac[192,82] <= 1
 frac[193,68] <= 1
 frac[66,202] <= 1
 frac[50,16] <= 1
 frac[196,55] <= 1
 frac[72,253] <= 1
 frac[91,254] <= 1
 frac[29,160] <= 1
 frac[78,55] <= 1
 frac[192,193] <= 1
 frac[193,245] <= 1
 frac[65,2] <= 1
 frac[195,249] <= 1
 frac[196,70] <= 1
 frac[124,126] <= 1
 frac[197,80] <= 1
 frac[77,197] <= 1
 frac[169,124] <= 1
 frac[117,137] <= 1
 frac[9,48] <= 1
 frac[10,142] <= 1
 frac[194,108] <= 1
 frac[195,134] <= 1
 frac[160,164] <= 1
 frac[196,213] <= 1
 frac[197,225] <= 1
 frac[198,241] <= 1
 frac[18,190] <= 1
 frac[36,161] <= 1
 frac[179,183] <= 1
 frac[61,167] <= 1
 frac[192,58] <= 1
 frac[65,249] <= 1
 frac[120,18] <= 1
 frac[157,222] <= 1
 frac[105,149] <= 1
 frac[175,163] <= 1
 frac[89,243] <= 1
 frac[49,42] <= 1
 frac[35,107] <= 1
 frac[158,169] <= 1
 frac[37,142] <= 1
 frac[168,51] <= 1
 frac[116,70] <= 1
 frac[117,112] <= 1
 frac[192,169] <= 1
 frac[193,93] <= 1
 frac[140,137] <= 1
 frac[196,62] <= 1
 frac[69,253] <= 1
 frac[32,93] <= 1
 frac[73,250] <= 1
 frac[163,143] <= 1
 frac[164,7] <= 1
 frac[117,225] <= 1
 frac[192,216] <= 1
 frac[193,206] <= 1
 frac[194,244] <= 1
 frac[195,254] <= 1
 frac[196,77] <= 1
 frac[197,89] <= 1
 frac[52,202] <= 1
 frac[143,61] <= 1
 frac[36,89] <= 1
 frac[54,39] <= 1
 frac[71,229] <= 1
 frac[133,119] <= 1
 frac[9,57] <= 1
 frac[12,34] <= 1
 frac[195,143] <= 1
 frac[196,220] <= 1
 frac[198,232] <= 1
 frac[55,220] <= 1
 frac[94,41] <= 1
 frac[5,77] <= 1
 frac[61,160] <= 1
 frac[192,49] <= 1
 frac[142,110] <= 1
 frac[198,123] <= 1
 frac[74,61] <= 1
 frac[40,101] <= 1
 frac[3,123] <= 1
 frac[24,148] <= 1
 frac[25,10] <= 1
 frac[192,160] <= 1
 frac[48,229] <= 1
 frac[63,29] <= 1
 frac[139,62] <= 1
 frac[193,86] <= 1
 frac[196,37] <= 1
 frac[26,43] <= 1
 frac[100,22] <= 1
 frac[192,223] <= 1
 frac[193,199] <= 1
 frac[29,14] <= 1
 frac[195,231] <= 1
 frac[194,207] <= 1
 frac[196,180] <= 1
 frac[197,66] <= 1
 frac[179,20] <= 1
 frac[53,184] <= 1
 frac[169,110] <= 1
 frac[5,37] <= 1
 frac[6,189] <= 1
 frac[99,237] <= 1
 frac[79,34] <= 1
 frac[194,126] <= 1
 frac[195,116] <= 1
 frac[174,210] <= 1
 frac[196,219] <= 1
 frac[197,211] <= 1
 frac[198,227] <= 1
 frac[103,152] <= 1
 frac[109,113] <= 1
 frac[108,101] <= 1
 frac[94,16] <= 1
 frac[58,164] <= 1
 frac[192,8] <= 1
 frac[10,3] <= 1
 frac[85,248] <= 1
 frac[103,233] <= 1
 frac[195,5] <= 1
 frac[161,149] <= 1
 frac[90,245] <= 1
 frac[198,114] <= 1
 frac[109,2] <= 1
 frac[18,195] <= 1
 frac[183,177] <= 1
 frac[40,108] <= 1
 frac[111,153] <= 1
 frac[117,66] <= 1
 frac[192,167] <= 1
 frac[193,47] <= 1
 frac[156,85] <= 1
 frac[196,44] <= 1
 frac[141,210] <= 1
 frac[90,100] <= 1
 frac[110,99] <= 1
 frac[130,101] <= 1
 frac[22,191] <= 1
 frac[189,27] <= 1
 frac[190,155] <= 1
 frac[193,216] <= 1
 frac[194,198] <= 1
 frac[33,204] <= 1
 frac[197,75] <= 1
 frac[144,73] <= 1
 frac[145,93] <= 1
 frac[196,179] <= 1
 frac[112,8] <= 1
 frac[77,240] <= 1
 frac[151,195] <= 1
 frac[7,133] <= 1
 frac[194,121] <= 1
 frac[195,125] <= 1
 frac[12,208] <= 1
 frac[196,194] <= 1
 frac[197,220] <= 1
 frac[36,222] <= 1
 frac[77,65] <= 1
 frac[131,47] <= 1
 frac[43,177] <= 1
 frac[134,8] <= 1
 frac[192,15] <= 1
 frac[156,61] <= 1
 frac[195,10] <= 1
 frac[105,136] <= 1
 frac[72,138] <= 1
 frac[198,117] <= 1
 frac[127,81] <= 1
 frac[146,7] <= 1
 frac[126,93] <= 1
 frac[166,66] <= 1
 frac[25,20] <= 1
 frac[192,190] <= 1
 frac[122,134] <= 1
 frac[194,174] <= 1
 frac[196,43] <= 1
 frac[186,1] <= 1
 frac[115,157] <= 1
 frac[188,170] <= 1
 frac[190,130] <= 1
 frac[155,108] <= 1
 frac[193,209] <= 1
 frac[195,213] <= 1
 frac[196,186] <= 1
 frac[197,52] <= 1
 frac[16,123] <= 1
 frac[53,138] <= 1
 frac[90,30] <= 1
 frac[77,233] <= 1
 frac[78,130] <= 1
 frac[82,244] <= 1
 frac[119,153] <= 1
 frac[194,112] <= 1
 frac[177,8] <= 1
 frac[195,98] <= 1
 frac[13,215] <= 1
 frac[197,197] <= 1
 frac[198,221] <= 1
 frac[196,201] <= 1
 frac[123,90] <= 1
 frac[0,116] <= 1
 frac[130,158] <= 1
 frac[187,83] <= 1
 frac[43,182] <= 1
 frac[133,243] <= 1
 frac[134,3] <= 1
 frac[192,6] <= 1
 frac[62,43] <= 1
 frac[189,62] <= 1
 frac[157,162] <= 1
 frac[70,27] <= 1
 frac[198,108] <= 1
 frac[19,244] <= 1
 frac[39,130] <= 1
 frac[167,73] <= 1
 frac[192,181] <= 1
 frac[193,57] <= 1
 frac[194,169] <= 1
 frac[47,194] <= 1
 frac[196,18] <= 1
 frac[130,119] <= 1
 frac[95,182] <= 1
 frac[193,170] <= 1
 frac[194,216] <= 1
 frac[30,153] <= 1
 frac[68,84] <= 1
 frac[124,153] <= 1
 frac[197,61] <= 1
 frac[196,161] <= 1
 frac[57,208] <= 1
 frac[131,140] <= 1
 frac[95,199] <= 1
 frac[152,7] <= 1
 frac[5,16] <= 1
 frac[104,213] <= 1
 frac[194,75] <= 1
 frac[13,208] <= 1
 frac[195,107] <= 1
 frac[16,129] <= 1
 frac[198,212] <= 1
 frac[17,21] <= 1
 frac[197,206] <= 1
 frac[18,157] <= 1
 frac[20,6] <= 1
 frac[130,153] <= 1
 frac[24,65] <= 1
 frac[159,6] <= 1
 frac[182,173] <= 1
 frac[39,187] <= 1
 frac[192,140] <= 1
 frac[65,79] <= 1
 frac[194,160] <= 1
 frac[193,50] <= 1
 frac[32,112] <= 1
 frac[196,25] <= 1
 frac[165,188] <= 1
 frac[8,249] <= 1
 frac[193,163] <= 1
 frac[194,211] <= 1
 frac[195,195] <= 1
 frac[105,65] <= 1
 frac[196,168] <= 1
 frac[197,38] <= 1
 frac[198,188] <= 1
 frac[143,30] <= 1
 frac[125,30] <= 1
 frac[2,52] <= 1
 frac[93,117] <= 1
 frac[169,74] <= 1
 frac[5,25] <= 1
 frac[79,6] <= 1
 frac[194,66] <= 1
 frac[12,253] <= 1
 frac[47,41] <= 1
 frac[195,80] <= 1
 frac[197,183] <= 1
 frac[198,207] <= 1
 frac[125,143] <= 1
 frac[58,184] <= 1
 frac[77,108] <= 1
 frac[25,126] <= 1
 frac[192,20] <= 1
 frac[85,156] <= 1
 frac[31,48] <= 1
 frac[198,94] <= 1
 frac[40,72] <= 1
 frac[23,129] <= 1
 frac[192,131] <= 1
 frac[193,11] <= 1
 frac[194,187] <= 1
 frac[139,83] <= 1
 frac[29,122] <= 1
 frac[196,0] <= 1
 frac[78,100] <= 1
 frac[189,63] <= 1
 frac[117,215] <= 1
 frac[193,164] <= 1
 frac[48,23] <= 1
 frac[195,200] <= 1
 frac[68,90] <= 1
 frac[196,151] <= 1
 frac[197,47] <= 1
 frac[198,183] <= 1
 frac[35,220] <= 1
 frac[194,42] <= 1
 frac[66,16] <= 1
 frac[194,93] <= 1
 frac[195,89] <= 1
 frac[197,176] <= 1
 frac[198,198] <= 1
 frac[18,143] <= 1
 frac[166,31] <= 1
 frac[187,102] <= 1
 frac[192,107] <= 1
 frac[140,203] <= 1
 frac[106,122] <= 1
 frac[14,99] <= 1
 frac[198,81] <= 1
 frac[35,58] <= 1
 frac[116,183] <= 1
 frac[99,89] <= 1
 frac[192,154] <= 1
 frac[193,12] <= 1
 frac[194,178] <= 1
 frac[196,15] <= 1
 frac[146,250] <= 1
 frac[93,214] <= 1
 frac[99,166] <= 1
 frac[9,255] <= 1
 frac[189,56] <= 1
 frac[193,189] <= 1
 frac[195,177] <= 1
 frac[33,225] <= 1
 frac[196,158] <= 1
 frac[198,174] <= 1
 frac[53,238] <= 1
 frac[197,40] <= 1
 frac[42,113] <= 1
 frac[132,243] <= 1
 frac[82,216] <= 1
 frac[173,47] <= 1
 frac[194,84] <= 1
 frac[195,94] <= 1
 frac[29,149] <= 1
 frac[197,185] <= 1
 frac[198,57] <= 1
 frac[35,82] <= 1
 frac[82,107] <= 1
 frac[192,98] <= 1
 frac[83,190] <= 1
 frac[139,240] <= 1
 frac[198,72] <= 1
 frac[110,158] <= 1
 frac[132,17] <= 1
 frac[192,145] <= 1
 frac[193,5] <= 1
 frac[194,141] <= 1
 frac[28,66] <= 1
 frac[164,95] <= 1
 frac[115,126] <= 1
 frac[170,162] <= 1
 frac[169,44] <= 1
 frac[99,175] <= 1
 frac[152,244] <= 1
 frac[173,151] <= 1
 frac[193,182] <= 1
 frac[47,87] <= 1
 frac[195,182] <= 1
 frac[194,60] <= 1
 frac[196,133] <= 1
 frac[197,17] <= 1
 frac[17,96] <= 1
 frac[198,161] <= 1
 frac[32,244] <= 1
 frac[165,56] <= 1
 frac[93,64] <= 1
 frac[133,79] <= 1
 frac[195,71] <= 1
 frac[103,171] <= 1
 frac[107,217] <= 1
 frac[197,162] <= 1
 frac[198,48] <= 1
 frac[108,182] <= 1
 frac[166,9] <= 1
 frac[77,23] <= 1
 frac[187,116] <= 1
 frac[81,92] <= 1
 frac[192,121] <= 1
 frac[191,19] <= 1
 frac[90,162] <= 1
 frac[198,67] <= 1
 frac[97,188] <= 1
 frac[22,253] <= 1
 frac[116,133] <= 1
 frac[192,232] <= 1
 frac[63,85] <= 1
 frac[194,132] <= 1
 frac[193,30] <= 1
 frac[135,125] <= 1
 frac[107,7] <= 1
 frac[169,21] <= 1
 frac[133,39] <= 1
 frac[172,6] <= 1
 frac[193,143] <= 1
 frac[194,55] <= 1
 frac[195,191] <= 1
 frac[86,9] <= 1
 frac[31,246] <= 1
 frac[196,140] <= 1
 frac[198,152] <= 1
 frac[145,44] <= 1
 frac[54,230] <= 1
 frac[197,26] <= 1
 frac[165,33] <= 1
 frac[60,217] <= 1
 frac[195,76] <= 1
 frac[197,171] <= 1
 frac[198,43] <= 1
 frac[108,173] <= 1
 frac[56,59] <= 1
 frac[77,16] <= 1
 frac[135,229] <= 1
 frac[193,102] <= 1
 frac[192,112] <= 1
 frac[28,33] <= 1
 frac[165,200] <= 1
 frac[113,143] <= 1
 frac[114,23] <= 1
 frac[95,11] <= 1
 frac[132,7] <= 1
 frac[100,70] <= 1
 frac[192,239] <= 1
 frac[193,23] <= 1
 frac[194,159] <= 1
 frac[47,244] <= 1
 frac[67,151] <= 1
 frac[196,100] <= 1
 frac[197,114] <= 1
 frac[53,72] <= 1
 frac[73,188] <= 1
 frac[182,51] <= 1
 frac[2,96] <= 1
 frac[166,226] <= 1
 frac[95,152] <= 1
 frac[78,64] <= 1
 frac[61,24] <= 1
 frac[170,148] <= 1
 frac[133,32] <= 1
 frac[193,128] <= 1
 frac[194,14] <= 1
 frac[177,206] <= 1
 frac[160,130] <= 1
 frac[195,164] <= 1
 frac[197,3] <= 1
 frac[198,147] <= 1
 frac[196,139] <= 1
 frac[51,30] <= 1
 frac[5,102] <= 1
 frac[79,227] <= 1
 frac[101,50] <= 1
 frac[157,224] <= 1
 frac[195,53] <= 1
 frac[106,207] <= 1
 frac[103,217] <= 1
 frac[197,148] <= 1
 frac[198,34] <= 1
 frac[180,252] <= 1
 frac[110,192] <= 1
 frac[107,215] <= 1
 frac[185,239] <= 1
 frac[77,9] <= 1
 frac[134,208] <= 1
 frac[192,119] <= 1
 frac[193,127] <= 1
 frac[28,56] <= 1
 frac[142,48] <= 1
 frac[71,122] <= 1
 frac[2,200] <= 1
 frac[38,65] <= 1
 frac[57,37] <= 1
 frac[169,246] <= 1
 frac[117,3] <= 1
 frac[154,63] <= 1
 frac[192,230] <= 1
 frac[193,232] <= 1
 frac[194,150] <= 1
 frac[174,58] <= 1
 frac[196,99] <= 1
 frac[197,123] <= 1
 frac[51,246] <= 1
 frac[124,91] <= 1
 frac[54,65] <= 1
 frac[56,235] <= 1
 frac[165,66] <= 1
 frac[2,123] <= 1
 frac[81,165] <= 1
 frac[193,153] <= 1
 frac[195,173] <= 1
 frac[196,242] <= 1
 frac[197,12] <= 1
 frac[179,90] <= 1
 frac[17,75] <= 1
 frac[198,138] <= 1
 frac[41,93] <= 1
 frac[27,30] <= 1
 frac[118,93] <= 1
 frac[195,58] <= 1
 frac[103,214] <= 1
 frac[86,138] <= 1
 frac[197,157] <= 1
 frac[198,37] <= 1
 frac[125,165] <= 1
 frac[92,101] <= 1
 frac[3,25] <= 1
 frac[192,78] <= 1
 frac[193,112] <= 1
 frac[49,158] <= 1
 frac[50,100] <= 1
 frac[72,201] <= 1
 frac[181,100] <= 1
 frac[39,74] <= 1
 frac[152,161] <= 1
 frac[188,250] <= 1
 frac[192,253] <= 1
 frac[193,225] <= 1
 frac[194,145] <= 1
 frac[196,106] <= 1
 frac[197,100] <= 1
 frac[90,46] <= 1
 frac[73,174] <= 1
 frac[146,217] <= 1
 frac[58,23] <= 1
 frac[193,146] <= 1
 frac[28,213] <= 1
 frac[103,126] <= 1
 frac[88,120] <= 1
 frac[196,249] <= 1
 frac[197,245] <= 1
 frac[198,141] <= 1
 frac[52,102] <= 1
 frac[160,144] <= 1
 frac[194,0] <= 1
 frac[158,42] <= 1
 frac[139,132] <= 1
 frac[195,35] <= 1
 frac[197,134] <= 1
 frac[18,229] <= 1
 frac[134,194] <= 1
 frac[192,69] <= 1
 frac[193,73] <= 1
 frac[142,2] <= 1
 frac[166,132] <= 1
 frac[152,168] <= 1
 frac[44,142] <= 1
 frac[190,245] <= 1
 frac[192,244] <= 1
 frac[193,250] <= 1
 frac[194,232] <= 1
 frac[85,60] <= 1
 frac[140,108] <= 1
 frac[196,81] <= 1
 frac[197,109] <= 1
 frac[63,73] <= 1
 frac[174,183] <= 1
 frac[195,155] <= 1
 frac[194,27] <= 1
 frac[196,224] <= 1
 frac[197,254] <= 1
 frac[35,239] <= 1
 frac[198,132] <= 1
 frac[192,45] <= 1
 frac[102,181] <= 1
 frac[49,63] <= 1
 frac[195,40] <= 1
 frac[197,143] <= 1
 frac[198,23] <= 1
 frac[39,235] <= 1
 frac[192,92] <= 1
 frac[193,66] <= 1
 frac[53,59] <= 1
 frac[57,56] <= 1
 frac[93,148] <= 1
 frac[135,94] <= 1
 frac[193,243] <= 1
 frac[194,227] <= 1
 frac[195,243] <= 1
 frac[196,88] <= 1
 frac[197,86] <= 1
 frac[92,153] <= 1
 frac[151,190] <= 1
 frac[153,208] <= 1
 frac[66,41] <= 1
 frac[194,18] <= 1
 frac[102,221] <= 1
 frac[13,185] <= 1
 frac[196,239] <= 1
 frac[197,231] <= 1
 frac[198,255] <= 1
 frac[192,36] <= 1
 frac[49,48] <= 1
 frac[195,17] <= 1
 frac[197,136] <= 1
 frac[198,14] <= 1
 frac[0,1] <= 1
 frac[74,8] <= 1
 frac[127,188] <= 1
 frac[20,204] <= 1
 frac[37,132] <= 1
 frac[116,120] <= 1
 frac[26,183] <= 1
 frac[65,144] <= 1
 frac[193,91] <= 1
 frac[192,83] <= 1
 frac[196,48] <= 1
 frac[163,137] <= 1
 frac[40,7] <= 1
 frac[152,182] <= 1
 frac[116,239] <= 1
 frac[192,194] <= 1
 frac[193,244] <= 1
 frac[104,100] <= 1
 frac[195,248] <= 1
 frac[122,210] <= 1
 frac[196,71] <= 1
 frac[197,95] <= 1
 frac[194,250] <= 1
 frac[92,144] <= 1
 frac[167,141] <= 1
 frac[189,96] <= 1
 frac[27,77] <= 1
 frac[7,153] <= 1
 frac[194,109] <= 1
 frac[195,137] <= 1
 frac[12,36] <= 1
 frac[196,214] <= 1
 frac[197,224] <= 1
 frac[198,246] <= 1
 frac[174,193] <= 1
 frac[94,39] <= 1
 frac[151,36] <= 1
 frac[135,174] <= 1
 frac[192,59] <= 1
 frac[157,221] <= 1
 frac[30,85] <= 1
 frac[198,1] <= 1
 frac[18,206] <= 1
 frac[17,192] <= 1
 frac[55,119] <= 1
 frac[96,12] <= 1
 frac[116,71] <= 1
 frac[192,170] <= 1
 frac[193,92] <= 1
 frac[196,63] <= 1
 frac[57,74] <= 1
 frac[97,227] <= 1
 frac[171,190] <= 1
 frac[192,217] <= 1
 frac[193,205] <= 1
 frac[194,245] <= 1
 frac[195,225] <= 1
 frac[28,138] <= 1
 frac[196,78] <= 1
 frac[197,88] <= 1
 frac[121,229] <= 1
 frac[19,2] <= 1
 frac[37,84] <= 1
 frac[148,169] <= 1
 frac[78,174] <= 1
 frac[194,100] <= 1
 frac[195,142] <= 1
 frac[196,221] <= 1
 frac[197,233] <= 1
 frac[198,233] <= 1
 frac[152,91] <= 1
 frac[192,50] <= 1
 frac[63,143] <= 1
 frac[195,31] <= 1
 frac[88,195] <= 1
 frac[86,169] <= 1
 frac[70,55] <= 1
 frac[198,120] <= 1
 frac[19,224] <= 1
 frac[184,161] <= 1
 frac[21,204] <= 1
 frac[117,120] <= 1
 frac[120,137] <= 1
 frac[193,85] <= 1
 frac[192,161] <= 1
 frac[135,92] <= 1
 frac[46,188] <= 1
 frac[196,38] <= 1
 frac[33,89] <= 1
 frac[69,229] <= 1
 frac[14,37] <= 1
 frac[1,48] <= 1
 frac[93,255] <= 1
 frac[58,67] <= 1
 frac[115,142] <= 1
 frac[192,208] <= 1
 frac[193,198] <= 1
 frac[194,204] <= 1
 frac[123,222] <= 1
 frac[195,230] <= 1
 frac[196,181] <= 1
 frac[197,65] <= 1
 frac[70,209] <= 1
 frac[125,121] <= 1
 frac[130,26] <= 1
 frac[131,152] <= 1
 frac[95,235] <= 1
 frac[97,85] <= 1
 frac[194,127] <= 1
 frac[195,119] <= 1
 frac[196,196] <= 1
 frac[197,210] <= 1
 frac[198,224] <= 1
 frac[61,184] <= 1
 frac[192,9] <= 1
 frac[119,59] <= 1
 frac[138,221] <= 1
 frac[139,217] <= 1
 frac[195,4] <= 1
 frac[198,115] <= 1
 frac[92,95] <= 1
 frac[39,143] <= 1
 frac[192,184] <= 1
 frac[193,46] <= 1
 frac[157,64] <= 1
 frac[196,45] <= 1
 frac[161,5] <= 1
 frac[91,196] <= 1
 frac[94,254] <= 1
 frac[149,113] <= 1
 frac[98,229] <= 1
 frac[22,188] <= 1
 frac[192,215] <= 1
 frac[193,223] <= 1
 frac[194,199] <= 1
 frac[195,239] <= 1
 frac[196,188] <= 1
 frac[51,41] <= 1
 frac[197,74] <= 1
 frac[16,5] <= 1
 frac[41,3] <= 1
 frac[115,4] <= 1
 frac[152,58] <= 1
 frac[136,212] <= 1
 frac[195,124] <= 1
 frac[196,195] <= 1
 frac[197,219] <= 1
 frac[198,219] <= 1
 frac[16,148] <= 1
 frac[153,93] <= 1
 frac[63,189] <= 1
 frac[192,0] <= 1
 frac[84,22] <= 1
 frac[195,13] <= 1
 frac[31,68] <= 1
 frac[88,209] <= 1
 frac[198,106] <= 1
 frac[180,180] <= 1
 frac[40,84] <= 1
 frac[116,92] <= 1
 frac[82,28] <= 1
 frac[192,191] <= 1
 frac[193,39] <= 1
 frac[194,175] <= 1
 frac[196,20] <= 1
 frac[161,30] <= 1
 frac[147,223] <= 1
 frac[149,122] <= 1
 frac[27,184] <= 1
 frac[193,208] <= 1
 frac[194,222] <= 1
 frac[195,212] <= 1
 frac[122,246] <= 1
 frac[196,187] <= 1
 frac[197,51] <= 1
 frac[152,1] <= 1
 frac[119,152] <= 1
 frac[156,134] <= 1
 frac[194,113] <= 1
 frac[195,101] <= 1
 frac[103,137] <= 1
 frac[196,202] <= 1
 frac[125,252] <= 1
 frac[124,242] <= 1
 frac[198,210] <= 1
 frac[56,18] <= 1
 frac[102,240] <= 1
 frac[197,196] <= 1
 frac[132,175] <= 1
 frac[24,123] <= 1
 frac[136,170] <= 1
 frac[192,7] <= 1
 frac[83,165] <= 1
 frac[14,143] <= 1
 frac[198,109] <= 1
 frac[39,189] <= 1
 frac[114,254] <= 1
 frac[168,16] <= 1
 frac[192,182] <= 1
 frac[193,56] <= 1
 frac[11,17] <= 1
 frac[194,166] <= 1
 frac[120,158] <= 1
 frac[196,19] <= 1
 frac[33,108] <= 1
 frac[71,59] <= 1
 frac[96,135] <= 1
 frac[115,101] <= 1
 frac[78,27] <= 1
 frac[8,243] <= 1
 frac[193,169] <= 1
 frac[194,217] <= 1
 frac[195,221] <= 1
 frac[30,158] <= 1
 frac[196,162] <= 1
 frac[69,97] <= 1
 frac[198,186] <= 1
 frac[53,146] <= 1
 frac[197,60] <= 1
 frac[148,77] <= 1
 frac[173,195] <= 1
 frac[194,72] <= 1
 frac[16,130] <= 1
 frac[198,213] <= 1
 frac[197,205] <= 1
 frac[0,76] <= 1
 frac[111,70] <= 1
 frac[134,251] <= 1
 frac[192,30] <= 1
 frac[65,221] <= 1
 frac[140,214] <= 1
 frac[198,100] <= 1
 frac[185,49] <= 1
 frac[192,141] <= 1
 frac[194,161] <= 1
 frac[157,43] <= 1
 frac[196,26] <= 1
 frac[141,252] <= 1
 frac[54,104] <= 1
 frac[182,93] <= 1
 frac[148,229] <= 1
 frac[22,65] <= 1
 frac[5,247] <= 1
 frac[193,162] <= 1
 frac[194,208] <= 1
 frac[195,194] <= 1
 frac[28,229] <= 1
 frac[196,169] <= 1
 frac[197,37] <= 1
 frac[198,189] <= 1
 frac[126,149] <= 1
 frac[184,102] <= 1
 frac[57,216] <= 1
 frac[76,12] <= 1
 frac[4,142] <= 1
 frac[134,99] <= 1
 frac[194,67] <= 1
 frac[195,83] <= 1
 frac[141,19] <= 1
 frac[197,182] <= 1
 frac[198,204] <= 1
 frac[127,254] <= 1
 frac[36,244] <= 1
 frac[52,37] <= 1
 frac[16,249] <= 1
 frac[60,165] <= 1
 frac[192,21] <= 1
 frac[65,214] <= 1
 frac[120,61] <= 1
 frac[198,95] <= 1
 frac[127,79] <= 1
 frac[147,133] <= 1
 frac[168,30] <= 1
 frac[21,233] <= 1
 frac[169,144] <= 1
 frac[171,123] <= 1
 frac[192,132] <= 1
 frac[193,10] <= 1
 frac[194,184] <= 1
 frac[196,1] <= 1
 frac[71,41] <= 1
 frac[78,101] <= 1
 frac[116,216] <= 1
 frac[80,63] <= 1
 frac[188,176] <= 1
 frac[137,8] <= 1
 frac[193,187] <= 1
 frac[194,43] <= 1
 frac[12,102] <= 1
 frac[195,203] <= 1
 frac[196,144] <= 1
 frac[197,46] <= 1
 frac[52,189] <= 1
 frac[198,180] <= 1
 frac[54,18] <= 1
 frac[128,144] <= 1
 frac[177,161] <= 1
 frac[61,100] <= 1
 frac[64,136] <= 1
 frac[118,127] <= 1
 frac[9,70] <= 1
 frac[194,90] <= 1
 frac[195,88] <= 1
 frac[141,12] <= 1
 frac[197,191] <= 1
 frac[198,199] <= 1
 frac[128,15] <= 1
 frac[96,78] <= 1
 frac[114,152] <= 1
 frac[132,132] <= 1
 frac[192,108] <= 1
 frac[10,47] <= 1
 frac[156,2] <= 1
 frac[198,86] <= 1
 frac[2,229] <= 1
 frac[20,132] <= 1
 frac[149,45] <= 1
 frac[191,125] <= 1
 frac[194,179] <= 1
 frac[176,69] <= 1
 frac[139,75] <= 1
 frac[196,8] <= 1
 frac[141,238] <= 1
 frac[53,124] <= 1
 frac[143,254] <= 1
 frac[150,4] <= 1
 frac[153,224] <= 1
 frac[44,68] <= 1
 frac[154,75] <= 1
 frac[155,75] <= 1
 frac[193,188] <= 1
 frac[194,34] <= 1
 frac[195,176] <= 1
 frac[196,159] <= 1
 frac[197,23] <= 1
 frac[198,175] <= 1
 frac[125,47] <= 1
 frac[15,48] <= 1
 frac[128,151] <= 1
 frac[179,65] <= 1
 frac[148,90] <= 1
 frac[6,136] <= 1
 frac[82,217] <= 1
 frac[138,16] <= 1
 frac[194,85] <= 1
 frac[195,65] <= 1
 frac[87,87] <= 1
 frac[197,184] <= 1
 frac[198,62] <= 1
 frac[192,99] <= 1
 frac[193,107] <= 1
 frac[68,171] <= 1
 frac[106,98] <= 1
 frac[198,73] <= 1
 frac[109,215] <= 1
 frac[182,183] <= 1
 frac[43,34] <= 1
 frac[192,146] <= 1
 frac[193,4] <= 1
 frac[196,119] <= 1
 frac[73,177] <= 1
 frac[128,255] <= 1
 frac[168,155] <= 1
 frac[167,253] <= 1
 frac[150,31] <= 1
 frac[154,82] <= 1
 frac[121,157] <= 1
 frac[194,61] <= 1
 frac[195,185] <= 1
 frac[174,145] <= 1
 frac[196,134] <= 1
 frac[197,16] <= 1
 frac[198,166] <= 1
 frac[35,205] <= 1
 frac[56,70] <= 1
 frac[128,110] <= 1
 frac[132,251] <= 1
 frac[98,10] <= 1
 frac[133,78] <= 1
 frac[151,244] <= 1
 frac[195,70] <= 1
 frac[123,126] <= 1
 frac[103,170] <= 1
 frac[197,161] <= 1
 frac[15,174] <= 1
 frac[198,49] <= 1
 frac[183,114] <= 1
 frac[192,122] <= 1
 frac[193,108] <= 1
 frac[178,5] <= 1
 frac[198,64] <= 1
 frac[92,8] <= 1
 frac[22,242] <= 1
 frac[8,7] <= 1
 frac[192,233] <= 1
 frac[193,29] <= 1
 frac[194,133] <= 1
 frac[49,251] <= 1
 frac[196,126] <= 1
 frac[90,82] <= 1
 frac[70,142] <= 1
 frac[1,120] <= 1
 frac[58,123] <= 1
 frac[115,70] <= 1
 frac[137,35] <= 1
 frac[193,142] <= 1
 frac[104,170] <= 1
 frac[195,190] <= 1
 frac[103,82] <= 1
 frac[196,141] <= 1
 frac[197,25] <= 1
 frac[198,153] <= 1
 frac[17,88] <= 1
 frac[191,240] <= 1
 frac[129,220] <= 1
 frac[42,96] <= 1
 frac[150,129] <= 1
 frac[195,79] <= 1
 frac[88,19] <= 1
 frac[13,236] <= 1
 frac[197,170] <= 1
 frac[198,40] <= 1
 frac[43,131] <= 1
 frac[151,98] <= 1
 frac[192,113] <= 1
 frac[193,101] <= 1
 frac[110,169] <= 1
 frac[93,143] <= 1
 frac[7,46] <= 1
 frac[192,224] <= 1
 frac[193,22] <= 1
 frac[194,156] <= 1
 frac[196,101] <= 1
 frac[197,113] <= 1
 frac[73,163] <= 1
 frac[128,205] <= 1
 frac[92,182] <= 1
 frac[75,214] <= 1
 frac[1,113] <= 1
 frac[8,157] <= 1
 frac[193,135] <= 1
 frac[194,15] <= 1
 frac[195,167] <= 1
 frac[196,244] <= 1
 frac[197,2] <= 1
 frac[17,81] <= 1
 frac[53,248] <= 1
 frac[198,144] <= 1
 frac[94,65] <= 1
 frac[130,221] <= 1
 frac[5,101] <= 1
 frac[100,165] <= 1
 frac[155,207] <= 1
 frac[66,117] <= 1
 frac[195,52] <= 1
 frac[159,90] <= 1
 frac[197,147] <= 1
 frac[63,245] <= 1
 frac[28,57] <= 1
 frac[138,156] <= 1
 frac[192,72] <= 1
 frac[193,126] <= 1
 frac[159,235] <= 1
 frac[0,213] <= 1
 frac[57,36] <= 1
 frac[185,95] <= 1
 frac[187,242] <= 1
 frac[117,2] <= 1
 frac[192,231] <= 1
 frac[135,114] <= 1
 frac[194,151] <= 1
 frac[49,245] <= 1
 frac[193,239] <= 1
 frac[196,108] <= 1
 frac[197,122] <= 1
 frac[63,90] <= 1
 frac[4,193] <= 1
 frac[117,179] <= 1
 frac[83,127] <= 1
 frac[193,152] <= 1
 frac[194,6] <= 1
 frac[195,172] <= 1
 frac[196,243] <= 1
 frac[197,11] <= 1
 frac[179,93] <= 1
 frac[198,139] <= 1
 frac[56,91] <= 1
 frac[131,222] <= 1
 frac[172,132] <= 1
 frac[195,61] <= 1
 frac[197,156] <= 1
 frac[198,26] <= 1
 frac[181,242] <= 1
 frac[112,167] <= 1
 frac[111,113] <= 1
 frac[3,24] <= 1
 frac[131,111] <= 1
 frac[166,51] <= 1
 frac[168,89] <= 1
 frac[192,79] <= 1
 frac[193,119] <= 1
 frac[46,158] <= 1
 frac[50,101] <= 1
 frac[74,111] <= 1
 frac[75,119] <= 1
 frac[111,230] <= 1
 frac[192,254] <= 1
 frac[194,238] <= 1
 frac[196,107] <= 1
 frac[125,91] <= 1
 frac[197,99] <= 1
 frac[145,133] <= 1
 frac[128,219] <= 1
 frac[150,59] <= 1
 frac[189,84] <= 1
 frac[193,145] <= 1
 frac[194,1] <= 1
 frac[195,149] <= 1
 frac[174,173] <= 1
 frac[196,250] <= 1
 frac[197,244] <= 1
 frac[198,130] <= 1
 frac[33,133] <= 1
 frac[124,194] <= 1
 frac[20,72] <= 1
 frac[195,34] <= 1
 frac[13,23] <= 1
 frac[30,97] <= 1
 frac[197,133] <= 1
 frac[158,186] <= 1
 frac[198,29] <= 1
 frac[88,232] <= 1
 frac[55,107] <= 1
 frac[134,195] <= 1
 frac[192,70] <= 1
 frac[193,72] <= 1
 frac[14,78] <= 1
 frac[93,154] <= 1
 frac[75,108] <= 1
 frac[39,66] <= 1
 frac[43,7] <= 1
 frac[192,245] <= 1
 frac[193,249] <= 1
 frac[194,233] <= 1
 frac[28,190] <= 1
 frac[196,82] <= 1
 frac[69,145] <= 1
 frac[34,149] <= 1
 frac[142,178] <= 1
 frac[197,108] <= 1
 frac[182,5] <= 1
 frac[147,37] <= 1
 frac[41,61] <= 1
 frac[115,34] <= 1
 frac[194,24] <= 1
 frac[178,254] <= 1
 frac[196,225] <= 1
 frac[197,253] <= 1
 frac[17,68] <= 1
 frac[198,133] <= 1
 frac[36,189] <= 1
 frac[143,81] <= 1
 frac[24,18] <= 1
 frac[192,46] <= 1
 frac[137,168] <= 1
 frac[66,94] <= 1
 frac[139,188] <= 1
 frac[195,43] <= 1
 frac[88,239] <= 1
 frac[141,91] <= 1
 frac[197,142] <= 1
 frac[198,20] <= 1
 frac[125,182] <= 1
 frac[92,116] <= 1
 frac[184,141] <= 1
 frac[38,24] <= 1
 frac[62,146] <= 1
 frac[192,93] <= 1
 frac[193,65] <= 1
 frac[84,91] <= 1
 frac[73,206] <= 1
 frac[143,184] <= 1
 frac[183,220] <= 1
 frac[20,181] <= 1
 frac[167,6] <= 1
 frac[170,230] <= 1
 frac[3,151] <= 1
 frac[172,83] <= 1
 frac[192,204] <= 1
 frac[193,242] <= 1
 frac[194,224] <= 1
 frac[195,242] <= 1
 frac[100,123] <= 1
 frac[196,89] <= 1
 frac[197,85] <= 1
 frac[34,172] <= 1
 frac[147,42] <= 1
 frac[76,252] <= 1
 frac[77,202] <= 1
 frac[150,53] <= 1
 frac[154,184] <= 1
 frac[194,19] <= 1
 frac[195,131] <= 1
 frac[158,21] <= 1
 frac[196,232] <= 1
 frac[197,230] <= 1
 frac[198,252] <= 1
 frac[128,88] <= 1
 frac[136,136] <= 1
 frac[192,37] <= 1
 frac[83,131] <= 1
 frac[84,243] <= 1
 frac[195,16] <= 1
 frac[30,83] <= 1
 frac[198,15] <= 1
 frac[40,121] <= 1
 frac[64,111] <= 1
 frac[192,84] <= 1
 frac[193,90] <= 1
 frac[69,242] <= 1
 frac[87,247] <= 1
 frac[196,49] <= 1
 frac[33,78] <= 1
 frac[127,12] <= 1
 frac[23,193] <= 1
 frac[192,195] <= 1
 frac[46,34] <= 1
 frac[193,203] <= 1
 frac[194,251] <= 1
 frac[195,251] <= 1
 frac[196,64] <= 1
 frac[197,94] <= 1
 frac[2,92] <= 1
 frac[58,33] <= 1
 frac[147,51] <= 1
 frac[170,112] <= 1
 frac[122,66] <= 1
 frac[195,136] <= 1
 frac[194,106] <= 1
 frac[196,215] <= 1
 frac[197,239] <= 1
 frac[198,247] <= 1
 frac[33,168] <= 1
 frac[111,36] <= 1
 frac[170,3] <= 1
 frac[120,20] <= 1
 frac[45,43] <= 1
 frac[192,60] <= 1
 frac[195,25] <= 1
 frac[103,245] <= 1
 frac[86,151] <= 1
 frac[12,180] <= 1
 frac[198,6] <= 1
 frac[37,140] <= 1
 frac[113,203] <= 1
 frac[168,53] <= 1
 frac[99,8] <= 1
 frac[193,83] <= 1
 frac[196,56] <= 1
 frac[107,56] <= 1
 frac[124,0] <= 1
 frac[0,136] <= 1
 frac[57,73] <= 1
 frac[2,164] <= 1
 frac[3,133] <= 1
 frac[95,92] <= 1
 frac[100,105] <= 1
 frac[192,218] <= 1
 frac[46,37] <= 1
 frac[47,153] <= 1
 frac[195,224] <= 1
 frac[194,242] <= 1
 frac[196,79] <= 1
 frac[197,71] <= 1
 frac[72,117] <= 1
 frac[176,4] <= 1
 frac[193,204] <= 1
 frac[55,169] <= 1
 frac[54,37] <= 1
 frac[128,167] <= 1
 frac[41,16] <= 1
 frac[8,167] <= 1
 frac[190,126] <= 1
 frac[194,101] <= 1
 frac[195,113] <= 1
 frac[13,170] <= 1
 frac[196,222] <= 1
 frac[197,232] <= 1
 frac[113,115] <= 1
 frac[58,155] <= 1
 frac[187,78] <= 1
 frac[78,222] <= 1
 frac[192,51] <= 1
 frac[102,95] <= 1
 frac[157,213] <= 1
 frac[195,30] <= 1
 frac[14,187] <= 1
 frac[180,167] <= 1
 frac[198,121] <= 1
 frac[95,52] <= 1
 frac[62,135] <= 1
 frac[192,162] <= 1
 frac[193,84] <= 1
 frac[32,86] <= 1
 frac[196,39] <= 1
 frac[89,107] <= 1
 frac[45,198] <= 1
 frac[193,197] <= 1
 frac[194,205] <= 1
 frac[177,131] <= 1
 frac[140,17] <= 1
 frac[196,182] <= 1
 frac[52,211] <= 1
 frac[142,142] <= 1
 frac[53,134] <= 1
 frac[197,64] <= 1
 frac[79,44] <= 1
 frac[100,231] <= 1
 frac[173,215] <= 1
 frac[194,124] <= 1
 frac[122,84] <= 1
 frac[196,197] <= 1
 frac[197,209] <= 1
 frac[17,32] <= 1
 frac[15,126] <= 1
 frac[198,225] <= 1
 frac[25,96] <= 1
 frac[6,66] <= 1
 frac[133,143] <= 1
 frac[192,10] <= 1
 frac[155,190] <= 1
 frac[139,216] <= 1
 frac[195,7] <= 1
 frac[160,35] <= 1
 frac[157,174] <= 1
 frac[198,112] <= 1
 frac[163,46] <= 1
 frac[1,159] <= 1
 frac[114,205] <= 1
 frac[167,85] <= 1
 frac[192,185] <= 1
 frac[193,45] <= 1
 frac[66,213] <= 1
 frac[121,5] <= 1
 frac[196,46] <= 1
 frac[71,68] <= 1
 frac[98,226] <= 1
 frac[100,31] <= 1
 frac[118,225] <= 1
 frac[193,222] <= 1
 frac[29,5] <= 1
 frac[174,104] <= 1
 frac[194,196] <= 1
 frac[196,189] <= 1
 frac[197,73] <= 1
 frac[58,58] <= 1
 frac[150,209] <= 1
 frac[172,198] <= 1
 frac[104,233] <= 1
 frac[194,119] <= 1
 frac[123,71] <= 1
 frac[196,204] <= 1
 frac[124,244] <= 1
 frac[198,216] <= 1
 frac[162,108] <= 1
 frac[36,216] <= 1
 frac[132,161] <= 1
 frac[192,1] <= 1
 frac[195,12] <= 1
 frac[198,107] <= 1
 frac[115,238] <= 1
 frac[60,8] <= 1
 frac[192,176] <= 1
 frac[121,14] <= 1
 frac[193,38] <= 1
 frac[11,27] <= 1
 frac[194,172] <= 1
 frac[196,21] <= 1
 frac[141,201] <= 1
 frac[120,152] <= 1
 frac[18,75] <= 1
 frac[146,148] <= 1
 frac[168,185] <= 1
 frac[96,129] <= 1
 frac[23,229] <= 1
 frac[155,110] <= 1
 frac[193,215] <= 1
 frac[194,223] <= 1
 frac[195,215] <= 1
 frac[175,99] <= 1
 frac[196,164] <= 1
 frac[197,50] <= 1
 frac[48,100] <= 1
 frac[69,103] <= 1
 frac[173,184] <= 1
 frac[133,96] <= 1
 frac[62,94] <= 1
 frac[64,164] <= 1
 frac[66,5] <= 1
 frac[194,78] <= 1
 frac[195,100] <= 1
 frac[196,203] <= 1
 frac[197,195] <= 1
 frac[198,211] <= 1
 frac[55,37] <= 1
 frac[148,54] <= 1
 frac[23,28] <= 1
 frac[192,24] <= 1
 frac[119,8] <= 1
 frac[63,165] <= 1
 frac[155,140] <= 1
 frac[198,98] <= 1
 frac[74,84] <= 1
 frac[167,75] <= 1
 frac[188,204] <= 1
 frac[136,58] <= 1
 frac[192,183] <= 1
 frac[193,63] <= 1
 frac[194,167] <= 1
 frac[33,99] <= 1
 frac[196,28] <= 1
 frac[72,2] <= 1
 frac[144,42] <= 1
 frac[163,173] <= 1
 frac[58,93] <= 1
 frac[23,226] <= 1
 frac[43,86] <= 1
 frac[8,244] <= 1
 frac[193,168] <= 1
 frac[194,214] <= 1
 frac[195,220] <= 1
 frac[196,163] <= 1
 frac[197,59] <= 1
 frac[198,187] <= 1
 frac[165,2] <= 1
 frac[172,180] <= 1
 frac[194,73] <= 1
 frac[176,191] <= 1
 frac[105,239] <= 1
 frac[87,75] <= 1
 frac[197,204] <= 1
 frac[15,85] <= 1
 frac[198,202] <= 1
 frac[43,161] <= 1
 frac[192,31] <= 1
 frac[83,173] <= 1
 frac[66,143] <= 1
 frac[157,185] <= 1
 frac[32,3] <= 1
 frac[159,0] <= 1
 frac[198,101] <= 1
 frac[0,252] <= 1
 frac[108,227] <= 1
 frac[39,181] <= 1
 frac[114,246] <= 1
 frac[117,91] <= 1
 frac[192,142] <= 1
 frac[193,48] <= 1
 frac[194,190] <= 1
 frac[47,213] <= 1
 frac[196,27] <= 1
 frac[75,52] <= 1
 frac[137,6] <= 1
 frac[193,161] <= 1
 frac[195,197] <= 1
 frac[196,170] <= 1
 frac[197,36] <= 1
 frac[165,11] <= 1
 frac[195,82] <= 1
 frac[197,181] <= 1
 frac[198,205] <= 1
 frac[55,59] <= 1
 frac[111,78] <= 1
 frac[96,72] <= 1
 frac[41,134] <= 1
 frac[82,119] <= 1
 frac[136,89] <= 1
 frac[70,11] <= 1
 frac[198,92] <= 1
 frac[163,202] <= 1
 frac[192,133] <= 1
 frac[193,9] <= 1
 frac[196,2] <= 1
 frac[56,202] <= 1
 frac[92,211] <= 1
 frac[74,169] <= 1
 frac[193,186] <= 1
 frac[195,202] <= 1
 frac[196,145] <= 1
 frac[197,45] <= 1
 frac[198,181] <= 1
 frac[130,246] <= 1
 frac[148,92] <= 1
 frac[60,206] <= 1
 frac[194,91] <= 1
 frac[29,154] <= 1
 frac[195,91] <= 1
 frac[197,190] <= 1
 frac[198,196] <= 1
 frac[56,40] <= 1
 frac[186,193] <= 1
 frac[43,159] <= 1
 frac[82,110] <= 1
 frac[192,109] <= 1
 frac[66,129] <= 1
 frac[198,87] <= 1
 frac[162,227] <= 1
 frac[0,202] <= 1
 frac[192,156] <= 1
 frac[193,2] <= 1
 frac[104,46] <= 1
 frac[50,214] <= 1
 frac[196,9] <= 1
 frac[147,250] <= 1
 frac[97,17] <= 1
 frac[190,188] <= 1
 frac[193,179] <= 1
 frac[194,35] <= 1
 frac[196,152] <= 1
 frac[197,22] <= 1
 frac[198,172] <= 1
 frac[124,160] <= 1
 frac[120,232] <= 1
 frac[75,179] <= 1
 frac[170,72] <= 1
 frac[136,248] <= 1
 frac[27,52] <= 1
 frac[194,82] <= 1
 frac[195,64] <= 1
 frac[107,226] <= 1
 frac[197,167] <= 1
 frac[41,240] <= 1
 frac[131,114] <= 1
 frac[45,115] <= 1
 frac[192,100] <= 1
 frac[102,108] <= 1
 frac[65,167] <= 1
 frac[140,252] <= 1
 frac[193,106] <= 1
 frac[198,78] <= 1
 frac[2,253] <= 1
 frac[192,147] <= 1
 frac[193,27] <= 1
 frac[135,6] <= 1
 frac[194,139] <= 1
 frac[140,83] <= 1
 frac[124,72] <= 1
 frac[141,230] <= 1
 frac[90,72] <= 1
 frac[196,112] <= 1
 frac[186,42] <= 1
 frac[96,164] <= 1
 frac[151,134] <= 1
 frac[193,180] <= 1
 frac[176,204] <= 1
 frac[195,184] <= 1
 frac[68,74] <= 1
 frac[196,135] <= 1
 frac[197,31] <= 1
 frac[198,167] <= 1
 frac[11,141] <= 1
 frac[194,58] <= 1
 frac[105,58] <= 1
 frac[143,119] <= 1
 frac[60,220] <= 1
 frac[10,79] <= 1
 frac[84,202] <= 1
 frac[156,226] <= 1
 frac[195,73] <= 1
 frac[106,219] <= 1
 frac[197,160] <= 1
 frac[198,54] <= 1
 frac[55,6] <= 1
 frac[114,139] <= 1
 frac[192,123] <= 1
 frac[193,99] <= 1
 frac[68,147] <= 1
 frac[178,2] <= 1
 frac[141,142] <= 1
 frac[89,50] <= 1
 frac[198,65] <= 1
 frac[2,244] <= 1
 frac[21,3] <= 1
 frac[9,158] <= 1
 frac[193,28] <= 1
 frac[121,92] <= 1
 frac[196,127] <= 1
 frac[197,119] <= 1
 frac[73,185] <= 1
 frac[182,62] <= 1
 frac[164,64] <= 1
 frac[81,169] <= 1
 frac[6,232] <= 1
 frac[189,72] <= 1
 frac[46,100] <= 1
 frac[195,161] <= 1
 frac[106,179] <= 1
 frac[196,142] <= 1
 frac[197,24] <= 1
 frac[198,158] <= 1
 frac[13,154] <= 1
 frac[34,225] <= 1
 frac[53,254] <= 1
 frac[119,77] <= 1
 frac[157,229] <= 1
 frac[195,78] <= 1
 frac[31,129] <= 1
 frac[67,123] <= 1
 frac[197,169] <= 1
 frac[198,41] <= 1
 frac[179,255] <= 1
 frac[36,233] <= 1
 frac[20,35] <= 1
 frac[60,74] <= 1
 frac[188,119] <= 1
 frac[192,114] <= 1
 frac[193,100] <= 1
 frac[70,119] <= 1
 frac[164,248] <= 1
 frac[74,122] <= 1
 frac[146,82] <= 1
 frac[39,86] <= 1
 frac[132,1] <= 1
 frac[8,15] <= 1
 frac[192,225] <= 1
 frac[48,166] <= 1
 frac[194,157] <= 1
 frac[11,42] <= 1
 frac[50,195] <= 1
 frac[196,102] <= 1
 frac[197,112] <= 1
 frac[74,205] <= 1
 frac[78,70] <= 1
 frac[118,41] <= 1
 frac[193,134] <= 1
 frac[194,12] <= 1
 frac[195,166] <= 1
 frac[86,6] <= 1
 frac[196,245] <= 1
 frac[197,1] <= 1
 frac[198,145] <= 1
 frac[123,158] <= 1
 frac[177,204] <= 1
 frac[41,90] <= 1
 frac[133,191] <= 1
 frac[134,71] <= 1
 frac[154,212] <= 1
 frac[155,206] <= 1
 frac[102,134] <= 1
 frac[6,114] <= 1
 frac[195,55] <= 1
 frac[31,142] <= 1
 frac[16,221] <= 1
 frac[198,32] <= 1
 frac[197,146] <= 1
 frac[0,39] <= 1
 frac[55,20] <= 1
 frac[39,254] <= 1
 frac[192,73] <= 1
 frac[45,94] <= 1
 frac[102,17] <= 1
 frac[67,241] <= 1
 frac[49,155] <= 1
 frac[168,226] <= 1
 frac[95,114] <= 1
 frac[166,136] <= 1
 frac[190,201] <= 1
 frac[45,239] <= 1
 frac[193,238] <= 1
 frac[194,148] <= 1
 frac[122,188] <= 1
 frac[174,56] <= 1
 frac[196,109] <= 1
 frac[197,121] <= 1
 frac[94,190] <= 1
 frac[58,10] <= 1
 frac[79,69] <= 1
 frac[193,159] <= 1
 frac[194,7] <= 1
 frac[195,175] <= 1
 frac[106,165] <= 1
 frac[196,252] <= 1
 frac[197,10] <= 1
 frac[17,73] <= 1
 frac[160,155] <= 1
 frac[16,69] <= 1
 frac[53,192] <= 1
 frac[198,136] <= 1
 frac[82,186] <= 1
 frac[136,148] <= 1
 frac[177,86] <= 1
 frac[195,60] <= 1
 frac[197,155] <= 1
 frac[198,27] <= 1
 frac[18,232] <= 1
 frac[192,64] <= 1
 frac[156,126] <= 1
 frac[138,132] <= 1
 frac[70,97] <= 1
 frac[90,189] <= 1
 frac[147,174] <= 1
 frac[149,73] <= 1
 frac[169,253] <= 1
 frac[98,205] <= 1
 frac[120,215] <= 1
 frac[193,231] <= 1
 frac[194,239] <= 1
 frac[192,255] <= 1
 frac[67,135] <= 1
 frac[196,84] <= 1
 frac[197,98] <= 1
 frac[41,59] <= 1
 frac[83,7] <= 1
 frac[193,144] <= 1
 frac[194,30] <= 1
 frac[195,148] <= 1
 frac[103,120] <= 1
 frac[196,251] <= 1
 frac[10,144] <= 1
 frac[198,131] <= 1
 frac[126,171] <= 1
 frac[143,75] <= 1
 frac[197,243] <= 1
 frac[38,151] <= 1
 frac[56,99] <= 1
 frac[28,51] <= 1
 frac[134,49] <= 1
 frac[136,155] <= 1
 frac[156,198] <= 1
 frac[85,216] <= 1
 frac[195,37] <= 1
 frac[105,167] <= 1
 frac[197,132] <= 1
 frac[198,18] <= 1
 frac[0,53] <= 1
 frac[112,223] <= 1
 frac[183,81] <= 1
 frac[186,247] <= 1
 frac[25,51] <= 1
 frac[62,232] <= 1
 frac[193,79] <= 1
 frac[86,242] <= 1
 frac[90,132] <= 1
 frac[18,114] <= 1
 frac[146,191] <= 1
 frac[97,222] <= 1
 frac[117,19] <= 1
 frac[154,15] <= 1
 frac[192,246] <= 1
 frac[193,248] <= 1
 frac[194,230] <= 1
 frac[176,16] <= 1
 frac[68,6] <= 1
 frac[196,83] <= 1
 frac[197,107] <= 1
 frac[90,55] <= 1
 frac[147,36] <= 1
 frac[58,28] <= 1
 frac[97,79] <= 1
 frac[170,143] <= 1
 frac[79,75] <= 1
 frac[173,242] <= 1
 frac[194,25] <= 1
 frac[195,157] <= 1
 frac[11,174] <= 1
 frac[196,226] <= 1
 frac[197,252] <= 1
 frac[198,250] <= 1
 frac[103,113] <= 1
 frac[15,101] <= 1
 frac[37,8] <= 1
 frac[71,136] <= 1
 frac[179,170] <= 1
 frac[59,247] <= 1
 frac[111,17] <= 1
 frac[192,47] <= 1
 frac[137,175] <= 1
 frac[138,39] <= 1
 frac[67,87] <= 1
 frac[195,42] <= 1
 frac[13,15] <= 1
 frac[197,141] <= 1
 frac[198,21] <= 1
 frac[60,110] <= 1
 frac[134,187] <= 1
 frac[192,94] <= 1
 frac[193,64] <= 1
 frac[157,122] <= 1
 frac[142,11] <= 1
 frac[53,57] <= 1
 frac[109,172] <= 1
 frac[127,6] <= 1
 frac[113,169] <= 1
 frac[96,239] <= 1
 frac[116,226] <= 1
 frac[27,217] <= 1
 frac[192,205] <= 1
 frac[193,241] <= 1
 frac[194,225] <= 1
 frac[195,245] <= 1
 frac[196,90] <= 1
 frac[197,84] <= 1
 frac[74,225] <= 1
 frac[57,175] <= 1
 frac[58,39] <= 1
 frac[115,42] <= 1
 frac[7,162] <= 1
 frac[173,235] <= 1
 frac[194,16] <= 1
 frac[195,130] <= 1
 frac[196,233] <= 1
 frac[197,229] <= 1
 frac[198,253] <= 1
 frac[40,235] <= 1
 frac[38,129] <= 1
 frac[60,246] <= 1
 frac[192,38] <= 1
 frac[155,170] <= 1
 frac[120,14] <= 1
 frac[195,19] <= 1
 frac[198,12] <= 1
 frac[96,119] <= 1
 frac[82,54] <= 1
 frac[192,85] <= 1
 frac[193,89] <= 1
 frac[196,50] <= 1
 frac[71,88] <= 1
 frac[18,100] <= 1
 frac[182,101] <= 1
 frac[96,230] <= 1
 frac[80,96] <= 1
 frac[78,58] <= 1
 frac[25,174] <= 1
 frac[192,196] <= 1
 frac[193,202] <= 1
 frac[194,248] <= 1
 frac[195,250] <= 1
 frac[177,144] <= 1
 frac[196,65] <= 1
 frac[197,93] <= 1
 frac[52,206] <= 1
 frac[131,172] <= 1
 frac[61,51] <= 1
 frac[196,208] <= 1
 frac[198,244] <= 1
 frac[79,206] <= 1
 frac[81,16] <= 1
 frac[192,61] <= 1
 frac[195,24] <= 1
 frac[86,148] <= 1
 frac[198,7] <= 1
 frac[127,167] <= 1
 frac[17,198] <= 1
 frac[183,188] <= 1
 frac[38,11] <= 1
 frac[188,41] <= 1
 frac[192,172] <= 1
 frac[193,82] <= 1
 frac[50,6] <= 1
 frac[196,57] <= 1
 frac[23,217] <= 1
 frac[192,219] <= 1
 frac[193,195] <= 1
 frac[194,243] <= 1
 frac[195,227] <= 1
 frac[47,152] <= 1
 frac[196,72] <= 1
 frac[197,70] <= 1
 frac[33,55] <= 1
 frac[48,112] <= 1
 frac[147,59] <= 1
 frac[171,193] <= 1
 frac[10,180] <= 1
 frac[194,98] <= 1
 frac[195,112] <= 1
 frac[196,223] <= 1
 frac[197,215] <= 1
 frac[198,239] <= 1
 frac[109,117] <= 1
 frac[34,46] <= 1
 frac[40,217] <= 1
 frac[131,34] <= 1
 frac[187,65] <= 1
 frac[170,11] <= 1
 frac[82,153] <= 1
 frac[192,52] <= 1
 frac[175,181] <= 1
 frac[195,1] <= 1
 frac[14,184] <= 1
 frac[72,159] <= 1
 frac[198,126] <= 1
 frac[96,5] <= 1
 frac[192,163] <= 1
 frac[193,43] <= 1
 frac[45,180] <= 1
 frac[196,32] <= 1
 frac[75,11] <= 1
 frac[118,239] <= 1
 frac[63,111] <= 1
 frac[193,196] <= 1
 frac[192,210] <= 1
 frac[195,232] <= 1
 frac[194,202] <= 1
 frac[196,183] <= 1
 frac[197,79] <= 1
 frac[34,182] <= 1
 frac[53,133] <= 1
 frac[98,91] <= 1
 frac[190,102] <= 1
 frac[66,48] <= 1
 frac[194,125] <= 1
 frac[195,121] <= 1
 frac[160,181] <= 1
 frac[196,198] <= 1
 frac[197,208] <= 1
 frac[198,230] <= 1
 frac[192,11] <= 1
 frac[83,153] <= 1
 frac[195,6] <= 1
 frac[159,20] <= 1
 frac[161,170] <= 1
 frac[198,113] <= 1
 frac[15,238] <= 1
 frac[92,89] <= 1
 frac[126,89] <= 1
 frac[112,242] <= 1
 frac[21,211] <= 1
 frac[24,158] <= 1
 frac[23,126] <= 1
 frac[122,55] <= 1
 frac[193,44] <= 1
 frac[196,47] <= 1
 frac[71,71] <= 1
 frac[115,153] <= 1
 frac[100,24] <= 1
 frac[172,48] <= 1
 frac[193,221] <= 1
 frac[195,209] <= 1
 frac[196,190] <= 1
 frac[197,72] <= 1
 frac[71,212] <= 1
 frac[147,73] <= 1
 frac[75,145] <= 1
 frac[149,224] <= 1
 frac[133,102] <= 1
 frac[27,82] <= 1
 frac[29,181] <= 1
 frac[194,116] <= 1
 frac[195,126] <= 1
 frac[196,205] <= 1
 frac[197,217] <= 1
 frac[198,217] <= 1
 frac[18,166] <= 1
 frac[54,167] <= 1
 frac[41,146] <= 1
 frac[192,2] <= 1
 frac[155,182] <= 1
 frac[195,15] <= 1
 frac[123,55] <= 1
 frac[198,104] <= 1
 frac[18,57] <= 1
 frac[186,157] <= 1
 frac[25,25] <= 1
 frac[117,72] <= 1
 frac[192,177] <= 1
 frac[63,12] <= 1
 frac[194,173] <= 1
 frac[193,37] <= 1
 frac[196,22] <= 1
 frac[14,53] <= 1
 frac[181,81] <= 1
 frac[149,120] <= 1
 frac[170,194] <= 1
 frac[193,214] <= 1
 frac[194,220] <= 1
 frac[195,214] <= 1
 frac[103,58] <= 1
 frac[196,165] <= 1
 frac[197,49] <= 1
 frac[73,99] <= 1
 frac[128,141] <= 1
 frac[77,230] <= 1
 frac[80,219] <= 1
 frac[81,227] <= 1
 frac[61,87] <= 1
 frac[194,79] <= 1
 frac[159,117] <= 1
 frac[197,194] <= 1
 frac[198,208] <= 1
 frac[181,184] <= 1
 frac[56,20] <= 1
 frac[128,60] <= 1
 frac[188,94] <= 1
 frac[133,240] <= 1
 frac[82,130] <= 1
 frac[192,25] <= 1
 frac[45,14] <= 1
 frac[63,164] <= 1
 frac[139,201] <= 1
 frac[140,217] <= 1
 frac[13,37] <= 1
 frac[89,204] <= 1
 frac[198,99] <= 1
 frac[181,41] <= 1
 frac[126,75] <= 1
 frac[134,145] <= 1
 frac[192,136] <= 1
 frac[194,164] <= 1
 frac[32,124] <= 1
 frac[50,42] <= 1
 frac[161,21] <= 1
 frac[90,117] <= 1
 frac[196,29] <= 1
 frac[181,90] <= 1
 frac[26,3] <= 1
 frac[193,175] <= 1
 frac[194,215] <= 1
 frac[104,73] <= 1
 frac[195,223] <= 1
 frac[33,211] <= 1
 frac[197,58] <= 1
 frac[198,184] <= 1
 frac[196,172] <= 1
 frac[148,79] <= 1
 frac[10,168] <= 1
 frac[194,70] <= 1
 frac[195,108] <= 1
 frac[159,114] <= 1
 frac[197,203] <= 1
 frac[198,203] <= 1
 frac[91,15] <= 1
 frac[125,243] <= 1
 frac[23,20] <= 1
 frac[192,16] <= 1
 frac[67,38] <= 1
 frac[160,73] <= 1
 frac[159,3] <= 1
 frac[198,90] <= 1
 frac[192,143] <= 1
 frac[193,55] <= 1
 frac[194,191] <= 1
 frac[196,4] <= 1
 frac[164,45] <= 1
 frac[166,194] <= 1
 frac[171,149] <= 1
 frac[193,160] <= 1
 frac[194,46] <= 1
 frac[196,171] <= 1
 frac[197,35] <= 1
 frac[69,104] <= 1
 frac[145,69] <= 1
 frac[198,179] <= 1
 frac[55,133] <= 1
 frac[170,71] <= 1
 frac[101,18] <= 1
 frac[174,237] <= 1
 frac[195,85] <= 1
 frac[194,65] <= 1
 frac[197,180] <= 1
 frac[198,194] <= 1
 frac[36,246] <= 1
 frac[6,47] <= 1
 frac[151,88] <= 1
 frac[192,23] <= 1
 frac[153,54] <= 1
 frac[173,75] <= 1
 frac[198,93] <= 1
 frac[18,34] <= 1
 frac[146,111] <= 1
 frac[182,187] <= 1
 frac[149,34] <= 1
 frac[192,134] <= 1
 frac[193,8] <= 1
 frac[194,182] <= 1
 frac[48,187] <= 1
 frac[33,124] <= 1
 frac[107,21] <= 1
 frac[169,39] <= 1
 frac[8,227] <= 1
 frac[27,129] <= 1
 frac[193,185] <= 1
 frac[194,41] <= 1
 frac[195,205] <= 1
 frac[196,146] <= 1
 frac[197,44] <= 1
 frac[162,30] <= 1
 frac[198,170] <= 1
 frac[15,53] <= 1
 frac[184,111] <= 1
 frac[73,22] <= 1
 frac[81,214] <= 1
 frac[10,90] <= 1
 frac[194,88] <= 1
 frac[47,51] <= 1
 frac[88,0] <= 1
 frac[195,90] <= 1
 frac[197,189] <= 1
 frac[107,236] <= 1
 frac[198,197] <= 1
 frac[36,253] <= 1
 frac[184,222] <= 1
 frac[40,163] <= 1
 frac[54,131] <= 1
 frac[17,4] <= 1
 frac[186,206] <= 1
 frac[127,225] <= 1
 frac[157,138] <= 1
 frac[89,39] <= 1
 frac[198,84] <= 1
 frac[99,90] <= 1
 frac[192,157] <= 1
 frac[194,177] <= 1
 frac[196,10] <= 1
 frac[147,253] <= 1
 frac[81,190] <= 1
 frac[9,252] <= 1
 frac[119,201] <= 1
 frac[194,32] <= 1
 frac[29,225] <= 1
 frac[195,178] <= 1
 frac[196,153] <= 1
 frac[197,21] <= 1
 frac[51,12] <= 1
 frac[17,108] <= 1
 frac[198,173] <= 1
 frac[19,55] <= 1
 frac[129,192] <= 1
 frac[2,5] <= 1
 frac[60,198] <= 1
 frac[82,215] <= 1
 frac[194,83] <= 1
 frac[195,67] <= 1
 frac[88,7] <= 1
 frac[141,3] <= 1
 frac[197,166] <= 1
 frac[198,60] <= 1
 frac[112,189] <= 1
 frac[3,62] <= 1
 frac[132,141] <= 1
 frac[136,72] <= 1
 frac[192,101] <= 1
 frac[49,119] <= 1
 frac[85,171] <= 1
 frac[159,254] <= 1
 frac[198,79] <= 1
 frac[108,201] <= 1
 frac[111,204] <= 1
 frac[60,36] <= 1
 frac[8,2] <= 1
 frac[192,148] <= 1
 frac[120,188] <= 1
 frac[193,26] <= 1
 frac[194,136] <= 1
 frac[196,113] <= 1
 frac[87,183] <= 1
 frac[141,229] <= 1
 frac[2,109] <= 1
 frac[98,153] <= 1
 frac[171,248] <= 1
 frac[7,75] <= 1
 frac[155,66] <= 1
 frac[48,56] <= 1
 frac[194,59] <= 1
 frac[193,139] <= 1
 frac[195,187] <= 1
 frac[196,128] <= 1
 frac[197,30] <= 1
 frac[198,164] <= 1
 frac[162,16] <= 1
 frac[15,59] <= 1
 frac[80,190] <= 1
 frac[7,216] <= 1
 frac[139,157] <= 1
 frac[195,72] <= 1
 frac[85,195] <= 1
 frac[197,175] <= 1
 frac[198,55] <= 1
 frac[20,37] <= 1
 frac[134,229] <= 1
 frac[192,124] <= 1
 frac[193,98] <= 1
 frac[102,100] <= 1
 frac[120,84] <= 1
 frac[32,32] <= 1
 frac[198,70] <= 1
 frac[110,164] <= 1
 frac[57,24] <= 1
 frac[97,177] <= 1
 frac[192,235] <= 1
 frac[193,19] <= 1
 frac[194,131] <= 1
 frac[139,123] <= 1
 frac[196,120] <= 1
 frac[72,38] <= 1
 frac[143,206] <= 1
 frac[114,106] <= 1
 frac[80,38] <= 1
 frac[8,152] <= 1
 frac[100,41] <= 1
 frac[119,223] <= 1
 frac[9,238] <= 1
 frac[194,50] <= 1
 frac[195,160] <= 1
 frac[31,243] <= 1
 frac[196,143] <= 1
 frac[197,7] <= 1
 frac[180,65] <= 1
 frac[198,159] <= 1
 frac[35,244] <= 1
 frac[193,140] <= 1
 frac[195,49] <= 1
 frac[158,71] <= 1
 frac[177,91] <= 1
 frac[197,168] <= 1
 frac[198,46] <= 1
 frac[16,231] <= 1
 frac[59,91] <= 1
 frac[6,11] <= 1
 frac[192,115] <= 1
 frac[193,123] <= 1
 frac[164,249] <= 1
 frac[8,48] <= 1
 frac[117,7] <= 1
 frac[192,226] <= 1
 frac[193,20] <= 1
 frac[194,154] <= 1
 frac[196,103] <= 1
 frac[197,127] <= 1
 frac[72,45] <= 1
 frac[143,215] <= 1
 frac[74,202] <= 1
 frac[2,127] <= 1
 frac[94,180] <= 1
 frac[146,226] <= 1
 frac[171,246] <= 1
 frac[193,133] <= 1
 frac[194,13] <= 1
 frac[195,169] <= 1
 frac[196,246] <= 1
 frac[52,147] <= 1
 frac[198,150] <= 1
 frac[197,0] <= 1
 frac[113,91] <= 1
 frac[76,41] <= 1
 frac[131,219] <= 1
 frac[84,217] <= 1
 frac[195,54] <= 1
 frac[197,145] <= 1
 frac[198,33] <= 1
 frac[109,47] <= 1
 frac[183,66] <= 1
 frac[38,245] <= 1
 frac[192,74] <= 1
 frac[193,124] <= 1
 frac[186,85] <= 1
 frac[188,254] <= 1
 frac[118,182] <= 1
 frac[192,249] <= 1
 frac[191,147] <= 1
 frac[194,149] <= 1
 frac[65,58] <= 1
 frac[135,108] <= 1
 frac[196,110] <= 1
 frac[197,120] <= 1
 frac[34,129] <= 1
 frac[44,42] <= 1
 frac[190,89] <= 1
 frac[119,173] <= 1
 frac[193,158] <= 1
 frac[194,4] <= 1
 frac[195,174] <= 1
 frac[121,182] <= 1
 frac[196,253] <= 1
 frac[197,9] <= 1
 frac[198,137] <= 1
 frac[56,93] <= 1
 frac[38,157] <= 1
 frac[111,2] <= 1
 frac[61,159] <= 1
 frac[123,7] <= 1
 frac[195,63] <= 1
 frac[177,85] <= 1
 frac[197,154] <= 1
 frac[198,24] <= 1
 frac[110,206] <= 1
 frac[92,96] <= 1
 frac[38,236] <= 1
 frac[41,211] <= 1
 frac[132,97] <= 1
 frac[192,65] <= 1
 frac[193,117] <= 1
 frac[84,87] <= 1
 frac[106,68] <= 1
 frac[71,108] <= 1
 frac[92,23] <= 1
 frac[98,202] <= 1
 frac[190,241] <= 1
 frac[192,240] <= 1
 frac[193,230] <= 1
 frac[194,236] <= 1
 frac[157,24] <= 1
 frac[174,64] <= 1
 frac[196,85] <= 1
 frac[197,97] <= 1
 frac[104,114] <= 1
 frac[67,134] <= 1
 frac[149,185] <= 1
 frac[98,189] <= 1
 frac[152,211] <= 1
 frac[10,145] <= 1
 frac[46,126] <= 1
 frac[194,31] <= 1
 frac[193,151] <= 1
 frac[195,151] <= 1
 frac[196,228] <= 1
 frac[197,242] <= 1
 frac[198,128] <= 1
 frac[180,94] <= 1
 frac[69,39] <= 1
 frac[34,11] <= 1
 frac[111,11] <= 1
 frac[36,176] <= 1
 frac[24,13] <= 1
 frac[98,44] <= 1
 frac[64,228] <= 1
 frac[192,41] <= 1
 frac[137,165] <= 1
 frac[157,207] <= 1
 frac[195,36] <= 1
 frac[197,131] <= 1
 frac[198,19] <= 1
 frac[16,204] <= 1
 frac[184,136] <= 1
 frac[77,56] <= 1
 frac[44,250] <= 1
 frac[82,69] <= 1
 frac[63,229] <= 1
 frac[193,78] <= 1
 frac[66,244] <= 1
 frac[192,88] <= 1
 frac[49,148] <= 1
 frac[178,39] <= 1
 frac[46,167] <= 1
 frac[40,28] <= 1
 frac[192,247] <= 1
 frac[193,255] <= 1
 frac[194,231] <= 1
 frac[49,229] <= 1
 frac[104,121] <= 1
 frac[196,92] <= 1
 frac[197,106] <= 1
 frac[16,37] <= 1
 frac[143,34] <= 1
 frac[28,184] <= 1
 frac[146,207] <= 1
 frac[125,82] <= 1
 frac[115,36] <= 1
 frac[81,148] <= 1
 frac[44,24] <= 1
 frac[7,164] <= 1
 frac[137,93] <= 1
 frac[194,22] <= 1
 frac[196,227] <= 1
 frac[16,180] <= 1
 frac[197,251] <= 1
 frac[143,83] <= 1
 frac[145,13] <= 1
 frac[42,66] <= 1
 frac[61,145] <= 1
 frac[136,131] <= 1
 frac[192,32] <= 1
 frac[119,80] <= 1
 frac[139,190] <= 1
 frac[12,128] <= 1
 frac[195,45] <= 1
 frac[175,153] <= 1
 frac[197,140] <= 1
 frac[198,10] <= 1
 frac[40,116] <= 1
 frac[187,170] <= 1
 frac[193,71] <= 1
 frac[67,231] <= 1
 frac[196,52] <= 1
 frac[90,140] <= 1
 frac[71,98] <= 1
 frac[163,133] <= 1
 frac[2,208] <= 1
 frac[113,168] <= 1
 frac[96,224] <= 1
 frac[21,47] <= 1
 frac[39,117] <= 1
 frac[78,48] <= 1
 frac[171,69] <= 1
 frac[190,227] <= 1
 frac[192,206] <= 1
 frac[193,240] <= 1
 frac[194,254] <= 1
 frac[191,142] <= 1
 frac[195,244] <= 1
 frac[196,91] <= 1
 frac[124,99] <= 1
 frac[197,83] <= 1
 frac[100,101] <= 1
 frac[167,145] <= 1
 frac[194,17] <= 1
 frac[195,133] <= 1
 frac[29,208] <= 1
 frac[33,149] <= 1
 frac[197,228] <= 1
 frac[198,242] <= 1
 frac[112,127] <= 1
 frac[19,102] <= 1
 frac[77,89] <= 1
 frac[192,39] <= 1
 frac[137,215] <= 1
 frac[198,13] <= 1
 frac[164,181] <= 1
 frac[37,129] <= 1
 frac[40,123] <= 1
 frac[168,48] <= 1
 frac[134,179] <= 1
 frac[192,86] <= 1
 frac[193,88] <= 1
 frac[196,51] <= 1
 frac[107,69] <= 1
 frac[87,241] <= 1
 frac[109,164] <= 1
 frac[182,122] <= 1
 frac[193,201] <= 1
 frac[194,249] <= 1
 frac[195,253] <= 1
 frac[196,66] <= 1
 frac[197,92] <= 1
 frac[93,27] <= 1
 frac[80,240] <= 1
 frac[27,78] <= 1
 frac[173,227] <= 1
 frac[194,104] <= 1
 frac[195,138] <= 1
 frac[160,184] <= 1
 frac[196,209] <= 1
 frac[197,237] <= 1
 frac[198,245] <= 1
 frac[16,162] <= 1
 frac[41,190] <= 1
 frac[59,4] <= 1
 frac[150,189] <= 1
 frac[135,171] <= 1
 frac[192,62] <= 1
 frac[119,38] <= 1
 frac[137,216] <= 1
 frac[195,27] <= 1
 frac[123,35] <= 1
 frac[13,0] <= 1
 frac[198,4] <= 1
 frac[168,55] <= 1
 frac[189,148] <= 1
 frac[193,81] <= 1
 frac[140,133] <= 1
 frac[196,58] <= 1
 frac[94,243] <= 1
 frac[21,49] <= 1
 frac[154,25] <= 1
 frac[192,220] <= 1
 frac[193,194] <= 1
 frac[194,240] <= 1
 frac[195,226] <= 1
 frac[32,192] <= 1
 frac[196,73] <= 1
 frac[197,69] <= 1
 frac[30,161] <= 1
 frac[6,190] <= 1
 frac[8,169] <= 1
 frac[7,147] <= 1
 frac[10,181] <= 1
 frac[156,180] <= 1
 frac[194,99] <= 1
 frac[176,149] <= 1
 frac[196,216] <= 1
 frac[197,214] <= 1
 frac[35,135] <= 1
 frac[198,236] <= 1
 frac[108,106] <= 1
 frac[55,216] <= 1
 frac[183,7] <= 1
 frac[180,50] <= 1
 frac[42,63] <= 1
 frac[150,180] <= 1
 frac[136,184] <= 1
 frac[135,160] <= 1
 frac[192,53] <= 1
 frac[156,43] <= 1
 frac[195,0] <= 1
 frac[89,248] <= 1
 frac[198,127] <= 1
 frac[127,175] <= 1
 frac[44,222] <= 1
 frac[192,164] <= 1
 frac[193,42] <= 1
 frac[196,33] <= 1
 frac[127,60] <= 1
 frac[97,249] <= 1
 frac[5,144] <= 1
 frac[99,128] <= 1
 frac[7,123] <= 1
 frac[192,211] <= 1
 frac[193,219] <= 1
 frac[194,203] <= 1
 frac[195,235] <= 1
 frac[29,10] <= 1
 frac[122,227] <= 1
 frac[197,78] <= 1
 frac[90,8] <= 1
 frac[196,176] <= 1
 frac[144,86] <= 1
 frac[147,67] <= 1
 frac[153,168] <= 1
 frac[27,92] <= 1
 frac[194,122] <= 1
 frac[195,120] <= 1
 frac[122,82] <= 1
 frac[196,199] <= 1
 frac[197,223] <= 1
 frac[180,57] <= 1
 frac[198,231] <= 1
 frac[163,67] <= 1
 frac[192,12] <= 1
 frac[45,27] <= 1
 frac[195,9] <= 1
 frac[88,205] <= 1
 frac[141,125] <= 1
 frac[44,213] <= 1
 frac[192,187] <= 1
 frac[193,35] <= 1
 frac[46,202] <= 1
 frac[28,100] <= 1
 frac[157,93] <= 1
 frac[85,117] <= 1
 frac[87,156] <= 1
 frac[196,40] <= 1
 frac[188,175] <= 1
 frac[117,255] <= 1
 frac[8,200] <= 1
 frac[171,161] <= 1
 frac[193,220] <= 1
 frac[9,222] <= 1
 frac[195,208] <= 1
 frac[194,194] <= 1
 frac[196,191] <= 1
 frac[197,55] <= 1
 frac[153,177] <= 1
 frac[64,175] <= 1
 frac[27,85] <= 1
 frac[194,117] <= 1
 frac[195,97] <= 1
 frac[196,206] <= 1
 frac[197,216] <= 1
 frac[198,222] <= 1
 frac[42,33] <= 1
 frac[192,3] <= 1
 frac[10,54] <= 1
 frac[195,14] <= 1
 frac[89,202] <= 1
 frac[198,105] <= 1
 frac[181,47] <= 1
 frac[146,3] <= 1
 frac[164,169] <= 1
 frac[20,227] <= 1
 frac[166,70] <= 1
 frac[99,33] <= 1
 frac[120,154] <= 1
 frac[193,36] <= 1
 frac[191,90] <= 1
 frac[194,170] <= 1
 frac[196,23] <= 1
 frac[161,19] <= 1
 frac[22,74] <= 1
 frac[170,195] <= 1
 frac[27,189] <= 1
 frac[193,213] <= 1
 frac[194,221] <= 1
 frac[123,225] <= 1
 frac[47,182] <= 1
 frac[196,166] <= 1
 frac[197,48] <= 1
 frac[195,217] <= 1
 frac[195,102] <= 1
 frac[14,227] <= 1
 frac[197,193] <= 1
 frac[198,209] <= 1
 frac[163,113] <= 1
 frac[0,72] <= 1
 frac[18,158] <= 1
 frac[148,48] <= 1
 frac[24,126] <= 1
 frac[198,96] <= 1
 frac[166,121] <= 1
 frac[115,249] <= 1
 frac[25,225] <= 1
 frac[193,61] <= 1
 frac[191,67] <= 1
 frac[194,165] <= 1
 frac[50,43] <= 1
 frac[196,30] <= 1
 frac[145,178] <= 1
 frac[109,129] <= 1
 frac[78,30] <= 1
 frac[155,121] <= 1
 frac[193,174] <= 1
 frac[194,212] <= 1
 frac[195,222] <= 1
 frac[196,173] <= 1
 frac[197,57] <= 1
 frac[198,185] <= 1
 frac[52,170] <= 1
 frac[35,210] <= 1
 frac[55,143] <= 1
 frac[75,142] <= 1
 frac[148,72] <= 1
 frac[73,107] <= 1
 frac[194,71] <= 1
 frac[105,237] <= 1
 frac[195,111] <= 1
 frac[197,202] <= 1
 frac[192,17] <= 1
 frac[135,132] <= 1
 frac[158,240] <= 1
 frac[13,93] <= 1
 frac[198,91] <= 1
 frac[96,34] <= 1
 frac[97,148] <= 1
 frac[4,107] <= 1
 frac[192,128] <= 1
 frac[193,54] <= 1
 frac[174,16] <= 1
 frac[194,188] <= 1
 frac[104,34] <= 1
 frac[196,5] <= 1
 frac[142,249] <= 1
 frac[76,184] <= 1
 frac[130,74] <= 1
 frac[193,167] <= 1
 frac[194,47] <= 1
 frac[195,199] <= 1
 frac[196,148] <= 1
 frac[197,34] <= 1
 frac[15,47] <= 1
 frac[198,176] <= 1
 frac[53,152] <= 1
 frac[56,180] <= 1
 frac[2,48] <= 1
 frac[58,213] <= 1
 frac[62,78] <= 1
 frac[136,204] <= 1
 frac[173,57] <= 1
 frac[154,247] <= 1
 frac[194,94] <= 1
 frac[195,84] <= 1
 frac[197,179] <= 1
 frac[198,195] <= 1
 frac[187,101] <= 1
 frac[61,217] <= 1
 frac[192,104] <= 1
 frac[107,135] <= 1
 frac[198,82] <= 1
 frac[109,226] <= 1
 frac[182,184] <= 1
 frac[93,168] <= 1
 frac[192,135] <= 1
 frac[193,15] <= 1
 frac[194,183] <= 1
 frac[196,12] <= 1
 frac[2,152] <= 1
 frac[166,250] <= 1
 frac[4,225] <= 1
 frac[169,38] <= 1
 frac[99,165] <= 1
 frac[100,61] <= 1
 frac[27,128] <= 1
 frac[193,184] <= 1
 frac[194,38] <= 1
 frac[195,204] <= 1
 frac[68,70] <= 1
 frac[196,147] <= 1
 frac[197,43] <= 1
 frac[198,171] <= 1
 frac[173,161] <= 1
 frac[45,45] <= 1
 frac[132,240] <= 1
 frac[134,89] <= 1
 frac[7,229] <= 1
 frac[62,113] <= 1
 frac[10,91] <= 1
 frac[194,89] <= 1
 frac[195,93] <= 1
 frac[178,191] <= 1
 frac[197,188] <= 1
 frac[198,58] <= 1
 frac[21,142] <= 1
 frac[3,56] <= 1
 frac[192,111] <= 1
 frac[138,231] <= 1
 frac[159,240] <= 1
 frac[161,78] <= 1
 frac[198,85] <= 1
 frac[108,211] <= 1
 frac[75,87] <= 1
 frac[20,135] <= 1
 frac[111,198] <= 1
 frac[113,152] <= 1
 frac[188,219] <= 1
 frac[192,158] <= 1
 frac[193,0] <= 1
 frac[194,142] <= 1
 frac[65,93] <= 1
 frac[196,11] <= 1
 frac[33,116] <= 1
 frac[161,63] <= 1
 frac[56,211] <= 1
 frac[1,99] <= 1
 frac[182,34] <= 1
 frac[8,235] <= 1
 frac[193,177] <= 1
 frac[194,33] <= 1
 frac[195,181] <= 1
 frac[196,154] <= 1
 frac[197,20] <= 1
 frac[198,162] <= 1
 frac[78,226] <= 1
 frac[118,117] <= 1
 frac[10,66] <= 1
 frac[83,213] <= 1
 frac[174,252] <= 1
 frac[195,66] <= 1
 frac[194,80] <= 1
 frac[14,255] <= 1
 frac[197,165] <= 1
 frac[198,61] <= 1
 frac[77,26] <= 1
 frac[192,102] <= 1
 frac[193,104] <= 1
 frac[120,78] <= 1
 frac[198,76] <= 1
 frac[113,145] <= 1
 frac[186,185] <= 1
 frac[40,58] <= 1
 frac[116,186] <= 1
 frac[100,92] <= 1
 frac[192,149] <= 1
 frac[194,137] <= 1
 frac[196,114] <= 1
 frac[107,2] <= 1
 frac[71,24] <= 1
 frac[146,241] <= 1
 frac[96,166] <= 1
 frac[193,138] <= 1
 frac[191,244] <= 1
 frac[195,186] <= 1
 frac[194,56] <= 1
 frac[196,129] <= 1
 frac[179,75] <= 1
 frac[198,165] <= 1
 frac[17,100] <= 1
 frac[197,29] <= 1
 frac[103,86] <= 1
 frac[57,240] <= 1
 frac[170,49] <= 1
 frac[153,159] <= 1
 frac[138,6] <= 1
 frac[195,75] <= 1
 frac[12,230] <= 1
 frac[161,239] <= 1
 frac[197,174] <= 1
 frac[198,52] <= 1
 frac[178,177] <= 1
 frac[128,16] <= 1
 frac[37,250] <= 1
 frac[112,181] <= 1
 frac[59,69] <= 1
 frac[132,117] <= 1
 frac[79,142] <= 1
 frac[193,97] <= 1
 frac[157,155] <= 1
 frac[140,245] <= 1
 frac[126,111] <= 1
 frac[198,71] <= 1
 frac[15,216] <= 1
 frac[74,113] <= 1
 frac[38,75] <= 1
 frac[3,183] <= 1
 frac[109,205] <= 1
 frac[93,179] <= 1
 frac[39,91] <= 1
 frac[9,156] <= 1
 frac[192,236] <= 1
 frac[193,18] <= 1
 frac[194,128] <= 1
 frac[121,58] <= 1
 frac[50,198] <= 1
 frac[196,121] <= 1
 frac[197,117] <= 1
 frac[125,77] <= 1
 frac[65,47] <= 1
 frac[144,15] <= 1
 frac[95,159] <= 1
 frac[150,21] <= 1
 frac[153,247] <= 1
 frac[193,131] <= 1
 frac[176,197] <= 1
 frac[195,163] <= 1
 frac[29,242] <= 1
 frac[196,136] <= 1
 frac[197,6] <= 1
 frac[198,156] <= 1
 frac[19,68] <= 1
 frac[150,132] <= 1
 frac[138,1] <= 1
 frac[195,48] <= 1
 frac[178,168] <= 1
 frac[197,151] <= 1
 frac[198,47] <= 1
 frac[162,155] <= 1
 frac[192,116] <= 1
 frac[193,122] <= 1
 frac[45,99] <= 1
 frac[161,81] <= 1
 frac[113,131] <= 1
 frac[192,227] <= 1
 frac[193,235] <= 1
 frac[191,149] <= 1
 frac[194,155] <= 1
 frac[196,96] <= 1
 frac[197,126] <= 1
 frac[33,31] <= 1
 frac[147,19] <= 1
 frac[57,145] <= 1
 frac[169,2] <= 1
 frac[98,168] <= 1
 frac[152,198] <= 1
 frac[8,128] <= 1
 frac[193,132] <= 1
 frac[194,10] <= 1
 frac[195,168] <= 1
 frac[196,247] <= 1
 frac[197,15] <= 1
 frac[198,151] <= 1
 frac[19,77] <= 1
 frac[172,136] <= 1
 frac[195,57] <= 1
 frac[197,144] <= 1
 frac[198,38] <= 1
 frac[16,223] <= 1
 frac[6,3] <= 1
 frac[192,75] <= 1
 frac[135,222] <= 1
 frac[31,25] <= 1
 frac[106,90] <= 1
 frac[93,133] <= 1
 frac[188,255] <= 1
 frac[9,142] <= 1
 frac[192,250] <= 1
 frac[48,95] <= 1
 frac[104,12] <= 1
 frac[194,146] <= 1
 frac[174,62] <= 1
 frac[196,111] <= 1
 frac[81,153] <= 1
 frac[172,240] <= 1
 frac[27,101] <= 1
 frac[28,218] <= 1
 frac[194,5] <= 1
 frac[195,145] <= 1
 frac[86,31] <= 1
 frac[196,254] <= 1
 frac[197,8] <= 1
 frac[198,142] <= 1
 frac[111,13] <= 1
 frac[80,164] <= 1
 frac[61,158] <= 1
 frac[100,175] <= 1
 frac[101,39] <= 1
 frac[67,107] <= 1
 frac[195,62] <= 1
 frac[86,142] <= 1
 frac[141,38] <= 1
 frac[197,153] <= 1
 frac[198,25] <= 1
 frac[60,90] <= 1
 frac[192,66] <= 1
 frac[193,116] <= 1
 mem_tot_CPU1 <= 32768
 mem_tot_CPU2 <= 32768
 mem_tot_CPU3 <= 32768
 mem_tot_CPU4 <= 32768
 mem_tot_CPU5 <= 32768
 mem_tot_CPU6 <= 32768
 mem_tot_CPU7 <= 32768
 mem_tot_CPU8 <= 32768
 mem_tot_CPU9 <= 32768
 mem_tot_CPU10 <= 32768
 mem_tot_CPU11 <= 32768
 mem_tot_CPU12 <= 32768
 mem_tot_CPU13 <= 32768
 mem_tot_CPU14 <= 32768
 mem_tot_CPU15 <= 32768
 mem_tot_CPU16 <= 32768
 mem_tot_CPU17 <= 32768
 mem_tot_CPU18 <= 32768
 mem_tot_CPU19 <= 32768
 mem_tot_CPU20 <= 32768
 mem_tot_CPU21 <= 32768
 mem_tot_CPU22 <= 32768
 mem_tot_CPU23 <= 32768
 mem_tot_CPU24 <= 32768
 mem_tot_CPU25 <= 32768
 mem_tot_CPU26 <= 32768
 mem_tot_CPU27 <= 32768
 mem_tot_CPU28 <= 32768
 mem_tot_CPU29 <= 32768
 mem_tot_CPU30 <= 32768
 mem_tot_CPU31 <= 32768
 mem_tot_CPU32 <= 32768
 mem_tot_CPU33 <= 32768
 mem_tot_CPU34 <= 32768
 mem_tot_CPU35 <= 32768
 mem_tot_CPU36 <= 32768
 mem_tot_CPU37 <= 32768
 mem_tot_CPU38 <= 32768
 mem_tot_CPU39 <= 32768
 mem_tot_CPU40 <= 32768
 mem_tot_CPU41 <= 32768
 mem_tot_CPU42 <= 32768
 mem_tot_CPU43 <= 32768
 mem_tot_CPU44 <= 32768
 mem_tot_CPU45 <= 32768
 mem_tot_CPU46 <= 32768
 mem_tot_CPU47 <= 32768
 mem_tot_CPU48 <= 32768
 mem_tot_CPU49 <= 32768
 mem_tot_CPU50 <= 32768
 mem_tot_CPU51 <= 32768
 mem_tot_CPU52 <= 32768
 mem_tot_CPU53 <= 32768
 mem_tot_CPU54 <= 32768
 mem_tot_CPU55 <= 32768
 mem_tot_CPU56 <= 32768
 mem_tot_CPU57 <= 32768
 mem_tot_CPU58 <= 32768
 mem_tot_CPU59 <= 32768
 mem_tot_CPU60 <= 32768
 mem_tot_CPU61 <= 32768
 mem_tot_CPU62 <= 32768
 mem_tot_CPU63 <= 32768
 mem_tot_CPU64 <= 32768
 mem_tot_CPU65 <= 32768
 mem_tot_CPU66 <= 32768
 mem_tot_CPU67 <= 32768
 mem_tot_CPU68 <= 32768
 mem_tot_CPU69 <= 32768
 mem_tot_CPU70 <= 32768
 mem_tot_CPU71 <= 32768
 mem_tot_CPU72 <= 32768
 mem_tot_CPU73 <= 32768
 mem_tot_CPU74 <= 32768
 mem_tot_CPU75 <= 32768
 mem_tot_CPU76 <= 32768
 mem_tot_CPU77 <= 32768
 mem_tot_CPU78 <= 32768
 mem_tot_CPU79 <= 32768
 mem_tot_CPU80 <= 32768
 mem_tot_CPU81 <= 32768
 mem_tot_CPU82 <= 32768
 mem_tot_CPU83 <= 32768
 mem_tot_CPU84 <= 32768
 mem_tot_CPU85 <= 32768
 mem_tot_CPU86 <= 32768
 mem_tot_CPU87 <= 32768
 mem_tot_CPU88 <= 32768
 mem_tot_CPU89 <= 32768
 mem_tot_CPU90 <= 32768
 mem_tot_CPU91 <= 32768
 mem_tot_CPU92 <= 32768
 mem_tot_CPU93 <= 32768
 mem_tot_CPU94 <= 32768
 mem_tot_CPU95 <= 32768
 mem_tot_CPU96 <= 32768
 mem_tot_CPU97 <= 32768
 mem_tot_CPU98 <= 32768
 mem_tot_CPU99 <= 32768
 mem_tot_CPU100 <= 32768
 mem_tot_CPU101 <= 32768
 mem_tot_CPU102 <= 32768
 mem_tot_CPU103 <= 32768
 mem_tot_CPU104 <= 32768
 mem_tot_CPU105 <= 32768
 mem_tot_CPU106 <= 32768
 mem_tot_CPU107 <= 32768
 mem_tot_CPU108 <= 32768
 mem_tot_CPU109 <= 32768
 mem_tot_CPU110 <= 32768
 mem_tot_CPU111 <= 32768
 mem_tot_CPU112 <= 32768
 mem_tot_CPU113 <= 32768
 mem_tot_CPU114 <= 32768
 mem_tot_CPU115 <= 32768
 mem_tot_CPU116 <= 32768
 mem_tot_CPU117 <= 32768
 mem_tot_CPU118 <= 32768
 mem_tot_CPU119 <= 32768
 mem_tot_CPU120 <= 32768
 mem_tot_CPU121 <= 32768
 mem_tot_CPU122 <= 32768
 mem_tot_CPU123 <= 32768
 mem_tot_CPU124 <= 32768
 mem_tot_CPU125 <= 32768
 mem_tot_CPU126 <= 32768
 mem_tot_CPU127 <= 32768
 mem_tot_CPU128 <= 32768
 mem_tot_CPU129 <= 32768
 mem_tot_CPU130 <= 32768
 mem_tot_CPU131 <= 32768
 mem_tot_CPU132 <= 32768
 mem_tot_CPU133 <= 32768
 mem_tot_CPU134 <= 32768
 mem_tot_CPU135 <= 32768
 mem_tot_CPU136 <= 32768
 mem_tot_CPU137 <= 32768
 mem_tot_CPU138 <= 32768
 mem_tot_CPU139 <= 32768
 mem_tot_CPU140 <= 32768
 mem_tot_CPU141 <= 32768
 mem_tot_CPU142 <= 32768
 mem_tot_CPU143 <= 32768
 mem_tot_CPU144 <= 32768
 mem_tot_CPU145 <= 32768
 mem_tot_CPU146 <= 32768
 mem_tot_CPU147 <= 32768
 mem_tot_CPU148 <= 32768
 mem_tot_CPU149 <= 32768
 mem_tot_CPU150 <= 32768
 mem_tot_CPU151 <= 32768
 mem_tot_CPU152 <= 32768
 mem_tot_CPU153 <= 32768
 mem_tot_CPU154 <= 32768
 mem_tot_CPU155 <= 32768
 mem_tot_CPU156 <= 32768
 mem_tot_CPU157 <= 32768
 mem_tot_CPU158 <= 32768
 mem_tot_CPU159 <= 32768
 mem_tot_CPU160 <= 32768
 mem_tot_CPU161 <= 32768
 mem_tot_CPU162 <= 32768
 mem_tot_CPU163 <= 32768
 mem_tot_CPU164 <= 32768
 mem_tot_CPU165 <= 32768
 mem_tot_CPU166 <= 32768
 mem_tot_CPU167 <= 32768
 mem_tot_CPU168 <= 32768
 mem_tot_CPU169 <= 32768
 mem_tot_CPU170 <= 32768
 mem_tot_CPU171 <= 32768
 mem_tot_CPU172 <= 32768
 mem_tot_CPU173 <= 32768
 mem_tot_CPU174 <= 32768
 mem_tot_CPU175 <= 32768
 mem_tot_CPU176 <= 32768
 mem_tot_CPU177 <= 32768
 mem_tot_CPU178 <= 32768
 mem_tot_CPU179 <= 32768
 mem_tot_CPU180 <= 32768
 mem_tot_CPU181 <= 32768
 mem_tot_CPU182 <= 32768
 mem_tot_CPU183 <= 32768
 mem_tot_CPU184 <= 32768
 mem_tot_CPU185 <= 32768
 mem_tot_CPU186 <= 32768
 mem_tot_CPU187 <= 32768
 mem_tot_CPU188 <= 32768
 mem_tot_CPU189 <= 32768
 mem_tot_CPU190 <= 32768
 mem_tot_CPU191 <= 32768
 mem_tot_CPU192 <= 32768
 mem_tot_tor_P41 <= 576
 mem_tot_tor_P42 <= 576
 mem_tot_tor_P43 <= 576
 mem_tot_tor_P44 <= 576
 mem_tot_l1_P41 <= 576
 mem_tot_l1_P42 <= 576
 mem_tot_l2_P4 <= 576
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_tor_P41 , mem_tot_tor_P42 ,
   mem_tot_tor_P43 , mem_tot_tor_P44 , mem_tot_l1_P41 , mem_tot_l1_P42 ,
   mem_tot_l2_P4 )
 mem_overall_CPU: max_mem_CPU = MAX ( mem_tot_CPU1 , mem_tot_CPU2 ,
   mem_tot_CPU3 , mem_tot_CPU4 , mem_tot_CPU5 , mem_tot_CPU6 ,
   mem_tot_CPU7 , mem_tot_CPU8 , mem_tot_CPU9 , mem_tot_CPU10 ,
   mem_tot_CPU11 , mem_tot_CPU12 , mem_tot_CPU13 , mem_tot_CPU14 ,
   mem_tot_CPU15 , mem_tot_CPU16 , mem_tot_CPU17 , mem_tot_CPU18 ,
   mem_tot_CPU19 , mem_tot_CPU20 , mem_tot_CPU21 , mem_tot_CPU22 ,
   mem_tot_CPU23 , mem_tot_CPU24 , mem_tot_CPU25 , mem_tot_CPU26 ,
   mem_tot_CPU27 , mem_tot_CPU28 , mem_tot_CPU29 , mem_tot_CPU30 ,
   mem_tot_CPU31 , mem_tot_CPU32 , mem_tot_CPU33 , mem_tot_CPU34 ,
   mem_tot_CPU35 , mem_tot_CPU36 , mem_tot_CPU37 , mem_tot_CPU38 ,
   mem_tot_CPU39 , mem_tot_CPU40 , mem_tot_CPU41 , mem_tot_CPU42 ,
   mem_tot_CPU43 , mem_tot_CPU44 , mem_tot_CPU45 , mem_tot_CPU46 ,
   mem_tot_CPU47 , mem_tot_CPU48 , mem_tot_CPU49 , mem_tot_CPU50 ,
   mem_tot_CPU51 , mem_tot_CPU52 , mem_tot_CPU53 , mem_tot_CPU54 ,
   mem_tot_CPU55 , mem_tot_CPU56 , mem_tot_CPU57 , mem_tot_CPU58 ,
   mem_tot_CPU59 , mem_tot_CPU60 , mem_tot_CPU61 , mem_tot_CPU62 ,
   mem_tot_CPU63 , mem_tot_CPU64 , mem_tot_CPU65 , mem_tot_CPU66 ,
   mem_tot_CPU67 , mem_tot_CPU68 , mem_tot_CPU69 , mem_tot_CPU70 ,
   mem_tot_CPU71 , mem_tot_CPU72 , mem_tot_CPU73 , mem_tot_CPU74 ,
   mem_tot_CPU75 , mem_tot_CPU76 , mem_tot_CPU77 , mem_tot_CPU78 ,
   mem_tot_CPU79 , mem_tot_CPU80 , mem_tot_CPU81 , mem_tot_CPU82 ,
   mem_tot_CPU83 , mem_tot_CPU84 , mem_tot_CPU85 , mem_tot_CPU86 ,
   mem_tot_CPU87 , mem_tot_CPU88 , mem_tot_CPU89 , mem_tot_CPU90 ,
   mem_tot_CPU91 , mem_tot_CPU92 , mem_tot_CPU93 , mem_tot_CPU94 ,
   mem_tot_CPU95 , mem_tot_CPU96 , mem_tot_CPU97 , mem_tot_CPU98 ,
   mem_tot_CPU99 , mem_tot_CPU100 , mem_tot_CPU101 , mem_tot_CPU102 ,
   mem_tot_CPU103 , mem_tot_CPU104 , mem_tot_CPU105 , mem_tot_CPU106 ,
   mem_tot_CPU107 , mem_tot_CPU108 , mem_tot_CPU109 , mem_tot_CPU110 ,
   mem_tot_CPU111 , mem_tot_CPU112 , mem_tot_CPU113 , mem_tot_CPU114 ,
   mem_tot_CPU115 , mem_tot_CPU116 , mem_tot_CPU117 , mem_tot_CPU118 ,
   mem_tot_CPU119 , mem_tot_CPU120 , mem_tot_CPU121 , mem_tot_CPU122 ,
   mem_tot_CPU123 , mem_tot_CPU124 , mem_tot_CPU125 , mem_tot_CPU126 ,
   mem_tot_CPU127 , mem_tot_CPU128 , mem_tot_CPU129 , mem_tot_CPU130 ,
   mem_tot_CPU131 , mem_tot_CPU132 , mem_tot_CPU133 , mem_tot_CPU134 ,
   mem_tot_CPU135 , mem_tot_CPU136 , mem_tot_CPU137 , mem_tot_CPU138 ,
   mem_tot_CPU139 , mem_tot_CPU140 , mem_tot_CPU141 , mem_tot_CPU142 ,
   mem_tot_CPU143 , mem_tot_CPU144 , mem_tot_CPU145 , mem_tot_CPU146 ,
   mem_tot_CPU147 , mem_tot_CPU148 , mem_tot_CPU149 , mem_tot_CPU150 ,
   mem_tot_CPU151 , mem_tot_CPU152 , mem_tot_CPU153 , mem_tot_CPU154 ,
   mem_tot_CPU155 , mem_tot_CPU156 , mem_tot_CPU157 , mem_tot_CPU158 ,
   mem_tot_CPU159 , mem_tot_CPU160 , mem_tot_CPU161 , mem_tot_CPU162 ,
   mem_tot_CPU163 , mem_tot_CPU164 , mem_tot_CPU165 , mem_tot_CPU166 ,
   mem_tot_CPU167 , mem_tot_CPU168 , mem_tot_CPU169 , mem_tot_CPU170 ,
   mem_tot_CPU171 , mem_tot_CPU172 , mem_tot_CPU173 , mem_tot_CPU174 ,
   mem_tot_CPU175 , mem_tot_CPU176 , mem_tot_CPU177 , mem_tot_CPU178 ,
   mem_tot_CPU179 , mem_tot_CPU180 , mem_tot_CPU181 , mem_tot_CPU182 ,
   mem_tot_CPU183 , mem_tot_CPU184 , mem_tot_CPU185 , mem_tot_CPU186 ,
   mem_tot_CPU187 , mem_tot_CPU188 , mem_tot_CPU189 , mem_tot_CPU190 ,
   mem_tot_CPU191 , mem_tot_CPU192 )
End

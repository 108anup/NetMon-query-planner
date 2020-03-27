\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  OBJ0: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ1: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_rows_P4: Priority=80 Weight=1 AbsTol=0 RelTol=0
   rows_tot_l2_P4
  P4_rows_load: Priority=70 Weight=1 AbsTol=0 RelTol=0
   max_rows_P4
  OBJ6: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ7: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_mem_P4: Priority=40 Weight=1 AbsTol=0 RelTol=0
   mem_tot_l2_P4
  P4_load_mem: Priority=30 Weight=1 AbsTol=0 RelTol=0
   max_mem_P4
Subject To
 cov_0_1: frac[0,1] >= 0.2
 cov_1_1: frac[0,1] >= 0.2
 cov_2_9: frac[0,9] >= 0.2928312467973777
 cov_3_9: frac[0,9] >= 0.2928312467973777
 cov_4_46: frac[0,46] >= 0.4964156233986888
 cov_5_46: frac[0,46] >= 0.4964156233986888
 cov_6_55: frac[0,55] >= 0.1
 cov_7_55: frac[0,55] >= 0.1
 cov_8_55: frac[0,55] >= 0.1
 cov_9_57: frac[0,57] >= 0.0964156233986888
 cov_10_57: frac[0,57] >= 0.0964156233986888
 cov_11_56: frac[0,56] >= 0.3982078116993445
 cov_12_57: frac[0,57] >= 0.0964156233986888
 cov_13_56: frac[0,56] >= 0.3982078116993445
 cov_14_66: frac[0,66] >= 0.3964156233986889
 cov_15_66: frac[0,66] >= 0.3964156233986889
 cov_16_66: frac[0,66] >= 0.3964156233986889
 cov_17_66: frac[0,66] >= 0.3964156233986889
 cov_18_66: frac[0,66] >= 0.3964156233986889
 cov_19_82: frac[0,82] >= 0.3982078116993444
 cov_20_82: frac[0,82] >= 0.3982078116993444
 cov_21_82: frac[0,82] >= 0.3982078116993444
 cov_22_95: frac[0,95] >= 0.2928312467973777
 cov_23_95: frac[0,95] >= 0.2928312467973777
 cov_24_106: frac[0,106] >= 0.3179509379688777
 cov_25_106: frac[0,106] >= 0.3179509379688777
 cov_26_115: frac[0,115] >= 0.4964156233986888
 cov_27_115: frac[0,115] >= 0.4964156233986888
 cov_28_130: frac[0,130] >= 0.6964156233986889
 cov_29_130: frac[0,130] >= 0.6964156233986889
 cov_30_141: frac[0,141] >= 0.5928312467973778
 cov_31_141: frac[0,141] >= 0.5928312467973778
 cov_32_164: frac[0,164] >= 0.5964156233986888
 cov_33_164: frac[0,164] >= 0.5964156233986888
 cov_34_168: frac[0,168] >= 0.2928312467973777
 cov_35_168: frac[0,168] >= 0.2928312467973777
 cov_36_178: frac[0,178] >= 0.8
 cov_37_178: frac[0,178] >= 0.8
 cov_38_190: frac[0,190] >= 0.7964156233986889
 cov_39_191: frac[0,191] >= 0.6964156233986889
 cov_40_191: frac[0,191] >= 0.6964156233986889
 cov_41_190: frac[0,190] >= 0.7964156233986889
 cov_42_190: frac[0,190] >= 0.7964156233986889
 cov_43_190: frac[0,190] >= 0.7964156233986889
 cov_44_191: frac[0,191] >= 0.6964156233986889
 cov_45_202: frac[0,202] >= 0.4928312467973777
 cov_46_202: frac[0,202] >= 0.4928312467973777
 cov_47_202: frac[0,202] >= 0.4928312467973777
 cov_48_245: frac[0,245] >= 0.9
 cov_49_245: frac[0,245] >= 0.9
 cov_50_254: frac[0,254] >= 0.8964156233986887
 cov_51_254: frac[0,254] >= 0.8964156233986887
 cov_52_278: frac[0,278] >= 0.4928312467973778
 cov_53_278: frac[0,278] >= 0.4928312467973778
 cov_54_286: frac[0,286] >= 0.4928312467973778
 cov_55_286: frac[0,286] >= 0.4928312467973778
 cov_56_288: frac[0,288] >= 0.6964156233986887
 cov_57_288: frac[0,288] >= 0.6964156233986887
 cov_58_346: frac[0,346] >= 0.8
 cov_59_346: frac[0,346] >= 0.8
 cov_60_346: frac[0,346] >= 0.8
 cov_61_379: frac[0,379] >= 0.2964156233986889
 cov_62_379: frac[0,379] >= 0.2964156233986889
 cov_63_379: frac[0,379] >= 0.2964156233986889
 cov_64_419: frac[0,419] >= 0.796415623398689
 cov_65_419: frac[0,419] >= 0.796415623398689
 cov_66_419: frac[0,419] >= 0.796415623398689
 cov_67_423: frac[0,423] >= 0.3
 cov_68_423: frac[0,423] >= 0.3
 cov_69_425: frac[0,425] >= 0.4964156233986888
 cov_70_425: frac[0,425] >= 0.4964156233986888
 cov_71_437: frac[0,437] >= 0.4964156233986888
 cov_72_437: frac[0,437] >= 0.4964156233986888
 cov_73_437: frac[0,437] >= 0.4964156233986888
 cov_74_437: frac[0,437] >= 0.4964156233986888
 cov_75_449: frac[0,449] >= 0.4928312467973778
 cov_76_449: frac[0,449] >= 0.4928312467973778
 cov_77_460: frac[0,460] >= 0.3928312467973776
 cov_78_460: frac[0,460] >= 0.3928312467973776
 accuracy_0_437: - 5.09765625 frac[0,437] + mem[0,437] = 0
 accuracy_0_115: - 5.09765625 frac[0,115] + mem[0,115] = 0
 accuracy_0_56: - 5.09765625 frac[0,56] + mem[0,56] = 0
 accuracy_0_82: - 5.09765625 frac[0,82] + mem[0,82] = 0
 accuracy_0_460: - 5.09765625 frac[0,460] + mem[0,460] = 0
 accuracy_0_164: - 5.09765625 frac[0,164] + mem[0,164] = 0
 accuracy_0_449: - 5.09765625 frac[0,449] + mem[0,449] = 0
 accuracy_0_379: - 5.09765625 frac[0,379] + mem[0,379] = 0
 accuracy_0_288: - 5.09765625 frac[0,288] + mem[0,288] = 0
 accuracy_0_346: - 5.09765625 frac[0,346] + mem[0,346] = 0
 accuracy_0_57: - 5.09765625 frac[0,57] + mem[0,57] = 0
 accuracy_0_178: - 5.09765625 frac[0,178] + mem[0,178] = 0
 accuracy_0_106: - 5.09765625 frac[0,106] + mem[0,106] = 0
 accuracy_0_95: - 5.09765625 frac[0,95] + mem[0,95] = 0
 accuracy_0_190: - 5.09765625 frac[0,190] + mem[0,190] = 0
 accuracy_0_168: - 5.09765625 frac[0,168] + mem[0,168] = 0
 accuracy_0_66: - 5.09765625 frac[0,66] + mem[0,66] = 0
 accuracy_0_55: - 5.09765625 frac[0,55] + mem[0,55] = 0
 accuracy_0_278: - 5.09765625 frac[0,278] + mem[0,278] = 0
 accuracy_0_245: - 5.09765625 frac[0,245] + mem[0,245] = 0
 accuracy_0_202: - 5.09765625 frac[0,202] + mem[0,202] = 0
 accuracy_0_191: - 5.09765625 frac[0,191] + mem[0,191] = 0
 accuracy_0_425: - 5.09765625 frac[0,425] + mem[0,425] = 0
 accuracy_0_286: - 5.09765625 frac[0,286] + mem[0,286] = 0
 accuracy_0_1: - 5.09765625 frac[0,1] + mem[0,1] = 0
 accuracy_0_46: - 5.09765625 frac[0,46] + mem[0,46] = 0
 accuracy_0_141: - 5.09765625 frac[0,141] + mem[0,141] = 0
 accuracy_0_423: - 5.09765625 frac[0,423] + mem[0,423] = 0
 accuracy_0_9: - 5.09765625 frac[0,9] + mem[0,9] = 0
 accuracy_0_254: - 5.09765625 frac[0,254] + mem[0,254] = 0
 accuracy_0_419: - 5.09765625 frac[0,419] + mem[0,419] = 0
 accuracy_0_130: - 5.09765625 frac[0,130] + mem[0,130] = 0
 mem_tot_l2_P4: - mem[0,437] - mem[0,115] - mem[0,56] - mem[0,82]
   - mem[0,460] - mem[0,164] - mem[0,449] - mem[0,379] - mem[0,288]
   - mem[0,346] - mem[0,57] - mem[0,178] - mem[0,106] - mem[0,95]
   - mem[0,190] - mem[0,168] - mem[0,66] - mem[0,55] - mem[0,278]
   - mem[0,245] - mem[0,202] - mem[0,191] - mem[0,425] - mem[0,286]
   - mem[0,1] - mem[0,46] - mem[0,141] - mem[0,423] - mem[0,9] - mem[0,254]
   - mem[0,419] - mem[0,130] + mem_tot_l2_P4 = 0
 R112: - 0.00173611111111111 mem_tot_l2_P4 + normalized_mem_tot_l2_P4 = 0
 rows_tot_l2_P4: - 3 frac[0,437] - 3 frac[0,115] - 3 frac[0,56]
   - 3 frac[0,82] - 3 frac[0,460] - 3 frac[0,164] - 3 frac[0,449]
   - 3 frac[0,379] - 3 frac[0,288] - 3 frac[0,346] - 3 frac[0,57]
   - 3 frac[0,178] - 3 frac[0,106] - 3 frac[0,95] - 3 frac[0,190]
   - 3 frac[0,168] - 3 frac[0,66] - 3 frac[0,55] - 3 frac[0,278]
   - 3 frac[0,245] - 3 frac[0,202] - 3 frac[0,191] - 3 frac[0,425]
   - 3 frac[0,286] - 3 frac[0,1] - 3 frac[0,46] - 3 frac[0,141]
   - 3 frac[0,423] - 3 frac[0,9] - 3 frac[0,254] - 3 frac[0,419]
   - 3 frac[0,130] + rows_tot_l2_P4 = 0
 R114: - 0.0208333333333333 rows_tot_l2_P4 + normalized_rows_tot_l2_P4 = 0
 row_capacity_l2_P4: rows_tot_l2_P4 <= 48
Bounds
 frac[0,437] <= 1
 frac[0,115] <= 1
 frac[0,56] <= 1
 frac[0,82] <= 1
 frac[0,460] <= 1
 frac[0,164] <= 1
 frac[0,449] <= 1
 frac[0,379] <= 1
 frac[0,288] <= 1
 frac[0,346] <= 1
 frac[0,57] <= 1
 frac[0,178] <= 1
 frac[0,106] <= 1
 frac[0,95] <= 1
 frac[0,190] <= 1
 frac[0,168] <= 1
 frac[0,66] <= 1
 frac[0,55] <= 1
 frac[0,278] <= 1
 frac[0,245] <= 1
 frac[0,202] <= 1
 frac[0,191] <= 1
 frac[0,425] <= 1
 frac[0,286] <= 1
 frac[0,1] <= 1
 frac[0,46] <= 1
 frac[0,141] <= 1
 frac[0,423] <= 1
 frac[0,9] <= 1
 frac[0,254] <= 1
 frac[0,419] <= 1
 frac[0,130] <= 1
 mem_tot_l2_P4 <= 576
 normalized_mem_tot_l2_P4 <= 1
 normalized_rows_tot_l2_P4 <= 1
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_l2_P4 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_l2_P4 )
End

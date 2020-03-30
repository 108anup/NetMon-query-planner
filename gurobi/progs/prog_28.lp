\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU_Cluster: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU626 + rows_tot_CPU631 + rows_tot_CPU632 + rows_tot_CPU633
   + rows_tot_CPU635 + rows_tot_CPU639 + rows_tot_CPU640 + rows_tot_CPU642
   + rows_tot_CPU643 + rows_tot_CPU645 + rows_tot_CPU651 + rows_tot_CPU653
   + rows_tot_CPU655 + rows_tot_CPU656 + rows_tot_CPU657 + rows_tot_CPU659
   + rows_tot_CPU660 + rows_tot_CPU666 + rows_tot_CPU668 + rows_tot_CPU669
   + rows_tot_CPU670 + rows_tot_CPU672
  CPU_Cluster_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU_Cluster
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ4: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ5: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_mem_CPU_Cluster: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU626 + mem_tot_CPU631 + mem_tot_CPU632 + mem_tot_CPU633
   + mem_tot_CPU635 + mem_tot_CPU639 + mem_tot_CPU640 + mem_tot_CPU642
   + mem_tot_CPU643 + mem_tot_CPU645 + mem_tot_CPU651 + mem_tot_CPU653
   + mem_tot_CPU655 + mem_tot_CPU656 + mem_tot_CPU657 + mem_tot_CPU659
   + mem_tot_CPU660 + mem_tot_CPU666 + mem_tot_CPU668 + mem_tot_CPU669
   + mem_tot_CPU670 + mem_tot_CPU672
  CPU_Cluster_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
Subject To
 cov_0_3: frac[18,3] >= 0.2035843766013112
 cov_1_3: frac[18,3] >= 0.2035843766013112
 cov_2_35: frac[10,35] >= 0.2035843766013111
 cov_3_35: frac[10,35] >= 0.2035843766013111
 cov_4_50: frac[13,50] >= 0.2035843766013112
 cov_5_50: frac[13,50] >= 0.2035843766013112
 cov_6_72: frac[21,72] >= 0.2035843766013112
 cov_7_142: frac[7,142] >= 0.2035843766013112
 cov_8_142: frac[7,142] >= 0.2035843766013112
 cov_9_154: frac[0,154] >= 0.2035843766013112
 cov_10_154: frac[0,154] >= 0.2035843766013112
 cov_11_202: frac[20,202] + frac[1,202] >= 0.4071687532026223
 cov_12_202: frac[1,202] >= 0.2035843766013111
 cov_13_208: frac[14,208] >= 0.2
 cov_14_208: frac[14,208] >= 0.2
 cov_15_211: frac[14,211] >= 0.00358437660131122
 cov_16_243: frac[2,243] >= 0.2035843766013112
 cov_17_243: frac[2,243] >= 0.2035843766013112
 cov_18_259: frac[12,259] >= 0.2035843766013112
 cov_19_259: frac[12,259] >= 0.2035843766013112
 cov_20_259: frac[12,259] >= 0.2035843766013112
 cov_21_259: frac[12,259] >= 0.2035843766013112
 cov_22_278: frac[9,278] >= 0.1035843766013111
 cov_23_277: frac[9,277] >= 0.1
 cov_24_277: frac[9,277] >= 0.1
 cov_25_432: frac[4,432] >= 0.2035843766013112
 cov_26_437: frac[16,437] >= 0.2035843766013112
 cov_27_437: frac[16,437] >= 0.2035843766013112
 cov_28_457: frac[8,457] >= 0.1
 cov_29_464: frac[17,464] >= 0.00301022
 cov_30_472: frac[3,472] >= 0.2035843766013112
 accuracy_14_211: - 5.09765625 frac[14,211] + mem[14,211] = 0
 accuracy_9_278: - 5.09765625 frac[9,278] + mem[9,278] = 0
 accuracy_20_202: - 5.09765625 frac[20,202] + mem[20,202] = 0
 accuracy_13_50: - 5.09765625 frac[13,50] + mem[13,50] = 0
 accuracy_14_208: - 5.09765625 frac[14,208] + mem[14,208] = 0
 accuracy_9_277: - 5.09765625 frac[9,277] + mem[9,277] = 0
 accuracy_4_432: - 5.09765625 frac[4,432] + mem[4,432] = 0
 accuracy_10_35: - 5.09765625 frac[10,35] + mem[10,35] = 0
 accuracy_12_259: - 5.09765625 frac[12,259] + mem[12,259] = 0
 accuracy_7_142: - 5.09765625 frac[7,142] + mem[7,142] = 0
 accuracy_2_243: - 5.09765625 frac[2,243] + mem[2,243] = 0
 accuracy_8_457: - 5.09765625 frac[8,457] + mem[8,457] = 0
 accuracy_17_464: - 5.09765625 frac[17,464] + mem[17,464] = 0
 accuracy_3_472: - 5.09765625 frac[3,472] + mem[3,472] = 0
 accuracy_18_3: - 5.09765625 frac[18,3] + mem[18,3] = 0
 accuracy_0_154: - 5.09765625 frac[0,154] + mem[0,154] = 0
 accuracy_21_72: - 5.09765625 frac[21,72] + mem[21,72] = 0
 accuracy_16_437: - 5.09765625 frac[16,437] + mem[16,437] = 0
 accuracy_1_202: - 5.09765625 frac[1,202] + mem[1,202] = 0
 mem_tot_CPU626: - mem[0,154] + mem_tot_CPU626 = 0
 R51: - 3.05176e-05 mem_tot_CPU626 + normalized_mem_tot_CPU626 = 0
 rows_tot_CPU626: - 3 frac[0,154] + rows_tot_CPU626 = 0
 R53: - 0.0833333333333333 rows_tot_CPU626 + normalized_rows_tot_CPU626
   = 0
 mem_tot_CPU631: - mem[1,202] + mem_tot_CPU631 = 0
 R55: - 3.05176e-05 mem_tot_CPU631 + normalized_mem_tot_CPU631 = 0
 rows_tot_CPU631: - 3 frac[1,202] + rows_tot_CPU631 = 0
 R57: - 0.0833333333333333 rows_tot_CPU631 + normalized_rows_tot_CPU631
   = 0
 mem_tot_CPU632: - mem[2,243] + mem_tot_CPU632 = 0
 R59: - 3.05176e-05 mem_tot_CPU632 + normalized_mem_tot_CPU632 = 0
 rows_tot_CPU632: - 3 frac[2,243] + rows_tot_CPU632 = 0
 R61: - 0.0833333333333333 rows_tot_CPU632 + normalized_rows_tot_CPU632
   = 0
 mem_tot_CPU633: - mem[3,472] + mem_tot_CPU633 = 0
 R63: - 3.05176e-05 mem_tot_CPU633 + normalized_mem_tot_CPU633 = 0
 rows_tot_CPU633: - 3 frac[3,472] + rows_tot_CPU633 = 0
 R65: - 0.0833333333333333 rows_tot_CPU633 + normalized_rows_tot_CPU633
   = 0
 mem_tot_CPU635: - mem[4,432] + mem_tot_CPU635 = 0
 R67: - 3.05176e-05 mem_tot_CPU635 + normalized_mem_tot_CPU635 = 0
 rows_tot_CPU635: - 3 frac[4,432] + rows_tot_CPU635 = 0
 R69: - 0.0833333333333333 rows_tot_CPU635 + normalized_rows_tot_CPU635
   = 0
 mem_tot_CPU639: mem_tot_CPU639 = 0
 R71: - 3.05176e-05 mem_tot_CPU639 + normalized_mem_tot_CPU639 = 0
 rows_tot_CPU639: rows_tot_CPU639 = 0
 R73: - 0.0833333333333333 rows_tot_CPU639 + normalized_rows_tot_CPU639
   = 0
 mem_tot_CPU640: mem_tot_CPU640 = 0
 R75: - 3.05176e-05 mem_tot_CPU640 + normalized_mem_tot_CPU640 = 0
 rows_tot_CPU640: rows_tot_CPU640 = 0
 R77: - 0.0833333333333333 rows_tot_CPU640 + normalized_rows_tot_CPU640
   = 0
 mem_tot_CPU642: - mem[7,142] + mem_tot_CPU642 = 0
 R79: - 3.05176e-05 mem_tot_CPU642 + normalized_mem_tot_CPU642 = 0
 rows_tot_CPU642: - 3 frac[7,142] + rows_tot_CPU642 = 0
 R81: - 0.0833333333333333 rows_tot_CPU642 + normalized_rows_tot_CPU642
   = 0
 mem_tot_CPU643: - mem[8,457] + mem_tot_CPU643 = 0
 R83: - 3.05176e-05 mem_tot_CPU643 + normalized_mem_tot_CPU643 = 0
 rows_tot_CPU643: - 3 frac[8,457] + rows_tot_CPU643 = 0
 R85: - 0.0833333333333333 rows_tot_CPU643 + normalized_rows_tot_CPU643
   = 0
 mem_tot_CPU645: - mem[9,278] - mem[9,277] + mem_tot_CPU645 = 0
 R87: - 3.05176e-05 mem_tot_CPU645 + normalized_mem_tot_CPU645 = 0
 rows_tot_CPU645: - 3 frac[9,278] - 3 frac[9,277] + rows_tot_CPU645 = 0
 R89: - 0.0833333333333333 rows_tot_CPU645 + normalized_rows_tot_CPU645
   = 0
 mem_tot_CPU651: - mem[10,35] + mem_tot_CPU651 = 0
 R91: - 3.05176e-05 mem_tot_CPU651 + normalized_mem_tot_CPU651 = 0
 rows_tot_CPU651: - 3 frac[10,35] + rows_tot_CPU651 = 0
 R93: - 0.0833333333333333 rows_tot_CPU651 + normalized_rows_tot_CPU651
   = 0
 mem_tot_CPU653: mem_tot_CPU653 = 0
 R95: - 3.05176e-05 mem_tot_CPU653 + normalized_mem_tot_CPU653 = 0
 rows_tot_CPU653: rows_tot_CPU653 = 0
 R97: - 0.0833333333333333 rows_tot_CPU653 + normalized_rows_tot_CPU653
   = 0
 mem_tot_CPU655: - mem[12,259] + mem_tot_CPU655 = 0
 R99: - 3.05176e-05 mem_tot_CPU655 + normalized_mem_tot_CPU655 = 0
 rows_tot_CPU655: - 3 frac[12,259] + rows_tot_CPU655 = 0
 R101: - 0.0833333333333333 rows_tot_CPU655 + normalized_rows_tot_CPU655
   = 0
 mem_tot_CPU656: - mem[13,50] + mem_tot_CPU656 = 0
 R103: - 3.05176e-05 mem_tot_CPU656 + normalized_mem_tot_CPU656 = 0
 rows_tot_CPU656: - 3 frac[13,50] + rows_tot_CPU656 = 0
 R105: - 0.0833333333333333 rows_tot_CPU656 + normalized_rows_tot_CPU656
   = 0
 mem_tot_CPU657: - mem[14,211] - mem[14,208] + mem_tot_CPU657 = 0
 R107: - 3.05176e-05 mem_tot_CPU657 + normalized_mem_tot_CPU657 = 0
 rows_tot_CPU657: - 3 frac[14,211] - 3 frac[14,208] + rows_tot_CPU657 = 0
 R109: - 0.0833333333333333 rows_tot_CPU657 + normalized_rows_tot_CPU657
   = 0
 mem_tot_CPU659: mem_tot_CPU659 = 0
 R111: - 3.05176e-05 mem_tot_CPU659 + normalized_mem_tot_CPU659 = 0
 rows_tot_CPU659: rows_tot_CPU659 = 0
 R113: - 0.0833333333333333 rows_tot_CPU659 + normalized_rows_tot_CPU659
   = 0
 mem_tot_CPU660: - mem[16,437] + mem_tot_CPU660 = 0
 R115: - 3.05176e-05 mem_tot_CPU660 + normalized_mem_tot_CPU660 = 0
 rows_tot_CPU660: - 3 frac[16,437] + rows_tot_CPU660 = 0
 R117: - 0.0833333333333333 rows_tot_CPU660 + normalized_rows_tot_CPU660
   = 0
 mem_tot_CPU666: - mem[17,464] + mem_tot_CPU666 = 0
 R119: - 3.05176e-05 mem_tot_CPU666 + normalized_mem_tot_CPU666 = 0
 rows_tot_CPU666: - 3 frac[17,464] + rows_tot_CPU666 = 0
 R121: - 0.0833333333333333 rows_tot_CPU666 + normalized_rows_tot_CPU666
   = 0
 mem_tot_CPU668: - mem[18,3] + mem_tot_CPU668 = 0
 R123: - 3.05176e-05 mem_tot_CPU668 + normalized_mem_tot_CPU668 = 0
 rows_tot_CPU668: - 3 frac[18,3] + rows_tot_CPU668 = 0
 R125: - 0.0833333333333333 rows_tot_CPU668 + normalized_rows_tot_CPU668
   = 0
 mem_tot_CPU669: mem_tot_CPU669 = 0
 R127: - 3.05176e-05 mem_tot_CPU669 + normalized_mem_tot_CPU669 = 0
 rows_tot_CPU669: rows_tot_CPU669 = 0
 R129: - 0.0833333333333333 rows_tot_CPU669 + normalized_rows_tot_CPU669
   = 0
 mem_tot_CPU670: - mem[20,202] + mem_tot_CPU670 = 0
 R131: - 3.05176e-05 mem_tot_CPU670 + normalized_mem_tot_CPU670 = 0
 rows_tot_CPU670: - 3 frac[20,202] + rows_tot_CPU670 = 0
 R133: - 0.0833333333333333 rows_tot_CPU670 + normalized_rows_tot_CPU670
   = 0
 mem_tot_CPU672: - mem[21,72] + mem_tot_CPU672 = 0
 R135: - 3.05176e-05 mem_tot_CPU672 + normalized_mem_tot_CPU672 = 0
 rows_tot_CPU672: - 3 frac[21,72] + rows_tot_CPU672 = 0
 R137: - 0.0833333333333333 rows_tot_CPU672 + normalized_rows_tot_CPU672
   = 0
 row_capacity_CPU626: rows_tot_CPU626 <= 12
 row_capacity_CPU631: rows_tot_CPU631 <= 12
 row_capacity_CPU632: rows_tot_CPU632 <= 12
 row_capacity_CPU633: rows_tot_CPU633 <= 12
 row_capacity_CPU635: rows_tot_CPU635 <= 12
 row_capacity_CPU639: rows_tot_CPU639 <= 12
 row_capacity_CPU640: rows_tot_CPU640 <= 12
 row_capacity_CPU642: rows_tot_CPU642 <= 12
 row_capacity_CPU643: rows_tot_CPU643 <= 12
 row_capacity_CPU645: rows_tot_CPU645 <= 12
 row_capacity_CPU651: rows_tot_CPU651 <= 12
 row_capacity_CPU653: rows_tot_CPU653 <= 12
 row_capacity_CPU655: rows_tot_CPU655 <= 12
 row_capacity_CPU656: rows_tot_CPU656 <= 12
 row_capacity_CPU657: rows_tot_CPU657 <= 12
 row_capacity_CPU659: rows_tot_CPU659 <= 12
 row_capacity_CPU660: rows_tot_CPU660 <= 12
 row_capacity_CPU666: rows_tot_CPU666 <= 12
 row_capacity_CPU668: rows_tot_CPU668 <= 12
 row_capacity_CPU669: rows_tot_CPU669 <= 12
 row_capacity_CPU670: rows_tot_CPU670 <= 12
 row_capacity_CPU672: rows_tot_CPU672 <= 12
Bounds
 frac[14,211] <= 1
 frac[9,278] <= 1
 frac[20,202] <= 1
 frac[13,50] <= 1
 frac[14,208] <= 1
 frac[9,277] <= 1
 frac[4,432] <= 1
 frac[10,35] <= 1
 frac[12,259] <= 1
 frac[7,142] <= 1
 frac[2,243] <= 1
 frac[8,457] <= 1
 frac[17,464] <= 1
 frac[3,472] <= 1
 frac[18,3] <= 1
 frac[0,154] <= 1
 frac[21,72] <= 1
 frac[16,437] <= 1
 frac[1,202] <= 1
 mem_tot_CPU626 <= 32768
 normalized_mem_tot_CPU626 <= 1
 normalized_rows_tot_CPU626 <= 1
 mem_tot_CPU631 <= 32768
 normalized_mem_tot_CPU631 <= 1
 normalized_rows_tot_CPU631 <= 1
 mem_tot_CPU632 <= 32768
 normalized_mem_tot_CPU632 <= 1
 normalized_rows_tot_CPU632 <= 1
 mem_tot_CPU633 <= 32768
 normalized_mem_tot_CPU633 <= 1
 normalized_rows_tot_CPU633 <= 1
 mem_tot_CPU635 <= 32768
 normalized_mem_tot_CPU635 <= 1
 normalized_rows_tot_CPU635 <= 1
 mem_tot_CPU639 <= 32768
 normalized_mem_tot_CPU639 <= 1
 normalized_rows_tot_CPU639 <= 1
 mem_tot_CPU640 <= 32768
 normalized_mem_tot_CPU640 <= 1
 normalized_rows_tot_CPU640 <= 1
 mem_tot_CPU642 <= 32768
 normalized_mem_tot_CPU642 <= 1
 normalized_rows_tot_CPU642 <= 1
 mem_tot_CPU643 <= 32768
 normalized_mem_tot_CPU643 <= 1
 normalized_rows_tot_CPU643 <= 1
 mem_tot_CPU645 <= 32768
 normalized_mem_tot_CPU645 <= 1
 normalized_rows_tot_CPU645 <= 1
 mem_tot_CPU651 <= 32768
 normalized_mem_tot_CPU651 <= 1
 normalized_rows_tot_CPU651 <= 1
 mem_tot_CPU653 <= 32768
 normalized_mem_tot_CPU653 <= 1
 normalized_rows_tot_CPU653 <= 1
 mem_tot_CPU655 <= 32768
 normalized_mem_tot_CPU655 <= 1
 normalized_rows_tot_CPU655 <= 1
 mem_tot_CPU656 <= 32768
 normalized_mem_tot_CPU656 <= 1
 normalized_rows_tot_CPU656 <= 1
 mem_tot_CPU657 <= 32768
 normalized_mem_tot_CPU657 <= 1
 normalized_rows_tot_CPU657 <= 1
 mem_tot_CPU659 <= 32768
 normalized_mem_tot_CPU659 <= 1
 normalized_rows_tot_CPU659 <= 1
 mem_tot_CPU660 <= 32768
 normalized_mem_tot_CPU660 <= 1
 normalized_rows_tot_CPU660 <= 1
 mem_tot_CPU666 <= 32768
 normalized_mem_tot_CPU666 <= 1
 normalized_rows_tot_CPU666 <= 1
 mem_tot_CPU668 <= 32768
 normalized_mem_tot_CPU668 <= 1
 normalized_rows_tot_CPU668 <= 1
 mem_tot_CPU669 <= 32768
 normalized_mem_tot_CPU669 <= 1
 normalized_rows_tot_CPU669 <= 1
 mem_tot_CPU670 <= 32768
 normalized_mem_tot_CPU670 <= 1
 normalized_rows_tot_CPU670 <= 1
 mem_tot_CPU672 <= 32768
 normalized_mem_tot_CPU672 <= 1
 normalized_rows_tot_CPU672 <= 1
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU626 , normalized_mem_tot_CPU631 ,
   normalized_mem_tot_CPU632 , normalized_mem_tot_CPU633 ,
   normalized_mem_tot_CPU635 , normalized_mem_tot_CPU639 ,
   normalized_mem_tot_CPU640 , normalized_mem_tot_CPU642 ,
   normalized_mem_tot_CPU643 , normalized_mem_tot_CPU645 ,
   normalized_mem_tot_CPU651 , normalized_mem_tot_CPU653 ,
   normalized_mem_tot_CPU655 , normalized_mem_tot_CPU656 ,
   normalized_mem_tot_CPU657 , normalized_mem_tot_CPU659 ,
   normalized_mem_tot_CPU660 , normalized_mem_tot_CPU666 ,
   normalized_mem_tot_CPU668 , normalized_mem_tot_CPU669 ,
   normalized_mem_tot_CPU670 , normalized_mem_tot_CPU672 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX (
   normalized_rows_tot_CPU626 , normalized_rows_tot_CPU631 ,
   normalized_rows_tot_CPU632 , normalized_rows_tot_CPU633 ,
   normalized_rows_tot_CPU635 , normalized_rows_tot_CPU639 ,
   normalized_rows_tot_CPU640 , normalized_rows_tot_CPU642 ,
   normalized_rows_tot_CPU643 , normalized_rows_tot_CPU645 ,
   normalized_rows_tot_CPU651 , normalized_rows_tot_CPU653 ,
   normalized_rows_tot_CPU655 , normalized_rows_tot_CPU656 ,
   normalized_rows_tot_CPU657 , normalized_rows_tot_CPU659 ,
   normalized_rows_tot_CPU660 , normalized_rows_tot_CPU666 ,
   normalized_rows_tot_CPU668 , normalized_rows_tot_CPU669 ,
   normalized_rows_tot_CPU670 , normalized_rows_tot_CPU672 )
End

\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  OBJ0: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ1: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_rows_P4: Priority=80 Weight=1 AbsTol=0 RelTol=0
   rows_tot_P4_0 + rows_tot_P4_1 + rows_tot_P4_2 + rows_tot_P4_3
  P4_rows_load: Priority=70 Weight=1 AbsTol=0 RelTol=0
   max_rows_P4
  OBJ4: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ5: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_mem_P4: Priority=40 Weight=1 AbsTol=0 RelTol=0
   mem_tot_P4_0 + mem_tot_P4_1 + mem_tot_P4_2 + mem_tot_P4_3
  P4_load_mem: Priority=30 Weight=1 AbsTol=0 RelTol=0
   max_mem_P4
Subject To
 cov_0_1: frac[0,1] + frac[1,1] >= 0.3559835566142286
 cov_0_0: frac[0,0] + frac[1,0] >= 1
 cov_1_1: frac[2,1] + frac[1,1] >= 0.3559835566142286
 cov_1_0: frac[2,0] + frac[1,0] >= 1
 cov_2_1: frac[3,1] + frac[2,1] >= 0.3559835566142286
 cov_2_0: frac[3,0] + frac[2,0] >= 1
 cov_3_1: frac[3,1] >= 0.3559835566142286
 cov_3_0: frac[3,0] >= 1
 cov_4_1: frac[0,1] >= 0.3559835566142286
 cov_4_0: frac[0,0] >= 1
 accuracy_0_1: - 1698.9296875 frac[0,1] + mem[0,1] = 0
 accuracy_0_0: - 1698.9296875 frac[0,0] + mem[0,0] = 0
 accuracy_3_0: - 1698.9296875 frac[3,0] + mem[3,0] = 0
 accuracy_3_1: - 1698.9296875 frac[3,1] + mem[3,1] = 0
 accuracy_2_1: - 1698.9296875 frac[2,1] + mem[2,1] = 0
 accuracy_2_0: - 1698.9296875 frac[2,0] + mem[2,0] = 0
 accuracy_1_0: - 1698.9296875 frac[1,0] + mem[1,0] = 0
 accuracy_1_1: - 1698.9296875 frac[1,1] + mem[1,1] = 0
 mem_tot_P4_0: - mem[0,1] - mem[0,0] + mem_tot_P4_0 = 0
 rows_tot_P4_0: - frac[0,1] - frac[0,0] + rows_tot_P4_0 = 0
 mem_tot_P4_1: - mem[1,0] - mem[1,1] + mem_tot_P4_1 = 0
 rows_tot_P4_1: - frac[1,0] - frac[1,1] + rows_tot_P4_1 = 0
 mem_tot_P4_2: - mem[2,1] - mem[2,0] + mem_tot_P4_2 = 0
 rows_tot_P4_2: - frac[2,1] - frac[2,0] + rows_tot_P4_2 = 0
 mem_tot_P4_3: - mem[3,0] - mem[3,1] + mem_tot_P4_3 = 0
 rows_tot_P4_3: - frac[3,0] - frac[3,1] + rows_tot_P4_3 = 0
 tot_mem: - mem_tot_P4_0 - mem_tot_P4_1 - mem_tot_P4_2 - mem_tot_P4_3
   + tot_mem = 0
 tot_rows: - rows_tot_P4_0 - rows_tot_P4_1 - rows_tot_P4_2 - rows_tot_P4_3
   + tot_rows = 0
Bounds
 frac[0,1] <= 1
 frac[0,0] <= 1
 frac[3,0] <= 1
 frac[3,1] <= 1
 frac[2,1] <= 1
 frac[2,0] <= 1
 frac[1,0] <= 1
 frac[1,1] <= 1
 mem_tot_P4_0 <= 576
 mem_tot_P4_1 <= 576
 mem_tot_P4_2 <= 576
 mem_tot_P4_3 <= 576
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_P4_0 , mem_tot_P4_1 ,
   mem_tot_P4_2 , mem_tot_P4_3 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_P4_0 , rows_tot_P4_1 ,
   rows_tot_P4_2 , rows_tot_P4_3 )
End

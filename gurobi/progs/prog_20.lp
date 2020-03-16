\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU7 + rows_tot_CPU8
  CPU_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ4: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ5: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_mem_CPU: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU7 + mem_tot_CPU8
  CPU_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU
Subject To
 cov_0_3: frac[0,3] >= 0.7502251961685821
 cov_1_4: frac[1,4] >= 0.8
 cov_2_3: frac[1,3] + frac[0,3] >= 0.7502251961685821
 cov_3_1: frac[1,1] >= 0.1
 cov_4_4: frac[1,4] >= 1
 cov_5_0: frac[1,0] >= 0.2
 cov_6_2: frac[1,2] >= 0.2
 cov_7_1: frac[0,1] >= 0.1
 cov_8_1: frac[1,1] >= 0.1
 cov_9_0: frac[1,0] >= 0.2
 cov_10_2: frac[0,2] >= 0.2
 accuracy_1_2: - 5.09765625 frac[1,2] + mem[1,2] = 0
 accuracy_0_1: - 5.09765625 frac[0,1] + mem[0,1] = 0
 accuracy_1_3: - 5.09765625 frac[1,3] + mem[1,3] = 0
 accuracy_0_2: - 5.09765625 frac[0,2] + mem[0,2] = 0
 accuracy_1_4: - 5.09765625 frac[1,4] + mem[1,4] = 0
 accuracy_1_0: - 5.09765625 frac[1,0] + mem[1,0] = 0
 accuracy_1_1: - 5.09765625 frac[1,1] + mem[1,1] = 0
 accuracy_0_3: - 5.09765625 frac[0,3] + mem[0,3] = 0
 mem_tot_CPU7: - mem[0,1] - mem[0,2] - mem[0,3] + mem_tot_CPU7 = 0
 rows_tot_CPU7: - 3 frac[0,1] - 3 frac[0,2] - 3 frac[0,3] + rows_tot_CPU7
   = 0
 mem_tot_CPU8: - mem[1,2] - mem[1,3] - mem[1,4] - mem[1,0] - mem[1,1]
   + mem_tot_CPU8 = 0
 rows_tot_CPU8: - 3 frac[1,2] - 3 frac[1,3] - 3 frac[1,4] - 3 frac[1,0]
   - 3 frac[1,1] + rows_tot_CPU8 = 0
 row_capacity_CPU7: rows_tot_CPU7 <= 12
 row_capacity_CPU8: rows_tot_CPU8 <= 12
 tot_mem: - mem_tot_CPU7 - mem_tot_CPU8 + tot_mem = 0
 tot_rows: - rows_tot_CPU7 - rows_tot_CPU8 + tot_rows = 0
Bounds
 frac[1,2] <= 1
 frac[0,1] <= 1
 frac[1,3] <= 1
 frac[0,2] <= 1
 frac[1,4] <= 1
 frac[1,0] <= 1
 frac[1,1] <= 1
 frac[0,3] <= 1
 mem_tot_CPU7 <= 32768
 mem_tot_CPU8 <= 32768
General Constraints
 mem_overall_CPU: max_mem_CPU = MAX ( mem_tot_CPU7 , mem_tot_CPU8 )
 rows_overall_CPU: max_rows_CPU = MAX ( rows_tot_CPU7 , rows_tot_CPU8 )
End

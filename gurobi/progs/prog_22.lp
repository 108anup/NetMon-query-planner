\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU_0 + rows_tot_CPU_1 + rows_tot_CPU_2 + rows_tot_CPU_3
   + rows_tot_CPU_4
  CPU_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ4: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ5: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_mem_CPU: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU_0 + mem_tot_CPU_1 + mem_tot_CPU_2 + mem_tot_CPU_3
   + mem_tot_CPU_4
  CPU_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU
Subject To
 cov_0_0: frac[0,0] + frac[1,0] + frac[4,0] >= 1
 cov_1_1: frac[2,1] + frac[4,1] + frac[1,1] >= 1
 cov_2_2: frac[3,2] + frac[2,2] + frac[4,2] >= 1
 cov_3_3: frac[3,3] + frac[0,3] + frac[4,3] >= 1
 accuracy_3_2: - 5.09765625 frac[3,2] + mem[3,2] = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 accuracy_3_3: - 5.09765625 frac[3,3] + mem[3,3] = 0
 accuracy_2_1: - 5.09765625 frac[2,1] + mem[2,1] = 0
 accuracy_0_3: - 5.09765625 frac[0,3] + mem[0,3] = 0
 accuracy_4_3: - 5.09765625 frac[4,3] + mem[4,3] = 0
 accuracy_2_2: - 5.09765625 frac[2,2] + mem[2,2] = 0
 accuracy_4_2: - 5.09765625 frac[4,2] + mem[4,2] = 0
 accuracy_1_0: - 5.09765625 frac[1,0] + mem[1,0] = 0
 accuracy_4_1: - 5.09765625 frac[4,1] + mem[4,1] = 0
 accuracy_1_1: - 5.09765625 frac[1,1] + mem[1,1] = 0
 accuracy_4_0: - 5.09765625 frac[4,0] + mem[4,0] = 0
 mem_tot_CPU_0: - mem[0,0] - mem[0,3] + mem_tot_CPU_0 = 0
 rows_tot_CPU_0: - 3 frac[0,0] - 3 frac[0,3] + rows_tot_CPU_0 = 0
 mem_tot_CPU_1: - mem[1,0] - mem[1,1] + mem_tot_CPU_1 = 0
 rows_tot_CPU_1: - 3 frac[1,0] - 3 frac[1,1] + rows_tot_CPU_1 = 0
 mem_tot_CPU_2: - mem[2,1] - mem[2,2] + mem_tot_CPU_2 = 0
 rows_tot_CPU_2: - 3 frac[2,1] - 3 frac[2,2] + rows_tot_CPU_2 = 0
 mem_tot_CPU_3: - mem[3,2] - mem[3,3] + mem_tot_CPU_3 = 0
 rows_tot_CPU_3: - 3 frac[3,2] - 3 frac[3,3] + rows_tot_CPU_3 = 0
 mem_tot_CPU_4: - mem[4,3] - mem[4,2] - mem[4,1] - mem[4,0] + mem_tot_CPU_4
   = 0
 rows_tot_CPU_4: - 3 frac[4,3] - 3 frac[4,2] - 3 frac[4,1] - 3 frac[4,0]
   + rows_tot_CPU_4 = 0
 row_capacity_CPU_0: rows_tot_CPU_0 <= 12
 row_capacity_CPU_1: rows_tot_CPU_1 <= 12
 row_capacity_CPU_2: rows_tot_CPU_2 <= 12
 row_capacity_CPU_3: rows_tot_CPU_3 <= 12
 row_capacity_CPU_4: rows_tot_CPU_4 <= 12
 tot_mem: - mem_tot_CPU_0 - mem_tot_CPU_1 - mem_tot_CPU_2 - mem_tot_CPU_3
   - mem_tot_CPU_4 + tot_mem = 0
 tot_rows: - rows_tot_CPU_0 - rows_tot_CPU_1 - rows_tot_CPU_2
   - rows_tot_CPU_3 - rows_tot_CPU_4 + tot_rows = 0
Bounds
 mem_tot_CPU_0 <= 32768
 mem_tot_CPU_1 <= 32768
 mem_tot_CPU_2 <= 32768
 mem_tot_CPU_3 <= 32768
 mem_tot_CPU_4 <= 32768
Binaries
 frac[3,2] frac[0,0] frac[3,3] frac[2,1] frac[0,3] frac[4,3] frac[2,2]
 frac[4,2] frac[1,0] frac[4,1] frac[1,1] frac[4,0]
General Constraints
 mem_overall_CPU: max_mem_CPU = MAX ( mem_tot_CPU_0 , mem_tot_CPU_1 ,
   mem_tot_CPU_2 , mem_tot_CPU_3 , mem_tot_CPU_4 )
 rows_overall_CPU: max_rows_CPU = MAX ( rows_tot_CPU_0 , rows_tot_CPU_1 ,
   rows_tot_CPU_2 , rows_tot_CPU_3 , rows_tot_CPU_4 )
End

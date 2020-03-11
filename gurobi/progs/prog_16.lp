\ Model Univmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  max_mem: Priority=10 Weight=1 AbsTol=0 RelTol=0
   max_mem
  tot_mem: Priority=5 Weight=1 AbsTol=0 RelTol=0
   tot_mem
Subject To
 cov_0_0: frac[0,0] + frac[3,0] + frac[5,0] + frac[1,0] >= 0.6
 cov_0_1: frac[0,1] + frac[3,1] + frac[5,1] + frac[1,1] >= 0.8
 cov_1_1: frac[3,1] + frac[2,1] + frac[4,1] >= 0.6
 cov_1_2: frac[3,2] + frac[2,2] + frac[4,2] >= 0.9
 cov_2_0: frac[5,0] + frac[1,0] >= 1
 accuracy_0_1: - 5.09765625 frac[0,1] + mem[0,1] = 0
 accuracy_3_2: - 5.09765625 frac[3,2] + mem[3,2] = 0
 accuracy_0_0: - 5.09765625 frac[0,0] + mem[0,0] = 0
 accuracy_3_0: - 5.09765625 frac[3,0] + mem[3,0] = 0
 accuracy_3_1: - 5.09765625 frac[3,1] + mem[3,1] = 0
 accuracy_2_1: - 5.09765625 frac[2,1] + mem[2,1] = 0
 accuracy_5_0: - 5.09765625 frac[5,0] + mem[5,0] = 0
 accuracy_2_2: - 5.09765625 frac[2,2] + mem[2,2] = 0
 accuracy_4_2: - 5.09765625 frac[4,2] + mem[4,2] = 0
 accuracy_5_1: - 5.09765625 frac[5,1] + mem[5,1] = 0
 accuracy_1_0: - 5.09765625 frac[1,0] + mem[1,0] = 0
 accuracy_4_1: - 5.09765625 frac[4,1] + mem[4,1] = 0
 accuracy_1_1: - 5.09765625 frac[1,1] + mem[1,1] = 0
 mem_tot_CPU_0: - mem[0,1] - mem[0,0] + mem_tot_CPU_0 = 0
 rows_tot_CPU_0: - 3 frac[0,1] - 3 frac[0,0] + rows_tot_CPU_0 = 0
 mem_tot_CPU_1: - mem[1,0] - mem[1,1] + mem_tot_CPU_1 = 0
 rows_tot_CPU_1: - 3 frac[1,0] - 3 frac[1,1] + rows_tot_CPU_1 = 0
 mem_tot_P4_2: - mem[2,1] - mem[2,2] + mem_tot_P4_2 = 0
 rows_tot_P4_2: - 3 frac[2,1] - 3 frac[2,2] + rows_tot_P4_2 = 0
 mem_tot_P4_3: - mem[3,2] - mem[3,0] - mem[3,1] + mem_tot_P4_3 = 0
 rows_tot_P4_3: - 3 frac[3,2] - 3 frac[3,0] - 3 frac[3,1] + rows_tot_P4_3
   = 0
 mem_tot_P4_4: - mem[4,2] - mem[4,1] + mem_tot_P4_4 = 0
 rows_tot_P4_4: - 3 frac[4,2] - 3 frac[4,1] + rows_tot_P4_4 = 0
 mem_tot_P4_5: - mem[5,0] - mem[5,1] + mem_tot_P4_5 = 0
 rows_tot_P4_5: - 3 frac[5,0] - 3 frac[5,1] + rows_tot_P4_5 = 0
 tot_mem: - mem_tot_CPU_0 - mem_tot_CPU_1 - mem_tot_P4_2 - mem_tot_P4_3
   - mem_tot_P4_4 - mem_tot_P4_5 + tot_mem = 0
Bounds
 mem_tot_CPU_0 <= 32768
 mem_tot_CPU_1 <= 32768
 mem_tot_P4_2 <= 576
 mem_tot_P4_3 <= 576
 mem_tot_P4_4 <= 576
 mem_tot_P4_5 <= 576
Binaries
 frac[0,1] frac[3,2] frac[0,0] frac[3,0] frac[3,1] frac[2,1] frac[5,0]
 frac[2,2] frac[4,2] frac[5,1] frac[1,0] frac[4,1] frac[1,1]
General Constraints
 mem_overall: max_mem = MAX ( mem_tot_CPU_0 , mem_tot_CPU_1 ,
   mem_tot_P4_2 , mem_tot_P4_3 , mem_tot_P4_4 , mem_tot_P4_5 )
End

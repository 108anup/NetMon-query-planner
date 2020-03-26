\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU_Cluster: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU21 + rows_tot_CPU13 + rows_tot_CPU108 + rows_tot_CPU182
   + rows_tot_CPU187 + rows_tot_CPU5 + rows_tot_CPU155 + rows_tot_CPU174
   + rows_tot_CPU49 + rows_tot_CPU136 + rows_tot_CPU52 + rows_tot_CPU2
   + rows_tot_CPU146 + rows_tot_CPU117 + rows_tot_CPU173 + rows_tot_CPU55
   + rows_tot_CPU169 + rows_tot_CPU107 + rows_tot_CPU38
  CPU_Cluster_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU_Cluster
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ4: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ5: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_mem_CPU_Cluster: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU21 + mem_tot_CPU13 + mem_tot_CPU108 + mem_tot_CPU182
   + mem_tot_CPU187 + mem_tot_CPU5 + mem_tot_CPU155 + mem_tot_CPU174
   + mem_tot_CPU49 + mem_tot_CPU136 + mem_tot_CPU52 + mem_tot_CPU2
   + mem_tot_CPU146 + mem_tot_CPU117 + mem_tot_CPU173 + mem_tot_CPU55
   + mem_tot_CPU169 + mem_tot_CPU107 + mem_tot_CPU38
  CPU_Cluster_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
Subject To
 cov_0_0: frac[13,0] >= 2.78177e-08
 accuracy_13_0: - 5.09765625 frac[13,0] + mem[13,0] = 0
 mem_tot_CPU21: mem_tot_CPU21 = 0
 R3: - 3.05176e-05 mem_tot_CPU21 + normalized_mem_tot_CPU21 = 0
 rows_tot_CPU21: rows_tot_CPU21 = 0
 R5: - 0.0833333333333333 rows_tot_CPU21 + normalized_rows_tot_CPU21 = 0
 mem_tot_CPU13: mem_tot_CPU13 = 0
 R7: - 3.05176e-05 mem_tot_CPU13 + normalized_mem_tot_CPU13 = 0
 rows_tot_CPU13: rows_tot_CPU13 = 0
 R9: - 0.0833333333333333 rows_tot_CPU13 + normalized_rows_tot_CPU13 = 0
 mem_tot_CPU108: mem_tot_CPU108 = 0
 R11: - 3.05176e-05 mem_tot_CPU108 + normalized_mem_tot_CPU108 = 0
 rows_tot_CPU108: rows_tot_CPU108 = 0
 R13: - 0.0833333333333333 rows_tot_CPU108 + normalized_rows_tot_CPU108
   = 0
 mem_tot_CPU182: mem_tot_CPU182 = 0
 R15: - 3.05176e-05 mem_tot_CPU182 + normalized_mem_tot_CPU182 = 0
 rows_tot_CPU182: rows_tot_CPU182 = 0
 R17: - 0.0833333333333333 rows_tot_CPU182 + normalized_rows_tot_CPU182
   = 0
 mem_tot_CPU187: mem_tot_CPU187 = 0
 R19: - 3.05176e-05 mem_tot_CPU187 + normalized_mem_tot_CPU187 = 0
 rows_tot_CPU187: rows_tot_CPU187 = 0
 R21: - 0.0833333333333333 rows_tot_CPU187 + normalized_rows_tot_CPU187
   = 0
 mem_tot_CPU5: mem_tot_CPU5 = 0
 R23: - 3.05176e-05 mem_tot_CPU5 + normalized_mem_tot_CPU5 = 0
 rows_tot_CPU5: rows_tot_CPU5 = 0
 R25: - 0.0833333333333333 rows_tot_CPU5 + normalized_rows_tot_CPU5 = 0
 mem_tot_CPU155: mem_tot_CPU155 = 0
 R27: - 3.05176e-05 mem_tot_CPU155 + normalized_mem_tot_CPU155 = 0
 rows_tot_CPU155: rows_tot_CPU155 = 0
 R29: - 0.0833333333333333 rows_tot_CPU155 + normalized_rows_tot_CPU155
   = 0
 mem_tot_CPU174: mem_tot_CPU174 = 0
 R31: - 3.05176e-05 mem_tot_CPU174 + normalized_mem_tot_CPU174 = 0
 rows_tot_CPU174: rows_tot_CPU174 = 0
 R33: - 0.0833333333333333 rows_tot_CPU174 + normalized_rows_tot_CPU174
   = 0
 mem_tot_CPU49: mem_tot_CPU49 = 0
 R35: - 3.05176e-05 mem_tot_CPU49 + normalized_mem_tot_CPU49 = 0
 rows_tot_CPU49: rows_tot_CPU49 = 0
 R37: - 0.0833333333333333 rows_tot_CPU49 + normalized_rows_tot_CPU49 = 0
 mem_tot_CPU136: mem_tot_CPU136 = 0
 R39: - 3.05176e-05 mem_tot_CPU136 + normalized_mem_tot_CPU136 = 0
 rows_tot_CPU136: rows_tot_CPU136 = 0
 R41: - 0.0833333333333333 rows_tot_CPU136 + normalized_rows_tot_CPU136
   = 0
 mem_tot_CPU52: mem_tot_CPU52 = 0
 R43: - 3.05176e-05 mem_tot_CPU52 + normalized_mem_tot_CPU52 = 0
 rows_tot_CPU52: rows_tot_CPU52 = 0
 R45: - 0.0833333333333333 rows_tot_CPU52 + normalized_rows_tot_CPU52 = 0
 mem_tot_CPU2: mem_tot_CPU2 = 0
 R47: - 3.05176e-05 mem_tot_CPU2 + normalized_mem_tot_CPU2 = 0
 rows_tot_CPU2: rows_tot_CPU2 = 0
 R49: - 0.0833333333333333 rows_tot_CPU2 + normalized_rows_tot_CPU2 = 0
 mem_tot_CPU146: mem_tot_CPU146 = 0
 R51: - 3.05176e-05 mem_tot_CPU146 + normalized_mem_tot_CPU146 = 0
 rows_tot_CPU146: rows_tot_CPU146 = 0
 R53: - 0.0833333333333333 rows_tot_CPU146 + normalized_rows_tot_CPU146
   = 0
 mem_tot_CPU117: - mem[13,0] + mem_tot_CPU117 = 0
 R55: - 3.05176e-05 mem_tot_CPU117 + normalized_mem_tot_CPU117 = 0
 rows_tot_CPU117: - 3 frac[13,0] + rows_tot_CPU117 = 0
 R57: - 0.0833333333333333 rows_tot_CPU117 + normalized_rows_tot_CPU117
   = 0
 mem_tot_CPU173: mem_tot_CPU173 = 0
 R59: - 3.05176e-05 mem_tot_CPU173 + normalized_mem_tot_CPU173 = 0
 rows_tot_CPU173: rows_tot_CPU173 = 0
 R61: - 0.0833333333333333 rows_tot_CPU173 + normalized_rows_tot_CPU173
   = 0
 mem_tot_CPU55: mem_tot_CPU55 = 0
 R63: - 3.05176e-05 mem_tot_CPU55 + normalized_mem_tot_CPU55 = 0
 rows_tot_CPU55: rows_tot_CPU55 = 0
 R65: - 0.0833333333333333 rows_tot_CPU55 + normalized_rows_tot_CPU55 = 0
 mem_tot_CPU169: mem_tot_CPU169 = 0
 R67: - 3.05176e-05 mem_tot_CPU169 + normalized_mem_tot_CPU169 = 0
 rows_tot_CPU169: rows_tot_CPU169 = 0
 R69: - 0.0833333333333333 rows_tot_CPU169 + normalized_rows_tot_CPU169
   = 0
 mem_tot_CPU107: mem_tot_CPU107 = 0
 R71: - 3.05176e-05 mem_tot_CPU107 + normalized_mem_tot_CPU107 = 0
 rows_tot_CPU107: rows_tot_CPU107 = 0
 R73: - 0.0833333333333333 rows_tot_CPU107 + normalized_rows_tot_CPU107
   = 0
 mem_tot_CPU38: mem_tot_CPU38 = 0
 R75: - 3.05176e-05 mem_tot_CPU38 + normalized_mem_tot_CPU38 = 0
 rows_tot_CPU38: rows_tot_CPU38 = 0
 R77: - 0.0833333333333333 rows_tot_CPU38 + normalized_rows_tot_CPU38 = 0
 row_capacity_CPU21: rows_tot_CPU21 <= 12
 row_capacity_CPU13: rows_tot_CPU13 <= 12
 row_capacity_CPU108: rows_tot_CPU108 <= 12
 row_capacity_CPU182: rows_tot_CPU182 <= 12
 row_capacity_CPU187: rows_tot_CPU187 <= 12
 row_capacity_CPU5: rows_tot_CPU5 <= 12
 row_capacity_CPU155: rows_tot_CPU155 <= 12
 row_capacity_CPU174: rows_tot_CPU174 <= 12
 row_capacity_CPU49: rows_tot_CPU49 <= 12
 row_capacity_CPU136: rows_tot_CPU136 <= 12
 row_capacity_CPU52: rows_tot_CPU52 <= 12
 row_capacity_CPU2: rows_tot_CPU2 <= 12
 row_capacity_CPU146: rows_tot_CPU146 <= 12
 row_capacity_CPU117: rows_tot_CPU117 <= 12
 row_capacity_CPU173: rows_tot_CPU173 <= 12
 row_capacity_CPU55: rows_tot_CPU55 <= 12
 row_capacity_CPU169: rows_tot_CPU169 <= 12
 row_capacity_CPU107: rows_tot_CPU107 <= 12
 row_capacity_CPU38: rows_tot_CPU38 <= 12
Bounds
 frac[13,0] <= 1
 mem_tot_CPU21 <= 32768
 normalized_mem_tot_CPU21 <= 1
 normalized_rows_tot_CPU21 <= 1
 mem_tot_CPU13 <= 32768
 normalized_mem_tot_CPU13 <= 1
 normalized_rows_tot_CPU13 <= 1
 mem_tot_CPU108 <= 32768
 normalized_mem_tot_CPU108 <= 1
 normalized_rows_tot_CPU108 <= 1
 mem_tot_CPU182 <= 32768
 normalized_mem_tot_CPU182 <= 1
 normalized_rows_tot_CPU182 <= 1
 mem_tot_CPU187 <= 32768
 normalized_mem_tot_CPU187 <= 1
 normalized_rows_tot_CPU187 <= 1
 mem_tot_CPU5 <= 32768
 normalized_mem_tot_CPU5 <= 1
 normalized_rows_tot_CPU5 <= 1
 mem_tot_CPU155 <= 32768
 normalized_mem_tot_CPU155 <= 1
 normalized_rows_tot_CPU155 <= 1
 mem_tot_CPU174 <= 32768
 normalized_mem_tot_CPU174 <= 1
 normalized_rows_tot_CPU174 <= 1
 mem_tot_CPU49 <= 32768
 normalized_mem_tot_CPU49 <= 1
 normalized_rows_tot_CPU49 <= 1
 mem_tot_CPU136 <= 32768
 normalized_mem_tot_CPU136 <= 1
 normalized_rows_tot_CPU136 <= 1
 mem_tot_CPU52 <= 32768
 normalized_mem_tot_CPU52 <= 1
 normalized_rows_tot_CPU52 <= 1
 mem_tot_CPU2 <= 32768
 normalized_mem_tot_CPU2 <= 1
 normalized_rows_tot_CPU2 <= 1
 mem_tot_CPU146 <= 32768
 normalized_mem_tot_CPU146 <= 1
 normalized_rows_tot_CPU146 <= 1
 mem_tot_CPU117 <= 32768
 normalized_mem_tot_CPU117 <= 1
 normalized_rows_tot_CPU117 <= 1
 mem_tot_CPU173 <= 32768
 normalized_mem_tot_CPU173 <= 1
 normalized_rows_tot_CPU173 <= 1
 mem_tot_CPU55 <= 32768
 normalized_mem_tot_CPU55 <= 1
 normalized_rows_tot_CPU55 <= 1
 mem_tot_CPU169 <= 32768
 normalized_mem_tot_CPU169 <= 1
 normalized_rows_tot_CPU169 <= 1
 mem_tot_CPU107 <= 32768
 normalized_mem_tot_CPU107 <= 1
 normalized_rows_tot_CPU107 <= 1
 mem_tot_CPU38 <= 32768
 normalized_mem_tot_CPU38 <= 1
 normalized_rows_tot_CPU38 <= 1
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU21 , normalized_mem_tot_CPU13 ,
   normalized_mem_tot_CPU108 , normalized_mem_tot_CPU182 ,
   normalized_mem_tot_CPU187 , normalized_mem_tot_CPU5 ,
   normalized_mem_tot_CPU155 , normalized_mem_tot_CPU174 ,
   normalized_mem_tot_CPU49 , normalized_mem_tot_CPU136 ,
   normalized_mem_tot_CPU52 , normalized_mem_tot_CPU2 ,
   normalized_mem_tot_CPU146 , normalized_mem_tot_CPU117 ,
   normalized_mem_tot_CPU173 , normalized_mem_tot_CPU55 ,
   normalized_mem_tot_CPU169 , normalized_mem_tot_CPU107 ,
   normalized_mem_tot_CPU38 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX ( normalized_rows_tot_CPU21 ,
   normalized_rows_tot_CPU13 , normalized_rows_tot_CPU108 ,
   normalized_rows_tot_CPU182 , normalized_rows_tot_CPU187 ,
   normalized_rows_tot_CPU5 , normalized_rows_tot_CPU155 ,
   normalized_rows_tot_CPU174 , normalized_rows_tot_CPU49 ,
   normalized_rows_tot_CPU136 , normalized_rows_tot_CPU52 ,
   normalized_rows_tot_CPU2 , normalized_rows_tot_CPU146 ,
   normalized_rows_tot_CPU117 , normalized_rows_tot_CPU173 ,
   normalized_rows_tot_CPU55 , normalized_rows_tot_CPU169 ,
   normalized_rows_tot_CPU107 , normalized_rows_tot_CPU38 )
End

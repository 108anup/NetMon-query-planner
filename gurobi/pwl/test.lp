\ Model pdt_test
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 col_bits + mem
Subject To
 R0: - cols + mem >= 0
 R1: cols >= 5
 R2: cols - exp_col_bits = 0
Bounds
 col_bits <= 20
 cols <= 1.048576e+06
 exp_col_bits <= 1.048576e+06
Generals
 col_bits cols exp_col_bits
General Constraints
 GC0: exp_col_bits = EXPA ( 2 ^ col_bits )
End

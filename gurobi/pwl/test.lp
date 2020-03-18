\ Model pdt_test
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x + y + 0 z + 0 logz + 0 ez + pdt_xy
Subject To
 pdt_xy: pdt_xy + [ - x * y ] = 0
Bounds
 x >= 300
 y >= 4.32
 -1 <= logz <= -0.5
General Constraints
 GC0: logz = LOG_2 ( z )
 GC1: ez = EXPA ( 2 ^ z )
End

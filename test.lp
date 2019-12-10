\ Model pdt_test
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x + y + 0 z + 0 logz + 0 ez + pdt_xy
Subject To
 pdt_xy: - log_x - log_y + log_pdt_xy = 0
Bounds
 5 <= x <= 6
 6 <= y <= 7
 z = 11
General Constraints
 GC0: logz = LOG_2 ( z )
 GC1: ez = EXPA ( 2 ^ z )
 log_pdt_xy: ( FuncPieces=-1 ) log_pdt_xy = LOG_2 ( pdt_xy )
 log_x: log_x = LOG_2 ( x )
 log_y: log_y = LOG_2 ( y )
End

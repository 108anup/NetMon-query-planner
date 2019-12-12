\ Model pdt_test
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x + y + 0 z + 0 logz + 0 ez + pdt_xy
Subject To
 pdt_xy: - log_x - log_y + log_pdt_xy = 0
Bounds
 -1 <= logz <= -0.5
 log_x free
 log_y free
 log_pdt_xy free
General Constraints
 GC0: logz = LOG_2 ( z )
 GC1: ez = EXPA ( 2 ^ z )
 log_pdt_xy: ( FuncPieces=-1 FuncPieceError=1e-05 ) pdt_xy =
   EXPA ( 2 ^ log_pdt_xy )
 log_x: x = EXPA ( 2 ^ log_x )
 log_y: y = EXPA ( 2 ^ log_y )
End

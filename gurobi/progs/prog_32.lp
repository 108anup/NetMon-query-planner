\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_others + 0 max_rows_others + 0 m_access_time_Netronome_1 + res
Subject To
 cov_0_0: frac[0,0] >= 1
 accuracy_0_0: - 8192 frac[0,0] + mem[0,0] = 0
 mem_tot_Netronome_1: - mem[0,0] + mem_tot_Netronome_1 = 0
 R3: - 5e-06 mem_tot_Netronome_1 + normalized_mem_tot_Netronome_1 = 0
 rows_tot_Netronome_1: - 4 frac[0,0] + rows_tot_Netronome_1 = 0
 R5: - 0.0833333333333333 rows_tot_Netronome_1
   + normalized_rows_tot_Netronome_1 = 0
 row_capacity_Netronome_1: rows_tot_Netronome_1 <= 12
 ns_mem_max_Netronome_1: - ns_mem_max_Netronome_1 + pdt_m_rows_Netronome_1
   = -13.40036284744041
 ns_hash_max_Netronome_1: - 1.01427115498035 rows_tot_Netronome_1
   + ns_hash_max_Netronome_1 = 33.22322438313901
 ns_req_hash_Netronome_1: - 54 ns_hash_max_Netronome_1
   + 43.21393312771375 micro_engines_Netronome_1 >= 0
 ns_req_mem_Netronome_1: ns_mem_max_Netronome_1 <= 43.21393312771375
 ns_req_fwd_Netronome_1: 43.21393312771375 micro_engines_Netronome_1
   >= 1960.072595281307
 res_acc: 3.2552083333333332e-04 mem_tot_Netronome_1
   + 5 micro_engines_Netronome_1 - res = 0
 pdt_m_rows_Netronome_1: pdt_m_rows_Netronome_1 + [
   - rows_tot_Netronome_1 * m_access_time_Netronome_1 ] = 0
Bounds
 mem_tot_Netronome_1 <= 200000
 normalized_mem_tot_Netronome_1 <= 1
 normalized_rows_tot_Netronome_1 <= 1
 micro_engines_Netronome_1 <= 54
Binaries
 frac[0,0]
Generals
 micro_engines_Netronome_1
General Constraints
 mem_overall_others: max_mem_others = MAX ( normalized_mem_tot_Netronome_1 )
 rows_overall_others: max_rows_others = MAX (
   normalized_rows_tot_Netronome_1 )
 mem_access_time_Netronome_1:
   m_access_time_Netronome_1 = PWL ( mem_tot_Netronome_1 ) : 
  (0, 2.437806463533801) (0.0390625, 2.437806463533801) 
  (0.078125, 3.001397701320998) (0.625, 3.001397701320998) 
  (1.25, 3.419718427960497) (1280, 3.419718427960497) 
  (5120, 5.797905875083051) (10240, 8.557025366725194) 
  (20480, 10.33170622821247) (40960, 11.79222979807959) 
  (163840, 13.76708909669199) (200000, 14.34823258821074)
End

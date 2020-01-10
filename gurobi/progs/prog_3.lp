\ Model netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0.05
   0 m_access_time_cpu_1 + 0 ns_sketch_cpu_1 + 0 ns_dpdk_cpu_1 + 0 ns_cpu_1
   + ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0.05
   res_overall
Subject To
 cov_0: frac[0,0] + frac[1,0] = 1
 cov_1: frac[0,1] + frac[1,1] = 1
 cov_2: frac[0,2] + frac[1,2] = 1
 cov_3: frac[0,3] + frac[1,3] = 1
 accuracy_(1,_sk_1)[0]: - 212.3671875 frac[0,0] + mem[0,0] >= 0
 accuracy_(1,_sk_1)[1]: - 212.3671875 frac[1,0] + mem[1,0] >= 0
 accuracy_(2,_sk_1)[0]: - 212.3671875 frac[0,1] + mem[0,1] >= 0
 accuracy_(2,_sk_1)[1]: - 212.3671875 frac[1,1] + mem[1,1] >= 0
 accuracy_(3,_sk_1)[0]: - 212.3671875 frac[0,2] + mem[0,2] >= 0
 accuracy_(3,_sk_1)[1]: - 212.3671875 frac[1,2] + mem[1,2] >= 0
 accuracy_(4,_sk_1)[0]: - 212.3671875 frac[0,3] + mem[0,3] >= 0
 accuracy_(4,_sk_1)[1]: - 212.3671875 frac[1,3] + mem[1,3] >= 0
 rows_tot_cpu_1: - frac[0,0] - frac[0,1] - frac[0,2] - frac[0,3]
   + rows_tot_cpu_1 = 0
 mem_tot_cpu_1: - mem[0,0] - mem[0,1] - mem[0,2] - mem[0,3] + mem_tot_cpu_1
   = 0
 pdt_m_rows_cpu_1: - log_m_access_time_cpu_1 - log_rows_tot_cpu_1
   + log_pdt_m_rows_cpu_1 = 0
 ns_single_cpu_1: 3.5 rows_tot_cpu_1 - ns_single_cpu_1 + pdt_m_rows_cpu_1
   <= 0
 capacity_cores_cpu_1: cores_sketch_cpu_1 + cores_dpdk_cpu_1 <= 7
 pdt_nsk_c_cpu_1: - log_ns_sketch_cpu_1 - log_cores_sketch_cpu_1
   + log_pdt_nsk_c_cpu_1 = 0
 ns_sketch_cpu_1: - ns_single_cpu_1 + pdt_nsk_c_cpu_1 = 0
 pdt_nsc_dpdk_cpu_1: - log_ns_dpdk_cpu_1 - log_cores_dpdk_cpu_1
   + log_pdt_nsc_dpdk_cpu_1 = 0
 ns_dpdk_cpu_1: 7.142857142857143 cores_dpdk_cpu_1 - pdt_nsc_dpdk_cpu_1
   = -21.42857142857143
 rows_tot_p4_1: - frac[1,0] - frac[1,1] - frac[1,2] - frac[1,3]
   + rows_tot_p4_1 = 0
 mem_tot_p4_1: - mem[1,0] - mem[1,1] - mem[1,2] - mem[1,3] + mem_tot_p4_1
   = 0
 capacity_mem_par_p4_1: mem[1,0] <= 48
 capacity_mem_par_p4_1: mem[1,1] <= 48
 capacity_mem_par_p4_1: mem[1,2] <= 48
 capacity_mem_par_p4_1: mem[1,3] <= 48
 ns_p4_1: ns_p4_1 = 6.756756756756757
 res: - 1.220703125e-04 mem_tot_cpu_1 - 10 cores_sketch_cpu_1
   - 10 cores_dpdk_cpu_1 - 0.25 rows_tot_p4_1
   - 0.0208333333333333 mem_tot_p4_1 + res_overall = 0
Bounds
 frac[0,0] <= 1
 frac[0,1] <= 1
 frac[0,2] <= 1
 frac[0,3] <= 1
 frac[1,0] <= 1
 frac[1,1] <= 1
 frac[1,2] <= 1
 frac[1,3] <= 1
 rows_tot_cpu_1 <= 9
 mem_tot_cpu_1 <= 32768
 log_m_access_time_cpu_1 free
 log_rows_tot_cpu_1 free
 log_pdt_m_rows_cpu_1 free
 cores_sketch_cpu_1 <= 7
 1 <= cores_dpdk_cpu_1 <= 7
 log_ns_sketch_cpu_1 free
 log_cores_sketch_cpu_1 free
 log_pdt_nsk_c_cpu_1 free
 log_ns_dpdk_cpu_1 free
 log_cores_dpdk_cpu_1 free
 log_pdt_nsc_dpdk_cpu_1 free
 rows_tot_p4_1 <= 12
 mem_tot_p4_1 <= 576
Generals
 cores_sketch_cpu_1 cores_dpdk_cpu_1
General Constraints
 mem_access_time_cpu_1: m_access_time_cpu_1 = PWL ( mem_tot_cpu_1 ) : 
  (1.1875, 0.5) (32, 0.510892) (1448.15625, 5.04469) (5792.625, 5.84114) 
  (32768, 30.6627) (440871.90625, 39.6981)
 log_pdt_m_rows_cpu_1: ( FuncPieces=-1 FuncPieceError=0.0001 )
   pdt_m_rows_cpu_1 = EXPA ( 2 ^ log_pdt_m_rows_cpu_1 )
 log_m_access_time_cpu_1: ( FuncPieces=-1 FuncPieceError=0.0001 )
   m_access_time_cpu_1 = EXPA ( 2 ^ log_m_access_time_cpu_1 )
 log_rows_tot_cpu_1: ( FuncPieces=-1 FuncPieceError=0.0001 )
   rows_tot_cpu_1 = EXPA ( 2 ^ log_rows_tot_cpu_1 )
 log_pdt_nsk_c_cpu_1: ( FuncPieces=-1 FuncPieceError=0.0001 )
   pdt_nsk_c_cpu_1 = EXPA ( 2 ^ log_pdt_nsk_c_cpu_1 )
 log_ns_sketch_cpu_1: ( FuncPieces=-1 FuncPieceError=0.0001 )
   ns_sketch_cpu_1 = EXPA ( 2 ^ log_ns_sketch_cpu_1 )
 log_cores_sketch_cpu_1: ( FuncPieces=-1 FuncPieceError=0.0001 )
   cores_sketch_cpu_1 = EXPA ( 2 ^ log_cores_sketch_cpu_1 )
 log_pdt_nsc_dpdk_cpu_1: ( FuncPieces=-1 FuncPieceError=0.0001 )
   pdt_nsc_dpdk_cpu_1 = EXPA ( 2 ^ log_pdt_nsc_dpdk_cpu_1 )
 log_ns_dpdk_cpu_1: ( FuncPieces=-1 FuncPieceError=0.0001 ) ns_dpdk_cpu_1 =
   EXPA ( 2 ^ log_ns_dpdk_cpu_1 )
 log_cores_dpdk_cpu_1: ( FuncPieces=-1 FuncPieceError=0.0001 )
   cores_dpdk_cpu_1 = EXPA ( 2 ^ log_cores_dpdk_cpu_1 )
 ns_cpu_1: ns_cpu_1 = MAX ( ns_dpdk_cpu_1 , ns_sketch_cpu_1 )
 ns_overall: ns = MAX ( ns_cpu_1 , ns_p4_1 )
End

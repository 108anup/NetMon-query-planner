\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 m_access_time_CPU_0 + 0 ns_sketch_CPU_0 + 0 ns_dpdk_CPU_0 + 0 ns_CPU_0
   + 0 m_access_time_CPU_1 + 0 ns_sketch_CPU_1 + 0 ns_dpdk_CPU_1
   + 0 ns_CPU_1 + ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0
   res
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
 ns_single_CPU_0: 3.5 rows_tot_CPU_0 - ns_single_CPU_0 + pdt_m_rows_CPU_0
   = 0
 capacity_cores_CPU_0: cores_sketch_CPU_0 + cores_dpdk_CPU_0 <= 7
 ns_sketch_CPU_0: - ns_single_CPU_0 + pdt_nsk_c_CPU_0 = 0
 ns_dpdk_CPU_0: 7.142857142857143 cores_dpdk_CPU_0 - pdt_nsc_dpdk_CPU_0
   = -21.42857142857143
 ns_single_CPU_1: 3.5 rows_tot_CPU_1 - ns_single_CPU_1 + pdt_m_rows_CPU_1
   = 0
 capacity_cores_CPU_1: cores_sketch_CPU_1 + cores_dpdk_CPU_1 <= 7
 ns_sketch_CPU_1: - ns_single_CPU_1 + pdt_nsk_c_CPU_1 = 0
 ns_dpdk_CPU_1: 7.142857142857143 cores_dpdk_CPU_1 - pdt_nsc_dpdk_CPU_1
   = -21.42857142857143
 ns_P4_2: ns_P4_2 = 6.756756756756757
 ns_P4_3: ns_P4_3 = 6.756756756756757
 ns_P4_4: ns_P4_4 = 6.756756756756757
 ns_P4_5: ns_P4_5 = 6.756756756756757
 res_acc: 1.220703125e-04 mem_tot_CPU_0 + 1.220703125e-04 mem_tot_CPU_1
   + 0.0208333333333333 mem_tot_P4_2 + 0.25 rows_tot_P4_2
   + 0.0208333333333333 mem_tot_P4_3 + 0.25 rows_tot_P4_3
   + 0.0208333333333333 mem_tot_P4_4 + 0.25 rows_tot_P4_4
   + 0.0208333333333333 mem_tot_P4_5 + 0.25 rows_tot_P4_5
   + 10 cores_sketch_CPU_0 + 10 cores_dpdk_CPU_0 + 10 cores_sketch_CPU_1
   + 10 cores_dpdk_CPU_1 - res = 0
 pdt_m_rows_CPU_0: pdt_m_rows_CPU_0 + [
   - rows_tot_CPU_0 * m_access_time_CPU_0 ] = 0
 pdt_nsk_c_CPU_0: pdt_nsk_c_CPU_0 + [
   - cores_sketch_CPU_0 * ns_sketch_CPU_0 ] = 0
 pdt_nsc_dpdk_CPU_0: pdt_nsc_dpdk_CPU_0 + [
   - cores_dpdk_CPU_0 * ns_dpdk_CPU_0 ] = 0
 pdt_m_rows_CPU_1: pdt_m_rows_CPU_1 + [
   - rows_tot_CPU_1 * m_access_time_CPU_1 ] = 0
 pdt_nsk_c_CPU_1: pdt_nsk_c_CPU_1 + [
   - cores_sketch_CPU_1 * ns_sketch_CPU_1 ] = 0
 pdt_nsc_dpdk_CPU_1: pdt_nsc_dpdk_CPU_1 + [
   - cores_dpdk_CPU_1 * ns_dpdk_CPU_1 ] = 0
Bounds
 mem_tot_CPU_0 <= 32768
 mem_tot_CPU_1 <= 32768
 mem_tot_P4_2 <= 576
 mem_tot_P4_3 <= 576
 mem_tot_P4_4 <= 576
 mem_tot_P4_5 <= 576
 cores_sketch_CPU_0 <= 7
 1 <= cores_dpdk_CPU_0 <= 7
 cores_sketch_CPU_1 <= 7
 1 <= cores_dpdk_CPU_1 <= 7
Binaries
 frac[0,1] frac[3,2] frac[0,0] frac[3,0] frac[3,1] frac[2,1] frac[5,0]
 frac[2,2] frac[4,2] frac[5,1] frac[1,0] frac[4,1] frac[1,1]
Generals
 cores_sketch_CPU_0 cores_dpdk_CPU_0 cores_sketch_CPU_1 cores_dpdk_CPU_1
General Constraints
 mem_access_time_CPU_0: m_access_time_CPU_0 = PWL ( mem_tot_CPU_0 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_0: ns_CPU_0 = MAX ( ns_dpdk_CPU_0 , ns_sketch_CPU_0 )
 mem_access_time_CPU_1: m_access_time_CPU_1 = PWL ( mem_tot_CPU_1 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_CPU_1: ns_CPU_1 = MAX ( ns_dpdk_CPU_1 , ns_sketch_CPU_1 )
 ns_overall: ns = MAX ( ns_CPU_0 , ns_CPU_1 , ns_P4_2 , ns_P4_3 , ns_P4_4 ,
   ns_P4_5 )
End

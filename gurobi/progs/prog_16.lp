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
 cov_0_0: frac[3,0] + frac[5,0] + frac[0,0] + frac[1,0] >= 0.6
 cov_0_1: frac[5,1] + frac[1,1] + frac[0,1] + frac[3,1] >= 0.6
 cov_0_2: frac[1,2] + frac[3,2] + frac[5,2] + frac[0,2] >= 0.6
 cov_0_3: frac[1,3] + frac[0,3] + frac[3,3] + frac[5,3] >= 0.8
 cov_0_4: frac[0,4] + frac[5,4] + frac[1,4] + frac[3,4] >= 0.8
 cov_0_5: frac[5,5] + frac[1,5] + frac[0,5] + frac[3,5] >= 0.8
 cov_1_3: frac[3,3] + frac[2,3] + frac[4,3] >= 0.6
 cov_1_4: frac[2,4] + frac[4,4] + frac[3,4] >= 0.6
 cov_1_5: frac[2,5] + frac[4,5] + frac[3,5] >= 0.6
 cov_1_6: frac[3,6] + frac[2,6] + frac[4,6] >= 0.9
 cov_1_7: frac[4,7] + frac[3,7] + frac[2,7] >= 0.9
 cov_1_8: frac[4,8] + frac[2,8] + frac[3,8] >= 0.9
 cov_2_0: frac[5,0] + frac[1,0] >= 1
 cov_2_1: frac[5,1] + frac[1,1] >= 1
 cov_2_2: frac[1,2] + frac[5,2] >= 1
 accuracy_4_7: - 1.69921875 frac[4,7] + mem[4,7] = 0
 accuracy_1_3: - 1.69921875 frac[1,3] + mem[1,3] = 0
 accuracy_4_8: - 1.69921875 frac[4,8] + mem[4,8] = 0
 accuracy_3_0: - 1.69921875 frac[3,0] + mem[3,0] = 0
 accuracy_2_8: - 1.69921875 frac[2,8] + mem[2,8] = 0
 accuracy_5_1: - 1.69921875 frac[5,1] + mem[5,1] = 0
 accuracy_0_3: - 1.69921875 frac[0,3] + mem[0,3] = 0
 accuracy_2_5: - 1.69921875 frac[2,5] + mem[2,5] = 0
 accuracy_3_7: - 1.69921875 frac[3,7] + mem[3,7] = 0
 accuracy_2_4: - 1.69921875 frac[2,4] + mem[2,4] = 0
 accuracy_1_2: - 1.69921875 frac[1,2] + mem[1,2] = 0
 accuracy_3_3: - 1.69921875 frac[3,3] + mem[3,3] = 0
 accuracy_5_5: - 1.69921875 frac[5,5] + mem[5,5] = 0
 accuracy_4_4: - 1.69921875 frac[4,4] + mem[4,4] = 0
 accuracy_1_5: - 1.69921875 frac[1,5] + mem[1,5] = 0
 accuracy_5_0: - 1.69921875 frac[5,0] + mem[5,0] = 0
 accuracy_0_4: - 1.69921875 frac[0,4] + mem[0,4] = 0
 accuracy_3_6: - 1.69921875 frac[3,6] + mem[3,6] = 0
 accuracy_1_1: - 1.69921875 frac[1,1] + mem[1,1] = 0
 accuracy_3_2: - 1.69921875 frac[3,2] + mem[3,2] = 0
 accuracy_0_0: - 1.69921875 frac[0,0] + mem[0,0] = 0
 accuracy_5_4: - 1.69921875 frac[5,4] + mem[5,4] = 0
 accuracy_2_6: - 1.69921875 frac[2,6] + mem[2,6] = 0
 accuracy_4_5: - 1.69921875 frac[4,5] + mem[4,5] = 0
 accuracy_5_2: - 1.69921875 frac[5,2] + mem[5,2] = 0
 accuracy_1_4: - 1.69921875 frac[1,4] + mem[1,4] = 0
 accuracy_0_5: - 1.69921875 frac[0,5] + mem[0,5] = 0
 accuracy_2_3: - 1.69921875 frac[2,3] + mem[2,3] = 0
 accuracy_1_0: - 1.69921875 frac[1,0] + mem[1,0] = 0
 accuracy_5_3: - 1.69921875 frac[5,3] + mem[5,3] = 0
 accuracy_0_1: - 1.69921875 frac[0,1] + mem[0,1] = 0
 accuracy_3_5: - 1.69921875 frac[3,5] + mem[3,5] = 0
 accuracy_2_7: - 1.69921875 frac[2,7] + mem[2,7] = 0
 accuracy_4_6: - 1.69921875 frac[4,6] + mem[4,6] = 0
 accuracy_3_1: - 1.69921875 frac[3,1] + mem[3,1] = 0
 accuracy_3_8: - 1.69921875 frac[3,8] + mem[3,8] = 0
 accuracy_4_3: - 1.69921875 frac[4,3] + mem[4,3] = 0
 accuracy_3_4: - 1.69921875 frac[3,4] + mem[3,4] = 0
 accuracy_0_2: - 1.69921875 frac[0,2] + mem[0,2] = 0
 mem_tot_CPU_0: - mem[0,3] - mem[0,4] - mem[0,0] - mem[0,5] - mem[0,1]
   - mem[0,2] + mem_tot_CPU_0 = 0
 rows_tot_CPU_0: - frac[0,3] - frac[0,4] - frac[0,0] - frac[0,5]
   - frac[0,1] - frac[0,2] + rows_tot_CPU_0 = 0
 mem_tot_CPU_1: - mem[1,3] - mem[1,2] - mem[1,5] - mem[1,1] - mem[1,4]
   - mem[1,0] + mem_tot_CPU_1 = 0
 rows_tot_CPU_1: - frac[1,3] - frac[1,2] - frac[1,5] - frac[1,1]
   - frac[1,4] - frac[1,0] + rows_tot_CPU_1 = 0
 mem_tot_P4_2: - mem[2,8] - mem[2,5] - mem[2,4] - mem[2,6] - mem[2,3]
   - mem[2,7] + mem_tot_P4_2 = 0
 rows_tot_P4_2: - frac[2,8] - frac[2,5] - frac[2,4] - frac[2,6] - frac[2,3]
   - frac[2,7] + rows_tot_P4_2 = 0
 mem_tot_P4_3: - mem[3,0] - mem[3,7] - mem[3,3] - mem[3,6] - mem[3,2]
   - mem[3,5] - mem[3,1] - mem[3,8] - mem[3,4] + mem_tot_P4_3 = 0
 rows_tot_P4_3: - frac[3,0] - frac[3,7] - frac[3,3] - frac[3,6] - frac[3,2]
   - frac[3,5] - frac[3,1] - frac[3,8] - frac[3,4] + rows_tot_P4_3 = 0
 mem_tot_P4_4: - mem[4,7] - mem[4,8] - mem[4,4] - mem[4,5] - mem[4,6]
   - mem[4,3] + mem_tot_P4_4 = 0
 rows_tot_P4_4: - frac[4,7] - frac[4,8] - frac[4,4] - frac[4,5] - frac[4,6]
   - frac[4,3] + rows_tot_P4_4 = 0
 mem_tot_P4_5: - mem[5,1] - mem[5,5] - mem[5,0] - mem[5,4] - mem[5,2]
   - mem[5,3] + mem_tot_P4_5 = 0
 rows_tot_P4_5: - frac[5,1] - frac[5,5] - frac[5,0] - frac[5,4] - frac[5,2]
   - frac[5,3] + rows_tot_P4_5 = 0
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
 frac[4,7] <= 1
 frac[1,3] <= 1
 frac[4,8] <= 1
 frac[3,0] <= 1
 frac[2,8] <= 1
 frac[5,1] <= 1
 frac[0,3] <= 1
 frac[2,5] <= 1
 frac[3,7] <= 1
 frac[2,4] <= 1
 frac[1,2] <= 1
 frac[3,3] <= 1
 frac[5,5] <= 1
 frac[4,4] <= 1
 frac[1,5] <= 1
 frac[5,0] <= 1
 frac[0,4] <= 1
 frac[3,6] <= 1
 frac[1,1] <= 1
 frac[3,2] <= 1
 frac[0,0] <= 1
 frac[5,4] <= 1
 frac[2,6] <= 1
 frac[4,5] <= 1
 frac[5,2] <= 1
 frac[1,4] <= 1
 frac[0,5] <= 1
 frac[2,3] <= 1
 frac[1,0] <= 1
 frac[5,3] <= 1
 frac[0,1] <= 1
 frac[3,5] <= 1
 frac[2,7] <= 1
 frac[4,6] <= 1
 frac[3,1] <= 1
 frac[3,8] <= 1
 frac[4,3] <= 1
 frac[3,4] <= 1
 frac[0,2] <= 1
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

\ Model netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 m_access_time_cpu_1 + 0 ns_sketch_cpu_1 + 0 ns_dpdk_cpu_1 + 0 ns_cpu_1
   + 0 m_access_time_cpu_2 + 0 ns_sketch_cpu_2 + 0 ns_dpdk_cpu_2
   + 0 ns_cpu_2 + ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0
   res
Subject To
 cov_0_0: frac[0,0] + frac[1,0] + frac[2,0] + frac[3,0] >= 1
 cov_0_1: frac[0,1] + frac[1,1] + frac[2,1] + frac[3,1] >= 1
 cov_0_2: frac[0,2] + frac[1,2] + frac[2,2] + frac[3,2] >= 1
 accuracy_0_sk_0[0]: - 424.734375 frac[0,0] + mem[0,0] >= 0
 accuracy_0_sk_0[1]: - 424.734375 frac[1,0] + mem[1,0] >= 0
 accuracy_0_sk_0[2]: - 424.734375 frac[2,0] + mem[2,0] >= 0
 accuracy_0_sk_0[3]: - 424.734375 frac[3,0] + mem[3,0] >= 0
 accuracy_1_sk_1[0]: - 25483.8984375 frac[0,1] + mem[0,1] >= 0
 accuracy_1_sk_1[1]: - 25483.8984375 frac[1,1] + mem[1,1] >= 0
 accuracy_1_sk_1[2]: - 25483.8984375 frac[2,1] + mem[2,1] >= 0
 accuracy_1_sk_1[3]: - 25483.8984375 frac[3,1] + mem[3,1] >= 0
 accuracy_2_sk_2[0]: - 67.96875 frac[0,2] + mem[0,2] >= 0
 accuracy_2_sk_2[1]: - 67.96875 frac[1,2] + mem[1,2] >= 0
 accuracy_2_sk_2[2]: - 67.96875 frac[2,2] + mem[2,2] >= 0
 accuracy_2_sk_2[3]: - 67.96875 frac[3,2] + mem[3,2] >= 0
 mem_tot_cpu_1: - mem[0,0] - mem[0,1] - mem[0,2] + mem_tot_cpu_1 = 0
 rows_tot_cpu_1: - 3 frac[0,0] - 3 frac[0,1] - 4 frac[0,2] + rows_tot_cpu_1
   = 0
 mem_tot_cpu_2: - mem[1,0] - mem[1,1] - mem[1,2] + mem_tot_cpu_2 = 0
 rows_tot_cpu_2: - 3 frac[1,0] - 3 frac[1,1] - 4 frac[1,2] + rows_tot_cpu_2
   = 0
 mem_tot_p4_1: - mem[2,0] - mem[2,1] - mem[2,2] + mem_tot_p4_1 = 0
 rows_tot_p4_1: - 3 frac[2,0] - 3 frac[2,1] - 4 frac[2,2] + rows_tot_p4_1
   = 0
 mem_tot_p4_2: - mem[3,0] - mem[3,1] - mem[3,2] + mem_tot_p4_2 = 0
 rows_tot_p4_2: - 3 frac[3,0] - 3 frac[3,1] - 4 frac[3,2] + rows_tot_p4_2
   = 0
 row_capacity_cpu_1: rows_tot_cpu_1 <= 12
 row_capacity_cpu_2: rows_tot_cpu_2 <= 12
 row_capacity_p4_1: rows_tot_p4_1 <= 48
 row_capacity_p4_2: rows_tot_p4_2 <= 48
 ns_single_cpu_1: 3.5 rows_tot_cpu_1 - ns_single_cpu_1 + pdt_m_rows_cpu_1
   = 0
 capacity_cores_cpu_1: cores_sketch_cpu_1 + cores_dpdk_cpu_1 <= 7
 ns_sketch_cpu_1: - ns_single_cpu_1 + pdt_nsk_c_cpu_1 = 0
 ns_dpdk_cpu_1: 7.142857142857143 cores_dpdk_cpu_1 - pdt_nsc_dpdk_cpu_1
   = -21.42857142857143
 ns_single_cpu_2: 3.5 rows_tot_cpu_2 - ns_single_cpu_2 + pdt_m_rows_cpu_2
   = 0
 capacity_cores_cpu_2: cores_sketch_cpu_2 + cores_dpdk_cpu_2 <= 7
 ns_sketch_cpu_2: - ns_single_cpu_2 + pdt_nsk_c_cpu_2 = 0
 ns_dpdk_cpu_2: 7.142857142857143 cores_dpdk_cpu_2 - pdt_nsc_dpdk_cpu_2
   = -21.42857142857143
 ns_p4_1: ns_p4_1 = 6.756756756756757
 ns_p4_2: ns_p4_2 = 6.756756756756757
 res_acc: 1.220703125e-04 mem_tot_cpu_1 + 1.220703125e-04 mem_tot_cpu_2
   + 0.0208333333333333 mem_tot_p4_1 + 0.25 rows_tot_p4_1
   + 0.0208333333333333 mem_tot_p4_2 + 0.25 rows_tot_p4_2
   + 10 cores_sketch_cpu_1 + 10 cores_dpdk_cpu_1 + 10 cores_sketch_cpu_2
   + 10 cores_dpdk_cpu_2 - res = 0
 pdt_m_rows_cpu_1: pdt_m_rows_cpu_1 + [
   - rows_tot_cpu_1 * m_access_time_cpu_1 ] = 0
 pdt_nsk_c_cpu_1: pdt_nsk_c_cpu_1 + [
   - cores_sketch_cpu_1 * ns_sketch_cpu_1 ] = 0
 pdt_nsc_dpdk_cpu_1: pdt_nsc_dpdk_cpu_1 + [
   - cores_dpdk_cpu_1 * ns_dpdk_cpu_1 ] = 0
 pdt_m_rows_cpu_2: pdt_m_rows_cpu_2 + [
   - rows_tot_cpu_2 * m_access_time_cpu_2 ] = 0
 pdt_nsk_c_cpu_2: pdt_nsk_c_cpu_2 + [
   - cores_sketch_cpu_2 * ns_sketch_cpu_2 ] = 0
 pdt_nsc_dpdk_cpu_2: pdt_nsc_dpdk_cpu_2 + [
   - cores_dpdk_cpu_2 * ns_dpdk_cpu_2 ] = 0
Bounds
 mem_tot_cpu_1 <= 32768
 mem_tot_cpu_2 <= 32768
 mem_tot_p4_1 <= 576
 mem_tot_p4_2 <= 576
 cores_sketch_cpu_1 <= 7
 1 <= cores_dpdk_cpu_1 <= 7
 cores_sketch_cpu_2 <= 7
 1 <= cores_dpdk_cpu_2 <= 7
Binaries
 frac[0,0] frac[0,1] frac[0,2] frac[1,0] frac[1,1] frac[1,2] frac[2,0]
 frac[2,1] frac[2,2] frac[3,0] frac[3,1] frac[3,2]
Generals
 cores_sketch_cpu_1 cores_dpdk_cpu_1 cores_sketch_cpu_2 cores_dpdk_cpu_2
General Constraints
 mem_access_time_cpu_1: m_access_time_cpu_1 = PWL ( mem_tot_cpu_1 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_1: ns_cpu_1 = MAX ( ns_dpdk_cpu_1 , ns_sketch_cpu_1 )
 mem_access_time_cpu_2: m_access_time_cpu_2 = PWL ( mem_tot_cpu_2 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_2: ns_cpu_2 = MAX ( ns_dpdk_cpu_2 , ns_sketch_cpu_2 )
 ns_overall: ns = MAX ( ns_cpu_1 , ns_cpu_2 , ns_p4_1 , ns_p4_2 )
End

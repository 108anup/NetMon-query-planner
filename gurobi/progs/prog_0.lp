\ Model netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0.05
   0 m_access_time_cpu_1 + 0 ns_sketch_cpu_1 + 0 ns_dpdk_cpu_1 + 0 ns_cpu_1
   + 0 m_access_time_cpu_2 + 0 ns_sketch_cpu_2 + 0 ns_dpdk_cpu_2
   + 0 ns_cpu_2 + ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0.05
   res_overall
Subject To
 cov_0: frac[0,0] + frac[1,0] + frac[2,0] + frac[3,0] = 1
 cov_1: frac[0,1] + frac[1,1] + frac[2,1] + frac[3,1] = 1
 cov_2: frac[0,2] + frac[1,2] + frac[2,2] + frac[3,2] = 1
 cov_3: frac[0,3] + frac[1,3] + frac[2,3] + frac[3,3] = 1
 cov_4: frac[0,4] + frac[1,4] + frac[2,4] + frac[3,4] = 1
 cov_5: frac[0,5] + frac[1,5] + frac[2,5] + frac[3,5] = 1
 cov_6: frac[0,6] + frac[1,6] + frac[2,6] + frac[3,6] = 1
 cov_7: frac[0,7] + frac[1,7] + frac[2,7] + frac[3,7] = 1
 cov_8: frac[0,8] + frac[1,8] + frac[2,8] + frac[3,8] = 1
 cov_9: frac[0,9] + frac[1,9] + frac[2,9] + frac[3,9] = 1
 cov_10: frac[0,10] + frac[1,10] + frac[2,10] + frac[3,10] = 1
 cov_11: frac[0,11] + frac[1,11] + frac[2,11] + frac[3,11] = 1
 cov_12: frac[0,12] + frac[1,12] + frac[2,12] + frac[3,12] = 1
 accuracy_(1,_sk_1)[0]: - 21.23828125 frac[0,0] + mem[0,0] >= 0
 accuracy_(1,_sk_1)[1]: - 21.23828125 frac[1,0] + mem[1,0] >= 0
 accuracy_(1,_sk_1)[2]: - 21.23828125 frac[2,0] + mem[2,0] >= 0
 accuracy_(1,_sk_1)[3]: - 21.23828125 frac[3,0] + mem[3,0] >= 0
 accuracy_(2,_sk_1)[0]: - 21.23828125 frac[0,1] + mem[0,1] >= 0
 accuracy_(2,_sk_1)[1]: - 21.23828125 frac[1,1] + mem[1,1] >= 0
 accuracy_(2,_sk_1)[2]: - 21.23828125 frac[2,1] + mem[2,1] >= 0
 accuracy_(2,_sk_1)[3]: - 21.23828125 frac[3,1] + mem[3,1] >= 0
 accuracy_(3,_sk_1)[0]: - 21.23828125 frac[0,2] + mem[0,2] >= 0
 accuracy_(3,_sk_1)[1]: - 21.23828125 frac[1,2] + mem[1,2] >= 0
 accuracy_(3,_sk_1)[2]: - 21.23828125 frac[2,2] + mem[2,2] >= 0
 accuracy_(3,_sk_1)[3]: - 21.23828125 frac[3,2] + mem[3,2] >= 0
 accuracy_(4,_sk_1)[0]: - 21.23828125 frac[0,3] + mem[0,3] >= 0
 accuracy_(4,_sk_1)[1]: - 21.23828125 frac[1,3] + mem[1,3] >= 0
 accuracy_(4,_sk_1)[2]: - 21.23828125 frac[2,3] + mem[2,3] >= 0
 accuracy_(4,_sk_1)[3]: - 21.23828125 frac[3,3] + mem[3,3] >= 0
 accuracy_(1,_sk_2)[0]: - 5309.14453125 frac[0,4] + mem[0,4] >= 0
 accuracy_(1,_sk_2)[1]: - 5309.14453125 frac[1,4] + mem[1,4] >= 0
 accuracy_(1,_sk_2)[2]: - 5309.14453125 frac[2,4] + mem[2,4] >= 0
 accuracy_(1,_sk_2)[3]: - 5309.14453125 frac[3,4] + mem[3,4] >= 0
 accuracy_(2,_sk_2)[0]: - 5309.14453125 frac[0,5] + mem[0,5] >= 0
 accuracy_(2,_sk_2)[1]: - 5309.14453125 frac[1,5] + mem[1,5] >= 0
 accuracy_(2,_sk_2)[2]: - 5309.14453125 frac[2,5] + mem[2,5] >= 0
 accuracy_(2,_sk_2)[3]: - 5309.14453125 frac[3,5] + mem[3,5] >= 0
 accuracy_(3,_sk_2)[0]: - 5309.14453125 frac[0,6] + mem[0,6] >= 0
 accuracy_(3,_sk_2)[1]: - 5309.14453125 frac[1,6] + mem[1,6] >= 0
 accuracy_(3,_sk_2)[2]: - 5309.14453125 frac[2,6] + mem[2,6] >= 0
 accuracy_(3,_sk_2)[3]: - 5309.14453125 frac[3,6] + mem[3,6] >= 0
 accuracy_(4,_sk_2)[0]: - 5309.14453125 frac[0,7] + mem[0,7] >= 0
 accuracy_(4,_sk_2)[1]: - 5309.14453125 frac[1,7] + mem[1,7] >= 0
 accuracy_(4,_sk_2)[2]: - 5309.14453125 frac[2,7] + mem[2,7] >= 0
 accuracy_(4,_sk_2)[3]: - 5309.14453125 frac[3,7] + mem[3,7] >= 0
 accuracy_(1,_sk_3)[0]: - 10.62109375 frac[0,8] + mem[0,8] >= 0
 accuracy_(1,_sk_3)[1]: - 10.62109375 frac[1,8] + mem[1,8] >= 0
 accuracy_(1,_sk_3)[2]: - 10.62109375 frac[2,8] + mem[2,8] >= 0
 accuracy_(1,_sk_3)[3]: - 10.62109375 frac[3,8] + mem[3,8] >= 0
 accuracy_(2,_sk_3)[0]: - 10.62109375 frac[0,9] + mem[0,9] >= 0
 accuracy_(2,_sk_3)[1]: - 10.62109375 frac[1,9] + mem[1,9] >= 0
 accuracy_(2,_sk_3)[2]: - 10.62109375 frac[2,9] + mem[2,9] >= 0
 accuracy_(2,_sk_3)[3]: - 10.62109375 frac[3,9] + mem[3,9] >= 0
 accuracy_(3,_sk_3)[0]: - 10.62109375 frac[0,10] + mem[0,10] >= 0
 accuracy_(3,_sk_3)[1]: - 10.62109375 frac[1,10] + mem[1,10] >= 0
 accuracy_(3,_sk_3)[2]: - 10.62109375 frac[2,10] + mem[2,10] >= 0
 accuracy_(3,_sk_3)[3]: - 10.62109375 frac[3,10] + mem[3,10] >= 0
 accuracy_(4,_sk_3)[0]: - 10.62109375 frac[0,11] + mem[0,11] >= 0
 accuracy_(4,_sk_3)[1]: - 10.62109375 frac[1,11] + mem[1,11] >= 0
 accuracy_(4,_sk_3)[2]: - 10.62109375 frac[2,11] + mem[2,11] >= 0
 accuracy_(4,_sk_3)[3]: - 10.62109375 frac[3,11] + mem[3,11] >= 0
 accuracy_(5,_sk_3)[0]: - 10.62109375 frac[0,12] + mem[0,12] >= 0
 accuracy_(5,_sk_3)[1]: - 10.62109375 frac[1,12] + mem[1,12] >= 0
 accuracy_(5,_sk_3)[2]: - 10.62109375 frac[2,12] + mem[2,12] >= 0
 accuracy_(5,_sk_3)[3]: - 10.62109375 frac[3,12] + mem[3,12] >= 0
 rows_tot_cpu_1: - frac[0,0] - frac[0,1] - frac[0,2] - frac[0,3]
   - frac[0,4] - frac[0,5] - frac[0,6] - frac[0,7] - frac[0,8] - frac[0,9]
   - frac[0,10] - frac[0,11] - frac[0,12] + rows_tot_cpu_1 = 0
 mem_tot_cpu_1: - mem[0,0] - mem[0,1] - mem[0,2] - mem[0,3] - mem[0,4]
   - mem[0,5] - mem[0,6] - mem[0,7] - mem[0,8] - mem[0,9] - mem[0,10]
   - mem[0,11] - mem[0,12] + mem_tot_cpu_1 = 0
 ns_single_cpu_1: 3.5 rows_tot_cpu_1 - ns_single_cpu_1 + pdt_m_rows_cpu_1
   <= 0
 capacity_cores_cpu_1: cores_sketch_cpu_1 + cores_dpdk_cpu_1 <= 7
 ns_sketch_cpu_1: - ns_single_cpu_1 + pdt_nsk_c_cpu_1 = 0
 ns_dpdk_cpu_1: 7.142857142857143 cores_dpdk_cpu_1 - pdt_nsc_dpdk_cpu_1
   = -21.42857142857143
 rows_tot_cpu_2: - frac[1,0] - frac[1,1] - frac[1,2] - frac[1,3]
   - frac[1,4] - frac[1,5] - frac[1,6] - frac[1,7] - frac[1,8] - frac[1,9]
   - frac[1,10] - frac[1,11] - frac[1,12] + rows_tot_cpu_2 = 0
 mem_tot_cpu_2: - mem[1,0] - mem[1,1] - mem[1,2] - mem[1,3] - mem[1,4]
   - mem[1,5] - mem[1,6] - mem[1,7] - mem[1,8] - mem[1,9] - mem[1,10]
   - mem[1,11] - mem[1,12] + mem_tot_cpu_2 = 0
 ns_single_cpu_2: 3.5 rows_tot_cpu_2 - ns_single_cpu_2 + pdt_m_rows_cpu_2
   <= 0
 capacity_cores_cpu_2: cores_sketch_cpu_2 + cores_dpdk_cpu_2 <= 7
 ns_sketch_cpu_2: - ns_single_cpu_2 + pdt_nsk_c_cpu_2 = 0
 ns_dpdk_cpu_2: 7.142857142857143 cores_dpdk_cpu_2 - pdt_nsc_dpdk_cpu_2
   = -21.42857142857143
 rows_tot_p4_1: - frac[2,0] - frac[2,1] - frac[2,2] - frac[2,3] - frac[2,4]
   - frac[2,5] - frac[2,6] - frac[2,7] - frac[2,8] - frac[2,9] - frac[2,10]
   - frac[2,11] - frac[2,12] + rows_tot_p4_1 = 0
 mem_tot_p4_1: - mem[2,0] - mem[2,1] - mem[2,2] - mem[2,3] - mem[2,4]
   - mem[2,5] - mem[2,6] - mem[2,7] - mem[2,8] - mem[2,9] - mem[2,10]
   - mem[2,11] - mem[2,12] + mem_tot_p4_1 = 0
 capacity_mem_par_p4_1: mem[2,0] <= 48
 capacity_mem_par_p4_1: mem[2,1] <= 48
 capacity_mem_par_p4_1: mem[2,2] <= 48
 capacity_mem_par_p4_1: mem[2,3] <= 48
 capacity_mem_par_p4_1: mem[2,4] <= 48
 capacity_mem_par_p4_1: mem[2,5] <= 48
 capacity_mem_par_p4_1: mem[2,6] <= 48
 capacity_mem_par_p4_1: mem[2,7] <= 48
 capacity_mem_par_p4_1: mem[2,8] <= 48
 capacity_mem_par_p4_1: mem[2,9] <= 48
 capacity_mem_par_p4_1: mem[2,10] <= 48
 capacity_mem_par_p4_1: mem[2,11] <= 48
 capacity_mem_par_p4_1: mem[2,12] <= 48
 ns_p4_1: ns_p4_1 = 6.756756756756757
 rows_tot_p4_2: - frac[3,0] - frac[3,1] - frac[3,2] - frac[3,3] - frac[3,4]
   - frac[3,5] - frac[3,6] - frac[3,7] - frac[3,8] - frac[3,9] - frac[3,10]
   - frac[3,11] - frac[3,12] + rows_tot_p4_2 = 0
 mem_tot_p4_2: - mem[3,0] - mem[3,1] - mem[3,2] - mem[3,3] - mem[3,4]
   - mem[3,5] - mem[3,6] - mem[3,7] - mem[3,8] - mem[3,9] - mem[3,10]
   - mem[3,11] - mem[3,12] + mem_tot_p4_2 = 0
 capacity_mem_par_p4_2: mem[3,0] <= 48
 capacity_mem_par_p4_2: mem[3,1] <= 48
 capacity_mem_par_p4_2: mem[3,2] <= 48
 capacity_mem_par_p4_2: mem[3,3] <= 48
 capacity_mem_par_p4_2: mem[3,4] <= 48
 capacity_mem_par_p4_2: mem[3,5] <= 48
 capacity_mem_par_p4_2: mem[3,6] <= 48
 capacity_mem_par_p4_2: mem[3,7] <= 48
 capacity_mem_par_p4_2: mem[3,8] <= 48
 capacity_mem_par_p4_2: mem[3,9] <= 48
 capacity_mem_par_p4_2: mem[3,10] <= 48
 capacity_mem_par_p4_2: mem[3,11] <= 48
 capacity_mem_par_p4_2: mem[3,12] <= 48
 ns_p4_2: ns_p4_2 = 6.756756756756757
 res: - 1.220703125e-04 mem_tot_cpu_1 - 10 cores_sketch_cpu_1
   - 10 cores_dpdk_cpu_1 - 1.220703125e-04 mem_tot_cpu_2
   - 10 cores_sketch_cpu_2 - 10 cores_dpdk_cpu_2 - 0.25 rows_tot_p4_1
   - 0.0208333333333333 mem_tot_p4_1 - 0.25 rows_tot_p4_2
   - 0.0208333333333333 mem_tot_p4_2 + res_overall = 0
 pdt_m_rows_cpu_1: pdt_m_rows_cpu_1 + [
   - rows_tot_cpu_1 * m_access_time_cpu_1 ] >= 0
 pdt_nsk_c_cpu_1: pdt_nsk_c_cpu_1 + [
   - cores_sketch_cpu_1 * ns_sketch_cpu_1 ] <= 0
 pdt_nsc_dpdk_cpu_1: pdt_nsc_dpdk_cpu_1 + [
   - cores_dpdk_cpu_1 * ns_dpdk_cpu_1 ] <= 0
 pdt_m_rows_cpu_2: pdt_m_rows_cpu_2 + [
   - rows_tot_cpu_2 * m_access_time_cpu_2 ] >= 0
 pdt_nsk_c_cpu_2: pdt_nsk_c_cpu_2 + [
   - cores_sketch_cpu_2 * ns_sketch_cpu_2 ] <= 0
 pdt_nsc_dpdk_cpu_2: pdt_nsc_dpdk_cpu_2 + [
   - cores_dpdk_cpu_2 * ns_dpdk_cpu_2 ] <= 0
Bounds
 frac[0,0] <= 1
 frac[0,1] <= 1
 frac[0,2] <= 1
 frac[0,3] <= 1
 frac[0,4] <= 1
 frac[0,5] <= 1
 frac[0,6] <= 1
 frac[0,7] <= 1
 frac[0,8] <= 1
 frac[0,9] <= 1
 frac[0,10] <= 1
 frac[0,11] <= 1
 frac[0,12] <= 1
 frac[1,0] <= 1
 frac[1,1] <= 1
 frac[1,2] <= 1
 frac[1,3] <= 1
 frac[1,4] <= 1
 frac[1,5] <= 1
 frac[1,6] <= 1
 frac[1,7] <= 1
 frac[1,8] <= 1
 frac[1,9] <= 1
 frac[1,10] <= 1
 frac[1,11] <= 1
 frac[1,12] <= 1
 frac[2,0] <= 1
 frac[2,1] <= 1
 frac[2,2] <= 1
 frac[2,3] <= 1
 frac[2,4] <= 1
 frac[2,5] <= 1
 frac[2,6] <= 1
 frac[2,7] <= 1
 frac[2,8] <= 1
 frac[2,9] <= 1
 frac[2,10] <= 1
 frac[2,11] <= 1
 frac[2,12] <= 1
 frac[3,0] <= 1
 frac[3,1] <= 1
 frac[3,2] <= 1
 frac[3,3] <= 1
 frac[3,4] <= 1
 frac[3,5] <= 1
 frac[3,6] <= 1
 frac[3,7] <= 1
 frac[3,8] <= 1
 frac[3,9] <= 1
 frac[3,10] <= 1
 frac[3,11] <= 1
 frac[3,12] <= 1
 rows_tot_cpu_1 <= 9
 mem_tot_cpu_1 <= 32768
 cores_sketch_cpu_1 <= 7
 3 <= cores_dpdk_cpu_1 <= 7
 rows_tot_cpu_2 <= 9
 mem_tot_cpu_2 <= 32768
 cores_sketch_cpu_2 <= 7
 3 <= cores_dpdk_cpu_2 <= 7
 rows_tot_p4_1 <= 12
 mem_tot_p4_1 <= 576
 rows_tot_p4_2 <= 12
 mem_tot_p4_2 <= 576
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

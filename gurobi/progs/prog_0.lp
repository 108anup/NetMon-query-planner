\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_CPU_Cluster + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU4175
   + 0 m_access_time_CPU5912 + 0 m_access_time_CPU3785
   + 0 m_access_time_CPU9475 + 0 m_access_time_CPU4065
   + 0 m_access_time_CPU4863 + 0 m_access_time_CPU899
   + 0 m_access_time_CPU9275 + res
Subject To
 cov_0_0: frac[0,0] + frac[1,0] >= 0.7
 cov_1_1: frac[5,1] + frac[1,1] >= 0.7
 cov_2_0: frac[7,0] >= 0.9
 cov_3_1: frac[5,1] >= 0.9
 cov_4_0: frac[7,0] + frac[2,0] >= 0.7
 cov_5_1: frac[3,1] + frac[4,1] >= 0.8
 cov_6_1: frac[7,1] + frac[2,1] >= 0.9
 accuracy_0_0: - 509.68359375 frac[0,0] + mem[0,0] = 0
 accuracy_7_0: - 509.68359375 frac[7,0] + mem[7,0] = 0
 accuracy_7_1: - 509.68359375 frac[7,1] + mem[7,1] = 0
 accuracy_3_1: - 509.68359375 frac[3,1] + mem[3,1] = 0
 accuracy_2_1: - 509.68359375 frac[2,1] + mem[2,1] = 0
 accuracy_2_0: - 509.68359375 frac[2,0] + mem[2,0] = 0
 accuracy_5_1: - 509.68359375 frac[5,1] + mem[5,1] = 0
 accuracy_1_0: - 509.68359375 frac[1,0] + mem[1,0] = 0
 accuracy_4_1: - 509.68359375 frac[4,1] + mem[4,1] = 0
 accuracy_1_1: - 509.68359375 frac[1,1] + mem[1,1] = 0
 mem_tot_CPU4175: - mem[0,0] + mem_tot_CPU4175 = 0
 R18: - 3.05176e-05 mem_tot_CPU4175 + normalized_mem_tot_CPU4175 = 0
 rows_tot_CPU4175: - 3 frac[0,0] + rows_tot_CPU4175 = 0
 R20: - 0.0833333333333333 rows_tot_CPU4175 + normalized_rows_tot_CPU4175
   = 0
 mem_tot_CPU5912: - mem[1,0] - mem[1,1] + mem_tot_CPU5912 = 0
 R22: - 3.05176e-05 mem_tot_CPU5912 + normalized_mem_tot_CPU5912 = 0
 rows_tot_CPU5912: - 3 frac[1,0] - 3 frac[1,1] + rows_tot_CPU5912 = 0
 R24: - 0.0833333333333333 rows_tot_CPU5912 + normalized_rows_tot_CPU5912
   = 0
 mem_tot_CPU3785: - mem[2,1] - mem[2,0] + mem_tot_CPU3785 = 0
 R26: - 3.05176e-05 mem_tot_CPU3785 + normalized_mem_tot_CPU3785 = 0
 rows_tot_CPU3785: - 3 frac[2,1] - 3 frac[2,0] + rows_tot_CPU3785 = 0
 R28: - 0.0833333333333333 rows_tot_CPU3785 + normalized_rows_tot_CPU3785
   = 0
 mem_tot_CPU9475: - mem[3,1] + mem_tot_CPU9475 = 0
 R30: - 3.05176e-05 mem_tot_CPU9475 + normalized_mem_tot_CPU9475 = 0
 rows_tot_CPU9475: - 3 frac[3,1] + rows_tot_CPU9475 = 0
 R32: - 0.0833333333333333 rows_tot_CPU9475 + normalized_rows_tot_CPU9475
   = 0
 mem_tot_CPU4065: - mem[4,1] + mem_tot_CPU4065 = 0
 R34: - 3.05176e-05 mem_tot_CPU4065 + normalized_mem_tot_CPU4065 = 0
 rows_tot_CPU4065: - 3 frac[4,1] + rows_tot_CPU4065 = 0
 R36: - 0.0833333333333333 rows_tot_CPU4065 + normalized_rows_tot_CPU4065
   = 0
 mem_tot_CPU4863: - mem[5,1] + mem_tot_CPU4863 = 0
 R38: - 3.05176e-05 mem_tot_CPU4863 + normalized_mem_tot_CPU4863 = 0
 rows_tot_CPU4863: - 3 frac[5,1] + rows_tot_CPU4863 = 0
 R40: - 0.0833333333333333 rows_tot_CPU4863 + normalized_rows_tot_CPU4863
   = 0
 mem_tot_CPU899: mem_tot_CPU899 = 0
 R42: - 3.05176e-05 mem_tot_CPU899 + normalized_mem_tot_CPU899 = 0
 rows_tot_CPU899: rows_tot_CPU899 = 0
 R44: - 0.0833333333333333 rows_tot_CPU899 + normalized_rows_tot_CPU899
   = 0
 mem_tot_CPU9275: - mem[7,0] - mem[7,1] + mem_tot_CPU9275 = 0
 R46: - 3.05176e-05 mem_tot_CPU9275 + normalized_mem_tot_CPU9275 = 0
 rows_tot_CPU9275: - 3 frac[7,0] - 3 frac[7,1] + rows_tot_CPU9275 = 0
 R48: - 0.0833333333333333 rows_tot_CPU9275 + normalized_rows_tot_CPU9275
   = 0
 row_capacity_CPU4175: rows_tot_CPU4175 <= 12
 row_capacity_CPU5912: rows_tot_CPU5912 <= 12
 row_capacity_CPU3785: rows_tot_CPU3785 <= 12
 row_capacity_CPU9475: rows_tot_CPU9475 <= 12
 row_capacity_CPU4065: rows_tot_CPU4065 <= 12
 row_capacity_CPU4863: rows_tot_CPU4863 <= 12
 row_capacity_CPU899: rows_tot_CPU899 <= 12
 row_capacity_CPU9275: rows_tot_CPU9275 <= 12
 ns_single_CPU4175: 3.5 rows_tot_CPU4175 - ns_single_CPU4175
   + pdt_m_rows_CPU4175 = 0
 capacity_cores_CPU4175: cores_sketch_CPU4175 + cores_dpdk_CPU4175 <= 7
 ns_sketch_CPU4175: - ns_single_CPU4175
   + 11.42857142857143 cores_sketch_CPU4175 >= 0
 ns_dpdk_CPU4175: - 4.285714285714287 cores_dpdk_CPU4175
   <= -21.42857142857143
 ns_single_CPU5912: 3.5 rows_tot_CPU5912 - ns_single_CPU5912
   + pdt_m_rows_CPU5912 = 0
 capacity_cores_CPU5912: cores_sketch_CPU5912 + cores_dpdk_CPU5912 <= 7
 ns_sketch_CPU5912: - ns_single_CPU5912
   + 11.42857142857143 cores_sketch_CPU5912 >= 0
 ns_dpdk_CPU5912: - 4.285714285714287 cores_dpdk_CPU5912
   <= -21.42857142857143
 ns_single_CPU3785: 3.5 rows_tot_CPU3785 - ns_single_CPU3785
   + pdt_m_rows_CPU3785 = 0
 capacity_cores_CPU3785: cores_sketch_CPU3785 + cores_dpdk_CPU3785 <= 7
 ns_sketch_CPU3785: - ns_single_CPU3785
   + 11.42857142857143 cores_sketch_CPU3785 >= 0
 ns_dpdk_CPU3785: - 4.285714285714287 cores_dpdk_CPU3785
   <= -21.42857142857143
 ns_single_CPU9475: 3.5 rows_tot_CPU9475 - ns_single_CPU9475
   + pdt_m_rows_CPU9475 = 0
 capacity_cores_CPU9475: cores_sketch_CPU9475 + cores_dpdk_CPU9475 <= 7
 ns_sketch_CPU9475: - ns_single_CPU9475
   + 11.42857142857143 cores_sketch_CPU9475 >= 0
 ns_dpdk_CPU9475: - 4.285714285714287 cores_dpdk_CPU9475
   <= -21.42857142857143
 ns_single_CPU4065: 3.5 rows_tot_CPU4065 - ns_single_CPU4065
   + pdt_m_rows_CPU4065 = 0
 capacity_cores_CPU4065: cores_sketch_CPU4065 + cores_dpdk_CPU4065 <= 7
 ns_sketch_CPU4065: - ns_single_CPU4065
   + 11.42857142857143 cores_sketch_CPU4065 >= 0
 ns_dpdk_CPU4065: - 4.285714285714287 cores_dpdk_CPU4065
   <= -21.42857142857143
 ns_single_CPU4863: 3.5 rows_tot_CPU4863 - ns_single_CPU4863
   + pdt_m_rows_CPU4863 = 0
 capacity_cores_CPU4863: cores_sketch_CPU4863 + cores_dpdk_CPU4863 <= 7
 ns_sketch_CPU4863: - ns_single_CPU4863
   + 11.42857142857143 cores_sketch_CPU4863 >= 0
 ns_dpdk_CPU4863: - 4.285714285714287 cores_dpdk_CPU4863
   <= -21.42857142857143
 ns_single_CPU899: 3.5 rows_tot_CPU899 - ns_single_CPU899
   + pdt_m_rows_CPU899 = 0
 capacity_cores_CPU899: cores_sketch_CPU899 + cores_dpdk_CPU899 <= 7
 ns_sketch_CPU899: - ns_single_CPU899
   + 11.42857142857143 cores_sketch_CPU899 >= 0
 ns_dpdk_CPU899: - 4.285714285714287 cores_dpdk_CPU899
   <= -21.42857142857143
 ns_single_CPU9275: 3.5 rows_tot_CPU9275 - ns_single_CPU9275
   + pdt_m_rows_CPU9275 = 0
 capacity_cores_CPU9275: cores_sketch_CPU9275 + cores_dpdk_CPU9275 <= 7
 ns_sketch_CPU9275: - ns_single_CPU9275
   + 11.42857142857143 cores_sketch_CPU9275 >= 0
 ns_dpdk_CPU9275: - 4.285714285714287 cores_dpdk_CPU9275
   <= -21.42857142857143
 res_acc: 1.220703125e-04 mem_tot_CPU4175 + 1.220703125e-04 mem_tot_CPU5912
   + 1.220703125e-04 mem_tot_CPU3785 + 1.220703125e-04 mem_tot_CPU9475
   + 1.220703125e-04 mem_tot_CPU4065 + 1.220703125e-04 mem_tot_CPU4863
   + 1.220703125e-04 mem_tot_CPU899 + 1.220703125e-04 mem_tot_CPU9275
   + 10 cores_sketch_CPU4175 + 10 cores_dpdk_CPU4175
   + 10 cores_sketch_CPU5912 + 10 cores_dpdk_CPU5912
   + 10 cores_sketch_CPU3785 + 10 cores_dpdk_CPU3785
   + 10 cores_sketch_CPU9475 + 10 cores_dpdk_CPU9475
   + 10 cores_sketch_CPU4065 + 10 cores_dpdk_CPU4065
   + 10 cores_sketch_CPU4863 + 10 cores_dpdk_CPU4863
   + 10 cores_sketch_CPU899 + 10 cores_dpdk_CPU899
   + 10 cores_sketch_CPU9275 + 10 cores_dpdk_CPU9275 - res = 0
 pdt_m_rows_CPU4175: pdt_m_rows_CPU4175 + [
   - rows_tot_CPU4175 * m_access_time_CPU4175 ] = 0
 pdt_m_rows_CPU5912: pdt_m_rows_CPU5912 + [
   - rows_tot_CPU5912 * m_access_time_CPU5912 ] = 0
 pdt_m_rows_CPU3785: pdt_m_rows_CPU3785 + [
   - rows_tot_CPU3785 * m_access_time_CPU3785 ] = 0
 pdt_m_rows_CPU9475: pdt_m_rows_CPU9475 + [
   - rows_tot_CPU9475 * m_access_time_CPU9475 ] = 0
 pdt_m_rows_CPU4065: pdt_m_rows_CPU4065 + [
   - rows_tot_CPU4065 * m_access_time_CPU4065 ] = 0
 pdt_m_rows_CPU4863: pdt_m_rows_CPU4863 + [
   - rows_tot_CPU4863 * m_access_time_CPU4863 ] = 0
 pdt_m_rows_CPU899: pdt_m_rows_CPU899 + [
   - rows_tot_CPU899 * m_access_time_CPU899 ] = 0
 pdt_m_rows_CPU9275: pdt_m_rows_CPU9275 + [
   - rows_tot_CPU9275 * m_access_time_CPU9275 ] = 0
Bounds
 mem_tot_CPU4175 <= 32768
 normalized_mem_tot_CPU4175 <= 1
 normalized_rows_tot_CPU4175 <= 1
 mem_tot_CPU5912 <= 32768
 normalized_mem_tot_CPU5912 <= 1
 normalized_rows_tot_CPU5912 <= 1
 mem_tot_CPU3785 <= 32768
 normalized_mem_tot_CPU3785 <= 1
 normalized_rows_tot_CPU3785 <= 1
 mem_tot_CPU9475 <= 32768
 normalized_mem_tot_CPU9475 <= 1
 normalized_rows_tot_CPU9475 <= 1
 mem_tot_CPU4065 <= 32768
 normalized_mem_tot_CPU4065 <= 1
 normalized_rows_tot_CPU4065 <= 1
 mem_tot_CPU4863 <= 32768
 normalized_mem_tot_CPU4863 <= 1
 normalized_rows_tot_CPU4863 <= 1
 mem_tot_CPU899 <= 32768
 normalized_mem_tot_CPU899 <= 1
 normalized_rows_tot_CPU899 <= 1
 mem_tot_CPU9275 <= 32768
 normalized_mem_tot_CPU9275 <= 1
 normalized_rows_tot_CPU9275 <= 1
 cores_sketch_CPU4175 <= 7
 1 <= cores_dpdk_CPU4175 <= 7
 cores_sketch_CPU5912 <= 7
 1 <= cores_dpdk_CPU5912 <= 7
 cores_sketch_CPU3785 <= 7
 1 <= cores_dpdk_CPU3785 <= 7
 cores_sketch_CPU9475 <= 7
 1 <= cores_dpdk_CPU9475 <= 7
 cores_sketch_CPU4065 <= 7
 1 <= cores_dpdk_CPU4065 <= 7
 cores_sketch_CPU4863 <= 7
 1 <= cores_dpdk_CPU4863 <= 7
 cores_sketch_CPU899 <= 7
 1 <= cores_dpdk_CPU899 <= 7
 cores_sketch_CPU9275 <= 7
 1 <= cores_dpdk_CPU9275 <= 7
Binaries
 frac[0,0] frac[7,0] frac[7,1] frac[3,1] frac[2,1] frac[2,0] frac[5,1]
 frac[1,0] frac[4,1] frac[1,1]
Generals
 cores_sketch_CPU4175 cores_dpdk_CPU4175 cores_sketch_CPU5912
 cores_dpdk_CPU5912 cores_sketch_CPU3785 cores_dpdk_CPU3785
 cores_sketch_CPU9475 cores_dpdk_CPU9475 cores_sketch_CPU4065
 cores_dpdk_CPU4065 cores_sketch_CPU4863 cores_dpdk_CPU4863
 cores_sketch_CPU899 cores_dpdk_CPU899 cores_sketch_CPU9275
 cores_dpdk_CPU9275
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU4175 , normalized_mem_tot_CPU5912 ,
   normalized_mem_tot_CPU3785 , normalized_mem_tot_CPU9475 ,
   normalized_mem_tot_CPU4065 , normalized_mem_tot_CPU4863 ,
   normalized_mem_tot_CPU899 , normalized_mem_tot_CPU9275 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX (
   normalized_rows_tot_CPU4175 , normalized_rows_tot_CPU5912 ,
   normalized_rows_tot_CPU3785 , normalized_rows_tot_CPU9475 ,
   normalized_rows_tot_CPU4065 , normalized_rows_tot_CPU4863 ,
   normalized_rows_tot_CPU899 , normalized_rows_tot_CPU9275 )
 mem_access_time_CPU4175:
   m_access_time_CPU4175 = PWL ( mem_tot_CPU4175 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU5912:
   m_access_time_CPU5912 = PWL ( mem_tot_CPU5912 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU3785:
   m_access_time_CPU3785 = PWL ( mem_tot_CPU3785 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU9475:
   m_access_time_CPU9475 = PWL ( mem_tot_CPU9475 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU4065:
   m_access_time_CPU4065 = PWL ( mem_tot_CPU4065 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU4863:
   m_access_time_CPU4863 = PWL ( mem_tot_CPU4863 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU899: m_access_time_CPU899 = PWL ( mem_tot_CPU899 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU9275:
   m_access_time_CPU9275 = PWL ( mem_tot_CPU9275 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

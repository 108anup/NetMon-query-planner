\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_CPU_Cluster + 0 max_rows_CPU_Cluster + 0 m_access_time_CPU8997
   + 0 m_access_time_CPU2275 + 0 m_access_time_CPU4780
   + 0 m_access_time_CPU45 + 0 m_access_time_CPU9485
   + 0 m_access_time_CPU8187 + 0 m_access_time_CPU3809
   + 0 m_access_time_CPU8827 + 0 m_access_time_CPU5598
   + 0 m_access_time_CPU9261 + 0 m_access_time_CPU501
   + 0 m_access_time_CPU7283 + 0 m_access_time_CPU4767
   + 0 m_access_time_CPU9211 + 0 m_access_time_CPU4550
   + 0 m_access_time_CPU3713 + 0 m_access_time_CPU6795
   + 0 m_access_time_CPU9283 + 0 m_access_time_CPU1799
   + 0 m_access_time_CPU854 + res
Subject To
 cov_0_6: frac[17,6] >= 0.1029424881704133
 cov_1_6: frac[17,6] >= 0.1029424881704133
 cov_2_1: frac[4,1] >= 0.7
 cov_3_5: frac[5,5] >= 0.8
 cov_4_2: frac[0,2] >= 1
 cov_5_2: frac[0,2] >= 0.9
 cov_6_4: frac[6,4] >= 0.8285252342776411
 cov_7_3: frac[9,3] >= 1
 cov_8_3: frac[9,3] >= 1
 cov_9_0: frac[8,0] >= 0.7011046421352773
 cov_10_0: frac[8,0] >= 0.7011046421352773
 accuracy_6_4: - 5.09765625 frac[6,4] + mem[6,4] = 0
 accuracy_17_6: - 5.09765625 frac[17,6] + mem[17,6] = 0
 accuracy_5_5: - 5.09765625 frac[5,5] + mem[5,5] = 0
 accuracy_9_3: - 5.09765625 frac[9,3] + mem[9,3] = 0
 accuracy_8_0: - 5.09765625 frac[8,0] + mem[8,0] = 0
 accuracy_4_1: - 5.09765625 frac[4,1] + mem[4,1] = 0
 accuracy_0_2: - 5.09765625 frac[0,2] + mem[0,2] = 0
 mem_tot_CPU8997: - mem[0,2] + mem_tot_CPU8997 = 0
 R19: - 3.05176e-05 mem_tot_CPU8997 + normalized_mem_tot_CPU8997 = 0
 rows_tot_CPU8997: - 3 frac[0,2] + rows_tot_CPU8997 = 0
 R21: - 0.0833333333333333 rows_tot_CPU8997 + normalized_rows_tot_CPU8997
   = 0
 mem_tot_CPU2275: mem_tot_CPU2275 = 0
 R23: - 3.05176e-05 mem_tot_CPU2275 + normalized_mem_tot_CPU2275 = 0
 rows_tot_CPU2275: rows_tot_CPU2275 = 0
 R25: - 0.0833333333333333 rows_tot_CPU2275 + normalized_rows_tot_CPU2275
   = 0
 mem_tot_CPU4780: mem_tot_CPU4780 = 0
 R27: - 3.05176e-05 mem_tot_CPU4780 + normalized_mem_tot_CPU4780 = 0
 rows_tot_CPU4780: rows_tot_CPU4780 = 0
 R29: - 0.0833333333333333 rows_tot_CPU4780 + normalized_rows_tot_CPU4780
   = 0
 mem_tot_CPU45: mem_tot_CPU45 = 0
 R31: - 3.05176e-05 mem_tot_CPU45 + normalized_mem_tot_CPU45 = 0
 rows_tot_CPU45: rows_tot_CPU45 = 0
 R33: - 0.0833333333333333 rows_tot_CPU45 + normalized_rows_tot_CPU45 = 0
 mem_tot_CPU9485: - mem[4,1] + mem_tot_CPU9485 = 0
 R35: - 3.05176e-05 mem_tot_CPU9485 + normalized_mem_tot_CPU9485 = 0
 rows_tot_CPU9485: - 3 frac[4,1] + rows_tot_CPU9485 = 0
 R37: - 0.0833333333333333 rows_tot_CPU9485 + normalized_rows_tot_CPU9485
   = 0
 mem_tot_CPU8187: - mem[5,5] + mem_tot_CPU8187 = 0
 R39: - 3.05176e-05 mem_tot_CPU8187 + normalized_mem_tot_CPU8187 = 0
 rows_tot_CPU8187: - 3 frac[5,5] + rows_tot_CPU8187 = 0
 R41: - 0.0833333333333333 rows_tot_CPU8187 + normalized_rows_tot_CPU8187
   = 0
 mem_tot_CPU3809: - mem[6,4] + mem_tot_CPU3809 = 0
 R43: - 3.05176e-05 mem_tot_CPU3809 + normalized_mem_tot_CPU3809 = 0
 rows_tot_CPU3809: - 3 frac[6,4] + rows_tot_CPU3809 = 0
 R45: - 0.0833333333333333 rows_tot_CPU3809 + normalized_rows_tot_CPU3809
   = 0
 mem_tot_CPU8827: mem_tot_CPU8827 = 0
 R47: - 3.05176e-05 mem_tot_CPU8827 + normalized_mem_tot_CPU8827 = 0
 rows_tot_CPU8827: rows_tot_CPU8827 = 0
 R49: - 0.0833333333333333 rows_tot_CPU8827 + normalized_rows_tot_CPU8827
   = 0
 mem_tot_CPU5598: - mem[8,0] + mem_tot_CPU5598 = 0
 R51: - 3.05176e-05 mem_tot_CPU5598 + normalized_mem_tot_CPU5598 = 0
 rows_tot_CPU5598: - 3 frac[8,0] + rows_tot_CPU5598 = 0
 R53: - 0.0833333333333333 rows_tot_CPU5598 + normalized_rows_tot_CPU5598
   = 0
 mem_tot_CPU9261: - mem[9,3] + mem_tot_CPU9261 = 0
 R55: - 3.05176e-05 mem_tot_CPU9261 + normalized_mem_tot_CPU9261 = 0
 rows_tot_CPU9261: - 3 frac[9,3] + rows_tot_CPU9261 = 0
 R57: - 0.0833333333333333 rows_tot_CPU9261 + normalized_rows_tot_CPU9261
   = 0
 mem_tot_CPU501: mem_tot_CPU501 = 0
 R59: - 3.05176e-05 mem_tot_CPU501 + normalized_mem_tot_CPU501 = 0
 rows_tot_CPU501: rows_tot_CPU501 = 0
 R61: - 0.0833333333333333 rows_tot_CPU501 + normalized_rows_tot_CPU501
   = 0
 mem_tot_CPU7283: mem_tot_CPU7283 = 0
 R63: - 3.05176e-05 mem_tot_CPU7283 + normalized_mem_tot_CPU7283 = 0
 rows_tot_CPU7283: rows_tot_CPU7283 = 0
 R65: - 0.0833333333333333 rows_tot_CPU7283 + normalized_rows_tot_CPU7283
   = 0
 mem_tot_CPU4767: mem_tot_CPU4767 = 0
 R67: - 3.05176e-05 mem_tot_CPU4767 + normalized_mem_tot_CPU4767 = 0
 rows_tot_CPU4767: rows_tot_CPU4767 = 0
 R69: - 0.0833333333333333 rows_tot_CPU4767 + normalized_rows_tot_CPU4767
   = 0
 mem_tot_CPU9211: mem_tot_CPU9211 = 0
 R71: - 3.05176e-05 mem_tot_CPU9211 + normalized_mem_tot_CPU9211 = 0
 rows_tot_CPU9211: rows_tot_CPU9211 = 0
 R73: - 0.0833333333333333 rows_tot_CPU9211 + normalized_rows_tot_CPU9211
   = 0
 mem_tot_CPU4550: mem_tot_CPU4550 = 0
 R75: - 3.05176e-05 mem_tot_CPU4550 + normalized_mem_tot_CPU4550 = 0
 rows_tot_CPU4550: rows_tot_CPU4550 = 0
 R77: - 0.0833333333333333 rows_tot_CPU4550 + normalized_rows_tot_CPU4550
   = 0
 mem_tot_CPU3713: mem_tot_CPU3713 = 0
 R79: - 3.05176e-05 mem_tot_CPU3713 + normalized_mem_tot_CPU3713 = 0
 rows_tot_CPU3713: rows_tot_CPU3713 = 0
 R81: - 0.0833333333333333 rows_tot_CPU3713 + normalized_rows_tot_CPU3713
   = 0
 mem_tot_CPU6795: mem_tot_CPU6795 = 0
 R83: - 3.05176e-05 mem_tot_CPU6795 + normalized_mem_tot_CPU6795 = 0
 rows_tot_CPU6795: rows_tot_CPU6795 = 0
 R85: - 0.0833333333333333 rows_tot_CPU6795 + normalized_rows_tot_CPU6795
   = 0
 mem_tot_CPU9283: - mem[17,6] + mem_tot_CPU9283 = 0
 R87: - 3.05176e-05 mem_tot_CPU9283 + normalized_mem_tot_CPU9283 = 0
 rows_tot_CPU9283: - 3 frac[17,6] + rows_tot_CPU9283 = 0
 R89: - 0.0833333333333333 rows_tot_CPU9283 + normalized_rows_tot_CPU9283
   = 0
 mem_tot_CPU1799: mem_tot_CPU1799 = 0
 R91: - 3.05176e-05 mem_tot_CPU1799 + normalized_mem_tot_CPU1799 = 0
 rows_tot_CPU1799: rows_tot_CPU1799 = 0
 R93: - 0.0833333333333333 rows_tot_CPU1799 + normalized_rows_tot_CPU1799
   = 0
 mem_tot_CPU854: mem_tot_CPU854 = 0
 R95: - 3.05176e-05 mem_tot_CPU854 + normalized_mem_tot_CPU854 = 0
 rows_tot_CPU854: rows_tot_CPU854 = 0
 R97: - 0.0833333333333333 rows_tot_CPU854 + normalized_rows_tot_CPU854
   = 0
 row_capacity_CPU8997: rows_tot_CPU8997 <= 12
 row_capacity_CPU2275: rows_tot_CPU2275 <= 12
 row_capacity_CPU4780: rows_tot_CPU4780 <= 12
 row_capacity_CPU45: rows_tot_CPU45 <= 12
 row_capacity_CPU9485: rows_tot_CPU9485 <= 12
 row_capacity_CPU8187: rows_tot_CPU8187 <= 12
 row_capacity_CPU3809: rows_tot_CPU3809 <= 12
 row_capacity_CPU8827: rows_tot_CPU8827 <= 12
 row_capacity_CPU5598: rows_tot_CPU5598 <= 12
 row_capacity_CPU9261: rows_tot_CPU9261 <= 12
 row_capacity_CPU501: rows_tot_CPU501 <= 12
 row_capacity_CPU7283: rows_tot_CPU7283 <= 12
 row_capacity_CPU4767: rows_tot_CPU4767 <= 12
 row_capacity_CPU9211: rows_tot_CPU9211 <= 12
 row_capacity_CPU4550: rows_tot_CPU4550 <= 12
 row_capacity_CPU3713: rows_tot_CPU3713 <= 12
 row_capacity_CPU6795: rows_tot_CPU6795 <= 12
 row_capacity_CPU9283: rows_tot_CPU9283 <= 12
 row_capacity_CPU1799: rows_tot_CPU1799 <= 12
 row_capacity_CPU854: rows_tot_CPU854 <= 12
 ns_single_CPU8997: 3.5 rows_tot_CPU8997 - ns_single_CPU8997
   + pdt_m_rows_CPU8997 = 0
 capacity_cores_CPU8997: cores_sketch_CPU8997 + cores_dpdk_CPU8997 <= 7
 ns_sketch_CPU8997: - ns_single_CPU8997
   + 11.42857142857143 cores_sketch_CPU8997 >= 0
 ns_dpdk_CPU8997: - 4.285714285714287 cores_dpdk_CPU8997
   <= -21.42857142857143
 ns_single_CPU2275: 3.5 rows_tot_CPU2275 - ns_single_CPU2275
   + pdt_m_rows_CPU2275 = 0
 capacity_cores_CPU2275: cores_sketch_CPU2275 + cores_dpdk_CPU2275 <= 7
 ns_sketch_CPU2275: - ns_single_CPU2275
   + 11.42857142857143 cores_sketch_CPU2275 >= 0
 ns_dpdk_CPU2275: - 4.285714285714287 cores_dpdk_CPU2275
   <= -21.42857142857143
 ns_single_CPU4780: 3.5 rows_tot_CPU4780 - ns_single_CPU4780
   + pdt_m_rows_CPU4780 = 0
 capacity_cores_CPU4780: cores_sketch_CPU4780 + cores_dpdk_CPU4780 <= 7
 ns_sketch_CPU4780: - ns_single_CPU4780
   + 11.42857142857143 cores_sketch_CPU4780 >= 0
 ns_dpdk_CPU4780: - 4.285714285714287 cores_dpdk_CPU4780
   <= -21.42857142857143
 ns_single_CPU45: 3.5 rows_tot_CPU45 - ns_single_CPU45 + pdt_m_rows_CPU45
   = 0
 capacity_cores_CPU45: cores_sketch_CPU45 + cores_dpdk_CPU45 <= 7
 ns_sketch_CPU45: - ns_single_CPU45 + 11.42857142857143 cores_sketch_CPU45
   >= 0
 ns_dpdk_CPU45: - 4.285714285714287 cores_dpdk_CPU45 <= -21.42857142857143
 ns_single_CPU9485: 3.5 rows_tot_CPU9485 - ns_single_CPU9485
   + pdt_m_rows_CPU9485 = 0
 capacity_cores_CPU9485: cores_sketch_CPU9485 + cores_dpdk_CPU9485 <= 7
 ns_sketch_CPU9485: - ns_single_CPU9485
   + 11.42857142857143 cores_sketch_CPU9485 >= 0
 ns_dpdk_CPU9485: - 4.285714285714287 cores_dpdk_CPU9485
   <= -21.42857142857143
 ns_single_CPU8187: 3.5 rows_tot_CPU8187 - ns_single_CPU8187
   + pdt_m_rows_CPU8187 = 0
 capacity_cores_CPU8187: cores_sketch_CPU8187 + cores_dpdk_CPU8187 <= 7
 ns_sketch_CPU8187: - ns_single_CPU8187
   + 11.42857142857143 cores_sketch_CPU8187 >= 0
 ns_dpdk_CPU8187: - 4.285714285714287 cores_dpdk_CPU8187
   <= -21.42857142857143
 ns_single_CPU3809: 3.5 rows_tot_CPU3809 - ns_single_CPU3809
   + pdt_m_rows_CPU3809 = 0
 capacity_cores_CPU3809: cores_sketch_CPU3809 + cores_dpdk_CPU3809 <= 7
 ns_sketch_CPU3809: - ns_single_CPU3809
   + 11.42857142857143 cores_sketch_CPU3809 >= 0
 ns_dpdk_CPU3809: - 4.285714285714287 cores_dpdk_CPU3809
   <= -21.42857142857143
 ns_single_CPU8827: 3.5 rows_tot_CPU8827 - ns_single_CPU8827
   + pdt_m_rows_CPU8827 = 0
 capacity_cores_CPU8827: cores_sketch_CPU8827 + cores_dpdk_CPU8827 <= 7
 ns_sketch_CPU8827: - ns_single_CPU8827
   + 11.42857142857143 cores_sketch_CPU8827 >= 0
 ns_dpdk_CPU8827: - 4.285714285714287 cores_dpdk_CPU8827
   <= -21.42857142857143
 ns_single_CPU5598: 3.5 rows_tot_CPU5598 - ns_single_CPU5598
   + pdt_m_rows_CPU5598 = 0
 capacity_cores_CPU5598: cores_sketch_CPU5598 + cores_dpdk_CPU5598 <= 7
 ns_sketch_CPU5598: - ns_single_CPU5598
   + 11.42857142857143 cores_sketch_CPU5598 >= 0
 ns_dpdk_CPU5598: - 4.285714285714287 cores_dpdk_CPU5598
   <= -21.42857142857143
 ns_single_CPU9261: 3.5 rows_tot_CPU9261 - ns_single_CPU9261
   + pdt_m_rows_CPU9261 = 0
 capacity_cores_CPU9261: cores_sketch_CPU9261 + cores_dpdk_CPU9261 <= 7
 ns_sketch_CPU9261: - ns_single_CPU9261
   + 11.42857142857143 cores_sketch_CPU9261 >= 0
 ns_dpdk_CPU9261: - 4.285714285714287 cores_dpdk_CPU9261
   <= -21.42857142857143
 ns_single_CPU501: 3.5 rows_tot_CPU501 - ns_single_CPU501
   + pdt_m_rows_CPU501 = 0
 capacity_cores_CPU501: cores_sketch_CPU501 + cores_dpdk_CPU501 <= 7
 ns_sketch_CPU501: - ns_single_CPU501
   + 11.42857142857143 cores_sketch_CPU501 >= 0
 ns_dpdk_CPU501: - 4.285714285714287 cores_dpdk_CPU501
   <= -21.42857142857143
 ns_single_CPU7283: 3.5 rows_tot_CPU7283 - ns_single_CPU7283
   + pdt_m_rows_CPU7283 = 0
 capacity_cores_CPU7283: cores_sketch_CPU7283 + cores_dpdk_CPU7283 <= 7
 ns_sketch_CPU7283: - ns_single_CPU7283
   + 11.42857142857143 cores_sketch_CPU7283 >= 0
 ns_dpdk_CPU7283: - 4.285714285714287 cores_dpdk_CPU7283
   <= -21.42857142857143
 ns_single_CPU4767: 3.5 rows_tot_CPU4767 - ns_single_CPU4767
   + pdt_m_rows_CPU4767 = 0
 capacity_cores_CPU4767: cores_sketch_CPU4767 + cores_dpdk_CPU4767 <= 7
 ns_sketch_CPU4767: - ns_single_CPU4767
   + 11.42857142857143 cores_sketch_CPU4767 >= 0
 ns_dpdk_CPU4767: - 4.285714285714287 cores_dpdk_CPU4767
   <= -21.42857142857143
 ns_single_CPU9211: 3.5 rows_tot_CPU9211 - ns_single_CPU9211
   + pdt_m_rows_CPU9211 = 0
 capacity_cores_CPU9211: cores_sketch_CPU9211 + cores_dpdk_CPU9211 <= 7
 ns_sketch_CPU9211: - ns_single_CPU9211
   + 11.42857142857143 cores_sketch_CPU9211 >= 0
 ns_dpdk_CPU9211: - 4.285714285714287 cores_dpdk_CPU9211
   <= -21.42857142857143
 ns_single_CPU4550: 3.5 rows_tot_CPU4550 - ns_single_CPU4550
   + pdt_m_rows_CPU4550 = 0
 capacity_cores_CPU4550: cores_sketch_CPU4550 + cores_dpdk_CPU4550 <= 7
 ns_sketch_CPU4550: - ns_single_CPU4550
   + 11.42857142857143 cores_sketch_CPU4550 >= 0
 ns_dpdk_CPU4550: - 4.285714285714287 cores_dpdk_CPU4550
   <= -21.42857142857143
 ns_single_CPU3713: 3.5 rows_tot_CPU3713 - ns_single_CPU3713
   + pdt_m_rows_CPU3713 = 0
 capacity_cores_CPU3713: cores_sketch_CPU3713 + cores_dpdk_CPU3713 <= 7
 ns_sketch_CPU3713: - ns_single_CPU3713
   + 11.42857142857143 cores_sketch_CPU3713 >= 0
 ns_dpdk_CPU3713: - 4.285714285714287 cores_dpdk_CPU3713
   <= -21.42857142857143
 ns_single_CPU6795: 3.5 rows_tot_CPU6795 - ns_single_CPU6795
   + pdt_m_rows_CPU6795 = 0
 capacity_cores_CPU6795: cores_sketch_CPU6795 + cores_dpdk_CPU6795 <= 7
 ns_sketch_CPU6795: - ns_single_CPU6795
   + 11.42857142857143 cores_sketch_CPU6795 >= 0
 ns_dpdk_CPU6795: - 4.285714285714287 cores_dpdk_CPU6795
   <= -21.42857142857143
 ns_single_CPU9283: 3.5 rows_tot_CPU9283 - ns_single_CPU9283
   + pdt_m_rows_CPU9283 = 0
 capacity_cores_CPU9283: cores_sketch_CPU9283 + cores_dpdk_CPU9283 <= 7
 ns_sketch_CPU9283: - ns_single_CPU9283
   + 11.42857142857143 cores_sketch_CPU9283 >= 0
 ns_dpdk_CPU9283: - 4.285714285714287 cores_dpdk_CPU9283
   <= -21.42857142857143
 ns_single_CPU1799: 3.5 rows_tot_CPU1799 - ns_single_CPU1799
   + pdt_m_rows_CPU1799 = 0
 capacity_cores_CPU1799: cores_sketch_CPU1799 + cores_dpdk_CPU1799 <= 7
 ns_sketch_CPU1799: - ns_single_CPU1799
   + 11.42857142857143 cores_sketch_CPU1799 >= 0
 ns_dpdk_CPU1799: - 4.285714285714287 cores_dpdk_CPU1799
   <= -21.42857142857143
 ns_single_CPU854: 3.5 rows_tot_CPU854 - ns_single_CPU854
   + pdt_m_rows_CPU854 = 0
 capacity_cores_CPU854: cores_sketch_CPU854 + cores_dpdk_CPU854 <= 7
 ns_sketch_CPU854: - ns_single_CPU854
   + 11.42857142857143 cores_sketch_CPU854 >= 0
 ns_dpdk_CPU854: - 4.285714285714287 cores_dpdk_CPU854
   <= -21.42857142857143
 res_acc: 1.220703125e-04 mem_tot_CPU8997 + 1.220703125e-04 mem_tot_CPU2275
   + 1.220703125e-04 mem_tot_CPU4780 + 1.220703125e-04 mem_tot_CPU45
   + 1.220703125e-04 mem_tot_CPU9485 + 1.220703125e-04 mem_tot_CPU8187
   + 1.220703125e-04 mem_tot_CPU3809 + 1.220703125e-04 mem_tot_CPU8827
   + 1.220703125e-04 mem_tot_CPU5598 + 1.220703125e-04 mem_tot_CPU9261
   + 1.220703125e-04 mem_tot_CPU501 + 1.220703125e-04 mem_tot_CPU7283
   + 1.220703125e-04 mem_tot_CPU4767 + 1.220703125e-04 mem_tot_CPU9211
   + 1.220703125e-04 mem_tot_CPU4550 + 1.220703125e-04 mem_tot_CPU3713
   + 1.220703125e-04 mem_tot_CPU6795 + 1.220703125e-04 mem_tot_CPU9283
   + 1.220703125e-04 mem_tot_CPU1799 + 1.220703125e-04 mem_tot_CPU854
   + 10 cores_sketch_CPU8997 + 10 cores_dpdk_CPU8997
   + 10 cores_sketch_CPU2275 + 10 cores_dpdk_CPU2275
   + 10 cores_sketch_CPU4780 + 10 cores_dpdk_CPU4780
   + 10 cores_sketch_CPU45 + 10 cores_dpdk_CPU45 + 10 cores_sketch_CPU9485
   + 10 cores_dpdk_CPU9485 + 10 cores_sketch_CPU8187
   + 10 cores_dpdk_CPU8187 + 10 cores_sketch_CPU3809
   + 10 cores_dpdk_CPU3809 + 10 cores_sketch_CPU8827
   + 10 cores_dpdk_CPU8827 + 10 cores_sketch_CPU5598
   + 10 cores_dpdk_CPU5598 + 10 cores_sketch_CPU9261
   + 10 cores_dpdk_CPU9261 + 10 cores_sketch_CPU501 + 10 cores_dpdk_CPU501
   + 10 cores_sketch_CPU7283 + 10 cores_dpdk_CPU7283
   + 10 cores_sketch_CPU4767 + 10 cores_dpdk_CPU4767
   + 10 cores_sketch_CPU9211 + 10 cores_dpdk_CPU9211
   + 10 cores_sketch_CPU4550 + 10 cores_dpdk_CPU4550
   + 10 cores_sketch_CPU3713 + 10 cores_dpdk_CPU3713
   + 10 cores_sketch_CPU6795 + 10 cores_dpdk_CPU6795
   + 10 cores_sketch_CPU9283 + 10 cores_dpdk_CPU9283
   + 10 cores_sketch_CPU1799 + 10 cores_dpdk_CPU1799
   + 10 cores_sketch_CPU854 + 10 cores_dpdk_CPU854 - res = 0
 pdt_m_rows_CPU8997: pdt_m_rows_CPU8997 + [
   - rows_tot_CPU8997 * m_access_time_CPU8997 ] = 0
 pdt_m_rows_CPU2275: pdt_m_rows_CPU2275 + [
   - rows_tot_CPU2275 * m_access_time_CPU2275 ] = 0
 pdt_m_rows_CPU4780: pdt_m_rows_CPU4780 + [
   - rows_tot_CPU4780 * m_access_time_CPU4780 ] = 0
 pdt_m_rows_CPU45: pdt_m_rows_CPU45 + [
   - rows_tot_CPU45 * m_access_time_CPU45 ] = 0
 pdt_m_rows_CPU9485: pdt_m_rows_CPU9485 + [
   - rows_tot_CPU9485 * m_access_time_CPU9485 ] = 0
 pdt_m_rows_CPU8187: pdt_m_rows_CPU8187 + [
   - rows_tot_CPU8187 * m_access_time_CPU8187 ] = 0
 pdt_m_rows_CPU3809: pdt_m_rows_CPU3809 + [
   - rows_tot_CPU3809 * m_access_time_CPU3809 ] = 0
 pdt_m_rows_CPU8827: pdt_m_rows_CPU8827 + [
   - rows_tot_CPU8827 * m_access_time_CPU8827 ] = 0
 pdt_m_rows_CPU5598: pdt_m_rows_CPU5598 + [
   - rows_tot_CPU5598 * m_access_time_CPU5598 ] = 0
 pdt_m_rows_CPU9261: pdt_m_rows_CPU9261 + [
   - rows_tot_CPU9261 * m_access_time_CPU9261 ] = 0
 pdt_m_rows_CPU501: pdt_m_rows_CPU501 + [
   - rows_tot_CPU501 * m_access_time_CPU501 ] = 0
 pdt_m_rows_CPU7283: pdt_m_rows_CPU7283 + [
   - rows_tot_CPU7283 * m_access_time_CPU7283 ] = 0
 pdt_m_rows_CPU4767: pdt_m_rows_CPU4767 + [
   - rows_tot_CPU4767 * m_access_time_CPU4767 ] = 0
 pdt_m_rows_CPU9211: pdt_m_rows_CPU9211 + [
   - rows_tot_CPU9211 * m_access_time_CPU9211 ] = 0
 pdt_m_rows_CPU4550: pdt_m_rows_CPU4550 + [
   - rows_tot_CPU4550 * m_access_time_CPU4550 ] = 0
 pdt_m_rows_CPU3713: pdt_m_rows_CPU3713 + [
   - rows_tot_CPU3713 * m_access_time_CPU3713 ] = 0
 pdt_m_rows_CPU6795: pdt_m_rows_CPU6795 + [
   - rows_tot_CPU6795 * m_access_time_CPU6795 ] = 0
 pdt_m_rows_CPU9283: pdt_m_rows_CPU9283 + [
   - rows_tot_CPU9283 * m_access_time_CPU9283 ] = 0
 pdt_m_rows_CPU1799: pdt_m_rows_CPU1799 + [
   - rows_tot_CPU1799 * m_access_time_CPU1799 ] = 0
 pdt_m_rows_CPU854: pdt_m_rows_CPU854 + [
   - rows_tot_CPU854 * m_access_time_CPU854 ] = 0
Bounds
 frac[6,4] <= 1
 frac[17,6] <= 1
 frac[5,5] <= 1
 frac[9,3] <= 1
 frac[8,0] <= 1
 frac[4,1] <= 1
 frac[0,2] <= 1
 mem_tot_CPU8997 <= 32768
 normalized_mem_tot_CPU8997 <= 1
 normalized_rows_tot_CPU8997 <= 1
 mem_tot_CPU2275 <= 32768
 normalized_mem_tot_CPU2275 <= 1
 normalized_rows_tot_CPU2275 <= 1
 mem_tot_CPU4780 <= 32768
 normalized_mem_tot_CPU4780 <= 1
 normalized_rows_tot_CPU4780 <= 1
 mem_tot_CPU45 <= 32768
 normalized_mem_tot_CPU45 <= 1
 normalized_rows_tot_CPU45 <= 1
 mem_tot_CPU9485 <= 32768
 normalized_mem_tot_CPU9485 <= 1
 normalized_rows_tot_CPU9485 <= 1
 mem_tot_CPU8187 <= 32768
 normalized_mem_tot_CPU8187 <= 1
 normalized_rows_tot_CPU8187 <= 1
 mem_tot_CPU3809 <= 32768
 normalized_mem_tot_CPU3809 <= 1
 normalized_rows_tot_CPU3809 <= 1
 mem_tot_CPU8827 <= 32768
 normalized_mem_tot_CPU8827 <= 1
 normalized_rows_tot_CPU8827 <= 1
 mem_tot_CPU5598 <= 32768
 normalized_mem_tot_CPU5598 <= 1
 normalized_rows_tot_CPU5598 <= 1
 mem_tot_CPU9261 <= 32768
 normalized_mem_tot_CPU9261 <= 1
 normalized_rows_tot_CPU9261 <= 1
 mem_tot_CPU501 <= 32768
 normalized_mem_tot_CPU501 <= 1
 normalized_rows_tot_CPU501 <= 1
 mem_tot_CPU7283 <= 32768
 normalized_mem_tot_CPU7283 <= 1
 normalized_rows_tot_CPU7283 <= 1
 mem_tot_CPU4767 <= 32768
 normalized_mem_tot_CPU4767 <= 1
 normalized_rows_tot_CPU4767 <= 1
 mem_tot_CPU9211 <= 32768
 normalized_mem_tot_CPU9211 <= 1
 normalized_rows_tot_CPU9211 <= 1
 mem_tot_CPU4550 <= 32768
 normalized_mem_tot_CPU4550 <= 1
 normalized_rows_tot_CPU4550 <= 1
 mem_tot_CPU3713 <= 32768
 normalized_mem_tot_CPU3713 <= 1
 normalized_rows_tot_CPU3713 <= 1
 mem_tot_CPU6795 <= 32768
 normalized_mem_tot_CPU6795 <= 1
 normalized_rows_tot_CPU6795 <= 1
 mem_tot_CPU9283 <= 32768
 normalized_mem_tot_CPU9283 <= 1
 normalized_rows_tot_CPU9283 <= 1
 mem_tot_CPU1799 <= 32768
 normalized_mem_tot_CPU1799 <= 1
 normalized_rows_tot_CPU1799 <= 1
 mem_tot_CPU854 <= 32768
 normalized_mem_tot_CPU854 <= 1
 normalized_rows_tot_CPU854 <= 1
 cores_sketch_CPU8997 <= 7
 1 <= cores_dpdk_CPU8997 <= 7
 cores_sketch_CPU2275 <= 7
 1 <= cores_dpdk_CPU2275 <= 7
 cores_sketch_CPU4780 <= 7
 1 <= cores_dpdk_CPU4780 <= 7
 cores_sketch_CPU45 <= 7
 1 <= cores_dpdk_CPU45 <= 7
 cores_sketch_CPU9485 <= 7
 1 <= cores_dpdk_CPU9485 <= 7
 cores_sketch_CPU8187 <= 7
 1 <= cores_dpdk_CPU8187 <= 7
 cores_sketch_CPU3809 <= 7
 1 <= cores_dpdk_CPU3809 <= 7
 cores_sketch_CPU8827 <= 7
 1 <= cores_dpdk_CPU8827 <= 7
 cores_sketch_CPU5598 <= 7
 1 <= cores_dpdk_CPU5598 <= 7
 cores_sketch_CPU9261 <= 7
 1 <= cores_dpdk_CPU9261 <= 7
 cores_sketch_CPU501 <= 7
 1 <= cores_dpdk_CPU501 <= 7
 cores_sketch_CPU7283 <= 7
 1 <= cores_dpdk_CPU7283 <= 7
 cores_sketch_CPU4767 <= 7
 1 <= cores_dpdk_CPU4767 <= 7
 cores_sketch_CPU9211 <= 7
 1 <= cores_dpdk_CPU9211 <= 7
 cores_sketch_CPU4550 <= 7
 1 <= cores_dpdk_CPU4550 <= 7
 cores_sketch_CPU3713 <= 7
 1 <= cores_dpdk_CPU3713 <= 7
 cores_sketch_CPU6795 <= 7
 1 <= cores_dpdk_CPU6795 <= 7
 cores_sketch_CPU9283 <= 7
 1 <= cores_dpdk_CPU9283 <= 7
 cores_sketch_CPU1799 <= 7
 1 <= cores_dpdk_CPU1799 <= 7
 cores_sketch_CPU854 <= 7
 1 <= cores_dpdk_CPU854 <= 7
Generals
 cores_sketch_CPU8997 cores_dpdk_CPU8997 cores_sketch_CPU2275
 cores_dpdk_CPU2275 cores_sketch_CPU4780 cores_dpdk_CPU4780
 cores_sketch_CPU45 cores_dpdk_CPU45 cores_sketch_CPU9485
 cores_dpdk_CPU9485 cores_sketch_CPU8187 cores_dpdk_CPU8187
 cores_sketch_CPU3809 cores_dpdk_CPU3809 cores_sketch_CPU8827
 cores_dpdk_CPU8827 cores_sketch_CPU5598 cores_dpdk_CPU5598
 cores_sketch_CPU9261 cores_dpdk_CPU9261 cores_sketch_CPU501
 cores_dpdk_CPU501 cores_sketch_CPU7283 cores_dpdk_CPU7283
 cores_sketch_CPU4767 cores_dpdk_CPU4767 cores_sketch_CPU9211
 cores_dpdk_CPU9211 cores_sketch_CPU4550 cores_dpdk_CPU4550
 cores_sketch_CPU3713 cores_dpdk_CPU3713 cores_sketch_CPU6795
 cores_dpdk_CPU6795 cores_sketch_CPU9283 cores_dpdk_CPU9283
 cores_sketch_CPU1799 cores_dpdk_CPU1799 cores_sketch_CPU854
 cores_dpdk_CPU854
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU8997 , normalized_mem_tot_CPU2275 ,
   normalized_mem_tot_CPU4780 , normalized_mem_tot_CPU45 ,
   normalized_mem_tot_CPU9485 , normalized_mem_tot_CPU8187 ,
   normalized_mem_tot_CPU3809 , normalized_mem_tot_CPU8827 ,
   normalized_mem_tot_CPU5598 , normalized_mem_tot_CPU9261 ,
   normalized_mem_tot_CPU501 , normalized_mem_tot_CPU7283 ,
   normalized_mem_tot_CPU4767 , normalized_mem_tot_CPU9211 ,
   normalized_mem_tot_CPU4550 , normalized_mem_tot_CPU3713 ,
   normalized_mem_tot_CPU6795 , normalized_mem_tot_CPU9283 ,
   normalized_mem_tot_CPU1799 , normalized_mem_tot_CPU854 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX (
   normalized_rows_tot_CPU8997 , normalized_rows_tot_CPU2275 ,
   normalized_rows_tot_CPU4780 , normalized_rows_tot_CPU45 ,
   normalized_rows_tot_CPU9485 , normalized_rows_tot_CPU8187 ,
   normalized_rows_tot_CPU3809 , normalized_rows_tot_CPU8827 ,
   normalized_rows_tot_CPU5598 , normalized_rows_tot_CPU9261 ,
   normalized_rows_tot_CPU501 , normalized_rows_tot_CPU7283 ,
   normalized_rows_tot_CPU4767 , normalized_rows_tot_CPU9211 ,
   normalized_rows_tot_CPU4550 , normalized_rows_tot_CPU3713 ,
   normalized_rows_tot_CPU6795 , normalized_rows_tot_CPU9283 ,
   normalized_rows_tot_CPU1799 , normalized_rows_tot_CPU854 )
 mem_access_time_CPU8997:
   m_access_time_CPU8997 = PWL ( mem_tot_CPU8997 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU2275:
   m_access_time_CPU2275 = PWL ( mem_tot_CPU2275 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU4780:
   m_access_time_CPU4780 = PWL ( mem_tot_CPU4780 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU45: m_access_time_CPU45 = PWL ( mem_tot_CPU45 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU9485:
   m_access_time_CPU9485 = PWL ( mem_tot_CPU9485 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU8187:
   m_access_time_CPU8187 = PWL ( mem_tot_CPU8187 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU3809:
   m_access_time_CPU3809 = PWL ( mem_tot_CPU3809 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU8827:
   m_access_time_CPU8827 = PWL ( mem_tot_CPU8827 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU5598:
   m_access_time_CPU5598 = PWL ( mem_tot_CPU5598 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU9261:
   m_access_time_CPU9261 = PWL ( mem_tot_CPU9261 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU501: m_access_time_CPU501 = PWL ( mem_tot_CPU501 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU7283:
   m_access_time_CPU7283 = PWL ( mem_tot_CPU7283 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU4767:
   m_access_time_CPU4767 = PWL ( mem_tot_CPU4767 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU9211:
   m_access_time_CPU9211 = PWL ( mem_tot_CPU9211 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU4550:
   m_access_time_CPU4550 = PWL ( mem_tot_CPU4550 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU3713:
   m_access_time_CPU3713 = PWL ( mem_tot_CPU3713 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU6795:
   m_access_time_CPU6795 = PWL ( mem_tot_CPU6795 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU9283:
   m_access_time_CPU9283 = PWL ( mem_tot_CPU9283 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU1799:
   m_access_time_CPU1799 = PWL ( mem_tot_CPU1799 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 mem_access_time_CPU854: m_access_time_CPU854 = PWL ( mem_tot_CPU854 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
End

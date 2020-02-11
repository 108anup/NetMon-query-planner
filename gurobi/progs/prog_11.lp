\ Model netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0.05
   0 m_access_time_cpu_0 + 0 ns_sketch_cpu_0 + 0 ns_dpdk_cpu_0 + 0 ns_cpu_0
   + 0 m_access_time_cpu_1 + 0 ns_sketch_cpu_1 + 0 ns_dpdk_cpu_1
   + 0 ns_cpu_1 + 0 m_access_time_cpu_2 + 0 ns_sketch_cpu_2
   + 0 ns_dpdk_cpu_2 + 0 ns_cpu_2 + 0 m_access_time_cpu_3
   + 0 ns_sketch_cpu_3 + 0 ns_dpdk_cpu_3 + 0 ns_cpu_3
   + 0 m_access_time_cpu_4 + 0 ns_sketch_cpu_4 + 0 ns_dpdk_cpu_4
   + 0 ns_cpu_4 + 0 m_access_time_cpu_5 + 0 ns_sketch_cpu_5
   + 0 ns_dpdk_cpu_5 + 0 ns_cpu_5 + 0 m_access_time_cpu_6
   + 0 ns_sketch_cpu_6 + 0 ns_dpdk_cpu_6 + 0 ns_cpu_6
   + 0 m_access_time_cpu_7 + 0 ns_sketch_cpu_7 + 0 ns_dpdk_cpu_7
   + 0 ns_cpu_7 + 0 m_access_time_cpu_8 + 0 ns_sketch_cpu_8
   + 0 ns_dpdk_cpu_8 + 0 ns_cpu_8 + 0 m_access_time_cpu_9
   + 0 ns_sketch_cpu_9 + 0 ns_dpdk_cpu_9 + 0 ns_cpu_9
   + 0 m_access_time_cpu_10 + 0 ns_sketch_cpu_10 + 0 ns_dpdk_cpu_10
   + 0 ns_cpu_10 + 0 m_access_time_cpu_11 + 0 ns_sketch_cpu_11
   + 0 ns_dpdk_cpu_11 + 0 ns_cpu_11 + 0 m_access_time_cpu_12
   + 0 ns_sketch_cpu_12 + 0 ns_dpdk_cpu_12 + 0 ns_cpu_12
   + 0 m_access_time_cpu_13 + 0 ns_sketch_cpu_13 + 0 ns_dpdk_cpu_13
   + 0 ns_cpu_13 + 0 m_access_time_cpu_14 + 0 ns_sketch_cpu_14
   + 0 ns_dpdk_cpu_14 + 0 ns_cpu_14 + 0 m_access_time_cpu_15
   + 0 ns_sketch_cpu_15 + 0 ns_dpdk_cpu_15 + 0 ns_cpu_15
   + 0 m_access_time_cpu_16 + 0 ns_sketch_cpu_16 + 0 ns_dpdk_cpu_16
   + 0 ns_cpu_16 + 0 m_access_time_cpu_17 + 0 ns_sketch_cpu_17
   + 0 ns_dpdk_cpu_17 + 0 ns_cpu_17 + 0 m_access_time_cpu_18
   + 0 ns_sketch_cpu_18 + 0 ns_dpdk_cpu_18 + 0 ns_cpu_18
   + 0 m_access_time_cpu_19 + 0 ns_sketch_cpu_19 + 0 ns_dpdk_cpu_19
   + 0 ns_cpu_19 + ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0.05
   res
Subject To
 cov_0_0: frac[0,0] + frac[1,0] + frac[20,0] >= 1
 cov_1_1: frac[1,1] + frac[2,1] + frac[20,1] >= 1
 cov_2_2: frac[2,2] + frac[3,2] + frac[20,2] >= 1
 cov_3_3: frac[3,3] + frac[4,3] + frac[20,3] >= 1
 cov_4_4: frac[4,4] + frac[5,4] + frac[20,4] >= 1
 cov_5_5: frac[5,5] + frac[6,5] + frac[20,5] >= 1
 cov_6_6: frac[6,6] + frac[7,6] + frac[20,6] >= 1
 cov_7_7: frac[7,7] + frac[8,7] + frac[20,7] >= 1
 cov_8_8: frac[8,8] + frac[9,8] + frac[20,8] >= 1
 cov_9_9: frac[9,9] + frac[10,9] + frac[20,9] >= 1
 cov_10_10: frac[10,10] + frac[11,10] + frac[20,10] >= 1
 cov_11_11: frac[11,11] + frac[12,11] + frac[20,11] >= 1
 cov_12_12: frac[12,12] + frac[13,12] + frac[20,12] >= 1
 cov_13_13: frac[13,13] + frac[14,13] + frac[20,13] >= 1
 cov_14_14: frac[14,14] + frac[15,14] + frac[20,14] >= 1
 cov_15_15: frac[15,15] + frac[16,15] + frac[20,15] >= 1
 cov_16_16: frac[16,16] + frac[17,16] + frac[20,16] >= 1
 cov_17_17: frac[17,17] + frac[18,17] + frac[20,17] >= 1
 cov_18_18: frac[18,18] + frac[19,18] + frac[20,18] >= 1
 cov_19_19: frac[0,19] + frac[19,19] + frac[20,19] >= 1
 accuracy_0_sk_0[0]: - 5.09765625 frac[0,0] + mem[0,0] = 0
 accuracy_0_sk_0[1]: - 5.09765625 frac[1,0] + mem[1,0] = 0
 accuracy_0_sk_0[2]: - 5.09765625 frac[2,0] + mem[2,0] = 0
 accuracy_0_sk_0[3]: - 5.09765625 frac[3,0] + mem[3,0] = 0
 accuracy_0_sk_0[4]: - 5.09765625 frac[4,0] + mem[4,0] = 0
 accuracy_0_sk_0[5]: - 5.09765625 frac[5,0] + mem[5,0] = 0
 accuracy_0_sk_0[6]: - 5.09765625 frac[6,0] + mem[6,0] = 0
 accuracy_0_sk_0[7]: - 5.09765625 frac[7,0] + mem[7,0] = 0
 accuracy_0_sk_0[8]: - 5.09765625 frac[8,0] + mem[8,0] = 0
 accuracy_0_sk_0[9]: - 5.09765625 frac[9,0] + mem[9,0] = 0
 accuracy_0_sk_0[10]: - 5.09765625 frac[10,0] + mem[10,0] = 0
 accuracy_0_sk_0[11]: - 5.09765625 frac[11,0] + mem[11,0] = 0
 accuracy_0_sk_0[12]: - 5.09765625 frac[12,0] + mem[12,0] = 0
 accuracy_0_sk_0[13]: - 5.09765625 frac[13,0] + mem[13,0] = 0
 accuracy_0_sk_0[14]: - 5.09765625 frac[14,0] + mem[14,0] = 0
 accuracy_0_sk_0[15]: - 5.09765625 frac[15,0] + mem[15,0] = 0
 accuracy_0_sk_0[16]: - 5.09765625 frac[16,0] + mem[16,0] = 0
 accuracy_0_sk_0[17]: - 5.09765625 frac[17,0] + mem[17,0] = 0
 accuracy_0_sk_0[18]: - 5.09765625 frac[18,0] + mem[18,0] = 0
 accuracy_0_sk_0[19]: - 5.09765625 frac[19,0] + mem[19,0] = 0
 accuracy_0_sk_0[20]: - 5.09765625 frac[20,0] + mem[20,0] = 0
 accuracy_1_sk_1[0]: - 5.09765625 frac[0,1] + mem[0,1] = 0
 accuracy_1_sk_1[1]: - 5.09765625 frac[1,1] + mem[1,1] = 0
 accuracy_1_sk_1[2]: - 5.09765625 frac[2,1] + mem[2,1] = 0
 accuracy_1_sk_1[3]: - 5.09765625 frac[3,1] + mem[3,1] = 0
 accuracy_1_sk_1[4]: - 5.09765625 frac[4,1] + mem[4,1] = 0
 accuracy_1_sk_1[5]: - 5.09765625 frac[5,1] + mem[5,1] = 0
 accuracy_1_sk_1[6]: - 5.09765625 frac[6,1] + mem[6,1] = 0
 accuracy_1_sk_1[7]: - 5.09765625 frac[7,1] + mem[7,1] = 0
 accuracy_1_sk_1[8]: - 5.09765625 frac[8,1] + mem[8,1] = 0
 accuracy_1_sk_1[9]: - 5.09765625 frac[9,1] + mem[9,1] = 0
 accuracy_1_sk_1[10]: - 5.09765625 frac[10,1] + mem[10,1] = 0
 accuracy_1_sk_1[11]: - 5.09765625 frac[11,1] + mem[11,1] = 0
 accuracy_1_sk_1[12]: - 5.09765625 frac[12,1] + mem[12,1] = 0
 accuracy_1_sk_1[13]: - 5.09765625 frac[13,1] + mem[13,1] = 0
 accuracy_1_sk_1[14]: - 5.09765625 frac[14,1] + mem[14,1] = 0
 accuracy_1_sk_1[15]: - 5.09765625 frac[15,1] + mem[15,1] = 0
 accuracy_1_sk_1[16]: - 5.09765625 frac[16,1] + mem[16,1] = 0
 accuracy_1_sk_1[17]: - 5.09765625 frac[17,1] + mem[17,1] = 0
 accuracy_1_sk_1[18]: - 5.09765625 frac[18,1] + mem[18,1] = 0
 accuracy_1_sk_1[19]: - 5.09765625 frac[19,1] + mem[19,1] = 0
 accuracy_1_sk_1[20]: - 5.09765625 frac[20,1] + mem[20,1] = 0
 accuracy_2_sk_2[0]: - 5.09765625 frac[0,2] + mem[0,2] = 0
 accuracy_2_sk_2[1]: - 5.09765625 frac[1,2] + mem[1,2] = 0
 accuracy_2_sk_2[2]: - 5.09765625 frac[2,2] + mem[2,2] = 0
 accuracy_2_sk_2[3]: - 5.09765625 frac[3,2] + mem[3,2] = 0
 accuracy_2_sk_2[4]: - 5.09765625 frac[4,2] + mem[4,2] = 0
 accuracy_2_sk_2[5]: - 5.09765625 frac[5,2] + mem[5,2] = 0
 accuracy_2_sk_2[6]: - 5.09765625 frac[6,2] + mem[6,2] = 0
 accuracy_2_sk_2[7]: - 5.09765625 frac[7,2] + mem[7,2] = 0
 accuracy_2_sk_2[8]: - 5.09765625 frac[8,2] + mem[8,2] = 0
 accuracy_2_sk_2[9]: - 5.09765625 frac[9,2] + mem[9,2] = 0
 accuracy_2_sk_2[10]: - 5.09765625 frac[10,2] + mem[10,2] = 0
 accuracy_2_sk_2[11]: - 5.09765625 frac[11,2] + mem[11,2] = 0
 accuracy_2_sk_2[12]: - 5.09765625 frac[12,2] + mem[12,2] = 0
 accuracy_2_sk_2[13]: - 5.09765625 frac[13,2] + mem[13,2] = 0
 accuracy_2_sk_2[14]: - 5.09765625 frac[14,2] + mem[14,2] = 0
 accuracy_2_sk_2[15]: - 5.09765625 frac[15,2] + mem[15,2] = 0
 accuracy_2_sk_2[16]: - 5.09765625 frac[16,2] + mem[16,2] = 0
 accuracy_2_sk_2[17]: - 5.09765625 frac[17,2] + mem[17,2] = 0
 accuracy_2_sk_2[18]: - 5.09765625 frac[18,2] + mem[18,2] = 0
 accuracy_2_sk_2[19]: - 5.09765625 frac[19,2] + mem[19,2] = 0
 accuracy_2_sk_2[20]: - 5.09765625 frac[20,2] + mem[20,2] = 0
 accuracy_3_sk_3[0]: - 5.09765625 frac[0,3] + mem[0,3] = 0
 accuracy_3_sk_3[1]: - 5.09765625 frac[1,3] + mem[1,3] = 0
 accuracy_3_sk_3[2]: - 5.09765625 frac[2,3] + mem[2,3] = 0
 accuracy_3_sk_3[3]: - 5.09765625 frac[3,3] + mem[3,3] = 0
 accuracy_3_sk_3[4]: - 5.09765625 frac[4,3] + mem[4,3] = 0
 accuracy_3_sk_3[5]: - 5.09765625 frac[5,3] + mem[5,3] = 0
 accuracy_3_sk_3[6]: - 5.09765625 frac[6,3] + mem[6,3] = 0
 accuracy_3_sk_3[7]: - 5.09765625 frac[7,3] + mem[7,3] = 0
 accuracy_3_sk_3[8]: - 5.09765625 frac[8,3] + mem[8,3] = 0
 accuracy_3_sk_3[9]: - 5.09765625 frac[9,3] + mem[9,3] = 0
 accuracy_3_sk_3[10]: - 5.09765625 frac[10,3] + mem[10,3] = 0
 accuracy_3_sk_3[11]: - 5.09765625 frac[11,3] + mem[11,3] = 0
 accuracy_3_sk_3[12]: - 5.09765625 frac[12,3] + mem[12,3] = 0
 accuracy_3_sk_3[13]: - 5.09765625 frac[13,3] + mem[13,3] = 0
 accuracy_3_sk_3[14]: - 5.09765625 frac[14,3] + mem[14,3] = 0
 accuracy_3_sk_3[15]: - 5.09765625 frac[15,3] + mem[15,3] = 0
 accuracy_3_sk_3[16]: - 5.09765625 frac[16,3] + mem[16,3] = 0
 accuracy_3_sk_3[17]: - 5.09765625 frac[17,3] + mem[17,3] = 0
 accuracy_3_sk_3[18]: - 5.09765625 frac[18,3] + mem[18,3] = 0
 accuracy_3_sk_3[19]: - 5.09765625 frac[19,3] + mem[19,3] = 0
 accuracy_3_sk_3[20]: - 5.09765625 frac[20,3] + mem[20,3] = 0
 accuracy_4_sk_4[0]: - 5.09765625 frac[0,4] + mem[0,4] = 0
 accuracy_4_sk_4[1]: - 5.09765625 frac[1,4] + mem[1,4] = 0
 accuracy_4_sk_4[2]: - 5.09765625 frac[2,4] + mem[2,4] = 0
 accuracy_4_sk_4[3]: - 5.09765625 frac[3,4] + mem[3,4] = 0
 accuracy_4_sk_4[4]: - 5.09765625 frac[4,4] + mem[4,4] = 0
 accuracy_4_sk_4[5]: - 5.09765625 frac[5,4] + mem[5,4] = 0
 accuracy_4_sk_4[6]: - 5.09765625 frac[6,4] + mem[6,4] = 0
 accuracy_4_sk_4[7]: - 5.09765625 frac[7,4] + mem[7,4] = 0
 accuracy_4_sk_4[8]: - 5.09765625 frac[8,4] + mem[8,4] = 0
 accuracy_4_sk_4[9]: - 5.09765625 frac[9,4] + mem[9,4] = 0
 accuracy_4_sk_4[10]: - 5.09765625 frac[10,4] + mem[10,4] = 0
 accuracy_4_sk_4[11]: - 5.09765625 frac[11,4] + mem[11,4] = 0
 accuracy_4_sk_4[12]: - 5.09765625 frac[12,4] + mem[12,4] = 0
 accuracy_4_sk_4[13]: - 5.09765625 frac[13,4] + mem[13,4] = 0
 accuracy_4_sk_4[14]: - 5.09765625 frac[14,4] + mem[14,4] = 0
 accuracy_4_sk_4[15]: - 5.09765625 frac[15,4] + mem[15,4] = 0
 accuracy_4_sk_4[16]: - 5.09765625 frac[16,4] + mem[16,4] = 0
 accuracy_4_sk_4[17]: - 5.09765625 frac[17,4] + mem[17,4] = 0
 accuracy_4_sk_4[18]: - 5.09765625 frac[18,4] + mem[18,4] = 0
 accuracy_4_sk_4[19]: - 5.09765625 frac[19,4] + mem[19,4] = 0
 accuracy_4_sk_4[20]: - 5.09765625 frac[20,4] + mem[20,4] = 0
 accuracy_5_sk_5[0]: - 5.09765625 frac[0,5] + mem[0,5] = 0
 accuracy_5_sk_5[1]: - 5.09765625 frac[1,5] + mem[1,5] = 0
 accuracy_5_sk_5[2]: - 5.09765625 frac[2,5] + mem[2,5] = 0
 accuracy_5_sk_5[3]: - 5.09765625 frac[3,5] + mem[3,5] = 0
 accuracy_5_sk_5[4]: - 5.09765625 frac[4,5] + mem[4,5] = 0
 accuracy_5_sk_5[5]: - 5.09765625 frac[5,5] + mem[5,5] = 0
 accuracy_5_sk_5[6]: - 5.09765625 frac[6,5] + mem[6,5] = 0
 accuracy_5_sk_5[7]: - 5.09765625 frac[7,5] + mem[7,5] = 0
 accuracy_5_sk_5[8]: - 5.09765625 frac[8,5] + mem[8,5] = 0
 accuracy_5_sk_5[9]: - 5.09765625 frac[9,5] + mem[9,5] = 0
 accuracy_5_sk_5[10]: - 5.09765625 frac[10,5] + mem[10,5] = 0
 accuracy_5_sk_5[11]: - 5.09765625 frac[11,5] + mem[11,5] = 0
 accuracy_5_sk_5[12]: - 5.09765625 frac[12,5] + mem[12,5] = 0
 accuracy_5_sk_5[13]: - 5.09765625 frac[13,5] + mem[13,5] = 0
 accuracy_5_sk_5[14]: - 5.09765625 frac[14,5] + mem[14,5] = 0
 accuracy_5_sk_5[15]: - 5.09765625 frac[15,5] + mem[15,5] = 0
 accuracy_5_sk_5[16]: - 5.09765625 frac[16,5] + mem[16,5] = 0
 accuracy_5_sk_5[17]: - 5.09765625 frac[17,5] + mem[17,5] = 0
 accuracy_5_sk_5[18]: - 5.09765625 frac[18,5] + mem[18,5] = 0
 accuracy_5_sk_5[19]: - 5.09765625 frac[19,5] + mem[19,5] = 0
 accuracy_5_sk_5[20]: - 5.09765625 frac[20,5] + mem[20,5] = 0
 accuracy_6_sk_6[0]: - 5.09765625 frac[0,6] + mem[0,6] = 0
 accuracy_6_sk_6[1]: - 5.09765625 frac[1,6] + mem[1,6] = 0
 accuracy_6_sk_6[2]: - 5.09765625 frac[2,6] + mem[2,6] = 0
 accuracy_6_sk_6[3]: - 5.09765625 frac[3,6] + mem[3,6] = 0
 accuracy_6_sk_6[4]: - 5.09765625 frac[4,6] + mem[4,6] = 0
 accuracy_6_sk_6[5]: - 5.09765625 frac[5,6] + mem[5,6] = 0
 accuracy_6_sk_6[6]: - 5.09765625 frac[6,6] + mem[6,6] = 0
 accuracy_6_sk_6[7]: - 5.09765625 frac[7,6] + mem[7,6] = 0
 accuracy_6_sk_6[8]: - 5.09765625 frac[8,6] + mem[8,6] = 0
 accuracy_6_sk_6[9]: - 5.09765625 frac[9,6] + mem[9,6] = 0
 accuracy_6_sk_6[10]: - 5.09765625 frac[10,6] + mem[10,6] = 0
 accuracy_6_sk_6[11]: - 5.09765625 frac[11,6] + mem[11,6] = 0
 accuracy_6_sk_6[12]: - 5.09765625 frac[12,6] + mem[12,6] = 0
 accuracy_6_sk_6[13]: - 5.09765625 frac[13,6] + mem[13,6] = 0
 accuracy_6_sk_6[14]: - 5.09765625 frac[14,6] + mem[14,6] = 0
 accuracy_6_sk_6[15]: - 5.09765625 frac[15,6] + mem[15,6] = 0
 accuracy_6_sk_6[16]: - 5.09765625 frac[16,6] + mem[16,6] = 0
 accuracy_6_sk_6[17]: - 5.09765625 frac[17,6] + mem[17,6] = 0
 accuracy_6_sk_6[18]: - 5.09765625 frac[18,6] + mem[18,6] = 0
 accuracy_6_sk_6[19]: - 5.09765625 frac[19,6] + mem[19,6] = 0
 accuracy_6_sk_6[20]: - 5.09765625 frac[20,6] + mem[20,6] = 0
 accuracy_7_sk_7[0]: - 5.09765625 frac[0,7] + mem[0,7] = 0
 accuracy_7_sk_7[1]: - 5.09765625 frac[1,7] + mem[1,7] = 0
 accuracy_7_sk_7[2]: - 5.09765625 frac[2,7] + mem[2,7] = 0
 accuracy_7_sk_7[3]: - 5.09765625 frac[3,7] + mem[3,7] = 0
 accuracy_7_sk_7[4]: - 5.09765625 frac[4,7] + mem[4,7] = 0
 accuracy_7_sk_7[5]: - 5.09765625 frac[5,7] + mem[5,7] = 0
 accuracy_7_sk_7[6]: - 5.09765625 frac[6,7] + mem[6,7] = 0
 accuracy_7_sk_7[7]: - 5.09765625 frac[7,7] + mem[7,7] = 0
 accuracy_7_sk_7[8]: - 5.09765625 frac[8,7] + mem[8,7] = 0
 accuracy_7_sk_7[9]: - 5.09765625 frac[9,7] + mem[9,7] = 0
 accuracy_7_sk_7[10]: - 5.09765625 frac[10,7] + mem[10,7] = 0
 accuracy_7_sk_7[11]: - 5.09765625 frac[11,7] + mem[11,7] = 0
 accuracy_7_sk_7[12]: - 5.09765625 frac[12,7] + mem[12,7] = 0
 accuracy_7_sk_7[13]: - 5.09765625 frac[13,7] + mem[13,7] = 0
 accuracy_7_sk_7[14]: - 5.09765625 frac[14,7] + mem[14,7] = 0
 accuracy_7_sk_7[15]: - 5.09765625 frac[15,7] + mem[15,7] = 0
 accuracy_7_sk_7[16]: - 5.09765625 frac[16,7] + mem[16,7] = 0
 accuracy_7_sk_7[17]: - 5.09765625 frac[17,7] + mem[17,7] = 0
 accuracy_7_sk_7[18]: - 5.09765625 frac[18,7] + mem[18,7] = 0
 accuracy_7_sk_7[19]: - 5.09765625 frac[19,7] + mem[19,7] = 0
 accuracy_7_sk_7[20]: - 5.09765625 frac[20,7] + mem[20,7] = 0
 accuracy_8_sk_8[0]: - 5.09765625 frac[0,8] + mem[0,8] = 0
 accuracy_8_sk_8[1]: - 5.09765625 frac[1,8] + mem[1,8] = 0
 accuracy_8_sk_8[2]: - 5.09765625 frac[2,8] + mem[2,8] = 0
 accuracy_8_sk_8[3]: - 5.09765625 frac[3,8] + mem[3,8] = 0
 accuracy_8_sk_8[4]: - 5.09765625 frac[4,8] + mem[4,8] = 0
 accuracy_8_sk_8[5]: - 5.09765625 frac[5,8] + mem[5,8] = 0
 accuracy_8_sk_8[6]: - 5.09765625 frac[6,8] + mem[6,8] = 0
 accuracy_8_sk_8[7]: - 5.09765625 frac[7,8] + mem[7,8] = 0
 accuracy_8_sk_8[8]: - 5.09765625 frac[8,8] + mem[8,8] = 0
 accuracy_8_sk_8[9]: - 5.09765625 frac[9,8] + mem[9,8] = 0
 accuracy_8_sk_8[10]: - 5.09765625 frac[10,8] + mem[10,8] = 0
 accuracy_8_sk_8[11]: - 5.09765625 frac[11,8] + mem[11,8] = 0
 accuracy_8_sk_8[12]: - 5.09765625 frac[12,8] + mem[12,8] = 0
 accuracy_8_sk_8[13]: - 5.09765625 frac[13,8] + mem[13,8] = 0
 accuracy_8_sk_8[14]: - 5.09765625 frac[14,8] + mem[14,8] = 0
 accuracy_8_sk_8[15]: - 5.09765625 frac[15,8] + mem[15,8] = 0
 accuracy_8_sk_8[16]: - 5.09765625 frac[16,8] + mem[16,8] = 0
 accuracy_8_sk_8[17]: - 5.09765625 frac[17,8] + mem[17,8] = 0
 accuracy_8_sk_8[18]: - 5.09765625 frac[18,8] + mem[18,8] = 0
 accuracy_8_sk_8[19]: - 5.09765625 frac[19,8] + mem[19,8] = 0
 accuracy_8_sk_8[20]: - 5.09765625 frac[20,8] + mem[20,8] = 0
 accuracy_9_sk_9[0]: - 5.09765625 frac[0,9] + mem[0,9] = 0
 accuracy_9_sk_9[1]: - 5.09765625 frac[1,9] + mem[1,9] = 0
 accuracy_9_sk_9[2]: - 5.09765625 frac[2,9] + mem[2,9] = 0
 accuracy_9_sk_9[3]: - 5.09765625 frac[3,9] + mem[3,9] = 0
 accuracy_9_sk_9[4]: - 5.09765625 frac[4,9] + mem[4,9] = 0
 accuracy_9_sk_9[5]: - 5.09765625 frac[5,9] + mem[5,9] = 0
 accuracy_9_sk_9[6]: - 5.09765625 frac[6,9] + mem[6,9] = 0
 accuracy_9_sk_9[7]: - 5.09765625 frac[7,9] + mem[7,9] = 0
 accuracy_9_sk_9[8]: - 5.09765625 frac[8,9] + mem[8,9] = 0
 accuracy_9_sk_9[9]: - 5.09765625 frac[9,9] + mem[9,9] = 0
 accuracy_9_sk_9[10]: - 5.09765625 frac[10,9] + mem[10,9] = 0
 accuracy_9_sk_9[11]: - 5.09765625 frac[11,9] + mem[11,9] = 0
 accuracy_9_sk_9[12]: - 5.09765625 frac[12,9] + mem[12,9] = 0
 accuracy_9_sk_9[13]: - 5.09765625 frac[13,9] + mem[13,9] = 0
 accuracy_9_sk_9[14]: - 5.09765625 frac[14,9] + mem[14,9] = 0
 accuracy_9_sk_9[15]: - 5.09765625 frac[15,9] + mem[15,9] = 0
 accuracy_9_sk_9[16]: - 5.09765625 frac[16,9] + mem[16,9] = 0
 accuracy_9_sk_9[17]: - 5.09765625 frac[17,9] + mem[17,9] = 0
 accuracy_9_sk_9[18]: - 5.09765625 frac[18,9] + mem[18,9] = 0
 accuracy_9_sk_9[19]: - 5.09765625 frac[19,9] + mem[19,9] = 0
 accuracy_9_sk_9[20]: - 5.09765625 frac[20,9] + mem[20,9] = 0
 accuracy_10_sk_10[0]: - 5.09765625 frac[0,10] + mem[0,10] = 0
 accuracy_10_sk_10[1]: - 5.09765625 frac[1,10] + mem[1,10] = 0
 accuracy_10_sk_10[2]: - 5.09765625 frac[2,10] + mem[2,10] = 0
 accuracy_10_sk_10[3]: - 5.09765625 frac[3,10] + mem[3,10] = 0
 accuracy_10_sk_10[4]: - 5.09765625 frac[4,10] + mem[4,10] = 0
 accuracy_10_sk_10[5]: - 5.09765625 frac[5,10] + mem[5,10] = 0
 accuracy_10_sk_10[6]: - 5.09765625 frac[6,10] + mem[6,10] = 0
 accuracy_10_sk_10[7]: - 5.09765625 frac[7,10] + mem[7,10] = 0
 accuracy_10_sk_10[8]: - 5.09765625 frac[8,10] + mem[8,10] = 0
 accuracy_10_sk_10[9]: - 5.09765625 frac[9,10] + mem[9,10] = 0
 accuracy_10_sk_10[10]: - 5.09765625 frac[10,10] + mem[10,10] = 0
 accuracy_10_sk_10[11]: - 5.09765625 frac[11,10] + mem[11,10] = 0
 accuracy_10_sk_10[12]: - 5.09765625 frac[12,10] + mem[12,10] = 0
 accuracy_10_sk_10[13]: - 5.09765625 frac[13,10] + mem[13,10] = 0
 accuracy_10_sk_10[14]: - 5.09765625 frac[14,10] + mem[14,10] = 0
 accuracy_10_sk_10[15]: - 5.09765625 frac[15,10] + mem[15,10] = 0
 accuracy_10_sk_10[16]: - 5.09765625 frac[16,10] + mem[16,10] = 0
 accuracy_10_sk_10[17]: - 5.09765625 frac[17,10] + mem[17,10] = 0
 accuracy_10_sk_10[18]: - 5.09765625 frac[18,10] + mem[18,10] = 0
 accuracy_10_sk_10[19]: - 5.09765625 frac[19,10] + mem[19,10] = 0
 accuracy_10_sk_10[20]: - 5.09765625 frac[20,10] + mem[20,10] = 0
 accuracy_11_sk_11[0]: - 5.09765625 frac[0,11] + mem[0,11] = 0
 accuracy_11_sk_11[1]: - 5.09765625 frac[1,11] + mem[1,11] = 0
 accuracy_11_sk_11[2]: - 5.09765625 frac[2,11] + mem[2,11] = 0
 accuracy_11_sk_11[3]: - 5.09765625 frac[3,11] + mem[3,11] = 0
 accuracy_11_sk_11[4]: - 5.09765625 frac[4,11] + mem[4,11] = 0
 accuracy_11_sk_11[5]: - 5.09765625 frac[5,11] + mem[5,11] = 0
 accuracy_11_sk_11[6]: - 5.09765625 frac[6,11] + mem[6,11] = 0
 accuracy_11_sk_11[7]: - 5.09765625 frac[7,11] + mem[7,11] = 0
 accuracy_11_sk_11[8]: - 5.09765625 frac[8,11] + mem[8,11] = 0
 accuracy_11_sk_11[9]: - 5.09765625 frac[9,11] + mem[9,11] = 0
 accuracy_11_sk_11[10]: - 5.09765625 frac[10,11] + mem[10,11] = 0
 accuracy_11_sk_11[11]: - 5.09765625 frac[11,11] + mem[11,11] = 0
 accuracy_11_sk_11[12]: - 5.09765625 frac[12,11] + mem[12,11] = 0
 accuracy_11_sk_11[13]: - 5.09765625 frac[13,11] + mem[13,11] = 0
 accuracy_11_sk_11[14]: - 5.09765625 frac[14,11] + mem[14,11] = 0
 accuracy_11_sk_11[15]: - 5.09765625 frac[15,11] + mem[15,11] = 0
 accuracy_11_sk_11[16]: - 5.09765625 frac[16,11] + mem[16,11] = 0
 accuracy_11_sk_11[17]: - 5.09765625 frac[17,11] + mem[17,11] = 0
 accuracy_11_sk_11[18]: - 5.09765625 frac[18,11] + mem[18,11] = 0
 accuracy_11_sk_11[19]: - 5.09765625 frac[19,11] + mem[19,11] = 0
 accuracy_11_sk_11[20]: - 5.09765625 frac[20,11] + mem[20,11] = 0
 accuracy_12_sk_12[0]: - 5.09765625 frac[0,12] + mem[0,12] = 0
 accuracy_12_sk_12[1]: - 5.09765625 frac[1,12] + mem[1,12] = 0
 accuracy_12_sk_12[2]: - 5.09765625 frac[2,12] + mem[2,12] = 0
 accuracy_12_sk_12[3]: - 5.09765625 frac[3,12] + mem[3,12] = 0
 accuracy_12_sk_12[4]: - 5.09765625 frac[4,12] + mem[4,12] = 0
 accuracy_12_sk_12[5]: - 5.09765625 frac[5,12] + mem[5,12] = 0
 accuracy_12_sk_12[6]: - 5.09765625 frac[6,12] + mem[6,12] = 0
 accuracy_12_sk_12[7]: - 5.09765625 frac[7,12] + mem[7,12] = 0
 accuracy_12_sk_12[8]: - 5.09765625 frac[8,12] + mem[8,12] = 0
 accuracy_12_sk_12[9]: - 5.09765625 frac[9,12] + mem[9,12] = 0
 accuracy_12_sk_12[10]: - 5.09765625 frac[10,12] + mem[10,12] = 0
 accuracy_12_sk_12[11]: - 5.09765625 frac[11,12] + mem[11,12] = 0
 accuracy_12_sk_12[12]: - 5.09765625 frac[12,12] + mem[12,12] = 0
 accuracy_12_sk_12[13]: - 5.09765625 frac[13,12] + mem[13,12] = 0
 accuracy_12_sk_12[14]: - 5.09765625 frac[14,12] + mem[14,12] = 0
 accuracy_12_sk_12[15]: - 5.09765625 frac[15,12] + mem[15,12] = 0
 accuracy_12_sk_12[16]: - 5.09765625 frac[16,12] + mem[16,12] = 0
 accuracy_12_sk_12[17]: - 5.09765625 frac[17,12] + mem[17,12] = 0
 accuracy_12_sk_12[18]: - 5.09765625 frac[18,12] + mem[18,12] = 0
 accuracy_12_sk_12[19]: - 5.09765625 frac[19,12] + mem[19,12] = 0
 accuracy_12_sk_12[20]: - 5.09765625 frac[20,12] + mem[20,12] = 0
 accuracy_13_sk_13[0]: - 5.09765625 frac[0,13] + mem[0,13] = 0
 accuracy_13_sk_13[1]: - 5.09765625 frac[1,13] + mem[1,13] = 0
 accuracy_13_sk_13[2]: - 5.09765625 frac[2,13] + mem[2,13] = 0
 accuracy_13_sk_13[3]: - 5.09765625 frac[3,13] + mem[3,13] = 0
 accuracy_13_sk_13[4]: - 5.09765625 frac[4,13] + mem[4,13] = 0
 accuracy_13_sk_13[5]: - 5.09765625 frac[5,13] + mem[5,13] = 0
 accuracy_13_sk_13[6]: - 5.09765625 frac[6,13] + mem[6,13] = 0
 accuracy_13_sk_13[7]: - 5.09765625 frac[7,13] + mem[7,13] = 0
 accuracy_13_sk_13[8]: - 5.09765625 frac[8,13] + mem[8,13] = 0
 accuracy_13_sk_13[9]: - 5.09765625 frac[9,13] + mem[9,13] = 0
 accuracy_13_sk_13[10]: - 5.09765625 frac[10,13] + mem[10,13] = 0
 accuracy_13_sk_13[11]: - 5.09765625 frac[11,13] + mem[11,13] = 0
 accuracy_13_sk_13[12]: - 5.09765625 frac[12,13] + mem[12,13] = 0
 accuracy_13_sk_13[13]: - 5.09765625 frac[13,13] + mem[13,13] = 0
 accuracy_13_sk_13[14]: - 5.09765625 frac[14,13] + mem[14,13] = 0
 accuracy_13_sk_13[15]: - 5.09765625 frac[15,13] + mem[15,13] = 0
 accuracy_13_sk_13[16]: - 5.09765625 frac[16,13] + mem[16,13] = 0
 accuracy_13_sk_13[17]: - 5.09765625 frac[17,13] + mem[17,13] = 0
 accuracy_13_sk_13[18]: - 5.09765625 frac[18,13] + mem[18,13] = 0
 accuracy_13_sk_13[19]: - 5.09765625 frac[19,13] + mem[19,13] = 0
 accuracy_13_sk_13[20]: - 5.09765625 frac[20,13] + mem[20,13] = 0
 accuracy_14_sk_14[0]: - 5.09765625 frac[0,14] + mem[0,14] = 0
 accuracy_14_sk_14[1]: - 5.09765625 frac[1,14] + mem[1,14] = 0
 accuracy_14_sk_14[2]: - 5.09765625 frac[2,14] + mem[2,14] = 0
 accuracy_14_sk_14[3]: - 5.09765625 frac[3,14] + mem[3,14] = 0
 accuracy_14_sk_14[4]: - 5.09765625 frac[4,14] + mem[4,14] = 0
 accuracy_14_sk_14[5]: - 5.09765625 frac[5,14] + mem[5,14] = 0
 accuracy_14_sk_14[6]: - 5.09765625 frac[6,14] + mem[6,14] = 0
 accuracy_14_sk_14[7]: - 5.09765625 frac[7,14] + mem[7,14] = 0
 accuracy_14_sk_14[8]: - 5.09765625 frac[8,14] + mem[8,14] = 0
 accuracy_14_sk_14[9]: - 5.09765625 frac[9,14] + mem[9,14] = 0
 accuracy_14_sk_14[10]: - 5.09765625 frac[10,14] + mem[10,14] = 0
 accuracy_14_sk_14[11]: - 5.09765625 frac[11,14] + mem[11,14] = 0
 accuracy_14_sk_14[12]: - 5.09765625 frac[12,14] + mem[12,14] = 0
 accuracy_14_sk_14[13]: - 5.09765625 frac[13,14] + mem[13,14] = 0
 accuracy_14_sk_14[14]: - 5.09765625 frac[14,14] + mem[14,14] = 0
 accuracy_14_sk_14[15]: - 5.09765625 frac[15,14] + mem[15,14] = 0
 accuracy_14_sk_14[16]: - 5.09765625 frac[16,14] + mem[16,14] = 0
 accuracy_14_sk_14[17]: - 5.09765625 frac[17,14] + mem[17,14] = 0
 accuracy_14_sk_14[18]: - 5.09765625 frac[18,14] + mem[18,14] = 0
 accuracy_14_sk_14[19]: - 5.09765625 frac[19,14] + mem[19,14] = 0
 accuracy_14_sk_14[20]: - 5.09765625 frac[20,14] + mem[20,14] = 0
 accuracy_15_sk_15[0]: - 5.09765625 frac[0,15] + mem[0,15] = 0
 accuracy_15_sk_15[1]: - 5.09765625 frac[1,15] + mem[1,15] = 0
 accuracy_15_sk_15[2]: - 5.09765625 frac[2,15] + mem[2,15] = 0
 accuracy_15_sk_15[3]: - 5.09765625 frac[3,15] + mem[3,15] = 0
 accuracy_15_sk_15[4]: - 5.09765625 frac[4,15] + mem[4,15] = 0
 accuracy_15_sk_15[5]: - 5.09765625 frac[5,15] + mem[5,15] = 0
 accuracy_15_sk_15[6]: - 5.09765625 frac[6,15] + mem[6,15] = 0
 accuracy_15_sk_15[7]: - 5.09765625 frac[7,15] + mem[7,15] = 0
 accuracy_15_sk_15[8]: - 5.09765625 frac[8,15] + mem[8,15] = 0
 accuracy_15_sk_15[9]: - 5.09765625 frac[9,15] + mem[9,15] = 0
 accuracy_15_sk_15[10]: - 5.09765625 frac[10,15] + mem[10,15] = 0
 accuracy_15_sk_15[11]: - 5.09765625 frac[11,15] + mem[11,15] = 0
 accuracy_15_sk_15[12]: - 5.09765625 frac[12,15] + mem[12,15] = 0
 accuracy_15_sk_15[13]: - 5.09765625 frac[13,15] + mem[13,15] = 0
 accuracy_15_sk_15[14]: - 5.09765625 frac[14,15] + mem[14,15] = 0
 accuracy_15_sk_15[15]: - 5.09765625 frac[15,15] + mem[15,15] = 0
 accuracy_15_sk_15[16]: - 5.09765625 frac[16,15] + mem[16,15] = 0
 accuracy_15_sk_15[17]: - 5.09765625 frac[17,15] + mem[17,15] = 0
 accuracy_15_sk_15[18]: - 5.09765625 frac[18,15] + mem[18,15] = 0
 accuracy_15_sk_15[19]: - 5.09765625 frac[19,15] + mem[19,15] = 0
 accuracy_15_sk_15[20]: - 5.09765625 frac[20,15] + mem[20,15] = 0
 accuracy_16_sk_16[0]: - 5.09765625 frac[0,16] + mem[0,16] = 0
 accuracy_16_sk_16[1]: - 5.09765625 frac[1,16] + mem[1,16] = 0
 accuracy_16_sk_16[2]: - 5.09765625 frac[2,16] + mem[2,16] = 0
 accuracy_16_sk_16[3]: - 5.09765625 frac[3,16] + mem[3,16] = 0
 accuracy_16_sk_16[4]: - 5.09765625 frac[4,16] + mem[4,16] = 0
 accuracy_16_sk_16[5]: - 5.09765625 frac[5,16] + mem[5,16] = 0
 accuracy_16_sk_16[6]: - 5.09765625 frac[6,16] + mem[6,16] = 0
 accuracy_16_sk_16[7]: - 5.09765625 frac[7,16] + mem[7,16] = 0
 accuracy_16_sk_16[8]: - 5.09765625 frac[8,16] + mem[8,16] = 0
 accuracy_16_sk_16[9]: - 5.09765625 frac[9,16] + mem[9,16] = 0
 accuracy_16_sk_16[10]: - 5.09765625 frac[10,16] + mem[10,16] = 0
 accuracy_16_sk_16[11]: - 5.09765625 frac[11,16] + mem[11,16] = 0
 accuracy_16_sk_16[12]: - 5.09765625 frac[12,16] + mem[12,16] = 0
 accuracy_16_sk_16[13]: - 5.09765625 frac[13,16] + mem[13,16] = 0
 accuracy_16_sk_16[14]: - 5.09765625 frac[14,16] + mem[14,16] = 0
 accuracy_16_sk_16[15]: - 5.09765625 frac[15,16] + mem[15,16] = 0
 accuracy_16_sk_16[16]: - 5.09765625 frac[16,16] + mem[16,16] = 0
 accuracy_16_sk_16[17]: - 5.09765625 frac[17,16] + mem[17,16] = 0
 accuracy_16_sk_16[18]: - 5.09765625 frac[18,16] + mem[18,16] = 0
 accuracy_16_sk_16[19]: - 5.09765625 frac[19,16] + mem[19,16] = 0
 accuracy_16_sk_16[20]: - 5.09765625 frac[20,16] + mem[20,16] = 0
 accuracy_17_sk_17[0]: - 5.09765625 frac[0,17] + mem[0,17] = 0
 accuracy_17_sk_17[1]: - 5.09765625 frac[1,17] + mem[1,17] = 0
 accuracy_17_sk_17[2]: - 5.09765625 frac[2,17] + mem[2,17] = 0
 accuracy_17_sk_17[3]: - 5.09765625 frac[3,17] + mem[3,17] = 0
 accuracy_17_sk_17[4]: - 5.09765625 frac[4,17] + mem[4,17] = 0
 accuracy_17_sk_17[5]: - 5.09765625 frac[5,17] + mem[5,17] = 0
 accuracy_17_sk_17[6]: - 5.09765625 frac[6,17] + mem[6,17] = 0
 accuracy_17_sk_17[7]: - 5.09765625 frac[7,17] + mem[7,17] = 0
 accuracy_17_sk_17[8]: - 5.09765625 frac[8,17] + mem[8,17] = 0
 accuracy_17_sk_17[9]: - 5.09765625 frac[9,17] + mem[9,17] = 0
 accuracy_17_sk_17[10]: - 5.09765625 frac[10,17] + mem[10,17] = 0
 accuracy_17_sk_17[11]: - 5.09765625 frac[11,17] + mem[11,17] = 0
 accuracy_17_sk_17[12]: - 5.09765625 frac[12,17] + mem[12,17] = 0
 accuracy_17_sk_17[13]: - 5.09765625 frac[13,17] + mem[13,17] = 0
 accuracy_17_sk_17[14]: - 5.09765625 frac[14,17] + mem[14,17] = 0
 accuracy_17_sk_17[15]: - 5.09765625 frac[15,17] + mem[15,17] = 0
 accuracy_17_sk_17[16]: - 5.09765625 frac[16,17] + mem[16,17] = 0
 accuracy_17_sk_17[17]: - 5.09765625 frac[17,17] + mem[17,17] = 0
 accuracy_17_sk_17[18]: - 5.09765625 frac[18,17] + mem[18,17] = 0
 accuracy_17_sk_17[19]: - 5.09765625 frac[19,17] + mem[19,17] = 0
 accuracy_17_sk_17[20]: - 5.09765625 frac[20,17] + mem[20,17] = 0
 accuracy_18_sk_18[0]: - 5.09765625 frac[0,18] + mem[0,18] = 0
 accuracy_18_sk_18[1]: - 5.09765625 frac[1,18] + mem[1,18] = 0
 accuracy_18_sk_18[2]: - 5.09765625 frac[2,18] + mem[2,18] = 0
 accuracy_18_sk_18[3]: - 5.09765625 frac[3,18] + mem[3,18] = 0
 accuracy_18_sk_18[4]: - 5.09765625 frac[4,18] + mem[4,18] = 0
 accuracy_18_sk_18[5]: - 5.09765625 frac[5,18] + mem[5,18] = 0
 accuracy_18_sk_18[6]: - 5.09765625 frac[6,18] + mem[6,18] = 0
 accuracy_18_sk_18[7]: - 5.09765625 frac[7,18] + mem[7,18] = 0
 accuracy_18_sk_18[8]: - 5.09765625 frac[8,18] + mem[8,18] = 0
 accuracy_18_sk_18[9]: - 5.09765625 frac[9,18] + mem[9,18] = 0
 accuracy_18_sk_18[10]: - 5.09765625 frac[10,18] + mem[10,18] = 0
 accuracy_18_sk_18[11]: - 5.09765625 frac[11,18] + mem[11,18] = 0
 accuracy_18_sk_18[12]: - 5.09765625 frac[12,18] + mem[12,18] = 0
 accuracy_18_sk_18[13]: - 5.09765625 frac[13,18] + mem[13,18] = 0
 accuracy_18_sk_18[14]: - 5.09765625 frac[14,18] + mem[14,18] = 0
 accuracy_18_sk_18[15]: - 5.09765625 frac[15,18] + mem[15,18] = 0
 accuracy_18_sk_18[16]: - 5.09765625 frac[16,18] + mem[16,18] = 0
 accuracy_18_sk_18[17]: - 5.09765625 frac[17,18] + mem[17,18] = 0
 accuracy_18_sk_18[18]: - 5.09765625 frac[18,18] + mem[18,18] = 0
 accuracy_18_sk_18[19]: - 5.09765625 frac[19,18] + mem[19,18] = 0
 accuracy_18_sk_18[20]: - 5.09765625 frac[20,18] + mem[20,18] = 0
 accuracy_19_sk_19[0]: - 5.09765625 frac[0,19] + mem[0,19] = 0
 accuracy_19_sk_19[1]: - 5.09765625 frac[1,19] + mem[1,19] = 0
 accuracy_19_sk_19[2]: - 5.09765625 frac[2,19] + mem[2,19] = 0
 accuracy_19_sk_19[3]: - 5.09765625 frac[3,19] + mem[3,19] = 0
 accuracy_19_sk_19[4]: - 5.09765625 frac[4,19] + mem[4,19] = 0
 accuracy_19_sk_19[5]: - 5.09765625 frac[5,19] + mem[5,19] = 0
 accuracy_19_sk_19[6]: - 5.09765625 frac[6,19] + mem[6,19] = 0
 accuracy_19_sk_19[7]: - 5.09765625 frac[7,19] + mem[7,19] = 0
 accuracy_19_sk_19[8]: - 5.09765625 frac[8,19] + mem[8,19] = 0
 accuracy_19_sk_19[9]: - 5.09765625 frac[9,19] + mem[9,19] = 0
 accuracy_19_sk_19[10]: - 5.09765625 frac[10,19] + mem[10,19] = 0
 accuracy_19_sk_19[11]: - 5.09765625 frac[11,19] + mem[11,19] = 0
 accuracy_19_sk_19[12]: - 5.09765625 frac[12,19] + mem[12,19] = 0
 accuracy_19_sk_19[13]: - 5.09765625 frac[13,19] + mem[13,19] = 0
 accuracy_19_sk_19[14]: - 5.09765625 frac[14,19] + mem[14,19] = 0
 accuracy_19_sk_19[15]: - 5.09765625 frac[15,19] + mem[15,19] = 0
 accuracy_19_sk_19[16]: - 5.09765625 frac[16,19] + mem[16,19] = 0
 accuracy_19_sk_19[17]: - 5.09765625 frac[17,19] + mem[17,19] = 0
 accuracy_19_sk_19[18]: - 5.09765625 frac[18,19] + mem[18,19] = 0
 accuracy_19_sk_19[19]: - 5.09765625 frac[19,19] + mem[19,19] = 0
 accuracy_19_sk_19[20]: - 5.09765625 frac[20,19] + mem[20,19] = 0
 mem_tot_cpu_0: - mem[0,0] - mem[0,1] - mem[0,2] - mem[0,3] - mem[0,4]
   - mem[0,5] - mem[0,6] - mem[0,7] - mem[0,8] - mem[0,9] - mem[0,10]
   - mem[0,11] - mem[0,12] - mem[0,13] - mem[0,14] - mem[0,15] - mem[0,16]
   - mem[0,17] - mem[0,18] - mem[0,19] + mem_tot_cpu_0 = 0
 rows_tot_cpu_0: - 3 frac[0,0] - 3 frac[0,1] - 3 frac[0,2] - 3 frac[0,3]
   - 3 frac[0,4] - 3 frac[0,5] - 3 frac[0,6] - 3 frac[0,7] - 3 frac[0,8]
   - 3 frac[0,9] - 3 frac[0,10] - 3 frac[0,11] - 3 frac[0,12]
   - 3 frac[0,13] - 3 frac[0,14] - 3 frac[0,15] - 3 frac[0,16]
   - 3 frac[0,17] - 3 frac[0,18] - 3 frac[0,19] + rows_tot_cpu_0 = 0
 mem_tot_cpu_1: - mem[1,0] - mem[1,1] - mem[1,2] - mem[1,3] - mem[1,4]
   - mem[1,5] - mem[1,6] - mem[1,7] - mem[1,8] - mem[1,9] - mem[1,10]
   - mem[1,11] - mem[1,12] - mem[1,13] - mem[1,14] - mem[1,15] - mem[1,16]
   - mem[1,17] - mem[1,18] - mem[1,19] + mem_tot_cpu_1 = 0
 rows_tot_cpu_1: - 3 frac[1,0] - 3 frac[1,1] - 3 frac[1,2] - 3 frac[1,3]
   - 3 frac[1,4] - 3 frac[1,5] - 3 frac[1,6] - 3 frac[1,7] - 3 frac[1,8]
   - 3 frac[1,9] - 3 frac[1,10] - 3 frac[1,11] - 3 frac[1,12]
   - 3 frac[1,13] - 3 frac[1,14] - 3 frac[1,15] - 3 frac[1,16]
   - 3 frac[1,17] - 3 frac[1,18] - 3 frac[1,19] + rows_tot_cpu_1 = 0
 mem_tot_cpu_2: - mem[2,0] - mem[2,1] - mem[2,2] - mem[2,3] - mem[2,4]
   - mem[2,5] - mem[2,6] - mem[2,7] - mem[2,8] - mem[2,9] - mem[2,10]
   - mem[2,11] - mem[2,12] - mem[2,13] - mem[2,14] - mem[2,15] - mem[2,16]
   - mem[2,17] - mem[2,18] - mem[2,19] + mem_tot_cpu_2 = 0
 rows_tot_cpu_2: - 3 frac[2,0] - 3 frac[2,1] - 3 frac[2,2] - 3 frac[2,3]
   - 3 frac[2,4] - 3 frac[2,5] - 3 frac[2,6] - 3 frac[2,7] - 3 frac[2,8]
   - 3 frac[2,9] - 3 frac[2,10] - 3 frac[2,11] - 3 frac[2,12]
   - 3 frac[2,13] - 3 frac[2,14] - 3 frac[2,15] - 3 frac[2,16]
   - 3 frac[2,17] - 3 frac[2,18] - 3 frac[2,19] + rows_tot_cpu_2 = 0
 mem_tot_cpu_3: - mem[3,0] - mem[3,1] - mem[3,2] - mem[3,3] - mem[3,4]
   - mem[3,5] - mem[3,6] - mem[3,7] - mem[3,8] - mem[3,9] - mem[3,10]
   - mem[3,11] - mem[3,12] - mem[3,13] - mem[3,14] - mem[3,15] - mem[3,16]
   - mem[3,17] - mem[3,18] - mem[3,19] + mem_tot_cpu_3 = 0
 rows_tot_cpu_3: - 3 frac[3,0] - 3 frac[3,1] - 3 frac[3,2] - 3 frac[3,3]
   - 3 frac[3,4] - 3 frac[3,5] - 3 frac[3,6] - 3 frac[3,7] - 3 frac[3,8]
   - 3 frac[3,9] - 3 frac[3,10] - 3 frac[3,11] - 3 frac[3,12]
   - 3 frac[3,13] - 3 frac[3,14] - 3 frac[3,15] - 3 frac[3,16]
   - 3 frac[3,17] - 3 frac[3,18] - 3 frac[3,19] + rows_tot_cpu_3 = 0
 mem_tot_cpu_4: - mem[4,0] - mem[4,1] - mem[4,2] - mem[4,3] - mem[4,4]
   - mem[4,5] - mem[4,6] - mem[4,7] - mem[4,8] - mem[4,9] - mem[4,10]
   - mem[4,11] - mem[4,12] - mem[4,13] - mem[4,14] - mem[4,15] - mem[4,16]
   - mem[4,17] - mem[4,18] - mem[4,19] + mem_tot_cpu_4 = 0
 rows_tot_cpu_4: - 3 frac[4,0] - 3 frac[4,1] - 3 frac[4,2] - 3 frac[4,3]
   - 3 frac[4,4] - 3 frac[4,5] - 3 frac[4,6] - 3 frac[4,7] - 3 frac[4,8]
   - 3 frac[4,9] - 3 frac[4,10] - 3 frac[4,11] - 3 frac[4,12]
   - 3 frac[4,13] - 3 frac[4,14] - 3 frac[4,15] - 3 frac[4,16]
   - 3 frac[4,17] - 3 frac[4,18] - 3 frac[4,19] + rows_tot_cpu_4 = 0
 mem_tot_cpu_5: - mem[5,0] - mem[5,1] - mem[5,2] - mem[5,3] - mem[5,4]
   - mem[5,5] - mem[5,6] - mem[5,7] - mem[5,8] - mem[5,9] - mem[5,10]
   - mem[5,11] - mem[5,12] - mem[5,13] - mem[5,14] - mem[5,15] - mem[5,16]
   - mem[5,17] - mem[5,18] - mem[5,19] + mem_tot_cpu_5 = 0
 rows_tot_cpu_5: - 3 frac[5,0] - 3 frac[5,1] - 3 frac[5,2] - 3 frac[5,3]
   - 3 frac[5,4] - 3 frac[5,5] - 3 frac[5,6] - 3 frac[5,7] - 3 frac[5,8]
   - 3 frac[5,9] - 3 frac[5,10] - 3 frac[5,11] - 3 frac[5,12]
   - 3 frac[5,13] - 3 frac[5,14] - 3 frac[5,15] - 3 frac[5,16]
   - 3 frac[5,17] - 3 frac[5,18] - 3 frac[5,19] + rows_tot_cpu_5 = 0
 mem_tot_cpu_6: - mem[6,0] - mem[6,1] - mem[6,2] - mem[6,3] - mem[6,4]
   - mem[6,5] - mem[6,6] - mem[6,7] - mem[6,8] - mem[6,9] - mem[6,10]
   - mem[6,11] - mem[6,12] - mem[6,13] - mem[6,14] - mem[6,15] - mem[6,16]
   - mem[6,17] - mem[6,18] - mem[6,19] + mem_tot_cpu_6 = 0
 rows_tot_cpu_6: - 3 frac[6,0] - 3 frac[6,1] - 3 frac[6,2] - 3 frac[6,3]
   - 3 frac[6,4] - 3 frac[6,5] - 3 frac[6,6] - 3 frac[6,7] - 3 frac[6,8]
   - 3 frac[6,9] - 3 frac[6,10] - 3 frac[6,11] - 3 frac[6,12]
   - 3 frac[6,13] - 3 frac[6,14] - 3 frac[6,15] - 3 frac[6,16]
   - 3 frac[6,17] - 3 frac[6,18] - 3 frac[6,19] + rows_tot_cpu_6 = 0
 mem_tot_cpu_7: - mem[7,0] - mem[7,1] - mem[7,2] - mem[7,3] - mem[7,4]
   - mem[7,5] - mem[7,6] - mem[7,7] - mem[7,8] - mem[7,9] - mem[7,10]
   - mem[7,11] - mem[7,12] - mem[7,13] - mem[7,14] - mem[7,15] - mem[7,16]
   - mem[7,17] - mem[7,18] - mem[7,19] + mem_tot_cpu_7 = 0
 rows_tot_cpu_7: - 3 frac[7,0] - 3 frac[7,1] - 3 frac[7,2] - 3 frac[7,3]
   - 3 frac[7,4] - 3 frac[7,5] - 3 frac[7,6] - 3 frac[7,7] - 3 frac[7,8]
   - 3 frac[7,9] - 3 frac[7,10] - 3 frac[7,11] - 3 frac[7,12]
   - 3 frac[7,13] - 3 frac[7,14] - 3 frac[7,15] - 3 frac[7,16]
   - 3 frac[7,17] - 3 frac[7,18] - 3 frac[7,19] + rows_tot_cpu_7 = 0
 mem_tot_cpu_8: - mem[8,0] - mem[8,1] - mem[8,2] - mem[8,3] - mem[8,4]
   - mem[8,5] - mem[8,6] - mem[8,7] - mem[8,8] - mem[8,9] - mem[8,10]
   - mem[8,11] - mem[8,12] - mem[8,13] - mem[8,14] - mem[8,15] - mem[8,16]
   - mem[8,17] - mem[8,18] - mem[8,19] + mem_tot_cpu_8 = 0
 rows_tot_cpu_8: - 3 frac[8,0] - 3 frac[8,1] - 3 frac[8,2] - 3 frac[8,3]
   - 3 frac[8,4] - 3 frac[8,5] - 3 frac[8,6] - 3 frac[8,7] - 3 frac[8,8]
   - 3 frac[8,9] - 3 frac[8,10] - 3 frac[8,11] - 3 frac[8,12]
   - 3 frac[8,13] - 3 frac[8,14] - 3 frac[8,15] - 3 frac[8,16]
   - 3 frac[8,17] - 3 frac[8,18] - 3 frac[8,19] + rows_tot_cpu_8 = 0
 mem_tot_cpu_9: - mem[9,0] - mem[9,1] - mem[9,2] - mem[9,3] - mem[9,4]
   - mem[9,5] - mem[9,6] - mem[9,7] - mem[9,8] - mem[9,9] - mem[9,10]
   - mem[9,11] - mem[9,12] - mem[9,13] - mem[9,14] - mem[9,15] - mem[9,16]
   - mem[9,17] - mem[9,18] - mem[9,19] + mem_tot_cpu_9 = 0
 rows_tot_cpu_9: - 3 frac[9,0] - 3 frac[9,1] - 3 frac[9,2] - 3 frac[9,3]
   - 3 frac[9,4] - 3 frac[9,5] - 3 frac[9,6] - 3 frac[9,7] - 3 frac[9,8]
   - 3 frac[9,9] - 3 frac[9,10] - 3 frac[9,11] - 3 frac[9,12]
   - 3 frac[9,13] - 3 frac[9,14] - 3 frac[9,15] - 3 frac[9,16]
   - 3 frac[9,17] - 3 frac[9,18] - 3 frac[9,19] + rows_tot_cpu_9 = 0
 mem_tot_cpu_10: - mem[10,0] - mem[10,1] - mem[10,2] - mem[10,3]
   - mem[10,4] - mem[10,5] - mem[10,6] - mem[10,7] - mem[10,8] - mem[10,9]
   - mem[10,10] - mem[10,11] - mem[10,12] - mem[10,13] - mem[10,14]
   - mem[10,15] - mem[10,16] - mem[10,17] - mem[10,18] - mem[10,19]
   + mem_tot_cpu_10 = 0
 rows_tot_cpu_10: - 3 frac[10,0] - 3 frac[10,1] - 3 frac[10,2]
   - 3 frac[10,3] - 3 frac[10,4] - 3 frac[10,5] - 3 frac[10,6]
   - 3 frac[10,7] - 3 frac[10,8] - 3 frac[10,9] - 3 frac[10,10]
   - 3 frac[10,11] - 3 frac[10,12] - 3 frac[10,13] - 3 frac[10,14]
   - 3 frac[10,15] - 3 frac[10,16] - 3 frac[10,17] - 3 frac[10,18]
   - 3 frac[10,19] + rows_tot_cpu_10 = 0
 mem_tot_cpu_11: - mem[11,0] - mem[11,1] - mem[11,2] - mem[11,3]
   - mem[11,4] - mem[11,5] - mem[11,6] - mem[11,7] - mem[11,8] - mem[11,9]
   - mem[11,10] - mem[11,11] - mem[11,12] - mem[11,13] - mem[11,14]
   - mem[11,15] - mem[11,16] - mem[11,17] - mem[11,18] - mem[11,19]
   + mem_tot_cpu_11 = 0
 rows_tot_cpu_11: - 3 frac[11,0] - 3 frac[11,1] - 3 frac[11,2]
   - 3 frac[11,3] - 3 frac[11,4] - 3 frac[11,5] - 3 frac[11,6]
   - 3 frac[11,7] - 3 frac[11,8] - 3 frac[11,9] - 3 frac[11,10]
   - 3 frac[11,11] - 3 frac[11,12] - 3 frac[11,13] - 3 frac[11,14]
   - 3 frac[11,15] - 3 frac[11,16] - 3 frac[11,17] - 3 frac[11,18]
   - 3 frac[11,19] + rows_tot_cpu_11 = 0
 mem_tot_cpu_12: - mem[12,0] - mem[12,1] - mem[12,2] - mem[12,3]
   - mem[12,4] - mem[12,5] - mem[12,6] - mem[12,7] - mem[12,8] - mem[12,9]
   - mem[12,10] - mem[12,11] - mem[12,12] - mem[12,13] - mem[12,14]
   - mem[12,15] - mem[12,16] - mem[12,17] - mem[12,18] - mem[12,19]
   + mem_tot_cpu_12 = 0
 rows_tot_cpu_12: - 3 frac[12,0] - 3 frac[12,1] - 3 frac[12,2]
   - 3 frac[12,3] - 3 frac[12,4] - 3 frac[12,5] - 3 frac[12,6]
   - 3 frac[12,7] - 3 frac[12,8] - 3 frac[12,9] - 3 frac[12,10]
   - 3 frac[12,11] - 3 frac[12,12] - 3 frac[12,13] - 3 frac[12,14]
   - 3 frac[12,15] - 3 frac[12,16] - 3 frac[12,17] - 3 frac[12,18]
   - 3 frac[12,19] + rows_tot_cpu_12 = 0
 mem_tot_cpu_13: - mem[13,0] - mem[13,1] - mem[13,2] - mem[13,3]
   - mem[13,4] - mem[13,5] - mem[13,6] - mem[13,7] - mem[13,8] - mem[13,9]
   - mem[13,10] - mem[13,11] - mem[13,12] - mem[13,13] - mem[13,14]
   - mem[13,15] - mem[13,16] - mem[13,17] - mem[13,18] - mem[13,19]
   + mem_tot_cpu_13 = 0
 rows_tot_cpu_13: - 3 frac[13,0] - 3 frac[13,1] - 3 frac[13,2]
   - 3 frac[13,3] - 3 frac[13,4] - 3 frac[13,5] - 3 frac[13,6]
   - 3 frac[13,7] - 3 frac[13,8] - 3 frac[13,9] - 3 frac[13,10]
   - 3 frac[13,11] - 3 frac[13,12] - 3 frac[13,13] - 3 frac[13,14]
   - 3 frac[13,15] - 3 frac[13,16] - 3 frac[13,17] - 3 frac[13,18]
   - 3 frac[13,19] + rows_tot_cpu_13 = 0
 mem_tot_cpu_14: - mem[14,0] - mem[14,1] - mem[14,2] - mem[14,3]
   - mem[14,4] - mem[14,5] - mem[14,6] - mem[14,7] - mem[14,8] - mem[14,9]
   - mem[14,10] - mem[14,11] - mem[14,12] - mem[14,13] - mem[14,14]
   - mem[14,15] - mem[14,16] - mem[14,17] - mem[14,18] - mem[14,19]
   + mem_tot_cpu_14 = 0
 rows_tot_cpu_14: - 3 frac[14,0] - 3 frac[14,1] - 3 frac[14,2]
   - 3 frac[14,3] - 3 frac[14,4] - 3 frac[14,5] - 3 frac[14,6]
   - 3 frac[14,7] - 3 frac[14,8] - 3 frac[14,9] - 3 frac[14,10]
   - 3 frac[14,11] - 3 frac[14,12] - 3 frac[14,13] - 3 frac[14,14]
   - 3 frac[14,15] - 3 frac[14,16] - 3 frac[14,17] - 3 frac[14,18]
   - 3 frac[14,19] + rows_tot_cpu_14 = 0
 mem_tot_cpu_15: - mem[15,0] - mem[15,1] - mem[15,2] - mem[15,3]
   - mem[15,4] - mem[15,5] - mem[15,6] - mem[15,7] - mem[15,8] - mem[15,9]
   - mem[15,10] - mem[15,11] - mem[15,12] - mem[15,13] - mem[15,14]
   - mem[15,15] - mem[15,16] - mem[15,17] - mem[15,18] - mem[15,19]
   + mem_tot_cpu_15 = 0
 rows_tot_cpu_15: - 3 frac[15,0] - 3 frac[15,1] - 3 frac[15,2]
   - 3 frac[15,3] - 3 frac[15,4] - 3 frac[15,5] - 3 frac[15,6]
   - 3 frac[15,7] - 3 frac[15,8] - 3 frac[15,9] - 3 frac[15,10]
   - 3 frac[15,11] - 3 frac[15,12] - 3 frac[15,13] - 3 frac[15,14]
   - 3 frac[15,15] - 3 frac[15,16] - 3 frac[15,17] - 3 frac[15,18]
   - 3 frac[15,19] + rows_tot_cpu_15 = 0
 mem_tot_cpu_16: - mem[16,0] - mem[16,1] - mem[16,2] - mem[16,3]
   - mem[16,4] - mem[16,5] - mem[16,6] - mem[16,7] - mem[16,8] - mem[16,9]
   - mem[16,10] - mem[16,11] - mem[16,12] - mem[16,13] - mem[16,14]
   - mem[16,15] - mem[16,16] - mem[16,17] - mem[16,18] - mem[16,19]
   + mem_tot_cpu_16 = 0
 rows_tot_cpu_16: - 3 frac[16,0] - 3 frac[16,1] - 3 frac[16,2]
   - 3 frac[16,3] - 3 frac[16,4] - 3 frac[16,5] - 3 frac[16,6]
   - 3 frac[16,7] - 3 frac[16,8] - 3 frac[16,9] - 3 frac[16,10]
   - 3 frac[16,11] - 3 frac[16,12] - 3 frac[16,13] - 3 frac[16,14]
   - 3 frac[16,15] - 3 frac[16,16] - 3 frac[16,17] - 3 frac[16,18]
   - 3 frac[16,19] + rows_tot_cpu_16 = 0
 mem_tot_cpu_17: - mem[17,0] - mem[17,1] - mem[17,2] - mem[17,3]
   - mem[17,4] - mem[17,5] - mem[17,6] - mem[17,7] - mem[17,8] - mem[17,9]
   - mem[17,10] - mem[17,11] - mem[17,12] - mem[17,13] - mem[17,14]
   - mem[17,15] - mem[17,16] - mem[17,17] - mem[17,18] - mem[17,19]
   + mem_tot_cpu_17 = 0
 rows_tot_cpu_17: - 3 frac[17,0] - 3 frac[17,1] - 3 frac[17,2]
   - 3 frac[17,3] - 3 frac[17,4] - 3 frac[17,5] - 3 frac[17,6]
   - 3 frac[17,7] - 3 frac[17,8] - 3 frac[17,9] - 3 frac[17,10]
   - 3 frac[17,11] - 3 frac[17,12] - 3 frac[17,13] - 3 frac[17,14]
   - 3 frac[17,15] - 3 frac[17,16] - 3 frac[17,17] - 3 frac[17,18]
   - 3 frac[17,19] + rows_tot_cpu_17 = 0
 mem_tot_cpu_18: - mem[18,0] - mem[18,1] - mem[18,2] - mem[18,3]
   - mem[18,4] - mem[18,5] - mem[18,6] - mem[18,7] - mem[18,8] - mem[18,9]
   - mem[18,10] - mem[18,11] - mem[18,12] - mem[18,13] - mem[18,14]
   - mem[18,15] - mem[18,16] - mem[18,17] - mem[18,18] - mem[18,19]
   + mem_tot_cpu_18 = 0
 rows_tot_cpu_18: - 3 frac[18,0] - 3 frac[18,1] - 3 frac[18,2]
   - 3 frac[18,3] - 3 frac[18,4] - 3 frac[18,5] - 3 frac[18,6]
   - 3 frac[18,7] - 3 frac[18,8] - 3 frac[18,9] - 3 frac[18,10]
   - 3 frac[18,11] - 3 frac[18,12] - 3 frac[18,13] - 3 frac[18,14]
   - 3 frac[18,15] - 3 frac[18,16] - 3 frac[18,17] - 3 frac[18,18]
   - 3 frac[18,19] + rows_tot_cpu_18 = 0
 mem_tot_cpu_19: - mem[19,0] - mem[19,1] - mem[19,2] - mem[19,3]
   - mem[19,4] - mem[19,5] - mem[19,6] - mem[19,7] - mem[19,8] - mem[19,9]
   - mem[19,10] - mem[19,11] - mem[19,12] - mem[19,13] - mem[19,14]
   - mem[19,15] - mem[19,16] - mem[19,17] - mem[19,18] - mem[19,19]
   + mem_tot_cpu_19 = 0
 rows_tot_cpu_19: - 3 frac[19,0] - 3 frac[19,1] - 3 frac[19,2]
   - 3 frac[19,3] - 3 frac[19,4] - 3 frac[19,5] - 3 frac[19,6]
   - 3 frac[19,7] - 3 frac[19,8] - 3 frac[19,9] - 3 frac[19,10]
   - 3 frac[19,11] - 3 frac[19,12] - 3 frac[19,13] - 3 frac[19,14]
   - 3 frac[19,15] - 3 frac[19,16] - 3 frac[19,17] - 3 frac[19,18]
   - 3 frac[19,19] + rows_tot_cpu_19 = 0
 mem_tot_p4_1: - mem[20,0] - mem[20,1] - mem[20,2] - mem[20,3] - mem[20,4]
   - mem[20,5] - mem[20,6] - mem[20,7] - mem[20,8] - mem[20,9] - mem[20,10]
   - mem[20,11] - mem[20,12] - mem[20,13] - mem[20,14] - mem[20,15]
   - mem[20,16] - mem[20,17] - mem[20,18] - mem[20,19] + mem_tot_p4_1 = 0
 rows_tot_p4_1: - 3 frac[20,0] - 3 frac[20,1] - 3 frac[20,2] - 3 frac[20,3]
   - 3 frac[20,4] - 3 frac[20,5] - 3 frac[20,6] - 3 frac[20,7]
   - 3 frac[20,8] - 3 frac[20,9] - 3 frac[20,10] - 3 frac[20,11]
   - 3 frac[20,12] - 3 frac[20,13] - 3 frac[20,14] - 3 frac[20,15]
   - 3 frac[20,16] - 3 frac[20,17] - 3 frac[20,18] - 3 frac[20,19]
   + rows_tot_p4_1 = 0
 row_capacity_cpu_0: rows_tot_cpu_0 <= 12
 row_capacity_cpu_1: rows_tot_cpu_1 <= 12
 row_capacity_cpu_2: rows_tot_cpu_2 <= 12
 row_capacity_cpu_3: rows_tot_cpu_3 <= 12
 row_capacity_cpu_4: rows_tot_cpu_4 <= 12
 row_capacity_cpu_5: rows_tot_cpu_5 <= 12
 row_capacity_cpu_6: rows_tot_cpu_6 <= 12
 row_capacity_cpu_7: rows_tot_cpu_7 <= 12
 row_capacity_cpu_8: rows_tot_cpu_8 <= 12
 row_capacity_cpu_9: rows_tot_cpu_9 <= 12
 row_capacity_cpu_10: rows_tot_cpu_10 <= 12
 row_capacity_cpu_11: rows_tot_cpu_11 <= 12
 row_capacity_cpu_12: rows_tot_cpu_12 <= 12
 row_capacity_cpu_13: rows_tot_cpu_13 <= 12
 row_capacity_cpu_14: rows_tot_cpu_14 <= 12
 row_capacity_cpu_15: rows_tot_cpu_15 <= 12
 row_capacity_cpu_16: rows_tot_cpu_16 <= 12
 row_capacity_cpu_17: rows_tot_cpu_17 <= 12
 row_capacity_cpu_18: rows_tot_cpu_18 <= 12
 row_capacity_cpu_19: rows_tot_cpu_19 <= 12
 row_capacity_p4_1: rows_tot_p4_1 <= 48
 ns_single_cpu_0: 3.5 rows_tot_cpu_0 - ns_single_cpu_0 + pdt_m_rows_cpu_0
   = 0
 capacity_cores_cpu_0: cores_sketch_cpu_0 + cores_dpdk_cpu_0 <= 7
 ns_sketch_cpu_0: - ns_single_cpu_0 + pdt_nsk_c_cpu_0 = 0
 ns_dpdk_cpu_0: 7.142857142857143 cores_dpdk_cpu_0 - pdt_nsc_dpdk_cpu_0
   = -21.42857142857143
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
 ns_single_cpu_3: 3.5 rows_tot_cpu_3 - ns_single_cpu_3 + pdt_m_rows_cpu_3
   = 0
 capacity_cores_cpu_3: cores_sketch_cpu_3 + cores_dpdk_cpu_3 <= 7
 ns_sketch_cpu_3: - ns_single_cpu_3 + pdt_nsk_c_cpu_3 = 0
 ns_dpdk_cpu_3: 7.142857142857143 cores_dpdk_cpu_3 - pdt_nsc_dpdk_cpu_3
   = -21.42857142857143
 ns_single_cpu_4: 3.5 rows_tot_cpu_4 - ns_single_cpu_4 + pdt_m_rows_cpu_4
   = 0
 capacity_cores_cpu_4: cores_sketch_cpu_4 + cores_dpdk_cpu_4 <= 7
 ns_sketch_cpu_4: - ns_single_cpu_4 + pdt_nsk_c_cpu_4 = 0
 ns_dpdk_cpu_4: 7.142857142857143 cores_dpdk_cpu_4 - pdt_nsc_dpdk_cpu_4
   = -21.42857142857143
 ns_single_cpu_5: 3.5 rows_tot_cpu_5 - ns_single_cpu_5 + pdt_m_rows_cpu_5
   = 0
 capacity_cores_cpu_5: cores_sketch_cpu_5 + cores_dpdk_cpu_5 <= 7
 ns_sketch_cpu_5: - ns_single_cpu_5 + pdt_nsk_c_cpu_5 = 0
 ns_dpdk_cpu_5: 7.142857142857143 cores_dpdk_cpu_5 - pdt_nsc_dpdk_cpu_5
   = -21.42857142857143
 ns_single_cpu_6: 3.5 rows_tot_cpu_6 - ns_single_cpu_6 + pdt_m_rows_cpu_6
   = 0
 capacity_cores_cpu_6: cores_sketch_cpu_6 + cores_dpdk_cpu_6 <= 7
 ns_sketch_cpu_6: - ns_single_cpu_6 + pdt_nsk_c_cpu_6 = 0
 ns_dpdk_cpu_6: 7.142857142857143 cores_dpdk_cpu_6 - pdt_nsc_dpdk_cpu_6
   = -21.42857142857143
 ns_single_cpu_7: 3.5 rows_tot_cpu_7 - ns_single_cpu_7 + pdt_m_rows_cpu_7
   = 0
 capacity_cores_cpu_7: cores_sketch_cpu_7 + cores_dpdk_cpu_7 <= 7
 ns_sketch_cpu_7: - ns_single_cpu_7 + pdt_nsk_c_cpu_7 = 0
 ns_dpdk_cpu_7: 7.142857142857143 cores_dpdk_cpu_7 - pdt_nsc_dpdk_cpu_7
   = -21.42857142857143
 ns_single_cpu_8: 3.5 rows_tot_cpu_8 - ns_single_cpu_8 + pdt_m_rows_cpu_8
   = 0
 capacity_cores_cpu_8: cores_sketch_cpu_8 + cores_dpdk_cpu_8 <= 7
 ns_sketch_cpu_8: - ns_single_cpu_8 + pdt_nsk_c_cpu_8 = 0
 ns_dpdk_cpu_8: 7.142857142857143 cores_dpdk_cpu_8 - pdt_nsc_dpdk_cpu_8
   = -21.42857142857143
 ns_single_cpu_9: 3.5 rows_tot_cpu_9 - ns_single_cpu_9 + pdt_m_rows_cpu_9
   = 0
 capacity_cores_cpu_9: cores_sketch_cpu_9 + cores_dpdk_cpu_9 <= 7
 ns_sketch_cpu_9: - ns_single_cpu_9 + pdt_nsk_c_cpu_9 = 0
 ns_dpdk_cpu_9: 7.142857142857143 cores_dpdk_cpu_9 - pdt_nsc_dpdk_cpu_9
   = -21.42857142857143
 ns_single_cpu_10: 3.5 rows_tot_cpu_10 - ns_single_cpu_10
   + pdt_m_rows_cpu_10 = 0
 capacity_cores_cpu_10: cores_sketch_cpu_10 + cores_dpdk_cpu_10 <= 7
 ns_sketch_cpu_10: - ns_single_cpu_10 + pdt_nsk_c_cpu_10 = 0
 ns_dpdk_cpu_10: 7.142857142857143 cores_dpdk_cpu_10 - pdt_nsc_dpdk_cpu_10
   = -21.42857142857143
 ns_single_cpu_11: 3.5 rows_tot_cpu_11 - ns_single_cpu_11
   + pdt_m_rows_cpu_11 = 0
 capacity_cores_cpu_11: cores_sketch_cpu_11 + cores_dpdk_cpu_11 <= 7
 ns_sketch_cpu_11: - ns_single_cpu_11 + pdt_nsk_c_cpu_11 = 0
 ns_dpdk_cpu_11: 7.142857142857143 cores_dpdk_cpu_11 - pdt_nsc_dpdk_cpu_11
   = -21.42857142857143
 ns_single_cpu_12: 3.5 rows_tot_cpu_12 - ns_single_cpu_12
   + pdt_m_rows_cpu_12 = 0
 capacity_cores_cpu_12: cores_sketch_cpu_12 + cores_dpdk_cpu_12 <= 7
 ns_sketch_cpu_12: - ns_single_cpu_12 + pdt_nsk_c_cpu_12 = 0
 ns_dpdk_cpu_12: 7.142857142857143 cores_dpdk_cpu_12 - pdt_nsc_dpdk_cpu_12
   = -21.42857142857143
 ns_single_cpu_13: 3.5 rows_tot_cpu_13 - ns_single_cpu_13
   + pdt_m_rows_cpu_13 = 0
 capacity_cores_cpu_13: cores_sketch_cpu_13 + cores_dpdk_cpu_13 <= 7
 ns_sketch_cpu_13: - ns_single_cpu_13 + pdt_nsk_c_cpu_13 = 0
 ns_dpdk_cpu_13: 7.142857142857143 cores_dpdk_cpu_13 - pdt_nsc_dpdk_cpu_13
   = -21.42857142857143
 ns_single_cpu_14: 3.5 rows_tot_cpu_14 - ns_single_cpu_14
   + pdt_m_rows_cpu_14 = 0
 capacity_cores_cpu_14: cores_sketch_cpu_14 + cores_dpdk_cpu_14 <= 7
 ns_sketch_cpu_14: - ns_single_cpu_14 + pdt_nsk_c_cpu_14 = 0
 ns_dpdk_cpu_14: 7.142857142857143 cores_dpdk_cpu_14 - pdt_nsc_dpdk_cpu_14
   = -21.42857142857143
 ns_single_cpu_15: 3.5 rows_tot_cpu_15 - ns_single_cpu_15
   + pdt_m_rows_cpu_15 = 0
 capacity_cores_cpu_15: cores_sketch_cpu_15 + cores_dpdk_cpu_15 <= 7
 ns_sketch_cpu_15: - ns_single_cpu_15 + pdt_nsk_c_cpu_15 = 0
 ns_dpdk_cpu_15: 7.142857142857143 cores_dpdk_cpu_15 - pdt_nsc_dpdk_cpu_15
   = -21.42857142857143
 ns_single_cpu_16: 3.5 rows_tot_cpu_16 - ns_single_cpu_16
   + pdt_m_rows_cpu_16 = 0
 capacity_cores_cpu_16: cores_sketch_cpu_16 + cores_dpdk_cpu_16 <= 7
 ns_sketch_cpu_16: - ns_single_cpu_16 + pdt_nsk_c_cpu_16 = 0
 ns_dpdk_cpu_16: 7.142857142857143 cores_dpdk_cpu_16 - pdt_nsc_dpdk_cpu_16
   = -21.42857142857143
 ns_single_cpu_17: 3.5 rows_tot_cpu_17 - ns_single_cpu_17
   + pdt_m_rows_cpu_17 = 0
 capacity_cores_cpu_17: cores_sketch_cpu_17 + cores_dpdk_cpu_17 <= 7
 ns_sketch_cpu_17: - ns_single_cpu_17 + pdt_nsk_c_cpu_17 = 0
 ns_dpdk_cpu_17: 7.142857142857143 cores_dpdk_cpu_17 - pdt_nsc_dpdk_cpu_17
   = -21.42857142857143
 ns_single_cpu_18: 3.5 rows_tot_cpu_18 - ns_single_cpu_18
   + pdt_m_rows_cpu_18 = 0
 capacity_cores_cpu_18: cores_sketch_cpu_18 + cores_dpdk_cpu_18 <= 7
 ns_sketch_cpu_18: - ns_single_cpu_18 + pdt_nsk_c_cpu_18 = 0
 ns_dpdk_cpu_18: 7.142857142857143 cores_dpdk_cpu_18 - pdt_nsc_dpdk_cpu_18
   = -21.42857142857143
 ns_single_cpu_19: 3.5 rows_tot_cpu_19 - ns_single_cpu_19
   + pdt_m_rows_cpu_19 = 0
 capacity_cores_cpu_19: cores_sketch_cpu_19 + cores_dpdk_cpu_19 <= 7
 ns_sketch_cpu_19: - ns_single_cpu_19 + pdt_nsk_c_cpu_19 = 0
 ns_dpdk_cpu_19: 7.142857142857143 cores_dpdk_cpu_19 - pdt_nsc_dpdk_cpu_19
   = -21.42857142857143
 ns_p4_1: ns_p4_1 = 6.756756756756757
 res_acc: 1.220703125e-04 mem_tot_cpu_0 + 1.220703125e-04 mem_tot_cpu_1
   + 1.220703125e-04 mem_tot_cpu_2 + 1.220703125e-04 mem_tot_cpu_3
   + 1.220703125e-04 mem_tot_cpu_4 + 1.220703125e-04 mem_tot_cpu_5
   + 1.220703125e-04 mem_tot_cpu_6 + 1.220703125e-04 mem_tot_cpu_7
   + 1.220703125e-04 mem_tot_cpu_8 + 1.220703125e-04 mem_tot_cpu_9
   + 1.220703125e-04 mem_tot_cpu_10 + 1.220703125e-04 mem_tot_cpu_11
   + 1.220703125e-04 mem_tot_cpu_12 + 1.220703125e-04 mem_tot_cpu_13
   + 1.220703125e-04 mem_tot_cpu_14 + 1.220703125e-04 mem_tot_cpu_15
   + 1.220703125e-04 mem_tot_cpu_16 + 1.220703125e-04 mem_tot_cpu_17
   + 1.220703125e-04 mem_tot_cpu_18 + 1.220703125e-04 mem_tot_cpu_19
   + 0.0208333333333333 mem_tot_p4_1 + 0.25 rows_tot_p4_1
   + 10 cores_sketch_cpu_0 + 10 cores_dpdk_cpu_0 + 10 cores_sketch_cpu_1
   + 10 cores_dpdk_cpu_1 + 10 cores_sketch_cpu_2 + 10 cores_dpdk_cpu_2
   + 10 cores_sketch_cpu_3 + 10 cores_dpdk_cpu_3 + 10 cores_sketch_cpu_4
   + 10 cores_dpdk_cpu_4 + 10 cores_sketch_cpu_5 + 10 cores_dpdk_cpu_5
   + 10 cores_sketch_cpu_6 + 10 cores_dpdk_cpu_6 + 10 cores_sketch_cpu_7
   + 10 cores_dpdk_cpu_7 + 10 cores_sketch_cpu_8 + 10 cores_dpdk_cpu_8
   + 10 cores_sketch_cpu_9 + 10 cores_dpdk_cpu_9 + 10 cores_sketch_cpu_10
   + 10 cores_dpdk_cpu_10 + 10 cores_sketch_cpu_11 + 10 cores_dpdk_cpu_11
   + 10 cores_sketch_cpu_12 + 10 cores_dpdk_cpu_12 + 10 cores_sketch_cpu_13
   + 10 cores_dpdk_cpu_13 + 10 cores_sketch_cpu_14 + 10 cores_dpdk_cpu_14
   + 10 cores_sketch_cpu_15 + 10 cores_dpdk_cpu_15 + 10 cores_sketch_cpu_16
   + 10 cores_dpdk_cpu_16 + 10 cores_sketch_cpu_17 + 10 cores_dpdk_cpu_17
   + 10 cores_sketch_cpu_18 + 10 cores_dpdk_cpu_18 + 10 cores_sketch_cpu_19
   + 10 cores_dpdk_cpu_19 - res = 0
 pdt_m_rows_cpu_0: pdt_m_rows_cpu_0 + [
   - rows_tot_cpu_0 * m_access_time_cpu_0 ] = 0
 pdt_nsk_c_cpu_0: pdt_nsk_c_cpu_0 + [
   - cores_sketch_cpu_0 * ns_sketch_cpu_0 ] = 0
 pdt_nsc_dpdk_cpu_0: pdt_nsc_dpdk_cpu_0 + [
   - cores_dpdk_cpu_0 * ns_dpdk_cpu_0 ] = 0
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
 pdt_m_rows_cpu_3: pdt_m_rows_cpu_3 + [
   - rows_tot_cpu_3 * m_access_time_cpu_3 ] = 0
 pdt_nsk_c_cpu_3: pdt_nsk_c_cpu_3 + [
   - cores_sketch_cpu_3 * ns_sketch_cpu_3 ] = 0
 pdt_nsc_dpdk_cpu_3: pdt_nsc_dpdk_cpu_3 + [
   - cores_dpdk_cpu_3 * ns_dpdk_cpu_3 ] = 0
 pdt_m_rows_cpu_4: pdt_m_rows_cpu_4 + [
   - rows_tot_cpu_4 * m_access_time_cpu_4 ] = 0
 pdt_nsk_c_cpu_4: pdt_nsk_c_cpu_4 + [
   - cores_sketch_cpu_4 * ns_sketch_cpu_4 ] = 0
 pdt_nsc_dpdk_cpu_4: pdt_nsc_dpdk_cpu_4 + [
   - cores_dpdk_cpu_4 * ns_dpdk_cpu_4 ] = 0
 pdt_m_rows_cpu_5: pdt_m_rows_cpu_5 + [
   - rows_tot_cpu_5 * m_access_time_cpu_5 ] = 0
 pdt_nsk_c_cpu_5: pdt_nsk_c_cpu_5 + [
   - cores_sketch_cpu_5 * ns_sketch_cpu_5 ] = 0
 pdt_nsc_dpdk_cpu_5: pdt_nsc_dpdk_cpu_5 + [
   - cores_dpdk_cpu_5 * ns_dpdk_cpu_5 ] = 0
 pdt_m_rows_cpu_6: pdt_m_rows_cpu_6 + [
   - rows_tot_cpu_6 * m_access_time_cpu_6 ] = 0
 pdt_nsk_c_cpu_6: pdt_nsk_c_cpu_6 + [
   - cores_sketch_cpu_6 * ns_sketch_cpu_6 ] = 0
 pdt_nsc_dpdk_cpu_6: pdt_nsc_dpdk_cpu_6 + [
   - cores_dpdk_cpu_6 * ns_dpdk_cpu_6 ] = 0
 pdt_m_rows_cpu_7: pdt_m_rows_cpu_7 + [
   - rows_tot_cpu_7 * m_access_time_cpu_7 ] = 0
 pdt_nsk_c_cpu_7: pdt_nsk_c_cpu_7 + [
   - cores_sketch_cpu_7 * ns_sketch_cpu_7 ] = 0
 pdt_nsc_dpdk_cpu_7: pdt_nsc_dpdk_cpu_7 + [
   - cores_dpdk_cpu_7 * ns_dpdk_cpu_7 ] = 0
 pdt_m_rows_cpu_8: pdt_m_rows_cpu_8 + [
   - rows_tot_cpu_8 * m_access_time_cpu_8 ] = 0
 pdt_nsk_c_cpu_8: pdt_nsk_c_cpu_8 + [
   - cores_sketch_cpu_8 * ns_sketch_cpu_8 ] = 0
 pdt_nsc_dpdk_cpu_8: pdt_nsc_dpdk_cpu_8 + [
   - cores_dpdk_cpu_8 * ns_dpdk_cpu_8 ] = 0
 pdt_m_rows_cpu_9: pdt_m_rows_cpu_9 + [
   - rows_tot_cpu_9 * m_access_time_cpu_9 ] = 0
 pdt_nsk_c_cpu_9: pdt_nsk_c_cpu_9 + [
   - cores_sketch_cpu_9 * ns_sketch_cpu_9 ] = 0
 pdt_nsc_dpdk_cpu_9: pdt_nsc_dpdk_cpu_9 + [
   - cores_dpdk_cpu_9 * ns_dpdk_cpu_9 ] = 0
 pdt_m_rows_cpu_10: pdt_m_rows_cpu_10 + [
   - rows_tot_cpu_10 * m_access_time_cpu_10 ] = 0
 pdt_nsk_c_cpu_10: pdt_nsk_c_cpu_10 + [
   - cores_sketch_cpu_10 * ns_sketch_cpu_10 ] = 0
 pdt_nsc_dpdk_cpu_10: pdt_nsc_dpdk_cpu_10 + [
   - cores_dpdk_cpu_10 * ns_dpdk_cpu_10 ] = 0
 pdt_m_rows_cpu_11: pdt_m_rows_cpu_11 + [
   - rows_tot_cpu_11 * m_access_time_cpu_11 ] = 0
 pdt_nsk_c_cpu_11: pdt_nsk_c_cpu_11 + [
   - cores_sketch_cpu_11 * ns_sketch_cpu_11 ] = 0
 pdt_nsc_dpdk_cpu_11: pdt_nsc_dpdk_cpu_11 + [
   - cores_dpdk_cpu_11 * ns_dpdk_cpu_11 ] = 0
 pdt_m_rows_cpu_12: pdt_m_rows_cpu_12 + [
   - rows_tot_cpu_12 * m_access_time_cpu_12 ] = 0
 pdt_nsk_c_cpu_12: pdt_nsk_c_cpu_12 + [
   - cores_sketch_cpu_12 * ns_sketch_cpu_12 ] = 0
 pdt_nsc_dpdk_cpu_12: pdt_nsc_dpdk_cpu_12 + [
   - cores_dpdk_cpu_12 * ns_dpdk_cpu_12 ] = 0
 pdt_m_rows_cpu_13: pdt_m_rows_cpu_13 + [
   - rows_tot_cpu_13 * m_access_time_cpu_13 ] = 0
 pdt_nsk_c_cpu_13: pdt_nsk_c_cpu_13 + [
   - cores_sketch_cpu_13 * ns_sketch_cpu_13 ] = 0
 pdt_nsc_dpdk_cpu_13: pdt_nsc_dpdk_cpu_13 + [
   - cores_dpdk_cpu_13 * ns_dpdk_cpu_13 ] = 0
 pdt_m_rows_cpu_14: pdt_m_rows_cpu_14 + [
   - rows_tot_cpu_14 * m_access_time_cpu_14 ] = 0
 pdt_nsk_c_cpu_14: pdt_nsk_c_cpu_14 + [
   - cores_sketch_cpu_14 * ns_sketch_cpu_14 ] = 0
 pdt_nsc_dpdk_cpu_14: pdt_nsc_dpdk_cpu_14 + [
   - cores_dpdk_cpu_14 * ns_dpdk_cpu_14 ] = 0
 pdt_m_rows_cpu_15: pdt_m_rows_cpu_15 + [
   - rows_tot_cpu_15 * m_access_time_cpu_15 ] = 0
 pdt_nsk_c_cpu_15: pdt_nsk_c_cpu_15 + [
   - cores_sketch_cpu_15 * ns_sketch_cpu_15 ] = 0
 pdt_nsc_dpdk_cpu_15: pdt_nsc_dpdk_cpu_15 + [
   - cores_dpdk_cpu_15 * ns_dpdk_cpu_15 ] = 0
 pdt_m_rows_cpu_16: pdt_m_rows_cpu_16 + [
   - rows_tot_cpu_16 * m_access_time_cpu_16 ] = 0
 pdt_nsk_c_cpu_16: pdt_nsk_c_cpu_16 + [
   - cores_sketch_cpu_16 * ns_sketch_cpu_16 ] = 0
 pdt_nsc_dpdk_cpu_16: pdt_nsc_dpdk_cpu_16 + [
   - cores_dpdk_cpu_16 * ns_dpdk_cpu_16 ] = 0
 pdt_m_rows_cpu_17: pdt_m_rows_cpu_17 + [
   - rows_tot_cpu_17 * m_access_time_cpu_17 ] = 0
 pdt_nsk_c_cpu_17: pdt_nsk_c_cpu_17 + [
   - cores_sketch_cpu_17 * ns_sketch_cpu_17 ] = 0
 pdt_nsc_dpdk_cpu_17: pdt_nsc_dpdk_cpu_17 + [
   - cores_dpdk_cpu_17 * ns_dpdk_cpu_17 ] = 0
 pdt_m_rows_cpu_18: pdt_m_rows_cpu_18 + [
   - rows_tot_cpu_18 * m_access_time_cpu_18 ] = 0
 pdt_nsk_c_cpu_18: pdt_nsk_c_cpu_18 + [
   - cores_sketch_cpu_18 * ns_sketch_cpu_18 ] = 0
 pdt_nsc_dpdk_cpu_18: pdt_nsc_dpdk_cpu_18 + [
   - cores_dpdk_cpu_18 * ns_dpdk_cpu_18 ] = 0
 pdt_m_rows_cpu_19: pdt_m_rows_cpu_19 + [
   - rows_tot_cpu_19 * m_access_time_cpu_19 ] = 0
 pdt_nsk_c_cpu_19: pdt_nsk_c_cpu_19 + [
   - cores_sketch_cpu_19 * ns_sketch_cpu_19 ] = 0
 pdt_nsc_dpdk_cpu_19: pdt_nsc_dpdk_cpu_19 + [
   - cores_dpdk_cpu_19 * ns_dpdk_cpu_19 ] = 0
Bounds
 mem_tot_cpu_0 <= 32768
 mem_tot_cpu_1 <= 32768
 mem_tot_cpu_2 <= 32768
 mem_tot_cpu_3 <= 32768
 mem_tot_cpu_4 <= 32768
 mem_tot_cpu_5 <= 32768
 mem_tot_cpu_6 <= 32768
 mem_tot_cpu_7 <= 32768
 mem_tot_cpu_8 <= 32768
 mem_tot_cpu_9 <= 32768
 mem_tot_cpu_10 <= 32768
 mem_tot_cpu_11 <= 32768
 mem_tot_cpu_12 <= 32768
 mem_tot_cpu_13 <= 32768
 mem_tot_cpu_14 <= 32768
 mem_tot_cpu_15 <= 32768
 mem_tot_cpu_16 <= 32768
 mem_tot_cpu_17 <= 32768
 mem_tot_cpu_18 <= 32768
 mem_tot_cpu_19 <= 32768
 mem_tot_p4_1 <= 576
 cores_sketch_cpu_0 <= 7
 1 <= cores_dpdk_cpu_0 <= 7
 cores_sketch_cpu_1 <= 7
 1 <= cores_dpdk_cpu_1 <= 7
 cores_sketch_cpu_2 <= 7
 1 <= cores_dpdk_cpu_2 <= 7
 cores_sketch_cpu_3 <= 7
 1 <= cores_dpdk_cpu_3 <= 7
 cores_sketch_cpu_4 <= 7
 1 <= cores_dpdk_cpu_4 <= 7
 cores_sketch_cpu_5 <= 7
 1 <= cores_dpdk_cpu_5 <= 7
 cores_sketch_cpu_6 <= 7
 1 <= cores_dpdk_cpu_6 <= 7
 cores_sketch_cpu_7 <= 7
 1 <= cores_dpdk_cpu_7 <= 7
 cores_sketch_cpu_8 <= 7
 1 <= cores_dpdk_cpu_8 <= 7
 cores_sketch_cpu_9 <= 7
 1 <= cores_dpdk_cpu_9 <= 7
 cores_sketch_cpu_10 <= 7
 1 <= cores_dpdk_cpu_10 <= 7
 cores_sketch_cpu_11 <= 7
 1 <= cores_dpdk_cpu_11 <= 7
 cores_sketch_cpu_12 <= 7
 1 <= cores_dpdk_cpu_12 <= 7
 cores_sketch_cpu_13 <= 7
 1 <= cores_dpdk_cpu_13 <= 7
 cores_sketch_cpu_14 <= 7
 1 <= cores_dpdk_cpu_14 <= 7
 cores_sketch_cpu_15 <= 7
 1 <= cores_dpdk_cpu_15 <= 7
 cores_sketch_cpu_16 <= 7
 1 <= cores_dpdk_cpu_16 <= 7
 cores_sketch_cpu_17 <= 7
 1 <= cores_dpdk_cpu_17 <= 7
 cores_sketch_cpu_18 <= 7
 1 <= cores_dpdk_cpu_18 <= 7
 cores_sketch_cpu_19 <= 7
 1 <= cores_dpdk_cpu_19 <= 7
Binaries
 frac[0,0] frac[0,1] frac[0,2] frac[0,3] frac[0,4] frac[0,5] frac[0,6]
 frac[0,7] frac[0,8] frac[0,9] frac[0,10] frac[0,11] frac[0,12] frac[0,13]
 frac[0,14] frac[0,15] frac[0,16] frac[0,17] frac[0,18] frac[0,19]
 frac[1,0] frac[1,1] frac[1,2] frac[1,3] frac[1,4] frac[1,5] frac[1,6]
 frac[1,7] frac[1,8] frac[1,9] frac[1,10] frac[1,11] frac[1,12] frac[1,13]
 frac[1,14] frac[1,15] frac[1,16] frac[1,17] frac[1,18] frac[1,19]
 frac[2,0] frac[2,1] frac[2,2] frac[2,3] frac[2,4] frac[2,5] frac[2,6]
 frac[2,7] frac[2,8] frac[2,9] frac[2,10] frac[2,11] frac[2,12] frac[2,13]
 frac[2,14] frac[2,15] frac[2,16] frac[2,17] frac[2,18] frac[2,19]
 frac[3,0] frac[3,1] frac[3,2] frac[3,3] frac[3,4] frac[3,5] frac[3,6]
 frac[3,7] frac[3,8] frac[3,9] frac[3,10] frac[3,11] frac[3,12] frac[3,13]
 frac[3,14] frac[3,15] frac[3,16] frac[3,17] frac[3,18] frac[3,19]
 frac[4,0] frac[4,1] frac[4,2] frac[4,3] frac[4,4] frac[4,5] frac[4,6]
 frac[4,7] frac[4,8] frac[4,9] frac[4,10] frac[4,11] frac[4,12] frac[4,13]
 frac[4,14] frac[4,15] frac[4,16] frac[4,17] frac[4,18] frac[4,19]
 frac[5,0] frac[5,1] frac[5,2] frac[5,3] frac[5,4] frac[5,5] frac[5,6]
 frac[5,7] frac[5,8] frac[5,9] frac[5,10] frac[5,11] frac[5,12] frac[5,13]
 frac[5,14] frac[5,15] frac[5,16] frac[5,17] frac[5,18] frac[5,19]
 frac[6,0] frac[6,1] frac[6,2] frac[6,3] frac[6,4] frac[6,5] frac[6,6]
 frac[6,7] frac[6,8] frac[6,9] frac[6,10] frac[6,11] frac[6,12] frac[6,13]
 frac[6,14] frac[6,15] frac[6,16] frac[6,17] frac[6,18] frac[6,19]
 frac[7,0] frac[7,1] frac[7,2] frac[7,3] frac[7,4] frac[7,5] frac[7,6]
 frac[7,7] frac[7,8] frac[7,9] frac[7,10] frac[7,11] frac[7,12] frac[7,13]
 frac[7,14] frac[7,15] frac[7,16] frac[7,17] frac[7,18] frac[7,19]
 frac[8,0] frac[8,1] frac[8,2] frac[8,3] frac[8,4] frac[8,5] frac[8,6]
 frac[8,7] frac[8,8] frac[8,9] frac[8,10] frac[8,11] frac[8,12] frac[8,13]
 frac[8,14] frac[8,15] frac[8,16] frac[8,17] frac[8,18] frac[8,19]
 frac[9,0] frac[9,1] frac[9,2] frac[9,3] frac[9,4] frac[9,5] frac[9,6]
 frac[9,7] frac[9,8] frac[9,9] frac[9,10] frac[9,11] frac[9,12] frac[9,13]
 frac[9,14] frac[9,15] frac[9,16] frac[9,17] frac[9,18] frac[9,19]
 frac[10,0] frac[10,1] frac[10,2] frac[10,3] frac[10,4] frac[10,5]
 frac[10,6] frac[10,7] frac[10,8] frac[10,9] frac[10,10] frac[10,11]
 frac[10,12] frac[10,13] frac[10,14] frac[10,15] frac[10,16] frac[10,17]
 frac[10,18] frac[10,19] frac[11,0] frac[11,1] frac[11,2] frac[11,3]
 frac[11,4] frac[11,5] frac[11,6] frac[11,7] frac[11,8] frac[11,9]
 frac[11,10] frac[11,11] frac[11,12] frac[11,13] frac[11,14] frac[11,15]
 frac[11,16] frac[11,17] frac[11,18] frac[11,19] frac[12,0] frac[12,1]
 frac[12,2] frac[12,3] frac[12,4] frac[12,5] frac[12,6] frac[12,7]
 frac[12,8] frac[12,9] frac[12,10] frac[12,11] frac[12,12] frac[12,13]
 frac[12,14] frac[12,15] frac[12,16] frac[12,17] frac[12,18] frac[12,19]
 frac[13,0] frac[13,1] frac[13,2] frac[13,3] frac[13,4] frac[13,5]
 frac[13,6] frac[13,7] frac[13,8] frac[13,9] frac[13,10] frac[13,11]
 frac[13,12] frac[13,13] frac[13,14] frac[13,15] frac[13,16] frac[13,17]
 frac[13,18] frac[13,19] frac[14,0] frac[14,1] frac[14,2] frac[14,3]
 frac[14,4] frac[14,5] frac[14,6] frac[14,7] frac[14,8] frac[14,9]
 frac[14,10] frac[14,11] frac[14,12] frac[14,13] frac[14,14] frac[14,15]
 frac[14,16] frac[14,17] frac[14,18] frac[14,19] frac[15,0] frac[15,1]
 frac[15,2] frac[15,3] frac[15,4] frac[15,5] frac[15,6] frac[15,7]
 frac[15,8] frac[15,9] frac[15,10] frac[15,11] frac[15,12] frac[15,13]
 frac[15,14] frac[15,15] frac[15,16] frac[15,17] frac[15,18] frac[15,19]
 frac[16,0] frac[16,1] frac[16,2] frac[16,3] frac[16,4] frac[16,5]
 frac[16,6] frac[16,7] frac[16,8] frac[16,9] frac[16,10] frac[16,11]
 frac[16,12] frac[16,13] frac[16,14] frac[16,15] frac[16,16] frac[16,17]
 frac[16,18] frac[16,19] frac[17,0] frac[17,1] frac[17,2] frac[17,3]
 frac[17,4] frac[17,5] frac[17,6] frac[17,7] frac[17,8] frac[17,9]
 frac[17,10] frac[17,11] frac[17,12] frac[17,13] frac[17,14] frac[17,15]
 frac[17,16] frac[17,17] frac[17,18] frac[17,19] frac[18,0] frac[18,1]
 frac[18,2] frac[18,3] frac[18,4] frac[18,5] frac[18,6] frac[18,7]
 frac[18,8] frac[18,9] frac[18,10] frac[18,11] frac[18,12] frac[18,13]
 frac[18,14] frac[18,15] frac[18,16] frac[18,17] frac[18,18] frac[18,19]
 frac[19,0] frac[19,1] frac[19,2] frac[19,3] frac[19,4] frac[19,5]
 frac[19,6] frac[19,7] frac[19,8] frac[19,9] frac[19,10] frac[19,11]
 frac[19,12] frac[19,13] frac[19,14] frac[19,15] frac[19,16] frac[19,17]
 frac[19,18] frac[19,19] frac[20,0] frac[20,1] frac[20,2] frac[20,3]
 frac[20,4] frac[20,5] frac[20,6] frac[20,7] frac[20,8] frac[20,9]
 frac[20,10] frac[20,11] frac[20,12] frac[20,13] frac[20,14] frac[20,15]
 frac[20,16] frac[20,17] frac[20,18] frac[20,19]
Generals
 cores_sketch_cpu_0 cores_dpdk_cpu_0 cores_sketch_cpu_1 cores_dpdk_cpu_1
 cores_sketch_cpu_2 cores_dpdk_cpu_2 cores_sketch_cpu_3 cores_dpdk_cpu_3
 cores_sketch_cpu_4 cores_dpdk_cpu_4 cores_sketch_cpu_5 cores_dpdk_cpu_5
 cores_sketch_cpu_6 cores_dpdk_cpu_6 cores_sketch_cpu_7 cores_dpdk_cpu_7
 cores_sketch_cpu_8 cores_dpdk_cpu_8 cores_sketch_cpu_9 cores_dpdk_cpu_9
 cores_sketch_cpu_10 cores_dpdk_cpu_10 cores_sketch_cpu_11
 cores_dpdk_cpu_11 cores_sketch_cpu_12 cores_dpdk_cpu_12
 cores_sketch_cpu_13 cores_dpdk_cpu_13 cores_sketch_cpu_14
 cores_dpdk_cpu_14 cores_sketch_cpu_15 cores_dpdk_cpu_15
 cores_sketch_cpu_16 cores_dpdk_cpu_16 cores_sketch_cpu_17
 cores_dpdk_cpu_17 cores_sketch_cpu_18 cores_dpdk_cpu_18
 cores_sketch_cpu_19 cores_dpdk_cpu_19
General Constraints
 mem_access_time_cpu_0: m_access_time_cpu_0 = PWL ( mem_tot_cpu_0 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_0: ns_cpu_0 = MAX ( ns_dpdk_cpu_0 , ns_sketch_cpu_0 )
 mem_access_time_cpu_1: m_access_time_cpu_1 = PWL ( mem_tot_cpu_1 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_1: ns_cpu_1 = MAX ( ns_dpdk_cpu_1 , ns_sketch_cpu_1 )
 mem_access_time_cpu_2: m_access_time_cpu_2 = PWL ( mem_tot_cpu_2 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_2: ns_cpu_2 = MAX ( ns_dpdk_cpu_2 , ns_sketch_cpu_2 )
 mem_access_time_cpu_3: m_access_time_cpu_3 = PWL ( mem_tot_cpu_3 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_3: ns_cpu_3 = MAX ( ns_dpdk_cpu_3 , ns_sketch_cpu_3 )
 mem_access_time_cpu_4: m_access_time_cpu_4 = PWL ( mem_tot_cpu_4 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_4: ns_cpu_4 = MAX ( ns_dpdk_cpu_4 , ns_sketch_cpu_4 )
 mem_access_time_cpu_5: m_access_time_cpu_5 = PWL ( mem_tot_cpu_5 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_5: ns_cpu_5 = MAX ( ns_dpdk_cpu_5 , ns_sketch_cpu_5 )
 mem_access_time_cpu_6: m_access_time_cpu_6 = PWL ( mem_tot_cpu_6 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_6: ns_cpu_6 = MAX ( ns_dpdk_cpu_6 , ns_sketch_cpu_6 )
 mem_access_time_cpu_7: m_access_time_cpu_7 = PWL ( mem_tot_cpu_7 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_7: ns_cpu_7 = MAX ( ns_dpdk_cpu_7 , ns_sketch_cpu_7 )
 mem_access_time_cpu_8: m_access_time_cpu_8 = PWL ( mem_tot_cpu_8 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_8: ns_cpu_8 = MAX ( ns_dpdk_cpu_8 , ns_sketch_cpu_8 )
 mem_access_time_cpu_9: m_access_time_cpu_9 = PWL ( mem_tot_cpu_9 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_9: ns_cpu_9 = MAX ( ns_dpdk_cpu_9 , ns_sketch_cpu_9 )
 mem_access_time_cpu_10: m_access_time_cpu_10 = PWL ( mem_tot_cpu_10 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_10: ns_cpu_10 = MAX ( ns_dpdk_cpu_10 , ns_sketch_cpu_10 )
 mem_access_time_cpu_11: m_access_time_cpu_11 = PWL ( mem_tot_cpu_11 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_11: ns_cpu_11 = MAX ( ns_dpdk_cpu_11 , ns_sketch_cpu_11 )
 mem_access_time_cpu_12: m_access_time_cpu_12 = PWL ( mem_tot_cpu_12 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_12: ns_cpu_12 = MAX ( ns_dpdk_cpu_12 , ns_sketch_cpu_12 )
 mem_access_time_cpu_13: m_access_time_cpu_13 = PWL ( mem_tot_cpu_13 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_13: ns_cpu_13 = MAX ( ns_dpdk_cpu_13 , ns_sketch_cpu_13 )
 mem_access_time_cpu_14: m_access_time_cpu_14 = PWL ( mem_tot_cpu_14 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_14: ns_cpu_14 = MAX ( ns_dpdk_cpu_14 , ns_sketch_cpu_14 )
 mem_access_time_cpu_15: m_access_time_cpu_15 = PWL ( mem_tot_cpu_15 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_15: ns_cpu_15 = MAX ( ns_dpdk_cpu_15 , ns_sketch_cpu_15 )
 mem_access_time_cpu_16: m_access_time_cpu_16 = PWL ( mem_tot_cpu_16 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_16: ns_cpu_16 = MAX ( ns_dpdk_cpu_16 , ns_sketch_cpu_16 )
 mem_access_time_cpu_17: m_access_time_cpu_17 = PWL ( mem_tot_cpu_17 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_17: ns_cpu_17 = MAX ( ns_dpdk_cpu_17 , ns_sketch_cpu_17 )
 mem_access_time_cpu_18: m_access_time_cpu_18 = PWL ( mem_tot_cpu_18 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_18: ns_cpu_18 = MAX ( ns_dpdk_cpu_18 , ns_sketch_cpu_18 )
 mem_access_time_cpu_19: m_access_time_cpu_19 = PWL ( mem_tot_cpu_19 ) : 
  (0, 0) (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_cpu_19: ns_cpu_19 = MAX ( ns_dpdk_cpu_19 , ns_sketch_cpu_19 )
 ns_overall: ns = MAX ( ns_cpu_0 , ns_cpu_1 , ns_cpu_2 , ns_cpu_3 ,
   ns_cpu_4 , ns_cpu_5 , ns_cpu_6 , ns_cpu_7 , ns_cpu_8 , ns_cpu_9 ,
   ns_cpu_10 , ns_cpu_11 , ns_cpu_12 , ns_cpu_13 , ns_cpu_14 , ns_cpu_15 ,
   ns_cpu_16 , ns_cpu_17 , ns_cpu_18 , ns_cpu_19 , ns_p4_1 )
End

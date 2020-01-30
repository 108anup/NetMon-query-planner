\ Model netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  ns: Priority=10 Weight=1 AbsTol=0 RelTol=0.05
   0 m_access_time_endhost_cpu1 + 0 ns_sketch_endhost_cpu1
   + 0 ns_dpdk_endhost_cpu1 + 0 ns_endhost_cpu1
   + 0 m_access_time_endhost_cpu2 + 0 ns_sketch_endhost_cpu2
   + 0 ns_dpdk_endhost_cpu2 + 0 ns_endhost_cpu2
   + 0 m_access_time_endhost_cpu3 + 0 ns_sketch_endhost_cpu3
   + 0 ns_dpdk_endhost_cpu3 + 0 ns_endhost_cpu3
   + 0 m_access_time_endhost_cpu4 + 0 ns_sketch_endhost_cpu4
   + 0 ns_dpdk_endhost_cpu4 + 0 ns_endhost_cpu4
   + 0 m_access_time_endhost_cpu5 + 0 ns_sketch_endhost_cpu5
   + 0 ns_dpdk_endhost_cpu5 + 0 ns_endhost_cpu5
   + 0 m_access_time_endhost_cpu6 + 0 ns_sketch_endhost_cpu6
   + 0 ns_dpdk_endhost_cpu6 + 0 ns_endhost_cpu6
   + 0 m_access_time_endhost_cpu7 + 0 ns_sketch_endhost_cpu7
   + 0 ns_dpdk_endhost_cpu7 + 0 ns_endhost_cpu7
   + 0 m_access_time_endhost_cpu8 + 0 ns_sketch_endhost_cpu8
   + 0 ns_dpdk_endhost_cpu8 + 0 ns_endhost_cpu8
   + 0 m_access_time_endhost_cpu9 + 0 ns_sketch_endhost_cpu9
   + 0 ns_dpdk_endhost_cpu9 + 0 ns_endhost_cpu9
   + 0 m_access_time_endhost_cpu10 + 0 ns_sketch_endhost_cpu10
   + 0 ns_dpdk_endhost_cpu10 + 0 ns_endhost_cpu10
   + 0 m_access_time_endhost_cpu11 + 0 ns_sketch_endhost_cpu11
   + 0 ns_dpdk_endhost_cpu11 + 0 ns_endhost_cpu11
   + 0 m_access_time_endhost_cpu12 + 0 ns_sketch_endhost_cpu12
   + 0 ns_dpdk_endhost_cpu12 + 0 ns_endhost_cpu12
   + 0 m_access_time_endhost_cpu13 + 0 ns_sketch_endhost_cpu13
   + 0 ns_dpdk_endhost_cpu13 + 0 ns_endhost_cpu13
   + 0 m_access_time_endhost_cpu14 + 0 ns_sketch_endhost_cpu14
   + 0 ns_dpdk_endhost_cpu14 + 0 ns_endhost_cpu14
   + 0 m_access_time_endhost_cpu15 + 0 ns_sketch_endhost_cpu15
   + 0 ns_dpdk_endhost_cpu15 + 0 ns_endhost_cpu15
   + 0 m_access_time_endhost_cpu16 + 0 ns_sketch_endhost_cpu16
   + 0 ns_dpdk_endhost_cpu16 + 0 ns_endhost_cpu16
   + 0 m_access_time_endhost_cpu17 + 0 ns_sketch_endhost_cpu17
   + 0 ns_dpdk_endhost_cpu17 + 0 ns_endhost_cpu17
   + 0 m_access_time_endhost_cpu18 + 0 ns_sketch_endhost_cpu18
   + 0 ns_dpdk_endhost_cpu18 + 0 ns_endhost_cpu18
   + 0 m_access_time_endhost_cpu19 + 0 ns_sketch_endhost_cpu19
   + 0 ns_dpdk_endhost_cpu19 + 0 ns_endhost_cpu19
   + 0 m_access_time_endhost_cpu20 + 0 ns_sketch_endhost_cpu20
   + 0 ns_dpdk_endhost_cpu20 + 0 ns_endhost_cpu20
   + 0 m_access_time_endhost_cpu21 + 0 ns_sketch_endhost_cpu21
   + 0 ns_dpdk_endhost_cpu21 + 0 ns_endhost_cpu21
   + 0 m_access_time_endhost_cpu22 + 0 ns_sketch_endhost_cpu22
   + 0 ns_dpdk_endhost_cpu22 + 0 ns_endhost_cpu22
   + 0 m_access_time_endhost_cpu23 + 0 ns_sketch_endhost_cpu23
   + 0 ns_dpdk_endhost_cpu23 + 0 ns_endhost_cpu23
   + 0 m_access_time_endhost_cpu24 + 0 ns_sketch_endhost_cpu24
   + 0 ns_dpdk_endhost_cpu24 + 0 ns_endhost_cpu24
   + 0 m_access_time_endhost_cpu25 + 0 ns_sketch_endhost_cpu25
   + 0 ns_dpdk_endhost_cpu25 + 0 ns_endhost_cpu25
   + 0 m_access_time_endhost_cpu26 + 0 ns_sketch_endhost_cpu26
   + 0 ns_dpdk_endhost_cpu26 + 0 ns_endhost_cpu26
   + 0 m_access_time_endhost_cpu27 + 0 ns_sketch_endhost_cpu27
   + 0 ns_dpdk_endhost_cpu27 + 0 ns_endhost_cpu27
   + 0 m_access_time_endhost_cpu28 + 0 ns_sketch_endhost_cpu28
   + 0 ns_dpdk_endhost_cpu28 + 0 ns_endhost_cpu28
   + 0 m_access_time_endhost_cpu29 + 0 ns_sketch_endhost_cpu29
   + 0 ns_dpdk_endhost_cpu29 + 0 ns_endhost_cpu29
   + 0 m_access_time_endhost_cpu30 + 0 ns_sketch_endhost_cpu30
   + 0 ns_dpdk_endhost_cpu30 + 0 ns_endhost_cpu30
   + 0 m_access_time_endhost_cpu31 + 0 ns_sketch_endhost_cpu31
   + 0 ns_dpdk_endhost_cpu31 + 0 ns_endhost_cpu31
   + 0 m_access_time_endhost_cpu32 + 0 ns_sketch_endhost_cpu32
   + 0 ns_dpdk_endhost_cpu32 + 0 ns_endhost_cpu32 + ns
  res: Priority=5 Weight=1 AbsTol=0 RelTol=0.05
   res_overall
Subject To
 cov_0_0: frac[15,0] + frac[26,0] + frac[33,0] + frac[35,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.8
 cov_0_1: frac[15,1] + frac[26,1] + frac[33,1] + frac[35,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.8
 cov_0_2: frac[15,2] + frac[26,2] + frac[33,2] + frac[35,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.8
 cov_0_3: frac[15,3] + frac[26,3] + frac[33,3] + frac[35,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.8
 cov_1_0: frac[7,0] + frac[28,0] + frac[32,0] + frac[35,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.8
 cov_1_1: frac[7,1] + frac[28,1] + frac[32,1] + frac[35,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.8
 cov_1_2: frac[7,2] + frac[28,2] + frac[32,2] + frac[35,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.8
 cov_1_3: frac[7,3] + frac[28,3] + frac[32,3] + frac[35,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.8
 cov_2_8: frac[27,8] + frac[30,8] + frac[35,8] >= 0.9
 cov_2_9: frac[27,9] + frac[30,9] + frac[35,9] >= 0.9
 cov_2_10: frac[27,10] + frac[30,10] + frac[35,10] >= 0.9
 cov_2_11: frac[27,11] + frac[30,11] + frac[35,11] >= 0.9
 cov_2_12: frac[27,12] + frac[30,12] + frac[35,12] >= 0.9
 cov_3_4: frac[15,4] + frac[31,4] + frac[33,4] + frac[35,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.7
 cov_3_5: frac[15,5] + frac[31,5] + frac[33,5] + frac[35,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.7
 cov_3_6: frac[15,6] + frac[31,6] + frac[33,6] + frac[35,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.7
 cov_3_7: frac[15,7] + frac[31,7] + frac[33,7] + frac[35,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.7
 cov_4_8: frac[17,8] + frac[27,8] + frac[34,8] + frac[35,8] + frac[37,8]
   >= 0.9
 cov_4_9: frac[17,9] + frac[27,9] + frac[34,9] + frac[35,9] + frac[37,9]
   >= 0.9
 cov_4_10: frac[17,10] + frac[27,10] + frac[34,10] + frac[35,10]
   + frac[37,10] >= 0.9
 cov_4_11: frac[17,11] + frac[27,11] + frac[34,11] + frac[35,11]
   + frac[37,11] >= 0.9
 cov_4_12: frac[17,12] + frac[27,12] + frac[34,12] + frac[35,12]
   + frac[37,12] >= 0.9
 cov_5_0: frac[2,0] + frac[6,0] + frac[32,0] >= 0.7
 cov_5_1: frac[2,1] + frac[6,1] + frac[32,1] >= 0.7
 cov_5_2: frac[2,2] + frac[6,2] + frac[32,2] >= 0.7
 cov_5_3: frac[2,3] + frac[6,3] + frac[32,3] >= 0.7
 cov_6_4: frac[3,4] + frac[15,4] + frac[32,4] + frac[33,4] + frac[36,4]
   >= 0.8
 cov_6_5: frac[3,5] + frac[15,5] + frac[32,5] + frac[33,5] + frac[36,5]
   >= 0.8
 cov_6_6: frac[3,6] + frac[15,6] + frac[32,6] + frac[33,6] + frac[36,6]
   >= 0.8
 cov_6_7: frac[3,7] + frac[15,7] + frac[32,7] + frac[33,7] + frac[36,7]
   >= 0.8
 cov_7_8: frac[8,8] + frac[25,8] + frac[33,8] + frac[35,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.9
 cov_7_9: frac[8,9] + frac[25,9] + frac[33,9] + frac[35,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.9
 cov_7_10: frac[8,10] + frac[25,10] + frac[33,10] + frac[35,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.9
 cov_7_11: frac[8,11] + frac[25,11] + frac[33,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.9
 cov_7_12: frac[8,12] + frac[25,12] + frac[33,12] + frac[35,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.9
 cov_8_4: frac[14,4] + frac[20,4] + frac[33,4] + frac[34,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.8
 cov_8_5: frac[14,5] + frac[20,5] + frac[33,5] + frac[34,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.8
 cov_8_6: frac[14,6] + frac[20,6] + frac[33,6] + frac[34,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.8
 cov_8_7: frac[14,7] + frac[20,7] + frac[33,7] + frac[34,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.8
 cov_9_8: frac[20,8] + frac[23,8] + frac[34,8] >= 0.8
 cov_9_9: frac[20,9] + frac[23,9] + frac[34,9] >= 0.8
 cov_9_10: frac[20,10] + frac[23,10] + frac[34,10] >= 0.8
 cov_9_11: frac[20,11] + frac[23,11] + frac[34,11] >= 0.8
 cov_9_12: frac[20,12] + frac[23,12] + frac[34,12] >= 0.8
 cov_10_8: frac[13,8] + frac[30,8] + frac[33,8] + frac[35,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.8
 cov_10_9: frac[13,9] + frac[30,9] + frac[33,9] + frac[35,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.8
 cov_10_10: frac[13,10] + frac[30,10] + frac[33,10] + frac[35,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.8
 cov_10_11: frac[13,11] + frac[30,11] + frac[33,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.8
 cov_10_12: frac[13,12] + frac[30,12] + frac[33,12] + frac[35,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.8
 cov_11_8: frac[30,8] + frac[31,8] + frac[35,8] >= 0.7
 cov_11_9: frac[30,9] + frac[31,9] + frac[35,9] >= 0.7
 cov_11_10: frac[30,10] + frac[31,10] + frac[35,10] >= 0.7
 cov_11_11: frac[30,11] + frac[31,11] + frac[35,11] >= 0.7
 cov_11_12: frac[30,12] + frac[31,12] + frac[35,12] >= 0.7
 cov_12_8: frac[6,8] + frac[17,8] + frac[32,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.8
 cov_12_9: frac[6,9] + frac[17,9] + frac[32,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.8
 cov_12_10: frac[6,10] + frac[17,10] + frac[32,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.8
 cov_12_11: frac[6,11] + frac[17,11] + frac[32,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.8
 cov_12_12: frac[6,12] + frac[17,12] + frac[32,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.8
 cov_13_4: frac[18,4] + frac[26,4] + frac[34,4] + frac[35,4] + frac[37,4]
   >= 0.7
 cov_13_5: frac[18,5] + frac[26,5] + frac[34,5] + frac[35,5] + frac[37,5]
   >= 0.7
 cov_13_6: frac[18,6] + frac[26,6] + frac[34,6] + frac[35,6] + frac[37,6]
   >= 0.7
 cov_13_7: frac[18,7] + frac[26,7] + frac[34,7] + frac[35,7] + frac[37,7]
   >= 0.7
 cov_14_4: frac[2,4] + frac[3,4] + frac[32,4] >= 1
 cov_14_5: frac[2,5] + frac[3,5] + frac[32,5] >= 1
 cov_14_6: frac[2,6] + frac[3,6] + frac[32,6] >= 1
 cov_14_7: frac[2,7] + frac[3,7] + frac[32,7] >= 1
 cov_15_8: frac[5,8] + frac[16,8] + frac[32,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 1
 cov_15_9: frac[5,9] + frac[16,9] + frac[32,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 1
 cov_15_10: frac[5,10] + frac[16,10] + frac[32,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 1
 cov_15_11: frac[5,11] + frac[16,11] + frac[32,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 1
 cov_15_12: frac[5,12] + frac[16,12] + frac[32,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 1
 cov_16_4: frac[4,4] + frac[5,4] + frac[32,4] >= 0.8
 cov_16_5: frac[4,5] + frac[5,5] + frac[32,5] >= 0.8
 cov_16_6: frac[4,6] + frac[5,6] + frac[32,6] >= 0.8
 cov_16_7: frac[4,7] + frac[5,7] + frac[32,7] >= 0.8
 cov_17_0: frac[13,0] + frac[26,0] + frac[33,0] + frac[35,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.9
 cov_17_1: frac[13,1] + frac[26,1] + frac[33,1] + frac[35,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.9
 cov_17_2: frac[13,2] + frac[26,2] + frac[33,2] + frac[35,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.9
 cov_17_3: frac[13,3] + frac[26,3] + frac[33,3] + frac[35,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.9
 cov_18_8: frac[1,8] + frac[12,8] + frac[32,8] + frac[33,8] + frac[36,8]
   >= 0.8
 cov_18_9: frac[1,9] + frac[12,9] + frac[32,9] + frac[33,9] + frac[36,9]
   >= 0.8
 cov_18_10: frac[1,10] + frac[12,10] + frac[32,10] + frac[33,10]
   + frac[36,10] >= 0.8
 cov_18_11: frac[1,11] + frac[12,11] + frac[32,11] + frac[33,11]
   + frac[36,11] >= 0.8
 cov_18_12: frac[1,12] + frac[12,12] + frac[32,12] + frac[33,12]
   + frac[36,12] >= 0.8
 cov_19_4: frac[24,4] + frac[27,4] + frac[35,4] >= 1
 cov_19_5: frac[24,5] + frac[27,5] + frac[35,5] >= 1
 cov_19_6: frac[24,6] + frac[27,6] + frac[35,6] >= 1
 cov_19_7: frac[24,7] + frac[27,7] + frac[35,7] >= 1
 cov_20_4: frac[4,4] + frac[5,4] + frac[32,4] >= 1
 cov_20_5: frac[4,5] + frac[5,5] + frac[32,5] >= 1
 cov_20_6: frac[4,6] + frac[5,6] + frac[32,6] >= 1
 cov_20_7: frac[4,7] + frac[5,7] + frac[32,7] >= 1
 cov_21_0: frac[22,0] + frac[25,0] + frac[34,0] + frac[35,0] + frac[37,0]
   >= 0.9
 cov_21_1: frac[22,1] + frac[25,1] + frac[34,1] + frac[35,1] + frac[37,1]
   >= 0.9
 cov_21_2: frac[22,2] + frac[25,2] + frac[34,2] + frac[35,2] + frac[37,2]
   >= 0.9
 cov_21_3: frac[22,3] + frac[25,3] + frac[34,3] + frac[35,3] + frac[37,3]
   >= 0.9
 cov_22_8: frac[1,8] + frac[31,8] + frac[32,8] + frac[35,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.8
 cov_22_9: frac[1,9] + frac[31,9] + frac[32,9] + frac[35,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.8
 cov_22_10: frac[1,10] + frac[31,10] + frac[32,10] + frac[35,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.8
 cov_22_11: frac[1,11] + frac[31,11] + frac[32,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.8
 cov_22_12: frac[1,12] + frac[31,12] + frac[32,12] + frac[35,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.8
 cov_23_4: frac[8,4] + frac[22,4] + frac[33,4] + frac[34,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.9
 cov_23_5: frac[8,5] + frac[22,5] + frac[33,5] + frac[34,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.9
 cov_23_6: frac[8,6] + frac[22,6] + frac[33,6] + frac[34,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.9
 cov_23_7: frac[8,7] + frac[22,7] + frac[33,7] + frac[34,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.9
 cov_24_8: frac[4,8] + frac[22,8] + frac[32,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.8
 cov_24_9: frac[4,9] + frac[22,9] + frac[32,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.8
 cov_24_10: frac[4,10] + frac[22,10] + frac[32,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.8
 cov_24_11: frac[4,11] + frac[22,11] + frac[32,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.8
 cov_24_12: frac[4,12] + frac[22,12] + frac[32,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.8
 cov_25_0: frac[3,0] + frac[28,0] + frac[32,0] + frac[35,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.9
 cov_25_1: frac[3,1] + frac[28,1] + frac[32,1] + frac[35,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.9
 cov_25_2: frac[3,2] + frac[28,2] + frac[32,2] + frac[35,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.9
 cov_25_3: frac[3,3] + frac[28,3] + frac[32,3] + frac[35,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.9
 cov_26_8: frac[11,8] + frac[12,8] + frac[33,8] >= 0.9
 cov_26_9: frac[11,9] + frac[12,9] + frac[33,9] >= 0.9
 cov_26_10: frac[11,10] + frac[12,10] + frac[33,10] >= 0.9
 cov_26_11: frac[11,11] + frac[12,11] + frac[33,11] >= 0.9
 cov_26_12: frac[11,12] + frac[12,12] + frac[33,12] >= 0.9
 cov_27_8: frac[2,8] + frac[19,8] + frac[32,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.8
 cov_27_9: frac[2,9] + frac[19,9] + frac[32,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.8
 cov_27_10: frac[2,10] + frac[19,10] + frac[32,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.8
 cov_27_11: frac[2,11] + frac[19,11] + frac[32,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.8
 cov_27_12: frac[2,12] + frac[19,12] + frac[32,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.8
 cov_28_8: frac[7,8] + frac[20,8] + frac[32,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 1
 cov_28_9: frac[7,9] + frac[20,9] + frac[32,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 1
 cov_28_10: frac[7,10] + frac[20,10] + frac[32,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 1
 cov_28_11: frac[7,11] + frac[20,11] + frac[32,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 1
 cov_28_12: frac[7,12] + frac[20,12] + frac[32,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 1
 cov_29_8: frac[7,8] + frac[11,8] + frac[32,8] + frac[33,8] + frac[36,8]
   >= 0.8
 cov_29_9: frac[7,9] + frac[11,9] + frac[32,9] + frac[33,9] + frac[36,9]
   >= 0.8
 cov_29_10: frac[7,10] + frac[11,10] + frac[32,10] + frac[33,10]
   + frac[36,10] >= 0.8
 cov_29_11: frac[7,11] + frac[11,11] + frac[32,11] + frac[33,11]
   + frac[36,11] >= 0.8
 cov_29_12: frac[7,12] + frac[11,12] + frac[32,12] + frac[33,12]
   + frac[36,12] >= 0.8
 cov_30_0: frac[14,0] + frac[17,0] + frac[33,0] + frac[34,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.8
 cov_30_1: frac[14,1] + frac[17,1] + frac[33,1] + frac[34,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.8
 cov_30_2: frac[14,2] + frac[17,2] + frac[33,2] + frac[34,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.8
 cov_30_3: frac[14,3] + frac[17,3] + frac[33,3] + frac[34,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.8
 cov_31_0: frac[1,0] + frac[26,0] + frac[32,0] + frac[35,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 1
 cov_31_1: frac[1,1] + frac[26,1] + frac[32,1] + frac[35,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 1
 cov_31_2: frac[1,2] + frac[26,2] + frac[32,2] + frac[35,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 1
 cov_31_3: frac[1,3] + frac[26,3] + frac[32,3] + frac[35,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 1
 cov_32_8: frac[15,8] + frac[31,8] + frac[33,8] + frac[35,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.8
 cov_32_9: frac[15,9] + frac[31,9] + frac[33,9] + frac[35,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.8
 cov_32_10: frac[15,10] + frac[31,10] + frac[33,10] + frac[35,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.8
 cov_32_11: frac[15,11] + frac[31,11] + frac[33,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.8
 cov_32_12: frac[15,12] + frac[31,12] + frac[33,12] + frac[35,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.8
 cov_33_4: frac[1,4] + frac[13,4] + frac[32,4] + frac[33,4] + frac[36,4]
   >= 1
 cov_33_5: frac[1,5] + frac[13,5] + frac[32,5] + frac[33,5] + frac[36,5]
   >= 1
 cov_33_6: frac[1,6] + frac[13,6] + frac[32,6] + frac[33,6] + frac[36,6]
   >= 1
 cov_33_7: frac[1,7] + frac[13,7] + frac[32,7] + frac[33,7] + frac[36,7]
   >= 1
 cov_34_8: frac[15,8] + frac[17,8] + frac[33,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 1
 cov_34_9: frac[15,9] + frac[17,9] + frac[33,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 1
 cov_34_10: frac[15,10] + frac[17,10] + frac[33,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 1
 cov_34_11: frac[15,11] + frac[17,11] + frac[33,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 1
 cov_34_12: frac[15,12] + frac[17,12] + frac[33,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 1
 cov_35_0: frac[16,0] + frac[31,0] + frac[34,0] + frac[35,0] + frac[37,0]
   >= 0.7
 cov_35_1: frac[16,1] + frac[31,1] + frac[34,1] + frac[35,1] + frac[37,1]
   >= 0.7
 cov_35_2: frac[16,2] + frac[31,2] + frac[34,2] + frac[35,2] + frac[37,2]
   >= 0.7
 cov_35_3: frac[16,3] + frac[31,3] + frac[34,3] + frac[35,3] + frac[37,3]
   >= 0.7
 cov_36_8: frac[21,8] + frac[25,8] + frac[34,8] + frac[35,8] + frac[37,8]
   >= 1
 cov_36_9: frac[21,9] + frac[25,9] + frac[34,9] + frac[35,9] + frac[37,9]
   >= 1
 cov_36_10: frac[21,10] + frac[25,10] + frac[34,10] + frac[35,10]
   + frac[37,10] >= 1
 cov_36_11: frac[21,11] + frac[25,11] + frac[34,11] + frac[35,11]
   + frac[37,11] >= 1
 cov_36_12: frac[21,12] + frac[25,12] + frac[34,12] + frac[35,12]
   + frac[37,12] >= 1
 cov_37_8: frac[7,8] + frac[15,8] + frac[32,8] + frac[33,8] + frac[36,8]
   >= 0.8
 cov_37_9: frac[7,9] + frac[15,9] + frac[32,9] + frac[33,9] + frac[36,9]
   >= 0.8
 cov_37_10: frac[7,10] + frac[15,10] + frac[32,10] + frac[33,10]
   + frac[36,10] >= 0.8
 cov_37_11: frac[7,11] + frac[15,11] + frac[32,11] + frac[33,11]
   + frac[36,11] >= 0.8
 cov_37_12: frac[7,12] + frac[15,12] + frac[32,12] + frac[33,12]
   + frac[36,12] >= 0.8
 cov_38_4: frac[7,4] + frac[28,4] + frac[32,4] + frac[35,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 1
 cov_38_5: frac[7,5] + frac[28,5] + frac[32,5] + frac[35,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 1
 cov_38_6: frac[7,6] + frac[28,6] + frac[32,6] + frac[35,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 1
 cov_38_7: frac[7,7] + frac[28,7] + frac[32,7] + frac[35,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 1
 cov_39_8: frac[12,8] + frac[24,8] + frac[33,8] + frac[35,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.8
 cov_39_9: frac[12,9] + frac[24,9] + frac[33,9] + frac[35,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.8
 cov_39_10: frac[12,10] + frac[24,10] + frac[33,10] + frac[35,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.8
 cov_39_11: frac[12,11] + frac[24,11] + frac[33,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.8
 cov_39_12: frac[12,12] + frac[24,12] + frac[33,12] + frac[35,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.8
 cov_40_4: frac[4,4] + frac[31,4] + frac[32,4] + frac[35,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.9
 cov_40_5: frac[4,5] + frac[31,5] + frac[32,5] + frac[35,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.9
 cov_40_6: frac[4,6] + frac[31,6] + frac[32,6] + frac[35,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.9
 cov_40_7: frac[4,7] + frac[31,7] + frac[32,7] + frac[35,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.9
 cov_41_4: frac[11,4] + frac[26,4] + frac[33,4] + frac[35,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.8
 cov_41_5: frac[11,5] + frac[26,5] + frac[33,5] + frac[35,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.8
 cov_41_6: frac[11,6] + frac[26,6] + frac[33,6] + frac[35,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.8
 cov_41_7: frac[11,7] + frac[26,7] + frac[33,7] + frac[35,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.8
 cov_42_0: frac[8,0] + frac[27,0] + frac[33,0] + frac[35,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.9
 cov_42_1: frac[8,1] + frac[27,1] + frac[33,1] + frac[35,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.9
 cov_42_2: frac[8,2] + frac[27,2] + frac[33,2] + frac[35,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.9
 cov_42_3: frac[8,3] + frac[27,3] + frac[33,3] + frac[35,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.9
 cov_43_8: frac[10,8] + frac[12,8] + frac[33,8] >= 0.7
 cov_43_9: frac[10,9] + frac[12,9] + frac[33,9] >= 0.7
 cov_43_10: frac[10,10] + frac[12,10] + frac[33,10] >= 0.7
 cov_43_11: frac[10,11] + frac[12,11] + frac[33,11] >= 0.7
 cov_43_12: frac[10,12] + frac[12,12] + frac[33,12] >= 0.7
 cov_44_8: frac[6,8] + frac[15,8] + frac[32,8] + frac[33,8] + frac[36,8]
   >= 0.8
 cov_44_9: frac[6,9] + frac[15,9] + frac[32,9] + frac[33,9] + frac[36,9]
   >= 0.8
 cov_44_10: frac[6,10] + frac[15,10] + frac[32,10] + frac[33,10]
   + frac[36,10] >= 0.8
 cov_44_11: frac[6,11] + frac[15,11] + frac[32,11] + frac[33,11]
   + frac[36,11] >= 0.8
 cov_44_12: frac[6,12] + frac[15,12] + frac[32,12] + frac[33,12]
   + frac[36,12] >= 0.8
 cov_45_0: frac[1,0] + frac[3,0] + frac[32,0] >= 1
 cov_45_1: frac[1,1] + frac[3,1] + frac[32,1] >= 1
 cov_45_2: frac[1,2] + frac[3,2] + frac[32,2] >= 1
 cov_45_3: frac[1,3] + frac[3,3] + frac[32,3] >= 1
 cov_46_8: frac[15,8] >= 1
 cov_46_9: frac[15,9] >= 1
 cov_46_10: frac[15,10] >= 1
 cov_46_11: frac[15,11] >= 1
 cov_46_12: frac[15,12] >= 1
 cov_47_4: frac[17,4] + frac[21,4] + frac[34,4] >= 0.9
 cov_47_5: frac[17,5] + frac[21,5] + frac[34,5] >= 0.9
 cov_47_6: frac[17,6] + frac[21,6] + frac[34,6] >= 0.9
 cov_47_7: frac[17,7] + frac[21,7] + frac[34,7] >= 0.9
 cov_48_8: frac[2,8] + frac[27,8] + frac[32,8] + frac[35,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.9
 cov_48_9: frac[2,9] + frac[27,9] + frac[32,9] + frac[35,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.9
 cov_48_10: frac[2,10] + frac[27,10] + frac[32,10] + frac[35,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.9
 cov_48_11: frac[2,11] + frac[27,11] + frac[32,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.9
 cov_48_12: frac[2,12] + frac[27,12] + frac[32,12] + frac[35,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.9
 cov_49_0: frac[2,0] + frac[9,0] + frac[32,0] + frac[33,0] + frac[36,0]
   >= 0.7
 cov_49_1: frac[2,1] + frac[9,1] + frac[32,1] + frac[33,1] + frac[36,1]
   >= 0.7
 cov_49_2: frac[2,2] + frac[9,2] + frac[32,2] + frac[33,2] + frac[36,2]
   >= 0.7
 cov_49_3: frac[2,3] + frac[9,3] + frac[32,3] + frac[33,3] + frac[36,3]
   >= 0.7
 cov_50_0: frac[1,0] + frac[28,0] + frac[32,0] + frac[35,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.7
 cov_50_1: frac[1,1] + frac[28,1] + frac[32,1] + frac[35,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.7
 cov_50_2: frac[1,2] + frac[28,2] + frac[32,2] + frac[35,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.7
 cov_50_3: frac[1,3] + frac[28,3] + frac[32,3] + frac[35,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.7
 cov_51_0: frac[7,0] + frac[11,0] + frac[32,0] + frac[33,0] + frac[36,0]
   >= 0.7
 cov_51_1: frac[7,1] + frac[11,1] + frac[32,1] + frac[33,1] + frac[36,1]
   >= 0.7
 cov_51_2: frac[7,2] + frac[11,2] + frac[32,2] + frac[33,2] + frac[36,2]
   >= 0.7
 cov_51_3: frac[7,3] + frac[11,3] + frac[32,3] + frac[33,3] + frac[36,3]
   >= 0.7
 cov_52_4: frac[3,4] + frac[9,4] + frac[32,4] + frac[33,4] + frac[36,4]
   >= 0.9
 cov_52_5: frac[3,5] + frac[9,5] + frac[32,5] + frac[33,5] + frac[36,5]
   >= 0.9
 cov_52_6: frac[3,6] + frac[9,6] + frac[32,6] + frac[33,6] + frac[36,6]
   >= 0.9
 cov_52_7: frac[3,7] + frac[9,7] + frac[32,7] + frac[33,7] + frac[36,7]
   >= 0.9
 cov_53_8: frac[10,8] + frac[13,8] + frac[33,8] >= 0.7
 cov_53_9: frac[10,9] + frac[13,9] + frac[33,9] >= 0.7
 cov_53_10: frac[10,10] + frac[13,10] + frac[33,10] >= 0.7
 cov_53_11: frac[10,11] + frac[13,11] + frac[33,11] >= 0.7
 cov_53_12: frac[10,12] + frac[13,12] + frac[33,12] >= 0.7
 cov_54_4: frac[0,4] + frac[13,4] + frac[32,4] + frac[33,4] + frac[36,4]
   >= 0.9
 cov_54_5: frac[0,5] + frac[13,5] + frac[32,5] + frac[33,5] + frac[36,5]
   >= 0.9
 cov_54_6: frac[0,6] + frac[13,6] + frac[32,6] + frac[33,6] + frac[36,6]
   >= 0.9
 cov_54_7: frac[0,7] + frac[13,7] + frac[32,7] + frac[33,7] + frac[36,7]
   >= 0.9
 cov_55_4: frac[15,4] + frac[23,4] + frac[33,4] + frac[34,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 1
 cov_55_5: frac[15,5] + frac[23,5] + frac[33,5] + frac[34,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 1
 cov_55_6: frac[15,6] + frac[23,6] + frac[33,6] + frac[34,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 1
 cov_55_7: frac[15,7] + frac[23,7] + frac[33,7] + frac[34,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 1
 cov_56_0: frac[15,0] + frac[26,0] + frac[33,0] + frac[35,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.7
 cov_56_1: frac[15,1] + frac[26,1] + frac[33,1] + frac[35,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.7
 cov_56_2: frac[15,2] + frac[26,2] + frac[33,2] + frac[35,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.7
 cov_56_3: frac[15,3] + frac[26,3] + frac[33,3] + frac[35,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.7
 cov_57_4: frac[3,4] + frac[13,4] + frac[32,4] + frac[33,4] + frac[36,4]
   >= 0.9
 cov_57_5: frac[3,5] + frac[13,5] + frac[32,5] + frac[33,5] + frac[36,5]
   >= 0.9
 cov_57_6: frac[3,6] + frac[13,6] + frac[32,6] + frac[33,6] + frac[36,6]
   >= 0.9
 cov_57_7: frac[3,7] + frac[13,7] + frac[32,7] + frac[33,7] + frac[36,7]
   >= 0.9
 cov_58_8: frac[8,8] + frac[29,8] + frac[33,8] + frac[35,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.7
 cov_58_9: frac[8,9] + frac[29,9] + frac[33,9] + frac[35,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.7
 cov_58_10: frac[8,10] + frac[29,10] + frac[33,10] + frac[35,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.7
 cov_58_11: frac[8,11] + frac[29,11] + frac[33,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.7
 cov_58_12: frac[8,12] + frac[29,12] + frac[33,12] + frac[35,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.7
 cov_59_0: frac[0,0] + frac[2,0] + frac[32,0] >= 1
 cov_59_1: frac[0,1] + frac[2,1] + frac[32,1] >= 1
 cov_59_2: frac[0,2] + frac[2,2] + frac[32,2] >= 1
 cov_59_3: frac[0,3] + frac[2,3] + frac[32,3] >= 1
 cov_60_0: frac[19,0] + frac[31,0] + frac[34,0] + frac[35,0] + frac[37,0]
   >= 0.9
 cov_60_1: frac[19,1] + frac[31,1] + frac[34,1] + frac[35,1] + frac[37,1]
   >= 0.9
 cov_60_2: frac[19,2] + frac[31,2] + frac[34,2] + frac[35,2] + frac[37,2]
   >= 0.9
 cov_60_3: frac[19,3] + frac[31,3] + frac[34,3] + frac[35,3] + frac[37,3]
   >= 0.9
 cov_61_0: frac[10,0] + frac[24,0] + frac[33,0] + frac[35,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.9
 cov_61_1: frac[10,1] + frac[24,1] + frac[33,1] + frac[35,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.9
 cov_61_2: frac[10,2] + frac[24,2] + frac[33,2] + frac[35,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.9
 cov_61_3: frac[10,3] + frac[24,3] + frac[33,3] + frac[35,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.9
 cov_62_8: frac[29,8] + frac[30,8] + frac[35,8] >= 0.7
 cov_62_9: frac[29,9] + frac[30,9] + frac[35,9] >= 0.7
 cov_62_10: frac[29,10] + frac[30,10] + frac[35,10] >= 0.7
 cov_62_11: frac[29,11] + frac[30,11] + frac[35,11] >= 0.7
 cov_62_12: frac[29,12] + frac[30,12] + frac[35,12] >= 0.7
 cov_63_4: frac[3,4] + frac[29,4] + frac[32,4] + frac[35,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.9
 cov_63_5: frac[3,5] + frac[29,5] + frac[32,5] + frac[35,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.9
 cov_63_6: frac[3,6] + frac[29,6] + frac[32,6] + frac[35,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.9
 cov_63_7: frac[3,7] + frac[29,7] + frac[32,7] + frac[35,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.9
 cov_64_8: frac[12,8] + frac[16,8] + frac[33,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.7
 cov_64_9: frac[12,9] + frac[16,9] + frac[33,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.7
 cov_64_10: frac[12,10] + frac[16,10] + frac[33,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.7
 cov_64_11: frac[12,11] + frac[16,11] + frac[33,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.7
 cov_64_12: frac[12,12] + frac[16,12] + frac[33,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.7
 cov_65_0: frac[14,0] + frac[23,0] + frac[33,0] + frac[34,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.9
 cov_65_1: frac[14,1] + frac[23,1] + frac[33,1] + frac[34,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.9
 cov_65_2: frac[14,2] + frac[23,2] + frac[33,2] + frac[34,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.9
 cov_65_3: frac[14,3] + frac[23,3] + frac[33,3] + frac[34,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.9
 cov_66_8: frac[3,8] + frac[26,8] + frac[32,8] + frac[35,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.9
 cov_66_9: frac[3,9] + frac[26,9] + frac[32,9] + frac[35,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.9
 cov_66_10: frac[3,10] + frac[26,10] + frac[32,10] + frac[35,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.9
 cov_66_11: frac[3,11] + frac[26,11] + frac[32,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.9
 cov_66_12: frac[3,12] + frac[26,12] + frac[32,12] + frac[35,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.9
 cov_67_0: frac[25,0] + frac[28,0] + frac[35,0] >= 0.8
 cov_67_1: frac[25,1] + frac[28,1] + frac[35,1] >= 0.8
 cov_67_2: frac[25,2] + frac[28,2] + frac[35,2] >= 0.8
 cov_67_3: frac[25,3] + frac[28,3] + frac[35,3] >= 0.8
 cov_68_4: frac[21,4] + frac[29,4] + frac[34,4] + frac[35,4] + frac[37,4]
   >= 0.8
 cov_68_5: frac[21,5] + frac[29,5] + frac[34,5] + frac[35,5] + frac[37,5]
   >= 0.8
 cov_68_6: frac[21,6] + frac[29,6] + frac[34,6] + frac[35,6] + frac[37,6]
   >= 0.8
 cov_68_7: frac[21,7] + frac[29,7] + frac[34,7] + frac[35,7] + frac[37,7]
   >= 0.8
 cov_69_4: frac[14,4] + frac[19,4] + frac[33,4] + frac[34,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.9
 cov_69_5: frac[14,5] + frac[19,5] + frac[33,5] + frac[34,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.9
 cov_69_6: frac[14,6] + frac[19,6] + frac[33,6] + frac[34,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.9
 cov_69_7: frac[14,7] + frac[19,7] + frac[33,7] + frac[34,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.9
 cov_70_4: frac[7,4] + frac[22,4] + frac[32,4] + frac[34,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 1
 cov_70_5: frac[7,5] + frac[22,5] + frac[32,5] + frac[34,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 1
 cov_70_6: frac[7,6] + frac[22,6] + frac[32,6] + frac[34,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 1
 cov_70_7: frac[7,7] + frac[22,7] + frac[32,7] + frac[34,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 1
 cov_71_8: frac[0,8] + frac[21,8] + frac[32,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.7
 cov_71_9: frac[0,9] + frac[21,9] + frac[32,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.7
 cov_71_10: frac[0,10] + frac[21,10] + frac[32,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.7
 cov_71_11: frac[0,11] + frac[21,11] + frac[32,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.7
 cov_71_12: frac[0,12] + frac[21,12] + frac[32,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.7
 cov_72_4: frac[18,4] + frac[21,4] + frac[34,4] >= 0.7
 cov_72_5: frac[18,5] + frac[21,5] + frac[34,5] >= 0.7
 cov_72_6: frac[18,6] + frac[21,6] + frac[34,6] >= 0.7
 cov_72_7: frac[18,7] + frac[21,7] + frac[34,7] >= 0.7
 cov_73_0: frac[19,0] + frac[22,0] + frac[34,0] >= 0.8
 cov_73_1: frac[19,1] + frac[22,1] + frac[34,1] >= 0.8
 cov_73_2: frac[19,2] + frac[22,2] + frac[34,2] >= 0.8
 cov_73_3: frac[19,3] + frac[22,3] + frac[34,3] >= 0.8
 cov_74_0: frac[11,0] + frac[26,0] + frac[33,0] + frac[35,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.7
 cov_74_1: frac[11,1] + frac[26,1] + frac[33,1] + frac[35,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.7
 cov_74_2: frac[11,2] + frac[26,2] + frac[33,2] + frac[35,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.7
 cov_74_3: frac[11,3] + frac[26,3] + frac[33,3] + frac[35,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.7
 cov_75_8: frac[0,8] + frac[13,8] + frac[32,8] + frac[33,8] + frac[36,8]
   >= 0.9
 cov_75_9: frac[0,9] + frac[13,9] + frac[32,9] + frac[33,9] + frac[36,9]
   >= 0.9
 cov_75_10: frac[0,10] + frac[13,10] + frac[32,10] + frac[33,10]
   + frac[36,10] >= 0.9
 cov_75_11: frac[0,11] + frac[13,11] + frac[32,11] + frac[33,11]
   + frac[36,11] >= 0.9
 cov_75_12: frac[0,12] + frac[13,12] + frac[32,12] + frac[33,12]
   + frac[36,12] >= 0.9
 cov_76_4: frac[2,4] + frac[25,4] + frac[32,4] + frac[35,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.7
 cov_76_5: frac[2,5] + frac[25,5] + frac[32,5] + frac[35,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.7
 cov_76_6: frac[2,6] + frac[25,6] + frac[32,6] + frac[35,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.7
 cov_76_7: frac[2,7] + frac[25,7] + frac[32,7] + frac[35,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.7
 cov_77_8: frac[5,8] + frac[30,8] + frac[32,8] + frac[35,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 1
 cov_77_9: frac[5,9] + frac[30,9] + frac[32,9] + frac[35,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 1
 cov_77_10: frac[5,10] + frac[30,10] + frac[32,10] + frac[35,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 1
 cov_77_11: frac[5,11] + frac[30,11] + frac[32,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 1
 cov_77_12: frac[5,12] + frac[30,12] + frac[32,12] + frac[35,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 1
 cov_78_4: frac[0,4] + frac[4,4] + frac[32,4] >= 0.8
 cov_78_5: frac[0,5] + frac[4,5] + frac[32,5] >= 0.8
 cov_78_6: frac[0,6] + frac[4,6] + frac[32,6] >= 0.8
 cov_78_7: frac[0,7] + frac[4,7] + frac[32,7] >= 0.8
 cov_79_0: frac[17,0] + frac[31,0] + frac[34,0] + frac[35,0] + frac[37,0]
   >= 0.8
 cov_79_1: frac[17,1] + frac[31,1] + frac[34,1] + frac[35,1] + frac[37,1]
   >= 0.8
 cov_79_2: frac[17,2] + frac[31,2] + frac[34,2] + frac[35,2] + frac[37,2]
   >= 0.8
 cov_79_3: frac[17,3] + frac[31,3] + frac[34,3] + frac[35,3] + frac[37,3]
   >= 0.8
 cov_80_0: frac[11,0] + frac[25,0] + frac[33,0] + frac[35,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.9
 cov_80_1: frac[11,1] + frac[25,1] + frac[33,1] + frac[35,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.9
 cov_80_2: frac[11,2] + frac[25,2] + frac[33,2] + frac[35,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.9
 cov_80_3: frac[11,3] + frac[25,3] + frac[33,3] + frac[35,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.9
 cov_81_4: frac[9,4] + frac[23,4] + frac[33,4] + frac[34,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 1
 cov_81_5: frac[9,5] + frac[23,5] + frac[33,5] + frac[34,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 1
 cov_81_6: frac[9,6] + frac[23,6] + frac[33,6] + frac[34,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 1
 cov_81_7: frac[9,7] + frac[23,7] + frac[33,7] + frac[34,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 1
 cov_82_0: frac[7,0] + frac[14,0] + frac[32,0] + frac[33,0] + frac[36,0]
   >= 0.8
 cov_82_1: frac[7,1] + frac[14,1] + frac[32,1] + frac[33,1] + frac[36,1]
   >= 0.8
 cov_82_2: frac[7,2] + frac[14,2] + frac[32,2] + frac[33,2] + frac[36,2]
   >= 0.8
 cov_82_3: frac[7,3] + frac[14,3] + frac[32,3] + frac[33,3] + frac[36,3]
   >= 0.8
 cov_83_4: frac[11,4] + frac[22,4] + frac[33,4] + frac[34,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 1
 cov_83_5: frac[11,5] + frac[22,5] + frac[33,5] + frac[34,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 1
 cov_83_6: frac[11,6] + frac[22,6] + frac[33,6] + frac[34,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 1
 cov_83_7: frac[11,7] + frac[22,7] + frac[33,7] + frac[34,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 1
 cov_84_4: frac[0,4] + frac[14,4] + frac[32,4] + frac[33,4] + frac[36,4]
   >= 0.9
 cov_84_5: frac[0,5] + frac[14,5] + frac[32,5] + frac[33,5] + frac[36,5]
   >= 0.9
 cov_84_6: frac[0,6] + frac[14,6] + frac[32,6] + frac[33,6] + frac[36,6]
   >= 0.9
 cov_84_7: frac[0,7] + frac[14,7] + frac[32,7] + frac[33,7] + frac[36,7]
   >= 0.9
 cov_85_8: frac[25,8] + frac[30,8] + frac[35,8] >= 0.9
 cov_85_9: frac[25,9] + frac[30,9] + frac[35,9] >= 0.9
 cov_85_10: frac[25,10] + frac[30,10] + frac[35,10] >= 0.9
 cov_85_11: frac[25,11] + frac[30,11] + frac[35,11] >= 0.9
 cov_85_12: frac[25,12] + frac[30,12] + frac[35,12] >= 0.9
 cov_86_4: frac[0,4] + frac[6,4] + frac[32,4] >= 0.9
 cov_86_5: frac[0,5] + frac[6,5] + frac[32,5] >= 0.9
 cov_86_6: frac[0,6] + frac[6,6] + frac[32,6] >= 0.9
 cov_86_7: frac[0,7] + frac[6,7] + frac[32,7] >= 0.9
 cov_87_8: frac[2,8] + frac[18,8] + frac[32,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.8
 cov_87_9: frac[2,9] + frac[18,9] + frac[32,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.8
 cov_87_10: frac[2,10] + frac[18,10] + frac[32,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.8
 cov_87_11: frac[2,11] + frac[18,11] + frac[32,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.8
 cov_87_12: frac[2,12] + frac[18,12] + frac[32,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.8
 cov_88_0: frac[10,0] + frac[16,0] + frac[33,0] + frac[34,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.9
 cov_88_1: frac[10,1] + frac[16,1] + frac[33,1] + frac[34,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.9
 cov_88_2: frac[10,2] + frac[16,2] + frac[33,2] + frac[34,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.9
 cov_88_3: frac[10,3] + frac[16,3] + frac[33,3] + frac[34,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.9
 cov_89_8: frac[5,8] + frac[26,8] + frac[32,8] + frac[35,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.7
 cov_89_9: frac[5,9] + frac[26,9] + frac[32,9] + frac[35,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.7
 cov_89_10: frac[5,10] + frac[26,10] + frac[32,10] + frac[35,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.7
 cov_89_11: frac[5,11] + frac[26,11] + frac[32,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.7
 cov_89_12: frac[5,12] + frac[26,12] + frac[32,12] + frac[35,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.7
 cov_90_8: frac[19,8] + frac[25,8] + frac[34,8] + frac[35,8] + frac[37,8]
   >= 0.7
 cov_90_9: frac[19,9] + frac[25,9] + frac[34,9] + frac[35,9] + frac[37,9]
   >= 0.7
 cov_90_10: frac[19,10] + frac[25,10] + frac[34,10] + frac[35,10]
   + frac[37,10] >= 0.7
 cov_90_11: frac[19,11] + frac[25,11] + frac[34,11] + frac[35,11]
   + frac[37,11] >= 0.7
 cov_90_12: frac[19,12] + frac[25,12] + frac[34,12] + frac[35,12]
   + frac[37,12] >= 0.7
 cov_91_4: frac[0,4] + frac[18,4] + frac[32,4] + frac[34,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.8
 cov_91_5: frac[0,5] + frac[18,5] + frac[32,5] + frac[34,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.8
 cov_91_6: frac[0,6] + frac[18,6] + frac[32,6] + frac[34,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.8
 cov_91_7: frac[0,7] + frac[18,7] + frac[32,7] + frac[34,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.8
 cov_92_4: frac[9,4] + frac[18,4] + frac[33,4] + frac[34,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.7
 cov_92_5: frac[9,5] + frac[18,5] + frac[33,5] + frac[34,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.7
 cov_92_6: frac[9,6] + frac[18,6] + frac[33,6] + frac[34,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.7
 cov_92_7: frac[9,7] + frac[18,7] + frac[33,7] + frac[34,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.7
 cov_93_8: frac[15,8] + frac[20,8] + frac[33,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.9
 cov_93_9: frac[15,9] + frac[20,9] + frac[33,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.9
 cov_93_10: frac[15,10] + frac[20,10] + frac[33,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.9
 cov_93_11: frac[15,11] + frac[20,11] + frac[33,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.9
 cov_93_12: frac[15,12] + frac[20,12] + frac[33,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.9
 cov_94_4: frac[3,4] + frac[19,4] + frac[32,4] + frac[34,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.7
 cov_94_5: frac[3,5] + frac[19,5] + frac[32,5] + frac[34,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.7
 cov_94_6: frac[3,6] + frac[19,6] + frac[32,6] + frac[34,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.7
 cov_94_7: frac[3,7] + frac[19,7] + frac[32,7] + frac[34,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.7
 cov_95_0: frac[9,0] + frac[30,0] + frac[33,0] + frac[35,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 1
 cov_95_1: frac[9,1] + frac[30,1] + frac[33,1] + frac[35,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 1
 cov_95_2: frac[9,2] + frac[30,2] + frac[33,2] + frac[35,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 1
 cov_95_3: frac[9,3] + frac[30,3] + frac[33,3] + frac[35,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 1
 cov_96_4: frac[11,4] + frac[17,4] + frac[33,4] + frac[34,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 1
 cov_96_5: frac[11,5] + frac[17,5] + frac[33,5] + frac[34,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 1
 cov_96_6: frac[11,6] + frac[17,6] + frac[33,6] + frac[34,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 1
 cov_96_7: frac[11,7] + frac[17,7] + frac[33,7] + frac[34,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 1
 cov_97_0: frac[10,0] + frac[14,0] + frac[33,0] >= 0.7
 cov_97_1: frac[10,1] + frac[14,1] + frac[33,1] >= 0.7
 cov_97_2: frac[10,2] + frac[14,2] + frac[33,2] >= 0.7
 cov_97_3: frac[10,3] + frac[14,3] + frac[33,3] >= 0.7
 cov_98_4: frac[15,4] + frac[31,4] + frac[33,4] + frac[35,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 1
 cov_98_5: frac[15,5] + frac[31,5] + frac[33,5] + frac[35,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 1
 cov_98_6: frac[15,6] + frac[31,6] + frac[33,6] + frac[35,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 1
 cov_98_7: frac[15,7] + frac[31,7] + frac[33,7] + frac[35,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 1
 cov_99_8: frac[12,8] + frac[17,8] + frac[33,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.8
 cov_99_9: frac[12,9] + frac[17,9] + frac[33,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.8
 cov_99_10: frac[12,10] + frac[17,10] + frac[33,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.8
 cov_99_11: frac[12,11] + frac[17,11] + frac[33,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.8
 cov_99_12: frac[12,12] + frac[17,12] + frac[33,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.8
 cov_100_8: frac[0,8] + frac[26,8] + frac[32,8] + frac[35,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.9
 cov_100_9: frac[0,9] + frac[26,9] + frac[32,9] + frac[35,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.9
 cov_100_10: frac[0,10] + frac[26,10] + frac[32,10] + frac[35,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.9
 cov_100_11: frac[0,11] + frac[26,11] + frac[32,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.9
 cov_100_12: frac[0,12] + frac[26,12] + frac[32,12] + frac[35,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.9
 cov_101_8: frac[13,8] + frac[17,8] + frac[33,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 1
 cov_101_9: frac[13,9] + frac[17,9] + frac[33,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 1
 cov_101_10: frac[13,10] + frac[17,10] + frac[33,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 1
 cov_101_11: frac[13,11] + frac[17,11] + frac[33,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 1
 cov_101_12: frac[13,12] + frac[17,12] + frac[33,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 1
 cov_102_8: frac[3,8] + frac[26,8] + frac[32,8] + frac[35,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.7
 cov_102_9: frac[3,9] + frac[26,9] + frac[32,9] + frac[35,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.7
 cov_102_10: frac[3,10] + frac[26,10] + frac[32,10] + frac[35,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.7
 cov_102_11: frac[3,11] + frac[26,11] + frac[32,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.7
 cov_102_12: frac[3,12] + frac[26,12] + frac[32,12] + frac[35,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.7
 cov_103_4: frac[21,4] + frac[26,4] + frac[34,4] + frac[35,4] + frac[37,4]
   >= 0.9
 cov_103_5: frac[21,5] + frac[26,5] + frac[34,5] + frac[35,5] + frac[37,5]
   >= 0.9
 cov_103_6: frac[21,6] + frac[26,6] + frac[34,6] + frac[35,6] + frac[37,6]
   >= 0.9
 cov_103_7: frac[21,7] + frac[26,7] + frac[34,7] + frac[35,7] + frac[37,7]
   >= 0.9
 cov_104_0: frac[4,0] + frac[24,0] + frac[32,0] + frac[35,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.7
 cov_104_1: frac[4,1] + frac[24,1] + frac[32,1] + frac[35,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.7
 cov_104_2: frac[4,2] + frac[24,2] + frac[32,2] + frac[35,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.7
 cov_104_3: frac[4,3] + frac[24,3] + frac[32,3] + frac[35,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.7
 cov_105_4: frac[3,4] + frac[31,4] + frac[32,4] + frac[35,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.9
 cov_105_5: frac[3,5] + frac[31,5] + frac[32,5] + frac[35,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.9
 cov_105_6: frac[3,6] + frac[31,6] + frac[32,6] + frac[35,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.9
 cov_105_7: frac[3,7] + frac[31,7] + frac[32,7] + frac[35,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.9
 cov_106_4: frac[5,4] + frac[30,4] + frac[32,4] + frac[35,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.9
 cov_106_5: frac[5,5] + frac[30,5] + frac[32,5] + frac[35,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.9
 cov_106_6: frac[5,6] + frac[30,6] + frac[32,6] + frac[35,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.9
 cov_106_7: frac[5,7] + frac[30,7] + frac[32,7] + frac[35,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.9
 cov_107_0: frac[15,0] + frac[21,0] + frac[33,0] + frac[34,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 1
 cov_107_1: frac[15,1] + frac[21,1] + frac[33,1] + frac[34,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 1
 cov_107_2: frac[15,2] + frac[21,2] + frac[33,2] + frac[34,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 1
 cov_107_3: frac[15,3] + frac[21,3] + frac[33,3] + frac[34,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 1
 cov_108_4: frac[10,4] + frac[31,4] + frac[33,4] + frac[35,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 1
 cov_108_5: frac[10,5] + frac[31,5] + frac[33,5] + frac[35,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 1
 cov_108_6: frac[10,6] + frac[31,6] + frac[33,6] + frac[35,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 1
 cov_108_7: frac[10,7] + frac[31,7] + frac[33,7] + frac[35,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 1
 cov_109_4: frac[24,4] + frac[27,4] + frac[35,4] >= 0.7
 cov_109_5: frac[24,5] + frac[27,5] + frac[35,5] >= 0.7
 cov_109_6: frac[24,6] + frac[27,6] + frac[35,6] >= 0.7
 cov_109_7: frac[24,7] + frac[27,7] + frac[35,7] >= 0.7
 cov_110_0: frac[7,0] + frac[15,0] + frac[32,0] + frac[33,0] + frac[36,0]
   >= 0.8
 cov_110_1: frac[7,1] + frac[15,1] + frac[32,1] + frac[33,1] + frac[36,1]
   >= 0.8
 cov_110_2: frac[7,2] + frac[15,2] + frac[32,2] + frac[33,2] + frac[36,2]
   >= 0.8
 cov_110_3: frac[7,3] + frac[15,3] + frac[32,3] + frac[33,3] + frac[36,3]
   >= 0.8
 cov_111_0: frac[1,0] + frac[16,0] + frac[32,0] + frac[34,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.9
 cov_111_1: frac[1,1] + frac[16,1] + frac[32,1] + frac[34,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.9
 cov_111_2: frac[1,2] + frac[16,2] + frac[32,2] + frac[34,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.9
 cov_111_3: frac[1,3] + frac[16,3] + frac[32,3] + frac[34,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.9
 cov_112_0: frac[9,0] + frac[18,0] + frac[33,0] + frac[34,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.9
 cov_112_1: frac[9,1] + frac[18,1] + frac[33,1] + frac[34,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.9
 cov_112_2: frac[9,2] + frac[18,2] + frac[33,2] + frac[34,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.9
 cov_112_3: frac[9,3] + frac[18,3] + frac[33,3] + frac[34,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.9
 cov_113_8: frac[0,8] + frac[3,8] + frac[32,8] >= 0.7
 cov_113_9: frac[0,9] + frac[3,9] + frac[32,9] >= 0.7
 cov_113_10: frac[0,10] + frac[3,10] + frac[32,10] >= 0.7
 cov_113_11: frac[0,11] + frac[3,11] + frac[32,11] >= 0.7
 cov_113_12: frac[0,12] + frac[3,12] + frac[32,12] >= 0.7
 cov_114_8: frac[4,8] + frac[7,8] + frac[32,8] >= 0.9
 cov_114_9: frac[4,9] + frac[7,9] + frac[32,9] >= 0.9
 cov_114_10: frac[4,10] + frac[7,10] + frac[32,10] >= 0.9
 cov_114_11: frac[4,11] + frac[7,11] + frac[32,11] >= 0.9
 cov_114_12: frac[4,12] + frac[7,12] + frac[32,12] >= 0.9
 cov_115_8: frac[17,8] + frac[31,8] + frac[34,8] + frac[35,8] + frac[37,8]
   >= 0.8
 cov_115_9: frac[17,9] + frac[31,9] + frac[34,9] + frac[35,9] + frac[37,9]
   >= 0.8
 cov_115_10: frac[17,10] + frac[31,10] + frac[34,10] + frac[35,10]
   + frac[37,10] >= 0.8
 cov_115_11: frac[17,11] + frac[31,11] + frac[34,11] + frac[35,11]
   + frac[37,11] >= 0.8
 cov_115_12: frac[17,12] + frac[31,12] + frac[34,12] + frac[35,12]
   + frac[37,12] >= 0.8
 cov_116_8: frac[11,8] + frac[18,8] + frac[33,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.9
 cov_116_9: frac[11,9] + frac[18,9] + frac[33,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.9
 cov_116_10: frac[11,10] + frac[18,10] + frac[33,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.9
 cov_116_11: frac[11,11] + frac[18,11] + frac[33,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.9
 cov_116_12: frac[11,12] + frac[18,12] + frac[33,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.9
 cov_117_4: frac[8,4] + frac[26,4] + frac[33,4] + frac[35,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.8
 cov_117_5: frac[8,5] + frac[26,5] + frac[33,5] + frac[35,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.8
 cov_117_6: frac[8,6] + frac[26,6] + frac[33,6] + frac[35,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.8
 cov_117_7: frac[8,7] + frac[26,7] + frac[33,7] + frac[35,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.8
 cov_118_8: frac[21,8] + frac[24,8] + frac[34,8] + frac[35,8] + frac[37,8]
   >= 0.9
 cov_118_9: frac[21,9] + frac[24,9] + frac[34,9] + frac[35,9] + frac[37,9]
   >= 0.9
 cov_118_10: frac[21,10] + frac[24,10] + frac[34,10] + frac[35,10]
   + frac[37,10] >= 0.9
 cov_118_11: frac[21,11] + frac[24,11] + frac[34,11] + frac[35,11]
   + frac[37,11] >= 0.9
 cov_118_12: frac[21,12] + frac[24,12] + frac[34,12] + frac[35,12]
   + frac[37,12] >= 0.9
 cov_119_8: frac[3,8] + frac[19,8] + frac[32,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 1
 cov_119_9: frac[3,9] + frac[19,9] + frac[32,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 1
 cov_119_10: frac[3,10] + frac[19,10] + frac[32,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 1
 cov_119_11: frac[3,11] + frac[19,11] + frac[32,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 1
 cov_119_12: frac[3,12] + frac[19,12] + frac[32,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 1
 cov_120_0: frac[15,0] + frac[16,0] + frac[33,0] + frac[34,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 1
 cov_120_1: frac[15,1] + frac[16,1] + frac[33,1] + frac[34,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 1
 cov_120_2: frac[15,2] + frac[16,2] + frac[33,2] + frac[34,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 1
 cov_120_3: frac[15,3] + frac[16,3] + frac[33,3] + frac[34,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 1
 cov_121_4: frac[16,4] >= 0.7
 cov_121_5: frac[16,5] >= 0.7
 cov_121_6: frac[16,6] >= 0.7
 cov_121_7: frac[16,7] >= 0.7
 cov_122_0: frac[5,0] + frac[9,0] + frac[32,0] + frac[33,0] + frac[36,0]
   >= 1
 cov_122_1: frac[5,1] + frac[9,1] + frac[32,1] + frac[33,1] + frac[36,1]
   >= 1
 cov_122_2: frac[5,2] + frac[9,2] + frac[32,2] + frac[33,2] + frac[36,2]
   >= 1
 cov_122_3: frac[5,3] + frac[9,3] + frac[32,3] + frac[33,3] + frac[36,3]
   >= 1
 cov_123_0: frac[16,0] + frac[28,0] + frac[34,0] + frac[35,0] + frac[37,0]
   >= 0.7
 cov_123_1: frac[16,1] + frac[28,1] + frac[34,1] + frac[35,1] + frac[37,1]
   >= 0.7
 cov_123_2: frac[16,2] + frac[28,2] + frac[34,2] + frac[35,2] + frac[37,2]
   >= 0.7
 cov_123_3: frac[16,3] + frac[28,3] + frac[34,3] + frac[35,3] + frac[37,3]
   >= 0.7
 cov_124_0: frac[14,0] + frac[21,0] + frac[33,0] + frac[34,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 1
 cov_124_1: frac[14,1] + frac[21,1] + frac[33,1] + frac[34,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 1
 cov_124_2: frac[14,2] + frac[21,2] + frac[33,2] + frac[34,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 1
 cov_124_3: frac[14,3] + frac[21,3] + frac[33,3] + frac[34,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 1
 cov_125_4: frac[14,4] + frac[19,4] + frac[33,4] + frac[34,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.7
 cov_125_5: frac[14,5] + frac[19,5] + frac[33,5] + frac[34,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.7
 cov_125_6: frac[14,6] + frac[19,6] + frac[33,6] + frac[34,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.7
 cov_125_7: frac[14,7] + frac[19,7] + frac[33,7] + frac[34,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.7
 cov_126_0: frac[12,0] + frac[17,0] + frac[33,0] + frac[34,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 1
 cov_126_1: frac[12,1] + frac[17,1] + frac[33,1] + frac[34,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 1
 cov_126_2: frac[12,2] + frac[17,2] + frac[33,2] + frac[34,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 1
 cov_126_3: frac[12,3] + frac[17,3] + frac[33,3] + frac[34,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 1
 cov_127_0: frac[18,0] + frac[22,0] + frac[34,0] >= 0.8
 cov_127_1: frac[18,1] + frac[22,1] + frac[34,1] >= 0.8
 cov_127_2: frac[18,2] + frac[22,2] + frac[34,2] >= 0.8
 cov_127_3: frac[18,3] + frac[22,3] + frac[34,3] >= 0.8
 cov_128_0: frac[10,0] + frac[20,0] + frac[33,0] + frac[34,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 1
 cov_128_1: frac[10,1] + frac[20,1] + frac[33,1] + frac[34,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 1
 cov_128_2: frac[10,2] + frac[20,2] + frac[33,2] + frac[34,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 1
 cov_128_3: frac[10,3] + frac[20,3] + frac[33,3] + frac[34,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 1
 cov_129_8: frac[1,8] + frac[3,8] + frac[32,8] >= 0.8
 cov_129_9: frac[1,9] + frac[3,9] + frac[32,9] >= 0.8
 cov_129_10: frac[1,10] + frac[3,10] + frac[32,10] >= 0.8
 cov_129_11: frac[1,11] + frac[3,11] + frac[32,11] >= 0.8
 cov_129_12: frac[1,12] + frac[3,12] + frac[32,12] >= 0.8
 cov_130_8: frac[19,8] + frac[29,8] + frac[34,8] + frac[35,8] + frac[37,8]
   >= 1
 cov_130_9: frac[19,9] + frac[29,9] + frac[34,9] + frac[35,9] + frac[37,9]
   >= 1
 cov_130_10: frac[19,10] + frac[29,10] + frac[34,10] + frac[35,10]
   + frac[37,10] >= 1
 cov_130_11: frac[19,11] + frac[29,11] + frac[34,11] + frac[35,11]
   + frac[37,11] >= 1
 cov_130_12: frac[19,12] + frac[29,12] + frac[34,12] + frac[35,12]
   + frac[37,12] >= 1
 cov_131_0: frac[1,0] + frac[24,0] + frac[32,0] + frac[35,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.9
 cov_131_1: frac[1,1] + frac[24,1] + frac[32,1] + frac[35,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.9
 cov_131_2: frac[1,2] + frac[24,2] + frac[32,2] + frac[35,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.9
 cov_131_3: frac[1,3] + frac[24,3] + frac[32,3] + frac[35,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.9
 cov_132_0: frac[1,0] >= 0.7
 cov_132_1: frac[1,1] >= 0.7
 cov_132_2: frac[1,2] >= 0.7
 cov_132_3: frac[1,3] >= 0.7
 cov_133_8: frac[4,8] + frac[23,8] + frac[32,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 1
 cov_133_9: frac[4,9] + frac[23,9] + frac[32,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 1
 cov_133_10: frac[4,10] + frac[23,10] + frac[32,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 1
 cov_133_11: frac[4,11] + frac[23,11] + frac[32,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 1
 cov_133_12: frac[4,12] + frac[23,12] + frac[32,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 1
 cov_134_8: frac[5,8] + frac[31,8] + frac[32,8] + frac[35,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.7
 cov_134_9: frac[5,9] + frac[31,9] + frac[32,9] + frac[35,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.7
 cov_134_10: frac[5,10] + frac[31,10] + frac[32,10] + frac[35,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.7
 cov_134_11: frac[5,11] + frac[31,11] + frac[32,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.7
 cov_134_12: frac[5,12] + frac[31,12] + frac[32,12] + frac[35,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.7
 cov_135_4: frac[9,4] + frac[29,4] + frac[33,4] + frac[35,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.8
 cov_135_5: frac[9,5] + frac[29,5] + frac[33,5] + frac[35,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.8
 cov_135_6: frac[9,6] + frac[29,6] + frac[33,6] + frac[35,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.8
 cov_135_7: frac[9,7] + frac[29,7] + frac[33,7] + frac[35,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.8
 cov_136_8: frac[12,8] + frac[29,8] + frac[33,8] + frac[35,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.7
 cov_136_9: frac[12,9] + frac[29,9] + frac[33,9] + frac[35,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.7
 cov_136_10: frac[12,10] + frac[29,10] + frac[33,10] + frac[35,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.7
 cov_136_11: frac[12,11] + frac[29,11] + frac[33,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.7
 cov_136_12: frac[12,12] + frac[29,12] + frac[33,12] + frac[35,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.7
 cov_137_4: frac[3,4] + frac[10,4] + frac[32,4] + frac[33,4] + frac[36,4]
   >= 0.9
 cov_137_5: frac[3,5] + frac[10,5] + frac[32,5] + frac[33,5] + frac[36,5]
   >= 0.9
 cov_137_6: frac[3,6] + frac[10,6] + frac[32,6] + frac[33,6] + frac[36,6]
   >= 0.9
 cov_137_7: frac[3,7] + frac[10,7] + frac[32,7] + frac[33,7] + frac[36,7]
   >= 0.9
 cov_138_8: frac[7,8] + frac[28,8] + frac[32,8] + frac[35,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.9
 cov_138_9: frac[7,9] + frac[28,9] + frac[32,9] + frac[35,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.9
 cov_138_10: frac[7,10] + frac[28,10] + frac[32,10] + frac[35,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.9
 cov_138_11: frac[7,11] + frac[28,11] + frac[32,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.9
 cov_138_12: frac[7,12] + frac[28,12] + frac[32,12] + frac[35,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.9
 cov_139_8: frac[11,8] + frac[16,8] + frac[33,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.9
 cov_139_9: frac[11,9] + frac[16,9] + frac[33,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.9
 cov_139_10: frac[11,10] + frac[16,10] + frac[33,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.9
 cov_139_11: frac[11,11] + frac[16,11] + frac[33,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.9
 cov_139_12: frac[11,12] + frac[16,12] + frac[33,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.9
 cov_140_8: frac[9,8] + frac[28,8] + frac[33,8] + frac[35,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.7
 cov_140_9: frac[9,9] + frac[28,9] + frac[33,9] + frac[35,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.7
 cov_140_10: frac[9,10] + frac[28,10] + frac[33,10] + frac[35,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.7
 cov_140_11: frac[9,11] + frac[28,11] + frac[33,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.7
 cov_140_12: frac[9,12] + frac[28,12] + frac[33,12] + frac[35,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.7
 cov_141_8: frac[11,8] + frac[31,8] + frac[33,8] + frac[35,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.7
 cov_141_9: frac[11,9] + frac[31,9] + frac[33,9] + frac[35,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.7
 cov_141_10: frac[11,10] + frac[31,10] + frac[33,10] + frac[35,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.7
 cov_141_11: frac[11,11] + frac[31,11] + frac[33,11] + frac[35,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.7
 cov_141_12: frac[11,12] + frac[31,12] + frac[33,12] + frac[35,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.7
 cov_142_0: frac[0,0] + frac[9,0] + frac[32,0] + frac[33,0] + frac[36,0]
   >= 0.8
 cov_142_1: frac[0,1] + frac[9,1] + frac[32,1] + frac[33,1] + frac[36,1]
   >= 0.8
 cov_142_2: frac[0,2] + frac[9,2] + frac[32,2] + frac[33,2] + frac[36,2]
   >= 0.8
 cov_142_3: frac[0,3] + frac[9,3] + frac[32,3] + frac[33,3] + frac[36,3]
   >= 0.8
 cov_143_4: frac[11,4] + frac[20,4] + frac[33,4] + frac[34,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.7
 cov_143_5: frac[11,5] + frac[20,5] + frac[33,5] + frac[34,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.7
 cov_143_6: frac[11,6] + frac[20,6] + frac[33,6] + frac[34,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.7
 cov_143_7: frac[11,7] + frac[20,7] + frac[33,7] + frac[34,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.7
 cov_144_8: frac[3,8] + frac[8,8] + frac[32,8] + frac[33,8] + frac[36,8]
   >= 0.7
 cov_144_9: frac[3,9] + frac[8,9] + frac[32,9] + frac[33,9] + frac[36,9]
   >= 0.7
 cov_144_10: frac[3,10] + frac[8,10] + frac[32,10] + frac[33,10]
   + frac[36,10] >= 0.7
 cov_144_11: frac[3,11] + frac[8,11] + frac[32,11] + frac[33,11]
   + frac[36,11] >= 0.7
 cov_144_12: frac[3,12] + frac[8,12] + frac[32,12] + frac[33,12]
   + frac[36,12] >= 0.7
 cov_145_8: frac[19,8] + frac[26,8] + frac[34,8] + frac[35,8] + frac[37,8]
   >= 0.9
 cov_145_9: frac[19,9] + frac[26,9] + frac[34,9] + frac[35,9] + frac[37,9]
   >= 0.9
 cov_145_10: frac[19,10] + frac[26,10] + frac[34,10] + frac[35,10]
   + frac[37,10] >= 0.9
 cov_145_11: frac[19,11] + frac[26,11] + frac[34,11] + frac[35,11]
   + frac[37,11] >= 0.9
 cov_145_12: frac[19,12] + frac[26,12] + frac[34,12] + frac[35,12]
   + frac[37,12] >= 0.9
 cov_146_4: frac[14,4] + frac[18,4] + frac[33,4] + frac[34,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.9
 cov_146_5: frac[14,5] + frac[18,5] + frac[33,5] + frac[34,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.9
 cov_146_6: frac[14,6] + frac[18,6] + frac[33,6] + frac[34,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.9
 cov_146_7: frac[14,7] + frac[18,7] + frac[33,7] + frac[34,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.9
 cov_147_4: frac[1,4] + frac[4,4] + frac[32,4] >= 0.8
 cov_147_5: frac[1,5] + frac[4,5] + frac[32,5] >= 0.8
 cov_147_6: frac[1,6] + frac[4,6] + frac[32,6] >= 0.8
 cov_147_7: frac[1,7] + frac[4,7] + frac[32,7] >= 0.8
 cov_148_8: frac[0,8] + frac[23,8] + frac[32,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 1
 cov_148_9: frac[0,9] + frac[23,9] + frac[32,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 1
 cov_148_10: frac[0,10] + frac[23,10] + frac[32,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 1
 cov_148_11: frac[0,11] + frac[23,11] + frac[32,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 1
 cov_148_12: frac[0,12] + frac[23,12] + frac[32,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 1
 cov_149_0: frac[6,0] + frac[30,0] + frac[32,0] + frac[35,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.8
 cov_149_1: frac[6,1] + frac[30,1] + frac[32,1] + frac[35,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.8
 cov_149_2: frac[6,2] + frac[30,2] + frac[32,2] + frac[35,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.8
 cov_149_3: frac[6,3] + frac[30,3] + frac[32,3] + frac[35,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.8
 cov_150_4: frac[8,4] + frac[28,4] + frac[33,4] + frac[35,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 1
 cov_150_5: frac[8,5] + frac[28,5] + frac[33,5] + frac[35,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 1
 cov_150_6: frac[8,6] + frac[28,6] + frac[33,6] + frac[35,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 1
 cov_150_7: frac[8,7] + frac[28,7] + frac[33,7] + frac[35,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 1
 cov_151_8: frac[22,8] + frac[30,8] + frac[34,8] + frac[35,8] + frac[37,8]
   >= 0.8
 cov_151_9: frac[22,9] + frac[30,9] + frac[34,9] + frac[35,9] + frac[37,9]
   >= 0.8
 cov_151_10: frac[22,10] + frac[30,10] + frac[34,10] + frac[35,10]
   + frac[37,10] >= 0.8
 cov_151_11: frac[22,11] + frac[30,11] + frac[34,11] + frac[35,11]
   + frac[37,11] >= 0.8
 cov_151_12: frac[22,12] + frac[30,12] + frac[34,12] + frac[35,12]
   + frac[37,12] >= 0.8
 cov_152_4: frac[5,4] + frac[11,4] + frac[32,4] + frac[33,4] + frac[36,4]
   >= 0.9
 cov_152_5: frac[5,5] + frac[11,5] + frac[32,5] + frac[33,5] + frac[36,5]
   >= 0.9
 cov_152_6: frac[5,6] + frac[11,6] + frac[32,6] + frac[33,6] + frac[36,6]
   >= 0.9
 cov_152_7: frac[5,7] + frac[11,7] + frac[32,7] + frac[33,7] + frac[36,7]
   >= 0.9
 cov_153_4: frac[15,4] + frac[28,4] + frac[33,4] + frac[35,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.8
 cov_153_5: frac[15,5] + frac[28,5] + frac[33,5] + frac[35,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.8
 cov_153_6: frac[15,6] + frac[28,6] + frac[33,6] + frac[35,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.8
 cov_153_7: frac[15,7] + frac[28,7] + frac[33,7] + frac[35,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.8
 cov_154_4: frac[4,4] + frac[25,4] + frac[32,4] + frac[35,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 0.9
 cov_154_5: frac[4,5] + frac[25,5] + frac[32,5] + frac[35,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 0.9
 cov_154_6: frac[4,6] + frac[25,6] + frac[32,6] + frac[35,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 0.9
 cov_154_7: frac[4,7] + frac[25,7] + frac[32,7] + frac[35,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 0.9
 cov_155_8: frac[5,8] + frac[22,8] + frac[32,8] + frac[34,8] + frac[36,8]
   + frac[37,8] + frac[38,8] >= 0.9
 cov_155_9: frac[5,9] + frac[22,9] + frac[32,9] + frac[34,9] + frac[36,9]
   + frac[37,9] + frac[38,9] >= 0.9
 cov_155_10: frac[5,10] + frac[22,10] + frac[32,10] + frac[34,10]
   + frac[36,10] + frac[37,10] + frac[38,10] >= 0.9
 cov_155_11: frac[5,11] + frac[22,11] + frac[32,11] + frac[34,11]
   + frac[36,11] + frac[37,11] + frac[38,11] >= 0.9
 cov_155_12: frac[5,12] + frac[22,12] + frac[32,12] + frac[34,12]
   + frac[36,12] + frac[37,12] + frac[38,12] >= 0.9
 cov_156_0: frac[0,0] + frac[20,0] + frac[32,0] + frac[34,0] + frac[36,0]
   + frac[37,0] + frac[38,0] >= 0.7
 cov_156_1: frac[0,1] + frac[20,1] + frac[32,1] + frac[34,1] + frac[36,1]
   + frac[37,1] + frac[38,1] >= 0.7
 cov_156_2: frac[0,2] + frac[20,2] + frac[32,2] + frac[34,2] + frac[36,2]
   + frac[37,2] + frac[38,2] >= 0.7
 cov_156_3: frac[0,3] + frac[20,3] + frac[32,3] + frac[34,3] + frac[36,3]
   + frac[37,3] + frac[38,3] >= 0.7
 cov_157_4: frac[13,4] + frac[18,4] + frac[33,4] + frac[34,4] + frac[36,4]
   + frac[37,4] + frac[38,4] >= 1
 cov_157_5: frac[13,5] + frac[18,5] + frac[33,5] + frac[34,5] + frac[36,5]
   + frac[37,5] + frac[38,5] >= 1
 cov_157_6: frac[13,6] + frac[18,6] + frac[33,6] + frac[34,6] + frac[36,6]
   + frac[37,6] + frac[38,6] >= 1
 cov_157_7: frac[13,7] + frac[18,7] + frac[33,7] + frac[34,7] + frac[36,7]
   + frac[37,7] + frac[38,7] >= 1
 cov_158_4: frac[21,4] + frac[29,4] + frac[34,4] + frac[35,4] + frac[37,4]
   >= 0.8
 cov_158_5: frac[21,5] + frac[29,5] + frac[34,5] + frac[35,5] + frac[37,5]
   >= 0.8
 cov_158_6: frac[21,6] + frac[29,6] + frac[34,6] + frac[35,6] + frac[37,6]
   >= 0.8
 cov_158_7: frac[21,7] + frac[29,7] + frac[34,7] + frac[35,7] + frac[37,7]
   >= 0.8
 cov_159_0: frac[8,0] + frac[10,0] + frac[33,0] >= 1
 cov_159_1: frac[8,1] + frac[10,1] + frac[33,1] >= 1
 cov_159_2: frac[8,2] + frac[10,2] + frac[33,2] >= 1
 cov_159_3: frac[8,3] + frac[10,3] + frac[33,3] >= 1
 accuracy_(0,_sk_0)[0]: - 21.23828125 frac[0,0] + mem[0,0] >= 0
 accuracy_(0,_sk_0)[1]: - 21.23828125 frac[1,0] + mem[1,0] >= 0
 accuracy_(0,_sk_0)[2]: - 21.23828125 frac[2,0] + mem[2,0] >= 0
 accuracy_(0,_sk_0)[3]: - 21.23828125 frac[3,0] + mem[3,0] >= 0
 accuracy_(0,_sk_0)[4]: - 21.23828125 frac[4,0] + mem[4,0] >= 0
 accuracy_(0,_sk_0)[5]: - 21.23828125 frac[5,0] + mem[5,0] >= 0
 accuracy_(0,_sk_0)[6]: - 21.23828125 frac[6,0] + mem[6,0] >= 0
 accuracy_(0,_sk_0)[7]: - 21.23828125 frac[7,0] + mem[7,0] >= 0
 accuracy_(0,_sk_0)[8]: - 21.23828125 frac[8,0] + mem[8,0] >= 0
 accuracy_(0,_sk_0)[9]: - 21.23828125 frac[9,0] + mem[9,0] >= 0
 accuracy_(0,_sk_0)[10]: - 21.23828125 frac[10,0] + mem[10,0] >= 0
 accuracy_(0,_sk_0)[11]: - 21.23828125 frac[11,0] + mem[11,0] >= 0
 accuracy_(0,_sk_0)[12]: - 21.23828125 frac[12,0] + mem[12,0] >= 0
 accuracy_(0,_sk_0)[13]: - 21.23828125 frac[13,0] + mem[13,0] >= 0
 accuracy_(0,_sk_0)[14]: - 21.23828125 frac[14,0] + mem[14,0] >= 0
 accuracy_(0,_sk_0)[15]: - 21.23828125 frac[15,0] + mem[15,0] >= 0
 accuracy_(0,_sk_0)[16]: - 21.23828125 frac[16,0] + mem[16,0] >= 0
 accuracy_(0,_sk_0)[17]: - 21.23828125 frac[17,0] + mem[17,0] >= 0
 accuracy_(0,_sk_0)[18]: - 21.23828125 frac[18,0] + mem[18,0] >= 0
 accuracy_(0,_sk_0)[19]: - 21.23828125 frac[19,0] + mem[19,0] >= 0
 accuracy_(0,_sk_0)[20]: - 21.23828125 frac[20,0] + mem[20,0] >= 0
 accuracy_(0,_sk_0)[21]: - 21.23828125 frac[21,0] + mem[21,0] >= 0
 accuracy_(0,_sk_0)[22]: - 21.23828125 frac[22,0] + mem[22,0] >= 0
 accuracy_(0,_sk_0)[23]: - 21.23828125 frac[23,0] + mem[23,0] >= 0
 accuracy_(0,_sk_0)[24]: - 21.23828125 frac[24,0] + mem[24,0] >= 0
 accuracy_(0,_sk_0)[25]: - 21.23828125 frac[25,0] + mem[25,0] >= 0
 accuracy_(0,_sk_0)[26]: - 21.23828125 frac[26,0] + mem[26,0] >= 0
 accuracy_(0,_sk_0)[27]: - 21.23828125 frac[27,0] + mem[27,0] >= 0
 accuracy_(0,_sk_0)[28]: - 21.23828125 frac[28,0] + mem[28,0] >= 0
 accuracy_(0,_sk_0)[29]: - 21.23828125 frac[29,0] + mem[29,0] >= 0
 accuracy_(0,_sk_0)[30]: - 21.23828125 frac[30,0] + mem[30,0] >= 0
 accuracy_(0,_sk_0)[31]: - 21.23828125 frac[31,0] + mem[31,0] >= 0
 accuracy_(0,_sk_0)[32]: - 21.23828125 frac[32,0] + mem[32,0] >= 0
 accuracy_(0,_sk_0)[33]: - 21.23828125 frac[33,0] + mem[33,0] >= 0
 accuracy_(0,_sk_0)[34]: - 21.23828125 frac[34,0] + mem[34,0] >= 0
 accuracy_(0,_sk_0)[35]: - 21.23828125 frac[35,0] + mem[35,0] >= 0
 accuracy_(0,_sk_0)[36]: - 21.23828125 frac[36,0] + mem[36,0] >= 0
 accuracy_(0,_sk_0)[37]: - 21.23828125 frac[37,0] + mem[37,0] >= 0
 accuracy_(0,_sk_0)[38]: - 21.23828125 frac[38,0] + mem[38,0] >= 0
 accuracy_(1,_sk_0)[0]: - 21.23828125 frac[0,1] + mem[0,1] >= 0
 accuracy_(1,_sk_0)[1]: - 21.23828125 frac[1,1] + mem[1,1] >= 0
 accuracy_(1,_sk_0)[2]: - 21.23828125 frac[2,1] + mem[2,1] >= 0
 accuracy_(1,_sk_0)[3]: - 21.23828125 frac[3,1] + mem[3,1] >= 0
 accuracy_(1,_sk_0)[4]: - 21.23828125 frac[4,1] + mem[4,1] >= 0
 accuracy_(1,_sk_0)[5]: - 21.23828125 frac[5,1] + mem[5,1] >= 0
 accuracy_(1,_sk_0)[6]: - 21.23828125 frac[6,1] + mem[6,1] >= 0
 accuracy_(1,_sk_0)[7]: - 21.23828125 frac[7,1] + mem[7,1] >= 0
 accuracy_(1,_sk_0)[8]: - 21.23828125 frac[8,1] + mem[8,1] >= 0
 accuracy_(1,_sk_0)[9]: - 21.23828125 frac[9,1] + mem[9,1] >= 0
 accuracy_(1,_sk_0)[10]: - 21.23828125 frac[10,1] + mem[10,1] >= 0
 accuracy_(1,_sk_0)[11]: - 21.23828125 frac[11,1] + mem[11,1] >= 0
 accuracy_(1,_sk_0)[12]: - 21.23828125 frac[12,1] + mem[12,1] >= 0
 accuracy_(1,_sk_0)[13]: - 21.23828125 frac[13,1] + mem[13,1] >= 0
 accuracy_(1,_sk_0)[14]: - 21.23828125 frac[14,1] + mem[14,1] >= 0
 accuracy_(1,_sk_0)[15]: - 21.23828125 frac[15,1] + mem[15,1] >= 0
 accuracy_(1,_sk_0)[16]: - 21.23828125 frac[16,1] + mem[16,1] >= 0
 accuracy_(1,_sk_0)[17]: - 21.23828125 frac[17,1] + mem[17,1] >= 0
 accuracy_(1,_sk_0)[18]: - 21.23828125 frac[18,1] + mem[18,1] >= 0
 accuracy_(1,_sk_0)[19]: - 21.23828125 frac[19,1] + mem[19,1] >= 0
 accuracy_(1,_sk_0)[20]: - 21.23828125 frac[20,1] + mem[20,1] >= 0
 accuracy_(1,_sk_0)[21]: - 21.23828125 frac[21,1] + mem[21,1] >= 0
 accuracy_(1,_sk_0)[22]: - 21.23828125 frac[22,1] + mem[22,1] >= 0
 accuracy_(1,_sk_0)[23]: - 21.23828125 frac[23,1] + mem[23,1] >= 0
 accuracy_(1,_sk_0)[24]: - 21.23828125 frac[24,1] + mem[24,1] >= 0
 accuracy_(1,_sk_0)[25]: - 21.23828125 frac[25,1] + mem[25,1] >= 0
 accuracy_(1,_sk_0)[26]: - 21.23828125 frac[26,1] + mem[26,1] >= 0
 accuracy_(1,_sk_0)[27]: - 21.23828125 frac[27,1] + mem[27,1] >= 0
 accuracy_(1,_sk_0)[28]: - 21.23828125 frac[28,1] + mem[28,1] >= 0
 accuracy_(1,_sk_0)[29]: - 21.23828125 frac[29,1] + mem[29,1] >= 0
 accuracy_(1,_sk_0)[30]: - 21.23828125 frac[30,1] + mem[30,1] >= 0
 accuracy_(1,_sk_0)[31]: - 21.23828125 frac[31,1] + mem[31,1] >= 0
 accuracy_(1,_sk_0)[32]: - 21.23828125 frac[32,1] + mem[32,1] >= 0
 accuracy_(1,_sk_0)[33]: - 21.23828125 frac[33,1] + mem[33,1] >= 0
 accuracy_(1,_sk_0)[34]: - 21.23828125 frac[34,1] + mem[34,1] >= 0
 accuracy_(1,_sk_0)[35]: - 21.23828125 frac[35,1] + mem[35,1] >= 0
 accuracy_(1,_sk_0)[36]: - 21.23828125 frac[36,1] + mem[36,1] >= 0
 accuracy_(1,_sk_0)[37]: - 21.23828125 frac[37,1] + mem[37,1] >= 0
 accuracy_(1,_sk_0)[38]: - 21.23828125 frac[38,1] + mem[38,1] >= 0
 accuracy_(2,_sk_0)[0]: - 21.23828125 frac[0,2] + mem[0,2] >= 0
 accuracy_(2,_sk_0)[1]: - 21.23828125 frac[1,2] + mem[1,2] >= 0
 accuracy_(2,_sk_0)[2]: - 21.23828125 frac[2,2] + mem[2,2] >= 0
 accuracy_(2,_sk_0)[3]: - 21.23828125 frac[3,2] + mem[3,2] >= 0
 accuracy_(2,_sk_0)[4]: - 21.23828125 frac[4,2] + mem[4,2] >= 0
 accuracy_(2,_sk_0)[5]: - 21.23828125 frac[5,2] + mem[5,2] >= 0
 accuracy_(2,_sk_0)[6]: - 21.23828125 frac[6,2] + mem[6,2] >= 0
 accuracy_(2,_sk_0)[7]: - 21.23828125 frac[7,2] + mem[7,2] >= 0
 accuracy_(2,_sk_0)[8]: - 21.23828125 frac[8,2] + mem[8,2] >= 0
 accuracy_(2,_sk_0)[9]: - 21.23828125 frac[9,2] + mem[9,2] >= 0
 accuracy_(2,_sk_0)[10]: - 21.23828125 frac[10,2] + mem[10,2] >= 0
 accuracy_(2,_sk_0)[11]: - 21.23828125 frac[11,2] + mem[11,2] >= 0
 accuracy_(2,_sk_0)[12]: - 21.23828125 frac[12,2] + mem[12,2] >= 0
 accuracy_(2,_sk_0)[13]: - 21.23828125 frac[13,2] + mem[13,2] >= 0
 accuracy_(2,_sk_0)[14]: - 21.23828125 frac[14,2] + mem[14,2] >= 0
 accuracy_(2,_sk_0)[15]: - 21.23828125 frac[15,2] + mem[15,2] >= 0
 accuracy_(2,_sk_0)[16]: - 21.23828125 frac[16,2] + mem[16,2] >= 0
 accuracy_(2,_sk_0)[17]: - 21.23828125 frac[17,2] + mem[17,2] >= 0
 accuracy_(2,_sk_0)[18]: - 21.23828125 frac[18,2] + mem[18,2] >= 0
 accuracy_(2,_sk_0)[19]: - 21.23828125 frac[19,2] + mem[19,2] >= 0
 accuracy_(2,_sk_0)[20]: - 21.23828125 frac[20,2] + mem[20,2] >= 0
 accuracy_(2,_sk_0)[21]: - 21.23828125 frac[21,2] + mem[21,2] >= 0
 accuracy_(2,_sk_0)[22]: - 21.23828125 frac[22,2] + mem[22,2] >= 0
 accuracy_(2,_sk_0)[23]: - 21.23828125 frac[23,2] + mem[23,2] >= 0
 accuracy_(2,_sk_0)[24]: - 21.23828125 frac[24,2] + mem[24,2] >= 0
 accuracy_(2,_sk_0)[25]: - 21.23828125 frac[25,2] + mem[25,2] >= 0
 accuracy_(2,_sk_0)[26]: - 21.23828125 frac[26,2] + mem[26,2] >= 0
 accuracy_(2,_sk_0)[27]: - 21.23828125 frac[27,2] + mem[27,2] >= 0
 accuracy_(2,_sk_0)[28]: - 21.23828125 frac[28,2] + mem[28,2] >= 0
 accuracy_(2,_sk_0)[29]: - 21.23828125 frac[29,2] + mem[29,2] >= 0
 accuracy_(2,_sk_0)[30]: - 21.23828125 frac[30,2] + mem[30,2] >= 0
 accuracy_(2,_sk_0)[31]: - 21.23828125 frac[31,2] + mem[31,2] >= 0
 accuracy_(2,_sk_0)[32]: - 21.23828125 frac[32,2] + mem[32,2] >= 0
 accuracy_(2,_sk_0)[33]: - 21.23828125 frac[33,2] + mem[33,2] >= 0
 accuracy_(2,_sk_0)[34]: - 21.23828125 frac[34,2] + mem[34,2] >= 0
 accuracy_(2,_sk_0)[35]: - 21.23828125 frac[35,2] + mem[35,2] >= 0
 accuracy_(2,_sk_0)[36]: - 21.23828125 frac[36,2] + mem[36,2] >= 0
 accuracy_(2,_sk_0)[37]: - 21.23828125 frac[37,2] + mem[37,2] >= 0
 accuracy_(2,_sk_0)[38]: - 21.23828125 frac[38,2] + mem[38,2] >= 0
 accuracy_(3,_sk_0)[0]: - 21.23828125 frac[0,3] + mem[0,3] >= 0
 accuracy_(3,_sk_0)[1]: - 21.23828125 frac[1,3] + mem[1,3] >= 0
 accuracy_(3,_sk_0)[2]: - 21.23828125 frac[2,3] + mem[2,3] >= 0
 accuracy_(3,_sk_0)[3]: - 21.23828125 frac[3,3] + mem[3,3] >= 0
 accuracy_(3,_sk_0)[4]: - 21.23828125 frac[4,3] + mem[4,3] >= 0
 accuracy_(3,_sk_0)[5]: - 21.23828125 frac[5,3] + mem[5,3] >= 0
 accuracy_(3,_sk_0)[6]: - 21.23828125 frac[6,3] + mem[6,3] >= 0
 accuracy_(3,_sk_0)[7]: - 21.23828125 frac[7,3] + mem[7,3] >= 0
 accuracy_(3,_sk_0)[8]: - 21.23828125 frac[8,3] + mem[8,3] >= 0
 accuracy_(3,_sk_0)[9]: - 21.23828125 frac[9,3] + mem[9,3] >= 0
 accuracy_(3,_sk_0)[10]: - 21.23828125 frac[10,3] + mem[10,3] >= 0
 accuracy_(3,_sk_0)[11]: - 21.23828125 frac[11,3] + mem[11,3] >= 0
 accuracy_(3,_sk_0)[12]: - 21.23828125 frac[12,3] + mem[12,3] >= 0
 accuracy_(3,_sk_0)[13]: - 21.23828125 frac[13,3] + mem[13,3] >= 0
 accuracy_(3,_sk_0)[14]: - 21.23828125 frac[14,3] + mem[14,3] >= 0
 accuracy_(3,_sk_0)[15]: - 21.23828125 frac[15,3] + mem[15,3] >= 0
 accuracy_(3,_sk_0)[16]: - 21.23828125 frac[16,3] + mem[16,3] >= 0
 accuracy_(3,_sk_0)[17]: - 21.23828125 frac[17,3] + mem[17,3] >= 0
 accuracy_(3,_sk_0)[18]: - 21.23828125 frac[18,3] + mem[18,3] >= 0
 accuracy_(3,_sk_0)[19]: - 21.23828125 frac[19,3] + mem[19,3] >= 0
 accuracy_(3,_sk_0)[20]: - 21.23828125 frac[20,3] + mem[20,3] >= 0
 accuracy_(3,_sk_0)[21]: - 21.23828125 frac[21,3] + mem[21,3] >= 0
 accuracy_(3,_sk_0)[22]: - 21.23828125 frac[22,3] + mem[22,3] >= 0
 accuracy_(3,_sk_0)[23]: - 21.23828125 frac[23,3] + mem[23,3] >= 0
 accuracy_(3,_sk_0)[24]: - 21.23828125 frac[24,3] + mem[24,3] >= 0
 accuracy_(3,_sk_0)[25]: - 21.23828125 frac[25,3] + mem[25,3] >= 0
 accuracy_(3,_sk_0)[26]: - 21.23828125 frac[26,3] + mem[26,3] >= 0
 accuracy_(3,_sk_0)[27]: - 21.23828125 frac[27,3] + mem[27,3] >= 0
 accuracy_(3,_sk_0)[28]: - 21.23828125 frac[28,3] + mem[28,3] >= 0
 accuracy_(3,_sk_0)[29]: - 21.23828125 frac[29,3] + mem[29,3] >= 0
 accuracy_(3,_sk_0)[30]: - 21.23828125 frac[30,3] + mem[30,3] >= 0
 accuracy_(3,_sk_0)[31]: - 21.23828125 frac[31,3] + mem[31,3] >= 0
 accuracy_(3,_sk_0)[32]: - 21.23828125 frac[32,3] + mem[32,3] >= 0
 accuracy_(3,_sk_0)[33]: - 21.23828125 frac[33,3] + mem[33,3] >= 0
 accuracy_(3,_sk_0)[34]: - 21.23828125 frac[34,3] + mem[34,3] >= 0
 accuracy_(3,_sk_0)[35]: - 21.23828125 frac[35,3] + mem[35,3] >= 0
 accuracy_(3,_sk_0)[36]: - 21.23828125 frac[36,3] + mem[36,3] >= 0
 accuracy_(3,_sk_0)[37]: - 21.23828125 frac[37,3] + mem[37,3] >= 0
 accuracy_(3,_sk_0)[38]: - 21.23828125 frac[38,3] + mem[38,3] >= 0
 accuracy_(0,_sk_1)[0]: - 5309.14453125 frac[0,4] + mem[0,4] >= 0
 accuracy_(0,_sk_1)[1]: - 5309.14453125 frac[1,4] + mem[1,4] >= 0
 accuracy_(0,_sk_1)[2]: - 5309.14453125 frac[2,4] + mem[2,4] >= 0
 accuracy_(0,_sk_1)[3]: - 5309.14453125 frac[3,4] + mem[3,4] >= 0
 accuracy_(0,_sk_1)[4]: - 5309.14453125 frac[4,4] + mem[4,4] >= 0
 accuracy_(0,_sk_1)[5]: - 5309.14453125 frac[5,4] + mem[5,4] >= 0
 accuracy_(0,_sk_1)[6]: - 5309.14453125 frac[6,4] + mem[6,4] >= 0
 accuracy_(0,_sk_1)[7]: - 5309.14453125 frac[7,4] + mem[7,4] >= 0
 accuracy_(0,_sk_1)[8]: - 5309.14453125 frac[8,4] + mem[8,4] >= 0
 accuracy_(0,_sk_1)[9]: - 5309.14453125 frac[9,4] + mem[9,4] >= 0
 accuracy_(0,_sk_1)[10]: - 5309.14453125 frac[10,4] + mem[10,4] >= 0
 accuracy_(0,_sk_1)[11]: - 5309.14453125 frac[11,4] + mem[11,4] >= 0
 accuracy_(0,_sk_1)[12]: - 5309.14453125 frac[12,4] + mem[12,4] >= 0
 accuracy_(0,_sk_1)[13]: - 5309.14453125 frac[13,4] + mem[13,4] >= 0
 accuracy_(0,_sk_1)[14]: - 5309.14453125 frac[14,4] + mem[14,4] >= 0
 accuracy_(0,_sk_1)[15]: - 5309.14453125 frac[15,4] + mem[15,4] >= 0
 accuracy_(0,_sk_1)[16]: - 5309.14453125 frac[16,4] + mem[16,4] >= 0
 accuracy_(0,_sk_1)[17]: - 5309.14453125 frac[17,4] + mem[17,4] >= 0
 accuracy_(0,_sk_1)[18]: - 5309.14453125 frac[18,4] + mem[18,4] >= 0
 accuracy_(0,_sk_1)[19]: - 5309.14453125 frac[19,4] + mem[19,4] >= 0
 accuracy_(0,_sk_1)[20]: - 5309.14453125 frac[20,4] + mem[20,4] >= 0
 accuracy_(0,_sk_1)[21]: - 5309.14453125 frac[21,4] + mem[21,4] >= 0
 accuracy_(0,_sk_1)[22]: - 5309.14453125 frac[22,4] + mem[22,4] >= 0
 accuracy_(0,_sk_1)[23]: - 5309.14453125 frac[23,4] + mem[23,4] >= 0
 accuracy_(0,_sk_1)[24]: - 5309.14453125 frac[24,4] + mem[24,4] >= 0
 accuracy_(0,_sk_1)[25]: - 5309.14453125 frac[25,4] + mem[25,4] >= 0
 accuracy_(0,_sk_1)[26]: - 5309.14453125 frac[26,4] + mem[26,4] >= 0
 accuracy_(0,_sk_1)[27]: - 5309.14453125 frac[27,4] + mem[27,4] >= 0
 accuracy_(0,_sk_1)[28]: - 5309.14453125 frac[28,4] + mem[28,4] >= 0
 accuracy_(0,_sk_1)[29]: - 5309.14453125 frac[29,4] + mem[29,4] >= 0
 accuracy_(0,_sk_1)[30]: - 5309.14453125 frac[30,4] + mem[30,4] >= 0
 accuracy_(0,_sk_1)[31]: - 5309.14453125 frac[31,4] + mem[31,4] >= 0
 accuracy_(0,_sk_1)[32]: - 5309.14453125 frac[32,4] + mem[32,4] >= 0
 accuracy_(0,_sk_1)[33]: - 5309.14453125 frac[33,4] + mem[33,4] >= 0
 accuracy_(0,_sk_1)[34]: - 5309.14453125 frac[34,4] + mem[34,4] >= 0
 accuracy_(0,_sk_1)[35]: - 5309.14453125 frac[35,4] + mem[35,4] >= 0
 accuracy_(0,_sk_1)[36]: - 5309.14453125 frac[36,4] + mem[36,4] >= 0
 accuracy_(0,_sk_1)[37]: - 5309.14453125 frac[37,4] + mem[37,4] >= 0
 accuracy_(0,_sk_1)[38]: - 5309.14453125 frac[38,4] + mem[38,4] >= 0
 accuracy_(1,_sk_1)[0]: - 5309.14453125 frac[0,5] + mem[0,5] >= 0
 accuracy_(1,_sk_1)[1]: - 5309.14453125 frac[1,5] + mem[1,5] >= 0
 accuracy_(1,_sk_1)[2]: - 5309.14453125 frac[2,5] + mem[2,5] >= 0
 accuracy_(1,_sk_1)[3]: - 5309.14453125 frac[3,5] + mem[3,5] >= 0
 accuracy_(1,_sk_1)[4]: - 5309.14453125 frac[4,5] + mem[4,5] >= 0
 accuracy_(1,_sk_1)[5]: - 5309.14453125 frac[5,5] + mem[5,5] >= 0
 accuracy_(1,_sk_1)[6]: - 5309.14453125 frac[6,5] + mem[6,5] >= 0
 accuracy_(1,_sk_1)[7]: - 5309.14453125 frac[7,5] + mem[7,5] >= 0
 accuracy_(1,_sk_1)[8]: - 5309.14453125 frac[8,5] + mem[8,5] >= 0
 accuracy_(1,_sk_1)[9]: - 5309.14453125 frac[9,5] + mem[9,5] >= 0
 accuracy_(1,_sk_1)[10]: - 5309.14453125 frac[10,5] + mem[10,5] >= 0
 accuracy_(1,_sk_1)[11]: - 5309.14453125 frac[11,5] + mem[11,5] >= 0
 accuracy_(1,_sk_1)[12]: - 5309.14453125 frac[12,5] + mem[12,5] >= 0
 accuracy_(1,_sk_1)[13]: - 5309.14453125 frac[13,5] + mem[13,5] >= 0
 accuracy_(1,_sk_1)[14]: - 5309.14453125 frac[14,5] + mem[14,5] >= 0
 accuracy_(1,_sk_1)[15]: - 5309.14453125 frac[15,5] + mem[15,5] >= 0
 accuracy_(1,_sk_1)[16]: - 5309.14453125 frac[16,5] + mem[16,5] >= 0
 accuracy_(1,_sk_1)[17]: - 5309.14453125 frac[17,5] + mem[17,5] >= 0
 accuracy_(1,_sk_1)[18]: - 5309.14453125 frac[18,5] + mem[18,5] >= 0
 accuracy_(1,_sk_1)[19]: - 5309.14453125 frac[19,5] + mem[19,5] >= 0
 accuracy_(1,_sk_1)[20]: - 5309.14453125 frac[20,5] + mem[20,5] >= 0
 accuracy_(1,_sk_1)[21]: - 5309.14453125 frac[21,5] + mem[21,5] >= 0
 accuracy_(1,_sk_1)[22]: - 5309.14453125 frac[22,5] + mem[22,5] >= 0
 accuracy_(1,_sk_1)[23]: - 5309.14453125 frac[23,5] + mem[23,5] >= 0
 accuracy_(1,_sk_1)[24]: - 5309.14453125 frac[24,5] + mem[24,5] >= 0
 accuracy_(1,_sk_1)[25]: - 5309.14453125 frac[25,5] + mem[25,5] >= 0
 accuracy_(1,_sk_1)[26]: - 5309.14453125 frac[26,5] + mem[26,5] >= 0
 accuracy_(1,_sk_1)[27]: - 5309.14453125 frac[27,5] + mem[27,5] >= 0
 accuracy_(1,_sk_1)[28]: - 5309.14453125 frac[28,5] + mem[28,5] >= 0
 accuracy_(1,_sk_1)[29]: - 5309.14453125 frac[29,5] + mem[29,5] >= 0
 accuracy_(1,_sk_1)[30]: - 5309.14453125 frac[30,5] + mem[30,5] >= 0
 accuracy_(1,_sk_1)[31]: - 5309.14453125 frac[31,5] + mem[31,5] >= 0
 accuracy_(1,_sk_1)[32]: - 5309.14453125 frac[32,5] + mem[32,5] >= 0
 accuracy_(1,_sk_1)[33]: - 5309.14453125 frac[33,5] + mem[33,5] >= 0
 accuracy_(1,_sk_1)[34]: - 5309.14453125 frac[34,5] + mem[34,5] >= 0
 accuracy_(1,_sk_1)[35]: - 5309.14453125 frac[35,5] + mem[35,5] >= 0
 accuracy_(1,_sk_1)[36]: - 5309.14453125 frac[36,5] + mem[36,5] >= 0
 accuracy_(1,_sk_1)[37]: - 5309.14453125 frac[37,5] + mem[37,5] >= 0
 accuracy_(1,_sk_1)[38]: - 5309.14453125 frac[38,5] + mem[38,5] >= 0
 accuracy_(2,_sk_1)[0]: - 5309.14453125 frac[0,6] + mem[0,6] >= 0
 accuracy_(2,_sk_1)[1]: - 5309.14453125 frac[1,6] + mem[1,6] >= 0
 accuracy_(2,_sk_1)[2]: - 5309.14453125 frac[2,6] + mem[2,6] >= 0
 accuracy_(2,_sk_1)[3]: - 5309.14453125 frac[3,6] + mem[3,6] >= 0
 accuracy_(2,_sk_1)[4]: - 5309.14453125 frac[4,6] + mem[4,6] >= 0
 accuracy_(2,_sk_1)[5]: - 5309.14453125 frac[5,6] + mem[5,6] >= 0
 accuracy_(2,_sk_1)[6]: - 5309.14453125 frac[6,6] + mem[6,6] >= 0
 accuracy_(2,_sk_1)[7]: - 5309.14453125 frac[7,6] + mem[7,6] >= 0
 accuracy_(2,_sk_1)[8]: - 5309.14453125 frac[8,6] + mem[8,6] >= 0
 accuracy_(2,_sk_1)[9]: - 5309.14453125 frac[9,6] + mem[9,6] >= 0
 accuracy_(2,_sk_1)[10]: - 5309.14453125 frac[10,6] + mem[10,6] >= 0
 accuracy_(2,_sk_1)[11]: - 5309.14453125 frac[11,6] + mem[11,6] >= 0
 accuracy_(2,_sk_1)[12]: - 5309.14453125 frac[12,6] + mem[12,6] >= 0
 accuracy_(2,_sk_1)[13]: - 5309.14453125 frac[13,6] + mem[13,6] >= 0
 accuracy_(2,_sk_1)[14]: - 5309.14453125 frac[14,6] + mem[14,6] >= 0
 accuracy_(2,_sk_1)[15]: - 5309.14453125 frac[15,6] + mem[15,6] >= 0
 accuracy_(2,_sk_1)[16]: - 5309.14453125 frac[16,6] + mem[16,6] >= 0
 accuracy_(2,_sk_1)[17]: - 5309.14453125 frac[17,6] + mem[17,6] >= 0
 accuracy_(2,_sk_1)[18]: - 5309.14453125 frac[18,6] + mem[18,6] >= 0
 accuracy_(2,_sk_1)[19]: - 5309.14453125 frac[19,6] + mem[19,6] >= 0
 accuracy_(2,_sk_1)[20]: - 5309.14453125 frac[20,6] + mem[20,6] >= 0
 accuracy_(2,_sk_1)[21]: - 5309.14453125 frac[21,6] + mem[21,6] >= 0
 accuracy_(2,_sk_1)[22]: - 5309.14453125 frac[22,6] + mem[22,6] >= 0
 accuracy_(2,_sk_1)[23]: - 5309.14453125 frac[23,6] + mem[23,6] >= 0
 accuracy_(2,_sk_1)[24]: - 5309.14453125 frac[24,6] + mem[24,6] >= 0
 accuracy_(2,_sk_1)[25]: - 5309.14453125 frac[25,6] + mem[25,6] >= 0
 accuracy_(2,_sk_1)[26]: - 5309.14453125 frac[26,6] + mem[26,6] >= 0
 accuracy_(2,_sk_1)[27]: - 5309.14453125 frac[27,6] + mem[27,6] >= 0
 accuracy_(2,_sk_1)[28]: - 5309.14453125 frac[28,6] + mem[28,6] >= 0
 accuracy_(2,_sk_1)[29]: - 5309.14453125 frac[29,6] + mem[29,6] >= 0
 accuracy_(2,_sk_1)[30]: - 5309.14453125 frac[30,6] + mem[30,6] >= 0
 accuracy_(2,_sk_1)[31]: - 5309.14453125 frac[31,6] + mem[31,6] >= 0
 accuracy_(2,_sk_1)[32]: - 5309.14453125 frac[32,6] + mem[32,6] >= 0
 accuracy_(2,_sk_1)[33]: - 5309.14453125 frac[33,6] + mem[33,6] >= 0
 accuracy_(2,_sk_1)[34]: - 5309.14453125 frac[34,6] + mem[34,6] >= 0
 accuracy_(2,_sk_1)[35]: - 5309.14453125 frac[35,6] + mem[35,6] >= 0
 accuracy_(2,_sk_1)[36]: - 5309.14453125 frac[36,6] + mem[36,6] >= 0
 accuracy_(2,_sk_1)[37]: - 5309.14453125 frac[37,6] + mem[37,6] >= 0
 accuracy_(2,_sk_1)[38]: - 5309.14453125 frac[38,6] + mem[38,6] >= 0
 accuracy_(3,_sk_1)[0]: - 5309.14453125 frac[0,7] + mem[0,7] >= 0
 accuracy_(3,_sk_1)[1]: - 5309.14453125 frac[1,7] + mem[1,7] >= 0
 accuracy_(3,_sk_1)[2]: - 5309.14453125 frac[2,7] + mem[2,7] >= 0
 accuracy_(3,_sk_1)[3]: - 5309.14453125 frac[3,7] + mem[3,7] >= 0
 accuracy_(3,_sk_1)[4]: - 5309.14453125 frac[4,7] + mem[4,7] >= 0
 accuracy_(3,_sk_1)[5]: - 5309.14453125 frac[5,7] + mem[5,7] >= 0
 accuracy_(3,_sk_1)[6]: - 5309.14453125 frac[6,7] + mem[6,7] >= 0
 accuracy_(3,_sk_1)[7]: - 5309.14453125 frac[7,7] + mem[7,7] >= 0
 accuracy_(3,_sk_1)[8]: - 5309.14453125 frac[8,7] + mem[8,7] >= 0
 accuracy_(3,_sk_1)[9]: - 5309.14453125 frac[9,7] + mem[9,7] >= 0
 accuracy_(3,_sk_1)[10]: - 5309.14453125 frac[10,7] + mem[10,7] >= 0
 accuracy_(3,_sk_1)[11]: - 5309.14453125 frac[11,7] + mem[11,7] >= 0
 accuracy_(3,_sk_1)[12]: - 5309.14453125 frac[12,7] + mem[12,7] >= 0
 accuracy_(3,_sk_1)[13]: - 5309.14453125 frac[13,7] + mem[13,7] >= 0
 accuracy_(3,_sk_1)[14]: - 5309.14453125 frac[14,7] + mem[14,7] >= 0
 accuracy_(3,_sk_1)[15]: - 5309.14453125 frac[15,7] + mem[15,7] >= 0
 accuracy_(3,_sk_1)[16]: - 5309.14453125 frac[16,7] + mem[16,7] >= 0
 accuracy_(3,_sk_1)[17]: - 5309.14453125 frac[17,7] + mem[17,7] >= 0
 accuracy_(3,_sk_1)[18]: - 5309.14453125 frac[18,7] + mem[18,7] >= 0
 accuracy_(3,_sk_1)[19]: - 5309.14453125 frac[19,7] + mem[19,7] >= 0
 accuracy_(3,_sk_1)[20]: - 5309.14453125 frac[20,7] + mem[20,7] >= 0
 accuracy_(3,_sk_1)[21]: - 5309.14453125 frac[21,7] + mem[21,7] >= 0
 accuracy_(3,_sk_1)[22]: - 5309.14453125 frac[22,7] + mem[22,7] >= 0
 accuracy_(3,_sk_1)[23]: - 5309.14453125 frac[23,7] + mem[23,7] >= 0
 accuracy_(3,_sk_1)[24]: - 5309.14453125 frac[24,7] + mem[24,7] >= 0
 accuracy_(3,_sk_1)[25]: - 5309.14453125 frac[25,7] + mem[25,7] >= 0
 accuracy_(3,_sk_1)[26]: - 5309.14453125 frac[26,7] + mem[26,7] >= 0
 accuracy_(3,_sk_1)[27]: - 5309.14453125 frac[27,7] + mem[27,7] >= 0
 accuracy_(3,_sk_1)[28]: - 5309.14453125 frac[28,7] + mem[28,7] >= 0
 accuracy_(3,_sk_1)[29]: - 5309.14453125 frac[29,7] + mem[29,7] >= 0
 accuracy_(3,_sk_1)[30]: - 5309.14453125 frac[30,7] + mem[30,7] >= 0
 accuracy_(3,_sk_1)[31]: - 5309.14453125 frac[31,7] + mem[31,7] >= 0
 accuracy_(3,_sk_1)[32]: - 5309.14453125 frac[32,7] + mem[32,7] >= 0
 accuracy_(3,_sk_1)[33]: - 5309.14453125 frac[33,7] + mem[33,7] >= 0
 accuracy_(3,_sk_1)[34]: - 5309.14453125 frac[34,7] + mem[34,7] >= 0
 accuracy_(3,_sk_1)[35]: - 5309.14453125 frac[35,7] + mem[35,7] >= 0
 accuracy_(3,_sk_1)[36]: - 5309.14453125 frac[36,7] + mem[36,7] >= 0
 accuracy_(3,_sk_1)[37]: - 5309.14453125 frac[37,7] + mem[37,7] >= 0
 accuracy_(3,_sk_1)[38]: - 5309.14453125 frac[38,7] + mem[38,7] >= 0
 accuracy_(0,_sk_2)[0]: - 10.62109375 frac[0,8] + mem[0,8] >= 0
 accuracy_(0,_sk_2)[1]: - 10.62109375 frac[1,8] + mem[1,8] >= 0
 accuracy_(0,_sk_2)[2]: - 10.62109375 frac[2,8] + mem[2,8] >= 0
 accuracy_(0,_sk_2)[3]: - 10.62109375 frac[3,8] + mem[3,8] >= 0
 accuracy_(0,_sk_2)[4]: - 10.62109375 frac[4,8] + mem[4,8] >= 0
 accuracy_(0,_sk_2)[5]: - 10.62109375 frac[5,8] + mem[5,8] >= 0
 accuracy_(0,_sk_2)[6]: - 10.62109375 frac[6,8] + mem[6,8] >= 0
 accuracy_(0,_sk_2)[7]: - 10.62109375 frac[7,8] + mem[7,8] >= 0
 accuracy_(0,_sk_2)[8]: - 10.62109375 frac[8,8] + mem[8,8] >= 0
 accuracy_(0,_sk_2)[9]: - 10.62109375 frac[9,8] + mem[9,8] >= 0
 accuracy_(0,_sk_2)[10]: - 10.62109375 frac[10,8] + mem[10,8] >= 0
 accuracy_(0,_sk_2)[11]: - 10.62109375 frac[11,8] + mem[11,8] >= 0
 accuracy_(0,_sk_2)[12]: - 10.62109375 frac[12,8] + mem[12,8] >= 0
 accuracy_(0,_sk_2)[13]: - 10.62109375 frac[13,8] + mem[13,8] >= 0
 accuracy_(0,_sk_2)[14]: - 10.62109375 frac[14,8] + mem[14,8] >= 0
 accuracy_(0,_sk_2)[15]: - 10.62109375 frac[15,8] + mem[15,8] >= 0
 accuracy_(0,_sk_2)[16]: - 10.62109375 frac[16,8] + mem[16,8] >= 0
 accuracy_(0,_sk_2)[17]: - 10.62109375 frac[17,8] + mem[17,8] >= 0
 accuracy_(0,_sk_2)[18]: - 10.62109375 frac[18,8] + mem[18,8] >= 0
 accuracy_(0,_sk_2)[19]: - 10.62109375 frac[19,8] + mem[19,8] >= 0
 accuracy_(0,_sk_2)[20]: - 10.62109375 frac[20,8] + mem[20,8] >= 0
 accuracy_(0,_sk_2)[21]: - 10.62109375 frac[21,8] + mem[21,8] >= 0
 accuracy_(0,_sk_2)[22]: - 10.62109375 frac[22,8] + mem[22,8] >= 0
 accuracy_(0,_sk_2)[23]: - 10.62109375 frac[23,8] + mem[23,8] >= 0
 accuracy_(0,_sk_2)[24]: - 10.62109375 frac[24,8] + mem[24,8] >= 0
 accuracy_(0,_sk_2)[25]: - 10.62109375 frac[25,8] + mem[25,8] >= 0
 accuracy_(0,_sk_2)[26]: - 10.62109375 frac[26,8] + mem[26,8] >= 0
 accuracy_(0,_sk_2)[27]: - 10.62109375 frac[27,8] + mem[27,8] >= 0
 accuracy_(0,_sk_2)[28]: - 10.62109375 frac[28,8] + mem[28,8] >= 0
 accuracy_(0,_sk_2)[29]: - 10.62109375 frac[29,8] + mem[29,8] >= 0
 accuracy_(0,_sk_2)[30]: - 10.62109375 frac[30,8] + mem[30,8] >= 0
 accuracy_(0,_sk_2)[31]: - 10.62109375 frac[31,8] + mem[31,8] >= 0
 accuracy_(0,_sk_2)[32]: - 10.62109375 frac[32,8] + mem[32,8] >= 0
 accuracy_(0,_sk_2)[33]: - 10.62109375 frac[33,8] + mem[33,8] >= 0
 accuracy_(0,_sk_2)[34]: - 10.62109375 frac[34,8] + mem[34,8] >= 0
 accuracy_(0,_sk_2)[35]: - 10.62109375 frac[35,8] + mem[35,8] >= 0
 accuracy_(0,_sk_2)[36]: - 10.62109375 frac[36,8] + mem[36,8] >= 0
 accuracy_(0,_sk_2)[37]: - 10.62109375 frac[37,8] + mem[37,8] >= 0
 accuracy_(0,_sk_2)[38]: - 10.62109375 frac[38,8] + mem[38,8] >= 0
 accuracy_(1,_sk_2)[0]: - 10.62109375 frac[0,9] + mem[0,9] >= 0
 accuracy_(1,_sk_2)[1]: - 10.62109375 frac[1,9] + mem[1,9] >= 0
 accuracy_(1,_sk_2)[2]: - 10.62109375 frac[2,9] + mem[2,9] >= 0
 accuracy_(1,_sk_2)[3]: - 10.62109375 frac[3,9] + mem[3,9] >= 0
 accuracy_(1,_sk_2)[4]: - 10.62109375 frac[4,9] + mem[4,9] >= 0
 accuracy_(1,_sk_2)[5]: - 10.62109375 frac[5,9] + mem[5,9] >= 0
 accuracy_(1,_sk_2)[6]: - 10.62109375 frac[6,9] + mem[6,9] >= 0
 accuracy_(1,_sk_2)[7]: - 10.62109375 frac[7,9] + mem[7,9] >= 0
 accuracy_(1,_sk_2)[8]: - 10.62109375 frac[8,9] + mem[8,9] >= 0
 accuracy_(1,_sk_2)[9]: - 10.62109375 frac[9,9] + mem[9,9] >= 0
 accuracy_(1,_sk_2)[10]: - 10.62109375 frac[10,9] + mem[10,9] >= 0
 accuracy_(1,_sk_2)[11]: - 10.62109375 frac[11,9] + mem[11,9] >= 0
 accuracy_(1,_sk_2)[12]: - 10.62109375 frac[12,9] + mem[12,9] >= 0
 accuracy_(1,_sk_2)[13]: - 10.62109375 frac[13,9] + mem[13,9] >= 0
 accuracy_(1,_sk_2)[14]: - 10.62109375 frac[14,9] + mem[14,9] >= 0
 accuracy_(1,_sk_2)[15]: - 10.62109375 frac[15,9] + mem[15,9] >= 0
 accuracy_(1,_sk_2)[16]: - 10.62109375 frac[16,9] + mem[16,9] >= 0
 accuracy_(1,_sk_2)[17]: - 10.62109375 frac[17,9] + mem[17,9] >= 0
 accuracy_(1,_sk_2)[18]: - 10.62109375 frac[18,9] + mem[18,9] >= 0
 accuracy_(1,_sk_2)[19]: - 10.62109375 frac[19,9] + mem[19,9] >= 0
 accuracy_(1,_sk_2)[20]: - 10.62109375 frac[20,9] + mem[20,9] >= 0
 accuracy_(1,_sk_2)[21]: - 10.62109375 frac[21,9] + mem[21,9] >= 0
 accuracy_(1,_sk_2)[22]: - 10.62109375 frac[22,9] + mem[22,9] >= 0
 accuracy_(1,_sk_2)[23]: - 10.62109375 frac[23,9] + mem[23,9] >= 0
 accuracy_(1,_sk_2)[24]: - 10.62109375 frac[24,9] + mem[24,9] >= 0
 accuracy_(1,_sk_2)[25]: - 10.62109375 frac[25,9] + mem[25,9] >= 0
 accuracy_(1,_sk_2)[26]: - 10.62109375 frac[26,9] + mem[26,9] >= 0
 accuracy_(1,_sk_2)[27]: - 10.62109375 frac[27,9] + mem[27,9] >= 0
 accuracy_(1,_sk_2)[28]: - 10.62109375 frac[28,9] + mem[28,9] >= 0
 accuracy_(1,_sk_2)[29]: - 10.62109375 frac[29,9] + mem[29,9] >= 0
 accuracy_(1,_sk_2)[30]: - 10.62109375 frac[30,9] + mem[30,9] >= 0
 accuracy_(1,_sk_2)[31]: - 10.62109375 frac[31,9] + mem[31,9] >= 0
 accuracy_(1,_sk_2)[32]: - 10.62109375 frac[32,9] + mem[32,9] >= 0
 accuracy_(1,_sk_2)[33]: - 10.62109375 frac[33,9] + mem[33,9] >= 0
 accuracy_(1,_sk_2)[34]: - 10.62109375 frac[34,9] + mem[34,9] >= 0
 accuracy_(1,_sk_2)[35]: - 10.62109375 frac[35,9] + mem[35,9] >= 0
 accuracy_(1,_sk_2)[36]: - 10.62109375 frac[36,9] + mem[36,9] >= 0
 accuracy_(1,_sk_2)[37]: - 10.62109375 frac[37,9] + mem[37,9] >= 0
 accuracy_(1,_sk_2)[38]: - 10.62109375 frac[38,9] + mem[38,9] >= 0
 accuracy_(2,_sk_2)[0]: - 10.62109375 frac[0,10] + mem[0,10] >= 0
 accuracy_(2,_sk_2)[1]: - 10.62109375 frac[1,10] + mem[1,10] >= 0
 accuracy_(2,_sk_2)[2]: - 10.62109375 frac[2,10] + mem[2,10] >= 0
 accuracy_(2,_sk_2)[3]: - 10.62109375 frac[3,10] + mem[3,10] >= 0
 accuracy_(2,_sk_2)[4]: - 10.62109375 frac[4,10] + mem[4,10] >= 0
 accuracy_(2,_sk_2)[5]: - 10.62109375 frac[5,10] + mem[5,10] >= 0
 accuracy_(2,_sk_2)[6]: - 10.62109375 frac[6,10] + mem[6,10] >= 0
 accuracy_(2,_sk_2)[7]: - 10.62109375 frac[7,10] + mem[7,10] >= 0
 accuracy_(2,_sk_2)[8]: - 10.62109375 frac[8,10] + mem[8,10] >= 0
 accuracy_(2,_sk_2)[9]: - 10.62109375 frac[9,10] + mem[9,10] >= 0
 accuracy_(2,_sk_2)[10]: - 10.62109375 frac[10,10] + mem[10,10] >= 0
 accuracy_(2,_sk_2)[11]: - 10.62109375 frac[11,10] + mem[11,10] >= 0
 accuracy_(2,_sk_2)[12]: - 10.62109375 frac[12,10] + mem[12,10] >= 0
 accuracy_(2,_sk_2)[13]: - 10.62109375 frac[13,10] + mem[13,10] >= 0
 accuracy_(2,_sk_2)[14]: - 10.62109375 frac[14,10] + mem[14,10] >= 0
 accuracy_(2,_sk_2)[15]: - 10.62109375 frac[15,10] + mem[15,10] >= 0
 accuracy_(2,_sk_2)[16]: - 10.62109375 frac[16,10] + mem[16,10] >= 0
 accuracy_(2,_sk_2)[17]: - 10.62109375 frac[17,10] + mem[17,10] >= 0
 accuracy_(2,_sk_2)[18]: - 10.62109375 frac[18,10] + mem[18,10] >= 0
 accuracy_(2,_sk_2)[19]: - 10.62109375 frac[19,10] + mem[19,10] >= 0
 accuracy_(2,_sk_2)[20]: - 10.62109375 frac[20,10] + mem[20,10] >= 0
 accuracy_(2,_sk_2)[21]: - 10.62109375 frac[21,10] + mem[21,10] >= 0
 accuracy_(2,_sk_2)[22]: - 10.62109375 frac[22,10] + mem[22,10] >= 0
 accuracy_(2,_sk_2)[23]: - 10.62109375 frac[23,10] + mem[23,10] >= 0
 accuracy_(2,_sk_2)[24]: - 10.62109375 frac[24,10] + mem[24,10] >= 0
 accuracy_(2,_sk_2)[25]: - 10.62109375 frac[25,10] + mem[25,10] >= 0
 accuracy_(2,_sk_2)[26]: - 10.62109375 frac[26,10] + mem[26,10] >= 0
 accuracy_(2,_sk_2)[27]: - 10.62109375 frac[27,10] + mem[27,10] >= 0
 accuracy_(2,_sk_2)[28]: - 10.62109375 frac[28,10] + mem[28,10] >= 0
 accuracy_(2,_sk_2)[29]: - 10.62109375 frac[29,10] + mem[29,10] >= 0
 accuracy_(2,_sk_2)[30]: - 10.62109375 frac[30,10] + mem[30,10] >= 0
 accuracy_(2,_sk_2)[31]: - 10.62109375 frac[31,10] + mem[31,10] >= 0
 accuracy_(2,_sk_2)[32]: - 10.62109375 frac[32,10] + mem[32,10] >= 0
 accuracy_(2,_sk_2)[33]: - 10.62109375 frac[33,10] + mem[33,10] >= 0
 accuracy_(2,_sk_2)[34]: - 10.62109375 frac[34,10] + mem[34,10] >= 0
 accuracy_(2,_sk_2)[35]: - 10.62109375 frac[35,10] + mem[35,10] >= 0
 accuracy_(2,_sk_2)[36]: - 10.62109375 frac[36,10] + mem[36,10] >= 0
 accuracy_(2,_sk_2)[37]: - 10.62109375 frac[37,10] + mem[37,10] >= 0
 accuracy_(2,_sk_2)[38]: - 10.62109375 frac[38,10] + mem[38,10] >= 0
 accuracy_(3,_sk_2)[0]: - 10.62109375 frac[0,11] + mem[0,11] >= 0
 accuracy_(3,_sk_2)[1]: - 10.62109375 frac[1,11] + mem[1,11] >= 0
 accuracy_(3,_sk_2)[2]: - 10.62109375 frac[2,11] + mem[2,11] >= 0
 accuracy_(3,_sk_2)[3]: - 10.62109375 frac[3,11] + mem[3,11] >= 0
 accuracy_(3,_sk_2)[4]: - 10.62109375 frac[4,11] + mem[4,11] >= 0
 accuracy_(3,_sk_2)[5]: - 10.62109375 frac[5,11] + mem[5,11] >= 0
 accuracy_(3,_sk_2)[6]: - 10.62109375 frac[6,11] + mem[6,11] >= 0
 accuracy_(3,_sk_2)[7]: - 10.62109375 frac[7,11] + mem[7,11] >= 0
 accuracy_(3,_sk_2)[8]: - 10.62109375 frac[8,11] + mem[8,11] >= 0
 accuracy_(3,_sk_2)[9]: - 10.62109375 frac[9,11] + mem[9,11] >= 0
 accuracy_(3,_sk_2)[10]: - 10.62109375 frac[10,11] + mem[10,11] >= 0
 accuracy_(3,_sk_2)[11]: - 10.62109375 frac[11,11] + mem[11,11] >= 0
 accuracy_(3,_sk_2)[12]: - 10.62109375 frac[12,11] + mem[12,11] >= 0
 accuracy_(3,_sk_2)[13]: - 10.62109375 frac[13,11] + mem[13,11] >= 0
 accuracy_(3,_sk_2)[14]: - 10.62109375 frac[14,11] + mem[14,11] >= 0
 accuracy_(3,_sk_2)[15]: - 10.62109375 frac[15,11] + mem[15,11] >= 0
 accuracy_(3,_sk_2)[16]: - 10.62109375 frac[16,11] + mem[16,11] >= 0
 accuracy_(3,_sk_2)[17]: - 10.62109375 frac[17,11] + mem[17,11] >= 0
 accuracy_(3,_sk_2)[18]: - 10.62109375 frac[18,11] + mem[18,11] >= 0
 accuracy_(3,_sk_2)[19]: - 10.62109375 frac[19,11] + mem[19,11] >= 0
 accuracy_(3,_sk_2)[20]: - 10.62109375 frac[20,11] + mem[20,11] >= 0
 accuracy_(3,_sk_2)[21]: - 10.62109375 frac[21,11] + mem[21,11] >= 0
 accuracy_(3,_sk_2)[22]: - 10.62109375 frac[22,11] + mem[22,11] >= 0
 accuracy_(3,_sk_2)[23]: - 10.62109375 frac[23,11] + mem[23,11] >= 0
 accuracy_(3,_sk_2)[24]: - 10.62109375 frac[24,11] + mem[24,11] >= 0
 accuracy_(3,_sk_2)[25]: - 10.62109375 frac[25,11] + mem[25,11] >= 0
 accuracy_(3,_sk_2)[26]: - 10.62109375 frac[26,11] + mem[26,11] >= 0
 accuracy_(3,_sk_2)[27]: - 10.62109375 frac[27,11] + mem[27,11] >= 0
 accuracy_(3,_sk_2)[28]: - 10.62109375 frac[28,11] + mem[28,11] >= 0
 accuracy_(3,_sk_2)[29]: - 10.62109375 frac[29,11] + mem[29,11] >= 0
 accuracy_(3,_sk_2)[30]: - 10.62109375 frac[30,11] + mem[30,11] >= 0
 accuracy_(3,_sk_2)[31]: - 10.62109375 frac[31,11] + mem[31,11] >= 0
 accuracy_(3,_sk_2)[32]: - 10.62109375 frac[32,11] + mem[32,11] >= 0
 accuracy_(3,_sk_2)[33]: - 10.62109375 frac[33,11] + mem[33,11] >= 0
 accuracy_(3,_sk_2)[34]: - 10.62109375 frac[34,11] + mem[34,11] >= 0
 accuracy_(3,_sk_2)[35]: - 10.62109375 frac[35,11] + mem[35,11] >= 0
 accuracy_(3,_sk_2)[36]: - 10.62109375 frac[36,11] + mem[36,11] >= 0
 accuracy_(3,_sk_2)[37]: - 10.62109375 frac[37,11] + mem[37,11] >= 0
 accuracy_(3,_sk_2)[38]: - 10.62109375 frac[38,11] + mem[38,11] >= 0
 accuracy_(4,_sk_2)[0]: - 10.62109375 frac[0,12] + mem[0,12] >= 0
 accuracy_(4,_sk_2)[1]: - 10.62109375 frac[1,12] + mem[1,12] >= 0
 accuracy_(4,_sk_2)[2]: - 10.62109375 frac[2,12] + mem[2,12] >= 0
 accuracy_(4,_sk_2)[3]: - 10.62109375 frac[3,12] + mem[3,12] >= 0
 accuracy_(4,_sk_2)[4]: - 10.62109375 frac[4,12] + mem[4,12] >= 0
 accuracy_(4,_sk_2)[5]: - 10.62109375 frac[5,12] + mem[5,12] >= 0
 accuracy_(4,_sk_2)[6]: - 10.62109375 frac[6,12] + mem[6,12] >= 0
 accuracy_(4,_sk_2)[7]: - 10.62109375 frac[7,12] + mem[7,12] >= 0
 accuracy_(4,_sk_2)[8]: - 10.62109375 frac[8,12] + mem[8,12] >= 0
 accuracy_(4,_sk_2)[9]: - 10.62109375 frac[9,12] + mem[9,12] >= 0
 accuracy_(4,_sk_2)[10]: - 10.62109375 frac[10,12] + mem[10,12] >= 0
 accuracy_(4,_sk_2)[11]: - 10.62109375 frac[11,12] + mem[11,12] >= 0
 accuracy_(4,_sk_2)[12]: - 10.62109375 frac[12,12] + mem[12,12] >= 0
 accuracy_(4,_sk_2)[13]: - 10.62109375 frac[13,12] + mem[13,12] >= 0
 accuracy_(4,_sk_2)[14]: - 10.62109375 frac[14,12] + mem[14,12] >= 0
 accuracy_(4,_sk_2)[15]: - 10.62109375 frac[15,12] + mem[15,12] >= 0
 accuracy_(4,_sk_2)[16]: - 10.62109375 frac[16,12] + mem[16,12] >= 0
 accuracy_(4,_sk_2)[17]: - 10.62109375 frac[17,12] + mem[17,12] >= 0
 accuracy_(4,_sk_2)[18]: - 10.62109375 frac[18,12] + mem[18,12] >= 0
 accuracy_(4,_sk_2)[19]: - 10.62109375 frac[19,12] + mem[19,12] >= 0
 accuracy_(4,_sk_2)[20]: - 10.62109375 frac[20,12] + mem[20,12] >= 0
 accuracy_(4,_sk_2)[21]: - 10.62109375 frac[21,12] + mem[21,12] >= 0
 accuracy_(4,_sk_2)[22]: - 10.62109375 frac[22,12] + mem[22,12] >= 0
 accuracy_(4,_sk_2)[23]: - 10.62109375 frac[23,12] + mem[23,12] >= 0
 accuracy_(4,_sk_2)[24]: - 10.62109375 frac[24,12] + mem[24,12] >= 0
 accuracy_(4,_sk_2)[25]: - 10.62109375 frac[25,12] + mem[25,12] >= 0
 accuracy_(4,_sk_2)[26]: - 10.62109375 frac[26,12] + mem[26,12] >= 0
 accuracy_(4,_sk_2)[27]: - 10.62109375 frac[27,12] + mem[27,12] >= 0
 accuracy_(4,_sk_2)[28]: - 10.62109375 frac[28,12] + mem[28,12] >= 0
 accuracy_(4,_sk_2)[29]: - 10.62109375 frac[29,12] + mem[29,12] >= 0
 accuracy_(4,_sk_2)[30]: - 10.62109375 frac[30,12] + mem[30,12] >= 0
 accuracy_(4,_sk_2)[31]: - 10.62109375 frac[31,12] + mem[31,12] >= 0
 accuracy_(4,_sk_2)[32]: - 10.62109375 frac[32,12] + mem[32,12] >= 0
 accuracy_(4,_sk_2)[33]: - 10.62109375 frac[33,12] + mem[33,12] >= 0
 accuracy_(4,_sk_2)[34]: - 10.62109375 frac[34,12] + mem[34,12] >= 0
 accuracy_(4,_sk_2)[35]: - 10.62109375 frac[35,12] + mem[35,12] >= 0
 accuracy_(4,_sk_2)[36]: - 10.62109375 frac[36,12] + mem[36,12] >= 0
 accuracy_(4,_sk_2)[37]: - 10.62109375 frac[37,12] + mem[37,12] >= 0
 accuracy_(4,_sk_2)[38]: - 10.62109375 frac[38,12] + mem[38,12] >= 0
 rows_tot_endhost_cpu1: - frac[0,0] - frac[0,1] - frac[0,2] - frac[0,3]
   - frac[0,4] - frac[0,5] - frac[0,6] - frac[0,7] - frac[0,8] - frac[0,9]
   - frac[0,10] - frac[0,11] - frac[0,12] + rows_tot_endhost_cpu1 = 0
 mem_tot_endhost_cpu1: - mem[0,0] - mem[0,1] - mem[0,2] - mem[0,3]
   - mem[0,4] - mem[0,5] - mem[0,6] - mem[0,7] - mem[0,8] - mem[0,9]
   - mem[0,10] - mem[0,11] - mem[0,12] + mem_tot_endhost_cpu1 = 0
 ns_single_endhost_cpu1: 3.5 rows_tot_endhost_cpu1 - ns_single_endhost_cpu1
   + pdt_m_rows_endhost_cpu1 <= 0
 capacity_cores_endhost_cpu1: cores_sketch_endhost_cpu1
   + cores_dpdk_endhost_cpu1 <= 7
 ns_sketch_endhost_cpu1: - ns_single_endhost_cpu1 + pdt_nsk_c_endhost_cpu1
   = 0
 ns_dpdk_endhost_cpu1: 7.142857142857143 cores_dpdk_endhost_cpu1
   - pdt_nsc_dpdk_endhost_cpu1 = -21.42857142857143
 rows_tot_endhost_cpu2: - frac[1,0] - frac[1,1] - frac[1,2] - frac[1,3]
   - frac[1,4] - frac[1,5] - frac[1,6] - frac[1,7] - frac[1,8] - frac[1,9]
   - frac[1,10] - frac[1,11] - frac[1,12] + rows_tot_endhost_cpu2 = 0
 mem_tot_endhost_cpu2: - mem[1,0] - mem[1,1] - mem[1,2] - mem[1,3]
   - mem[1,4] - mem[1,5] - mem[1,6] - mem[1,7] - mem[1,8] - mem[1,9]
   - mem[1,10] - mem[1,11] - mem[1,12] + mem_tot_endhost_cpu2 = 0
 ns_single_endhost_cpu2: 3.5 rows_tot_endhost_cpu2 - ns_single_endhost_cpu2
   + pdt_m_rows_endhost_cpu2 <= 0
 capacity_cores_endhost_cpu2: cores_sketch_endhost_cpu2
   + cores_dpdk_endhost_cpu2 <= 7
 ns_sketch_endhost_cpu2: - ns_single_endhost_cpu2 + pdt_nsk_c_endhost_cpu2
   = 0
 ns_dpdk_endhost_cpu2: 7.142857142857143 cores_dpdk_endhost_cpu2
   - pdt_nsc_dpdk_endhost_cpu2 = -21.42857142857143
 rows_tot_endhost_cpu3: - frac[2,0] - frac[2,1] - frac[2,2] - frac[2,3]
   - frac[2,4] - frac[2,5] - frac[2,6] - frac[2,7] - frac[2,8] - frac[2,9]
   - frac[2,10] - frac[2,11] - frac[2,12] + rows_tot_endhost_cpu3 = 0
 mem_tot_endhost_cpu3: - mem[2,0] - mem[2,1] - mem[2,2] - mem[2,3]
   - mem[2,4] - mem[2,5] - mem[2,6] - mem[2,7] - mem[2,8] - mem[2,9]
   - mem[2,10] - mem[2,11] - mem[2,12] + mem_tot_endhost_cpu3 = 0
 ns_single_endhost_cpu3: 3.5 rows_tot_endhost_cpu3 - ns_single_endhost_cpu3
   + pdt_m_rows_endhost_cpu3 <= 0
 capacity_cores_endhost_cpu3: cores_sketch_endhost_cpu3
   + cores_dpdk_endhost_cpu3 <= 7
 ns_sketch_endhost_cpu3: - ns_single_endhost_cpu3 + pdt_nsk_c_endhost_cpu3
   = 0
 ns_dpdk_endhost_cpu3: 7.142857142857143 cores_dpdk_endhost_cpu3
   - pdt_nsc_dpdk_endhost_cpu3 = -21.42857142857143
 rows_tot_endhost_cpu4: - frac[3,0] - frac[3,1] - frac[3,2] - frac[3,3]
   - frac[3,4] - frac[3,5] - frac[3,6] - frac[3,7] - frac[3,8] - frac[3,9]
   - frac[3,10] - frac[3,11] - frac[3,12] + rows_tot_endhost_cpu4 = 0
 mem_tot_endhost_cpu4: - mem[3,0] - mem[3,1] - mem[3,2] - mem[3,3]
   - mem[3,4] - mem[3,5] - mem[3,6] - mem[3,7] - mem[3,8] - mem[3,9]
   - mem[3,10] - mem[3,11] - mem[3,12] + mem_tot_endhost_cpu4 = 0
 ns_single_endhost_cpu4: 3.5 rows_tot_endhost_cpu4 - ns_single_endhost_cpu4
   + pdt_m_rows_endhost_cpu4 <= 0
 capacity_cores_endhost_cpu4: cores_sketch_endhost_cpu4
   + cores_dpdk_endhost_cpu4 <= 7
 ns_sketch_endhost_cpu4: - ns_single_endhost_cpu4 + pdt_nsk_c_endhost_cpu4
   = 0
 ns_dpdk_endhost_cpu4: 7.142857142857143 cores_dpdk_endhost_cpu4
   - pdt_nsc_dpdk_endhost_cpu4 = -21.42857142857143
 rows_tot_endhost_cpu5: - frac[4,0] - frac[4,1] - frac[4,2] - frac[4,3]
   - frac[4,4] - frac[4,5] - frac[4,6] - frac[4,7] - frac[4,8] - frac[4,9]
   - frac[4,10] - frac[4,11] - frac[4,12] + rows_tot_endhost_cpu5 = 0
 mem_tot_endhost_cpu5: - mem[4,0] - mem[4,1] - mem[4,2] - mem[4,3]
   - mem[4,4] - mem[4,5] - mem[4,6] - mem[4,7] - mem[4,8] - mem[4,9]
   - mem[4,10] - mem[4,11] - mem[4,12] + mem_tot_endhost_cpu5 = 0
 ns_single_endhost_cpu5: 3.5 rows_tot_endhost_cpu5 - ns_single_endhost_cpu5
   + pdt_m_rows_endhost_cpu5 <= 0
 capacity_cores_endhost_cpu5: cores_sketch_endhost_cpu5
   + cores_dpdk_endhost_cpu5 <= 7
 ns_sketch_endhost_cpu5: - ns_single_endhost_cpu5 + pdt_nsk_c_endhost_cpu5
   = 0
 ns_dpdk_endhost_cpu5: 7.142857142857143 cores_dpdk_endhost_cpu5
   - pdt_nsc_dpdk_endhost_cpu5 = -21.42857142857143
 rows_tot_endhost_cpu6: - frac[5,0] - frac[5,1] - frac[5,2] - frac[5,3]
   - frac[5,4] - frac[5,5] - frac[5,6] - frac[5,7] - frac[5,8] - frac[5,9]
   - frac[5,10] - frac[5,11] - frac[5,12] + rows_tot_endhost_cpu6 = 0
 mem_tot_endhost_cpu6: - mem[5,0] - mem[5,1] - mem[5,2] - mem[5,3]
   - mem[5,4] - mem[5,5] - mem[5,6] - mem[5,7] - mem[5,8] - mem[5,9]
   - mem[5,10] - mem[5,11] - mem[5,12] + mem_tot_endhost_cpu6 = 0
 ns_single_endhost_cpu6: 3.5 rows_tot_endhost_cpu6 - ns_single_endhost_cpu6
   + pdt_m_rows_endhost_cpu6 <= 0
 capacity_cores_endhost_cpu6: cores_sketch_endhost_cpu6
   + cores_dpdk_endhost_cpu6 <= 7
 ns_sketch_endhost_cpu6: - ns_single_endhost_cpu6 + pdt_nsk_c_endhost_cpu6
   = 0
 ns_dpdk_endhost_cpu6: 7.142857142857143 cores_dpdk_endhost_cpu6
   - pdt_nsc_dpdk_endhost_cpu6 = -21.42857142857143
 rows_tot_endhost_cpu7: - frac[6,0] - frac[6,1] - frac[6,2] - frac[6,3]
   - frac[6,4] - frac[6,5] - frac[6,6] - frac[6,7] - frac[6,8] - frac[6,9]
   - frac[6,10] - frac[6,11] - frac[6,12] + rows_tot_endhost_cpu7 = 0
 mem_tot_endhost_cpu7: - mem[6,0] - mem[6,1] - mem[6,2] - mem[6,3]
   - mem[6,4] - mem[6,5] - mem[6,6] - mem[6,7] - mem[6,8] - mem[6,9]
   - mem[6,10] - mem[6,11] - mem[6,12] + mem_tot_endhost_cpu7 = 0
 ns_single_endhost_cpu7: 3.5 rows_tot_endhost_cpu7 - ns_single_endhost_cpu7
   + pdt_m_rows_endhost_cpu7 <= 0
 capacity_cores_endhost_cpu7: cores_sketch_endhost_cpu7
   + cores_dpdk_endhost_cpu7 <= 7
 ns_sketch_endhost_cpu7: - ns_single_endhost_cpu7 + pdt_nsk_c_endhost_cpu7
   = 0
 ns_dpdk_endhost_cpu7: 7.142857142857143 cores_dpdk_endhost_cpu7
   - pdt_nsc_dpdk_endhost_cpu7 = -21.42857142857143
 rows_tot_endhost_cpu8: - frac[7,0] - frac[7,1] - frac[7,2] - frac[7,3]
   - frac[7,4] - frac[7,5] - frac[7,6] - frac[7,7] - frac[7,8] - frac[7,9]
   - frac[7,10] - frac[7,11] - frac[7,12] + rows_tot_endhost_cpu8 = 0
 mem_tot_endhost_cpu8: - mem[7,0] - mem[7,1] - mem[7,2] - mem[7,3]
   - mem[7,4] - mem[7,5] - mem[7,6] - mem[7,7] - mem[7,8] - mem[7,9]
   - mem[7,10] - mem[7,11] - mem[7,12] + mem_tot_endhost_cpu8 = 0
 ns_single_endhost_cpu8: 3.5 rows_tot_endhost_cpu8 - ns_single_endhost_cpu8
   + pdt_m_rows_endhost_cpu8 <= 0
 capacity_cores_endhost_cpu8: cores_sketch_endhost_cpu8
   + cores_dpdk_endhost_cpu8 <= 7
 ns_sketch_endhost_cpu8: - ns_single_endhost_cpu8 + pdt_nsk_c_endhost_cpu8
   = 0
 ns_dpdk_endhost_cpu8: 7.142857142857143 cores_dpdk_endhost_cpu8
   - pdt_nsc_dpdk_endhost_cpu8 = -21.42857142857143
 rows_tot_endhost_cpu9: - frac[8,0] - frac[8,1] - frac[8,2] - frac[8,3]
   - frac[8,4] - frac[8,5] - frac[8,6] - frac[8,7] - frac[8,8] - frac[8,9]
   - frac[8,10] - frac[8,11] - frac[8,12] + rows_tot_endhost_cpu9 = 0
 mem_tot_endhost_cpu9: - mem[8,0] - mem[8,1] - mem[8,2] - mem[8,3]
   - mem[8,4] - mem[8,5] - mem[8,6] - mem[8,7] - mem[8,8] - mem[8,9]
   - mem[8,10] - mem[8,11] - mem[8,12] + mem_tot_endhost_cpu9 = 0
 ns_single_endhost_cpu9: 3.5 rows_tot_endhost_cpu9 - ns_single_endhost_cpu9
   + pdt_m_rows_endhost_cpu9 <= 0
 capacity_cores_endhost_cpu9: cores_sketch_endhost_cpu9
   + cores_dpdk_endhost_cpu9 <= 7
 ns_sketch_endhost_cpu9: - ns_single_endhost_cpu9 + pdt_nsk_c_endhost_cpu9
   = 0
 ns_dpdk_endhost_cpu9: 7.142857142857143 cores_dpdk_endhost_cpu9
   - pdt_nsc_dpdk_endhost_cpu9 = -21.42857142857143
 rows_tot_endhost_cpu10: - frac[9,0] - frac[9,1] - frac[9,2] - frac[9,3]
   - frac[9,4] - frac[9,5] - frac[9,6] - frac[9,7] - frac[9,8] - frac[9,9]
   - frac[9,10] - frac[9,11] - frac[9,12] + rows_tot_endhost_cpu10 = 0
 mem_tot_endhost_cpu10: - mem[9,0] - mem[9,1] - mem[9,2] - mem[9,3]
   - mem[9,4] - mem[9,5] - mem[9,6] - mem[9,7] - mem[9,8] - mem[9,9]
   - mem[9,10] - mem[9,11] - mem[9,12] + mem_tot_endhost_cpu10 = 0
 ns_single_endhost_cpu10: 3.5 rows_tot_endhost_cpu10
   - ns_single_endhost_cpu10 + pdt_m_rows_endhost_cpu10 <= 0
 capacity_cores_endhost_cpu10: cores_sketch_endhost_cpu10
   + cores_dpdk_endhost_cpu10 <= 7
 ns_sketch_endhost_cpu10: - ns_single_endhost_cpu10
   + pdt_nsk_c_endhost_cpu10 = 0
 ns_dpdk_endhost_cpu10: 7.142857142857143 cores_dpdk_endhost_cpu10
   - pdt_nsc_dpdk_endhost_cpu10 = -21.42857142857143
 rows_tot_endhost_cpu11: - frac[10,0] - frac[10,1] - frac[10,2]
   - frac[10,3] - frac[10,4] - frac[10,5] - frac[10,6] - frac[10,7]
   - frac[10,8] - frac[10,9] - frac[10,10] - frac[10,11] - frac[10,12]
   + rows_tot_endhost_cpu11 = 0
 mem_tot_endhost_cpu11: - mem[10,0] - mem[10,1] - mem[10,2] - mem[10,3]
   - mem[10,4] - mem[10,5] - mem[10,6] - mem[10,7] - mem[10,8] - mem[10,9]
   - mem[10,10] - mem[10,11] - mem[10,12] + mem_tot_endhost_cpu11 = 0
 ns_single_endhost_cpu11: 3.5 rows_tot_endhost_cpu11
   - ns_single_endhost_cpu11 + pdt_m_rows_endhost_cpu11 <= 0
 capacity_cores_endhost_cpu11: cores_sketch_endhost_cpu11
   + cores_dpdk_endhost_cpu11 <= 7
 ns_sketch_endhost_cpu11: - ns_single_endhost_cpu11
   + pdt_nsk_c_endhost_cpu11 = 0
 ns_dpdk_endhost_cpu11: 7.142857142857143 cores_dpdk_endhost_cpu11
   - pdt_nsc_dpdk_endhost_cpu11 = -21.42857142857143
 rows_tot_endhost_cpu12: - frac[11,0] - frac[11,1] - frac[11,2]
   - frac[11,3] - frac[11,4] - frac[11,5] - frac[11,6] - frac[11,7]
   - frac[11,8] - frac[11,9] - frac[11,10] - frac[11,11] - frac[11,12]
   + rows_tot_endhost_cpu12 = 0
 mem_tot_endhost_cpu12: - mem[11,0] - mem[11,1] - mem[11,2] - mem[11,3]
   - mem[11,4] - mem[11,5] - mem[11,6] - mem[11,7] - mem[11,8] - mem[11,9]
   - mem[11,10] - mem[11,11] - mem[11,12] + mem_tot_endhost_cpu12 = 0
 ns_single_endhost_cpu12: 3.5 rows_tot_endhost_cpu12
   - ns_single_endhost_cpu12 + pdt_m_rows_endhost_cpu12 <= 0
 capacity_cores_endhost_cpu12: cores_sketch_endhost_cpu12
   + cores_dpdk_endhost_cpu12 <= 7
 ns_sketch_endhost_cpu12: - ns_single_endhost_cpu12
   + pdt_nsk_c_endhost_cpu12 = 0
 ns_dpdk_endhost_cpu12: 7.142857142857143 cores_dpdk_endhost_cpu12
   - pdt_nsc_dpdk_endhost_cpu12 = -21.42857142857143
 rows_tot_endhost_cpu13: - frac[12,0] - frac[12,1] - frac[12,2]
   - frac[12,3] - frac[12,4] - frac[12,5] - frac[12,6] - frac[12,7]
   - frac[12,8] - frac[12,9] - frac[12,10] - frac[12,11] - frac[12,12]
   + rows_tot_endhost_cpu13 = 0
 mem_tot_endhost_cpu13: - mem[12,0] - mem[12,1] - mem[12,2] - mem[12,3]
   - mem[12,4] - mem[12,5] - mem[12,6] - mem[12,7] - mem[12,8] - mem[12,9]
   - mem[12,10] - mem[12,11] - mem[12,12] + mem_tot_endhost_cpu13 = 0
 ns_single_endhost_cpu13: 3.5 rows_tot_endhost_cpu13
   - ns_single_endhost_cpu13 + pdt_m_rows_endhost_cpu13 <= 0
 capacity_cores_endhost_cpu13: cores_sketch_endhost_cpu13
   + cores_dpdk_endhost_cpu13 <= 7
 ns_sketch_endhost_cpu13: - ns_single_endhost_cpu13
   + pdt_nsk_c_endhost_cpu13 = 0
 ns_dpdk_endhost_cpu13: 7.142857142857143 cores_dpdk_endhost_cpu13
   - pdt_nsc_dpdk_endhost_cpu13 = -21.42857142857143
 rows_tot_endhost_cpu14: - frac[13,0] - frac[13,1] - frac[13,2]
   - frac[13,3] - frac[13,4] - frac[13,5] - frac[13,6] - frac[13,7]
   - frac[13,8] - frac[13,9] - frac[13,10] - frac[13,11] - frac[13,12]
   + rows_tot_endhost_cpu14 = 0
 mem_tot_endhost_cpu14: - mem[13,0] - mem[13,1] - mem[13,2] - mem[13,3]
   - mem[13,4] - mem[13,5] - mem[13,6] - mem[13,7] - mem[13,8] - mem[13,9]
   - mem[13,10] - mem[13,11] - mem[13,12] + mem_tot_endhost_cpu14 = 0
 ns_single_endhost_cpu14: 3.5 rows_tot_endhost_cpu14
   - ns_single_endhost_cpu14 + pdt_m_rows_endhost_cpu14 <= 0
 capacity_cores_endhost_cpu14: cores_sketch_endhost_cpu14
   + cores_dpdk_endhost_cpu14 <= 7
 ns_sketch_endhost_cpu14: - ns_single_endhost_cpu14
   + pdt_nsk_c_endhost_cpu14 = 0
 ns_dpdk_endhost_cpu14: 7.142857142857143 cores_dpdk_endhost_cpu14
   - pdt_nsc_dpdk_endhost_cpu14 = -21.42857142857143
 rows_tot_endhost_cpu15: - frac[14,0] - frac[14,1] - frac[14,2]
   - frac[14,3] - frac[14,4] - frac[14,5] - frac[14,6] - frac[14,7]
   - frac[14,8] - frac[14,9] - frac[14,10] - frac[14,11] - frac[14,12]
   + rows_tot_endhost_cpu15 = 0
 mem_tot_endhost_cpu15: - mem[14,0] - mem[14,1] - mem[14,2] - mem[14,3]
   - mem[14,4] - mem[14,5] - mem[14,6] - mem[14,7] - mem[14,8] - mem[14,9]
   - mem[14,10] - mem[14,11] - mem[14,12] + mem_tot_endhost_cpu15 = 0
 ns_single_endhost_cpu15: 3.5 rows_tot_endhost_cpu15
   - ns_single_endhost_cpu15 + pdt_m_rows_endhost_cpu15 <= 0
 capacity_cores_endhost_cpu15: cores_sketch_endhost_cpu15
   + cores_dpdk_endhost_cpu15 <= 7
 ns_sketch_endhost_cpu15: - ns_single_endhost_cpu15
   + pdt_nsk_c_endhost_cpu15 = 0
 ns_dpdk_endhost_cpu15: 7.142857142857143 cores_dpdk_endhost_cpu15
   - pdt_nsc_dpdk_endhost_cpu15 = -21.42857142857143
 rows_tot_endhost_cpu16: - frac[15,0] - frac[15,1] - frac[15,2]
   - frac[15,3] - frac[15,4] - frac[15,5] - frac[15,6] - frac[15,7]
   - frac[15,8] - frac[15,9] - frac[15,10] - frac[15,11] - frac[15,12]
   + rows_tot_endhost_cpu16 = 0
 mem_tot_endhost_cpu16: - mem[15,0] - mem[15,1] - mem[15,2] - mem[15,3]
   - mem[15,4] - mem[15,5] - mem[15,6] - mem[15,7] - mem[15,8] - mem[15,9]
   - mem[15,10] - mem[15,11] - mem[15,12] + mem_tot_endhost_cpu16 = 0
 ns_single_endhost_cpu16: 3.5 rows_tot_endhost_cpu16
   - ns_single_endhost_cpu16 + pdt_m_rows_endhost_cpu16 <= 0
 capacity_cores_endhost_cpu16: cores_sketch_endhost_cpu16
   + cores_dpdk_endhost_cpu16 <= 7
 ns_sketch_endhost_cpu16: - ns_single_endhost_cpu16
   + pdt_nsk_c_endhost_cpu16 = 0
 ns_dpdk_endhost_cpu16: 7.142857142857143 cores_dpdk_endhost_cpu16
   - pdt_nsc_dpdk_endhost_cpu16 = -21.42857142857143
 rows_tot_endhost_cpu17: - frac[16,0] - frac[16,1] - frac[16,2]
   - frac[16,3] - frac[16,4] - frac[16,5] - frac[16,6] - frac[16,7]
   - frac[16,8] - frac[16,9] - frac[16,10] - frac[16,11] - frac[16,12]
   + rows_tot_endhost_cpu17 = 0
 mem_tot_endhost_cpu17: - mem[16,0] - mem[16,1] - mem[16,2] - mem[16,3]
   - mem[16,4] - mem[16,5] - mem[16,6] - mem[16,7] - mem[16,8] - mem[16,9]
   - mem[16,10] - mem[16,11] - mem[16,12] + mem_tot_endhost_cpu17 = 0
 ns_single_endhost_cpu17: 3.5 rows_tot_endhost_cpu17
   - ns_single_endhost_cpu17 + pdt_m_rows_endhost_cpu17 <= 0
 capacity_cores_endhost_cpu17: cores_sketch_endhost_cpu17
   + cores_dpdk_endhost_cpu17 <= 7
 ns_sketch_endhost_cpu17: - ns_single_endhost_cpu17
   + pdt_nsk_c_endhost_cpu17 = 0
 ns_dpdk_endhost_cpu17: 7.142857142857143 cores_dpdk_endhost_cpu17
   - pdt_nsc_dpdk_endhost_cpu17 = -21.42857142857143
 rows_tot_endhost_cpu18: - frac[17,0] - frac[17,1] - frac[17,2]
   - frac[17,3] - frac[17,4] - frac[17,5] - frac[17,6] - frac[17,7]
   - frac[17,8] - frac[17,9] - frac[17,10] - frac[17,11] - frac[17,12]
   + rows_tot_endhost_cpu18 = 0
 mem_tot_endhost_cpu18: - mem[17,0] - mem[17,1] - mem[17,2] - mem[17,3]
   - mem[17,4] - mem[17,5] - mem[17,6] - mem[17,7] - mem[17,8] - mem[17,9]
   - mem[17,10] - mem[17,11] - mem[17,12] + mem_tot_endhost_cpu18 = 0
 ns_single_endhost_cpu18: 3.5 rows_tot_endhost_cpu18
   - ns_single_endhost_cpu18 + pdt_m_rows_endhost_cpu18 <= 0
 capacity_cores_endhost_cpu18: cores_sketch_endhost_cpu18
   + cores_dpdk_endhost_cpu18 <= 7
 ns_sketch_endhost_cpu18: - ns_single_endhost_cpu18
   + pdt_nsk_c_endhost_cpu18 = 0
 ns_dpdk_endhost_cpu18: 7.142857142857143 cores_dpdk_endhost_cpu18
   - pdt_nsc_dpdk_endhost_cpu18 = -21.42857142857143
 rows_tot_endhost_cpu19: - frac[18,0] - frac[18,1] - frac[18,2]
   - frac[18,3] - frac[18,4] - frac[18,5] - frac[18,6] - frac[18,7]
   - frac[18,8] - frac[18,9] - frac[18,10] - frac[18,11] - frac[18,12]
   + rows_tot_endhost_cpu19 = 0
 mem_tot_endhost_cpu19: - mem[18,0] - mem[18,1] - mem[18,2] - mem[18,3]
   - mem[18,4] - mem[18,5] - mem[18,6] - mem[18,7] - mem[18,8] - mem[18,9]
   - mem[18,10] - mem[18,11] - mem[18,12] + mem_tot_endhost_cpu19 = 0
 ns_single_endhost_cpu19: 3.5 rows_tot_endhost_cpu19
   - ns_single_endhost_cpu19 + pdt_m_rows_endhost_cpu19 <= 0
 capacity_cores_endhost_cpu19: cores_sketch_endhost_cpu19
   + cores_dpdk_endhost_cpu19 <= 7
 ns_sketch_endhost_cpu19: - ns_single_endhost_cpu19
   + pdt_nsk_c_endhost_cpu19 = 0
 ns_dpdk_endhost_cpu19: 7.142857142857143 cores_dpdk_endhost_cpu19
   - pdt_nsc_dpdk_endhost_cpu19 = -21.42857142857143
 rows_tot_endhost_cpu20: - frac[19,0] - frac[19,1] - frac[19,2]
   - frac[19,3] - frac[19,4] - frac[19,5] - frac[19,6] - frac[19,7]
   - frac[19,8] - frac[19,9] - frac[19,10] - frac[19,11] - frac[19,12]
   + rows_tot_endhost_cpu20 = 0
 mem_tot_endhost_cpu20: - mem[19,0] - mem[19,1] - mem[19,2] - mem[19,3]
   - mem[19,4] - mem[19,5] - mem[19,6] - mem[19,7] - mem[19,8] - mem[19,9]
   - mem[19,10] - mem[19,11] - mem[19,12] + mem_tot_endhost_cpu20 = 0
 ns_single_endhost_cpu20: 3.5 rows_tot_endhost_cpu20
   - ns_single_endhost_cpu20 + pdt_m_rows_endhost_cpu20 <= 0
 capacity_cores_endhost_cpu20: cores_sketch_endhost_cpu20
   + cores_dpdk_endhost_cpu20 <= 7
 ns_sketch_endhost_cpu20: - ns_single_endhost_cpu20
   + pdt_nsk_c_endhost_cpu20 = 0
 ns_dpdk_endhost_cpu20: 7.142857142857143 cores_dpdk_endhost_cpu20
   - pdt_nsc_dpdk_endhost_cpu20 = -21.42857142857143
 rows_tot_endhost_cpu21: - frac[20,0] - frac[20,1] - frac[20,2]
   - frac[20,3] - frac[20,4] - frac[20,5] - frac[20,6] - frac[20,7]
   - frac[20,8] - frac[20,9] - frac[20,10] - frac[20,11] - frac[20,12]
   + rows_tot_endhost_cpu21 = 0
 mem_tot_endhost_cpu21: - mem[20,0] - mem[20,1] - mem[20,2] - mem[20,3]
   - mem[20,4] - mem[20,5] - mem[20,6] - mem[20,7] - mem[20,8] - mem[20,9]
   - mem[20,10] - mem[20,11] - mem[20,12] + mem_tot_endhost_cpu21 = 0
 ns_single_endhost_cpu21: 3.5 rows_tot_endhost_cpu21
   - ns_single_endhost_cpu21 + pdt_m_rows_endhost_cpu21 <= 0
 capacity_cores_endhost_cpu21: cores_sketch_endhost_cpu21
   + cores_dpdk_endhost_cpu21 <= 7
 ns_sketch_endhost_cpu21: - ns_single_endhost_cpu21
   + pdt_nsk_c_endhost_cpu21 = 0
 ns_dpdk_endhost_cpu21: 7.142857142857143 cores_dpdk_endhost_cpu21
   - pdt_nsc_dpdk_endhost_cpu21 = -21.42857142857143
 rows_tot_endhost_cpu22: - frac[21,0] - frac[21,1] - frac[21,2]
   - frac[21,3] - frac[21,4] - frac[21,5] - frac[21,6] - frac[21,7]
   - frac[21,8] - frac[21,9] - frac[21,10] - frac[21,11] - frac[21,12]
   + rows_tot_endhost_cpu22 = 0
 mem_tot_endhost_cpu22: - mem[21,0] - mem[21,1] - mem[21,2] - mem[21,3]
   - mem[21,4] - mem[21,5] - mem[21,6] - mem[21,7] - mem[21,8] - mem[21,9]
   - mem[21,10] - mem[21,11] - mem[21,12] + mem_tot_endhost_cpu22 = 0
 ns_single_endhost_cpu22: 3.5 rows_tot_endhost_cpu22
   - ns_single_endhost_cpu22 + pdt_m_rows_endhost_cpu22 <= 0
 capacity_cores_endhost_cpu22: cores_sketch_endhost_cpu22
   + cores_dpdk_endhost_cpu22 <= 7
 ns_sketch_endhost_cpu22: - ns_single_endhost_cpu22
   + pdt_nsk_c_endhost_cpu22 = 0
 ns_dpdk_endhost_cpu22: 7.142857142857143 cores_dpdk_endhost_cpu22
   - pdt_nsc_dpdk_endhost_cpu22 = -21.42857142857143
 rows_tot_endhost_cpu23: - frac[22,0] - frac[22,1] - frac[22,2]
   - frac[22,3] - frac[22,4] - frac[22,5] - frac[22,6] - frac[22,7]
   - frac[22,8] - frac[22,9] - frac[22,10] - frac[22,11] - frac[22,12]
   + rows_tot_endhost_cpu23 = 0
 mem_tot_endhost_cpu23: - mem[22,0] - mem[22,1] - mem[22,2] - mem[22,3]
   - mem[22,4] - mem[22,5] - mem[22,6] - mem[22,7] - mem[22,8] - mem[22,9]
   - mem[22,10] - mem[22,11] - mem[22,12] + mem_tot_endhost_cpu23 = 0
 ns_single_endhost_cpu23: 3.5 rows_tot_endhost_cpu23
   - ns_single_endhost_cpu23 + pdt_m_rows_endhost_cpu23 <= 0
 capacity_cores_endhost_cpu23: cores_sketch_endhost_cpu23
   + cores_dpdk_endhost_cpu23 <= 7
 ns_sketch_endhost_cpu23: - ns_single_endhost_cpu23
   + pdt_nsk_c_endhost_cpu23 = 0
 ns_dpdk_endhost_cpu23: 7.142857142857143 cores_dpdk_endhost_cpu23
   - pdt_nsc_dpdk_endhost_cpu23 = -21.42857142857143
 rows_tot_endhost_cpu24: - frac[23,0] - frac[23,1] - frac[23,2]
   - frac[23,3] - frac[23,4] - frac[23,5] - frac[23,6] - frac[23,7]
   - frac[23,8] - frac[23,9] - frac[23,10] - frac[23,11] - frac[23,12]
   + rows_tot_endhost_cpu24 = 0
 mem_tot_endhost_cpu24: - mem[23,0] - mem[23,1] - mem[23,2] - mem[23,3]
   - mem[23,4] - mem[23,5] - mem[23,6] - mem[23,7] - mem[23,8] - mem[23,9]
   - mem[23,10] - mem[23,11] - mem[23,12] + mem_tot_endhost_cpu24 = 0
 ns_single_endhost_cpu24: 3.5 rows_tot_endhost_cpu24
   - ns_single_endhost_cpu24 + pdt_m_rows_endhost_cpu24 <= 0
 capacity_cores_endhost_cpu24: cores_sketch_endhost_cpu24
   + cores_dpdk_endhost_cpu24 <= 7
 ns_sketch_endhost_cpu24: - ns_single_endhost_cpu24
   + pdt_nsk_c_endhost_cpu24 = 0
 ns_dpdk_endhost_cpu24: 7.142857142857143 cores_dpdk_endhost_cpu24
   - pdt_nsc_dpdk_endhost_cpu24 = -21.42857142857143
 rows_tot_endhost_cpu25: - frac[24,0] - frac[24,1] - frac[24,2]
   - frac[24,3] - frac[24,4] - frac[24,5] - frac[24,6] - frac[24,7]
   - frac[24,8] - frac[24,9] - frac[24,10] - frac[24,11] - frac[24,12]
   + rows_tot_endhost_cpu25 = 0
 mem_tot_endhost_cpu25: - mem[24,0] - mem[24,1] - mem[24,2] - mem[24,3]
   - mem[24,4] - mem[24,5] - mem[24,6] - mem[24,7] - mem[24,8] - mem[24,9]
   - mem[24,10] - mem[24,11] - mem[24,12] + mem_tot_endhost_cpu25 = 0
 ns_single_endhost_cpu25: 3.5 rows_tot_endhost_cpu25
   - ns_single_endhost_cpu25 + pdt_m_rows_endhost_cpu25 <= 0
 capacity_cores_endhost_cpu25: cores_sketch_endhost_cpu25
   + cores_dpdk_endhost_cpu25 <= 7
 ns_sketch_endhost_cpu25: - ns_single_endhost_cpu25
   + pdt_nsk_c_endhost_cpu25 = 0
 ns_dpdk_endhost_cpu25: 7.142857142857143 cores_dpdk_endhost_cpu25
   - pdt_nsc_dpdk_endhost_cpu25 = -21.42857142857143
 rows_tot_endhost_cpu26: - frac[25,0] - frac[25,1] - frac[25,2]
   - frac[25,3] - frac[25,4] - frac[25,5] - frac[25,6] - frac[25,7]
   - frac[25,8] - frac[25,9] - frac[25,10] - frac[25,11] - frac[25,12]
   + rows_tot_endhost_cpu26 = 0
 mem_tot_endhost_cpu26: - mem[25,0] - mem[25,1] - mem[25,2] - mem[25,3]
   - mem[25,4] - mem[25,5] - mem[25,6] - mem[25,7] - mem[25,8] - mem[25,9]
   - mem[25,10] - mem[25,11] - mem[25,12] + mem_tot_endhost_cpu26 = 0
 ns_single_endhost_cpu26: 3.5 rows_tot_endhost_cpu26
   - ns_single_endhost_cpu26 + pdt_m_rows_endhost_cpu26 <= 0
 capacity_cores_endhost_cpu26: cores_sketch_endhost_cpu26
   + cores_dpdk_endhost_cpu26 <= 7
 ns_sketch_endhost_cpu26: - ns_single_endhost_cpu26
   + pdt_nsk_c_endhost_cpu26 = 0
 ns_dpdk_endhost_cpu26: 7.142857142857143 cores_dpdk_endhost_cpu26
   - pdt_nsc_dpdk_endhost_cpu26 = -21.42857142857143
 rows_tot_endhost_cpu27: - frac[26,0] - frac[26,1] - frac[26,2]
   - frac[26,3] - frac[26,4] - frac[26,5] - frac[26,6] - frac[26,7]
   - frac[26,8] - frac[26,9] - frac[26,10] - frac[26,11] - frac[26,12]
   + rows_tot_endhost_cpu27 = 0
 mem_tot_endhost_cpu27: - mem[26,0] - mem[26,1] - mem[26,2] - mem[26,3]
   - mem[26,4] - mem[26,5] - mem[26,6] - mem[26,7] - mem[26,8] - mem[26,9]
   - mem[26,10] - mem[26,11] - mem[26,12] + mem_tot_endhost_cpu27 = 0
 ns_single_endhost_cpu27: 3.5 rows_tot_endhost_cpu27
   - ns_single_endhost_cpu27 + pdt_m_rows_endhost_cpu27 <= 0
 capacity_cores_endhost_cpu27: cores_sketch_endhost_cpu27
   + cores_dpdk_endhost_cpu27 <= 7
 ns_sketch_endhost_cpu27: - ns_single_endhost_cpu27
   + pdt_nsk_c_endhost_cpu27 = 0
 ns_dpdk_endhost_cpu27: 7.142857142857143 cores_dpdk_endhost_cpu27
   - pdt_nsc_dpdk_endhost_cpu27 = -21.42857142857143
 rows_tot_endhost_cpu28: - frac[27,0] - frac[27,1] - frac[27,2]
   - frac[27,3] - frac[27,4] - frac[27,5] - frac[27,6] - frac[27,7]
   - frac[27,8] - frac[27,9] - frac[27,10] - frac[27,11] - frac[27,12]
   + rows_tot_endhost_cpu28 = 0
 mem_tot_endhost_cpu28: - mem[27,0] - mem[27,1] - mem[27,2] - mem[27,3]
   - mem[27,4] - mem[27,5] - mem[27,6] - mem[27,7] - mem[27,8] - mem[27,9]
   - mem[27,10] - mem[27,11] - mem[27,12] + mem_tot_endhost_cpu28 = 0
 ns_single_endhost_cpu28: 3.5 rows_tot_endhost_cpu28
   - ns_single_endhost_cpu28 + pdt_m_rows_endhost_cpu28 <= 0
 capacity_cores_endhost_cpu28: cores_sketch_endhost_cpu28
   + cores_dpdk_endhost_cpu28 <= 7
 ns_sketch_endhost_cpu28: - ns_single_endhost_cpu28
   + pdt_nsk_c_endhost_cpu28 = 0
 ns_dpdk_endhost_cpu28: 7.142857142857143 cores_dpdk_endhost_cpu28
   - pdt_nsc_dpdk_endhost_cpu28 = -21.42857142857143
 rows_tot_endhost_cpu29: - frac[28,0] - frac[28,1] - frac[28,2]
   - frac[28,3] - frac[28,4] - frac[28,5] - frac[28,6] - frac[28,7]
   - frac[28,8] - frac[28,9] - frac[28,10] - frac[28,11] - frac[28,12]
   + rows_tot_endhost_cpu29 = 0
 mem_tot_endhost_cpu29: - mem[28,0] - mem[28,1] - mem[28,2] - mem[28,3]
   - mem[28,4] - mem[28,5] - mem[28,6] - mem[28,7] - mem[28,8] - mem[28,9]
   - mem[28,10] - mem[28,11] - mem[28,12] + mem_tot_endhost_cpu29 = 0
 ns_single_endhost_cpu29: 3.5 rows_tot_endhost_cpu29
   - ns_single_endhost_cpu29 + pdt_m_rows_endhost_cpu29 <= 0
 capacity_cores_endhost_cpu29: cores_sketch_endhost_cpu29
   + cores_dpdk_endhost_cpu29 <= 7
 ns_sketch_endhost_cpu29: - ns_single_endhost_cpu29
   + pdt_nsk_c_endhost_cpu29 = 0
 ns_dpdk_endhost_cpu29: 7.142857142857143 cores_dpdk_endhost_cpu29
   - pdt_nsc_dpdk_endhost_cpu29 = -21.42857142857143
 rows_tot_endhost_cpu30: - frac[29,0] - frac[29,1] - frac[29,2]
   - frac[29,3] - frac[29,4] - frac[29,5] - frac[29,6] - frac[29,7]
   - frac[29,8] - frac[29,9] - frac[29,10] - frac[29,11] - frac[29,12]
   + rows_tot_endhost_cpu30 = 0
 mem_tot_endhost_cpu30: - mem[29,0] - mem[29,1] - mem[29,2] - mem[29,3]
   - mem[29,4] - mem[29,5] - mem[29,6] - mem[29,7] - mem[29,8] - mem[29,9]
   - mem[29,10] - mem[29,11] - mem[29,12] + mem_tot_endhost_cpu30 = 0
 ns_single_endhost_cpu30: 3.5 rows_tot_endhost_cpu30
   - ns_single_endhost_cpu30 + pdt_m_rows_endhost_cpu30 <= 0
 capacity_cores_endhost_cpu30: cores_sketch_endhost_cpu30
   + cores_dpdk_endhost_cpu30 <= 7
 ns_sketch_endhost_cpu30: - ns_single_endhost_cpu30
   + pdt_nsk_c_endhost_cpu30 = 0
 ns_dpdk_endhost_cpu30: 7.142857142857143 cores_dpdk_endhost_cpu30
   - pdt_nsc_dpdk_endhost_cpu30 = -21.42857142857143
 rows_tot_endhost_cpu31: - frac[30,0] - frac[30,1] - frac[30,2]
   - frac[30,3] - frac[30,4] - frac[30,5] - frac[30,6] - frac[30,7]
   - frac[30,8] - frac[30,9] - frac[30,10] - frac[30,11] - frac[30,12]
   + rows_tot_endhost_cpu31 = 0
 mem_tot_endhost_cpu31: - mem[30,0] - mem[30,1] - mem[30,2] - mem[30,3]
   - mem[30,4] - mem[30,5] - mem[30,6] - mem[30,7] - mem[30,8] - mem[30,9]
   - mem[30,10] - mem[30,11] - mem[30,12] + mem_tot_endhost_cpu31 = 0
 ns_single_endhost_cpu31: 3.5 rows_tot_endhost_cpu31
   - ns_single_endhost_cpu31 + pdt_m_rows_endhost_cpu31 <= 0
 capacity_cores_endhost_cpu31: cores_sketch_endhost_cpu31
   + cores_dpdk_endhost_cpu31 <= 7
 ns_sketch_endhost_cpu31: - ns_single_endhost_cpu31
   + pdt_nsk_c_endhost_cpu31 = 0
 ns_dpdk_endhost_cpu31: 7.142857142857143 cores_dpdk_endhost_cpu31
   - pdt_nsc_dpdk_endhost_cpu31 = -21.42857142857143
 rows_tot_endhost_cpu32: - frac[31,0] - frac[31,1] - frac[31,2]
   - frac[31,3] - frac[31,4] - frac[31,5] - frac[31,6] - frac[31,7]
   - frac[31,8] - frac[31,9] - frac[31,10] - frac[31,11] - frac[31,12]
   + rows_tot_endhost_cpu32 = 0
 mem_tot_endhost_cpu32: - mem[31,0] - mem[31,1] - mem[31,2] - mem[31,3]
   - mem[31,4] - mem[31,5] - mem[31,6] - mem[31,7] - mem[31,8] - mem[31,9]
   - mem[31,10] - mem[31,11] - mem[31,12] + mem_tot_endhost_cpu32 = 0
 ns_single_endhost_cpu32: 3.5 rows_tot_endhost_cpu32
   - ns_single_endhost_cpu32 + pdt_m_rows_endhost_cpu32 <= 0
 capacity_cores_endhost_cpu32: cores_sketch_endhost_cpu32
   + cores_dpdk_endhost_cpu32 <= 7
 ns_sketch_endhost_cpu32: - ns_single_endhost_cpu32
   + pdt_nsk_c_endhost_cpu32 = 0
 ns_dpdk_endhost_cpu32: 7.142857142857143 cores_dpdk_endhost_cpu32
   - pdt_nsc_dpdk_endhost_cpu32 = -21.42857142857143
 rows_tot_tor_p41: - frac[32,0] - frac[32,1] - frac[32,2] - frac[32,3]
   - frac[32,4] - frac[32,5] - frac[32,6] - frac[32,7] - frac[32,8]
   - frac[32,9] - frac[32,10] - frac[32,11] - frac[32,12]
   + rows_tot_tor_p41 = 0
 mem_tot_tor_p41: - mem[32,0] - mem[32,1] - mem[32,2] - mem[32,3]
   - mem[32,4] - mem[32,5] - mem[32,6] - mem[32,7] - mem[32,8] - mem[32,9]
   - mem[32,10] - mem[32,11] - mem[32,12] + mem_tot_tor_p41 = 0
 capacity_mem_par_tor_p41: mem[32,0] <= 48
 capacity_mem_par_tor_p41: mem[32,1] <= 48
 capacity_mem_par_tor_p41: mem[32,2] <= 48
 capacity_mem_par_tor_p41: mem[32,3] <= 48
 capacity_mem_par_tor_p41: mem[32,4] <= 48
 capacity_mem_par_tor_p41: mem[32,5] <= 48
 capacity_mem_par_tor_p41: mem[32,6] <= 48
 capacity_mem_par_tor_p41: mem[32,7] <= 48
 capacity_mem_par_tor_p41: mem[32,8] <= 48
 capacity_mem_par_tor_p41: mem[32,9] <= 48
 capacity_mem_par_tor_p41: mem[32,10] <= 48
 capacity_mem_par_tor_p41: mem[32,11] <= 48
 capacity_mem_par_tor_p41: mem[32,12] <= 48
 ns_tor_p41: ns_tor_p41 = 6.756756756756757
 rows_tot_tor_p42: - frac[33,0] - frac[33,1] - frac[33,2] - frac[33,3]
   - frac[33,4] - frac[33,5] - frac[33,6] - frac[33,7] - frac[33,8]
   - frac[33,9] - frac[33,10] - frac[33,11] - frac[33,12]
   + rows_tot_tor_p42 = 0
 mem_tot_tor_p42: - mem[33,0] - mem[33,1] - mem[33,2] - mem[33,3]
   - mem[33,4] - mem[33,5] - mem[33,6] - mem[33,7] - mem[33,8] - mem[33,9]
   - mem[33,10] - mem[33,11] - mem[33,12] + mem_tot_tor_p42 = 0
 capacity_mem_par_tor_p42: mem[33,0] <= 48
 capacity_mem_par_tor_p42: mem[33,1] <= 48
 capacity_mem_par_tor_p42: mem[33,2] <= 48
 capacity_mem_par_tor_p42: mem[33,3] <= 48
 capacity_mem_par_tor_p42: mem[33,4] <= 48
 capacity_mem_par_tor_p42: mem[33,5] <= 48
 capacity_mem_par_tor_p42: mem[33,6] <= 48
 capacity_mem_par_tor_p42: mem[33,7] <= 48
 capacity_mem_par_tor_p42: mem[33,8] <= 48
 capacity_mem_par_tor_p42: mem[33,9] <= 48
 capacity_mem_par_tor_p42: mem[33,10] <= 48
 capacity_mem_par_tor_p42: mem[33,11] <= 48
 capacity_mem_par_tor_p42: mem[33,12] <= 48
 ns_tor_p42: ns_tor_p42 = 6.756756756756757
 rows_tot_tor_p43: - frac[34,0] - frac[34,1] - frac[34,2] - frac[34,3]
   - frac[34,4] - frac[34,5] - frac[34,6] - frac[34,7] - frac[34,8]
   - frac[34,9] - frac[34,10] - frac[34,11] - frac[34,12]
   + rows_tot_tor_p43 = 0
 mem_tot_tor_p43: - mem[34,0] - mem[34,1] - mem[34,2] - mem[34,3]
   - mem[34,4] - mem[34,5] - mem[34,6] - mem[34,7] - mem[34,8] - mem[34,9]
   - mem[34,10] - mem[34,11] - mem[34,12] + mem_tot_tor_p43 = 0
 capacity_mem_par_tor_p43: mem[34,0] <= 48
 capacity_mem_par_tor_p43: mem[34,1] <= 48
 capacity_mem_par_tor_p43: mem[34,2] <= 48
 capacity_mem_par_tor_p43: mem[34,3] <= 48
 capacity_mem_par_tor_p43: mem[34,4] <= 48
 capacity_mem_par_tor_p43: mem[34,5] <= 48
 capacity_mem_par_tor_p43: mem[34,6] <= 48
 capacity_mem_par_tor_p43: mem[34,7] <= 48
 capacity_mem_par_tor_p43: mem[34,8] <= 48
 capacity_mem_par_tor_p43: mem[34,9] <= 48
 capacity_mem_par_tor_p43: mem[34,10] <= 48
 capacity_mem_par_tor_p43: mem[34,11] <= 48
 capacity_mem_par_tor_p43: mem[34,12] <= 48
 ns_tor_p43: ns_tor_p43 = 6.756756756756757
 rows_tot_tor_p44: - frac[35,0] - frac[35,1] - frac[35,2] - frac[35,3]
   - frac[35,4] - frac[35,5] - frac[35,6] - frac[35,7] - frac[35,8]
   - frac[35,9] - frac[35,10] - frac[35,11] - frac[35,12]
   + rows_tot_tor_p44 = 0
 mem_tot_tor_p44: - mem[35,0] - mem[35,1] - mem[35,2] - mem[35,3]
   - mem[35,4] - mem[35,5] - mem[35,6] - mem[35,7] - mem[35,8] - mem[35,9]
   - mem[35,10] - mem[35,11] - mem[35,12] + mem_tot_tor_p44 = 0
 capacity_mem_par_tor_p44: mem[35,0] <= 48
 capacity_mem_par_tor_p44: mem[35,1] <= 48
 capacity_mem_par_tor_p44: mem[35,2] <= 48
 capacity_mem_par_tor_p44: mem[35,3] <= 48
 capacity_mem_par_tor_p44: mem[35,4] <= 48
 capacity_mem_par_tor_p44: mem[35,5] <= 48
 capacity_mem_par_tor_p44: mem[35,6] <= 48
 capacity_mem_par_tor_p44: mem[35,7] <= 48
 capacity_mem_par_tor_p44: mem[35,8] <= 48
 capacity_mem_par_tor_p44: mem[35,9] <= 48
 capacity_mem_par_tor_p44: mem[35,10] <= 48
 capacity_mem_par_tor_p44: mem[35,11] <= 48
 capacity_mem_par_tor_p44: mem[35,12] <= 48
 ns_tor_p44: ns_tor_p44 = 6.756756756756757
 rows_tot_l1_p41: - frac[36,0] - frac[36,1] - frac[36,2] - frac[36,3]
   - frac[36,4] - frac[36,5] - frac[36,6] - frac[36,7] - frac[36,8]
   - frac[36,9] - frac[36,10] - frac[36,11] - frac[36,12] + rows_tot_l1_p41
   = 0
 mem_tot_l1_p41: - mem[36,0] - mem[36,1] - mem[36,2] - mem[36,3]
   - mem[36,4] - mem[36,5] - mem[36,6] - mem[36,7] - mem[36,8] - mem[36,9]
   - mem[36,10] - mem[36,11] - mem[36,12] + mem_tot_l1_p41 = 0
 capacity_mem_par_l1_p41: mem[36,0] <= 48
 capacity_mem_par_l1_p41: mem[36,1] <= 48
 capacity_mem_par_l1_p41: mem[36,2] <= 48
 capacity_mem_par_l1_p41: mem[36,3] <= 48
 capacity_mem_par_l1_p41: mem[36,4] <= 48
 capacity_mem_par_l1_p41: mem[36,5] <= 48
 capacity_mem_par_l1_p41: mem[36,6] <= 48
 capacity_mem_par_l1_p41: mem[36,7] <= 48
 capacity_mem_par_l1_p41: mem[36,8] <= 48
 capacity_mem_par_l1_p41: mem[36,9] <= 48
 capacity_mem_par_l1_p41: mem[36,10] <= 48
 capacity_mem_par_l1_p41: mem[36,11] <= 48
 capacity_mem_par_l1_p41: mem[36,12] <= 48
 ns_l1_p41: ns_l1_p41 = 6.756756756756757
 rows_tot_l1_p42: - frac[37,0] - frac[37,1] - frac[37,2] - frac[37,3]
   - frac[37,4] - frac[37,5] - frac[37,6] - frac[37,7] - frac[37,8]
   - frac[37,9] - frac[37,10] - frac[37,11] - frac[37,12] + rows_tot_l1_p42
   = 0
 mem_tot_l1_p42: - mem[37,0] - mem[37,1] - mem[37,2] - mem[37,3]
   - mem[37,4] - mem[37,5] - mem[37,6] - mem[37,7] - mem[37,8] - mem[37,9]
   - mem[37,10] - mem[37,11] - mem[37,12] + mem_tot_l1_p42 = 0
 capacity_mem_par_l1_p42: mem[37,0] <= 48
 capacity_mem_par_l1_p42: mem[37,1] <= 48
 capacity_mem_par_l1_p42: mem[37,2] <= 48
 capacity_mem_par_l1_p42: mem[37,3] <= 48
 capacity_mem_par_l1_p42: mem[37,4] <= 48
 capacity_mem_par_l1_p42: mem[37,5] <= 48
 capacity_mem_par_l1_p42: mem[37,6] <= 48
 capacity_mem_par_l1_p42: mem[37,7] <= 48
 capacity_mem_par_l1_p42: mem[37,8] <= 48
 capacity_mem_par_l1_p42: mem[37,9] <= 48
 capacity_mem_par_l1_p42: mem[37,10] <= 48
 capacity_mem_par_l1_p42: mem[37,11] <= 48
 capacity_mem_par_l1_p42: mem[37,12] <= 48
 ns_l1_p42: ns_l1_p42 = 6.756756756756757
 rows_tot_l2_p4: - frac[38,0] - frac[38,1] - frac[38,2] - frac[38,3]
   - frac[38,4] - frac[38,5] - frac[38,6] - frac[38,7] - frac[38,8]
   - frac[38,9] - frac[38,10] - frac[38,11] - frac[38,12] + rows_tot_l2_p4
   = 0
 mem_tot_l2_p4: - mem[38,0] - mem[38,1] - mem[38,2] - mem[38,3] - mem[38,4]
   - mem[38,5] - mem[38,6] - mem[38,7] - mem[38,8] - mem[38,9] - mem[38,10]
   - mem[38,11] - mem[38,12] + mem_tot_l2_p4 = 0
 capacity_mem_par_l2_p4: mem[38,0] <= 48
 capacity_mem_par_l2_p4: mem[38,1] <= 48
 capacity_mem_par_l2_p4: mem[38,2] <= 48
 capacity_mem_par_l2_p4: mem[38,3] <= 48
 capacity_mem_par_l2_p4: mem[38,4] <= 48
 capacity_mem_par_l2_p4: mem[38,5] <= 48
 capacity_mem_par_l2_p4: mem[38,6] <= 48
 capacity_mem_par_l2_p4: mem[38,7] <= 48
 capacity_mem_par_l2_p4: mem[38,8] <= 48
 capacity_mem_par_l2_p4: mem[38,9] <= 48
 capacity_mem_par_l2_p4: mem[38,10] <= 48
 capacity_mem_par_l2_p4: mem[38,11] <= 48
 capacity_mem_par_l2_p4: mem[38,12] <= 48
 ns_l2_p4: ns_l2_p4 = 6.756756756756757
 res: - 1.220703125e-04 mem_tot_endhost_cpu1 - 10 cores_sketch_endhost_cpu1
   - 10 cores_dpdk_endhost_cpu1 - 1.220703125e-04 mem_tot_endhost_cpu2
   - 10 cores_sketch_endhost_cpu2 - 10 cores_dpdk_endhost_cpu2
   - 1.220703125e-04 mem_tot_endhost_cpu3 - 10 cores_sketch_endhost_cpu3
   - 10 cores_dpdk_endhost_cpu3 - 1.220703125e-04 mem_tot_endhost_cpu4
   - 10 cores_sketch_endhost_cpu4 - 10 cores_dpdk_endhost_cpu4
   - 1.220703125e-04 mem_tot_endhost_cpu5 - 10 cores_sketch_endhost_cpu5
   - 10 cores_dpdk_endhost_cpu5 - 1.220703125e-04 mem_tot_endhost_cpu6
   - 10 cores_sketch_endhost_cpu6 - 10 cores_dpdk_endhost_cpu6
   - 1.220703125e-04 mem_tot_endhost_cpu7 - 10 cores_sketch_endhost_cpu7
   - 10 cores_dpdk_endhost_cpu7 - 1.220703125e-04 mem_tot_endhost_cpu8
   - 10 cores_sketch_endhost_cpu8 - 10 cores_dpdk_endhost_cpu8
   - 1.220703125e-04 mem_tot_endhost_cpu9 - 10 cores_sketch_endhost_cpu9
   - 10 cores_dpdk_endhost_cpu9 - 1.220703125e-04 mem_tot_endhost_cpu10
   - 10 cores_sketch_endhost_cpu10 - 10 cores_dpdk_endhost_cpu10
   - 1.220703125e-04 mem_tot_endhost_cpu11 - 10 cores_sketch_endhost_cpu11
   - 10 cores_dpdk_endhost_cpu11 - 1.220703125e-04 mem_tot_endhost_cpu12
   - 10 cores_sketch_endhost_cpu12 - 10 cores_dpdk_endhost_cpu12
   - 1.220703125e-04 mem_tot_endhost_cpu13 - 10 cores_sketch_endhost_cpu13
   - 10 cores_dpdk_endhost_cpu13 - 1.220703125e-04 mem_tot_endhost_cpu14
   - 10 cores_sketch_endhost_cpu14 - 10 cores_dpdk_endhost_cpu14
   - 1.220703125e-04 mem_tot_endhost_cpu15 - 10 cores_sketch_endhost_cpu15
   - 10 cores_dpdk_endhost_cpu15 - 1.220703125e-04 mem_tot_endhost_cpu16
   - 10 cores_sketch_endhost_cpu16 - 10 cores_dpdk_endhost_cpu16
   - 1.220703125e-04 mem_tot_endhost_cpu17 - 10 cores_sketch_endhost_cpu17
   - 10 cores_dpdk_endhost_cpu17 - 1.220703125e-04 mem_tot_endhost_cpu18
   - 10 cores_sketch_endhost_cpu18 - 10 cores_dpdk_endhost_cpu18
   - 1.220703125e-04 mem_tot_endhost_cpu19 - 10 cores_sketch_endhost_cpu19
   - 10 cores_dpdk_endhost_cpu19 - 1.220703125e-04 mem_tot_endhost_cpu20
   - 10 cores_sketch_endhost_cpu20 - 10 cores_dpdk_endhost_cpu20
   - 1.220703125e-04 mem_tot_endhost_cpu21 - 10 cores_sketch_endhost_cpu21
   - 10 cores_dpdk_endhost_cpu21 - 1.220703125e-04 mem_tot_endhost_cpu22
   - 10 cores_sketch_endhost_cpu22 - 10 cores_dpdk_endhost_cpu22
   - 1.220703125e-04 mem_tot_endhost_cpu23 - 10 cores_sketch_endhost_cpu23
   - 10 cores_dpdk_endhost_cpu23 - 1.220703125e-04 mem_tot_endhost_cpu24
   - 10 cores_sketch_endhost_cpu24 - 10 cores_dpdk_endhost_cpu24
   - 1.220703125e-04 mem_tot_endhost_cpu25 - 10 cores_sketch_endhost_cpu25
   - 10 cores_dpdk_endhost_cpu25 - 1.220703125e-04 mem_tot_endhost_cpu26
   - 10 cores_sketch_endhost_cpu26 - 10 cores_dpdk_endhost_cpu26
   - 1.220703125e-04 mem_tot_endhost_cpu27 - 10 cores_sketch_endhost_cpu27
   - 10 cores_dpdk_endhost_cpu27 - 1.220703125e-04 mem_tot_endhost_cpu28
   - 10 cores_sketch_endhost_cpu28 - 10 cores_dpdk_endhost_cpu28
   - 1.220703125e-04 mem_tot_endhost_cpu29 - 10 cores_sketch_endhost_cpu29
   - 10 cores_dpdk_endhost_cpu29 - 1.220703125e-04 mem_tot_endhost_cpu30
   - 10 cores_sketch_endhost_cpu30 - 10 cores_dpdk_endhost_cpu30
   - 1.220703125e-04 mem_tot_endhost_cpu31 - 10 cores_sketch_endhost_cpu31
   - 10 cores_dpdk_endhost_cpu31 - 1.220703125e-04 mem_tot_endhost_cpu32
   - 10 cores_sketch_endhost_cpu32 - 10 cores_dpdk_endhost_cpu32
   - 0.25 rows_tot_tor_p41 - 0.0208333333333333 mem_tot_tor_p41
   - 0.25 rows_tot_tor_p42 - 0.0208333333333333 mem_tot_tor_p42
   - 0.25 rows_tot_tor_p43 - 0.0208333333333333 mem_tot_tor_p43
   - 0.25 rows_tot_tor_p44 - 0.0208333333333333 mem_tot_tor_p44
   - 0.25 rows_tot_l1_p41 - 0.0208333333333333 mem_tot_l1_p41
   - 0.25 rows_tot_l1_p42 - 0.0208333333333333 mem_tot_l1_p42
   - 0.25 rows_tot_l2_p4 - 0.0208333333333333 mem_tot_l2_p4 + res_overall
   = 0
 pdt_m_rows_endhost_cpu1: pdt_m_rows_endhost_cpu1 + [
   - rows_tot_endhost_cpu1 * m_access_time_endhost_cpu1 ] = 0
 pdt_nsk_c_endhost_cpu1: pdt_nsk_c_endhost_cpu1 + [
   - cores_sketch_endhost_cpu1 * ns_sketch_endhost_cpu1 ] = 0
 pdt_nsc_dpdk_endhost_cpu1: pdt_nsc_dpdk_endhost_cpu1 + [
   - cores_dpdk_endhost_cpu1 * ns_dpdk_endhost_cpu1 ] = 0
 pdt_m_rows_endhost_cpu2: pdt_m_rows_endhost_cpu2 + [
   - rows_tot_endhost_cpu2 * m_access_time_endhost_cpu2 ] = 0
 pdt_nsk_c_endhost_cpu2: pdt_nsk_c_endhost_cpu2 + [
   - cores_sketch_endhost_cpu2 * ns_sketch_endhost_cpu2 ] = 0
 pdt_nsc_dpdk_endhost_cpu2: pdt_nsc_dpdk_endhost_cpu2 + [
   - cores_dpdk_endhost_cpu2 * ns_dpdk_endhost_cpu2 ] = 0
 pdt_m_rows_endhost_cpu3: pdt_m_rows_endhost_cpu3 + [
   - rows_tot_endhost_cpu3 * m_access_time_endhost_cpu3 ] = 0
 pdt_nsk_c_endhost_cpu3: pdt_nsk_c_endhost_cpu3 + [
   - cores_sketch_endhost_cpu3 * ns_sketch_endhost_cpu3 ] = 0
 pdt_nsc_dpdk_endhost_cpu3: pdt_nsc_dpdk_endhost_cpu3 + [
   - cores_dpdk_endhost_cpu3 * ns_dpdk_endhost_cpu3 ] = 0
 pdt_m_rows_endhost_cpu4: pdt_m_rows_endhost_cpu4 + [
   - rows_tot_endhost_cpu4 * m_access_time_endhost_cpu4 ] = 0
 pdt_nsk_c_endhost_cpu4: pdt_nsk_c_endhost_cpu4 + [
   - cores_sketch_endhost_cpu4 * ns_sketch_endhost_cpu4 ] = 0
 pdt_nsc_dpdk_endhost_cpu4: pdt_nsc_dpdk_endhost_cpu4 + [
   - cores_dpdk_endhost_cpu4 * ns_dpdk_endhost_cpu4 ] = 0
 pdt_m_rows_endhost_cpu5: pdt_m_rows_endhost_cpu5 + [
   - rows_tot_endhost_cpu5 * m_access_time_endhost_cpu5 ] = 0
 pdt_nsk_c_endhost_cpu5: pdt_nsk_c_endhost_cpu5 + [
   - cores_sketch_endhost_cpu5 * ns_sketch_endhost_cpu5 ] = 0
 pdt_nsc_dpdk_endhost_cpu5: pdt_nsc_dpdk_endhost_cpu5 + [
   - cores_dpdk_endhost_cpu5 * ns_dpdk_endhost_cpu5 ] = 0
 pdt_m_rows_endhost_cpu6: pdt_m_rows_endhost_cpu6 + [
   - rows_tot_endhost_cpu6 * m_access_time_endhost_cpu6 ] = 0
 pdt_nsk_c_endhost_cpu6: pdt_nsk_c_endhost_cpu6 + [
   - cores_sketch_endhost_cpu6 * ns_sketch_endhost_cpu6 ] = 0
 pdt_nsc_dpdk_endhost_cpu6: pdt_nsc_dpdk_endhost_cpu6 + [
   - cores_dpdk_endhost_cpu6 * ns_dpdk_endhost_cpu6 ] = 0
 pdt_m_rows_endhost_cpu7: pdt_m_rows_endhost_cpu7 + [
   - rows_tot_endhost_cpu7 * m_access_time_endhost_cpu7 ] = 0
 pdt_nsk_c_endhost_cpu7: pdt_nsk_c_endhost_cpu7 + [
   - cores_sketch_endhost_cpu7 * ns_sketch_endhost_cpu7 ] = 0
 pdt_nsc_dpdk_endhost_cpu7: pdt_nsc_dpdk_endhost_cpu7 + [
   - cores_dpdk_endhost_cpu7 * ns_dpdk_endhost_cpu7 ] = 0
 pdt_m_rows_endhost_cpu8: pdt_m_rows_endhost_cpu8 + [
   - rows_tot_endhost_cpu8 * m_access_time_endhost_cpu8 ] = 0
 pdt_nsk_c_endhost_cpu8: pdt_nsk_c_endhost_cpu8 + [
   - cores_sketch_endhost_cpu8 * ns_sketch_endhost_cpu8 ] = 0
 pdt_nsc_dpdk_endhost_cpu8: pdt_nsc_dpdk_endhost_cpu8 + [
   - cores_dpdk_endhost_cpu8 * ns_dpdk_endhost_cpu8 ] = 0
 pdt_m_rows_endhost_cpu9: pdt_m_rows_endhost_cpu9 + [
   - rows_tot_endhost_cpu9 * m_access_time_endhost_cpu9 ] = 0
 pdt_nsk_c_endhost_cpu9: pdt_nsk_c_endhost_cpu9 + [
   - cores_sketch_endhost_cpu9 * ns_sketch_endhost_cpu9 ] = 0
 pdt_nsc_dpdk_endhost_cpu9: pdt_nsc_dpdk_endhost_cpu9 + [
   - cores_dpdk_endhost_cpu9 * ns_dpdk_endhost_cpu9 ] = 0
 pdt_m_rows_endhost_cpu10: pdt_m_rows_endhost_cpu10 + [
   - rows_tot_endhost_cpu10 * m_access_time_endhost_cpu10 ] = 0
 pdt_nsk_c_endhost_cpu10: pdt_nsk_c_endhost_cpu10 + [
   - cores_sketch_endhost_cpu10 * ns_sketch_endhost_cpu10 ] = 0
 pdt_nsc_dpdk_endhost_cpu10: pdt_nsc_dpdk_endhost_cpu10 + [
   - cores_dpdk_endhost_cpu10 * ns_dpdk_endhost_cpu10 ] = 0
 pdt_m_rows_endhost_cpu11: pdt_m_rows_endhost_cpu11 + [
   - rows_tot_endhost_cpu11 * m_access_time_endhost_cpu11 ] = 0
 pdt_nsk_c_endhost_cpu11: pdt_nsk_c_endhost_cpu11 + [
   - cores_sketch_endhost_cpu11 * ns_sketch_endhost_cpu11 ] = 0
 pdt_nsc_dpdk_endhost_cpu11: pdt_nsc_dpdk_endhost_cpu11 + [
   - cores_dpdk_endhost_cpu11 * ns_dpdk_endhost_cpu11 ] = 0
 pdt_m_rows_endhost_cpu12: pdt_m_rows_endhost_cpu12 + [
   - rows_tot_endhost_cpu12 * m_access_time_endhost_cpu12 ] = 0
 pdt_nsk_c_endhost_cpu12: pdt_nsk_c_endhost_cpu12 + [
   - cores_sketch_endhost_cpu12 * ns_sketch_endhost_cpu12 ] = 0
 pdt_nsc_dpdk_endhost_cpu12: pdt_nsc_dpdk_endhost_cpu12 + [
   - cores_dpdk_endhost_cpu12 * ns_dpdk_endhost_cpu12 ] = 0
 pdt_m_rows_endhost_cpu13: pdt_m_rows_endhost_cpu13 + [
   - rows_tot_endhost_cpu13 * m_access_time_endhost_cpu13 ] = 0
 pdt_nsk_c_endhost_cpu13: pdt_nsk_c_endhost_cpu13 + [
   - cores_sketch_endhost_cpu13 * ns_sketch_endhost_cpu13 ] = 0
 pdt_nsc_dpdk_endhost_cpu13: pdt_nsc_dpdk_endhost_cpu13 + [
   - cores_dpdk_endhost_cpu13 * ns_dpdk_endhost_cpu13 ] = 0
 pdt_m_rows_endhost_cpu14: pdt_m_rows_endhost_cpu14 + [
   - rows_tot_endhost_cpu14 * m_access_time_endhost_cpu14 ] = 0
 pdt_nsk_c_endhost_cpu14: pdt_nsk_c_endhost_cpu14 + [
   - cores_sketch_endhost_cpu14 * ns_sketch_endhost_cpu14 ] = 0
 pdt_nsc_dpdk_endhost_cpu14: pdt_nsc_dpdk_endhost_cpu14 + [
   - cores_dpdk_endhost_cpu14 * ns_dpdk_endhost_cpu14 ] = 0
 pdt_m_rows_endhost_cpu15: pdt_m_rows_endhost_cpu15 + [
   - rows_tot_endhost_cpu15 * m_access_time_endhost_cpu15 ] = 0
 pdt_nsk_c_endhost_cpu15: pdt_nsk_c_endhost_cpu15 + [
   - cores_sketch_endhost_cpu15 * ns_sketch_endhost_cpu15 ] = 0
 pdt_nsc_dpdk_endhost_cpu15: pdt_nsc_dpdk_endhost_cpu15 + [
   - cores_dpdk_endhost_cpu15 * ns_dpdk_endhost_cpu15 ] = 0
 pdt_m_rows_endhost_cpu16: pdt_m_rows_endhost_cpu16 + [
   - rows_tot_endhost_cpu16 * m_access_time_endhost_cpu16 ] = 0
 pdt_nsk_c_endhost_cpu16: pdt_nsk_c_endhost_cpu16 + [
   - cores_sketch_endhost_cpu16 * ns_sketch_endhost_cpu16 ] = 0
 pdt_nsc_dpdk_endhost_cpu16: pdt_nsc_dpdk_endhost_cpu16 + [
   - cores_dpdk_endhost_cpu16 * ns_dpdk_endhost_cpu16 ] = 0
 pdt_m_rows_endhost_cpu17: pdt_m_rows_endhost_cpu17 + [
   - rows_tot_endhost_cpu17 * m_access_time_endhost_cpu17 ] = 0
 pdt_nsk_c_endhost_cpu17: pdt_nsk_c_endhost_cpu17 + [
   - cores_sketch_endhost_cpu17 * ns_sketch_endhost_cpu17 ] = 0
 pdt_nsc_dpdk_endhost_cpu17: pdt_nsc_dpdk_endhost_cpu17 + [
   - cores_dpdk_endhost_cpu17 * ns_dpdk_endhost_cpu17 ] = 0
 pdt_m_rows_endhost_cpu18: pdt_m_rows_endhost_cpu18 + [
   - rows_tot_endhost_cpu18 * m_access_time_endhost_cpu18 ] = 0
 pdt_nsk_c_endhost_cpu18: pdt_nsk_c_endhost_cpu18 + [
   - cores_sketch_endhost_cpu18 * ns_sketch_endhost_cpu18 ] = 0
 pdt_nsc_dpdk_endhost_cpu18: pdt_nsc_dpdk_endhost_cpu18 + [
   - cores_dpdk_endhost_cpu18 * ns_dpdk_endhost_cpu18 ] = 0
 pdt_m_rows_endhost_cpu19: pdt_m_rows_endhost_cpu19 + [
   - rows_tot_endhost_cpu19 * m_access_time_endhost_cpu19 ] = 0
 pdt_nsk_c_endhost_cpu19: pdt_nsk_c_endhost_cpu19 + [
   - cores_sketch_endhost_cpu19 * ns_sketch_endhost_cpu19 ] = 0
 pdt_nsc_dpdk_endhost_cpu19: pdt_nsc_dpdk_endhost_cpu19 + [
   - cores_dpdk_endhost_cpu19 * ns_dpdk_endhost_cpu19 ] = 0
 pdt_m_rows_endhost_cpu20: pdt_m_rows_endhost_cpu20 + [
   - rows_tot_endhost_cpu20 * m_access_time_endhost_cpu20 ] = 0
 pdt_nsk_c_endhost_cpu20: pdt_nsk_c_endhost_cpu20 + [
   - cores_sketch_endhost_cpu20 * ns_sketch_endhost_cpu20 ] = 0
 pdt_nsc_dpdk_endhost_cpu20: pdt_nsc_dpdk_endhost_cpu20 + [
   - cores_dpdk_endhost_cpu20 * ns_dpdk_endhost_cpu20 ] = 0
 pdt_m_rows_endhost_cpu21: pdt_m_rows_endhost_cpu21 + [
   - rows_tot_endhost_cpu21 * m_access_time_endhost_cpu21 ] = 0
 pdt_nsk_c_endhost_cpu21: pdt_nsk_c_endhost_cpu21 + [
   - cores_sketch_endhost_cpu21 * ns_sketch_endhost_cpu21 ] = 0
 pdt_nsc_dpdk_endhost_cpu21: pdt_nsc_dpdk_endhost_cpu21 + [
   - cores_dpdk_endhost_cpu21 * ns_dpdk_endhost_cpu21 ] = 0
 pdt_m_rows_endhost_cpu22: pdt_m_rows_endhost_cpu22 + [
   - rows_tot_endhost_cpu22 * m_access_time_endhost_cpu22 ] = 0
 pdt_nsk_c_endhost_cpu22: pdt_nsk_c_endhost_cpu22 + [
   - cores_sketch_endhost_cpu22 * ns_sketch_endhost_cpu22 ] = 0
 pdt_nsc_dpdk_endhost_cpu22: pdt_nsc_dpdk_endhost_cpu22 + [
   - cores_dpdk_endhost_cpu22 * ns_dpdk_endhost_cpu22 ] = 0
 pdt_m_rows_endhost_cpu23: pdt_m_rows_endhost_cpu23 + [
   - rows_tot_endhost_cpu23 * m_access_time_endhost_cpu23 ] = 0
 pdt_nsk_c_endhost_cpu23: pdt_nsk_c_endhost_cpu23 + [
   - cores_sketch_endhost_cpu23 * ns_sketch_endhost_cpu23 ] = 0
 pdt_nsc_dpdk_endhost_cpu23: pdt_nsc_dpdk_endhost_cpu23 + [
   - cores_dpdk_endhost_cpu23 * ns_dpdk_endhost_cpu23 ] = 0
 pdt_m_rows_endhost_cpu24: pdt_m_rows_endhost_cpu24 + [
   - rows_tot_endhost_cpu24 * m_access_time_endhost_cpu24 ] = 0
 pdt_nsk_c_endhost_cpu24: pdt_nsk_c_endhost_cpu24 + [
   - cores_sketch_endhost_cpu24 * ns_sketch_endhost_cpu24 ] = 0
 pdt_nsc_dpdk_endhost_cpu24: pdt_nsc_dpdk_endhost_cpu24 + [
   - cores_dpdk_endhost_cpu24 * ns_dpdk_endhost_cpu24 ] = 0
 pdt_m_rows_endhost_cpu25: pdt_m_rows_endhost_cpu25 + [
   - rows_tot_endhost_cpu25 * m_access_time_endhost_cpu25 ] = 0
 pdt_nsk_c_endhost_cpu25: pdt_nsk_c_endhost_cpu25 + [
   - cores_sketch_endhost_cpu25 * ns_sketch_endhost_cpu25 ] = 0
 pdt_nsc_dpdk_endhost_cpu25: pdt_nsc_dpdk_endhost_cpu25 + [
   - cores_dpdk_endhost_cpu25 * ns_dpdk_endhost_cpu25 ] = 0
 pdt_m_rows_endhost_cpu26: pdt_m_rows_endhost_cpu26 + [
   - rows_tot_endhost_cpu26 * m_access_time_endhost_cpu26 ] = 0
 pdt_nsk_c_endhost_cpu26: pdt_nsk_c_endhost_cpu26 + [
   - cores_sketch_endhost_cpu26 * ns_sketch_endhost_cpu26 ] = 0
 pdt_nsc_dpdk_endhost_cpu26: pdt_nsc_dpdk_endhost_cpu26 + [
   - cores_dpdk_endhost_cpu26 * ns_dpdk_endhost_cpu26 ] = 0
 pdt_m_rows_endhost_cpu27: pdt_m_rows_endhost_cpu27 + [
   - rows_tot_endhost_cpu27 * m_access_time_endhost_cpu27 ] = 0
 pdt_nsk_c_endhost_cpu27: pdt_nsk_c_endhost_cpu27 + [
   - cores_sketch_endhost_cpu27 * ns_sketch_endhost_cpu27 ] = 0
 pdt_nsc_dpdk_endhost_cpu27: pdt_nsc_dpdk_endhost_cpu27 + [
   - cores_dpdk_endhost_cpu27 * ns_dpdk_endhost_cpu27 ] = 0
 pdt_m_rows_endhost_cpu28: pdt_m_rows_endhost_cpu28 + [
   - rows_tot_endhost_cpu28 * m_access_time_endhost_cpu28 ] = 0
 pdt_nsk_c_endhost_cpu28: pdt_nsk_c_endhost_cpu28 + [
   - cores_sketch_endhost_cpu28 * ns_sketch_endhost_cpu28 ] = 0
 pdt_nsc_dpdk_endhost_cpu28: pdt_nsc_dpdk_endhost_cpu28 + [
   - cores_dpdk_endhost_cpu28 * ns_dpdk_endhost_cpu28 ] = 0
 pdt_m_rows_endhost_cpu29: pdt_m_rows_endhost_cpu29 + [
   - rows_tot_endhost_cpu29 * m_access_time_endhost_cpu29 ] = 0
 pdt_nsk_c_endhost_cpu29: pdt_nsk_c_endhost_cpu29 + [
   - cores_sketch_endhost_cpu29 * ns_sketch_endhost_cpu29 ] = 0
 pdt_nsc_dpdk_endhost_cpu29: pdt_nsc_dpdk_endhost_cpu29 + [
   - cores_dpdk_endhost_cpu29 * ns_dpdk_endhost_cpu29 ] = 0
 pdt_m_rows_endhost_cpu30: pdt_m_rows_endhost_cpu30 + [
   - rows_tot_endhost_cpu30 * m_access_time_endhost_cpu30 ] = 0
 pdt_nsk_c_endhost_cpu30: pdt_nsk_c_endhost_cpu30 + [
   - cores_sketch_endhost_cpu30 * ns_sketch_endhost_cpu30 ] = 0
 pdt_nsc_dpdk_endhost_cpu30: pdt_nsc_dpdk_endhost_cpu30 + [
   - cores_dpdk_endhost_cpu30 * ns_dpdk_endhost_cpu30 ] = 0
 pdt_m_rows_endhost_cpu31: pdt_m_rows_endhost_cpu31 + [
   - rows_tot_endhost_cpu31 * m_access_time_endhost_cpu31 ] = 0
 pdt_nsk_c_endhost_cpu31: pdt_nsk_c_endhost_cpu31 + [
   - cores_sketch_endhost_cpu31 * ns_sketch_endhost_cpu31 ] = 0
 pdt_nsc_dpdk_endhost_cpu31: pdt_nsc_dpdk_endhost_cpu31 + [
   - cores_dpdk_endhost_cpu31 * ns_dpdk_endhost_cpu31 ] = 0
 pdt_m_rows_endhost_cpu32: pdt_m_rows_endhost_cpu32 + [
   - rows_tot_endhost_cpu32 * m_access_time_endhost_cpu32 ] = 0
 pdt_nsk_c_endhost_cpu32: pdt_nsk_c_endhost_cpu32 + [
   - cores_sketch_endhost_cpu32 * ns_sketch_endhost_cpu32 ] = 0
 pdt_nsc_dpdk_endhost_cpu32: pdt_nsc_dpdk_endhost_cpu32 + [
   - cores_dpdk_endhost_cpu32 * ns_dpdk_endhost_cpu32 ] = 0
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
 frac[4,0] <= 1
 frac[4,1] <= 1
 frac[4,2] <= 1
 frac[4,3] <= 1
 frac[4,4] <= 1
 frac[4,5] <= 1
 frac[4,6] <= 1
 frac[4,7] <= 1
 frac[4,8] <= 1
 frac[4,9] <= 1
 frac[4,10] <= 1
 frac[4,11] <= 1
 frac[4,12] <= 1
 frac[5,0] <= 1
 frac[5,1] <= 1
 frac[5,2] <= 1
 frac[5,3] <= 1
 frac[5,4] <= 1
 frac[5,5] <= 1
 frac[5,6] <= 1
 frac[5,7] <= 1
 frac[5,8] <= 1
 frac[5,9] <= 1
 frac[5,10] <= 1
 frac[5,11] <= 1
 frac[5,12] <= 1
 frac[6,0] <= 1
 frac[6,1] <= 1
 frac[6,2] <= 1
 frac[6,3] <= 1
 frac[6,4] <= 1
 frac[6,5] <= 1
 frac[6,6] <= 1
 frac[6,7] <= 1
 frac[6,8] <= 1
 frac[6,9] <= 1
 frac[6,10] <= 1
 frac[6,11] <= 1
 frac[6,12] <= 1
 frac[7,0] <= 1
 frac[7,1] <= 1
 frac[7,2] <= 1
 frac[7,3] <= 1
 frac[7,4] <= 1
 frac[7,5] <= 1
 frac[7,6] <= 1
 frac[7,7] <= 1
 frac[7,8] <= 1
 frac[7,9] <= 1
 frac[7,10] <= 1
 frac[7,11] <= 1
 frac[7,12] <= 1
 frac[8,0] <= 1
 frac[8,1] <= 1
 frac[8,2] <= 1
 frac[8,3] <= 1
 frac[8,4] <= 1
 frac[8,5] <= 1
 frac[8,6] <= 1
 frac[8,7] <= 1
 frac[8,8] <= 1
 frac[8,9] <= 1
 frac[8,10] <= 1
 frac[8,11] <= 1
 frac[8,12] <= 1
 frac[9,0] <= 1
 frac[9,1] <= 1
 frac[9,2] <= 1
 frac[9,3] <= 1
 frac[9,4] <= 1
 frac[9,5] <= 1
 frac[9,6] <= 1
 frac[9,7] <= 1
 frac[9,8] <= 1
 frac[9,9] <= 1
 frac[9,10] <= 1
 frac[9,11] <= 1
 frac[9,12] <= 1
 frac[10,0] <= 1
 frac[10,1] <= 1
 frac[10,2] <= 1
 frac[10,3] <= 1
 frac[10,4] <= 1
 frac[10,5] <= 1
 frac[10,6] <= 1
 frac[10,7] <= 1
 frac[10,8] <= 1
 frac[10,9] <= 1
 frac[10,10] <= 1
 frac[10,11] <= 1
 frac[10,12] <= 1
 frac[11,0] <= 1
 frac[11,1] <= 1
 frac[11,2] <= 1
 frac[11,3] <= 1
 frac[11,4] <= 1
 frac[11,5] <= 1
 frac[11,6] <= 1
 frac[11,7] <= 1
 frac[11,8] <= 1
 frac[11,9] <= 1
 frac[11,10] <= 1
 frac[11,11] <= 1
 frac[11,12] <= 1
 frac[12,0] <= 1
 frac[12,1] <= 1
 frac[12,2] <= 1
 frac[12,3] <= 1
 frac[12,4] <= 1
 frac[12,5] <= 1
 frac[12,6] <= 1
 frac[12,7] <= 1
 frac[12,8] <= 1
 frac[12,9] <= 1
 frac[12,10] <= 1
 frac[12,11] <= 1
 frac[12,12] <= 1
 frac[13,0] <= 1
 frac[13,1] <= 1
 frac[13,2] <= 1
 frac[13,3] <= 1
 frac[13,4] <= 1
 frac[13,5] <= 1
 frac[13,6] <= 1
 frac[13,7] <= 1
 frac[13,8] <= 1
 frac[13,9] <= 1
 frac[13,10] <= 1
 frac[13,11] <= 1
 frac[13,12] <= 1
 frac[14,0] <= 1
 frac[14,1] <= 1
 frac[14,2] <= 1
 frac[14,3] <= 1
 frac[14,4] <= 1
 frac[14,5] <= 1
 frac[14,6] <= 1
 frac[14,7] <= 1
 frac[14,8] <= 1
 frac[14,9] <= 1
 frac[14,10] <= 1
 frac[14,11] <= 1
 frac[14,12] <= 1
 frac[15,0] <= 1
 frac[15,1] <= 1
 frac[15,2] <= 1
 frac[15,3] <= 1
 frac[15,4] <= 1
 frac[15,5] <= 1
 frac[15,6] <= 1
 frac[15,7] <= 1
 frac[15,8] <= 1
 frac[15,9] <= 1
 frac[15,10] <= 1
 frac[15,11] <= 1
 frac[15,12] <= 1
 frac[16,0] <= 1
 frac[16,1] <= 1
 frac[16,2] <= 1
 frac[16,3] <= 1
 frac[16,4] <= 1
 frac[16,5] <= 1
 frac[16,6] <= 1
 frac[16,7] <= 1
 frac[16,8] <= 1
 frac[16,9] <= 1
 frac[16,10] <= 1
 frac[16,11] <= 1
 frac[16,12] <= 1
 frac[17,0] <= 1
 frac[17,1] <= 1
 frac[17,2] <= 1
 frac[17,3] <= 1
 frac[17,4] <= 1
 frac[17,5] <= 1
 frac[17,6] <= 1
 frac[17,7] <= 1
 frac[17,8] <= 1
 frac[17,9] <= 1
 frac[17,10] <= 1
 frac[17,11] <= 1
 frac[17,12] <= 1
 frac[18,0] <= 1
 frac[18,1] <= 1
 frac[18,2] <= 1
 frac[18,3] <= 1
 frac[18,4] <= 1
 frac[18,5] <= 1
 frac[18,6] <= 1
 frac[18,7] <= 1
 frac[18,8] <= 1
 frac[18,9] <= 1
 frac[18,10] <= 1
 frac[18,11] <= 1
 frac[18,12] <= 1
 frac[19,0] <= 1
 frac[19,1] <= 1
 frac[19,2] <= 1
 frac[19,3] <= 1
 frac[19,4] <= 1
 frac[19,5] <= 1
 frac[19,6] <= 1
 frac[19,7] <= 1
 frac[19,8] <= 1
 frac[19,9] <= 1
 frac[19,10] <= 1
 frac[19,11] <= 1
 frac[19,12] <= 1
 frac[20,0] <= 1
 frac[20,1] <= 1
 frac[20,2] <= 1
 frac[20,3] <= 1
 frac[20,4] <= 1
 frac[20,5] <= 1
 frac[20,6] <= 1
 frac[20,7] <= 1
 frac[20,8] <= 1
 frac[20,9] <= 1
 frac[20,10] <= 1
 frac[20,11] <= 1
 frac[20,12] <= 1
 frac[21,0] <= 1
 frac[21,1] <= 1
 frac[21,2] <= 1
 frac[21,3] <= 1
 frac[21,4] <= 1
 frac[21,5] <= 1
 frac[21,6] <= 1
 frac[21,7] <= 1
 frac[21,8] <= 1
 frac[21,9] <= 1
 frac[21,10] <= 1
 frac[21,11] <= 1
 frac[21,12] <= 1
 frac[22,0] <= 1
 frac[22,1] <= 1
 frac[22,2] <= 1
 frac[22,3] <= 1
 frac[22,4] <= 1
 frac[22,5] <= 1
 frac[22,6] <= 1
 frac[22,7] <= 1
 frac[22,8] <= 1
 frac[22,9] <= 1
 frac[22,10] <= 1
 frac[22,11] <= 1
 frac[22,12] <= 1
 frac[23,0] <= 1
 frac[23,1] <= 1
 frac[23,2] <= 1
 frac[23,3] <= 1
 frac[23,4] <= 1
 frac[23,5] <= 1
 frac[23,6] <= 1
 frac[23,7] <= 1
 frac[23,8] <= 1
 frac[23,9] <= 1
 frac[23,10] <= 1
 frac[23,11] <= 1
 frac[23,12] <= 1
 frac[24,0] <= 1
 frac[24,1] <= 1
 frac[24,2] <= 1
 frac[24,3] <= 1
 frac[24,4] <= 1
 frac[24,5] <= 1
 frac[24,6] <= 1
 frac[24,7] <= 1
 frac[24,8] <= 1
 frac[24,9] <= 1
 frac[24,10] <= 1
 frac[24,11] <= 1
 frac[24,12] <= 1
 frac[25,0] <= 1
 frac[25,1] <= 1
 frac[25,2] <= 1
 frac[25,3] <= 1
 frac[25,4] <= 1
 frac[25,5] <= 1
 frac[25,6] <= 1
 frac[25,7] <= 1
 frac[25,8] <= 1
 frac[25,9] <= 1
 frac[25,10] <= 1
 frac[25,11] <= 1
 frac[25,12] <= 1
 frac[26,0] <= 1
 frac[26,1] <= 1
 frac[26,2] <= 1
 frac[26,3] <= 1
 frac[26,4] <= 1
 frac[26,5] <= 1
 frac[26,6] <= 1
 frac[26,7] <= 1
 frac[26,8] <= 1
 frac[26,9] <= 1
 frac[26,10] <= 1
 frac[26,11] <= 1
 frac[26,12] <= 1
 frac[27,0] <= 1
 frac[27,1] <= 1
 frac[27,2] <= 1
 frac[27,3] <= 1
 frac[27,4] <= 1
 frac[27,5] <= 1
 frac[27,6] <= 1
 frac[27,7] <= 1
 frac[27,8] <= 1
 frac[27,9] <= 1
 frac[27,10] <= 1
 frac[27,11] <= 1
 frac[27,12] <= 1
 frac[28,0] <= 1
 frac[28,1] <= 1
 frac[28,2] <= 1
 frac[28,3] <= 1
 frac[28,4] <= 1
 frac[28,5] <= 1
 frac[28,6] <= 1
 frac[28,7] <= 1
 frac[28,8] <= 1
 frac[28,9] <= 1
 frac[28,10] <= 1
 frac[28,11] <= 1
 frac[28,12] <= 1
 frac[29,0] <= 1
 frac[29,1] <= 1
 frac[29,2] <= 1
 frac[29,3] <= 1
 frac[29,4] <= 1
 frac[29,5] <= 1
 frac[29,6] <= 1
 frac[29,7] <= 1
 frac[29,8] <= 1
 frac[29,9] <= 1
 frac[29,10] <= 1
 frac[29,11] <= 1
 frac[29,12] <= 1
 frac[30,0] <= 1
 frac[30,1] <= 1
 frac[30,2] <= 1
 frac[30,3] <= 1
 frac[30,4] <= 1
 frac[30,5] <= 1
 frac[30,6] <= 1
 frac[30,7] <= 1
 frac[30,8] <= 1
 frac[30,9] <= 1
 frac[30,10] <= 1
 frac[30,11] <= 1
 frac[30,12] <= 1
 frac[31,0] <= 1
 frac[31,1] <= 1
 frac[31,2] <= 1
 frac[31,3] <= 1
 frac[31,4] <= 1
 frac[31,5] <= 1
 frac[31,6] <= 1
 frac[31,7] <= 1
 frac[31,8] <= 1
 frac[31,9] <= 1
 frac[31,10] <= 1
 frac[31,11] <= 1
 frac[31,12] <= 1
 frac[32,0] <= 1
 frac[32,1] <= 1
 frac[32,2] <= 1
 frac[32,3] <= 1
 frac[32,4] <= 1
 frac[32,5] <= 1
 frac[32,6] <= 1
 frac[32,7] <= 1
 frac[32,8] <= 1
 frac[32,9] <= 1
 frac[32,10] <= 1
 frac[32,11] <= 1
 frac[32,12] <= 1
 frac[33,0] <= 1
 frac[33,1] <= 1
 frac[33,2] <= 1
 frac[33,3] <= 1
 frac[33,4] <= 1
 frac[33,5] <= 1
 frac[33,6] <= 1
 frac[33,7] <= 1
 frac[33,8] <= 1
 frac[33,9] <= 1
 frac[33,10] <= 1
 frac[33,11] <= 1
 frac[33,12] <= 1
 frac[34,0] <= 1
 frac[34,1] <= 1
 frac[34,2] <= 1
 frac[34,3] <= 1
 frac[34,4] <= 1
 frac[34,5] <= 1
 frac[34,6] <= 1
 frac[34,7] <= 1
 frac[34,8] <= 1
 frac[34,9] <= 1
 frac[34,10] <= 1
 frac[34,11] <= 1
 frac[34,12] <= 1
 frac[35,0] <= 1
 frac[35,1] <= 1
 frac[35,2] <= 1
 frac[35,3] <= 1
 frac[35,4] <= 1
 frac[35,5] <= 1
 frac[35,6] <= 1
 frac[35,7] <= 1
 frac[35,8] <= 1
 frac[35,9] <= 1
 frac[35,10] <= 1
 frac[35,11] <= 1
 frac[35,12] <= 1
 frac[36,0] <= 1
 frac[36,1] <= 1
 frac[36,2] <= 1
 frac[36,3] <= 1
 frac[36,4] <= 1
 frac[36,5] <= 1
 frac[36,6] <= 1
 frac[36,7] <= 1
 frac[36,8] <= 1
 frac[36,9] <= 1
 frac[36,10] <= 1
 frac[36,11] <= 1
 frac[36,12] <= 1
 frac[37,0] <= 1
 frac[37,1] <= 1
 frac[37,2] <= 1
 frac[37,3] <= 1
 frac[37,4] <= 1
 frac[37,5] <= 1
 frac[37,6] <= 1
 frac[37,7] <= 1
 frac[37,8] <= 1
 frac[37,9] <= 1
 frac[37,10] <= 1
 frac[37,11] <= 1
 frac[37,12] <= 1
 frac[38,0] <= 1
 frac[38,1] <= 1
 frac[38,2] <= 1
 frac[38,3] <= 1
 frac[38,4] <= 1
 frac[38,5] <= 1
 frac[38,6] <= 1
 frac[38,7] <= 1
 frac[38,8] <= 1
 frac[38,9] <= 1
 frac[38,10] <= 1
 frac[38,11] <= 1
 frac[38,12] <= 1
 rows_tot_endhost_cpu1 <= 9
 mem_tot_endhost_cpu1 <= 32768
 cores_sketch_endhost_cpu1 <= 7
 3 <= cores_dpdk_endhost_cpu1 <= 7
 rows_tot_endhost_cpu2 <= 9
 mem_tot_endhost_cpu2 <= 32768
 cores_sketch_endhost_cpu2 <= 7
 3 <= cores_dpdk_endhost_cpu2 <= 7
 rows_tot_endhost_cpu3 <= 9
 mem_tot_endhost_cpu3 <= 32768
 cores_sketch_endhost_cpu3 <= 7
 3 <= cores_dpdk_endhost_cpu3 <= 7
 rows_tot_endhost_cpu4 <= 9
 mem_tot_endhost_cpu4 <= 32768
 cores_sketch_endhost_cpu4 <= 7
 3 <= cores_dpdk_endhost_cpu4 <= 7
 rows_tot_endhost_cpu5 <= 9
 mem_tot_endhost_cpu5 <= 32768
 cores_sketch_endhost_cpu5 <= 7
 3 <= cores_dpdk_endhost_cpu5 <= 7
 rows_tot_endhost_cpu6 <= 9
 mem_tot_endhost_cpu6 <= 32768
 cores_sketch_endhost_cpu6 <= 7
 3 <= cores_dpdk_endhost_cpu6 <= 7
 rows_tot_endhost_cpu7 <= 9
 mem_tot_endhost_cpu7 <= 32768
 cores_sketch_endhost_cpu7 <= 7
 3 <= cores_dpdk_endhost_cpu7 <= 7
 rows_tot_endhost_cpu8 <= 9
 mem_tot_endhost_cpu8 <= 32768
 cores_sketch_endhost_cpu8 <= 7
 3 <= cores_dpdk_endhost_cpu8 <= 7
 rows_tot_endhost_cpu9 <= 9
 mem_tot_endhost_cpu9 <= 32768
 cores_sketch_endhost_cpu9 <= 7
 3 <= cores_dpdk_endhost_cpu9 <= 7
 rows_tot_endhost_cpu10 <= 9
 mem_tot_endhost_cpu10 <= 32768
 cores_sketch_endhost_cpu10 <= 7
 3 <= cores_dpdk_endhost_cpu10 <= 7
 rows_tot_endhost_cpu11 <= 9
 mem_tot_endhost_cpu11 <= 32768
 cores_sketch_endhost_cpu11 <= 7
 3 <= cores_dpdk_endhost_cpu11 <= 7
 rows_tot_endhost_cpu12 <= 9
 mem_tot_endhost_cpu12 <= 32768
 cores_sketch_endhost_cpu12 <= 7
 3 <= cores_dpdk_endhost_cpu12 <= 7
 rows_tot_endhost_cpu13 <= 9
 mem_tot_endhost_cpu13 <= 32768
 cores_sketch_endhost_cpu13 <= 7
 3 <= cores_dpdk_endhost_cpu13 <= 7
 rows_tot_endhost_cpu14 <= 9
 mem_tot_endhost_cpu14 <= 32768
 cores_sketch_endhost_cpu14 <= 7
 3 <= cores_dpdk_endhost_cpu14 <= 7
 rows_tot_endhost_cpu15 <= 9
 mem_tot_endhost_cpu15 <= 32768
 cores_sketch_endhost_cpu15 <= 7
 3 <= cores_dpdk_endhost_cpu15 <= 7
 rows_tot_endhost_cpu16 <= 9
 mem_tot_endhost_cpu16 <= 32768
 cores_sketch_endhost_cpu16 <= 7
 3 <= cores_dpdk_endhost_cpu16 <= 7
 rows_tot_endhost_cpu17 <= 9
 mem_tot_endhost_cpu17 <= 32768
 cores_sketch_endhost_cpu17 <= 7
 3 <= cores_dpdk_endhost_cpu17 <= 7
 rows_tot_endhost_cpu18 <= 9
 mem_tot_endhost_cpu18 <= 32768
 cores_sketch_endhost_cpu18 <= 7
 3 <= cores_dpdk_endhost_cpu18 <= 7
 rows_tot_endhost_cpu19 <= 9
 mem_tot_endhost_cpu19 <= 32768
 cores_sketch_endhost_cpu19 <= 7
 3 <= cores_dpdk_endhost_cpu19 <= 7
 rows_tot_endhost_cpu20 <= 9
 mem_tot_endhost_cpu20 <= 32768
 cores_sketch_endhost_cpu20 <= 7
 3 <= cores_dpdk_endhost_cpu20 <= 7
 rows_tot_endhost_cpu21 <= 9
 mem_tot_endhost_cpu21 <= 32768
 cores_sketch_endhost_cpu21 <= 7
 3 <= cores_dpdk_endhost_cpu21 <= 7
 rows_tot_endhost_cpu22 <= 9
 mem_tot_endhost_cpu22 <= 32768
 cores_sketch_endhost_cpu22 <= 7
 3 <= cores_dpdk_endhost_cpu22 <= 7
 rows_tot_endhost_cpu23 <= 9
 mem_tot_endhost_cpu23 <= 32768
 cores_sketch_endhost_cpu23 <= 7
 3 <= cores_dpdk_endhost_cpu23 <= 7
 rows_tot_endhost_cpu24 <= 9
 mem_tot_endhost_cpu24 <= 32768
 cores_sketch_endhost_cpu24 <= 7
 3 <= cores_dpdk_endhost_cpu24 <= 7
 rows_tot_endhost_cpu25 <= 9
 mem_tot_endhost_cpu25 <= 32768
 cores_sketch_endhost_cpu25 <= 7
 3 <= cores_dpdk_endhost_cpu25 <= 7
 rows_tot_endhost_cpu26 <= 9
 mem_tot_endhost_cpu26 <= 32768
 cores_sketch_endhost_cpu26 <= 7
 3 <= cores_dpdk_endhost_cpu26 <= 7
 rows_tot_endhost_cpu27 <= 9
 mem_tot_endhost_cpu27 <= 32768
 cores_sketch_endhost_cpu27 <= 7
 3 <= cores_dpdk_endhost_cpu27 <= 7
 rows_tot_endhost_cpu28 <= 9
 mem_tot_endhost_cpu28 <= 32768
 cores_sketch_endhost_cpu28 <= 7
 3 <= cores_dpdk_endhost_cpu28 <= 7
 rows_tot_endhost_cpu29 <= 9
 mem_tot_endhost_cpu29 <= 32768
 cores_sketch_endhost_cpu29 <= 7
 3 <= cores_dpdk_endhost_cpu29 <= 7
 rows_tot_endhost_cpu30 <= 9
 mem_tot_endhost_cpu30 <= 32768
 cores_sketch_endhost_cpu30 <= 7
 3 <= cores_dpdk_endhost_cpu30 <= 7
 rows_tot_endhost_cpu31 <= 9
 mem_tot_endhost_cpu31 <= 32768
 cores_sketch_endhost_cpu31 <= 7
 3 <= cores_dpdk_endhost_cpu31 <= 7
 rows_tot_endhost_cpu32 <= 9
 mem_tot_endhost_cpu32 <= 32768
 cores_sketch_endhost_cpu32 <= 7
 3 <= cores_dpdk_endhost_cpu32 <= 7
 rows_tot_tor_p41 <= 12
 mem_tot_tor_p41 <= 576
 rows_tot_tor_p42 <= 12
 mem_tot_tor_p42 <= 576
 rows_tot_tor_p43 <= 12
 mem_tot_tor_p43 <= 576
 rows_tot_tor_p44 <= 12
 mem_tot_tor_p44 <= 576
 rows_tot_l1_p41 <= 12
 mem_tot_l1_p41 <= 576
 rows_tot_l1_p42 <= 12
 mem_tot_l1_p42 <= 576
 rows_tot_l2_p4 <= 12
 mem_tot_l2_p4 <= 576
Generals
 cores_sketch_endhost_cpu1 cores_dpdk_endhost_cpu1
 cores_sketch_endhost_cpu2 cores_dpdk_endhost_cpu2
 cores_sketch_endhost_cpu3 cores_dpdk_endhost_cpu3
 cores_sketch_endhost_cpu4 cores_dpdk_endhost_cpu4
 cores_sketch_endhost_cpu5 cores_dpdk_endhost_cpu5
 cores_sketch_endhost_cpu6 cores_dpdk_endhost_cpu6
 cores_sketch_endhost_cpu7 cores_dpdk_endhost_cpu7
 cores_sketch_endhost_cpu8 cores_dpdk_endhost_cpu8
 cores_sketch_endhost_cpu9 cores_dpdk_endhost_cpu9
 cores_sketch_endhost_cpu10 cores_dpdk_endhost_cpu10
 cores_sketch_endhost_cpu11 cores_dpdk_endhost_cpu11
 cores_sketch_endhost_cpu12 cores_dpdk_endhost_cpu12
 cores_sketch_endhost_cpu13 cores_dpdk_endhost_cpu13
 cores_sketch_endhost_cpu14 cores_dpdk_endhost_cpu14
 cores_sketch_endhost_cpu15 cores_dpdk_endhost_cpu15
 cores_sketch_endhost_cpu16 cores_dpdk_endhost_cpu16
 cores_sketch_endhost_cpu17 cores_dpdk_endhost_cpu17
 cores_sketch_endhost_cpu18 cores_dpdk_endhost_cpu18
 cores_sketch_endhost_cpu19 cores_dpdk_endhost_cpu19
 cores_sketch_endhost_cpu20 cores_dpdk_endhost_cpu20
 cores_sketch_endhost_cpu21 cores_dpdk_endhost_cpu21
 cores_sketch_endhost_cpu22 cores_dpdk_endhost_cpu22
 cores_sketch_endhost_cpu23 cores_dpdk_endhost_cpu23
 cores_sketch_endhost_cpu24 cores_dpdk_endhost_cpu24
 cores_sketch_endhost_cpu25 cores_dpdk_endhost_cpu25
 cores_sketch_endhost_cpu26 cores_dpdk_endhost_cpu26
 cores_sketch_endhost_cpu27 cores_dpdk_endhost_cpu27
 cores_sketch_endhost_cpu28 cores_dpdk_endhost_cpu28
 cores_sketch_endhost_cpu29 cores_dpdk_endhost_cpu29
 cores_sketch_endhost_cpu30 cores_dpdk_endhost_cpu30
 cores_sketch_endhost_cpu31 cores_dpdk_endhost_cpu31
 cores_sketch_endhost_cpu32 cores_dpdk_endhost_cpu32
General Constraints
 mem_access_time_endhost_cpu1:
   m_access_time_endhost_cpu1 = PWL ( mem_tot_endhost_cpu1 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu1: ns_endhost_cpu1 = MAX ( ns_dpdk_endhost_cpu1 ,
   ns_sketch_endhost_cpu1 )
 mem_access_time_endhost_cpu2:
   m_access_time_endhost_cpu2 = PWL ( mem_tot_endhost_cpu2 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu2: ns_endhost_cpu2 = MAX ( ns_dpdk_endhost_cpu2 ,
   ns_sketch_endhost_cpu2 )
 mem_access_time_endhost_cpu3:
   m_access_time_endhost_cpu3 = PWL ( mem_tot_endhost_cpu3 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu3: ns_endhost_cpu3 = MAX ( ns_dpdk_endhost_cpu3 ,
   ns_sketch_endhost_cpu3 )
 mem_access_time_endhost_cpu4:
   m_access_time_endhost_cpu4 = PWL ( mem_tot_endhost_cpu4 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu4: ns_endhost_cpu4 = MAX ( ns_dpdk_endhost_cpu4 ,
   ns_sketch_endhost_cpu4 )
 mem_access_time_endhost_cpu5:
   m_access_time_endhost_cpu5 = PWL ( mem_tot_endhost_cpu5 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu5: ns_endhost_cpu5 = MAX ( ns_dpdk_endhost_cpu5 ,
   ns_sketch_endhost_cpu5 )
 mem_access_time_endhost_cpu6:
   m_access_time_endhost_cpu6 = PWL ( mem_tot_endhost_cpu6 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu6: ns_endhost_cpu6 = MAX ( ns_dpdk_endhost_cpu6 ,
   ns_sketch_endhost_cpu6 )
 mem_access_time_endhost_cpu7:
   m_access_time_endhost_cpu7 = PWL ( mem_tot_endhost_cpu7 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu7: ns_endhost_cpu7 = MAX ( ns_dpdk_endhost_cpu7 ,
   ns_sketch_endhost_cpu7 )
 mem_access_time_endhost_cpu8:
   m_access_time_endhost_cpu8 = PWL ( mem_tot_endhost_cpu8 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu8: ns_endhost_cpu8 = MAX ( ns_dpdk_endhost_cpu8 ,
   ns_sketch_endhost_cpu8 )
 mem_access_time_endhost_cpu9:
   m_access_time_endhost_cpu9 = PWL ( mem_tot_endhost_cpu9 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu9: ns_endhost_cpu9 = MAX ( ns_dpdk_endhost_cpu9 ,
   ns_sketch_endhost_cpu9 )
 mem_access_time_endhost_cpu10:
   m_access_time_endhost_cpu10 = PWL ( mem_tot_endhost_cpu10 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu10: ns_endhost_cpu10 = MAX ( ns_dpdk_endhost_cpu10 ,
   ns_sketch_endhost_cpu10 )
 mem_access_time_endhost_cpu11:
   m_access_time_endhost_cpu11 = PWL ( mem_tot_endhost_cpu11 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu11: ns_endhost_cpu11 = MAX ( ns_dpdk_endhost_cpu11 ,
   ns_sketch_endhost_cpu11 )
 mem_access_time_endhost_cpu12:
   m_access_time_endhost_cpu12 = PWL ( mem_tot_endhost_cpu12 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu12: ns_endhost_cpu12 = MAX ( ns_dpdk_endhost_cpu12 ,
   ns_sketch_endhost_cpu12 )
 mem_access_time_endhost_cpu13:
   m_access_time_endhost_cpu13 = PWL ( mem_tot_endhost_cpu13 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu13: ns_endhost_cpu13 = MAX ( ns_dpdk_endhost_cpu13 ,
   ns_sketch_endhost_cpu13 )
 mem_access_time_endhost_cpu14:
   m_access_time_endhost_cpu14 = PWL ( mem_tot_endhost_cpu14 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu14: ns_endhost_cpu14 = MAX ( ns_dpdk_endhost_cpu14 ,
   ns_sketch_endhost_cpu14 )
 mem_access_time_endhost_cpu15:
   m_access_time_endhost_cpu15 = PWL ( mem_tot_endhost_cpu15 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu15: ns_endhost_cpu15 = MAX ( ns_dpdk_endhost_cpu15 ,
   ns_sketch_endhost_cpu15 )
 mem_access_time_endhost_cpu16:
   m_access_time_endhost_cpu16 = PWL ( mem_tot_endhost_cpu16 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu16: ns_endhost_cpu16 = MAX ( ns_dpdk_endhost_cpu16 ,
   ns_sketch_endhost_cpu16 )
 mem_access_time_endhost_cpu17:
   m_access_time_endhost_cpu17 = PWL ( mem_tot_endhost_cpu17 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu17: ns_endhost_cpu17 = MAX ( ns_dpdk_endhost_cpu17 ,
   ns_sketch_endhost_cpu17 )
 mem_access_time_endhost_cpu18:
   m_access_time_endhost_cpu18 = PWL ( mem_tot_endhost_cpu18 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu18: ns_endhost_cpu18 = MAX ( ns_dpdk_endhost_cpu18 ,
   ns_sketch_endhost_cpu18 )
 mem_access_time_endhost_cpu19:
   m_access_time_endhost_cpu19 = PWL ( mem_tot_endhost_cpu19 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu19: ns_endhost_cpu19 = MAX ( ns_dpdk_endhost_cpu19 ,
   ns_sketch_endhost_cpu19 )
 mem_access_time_endhost_cpu20:
   m_access_time_endhost_cpu20 = PWL ( mem_tot_endhost_cpu20 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu20: ns_endhost_cpu20 = MAX ( ns_dpdk_endhost_cpu20 ,
   ns_sketch_endhost_cpu20 )
 mem_access_time_endhost_cpu21:
   m_access_time_endhost_cpu21 = PWL ( mem_tot_endhost_cpu21 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu21: ns_endhost_cpu21 = MAX ( ns_dpdk_endhost_cpu21 ,
   ns_sketch_endhost_cpu21 )
 mem_access_time_endhost_cpu22:
   m_access_time_endhost_cpu22 = PWL ( mem_tot_endhost_cpu22 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu22: ns_endhost_cpu22 = MAX ( ns_dpdk_endhost_cpu22 ,
   ns_sketch_endhost_cpu22 )
 mem_access_time_endhost_cpu23:
   m_access_time_endhost_cpu23 = PWL ( mem_tot_endhost_cpu23 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu23: ns_endhost_cpu23 = MAX ( ns_dpdk_endhost_cpu23 ,
   ns_sketch_endhost_cpu23 )
 mem_access_time_endhost_cpu24:
   m_access_time_endhost_cpu24 = PWL ( mem_tot_endhost_cpu24 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu24: ns_endhost_cpu24 = MAX ( ns_dpdk_endhost_cpu24 ,
   ns_sketch_endhost_cpu24 )
 mem_access_time_endhost_cpu25:
   m_access_time_endhost_cpu25 = PWL ( mem_tot_endhost_cpu25 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu25: ns_endhost_cpu25 = MAX ( ns_dpdk_endhost_cpu25 ,
   ns_sketch_endhost_cpu25 )
 mem_access_time_endhost_cpu26:
   m_access_time_endhost_cpu26 = PWL ( mem_tot_endhost_cpu26 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu26: ns_endhost_cpu26 = MAX ( ns_dpdk_endhost_cpu26 ,
   ns_sketch_endhost_cpu26 )
 mem_access_time_endhost_cpu27:
   m_access_time_endhost_cpu27 = PWL ( mem_tot_endhost_cpu27 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu27: ns_endhost_cpu27 = MAX ( ns_dpdk_endhost_cpu27 ,
   ns_sketch_endhost_cpu27 )
 mem_access_time_endhost_cpu28:
   m_access_time_endhost_cpu28 = PWL ( mem_tot_endhost_cpu28 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu28: ns_endhost_cpu28 = MAX ( ns_dpdk_endhost_cpu28 ,
   ns_sketch_endhost_cpu28 )
 mem_access_time_endhost_cpu29:
   m_access_time_endhost_cpu29 = PWL ( mem_tot_endhost_cpu29 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu29: ns_endhost_cpu29 = MAX ( ns_dpdk_endhost_cpu29 ,
   ns_sketch_endhost_cpu29 )
 mem_access_time_endhost_cpu30:
   m_access_time_endhost_cpu30 = PWL ( mem_tot_endhost_cpu30 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu30: ns_endhost_cpu30 = MAX ( ns_dpdk_endhost_cpu30 ,
   ns_sketch_endhost_cpu30 )
 mem_access_time_endhost_cpu31:
   m_access_time_endhost_cpu31 = PWL ( mem_tot_endhost_cpu31 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu31: ns_endhost_cpu31 = MAX ( ns_dpdk_endhost_cpu31 ,
   ns_sketch_endhost_cpu31 )
 mem_access_time_endhost_cpu32:
   m_access_time_endhost_cpu32 = PWL ( mem_tot_endhost_cpu32 ) : (0, 0) 
  (1.1875, 0.539759) (32, 0.510892) (1448.15625, 5.04469) 
  (5792.625, 5.84114) (32768, 30.6627) (440871.90625, 39.6981)
 ns_endhost_cpu32: ns_endhost_cpu32 = MAX ( ns_dpdk_endhost_cpu32 ,
   ns_sketch_endhost_cpu32 )
 ns_overall: ns = MAX ( ns_endhost_cpu1 , ns_endhost_cpu2 ,
   ns_endhost_cpu3 , ns_endhost_cpu4 , ns_endhost_cpu5 , ns_endhost_cpu6 ,
   ns_endhost_cpu7 , ns_endhost_cpu8 , ns_endhost_cpu9 , ns_endhost_cpu10 ,
   ns_endhost_cpu11 , ns_endhost_cpu12 , ns_endhost_cpu13 ,
   ns_endhost_cpu14 , ns_endhost_cpu15 , ns_endhost_cpu16 ,
   ns_endhost_cpu17 , ns_endhost_cpu18 , ns_endhost_cpu19 ,
   ns_endhost_cpu20 , ns_endhost_cpu21 , ns_endhost_cpu22 ,
   ns_endhost_cpu23 , ns_endhost_cpu24 , ns_endhost_cpu25 ,
   ns_endhost_cpu26 , ns_endhost_cpu27 , ns_endhost_cpu28 ,
   ns_endhost_cpu29 , ns_endhost_cpu30 , ns_endhost_cpu31 ,
   ns_endhost_cpu32 , ns_tor_p41 , ns_tor_p42 , ns_tor_p43 , ns_tor_p44 ,
   ns_l1_p41 , ns_l1_p42 , ns_l2_p4 )
End

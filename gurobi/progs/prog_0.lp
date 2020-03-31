\ Model Netmon
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  res: Priority=10 Weight=1 AbsTol=0 RelTol=0
   0 max_mem_P4 + 0 max_rows_P4 + res
Subject To
 cov_0_7: frac[0,7] >= 1
 cov_1_7: frac[0,7] >= 0.7
 cov_2_7: frac[0,7] >= 0.7
 cov_3_7: frac[0,7] >= 0.7
 cov_4_7: frac[0,7] >= 0.9
 cov_5_102: frac[0,102] >= 0.7
 cov_6_102: frac[0,102] >= 0.9
 cov_7_102: frac[0,102] >= 0.9
 cov_8_184: frac[0,184] >= 0.7
 cov_9_184: frac[0,184] >= 1
 cov_10_184: frac[0,184] >= 0.7
 cov_11_184: frac[0,184] >= 0.9
 cov_12_302: frac[0,302] >= 0.7
 cov_13_302: frac[0,302] >= 1
 cov_14_302: frac[0,302] >= 0.7
 cov_15_302: frac[0,302] >= 1
 cov_16_582: frac[0,582] >= 0.9
 cov_17_582: frac[0,582] >= 0.7
 cov_18_582: frac[0,582] >= 0.7
 cov_19_582: frac[0,582] >= 0.8
 cov_20_1052: frac[0,1052] >= 1
 cov_21_1052: frac[0,1052] >= 0.9
 cov_22_1052: frac[0,1052] >= 1
 cov_23_1052: frac[0,1052] >= 0.7
 cov_24_1103: frac[0,1103] >= 0.9
 cov_25_1103: frac[0,1103] >= 1
 cov_26_1103: frac[0,1103] >= 0.9
 cov_27_1103: frac[0,1103] >= 0.8
 cov_28_1212: frac[0,1212] >= 0.7
 cov_29_1212: frac[0,1212] >= 0.7
 cov_30_1212: frac[0,1212] >= 1
 cov_31_1212: frac[0,1212] >= 0.8
 cov_32_1212: frac[0,1212] >= 0.7
 cov_33_1395: frac[0,1395] >= 1
 cov_34_1395: frac[0,1395] >= 0.9
 cov_35_1395: frac[0,1395] >= 0.9
 cov_36_1395: frac[0,1395] >= 1
 cov_37_1681: frac[0,1681] >= 0.8
 cov_38_1681: frac[0,1681] >= 0.9
 cov_39_1681: frac[0,1681] >= 0.9
 cov_40_1681: frac[0,1681] >= 1
 cov_41_1681: frac[0,1681] >= 1
 cov_42_1715: frac[0,1715] >= 0.9
 cov_43_1715: frac[0,1715] >= 1
 cov_44_1715: frac[0,1715] >= 0.8
 cov_45_1715: frac[0,1715] >= 0.9
 cov_46_1828: frac[0,1828] >= 0.9
 cov_47_1828: frac[0,1828] >= 0.7
 cov_48_1828: frac[0,1828] >= 0.9
 cov_49_1828: frac[0,1828] >= 0.8
 cov_50_1884: frac[0,1884] >= 0.9
 cov_51_1884: frac[0,1884] >= 0.7
 cov_52_1884: frac[0,1884] >= 0.7
 cov_53_1884: frac[0,1884] >= 1
 cov_54_1923: frac[0,1923] >= 1
 cov_55_1923: frac[0,1923] >= 0.8
 cov_56_1923: frac[0,1923] >= 0.9
 cov_57_1985: frac[0,1985] >= 0.7
 cov_58_1985: frac[0,1985] >= 1
 cov_59_1985: frac[0,1985] >= 1
 cov_60_1985: frac[0,1985] >= 0.8
 cov_61_2369: frac[0,2369] >= 1
 cov_62_2369: frac[0,2369] >= 0.9
 cov_63_2369: frac[0,2369] >= 0.7
 accuracy_0_1985: - 5.09765625 frac[0,1985] + mem[0,1985] = 0
 accuracy_0_2369: - 5.09765625 frac[0,2369] + mem[0,2369] = 0
 accuracy_0_1103: - 5.09765625 frac[0,1103] + mem[0,1103] = 0
 accuracy_0_302: - 5.09765625 frac[0,302] + mem[0,302] = 0
 accuracy_0_1395: - 5.09765625 frac[0,1395] + mem[0,1395] = 0
 accuracy_0_184: - 5.09765625 frac[0,184] + mem[0,184] = 0
 accuracy_0_1715: - 5.09765625 frac[0,1715] + mem[0,1715] = 0
 accuracy_0_7: - 5.09765625 frac[0,7] + mem[0,7] = 0
 accuracy_0_1681: - 5.09765625 frac[0,1681] + mem[0,1681] = 0
 accuracy_0_102: - 5.09765625 frac[0,102] + mem[0,102] = 0
 accuracy_0_582: - 5.09765625 frac[0,582] + mem[0,582] = 0
 accuracy_0_1884: - 5.09765625 frac[0,1884] + mem[0,1884] = 0
 accuracy_0_1828: - 5.09765625 frac[0,1828] + mem[0,1828] = 0
 accuracy_0_1923: - 5.09765625 frac[0,1923] + mem[0,1923] = 0
 accuracy_0_1212: - 5.09765625 frac[0,1212] + mem[0,1212] = 0
 accuracy_0_1052: - 5.09765625 frac[0,1052] + mem[0,1052] = 0
 mem_tot_l2_P4: - mem[0,1985] - mem[0,2369] - mem[0,1103] - mem[0,302]
   - mem[0,1395] - mem[0,184] - mem[0,1715] - mem[0,7] - mem[0,1681]
   - mem[0,102] - mem[0,582] - mem[0,1884] - mem[0,1828] - mem[0,1923]
   - mem[0,1212] - mem[0,1052] + mem_tot_l2_P4 = 0
 R81: - 0.00173611111111111 mem_tot_l2_P4 + normalized_mem_tot_l2_P4 = 0
 rows_tot_l2_P4: - 3 frac[0,1985] - 3 frac[0,2369] - 3 frac[0,1103]
   - 3 frac[0,302] - 3 frac[0,1395] - 3 frac[0,184] - 3 frac[0,1715]
   - 3 frac[0,7] - 3 frac[0,1681] - 3 frac[0,102] - 3 frac[0,582]
   - 3 frac[0,1884] - 3 frac[0,1828] - 3 frac[0,1923] - 3 frac[0,1212]
   - 3 frac[0,1052] + rows_tot_l2_P4 = 0
 R83: - 0.0208333333333333 rows_tot_l2_P4 + normalized_rows_tot_l2_P4 = 0
 row_capacity_l2_P4: rows_tot_l2_P4 <= 48
 ns_l2_P4: ns_l2_P4 = 6.756756756756757
 max_ns_l2_P4: ns_l2_P4 <= 6.756756756756757
 res_acc: 0.0208333333333333 mem_tot_l2_P4 + 0.25 rows_tot_l2_P4 - res = 0
Bounds
 mem_tot_l2_P4 <= 576
 normalized_mem_tot_l2_P4 <= 1
 normalized_rows_tot_l2_P4 <= 1
Binaries
 frac[0,1985] frac[0,2369] frac[0,1103] frac[0,302] frac[0,1395]
 frac[0,184] frac[0,1715] frac[0,7] frac[0,1681] frac[0,102] frac[0,582]
 frac[0,1884] frac[0,1828] frac[0,1923] frac[0,1212] frac[0,1052]
General Constraints
 mem_overall_P4: max_mem_P4 = MAX ( mem_tot_l2_P4 )
 rows_overall_P4: max_rows_P4 = MAX ( rows_tot_l2_P4 )
End

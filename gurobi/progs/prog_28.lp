\ Model UnivmonGreedyRows
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  tot_rows_CPU_Cluster: Priority=100 Weight=1 AbsTol=0 RelTol=0
   rows_tot_CPU1681 + rows_tot_CPU1684 + rows_tot_CPU1685 + rows_tot_CPU1686
   + rows_tot_CPU1687 + rows_tot_CPU1688 + rows_tot_CPU1689
   + rows_tot_CPU1690 + rows_tot_CPU1691 + rows_tot_CPU1692
   + rows_tot_CPU1693 + rows_tot_CPU1694 + rows_tot_CPU1695
   + rows_tot_CPU1696 + rows_tot_CPU1698 + rows_tot_CPU1699
   + rows_tot_CPU1700 + rows_tot_CPU1701 + rows_tot_CPU1702
   + rows_tot_CPU1703 + rows_tot_CPU1704 + rows_tot_CPU1705
   + rows_tot_CPU1706 + rows_tot_CPU1708 + rows_tot_CPU1709
   + rows_tot_CPU1710 + rows_tot_CPU1711 + rows_tot_CPU1712
   + rows_tot_CPU1713 + rows_tot_CPU1714 + rows_tot_CPU1715
   + rows_tot_CPU1716 + rows_tot_CPU1717 + rows_tot_CPU1718
   + rows_tot_CPU1719 + rows_tot_CPU1720 + rows_tot_CPU1721
   + rows_tot_CPU1722 + rows_tot_CPU1723 + rows_tot_CPU1724
   + rows_tot_CPU1725 + rows_tot_CPU1727 + rows_tot_CPU1728
  CPU_Cluster_rows_load: Priority=90 Weight=1 AbsTol=0 RelTol=0
   max_rows_CPU_Cluster
  OBJ2: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ3: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ4: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ5: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  tot_mem_CPU_Cluster: Priority=60 Weight=1 AbsTol=0 RelTol=0
   mem_tot_CPU1681 + mem_tot_CPU1684 + mem_tot_CPU1685 + mem_tot_CPU1686
   + mem_tot_CPU1687 + mem_tot_CPU1688 + mem_tot_CPU1689 + mem_tot_CPU1690
   + mem_tot_CPU1691 + mem_tot_CPU1692 + mem_tot_CPU1693 + mem_tot_CPU1694
   + mem_tot_CPU1695 + mem_tot_CPU1696 + mem_tot_CPU1698 + mem_tot_CPU1699
   + mem_tot_CPU1700 + mem_tot_CPU1701 + mem_tot_CPU1702 + mem_tot_CPU1703
   + mem_tot_CPU1704 + mem_tot_CPU1705 + mem_tot_CPU1706 + mem_tot_CPU1708
   + mem_tot_CPU1709 + mem_tot_CPU1710 + mem_tot_CPU1711 + mem_tot_CPU1712
   + mem_tot_CPU1713 + mem_tot_CPU1714 + mem_tot_CPU1715 + mem_tot_CPU1716
   + mem_tot_CPU1717 + mem_tot_CPU1718 + mem_tot_CPU1719 + mem_tot_CPU1720
   + mem_tot_CPU1721 + mem_tot_CPU1722 + mem_tot_CPU1723 + mem_tot_CPU1724
   + mem_tot_CPU1725 + mem_tot_CPU1727 + mem_tot_CPU1728
  CPU_Cluster_load_mem: Priority=50 Weight=1 AbsTol=0 RelTol=0
   max_mem_CPU_Cluster
Subject To
 cov_0_28: frac[14,28] >= 1
 cov_1_28: frac[14,28] >= 1
 cov_2_33: frac[20,33] >= 0.7
 cov_3_7: frac[24,7] >= 0.7
 cov_4_7: frac[24,7] >= 0.7
 cov_5_3: frac[37,3] >= 0.1
 cov_6_35: frac[33,35] >= 0.7
 cov_7_8: frac[23,8] >= 0.7
 cov_8_13: frac[5,13] >= 0.1
 cov_9_34: frac[5,34] >= 0.7
 cov_10_37: frac[32,37] >= 0.8
 cov_11_2: frac[10,2] >= 0.2
 cov_12_31: frac[29,31] >= 0.7
 cov_13_1: frac[29,1] >= 1
 cov_14_25: frac[4,25] >= 0.8
 cov_15_25: frac[4,25] >= 0.8
 cov_16_22: frac[15,22] >= 0.7
 cov_17_12: frac[27,12] >= 0.7
 cov_18_15: frac[35,15] >= 0.8
 cov_19_5: frac[11,5] >= 0.5
 cov_20_21: frac[11,21] >= 1
 cov_21_5: frac[11,5] >= 0.5
 cov_22_5: frac[11,5] >= 0.5
 cov_23_36: frac[1,36] >= 0.2
 cov_24_4: frac[37,4] >= 0.3243655333647049
 cov_25_23: frac[37,23] >= 0.9
 cov_26_23: frac[37,23] >= 0.9
 cov_27_4: frac[37,4] >= 0.3243655333647049
 cov_28_18: frac[8,18] >= 0.7
 cov_29_18: frac[8,18] >= 0.7
 cov_30_10: frac[2,10] >= 0.7
 cov_31_10: frac[2,10] >= 0.7
 cov_32_19: frac[22,19] >= 0.4015793407999981
 cov_33_14: frac[22,14] >= 0.7
 cov_34_27: frac[13,27] >= 0.3
 cov_35_26: frac[13,26] >= 0.9
 cov_36_29: frac[1,29] >= 0.7
 cov_37_29: frac[1,29] >= 1
 cov_38_20: frac[1,20] >= 0.9
 cov_39_11: frac[12,11] >= 0.7
 cov_40_11: frac[41,11] >= 0.7
 cov_41_30: frac[3,30] >= 1
 cov_42_16: frac[36,16] >= 0.4717214065875296
 cov_43_17: frac[36,17] >= 0.7
 cov_44_9: frac[26,9] >= 0.8
 cov_45_9: frac[26,9] >= 0.8
 cov_46_6: frac[9,6] >= 0.4045984936192675
 cov_47_32: frac[9,32] >= 0.8
 cov_48_24: frac[7,24] >= 0.8
 cov_49_24: frac[7,24] >= 1
 cov_50_0: frac[40,0] >= 0.7
 cov_51_38: frac[25,38] >= 0.1
 accuracy_23_8: - 5.09765625 frac[23,8] + mem[23,8] = 0
 accuracy_5_13: - 5.09765625 frac[5,13] + mem[5,13] = 0
 accuracy_13_26: - 5.09765625 frac[13,26] + mem[13,26] = 0
 accuracy_36_17: - 5.09765625 frac[36,17] + mem[36,17] = 0
 accuracy_22_19: - 5.09765625 frac[22,19] + mem[22,19] = 0
 accuracy_32_37: - 5.09765625 frac[32,37] + mem[32,37] = 0
 accuracy_1_29: - 5.09765625 frac[1,29] + mem[1,29] = 0
 accuracy_1_20: - 5.09765625 frac[1,20] + mem[1,20] = 0
 accuracy_37_3: - 5.09765625 frac[37,3] + mem[37,3] = 0
 accuracy_35_15: - 5.09765625 frac[35,15] + mem[35,15] = 0
 accuracy_11_5: - 5.09765625 frac[11,5] + mem[11,5] = 0
 accuracy_29_31: - 5.09765625 frac[29,31] + mem[29,31] = 0
 accuracy_7_24: - 5.09765625 frac[7,24] + mem[7,24] = 0
 accuracy_40_0: - 5.09765625 frac[40,0] + mem[40,0] = 0
 accuracy_1_36: - 5.09765625 frac[1,36] + mem[1,36] = 0
 accuracy_22_14: - 5.09765625 frac[22,14] + mem[22,14] = 0
 accuracy_37_23: - 5.09765625 frac[37,23] + mem[37,23] = 0
 accuracy_14_28: - 5.09765625 frac[14,28] + mem[14,28] = 0
 accuracy_15_22: - 5.09765625 frac[15,22] + mem[15,22] = 0
 accuracy_33_35: - 5.09765625 frac[33,35] + mem[33,35] = 0
 accuracy_9_32: - 5.09765625 frac[9,32] + mem[9,32] = 0
 accuracy_25_38: - 5.09765625 frac[25,38] + mem[25,38] = 0
 accuracy_3_30: - 5.09765625 frac[3,30] + mem[3,30] = 0
 accuracy_2_10: - 5.09765625 frac[2,10] + mem[2,10] = 0
 accuracy_26_9: - 5.09765625 frac[26,9] + mem[26,9] = 0
 accuracy_41_11: - 5.09765625 frac[41,11] + mem[41,11] = 0
 accuracy_4_25: - 5.09765625 frac[4,25] + mem[4,25] = 0
 accuracy_12_11: - 5.09765625 frac[12,11] + mem[12,11] = 0
 accuracy_11_21: - 5.09765625 frac[11,21] + mem[11,21] = 0
 accuracy_5_34: - 5.09765625 frac[5,34] + mem[5,34] = 0
 accuracy_9_6: - 5.09765625 frac[9,6] + mem[9,6] = 0
 accuracy_37_4: - 5.09765625 frac[37,4] + mem[37,4] = 0
 accuracy_24_7: - 5.09765625 frac[24,7] + mem[24,7] = 0
 accuracy_20_33: - 5.09765625 frac[20,33] + mem[20,33] = 0
 accuracy_27_12: - 5.09765625 frac[27,12] + mem[27,12] = 0
 accuracy_13_27: - 5.09765625 frac[13,27] + mem[13,27] = 0
 accuracy_36_16: - 5.09765625 frac[36,16] + mem[36,16] = 0
 accuracy_29_1: - 5.09765625 frac[29,1] + mem[29,1] = 0
 accuracy_8_18: - 5.09765625 frac[8,18] + mem[8,18] = 0
 accuracy_10_2: - 5.09765625 frac[10,2] + mem[10,2] = 0
 mem_tot_CPU1681: mem_tot_CPU1681 = 0
 R93: - 3.05176e-05 mem_tot_CPU1681 + normalized_mem_tot_CPU1681 = 0
 rows_tot_CPU1681: rows_tot_CPU1681 = 0
 R95: - 0.0833333333333333 rows_tot_CPU1681 + normalized_rows_tot_CPU1681
   = 0
 mem_tot_CPU1684: - mem[1,29] - mem[1,20] - mem[1,36] + mem_tot_CPU1684
   = 0
 R97: - 3.05176e-05 mem_tot_CPU1684 + normalized_mem_tot_CPU1684 = 0
 rows_tot_CPU1684: - 3 frac[1,29] - 3 frac[1,20] - 3 frac[1,36]
   + rows_tot_CPU1684 = 0
 R99: - 0.0833333333333333 rows_tot_CPU1684 + normalized_rows_tot_CPU1684
   = 0
 mem_tot_CPU1685: - mem[2,10] + mem_tot_CPU1685 = 0
 R101: - 3.05176e-05 mem_tot_CPU1685 + normalized_mem_tot_CPU1685 = 0
 rows_tot_CPU1685: - 3 frac[2,10] + rows_tot_CPU1685 = 0
 R103: - 0.0833333333333333 rows_tot_CPU1685 + normalized_rows_tot_CPU1685
   = 0
 mem_tot_CPU1686: - mem[3,30] + mem_tot_CPU1686 = 0
 R105: - 3.05176e-05 mem_tot_CPU1686 + normalized_mem_tot_CPU1686 = 0
 rows_tot_CPU1686: - 3 frac[3,30] + rows_tot_CPU1686 = 0
 R107: - 0.0833333333333333 rows_tot_CPU1686 + normalized_rows_tot_CPU1686
   = 0
 mem_tot_CPU1687: - mem[4,25] + mem_tot_CPU1687 = 0
 R109: - 3.05176e-05 mem_tot_CPU1687 + normalized_mem_tot_CPU1687 = 0
 rows_tot_CPU1687: - 3 frac[4,25] + rows_tot_CPU1687 = 0
 R111: - 0.0833333333333333 rows_tot_CPU1687 + normalized_rows_tot_CPU1687
   = 0
 mem_tot_CPU1688: - mem[5,13] - mem[5,34] + mem_tot_CPU1688 = 0
 R113: - 3.05176e-05 mem_tot_CPU1688 + normalized_mem_tot_CPU1688 = 0
 rows_tot_CPU1688: - 3 frac[5,13] - 3 frac[5,34] + rows_tot_CPU1688 = 0
 R115: - 0.0833333333333333 rows_tot_CPU1688 + normalized_rows_tot_CPU1688
   = 0
 mem_tot_CPU1689: mem_tot_CPU1689 = 0
 R117: - 3.05176e-05 mem_tot_CPU1689 + normalized_mem_tot_CPU1689 = 0
 rows_tot_CPU1689: rows_tot_CPU1689 = 0
 R119: - 0.0833333333333333 rows_tot_CPU1689 + normalized_rows_tot_CPU1689
   = 0
 mem_tot_CPU1690: - mem[7,24] + mem_tot_CPU1690 = 0
 R121: - 3.05176e-05 mem_tot_CPU1690 + normalized_mem_tot_CPU1690 = 0
 rows_tot_CPU1690: - 3 frac[7,24] + rows_tot_CPU1690 = 0
 R123: - 0.0833333333333333 rows_tot_CPU1690 + normalized_rows_tot_CPU1690
   = 0
 mem_tot_CPU1691: - mem[8,18] + mem_tot_CPU1691 = 0
 R125: - 3.05176e-05 mem_tot_CPU1691 + normalized_mem_tot_CPU1691 = 0
 rows_tot_CPU1691: - 3 frac[8,18] + rows_tot_CPU1691 = 0
 R127: - 0.0833333333333333 rows_tot_CPU1691 + normalized_rows_tot_CPU1691
   = 0
 mem_tot_CPU1692: - mem[9,32] - mem[9,6] + mem_tot_CPU1692 = 0
 R129: - 3.05176e-05 mem_tot_CPU1692 + normalized_mem_tot_CPU1692 = 0
 rows_tot_CPU1692: - 3 frac[9,32] - 3 frac[9,6] + rows_tot_CPU1692 = 0
 R131: - 0.0833333333333333 rows_tot_CPU1692 + normalized_rows_tot_CPU1692
   = 0
 mem_tot_CPU1693: - mem[10,2] + mem_tot_CPU1693 = 0
 R133: - 3.05176e-05 mem_tot_CPU1693 + normalized_mem_tot_CPU1693 = 0
 rows_tot_CPU1693: - 3 frac[10,2] + rows_tot_CPU1693 = 0
 R135: - 0.0833333333333333 rows_tot_CPU1693 + normalized_rows_tot_CPU1693
   = 0
 mem_tot_CPU1694: - mem[11,5] - mem[11,21] + mem_tot_CPU1694 = 0
 R137: - 3.05176e-05 mem_tot_CPU1694 + normalized_mem_tot_CPU1694 = 0
 rows_tot_CPU1694: - 3 frac[11,5] - 3 frac[11,21] + rows_tot_CPU1694 = 0
 R139: - 0.0833333333333333 rows_tot_CPU1694 + normalized_rows_tot_CPU1694
   = 0
 mem_tot_CPU1695: - mem[12,11] + mem_tot_CPU1695 = 0
 R141: - 3.05176e-05 mem_tot_CPU1695 + normalized_mem_tot_CPU1695 = 0
 rows_tot_CPU1695: - 3 frac[12,11] + rows_tot_CPU1695 = 0
 R143: - 0.0833333333333333 rows_tot_CPU1695 + normalized_rows_tot_CPU1695
   = 0
 mem_tot_CPU1696: - mem[13,26] - mem[13,27] + mem_tot_CPU1696 = 0
 R145: - 3.05176e-05 mem_tot_CPU1696 + normalized_mem_tot_CPU1696 = 0
 rows_tot_CPU1696: - 3 frac[13,26] - 3 frac[13,27] + rows_tot_CPU1696 = 0
 R147: - 0.0833333333333333 rows_tot_CPU1696 + normalized_rows_tot_CPU1696
   = 0
 mem_tot_CPU1698: - mem[14,28] + mem_tot_CPU1698 = 0
 R149: - 3.05176e-05 mem_tot_CPU1698 + normalized_mem_tot_CPU1698 = 0
 rows_tot_CPU1698: - 3 frac[14,28] + rows_tot_CPU1698 = 0
 R151: - 0.0833333333333333 rows_tot_CPU1698 + normalized_rows_tot_CPU1698
   = 0
 mem_tot_CPU1699: - mem[15,22] + mem_tot_CPU1699 = 0
 R153: - 3.05176e-05 mem_tot_CPU1699 + normalized_mem_tot_CPU1699 = 0
 rows_tot_CPU1699: - 3 frac[15,22] + rows_tot_CPU1699 = 0
 R155: - 0.0833333333333333 rows_tot_CPU1699 + normalized_rows_tot_CPU1699
   = 0
 mem_tot_CPU1700: mem_tot_CPU1700 = 0
 R157: - 3.05176e-05 mem_tot_CPU1700 + normalized_mem_tot_CPU1700 = 0
 rows_tot_CPU1700: rows_tot_CPU1700 = 0
 R159: - 0.0833333333333333 rows_tot_CPU1700 + normalized_rows_tot_CPU1700
   = 0
 mem_tot_CPU1701: mem_tot_CPU1701 = 0
 R161: - 3.05176e-05 mem_tot_CPU1701 + normalized_mem_tot_CPU1701 = 0
 rows_tot_CPU1701: rows_tot_CPU1701 = 0
 R163: - 0.0833333333333333 rows_tot_CPU1701 + normalized_rows_tot_CPU1701
   = 0
 mem_tot_CPU1702: mem_tot_CPU1702 = 0
 R165: - 3.05176e-05 mem_tot_CPU1702 + normalized_mem_tot_CPU1702 = 0
 rows_tot_CPU1702: rows_tot_CPU1702 = 0
 R167: - 0.0833333333333333 rows_tot_CPU1702 + normalized_rows_tot_CPU1702
   = 0
 mem_tot_CPU1703: mem_tot_CPU1703 = 0
 R169: - 3.05176e-05 mem_tot_CPU1703 + normalized_mem_tot_CPU1703 = 0
 rows_tot_CPU1703: rows_tot_CPU1703 = 0
 R171: - 0.0833333333333333 rows_tot_CPU1703 + normalized_rows_tot_CPU1703
   = 0
 mem_tot_CPU1704: - mem[20,33] + mem_tot_CPU1704 = 0
 R173: - 3.05176e-05 mem_tot_CPU1704 + normalized_mem_tot_CPU1704 = 0
 rows_tot_CPU1704: - 3 frac[20,33] + rows_tot_CPU1704 = 0
 R175: - 0.0833333333333333 rows_tot_CPU1704 + normalized_rows_tot_CPU1704
   = 0
 mem_tot_CPU1705: mem_tot_CPU1705 = 0
 R177: - 3.05176e-05 mem_tot_CPU1705 + normalized_mem_tot_CPU1705 = 0
 rows_tot_CPU1705: rows_tot_CPU1705 = 0
 R179: - 0.0833333333333333 rows_tot_CPU1705 + normalized_rows_tot_CPU1705
   = 0
 mem_tot_CPU1706: - mem[22,19] - mem[22,14] + mem_tot_CPU1706 = 0
 R181: - 3.05176e-05 mem_tot_CPU1706 + normalized_mem_tot_CPU1706 = 0
 rows_tot_CPU1706: - 3 frac[22,19] - 3 frac[22,14] + rows_tot_CPU1706 = 0
 R183: - 0.0833333333333333 rows_tot_CPU1706 + normalized_rows_tot_CPU1706
   = 0
 mem_tot_CPU1708: - mem[23,8] + mem_tot_CPU1708 = 0
 R185: - 3.05176e-05 mem_tot_CPU1708 + normalized_mem_tot_CPU1708 = 0
 rows_tot_CPU1708: - 3 frac[23,8] + rows_tot_CPU1708 = 0
 R187: - 0.0833333333333333 rows_tot_CPU1708 + normalized_rows_tot_CPU1708
   = 0
 mem_tot_CPU1709: - mem[24,7] + mem_tot_CPU1709 = 0
 R189: - 3.05176e-05 mem_tot_CPU1709 + normalized_mem_tot_CPU1709 = 0
 rows_tot_CPU1709: - 3 frac[24,7] + rows_tot_CPU1709 = 0
 R191: - 0.0833333333333333 rows_tot_CPU1709 + normalized_rows_tot_CPU1709
   = 0
 mem_tot_CPU1710: - mem[25,38] + mem_tot_CPU1710 = 0
 R193: - 3.05176e-05 mem_tot_CPU1710 + normalized_mem_tot_CPU1710 = 0
 rows_tot_CPU1710: - 3 frac[25,38] + rows_tot_CPU1710 = 0
 R195: - 0.0833333333333333 rows_tot_CPU1710 + normalized_rows_tot_CPU1710
   = 0
 mem_tot_CPU1711: - mem[26,9] + mem_tot_CPU1711 = 0
 R197: - 3.05176e-05 mem_tot_CPU1711 + normalized_mem_tot_CPU1711 = 0
 rows_tot_CPU1711: - 3 frac[26,9] + rows_tot_CPU1711 = 0
 R199: - 0.0833333333333333 rows_tot_CPU1711 + normalized_rows_tot_CPU1711
   = 0
 mem_tot_CPU1712: - mem[27,12] + mem_tot_CPU1712 = 0
 R201: - 3.05176e-05 mem_tot_CPU1712 + normalized_mem_tot_CPU1712 = 0
 rows_tot_CPU1712: - 3 frac[27,12] + rows_tot_CPU1712 = 0
 R203: - 0.0833333333333333 rows_tot_CPU1712 + normalized_rows_tot_CPU1712
   = 0
 mem_tot_CPU1713: mem_tot_CPU1713 = 0
 R205: - 3.05176e-05 mem_tot_CPU1713 + normalized_mem_tot_CPU1713 = 0
 rows_tot_CPU1713: rows_tot_CPU1713 = 0
 R207: - 0.0833333333333333 rows_tot_CPU1713 + normalized_rows_tot_CPU1713
   = 0
 mem_tot_CPU1714: - mem[29,31] - mem[29,1] + mem_tot_CPU1714 = 0
 R209: - 3.05176e-05 mem_tot_CPU1714 + normalized_mem_tot_CPU1714 = 0
 rows_tot_CPU1714: - 3 frac[29,31] - 3 frac[29,1] + rows_tot_CPU1714 = 0
 R211: - 0.0833333333333333 rows_tot_CPU1714 + normalized_rows_tot_CPU1714
   = 0
 mem_tot_CPU1715: mem_tot_CPU1715 = 0
 R213: - 3.05176e-05 mem_tot_CPU1715 + normalized_mem_tot_CPU1715 = 0
 rows_tot_CPU1715: rows_tot_CPU1715 = 0
 R215: - 0.0833333333333333 rows_tot_CPU1715 + normalized_rows_tot_CPU1715
   = 0
 mem_tot_CPU1716: mem_tot_CPU1716 = 0
 R217: - 3.05176e-05 mem_tot_CPU1716 + normalized_mem_tot_CPU1716 = 0
 rows_tot_CPU1716: rows_tot_CPU1716 = 0
 R219: - 0.0833333333333333 rows_tot_CPU1716 + normalized_rows_tot_CPU1716
   = 0
 mem_tot_CPU1717: - mem[32,37] + mem_tot_CPU1717 = 0
 R221: - 3.05176e-05 mem_tot_CPU1717 + normalized_mem_tot_CPU1717 = 0
 rows_tot_CPU1717: - 3 frac[32,37] + rows_tot_CPU1717 = 0
 R223: - 0.0833333333333333 rows_tot_CPU1717 + normalized_rows_tot_CPU1717
   = 0
 mem_tot_CPU1718: - mem[33,35] + mem_tot_CPU1718 = 0
 R225: - 3.05176e-05 mem_tot_CPU1718 + normalized_mem_tot_CPU1718 = 0
 rows_tot_CPU1718: - 3 frac[33,35] + rows_tot_CPU1718 = 0
 R227: - 0.0833333333333333 rows_tot_CPU1718 + normalized_rows_tot_CPU1718
   = 0
 mem_tot_CPU1719: mem_tot_CPU1719 = 0
 R229: - 3.05176e-05 mem_tot_CPU1719 + normalized_mem_tot_CPU1719 = 0
 rows_tot_CPU1719: rows_tot_CPU1719 = 0
 R231: - 0.0833333333333333 rows_tot_CPU1719 + normalized_rows_tot_CPU1719
   = 0
 mem_tot_CPU1720: - mem[35,15] + mem_tot_CPU1720 = 0
 R233: - 3.05176e-05 mem_tot_CPU1720 + normalized_mem_tot_CPU1720 = 0
 rows_tot_CPU1720: - 3 frac[35,15] + rows_tot_CPU1720 = 0
 R235: - 0.0833333333333333 rows_tot_CPU1720 + normalized_rows_tot_CPU1720
   = 0
 mem_tot_CPU1721: - mem[36,17] - mem[36,16] + mem_tot_CPU1721 = 0
 R237: - 3.05176e-05 mem_tot_CPU1721 + normalized_mem_tot_CPU1721 = 0
 rows_tot_CPU1721: - 3 frac[36,17] - 3 frac[36,16] + rows_tot_CPU1721 = 0
 R239: - 0.0833333333333333 rows_tot_CPU1721 + normalized_rows_tot_CPU1721
   = 0
 mem_tot_CPU1722: - mem[37,3] - mem[37,23] - mem[37,4] + mem_tot_CPU1722
   = 0
 R241: - 3.05176e-05 mem_tot_CPU1722 + normalized_mem_tot_CPU1722 = 0
 rows_tot_CPU1722: - 3 frac[37,3] - 3 frac[37,23] - 3 frac[37,4]
   + rows_tot_CPU1722 = 0
 R243: - 0.0833333333333333 rows_tot_CPU1722 + normalized_rows_tot_CPU1722
   = 0
 mem_tot_CPU1723: mem_tot_CPU1723 = 0
 R245: - 3.05176e-05 mem_tot_CPU1723 + normalized_mem_tot_CPU1723 = 0
 rows_tot_CPU1723: rows_tot_CPU1723 = 0
 R247: - 0.0833333333333333 rows_tot_CPU1723 + normalized_rows_tot_CPU1723
   = 0
 mem_tot_CPU1724: mem_tot_CPU1724 = 0
 R249: - 3.05176e-05 mem_tot_CPU1724 + normalized_mem_tot_CPU1724 = 0
 rows_tot_CPU1724: rows_tot_CPU1724 = 0
 R251: - 0.0833333333333333 rows_tot_CPU1724 + normalized_rows_tot_CPU1724
   = 0
 mem_tot_CPU1725: - mem[40,0] + mem_tot_CPU1725 = 0
 R253: - 3.05176e-05 mem_tot_CPU1725 + normalized_mem_tot_CPU1725 = 0
 rows_tot_CPU1725: - 3 frac[40,0] + rows_tot_CPU1725 = 0
 R255: - 0.0833333333333333 rows_tot_CPU1725 + normalized_rows_tot_CPU1725
   = 0
 mem_tot_CPU1727: - mem[41,11] + mem_tot_CPU1727 = 0
 R257: - 3.05176e-05 mem_tot_CPU1727 + normalized_mem_tot_CPU1727 = 0
 rows_tot_CPU1727: - 3 frac[41,11] + rows_tot_CPU1727 = 0
 R259: - 0.0833333333333333 rows_tot_CPU1727 + normalized_rows_tot_CPU1727
   = 0
 mem_tot_CPU1728: mem_tot_CPU1728 = 0
 R261: - 3.05176e-05 mem_tot_CPU1728 + normalized_mem_tot_CPU1728 = 0
 rows_tot_CPU1728: rows_tot_CPU1728 = 0
 R263: - 0.0833333333333333 rows_tot_CPU1728 + normalized_rows_tot_CPU1728
   = 0
 row_capacity_CPU1681: rows_tot_CPU1681 <= 12
 row_capacity_CPU1684: rows_tot_CPU1684 <= 12
 row_capacity_CPU1685: rows_tot_CPU1685 <= 12
 row_capacity_CPU1686: rows_tot_CPU1686 <= 12
 row_capacity_CPU1687: rows_tot_CPU1687 <= 12
 row_capacity_CPU1688: rows_tot_CPU1688 <= 12
 row_capacity_CPU1689: rows_tot_CPU1689 <= 12
 row_capacity_CPU1690: rows_tot_CPU1690 <= 12
 row_capacity_CPU1691: rows_tot_CPU1691 <= 12
 row_capacity_CPU1692: rows_tot_CPU1692 <= 12
 row_capacity_CPU1693: rows_tot_CPU1693 <= 12
 row_capacity_CPU1694: rows_tot_CPU1694 <= 12
 row_capacity_CPU1695: rows_tot_CPU1695 <= 12
 row_capacity_CPU1696: rows_tot_CPU1696 <= 12
 row_capacity_CPU1698: rows_tot_CPU1698 <= 12
 row_capacity_CPU1699: rows_tot_CPU1699 <= 12
 row_capacity_CPU1700: rows_tot_CPU1700 <= 12
 row_capacity_CPU1701: rows_tot_CPU1701 <= 12
 row_capacity_CPU1702: rows_tot_CPU1702 <= 12
 row_capacity_CPU1703: rows_tot_CPU1703 <= 12
 row_capacity_CPU1704: rows_tot_CPU1704 <= 12
 row_capacity_CPU1705: rows_tot_CPU1705 <= 12
 row_capacity_CPU1706: rows_tot_CPU1706 <= 12
 row_capacity_CPU1708: rows_tot_CPU1708 <= 12
 row_capacity_CPU1709: rows_tot_CPU1709 <= 12
 row_capacity_CPU1710: rows_tot_CPU1710 <= 12
 row_capacity_CPU1711: rows_tot_CPU1711 <= 12
 row_capacity_CPU1712: rows_tot_CPU1712 <= 12
 row_capacity_CPU1713: rows_tot_CPU1713 <= 12
 row_capacity_CPU1714: rows_tot_CPU1714 <= 12
 row_capacity_CPU1715: rows_tot_CPU1715 <= 12
 row_capacity_CPU1716: rows_tot_CPU1716 <= 12
 row_capacity_CPU1717: rows_tot_CPU1717 <= 12
 row_capacity_CPU1718: rows_tot_CPU1718 <= 12
 row_capacity_CPU1719: rows_tot_CPU1719 <= 12
 row_capacity_CPU1720: rows_tot_CPU1720 <= 12
 row_capacity_CPU1721: rows_tot_CPU1721 <= 12
 row_capacity_CPU1722: rows_tot_CPU1722 <= 12
 row_capacity_CPU1723: rows_tot_CPU1723 <= 12
 row_capacity_CPU1724: rows_tot_CPU1724 <= 12
 row_capacity_CPU1725: rows_tot_CPU1725 <= 12
 row_capacity_CPU1727: rows_tot_CPU1727 <= 12
 row_capacity_CPU1728: rows_tot_CPU1728 <= 12
Bounds
 frac[23,8] <= 1
 frac[5,13] <= 1
 frac[13,26] <= 1
 frac[36,17] <= 1
 frac[22,19] <= 1
 frac[32,37] <= 1
 frac[1,29] <= 1
 frac[1,20] <= 1
 frac[37,3] <= 1
 frac[35,15] <= 1
 frac[11,5] <= 1
 frac[29,31] <= 1
 frac[7,24] <= 1
 frac[40,0] <= 1
 frac[1,36] <= 1
 frac[22,14] <= 1
 frac[37,23] <= 1
 frac[14,28] <= 1
 frac[15,22] <= 1
 frac[33,35] <= 1
 frac[9,32] <= 1
 frac[25,38] <= 1
 frac[3,30] <= 1
 frac[2,10] <= 1
 frac[26,9] <= 1
 frac[41,11] <= 1
 frac[4,25] <= 1
 frac[12,11] <= 1
 frac[11,21] <= 1
 frac[5,34] <= 1
 frac[9,6] <= 1
 frac[37,4] <= 1
 frac[24,7] <= 1
 frac[20,33] <= 1
 frac[27,12] <= 1
 frac[13,27] <= 1
 frac[36,16] <= 1
 frac[29,1] <= 1
 frac[8,18] <= 1
 frac[10,2] <= 1
 mem_tot_CPU1681 <= 32768
 normalized_mem_tot_CPU1681 <= 1
 normalized_rows_tot_CPU1681 <= 1
 mem_tot_CPU1684 <= 32768
 normalized_mem_tot_CPU1684 <= 1
 normalized_rows_tot_CPU1684 <= 1
 mem_tot_CPU1685 <= 32768
 normalized_mem_tot_CPU1685 <= 1
 normalized_rows_tot_CPU1685 <= 1
 mem_tot_CPU1686 <= 32768
 normalized_mem_tot_CPU1686 <= 1
 normalized_rows_tot_CPU1686 <= 1
 mem_tot_CPU1687 <= 32768
 normalized_mem_tot_CPU1687 <= 1
 normalized_rows_tot_CPU1687 <= 1
 mem_tot_CPU1688 <= 32768
 normalized_mem_tot_CPU1688 <= 1
 normalized_rows_tot_CPU1688 <= 1
 mem_tot_CPU1689 <= 32768
 normalized_mem_tot_CPU1689 <= 1
 normalized_rows_tot_CPU1689 <= 1
 mem_tot_CPU1690 <= 32768
 normalized_mem_tot_CPU1690 <= 1
 normalized_rows_tot_CPU1690 <= 1
 mem_tot_CPU1691 <= 32768
 normalized_mem_tot_CPU1691 <= 1
 normalized_rows_tot_CPU1691 <= 1
 mem_tot_CPU1692 <= 32768
 normalized_mem_tot_CPU1692 <= 1
 normalized_rows_tot_CPU1692 <= 1
 mem_tot_CPU1693 <= 32768
 normalized_mem_tot_CPU1693 <= 1
 normalized_rows_tot_CPU1693 <= 1
 mem_tot_CPU1694 <= 32768
 normalized_mem_tot_CPU1694 <= 1
 normalized_rows_tot_CPU1694 <= 1
 mem_tot_CPU1695 <= 32768
 normalized_mem_tot_CPU1695 <= 1
 normalized_rows_tot_CPU1695 <= 1
 mem_tot_CPU1696 <= 32768
 normalized_mem_tot_CPU1696 <= 1
 normalized_rows_tot_CPU1696 <= 1
 mem_tot_CPU1698 <= 32768
 normalized_mem_tot_CPU1698 <= 1
 normalized_rows_tot_CPU1698 <= 1
 mem_tot_CPU1699 <= 32768
 normalized_mem_tot_CPU1699 <= 1
 normalized_rows_tot_CPU1699 <= 1
 mem_tot_CPU1700 <= 32768
 normalized_mem_tot_CPU1700 <= 1
 normalized_rows_tot_CPU1700 <= 1
 mem_tot_CPU1701 <= 32768
 normalized_mem_tot_CPU1701 <= 1
 normalized_rows_tot_CPU1701 <= 1
 mem_tot_CPU1702 <= 32768
 normalized_mem_tot_CPU1702 <= 1
 normalized_rows_tot_CPU1702 <= 1
 mem_tot_CPU1703 <= 32768
 normalized_mem_tot_CPU1703 <= 1
 normalized_rows_tot_CPU1703 <= 1
 mem_tot_CPU1704 <= 32768
 normalized_mem_tot_CPU1704 <= 1
 normalized_rows_tot_CPU1704 <= 1
 mem_tot_CPU1705 <= 32768
 normalized_mem_tot_CPU1705 <= 1
 normalized_rows_tot_CPU1705 <= 1
 mem_tot_CPU1706 <= 32768
 normalized_mem_tot_CPU1706 <= 1
 normalized_rows_tot_CPU1706 <= 1
 mem_tot_CPU1708 <= 32768
 normalized_mem_tot_CPU1708 <= 1
 normalized_rows_tot_CPU1708 <= 1
 mem_tot_CPU1709 <= 32768
 normalized_mem_tot_CPU1709 <= 1
 normalized_rows_tot_CPU1709 <= 1
 mem_tot_CPU1710 <= 32768
 normalized_mem_tot_CPU1710 <= 1
 normalized_rows_tot_CPU1710 <= 1
 mem_tot_CPU1711 <= 32768
 normalized_mem_tot_CPU1711 <= 1
 normalized_rows_tot_CPU1711 <= 1
 mem_tot_CPU1712 <= 32768
 normalized_mem_tot_CPU1712 <= 1
 normalized_rows_tot_CPU1712 <= 1
 mem_tot_CPU1713 <= 32768
 normalized_mem_tot_CPU1713 <= 1
 normalized_rows_tot_CPU1713 <= 1
 mem_tot_CPU1714 <= 32768
 normalized_mem_tot_CPU1714 <= 1
 normalized_rows_tot_CPU1714 <= 1
 mem_tot_CPU1715 <= 32768
 normalized_mem_tot_CPU1715 <= 1
 normalized_rows_tot_CPU1715 <= 1
 mem_tot_CPU1716 <= 32768
 normalized_mem_tot_CPU1716 <= 1
 normalized_rows_tot_CPU1716 <= 1
 mem_tot_CPU1717 <= 32768
 normalized_mem_tot_CPU1717 <= 1
 normalized_rows_tot_CPU1717 <= 1
 mem_tot_CPU1718 <= 32768
 normalized_mem_tot_CPU1718 <= 1
 normalized_rows_tot_CPU1718 <= 1
 mem_tot_CPU1719 <= 32768
 normalized_mem_tot_CPU1719 <= 1
 normalized_rows_tot_CPU1719 <= 1
 mem_tot_CPU1720 <= 32768
 normalized_mem_tot_CPU1720 <= 1
 normalized_rows_tot_CPU1720 <= 1
 mem_tot_CPU1721 <= 32768
 normalized_mem_tot_CPU1721 <= 1
 normalized_rows_tot_CPU1721 <= 1
 mem_tot_CPU1722 <= 32768
 normalized_mem_tot_CPU1722 <= 1
 normalized_rows_tot_CPU1722 <= 1
 mem_tot_CPU1723 <= 32768
 normalized_mem_tot_CPU1723 <= 1
 normalized_rows_tot_CPU1723 <= 1
 mem_tot_CPU1724 <= 32768
 normalized_mem_tot_CPU1724 <= 1
 normalized_rows_tot_CPU1724 <= 1
 mem_tot_CPU1725 <= 32768
 normalized_mem_tot_CPU1725 <= 1
 normalized_rows_tot_CPU1725 <= 1
 mem_tot_CPU1727 <= 32768
 normalized_mem_tot_CPU1727 <= 1
 normalized_rows_tot_CPU1727 <= 1
 mem_tot_CPU1728 <= 32768
 normalized_mem_tot_CPU1728 <= 1
 normalized_rows_tot_CPU1728 <= 1
General Constraints
 mem_overall_CPU_Cluster: max_mem_CPU_Cluster = MAX (
   normalized_mem_tot_CPU1681 , normalized_mem_tot_CPU1684 ,
   normalized_mem_tot_CPU1685 , normalized_mem_tot_CPU1686 ,
   normalized_mem_tot_CPU1687 , normalized_mem_tot_CPU1688 ,
   normalized_mem_tot_CPU1689 , normalized_mem_tot_CPU1690 ,
   normalized_mem_tot_CPU1691 , normalized_mem_tot_CPU1692 ,
   normalized_mem_tot_CPU1693 , normalized_mem_tot_CPU1694 ,
   normalized_mem_tot_CPU1695 , normalized_mem_tot_CPU1696 ,
   normalized_mem_tot_CPU1698 , normalized_mem_tot_CPU1699 ,
   normalized_mem_tot_CPU1700 , normalized_mem_tot_CPU1701 ,
   normalized_mem_tot_CPU1702 , normalized_mem_tot_CPU1703 ,
   normalized_mem_tot_CPU1704 , normalized_mem_tot_CPU1705 ,
   normalized_mem_tot_CPU1706 , normalized_mem_tot_CPU1708 ,
   normalized_mem_tot_CPU1709 , normalized_mem_tot_CPU1710 ,
   normalized_mem_tot_CPU1711 , normalized_mem_tot_CPU1712 ,
   normalized_mem_tot_CPU1713 , normalized_mem_tot_CPU1714 ,
   normalized_mem_tot_CPU1715 , normalized_mem_tot_CPU1716 ,
   normalized_mem_tot_CPU1717 , normalized_mem_tot_CPU1718 ,
   normalized_mem_tot_CPU1719 , normalized_mem_tot_CPU1720 ,
   normalized_mem_tot_CPU1721 , normalized_mem_tot_CPU1722 ,
   normalized_mem_tot_CPU1723 , normalized_mem_tot_CPU1724 ,
   normalized_mem_tot_CPU1725 , normalized_mem_tot_CPU1727 ,
   normalized_mem_tot_CPU1728 )
 rows_overall_CPU: max_rows_CPU_Cluster = MAX (
   normalized_rows_tot_CPU1681 , normalized_rows_tot_CPU1684 ,
   normalized_rows_tot_CPU1685 , normalized_rows_tot_CPU1686 ,
   normalized_rows_tot_CPU1687 , normalized_rows_tot_CPU1688 ,
   normalized_rows_tot_CPU1689 , normalized_rows_tot_CPU1690 ,
   normalized_rows_tot_CPU1691 , normalized_rows_tot_CPU1692 ,
   normalized_rows_tot_CPU1693 , normalized_rows_tot_CPU1694 ,
   normalized_rows_tot_CPU1695 , normalized_rows_tot_CPU1696 ,
   normalized_rows_tot_CPU1698 , normalized_rows_tot_CPU1699 ,
   normalized_rows_tot_CPU1700 , normalized_rows_tot_CPU1701 ,
   normalized_rows_tot_CPU1702 , normalized_rows_tot_CPU1703 ,
   normalized_rows_tot_CPU1704 , normalized_rows_tot_CPU1705 ,
   normalized_rows_tot_CPU1706 , normalized_rows_tot_CPU1708 ,
   normalized_rows_tot_CPU1709 , normalized_rows_tot_CPU1710 ,
   normalized_rows_tot_CPU1711 , normalized_rows_tot_CPU1712 ,
   normalized_rows_tot_CPU1713 , normalized_rows_tot_CPU1714 ,
   normalized_rows_tot_CPU1715 , normalized_rows_tot_CPU1716 ,
   normalized_rows_tot_CPU1717 , normalized_rows_tot_CPU1718 ,
   normalized_rows_tot_CPU1719 , normalized_rows_tot_CPU1720 ,
   normalized_rows_tot_CPU1721 , normalized_rows_tot_CPU1722 ,
   normalized_rows_tot_CPU1723 , normalized_rows_tot_CPU1724 ,
   normalized_rows_tot_CPU1725 , normalized_rows_tot_CPU1727 ,
   normalized_rows_tot_CPU1728 )
End

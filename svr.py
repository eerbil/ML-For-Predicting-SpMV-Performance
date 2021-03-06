import numpy as np
from sklearn import svm
import random

np.set_printoptions(threshold='nan')

csr_data = open("timesCSR", 'r')
ell_data = open("timesELL", 'r')
coo_data = open("timesCOO", 'r')
feature_data = open("features", 'r')
output = open("svr-scoring.txt", 'w')

csr_times = []
coo_times = []
ell_times = []
ell_times_final = []
n = []
nnz = []
dis = []
mu = []
sd = []
nmax = []
ell_n = []
ell_nnz = []
ell_dis = []
ell_nmax = []
names = []
n_count = 0

# Data Parsing
for line in feature_data:
    features = line.rstrip().split(' ')
    names.append(features[0] + " - " + features[1])
    n.append(float(features[3]))
    nnz.append(float(features[5]))
    dis.append(float(features[7]))
    mu.append(float(features[9]))
    sd.append(float(features[11]))
    nmax.append(float(features[13]))

for line in csr_data:
    time = line.rstrip().split(' ')
    csr_times.append(float(time[2]))
for line in coo_data:
    time = line.rstrip().split(' ')
    coo_times.append(float(time[2]))
for line in ell_data:
    time = line.rstrip().split(' ')
    if time[2] == 'N/A':
        ell_times.append("N/A")
    else:
        #print names[n_count]
        ell_times.append(float(time[2]))
    n_count += 1
for i in range(0, len(ell_times)):
    if ell_times[i] != 'N/A':
        ell_times_final.append(ell_times[i])
        ell_n.append(n[i])
        ell_nnz.append(nnz[i])
        ell_dis.append(dis[i])
        ell_nmax.append(nmax[i])

ell_X = []
coo_X = []
csr_X = []

for i in range(0, len(n)):
    coo_X.append([n[i], nnz[i], dis[i]])
    csr_X.append([n[i], nnz[i], dis[i], mu[i], sd[i]])
for i in range(0, len(ell_n)):
    ell_X.append([ell_n[i], ell_nnz[i], ell_dis[i], ell_nmax[i]])

ell_X_shuffled = []
coo_X_shuffled = []
csr_X_shuffled = []

"""
#randomizing the list indexes in every run
shuffleList = list(range(len(n)))
shuffleListELL = list(range(len(ell_n)))
random.shuffle(shuffleList)
random.shuffle(shuffleListELL)
print shuffleList
print shuffleListELL
"""

"""
Using pre-generated random lists with a fixed size for the large dataset (1878 for CSR/COO and 1077 for ELL)
"""
shuffleList = [1001, 1505, 32, 1320, 1052, 1829, 751, 1814, 895, 1079, 1620, 1363, 511, 14, 551, 1706, 1309, 891, 689, 658, 61, 1414, 1761, 1691, 542, 11, 1413, 1353, 1074, 793, 1016, 742, 1811, 1259, 728, 543, 62, 1757, 594, 1635, 1229, 17, 1219, 951, 223, 938, 450, 1128, 1086, 1237, 1739, 1700, 693, 1760, 327, 692, 1741, 676, 933, 1670, 439, 615, 342, 1558, 214, 617, 638, 0, 1084, 1154, 1601, 1217, 1092, 1487, 949, 1271, 952, 1456, 221, 1255, 1152, 1332, 1778, 700, 758, 1554, 1377, 1046, 1101, 207, 1181, 260, 1718, 1513, 1218, 1038, 1789, 345, 824, 1222, 1269, 1869, 105, 743, 1746, 1582, 1726, 1010, 1183, 1649, 299, 114, 166, 1689, 227, 593, 1551, 1771, 867, 910, 118, 1560, 570, 1584, 286, 403, 900, 1040, 1834, 1732, 1701, 306, 1442, 1752, 1385, 1728, 1824, 1573, 706, 581, 1403, 123, 1277, 1445, 197, 1245, 271, 446, 1088, 285, 1268, 1795, 1800, 1659, 1061, 948, 994, 1018, 443, 1300, 1274, 1769, 1114, 1187, 63, 516, 1246, 1482, 1669, 1768, 1581, 1136, 390, 438, 1311, 1051, 1796, 1600, 1603, 1017, 96, 1597, 621, 66, 1775, 1661, 1401, 1547, 479, 1532, 719, 1090, 1070, 898, 1236, 622, 1478, 1387, 244, 1587, 1790, 1643, 1443, 379, 175, 289, 39, 847, 704, 1112, 826, 386, 514, 1164, 523, 319, 708, 229, 889, 122, 237, 44, 1032, 374, 481, 1324, 1251, 908, 99, 1833, 661, 515, 1561, 10, 349, 1830, 655, 1145, 235, 152, 1665, 292, 765, 1400, 144, 1753, 274, 129, 596, 710, 23, 1386, 1852, 370, 180, 405, 1050, 1679, 425, 944, 819, 18, 586, 999, 1598, 779, 804, 901, 1815, 38, 37, 1780, 1382, 1042, 1559, 876, 230, 1149, 1278, 1774, 1798, 1529, 505, 111, 525, 703, 998, 138, 1867, 88, 83, 1533, 311, 1819, 239, 461, 959, 518, 1341, 634, 482, 1764, 134, 711, 1632, 536, 985, 1541, 1521, 1047, 792, 280, 1725, 1123, 149, 1642, 133, 588, 1480, 35, 695, 269, 106, 1364, 404, 1158, 337, 1692, 564, 1490, 1028, 688, 1224, 1433, 1297, 1134, 580, 132, 1809, 24, 797, 806, 493, 1106, 960, 800, 234, 1397, 885, 636, 353, 67, 187, 1853, 654, 836, 43, 955, 1696, 1213, 828, 1291, 772, 233, 1182, 1165, 569, 1411, 211, 734, 153, 1022, 1143, 568, 931, 113, 786, 549, 1783, 749, 1693, 1171, 1384, 309, 1281, 954, 669, 1069, 612, 359, 744, 47, 694, 491, 4, 1489, 414, 1875, 1402, 350, 809, 1849, 599, 1389, 1312, 1080, 1029, 1221, 1355, 73, 251, 1030, 1315, 858, 423, 1495, 1365, 618, 578, 406, 1748, 458, 943, 808, 1067, 304, 1745, 739, 855, 1099, 1660, 218, 795, 1605, 509, 266, 1609, 628, 1788, 1452, 650, 307, 1045, 1319, 20, 890, 437, 1707, 723, 1370, 1568, 539, 466, 116, 53, 1133, 1674, 1845, 672, 1610, 1719, 1498, 1864, 557, 1754, 282, 796, 1585, 712, 281, 1514, 613, 822, 892, 1816, 777, 566, 1801, 997, 1477, 1110, 646, 1422, 915, 610, 821, 158, 1740, 1637, 163, 1240, 291, 435, 1346, 845, 1410, 1369, 630, 1705, 632, 1415, 81, 49, 1538, 1492, 1014, 421, 1736, 1094, 1405, 1207, 917, 1323, 919, 988, 886, 769, 1356, 619, 431, 457, 607, 97, 1267, 1536, 1098, 1262, 788, 399, 1390, 1871, 1457, 965, 489, 203, 1429, 964, 571, 1373, 220, 1117, 1537, 135, 255, 953, 531, 265, 1141, 1797, 1808, 448, 1596, 1651, 848, 380, 643, 1203, 1296, 76, 811, 771, 721, 647, 1455, 1630, 1258, 1130, 1057, 1148, 1459, 78, 191, 1727, 1119, 155, 902, 1470, 480, 1520, 1111, 716, 1623, 128, 1488, 1118, 278, 544, 186, 1758, 652, 970, 436, 1722, 1326, 1865, 1002, 1806, 1449, 169, 320, 1006, 136, 837, 1804, 283, 585, 888, 1290, 12, 396, 903, 977, 1, 1081, 1024, 86, 905, 616, 929, 782, 1628, 1510, 441, 87, 1747, 857, 1840, 851, 1508, 1842, 1751, 417, 322, 1851, 108, 832, 340, 705, 464, 194, 791, 1238, 1713, 1055, 587, 1321, 1534, 1215, 487, 146, 1330, 666, 303, 1548, 1282, 1036, 1256, 923, 91, 1095, 1785, 1854, 1644, 393, 745, 1197, 407, 21, 963, 686, 904, 305, 332, 1167, 42, 465, 1209, 1858, 1629, 25, 608, 1564, 683, 1172, 13, 213, 196, 550, 1409, 382, 1432, 89, 1059, 1688, 1464, 1777, 584, 1417, 691, 473, 913, 980, 1542, 1699, 863, 1035, 1108, 469, 701, 1196, 1473, 1710, 754, 979, 1325, 198, 277, 560, 513, 1779, 208, 1334, 750, 1392, 524, 798, 408, 506, 1773, 853, 275, 56, 1166, 336, 1011, 1653, 921, 373, 1714, 726, 829, 1695, 1434, 818, 897, 1416, 1248, 145, 318, 1178, 1862, 510, 1347, 394, 1678, 1212, 926, 932, 737, 927, 674, 262, 1447, 190, 1358, 117, 884, 1847, 1589, 7, 1102, 397, 478, 1302, 770, 378, 276, 1743, 962, 294, 1636, 1619, 1500, 193, 1244, 633, 288, 668, 1562, 1087, 1317, 1292, 987, 361, 582, 1639, 80, 29, 1314, 33, 3, 324, 1523, 195, 48, 1435, 852, 1793, 664, 667, 916, 1504, 637, 110, 785, 1261, 1625, 992, 1265, 725, 881, 1595, 184, 1125, 1493, 1593, 872, 1228, 781, 715, 682, 801, 1372, 357, 1590, 849, 248, 296, 1199, 51, 496, 912, 1613, 364, 1179, 1575, 1526, 1486, 1518, 1683, 455, 946, 991, 732, 854, 760, 176, 358, 1645, 1602, 893, 1072, 1249, 583, 1828, 1025, 415, 36, 28, 1704, 1367, 179, 1049, 1116, 840, 355, 538, 746, 164, 1813, 825, 1451, 565, 200, 68, 50, 1448, 671, 344, 1638, 1425, 1615, 1190, 930, 659, 556, 1129, 1147, 1614, 1408, 253, 1306, 1193, 160, 1506, 1652, 1454, 453, 1557, 1223, 1654, 1362, 1186, 862, 420, 657, 774, 1646, 767, 799, 1004, 649, 577, 856, 803, 1294, 1592, 419, 1007, 1525, 945, 1666, 308, 1075, 1043, 1439, 199, 1735, 284, 1109, 1169, 714, 1511, 1344, 34, 368, 77, 1767, 529, 1121, 463, 171, 1285, 1137, 391, 866, 1093, 940, 1723, 1163, 1208, 802, 937, 1065, 1388, 1307, 1647, 986, 1676, 733, 323, 820, 328, 1606, 1192, 1708, 1056, 1698, 1383, 90, 1856, 879, 1189, 877, 909, 70, 1484, 172, 713, 440, 1064, 702, 376, 1428, 1832, 545, 27, 752, 1287, 747, 1437, 483, 842, 258, 1588, 64, 354, 15, 348, 1012, 1634, 512, 1034, 841, 413, 527, 1227, 400, 428, 1009, 838, 477, 1225, 451, 126, 242, 130, 267, 1671, 45, 805, 1441, 109, 665, 41, 640, 238, 454, 1162, 1662, 168, 1873, 1711, 459, 660, 351, 1142, 102, 371, 385, 918, 1333, 1396, 209, 653, 94, 753, 395, 896, 812, 383, 1839, 475, 1023, 369, 1860, 1096, 1465, 312, 532, 757, 31, 331, 1682, 1140, 1501, 161, 456, 1468, 1031, 120, 727, 1687, 1857, 729, 1308, 521, 1279, 205, 498, 490, 830, 1342, 503, 148, 592, 1843, 1336, 1818, 1436, 356, 1563, 789, 1576, 741, 684, 1657, 1471, 1690, 778, 1063, 648, 831, 1283, 95, 1822, 1073, 1103, 911, 216, 1680, 600, 250, 1044, 1608, 338, 347, 775, 968, 1026, 231, 1083, 384, 598, 300, 433, 1567, 1361, 947, 1876, 1821, 1376, 768, 1153, 1348, 1295, 1293, 1607, 1194, 1235, 1742, 1289, 766, 1835, 969, 983, 310, 315, 1380, 546, 1021, 1343, 1206, 1502, 541, 1185, 787, 232, 859, 1054, 5, 572, 814, 316, 1155, 748, 137, 1263, 1272, 1782, 1499, 1062, 127, 1509, 154, 1453, 1460, 1350, 1475, 177, 590, 270, 1431, 762, 486, 219, 445, 528, 1731, 335, 878, 1053, 352, 54, 426, 442, 555, 833, 1071, 69, 377, 1066, 143, 471, 522, 1519, 1717, 576, 1368, 834, 1378, 1528, 279, 1233, 1877, 907, 1231, 696, 601, 1627, 165, 816, 1280, 1345, 1160, 263, 1472, 341, 1375, 1621, 604, 1450, 873, 614, 730, 1015, 1570, 860, 1729, 467, 606, 298, 1331, 1802, 1027, 526, 1763, 449, 925, 1874, 724, 1512, 333, 476, 1132, 226, 1491, 1252, 343, 1626, 1507, 157, 188, 173, 1157, 1622, 1446, 1299, 547, 1656, 966, 958, 1304, 807, 468, 1068, 1611, 1337, 167, 1257, 662, 1359, 984, 58, 1762, 125, 1266, 1349, 141, 178, 783, 1462, 1827, 1786, 1583, 561, 589, 330, 709, 1838, 989, 1517, 1539, 1104, 663, 763, 1420, 254, 79, 1107, 575, 920, 835, 670, 924, 1284, 936, 360, 1702, 488, 1318, 1138, 1077, 961, 935, 1161, 346, 1033, 563, 422, 363, 645, 602, 1273, 1799, 147, 1565, 325, 1037, 264, 241, 784, 1580, 1792, 1076, 844, 430, 1122, 956, 1339, 1013, 874, 156, 1550, 1379, 1264, 409, 272, 1275, 1131, 273, 871, 517, 790, 1220, 74, 861, 381, 722, 899, 1497, 978, 597, 1697, 256, 1335, 1173, 1552, 1756, 558, 827, 112, 1168, 1694, 1085, 362, 401, 718, 1681, 975, 1716, 934, 317, 1230, 501, 717, 1543, 623, 72, 1322, 794, 1872, 678, 1310, 554, 1374, 46, 22, 313, 690, 1371, 1115, 1170, 1737, 329, 1750, 192, 185, 1399, 246, 1631, 759, 1260, 573, 533, 976, 82, 1810, 967, 367, 1113, 301, 92, 297, 537, 1640, 392, 429, 334, 1407, 1250, 295, 1686, 1825, 865, 1127, 1316, 974, 1174, 882, 1418, 720, 418, 1177, 1720, 243, 1254, 1146, 1826, 103, 773, 1791, 520, 492, 1078, 1156, 1427, 1759, 941, 1765, 1469, 1641, 9, 1772, 1616, 1360, 1205, 1667, 502, 1276, 1817, 639, 60, 1574, 1776, 764, 1837, 1545, 1770, 982, 626, 1288, 1424, 1467, 1733, 755, 535, 656, 224, 698, 736, 107, 460, 1803, 508, 1351, 1787, 870, 1848, 1859, 1135, 567, 939, 98, 1393, 864, 548, 1496, 1391, 212, 410, 201, 642, 1126, 119, 474, 519, 1247, 624, 240, 552, 591, 1444, 1730, 1211, 1572, 1286, 894, 1159, 1516, 75, 780, 1494, 1846, 1232, 1624, 813, 1241, 1329, 1812, 679, 1463, 1684, 605, 411, 1120, 993, 1458, 1530, 631, 1204, 611, 553, 494, 1301, 880, 427, 1313, 249, 1544, 1476, 321, 1020, 559, 1805, 162, 1270, 1784, 1594, 452, 1870, 1019, 1466, 1082, 869, 1555, 1836, 868, 1485, 85, 1612, 1766, 1556, 776, 398, 236, 1685, 1305, 906, 1535, 731, 1398, 170, 1201, 470, 259, 462, 1604, 412, 1048, 1866, 293, 1850, 1327, 1633, 1586, 204, 1243, 1781, 680, 30, 1578, 1357, 1672, 687, 579, 1577, 183, 1242, 1549, 206, 71, 182, 609, 115, 1579, 1664, 1673, 1531, 973, 1216, 1340, 1650, 1150, 150, 1404, 1599, 495, 181, 603, 2, 366, 507, 1461, 1868, 1426, 1483, 434, 1214, 850, 1703, 815, 562, 202, 625, 1003, 635, 883, 707, 1712, 540, 1200, 8, 1527, 124, 620, 738, 1239, 740, 677, 1366, 499, 1151, 1440, 1522, 387, 644, 1005, 228, 1198, 52, 1430, 247, 1709, 1381, 16, 497, 1419, 1184, 1617, 174, 1807, 1648, 447, 472, 1569, 1438, 595, 6, 673, 875, 1841, 699, 627, 1540, 887, 1008, 139, 1823, 1677, 142, 1479, 675, 388, 1566, 823, 1328, 104, 1546, 365, 1097, 101, 1234, 756, 1668, 1744, 485, 424, 339, 1395, 950, 1658, 971, 225, 1820, 942, 1663, 1794, 55, 1303, 641, 1591, 1180, 651, 222, 40, 1755, 761, 1191, 1124, 1618, 1354, 1749, 1039, 215, 972, 504, 1831, 375, 1202, 1298, 444, 1144, 1421, 131, 261, 1058, 416, 981, 1675, 159, 1715, 59, 140, 1423, 574, 1089, 302, 1524, 817, 697, 846, 93, 1844, 1863, 1412, 1394, 1553, 1855, 681, 19, 990, 1338, 1571, 1210, 735, 1861, 1352, 1175, 843, 928, 1724, 432, 1060, 629, 1041, 121, 810, 1188, 957, 996, 290, 1738, 1406, 839, 402, 1176, 484, 1195, 1474, 1515, 1000, 252, 189, 268, 84, 1105, 257, 1734, 1721, 65, 685, 151, 1100, 1481, 1139, 1253, 530, 389, 210, 1091, 922, 1655, 914, 314, 287, 100, 326, 500, 1226, 26, 217, 372, 995, 245, 57, 534, 1503]
shuffleListELL = [590, 913, 657, 196, 489, 1048, 664, 432, 478, 492, 884, 526, 706, 7, 428, 238, 634, 237, 202, 858, 951, 771, 500, 640, 377, 305, 896, 6, 219, 485, 174, 250, 367, 379, 15, 733, 566, 537, 539, 504, 702, 248, 560, 887, 215, 1012, 355, 461, 893, 1021, 388, 1031, 721, 177, 68, 592, 334, 347, 454, 939, 746, 701, 203, 751, 996, 910, 462, 943, 265, 965, 840, 358, 579, 756, 517, 486, 438, 1041, 783, 737, 10, 944, 898, 218, 262, 110, 88, 595, 741, 866, 776, 542, 448, 914, 254, 988, 873, 267, 421, 983, 220, 11, 397, 1070, 366, 541, 221, 1016, 565, 292, 970, 908, 610, 1074, 928, 1035, 291, 780, 864, 306, 246, 195, 146, 157, 402, 240, 181, 469, 623, 710, 933, 1042, 427, 401, 794, 247, 24, 703, 130, 872, 830, 390, 619, 151, 139, 589, 824, 58, 378, 835, 482, 818, 812, 962, 82, 515, 183, 191, 418, 251, 503, 956, 995, 429, 954, 987, 687, 136, 608, 734, 778, 495, 811, 520, 895, 774, 480, 1045, 400, 909, 591, 1038, 315, 313, 1076, 1034, 148, 982, 960, 925, 230, 1052, 490, 534, 180, 658, 856, 102, 615, 79, 882, 376, 753, 936, 156, 570, 295, 394, 27, 796, 609, 718, 929, 138, 93, 103, 431, 620, 994, 947, 1073, 684, 842, 359, 952, 327, 675, 96, 1002, 804, 1018, 22, 792, 624, 1036, 161, 1037, 611, 164, 569, 158, 457, 81, 440, 819, 548, 423, 273, 531, 554, 233, 937, 249, 53, 236, 665, 309, 1033, 198, 514, 571, 907, 476, 208, 322, 277, 419, 228, 1010, 508, 36, 885, 137, 963, 874, 667, 735, 300, 679, 119, 789, 481, 123, 74, 20, 131, 498, 317, 1059, 269, 209, 392, 601, 616, 387, 1063, 707, 443, 87, 790, 224, 472, 363, 1020, 984, 639, 116, 587, 850, 301, 225, 1026, 769, 193, 59, 56, 764, 795, 807, 222, 976, 405, 803, 991, 332, 510, 303, 169, 54, 362, 578, 676, 104, 622, 698, 605, 1062, 8, 55, 860, 491, 524, 989, 1008, 144, 651, 258, 926, 584, 447, 23, 178, 931, 844, 547, 852, 577, 284, 600, 312, 41, 90, 831, 114, 512, 709, 108, 398, 213, 704, 77, 12, 120, 754, 549, 568, 113, 115, 855, 631, 561, 145, 410, 415, 37, 129, 493, 135, 25, 333, 915, 876, 133, 617, 370, 875, 436, 647, 199, 253, 886, 125, 626, 375, 94, 699, 474, 847, 117, 256, 172, 823, 310, 282, 538, 999, 879, 467, 14, 171, 822, 574, 562, 806, 563, 1025, 938, 1043, 650, 559, 28, 396, 550, 1023, 232, 187, 424, 78, 745, 433, 567, 290, 382, 152, 744, 385, 43, 871, 655, 336, 97, 71, 586, 63, 558, 507, 897, 576, 162, 35, 509, 749, 458, 868, 670, 67, 89, 649, 204, 98, 516, 1054, 768, 349, 519, 883, 348, 425, 869, 308, 340, 511, 839, 506, 1046, 430, 149, 127, 1001, 1039, 905, 160, 307, 668, 901, 727, 785, 920, 101, 661, 207, 60, 594, 168, 257, 26, 838, 546, 826, 953, 373, 325, 736, 761, 344, 627, 1067, 234, 372, 3, 833, 606, 92, 582, 38, 399, 903, 380, 17, 997, 881, 45, 352, 588, 338, 724, 671, 985, 170, 487, 1, 740, 361, 354, 31, 345, 986, 235, 845, 496, 767, 1056, 924, 466, 244, 645, 880, 930, 641, 386, 536, 593, 271, 452, 1040, 201, 165, 1050, 407, 782, 445, 360, 197, 13, 4, 1071, 311, 628, 799, 212, 239, 809, 106, 813, 972, 328, 450, 916, 350, 551, 959, 877, 255, 163, 888, 484, 614, 18, 697, 304, 95, 596, 286, 535, 705, 0, 383, 912, 757, 289, 368, 80, 279, 573, 760, 821, 497, 446, 902, 264, 381, 365, 841, 33, 581, 302, 216, 260, 899, 713, 743, 717, 124, 602, 923, 320, 471, 51, 444, 728, 787, 981, 638, 1066, 1049, 663, 723, 167, 331, 426, 816, 525, 1065, 633, 890, 750, 30, 625, 854, 708, 441, 245, 128, 555, 932, 229, 805, 648, 100, 1032, 19, 65, 643, 272, 243, 1019, 662, 166, 411, 1055, 44, 1027, 919, 979, 786, 66, 316, 1022, 918, 791, 111, 268, 48, 523, 330, 765, 900, 642, 1007, 1064, 1051, 150, 935, 518, 817, 369, 921, 669, 475, 859, 853, 677, 456, 459, 34, 159, 865, 851, 654, 575, 1003, 801, 449, 460, 357, 878, 1057, 659, 672, 468, 660, 974, 453, 1072, 513, 227, 276, 680, 73, 437, 501, 922, 849, 389, 730, 825, 892, 434, 16, 337, 696, 75, 607, 85, 473, 122, 175, 681, 134, 820, 762, 46, 39, 894, 990, 406, 1068, 323, 583, 1069, 948, 673, 557, 298, 261, 293, 1047, 194, 798, 815, 147, 810, 408, 1061, 716, 725, 961, 543, 141, 205, 686, 502, 226, 779, 942, 69, 341, 283, 889, 29, 533, 966, 604, 184, 266, 720, 603, 747, 62, 978, 479, 420, 176, 635, 599, 1014, 808, 945, 346, 973, 545, 112, 685, 288, 828, 483, 412, 867, 1000, 766, 391, 748, 846, 861, 499, 451, 598, 941, 70, 832, 1017, 126, 998, 597, 674, 693, 404, 652, 494, 632, 950, 153, 722, 758, 719, 442, 190, 637, 1013, 61, 109, 975, 1060, 417, 2, 715, 335, 732, 694, 690, 179, 278, 521, 682, 72, 356, 862, 528, 1006, 280, 691, 678, 612, 214, 540, 252, 993, 834, 580, 287, 618, 870, 188, 904, 1075, 971, 107, 242, 739, 646, 688, 259, 274, 192, 755, 906, 384, 797, 556, 530, 955, 40, 121, 413, 1009, 299, 1004, 339, 656, 967, 154, 711, 857, 827, 731, 784, 84, 957, 393, 285, 585, 814, 488, 32, 644, 319, 422, 980, 86, 297, 329, 1015, 772, 294, 140, 829, 666, 275, 105, 759, 342, 977, 836, 182, 689, 802, 934, 200, 553, 143, 505, 946, 189, 968, 692, 1053, 777, 572, 477, 564, 132, 969, 781, 552, 958, 621, 714, 891, 770, 773, 630, 843, 241, 21, 91, 435, 371, 964, 729, 464, 949, 263, 57, 155, 321, 374, 318, 1005, 364, 231, 351, 409, 395, 296, 927, 414, 1028, 837, 752, 544, 206, 470, 324, 940, 1058, 326, 118, 47, 211, 683, 532, 52, 529, 281, 463, 629, 522, 726, 455, 343, 223, 636, 848, 9, 793, 1030, 314, 5, 64, 142, 788, 712, 42, 416, 173, 1044, 527, 1024, 653, 185, 863, 695, 217, 270, 99, 49, 738, 763, 439, 76, 700, 992, 353, 742, 186, 50, 465, 775, 613, 403, 800, 917, 911, 83, 210, 1029, 1011]

for i in range(0, len(n)):
    coo_X_shuffled.append(coo_X[shuffleList[i]])
    csr_X_shuffled.append(csr_X[shuffleList[i]])
for i in range(0, len(ell_n)):
    ell_X_shuffled.append(ell_X[shuffleListELL[i]])

ell_Y_shuffled = []
coo_Y_shuffled = []
csr_Y_shuffled = []

for i in range(0, len(n)):
    coo_Y_shuffled.append(coo_times[shuffleList[i]])
    csr_Y_shuffled.append(csr_times[shuffleList[i]])
for i in range(0, len(ell_n)):
    ell_Y_shuffled.append(ell_times_final[shuffleListELL[i]])


"""
#COO
"""

# Split the targets into training/testing sets
### Preprocessing X
coo_eps = np.add(coo_X_shuffled, 1)
coo_X_log = np.log10(np.array(coo_eps))
mean_X = np.mean(coo_X_log,axis=0)
std_X = np.std(coo_X_log, axis=0)
coo_X_norm = np.multiply((coo_X_log - mean_X), (1 / std_X))

### Preprocessing Y
coo_log_times = np.log10(np.array(coo_Y_shuffled))
coo_mean_Y = np.mean(coo_log_times, axis = 0)
coo_std_Y = np.std(coo_log_times, axis = 0)
coo_norm_times = np.multiply((coo_log_times - coo_mean_Y), (1 / coo_std_Y))

### Split
train_coo_X = []
test_coo_X = []
train_coo_Y = []
test_coo_Y = []

coo_len = len(coo_X)

for i in range(0, 10):
    test_coo_X.append(coo_X_norm[(coo_len/10)*i: (coo_len/10)*(i+1)])
    x_temp = np.concatenate((coo_X_norm[0: (coo_len/10)*i], coo_X_norm[(coo_len/10)*(i+1): coo_len]), axis=0)
    train_coo_X.append(x_temp)
    test_coo_Y.append(coo_norm_times[(coo_len/10) * i: (coo_len/10) * (i + 1)])
    y_temp = np.concatenate((coo_norm_times[0: (coo_len/10) * i], coo_norm_times[(coo_len/10) * (i + 1): coo_len]), axis=0)
    train_coo_Y.append(y_temp)
    x_temp = []
    y_temp = []

output.write("COO" + "\n")
for test in range(0, 10):
    # Create linear regression object
    regr_coo = svm.SVR(C=400, epsilon=1e-5, kernel='rbf', verbose=5)
    coo_model = regr_coo.fit(train_coo_X[test], train_coo_Y[test])
    #print(coo_model)
    prediction_coo = coo_model.predict(test_coo_X[test])
    print "score of coo testing set " + str(test) + ": " + str(regr_coo.score(test_coo_X[test], test_coo_Y[test]))
    # t_pred = coo_model.predict(coo_X_train)
    sum_coo = 0

    coo_mult = np.multiply(prediction_coo, coo_std_Y)
    coo_sum_mean = coo_mult + coo_mean_Y
    coo_pred = np.power(10, coo_sum_mean)

    coo_y_data = coo_Y_shuffled[(coo_len/10)*test: (coo_len/10)*(test+1)]
    for i in range(0, len(test_coo_X[test])):
        # print str(prediction_coo[i]) + " vs " + str(coo_y_test[i])
        #print str(coo_pred[i]) + " vs " + str(coo_y_data[i])
        #if abs((coo_pred[i] - coo_y_data[i]) / coo_y_data[i]) > 0.5:
            #print abs((coo_pred[i] - coo_y_data[i]) / coo_y_data[i])
        sum_coo += abs((coo_pred[i] - coo_y_data[i]) / coo_y_data[i])
    print "rme of coo: " + str(sum_coo / len(test_coo_X[test]))
    output.write(str(test) + " " + str(regr_coo.score(test_coo_X[test], test_coo_Y[test])) + " " +
                 str(sum_coo / len(test_coo_X[test]))+ "\n")
output.write("ELL" + "\n")
"""
#ELL
"""

# Split the data into training/testing sets
### Preprocessing X
ell_eps = np.add(ell_X_shuffled, 1)
ell_X_log = np.log10(np.array(ell_eps))
mean_X = np.mean(ell_X_log,axis = 0)
std_X = np.std(ell_X_log, axis=0)
ell_X_norm = np.multiply((ell_X_log - mean_X), (1 / std_X))

### Preprocessing Y
ell_log_times = np.log10(np.array(ell_Y_shuffled))
ell_mean_Y = np.mean(ell_log_times, axis = 0)
ell_std_Y = np.std(ell_log_times, axis = 0)
ell_norm_times = np.multiply((ell_log_times - ell_mean_Y), (1 / ell_std_Y))

ell_len = len(ell_X)

### Split
train_ell_X = []
test_ell_X = []
train_ell_Y = []
test_ell_Y = []

for i in range(0, 10):
    test_ell_X.append(ell_X_norm[(ell_len/10)*i: (ell_len/10)*(i+1)])
    x_temp = np.concatenate((ell_X_norm[0: (ell_len/10)*i], ell_X_norm[(ell_len/10)*(i+1): ell_len]), axis=0)
    train_ell_X.append(x_temp)
    test_ell_Y.append(ell_norm_times[(ell_len/10) * i: (ell_len/10) * (i + 1)])
    y_temp = np.concatenate((ell_norm_times[0: (ell_len/10) * i], ell_norm_times[(ell_len/10) * (i + 1): ell_len]), axis=0)
    train_ell_Y.append(y_temp)
    x_temp = []
    y_temp = []

for test in range(0, 10):
    # Create linear regression object
    regr_ell = svm.SVR(C=0.5, epsilon=1e-6, kernel='rbf', verbose = 5)
    ell_model = regr_ell.fit(train_ell_X[test], train_ell_Y[test])
    #print(ell_model)
    prediction_ell = ell_model.predict(test_ell_X[test])
    print "score of ell testing set " + str(test) + ": " + str(regr_ell.score(test_ell_X[test], test_ell_Y[test]))
    # t_pred = ell_model.predict(ell_X_train)
    sum_ell = 0

    ell_mult = np.multiply(prediction_ell, ell_std_Y)
    ell_sum_mean = ell_mult + ell_mean_Y
    ell_pred = np.power(10, ell_sum_mean)

    ell_y_data = ell_Y_shuffled[(ell_len/10)*test: (ell_len/10)*(test+1)]
    for i in range(0, len(test_ell_X[test])):
        # print str(prediction_ell[i]) + " vs " + str(ell_y_test[i])
        #print str(ell_pred[i]) + " vs " + str(ell_y_data[i])
        #if abs((ell_pred[i] - ell_y_data[i]) / ell_y_data[i]) > 0.5:
            #print abs((ell_pred[i] - ell_y_data[i]) / ell_y_data[i])
        sum_ell += abs((ell_pred[i] - ell_y_data[i]) / ell_y_data[i])
    print "rme of ell: " + str(sum_ell / len(test_ell_X[test]))
    output.write(str(test)+ " " + str(regr_ell.score(test_ell_X[test], test_ell_Y[test])) + " " +
                 str(sum_ell / len(test_ell_X[test]))+ "\n")


"""
#CSR
"""

# Split the data into training/testing sets
### Preprocessing X
csr_eps = np.add(csr_X_shuffled, 1)
csr_X_log = np.log10(np.array(csr_eps))
csr_mean_X = np.mean(csr_X_log,axis = 0)
csr_std_X = np.std(csr_X_log, axis=0)
csr_X_norm = np.multiply((csr_X_log - csr_mean_X), (1 / csr_std_X))

### Preprocessing Y
csr_log_times = np.log10(np.array(csr_Y_shuffled))
csr_mean_Y = np.mean(csr_log_times, axis = 0)
csr_std_Y = np.std(csr_log_times, axis = 0)
csr_norm_times = np.multiply((csr_log_times - csr_mean_Y), (1 / csr_std_Y))

csr_len = len(csr_X)

### Split
train_csr_X = []
test_csr_X = []
train_csr_Y = []
test_csr_Y = []

for i in range(0, 10):
    test_csr_X.append(csr_X_norm[(csr_len/10)*i: (csr_len/10)*(i+1)])
    x_temp = np.concatenate((csr_X_norm[0: (csr_len/10)*i], csr_X_norm[(csr_len/10)*(i+1): csr_len]), axis=0)
    train_csr_X.append(x_temp)
    test_csr_Y.append(csr_norm_times[(csr_len/10) * i: (csr_len/10) * (i + 1)])
    y_temp = np.concatenate((csr_norm_times[0: (csr_len/10) * i], csr_norm_times[(csr_len/10) * (i + 1): csr_len]), axis=0)
    train_csr_Y.append(y_temp)
    x_temp = []
    y_temp = []

output.write("CSR" + "\n")
for test in range(0, 10):
    # Create linear regression object
    regr_csr = svm.SVR(C=1, epsilon=1e-7, kernel='rbf', verbose= 5)
    csr_model = regr_csr.fit(train_csr_X[test], train_csr_Y[test])
    #print(csr_model)
    prediction_csr = csr_model.predict(test_csr_X[test])
    print "score of csr testing set " + str(test) + ": " + str(regr_csr.score(test_csr_X[test], test_csr_Y[test]))
    # t_pred = csr_model.predict(csr_X_train)
    sum_csr = 0

    csr_mult = np.multiply(prediction_csr, csr_std_Y)
    csr_sum_mean = csr_mult + csr_mean_Y
    csr_pred = np.power(10, csr_sum_mean)

    csr_y_data = csr_Y_shuffled[(csr_len/10)*test: (csr_len/10)*(test+1)]
    for i in range(0, len(test_csr_X[test])):
        # print str(prediction_csr[i]) + " vs " + str(csr_y_test[i])
        #print str(csr_pred[i]) + " vs " + str(csr_y_data[i])
        #if abs((csr_pred[i] - csr_y_data[i]) / csr_y_data[i]) > 0.5:
            #print abs((csr_pred[i] - csr_y_data[i]) / csr_y_data[i])
        sum_csr += abs((csr_pred[i] - csr_y_data[i]) / csr_y_data[i])
    print "rme of csr: " + str(sum_csr / len(test_csr_X[test]))
    output.write(str(test) + " " + str(regr_csr.score(test_csr_X[test], test_csr_Y[test])) + " " +
                 str(sum_csr / len(test_csr_X[test])) + "\n")
import os
import glob
import code
import numpy as np


s = '''p3_c25_bin2
p3_c27_bin2
p2_c58
p3_c19_bin2
p3_c4_bin2
p3_c32_bin2
p2_c10
p6_c14
p3_c22_bin2
p6_c41
p4_c9
p3_c7_bin2
p2_c53
p6_c26
p3_c15_bin2
p3_c28_bin2
p6_c18
p9_c31
p4_c17
p9_c49
p3_c5_bin2
p4_c14
p3_c3_bin2
p4_c15
p3_c30_bin2
p4_c11
p2_c56
p4_c4
p3_c33_bin2
p3_c13_bin2
p3_c29_bin2
p3_c1_bin2
p4_c10
p9_c44
p2_c45
p3_c8_bin2
p4_c16
p3_c6_bin2
p6_c35
p3_c31_bin2
p4_c3
p4_c2
p3_c18_bin2
p3_c11_bin2
p6_c9
p4_c12
p4_c6
p3_c20_bin2
p3_c12_bin2
p2_c40'''

l = s.split()
l = [x.replace('_bin2','') for x in l]
l = np.asarray(l)

k = np.genfromtxt('./names_and_counts.txt', dtype=str, delimiter=',')
names = k[:,0]

count = 0

for each in names:
	indx = (np.where(l == each))
	print indx
	


code.interact(local=locals())

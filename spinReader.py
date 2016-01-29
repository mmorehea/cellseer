import os
import sys

path = sys.argv[1]

import os
count = 0;
string = '/pcd_spin_image '
for u in os.listdir(path):



	if u[-3:] == 'bin':
		
		filePath = path+u
		print filePath	
		f = open(filePath, 'rb')
		lineCount = 0
		for line in f:
			lineCount += 1;
			#g = line[2:-1]
			#gSplit = g.split(',')
			#print(len(gSplit))
		print(lineCount)
		count = count+lineCount
print count
		
	
			
		



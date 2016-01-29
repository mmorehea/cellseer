import os
import sys

path = sys.argv[1]

import os


for u in os.listdir(path):



	if u[-3:] == 'txt':
		s = path+ u[0:-4]
		os.rename(path+u, s)
		
			
		



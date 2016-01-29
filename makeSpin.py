import os
import sys
import code
path = sys.argv[1]

import os

string = '/home/mdm/Projects/cellseer_build/pcd_spin_image '
for c in os.listdir(path):



	if c[-3:] == 'pcd':
		
		string +=c
		code.interact(local=locals())
		print string		
		os.system(string)
		string = '/home/mdm/Projects/cellseer_build/pcd_spin_image '
	
			
		



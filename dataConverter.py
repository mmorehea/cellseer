import os
import sys
import glob
import code
path = sys.argv[1]

import os

string = 'pcl_obj2pcd '
l = glob.glob(path + '*.obj')
for i in l:
	print i
	   
	
	
	string +=i + ' '+i[0:-3]+'pcd'+ ' -copy_normals 1'
	print string		
	
	os.system(string)
	string = 'pcl_obj2pcd '
			
		



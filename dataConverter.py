import os
import sys

path = sys.argv[1]

import os

string = 'pcl_obj2pcd '
for i in os.listdir(path):
    x = os.path.join(path,i)
    for u in os.listdir(x):
	
	c = os.path.join(x,u)
        
    	if c[-3:] == 'obj':
                #print u
		string +=c+' ';
		string +=x+u[0:-3]+'pcd'+ ' -copy_normals 1'
		print string		
		os.system(string)
		string = 'pcl_obj2pcd '
			
		



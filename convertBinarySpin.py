import os
import sys
import cPickle as pickle
path = sys.argv[1]

import os
w = open('textSpin.txt', 'w')
fi = open(path,'rb')
for each in fi:
	w.write(each)
	

print fi

#g = line[2:-1]
#gSplit = g.split(',')
#print(len(gSplit))

		
	
			

import os
import sys
import cPickle as pickle
import os
import re
import numpy

path = sys.argv[1]




bad_chars = '() \n'
rgx = re.compile('[%s]' % bad_chars)

kmeansList = []



labelRest = open('restLabel.txt', 'w')
klabelcount = 0
rlabelcount = 0
for u in os.listdir(path):


	if os.path.isfile(path+u):

		count = 0
		filePath = path+u
		print filePath	
		with open(filePath, 'r') as f:
			count = len(f.readlines())
		print count		
		a = numpy.linspace(1, count, 1000)
		a = a.astype(int)
		#raw_input("Press Enter to continue...")
		
		with open(filePath, 'r') as f:			
			for i, line in enumerate(f):
				if i in a:
					o= rgx.sub('', line)
					i = o.split(',')
					klabelcount += 1
					kmeansList.append(i)
					
					
				
		#raw_input("Press Enter to continue...")	
		kmeans = open(filePath+'kmeanList', 'wb')
		pickle.dump(kmeansList, kmeans)
		kmeans.close()
		kmeansList = []
		
					
		
			






				
					
				
				

				
			
				
			

			
		



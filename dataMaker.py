import cPickle as pickle
import numpy
import os
import sys
import svmutil


path = sys.argv[1] 
label = []
points = []
st = ''
data = open('neuronData','w')
for u in os.listdir(path): 
	if u[-2:] == 'WC':
		st = ''
		filePath = path+u
		WC = pickle.load(open(filePath, 'rb'))
		st+=u[1]
		for i, x in enumerate(WC):
			st+=' '+str(i+1)+':'+str(x)
		st+='\n'
		data.write(st)

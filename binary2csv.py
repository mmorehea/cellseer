import cPickle as pickle
import numpy
import os
import sys
import svmutil

def main(path):
	
	label = []
	points = []
	for u in os.listdir(path): 
		if u[-2:] == 'WC':
			
			filePath = path+u
			WC = pickle.load(open(filePath, 'rb'))
			label.append(u[1])
			f = open(filePath + '.csv', 'w')
			print WC
			for each in WC:
				print type(each)
				f.write(str(each) + ', ')
	s = open('./labels.csv', 'w')
	for ll in label:
		s.write(ll+ ', ')
			
			
if __name__ == "__main__":
	path = sys.argv[1] 
	main(path)			

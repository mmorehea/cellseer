import os
import sys
import cPickle as p
import re
import numpy as np
import cv2
import math
import random
from matplotlib import pyplot as plt

def knn_search(x, D, K):
 """ find K nearest neighbours of data among D """
  # euclidean distances from the other points
  #must take in data like D = array(10,3)
  # and x = array(1,3)
  #returns indicie of center (0 start)
 cc = ((D - x)**2)
 
 sqd = np.sqrt(cc.sum(axis=1))
 idx = np.argsort(sqd) # sorting
 # return the indexes of K nearest neighbours
 return idx[:K]


def main(path, numberPoints):
	bad_chars = '() \n'
	rgx = re.compile('[%s]' % bad_chars) 
	path = sys.argv[1] 
	r = open(path+'centers', 'rb')
	centers = p.load(r)


	ooo = 0
	for u in os.listdir(path): 
		if len(u) == 6:
			ooo += 1
			print ooo
			with open(path+u, 'r') as f:
				count = len(f.readlines())
				
				a = np.linspace(1, count, 1000)
				a = a.astype(int)
			WC = open(path+u+'_WC', 'rb')
			wordCount = p.load(WC)
			WC.close()
			
			
			pointsToGet = []
			
			cc=0
			while (cc < int(numberPoints)):
				cc = cc+1
				
				pointsToGet.append(random.randint(1, count))
			
			
			with open(path+u, 'r') as f:			
				for i, line in enumerate(f):
					if i in pointsToGet:
						o= rgx.sub('', line)
						ii = o.split(',')
						ii = np.array(ii)
						
						rr = np.float32(ii)
						rr.shape=(1,153)
						
						centNum = knn_search(rr, centers, 1)
						wordCount[centNum[0]] = wordCount[centNum[0]]+1
			WC2 = open(path+u+'_WC', 'wb')
			p.dump(wordCount, WC2)
			print wordCount
			WC2.close()

if __name__ == "__main__":	
	path = sys.argv[1] 
	numberPoints = sys.argv[2]
	main(path, int(numberPoints))
						

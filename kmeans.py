import os
import sys
import cPickle as pickle
import os
import re
import numpy as np
import cv2
from matplotlib import pyplot as plt




def main(path, K):
	kLabels = []
	wholeK = np.zeros([1,153])
	labelLines = 0
	for u in os.listdir(path): 
		if u[-4:] == 'List':
			labelLines += 1
			filePath = path+u
			kPoints = pickle.load(open(filePath, 'rb'))
			kPoints = np.asarray(kPoints)
			kLabels.append(u[0:6])
			wholeK = np.append(wholeK, kPoints, axis=0)
		
			


	Z = np.float32(wholeK)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_PP_CENTERS)

	cen = open(path+'centers', 'wb')
	pickle.dump(center, cen)

	for y in range(0,labelLines):
		l = label[0+(1000*y):999+(1000*y)]
		ll = l.tolist()
		lll = [item for sublist in ll for item in sublist] 
		
		wordcount = []
		for x in range(K):
			wordcount.append(lll.count(x))
		stringSave = path+kLabels[y] + '_WC'
		
		pickle.dump(wordcount, open(stringSave, 'wb'))	

if __name__ == "__main__":	
	path = sys.argv[1] 
	k = sys.argv[2]
	main(path, k)

	 
		


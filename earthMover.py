from emd import emd
import numpy as np
from cv2 import *

def calcEM(hist1,hist2,h_bins,s_bins):

	#Define number of rows
	numRows = h_bins*s_bins

	sig1 = cv.CreateMat(numRows, 3, cv.CV_32FC1)
	sig2 = cv.CreateMat(numRows, 3, cv.CV_32FC1)    

	for h in range(h_bins):
		for s in range(s_bins): 
			bin_val = cv.QueryHistValue_2D(hist1, h, s)
			cv.Set2D(sig1, h*s_bins+s, 0, cv.Scalar(bin_val))
			cv.Set2D(sig1, h*s_bins+s, 1, cv.Scalar(h))
			cv.Set2D(sig1, h*s_bins+s, 2, cv.Scalar(s))

			bin_val = cv.QueryHistValue_2D(hist2, h, s)
			cv.Set2D(sig2, h*s_bins+s, 0, cv.Scalar(bin_val))
			cv.Set2D(sig2, h*s_bins+s, 1, cv.Scalar(h))
			cv.Set2D(sig2, h*s_bins+s, 2, cv.Scalar(s))

#This is the important line were the OpenCV EM algorithm is called
	return cv.CalcEMD2(sig1,sig2,cv.CV_DIST_L2)

l = np.genfromtxt('./names_and_counts.txt', dtype=str, delimiter=',')
l= l[:,1:]
l=l.astype(float)
row_sums = l.sum(axis=1)
new_matrix = l / row_sums[:, np.newaxis]
xx, yy = new_matrix.shape

simMatrix = np.zeros((xx,xx))

for ii in range(xx):
	print ii
	for jj in range(xx):
		
		simMatrix[ii,jj] = calcEM(new_matrix[ii,:], new_matrix[jj,:], 400, 400)
		
code.interact(local=locals())
		

np.savetxt("EMD_sim.csv.csv", simMatrix, delimiter=",")

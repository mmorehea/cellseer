import numpy as np
import os
import code
import glob
import math

l = glob.glob('./spinimages/*.pcdspinImages.bin')



		
cnt = 0
for each in l:
	print each
	print cnt
	cnt += 1
	w = open(each)
	q = w.readlines()
	ww = [l.strip()[1:-1].split(',') for l in q]
	ww = np.asarray(ww)
	ww = ww.astype('float')
	xx, yy = ww.shape
	xToPick = 2000
	picks = np.random.choice(xx, xToPick, replace=False)
	
	#parse file name
	age = each.split('/')[-1].split('_')[0][1:]
	#cell = each.split('/')[-1].split('_')[2].split('.')[1:]
	cell = each.split('/')[-1].split('_')[2].split('.')[0][1:]
	print age, ' ', cell
	#
	
	nonPick = np.setdiff1d(np.asarray(range(0,xx)),picks)
	
		
	
	
	ss = ww[picks]
	code.interact(local=locals())
	fileName = './picked1/'+age + '_' + cell +'_vectorspicked.csv'
	np.savetxt(fileName, ss, delimiter=',')
	ss= None
		
		
	dd = ww[nonPick]
	fileName = './picked1/'+age+'_'+cell+'_vectorsnotpicked.csv'
	np.savetxt(fileName, dd, delimiter = ',')
	dd = None
	
	
	

	

	
	



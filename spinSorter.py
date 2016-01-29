import numpy as np
import os
import code
import glob
import math

l = glob.glob('./data/spinimages/*.pcdspinImages.bin')


f = open('picksIndex.txt', 'a+')
		
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
	xToPick = math.floor(xx*.01)
	picks = np.random.choice(xx, xToPick, replace=False)
	
	#parse file name
	age = each.split('/')[-1].split('_')[0][1:]
	#cell = each.split('/')[-1].split('_')[2].split('.')[1:]
	cell = each.split('/')[-1].split('_')[2].split('.')[0][1:]
	print age, ' ', cell
	#code.interact(local=locals())
	
	picks = np.insert(picks, 0, cell)
	picks = np.insert(picks, 0, age)
	
	
	ss = ww[picks]
	fileName = age + '_' + cell +'_vectors.txt'
	np.savetxt(fileName, ss, delimiter=',')
	ss=[]
		
	
	f.write(picks)
	

	
	
code.interact(local=locals())
	
	



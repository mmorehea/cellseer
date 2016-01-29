from emd import emd
import numpy as np
from sklearn.metrics.pairwise import chi2_kernel
import code


l = np.genfromtxt('./names_and_counts.txt', dtype=str, delimiter=',')
l= l[:,1:]
l=l.astype(float)
row_sums = l.sum(axis=1)
new_matrix = l / row_sums[:, np.newaxis]
xx, yy = new_matrix.shape


simMatrix = chi2_kernel(new_matrix)
		
code.interact(local=locals())
		

np.savetxt("chi_sim.csv", simMatrix, delimiter=",")

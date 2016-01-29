import numpy as np
import code
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine


def fast_hik(x, y):
    return np.minimum(x, y).sum()
    
    
l = np.genfromtxt('./output.txt', delimiter=',')
labels = l[:,0]
l = l[:,1:]
hist = pairwise_distances(l, metric=fast_hik)
histNorm = hist/np.max(hist)
cosineSim = 1-pairwise_distances(l, metric="cosine")

np.savetxt("hist_intersect_similarity.csv", hist, delimiter=",")
np.savetxt("hist_intersect_similarity_normed.csv", histNorm, delimiter=",")
np.savetxt('cosine_sim.csv', cosineSim, delimiter=',')

np.savetxt('labels.csv', labels, delimiter=',')


code.interact(local=locals())

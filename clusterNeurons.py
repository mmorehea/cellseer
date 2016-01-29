import numpy as np
import code
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import cluster

#def MDPA(wordCount1, wordCount2):
	#s = 0
	#for i in range(len(wordCount1)):
		#for j in range(i):
				#s += abs(wordCount1[j] - wordCount2[j])
	#return s

#l = np.genfromtxt('./hist_intersect_similarity.csv', delimiter=',')
#l = np.genfromtxt('./names_and_counts.txt', dtype=str, delimiter=',')
#l= l[:,1:]
#l=l.astype(float)
#row_sums = l.sum(axis=1)
#new_matrix = l / row_sums[:, np.newaxis]
#xx, yy = new_matrix.shape
#print yy
#simMatrix = np.zeros((xx,xx))

#for ii in range(xx):
	#print ii
	#for jj in range(xx):
		
		#simMatrix[ii,jj] = MDPA(new_matrix[ii,:], new_matrix[jj,:])
		
#code.interact(local=locals())
		

#np.savetxt("MDPA_sim.csv.csv", simMatrix, delimiter=",")

	
	
l = np.genfromtxt('./chi_sim.csv', dtype=float, delimiter=',')


#l = l/np.max(l)

trueLabels = np.genfromtxt('./labels.csv', delimiter=',')
trueLabels[np.where(trueLabels==2)] = 0
trueLabels[np.where(trueLabels==3)] = 1
trueLabels[np.where(trueLabels==4)] = 2
trueLabels[np.where(trueLabels==6)] = 3
trueLabels[np.where(trueLabels==9)] = 4
kmeans = cluster.KMeans(init='k-means++', n_clusters=5, n_init=10)
kmeans.fit(l)
labels = kmeans.labels_
code.interact(local=locals())
print "Raw, kmeans 5"
print("Completeness: %0.3f" % metrics.completeness_score(trueLabels, labels))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(trueLabels, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(trueLabels, labels))

af = cluster.AffinityPropagation(affinity="precomputed").fit(l)
labels = af.labels_
print "Raw, AP"
print("Completeness: %0.3f" % metrics.completeness_score(trueLabels, labels))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(trueLabels, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(trueLabels, labels))
code.interact(local=locals())

#l = np.genfromtxt('./cosine_sim.csv', delimiter=',')
#l=l.astype(float)
#l = l/np.max(l)
#kmeans = cluster.KMeans(init='k-means++', n_clusters=5, n_init=10)
#kmeans.fit(l)
#labels = kmeans.labels_
#print "hist_intersect, kmeans"
#print("Completeness: %0.3f" % metrics.completeness_score(trueLabels, labels))
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(trueLabels, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(trueLabels, labels))

#af = cluster.AffinityPropagation().fit(l)
#labels = af.labels_
#print "hist_intersection, AP"
#print("Completeness: %0.3f" % metrics.completeness_score(trueLabels, labels))
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(trueLabels, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(trueLabels, labels))

#l = np.genfromtxt('./hist_intersect_similarity.csv', delimiter=',')
#l=l.astype(float)
#l = l/np.max(l)
#kmeans = cluster.KMeans(init='k-means++', n_clusters=5, n_init=10)
#kmeans.fit(l)
#labels = kmeans.labels_
#print "hist_intersect, kmeans"
#print("Completeness: %0.3f" % metrics.completeness_score(trueLabels, labels))
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(trueLabels, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(trueLabels, labels))

#af = cluster.AffinityPropagation().fit(l)
#labels = af.labels_
#print "hist_intersection, AP"
#print("Completeness: %0.3f" % metrics.completeness_score(trueLabels, labels))
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(trueLabels, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(trueLabels, labels))

#code.interact(local=locals())

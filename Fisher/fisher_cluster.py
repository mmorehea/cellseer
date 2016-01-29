# -*- coding: utf-8 -*-

# /*
#  ██████ ███████ ██	  ██	  ███████ ███████ ███████ ██████
# ██	  ██	  ██	  ██	  ██	  ██	  ██	  ██   ██
# ██	  █████   ██	  ██	  ███████ █████   █████   ██████
# ██	  ██	  ██	  ██		   ██ ██	  ██	  ██   ██
#  ██████ ███████ ███████ ███████ ███████ ███████ ███████ ██   ██
# Written by Michael Morehead
# WVU Center for Neuroscience, 2016

import numpy as np
import code
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import cluster
import math
import svmutil
from sklearn import svm, grid_search
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import LeaveOneOut
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import time


# ████████ ██ ███	███ ███████ ██████
#	██	██ ████  ████ ██	  ██   ██
#	██	██ ██ ████ ██ █████   ██████
#	██	██ ██  ██  ██ ██	  ██   ██
#	██	██ ██	  ██ ███████ ██   ██

def timeme(method):
	def wrapper(*args, **kw):
		print('Starting ' + method.__name__)
		start_time = time.time()
		startTime = int(round(time.time() * 1000))
		result = method(*args, **kw)
		endTime = int(round(time.time() * 1000))
		printt('Function ' + method.__name__ + ' complete ', time.time() - start_time)
		print(endTime - startTime, 'ms')
		return result

	return wrapper


# ██   ██ ███████  ██████  ██	  ██████
# ██  ██  ██	  ██	██ ██	  ██   ██
# █████   █████   ██	██ ██	  ██   ██
# ██  ██  ██	  ██	██ ██	  ██   ██
# ██   ██ ██	   ██████  ███████ ██████

@timeme
def kfold(data, labels, k):
	prabs = []

	for xxx in range(0, 10):
		picks = np.random.choice(len(data), len(data) / k, replace=False)
		testLabel = labels[picks]
		testPoint = data[picks]
		trainPoint = data[np.setdiff1d(range(0, len(data)), picks)]
		trainLabel = labels[np.setdiff1d(range(0, len(data)), picks)]

		trainLabel = trainLabel.tolist()
		trainPoint = trainPoint.tolist()

		prob = svmutil.svm_problem(trainLabel, trainPoint)
		param = svmutil.svm_parameter('-t 3 -c 4 -b 1 -q')
		testLabel = testLabel.tolist()
		testPoint = testPoint.tolist()

		m = svmutil.svm_train(prob, param)
		svmutil.svm_save_model('n.model', m)

		p_label, p_acc, p_val = svmutil.svm_predict(testLabel, testPoint, m, '-b 1')

		prabs.append(p_acc[0])

	print sum(prabs) / float(len(prabs))
	print 'std' + str(np.std(prabs))
	return sum(prabs) / float(len(prabs))


# ██ ███	███ ██████   ██████  ██████  ████████
# ██ ████  ████ ██   ██ ██	██ ██   ██	██
# ██ ██ ████ ██ ██████  ██	██ ██████	 ██
# ██ ██  ██  ██ ██	  ██	██ ██   ██	██
# ██ ██	  ██ ██	   ██████  ██   ██	██

@timeme
def import_csv_data(data_path, response_path):
	data = np.genfromtxt(data_path, dtype=float, delimiter=',')
	response = np.genfromtxt(response_path, dtype=str, delimiter=',')

	response = [x[0] for x in response]
	response = np.asarray(response)
	nans = [i for i, j in enumerate(data) if math.isnan(j[0])]
	data = np.delete(data, nans, axis=0)

	response = np.delete(response, nans, axis=0)
	response = response.astype(int)
	response[np.where(response == 9)] = 6
	response[np.where(response == 2)] = 0
	response[np.where(response == 3)] = 1
	response[np.where(response == 4)] = 2
	response[np.where(response == 6)] = 4
	scaler = StandardScaler().fit(data)
	data = scaler.transform(data)

	return data, response


# ██████  ██	   █████  ██ ███	██ ███████ ██	██ ███	███
# ██   ██ ██	  ██   ██ ██ ████   ██ ██	  ██	██ ████  ████
# ██████  ██	  ███████ ██ ██ ██  ██ ███████ ██	██ ██ ████ ██
# ██	  ██	  ██   ██ ██ ██  ██ ██	  ██  ██  ██  ██  ██  ██
# ██	  ███████ ██   ██ ██ ██   ████ ███████   ████   ██	  ██
# kern = 'rbf', 'linear', 'poly', 'sigmoid'
@timeme
def plain_svm(data, response, kern='linear'):
	clf = svm.SVC(kernel=kern)
	scores = cross_validation.cross_val_score(clf, data, response)
	print scores
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# /*
#  ██████  ██████  ███    ██ ███████ ██    ██ ███████ ██  ██████  ███    ██         ███    ███  █████  ████████ ██████  ██ ██   ██
# ██      ██    ██ ████   ██ ██      ██    ██ ██      ██ ██    ██ ████   ██         ████  ████ ██   ██    ██    ██   ██ ██  ██ ██
# ██      ██    ██ ██ ██  ██ █████   ██    ██ ███████ ██ ██    ██ ██ ██  ██         ██ ████ ██ ███████    ██    ██████  ██   ███
# ██      ██    ██ ██  ██ ██ ██      ██    ██      ██ ██ ██    ██ ██  ██ ██         ██  ██  ██ ██   ██    ██    ██   ██ ██  ██ ██
#  ██████  ██████  ██   ████ ██       ██████  ███████ ██  ██████  ██   ████ ███████ ██      ██ ██   ██    ██    ██   ██ ██ ██   ██
# */
@timeme
def make_confusion_matrix(data, response, kern='linear'):
	X_train, X_test, y_train, y_test = train_test_split(data, response)
	classifier = svm.SVC(kernel=kern)
	y_pred = classifier.fit(X_train, y_train).predict(X_test)
	cm = confusion_matrix(y_test, y_pred)
	print cm


# ██	   ██████   ██████
# ██	  ██	██ ██	██
# ██	  ██	██ ██	██
# ██	  ██	██ ██	██
# ███████  ██████   ██████
@timeme
def loo(data, response, kern, number_of_runs):
	clf = svm.SVC(kernel=kern)
	loo = LeaveOneOut(len(data))
	accuracies = []
	for run in range(0, number_of_runs - 1):
		for train, test in loo:
				score = clf.fit(data[train], response[train]).score(data[test], response[test])
				accuracies.append(score)

	print 'LOO: '
	print sum(accuracies) / float(len(accuracies))


#  ██████  ██████  ██ ██████
# ██	   ██   ██ ██ ██   ██
# ██   ███ ██████  ██ ██   ██
# ██	██ ██   ██ ██ ██   ██
#  ██████  ██   ██ ██ ██████
@timeme
def grid(data, response):
	X_train, X_test, y_train, y_test = train_test_split(data, response)
	tuned_parameters = [
		{'kernel': ['rbf', 'linear', 'sigmoid', 'poly'],
			'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]

	print("# Tuning hyper-parameters")
	print()

	clf = grid_search.GridSearchCV(SVC(), tuned_parameters, cv=5)
	clf.fit(X_train, y_train)
	print("Best parameters set found on development set:")
	print()
	print(clf.best_params_)
	print()
	print("Grid scores on development set:")
	print()
	for params, mean_score, scores in clf.grid_scores_:
		print("%0.3f (+/-%0.03f) for %r"
								% (mean_score, scores.std() * 2, params))
	print()

	print("Detailed classification report:")
	print()
	print("The model is trained on the full development set.")
	print("The scores are computed on the full evaluation set.")
	print()
	y_true, y_pred = y_test, clf.predict(X_test)
	print(metrics.classification_report(y_true, y_pred))
	print()


# /*
#  █████  ██████
# ██   ██ ██   ██
# ███████ ██████
# ██   ██ ██
# ██   ██ ██
# */
@timeme
def ap(data, response):
	af = cluster.AffinityPropagation().fit(data)
	labels = af.labels_
	clusternames = defaultdict(list)
	for i, label in enumerate(labels):
		clusternames[label].append(response[i])
	# clusternames now holds a map from cluster label to list of sequence names
	for k, v in clusternames.items():
		print k, v


# /*
# ███    ███ ███████  █████  ███    ██ ███████ ██   ██ ██ ███████ ████████
# ████  ████ ██      ██   ██ ████   ██ ██      ██   ██ ██ ██         ██
# ██ ████ ██ █████   ███████ ██ ██  ██ ███████ ███████ ██ █████      ██
# ██  ██  ██ ██      ██   ██ ██  ██ ██      ██ ██   ██ ██ ██         ██
# ██      ██ ███████ ██   ██ ██   ████ ███████ ██   ██ ██ ██         ██
# */
@timeme
def mean_shift(data, response):
	print 'MEANSHIFT'
	af = cluster.MeanShift().fit(data)
	labels = af.labels_
	clusternames = defaultdict(list)

	for i, label in enumerate(labels):
		clusternames[label].append(response[i])

	# clusternames now holds a map from cluster label to list of sequence names
	# Print out the label with the list
	for k, v in clusternames.items():
		print k, v


# /*
# ██████  ██████  ███████  ██████  █████  ███	██
# ██   ██ ██   ██ ██	  ██	  ██   ██ ████   ██
# ██   ██ ██████  ███████ ██	  ███████ ██ ██  ██
# ██   ██ ██   ██	  ██ ██	  ██   ██ ██  ██ ██
# ██████  ██████  ███████  ██████ ██   ██ ██   ████
# */
@timeme
def dbscan(data, response):
	print 'DBSCAN'
	af = cluster.DBSCAN().fit(data)
	labels = af.labels_

	clusternames = defaultdict(list)

	for i, label in enumerate(labels):
		clusternames[label].append(response[i])
	for k, v in clusternames.items():
		print k, v


# /*
# ██   ██ ███	███ ███████  █████  ███	██ ███████
# ██  ██  ████  ████ ██	  ██   ██ ████   ██ ██
# █████   ██ ████ ██ █████   ███████ ██ ██  ██ ███████
# ██  ██  ██  ██  ██ ██	  ██   ██ ██  ██ ██	  ██
# ██   ██ ██	  ██ ███████ ██   ██ ██   ████ ███████
# */
def kmeans(data, response):
	km = KMeans(init='k-means++', n_clusters=4, n_init=10)
	print 'KMEANS'
	af = km.fit(data)
	labels = af.labels_

	clusternames = defaultdict(list)

	for i, label in enumerate(labels):
		clusternames[label].append(response[i])
	for k, v in clusternames.items():
		print k, v


# /*
# ████████ ██ ███	███ ███████		 ██████  ██████  ██ ███	██ ████████
#	██	██ ████  ████ ██			  ██   ██ ██   ██ ██ ████   ██	██
#	██	██ ██ ████ ██ █████		   ██████  ██████  ██ ██ ██  ██	██
#	██	██ ██  ██  ██ ██			  ██	  ██   ██ ██ ██  ██ ██	██
#	██	██ ██	  ██ ███████ ███████ ██	  ██   ██ ██ ██   ████	██
# */
def printt(message, time):
	string = '{0:.<50}'.format(message) + ' ' + str(time) + ' seconds'
	print(string)


# /*
# ████████ ███████ ███████ ████████	  ███████ ██	██ ██ ████████ ███████
#	██	██	  ██		 ██		 ██	  ██	██ ██	██	██
#	██	█████   ███████	██		 ███████ ██	██ ██	██	█████
#	██	██		   ██	██			  ██ ██	██ ██	██	██
#	██	███████ ███████	██ ███████ ███████  ██████  ██	██	███████
# */
def test_suite():
	print('Importing')
	data, response = import_csv_data('./encodings.csv', './junk2.csv')
	# kfold(data, response, 2)
	plain_svm(data, response, 'linear')
	make_confusion_matrix(data, response, 'linear')
	make_confusion_matrix(data, response, 'poly')
	make_confusion_matrix(data, response, 'rbf')
	# loo(data, response, 'linear', 3)
	# grid(data, response)
	ap(data, response)
	# mean_shift(data, response)
	# dbscan(data, response)
	kmeans(data, response)

test_suite()

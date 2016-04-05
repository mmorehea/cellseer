# -*- coding: utf-8 -*-

# /*
#  ██████ ███████ ██	  ██	  ███████ ███████ ███████ ██████
# ██	  ██	  ██	  ██	  ██	  ██	  ██	  ██   ██
# ██	  █████   ██	  ██	  ███████ █████   █████   ██████
# ██	  ██	  ██	  ██	       ██ ██	  ██	  ██   ██
#  ██████ ███████ ███████ ███████ ███████ ███████ ███████ ██   ██
# Written by Michael Morehead
# WVU Center for Neuroscience, 2016
# A library for processing SWCs and OBJs of neuron models
import sys
import matlab.engine
import numpy as np
import code
import os
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import cluster
import math
import glob
from sklearn import svm, grid_search
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import LeaveOneOut
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
import time
import flatten
import matplotlib.pyplot as plt
import shutil
import pcl
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call
import subprocess
import flatten
import matplotlib.pyplot as plt
import shutil


# /*
# ████████ ██ ███    ███ ███████ ██████
#    ██    ██ ████  ████ ██      ██   ██
#    ██    ██ ██ ████ ██ █████   ██████
#    ██    ██ ██  ██  ██ ██      ██   ██
#    ██    ██ ██      ██ ███████ ██   ██
# */
def timeme(method):
	def wrapper(*args, **kw):
		print('Starting ' + method.__name__)
		start_time = time.time()
		startTime = int(round(time.time() * 1000))
		result = method(*args, **kw)
		endTime = int(round(time.time() * 1000))
		timer_print('Function ' + method.__name__ + ' complete ', time.time() - start_time)
		print(endTime - startTime, 'ms')
		return result

	return wrapper


# /*
# ████████ ██ ███    ███ ███████ ██████          ██████  ██████  ██ ███    ██ ████████
#    ██    ██ ████  ████ ██      ██   ██         ██   ██ ██   ██ ██ ████   ██    ██
#    ██    ██ ██ ████ ██ █████   ██████          ██████  ██████  ██ ██ ██  ██    ██
#    ██    ██ ██  ██  ██ ██      ██   ██         ██      ██   ██ ██ ██  ██ ██    ██
#    ██    ██ ██      ██ ███████ ██   ██ ███████ ██      ██   ██ ██ ██   ████    ██
# */
def timer_print(message, time):
	string = '{0:.<50}'.format(message) + ' ' + str(time) + ' seconds'
	print(string)


# ██   ██ ███████  ██████  ██	  ██████
# ██  ██  ██	  ██	██ ██	  ██   ██
# █████   █████   ██	██ ██	  ██   ██
# ██  ██  ██	  ██	██ ██	  ██   ██
# ██   ██ ██	   ██████  ███████ ██████

@timeme
def kfold(data, labels, k):
	try:
		import svmutil
	except:
		return 0
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


# /*
# ██ ███    ███ ██████   ██████  ██████  ████████
# ██ ████  ████ ██   ██ ██    ██ ██   ██    ██
# ██ ██ ████ ██ ██████  ██    ██ ██████     ██
# ██ ██  ██  ██ ██      ██    ██ ██   ██    ██
# ██ ██      ██ ██       ██████  ██   ██    ██
# */
@timeme
def import_csv_data(data_path, response_path):
	data = np.genfromtxt(data_path, dtype=float, delimiter=',')
	response = np.genfromtxt(response_path, dtype=str, delimiter=',')
	aa = [i.split('_') for i in response]
	names = ['p' + str(aa[i][0]) + '_c' + str(aa[i][1]) for i in range(len(aa))]
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
	response[np.where(response == 6)] = 3
	scaler = StandardScaler().fit(data)
	data = scaler.transform(data)
	return data, response, names


# /*
# ██████  ██       █████  ██ ███    ██         ███████ ██    ██ ███    ███
# ██   ██ ██      ██   ██ ██ ████   ██         ██      ██    ██ ████  ████
# ██████  ██      ███████ ██ ██ ██  ██         ███████ ██    ██ ██ ████ ██
# ██      ██      ██   ██ ██ ██  ██ ██              ██  ██  ██  ██  ██  ██
# ██      ███████ ██   ██ ██ ██   ████ ███████ ███████   ████   ██      ██
# */
# kern = 'rbf', 'linear', 'poly', 'sigmoid'
@timeme
def plain_svm(data, response, kern='linear'):
	clf = svm.SVC(kernel=kern)
	scores = cross_validation.cross_val_score(clf, data, response, cv=10)
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
def make_confusion_matrix(data, response, kern='linear', percent_train=.25):
	X_train, X_test, y_train, y_test = train_test_split(data, response, train_size=percent_train)
	print kern
	classifier = svm.SVC(kernel=kern)  # KNeighborsClassifier(n_neighbors=1)   # svm.SVC(kernel=kern)
	y_pred = classifier.fit(X_train, y_train).predict(X_test)
	cm = confusion_matrix(y_test, y_pred)

	# code.interact(local=locals())

	print cm
	plt.clf()
	names = np.array(['Postnatel Day 2', 'Postnatel Day 3', 'Postnatel Day 4', 'Postnatel Day > 6'])
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Linear SVM at ' + str(percent_train * 100) + '% Population Trained (K=1)')
	plt.colorbar()
	tick_marks = np.arange(len(names))
	plt.xticks(tick_marks, names, rotation=45)
	plt.yticks(tick_marks, names)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	age = raw_input("Press l to save figure")
	if age == 'l':
		plt.savefig('svm.png', bbox_inches='tight')

	age = raw_input()
	if age == 'l':
		plt.savefig('svm.eps', format='eps', dpi=1200, bbox_inches='tight')

		plt.show()


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
def ap(data, response, names, show_graph=0):
	af = cluster.AffinityPropagation().fit(data)
	labels = af.labels_
	clusternames = defaultdict(list)
	cm = np.zeros([len(set(response)), len(set(labels))])
	for i, label in enumerate(labels):
		clusternames[label].append(names[i])
		cm[response[i], label] += 1
	# clusternames now holds a map from cluster label to list of sequence names
	for k, v in clusternames.items():
		print k, v
	print cm
	if show_graph:
		y_names = np.array(['Postnatel Day 2', 'Postnatel Day 3', 'Postnatel Day 4', 'Postnatel Day > 6'])
		plt.imshow(cm, interpolation='none', cmap=plt.cm.Blues)
		plt.title('AP Stock Parameters')
		# plt.colorbar()
		tick_marks = np.arange(len(names))
		plt.xticks(np.arange(max(labels) + 1), range(max(labels) + 1), rotation=45)
		plt.yticks(tick_marks, names)
		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted cluster label')
		plt.savefig('cm_ap_high.pdf', format='pdf', dpi=1200, bbox_inches='tight')
		plt.show()
	return clusternames


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
# ██████  ██████  ███████  ██████  █████  ███    ██
# ██   ██ ██   ██ ██      ██      ██   ██ ████   ██
# ██   ██ ██████  ███████ ██      ███████ ██ ██  ██
# ██   ██ ██   ██      ██ ██      ██   ██ ██  ██ ██
# ██████  ██████  ███████  ██████ ██   ██ ██   ████
# */
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
# ██   ██ ███    ███ ███████  █████  ███    ██ ███████
# ██  ██  ████  ████ ██      ██   ██ ████   ██ ██
# █████   ██ ████ ██ █████   ███████ ██ ██  ██ ███████
# ██  ██  ██  ██  ██ ██      ██   ██ ██  ██ ██      ██
# ██   ██ ██      ██ ███████ ██   ██ ██   ████ ███████
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
# ██   ██ ███    ██ ███    ██          ██████ ██       █████  ███████ ███████ ██ ███████ ██ ███████ ██████
# ██  ██  ████   ██ ████   ██         ██      ██      ██   ██ ██      ██      ██ ██      ██ ██      ██   ██
# █████   ██ ██  ██ ██ ██  ██         ██      ██      ███████ ███████ ███████ ██ █████   ██ █████   ██████
# ██  ██  ██  ██ ██ ██  ██ ██         ██      ██      ██   ██      ██      ██ ██ ██      ██ ██      ██   ██
# ██   ██ ██   ████ ██   ████ ███████  ██████ ███████ ██   ██ ███████ ███████ ██ ██      ██ ███████ ██   ██
# */
def knn_classifier(data, response):
	X_train, X_test, y_train, y_test = train_test_split(data, response)
	neigh = KNeighborsClassifier(n_neighbors=3)
	d = neigh.fit(X_train, y_train).score(X_test, y_test)
	print 'knn classifier accuracy: ' + str(d)


def radius_knn(data, response, rad):
	X_train, X_test, y_train, y_test = train_test_split(data, response)
	neigh = RadiusNeighborsClassifier(radius=rad)
	d = neigh.fit(X_train, y_train).score(X_test, y_test)
	print 'knn radius classifier accuracy: ' + str(d)


# /*
# ██████  ██    ██ ██ ██      ██████           ██████ ██      ██    ██ ███████ ████████ ███████ ██████          ██ ███    ███  ██████  ███████
# ██   ██ ██    ██ ██ ██      ██   ██         ██      ██      ██    ██ ██         ██    ██      ██   ██         ██ ████  ████ ██       ██
# ██████  ██    ██ ██ ██      ██   ██         ██      ██      ██    ██ ███████    ██    █████   ██████          ██ ██ ████ ██ ██   ███ ███████
# ██   ██ ██    ██ ██ ██      ██   ██         ██      ██      ██    ██      ██    ██    ██      ██   ██         ██ ██  ██  ██ ██    ██      ██
# ██████   ██████  ██ ███████ ██████  ███████  ██████ ███████  ██████  ███████    ██    ███████ ██   ██ ███████ ██ ██      ██  ██████  ███████
# */
def build_cluster_images(cm, names, path_to_images):
	failures = []
	counter = 0
	for row_number in cm:
		row = cm[row_number]
		counter += 1
		foldername = './clusterFolder/cluster' + str(counter)
		fname = 'cluster' + str(counter)
		try:
			os.mkdir(foldername)
		except:
			pass
		for each in row:
			try:
				shutil.copy2('./images/' + each, foldername + '/' + each)
			except:
				failures.append(each)

	print failures


# /*
# ████████ ███████ ███████ ████████
#    ██    ██      ██         ██
#    ██    █████   ███████    ██
#    ██    ██           ██    ██
#    ██    ███████ ███████    ██
# */
def test_suite():
	print("Starting Tests")
	print('Importing')
	# data, response, names = import_csv_data('./encodings.csv', './junk2.csv')
	# kfold(data, response, 2)
	# plain_svm(data, response, 'linear')
	# cm = make_confusion_matrix(data, response, 'linear', .33)

	# cm = ap(data, response, names)
	# build_cluster_images(cm, names, './images/')
	# cm = ap(data, response, names)
	# build_cluster_images(cm, names, './images/')
	# # loo(data, response, 'linear', 3)
	# # grid(data, response)
	# ap(data, response)
	# # mean_shift(data, response)
	# # dbscan(data, response)
	# kmeans(data, response)
	# knn_classifier(data, response)
	# radius_knn(data, response, 1000.0)
	# obj2spin('/home/mdm/Projects/cellseer/Fisher/objout/')
	# readPCL('./plc_spin_images/')
	# split_and_write_spin_images('./spin_images/', 5)
	# clean_spins('./spin_images/*')


# /*
#  ██████  ███████ ████████      ███████ ██ ██      ███████         ██      ███████ ███    ██
# ██       ██         ██         ██      ██ ██      ██              ██      ██      ████   ██
# ██   ███ █████      ██         █████   ██ ██      █████           ██      █████   ██ ██  ██
# ██    ██ ██         ██         ██      ██ ██      ██              ██      ██      ██  ██ ██
#  ██████  ███████    ██ ███████ ██      ██ ███████ ███████ ███████ ███████ ███████ ██   ████
# */
def get_file_len(fname):
	p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	result, err = p.communicate()
	if p.returncode != 0:
		raise IOError(err)
	return int(result.strip().split()[0])


# /*
# ███████ ██████  ██      ██ ████████       █████  ███    ██ ██████          ██     ██ ██████  ██ ████████ ███████
# ██      ██   ██ ██      ██    ██         ██   ██ ████   ██ ██   ██         ██     ██ ██   ██ ██    ██    ██
# ███████ ██████  ██      ██    ██         ███████ ██ ██  ██ ██   ██         ██  █  ██ ██████  ██    ██    █████
#      ██ ██      ██      ██    ██         ██   ██ ██  ██ ██ ██   ██         ██ ███ ██ ██   ██ ██    ██    ██
# ███████ ██      ███████ ██    ██ ███████ ██   ██ ██   ████ ██████  ███████  ███ ███  ██   ██ ██    ██    ███████
# */
def split_and_write_spin_images(path, percent_to_pick):
	list_of_spins = glob.glob(path + '*.pcdspinImages.bin')
	cnt = 0

	list_of_splits = glob.glob('./split_spin/*')
	list_of_splits = [os.path.basename(x) for x in list_of_splits]

	for each in list_of_spins:
		if os.path.basename(each) + '_vectorspicked.csv' in list_of_splits:
			continue
		print each
		print cnt
		# code.interact(local=locals())

	for each in list_of_spins:
		print each
		print cnt

		cnt += 1

		spin_file = open(each)
		spin_file_lines = spin_file.readlines()
		spin_file.close()

		spin_file_lines = [l.strip()[1:-1].split(',') for l in spin_file_lines]
		spin_file_lines = np.asarray(spin_file_lines)
		spin_file_lines = spin_file_lines.astype('float')
		xx, yy = spin_file_lines.shape

		xToPick = .05 * xx

		picks = np.random.choice(xx, xToPick, replace=False)

		# parse file name
		# age = each.split('/')[-1].split('_')[0][1:]
		# cell = each.split('/')[-1].split('_')[2].split('.')[1:]
		# cell = each.split('/')[-1].split('_')[2].split('.')[0][1:]

		name = os.path.basename(each)
		# print age, ' ', cell
		nonPick = np.setdiff1d(np.asarray(range(0, xx)), picks)
		if not os.path.exists('./split_spin/'):
			os.makedirs('./split_spin/')

		spin_file_lines_picked = spin_file_lines[picks]
		fileName = './split_spin/' + name + '_vectorspicked.csv'
		xToPick = 2000
		picks = np.random.choice(xx, xToPick, replace=False)

		# parse file name
		age = each.split('/')[-1].split('_')[0][1:]
		# cell = each.split('/')[-1].split('_')[2].split('.')[1:]
		cell = each.split('/')[-1].split('_')[2].split('.')[0][1:]
		print age, ' ', cell
		nonPick = np.setdiff1d(np.asarray(range(0, xx)), picks)
		if not os.path.exists('./picked/'):
			os.makedirs('./picked/')
		spin_file_lines_picked = spin_file_lines[picks]
		fileName = './picked1/' + age + '_' + cell + '_vectorspicked.csv'

		np.savetxt(fileName, spin_file_lines_picked, delimiter=',')
		spin_file_lines_picked = None

		spin_file_lines_notpicked = spin_file_lines[nonPick]

		fileName = './split_spin/' + name + '_vectorsnotpicked.csv'

		fileName = './picked1/' + age + '_' + cell + '_vectorsnotpicked.csv'

		np.savetxt(fileName, spin_file_lines_notpicked, delimiter=',')
		spin_file_lines_notpicked = None


# /*
# ██████  ██    ██ ██ ██      ██████          ███████ ██ ███████ ██   ██ ███████ ██████
# ██   ██ ██    ██ ██ ██      ██   ██         ██      ██ ██      ██   ██ ██      ██   ██
# ██████  ██    ██ ██ ██      ██   ██         █████   ██ ███████ ███████ █████   ██████
# ██   ██ ██    ██ ██ ██      ██   ██         ██      ██      ██ ██   ██ ██      ██   ██
# ██████   ██████  ██ ███████ ██████  ███████ ██      ██ ███████ ██   ██ ███████ ██   ██
# */
def matlab_make_fisher(path_to_split_spins, number_of_clusters):
	process = subprocess.Popen('sudo matlab -nodisplay -nodesktop -r run getAndSave(' + path_to_split_spins + ', ' + str(number_of_clusters) + ')')


# /*
# ███████ ██     ██  ██████ ██████   ██████  ██████       ██
# ██      ██     ██ ██           ██ ██    ██ ██   ██      ██
# ███████ ██  █  ██ ██       █████  ██    ██ ██████       ██
#      ██ ██ ███ ██ ██      ██      ██    ██ ██   ██ ██   ██
# ███████  ███ ███   ██████ ███████  ██████  ██████   █████
# */
@timeme
def swc2obj(path):
	print('Beginning swc2obj conversion')
	matlab_engine = matlab.engine.start_matlab()

	if not os.path.exists('./tempobj/'):
		os.makedirs('./tempobj/')
	else:
		shutil.rmtree('./tempobj/')
		os.makedirs('./tempobj/')
	if not os.path.exists('./convertedOBJs/'):
		os.makedirs('./convertedOBJs/')
	list_of_swcs = glob.glob(os.path.join(path, '*.swc'))
	list_of_objs = glob.glob('./objout/*.obj')

	print('Processing ' + str(len(list_of_swcs)) + ' SWCs')
	for each in list_of_swcs:
		if len(each) < 3:
			continue
		# code.interact(local=locals())
		if each + '.obj' in list_of_objs:
			print(each + ' OBJ already created, skipping')
			continue
		print(each)
		# process = subprocess.Popen('sudo matlab -nodisplay -nodesktop -r \"swc2obj(each);quit;\"')
		try:
			matlab_engine.swc2obj(each, nargout=0)
			flatten.main('tempobj/', each)
		except:
			continue


# /*
#  ██████  ██████       ██ ██████  ██████   ██████ ██████
# ██    ██ ██   ██      ██      ██ ██   ██ ██      ██   ██
# ██    ██ ██████       ██  █████  ██████  ██      ██   ██
# ██    ██ ██   ██ ██   ██ ██      ██      ██      ██   ██
#  ██████  ██████   █████  ███████ ██       ██████ ██████
# */
def obj2pcd(path):
	string = 'pcl_obj2pcd '
	list_of_objs = glob.glob(path + '*.obj')
	if not os.path.exists('./plc_spin_images/'):
		os.makedirs('./plc_spin_images/')
	for i in list_of_objs:
		print i
		string += i + ' ' + './plc_spin_images/' + os.path.basename(i)[0:-3] + 'pcd' + ' -copy_normals 1'
		print string
		os.system(string)
		string = 'pcl_obj2pcd '


# /*
# ██████   ██████ ██████  ██████  ███████ ██████  ██ ███    ██
# ██   ██ ██      ██   ██      ██ ██      ██   ██ ██ ████   ██
# ██████  ██      ██   ██  █████  ███████ ██████  ██ ██ ██  ██
# ██      ██      ██   ██ ██           ██ ██      ██ ██  ██ ██
# ██       ██████ ██████  ███████ ███████ ██      ██ ██   ████
# */
def pcd2spin(path):
	string = '/home/mdm/Projects/cellseer_build/pcd_spin_image '
	if not os.path.exists('./spin_images/'):
		os.makedirs('./spin_images/')
	list_of_spins = glob.glob(path + '*.bin')
	list_of_pcl = glob.glob(path + '*.pcd')
	compare_list_of_spins = [x[:-14] for x in list_of_spins]
	set_spins = set(compare_list_of_spins)
	set_pcl = set(list_of_pcl)
	set_to_do = set_pcl - set_spins
	sorted_files = sorted(set_to_do, key=os.path.getsize)
	print(str(len(set_to_do)) + 's PLCs to convert')
	commands_to_run = [string + x for x in sorted_files]
	pool = Pool(4)

	for i, returncode in enumerate(pool.imap(partial(call, shell=True), commands_to_run)):
		if returncode != 0:
			print("%d command failed: %d" % (i, returncode))
		else:
			print("%d command done! %d " % (i, returncode))
	# code.interact(local=locals())
	# for each in list_of_pcl:
	# 	if each not in list_of_spins:
	# 		code.interact(local=locals())
	# 		print each
	# 		string += each
	# 		print string
	# 		os.system(string)
	# 		string = '/home/mdm/Projects/cellseer_build/pcd_spin_image '


# def build_bow(path):
# /*
#  ██████ ██      ███████  █████  ███    ██ ███████ ██████  ██ ███    ██ ███████
# ██      ██      ██      ██   ██ ████   ██ ██      ██   ██ ██ ████   ██ ██
# ██      ██      █████   ███████ ██ ██  ██ ███████ ██████  ██ ██ ██  ██ ███████
# ██      ██      ██      ██   ██ ██  ██ ██      ██ ██      ██ ██  ██ ██      ██
#  ██████ ███████ ███████ ██   ██ ██   ████ ███████ ██      ██ ██   ████ ███████
# */
def clean_spins(path):
	l = glob.glob(path)
	for num, each in enumerate(l):
		print each
		print num
		print len(l)
		xx = np.genfromtxt(each, delimiter=',')
		xx = xx[:, 1:-1]
		np.savetxt('./cleaned_spin/' + os.path.basename(each), xx, delimiter=",")


def main(argv):
	print("Starting Tests")
	p = argv[1]
	print("Beginning to work on: " + p)
	#swc2obj(p)
	#obj2spin(p)
	#pcd2spin(p)
	clean_spin(p)
	# data, response, names = import_csv_data('./encodings.csv', './junk2.csv')
	# kfold(data, response, 2)
	# plain_svm(data, response, 'linear')
	# cm = make_confusion_matrix(data, response, 'linear', .33)

	# cm = ap(data, response, names)
	# build_cluster_images(cm, names, './images/')
	# cm = ap(data, response, names)
	# build_cluster_images(cm, names, './images/')
	# # loo(data, response, 'linear', 3)
	# # grid(data, response)
	# ap(data, response)
	# # mean_shift(data, response)
	# # dbscan(data, response)
	# kmeans(data, response)
	# knn_classifier(data, response)
	# radius_knn(data, response, 1000.0)
	# obj2spin('/home/mdm/Projects/cellseer/Fisher/objout/')
	# readPCL('./plc_spin_images/')
	# split_and_write_spin_images('./spin_images/', 5)
	# clean_spins('./spin_images/*')

if __name__ == "__main__":
	main(sys.argv)

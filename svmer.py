import cPickle as pickle
import numpy
import os
import sys
import svmutil

def main(path):
	
	label = []
	points = []
	for u in os.listdir(path): 
		if u[-2:] == 'WC':
			
			filePath = path+u
			WC = pickle.load(open(filePath, 'rb'))
			label.append(u[1])
			points.append(WC)
	label = [int(i) for i in label]
	
	prob = svmutil.svm_problem(label, points)
	param = svmutil.svm_parameter('-t 0 -c 4 -b 1')
	
	m = svmutil.svm_train(prob, param)
	svmutil.svm_save_model('n.model', m)
	
	p_label, p_acc, p_val = svmutil.svm_predict(label, points, m, '-b 1')
	
	return p_acc
			
if __name__ == "__main__":
	path = sys.argv[1] 
	main(path)			
			
			

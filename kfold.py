import cPickle as pickle
import numpy
import os
import sys
import svmutil

def main(path, k):
	
	prabs = []
	lns = []
	for kk in range(0,k-1):
		testLabel = []
		trainPoint = []
		trainLabel = []
		testPoint = []
		wcCount = 0
		for u in os.listdir(path): 
			if u[-2:] == 'WC':
				wcCount += 1
				filePath = path+u
				WC = pickle.load(open(filePath, 'rb'))
				if wcCount % k == 0 + kk:
					testLabel.append(int(u[1]))
					testPoint.append(WC)
					
				else:
					trainLabel.append(int(u[1]))
					trainPoint.append(WC)

		lns.append(len(testLabel))
		prob = svmutil.svm_problem(trainLabel, trainPoint)
		param = svmutil.svm_parameter('-t 0 -c 4 -b 1 -q')


		m = svmutil.svm_train(prob, param)
		svmutil.svm_save_model('n.model', m)
		p_label, p_acc, p_val = svmutil.svm_predict(testLabel, testPoint, m, '-b 1')
		prabs.append(p_acc[0])
	
	print prabs		
	print lns
	print sum(prabs)/float(len(prabs))
	print 'ba'
	return sum(prabs)/float(len(prabs))

if __name__ == "__main__":			
	path = sys.argv[1] 
	k = int(sys.argv[2])
	main(path, k)		

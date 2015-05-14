import kmeans
import centroidFinderP
import kfold
import sys

def main(path):
	d = [10, 50, 100, 200, 300, 400, 500, 1000, 2000]
	l = []
	for x in d:
		print 'Starting kmeans with ' + str(x) +' centroids...'
		kmeans.main(path, 400)
		print 'Finished kmeans, proceding with centroidFinder...'
		centroidFinderP.main(path, x)
		print 'Finished CF, running kfold10...'
		a = kfold.main(path, 10)
		l.append(a)
	print l
	



if __name__ == "__main__":	
	path = sys.argv[1] 
	
	main(path)



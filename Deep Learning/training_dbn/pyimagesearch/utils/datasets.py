# import the necessary packages
import cPickle

def load_cifar10(path):
	# load the CIFAR-10 dataset batch
	f = open(path, "rb")
	data = cPickle.load(f)
	f.close()

	# return the CIFAR-10 batch
	return data
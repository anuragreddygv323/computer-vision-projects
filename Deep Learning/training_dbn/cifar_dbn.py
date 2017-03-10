# USAGE
# python cifar_dbn.py --dataset cifar-10-batches-py

# import the necessary packages
from __future__ import print_function
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from pyimagesearch.utils import datasets
from nolearn.dbn import DBN
import numpy as np
import argparse
import glob

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the CIFAR-10 dataset")
args = vars(ap.parse_args())

# initialize the training data and labels lists
print("[INFO] loading training data...")
trainData = None
trainlabels = None

# loop over the CIFAR-10 training batches
for p in sorted(glob.glob("{}/data_batch_*".format(args["dataset"]))):
	# load the batch
	batch = datasets.load_cifar10(p)

	# if the training data or labels are None, initialize them
	if trainData is None or trainLabels is None:
		trainData = batch["data"]
		trainLabels = batch["labels"]

	# otherwise, stack the data and labels
	else:
		trainData = np.vstack([trainData, batch["data"]])
		trainLabels = np.hstack([trainLabels, batch["labels"]])

# give the training data a good shuffle and scale it to the range [0, 1]
(trainData, trainLabels) = shuffle(trainData, trainLabels)
trainData = trainData.astype("float") / 255.0

# load the testing data and also scale it to the range [0, 1]
print("[INFO] loading testing data...")
batch = datasets.load_cifar10("{}/test_batch".format(args["dataset"]))
(testData, testLabels) = (batch["data"], np.array(batch["labels"]))
testData = testData.astype("float") / 255.0

# train a Deep Belief Network with 3072 inputs (i.e., the flattened 32 x 32 x 3
# color images), 1,500 hidden units, and 10 output units (one for each of the
# possible output classifications)
dbn = DBN(
	[trainData.shape[1], 1500, 10],
	learn_rates=0.3,
	learn_rate_decays=0.9,
	epochs=10,
	verbose=1)
dbn.fit(trainData, trainLabels)

# compute predictions for the test data and show a classification report
print("[INFO] evaluating...")
predictions = dbn.predict(testData)
print(classification_report(testLabels, predictions))
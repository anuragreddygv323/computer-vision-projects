# import the necessary packages
from __future__ import print_function
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
import argparse
import cPickle
import h5py

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features-db", required=True,
	help="Path the features database")
ap.add_argument("-b", "--bovw-db", required=True,
	help="Path to where the bag-of-visual-words database")
ap.add_argument("-m", "--model", required=True,
	help="Path to the output classifier")
ap.add_argument("-t", "--training-size", type=float, default=0.75,
	help="Percentage of training data to be used")
args = vars(ap.parse_args())

# open the features and bag-of-visual-words databases
featuresDB = h5py.File(args["features_db"])
bovwDB = h5py.File(args["bovw_db"])

# construct the training and testing splits of the data
i = int(bovwDB["bovw"].shape[0] * args["training_size"])
(trainData, trainLabels) = (bovwDB["bovw"][:i], featuresDB["image_ids"][:i])
(testData, testLabels) = (bovwDB["bovw"][i:], featuresDB["image_ids"][i:])

# prepare the labels by removing the filename from the image ID, leaving
# us with just the class name
trainLabels = [l.split(":")[0] for l in trainLabels]
testLabels = [l.split(":")[0] for l in testLabels]

# define the grid of parameters to explore, then start the grid search where
# we evaluate a Linear SVM for each value of C
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LinearSVC(), params, cv=3)
model.fit(trainData, trainLabels)

# print the accuracy of the classifier on the validation data
print(model.score(testData, testLabels))

# close the databases
featuresDB.close()
bovwDB.close()

# dump the classifier to file
f = open(args["model"], "w")
f.write(cPickle.dumps(model))
f.close()
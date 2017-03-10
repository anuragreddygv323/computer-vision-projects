# USAGE
# python train.py --conf conf/flowers17.json

# import the necessary packages
from __future__ import print_function
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from pyimagesearch.utils import Conf
from pyimagesearch.utils import dataset
import numpy as np
import argparse
import cPickle
import h5py

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
args = vars(ap.parse_args())

# load the configuration and label encoder
conf = Conf(args["conf"])
le = cPickle.loads(open(conf["label_encoder_path"]).read())

# open the database and split the data into their respective training and
# testing splits
print("[INFO] gathering train/test splits...")
db = h5py.File(conf["features_path"])
split = int(db["image_ids"].shape[0] * conf["training_size"])
(trainData, trainLabels) = (db["features"][:split], db["image_ids"][:split])
(testData, testLabels) = (db["features"][split:], db["image_ids"][split:])

# use the label encoder to encode the training and testing labels
trainLabels = [le.transform(l.split(":")[0]) for l in trainLabels]
testLabels = [le.transform(l.split(":")[0]) for l in testLabels]

# define the grid of parameters to explore, then start the grid search where
# we evaluate a Linear SVM for each value of C
print("[INFO] tuning hyperparameters...")
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3)
model.fit(trainData, trainLabels)
print("[INFO] best hyperparameters: {}".format(model.best_params_))

# open the results file for writing and initialize the total number of accurate
# rank-1 and rank-5 predictions
print("[INFO] evaluating...")
f = open(conf["results_path"], "w")
rank1 = 0
rank5 = 0

# loop over the testing data
for (label, features) in zip(testLabels, testData):
	# predict the probability of each class label and grab the top-5 labels
	# (based on probabiltiy)
	preds = model.predict_proba(np.atleast_2d(features))[0]
	preds = np.argsort(preds)[::-1][:5]

	# if the correct label if the first entry in the predicted labels list,
	# increment the number of correct rank-1 predictions
	if label == preds[0]:
		rank1 += 1

	# if the correct label is in the top-5 predicted labels, then increment
	# the number of correct rank-5 predictions
	if label in preds:
		rank5 += 1

# convert the accuracies to percents and write them to file
rank1 = (rank1 / float(len(testLabels))) * 100
rank5 = (rank5 / float(len(testLabels))) * 100
f.write("rank-1: {:.2f}%\n".format(rank1))
f.write("rank-5: {:.2f}%\n\n".format(rank5))

# write the classification report to file and close the output file
predictions = model.predict(testData)
f.write("{}\n".format(classification_report(testLabels, predictions,
	target_names=le.classes_)))
f.close()

# dump classifier to file
print("[INFO] dumping classifier...")
f = open(conf["classifier_path"], "w")
f.write(cPickle.dumps(model))
f.close()

# close the database
db.close()
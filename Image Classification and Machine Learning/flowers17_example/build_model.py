# USAGE
# python build_model.py --conf conf/flowers17.json

# import the necessary packages
from __future__ import print_function
from collections import OrderedDict
from pyimagesearch.utils import Conf
import numpy as np
import argparse
import cPickle
import sh

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
args = vars(ap.parse_args())

# load the configuration file and initialize the dictionary of accuracies
conf = Conf(args["conf"])
accuracies = OrderedDict()

# construct and execute the command to index features from our image dataset
print("[INFO] indexing features...")
cmd = ("index_features.py --dataset {image_dataset} --features-db {features_db} "
	   "--kp-detector {kp_detector} --desc {descriptor}").format(image_dataset=conf["image_dataset"],
	features_db=conf["features_path"], kp_detector=conf["kp_detector"],
	descriptor=conf["descriptor"])
indexer = sh.Command("python")
indexer(cmd.split(" "))

# loop over the vocabulary sizes to investigate
for k in conf["vocab_sizes"]:
	# initialize the list of vocab accuracies
	vocabAccuracies = []

	# loop over the number of times to construct each vocabulary size
	for i in np.arange(0, conf["num_passes"]):
		# construct and execute the command to cluster the extracted features
		print("[INFO] clustering pass {}/{} for k={}".format(i + 1, conf["num_passes"], k))
		cmd = ("cluster_features.py --features-db {features_db} --codebook {vocab} --clusters {k} "
			   "--percentage {sample_size}").format(features_db=conf["features_path"],
			vocab=conf["vocab_path"], k=k, sample_size=conf["sample_size"])
		cluster = sh.Command("python")
		cluster(cmd.split(" "))

		# construct and execute the comand to extract bag of visual words
		# representations for each image in the dataset
		print("[INFO] extracting bovw represenations...")
		cmd = ("extract_bovw.py --features-db {features_db} --codebook {vocab} "
			   "--bovw-db {bovw}").format(features_db=conf["features_path"],
			vocab=conf["vocab_path"], bovw=conf["bovw_path"])
		bovw = sh.Command("python")
		bovw(cmd.split(" "))

		# construct and execute the command to train a classifier on the
		# bag of visual words representations and determine the accuracy
		print("[INFO] training and evaluating model...")
		cmd = ("train_model.py --features-db {features_db} --bovw-db {bovw} "
			   "--model {classifier_path}").format(features_db=conf["features_path"],
			bovw=conf["bovw_path"], classifier_path=conf["classifier_path"])
		train = sh.Command("python")
		accuracy = float(train(cmd.split(" ")).strip())
		vocabAccuracies.append(accuracy)

	# update the dictionary of accuracies
	accuracies[k] = np.mean(vocabAccuracies)

# dump the accuracies to file
f = open(conf["accuracies_path"], "w")
f.write(cPickle.dumps(accuracies))
f.close()
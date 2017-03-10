# USAGE
# python feature_extractor_demo.py --features-data output/ukbench/hadoop_output

# import the necessary packages
from __future__ import print_function
import numpy as np
import argparse
import imutils
import random
import glob
import json
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features-data", required=True, help="path to the features output directory")
ap.add_argument("-s", "--sample", type=int, default=10, help="# of features samples to use")
args = vars(ap.parse_args())

# grab the list of parts files
partsFiles = list(glob.glob(args["features_data"] + "/part-*"))

# loop over the randomly number of images to sample
for i in range(0, args["sample"]):
	# randomly sample one of the features files, then load the output file
	p = random.choice(partsFiles)
	d = open(p).read().strip().split("\n")

	# randomly sample a row and unpack it
	row = random.choice(d)
	(imageID, path, features) = row.strip().split("\t")
	features = json.loads(features)

	# load the image and resize it
	image = cv2.imread(path)
	image = imutils.resize(image, width=320)
	orig = image.copy()
	descs = []

	# loop over the keypoints and features
	for (kp, vec) in features:
		# draw the keypoint on the image, then update the list of descriptors
		cv2.circle(image, (kp[0], kp[1]), 3, (0, 255, 0), 2)
		descs.append(vec)

	# show the image
	print("[INFO] {}: {}".format(path, np.array(descs).shape))
	cv2.imshow("Original", orig)
	cv2.imshow("Keypoints", image)
	cv2.waitKey(0)
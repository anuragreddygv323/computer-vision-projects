# USAGE
# python train_recognizer.py --selfies output/faces --classifier output/classifier --sample-size 100

# import the necessary packages
from __future__ import print_function
from pyimagesearch.face_recognition import FaceRecognizer
from imutils import encodings
import numpy as np
import argparse
import random
import glob
import cv2

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--selfies", required=True, help="path to the selfies directory")
ap.add_argument("-c", "--classifier", required=True, help="path to the output classifier directory")
ap.add_argument("-n", "--sample-size", type=int, default=100, help="maximum sample size for each face")
args = vars(ap.parse_args())

# initialize the face recognizer and the list of labels
fr = FaceRecognizer(cv2.createLBPHFaceRecognizer(radius=1, neighbors=8, grid_x=8, grid_y=8))
labels = []

# loop over the input faces for training
for (i, path) in enumerate(glob.glob(args["selfies"] + "/*.txt")):
	# extract the person from the file name,
	name = path[path.rfind("/") + 1:].replace(".txt", "")
	print("[INFO] training on '{}'".format(name))

	# load the faces file, sample it, and initialize the list of faces
	sample = open(path).read().strip().split("\n")
	sample = random.sample(sample, min(len(sample), args["sample_size"]))
	faces = []

	# loop over the faces in the sample
	for face in sample:
		# decode the face and update the list of faces
		faces.append(encodings.base64_decode_image(face))

	# train the face detector on the faces and update the list of labels
	fr.train(faces, np.array([i] * len(faces)))
	labels.append(name)

# update the face recognizer to include the face name labels, then write the model to file
fr.setLabels(labels)
fr.save(args["classifier"])
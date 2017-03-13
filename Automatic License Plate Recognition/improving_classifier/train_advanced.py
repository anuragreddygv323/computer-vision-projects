# USAGE
# python train_advanced.py --samples output/examples --char-classifier output/adv_char.cpickle \
#	--digit-classifier output/adv_digit.cpickle

# import the necessary packages
from __future__ import print_function
from pyimagesearch.license_plate import LicensePlateDetector
from pyimagesearch.descriptors import BlockBinaryPixelSum
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cPickle
import random
import glob
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--samples", required=True,
	help="path to the training samples directory")
ap.add_argument("-c", "--char-classifier", required=True,
	help="path to the output character classifier")
ap.add_argument("-d", "--digit-classifier", required=True,
	help="path to the output digit classifier")
ap.add_argument("-m", "--min-samples", type=int, default=20,
	help="minimum # of samples per character")
args = vars(ap.parse_args())

# initialize the descriptor
blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)

# initialize the data and labels for the alphabet and digits
alphabetData = []
digitsData = []
alphabetLabels = []
digitsLabels = []

# loop over the sample character paths
for samplePath in sorted(glob.glob(args["samples"] + "/*")):
	# extract the sample name, grab all images in the sample path, and sample them
	sampleName = samplePath[samplePath.rfind("/") + 1:]
	imagePaths = list(paths.list_images(samplePath))
	imagePaths = random.sample(imagePaths, min(len(imagePaths), args["min_samples"]))

	# loop over all images in the sample path
	for imagePath in imagePaths:
		# load the character, convert it to grayscale, preprocess it, and describe it
		char = cv2.imread(imagePath)
		char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
		char = LicensePlateDetector.preprocessChar(char)
		features = desc.describe(char)

		# check to see if we are examining a digit
		if sampleName.isdigit():
			digitsData.append(features)
			digitsLabels.append(sampleName)

		# otherwise, we are examining an alphabetical character
		else:
			alphabetData.append(features)
			alphabetLabels.append(sampleName)

# train the character classifier
print("[INFO] fitting character model...")
charModel = LinearSVC(C=1.0, random_state=42)
charModel.fit(alphabetData, alphabetLabels)

# train the digit classifier
print("[INFO] fitting digit model...")
digitModel = LinearSVC(C=1.0, random_state=42)
digitModel.fit(digitsData, digitsLabels)

# dump the character classifer to file
print("[INFO] dumping character model...")
f = open(args["char_classifier"], "w")
f.write(cPickle.dumps(charModel))
f.close()

# dump the digit classifer to file
print("[INFO] dumping digit model...")
f = open(args["digit_classifier"], "w")
f.write(cPickle.dumps(digitModel))
f.close()
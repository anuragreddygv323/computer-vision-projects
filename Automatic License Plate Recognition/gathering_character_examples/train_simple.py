# USAGE
# python train_simple.py --fonts input/example_fonts --char-classifier output/simple_char.cpickle \
#	--digit-classifier output/simple_digit.cpickle

# import the necessary packages
from __future__ import print_function
from pyimagesearch.descriptors import BlockBinaryPixelSum
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cPickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--fonts", required=True, help="path to the fonts dataset")
ap.add_argument("-c", "--char-classifier", required=True,
	help="path to the output character classifier")
ap.add_argument("-d", "--digit-classifier", required=True,
	help="path to the output digit classifier")
args = vars(ap.parse_args())

# initialize characters string
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"

# initialize the data and labels for the alphabet and digits
alphabetData = []
digitsData = []
alphabetLabels = []
digitsLabels = []

# initialize the descriptor
print("[INFO] describing font examples...")
blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)

# loop over the font paths
for fontPath in paths.list_images(args["fonts"]):
	# load the font image, convert it to grayscale and threshold it
	font = cv2.imread(fontPath)
	font = cv2.cvtColor(font, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(font, 128, 255, cv2.THRESH_BINARY_INV)[1]

	# detect contours in the thresholded image and sort them from left to right
	(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key=lambda c:(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[1]))

	# loop over the contours
	for (i, c) in enumerate(cnts):
		# grab the bounding box for the contour, extract the ROI, and extract features
		(x, y, w, h) = cv2.boundingRect(c)
		roi = thresh[y:y + h, x:x + w]
		features = desc.describe(roi)

		# check to see if this is an alphabet character
		if i < 26:
			alphabetData.append(features)
			alphabetLabels.append(alphabet[i])

		# otherwise this is a digit
		else:
			digitsData.append(features)
			digitsLabels.append(alphabet[i])

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
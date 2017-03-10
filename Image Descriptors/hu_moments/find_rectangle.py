# USAGE
# python find_rectangle.py --dataset output

# import the necessary packages
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import argparse
import glob
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the dataset directory")
args = vars(ap.parse_args())

# grab the image paths from disk and initialize the data matrix
imagePaths = sorted(glob.glob(args["dataset"] + "/*.jpg"))
data = []

# loop over the images in the dataset directory
for imagePath in imagePaths:
	# load the image, convert it to grayscale, and threshold it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)[1]

	# find contours in the image, keeping only the largest one
	(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	c = max(cnts, key=cv2.contourArea)

	# extract the ROI from the image, resize it to a canonical size,
	# compute the Hu Moments feature vector for the ROI, and update
	# the data matrix
	(x, y, w, h) = cv2.boundingRect(c)
	roi = cv2.resize(thresh[y:y + h, x:x + w], (50, 50))
	moments = cv2.HuMoments(cv2.moments(roi)).flatten()
	data.append(moments)

# compute the distance between all entries in the data matrix, then
# take the sum of the distances for each row, followed by grabbing
# the row with the largest distance
D = pairwise_distances(data).sum(axis=1)
i = np.argmax(D)

# display the outlier image
image = cv2.imread(imagePaths[i])
print "Found square: {}".format(imagePaths[i])
cv2.imshow("Outlier", image)
cv2.waitKey(0)
# USAGE
# python search_shirts.py --dataset shirts --query queries/query_01.jpg

# import the necessary packages
from __future__ import print_function
from pyimagesearch import LocalBinaryPatterns
from imutils import paths
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to the dataset of shirt images")
ap.add_argument("-q", "--query", required=True, help="path to the query image")
args = vars(ap.parse_args())

# initialize the local binary patterns descriptor and initialize the index dictionary
# where the image filename is the key and the features are the value
desc = LocalBinaryPatterns(24, 8)
index = {}

# loop over the shirt images
for imagePath in paths.list_images(args["dataset"]):
	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)

	# update the index dictionary
	filename = imagePath[imagePath.rfind("/") + 1:]
	index[filename] = hist

# load the query image and extract Local Binary Patterns from it
query = cv2.imread(args["query"])
queryFeatures = desc.describe(cv2.cvtColor(query, cv2.COLOR_BGR2GRAY))

# show the query image and initialize the results dictionary
cv2.imshow("Query", query)
results = {}

# loop over the index
for (k, features) in index.items():
	# compute the chi-squared distance between the current features and the query
	# features, then update the dictionary of results
	d = 0.5 * np.sum(((features - queryFeatures) ** 2) / (features + queryFeatures + 1e-10))
	results[k] = d

# sort the results
results = sorted([(v, k) for (k, v) in results.items()])[:3]

# loop over the results
for (i, (score, filename)) in enumerate(results):
	# show the result image
	print("#%d. %s: %.4f" % (i + 1, filename, score))
	image = cv2.imread(args["dataset"] + "/" + filename)
	cv2.imshow("Result #{}".format(i + 1), image)
	cv2.waitKey(0)
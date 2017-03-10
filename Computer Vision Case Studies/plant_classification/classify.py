# USAGE
# python classify.py --images dataset/images --masks dataset/masks

# import the necessary packages
from __future__ import print_function
from pyimagesearch.descriptors import RGBHistogram
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to the image dataset")
ap.add_argument("-m", "--masks", required=True, help="path to the image masks")
args = vars(ap.parse_args())

# grab the image and mask paths
imagePaths = sorted(list(paths.list_images(args["images"])))
maskPaths = sorted(list(paths.list_images(args["masks"])))

# initialize the list of data and class label targets
data = []
target = []

# initialize the image descriptor
desc = RGBHistogram([8, 8, 8])

# loop over the image and mask paths
for (imagePath, maskPath) in zip(imagePaths, maskPaths):
	# load the image and mask
	image = cv2.imread(imagePath)
	mask = cv2.imread(maskPath)
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

	# describe the image
	features = desc.describe(image, mask)

	# update the list of data and targets
	data.append(features)
	target.append(imagePath.split("_")[-2])

# grab the unique target names and encode the labels
targetNames = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)

# construct the training and testing splits
(trainData, testData, trainTarget, testTarget) = train_test_split(data, target,
	test_size=0.3, random_state=42)

# train the classifier
model = RandomForestClassifier(n_estimators=25, random_state=84)
model.fit(trainData, trainTarget)

# evaluate the classifier
print(classification_report(testTarget, model.predict(testData),
	target_names = targetNames))

# loop over a sample of the images
for i in np.random.choice(np.arange(0, len(imagePaths)), 10):
	# grab the image and mask paths
	imagePath = imagePaths[i]
	maskPath = maskPaths[i]

	# load the image and mask
	image = cv2.imread(imagePath)
	mask = cv2.imread(maskPath)
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

	# describe the image
	features = desc.describe(image, mask)

	# predict what type of flower the image is
	flower = le.inverse_transform(model.predict(features.reshape(1, -1)))[0]
	print("[INFO] prediction: {}, path: {}".format(flower.upper(), imagePath))
	cv2.imshow("image", image)
	cv2.waitKey(0)
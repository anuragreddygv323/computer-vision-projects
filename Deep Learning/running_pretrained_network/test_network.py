# USAGE
# python test_network.py --model output/cifar10_shallownet.hdf5 --test-images test_images

# import the necessary packages
from __future__ import print_function
from keras.models import load_model
from keras.datasets import cifar10
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to output model file")
ap.add_argument("-t", "--test-images", required=True,
	help="path to the directory of testing images")
ap.add_argument("-b", "--batch-size", type=int, default=32,
	help="size of mini-batches passed to network")
args = vars(ap.parse_args())

# initialize the ground-truth labels for the CIFAR-10 dataset
gtLabels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
	"ship", "truck"]

# load the network
print("[INFO] loading network architecture and weights...")
model = load_model(args["model"])

# randomly select a few testing examples from the CIFAR-10 dataset and then
# scale the data points into the range [0, 1]
print("[INFO] sampling CIFAR-10...")
(testData, testLabels) = cifar10.load_data()[1]
testData = testData.astype("float") / 255.0
np.random.seed(42)
idxs = np.random.choice(testData.shape[0], size=(15,), replace=False)
(testData, testLabels) = (testData[idxs], testLabels[idxs])
testLabels = testLabels.flatten()

# make predictions on the sample of testing data
print("[INFO] predicting on testing data...")
probs = model.predict(testData, batch_size=args["batch_size"])
predictions = probs.argmax(axis=1)

# loop over each of the testing data points
for (i, prediction) in enumerate(predictions):
	# convert the image from shape (3, 32, 32) to (32, 32, 3) and then resize
	# it to a larger size
	(R, G, B) = testData[i]
	image = np.dstack([B, G, R])
	image = imutils.resize(image, width=128, inter=cv2.INTER_CUBIC)
	
	# show the image along with the predicted label
	print("[INFO] predicted: {}, actual: {}".format(gtLabels[prediction],
		gtLabels[testLabels[i]]))
	cv2.imshow("Image", image)
	cv2.waitKey(0)

# close all open windows in preparation for the images not part of the CIFAR-10
# dataset
cv2.destroyAllWindows()
print("[INFO] testing on images NOT part of CIFAR-10")

# loop over the images not part of the CIFAR-10 dataset
for imagePath in paths.list_images(args["test_images"]):
	# load the image, resize it to a fixed 32 x 32 pixels (ignoring aspect ratio),
	# and then conver the shape from (32, 32, 3) to (3, 32, 32) using RGB order
	# to make it compatible with our network
	print("[INFO] classifying {}".format(imagePath[imagePath.rfind("/") + 1:]))
	image = cv2.imread(imagePath)
	(B, G, R) = cv2.split(cv2.resize(image, (32, 32)))
	kerasImage = np.array([R, G, B], dtype="float") / 255.0

	# add an extra dimension to the image so we can pass it through the network,
	# then make a prediction on the image (normally we would make predictions on
	# an *array* of images instead one at a time)
	kerasImage = kerasImage[np.newaxis, ...]
	probs = model.predict(kerasImage, batch_size=args["batch_size"])
	prediction = probs.argmax(axis=1)[0]

	# draw the prediction on the test image and display it
	cv2.putText(image, gtLabels[prediction], (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 255, 0), 3)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
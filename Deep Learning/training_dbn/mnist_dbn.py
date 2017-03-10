# USAGE
# python mnist_dbn.py

# import the necessary packages
from __future__ import print_function
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np
import cv2

# grab the MNIST dataset (if this is your first time running this script, the
# download make take a minute -- the 55mb MNIST dataset will be downloaded)
print("[INFO] downloading MNIST...")
dataset = datasets.fetch_mldata("MNIST Original")

# scale the data to the range [0, 1] and then construct the training and testing
# splits
(trainData, testData, trainLabels, testLabels) = train_test_split(
	dataset.data / 255.0, dataset.target.astype("int"), test_size=0.33)

# train the Deep Belief Network with 764 input units (i.e., the flattened,
# 28 x 28 grayscale images), 300 hidden units, and 10 output units (one for
# each of the possible output classifications)
dbn = DBN(
	[trainData.shape[1], 300, 10],
	learn_rates=0.3,
	learn_rate_decays=0.9,
	epochs=10,
	verbose=1)
dbn.fit(trainData, trainLabels)

# compute the predictions for the test data and show a classification report
print("[INFO] evaluating...")
predictions = dbn.predict(testData)
print(classification_report(testLabels, predictions))

# randomly select a few digits
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
	# classify the digit
	prediction = dbn.predict(np.atleast_2d(testData[i]))

	# reshape the feature vector to be a 28 x 28 pixel image, then resize it
	# so we can visualize it better
	image = (testData[i] * 255).reshape((28, 28)).astype("uint8")
	image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)

	# show the image and prediction
	print("[INFO] Predicted: {}, Actual: {}".format(prediction[0], testLabels[i]))
	cv2.imshow("Digit", image)
	cv2.waitKey(0)
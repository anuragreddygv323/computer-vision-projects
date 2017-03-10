# USAGE
# python perceptron.py

# import the necessary packages
from __future__ import print_function
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron
from sklearn import datasets

# load the Iris dataset and split it into training and testing data
print("[INFO] loading data...")
iris = datasets.load_iris()
(trainData, testData, trainLabels, testLabels) = train_test_split(iris.data, iris.target,
	test_size=0.25, random_state=42)

# train the Perceptron
print("[INFO] training...")
model = Perceptron(n_iter=10, eta0=1.0, random_state=84)
model.fit(trainData, trainLabels)

# evaluate the Perceptron
print("[INFO] evaluating...")
predictions = model.predict(testData)
print(classification_report(predictions, testLabels, target_names=iris.target_names))
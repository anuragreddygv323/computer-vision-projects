# import the necessary packages
from collections import namedtuple
import cPickle
import cv2
import os

# define the face recognizer instance
FaceRecognizerInstance = namedtuple("FaceRecognizerInstance", ["trained", "labels"])

class FaceRecognizer:
	def __init__(self, recognizer, trained=False, labels=None):
		# store the face recognizer, whether or not the face recognizer has already been trained,
		# and the list of face name labels
		self.recognizer = recognizer
		self.trained = trained
		self.labels = labels

	def setLabels(self, labels):
		# store the face name labels
		self.labels = labels

	def setConfidenceThreshold(self, confidenceThreshold):
		# set the confidence threshold for the classifier
		self.recognizer.setDouble("threshold", confidenceThreshold)

	def train(self, data, labels):
		# if the model has not been trained, train it
		if not self.trained:
			self.recognizer.train(data, labels)
			self.trained = True
			return

		# otherwise, update the model
		self.recognizer.update(data, labels)

	def predict(self, face):
		# predict the face
		(prediction, confidence) = self.recognizer.predict(face)

		# if the prediction is `-1`, then the confidence is greater than the threshold, implying
		# that the face cannot be recognized
		if prediction == -1:
			return ("Unknown", 0)

		# return a tuple of the face label and the confidence
		return (self.labels[prediction], confidence)

	def save(self, basePath):
		# construct the face recognizer instance
		fri = FaceRecognizerInstance(trained=self.trained, labels=self.labels)

		# due to strange behavior with OpenCV, we need to make sure the output classifier file
		# exists prior to writing it to file
		if not os.path.exists(basePath + "/classifier.model"):
			f = open(basePath + "/classifier.model", "w")
			f.close()

		# write the actual recognizer along with the parameters to file
		self.recognizer.save(basePath + "/classifier.model")
		f = open(basePath + "/fr.cpickle", "w")
		f.write(cPickle.dumps(fri))
		f.close()

	@staticmethod
	def load(basePath):
		# load the face recognition instance and construct the OpenCV face recognizer
		fri = cPickle.loads(open(basePath + "/fr.cpickle").read())
		recognizer = cv2.createLBPHFaceRecognizer()
		recognizer.load(basePath + "/classifier.model")

		# construct and return the face recognizer
		return FaceRecognizer(recognizer, trained=fri.trained, labels=fri.labels)
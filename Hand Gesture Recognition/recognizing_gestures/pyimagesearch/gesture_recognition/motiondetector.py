# import the necessary packages
import cv2

class MotionDetector:
	def __init__(self, accumWeight=0.5):
		# store the accumulated weight factor
		self.accumWeight = accumWeight

		# initialize the background model
		self.bg = None

	def update(self, image):
		# if the background model is None, initialize it
		if self.bg is None:
			self.bg = image.copy().astype("float")
			return

		# update the background model by accumulating the weighted average
		cv2.accumulateWeighted(image, self.bg, self.accumWeight)

	def detect(self, image, tVal=25):
		# compute the absolute difference between the background model and the image
		# passed in, then threshold the delta image
		delta = cv2.absdiff(self.bg.astype("uint8"), image)
		thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]

		# find contours in the thresholded image
		(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# if no contours were found, return None
		if len(cnts) == 0:
			return None

		# otherwise, return a tuple of the thresholded image along with the contour area
		return (thresh, max(cnts, key=cv2.contourArea))
# import the necessary packages
from sklearn.metrics import pairwise
import numpy as np
import cv2

class GestureDetector:
	def __init__(self):
		pass

	def detect(self, thresh, cnt):
		# compute the convex hull of the contour and then find the extreme most points along
		# the hull
		hull = cv2.convexHull(cnt)
		extLeft = tuple(hull[hull[:, :, 0].argmin()][0])
		extRight = tuple(hull[hull[:, :, 0].argmax()][0])
		extTop = tuple(hull[hull[:, :, 1].argmin()][0])
		extBot = tuple(hull[hull[:, :, 1].argmax()][0])

		# compute the center (x, y)-coordinates based on the extreme points, then add a little
		# extra percentage to the y-coordinate to lower the region towards the center of the
		# palm
		cX = (extLeft[0] + extRight[0]) / 2
		cY = (extTop[1] + extBot[1]) / 2
		cY += (cY * 0.15)
		cY = int(cY)

		# compute the distances between the centroid and the extreme coordinates, then find the
		# largest distance, and use it to complete radius of palm region
		D = pairwise.euclidean_distances([(cX, cY)], Y=[extLeft, extRight, extTop, extBot])[0]
		maxDist = D[D.argmax()]
		r = int(0.7 * maxDist)
		circum = 2 * np.pi * r

		# construct the circular ROI that includes the palm + fingers
		circleROI = np.zeros(thresh.shape[:2], dtype="uint8")
		cv2.circle(circleROI, (cX, cY), r, 255, 1)
		circleROI = cv2.bitwise_and(thresh, thresh, mask=circleROI)

		# find contours in the circular ROI and initialize the total number of fingers counted
		# in the frame
		(cnts, _) = cv2.findContours(circleROI.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		total = 0

		# loop over the contours
		for c in cnts:
			# compute the bounding box of the contour
			(x, y, w, h) = cv2.boundingRect(c)

			# increment the total number of fingers only if (1) the number of points along the
			# contour does not exceed 25% of the circumfrence and (2) the contour region is not
			# at the bottom of the circle (which is the wrist area)
			if c.shape[0] < circum * 0.25 and (y + h) < cY + (cY * 0.25):
				total += 1

		# return the total number of fingers detected
		return total

	@staticmethod
	def drawText(roi, i, val, color=(0, 0, 255)):
		# draw the text on the ROI
		cv2.putText(roi, str(val), ((i * 50) + 20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
			color, 3)

	@staticmethod
	def drawBox(roi, i, color=(0, 0, 255)):
		# draw the box on the ROI
		cv2.rectangle(roi, ((i * 50) + 10, 10), ((i * 50) + 50, 60), color, 2)
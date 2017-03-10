# import the necessary packages
import cv2

class DistanceFinder:
	def __init__(self, knownWidth, knownDistance):
		# store the known width of the object (in inches, meters, etc.) and known distance
		# to the object (again, in inches, meters, etc.)
		self.knownWidth = knownWidth
		self.knownDistance = knownDistance

		# initialize the focal length
		self.focalLength = 0

	def calibrate(self, width):
		# compute and store the focal length for calibration
		self.focalLength = (width * self.knownDistance) / self.knownWidth

	def distance(self, perceivedWidth):
		# compute and return the distance from the marker to the camera
		return (self.knownWidth * self.focalLength) / perceivedWidth

	@staticmethod
	def findSquareMarker(image):
		# convert the image to grayscale, blur it, and find edges in the image
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (5, 5), 0)
		edged = cv2.Canny(gray, 35, 125)

		# find contours in the edged image, sort them according to their area (from largest to
		# smallest), and initialize the marker dimensions
		(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
		markerDim = None

		# loop over the contours
		for c in cnts:
			# approximate the contour
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)

			# ensure that the contour is a rectangle
			if len(approx) == 4:
				# compute the bounding box and aspect ratio of the approximated contour
				(x, y, w, h) = cv2.boundingRect(approx)
				aspectRatio = w / float(h)

				# check to see if the aspect ratio is within tolerable bounds; if so, store
				# the marker dimensions and break from the loop
				if aspectRatio > 0.9 and aspectRatio < 1.1:
					markerDim = (x, y, w, h)
					break

		# return the marker dimensions
		return markerDim

	@staticmethod
	def draw(image, boundingBox, dist, color=(0, 255, 0), thickness=2):
		# draw a bounding box around the marker and display the distance to the marker on the
		# image
		(x, y, w, h) = boundingBox
		cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
		cv2.putText(image, "%.2fft" % (dist / 12), (image.shape[1] - 200, image.shape[0] - 20),
			cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)

		# return the image
		return image
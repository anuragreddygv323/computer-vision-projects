# import the necessary packages
from collections import namedtuple
from skimage.filters import threshold_adaptive
from skimage import segmentation
from skimage import measure
from imutils import perspective
import numpy as np
import imutils
import cv2

# define the named tupled to store the license plate
LicensePlate = namedtuple("LicensePlateRegion", ["success", "plate", "thresh", "candidates"])

class LicensePlateDetector:
	def __init__(self, image, minPlateW=60, minPlateH=20, numChars=7, minCharW=40):
		# store the image to detect license plates in, the minimum width and height of the
		# license plate region, the number of characters to be detected in the license plate,
		# and the minimum width of the extracted characters
		self.image = image
		self.minPlateW = minPlateW
		self.minPlateH = minPlateH
		self.numChars = numChars
		self.minCharW = minCharW

	def detect(self):
		# detect license plate regions in the image
		lpRegions = self.detectPlates()

		# loop over the license plate regions
		for lpRegion in lpRegions:
			# detect character candidates in the current license plate region
			lp = self.detectCharacterCandidates(lpRegion)

			# only continue if characters were successfully detected
			if lp.success:
				# scissor the candidates into characters
				chars = self.scissor(lp)

				# yield a tuple of the license plate region and the characters
				yield (lpRegion, chars)

	def detectPlates(self):
		# initialize the rectangular and square kernels to be applied to the image,
		# then initialize the list of license plate regions
		rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
		squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		regions = []

		# convert the image to grayscale, and apply the blackhat operation
		gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

		# find regions in the image that are light
		light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKernel)
		light = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY)[1]

		# compute the Scharr gradient representation of the blackhat image in the x-direction,
		# and scale the resulting image into the range [0, 255]
		gradX = cv2.Sobel(blackhat, ddepth=cv2.cv.CV_32F, dx=1, dy=0, ksize=-1)
		gradX = np.absolute(gradX)
		(minVal, maxVal) = (np.min(gradX), np.max(gradX))
		gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

		# blur the gradient representation, apply a closing operating, and threshold the
		# image using Otsu's method
		gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
		gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
		thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

		# perform a series of erosions and dilations on the image
		thresh = cv2.erode(thresh, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=2)

		# take the bitwise 'and' between the 'light' regions of the image, then perform
		# another series of erosions and dilations
		thresh = cv2.bitwise_and(thresh, thresh, mask=light)
		thresh = cv2.dilate(thresh, None, iterations=2)
		thresh = cv2.erode(thresh, None, iterations=1)

		# find contours in the thresholded image
		(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# loop over the contours
		for c in cnts:
			# grab the bounding box associated with the contour and compute the area and
			# aspect ratio
			(w, h) = cv2.boundingRect(c)[2:]
			aspectRatio = w / float(h)

			# compute the rotated bounding box of the region
			rect = cv2.minAreaRect(c)
			box = np.int0(cv2.cv.BoxPoints(rect))

			# ensure the aspect ratio, width, and height of the bounding box fall within
			# tolerable limits, then update the list of license plate regions
			if (aspectRatio > 3 and aspectRatio < 6) and h > self.minPlateH and w > self.minPlateW:
				regions.append(box)

		# return the list of license plate regions
		return regions

	def detectCharacterCandidates(self, region):
		# apply a 4-point transform to extract the license plate
		plate = perspective.four_point_transform(self.image, region)
		cv2.imshow("Perspective Transform", imutils.resize(plate, width=400))

		# extract the Value component from the HSV color space and apply adaptive thresholding
		# to reveal the characters on the license plate
		V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
		thresh = threshold_adaptive(V, 29, offset=15).astype("uint8") * 255
		thresh = cv2.bitwise_not(thresh)

		# resize the license plate region to a canonical size
		plate = imutils.resize(plate, width=400)
		thresh = imutils.resize(thresh, width=400)
		cv2.imshow("LP Threshold", thresh)

		# perform a connected components analysis and initialize the mask to store the locations
		# of the character candidates
		labels = measure.label(thresh, neighbors=8, background=0)
		charCandidates = np.zeros(thresh.shape, dtype="uint8")

		# loop over the unique components
		for label in np.unique(labels):
			# if this is the background label, ignore it
			if label == 0:
				continue

			# otherwise, construct the label mask to display only connected components for the
			# current label, then find contours in the label mask
			labelMask = np.zeros(thresh.shape, dtype="uint8")
			labelMask[labels == label] = 255
			(cnts, _) = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			# ensure at least one contour was found in the mask
			if len(cnts) > 0:
				# grab the largest contour which corresponds to the component in the mask, then
				# grab the bounding box for the contour
				c = max(cnts, key=cv2.contourArea)
				(boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

				# compute the aspect ratio, solidity, and height ratio for the component
				aspectRatio = boxW / float(boxH)
				solidity = cv2.contourArea(c) / float(boxW * boxH)
				heightRatio = boxH / float(plate.shape[0])

				# determine if the aspect ratio, solidity, and height of the contour pass
				# the rules tests
				keepAspectRatio = aspectRatio < 1.0
				keepSolidity = solidity > 0.15
				keepHeight = heightRatio > 0.4 and heightRatio < 0.95

				# check to see if the component passes all the tests
				if keepAspectRatio and keepSolidity and keepHeight:
					# compute the convex hull of the contour and draw it on the character
					# candidates mask
					hull = cv2.convexHull(c)
					cv2.drawContours(charCandidates, [hull], -1, 255, -1)

		# clear pixels that touch the borders of the character candidates mask and detect
		# contours in the candidates mask
		charCandidates = segmentation.clear_border(charCandidates)
		(cnts, _) = cv2.findContours(charCandidates.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cv2.imshow("Original Candidates", charCandidates)

		# if there are more character candidates than the supplied number, then prune
		# the candidates
		if len(cnts) > self.numChars:
			(charCandidates, cnts) = self.pruneCandidates(charCandidates, cnts)
			cv2.imshow("Pruned Canidates", charCandidates)

		# take bitwise AND of raw thresholded image and character candidates to get a more
		# clean segmentation of the characters
		thresh = cv2.bitwise_and(thresh, thresh, mask=charCandidates)
		cv2.imshow("Char Threshold", thresh)

		# return the license plate region object containing the license plate, the thresholded
		# license plate, and the character candidates
		return LicensePlate(success=True, plate=plate, thresh=thresh,
			candidates=charCandidates)

	def pruneCandidates(self, charCandidates, cnts):
		# initialize the pruned candidates mask and the list of dimensions
		prunedCandidates = np.zeros(charCandidates.shape, dtype="uint8")
		dims = []

		# loop over the contours
		for c in cnts:
			# compute the bounding box for the contour and update the list of dimensions
			(boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
			dims.append(boxY + boxH)

		# convert the dimensions into a NumPy array and initialize the list of differences
		# and selected contours
		dims = np.array(dims)
		diffs = []
		selected = []

		# loop over the dimensions
		for i in xrange(0, len(dims)):
			# compute the sum of differences between the current dimension and and all other
			# dimensions, then update the differences list
			diffs.append(np.absolute(dims - dims[i]).sum())

		# find the top number of candidates with the most similar dimensions and loop over
		# the selected contours
		for i in np.argsort(diffs)[:self.numChars]:
			# draw the contour on the pruned candidates mask and add it to the list of selected
			# contours
			cv2.drawContours(prunedCandidates, [cnts[i]], -1, 255, -1)
			selected.append(cnts[i])

		# return a tuple of the pruned candidates mask and selected contours
		return (prunedCandidates, selected)

	def scissor(self, lp):
		# detect contours in the candidates and initialize the list of bounding boxes and
		# list of extracted characters
		(cnts, _) = cv2.findContours(lp.candidates.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		boxes = []
		chars = []

		# loop over the contours
		for c in cnts:
			# compute the bounding box for the contour while maintaining the minimum width
			(boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
			dX = min(self.minCharW, self.minCharW - boxW) / 2
			boxX -= dX
			boxW += (dX * 2)

			# update the list of bounding boxes
			boxes.append((boxX, boxY, boxX + boxW, boxY + boxH))

		# sort the bounding boxes from left to right
		boxes = sorted(boxes, key=lambda b:b[0])

		# loop over the started bounding boxes
		for (startX, startY, endX, endY) in boxes:
			# extract the ROI from the thresholded license plate and update the characters
			# list
			chars.append(lp.thresh[startY:endY, startX:endX])

		# return the list of characters
		return chars

	@staticmethod
	def preprocessChar(char):
		# find the largest contour in the character, grab its bounding box, and crop it
		(cnts, _) = cv2.findContours(char.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		c = max(cnts, key=cv2.contourArea)
		(x, y, w, h) = cv2.boundingRect(c)
		char = char[y:y + h, x:x + w]

		# return the processed character
		return char
# import the necessary packages
import numpy as np
import cv2

class BlockBinaryPixelSum:
	def __init__(self, targetSize=(30, 15), blockSizes=((5, 5),)):
		# store the target size of the image to be described along with the set of block sizes
		self.targetSize = targetSize
		self.blockSizes = blockSizes

	def describe(self, image):
		# resize the image to the target size and initialize the feature vector
		image = cv2.resize(image, (self.targetSize[1], self.targetSize[0]))
		features = []

		# loop over the block sizes
		for (blockW, blockH) in self.blockSizes:
			# loop over the image for the current block size
			for y in xrange(0, image.shape[0], blockH):
				for x in xrange(0, image.shape[1], blockW):
					# extract the current ROI, count the total number of non-zero pixels in the
					# ROI, normalizing by the total size of the block
					roi = image[y:y + blockH, x:x + blockW]
					total = cv2.countNonZero(roi) / float(roi.shape[0] * roi.shape[1])

					# update the feature vector
					features.append(total)

		# return the features
		return np.array(features)
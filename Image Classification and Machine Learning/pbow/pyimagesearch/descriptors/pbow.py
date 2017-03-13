# import the necessary packages
from scipy import sparse
import numpy as np

class PBOW:
	def __init__(self, bovw, numLevels=2):
		# store the bag of visual words used to build a histogram along with the number
		# of pyramid levels to generate
		self.bovw = bovw
		self.numLevels = numLevels

	def describe(self, imageWidth, imageHeight, kps, features):
		# initialize the keypoint mask and concatenated visual words histogram
		kpMask = np.zeros((imageHeight, imageWidth), dtype="int")
		concatHist = None

		# loop over the keypoints and assign a unique ID to each (x, y)-coordinate in the
		# keypoint mask
		for (i, (x, y)) in enumerate(kps):
			kpMask[y, x] = i + 1

		# loop over the number of levels
		for level in np.arange(self.numLevels, -1, -1):
			# compute the number of partitions and the weight for the current level then
			numParts = 2 ** level
			weight = 1.0 / (2 ** (self.numLevels - level + 1))

			# if this is the first level, then adjust the weight
			if level == 0:
				weight = 1.0 / (2 ** self.numLevels)

			# determine the partitions for both the x and y direction
			X = np.linspace(imageWidth / numParts, imageWidth, numParts)
			Y = np.linspace(imageHeight / numParts, imageHeight, numParts)
			xParts = np.hstack([[0], X]).astype("int")
			yParts = np.hstack([[0], Y]).astype("int")

			# loop over the partitions
			for i in np.arange(1, len(xParts)):
				for j in np.arange(1, len(yParts)):
					# determine the window coordinates, then extract the indexes of the
					# keypoints residing in the current window
					(startX, endX) = (xParts[i - 1], xParts[i])
					(startY, endY) = (yParts[j - 1], yParts[j])
					idxs = np.unique(kpMask[startY:endY, startX:endX])[1:] - 1
					hist = sparse.csr_matrix((1, self.bovw.codebook.shape[0]), dtype="float")

					# ensure at least some features exist inside the current sub-region
					if len(features[idxs]) > 0:
						# quantize the current set of features, then L1-normalize the
						# histogram and multiply by the weight of the current level
						hist = self.bovw.describe(features[idxs])
						hist = weight * (hist / hist.sum())

					# if the concatenated histogram is None, initialize it
					if concatHist is None:
						concatHist = hist

					# otherwise, stack the histograms
					else:
						concatHist = sparse.hstack([concatHist, hist])

		# return the concatenated visual words histogram
		return concatHist

	@staticmethod
	def featureDim(numClusters, numLevels):
		# compute and return the reuslting dimensionality of the PBOW feautre representation
		return int(round(numClusters * (1 / 3.0) * ((4 ** (numLevels + 1)) - 1)))
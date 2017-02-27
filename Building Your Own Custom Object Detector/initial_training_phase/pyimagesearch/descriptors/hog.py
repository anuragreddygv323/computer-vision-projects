# import the necessary packages
from skimage import feature

class HOG:
	def __init__(self, orientations=12, pixelsPerCell=(4, 4), cellsPerBlock=(2, 2), normalize=True):
		# store the number of orientations, pixels per cell, cells per block, and
		# whether normalization should be applied to the image
		self.orientations = orientations
		self.pixelsPerCell = pixelsPerCell
		self.cellsPerBlock = cellsPerBlock
		self.normalize = normalize

	def describe(self, image):
		# compute Histogram of Oriented Gradients features
		hist = feature.hog(image, orientations=self.orientations, pixels_per_cell=self.pixelsPerCell,
			cells_per_block=self.cellsPerBlock, transform_sqrt=self.normalize)
		hist[hist < 0] = 0

		# return the histogram
		return hist
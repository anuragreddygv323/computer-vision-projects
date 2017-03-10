# import the necessary packages
import cv2

class RGBHistogram:
	def __init__(self, bins):
		# store the number of bins the histogram will use
		self.bins = bins

	def describe(self, image, mask=None):
		# compute a 3D histogram in the RGB colorspace then normalize the histogram so
		# that images with the same content, but either scaled larger or smaller will
		# have (roughly) the same histogram
		hist = cv2.calcHist([image], [0, 1, 2],
			mask, self.bins, [0, 256, 0, 256, 0, 256])
		cv2.normalize(hist, hist)

		# return out 3D histogram as a flattened array
		return hist.flatten()
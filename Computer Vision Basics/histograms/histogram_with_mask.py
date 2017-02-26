# USAGE
# python histogram_with_mask.py

# import the necessary packages
from matplotlib import pyplot as plt
import numpy as np
import cv2

def plot_histogram(image, title, mask=None):
	# grab the image channels, initialize the tuple of colors and
	# the figure
	chans = cv2.split(image)
	colors = ("b", "g", "r")
	plt.figure()
	plt.title(title)
	plt.xlabel("Bins")
	plt.ylabel("# of Pixels")

	# loop over the image channels
	for (chan, color) in zip(chans, colors):
		# create a histogram for the current channel and plot it
		hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
		plt.plot(hist, color=color)
		plt.xlim([0, 256])

# load the beach image and plot a histogram for it
image = cv2.imread("beach.png")
cv2.imshow("Original", image)
plot_histogram(image, "Histogram for Original Image")

# construct a mask for our image -- our mask will be BLACK for regions
# we want to IGNORE and WHITE for regions we want to EXAMINE
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (60, 290), (210, 390), 255, -1)
cv2.imshow("Mask", mask)

# what does masking our image look like?
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Applying the Mask", masked)

# compute a histogram for our image, but we'll only include pixels in
# the masked region
plot_histogram(image, "Histogram for Masked Image", mask=mask)

# show our plots
plt.show()
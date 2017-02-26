# USAGE
# python connected_components_labeling.py

# import the necessary packages
from __future__ import print_function
from skimage.filters import threshold_adaptive
from skimage import measure
import numpy as np
import cv2

# load the license plate image from disk
plate = cv2.imread("license_plate.png")

# extract the Value component from the HSV color space and apply adaptive thresholding
# to reveal the characters on the license plate
V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
thresh = threshold_adaptive(V, 29, offset=15).astype("uint8") * 255
thresh = cv2.bitwise_not(thresh)

# show the images
cv2.imshow("License Plate", plate)
cv2.imshow("Thresh", thresh)

# perform connected components analysis on the thresholded images and initialize the
# mask to hold only the "large" components we are interested in
labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")
print("[INFO] found {} blobs".format(len(np.unique(labels))))

# loop over the unique components
for (i, label) in enumerate(np.unique(labels)):
	# if this is the background label, ignore it
	if label == 0:
		print("[INFO] label: 0 (background)")
		continue

	# otherwise, construct the label mask to display only connected components for
	# the current label
	print("[INFO] label: {} (foreground)".format(i))
	labelMask = np.zeros(thresh.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)

	# if the number of pixels in the component is sufficiently large, add it to our
	# mask of "large" blobs
	if numPixels > 300 and numPixels < 1500:
		mask = cv2.add(mask, labelMask)

	# show the label mask
	cv2.imshow("Label", labelMask)
	cv2.waitKey(0)

# show the large components in the image
cv2.imshow("Large Blobs", mask)
cv2.waitKey(0)
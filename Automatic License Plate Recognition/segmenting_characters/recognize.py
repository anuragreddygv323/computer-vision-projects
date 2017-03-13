# USAGE
# python recognize.py --images ../testing_lp_dataset

# import the necessary packages
from __future__ import print_function
from pyimagesearch.license_plate import LicensePlateDetector
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to the images to be classified")
args = vars(ap.parse_args())

# loop over the images
for imagePath in sorted(list(paths.list_images(args["images"]))):
	# load the image
	image = cv2.imread(imagePath)
	print(imagePath)

	# if the width is greater than 640 pixels, then resize the image
	if image.shape[1] > 640:
		image = imutils.resize(image, width=640)

	# initialize the license plate detector and detect the license plates and candidates
	lpd = LicensePlateDetector(image)
	plates = lpd.detect()

	# loop over the license plate regions
	for (i, (lp, lpBox)) in enumerate(plates):
		# draw the bounding box surrounding the license plate
		cv2.drawContours(image, [lpBox], -1, (0, 255, 0), 2)

		# show the output images
		candidates = np.dstack([lp.candidates] * 3)
		thresh = np.dstack([lp.thresh] * 3)
		output = np.vstack([lp.plate, thresh, candidates])
		cv2.imshow("Plate & Candidates #{}".format(i + 1), output)

	# display the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
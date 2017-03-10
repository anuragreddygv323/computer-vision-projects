#!/usr/bin/env python

# import the necessary packages
import sys

# import the zipped packages and finish import packages
sys.path.insert(0, "pyimagesearch.zip")
from pyimagesearch.hadoop.mapper import Mapper
from pyimagesearch.descriptors import DetectAndDescribe
from pyimagesearch.descriptors import RootSIFT
import imutils
import cv2

def job():
	# initialize the keypoint detector, local invariant descriptor, and the
	# descriptor
	# pipeline
	detector = cv2.FeatureDetector_create("SURF")
	descriptor = RootSIFT()
	dad = DetectAndDescribe(detector, descriptor)

	# loop over the lines of input
	for line in Mapper.parse_input(sys.stdin):
		# parse the line into the image ID, path, and image
		(imageID, path, image) = Mapper.handle_input(line.strip())

		# describe the image and initialize the output list
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = imutils.resize(image, width=320)
		(kps, descs) = dad.describe(image)
		output = []

		# loop over the keypoints and descriptors
		for (kp, vec) in zip(kps, descs):
			# update the output list as a 2-tuple of the keypoint (x, y)-coordinates
			# and the feature vector
			output.append((kp.tolist(), vec.tolist()))

		# output the row to the reducer
		Mapper.output_row(imageID, path, output, sep="\t")

# handle if the job is being executed
if __name__ == "__main__":
	job()
# USAGE
# python detect_dense_simple.py

# import the necessary packages
from __future__ import print_function
import numpy as np
import argparse
import cv2

#  construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--step", type=int, default=28, help="step (in pixels) of the dense detector")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread("next.png")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect the raw Dense keypoints in the image
detector = cv2.FeatureDetector_create("Dense")
detector.setInt("initXyStep", args["step"])
print("using step size of: {}".format(detector.getInt("initXyStep")))
rawKps = detector.detect(gray)
kps = []

# loop over the raw keypoints
for rawKp in rawKps:
	# loop over the various radii we are going to use
	for r in (4, 8, 12):
		# construct a keypoint manually and then update the keypoitns list
		kp = cv2.KeyPoint(x=rawKp.pt[0], y=rawKp.pt[1], _size=r * 2)
		kps.append(kp)

# show some information regarding the number of keypoints detected
print("# dense keypoints: {}".format(len(rawKps)))
print("# dense + multi radii keypoints: {}".format(len(kps)))

# loop over the keypoints and draw them
for kp in kps:
	r = int(0.5 * kp.size)
	(x, y) = np.int0(kp.pt)
	cv2.circle(image, (x, y), r, (0, 255, 255), 1)

# show the image
cv2.imshow("Images", np.hstack([orig, image]))
cv2.waitKey(0)
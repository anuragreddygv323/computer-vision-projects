# USAGE
# python detect_dense_simple.py

# import the necessary packages
from __future__ import print_function
import numpy as np
import argparse
import cv2

#  construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--step", type=int, default=6, help="step (in pixels) of the dense detector")
ap.add_argument("-r", "--size", type=int, default=1, help="default diameter of keypoint")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread("next.png")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect Dense keypoints in the image
detector = cv2.FeatureDetector_create("Dense")
detector.setInt("initXyStep", args["step"])
print("using step size of: {}".format(detector.getInt("initXyStep")))
kps = detector.detect(gray)
print("# of keypoints: {}".format(len(kps)))

# loop over the keypoints and explicity adjust the keypoint size
for kp in kps:
	kp.size = args["size"]

# loop over the keypoints and draw them
for kp in kps:
	r = int(0.5 * kp.size)
	(x, y) = np.int0(kp.pt)
	cv2.circle(image, (x, y), r, (0, 255, 255), 2)

# show the image
cv2.imshow("Images", np.hstack([orig, image]))
cv2.waitKey(0)
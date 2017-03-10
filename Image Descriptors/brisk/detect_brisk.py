# USAGE
# python detect_brisk.py

# import the necessary packages
from __future__ import print_function
import numpy as np
import cv2

# load the image and convert it to grayscale
image = cv2.imread("next.png")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect BRISK keypoints in the image
detector = cv2.FeatureDetector_create("BRISK")
kps = detector.detect(gray)
print("# of keypoints: {}".format(len(kps)))

# loop over the keypoints and draw them
for kp in kps:
	r = int(0.5 * kp.size)
	(x, y) = np.int0(kp.pt)
	cv2.circle(image, (x, y), r, (0, 255, 255), 2)

# show the image
cv2.imshow("Images", np.hstack([orig, image]))
cv2.waitKey(0)
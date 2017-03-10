# USAGE
# python draw_matches.py --first jp_01.png --second jp_02.png --detector SURF --extractor SIFT

# import the necessary packages
from __future__ import print_function
from pyimagesearch.descriptors import RootSIFT
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="Path to first image")
ap.add_argument("-s", "--second", required=True, help="Path to second image")
ap.add_argument("-d", "--detector", type=str, default="SURF",
	help="Kepyoint detector to use")
ap.add_argument("-e", "--extractor", type=str, default="SIFT",
	help="Feature extractor to use")
ap.add_argument("-m", "--matcher", type=str, default="BruteForce",
	help="Feature matcher to use")
ap.add_argument("-v", "--visualize-each", type=int, default=-1,
	help="Whether or not each match should be visualized individually")
args = vars(ap.parse_args())

# initialize the keypoint detector and keypoint matcher
detector = cv2.FeatureDetector_create(args["detector"])
matcher = cv2.DescriptorMatcher_create(args["matcher"])

# if the feature extractor is `RootSIFT`, then initialize our custom
# RootSIFT class
if args["extractor"] == "RootSIFT":
	extractor = RootSIFT()

# otherwise, initialize the extractor using the built-in OpenCV method
else:
	extractor = cv2.DescriptorExtractor_create(args["extractor"])

# load the two images and convert them to grayscale
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# detect keypoints in the two images
kpsA = detector.detect(grayA)
kpsB = detector.detect(grayB)

# extract features from each of the keypoint regions in the images
(kpsA, featuresA) = extractor.compute(grayA, kpsA)
(kpsB, featuresB) = extractor.compute(grayB, kpsB)

# match the keypoints using the Euclidean distance and initialize
# the list of actual matches
rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
matches = []

# loop over the raw matches
for m in rawMatches:
	# ensure the distance passes David Lowe's ratio test
	if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
		matches.append((m[0].trainIdx, m[0].queryIdx))

# show some diagnostic information
print("# of keypoints from first image: {}".format(len(kpsA)))
print("# of keypoints from second image: {}".format(len(kpsB)))
print("# of matched keypoints: {}".format(len(matches)))

# initialize the output visualization image
(hA, wA) = imageA.shape[:2]
(hB, wB) = imageB.shape[:2]
vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
vis[0:hA, 0:wA] = imageA
vis[0:hB, wA:] = imageB

# loop over the matches
for (trainIdx, queryIdx) in matches:
	# generate a random color and draw the match
	color = np.random.randint(0, high=255, size=(3,))
	ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
	ptB = (int(kpsB[trainIdx].pt[0] + wA), int(kpsB[trainIdx].pt[1]))
	cv2.line(vis, ptA, ptB, color, 2)

	# check to see if each match should be visualized individually
	if args["visualize_each"] > 0:
		cv2.imshow("Matched", vis)
		cv2.waitKey(0)

# show the visualization
if args["visualize_each"] <= 0:
	cv2.imshow("Matched", vis)
	cv2.waitKey(0)
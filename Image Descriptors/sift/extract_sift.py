# USAGE
# python extract_sift.py --image jp_01.png

# import the necessary packages
from __future__ import print_function
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# initialize the keypoint detector and local invariant descriptor
detector = cv2.FeatureDetector_create("SIFT")
extractor = cv2.DescriptorExtractor_create("SIFT")

# load the input image, convert it to grayscale, detect keypoints, and then
# extract local invariant descriptors
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kps = detector.detect(gray)
(kps, descs) = extractor.compute(gray, kps)

# show the shape of the keypoints and local invariant descriptors array
print("[INFO] # of keypoints detected: {}".format(len(kps)))
print("[INFO] feature vector shape: {}".format(descs.shape))
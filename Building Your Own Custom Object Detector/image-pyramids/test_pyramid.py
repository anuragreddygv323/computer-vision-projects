# USAGE
# python test_pyramid.py --image florida_trip.png --scale 1.5

# import the necessary packages
from pyimagesearch.object_detection.helpers import pyramid
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factor size")
args = vars(ap.parse_args())

# load the input image
image = cv2.imread(args["image"])

# loop over the layers of the image pyramid and display them
for (i, layer) in enumerate(pyramid(image, scale=args["scale"])):
	cv2.imshow("Layer {}".format(i + 1), layer)
	cv2.waitKey(0)
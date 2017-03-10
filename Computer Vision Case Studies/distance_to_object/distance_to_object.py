# USAGE
# python distance_to_object.py --reference reference/ref_24in.jpg --ref-width-inches 4.0 \
#	--ref-distance-inches 24.0 --images images

# import the necessary packages
from pyimagesearch.markers import DistanceFinder
from imutils import paths
import argparse
import imutils
import cv2

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--reference", required=True, help="path to the reference image")
ap.add_argument("-w", "--ref-width-inches", required=True, type=float,
	help="reference object width in inches")
ap.add_argument("-d", "--ref-distance-inches", required=True, type=float,
	help="distance to reference object in inches")
ap.add_argument("-i", "--images", required=True,
	help="path to the directory containing images to test")
args = vars(ap.parse_args())

# load the reference image and resize it
refImage = cv2.imread(args["reference"])
refImage = imutils.resize(refImage, height=700)

# initialize the distance finder, find the marker in the image, and calibrate the distance
# finder
df = DistanceFinder(args["ref_width_inches"], args["ref_distance_inches"])
refMarker = DistanceFinder.findSquareMarker(refImage)
df.calibrate(refMarker[2])

# visualize the results on the reference image and display it
refImage = df.draw(refImage, refMarker, df.distance(refMarker[2]))
cv2.imshow("Reference", refImage)

# loop over the image paths
for imagePath in paths.list_images(args["images"]):
	# extract the filename, load the image, and resize it
	filename = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	image = imutils.resize(image, height=700)
	print "[INFO] processing {}".format(filename)

	# find the marker in the image
	marker = DistanceFinder.findSquareMarker(image)

	# if the marker is None, then the square marker could not be found in the image
	if marker is None:
		print "[INFO] could not find marker for {}".format(filename)
		continue

	# determine the distance to the marker
	distance = df.distance(marker[2])

	# visualize the results on the image and display it
	image = df.draw(image, marker, distance)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
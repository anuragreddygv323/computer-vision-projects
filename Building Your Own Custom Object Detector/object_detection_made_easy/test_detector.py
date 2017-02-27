# USAGE
# python test_detector.py --detector output/stop_sign_detector.svm --testing stop_sign_testing

# import the necessary packages
from imutils import paths
import argparse
import dlib
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True, help="Path to trained object detector")
ap.add_argument("-t", "--testing", required=True, help="Path to directory of testing images")
args = vars(ap.parse_args())

# load the detector
detector = dlib.simple_object_detector(args["detector"])

# loop over the testing images
for testingPath in paths.list_images(args["testing"]):
	# load the image and make predictions
	image = cv2.imread(testingPath)
	boxes = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

	# loop over the bounding boxes and draw them
	for b in boxes:
		(x, y, w, h) = (b.left(), b.top(), b.right(), b.bottom())
		cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

	# show the image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
# USAGE
# python face_detector_demo.py --face-data output/faces/hadoop_output

# import the necessary packages
import argparse
import random
import glob
import json
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face-data", required=True, help="path to the face output directory")
ap.add_argument("-s", "--sample", type=int, default=10, help="# of face samples to use")
args = vars(ap.parse_args())

# initialize the list of face data
faceData = []

# loop over the output files from Hadoop
for p in glob.glob(args["face_data"] + "/part-*"):
	# load the output file, split it into lines, and update the face data
	d = open(p).read().strip().split("\n")
	faceData.extend(d)

# randomly sample the face data
faceData = random.sample(faceData, args["sample"])

# loop over the sampled data
for row in faceData:
	# unpack the row, then load the image and the face bounding boxes
	(imageID, path, faceBoxes) = row.strip().split("\t")
	image = cv2.imread(path)
	faceBoxes = json.loads(faceBoxes)

	# loop over the faces bounding boxes and draw them
	for (x, y, w, h) in faceBoxes:
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the output image
	print "[INFO] {num_faces} face(s) were found in {filename}".format(num_faces=len(faceBoxes),
		filename=path[path.rfind("/") + 1:])
	cv2.imshow("image", image)
	cv2.waitKey(0)
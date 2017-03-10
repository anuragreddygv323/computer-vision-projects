#!/usr/bin/env python

# import the necessary packages
import sys

# import the zipped packages and finish import packages
sys.path.insert(0, "pyimagesearch.zip")
from pyimagesearch.hadoop.mapper import Mapper
from pyimagesearch.face_detection import FaceDetector

def job():
	# initialize the face detector
	fd = FaceDetector("haarcascade_frontalface_default.xml")

	# loop over the lines of input
	for line in Mapper.parse_input(sys.stdin):
		# parse the line into the image ID, path, and image
		(imageID, path, image) = Mapper.handle_input(line.strip())

		# detect the raw faces in the image and initialize the list of faces we'll
		# be using
		rawFaces = fd.detect(image, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
		faces = []

		# loop over the raw faces and convert them to normal Python lists rather than
		# NumPy arrays so they can be serialized via JSON
		for face in rawFaces:
			faces.append(face.tolist())

		# output the row to the reducer
		Mapper.output_row(imageID, path, faces, sep="\t")

# handle if the job is being executed
if __name__ == "__main__":
	job()
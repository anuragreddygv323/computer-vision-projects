# USAGE
# python gather_selfies.py --face-cascade cascades/haarcascade_frontalface_default.xml \
#	--output output/faces/adrian.txt

# import the necessary packages
from __future__ import print_function
from pyimagesearch.face_recognition import FaceDetector
from imutils import encodings
import argparse
import imutils
import cv2

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face-cascade", required=True, help="path to face detection cascade")
ap.add_argument("-o", "--output", required=True, help="path to output file")
ap.add_argument("-w", "--write-mode", type=str, default="a+", help="write method for the output file")
args = vars(ap.parse_args())

# initialize the face detector, boolean indicating if we are in capturing mode or not, and
# the bounding box color
fd = FaceDetector(args["face_cascade"])
captureMode = False
color = (0, 255, 0)

# grab a reference to the webcam and open the output file for writing
camera = cv2.VideoCapture(0)
f = open(args["output"], args["write_mode"])
total = 0

# loop over the frames of the video
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if the frame could not be grabbed, then we have reached the end of the video
	if not grabbed:
		break

	# resize the frame, convert the frame to grayscale, and detect faces in the frame
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=9, minSize=(100, 100))

	# ensure that at least one face was detected
	if len(faceRects) > 0:
		# sort the bounding boxes, keeping only the largest one
		(x, y, w, h) = max(faceRects, key=lambda b:(b[2] * b[3]))

		# if we are in capture mode, extract the face ROI, encode it, and write it to file
		if captureMode:
			face = gray[y:y + h, x:x + w].copy(order="C")
			f.write("{}\n".format(encodings.base64_encode_image(face)))
			total += 1

		# draw bounding box on the frame
		cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

	# show the frame and record if the user presses a key
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `c` key is pressed, then go into capture mode
	if key == ord("c"):
		# if we are not already in capture mode, drop into capture mode
		if not captureMode:
			captureMode = True
			color = (0, 0, 255)

		# otherwise, back out of capture mode
		else:
			captureMode = False
			color = (0, 255, 0)

	# if the `q` key is pressed, break from the loop
	elif key == ord("q"):
		break

# close the output file, cleanup the camera, and close any open windows
print("[INFO] wrote {} frames to file".format(total))
f.close()
camera.release()
cv2.destroyAllWindows()
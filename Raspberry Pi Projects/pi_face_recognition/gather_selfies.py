# USAGE
# python gather_selfies.py --face-cascade cascades/haarcascade_frontalface_default.xml \
#	--output output/faces/adrian.txt

# import the necessary packages
from __future__ import print_function
from pyimagesearch.face_recognition import FaceDetector
from imutils.video import VideoStream
from imutils import encodings
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face-cascade", required=True, help="path to face detection cascade")
ap.add_argument("-o", "--output", required=True, help="path to output file")
ap.add_argument("-w", "--write-mode", type=str, default="a+", help="write method for the output file")
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] warming up camera...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# initialize the face detector, boolean indicating if we are in capturing mode or not, and
# the bounding box color
fd = FaceDetector(args["face_cascade"])
captureMode = False
color = (0, 255, 0)

# open the output file for writing
f = open(args["output"], args["write_mode"])
total = 0

# loop over the frames of the video
while True:
	# grab the next frame from the stream, resize the it, convert it to grayscale, and detect
	# faces
	frame = vs.read()
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
cv2.destroyAllWindows()
vs.stop()
# USAGE
# python security_cam.py --face-cascade cascades/haarcascade_frontalface_default.xml --classifier output/classifier \
#	--conf conf/alerts.json

# import the necessary packages
from __future__ import print_function
from pyimagesearch.face_recognition import FaceDetector
from pyimagesearch.face_recognition import FaceRecognizer
from pyimagesearch.notifications import TwilioNotifier
from pyimagesearch.utils import Conf
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
ap.add_argument("-f", "--face-cascade", required=True, help="path to face detection cascade")
ap.add_argument("-m", "--classifier", required=True, help="path to the classifier")
ap.add_argument("-t", "--confidence", type=float, default=100.0,
	help="maximum confidence threshold for positive face identification")
ap.add_argument("-n", "--consec-frames", type=int, default=90,
	help="# of consecutive frames containing an unknown face before sending alert")
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# load the configuration file and initialize the Twilio notifier
conf = Conf(args["conf"])
tn = TwilioNotifier(conf)

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] warming up camera...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# initialize the face detector, load the face recognizer, and set the confidence
# threshold
fd = FaceDetector(args["face_cascade"])
fr = FaceRecognizer.load(args["classifier"])
fr.setConfidenceThreshold(args["confidence"])

# initialize the number of consecutive frames list that will keep track of (1) the
# name of the face in the image and (2) the number of *consecutive* frames the face
# has appeared in
consec = None

# initialize the color of the bounding box used for the face and the last time
# we sent a MMS notification
color = (0, 255, 0)
lastSent = None

# loop over the frames of the video
while True:
	# grab the next frame from the stream and initialize the intruder boolean
	frame = vs.read()
	intruder = False

	# resize the frame, convert the frame to grayscale, and detect faces in the frame
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

	# draw the timestamp on the frame
	timestamp = datetime.datetime.now()
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.35, (0, 0, 255), 1)

	# loop over the face bounding boxes
	for (i, (x, y, w, h)) in enumerate(faceRects):
		# grab the face to predict and predict it
		face = gray[y:y + h, x:x + w]
		(prediction, confidence) = fr.predict(face)

		# if the consecutive frames list is None, or the prediction does not match the
		# name from the previous frame, re-initialize the list
		if consec is None:
			consec = [prediction, 1]
			color = (0, 255, 0)

		# if predicted face matches the name in the consecutive list, then update the
		# total count
		elif prediction == consec[0]:
			consec[1] += 1

		# if we the prediction has been "unknown" for a sufficient number of frames,
		# then we have an intruder
		if consec[0] == "Unknown" and consec[1] >= args["consec_frames"]:
			# change the color of the bounding box and text
			color = (0, 0, 255)
			intruder = True

		# display the text prediction on the image, followed by drawing a bounding box
		# around the face
		text = "{}: {:.2f}".format(prediction, confidence)
		cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
		cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

		# check to see if we (1) have an intruder and (2) enough time has passed
		# between message sends
		if intruder:
			if lastSent is None or (timestamp - lastSent).seconds >= conf["wait_n_seconds"]:
				# send the frame via Twilio, and update the last send timestamp
				print("[INFO] intruder: {}".format(timestamp))
				tn.send(frame)
				lastSent = timestamp

	# show the frame and record if the user presses a key
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
cv2.destroyAllWindows()
vs.stop()
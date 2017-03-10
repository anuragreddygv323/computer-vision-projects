# USAGE
# python detect.py --bounding-box "10,350,225,590"

# import the necessary packages
from pyimagesearch.gesture_recognition import MotionDetector
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--bounding-box", required=True,
	help="comma separted list of top, right, bottom, left coordinates of hand ROI")
ap.add_argument("-v", "--video", required=False, help="path to the (optional) video file")
args = vars(ap.parse_args())

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

# unpack the hand ROI, then initialize the motion detector and the total number of
# frames read thus far
(top, right, bot, left) = np.int32(args["bounding_box"].split(","))
md = MotionDetector()
numFrames = 0

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a frame, then we have reached the
	# end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame and flip it so the frame is no longer a mirror view
	frame = imutils.resize(frame, width=600)
	frame = cv2.flip(frame, 1)
	clone = frame.copy()
	(frameH, frameW) = frame.shape[:2]

	# extract the ROI, passing in right:left since the image is mirrored, then
	# blur it slightly
	roi = frame[top:bot, right:left]
	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)

	# if we not reached 32 initial frames, then calibrate the skin detector
	if numFrames < 32:
		md.update(gray)

	# otherwise, detect skin in the ROI
	else:
		# detect motion (i.e., skin) in the image
		skin = md.detect(gray)

		# check to see if skin has been detected
		if skin is not None:
			# unpack the tuple and draw the contours on the image
			(thresh, c) = skin
			cv2.drawContours(clone, [c + (right, top)], -1, (0, 255, 0), 2)
			cv2.imshow("Thresh", thresh)

	# draw the hand ROI and increment the number of processed frames
	cv2.rectangle(clone, (left, top), (right, bot), (0, 0, 255), 2)
	numFrames += 1

	# show the frame to our screen
	cv2.imshow("Frame", clone)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
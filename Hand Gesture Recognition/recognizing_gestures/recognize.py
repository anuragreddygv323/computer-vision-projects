# USAGE
# python recognize.py --bounding-box "10,350,225,590"

# import the necessary packages
from pyimagesearch.gesture_recognition import GestureDetector
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

# unpack the hand ROI, then initialize the motion detector and gesture detector
(top, right, bot, left) = np.int32(args["bounding_box"].split(","))
gd = GestureDetector()
md = MotionDetector()

# initialize the total number of frames read thus far, a bookkeeping variable used to
# keep track of the number of consecutive frames a gesture has appeared in, along
# with the values recognized by the gesture detector
numFrames = 0
gesture = None
values = []

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
			# unpack the tuple and detect the gesture in the thresholded image
			(thresh, c) = skin
			cv2.drawContours(clone, [c + (right, top)], -1, (0, 255, 0), 2)
			fingers = gd.detect(thresh, c)

			# if the current gesture count is None, initialize it
			if gesture is None:
				gesture = [1, fingers]

			# otherwise the finger count has been initialized
			else:
				# if the finger counts are the same, increment the number of frames
				if gesture[1] == fingers:
					gesture[0] += 1

					# if we have reached a sufficient number of frames, draw the number of the
					# screen
					if gesture[0] >= 25:
						# if the values list is already full, reset it
						if len(values) == 2:
							values = []

						# update the values list and reset the gesture
						values.append(fingers)
						gesture = None

				# otherwise, the finger counts do not match up, so reset the bookkeeping variable
				else:
					gesture = None

	# check to see if there is at least one entry in the values list
	if len(values) > 0:
		# draw the first digit and the plus sign
		GestureDetector.drawBox(clone, 0)
		GestureDetector.drawText(clone, 0, values[0])
		GestureDetector.drawText(clone, 1, "+")

	# check to see if there is a second entry in the values list
	if len(values) == 2:
		# draw the second digit, the equal sign, and fianlly the answer
		GestureDetector.drawBox(clone, 2)
		GestureDetector.drawText(clone, 2, values[1])
		GestureDetector.drawText(clone, 3, "=")
		GestureDetector.drawBox(clone, 4, color=(0, 255, 0))
		GestureDetector.drawText(clone, 4, values[0] + values[1], color=(0, 255, 0))

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
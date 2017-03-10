# USAGE
# python detect.py --output output

# import the necessary packages
from __future__ import print_function
from pyimagesearch.motion_detection import SingleMotionDetector
from scipy.spatial import distance as dist
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output directory")
ap.add_argument("-m", "--min-frames", type=int, default=120,
	help="minimum # of frames containing motion before writing to file")
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] warming up camera...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# initialize the motion detector, the total number of frames read thus far, the
# number of consecutive frames that have contained motion, and the spatial
# dimensions of the frame
md = SingleMotionDetector(accumWeight=0.1)
total = 0
consec = None
frameShape = None

# loop over frames
while True:
	# read the next frame from the video stream, resize it, convert the frame to
	# grayscale, and blur it
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)

	# grab the current timestamp and draw it on the frame
	timestamp = datetime.datetime.now()
	cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# if we do not already have the dimensions of the frame, initialize it
	if frameShape is None:
		frameShape = (int(gray.shape[0] / 2), int(gray.shape[1] / 2))

	# if the total number of frames has reached a sufficient number to construct a
	# reasonable background model, then continue to process the frame
	if total > 32:
		# detect motion in the image
		motion = md.detect(gray)

		# if the `motion` object None, then motion has occurred in the image
		if motion is not None:
			# unpack the motion tuple, compute the center (x, y)-coordinates of the
			# bounding box, and draw the bounding box of the motion on the output frame
			(thresh, (minX, minY, maxX, maxY)) = motion
			cX = int((minX + maxX) / 2)
			cY = int((minY + maxY) / 2)
			cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 255), 2)

			# if the number of consecutive frames is None, initialize it using a list
			# of the number of total frames, the frame itself, along with distance of
			# the centroid to the center of the image
			if consec is None:
				consec = [1, frame, dist.euclidean((cY, cX), frameShape)]

			# otherwise, we need to update the bookkeeping variable
			else:
				# compute the Euclidean distance between the motion centroid and the
				# center of the frame, then increment the total number of *consecutive
				# frames* that contain motion
				d = dist.euclidean((cY, cX), frameShape)
				consec[0] += 1

				# if the distance is smaller than the current distance, then update the
				# bookkeeping variable
				if d < consec[2]:
					consec[1:] = (frame, d)

			# if a sufficient number of frames have contained motion, log the motion
			if consec[0] == args["min_frames"]:
				# write the frame to file, then reset the consecutive bookkeeing variable
				print("[INFO] logging motion to file: {}".format(timestamp))
				outputPath = "{}/{}.jpg".format(args["output"],
					timestamp.strftime("%Y%m%d-%H%M%S"))
				cv2.imwrite(outputPath, consec[1])
				consec = None

		# otherwise, there is no motion in the frame so reset the consecutive bookkeeing
		# variable
		else:
			consec = None

	# update the background model and increment the total number of frames read thus far
	md.update(gray)
	total += 1

	# show the frame and record if the user presses a key
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
cv2.destroyAllWindows()
vs.stop()
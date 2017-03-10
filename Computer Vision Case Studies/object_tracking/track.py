# USAGE
# python track.py
# python track.py --video BallTracking_01.mp4

# import the necessary packages
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

# define the color ranges
colorRanges = [
	((29, 86, 6), (64, 255, 255), "green"),
	((57, 68, 0), (151, 255, 255), "blue")]

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a frame, then we have
	# reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame, blur it, and convert it to the HSV color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# loop over the color ranges
	for (lower, upper, colorName) in colorRanges:
		# construct a mask for all colors in the current HSV range, then
		# perform a series of dilations and erosions to remove any small
		# blobs left in the mask
		mask = cv2.inRange(hsv, lower, upper)
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)

		# find contours in the mask
		(cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)

		# only proceed if at least one contour was found
		if len(cnts) > 0:
			# find the largest contour in the mask, then use it to compute
			# the minimum enclosing circle and centroid
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			(cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			# only draw the enclosing circle and text if the radious meets
			# a minimum size
			if radius > 10:
				cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
				cv2.putText(frame, colorName, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
					1.0, (0, 255, 255), 2)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
# USAGE
# python approx_simple.py

# import the necessary packages
import cv2

# load the the cirles and squares image and convert it to grayscale
image = cv2.imread("images/circles_and_squares.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# find contours in the image
(_, cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.01 * peri, True)

	# if the approximated contour has 4 vertices, then we are examining
	# a rectangle
	if len(approx) == 4:
		# draw the outline of the contour and draw the text on the image
		cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
		(x, y, w, h) = cv2.boundingRect(approx)
		cv2.putText(image, "Rectangle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 255, 255), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
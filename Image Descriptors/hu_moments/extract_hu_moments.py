# USAGE
# python extract_hu_moments.py

# import the necessary packages
import cv2

# load the input image and convert it to grayscale
image = cv2.imread("planes.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# compute the Hu Moments feature vector for the entire image and show it
moments = cv2.HuMoments(cv2.moments(image)).flatten()
print "ORIGINAL MOMENTS: {}".format(moments)
cv2.imshow("Image", image)
cv2.waitKey(0)

# find the contours of the three planes in the image
(cnts, _) = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

# loop over each contour
for (i, c) in enumerate(cnts):
	# extract the ROI from the image and compute the Hu Moments feature
	# vector for the ROI
	(x, y, w, h) = cv2.boundingRect(c)
	roi = image[y:y + h, x:x + w]
	moments = cv2.HuMoments(cv2.moments(roi)).flatten()

	# show the moments and ROI
	print "MOMENTS FOR PLANE #{}: {}".format(i + 1, moments)
	cv2.imshow("ROI #{}".format(i + 1), roi)
	cv2.waitKey(0)
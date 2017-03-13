# USAGE
# python cluster_colors.py

# import the necessary packages
from sklearn.cluster import KMeans
import numpy as np
import random
import cv2

# initialize the list of color choices
colors = [
	# shades of red, green, and blue
	(138, 8, 8), (180, 4, 4), (223, 1, 1), (255, 0, 0), (250, 88, 88),
	(8, 138, 8), (4, 180, 4), (1, 223, 1), (0, 255, 0), (46, 254, 46),
	(11, 11, 97), (8, 8, 138), (4, 4, 180), (0, 0, 255), (46, 46, 254)]

# initialize the canvas
canvas = np.ones((400, 600, 3), dtype="uint8") * 255

# loop over the canvas
for y in xrange(0, 400, 20):
	for x in xrange(0, 600, 20):
		# generate a random (x, y) coordinate, radius, and color for
		# the circle
		(dX, dY) = np.random.randint(5, 10, size=(2,))
		r = np.random.randint(5, 8)
		color = random.choice(colors)[::-1]

		# draw the circle on the canvas
		cv2.circle(canvas, (x + dX, y + dY), r, color, -1)

# pad the border of the image
canvas = cv2.copyMakeBorder(canvas, 5, 5, 5, 5, cv2.BORDER_CONSTANT,
	value=(255, 255, 255))

# convert the canvas to grayscale, threshold it, and detect contours
# in the image
gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
(cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# initialize the data matrix
data = []

# loop over the contours
for c in cnts:
	# construct a mask from the contour
	mask = np.zeros(canvas.shape[:2], dtype="uint8")
	cv2.drawContours(mask, [c], -1, 255, -1)
	features = cv2.mean(canvas, mask=mask)[:3]
	data.append(features)

# cluster the color features
clt = KMeans(n_clusters=3)
clt.fit(data)
cv2.imshow("Canvas", canvas)

# loop over the unique cluster identifiers
for i in np.unique(clt.labels_):
	# construct a mask for the current cluster
	mask = np.zeros(canvas.shape[:2], dtype="uint8")

	# loop over the indexes of the current cluster and draw them
	for j in np.where(clt.labels_ == i)[0]:
		cv2.drawContours(mask, [cnts[j]], -1, 255, -1)

	# show the output image for the cluster
	cv2.imshow("Cluster", cv2.bitwise_and(canvas, canvas, mask=mask))
	cv2.waitKey(0)
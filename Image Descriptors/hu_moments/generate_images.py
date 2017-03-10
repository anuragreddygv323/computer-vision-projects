# USAGE
# python generate_images.py --output output

# import the necessary packages
import numpy as np
import argparse
import uuid
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="Path to the output directory")
ap.add_argument("-n", "--num-images", type=int, default=500,
	help="# of disctrator images to generate")
args = vars(ap.parse_args())

# loop over the number of distractor images to generate
for i in xrange(0, args["num_images"]):
	# allocate memory for the image, then generate the (x, y)
	# center of the circle, then generate the radius of the
	# circle, ensuring that the circle is fully contained in
	# the image
	image = np.zeros((500, 500, 3), dtype="uint8")
	(x, y) = np.random.uniform(low=105, high=405, size=(2,)).astype("int0")
	r = np.random.uniform(low=25, high=100, size=(1,)).astype("int0")[0]

	# randomly generate a color for the circle, draw it, and
	# write the image to file using a random filename
	color = np.random.uniform(low=0, high=255, size=(3,)).astype("int0")
	cv2.circle(image, (x, y), r, color, -1)
	cv2.imwrite("{}/{}.jpg".format(args["output"], uuid.uuid4()), image)

# allocate memory for the rectangle image, then generate the starting
# and ending (x, y) coordinates of the square
image = np.zeros((500, 500, 3), dtype="uint8")
topLeft = np.random.uniform(low=25, high=225, size=(2,)).astype("int0")
botRight = np.random.uniform(low=250, high=400, size=(2,)).astype("int0")

# draw the rectangle on the image and write it to file, again,
# using a random filename
color = np.random.uniform(low=0, high=255, size=(3,)).astype("int0")
cv2.rectangle(image, tuple(topLeft), tuple(botRight), color, -1)
cv2.imwrite("{}/{}.jpg".format(args["output"], uuid.uuid4()), image)
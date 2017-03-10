# import the necessary packages
import numpy as np
import cv2

def prepare_image(image, fixedSize):
	# convert the image from BGR to RGB, then resize it to a fixed size,
	# ignoring aspect ratio
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, tuple(fixedSize))

	# return the image
	return image

def build_batch(paths, fixedSize):
	# load the images from disk, prepare them for extraction, and convert
	# the list to a NumPy array
	images = [prepare_image(cv2.imread(p), fixedSize) for p in paths]
	images = np.array(images, dtype="float")

	# extract the labels from the image paths
	labels = [":".join(p.split("/")[-2:]) for p in paths]

	# return the labels and images
	return (labels, images)

def chunk(l, n):
	# loop over the list `l`, yielding chunks of size `n`
	for i in np.arange(0, len(l), n):
		yield l[i:i + n]
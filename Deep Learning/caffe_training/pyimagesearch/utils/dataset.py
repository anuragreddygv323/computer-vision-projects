# import the necessary packages
import numpy as np
import cPickle
import cv2

def build_cifar10(inputPaths, outputPath, outputFile):
	# loop over the input CIFAR-10 files
	for path in inputPaths:
		# load the data
		data = cPickle.loads(open(path).read())

		# loop over the data
		for (i, image) in enumerate(data["data"]):
			# unpack the RGB components of the image
			(R, G, B) = (image[:1024], image[1024:2048], image[2048:])
			image = np.dstack([B, G, R]).reshape((32, 32, 3))

			# construct the path to the output image file and write it to disk
			p = "{}/{}".format(outputPath, data["filenames"][i])
			cv2.imwrite(p, image)

			# update the training file with the path and class label
			outputFile.write("{} {}\n".format(p, data["labels"][i]))
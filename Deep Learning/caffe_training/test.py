# USAGE
# python test.py --arch cifar10_deploy.prototxt \
#	--snapshot output/cifar10/snapshots/cifar10_iter_70000.caffemodel.h5 \
#	--mean output/cifar10/dataset_mean.binaryproto --val output/cifar10/val.txt \
#	--test-images test_images

# import the necessary packages
from __future__ import print_function
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import caffe
import cv2

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--arch", required=True,
	help="path to network architecture definition")
ap.add_argument("-s", "--snapshot", required=True, help="path network snapsot")
ap.add_argument("-m", "--mean", required=True, help="path to mean image")
ap.add_argument("-v", "--val", required=True, help="path to evaluation file")
ap.add_argument("-t", "--test-images", required=True,
	help="path to the directory of testing images")
ap.add_argument("-g", "--gpu", type=int, default=-1, help="GPU device index")
args = vars(ap.parse_args())

# initialize the ground-truth labels for the CIFAR-10 dataset
gtLabels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
	"ship", "truck"]

# check to see if the GPU should be utilized
if args["gpu"] > -1:
	print("[INFO] using GPU: {}".format(args["gpu"]))
	caffe.set_mode_gpu()
	caffe.set_device(args["gpu"])

# otherwise, the CPU is being used
else:
	print("[INFO] using CPU")

# load the mean image
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(args["mean"], "rb").read()
blob.ParseFromString(data)
mean = np.array(caffe.io.blobproto_to_array(blob))

# load the network
net = caffe.Classifier(args["arch"], args["snapshot"], image_dims=(32, 32),
	mean=mean[0], raw_scale=255)

# load the evaluation file and randomly select rows from the data
print("[INFO] testing CIFAR-10 evaluation data...")
rows = open(args["val"]).read().strip().split("\n")
rows = random.sample(rows, 10)

# loop over the randomly selected rows
for row in rows:
	# unpack the row and load the image from disk
	(path, label) = row.split(" ")
	image = caffe.io.load_image(path)
	
	# make prediction on image
	pred = net.predict([image])
	i = pred[0].argmax()

	# convert the image to BGR ordering and resize it
	(R, G, B) = cv2.split(image)
	image = cv2.merge([B, G, R])
	image = imutils.resize(image, width=256, inter=cv2.INTER_CUBIC)

	# display the output prediction to our screen
	print("[INFO] predicted: {}, actual: {}".format(gtLabels[i],
		gtLabels[int(label)]))
	cv2.putText(image, "{}: {}%".format(gtLabels[i], int(pred[0][i] * 100)),
		(5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)

# move on to extra testing images...
print("[INFO] testing images *not* part of CIFAR-10...")

# loop over the images not part of the CIFAR-10 dataset
for imagePath in paths.list_images(args["test_images"]):
	# load the image and resize it to a fixed size
	print("[INFO] classifying {}".format(imagePath[imagePath.rfind("/") + 1:]))
	image = caffe.io.load_image(imagePath)
	orig = image.copy()
	image = cv2.resize(image, (32, 32))

	# make a prediction on the image
	pred = net.predict([image])
	i = pred[0].argmax()

	# convert the image to BGR ordering, then display the prediction to
	# our screen
	(R, G, B) = cv2.split(orig)
	orig = cv2.merge([B, G, R])
	cv2.putText(orig, "{}: {}%".format(gtLabels[i], int(pred[0][i] * 100)),
		(5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
	cv2.imshow("Image", orig)
	cv2.waitKey(0)
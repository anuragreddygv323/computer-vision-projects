# USAGE
# python index_features.py --dataset output/data/training --features-db output/training_features.hdf5

# import the necessary packages
from __future__ import print_function
from pyimagesearch.descriptors import DetectAndDescribe
from pyimagesearch.descriptors import RootSIFT
from pyimagesearch.indexer import FeatureIndexer
from imutils import paths
import argparse
import imutils
import random
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="Path to the directory that contains the images to be indexed")
ap.add_argument("-f", "--features-db", required=True,
	help="Path to where the features database will be stored")
ap.add_argument("-a", "--approx-images", type=int, default=250,
	help="Approximate # of images in the dataset")
ap.add_argument("-b", "--max-buffer-size", type=int, default=50000,
	help="Maximum buffer size for # of features to be stored in memory")
args = vars(ap.parse_args())

# initialize the keypoint detector, local invariant descriptor, and the descriptor
# pipeline
detector = cv2.FeatureDetector_create("GFTT")
descriptor = RootSIFT()
dad = DetectAndDescribe(detector, descriptor)

# initialize the feature indexer
fi = FeatureIndexer(args["features_db"], estNumImages=args["approx_images"],
	maxBufferSize=args["max_buffer_size"], verbose=True)

# grab the image paths and randomly shuffle them
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

# loop over the images in the dataset
for (i, imagePath) in enumerate(imagePaths):
	# check to see if progress should be displayed
	if i > 0 and i % 10 == 0:
		fi._debug("processed {} images".format(i), msgType="[PROGRESS]")

	# load the image and pre-process it
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=320)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# describe the image
	(kps, descs) = dad.describe(image)

	# if either the keypoints or descriptors are None, then ignore the image
	if kps is None or descs is None:
		continue

	# extract the image filename and label from the path, then index the features
	(label, filename) = imagePath.split("/")[-2:]
	k = "{}:{}".format(label, filename)
	fi.add(k, image.shape, kps, descs)

# finish the indexing process
fi.finish()
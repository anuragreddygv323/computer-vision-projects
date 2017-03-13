# USAGE
# python index_features.py --dataset caltech5 --features-db output/features.hdf5 --approx-images 500

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
ap.add_argument("-a", "--approx-images", type=int, default=500,
	help="Approximate # of images in the dataset")
ap.add_argument("-b", "--max-buffer-size", type=int, default=50000,
	help="Maximum buffer size for # of features to be stored in memory")
args = vars(ap.parse_args())

# initialize the keypoint detector, local invariant descriptor, and the descriptor
# pipeline
detector = cv2.FeatureDetector_create("GFTT")
descriptor = RootSIFT()
dad = DetectAndDescribe(detector, descriptor)

# initialize the feature indexer, then grab the image paths and randomly shuffle
# them
fi = FeatureIndexer(args["features_db"], estNumImages=args["approx_images"],
	maxBufferSize=args["max_buffer_size"], verbose=True)
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

# loop over the images in the dataset
for (i, imagePath) in enumerate(imagePaths):
	# check to see if progress should be displayed
	if i > 0 and i % 10 == 0:
		fi._debug("processed {} images".format(i), msgType="[PROGRESS]")

	# extract the filename and image class from the image path and use it to
	# construct the unique image ID
	p = imagePath.split("/")
	imageID = "{}:{}".format(p[-2], p[-1])

	# load the image and prepare it from description
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=min(320, image.shape[1]))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# describe the image
	(kps, descs) = dad.describe(image)

	# if either the keypoints or descriptors are None, then ignore the image
	if kps is None or descs is None:
		continue

	# index the features
	fi.add(imageID, kps, descs)

# finish the indexing process
fi.finish()
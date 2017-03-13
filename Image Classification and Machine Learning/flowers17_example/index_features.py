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
ap.add_argument("-k", "--kp-detector", type=str, default="GFTT",
	help="Keypoint detector method")
ap.add_argument("-l", "--desc", type=str, default="RootSIFT",
	help="Local invariant descriptor method")
args = vars(ap.parse_args())

# initialize the keypoint detector and local invariant descriptor
detector = cv2.FeatureDetector_create(args["kp_detector"])
descriptor = RootSIFT()

# if the descriptor method is not 'RootSIFT', then update the descriptor
if args["desc"].lower() != "rootsift":
	descriptor = cv2.DescriptorExtractor_create(args["desc"])

# initialize the descriptor pipeline
dad = DetectAndDescribe(detector, descriptor)

# initialize the feature indexer, then grab the image paths and randomly shuffle
# them
fi = FeatureIndexer(args["features_db"], estNumImages=args["approx_images"],
	maxBufferSize=args["max_buffer_size"], verbose=True)
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

# loop over the images in the dataset
for (i, imagePath) in enumerate(imagePaths):
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
# USAGE
# python test_model.py --images test_images --codebook output/vocab.cpickle --model output/model.cpickle

# import the necessary packages
from __future__ import print_function
from pyimagesearch.descriptors import DetectAndDescribe
from pyimagesearch.descriptors import RootSIFT
from pyimagesearch.ir import BagOfVisualWords
from imutils import paths
import argparse
import cPickle
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="Path to input images directory")
ap.add_argument("-c", "--codebook", required=True,
	help="Path to the codebook")
ap.add_argument("-m", "--model", required=True,
	help="Path to the classifier")
args = vars(ap.parse_args())

# initialize the keypoint detector, local invariant descriptor, and the descriptor
# pipeline
detector = cv2.FeatureDetector_create("GFTT")
descriptor = RootSIFT()
dad = DetectAndDescribe(detector, descriptor)

# load the codebook vocabulary and initialize the bag-of-visual-words transformer
vocab = cPickle.loads(open(args["codebook"]).read())
bovw = BagOfVisualWords(vocab)

# load the classifier
model = cPickle.loads(open(args["model"]).read())

# loop over the image paths
for imagePath in paths.list_images(args["images"]):
	# load the image and prepare it from description
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = imutils.resize(gray, width=min(320, image.shape[1]))

	# describe the image and classify it
	(kps, descs) = dad.describe(gray)
	hist = bovw.describe(descs)
	hist /= hist.sum()
	prediction = model.predict(hist)[0]

	# show the prediction
	filename = imagePath[imagePath.rfind("/") + 1:]
	print("[PREDICTION] {}: {}".format(filename, prediction))
	cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
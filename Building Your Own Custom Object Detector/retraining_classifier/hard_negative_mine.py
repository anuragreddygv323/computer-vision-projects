# USAGE
# python hard_negative_mine.py --conf conf/cars.json

# import the necessary packages
from __future__ import print_function
from pyimagesearch.object_detection import ObjectDetector
from pyimagesearch.descriptors import HOG
from pyimagesearch.utils import dataset
from pyimagesearch.utils import Conf
from imutils import paths
import numpy as np
import progressbar
import argparse
import cPickle
import random
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
args = vars(ap.parse_args())

# load the configuration file and initialize the data list
conf = Conf(args["conf"])
data = []

# load the classifier, then initialize the Histogram of Oriented Gradients descriptor
# and the object detector
model = cPickle.loads(open(conf["classifier_path"]).read())
hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
	cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
od = ObjectDetector(model, hog)

# grab the set of distraction paths and randomly sample them
dstPaths = list(paths.list_images(conf["image_distractions"]))
dstPaths = random.sample(dstPaths, conf["hn_num_distraction_images"])

# setup the progress bar
widgets = ["Mining: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(dstPaths), widgets=widgets).start()

# loop over the distraction paths
for (i, imagePath) in enumerate(dstPaths):
	# load the image and convert it to grayscale
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect objects in the image
	(boxes, probs) = od.detect(gray, conf["window_dim"], winStep=conf["hn_window_step"],
		pyramidScale=conf["hn_pyramid_scale"], minProb=conf["hn_min_probability"])

	# loop over the bounding boxes
	for (prob, (startX, startY, endX, endY)) in zip(probs, boxes):
		# extract the ROI from the image, resize it to a known, canonical size, extract
		# HOG features from teh ROI, and finally update the data
		roi = cv2.resize(gray[startY:endY, startX:endX], tuple(conf["window_dim"]),
			interpolation=cv2.INTER_AREA)
		features = hog.describe(roi)
		data.append(np.hstack([[prob], features]))

	# update the progress bar
	pbar.update(i)

# sort the data points by confidence
pbar.finish()
print("[INFO] sorting by probability...")
data = np.array(data)
data = data[data[:, 0].argsort()[::-1]]

# dump the dataset to file
print("[INFO] dumping hard negatives to file...")
dataset.dump_dataset(data[:, 1:], [-1] * len(data), conf["features_path"], "hard_negatives",
	writeMethod="a")
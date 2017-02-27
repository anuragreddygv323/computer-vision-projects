# USAGE
# python train_detector.py --class stop_sign_images --annotations stop_sign_annotations \
#	--output output/stop_sign_detector.svm

# import the necessary packages
from __future__ import print_function
from imutils import paths
from scipy.io import loadmat
from skimage import io
import argparse
import dlib

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--class", required=True,
	help="Path to the CALTECH-101 class images")
ap.add_argument("-a", "--annotations", required=True,
	help="Path to the CALTECH-101 class annotations")
ap.add_argument("-o", "--output", required=True,
	help="Path to the output detector")
args = vars(ap.parse_args())

# grab the default training options for our HOG + Linear SVM detector initialize the
# list of images and bounding boxes used to train the classifier
print("[INFO] gathering images and bounding boxes...")
options = dlib.simple_object_detector_training_options()
images = []
boxes = []

# loop over the image paths
for imagePath in paths.list_images(args["class"]):
	# extract the image ID from the image path and load the annotations file
	imageID = imagePath[imagePath.rfind("/") + 1:].split("_")[1]
	imageID = imageID.replace(".jpg", "")
	p = "{}/annotation_{}.mat".format(args["annotations"], imageID)
	annotations = loadmat(p)["box_coord"]

	# loop over the annotations and add each annotation to the list of bounding
	# boxes
	bb = [dlib.rectangle(left=long(x), top=long(y), right=long(w), bottom=long(h))
			for (y, h, x, w) in annotations]
	boxes.append(bb)

	# add the image to the list of images
	images.append(io.imread(imagePath))

# train the object detector
print("[INFO] training detector...")
detector = dlib.train_simple_object_detector(images, boxes, options)

# dump the classifier to file
print("[INFO] dumping classifier to file...")
detector.save(args["output"])

# visualize the results of the detector
win = dlib.image_window()
win.set_image(detector)
dlib.hit_enter_to_continue()
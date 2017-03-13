# USAGE
# python sample_dataset.py --input ~/PyImageSearch/Datasets/caltech5 --output output/data --training-size 0.75

# import the necessary packages
from imutils import paths
import argparse
import random
import shutil
import glob
import os

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of image classes")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory to store training and testing images")
ap.add_argument("-t", "--training-size", type=float, default=0.75,
	help="% of images to use for training data")
args = vars(ap.parse_args())

# if the output directory exists, delete it
if os.path.exists(args["output"]):
	shutil.rmtree(args["output"])

# create the output directories
os.makedirs(args["output"])
os.makedirs("{}/training".format(args["output"]))
os.makedirs("{}/testing".format(args["output"]))

# loop over the image classies in the input directory
for labelPath in glob.glob(args["input"] + "/*"):
	# extract the label from the path and create the sub-directories for the label in
	# the output directory
	label = labelPath[labelPath.rfind("/") + 1:]
	os.makedirs("{}/training/{}".format(args["output"], label))
	os.makedirs("{}/testing/{}".format(args["output"], label))

	# grab the image paths for the current label and shuffle them
	imagePaths = list(paths.list_images(labelPath))
	random.shuffle(imagePaths)
	i = int(len(imagePaths) * args["training_size"])

	# loop over the randomly sampled training paths and copy them into the appropriate
	# output directory
	for imagePath in imagePaths[:i]:
		filename = imagePath[imagePath.rfind("/") + 1:]
		shutil.copy(imagePath, "{}/training/{}/{}".format(args["output"], label, filename))

	# loop over the randomly sampled testing paths and copy them into the appropriate
	# output directory
	for imagePath in imagePaths[i:]:
		filename = imagePath[imagePath.rfind("/") + 1:]
		shutil.copy(imagePath, "{}/testing/{}/{}".format(args["output"], label, filename))
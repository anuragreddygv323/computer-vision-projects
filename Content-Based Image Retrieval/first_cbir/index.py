# USAGE
# python index.py --dataset ../ukbench --index index.csv

# import the necessary packages
from __future__ import print_function
from pyimagesearch.cbir import HSVDescriptor
from imutils import paths
import progressbar
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required=True,
	help = "Path to where the features index will be stored")
args = vars(ap.parse_args())

# initialize the color descriptor and open the output index file for writing
desc = HSVDescriptor((4, 6, 3))
output = open(args["index"], "w")

# grab the list of image paths and initialize the progress bar
imagePaths = list(paths.list_images(args["dataset"]))
widgets = ["Indexing: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets)
pbar.start()

# loop over the image paths in the dataset directory
for (i, imagePath) in enumerate(imagePaths):
	# extract the image filename (i.e. the unique image ID) from the image
	# path, then load the image itself
	filename = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)

	# describe the image
	features = desc.describe(image)

	# write the features to our index file
	features = [str(x) for x in features]
	output.write("{},{}\n".format(filename, ",".join(features)))
	pbar.update(i)

# close the output index file
pbar.finish()
print("[INFO] indexed {} images".format(len(imagePaths)))
output.close()
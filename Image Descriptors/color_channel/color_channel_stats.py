# USAGE
# python color_channel_stats.py

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import paths
import numpy as np
import cv2

# grab the list of image paths and initialize the index to store the image filename
# and feature vector
imagePaths = sorted(list(paths.list_images("dinos")))
index = {}

# loop over the image paths
for imagePath in imagePaths:
	# load the image and extract the filename
	image = cv2.imread(imagePath)
	filename = imagePath[imagePath.rfind("/") + 1:]

	# extract the mean and standard deviation from each channel of the
	# BGR image, then update the index with the feature vector
	(means, stds) = cv2.meanStdDev(image)
	features = np.concatenate([means, stds]).flatten()
	index[filename] = features

# display the query image and grab the sorted keys of the index dictionary
query = cv2.imread(imagePaths[0])
cv2.imshow("Query (trex_01.png)", query)
keys = sorted(index.keys())

# loop over the filenames in the dictionary
for (i, k) in enumerate(keys):
	# if this is the query image, ignore it
	if k == "trex_01.png":
		continue

	# load the current image and compute the Euclidean distance between the
	# query image (i.e. the 1st image) and the current image
	image = cv2.imread(imagePaths[i])
	d = dist.euclidean(index["trex_01.png"], index[k])

	# display the distance between the query image and the current image
	cv2.putText(image, "%.2f" % (d), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
	cv2.imshow(k, image)

# wait for a keypress
cv2.waitKey(0)
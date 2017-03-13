# USAGE
# python visualize_centers.py --dataset ~/Desktop/ukbench_sample --features-db output/features.hdf5 \
#	--codebook output/vocab.cpickle --output output/vw_vis

# import the necessary packages
from __future__ import print_function
from pyimagesearch import ResultsMontage
from sklearn.metrics import pairwise
import numpy as np
import progressbar
import argparse
import cPickle
import h5py
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory of indexed images")
ap.add_argument("-f", "--features-db", required=True, help="Path to the features database")
ap.add_argument("-c", "--codebook", required=True, help="Path to the codebook")
ap.add_argument("-o", "--output", required=True, help="Path to output directory")
args = vars(ap.parse_args())

# load the codebook and open the features database
vocab = cPickle.loads(open(args["codebook"]).read())
featuresDB = h5py.File(args["features_db"], mode="r")
print("[INFO] starting distance computations...")

# initialize the visualizations dictionary and initialize the progress bar
vis = {i:[] for i in np.arange(0, len(vocab))}
widgets = ["Comparing: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=featuresDB["image_ids"].shape[0], widgets=widgets).start()

# loop over the image IDs
for (i, imageID) in enumerate(featuresDB["image_ids"]):
	# grab the rows for the features database and split them into keypoints and
	# feature vectors
	(start, end) = featuresDB["index"][i]
	rows = featuresDB["features"][start:end]
	(kps, descs) = (rows[:, :2], rows[:, 2:])

	# loop over each of the individual keypoints and feature vectors
	for (kp, features) in zip(kps, descs):
		# compute the distance between the feature vector and all clusters,
		# meaning that we'll have one distance value for each cluster
		D = pairwise.euclidean_distances(features.reshape(1, -1), Y=vocab)[0]

		# loop over the distances dictionary
		for j in np.arange(0, len(vocab)):
			# grab the set of top visualization results for the current
			# visual word and update the top reults with a tuple of the
			# distance, keypoint, and image ID
			topResults = vis.get(j)
			topResults.append((D[j], kp, imageID))

			# sort the top results list by their distance, keeping only
			# the best 16, then update the visualizations dictionary
			topResults = sorted(topResults, key=lambda r:r[0])[:16]
			vis[j] = topResults

	# update the progress bar
	pbar.update(i)

# close the features database
pbar.finish()
featuresDB.close()
print("[INFO] writing visualizations to file...")

# loop over the top results
for (vwID, results) in vis.items():
	# initialize the results montage
	montage = ResultsMontage((64, 64), 4, 16)

	# loop over the results
	for (_, (x, y), imageID) in results:
		# load the current image
		p = "{}/{}".format(args["dataset"], imageID)
		image = cv2.imread(p)
		(h, w) = image.shape[:2]

		# extract a 64x64 region surrounding the keypoint
		(startX, endX) = (max(0, x - 32), min(w, x + 32))
		(startY, endY) = (max(0, y - 32), min(h, y + 32))
		roi = image[startY:endY, startX:endX]

		# add the ROI to the montage
		montage.addResult(roi)

	# write the visualization to file
	p = "{}/vis_{}.jpg".format(args["output"], vwID)
	cv2.imwrite(p, cv2.cvtColor(montage.montage, cv2.COLOR_BGR2GRAY))
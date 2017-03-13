# USAGE
# python cluster_features.py --features-db output/features.hdf5 --codebook output/vocab.cpickle --clusters 1536 \
#	--percentage 0.25

# import the necessary packages
from __future__ import print_function
from pyimagesearch.ir import Vocabulary
import argparse
import cPickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features-db", required=True,
	help="Path to where the features database will be stored")
ap.add_argument("-c", "--codebook", required=True,
	help="Path to the output codebook")
ap.add_argument("-k", "--clusters", type=int, default=64,
	help="# of clusters to generate")
ap.add_argument("-p", "--percentage", type=float, default=0.25,
	help="Percentage of total features to use when clustering")
args = vars(ap.parse_args())

# create the visual words vocabulary
voc = Vocabulary(args["features_db"])
vocab = voc.fit(args["clusters"], args["percentage"])

# dump the clusters to file
print("[INFO] storing cluster centers...")
f = open(args["codebook"], "w")
f.write(cPickle.dumps(vocab))
f.close()
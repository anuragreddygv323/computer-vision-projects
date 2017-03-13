# USAGE
# python quantize_example.py

# import the necessary packages
from __future__ import print_function
from pyimagesearch.ir import BagOfVisualWords
from sklearn.metrics import pairwise
import numpy as np

# randomly generate the vocabulary/cluster centers along with the feature
# vectors -- we'll generate 10 feature vectos containing 6 real-valued
# entries, along with a codebook containing 3 'visual words'
np.random.seed(42)
vocab = np.random.uniform(size=(3, 6))
features = np.random.uniform(size=(10, 6))
print("[INFO] vocabulary:\n{}\n".format(vocab))
print("[INFO] features:\n{}\n".format(features))

# initialize our bag of visual words histogram -- it will contain 3 entries,
# one for each of the possible visual words
hist = np.zeros((3,), dtype="int32")

# loop over the inidividual feature vectors
for (i, f) in enumerate(features):
	# compute the Euclidean distance between the current feature vector
	# and the 3 visual words; then, find the index of the visual word
	# with the smallest distance
	D = pairwise.euclidean_distances(f.reshape(1, -1), Y=vocab)
	j = np.argmin(D)

	print("[INFO] Closest visual word to feature #{}: {}".format(i, j))
	hist[j] += 1
	print("[INFO] Updated histogram: {}".format(hist))

# this apply our `BagOfVisualWords` class and make this process super
# speedy
bovw = BagOfVisualWords(vocab, sparse=False)
hist = bovw.describe(features)
print("[INFO] BOVW histogram: {}".format(hist))
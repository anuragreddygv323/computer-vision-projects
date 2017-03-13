# import the necessary packages
import numpy as np

def chi2_distance(histA, histB, eps=1e-10):
	# compute the chi-squared distance
	d = 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))

	# return the chi-squared distance
	return d
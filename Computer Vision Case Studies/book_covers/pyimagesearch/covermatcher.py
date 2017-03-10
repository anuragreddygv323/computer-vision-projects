# import the necessary packages
import numpy as np
import cv2

class CoverMatcher:
	def __init__(self, descriptor, coverPaths):
		# store the descriptor and book cover paths
		self.descriptor = descriptor
		self.coverPaths = coverPaths

	def search(self, queryKps, queryDescs):
		# initialize the dictionary of results
		results = {}

		# loop over the book cover images
		for coverPath in self.coverPaths:
			# load the query image, convert it to grayscale, and extract
			# keypoints and descriptors
			cover = cv2.imread(coverPath)
			gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
			(kps, descs) = self.descriptor.describe(gray)

			# determine the number of matched, inlier keypoints,
			# then update the results
			score = self.match(queryKps, queryDescs, kps, descs)
			results[coverPath] = score

		# if matches were found, sort them
		if len(results) > 0:
			results = sorted([(v, k) for (k, v) in results.items() if v > 0],
				reverse = True)

		# return the results
		return results

	def match(self, kpsA, featuresA, kpsB, featuresB, ratio=0.7, minMatches=50):
		# compute the raw matches and initialize the list of actual
		# matches
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresB, featuresA, 2)
		matches = []

		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# check to see if there are enough matches to process
		if len(matches) > minMatches:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (i, _) in matches])
			ptsB = np.float32([kpsB[j] for (_, j) in matches])

			# compute the homography between the two sets of points
			# and compute the ratio of matched points
			(_, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)

			# return the ratio of the number of matched keypoints
			# to the total number of keypoints
			return float(status.sum()) / status.size

		# no matches were found
		return -1.0
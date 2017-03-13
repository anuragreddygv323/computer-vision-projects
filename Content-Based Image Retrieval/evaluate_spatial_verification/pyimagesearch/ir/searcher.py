# import the necessary packages
from searchresult import SearchResult
import numpy as np
import datetime
import dists
import h5py

class Searcher:
	def __init__(self, redisDB, bovwDBPath, featuresDBPath, idf=None,
		distanceMetric=dists.chi2_distance):
		# store the redis database reference, the idf array, and the distance
		# metric
		self.redisDB = redisDB
		self.idf = idf
		self.distanceMetric = distanceMetric

		# open both the bag-of-visual-words database and the features database
		# for reading
		self.bovwDB = h5py.File(bovwDBPath, mode="r")
		self.featuresDB = h5py.File(featuresDBPath, "r")

	def search(self, queryHist, numResults=10, maxCandidates=200):
		# start the timer to track how long the search took
		startTime = datetime.datetime.now()

		# determine the candidates and sort them in ascending order so they can
		# be read from the bag-of-visual-words database
		candidateIdxs = self.buildCandidates(queryHist, maxCandidates=maxCandidates)
		candidateIdxs.sort()

		# grab the histograms for the candidates from the bag-of-visual-words
		# database and initialize the results dictionary
		hists = self.bovwDB["bovw"][candidateIdxs]
		queryHist = queryHist.toarray()
		results = {}

		# if the inverse document frequency array has been supplied, multiply the
		# query by it
		if self.idf is not None:
			queryHist *= self.idf

		# loop over the histograms
		for (candidate, hist) in zip(candidateIdxs, hists):
			# if the inverse document frequency array has been supplied, multiply
			# the histogram by it
			if self.idf is not None:
				hist *= self.idf

			# compute the distance between the histograms and updated the results
			# dictionary
			d = self.distanceMetric(hist, queryHist)
			results[candidate] = d

		# sort the results, this time replacing the image indexes with the image
		# IDs themselves
		results = sorted([(v, self.featuresDB["image_ids"][k], k)
			for (k, v) in results.items()])
		results = results[:numResults]

		# return the search results
		return SearchResult(results, (datetime.datetime.now() - startTime).total_seconds())

	def buildCandidates(self, hist, maxCandidates):
		# initialize the redis pipeline
		p = self.redisDB.pipeline()

		# loop over the columns of the (sparse) matrix and create a query to
		# grab all images with an occurrence of the current visual word
		for i in hist.col:
			p.lrange("vw:{}".format(i), 0, -1)

		# execute the pipeline and initialize the candidates list
		pipelineResults = p.execute()
		candidates = []

		# loop over the pipeline results, extract the image index, and update
		# the candidates list
		for results in pipelineResults:
			results = [int(r) for r in results]
			candidates.extend(results)

		# count the occurrence of each of the canidates and sort in descending
		# order
		(imageIdxs, counts) = np.unique(candidates, return_counts=True)
		imageIdxs = [i for (c, i) in sorted(zip(counts, imageIdxs), reverse=True)]

		# return the image indexes of the candidates
		return imageIdxs[:maxCandidates]

	def finish(self):
		# close the bag-of-visual-words database and the features database
		self.bovwDB.close()
		self.featuresDB.close()
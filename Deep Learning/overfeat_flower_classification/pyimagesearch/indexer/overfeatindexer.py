# import the necessary packages
from baseindexer import BaseIndexer
import numpy as np
import h5py

class OverfeatIndexer(BaseIndexer):
	def __init__(self, dbPath, estNumImages=500, maxBufferSize=5000, dbResizeFactor=2,
		verbose=True):
		# call the parent constructor
		super(OverfeatIndexer, self).__init__(dbPath, estNumImages=estNumImages,
			maxBufferSize=maxBufferSize, dbResizeFactor=dbResizeFactor,
			verbose=verbose)

		# open the HDF5 database for writing and initialize the datasets within
		# the group
		self.db = h5py.File(self.dbPath, mode="w")
		self.imageIDDB = None
		self.featuresDB = None

		# initialize the image IDs buffer and features buffer
		self.imageIDBuffer = []
		self.featuresBuffer = None

		# initialize the total number of features in the buffer along with the
		# indexes dictionary
		self.totalFeatures = 0
		self.idxs = {"index": 0, "features": 0}

	def add(self, imageID, features):
		# update the image IDs buffer and features buffer, followed by
		# incrementing the feature count
		self.imageIDBuffer.append(imageID)
		self.featuresBuffer = BaseIndexer.featureStack(features, self.featuresBuffer)
		self.totalFeatures += 1

		# check to see if we have reached the maximum buffer size
		if self.totalFeatures >= self.maxBufferSize:
			# if the databases have not been created yet, create them
			if None in (self.imageIDDB, self.featuresDB):
				self._debug("initial buffer full")
				self._createDatasets()

			# write the buffers to file
			self._writeBuffers()

	def _createDatasets(self):
		# grab the feature vector size
		fvectorSize = self.featuresBuffer.shape[1]

		# initialize the datasets
		self._debug("creating datasets...")
		self.imageIDDB = self.db.create_dataset("image_ids", (self.estNumImages,),
			maxshape=(None,), dtype=h5py.special_dtype(vlen=unicode))
		self.featuresDB = self.db.create_dataset("features",
			(self.estNumImages, fvectorSize), maxshape=(None, fvectorSize),
			dtype="float")

	def _writeBuffers(self):
		# write the buffers to disk
		self._writeBuffer(self.imageIDDB, "image_ids", self.imageIDBuffer,
			"index")
		self._writeBuffer(self.featuresDB, "features", self.featuresBuffer,
			"features")

		# increment the indexes
		self.idxs["index"] += len(self.imageIDBuffer)
		self.idxs["features"] += self.totalFeatures

		# reset the buffers and feature counts
		self.imageIDBuffer = []
		self.featuresBuffer = None
		self.totalFeatures = 0

	def finish(self):
		# if the databases have not been initialized, then the original
		# buffers were never filled up
		if None in (self.imageIDDB, self.featuresDB):
			self._debug("minimum init buffer not reached", msgType="[WARN]")
			self._createDatasets()

		# write any unempty buffers to file
		self._debug("writing un-empty buffers...")
		self._writeBuffers()

		# compact datasets
		self._debug("compacting datasets...")
		self._resizeDataset(self.imageIDDB, "image_ids", finished=self.idxs["index"])
		self._resizeDataset(self.featuresDB, "features", finished=self.idxs["features"])

		# close the database
		self.db.close()
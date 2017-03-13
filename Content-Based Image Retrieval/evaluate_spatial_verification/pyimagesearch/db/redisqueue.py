# import the necessary packages
import numpy as np

class RedisQueue:
	def __init__(self, redisDB):
		# store the redis database object
		self.redisDB = redisDB

	def add(self, imageIdx, hist):
		# initialize the redis pipeline
		p = self.redisDB.pipeline()

		# loop over all non-zero entries for the histogram, creating a
		# (visual word -> document) record for each visual word in the
		# histogram
		for i in np.where(hist > 0)[0]:
			p.rpush("vw:{}".format(i), imageIdx)

		# execute the pipeline
		p.execute()

	def finish(self):
		# save the state of the Redis database
		self.redisDB.save()
# import the necessary packages
from sklearn_theano.feature_extraction.overfeat import SMALL_NETWORK_FILTER_SHAPES
from sklearn_theano.feature_extraction import OverfeatTransformer

class OverfeatExtractor:
	def __init__(self, layerNum):
		# store the layer number and initialize the Overfeat transformer
		self.layerNum = layerNum
		self.of = OverfeatTransformer(output_layers=[layerNum])

	def describe(self, data):
		# apply the Overfeat transfrom to the images
		return self.of.transform(data)

	def getFeatureDim(self):
		# return the feature dimensionality from the supplied layer
		return SMALL_NETWORK_FILTER_SHAPES[self.layerNum][0]
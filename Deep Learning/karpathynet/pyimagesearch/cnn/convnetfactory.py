# import the necessary packages
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.models import Sequential

class ConvNetFactory:
	def __init__(self):
		pass

	@staticmethod
	def build(name, *args, **kargs):
		# define the network (i.e., string => function) mappings
		mappings = {
			"shallownet": ConvNetFactory.ShallowNet,
			"lenet": ConvNetFactory.LeNet,
			"karpathynet": ConvNetFactory.KarpathyNet,
			"minivggnet": ConvNetFactory.MiniVGGNet}

		# grab the builder function from the mappings dictionary
		builder = mappings.get(name, None)

		# if the builder is None, then there is not a function that can be used
		# to build to the network, so return None
		if builder is None:
			return None

		# otherwise, build the network architecture
		return builder(*args, **kargs)

	@staticmethod
	def ShallowNet(numChannels, imgRows, imgCols, numClasses, **kwargs):
		# initialzie the model
		model = Sequential()

		# define the first (and only) CONV => RELU layer
		model.add(Convolution2D(32, 3, 3, border_mode="same",
			input_shape=(numChannels, imgRows, imgCols)))
		model.add(Activation("relu"))

		# add a FC layer followed by the soft-max classifier
		model.add(Flatten())
		model.add(Dense(numClasses))
		model.add(Activation("softmax"))

		# return the network architecture
		return model

	@staticmethod
	def LeNet(numChannels, imgRows, imgCols, numClasses, activation="tanh", **kwargs):
		# initialize the model
		model = Sequential()

		# define the first set of CONV => ACTIVATION => POOL layers
		model.add(Convolution2D(20, 5, 5, border_mode="same",
			input_shape=(numChannels, imgRows, imgCols)))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# define the second set of CONV => ACTIVATION => POOL layers
		model.add(Convolution2D(50, 5, 5, border_mode="same"))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# define the first FC => ACTIVATION layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation(activation))

		# define the second FC layer
		model.add(Dense(numClasses))

		# lastly, define the soft-max classifier
		model.add(Activation("softmax"))

		# return the network architecture
		return model

	@staticmethod
	def KarpathyNet(numChannels, imgRows, imgCols, numClasses, dropout=False, **kwargs):
		# initialize the model
		model = Sequential()

		# define the first set of CONV => RELU => POOL layers
		model.add(Convolution2D(16, 5, 5, border_mode="same",
			input_shape=(numChannels, imgRows, imgCols)))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# check to see if dropout should be applied to reduce overfitting
		if dropout:
			model.add(Dropout(0.25))

		# define the second set of CONV => RELU => POOL layers
		model.add(Convolution2D(20, 5, 5, border_mode="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# check to see if dropout should be applied to reduce overfitting
		if dropout:
			model.add(Dropout(0.25))

		# define the third set of CONV => RELU => POOL layers
		model.add(Convolution2D(20, 5, 5, border_mode="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# check to see if dropout should be applied to reduce overfitting
		if dropout:
			model.add(Dropout(0.5))

		# define the soft-max classifier
		model.add(Flatten())
		model.add(Dense(numClasses))
		model.add(Activation("softmax"))

		# return the network architecture
		return model

	@staticmethod
	def MiniVGGNet(numChannels, imgRows, imgCols, numClasses, dropout=False, **kwargs):
		pass
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class SmallerVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		# The classes equals number of classes which
        # will affect the last layer of our
        # model
        model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
        # depth = number of channels
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

        # Convolution layer has 32 filters with a
        # 3 x 3 kernel. RELU = the activation function
        # followed by batch normalization.

        # Our Pool layer uses a 3 by 3 pool size to reduce
        # spatial demensions quickly from 96 by 96 to 32 by 32.

        # Using dropout, which works by randomly disconnecting
        # nodes from the current layer to the next layer. This helps
        # by naturally introduce redundancy into the model.
        # No single node is responsible for predicting a certain class,
        # object, edge or corner.

        # Stacking multiple conv and relu layers together (prior to
        # reducing spatial dimensions of the volume) allows us to
        # learn a richer set of features.

        # CONV => RELU => POOL
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3)))
		model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
		model.add(Conv2D(64, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
		model.add(Conv2D(128, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model

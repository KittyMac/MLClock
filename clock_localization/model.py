from __future__ import division

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from keras.optimizers import SGD
import os

INCLUDE_SECONDS_HAND = False
MODEL_H5_NAME = "clock.h5"
MODEL_COREML_NAME = "../ios/MLclock/Assets/main/clock.mlmodel"
IMG_SIZE = [128,128,1]

def doesModelExist():
	return os.path.isfile(MODEL_H5_NAME)

def createModel(loadFromDisk):

	model = Sequential()

	# feature extractor
	model.add(Conv2D(16, (3, 3), input_shape=(IMG_SIZE[1], IMG_SIZE[0], IMG_SIZE[2])))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.1))
	
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3, 3)))
	model.add(Dropout(0.1))
	
	# yolo part
	model.add(Conv2D(16, (3, 3)))
	model.add(MaxPooling2D(pool_size=(3, 3)))
	
	model.add(Conv2D(8, (3, 3)))
	model.add(MaxPooling2D(pool_size=(3, 3)))
	
	model.add(Conv2D(4, (1, 1), padding="same"))
	print(model.summary())
	model.add(Reshape((4,)))
	

	model.compile(loss='mean_squared_error', optimizer="rmsprop", metrics=['accuracy'])

	print(model.summary())
	
	if loadFromDisk and os.path.isfile(MODEL_H5_NAME):
		model.load_weights(MODEL_H5_NAME)
	
	return model
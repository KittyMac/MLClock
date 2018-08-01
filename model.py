from __future__ import division

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from keras.optimizers import SGD
import os

MODEL_H5_NAME = "clock.h5"
MODEL_COREML_NAME = "clock.mlmodel"
IMG_SIZE = [128,128,1]

def doesModelExist():
	return os.path.isfile(MODEL_H5_NAME)

def createModel(loadFromDisk):

	model = Sequential()

	model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_SIZE[1], IMG_SIZE[0], IMG_SIZE[2]), activation='relu'))
	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(784, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='mse', optimizer="adadelta")

	print(model.summary())
	
	if loadFromDisk and os.path.isfile(MODEL_H5_NAME):
		model.load_weights(MODEL_H5_NAME)
	
	return model
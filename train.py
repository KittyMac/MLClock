from __future__ import division

from keras import backend as keras

from keras.preprocessing import sequence
import numpy as np
import coremltools
import model
import data
import json
import operator
import keras.callbacks
import random
import time
import sys
import math

import signal
import time

def Learn():
	
	# 1. create the model
	print("creating the model")
	_model = model.createModel(True)

	# 2. train the model
	print("initializing the generator")
	batch_size = 1
	generator = data.ClockGenerator()
	
	iterations = 50000
		
	print("beginning training")
	i = 0
	while True:
		
		n = int(12 + random.random() * 360)
		print(i)
		Train(generator,_model,n)
		i += n
		
		if i >= iterations:
			break
				
	
	_model.save(model.MODEL_H5_NAME)



def Train(generator,_model,n):
	
	train,label = generator.generateClockFaces(n)
	
	batch_size = 32
	if n < batch_size:
		batch_size = n
	
	_model.fit(train,label,batch_size=batch_size,shuffle=True,validation_split=0.2,epochs=1,verbose=1)


if __name__ == '__main__':
	Learn()
	
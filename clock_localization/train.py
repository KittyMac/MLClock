from __future__ import division

import sys
sys.path.insert(0, '../')

from keras import backend as keras

from keras.preprocessing import sequence
from dateutil import parser
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
import coremltools

######
# allows us to used ctrl-c to end gracefully instead of just dying
######
class SignalHandler:
  stop_processing = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.stop_processing = True
######

def Learn():
	
	random.seed(90021)
	
	# 1. create the model
	print("creating the model")
	_model = model.createModel(True)

	# 2. train the model
	print("initializing the generator")
	batch_size = 1
	generator = data.ClockGenerator(model.IMG_SIZE,model.INCLUDE_SECONDS_HAND,0.5)
	generator.shakeVariance = 0
	
	iterations = 50000
		
	print("beginning training")
	handler = SignalHandler()
	i = 0
	while True:
		
		if handler.stop_processing:
			break
		
		#n = int(random.random() * 43200)
		n = 25000
		print(i)
		Train(generator,_model,n)
		i += n
		
		if i >= iterations:
			break
				
	
	_model.save(model.MODEL_H5_NAME)
	

def Convert():
	output_labels = []
	output_labels.append("notclock")
	output_labels.append("clock")
		
	coreml_model = coremltools.converters.keras.convert(model.MODEL_H5_NAME,input_names='image',image_input_names='image',class_labels=output_labels, image_scale=1/255.0)
	coreml_model.author = 'Rocco Bowling'   
	coreml_model.short_description = 'is the image good clock face to send to the time detector'
	coreml_model.input_description['image'] = 'image of the clock face'
	coreml_model.save(model.MODEL_COREML_NAME)


def Train(generator,_model,n):
	
	train,label = generator.generateClockFaces(n)
	label = FixLabels(label)
	
	batch_size = 32
	if n < batch_size:
		batch_size = n
	
	_model.fit(train,label,batch_size=batch_size,shuffle=True,epochs=1,verbose=1)

def Test():
	_model = model.createModel(True)
	
	generator = data.ClockGenerator(model.IMG_SIZE,model.INCLUDE_SECONDS_HAND,0.5)
	generator.shakeVariance = 0
	
	train,label = generator.generateClockFaces(12*60*60)
	label = FixLabels(label)
	
	results = _model.predict(train)
	
	
	correct = 0
	for i in range(0,len(label)):
		if np.argmax(label[i]) == np.argmax(results[i]):
			correct += 1
	print("correct", correct, "total", len(label))
	

def FixLabels(label):
	newLabels = np.zeros((len(label),2), dtype='float32')
	for i in range(0,len(label)):
		if label[i][0] == 1:
			newLabels[i][0] = 1
			newLabels[i][1] = 0
		else:
			newLabels[i][0] = 0
			newLabels[i][1] = 1
	return newLabels

if __name__ == '__main__':
	if sys.argv >= 2:
		if sys.argv[1] == "test":
			Test()
		elif sys.argv[1] == "learn":
			Learn()
		elif sys.argv[1] == "convert":
			Convert()
		else:
			Test2(sys.argv[2])
	else:
		Test()
	
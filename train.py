from __future__ import division

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
	
	# 1. create the model
	print("creating the model")
	_model = model.createModel(True)

	# 2. train the model
	print("initializing the generator")
	batch_size = 1
	generator = data.ClockGenerator()
	
	iterations = 50000
		
	print("beginning training")
	handler = SignalHandler()
	i = 0
	while True:
		
		if handler.stop_processing:
			break
		
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
	
	_model.fit(train,label,batch_size=batch_size,shuffle=True,epochs=1,verbose=1)

def Test():
	_model = model.createModel(True)
	
	generator = data.ClockGenerator()
	
	train,label = generator.generateClockFaces(720)
	
	results = _model.predict(train)
	
	
	correct = 0
	for i in range(0,len(label)):
		expected = generator.convertOutputToTime(label[i])
		predicted = generator.convertOutputToTime(results[i])
		if expected == predicted:
			correct += 1
		print("expected", expected, "predicted", predicted)
	print("correct", correct, "total", len(label))
	

def Test2(timeAsString):
	parsedTime = parser.parse(timeAsString)
	
	_model = model.createModel(True)
	
	generator = data.ClockGenerator()
	
	train,label = generator.generateClockFace(parsedTime.hour, parsedTime.minute)
	results = _model.predict(train)
	
	for i in range(0,len(label)):
		filepath = '/tmp/clock_%s.png' % (generator.convertOutputToTime(results[i]))
		generator.saveImageToFile(train[i], filepath)
		print("expected", generator.convertOutputToTime(label[i]), "predicted", generator.convertOutputToTime(results[i]), "file", filepath)
	

if __name__ == '__main__':
	if sys.argv >= 2:
		if sys.argv[1] == "test":
			Test()
		elif sys.argv[1] == "learn":
			Learn()
		else:
			Test2(sys.argv[2])
	else:
		Test()
	
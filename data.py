from __future__ import division

from keras import backend as keras

from keras.preprocessing import sequence
import numpy as np
import os
import coremltools
import model
import json
import operator
import keras.callbacks
import random
import time
import math
import model
from model import IMG_SIZE
from PIL import Image,ImageDraw

import signal
import time

def frange(start, stop, step):
	i = start
	while i < stop:
		yield i
		i += step

class ClockGenerator(keras.utils.Sequence):
	
	def __init__(self):
		self.generated_turns = []
			
	def __len__(self):
		return 1
	
	def rotate(self, point, angle):
	    px, py = point
	    return math.cos(angle) * (px) - math.sin(angle) * (py), math.sin(angle) * (px) + math.cos(angle) * (py)
		
	def generateClockFace(self,hourHandAngle):
		img = Image.new('RGB', (IMG_SIZE[1], IMG_SIZE[0]), (255, 255, 255, 255))
		
		draw = ImageDraw.Draw(img)
		
		origin = (IMG_SIZE[1]/2,IMG_SIZE[0]/2)
		hand = self.rotate((IMG_SIZE[1]*0.4,0), math.radians(hourHandAngle-90))
		draw.line((origin[0], origin[1], origin[0]+hand[0], origin[1]+hand[1]), width=3, fill=(0,0,0))
		
		return img.convert('L')
	
	def generateClockFaces(self, num):
		# generate num images which represent one full rotation of the clock hand
		start = 0
		end = 360
		delta = 360 / num
		
		input_images = np.zeros((num,IMG_SIZE[1],IMG_SIZE[0],IMG_SIZE[2]), dtype='float32')
		output_values = np.zeros((num,1), dtype='float32')
		
		for idx in range(0,num):
			angle = start + delta * idx
			img = self.generateClockFace(angle)
			np.copyto(input_images[idx],np.array(img).reshape(IMG_SIZE[1],IMG_SIZE[0],IMG_SIZE[2]))
			output_values[idx] = angle / 360
		
		return input_images,output_values
			


if __name__ == '__main__':
	generator = ClockGenerator()
	
	input,output = generator.generateClockFaces(24)
	for n in range(0,len(input)):
		(Image.fromarray(input[n].reshape(IMG_SIZE[1],IMG_SIZE[0])).convert("L")).save('/tmp/clock_%f.png' % (output[n]))
	
	
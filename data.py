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
from model import INCLUDE_SECONDS_HAND
from model import IMG_SIZE
from PIL import Image,ImageDraw

import signal
import time

faceImg = Image.open('meta/face.png', 'r').convert('RGBA')
secondImg = Image.open('meta/second.png', 'r').convert('RGBA')
minuteImg = Image.open('meta/minute.png', 'r').convert('RGBA')
hourImg = Image.open('meta/hour.png', 'r').convert('RGBA')

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
		
	def generateClockImage(self,hourHandAngle,minuteHandAngle,secondHandAngle):
		
		# simulated real clock with photo drawing
		img = Image.new('RGBA', (IMG_SIZE[1], IMG_SIZE[0]), (255, 255, 255, 255))
		
		if INCLUDE_SECONDS_HAND:
			secondImgRotated = secondImg.rotate( (-secondHandAngle)+90 )
		hourImgRotated = hourImg.rotate( (-hourHandAngle)+90 )
		minuteImgRotated = minuteImg.rotate( (-minuteHandAngle)+90 )
		
		img.paste(faceImg, (0,0), faceImg)
		if INCLUDE_SECONDS_HAND:
			img.paste(secondImgRotated, (0,0), secondImgRotated)
		img.paste(hourImgRotated, (0,0), hourImgRotated)
		img.paste(minuteImgRotated, (0,0), minuteImgRotated)
		
		return img.convert('L')
		
		# simulated clock with polygon drawing
		'''
		img = Image.new('RGB', (IMG_SIZE[1], IMG_SIZE[0]), (255, 255, 255, 255))
		
		draw = ImageDraw.Draw(img)
		
		origin = (IMG_SIZE[1]/2,IMG_SIZE[0]/2)
		hourHand = self.rotate((IMG_SIZE[1]*0.25,0), math.radians(hourHandAngle-90))
		draw.line((origin[0], origin[1], origin[0]+hourHand[0], origin[1]+hourHand[1]), width=4, fill=(0,0,0))
		
		minuteHand = self.rotate((IMG_SIZE[1]*0.4,0), math.radians(minuteHandAngle-90))
		draw.line((origin[0], origin[1], origin[0]+minuteHand[0], origin[1]+minuteHand[1]), width=2, fill=(0,0,0))
		
		secondHand = self.rotate((IMG_SIZE[1]*0.5,0), math.radians(secondHandAngle-90))
		draw.line((origin[0], origin[1], origin[0]+secondHand[0], origin[1]+secondHand[1]), width=1, fill=(100,100,100))
		
		return img.convert('L')
		'''
	
	def generateClockFace(self, hours, minutes):
		start = 0
		end = 360
		delta = 360
		
		input_images = np.zeros((1,IMG_SIZE[1],IMG_SIZE[0],IMG_SIZE[2]), dtype='float32')
		
		if INCLUDE_SECONDS_HAND:
			output_values = np.zeros((1,12+60+60), dtype='float32')
		else:
			output_values = np.zeros((1,12+60), dtype='float32')
		
		combined_seconds = hours * 60 + minutes * 60 + seconds
		
		hour_normalized = (combined_seconds / 3600) / 12
		minute_normalized = ((combined_seconds / 60) % 60) / 60
		second_normalized = (combined_seconds % 60) / 60
		
		hour_idx = int(hour_normalized * 12)
		minute_idx = int(minute_normalized * 60)
		second_idx = int(second_normalized * 60)
		
		img = self.generateClockImage(
			start + (end-start) * hour_normalized, 
			start + (end-start) * minute_normalized,
			start + (end-start) * second_normalized)
		np.copyto(input_images[0],np.array(img).reshape(IMG_SIZE[1],IMG_SIZE[0],IMG_SIZE[2]))
		input_images[0] /= 255.0
		
		output_values[0][hour_idx] = 1
		output_values[0][minute_idx+12] = 1
		if INCLUDE_SECONDS_HAND:
			output_values[0][second_idx+12+60] = 1
		
		return input_images,output_values
		
	
	def generateClockFaces(self, num):
		# generate num images which represent one full rotation of the clock hand
		start = 0
		end = 360
		delta = 360 / num
		
		input_images = np.zeros((num,IMG_SIZE[1],IMG_SIZE[0],IMG_SIZE[2]), dtype='float32')
		
		if INCLUDE_SECONDS_HAND:
			output_values = np.zeros((num,12+60+60), dtype='float32')
		else:
			output_values = np.zeros((num,12+60), dtype='float32')
		
		for idx in range(0,num):
			
			combined_seconds = int((idx / num) * 43200)
					
			hour_normalized = (combined_seconds / 3600) / 12
			minute_normalized = ((combined_seconds / 60) % 60) / 60
			second_normalized = (combined_seconds % 60) / 60
		
			hour_idx = int(hour_normalized * 12)
			minute_idx = int(minute_normalized * 60)
			second_idx = int(second_normalized * 60)
						
			img = self.generateClockImage(
				start + (end-start) * hour_normalized, 
				start + (end-start) * minute_normalized,
				start + (end-start) * second_normalized)
			np.copyto(input_images[idx],np.array(img).reshape(IMG_SIZE[1],IMG_SIZE[0],IMG_SIZE[2]))
			input_images[idx] /= 255.0
			
			output_values[idx][hour_idx] = 1
			output_values[idx][minute_idx+12] = 1
			if INCLUDE_SECONDS_HAND:
				output_values[idx][second_idx+12+60] = 1
				
		return input_images,output_values
	
	def convertOutputToTime(self,output):
		hour = np.argmax(output[0:12])
		minute = np.argmax(output[12:72])
		if INCLUDE_SECONDS_HAND:
			second = np.argmax(output[72:132])
			
		if hour == 0:
			hour = 12
		
		if INCLUDE_SECONDS_HAND:
			return "%02d.%02d.%02d" % (hour,minute,second)
		return "%02d.%02d" % (hour,minute)		
	
	def saveImageToFile(self,img,filepath):
		img = img.reshape(IMG_SIZE[1],IMG_SIZE[0]) * 255.0
		Image.fromarray(img).convert("L").save(filepath)


if __name__ == '__main__':
	generator = ClockGenerator()
	
	input,output = generator.generateClockFaces(13)	
	for n in range(0,len(input)):
		generator.saveImageToFile(input[n], '/tmp/clock_%s.png' % (generator.convertOutputToTime(output[n])))
	
	
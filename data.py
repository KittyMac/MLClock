from __future__ import division

from keras import backend as keras

from keras.preprocessing import sequence
import numpy as np
import os
import coremltools
import json
import operator
import keras.callbacks
import random
import time
import math
from PIL import Image,ImageDraw

import signal
import time

META_PATH = '../meta'

def frange(start, stop, step):
	i = start
	while i < stop:
		yield i
		i += step

class ClockGenerator(keras.utils.Sequence):
	
	def __init__(self,imgSize,includeSecondsHands,notClockFaceThreshold):
		global faceImg, secondImg, minuteImg, hourImg
		
		faceImg = Image.open('%s/face.png' % META_PATH, 'r').convert('RGBA')
		secondImg = Image.open('%s/second.png' % META_PATH, 'r').convert('RGBA')
		minuteImg = Image.open('%s/minute.png' % META_PATH, 'r').convert('RGBA')
		hourImg = Image.open('%s/hour.png' % META_PATH, 'r').convert('RGBA')
		
		self.imgSize = imgSize
		self.includeSecondsHands = includeSecondsHands
		self.notClockFaceThreshold = notClockFaceThreshold
		self.generated_turns = []
			
	def __len__(self):
		return 1
	
	def rotate(self, point, angle):
	    px, py = point
	    return math.cos(angle) * (px) - math.sin(angle) * (py), math.sin(angle) * (px) + math.cos(angle) * (py)
		
	
	def generateNotClockImage(self):
		# TODO: make this better at generating random images
		img = Image.new('RGBA', (self.imgSize[1], self.imgSize[0]), 
			(int(30 + random.random() * 210),
			int(30 + random.random() * 210),
			int(30 + random.random() * 210)))
		
		randomImg1 = Image.open('%s/random/%d.png' % (META_PATH, int(random.random()*230)), 'r').convert('RGBA')
		randomImg2 = Image.open('%s/random/%d.png' % (META_PATH, int(random.random()*230)), 'r').convert('RGBA')
		
		img.paste(randomImg1, (0,0), img)
		img.paste(randomImg2, (0,0), img)
		
		return img.convert('L')
		
	def generateClockImage(self,hourHandAngle,minuteHandAngle,secondHandAngle):
		
		variance = 4
		
		offset = (int(random.random() * variance - variance / 2), int(random.random() * variance - variance / 2))
		rotation_offset = (random.random() * variance - variance / 2) * 2
		
		# simulated real clock with photo drawing
		img = Image.new('RGBA', (self.imgSize[1], self.imgSize[0]), (int(127 + random.random() * 128),
			int(127 + random.random() * 128),
			int(127 + random.random() * 128)))
		
		if self.includeSecondsHands:
			secondImgRotated = secondImg.rotate( (-secondHandAngle)+90+rotation_offset )
		hourImgRotated = hourImg.rotate( (-hourHandAngle)+90+rotation_offset )
		minuteImgRotated = minuteImg.rotate( (-minuteHandAngle)+90+rotation_offset )
		
		faceImgRotated = faceImg.rotate( rotation_offset )
		
		img.paste(faceImgRotated, offset, faceImgRotated)
		if self.includeSecondsHands:
			img.paste(secondImgRotated, offset, secondImgRotated)
		img.paste(hourImgRotated, offset, hourImgRotated)
		img.paste(minuteImgRotated, offset, minuteImgRotated)
		
		'''
		filter = Image.new('RGBA', (self.imgSize[1], self.imgSize[0]), (int(127 + random.random() * 128),
			int(127 + random.random() * 128),
			int(127 + random.random() * 128),
			int(127 + random.random() * 128)))
		
		img.paste(filter, (0,0), filter)
		'''
		
		return img.convert('L')
		
		# simulated clock with polygon drawing
		'''
		img = Image.new('RGB', (self.imgSize[1], self.imgSize[0]), (255, 255, 255, 255))
		
		draw = ImageDraw.Draw(img)
		
		origin = (self.imgSize[1]/2,self.imgSize[0]/2)
		hourHand = self.rotate((self.imgSize[1]*0.25,0), math.radians(hourHandAngle-90))
		draw.line((origin[0], origin[1], origin[0]+hourHand[0], origin[1]+hourHand[1]), width=4, fill=(0,0,0))
		
		minuteHand = self.rotate((self.imgSize[1]*0.4,0), math.radians(minuteHandAngle-90))
		draw.line((origin[0], origin[1], origin[0]+minuteHand[0], origin[1]+minuteHand[1]), width=2, fill=(0,0,0))
		
		secondHand = self.rotate((self.imgSize[1]*0.5,0), math.radians(secondHandAngle-90))
		draw.line((origin[0], origin[1], origin[0]+secondHand[0], origin[1]+secondHand[1]), width=1, fill=(100,100,100))
		
		return img.convert('L')
		'''
	
	def generateClockFace(self, hours, minutes):
		start = 0
		end = 360
		delta = 360
		
		input_images = np.zeros((1,self.imgSize[1],self.imgSize[0],self.imgSize[2]), dtype='float32')
		
		if self.includeSecondsHands:
			output_values = np.zeros((1,1+12+60+60), dtype='float32')
		else:
			output_values = np.zeros((1,1+12+60), dtype='float32')
		
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
		np.copyto(input_images[0],np.array(img).reshape(self.imgSize[1],self.imgSize[0],self.imgSize[2]))
		input_images[0] /= 255.0
		
		output_values[0][hour_idx+1] = 1
		output_values[0][minute_idx+12+1] = 1
		if self.includeSecondsHands:
			output_values[0][second_idx+12+60+1] = 1
		
		return input_images,output_values
		
	
	def generateClockFaces(self, num):
		# generate num images which represent one full rotation of the clock hand
		start = 0
		end = 360
		delta = 360 / num
		
		input_images = np.zeros((num,self.imgSize[1],self.imgSize[0],self.imgSize[2]), dtype='float32')
		
		if self.includeSecondsHands:
			output_values = np.zeros((num,1+12+60+60), dtype='float32')
		else:
			output_values = np.zeros((num,1+12+60), dtype='float32')
		
		for idx in range(0,num):
			
			# not clock faces should be all zeros
			if random.random() < self.notClockFaceThreshold:
				img = self.generateNotClockImage()
				np.copyto(input_images[idx],np.array(img).reshape(self.imgSize[1],self.imgSize[0],self.imgSize[2]))
				input_images[idx] /= 255.0
				output_values[idx][0] = 1
				continue
			
			
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
			np.copyto(input_images[idx],np.array(img).reshape(self.imgSize[1],self.imgSize[0],self.imgSize[2]))
			input_images[idx] /= 255.0
			
			output_values[idx][hour_idx+1] = 1
			output_values[idx][minute_idx+12+1] = 1
			if self.includeSecondsHands:
				output_values[idx][second_idx+12+60+1] = 1
				
		return input_images,output_values
		
	
	def convertOutputToTime(self,output):
		
		if output[0] > 0.5:
			if self.includeSecondsHands:
				return "00.00.00"
			return "00.00"
		
		hour = np.argmax(output[1:13])
		minute = np.argmax(output[13:73])
		if self.includeSecondsHands:
			second = np.argmax(output[73:133])
			
		if hour == 0:
			hour = 12
		
		if self.includeSecondsHands:
			return "%02d.%02d.%02d" % (hour,minute,second)
		return "%02d.%02d" % (hour,minute)
	
	def saveImageToFile(self,img,filepath):
		img = img.reshape(self.imgSize[1],self.imgSize[0]) * 255.0
		Image.fromarray(img).convert("L").save(filepath)


if __name__ == '__main__':
	
	META_PATH = "./meta"
	
	generator = ClockGenerator([128,128,1],True,0.5)
	
	input,output = generator.generateClockFaces(13)	
	for n in range(0,len(input)):
		generator.saveImageToFile(input[n], '/tmp/clock_%s.png' % (generator.convertOutputToTime(output[n])))
	
	
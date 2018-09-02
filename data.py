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
		
		self.shakeVariance = 4
		self.randomImages = None
		self.imgSize = imgSize
		self.includeSecondsHands = includeSecondsHands
		self.notClockFaceThreshold = notClockFaceThreshold
		self.generated_turns = []
			
	def __len__(self):
		return 1
	
	def rotate(self, point, angle):
	    px, py = point
	    return math.cos(angle) * (px) - math.sin(angle) * (py), math.sin(angle) * (px) + math.cos(angle) * (py)
	
	
	def getRandomImage(self,idx):
		if self.randomImages == None:
			self.randomImages = []
			for i in range(0,236):
				self.randomImages.append(Image.open('%s/random/%d.png' % (META_PATH, i), 'r').convert('RGBA').resize((self.imgSize[1],self.imgSize[0]), Image.ANTIALIAS))
		return self.randomImages[idx]
	
	def generateNotClockImage(self):
		# TODO: make this better at generating random images
		img = Image.new('RGBA', (self.imgSize[1], self.imgSize[0]), 
			(int(30 + random.random() * 210),
			int(30 + random.random() * 210),
			int(30 + random.random() * 210),
			int(30 + random.random() * 210)))
		
		# allow empty images through
		if random.random() > 0.2:
			randomImg1 = self.getRandomImage(int(random.random()*230))
			randomImg2 = self.getRandomImage(int(random.random()*230))
		
			img.paste(randomImg1, (0,0), img)
			if random.random() > 0.5:
				img.paste(randomImg2, (0,0), img)
		
		return img
		
	def generateClockImage(self,hourHandAngle,minuteHandAngle,secondHandAngle,hasBackground=True):

		offset = (int(random.random() * self.shakeVariance - self.shakeVariance / 2), int(random.random() * self.shakeVariance - self.shakeVariance / 2))
		rotation_offset = ((random.random() * self.shakeVariance - self.shakeVariance / 2) * 2) / 4
		
		scaleVariance = int(128 + ((random.random() - 0.5) * self.shakeVariance * 2.0))
		scaleVariance = (scaleVariance,scaleVariance)
		
		# simulated real clock with photo drawing
		alpha = 0
		if hasBackground:
			alpha = 255
		
		img = Image.new('RGBA', (128, 128), (int(127 + random.random() * 128),
			int(127 + random.random() * 128),
			int(127 + random.random() * 128),
			alpha))
		
		if self.includeSecondsHands:
			secondImgRotated = secondImg.resize(scaleVariance).rotate( (-secondHandAngle)+90+rotation_offset )
		hourImgRotated = hourImg.resize(scaleVariance).rotate( (-hourHandAngle)+90+rotation_offset )
		minuteImgRotated = minuteImg.resize(scaleVariance).rotate( (-minuteHandAngle)+90+rotation_offset )
		
		faceImgRotated = faceImg.resize(scaleVariance).rotate( rotation_offset )
				
		img.paste(faceImgRotated, offset, faceImgRotated)
		if self.includeSecondsHands:
			img.paste(secondImgRotated, offset, secondImgRotated)
		img.paste(hourImgRotated, offset, hourImgRotated)
		img.paste(minuteImgRotated, offset, minuteImgRotated)
		
		return img.resize((self.imgSize[1],self.imgSize[0]), Image.ANTIALIAS)
		
		
		
		'''
		# simulated clock with polygon drawing
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
			start + (end-start) * second_normalized).convert('L')
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
				img = self.generateNotClockImage().convert('L')
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
				start + (end-start) * second_normalized).convert('L')
			np.copyto(input_images[idx],np.array(img).reshape(self.imgSize[1],self.imgSize[0],self.imgSize[2]))
			input_images[idx] /= 255.0
			
			output_values[idx][hour_idx+1] = 1
			output_values[idx][minute_idx+12+1] = 1
			if self.includeSecondsHands:
				output_values[idx][second_idx+12+60+1] = 1
				
		return input_images,output_values
	
	def easeInExpo (self, start, end, val):
		return (end - start) * pow(2, 10 * (val / 1 - 1)) + start
	
	def easeInQuad (self, start, end, val):
		end -= start
		return end * val * val + start
	
	
	def generateClocksForLocalization(self, subdiv, num):
		input_images = np.zeros((num,self.imgSize[1],self.imgSize[0],self.imgSize[2]), dtype='float32')
		output_values = np.zeros((num,subdiv+subdiv), dtype='float32')

		for idx in range(0,num):
			
			# make a random background
			img = self.generateNotClockImage()
			
			if random.random() > self.notClockFaceThreshold:
				# make a random clock
				clock = self.generateClockImage(random.random() * 360, random.random() * 360, random.random() * 360, False)
			
				# place clock somewhere randomly in the image
				# Note about scale: we want tot weight it so that images with smaller scale
				# happen more often, as smaller scale images can occupy more spaces on the screen.
			
				scale = self.easeInQuad(0.1, 0.9, random.random())
				aspect = random.random() * 0.2 + 0.9
				xscale = scale * aspect
				yscale = scale / aspect
			
				xmin = int((random.random() - xscale / 2) * self.imgSize[1])
				ymin = int((random.random() - yscale / 2) * self.imgSize[0])
				width = int(self.imgSize[1]*xscale)
				height = int(self.imgSize[0]*yscale)
			
				if xmin < 0:
					xmin = 0
				if ymin < 0:
					ymin = 0
				if xmin + width > self.imgSize[1]:
					xmin = self.imgSize[1] - width
				if ymin + height > self.imgSize[0]:
					ymin = self.imgSize[1] - height
			
				xmax = (xmin + width)
				ymax = (ymin + height)
			
				clock = clock.resize((width,height), Image.ANTIALIAS)
			
				rotation_offset = ((random.random() * 32 - 32 / 2) * 2)
				rotation_offset = 0
				clockRotated = clock.rotate( rotation_offset )
			
				img.paste(clockRotated, (xmin,ymin), clockRotated)
				
				# 10 columns on X, 10 rows on Y, if the clock is in said row or column put a 1 in it
				xdelta = (self.imgSize[1] / subdiv)
				ydelta = (self.imgSize[0] / subdiv)
				for x in range(0, subdiv):
					for y in range(0, subdiv):
						xValue = x * xdelta
						yValue = y * ydelta
						if xValue+xdelta >= xmin and xValue <= xmax:
							output_values[idx][x] = 1
						if yValue+ydelta >= ymin and yValue <= ymax:
							output_values[idx][subdiv+y] = 1
			
			img = img.convert('L')
			np.copyto(input_images[idx],np.array(img).reshape(self.imgSize[1],self.imgSize[0],self.imgSize[2]))
			input_images[idx] /= 255.0
					
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
	
	def convertOutputToRect(self,output):
		return np.array2string(output.astype(int), separator='.')
	
	def saveImageToFile(self,img,filepath):
		img = img.reshape(self.imgSize[1],self.imgSize[0]) * 255.0
		Image.fromarray(img).convert("L").save(filepath)

	def GetCoordsFromOutput(self,output,size):
		subdiv = int(len(output)/2)
		
		xdelta = 1.0 / subdiv
		ydelta = 1.0 / subdiv
	
		xmin = 1.0
		xmax = 0.0
		ymin = 1.0
		ymax = 0.0
	
		for x in range(0,subdiv):
			for y in range(0,subdiv):
				xValue = (x*xdelta)
				yValue = (y*ydelta)
				
				if output[x] >= 0.5:
					if xValue < xmin:
						xmin = xValue
					if xValue > xmax:
						xmax = xValue
				if output[subdiv+y] >= 0.5:
					if yValue < ymin:
						ymin = yValue
					if yValue > ymax:
						ymax = yValue

		return (xmin*size[1],ymin*size[0],xmax*size[1],ymax*size[0])

if __name__ == '__main__':
	
	META_PATH = "./meta"
	
	size = [100,100,1]
	
	generator = ClockGenerator(size,True,0.2)
	generator.shakeVariance = 0
	
	np.set_printoptions(threshold=20)
	
	input,output = generator.generateClocksForLocalization(100,64)	
	for n in range(0,len(input)):
		sourceImg = Image.fromarray(input[n].reshape(size[1],size[0]) * 255.0).convert("RGB")
				
		draw = ImageDraw.Draw(sourceImg)
		draw.rectangle(generator.GetCoordsFromOutput(output[n],size), outline="green")		
		
		sourceImg.save('/tmp/clock_%s_%d.png' % (generator.convertOutputToRect(output[n]), n))
	
	
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import random
import cv2

np.set_printoptions(threshold=np.inf)

random.seed(7)
#current we use gray image firstly
# Ntrainingset=700 #60*2 #60 positive instance and 60 negative instance
# Ntestingset=50
# imageLength=784
extraGen=3
vectorPerImage=extraGen+1

class Dataset(object):
	def __init__(self,folder,Ntrainingset=700,Ntestingset=50,imagewidth=28,imageheight=28,shuffle=False):		
		#for posintimgfile in random.shuffle(sorted(os.listdir(folder+"/positive/intensity/64x48"))):			
		self.Ntrainingset=Ntrainingset
		self.Ntestingset=Ntestingset
		self.imageheight=imageheight
		self.imagewidth=imagewidth
		self.imagelength=imageheight*imagewidth
		
		posFolder = "positive/"
		negFolder = "negative/"
		dirList = sorted(os.listdir(folder+posFolder))
		for i in range(len(dirList)):
			dirList[i] = folder+posFolder+dirList[i]
		
		self.posFileList=dirList

		dirList = sorted(os.listdir(folder+negFolder))

		for i in range(len(dirList)):
			dirList[i] = folder+negFolder+dirList[i]
		
		self.negFileList = dirList

		#When creating 2 independent sets (trainging and testing), we shuffle the data firstly
		if shuffle==True:
			random.shuffle(self.posFileList);
			random.shuffle(self.negFileList);

		self.trainingFileList = self.posFileList[0:self.Ntrainingset]+self.negFileList[0:self.Ntrainingset];
		self.testingFileList = self.posFileList[self.Ntrainingset:self.Ntrainingset+self.Ntestingset]+self.negFileList[self.Ntrainingset:self.Ntrainingset+self.Ntestingset]

		randTrainingOrder = range(self.Ntrainingset*2)
		randTestingOrder = range(self.Ntestingset*2) 

		random.shuffle(randTrainingOrder)
		random.shuffle(randTestingOrder)

		self.randTrainingOrder = randTrainingOrder
		self.randTestingOrder = randTestingOrder

		poshot=[1,0]
		neghot=[0,1]

		#Training Set
		oriTrainingDecisionAttribute = np.array([poshot for i in range(self.Ntrainingset)]+[neghot for i in range(self.Ntrainingset)])
		trainingDecisionAttribute =np.array([poshot for i in range(self.Ntrainingset*vectorPerImage)]+[neghot for i in range(self.Ntrainingset*vectorPerImage)])
		for i in range(len(self.randTrainingOrder)):
			for v in range(vectorPerImage):
				trainingDecisionAttribute[i*vectorPerImage+v]=oriTrainingDecisionAttribute[self.randTrainingOrder[i]]

		trainingImageAttribute=np.zeros((self.Ntrainingset*2*vectorPerImage,self.imagelength))
		count = 0 #This is corresponding to source image as a source image can be used to generate more than one training vector

		for i in randTrainingOrder:
			#print("Load %d image"%(i))
		 	img = cv2.imread(self.trainingFileList[i],cv2.CV_LOAD_IMAGE_GRAYSCALE)
		 	img = img.astype(np.float)/255
		 	imgvector = np.reshape(img,(1,self.imagelength))
		  	trainingImageAttribute[count*vectorPerImage]=imgvector

			# cv2.imshow("flipimg",img)
			# print(trainingDecisionAttribute[count*vectorPerImage])
		 	# cv2.waitKey(500)

		  	flipimg=cv2.flip(img,0)
		 	imgvector = np.reshape(flipimg,(1,self.imagelength))
		 	trainingImageAttribute[count*vectorPerImage+1]=imgvector
		 	
		 	flipimg=cv2.flip(img,1)
		 	imgvector = np.reshape(flipimg,(1,self.imagelength))
		 	trainingImageAttribute[count*vectorPerImage+2]=imgvector

		 	flipimg=cv2.flip(img,-1)
		 	imgvector = np.reshape(flipimg,(1,self.imagelength))
		 	trainingImageAttribute[count*vectorPerImage+3]=imgvector

			count+=1

		self.trainingSet=(trainingImageAttribute,trainingDecisionAttribute)
		
		#Testing Set
		oriTestingDecisionAttribute = np.array([poshot for i in range(self.Ntestingset)]+[neghot for i in range(self.Ntestingset)])
		testingDecisionAttribute = np.array([poshot for i in range(self.Ntestingset*vectorPerImage)]+[neghot for i in range(self.Ntestingset*vectorPerImage)])
		
		for i in range(len(self.randTestingOrder)):
			for v in range(vectorPerImage):
				testingDecisionAttribute[i*vectorPerImage+v]=oriTestingDecisionAttribute[self.randTestingOrder[i]]
		
		testingImageAttribute = np.zeros((self.Ntestingset*2*vectorPerImage,self.imagelength))
		count = 0
		
		for i in randTestingOrder:
		 	img=cv2.imread(self.testingFileList[i],cv2.CV_LOAD_IMAGE_GRAYSCALE)
		 	img=img.astype(np.float)/255
		 	imgvector = np.reshape(img,(1,self.imagelength))
		 	testingImageAttribute[count*vectorPerImage] = imgvector

		 	flipimg=cv2.flip(img,0)
		 	imgvector = np.reshape(flipimg,(1,self.imagelength))
		 	testingImageAttribute[count*vectorPerImage+1]=imgvector
		 	
		 	flipimg=cv2.flip(img,1)
		 	imgvector = np.reshape(flipimg,(1,self.imagelength))
		 	testingImageAttribute[count*vectorPerImage+2]=imgvector

		 	flipimg=cv2.flip(img,-1)
		 	imgvector = np.reshape(flipimg,(1,self.imagelength))
		 	testingImageAttribute[count*vectorPerImage+3]=imgvector

		 	count+=1

		self.testingSet=(testingImageAttribute, testingDecisionAttribute)

	def showimg(self,trainortest,imindex):
		if(trainortest==0):
			imset=self.trainingSet[0]
		else:
			imset=self.testingSet[0]

		cv2.imshow("Mat",np.reshape(imset[imindex],(self.imageheight,self.imagewidth)))
		cv2.waitKey(10)

	def showlabel(self,trainortest,imindex):
		if(trainortest==0):
			label=self.trainingSet[1]
		else:
			label=self.testingSet[1]
		
		return(label[imindex])

if __name__=="__main__":
	dataSet = Dataset("../data/gridmap417/",Ntrainingset=500,Ntestingset=100,shuffle=True)
	np.savetxt("../data/map_extrainingset_image_417.txt",dataSet.trainingSet[0])
	np.savetxt("../data/map_extrainingset_label_417.txt",dataSet.trainingSet[1],fmt="%d")
	np.savetxt("../data/map_extestingset_image_417.txt",dataSet.testingSet[0])
	np.savetxt("../data/map_extestingset_label_417.txt",dataSet.testingSet[1],fmt="%d")
	#np.save("../data/map_trainingset",dataSet.trainingSet)
	#np.save("../data/map_testingset",dataSet.traingSet)
	for i in range(10):
	 	dataSet.showimg(1,i)
	 	print(dataSet.showlabel(1,i))
	 	cv2.waitKey(500)



	# for i in range(120):
	# 	dataSet.showimg(0,i)
	# 	print(dataSet.showlabel(0,i))
	# 	cv2.waitKey(1000)





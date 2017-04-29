#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import cv2 as cv

#current we use gray image firstly
Ntrainingset=700*2 
Ntestingset=50*2

batchsize=100
trainingsetsize=1400

patch_h=28
patch_w=28
patch_size=patch_h*patch_w
patch_skip=1

def weight_variable(shape,vname=None):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial,name=vname)

def bias_variable(shape,vname=None):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial,name=vname)

def conv2d(x,W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def main():
	##Build the Network
	#First Convolutional Layer
	sess = tf.Session()

	x = tf.placeholder(tf.float32, shape=[None, patch_size])
	y_ = tf.placeholder(tf.float32, shape=[None, 2])

	W_conv1=weight_variable([5,5,1,32],"W_conv1")
	b_conv1=bias_variable([32],"b_conv1")

	x_image=tf.reshape(x,[-1,patch_h,patch_w,1])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	#Second Convolutional Layer
	W_conv2 = weight_variable([5,5,32,64],"W_conv2")
	b_conv2 = bias_variable([64],"b_conv2")

	h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	#Densely Connected Layer
	W_fc1 = weight_variable([patch_h*patch_w//16*64,1024],"W_fc1")
	b_fc1 = bias_variable([1024],"b_fc1")

	h_pool2_flat = tf.reshape(h_pool2,[-1, patch_h*patch_w//16*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

	#Dropout
	keep_prob =tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	#Readout Layer
	W_fc2 = weight_variable([1024,2],"W_fc2")
	b_fc2 = bias_variable([2],"b_fc2")

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#Specified to predict 
	classfication=tf.argmax(y_conv,1)

	saver = tf.train.Saver()
	sess=tf.Session()
	sess.run(tf.global_variables_initializer())
	saver.restore(sess, "../model/model417/model")
  	print("Model restored.")

  	mapimg=cv.imread("../image/map.pgm",0)
  	#polish_mapimg=np.zeros(mapimg.shape)
  	#polish_mapimg=np.copy(mapimg)
  	polish_mapimg = cv.merge((mapimg,mapimg,mapimg))
  	patchimg=np.zeros((patch_h,patch_w))
  	 #although this is useless, but it should be feeded into the graph to allow it be able to compute results
  	fakepredict=[[0,0]]
  	for h in range(patch_h//2,mapimg.shape[0]-patch_h//2,patch_skip):
  		for w in range(patch_w//2,mapimg.shape[1]-patch_w//2,patch_skip):
  			if(mapimg[h,w]!=205):
	  			patchimg=mapimg[h-patch_h//2:h+patch_h//2,w-patch_w//2:w+patch_w//2]
	  			arraypatch=[patchimg.reshape(patch_size)/255]
	  			value=classfication.eval(session=sess,feed_dict={x: arraypatch, y_: fakepredict, keep_prob: 1})*255
	  			polish_mapimg[h,w,1]= 255-value
	  			polish_mapimg[h,w,2]= value
  			
  	cv.imshow("polishimg",polish_mapimg)
  	cv.waitKey(0)
  	cv.imwrite("./polishimg.png",polish_mapimg)

if(__name__=="__main__"):
	main()

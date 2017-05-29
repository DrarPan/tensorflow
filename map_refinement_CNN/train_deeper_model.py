#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import os

#current we use gray image firstly

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
	batchsize=100
	trainingsetsize=4000
	trainingstep=300
	shuffle=True
	record=True

	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 2])
	sess = tf.Session()

	W_conv1=weight_variable([5,5,1,32],"W_conv1")
	b_conv1=bias_variable([32],"b_conv1")

	x_image=tf.reshape(x,[-1,28,28,1])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	#Second Convolutional Layer
	W_conv2 = weight_variable([5,5,32,64],"W_conv2")
	b_conv2 = bias_variable([64],"b_conv2")

	h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_conv3 = weight_variable([3,3,64,192],"W_conv3")
	b_conv3 = bias_variable([192],"b_conv3")

	h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)

	W_conv4 = weight_variable([3,3,192,128],"W_conv4")
	b_conv4 = bias_variable([128],"b_conv4")

	h_conv4 = tf.nn.relu(conv2d(h_conv3,W_conv4)+b_conv4)
	
	#Densely Connected Layer
	#W_fc1 = weight_variable([7*7*64,1024],"W_fc1")
	W_fc1 = weight_variable([7*7*128,1024],"W_fc1")
	b_fc1 = bias_variable([1024],"b_fc1")

	#h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*64])
	h_pool2_flat = tf.reshape(h_conv4,[-1, 7*7*128])
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
	saver=tf.train.Saver()

	accuracyRecord=[]
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# for v in tf.global_variables():
		# 	print(v)
		
		#Build data
		trainingset_img=np.loadtxt("../data/map_extrainingset_image_417.txt",dtype=float)
		trainingset_lab=np.loadtxt("../data/map_extrainingset_label_417.txt",dtype=int)
		testingset_img=np.loadtxt("../data/map_extestingset_image_417.txt",dtype=float)
		testingset_lab=np.loadtxt("../data/map_extestingset_label_417.txt",dtype=int)

		shufflenum=0

		for i in range(trainingstep):
			if(shuffle and i%(trainingsetsize/batchsize)==0):
				shufflenum=int(random.random()*batchsize)
				#print("shufflenum=%d"%shufflenum)

			start_index=(i*batchsize)%trainingsetsize
			#print("start_index=%d"%start_index)

			if(start_index+shufflenum+batchsize>trainingsetsize):
				batch=(np.row_stack((trainingset_img[start_index+shufflenum:trainingsetsize],trainingset_img[0:shufflenum])),
					   np.row_stack((trainingset_lab[start_index+shufflenum:trainingsetsize],trainingset_lab[0:shufflenum])))
			else:
				batch=(trainingset_img[start_index+shufflenum:start_index+shufflenum+batchsize],trainingset_lab[start_index+shufflenum:start_index+shufflenum+batchsize])

			print("step %d"%i)
		   	#train_accuracy = accuracy.eval(session=sess,feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})		
			#print("training accuracy %g"%(train_accuracy))
		  	
		  	c,_=sess.run([cross_entropy,train_step],feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.6})
		  	print('Current cost: ',c)

			test_accuracy=accuracy.eval(session=sess,feed_dict={x: testingset_img, y_: testingset_lab, keep_prob: 1})
			print("test accuracy %g"%(test_accuracy))
			#accuracyRecord.append([train_accuracy,test_accuracy])

		##Save Model
		save_path=saver.save(sess,"../model/model417/exmodel2")
		print("Model saved in file: %s" % save_path)
		
		# with open("./AccuracyRecord.txt",'w') as f:
		# 	for acc in accuracyRecord:
		# 		f.write("%f\n" % acc[1])

if(__name__=="__main__"):
	main()
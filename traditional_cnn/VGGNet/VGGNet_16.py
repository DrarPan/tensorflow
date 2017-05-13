#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import math
from datetime import datetime
import time

batch_size=32
num_batches=100

def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
	n_in = input_op.get_shape()[-1].value
	with tf.name_scope(name) as scope:
		kernel = tf.get_variable(scope+'w',shape=[kh,kw,n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
		conv=tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding="SAME")
		biases=tf.Variable(tf.constant(0.0,shape=[n_out],dtype=tf.float32),name='b')
		z=tf.nn.bias_add(conv,biases)
		activation=tf.nn.relu(z,name=scope)
		p+=[kernel,biases]
		return activation

def fc_op(input_op,name,n_out,p):
	n_in=input_op.get_shape()[-1].value
	with tf.name_scope(name) as scope:
		weight = tf.get_variable(scope+'w',shape=[n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
		biases=tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='b')
		activation=tf.nn.relu_layer(input_op,weight,biases,name=scope)
		
		p+=[weight,biases]
		return activation

def mpool_op(input_op,name,kh,kw,dh,dw):
	return tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding='SAME',name=name)

def inference_op(input_op,keep_prob):
	p=[]
	conv1_1=conv_op(input_op,name="conv1_1",kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
	conv1_2=conv_op(conv1_1,name="conv1_2",kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
	pool1=mpool_op(conv1_2,name="pool1",kh=2,kw=2,dw=2,dh=2)

	conv2_1=conv_op(pool1,"conv2_1",3,3,128,1,1,p)
	conv2_2=conv_op(conv2_1,"conv2_2",3,3,128,1,1,p)
	pool2=mpool_op(conv2_2,"pool2",2,2,2,2)

	conv3_1=conv_op(pool2,"conv3_1",3,3,256,1,1,p)
	conv3_2=conv_op(conv3_1,"conv3_2",3,3,256,1,1,p)
	conv3_3=conv_op(conv3_2,"conv3_3",3,3,256,1,1,p)
	pool3=mpool_op(conv3_3,"pool3",2,2,2,2)

	conv4_1=conv_op(pool3,"conv4_1",3,3,512,1,1,p)
	conv4_2=conv_op(conv4_1,"conv4_2",3,3,512,1,1,p)
	conv4_3=conv_op(conv4_2,"conv4_3",3,3,512,1,1,p)
	pool4=mpool_op(conv4_3,"pool4",2,2,2,2)

	conv5_1=conv_op(pool4,"conv5_1",3,3,512,1,1,p)
	conv5_2=conv_op(conv5_1,"conv5_2",3,3,512,1,1,p)
	conv5_3=conv_op(conv5_2,"conv5_3",3,3,512,1,1,p)
	pool5=mpool_op(conv5_3,"pool5",2,2,2,2)

	flat=tf.reshape(pool5,[batch_size,-1])
	
	fc6=fc_op(flat,name="fc6_drop",n_out=4096,p=p)
	fc6_drop=tf.nn.dropout(fc6,keep_prob,name="fc6_drop")

	fc7=fc_op(fc6_drop,name="fc7",n_out=4096,p=p)
	fc7_drop=tf.nn.dropout(fc7,keep_prob,name="fc7_drop")

	fc8=fc_op(fc7_drop,name="fc8",n_out=1000,p=p)
	softmax=tf.nn.softmax(fc8)
	prediction=tf.argmax(softmax,1)
	return prediction,softmax,fc8,p


def time_tensorflow_run(session,target,feed,info_string,f=None):
	num_steps_burn_in=10
	total_duration=0.0
	total_duration_squared=0.0
	print("Start training")
	for i in range(num_batches+num_steps_burn_in):
		print("Epoch %d"%i)
		start_time=time.time()
		_ = session.run(target,feed_dict=feed)
		duration=time.time()-start_time
		if i>num_steps_burn_in:
			if not i%10:
				print('%s: step %d, duration = %.3f'%(datetime.now(),i-num_steps_burn_in,duration))
				if f is not None:
					f.write('%s: step %d, duration = %.3f\n'%(datetime.now(),i-num_steps_burn_in,duration))
			total_duration+=duration
			total_duration_squared+=duration**2

	mn = total_duration/num_batches
	vr=total_duration_squared/num_batches-mn*mn
	sd=math.sqrt(vr)
	print('%s: %s across %d steps, %.3f +/- %.3f sec/batch'%(datetime.now(),info_string,num_batches,mn,sd))
	if f is not None:
		f.write('%s: %s across %d steps, %.3f +/- %.3f sec/batch\n'%(datetime.now(),info_string,num_batches,mn,sd))

def run_benchmark():
	with tf.Graph().as_default():
		image_size=224
		images=tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],dtype=tf.float32,stddev=0.1))

		keep_prob = tf.placeholder(tf.float32)

		prediction,softmax,fc8,p=inference_op(images,keep_prob)

		sess=tf.Session()
		sess.run(tf.global_variables_initializer())
		f=open('./VGGNetRunningTime.txt','w');
		time_tensorflow_run(sess,prediction,{keep_prob: 0.5},"Forward",f)
		objective=tf.nn.l2_loss(fc8)
		grad=tf.gradients(objective,p)#tensor
		time_tensorflow_run(sess,grad,{keep_prob: 0.5},"Backward",f)
		f.close()

run_benchmark()
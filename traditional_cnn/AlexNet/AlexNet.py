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

def print_activations(t):
	print(t.op.name,'',t.get_shape().as_list())

def inference(images,apply_fc=False):
	parameter=[]
	with tf.name_scope('conv1') as scope:
		kernel=tf.Variable(tf.truncated_normal([11,11,3,64],dtype=tf.float32,stddev=0.1),name='weights')
		bias=tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),name='biases')
		conv1=tf.nn.relu(tf.nn.conv2d(images,kernel,[1,4,4,1],padding='SAME')+bias,name=scope)
		print_activations(conv1)
		parameter+=[kernel,bias]

	lrn1=tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn1')
	pool1=tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool1')
	print_activations(pool1)

	with tf.name_scope('conv2') as scope:
		kernel=tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=0.1),name='weights')
		bias=tf.Variable(tf.constant(0.0,shape=[192],dtype=tf.float32),name='biases')
		conv2=tf.nn.relu(tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME')+bias,name=scope)
		print_activations(conv2)
		parameter+=[kernel,bias]

	lrn2=tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn2')
	pool2=tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool2')
	print_activations(pool2)

	with tf.name_scope('conv3') as scope:
		kernel=tf.Variable(tf.truncated_normal([3,3,192,384],dtype=tf.float32,stddev=0.1),name='weights')
		bias=tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),name='biases')
		conv3=tf.nn.relu(tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')+bias,name=scope)
		print_activations(conv3)
		parameter+=[kernel,bias]

	with tf.name_scope('conv4') as scope:
		kernel=tf.Variable(tf.truncated_normal([3,3,384,256],dtype=tf.float32,stddev=0.1),name='weights')
		bias=tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),name='biases')
		conv4=tf.nn.relu(tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')+bias,name=scope)
		print_activations(conv4)
		parameter+=[kernel,bias]

	with tf.name_scope('conv5') as scope:
		kernel=tf.Variable(tf.truncated_normal([3,3,256,256],dtype=tf.float32,stddev=0.1),name='weights')
		bias=tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),name='biases')
		conv5=tf.nn.relu(tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding='SAME')+bias,name=scope)
		print_activations(conv5)
		parameter+=[kernel,bias]

	pool5=tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool5')
	print_activations(pool5)

	if not apply_fc:
		return pool5,parameter;
	else:
		reshape=tf.reshape(pool5,[batch_size,-1])
		dim=reshape.get_shape()[1].value
		weight_fc1=tf.Variable(tf.truncated_normal([dim,4096],dtype=tf.float32,stddev=1e-8));
		bias_fc1=tf.Variable(tf.constant(0.0,shape=[4096],dtype=tf.float32),name='biases')
		fc1=tf.nn.relu(tf.matmul(reshape,weight_fc1)+bias_fc1,name="fc1");
		weight_fc2=tf.Variable(tf.truncated_normal([4096,4096],dtype=tf.float32,stddev=1e-8));
		bias_fc2=tf.Variable(tf.constant(0.0,shape=[4096],dtype=tf.float32),name='biases')
		fc2=tf.nn.relu(tf.matmul(fc1,weight_fc2)+bias_fc2,name="fc2");
		weight_fc3=tf.Variable(tf.truncated_normal([4096,1000],dtype=tf.float32,stddev=1e-5));
		bias_fc3=tf.Variable(tf.constant(0.0,shape=[1000],dtype=tf.float32),name='biases')
		fc3=tf.nn.relu(tf.matmul(fc2,weight_fc3)+bias_fc3,name="fc3");
		parameter+=[weight_fc1,bias_fc1,weight_fc2,bias_fc2,weight_fc3,bias_fc3]
		print_activations(fc1)
		print_activations(fc2)
		print_activations(fc3)
		return fc3,parameter

def time_tensorflow_run(session,target,info_string,f=None):
	num_steps_burn_in=10
	total_duration=0.0
	total_duration_squared=0.0

	for i in range(num_batches+num_steps_burn_in):
		start_time=time.time()
		_ = session.run(target)
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

		result,parameters=inference(images,False)

		init=tf.global_variables_initializer()
		sess=tf.Session()
		sess.run(init)
		f=open('./AlexNetRunningTime.txt','w');
		time_tensorflow_run(sess,result,"Forward",f)
		objective=tf.nn.l2_loss(result)
		grad=tf.gradients(objective,parameters)#tensor
		time_tensorflow_run(sess,grad,"Backward",f)
		f.close()

		

run_benchmark()

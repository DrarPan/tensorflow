#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname('__file__')), 'cifar10'))
import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time
import math

max_steps = 3200
batch_size = 128
data_dir="./cifar10_data/cifar-10-batches-bin/"


#cifar10.maybe_download_and_extract(data_dir)

images_train,labels_train=cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)
images_test,labels_test=cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)

def variable_with_weight_loss(shape,stddev,wl):
	var=tf.Variable(tf.truncated_normal(shape,stddev=stddev,dtype=tf.float32))
	if wl is not None:
		weight_loss=tf.multiply(tf.nn.l2_loss(var),wl,name="weight_loss")
	tf.add_to_collection('losses',weight_loss)
	return var

def loss_function(logits,labels):
	labels=tf.cast(labels,tf.int64)
	cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy_per_example')
	cross_entropy_mean=tf.reduce_mean(cross_entropy,name='cross_entropy')
	tf.add_to_collection('losses',cross_entropy_mean)

	return tf.add_n(tf.get_collection('losses'),name='total_loss')

def main():
	apply_lrn=True
	image_holder=tf.placeholder(tf.float32,[batch_size,24,24,3]) #batch_size can be set as None
	label_holder=tf.placeholder(tf.int32,[batch_size]) #This is not one_hot
	#net
	weight1=variable_with_weight_loss([5,5,3,64],stddev=5e-2,wl=0.0)
	kernel1=tf.nn.conv2d(image_holder,weight1,[1,1,1,1],padding='SAME')
	bias1=tf.Variable(tf.zeros([64]))
	conv1=tf.nn.relu(tf.nn.bias_add(kernel1,bias1))
	pool1=tf.nn.max_pool(conv1,[1,3,3,1],[1,2,2,1],padding='SAME')
	if(apply_lrn):
		norm1=tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75) #local response normalization
	else:
		norm1=tf.identity(pool1)

	weight2=variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,wl=0.0)
	kernel2=tf.nn.conv2d(norm1,weight2,[1,1,1,1],padding='SAME')
	bias2=tf.Variable(tf.constant(0.1,shape=[64]))
	conv2=tf.nn.relu(tf.nn.bias_add(kernel2,bias2))
	if(apply_lrn):
		norm2=tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75) #local response normalization
	else:
		norm2=tf.identity(pool2)
	pool2=tf.nn.max_pool(norm2,[1,3,3,1],[1,2,2,1],'SAME')

	reshape=tf.reshape(pool2,[batch_size,-1])
	dim=reshape.get_shape()[1].value
	weight3=variable_with_weight_loss(shape=[dim,384],stddev=0.04,wl=0.004)
	bias3=tf.Variable(tf.constant(0.1,shape=[384]))
	local3=tf.nn.relu(tf.matmul(reshape,weight3)+bias3)
	
	weight4=variable_with_weight_loss(shape=[384,192],stddev=0.04,wl=0.004)	
	bias4=tf.Variable(tf.constant(0.1,shape=[192]))
	local4=tf.nn.relu(tf.matmul(local3,weight4)+bias4)

	weight5=variable_with_weight_loss(shape=[192,10],stddev=1/192,wl=0.0)	
	bias5=tf.Variable(tf.constant(0.0,shape=[10]))
	logits=tf.add(tf.matmul(local4,weight5),bias5) #As it the output layer, use softmax to activate it

	loss=loss_function(logits,label_holder)
	train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)

	top_k_op=tf.nn.in_top_k(logits,label_holder,1)

	sess=tf.InteractiveSession()
	tf.global_variables_initializer().run()

	tf.train.start_queue_runners()

	for step in xrange(max_steps):
		start_time=time.time()
		image_batch,label_batch=sess.run([images_train,labels_train])
		_,loss_value=sess.run([train_op,loss],feed_dict={image_holder: image_batch, label_holder: label_batch})

		duration=time.time()-start_time

		if step%10==0:
			example_per_sec=batch_size/duration
			sec_per_batch=float(duration)

			format_str=('step %d, loss %.2f (%.1f example_per_sec; %.3f sec/batch)')
			print(format_str%(step,loss_value,example_per_sec,sec_per_batch))
	#save model
	save_path=tf.train.Saver().save(sess,"./model/advance_cnn")
	print("Model saved in file: %s" % save_path)
	
	#test set
	num_test_example=10000
	num_iter=int(math.ceil(num_test_example/batch_size))
	true_count=0
	total_count=num_iter*batch_size
	step=0
	while step<num_iter:
		image_batch,label_batch=sess.run([images_test,labels_test])
		predictions=sess.run([top_k_op],feed_dict={image_holder: image_batch, label_holder: label_batch})
		#print(predictions) #a array with 128 results corresponding one batch of data
		true_count+=np.sum(predictions)
		step+=1

	precision=true_count/total_count
	print('precision: %.3f'%precision)

if(__name__=="__main__"):
	main()
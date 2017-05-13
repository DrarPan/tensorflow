#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data",one_hot=True)

learning_rate=0.1
max_samples=400000
batch_size=128
display_step=10

n_input=28
n_steps=28
n_hidden=256
n_class=10

def BiRNN(x,weights,biases):
	x=tf.transpose(x,[1,0,2])
	x=tf.reshape(x,[-1,n_input])
	x=tf.split(0,n_steps,x,);

	lstm_fw_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden,forget_bias=1.0)
	lstm_bw_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden,forget_bias=1.0)
	
	output,_,_=tf.nn.bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype=tf.float32)

	return tf.matmul(output[-1],weights)+biases

x=tf.placeholder(tf.float32,[None,n_steps,n_input])
y=tf.placeholder(tf.float32,[None,n_class])

weights=tf.Variable(tf.random_normal([2*n_hidden,n_class]))
biases=tf.Variable(tf.random_normal([n_class]))

pred=BiRNN(x,weights,biases)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))

optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	step=1
	while step*batch_size<max_samples:
		batch_x,batch_y=mnist.train.next_batch(batch_size)
		batch_x=batch_x.reshape((batch_size,n_steps,n_input))
		sess.run(optimizer,feed_dict={x: batch_x, y: batch_y})
		if step% display_step==0:
			acc=sess.run(accuracy,feed_dict={x: batch_x, y: batch_y})
			loss=sess.run(cost,feed_dict={x: batch_x, y: batch_y})
			print("Iter ",step*batch_size, ", loss = ",loss,", accuracy = ",acc)
		step+=1
	print("Optimization Finished")

	test_len=10000
	test_data=mnist.test.images[:test_len].reshape((-1,n_steps,n_input))
	test_label=mnist.test.labels[:test_len]
	print("Testing Accuracy: ",sess.run(accuracy,feed_dict={x: test_data, y: test_label}))

#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()

in_unit=784
hi_unit=300
w1 = tf.Variable(tf.truncated_normal([in_unit,hi_unit],stddev=0.1))
b1 = tf.Variable(tf.zeros([hi_unit]))
w2 = tf.Variable(tf.zeros([hi_unit,10]))
b2 = tf.Variable(tf.zeros([10]))

x=tf.placeholder(tf.float32,[None,in_unit])
y_=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)

hidden1=tf.nn.relu(tf.matmul(x,w1)+b1)
hidden1_drop = tf.nn.dropout(hidden1,keep_prob)

y=tf.nn.softmax(tf.matmul(hidden1_drop,w2)+b2)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y+1e-8),reduction_indices=[1]))#the 1e-8 is important to avoid accuracy suddenly falling down
train_step =tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

tf.global_variables_initializer().run()
f=open("./accuracy.txt",'w')

for i in range(10000):
	batch_xs,batch_ys = mnist.train.next_batch(100)
	train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})
	if i%100==0:
		print("Epoch %d"%i)
		correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
		accuracy_train=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
		accuracy_test=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
		f.write("%d: %f,%f\n" % (i,accuracy_train.eval(({x: mnist.train.images, y_: mnist.train.labels, keep_prob: 1})),accuracy_test.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1})))

f.close()

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1}))



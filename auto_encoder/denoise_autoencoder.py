#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import cv2 as cv

def xavier_init(fan_in, fan_out, constant = 1):
	low=-constant*np.sqrt(6.0/(fan_in+fan_out))
	high=-low
	return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
	def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(),scale=0.1):
		self.n_input=n_input
		self.n_hidden=n_hidden
		self.transfer=transfer_function
		self.scale=tf.placeholder(tf.float32)
		self.training_scale=scale
		network_weights=self._initialize_weights()
		self.weights=network_weights

		self.x=tf.placeholder(tf.float32,[None,self.n_input])
		self.output_x=tf.add(
			tf.matmul(self.x+scale*tf.random_normal((n_input,)),self.weights['w1']),
			self.weights['b1'])
		self.hidden=self.transfer(self.output_x)

		self.reconstruction=tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])

		self.cost=0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
		self.optimizer=optimizer.minimize(self.cost)

		self.sess=tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def _initialize_weights(self):
		all_weights=dict()
		all_weights['w1']=tf.Variable(xavier_init(self.n_input,self.n_hidden))
		all_weights['b1']=tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
		all_weights['w2']=tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32))
		all_weights['b2']=tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
		return all_weights

	def partial_fit(self,X):
		cost,opt=self.sess.run((self.cost,self.optimizer),
			feed_dict={self.x: X, self.scale: self.training_scale})
		return cost

	def calc_total_cost(self, X):
		return self.sess.run(self.cost, 
			feed_dict = {self.x: X, self.scale: self.training_scale})

	def transform(self, X): #for obtainning the value of the hidden layer, which is the main target of AutoEncoder
		return self.sess.run(self.hidden, 
			feed_dict = {self.x: X, self.scale: self.training_scale})

	def generate(self, hidden=None): #restore
		if hidden is None:
			hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
		self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})

	def reconstruct(self,X):
		return self.sess.run(self.reconstruction,
			feed_dict={self.x: X, self.scale: self.training_scale})

	def getWeights(self):
		return self.sess.run(self.weight['w1'])

	def getBiases(self):
		return self.sess.run(self.weight['b1'])

def standard_scale(X_train,X_test):
	preprocessor=prep.StandardScaler().fit(X_train)
	X_train=preprocessor.transform(X_train)
	X_test=preprocessor.transform(X_test)
	return X_train,X_test

def get_random_block_from_data(data,batch_size):
	start_index = np.random.randint(0,len(data)-batch_size)
	return data[start_index:(start_index+batch_size)]

def main():
	mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
	X_train, X_test = standard_scale(mnist.train.images,mnist.train.images)
	n_sample=int(mnist.train.num_examples)
	training_epochs=20
	batch_size=128
	display_step=1

	autoencoder=AdditiveGaussianNoiseAutoencoder(n_input=784,n_hidden=16,scale=0.0)
	for epoch in range(training_epochs):
		avg_cost=0.
		total_batch=int(n_sample/batch_size)
		for i in range(total_batch):
			batch_xs=get_random_block_from_data(X_train,batch_size)
			cost=autoencoder.partial_fit(batch_xs)
			avg_cost+=cost/n_sample*batch_size
			
		if epoch%display_step==0:
			print("Epoch:", '%04d'%(epoch+1), "cost=", "{:.9f}".format(avg_cost))

		weight_showimg=np.ones((120,120),dtype=np.float32)*0.78
		for wi in range(16):
			row=wi//4
			col=wi%4
			
			encoder_weight=autoencoder.sess.run(autoencoder.weights['w1'])[0:784,wi]
			#encoder_weight_max=encoder_weight.max()
			#encoder_weight_min=encoder_weight.min()
			#encoder_weight=(encoder_weight-encoder_weight_min)/(encoder_weight_max-encoder_weight_min)
			weight_image=encoder_weight.reshape((28,28))*10 #we only show single-side of weight 
			weight_showimg[1+row*30:29+row*30,1+col*30:29+col*30]=weight_image;

		cv.namedWindow("weight",-1)
		cv.imshow("weight",weight_showimg)
		cv.imwrite("./weightimg_epoch"+str(epoch)+".png",weight_showimg*255)
		cv.waitKey(2000)
	# #test reconstruct
	# ori_vector=X_train[100,:].reshape(1,784)
	# ori_img=ori_vector.reshape((28,28))
	# out_img=autoencoder.reconstruct(ori_vector).reshape((28,28))
	# cv.namedWindow("Original",-1)
	# cv.imshow("Original",ori_img)
	# cv.namedWindow("After reconstruction",-1)
	# cv.imshow("After reconstruction",out_img)
	# cv.waitKey(100000)

	# #show weight
	# weight_showimg=np.ones((120,120),dtype=np.float32)*0.78
	# for wi in range(16):
	# 	row=wi//4
	# 	col=wi%4
		
	# 	encoder_weight=autoencoder.sess.run(autoencoder.weights['w1'])[0:784,wi]
	# 	#encoder_weight_max=encoder_weight.max()
	# 	#encoder_weight_min=encoder_weight.min()
	# 	#encoder_weight=(encoder_weight-encoder_weight_min)/(encoder_weight_max-encoder_weight_min)
	# 	weight_image=encoder_weight.reshape((28,28))*10 #we only show single-side of weight 
	# 	weight_showimg[1+row*30:29+row*30,1+col*30:29+col*30]=weight_image;

	# cv.namedWindow("weight",-1)
	# cv.imshow("weight",weight_showimg)
	# cv.imwrite("./weightimg_epoch"+str(training_epochs)+".png",weight_showimg*255)
	# cv.waitKey(2000)


if __name__ == '__main__':
	main()
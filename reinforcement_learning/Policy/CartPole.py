#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2 as cv
import time
import math
import numpy as np
import tensorflow as tf
import gym

env=gym.make('CartPole-v0')

env.reset()
random_episodes=0
reward_sum=0

# # RANDOM
# while random_episodes<10:
# 	env.render()
# 	observation,reward,done,_=env.step(np.random.randint(0,2))
	
# 	print(reward)
# 	reward_sum+=reward
# 	if done:
# 		random_episodes+=1
# 		print("Reward for this episode was: ",reward_sum)
# 		reward_sum=0
# 		env.reset()

H=50
batch_size=25
learning_rate=0.1
D=4
gamma=0.99#reward discount

observations=tf.placeholder(tf.float32,[None,D],name="input_x")
W1=tf.get_variable("W1",shape=[D,H],initializer=tf.contrib.layers.xavier_initializer())
layer1=tf.nn.relu(tf.matmul(observations,W1))
W2=tf.get_variable("W2",shape=[H,1],initializer=tf.contrib.layers.xavier_initializer())
tvars=tf.trainable_variables()#i.e. W1,W2

score=tf.matmul(layer1,W2)
probability=tf.nn.sigmoid(score)

adam=tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad=tf.placeholder(tf.float32,name="batch_grad1")
W2Grad=tf.placeholder(tf.float32,name="batch_grad2")
batchGrad=[W1Grad,W2Grad]

def discount_rewards(r):
	discounted_r=np.zeros_like(r)
	running_add=0;
	for t in reversed(range(r.size)):
		running_add = running_add*gamma+r[t]
		discounted_r[t]=running_add
	return discounted_r

input_y=tf.placeholder(tf.float32,[None,1],name="input_y")#inverse action?
advantages=tf.placeholder(tf.float32,name="reward_signal")
loglik = tf.log(input_y*(input_y-probability)+ 
				(1-input_y)*(input_y+probability)) #Action 1
loss=-tf.reduce_mean(loglik*advantages) #to maximize tf.reduce_mean(loglik)*advantages

updateGrads=adam.apply_gradients(zip(batchGrad,tvars))
newGrads=tf.gradients(loss,tvars) #tf.gradients(y,x) => dy/dx

xs,ys,drs=[],[],[]
reward_sum=0
episode_number=1
total_episodes=10000

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	rendering=False
	observation=env.reset()
	gradBuffer=sess.run(tvars)
	for ix,grad in enumerate(gradBuffer):
		gradBuffer[ix]=grad*0

	while episode_number<=total_episodes:
		if reward_sum/batch_size>160 or rendering==True:
			env.render()
			rendering=True
		#print(observation)
		x=np.reshape(observation,[-1,D])#change dimension from 1 to 2
		#print(x)
		tfprob=sess.run(probability,feed_dict={observations:x})
		action= 1 if np.random.uniform()<tfprob else 0
		xs.append(x)
		y=1-action
		ys.append(y)

		observation,reward,done,info=env.step(action)

		reward_sum+=reward
		drs.append(reward)

		if done:
			episode_number+=1
			epx=np.vstack(xs)
			epy=np.vstack(ys)
			epr=np.vstack(drs)
			xs,ys,drs=[],[],[]

			discounted_epr=discount_rewards(epr)
			discounted_epr-=np.mean(discounted_epr)
			discounted_epr/=np.std(discounted_epr)

			tGrad=sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})

			for ix, grad in enumerate(tGrad):
				gradBuffer[ix]+=grad

			if episode_number % batch_size==0:
				sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})

				for ix, grad in enumerate(gradBuffer):
					gradBuffer[ix]=grad*0

				print("Average reward for episode %d : %f"%(episode_number, reward_sum/batch_size))

				if reward_sum/batch_size>200:
					print("Task solved in", episode_number, "episodes!")

				reward_sum=0

			observation=env.reset()
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf

url='http://mattmahoney.net/dc/'

def maybe_download(folder,filename,expected_bytes):
	if not os.path.exists(folder+filename):
		filename,_=urllib.urlretrieve(url+filename,folder+filename)
	statinfo=os.stat(folder+filename)
	if statinfo.st_size==expected_bytes:
		print('Found and verified',folder+filename)
	else:
		print(statinfo.st_size)
		raise Exception('Failed to verify' + filename)
	return folder+filename

filename=maybe_download("../data/",'text8.zip',31344016)

def read_data(filename):
	with zipfile.ZipFile(filename) as f:
		data=tf.compat.as_str(f.read(f.namelist()[0])).split()
	return data

words = read_data(filename)
print('Data size',len(words))

vocabulary_size=50000

def build_dataset(words):
	count=[['UNK',-1]]
	count.extend(collections.Counter(words).most_common(vocabulary_size-1))
	dictionary=dict()
	for word,_ in count:
		dictionary[word]=len(dictionary)
	data=list()
	unk_count=0;
	for word in words:
		if word in dictionary:
			index=dictionary[word]
		else:
			index=0
			unk_count+=1
		data.append(index)
	count[0][1]=unk_count;
	reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
	return data,count,dictionary,reverse_dictionary

data,count,dictionary,reverse_dictionary=build_dataset(words)

del words
#print('Most common words',count[:5])
#print('Sample data',data[:10],[reverse_dictionary[i] for i in data[:10]])

data_index=0

def generate_batch(batch_size,num_skips,skip_window):
	global data_index
	assert batch_size%num_skips==0
	assert num_skips<=2*skip_window
	batch=np.ndarray(shape=(batch_size),dtype=np.int32)
	labels=np.ndarray(shape=(batch_size,1),dtype=np.int32)
	span=2*skip_window+1
	buffers=collections.deque(maxlen=span)

	for _ in range(span):#if needs not to change value, can use global variable data
		buffers.append(data[data_index])
		data_index=(data_index+1)%len(data)

	for i in range(batch_size //num_skips):
		target=skip_window
		targets_to_avoid=[skip_window]
		for j in range(num_skips):
			while target in targets_to_avoid:
				target=random.randint(0,span-1)
			targets_to_avoid.append(target)
			batch[i*num_skips+j]=buffers[skip_window]
			labels[i*num_skips+j,0]=buffers[target]
		buffers.append(data[data_index])#as buffers is a double end queue, the front element will be discarded
		data_index=(data_index+1)%len(data)
	return batch,labels

batch, labels=generate_batch(batch_size=8,num_skips=2,skip_window=1)

batch_size=128
embedding_size=128
skip_window=1
num_skips=2;
valid_size=16
valid_window=100
valid_examples=np.random.choice(valid_window,valid_size,replace=False)

num_sampled=64 #negative sample

graph=tf.Graph()
with graph.as_default():
	train_inputs=tf.placeholder(tf.int32,shape=[batch_size])
	train_labels=tf.placeholder(tf.int32,shape=[batch_size,1])
	valid_dataset=tf.constant(valid_examples,dtype=tf.int32)

	#:( we don't have a GPU
	embeddings=tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
	embed=tf.nn.embedding_lookup(embeddings,train_inputs)
	nce_weights=tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],stddev=1.0/math.sqrt(embedding_size)))
	nce_biases=tf.Variable(tf.zeros([vocabulary_size]))

	loss=tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,biases=nce_biases,labels=train_labels,inputs=embed,num_sampled=num_sampled,num_classes=vocabulary_size))
	optimizer=tf.train.GradientDescentOptimizer(1.0).minimize(loss)

	norm=tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
	normalized_embeddings=embeddings/norm
	valid_embeddings=tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)

	similarity=tf.matmul(valid_embeddings,normalized_embeddings,transpose_b=True)

num_steps=200001
with tf.Session(graph=graph) as sess:
	sess.run(tf.global_variables_initializer())
	print("Initialized")
	average_loss=0
	for step in range(num_steps):
		batch_inputs,batch_labels=generate_batch(batch_size,num_skips,skip_window)
		feed_dict={train_inputs: batch_inputs, train_labels: batch_labels}

		_,loss_val=sess.run([optimizer,loss],feed_dict=feed_dict)
		average_loss+=loss_val
		
		if step%5000==0:
			if step>0:
				average_loss/=2000
			print("Average loss at step ",step,": ",average_loss)
			average_loss=0
		if step%20000==0:
			sim=similarity.eval()
			for i in range(valid_size):
				valid_word=reverse_dictionary[valid_examples[i]]
				top_k=8
				nearest=(-sim[i,:]).argsort()[1:top_k+1]
				log_str="Nearest to %s: " % valid_word
				for k in range(top_k):
					close_word=reverse_dictionary[nearest[k]]
					log_str="%s %s,"%(log_str,close_word)
				print(log_str)
	final_embeddings=normalized_embeddings.eval()

import matplotlib.pyplot as plt

def plot_with_labels(low_dim_embs,labels,filename='tesn.png'):
	assert low_dim_embs.shape[0]>=len(labels),"More labels than embeddings"
	plt.figure(figsize=(18,18))
	for i, label in enumerate(labels):
		x,y=low_dim_embs[i,:]
		plt.scatter(x,y)
		plt.annotate(label,xy=(x,y),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')
	plt.savefig(filename)
	plt.show()

from sklearn.manifold import TSNE

tsne=TSNE(perplexity=30,n_components=2,init="pca",n_iter=5000)
plot_only=500

low_dim_embs=tsne.fit_transform(final_embeddings[:plot_only,:])
labels=[reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs,labels)



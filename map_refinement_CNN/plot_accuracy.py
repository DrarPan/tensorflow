#!/usr/bin/env python
import matplotlib.pyplot as plt

acc=[]	
with open("AccuracyRecord.txt",'r') as f:
	for line in f:
		acc.append(float(line))

plt.plot(acc)
plt.show()
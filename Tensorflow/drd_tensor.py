# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:39:31 2017

@author: ajalali
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction import DictVectorizer

v = DictVectorizer(sparse=False)
cwd = os.getcwd()
os.chdir('C:/Users/ajalali/Documents/python_tensorflow/diabetics_tensor')

drd_data = pd.read_csv('After_Rand_Data.csv')

#list(drd_data.columns.values)

out=drd_data[drd_data.columns[0]]
inp=drd_data[drd_data.columns[1:19]]

b, c = np.unique(out, return_inverse=True)
out=2*c-1
s=np.zeros((len(out),2))

for i in range(len(out)):
    if out[i]==-1:
        s[i,0]=1
    else:
        s[i,1]=1
        
out_net=s
    

#xx = v.fit_transform(out)
n=len(drd_data.columns)-1

l1=50
l2=100
l3=80
l4=2

Y=tf.placeholder(tf.float32, [None, 2])
X=tf.placeholder(tf.float32,[None,n])
W1=tf.Variable(tf.zeros([n,l1]))
b1=tf.Variable(tf.zeros([l1]))
W2=tf.Variable(tf.zeros([l1,l2]))
b2=tf.Variable(tf.zeros([l2]))
W3=tf.Variable(tf.zeros([l2,l3]))
b3=tf.Variable(tf.zeros([l3]))
W4=tf.Variable(tf.zeros([l3,l4]))
b4=tf.Variable(tf.zeros([l4]))

init=tf.global_variables_initializer()

#Y_hat=tf.nn.softmax(tf.matmul(X,W)+b)
#Y=tf.placeholder(tf.float32,[None,10])

#########################
Y1=tf.nn.relu(tf.matmul(X,W1)+b1) # 1*l1
Y2=tf.nn.sigmoid(tf.matmul(Y1,W2)+b2) #1*l2
Y3=tf.nn.relu(tf.matmul(Y2,W3)+b3) #1*l3
Y_hat=tf.nn.softmax(tf.matmul(Y3,W4)+b4) #1*l4



cross_entropy=-tf.reduce_sum(Y*tf.log(Y_hat))

optimizer=tf.train.GradientDescentOptimizer(0.03)
loss=optimizer.minimize(cross_entropy)

sess=tf.Session()
sess.run(init)

for i in range(1000):
  sess.run(loss, {X:inp, Y:out_net})

curr_W = sess.run([W4])
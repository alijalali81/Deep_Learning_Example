# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:33:04 2017

@author: ajalali
"""


#import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#import sys
#from sklearn.feature_extraction import DictVectorizer

#v = DictVectorizer(sparse=False)
cwd = os.getcwd()
os.chdir('C:/Users/ajalali/Documents/python_tensorflow/diabetics_tensor')
drd_data = pd.read_csv('After_Rand_Data.csv')

import cntk as ck
from cntk import Trainer, learning_rate_schedule, UnitType, sgd
from cntk.initializer import glorot_uniform
from cntk.layers import default_options, Dense


#list(drd_data.columns.values)

out=drd_data[drd_data.columns[0]]
inp=drd_data[drd_data.columns[1:19]]

perms=np.random.permutation(len(out))
new_inp=inp.iloc[perms,:]
new_out=out[perms]
out=new_out
inp=new_inp


colors=['r' if l==0 else 'b' for l in out]
plt.scatter(inp[inp.columns[4]],inp[inp.columns[5]],c=colors)


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

num_hidden_layers=3
hidden_layer_dim=[l1,l2,l3]
num_out_class=2

input1=ck.input(n)
label=ck.input(2)


def linear_layer(inp_var,output_dim):
    inp_dim=inp_var.shape[1]
    w=ck.parameter(shape=(inp_dim,output_dim))
    b=ck.parameter(shape=(output_dim))
    
    return b+ck.times(inp_var,w)
    
#inp.iloc[3,:]
    
def dense_layer(inp_var,output_dim, nonlinearity):
    l=linear_layer(inp_var, output_dim)
    
    return nonlinearity(l)
    

def full_ff_net(inp_var, num_out_class, hidden_layer_dim, num_hidden_layers, nonlinearity):
    h=dense_layer(inp_var,hidden_layer_dim[0],nonlinearity)
    for i in range(1,num_hidden_layers):
        h=dense_layer(inp_var,hidden_layer_dim[i],nonlinearity)
        
        return linear_layer(h,num_out_class)
        

z=full_ff_net(inp,2,hidden_layer_dim,num_hidden_layers,ck.relu)

def create_model(features):
    with default_options(init=glorot_uniform(),activation=ck.relu):
        h=features
        for i in range(num_hidden_layers):
            h=Dense(hidden_layer_dim[i])(h)
        last_layer=Dense(num_out_class,activation=ck.softmax)
        
        return last_layer(h)
        
z1=create_model(input1)

loss=ck.cross_entropy_with_softmax(z1,label)
eval_error=ck.classification_error(z1,label)


learning_rate=0.5
lr_schedule=ck.learning_rate_schedule(learning_rate,UnitType.minibatch)
learner=ck.sgd(z1.parameters,lr_schedule)
trainer=Trainer(z1,(loss,eval_error),[learner])


def moving_average(a,w=10):
    if len(a)<w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]
    
def print_training_progress(trainer,mb,frequency,verbose=1):
    training_loss="NA"
    eval_error="NA"
    
    if mb%frequency == 0:
        training_loss=trainer.previous_minibatch_loss_average
        eval_error=trainer.previous_minibatch_evaluation_average
        if verbose:
            print("Minibatch: {}, Train_loss: {}, Train Error: {}".format(mb,training_loss,eval_error))
            
        return mb,training_loss,eval_error

mb_size=200
num_samples=80000
num_mb_to_train=num_samples/mb_size

training_progress_out_freq=50

plotdata={"batchsize":[],"loss":[],"error":[]}

for i in range(0,int(num_mb_to_train)):
    inpu=inp.iloc[range((mb_size*i),(mb_size*(i+1))),:]
    X=inpu.astype(np.float32) 
    X=np.asarray(X,dtype=np.float32)
    outp=out_net[range((mb_size*i),(mb_size*(i+1)))]
    Y=np.asarray(outp,dtype=np.float32)
    trainer.train_minibatch({input1:X,label:Y})
    m=i
    if i%training_progress_out_freq ==0:
        training_loss=trainer.previous_minibatch_loss_average
        eval_error=trainer.previous_minibatch_evaluation_average
        print("Minibatch: {}, Train_loss: {}, Train Error: {}".format(i,training_loss,eval_error))
#   batchsize,los,erro=print_training_progress(trainer,m,training_progress_out_freq,verbose=0)
    
    if not (training_loss=="NA" or eval_error=="NA"):
        plotdata["batchsize"].append(i)
        plotdata["loss"].append(training_loss)
        plotdata["error"].append(eval_error)
        

plotdata["avgloss"] = moving_average(plotdata["loss"])
plotdata["avgerror"] = moving_average(plotdata["error"])


plt.figure(1)
plt.subplot(211)
plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')
plt.show()

plt.subplot(212)
plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--')
plt.xlabel('Minibatch number')
plt.ylabel('Label Prediction Error')
plt.title('Minibatch run vs. Label Prediction Error')
plt.show()


test_minibatch_size = 2000
inpuu=inp.iloc[range(90000,90000+test_minibatch_size),:]
X_test=inpuu.astype(np.float32) 
X_test=np.asarray(X_test,dtype=np.float32)
outpp=out_net[range(90000,90000+test_minibatch_size)]
Y_test=np.asarray(outpp,dtype=np.float32)
trainer.test_minibatch({input1 : X_test, label : Y_test})



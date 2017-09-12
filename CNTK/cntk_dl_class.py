# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:59:00 2017

@author: ajalali
"""


#import tensorflow as tf
import numpy as np
import pandas as pd


import cntk as ck
from cntk import Trainer, learning_rate_schedule, UnitType, sgd
from cntk.initializer import glorot_uniform
from cntk.layers import default_options, Dense


#list(drd_data.columns.values)



class DeepMLP:
    
    learning_rate=0.5
    training_progress_out_freq=50
    mb_size=200
    num_samples=80000
    num_mb_to_train=num_samples/mb_size

    
    
    def __init__(self,inp,out,hidden_layer_dim):
        perms=np.random.permutation(len(out))
        new_inp=inp.iloc[perms,:]
        new_out=out[perms]
        out=new_out
        inp=new_inp
        b, c = np.unique(out, return_inverse=True)
        out=2*c-1
        s=np.zeros((len(out),2))
        num_out_class=b
        for i in range(len(out)):
            if out[i]==-1:
                s[i,0]=1
            else:
                s[i,1]=1
        
        out_net=s
        n=len(inp.columns)
        input1=ck.input(n)
        label=ck.input(len(num_out_class))
        self.inputtemplate=input1
        self.outtemplate=label
        self.input=inp
        self.output=out
        self.out_net=out_net
        self.hidden_layer_dim=hidden_layer_dim
        self.num_hidden_layer=len(hidden_layer_dim)
        self.outdim=len(num_out_class)
        
    def create_model(self):
        with default_options(init=glorot_uniform(),activation=ck.relu):
            h=self.inputtemplate
            for i in range(self.num_hidden_layer):
                h=Dense(self.hidden_layer_dim[i])(h)
            last_layer=Dense(self.outdim,activation=ck.softmax)
        
            return last_layer(h)
    
    def create_deep_model(self,features):
        with default_options(init=glorot_uniform(),activation=ck.relu):
            h=features
            for i in range(self.num_hidden_layer):
                h=Dense(self.hidden_layer_dim[i])(h)
            last_layer=Dense(self.outdim,activation=ck.softmax)
        
            return last_layer(h)
    
    
    
    
    def Training(self):
#        input1=self.inputtemplate
#        z1=self.create_model(input1)
        z1=self.create_model()
        loss=ck.cross_entropy_with_softmax(z1,self.outtemplate)
        eval_error=ck.classification_error(z1,self.outtemplate)
        lr_schedule=ck.learning_rate_schedule(self.learning_rate,UnitType.minibatch)
        learner=ck.sgd(z1.parameters,lr_schedule)
        trainer=Trainer(z1,(loss,eval_error),[learner])
        
        plotdata={"batchsize":[],"loss":[],"error":[]}

        for i in range(0,int(self.num_mb_to_train)):
            inpu=self.input.iloc[range((self.mb_size*i),(self.mb_size*(i+1))),:]
            X=inpu.astype(np.float32) 
            X=np.asarray(X,dtype=np.float32)
            outp=self.out_net[range((self.mb_size*i),(self.mb_size*(i+1)))]
            Y=np.asarray(outp,dtype=np.float32)
            trainer.train_minibatch({self.inputtemplate:X,self.outtemplate:Y})
            if i%self.training_progress_out_freq ==0:
                training_loss=trainer.previous_minibatch_loss_average
                eval_error=trainer.previous_minibatch_evaluation_average
                print("Minibatch: {}, Train_loss: {}, Train Error: {}".format(i,training_loss,eval_error))
        #   batchsize,los,erro=print_training_progress(trainer,m,training_progress_out_freq,verbose=0)
            
            if not (training_loss=="NA" or eval_error=="NA"):
                plotdata["batchsize"].append(i)
                plotdata["loss"].append(training_loss)
                plotdata["error"].append(eval_error)
                
        self.trained_model=trainer
        self.training_vars=plotdata
        return trainer,plotdata
        
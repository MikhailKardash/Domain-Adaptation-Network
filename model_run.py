# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:45:51 2019

@author: Michael K
"""

# mycode start
import torch
#import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#from sklearn import naive_bayes
#from sklearn.neighbors import KNeighborsClassifier
from Libraries.theLoss import theLoss
from Libraries.DANet2 import DANet2


#load model
DA_nn = torch.load('./data_use/model.pt')
#load data
x_dat = torch.load("./data_use/Target_dat.pt")


#run data through model

#preallocate space
y_pred = torch.zeros(len(x_dat),56)

#run labeled data
zs,h_tanh = DA_nn.forward(x_dat)
      
#y_pred = torch.transpose(y_pred,0,1)
  
output = h_tanh[-1]

torch.save(output,'./data_use/output.pt')
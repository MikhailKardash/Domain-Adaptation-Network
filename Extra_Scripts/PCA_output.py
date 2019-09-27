# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 15:43:28 2019

@author: Michael K
"""

import torch

X_tens = torch.load('./data_use/output.pt')


Xmean = torch.sum(X_tens,0)/X_tens.shape[0]
Xmean = torch.unsqueeze(Xmean,0)
print(Xmean.shape)
j = X_tens - Xmean
print(j.shape)
U,S,V = torch.svd(j)
print(U.shape)
#can use something like this to grab N = 5 dimensions:
V = V[:,:3]

out = torch.mm(X_tens,V)
print(out.shape)

torch.save(out,'./data_use/output_PCA.pt')
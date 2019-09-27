# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:25:52 2019

@author: Misha
"""

import matplotlib.pyplot as plt


loss = list()
with open('./data_use/loss2.txt','r') as f:
    for line in f:
        loss.append(float(line))
        
loss = loss[2:]

plt.plot(loss)
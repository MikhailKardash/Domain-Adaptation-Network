# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 00:59:00 2019

@author: Michael K
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import json


output = torch.load('./data_use/output_PCA.pt')
labels_tot = torch.load("./data_use/t_labels.pt")
train_dat = output.data.numpy()
train_labl = labels_tot.data.numpy()
train_labl = train_labl.astype(int)

x = train_labl
y = torch.load("./data_use/Target_dat.pt")
y = y.data.numpy()

x1 = torch.load("./data_use/t_labels_fixed.pt")
y1 = torch.load("./data_use/Target_dat_fixed.pt")
x1 = x1.data.numpy()
x1 = x1.astype(int)
y1 = y1.data.numpy()

x_s = torch.load("./data_use/labels.pt")
x_s = x_s.data.numpy()
x_s = x_s.astype(int)
y_s = torch.load("./data_use/Source_dat.pt")
y_s = y_s.data.numpy()




def plot_hyperspace(points, labels):
  print("hyper space")
  #point_labels = [points, labels]
  #points = [[x1, x2, x3, ... xn], ...]
  marker_symbols = ['s','^','o','v','.','>','<']
  color_symbols = ['b', 'g', 'r','c','m','y','k']
  marks = [marker_symbols[l] for l in labels]
  colors = [color_symbols[l] for l in labels]
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  for i in range(len(labels)):
      ax.scatter(points[i][0], points[i][1], points[i][2], marker=marks[i], color=colors[i])
  plt.show()

plot_hyperspace(train_dat, train_labl)
#plot_hyperspace(y,x)
#plot_hyperspace(y1,x1)
#plot_hyperspace(y_s[0:1000],x_s[0:1000])

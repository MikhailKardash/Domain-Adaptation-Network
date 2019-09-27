# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 13:42:22 2019

@author: Michael K
"""

import torch
import numpy as np
from sklearn import naive_bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

NN_points = torch.load('./data_use/output.pt').data.numpy()
original = torch.load('./data_use/Target_dat.pt').data.numpy()
labels = torch.load("./data_use/t_labels.pt").data.numpy()

#NN_points = torch.load('./data_use/output.pt').data.numpy()
#original = torch.load('./data_use/Source_dat.pt').data.numpy()
#labels = torch.load("./data_use/labels.pt").data.numpy()

inds = np.random.permutation(len(original))
NN_points = NN_points[inds]
original = original[inds]
labels = labels[inds]

n1 = int(np.floor(len(labels)*0.8))
n2 = len(labels)

#test against this
neigh = KNeighborsClassifier(n_neighbors=5, algorithm = 'brute')
neigh.fit(original[0:n1],labels[0:n1])
gauss = naive_bayes.GaussianNB()
gauss.fit(original[0:n1],labels[0:n1])

out = neigh.predict(original[n1+1:n2])
out2 = gauss.predict(original[n1+1:n2])

correct = sum(out == labels[n1+1:n2])
error = (len(out)-correct)/(len(out))
print(error)
correct = sum(out2 == labels[n1+1:n2])
error = (len(out2)-correct)/(len(out2))
print(error)
print(confusion_matrix(labels[n1+1:n2],out))

neigh = KNeighborsClassifier(n_neighbors=5, algorithm = 'brute')
neigh.fit(NN_points[0:n1],labels[0:n1])
gauss = naive_bayes.GaussianNB()
gauss.fit(NN_points[0:n1],labels[0:n1])

out = neigh.predict(NN_points[n1+1:n2])
out2 = gauss.predict(NN_points[n1+1:n2])

correct = sum(out == labels[n1+1:n2])
error = (len(out)-correct)/(len(out))
print(error)
correct = sum(out2 == labels[n1+1:n2])
error = (len(out2)-correct)/(len(out2))
print(error)

print(confusion_matrix(labels[n1+1:n2],out))

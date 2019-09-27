Mikhail Kardash

# Implementation of Deep Transfer Metric Learning

## Introdution

This repository is an implementation of Deep Transfer Metric Learning. The associated research paper is located here:  https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Hu_Deep_Transfer_Metric_2015_CVPR_paper.pdf

This network is meant for data transformation. Data in a known space with class labels is fed to create a decision boundary. That decision boundary is then transferred to a new data space based on unlabeled data.

It is possible to feed the same data to both the labeled and unlabeled parts, but good classification accuracy is not guaranteed.

Refer to model_train and model_run scripts as an example of how to implement this code.

Required libraries: Pytorch, torchvision, numpy.

This network currently only runs on CPU.

# Functions

## *DANet(dims, act_func)* creates a linear network object.

dims:  a list of dimensions. dims[0] should be the same as the dimensionality of your input. Every subsequent entry is the output of the next linear layer.

act_func:  activation function of the linear layer. currently supports torch.tanh, torch.sigmoid, and F.relu

## *z,h = network_object.forward(data)* pushes data forward through the network.

z and h results will need to be passed to the loss function later so make sure to save them. 

## *theLoss(alpha, beta, gamma, omega, tau, k1, k2)* creates a loss object.

refer to the paper in the introduction for explanation of hyperparameters.

note that *omega* and *tau* need to be lists of length N-1 where N is the number of network layers.

## *loss, j = loss_object.forward(Model, xlabl, xun, actS, actT, labels)* pushes network results through the loss function.

Output parameters:

loss:  network loss

j: layer loss, needs to be passed to the backward function.

Input parameters:

Model: pass the network_object created by DANet

xlabl: labeled data that was input to network_object.forward()

xun: unlabeled data that was input to network_object.forward()

actS: h output of network_object.forward(x_labeled)

actT: h output of network_object.forward(x_unlabeled)

labels: class labels for xlabl

## grad_out,bias_out = criterion.backward(Model,xlabl,xun,actS,actT,zS,zT,j,labels)

Outputs:

grad_out: updated weights

bias_out: updated biases

Inputs:

xlabl, xun, actS, actT, labels:  see previous function

zS: z output of network_object.forward(x_labeled)

zT: z output of network_object.forward(x_unlabeled)

j: j output of loss_object.forward()

# Libraries Folder

## DANet

Constructs network architecture and handles identification of activation function.

## theLoss

Contains loss class including forward and backward functions.

# Extra_Scripts

## Classifiers

Example of classifiers using scipy. 

## graph_loss

Script to graph loss vs iteration of network after training has been completed.

## graph_output

Graphs 3-d output space after running PCA_Output script.

## PCA_Output

Performs PCA on network output. Reduces dimensionality of data using SVD to make data easier to visualize.


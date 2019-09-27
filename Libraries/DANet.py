# model definition
import torch
import torch.nn as nn
import torch.nn.functional as F

class DANet(torch.nn.Module):
    #requires list of dims and an activation function to be passed, i.e. [5,6],torch.tanh
    def __init__(self, dims,act):
        super(DANet2, self).__init__()
        self.dim = dims
        self.numLayers = len(dims) - 1
        self.linears = nn.ModuleList([nn.Linear(dims[i],dims[i+1]) for i in range(self.numLayers)])
        self.act = act
        self.derivFunc = self.getDeriv(act)
        #for i in range(self.numLayers):
        #    self.linears[i

    def forward(self, x):     
        z = []
        activ = []
        dat = x  
        for i in range(self.numLayers):
            temp = torch.Tensor(torch.zeros(len(dat),self.dim[i+1]))
            temp2 = torch.Tensor(torch.zeros(len(dat),self.dim[i+1]))
            for j in range(len(x)):
                temp[j][:] = self.linears[i](dat[j])
                temp2[j][:] =self.act(temp[j])
            dat = temp2
            z.append(temp)
            activ.append(dat)
            
            
        return [z,activ]

    #various derivatives
    def tanhDeriv(self,z,inp):
        return 1-inp*inp
    
    def sigDeriv(self,z,inp):
        return inp*(1-inp)
    
    def reluDeriv(self,z,inp):
        return torch.gt(z,0).float()
    
    #def gaussDeriv(self,z,inp):
    #    return -2*z*inp
    
    #function to assign derivatives
    def getDeriv(self,act):
        switcher = {
            torch.tanh: self.tanhDeriv,
            torch.sigmoid: self.sigDeriv,
            F.relu: self.reluDeriv
                }
        return switcher.get(act,0)
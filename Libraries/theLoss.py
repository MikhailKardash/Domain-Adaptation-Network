# looking like this is gonna require large batch training
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class theLoss:
    def __init__(self, alpha, beta, gamma, omega, tau, k1, k2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega = omega
        self.tau = tau
        self.k1 = k1
        self.k2 = k2

    # used for euclidean distance and other things.

    def distance_matrix(self, mat1, mat2):
        # get the product x * y
        # here, y = x.t()
        r = torch.mm(mat1, torch.transpose(mat2, 0, 1))
        # get the diagonal elements
        diag = r.diag().unsqueeze(0)
        diag = diag.expand_as(r)
        # compute the distance matrix
        D = diag + torch.transpose(diag, 0, 1) - 2 * r
        return torch.sqrt(D)

    def similarity_matrix(self, mat):
        return self.distance_matrix(mat, mat)

    def similarity_new(self, x):
        n = x
        #m1 = torch.max(n)
        #m2 = torch.min(n)
        #var = m1 - m2
        #n = n - m2
        #n = n/var
        #xixi = torch.sum(n * n, 1)
        #xjxj = torch.unsqueeze(xixi, 0)
        #xixi = torch.unsqueeze(xixi, 1)
        #xixj = torch.sum(torch.unsqueeze(n,1)*torch.unsqueeze(n, 0),2)
        #temp = xixi + xjxj
        #outmat = torch.squeeze(temp - 2*xixj)
        #outmat = torch.abs(outmat)
        
        n2 = torch.unsqueeze(n,0)
        n = torch.unsqueeze(n,1)
        outmat = n-n2
        outmat = outmat*outmat
        outmat = torch.sum(outmat,2)
        outmat = torch.squeeze(outmat)

        
        #print(temp[2,9])
        #print(-2*xixj[2,9])

        return torch.sqrt(outmat)

    # used to remove zero elements before argsorting
    def sparse_argsort(self, arr):
        ind = torch.squeeze(torch.nonzero(arr))
        return ind[torch.argsort(arr[ind], descending=False)]

    # get rid of y.
    def make_partial_loss(self, P, Q, ys, yt, y, labels):
        # loopable part of the loss function, Sc, Sd, and Dterm
        #actual_d2mat = self.similarity_matrix(ys)
        actual_d2mat = self.similarity_new(ys)
        actual_d2mat = torch.squeeze(actual_d2mat)
        

        # compute P*d^2 and Q*d^2
        Pd = P * actual_d2mat
        # Pd = torch.squeeze(Pd)
        Qd = Q * actual_d2mat
        # Qd = torch.squeeze(Qd)

        # compute Sc and Sb
        Sc = 0
        Sd = 0
        for i in range(0, P.shape[0]):
            Sc = Sc + torch.sum(Pd[i][self.sparse_argsort(torch.squeeze(Pd[i]))[:self.k1]])
            Sd = Sd + torch.sum(Qd[i][self.sparse_argsort(torch.squeeze(Qd[i]))[:self.k2]])

        Sc = Sc / self.k1 / len(labels)
        Sd = self.alpha * Sd / self.k2 / len(labels)
        # make D function
        Dterm = torch.sum(yt, 0) / len(yt) - torch.sum(ys, 0) / len(ys)
        Dterm = torch.norm(Dterm, 2)
        Dterm = self.beta * Dterm

        print("SC:" + str(float(Sc)))
        print("SD:" + str(float(Sd)))
        print("Dterm:" + str(float(Dterm)))

        return Sc - Sd + Dterm

    def forward(self, Model, xlabl, xun, actS, actT, labels):
        #actS is all source activations.
        #actT is all target activations.
        a = labels.reshape(len(labels), 1)
        b = labels.reshape(1, len(labels))
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)

        # rewrite b as torch.transpose(a)
        P = torch.eq(a, b)  # same-class identifier matrix
        Q = 1 - P  # different-class identifier matrix
        P = P.type(torch.FloatTensor)
        Q = Q.type(torch.FloatTensor)
        
        # extract weights
        L = []
        for i in range(0,Model.numLayers):
            temp = torch.Tensor(Model.linears[i].weight)
            temp2 = torch.Tensor(Model.linears[i].bias)
            temp2 = torch.unsqueeze(temp2,1)
            temp = torch.cat((temp,temp2),1)
            L.append(temp)

        # make norm terms
        wTerm = []
        for i in range(0,Model.numLayers):
            wTerm.append(self.gamma*torch.sum(torch.norm(L[i],'fro',dim=1)))
            print("Wterm"+str(i)+": " + str(float(wTerm[i])))

        
        #create total activation tensors.
        act = []
        for i in range(0,Model.numLayers):
            act.append(torch.cat((actS[i],actT[i]),0))
        
        # call make_partial_loss for each layer activations
        j = []
        for i in range(0,Model.numLayers):
            temp = self.make_partial_loss(P,Q,actS[i],actT[i],act[i],labels)
            j.append(temp + wTerm[i])

        # finally, update loss
        loss = j[Model.numLayers-1]
        for i in range(0,Model.numLayers-2):
            loss = loss + self.omega[i]*F.relu(j[i] - self.tau[i])

        return [loss, j]

    # forms h matrix
    def differencemat(self, mat):
        x1 = torch.unsqueeze(mat, 1)
        x2 = torch.unsqueeze(mat, 0)
        return x1 - x2

    # batch dot multiplication. Rewrite this later.
    def batchdot(self, mat, todot):
        temp = mat
        temp2 = todot.expand(temp.shape[0],todot.shape[0])       
        return temp*temp2

    # transposing Lmat
    def trans3d(self, arr):
        return torch.transpose(arr, 0, 1)

    def transp1(self,P,mat):
      temp = P.expand(mat.shape[2],P.shape[0],P.shape[1])
      temp = torch.transpose(temp,0,1)
      temp = torch.transpose(temp,1,2)
      return torch.transpose(torch.transpose(temp*mat,0,1),1,2)
    
    def transp2(self,P,mat):
      temp = P.expand(mat.shape[2],P.shape[0],P.shape[1])
      temp = torch.transpose(temp,0,1)
      temp = torch.transpose(temp,1,2)
      return torch.transpose(temp*mat,1,2)
    
    def JacobianTerms(self,Model_layer_weights,Lij,Lji,Lti,Lsi,xl,xu,P,Q):
      J = torch.zeros([Lij.shape[2],xl.shape[1]])
      xtemp = xl.expand(xl.shape[0],xl.shape[0],xl.shape[1])
      J = J + 2*torch.sum(torch.bmm(self.transp1(P,Lij),xtemp),0)/len(Lij)/self.k1
      J = J + 2*torch.sum(torch.bmm(self.transp2(P,Lji),xtemp),0)/len(Lij)/self.k1 
      J = J - 2*torch.sum(torch.bmm(self.transp1(Q,Lij),xtemp),0)*self.alpha/self.k2/len(Lij)
      J = J - 2*torch.sum(torch.bmm(self.transp2(Q,Lji),xtemp),0)*self.alpha/self.k2/len(Lij)
      
      J = J + 2*self.beta*(torch.mm(torch.transpose(Lti,0,1),xu)/len(Lti) + torch.mm(torch.transpose(Lsi,0,1),xl)/len(Lsi))
      J = J + 2*self.gamma*Model_layer_weights
      #print(J)
      
      return J
    
    def JacobianBias(self,Model_Biases,Lij,Lji,Lti,Lsi,P,Q):
      J = torch.zeros(Model_Biases.shape)

      J = J + 2*torch.sum(torch.sum(self.transp1(P,Lij),2),0)/len(Lij)/self.k1
      J = J + 2*torch.sum(torch.sum(self.transp2(P,Lji),2),0)/len(Lij)/self.k1

      J = J - 2*torch.sum(torch.sum(self.transp1(Q,Lji),2),0)*self.alpha/self.k2/len(Lij)
      J = J - 2*torch.sum(torch.sum(self.transp2(Q,Lji),2),0)*self.alpha/self.k2/len(Lij)

      J = J + 2*self.beta*(torch.sum(Lti,0)/len(Lti) + torch.sum(Lsi,0)/len(Lsi))
      
      return J + 2*self.gamma*Model_Biases

    def backward(self,Model,xlabl,xun,actS,actT,zS,zT,j,labels):
      a = labels.reshape(len(labels),1)
      b = labels.reshape(1,len(labels))
      a = torch.from_numpy(a)
      b = torch.from_numpy(b)
      
      #make P and Q matrices.
      P = torch.eq(a,b) #same-class identifier matrix
      Q = 1-P #different-class identifier matrix
      P = P.type(torch.FloatTensor)
      Q = Q.type(torch.FloatTensor)
      #ctual_d2mat = self.similarity_matrix(actS[-1])
      actual_d2mat = self.similarity_new(actS[-1])
      Pd = P*actual_d2mat
      Qd = Q*actual_d2mat
      
      P = torch.zeros(Pd.shape[0],Pd.shape[1])
      Q = torch.zeros(Qd.shape[0],Qd.shape[1])
      
      for i in range(0,P.shape[0]):
        P[i][self.sparse_argsort(torch.squeeze(Pd[i]))[:self.k1]] = torch.ones(self.k1)
        Q[i][self.sparse_argsort(torch.squeeze(Qd[i]))[:self.k2]] = torch.ones(self.k2)

      
      #make Lij and Lji for the 2 layers. These have to be labeled, thus z1s, source data.
      source_derivs = []
      target_derivs = []
      for i in range(Model.numLayers):
          #source_derivs.append(1 - actS[i]*actS[i])
          #target_derivs.append(1 - actT[i]*actT[i])
          source_derivs.append(Model.derivFunc(zS[i],actS[i]))
          target_derivs.append(Model.derivFunc(zT[i],actT[i]))
      
      L = []
      Lji = []
      for i in range(Model.numLayers):
          #topmost layer
          t = self.differencemat(actS[i])
          tji = torch.transpose(t,0,1)
          t = t*torch.transpose(source_derivs[i].expand(
                      source_derivs[i].shape[0],source_derivs[i].shape[0],source_derivs[i].shape[1]),0,1)
          tji = tji*source_derivs[i].expand(
                      source_derivs[i].shape[0],source_derivs[i].shape[0],source_derivs[i].shape[1])
          t = [t]
          tji = [tji]
          k = i
          #if not top layer:
          while k > 0:
              W = torch.Tensor(Model.linears[k].weight)
              k = k - 1
              tempji = torch.bmm(tji[-1],W.expand(tji[-1].shape[0],W.shape[0],W.shape[1]))
              temp = torch.transpose(tempji,0,1)
              temp = temp*torch.transpose(source_derivs[k].expand(
                      source_derivs[k].shape[0],source_derivs[k].shape[0],source_derivs[k].shape[1]),0,1)
              tempji = tempji*source_derivs[k].expand(
                      source_derivs[k].shape[0],source_derivs[k].shape[0],source_derivs[k].shape[1])
              t.append(temp)
              tji.append(tempji)
          L.append(t[::-1]) #because want the last element to be topmost layer
          Lji.append(tji[::-1])
      
      #now make Lti and Lsi matrices      
      sums = []
      for i in range(Model.numLayers):
          sums.append(torch.sum(actT[i],0)/len(actT[i]) - torch.sum(actS[i],0)/len(actS[i]))
     
      Lti = []
      Lsi = []
      for i in range(Model.numLayers):
          ti = [self.batchdot(target_derivs[i],sums[i])]
          si = [self.batchdot(source_derivs[i],-1*sums[i])]
          k = i
          while k > 0:
              W = torch.Tensor(Model.linears[k].weight)
              k = k - 1
              tempti = torch.mm(ti[-1],W)*target_derivs[k]
              tempsi = torch.mm(si[-1],W)*source_derivs[k]
              ti.append(tempti)
              si.append(tempsi)
          Lti.append(ti[::-1])
          Lsi.append(si[::-1])
      #compute Jacobian of each layer. 
      J = []
      xL = [xlabl]
      xU = [xun]
      for i in range(Model.numLayers-1):
          xL.append(actS[i])
          xU.append(actT[i])
          
      for i in range(Model.numLayers):
          top = self.JacobianTerms(Model_layer_weights=torch.Tensor(Model.linears[i].weight),
                                   Lij=L[i][i],Lji=Lji[i][i],Lti=Lti[i][i],Lsi=Lsi[i][i],xl=xL[i],xu=xU[i],P=P,Q=Q)
          top = [top]
          k = i
          while k > 0:
              k = k - 1
              temp = self.JacobianTerms(Model_layer_weights=torch.Tensor(Model.linears[k].weight),
                                   Lij=L[i][k],Lji=Lji[i][k],Lti=Lti[i][k],Lsi=Lsi[i][k],xl=xL[k],xu=xU[k],P=P,Q=Q)
              top.append(temp)
          J.append(top[::-1])
      Jb = []
      for i in range(Model.numLayers):
          top = [self.JacobianBias(Model_Biases=torch.Tensor(Model.linears[i].bias),
                                   Lij=L[i][i],Lji=Lji[i][i],Lti=Lti[i][i],Lsi=Lsi[i][i],P=P,Q=Q)]
          k = i
          while k > 0:
              k = k - 1
              temp = self.JacobianBias(Model_Biases=torch.Tensor(Model.linears[k].bias),
                                   Lij=L[i][k],Lji=Lji[i][k],Lti=Lti[i][k],Lsi=Lsi[i][k],P=P,Q=Q)
              top.append(temp)
          Jb.append(top[::-1])      

      #deep layer
      for i in range(Model.numLayers-1):
          if (j[i] - self.tau[i]):
              k = i
              while k >= 0:
                  J[-1][k] = J[-1][k] + self.omega[i]*J[i][k]
                  Jb[-1][k] = Jb[-1][k]+ self.omega[i]*Jb[i][k]
                  k = k - 1
      
      
      return [J[-1], Jb[-1]]
    
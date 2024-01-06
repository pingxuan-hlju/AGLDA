import torch
import math
import numpy as np
import torch.nn as nn

class Net1(nn.Module):
    def __init__(self):
        super(Net1,self).__init__()
        self.c11=nn.Conv2d(1,32,kernel_size=(2,7),stride=1,padding=0)
        self.s2=nn.MaxPool2d(kernel_size=(1,7))
        self.c31=nn.Conv2d(32,64,kernel_size=(1,7),stride=1,padding=0)
        self.c32=nn.Conv2d(32,64,kernel_size=(1,7),stride=1,padding=0)
        self.s4=nn.MaxPool2d(kernel_size=(1,7))
        self.cf1=nn.Linear(22*64,200)#1140->70
        self.f3=nn.Linear(200,2)
        self.cd1=nn.Dropout(0.5)
        self.cd2=nn.Dropout2d(0.5)
        self.leakrelu=nn.LeakyReLU()
        self.tanh=nn.Tanh()
    def forward(self,x1,x2,fea,G1):
        x3=x2+240
        x=torch.cat([fea[x1][:,None,None,:],fea[x3][:,None,None,:]],dim=2)
        x=self.s2(self.leakrelu(self.c11(x)))
        att=self.tanh(self.c32(x))
        x=self.s4(self.leakrelu(self.c31(x)*att+self.c31(x)))
        x=self.cd2(x)
        x=x.reshape(x.shape[0],-1)
        x=self.leakrelu(self.cf1(x))
        x=self.cd1(x)
        x=self.f3(x)
        return x
    
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))  
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)     
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)          
        return x
class Net2(nn.Module):
    def __init__(self):
        super(Net2,self).__init__()
        
        self.ne=nn.Parameter(torch.rand(500,300))
        self.w1=nn.Parameter(torch.rand(1140,300))
        #self.w2=nn.Parameter(torch.randn(1440,300))
        self.w2=nn.Linear(1440,300)
        self.h3=HGNN_conv(1140,600)
        self.h4=HGNN_conv(600,300)
        self.h1=HGNN_conv(1140,600)
        self.h2=HGNN_conv(600,300)        
        self.hf1=nn.Linear(1440*2,50)
        self.f3=nn.Linear(50,2)
        self.cd1=nn.Dropout(0.5)
        self.cd2=nn.Dropout2d(0.5)
        
        self.leakrelu=nn.LeakyReLU()
        self.tanh=nn.Tanh()
    def forward(self,x1,x2,fea,G1):
        
        x3=x2+240

        hyperH=self.leakrelu(G1@self.w1@self.ne.T)
        self.H=hyperH
        hyperH=(hyperH-hyperH.min())/(hyperH.max()-hyperH.min())
        
        dv=(torch.sum(hyperH,dim=1)**(-1/2))[:,None]
        de=(torch.sum(hyperH,dim=0)**(-1))[None,:]
        dv[torch.where(dv==torch.inf)]=0
        de[torch.where(de==torch.inf)]=0
        hyperH=dv*hyperH*de@hyperH.T*dv

        fea1=self.leakrelu(self.h1(fea,G1))
        fea2=self.leakrelu(self.h3(fea,hyperH))
        fea3=self.leakrelu(self.h2(fea1,G1))
        fea4=self.leakrelu(self.h4(fea2+fea1,hyperH))
        fea3=torch.cat([fea,fea3+fea4],dim=1)

        self.oe=self.w2(self.H.T@fea3)
        
        fea3=torch.cat([fea3[x1],fea3[x3]],dim=1)
        fea3=self.leakrelu(self.hf1(fea3))
        x=self.cd1(fea3)
        x=self.f3(x)
        return x
    
    def loss(self):
        p=torch.softmax(self.oe/1e+7,dim=-1)
        q=torch.log_softmax(self.ne,dim=-1)
        return nn.KLDivLoss(reduction='batchmean')(q,p)


class Net3(nn.Module):
    def __init__(self):
        super(Net3,self).__init__()

        self.h1=HGNN_conv(1140,600)
        self.h2=HGNN_conv(600,300)        
        self.hf1=nn.Linear(1440*2,50)

        self.c11=nn.Conv2d(1,32,kernel_size=(2,7),stride=1,padding=0)
        self.s2=nn.MaxPool2d(kernel_size=(1,7))
        self.c31=nn.Conv2d(32,64,kernel_size=(1,7),stride=1,padding=0)
        self.c32=nn.Conv2d(32,64,kernel_size=(1,7),stride=1,padding=0)
        self.s4=nn.MaxPool2d(kernel_size=(1,7))
        self.cf1=nn.Linear(22*64,200)#1140->70

        self.f3=nn.Linear(250,2)
        self.cd1=nn.Dropout(0.5)
        self.cd2=nn.Dropout2d(0.5)
        
        self.leakrelu=nn.LeakyReLU()
        self.tanh=nn.Tanh()
    def forward(self,x1,x2,fea,G1):
        
        x3=x2+240
        x=torch.cat([fea[x1][:,None,None,:],fea[x3][:,None,None,:]],dim=2)
        
        fea1=self.leakrelu(self.h1(fea,G1))
        fea3=self.leakrelu(self.h2(fea1,G1))
        fea3=torch.cat([fea,fea3],dim=1)
        
        fea3=torch.cat([fea3[x1],fea3[x3]],dim=1)
        fea3=self.leakrelu(self.hf1(fea3))
        
        x=self.s2(self.leakrelu(self.c11(x)))
        att=self.tanh(self.c32(x))
        x=self.s4(self.leakrelu(self.c31(x)*att+self.c31(x)))
        x=self.cd2(x)
        x=x.reshape(x.shape[0],-1)
        x=self.leakrelu(self.cf1(x))
        
        x=torch.cat([x,fea3],dim=1)
        
        x=self.cd1(x)
        x=self.f3(x)

        return x
    
class Net4(nn.Module):
    def __init__(self):
        super(Net4,self).__init__()
        
        self.ne=nn.Parameter(torch.rand(500,300))
        self.w1=nn.Parameter(torch.rand(1140,300))
        self.w2=nn.Linear(1440,300)
        self.h3=HGNN_conv(1140,600)
        self.h4=HGNN_conv(600,300)
        self.hf1=nn.Linear(1440*2,50)

        self.c11=nn.Conv2d(1,32,kernel_size=(2,7),stride=1,padding=0)
        self.s2=nn.MaxPool2d(kernel_size=(1,7))
        self.c31=nn.Conv2d(32,64,kernel_size=(1,7),stride=1,padding=0)
        self.c32=nn.Conv2d(32,64,kernel_size=(1,7),stride=1,padding=0)
        self.s4=nn.MaxPool2d(kernel_size=(1,7))
        self.cf1=nn.Linear(22*64,200)#1140->70

        self.f3=nn.Linear(250,2)
        self.cd1=nn.Dropout(0.5)
        self.cd2=nn.Dropout2d(0.5)
        
        self.leakrelu=nn.LeakyReLU()
        self.tanh=nn.Tanh()
    def forward(self,x1,x2,fea,G1):
        
        x3=x2+240
        x=torch.cat([fea[x1][:,None,None,:],fea[x3][:,None,None,:]],dim=2)
        
        hyperH=self.leakrelu(G1@self.w1@self.ne.T)
        self.H=hyperH
        hyperH=(hyperH-hyperH.min())/(hyperH.max()-hyperH.min())
        
        dv=(torch.sum(hyperH,dim=1)**(-1/2))[:,None]
        de=(torch.sum(hyperH,dim=0)**(-1))[None,:]
        dv[torch.where(dv==torch.inf)]=0
        de[torch.where(de==torch.inf)]=0
        hyperH=dv*hyperH*de@hyperH.T*dv

        fea2=self.leakrelu(self.h3(fea,hyperH))
        fea4=self.leakrelu(self.h4(fea2,hyperH))
        fea3=torch.cat([fea,fea4],dim=1)

        self.oe=self.w2(self.H.T@fea3)
        
        fea3=torch.cat([fea3[x1],fea3[x3]],dim=1)
        fea3=self.leakrelu(self.hf1(fea3))
        
        x=self.s2(self.leakrelu(self.c11(x)))
        att=self.tanh(self.c32(x))
        x=self.s4(self.leakrelu(self.c31(x)*att+self.c31(x)))
        x=self.cd2(x)
        x=x.reshape(x.shape[0],-1)
        x=self.leakrelu(self.cf1(x))
        
        x=torch.cat([x,fea3],dim=1)
        
        x=self.cd1(x)
        x=self.f3(x)

        return x
    
    def loss(self):
        p=torch.softmax(self.oe/1e+7,dim=-1)
        q=torch.log_softmax(self.ne,dim=-1)
        return nn.KLDivLoss(reduction='batchmean')(q,p)
    
class Net5(nn.Module):
    def __init__(self):
        super(Net5,self).__init__()
        
        self.ne=nn.Parameter(torch.rand(500,300))
        self.w1=nn.Parameter(torch.rand(1140,300))
        self.w2=nn.Linear(1440,300)
        self.h3=HGNN_conv(1140,600)
        self.h4=HGNN_conv(600,300)
        self.hf1=nn.Linear(1440*2,50)

        self.c11=nn.Conv2d(1,32,kernel_size=(2,7),stride=1,padding=0)
        self.s2=nn.MaxPool2d(kernel_size=(1,7))
        self.c31=nn.Conv2d(32,64,kernel_size=(1,7),stride=1,padding=0)
        self.c32=nn.Conv2d(32,64,kernel_size=(1,7),stride=1,padding=0)
        self.s4=nn.MaxPool2d(kernel_size=(1,7))
        self.cf1=nn.Linear(22*64,200)#1140->70

        self.f3=nn.Linear(250,2)
        self.cd1=nn.Dropout(0.5)
        self.cd2=nn.Dropout2d(0.5)
        
        self.leakrelu=nn.LeakyReLU()
        self.tanh=nn.Tanh()
    def forward(self,x1,x2,fea,G1):
        
        x3=x2+240
        x=torch.cat([fea[x1][:,None,None,:],fea[x3][:,None,None,:]],dim=2)
        
        hyperH=self.leakrelu(G1@self.w1@self.ne.T)
        self.H=hyperH
        hyperH=(hyperH-hyperH.min())/(hyperH.max()-hyperH.min())
        
        dv=(torch.sum(hyperH,dim=1)**(-1/2))[:,None]
        de=(torch.sum(hyperH,dim=0)**(-1))[None,:]
        dv[torch.where(dv==torch.inf)]=0
        de[torch.where(de==torch.inf)]=0
        hyperH=dv*hyperH*de@hyperH.T*dv

        fea2=self.leakrelu(self.h3(fea,hyperH))
        fea4=self.leakrelu(self.h4(fea2,hyperH))
        fea3=torch.cat([fea,fea4],dim=1)

        self.oe=self.w2(self.H.T@fea3)
        
        fea3=torch.cat([fea3[x1],fea3[x3]],dim=1)
        fea3=self.leakrelu(self.hf1(fea3))
        
        x=self.s2(self.leakrelu(self.c11(x)))
        att=self.tanh(self.c32(x))
        x=self.s4(self.leakrelu(self.c31(x)*att+self.c31(x)))
        x=self.cd2(x)
        x=x.reshape(x.shape[0],-1)
        x=self.leakrelu(self.cf1(x))
        
        x=torch.cat([x,fea3],dim=1)
        
        x=self.cd1(x)
        x=self.f3(x)

        return x
    
    def loss(self):
        p=torch.softmax(self.oe/1e+7,dim=-1)
        q=torch.log_softmax(self.ne,dim=-1)
        return nn.KLDivLoss(reduction='batchmean')(q,p)

class Net6(nn.Module):
    def __init__(self):
        super(Net6,self).__init__()
        
        self.ne=nn.Parameter(torch.rand(500,300))
        self.w1=nn.Parameter(torch.rand(1140,300))
        self.h3=HGNN_conv(1140,600)
        self.h4=HGNN_conv(600,300)
        self.h1=HGNN_conv(1140,600)
        self.h2=HGNN_conv(600,300)        
        self.hf1=nn.Linear(1440*2,50)

        self.c11=nn.Conv2d(1,32,kernel_size=(2,7),stride=1,padding=0)
        self.s2=nn.MaxPool2d(kernel_size=(1,7))
        self.c31=nn.Conv2d(32,64,kernel_size=(1,7),stride=1,padding=0)
        self.c32=nn.Conv2d(32,64,kernel_size=(1,7),stride=1,padding=0)
        self.s4=nn.MaxPool2d(kernel_size=(1,7))
        self.cf1=nn.Linear(22*64,200)#1140->70

        self.f3=nn.Linear(250,2)
        self.cd1=nn.Dropout(0.5)
        self.cd2=nn.Dropout2d(0.5)
        
        self.leakrelu=nn.LeakyReLU()
        self.tanh=nn.Tanh()
    def forward(self,x1,x2,fea,G1):
        
        x3=x2+240
        x=torch.cat([fea[x1][:,None,None,:],fea[x3][:,None,None,:]],dim=2)
        
        hyperH=self.leakrelu(G1@self.w1@self.ne.T)
        self.H=hyperH
        hyperH=(hyperH-hyperH.min())/(hyperH.max()-hyperH.min())
        
        dv=(torch.sum(hyperH,dim=1)**(-1/2))[:,None]
        de=(torch.sum(hyperH,dim=0)**(-1))[None,:]
        dv[torch.where(dv==torch.inf)]=0
        de[torch.where(de==torch.inf)]=0
        hyperH=dv*hyperH*de@hyperH.T*dv

        fea1=self.leakrelu(self.h1(fea,G1))
        fea2=self.leakrelu(self.h3(fea,hyperH))
        fea3=self.leakrelu(self.h2(fea1,G1))
        fea4=self.leakrelu(self.h4(fea2+fea1,hyperH))
        fea3=torch.cat([fea,fea3+fea4],dim=1)
  
        fea3=torch.cat([fea3[x1],fea3[x3]],dim=1)
        fea3=self.leakrelu(self.hf1(fea3))
        
        x=self.s2(self.leakrelu(self.c11(x)))
        att=self.tanh(self.c32(x))
        x=self.s4(self.leakrelu(self.c31(x)*att+self.c31(x)))
        x=self.cd2(x)
        x=x.reshape(x.shape[0],-1)
        x=self.leakrelu(self.cf1(x))
        
        x=torch.cat([x,fea3],dim=1)
        
        x=self.cd1(x)
        x=self.f3(x)

        return x
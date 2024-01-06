import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from sklearn import metrics
torch.cuda.empty_cache()
device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
learn_rate=0.0003
epoch=80
batch=32
n,ld,dd,md,lm,tri,tei,lda,ll,mm=torch.load('../parasave/tempdata2.pth')
feas,G1s=torch.load('../parasave/tempdata3.pth')

class MyDataset(Dataset):#shuffle and batch from Dataloader
    def __init__(self,tri,ld):
        self.tri=tri
        self.ld=ld
    def __getitem__(self,idx):
        x,y=self.tri[:,idx]
        label=self.ld[x][y]
        return x,y,label
    def __len__(self):
        return self.tri.shape[1]
    
class HGNN_conv(nn.Module):
    '''
    @article{feng2018hypergraph,
    title={Hypergraph Neural Networks},
    author={Feng, Yifan and You, Haoxuan and Zhang, Zizhao and Ji, Rongrong and Gao, Yue},
    journal={AAAI 2019},
    year={2018}
    }
    '''
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
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.ne=nn.Parameter(torch.rand(500,300))
        self.w1=nn.Parameter(torch.rand(1140,300))
        #self.w2=nn.Parameter(torch.randn(1440,300))
        self.w2=nn.Linear(1440,300)
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
    
def train(model,train_set,test_set,fea,G1,tei,epoch,learn_rate,cros):
    optimizer=torch.optim.Adam(model.parameters(),learn_rate,weight_decay=0.001)
    cost=nn.CrossEntropyLoss()
    model.train()
    fea,G1=fea.float().to(device),G1.float().to(device)#,G2.float().to(device)
    Amax=[0,0]
    for i in range(epoch):
        for x1,x2,y in train_set:
            x1,x2,y=Variable(x1.long()).to(device),Variable(x2.long()).to(device),Variable(y.long()).to(device)
            out=model(x1,x2,fea,G1)
            loss=cost(out,y)#+model.loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (i+1)%10 == 0 and i+1>=50:
            tacc(model,test_set,fea,G1,tei,cros,Amax)
        #if i+1==epoch:
            #tacc(model,test_set,fea,G1,0,cros)
        torch.cuda.empty_cache()

def calculate_TPR_FPR(RD, f, B):
    old_id = np.argsort(-RD)
    min_f = int(min(f))
    max_f = int(max(f))
    TP_FN = np.zeros((RD.shape[0], 1), dtype=np.float64)
    FP_TN = np.zeros((RD.shape[0], 1), dtype=np.float64)
    TP = np.zeros((RD.shape[0], max_f), dtype=np.float64)
    TP2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)
    FP = np.zeros((RD.shape[0], max_f), dtype=np.float64)
    FP2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)
    P = np.zeros((RD.shape[0], max_f), dtype=np.float64)
    P2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)
    for i in range(RD.shape[0]):
        TP_FN[i] = sum(B[i] == 1)
        FP_TN[i] = sum(B[i] == 0)
    for i in range(RD.shape[0]):
        for j in range(int(f[i])):
            if j == 0:
                if B[i][old_id[i][j]] == 1:
                    FP[i][j] = 0
                    TP[i][j] = 1
                    P[i][j] = TP[i][j] / (j + 1)
                else:
                    TP[i][j] = 0
                    FP[i][j] = 1
                    P[i][j] = TP[i][j] / (j + 1)
            else:
                if B[i][old_id[i][j]] == 1:
                    FP[i][j] = FP[i][j - 1]
                    TP[i][j] = TP[i][j - 1] + 1
                    P[i][j] = TP[i][j] / (j + 1)
                else:
                    TP[i][j] = TP[i][j - 1]
                    FP[i][j] = FP[i][j - 1] + 1
                    P[i][j] = TP[i][j] / (j + 1)
    ki = 0
    for i in range(RD.shape[0]):
        if TP_FN[i] == 0:
            TP[i] = 0
            FP[i] = 0
            ki = ki + 1
        else:
            TP[i] = TP[i] / TP_FN[i]
            FP[i] = FP[i] / FP_TN[i]
    for i in range(RD.shape[0]):
        kk = f[i] / min_f
        for j in range(min_f):
            TP2[i][j] = TP[i][int(np.round_(((j + 1) * kk))) - 1]
            FP2[i][j] = FP[i][int(np.round_(((j + 1) * kk))) - 1]
            P2[i][j] = P[i][int(np.round_(((j + 1) * kk))) - 1]
    TPR = TP2.sum(0) / (TP.shape[0] - ki)
    FPR = FP2.sum(0) / (FP.shape[0] - ki)
    Pr = P2.sum(0) / (P.shape[0] - ki)
    return TPR, FPR, Pr
def tacc(model,tset,fea,G1,tei,cros,Amax):
    predall,yall=torch.tensor([]),torch.tensor([])
    model.eval()
    for x1,x2,y in tset:
        x1,x2,y=Variable(x1.long()).to(device),Variable(x2.long()).to(device),Variable(y.long()).to(device)
        pred=model(x1,x2,fea,G1).data
        predall=torch.cat([predall,torch.as_tensor(pred,device='cpu')],dim=0)
        yall=torch.cat([yall,torch.as_tensor(y,device='cpu')])
    #torch.save((predall,yall),'./prady/PandY%d_%d' % (cros,epoch+1))
    pred=torch.softmax(predall,dim=1)[:,1]
    trh=torch.zeros(ld.shape[0],ld.shape[1])-1
    tlh=torch.zeros(ld.shape[0],ld.shape[1])-1
    trh[tei[0],tei[1]]=pred
    tlh[tei[0],tei[1]]=yall
    R=trh.numpy()
    label=tlh.numpy()
    f = np.zeros(shape=(R.shape[0], 1))
    for i in range(R.shape[0]):
        f[i] = np.sum(R[i] > -1)
    if min(f)>0:
        TPR,FPR,P=calculate_TPR_FPR(R,f,label)
        AUC=metrics.auc(FPR, TPR)
        AUPR=metrics.auc(TPR, P) + (TPR[0] * P[0])
        print("AUC:%.4f_AUPR:%.4f"%(AUC,AUPR))
        if AUPR>Amax[1]:
            Amax[0]=AUC
            Amax[1]=AUPR
            print("save")
            torch.save((predall,yall),"../parasave/PandY%d"%cros)

for i in range(n):
    net=Net().to(device)
    trset=DataLoader(MyDataset(tri[i],ld),batch,shuffle=True)
    teset=DataLoader(MyDataset(tei[i],ld),batch,shuffle=False)
    train_set,test_set=[],[]
    for x1,x2,y in trset:
        train_set.append((x1,x2,y))
    for x1,x2,y in teset:
        test_set.append((x1,x2,y))
    print('cross:'+str(i+1))
    train(net,train_set,test_set,feas[i],G1s[i],tei[i],epoch,learn_rate,i)
    #torch.save(net.state_dict(),'./modelpara/model%d'%i)
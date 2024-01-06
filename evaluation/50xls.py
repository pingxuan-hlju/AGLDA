import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import itertools
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
torch.cuda.empty_cache()
device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def load_data():  # lnc 240 dis 405 mi 495
    ld=np.loadtxt("../data/lnc_dis_association.txt",dtype=int)
    dd=np.loadtxt("../data/dis_sim_matrix_process.txt",dtype=float)
    md=np.loadtxt("../data/mi_dis.txt",dtype=int)
    lm=np.loadtxt("../data/yuguoxian_lnc_mi.txt",dtype=int)
    ll=np.loadtxt("../data/lnc_sim.txt",dtype=float)
    return torch.tensor(ld),torch.tensor(dd),torch.tensor(md),torch.tensor(lm),torch.tensor(ll)
def calculate_sim(ld,dd):
    s1=ld.shape[0]
    ll=torch.eye(s1)
    m2=dd*ld[:,None,:]
    m1=ld[:,:,None]
    for x,y in itertools.permutations(torch.linspace(0,s1-1,s1,dtype=torch.long),2):
        x,y=x.item(),y.item()
        m=m1[x,:,:]*m2[y,:,:]
        if ld[x].sum()+ld[y].sum()==0:
            ll[x,y]=0
        else:
            ll[x,y]=(m.max(dim=0,keepdim=True)[0].sum()+m.max(dim=1,keepdim=True)[0].sum())/(ld[x].sum()+ld[y].sum())
    return ll
def cfm(ll,ld,dd,md,lm,mm):
    r1=torch.cat([ll,ld,lm],dim=1)
    r2=torch.cat([ld.T,dd,md.T],dim=1)
    r3=torch.cat([lm.T,md,mm],dim=1)
    fea=torch.cat([r1,r2,r3],dim=0)
    deg=torch.diag((torch.sum(fea>0,dim=1))**(-1/2)).double()
    G1=deg@fea@deg
    return fea,G1
class MyDataset(Dataset):
    def __init__(self,tri,ld):
        self.tri=tri
        self.ld=ld
    def __getitem__(self,idx):
        x,y=self.tri[idx,:]
        label=self.ld[x][y]
        return x,y,label
    def __len__(self):
        return self.tri.shape[0]
batch=32
ld,dd,md,lm,ll=load_data()
mm=calculate_sim(md,dd)
fea,G1=cfm(ll,ld,dd,md,lm,mm)
ti=torch.argwhere(ld>-1)
trset=DataLoader(MyDataset(ti,ld),batch,shuffle=True)
teset=DataLoader(MyDataset(ti,ld),batch,shuffle=False)
train_set,test_set=[],[]
for x1,x2,y in trset:
    train_set.append((x1,x2,y))
for x1,x2,y in teset:
    test_set.append((x1,x2,y))
# torch.save(train_set,'../parasave/trainset.pth')
# torch.save(test_set,'../parasave/testset.pth')
# torch.save([ti,G1,fea],'../parasave/par.pth')
    
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

def train(model,train_set,test_set,fea,G1,tei,epoch,learn_rate):
    optimizer=torch.optim.Adam(model.parameters(),learn_rate,weight_decay=0.001)
    cost=nn.CrossEntropyLoss()
    model.train()
    fea,G1=fea.float().to(device),G1.float().to(device)
    Amax=[0,0]
    for i in range(epoch):
        for x1,x2,y in train_set:
            x1,x2,y=Variable(x1.long()).to(device),Variable(x2.long()).to(device),Variable(y.long()).to(device)
            out=model(x1,x2,fea,G1)
            loss=cost(out,y)+model.loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (i+1)%1 == 0: #and i+1>=50:
            print(i)
            tacc(model,test_set,fea,G1,tei,Amax)
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
def tacc(model,tset,fea,G1,tei,Amax):
    predall,yall=torch.tensor([]),torch.tensor([])
    model.eval()
    for x1,x2,y in tset:
        x1,x2,y=Variable(x1.long()).to(device),Variable(x2.long()).to(device),Variable(y.long()).to(device)
        pred=model(x1,x2,fea,G1).data
        predall=torch.cat([predall,torch.as_tensor(pred,device='cpu')],dim=0)
        yall=torch.cat([yall,torch.as_tensor(y,device='cpu')])
    pred=torch.softmax(predall,dim=1)[:,1]
    trh=torch.zeros(240,405)-1
    tlh=torch.zeros(240,405)-1
    trh[tei[:,0],tei[:,1]]=pred
    tlh[tei[:,0],tei[:,1]]=yall
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
            torch.save((predall,yall),"PandY")
# ti,G1,fea=torch.load('../parasave/par.pth')
# train_set=torch.load('../parasave/trainset.pth')
# test_set=torch.load('../parasave/testset.pth')
net=Net().to(device)
train(net,train_set,test_set,fea,G1,ti,epoch=15,learn_rate=0.0003)

# ti,_,_=torch.load('../parasave/par.pth')
pred,_=torch.load('../parasave/PandY')
pred=torch.softmax(pred,dim=1)[:,1]
trh=torch.zeros(240,405)
trh[ti[:,0],ti[:,1]]=pred
lnc_dis_score=trh
index = np.argsort(-lnc_dis_score, axis=0) 
lncRNA_name=np.loadtxt("../data/yuguoxian_lncRNA_name.txt",dtype=str)
dis_name=pd.read_csv("../data/disease_name.txt",header=None,sep='\t')
lncRNA_all_50 = index[:30,:]
lncRNA_name_50 = lncRNA_all_50.T
candidate_list = []
for i in range(lncRNA_name_50.shape[0]): # 405
    for j in range(lncRNA_name_50.shape[1]): #50
        candidate_list.append([dis_name[2][i],lncRNA_name[lncRNA_name_50[i][j]],j+1,lnc_dis_score[lncRNA_name_50[i][j]][i]])
result = open('../parasave/ST1.csv', 'w', encoding='gbk')
result.write('Disease Name,Candidate lncRNA name,Rank,Association score\n')
for m in range(len(candidate_list)): # 遍历的是405 * 50 的长度
    for n in range(len(candidate_list[m])): # 将每一行的元素求一下长度
        result.write(str(candidate_list[m][n])) # 一个一个的写入
        result.write(',') # 以 \t结束一行的写入
    result.write('\n') # 换行重新写
result.close() # 写完关闭文件
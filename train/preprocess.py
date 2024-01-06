import torch
import numpy as np
import itertools

def load_data():  # lnc 240 dis 405 mi 495
    ld=np.loadtxt("../data/lnc_dis_association.txt",dtype=int)
    dd=np.loadtxt("../data/dis_sim_matrix_process.txt",dtype=float)
    md=np.loadtxt("../data/mi_dis.txt",dtype=int)
    lm=np.loadtxt("../data/yuguoxian_lnc_mi.txt",dtype=int)
    return torch.tensor(ld),torch.tensor(dd),torch.tensor(md),torch.tensor(lm)

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

def split_dataset(al,dd,n):#5 cross
    rand_index=torch.randperm(al.sum())
    ps=torch.argwhere(al==1).index_select(0,rand_index)
    ns=torch.argwhere(al==0)
    ns=ns.index_select(0,torch.randperm(ns.shape[0]))
    sf=int(ps.shape[0]/n)
    tri,tei,lda,ll=[],[],[],[]
    for i in range(n):
        ptrn=torch.cat([ps[:(i*sf),:],ps[((i+1)*sf):(n*sf),:]],dim=0).T
        ntrn=torch.cat([ns[:(i*sf),:],ns[((i+1)*sf):(n*sf),:]],dim=0).T
        trn=torch.cat([ptrn,ntrn],dim=1)
        pten=torch.cat([ps[(i*sf):((i+1)*sf),:],ps[(n*sf):,:]],dim=0).T
        nten=torch.cat([ns[(i*sf):((i+1)*sf),:],ns[(n*sf):,:]],dim=0).T
        ten=torch.cat([pten,nten],dim=1)
        tri.append(trn)
        tei.append(ten)
        ldt=al.clone()
        ldt[pten[0,:],pten[1,:]]=0
        lda.append(ldt)
        ll.append(calculate_sim(ldt,dd))  
    return tri,tei,lda,ll

n=5
ld,dd,md,lm=load_data()
tri,tei,lda,ll=split_dataset(ld,dd,n)
mm=calculate_sim(md,dd)
datasave=(n,ld,dd,md,lm,tri,tei,lda,ll,mm)
torch.save(datasave,'../parasave/tempdata2.pth')

# n,ld,dd,md,lm,tri,tei,lda,ll,mm=torch.load('./tempdata2.pth')
def chm(m1,m2,th1=0,th2=0,th3=0):
    #th is threshold
    poi=[]
    m1idx=torch.argwhere(m1)
    for i in range(m1idx.shape[0]):
        p1,p2=m1idx[i]
        m2idx=torch.argwhere(m2[p2])
        for j in range(m2idx.shape[0]):
            poi.append([p1+th1,p2+th2,m2idx[j][0]+th3])
    return poi
def cfm(ll,ld,dd,md,lm,mm):
    #lld ldd lmd
    #  e1 e2 e3 ..
    #l
    #d
    #m
    #llidx=torch.argwhere(ll)  
    #ldidx=torch.argwhere(ld)
    #ddidx=torch.argwhere(dd)
    #mdidx=torch.argwhere(md)
    #lmidx=torch.argwhere(lm)
    
    r1=torch.cat([ll,ld,lm],dim=1)
    r2=torch.cat([ld.T,dd,md.T],dim=1)
    r3=torch.cat([lm.T,md,mm],dim=1)
    fea=torch.cat([r1,r2,r3],dim=0)
    
    deg=torch.diag((torch.sum(fea,dim=1))**(-1/2)).double()
    #G1=deg@(fea+torch.diag(torch.ones(fea.shape[0])))@deg
    G1=deg@fea@deg

    '''计算超图G2包含元路径信息
    #这里容易使得超图过大，将大矩阵分块计算
    pois1=chm(ll,ld,0,0,240)+chm(ld,dd,0,240,240)+chm(lm,md,0,645,240)
    pois=pois1
    # pois=[]
    #for l1 in pois1:   删除冗余。不过代价太大
    #    if l1 not in pois:
    #        pois.append(l1)
    H=[]
    Dvs=[]
    Des=[]
    Hsize=1000
    Hl=len(pois)//Hsize
    for i in range(Hl+1):
        if i==Hl:
            Ht=torch.zeros(1140,len(pois)-Hl*Hsize)
        else:
            Ht=torch.zeros(1140,Hsize)
        for j in range(Ht.shape[1]):
            Ht[pois[j+i*Hsize],j]=1
        Dvs.append(torch.sum(Ht,dim=1,keepdim=True))
        Des.append(torch.sum(Ht,dim=0,keepdim=True))
        H.append(Ht)
    Dv=torch.zeros(1140,1)
    for i in range(len(Dvs)):
        Dv+=Dvs[i]
    Dv=Dv**(-1/2)
    Dv[torch.where(Dv==torch.inf)]=0
    for i in range(len(Des)):
        Des[i]=Des[i]**(-1)
    for i in range(len(H)):
        H[i]=Dv*H[i]*Des[i]@H[i].T*Dv
    G2=torch.zeros(1140,1140)
    for i in range(len(H)):
        G2+=H[i]
    '''
    return fea,G1#,G2

feas=[]
G1s=[]
#G2s=[]
for i in range(n):
    fea,G1=cfm(ll[i],lda[i],dd,md,lm,mm)
    feas.append(fea)
    G1s.append(G1)
    #G2s.append(G2)
torch.save((feas,G1s),'../parasave/tempdata3.pth')
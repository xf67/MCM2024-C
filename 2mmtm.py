import pandas as pd
import numpy as np

tau=4
salt=np.e**(-1/tau)
set_salt_xp=10
game_salt_xp=8

data_all=pd.read_csv("performance.csv")
data=data_all[["DATA_NUM","set_no","game_no","point_no","per1","per2","p1_score","p2_score","p1_sets","p2_sets","p1_games","p2_games","p1_double_fault","p2_double_fault","p1_unf_err","p2_unf_err","p1_ace","p2_ace","p1_break_pt_won","p2_break_pt_won"]]
mmtm1=[]
mmtm2=[]
ge_pt_p1=[]
ge_pt_p2=[]
wwin1=0
wwin2=0
confidence1=[]
confidence2=[]
new_mmtm1=[]
new_mmtm2=[]

def p2_get():
    ge_pt_p1.append(0)
    ge_pt_p2.append(1)
    global wwin1
    global wwin2
    wwin2=wwin2+1
    wwin1=wwin1-2
    if wwin1<0 :
        wwin1=0
def p1_get():
    ge_pt_p1.append(1)
    ge_pt_p2.append(0)
    global wwin1
    global wwin2
    wwin1=wwin1+1
    wwin2=wwin2-2
    if wwin2<0 :
        wwin2=0
def atart():
    ge_pt_p1.append(0)
    ge_pt_p2.append(0)
    global wwin1
    global wwin2
    wwin1=0
    wwin2=0


for i in data["DATA_NUM"] :
    if data["point_no"][i]==1:
    # a new player
        mmtm1.append(data["per1"][i])
        mmtm2.append(data["per2"][i])
    elif data["set_no"][i]!=data["set_no"][i-1]:
    # a new set
        mmtm1.append(mmtm1[i-1]*(salt**set_salt_xp)+data["per1"][i])
        mmtm2.append(mmtm2[i-1]*(salt**set_salt_xp)+data["per2"][i])
    elif data["game_no"][i]!=data["game_no"][i-1]:
    # a new game
        mmtm1.append(mmtm1[i-1]*(salt**game_salt_xp)+data["per1"][i])
        mmtm2.append(mmtm2[i-1]*(salt**game_salt_xp)+data["per2"][i])
    else:
    # normal
        mmtm1.append(mmtm1[i-1]*salt+data["per1"][i])
        mmtm2.append(mmtm2[i-1]*salt+data["per2"][i])

    if data["p1_score"][i]==0 and data["p2_score"][i]==0:
        atart()
    elif data["p1_score"][i]==-1:
        p1_get()
    elif data["p2_score"][i]==-1:
        p2_get()
    elif data["p1_score"][i-1]==-1 and data["p1_score"][i]==40:
        p2_get()
    elif data["p1_score"][i-1]==-1 and data["p1_score"][i]==40:
        p1_get()
    elif data["p1_score"][i]>data["p1_score"][i-1]:
        p1_get()
    elif data["p2_score"][i]>data["p2_score"][i-1]:
        p2_get()
    
    set_delta = data["p1_sets"][i] - data["p2_sets"][i]
    game_delta = data["p1_games"][i] - data["p2_games"][i]
    score_delta = 0
    if data["p1_score"][i] < 0:
        score_delta = 1
    elif data["p2_score"][i] < 0:
        score_delta = -1
    # AD
    elif data["p1_score"][i] <= 10 and data["p2_score"][i] <= 10:
        score_delta = data["p1_score"][i]-data["p2_score"][i]
    # QIANG 7
    else:
        p1_score_ = (data["p1_score"][i]+6)//15
        p2_score_ = (data["p2_score"][i]+6)//15
        score_delta = p1_score_-p2_score_

    technic1=data["p1_break_pt_won"][i]*(-4)+data["p1_ace"][i]*4-data["p1_unf_err"][i]*5-data["p1_double_fault"][i]*5
    technic2=data["p2_break_pt_won"][i]*(-4)+data["p2_ace"][i]*4-data["p2_unf_err"][i]*5-data["p2_double_fault"][i]*5

    confidence1.append(wwin1*0.5+set_delta*0.9+game_delta*0.5+score_delta*0.4+technic1*1)
    confidence2.append(wwin2*0.5-set_delta*0.9-game_delta*0.5-score_delta*0.4+technic2*1)

for i in data["DATA_NUM"] :
    new_mmtm1.append(mmtm1[i]*0.5) 
    #new_mmtm1.append(0) 
    if confidence1[i]>=2:
        new_mmtm1[i]+= confidence1[i]
    new_mmtm2.append(mmtm2[i]*0.5) 
    #new_mmtm2.append(0) 
    if confidence2[i]>=2:
        new_mmtm2[i]+= confidence2[i]


data_all["mmtm1"]=mmtm1
data_all["mmtm2"]=mmtm2
data_all["new_mmtm1"]=new_mmtm1
data_all["new_mmtm2"]=new_mmtm2
data_all.to_csv('data_w_mmtm.csv', index=False)

mmtm1_ = mmtm1[:-1]
mmtm2_ = mmtm2[:-1]
new_mmtm1_ = new_mmtm1[:-1]
new_mmtm2_ = new_mmtm2[:-1]
ge_pt_p1_ =ge_pt_p1[1:]
ge_pt_p2_ =ge_pt_p2[1:]

ge_5pt_p1_=[]
ge_5pt_p2_=[]

for i,_ in enumerate(ge_pt_p1_):
    # ge_5pt_p1_.append(ge_pt_p1_[i])
    # ge_5pt_p2_.append(ge_pt_p2_[i])
    try:
        ge_5pt_p1_.append(ge_pt_p1_[i]+ge_pt_p1_[i+1]*salt**4+ge_pt_p1_[i+2]*salt**8+ge_pt_p1_[i+3]*salt**16)
        ge_5pt_p2_.append(ge_pt_p2_[i]+ge_pt_p2_[i+1]*salt**4+ge_pt_p2_[i+2]*salt**8+ge_pt_p2_[i+3]*salt**16)
    except:
        try:
            ge_5pt_p1_.append(ge_pt_p1_[i]+ge_pt_p1_[i+1]*salt**4+ge_pt_p1_[i+2]*salt**8)
            ge_5pt_p2_.append(ge_pt_p2_[i]+ge_pt_p2_[i+1]*salt**4+ge_pt_p2_[i+2]*salt**8)
        except:
            try:
                ge_5pt_p1_.append(ge_pt_p1_[i]+ge_pt_p1_[i+1]*salt**4)
                ge_5pt_p2_.append(ge_pt_p2_[i]+ge_pt_p2_[i+1]*salt**4)
            except:
                ge_5pt_p1_.append(sum(ge_pt_p1_[i:-1]))
                ge_5pt_p2_.append(sum(ge_pt_p2_[i:-1]))


cov1=np.cov(mmtm1_,ge_5pt_p1_,ddof=0)[1][0]
cov2=np.cov(mmtm2_,ge_5pt_p2_,ddof=0)[1][0]
dx1=np.var(mmtm1_)
dy1=np.var(ge_5pt_p1_)
rho1=cov1/np.sqrt(dx1*dy1)
dx2=np.var(mmtm2_)
dy2=np.var(ge_5pt_p2_)
rho2=cov2/np.sqrt(dx2*dy2)
print(rho1,rho2)

cov1=np.cov(new_mmtm1_,ge_5pt_p1_,ddof=0)[1][0]
cov2=np.cov(new_mmtm2_,ge_5pt_p2_,ddof=0)[1][0]
dx1=np.var(new_mmtm1_)
dy1=np.var(ge_5pt_p1_)
rho1=cov1/np.sqrt(dx1*dy1)
dx2=np.var(new_mmtm2_)
dy2=np.var(ge_5pt_p2_)
rho2=cov2/np.sqrt(dx2*dy2)
print(rho1,rho2)


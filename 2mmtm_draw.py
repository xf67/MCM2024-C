import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_all=pd.read_csv("data_w_mmtm.csv")
data=data_all[["DATA_NUM","set_no","game_no","point_no","mmtm1","mmtm2","new_mmtm1","new_mmtm2","p1_score","p2_score","p1_sets","p2_sets","p1_games","p2_games"]]

# head=0
# num=0

# fig=plt.figure(dpi=1000)
# fig.set_size_inches(8,8)

# for i in data["DATA_NUM"]:
#     if i+1==len(data["DATA_NUM"]) or data["point_no"][i+1]<data["point_no"][i] :
#         num+=1
#         plt.subplot(6, 6, num)
#         plt.plot(data["point_no"][head:i],data["per1"][head:i],linewidth=1)
#         plt.plot(data["point_no"][head:i],data["per2"][head:i],linewidth=1)

# plt.show()


range_lines=(6950,7283)
fig=plt.figure()
# plt.subplot(3,1,1)
# plt.plot(data["point_no"][range_lines[0]:range_lines[1]],data["new_mmtm1"][range_lines[0]:range_lines[1]],linewidth=1,color='r') #CA
# plt.subplot(3,1,2)
# plt.plot(data["point_no"][range_lines[0]:range_lines[1]],data["new_mmtm2"][range_lines[0]:range_lines[1]],linewidth=1,color='c') #ND
# plt.subplot(3,1,3)

# delta = [(x-y) for x, y in list(zip(data["new_mmtm1"],data["new_mmtm2"]))]
# zeros = [0 for x in data["new_mmtm1"] ]
# twos = [2 for x in data["new_mmtm1"] ]
# twos_ = [-2 for x in data["new_mmtm1"] ]
# plt.plot(data["point_no"][range_lines[0]:range_lines[1]],delta[range_lines[0]:range_lines[1]],linewidth=2,color='b') #DELTA
# plt.plot(data["point_no"][range_lines[0]:range_lines[1]],twos[range_lines[0]:range_lines[1]],linewidth=1,color='m') 
# plt.plot(data["point_no"][range_lines[0]:range_lines[1]],twos_[range_lines[0]:range_lines[1]],linewidth=1,color='m') 

# for i in range(range_lines[0],range_lines[1]):
#     if i==range_lines[1] or data["game_no"][i]!=data["game_no"][i+1] :
#         lines = [data["point_no"][i] for x in range(-7,6)]
#         grows = [x for x in range(-7,6)]
#         plt.plot(lines,grows,linewidth=1,color='grey')
#         plt.text(data["point_no"][i], -8, f'{data["p1_sets"][i]}:{data["p2_sets"][i]} \n {data["p1_games"][i]}:{data["p2_games"][i]} \n {data["p1_score"][i]}:{data["p2_score"][i]} ', horizontalalignment='center', verticalalignment='bottom',fontsize=5)
# plt.show()

windows=5
p_new_mmtm1=pd.DataFrame(data["new_mmtm1"][range_lines[0]:range_lines[1]])
rp_new_mmtm1=p_new_mmtm1.rolling(window=windows).mean()
r_new_mmtm1=rp_new_mmtm1.values.tolist()
p_new_mmtm2=pd.DataFrame(data["new_mmtm2"][range_lines[0]:range_lines[1]])
rp_new_mmtm2=p_new_mmtm2.rolling(window=windows).mean()
r_new_mmtm2=rp_new_mmtm2.values.tolist()
for i in range(0,windows-1):
    r_new_mmtm1[i][0]=0
    r_new_mmtm2[i][0]=0

data1=pd.DataFrame({'Player1mmtm': r_new_mmtm1, 'Player2mmtm': r_new_mmtm2})
data1.to_csv("data_w2_mmtm.csv")



r_delta=[(x[0]-y[0]) for x, y in list(zip(r_new_mmtm1,r_new_mmtm2))]
zeros = [0 for x in r_new_mmtm1 ]
twos = [1.5 for x in r_new_mmtm1 ]
twos_ = [-1.5 for x in r_new_mmtm1 ]
plt.plot(data["point_no"][range_lines[0]:range_lines[1]],r_delta,linewidth=2,color='b') #DELTA
plt.plot(data["point_no"][range_lines[0]:range_lines[1]],twos,linewidth=1,color='m') 
plt.plot(data["point_no"][range_lines[0]:range_lines[1]],twos_,linewidth=1,color='m') 
for i in range(range_lines[0],range_lines[1]):
    if i==range_lines[1] or data["game_no"][i]!=data["game_no"][i+1] :
        lines = [data["point_no"][i] for x in range(-5,5)]
        grows = [x for x in range(-5,5)]
        plt.plot(lines,grows,linewidth=1,color='grey')
        plt.text(data["point_no"][i], -6, f'{data["p1_sets"][i]}:{data["p2_sets"][i]} \n {data["p1_games"][i]}:{data["p2_games"][i]} \n {data["p1_score"][i]}:{data["p2_score"][i]} ', horizontalalignment='center', verticalalignment='bottom',fontsize=5)
plt.show()
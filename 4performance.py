from math import sqrt, exp, atan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#df = pd.read_csv('Wimbledon_featured_matches.csv')
df = pd.read_csv('Wim4.csv')
data = df.values
games_cnt=2

# TENSION
weight = {'game_point': 0.1, 'set_game': 0.2, 'match_set': 0.3, 'set_gap': 0.1, 'game_gap':0.08, 'score_gap':0.04, 'tech':0.1 ,'phys':0.1}
tension_p1 = []
tension_p2=[]

for i in range(len(data)):
    p1_sets = data[i][7]
    p2_sets = data[i][8]
    p1_games = data[i][9]
    p2_games = data[i][10]
    p1_score = data[i][11]
    p2_score = data[i][12]
    sum = 0
    if p1_score == 40 or p2_score == 40:
        for j in range(i - 1, -1, -1):
            # print(j)
            sum += 1
            if data[j][11] != 40 and data[j][12] != 40:
                break
    sum = sum * weight['game_point']
    cnt = 0
    if (6 <= p1_score < 15 or 6 <= p2_score < 15) and data[i][4] != 5 or (9 <= p1_score < 15 or 9 <= p2_score < 15) and data[i][4] == 5:
        for j in range(i-1, -1, -1):
            cnt += 1
            if (data[j][11] < 6 and data[j][12] < 6) and data[i][4] != 5 or (data[j][11] < 9 and data[j][12] < 9) and data[i][4] == 5:
                break
    sum += 2 * weight['game_point'] * cnt
    if p1_games == 6 and p2_games == 6:
        sum += 2 * weight['set_game']
    elif p1_games == 6 or p2_games == 6:
        sum += weight['set_game']
    if p1_sets == 2 and p2_sets == 2:
        sum += 2 * weight['match_set']
    elif p1_sets == 2 or p2_sets == 2:
        sum += weight['match_set']

    set_delta = p1_sets - p2_sets
    game_delta = p1_games - p2_games
    score_delta = 0
    if p1_score < 0:
        score_delta = 1
    elif p2_score < 0:
        score_delta = -1
    # AD
    elif p1_score <= 10 and p2_score <= 10:
        score_delta = p1_score-p2_score
    # QIANG 7
    else:
        p1_score_ = (p1_score+6)//15
        p2_score_ = (p2_score+6)//15
        score_delta = p1_score_-p2_score_
    # NORMAL
    # print(set_delta," ",game_delta," ",score_delta)
    sump2=sum+set_delta*weight['set_gap']+game_delta*weight['game_gap']+score_delta*weight['score_gap']
    sum -= set_delta*weight['set_gap']+game_delta*weight['game_gap']+score_delta*weight['score_gap']
    tension_p1.append(sum)
    tension_p2.append(sump2)
    #print(f"tension: {round(100*sum)},  {round(100*sump2)}")

# TECHNIC, PLAYER INFO, PHYSICAL LOSS
technic_p1=[]
technic_p2=[]
rank_p1=[]
rank_p2=[]
phys_p1=[]
phys_p2=[]
age_multiplier=0.9
age_avg=26.5
dis_threshold=1.3
for score in data:
    if score[13]==1:
        sump1=-1
        sump2=1
    else:
        sump1=1
        sump2=-1
    if score[20]==1:
        sump1+=3
    if score[21]==1:
        sump2+=3
    if score[25]==1:
        sump1-=2
    if score[26]==1:
        sump2-=2
    if score[27]==1:
        sump1-=1.5
    if score[28]==1:
        sump2-=1.5
    if score[35]==1:
        sump1+=4
    if score[36]==1:
        sump2+=4
    technic_p1.append(sump1)
    technic_p2.append(sump2)
    #print(f"tech: {round(100*sump1)},  {round(100*sump2)}")
    rank_p1.append(score[46])
    rank_p2.append(score[47])
    dis_p1=score[39]
    dis_p2=score[40]
    if score[48]>age_avg:
        dis_p1*=age_multiplier
    if score[49]>age_avg:
        dis_p2*=age_multiplier
    if dis_p1==0 and dis_p2==0:
        sump1=0
        sump2=0
    elif dis_p1==-1 or dis_p1*dis_p2==0:
        sump1='artist'
        sump2='artist'
    elif dis_p1>dis_p2*dis_threshold:
        sump1=-dis_p1/dis_p2
        sump2=dis_p1/dis_p2
    elif dis_p2>dis_p1*dis_threshold:
        sump2=-dis_p2/dis_p1
        sump1=dis_p2/dis_p1
    else:
        sump1=0
        sump2=0
    phys_p1.append(0)
    phys_p2.append(0)
    

# MIXING TENSION, TECHNIC, PHYSICAL LOSS
performance=[]
for i in range(len(data)):
    p1_win=True
    if data[i][15]==2:
        p1_win=False
    # if i==0 and data[i][12]>0:
    #     p1_win=False
    # elif data[i][11]==0 and data[i][12]==0 and data[i-1][12]==-1:
    #     p1_win=False
    # elif data[i][11]==0 and data[i][12]==0 and data[i-1][12]==40 and data[i-1][11]<40:
    #     p1_win=False
    # elif data[i][11]==0 and data[i][12]==0 and data[i-1][12]==7 and data[i-1][11]<40:
    if p1_win:
        prf_p1=1
        prf_p2=-1
    else:
        prf_p1=-1
        prf_p2=1
    if phys_p1[i]=='artist':
        performance.append([prf_p1,prf_p2])
        continue
    print(f"tech:{technic_p1[i]},phys:{phys_p1[i]},tension:{sqrt(exp(atan(tension_p1[i])))},prf_p1:{prf_p1}")
    prf_p1=(prf_p1+technic_p1[i]*weight['tech']+phys_p1[i]*weight['phys'])*sqrt(exp(atan(tension_p1[i])))
    prf_p2=(prf_p2+technic_p2[i]*weight['tech']+phys_p2[i]*weight['phys'])*sqrt(exp(atan(tension_p2[i])))
    
    performance.append([prf_p1,prf_p2])

# print(performance)
# output=pd.DataFrame(performance)
# output.to_csv('example.csv', mode='a', index=False,)
    
# prf_avg=[]
# cnt=[]
# rank=[]
# for i in range(games_cnt):
#     prf_avg.append(0)
#     cnt.append(0)
#     rank.append(0)
# for i in range(len(data)):
#     for j in range(games_cnt):
#         if data[i][1]==data[j][52]:
#             prf_avg[j]+=performance[i][0]
#             cnt[j]+=1
#         if data[i][2]==data[j][52]:
#             prf_avg[j]+=performance[i][1]
#             cnt[j]+=1
            
# print(cnt)
# for i in range(games_cnt):
#     prf_avg[i]/=cnt[i]
#     prf_avg[i]*=200
#     prf_avg[i]=round(prf_avg[i])
#     rank[i]=np.log(data[i][53])
# # print(prf_avg)
# # print(rank)
# plt.scatter(prf_avg, rank)
# plt.show()

performance=np.array(performance)
df["per1"]=performance[:,0]
df["per2"]=performance[:,1]
df.to_csv('performance4.csv',index=False)


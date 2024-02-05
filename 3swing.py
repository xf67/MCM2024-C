import pandas as pd

df=pd.read_excel("super_smooth.xlsx")
data=df.values

parameters=(15,1.2,23)
swings=[]
cnt=0

def slopy(i,j,length):
    MIN=100
    MAX=-100
    # if cnt==0:
    #     prev=0
    # else:
    #     prev=swings[j][cnt-1][0]
    prev=0
    follow=length
    for k in range(i-1,-1,-1):
        if data[k][j]*data[k+1][j]<=0:
            prev=k
            break
    for k in range(i+1,length-1):
        if data[k][j]*data[k+1][j]<=0:
            follow=k
            break
    for k in range(max(prev,i-parameters[0]),min(follow,i+1+parameters[0])):
        if k==length:
            break
        if MIN>data[k][j]:
            MIN=data[k][j]
            MINK=k
        if MAX<data[k][j]:
            MAX=data[k][j]
            MAXK=k
    # print(MAXK,MAX,MINK,MIN)
    if MAX-MIN>parameters[1]:
        return MAXK-MINK
    return 114514

for j in range(0,31):
    cnt=0
    length=None
    swings.append([])
    for i in range(120,1000):
        if i==len(data):
            length=i
            break
        if not -100<data[i][j]<100:
            length=i
            break
    for i in range(0,length-1):
        tmp=slopy(i,j,length)
        if data[i][j]*data[i+1][j]<=0 and tmp!=114514:
            if j==25:
                print(i,cnt,tmp)
            if cnt>0 and swings[j][cnt-1][0]>i-parameters[2]:
                if swings[j][cnt-1][1]*tmp<=0:
                    cnt-=1
                    swings[j].pop()
                else:
                    swings[j][cnt-1][0]=(i+swings[j][cnt-1][0])//2
            elif i<length-19:
                swings[j].append([i,tmp])
                cnt+=1

output=[]
for i in range(0,31):
    output.append([])
    for j in swings[i]:
        if j[0]>19:
            output[i].append(j[0])
print(output)
df=pd.DataFrame(output)
df.to_excel('swings2.xlsx')
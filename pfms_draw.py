import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_all=pd.read_csv("performance.csv")
data=data_all[["DATA_NUM","set_no","game_no","point_no","per1","per2"]]

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

fig=plt.figure()
plt.subplot(2,1,1)
plt.plot(data["point_no"][6950:-1],data["per1"][6950:-1],linewidth=1,color='r') #CA
plt.subplot(2,1,2)
plt.plot(data["point_no"][6950:-1],data["per2"][6950:-1],linewidth=1,color='c') #ND
plt.show()
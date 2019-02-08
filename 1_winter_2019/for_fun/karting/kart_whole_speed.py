import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('alldata.xls',header=None)
n = data.shape[0]
index = []
for i in range(n):
    if float(data.iloc[i])>90:
        index.append(i)

anomaly = data.iloc[index]
print('anomaly: ', anomaly)
dropped = data.drop(index)

plt.ion()
plt.plot(data)
plt.grid()
plt.show()
plt.savefig('./Team31.jpg')

"""flag0 = 0
flag1 = index[0]
plt.figure()
plt.plot(data.iloc[flag0:flag1])

flag0 = flag1 + 1
flag1 = index [1]
plt.figure()
plt.plot(data.iloc[flag0:flag1])
"""
index_plot = index
index_plot.append(279)
flag0 = 0
legend = ['1: Bobo1','2: AFu1','3: Xiaobai1','3: Xiaobai2','4: Xiaopan','5: AFu2','6: Bobo2']
for i in range(len(index)):
    flag1 = index[i]
    plt.figure()
    plt.grid()
    plt.plot(data.iloc[flag0:flag1],label=legend[i])
    plt.legend()
    plt.savefig(legend[i])
    flag0 = flag1 + 1

ax= anomaly.plot.barh()
plt.savefig('anomalytime.jpg')

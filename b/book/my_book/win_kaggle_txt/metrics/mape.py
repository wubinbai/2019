import matplotlib.pyplot as plt
import numpy as np
data = np.array([10,20,30,40,50,270])

samples = np.array(list(range(300)))

def compute_mae(i,data):
    err = 0
    n = len(data)
    for j in data:
        err += np.abs(j - i)
    mse = err/n
    return mse


def compute_mape(i,data):
    err = 0
    n = len(data)
    for j in data:
        err += np.abs((j - i)/j)
    mape = err/n
    return mape

mape = []
for i in samples:
    result = compute_mape(i,data)
    mape.append(result)
#print(mape)
print(np.argmin(mape))
plt.plot(mape)
plt.ion()
plt.show()


'''
data2 = np.array([10,20,30,31,32,32,33,34,35,36,40,50,270])

samples = np.array(list(range(300)))



mae2 = []
for i in samples:
    result = compute_mae(i,data2)
    mae2.append(result)
#print(mae2)
print(np.argmin(mae2))

counts = np.bincount(data2)
#返回众数
print(np.argmax(counts))

'''

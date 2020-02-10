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

mae = []
for i in samples:
    result = compute_mae(i,data)
    mae.append(result)
#print(mae)
print(np.argmin(mae))

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

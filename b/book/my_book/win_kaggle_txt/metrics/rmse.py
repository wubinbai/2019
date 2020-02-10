import numpy as np
data = np.array([10,20,30,40,50,270])

samples = np.array(list(range(300)))

def compute_mse(i,data):
    err = 0
    n = len(data)
    for j in data:
        err += (j - i)**2
    mse = err/n
    return mse
def compute_rmse(i,data):
    temp = compute_mse(i,data)
    return temp**0.5
rmse = []

for i in samples:
    result = compute_rmse(i,data)
    rmse.append(result)
#print(rmse)
print(np.argmin(rmse))

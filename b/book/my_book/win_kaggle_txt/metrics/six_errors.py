import numpy as np

data = np.array([5,9,8,6,27])
li = list(range(300))
samples = [j/10 for j in li]
samples = np.array(samples)

def compute_mae(i,data):
    err = 0
    n = len(data)
    for j in data:
        err += np.abs(j - i)
    mse = err/n
    return mse
    

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


def compute_mape(i,data):
    err = 0
    n = len(data)
    for j in data:
        err += np.abs((j - i)/j)
    mape = err/n
    return mape

def compute_mspe(i,data):
    err = 0
    n = len(data)
    for j in data:
        err += ((j - i)/j)**2
        mspe = err/n
    return mspe

def compute_rmsle(i,data):
    log0 = np.log(i+1)
    log1 = np.log(data+1)
    rmsle = compute_rmse(log0,log1)
    return rmsle
    ''''
    err = 0
    n = len(data)
    for j in data:
        err += ((j - i)/j)**2
        rmsle = err/n
    return rmsle
    '''

grid = [[] for i in range(6) ]

for i in samples:
    temp = compute_mae(i,data)
    grid[0].append(temp)

    temp = compute_mse(i,data)
    grid[1].append(temp)

    temp = compute_rmse(i,data)
    grid[2].append(temp)

    temp = compute_mape(i,data)
    grid[3].append(temp)

    temp = compute_mspe(i,data)
    grid[4].append(temp)

    temp = compute_rmsle(i,data)
    grid[5].append(temp)

print('val pred for mae, mse, rmse, mape, mspe, rmsle: ')
ans = [np.argmin(j) for j in grid]
ans = [k/10 for k in ans]
print(ans)


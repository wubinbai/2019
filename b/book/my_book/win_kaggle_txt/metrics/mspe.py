import matplotlib.pyplot as plt
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
def compute_mspe(i,data):
    err = 0
    n = len(data)
    for j in data:
        err += ((j - i)/j)**2
    mse = err/n
    return mse

mspe = []
for i in samples:
    result = compute_mspe(i,data)
    mspe.append(result)
#print(mspe)
print(np.argmin(mspe))


plt.figure()
plt.ion()
plt.plot(mspe)

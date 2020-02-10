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

mse = []
for i in samples:
    result = compute_mse(i,data)
    mse.append(result)
#print(mse)
print(np.argmin(mse))


plt.figure()
plt.ion()
plt.plot(mse)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming data
x = np.array([[0,1,1],[0,0,1],[1,1,1],[1,0,1]])
y = np.array([[0,0,1,1]]).T
# Setting Random Seed
np.random.seed(0)
# Three layer: 3 by one parameters

weights1 = 2 * np.random.random((3,4))-1
weights2 = 2 * np.random.random((4,1))-1
 
# Prepare sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

error2=0
err_cum = []
# Let's loop and train our neural network!
for i in range(300):
    et=error2
    z1 = np.dot(x,weights1)# z1 is 4 by 4
    a1 = sigmoid(z1)# a1 is also 4 by 4
    z2 = np.dot(a1,weights2) # z2 is 4 by 1
    a2 = sigmoid(z2) # a2 is 4 by 1
    error2 = a2 - y # error is 4 by 1

    # update
    weights2 = weights2 - np.dot(a1.T, error2 * a2 * (1-a2)) # a * (1-a) represents the derivative of a sigmoid function since the derivative of 1/(1+e**(-x)) is "the sigmoid function multiplied by 1 minus the sigmoid function"
    error1 = (error2 * a2 * (1-a2)).dot(weights2.T)
    weights1 = weights1 - np.dot(x.T, error1 * a1 * (1-a1))
    err_cum.append(error2)
print('Predicted Result: ', a2)

x1,x2,x3,x4 = [],[],[],[]
for i in range(len(err_cum)):
    x1.append(float(err_cum[i][0])**2)
    x2.append(float(err_cum[i][1])**2)
    x3.append(float(err_cum[i][2])**2)
    x4.append(float(err_cum[i][3])**2)
    
plt.subplot(1,4,1)
plt.plot(range(len(x1)),x1)
plt.subplot(1,4,2)
plt.plot(range(len(x2)),x2)
plt.subplot(1,4,3)
plt.plot(range(len(x3)),x3)
plt.subplot(1,4,4)
plt.plot(range(len(x4)),x4)

plt.plot()
plt.show()

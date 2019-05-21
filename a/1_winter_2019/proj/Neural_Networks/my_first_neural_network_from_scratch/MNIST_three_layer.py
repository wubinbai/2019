import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

# import data

data = pd.read_csv("train.csv")
x_train=data.iloc[:210,1:]
y_train=data.iloc[:210,:1]
x_test=data.iloc[210:420,1:]
y_test=data.iloc[210:420,:1]
# Assuming:
x = x_train # 210 by 784  
y = y_train # 210 by 1
ya=np.array(y)
yzeros=np.zeros((210,10))
for i in range(len(yzeros)):
    temp = ya[i]
    yzeros[i][temp]=1
y=yzeros # 210 by 10

# Setting Random Seed
np.random.seed(0)
# Three layer: 3 by one parameters

weights1 = 2 * np.random.random((784,16))-1
weights2 = 2 * np.random.random((16,10))-1
 
# Prepare sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))
# Define cost function
def compute_cost(pred,true):
    cost = np.sum((pred-true)**2)/(2 * len(true))
    return cost



# Define learning rate alpha:
alpha = 0.03

error2 = 0
err_cum = [] 
cost = []
# Let's loop and train our neural network!
for i in range(800):
    et = error2
    z1 = np.dot(x,weights1)# z1 is 210 by 16
    a1 = sigmoid(z1)# same size as z1
    z2 = np.dot(a1,weights2) # z2 is 210 by 10
    a2 = sigmoid(z2) # same size as z2
    error2 = a2 - y # same size as a2

    # update
    weights2 = weights2 - alpha * np.dot(a1.T, error2 * a2 * (1-a2)) # a * (1-a) represents the derivative of a sigmoid function since the derivative of 1/(1+e**(-x)) is "the sigmoid function multiplied by 1 minus the sigmoid function"
    error1 = (error2 * a2 * (1-a2)).dot(weights2.T)
    weights1 = weights1 - alpha * np.dot(x.T, error1 * a1 * (1-a1))
    err_cum.append(error2)
    cost.append(compute_cost(a2,y))
print('Predicted Result: ', a2)


plt.plot(cost)
print('compare a2 [0] and y [0] +======')
print(a2[0])
print(y[0])
"""
x1,x2,x3,x4 = [],[],[],[]
for i in range(4):#range(len(err_cum)):
    x1.append(float(err_cum[i][0])**2)
    x2.append(float(err_cum[i][1])**2)
    x3.append(float(err_cum[i][2])**2)
    x4.append(float(err_cum[i][3])**2)

plt.ion()
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
"""

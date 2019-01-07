import numpy as np
import pandas as pd
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

# Let's loop and train our neural network!
for i in range(27000):
    z1 = np.dot(x,weights1)# z1 is 210 by 16
    a1 = sigmoid(z1)# same size as z1
    z2 = np.dot(a1,weights2) # z2 is 210 by 10
    a2 = sigmoid(z2) # same size as z2
    error2 = a2 - y # same size as a2

    # update
    weights2 = weights2 - np.dot(a1.T, error2 * a2 * (1-a2)) # a * (1-a) represents the derivative of a sigmoid function since the derivative of 1/(1+e**(-x)) is "the sigmoid function multiplied by 1 minus the sigmoid function"
    error1 = (error2 * a2 * (1-a2)).dot(weights2.T)
    weights1 = weights1 - np.dot(x.T, error1 * a1 * (1-a1))

print('Predicted Result: ', a2)


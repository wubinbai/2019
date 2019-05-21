import numpy as np
#np.set_printoptions(threshold=np.nan)
import pandas as pd

# import data

data = pd.read_csv("train.csv")
# n=int(input('Please enter the number of trainng sets(samples) you would like to train our model: '))

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
# Two layer: 3 by one parameters
weights = 2 * np.random.random((784,10))-1
# Prepare sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Let's loop and train our neural network!
for i in range(9000):
    z1 = np.dot(x,weights)# z1 is 210 by 10
    a1 = sigmoid(z1)# a1 is also 210 by 10
    error = a1 - y # error is 210 by 10

    # update
    weights = weights - 0.01 * np.dot(x.T, error * a1 * (1-a1)) # a1 * (1-a1) represents the derivative of a sigmoid function since the derivative of 1/(1+e**(-x)) is "the sigmoid function multiplied by 1 minus the sigmoid function"
    
print('Predicted Result: ', a1)

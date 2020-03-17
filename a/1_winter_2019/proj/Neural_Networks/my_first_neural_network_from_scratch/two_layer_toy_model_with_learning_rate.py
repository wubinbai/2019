import numpy as np
# Assuming:
x = np.array([[0,1,1],[0,0,1],[1,1,1],[1,0,1]])
y = np.array([[0,0,1,1]]).T
# Setting Random Seed
np.random.seed(0)
# Two layer: 3 by one parameters
weights = 2 * np.random.random((3,1))-1
# Prepare sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Let's loop and train our neural network!
for i in range(9000):
    z1 = np.dot(x,weights)# z1 is 4 by 1
    a1 = sigmoid(z1)# a1 is also 4 by 1
    error = a1 - y # error is 4 by 1

    # update
    weights = weights - 0.01 * np.dot(x.T, error * a1 * (1-a1)) # a1 * (1-a1) represents the derivative of a sigmoid function since the derivative of 1/(1+e**(-x)) is "the sigmoid function multiplied by 1 minus the sigmoid function"
    
print('Predicted Result: ', a1)
print("Let's see how our 2-layer-neural-network works with new input(e.g. x = np.array([[1,0,0]]))")
x = np.array([[1,0,0]])
z1 = np.dot(x,weights)# z1 is 4 by 1
a1 = sigmoid(z1)# a1 is also 4 by 1
print("predicted result of [1,0,0]: ", a1)

x = np.array([[0,1,0]])
z1 = np.dot(x,weights)# z1 is 4 by 1
a1 = sigmoid(z1)# a1 is also 4 by 1
print("predicted result of [0,1,0]: ", a1)


x = np.array([[0,0,1]])
z1 = np.dot(x,weights)# z1 is 4 by 1
a1 = sigmoid(z1)# a1 is also 4 by 1
print("predicted result of [0,0,1]: ", a1)

x = np.array([[1,1,0]])
z1 = np.dot(x,weights)# z1 is 4 by 1
a1 = sigmoid(z1)# a1 is also 4 by 1
print("predicted result of [1,1,0]: ", a1)


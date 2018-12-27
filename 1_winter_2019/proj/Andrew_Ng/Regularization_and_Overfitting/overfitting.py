import numpy as np
import matplotlib.pyplot as plt
# Assuming we have dataset
x=np.linspace(0,10,20)
y=np.sqrt(x)+np.random.random(20)
# First plot them
plt.plot(x,y)
# plt.show()
# Suppose we want to fit them to the 6th polynomial y = theta0 + theta1 * x + theta2 * x**2 + ... + theta6 * x**2
# Using Gradient Descent
# Randomly assign chose theta0 to theta6
theta = np.random.random(7)
# Define hypothesis
def h(x):
    global theta
    global x_poly
    x_poly = np.array([x**i for i in range(7)]) 
    y = theta * x_poly
    return y


n = 50000 # iterate over n times
number = 0
alpha = 0.01
while number < n:
    h_val = []
    for i in range(len(x)):
        h_val.append(h(x[i]))
    h_minus_y=[]
    for i in range(len(h_val)):
        h_minus_y.append(h_val[i]-y[i])
    temp=np.zeros(7)
#    temp[0]=theta[0]-alpha*sum(h_minus_y)/len(x)
    for i in range(7):
        temp[i]=theta[i]-alpha*sum(np.array(h_minus_y) * x_poly[i]) /len(x_poly)
    theta=temp
    number+=1

print("theta: ", theta)

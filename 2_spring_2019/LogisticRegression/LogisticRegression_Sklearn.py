from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ex2data1.txt',delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
fig=plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.show()

LR = LogisticRegression()
LR.fit(X,y)

print("LR comes from LogisticRegression.fit(X,y)")

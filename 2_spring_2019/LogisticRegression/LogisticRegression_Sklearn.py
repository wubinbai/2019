from sklearn.linear_model import LogisticRegression
import numpy as np
data = np.loadtxt('ex2data1.txt',delimiter=',')
X = data[:, 0:2]
y = data[:, 2]

LR = LogisticRegression()
LR.fit(X,y)



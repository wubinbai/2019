import numpy as np
import matplotlib.pyplot as plt
x = np.array([[0,1],[1,0],[0,2],[1,0.8],[4,0.5],[2,0.5]])
y = np.array([1,0,1,1,0,0])
from sklearn.svm import LinearSVC
from sklearn import svm

svm_clf = LinearSVC()
svm_clf.fit(x,y)
clf = svm.SVC(kernel='linear')
clf.fit(x,y)

x_linspace = np.linspace(0,10,20)
y_line = ( - svm_clf.intercept_[0] - x_linspace * svm_clf.coef_[0][0] ) / svm_clf.coef_[0][1] 
plt.figure()
plt.scatter(x_linspace,y_line,marker='*')
plt.scatter(x[:,0],x[:,1],marker='o',color='y')
plt.ion()
plt.show()

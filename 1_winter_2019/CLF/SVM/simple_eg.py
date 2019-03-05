import numpy as np
x = np.array([[0,1],[1,0],[0,2]])
y = np.array([1,0,1])
from sklearn.svm import LinearSVC
from sklearn import svm

svm_clf = LinearSVC()
svm_clf.fit(x,y)
clf = svm.SVC(kernel='linear')
clf.fit(x,y)


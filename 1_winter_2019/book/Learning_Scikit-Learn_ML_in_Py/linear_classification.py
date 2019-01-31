from sklearn import datasets
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

X, y = X_iris[:,:2], y_iris
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=33)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib.pyplot as plt

plt.ion()
colors = ['red','greenyellow','blue']
for i in range(len(colors)):
    xs = X_train[:,0][y_train==i]
    ys = X_train[:,1][y_train==i]
    plt.scatter(xs,ys,c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(X_train, y_train)

# print(clf.coef_,clf.intercept_)

x_min, x_max = X_train[:,0].min()-0.5, X_train[:,0].max()+0.5

y_min, y_max = X_train[:,1].min()-0.5, X_train[:,1].max()+0.5

import numpy as np
xs = np.arange(x_min,x_max,0.5)
fig,axes = plt.subplots(1,3)
fig.set_size_inches(10,6)
for i in [0,1,2]:
    axes[i].set_aspect('equal')
    axes[i].set_title('Class '+str(i)+' versus the rest')
    axes[i].set_xlabel('Sepal Length')
    axes[i].set_ylabel('Sepal Width')
    axes[i].set_xlim(x_min,x_max)
    axes[i].set_ylim(y_min,y_max)
 #   sca(axes[i])
    plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.prism)
    ys=(-clf.intercept_[i]-xs*clf.coef_[i,0])/clf.coef_[i,1]
    plt.plot(xs,ys,hold=True)

from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline
clf =  Pipeline([('scaler', StandardScaler()),('linear_model', SGDClassifier())])
cv = KFold(X.shape[0], 5 ,shuffle=True, random_state = 33)
score = cross_val_score(clf,X,y,cv=cv)
print(scores)

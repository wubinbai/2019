import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
boston=load_boston()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(boston.data,boston.target,test_size=0.25,random_state=33)
from sklearn.preprocessing import StandardScaler
scalerX=StandardScaler().fit(X_train)
scalery=StandardScaler().fit(y_train)
X_train = scalerX.transform(X_train)
y_train = scalery.transform(y_train)
X_test = scalerX.tranform(X_test)
y_test = scalery.transform(y_test)



import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv("train.csv")
x_train=data.iloc[:210,1:]
y_train=data.iloc[:210,:1]
x_test=data.iloc[210:420,1:]
y_test=data.iloc[210:420,:1]
dtc=DecisionTreeClassifier()

dtc.fit(x_train,y_train)

d=x_test.iloc[8]
dv=d.values# otherwise, error.
dv.shape=(28,28)
plt.imshow(255-dv,cmap='gray')
plt.show()
#p=dtc.predict(x_test)
count = 0
"""
for i in range(len(y_test)):
    count+=1 if p[i] == y_test.iloc[i]
"""
#print("ACCURACY: ", (count/len(y_test))*100)



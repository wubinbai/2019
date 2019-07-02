# What is XGBoost
# XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements M.L. algos. under the Gradient Boosting framework. XGBoost provides a parallel tree boosting, (AKA GDBT, GBM) that solve many data science problesm in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples.
#( From XGBoost Documentation)

# 
import pandas as pd
from xgboost import XGBClassifier
xgbc = XGBClassifier()
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

train = pd.read_csv("Affairs.csv")
train["gender"] =enc.fit_transform(train["gender"])
train["children"] =enc.fit_transform(train["children"])
train.age=train.age.astype(int)
train.yearsmarried=train.yearsmarried.astype(int)

target = 'affairs'
y=train[target]
X = train.drop(target,axis=1)
X = X.drop(train.columns[0],axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

xgbc.fit(X_train,y_train)
print('Score on test set: ', xgbc.score(X_test,y_test))





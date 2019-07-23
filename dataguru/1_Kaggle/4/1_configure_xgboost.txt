# -*- coding: utf-8 -*-

from sklearn.model_selection import GridSearchCV
import pandas as pd

df_train = pd.read_csv('./Affairs.csv', index_col=0)
df_train.head()

df_train['affairs'] = (df_train['affairs']>0).astype(float)
df_train['gender'] = (df_train['gender']=='male').astype(float)
df_train['children'] = (df_train['children']=='yes').astype(float)

df_train.head()

from sklearn.model_selection import train_test_split
train_xy,val = train_test_split(df_train, test_size = 0.3,random_state=1)
y = train_xy.affairs
X = train_xy.drop(['affairs'],axis=1)
val_y = val.affairs
val_X = val.drop(['affairs'],axis=1)

import xgboost as xgb
xgb_val = xgb.DMatrix(val_X,label=val_y)
xgb_train = xgb.DMatrix(X, label=y)

#param = {'max_depth':5, 'eta':0.02, 'silent':0, 'objective':'binary:logistic',
#'eval_metric':'logloss', 'lambda':3, 'colsample_bytree':0.9 }
num_round = 100
watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]

param_grid = {'max_depth':[5], 'eta':[0.001,0.003,0.009,0.01,0.03,0.09,0.1,0.3,0.9,1,3,9], 'silent': 0, 'objective':['binary:logistic'], 'eval_metric':['logloss'], 'lambda':[3], 'colsample_bytree':[0.9] }
#grid = GridSearchCV(xgb,param_grid,cv=10,scoring='accuracy',return_train_score=False)
# I don't know how to make the following work

#model = grid.fit(xgb_train,watchlist)
#model = xgb.train(param, xgb_train, num_round, watchlist)
model_ori = xgb.train(param_grid,xgb_train,num_round,watchlist)

# I really don't know how to make the above working, any method to KNOW this?

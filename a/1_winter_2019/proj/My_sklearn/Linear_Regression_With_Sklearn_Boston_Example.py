import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
boston = load_boston()
df_x = pd.DataFrame(boston.data,columns = boston.feature_names)
df_y=pd.DataFrame(boston.target)
df_x.describe()
reg=linear_model.LinearRegression()
X_train,X_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=4)
reg.fit(X_train,y_train)
reg.coef_
y_pred=reg.predict(X_test)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_pred,y_test)
rmse = np.sqrt(mse)
print('log_reg rmse:', rmse)

# Try another model, in page 69 of hands on ML

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train,y_train)
tree_pred = tree_reg.predict(X_test)
tree_rmse = np.sqrt(mean_squared_error(tree_pred,y_test))
print('DTR rmse', tree_rmse)

# Better eval using cross val score

from sklearn.model_selection import cross_val_score as crvs
scores = crvs(tree_reg,df_x,df_y,scoring="neg_mean_squared_error",cv=10)
rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print('======')
    
    print('Scores: ', scores)
    print("Mean: ", scores.mean())
    print("STD: ", scores.std())
    print('==end==')
display_scores(rmse_scores)

# display lin reg scores
lin_scores = crvs(reg,df_x,df_y,scoring="neg_mean_squared_error",cv=10)
lin_scores = np.sqrt(-lin_scores)
display_scores(lin_scores)


# Try one more RFR.

from sklearn.ensemble import RandomForestRegressor as RFR
rfr = RFR()

rfr_scores = crvs(rfr,df_x,np.ravel(df_y),scoring="neg_mean_squared_error",cv=10)
rfr_scores = np.sqrt(-rfr_scores)
display_scores(rfr_scores)

# Use Ridge Regression

from sklearn.linear_model import Ridge as Ri
ri = Ri()
# display Ridge scores
ri_scores = crvs(ri,df_x,df_y,scoring="neg_mean_squared_error",cv=10)
ri_scores = np.sqrt(-ri_scores)
print("Ridge Regression: ")
display_scores(ri_scores)


# Use Lasso Regression

from sklearn.linear_model import Lasso as La
la = La()
# display Ridge scores
la_scores = crvs(la,df_x,df_y,scoring="neg_mean_squared_error",cv=10)
la_scores = np.sqrt(-la_scores)
print("Lasso Regression: ")
display_scores(la_scores)
la.fit(X_train,y_train)
la_pred = la.predict(X_test)











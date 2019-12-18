
import pandas as pd
from rmse import rmse_cv
from sklearn.linear_model import Ridge, Lasso

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
corr = train.corr()
y_train = train.SalePrice
train.drop('SalePrice',axis=1,inplace=True)
y_train_transformed = np.log1p(y_train)
test_id = test.Id

useless = ['Utilities', 'BsmtFinType2','BsmtFinSF2','BsmtUnfSF','HalfBath','YrSold']
DROP_USELESS = True
def drop_useless(df):
    print('before drop_useless: df has shape: ', df.shape)
    for i in useless:
        df.drop(i,axis=1,inplace=True)
    print('after drop_useless: df has shape: ', df.shape)

if DROP_USELESS:
    drop_useless(train)
    drop_useless(test)
LEVEL = 0
if LEVEL == 0:
    train_test = pd.concat([train,test],axis=0)
    train_test_dummies = pd.get_dummies(train_test)
    train_d = train_dummies = train_test_dummies[:train.shape[0]]
    test_d = test_dummies = train_test_dummies[train.shape[0]:]
    train_fill = train_d.fillna(train_d.mean())
    test_fill = test_d.fillna(test_d.mean())
    
#    tr0 = pd.get_dummies(train)
#    tr1 = tr0.fillna(tr0.mean())
#    te0 = pd.get_dummies(test)
#    te1 = te0.fillna(te0.mean())

model_ridge = Ridge(alpha=10)
scores_ridge = rmse_cv(model_ridge,train_fill,y_train_transformed)
print('scores_ridge: ', scores_ridge)
model_ridge.fit(train_fill,y_train_transformed)
ridge_pred = model_ridge.predict(test_fill)

ridge_pred_transformed = np.expm1(ridge_pred)
ridge_dict = {'id':test_id, 'SalePrice':ridge_pred_transformed}
ridge_df = pd.DataFrame(ridge_dict)
ridge_df.to_csv('ridge_sub.csv',index=False)

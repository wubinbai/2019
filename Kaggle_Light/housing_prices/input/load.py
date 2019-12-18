
import pandas as pd
from rmse import rmse_cv
from sklearn.linear_model import Ridge

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
corr = train.corr()
y_train = train.SalePrice
y_train_transformed = np.log1p(y_train)
useless = ['Utilities', 'BsmtFinType2','BsmtFinSF2','BsmtUnfSF','HalfBath','YrSold']
DROP_USELESS = True
def drop_useless(df):
    print('before drop_useless: df has shape: ', df.shape)
    for i in useless:
        df.drop(i,axis=1,inplace=True)
    print('after drop_useless: df has shape: ', df.shape)

if DROP_USELESS:
    drop_useless(train)

LEVEL = 0
if LEVEL == 0:
    tr0 = pd.get_dummies(train)
    tr1 = tr0.fillna(tr0.mean())

model_ridge = Ridge()
scores_ridge = rmse_cv(model_ridge,tr1,y_train_transformed)
print('scores_ridge: ', scores_ridge)


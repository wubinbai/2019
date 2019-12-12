

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')



na_means_no_or_none_tuple = ('Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature')
repeat_tuple = (('BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'),('GarageType','GarageFinish','GarageQual','GarageCond'))

def f1(train):
    for i in na_means_no_or_none_tuple:
        train.loc[:,i]= train.loc[:,i].fillna('No')
    return train

print('After these feature have been fillna, still we have LogFrontage 259 MaxVnrType 8 MaxVnrArea 8 Electrical 1 GarageYrBlt 81 = 357')
print('use train.isna().sum().sum() to confirm')
print('call train = d2.d2_fun2(train) to clear')

def f2(train):
    train.loc[:, "LotFrontage"] = train.loc[:, "LotFrontage"].fillna(0)
    train.loc[:, "MasVnrType"] = train.loc[:, "MasVnrType"].fillna("None")
    train.loc[:, "MasVnrArea"] = train.loc[:, "MasVnrArea"].fillna(0)
    train.loc[:, "GarageYrBlt"] = train.loc[:, "GarageYrBlt"].fillna(0)
    train.loc[:, "Electrical"] = train.loc[:, "Electrical"].fillna('SBrkr')
    return train

g1 = f1

def g2():
    '''
    test after g1 we have missing: using
    hey = test.isna().sum().loc[test.isna().sum().values!=0]     
    In [44]: hey
Out[44]:
MSZoning        4
Utilities       2
Exterior1st     1
Exterior2nd     1
BsmtFinSF1      1
BsmtFinSF2      1
BsmtUnfSF       1
TotalBsmtSF     1
BsmtFullBath    2
BsmtHalfBath    2
KitchenQual     1
Functional      2
GarageCars      1
GarageArea      1
SaleType        1
dtype: int64

    '''


train = f1(train)
train = f2(train)
test = g1(test)
test = g2(test)

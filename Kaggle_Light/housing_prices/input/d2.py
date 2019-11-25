na_means_no_or_none_tuple = ('Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature')
repeat_tuple = (('BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'),('GarageType','GarageFinish','GarageQual','GaregeCond'))

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

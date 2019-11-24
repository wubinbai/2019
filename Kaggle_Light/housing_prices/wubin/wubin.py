
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


train = pd.read_csv('../input/train.csv')
train_b = train.copy()
test = pd.read_csv('../input/test.csv')
test_b = test.copy()

def f1(df,test):
    '''
    drop missing, from comprehensive E. E. w/t Py
    '''
    
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    nothing = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
    missing = nothing[nothing['Total']>1]
    res = df.drop(missing.index,axis = 1)
    res.drop(res.loc[res['Electrical'].isnull()].index,inplace=True)
    test = test.drop(missing.index,axis = 1)
        
    # for test missing: extra missing:
    #BsmtFullBath    2
    #BsmtHalfBath    2
    #Functional      2
    #MSZoning        4
    #Utilities       2
    test.loc[:,'BsmtFullBath'] = test.loc[:,'BsmtFullBath'].fillna(0)
    test.loc[:,'BsmtHalfBath'] = test.loc[:,'BsmtHalfBath'].fillna(0)
    test.loc[:,'Functional'] = test.loc[:,'Functional'].fillna('Typ')
    test.loc[:,'MSZoning'] = test.loc[:,'MSZoning'].fillna('RL')
    test.loc[:,'Utilities'] = test.loc[:,'Utilities'].fillna('AllPub')
    # more
    index = ['BsmtFinSF2',
 'BsmtFinSF1',
 'Exterior2nd',
 'BsmtUnfSF',
 'TotalBsmtSF',
 'SaleType',
 'Exterior1st',
 'KitchenQual',
 'GarageArea',
 'GarageCars']
    for i in index:
        val = test.loc[:,i].mode().values[0]
        test.loc[:,i] = test.loc[:,i].fillna(val)
 
    return res, test
                

def f2(df1,df2):
    '''
    new feat: has basement
    '''
    df1['HasBsmt'] = 0
    df1.loc[df1['TotalBsmtSF']>0,'HasBsmt'] = 1
    df2['HasBsmt'] = 0
    df2.loc[df2['TotalBsmtSF']>0,'HasBsmt'] = 1 
    return df1,df2

def f3(df):
    '''
    log1p Saleprice
    '''

    df['SalePrice'] = np.log1p(df.SalePrice)
    return df
def f4(train,test):
    '''
    log1p skewed_feats
    '''
    numeric_feats = train.dtypes[train.dtypes!='object'].index
    skewed_feats = train[numeric_feats].apply(lambda x: scipy.stats.skew(x.dropna()))
    transf_feats = skewed_feats[skewed_feats > 0.75]
    transf_index = transf_feats.index
    train[transf_index] = np.log1p(train[transf_index])
    test[transf_index] = np.log1p(test[transf_index])
    return train,test

def info_4_1(train):
    '''
    get info corr
    '''
    corr = train.corr()
    corr.sort_values(['SalePrice'], ascending = False, inplace = True)
    print(corr.SalePrice[:10])


def f5(train):
    '''
    Create new features
    '''
    # 3* Polynomials on the top 5(comment to increase to 7 since some will create error when predicting test because of nans) existing features
    train["OverallQual-s2"] = train["OverallQual"] ** 2
    train["OverallQual-s3"] = train["OverallQual"] ** 3
    train["OverallQual-Sq"] = np.sqrt(train["OverallQual"])
    train["GrLivArea-2"] = train["GrLivArea"] ** 2
    train["GrLivArea-3"] = train["GrLivArea"] ** 3
    train["GrLivArea-Sq"] = np.sqrt(train["GrLivArea"])
#    train["GarageCars-2"] = train["GarageCars"] ** 2
#    train["GarageCars-3"] = train["GarageCars"] ** 3
#    train["GarageCars-Sq"] = np.sqrt(train["GarageCars"])
    train["1stFlrSF-s2"] = train["1stFlrSF"] ** 2
    train["1stFlrSF-s3"] = train["1stFlrSF"] ** 3
    train["1stFlrSF-Sq"] = np.sqrt(train["1stFlrSF"])
    train["FullBath-2"] = train["FullBath"] ** 2
    train["FullBath-3"] = train["FullBath"] ** 3
    train["FullBath-Sq"] = np.sqrt(train["FullBath"])
    train["YearBuilt-2"] = train["YearBuilt"] ** 2
    train["YearBuilt-3"] = train["YearBuilt"] ** 3
    train["YearBuilt-Sq"] = np.sqrt(train["YearBuilt"])
#    train["GarageArea-2"] = train["GarageArea"] ** 2
#    train["GarageArea-3"] = train["GarageArea"] ** 3
#    train["GarageArea-Sq"] = np.sqrt(train["GarageArea"])
    return train

def f6(train,test):
    '''
    StdScl train and test
    '''
    num_feats = train.select_dtypes(exclude='object').columns
    num_feats = num_feats.drop('SalePrice')
    ss = StandardScaler()
    ss.fit_transform(train[num_feats])
    ss.transform(test[num_feats])
    return train,test

def p(train,test):
    train = pd.get_dummies(train)
    test = pd.get_dummies(test)

    y = train.SalePrice
    X = train.drop('SalePrice',axis=1)
    lr = LinearRegression()
    lr.fit(X,y)
    pred = lr.predict(test)
    return pred

def t0(pred):
    '''
    expm1 back for pred
    '''
    res = np.expm1(pred)
    return res


#drop missing
train,test = f1(train,test)
#new feat: HasBasmt
train,test = f2(train,test)
#log1p train.SalePrice
train = f3(train)
#log1p skewed_feats
train,test = f4(train,test)
#info4_1 get corr top 10
info_4_1(train)
# new feats for train and test
train = f5(train)
test = f5(test)
# StdScl train and test
train,test = f6(train,test)

# fit and predict with lr
pred = p(train,test)
# expm1 back for pred
pred = t0(pred)


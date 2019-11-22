import scipy


train = pd.read_csv('../input/train.csv')
train_b = train.copy()
test = pd.read_csv('../input/test.csv')
test_b = test.copy()

def f1(df,df2):
    '''
    drop missing, from comprehensive E. E. w/t Py
    '''
    
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    nothing = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
    missing = nothing[nothing['Total']>1]
    res = df.drop(missing.index,axis = 1)
    res.drop(res.loc[res['Electrical'].isnull()].index,inplace=True)
    res2 = df2.drop(missing.index,axis = 1)
    return res,res2
                

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

def t0(pred):
    '''
    expm1 back for pred
    '''
    res = np.expm1(pred)
    return res

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


#drop missing
train,test = f1(train,test)
#new feat: HasBasmt
train,test = f2(train,test)
#log1p train.SalePrice
train = f3(train)
#log1p skewed_feats
train,test = f4(train,test)

# pred = t0(pred)


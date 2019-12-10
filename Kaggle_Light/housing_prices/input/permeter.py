import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train['permeter'] = train.SalePrice/train.GrLivArea
train['MainVal'] = train.SalePrice - train.MiscVal
corr = train.corr()

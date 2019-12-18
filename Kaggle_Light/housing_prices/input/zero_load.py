train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
corr = train.corr()
y_train = train.SalePrice
train.drop('SalePrice',axis=1,inplace=True)
y_train_transformed = np.log1p(y_train)
test_id = test.Id


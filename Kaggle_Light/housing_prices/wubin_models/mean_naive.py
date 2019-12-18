train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

test['SalePrice'] = train.SalePrice.mean()
df = test[['Id','SalePrice']]
df = df.set_index('Id',drop=True)
df.to_csv('naive_sub_mean.csv')

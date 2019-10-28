train = pd.read_csv('../input/train.csv')
test = pd.read_csv("../input/test.csv")


print(train.shape)
#print(train.columns)
print(test.shape)
#print(test.columns)

def find_dtypes(train):
    d = {}
    for i in train.columns:
        if not train[i].dtype in d:
            d[train[i].dtype] = 1
        else:
            d[train[i].dtype] += 1
    print(".columns have ", len(d), 'dtypes')
    print('Namely: ', d)
find_dtypes(train)
find_dtypes(test)

def create_new_df(train):
    new_train = pd.DataFrame()
    for i in train.columns:
        if train[i].dtype == np.dtype('int64') or train[i].dtype == np.dtype('float64'):
            new_train = pd.concat([new_train,train[i]],axis=1)
    return new_train

new_train = create_new_df(train)
new_test = create_new_df(test)

# using new_train.info() we found that:
missing_col = ['LotFrontage','MasVnrArea','GarageYrBlt']

# Fill With Missing Value, with mean/median, FOR LINEAR MODEL, like NN
for i in range(len(missing_col)):
    if i == 2:
        temp = new_train[missing_col[i]].median()
        new_train[missing_col[i]].loc[new_train[missing_col[i]].isnull()] = temp
    else:
        temp = new_train[missing_col[i]].mean()
        new_train[missing_col[i]].loc[new_train[missing_col[i]].isnull()] = temp


y_train = new_train.SalePrice
x_train = new_train.drop(['SalePrice','Id'],axis=1)

y_test = None
x_test = new_test.copy()
x_test_id = x_test.Id
x_test = x_test.drop(['Id'],axis=1)


mean = x_train.mean(axis=0)
x_train -= mean
x_test -= mean

std = x_train.std(axis=0)
x_train /= std
x_test /= std

# let's also try to scale the label, BUT ALSO KEEP IN MIND, AFTER PREDICTION, SCALE IT BACK!
# but just scale it, say 1e-5 factor
y_train *= 1e-4

from sklearn.model_selection import train_test_split
train_data, val_data, train_label, val_label = train_test_split(x_train,y_train)





# Model Def
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_data,train_label)
pred = lr.predict(x_test)
# scale pred back, because our input labels had been scaled.
pred *= 1e4

df2 = pd.concat([new_test.Id,pred],axis=1)
df = df2.set_index('Id',drop=True)
df.columns = ['SalePrice']
temp_mean = df.mean(numeric_only=True)
df[df.isnull()] = temp_mean.values[0]
df.to_csv('sub.csv')

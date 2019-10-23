train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

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

y_train = new_train.SalePrice
x_train = new_train.drop('SalePrice',axis=1)
y_test = None
x_test = new_test


mean = x_train.mean(axis=0)
x_train -= mean
x_test -= mean

std = x_train.std(axis=0)
x_train /= std
x_test /= std

import keras
from keras.models import Sequential
from keras.layers import Dense
EPOCHS = 100


m = Sequential()
m.add(Dense(32,activation='relu',input_shape = x_train.loc[0].shape))
m.add(Dense(32,activation='relu'))
m.add(Dense(1))

m.summary()

m.compile(optimizer=keras.optimizers.RMSprop(lr=0.00000000000000001),loss='mse',metrics=['mae'])


history = m.fit(x_train,y_train,epochs=EPOCHS,batch_size=512)

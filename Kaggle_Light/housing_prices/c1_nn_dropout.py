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


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
EPOCHS = 1000


m = Sequential()
m.add(Dense(64,activation='relu',input_shape = x_train.loc[0].shape))
m.add(Dropout(0.2))
m.add(Dense(64,activation='relu'))

m.add(Dropout(0.35))
m.add(Dense(1))

m.summary()

m.compile(optimizer=keras.optimizers.RMSprop(),loss='mse',metrics=['mae'])


history = m.fit(train_data,train_label,epochs=EPOCHS,batch_size=128, validation_data=(val_data,val_label))

pred = pd.DataFrame(m.predict(x_test))
# scale pred back, because our input labels had been scaled.
pred *= 1e4
df2 = pd.concat([new_test.Id,pred],axis=1)
df = df2.set_index('Id',drop=True)
df.columns = ['SalePrice']     
temp_mean = df.mean(numeric_only=True) 
df[df.isnull()] = temp_mean.values[0]
df.to_csv('sub.csv')




mae = history.history['mean_absolute_error']
val_mae = history.history['val_mean_absolute_error']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(mae) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, mae, 'bo', label='Training mae')
plt.plot(epochs, val_mae, 'b', label='Validation mae')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

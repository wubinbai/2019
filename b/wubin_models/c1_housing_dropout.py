from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
EPOCHS = 100

(x_train, y_train),(x_test, y_test) = boston_housing.load_data()
x_train_backup = x_train.copy()
x_test_backup = x_test.copy()

mean = x_train.mean(axis=0)
x_train -= mean
x_test -= mean

std = x_train.std(axis=0)
x_train /= std
x_test /= std

m = Sequential()
m.add(Dense(64,activation='relu',input_shape = x_train[0,:].shape))
m.add(Dropout(0.5))
m.add(Dense(32,activation='relu'))

m.add(Dense(1))

m.summary()

m.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])


history = m.fit(x_train,y_train,epochs=EPOCHS,validation_data=(x_test,y_test))


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


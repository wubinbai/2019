import time
import keras
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def get_features_labels(df):
    features = df.values[:,1:]/255
    labels = df['label'].values
    return features, labels

train_features, train_labels = get_features_labels(train)
test_features = test
test_copy = test.copy()
train_labels = keras.utils.to_categorical(train_labels)

x_train = train_features
y_train = train_labels
x_test = np.array(test_features)

img_rows = img_cols = 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# Let's train_test_split before creating our neural network!
# UNDERSCORE means: interal train or test.
# Since actually there's option found in model.fit validation_split -> [0,1] parameter, following may not be necc.
from sklearn.model_selection import train_test_split
_x_train,_x_test,_y_train,_y_test = train_test_split(x_train,y_train,test_size=0.5)

num_classes = 10
batch_size = 128
epochs = 20 

model = keras.Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
# Add my one MaxPooling below
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#Adding one more conv
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
history = model.fit(_x_train, _y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(_x_test,_y_test)
          )
#          validation_split=0.2)

acc = history.history['acc'] 
val_acc = history.history['val_acc'] 
loss = history.history['loss'] 
val_loss = history.history['val_loss'] 

epochs = range(1, len(acc) + 1)

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
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()



result_pred = model.predict_classes(x_test)
df = pd.DataFrame(result_pred)
df.index = range(1,28001)

df.to_csv("my_cnn.csv")



'''model.add(keras.layers.Dense(400,activation='relu',input_shape=(784,)))

model.add(keras.layers.Dense(320,activation='relu'))
model.add(keras.layers.Dense(240,activation='relu'))
model.add(keras.layers.Dense(160,activation='relu'))
model.add(keras.layers.Dense(80,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])
model.summary()
EPOCHS = 10
BATCH_SIZE = 64


tic = time.time()
'''
'''
models = [model for i in range(10)]
x = 40
times = []
for modeli in models:
    t1 = time.time()
    modeli.fit(train_features,train_labels,epochs=EPOCHS, batch_size = x)
    t2 = time.time()
    times.append(t2-t1)
    x+=40
'''

'''
model.fit(train_features,train_labels, epochs = EPOCHS, batch_size = BATCH_SIZE)

toc = time.time()

diff = toc - tic
print("time running keras.Sequential().fit: ", diff, 's')
#test_loss,test_acc = model.evaluate(test_features,test_labels)

pred_test = model.predict(test_features)
result_pred = pred_test.argmax(axis=1)
#result_ground = test_labels.argmax(axis=1)
#pred_acc = (result_pred == result_ground).sum()/result_ground.shape[0]
df = pd.DataFrame(result_pred)
#df.shape
df.index = range(1,28001)
#df.columns = ['ImageId','Label']
df.to_csv("sub.csv")

'''

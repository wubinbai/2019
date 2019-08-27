import time

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def get_features_labels(df):
    features = df.values[:,1:]/255
    labels = df['label'].values
    return features, labels

train_features, train_labels = get_features_labels(train)
test_features = test
'''repeat = 0
while repeat < 10:
    repeat += 1
    example_index = np.random.randint(300)
    plt.ion()
    plt.figure()
    _ = plt.imshow(np.reshape(train_features[example_index,:],(28,28)),'gray')'''

import keras
train_labels = keras.utils.to_categorical(train_labels)
#test_labels = keras.utils.to_categorical(test_labels)
model = keras.Sequential()
model.add(keras.layers.Dense(30,activation='relu',input_shape=(784,)))
model.add(keras.layers.Dense(20,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])
model.summary()
EPOCHS = 10
BATCH_SIZE = 64

tic = time.time()

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
df.shape
df.index = range(1,28001)
df.to_csv("sub.csv")

train = pd.read_csv("fashion-mnist_train.csv")
test = pd.read_csv("fashion-mnist_test.csv")

def get_features_labels(df):
    features = df.values[:,1:]/255
    labels = df['label'].values
    return features, labels

train_features, train_labels = get_features_labels(train)
test_features, test_labels = get_features_labels(test)
'''repeat = 0
while repeat < 10:
    repeat += 1
    example_index = np.random.randint(300)
    plt.ion()
    plt.figure()
    _ = plt.imshow(np.reshape(train_features[example_index,:],(28,28)),'gray')'''

import tensorflow as tf
from tensorflow import keras
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(30,activation=tf.nn.relu,input_shape=(784,)))
model.add(tf.keras.layers.Dense(20,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])
model.summary()
EPOCHS = 2
BATCH_SIZE = 128
model.fit(train_features,train_labels, epochs = EPOCHS, batch_size = BATCH_SIZE)
test_loss,test_acc = model.evaluate(test_features,test_labels)


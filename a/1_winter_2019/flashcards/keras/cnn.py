from keras.layers import Activation, Conv2D, MaxPooling2D, Flatten
from seq import *
from keras.layers import Dropout
from data import *

model2.add(Conv2D(32,(3,3),padding='same',input_shape=X_train.shape[1:]))
model2.add(Activation('relu'))
model2.add(Conv2D(32,(3,3)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.25))

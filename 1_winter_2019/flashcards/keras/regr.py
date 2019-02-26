from seq import *
from keras.layers import Dense
from data import *
train_data = X_train
model.add(Dense(64,activation='relu',input_dim=train_data.shape[1]))
model.add(Dense(1))

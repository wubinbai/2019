from seq import *
from keras.layers import Dense
from keras.layers import Embedding, LSTM

model3.add(Embedding(20000,128))
model3.add(LSTM(128,dropout=0.2,recurrent_dropout=0.21))
model3.add(Dense(1,activation='sigmoid'))

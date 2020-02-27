from keras import Input
from keras.layers import Dense
from keras import Model
from keras import  Sequential

input_tensor = Input(shape=(32,))
dense0 = Dense(32,activation='relu')(input_tensor)
dense1 = Dense(32,activation='relu')(dense0)
output_tensor = Dense(10,activation='softmax')(dense1)

model = Model(input_tensor, output_tensor)
print(model.summary())

model_s = Sequential()
model_s.add(Dense(32,activation='relu',input_shape=(32,)))
model_s.add(Dense(32,activation='relu'))
model_s.add(Dense(10,activation='softmax'))
print(model_s.summary())

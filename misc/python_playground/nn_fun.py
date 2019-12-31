from keras.layers import *
from keras import Sequential
from keras.optimizers import *

x = list(range(100))
y = [i+5 for i in x]

model = Sequential()
model.add(Dense(1,activation='relu',input_shape=(1,),use_bias=True))

model.compile(optimizer=rmsprop(lr=0.3),metrics=['mae'],loss='mse')
history = model.fit(x,y,epochs=100)
plt.plot(history.history['loss'])
'''
z = [i**2 for i in x]

m = Sequential()
m.add(Dense(1,activation='relu',input_shape=(1,)))
m.compile(optimizer=rmsprop(lr=14),metrics=['mae'],loss='mse')
history = m.fit(x,z,epochs=100)
plt.plot(history.history['loss'])
'''

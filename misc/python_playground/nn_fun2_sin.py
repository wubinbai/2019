from keras.layers import *
from keras import Sequential
from keras.optimizers import *

x0 = list(range(100))
q = [x0,x0,x0]
x = np.array(q).transpose()
y = [np.sin(i) for i in x0]

model = Sequential()
model.add(Dense(5,activation='relu',input_shape=(3,)))
model.add(Dense(1,activation='relu')) 

model.compile(optimizer=rmsprop(lr=0.01),metrics=['mae'],loss='mse')
history = model.fit(x,y,epochs=100)
plt.plot(history.history['loss'])

def predict(a):
    l = [a,a,a]
    prep = np.array(l).transpose()
    return model.predict(prep)

'''
z = [i**2 for i in x]

m = Sequential()
m.add(Dense(1,activation='relu',input_shape=(1,)))
m.compile(optimizer=rmsprop(lr=14),metrics=['mae'],loss='mse')
history = m.fit(x,z,epochs=100)
plt.plot(history.history['loss'])
'''

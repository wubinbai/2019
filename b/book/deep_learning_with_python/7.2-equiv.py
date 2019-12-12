from keras import Input,Model,layers

input_tensor = Input(shape=(64,))
x0 = layers.Dense(32, activation='relu')(input_tensor)
x1 = layers.Dense(32, activation='relu')(x0)
output_tensor = layers.Dense(10, activation='softmax')(x1)

model = Model(input_tensor,output_tensor)
print(model.summary())

model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
x = np.random.random((1000,64))
y = np.random.random((1000,10))

model.fit(x,y,epochs=10,batch_size=128)
score = model.evaluate(x,y)

from keras import layers, Input, Model, LSTM, concatenate, Dense

lstm = LSTM(32)

l_input = Input(shape=(None, 128))
l_output = lstm(l_input)

r_input = Input(shape=(None, 128))
r_output = lstm(r_input)

merged = concatenate([l_output, r_output], axis=-1)
predictions = Dense(1, activation='sigmoid')(merged)
model = Model([l_i,r_i], predictions)
model.fit([l_data,r_data], targets)

from keras import Input, layers, Model

text_vocab_size = 10000
ques_vocab_size = 10000
ans_vocab_size = 500

text_input = Input(shape=(None,),dtype='int32',name='text')
embedded_text = layers.Embedding(64,text_vocab_size)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape=(None,), dtype='int32',name = 'question')
embedded_question = layers.Embedding(32, ques_vocab_size)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([encoded_text, encoded_question], axis = -1)
answer = layers.Dense(ans_vocab_size, activation = 'softmax')(concatenated)
model = Model([text_input, question_input],answer)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['acc'])

'''
num_samples = 100
max_length = 32
text = np.random.randint(1,text_vocab_size,size=(num_samples, max_length))
question = np.random.randint(1,ques_vocab_size, size = (num_samples, max_length))
answers = np.random.randint(0,1,size=(num_samples, ans_vocab_size))

model.fit([text,question],answers,epochs=10,batch_size=128)

'''

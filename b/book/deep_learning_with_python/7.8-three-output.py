from keras import layers, Input, Model
vocab_size = 50000
num_income_groups = 10
posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocab_size)(posts_input)
x0 = layers.Conv1D(128,5,activation='relu')(embedded_posts)
x1 = layers.MaxPooling1D(5)(x0)
x2 = layers.Conv1D(256,5,activation='relu')(x1)
x3 = layers.Conv1D(256,5,activation='relu')(x2)
x4 = layers.MaxPooling1D(5)(x3)

x5 = layers.Conv1D(256,5,activation='relu')(x4)
x6 = layers.Conv1D(256,5,activation='relu')(x5)
x7 = layers.GlobalMaxPooling1D()(x6)

x8 = layers.Dense(128, activation='relu')(x7)

age_prediction = layers.Dense(1,name='age')(x8)
income_prediction = layers.Dense(num_income_groups,activation='softmax',name='income')(x8)
gender_prediction = layers.Dense(1,activation='sigmoid')(x8)
model = Model(input_posts,[gender_prediction,income_prediction,gender_prediction])



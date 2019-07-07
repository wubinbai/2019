'''from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

'''

import pandas as pd
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

train = pd.read_csv("Affairs.csv")
train["gender"] =enc.fit_transform(train["gender"])
train["children"] =enc.fit_transform(train["children"])
train.age=train.age.astype(int)
train.yearsmarried=train.yearsmarried.astype(int)

target = 'affairs'
y=train[target]
X = train.drop(target,axis=1)
X = X.drop(train.columns[0],axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)
INPUT_SHAPE = len(X_train.columns)
import keras
from keras.layers import Dense
model = keras.Sequential()
model.add(Dense(4,activation='relu',input_shape=(INPUT_SHAPE,)))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='tanh'))
model.compile(loss='mean_absolute_error',optimizer='rmsprop',metrics=['accuracy'])
model.summary()
EPOCHS = 300
BATCH_SIZE = 900
model.fit(X_train,y_train,epochs = EPOCHS, batch_size = BATCH_SIZE)
test_loss,test_acc=model.evaluate(X_test,y_test)



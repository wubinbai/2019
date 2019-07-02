from keras.models import Sequential
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

#xgbc.fit(X_train,y_train)
#print('Score on test set: ', xgbc.score(X_test,y_test))





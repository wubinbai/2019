train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

print(train.shape)
print(train.columns)
print(test.shape)
print(test.columns)

y_train = train.target
x_train = train.drop('target',axis=1)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
rffi = rf.feature_importances_
print(rffi)

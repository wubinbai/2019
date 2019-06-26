import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.info()
test.info()
#print("train.info(): ", train.info())
#print("test.info(): ", test.info())
print("train columns: ", train.columns)
print("test columns: ", test.columns)


from tqdm import tqdm


train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
# Whether or not to use the fllowing to drop_duplicates of rows like around within 100 rows?
#new_train = train.T.drop_duplicates()
#new = new_train.T
Y = train.target
test_ID = test.ID
traintest = pd.concat([train,test],axis=0)

feats_counts = train.nunique(dropna=False)
temp = feats_counts.loc[feats_counts==1].index
constant_features = temp.tolist()
traintest.drop(constant_features,axis=1,inplace=True)

traintest.fillna('NaN',inplace=True)
train_enc = pd.DataFrame(index=train.index)
for col in tqdm_notebook(traintest.columns):
    train_enc[col] = train[col].factorize()[0]
# or
# train_enc[col] = train[col].map(train[col.value_counts()])

dup_cols = {}

for i, c1 in tqdm(train_enc.columns):
    for c2 in train_enc.columns[i+1:]:
        if c2 not in dup_cols and np.all(train_enc[c1]==train_enc[c2]):
            dup_cols[c2] = c1



trnu = train.nunique()


trnu1=(trnu==1)

trnu1.sum()


plot_whole(train.head())


tris=train.isnull()



triss=tris.sum(axis=1)



triss.shape



triss2=tris.sum(axis=0)


triss2.head(20)


traintest = pd.concat([train,test],axis=0)


feats_counts=train.nunique(dropna=False)


feats_counts.sort_values()[:30]



feats_counts.sort_values()[:5]


new_train = train.T.drop_duplicates()

new_train.shape



train.shape



import os
root_name = os.listdir('../input')[0]

path = os.path.join('../input',root_name)
files = os.listdir(path)
train_curated = pd.read_csv(os.path.join(path,'train_curated.csv'))

test_files = os.listdir(os.path.join(path,'test'))
print(len(test_files))

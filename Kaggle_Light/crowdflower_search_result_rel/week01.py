# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:15:50 2017

@author: tracy
"""

import pandas as pd
import numpy as np
import nltk



train=pd.read_csv("./train.csv")
test=pd.read_csv("./test.csv")

train.head()
test.head()

train.columns
test.columns

len(train)
len(test)

train['query'].unique()[0:10]

len(train['query'].unique())

len(test['query'].unique())

train['product_title'].unique()[0:10]

len(train['product_title'].unique())

len(test['product_title'].unique())

len(np.setdiff1d(test['product_title'].unique(),train['product_title'].unique()))

len(np.intersect1d(test['product_title'].unique(),train['product_title'].unique()))

query = train['query'].map(nltk.tokenize.word_tokenize)

from nltk.corpus import stopwords
stopset = set(stopwords.words('english'))

def key_plot(data,col,top_num=10):  
    s= data[col].map(nltk.tokenize.word_tokenize)
    fdist=nltk.FreqDist( words.lower()  for x in s
                    for words in x if words.lower() not in stopset )
    top=pd.DataFrame(fdist.most_common(top_num),columns=['query','times'])
    top=top.set_index('query')
    top.plot(kind='bar')


key_plot(train,'query')
key_plot(test,'query')



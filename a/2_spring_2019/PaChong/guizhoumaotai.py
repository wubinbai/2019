#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import csv
# from pandas import DataFrame
import pandas

mlist=[]
with open(r'D:\python操作文件\贵州茅台历史数据.txt','r',encoding='gbk') as f:
    soup=BeautifulSoup(f.read())
    tar=soup.body.find('table', class_="data", style="width:100%; border-collapse: collapse;")
#     print(tar.tr.td.string)
    for i in tar.findAll('tr')[1:-1]:
        m=[]
        for j in i.findAll('td'):
            m.append(j.string)
        mlist.append(m)
# with open(r'D:\python操作文件\贵州茅台历史数据.csv','w',encoding='gbk',newline='') as f1:
#     g=csv.writer(f1)
#     for row in mlist:
#         g.writerow(row)

df=pandas.DataFrame(mlist,columns=['日期','开盘','最高','最低','收盘','成交量','成交金额','升跌$','升跌%','缩','高低差%','SH上证','SH%'])
df['日期']=pandas.to_datetime(df['日期'])
df=df.set_index('日期')
df
# df.to_csv(r'D:\python操作文件\贵州茅台历史数据2.csv',encoding='gbk')


# In[38]:


import pandas 
# from pandas import DataFrame
with open(r'D:\python操作文件\贵州茅台历史数据2.csv','r',encoding='gbk') as f:
    file=pandas.read_csv(f)
# file
# file['日期']=pandas.to_datetime(file['日期'])
file=file.set_index('日期')   #如果将日期转为标准时间,则最终会导致无法删除

f=file.drop('2018-11-09')
f


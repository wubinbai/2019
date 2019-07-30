#!/usr/bin/env python
# coding: utf-8

# In[11]:


from bs4 import BeautifulSoup           #保存路径\获取数据封装
import pandas as pd
from pandas import DataFrame
import requests

def get_data(text):
    scode=BeautifulSoup(text).body.find('table', class_="text-center queryRecruitTable", width="778").findAll('tr')
    M=[]
    for i in scode[1:]:
        m=[j.a.text for j in i.findAll('td')[1:8]]
        M.append(m)
    return M
def save_data(data,path):
    for c in data.columns:
        data[c]=data[c].str.replace(' ','').str.strip()
    data.to_csv(path,encoding='gbk')
    
A=[]
for i in range(1,5):
    url='https://www.xmrc.com.cn/net/info/Resultg.aspx?a=a&g=g&recordtype=1&searchtype=3&keyword=%E5%A4%A7%E6%95%B0%E6%8D%AE&releasetime=365&worklengthflag=0&sortby=updatetime&ascdesc=Desc&PageIndex={0}'.format(str(i))
    head={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:63.0) Gecko/20100101 Firefox/63.0'}
    html=requests.Session().get(url,headers=head)
    html.encoding='utf-8'
    A.extend(get_data(html.text))

data=DataFrame(A,columns=['职位名称','公司名称','工作地点','参考月薪','学历要求','性别要求','发布时间'])
path=r'D:\python操作文件\人才网爬取的数据3.csv'
save_data(data,path)


    


# In[4]:


from bs4 import BeautifulSoup
import pandas as pd
from pandas import DataFrame
import requests
A=[]
for i in range(1,5):
    url='https://www.xmrc.com.cn/net/info/Resultg.aspx?a=a&g=g&recordtype=1&searchtype=3&keyword=%E5%A4%A7%E6%95%B0%E6%8D%AE&releasetime=365&worklengthflag=0&sortby=updatetime&ascdesc=Desc&PageIndex={0}'.format(str(i))
    head={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:63.0) Gecko/20100101 Firefox/63.0'}
    html=requests.Session().get(url,headers=head)
    html.encoding='utf-8'
    scode=BeautifulSoup(html.text).body.find('table', class_="text-center queryRecruitTable", width="778").findAll('tr')
    for i in scode[1:]:
        m=[j.a.text for j in i.findAll('td')[1:8]]
        A.append(m)

def save_data(data,path):
    for c in data.columns:
        data[c]=data[c].str.replace(' ','').str.strip()
    data.to_csv(path,encoding='gbk')
    
data=DataFrame(A,columns=['职位名称','公司名称','工作地点','参考月薪','学历要求','性别要求','发布时间'])
# for c in data.columns:
#     data[c]=data[c].str.replace(' ','').str.strip()
# data.to_csv(r'D:\python操作文件\人才网爬取的数据3.csv',encoding='gbk')

path=r'D:\python操作文件\人才网爬取的数据3.csv'
save_data(data,path)


# In[13]:


import pandas 
from pandas import DataFrame
from bs4 import BeautifulSoup
with open(r'D:\python操作文件\厦门人才网大数据源码.txt','r',encoding='gbk') as file:
    f=file.read()
    A=[]
    soup=BeautifulSoup(f).body.find('table', class_="text-center queryRecruitTable", width="778")
#     for i in soup.find('tr').findAll('th')[1:-1]:
#         b.append(i.string)                  #爬出来的列索引存在太多符号,所以采用以下columns方法
    m=soup.findAll('tr', class_="bg")
    for j in m:
        c=[]
        for h in j.findAll('td')[1:-1]:
            c.append(h.a.text)             #直接用h.text可以索引出时间栏的隐藏文本
        A.append(c)
#     print(A)
df=DataFrame(A,columns=['职位名称','公司名称','工作地点','参考月薪','学历要求','性别要求','发布时间'])
for i in df.columns:
#     if i=='发布时间':
#         df[i]=df[i].str.replace(' ','').str.replace(r'\n','').str.strip()    #如果发布时间一栏用text爬出数据,会爬出很多文本
#     else:                                                                    #此代码专门清洗'发布时间'这列数据
    df[i]=df[i].str.replace(' ','').str.strip().str.strip(r'\n')
df.to_csv(r'D:\python操作文件\人才网爬取的数据2.csv',encoding='gbk')


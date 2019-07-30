#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests                              #保存为字典的
from bs4 import BeautifulSoup
import pandas as pd
from pandas import DataFrame

info1={'rank':[],'movie_title':[],'director':[],'year':[],'country':[],'type':[],'score':[],'comment_num':[],'comm_one':[]}
# info2={'director':[],'lead_role':[],'language':[],'flength':[],}

def get_info1(text,info1):          #获取文本中所需的数据并保存在列表info1中
    soup=BeautifulSoup(text).find('ol', class_="grid_view").findAll('li')      
    for j in soup:  
        info1['rank'].append(j.find('em', class_="").string)
        info1['movie_title'].append(j.find('span', class_="title").string)
        meg=j.find('p', class_="").text.strip().split('\n')
        info1['director'].append(meg[0].strip().split('\xa0\xa0')[0].strip('导演: '))    #导演名字太长导致有些切割三个\ax0时出现错误
        info1['year'].append(meg[1].strip().split('\xa0/\xa0')[0])                       
        info1['country'].append(meg[1].strip().split('\xa0/\xa0')[1])
        info1['type'].append(meg[1].strip().split('\xa0/\xa0')[-1])
        info1['score'].append(j.find('div', class_="star").findAll('span')[1].string)
        info1['comment_num'].append(j.find('div', class_="star").findAll('span')[3].string.strip('人评价'))
        if len(j.find('div', class_="bd").findAll('p'))==2:                 #第7页某电影不存在一句影评模块,因此会导致读取不到信息
            info1['comm_one'].append(j.find('span', class_="inq").string)
        else:
            info1['comm_one'].append('无')
    return DataFrame(info1)

for i in range(0,230,25):
    head={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:63.0) Gecko/20100101 Firefox/63.0'}
    url=r'https://movie.douban.com/top250?start={0}&filter='.format(str(i))
    html=requests.get(url,headers=head)
    html.encoding='utf-8'
    Info1=get_info1(html.text,info1)
        
#         url2=j.find('div', class_="hd").a['href']
#         html2=requests.get(url2,headers=head)
#         html2.encoding='utf-8'
#         soup2=BeautifulSoup(html2.text).find('div', id="info")
#         info2['director'].append(soup2.find('span', class_="attrs").a.string)
#         info2['lead_role'].append(soup2.find('span', class_="actor").a.string)
#         info2['language'].append(soup2.findAll('span', class_="pl")[5].string)
#         info2['flength'].append(soup2.find('span', property="v:runtime").string)

Info1.to_excel(r'D:\python操作文件\豆瓣\数据1.xlsx',encoding='utf-8')    #出现\xf4编码在保存csv文件时出现问题


# In[19]:


import requests                           #列表形式
from bs4 import BeautifulSoup
import pandas as pd
from pandas import DataFrame
import csv
ListFinal=[]

def get_info1(text):          #获取文本中所需的数据并保存在列表info1中
    soup=BeautifulSoup(text).find('ol', class_="grid_view").findAll('li')  
    M=[]
    for j in soup:  
        m=[]
        rank=j.find('em', class_="").string
        movie_title=j.find('span', class_="title").string
        meg=j.find('p', class_="").text.strip().split('\n')
        director=meg[0].strip().split('\xa0\xa0')[0].strip('导演: ')   #导演名字太长导致有些切割三个\ax0时出现错误
        year=meg[1].strip().split('\xa0/\xa0')[0]                     
        country=meg[1].strip().split('\xa0/\xa0')[1]
        types=meg[1].strip().split('\xa0/\xa0')[-1]
        score=j.find('div', class_="star").findAll('span')[1].string
        comment_num=j.find('div', class_="star").findAll('span')[3].string.strip('人评价')
        if len(j.find('div', class_="bd").findAll('p'))==2:                 #第7页某电影不存在一句影评模块,因此会导致读取不到信息
            comm_one=j.find('span', class_="inq").string
        else:
            comm_one='无'
        m=[rank,movie_title,director,year,country,types,score,comment_num,comm_one]
        M.append(m)
    return M

for i in range(0,230,25):
    head={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:63.0) Gecko/20100101 Firefox/63.0'}
    url=r'https://movie.douban.com/top250?start={0}&filter='.format(str(i))
    html=requests.get(url,headers=head)
    html.encoding='utf-8'
    ListFinal.extend(get_info1(html.text))
with open(r'D:\python操作文件\豆瓣\数据2.csv','w',newline='',encoding='gb18030') as file:
    w=csv.writer(file)
    w.writerow(['rank','movie_title','director','year','country','types','score','comment_num','comm_one'])
    w.writerows(ListFinal)
    


# In[15]:


import requests                          #课堂使用try提起年份\国家\类型
from bs4 import BeautifulSoup
import pandas as pd
from pandas import DataFrame
import csv

listA=[]
for i in range(0,230,25):    #230
    head={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:63.0) Gecko/20100101 Firefox/63.0'}
    url=r'https://movie.douban.com/top250?start={0}&filter='.format(str(i))
    html=requests.get(url,headers=head)
    html.encoding='utf-8'
    soup=BeautifulSoup(html.text).find('ol', class_="grid_view").findAll('li')
    for j in soup:
        lista=[]
        a=j.find('div', class_="bd").text.strip().split('\n')
        try:
            year=a[1].strip().split('\xa0/\xa0')[0]                     
            country=a[1].strip().split('\xa0/\xa0')[1]
            types=a[1].strip().split('\xa0/\xa0')[-1]
        except:
            year='无'
            country='无'
            types='无'
        lista=[year,country,types]
        listA.append(lista)

with open(r'D:\python操作文件\豆瓣\数据3.csv','w',newline='',encoding='gb18030') as file:
    w=csv.writer(file)
    w.writerow(['year','country','types'])
    w.writerows(listA)
        
    
    


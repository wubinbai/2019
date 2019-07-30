#!/usr/bin/env python
# coding: utf-8

# In[6]:


from bs4 import BeautifulSoup        #初代版
import requests
import pandas as pd
from pandas import DataFrame
import time
import random

def get_data(text,Fdict):
    soup=BeautifulSoup(text,'html.parser').find('div', class_="dw_table", id="resultList")
    try:
        for p in soup.select('.t1 a'):
            Fdict['position'].append(p.string.strip())
        for c in soup.select('.t2 a'):
            Fdict['company'].append(c.string)
        for w in soup.select('.t3')[1:]:
            Fdict['workplace'].append(w.string)
        for s in soup.select('.t4')[1:]:
            Fdict['salary'].append(s.string)
        for t in soup.select('.t5')[1:]:
            Fdict['time'].append(t.string)
    except:
        print('错误')
    return Fdict

Fdict={'position':[],'company':[],'workplace':[],'salary':[],'time':[]}
for i in range(1,3):
    url='https://search.51job.com/list/000000,000000,0000,00,9,99,%25E5%25A4%25A7%25E6%2595%25B0%25E6%258D%25AE%25E5%2588%2586%25E6%259E%2590,2,{0}.html?lang=c&stype=1&postchannel=0000&workyear=99&cotype=99&degreefrom=99&jobterm=99&companysize=99&lonlat=0%2C0&radius=-1&ord_field=0&confirmdate=9&fromType=&dibiaoid=0&address=&line=&specialarea=00&from=&welfare='.format(i)
    head={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.80 Safari/537.36',
         'Referer':'https://search.51job.com/list/000000,000000,0000,00,9,99,%25E5%25A4%25A7%25E6%2595%25B0%25E6%258D%25AE%25E5%2588%2586%25E6%259E%2590,2,1.html?lang=c&stype=&postchannel=0000&workyear=99&cotype=99&degreefrom=99&jobterm=99&companysize=99&providesalary=99&lonlat=0%2C0&radius=-1&ord_field=0&confirmdate=9&fromType=&dibiaoid=0&address=&line=&specialarea=00&from=&welfare='
         }
    html=requests.get(url,headers=head)
    html.encoding='gbk'
    Fdict=get_data(html.text,Fdict)
    time.sleep(random.randint(3,9))
    print('完成%.2f%%'%(i*100/95))
print('已完成所有页面数据抓取,耐心等待数据保存!')
df=DataFrame(Fdict)
df.to_csv(r'D:\python操作文件\前程无忧\前程无忧.csv',encoding='gbk')


# In[3]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
from pandas import DataFrame
import time
import random

def get_data(text,Fdict):
    soup=BeautifulSoup(text,'html.parser').find('div', class_="dw_table", id="resultList")
    try:
        Fdict['position'].extend(list(map(lambda x:x.string.strip() , soup.select('.t1 a'))))
        Fdict['company'].extend(list(map(lambda x:x.string , soup.select('.t2 a'))))
        Fdict['workplace'].extend(list(map(lambda x:x.string , soup.select('.t3')[1:])))
        Fdict['salary'].extend(list(map(lambda x:x.string , soup.select('.t4')[1:])))
        Fdict['time'].extend(list(map(lambda x:x.string , soup.select('.t5')[1:])))
    except:
        print('错误')
    return Fdict

Fdict={'position':[],'company':[],'workplace':[],'salary':[],'time':[]}
for i in range(1,3):
    url='https://search.51job.com/list/000000,000000,0000,00,9,99,%25E5%25A4%25A7%25E6%2595%25B0%25E6%258D%25AE%25E5%2588%2586%25E6%259E%2590,2,{0}.html?lang=c&stype=1&postchannel=0000&workyear=99&cotype=99&degreefrom=99&jobterm=99&companysize=99&lonlat=0%2C0&radius=-1&ord_field=0&confirmdate=9&fromType=&dibiaoid=0&address=&line=&specialarea=00&from=&welfare='.format(i)
    head={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.80 Safari/537.36',
         'Referer':'https://search.51job.com/list/000000,000000,0000,00,9,99,%25E5%25A4%25A7%25E6%2595%25B0%25E6%258D%25AE%25E5%2588%2586%25E6%259E%2590,2,1.html?lang=c&stype=&postchannel=0000&workyear=99&cotype=99&degreefrom=99&jobterm=99&companysize=99&providesalary=99&lonlat=0%2C0&radius=-1&ord_field=0&confirmdate=9&fromType=&dibiaoid=0&address=&line=&specialarea=00&from=&welfare='
         }
    html=requests.get(url,headers=head)
    html.encoding='gbk'
    Fdict=get_data(html.text,Fdict)
    time.sleep(random.randint(3,9))
    print('完成%.2f%%'%(i*100/95))
df=DataFrame(Fdict)
df.head()
# print('已完成所有页面数据抓取,耐心等待数据保存!')
# df=DataFrame(Fdict)
# df.to_csv(r'D:\python操作文件\前程无忧\前程无忧.csv',encoding='gbk')


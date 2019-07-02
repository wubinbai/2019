#!/usr/bin/env python
# coding: utf-8

# In[102]:


import requests
import time
import random
from bs4 import BeautifulSoup as bs
import re
import pandas as pd
from pandas import DataFrame

def get_data(html,Fdict):
    soup=bs(html.text,'html.parser').body.find('div', class_="list-box").findAll( 'div', class_=re.compile('list-item.*') )
    num=len( Fdict['name'] )
    for i in soup:
        num+=1
        Fdict['name'].append( i.find('div', class_="pro-intro").h3.text.strip() ) 
        try:             #整理发布日期
            Fdict['date'].append( i.find('div', class_="price-box").find('span', class_="date").string )
        except:
            Fdict['date'].append( '未发布' )
        try:             #整理价格栏位
            Fdict['price'].append( i.find('div', class_="price-box").find('b', class_="price-type").string )    
        except:
            Fdict['price'].append( i.find('div', class_="price-box").find('span', class_="price price-normal").text.replace(' ','')[1:-6] )
        for j in i.findAll('li'):
            if j.span.string[:-1] not in Fdict:               #增加新的列
                Fdict[ j.span.string[:-1] ] = ['无']*(num-1)
            try:
                Fdict[ j.span.string[:-1] ].append( j['title'] )        #将拥有的数据加入列表中
            except:
                Fdict[ j.span.string[:-1] ].append( 'wu' )
        for k in Fdict:                                        #补充空白的数据
            if len( Fdict[k] ) != num:
                Fdict[k].append('无')
    print('目前收集了%s条数据'%num)
    return Fdict

Fdict={'name':[],'price':[],'date':[]}
head={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.80 Safari/537.36'}
for i in range(1,42):     #共41页
    url='http://detail.zol.com.cn/cell_phone_index/subcate57_0_list_1_0_1_1_0_{0}.html'.format(i)              
    html=requests.get( url,headers=head )
    Fdict=get_data(html,Fdict)
    time.sleep(random.randint(3,10))
    print('已完成%.2f%%'%(i*100/41))
df=DataFrame(Fdict) 
df.to_csv(r'D:\python操作文件\中关村\中关村手机数据.csv',encoding='gbk')
# for j in Fdict:
#     print(len(Fdict[j]))
df.head()


# In[1]:


import pandas as pd
from pandas import DataFrame
import numpy as np
import re
file=pd.read_csv(r'D:\python操作文件\中关村\中关村手机数据.csv',engine='python',encoding='gbk',index_col=0)
file=file[['name','price','date','主屏尺寸','CPU型号','CPU频率','电池容量','后置摄像头','操作系统','RAM容量']]
def get_price(text):
    if text!='概念产品' and text!='即将上市' and text!='价格面议' and text!='停产' and text!='暂无报价':
        if '万' in text:
            return float(text[:-1])*10000
        else:
            return int(text)
    else:
        return None
file['price'] = file['price'].apply(get_price)
file['screen'] = file['主屏尺寸'].str.split('英寸',expand=True)[0]
file['screen_xs'] = file['主屏尺寸'].str.split('英寸',expand=True)[1].str[:-2]
def get_cpu(text):
    if "高通" in text or '海思' in text or '苹果' in text:
        return 1
    elif text=='无': 
        return None
    else:
        return 0
file['CPU型号'] = file['CPU型号'].apply(get_cpu)
# file['CPU型号']=file['CPU型号'].astype('str')
# DataFrame(file['CPU型号']).apply(pd.value_counts)
def get_cpuh(text):
    if '十核' in text[-3:]:
        return 10
    elif '八核' in text[-3:]:
        return 8
    elif '六核' in text[-3:]:
        return 6
    elif '四核' in text[-3:]:
        return 4
    elif '双核' in text[-3:]:
        return 2
    elif '单核' in text[-3:]:
        return 1
    else:
        return None
file['CPUh']=file['CPU频率'].apply(get_cpuh)
def get_bt(text):
    if text=='无':
        return None
    else:
        return text[:text.find('m')]
file['bt(mAh)'] = file['电池容量'].apply(get_bt)
file['bt_type'] = file['电池容量'].str[-7:]
def get_bt_type(text):
    if '不' in text[-8:]:
        return 0
    elif 'mAh' in text[-8:] or '无' in text[-8:] or '时长' in text[-8:]:
        return None
    else:
        return 1
file['bt_type'] = file['电池容量'].apply(get_bt_type)

file=file[file['后置摄像头'].str.contains('\d')]
file['cam(万像素)'] = file['后置摄像头'].apply(lambda x:re.search('\d{2,4}|\d万',x).group()).str.replace('万','')     #有一个8万像素,还有2*1200万像素的,因此采取或者的方式
def get_ram(text):
    if 'M' in text:
        return round(int(text[:text.find('M')])/1024,3)
    elif text=='无':
        return None
    elif text=='6':
        return 6
    else:
        return float(text[:text.find('G')])
file['ram(GB)'] = file['RAM容量'].apply(get_ram)
def get_dbcam(text):
    if '+' in text or '副' in text or '双' in text:
        return 1
    else:
        return 0
file['dbcam']=file['后置摄像头'].apply(get_dbcam)
File = file[['name','price','date','CPU型号','CPUh','bt(mAh)','bt_type','screen','screen_xs','cam(万像素)','ram(GB)','dbcam']]
File.notnull().all()


# In[10]:


import datetime 
import time
File=File.dropna()
File['date'] = File['date'].astype('datetime64')
File['new_time']= File['date'].apply( lambda x:datetime.datetime.now()-x ).astype('str').apply( lambda x:x[:x.find('d')] )
File.head()
# File.to_excel(r'D:\python操作文件\中关村\手机数据1.xlsx')


# In[325]:


num=0
for i in a:
    print(i)
    print(re.search('\d{2,4}|\d万',i).group())
    num+=1
    print(num)
    print('````````````````````````````````')
# print('彩色1600万像素+彩色200+万像素'.count('+'))


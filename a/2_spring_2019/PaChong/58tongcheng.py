#!/usr/bin/env python
# coding: utf-8

# In[24]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
from pandas import DataFrame
import time
import csv
###############################自定义获取数据的函数#################################
def get_data(text):             ##传入网页完整文本信息    
    soup=BeautifulSoup(text).find('ul', class_="listUl").findAll('li')
    flist=[]
    for i in soup:
        try:
            roomtype=i.find('div', class_="des").p.string.split('\xa0\xa0\xa0\xa0')[0].strip()
            roomarea=i.find('div', class_="des").p.string.split('\xa0\xa0\xa0\xa0')[1].strip()
            region=i.find('p', class_="add").text.split('\xa0\xa0')[0].strip()
            address=i.find('p', class_="add").text.split('\xa0\xa0')[1].strip()
            price=i.find('div', class_="money").b.string
            data=i.find('div', class_="des").findAll('p')
            if len(data)==2:
                data1=i.find('div', class_="des").find('div', class_="jjr").text
                comefrom_type=data1.split('：')[0].strip()
                comefrom_contact=data1.split('：')[1].replace(' ','').replace('\n\n','')
            else:
                data2=i.find('div', class_="des").findAll('p')[2].text
                comefrom_type=data2.split('：')[0].strip()
                comefrom_contact=data2.split('：')[1].strip()
        except:
            roomtype='无'
            roomarea='无'
            region='无'
            address='无'
            price='无'
            comefrom_type='无'
            comefrom_contact='无'
        flist.append([ roomtype , roomarea , region , address , price, comefrom_type , comefrom_contact ])
    return flist
################################请求网页信息主体##################################
Flist=[]
for i in range(1,71):
    url='https://xm.58.com/chuzu/pn{0}/?utm_source=sem-baidu-pc&spm=105916146539.26420796315&PGTID=0d3090a7-0025-e14b-4966-8c14a46831c7&ClickID=2'.format(str(i))
    head={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:63.0) Gecko/20100101 Firefox/63.0'}
    html=requests.get(url,headers=head)
    html.encoding='utf-8'
    Flist.extend(get_data(html.text))                      ####得到的数据均保存在Flist列表中
    print('已经完成{:.2f}%'.format(i*100/70))
    time.sleep(15)
####################################数据保存######################################
DFlist=DataFrame(Flist,colunms=[ 'roomtype' , 'roomarea' , 'region' , 'address' , 'price' , 'comefrom_type' , 'comefrom_contact' ])
DFlist.to_csv(r'D:\python操作文件\58同城\shuju1.csv',encoding='gbk')
with open(r'D:\python操作文件\58同城\shuju2.csv','w',newline='',encoding='gbk') as f:
    file=csv.writer(f)
    file.writerow([ 'roomtype' , 'roomarea' , 'region' , 'address' , 'price' , 'comefrom_type' , 'comefrom_contact' ])
    file.writerows(Flist)


# In[3]:


from bs4 import BeautifulSoup                          #params使用尝试以及timesleep与random结合
import pandas as pd
from pandas import DataFrame
import requests
import random
import time
import csv
url='https://xm.58.com/chuzu/pn1/'
pdict={'utm_source':'sem-baidu-pc',
       'spm':'105916146539.26420796315',
       'PGTID':'0d3090a7-0025-e14b-4966-8c14a46831c7',
       'ClickID':'2'}
head={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:63.0) Gecko/20100101 Firefox/63.0'}
html=requests.get(url,params=pdict,headers=head)
html.encoding='UTF-8'
soup=BeautifulSoup(html.text).find('ul', class_="listUl").findAll('li')
flist=[]
for i in soup:
    try:
        roomtype=i.find('div', class_="des").p.string.split('\xa0\xa0\xa0\xa0')[0].strip()
        roomarea=i.find('div', class_="des").p.string.split('\xa0\xa0\xa0\xa0')[1].strip()
        m1=i.find('p', class_="add").text
        region=i.find('p', class_="add").text.split('\xa0\xa0')[0].strip()
        address=i.find('p', class_="add").text.split('\xa0\xa0')[1].strip()
        price=i.find('div', class_="money").b.string
        data=i.find('div', class_="des").findAll('p')
        if len(data)==2:
            data1=i.find('div', class_="des").find('div', class_="jjr").text
            comefrom_type=data1.split('：')[0].strip()
            comefrom_contact=data1.split('：')[1].replace(' ','').replace('\n\n','')
        else:
            data2=i.find('div', class_="des").findAll('p')[2].text
            comefrom_type=data2.split('：')[0].strip()
            comefrom_contact=data2.split('：')[1].strip()
    except:
        m1='w'
        roomtype='无'
        roomarea='无'
        region='无'
        address='无'
        price='无'
        comefrom_type='无'
        comefrom_contact='无'
    flist.append([ roomtype , roomarea , region , address , price, comefrom_type , comefrom_contact ])


print('程序运行完成,请等待%s秒'%random.randint(3,10))
time.sleep(random.randint(3,10))
with open(r'D:\python操作文件\58同城\sssss1.csv','w',newline='',encoding='gbk') as f:
    csv.writer(f).writerow([ 'roomtype' , 'roomarea' , 'region' , 'address' , 'price' , 'comefrom_type' , 'comefrom_contact' ])
    csv.writer(f).writerows(flist)
print('等待结束')


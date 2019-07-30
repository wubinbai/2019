#!/usr/bin/env python
# coding: utf-8

# In[28]:


import requests 
import json
import csv
url='https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv2124&productId=3311073&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1'
head={
    'Accept':'*/*',
    'Accept-Ecoding':'gzip, deflate, br',
    'Accept-Language':'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
    'Connection':'keep-alive',
    'Cookie':'__jda=122270672.222191952.1544061528.1544061528.1544061530.1; unpl=V2_ZzNtbUVVQxZ2AEVUfhsMAWJTE1tKUUsQfV1FVX4dWwFmUEYIclRCFXwURldnGFsUZgsZWUtcRhJFCEdkexhdBGYKGlRKVXMVcQ8oVRUZWw1hbSJtQVdzHUULRVd7EVwNYwAibUVnc8LcppDY3c3lr7CxkG1GUUATcQlHVXgpXTVmM1kzQxpAFnYITlRzHV81ZjMR; __jdb=122270672.5.222191952|1.1544061530; __jdc=122270672; __jdv=122270672|www.linkstars.com|t_1000089893_156_0_1697_|tuiguang|6312292143a54a0797949d2045650bed|1544061529627; __jdu=222191952; PCSYCityID=1315; shshshfp=508049d49e4f40bc61d1b98b08b17b4f; shshshfpa=3adaf07a-d523-290e-99aa-e7703cb8697e-1544061530; shshshsID=38676382b13d1bc259783ca70fadbd88_4_1544063289588; shshshfpb=1297677104d9d46f4a61023673eb8e766d56a84ca5556a6b55c088258c; 3AB9D23F7A4B3C9B=TTZSKXQMQZ67CXIMLCVGTA6IJ3ZV3NZIOY2ZFYCJLXSRNHQH2NLNWDAOLCNC7HWVGGI5JR7XDL3SVLFXXAKOVNAB6Y; _gcl_au=1.1.533661972.1544061572; JSESSIONID=5E8846A54C6AFB0734EDFBC654B05BB3.s1',
    'Host':'sclub.jd.com',
    'Referer':'https://item.jd.com/3311073.html',
    'TE':'Trailers',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:63.0) Gecko/20100101 Firefox/63.0',
     }
html=requests.get(url,headers=head)
html.encoding='gbk'

file=html.text.replace('fetchJSON_comment98vv2124(','').replace(');','')
soup=json.loads(file)

label=[]
com=[]
for i in soup['hotCommentTagStatistics']:
    label.append(i['name'])
for j in soup['comments']:
    c=[j['content'],j['creationTime'],j['score'],j['userLevelName'],j['userClientShow']]
    com.append(c)
# print(label)
# print(com)
with open(r'D:\python操作文件\京东评论\评论数据.csv','w',newline='',encoding='gbk') as f:
    csv.writer(f).writerow(label)
    csv.writer(f).writerow(['content','creationtime','score','userlevelname','userclientshow'])
    csv.writer(f).writerows(com)


# In[2]:


import requests
from bs4 import BeautifulSoup
url='https://item.jd.com/3311073.html'
html=requests.get(url)
html.encoding='gbk'
soup=BeautifulSoup(html.text,'lxml').find('div', class_="p-parameter")
print(soup)


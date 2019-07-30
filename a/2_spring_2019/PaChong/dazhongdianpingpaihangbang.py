#!/usr/bin/env python
# coding: utf-8

# In[20]:


import requests
import time
import json
from bs4 import BeautifulSoup
url='http://www.dianping.com/mylist/ajax/shoprank?rankId=9342352ff0d3807a775a93b688e382b6'
head={'Accept':'*/*',
      'Accept-Encoding': 'gzip, deflate',
      'Accept-Language': 'zh-CN,zh;q=0.9',
      'Connection': 'keep-alive',
      'Cookie': 'UM_distinctid=16787662ab1277-0f6db90f9a2a34-b78173e-144000-16787662ab229c; _lxsdk_cuid=16787662d14c8-008dd4e4a8c0ca-b78173e-144000-16787662d149a; _lxsdk=16787662d14c8-008dd4e4a8c0ca-b78173e-144000-16787662d149a; _hc.v=a005914f-caee-8b03-362b-7723b966cd73.1544164880; s_ViewType=10; cy=15; cye=xiamen; Hm_lvt_f5df380d5163c1cc4823c8d33ec5fa49=1544267312,1544267532,1544267584,1544269361; Hm_lpvt_f5df380d5163c1cc4823c8d33ec5fa49=1544408132; CNZZDATA1271442956=315510241-1544161777-https%253A%252F%252Fwww.baidu.com%252F%7C1544404066; _lx_utm=utm_source%3DBaidu%26utm_medium%3Dorganic; _lxsdk_s=16795e5c926-ceb-371-356%7C%7C22',
      'Host':'www.dianping.com',
      'Referer': 'http://www.dianping.com/shoplist/shopRank/pcChannelRankingV2?rankId=9342352ff0d3807a775a93b688e382b6',
      'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.80 Safari/537.36',
      'X-Requested-With': 'XMLHttpRequest'
     }
html=requests.get(url,headers=head)
# html.encoding='gbk'
tar=json.loads(html.text)
print(type(html.text))


# In[21]:


import pandas
from pandas import DataFrame
Flist=[]
for i in tar['shopBeans']:
    flist=[]
    add=i['address']
    avgPrice=i['avgPrice']
    branchName=i['branchName']
    types=i['mainCategoryName']
    region=i['mainRegionName']
    tasteScore=i['refinedScore1']
    envScore=i['refinedScore2']
    servScore=i['refinedScore3']
    shopId=i['shopId']
    shopName=i['shopName']
    flist=[shopId,shopName,region,add,branchName,avgPrice,types,tasteScore,envScore,servScore]
    Flist.append(flist)
Flist=DataFrame(Flist,columns=['shopId','shopName','region','add','branchName','avgPrice','types','tasteScore','envScore','servScore'])
Flist.to_excel(r'D:\python操作文件\大众点评\大众点评火锅排行榜.xlsx')    #使用此方法,无需对页面进行encoding
# Flist.to_csv(r'D:\python操作文件\大众点评\大众点评火锅排行榜.csv')   #乱码


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
import json
import time
import random
import hashlib
url='http://fanyi.youdao.com/translate_o?smartresult=dict&smartresult=rule'
content=input('请输入要翻译的内容:')

S = "fanyideskweb"
n = content
r =str(int(time.time() * 1000) + random.randint(0, 10))
D = "ebSeFb%=XZ%T[KZ)c(sy!"
sign = hashlib.md5((S + n + r + D).encode('utf-8')).hexdigest()


data={'action':'FY_BY_REALTIME',
      'bv':'e2a78ed30c66e16a857c5b6486a1d326',
      'client':'fanyideskweb',
      'doctype':'json',
      'from':'AUTO',
      'i':n,
      'keyfrom':'fanyi.web',
      'salt':r,
      'sign':sign,
      'smartresult':'dict',
      'to':'AUTO',
#       'ts':'1544009129987',
      'typoResult':'false',
      'version':'2.1'}
headers = {
           'Cookie': 'OUTFOX_SEARCH_USER_ID=782075878@10.169.0.84',
           'Referer': 'http://fanyi.youdao.com/',
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:63.0) Gecko/20100101 Firefox/63.0',
           }
html=requests.post(url,data=data,headers=headers).text
print(html)
text=json.loads(html)
print(text)
if 'translateResult' in text:
    try:
        result=text['translateResult'][0][0]['tgt']
    except:
        result='失败'
print('翻译结果为:%s'%result)


# In[ ]:





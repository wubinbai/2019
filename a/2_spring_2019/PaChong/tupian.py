#!/usr/bin/env python
# coding: utf-8

# In[17]:


from bs4 import BeautifulSoup
import requests
for i in range(1,2):
    F=open(r'D:\ppt图片源码%s.txt'%i).read()
    soup=BeautifulSoup(F)
    tar=soup.body.find('dl', class_="dlbox").find('ul', class_="tplist")
    for i in tar.findAll('li'):
        url=i.img.get('src')
        name=i.img.get('alt')
        path=r'E:\tupian\%s.jpg'%name
#         urlinfo=requests.get(url)      
#         f=open(path,'wb')
#         f.write(urlinfo.content)
#         f.close()
        save_poto(path,url)
    
def save_poto(path,url):
    urlinfo=requests.get(url)
    f=open(path,'wb')
    f.write(urlinfo.content)
    f.close()


# In[ ]:


import re
import requests
f=open('D:\ppt图片源码1.txt','r',encoding='gbk').read()
p=re.findall('<dl class="dlbox">(.*?)<table width=',f,re.S)[0]
urllist=re.findall('img src="(.*?)" alt',p,re.S)
namelist=re.findall(' alt="(.*?)"/></a>',p)
# print(url)

def save_photo(url,path):
    urlinfo=requests.get(url)
    ff=open(path,'wb')
    ff.write(urlinfo.content)
    ff.close()
    
for url,j in zip(urllist,namelist):
    path=r'E:\tupian\tupian\%s.jpg'%j
    save_photo(url,path)
    
#     urlinfo=requests.get(i)
#     poto=urlinfo.content
#     ff=open(path,'wb')
#     ff.write(poto)
#     ff.close


# def get_imag(path,url,name):
#     h=requests.get(url)
#     n='{0}.jpg'.format(name)
#     lujing=os.path.join(path,n)     #path.join的用法
#     f=open(lujing,'wb')
#     f.write(h.content)
#     f.close()


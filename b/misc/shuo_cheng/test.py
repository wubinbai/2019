import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

data = np.load("data.npy",allow_pickle=True)

cc = data

arr1 = cc[1] > 0.5e13
ratio1_meth1 = arr1.cumsum()[-1]/len(arr1)
df = pd.DataFrame(cc).transpose()
df[2] = df[1] > 0.5e13
vc = df[2].value_counts()
ratio1_meth2 = vc[1]/df[2].shape[0]
# Extract hour to column '3'
df[3] = df[0].apply(lambda x: x.hour)
select_df = df[df[2]]
gb = select_df.groupby(3).count()
print('')
print('======')
print("Answer to question 1:")
print('To view answer to question 1, use q1() function.')
print('To view steps to question 2, use q2(n) function, where n is the number of plots you want to plot.')
print('I think for question two, it may be better to first answer the question, is the noise continuous or not?')
def q1():
    print("ratio of anomaly is: ", ratio1_meth2)
    print("24 hours anomaly distribution is: ")
    print(gb)
    gb.plot.pie(0)
    gb.plot.bar(0)

df1 = df[1]
n = 0
def q2(start_from):
    print("Calling q2() once will plot 1 graph a time; each time you call this function, previous graph will be closed.")
    global n
    window = 24*4
    w = window

    s = start_from + n * w
    n+=1
    e = start_from + n * w
    global df1
    plt.close()
    df1[s:e].plot('line',style = 'r-*')
    plt.show()
    time.sleep(1)

def show_time(start_row):
    print(df[0][start_row:start_row+24])

# for visualizing some data:
# l = range(12000-24*6,12000+24*17,24)
# xv = list(l)
# for ii in xv:
#     plt.axvline(x=ii)
# df1 = df[1]
# df1[12000-24*6:12000+24*2+24*2].plot()

# finding's in the following:
print("Time starts from: ")
print('0            2019-02-21 15:12:17')
print('1     2019-02-21 15:12:17.041667')
print('to ...')
print('23    2019-02-21 15:12:17.958341')
print('to ...')
print('2072568           2019-02-22 15:12:13')
print('to ...')
print('2072591    2019-02-22 15:12:13.958341')










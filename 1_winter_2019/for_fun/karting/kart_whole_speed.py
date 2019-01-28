import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('alldata.xls',header=None)

data[data>60] = 60
plt.ion()
plt.plot(data)
plt.grid()
plt.show()



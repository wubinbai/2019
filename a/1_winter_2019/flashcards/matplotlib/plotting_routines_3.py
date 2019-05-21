from prepare_data import *
from create_plot import *

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
lines = ax.plot(x,y)
ax.scatter(x,y)
axes[0,0].bar([1,2,3],[3,4,5])
axes[1,0].barh([0.5,1,2.5],[0.5,1,2])
axes[1,1].axhline(0.45)
axes[0,1].axvline(0.65)
ax.fill(x,y,color='blue')
ax.fill_between(x,y,color='yellow')


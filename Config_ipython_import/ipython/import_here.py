import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=100)

import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns

def plot_whole(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
        print(df)

h = help
# Better help function he():

def he(): 
    global ar
    ar = input('Enter the function name for help:')
    help(eval(ar))
# for . operation of dir
# use eval(repr(xxx.i))

from pandas import read_csv as pdrc

# for ipython to display all results in the jupyter notebook:
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

def my_plot(data_array):
    plt.figure()
    plt.plot(data_array)
    plt.grid()

def my_plotas(data_array):
    plt.figure()
    plt.plot(data_array)
    plt.plot(data_array,'b*')
    plt.grid()

def save_model_keras(model,save_path):
    from keras.utils import plot_model
    plot_model(model,show_shapes=True,to_file=save_path)





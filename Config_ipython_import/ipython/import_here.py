import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=100)

import pandas as pd
import matplotlib.pyplot as plt
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

from pandas import read_csv as pdrc


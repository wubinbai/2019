import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_whole(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
        print(df)

h = help

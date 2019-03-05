import pandas as pd
import numpy as np
import seaborn as sns
uniform_data = np.random.rand(10,12)
data = pd.DataFrame({'x':np.arange(1,101),'y':np.random.normal(0,4,100)})

titanic = sns.load_dataset("titanic")
iris = sns.load_dataset("iris")


import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")
sns.set_style("whitegrid")
g = sns.lmplot(x="tip",
               y="total_bill",
               data=tips,
               aspect=2)
g = (g.set_axis_labels("Tip","Total bill(USD)").set(xlim=(0,10),ylim=(0,100)))
plt.show(g)

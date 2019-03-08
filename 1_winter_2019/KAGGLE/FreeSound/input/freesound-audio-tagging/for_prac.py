import pandas as pd
train = pd.read_csv("train.csv")
test = pd.read_csv("sample_submission.csv")
category_group = train.groupby(['label','manually_verified']).count()
toc = category_group.index.levels[0]
for i in range(len(toc)):
    print(toc[i],": ")
    print(category_group.loc[toc[i]])
    print('######')
    
plot = category_group.unstack()
plot_sum = plot.sum(axis=1)
plot_sum_sort = plot_sum.sort_values()
ind = plot_sum_sort.index
tempdf = plot.reindex(ind)

to_plot = tempdf.plot(kind='bar',stacked=True,title="Number of Audio Samples per CAtegory",figsize=(16,10))
to_plot.set_xlabel("Category")
to_plot.set_ylabel("Number of Samples")

print("Minimum samples per category = ", min(train.label.value_counts()))
print("Maximum samples per category = ", max(train.label.value_counts()))

import IPython.display as ipd
fname = ''

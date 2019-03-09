import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("sample_submission.csv")
category_group = train.groupby(['label','manually_verified']).count()
toc = category_group.index.levels[0]
"""for i in range(len(toc)):
    print(toc[i],": ")
    print(category_group.loc[toc[i]])
    print('######')
"""

plot = category_group.unstack()
plot_sum = plot.sum(axis=1)
plot_sum_sort = plot_sum.sort_values()
ind = plot_sum_sort.index
tempdf = plot.reindex(ind)
plt.ion()
to_plot = tempdf.plot(kind='bar',stacked=True,title="Number of Audio Samples per CAtegory",figsize=(16,10))
to_plot.set_xlabel("Category")
to_plot.set_ylabel("Number of Samples")
plt.show()

print("Minimum samples per category = ", min(train.label.value_counts()))
print("Maximum samples per category = ", max(train.label.value_counts()))

import IPython.display as ipd
fname = '00044347.wav'
ipd.Audio(fname)

import wave
wav = wave.open(fname)
print("Sampling (frame) rate = ", wav.getframerate())
print("Total samples (frames) = ", wav.getnframes())
print("Duration = ", wav.getnframes()/wav.getframerate(), 's')

from scipy.io import wavfile
rate, data = wavfile.read(fname)
print("Sampling (frame) rate = ", rate)
print("Total samples (frames) = ", data.shape)
print(data)

plt.figure()
plt.plot(data,'-',)

plt.figure(figsize=(16,4))
plt.plot(data[:500],'.')
plt.plot(data[:500],'-')

train['nframes'] = train['fname'].apply(lambda f: wave.open(f).getnframes())

test['nframes'] = test['fname'].apply(lambda f: wave.open(f).getnframes())





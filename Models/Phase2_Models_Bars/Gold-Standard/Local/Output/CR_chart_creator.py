import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('SGD_CR.csv', index_col='Opinion')
df.plot(kind='bar', ylim=(0,1), figsize=(16,9))
plt.xticks(rotation='horizontal')
ax = plt.axes()
ax.yaxis.grid()
ax.set_axisbelow(True)
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+0.04, i.get_height()+0.02, str(round(i.get_height(), 2)))

plt.show()

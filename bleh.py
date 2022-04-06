import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('whitegrid')
from scipy import interpolate
from scipy import stats

skips = np.array([1, 3, 6, 30, 60, 120])
skips = np.array([120, 60, 30, 6, 3, 1])

sz = np.array(3000/skips, dtype=int)
szz = np.array(sz, dtype=str)

d = {}
for skip in skips:
    base = pd.read_csv(f'perf_data2.base_{skip}.csv').iloc[:, 1]

    hy = pd.read_csv(f'perf_data2.hybrid_{skip}.csv').iloc[:, 1]

    loss = pd.read_csv(f'perf_data2.loss_{skip}.csv').iloc[:, 1]

    trans = pd.read_csv(f'perf_data2.transfer_{skip}.csv').iloc[:, 1]

    d[(f'l{int(3000 / skip)}', 'baseline')] = base.values
    d[(f'l{int(3000 / skip)}', 'hybrid')] = hy.values
    d[(f'l{int(3000 / skip)}', 'loss')] = loss.values
    d[(f'l{int(3000 / skip)}', 'transfer')] = trans.values
    # d2[(f'{int(3000 / skip)}', 'baseline')] = d1
df = pd.DataFrame(d)


plt.violinplot(df.l25)
plt.title('25 datapoints')
plt.ylabel('MAE')
plt.show()
plt.violinplot(df.l50)
plt.title('50 datapoints')
plt.ylabel('MAE')
plt.show()
plt.violinplot(df.l100)
plt.title('100 datapoints')
plt.ylabel('MAE')
plt.show()
plt.violinplot(df.l500)
plt.title('500 datapoints')
plt.ylabel('MAE')
plt.show()
plt.violinplot(df.l1000)
plt.title('1000 datapoints')
plt.ylabel('MAE')
plt.show()
plt.violinplot(df.l3000)
plt.title('3000 datapoints')
plt.ylabel('MAE')
plt.show()



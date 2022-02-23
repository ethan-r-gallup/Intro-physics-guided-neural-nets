import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import winsound


baseline = pd.read_csv('training data/baseline_training.csv').drop(['Unnamed: 0'], axis=1)
print(baseline.columns)

phyloss = pd.read_csv('training data/hybrid_training.csv').drop(['Unnamed: 0'], axis=1)
print(phyloss.columns)

hlen = []
for skipsize in range(1, 100, 1):
    mask = np.arange(100, 3100, skipsize)
    lm = len(mask)
    hlen.append(lm)

phy = np.array([])
base = np.array([])
for col, hcol in zip(baseline.columns, hlen):
    p = pd.read_csv(f'performance data/hybrid_perf_{hcol}.csv')
    print(p)
    pme = np.mean(abs(p.pred - p.Cc)**2)
    b = pd.read_csv(f'performance data/baseline_perf_{col}.csv')
    bme = np.mean(abs(b.Cc_pred - b.Cc)**2)
    phy = np.append(phy, pme)
    base = np.append(base, bme)

plt.plot(base, label='base')
plt.plot(phy, label='phy')
plt.legend()
plt.show()

# for i in baseline.columns:
#     plt.semilogy(baseline[[i]], label='base')
#     plt.semilogy(phyloss[[i]], label='phyloss')
#     plt.legend()
#     plt.title(f'{i}')
#     plt.ylim((0, 0.5))

x = np.arange(1, 501, 1)
print(phyloss.columns)
y = np.array(phyloss.columns, dtype=float)
print(y)
X, Y = np.meshgrid(x, y)
Z = np.array(phyloss.iloc[:, 0:100].T)
Zhat = savgol_filter(Z, 51, 3, axis=0)
Z2 = np.array(baseline.iloc[:, 0:100].T)
Zhat2 = np.array(savgol_filter(Z, 51, 3, axis=0))
print(Z.shape)
print(X.shape, Y.shape, Z.shape, Zhat.shape)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, np.log10(Z))
ax.plot_surface(X, Y, np.log10(Z2))
ax.set_title('wireframe')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

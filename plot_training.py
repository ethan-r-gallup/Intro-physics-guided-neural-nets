
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_theme()
sns.set_style('whitegrid')
mpl.rcParams['lines.markeredgecolor'] = 'w'
mpl.rcParams['lines.markeredgewidth'] = .5
# mpl.rcParams['font.family'] = 'fantasy'
import numpy as np
import pandas as pd

baseline = pd.read_csv('training data/baseline_training.csv').drop(['Unnamed: 0'], axis=1)
# print(baseline.columns)

phyloss = pd.read_csv('training data/phyloss_training4.csv').drop(['Unnamed: 0'], axis=1)
# print(phyloss.columns)
phytransfer = pd.read_csv('training data/transfer_training_2.csv').drop(['Unnamed: 0'], axis=1)
phyhybrid = pd.read_csv('training data/hybrid_training2.csv').drop(['Unnamed: 0'], axis=1)

hlen = []
for skipsize in range(1, 100, 1):
    mask = np.arange(100, 3100, skipsize)
    lm = len(mask)
    hlen.append(lm)

xs = np.array([])
for i in range(1, 100):
    mask = np.arange(100, 3100, i)
    xs = np.append(xs, len(mask))
print(xs)

phy = np.array([])
base = np.array([])
hybrid = np.array([])
transfer = np.array([])
transfer2 = np.array([])
cntrl = np.array([])

for col, hcol in zip(baseline.columns, hlen):
    p = pd.read_csv(f'performance data/phyloss_perf_{col}4.csv')
    # print(p)
    pme = np.mean(abs(p.Cc_pred - p.Cc)**2)
    h = pd.read_csv(f'performance data/hybrid_perf_{col}.csv')
    # print(h)
    hme = np.mean((h.pred - h.Cc)**2)
    t = pd.read_csv(f'performance data/transfer_perf_{col}.csv')
    # print(t)
    tme = np.mean((t.Cc_pred - t.Cc)**2)
    t2 = pd.read_csv(f'performance data/transfer_perf_{col}_2.csv')
    # print(t2)
    tme2 = np.mean((t2.Cc_pred - t2.Cc)**2)
    b = pd.read_csv(f'performance data/baseline_perf_{col}_2.csv')
    bme = np.mean(abs(b.Cc_pred - b.Cc)**2)

    phy = np.append(phy, pme)
    base = np.append(base, bme)
    hybrid = np.append(hybrid, hme)
    transfer = np.append(transfer, tme)
    transfer2 = np.append(transfer2, tme2)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.plot(base, linewidth=1.75, color='white')
ax.plot(base, label='base', linewidth=1.5, color='tab:blue')
ax.plot(phy, linewidth=1.75, color='white')
ax.plot(phy, label='phy loss', linewidth=1.5, color='tab:orange')
plt.xticks(np.arange(0, 101, 20), ('3000', '150', '75', '50', '38', '30'))
plt.xlabel('Data points')
plt.ylabel('Performance')
plt.legend()
plt.show()

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.plot(base, linewidth=1.75, color='white')
ax.plot(base, label='base', linewidth=1.5, color='tab:blue')
ax.plot(hybrid, linewidth=1.75, color='white')
ax.plot(hybrid, label='hybrid', linewidth=1.5, color='tab:green')
plt.xticks(np.arange(0, 101, 20), ('3000', '150', '75', '50', '38', '30'))
plt.xlabel('Data points')
plt.ylabel('Performance')
plt.legend()
plt.show()

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.plot(base, linewidth=1.75, color='white')
ax.plot(base, label='base', linewidth=1.5, color='tab:blue')
ax.plot(transfer2, linewidth=1.75, color='white')
ax.plot(transfer2, label='transfer2', linewidth=1.5, color='tab:purple')
plt.xticks(np.arange(0, 101, 20), ('3000', '150', '75', '50', '38', '30'))
plt.xlabel('Data points')
plt.ylabel('Performance')
plt.legend()
plt.show()

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.plot(base, linewidth=1.75, color='white')
ax.plot(base, label='base', linewidth=1.5, color='tab:blue')
ax.plot(phy, linewidth=1.75, color='white')
ax.plot(phy, label='phy loss', linewidth=1.5, color='tab:orange')
ax.plot(hybrid, linewidth=1.75, color='white')
ax.plot(hybrid, label='hybrid', linewidth=1.5, color='tab:green')
ax.plot(transfer2, linewidth=1.75, color='white')
ax.plot(transfer2, label='transfer2', linewidth=1.5, color='tab:purple')
plt.xticks(np.arange(0, 101, 20), ('3000', '150', '75', '50', '38', '30'))
plt.xlabel('Data points')
plt.ylabel('Performance')
plt.legend()
plt.show()


for i in baseline.columns:
    plt.semilogy(baseline[[i]], linewidth=1.75, color='white')
    plt.semilogy(baseline[[i]], label='base', linewidth=1.5, color='tab:blue')
    plt.semilogy(phyloss[[i]], linewidth=1.75, color='white')
    plt.semilogy(phyloss[[i]], label='phyloss', linewidth=1.5, color='tab:orange')
    plt.semilogy(phyhybrid[[i]], linewidth=1.75, color='white')
    plt.semilogy(phyhybrid[[i]], label='hybrid', linewidth=1.5, color='tab:green')
    plt.semilogy(phytransfer[[i]], linewidth=1.75, color='white')
    plt.semilogy(phytransfer[[i]], label='transfer', linewidth=1.5, color='tab:purple')
    plt.legend()
    plt.ylabel('MSE loss')
    plt.xlabel('epochs')
    plt.title(f'Training loss with {xs[int(i)-1]:.0f} data points')
    plt.show()


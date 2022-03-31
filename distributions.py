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
print(szz)

d = {}
# df = pd.DataFrame([])
for skip in skips:
    base = pd.read_csv(f'perf_data2.base_{skip}.csv').iloc[:, 1]

    hy = pd.read_csv(f'perf_data2.hybrid_{skip}.csv').iloc[:, 1]

    loss = pd.read_csv(f'perf_data2.loss_{skip}.csv').iloc[:, 1]

    trans = pd.read_csv(f'perf_data2.transfer_{skip}.csv').iloc[:, 1]

    d[(f'{int(3000 / skip)}', 'baseline')] = base.values
    d[(f'{int(3000 / skip)}', 'hybrid')] = hy.values
    d[(f'{int(3000 / skip)}', 'loss')] = loss.values
    d[(f'{int(3000 / skip)}', 'transfer')] = trans.values
    # d2[(f'{int(3000 / skip)}', 'baseline')] = d1
df = pd.DataFrame(d)
print(df)


baseline = pd.DataFrame([])
hybrid = pd.DataFrame([])
phyloss = pd.DataFrame([])
transfer = pd.DataFrame([])
for skip in skips:
    base = pd.read_csv(f'perf_data2.base_{skip}.csv').iloc[:, 1]
    baseline = pd.concat([baseline, base.rename(f'{int(3000 / skip)}')], axis=1)

    hy = pd.read_csv(f'perf_data2.hybrid_{skip}.csv').iloc[:, 1]
    hybrid = pd.concat([hybrid, hy.rename(f'{int(3000 / skip)}')], axis=1)

    loss = pd.read_csv(f'perf_data2.loss_{skip}.csv').iloc[:, 1]
    phyloss = pd.concat([phyloss, loss.rename(f'{int(3000 / skip)}')], axis=1)

    trans = pd.read_csv(f'perf_data2.transfer_{skip}.csv').iloc[:, 1]
    transfer = pd.concat([transfer, trans.rename(f'{int(3000 / skip)}')], axis=1)


# baseplot = plt.boxplot(baseline, positions=np.array(np.arange(len(sz)))*2.0, widths=0.3)
# hyplot = plt.boxplot(hybrid, positions=np.array(np.arange(len(sz)))*2.0+0.35, widths=0.3)
# lossplot = plt.boxplot(phyloss, positions=np.array(np.arange(len(sz)))*2.0+0.7, widths=0.3)
# transplot = plt.boxplot(transfer, positions=np.array(np.arange(len(sz)))*2.0+1.05, widths=0.3)
# x = np.linspace(1, 121, 300)
# # bmcs = interpolate.CubicSpline(skips, baseline.mean())
#
# # print(baseline.mean())
# # plt.plot(x, bmcs(x))
# # plt.plot(np.flip(sz), baseline.mean().iloc[::-1])
#
# def define_box_properties(plot_name, color_code, label):
#     for k, v in plot_name.items():
#         plt.setp(plot_name.get(k), color=color_code)
#
#     # use plot function to draw a small line to name the legend.
#     plt.plot([], c=color_code, label=label)
#     plt.legend()
#
#
# # setting colors for each groups
# define_box_properties(baseplot, '#D7191C', 'Baseline')
# define_box_properties(hyplot, 'green', 'Architecture')
# define_box_properties(lossplot, '#2C7BB6', 'PhyLoss')
# define_box_properties(transplot, 'orange', 'transfer')
#
#
# # set the x label values
# plt.xticks(np.arange(0, len(sz)* 2, 2), sz)
#
# # set the limit for x axis
# plt.xlim(-2, len(sz)*2)
# # plt.ylim(0, .00175)
#
#
# # set the title
# plt.title('Performance Distributions')
# df.to_csv('full_data.csv')
# baseline.to_csv('base_dist.csv')
# hybrid.to_csv('hybrid_dist.csv')
# phyloss.to_csv('loss_dist.csv')
# transfer.to_csv('transfer_dist.csv')
# plt.show()

df = pd.melt(df)
hy = df[df['variable_1'].isin(['baseline', 'hybrid'])]
loss = df[df['variable_1'].isin(['baseline', 'loss'])]
trans = df[df['variable_1'].isin(['baseline', 'transfer'])]

mypal = sns.color_palette()
p1 = sns.stripplot(x='variable_0', y='value', hue='variable_1', data=df, dodge=True, jitter=False, linewidth=.75, size=1.75)
p2 = sns.violinplot(x='variable_0', y='value', hue='variable_1', data=df, linewidth=1, scale='width', bw=.75, cut=0, inner='stick', alpha=.5)
h, l = p2.get_legend_handles_labels()
plt.legend(h[0:4], l[0:4])
plt.savefig('plots/df_violin_stick.png')
plt.show()

p1 = sns.stripplot(x='variable_0', y='value', hue='variable_1', data=hy, dodge=True, jitter=False, linewidth=.75, size=1.75)
p2 = sns.violinplot(x='variable_0', y='value', hue='variable_1', data=hy, linewidth=1, scale='width', bw=.5, cut=0, inner='stick', alpha=.5)
h, l = p2.get_legend_handles_labels()
plt.legend(h[0:2], l[0:2])
plt.savefig('plots/hy_violin_stick.png')
plt.show()

p1 = sns.stripplot(x='variable_0', y='value', hue='variable_1', data=loss, dodge=True, jitter=False, linewidth=.75, size=1.75, palette=[mypal[0], mypal[2]])
p2 = sns.violinplot(x='variable_0', y='value', hue='variable_1', data=loss, linewidth=1, scale='width', bw=.5, cut=0, inner='stick', alpha=.5, palette=[mypal[0], mypal[2]])
h, l = p2.get_legend_handles_labels()
plt.legend(h[0:2], l[0:2])
plt.savefig('plots/loss_violin_stick.png')
plt.show()


p1 = sns.stripplot(x='variable_0', y='value', hue='variable_1', data=trans, dodge=True, jitter=False, linewidth=.75, size=1.75, palette=[mypal[0], mypal[3]])
p2 = sns.violinplot(x='variable_0', y='value', hue='variable_1', data=trans, linewidth=1, scale='width', bw=.75, cut=0, inner='stick', alpha=.5, palette=[mypal[0], mypal[3]])
h, l = p2.get_legend_handles_labels()
plt.legend(h[0:2], l[0:2])
plt.savefig('plots/trans_violin_stick.png')
plt.show()


# INNER=BOX
# p1 = sns.stripplot(x='variable_0', y='value', hue='variable_1', data=df, dodge=True, jitter=False, linewidth=.75, size=1.75)
p2 = sns.violinplot(x='variable_0', y='value', hue='variable_1', data=df, linewidth=1, scale='width', bw=.75, cut=0, inner='box', alpha=.5)
h, l = p2.get_legend_handles_labels()
plt.legend(h[0:4], l[0:4])
plt.savefig('plots/df_violin_box.png')
plt.show()

# p1 = sns.stripplot(x='variable_0', y='value', hue='variable_1', data=hy, dodge=True, jitter=False, linewidth=.75, size=1.75)
p2 = sns.violinplot(x='variable_0', y='value', hue='variable_1', data=hy, linewidth=1, scale='width', bw=.5, cut=0, inner='box', alpha=.5)
h, l = p2.get_legend_handles_labels()
plt.legend(h[0:2], l[0:2])
plt.savefig('plots/hy_violin_box.png')
plt.show()

# p1 = sns.stripplot(x='variable_0', y='value', hue='variable_1', data=loss, dodge=True, jitter=False, linewidth=.75, size=1.75, palette=[mypal[0], mypal[2]])
p2 = sns.violinplot(x='variable_0', y='value', hue='variable_1', data=loss, linewidth=1, scale='width', bw=.5, cut=0, inner='box', alpha=.5, palette=[mypal[0], mypal[2]])
h, l = p2.get_legend_handles_labels()
plt.legend(h[0:2], l[0:2])
plt.savefig('plots/loss_violin_box.png')
plt.show()


# p1 = sns.stripplot(x='variable_0', y='value', hue='variable_1', data=trans, dodge=True, jitter=False, linewidth=.75, size=1.75, palette=[mypal[0], mypal[3]])
p2 = sns.violinplot(x='variable_0', y='value', hue='variable_1', data=trans, linewidth=1, scale='width', bw=.75, cut=0, inner='box', alpha=.5, palette=[mypal[0], mypal[3]])
h, l = p2.get_legend_handles_labels()
plt.legend(h[0:2], l[0:2])
plt.savefig('plots/trans_violin_box.png')
plt.show()



sns.boxplot(x='variable_0', y='value', hue='variable_1', data=df, linewidth=1)
plt.savefig('plots/df_box.png')
plt.show()

sns.boxplot(x='variable_0', y='value', hue='variable_1', data=hy, linewidth=1)
plt.savefig('plots/hy_box.png')
plt.show()
sns.boxplot(x='variable_0', y='value', hue='variable_1', data=loss, linewidth=1, palette=[mypal[0], mypal[2]])
plt.savefig('plots/loss_box.png')
plt.show()
sns.boxplot(x='variable_0', y='value', hue='variable_1', data=trans, linewidth=1, palette=[mypal[0], mypal[3]])
plt.savefig('plots/trans_box.png')
plt.show()

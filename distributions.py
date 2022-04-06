import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('whitegrid')
from scipy import interpolate
from scipy import stats


def patch_violinplot(palette, alpha, n):
    from matplotlib.collections import PolyCollection
    ax = plt.gca()
    violins = [art for art in ax.get_children() if isinstance(art, PolyCollection)]
    colors = sns.color_palette(palette, n_colors=n) * (len(violins)//n)
    for i in range(len(violins)):
        violins[i].set_edgecolor(colors[i])
        violins[i].set_alpha(alpha)




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
print(df)
for i in szz:
    dd = df[df['variable_0'].isin([i])]
    pal = sns.color_palette('tab10')

    alpha = 0.7
    b = sns.boxplot(x='variable_1', y='value', data=dd, linewidth=.65, showfliers=False, palette=pal, width=0.3)
    # p1 = sns.stripplot(x='variable_1', y='value', data=dd, dodge=True, jitter=False, linewidth=.5, size=3, palette=pal, ec="black", alpha=alpha)
    # p2 = sns.violinplot(x='variable_1', y='value', data=dd, linewidth=.5, scale='width', bw=.75, cut=0, inner='box', palette=pal)
    # for p, box in enumerate(b.artists):
    #     r_, g_, b_, a_ = box.get_facecolor()
    #     # box.set_edgecolor((r_, g_, b_, .75))
    #     box.set_facecolor((r_, g_, b_, .75))
    #     box.set_edgecolor((0, 0, 0, .75))
    #     # box.set_facecolor((0, 0, 0, 0))
    #     print(p)
    #     for q in range(5*p, 5*(p+1)):
    #         b.lines[q].set_color('black')
    #         print('   ', q)
    #     # for q in range(20, 28):
    #     #     b.lines[q].set_alpha(0)

    # patch_violinplot(['black', 'black', 'black', 'black', 'black'], alpha, 4)


    plt.ylabel('Error')
    plt.title(f'Performance when trained on {i} datapoints')

    plt.savefig(f'plots/{i}plot.png')
    plt.show()




hy = df[df['variable_1'].isin(['baseline', 'hybrid'])]
loss = df[df['variable_1'].isin(['baseline', 'loss'])]
trans = df[df['variable_1'].isin(['baseline', 'transfer'])]

mypal = sns.color_palette()

sns.boxplot(x='variable_0', y='value', hue='variable_1', data=hy, linewidth=.65, showfliers=False, palette=pal, width=0.5)
# plt.ylim(-.00025, .002)
plt.ylabel('Error')
plt.xlabel('Training Data Points')
plt.savefig('plots/hy_box.png')
plt.show()
sns.boxplot(x='variable_0', y='value', hue='variable_1', data=loss, linewidth=.65, showfliers=False, palette=[pal[0], pal[2]], width=0.5)
# plt.ylim(-.00025, .002)
plt.ylabel('Error')
plt.xlabel('Training Data Points')
plt.savefig('plots/loss_box.png')
plt.show()
sns.boxplot(x='variable_0', y='value', hue='variable_1', data=trans, linewidth=.65, showfliers=False, palette=[pal[0], pal[3]], width=0.75)
# plt.ylim(-.00025, .002)
plt.ylabel('Error')
plt.xlabel('Training Data Points')
plt.savefig('plots/trans_box.png')
plt.show()

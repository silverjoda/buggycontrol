import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import numpy as np
import os

def plot_vanilla(data_list, min_len):
    sns.set_style("whitegrid", {'axes.grid' : True,
                                'axes.edgecolor':'black'
                                })
    fig = plt.figure()
    plt.clf()
    ax = fig.gca()
    colors = ["red", "black", "green", "blue", "purple",  "darkcyan", "brown", "darkblue",]
    labels = ["DQN", "DDQN","Maxmin", "EnsembleDQN", "MaxminDQN"]
    color_patch = []
    for color, label, data in zip(colors, labels, data_list):
        sns.tsplot(time=range(min_len), data=data, color=color, ci=95)
        color_patch.append(mpatches.Patch(color=color, label=label))
    print(min_len)
    plt.xlim([0, min_len])
    plt.xlabel('Training Episodes $(\\times10^6)$', fontsize=22)
    plt.ylabel('Average return', fontsize=22)
    lgd=plt.legend(
    frameon=True, fancybox=True, \
    prop={'weight':'bold', 'size':14}, handles=color_patch, loc="best")
    plt.title('Title', fontsize=14)
    ax = plt.gca()
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.set_xticklabels([0.5, 1, 1.5, 2.5, 3.0])

    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    sns.despine()
    plt.tight_layout()
    plt.show()

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/opt/logs/evaluations.npz")
data = np.load(path)
print(data.files)
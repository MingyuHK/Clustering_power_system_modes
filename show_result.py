# show results

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\\fonts\\times.ttf", size=20)

y_pred = np.loadtxt('y_pred.txt', dtype=int)
low_data = np.loadtxt('low_data.txt', dtype='float32', delimiter=' ')
# print(y_pred)
colors = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tan', 'grey', 'cyan', 'brown', 'peru', 'sage', 'teal', 'pink'])
# colors = np.array(['chocolate', 'blueviolet', 'limegreen', 'c', 'm', 'y', 'k', 'tan', 'grey', 'cyan', 'brown', 'peru', 'sage', 'teal', 'pink'])
markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
title = u'2021 (9%)'
x_date = np.arange(1, 366)
y_pred[y_pred == 0] = 10
y_pred[y_pred == 1] = 11
y_pred[y_pred == 2] = 12

y_pred[y_pred == 10] = 0
y_pred[y_pred == 12] = 1
y_pred[y_pred == 11] = 2

with plt.style.context('seaborn-white'):
    plt.figure()
    for j in range(len(low_data)):
        plt.scatter(low_data[j, 0], low_data[j, 1], c=colors[y_pred[j]], marker=markers[y_pred[j]],
                    edgecolors='k',
                    linewidths=1, alpha=1)
    plt.xlabel('First principal feature', fontproperties=font)
    plt.ylabel('Second principal feature', fontproperties=font)
    plt.title('2021', fontproperties=font)
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    plt.subplots_adjust(top=0.915, bottom=0.13, left=0.145, right=0.96, hspace=0.2, wspace=0.2)
    # plt.savefig(title + u'散点图', dpi=400)
    plt.savefig('C:\\Users\\eee\\Desktop\\fig\\point.pdf')

with plt.style.context('seaborn-white'):
    plt.figure()
    for j in range(len(x_date)):
        plt.scatter(x_date[j], int(y_pred[j]+1), c=colors[y_pred[j]], linewidths=0.1, alpha=1)
    plt.plot(x_date, y_pred+1, 'k', linewidth=0.5, alpha=1)
    plt.xlabel('Day', fontproperties=font)
    plt.ylabel('Cluster', fontproperties=font)
    # 设定刻度范围
    plt.yticks(np.arange(1, 7, 1))
    plt.title('2021', fontproperties=font)
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    plt.xlim((1, 365))
    plt.subplots_adjust(top=0.915, bottom=0.13, left=0.145, right=0.96, hspace=0.2, wspace=0.2)
    plt.savefig('C:\\Users\\eee\\Desktop\\fig\\time.pdf')
    # plt.savefig(title + u'时序图_cluster', dpi=400)


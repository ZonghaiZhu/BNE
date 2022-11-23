# coding:utf-8
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt

# 构建数据
map_color = {0: '#36556A', 1: '#1B798C', 2: '#009F9B', 3: '#4AC295', 4: '#9CE181',\
             5: '#F9F871'}
data_sets = ['cora', 'citeseer', 'pubmed', 'photo', 'computers']
data_dir = os.listdir('log')
imb_ratio = 5.0
bar_width = 0.3

for i, data_set in enumerate(data_sets):
    # searcg the related dir
    dirs = []
    for dir in data_dir:
        if dir.split('_')[2] == str(imb_ratio):
            dirs.append(dir)

    result_RNs = []
    result_BNEs= []
    for dir in dirs:
        log_dir = 'log/' + dir + '/result.txt'
        imb_ratio = dir.split('_')[2]

        with open(log_dir, 'r') as f:
            results = f.readlines()
            result_RN = np.float(results[1].rstrip('\n'))
            result_BNE = np.float(results[3].rstrip('\n'))

            result_RNs.append(result_RN)
            result_BNEs.append(result_BNE)


plt.figure(1, figsize=(12,10))
# plt.bar(x=np.arange(len(dirs)), height=result_BNEs, label='BNE',
#         color='coral', alpha=0.6, width=bar_width)
# plt.bar(x=np.arange(len(dirs)) + bar_width, height=result_RNs, label='ReNode',
#         color='royalblue', alpha=0.6, width=bar_width)

plt.bar(x=np.arange(len(dirs)), height=result_BNEs, label='BNE',
        color=map_color[1], alpha=1, width=bar_width)
plt.bar(x=np.arange(len(dirs)) + bar_width, height=result_RNs, label='ReNode',
        color=map_color[3], alpha=1, width=bar_width)

# # 柱状图显示数值，ha控制水平，va控制垂直
for x, y in enumerate(result_BNEs):
    plt.text(x, y, format(y, '.3f'), ha='center', va='bottom', fontsize=22)

for x, y in enumerate(result_RNs):
    plt.text(x + bar_width, y, format(y, '.3f'), ha='center', va='bottom', fontsize=22)

imb_ratios = ['imb ratio='+dir.split('_')[2] for dir in dirs]
plt.xticks(range(5), data_sets, fontsize=24, horizontalalignment='left')
# plt.xlabel(data_set)
plt.ylabel('Time (second)', fontsize=28)
plt.legend(loc='upper left', fontsize=20)
plt.grid(axis='y', linestyle='--')
plt.show()

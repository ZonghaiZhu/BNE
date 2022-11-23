# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt


data_sets = ['cora', 'citeseer', 'pubmed', 'photo', 'computers']
params = [0, 1, 1.5, 2, 2.5, 3]
bar_width = 0.5

for i, data_set in enumerate(data_sets):
    log_dir = data_set + '_step_10.0_results.csv'
    results = np.loadtxt(log_dir)

    Ks = []
    for k in params:
        idxs = np.where(results[:,2] == k)
        Ks.append(np.mean(results[idxs, -2]))

    plt.figure(i+1, figsize=(12,10))
    plt.bar(x=np.arange(len(Ks)), height=Ks, label='True',
            color='dodgerblue', alpha=0.6, width=bar_width)
    for x, y in enumerate(Ks):
        plt.text(x, y, format(y, '.3f'), ha='center', va='bottom', fontsize=18)

    plt.xticks(range(6), params, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Exploration Coefficient k', fontsize=22)
    plt.ylabel('Marco-F1 Score (%)', fontsize=22)
    # plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle='-.')

    a = 1
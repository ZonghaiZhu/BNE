# coding:utf-8
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt

data_sets = ['cora', 'citeseer', 'pubmed', 'photo', 'computers']
data_dirs = os.listdir('./')
colors = ['b-', 'b--', 'r-', 'r--']

for i, data_set in enumerate(data_sets):
    Train_results5 = []
    val_results5 = []
    Train_results10 = []
    val_results10 = []
    dirs = []
    for dir in data_dirs:
        if dir.split('_')[0] == data_set:
            if dir.split('_')[2] == '5.0':
                results = np.loadtxt(dir)
                Train_results5 = results[:, -2]
                val_results5 = results[:, 3]
            if dir.split('_')[2] == '10.0':
                results = np.loadtxt(dir)
                Train_results10 = results[:, -2]
                val_results10 = results[:, 3]

    x = np.arange(0, 3, 0.1)
    plt.figure(i + 1, figsize=(12, 10))
    plt.plot(x, Train_results5, colors[0], mec='k', label='Training_Imb_' + '5', lw=2)
    plt.plot(x, val_results5, colors[1], mec='k', label='validation_Imb_' + '5', lw=2)
    plt.plot(x, Train_results10, colors[2], mec='k', label='Training_Imb_' + '10', lw=2)
    plt.plot(x, val_results10, colors[3], mec='k', label='validation_Imb_' + '10', lw=2)

    plt.legend(loc='upper right', fontsize=20)
    plt.xlabel('Values of k', fontsize=28)
    plt.ylabel('Marco-F1 Score', fontsize=28)
    plt.grid(axis='y', linestyle='--')
    plt.show()
# coding:utf-8
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
import numpy as np
import random


def get_equal_split(args, all_label, num_classes, shuffle_seed):
    random.seed(shuffle_seed)
    n = len(all_label)
    np.random.seed(shuffle_seed)
    all_idx = np.random.permutation(n)
    shuffled_labels = all_label[all_idx]

    base_train_each = args.ntrain_div_classes
    base_valid_each = int(1.5*base_train_each)

    idx2train, idx2valid = {}, {}
    idx2test = []
    num4classes = []
    for i in range(num_classes):
        num_train = base_train_each
        num4classes.append(num_train)

        temp = list(np.where(shuffled_labels == i)[0])
        temp_idx = np.array(all_idx)[temp].tolist()
        idx2train[i] = temp_idx[:num_train]
        idx2valid[i] = temp_idx[num_train:(num_train + base_valid_each)]
        idx2test.append(temp_idx[(num_train + base_valid_each):])

    return idx2train, idx2valid, idx2test, num4classes


def get_random_split(args, all_label, num_classes, shuffle_seed):
    random.seed(shuffle_seed)
    n = len(all_label)

    base_train_each = args.ntrain_div_classes*num_classes  # num of valid samples in each class
    base_valid_each = int(1.5*base_train_each)

    rnd = np.random.permutation(n)

    train_idx = np.sort(rnd[:base_train_each])
    val_idx = np.sort(rnd[base_train_each:base_train_each + base_valid_each])

    train_val_idx = np.concatenate((train_idx, val_idx))
    test_idx = np.sort(np.setdiff1d(np.arange(n), train_val_idx))

    idx2train, idx2valid = {}, {}
    idx2test = []
    num4classes = []
    for i in range(num_classes):
        temp_trn = [j for j in train_idx if all_label[j] == i]
        num4classes.append(len(temp_trn))
        idx2train[i] = temp_trn
        temp_val = [j for j in val_idx if all_label[j] == i]
        idx2valid[i] = temp_val
        temp_tst = [j for j in test_idx if all_label[j] == i]
        idx2test.append(temp_tst)

    return idx2train, idx2valid, idx2test, num4classes


def explore_node(args, idx, adj, num_classes, num4classes):
    idx_train = {}
    max_num = int(max(num4classes)*args.ec)
    for i in range(num_classes):
        if num4classes[i] == max_num or num4classes[i] == 0:
            idx_train[i] = idx[i]
        else:
            temp = adj[idx[i]].sum(0).getA().reshape(-1)
            top_k_idx = list(temp.argsort()[::-1][0:max_num])
            idx_train[i] = list(set(top_k_idx + idx[i]))

    return idx_train


def explore_weights(idx, adj, num_classes, num4classes):
    node_weights = []
    max_num = max(num4classes)
    min_num = max(num4classes)
    imb_ratio = max_num/min_num
    for i in range(num_classes):
        temp = adj[idx[i]].sum(axis=0).getA()
        if num4classes[i] == max_num:
            node_weights.extend(temp*imb_ratio)
        else:
            node_weights.extend(temp)
    node_weights = np.array(node_weights)

    return node_weights.T


def load_processed_data(args, feat, adj, labels, shuffle_seed=0):
    print("\nLoading {} data with shuffle_seed {}".format(args.data_name, shuffle_seed))
    random.seed(shuffle_seed)

    # split the data into train/val/test
    num_classes = np.max(labels) + 1
    n_train = num_classes * args.ntrain_div_classes
    n_val = int(n_train * 1.5)

    if args.size_imb_type == 'equal':
        train_idx, val_idx, test_idx, num4classes = get_equal_split(args, labels, num_classes, shuffle_seed)
    elif args.size_imb_type == 'random':
        train_idx, val_idx, test_idx, num4classes = get_random_split(args, labels, num_classes, shuffle_seed)

    if args.ec == 0:
        explored_idx = train_idx
    else:
        explored_idx = explore_node(args, train_idx, adj, num_classes, num4classes)

    train_feats, train_labels = [], []
    valid_feats, valid_labels = [], []
    test_feats, test_labels = [], []
    for i in range(num_classes):
        train_feats.append(feat[explored_idx[i]])
        valid_feats.append(feat[val_idx[i]])
        test_feats.append(feat[test_idx[i]])

        train_labels.extend([i] * len(explored_idx[i]))
        valid_labels.extend([i] * len(val_idx[i]))
        test_labels.extend([i] * len(test_idx[i]))

    train_feats = torch.tensor(np.concatenate(train_feats), dtype=torch.float32)
    valid_feats = torch.tensor(np.concatenate(valid_feats), dtype=torch.float32)
    test_feats = torch.tensor(np.concatenate(test_feats), dtype=torch.float32)

    return train_feats, valid_feats, test_feats, \
           torch.tensor(train_labels), torch.tensor(valid_labels), torch.tensor(test_labels)
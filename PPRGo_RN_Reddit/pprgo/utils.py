# import resource
import numpy as np
import scipy.sparse as sp
import sklearn
import time, json, random


def get_equal_split(ntrain_div_classes, all_label, num_classes, shuffle_seed):
    random.seed(shuffle_seed)
    n = len(all_label)
    np.random.seed(shuffle_seed)
    all_idx = np.random.permutation(n)
    shuffled_labels = all_label[all_idx]

    base_train_each = ntrain_div_classes
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

    train_idx, valid_idx, test_idx = [], [], []
    for i in range(num_classes):
        train_idx.extend(idx2train[i])
        valid_idx.extend(idx2valid[i])
        test_idx.extend(idx2test[i])

    train_idx = np.sort(train_idx)
    valid_idx = np.sort(valid_idx)
    test_idx = np.sort(test_idx)

    return train_idx, valid_idx, test_idx


def get_random_split(ntrain_div_classes, all_label, num_classes, shuffle_seed):
    random.seed(shuffle_seed)
    n = len(all_label)

    base_train_each = ntrain_div_classes*num_classes  # num of valid samples in each class
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

    train_idx, valid_idx, test_idx = [], [], []
    for i in range(num_classes):
        train_idx.extend(idx2train[i])
        valid_idx.extend(idx2valid[i])
        test_idx.extend(idx2test[i])

    train_idx = np.sort(train_idx)
    valid_idx = np.sort(valid_idx)
    test_idx = np.sort(test_idx)

    return train_idx, valid_idx, test_idx


def get_data(dataset_path, seed, ntrain_div_classes, ratio, issue_type='qinl'):

    adj_full = sp.load_npz('{}/adj_full.npz'.format(dataset_path)).astype(np.int)
    # adj_full = adj_full + sp.eye(adj_full.shape[0])
    feats = np.load('{}/feats.npy'.format(dataset_path))
    class_map = json.load(open('{}/class_map.json'.format(dataset_path)))
    class_map = {int(k): v for k, v in class_map.items()}
    labels = np.array([class_map[i] for i in range(len(class_map))])

    # split the data into train/val/test
    num_classes = labels.max() + 1

    if issue_type == 'random':
        train_idx, val_idx, test_idx = get_random_split(ntrain_div_classes, labels, num_classes, seed)
    elif issue_type == 'equal':
        train_idx, val_idx, test_idx = get_equal_split(ntrain_div_classes, labels, num_classes, seed)

    return adj_full, feats, labels, train_idx, val_idx, test_idx


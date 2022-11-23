# coding:utf-8
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
import numpy as np
import random
import scipy.sparse as sp


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def get_equal_split(args, all_idx, all_label, num_classes, shuffle_seed):
    base_valid_each = args.valid_each
    base_train_each = args.train_each

    shuffled_labels = all_label[all_idx]

    idx2train, idx2valid = {}, {}
    idx2test = []
    num4classes = []
    for i in range(num_classes):
        num_train = base_train_each
        num4classes.append(num_train)

        temp = list(np.where(shuffled_labels == i)[0])
        temp_idx = np.array(all_idx)[temp]
        idx2train[i] = temp_idx[:num_train].tolist()
        idx2valid[i] = temp_idx[num_train:(num_train + base_valid_each)].tolist()
        idx2test.append(temp_idx[(num_train + base_valid_each):].tolist())

    return idx2train, idx2valid, idx2test, num4classes


def get_step_split(args, all_idx, all_label, num_classes, shuffle_seed):
    base_valid_each = args.valid_each  # num of valid samples in each class
    imb_ratio = args.imb_ratio
    head_list = args.head_list if len(args.head_list) > 0 else [i for i in range(num_classes // 2)]

    shuffled_labels = all_label[all_idx]

    all_class_list = [i for i in range(num_classes)]
    tail_list = list(set(all_class_list) - set(head_list))
    h_num = len(head_list)  # head classes num
    t_num = len(tail_list)  # tail classes num

    base_train_each = int(len(all_idx) * args.labeling_ratio / (t_num + imb_ratio * h_num))
    idx2train, idx2valid = {}, {}
    idx2test = []
    num4classes = []
    for i in range(num_classes):
        if i in head_list:
            num_train = int(base_train_each * imb_ratio)
        else:
            num_train = int(base_train_each * 1)
        num4classes.append(num_train)

        temp = list(np.where(shuffled_labels == i)[0])
        temp_idx = np.array(all_idx)[temp]
        idx2train[i] = temp_idx[:num_train].tolist()
        idx2valid[i] = temp_idx[num_train:(num_train + base_valid_each)].tolist()
        idx2test.append(temp_idx[(num_train + base_valid_each):].tolist())

    return idx2train, idx2valid, idx2test, num4classes


def get_adj_feats(target_data):
    adj0 = sp.coo_matrix((np.ones(target_data.edge_index.shape[1]),
                         (target_data.edge_index[0, :], target_data.edge_index[1, :])),
                        shape=(target_data.num_nodes, target_data.num_nodes), dtype=np.float32)

    adj = adj0 + sp.eye(adj0.shape[0]) # add self loop, and coo_matrix ->(change to) csr matrix
    norm_adj = normalize_adj(adj)
    norm_adj2 = norm_adj @ norm_adj # coo_matrix @ coo_matrix -> (change to) csr_matrix
    feat0 = target_data.x
    # feat1 = norm_adj @ feat0
    feat2 = norm_adj2 @ feat0
    # feat = np.concatenate((feat0, feat1, feat2), axis=1)
    feat = feat2
    return feat, norm_adj2


def explore_node(args, idx, adj, all_idx, num_classes, num4classes):
    exp_idx = {}
    exp_unlabel_idx = {}
    unlabel_idx = all_idx
    relabel_idx = []
    max_num = int(max(num4classes)*args.ec)
    train_num = 0
    for i in range(num_classes):
        if num4classes[i] == max_num:
            exp_idx[i] = idx[i]
        else:
            temp = adj[idx[i]].sum(0).getA().reshape(-1)
            top_k_idx = list(temp.argsort()[::-1][0:max_num])
            exp_idx[i] = list(set(top_k_idx))
        train_num += len(idx[i])

    # for i in range(num_classes):
    #     exp_idx[i] = list(set(exp_idx[i]) - set(idx[i]))

    for i in range(num_classes):
        exp_unlabel_idx[i] = list(set(exp_idx[i]) - set(idx[i]) - set(relabel_idx))
        unlabel_idx = list(set(unlabel_idx) - set(idx[i]))
        relabel_idx = list(set(relabel_idx) | set(exp_unlabel_idx[i]))
    # exp_ratio denotes the ratio that explored unlabeled nodes occupies in all unlabeled nodes
    exp_num = sum([len(exp_unlabel_idx[i]) for i in range(num_classes)])
    # exp_ratio = exp_num/len(unlabel_idx)
    exp_ratio = (exp_num + train_num)/train_num
    return exp_idx, exp_num, exp_ratio


def load_processed_data(args, data_path, data_name, shuffle_seed=0):
    print("\nLoading {} data with shuffle_seed {}".format(data_name, shuffle_seed))
    random.seed(shuffle_seed)

    data_dict = {'cora': 'planetoid', 'citeseer': 'planetoid', 'pubmed': 'planetoid',
                 'photo': 'amazon', 'computers': 'amazon'}
    target_type = data_dict[data_name]
    if target_type == 'planetoid':
        target_dataset = Planetoid(data_path, name=data_name)
    elif target_type == 'amazon':
        target_dataset = Amazon(data_path, name=data_name)
    target_data = target_dataset[0]
    all_labels = target_data.y.numpy()
    num_classes = np.max(all_labels) + 1

    feat, adj = get_adj_feats(target_data)
    all_idx = [i for i in range(target_data.num_nodes)]
    random.shuffle(all_idx)

    if args.size_imb_type == 'equal':
        train_idx, val_idx, test_idx, num4classes = get_equal_split(args, all_idx, all_labels, num_classes, shuffle_seed)
    elif args.size_imb_type == 'step':
        train_idx, val_idx, test_idx, num4classes = get_step_split(args, all_idx, all_labels, num_classes, shuffle_seed)

    if args.ec == 0:
        explored_idx = train_idx
    else:
        explored_idx, exp_num, exp_ratio = explore_node(args, train_idx, adj, all_idx, num_classes, num4classes)

    exp_labels, exp_real_labels = [], []
    for i in range(num_classes):
        exp_labels.extend([i]*len(explored_idx[i])) # must assign new label, (labels[explored_idx] leak labels)
        exp_real_labels.extend(all_labels[explored_idx[i]])

    return exp_labels, exp_real_labels, exp_num, exp_ratio
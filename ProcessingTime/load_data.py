# coding:utf-8
import random
import math
import os,sys, time
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import numpy as np
from utils import index2dense
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def get_feats(target_data):
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
    return feat

def get_equal_split_BNE(args, all_idx, all_label, num_classes):
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


def get_step_split_BNE(args, all_idx, all_label, num_classes):
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


def get_equal_split(args, all_idx, all_label, nclass):
    base_valid_each = args.valid_each
    base_train_each = args.train_each

    shuffled_labels = all_label[all_idx]

    idx2train, idx2valid = {}, {}
    idx2test = []
    num4classes = []
    for i in range(nclass):
        num_train = base_train_each
        num4classes.append(num_train)

        temp = list(np.where(shuffled_labels == i)[0])
        temp_idx = np.array(all_idx)[temp]
        idx2train[i] = temp_idx[:num_train].tolist()
        idx2valid[i] = temp_idx[num_train:(num_train + base_valid_each)].tolist()
        idx2test.append(temp_idx[(num_train + base_valid_each):].tolist())

    train_node = []
    train_idx, valid_idx, test_idx = [], [], []
    for i in range(nclass):
        train_idx.extend(idx2train[i])
        valid_idx.extend(idx2valid[i])
        test_idx.extend(idx2test[i])
        train_node.append(idx2train[i])

    return train_idx, valid_idx, test_idx, train_node


def get_step_split(args, all_idx, all_label, nclass):
    base_valid_each = args.valid_each
    imb_ratio = args.imb_ratio
    head_list = args.head_list if len(args.head_list) > 0 else [i for i in range(nclass // 2)]

    shuffled_labels = all_label[all_idx]

    all_class_list = [i for i in range(nclass)]
    tail_list = list(set(all_class_list) - set(head_list))
    h_num = len(head_list)  # head classes num
    t_num = len(tail_list)  # tail classes num

    base_train_each = int(len(all_idx) * args.labeling_ratio / (t_num + imb_ratio * h_num))
    idx2train, idx2valid = {}, {}
    idx2test = []
    num4classes = []
    for i in range(nclass):
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

    train_node = []
    train_idx, valid_idx, test_idx = [], [], []
    for i in range(nclass):
        train_idx.extend(idx2train[i])
        valid_idx.extend(idx2valid[i])
        test_idx.extend(idx2test[i])
        train_node.append(idx2train[i])

    return train_idx, valid_idx, test_idx, train_node


# return the ReNode Weight
def get_renode_weight(opt, data):
    ppr_matrix = data.Pi  # personlized pagerank
    gpr_matrix = torch.tensor(data.gpr).float()

    base_w = opt.rn_base_weight
    scale_w = opt.rn_scale_weight
    nnode = ppr_matrix.size(0)
    unlabel_mask = data.train_mask.int().ne(1)

    # computing the Totoro values for labeled nodes
    gpr_sum = torch.sum(gpr_matrix, dim=1)
    gpr_rn = gpr_sum.unsqueeze(1) - gpr_matrix
    rn_matrix = torch.mm(ppr_matrix, gpr_rn)

    label_matrix = F.one_hot(data.y, gpr_matrix.size(1)).float()
    label_matrix[unlabel_mask] = 0

    rn_matrix = torch.sum(rn_matrix * label_matrix, dim=1)
    rn_matrix[unlabel_mask] = rn_matrix.max() + 99

    # computing the ReNode Weight
    train_size = torch.sum(data.train_mask.int()).item()
    totoro_list = rn_matrix.tolist()
    id2totoro = {i: totoro_list[i] for i in range(len(totoro_list))}
    sorted_totoro = sorted(id2totoro.items(), key=lambda x: x[1], reverse=False)
    id2rank = {sorted_totoro[i][0]: i for i in range(nnode)}
    totoro_rank = [id2rank[i] for i in range(nnode)]

    rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x * 1.0 * math.pi / (train_size - 1)))) for x in totoro_rank]
    rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor)
    rn_weight = rn_weight * data.train_mask.float()

    return rn_weight


# loading the processed data
def load_processed_data(args, data_path, data_name, shuffle_seed=0, ppr_file=''):
    print("\nLoading {} data with shuffle_seed {}".format(data_name, shuffle_seed))

    data_dict = {'cora': 'planetoid', 'citeseer': 'planetoid', 'pubmed': 'planetoid',
                 'photo': 'amazon', 'computers': 'amazon'}

    target_type = data_dict[data_name]
    if target_type == 'planetoid':
        target_dataset = Planetoid(data_path, name=data_name)
    elif target_type == 'amazon':
        target_dataset = Amazon(data_path, name=data_name)
    target_data = target_dataset[0]
    target_data.num_classes = np.max(target_data.y.numpy())+1
    feats = get_feats(target_data)

    # the random seed for the dataset splitting
    mask_list = [i for i in range(target_data.num_nodes)]
    random.seed(shuffle_seed)
    random.shuffle(mask_list)

    if args.size_imb_type == 'equal':
        train_mask_list, valid_mask_list, test_mask_list, target_data.train_node = \
            get_equal_split(args, mask_list, target_data.y.numpy(), nclass=target_data.num_classes)
    elif args.size_imb_type == 'step':
        train_mask_list, valid_mask_list, test_mask_list, target_data.train_node = \
            get_step_split(args, mask_list, target_data.y.numpy(), nclass=target_data.num_classes)

    target_data.train_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)
    target_data.valid_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)
    target_data.test_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)

    target_data.train_mask_list = train_mask_list

    target_data.train_mask[torch.tensor(train_mask_list).long()] = True
    target_data.valid_mask[torch.tensor(valid_mask_list).long()] = True
    target_data.test_mask[torch.tensor(test_mask_list).long()] = True

    # calculating the Personalized PageRank Matrix if not exists.
    start_time = time.time()
    pr_prob = 1 - args.pagerank_prob
    A = index2dense(target_data.edge_index, target_data.num_nodes)
    A_hat = A.to(args.device) + torch.eye(A.size(0)).to(args.device)  # add self-loop
    D = torch.diag(torch.sum(A_hat, 1))
    D = D.inverse().sqrt()
    A_hat = torch.mm(torch.mm(D, A_hat), D)
    target_data.Pi = pr_prob * (
        (torch.eye(A.size(0)).to(args.device) - (1 - pr_prob) * A_hat).inverse())
    target_data.Pi = target_data.Pi.cpu()

        # calculating the ReNode Weight
    gpr_matrix = []  # the class-level influence distribution
    for iter_c in range(target_data.num_classes):
        iter_Pi = target_data.Pi[torch.tensor(target_data.train_node[iter_c]).long()]
        iter_gpr = torch.mean(iter_Pi, dim=0).squeeze()
        gpr_matrix.append(iter_gpr)

    temp_gpr = torch.stack(gpr_matrix, dim=0)
    temp_gpr = temp_gpr.transpose(0, 1)
    target_data.gpr = temp_gpr

    target_data.rn_weight = get_renode_weight(args, target_data)
    end_time = time.time() - start_time

    return end_time


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


def explore_node(args, idx, adj, num_classes, num4classes):
    idx_train = {}
    max_num = int(max(num4classes)*args.ec)
    for i in range(num_classes):
        if num4classes[i] == max_num:
            idx_train[i] = idx[i]
        else:
            temp = adj[idx[i]].sum(0).getA().reshape(-1)
            top_k_idx = list(temp.argsort()[::-1][0:max_num])
            idx_train[i] = list(set(top_k_idx + idx[i]))

    return idx_train


def load_BNE_data(args, data_path, data_name, shuffle_seed=0):
    args.ec = 2
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
        train_idx, val_idx, test_idx, num4classes = get_equal_split_BNE(args, all_idx, all_labels, num_classes)
    elif args.size_imb_type == 'step':
        train_idx, val_idx, test_idx, num4classes = get_step_split_BNE(args, all_idx, all_labels, num_classes)

    start_time = time.time()
    if args.ec == 0:
        explored_idx = train_idx
    else:
        explored_idx = explore_node(args, train_idx, adj, num_classes, num4classes)
    # node_weights[explored_idx[0~6]].argmax(1) are approximately right
    # node_weights = explore_weights(train_idx, adj, num_classes, num4classes)

    train_feats, train_labels = [], []
    valid_feats, valid_labels = [], []
    test_feats, test_labels = [], []
    for i in range(num_classes):
        train_feats.append(feat[explored_idx[i]])
        valid_feats.append(feat[val_idx[i]])
        test_feats.append(feat[test_idx[i]])

        train_labels.extend([i]*len(explored_idx[i])) # must assign new label, (labels[explored_idx] leak labels)
        valid_labels.extend([i]*len(val_idx[i]))
        test_labels.extend([i]*len(test_idx[i]))

    train_feats = np.concatenate(train_feats)
    valid_feats = np.concatenate(valid_feats)
    test_feats = np.concatenate(test_feats)
    end_time = time.time() - start_time

    return end_time


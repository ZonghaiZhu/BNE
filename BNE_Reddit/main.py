# coding:utf-8
import os, time
import argparse
import numpy as np
import utils
from model import Model
from load_data import load_processed_data
import scipy.sparse as sp


parser = argparse.ArgumentParser()
# NN
parser.add_argument('--num_hidden', default=256, type=int)
parser.add_argument('--num_layer', default=2, type=int)

# Dataset
parser.add_argument('--data_path', default='../../DataSet/', type=str, help="data path (dictionary)")
parser.add_argument('--data_name', default='reddit', type=str, help="data name")  # reddit mag_coarse
parser.add_argument('--size_imb_type', default='random', type=str,
                    help="the imbalace type of the training set")  # equal random
parser.add_argument('--ntrain_div_classes', default=100, type=int,
                    help="the training size of each class, used in none imbe type") # 20, 50, 100
parser.add_argument('--valid_each', default=30, type=int, help="the validation size of each class")
parser.add_argument('--labeling_ratio', default=0.05, type=float,
                    help="the labeling ratio of the dataset, used in step imb type")
parser.add_argument('--head_list', default=[], type=int, nargs='+',
                    help="list of the majority class, used in step imb type")
parser.add_argument('--imb_ratio', default=5.0, type=float,
                    help="the ratio of the majority class size to the minoriry class size, used in step imb type")

# Training
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--lr_decay_epoch', default=20, type=int)
parser.add_argument('--lr_decay_rate', default=0.95, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--epoch', default=800, type=int)
parser.add_argument('--least_epoch', default=40, type=int)
parser.add_argument('--early_stop', default=30, type=int)
parser.add_argument('--print_freq', default=10, type=int)

parser.add_argument('--run_split_num', default=5, type=int, help='run N different split times')
parser.add_argument('--run_init_num', default=3, type=int, help='run N different init seeds')
parser.add_argument('--device', default='cuda', type=str, help='use GPU.')


def main(args, feats, adj, targets):
    print('\nSetting environment...')
    args.shuffle_seed_list = [i for i in range(args.run_split_num)]
    args.seed_list = [i for i in range(args.run_init_num)]

    if not os.path.exists('log'):
        os.makedirs('log')
    log_dir = args.data_name + '_' + str(args.imb_ratio) + '_' \
              + args.size_imb_type + '_' + time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join('log', log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    utils.configure_output_dir(log_dir)
    hyperparams = dict(args._get_kwargs())
    utils.save_hyperparams(hyperparams)

    # opt.run_split_num experiments
    run_time_val_result_weighted = [[] for _ in range(args.run_split_num)]
    run_time_val_result_macro = [[] for _ in range(args.run_split_num)]
    run_time_result_weighted = [[] for _ in range(args.run_split_num)]
    run_time_result_macro = [[] for _ in range(args.run_split_num)]

    for iter_split_seed in range(args.run_split_num):
        print('The [%d] / [%d] dataset spliting...' % (iter_split_seed + 1, args.run_split_num))

        # prepare related data
        print("=> preparing data sets: {}, imablanced ratio: {}, type: {}"
              .format(args.data_name, args.imb_ratio, args.size_imb_type))

        train_feats, valid_feats, test_feats, train_labels, valid_labels, test_labels = \
            load_processed_data(args, feats, adj, targets,
                            shuffle_seed=args.shuffle_seed_list[iter_split_seed])

        for iter_init_seed in range(args.run_init_num):
            print('--[%d] / [%d] seed...' % (iter_init_seed + 1, args.run_init_num))

            # set the seed for training initial
            utils.set_seed(args.seed_list[iter_init_seed], cuda=True)

            # initialize model
            print('\nSetting model...')
            model = Model(args, d_in=train_feats.shape[1], d_hid=args.num_hidden, d_out=train_labels.max().item()+1)

            print('\nTraining begining...')
            val_w_f1, val_m_f1 = model.fit(train_feats, valid_feats, train_labels, valid_labels)

            print('\nTesting begining..')
            weighted_f1, macro_f1 = model.predict(test_feats, test_labels, testing='True')

            run_time_val_result_weighted[iter_split_seed].append(val_w_f1)
            run_time_val_result_macro[iter_split_seed].append(val_m_f1)
            run_time_result_weighted[iter_split_seed].append(weighted_f1)
            run_time_result_macro[iter_split_seed].append(macro_f1)

    val_weighted_np = np.array(run_time_val_result_weighted)
    val_macro_np = np.array(run_time_val_result_macro)

    weighted_np = np.array(run_time_result_weighted)
    macro_np = np.array(run_time_result_macro)

    return val_weighted_np, val_macro_np, weighted_np, macro_np


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()


if __name__ == '__main__':
    args = parser.parse_args()
    args.data_path = args.data_path + args.data_name

    adj_full, feats, targets = utils.load_data(args.data_path, args.data_name)
    adj = normalize_adj(adj_full)
    feats = adj @ feats

    # hyperparameters: lr, explore coeff (ec), weight_decay (wd)
    # lrs = [0.01, 0.001, 0.0001]
    lrs = [0.001]
    wds = [1e-5]
    if args.size_imb_type == 'random':
        ecs = [0, 1, 1.5, 2, 2.5, 3]  # ec==0, use original imbalanced data
    elif args.size_imb_type == 'equal':
        ecs = [1, 1.5, 2, 2.5, 3]
    results = []
    for lr in lrs:
        for wd in wds:
            for ec in ecs:
                args.lr = lr
                args.weight_decay = wd
                args.ec = ec
                val_weighted_np, val_macro_np, weighted_np, macro_np = main(args, feats, adj, targets)
                print('\nThe overall performance:')
                val_weighted_mean = round(np.mean(val_weighted_np), 4)
                val_weighted_std = round(np.std(val_weighted_np), 4)

                val_macro_mean = round(np.mean(val_macro_np), 4)
                val_macro_std = round(np.std(val_macro_np), 4)

                weighted_mean = round(np.mean(weighted_np), 4)
                weighted_std = round(np.std(weighted_np), 4)

                macro_mean = round(np.mean(macro_np), 4)
                macro_std = round(np.std(macro_np), 4)

                temp = np.array([lr, wd, ec, val_macro_mean, val_macro_std,
                                 weighted_mean, weighted_std, macro_mean, macro_std])
                results.append(temp)

    temp_results = np.array(results, dtype=np.float32)
    if args.size_imb_type == 'equal':
        save_name = args.size_imb_type + '_' + str(args.ntrain_div_classes)
    elif args.size_imb_type == 'random':
        save_name = args.size_imb_type + '_' + str(args.ntrain_div_classes)
    elif args.size_imb_type == 'ratio':
        save_name = args.size_imb_type + '_' + str(args.ratio_per_class)
    np.savetxt(save_name + '_' + 'results.csv', temp_results, fmt='%.05f')





# coding:utf-8
# coding:utf-8
import os, time
import argparse
import numpy as np
import utils
from load_data import load_processed_data
from sklearn.metrics import f1_score, accuracy_score

parser = argparse.ArgumentParser()
# NN
parser.add_argument('--num_hidden', default=1024, type=int)
parser.add_argument('--num_layer', default=2, type=int)

# Dataset
parser.add_argument('--data_path', default='../../DataSet/', type=str, help="data path (dictionary)")
parser.add_argument('--data_name', default='cora', type=str, help="data name")  # cora citeseer pubmed photo computers
parser.add_argument('--size_imb_type', default='step', type=str,
                    help="the imbalace type of the training set")  # equal step
parser.add_argument('--train_each', default=20, type=int,
                    help="the training size of each class, used in none imbe type")
parser.add_argument('--valid_each', default=30, type=int, help="the validation size of each class")
parser.add_argument('--labeling_ratio', default=0.05, type=float,
                    help="the labeling ratio of the dataset, used in step imb type")
parser.add_argument('--head_list', default=[], type=int, nargs='+',
                    help="list of the majority class, used in step imb type")
parser.add_argument('--imb_ratio', default=10.0, type=float,
                    help="the ratio of the majority class size to the minoriry class size, used in step imb type") # 5.0 10.0

# Training
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--lr_decay_epoch', default=20, type=int)
parser.add_argument('--lr_decay_rate', default=0.95, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--least_epoch', default=40, type=int)
parser.add_argument('--early_stop', default=30, type=int)
parser.add_argument('--print_freq', default=10, type=int)

parser.add_argument('--run_split_num', default=5, type=int, help='run N different split times')
parser.add_argument('--run_init_num', default=3, type=int, help='run N different init seeds')
parser.add_argument('--device', default='cuda', type=str, help='use GPU.')


def main(args):
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
    m_f1s = []
    accs = []
    exp_nums = []
    ratios = []
    for iter_split_seed in range(args.run_split_num):
        print('The [%d] / [%d] dataset spliting...' % (iter_split_seed + 1, args.run_split_num))

        # prepare related data
        print("=> preparing data sets: {}, imablanced ratio: {}, type: {}"
              .format(args.data_name, args.imb_ratio, args.size_imb_type))

        exp_labels, exp_real_labels, exp_num, exp_ratio = load_processed_data(args, args.data_path, args.data_name,
                            shuffle_seed=args.shuffle_seed_list[iter_split_seed])

        # w_f1 = f1_score(exp_real_labels, exp_labels, average='weighted')
        # m_f1 = f1_score(exp_real_labels, exp_labels, average='macro')
        acc = accuracy_score(exp_real_labels, exp_labels)

        accs.append(acc)
        exp_nums.append(exp_num)
        ratios.append(exp_ratio)

    return accs, exp_nums, ratios


if __name__ == '__main__':
    args = parser.parse_args()
    args.data_path = args.data_path + args.data_name # amazon data need args.name to create the doc
    if args.size_imb_type == 'step':
        ecs = [1, 1.5, 2, 2.5, 3] # ec==0, use original imbalanced data
    else:
        ecs = [1, 1.5, 2, 2.5, 3]
    results = []

    for ec in ecs:
        args.ec = ec
        accs, exp_nums, ratios = main(args)
        print('\nThe overall performance:')

        acc_mean = round(np.mean(accs), 4)
        acc_std = round(np.std(accs), 4)

        num_mean = round(np.mean(exp_nums), 4)
        num_std = round(np.std(exp_nums), 4)

        ratio_mean = round(np.mean(ratios), 4)
        ratio_std = round(np.std(ratios), 4)

        temp = np.array([ec, acc_mean, acc_std, num_mean, num_std, ratio_mean, ratio_std])
        results.append(temp)

    temp_results = np.array(results, dtype=np.float32)
    if args.size_imb_type == 'step':
        save_name = args.data_name + '_' + args.size_imb_type + '_' + str(args.imb_ratio)
    else:
        save_name = args.data_name + '_' + args.size_imb_type
    np.savetxt(save_name + '_'+'results.csv', temp_results, fmt='%.05f')
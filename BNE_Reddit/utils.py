# coding:utf-8
import os, json, atexit, time, torch, random
import numpy as np
import matplotlib
import scipy.sparse as sp

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class G:
    output_dir = None
    output_file = None
    first_row = True
    log_headers = []
    log_current_row = {}


def configure_output_dir(dir=None):
    G.output_dir = dir
    G.output_file = open(os.path.join(G.output_dir, "log.txt"), 'w')
    atexit.register(G.output_file.close)
    print("Logging data to %s" % G.output_file.name)


def save_hyperparams(params):
    with open(os.path.join(G.output_dir, "hyperparams.json"), 'w') as out:
        out.write(json.dumps(params, separators=(',\n', '\t:\t'), sort_keys=True))


def log_tabular(key, val):
    if G.first_row:
        G.log_headers.append(key)
    else:
        assert key in G.log_headers
    assert key not in G.log_current_row
    G.log_current_row[key] = val


def dump_tabular():
    vals = []
    for key in G.log_headers:
        val = G.log_current_row.get(key, "")
        vals.append(val)
    if G.output_dir is not None:
        if G.first_row:
            G.output_file.write("\t".join(G.log_headers))
            G.output_file.write("\n")
        G.output_file.write("\t".join(map(str, vals)))
        G.output_file.write("\n")
        G.output_file.flush()
    G.log_current_row.clear()
    G.first_row = False


def save_pytorch_model(model):
    """
    Saves the entire pytorch Module
    """
    torch.save(model, os.path.join(G.output_dir, "model.pkl"))


def load_pytorch_model(model):
    """
    Saves the entire pytorch Module
    """
    temp = torch.load('model.pkl')
    model.resnet.load_state_dict(temp.resnet.state_dict())
    model.classifier.load_state_dict(temp.classifier.state_dict())


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def index2dense(edge_index, nnode=2708):
    indx = edge_index.numpy()
    adj = np.zeros((nnode, nnode), dtype='int8')
    adj[(indx[0], indx[1])] = 1
    new_adj = torch.from_numpy(adj).float()
    return new_adj


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if cuda: torch.cuda.manual_seed(seed)


def load_data(prefix, data_name, normalize=True):
    '''
        Inputs:
            prefix              string, directory containing the above graph related files
            normalize           bool, whether or not to normalize the node features
    '''

    adj_full = sp.load_npz('{}/adj_full.npz'.format(prefix)).astype(np.int)
    adj_full = adj_full + sp.eye(adj_full.shape[0])
    feats = np.load('{}/feats.npy'.format(prefix))
    class_map = json.load(open('{}/class_map.json'.format(prefix)))
    class_map = {int(k): v for k, v in class_map.items()}
    labels = np.array([class_map[i] for i in range(len(class_map))])
    # labels = np.array([v for k, v in class_map.items()])

    return adj_full, feats, labels
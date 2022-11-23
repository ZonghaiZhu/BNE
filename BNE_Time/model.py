# coding:utf-8
import torch, time, copy
import numpy as np
from network import Net
import utils
from sklearn.metrics import f1_score

class Model():
    def __init__(self, args, d_in, d_hid, d_out):
        self.args = args
        self.d_in = d_in
        self.d_hid = d_hid
        self.classes = d_out

        self.print_freq = args.print_freq
        self.device = args.device
        self.net = Net(self.d_in, self.d_hid, self.classes).to(self.device)
        self.lr = args.lr
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          self.lr, weight_decay=args.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()

    def fit(self, train_feats, valid_feats, train_labels, valid_labels):
        best_val_m_f1 = 0
        best_epoch = 0

        trn_x = train_feats.to(self.device)
        trn_y = train_labels.to(self.device)

        val_x = valid_feats.to(self.device)
        val_y = valid_labels

        reduce_times = 0
        for epoch in range(self.args.epoch):
            # self.adjust_learning_rate(epoch)
            # switch to train mode
            self.net.train()
            start_time = time.time()
            if epoch > self.args.lr_decay_epoch:
                new_lr = self.args.lr * pow(self.args.lr_decay_rate, (epoch - self.args.lr_decay_epoch))
                new_lr = max(new_lr, 1e-4)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr

            # compute outputs
            outputs = self.net(trn_x)
            loss = self.criterion(outputs, trn_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_time = time.time() - start_time

            # validation process
            val_w_f1, val_m_f1 = self.predict(val_x, val_y)
            if val_m_f1 > best_val_m_f1:
                best_net = copy.deepcopy(self.net)
                best_val_m_f1 = val_m_f1 # m_f1 is the most important
                best_val_w_f1 = val_w_f1
                best_epoch = epoch
            else:
                if epoch >= self.args.least_epoch and epoch - best_epoch > self.args.early_stop:
                    print('\nEarly stop at %d epoch. The best is in %d epoch' %
                          (epoch, best_epoch))
                    self.net = best_net
                    break

            if epoch % self.args.print_freq == 0:
                pred = outputs.cpu().max(1)[1].numpy()
                acc_trn = (pred == np.array(train_labels)).sum() / len(pred)
                print(('Epoch:[{}/{}], Epoch_time:{:.3f}\t'
                       'Train_Accuracy:{:.4f}, Train_Loss:{:.4f}\t'
                       'Best_m_f1:{:.3f}, Val_m_f1:{:.3f}').format(
                    epoch+1, self.args.epoch, epoch_time, acc_trn, loss.item(), best_val_m_f1, val_m_f1)
                )

            utils.log_tabular("Epoch", epoch)
            utils.log_tabular("Training_time", epoch_time)
            utils.log_tabular("Train_Loss", loss.item())
            utils.log_tabular("Val_Weighted_F1_macro", val_w_f1)
            utils.log_tabular("Val_F1_macro", val_m_f1)
            utils.dump_tabular()

        return best_val_w_f1, best_val_m_f1

    def predict(self, x, y):
        self.net.eval()

        out = self.net(x.to(self.device))
        pred = out.cpu().max(1)[1].numpy()

        w_f1 = f1_score(y, pred, average='weighted')
        m_f1 = f1_score(y, pred, average='macro')

        return w_f1, m_f1

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        epoch = epoch + 1
        if epoch <= 5:
            lr = self.lr * epoch / 5
        elif epoch > 80:
            lr = self.lr * 0.001
        elif epoch > 60:
            lr = self.lr * 0.01
        elif epoch > 40:
            lr = self.lr * 0.1
        else:
            lr = self.lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


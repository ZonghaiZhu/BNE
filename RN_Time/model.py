# coding:utf-8
import torch, copy, time
import utils
from network import Net

from imb_loss import IMB_LOSS
from sklearn.metrics import f1_score


class Models():
    def __init__(self, args, d_in, d_hid, d_out):
        self.args = args
        self.d_in = d_in
        self.d_hid = d_hid
        self.classes = d_out
        self.device = self.args.device

        self.nfeat = args.num_feature
        self.nclass = args.num_class
        self.nhid = args.num_hidden
        self.nlayer = args.num_layer
        self.dropout = args.dropout
        self.model = Net(self.d_in, self.d_hid, self.classes).to(self.device)

    def fit(self, data, feats, seed):
        feats_trn = feats[data.train_mask].to(self.args.device)
        labels = data.y
        y_trn = labels[data.train_mask].to(self.args.device)
        feats_val = feats[data.valid_mask].to(self.args.device)
        y_val = labels[data.valid_mask].numpy()
        rn_weights = data.rn_weight[data.train_mask].to(self.args.device)
        model_loss = IMB_LOSS(self.args, data)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                     betas=(0.9, 0.999), eps=1e-08, weight_decay=self.args.weight_decay)
        best_res = 0
        best_epoch = 0

        for epoch in range(self.args.epoch):
            start_time = time.time()
            if epoch > self.args.lr_decay_epoch:
                new_lr = self.args.lr * pow(self.args.lr_decay_rate, (epoch - self.args.lr_decay_epoch))
                new_lr = max(new_lr, 1e-4)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

            self.model.train()
            data.batch = None
            optimizer.zero_grad()

            sup_logits = self.model(feats_trn)
            # sup_logits[data.train_mask]->133,7
            cls_loss = model_loss.compute(sup_logits, y_trn)

            if self.args.renode_reweight == 1:
                cls_loss = torch.sum(cls_loss * rn_weights) / cls_loss.size(0)
            else:
                cls_loss = torch.mean(cls_loss)

            cls_loss.backward()
            optimizer.step()

            val_res = self.predict_val(y_val, feats_val)

            if val_res > best_res:
                best_model = copy.deepcopy(self.model)
                best_res = val_res
                best_epoch = epoch

            epoch_time = time.time() - start_time
            if epoch % self.args.print_freq == 0:
                print(('Epoch:[{}/{}], Epoch_time:{:.3f}\t'
                       'Train_Loss:{:.4f}\t'
                       'Best_Res:{:.3f}, Val_Res:{:.3f}').format(
                    epoch, self.args.epoch+1, epoch_time, cls_loss.item(), best_res, val_res)
                )
            utils.log_tabular("Seed", seed)
            utils.log_tabular("Epoch", epoch)
            utils.log_tabular("Epoch Time", round(epoch_time, 5))
            utils.log_tabular("Train Loss", round(cls_loss.item(), 5))
            utils.log_tabular("val_Res", round(val_res, 5))
            utils.log_tabular("Best_Res", round(best_res, 5))
            utils.dump_tabular()

            if self.args.early_stop > 0 and epoch >= self.args.least_epoch and epoch - best_epoch >= self.args.early_stop:
                print('\nEarly stop at %d epoch. Since there is no improve in %d epoch' % (epoch, self.args.early_stop))
                break

        utils.save_pytorch_model(best_model)
        print('best_epoch,best_val_result:%d, %.4f' % (best_epoch, best_res))

        self.model = best_model
        # return best_model
        return best_res

    def predict_val(self, y, feats, test_type=''):
        self.model.eval()
        target = y

        with torch.no_grad():
            out = self.model(feats)
        pred = out.cpu().max(1)[1].numpy()

        w_f1 = f1_score(target, pred, average='weighted')

        if test_type == 'test':
            m_f1 = f1_score(target, pred, average='macro')
            return w_f1, m_f1

        return w_f1

    def predict(self, data, feats, target_mask, test_type=''):
        self.model.eval()
        target = data.y[target_mask].numpy()

        with torch.no_grad():
            out = self.model(feats[target_mask].to(self.args.device))
        pred = out.cpu().max(1)[1].numpy()

        w_f1 = f1_score(target, pred, average='weighted')

        if test_type == 'test':
            m_f1 = f1_score(target, pred, average='macro')
            return w_f1, m_f1

        return w_f1

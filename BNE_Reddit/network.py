# coding:utf-8
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_hid, bias=False),
            nn.BatchNorm1d(d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_hid, bias=False),
            nn.BatchNorm1d(d_hid),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(d_hid, d_out)

    def forward(self, inputs):
        feats = self.fc(inputs)
        outputs = self.fc2(feats)

        return outputs
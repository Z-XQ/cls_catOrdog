#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：catOrdog 
@File    ：model.py
@IDE     ：PyCharm 
@Author  ：zxq
@Date    ：2021/12/16 10:01 
"""
import torch
from torch import nn


class BPModel(nn.Module):
    def __init__(self):
        super(BPModel, self).__init__()
        # 全连接层节点数
        self.layer1 = nn.Linear(6, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 3)

        self.dropout1 = nn.Dropout(p=0.15)
        self.dropout2 = nn.Dropout(p=0.15)
        self.BN0 = nn.BatchNorm1d(6, momentum=0.5)
        self.BN1 = nn.BatchNorm1d(32, momentum=0.5)
        self.BN2 = nn.BatchNorm1d(16, momentum=0.5)

    def forward(self, x):
        # 可自行添加dropout和BatchNorm1d层
        x = self.BN0(x)
        x = self.BN1(self.layer1(x))
        x = torch.tanh(x)
        x = self.BN2(self.layer2(x))
        x = torch.tanh(x)
        out = torch.relu(self.layer3(x))
        return out


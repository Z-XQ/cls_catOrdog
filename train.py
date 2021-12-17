#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：catOrdog 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：zxq
@Date    ：2021/12/16 10:02 
"""
import cv2
from torchvision import datasets, transforms
import torch.utils.data
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

from dataset import CustomDataset

if __name__ == '__main__':

    # 数据处理
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # train_dataset = datasets.ImageFolder(root=r'PetImages/train/', transform=data_transform)
    train_dataset = CustomDataset(data_root=r'PetImages/train/')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    test_dataset = CustomDataset(data_root=r'PetImages/test/')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

    # 损失函数
    criteon = nn.CrossEntropyLoss()

    # 加载预训练模型
    transfer_model = models.resnet18(pretrained=True)
    dim_in = transfer_model.fc.in_features  # 修改最后一层，全连接层
    transfer_model.fc = nn.Linear(dim_in, 2)

    # 优化器adam
    optimizer = optim.Adam(transfer_model.parameters(), lr=0.01)

    # 加载模型到GPU
    transfer_model = transfer_model.cuda()

    global_step = 0
    # 模型训练
    transfer_model.train()
    for epoch in range(100):
        train_acc_num = 0
        test_acc_num = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()  # data: torch.float32. [b, c, h, w]. target: torch.int64. [b,]

            # 投入数据，得到预测值
            logits = transfer_model(data)
            _, pred = torch.max(logits.data, 1)
            # print(pred, target)
            loss = criteon(logits, target)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # 准确度计算
            train_acc_num += pred.eq(target).float().sum().item()

            # print("准确数:",train_acc_num," ",batch_idx, " ",len(data))
            train_acc = train_acc_num / ((batch_idx + 1) * len(data))
            # print(train_acc)
            # print(train_acc.item())
            global_step += 1

            print("epoch {} train loss: {}".format(epoch, loss))

        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            logits = transfer_model(data)
            test_loss = criteon(logits, target).item()

            print(print("epoch {} test loss: {}".format(epoch, test_loss)))
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：catOrdog 
@File    ：dataset.py
@IDE     ：PyCharm 
@Author  ：zxq
@Date    ：2021/12/16 9:59 
"""

# 输入数据的容器，配合DataLoader，完成mini_bacth训练方法，要包含__init__， __len__和__getitem__三个属性
import os

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self,
                 data_root,
                 ):
        """
        索引到所有训练数据
        :param data_root:
        """
        super(CustomDataset, self).__init__()
        self.x_data_list = []
        self.y_data_list = []
        categories = os.listdir(data_root)  # 目录下有cat和dog两个类别目录
        for i, category_name in enumerate(categories):
            category_full_path = os.path.join(data_root, category_name)
            img_name_list = os.listdir(category_full_path)
            for img_name in img_name_list:
                self.x_data_list.append(os.path.join(category_full_path, img_name))
                self.y_data_list.append(i)

    def __len__(self):
        return len(self.x_data_list)

    def __getitem__(self, index):
        """
        转换为网络输入形式
        :param index:
        :return:
            x_data: torch.float32. (3, 128, 128)
            y_data: torch.int64. ndim==0
        """
        img_path = self.x_data_list[index]
        # opencv读取部分会读取失败None 图片不完全是.jpg格式，还有一些是4通道的图
        # img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        x_data = torch.tensor(img).permute(2, 0, 1)  # (h,w,c) to (c, h, w)
        y_data = torch.tensor(self.y_data_list[index])
        return x_data.float(), y_data.long()  # target: int

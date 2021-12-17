#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：catOrdog 
@File    ：visual_data.py
@IDE     ：PyCharm 
@Author  ：zxq
@Date    ：2021/12/16 10:16 
"""

import matplotlib.pyplot as plt
import numpy
import os
from PIL import Image  # 读取图片模块
from matplotlib.image import imread

source_path = r"PetImages"
# 分别从Dog,Cat文件夹中选取10张图片显示
train_Dog_dir = os.path.join(source_path, "train", "Dog")
train_Cat_dir = os.path.join(source_path, "train", "Cat")
Dog_image_list = os.listdir(train_Dog_dir)

Cat_image_list = os.listdir(train_Cat_dir)
show_image = [os.path.join(train_Dog_dir, Dog_image_list[i]) for i in range(10)]
show_image.extend([os.path.join(train_Cat_dir, Cat_image_list[i]) for i in range(10)])
for i in show_image:
    print(i)
plt.figure()

for i in range(1, 20):
    plt.subplot(4, 5, i)
    img = Image.open(show_image[i - 1])
    plt.imshow(img)

plt.show()
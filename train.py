#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mask R-CNN模型训练脚本

本脚本用于训练Mask R-CNN神经网络模型，以检测卫星图像中的运动场。
训练过程分为多个阶段，从预训练的COCO模型开始，逐步微调不同层次的网络参数。

作者: 原作者
日期: 2017
"""

# 从shapes示例文件修改而来

import sys
sys.path.append("Mask_RCNN")  # 添加Mask_RCNN库到Python路径

import os
import random
import math
import time
import numpy as np
import random
import glob
import skimage
import osmmodelconfig  # 导入OSM模型配置

from config import Config
import imagestoosm.config as osmcfg  # 项目配置
import utils
import model as modellib
from model import log

# 项目根目录
#ROOT_DIR = os.getcwd()
ROOT_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))

# 保存日志和训练模型的目录
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# COCO预训练权重的路径
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# 初始化模型配置
config = osmmodelconfig.OsmModelConfig()
config.ROOT_DIR = ROOT_DIR
config.display()  # 显示配置信息

# 获取所有训练图像的ID列表
fullTrainingDir = os.path.join(ROOT_DIR, osmcfg.trainDir, "*")
fullImageList = []
for imageDir in glob.glob(fullTrainingDir): 
    if (os.path.isdir(os.path.join(fullTrainingDir, imageDir))):
        id = os.path.split(imageDir)[1]
        fullImageList.append(id)

# 随机打乱图像列表，确保训练和验证数据的随机性
random.shuffle(fullImageList)

# 将数据集分为训练集(75%)和验证集(25%)
cutoffIndex = int(len(fullImageList) * 0.75)
trainingImages = fullImageList[0:cutoffIndex]
validationImages = fullImageList[cutoffIndex:-1]

# 加载训练数据集
dataset_train = osmmodelconfig.OsmImagesDataset(ROOT_DIR)
dataset_train.load(trainingImages, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# 加载验证数据集
dataset_val = osmmodelconfig.OsmImagesDataset(ROOT_DIR)
dataset_val.load(validationImages, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

# 创建训练模式的模型
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# 选择初始权重
init_with = "coco"  # 可选值: imagenet, coco, 或 last

if init_with == "imagenet":
    # 使用在ImageNet上预训练的权重
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # 加载在MS COCO数据集上训练的权重，但跳过因类别数不同而不同的层
    # 查看README获取下载COCO权重的说明
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # 加载上次训练的模型权重并继续训练
    print(model.find_last()[1])
    model.load_weights(model.find_last()[1], by_name=True)

# 如果不是从上次训练的模型继续，则执行完整的训练流程
if (init_with != "last"):
    # 训练阶段1 - 仅训练网络头部
    # 根据需要调整epochs和layers参数
    print("训练网络头部")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')

    # 训练阶段2 - 微调ResNet第4阶段及以上的层
    print("训练Resnet第3层及以上")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/10,
                epochs=100,
                layers='3+')
    
# 训练阶段3 - 微调ResNet第3阶段及以上的层
print("训练所有层")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 100,
            epochs=1000,
            layers='all')
            

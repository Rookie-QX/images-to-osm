#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OSM模型配置模块

本模块定义了用于训练Mask R-CNN模型的配置类和数据集类，
用于检测卫星图像中的运动场（棒球场、篮球场、网球场等）。

作者: 原作者
日期: 2017
"""

import sys
sys.path.append("Mask_RCNN")  # 添加Mask_RCNN库到Python路径

import os
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import random
import glob
import skimage

from config import Config
import imagestoosm.config as osmcfg  # 项目配置
import utils
import model as modellib
import visualize
from model import log

# 定义要检测的运动场类型及其对应的类别ID
featureNames = {
    "baseball": 1,
    "basketball": 2,
    "tennis": 3
#    "american_football": 4,
#    "soccer": 5,
}

class OsmModelConfig(Config):
    """
    OSM图像训练的配置类。
    
    继承自基础Config类，并覆盖特定于OSM数据集的值。
    """
    # 给配置一个可识别的名称
    NAME = "OSM Images Baseball,Basketball,Tennis"

    # 批量大小为(GPU数量 * 每个GPU的图像数)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    LEARNING_RATE = 0.001

    # 2分钟的epoch
    #STEPS_PER_EPOCH = 100 // IMAGES_PER_GPU

    # 1小时的epoch
    STEPS_PER_EPOCH = 12000 // IMAGES_PER_GPU

    # 类别数量（包括背景）
    NUM_CLASSES = 1 + len(featureNames)  # 背景 + 特征类型数量

    # 每个瓦片是256像素宽，训练数据是3x3瓦片
    TILES = 3
    IMAGE_MIN_DIM = 256 * TILES
    IMAGE_MAX_DIM = 256 * TILES

    # 小掩码形状，用于节省内存
    MINI_MASK_SHAPE = (128, 128) 
    #MASK_SHAPE = (IMAGE_MIN_DIM, IMAGE_MIN_DIM) 

    # 减少每个图像的训练ROI，因为图像很小且对象很少
    # 目标是允许ROI采样选择33%的正ROI
    #TRAIN_ROIS_PER_IMAGE = 64
    #DETECTION_MAX_INSTANCES = 64

    # 验证步骤数量
    VALIDATION_STEPS = 100

class OsmImagesDataset(utils.Dataset):
    """
    OSM图像数据集类。
    
    用于加载和处理OSM运动场图像数据集。
    """

    def __init__(self, rootDir):
        """
        初始化OSM图像数据集。
        
        参数:
            rootDir (str): 数据集根目录
        """
        utils.Dataset.__init__(self)
        self.ROOT_DIR = rootDir

    def load(self, imageDirs, height, width):
        """
        加载指定的图像目录列表。
        
        参数:
            imageDirs (list): 图像目录列表
            height (int): 图像高度
            width (int): 图像宽度
        """
        # 添加每种运动场类型作为一个类别
        for feature in featureNames:
            self.add_class("osm", featureNames[feature], feature)

        # 添加图像
        for i in range(len(imageDirs)):
            imgPath = os.path.join(self.ROOT_DIR, osmcfg.trainDir, imageDirs[i], imageDirs[i] + ".jpg")
            self.add_image("osm", image_id=imageDirs[i], path=imgPath, width=width, height=height)

    def load_mask(self, image_id):
        """
        为给定图像ID生成实例掩码。
        
        参数:
            image_id (str): 图像ID
            
        返回:
            tuple: (掩码数组, 类别ID数组)
        """
        info = self.image_info[image_id]

        # 查找图像目录中的所有PNG掩码文件
        imgDir = os.path.join(self.ROOT_DIR, osmcfg.trainDir, info['id'])
        wildcard = os.path.join(imgDir, "*.png")

        # 文件命名格式示例: 00015-american_football-0.png  00015-baseball-0.png  00015-baseball-1.png

        # 计算掩码数量
        maskCount = 0
        for filePath in glob.glob(wildcard): 
            filename = os.path.split(filePath)[1] 
            parts = filename.split("-")
            if (len(parts) == 3) and parts[1] in featureNames: 
                maskCount += 1

        # 创建掩码和类别ID数组
        mask = np.zeros([info['height'], info['width'], maskCount], dtype=np.uint8)
        class_ids = np.zeros((maskCount), np.int32)
 
        # 加载每个掩码并分配类别ID
        count = 0
        for filePath in glob.glob(wildcard): 
            filename = os.path.split(filePath)[1] 
            parts = filename.split("-")
            if (len(parts) == 3) and parts[1] in featureNames: 
                imgPath = filePath
                mask[:, :, count] = skimage.io.imread(filePath)
                class_ids[count] = featureNames[parts[1]]
                count += 1            
                
        return mask, class_ids

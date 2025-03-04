#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mask R-CNN - 在形状数据集上训练

本脚本展示如何在自定义数据集上训练Mask R-CNN。为了简化过程，我们使用一个合成的形状数据集
（方形、三角形和圆形），这使得训练速度更快。不过，您仍然需要GPU，因为网络骨干是Resnet101，
在CPU上训练会太慢。在GPU上，您可以在几分钟内获得不错的结果，在不到一小时内获得良好的结果。

形状数据集的代码包含在下面。它可以即时生成图像，因此不需要下载任何数据。
而且它可以生成任意大小的图像，所以我们选择较小的图像尺寸以加快训练速度。

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
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log

# 项目根目录
ROOT_DIR = os.getcwd()

# 保存日志和训练模型的目录
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# COCO预训练权重的路径
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


# ## 配置

class ShapesConfig(Config):
    """
    用于在玩具形状数据集上训练的配置。
    
    继承自基础Config类，并覆盖特定于玩具形状数据集的值。
    """
    # 给配置一个可识别的名称
    NAME = "shapes"

    # 在1个GPU上训练，每个GPU 8张图像。我们可以在每个GPU上放置多个图像，
    # 因为图像很小。批量大小为8（GPU数量 * 每个GPU的图像数）。
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # 类别数量（包括背景）
    NUM_CLASSES = 1 + 3  # 背景 + 3种形状

    # 使用小图像加快训练速度。设置小边和大边的限制，这决定了图像形状。
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # 使用较小的锚点，因为我们的图像和对象都很小
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # 锚点边长（像素）

    # 减少每个图像的训练ROI，因为图像很小且对象很少。
    # 目标是允许ROI采样选择33%的正ROI。
    TRAIN_ROIS_PER_IMAGE = 32

    # 使用小的epoch，因为数据很简单
    STEPS_PER_EPOCH = 100

    # 使用小的验证步骤，因为epoch很小
    VALIDATION_STEPS = 5
    
# 初始化配置
config = ShapesConfig()
config.display()  # 显示配置信息


# ## 笔记本偏好设置

def get_ax(rows=1, cols=1, size=8):
    """
    返回用于笔记本中所有可视化的Matplotlib Axes数组。
    提供一个中心点来控制图形大小。
    
    参数:
        rows (int): 行数
        cols (int): 列数
        size (int): 图形大小
        
    返回:
        matplotlib.axes.Axes: Matplotlib Axes对象
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# ## 数据集
# 
# 创建一个合成数据集
# 
# 扩展Dataset类并添加一个方法来加载形状数据集，`load_shapes()`，并覆盖以下方法：
# 
# * load_image()
# * load_mask()
# * image_reference()

class ShapesDataset(utils.Dataset):
    """
    生成形状合成数据集。
    
    该数据集由简单的形状（三角形、方形、圆形）组成，这些形状随机放置在空白表面上。
    图像是即时生成的，不需要文件访问。
    """

    def load_shapes(self, count, height, width):
        """
        生成请求数量的合成图像。
        
        参数:
            count (int): 要生成的图像数量
            height (int): 生成图像的高度
            width (int): 生成图像的宽度
        """
        # 添加类别
        self.add_class("shapes", 1, "square")
        self.add_class("shapes", 2, "circle")
        self.add_class("shapes", 3, "triangle")

        # 添加图像
        # 生成图像的随机规格（即颜色和形状大小和位置列表）。
        # 这比实际图像更紧凑。图像在load_image()中即时生成。
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def load_image(self, image_id):
        """
        从给定图像ID的规格生成图像。
        
        通常这个函数从文件加载图像，但在这种情况下，
        它根据image_info中的规格即时生成图像。
        
        参数:
            image_id (int): 图像ID
            
        返回:
            numpy.ndarray: 生成的图像
        """
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def image_reference(self, image_id):
        """
        返回图像的形状数据。
        
        参数:
            image_id (int): 图像ID
            
        返回:
            list: 形状数据列表
        """
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """
        为给定图像ID的形状生成实例掩码。
        
        参数:
            image_id (int): 图像ID
            
        返回:
            tuple: 包含掩码和类别ID的元组
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)
        # 处理遮挡
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # 将类名映射到类ID
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask, class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        """
        根据给定的规格绘制形状。
        
        参数:
            image (numpy.ndarray): 要绘制形状的图像
            shape (str): 形状类型（方形、圆形、三角形）
            dims (tuple): 形状的尺寸和位置
            color: 形状的颜色
            
        返回:
            numpy.ndarray: 绘制了形状的图像
        """
        # 获取中心x, y和大小s
        x, y, s = dims
        if shape == 'square':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
        elif shape == "circle":
            cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y-s),
                                (x-s/math.sin(math.radians(60)), y+s),
                                (x+s/math.sin(math.radians(60)), y+s),
                                ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        """
        生成位于给定高度和宽度边界内的随机形状的规格。
        
        返回三个值的元组:
        * 形状名称（方形、圆形等）
        * 形状颜色：一个包含3个值的元组，RGB
        * 形状尺寸：定义形状大小和位置的值元组。每种形状类型不同。
        
        参数:
            height (int): 图像高度
            width (int): 图像宽度
            
        返回:
            tuple: 包含形状名称、颜色和尺寸的元组
        """
        # 形状
        shape = random.choice(["square", "circle", "triangle"])
        # 颜色
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # 中心x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # 大小
        s = random.randint(buffer, height//4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """
        创建具有多个形状的图像的随机规格。
        
        返回图像的背景颜色和形状规格列表，可用于绘制图像。
        
        参数:
            height (int): 图像高度
            width (int): 图像宽度
            
        返回:
            tuple: 包含背景颜色和形状列表的元组
        """
        # 选择随机背景颜色
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # 生成几个随机形状并记录它们的边界框
        shapes = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y-s, x-s, y+s, x+s])
        # 应用非最大抑制，阈值为0.3，以避免形状相互覆盖
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes


# 加载训练数据集
dataset_train = ShapesDataset()
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# 加载验证数据集
dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()


# 加载并显示随机样本
#image_ids = np.random.choice(dataset_train.image_ids, 4)
#for image_id in image_ids:
#    image = dataset_train.load_image(image_id)
#    mask, class_ids = dataset_train.load_mask(image_id)
#    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


# ## 创建模型

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
    model.load_weights(model.find_last()[1], by_name=True)


# ## 训练
# 
# 分两个阶段训练:
# 1. 只训练头部。这里我们冻结所有骨干层，只训练随机初始化的层（即我们没有使用MS COCO预训练权重的层）。
#    要只训练头部层，将`layers='heads'`传递给`train()`函数。
# 
# 2. 微调所有层。对于这个简单的例子，这不是必需的，但我们包括它以展示过程。
#    只需将`layers="all"`传递给train()函数即可训练所有层。

# 训练头部分支
# 传递layers="heads"冻结除头部层之外的所有层
# 您也可以传递正则表达式，通过名称模式选择要训练的层
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=20, 
            layers='heads')


# 微调所有层
# 传递layers="all"训练所有层
# 您也可以传递正则表达式，通过名称模式选择要训练的层
#model.train(dataset_train, dataset_val, 
#            learning_rate=config.LEARNING_RATE / 10,
#            epochs=2, 
#            layers="all")


# 保存权重
# 通常不需要，因为回调在每个epoch后保存
# 取消注释以手动保存
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)


# ## 检测

class InferenceConfig(ShapesConfig):
    """
    用于推理的配置类，继承自ShapesConfig
    """
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# 初始化推理配置
inference_config = InferenceConfig()

# 在推理模式下重新创建模型
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# 获取保存的权重路径
# 可以设置特定路径或查找最后训练的权重
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# 加载训练好的权重（在此处填写训练权重的路径）
assert model_path != "", "请提供训练权重的路径"
print("从以下位置加载权重: ", model_path)
model.load_weights(model_path, by_name=True)


# 在随机图像上测试
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
   modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_bbox)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

# 显示原始图像和真实标注
visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))


# 使用模型进行检测
results = model.detect([original_image], verbose=1)

# 显示检测结果
r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())


# ## 评估

# 计算VOC风格的mAP @ IoU=0.5
# 在10张图像上运行。增加数量以获得更好的准确性。
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # 加载图像和真实标注数据
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
       modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # 运行对象检测
    results = model.detect([image], verbose=0)
    r = results[0]
    # 计算AP
    AP, precisions, recalls, overlaps = \
       utils.compute_ap(gt_bbox, gt_class_id,
                         r["rois"], r["class_ids"], r["scores"])
    APs.append(AP)
    
# 打印平均AP
print("mAP: ", np.mean(APs))


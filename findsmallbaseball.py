#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查找小型棒球场模块

本模块用于计算OSM中棒球场的面积（以平方米为单位），
帮助识别较小的棒球场，这些场地可能需要特殊处理。

作者: 原作者
日期: 2017
"""

import imagestoosm.config as cfg  # 项目配置
import os
import QuadKey.quadkey as quadkey  # 用于地图瓦片坐标转换
import numpy as np
import shapely.geometry as geometry  # 用于几何计算
from skimage import draw
from skimage import io
import csv

# 加载OSM特征到多边形数组的哈希表中，以像素为单位

# 仅处理棒球场类型
for classDir in os.listdir(cfg.rootOsmDir):
    if (classDir == 'baseball'):
        classDirFull = os.path.join(cfg.rootOsmDir, classDir)
        # 遍历每个棒球场文件
        for fileName in os.listdir(classDirFull):
            fullPath = os.path.join(cfg.rootOsmDir, classDir, fileName)
            with open(fullPath, "rt") as csvfile:
                csveader = csv.reader(csvfile, delimiter='\t')

                # 收集所有点的坐标
                pts = []
                for row in csveader:
                    latLot = (float(row[0]), float(row[1]))
                    # 将地理坐标转换为像素坐标
                    pixel = quadkey.TileSystem.geo_to_pixel(latLot, cfg.tileZoom)
                    pts.append(pixel)

                # 创建多边形
                poly = geometry.Polygon(pts)

                # 计算面积（平方米）
                # 在缩放级别18时，每个像素约为0.596米
                areaMeters = poly.area * 0.596 * 0.596

                # 输出文件名和面积
                print("{}\t{}".format(fileName, areaMeters))









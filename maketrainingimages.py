#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成训练图像模块

本模块用于从Bing地图瓦片创建Mask R-CNN训练数据集。
它将瓦片拼接成更大的图像，并为每个与图像相交的OSM运动场生成掩码。
这些图像和掩码将用于训练神经网络识别卫星图像中的运动场。

作者: 原作者
日期: 2017
"""

import imagestoosm.config as cfg  # 项目配置
import os
import QuadKey.quadkey as quadkey  # 地图瓦片坐标转换
import numpy as np
import shapely.geometry as geometry  # 几何处理
from skimage import draw  # 用于绘制多边形
from skimage import io  # 图像读写
import csv

# 特征裁剪的最小比例（如果裁剪后的特征面积小于原面积的30%，则忽略）
minFeatureClip = 0.3

# 生成训练数据图像的步骤：
# 1. 构建OSM数据的索引，每个点
# 2. 对于每个瓦片：
#    检查东、南和东南方向的瓦片，以创建512x512的瓦片，如果没有则跳过
# 3. 确定图像的边界框
# 4. 将图像保存为PNG
# 5. 查看哪些特征与图像重叠
# 
# 如果特征重叠超过20%：
# 生成增强列表（N +45,-45度旋转），N个偏移
# 为当前图像的特征生成掩码
#
# 训练时将进行翻转、强度和颜色偏移（如果需要）

# 清空并重新创建训练目录
os.system("rm -R " + cfg.trainDir)
os.mkdir(cfg.trainDir)

# 将OSM特征加载到多边形数组的哈希表中（以像素为单位）
features = {}

# 遍历所有运动场类型目录
for classDir in os.listdir(cfg.rootOsmDir):
    classDirFull = os.path.join(cfg.rootOsmDir, classDir)
    # 遍历每个运动场文件
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

            # 不要学习仅在钻石区轮廓的棒球场。它们标记错误，不想教神经网络
            # 这是正确的。OSM数据库中有超过1000个这样的字段，我们无法避免它们。
            if (classDir != "baseball" or areaMeters > 2500):                
                feature = {
                    "geometry": poly,
                    "filename": fullPath
                }

                # 为每种类型创建特征列表
                if ((classDir in features) == False):
                    features[classDir] = []

                features[classDir].append(feature)

# 图像写入计数器
imageWriteCounter = 0
# 遍历瓦片目录中的所有文件
for root, subFolders, files in os.walk(cfg.rootTileDir):
    for file in files: 
        # 从文件名获取四叉树键
        quadKeyStr = os.path.splitext(file)[0]

        qkRoot = quadkey.from_str(quadKeyStr)
        # 获取瓦片的像素坐标
        tilePixel = quadkey.TileSystem.geo_to_pixel(qkRoot.to_geo(), qkRoot.level)

        tileRootDir = os.path.split(root)[0]

        # 将相邻瓦片拼接在一起，创建更大的图像（最大为maxImageSize）
        maxImageSize = 256 * 3
        maxTileCount = maxImageSize // 256
        count = 0
        # 创建空白图像
        image = np.zeros([maxImageSize, maxImageSize, 3], dtype=np.uint8)
        # 填充图像
        for x in range(maxTileCount):
            for y in range(maxTileCount):
                pixel = (tilePixel[0] + 256*x, tilePixel[1] + 256*y)
                geo = quadkey.TileSystem.pixel_to_geo(pixel, qkRoot.level)
                qk = quadkey.from_geo(geo, qkRoot.level)

                qkStr = str(qk)

                tileCacheDir = os.path.join(tileRootDir, qkStr[-3:])
                tileFileName = "%s/%s.jpg" % (tileCacheDir, qkStr)

                # 如果瓦片文件存在，读取并添加到图像中
                if (os.path.exists(tileFileName)):
                    try:
                        image[y*256:(y+1)*256, x*256:(x+1)*256, 0:3] = io.imread(tileFileName)
                        count += 1
                    except:
                        # 下次再尝试获取瓦片
                        os.remove(tileFileName)
                        
        # 创建图像边界框多边形
        pts = []
        pts.append((tilePixel[0] + 0, tilePixel[1] + 0))
        pts.append((tilePixel[0] + 0, tilePixel[1] + maxImageSize))
        pts.append((tilePixel[0] + maxImageSize, tilePixel[1] + maxImageSize))
        pts.append((tilePixel[0] + maxImageSize, tilePixel[1] + 0))
        
        imageBoundingBoxPoly = geometry.Polygon(pts)

        # 创建特征掩码
        featureMask = np.zeros((maxImageSize, maxImageSize), dtype=np.uint8)
        featureCountTotal = 0
        usedFileNames = []
        # 检查每种特征类型
        for featureType in features:
            featureCount = 0
            # 检查每个特征
            for feature in features[featureType]:
                # 如果特征与图像边界框相交
                if (imageBoundingBoxPoly.intersects(feature['geometry'])):
                    area = feature['geometry'].area

                    # 获取特征的坐标
                    xs, ys = feature['geometry'].exterior.coords.xy
                    # 转换为相对于瓦片的坐标
                    xs = [x - tilePixel[0] for x in xs]
                    ys = [y - tilePixel[1] for y in ys]
    
                    # 裁剪坐标到图像边界
                    xsClipped = [min(max(x, 0), maxImageSize) for x in xs]
                    ysClipped = [min(max(y, 0), maxImageSize) for y in ys]

                    # 创建裁剪后的多边形
                    pts2 = []
                    for i in range(len(xs)):
                        pts2.append((xsClipped[i], ysClipped[i]))

                    clippedPoly = geometry.Polygon(pts2)
                    newArea = clippedPoly.area

                    # 如果裁剪后的面积足够大（相对于原始面积）
                    if (area > 0 and newArea/area > minFeatureClip):
                        # 创建训练图像目录
                        if (os.path.exists("%s/%06d" % (cfg.trainDir, imageWriteCounter)) == False):
                            os.mkdir("%s/%06d" % (cfg.trainDir, imageWriteCounter))
                        
                        # 创建特征掩码
                        featureMask.fill(0)
                        rr, cc = draw.polygon(xs, ys, (maxImageSize, maxImageSize))
                        featureMask[cc, rr] = 255
                        # 保存掩码图像
                        io.imsave("%s/%06d/%06d-%s-%d.png" % (cfg.trainDir, imageWriteCounter, imageWriteCounter, featureType, featureCount), featureMask)
                        usedFileNames.append(feature['filename'])
                        featureCount += 1
                        featureCountTotal += 1

        # 如果图像中包含特征，保存图像和相关信息
        if (featureCountTotal > 0):
            # 保存图像
            io.imsave("%s/%06d/%06d.jpg" % (cfg.trainDir, imageWriteCounter, imageWriteCounter), image, quality=100)
            
            # 保存元数据文件
            with open("%s/%06d/%06d.txt" % (cfg.trainDir, imageWriteCounter, imageWriteCounter), "wt") as text_file:
                text_file.write("%s\n" % (str(qkRoot)))
                text_file.write("%0.8f,%0.8f\n" % qkRoot.to_geo())
                # 记录使用的特征文件
                for f in usedFileNames:
                    text_file.write("%s\n" % (f))
                    
            imageWriteCounter += 1

            # 输出处理信息
            print("%s - %s - tiles %d - features %d" % (os.path.join(root, file), quadKeyStr, count, featureCountTotal))






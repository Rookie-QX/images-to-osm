#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建OSM异常检测模块

本模块使用训练好的Mask R-CNN模型检测卫星图像中的运动场，
并将检测到的但在OSM中缺失的运动场保存为OSM文件，以供后续审查。

"""

import sys
sys.path.append("Mask_RCNN")  # 添加Mask_RCNN库到Python路径

import os
import sys
import glob
import osmmodelconfig  # OSM模型配置
import skimage
import math
import imagestoosm.config as osmcfg  # 项目配置
import model as modellib  # Mask R-CNN模型
import visualize as vis  # 可视化工具
import numpy as np
import csv
import QuadKey.quadkey as quadkey  # 地图瓦片坐标转换
import shapely.geometry as geometry  # 几何处理
import shapely.affinity as affinity  # 几何变换
import matplotlib.pyplot as plt  # 绘图
import cv2  # 计算机视觉库
import scipy.optimize  # 优化算法
import time
from skimage import draw
from skimage import io

# 是否显示图形界面
showFigures = False

def toDegrees(rad):
    """
    将弧度转换为角度。
    
    参数:
        rad (float): 弧度值
        
    返回:
        float: 角度值
    """
    return rad * 180/math.pi

def writeOSM(osmFileName, featureName, simpleContour, tilePixel, qkRoot):
    """
    将检测到的运动场写入OSM文件。
    
    参数:
        osmFileName (str): 输出OSM文件名
        featureName (str): 特征类型（如baseball, tennis等）
        simpleContour (numpy.ndarray): 轮廓点
        tilePixel (tuple): 瓦片像素坐标
        qkRoot (QuadKey): 四叉树键根节点
    """
    with open(osmFileName, "wt", encoding="ascii") as f: 
        f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
        f.write("<osm version=\"0.6\">\n")
        id = -1
        # 写入节点
        for pt in simpleContour:
            # 将像素坐标转换回地理坐标
            geo = quadkey.TileSystem.pixel_to_geo((pt[0,0] + tilePixel[0], pt[0,1] + tilePixel[1]), qkRoot.level)
            f.write("  <node id=\"{}\" lat=\"{}\" lon=\"{}\" />\n".format(id, geo[0], geo[1]))
            id -= 1

        # 写入路径
        f.write("  <way id=\"{}\" visible=\"true\">\n".format(id))
        id = -1
        for pt in simpleContour:
            f.write("    <nd ref=\"{}\" />\n".format(id))
            id -= 1
        # 闭合路径（连接回第一个点）
        f.write("    <nd ref=\"{}\" />\n".format(-1))
        # 添加标签
        f.write("    <tag k=\"{}\" v=\"{}\" />\n".format("leisure", "pitch"))
        f.write("    <tag k=\"{}\" v=\"{}\" />\n".format("sport", featureName))
        f.write("  </way>\n")

        f.write("</osm>\n")   
        f.close             

def writeShape(wayNumber, finalShape, image, bbTop, bbHeight, bbLeft, bbWidth):
    """
    将拟合的形状写入文件并返回下一个可用的路径编号。
    
    参数:
        wayNumber (int): 当前路径编号
        finalShape (shapely.geometry): 最终形状
        image (numpy.ndarray): 图像数据
        bbTop, bbHeight, bbLeft, bbWidth (int): 边界框参数
        
    返回:
        int: 下一个可用的路径编号
    """
    # 限制点的数量
    nPts = int(finalShape.length)
    if (nPts > 5000):
        nPts = 5000
    fitContour = np.zeros((nPts, 1, 2), dtype=np.int32)

    if (nPts > 3):
        # 从形状中提取点
        for t in range(0, nPts):
            pt = finalShape.interpolate(t)
            fitContour[t, 0, 0] = pt.x
            fitContour[t, 0, 1] = pt.y
            
        # 简化轮廓
        fitContour = [fitContour]
        fitContour = [cv2.approxPolyDP(cnt, 2, True) for cnt in fitContour]
                        
        # 在图像上绘制轮廓
        image = np.copy(imageNoMasks)
        cv2.drawContours(image, fitContour, -1, (0, 255, 0), 2)
        if (showFigures):
            fig.add_subplot(2, 2, 3)
            plt.title(featureName + " " + str(r['scores'][i]) + " Fit")
            plt.imshow(image[bbTop:bbTop+bbHeight, bbLeft:bbLeft+bbWidth])

        # 查找可用的文件名
        while (os.path.exists("anomaly/add/{0:06d}.osm".format(wayNumber))):
            wayNumber += 1

        # 保存调试图像
        debugFileName = os.path.join(inference_config.ROOT_DIR, "anomaly", "add", "{0:06d}.jpg".format(wayNumber))
        io.imsave(debugFileName, image[bbTop:bbTop+bbHeight, bbLeft:bbLeft+bbWidth], quality=100)

        # 保存OSM文件
        osmFileName = os.path.join(inference_config.ROOT_DIR, "anomaly", "add", "{0:06d}.osm".format(wayNumber))
        writeOSM(osmFileName, featureName, fitContour[0], tilePixel, qkRoot)

    # 显示图形（如果启用）
    if (showFigures):
        plt.show(block=False)
        plt.pause(0.05)

    return wayNumber

# 确保异常目录存在
if (os.path.exists("anomaly") == False):
    os.mkdir("anomaly")

if (os.path.exists("anomaly/add") == False):
    os.mkdir("anomaly/add")

# 初始化模型配置
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# 创建推理配置
inference_config = osmmodelconfig.OsmModelConfig()
inference_config.ROOT_DIR = ROOT_DIR
inference_config.GPU_COUNT = 1
inference_config.IMAGES_PER_GPU = 1
inference_config.DETECTION_MIN_CONFIDENCE = 0.9
inference_config.display()

# 创建模型对象
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# 加载最新的模型权重
model_path = model.find_last()[1]
print("加载权重 ", model_path)
model.load_weights(model_path, by_name=True)

# 获取类别名称
class_names = []
for feature in osmmodelconfig.featureNames:
    class_names.append(feature)

# 创建图形窗口（如果启用）
if (showFigures):
    fig = plt.figure(figsize=(12, 12))

# 初始化路径编号
wayNumber = 1

# 遍历所有瓦片文件
for root, subFolders, files in os.walk(osmcfg.rootTileDir):
    for file in files:
        # 只处理JPG文件
        if (file.endswith(".jpg")):
            # 从文件名获取四叉树键
            quadKeyStr = os.path.splitext(file)[0]
            qkRoot = quadkey.from_str(quadKeyStr)
            
            # 获取瓦片的像素坐标
            tilePixel = quadkey.TileSystem.geo_to_pixel(qkRoot.to_geo(), qkRoot.level)
            
            # 获取瓦片的地理坐标
            geo = qkRoot.to_geo()
            
            # 创建图像边界框多边形
            pts = []
            pts.append((tilePixel[0] + 0, tilePixel[1] + 0))
            pts.append((tilePixel[0] + 0, tilePixel[1] + 256))
            pts.append((tilePixel[0] + 256, tilePixel[1] + 256))
            pts.append((tilePixel[0] + 256, tilePixel[1] + 0))
            
            imageBoundingBoxPoly = geometry.Polygon(pts)
            
            # 检查是否有任何已知的特征与此瓦片相交
            skipTile = False
            for featureType in osmmodelconfig.featureNames:
                # 跳过不存在的特征类型目录
                if (os.path.exists(os.path.join(osmcfg.rootOsmDir, featureType)) == False):
                    continue
                    
                classDirFull = os.path.join(osmcfg.rootOsmDir, featureType)
                for fileName in os.listdir(classDirFull):
                    fullPath = os.path.join(osmcfg.rootOsmDir, featureType, fileName)
                    with open(fullPath, "rt") as csvfile:
                        csveader = csv.reader(csvfile, delimiter='\t')
                        
                        # 收集所有点的坐标
                        pts = []
                        for row in csveader:
                            latLot = (float(row[0]), float(row[1]))
                            # 将地理坐标转换为像素坐标
                            pixel = quadkey.TileSystem.geo_to_pixel(latLot, osmcfg.tileZoom)
                            pts.append(pixel)
                            
                        # 创建多边形
                        poly = geometry.Polygon(pts)
                        
                        # 如果已知特征与瓦片相交，跳过此瓦片
                        if (imageBoundingBoxPoly.intersects(poly)):
                            skipTile = True
                            break
                            
                if (skipTile):
                    break
                    
            if (skipTile):
                continue
                
            # 读取瓦片图像
            image = skimage.io.imread(os.path.join(root, file))
            
            # 保存无掩码的图像副本
            imageNoMasks = np.copy(image)
            
            # 使用模型进行检测
            results = model.detect([image], verbose=0)
            r = results[0]
            
            # 显示检测结果（如果启用）
            if (showFigures):
                fig.add_subplot(2, 2, 1)
                plt.title(quadKeyStr)
                plt.imshow(image)
                
                fig.add_subplot(2, 2, 2)
                plt.title("Detections")
                vis.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                     class_names, r['scores'], ax=plt.gca())
                
            # 处理每个检测结果
            for i in range(len(r['class_ids'])):
                # 获取特征类型
                featureType = class_names[r['class_ids'][i] - 1]
                
                # 获取边界框
                y1, x1, y2, x2 = r['rois'][i]
                bbTop = y1
                bbHeight = y2 - y1
                bbLeft = x1
                bbWidth = x2 - x1
                
                # 获取掩码
                mask = r['masks'][:, :, i]
                
                # 查找掩码轮廓
                ret, thresh = cv2.threshold(mask.astype(np.uint8) * 255, 127, 255, 0)
                rawContours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # 如果找到轮廓
                if (len(rawContours) > 0):
                    # 根据特征类型选择不同的形状拟合方法
                    if (featureType == "baseball"):
                        # 棒球场通常是圆形或椭圆形
                        
                        # 定义椭圆形状生成函数
                        def makeEllipse(paramsX):
                            """
                            根据参数生成椭圆形状。
                            
                            参数:
                                paramsX (list): [centerX, centerY, width, height, angle]
                                
                            返回:
                                shapely.geometry: 椭圆形状
                            """
                            centerX = paramsX[0]
                            centerY = paramsX[1]
                            width = abs(paramsX[2])
                            height = abs(paramsX[3])
                            angle = paramsX[4]
                            
                            # 创建椭圆点
                            nPts = 24
                            pts = []
                            for i in range(nPts):
                                theta = i * 2 * math.pi / nPts
                                x = width/2 * math.cos(theta)
                                y = height/2 * math.sin(theta)
                                pts.append((x, y))
                                
                            # 创建形状并应用变换
                            fitShape = geometry.LinearRing(pts)
                            fitShape = affinity.rotate(fitShape, angle, use_radians=True)
                            fitShape = affinity.translate(fitShape, centerX, centerY)
                            
                            return fitShape
                            
                        # 定义椭圆拟合误差函数
                        def fitEllipse(paramsX):
                            """
                            计算椭圆拟合误差。
                            
                            参数:
                                paramsX (list): 椭圆参数
                                
                            返回:
                                float: 拟合误差
                            """
                            fitShape = makeEllipse(paramsX)
                            
                            sum = 0
                            # 计算轮廓点到椭圆的距离平方和
                            for cnt in rawContours:
                                for pt in cnt:
                                    p = geometry.Point(pt[0])
                                    d = p.distance(fitShape)
                                    sum += d*d
                            return sum
                            
                        # 计算轮廓的中心点
                        cm = np.mean(rawContours[0], axis=0)
                        
                        # 尝试不同的初始角度，找到最佳拟合
                        result = {}
                        angleStepCount = 8
                        for angleI in range(angleStepCount):
                            centerX = cm[0, 0]
                            centerY = cm[0, 1]
                            width = math.sqrt(cv2.contourArea(rawContours[0]))
                            height = width
                            angle = 2*math.pi * float(angleI)/angleStepCount
                            x0 = np.array([centerX, centerY, width, height, angle])
                            resultE = scipy.optimize.minimize(fitEllipse, x0, method='nelder-mead', options={'xtol': 1e-6, 'maxiter': 50})
                            
                            if (angleI == 0):                            
                                result = resultE
                                
                            if (resultE.fun < result.fun):
                                result = resultE
                                
                        # 进一步优化最佳结果
                        resultE = scipy.optimize.minimize(fitEllipse, result.x, method='nelder-mead', options={'xtol': 1e-6})
                        
                        # 创建最终椭圆形状
                        finalShape = makeEllipse(result.x)
                        
                        # 写入形状并更新路径编号
                        wayNumber = writeShape(wayNumber, finalShape, image, bbTop, bbHeight, bbLeft, bbWidth)
                        
                    elif (featureType == "basketball" or featureType == "tennis"):
                        # 篮球场和网球场通常是矩形
                        
                        # 定义矩形形状生成函数
                        def makeRect(paramsX):
                            """
                            根据参数生成矩形形状。
                            
                            参数:
                                paramsX (list): [centerX, centerY, width, height, angle]
                                
                            返回:
                                shapely.geometry: 矩形形状
                            """
                            centerX = paramsX[0]
                            centerY = paramsX[1]
                            width = abs(paramsX[2])
                            height = abs(paramsX[3])
                            angle = paramsX[4]
                            
                            # 创建矩形点
                            pts = [(width/2, -height/2),
                                   (width/2, height/2),
                                   (-width/2, height/2),
                                   (-width/2, -height/2)]
                                   
                            # 创建形状并应用变换
                            fitShape = geometry.LineString(pts)
                            fitShape = affinity.rotate(fitShape, angle, use_radians=True)
                            fitShape = affinity.translate(fitShape, centerX, centerY)
                            
                            return fitShape
                            
                        # 定义矩形拟合误差函数
                        def fitRect(paramsX):
                            """
                            计算矩形拟合误差。
                            
                            参数:
                                paramsX (list): 矩形参数
                                
                            返回:
                                float: 拟合误差
                            """
                            fitShape = makeRect(paramsX)
                            
                            sum = 0
                            # 计算轮廓点到矩形的距离平方和
                            for cnt in rawContours:
                                for pt in cnt:
                                    p = geometry.Point(pt[0])
                                    d = p.distance(fitShape)
                                    sum += d*d
                            return sum
                            
                        # 计算轮廓的中心点
                        cm = np.mean(rawContours[0], axis=0)
                        
                        # 尝试不同的初始角度，找到最佳拟合
                        result = {}
                        angleStepCount = 8
                        for angleI in range(angleStepCount):
                            centerX = cm[0, 0]
                            centerY = cm[0, 1]
                            width = math.sqrt(cv2.contourArea(rawContours[0]))
                            height = width
                            angle = 2*math.pi * float(angleI)/angleStepCount
                            x0 = np.array([centerX, centerY, width, height, angle])
                            resultR = scipy.optimize.minimize(fitRect, x0, method='nelder-mead', options={'xtol': 1e-6, 'maxiter': 50})
                            
                            if (angleI == 0):                            
                                result = resultR
                                
                            if (resultR.fun < result.fun):
                                result = resultR
                                
                        # 进一步优化最佳结果
                        resultR = scipy.optimize.minimize(fitRect, resultR.x, method='nelder-mead', options={'xtol': 1e-6})
                        
                        # 创建最终矩形形状
                        finalShape = makeRect(result.x)
                        
                        # 写入形状并更新路径编号
                        wayNumber = writeShape(wayNumber, finalShape, image, bbTop, bbHeight, bbLeft, bbWidth)



                
                


    

 
    

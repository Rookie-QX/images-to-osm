#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OSM异常审查脚本

本脚本用于审查由神经网络检测到的潜在OSM异常（缺失的运动场）。
它显示检测到的运动场的图像，并允许用户接受或拒绝这些建议。
接受的建议将在后续步骤中导入到OpenStreetMap。

作者: 原作者
日期: 2017
"""

import os
import sys
import glob
import imagestoosm.config as osmcfg  # 项目配置
import xml.etree.ElementTree as ET  # XML解析
import QuadKey.quadkey as quadkey  # 地图瓦片坐标转换
import shapely.geometry as geometry  # 几何处理
import matplotlib.pyplot as plt  # 图像显示
import skimage.io  # 图像读取
import shutil  # 文件操作

def _find_getch():
    """
    查找适合当前操作系统的getch函数实现。
    
    getch函数用于从控制台读取单个字符，无需按回车键。
    
    返回:
        function: 适合当前操作系统的getch函数
    """
    try:
        import termios
    except ImportError:
        # 非POSIX系统。返回msvcrt的(Windows)getch
        import msvcrt
        return msvcrt.getch

    # POSIX系统。创建并返回一个操作tty的getch
    import sys, tty
    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    return _getch

# 初始化getch函数
getch = _find_getch()

# 定义文件路径
addDirectory = os.path.join("anomaly", "add", "*.osm")  # 添加的OSM文件目录
anomalyStatusFile = os.path.join("anomaly", "status.csv")  # 异常状态文件

# 读取OSM文件，转换为像素坐标(z18)，创建shapely多边形

newWays = {}  # 存储新的路径数据
for osmFileName in glob.glob(addDirectory): 
    # 提取文件路径和文件名
    (path, filename) = os.path.split(osmFileName)
    wayNumber = os.path.splitext(filename)[0]

    # 创建新条目
    newEntry = { 
        "imageName": os.path.join(path, str(wayNumber) + ".jpg"),
        "osmFile": osmFileName,
        "tags": {},
        "status": ""
    }

    # 解析OSM XML文件
    tree = ET.parse(osmFileName)
    root = tree.getroot()

    # 提取标签
    for tag in root.findall('./way/tag'):
        key = tag.attrib["k"]
        val = tag.attrib["v"]
        newEntry['tags'][key] = val

    # 提取节点坐标并转换为像素坐标
    pts = []
    for node in root.iter('node'):
        pt = (float(node.attrib['lat']), float(node.attrib['lon']))
        pixel = quadkey.TileSystem.geo_to_pixel(pt, osmcfg.tileZoom)
        pts.append(pixel)

    # 如果有足够的点，创建多边形
    if (len(pts) > 2):
        newEntry["geometry"] = geometry.Polygon(pts)
        newWays[osmFileName] = newEntry

# 读取审查文件，获取状态(接受或拒绝)和OSM文件路径
if (os.path.exists(anomalyStatusFile)):
    with open(anomalyStatusFile, "rt", encoding="ascii") as f: 
        for line in f:
            (status, osmFileName) = line.split(',')
            osmFileName = osmFileName.strip()
            newWays[osmFileName]['status'] = status

# 创建图形窗口
fig = plt.figure()

# 遍历所有路径进行审查
for wayKey in sorted(newWays):
    way = newWays[wayKey]

    # 如果尚未审查，则进行审查
    if (len(way['status']) == 0):
        # 设置子图布局
        subPlotCols = 2
        subPlotRows = 2
        maxSubPlots = subPlotCols * subPlotRows

        # 查找相关的路径（相同标签且几何相交）
        reviewSet = [way]
        for otherKey in sorted(newWays):
            other = newWays[otherKey]
            if (other != way and len(other['status']) == 0 and way['tags'] == other['tags'] and other['geometry'].intersects(way['geometry'])):
                reviewSet.append(other)

        # 默认将所有路径标记为拒绝
        for wayIndex in range(len(reviewSet)):
            other = reviewSet[wayIndex]
            other['status'] = 'rejected'

        acceptedWay = {}  # 存储接受的路径
        viewSet = []  # 当前视图中的路径集合
        
        # 分批显示路径以供审查
        for wayIndex in range(len(reviewSet)):
            viewSet.append(reviewSet[wayIndex])

            # 当达到最大子图数量或已处理所有路径时，显示当前批次
            if (len(viewSet) == maxSubPlots or wayIndex + 1 >= len(reviewSet)):
                # 清除所有子图
                for plotIndex in range(maxSubPlots):
                    sb = fig.add_subplot(subPlotRows, subPlotCols, plotIndex + 1)
                    sb.cla()

                # 在子图中显示每个路径的图像
                for wayIndex in range(len(viewSet)):
                    fig.add_subplot(subPlotRows, subPlotCols, wayIndex + 1)
                    plt.title("{} {}".format(wayIndex + 1, viewSet[wayIndex]['osmFile']))
                    image = skimage.io.imread(viewSet[wayIndex]['imageName'])
                    plt.imshow(image)

                # 显示图形并等待用户输入
                plt.show(block=False)
                plt.pause(0.05)

                goodInput = False
                while goodInput == False:
                    print("{} - q退出, 0拒绝所有, 使用子图索引接受".format(way['osmFile']))
                    c = getch()

                    try:
                        index = int(c)
                        if (index > 0):
                            # 接受选定的路径
                            acceptedWay = viewSet[index - 1]
                            viewSet = [acceptedWay]
                            print("已选择 {} {}".format(acceptedWay['osmFile'], index))
                            goodInput = True
                        if (index == 0):
                            # 拒绝所有路径
                            viewSet = [reviewSet[0]]
                            acceptedWay = {}
                            print("拒绝所有")
                            goodInput = True
                                                    
                    except:
                        if (c == "q"):
                            sys.exit(0)
                        print("输入无效")

        # 如果有接受的路径，更新其状态
        if (bool(acceptedWay)):
            acceptedWay['status'] = 'accepted'
            print("已接受 {}".format(acceptedWay['osmFile']))
        
        # 备份状态文件
        if (os.path.exists(anomalyStatusFile + ".1")):
            shutil.copy(anomalyStatusFile + ".1", anomalyStatusFile + ".2")
        if (os.path.exists(anomalyStatusFile)):
            shutil.copy(anomalyStatusFile, anomalyStatusFile + ".1")

        # 写入更新后的状态文件
        with open(anomalyStatusFile, "wt", encoding="ascii") as f: 
            for otherKey in sorted(newWays):
                other = newWays[otherKey]
                if (len(other['status']) > 0):
                    f.write("{},{}\n".format(other['status'], other['osmFile']))


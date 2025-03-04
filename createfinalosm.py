#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建最终OSM文件模块

本模块用于处理已接受的OSM异常检测结果，将它们合并成最终的OSM文件，
以便上传到OpenStreetMap。同时处理节点和路径ID，确保它们在文件中是唯一的。

"""

import os
import sys
import imagestoosm.config as osmcfg  # 项目配置
import xml.etree.ElementTree as ET  # XML解析
import shapely.geometry as geometry  # 几何处理

def makeOsmFileName(fileNumber):
    """
    生成OSM文件名。
    
    参数:
        fileNumber (int): 文件编号
        
    返回:
        str: OSM文件的完整路径
    """
    return os.path.join("anomaly", "reviewed_{:02d}.osm".format(fileNumber))

# 清除旧的OSM文件
fileCount = 1
while (os.path.exists(makeOsmFileName(fileCount))):
    os.remove(makeOsmFileName(fileCount))
    fileCount += 1
fileCount = 1

# 除了将单个接受的OSM文件组合在一起，此脚本
# 还需要修复负/占位符ID，使其在文件中唯一
startId = 0

# 异常状态文件路径
anomalyStatusFile = os.path.join("anomaly", "status.csv")

# 创建OSM根元素
osmTreeRoot = ET.Element('osm')
osmTreeRoot.attrib['version'] = "0.6"

# 处理已接受的异常
if (os.path.exists(anomalyStatusFile)):
    with open(anomalyStatusFile, "rt", encoding="ascii") as f: 
        for line in f:
            (status, osmFileName) = line.split(',')                
            osmFileName = osmFileName.strip()

            # 只处理已接受的异常
            if (status == "accepted"):
                # 解析OSM文件
                tree = ET.parse(osmFileName)
                root = tree.getroot()

                wayCount = 0
                # 处理节点，修复ID
                for node in root.iter('node'):
                    node.attrib['id'] = "{0:d}".format(int(node.attrib['id']) - startId)
                    wayCount += 1
                    osmTreeRoot.append(node)
                
                # 处理路径中的节点引用
                for node in root.findall('./way/nd'):
                    node.attrib['ref'] = "{0:d}".format(int(node.attrib['ref']) - startId)

                # 处理路径，修复ID
                for node in root.iter('way'):
                    node.attrib['id'] = "{0:d}".format(int(node.attrib['id']) - startId)
                    wayCount += 1
                    osmTreeRoot.append(node)

                # 更新ID偏移量
                startId += wayCount

                # OSM只允许在单个变更集上传中包含10,000个元素，创建不同的OSM文件
                # 以便它们可以上传而不会出错
                if (startId > 9500):
                    tree = ET.ElementTree(osmTreeRoot)
                    tree.write(makeOsmFileName(fileCount))
                    fileCount += 1

                    # 创建新的OSM根元素
                    osmTreeRoot = ET.Element('osm')
                    osmTreeRoot.attrib['version'] = "0.6"
                    startId = 0

# 保存最终的OSM文件
tree = ET.ElementTree(osmTreeRoot)
tree.write(makeOsmFileName(fileCount))



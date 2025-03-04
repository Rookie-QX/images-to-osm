#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从OpenStreetMap下载运动场数据模块

本模块使用Overpass API从OpenStreetMap下载指定区域内的运动场数据，
包括棒球场、网球场、足球场、美式足球场和篮球场。数据将被保存为
CSV和GeoJSON格式，以便后续处理。

"""

import sys
import overpy  # Overpass API的Python接口
import imagestoosm.config as cfg  # 项目配置
import os
import shapely.geometry  # 用于处理地理空间几何
import shapely.wkt
import shapely.ops
import geojson  # 用于处理GeoJSON格式

# 初始化Overpass API客户端
api = overpy.Overpass()

# 用于统计各类运动场数量的字典
summary = {}

def saveOsmData(query):
    """
    执行Overpass查询并保存结果数据
    
    该函数执行指定的Overpass查询，获取运动场数据，
    并将结果保存为CSV和GeoJSON格式。同时更新统计信息。
    
    参数:
        query (str): Overpass API查询字符串
    
    返回:
        无返回值，但会将数据保存到文件系统
    """
    # 执行查询
    result = api.query(query)

    for way in result.ways:
        # 获取运动类型作为目录名
        # "leisure=pitch,sport=" , 不要在featureDirectoryName中使用"-"字符
        featureDirectoryName = way.tags.get("sport")

        # 为每种运动类型创建输出目录
        outputDirectoryName = os.path.join(cfg.rootOsmDir, featureDirectoryName)
        if (os.path.exists(outputDirectoryName) == False):
            os.makedirs(outputDirectoryName)

        # 更新统计计数
        if ((featureDirectoryName in summary) == False):
            summary[featureDirectoryName] = 1
        else:     
            summary[featureDirectoryName] += 1
        
        # 生成基础文件名（使用OSM way ID）
        filenameBase = os.path.join(cfg.rootOsmDir, featureDirectoryName, str(way.id))

        # 调试输出
        #print("Name: %d %s %s" % (way.id, way.tags.get("name", ""), filenameBase))

        # 保存为CSV格式
        # 保留CSV文件，直到下一阶段的脚本重写
        with open("%s.csv" % (filenameBase), "wt") as text_file:
            for node in way.nodes:
                text_file.write("%0.7f\t%0.7f\n" % (node.lat, node.lon))

        # 保存为GeoJSON格式
        with open("%s.GeoJSON" % (filenameBase), "wt") as text_file:
            # 收集所有节点坐标 (注意：GeoJSON使用经度,纬度顺序)
            rawNodes = []
            for node in way.nodes:
                rawNodes.append((node.lon, node.lat))
            
            try:
                # 创建多边形几何体
                geom = shapely.geometry.Polygon(rawNodes)

                # 添加所有标签和way ID到属性
                tags = way.tags
                tags['wayOSMId'] = way.id

                # 创建GeoJSON特征集合
                features = []            
                features.append(geojson.Feature(geometry=geom, properties=tags))
                featureC = geojson.FeatureCollection(features)

                # 保存GeoJSON数据到文件
                text_file.write(geojson.dumps(featureC))
            except Exception as e:
                print("处理way ID %s时出错: %s" % (way.id, e))


# 查询多个州的运动场数据
# 包括马萨诸塞州、纽约州、康涅狄格州、罗德岛和宾夕法尼亚州
queryFull = """[timeout:125];
    ( 
        area[admin_level=4][boundary=administrative][name="Massachusetts"];         
        area[admin_level=4][boundary=administrative][name="New York"];         
        area[admin_level=4][boundary=administrative][name="Connecticut"];         
        area[admin_level=4][boundary=administrative][name="Rhode Island"]; 
        area[admin_level=4][boundary=administrative][name="Pennsylvania"];                               
    )->.searchArea;
    (
    way["sport"="baseball"]["leisure"="pitch"](area.searchArea);
    way["sport"="tennis"]["leisure"="pitch"](area.searchArea);
    way["sport"="soccer"]["leisure"="pitch"](area.searchArea);
    way["sport"="american_football"]["leisure"="pitch"](area.searchArea);    
    way["sport"="basketball"]["leisure"="pitch"](area.searchArea);    
    );
    (._;>;);
    out body;
    """

# 仅查询马萨诸塞州的运动场数据
# 用于测试或者专注于单个州的数据
queryMA = """[timeout:125];
    ( 
        area[admin_level=4][boundary=administrative][name="Massachusetts"];         
    )->.searchArea;
    (
    way["sport"="baseball"]["leisure"="pitch"](area.searchArea);
    way["sport"="tennis"]["leisure"="pitch"](area.searchArea);
    way["sport"="soccer"]["leisure"="pitch"](area.searchArea);
    way["sport"="american_football"]["leisure"="pitch"](area.searchArea);    
    way["sport"="basketball"]["leisure"="pitch"](area.searchArea);    
    );
    (._;>;);
    out body;
    """

# 执行完整查询 - 获取所有指定州的数据
saveOsmData(queryFull)
    
# 其他可能查询的数据 - 未来扩展的潜在目标
#  - 桥梁
#  - 太阳能电池板农场
#  - 风力涡轮机 
#  - 铁路交叉口
#  - 活跃的铁路
#  - 水塔
#  - 水域/湖泊/河流
#  - 停车场
#  - 车道
#  - 加油站
#  - 建筑物（微软已经完成了这项工作）
#  - 跑道

# 打印各类运动场的数量统计
print(summary)

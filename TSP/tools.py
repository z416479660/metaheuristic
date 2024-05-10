# -*- coding: utf-8 -*-
"""
	基于贪心算法的旅行商问题解法Python源码

	Author:	Greatpan
	Date:	2018.9.30
"""
import pandas
import numpy as np
import math
import matplotlib.pyplot as plt
import time


# class Node:
#     """
#         类名：Node
#         类说明：	城市节点类
#     """
#
#     def __init__(self, CityNum):
#         """
#         函数名：GetData()
#         函数功能：	从外界读取城市数据并处理
#         输入	无
#         输出	1 Position：各个城市的位置矩阵
#         2 CityNum：城市数量
#         3 Dist：城市间距离矩阵
#         其他说明：无
#         """
#         self.visited = [False] * CityNum  # 记录城市是否走过
#         self.start = 0  # 起点城市
#         self.end = 0  # 目标城市
#         self.current = 0  # 当前所处城市
#         self.num = 0  # 走过的城市数量
#         self.pathsum = 0  # 走过的总路程
#         self.lb = 0  # 当前结点的下界
#         self.listc = []  # 记录依次走过的城市


def GetData(datapath):
    """
    读取城市数据并处理
    :param datapath: 文件路径
    :return:
    1 Position：各个城市的位置矩阵
    2 CityNum：城市数量
    3 Dist：城市间距离矩阵
    """
    dataframe = pandas.read_csv(datapath, sep=" ", header=None)
    Cities = dataframe.iloc[:, 1:3]
    Position = np.array(Cities)  # 从城市A到B的距离矩阵
    CityNum = Position.shape[0]  # CityNum:代表城市数量
    Dist = np.zeros((CityNum, CityNum))  # Dist(i,j)：城市i与城市j间的距离

    # 计算距离矩阵
    for i in range(CityNum):
        for j in range(CityNum):
            if i == j:
                Dist[i, j] = math.inf
            else:
                Dist[i, j] = math.sqrt(np.sum((Position[i, :] - Position[j, :]) ** 2))
    return Position, CityNum, Dist


def ResultShow(Min_Path, BestPath, CityNum, method=None):
    """

    :param Min_Path: 最优解TSP距离
    :param BestPath: 最优路径
    :param CityNum: TSP城市数量
    :param method: 算法
    :return:
    """
    if method:
        print("基于" + method + "求得的旅行商最短路径为：")
    for m in range(CityNum):
        print(str(BestPath[m]) + "—>", end="")
    print(BestPath[CityNum])
    print("总路径长为：" + str(Min_Path))


def draw(BestPath, Position, title):
    """
    绘制旅行商最优路径
    :param BestPath: 最优路径
    :param Position: 各个城市的坐标矩阵
    :param title: 图表的标题
    :return:
    """
    plt.title(title)
    plt.plot(Position[:, 0], Position[:, 1], 'bo')
    for i, city in enumerate(Position):
        plt.text(city[0], city[1], str(i))
    plt.plot(Position[BestPath, 0], Position[BestPath, 1], color='red')
    plt.show()


def PlotConvergence(iters, values):
    plt.plot(iters, values, '-r')
    plt.show()
def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.6f}s to run.")
        return result
    return wrapper

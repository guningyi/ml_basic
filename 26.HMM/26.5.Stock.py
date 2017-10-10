# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics.pairwise import pairwise_distances_argmin
import warnings


def expand(a, b):
    d = (b - a) * 0.05
    return a-d, b+d


if __name__ == "__main__":
    warnings.filterwarnings("ignore")   # hmmlearn(0.2.0) < sklearn(0.18)

    # 0日期  1开盘  2最高  3最低  4收盘  5成交量  6成交额
    # 26.SH6000000.txt中每一列的含义,经过
    # usecols = (4,5,6,2,3) 参数设置之后，在x中的存储为：
    # 0收盘价 1成交量 2成交额 3最高价 4最低价
    x = np.loadtxt('26.SH600000.txt', delimiter='\t', skiprows=2, usecols=(4, 5, 6, 2, 3))
    close_price = x[:, 0]  #取出所有的收盘价
    volumn = x[:, 1]   #取出所有的成交量
    amount = x[:, 2]   #取出所有的成交额
    amplitude_price = x[:, 3] - x[:, 4] # 每天的最高价与最低价的差
    diff_price = np.diff(close_price)   # 涨跌值 np.diff([1,3,4,6,10]) -> [2,1,2,4] , diff是用数组的array[i] - array[i+1], 遍历整个数组。
    volumn = volumn[1:]                 # 去掉成交量中的第一个数据
    amount = amount[1:]                 # 去掉成交额中的第一个数据
    amplitude_price = amplitude_price[1:]   # 去掉每日振幅中的第一个数据

    # np.column_stack 是将输入的一个由一维数组组成的tuple合成为二维数组
    # 例如： A = ([1,2,3,4,5]) B=([-1,-2,-3,-4,-5])
    # np.column_stack((A,B))
    #  -> ([[1,-1],[2,-2],[3,-3],[4,-4],[5,-5]])

    # 将整理出来的 [涨跌值],[成交量],[成交额],[每日振幅]　代入column_stack()中合成二维数组
    #  [[涨跌值1,成交量1,成交额1,每日振幅1], [涨跌值2,成交量2,成交额2,每日振幅2],....
    #    .......
    #    .....[涨跌值n,成交量n,成交额n,每日振幅n]]
    #  这个数据作为观测值
    sample = np.column_stack((diff_price, volumn, amount, amplitude_price))
    n = 5  # 设置5个隐状态
    # convariance_type(协方差类型) full：是指在每个马尔可夫隐含状态下，可观察态向量使用完全协方差矩阵。对应的协方差矩阵里面的元素都是不为零。
    # convariance_type取不同的值代表了使用不同的高斯概率密度分布函数
    # n_components 隐状态个数
    model = hmm.GaussianHMM(n_components=n, covariance_type='full')
    model.fit(sample)
    # Compute the posterior probability for each state in the model
    # 计算该模型中每个状态的的后验概率
    y = model.predict_proba(sample)
    np.set_printoptions(suppress=True)
    print y

    # predict : Find most likely state sequence corresponding to sample
    # 用来寻找针对观察序列sample的最有可能的隐状态序列z
    # z = model.predict(sample)
    # print z

    t = np.arange(len(diff_price))
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10,8), facecolor='w')
    plt.subplot(421)
    plt.plot(t, diff_price, 'r-')
    plt.grid(True)
    plt.title(u'涨跌幅')
    plt.subplot(422)
    plt.plot(t, volumn, 'g-')
    plt.grid(True)
    plt.title(u'交易量')

    # 创建了一个颜色库？
    # np.linspace(0, 0.8, n) -> [ 0. ,  0.2,  0.4,  0.6,  0.8]
    # plt.cm.terrain(np.linspace([ 0. ,  0.2,  0.4,  0.6,  0.8]) ->
    # [[ 0.2  ,  0.2  ,  0.6  ,  1.   ],
    #   [ 0.   ,  0.7  ,  0.7  ,  1.   ],
    #   [ 0.6  ,  0.92 ,  0.52 ,  1.   ],
    #   [ 0.8  ,  0.744,  0.492,  1.   ],
    #   [ 0.6  ,  0.488,  0.464,  1.   ]]

    clrs = plt.cm.terrain(np.linspace(0, 0.8, n))
    plt.subplot(423)
    # 遍历clrs, 这个二维数组的一个维度的长度是5,每个元素是一组色彩值。
    # 每个状态的后验概率用y[:,i]取出来
    # 二维数组中y[:,i]是按列取
    # y[:, 2] ->  [ 0.6  ,  0.7  ,  0.52 ,  0.492,  0.464]

    # y[i,]是按照行取
    # y[2,] -> [ 0.6 ,  0.92,  0.52,  1.  ]

    for i, clr in enumerate(clrs):
        plt.plot(t, y[:, i], '-', color=clr, alpha=0.7)
    plt.title(u'所有组分')
    plt.grid(True)
    for i, clr in enumerate(clrs):
        axes = plt.subplot(4, 2, i+4)
        plt.plot(t, y[:, i], '-', color=clr)
        plt.title(u'组分%d' % (i+1))
        plt.grid(True)
    plt.suptitle(u'SH600000股票：GaussianHMM分解隐变量', fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

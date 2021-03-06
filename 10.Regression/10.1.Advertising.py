#!/usr/bin/python
# -*- coding:utf-8 -*-

import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from pprint import pprint


if __name__ == "__main__":
    path = '10.Advertising.csv'
    # # 手写读取数据 - 请自行分析，在10.2.Iris代码中给出类似的例子
    # f = file(path)
    # x = []
    # y = []
    # for i, d in enumerate(f):
    #     if i == 0:
    #         continue
    #     d = d.strip()
    #     if not d:
    #         continue
    #     d = map(float, d.split(','))
    #     x.append(d[1:-1])
    #     y.append(d[-1])
    # pprint(x)
    # pprint(y)
    # x = np.array(x)
    # y = np.array(y)

    # # Python自带库
    # f = file(path, 'rb')
    # print f
    # d = csv.reader(f)
    # for line in d:
    #     print line
    # f.close()

    # # numpy读入
    # p = np.loadtxt(path, delimiter=',', skiprows=1)
    # print p
    # print '\n\n===============\n\n'

    # pandas读入
    data = pd.read_csv(path)    # TV、Radio、Newspaper、Sales
    # x = data[['TV', 'Radio', 'Newspaper']]
    x = data[['TV', 'Radio']]
    y = data['Sales']
    print x
    print y

    # rc配置变量称为matplotlib.rcParams
    # 可以修改其默认的配置变量
    mpl.rcParams['font.sans-serif'] = [u'simHei']  # 指定默认文本
    mpl.rcParams['axes.unicode_minus'] = False     # 解决保存图像是负号'-'显示为方块的问题

    # # 绘制1
    # plt.plot(data['TV'], y, 'ro', label='TV')
    # plt.plot(data['Radio'], y, 'g^', label='Radio')
    # plt.plot(data['Newspaper'], y, 'mv', label='Newspaer')
    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.show()
    # #
    # 绘制2
    plt.figure(figsize=(9,12)) # 参数9,12指定高度和宽度，默认的dpi是80象素，即宽度是9*80象素,长度12*80象素
    plt.subplot(311) #绘制子图
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')
    plt.grid()
    plt.subplot(312)
    plt.plot(data['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid()
    plt.subplot(313)
    plt.plot(data['Newspaper'], y, 'b*')
    plt.title('Newspaper')
    plt.grid()
    plt.tight_layout() # 紧凑显示图片，居中显示
    plt.show()

    # 功能是从样本中随机的按比例选取train data和test data
    # x：所要划分的样本特征集
    # y：所要划分的样本结果
    # train_size：样本占比，如果是整数的话就是样本的数量
    # random_state：是随机数的种子。
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    print 'x_train:\n',x_train
    print 'y_train:\n',y_train
    linreg = LinearRegression()  #sklearn 实例化
    # 使用fit函数进行回归计算，参数:x_train,y_train以矩阵的方式传入，而第三个参数sample_weight则是每条测试数据的权重，同样以array格式传入。
    # 得到线性回归模型
    model = linreg.fit(x_train, y_train)
    print model
    # # LinearRegression将方程分为两个部分存放，
    # # coef_存放回归系数，
    # # intercept_则存放截距，
    # # 因此要查看方程，就是查看这两个变量的取值。
    print linreg.coef_
    print linreg.intercept_

    # # 预测方法，将返回预测值y_hat
    # # 使用测试集去得到预测值
    y_hat = linreg.predict(np.array(x_test))

    # 我们需要评估我们的模型的好坏程度，对于线性回归来说，我们一般用均方差（Mean SquaredError, MSE）
    # 或者均方根差(Root Mean SquaredError, RMSE)在测试集上的表现来评价模型的好坏。
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error 均方差
    rmse = np.sqrt(mse)  # Root Mean Squared Error 均方根差
    print mse, rmse

    # # 显示图像
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.legend(loc='upper right')
    plt.title(u'线性回归预测销量', fontsize=18)
    plt.grid()
    plt.show()

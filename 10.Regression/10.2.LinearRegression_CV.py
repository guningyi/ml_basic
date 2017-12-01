#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":
    # pandas读入
    data = pd.read_csv('10.Advertising.csv')    # TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio', 'Newspaper']]
    # x = data[['TV', 'Radio']]
    y = data['Sales']
    print x
    print y

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    #获取Lasso的模型
    model = Lasso()
    # model = Ridge()
    alpha_can = np.logspace(-3, 2, 10)
    # print 'alpha_can:',alpha_can
    # # alpha_can 的值
    # #  [  1.00000000e-03   3.59381366e-03   1.29154967e-02   4.64158883e-02
    # #     1.66810054e-01   5.99484250e-01   2.15443469e+00   7.74263683e+00
    # #     2.78255940e+01   1.00000000e+02]

    # # 1.00000000e-03 是科学计数法，表示　1.00000000 * 10 ^ (-3)
    # # 3.59381366e-03 表示　3.59381366 * 10 ^ (-3)　

    # 使用np.linspace(-10,10,10)作为参数是否可行？
    # alpha_can = np.linspace(-10,10,10)

    # GridSearchCV通过交叉验证对参数空间进行求解，寻找最佳的参数。
    # param_grid就是我们要交叉验证的参数
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    # 用训练集数据学习(训练)
    lasso_model.fit(x_train, y_train)
    print '超参数：\n', lasso_model.best_params_

    # 用测试集数据预测
    y_hat = lasso_model.predict(np.array(x_test))
    print lasso_model.score(x_test, y_test)

    # # 评估模型的优劣
    # 损失函数 1:平方和损失
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error  1/n*∑(y_hat-y_test)^2
    # 损失函数 2
    rmse = np.sqrt(mse)  # Root Mean Squared Error  √(1/n*∑(y_hat-y_test)^2)
    print mse, rmse

    # print 'y_test:\n', y_test
    # print 'np.array(y_test):\n', np.array(y_test)

    t = np.arange(len(x_test))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.title(u'线性回归预测销量', fontsize=18)
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

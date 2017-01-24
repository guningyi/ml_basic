#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from pprint import pprint


def restore1(sigma, u, v, K):  # 奇异值、左特征向量、右特征向量
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))
    for k in range(K):
        uk = u[:, k].reshape(m, 1)
        vk = v[k].reshape(1, n)
        a += sigma[k] * np.dot(uk, vk)
    a[a < 0] = 0
    a[a > 255] = 255
    # a = a.clip(0, 255)
    return np.rint(a).astype('uint8')


def restore2(sigma, u, v, K):  # 奇异值、左特征向量、右特征向量
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))
    for k in range(K+1):
        for i in range(m):
            a[i] += sigma[k] * u[i][k] * v[k]
    a[a < 0] = 0
    a[a > 255] = 255
    return np.rint(a).astype('uint8')


if __name__ == "__main__":
    A = Image.open("7.son.png", 'r')
    output_path = r'.\Pic'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    a = np.array(A)
    #图片数组
    #print a
    #图片数组的分片尺寸，可以看出RGB三个尺寸
    #print a.shape
    K = 50
    u_r, sigma_r, v_r = np.linalg.svd(a[:, :, 0])  #红色通道矩阵
    u_g, sigma_g, v_g = np.linalg.svd(a[:, :, 1])  #绿色通道矩阵
    u_b, sigma_b, v_b = np.linalg.svd(a[:, :, 2])  #黑色通道矩阵
    plt.figure(figsize=(10,10), facecolor='w')
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    for k in range(1, K+1):
        print k
        R = restore1(sigma_r, u_r, v_r, k)
        G = restore1(sigma_g, u_g, v_g, k)
        B = restore1(sigma_b, u_b, v_b, k)
        I = np.stack((R, G, B), 2)#axis=2 这里为什么要用axis=2?
        Image.fromarray(I).save('%s\\svd_%d.png' % (output_path, k))
        if k <= 12:
            plt.subplot(3, 4, k)
            plt.imshow(I)
            plt.axis('off')
            plt.title(u'奇异值个数：%d' % k)
    plt.suptitle(u'SVD与图像分解', fontsize=20)
    plt.tight_layout(2) #减少边缘留白的尺寸，但也不能太小，否则图像会重叠在一起，在这里仅仅起到调整输出的布局而已
    plt.subplots_adjust(top=0.9) #手动指定顶部的高度，和css中的类似
    plt.show()
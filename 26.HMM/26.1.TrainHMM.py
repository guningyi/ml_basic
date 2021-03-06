# !/usr/bin/python
# -*- coding:utf-8 -*-

import math
import matplotlib.pyplot as plt
import numpy as np
import codecs
import random

infinite = float(-2**31)


def log_normalize(a):
    s = 0
    for x in a:
        s += x
    if s == 0:
        print "Error..from log_normalize."
        return
    s = math.log(s)
    for i in range(len(a)):
        if a[i] == 0:
            a[i] = infinite
        else:
            a[i] = math.log(a[i]) - s


def log_sum(a):
    if not a:   # a为空
        return infinite
    m = max(a)
    s = 0
    for t in a:
        s += math.exp(t-m)
    return m + math.log(s)


def calc_alpha(pi, A, B, o, alpha):
    for i in range(4):
        alpha[0][i] = pi[i] + B[i][ord(o[0])]
    T = len(o)
    temp = [0 for i in range(4)]
    del i
    for t in range(1, T):
        for i in range(4):
            for j in range(4):
                temp[j] = (alpha[t-1][j] + A[j][i])
            alpha[t][i] = log_sum(temp)
            alpha[t][i] += B[i][ord(o[t])]


def calc_beta(pi, A, B, o, beta):
    T = len(o)
    for i in range(4):
        beta[T-1][i] = 1
    temp = [0 for i in range(4)]
    del i
    for t in range(T-2, -1, -1):
        for i in range(4):
            beta[t][i] = 0
            for j in range(4):
                temp[j] = A[i][j] + B[j][ord(o[t+1])] + beta[t+1][j]
            beta[t][i] += log_sum(temp)


def calc_gamma(alpha, beta, gamma):
    for t in range(len(alpha)):
        for i in range(4):
            gamma[t][i] = alpha[t][i] + beta[t][i]
        s = log_sum(gamma[t])
        for i in range(4):
            gamma[t][i] -= s


def calc_ksi(alpha, beta, A, B, o, ksi):
    T = len(alpha)
    temp = [0 for x in range(16)]
    for t in range(T-1):
        k = 0
        for i in range(4):
            for j in range(4):
                ksi[t][i][j] = alpha[t][i] + A[i][j] + B[j][ord(o[t+1])] + beta[t+1][j]
                temp[k] =ksi[t][i][j]
                k += 1
        s = log_sum(temp)
        for i in range(4):
            for j in range(4):
                ksi[t][i][j] -= s


def bw(pi, A, B, alpha, beta, gamma, ksi, o):
    T = len(alpha)
    for i in range(4):
        pi[i] = gamma[0][i]
    s1 = [0 for x in range(T-1)]
    s2 = [0 for x in range(T-1)]
    for i in range(4):
        for j in range(4):
            for t in range(T-1):
                s1[t] = ksi[t][i][j]
                s2[t] = gamma[t][i]
            A[i][j] = log_sum(s1) - log_sum(s2)
    s1 = [0 for x in range(T)]
    s2 = [0 for x in range(T)]
    for i in range(4):
        print "bw", i
        for k in range(65536):
            valid = 0
            if k % 10000 == 0:
                print "bw - k", k
            for t in range(T):
                if ord(o[t]) == k:
                    s1[valid] = gamma[t][i]
                    valid += 1
                s2[t] = gamma[t][i]
            if valid == 0:
                B[i][k] = infinite
            else:
                B[i][k] = log_sum(s1[:valid]) - log_sum(s2)


def baum_welch(pi, A, B):
    f = file(".\\1.txt")
    sentence = f.read()[3:].decode('utf-8')
    f.close()
    T = len(sentence)
    alpha = [[0 for i in range(4)] for t in range(T)]
    print alpha
    beta = [[0 for i in range(4)] for t in range(T)]
    gamma = [[0 for i in range(4)] for t in range(T)]
    ksi = [[[0 for j in range(4)] for i in range(4)] for t in range(T-1)]
    for time in range(3):
        print "calc_alpha"
        calc_alpha(pi, A, B, sentence, alpha)    # alpha(t,i):给定lamda，在时刻t的状态为i且观测到o(1),o(2)...o(t)的概率
        print "calc_beta"
        calc_beta(pi, A, B, sentence, beta)      # beta(t,i)：给定lamda和时刻t的状态i，观测到o(t+1),o(t+2)...oT的概率
        print "calc_gamma"
        calc_gamma(alpha, beta, gamma)    # gamma(t,i)：给定lamda和O，在时刻t状态位于i的概率
        print "calc_ksi"
        calc_ksi(alpha, beta, A, B, sentence, ksi)    # ksi(t,i,j)：给定lamda和O，在时刻t状态位于i且在时刻i+1，状态位于j的概率
        print "bw"
        bw(pi, A, B, alpha, beta, gamma, ksi, sentence)
        print "time", time
        print "Pi:", pi
        print "A", A


def mle():  # 0B/1M/2E/3S
    pi = [0] * 4   # npi[i]：i状态的个数
    a = [[0] * 4 for x in range(4)]     # na[i][j]：从i状态到j状态的转移个数. a~转移矩阵
    b = [[0]* 65536 for x in range(4)]  # nb[i][o]：从i(某个隐状态)到o（某个字符）的个数, 4行,65536列. b~发射矩阵
    f = file(".\\26.pku_training.utf8")
    data = f.read()[3:].decode('utf-8') # 因为这个文件之前在windows下被打开过，所以需要作一些处理和转换，变成UNIX的编码形式。
    f.close()
    tokens = data.split('  ')
    # 增加英文词训练集
    f = file('26.Englishword.train')
    data = f.read().decode('utf-8')
    f.close()
    tokens.extend(data.split(' '))

    # 开始训练
    last_q = 2
    iii = 0
    old_progress = 0
    # 这里的k只是为了输出进度所用，
    print '进度：'
    for k, token in enumerate(tokens):
        progress = float(k) / float(len(tokens))
        if progress > old_progress + 0.1:
            print '%.3f%%' % (progress * 100)
            old_progress = progress
        token = token.strip() #过滤掉回车符号
        n = len(token)
        if n <= 0:  #如果词的长度小于等于0，那这个词就是个空格，直接忽略掉。
            continue
        if n == 1: #词的长度为1，单字成词
            pi[3] += 1  #单字出现的次数增加1
            a[last_q][3] += 1   # 上一个词的结束(last_q)到当前状态(3)出现的次数增加一，相当于在转移矩阵a中某个位置更新值，作+1操作
            b[3][ord(token[0])] += 1   # 从b[3]隐状态到token[0]这个字符的次数增加一
            last_q = 3 # 对于下一次的last_q而言，它的值是3。 也就是说：当下一次计算时向前看，其前一次的last_q是3
            continue
        # 初始向量
        # 假如不是单字成词，那它必然有begin ,end, 有没有middle暂且不说。
        pi[0] += 1 # begin 的次数加1
        pi[2] += 1 # end 的次数加1
        pi[1] += (n-2) # middle 的次数暂且定位为 (n-2)， 随着循环的进行，会不断更新这个。
        # 转移矩阵
        a[last_q][0] += 1 #上一个状态(隐变量)转移到0(起始位置 begin)的次数加1
        last_q = 2 # 对于下一次计算来说，last_q 就是2，
        if n == 2:
            a[0][2] += 1 # 假如词长为2 ，那么转移矩阵就是 begin -> end. a[0][2]加1
        else:
            a[0][1] += 1        # a[0][1] 从begin -> middle
            a[1][1] += (n-3)    # a[1][1] middle -> middle ~ 若干个 middle -> middle
            a[1][2] += 1        # a[1][2] middle -> end
        # 发射矩阵
        b[0][ord(token[0])] += 1 # 从b[0]隐状态到token[0]这个字符的次数加1 token[0]是首字符~begin
        b[2][ord(token[n-1])] += 1  # 从b[2]隐状态到token[n-1]这个字符的次数加1 token[n-1]是尾字符~end
        for i in range(1, n-1):
            b[1][ord(token[i])] += 1  # 从b[1]隐状态到token[i], i~[1, n-1]这些字符的次数分别加1 token[i] 是若干个中间字符，次数分别加1
    # 正则化
    log_normalize(pi)
    for i in range(4):
        log_normalize(a[i])  #按行正则化
        log_normalize(b[i])
    return [pi, a, b]


def list_write(f, v):
    for a in v:
        f.write(str(a))
        f.write(' ')
    f.write('\n')


def save_parameter(pi, A, B):
    f_pi = open(".\\pi.txt", "w")
    list_write(f_pi, pi)
    f_pi.close()
    f_A = open(".\\A.txt", "w")
    for a in A:
        list_write(f_A, a)
    f_A.close()
    f_B = open(".\\B.txt", "w")
    for b in B:
        list_write(f_B, b)
    f_B.close()


if __name__ == "__main__":
    pi, A, B = mle()
    print A
    save_parameter(pi, A, B)
    print "训练完成..."

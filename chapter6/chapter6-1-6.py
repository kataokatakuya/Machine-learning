"""
6.1  1次元入力2クラス分類
"""
"""
6.1.6  学習則の導出
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# データ生成
np.random.seed(seed=0)
X_min = 0
X_max = 2.5
X_n = 30
X = np.zeros(X_n)   # 入力データ

T = np.zeros(X_n, dtype=np.uint8)
Dist_s = [0.4, 0.8]   # 分布の開始地点
Dist_w = [0.8, 1.6]   # 分布の幅
Pi = 0.5   # クラス0の比率

for n in range(X_n):
    wk = np.random.rand()
    T[n] = 0 * (wk < Pi) + 1 * (wk >= Pi)
    X[n] = np.random.rand() * Dist_w[T[n]] + Dist_s[T[n]]


# シグモイド関数
def logistic(x, w):
    y = 1 / (1 + np.exp(-(w[0] * x + w[1])))
    return y



# 平均交差エントロピー誤差
def dcee_logistic(w, x, t):
    y = logistic(x, w)
    dcee = np.zeros(2)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t[n]) * x[n]
        dcee[1] = dcee[1] + (y[n] - t[n])
    dcee = dcee / X_n
    return dcee


W = [1, 1]
print(dcee_logistic(W, X, T))








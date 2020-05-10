"""
6.3  2次元入力3クラス分類
"""
"""
6.3.2  交差エントロピー誤差
"""

import numpy as np
import matplotlib.pyplot as plt

# データ生成
np.random.seed(seed=1)
N = 100   # データの数
K = 3     # 分布の数
T3 = np.zeros((N, 3), dtype=np.uint8)
T2 = np.zeros((N, 2), dtype=np.uint8)
X = np.zeros((N, 2))
X_range0 = [-3, 3]    # X0の範囲、表示用
X_range1 = [-3, 3]    # X1の範囲、表示用
Mu = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]])  # 分布の中心
Sig = np.array([[.7, .7], [.8, .3], [.3, .8]])  # 分布の分散
Pi = np.array([0.4, 0.8, 1])  # 各分布への割合 0.4, 0.8, 1

for n in range(N):
    wk = np.random.rand()
    for k in range(K):
        if wk < Pi[k]:
            T3[n, k] = 1
            break
    for k in range(2):
        X[n, k] = (np.random.randn() * Sig[T3[n, :] == 1, k] + Mu[T3[n, :] == 1, k])

T2[:, 0] = T3[:, 0]
T2[:, 1] = T3[:, 1] | T3[:, 2]    # T2とT3の論理和


# 3クラス用ロジスティック回帰モデル
def logistic3(x0, x1, w):
    K = 3
    w = w.reshape((3, 3))
    n = len(x1)
    y = np.zeros((n, K))
    for k in range(K):
        y[:, k] = np.exp(w[k, 0] * x0 + w[k, 1] * x1 + w[k, 2])
    wk = np.sum(y, axis=1)
    wk = y.T / wk
    y = wk.T
    return y


# 交差エントロピー誤差
def cee_logistic3(w, x, t):
    X_n = x.shape[0]
    y = logistic3(x[:, 0], x[:, 1], w)
    cee = 0
    N, K = y.shape
    for n in range(N):
        for k in range(K):
            cee = cee - (t[n, k] * np.log(y[n, k]))
    cee = cee / X_n
    return cee


W = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(cee_logistic3(W, X, T3))


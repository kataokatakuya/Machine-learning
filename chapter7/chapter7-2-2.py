"""
7.2  ニューラルネットワークモデル
"""
"""
7.2.2  2層フィードフォワードニューラルネットの実装
"""

import numpy as np
import matplotlib.pyplot as plt
# データ生成
np.random.seed(seed=1)    # 乱数を固定
N = 200    # データの数
K = 3    # 分布の数
T = np.zeros((N, 3), dtype=np.uint8)
X = np.zeros((N, 2))
X_range0 = [-3, 3]    # X0の範囲, 表示用
X_range1 = [-3, 3]    # X1の範囲, 表示用
Mu = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]])    # 分布の中心
Sig = np.array([[.7, .7], [.8, .3], [.3, .8]])    # 分布の分散
Pi = np.array([0.4, 0.8, 1])    # 各分布への割合

for n in range(N):
    wk = np.random.rand()
    for k in range(K):
        if wk < Pi[k]:
            T[n, k] = 1
            break
    for k in range(2):
        X[n, k] = np.random.randn() * Sig[T[n, :] == 1, k] + \
                    Mu[T[n, :] == 1, k]


# 2分類のデータをテスト・訓練データに分類
TestRatio = 0.5
X_n_training = int(N * TestRatio)
X_train = X[:X_n_training, :]
X_test = X[X_n_training:, :]
T_train = T[:X_n_training, :]
T_test = T[X_n_training:, :]

# データを'class_data.npz'に保存
np.savez('class_data.npz', X_train=X_train, T_train=T_train, X_test=X_test, T_test=T_test, X_range0=X_range0, X_range1=X_range1)


# データの図示
def Show_data(x, t):
    wk, n = t.shape
    c = [[0, 0, 0], [.5, .5, .5], [1, 1, 1]]
    for i in range(n):
        plt.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1], linestyle='none', marker='o', markeredgecolor='black', color=c[i], alpha=0.8)
    plt.grid(True)


plt.figure(1, figsize=(8, 3.7))
plt.subplot(1, 2, 1)
Show_data(X_train, T_train)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.title('Training Data')
plt.subplot(1, 2, 2)
Show_data(X_test, T_test)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.title('Test Data')
# plt.show()


# シグモイド関数
def Sigmoid(x):
    y = 1/ (1 + np.exp(-x))
    return y

def FNN(wv, M, K, x):
    N, D = x.shape      # 入力次元
    w = wv[:M * (D + 1)]    # 中間層ニューロンへの重み
    w = w.reshape(M, (D + 1))
    v = wv[M * (D + 1):]     # 出力層ニューロンへの重み
    v = v.reshape((K, M + 1))
    b = np.zeros((N, M + 1))    # 中間層ニューロンの入力総和
    z = np.zeros((N, M + 1))    # 中間層ニューロンの出力
    a = np.zeros((N, K))    # 出力層ニューロンの入力総和
    y = np.zeros((N, K))    # 出力層ニューロンの出力

    for n in range(N):
        # 中間層の計算
        for m in range(M):
            b[n, m] = np.dot(w[m, :], np.r_[x[n, :], 1])
            z[n, m] = Sigmoid(b[n, m])
        # 出力層の計算
        z[n, M] = 1     # ダミーニューロン
        wkz = 0
        for k in range(K):
            a[n, k] = np.dot(v[k, :], z[n, :])
            wkz = wkz + np.exp(a[n, k])
        for k in range(K):
            y[n, k] = np.exp(a[n, k]) / wkz
    return y, a, z, b

# test
WV = np.ones(15)
M = 2
K = 3
print(FNN(WV, M, K, X_train[:2, :]))







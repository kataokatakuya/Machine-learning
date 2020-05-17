"""
7.2  ニューラルネットワークモデル
"""
"""
7.2.4  数値微分法による勾配法
"""

import numpy as np
import matplotlib.pyplot as plt
import time


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



# データの図示
def Show_data(x, t):
    wk, n = t.shape
    c = [[0, 0, 0], [.5, .5, .5], [1, 1, 1]]
    for i in range(n):
        plt.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1], linestyle='none', marker='o', markeredgecolor='black', color=c[i], alpha=0.8)
    plt.grid(True)



# 平均交差エントロピー誤差
def CE_FNN(wv, M, K, x, t):
    N, D = x.shape
    y, a, z, b = FNN(wv, M, K, x)
    ce = -np.dot(np.log(y.reshape(-1)), t.reshape(-1)) / N
    return ce



# 数値微分
def dCE_FNN_num(wv, M, K, x, t):
    epsilon = 0.001
    dwv = np.zeros_like(wv)
    for iwv in range(len(wv)):
        wv_modified = wv.copy()
        wv_modified[iwv] = wv[iwv] - epsilon
        mse1 = CE_FNN(wv_modified, M, K, x, t)
        wv_modified[iwv] = wv[iwv] + epsilon
        mse2 = CE_FNN(wv_modified, M, K, x, t)
        dwv[iwv] = (mse2 - mse1) / (2 * epsilon)
    return dwv



# 数値微分法を使った勾配法
def Fit_FNN_num(wv_init, M, K, x_train, t_train, x_test, t_test, n, alpha):
    wvt = wv_init
    err_train = np.zeros(n)
    err_test = np.zeros(n)
    wv_hist = np.zeros((n, len(wv_init)))

    for i in range(n):
        wvt = wvt - alpha * dCE_FNN_num(wvt, M, K, x_train, t_train)
        err_train[i] = CE_FNN(wvt, M, K, x_train, t_train)
        err_test[i] = CE_FNN(wvt, M, K, x_test, t_test)
        wv_hist[i, :] = wvt
    return wvt, wv_hist, err_train, err_test



# 境界線表示関数
def show_FNN(wv, M, K):
    xn = 60     # 等高線表示の解像度
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    x = np.c_[np.reshape(xx0, xn * xn, 'F'), np.reshape(xx1, xn * xn, order= 'F')]
    y, a, z, b = FNN(wv, M, K, x)
    plt.figure(1, figsize=(4, 4))
    for ic in range(K):
        f = y[:, ic]
        f = f.reshape(xn, xn)
        f = f.T
        cont = plt.contour(xx0, xx1, f, levels=[0.8, 0.9], colors=['cornflowerblue', 'black'])
        cont.clabel(fmt='%1.1f', fontsize=9)
    plt.xlim(X_range0)
    plt.ylim(X_range1)

# メイン
startTime = time.time()
M = 2
K = 3
np.random.seed(1)
WV_init = np.random.normal(0, 0.01, M * 3 + K * (M + 1))
N_step = 1000
alpha = 0.5
WV, WV_hist, Err_train, Err_test = Fit_FNN_num(WV_init, M, K, X_train, T_train, X_test, T_test, N_step, alpha)
calculation_time = time.time() - startTime
print('Calculation time:{0:.3f} sec'.format(calculation_time))



# # 学習誤差の表示
# plt.figure(1, figsize=(3, 3))
# plt.plot(Err_train, 'black', label='training')
# plt.plot(Err_test, 'cornflowerblue', label='test')
# plt.legend()
# plt.show()

# # 重みの時間発展の表示
# plt.figure(1, figsize=(3, 3))
# plt.plot(WV_hist[:, :M * 3], 'black')
# plt.plot(WV_hist[:, M * 3:], 'cornflowerblue')
# plt.show()


# 境界線の表示
plt.figure(1, figsize=(3, 3))
Show_data(X_test, T_test)
show_FNN(WV, M, K)
plt.show()


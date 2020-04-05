"""
5.1  1次元入力の直線モデル
"""
"""
5.1.3  パラメータを求める(勾配法)
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# データを生成
np.random.seed(seed=1)    # 乱数を固定
X_min = 4                 # Xの下限(表示用)
X_max = 30                # Xの上限(表示用)
X_n = 16                  # データの個数

X = 5 + 25 * np.random.rand(X_n)
Prm_c = [170, 108, 0.2]   # 生成パラメータ
T = Prm_c[0] - Prm_c[1] * np.exp(-Prm_c[2] * X) \
            + 4 * np.random.randn(X_n)

# 平均誤差関数
def mse_line(x, t, w):
    y = w[0] * x + w[1]
    mse = np.mean((y - t) ** 2)
    return mse


# 平均二乗誤差の勾配
def dmse_line(x, t, w):
    y = w[0] * x + w[1]
    d_w0 = 2 * np.mean((y - t) * x)
    d_w1 = 2 * np.mean(y - t)
    return d_w0, d_w1


# 勾配法
def fit_line_num(x, t):
    w_init = [10.0, 165.0]   # 初期パラメータ
    alpha = 0.001    # 学習率
    i_max = 100000   # 繰り返しの最大数
    eps = 0.1        # 繰り返しをやめる勾配の絶対値の閾値
    w_i = np.zeros([i_max, 2])
    w_i[0, :] = w_init

    for i in range(1, i_max):
        dmse = dmse_line(x, t, w_i[i - 1])
        w_i[i, 0] = w_i[i - 1, 0] - alpha * dmse[0]
        w_i[i, 1] = w_i[i - 1, 1] - alpha * dmse[1]
        if max(np.absolute(dmse)) < eps:   # 終了判定、np.absoluteは絶対値
            break
    w0 = w_i[i, 0]
    w1 = w_i[i, 1]
    w_i = w_i[:i1, :]
    return w0, w1, dmse, w_i


# メイン
plt.figure(figsize=(4,4))
# MSEの等高線表示
xn = 100   # 等高線解像度
w0_range = [-25, 25]
w1_range = [120, 170]
x0 = np.linspace(w0_range[0], w0_range[1], xn)
x1 = np.linspace(w1_range[0], w1_range[1], xn)
xx0, xx1 = np.meshgrid(x0, x1)
J = np.zeros((len(x0), len(x1)))

for i0 in range(xn):
    for i1 in range(xn):
        J[i1, i0] = mse_line(X, T, (x0[i0], x1[i1]))

cont = plt.contour(xx0, xx1, J, 30, colors='black', levels=(100, 1000, 10000, 100000))
cont.clabel(fmt='%1.0f', fontsize=8)
plt.grid(True)

# 勾配法呼び出し
W0,W1, dMSE, W_history = fit_line_num(X, T)
# 結果表示
print('繰り返し回数 {0}'.format(W_history.shape[0]))
print('W=[{0:.6f}, {1:.6f}]'.format(W0, W1))
print('dMSE=[{0:.6f}, {1:.6f}]'.format(dMSE[0], dMSE[1]))
print('MSE={0:.6f}'.format(mse_line(X, T, [W0, W1])))
plt.plot(W_history[:, 0], W_history[:, 1], '.-', color='gray', markersize=10, markeredgecolor='cornflowerblue')
plt.show()


# 線の表示
def show_line(w):
    xb = np.linspace(X_min, X_max, 100)
    y = w[0] * xb + w[1]
    plt.plot(xb, y, color=(.5, .5, .5), linewidth=4)


# メイン
plt.figure(figsize=(4, 4))
W = np.array([W0, W1])
mse = mse_line(X, T, W)
print('w0={0:.3f}, w1={1:.3f}'.format(W0, W1))
print('SD={0:.3f} cm'.format(np.sqrt(mse)))
show_line(W)
plt.plot(X, T, marker='o', linestyle='None', color='cornflowerblue', markeredgecolor='black')
plt.xlim(X_min, X_max)
plt.grid(True)
plt.show()


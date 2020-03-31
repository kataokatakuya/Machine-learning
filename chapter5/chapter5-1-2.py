"""
5.1  1次元入力の直線モデル
"""
"""
5.1.2  二乗誤差関数
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
np.savez('ch5_data.npz', X = X, X_min = X_min, X_max = X_max, X_n = X_n, T=T)


# 平均誤差関数
def mse_line(x, t, w):
    y = w[0] * x + w[1]
    mse = np.mean((y - t) ** 2)
    return mse

# 計算
xn = 100  # 等高線表示の解像度
w0_range = [-25, 25]
w1_range = [120, 170]
x0 = np.linspace(w0_range[0], w0_range[1], xn)
x1 = np.linspace(w1_range[0], w1_range[1], xn)
xx0, xx1 = np.meshgrid(x0, x1)
J = np.zeros((len(x0), len(x1)))

for i0 in range(xn):
    for i1 in range(xn):
        J[i1, i0] = mse_line(X, T, (x0[i0], x1[i1]))

# 表示
plt.figure(figsize=(9.5, 4))
plt.subplots_adjust(wspace=0.5)

ax = plt.subplot(1, 2, 1, projection='3d')
ax.plot_surface(xx0, xx1, J, rstride=10, cstride=10, alpha=0.3, color='blue',
  edgecolor='black')
ax.set_xticks([-20, 0, 20])
ax.set_yticks([120, 140, 160])
ax.view_init(20, -60)

plt.subplot(1, 2, 2)
cont = plt.contour(xx0, xx1, J, 30, colors='black', levels=[100, 1000, 10000, 100000])
cont.clabel(fmt='%1.0f', fontsize=8)
plt.grid(True)
plt.show()

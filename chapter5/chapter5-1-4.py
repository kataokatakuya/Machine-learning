"""
5.1  1次元入力の直線モデル
"""
"""
5.1.4  直線モデルパラメータの解析解
"""
import numpy as np
import matplotlib.pyplot as plt

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

# 線の表示
def show_line(w):
    xb = np.linspace(X_min, X_max, 100)
    y = w[0] * xb + w[1]
    plt.plot(xb, y, color=(.5, .5, .5), linewidth=4)

def fit_line(x, t):
    mx = np.mean(x)
    mt = np.mean(t)
    mtx = np.mean(t * x)
    mxx = np.mean(x * x)
    w0 = (mtx - mt * mx) / (mxx - mx ** 2)
    w1 = mt - w0 * mx
    return np.array([w0, w1])

# メイン
W = fit_line(X, T)
print('w0={0:.3f}, w1={1:.3f}'.format(W[0], W[1]))
mse = mse_line(X, T, W)
print('SD={0:3f} cm'.format(np.sqrt(mse)))
plt.figure(figsize=(4, 4))
show_line(W)
plt.plot(X, T, marker='o', linestyle='None', color='cornflowerblue', markeredgecolor='black')
plt.xlim(X_min, X_max)
plt.grid(True)
plt.show()

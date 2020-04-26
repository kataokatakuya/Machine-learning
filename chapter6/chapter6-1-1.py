"""
6.1  1次元入力2クラス分類
"""
"""
6.1.1  問題設定
"""

import numpy as np
import matplotlib.pyplot as plt

# データ生成
np.random.seed(seed=0)              # 乱数を固定
X_min = 0
X_max = 2.5
X_n = 30
X_col = ['cornflowerblue', 'gray']
X = np.zeros(X_n)                   # 入力データ
T = np.zeros(X_n, dtype=np.uint8)   # 目標データ

Dist_s = [0.4, 0.8]                 # 分布の開始地点
Dist_w = [0.8, 1.6]                 # 分布の幅
Pi = 0.5                            # クラス0の比率

for n in range(X_n):
    wk = np.random.rand()
    T[n] = 0 * (wk < Pi) + 1 * (wk >= Pi)
    X[n] = np.random.rand() * Dist_w[T[n]] + Dist_s[T[n]]

# データ表示
print('X=' + str(np.round(X, 2)))
print('T=' + str(T))

# データ分布表示
def show_data1(x, t):
    K = np.max(t) + 1
    for k in range(K):
        plt.plot(x[t == k], t[t == k], alpha=0.5, linestyle='none', marker='o')
    plt.grid(True)
    plt.ylim(-.5, 1.5)
    plt.xlim(X_min, X_max)
    plt.yticks([0, 1])

# メイン
fig = plt.figure(figsize=(3, 3))
show_data1(X, T)
plt.show()

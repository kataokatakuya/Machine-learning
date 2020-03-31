"""
5.1  1次元入力の直線モデル
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
np.savez('ch5_data.npz', X = X, X_min = X_min, X_max = X_max, X_n = X_n, T=T)

# 年齢の表示
print(np.round(X, 2))
# 身長の表示
print(np.round(T, 2))

# データグラフの表示
plt.figure(figsize=(4, 4))
plt.plot(X, T, marker='o', linestyle='None', markeredgecolor='black', color='cornflowerblue')
plt.xlim(X_min, X_max)
plt.grid(True)
plt.show()


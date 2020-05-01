"""
6.1  1次元入力2クラス分類
"""
"""
6.1.4  ロジスティック回帰モデル
"""

import numpy as np
import matplotlib.pyplot as plt

# データ生成
np.random.seed(seed=0)              # 乱数を固定
X_min = 0
X_max = 2.5
X_n = 30


# シグモイド関数
def logistic(x, w):
    y = 1 / (1 + np.exp(-(w[0] * x + w[1])))
    return y


# 決定境界の表示
def show_logistic(w):
    xb = np.linspace(X_min, X_max, 100)
    y = logistic(xb, w)
    plt.plot(xb, y, color='gray', linewidth=4)
    # 決定境界
    i = np.min(np.where(y > 0.5))
    B = (xb[i - 1] + xb[i]) / 2
    plt.plot([B, B], [-.5, 1.5], color='k', linestyle='--')
    plt.grid(True)

    # 決定境界の近似値 出力
    print(B)

    return B


# メイン
fig = plt.figure(figsize=(5, 3))
W = [8, -10]
show_logistic(W)
plt.ylim(-.5, 1.5)
plt.xlim(X_min, X_max)
plt.show()

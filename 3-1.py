"""
3.1.1  ランダムなグラフを描く
"""
"""
import numpy as np
import matplotlib.pyplot as plt

# data作成
np.random.seed(1)
x = np.arange(10)
y = np.random.rand(10)

# グラフ表示
plt.plot(x, y)
plt.show()
"""


"""
3.1.3  3次関数 f(x) = (x-2)x(x+2)を描く
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x - 2) * x * (x + 2)

#print(f(1))
#print(f(np.array([1, 2, 3])))



"""
3.1.4  描画する範囲を決める
"""
"""
x = np.arange(-3, 3.5, 0.5)
print(x)

x = np.linspace(-3, 3, 10)
print(np.round(x, 2))
"""

"""
3.1.5  グラフを描画する
"""
"""
plt.plot(x, f(x))
plt.show()
"""

"""
3.1.6  グラフを装飾する
"""

def f2(x, w):
    return (x - w) * x * (x + 2)

x = np.linspace(-3, 3, 100)

# グラフ描写
plt.plot(x, f2(x, 2), color='black', label='$w=2$')
plt.plot(x, f2(x, 1), color='cornflowerblue', label='$w=1$')
plt.legend(loc="upper left")
plt.ylim(-15, 15)
plt.title('$f_2(x)$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid(True)
plt.show()




"""
3.1.7  グラフを複数並べる
"""

plt.figure(figsize=(10,3))
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.title(i + 1)
    plt.plot(x, f2(x, i), 'k')
    plt.ylim(-20, 20)
    plt.grid(True)
plt.show()

"""
4-7  指数関数と対数関数
"""
"""
4-7-8  ガウス関数
"""

import numpy as np
import matplotlib.pyplot as plt

def gauss(mu, sigma, a):
    return a * np.exp(-(x - mu) ** 2 / sigma ** 2)

x = np.linspace(-4, 4, 100)
plt.figure(figsize=(4,4))
plt.plot(x, gauss(0, 1, 1), 'black', linewidth=3)
plt.plot(x, gauss(2, 3, 0.5), 'gray', linewidth=3)

plt.ylim(-.5, 1.5)
plt.xlim(-4, 4)
plt.grid(True)
plt.show()

"""
pythonの基本
教科書の一部のみ抜粋する
"""

import numpy as np

"""
連続した整数のデータ作成
"""
"""
y = range(5,10)
print(y[0], y[1], y[2], y[3], y[4])

print(y)

z = list(range(5, 10))
print(z)

print(list(range(10)))
"""


"""
長さが1のタプル
"""
"""
a = (1)
print(type(a))

a = (1,)
print(type(a))
"""


"""
for文
"""
"""
for i in [1, 2, 3]:
    print(i)

num = [2, 4, 6, 8, 10]
for i in range(len(num)):
    num[i] = num[i] * 2
print(num)

num = [2, 4, 6, 8, 10]
for i, n in enumerate(num):
    num[i] = n * 2
print(num)
"""


"""
ベクトル
"""
"""
x = np.array([1, 2, 3])
print(x)

y = np.array([4, 5, 6])
print(x + y)

print(np.arange(10))
print(np.arange(5, 10))
"""


"""
ベクトルのコピー
"""
"""
print('b = a の場合')
a = np.array([1, 1])
b = a
print('a = ' + str(a))
print('b = ' + str(b))
b[0] = 100
print('a = ' + str(a))
print('b = ' + str(b))

print('b = a.copy() の場合')
a = np.array([1, 1])
b = a.copy()
print('a = ' + str(a))
print('b = ' + str(b))
b[0] = 100
print('a = ' + str(a))
print('b = ' + str(b))
"""


"""
行列の定義
"""
"""
x = np.array([[1, 2, 3], [4, 5, 6]])
print(x)
"""


"""
行列のサイズ
"""
"""
x = np.array([[1,2,3], [4,5,6]])
print(x.shape)

w, h = x.shape
print(w)
print(h)
"""


"""
要素の参照
"""
"""
x = np.array([[1, 2, 3], [4, 5, 6]])
print(x[1, 2])
"""


"""
要素の書き換え
"""
"""
x = np.array([[1, 2, 3], [4, 5, 6]])
x[1, 2] = 100
print(x)
"""

"""
要素が 0 や 1 の ndarray の作成
"""
"""
print(np.zeros(10))
print(np.ones(2, 10))
print(np.ones((2, 10)))
"""

"""
要素がランダムな行列の生成
"""
"""
print(np.random.rand(2,3))
"""

"""
行列のサイズの変更
"""
"""
a = np.arange(10)
print(a)
print(a.reshape(2, 5))
"""


"""
行列における四則演算
"""
"""
x = np.array([[4, 4, 4], [8, 8, 8]])
y = np.array([[1, 1, 1], [2, 2, 2]])
print(x + y)
"""

"""
スカラー × 行列
"""
"""
x = np.array([[4, 4, 4], [8, 8, 8]])
print(10 * x)
"""


"""
算術関数
"""
"""
x = np.array([[4, 4, 4], [8, 8, 8]])
print(np.exp(x))
"""


"""
行列積の計算
"""
"""
v = np.array([[1, 2, 3], [4, 5, 6]])
w = np.array([[1, 1], [2, 2], [3, 3]])
print(v.dot(w))
"""


"""
bool配列の利用
"""
"""
x = np.array([1, 1, 2, 3, 5, 8, 13])
print(x > 3)
print(x[x > 3])

x[x > 3] = 999
print(x)
"""


"""
1つのndarray型の保存
"""
"""
data = np.random.randn(5)
print(data)
np.save('datafile.npy', data)    #セーブ
data = []                       #データの消去
data = np.load('datafile.npy')  #ロード
print(data)
"""

data1 = np.array([1, 2, 3])
data2 = np.array([10, 20, 30])
np.savez('datafile2.npz', data1=data1, data2=data2)
data1 = []
data2 = []
outfile = np.load('datafile2.npz')
print(outfile.files)
data1 = outfile['data1']
data2 = outfile['data2']
print(data1)
print(data2)

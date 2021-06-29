# -*- coding: utf-8 -*-
"""
@Date  : 2021/6/28
@Refer : https://blog.csdn.net/qq_36302589/article/details/101602826
@Title : 
@Desc  : 
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# 输入数据
# 读数据
df = pd.read_csv(r"../LearnSeaborn/PhoneOldAndNew.csv")
x = df["area"].values.reshape((-1, 1))
y = pd.get_dummies(df["species"]).values[:, 0].ravel()
print(x, "\n", y)
w1 = []
w2 = []
for i, v in enumerate(y):
    if v == 0:
        w1.append([x[i][0], 1])
    else:
        w2.append([x[i][0], 1])
w1 = np.array(w1)
w2 = np.array(w2)
print(w1, w1.shape)

# 线性可分
# w1 = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1]])
# w2 = np.array([[1, 1, 1]])

# 线性不可分
# w1 = np.array([[0, 0, 1], [1, 1, 1]])
# w2 = np.array([[1, 0, 1], [0, 1, 1]])

w2 = -w2
c = 1
flag = 0
X = np.concatenate((w1, w2), axis=0)
b = np.ones(len(X))
print(b, b.shape, X)
# 求伪逆矩阵XX
XX = np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose())
w = np.matmul(XX, b)
e = np.matmul(X, w) - b
t = 0
iteration = 10000
while True:
    temp = min(e)
    temp1 = max(e)
    if 0 > temp > -1e-4:
        temp = 0
    if temp > 1e-3:
        deltab = e + abs(e)
        b = b + c * deltab
        w = w + c * np.matmul(XX, deltab)
        e = np.matmul(X, w) - b
    else:
        if 1e-4 > temp >= 0:
            break
        else:
            # 线性不可分
            if temp1 < 0:
                flag = 1
                break
            else:
                # 趋近时迭代
                deltab = e + abs(e)
                b = b + c * deltab
                w = w + c * np.matmul(XX, deltab)
                e = np.matmul(X, w) - b
                t = t + 1
                if t >= iteration:
                    break
print(XX, '\n', w, '\n', e)

x_v = -w[1] / w[0]
print(x_v)
y_pred = np.where(x[:, 0] > x_v, 0, 1)
print(y_pred)
print("准确率", accuracy_score(y_pred, y))

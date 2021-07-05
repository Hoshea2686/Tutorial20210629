# -*- coding: utf-8 -*-
"""
@Date  : 2021/7/2
@Refer :
@Title : 
@Desc  : 
"""
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set()

# 绘制熵图像，gini图像，和错分图像

xx = np.linspace(0, 1, 50)
y1 = [-x*np.log2(x)-(1-x)*np.log2(1-x) for x in xx]
y2 = [2*x*(1-x) for x in xx]
y3 = [1-max(x, (1-x)) for x in xx]

plt.figure(figsize=(6, 4))
plt.plot(xx, y1, label="entropy")
plt.plot(xx, y2, label="gini")
plt.plot(xx, y3, label="miss")
plt.xlabel("p+")
plt.ylabel("criterion")
plt.title("criterion of qulity as a function of p+(binary classification)")
plt.legend()
plt.show()

# 创建两类
np.random.seed(1)
# 1类
train_data = np.random.normal(size=(100, 2))
train_labels = np.zeros(100)
# 2类
train_data_2 = np.random.normal(size=(100, 2), loc=2)
train_labels_2 = np.ones(100)
train_data = np.r_[train_data, train_data_2]
train_labels = np.r_[train_labels, train_labels_2]

plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap="autumn")
plt.plot(range(-2, 5), range(4, -3, -1))

plt.show()

# 使用决策树将其分开

clf_tree = DecisionTreeClassifier(max_depth=3, criterion="entropy", random_state=1)
clf_tree.fit(train_data, train_labels)


# predict分区
# 创建分区网点集合
def get_grid(data):
    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()
    return np.mgrid[xmin:xmax:100j, ymin:ymax:100j]


xx, yy = get_grid(train_data)
predicted = clf_tree.predict(np.c_[xx.flat, yy.flat]).reshape(xx.shape)
plt.pcolormesh(xx, yy, predicted, cmap="autumn")
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap="autumn", edgecolors="black")
plt.show()


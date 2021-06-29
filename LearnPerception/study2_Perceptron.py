# -*- coding: utf-8 -*-
"""
@Date  : 2021/6/27
@Refer :
@Title : 利用sklearn载入iris数据集并利用感知机进行分类
@Desc  : 
"""

import sklearn
from sklearn import datasets
from sklearn.linear_model import Perceptron
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

# 加载iris数据集
# iris数据集为一个用于识别鸢尾花的机器学习数据集
# 通过四种特征(花瓣长度,花瓣宽度,花萼长度,花萼宽度)来实现三种鸢尾花的类别划分
iris = datasets.load_iris()

# iris.data大小为150*4,代表4种特征
# 这里只提取两类.花长，鄂长两个特征, 线性可分
X = iris.data[:100, [0, 2]]
# 标签
y = iris.target[:100]

# 可视化数据
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:, 0], X[50:, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()


# random_state = 0表示不设定随机数种子,每一次产生的随机数不一样
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 为了追求机器学习和最优化算法的最佳性能，我们将特征缩放。分类前先归一化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)  # 估算每个特征的平均值和标准差
# 查看特征的平均值，由于Iris我们只用了两个特征，结果是array([ 3.82857143,  1.22666667])
sc.mean_
# 查看特征的标准差，结果是array([ 1.79595918,  0.77769705])
sc.scale_
# 标准化训练集
X_train_std = sc.transform(X_train)
# 注意：这里我们要用同样的参数来标准化测试集，使得测试集和训练集之间有可比性
X_test_std = sc.transform(X_test)

# max_iter：可以理解成梯度下降中迭代的次数
# eta0：可以理解成梯度下降中的学习率
# random_state：设置随机种子的，为了每次迭代都有相同的训练集顺序
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

# 分类测试集，这将返回一个测试结果的数组
y_pred = ppn.predict(X_test_std)

# 计算模型在测试集上的准确性
print(accuracy_score(y_test, y_pred))   # 百分百

# 绘图
x = sc.transform(X)
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
grid_test = np.stack((x1.flat, x2.flat), axis=1)
print(grid_test)
grid_pred = ppn.predict(grid_test)
grid_pred = grid_pred.reshape(x1.shape)

plt.figure(2)
# 自定义颜色版
cm_bg = mpl.colors.ListedColormap(['#FFA0A0', '#A0A0FF'])
cm_dot = mpl.colors.ListedColormap(['r', 'b'])
# 画分类区域
plt.pcolormesh(x1, x2, grid_pred, shading='auto', cmap=cm_bg)
# 画点集，样本
plt.scatter(x[:, 0], x[:, 1], alpha=0.5, c=y, edgecolors='k', s=50, cmap=cm_dot)

plt.show()


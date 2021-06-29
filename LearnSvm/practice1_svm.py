# -*- coding: utf-8 -*-
"""
@Date  : 2021/6/27
@Refer : https://www.cnblogs.com/shenxiaolin/p/8854838.html
@Title : 实现鸢尾花数据集分类问题——基于skearn的SVM
@Desc  : 
"""
import pandas as pd
from sklearn import svm
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn import datasets
import seaborn as sns

iris = datasets.load_iris()
x = iris.data[:, 0:2]  # 取出前两个特征来分析
y = iris.target
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.3)

# 搭建模型，训练SVM分类器
# classifier=svm.SVC(kernel='linear',gamma=0.1,decision_function_shape='ovo',C=0.1)
# kernel='linear'时，为线性核函数，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
classifier=svm.SVC(kernel='rbf',gamma=0.1,decision_function_shape='ovo',C=0.8)
# kernel='rbf'（default）时，为高斯核函数，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
# decision_function_shape='ovo'时，为one v one分类问题，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
# decision_function_shape='ovr'时，为one v rest分类问题，即一个类别与其他类别进行划分。
#开始训练
classifier.fit(x_train,y_train.ravel())
#调用ravel()函数将矩阵转变成一维数组
# （ravel()函数与flatten()的区别）
# 两者所要实现的功能是一致的（将多维数组降为一维），
# 两者的区别在于返回拷贝（copy）还是返回视图（view），
# numpy.flatten() 返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵，
# 而numpy.ravel()返回的是视图（view），会影响（reflects）原始矩阵。

def show_accuracy(y_hat,y_train,str):
    pass
#（4）计算svm分类器的准确率
print("SVM-输出训练集的准确率为：",classifier.score(x_train,y_train))
y_hat=classifier.predict(x_train)
show_accuracy(y_hat,y_train,'训练集')
print("SVM-输出测试集的准确率为：",classifier.score(x_test,y_test))
y_hat=classifier.predict(x_test)
show_accuracy(y_hat,y_test,'测试集')
# SVM-输出训练集的准确率为： 0.838095238095
# SVM-输出测试集的准确率为： 0.777777777778

# 查看决策函数，可以通过decision_function()实现。decision_function中每一列的值代表距离各类别的距离。
# print('decision_function:\n', classifier.decision_function(x_train))
print('\npredict:\n', classifier.predict(x_train))

# (5)绘制图像
# 1.确定坐标轴范围，x，y轴分别表示两个特征
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围  x[:, 0] "："表示所有行，0表示第1列
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围  x[:, 0] "："表示所有行，1表示第2列
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点（用meshgrid函数生成两个网格矩阵X1和X2）
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点，再通过stack()函数，axis=1，生成测试点
# .flat 将矩阵转变成一维数组 （与ravel()的区别：flatten：返回的是拷贝

print("grid_test = \n", grid_test)
# print("x = \n",x)
grid_hat = classifier.predict(grid_test)       # 预测分类值

print("grid_hat = \n", grid_hat)
# print(x1.shape())
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同


# 2.指定默认字体
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 3.绘制
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

alpha=0.5

plt.pcolormesh(x1, x2, grid_hat, shading="auto", cmap=cm_light)  # 预测值的显示
# plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark, label='$like$')  # 样本
df = {
    "f1":x[:, 0],
    "f2":x[:, 1],
    "y":y
}
df = pd.DataFrame(df)
sns.scatterplot(x="f1", y="f2", hue="y", data=df, palette=["g", "r", "b"])
# plt.plot(x[:, 0], x[:, 1], 'o', alpha=alpha, color='blue', markeredgecolor='k')
plt.scatter(x_test[:, 0], x_test[:, 1], s=5, facecolors='yellow', zorder=10)  # 标记测试集样本
plt.xlabel(u'花萼长度', fontsize=13)
plt.ylabel(u'花萼宽度', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花SVM二特征分类', fontsize=15)
# plt.grid()
plt.legend()
plt.show()

# -*- coding: utf-8 -*-
"""
@Date  : 2021/7/3
@Refer : study5.ipynb
@Title : 决策树实验
@Desc  : 
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
sns.set()
warnings.filterwarnings("ignore")

# **（1）绘制决策的评价函数**
# - 熵：y = -xlon2(x)- (1-x)log2(1-x)
# - 基尼不确定性：y = 1-x**2-(1-x)**2 = 2*x*(x-1)
# - 错分率：y = 1-max(1, 1-x)
x = np.linspace(0, 1, 50)
y1 = [-v*np.log2(v)-(1-v)*np.log2(1-v) for v in x]
y2 = [2*v*(1-v) for v in x]
y3 = [1-max(v, 1-v) for v in x]
plt.figure(figsize=(6, 4))
plt.plot(x, y1, label="entropy")
plt.plot(x, y2, label="gini")
plt.plot(x, y3, label="missclass")
plt.xlabel("p+")
plt.ylabel("criterion")
plt.title("criterion of quality function (binary classifier) ")
plt.legend()
# plt.show()

# **(2)** 用决策树分类2类案例
np.random.seed(17)
# 第一类
train_data = np.random.normal(size=(100, 2))
train_labels = np.zeros(100)
# 第二类
train_data = np.r_[train_data, np.random.normal(size=(100, 2), loc=2)]
train_labels = np.r_[train_labels, np.ones(100)]

plt.figure(figsize=(10, 6))
plt.scatter(train_data[:, 0], train_data[:, 1], cmap="summer", s=50, c=train_labels)
plt.plot(range(-2, 4), range(4, -2, -1))
# plt.show()

# 创建决策树
clf_tree = DecisionTreeClassifier(max_depth=3, random_state=17, criterion="entropy")
clf_tree.fit(train_data, train_labels)


def get_grid(data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.mgrid[x_min:x_max:100j, y_min:y_max:100j]


# 预测决策区， 通过绘图表示
xx, yy = get_grid(train_data)
predicted = clf_tree.predict(np.c_[xx.flat, yy.flat]).reshape(xx.shape)
plt.figure(figsize=(10, 6))
plt.pcolormesh(xx, yy, predicted, cmap="summer")
plt.scatter(train_data[:, 0], train_data[:, 1], cmap="summer", s=50, c=train_labels, edgecolors="black")
# plt.show()


def plot_tree(decision_tree, feature_names, **kwargs):
    from io import StringIO, BytesIO
    import pydotplus
    from sklearn.tree import export_graphviz
    dot_data = StringIO()
    export_graphviz(decision_tree, out_file=dot_data, feature_names=feature_names, filled=True, **kwargs)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    plt.figure()
    img = plt.imread(BytesIO(graph.create_png()), "png")
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# 通过graphviz绘制决策数图
# plot_tree(clf_tree, ["x1", "x2"])


# **(3)在离网预测中(二分类问题)，使用最近邻，也使用决策树**
# 读取数据
df = pd.read_csv("telecom_churn.csv")
df["International plan"] = pd.factorize(df["International plan"])[0]
df["Voice mail plan"] = pd.factorize(df["Voice mail plan"])[0]
y = df["Churn"].astype("int")
state = df["State"]
df.drop(["Churn", "State"], axis=1, inplace=True)
print(df.head(), "\n", df.values)
# 训练两个模型
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train, X_holdout, y_train, y_holdout = train_test_split(df.values, y, test_size=0.3, random_state=17)
tree = DecisionTreeClassifier(max_depth=5, random_state=17)
knn = KNeighborsClassifier(n_neighbors=10)
tree.fit(X_train, y_train)
knn.fit(X_train, y_train)
# 评估Accuracy
from sklearn.metrics import accuracy_score
tree_pred = tree.predict(X_holdout)
knn_pred = knn.predict(X_holdout)
print("tree predicted:", accuracy_score(y_holdout, tree_pred))  # 0.94
print("knn predicted:", accuracy_score(y_holdout, knn_pred))    # 0.881

# **（4）网络调优-交叉验证法GridSearchCV()**
from sklearn.model_selection import GridSearchCV, cross_val_score

# 对tree调优
param_grid = {"max_depth": range(5, 7),
              "max_features": range(16, 18),
              }
tree_grid = GridSearchCV(tree, param_grid=param_grid, n_jobs=-1)
tree_grid.fit(X_train, y_train)
print("tree grid best_params_:", tree_grid.best_params_)  # {'max_depth': 6, 'max_features': 17}
print("tree grid accuracy: ", accuracy_score(tree_grid.predict(X_holdout), y_holdout))  # 0.946

# 对knn调优
# knn需要数据归一化（StanderScaler），最近邻与所有特征距离有关。
# 为了方便使用Pipeline将整个处理过程整合在一起
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", knn)
])
param_grid = {"knn__n_neighbors": range(6, 8)}
knn_grid = GridSearchCV(knn_pipe, param_grid=param_grid, n_jobs=-1)
knn_grid.fit(X_train, y_train)
print("knn grid best_params_:", knn_grid.best_params_)  # {'knn__n_neighbors': 7}
print("knn grid accuracy: ", accuracy_score(knn_grid.predict(X_holdout), y_holdout))  # 0.89

# **(5) 随机森林RandomForestClassifier()**
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=17)
# 直接用交叉验证cross_val_score评估模型
forest_cv_score = cross_val_score(forest, X_train, y_train, cv=5)
print(forest_cv_score, np.mean(forest_cv_score))   # [0.94860814 0.94646681 0.94646681 0.9527897  0.95708155] 0.9502826001047688
# 调优
param_grid = {
    "max_depth": range(8, 10),
    "max_features": range(5, 7),
}
forest_grid = GridSearchCV(forest, param_grid=param_grid, n_jobs=-1)
forest_grid.fit(X_train, y_train)
print("forest grid best_params_:", forest_grid.best_params_)  # {'max_depth': 9, 'max_features': 6}
print("forest grid accuracy: ", accuracy_score(forest_grid.predict(X_holdout), y_holdout))  # 0.954

# **(6)案例， MNIST 手写数字识别任务中应用决策树和 k-NN**
from sklearn.datasets import load_digits
# 加载数据集
mnist = load_digits()
X, y = mnist.data, mnist.target
print(X[0, :].reshape([8, 8]))


# 绘制一些 MNIST 手写数字。
def plot_mnist(X):
    f, axes = plt.subplots(1, 4, figsize=(10, 6))
    for i in range(4):
        axes[i].imshow(X[i, :].reshape([8, 8]), cmap='Greys')
    plt.show()
# plot_mnist(X)


# 分割训练集与测试集
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, random_state=17)

# 使用随机参数训练决策树和 k-NN。
tree = DecisionTreeClassifier(max_depth=5, random_state=17)
knn = KNeighborsClassifier(n_neighbors=10)
tree.fit(X_train, y_train)
knn.fit(X_train, y_train)
tree_pred = tree.predict(X_holdout)
knn_pred = knn.predict(X_holdout)
print("tree accuracy", accuracy_score(y_holdout, tree_pred))     # 0.666
print("knn accuracy", accuracy_score(y_holdout, knn_pred))      # 0.974

# 对tree调优，
param_grid = {"max_depth": [10, 20, 30],
              "max_features": [30, 50, 64],
              }
tree_grid = GridSearchCV(tree, param_grid=param_grid, n_jobs=-1)
tree_grid.fit(X_train, y_train)
print("tree grid best_params_:", tree_grid.best_params_)  # {'max_depth': 10, 'max_features': 50}
print("tree grid accuracy: ", accuracy_score(tree_grid.predict(X_holdout), y_holdout))  # 0.843

# knn不用调优10类已经是最优了, 用交叉验证，评估模型
param_grid = {"n_neighbors": range(1, 10)}
knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, n_jobs=-1)
knn_grid.fit(X_train, y_train)
print("knn grid best_params_:", knn_grid.best_params_)  # {'knn__n_neighbors': 1}
print("knn grid accuracy: ", accuracy_score(knn_grid.predict(X_holdout), y_holdout))  # 0.983

# **(7) 案例，某个特征直接和目标变量成比例的情况**
def form_noisy_data(n_obj=1000, n_feat=100, random_seed=17):
    np.seed = random_seed
    y = np.random.choice([-1, 1], size=n_obj)
    # 第一个特征与目标成比例
    x1 = 0.3 * y
    # 其他特征为噪声
    x_other = np.random.random(size=[n_obj, n_feat - 1])
    return np.hstack([x1.reshape([n_obj, 1]), x_other]), y


X, y = form_noisy_data()   # （1000, 100）, （1000,）
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, random_state=17)
cv_scores, holdout_scores = [], []
n_neighb = [1, 2, 3, 5] + list(range(50, 550, 50))
for k in n_neighb:
    knn_pipe = Pipeline([('scaler', StandardScaler()),
                         ('knn', KNeighborsClassifier(n_neighbors=k))])
    cv_scores.append(np.mean(cross_val_score(knn_pipe, X_train, y_train, cv=5)))
    knn_pipe.fit(X_train, y_train)
    holdout_scores.append(accuracy_score(y_holdout, knn_pipe.predict(X_holdout)))
plt.figure(figsize=(10, 6))
plt.plot(n_neighb, cv_scores, label='CV')
plt.plot(n_neighb, holdout_scores, label='holdout')
plt.title('Easy task. kNN fails')
plt.legend()
plt.show()

tree = DecisionTreeClassifier(random_state=17, max_depth=1)
tree_cv_score = np.mean(cross_val_score(tree, X_train, y_train, cv=5))
tree.fit(X_train, y_train)
tree_holdout_score = accuracy_score(y_holdout, tree.predict(X_holdout))
print('Decision tree. CV: {}, holdout: {}'.format(tree_cv_score, tree_holdout_score))
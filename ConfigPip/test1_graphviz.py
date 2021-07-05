# -*- coding: utf-8 -*-
"""
@Date  : 2021/7/1
@Refer :
@Title : 测试Graphviz软件安装成功？
@Desc  : 创建的一个决策树，将决策模型保存为iris.png
"""
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.create_png()

# graph.write_png('iris.png')
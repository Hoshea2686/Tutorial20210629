# -*- coding: utf-8 -*-
"""
@Date  : 2021/7/4
@Refer : study6.ipynb
@Title : 决策树和随机森林分析应用
@Desc  : 
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from IPython.display import SVG
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

import pydotplus
from sklearn.tree import export_graphviz
from io import StringIO, BytesIO

from typing import Union, Tuple, Dict, List, Any
"""
# **(1)创建数据**
# 创建一个示例数据集，该数据集表示了 A 会不会和 B 进行第二次约会。
# 而数据集中的特征包括：外貌，口才，酒精消费，以及第一次约会花了多少钱。

# 数据基本信息


features = ["Looks", "Alcoholic_beverage", "Eloquence", "Money_spent"]
df_train = dict()
df_train["Looks"] = ['handsome', 'handsome', 'handsome', 'repulsive', 'repulsive', 'repulsive', 'handsome']
df_train['Alcoholic_beverage'] = ['yes', 'yes', 'no', 'no', 'yes', 'yes', 'yes']
df_train['Eloquence'] = ['high', 'low', 'average', 'average', 'low', 'high', 'average']
df_train['Money_spent'] = ['lots', 'little', 'lots', 'little', 'lots', 'lots', 'lots']
df_train['Will_go'] = ['+', '-', '+', '-', '-', '+', '+']

# 数据整理，features用热独形式，Will_go 不用热独形式，转为数值标签
# 两种方法都可以将文字标签转为数字标签。例如
print(pd.factorize(['+', '-', '+', '-', '-', '+', '+'])[0])
# from sklearn.preprocessing import LabelEncoder
# df_train['Will_go'] = LabelEncoder().fit_transform(['+', '-', '+', '-', '-', '+', '+'])
# 其中Eloquence有3个值。需要另一种方法，如下
print(pd.get_dummies(['high', 'low', 'average', 'average', 'low', 'high', 'average']))


# -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]
def tidy_engage_data(data: Dict[str, List[str]]) -> pd.DataFrame or Tuple[pd.DataFrame, pd.Series]:
    _data = pd.DataFrame(data)
    _data = pd.get_dummies(_data)
    if "Will_go_+" in _data.columns:
        _data.drop(["Will_go_+"], axis=1, inplace=True)
        _data.rename(columns={"Will_go_-": "Will_go"}, inplace=True)
        return _data, _data["Will_go"]
    return _data


df_train, y = tidy_engage_data(df_train)
print(df_train)    # [7 rows x 10 columns]


df_test = dict()
df_test['Looks'] = ['handsome', 'handsome', 'repulsive']
df_test['Alcoholic_beverage'] = ['no', 'yes', 'yes']
df_test['Eloquence'] = ['average', 'high', 'average']
df_test['Money_spent'] = ['lots', 'little', 'lots']
df_test = tidy_engage_data(df_test)
print(df_test)  # [3 rows x 8 columns]


# 保证独热编码后的特征在训练和测试数据中同时存在,
# 即列的值要相同,保留公共特征多余的删除。
def intersect_features(train, test):
    common_feat = list(set(train.keys()) & set(test.keys()))
    return train[common_feat], test[common_feat]


df_train, df_test = intersect_features(train=df_train, test=df_test)
print(df_train)
print(df_test)
print(y)

# **(2) 分类**
dt = DecisionTreeClassifier(criterion='entropy', random_state=17)
dt.fit(df_train, y)

tree_str = export_graphviz(dt, feature_names=df_train.columns, out_file=None, filled=True)
graph = pydotplus.graph_from_dot_data(tree_str)
SVG(graph.create_svg())    # 只能在jupyter中显示（网页中显示）
plt.imshow(plt.imread(BytesIO(graph.create_png()), "png"))
plt.axis("off")
plt.show()


# **（3）自定义决策树**
# 计算熵函数
def entropy(data: list) -> float:
    _data = pd.DataFrame(data)
    _val_counts = _data.value_counts(normalize=True).values
    return sum([-v*np.log2(v) for v in _val_counts])


# 信息增益
def information_gain(root, left, right):
    e_root = entropy(root)
    e_left = entropy(left)
    e_right = entropy(right)
    return e_root - len(left)/len(root)*e_left - len(right)/len(root)*e_right


# 获得所有特征分割后的的信息增益
def gain_feature_to_split(data: pd.DataFrame, label: pd.Series) -> list:
    gain_features = []
    for feature_name in data.columns:
        root = data[feature_name]
        left = label[data[feature_name] == 0]
        right = label[data[feature_name] == 1]
        gain_features.append(information_gain(root, left, right))
    return gain_features


# 获得最好的决策树
def best_tree(data: pd.DataFrame, label: pd.Series):
    gain_features = gain_feature_to_split(data, label)
    idx = gain_features.index(max(gain_features))
    label_left = label[data.iloc[:, idx] == 0]
    label_right = label[data.iloc[:, idx] == 1]
    e_left = entropy(label_left)
    e_right = entropy(label_right)
    print("最好特征名为{}, 分割后的熵分别为{}和{}".format(
                                             data.columns[idx],
                                             e_left,
                                             e_right))
    if e_left != 0.0:
        data_left = data[data.iloc[:, idx] == 0]
        best_tree(data_left, label_left)
    if e_right != 0.0:
        data_right = data[data.iloc[:, idx] == 1]
        best_tree(data_right, label_right)


best_tree(df_train, y)
"""

# **(4)数据预处理案例。人口收入普查数据集。
# 预处理过程很繁琐，记录一下处理流程**

# 加载数据
data_train = pd.read_csv('adult_train.csv', sep=';')
data_test = pd.read_csv('adult_test.csv', sep=';')

# 简单观察一下，describe(), 以及画图（略）
print("观察数据")
print(data_train.tail())
print(data_test.tail())
# 可以看出有缺失的值，例如data_test 中的 Country 缺少
print(data_train.describe(include="all").T)
print(data_test.describe(include="all").T)

# 然后，对数据集进行一些必要的清洗。移除错误的数据.注意 >50K前有一个空格。 >50K.后有.
data_train = data_train[(data_train["Target"] == " >50K")|(data_train["Target"] == " <=50K")]
data_test = data_test[(data_test["Target"] == " >50K.")|(data_test["Target"] == " <=50K.")]

# 将目标值(Target)转换为 0，1 二元数值。
data_train.loc[data_train["Target"] == " <=50K", "Target"] = 0
data_train.loc[data_train["Target"] == " >50K", "Target"] = 1
data_test.loc[data_test["Target"] == " <=50K.", "Target"] = 0
data_test.loc[data_test["Target"] == " >50K.", "Target"] = 1
# 数据检查目标特征值
print("查看或者统计目标值数据：")
print(data_train["Target"].value_counts())
print(data_test["Target"].value_counts())

# 查看其他特征值类型
print("查看数据类型：")
print(data_train.dtypes)
print(data_test.dtypes)
# 类型需要要修改，类型需要一样，这里修改data_test,数据少一些
data_test["Age"] = data_test["Age"].astype(int)
data_test['fnlwgt'] = data_test['fnlwgt'].astype(int)
data_test['Education_Num'] = data_test['Education_Num'].astype(int)
data_test['Capital_Gain'] = data_test['Capital_Gain'].astype(int)
data_test['Capital_Loss'] = data_test['Capital_Loss'].astype(int)
data_test['Hours_per_week'] = data_test['Hours_per_week'].astype(int)

# 对空值需要填充。数值连续性（int特征）的用data_train中位数填充，而类别特征(object)用data_train众数填充。
categorical_columns = [c for c in data_train.columns if data_train[c].dtype.name == "object"]
numerical_columns = [c for c in data_train.columns if data_train[c].dtype.name != "object"]
print('categorical_columns:', categorical_columns)
print('numerical_columns:', numerical_columns)
for c in categorical_columns:
    data_train[c].fillna(data_train[c].mode().values[0], inplace=True)
    data_test[c].fillna(data_train[c].mode().values[0], inplace=True)
for c in numerical_columns:
    data_train[c].fillna(data_train[c].median(), inplace=True)
    data_test[c].fillna(data_train[c].median(), inplace=True)
# 可以看有无缺失的值
print(data_train.describe(include="all").T)
print(data_test.describe(include="all").T)

# 热独编码，数字连续型不能用热独编码， 而类别热独编码
data_train = pd.concat([data_train[numerical_columns],
                        pd.get_dummies(data_train[categorical_columns])], axis=1)
data_test = pd.concat([data_test[numerical_columns],
                       pd.get_dummies(data_test[categorical_columns])], axis=1)
# 热独编码后检查有无缺失特征
print(set(data_train.columns) - set(data_test.columns))     # {'Country_ Holand-Netherlands'}
# 缺失了{'Country_ Holand-Netherlands'}需要补0
data_test['Country_ Holand-Netherlands'] = 0
# 最后检查数据
print(data_train.head())
print(data_test.head())

# 最终结果
X_train = data_train.drop(['Target'], axis=1)
y_train = data_train['Target']
X_test = data_test.drop(['Target'], axis=1)
y_test = data_test['Target']


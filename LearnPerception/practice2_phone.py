"""
@Date  : 2021/6/27
@Refer :
@Title :
@Desc  :
"""

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import pandas as pd

# 读数据
df = pd.read_csv(r"../LearnSeaborn/PhoneOldAndNew.csv")
x = df["area"].values.reshape((-1, 1))
y = pd.get_dummies(df["species"]).values[:, 0].ravel()
print(x, "\n",  y)

# max_iter：可以理解成梯度下降中迭代的次数
# eta0：可以理解成梯度下降中的学习率
# random_state：设置随机种子的，为了每次迭代都有相同的训练集顺序
ppn = Perceptron(max_iter=1000, eta0=0.1, random_state=0)

ppn.fit(x, y)

# 分类测试集，这将返回一个测试结果的数组
y_pred = ppn.predict(x)
print(y_pred)
# 计算模型在测试集上的准确性
print(accuracy_score(y, y_pred))

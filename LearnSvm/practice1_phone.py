# -*- coding: utf-8 -*-
"""
@Date  : 2021/6/27
@Refer :
@Title : 
@Desc  : 
"""
import pandas as pd
from sklearn import svm

# 读数据
df = pd.read_csv(r"../LearnSeaborn/PhoneOldAndNew.csv")
x = df["area"].values.reshape((-1, 1))
y = pd.get_dummies(df["species"]).values[:, 0].ravel()
print(x, "\n",  y)

# 创建svm
classifier = svm.SVC(kernel='linear',gamma=0.1,decision_function_shape='ovo',C=1)
# classifier=svm.SVC(kernel='rbf',gamma=0.1,decision_function_shape='ovo',C=1)
classifier.fit(x, y)

print("SVM-输出训练集的准确率为：", classifier.score(x, y))
y_ = classifier.predict(x)
print(y_)

# -*- coding: utf-8 -*-
"""
@Date  : 2021/6/26
@Desc  : 计算错误个数
"""
import numpy as np
import pandas as pd

df = pd.read_csv(r"../LearnSeaborn/PhoneOldAndNew.csv")
area = df[['area', 'species']].values
print(area)
a1 = 0
a2 = 0
n1 = 0
n2 = 0
for ar in area:
    if ar[1] == "smartphone":
        a1 += ar[0]
        n1 += 1
    else:
        a2 += ar[0]
        n2 += 1
a11 = a1 / n1
a22 = a2 / n2
print(a1, a2, n1, n2)
print(a11, a22)
pipei = []
for ar in area:
    if abs(ar[0] - a11) < abs(ar[0] - a22):
        pipei.append("smartphone")
    else:
        pipei.append("feature Tech")
print(pipei)
error = 0
for i, p in enumerate(pipei):
    if p == area[i][1]:
        error += 0
    else:
        error += 1
        print("pipei error index", i)
print(error)




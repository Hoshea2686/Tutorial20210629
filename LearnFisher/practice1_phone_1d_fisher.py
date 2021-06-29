# -*- coding: utf-8 -*-
"""
@Date  : 2021/6/26
@Desc  : 计算错误个数
"""

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

# 计算s1,s2
u1 = a11
u2 = a22
s1 = 0
s2 = 0
print("---------")
print(u1)
print(u2)
for ar in area:
    if ar[1] == "smartphone":
        s1 += (ar[0] - u1) ** 2
    else:
        s2 += (ar[0] - u2) ** 2
w = (u1 - u2) / (s1 + s2)

y = w * area[:, 0]
print(w, y)
y01 = (49 * u1 + 50 * u2) * w / 99
y00 = (49 * u1 + 50 * u2) / 99
print("````````````")
print(y00)
print(y01)
fisher00 = []
fisher01 = []
for ar in area:
    if ar[0] > y00:
        fisher00.append("smartphone")
    else:
        fisher00.append("feature Tech")
    if ar[0] * w > y01:
        fisher01.append("smartphone")
    else:
        fisher01.append("feature Tech")
error01 = 0
error00 = 0
for i, v in enumerate(fisher00):
    if v == area[i][1]:
        error00 += 0
    else:
        error00 += 1
        print("fisher00 error index", i)
print(error00)
for i, v in enumerate(fisher01):
    if v == area[i][1]:
        error01 += 0
    else:
        error01 += 1
        print("fisher01 error index", i)
print(error01)

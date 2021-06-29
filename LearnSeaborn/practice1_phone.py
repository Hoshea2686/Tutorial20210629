# -*- coding: utf-8 -*-
"""
@Date  : 2021/6/25
@Desc  : 
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv("PhoneOldAndNew.csv")
print(df)

plt.subplot(121)
sns.stripplot(x="species", y="area", data=df)
plt.subplot(122)
sns.stripplot(x="species", y="perimeter", data=df)

sns.relplot(x="perimeter", y="area", hue="species", style="species", data=df, size="species")
plt.show()

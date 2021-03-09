# -*- coding:utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd


# 加载数据
df = pd.read_excel('titanic.xls')
# print(df.shape)  (1309, 14)
# print(df.head())
# print(df.tail())


# 去掉无用字段
df.drop(['body', 'name', 'ticket'], 1, inplace=True)
# print(df.info())#可以查看数据类型
df.convert_objects(convert_numeric=True)#将object格式转float64格式
df.fillna(0, inplace=True)  # 把NaN替换为0

# 把字符串映射为数字，例如{female:1, male:0}
df_map = {}  # 保存映射关系
cols = df.columns.values
print('cols:',cols)
for col in cols:
    if df[col].dtype != np.int64 and df[col].dtype != np.float64:
        temp = {}
        x = 0
        for ele in set(df[col].values.tolist()):
            if ele not in temp:
                temp[ele] = x
                x += 1

        df_map[df[col].name] = temp
        df[col] = list(map(lambda val: temp[val], df[col]))

for key, value in df_map.items():
   print(key,value)
# print(df.head())

# 由于是非监督学习，不使用label
x = np.array(df.drop(['survived'], 1).astype(float))
# 将每一列特征标准化为标准正太分布，注意，标准化是针对每一列而言的
x = preprocessing.scale(x)

clf = KMeans(n_clusters=2)
clf.fit(x)
# 上面已把数据分成两组

# 下面计算分组准确率是多少
y = np.array(df['survived'])

correct = 0
for i in range(len(x)):
    predict_data = np.array(x[i].astype(float))
    predict_data = predict_data.reshape(-1, len(predict_data))
    predict = clf.predict(predict_data)
    # print(predict[0], y[i])
    if predict[0] == y[i]:
        correct += 1

print(correct * 1.0 / len(x))

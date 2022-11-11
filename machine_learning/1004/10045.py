from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd

items = ['선풍기', '냉장고', '믹서', '믹서', '컴퓨터', '전자렌지', '선풍기']

# label encoding
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)

# 2차원 데이터로 변환
labels = labels.reshape(-1, 1)

# one-hot encoding
oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)

print('one-hot encoding data')
print(oh_labels.toarray())
# [[0. 0. 1. 0. 0.]
#  [1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 1.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 1. 0. 0.]]

print('one-hot encoding dimension')
print(oh_labels.shape)
# (7, 5)

df = pd.DataFrame({'item': items})
print(pd.get_dummies(df))
#    item_냉장고  item_믹서  item_선풍기  item_전자렌지  item_컴퓨터
# 0           0          0           1            0            0
# 1           1          0           0            0            0
# 2           0          1           0            0            0
# 3           0          1           0            0            0
# 4           0          0           0            0            1
# 5           0          0           0            1            0
# 6           0          0           1            0            0


# StandardScaler
from sklearn.preprocessing import StandardScaler

# 표준화를 위한 테스트용 데이터 생성
scaler = StandardScaler()
# StandardScaler 로 데이터셋 변환, fit() 과 transform() 호출
from sklearn.datasets import load_iris

iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data = iris_data, columns = iris.feature_names)
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

# transform()시 scale 변환된 데이터 셋이 numpy ndarry로 반환돼 이를 DataFrame으로 변환
iris_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print('feature들의 평균값')
print(iris_scaled.mean())
# sepal length (cm)   -1.690315e-15
# sepal width (cm)    -1.842970e-15
# petal length (cm)   -1.698641e-15
# petal width (cm)    -1.409243e-15
# dtype: float64

print('feature들의 분산값')
print(iris_scaled.var())
# sepal length (cm)    1.006711
# sepal width (cm)     1.006711
# petal length (cm)    1.006711
# petal width (cm)     1.006711
# dtype: float64

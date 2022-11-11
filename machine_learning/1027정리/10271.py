# K-Means clustering 연습
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# K-Means clustering
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0)
kmeans.fit(df)
df['cluster'] = kmeans.labels_

# 타깃별 중심점
df['target'] = iris.target
iris_result = df.groupby(['target', 'cluster'])['sepal_length'].count()
print(iris_result)
# target  cluster
# 0       1          50
# 1       0          48
#         2           2
# 2       0          14
#         2          36
# Name: sepal_length, dtype: int64

# 2차원 PCA로 축소
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(iris.data)
df['pca_x'] = pca_transformed[:, 0]
df['pca_y'] = pca_transformed[:, 1]

# 클러스터 값이 0, 1, 2인 데이터를 별도의 인덱스로 추출
marker0_ind = df[df['cluster'] == 0].index
marker1_ind = df[df['cluster'] == 1].index
marker2_ind = df[df['cluster'] == 2].index

# 각각 marker 표시
plt.scatter(x=df.loc[marker0_ind, 'pca_x'], y=df.loc[marker0_ind, 'pca_y'], marker='o')
plt.scatter(x=df.loc[marker1_ind, 'pca_x'], y=df.loc[marker1_ind, 'pca_y'], marker='s')
plt.scatter(x=df.loc[marker2_ind, 'pca_x'], y=df.loc[marker2_ind, 'pca_y'], marker='^')

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('3 Clusters Visualization by 2 PCA Components')
plt.show()

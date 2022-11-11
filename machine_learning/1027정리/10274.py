from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris = load_iris()
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

df = pd.DataFrame(data = iris.data, columns=features)
df['target'] = iris.target

# DBSCAN의 eps 값과 min_samples 값을 변경하면서 클러스터링 결과를 확인
dbscan = DBSCAN(eps=0.6, min_samples=16, metric='euclidean')
dbscan_labels = dbscan.fit_predict(iris.data)

df['dbscan_cluster'] = dbscan_labels
df['target'] = iris.target

iris_result = df.groupby(['target'])['dbscan_cluster'].value_counts()
# print(iris_result)
# target  dbscan_cluster
# 0        0                49
#         -1                 1
# 1        1                46
#         -1                 4
# 2        1                42
#         -1                 8
# Name: dbscan_cluster, dtype: int64

### 클러스터 결과를 담은 DataFrame과 사이킷런의 Cluster 객체등을 인자로 받아 클러스터링 결과를 시각화하는 함수
def visualize_cluster_plot(clusterobj, dataframe, label_name, iscenter=True):
    if iscenter:
        centers = clusterobj.cluster_centers_

    unique_labels = np.unique(dataframe[label_name].values)
    markers = ['o', 's', '^', 'x', '*']
    isNoise = False

    for label in unique_labels:
        label_cluster = dataframe[dataframe[label_name] == label]
        if label == -1:
            cluster_legend = 'Noise'
            isNoise = True
        else:
            cluster_legend = 'Cluster ' + str(label)

        plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], s=70,
                    edgecolor='k', marker=markers[label], label=cluster_legend)

        if iscenter:
            center_x_y = centers[label]
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=250, color='white',
                        alpha=0.9, edgecolor='k', marker=markers[label])
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k', \
                        edgecolor='k', marker='$%d$' % label)
    if isNoise:
        legend_loc = 'upper center'
    else:
        legend_loc = 'upper right'

    plt.legend(loc=legend_loc)
    plt.show()

# DBSCAN 적용
# 2차원으로 시각화하기 위해 n_components=2로 PCA 변환
pca = PCA(n_components=2, random_state=0)
pca_transformed = pca.fit_transform(iris.data)
# pca변환값을 DataFrame에 추가
df['ftr1'] = pca_transformed[:, 0]
df['ftr2'] = pca_transformed[:, 1]

visualize_cluster_plot(dbscan, df, 'dbscan_cluster', iscenter=False)
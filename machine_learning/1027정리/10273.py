# iris 실루엣 계수 계산
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris = load_iris()

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df = pd.DataFrame(iris.data, columns=features)

# K-Means clustering
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0).fit(df)
# 데이터당 클러스터 할당
kmeans_cluster_labels = kmeans.predict(iris.data)
df['cluster'] = kmeans.labels_
df['target'] = iris.target

# 모든 개별 데이터에 실루엣 계수를 구함
score_samples = silhouette_samples(iris.data, df['cluster'])
df['silhouette_coeff'] = score_samples
average_score = silhouette_score(iris.data, df['cluster'])
# print('silhouette analysis score: {0:.3f}'.format(average_score))
# silhouette analysis score: 0.553

# print(df.groupby('cluster')['silhouette_coeff'].mean())
# cluster
# 0    0.417320
# 1    0.798140 -> 군집화가 가장 잘 됨
# 2    0.451105
# Name: silhouette_coeff, dtype: float64

# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html?highlight=silhouette

# K-Means 결과
kmeans_result = df.groupby(['target'])['cluster'].value_counts()
print(kmeans_result)
# target  cluster
# 0       1          50
# 1       0          48
#         2           2
# 2       2          36
#         0          14
# Name: cluster, dtype: int64

#GMM과 kmeans 비교
gmm = GaussianMixture(n_components=3, random_state=0).fit(iris.data)
gmm_cluster_labels = gmm.predict(iris.data)

# 클러스터링 결과 저장
df['gmm_cluster'] = gmm_cluster_labels
df['target'] = iris.target

gmm_result = df.groupby(['target'])['gmm_cluster'].value_counts()
print(gmm_result)
# target  gmm_cluster
# 0       0              50
# 1       2              45
#         1               5
# 2       1              50
# Name: gmm_cluster, dtype: int64
# -> iris 분포에선 gmm이 더 잘 분류함

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

        plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], s=70, edgecolor='k', marker=markers[label], label=cluster_legend)

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

# make_blobs() 로 300개의 데이터 셋, 3개의 cluster 셋, cluster_std=0.5 을 만듬.
X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=0.5, random_state=0)

# 길게 늘어난 타원형의 데이터 셋을 생성하기 위해 변환함.
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
# feature 데이터 셋과 make_blobs( ) 의 y 결과 값을 DataFrame으로 저장
clusterDF = pd.DataFrame(data=X_aniso, columns=['ftr1', 'ftr2'])
clusterDF['target'] = y
# 생성된 데이터 셋을 target 별로 다른 marker 로 표시하여 시각화 함.
visualize_cluster_plot(None, clusterDF, 'target', iscenter=False)

# k-means 분류 시각화
kmeans = KMeans(3, random_state=0)
kmeans_label = kmeans.fit_predict(X_aniso)
clusterDF['kmeans_label'] = kmeans_label

visualize_cluster_plot(kmeans, clusterDF, 'kmeans_label', iscenter=True)
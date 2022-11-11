import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 임의의 데이터 생성
x, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.8, random_state=0)
unique, counts = np.unique(y, return_counts=True)

df = pd.DataFrame(x, columns=['ftr1', 'ftr2'])
df['target'] = y

target_list = np.unique(y)
markers = ['o', 's', '^', 'P', 'D', 'H', 'x']
# target이 3개 이므로 target_list는 [0, 1, 2]가 됨
# target == 0 ~ 2로 marker생성
for target in target_list:
    target_cluster = df[df['target'] == target]
    plt.scatter(x=target_cluster['ftr1'], y=target_cluster['ftr2'], edgecolor='k', marker=markers[target])
# plt.show()

# kmeans 군집화 후 개별 클러스터의 중심점 시각화
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=200, random_state=0)
cluster_labels = kmeans.fit_predict(x)
df['kmeans_label'] = cluster_labels

centers = kmeans.cluster_centers_
unique_labels = np.unique(cluster_labels)
markers = ['o', 's', '^', 'P', 'D', 'H', 'x']

for label in unique_labels:
    label_cluster = df[df['kmeans_label'] == label]
    center_x_y = centers[label]
    plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], edgecolor='k', marker=markers[label])
    # 중심점 위치 시각화
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=200, color='white', alpha=0.9, edgecolor='k', marker=markers[label])
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k', edgecolor='k', marker='$%d$' % label)
plt.show()

print(df.groupby('target')['kmeans_label'].value_counts())
# target  kmeans_label
# 0       0               66
#         1                1 -> 모이지 못한 부분 1개
# 1       2               67
# 2       1               65
#         2                1 -> 모이지 못한 부분 1개
# Name: kmeans_label, dtype: int64

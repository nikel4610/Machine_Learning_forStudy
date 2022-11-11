import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold, StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score

from imblearn.over_sampling import SMOTE
from imblearn.metrics import classification_report_imbalanced

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('D:/vsc_project/machinelearning_study/heart_disease_health_indicators_BRFSS2015.csv')
std_scaler = StandardScaler()
rob_scaler = RobustScaler()

# print(df.isnull().sum())
# print(df.describe())

df.drop(['Education', 'Income'], axis=1, inplace=True)
# df['scaled_age'] = std_scaler.fit_transform(df['Age'].values.reshape(-1, 1))
# scaled_age = df['scaled_age']
# df.drop(['Age'], axis=1, inplace=True)
# df.drop(['scaled_age'], axis=1, inplace=True)
# df.insert(0, 'scaled_age', scaled_age)

x = df.drop(['HeartDiseaseorAttack'], axis=1)
y = df['HeartDiseaseorAttack']

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
for train_index, test_index in sss.split(x, y):
    original_Xtrain, original_Xtest = x.iloc[train_index], x.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

df = df.sample(frac=1)

heartattack_df = df.loc[df['HeartDiseaseorAttack'] == 1]
non_heartattack_df = df.loc[df['HeartDiseaseorAttack'] == 0][:len(heartattack_df)]

normal_distributed_df = pd.concat([heartattack_df, non_heartattack_df])

new_df = normal_distributed_df.sample(frac=1, random_state=0)

X = new_df.drop('HeartDiseaseorAttack', axis=1)
y = new_df['HeartDiseaseorAttack']

# t-SNE
X_reduced_tsne = TSNE(n_components=2, random_state=0).fit_transform(X.values)
print('t-SNE done')

# PCA
X_reduced_pca = PCA(n_components=2, random_state=0).fit_transform(X.values)
print('PCA done')

# TruncatedSVD
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=0).fit_transform(X.values)
print('Truncated SVD done')

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
f.suptitle('Clusters after Dimensionality Reduction', fontsize=14)
labels = ['No Heart Disease', 'Heart Disease']
blue_patch = mpatches.Patch(color='#0A0AFF', label=labels[0])
red_patch = mpatches.Patch(color='#AF0000', label=labels[1])

# t-SNE scatter plot
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax1.set_title('t-SNE', fontsize=14)
ax1.grid(True)
ax1.legend(handles=[blue_patch, red_patch])

# PCA scatter plot
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax2.set_title('PCA', fontsize=14)
ax2.grid(True)
ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)
ax3.grid(True)
ax3.legend(handles=[blue_patch, red_patch])
plt.show()
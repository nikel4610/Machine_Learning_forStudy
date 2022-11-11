import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv('D:/vsc_project/machinelearning_study/titanic_train.csv')
# print(df.head())

# titanic 데이터셋 PCA변환
y_target = df['Survived']
x_features = df.drop('Survived', axis=1)

# NaN값 처리
x_features['Age'].fillna(x_features['Age'].mean(), inplace=True)
x_features['Cabin'].fillna('N', inplace=True)
x_features['Embarked'].fillna('N', inplace=True)
# print(x_features.isnull().sum())

x_features['Cabin'] = x_features['Cabin'].str[:1]
# print(x_features['Cabin'].head())

corr = x_features.corr()
plt.figure(figsize=(14, 14))
sns.heatmap(corr, annot=True, fmt='.1g')
# plt.show()

# 상관관계가 높은 feature들을 PCA 변환
col_names = ['Pclass', 'SibSp', 'Parch', 'Fare']
x_features = x_features[col_names]

scaler = StandardScaler()
x_features_scaled = scaler.fit_transform(x_features)
pca = PCA(n_components=3)
x_features_pca = pca.fit_transform(x_features_scaled)
scores_pca = cross_val_score(RandomForestClassifier(), x_features_pca, y_target, scoring='accuracy', cv=5)

print('PCA변환 데이터 세트의 개별 CV 정확도:', scores_pca)
print('PCA변환 데이터 세트의 개별 CV 정확도 평균: {0: .4f}'.format(scores_pca.mean()))
# PCA변환 데이터 세트의 개별 CV 정확도: [0.63687151 0.65730337 0.70224719 0.75842697 0.71348315]
# PCA변환 데이터 세트의 개별 CV 정확도 평균:  0.6948

# 원본 데이터 세트의 개별 CV 정확도
# 교차 검증 0 정확도: 0.7430
# 교차 검증 1 정확도: 0.7753
# 교차 검증 2 정확도: 0.7921
# 교차 검증 3 정확도: 0.7865
# 교차 검증 4 정확도: 0.8427
# 평균 검증 정확도: 0.7879
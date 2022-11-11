import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_excel('D:/vsc_project/machinelearning_study/default of credit card clients.xls', header=1, sheet_name='Data').iloc[0:, 1:]
# print(df.head())

df.rename(columns={'PAY_0':'PAY_1', 'default payment next month':'default'}, inplace=True)
y_target = df['default']
x_features = df.drop('default', axis=1)
# print(y_target.value_counts())

corr = x_features.corr()
plt.figure(figsize=(14, 14))
sns.heatmap(corr, annot=True, fmt='.1g')
plt.show()

# 상관관계 높은 feature들을 PCA 변환 후 확인
cols_bills = ['BILL_AMT' + str(i) for i in range(1, 7)]
print('대상 피처명:', cols_bills)
# 대상 피처명: ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

scaler = StandardScaler()
df_cols_scaled = scaler.fit_transform(x_features[cols_bills])
# pca = PCA(n_components=2)
# pca.fit(df_cols_scaled)
# print('PCA Component별 변동성:', pca.explained_variance_ratio_)
# PCA Component별 변동성: [0.90555253 0.0509867 ]

# 원본 데이터와 pca변환 데이터 간 랜덤 포레스트 비교
ref = RandomForestClassifier(n_estimators=300, random_state=156)

scores = cross_val_score(ref, x_features, y_target, scoring='accuracy', cv=3)
df_scaled = scaler.fit_transform(x_features)
pca = PCA(n_components=7)
df_pca = pca.fit_transform(df_scaled)
scores_pca = cross_val_score(ref, df_pca, y_target, scoring='accuracy', cv=3)

print('원본 데이터 CV=3 정확도:', scores)
print('원본 데이터 CV=3 정확도 평균:', scores.mean())
# 원본 데이터 CV=3 정확도: [0.8083 0.8196 0.8232]
# 원본 데이터 CV=3 정확도 평균: 0.8170333333333333

print('PCA 변환 데이터 CV=3 정확도:', scores_pca)
print('PCA 변환 데이터 CV=3 정확도 평균:', scores_pca.mean())
# PCA 변환 데이터 CV=3 정확도: [0.7939 0.8004 0.8021]
# PCA 변환 데이터 CV=3 정확도 평균: 0.7988
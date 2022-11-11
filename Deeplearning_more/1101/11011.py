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

df = pd.read_csv('D:/vsc_project/machinelearning_study/creditcard.csv')
# 데이터 30개 행 확인
# print(df.head(30))

# Missing 여부 확인
# print(df.isnull().value_counts())

# 데이터의 분포 확인
# print(df.describe())

std_scaler = StandardScaler()
rob_scaler = RobustScaler() # 이상치 무시하고 골고루 분포

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']
df.drop(['Time', 'Amount'], axis=1, inplace=True)

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

x = df.drop('Class', axis=1)
y = df['Class']

# 층위 샘플링으로 데이터 나누기
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

for train_index, test_index in sss.split(x, y):
    original_Xtrain, original_Xtest = x.iloc[train_index], x.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

# 클래스의 skew 정도가 매우 높기 때문에 클래스간 분포를 맞추는 것이 필요합니다.
# subsample 구축 전 셔플링을 통해 레이블이 한쪽에 몰려있지 않도록 하겠습니다.

df = df.sample(frac=1)

# 데이터 준비
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# 데이터 셔플하기
new_df = normal_distributed_df.sample(frac=1, random_state=0)

# PCA를 사용한 차원 축소
# 차원 축소할 데이터 준비
X = new_df.drop('Class', axis=1)
y = new_df['Class']
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
f.suptitle('Clusters after Dimensionality Reduction', fontsize=16)
labels = ['No Fraud', 'Fraud']
blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')

# # t-SNE scatter plot
# ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
# ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
# ax1.set_title('t-SNE', fontsize=14)
# ax1.grid(True)
# ax1.legend(handles=[blue_patch, red_patch])
#
# # PCA scatter plot
# ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
# ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
# ax2.set_title('PCA', fontsize=14)
# ax2.grid(True)
# ax2.legend(handles=[blue_patch, red_patch])
#
# # TruncatedSVD scatter plot
# ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
# ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
# ax3.set_title('Truncated SVD', fontsize=14)
# ax3.grid(True)
# ax3.legend(handles=[blue_patch, red_patch])
# plt.show()

# 재구축한 데이터의 클래스 분포 확인하기
new_df.groupby(by=['Class']).count()

# X와 y 데이터 셋 만들기
X = new_df.drop('Class', axis=1)
y = new_df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 모델 인풋에 들어가기 위한 데이터의 형태 바꾸기
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values


classifiers = {
    "Logisitic Regression": LogisticRegression(),
    "K Nearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "LigthGBM Classifier": LGBMClassifier(),
}

for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
#     print("Classifiers: ", classifier.__class__.__name__,':',round(training_score.mean(), 2) * 100, '% accuracy score')
# Classifiers:  LogisticRegression : 94.0 % accuracy score
# Classifiers:  KNeighborsClassifier : 93.0 % accuracy score
# Classifiers:  SVC : 93.0 % accuracy score
# Classifiers:  DecisionTreeClassifier : 89.0 % accuracy score
# Classifiers:  RandomForestClassifier : 93.0 % accuracy score
# Classifiers:  GradientBoostingClassifier : 93.0 % accuracy score
# Classifiers:  LGBMClassifier : 93.0 % accuracy score

# 분류 결과 확인
for key, classifier in classifiers.items():
    y_pred = classifier.predict(original_Xtest)
    results = classification_report(original_ytest, y_pred)
    # print(classifier.__class__.__name__, '-------','\n', results)
# LogisticRegression -------
#                precision    recall  f1-score   support
#
#            0       1.00      0.97      0.99     85295
#            1       0.06      0.96      0.11       148
#
#     accuracy                           0.97     85443
#    macro avg       0.53      0.97      0.55     85443
# weighted avg       1.00      0.97      0.98     85443
#
# KNeighborsClassifier -------
#                precision    recall  f1-score   support
#
#            0       1.00      0.97      0.98     85295
#            1       0.05      0.95      0.10       148
#
#     accuracy                           0.97     85443
#    macro avg       0.53      0.96      0.54     85443
# weighted avg       1.00      0.97      0.98     85443
#
# SVC -------
#                precision    recall  f1-score   support
#
#            0       1.00      0.98      0.99     85295
#            1       0.07      0.95      0.13       148
#
#     accuracy                           0.98     85443
#    macro avg       0.53      0.96      0.56     85443
# weighted avg       1.00      0.98      0.99     85443
#
# DecisionTreeClassifier -------
#                precision    recall  f1-score   support
#
#            0       1.00      0.88      0.94     85295
#            1       0.01      0.99      0.03       148
#
#     accuracy                           0.88     85443
#    macro avg       0.51      0.94      0.48     85443
# weighted avg       1.00      0.88      0.94     85443
#
# RandomForestClassifier -------
#                precision    recall  f1-score   support
#
#            0       1.00      0.98      0.99     85295
#            1       0.07      0.99      0.14       148
#
#     accuracy                           0.98     85443
#    macro avg       0.54      0.98      0.56     85443
# weighted avg       1.00      0.98      0.99     85443
#
# GradientBoostingClassifier -------
#                precision    recall  f1-score   support
#
#            0       1.00      0.97      0.98     85295
#            1       0.05      0.99      0.09       148
#
#     accuracy                           0.97     85443
#    macro avg       0.52      0.98      0.54     85443
# weighted avg       1.00      0.97      0.98     85443
#
# LGBMClassifier -------
#                precision    recall  f1-score   support
#
#            0       1.00      0.97      0.99     85295
#            1       0.06      0.99      0.11       148
#
#     accuracy                           0.97     85443
#    macro avg       0.53      0.98      0.55     85443
# weighted avg       1.00      0.97      0.98     85443

for key, classifier in classifiers.items():
    y_pred = classifier.predict(original_Xtest)
    cm = confusion_matrix(original_ytest, y_pred)
   #  print(classifier.__class__.__name__, '\n', cm, '\n')
# LogisticRegression
#  [[82947  2348]
#  [    6   142]]
#
# KNeighborsClassifier
#  [[82621  2674]
#  [    7   141]]
#
# SVC
#  [[83328  1967]
#  [    7   141]]
#
# DecisionTreeClassifier
#  [[75386  9909]
#  [    1   147]]
#
# RandomForestClassifier
#  [[83452  1843]
#  [    2   146]]
#
# GradientBoostingClassifier
#  [[82436  2859]
#  [    2   146]]
#
# LGBMClassifier
#  [[82812  2483]
#  [    2   146]]

# SMOTE로 오버샘플링 후 확인
sm = SMOTE()
X_resampled, y_resampled = sm.fit_resample(original_Xtrain,list(original_ytrain))

# print('Before SMOTE, original X_train: {}'.format(original_Xtrain.shape))
# print('Before SMOTE, original y_train: {}'.format(np.array(original_ytrain).shape))
# print('After SMOTE, resampled original X_train: {}'.format(X_resampled.shape))
# print('After SMOTE, resampled original y_train: {} \n'.format(np.array(y_resampled).shape))
# print("Before SMOTE, fraud counts: {}".format(sum(np.array(original_ytrain)==1)))
# print("Before SMOTE, non-fraud counts: {}".format(sum(np.array(original_ytrain)==0)))
# print("After SMOTE, fraud counts: {}".format(sum(np.array(y_resampled)==1)))
# print("After SMOTE, non-fraud counts: {}".format(sum(np.array(y_resampled)==0)))
# Before SMOTE, original X_train: (199364, 30)
# Before SMOTE, original y_train: (199364,)
# After SMOTE, resampled original X_train: (398040, 30)
# After SMOTE, resampled original y_train: (398040,)
#
# Before SMOTE, fraud counts: 344
# Before SMOTE, non-fraud counts: 199020
# After SMOTE, fraud counts: 199020
# After SMOTE, non-fraud counts: 199020

# 분류 모형 구현
# 불균형 클래스 weight지정
w = {1:0, 1:99}

# 모델 피팅
logreg_weighted = LogisticRegression(random_state=0, class_weight=w)
logreg_weighted.fit(original_Xtrain,original_ytrain)

# 예측값 구하기
y_pred = logreg_weighted.predict(original_Xtest)

# # 예측결과 확인하기
# print('Logistic Regression ------ Weighted')
# print(f'Accuracy: {accuracy_score(original_ytest,y_pred)}')
# print('\n')
# print(f'Confusion Matrix: \n{confusion_matrix(original_ytest, y_pred)}')
# print('\n')
# print(f'Recall: {recall_score(original_ytest,y_pred)}')
#
# label = ['non-fraud', 'fraud']
# print(classification_report_imbalanced(original_ytest, y_pred, target_names=label))

# ogistic Regression ------ Weighted
# Accuracy: 0.9955759980337769
#
#
# Confusion Matrix:
# [[84930   365]
#  [   13   135]]
#
#
# Recall: 0.9121621621621622
#                    pre       rec       spe        f1       geo       iba       sup
#
#   non-fraud       1.00      1.00      0.91      1.00      0.95      0.92     85295
#       fraud       0.27      0.91      1.00      0.42      0.95      0.90       148
#
# avg / total       1.00      1.00      0.91      1.00      0.95      0.92     85443

# 랜덤포레스트 확인
rf = RandomForestClassifier(random_state=0, class_weight=w)
rf.fit(original_Xtrain,original_ytrain)
y_pred = rf.predict(original_Xtest)

# print('Random Forest ------ Weighted')
# print(f'Accuracy: {accuracy_score(original_ytest,y_pred)}')
# print('\n')
# print(f'Confusion Matrix: \n{confusion_matrix(original_ytest, y_pred)}')
# print('\n')
# print(f'Recall: {recall_score(original_ytest,y_pred)}')
# Random Forest ------ Weighted
# Accuracy: 0.9996254813150287
# Confusion Matrix:
# [[85290     5]
#  [   27   121]]
# Recall: 0.8175675675675675
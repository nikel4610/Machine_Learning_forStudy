import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

titanic_df = pd.read_csv('titanic_train.csv')
# print(titanic_df.head())
# print(titanic_df.info())

titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
titanic_df['Cabin'].fillna('N', inplace=True)
titanic_df['Embarked'].fillna('N', inplace=True)
# null값 확인
# print(titanic_df.isnull().sum())

# 값 분포 확인
# print(titanic_df.['Sex'].value_counts())
# print(titanic_df.['Cabin'].value_counts())
# print(titanic_df.['Embarked'].value_counts())


# Cabin 값 정리
titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]

def encode_features(dataDF):
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(dataDF[feature])
        dataDF[feature] = le.transform(dataDF[feature])
    return dataDF

titanic_df = encode_features(titanic_df)
# print(titanic_df.head())
#    PassengerId  Survived  Pclass  ...     Fare  Cabin  Embarked
# 0            1         0       3  ...   7.2500      7         3
# 1            2         1       1  ...  71.2833      2         0
# 2            3         1       3  ...   7.9250      7         3
# 3            4         1       1  ...  53.1000      2         3
# 4            5         0       3  ...   8.0500      7         3

# Null 처리 함수
def fillna(df):
    df['Age'].fillna(df['Age'].mean(),inplace=True)
    df['Cabin'].fillna('N',inplace=True)
    df['Embarked'].fillna('N',inplace=True)
    df['Fare'].fillna(0,inplace=True)
    return df

# 머신러닝 알고리즘에 불필요한 속성 제거
def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
    return df

# 레이블 인코딩 수행. 
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

# 앞에서 설정한 Data Preprocessing 함수 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df

# 원본 데이터를 재로딩 하고, feature 데이터 셋과 label 데이터 셋 추출
titanic_df = pd.read_csv('titanic_train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived',axis=1)

X_titanic_df = transform_features(X_titanic_df)

X_train, X_test, Y_train, Y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=11)

# 결정트리, Random Forest, 로지스틱 회귀를 위한 사이킷런 Classifier 클래스 설정
# 실무에서는 셋다 안씀
dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression()

# DecisionTreeClassifier 학습/예측/평가
dt_clf.fit(X_train, Y_train)
dt_pred = dt_clf.predict(X_test)
# print('DecisionTreeClassifier 정확도: {0:.4f}'.format(accuracy_score(Y_test, dt_pred)))
# DecisionTreeClassifier 정확도: 0.7877

# RandomForestClassifier 학습/예측/평가
rf_clf.fit(X_train, Y_train)
rf_pred = rf_clf.predict(X_test)
# print('RandomForestClassifier 정확도: {0:.4f}'.format(accuracy_score(Y_test, rf_pred)))
# RandomForestClassifier 정확도: 0.8547

# LogisticRegression 학습/예측/평가
lr_clf.fit(X_train, Y_train)
lr_pred = lr_clf.predict(X_test)
# print('LogisticRegression 정확도: {0:.4f}'.format(accuracy_score(Y_test, lr_pred)))
# LogisticRegression 정확도: 0.8492

scores = cross_val_score(dt_clf, X_titanic_df, y_titanic_df, cv=5)
for iter_count, accuracy in enumerate(scores):
    print('교차 검증 {0} 정확도: {1:.4f}'.format(iter_count, accuracy))

print('평균 검증 정확도: {0:.4f}'.format(np.mean(scores)))

# 교차 검증 0 정확도: 0.7430
# 교차 검증 1 정확도: 0.7753
# 교차 검증 2 정확도: 0.7921
# 교차 검증 3 정확도: 0.7865
# 교차 검증 4 정확도: 0.8427
# 평균 검증 정확도: 0.7879 

parameters = {'max_depth':[2,3,5,10], 'min_samples_split':[2,3,5], 'min_samples_leaf':[1,5,8]}

grid_dcif = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=5)
grid_dcif.fit(X_train, Y_train)

print('GridSearchCV 최적 하이퍼 파라미터: ', grid_dcif.best_params_)
print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_dcif.best_score_))    
# GridSearchCV 최적 하이퍼 파라미터:  {'max_depth': 3, 'min_samples_leaf': 5, 'min_samples_split': 2}
# GridSearchCV 최고 정확도: 0.7992

best_dcif = grid_dcif.best_estimator_

# GridSearchCV의 최적 하이퍼 파라미터로 학습된 Estimator로 예측 및 평가 수행
dpredictions = best_dcif.predict(X_test)
accuracy = accuracy_score(Y_test, dpredictions)
print('테스트 세트에서의 DecisionTreeClassifier 정확도: {0:.4f}'.format(accuracy))
# 테스트 세트에서의 DecisionTreeClassifier 정확도: 0.8715
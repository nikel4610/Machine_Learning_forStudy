from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

# 데이터셋 로딩
iris = load_iris()

# iris 데이터셋 에서 feature만으로 된 데이터를 numpy로 가지고 있음
iris_data = iris.data

# iris.target은 붓꽃 데이터셋에서 레이블(껼정 값) 데이터를 numpy로 가지고 있음
iris_label = iris.target

# print('iris target 값:', iris_label)
# print('iris target 명:', iris.target_names)

# iris target 값: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]
# iris target 명: ['setosa' 'versicolor' 'virginica']

# 붓꽃 데이터셋을 DataFrame으로 변환
iris_df = pd.DataFrame(data = iris_data, columns = iris.feature_names)
iris_df['label'] = iris.target

# print(iris_df.head())
#    sepal length (cm)  sepal width (cm)  ...  petal width (cm)  label
# 0                5.1               3.5  ...               0.2      0
# 1                4.9               3.0  ...               0.2      0
# 2                4.7               3.2  ...               0.2      0

# [3 rows x 5 columns]

X_train, X_test, Y_train, Y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)

# DecisionTreeClassifier 객체 생성
dt_clf = DecisionTreeClassifier(random_state=11)

# 학습 수행
dt_clf.fit(X_train, Y_train)

# 학습이 완료된 DecisionTreeClassifier 객체에서 테스트 데이터 세트로 예측 수행
pred = dt_clf.predict(X_test)

# 예측 정확도 평가
print('예측 정확도: {0:.4f}'.format(accuracy_score(Y_test, pred)))

# 예측 정확도: 0.9333

# # 학습 데이터셋으로 예측 수행
# pred = dt_clf.predict(X_train)
# print('예측 정확도: {0:.4f}'.format(accuracy_score(Y_train, pred)))

# # 예측 정확도: 1.0000
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris

iris = load_iris()
features = iris.data

skf = StratifiedKFold(n_splits=3)
n_iter = 0

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target
label = iris.target

for train_index, test_index in skf.split(iris_df, iris_df['label']):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print('## 교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포:\n', label_train.value_counts())
    print('검증 레이블 데이터 분포:\n', label_test.value_counts())

# ## 교차 검증: 1
# 학습 레이블 데이터 분포:
#  2    34
# 0    33
# 1    33
# Name: label, dtype: int64
# 검증 레이블 데이터 분포:
#  0    17
# 1    17
# 2    16
# Name: label, dtype: int64
# ## 교차 검증: 2
# 학습 레이블 데이터 분포:
#  1    34
# 0    33
# 2    33
# Name: label, dtype: int64
# 검증 레이블 데이터 분포:
#  0    17
# 2    17
# 1    16
# Name: label, dtype: int64
# ## 교차 검증: 3
# 학습 레이블 데이터 분포:
#  0    34
# 1    33
# 2    33
# Name: label, dtype: int64
# 검증 레이블 데이터 분포:
#  1    17
# 2    17
# 0    16
# Name: label, dtype: int64

dt_clf = DecisionTreeClassifier(random_state=156)

skfold = StratifiedKFold(n_splits=3)
cv_accuracy = []
n_iter = 0

# StratifiedKFold의 split() 호출 시 반드시 레이블 데이터 세트도 추가 입력 필요
for train_index, test_index  in skfold.split(features, label):
    # split( )으로 반환된 인덱스를 이용하여 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    #학습 및 예측 
    dt_clf.fit(X_train , y_train)    
    pred = dt_clf.predict(X_test)

    # 반복 시 마다 정확도 측정 
    n_iter += 1
    accuracy = np.round(accuracy_score(y_test,pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'
          .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,test_index))
    cv_accuracy.append(accuracy)

# #1 교차 검증 정확도 :0.98, 학습 데이터 크기: 100, 검증 데이터 크기: 50
# #1 검증 세트 인덱스:[  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  50
#   51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66 100 101
#  102 103 104 105 106 107 108 109 110 111 112 113 114 115]

# #2 교차 검증 정확도 :0.94, 학습 데이터 크기: 100, 검증 데이터 크기: 50
# #2 검증 세트 인덱스:[ 17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  67
#   68  69  70  71  72  73  74  75  76  77  78  79  80  81  82 116 117 118
#  119 120 121 122 123 124 125 126 127 128 129 130 131 132]

# #3 교차 검증 정확도 :0.98, 학습 데이터 크기: 100, 검증 데이터 크기: 50
# #3 검증 세트 인덱스:[ 34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  83  84
#   85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 133 134 135
#  136 137 138 139 140 141 142 143 144 145 146 147 148 149]

# 교차 검증별 정확도 및 평균 정확도 계산
print('## 교차 검증별 정확도:', np.round(cv_accuracy, 4))
print('## 평균 검증 정확도:', np.mean(cv_accuracy))

## 교차 검증별 정확도: [0.98 0.94 0.98]
## 평균 검증 정확도: 0.9666666666666667
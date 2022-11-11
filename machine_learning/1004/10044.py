from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

# GridSearchCV

# 데이터 로딩하고 학습데이터와 테스트데이터로 분리
iris = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=121)

dtree = DecisionTreeClassifier()

# 파라미터를 딕셔너리 형태로 설정
parameters = {'max_depth': [1, 2, 3], 'min_samples_split': [2, 3]}

# param_grid의 하이퍼 파라미터들을 3개의 train, test set fold로 나누어 테스트 수행 설정
# refit=True가 default임. True이면 가장 좋은 파라미터 설정으로 재학습 시킴.
grid_dtree = GridSearchCV(dtree, param_grid=parameters, cv=3, refit=True)

# 붓꽃 Train 데이터로 param_grid의 하이퍼 파라미터를 순차적으로 학습/평가
grid_dtree.fit(X_train, Y_train)

# GridSearchCV 결과를 추출해 데이터 프레임으로 반환
scores_df = pd.DataFrame(grid_dtree.cv_results_)
print(scores_df[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score']])

# params                                       mean_test_score  rank_test_score  ...  split1_test_score  split2_test_score
# 0  {'max_depth': 1, 'min_samples_split': 2}         0.700000                5  ...                0.7               0.70
# 1  {'max_depth': 1, 'min_samples_split': 3}         0.700000                5  ...                0.7               0.70
# 2  {'max_depth': 2, 'min_samples_split': 2}         0.958333                3  ...                1.0               0.95
# 3  {'max_depth': 2, 'min_samples_split': 3}         0.958333                3  ...                1.0               0.95
# 4  {'max_depth': 3, 'min_samples_split': 2}         0.975000                1  ...                1.0               0.95
# 5  {'max_depth': 3, 'min_samples_split': 3}         0.975000                1  ...                1.0               0.95
# [6 rows x 6 columns]

print('GridSearchCV 최적 파라미터:', grid_dtree.best_params_)
print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_dtree.best_score_))

# GridSearchCV 최적 파라미터: {'max_depth': 3, 'min_samples_split': 2}
# GridSearchCV 최고 정확도: 0.9750

# GridSearchCV의 refit으로 이미 학습된 estimator 반환
estimator = grid_dtree.best_estimator_

# GridSearchCv의 best_estimator_는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음.
pred = estimator.predict(X_test)
print('테스트 데이터 세트 정확도: {0:.4f}'.format(accuracy_score(Y_test, pred)))

# 테스트 데이터 세트 정확도: 0.9667
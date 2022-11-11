# Ensemble Learning Rules!

import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

from sklearn.model_selection import GridSearchCV

# Voting Classifier

# cancer = load_breast_cancer()
# data_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

# # 개별 모델은 로지스틱 회귀와 KNN
# lr_clf = LogisticRegression()
# knn_clf = KNeighborsClassifier(n_neighbors=8)

# # 개별 모델을 소프트 보팅 기반의 앙상블 모델로 구현한 분류기
# vo_clf = VotingClassifier(estimators=[('LR', lr_clf), ('KNN', knn_clf)], voting='soft')
# x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=156)

# # VotingClassifier 학습/예측/평가
# vo_clf.fit(x_train, y_train)
# pred = vo_clf.predict(x_test)
# print('Voting 분류기 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))
# # Voting 분류기 정확도: 0.9474

# # 개별 모델의 학습/예측/평가
# classifiers = [lr_clf, knn_clf]
# for classifier in classifiers:
#     classifier.fit(x_train, y_train)
#     pred = classifier.predict(x_test)
#     class_name = classifier.__class__.__name__
#     print('{0} 정확도: {1:.4f}'.format(class_name, accuracy_score(y_test, pred)))
# # LogisticRegression 정확도: 0.9386
# # KNeighborsClassifier 정확도: 0.9386

# --------------------------------------------------

# # Bagging Classifier
def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(), columns=['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how='outer')
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x: x[0] + '_' + str(x[1]) if x[1] > 0 else x[0], axis=1)

    new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)
    return new_feature_name_df

def get_human_dataset( ):
    
    # 각 데이터 파일들은 공백으로 분리되어 있으므로 read_csv에서 공백 문자를 sep으로 할당.
    feature_name_df = pd.read_csv('human_activity/features.txt',sep='\s+',
                        header=None,names=['column_index','column_name'])
    
    # 중복된 피처명을 수정하는 get_new_feature_name_df()를 이용, 신규 피처명 DataFrame생성. 
    new_feature_name_df = get_new_feature_name_df(feature_name_df)
    
    # DataFrame에 피처명을 컬럼으로 부여하기 위해 리스트 객체로 다시 변환
    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()
    
    # 학습 피처 데이터 셋과 테스트 피처 데이터을 DataFrame으로 로딩. 컬럼명은 feature_name 적용
    X_train = pd.read_csv('human_activity/train/X_train.txt',sep='\s+', names=feature_name )
    X_test = pd.read_csv('human_activity/test/X_test.txt',sep='\s+', names=feature_name)
    
    # 학습 레이블과 테스트 레이블 데이터을 DataFrame으로 로딩하고 컬럼명은 action으로 부여
    y_train = pd.read_csv('human_activity/train/y_train.txt',sep='\s+',header=None,names=['action'])
    y_test = pd.read_csv('human_activity/test/y_test.txt',sep='\s+',header=None,names=['action'])
    
    # 로드된 학습/테스트용 DataFrame을 모두 반환 
    return X_train, X_test, y_train, y_test


# X_train, X_test, y_train, y_test = get_human_dataset()

# warnings.filterwarnings('ignore')

# # 결정 트리에서 사용할 get_human_dataset()을 이용해 학습/테스트용 DataFrame 반환
# X_train, X_test, y_train, y_test = get_human_dataset()

# # 랜덤 포레스트 학습 및 별도의 테스트셋으로 예측 성능 평가
# rf_clf = RandomForestClassifier(random_state=0)
# rf_clf.fit(X_train, y_train)
# pred = rf_clf.predict(X_test)
# accuracy = accuracy_score(y_test, pred)
# print('랜덤 포레스트 정확도: {0:.4f}'.format(accuracy))
# # 랜덤 포레스트 정확도: 0.9253

# # params = {
# #     'n_estimators': [20],
# #     'max_depth': [6, 8, 10, 12],
# #     'min_samples_leaf': [8, 12, 18],
# #     'min_samples_split': [8, 16, 20]
# # }

# # # RandomForestClassifier 객체 생성 후 GridSearchCV 수행 (n_jobs = -1 -> 모든 CPU 코어를 이용해 학습)
# # rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
# # grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=2, n_jobs=-1)
# # grid_cv.fit(X_train, y_train)

# # print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
# # print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))
# # 최적 하이퍼 파라미터:  {'max_depth': 12, 'min_samples_leaf': 18, 'min_samples_split': 8, 'n_estimators': 10}
# # 최고 예측 정확도: 0.9032

# # 최적 하이퍼 파라미터로 재학습 및 예측/평가
# rf_clf = RandomForestClassifier(n_estimators=10, max_depth=12, min_samples_leaf=18, min_samples_split=8, random_state=0)

# rf_clf.fit(X_train, y_train)
# pred = rf_clf.predict(X_test)
# print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))
# # 예측 정확도: 0.9006

# --------------------------------------------------

# Boosting -> 실무에서는 이거 씀 (시간 오래걸림)
# GBM  Hyper Parameter
from sklearn.ensemble import GradientBoostingClassifier
import time
import warnings
warnings.filterwarnings('ignore')

x_train, x_test, y_train, y_test = get_human_dataset()

# GBM 수행시간 측정
start_time = time.time()

gb_clf = GradientBoostingClassifier(random_state=0)
gb_clf.fit(x_train, y_train)
# gb_pred = gb_clf.predict(x_test)
# gb_accuracy = accuracy_score(y_test, gb_pred)

# print('GBM 정확도: {0:.4f}'.format(gb_accuracy))
# print('GBM 수행시간: {0:.1f} 초'.format(time.time() - start_time))
# GBM 정확도: 0.9389
# GBM 수행시간: 442.2 초

# --------------------------------------------------

from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [100, 500],
    'learning_rate': [0.05, 0.1]
}

grid_cv = GridSearchCV(gb_clf, param_grid=params, cv=2, verbose=1)
grid_cv.fit(x_train, y_train)
print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))

# GridSearchCV를 이용하여 최적으로 학습된 estimator로 predict 수행
gb_pred = grid_cv.best_estimator_.predict(x_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
print('GBM 정확도: {0:.4f}'.format(gb_accuracy))
import numpy as np
import pandas as pd
import gc
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, plot_importance

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
app_train = pd.read_csv('D:/vsc_project/machinelearning_study/home-credit-default-risk/application_train.csv')
app_test = pd.read_csv('D:/vsc_project/machinelearning_study/home-credit-default-risk/application_test.csv')

# app_train['TARGET'].value_counts()
# app_train['AMT_INCOME_TOTAL'].hist()
# plt.hist(app_train['AMT_INCOME_TOTAL'])
# sns.distplot(app_train['AMT_INCOME_TOTAL'])
# sns.boxplot(app_train['AMT_INCOME_TOTAL'])

# app_train[app_train['AMT_INCOME_TOTAL'] < 1000000]['AMT_INCOME_TOTAL'].hist()
# sns.distplot(app_train[app_train['AMT_INCOME_TOTAL'] < 1000000]['AMT_INCOME_TOTAL'])

# TARGET값에 따른 Filtering 조건 각각 설정.
cond1 = (app_train['TARGET'] == 1)
cond0 = (app_train['TARGET'] == 0)
# AMT_INCOME_TOTAL은 매우 큰 값이 있으므로 이는 제외.
cond_amt = (app_train['AMT_INCOME_TOTAL'] < 500000)
# # distplot으로 TARGET=1이면 빨간색으로, 0이면 푸른색으로 Histogram 표현
# sns.distplot(app_train[cond0 & cond_amt]['AMT_INCOME_TOTAL'], label='0', color='blue')
# sns.distplot(app_train[cond1 & cond_amt]['AMT_INCOME_TOTAL'], label='1', color='red')
#
# # violinplot을 이용하면 Category 값별로 연속형 값의 분포도를 알수 있음. x는 category컬럼, y는 연속형 컬럼
# sns.violinplot(x='TARGET', y='AMT_INCOME_TOTAL', data=app_train[cond_amt])
# # 2개의 subplot을 생성
# fig, axs = plt.subplots(figsize=(12, 4), nrows=1, ncols=2, squeeze=False)

# TARGET 값 유형에 따른 Boolean Indexing 조건
cond1 = (app_train['TARGET'] == 1)
cond0 = (app_train['TARGET'] == 0)
cond_amt = (app_train['AMT_INCOME_TOTAL'] < 500000)

# # 2개의 subplot을 생성하고 왼쪽에는 violinplot을 오른쪽에는 distplot을 표현
# # violin plot을 왼쪽 subplot에 그림.
# sns.violinplot(x='TARGET', y='AMT_INCOME_TOTAL', data=app_train[cond_amt], ax=axs[0][0] )
# # Histogram을 오른쪽 subplot에 그림.
# sns.distplot(app_train[cond0 & cond_amt]['AMT_INCOME_TOTAL'], ax=axs[0][1], label='0', color='blue')
# sns.distplot(app_train[cond1 & cond_amt]['AMT_INCOME_TOTAL'], ax=axs[0][1], label='1', color='red')

def show_column_hist_by_target(df, column, is_amt=False):
    cond1 = (df['TARGET'] == 1)
    cond0 = (df['TARGET'] == 0)
    fig, axs = plt.subplots(figsize=(12, 4), nrows=1, ncols=2, squeeze=False)
    # is_amt가 True이면 < 500000 조건으로 filtering
    cond_amt = True
    if is_amt:
        cond_amt = df[column] < 500000
    sns.violinplot(x='TARGET', y=column, data=df[cond_amt], ax=axs[0][0] )
    sns.distplot(df[cond0 & cond_amt][column], ax=axs[0][1], label='0', color='blue')
    sns.distplot(df[cond1 & cond_amt][column], ax=axs[0][1], label='1', color='red')
show_column_hist_by_target(app_train, 'AMT_CREDIT', is_amt=True)

apps = pd.concat([app_train, app_test])
apps['TARGET'].value_counts(dropna = False)

# pd.factorize() 사용 -> 편리하게 Category column을 label인코딩
apps['CODE_GENDER'] = pd.factorize(apps['CODE_GENDER'])[0]
# Label 인코딩을 위해 object 유형의 컬럼만 추출
object_columns = apps.dtypes[apps.dtypes == 'object'].index.tolist()
# pd.factorize()는 한개의 컬럼만 Label 인코딩이 가능하므로 object형 컬럼들을 iteration하면서 변환 수행.
for column in object_columns:
    apps[column] = pd.factorize(apps[column])[0]

# -999로 모든 컬럼들의 Null값 변환
apps = apps.fillna(-999)
# app_test의 TARGET 컬럼은 원래 null이었는데 앞에서 fillna(-999)로 -999로 변환됨. 이를 추출함.
app_train = apps[apps['TARGET'] != -999]
app_test = apps[apps['TARGET']== -999]
# app_test의 TARGET컬럼을 Drop
app_test = app_test.drop('TARGET', axis=1)

# prime_key 제외
ftr_app = app_train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
target_app = app_train['TARGET']

train_x, valid_x, train_y, valid_y = train_test_split(ftr_app, target_app, test_size=0.3, random_state=2020)

# LightGBM 모델 생성
clf = LGBMClassifier(
                        n_jobs=-1,
                        n_estimators=1000,
                        learning_rate=0.02,
                        num_leaves=32,
                        subsample=0.8,
                        max_depth=12,
                        silent=-1,
                        verbose=-1
                    )
clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric='auc', verbose=100, early_stopping_rounds=50)
plot_importance(clf, figsize=(16, 32))
# [100]	training's auc: 0.752205	training's binary_logloss: 0.250372	valid_1's auc: 0.744317	valid_1's binary_logloss: 0.251593
# [200]	training's auc: 0.771473	training's binary_logloss: 0.243554	valid_1's auc: 0.754053	valid_1's binary_logloss: 0.247539
# [300]	training's auc: 0.784885	training's binary_logloss: 0.239292	valid_1's auc: 0.757737	valid_1's binary_logloss: 0.246203
# [400]	training's auc: 0.796336	training's binary_logloss: 0.235948	valid_1's auc: 0.758946	valid_1's binary_logloss: 0.245732
# [500]	training's auc: 0.806016	training's binary_logloss: 0.233017	valid_1's auc: 0.759411	valid_1's binary_logloss: 0.24555

#학습된 classifier의 predict_proba()를 이용하여 binary classification에서 1이될 확률만 추출
preds = clf.predict_proba(app_test.drop(['SK_ID_CURR'], axis=1))[:, 1 ]
clf.predict_proba(app_test.drop(['SK_ID_CURR'],axis=1)) #0과 1이 될 확룔이 두개다 표시가 된다
# app_test의 TARGET으로 1이될 확률 Update
app_test['TARGET'] = preds
# SK_ID_CURR과 TARGET 값만 csv 형태로 생성.
app_test[['SK_ID_CURR', 'TARGET']].to_csv('app_baseline_01.csv', index=False)

import numpy as np
import pandas as pd
import gc
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, plot_importance
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, plot_importance


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_colwidth', 30)

app_train = pd.read_csv('D:/vsc_project/machinelearning_study/home-credit-default-risk/application_train.csv')
app_test = pd.read_csv('D:/vsc_project/machinelearning_study/home-credit-default-risk/application_test.csv')
num_columns = app_train.dtypes[app_train.dtypes != 'object']

cond_1 = (app_train['TARGET'] == 1)
cond_0 = (app_train['TARGET'] == 0)
cond_f = (app_train['CODE_GENDER'] == 'F')
cond_m = (app_train['CODE_GENDER'] == 'M')
# # 전체 건수 대비 남성과 여성의 비율 확인
# print(app_train['CODE_GENDER'].value_counts()/app_train.shape[0])
# # TARGET=1 일 경우 남성과 여성의 비율 확인
# print(app_train[cond_1]['CODE_GENDER'].value_counts()/app_train[cond_1].shape[0])
# # TARGET=0 일 경우 남성과 여성의 비율 확인
# print(app_train[cond_0]['CODE_GENDER'].value_counts()/app_train[cond_0].shape[0])

# LightGBM은 NULL값을 트리 모델 생성하는데 사용할 수 있으므로 일괄적으로 Null로 변환
app_train['DAYS_EMPLOYED'] = app_train['DAYS_EMPLOYED'].replace(365243, np.nan)
apps = pd.concat([app_train, app_test])

def get_apps_processed(apps):
    # EXT_SOURCE_X FEATURE 가공
    apps['APPS_EXT_SOURCE_MEAN'] = apps[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    apps['APPS_EXT_SOURCE_STD'] = apps[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    apps['APPS_EXT_SOURCE_STD'] = apps['APPS_EXT_SOURCE_STD'].fillna(apps['APPS_EXT_SOURCE_STD'].mean())
    # AMT_CREDIT 비율로 Feature 가공
    apps['APPS_ANNUITY_CREDIT_RATIO'] = apps['AMT_ANNUITY']/apps['AMT_CREDIT']
    apps['APPS_GOODS_CREDIT_RATIO'] = apps['AMT_GOODS_PRICE']/apps['AMT_CREDIT']
    # AMT_INCOME_TOTAL 비율로 Feature 가공
    apps['APPS_ANNUITY_INCOME_RATIO'] = apps['AMT_ANNUITY']/apps['AMT_INCOME_TOTAL']
    apps['APPS_CREDIT_INCOME_RATIO'] = apps['AMT_CREDIT']/apps['AMT_INCOME_TOTAL']
    apps['APPS_GOODS_INCOME_RATIO'] = apps['AMT_GOODS_PRICE']/apps['AMT_INCOME_TOTAL']
    apps['APPS_CNT_FAM_INCOME_RATIO'] = apps['AMT_INCOME_TOTAL']/apps['CNT_FAM_MEMBERS']
    # DAYS_BIRTH, DAYS_EMPLOYED 비율로 Feature 가공
    apps['APPS_EMPLOYED_BIRTH_RATIO'] = apps['DAYS_EMPLOYED']/apps['DAYS_BIRTH']
    apps['APPS_INCOME_EMPLOYED_RATIO'] = apps['AMT_INCOME_TOTAL']/apps['DAYS_EMPLOYED']
    apps['APPS_INCOME_BIRTH_RATIO'] = apps['AMT_INCOME_TOTAL']/apps['DAYS_BIRTH']
    apps['APPS_CAR_BIRTH_RATIO'] = apps['OWN_CAR_AGE'] / apps['DAYS_BIRTH']
    apps['APPS_CAR_EMPLOYED_RATIO'] = apps['OWN_CAR_AGE'] / apps['DAYS_EMPLOYED']
    return apps

object_columns = apps.dtypes[apps.dtypes == 'object'].index.tolist()
for column in object_columns:
    apps[column] = pd.factorize(apps[column])[0]

apps_train = apps[~apps['TARGET'].isnull()]
apps_test = apps[apps['TARGET'].isnull()]
apps_test = apps_test.drop('TARGET', axis=1)

ftr_app = apps_train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
target_app = app_train['TARGET']
train_x, valid_x, train_y, valid_y = train_test_split(ftr_app, target_app, test_size=0.3, random_state=2020)

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
clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric= 'auc', verbose= 100, early_stopping_rounds= 100)
plot_importance(clf, figsize=(16, 32))

preds = clf.predict_proba(apps_test.drop(['SK_ID_CURR'], axis=1))[:, 1 ]#확률값에서 1인 확률만 선정하고
app_test['TARGET'] = preds
app_test[['SK_ID_CURR', 'TARGET']].to_csv('apps_baseline_02.csv', index=False)
get_apps_processed(apps)
# [100]	training's auc: 0.754329	training's binary_logloss: 0.249469	valid_1's auc: 0.745049	valid_1's binary_logloss: 0.25117
# [200]	training's auc: 0.77357	    training's binary_logloss: 0.242628	valid_1's auc: 0.755094	valid_1's binary_logloss: 0.247157
# [300]	training's auc: 0.78675	    training's binary_logloss: 0.238402	valid_1's auc: 0.758331	valid_1's binary_logloss: 0.245941
# [400]	training's auc: 0.798154	training's binary_logloss: 0.234988	valid_1's auc: 0.75949	valid_1's binary_logloss: 0.245539
# [500]	training's auc: 0.808194	training's binary_logloss: 0.231936	valid_1's auc: 0.759691	valid_1's binary_logloss: 0.24541
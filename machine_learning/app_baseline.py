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

def get_apps_dataset():
    apps = pd.concat([app_train, app_test])
    return apps
apps = get_apps_dataset()

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

prev = pd.read_csv('D:/vsc_project/machinelearning_study/home-credit-default-risk/previous_application.csv')
prev_app_outer = prev.merge(apps['SK_ID_CURR'], on='SK_ID_CURR', how='outer', indicator=True)
# TARGET값을 application에서 가져오기 위해 조인.
app_prev = prev.merge(app_train[['SK_ID_CURR', 'TARGET']], on='SK_ID_CURR', how='left')

num_columns = app_prev.dtypes[app_prev.dtypes != 'object'].index.tolist()
num_columns = [column for column in num_columns if column not in['SK_ID_PREV', 'SK_ID_CURR', 'TARGET']]
object_columns = app_prev.dtypes[app_prev.dtypes=='object'].index.tolist()

# DataFrameGroupby 생성.
prev_group= prev.groupby('SK_ID_CURR')
# DataFrameGroupby 객체에 aggregation함수 수행 결과를 저장한 DataFrame 생성 및 aggregation값 저장.
prev_agg = pd.DataFrame()
prev_agg['CNT'] = prev_group['SK_ID_CURR'].count()
prev_agg['AVG_CREDIT'] = prev_group['AMT_CREDIT'].mean()
prev_agg['MAX_CREDIT'] = prev_group['AMT_CREDIT'].max()

# DataFrameGroupby의 agg() 함수를 이용하여 여러개의 aggregation 함수 적용
prev_agg1 = prev_group['AMT_CREDIT'].agg(['mean', 'max', 'sum'])
prev_agg2 = prev_group['AMT_ANNUITY'].agg(['mean', 'max', 'sum'])
# merge를 이용하여 두개의 DataFrame 결합.
prev_agg = prev_agg1.merge(prev_agg2, on='SK_ID_CURR', how='inner')

agg_dict = {
                'SK_ID_CURR':['count'],
                'AMT_CREDIT':['mean', 'max', 'sum'],
                'AMT_ANNUITY':['mean', 'max', 'sum'],
                'AMT_APPLICATION':['mean', 'max', 'sum'],
                'AMT_DOWN_PAYMENT':['mean', 'max', 'sum'],
                'AMT_GOODS_PRICE':['mean', 'max', 'sum']
            }
prev_group = prev.groupby('SK_ID_CURR')
prev_amt_agg = prev_group.agg(agg_dict)
# ravel을 사용하게 되면, index로 쓰여지게 된다
prev_amt_agg.columns = ["PREV_"+"_".join(x).upper() for x in prev_amt_agg.columns.ravel()]


# 대출 신청 금액과 실제 대출액/대출 상품금액 차이 및 비율
prev['PREV_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
prev['PREV_GOODS_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_GOODS_PRICE']
prev['PREV_CREDIT_APPL_RATIO'] = prev['AMT_CREDIT']/prev['AMT_APPLICATION']
prev['PREV_ANNUITY_APPL_RATIO'] = prev['AMT_ANNUITY']/prev['AMT_APPLICATION']
prev['PREV_GOODS_APPL_RATIO'] = prev['AMT_GOODS_PRICE']/prev['AMT_APPLICATION']

prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
# 첫번째 만기일과 마지막 만기일까지의 기간
prev['PREV_DAYS_LAST_DUE_DIFF'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_LAST_DUE']

# 매월 납부 금액과 납부 횟수 곱해서 전체 납부 금액 구함.
all_pay = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']
# 전체 납부 금액 대비 AMT_CREDIT 비율을 구하고 여기에 다시 납부횟수로 나누어서 이자율 계산.
prev['PREV_INTERESTS_RATE'] = (all_pay/prev['AMT_CREDIT'] - 1)/prev['CNT_PAYMENT']

# 새롭게 생성된 대출 신청액 대비 다른 금액 차이 및 비율로 aggregation 수행.
agg_dict = {
                # 기존 컬럼.
                'SK_ID_CURR':['count'],
                'AMT_CREDIT':['mean', 'max', 'sum'],
                'AMT_ANNUITY':['mean', 'max', 'sum'],
                'AMT_APPLICATION':['mean', 'max', 'sum'],
                'AMT_DOWN_PAYMENT':['mean', 'max', 'sum'],
                'AMT_GOODS_PRICE':['mean', 'max', 'sum'],
                'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
                'DAYS_DECISION': ['min', 'max', 'mean'],
                'CNT_PAYMENT': ['mean', 'sum'],
                # 가공 컬럼
                'PREV_CREDIT_DIFF':['mean', 'max', 'sum'],
                'PREV_CREDIT_APPL_RATIO':['mean', 'max'],
                'PREV_GOODS_DIFF':['mean', 'max', 'sum'],
                'PREV_GOODS_APPL_RATIO':['mean', 'max'],
                'PREV_DAYS_LAST_DUE_DIFF':['mean', 'max', 'sum'],
                'PREV_INTERESTS_RATE':['mean', 'max']
            }

prev_group = prev.groupby('SK_ID_CURR')
prev_amt_agg = prev_group.agg(agg_dict)

# multi index 컬럼을 '_'로 연결하여 컬럼명 변경
prev_amt_agg.columns = ["PREV_"+ "_".join(x).upper() for x in prev_amt_agg.columns.ravel()]

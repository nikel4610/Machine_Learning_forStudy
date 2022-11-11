import numpy as np
import pandas as pd
import gc
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from lightgbm import LGBMClassifier

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from math import sqrt

train_raw = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')
test_raw = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)

# nan값 제거
train_raw = train_raw.dropna()

# 데이터 제거 및 추가
train_raw = train_raw.drop(train_raw[(train_raw['walkDistance'] == 0) & (train_raw['weaponsAcquired'] >= 3)].index)
train_raw['distance'] = train_raw['walkDistance'] + train_raw['rideDistance'] + train_raw['swimDistance']
train_raw['allKills'] = train_raw['kills'] + train_raw['roadKills']
train_raw['kill_rate'] = train_raw['headshotKills'] / train_raw['allKills']
train_raw['bodyKillsRate'] = (train_raw['allKills'] - train_raw['headshotKills'] + train_raw['roadKills']) / train_raw[
    'allKills']

train_raw['headshotrate'] = train_raw['headshotKills'] / train_raw['kills']
train_raw['killStreakrate'] = train_raw['killStreaks'] / train_raw['kills']
train_raw['roadkillrate'] = train_raw['roadKills'] / train_raw['kills']
train_raw['bodyK'] = (train_raw['kills'] - (train_raw['headshotKills'] + train_raw['roadKills'])) / train_raw['kills']
train_raw['healthitems'] = train_raw['heals'] + train_raw['boosts']
train_raw['totalDistance'] = train_raw['rideDistance'] + train_raw["walkDistance"] + train_raw["swimDistance"]
train_raw['distance_over_weapons'] = train_raw['totalDistance'] / train_raw['weaponsAcquired']
train_raw['walkDistance_over_heals'] = train_raw['walkDistance'] / train_raw['heals']
train_raw['walkDistance_over_kills'] = train_raw['walkDistance'] / train_raw['kills']
train_raw['killsPerWalkDistance'] = train_raw['kills'] / train_raw['walkDistance']
train_raw["skill"] = train_raw["headshotKills"] + train_raw["roadKills"]

features = list(train_raw.columns)
features.remove("Id")
features.remove("matchId")
features.remove("groupId")
features.remove("matchType")
features.remove("winPlacePerc")

df_team = train_raw.copy()

df_max = train_raw.groupby(['matchId', 'groupId'])[features].agg('max')
df_team = pd.merge(train_raw, df_max.reset_index(), suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
df_team = df_team.drop(
    ["assists_max", "killPoints_max", "headshotKills_max", "numGroups_max", "revives_max", "teamKills_max",
     "roadKills_max", "vehicleDestroys_max"], axis=1)


# 데이터 줄이기
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    # start_mem = df.memory_usage().sum() / 1024**2
    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    return df


df_rank = df_max.groupby('matchId')[features].rank(pct=True).reset_index()
df_team = pd.merge(df_team, df_rank, suffixes=["", "_maxRank"], how='left', on=['matchId', 'groupId'])
df_team = df_team.drop(["roadKills_maxRank", "matchDuration_maxRank", "maxPlace_maxRank", "numGroups_maxRank"], axis=1)
del df_max
del df_rank
gc.collect()

df_sum = train_raw.groupby(['matchId', 'groupId'])[features].agg('sum')
df_team = pd.merge(df_team, df_sum.reset_index(), suffixes=["", "_sum"], how='left', on=['matchId', 'groupId'])
df_team = df_team.drop(
    ["assists_sum", "killPoints_sum", "headshotKills_sum", "numGroups_sum", "revives_sum", "teamKills_sum",
     "roadKills_sum", "vehicleDestroys_sum"], axis=1)
del df_sum
gc.collect()

df_team = reduce_mem_usage(df_team)
test_raw = reduce_mem_usage(test_raw)

df_pred = df_team.drop(['Id', 'groupId', 'matchId'], axis=1)
df_pred = reduce_mem_usage(df_pred)
df_pred = pd.get_dummies(df_pred)
# ,'assists','killPoints','kills','killStreaks','longestKill','matchDuration','maxPlace','numGroups','rankPoints','revives','roadKills','swimDistance','teamKills','vehicleDestroys','winPoints'
df_pred_y = df_pred['winPlacePerc']
df_pred_x = df_pred.drop(['winPlacePerc'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(df_pred_x, df_pred_y, test_size=0.33, random_state=42)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'nthread': -1,
    'verbose': 0,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'max_depth': -1,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.6,
    'reg_aplha': 1,
    'reg_lambda': 0.001,
    'metric': 'rmse',
    'min_split_gain': 0.5,
    'min_child_weight': 1,
    'min_child_samples': 10,
}

train_set = lgb.Dataset(X_train, y_train, silent=True)
model = lgb.train(params, train_set=train_set, num_boost_round=300)
pred_test_y = model.predict(X_test, num_iteration=model.best_iteration)

# rms값 출력
rms = sqrt(mean_squared_error(y_test, pred_test_y))
print(rms)
# 0.0754610993548763

df_result = pd.DataFrame(columns=['PRED', 'REAL'])
df_result['PRED'] = pred_test_y
df_result['REAL'] = y_test.reset_index(drop=True)

# 그래프로 예측값과 실제값 비교
plt.figure(figsize=(12, 6))
plt.plot(df_result['PRED'].head(80), label='PRED')
plt.plot(df_result['REAL'].head(80), label='REAL')
plt.legend()
plt.show()

# df_result를 csv로 저장
df_result.to_csv('resultwithprediction.csv', index=False)
features = list(test_raw.columns)
features.remove("Id")
features.remove("matchId")
features.remove("groupId")
features.remove("matchType")

df_max = test_raw.groupby(['matchId', 'groupId'])[features].agg('max')
df_team = pd.merge(test_raw, df_max.reset_index(), suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
df_team = df_team.drop(
    ["assists_max", "killPoints_max", "headshotKills_max", "numGroups_max", "revives_max", "teamKills_max",
     "roadKills_max", "vehicleDestroys_max"], axis=1)

df_rank = df_max.groupby('matchId')[features].rank(pct=True).reset_index()
df_team = pd.merge(df_team, df_rank, suffixes=["", "_maxRank"], how='left', on=['matchId', 'groupId'])
df_team = df_team.drop(["roadKills_maxRank", "matchDuration_maxRank", "maxPlace_maxRank", "numGroups_maxRank"], axis=1)
del df_max
del df_rank
gc.collect()

df_sum = train_raw.groupby(['matchId', 'groupId'])[features].agg('sum')
df_team = pd.merge(df_team, df_sum.reset_index(), suffixes=["", "_sum"], how='left', on=['matchId', 'groupId'])
df_team = df_team.drop(
    ["assists_sum", "killPoints_sum", "headshotKills_sum", "numGroups_sum", "revives_sum", "teamKills_sum",
     "roadKills_sum", "vehicleDestroys_sum"], axis=1)
del df_sum
gc.collect()

test = df_team.drop(['Id', 'groupId', 'matchId'], axis=1)
test = pd.get_dummies(test)
pred_test_y = model.predict(test, num_iteration=model.best_iteration, predict_disable_shape_check=True)

test = pd.DataFrame(columns=['Id', 'winPlacePerc'])
test['Id'] = test_raw['Id']
test['winPlacePerc'] = pred_test_y

test.to_csv('submission.csv', index=False)
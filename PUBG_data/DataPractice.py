import numpy as np
import pandas as pd
import gc
import time
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('/pubg-finish-placement-prediction/train_V2.csv')
test_df = pd.read_csv('/pubg-finish-placement-prediction/test_V2.csv')

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)

# nan값 확인
# print(train_df.isnull().sum())
# print(test_df.isnull().sum())
# winPlacePerc       1

# nan값 제거
train_df = train_df.dropna()

# rankpoint 제거
train_df = train_df.drop(['rankPoints', 'numGroups', 'Id', 'groupId', 'matchId'], axis=1)
test_df = test_df.drop(['rankPoints', 'numGroups', 'Id', 'groupId', 'matchId'], axis=1)

# walkDistance = 0 and weaponsAcquired >= 3 데이터 지우기
train_df = train_df.drop(train_df[(train_df['walkDistance'] == 0) & (train_df['weaponsAcquired'] >= 3)].index)
# print(len(train_df))

# solo, duo, squad 데이터 갯수 count
# print(train_df['matchType'].value_counts())

train_df['healsperwalkDistance'] = train_df['heals'] / (train_df['walkDistance'] + 1)
train_df['healsperwalkDistance'].fillna(0, inplace=True)
train_df['boostsperwalkDistance'] = train_df['boosts'] / (train_df['walkDistance'] + 1)
train_df['boostsperwalkDistance'].fillna(0, inplace=True)

train_df['killsandhadshotkills'] = train_df['kills'] + train_df['headshotKills']
train_df['killsperwalkDistance'] = train_df['killsandhadshotkills'] / (train_df['walkDistance'] + 1)
train_df['killsperwalkDistance'].fillna(0, inplace=True)

# killsperwalkDistance, killsperheadshotkills, healsperwalkDistance, boostsperwalkDistance 확인
# print(train_df[['killsperwalkDistance', 'killsperheadshotkills', 'healsperwalkDistance', 'boostsperwalkDistance']].head(10))

# solo, duo, squad, solo-fpp, duo-fpp, squad-fpp 나누기
solo_train = train_df[train_df['matchType'] == 'solo'] # 6
duo_train = train_df[train_df['matchType'] == 'duo'] # 5
squad_train = train_df[train_df['matchType'] == 'squad'] # 3
solo_fpp_train = train_df[train_df['matchType'] == 'solo-fpp'] # 4
duo_fpp_train = train_df[train_df['matchType'] == 'duo-fpp'] # 2
squad_fpp_train = train_df[train_df['matchType'] == 'squad-fpp'] # 1

solo_train = solo_train.drop(['revives', 'DBNOs'], axis=1)
solo_fpp_train = solo_fpp_train.drop(['revives', 'DBNOs'], axis=1)

# f, ax = plt.subplots(figsize = (15, 15))
# # sns.heatmap(solo_train.corr(), annot = True, linewidths = .5, fmt = '.1f', ax = ax)
# # sns.heatmap(solo_fpp_train.corr(), annot = True, linewidths = .5, fmt = '.1f', ax = ax)
# # sns.heatmap(duo_train.corr(), annot = True, linewidths = .5, fmt = '.1f', ax = ax)
# # sns.heatmap(duo_fpp_train.corr(), annot = True, linewidths = .5, fmt = '.1f', ax = ax)
# # sns.heatmap(squad_train.corr(), annot = True, linewidths = .5, fmt = '.1f', ax = ax)
# # sns.heatmap(squad_fpp_train.corr(), annot = True, linewidths = .5, fmt = '.1f', ax = ax)
# # plt.show()
#
# # solo_train의 heals, boosts 별 winPlacePerc
# sns.pointplot(x = 'heals', y = 'winPlacePerc', data = solo_train, color = 'lime', alpha = 0.8)
# sns.pointplot(x = 'boosts', y = 'winPlacePerc', data = solo_train, color = 'red', alpha = 0.8)
# plt.text(4, 0.6, 'heals', color = 'lime', fontsize = 17, style = 'italic')
# plt.text(4, 0.55, 'boosts', color = 'red', fontsize = 18, style = 'italic')
# plt.xlabel('Number of Heals/Boosts', fontsize = 15, color = 'blue')
# plt.ylabel('Win Place Percentage', fontsize = 15, color = 'blue')
# plt.title('Heals and Boosts vs Win Place Percentage', fontsize = 20, color = 'blue')
# plt.grid()
# plt.show()
#
# # solo_train의 kills, headshotKills, killStreaks 별 winPlacePerc
# sns.pointplot(x = 'kills', y = 'winPlacePerc', data = solo_train, color = 'lime', alpha = 0.8)
# sns.pointplot(x = 'headshotKills', y = 'winPlacePerc', data = solo_train, color = 'red', alpha = 0.8)
# sns.pointplot(x = 'killStreaks', y = 'winPlacePerc', data = solo_train, color = 'blue', alpha = 0.8)
# plt.text(4, 0.6, 'kills', color = 'lime', fontsize = 17, style = 'italic')
# plt.text(4, 0.55, 'headshotKills', color = 'red', fontsize = 18, style = 'italic')
# plt.text(4, 0.5, 'killStreaks', color = 'blue', fontsize = 18, style = 'italic')
# plt.xlabel('Number of Kills/HeadshotKills/killStreaks', fontsize = 15, color = 'blue')
# plt.ylabel('Win Place Percentage', fontsize = 15, color = 'blue')
# plt.title('Kills, HeadshotKills and killStreaks vs Win Place Percentage', fontsize = 20, color = 'blue')
# plt.grid()
# plt.show()
#
# # solo_train의 weaponsAcquired 별 winPlacePerc
# sns.barplot(x = 'weaponsAcquired', y = 'winPlacePerc', data = solo_train)
# plt.show()
#
# # solo_train의 walkDistance 별 winPlacePerc
# sns.scatterplot(x = 'winPlacePerc', y = 'walkDistance', data = solo_train)
# plt.show()
#
# # solo_train의 damageDealt 별 winPlacePerc
# sns.scatterplot(x = 'winPlacePerc', y = 'damageDealt', data = solo_train)
# plt.show()


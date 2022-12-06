import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as py
import pickle
import lightgbm as lgb
from math import sqrt

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor, LGBMClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, mean_squared_error, r2_score
from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, \
    RandomForestRegressor, GradientBoostingRegressor

path = 'D:/vsc_project/machinelearning_study/Project/searchData/Data_real_Final/'
# weather_df = pd.read_csv(path + 'Weather_Stand.csv', encoding='cp949')

df1 = pd.read_csv(os.path.join(path, 'all.csv'), encoding='cp949')
df1 = df1[['일시', 'RCP_N', 'K_Count', 'Age']]
df1 = df1.rename(columns={'일시': 'Date', 'RCP_N': 'RCP', 'K_Count': 'Count', 'Age': 'Age'})
df1 = df1.groupby(['Date', 'RCP', 'Age']).sum().reset_index()

drop_columns = ['감자보관', '교촌치킨', '단호박찌는법', '닭다리', '돼지고기', '두부', '멜론', '문어', '볶음', '볶은소금', '쑥', '어묵', '전', '젤리', '쥬스',
                '채소', '파프리카보관', '포도', '흑마늘', 'KFC비스킷', '옥수수삶는법', '아이스크림']
df1 = df1[~df1['RCP'].isin(drop_columns)]

# weather_df = weather_df.rename(
#     columns={'일시': 'Date', '평균기온(°C)': 'Tempertures', '일강수량(mm)': 'Humidity', '평균 상대습도(%)': 'Precipitation'})
# df_with_weather = pd.merge(df1, weather_df, on='Date', how='left')

# label encoding
le = LabelEncoder()
df1['RCP'] = le.fit_transform(df1['RCP'])

print(df1)
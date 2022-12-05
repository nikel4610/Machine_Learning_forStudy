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

df1['Date'] = pd.to_datetime(df1['Date'])
df1['Month'] = df1['Date'].dt.month
df1['Season'] = df1['Month'].apply(
    lambda x: 1 if x in [3, 4, 5] else 2 if x in [6, 7, 8] else 3 if x in [9, 10, 11] else 4)
df1['Weekday'] = df1['Date'].dt.dayofweek
df1 = df1.drop(['Date'], axis=1)
df1 = df1.reset_index(drop=True)
print(df1)

x = df1.drop(['Count'], axis=1)
y = df1['Count']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


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


reduce_mem_usage(df1)

rfr_t = RandomForestRegressor(random_state=1, n_estimators=500, max_depth=7, max_features='sqrt')
rfr_t.fit(x_train, y_train)
y_pred = rfr_t.predict(x_test)
print('RandomForestRegressor')
print('MAE:', mean_absolute_error(y_test, y_pred))  # (평균 절대 오차) 예측값과 실제값의 차이의 절대값에 대한 평균
print('MSE:', mean_squared_error(y_test, y_pred))  # (평균 제곱 오차) 예측값과 실제값의 차이의 제곱에 대한 평균
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))  # (평균 제곱근 오차) 예측값과 실제값의 차이의 제곱에 대한 평균의 제곱근
print('R2:', r2_score(y_test, y_pred))  # (결정 계수) 1에 가까울수록 예측값과 실제값이 가깝다는 의미

# 예측값과 실제값 비교 csv 파일로 저장
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df2.to_csv('result_rf.csv', index=False)

df1 = pd.read_csv(os.path.join(path, 'result_rf.csv'), encoding='cp949')
df1 = df1.head(100)

# 선 그래프 그리기
plt.figure(figsize=(16, 8))
plt.plot(df1['Actual'], label='Actual')
plt.plot(df1['Predicted'], label='Predicted')
plt.legend(loc='best')
plt.show()

import pandas as pd
import plotly.express as px
import os

path = 'D:/vsc_project/machinelearning_study/Project/searchData/Data_real_Final/'
# weather_df = pd.read_csv(path + 'Weather_Stand.csv', encoding='cp949')

df1 = pd.read_csv(os.path.join(path, 'all.csv'), encoding='cp949')
df1 = df1[['일시', 'RCP_N', 'K_Count', 'Age']]
df1 = df1.rename(columns={'일시': 'Date', 'RCP_N': 'RCP', 'K_Count': 'Count', 'Age': 'Age'})
df1 = df1.groupby(['Date', 'RCP', 'Age']).sum().reset_index()

drop_columns = ['감자보관', '교촌치킨', '단호박찌는법', '닭다리', '돼지고기', '두부', '멜론', '문어', '볶음', '볶은소금', '쑥', '어묵', '전', '젤리', '쥬스',
                '채소', '파프리카보관', '포도', '흑마늘', 'KFC비스킷', '옥수수삶는법', '아이스크림']
df1 = df1[~df1['RCP'].isin(drop_columns)]

df1['Date'] = pd.to_datetime(df1['Date'])
df1['Month'] = df1['Date'].dt.month
df1['Season'] = df1['Month'].apply(lambda x: 1 if x in [3, 4, 5] else 2 if x in [6, 7, 8] else 3 if x in [9, 10, 11] else 4)
df1['Weekday'] = df1['Date'].dt.dayofweek
df1 = df1.drop(['Date'], axis=1)
df1 = df1.reset_index(drop=True)
print(df1)


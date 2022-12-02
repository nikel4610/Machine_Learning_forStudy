import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as py
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor, LGBMClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier

path = 'D:/vsc_project/machinelearning_study/Project/searchData/Data_real_Final/'
# weather_df = pd.read_csv(path + 'Weather_Stand.csv', encoding='cp949')

df1 = pd.read_csv(os.path.join(path, 'all.csv'), encoding='cp949')
df1 = df1[['일시', 'RCP_N', 'K_Count', 'Age']]
df1 = df1.rename(columns={'일시': 'Date', 'RCP_N': 'RCP', 'K_Count': 'Count', 'Age': 'Age'})
df1 = df1.groupby(['Date', 'RCP', 'Age']).sum().reset_index()

# weather_df = weather_df.rename(
#     columns={'일시': 'Date', '평균기온(°C)': 'Tempertures', '일강수량(mm)': 'Humidity', '평균 상대습도(%)': 'Precipitation'})
# df_with_weather = pd.merge(df1, weather_df, on='Date', how='left')

# label encoding
le = LabelEncoder()
df1['RCP'] = le.fit_transform(df1['RCP'])

df1['Date'] = pd.to_datetime(df1['Date'])
df1['Month'] = df1['Date'].dt.month
df1['Season'] = df1['Month'].apply(lambda x: 1 if x in [3, 4, 5] else 2 if x in [6, 7, 8] else 3 if x in [9, 10, 11] else 4)
df1['Weekday'] = df1['Date'].dt.dayofweek
df1 = df1.drop(['Date'], axis=1)
# print(df1)

x = df1.drop(['Count'], axis=1)
y = df1['Count']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

classifiers={}
classifiers1=RandomForestClassifier(n_estimators=500)
classifiers2=GradientBoostingClassifier(n_estimators=500)
classifiers3=AdaBoostClassifier(n_estimators=500)

eclf = VotingClassifier(estimators=[('lr', classifiers1), ('rf', classifiers2), ('gnb', classifiers3)],voting='soft',weights=[6,1,1])
eclf = eclf.fit(x_train, y_train)

predictions=eclf.predict(x_test)
accuracy=accuracy_score(y_test,predictions) * 100.0
print(accuracy)

# 그래프로 예측값과 실제값 비교
plt.figure(figsize=(10, 10))
plt.plot(y_test, label='y_test')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()

# 모델 결과 저장
with open('D:/vsc_project/machinelearning_study/Project/searchData/Data_real_Final/model.pkl', 'wb') as f:
    pickle.dump(eclf, f)
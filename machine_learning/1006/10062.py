# titanic data 전처리 했던거 참고하기
# adult.csv 사용 (fnlwgt -> final weight)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

adult_df = pd.read_csv('adult.csv')
# print(adult_df.head())
# print(adult_df.info())

# print(adult_df.isnull().sum().sum())
# null값 없음

# ? 값 지우기
adult_df = adult_df[adult_df['workclass'] != ' ?']
adult_df = adult_df[adult_df['occupation'] != ' ?']
adult_df = adult_df[adult_df['native_country'] != ' ?']

# seaborn을 통한 그래프 확인
sns.barplot(x='workclass', y='age', data=adult_df)
plt.show()

sns.barplot(x='occupation', y='age', data=adult_df)
plt.show()

# wage 기준으로 예측하기
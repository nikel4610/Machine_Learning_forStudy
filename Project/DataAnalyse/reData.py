import pandas as pd
import os

path = 'D:/vsc_project/machinelearning_study/Project/searchData/Data_real_Final'
df1 = pd.read_csv(os.path.join(path, 'all_re.csv'), encoding='cp949')
df1 = df1.drop(['Unnamed: 0'], axis=1)

df1 = df1[df1['RCP_N'] != '닭가슴살']

df1.to_csv(os.path.join(path, 'all_re.csv'), encoding='cp949')

# print(df1)
# print(df1.columns)

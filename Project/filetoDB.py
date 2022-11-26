import pandas as pd
import openpyxl as xl
import os

file_path = 'D:/vsc_project/machinelearning_study/Project/searchData/Data_fn'
file_format = ".csv"
file_name_os = os.listdir(file_path)
print(file_name_os)

# # 나잇대 별로 합치기
# for i in range(0, len(file_name_os), 2):
#     df1 = pd.read_csv(f"{file_path}/{file_name_os[i]}", encoding = 'cp949')
#     df2 = pd.read_csv(f"{file_path}/{file_name_os[i+1]}", encoding = 'cp949')
#     df1 = df1.loc[:, (df1 != 0).any(axis=0)]
#     df2 = df2.loc[:, (df2 != 0).any(axis=0)]
#     df = df1 + df2
#     # df1에만 있는 칼럼들
#     df1_only = df1.columns.difference(df2.columns)
#     # df2에만 있는 칼럼들
#     df2_only = df2.columns.difference(df1.columns)
#     # df에 df1_only df2_only 추가
#     df[df1_only] = df1[df1_only]
#     df[df2_only] = df2[df2_only]
#     df.to_csv(f"{file_path}/Total_FM" + str(i) + ".csv", encoding = 'cp949')

df0 = pd.read_csv(f"{file_path}/{file_name_os[0]}", encoding = 'cp949')
df2 = pd.read_csv(f"{file_path}/{file_name_os[2]}", encoding = 'cp949')
df4 = pd.read_csv(f"{file_path}/{file_name_os[4]}", encoding = 'cp949')
df6 = pd.read_csv(f"{file_path}/{file_name_os[6]}", encoding = 'cp949')
df8 = pd.read_csv(f"{file_path}/{file_name_os[8]}", encoding = 'cp949')
df10 = pd.read_csv(f"{file_path}/{file_name_os[10]}", encoding = 'cp949')

df0 = df0.loc[:, (df0 != 0).any(axis=0)]
df2 = df2.loc[:, (df2 != 0).any(axis=0)]
df4 = df4.loc[:, (df4 != 0).any(axis=0)]
df6 = df6.loc[:, (df6 != 0).any(axis=0)]
df8 = df8.loc[:, (df8 != 0).any(axis=0)]
df10 = df10.loc[:, (df10 != 0).any(axis=0)]

df = df0 + df2 + df4 + df6 + df8 + df10

df.to_csv(f"{file_path}/Total_FM.csv", encoding = 'cp949')


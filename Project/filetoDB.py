import pandas as pd
import openpyxl as xl
import os

file_path_Male = 'D:/vsc_project/machinelearning_study/Project/searchData/Data_fn/Male'
file_path_Female = 'D:/vsc_project/machinelearning_study/Project/searchData/Data_fn/Female'
file_format = ".csv"
file_name_os_Male = os.listdir(file_path_Male)
file_name_os_Female = os.listdir(file_path_Female)
print(file_name_os_Male)
print(file_name_os_Female)

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


df1 = pd.read_csv('D:/vsc_project/machinelearning_study/Project/searchData/Data_fn/Female/Total_FM3.csv', encoding = 'cp949')
df2 = pd.read_csv('D:/vsc_project/machinelearning_study/Project/searchData/Data_fn/Female/Total_FM4.csv', encoding = 'cp949')
df1 = df1.loc[:, (df1 != 0).any(axis=0)]
df2 = df2.loc[:, (df2 != 0).any(axis=0)]
df = df1 + df2
# df1에만 있는 칼럼들
df1_only = df1.columns.difference(df2.columns)
# df2에만 있는 칼럼들
df2_only = df2.columns.difference(df1.columns)
# df에 df1_only df2_only 추가
df[df1_only] = df1[df1_only]
df[df2_only] = df2[df2_only]
df.to_csv(f"{file_path_Female}/Total_FM_real" + ".csv", encoding = 'cp949')

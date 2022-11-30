import pandas as pd
import os
import openpyxl as xls

# 키워드 카운트
keywords = pd.read_excel('D:/vsc_project/machinelearning_study/Project/searchData/Data/Keyword_Counts_all.xlsx')
keywords = keywords.iloc[:, 1:]
# print(keywords.head())

# 남자 10대 ~ 60대
file_format = ".xlsx"
file_path_M = 'D:/vsc_project/machinelearning_study/Project/searchData/Data/Total_Male_fn'
file_name_os_M = os.listdir(file_path_M)
file_list_M = [f"{file_path_M}/{file}" for file in os.listdir(file_path_M) if file_format in file]

# 여자 10대 ~ 60대
file_path_FM = 'D:/vsc_project/machinelearning_study/Project/searchData/Data/Total_Female_fn'
file_name_os_FM = os.listdir(file_path_FM)
file_list_FM = [f"{file_path_FM}/{file}" for file in os.listdir(file_path_FM) if file_format in file]

df = pd.DataFrame()
# print(pd.read_excel(file_list_M[0]).iloc[:, 2:].head())

# for i in range(len(file_name_os_M)):
#     df = pd.read_excel(file_list_M[i]).iloc[:, 2:]
#     for j in range(len(df.columns)):
#         if df.columns[j] in keywords.columns:
#             df.iloc[:, j] = (df.iloc[:, j] * keywords[df.columns[j]] / 100).astype(int)
#         else:
#             df.iloc[:, j] = 0
#     df = df.loc[:, (df != 0).any(axis=0)]
#     df.to_excel(file_path_M + '/' + 'keywordCount' + str(i) + '.xlsx')

for i in range(len(file_name_os_M)):
    df = pd.read_excel(file_list_M[i]).iloc[:, 2:]
    for j in range(len(df.columns)):
        if df.columns[j] in keywords.columns:
            df.iloc[:, j] = round((df.iloc[:, j] * keywords[df.columns[j]])).astype(int)
        else:
            df.iloc[:, j] = 0
    df.to_excel(file_path_M + '/' + 'keywordCount' + str(i) + '.xlsx')
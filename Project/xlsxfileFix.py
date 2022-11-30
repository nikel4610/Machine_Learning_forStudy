import openpyxl as xls
import pandas as pd
import os

file_format = ".xlsx"
file_path_FM = 'D:/vsc_project/machinelearning_study/Project/searchData/Data/11/1'
file_name_os_FM = os.listdir(file_path_FM)
file_list_FM = [f"{file_path_FM}/{file}" for file in os.listdir(file_path_FM) if file_format in file]

file_format = ".xlsx"
file_path_M = 'D:/vsc_project/machinelearning_study/Project/searchData/Data/11/2'
file_name_os_M = os.listdir(file_path_M)
file_list_M = [f"{file_path_M}/{file}" for file in os.listdir(file_path_M) if file_format in file]

# df = pd.DataFrame()
# for i in range(len(file_list_FM)):
#     df1 = pd.read_excel(file_list_FM[i])
#     df2 = pd.read_excel(file_list_M[i])
#     df = df1 + df2
#     df.to_excel(f"{file_path_FM}/Total_FM" + str(i) + ".xlsx")

df1 = pd.read_excel(file_list_M[8])
df2 = pd.read_excel(file_list_M[9])
df = df1 + df2
df.to_excel(f"{file_path_M}/Total_M5" + ".xlsx")
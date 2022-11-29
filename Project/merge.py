import pandas as pd
import os
import openpyxl as xls
import glob

file_format = ".xlsx"
file_path = 'D:/vsc_project/machinelearning_study/Project/searchData/Data/Total_Male'
file_name_os = os.listdir(file_path)
# # print(file_name_os)

file_list = [f"{file_path}/{file}" for file in os.listdir(file_path) if file_format in file]
print(file_list)

for i in range(0, len(file_list), 2):
    df1 = pd.read_excel(file_list[i])
    df2 = pd.read_excel(file_list[i+1])
    df1 = df1.drop(['Unnamed: 0'], axis = 1)
    df2 = df2.drop(['Unnamed: 0'], axis = 1)
    df = pd.concat([df1, df2], axis = 1)
    df.to_excel(f"{file_path}/Total_FM" + str(i) + ".xlsx")



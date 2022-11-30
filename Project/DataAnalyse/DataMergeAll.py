import pandas as pd
import os

file_format = ".xlsx"
file_path = 'D:/vsc_project/machinelearning_study/Project/searchData/Data_real_Final/N_T_G'
file_name_os = os.listdir(file_path)
# # print(file_name_os)

file_list = [f"{file_path}/{file}" for file in os.listdir(file_path) if file_format in file]
# print(file_list)

df = pd.DataFrame()
for i in range(len(file_list)):
    df = pd.concat([df, pd.read_excel(file_list[i])], axis=0)
    df = df
    df.to_excel('D:/vsc_project/machinelearning_study/Project/searchData/Data_real_Final/N_T_G.xlsx', index=False)